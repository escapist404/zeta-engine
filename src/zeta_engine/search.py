from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from math import log
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import (
    text_normalize,
    tokenize_with_positions,
)

DEFAULT_RERANKER_MODEL_PATH = Path("models/bge-reranker-base")
DEFAULT_RERANK_CANDIDATES = 50
DEFAULT_RERANK_BATCH_SIZE = 16
RERANK_MAX_TOKENS = 512
RERANK_TITLE_MAX_TOKENS = 64
RERANK_OVERLAP_TOKENS = 64

_RERANK_LOCK = Lock()


def search_phrase(
    storage: Storage,
    phrase: str,
) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    phrase = text_normalize(phrase)
    if not phrase:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    tokens = tokenize_with_positions(phrase, mode)
    if not tokens:
        return []

    matches: set[int] = set()
    first_offset = tokens[0][1]

    for field in (storage.index.TITLE, storage.index.TEXT):
        postings = [
            storage.index.lookup_posting(term, field)
            for term, _offset in tokens
        ]
        if any(not posting for posting in postings):
            continue

        document_ids = set(postings[0])
        for posting in postings[1:]:
            document_ids &= posting.keys()

        for document_id in document_ids:
            positions = [set(posting[document_id]) for posting in postings]
            if any(
                all(
                    start + offset - first_offset in positions[index]
                    for index, (_term, offset) in enumerate(tokens[1:], start=1)
                )
                for start in positions[0]
            ):
                matches.add(document_id)

    return sorted(matches)


def _score_bm25f(
        storage: Storage,
        query: str,
) -> dict[int, float]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    query = text_normalize(query)
    if not query:
        return {}

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    query_terms = [
        term
        for term, _ in tokenize_with_positions(query, mode)
        if term.strip()
    ]

    if not query_terms:
        return {}

    terms = list(dict.fromkeys(query_terms))

    index = storage.index
    document_count = index.count_documents()
    if document_count == 0:
        return {}

    postings = {
        term: (
            index.lookup_posting(term, index.TITLE),
            index.lookup_posting(term, index.TEXT),
        )
        for term in terms
    }
    matches = {
        document_id
        for field_postings in postings.values()
        for posting in field_postings
        for document_id in posting
    }
    if not matches:
        return {}

    total_title_len, total_text_len = index.get_total_len()
    average_lengths = {
        index.TITLE: total_title_len / document_count,
        index.TEXT: total_text_len / document_count,
    }
    document_lengths = index.get_document_lengths(matches)

    k1 = 1.2
    field_parameters = {
        index.TITLE: (2.0, 0.3),
        index.TEXT: (1.0, 0.75),
    }

    def combined_tf(document_id: int, term: str) -> float:
        total = 0.

        for field, posting in zip(
            (index.TITLE, index.TEXT),
            postings[term],
        ):
            tf = len(posting.get(document_id, ()))
            if not tf:
                continue

            weight, b = field_parameters[field]
            average_length = average_lengths[field]
            length_normalization = (
                1. - b
                + b * document_lengths[document_id][field] / average_length
                if average_length > 0.
                else 1.
            )
            total += weight * tf / length_normalization

        return total

    scores = dict.fromkeys(matches, 0.)

    for term, (title_posting, text_posting) in postings.items():
        term_matches = set(title_posting) | set(text_posting)
        document_frequency = len(term_matches)
        idf = log(
            1.
            + (document_count - document_frequency + .5)
            / (document_frequency + .5)
        )

        for document_id in term_matches:
            tf = combined_tf(document_id, term)
            scores[document_id] += idf * (k1 + 1.) * tf / (k1 + tf)

    return scores


def search_bm25f(
    storage: Storage,
    query: str,
) -> list[int]:
    scores = _score_bm25f(storage, query)
    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lowest = min(scores.values())
    highest = max(scores.values())
    if highest == lowest:
        return dict.fromkeys(scores, 1.)
    scale = highest - lowest
    return {
        document_id: (score - lowest) / scale
        for document_id, score in scores.items()
    }


def _fuse_scores(
    sparse_scores: dict[int, float],
    dense_scores: dict[int, float],
    *,
    alpha: float,
    limit: int,
) -> list[int]:
    """Min-max normalize each score set, then combine them linearly."""

    if not 0. <= alpha <= 1.:
        raise ValueError("alpha 必须在 0 到 1 之间")
    if limit <= 0:
        return []
    if alpha == 0.:
        combined = sparse_scores
    elif alpha == 1.:
        combined = dense_scores
    else:
        sparse = _normalize_scores(sparse_scores)
        dense = _normalize_scores(dense_scores)
        combined = {
            document_id: (
                (1. - alpha) * sparse.get(document_id, 0.)
                + alpha * dense.get(document_id, 0.)
            )
            for document_id in sparse.keys() | dense.keys()
        }

    return sorted(
        combined,
        key=lambda document_id: (-combined[document_id], document_id),
    )[:limit]


def search_hybrid_with_snippets(
    storage: Storage,
    query: str,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    *,
    limit: int = 20,
    alpha: float = .5,
    device: str | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Return fused document IDs and available Dense snippets."""

    if not 0. <= alpha <= 1.:
        raise ValueError("alpha 必须在 0 到 1 之间")
    if limit <= 0:
        return [], {}
    if alpha == 0.:
        return search_bm25f(storage, query)[:limit], {}
    if alpha == 1.:
        dense_hits = search_dense(
            query,
            dense_index,
            limit=limit,
            device=device,
        )
        return (
            [hit.document_id for hit in dense_hits],
            {hit.document_id: hit.snippet for hit in dense_hits},
        )

    candidate_limit = max(100, limit * 5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        dense_future = executor.submit(
            search_dense,
            query,
            dense_index,
            limit=candidate_limit,
            device=device,
        )
        all_sparse_scores = _score_bm25f(storage, query)
        dense_hits = dense_future.result()

    sparse_ids = sorted(
        all_sparse_scores,
        key=lambda document_id: (-all_sparse_scores[document_id], document_id),
    )[:candidate_limit]
    sparse_scores = {
        document_id: all_sparse_scores[document_id]
        for document_id in sparse_ids
    }
    dense_scores = {
        hit.document_id: hit.score
        for hit in dense_hits
    }
    return (
        _fuse_scores(
            sparse_scores,
            dense_scores,
            alpha=alpha,
            limit=limit,
        ),
        {hit.document_id: hit.snippet for hit in dense_hits},
    )


def search_hybrid(
    storage: Storage,
    query: str,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    *,
    limit: int = 20,
    alpha: float = .5,
    device: str | None = None,
) -> list[int]:
    """Combine per-query normalized BM25F and Dense scores linearly."""

    document_ids, _snippets = search_hybrid_with_snippets(
        storage,
        query,
        dense_index,
        limit=limit,
        alpha=alpha,
        device=device,
    )
    return document_ids


@lru_cache(maxsize=2)
def _load_cross_encoder(model_path: str, device: str | None):
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Reranker 模型目录不存在: {path}")

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        str(path),
        device=device,
        local_files_only=True,
        max_length=RERANK_MAX_TOKENS,
    )


def _rerank_passages(
    tokenizer: Any,
    query: str,
    title: str,
    text: str,
) -> list[str]:
    """Return one or two distinct passages with strong lexical query overlap."""

    title = text_normalize(title)
    text = text_normalize(text)
    if not title and not text:
        return []

    query_ids = tokenizer.encode(query, add_special_tokens=False)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    title_ids = tokenizer.encode(title, add_special_tokens=False)[
        :RERANK_TITLE_MAX_TOKENS
    ]
    separator_ids = tokenizer.encode("\n", add_special_tokens=False)
    prefix_ids = title_ids + separator_ids if title_ids and text else title_ids
    body_budget = (
        RERANK_MAX_TOKENS
        - special_tokens
        - min(len(query_ids), RERANK_MAX_TOKENS // 2)
        - len(prefix_ids)
    )

    if not text or body_budget <= 0:
        passage = tokenizer.decode(
            title_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return [passage] if passage else []

    body_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )
    overlap = min(RERANK_OVERLAP_TOKENS, max(0, body_budget - 1))
    step = body_budget - overlap
    passages = []

    for start in range(0, len(body_ids), step):
        chunk_ids = body_ids[start:start + body_budget]
        if not chunk_ids:
            break
        passage = tokenizer.decode(
            prefix_ids + chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if passage:
            passages.append(passage)
        if start + body_budget >= len(body_ids):
            break

    compact_query = "".join(
        character
        for character in text_normalize(query)
        if character.isalnum()
    )
    if not compact_query:
        return passages[:1]
    query_bigrams = {
        compact_query[index:index + 2]
        for index in range(len(compact_query) - 1)
    }

    def lexical_score(item: tuple[int, str]) -> tuple[bool, int, int, int]:
        index, passage = item
        compact_passage = "".join(
            character
            for character in text_normalize(passage)
            if character.isalnum()
        )
        return (
            compact_query in compact_passage,
            sum(bigram in compact_passage for bigram in query_bigrams),
            sum(character in compact_passage for character in set(compact_query)),
            -index,
        )

    ranked = sorted(enumerate(passages), key=lexical_score, reverse=True)
    best_index, best_passage = ranked[0]
    best_exact, best_bigrams, best_characters, _index = lexical_score(ranked[0])
    if best_exact:
        return [best_passage]

    best_relevance = 2 * best_bigrams + best_characters
    for candidate in ranked[1:]:
        index, passage = candidate
        if abs(index - best_index) <= 1:
            continue
        _exact, bigrams, characters, _index = lexical_score(candidate)
        relevance = 2 * bigrams + characters
        if relevance > 0 and relevance >= .8 * best_relevance:
            return [best_passage, passage]
    return [best_passage]


def search_reranked(
    storage: Storage,
    query: str,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    *,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    limit: int = 20,
    candidate_limit: int = DEFAULT_RERANK_CANDIDATES,
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    alpha: float = .5,
    device: str | None = None,
) -> list[int]:
    """Rerank Hybrid candidates with a query-document CrossEncoder."""

    if storage.documents is None:
        raise ValueError("Reranker 查询需要 document_db")
    if limit <= 0:
        return []
    if candidate_limit <= 0:
        raise ValueError("candidate_limit 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    query = text_normalize(query)
    candidates = search_hybrid(
        storage,
        query,
        dense_index,
        limit=max(limit, candidate_limit),
        alpha=alpha,
        device=device,
    )
    model = _load_cross_encoder(str(Path(reranker_model).resolve()), device)
    pair_document_ids = []
    pairs = []
    for document_id in candidates:
        document = storage.documents.get(document_id)
        if document is None:
            continue
        _url, title, text, _fetched_at = document
        for passage in _rerank_passages(model.tokenizer, query, title, text):
            pair_document_ids.append(document_id)
            pairs.append((query, passage))

    if not pairs:
        return []

    with _RERANK_LOCK:
        scores = np.asarray(model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )).reshape(-1)
    if len(scores) != len(pair_document_ids):
        raise ValueError("Reranker 返回的分数数量不正确")

    document_scores: dict[int, float] = {}
    for document_id, score in zip(pair_document_ids, scores, strict=True):
        document_scores[document_id] = max(
            document_scores.get(document_id, float("-inf")),
            float(score),
        )

    candidate_order = {
        document_id: index
        for index, document_id in enumerate(candidates)
    }
    return sorted(
        document_scores,
        key=lambda document_id: (
            -document_scores[document_id],
            candidate_order[document_id],
        ),
    )[:limit]
