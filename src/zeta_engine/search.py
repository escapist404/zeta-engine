from collections import Counter
from math import log
from pathlib import Path

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import (
    load_stopwords,
    text_normalize,
    tokenize_with_positions,
)

def search_term(
    storage: Storage,
    term: str,
) -> set[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    term = text_normalize(term)
    if not term or term in load_stopwords():
        return set()

    return set(storage.index.lookup_posting(term, storage.index.TITLE)) | set(
        storage.index.lookup_posting(term, storage.index.TEXT)
    )


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


def search_query(
        storage: Storage, 
        query: str, *, 
        phrase: bool = False
) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")
    if phrase:
        return search_phrase(storage, query)

    query = text_normalize(query)
    if not query:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    terms = list(dict.fromkeys(
        term for term, _position in tokenize_with_positions(query, mode)
        if term.strip()
    ))
    if not terms:
        return []

    postings = {
        term: (
            storage.index.lookup_posting(term, storage.index.TITLE),
            storage.index.lookup_posting(term, storage.index.TEXT),
        )
        for term in terms
    }
    matches = set(postings[terms[0]][0]) | set(postings[terms[0]][1])
    for title_posting, text_posting in postings.values():
        matches &= set(title_posting) | set(text_posting)

    scores = {
        document_id: sum(
            2 * len(title_posting.get(document_id, ()))
            + len(text_posting.get(document_id, ()))
            for title_posting, text_posting in postings.values()
        )
        for document_id in matches
    }

    return sorted(matches, key=lambda document_id: (-scores[document_id], document_id))


def search_tf_idf(
        storage: Storage,
        query: str,
) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    query = text_normalize(query)
    if not query:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    query_terms = [
        term
        for term, _ in tokenize_with_positions(query, mode)
        if term.strip()
    ]

    if not query_terms:
        return []

    terms = list(dict.fromkeys(query_terms))

    document_count = storage.index.count_documents()
    query_tf = Counter(query_terms)

    postings = {
        term: (
            storage.index.lookup_posting(term, storage.index.TITLE),
            storage.index.lookup_posting(term, storage.index.TEXT),
        )
        for term in terms
    }

    idfs = {}
    for term, (title_posting, text_posting) in postings.items():
        df = len(set(title_posting) | set(text_posting))
        idfs[term] = log((document_count + 1) / (df + 1)) + 1.
    query_weights = {
        term: (1 + log(query_tf[term])) * idfs[term]
        for term in terms
    }

    matches = set(postings[terms[0]][0]) | set(postings[terms[0]][1])
    for title_posting, text_posting in postings.values():
        matches |= set(title_posting) | set(text_posting)

    def document_weight(document_id: int, term: str) -> float:
        title_posting, text_posting = postings[term]
        tf = (
            len(title_posting.get(document_id, ()))
            + len(text_posting.get(document_id, ()))
        )
        return (1 + log(tf)) * idfs[term] if tf else 0.

    selected_tf_idf_norm = storage.index.get_tf_idf_norm(matches)

    scores = {
        document_id: sum(
            document_weight(document_id, term) * query_weights[term]
            for term in terms
        ) / selected_tf_idf_norm[document_id]
        for document_id in matches
        if selected_tf_idf_norm.get(document_id, 0.) > 0.
    }

    return sorted(matches, key=lambda document_id: (-scores[document_id], document_id))


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

    if not 0. <= alpha <= 1.:
        raise ValueError("alpha 必须在 0 到 1 之间")
    if limit <= 0:
        return []
    if alpha == 0.:
        return search_bm25f(storage, query)[:limit]
    if alpha == 1.:
        return [
            hit.document_id
            for hit in search_dense(
                query,
                dense_index,
                limit=limit,
                device=device,
            )
        ]

    candidate_limit = max(100, limit * 5)
    all_sparse_scores = _score_bm25f(storage, query)
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
        for hit in search_dense(
            query,
            dense_index,
            limit=candidate_limit,
            device=device,
        )
    }
    return _fuse_scores(
        sparse_scores,
        dense_scores,
        alpha=alpha,
        limit=limit,
    )
