import re
from datetime import date
from pathlib import Path

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense, warmup_dense
from zeta_engine.rag import AGENT_MAX_CYCLES, agentic_rag_answer
from zeta_engine.search import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid_with_snippets,
    search_phrase,
    search_reranked,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize, tokenize_with_positions

_PUBLISHED_DATE_RE = re.compile(
    r"(?:日期|发布时间|更新时间)\s*[:：]?\s*"
    r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})日?"
)
_URL_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})")
RAG_DOCUMENT_WINDOW_CHARS = 28_000


def _query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(
        term
        for term, _offset in tokenize_with_positions(
            text_normalize(query),
            mode="search",
        )
    ))


def _lexical_score(text: str, terms: list[str]) -> int:
    return sum(len(term) * text.count(term) for term in terms)


def _query_snippet(query: str, text: str, limit: int = 240) -> str:
    text = text_normalize(text)
    if len(text) <= limit:
        return text

    terms = _query_terms(query)
    starts = {0}
    for term in terms:
        offset = 0
        while (position := text.find(term, offset)) >= 0:
            starts.add(max(0, min(position - limit // 3, len(text) - limit)))
            offset = position + len(term)

    def score(start: int) -> tuple[int, int]:
        window = text[start:start + limit]
        return _lexical_score(window, terms), -start

    start = max(starts, key=score)
    return text[start:start + limit]


def _published_at(url: str, text: str) -> str:
    """Extract an ISO publication date from page text or its URL."""

    match = _PUBLISHED_DATE_RE.search(text) or _URL_DATE_RE.search(url)
    if not match:
        return ""
    year, month, day = map(int, match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def search_documents(
    document_db: str | Path,
    index_db: str | Path,
    query: str,
    limit: int = 20,
    *,
    ranking: str = "hybrid",
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    content_limit: int = 0,
) -> list[dict[str, str]]:
    if content_limit < 0:
        raise ValueError("content_limit 不能小于 0")

    with Storage(
        document_db=document_db,
        index_db=index_db if ranking != "dense" else None,
    ) as storage:
        assert storage.documents is not None
        snippets = {}
        if ranking == "dense":
            hits = search_dense(
                query,
                dense_index,
                limit=limit,
                device=device,
            )
            document_ids = [hit.document_id for hit in hits]
            snippets = {hit.document_id: hit.snippet for hit in hits}
        elif ranking == "phrase":
            document_ids = search_phrase(storage, query)[:limit]
        elif ranking == "bm25f":
            document_ids = search_bm25f(storage, query)[:limit]
        elif ranking == "hybrid":
            document_ids, snippets = search_hybrid_with_snippets(
                storage,
                query,
                dense_index,
                limit=limit,
                alpha=alpha,
                device=device,
            )
        elif ranking == "rerank":
            document_ids = search_reranked(
                storage,
                query,
                dense_index,
                reranker_model=reranker_model,
                limit=limit,
                candidate_limit=rerank_candidates,
                batch_size=reranker_batch_size,
                alpha=alpha,
                device=device,
            )
        else:
            raise ValueError(f"不支持的排名方式: {ranking}")

        results = []
        terms = _query_terms(query)
        for document_id in document_ids:
            document = storage.documents.get(document_id)
            if document is None:
                continue
            url, title, text, _fetched_at = document
            content_html = storage.documents.get_content_html(document_id)
            dense_text = text_normalize(snippets.get(document_id, ""))
            dense_snippet = _query_snippet(query, dense_text)
            lexical_snippet = _query_snippet(query, text)
            snippet = max(
                (item for item in (dense_snippet, lexical_snippet) if item),
                key=lambda item: _lexical_score(item, terms),
                default="",
            )
            result = {
                "title": text_normalize(title) or url,
                "url": url,
                "snippet": snippet[:240],
            }
            if published_at := _published_at(url, text):
                result["published_at"] = published_at
            if content_limit:
                lexical_content = _query_snippet(query, text, content_limit)
                dense_content = _query_snippet(query, dense_text, content_limit)
                result["content"] = max(
                    (item for item in (lexical_content, dense_content) if item),
                    key=lambda item: _lexical_score(item, terms),
                    default="",
                )
                if content_html and len(text_normalize(text)) <= content_limit:
                    result["structured_content"] = content_html
            results.append(result)
        return results


def answer_question(
    document_db: str | Path,
    index_db: str | Path,
    query: str,
    top_k: int = 5,
    *,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    max_cycles: int = AGENT_MAX_CYCLES,
    debug: bool = False,
) -> dict[str, object]:
    if alpha > 0.:
        warmup_dense(dense_index, device=device)

    def search_fn(search_query: str, limit: int) -> list[dict[str, str]]:
        return search_documents(
            document_db,
            index_db,
            search_query,
            limit,
            ranking="hybrid",
            dense_index=dense_index,
            device=device,
            alpha=alpha,
            content_limit=RAG_DOCUMENT_WINDOW_CHARS,
        )

    return agentic_rag_answer(
        query,
        search_fn,
        top_k=top_k,
        max_cycles=max_cycles,
        debug=debug,
    )
