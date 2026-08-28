import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense
from zeta_engine.rag import agentic_rag_answer
from zeta_engine.search import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid_with_snippets,
    search_reranked,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize, tokenize_with_positions

logger = logging.getLogger(__name__)


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


def search_documents(
    document_db: Path,
    index_db: Path,
    query: str,
    limit: int = 20,
    *,
    ranking: str = "hybrid",
    dense_index: Path = DEFAULT_INDEX_DIR,
    reranker_model: Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
    alpha: float = .5,
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
            if content_limit:
                lexical_content = _query_snippet(query, text, content_limit)
                dense_content = _query_snippet(query, dense_text, content_limit)
                result["content"] = max(
                    (item for item in (lexical_content, dense_content) if item),
                    key=lambda item: _lexical_score(item, terms),
                    default="",
                )
            results.append(result)
        return results


def create_server(
    host: str,
    port: int,
    document_db: Path,
    index_db: Path,
    frontend: Path,
    *,
    dense_index: Path = DEFAULT_INDEX_DIR,
    reranker_model: Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
) -> ThreadingHTTPServer:
    class SearchHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path in {"/", "/index.html"}:
                self._send(frontend.read_bytes(), "text/html; charset=utf-8")
                return
            if request.path != "/api/search":
                self._json({"error": "not found"}, 404)
                return

            parameters = parse_qs(request.query)
            query = parameters.get("q", [""])[0].strip()
            if not query:
                self._json({"error": "请输入搜索内容"}, 400)
                return

            ranking = parameters.get("ranking", ["hybrid"])[0]
            if ranking not in {"bm25f", "dense", "hybrid", "rerank", "rag"}:
                self._json({
                    "error": "ranking 必须是 bm25f、dense、hybrid、rerank 或 rag"
                }, 400)
                return

            try:
                default_limit = "5" if ranking == "rag" else "20"
                limit = min(
                    max(int(parameters.get("limit", [default_limit])[0]), 1),
                    100,
                )
            except ValueError:
                self._json({"error": "limit 必须是整数"}, 400)
                return

            try:
                alpha = float(parameters.get("alpha", ["0.5"])[0])
            except ValueError:
                self._json({"error": "alpha 必须是数字"}, 400)
                return
            if not 0. <= alpha <= 1.:
                self._json({"error": "alpha 必须在 0 到 1 之间"}, 400)
                return

            try:
                if ranking == "rag":
                    def search_fn(
                        search_query: str,
                        top_k: int,
                    ) -> list[dict[str, str]]:
                        return search_documents(
                            document_db,
                            index_db,
                            search_query,
                            top_k,
                            ranking="hybrid",
                            dense_index=dense_index,
                            device=device,
                            alpha=alpha,
                            content_limit=3000,
                        )

                    payload = agentic_rag_answer(query, search_fn, top_k=limit)
                    results = payload["results"]
                    self._json({
                        "query": query,
                        "answer": payload["answer"],
                        "count": len(results),
                        "results": results,
                    })
                    return

                results = search_documents(
                    document_db,
                    index_db,
                    query,
                    limit,
                    ranking=ranking,
                    dense_index=dense_index,
                    reranker_model=reranker_model,
                    rerank_candidates=rerank_candidates,
                    reranker_batch_size=reranker_batch_size,
                    device=device,
                    alpha=alpha,
                )
            except Exception:
                logger.exception("搜索请求失败")
                self._json({"error": "搜索服务暂时不可用"}, 500)
                return

            self._json({"query": query, "count": len(results), "results": results})

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(
                json.dumps(payload, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), SearchHandler)


def serve(
    host: str,
    port: int,
    document_db: Path,
    index_db: Path,
    frontend: Path,
    *,
    dense_index: Path = DEFAULT_INDEX_DIR,
    reranker_model: Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
) -> None:
    for path in (document_db, index_db, frontend):
        if not path.is_file():
            raise FileNotFoundError(path)

    with create_server(
        host,
        port,
        document_db,
        index_db,
        frontend,
        dense_index=dense_index,
        reranker_model=reranker_model,
        rerank_candidates=rerank_candidates,
        reranker_batch_size=reranker_batch_size,
        device=device,
    ) as server:
        print(f"ζ-engine: http://{host}:{server.server_port}")
        server.serve_forever()
