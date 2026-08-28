import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense
from zeta_engine.search import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid,
    search_reranked,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize

logger = logging.getLogger(__name__)


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
) -> list[dict[str, str]]:
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
            document_ids = search_hybrid(
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
        for document_id in document_ids:
            document = storage.documents.get(document_id)
            if document is None:
                continue
            url, title, text, _fetched_at = document
            results.append({
                "title": text_normalize(title) or url,
                "url": url,
                "snippet": text_normalize(
                    snippets.get(document_id, text),
                )[:240],
            })
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

            try:
                limit = min(max(int(parameters.get("limit", ["20"])[0]), 1), 100)
            except ValueError:
                self._json({"error": "limit 必须是整数"}, 400)
                return

            ranking = parameters.get("ranking", ["hybrid"])[0]
            if ranking not in {"bm25f", "dense", "hybrid", "rerank"}:
                self._json({
                    "error": "ranking 必须是 bm25f、dense、hybrid 或 rerank"
                }, 400)
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
