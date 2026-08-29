import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from zeta_engine.dense import DEFAULT_INDEX_DIR
from zeta_engine.rag import AGENT_MAX_CYCLES
from zeta_engine.search import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
)
from zeta_engine.service import answer_question, search_documents

logger = logging.getLogger(__name__)


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
                max_cycles = int(parameters.get(
                    "max_cycles",
                    [str(AGENT_MAX_CYCLES)],
                )[0])
            except ValueError:
                self._json({"error": "max_cycles 必须是整数"}, 400)
                return
            if not 1 <= max_cycles <= 8:
                self._json({"error": "max_cycles 必须在 1 到 8 之间"}, 400)
                return
            debug = parameters.get("debug", [""])[0].lower() in {
                "1", "true", "yes", "on",
            }

            try:
                if ranking == "rag":
                    payload = answer_question(
                        document_db,
                        index_db,
                        query,
                        top_k=limit,
                        dense_index=dense_index,
                        device=device,
                        alpha=alpha,
                        max_cycles=max_cycles,
                        debug=debug,
                    )
                    results = payload["results"]
                    response_payload = {
                        "query": query,
                        "answer": payload["answer"],
                        "claims": payload.get("claims", []),
                        "sources": payload.get("sources", []),
                        "count": len(results),
                        "results": results,
                    }
                    if debug:
                        response_payload["trace"] = payload.get("trace", [])
                    self._json(response_payload)
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
