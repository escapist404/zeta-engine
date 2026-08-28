import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from zeta_engine.search import search_bm25f
from zeta_engine.storage import Storage

logger = logging.getLogger(__name__)


def search_documents(
    document_db: Path,
    index_db: Path,
    query: str,
    limit: int = 10,
) -> list[dict[str, str]]:
    with Storage(document_db=document_db, index_db=index_db) as storage:
        assert storage.documents is not None
        document_ids = search_bm25f(storage, query)[:limit]
        results = []
        for document_id in document_ids:
            document = storage.documents.get(document_id)
            if document is None:
                continue
            url, title, text, _fetched_at = document
            results.append({
                "title": title or url,
                "url": url,
                "snippet": " ".join(text.split())[:240],
            })
        return results


def create_server(
    host: str,
    port: int,
    document_db: Path,
    index_db: Path,
    frontend: Path,
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
                limit = min(max(int(parameters.get("limit", ["10"])[0]), 1), 100)
                results = search_documents(document_db, index_db, query, limit)
            except ValueError:
                self._json({"error": "limit 必须是整数"}, 400)
                return
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
) -> None:
    for path in (document_db, index_db, frontend):
        if not path.is_file():
            raise FileNotFoundError(path)

    with create_server(host, port, document_db, index_db, frontend) as server:
        print(f"ζ-engine: http://{host}:{server.server_port}")
        server.serve_forever()
