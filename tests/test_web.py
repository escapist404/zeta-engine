import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from zeta_engine.index import build_index
from zeta_engine.storage import Storage
from zeta_engine.web import create_server


class WebTest(unittest.TestCase):
    def test_serves_frontend_and_search_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"
            frontend = Path(__file__).parents[1] / "index.html"

            with Storage(document_db=document_db, index_db=index_db) as storage:
                assert storage.documents is not None
                storage.documents.save(
                    url="https://info.ruc.edu.cn/example",
                    title="中国人民大学",
                    text="信息学院欢迎你",
                    fetched_at="2026-08-28T10:00:00",
                )
                build_index(storage, mode="search")

            server = create_server(
                "127.0.0.1", 0, document_db, index_db, frontend
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"

            try:
                with urlopen(base_url) as response:
                    self.assertIn("ζ-engine", response.read().decode())

                with urlopen(f"{base_url}/api/search?q={quote('中国人民大学')}") as response:
                    payload = json.load(response)
                self.assertEqual(payload["count"], 1)
                self.assertEqual(
                    payload["results"][0]["url"],
                    "https://info.ruc.edu.cn/example",
                )

                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/api/search?q=test&limit=nope")
                self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
