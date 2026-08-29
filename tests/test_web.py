import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from zeta_engine.dense import DenseHit
from zeta_engine.index import build_index
from zeta_engine.rag import AGENT_MAX_CYCLES
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
                    title="中国人民大学ＶＣＲ",
                    text="信息学院于１０月１４日欢迎你",
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
                    frontend_html = response.read().decode()
                self.assertIn("<span>ζ</span>engine", frontend_html)
                self.assertIn('value="hybrid" selected', frontend_html)
                self.assertIn('value="rag"', frontend_html)
                self.assertIn('id="answer"', frontend_html)
                self.assertIn('content: "「"', frontend_html)

                with urlopen(
                    f"{base_url}/api/search?q={quote('中国人民大学')}&ranking=bm25f"
                ) as response:
                    payload = json.load(response)
                self.assertEqual(payload["count"], 1)
                self.assertEqual(
                    payload["results"][0]["url"],
                    "https://info.ruc.edu.cn/example",
                )
                self.assertEqual(
                    payload["results"][0]["title"],
                    "中国人民大学vcr",
                )
                self.assertIn("10月14日", payload["results"][0]["snippet"])

                with patch(
                    "zeta_engine.service.search_dense",
                    return_value=[DenseHit(1, 0.9, "Ｄｅｎｓｅ 命中１０")],
                ):
                    with urlopen(
                        f"{base_url}/api/search?q=test&ranking=dense"
                    ) as response:
                        dense_payload = json.load(response)
                self.assertEqual(
                    dense_payload["results"][0]["snippet"],
                    "dense 命中10",
                )

                with patch(
                    "zeta_engine.service.search_hybrid_with_snippets",
                    return_value=([1], {1: "Hybrid 命中１０"}),
                ) as search_hybrid:
                    with urlopen(
                        f"{base_url}/api/search?q=test&alpha=0.7"
                    ) as response:
                        hybrid_payload = json.load(response)
                self.assertEqual(hybrid_payload["count"], 1)
                self.assertEqual(
                    hybrid_payload["results"][0]["snippet"],
                    "hybrid 命中10",
                )
                self.assertEqual(search_hybrid.call_args.kwargs["alpha"], .7)
                self.assertEqual(search_hybrid.call_args.kwargs["limit"], 20)

                with patch(
                    "zeta_engine.service.search_reranked",
                    return_value=[1],
                ) as search_reranked:
                    with urlopen(
                        f"{base_url}/api/search?q=test&ranking=rerank"
                    ) as response:
                        reranked_payload = json.load(response)
                self.assertEqual(reranked_payload["count"], 1)
                self.assertEqual(
                    search_reranked.call_args.kwargs["candidate_limit"],
                    50,
                )
                self.assertEqual(
                    search_reranked.call_args.kwargs["batch_size"],
                    16,
                )
                self.assertEqual(
                    search_reranked.call_args.kwargs["alpha"],
                    .38,
                )

                rag_results = [{
                    "title": "RAG 来源",
                    "url": "https://info.ruc.edu.cn/example",
                    "snippet": "RAG 相关片段",
                }]
                with patch(
                    "zeta_engine.web.answer_question",
                    return_value={
                        "answer": "这是模型回答。[文档1]",
                        "results": rag_results,
                        "trace": [{"cycle": 1, "action": "answer"}],
                    },
                ) as answer_question:
                    with urlopen(
                        f"{base_url}/api/search?q=test&ranking=rag&debug=1"
                    ) as response:
                        rag_payload = json.load(response)
                self.assertEqual(rag_payload["answer"], "这是模型回答。[文档1]")
                self.assertEqual(rag_payload["results"], rag_results)
                self.assertEqual(rag_payload["count"], 1)
                self.assertEqual(
                    rag_payload["trace"],
                    [{"cycle": 1, "action": "answer"}],
                )
                self.assertEqual(
                    answer_question.call_args.args,
                    (document_db, index_db, "test"),
                )
                self.assertEqual(answer_question.call_args.kwargs["top_k"], 5)
                self.assertEqual(
                    answer_question.call_args.kwargs["max_cycles"],
                    AGENT_MAX_CYCLES,
                )
                self.assertTrue(answer_question.call_args.kwargs["debug"])

                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/api/search?q=test&limit=nope")
                self.assertEqual(error.exception.code, 400)

                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/api/search?q=test&ranking=unknown")
                self.assertEqual(error.exception.code, 400)

                with self.assertRaises(HTTPError) as error:
                    urlopen(
                        f"{base_url}/api/search?q=test&ranking=rag&max_cycles=0"
                    )
                self.assertEqual(error.exception.code, 400)

                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/api/search?q=test&ranking=hybrid&alpha=2")
                self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
