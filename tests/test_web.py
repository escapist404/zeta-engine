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
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize
from zeta_engine.web import _query_snippet, create_server, search_documents


class WebTest(unittest.TestCase):
    def test_query_snippet_focuses_on_matching_text(self) -> None:
        snippet = _query_snippet(
            "目标证据",
            "无关开头" * 100 + "目标证据" + "无关结尾" * 100,
        )

        self.assertIn("目标证据", snippet)
        self.assertLessEqual(len(snippet), 240)

    def test_rag_content_prefers_complete_lexical_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"
            text = (
                "个人简介。教授课程：《人工智能综合设计》大一夏季学期。"
                + "其他介绍。" * 100
            )
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://example.test/teacher",
                    title="教师主页",
                    text=text,
                    fetched_at="2026-08-28T10:00:00",
                )

            with patch(
                "zeta_engine.web.search_hybrid_with_snippets",
                return_value=([document_id], {document_id: "无关论文成果"}),
            ):
                results = search_documents(
                    document_db,
                    index_db,
                    "教授课程 夏季",
                    ranking="hybrid",
                    content_limit=3000,
                )

            self.assertIn("人工智能综合设计", results[0]["snippet"])
            self.assertEqual(results[0]["content"], text_normalize(text))

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
                    "zeta_engine.web.search_dense",
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
                    "zeta_engine.web.search_hybrid_with_snippets",
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
                    "zeta_engine.web.search_reranked",
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

                rag_results = [{
                    "title": "RAG 来源",
                    "url": "https://info.ruc.edu.cn/example",
                    "snippet": "RAG 相关片段",
                }]
                with patch(
                    "zeta_engine.web.agentic_rag_answer",
                    return_value={
                        "answer": "这是模型回答。[文档1]",
                        "results": rag_results,
                    },
                ) as rag_answer:
                    with urlopen(
                        f"{base_url}/api/search?q=test&ranking=rag"
                    ) as response:
                        rag_payload = json.load(response)
                self.assertEqual(rag_payload["answer"], "这是模型回答。[文档1]")
                self.assertEqual(rag_payload["results"], rag_results)
                self.assertEqual(rag_payload["count"], 1)
                self.assertEqual(rag_answer.call_args.args[0], "test")
                self.assertEqual(rag_answer.call_args.kwargs["top_k"], 5)
                rag_search_fn = rag_answer.call_args.args[1]
                with patch(
                    "zeta_engine.web.search_documents",
                    return_value=[],
                ) as search_documents:
                    rag_search_fn("evidence", 3)
                self.assertEqual(
                    search_documents.call_args.kwargs["ranking"],
                    "hybrid",
                )
                self.assertEqual(
                    search_documents.call_args.kwargs["content_limit"],
                    3000,
                )

                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/api/search?q=test&limit=nope")
                self.assertEqual(error.exception.code, 400)

                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/api/search?q=test&ranking=unknown")
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
