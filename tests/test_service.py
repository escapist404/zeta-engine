import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zeta_engine.service import (
    _published_at,
    _query_snippet,
    answer_question,
    search_documents,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize


class ServiceTest(unittest.TestCase):
    def test_extracts_publication_date_from_text_or_url(self) -> None:
        self.assertEqual(
            _published_at("https://example.test/page", "日期：2025-4-3"),
            "2025-04-03",
        )
        self.assertEqual(
            _published_at("https://example.test/20260516003.html", "正文"),
            "2026-05-16",
        )
        self.assertEqual(
            _published_at("https://example.test/20261340003.html", "正文"),
            "",
        )

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
                "zeta_engine.service.search_hybrid_with_snippets",
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

    def test_answer_question_uses_shared_search_service(self) -> None:
        result = {
            "title": "标题",
            "url": "https://example.test",
            "snippet": "材料",
        }

        def answer(query, search_fn, **kwargs):
            self.assertEqual(search_fn("证据", 3), [result])
            return {"answer": query, "results": [result]}

        with (
            patch(
                "zeta_engine.service.search_documents",
                return_value=[result],
            ) as search,
            patch(
                "zeta_engine.service.agentic_rag_answer",
                side_effect=answer,
            ),
        ):
            response = answer_question(
                "documents.db",
                "index.db",
                "问题",
                top_k=3,
                ranking="dense",
                debug=True,
            )

        self.assertEqual(response["answer"], "问题")
        self.assertEqual(search.call_args.kwargs["ranking"], "dense")
        self.assertEqual(search.call_args.kwargs["content_limit"], 3000)


if __name__ == "__main__":
    unittest.main()
