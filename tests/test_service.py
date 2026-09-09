import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zeta_engine.retrieval.dense import PassageHit
from zeta_engine.retrieval.acquisition import _table_section_results
from zeta_engine.application.service import answer_question, search_passages
from zeta_engine.infrastructure.storage import Storage


class ServiceTest(unittest.TestCase):
    def test_complete_table_sections_keep_independent_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://example.test/camp",
                    title="夏令营营员名单",
                    text="学硕与直博营员名单",
                    fetched_at="2026-08-29T00:00:00Z",
                    content_html=(
                        "<p>一、学术硕士营员名单</p><p>拟推荐不少于2人</p>"
                        "<table><tr><td>甲</td><td>乙</td><td>丙</td></tr></table>"
                        "<p>二、直博营员名单</p>"
                        "<table><tr><td>丁</td><td>戊</td></tr></table>"
                    ),
                )
                results = _table_section_results(
                    storage,
                    [PassageHit(document_id, 0, 1.0, "学硕名单", "夏令营营员名单")],
                    "学术硕士营员最低录取比",
                )

        aggregate = next(
            item for item in results
            if item["evidence_scope"] == "complete_document_tables"
        )
        master = next(
            item for item in results
            if "学术硕士营员名单" in item["heading_path"]
        )
        doctorate = next(
            item for item in results
            if "直博营员名单" in item["heading_path"]
        )
        self.assertEqual(aggregate["collection_size"], "5")
        self.assertEqual(master["collection_size"], "3")
        self.assertIn("拟推荐不少于2人", master["content"])
        self.assertNotIn("derived_relations", master)
        self.assertEqual(doctorate["collection_size"], "2")

    def test_search_passages_separates_candidate_from_acquired_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://example.test/list.html",
                    title="项目名单",
                    text="项目名单 甲 乙 丙 联系方式",
                    fetched_at="2026-08-29T10:00:00",
                    content_html=(
                        "<main><h2>项目名单</h2><ul><li>甲</li>"
                        "<li>乙</li><li>丙</li></ul>"
                        "<h2>联系方式</h2><p>电话咨询</p></main>"
                    ),
                )

            with patch(
                "zeta_engine.application.service.search_reranked_passages",
                return_value=[PassageHit(
                    document_id,
                    2,
                    .9,
                    "项目名单 甲 乙 丙",
                    "项目名单",
                )],
            ):
                result = search_passages(document_db, "项目名单", limit=1)[0]

            self.assertEqual(result["candidate_content"], "项目名单 甲 乙 丙")
            self.assertEqual(result["content"], "甲 乙 丙")
            self.assertEqual(result["evidence_scope"], "ul")
            self.assertEqual(result["heading_path"], "项目名单")
            self.assertIn("<ul>", result["structured_content"])
            self.assertNotIn("电话咨询", result["content"])

    def test_agent_continues_when_collection_tool_declines(self) -> None:
        response = {"answer": "由常规 RAG 回答"}

        def run_agent(_query, _search_fn, **kwargs):
            self.assertEqual(kwargs["collection_fn"]("无法解析的集合"), [])
            return response

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("zeta_engine.application.service.warmup_dense"),
                patch(
                    "zeta_engine.application.service.search_reranked_passages",
                    return_value=[],
                ) as search,
                patch(
                    "zeta_engine.application.service._table_section_results",
                    return_value=[],
                ) as table_scan,
                patch(
                    "zeta_engine.application.service.agentic_rag_answer",
                    side_effect=run_agent,
                ) as passage_rag,
            ):
                actual = answer_question(
                    Path(directory) / "documents.db",
                    Path(directory) / "index.db",
                    "比较2025和2026项目：分别多少项，哪些主题共同出现？",
                )

        self.assertEqual(actual, response)
        search.assert_called_once()
        table_scan.assert_called_once()
        passage_rag.assert_called_once()


if __name__ == "__main__":
    unittest.main()
