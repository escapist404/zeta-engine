import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zeta_engine.dense import PassageHit
from zeta_engine.service import (
    _acquire_structured_evidence,
    _collection_title_matches_query,
    _complete_table_values,
    _published_at,
    _query_snippet,
    _table_section_results,
    answer_question,
    search_documents,
    search_passages,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize


class ServiceTest(unittest.TestCase):
    def test_complete_collection_title_must_match_requested_kind(self) -> None:
        query = "2020年优秀大学生夏令营营员名单"

        self.assertTrue(_collection_title_matches_query(
            query,
            "2020年高瓴人工智能学院夏令营营员名单",
        ))
        self.assertFalse(_collection_title_matches_query(
            query,
            "2020年高瓴人工智能学院夏令营获奖名单",
        ))

    def test_extracts_all_data_cells_from_complete_tables(self) -> None:
        values = _complete_table_values(
            "<table><tr><td colspan='2'>成员名单</td></tr>"
            "<tr><td>甲</td><td>乙</td></tr></table>"
            "<table><tr><th>姓名</th></tr>"
            "<tr><td>丙</td><td>-</td></tr></table>"
        )

        self.assertEqual(values, ("甲", "乙", "丙"))

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
        self.assertIn("比例下界为0.666667（66.67%）", master["content"])
        self.assertEqual(
            json.loads(master["derived_relations"]),
            [{
                "kind": "ratio_bound",
                "direction": "lower",
                "numerator": 2.0,
                "denominator": 3,
                "ratio": 2 / 3,
            }],
        )
        self.assertEqual(doctorate["collection_size"], "2")

    def test_cross_year_table_scan_keeps_each_requested_document_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            hits = []
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                for index, year in enumerate((2019, 2020, 2021)):
                    document_id = storage.documents.save(
                        url=f"https://example.test/{year}",
                        title=f"{year}年项目成员名单",
                        text=f"{year}年项目成员名单",
                        fetched_at="2026-08-30T00:00:00Z",
                        content_html=(
                            "<p>一、甲组名单</p>"
                            "<table><tr><td>甲</td><td>乙</td></tr></table>"
                            "<p>二、乙组名单</p>"
                            "<table><tr><td>丙</td></tr></table>"
                        ),
                    )
                    hits.append(PassageHit(
                        document_id,
                        0,
                        1.0 - index / 10,
                        f"{year}年名单",
                        f"{year}年项目成员名单",
                    ))
                results = _table_section_results(
                    storage,
                    hits,
                    "2019、2020和2021年项目人数分别是多少？",
                    limit=3,
                )

        self.assertEqual(
            [item["evidence_scope"] for item in results],
            ["complete_document_tables"] * 3,
        )
        self.assertEqual(
            [item["collection_size"] for item in results],
            ["3", "3", "3"],
        )

    def test_acquires_complete_list_from_candidate_passage(self) -> None:
        evidence = _acquire_structured_evidence(
            "<main><h2>入选名单</h2><p>以下人员通过审核：</p>"
            "<ol><li>张三</li><li>李四</li><li>王五</li></ol>"
            "<h2>联系方式</h2><p>电话咨询。</p></main>",
            "以下人员通过审核： 张三 李四 王五",
        )

        self.assertEqual(evidence.scope, "ol")
        self.assertEqual(evidence.text, "张三 李四 王五")
        self.assertEqual(evidence.heading_path, "入选名单")
        self.assertIn("<ol>", evidence.structured_content)
        self.assertNotIn("电话咨询", evidence.text)

    def test_acquires_neighboring_blocks_for_plain_passage(self) -> None:
        evidence = _acquire_structured_evidence(
            "<article><h2>申请条件</h2><p>申请对象为在校生。</p>"
            "<p>申请人需要提交证明。</p><p>截止日期为九月十日。</p>"
            "<h2>联系方式</h2><p>电话咨询。</p></article>",
            "申请人需要提交证明。",
        )

        self.assertEqual(evidence.scope, "block_window")
        self.assertIn("申请对象为在校生", evidence.text)
        self.assertIn("截止日期为九月十日", evidence.text)
        self.assertNotIn("电话咨询", evidence.text)
        self.assertEqual(evidence.heading_path, "申请条件")

    def test_keeps_passage_when_structured_anchor_is_uncertain(self) -> None:
        evidence = _acquire_structured_evidence(
            "<main><p>完全不同的正文。</p></main>",
            "精确 passage 证据",
        )

        self.assertEqual(evidence.scope, "passage")
        self.assertEqual(evidence.text, "精确 passage 证据")

    def test_query_disambiguates_a_passage_spanning_multiple_sections(self) -> None:
        evidence = _acquire_structured_evidence(
            "<main>"
            "<section><h2>教授课程</h2><p>机器学习基础：2022-2024春</p>"
            "</section>"
            "<section><h2>社会兼职</h2><p>期刊审稿、会议领域主席以及"
            "其他很长很长的社会服务介绍。</p></section>"
            "</main>",
            "教授课程 机器学习基础：2022-2024春 社会兼职 期刊审稿、"
            "会议领域主席以及其他很长很长的社会服务介绍。",
            query="哪门课程在2022年至2024年连续三个春季学期开设？",
        )

        self.assertEqual(evidence.scope, "section")
        self.assertEqual(evidence.heading_path, "教授课程")
        self.assertIn("机器学习基础", evidence.text)
        self.assertNotIn("社会兼职", evidence.text)

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
                    content_html=(
                        "<main><p>个人简介。</p><p>教授课程："
                        "《人工智能综合设计》大一夏季学期。</p></main>"
                    ),
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
            self.assertEqual(
                results[0]["structured_content"],
                "<main><p>个人简介。</p><p>教授课程："
                "《人工智能综合设计》大一夏季学期。</p></main>",
            )

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
                "zeta_engine.service.search_passages",
                return_value=[result],
            ) as search,
            patch(
                "zeta_engine.service.agentic_rag_answer",
                side_effect=answer,
            ),
            patch("zeta_engine.service.warmup_dense") as warmup,
        ):
            response = answer_question(
                "documents.db",
                "index.db",
                "问题",
                top_k=3,
                debug=True,
            )

        self.assertEqual(response["answer"], "问题")
        warmup.assert_called_once_with(Path("data/dense"), device=None)
        self.assertEqual(search.call_args.kwargs["rerank_candidates"], 50)
        self.assertEqual(search.call_args.kwargs["reranker_batch_size"], 16)

    def test_search_passages_keeps_exact_chunk_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://example.test/20260829001.html",
                    title="完整标题",
                    text="页面中的其他内容",
                    fetched_at="2026-08-29T10:00:00",
                )

            with patch(
                "zeta_engine.service.search_reranked_passages",
                return_value=[PassageHit(
                    document_id,
                    7,
                    .9,
                    "精确 passage 证据",
                    "完整标题",
                )],
            ):
                results = search_passages(document_db, "证据", limit=1)

            self.assertEqual(results[0]["passage_id"], f"{document_id}:7")
            self.assertEqual(results[0]["chunk_index"], "7")
            self.assertEqual(results[0]["content"], "精确 passage 证据")
            self.assertNotIn("页面中的其他内容", results[0]["content"])

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
                "zeta_engine.service.search_reranked_passages",
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

    def test_answer_question_warms_dense_before_agent(self) -> None:
        calls = []

        with (
            patch(
                "zeta_engine.service.warmup_dense",
                side_effect=lambda *_args, **_kwargs: calls.append("bge"),
            ),
            patch(
                "zeta_engine.service.agentic_rag_answer",
                side_effect=lambda *_args, **_kwargs: (
                    calls.append("agent") or {"answer": "完成"}
                ),
            ),
        ):
            answer_question("documents.db", "index.db", "复合问题")

        self.assertEqual(calls, ["bge", "agent"])

    def test_answer_question_exposes_collection_as_agent_tool(self) -> None:
        response = {"answer": "2025年1项，2026年2项", "complete": True}

        def run_agent(_query, _search_fn, **kwargs):
            tool_results = kwargs["collection_fn"]("完整集合查询")
            self.assertTrue(tool_results)
            self.assertIn("确定性完整集合工具执行完成", tool_results[0]["content"])
            return response

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("zeta_engine.service.warmup_dense"),
                patch(
                    "zeta_engine.service.search_reranked_passages",
                    return_value=[],
                ) as search,
                patch(
                    "zeta_engine.service.answer_collection_question",
                    return_value=response,
                ) as collection,
                patch(
                    "zeta_engine.service.agentic_rag_answer",
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
        collection.assert_called_once()
        passage_rag.assert_called_once()

    def test_agent_continues_when_collection_tool_declines(self) -> None:
        response = {"answer": "由常规 RAG 回答"}

        def run_agent(_query, _search_fn, **kwargs):
            self.assertEqual(kwargs["collection_fn"]("无法解析的集合"), [])
            return response

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("zeta_engine.service.warmup_dense"),
                patch(
                    "zeta_engine.service.search_reranked_passages",
                    return_value=[],
                ) as search,
                patch(
                    "zeta_engine.service.answer_collection_question",
                    return_value=None,
                ) as collection,
                patch(
                    "zeta_engine.service.agentic_rag_answer",
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
        collection.assert_called_once()
        passage_rag.assert_called_once()


if __name__ == "__main__":
    unittest.main()
