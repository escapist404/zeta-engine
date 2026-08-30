import json
import tempfile
import unittest
from pathlib import Path

from zeta_engine.collection_rag import (
    answer_collection_question,
    plan_collection_query,
)
from zeta_engine.dense import PassageHit
from zeta_engine.storage import Storage


class CollectionRagTest(unittest.TestCase):
    def test_planner_only_routes_explicit_cross_group_collections(self) -> None:
        self.assertIsNone(plan_collection_query("2025年的项目负责人是谁？"))
        self.assertIsNone(plan_collection_query("比较甲项目和乙项目"))

        plan = plan_collection_query(
            "比较2025和2026项目汇总：负责人包含张三的项目分别多少项，"
            "哪些主题同时出现？"
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.group_values, ("2025", "2026"))
        self.assertTrue(plan.needs_count)
        self.assertTrue(plan.needs_intersection)

    def test_executes_complete_generic_collection_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                first = storage.documents.save(
                    url="https://example.test/2025",
                    title="2025 项目汇总",
                    text="2025年共2项项目",
                    fetched_at="2026-08-29T00:00:00Z",
                    content_html=(
                        "<p>2025年共2项项目。</p>"
                        "<p>项目名称：检索系统</p><p>负责人：张三</p>"
                        "<p>项目简介：检索与排序</p>"
                        "<p>项目名称：图像系统</p><p>负责人：李四</p>"
                        "<p>项目简介：视觉识别</p>"
                    ),
                )
                second = storage.documents.save(
                    url="https://example.test/2026",
                    title="2026 项目汇总",
                    text="2026年共3项项目",
                    fetched_at="2026-08-29T00:00:00Z",
                    content_html=(
                        "<p>2026年共3项项目。</p>"
                        "<p>项目名称：智能检索</p><p>负责人：张三</p>"
                        "<p>项目简介：智能体检索</p>"
                        "<p>项目名称：文档排序</p><p>负责人：张三、王五</p>"
                        "<p>项目简介：检索排序</p>"
                        "<p>项目名称：语音系统</p><p>负责人：赵六</p>"
                        "<p>项目简介：语音识别</p>"
                    ),
                )

                def topic_mapper(_prompt: str, **_kwargs: object) -> str:
                    return json.dumps({
                        "topics": [{
                            "label": "检索与排序",
                            "evidence": {
                                "2025": [f"{first}:record:1"],
                                "2026": [f"{second}:record:1", f"{second}:record:2"],
                            },
                        }],
                    }, ensure_ascii=False)

                response = answer_collection_question(
                    storage,
                    (
                        "比较2025和2026项目汇总：负责人包含张三的项目"
                        "分别有多少项？哪些主题同时出现？"
                    ),
                    [
                        PassageHit(first, 0, 1.0, "2025年项目", "2025 项目汇总"),
                        PassageHit(second, 0, .9, "2026年项目", "2026 项目汇总"),
                    ],
                    topic_mapper=topic_mapper,
                    debug=True,
                )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertTrue(response["complete"])
        self.assertIn("2025年1项", response["answer"])
        self.assertIn("2026年2项", response["answer"])
        self.assertIn("检索与排序", response["answer"])
        self.assertEqual(
            response["collection_trace"]["filter"],
            {"field": "负责人", "operator": "contains", "value": "张三"},
        )
        self.assertEqual(
            [source["coverage"]["record_complete"] for source in response["sources"]],
            [True, True],
        )

    def test_rejects_semantic_topic_without_every_group_evidence(self) -> None:
        # This behavior is covered through the public path: invalid semantic
        # output must fall back to inspectable lexical topics, never be trusted.
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                ids = []
                for year in (2025, 2026):
                    ids.append(storage.documents.save(
                        url=f"https://example.test/{year}",
                        title=f"{year} 记录汇总",
                        text=f"{year}年共2项记录",
                        fetched_at="2026-08-29T00:00:00Z",
                        content_html=(
                            f"<p>{year}年共2项记录。</p>"
                            "<p>记录名称：Agentic Retrieval</p><p>负责人：张三</p>"
                            "<p>记录名称：Multimodal Embedding</p><p>负责人：张三</p>"
                        ),
                    ))
                response = answer_collection_question(
                    storage,
                    "比较2025和2026记录：负责人包含张三的记录分别多少项，哪些主题共同出现？",
                    [
                        PassageHit(ids[0], 0, 1, "", "2025 记录"),
                        PassageHit(ids[1], 0, .9, "", "2026 记录"),
                    ],
                    topic_mapper=lambda *_args, **_kwargs: json.dumps({
                        "topics": [{
                            "label": "伪主题",
                            "evidence": {"2025": [f"{ids[0]}:record:1"]},
                        }],
                    }),
                    debug=True,
                )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(
            response["collection_trace"]["topic_mode"],
            "lexical_fallback",
        )
        self.assertNotIn("伪主题", response["answer"])


if __name__ == "__main__":
    unittest.main()
