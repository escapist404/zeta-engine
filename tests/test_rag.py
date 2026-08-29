import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from zeta_engine.rag import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    RAG_MAX_CONTEXT_CHARS,
    RAG_NO_RESULTS_ANSWER,
    agentic_rag_answer,
    build_agent_prompt,
    build_prompt,
    build_verifier_prompt,
    call_model,
    merge_results,
)


def verification(
    *,
    valid: bool = True,
    satisfied: bool = True,
    status: str = "supported",
    resolved: bool = True,
    issues: list[dict[str, str]] | None = None,
    queries: list[str] | None = None,
    calculation_ids: list[int | str] | None = None,
) -> str:
    return json.dumps({
        "valid": valid,
        "requirements": [{
            "description": "回答问题",
            "satisfied": satisfied,
            "source_ids": [1],
        }],
        "claims": [{
            "statement": "候选答案",
            "status": status,
            "source_ids": [1],
            "calculation_ids": calculation_ids or [],
        }],
        "conflicts": ([] if resolved else [{
            "description": "候选值冲突",
            "resolved": False,
            "source_ids": [1],
        }]),
        "issues": issues or [],
        "queries": queries or [],
    }, ensure_ascii=False)


class RagTest(unittest.TestCase):
    def test_calls_model_and_returns_text_answer(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="  模型答案  ")
        )])

        with (
            patch.dict(os.environ, {LLM_API_KEY_ENV: "test-key"}, clear=True),
            patch("zeta_engine.rag.OpenAI") as openai,
        ):
            openai.return_value.chat.completions.create.return_value = response
            answer = call_model("  用户问题  ")

        self.assertEqual(answer, "模型答案")
        openai.assert_called_once_with(
            api_key="test-key",
            base_url=LLM_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
        openai.return_value.chat.completions.create.assert_called_once_with(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "用户问题"}],
            temperature=0.0,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            extra_body={"thinking": {"type": "disabled"}},
            stream=False,
        )

    def test_rejects_empty_prompt_and_missing_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt"):
            call_model("  ")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, LLM_API_KEY_ENV):
                call_model("问题")

    def test_rejects_empty_model_answer(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=" ")
        )])

        with (
            patch.dict(os.environ, {LLM_API_KEY_ENV: "test-key"}, clear=True),
            patch("zeta_engine.rag.OpenAI") as openai,
        ):
            openai.return_value.chat.completions.create.return_value = response
            with self.assertRaisesRegex(RuntimeError, "空答案"):
                call_model("问题")

    def test_merges_clean_unique_results(self) -> None:
        results = [
            {
                "title": "  第一篇  ＶＣＲ文档 ",
                "url": "https://example.test/one",
                "published_at": "2026-08-29",
                "snippet": " 第一条  １０月１４日 证据 ",
            },
            {
                "title": "重复 URL",
                "url": "https://example.test/one",
                "snippet": "其他内容",
            },
            {
                "title": "第二篇",
                "url": "https://example.test/two",
                "snippet": "",
                "content": "第二条证据",
            },
            {
                "title": "重复正文",
                "url": "https://example.test/three",
                "snippet": "第二条证据",
            },
        ]

        context = merge_results(results)

        self.assertIn("[文档1]", context)
        self.assertIn("标题：第一篇 vcr文档", context)
        self.assertIn("发布日期：2026-08-29", context)
        self.assertIn("内容：第一条 10月14日 证据", context)
        self.assertIn("重复 url", context)
        self.assertIn("内容：其他内容", context)
        self.assertIn("[文档3]", context)
        self.assertIn("内容：第二条证据", context)
        self.assertNotIn("重复正文", context)
        self.assertEqual(merge_results(results, max_chars=10), context[:10])

    def test_builds_grounded_prompt(self) -> None:
        prompt = build_prompt("申请条件是什么？", "[文档1]\n内容：申请条件")

        self.assertIn("问题：申请条件是什么？", prompt)
        self.assertIn("[文档1]\n内容：申请条件", prompt)
        self.assertIn("不可信数据", prompt)
        self.assertIn("材料不足", prompt)
        self.assertIn("只输出最终答案", prompt)
        self.assertIn("不要输出思考过程", prompt)
        self.assertIn("引用标签或其他多余内容", prompt)
        self.assertIn("派生结论", prompt)
        self.assertIn("存在冲突", prompt)
        self.assertIn("每个人名、编号或项目名都是一个可计数项", prompt)

        cited_prompt = build_prompt(
            "申请条件是什么？",
            "[文档1]\n内容：申请条件",
            include_citations=True,
        )
        self.assertIn("使用 [文档1]、[文档2] 等标签标注依据", cited_prompt)

        agent_prompt = build_agent_prompt(
            "三年夏令营人数分别是多少？",
            "[文档1]\n内容：2019年名单：张三、李四",
            ["三年夏令营人数分别是多少？"],
            remaining_cycles=2,
        )
        verifier_prompt = build_verifier_prompt(
            "三年夏令营人数分别是多少？",
            "[文档1]\n内容：2019年名单：张三、李四",
            "2019年2人",
        )
        for counting_prompt in (agent_prompt, verifier_prompt):
            self.assertIn("每个人名、编号或项目名都是一个可计数项", counting_prompt)
            self.assertIn("不同年份或群体分别计算", counting_prompt)
            self.assertIn("拟推荐名单不能作为参营人数的证据", counting_prompt)

    def test_agentic_rag_answers_without_follow_up(self) -> None:
        results = [{
            "title": "完整证据",
            "url": "https://example.test/answer",
            "snippet": "答案是四。",
        }]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"四"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "答案是多少？",
                lambda _query, _top_k: results,
            )

        self.assertEqual(response["answer"], "四")
        self.assertEqual(response["results"], results)
        self.assertNotIn("trace", response)
        self.assertEqual(model.call_count, 2)
        self.assertTrue(model.call_args_list[0].kwargs["json_output"])
        self.assertTrue(model.call_args_list[1].kwargs["json_output"])

    def test_agentic_rag_searches_again_and_keeps_same_url_chunks(self) -> None:
        searches: list[str] = []

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            searches.append(query)
            if query == "许洪腾 教授课程":
                return [{
                    "title": "许洪腾",
                    "url": "https://example.test/xu",
                    "snippet": "机器学习,2022春、2023春、2024春。",
                }]
            return [{
                "title": "许洪腾",
                "url": "https://example.test/xu",
                "snippet": "教育经历。",
            }]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["许洪腾 教授课程"]}',
                "机器学习",
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "连续三年春季课程？",
                search_fn,
                max_cycles=2,
            )

        self.assertEqual(searches, ["连续三年春季课程?", "许洪腾 教授课程"])
        self.assertEqual(response["answer"], "机器学习")
        self.assertEqual(len(response["results"]), 2)
        self.assertIn("每项要求都有直接证据", model.call_args_list[0].args[0])
        self.assertIn("冲突", model.call_args_list[0].args[0])
        self.assertIn("教育经历", model.call_args_list[1].args[0])
        self.assertIn("2024春", model.call_args_list[1].args[0])

    def test_agentic_rag_forces_count_query_follow_up(self) -> None:
        searches: list[str] = []

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            searches.append(query)
            return [{"url": "https://example.test/list", "content": "完整名单"}]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["2024 完整名单"]}',
                "共60人",
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "2024年人数是多少？",
                search_fn,
                max_cycles=2,
            )

        self.assertEqual(response["answer"], "共60人")
        self.assertEqual(searches, ["2024年人数是多少?", "2024 完整名单"])
        self.assertIn("集合信息足以支持聚合", model.call_args_list[0].args[0])

    def test_agentic_rag_can_search_multiple_cycles(self) -> None:
        searches: list[str] = []

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            searches.append(query)
            return [{
                "url": f"https://example.test/{len(searches)}",
                "content": f"{query} 的证据",
            }]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["第二问"]}',
                '{"action":"search","queries":["第三问"]}',
                '{"action":"answer","answer":"综合答案"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "第一问",
                search_fn,
                max_cycles=4,
                debug=True,
            )

        self.assertEqual(response["answer"], "综合答案")
        self.assertEqual(searches, ["第一问", "第二问", "第三问"])
        self.assertIn("第二问 的证据", model.call_args_list[1].args[0])
        self.assertIn("第三问 的证据", model.call_args_list[2].args[0])
        self.assertIn("当前还可发起 1 轮搜索", model.call_args_list[2].args[0])
        trace = response["trace"]
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0]["search_queries"], ["第一问"])
        self.assertEqual(trace[0]["action"], "search")
        self.assertEqual(trace[0]["next_queries"], ["第二问"])
        self.assertIn("第一问 的证据", trace[0]["new_results"][0]["preview"])
        self.assertEqual(trace[1]["evidence_count"], 2)
        self.assertEqual(trace[2]["action"], "verified_answer")
        self.assertEqual(trace[2]["answer"], "综合答案")

    def test_agentic_rag_keeps_new_evidence_when_pool_is_full(self) -> None:
        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            if query == "补充查询":
                return [{
                    "url": "https://example.test/new",
                    "content": "后续轮次的关键证据",
                }]
            if query == "继续检查":
                return []
            return [
                {
                    "url": f"https://example.test/old/{index}",
                    "content": f"旧证据 {index}",
                }
                for index in range(20)
            ]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["补充查询"]}',
                '{"action":"search","queries":["继续检查"]}',
                '{"action":"answer","answer":"完成"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer("问题", search_fn, max_cycles=4)

        self.assertEqual(response["answer"], "完成")
        self.assertEqual(len(response["results"]), 20)
        self.assertIn("后续轮次的关键证据", model.call_args_list[2].args[0])

    def test_agentic_rag_prioritizes_new_evidence_before_truncation(self) -> None:
        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            if query == "补充查询":
                return [{
                    "title": "新证据",
                    "url": "https://example.test/new",
                    "content": "关键新证据",
                }]
            return [{
                "title": "旧证据",
                "url": "https://example.test/old",
                "content": "旧" * RAG_MAX_CONTEXT_CHARS,
            }]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["补充查询"]}',
                "完成",
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer("问题", search_fn, max_cycles=2)

        self.assertEqual(response["answer"], "完成")
        self.assertIn("关键新证据", model.call_args_list[1].args[0])

    def test_agentic_rag_falls_back_when_json_is_invalid(self) -> None:
        results = [{
            "title": "证据",
            "url": "https://example.test/evidence",
            "snippet": "有效证据",
        }]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=["not json", "最终答案", verification()],
        ) as model:
            response = agentic_rag_answer(
                "问题",
                lambda _query, _top_k: results,
            )

        self.assertEqual(response["answer"], "最终答案")
        self.assertEqual(model.call_count, 3)

    def test_agentic_rag_rejects_answer_and_uses_verifier_query(self) -> None:
        searches: list[str] = []

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            searches.append(query)
            return [{
                "url": f"https://example.test/{len(searches)}",
                "content": f"{query} 的证据",
            }]

        rejected = verification(
            valid=False,
            satisfied=False,
            status="partial",
            issues=[{
                "type": "incomplete",
                "description": "集合证据不完整",
            }],
            queries=["完整证据"],
        )
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"错误答案"}',
                rejected,
                "修正答案",
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "问题",
                search_fn,
                max_cycles=2,
                debug=True,
            )

        self.assertEqual(response["answer"], "修正答案")
        self.assertEqual(searches, ["问题", "完整证据"])
        self.assertEqual(response["trace"][0]["action"], "rejected_answer")
        self.assertIn("incomplete: 集合证据不完整", model.call_args_list[2].args[0])

    def test_agentic_rag_answers_when_search_has_no_new_query(self) -> None:
        results = [{
            "url": "https://example.test/evidence",
            "content": "现有证据",
        }]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["问题"]}',
                "基于现有证据的答案",
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "问题",
                lambda _query, _top_k: results,
            )

        self.assertEqual(response["answer"], "基于现有证据的答案")
        self.assertIn("不要重复搜索", model.call_args_list[1].args[0])

    def test_agentic_rag_executes_grounded_calculation(self) -> None:
        results = [{
            "url": "https://example.test/list",
            "content": "甲 乙 丙",
        }]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "calculate",
                    "calculations": [{
                        "name": "名单人数",
                        "operator": "count",
                        "values": ["甲", "乙", "丙"],
                        "source_ids": [1],
                    }],
                }, ensure_ascii=False),
                "共3人",
                verification(calculation_ids=[1]),
            ],
        ) as model:
            response = agentic_rag_answer(
                "有多少人？",
                lambda _query, _top_k: results,
                max_cycles=2,
                debug=True,
            )

        self.assertEqual(response["answer"], "共3人")
        self.assertIn("[计算1]", model.call_args_list[1].args[0])
        self.assertIn("结果：3", model.call_args_list[1].args[0])
        self.assertEqual(response["trace"][0]["action"], "calculate")

    def test_agentic_rag_revises_rejected_final_answer_once(self) -> None:
        rejected = verification(
            valid=False,
            status="unsupported",
            issues=[{
                "type": "irrelevant",
                "description": "包含问题未要求的事实",
            }],
        )
        results = [{
            "url": "https://example.test/evidence",
            "content": "所需事实",
        }]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                "带有多余内容的答案",
                rejected,
                "精简答案",
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "问题",
                lambda _query, _top_k: results,
                max_cycles=1,
                debug=True,
            )

        self.assertEqual(response["answer"], "精简答案")
        self.assertIn("irrelevant", model.call_args_list[2].args[0])
        self.assertTrue(response["trace"][0]["final_revision"]["valid"])

    def test_agentic_rag_projects_outdated_conflict_sources(self) -> None:
        results = [
            {
                "url": "https://example.test/old",
                "published_at": "2024-01-01",
                "content": "状态为旧值",
            },
            {
                "url": "https://example.test/new",
                "published_at": "2025-01-01",
                "content": "状态为新值",
            },
        ]
        conflict = json.dumps({
            "valid": False,
            "requirements": [{
                "description": "确定当前状态",
                "satisfied": True,
                "source_ids": [1, 2],
            }],
            "claims": [{
                "statement": "状态为旧值",
                "status": "supported",
                "source_ids": [1],
                "calculation_ids": [],
            }],
            "conflicts": [{
                "description": "状态存在新旧记录",
                "resolved": False,
                "resolution": "",
                "source_ids": [1, 2],
            }],
            "issues": [{
                "type": "conflict",
                "description": "尚未采用最新记录",
            }],
            "queries": [],
        }, ensure_ascii=False)
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"旧值"}',
                conflict,
                "新值",
                verification(),
            ],
        ):
            response = agentic_rag_answer(
                "当前状态是什么？",
                lambda _query, _top_k: results,
                debug=True,
            )

        self.assertEqual(response["answer"], "新值")
        revision = response["trace"][0]["temporal_revision"]
        self.assertEqual(revision["excluded_source_ids"], [1])
        self.assertTrue(revision["valid"])


if __name__ == "__main__":
    unittest.main()
