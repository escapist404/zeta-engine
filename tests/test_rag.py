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
    build_prompt,
    call_model,
    merge_results,
    rag_answer,
)


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
        self.assertIn("完整名单", prompt)
        self.assertIn("逐项计数", prompt)

        cited_prompt = build_prompt(
            "申请条件是什么？",
            "[文档1]\n内容：申请条件",
            include_citations=True,
        )
        self.assertIn("使用 [文档1]、[文档2] 等标签标注依据", cited_prompt)

    def test_searches_builds_answer_and_returns_original_results(self) -> None:
        results = [{
            "title": "资助政策",
            "url": "https://example.test/policy",
            "snippet": "学生可以提交申请。",
        }]
        searches: list[tuple[str, int]] = []

        def search_fn(query: str, top_k: int) -> list[dict[str, str]]:
            searches.append((query, top_k))
            return results

        with patch("zeta_engine.rag.call_model", return_value="可以申请。") as model:
            response = rag_answer("  如何申请ＶＣＲ？ ", search_fn, top_k=3)

        self.assertEqual(searches, [("如何申请vcr?", 3)])
        self.assertEqual(response, {
            "answer": "可以申请。",
            "results": results,
        })
        self.assertIn("资助政策", model.call_args.args[0])

        with patch("zeta_engine.rag.call_model", return_value="可以申请。") as model:
            rag_answer("如何申请？", search_fn, include_citations=True)
        self.assertIn("使用 [文档1]、[文档2] 等标签", model.call_args.args[0])

    def test_skips_model_without_usable_results(self) -> None:
        with patch("zeta_engine.rag.call_model") as model:
            response = rag_answer("问题", lambda _query, _top_k: [])

        self.assertEqual(response, {
            "answer": RAG_NO_RESULTS_ANSWER,
            "results": [],
        })
        model.assert_not_called()

    def test_agentic_rag_answers_without_follow_up(self) -> None:
        results = [{
            "title": "完整证据",
            "url": "https://example.test/answer",
            "snippet": "答案是四。",
        }]

        with patch(
            "zeta_engine.rag.call_model",
            return_value='{"action":"answer","answer":"四"}',
        ) as model:
            response = agentic_rag_answer(
                "答案是多少？",
                lambda _query, _top_k: results,
            )

        self.assertEqual(response["answer"], "四")
        self.assertEqual(response["results"], results)
        self.assertEqual(model.call_count, 1)
        self.assertTrue(model.call_args.kwargs["json_output"])

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
            ],
        ) as model:
            response = agentic_rag_answer("连续三年春季课程？", search_fn)

        self.assertEqual(searches, ["连续三年春季课程?", "许洪腾 教授课程"])
        self.assertEqual(response["answer"], "机器学习")
        self.assertEqual(len(response["results"]), 2)
        self.assertIn("逐项核对", model.call_args_list[0].args[0])
        self.assertIn("分别检索两人的个人主页", model.call_args_list[0].args[0])
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
            ],
        ) as model:
            response = agentic_rag_answer("2024年人数是多少？", search_fn)

        self.assertEqual(response["answer"], "共60人")
        self.assertEqual(searches, ["2024年人数是多少?", "2024 完整名单"])
        self.assertIn("不得返回 answer", model.call_args_list[0].args[0])

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
            ],
        ) as model:
            response = agentic_rag_answer("问题", search_fn)

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
            side_effect=["not json", "最终答案"],
        ) as model:
            response = agentic_rag_answer(
                "问题",
                lambda _query, _top_k: results,
            )

        self.assertEqual(response["answer"], "最终答案")
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
