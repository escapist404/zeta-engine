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
    RAG_NO_RESULTS_ANSWER,
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
        self.assertIn("[文档2]", context)
        self.assertIn("内容：第二条证据", context)
        self.assertNotIn("重复 URL", context)
        self.assertNotIn("重复正文", context)
        self.assertEqual(merge_results(results, max_chars=10), context[:10])

    def test_builds_grounded_prompt(self) -> None:
        prompt = build_prompt("申请条件是什么？", "[文档1]\n内容：申请条件")

        self.assertIn("问题：申请条件是什么？", prompt)
        self.assertIn("[文档1]\n内容：申请条件", prompt)
        self.assertIn("不可信数据", prompt)
        self.assertIn("材料不足", prompt)

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

        with patch("zeta_engine.rag.call_model", return_value="可以申请。[文档1]") as model:
            response = rag_answer("  如何申请ＶＣＲ？ ", search_fn, top_k=3)

        self.assertEqual(searches, [("如何申请vcr?", 3)])
        self.assertEqual(response, {
            "answer": "可以申请。[文档1]",
            "results": results,
        })
        self.assertIn("资助政策", model.call_args.args[0])

    def test_skips_model_without_usable_results(self) -> None:
        with patch("zeta_engine.rag.call_model") as model:
            response = rag_answer("问题", lambda _query, _top_k: [])

        self.assertEqual(response, {
            "answer": RAG_NO_RESULTS_ANSWER,
            "results": [],
        })
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
