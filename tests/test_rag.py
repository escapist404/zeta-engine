import json
import os
import unittest
from unittest.mock import patch

from zeta_engine.rag import Evidence, LLM_API_KEY_ENV, agentic_rag_answer, call_model


def verification(
    *,
    valid: bool = True,
    satisfied: bool = True,
    status: str = "supported",
    resolved: bool = True,
    issues: list[dict[str, str]] | None = None,
    queries: list[str] | None = None,
    calculation_ids: list[int | str] | None = None,
    evidence_ids: list[str] | None = None,
) -> str:
    references = (
        {"evidence_ids": evidence_ids}
        if evidence_ids is not None
        else {"source_ids": [1]}
    )
    return json.dumps({
        "valid": valid,
        "requirements": [{
            "description": "回答问题",
            "satisfied": satisfied,
            **references,
        }],
        "claims": [{
            "statement": "候选答案",
            "status": status,
            **references,
            "calculation_ids": calculation_ids or [],
        }],
        "conflicts": ([] if resolved else [{
            "description": "候选值冲突",
            "resolved": False,
            **references,
        }]),
        "issues": issues or [],
        "queries": queries or [],
    }, ensure_ascii=False)


class RagTest(unittest.TestCase):
    def test_complete_table_ratio_uses_agent_calculation_and_verifier(self) -> None:
        initial = {
            "title": "2021年夏令营公告",
            "url": "https://example.test/camp",
            "content": "拟推荐优秀营员不少于34人。",
        }
        tool_result = {
            "title": "2021年夏令营公告 · 学术硕士营员名单",
            "url": "https://example.test/camp",
            "content": "完整扫描学硕名单共127人；拟推荐不少于34人。",
            "evidence_scope": "complete_table_section",
            "collection_complete": "true",
        }
        evidence_id = Evidence.from_result(tool_result).evidence_id
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "collection",
                    "query": "扫描完整学术硕士名单并计算最低录取比",
                    "slots": [],
                }),
                json.dumps({
                    "action": "calculate",
                    "calculations": [{
                        "name": "最低录取比",
                        "operator": "ratio",
                        "items": [
                            {"value": 34, "evidence_ids": [evidence_id]},
                            {"value": 127, "evidence_ids": [evidence_id]},
                        ],
                    }],
                    "slots": [],
                }),
                '{"action":"answer","answer":"最低录取比至少约为26.77%"}',
                verification(calculation_ids=["最低录取比"]),
            ],
        ) as model:
            response = agentic_rag_answer(
                "2021年夏令营学硕最低录取比",
                lambda _query, _top_k: [initial],
                collection_fn=lambda _tool_query: [tool_result],
                debug=True,
            )

        self.assertEqual(response["answer"], "最低录取比至少约为26.77%")
        self.assertEqual(model.call_count, 4)
        self.assertEqual(
            [item["action"] for item in response["trace"]],
            ["collection", "calculate", "verified_answer"],
        )

    def test_rejects_empty_prompt_and_missing_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt"):
            call_model("  ")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, LLM_API_KEY_ENV):
                call_model("问题")

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

    def test_agentic_rag_tracks_slots_across_search_cycles(self) -> None:
        first_result = {
            "url": "https://example.test/a",
            "content": "甲参加项目一",
        }
        second_result = {
            "url": "https://example.test/b",
            "content": "乙参加项目二",
        }
        first_id = Evidence.from_result(first_result).evidence_id
        second_id = Evidence.from_result(second_result).evidence_id

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            return [second_result] if query == "乙 项目名单" else [first_result]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "search",
                    "slots": [
                        {
                            "id": "a",
                            "question": "甲参加的项目",
                            "status": "answered",
                            "evidence_ids": [first_id],
                        },
                        {
                            "id": "b",
                            "question": "乙参加的项目",
                            "status": "missing",
                            "evidence_ids": [],
                        },
                    ],
                    "queries": [{"slot_id": "b", "query": "乙 项目名单"}],
                }, ensure_ascii=False),
                json.dumps({
                    "action": "answer",
                    "answer": "甲参加项目一，乙参加项目二",
                    "slots": [
                        {
                            "id": "a",
                            "question": "甲参加的项目",
                            "status": "answered",
                            "evidence_ids": [first_id],
                        },
                        {
                            "id": "b",
                            "question": "乙参加的项目",
                            "status": "answered",
                            "evidence_ids": [second_id],
                        },
                    ],
                    "claims": [],
                }, ensure_ascii=False),
                verification(evidence_ids=[first_id, second_id]),
            ],
        ):
            response = agentic_rag_answer(
                "甲和乙分别参加什么项目？",
                search_fn,
                max_cycles=2,
                debug=True,
            )

        self.assertEqual(
            response["trace"][0]["query_slots"],
            {"乙 项目名单": "b"},
        )
        self.assertEqual(
            [slot["status"] for slot in response["trace"][0]["answer_slots"]],
            ["answered", "missing"],
        )
        self.assertEqual(
            [slot["id"] for slot in response["answer_slots"]],
            ["a", "b"],
        )
        self.assertEqual(
            [slot["status"] for slot in response["answer_slots"]],
            ["answered", "answered"],
        )

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

    def test_returns_claim_level_stable_sources(self) -> None:
        result = {
            "title": "证据",
            "url": "https://example.test/evidence",
            "content": "答案是四。",
        }
        evidence_id = Evidence.from_result(result).evidence_id
        answer = json.dumps({
            "action": "answer",
            "answer": "四",
            "claims": [{
                "statement": "答案是四",
                "evidence_ids": [evidence_id],
                "calculation_ids": [],
            }],
        }, ensure_ascii=False)

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                answer,
                verification(evidence_ids=[evidence_id]),
            ],
        ):
            response = agentic_rag_answer(
                "答案是多少？",
                lambda _query, _top_k: [result],
            )

        self.assertEqual(response["claims"][0]["evidence_ids"], [evidence_id])
        self.assertEqual(response["sources"][0]["evidence_id"], evidence_id)

if __name__ == "__main__":
    unittest.main()
