import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from zeta_engine.rag import (
    AGENT_MAX_CYCLES,
    Evidence,
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_BASE_URL_ENV,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_MODEL_ENV,
    LLM_TIMEOUT_SECONDS,
    RAG_MAX_CONTEXT_CHARS,
    RAG_NO_RESULTS_ANSWER,
    _parse_agent_action,
    _parse_verification,
    _deterministic_numbered_event_answer,
    _deterministic_collection_bucket_answer,
    _deterministic_complete_table_ratio_answer,
    _range_classification_rules,
    _requires_complete_collection,
    _run_calculations,
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
    def test_default_cycle_budget_allows_post_retrieval_repair(self) -> None:
        self.assertGreaterEqual(AGENT_MAX_CYCLES, 5)

    def test_parses_collection_tool_action(self) -> None:
        decision = _parse_agent_action(json.dumps({
            "action": "collection",
            "query": "完整扫描目标名单并保留分组边界",
            "slots": [],
        }, ensure_ascii=False))

        self.assertEqual(decision.action, "collection")
        self.assertEqual(
            decision.collection_query,
            "完整扫描目标名单并保留分组边界",
        )

    def test_accepts_text_alias_for_agent_claim(self) -> None:
        decision = _parse_agent_action(json.dumps({
            "action": "answer",
            "answer": "答案",
            "claims": [{
                "text": "可验证断言",
                "evidence_ids": ["ev_1"],
                "calculation_ids": [],
            }],
            "slots": [],
        }, ensure_ascii=False))

        self.assertEqual(decision.claims[0]["statement"], "可验证断言")

    def test_grounded_ratio_uses_two_operands(self) -> None:
        evidence = Evidence.from_result({
            "title": "完整名单",
            "url": "https://example.test/list",
            "content": "拟推荐人数不少于34人，完整名单共127人。",
        })

        completed = _run_calculations([{
            "name": "最低比例",
            "operator": "ratio",
            "items": [
                {"value": 34, "evidence_ids": [evidence.evidence_id]},
                {"value": 127, "evidence_ids": [evidence.evidence_id]},
            ],
        }], [evidence], [])

        self.assertAlmostEqual(completed[0]["result"], 34 / 127)

    def test_parses_answer_slots_and_slot_scoped_queries(self) -> None:
        decision = _parse_agent_action(json.dumps({
            "action": "search",
            "slots": [
                {
                    "id": "person_a",
                    "question": "甲参加了哪些项目",
                    "status": "answered",
                    "evidence_ids": ["ev_a"],
                },
                {
                    "id": "person_b",
                    "question": "乙参加了哪些项目",
                    "status": "missing",
                    "evidence_ids": [],
                },
            ],
            "queries": [{
                "slot_id": "person_b",
                "query": "乙 完整项目名单",
            }],
        }, ensure_ascii=False))

        self.assertEqual(decision.action, "search")
        self.assertEqual(decision.queries, ["乙 完整项目名单"])
        self.assertEqual(
            decision.query_slots,
            {"乙 完整项目名单": "person_b"},
        )
        self.assertEqual(
            [slot.status for slot in decision.slots],
            ["answered", "missing"],
        )

    def test_rejects_query_for_unknown_answer_slot(self) -> None:
        with self.assertRaisesRegex(ValueError, "slot_id"):
            _parse_agent_action(json.dumps({
                "action": "search",
                "slots": [{
                    "id": "s1",
                    "question": "第一个答案",
                    "status": "missing",
                    "evidence_ids": [],
                }],
                "queries": [{"slot_id": "s2", "query": "补充搜索"}],
            }, ensure_ascii=False))

    def test_rejects_query_for_answered_slot(self) -> None:
        with self.assertRaisesRegex(ValueError, "已回答"):
            _parse_agent_action(json.dumps({
                "action": "search",
                "slots": [{
                    "id": "s1",
                    "question": "第一个答案",
                    "status": "answered",
                    "evidence_ids": ["ev_1"],
                }],
                "queries": [{"slot_id": "s1", "query": "重复搜索"}],
            }, ensure_ascii=False))

    def test_accepts_claim_alias_from_model_output(self) -> None:
        decision = _parse_agent_action(json.dumps({
            "action": "answer",
            "answer": "答案",
            "claims": [{
                "claim": "关键断言",
                "evidence_ids": ["ev_1"],
                "calculation_ids": [],
            }],
        }, ensure_ascii=False))

        self.assertEqual(decision.claims[0]["statement"], "关键断言")

    def test_accepts_string_issues_and_keeps_verifier_queries(self) -> None:
        response = json.dumps({
            "valid": False,
            "requirements": [{
                "description": "补齐缺失项",
                "satisfied": False,
                "evidence_ids": ["ev_1"],
            }],
            "claims": [{
                "statement": "现有部分答案",
                "status": "supported",
                "evidence_ids": ["ev_1"],
                "calculation_ids": [],
            }],
            "conflicts": [],
            "issues": ["还缺少一项明确要求"],
            "queries": ["缺失项 精确查询"],
        }, ensure_ascii=False)

        valid, feedback, queries, ledger = _parse_verification(
            response,
            allowed_evidence_ids=["ev_1"],
        )

        self.assertFalse(valid)
        self.assertIn("还缺少一项明确要求", feedback)
        self.assertEqual(queries, ["缺失项 精确查询"])
        self.assertEqual(ledger["issues"][0]["type"], "incomplete")

    def test_executes_grounded_set_operations_and_counts_results(self) -> None:
        evidence = [Evidence.from_result({
            "url": "https://example.test/lists",
            "content": "甲 乙 丙 丁",
        })]
        evidence_id = evidence[0].evidence_id
        set_a = {"values": ["甲", "乙", "丙"], "evidence_ids": [evidence_id]}
        set_b = {"values": ["乙", "丙", "丁"], "evidence_ids": [evidence_id]}

        calculations = _run_calculations(
            [
                {"name": "交集", "operator": "intersection", "sets": [set_a, set_b]},
                {"name": "并集", "operator": "union", "sets": [set_a, set_b]},
                {"name": "补集", "operator": "complement", "sets": [set_a, set_b]},
                {
                    "name": "交集人数",
                    "operator": "count",
                    "values": [{"calculation": "交集"}],
                    "evidence_ids": [evidence_id],
                },
            ],
            evidence,
            [],
        )

        self.assertEqual(calculations[0]["result"], ["乙", "丙"])
        self.assertEqual(calculations[1]["result"], ["甲", "乙", "丙", "丁"])
        self.assertEqual(calculations[2]["result"], ["甲"])
        self.assertEqual(calculations[3]["result"], 2)
        self.assertEqual(calculations[3]["evidence_ids"], [evidence_id])

    def test_rejects_identical_calculations_with_different_names(self) -> None:
        evidence = [Evidence.from_result({
            "url": "https://example.test/members",
            "content": "成员：甲、乙",
        })]
        evidence_id = evidence[0].evidence_id
        specifications = [
            {
                "name": "第一个名称",
                "operator": "count",
                "values": ["甲", "乙"],
                "evidence_ids": [evidence_id],
            },
            {
                "name": "另一个名称",
                "operator": "count",
                "values": ["甲", "乙"],
                "evidence_ids": [evidence_id],
            },
        ]

        with self.assertRaisesRegex(ValueError, "完全相同的计算"):
            _run_calculations(specifications, evidence, [])

    def test_allows_different_operations_on_the_same_input(self) -> None:
        evidence = [Evidence.from_result({
            "url": "https://example.test/members",
            "content": "成员：甲、甲、乙",
        })]
        evidence_id = evidence[0].evidence_id

        calculations = _run_calculations(
            [
                {
                    "name": "记录数",
                    "operator": "count",
                    "values": ["甲", "甲", "乙"],
                    "evidence_ids": [evidence_id],
                },
                {
                    "name": "去重数",
                    "operator": "count_unique",
                    "values": ["甲", "甲", "乙"],
                    "evidence_ids": [evidence_id],
                },
            ],
            evidence,
            [],
        )

        self.assertEqual([item["result"] for item in calculations], [3, 2])

    def test_classifies_a_grounded_number_with_explicit_boundaries(self) -> None:
        evidence = [Evidence.from_result({
            "url": "https://example.test/count",
            "content": "2019年共有59人",
        })]
        evidence_id = evidence[0].evidence_id

        calculations = _run_calculations(
            [
                {
                    "name": "人数",
                    "operator": "max",
                    "values": [59],
                    "evidence_ids": [evidence_id],
                },
                {
                    "name": "教室类型",
                    "operator": "classify",
                    "items": [{"calculation": "人数"}],
                    "rules": [
                        {"label": "小型教室", "lt": 60},
                        {"label": "中型教室", "gte": 60, "lte": 100},
                        {"label": "大型教室", "gte": 101, "lte": 200},
                        {"label": "特大型教室", "gt": 200},
                    ],
                },
            ],
            evidence,
            [],
        )

        self.assertEqual(calculations[1]["result"], "小型教室")
        self.assertEqual(calculations[1]["evidence_ids"], [evidence_id])

    def test_event_total_uses_terminal_grounded_calculation(self) -> None:
        results = [{
            "title": "第一次企业参访",
            "url": "https://example.test/one",
            "content": "第一次企业参访",
        }, {
            "title": "第二次企业参访",
            "url": "https://example.test/two",
            "content": "第二次企业参访",
        }]
        evidence_ids = [
            Evidence.from_result(result).evidence_id for result in results
        ]
        decision = json.dumps({
            "action": "calculate",
            "calculations": [
                {
                    "name": "第一类次数",
                    "operator": "count_unique",
                    "values": [evidence_ids[0]],
                    "evidence_ids": [evidence_ids[0]],
                },
                {
                    "name": "第二类次数",
                    "operator": "count_unique",
                    "values": [evidence_ids[1]],
                    "evidence_ids": [evidence_ids[1]],
                },
                {
                    "name": "总次数",
                    "operator": "sum",
                    "values": [
                        {"calculation": "第一类次数"},
                        {"calculation": "第二类次数"},
                    ],
                    "evidence_ids": evidence_ids,
                },
            ],
        }, ensure_ascii=False)

        with patch("zeta_engine.rag.call_model", return_value=decision) as model:
            response = agentic_rag_answer(
                "两类企业参访一共多少次？",
                lambda _query, _top_k: results,
                debug=True,
            )

        self.assertEqual(response["answer"], "一共2次。")
        self.assertEqual(response["trace"][0]["action"], "deterministic_calculation")
        model.assert_called_once()

    def test_numbered_event_count_deduplicates_listing_and_detail_pages(self) -> None:
        results = [
            {
                "title": "企业参访第七站：快手公司参访纪实",
                "url": "https://example.test/kuaishou-7",
                "content": "第七站前往快手公司。",
            },
            {
                "title": "活动列表",
                "url": "https://example.test/list",
                "content": (
                    "企业参访第7站：快手公司；"
                    "企业参访第14站：快手公司；"
                    "企业参访第15站：腾讯公司；"
                    "企业参访第16站：腾讯公司。"
                ),
            },
            {
                "title": "腾讯就业宣讲会",
                "url": "https://example.test/recruiting",
                "content": "腾讯公司举办就业宣讲会，不属于企业参访。",
            },
        ]
        evidence = [Evidence.from_result(result) for result in results]

        answer, claims = _deterministic_numbered_event_answer(
            "企业参访中，参访快手公司和腾讯公司一共多少次？",
            evidence,
        )

        self.assertEqual(answer, "一共4次。")
        self.assertEqual(len(claims), 4)

    def test_complete_collection_sizes_are_bucketed_by_query_rules(self) -> None:
        evidence = [
            Evidence.from_result({
                "title": f"{year}年项目成员名单",
                "url": f"https://example.test/{year}",
                "content": f"完整表格集合共{size}项",
                "collection_size": str(size),
                "collection_complete": "true",
            })
            for year, size in ((2019, 59), (2020, 65), (2021, 257))
        ]
        query = (
            "不满60人使用小型教室，60至100人使用中型教室，"
            "101至200人使用大型教室，超过200人使用特大型教室。"
            "2019、2020和2021年分别使用哪类教室？"
        )

        answer, claims = _deterministic_collection_bucket_answer(query, evidence)

        self.assertEqual(
            answer,
            "2019年使用小型教室；2020年使用中型教室；"
            "2021年使用特大型教室。",
        )
        self.assertEqual(len(claims), 3)

    def test_document_aggregate_wins_over_conflicting_subtable_sizes(self) -> None:
        evidence = []
        for year, total, subtables in (
            (2019, 59, (40, 19)),
            (2020, 65, (45, 20)),
            (2021, 257, (127, 130)),
        ):
            evidence.append(Evidence.from_result({
                "title": f"{year}年项目名单 · 全部表格",
                "content": f"文档级完整集合共{total}项",
                "collection_size": str(total),
                "collection_complete": "true",
                "evidence_scope": "complete_document_tables",
            }))
            evidence.extend(Evidence.from_result({
                "title": f"{year}年项目名单 · 子表{index}",
                "content": f"完整子表共{size}项",
                "collection_size": str(size),
                "collection_complete": "true",
                "evidence_scope": "complete_table_section",
            }) for index, size in enumerate(subtables, start=1))
        query = (
            "不满60人使用小型教室，60至100人使用中型教室，"
            "101至200人使用大型教室，超过200人使用特大型教室。"
            "2019、2020和2021年分别使用哪类教室？"
        )

        answer, claims = _deterministic_collection_bucket_answer(query, evidence)

        self.assertEqual(
            answer,
            "2019年使用小型教室；2020年使用中型教室；"
            "2021年使用特大型教室。",
        )
        self.assertEqual(len(claims), 3)

    def test_complete_table_ratio_bound_is_materialized_without_another_model_call(self) -> None:
        def result(section: str, numerator: int, denominator: int) -> dict[str, str]:
            return {
                "title": f"2021年夏令营名单 · {section}",
                "url": "https://example.test/camp",
                "content": "完整表格与同章节数量下界已扫描。",
                "evidence_scope": "complete_table_section",
                "collection_complete": "true",
                "derived_relations": json.dumps([{
                    "kind": "ratio_bound",
                    "direction": "lower",
                    "numerator": float(numerator),
                    "denominator": denominator,
                    "ratio": numerator / denominator,
                }]),
            }

        answer, claims = _deterministic_complete_table_ratio_answer(
            "2021年夏令营学硕最低录取比",
            [
                Evidence.from_result(result("学术硕士营员名单", 34, 127)),
                Evidence.from_result(result("直博营员名单", 39, 130)),
            ],
        )

        self.assertEqual(answer, "最低比例为26.77%（34/127≈0.2677）。")
        self.assertEqual(len(claims), 1)

    def test_bounded_ratio_requires_complete_collection(self) -> None:
        self.assertTrue(_requires_complete_collection("项目的最低录取比是多少？"))
        self.assertTrue(_requires_complete_collection("最高通过率"))
        self.assertTrue(_requires_complete_collection(
            "不满60人使用小型教室，60至100人使用中型教室，"
            "2019、2020年分别使用哪类教室？"
        ))
        self.assertFalse(_requires_complete_collection("录取人数是多少？"))
        self.assertFalse(_requires_complete_collection("录取比例是多少？"))

    def test_agentic_rag_closes_after_complete_table_ratio_tool_result(self) -> None:
        initial = {
            "title": "2021年夏令营公告",
            "url": "https://example.test/camp",
            "content": "拟推荐优秀营员不少于34人。",
        }
        tool_result = {
            "title": "2021年夏令营公告 · 学术硕士营员名单",
            "url": "https://example.test/camp",
            "content": "完整表格与同章节数量下界已扫描。",
            "evidence_scope": "complete_table_section",
            "collection_complete": "true",
            "derived_relations": json.dumps([{
                "kind": "ratio_bound",
                "direction": "lower",
                "numerator": 34.0,
                "denominator": 127,
                "ratio": 34 / 127,
            }]),
        }
        with patch(
            "zeta_engine.rag.call_model",
            return_value=json.dumps({
                "action": "collection",
                "query": "扫描完整学术硕士名单并计算最低录取比",
                "slots": [],
            }),
        ) as model:
            response = agentic_rag_answer(
                "2021年夏令营学硕最低录取比",
                lambda _query, _top_k: [initial],
                collection_fn=lambda _tool_query: [tool_result],
                debug=True,
            )

        self.assertEqual(response["answer"], "最低比例为26.77%（34/127≈0.2677）。")
        self.assertEqual(model.call_count, 1)
        self.assertEqual(
            [item["action"] for item in response["trace"]],
            ["collection", "deterministic_table_ratio"],
        )

    def test_bounded_ratio_overrides_agent_refusal_with_collection(self) -> None:
        initial = {
            "title": "项目公告",
            "url": "https://example.test/camp",
            "content": "入选者不少于3人。",
        }
        tool_result = {
            "title": "项目公告 · 完整名单",
            "url": "https://example.test/camp",
            "content": "完整表格与同章节数量下界已扫描。",
            "evidence_scope": "complete_table_section",
            "collection_complete": "true",
            "derived_relations": json.dumps([{
                "kind": "ratio_bound",
                "direction": "lower",
                "numerator": 3.0,
                "denominator": 8,
                "ratio": 3 / 8,
            }]),
        }
        with patch(
            "zeta_engine.rag.call_model",
            return_value=(
                '{"action":"answer","answer":"缺少申请人数，无法计算",'
                '"claims":[],"slots":[]}'
            ),
        ) as model:
            response = agentic_rag_answer(
                "项目最低录取比是多少？",
                lambda _query, _top_k: [initial],
                collection_fn=lambda _tool_query: [tool_result],
                debug=True,
            )

        self.assertEqual(response["answer"], "最低比例为37.50%（3/8≈0.3750）。")
        self.assertEqual(model.call_count, 1)
        self.assertEqual(response["trace"][0]["agent_action_overridden"], "answer")
        self.assertEqual(
            [item["action"] for item in response["trace"]],
            ["collection", "deterministic_table_ratio"],
        )

    def test_cross_year_bucketing_overrides_agent_guess_with_collection(self) -> None:
        query = (
            "不满60人使用小型教室，60至100人使用中型教室，"
            "101至200人使用大型教室，超过200人使用特大型教室。"
            "2019、2020年和2021年分别应使用哪类教室？"
        )
        sizes = ((2019, 59), (2020, 65), (2021, 257))
        tool_results = [{
            "title": f"{year}年项目完整名单",
            "url": f"https://example.test/{year}",
            "content": f"确定性工具已扫描完整名单，共{size}项。",
            "collection_complete": "true",
            "collection_size": str(size),
        } for year, size in sizes]
        with patch(
            "zeta_engine.rag.call_model",
            return_value=(
                '{"action":"answer","answer":"三年都使用小型教室",'
                '"claims":[],"slots":[]}'
            ),
        ) as model:
            response = agentic_rag_answer(
                query,
                lambda _query, _top_k: [],
                collection_fn=lambda _tool_query: tool_results,
                debug=True,
            )

        self.assertEqual(
            response["answer"],
            "2019年使用小型教室；2020年使用中型教室；"
            "2021年使用特大型教室。",
        )
        self.assertEqual(model.call_count, 1)
        self.assertEqual(
            [item["action"] for item in response["trace"]],
            ["collection", "deterministic_collection_bucket"],
        )

    def test_range_classification_rules_accept_normalized_commas(self) -> None:
        query = (
            "不满60人使用小型教室,60至100人使用中型教室,"
            "101至200人使用大型教室,超过200人使用特大型教室"
        )

        self.assertEqual(
            [rule[-1] for rule in _range_classification_rules(query)],
            ["小型教室", "中型教室", "大型教室", "特大型教室"],
        )

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

    def test_verifier_audits_only_evidence_cited_by_agent_claims(self) -> None:
        top = {
            "title": "双方签订合作协议",
            "url": "https://example.test/first",
            "content": "协议持续五年，重点加强学术合作、学生交换、教师互访。",
        }
        unrelated = {
            "title": "双方续签第二期协议",
            "url": "https://example.test/second",
            "content": "第二期协议持续四年，重点推进教学与科研合作。",
        }
        top_id = Evidence.from_result(top).evidence_id
        decision = json.dumps({
            "action": "answer",
            "answer": "协议持续五年，重点加强学术合作、学生交换、教师互访。",
            "claims": [{
                "statement": "协议持续五年并重点加强学术合作、学生交换、教师互访",
                "evidence_ids": [top_id],
            }],
            "slots": [],
        }, ensure_ascii=False)
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[decision, verification(evidence_ids=[top_id])],
        ) as model:
            response = agentic_rag_answer(
                "合作协议计划持续多少年，重点加强哪些合作？",
                lambda _query, _top_k: [top, unrelated],
                debug=True,
            )

        verifier_prompt = model.call_args_list[1].args[0]
        self.assertIn("协议持续五年", verifier_prompt)
        self.assertNotIn("第二期协议持续四年", verifier_prompt)
        self.assertEqual(response["answer"], "协议持续五年，重点加强学术合作、学生交换、教师互访。")
        self.assertEqual(response["trace"][0]["verification_evidence_ids"], [top_id])

    def test_rejects_empty_prompt_and_missing_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt"):
            call_model("  ")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, LLM_API_KEY_ENV):
                call_model("问题")

    def test_model_endpoint_can_be_overridden_from_environment(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="模型答案")
        )])
        environment = {
            LLM_API_KEY_ENV: "test-key",
            LLM_BASE_URL_ENV: "https://llm.example.test/v1",
            LLM_MODEL_ENV: "test-model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("zeta_engine.rag.OpenAI") as openai,
        ):
            openai.return_value.chat.completions.create.return_value = response
            call_model("问题")

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://llm.example.test/v1",
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
        self.assertEqual(
            openai.return_value.chat.completions.create.call_args.kwargs["model"],
            "test-model",
        )
        self.assertNotIn(
            "extra_body",
            openai.return_value.chat.completions.create.call_args.kwargs,
        )

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
        self.assertIn("相关性优先级：1（数字越小越相关）", context)
        self.assertIn("标题：第一篇 vcr文档", context)
        self.assertIn("发布日期：2026-08-29", context)
        self.assertIn("内容：第一条 10月14日 证据", context)
        self.assertIn("重复 url", context)
        self.assertIn("内容：其他内容", context)
        self.assertIn("[文档3]", context)
        self.assertIn("内容：第二条证据", context)
        self.assertNotIn("重复正文", context)
        self.assertEqual(merge_results(results, max_chars=10), "")

    def test_prefers_minimal_html_for_model_context(self) -> None:
        context = merge_results([{
            "url": "https://example.test/rules",
            "content": "申诉处理 十日内提交",
            "heading_path": "学生管理 > 申诉处理",
            "structured_content": (
                "<main><h2>申诉处理</h2><p>十日内提交</p></main>"
            ),
        }])

        self.assertIn(
            "<main><h2>申诉处理</h2><p>十日内提交</p></main>",
            context,
        )
        self.assertIn("章节：学生管理 > 申诉处理", context)

    def test_model_context_keeps_raw_candidate_when_section_is_lossy(self) -> None:
        context = merge_results([{
            "url": "https://example.test/profile",
            "content": "社会兼职：期刊审稿。",
            "structured_content": "<section><h2>社会兼职</h2></section>",
            "candidate_content": (
                "教授课程：机器学习基础：2022-2024春。"
                "社会兼职：期刊审稿。"
            ),
        }])

        self.assertIn("机器学习基础:2022-2024春", context)

    def test_builds_grounded_prompt(self) -> None:
        prompt = build_prompt("申请条件是什么？", "[文档1]\n内容：申请条件")

        self.assertIn("问题：申请条件是什么？", prompt)
        self.assertIn("[文档1]\n内容：申请条件", prompt)
        self.assertIn("优先给出有用答案", prompt)
        self.assertIn("只输出最终答案", prompt)
        self.assertIn("不输出思考过程", prompt)
        self.assertIn("不要求答案必须在原文中逐字出现", prompt)

        cited_prompt = build_prompt(
            "申请条件是什么？",
            "[文档1]\n内容：申请条件",
            include_citations=True,
        )
        self.assertIn("用证据ID标注依据", cited_prompt)

        agent_prompt = build_agent_prompt(
            "三个年度的成员数量分别是多少？",
            "[文档1]\n内容：第一年成员：甲、乙",
            ["三个年度的成员数量分别是多少？"],
            remaining_cycles=2,
        )
        verifier_prompt = build_verifier_prompt(
            "三个年度的成员数量分别是多少？",
            "[文档1]\n内容：第一年成员：甲、乙",
            "第一年2人",
        )
        self.assertIn("优先 answer", agent_prompt)
        self.assertIn("只有核心信息完全缺失时才 search", agent_prompt)
        self.assertIn("claims 和 slots 可以为空", agent_prompt)
        self.assertIn("不要因为材料不完整", verifier_prompt)
        self.assertIn("与检索材料明显矛盾", verifier_prompt)
        for leaked_rule in ("不少于N", "比例≥N/D", "目标子集合", "流程阶段"):
            self.assertNotIn(leaked_rule, prompt)
            self.assertNotIn(leaked_rule, agent_prompt)
            self.assertNotIn(leaked_rule, verifier_prompt)

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

    def test_agentic_rag_runs_collection_and_ratio_inside_one_loop(self) -> None:
        initial = {
            "title": "夏令营公告",
            "url": "https://example.test/camp",
            "content": "拟推荐优秀营员不少于34人。",
        }
        tool_result = {
            "title": "夏令营公告 · 学术硕士营员名单",
            "url": "https://example.test/camp",
            "content": "确定性工具完整扫描学术硕士营员名单，共127人；拟推荐不少于34人。",
            "passage_id": "collection-table:1:1",
        }
        tool_evidence_id = Evidence.from_result(tool_result).evidence_id
        calculation = json.dumps({
            "action": "calculate",
            "calculations": [{
                "name": "最低录取比",
                "operator": "ratio",
                "items": [
                    {"value": 34, "evidence_ids": [tool_evidence_id]},
                    {"value": 127, "evidence_ids": [tool_evidence_id]},
                ],
            }],
            "slots": [],
        }, ensure_ascii=False)
        collection_calls = []

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "collection",
                    "query": "完整扫描学术硕士营员名单并计算最低录取比",
                    "slots": [],
                }, ensure_ascii=False),
                calculation,
                '{"action":"answer","answer":"最低录取比至少约为26.77%"}',
                verification(calculation_ids=["最低录取比"]),
            ],
        ):
            response = agentic_rag_answer(
                "夏令营学硕最低录取比是多少？",
                lambda _query, _top_k: [initial],
                collection_fn=lambda tool_query: (
                    collection_calls.append(tool_query) or [tool_result]
                ),
                debug=True,
            )

        self.assertEqual(response["answer"], "最低录取比至少约为26.77%")
        self.assertEqual(len(collection_calls), 1)
        self.assertEqual(
            [item["action"] for item in response["trace"]],
            ["collection", "calculate", "verified_answer"],
        )

    def test_shared_answer_across_years_skips_requirement_planning(self) -> None:
        results = [{
            "url": "https://example.test/course",
            "content": "2022、2023、2024年春季均开设机器学习基础。",
        }]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"机器学习基础"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "2022、2023和2024年春季都开设哪门课程？",
                lambda _query, _top_k: results,
            )

        self.assertEqual(response["answer"], "机器学习基础")
        self.assertEqual(model.call_count, 2)

    def test_compound_derivation_stays_in_one_closed_loop(self) -> None:
        query = (
            "不满60人使用小型教室，60至100人使用中型教室，"
            "101至200人使用大型教室，超过200人使用特大型教室。"
            "2019、2020和2021年夏令营分别应使用哪类教室？"
        )
        results = [
            {"url": "https://example.test/2019", "content": "2019年59人"},
            {"url": "https://example.test/2020", "content": "2020年65人"},
            {"url": "https://example.test/2021", "content": "2021年257人"},
        ]
        evidence_ids = [Evidence.from_result(item).evidence_id for item in results]
        ledger = json.dumps({
            "valid": True,
            "requirements": [
                {
                    "description": f"回答{year}年教室类型",
                    "satisfied": True,
                    "evidence_ids": [evidence_id],
                }
                for year, evidence_id in zip(
                    (2019, 2020, 2021), evidence_ids, strict=True
                )
            ],
            "claims": [
                {
                    "statement": statement,
                    "status": "supported",
                    "evidence_ids": [evidence_id],
                    "calculation_ids": [],
                }
                for statement, evidence_id in zip(
                    (
                        "2019年使用小型教室",
                        "2020年使用中型教室",
                        "2021年使用特大型教室",
                    ),
                    evidence_ids,
                    strict=True,
                )
            ],
            "conflicts": [],
            "issues": [],
            "queries": [],
        }, ensure_ascii=False)

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "answer",
                    "answer": (
                        "2019年使用小型教室，2020年使用中型教室，"
                        "2021年使用特大型教室。"
                    ),
                    "claims": [],
                }, ensure_ascii=False),
                ledger,
            ],
        ) as model:
            response = agentic_rag_answer(
                query,
                lambda _query, _top_k: results,
                debug=True,
            )

        self.assertTrue(response["complete"])
        self.assertEqual(
            [item["status"] for item in response["requirements"]],
            ["answered", "answered", "answered"],
        )
        self.assertEqual(model.call_count, 2)
        for call in model.call_args_list:
            self.assertIn("超过200人使用特大型教室", call.args[0])
        self.assertNotIn("requirement_plan", response["trace"][0])

    def test_multi_slot_verdict_cannot_return_only_an_intermediate_fact(self) -> None:
        results = [
            {
                "url": "https://example.test/report",
                "content": "入选教师是林老师",
            },
            {
                "url": "https://example.test/profile",
                "content": "林老师的本科生课程是《知识表示》",
            },
        ]
        evidence_ids = [Evidence.from_result(item).evidence_id for item in results]
        ledger = json.dumps({
            "valid": True,
            "requirements": [
                {
                    "description": "确定入选教师",
                    "satisfied": True,
                    "evidence_ids": [evidence_ids[0]],
                },
                {
                    "description": "列出该教师的本科生课程",
                    "satisfied": True,
                    "evidence_ids": [evidence_ids[1]],
                },
            ],
            "claims": [
                {
                    "statement": "入选教师是林老师",
                    "status": "supported",
                    "evidence_ids": [evidence_ids[0]],
                    "calculation_ids": [],
                },
                {
                    "statement": "林老师的本科生课程是《知识表示》",
                    "status": "supported",
                    "evidence_ids": [evidence_ids[1]],
                    "calculation_ids": [],
                },
            ],
            "conflicts": [],
            "issues": [],
            "queries": [],
        }, ensure_ascii=False)

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"林老师"}',
                ledger,
            ],
        ):
            response = agentic_rag_answer(
                "入选教师的本科生课程是什么？",
                lambda _query, _top_k: results,
            )

        self.assertIn("林老师", response["answer"])
        self.assertIn("《知识表示》", response["answer"])

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
        self.assertIn("优先 answer", model.call_args_list[0].args[0])
        self.assertIn("核心信息完全缺失", model.call_args_list[0].args[0])
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
        self.assertIn("继续检索", model.call_args_list[0].args[0])

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
        self.assertEqual(len(response["results"]), 21)
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

    def test_agentic_rag_counts_independent_evidence_records(self) -> None:
        results = [
            {
                "title": "第一次参访",
                "url": "https://example.test/visit-one",
                "content": "学院开展企业参访。",
            },
            {
                "title": "第二次参访",
                "url": "https://example.test/visit-two",
                "content": "学院再次开展企业参访。",
            },
        ]
        evidence_ids = [
            Evidence.from_result(result).evidence_id for result in results
        ]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "calculate",
                    "calculations": [{
                        "name": "参访次数",
                        "operator": "count",
                        "values": evidence_ids,
                        "evidence_ids": evidence_ids,
                    }],
                }, ensure_ascii=False),
                json.dumps({
                    "action": "answer",
                    "answer": "共2次",
                    "claims": [{
                        "statement": "共参访2次",
                        "evidence_ids": [],
                        "calculation_ids": ["计算1"],
                    }],
                }, ensure_ascii=False),
                verification(
                    calculation_ids=["参访次数"],
                    evidence_ids=evidence_ids,
                ),
            ],
        ):
            response = agentic_rag_answer(
                "一共参访多少次？",
                lambda _query, _top_k: results,
                max_cycles=2,
                debug=True,
            )

        calculation = response["trace"][0]["calculations"][0]
        self.assertEqual(calculation["result"], 2)
        self.assertEqual(response["answer"], "共2次")

    def test_calculation_keeps_stable_evidence_after_later_search(self) -> None:
        first = {
            "url": "https://example.test/list",
            "content": "甲 乙",
        }
        first_id = Evidence.from_result(first).evidence_id

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            if query == "补充":
                return [{
                    "url": "https://example.test/new",
                    "content": "无关的新证据",
                }]
            return [first]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                json.dumps({
                    "action": "calculate",
                    "calculations": [{
                        "name": "人数",
                        "operator": "count",
                        "items": [
                            {"value": "甲", "evidence_id": first_id},
                            {"value": "乙", "evidence_id": first_id},
                        ],
                    }],
                }, ensure_ascii=False),
                '{"action":"search","queries":["补充"]}',
                '{"action":"answer","answer":"2人"}',
                verification(calculation_ids=["人数"], evidence_ids=[first_id]),
            ],
        ):
            response = agentic_rag_answer(
                "有多少人？",
                search_fn,
                max_cycles=4,
                debug=True,
            )

        calculation = response["trace"][0]["calculations"][0]
        self.assertEqual(calculation["evidence_ids"], [first_id])
        self.assertEqual(response["sources"][0]["evidence_id"], first_id)
        self.assertEqual(len(response["results"]), 2)

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
        self.assertEqual(revision["policy"], "latest_per_independent_fact")
        self.assertTrue(revision["valid"])

    def test_prefilters_older_dated_records_for_each_query_entity(self) -> None:
        results = [
            {
                "title": "星河公司旧记录",
                "url": "https://example.test/star-old",
                "published_at": "2024-01-01",
                "content": "星河公司是第2项",
            },
            {
                "title": "星河公司新记录",
                "url": "https://example.test/star-new",
                "published_at": "2025-01-01",
                "content": "星河公司是第4项",
            },
            {
                "title": "远山公司记录",
                "url": "https://example.test/mountain",
                "published_at": "2023-01-01",
                "content": "远山公司是第1项",
            },
        ]
        old_id = Evidence.from_result(results[0]).evidence_id

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"远山公司、星河公司"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "请按编号排列星河公司和远山公司",
                lambda _query, _top_k: results,
                debug=True,
            )

        initial_prompt = model.call_args_list[0].args[0]
        self.assertNotIn("星河公司是第2项", initial_prompt)
        self.assertIn("星河公司是第4项", initial_prompt)
        self.assertEqual(response["trace"][0]["superseded_evidence_ids"], [old_id])
        self.assertEqual(len(response["results"]), 3)

    def test_event_count_keeps_all_dated_records(self) -> None:
        results = [
            {
                "title": "快手公司参访纪实",
                "url": "https://example.test/kuaishou-old",
                "published_at": "2023-04-13",
                "content": "学院组织学生参访快手公司。",
            },
            {
                "title": "快手公司企业参访",
                "url": "https://example.test/kuaishou-new",
                "published_at": "2025-12-04",
                "content": "学院再次组织学生参访快手公司。",
            },
            {
                "title": "腾讯公司参访纪实",
                "url": "https://example.test/tencent-old",
                "published_at": "2025-12-19",
                "content": "学院组织学生参访腾讯公司。",
            },
            {
                "title": "腾讯公司企业参访",
                "url": "https://example.test/tencent-new",
                "published_at": "2026-05-16",
                "content": "学院再次组织学生参访腾讯公司。",
            },
        ]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"一共4次"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "参访快手公司和腾讯公司一共多少次？",
                lambda _query, _top_k: results,
                debug=True,
            )

        initial_prompt = model.call_args_list[0].args[0]
        self.assertIn("2023-04-13", initial_prompt)
        self.assertIn("2025-12-04", initial_prompt)
        self.assertIn("2025-12-19", initial_prompt)
        self.assertIn("2026-05-16", initial_prompt)
        self.assertEqual(response["answer"], "一共4次")
        self.assertEqual(response["trace"][0]["superseded_evidence_ids"], [])

    def test_sorts_title_facts_without_an_llm_call(self) -> None:
        results = [
            {
                "title": "远山公司企业参访第12站",
                "url": "https://example.test/mountain",
                "published_at": "2025-01-01",
                "content": "远山公司参访纪实",
            },
            {
                "title": "星河公司企业参访第3站",
                "url": "https://example.test/star",
                "published_at": "2025-01-02",
                "content": "星河公司参访纪实",
            },
        ]

        with patch("zeta_engine.rag.call_model") as model:
            response = agentic_rag_answer(
                "星河公司和远山公司的站次顺序是什么？",
                lambda _query, _top_k: results,
                debug=True,
            )

        self.assertEqual(
            response["answer"],
            "星河公司是第3站、远山公司是第12站。",
        )
        self.assertEqual(response["trace"][0]["action"], "deterministic_sort")
        self.assertEqual(len(response["claims"]), 2)
        model.assert_not_called()

    def test_single_explicit_year_does_not_drop_other_evidence(self) -> None:
        results = [
            {
                "title": "林老师入选2024年度人才计划",
                "url": "https://example.test/2024",
                "published_at": "2024-09-01",
                "content": "2024年度入选教师是林老师",
            },
            {
                "title": "王老师入选2025年度人才计划",
                "url": "https://example.test/2025",
                "published_at": "2025-09-01",
                "content": "2025年度入选教师是王老师",
            },
        ]
        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"林老师"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "2024年度人才计划入选教师是谁？",
                lambda _query, _top_k: results,
                debug=True,
            )

        initial_prompt = model.call_args_list[0].args[0]
        self.assertIn("2024年度入选教师是林老师", initial_prompt)
        self.assertIn("2025年度入选教师是王老师", initial_prompt)
        self.assertNotIn("scope_excluded_evidence_ids", response["trace"][0])

    def test_temporal_sorting_stops_after_grounded_format_revision(self) -> None:
        results = [
            {
                "url": "https://example.test/old",
                "published_at": "2024-01-01",
                "content": "乙对象曾经排第2",
            },
            {
                "url": "https://example.test/new",
                "published_at": "2025-01-01",
                "content": "甲对象当前排第1，乙对象当前排第3",
            },
        ]
        conflict = json.dumps({
            "valid": False,
            "requirements": [{
                "description": "按当前编号排序对象",
                "satisfied": True,
                "source_ids": [1, 2],
            }],
            "claims": [{
                "statement": "乙对象排第2",
                "status": "supported",
                "source_ids": [1],
                "calculation_ids": [],
            }],
            "conflicts": [{
                "description": "乙对象有新旧编号",
                "resolved": False,
                "resolution": "",
                "source_ids": [1, 2],
            }],
            "issues": [{
                "type": "conflict",
                "description": "需要采用最新记录",
            }],
            "queries": [],
        }, ensure_ascii=False)
        grounded_but_invalid = json.dumps({
            "valid": False,
            "requirements": [{
                "description": "按当前编号排序对象",
                "satisfied": True,
                "source_ids": [1],
            }],
            "claims": [
                {
                    "statement": "甲对象排第1",
                    "status": "supported",
                    "source_ids": [1],
                    "calculation_ids": [],
                },
                {
                    "statement": "乙对象排第3",
                    "status": "supported",
                    "source_ids": [1],
                    "calculation_ids": [],
                },
            ],
            "conflicts": [],
            "issues": [{
                "type": "format",
                "description": "候选答案只列出了编号",
            }],
            "queries": [],
        }, ensure_ascii=False)

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"乙对象第2"}',
                conflict,
                "第1、第3",
                grounded_but_invalid,
                grounded_but_invalid,
            ],
        ) as model:
            response = agentic_rag_answer(
                "请按当前编号顺序排列对象",
                lambda _query, _top_k: results,
                debug=True,
            )

        self.assertEqual(response["answer"], "甲对象排第1、乙对象排第3。")
        self.assertEqual(response["model_call_count"], 5)
        self.assertEqual(model.call_count, 5)
        revision = response["trace"][0]["temporal_revision"]
        self.assertTrue(revision["format_revision"]["accepted_as_grounded"])

    def test_compound_query_preserves_answered_requirements(self) -> None:
        results = [
            {
                "url": "https://example.test/2019",
                "content": "2019年人数为59人",
            },
            {
                "url": "https://example.test/2020",
                "content": "2020年人数为65人",
            },
        ]
        evidence_ids = [Evidence.from_result(item).evidence_id for item in results]
        partial_ledger = json.dumps({
            "valid": False,
            "requirements": [
                {
                    "description": "回答2019年人数",
                    "satisfied": True,
                    "evidence_ids": [evidence_ids[0]],
                },
                {
                    "description": "回答2020年人数",
                    "satisfied": True,
                    "evidence_ids": [evidence_ids[1]],
                },
                {
                    "description": "回答2021年人数",
                    "satisfied": False,
                    "evidence_ids": [],
                },
            ],
            "claims": [
                {
                    "statement": "2019年59人",
                    "status": "supported",
                    "evidence_ids": [evidence_ids[0]],
                    "calculation_ids": [],
                },
                {
                    "statement": "2020年65人",
                    "status": "supported",
                    "evidence_ids": [evidence_ids[1]],
                    "calculation_ids": [],
                },
                {
                    "statement": "2021年人数未知",
                    "status": "unsupported",
                    "evidence_ids": [],
                    "calculation_ids": [],
                },
            ],
            "conflicts": [],
            "issues": [{
                "type": "incomplete",
                "description": "缺少2021年人数",
            }],
            "queries": ["2021年人数"],
        }, ensure_ascii=False)

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"answer","answer":"2019年59人；2020年65人"}',
                partial_ledger,
                '{"action":"answer","answer":"2019年59人；2020年65人"}',
                partial_ledger,
            ],
        ) as model:
            response = agentic_rag_answer(
                "2019、2020和2021年人数分别是多少？",
                lambda _query, _top_k: results,
                max_cycles=1,
                debug=True,
            )

        self.assertEqual(response["status"], "partial")
        self.assertFalse(response["complete"])
        self.assertIn("2019年59人", response["answer"])
        self.assertIn("2020年65人", response["answer"])
        self.assertIn("2021年人数", response["answer"])
        self.assertEqual(
            [item["status"] for item in response["requirements"]],
            ["answered", "answered", "missing"],
        )
        self.assertEqual(response["model_call_count"], 4)
        self.assertNotIn("requirement_plan", response["trace"][0])
        self.assertIn(
            "2019、2020和2021年人数分别是多少",
            model.call_args_list[0].args[0],
        )

    def test_requirement_dependency_receives_upstream_answer(self) -> None:
        searches: list[str] = []

        def search_fn(query: str, _top_k: int) -> list[dict[str, str]]:
            searches.append(query)
            if "林老师" in query:
                return [{
                    "url": "https://example.test/teacher",
                    "content": "林老师教授《知识表示》",
                }]
            return [{
                "title": "2024年度入选报道",
                "url": "https://example.test/report",
                "content": "入选教师为林老师",
            }]

        with patch(
            "zeta_engine.rag.call_model",
            side_effect=[
                '{"action":"search","queries":["林老师 本科生课程"]}',
                '{"action":"answer","answer":"《知识表示》"}',
                verification(),
            ],
        ) as model:
            response = agentic_rag_answer(
                "2024年度入选教师是谁，在其个人主页中有哪些本科生课程？",
                search_fn,
                max_cycles=2,
            )

        self.assertTrue(response["complete"])
        self.assertEqual(response["status"], "answered")
        self.assertIn("林老师", searches[1])
        self.assertIn("《知识表示》", response["answer"])
        self.assertEqual(model.call_count, 3)
        for call in model.call_args_list:
            self.assertIn("2024年度入选教师是谁", call.args[0])

    def test_compound_query_still_answers_when_all_evidence_is_missing(self) -> None:
        with patch(
            "zeta_engine.rag.call_model",
            return_value="甲和乙的具体信息无法确认。",
        ) as model:
            response = agentic_rag_answer(
                "甲和乙分别是什么？",
                lambda _query, _top_k: [],
                max_cycles=1,
            )

        self.assertEqual(response["answer"], "甲和乙的具体信息无法确认。")
        self.assertEqual(response["status"], "answered")
        self.assertTrue(response["complete"])
        model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
