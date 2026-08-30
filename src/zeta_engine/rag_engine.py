"""Bounded Agent→Tools→Verifier orchestration for RAG."""

import json
import re
from collections.abc import Callable
from typing import cast

from zeta_engine.rag_config import (
    AGENT_MAX_CYCLES,
    AGENT_MAX_LLM_CALLS,
    RAG_NO_RESULTS_ANSWER,
)
from zeta_engine.rag_evidence import (
    AgentDecision,
    AnswerSlot,
    Evidence,
    EvidenceManager,
    _clean,
    _estimate_tokens,
    _normalize_text,
    _result_text,
)
from zeta_engine.rag_prompts import (
    build_agent_prompt,
    build_prompt,
    build_verifier_prompt,
)
from zeta_engine.rag_protocol import (
    _parse_agent_action,
    _parse_verification,
    _valid_calculation_reference,
)
from zeta_engine.rag_types import RagResponse
from zeta_engine.rag_calculations import (
    _deterministic_calculation_answer,
    _format_calculations,
    _run_calculations,
)
from zeta_engine.rag_policies import (
    _deterministic_collection_bucket_answer,
    _deterministic_complete_table_ratio_answer,
    _deterministic_numbered_event_answer,
    _deterministic_sorted_answer,
    _needs_sorted_object_format,
    _prefer_latest_record,
    _requires_complete_collection,
    _superseded_evidence_ids,
)
from zeta_engine.tokenizer import text_normalize


def _run_closed_loop(
    query: str,
    search_fn: Callable[[str, int], list[dict[str, str]]],
    top_k: int = 8,
    *,
    max_cycles: int = AGENT_MAX_CYCLES,
    max_llm_calls: int = AGENT_MAX_LLM_CALLS,
    include_citations: bool = False,
    debug: bool = False,
    model_fn: Callable[..., str],
    collection_fn: Callable[[str], list[dict[str, str]]] | None = None,
) -> RagResponse:
    """Run one bounded Agent→Tools→Verifier loop over the full question."""

    query = text_normalize(query)
    if not query:
        raise ValueError("query 不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if max_cycles <= 0:
        raise ValueError("max_cycles 必须大于 0")
    if max_llm_calls <= 0:
        raise ValueError("max_llm_calls 必须大于 0")

    pending_queries = [query]
    searched_queries: list[str] = []
    collection_queries: list[str] = []
    evidence_manager = EvidenceManager()
    calculations: list[dict[str, object]] = []
    verification_feedback = ""
    trace: list[dict[str, object]] | None = [] if debug else None
    model_call_count = 0
    answer_slots: dict[str, AnswerSlot] = {
        "s1": AnswerSlot("s1", query, "missing")
    }

    def ask(prompt: str, *, json_output: bool = False) -> str:
        nonlocal model_call_count
        if model_call_count >= max_llm_calls:
            raise RuntimeError("RAG 已达到模型调用上限")
        model_call_count += 1
        return model_fn(prompt, json_output=json_output)

    def update_answer_slots(slots: list[AnswerSlot]) -> None:
        """Replace the provisional slot plan with the latest stable plan."""

        nonlocal answer_slots
        if not slots:
            return
        previous = answer_slots
        answer_slots = {}
        for slot in slots:
            old = previous.get(slot.slot_id)
            evidence_ids = tuple(dict.fromkeys([
                *(old.evidence_ids if old is not None else ()),
                *slot.evidence_ids,
            ]))
            answer_slots[slot.slot_id] = AnswerSlot(
                slot_id=slot.slot_id,
                question=slot.question,
                status=slot.status,
                evidence_ids=evidence_ids,
            )

    def public_claims(
        ledger: dict[str, object] | None,
        fallback: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        raw_claims = ledger.get("claims", []) if isinstance(ledger, dict) else []
        claims = [
            {
                "statement": _clean(claim.get("statement")),
                "evidence_ids": claim.get("evidence_ids", []),
                "calculation_ids": claim.get("calculation_ids", []),
            }
            for claim in raw_claims
            if isinstance(claim, dict) and claim.get("status") == "supported"
        ]
        return claims or list(fallback or [])

    def finish(
        answer: str,
        *,
        ledger: dict[str, object] | None = None,
        claims: list[dict[str, object]] | None = None,
    ) -> RagResponse:
        final_claims = public_claims(ledger, claims)
        raw_requirements = (
            ledger.get("requirements", [])
            if isinstance(ledger, dict)
            else []
        )
        raw_ledger_claims = (
            ledger.get("claims", []) if isinstance(ledger, dict) else []
        )
        conflicts = ledger.get("conflicts", []) if isinstance(ledger, dict) else []
        unresolved_conflict = any(
            isinstance(conflict, dict) and conflict.get("resolved") is False
            for conflict in conflicts
        )
        ledger_incomplete = isinstance(ledger, dict) and (
            any(
                isinstance(item, dict) and item.get("satisfied") is not True
                for item in raw_requirements
            )
            or any(
                isinstance(item, dict) and item.get("status") != "supported"
                for item in raw_ledger_claims
            )
        )
        status = (
            "conflicting"
            if unresolved_conflict
            else "partial"
            if ledger_incomplete and final_claims
            else "missing"
            if answer in {"材料不足", RAG_NO_RESULTS_ANSWER} or ledger_incomplete
            else "answered"
        )
        cited_ids = list(dict.fromkeys(
            evidence_id
            for claim in final_claims
            for evidence_id in claim.get("evidence_ids", [])
            if isinstance(evidence_id, str)
        ))
        sources = []
        for evidence_id in cited_ids:
            evidence = evidence_manager.get(evidence_id)
            if evidence is None:
                continue
            source = dict(evidence.result)
            source["evidence_id"] = evidence_id
            sources.append(source)
        requirements = [
            {
                "id": f"r{index}",
                "question": _clean(item.get("description")) or query,
                "depends_on": [],
                "status": (
                    "answered" if item.get("satisfied") is True else "missing"
                ),
                "evidence_ids": list(item.get("evidence_ids", [])),
            }
            for index, item in enumerate(raw_requirements, start=1)
            if isinstance(item, dict)
        ]
        if not requirements:
            requirements = [slot.public() for slot in answer_slots.values()]
            if len(requirements) == 1 and status == "answered":
                requirements[0]["status"] = "answered"
                requirements[0]["evidence_ids"] = cited_ids
        public_answer_slots = [slot.public() for slot in answer_slots.values()]
        if len(public_answer_slots) == 1 and status == "answered":
            public_answer_slots[0]["status"] = "answered"
            public_answer_slots[0]["evidence_ids"] = cited_ids
        response: dict[str, object] = {
            "answer": answer,
            "status": status,
            "complete": status == "answered",
            "answer_slots": public_answer_slots,
            "requirements": requirements,
            "claims": final_claims,
            "sources": sources,
            # Backward-compatible complete retrieval pool.
            "results": evidence_manager.results(),
        }
        if trace is not None:
            response["model_call_count"] = model_call_count
            response["trace"] = trace
        return cast(RagResponse, response)

    def answer_from_supported_claims(
        ledger: dict[str, object],
    ) -> str:
        """Keep verified slots when the whole answer cannot be completed."""

        if any(
            isinstance(item, dict) and item.get("resolved") is False
            for item in ledger.get("conflicts", [])
        ):
            return ""
        statements = list(dict.fromkeys(
            _clean(item.get("statement"))
            for item in ledger.get("claims", [])
            if isinstance(item, dict)
            and item.get("status") == "supported"
            and _clean(item.get("statement"))
        ))
        if not statements:
            return ""
        missing = list(dict.fromkeys(
            _clean(item.get("description"))
            for item in ledger.get("requirements", [])
            if isinstance(item, dict)
            and item.get("satisfied") is False
            and _clean(item.get("description"))
        ))
        parts = statements
        if missing:
            parts.append(f"未能回答：{'、'.join(missing)}（材料不足）")
        return "；".join(part.rstrip("。；; ") for part in parts) + "。"

    def complete_answer(
        candidate: str,
        ledger: dict[str, object],
    ) -> str:
        """Materialize every verified final slot in a multi-slot answer."""

        requirements = ledger.get("requirements", [])
        if not isinstance(requirements, list) or len(requirements) <= 1:
            return candidate
        return answer_from_supported_claims(ledger) or candidate

    def ledger_is_grounded(ledger: dict[str, object]) -> bool:
        """Accept a semantically grounded answer despite presentation-only issues."""

        requirements = ledger.get("requirements", [])
        claims = ledger.get("claims", [])
        conflicts = ledger.get("conflicts", [])
        return (
            isinstance(requirements, list)
            and bool(requirements)
            and all(
                isinstance(item, dict) and item.get("satisfied") is True
                for item in requirements
            )
            and isinstance(claims, list)
            and bool(claims)
            and all(
                isinstance(item, dict) and item.get("status") == "supported"
                for item in claims
            )
            and isinstance(conflicts, list)
            and all(
                isinstance(item, dict) and item.get("resolved") is True
                for item in conflicts
            )
        )

    def audit(
        answer: str,
        context: str,
        visible_evidence: list[Evidence],
    ) -> tuple[bool, str, list[str], dict[str, object]]:
        verifier_response = ask(
            build_verifier_prompt(query, context, answer),
            json_output=True,
        )
        try:
            return _parse_verification(
                verifier_response,
                allowed_evidence_ids=[
                    evidence.evidence_id for evidence in visible_evidence
                ],
                calculation_names={
                    str(calculation["name"])
                    for calculation in calculations
                },
                calculation_count=len(calculations),
                evidence_dates={
                    evidence.evidence_id: evidence.published_at
                    for evidence in visible_evidence
                },
                prefer_latest=_prefer_latest_record(query),
            )
        except ValueError as error:
            return (
                False,
                f"证据审计输出无效：{error}",
                [],
                {"error": str(error), "raw": verifier_response},
            )

    def validate_claims(
        claims: list[dict[str, object]],
        visible_evidence: list[Evidence],
    ) -> str:
        # Plain-answer fallback remains supported; the semantic verifier then
        # supplies the claim ledger and stable citations.
        if not claims:
            return ""
        visible_ids = {evidence.evidence_id for evidence in visible_evidence}
        calculation_ids = {
            str(calculation["name"]) for calculation in calculations
        }
        for claim in claims:
            evidence_ids = claim.get("evidence_ids", [])
            referenced_calculations = claim.get("calculation_ids", [])
            if not evidence_ids and not referenced_calculations:
                return "每项 claim 必须引用证据或计算结果"
            if any(item not in visible_ids for item in evidence_ids):
                return "claim 引用了当前上下文之外的证据"
            if any(
                not _valid_calculation_reference(
                    item,
                    calculation_ids,
                    len(calculations),
                )
                for item in referenced_calculations
            ):
                return "claim 引用了不存在的计算结果"
        return ""

    def add_calculations(context: str, allowed_ids: set[str]) -> str:
        visible = [
            calculation
            for calculation in calculations
            if set(calculation.get("evidence_ids", [])) <= allowed_ids
        ]
        if not visible:
            return context
        return (
            f"{context}\n\n<计算结果>\n"
            f"{_format_calculations(visible)}\n</计算结果>"
        )

    def request_answer(
        context: str,
        feedback: str,
    ) -> tuple[str, list[dict[str, object]]]:
        response = ask(
            build_agent_prompt(
                query,
                context,
                searched_queries,
                remaining_cycles=0,
                verification_feedback=feedback,
                collection_available=collection_fn is not None,
                collection_queries=collection_queries,
            ),
            json_output=True,
        )
        try:
            decision = _parse_agent_action(response)
        except (ValueError, json.JSONDecodeError):
            return _clean(response), []
        update_answer_slots(decision.slots)
        if decision.action == "answer":
            return decision.answer, decision.claims
        return ask(build_prompt(
            query,
            context,
            include_citations=include_citations,
            verification_feedback=feedback,
        )), []

    def format_verified_answer(
        answer: str,
        ledger: dict[str, object],
    ) -> str:
        statements = [
            _clean(claim.get("statement"))
            for claim in ledger.get("claims", [])
            if isinstance(claim, dict) and claim.get("status") == "supported"
        ]
        if not statements:
            return answer
        if _needs_sorted_object_format(query):
            keyed_statements = []
            for statement in statements:
                keys = re.findall(r"第\s*(-?\d+(?:\.\d+)?)", statement)
                if len(keys) != 1:
                    keyed_statements = []
                    break
                keyed_statements.append((float(keys[0]), statement.rstrip("。；;，,")))
            if keyed_statements:
                keyed_statements.sort(key=lambda item: item[0])
                return "、".join(item[1] for item in keyed_statements) + "。"
        return ask(f"""请仅使用已验证事实，把候选答案改写成直接回答问题的一句话。
不得增加、删除或猜测事实，不要输出解释、依据或思考过程。
排序题必须输出问题要求排序的对象，不能只输出排序键。

问题：{query}
候选答案：{answer}
已验证事实：{json.dumps(statements, ensure_ascii=False)}

最终答案：""")

    for cycle in range(max_cycles):
        batches: list[tuple[str, list[dict[str, str]]]] = []
        cycle_queries = []
        for search_query in pending_queries:
            if search_query in searched_queries:
                continue
            searched_queries.append(search_query)
            cycle_queries.append(search_query)
            batches.append((search_query, search_fn(search_query, top_k)))

        interleaved = [
            (batch[rank], search_query)
            for rank in range(max((len(batch) for _query, batch in batches), default=0))
            for search_query, batch in batches
            if rank < len(batch)
        ]
        new_evidence = evidence_manager.add(interleaved)
        newest_ids = [evidence.evidence_id for evidence in new_evidence]
        context, visible_evidence = evidence_manager.build_context(
            newest_ids=newest_ids,
        )
        superseded_ids = _superseded_evidence_ids(query, visible_evidence)
        excluded_context_ids = superseded_ids
        if excluded_context_ids:
            retained_ids = {
                evidence.evidence_id for evidence in visible_evidence
            } - excluded_context_ids
            context, visible_evidence = evidence_manager.build_context(
                newest_ids=[
                    evidence_id
                    for evidence_id in newest_ids
                    if evidence_id in retained_ids
                ],
                only_ids=retained_ids,
            )
        visible_ids = {evidence.evidence_id for evidence in visible_evidence}
        context = add_calculations(context, visible_ids)
        cycle_trace = None
        if trace is not None:
            cycle_trace = {
                "cycle": cycle + 1,
                "search_queries": cycle_queries,
                "new_results": [
                    {
                        "evidence_id": Evidence.from_result(result).evidence_id,
                        "title": _normalize_text(result.get("title")),
                        "url": _clean(result.get("url")),
                        "published_at": _clean(result.get("published_at")),
                        "preview": _result_text(result)[:240],
                    }
                    for result, _search_query in interleaved
                ],
                "evidence_count": len(evidence_manager.all()),
                "context_evidence_ids": [
                    evidence.evidence_id for evidence in visible_evidence
                ],
                "superseded_evidence_ids": sorted(superseded_ids),
                "context_chars": len(context),
                "context_tokens_estimate": _estimate_tokens(context),
            }
            trace.append(cycle_trace)
        final_cycle = cycle == max_cycles - 1
        event_answer, event_claims = _deterministic_numbered_event_answer(
            query,
            visible_evidence,
        )
        if event_answer:
            if cycle_trace is not None:
                cycle_trace.update({
                    "action": "deterministic_event_count",
                    "answer": event_answer,
                })
            return finish(event_answer, claims=event_claims)
        bucket_answer, bucket_claims = _deterministic_collection_bucket_answer(
            query,
            visible_evidence,
        )
        if bucket_answer:
            if cycle_trace is not None:
                cycle_trace.update({
                    "action": "deterministic_collection_bucket",
                    "answer": bucket_answer,
                })
            return finish(bucket_answer, claims=bucket_claims)
        ratio_answer, ratio_claims = _deterministic_complete_table_ratio_answer(
            query,
            visible_evidence,
        )
        if ratio_answer:
            if cycle_trace is not None:
                cycle_trace.update({
                    "action": "deterministic_table_ratio",
                    "answer": ratio_answer,
                })
            return finish(ratio_answer, claims=ratio_claims)
        sorted_answer, sorted_claims = _deterministic_sorted_answer(
            query,
            visible_evidence,
        )
        if sorted_answer:
            if cycle_trace is not None:
                cycle_trace.update({
                    "action": "deterministic_sort",
                    "answer": sorted_answer,
                })
            return finish(sorted_answer, claims=sorted_claims)
        if not context and final_cycle:
            fallback_answer = ask(build_prompt(
                query,
                "（没有检索到材料，请直接给出最可能的答案并说明不确定性。）",
                include_citations=include_citations,
                verification_feedback=verification_feedback,
            ))
            if cycle_trace is not None:
                cycle_trace.update({
                    "action": "fallback_without_evidence",
                    "answer": fallback_answer,
                })
            return finish(fallback_answer)

        proposed_answer = ""
        proposed_claims: list[dict[str, object]] = []
        if final_cycle:
            proposed_answer, proposed_claims = request_answer(
                context,
                verification_feedback,
            )
        else:
            response = ask(
                build_agent_prompt(
                    query,
                    context or "（未检索到材料）",
                    searched_queries,
                    remaining_cycles=max_cycles - cycle - 1,
                    verification_feedback=verification_feedback,
                    collection_available=collection_fn is not None,
                    collection_queries=collection_queries,
                ),
                json_output=True,
            )
            if cycle_trace is not None:
                cycle_trace["model_response"] = response
            try:
                decision = _parse_agent_action(response)
            except ValueError as error:
                if not context:
                    proposed_answer = ask(build_prompt(
                        query,
                        "（没有检索到材料，请直接给出最可能的答案并说明不确定性。）",
                        include_citations=include_citations,
                        verification_feedback=verification_feedback,
                    ))
                    if cycle_trace is not None:
                        cycle_trace.update({
                            "action": "fallback_without_evidence",
                            "error": str(error),
                            "answer": proposed_answer,
                        })
                else:
                    proposed_answer, proposed_claims = request_answer(
                        context,
                        verification_feedback,
                    )
                    if cycle_trace is not None:
                        cycle_trace.update({
                            "action": "fallback_answer",
                            "error": str(error),
                            "answer": proposed_answer,
                        })
            else:
                update_answer_slots(decision.slots)
                if (
                    collection_fn is not None
                    and not collection_queries
                    and decision.action != "collection"
                    and _requires_complete_collection(query)
                ):
                    if cycle_trace is not None:
                        cycle_trace["agent_action_overridden"] = decision.action
                    decision = AgentDecision(
                        action="collection",
                        collection_query=query,
                        slots=decision.slots,
                    )
                if cycle_trace is not None:
                    cycle_trace["answer_slots"] = [
                        slot.public() for slot in answer_slots.values()
                    ]
                proposed_answer = decision.answer
                proposed_claims = decision.claims
                if decision.action == "collection":
                    if collection_fn is None:
                        verification_feedback = "当前没有可用的完整集合工具。"
                        if cycle_trace is not None:
                            cycle_trace["action"] = "collection_unavailable"
                        pending_queries = []
                        continue
                    collection_query = decision.collection_query
                    if collection_query in collection_queries:
                        verification_feedback = (
                            "该集合查询已经完整执行；请使用现有集合结果回答或计算，"
                            "不要重复调用。"
                        )
                        if cycle_trace is not None:
                            cycle_trace.update({
                                "action": "duplicate_collection",
                                "collection_query": collection_query,
                            })
                        pending_queries = []
                        continue
                    collection_queries.append(collection_query)
                    tool_results = collection_fn(collection_query)
                    added = evidence_manager.add([
                        (result, collection_query) for result in tool_results
                    ])
                    evidence_manager.pin([
                        evidence.evidence_id for evidence in added
                    ])
                    verification_feedback = (
                        "完整集合工具已返回结果；请使用其中的完整扫描、分组、计数"
                        "和边界信息继续计算或回答。"
                        if added
                        else "集合工具没有解析出完整集合；请改用现有证据或补充检索。"
                    )
                    if cycle_trace is not None:
                        cycle_trace.update({
                            "action": "collection",
                            "collection_query": collection_query,
                            "collection_results": [
                                {
                                    "evidence_id": evidence.evidence_id,
                                    "title": evidence.title,
                                    "url": evidence.url,
                                    "preview": evidence.text[:240],
                                }
                                for evidence in added
                            ],
                        })
                    pending_queries = []
                    continue
                if decision.action == "calculate":
                    try:
                        completed = _run_calculations(
                            decision.calculations,
                            visible_evidence,
                            calculations,
                        )
                    except ValueError as error:
                        verification_feedback = f"计算请求无效：{error}"
                        if cycle_trace is not None:
                            cycle_trace.update({
                                "action": "invalid_calculation",
                                "error": str(error),
                            })
                    else:
                        calculations.extend(completed)
                        for calculation in completed:
                            evidence_manager.pin(
                                list(calculation.get("evidence_ids", []))
                            )
                        verification_feedback = (
                            "通用计算已完成，请使用计算结果继续检查并回答问题。"
                        )
                        if cycle_trace is not None:
                            cycle_trace.update({
                                "action": "calculate",
                                "calculations": completed,
                            })
                        deterministic_answer, deterministic_claims = (
                            _deterministic_calculation_answer(
                                query,
                                calculations,
                            )
                        )
                        if deterministic_answer:
                            if cycle_trace is not None:
                                cycle_trace.update({
                                    "action": "deterministic_calculation",
                                    "answer": deterministic_answer,
                                })
                            return finish(
                                deterministic_answer,
                                claims=deterministic_claims,
                            )
                    pending_queries = []
                    continue
                if decision.action == "search":
                    pending_queries = [
                        item
                        for item in decision.queries
                        if item not in searched_queries
                    ]
                    if cycle_trace is not None:
                        cycle_trace.update({
                            "action": "search",
                            "next_queries": pending_queries,
                            "query_slots": {
                                item: decision.query_slots.get(item, "")
                                for item in pending_queries
                            },
                        })
                    if pending_queries:
                        continue
                    no_new_query = (
                        "搜索计划没有产生新的查询；请使用现有证据和先前审计反馈"
                        "完成推理，不要重复搜索。"
                    )
                    verification_feedback = "\n".join(filter(None, (
                        verification_feedback,
                        no_new_query,
                    )))
                    proposed_answer, proposed_claims = request_answer(
                        context,
                        verification_feedback,
                    )

        claim_error = validate_claims(proposed_claims, visible_evidence)
        if claim_error:
            valid, feedback, verifier_queries, ledger = (
                False,
                f"citation: {claim_error}",
                [],
                {"error": claim_error},
            )
        else:
            for claim in proposed_claims:
                evidence_manager.pin(list(claim.get("evidence_ids", [])))
            verification_context = context
            verification_evidence = visible_evidence
            cited_evidence_ids = {
                evidence_id
                for claim in proposed_claims
                for evidence_id in claim.get("evidence_ids", [])
                if isinstance(evidence_id, str)
            }
            if cited_evidence_ids:
                verification_context, verification_evidence = (
                    evidence_manager.build_context(only_ids=cited_evidence_ids)
                )
                verification_context = add_calculations(
                    verification_context,
                    cited_evidence_ids,
                )
                if cycle_trace is not None:
                    cycle_trace["verification_evidence_ids"] = [
                        evidence.evidence_id for evidence in verification_evidence
                    ]
            valid, feedback, verifier_queries, ledger = audit(
                proposed_answer,
                verification_context,
                verification_evidence,
            )

        if cycle_trace is not None:
            cycle_trace.update({
                "action": "verified_answer" if valid else "rejected_answer",
                "answer": proposed_answer,
                "verification": ledger,
            })
        if valid:
            return finish(
                complete_answer(proposed_answer, ledger),
                ledger=ledger,
                claims=proposed_claims,
            )

        harness_state = ledger.get("_harness")
        temporal_resolution_required = (
            harness_state.get("temporal_resolution_required") is True
            if isinstance(harness_state, dict)
            else False
        )
        if temporal_resolution_required:
            excluded = set(
                harness_state.get("temporal_excluded_evidence_ids", [])
            )
            projected_ids = visible_ids - excluded
            projected_context, projected_evidence = evidence_manager.build_context(
                only_ids=projected_ids,
            )
            projected_context = add_calculations(projected_context, projected_ids)
            projected_answer = ask(build_prompt(
                query,
                projected_context,
                include_citations=include_citations,
                verification_feedback=feedback,
            ))
            (
                projected_valid,
                projected_feedback,
                projected_queries,
                projected_ledger,
            ) = audit(
                projected_answer,
                projected_context,
                projected_evidence,
            )
            if cycle_trace is not None:
                cycle_trace["temporal_revision"] = {
                    "policy": "latest_per_independent_fact",
                    "excluded_evidence_ids": sorted(excluded),
                    "answer": projected_answer,
                    "verification": projected_ledger,
                    "valid": projected_valid,
                }
            if projected_valid:
                formatted_answer = (
                    format_verified_answer(projected_answer, projected_ledger)
                    if _needs_sorted_object_format(query)
                    else projected_answer
                )
                if cycle_trace is not None:
                    cycle_trace["temporal_revision"]["formatted_answer"] = (
                        formatted_answer
                    )
                return finish(
                    formatted_answer,
                    ledger=projected_ledger,
                )
            supported_projected_claims = [
                claim
                for claim in projected_ledger.get("claims", [])
                if isinstance(claim, dict) and claim.get("status") == "supported"
            ]
            if supported_projected_claims and _needs_sorted_object_format(query):
                formatted_answer = format_verified_answer(
                    projected_answer,
                    projected_ledger,
                )
                (
                    formatted_valid,
                    formatted_feedback,
                    formatted_queries,
                    formatted_ledger,
                ) = audit(
                    formatted_answer,
                    projected_context,
                    projected_evidence,
                )
                if cycle_trace is not None:
                    cycle_trace["temporal_revision"]["format_revision"] = {
                        "answer": formatted_answer,
                        "verification": formatted_ledger,
                        "valid": formatted_valid,
                    }
                if formatted_valid:
                    return finish(formatted_answer, ledger=formatted_ledger)
                if (
                    not formatted_queries
                    and ledger_is_grounded(formatted_ledger)
                ):
                    if cycle_trace is not None:
                        cycle_trace["temporal_revision"]["format_revision"][
                            "accepted_as_grounded"
                        ] = True
                    return finish(formatted_answer, ledger=formatted_ledger)
                projected_feedback = formatted_feedback
                projected_queries = formatted_queries
            elif not projected_queries and ledger_is_grounded(projected_ledger):
                return finish(projected_answer, ledger=projected_ledger)
            if not projected_queries:
                return finish(projected_answer, ledger=projected_ledger)
            feedback = projected_feedback
            verifier_queries = projected_queries

        if final_cycle:
            revised_answer, revised_claims = request_answer(
                context,
                feedback,
            )
            (
                revision_valid,
                _revision_feedback,
                _revision_queries,
                revision_ledger,
            ) = audit(revised_answer, context, visible_evidence)
            if cycle_trace is not None:
                cycle_trace["final_revision"] = {
                    "answer": revised_answer,
                    "verification": revision_ledger,
                    "valid": revision_valid,
                }
            if revision_valid:
                return finish(
                    complete_answer(revised_answer, revision_ledger),
                    ledger=revision_ledger,
                    claims=revised_claims,
                )
            for partial_ledger in (revision_ledger, ledger):
                partial_answer = answer_from_supported_claims(partial_ledger)
                if partial_answer:
                    return finish(partial_answer, ledger=partial_ledger)
            return finish(revised_answer, ledger=revision_ledger)

        verification_feedback = feedback
        pending_queries = [
            item
            for item in verifier_queries
            if item not in searched_queries
        ]

    raise RuntimeError("Agentic RAG 未生成答案")
