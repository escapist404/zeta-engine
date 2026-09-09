"""Parsing and validation for model-produced RAG protocol messages."""

import json
import re

from zeta_engine.rag.config import AGENT_MAX_QUERIES
from zeta_engine.rag.evidence import AnswerSlot, AgentDecision, _clean
from zeta_engine.infrastructure.tokenizer import text_normalize


def _parse_json_object(response: str) -> dict[str, object]:
    response = response.strip()
    if response.startswith("```"):
        response = response.removeprefix("```json").removeprefix("```")
        response = response.removesuffix("```").strip()
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Agent 未返回 JSON 对象")
    payload = json.loads(response[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型返回值必须是 JSON 对象")
    return payload


def _parse_answer_slots(payload: dict[str, object]) -> list[AnswerSlot]:
    raw_slots = payload.get("slots", [])
    if not isinstance(raw_slots, list) or not all(
        isinstance(slot, dict) for slot in raw_slots
    ):
        raise ValueError("Agent slots 必须是对象列表")
    slots = []
    seen_ids = set()
    for raw_slot in raw_slots:
        slot_id = _clean(raw_slot.get("id"))
        question = _clean(raw_slot.get("question"))
        status = raw_slot.get("status")
        evidence_ids = raw_slot.get("evidence_ids", [])
        if (
            not slot_id
            or slot_id in seen_ids
            or not question
            or status not in {"missing", "candidate", "answered"}
            or not isinstance(evidence_ids, list)
            or not all(isinstance(item, str) for item in evidence_ids)
        ):
            raise ValueError("Agent slot 格式不正确")
        seen_ids.add(slot_id)
        slots.append(AnswerSlot(
            slot_id=slot_id,
            question=question,
            status=str(status),
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        ))
    return slots


def _parse_agent_action(response: str) -> AgentDecision:
    payload = _parse_json_object(response)
    slots = _parse_answer_slots(payload)

    action = payload.get("action")
    if action == "answer":
        answer = _clean(payload.get("answer"))
        if not answer:
            raise ValueError("Agent 答案为空")
        raw_claims = payload.get("claims", [])
        if not isinstance(raw_claims, list) or not all(
            isinstance(claim, dict) for claim in raw_claims
        ):
            raise ValueError("Agent claims 必须是对象列表")
        claims = []
        for claim in raw_claims:
            statement = _clean(
                claim.get("statement") or claim.get("claim") or claim.get("text")
            )
            evidence_ids = claim.get("evidence_ids", [])
            calculation_ids = claim.get("calculation_ids", [])
            if (
                not statement
                or not isinstance(evidence_ids, list)
                or not all(isinstance(item, str) for item in evidence_ids)
                or not isinstance(calculation_ids, list)
                or not all(isinstance(item, (str, int)) for item in calculation_ids)
            ):
                raise ValueError("Agent claim 格式不正确")
            claims.append({
                "statement": statement,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "calculation_ids": calculation_ids,
            })
        return AgentDecision(
            action=action,
            answer=answer,
            claims=claims,
            slots=slots,
        )
    if action == "search":
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise ValueError("Agent 搜索查询必须是列表")
        slot_ids = {slot.slot_id for slot in slots}
        slot_statuses = {slot.slot_id: slot.status for slot in slots}
        queries = []
        query_slots = {}
        for raw_query in raw_queries:
            if isinstance(raw_query, str):
                search_query = text_normalize(raw_query)
                slot_id = ""
            elif isinstance(raw_query, dict):
                search_query = text_normalize(str(raw_query.get("query", "")))
                slot_id = _clean(raw_query.get("slot_id"))
                if not slot_id or (slot_ids and slot_id not in slot_ids):
                    raise ValueError("Agent 搜索引用了无效 slot_id")
                if slot_statuses.get(slot_id) == "answered":
                    raise ValueError("Agent 不得继续搜索已回答的 slot")
            else:
                continue
            if not search_query or search_query in queries:
                continue
            queries.append(search_query)
            if slot_id:
                query_slots[search_query] = slot_id
            if len(queries) == AGENT_MAX_QUERIES:
                break
        if not queries:
            raise ValueError("Agent 未返回有效搜索查询")
        return AgentDecision(
            action=action,
            queries=queries,
            query_slots=query_slots,
            slots=slots,
        )
    if action == "calculate":
        calculations = payload.get("calculations")
        if not isinstance(calculations, list) or not calculations or not all(
            isinstance(calculation, dict) for calculation in calculations
        ):
            raise ValueError("Agent 计算请求必须是非空对象列表")
        return AgentDecision(
            action=action,
            calculations=calculations,
            slots=slots,
        )
    if action == "collection":
        collection_query = text_normalize(str(payload.get("query", "")))
        if not collection_query:
            raise ValueError("Agent 集合工具查询不能为空")
        return AgentDecision(
            action=action,
            collection_query=collection_query,
            slots=slots,
        )
    raise ValueError(f"不支持的 Agent 动作: {action}")

def _valid_calculation_reference(
    value: object,
    names: set[str],
    count: int,
) -> bool:
    if isinstance(value, str):
        if value in names:
            return True
        match = re.fullmatch(r"\[?计算(\d+)\]?", _clean(value))
        return bool(match and 1 <= int(match.group(1)) <= count)
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= count
    )


def _parse_verification(
    response: str,
    *,
    allowed_evidence_ids: list[str],
    calculation_names: set[str] | None = None,
    calculation_count: int = 0,
) -> tuple[bool, str, list[str], dict[str, object]]:
    """Validate a verifier ledger and return its decision and feedback."""

    payload = _parse_json_object(response)
    calculation_names = calculation_names or set()
    allowed = set(allowed_evidence_ids)
    requirements = payload.get("requirements")
    claims = payload.get("claims")
    conflicts = payload.get("conflicts")
    raw_issues = payload.get("issues")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Verifier 必须列出回答要求")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Verifier 必须列出关键断言")
    if not isinstance(conflicts, list) or not isinstance(raw_issues, list):
        raise ValueError("Verifier 的冲突或问题格式不正确")
    issues: list[dict[str, str]] = []
    for issue in raw_issues:
        if isinstance(issue, str) and _clean(issue):
            issues.append({"type": "incomplete", "description": _clean(issue)})
        elif (
            isinstance(issue, dict)
            and isinstance(issue.get("type"), str)
            and isinstance(issue.get("description"), str)
        ):
            issues.append({
                "type": issue["type"],
                "description": issue["description"],
            })
        else:
            raise ValueError("Verifier 问题格式不正确")
    payload["issues"] = issues

    def referenced_evidence(item: object) -> list[str]:
        if not isinstance(item, dict):
            raise ValueError("Verifier 账本条目必须是对象")
        raw_ids = item.get("evidence_ids")
        if raw_ids is None:
            source_ids = item.get("source_ids")
            if not isinstance(source_ids, list) or not all(
                isinstance(source_id, int)
                and not isinstance(source_id, bool)
                and 1 <= source_id <= len(allowed_evidence_ids)
                for source_id in source_ids
            ):
                raise ValueError("Verifier 引用了无效文档编号")
            raw_ids = [allowed_evidence_ids[source_id - 1] for source_id in source_ids]
        if not isinstance(raw_ids, list) or not all(
            isinstance(evidence_id, str) and evidence_id in allowed
            for evidence_id in raw_ids
        ):
            raise ValueError("Verifier 引用了当前上下文之外的证据ID")
        normalized = list(dict.fromkeys(raw_ids))
        item["evidence_ids"] = normalized
        return normalized

    requirement_ok = True
    for requirement in requirements:
        referenced_evidence(requirement)
        if (
            not isinstance(requirement.get("description"), str)
            or not isinstance(requirement.get("satisfied"), bool)
        ):
            raise ValueError("Verifier 回答要求格式不正确")
        requirement_ok &= requirement["satisfied"]

    claim_ok = True
    for claim in claims:
        referenced_evidence(claim)
        calculation_ids = claim.get("calculation_ids", [])
        if not isinstance(calculation_ids, list) or not all(
            _valid_calculation_reference(
                calculation_id,
                calculation_names,
                calculation_count,
            )
            for calculation_id in calculation_ids
        ):
            raise ValueError("Verifier 引用了无效计算结果")
        if (
            not isinstance(claim.get("statement"), str)
            or claim.get("status")
            not in {"supported", "partial", "conflicting", "unsupported"}
        ):
            raise ValueError("Verifier 断言格式不正确")
        claim_ok &= claim["status"] == "supported"

    conflict_ok = True
    for conflict in conflicts:
        referenced_evidence(conflict)
        if (
            not isinstance(conflict.get("description"), str)
            or not isinstance(conflict.get("resolved"), bool)
            or (
                conflict.get("resolved")
                and not _clean(conflict.get("resolution"))
            )
        ):
            raise ValueError("Verifier 冲突格式不正确")
        conflict_ok &= conflict["resolved"]

    feedback = []
    for issue in issues:
        feedback.append(f"{issue['type']}: {issue['description']}")
    feedback.extend(
        f"conflict: {conflict['description']}"
        for conflict in conflicts
        if not conflict["resolved"]
    )
    feedback.extend(
        f"incomplete: {requirement['description']}"
        for requirement in requirements
        if not requirement["satisfied"]
    )

    raw_queries = payload.get("queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError("Verifier 检索查询必须是列表")
    queries = list(dict.fromkeys(
        text_normalize(query)
        for query in raw_queries
        if isinstance(query, str) and text_normalize(query)
    ))[:AGENT_MAX_QUERIES]
    valid = (
        payload.get("valid") is True
        and requirement_ok
        and claim_ok
        and conflict_ok
        and not issues
    )
    if not valid and not feedback:
        feedback.append("候选答案未通过证据审计")
    return valid, "\n".join(feedback), queries, payload
