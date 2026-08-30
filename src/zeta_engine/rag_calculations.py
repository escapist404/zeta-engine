"""Grounded deterministic calculation execution and rendering."""

import json
import re
from hashlib import sha256

from zeta_engine.rag_evidence import (
    Evidence,
    _clean,
    _grounding_text,
    _normalize_text,
)


def _run_calculations(
    specifications: list[dict[str, object]],
    evidence: list[Evidence],
    previous: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Execute grounded, declarative calculations without arbitrary code."""

    known_calculations = {
        str(calculation["name"]): calculation
        for calculation in previous
    }
    known = {
        name: calculation["result"]
        for name, calculation in known_calculations.items()
    }
    input_fingerprints = {
        fingerprint
        for calculation in previous
        if (fingerprint := _clean(calculation.get("input_fingerprint")))
    }
    evidence_by_id = {item.evidence_id: item for item in evidence}
    completed = []

    def reserve_input(
        operator: str,
        payload: object,
        evidence_ids: list[str],
        parameters: object = None,
    ) -> str:
        fingerprint = sha256(json.dumps(
            {
                "operator": operator,
                "payload": payload,
                "evidence_ids": sorted(set(evidence_ids)),
                "parameters": parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()).hexdigest()
        if fingerprint in input_fingerprints:
            raise ValueError("不得以不同名称重复提交完全相同的计算")
        input_fingerprints.add(fingerprint)
        return fingerprint

    for specification in specifications:
        name = _clean(specification.get("name"))
        operator = specification.get("operator")
        if not name or name in known:
            raise ValueError("计算名称为空或重复")
        set_operators = {"intersection", "union", "difference", "complement"}
        arithmetic_operators = {"add", "subtract", "multiply", "divide", "ratio"}
        if operator not in {
            "count", "count_unique", "sum", "min", "max", "sort", "classify",
            *set_operators, *arithmetic_operators,
        }:
            raise ValueError("不支持的计算操作")

        if operator in set_operators:
            raw_sets = specification.get("sets")
            if not isinstance(raw_sets, list) or not 2 <= len(raw_sets) <= 50:
                raise ValueError("集合运算需要 2 至 50 个集合")
            sets: list[list[object]] = []
            used_evidence_ids: list[str] = []
            for raw_set in raw_sets:
                if not isinstance(raw_set, dict):
                    raise ValueError("集合必须是对象")
                if "calculation" in raw_set:
                    reference = _clean(raw_set.get("calculation"))
                    if reference not in known or not isinstance(known[reference], list):
                        raise ValueError("集合引用必须指向已有集合结果")
                    sets.append(list(dict.fromkeys(known[reference])))
                    used_evidence_ids.extend(
                        known_calculations[reference].get("evidence_ids", [])
                    )
                    continue

                raw_values = raw_set.get("values")
                raw_ids = raw_set.get("evidence_ids")
                if not isinstance(raw_values, list) or not 1 <= len(raw_values) <= 2000:
                    raise ValueError("每个集合必须包含 1 至 2000 个值")
                if not isinstance(raw_ids, list) or not raw_ids or not all(
                    isinstance(evidence_id, str) and evidence_id in evidence_by_id
                    for evidence_id in raw_ids
                ):
                    raise ValueError("每个集合必须引用有效证据ID")
                source_text = " ".join(
                    _grounding_text(evidence_by_id[evidence_id])
                    for evidence_id in raw_ids
                )
                occurrences: dict[str, int] = {}
                values = []
                for raw_value in raw_values:
                    if (
                        isinstance(raw_value, bool)
                        or not isinstance(raw_value, (str, int, float))
                    ):
                        raise ValueError("集合输入只支持文本或数字")
                    value = (
                        _normalize_text(raw_value)
                        if isinstance(raw_value, str)
                        else raw_value
                    )
                    if value == "":
                        raise ValueError("集合输入不能为空")
                    key = str(value)
                    occurrences[key] = occurrences.get(key, 0) + 1
                    if source_text.count(key) < occurrences[key]:
                        raise ValueError(f"集合输入不受来源支持: {value}")
                    values.append(value)
                sets.append(list(dict.fromkeys(values)))
                used_evidence_ids.extend(raw_ids)

            if operator == "intersection":
                other_sets = [set(values) for values in sets[1:]]
                result = [
                    value for value in sets[0]
                    if all(value in values for values in other_sets)
                ]
            elif operator == "union":
                result = list(dict.fromkeys(
                    value for values in sets for value in values
                ))
            else:
                excluded = set().union(*(set(values) for values in sets[1:]))
                result = [value for value in sets[0] if value not in excluded]

            input_fingerprint = reserve_input(
                str(operator),
                sets,
                used_evidence_ids,
            )
            calculation = {
                "name": name,
                "operator": operator,
                "evidence_ids": list(dict.fromkeys(used_evidence_ids)),
                "input_count": sum(len(values) for values in sets),
                "result": result,
                "input_fingerprint": input_fingerprint,
            }
            completed.append(calculation)
            known[name] = result
            known_calculations[name] = calculation
            continue

        raw_items = specification.get("items")
        if raw_items is None:
            raw_values = specification.get("values")
            raw_evidence_ids = specification.get("evidence_ids")
            raw_source_ids = specification.get("source_ids")
            if raw_evidence_ids is None and isinstance(raw_source_ids, list):
                if not all(
                    isinstance(source_id, int)
                    and not isinstance(source_id, bool)
                    and 1 <= source_id <= len(evidence)
                    for source_id in raw_source_ids
                ):
                    raise ValueError("计算引用了无效文档编号")
                raw_evidence_ids = [
                    evidence[source_id - 1].evidence_id
                    for source_id in raw_source_ids
                ]
            if not isinstance(raw_values, list):
                raise ValueError("计算输入必须是列表")
            if not isinstance(raw_evidence_ids, list) or not all(
                isinstance(evidence_id, str) and evidence_id in evidence_by_id
                for evidence_id in raw_evidence_ids
            ):
                raise ValueError("计算引用了无效证据ID")
            raw_items = [
                raw_value
                if isinstance(raw_value, dict)
                else {"value": raw_value, "evidence_ids": raw_evidence_ids}
                for raw_value in raw_values
            ]
        if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 2000:
            raise ValueError("计算输入必须是 1 至 2000 项的列表")

        values = []
        used_evidence_ids: list[str] = []
        occurrences: dict[tuple[str, str], int] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise ValueError("计算 item 必须是对象")
            if "calculation" in raw_item:
                reference = _clean(raw_item.get("calculation"))
                if not reference or reference not in known:
                    raise ValueError("计算引用了未知的先前结果")
                referenced = known[reference]
                if isinstance(referenced, list):
                    values.extend(referenced)
                else:
                    values.append(referenced)
                used_evidence_ids.extend(
                    known_calculations[reference].get("evidence_ids", [])
                )
                continue

            raw_value = raw_item.get("value")
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (str, int, float))
            ):
                raise ValueError("计算输入只支持文本、数字或先前结果")
            value = _normalize_text(raw_value) if isinstance(raw_value, str) else raw_value
            if value == "":
                raise ValueError("计算输入不能为空")
            raw_ids = raw_item.get("evidence_ids")
            if raw_ids is None:
                raw_ids = [raw_item.get("evidence_id")]
            if not isinstance(raw_ids, list) or not raw_ids or not all(
                isinstance(evidence_id, str) and evidence_id in evidence_by_id
                for evidence_id in raw_ids
            ):
                raise ValueError("每个直接计算输入必须引用有效证据ID")
            source_text = " ".join(
                _grounding_text(evidence_by_id[evidence_id])
                for evidence_id in raw_ids
            )
            occurrence_key = ("\0".join(raw_ids), str(value))
            occurrences[occurrence_key] = occurrences.get(occurrence_key, 0) + 1
            if source_text.count(str(value)) < occurrences[occurrence_key]:
                raise ValueError(f"计算输入不受来源支持: {value}")
            values.append(value)
            used_evidence_ids.extend(raw_ids)

        if operator in arithmetic_operators:
            if (
                len(values) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in values
                )
            ):
                raise ValueError(f"{operator} 需要恰好两个数字输入")
            left, right = values
            if operator == "add":
                result: object = left + right
            elif operator == "subtract":
                result = left - right
            elif operator == "multiply":
                result = left * right
            else:
                if right == 0:
                    raise ValueError("除数不能为 0")
                result = left / right
        elif operator == "classify":
            if len(values) != 1 or not isinstance(values[0], (int, float)):
                raise ValueError("classify 需要恰好一个数字输入")
            raw_rules = specification.get("rules")
            if not isinstance(raw_rules, list) or not 1 <= len(raw_rules) <= 100:
                raise ValueError("classify 需要 1 至 100 条区间规则")
            matches = []
            value = values[0]
            for raw_rule in raw_rules:
                if not isinstance(raw_rule, dict):
                    raise ValueError("classify 的每条规则必须是对象")
                label = _clean(raw_rule.get("label"))
                bounds = {
                    key: raw_rule[key]
                    for key in ("gt", "gte", "lt", "lte")
                    if key in raw_rule
                }
                if not label or not bounds or not all(
                    isinstance(bound, (int, float))
                    and not isinstance(bound, bool)
                    for bound in bounds.values()
                ):
                    raise ValueError("classify 规则需要标签和数值边界")
                matched = (
                    ("gt" not in bounds or value > bounds["gt"])
                    and ("gte" not in bounds or value >= bounds["gte"])
                    and ("lt" not in bounds or value < bounds["lt"])
                    and ("lte" not in bounds or value <= bounds["lte"])
                )
                if matched:
                    matches.append(label)
            if len(matches) != 1:
                raise ValueError("classify 输入必须且只能匹配一条区间规则")
            result: object = matches[0]
        elif operator == "count":
            result: object = len(values)
        elif operator == "count_unique":
            result = len(dict.fromkeys(values))
        elif operator == "sum":
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                raise ValueError("sum 只支持数字")
            result = sum(values)
        else:
            if not (
                all(isinstance(value, str) for value in values)
                or all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in values
                )
            ):
                raise ValueError(f"{operator} 的输入类型必须一致")
            if operator == "min":
                result = min(values)
            elif operator == "max":
                result = max(values)
            else:
                result = sorted(values)

        input_fingerprint = reserve_input(
            str(operator),
            values,
            used_evidence_ids,
            specification.get("rules") if operator == "classify" else None,
        )
        calculation = {
            "name": name,
            "operator": operator,
            "evidence_ids": list(dict.fromkeys(used_evidence_ids)),
            "input_count": len(values),
            "result": result,
            "input_fingerprint": input_fingerprint,
        }
        completed.append(calculation)
        known[name] = result
        known_calculations[name] = calculation
    return completed


def _format_calculations(calculations: list[dict[str, object]]) -> str:
    return "\n".join(
        f"[计算{index}] 名称：{calculation['name']}；"
        f"操作：{calculation['operator']}；"
        f"来源证据：{calculation['evidence_ids']}；"
        f"输入项数：{calculation['input_count']}；"
        f"结果：{json.dumps(calculation['result'], ensure_ascii=False)}"
        for index, calculation in enumerate(calculations, start=1)
    )

def _deterministic_calculation_answer(
    query: str,
    calculations: list[dict[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    """Render a terminal scalar calculation when the requested unit is clear."""

    if not re.search(r"(?:一共|总共|合计).{0,20}(?:多少|几).{0,6}次", query):
        return "", []
    candidates = [
        calculation
        for calculation in calculations
        if calculation.get("operator") == "sum"
        and isinstance(calculation.get("result"), (int, float))
        and not isinstance(calculation.get("result"), bool)
    ]
    if not candidates:
        return "", []
    terminal = candidates[-1]
    result = terminal["result"]
    rendered = str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)
    statement = f"一共{rendered}次"
    return statement + "。", [{
        "statement": statement,
        "evidence_ids": list(terminal.get("evidence_ids", [])),
        "calculation_ids": [str(terminal["name"])],
    }]
