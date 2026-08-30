"""Deterministic answer-selection and temporal evidence policies."""

import json
import re

from zeta_engine.rag_evidence import Evidence, _clean, _grounding_text
from zeta_engine.tokenizer import tokenize_with_positions


def _range_classification_rules(
    query: str,
) -> list[tuple[float | None, bool, float | None, bool, str]]:
    unit = r"(?:人|名|个|项|条)?"
    label = r"([^，,。；;]+)"
    rules: list[tuple[float | None, bool, float | None, bool, str]] = []
    for match in re.finditer(rf"不满\s*(\d+(?:\.\d+)?)\s*{unit}\s*使用\s*{label}", query):
        rules.append((None, False, float(match.group(1)), False, _clean(match.group(2))))
    for match in re.finditer(
        rf"(\d+(?:\.\d+)?)\s*至\s*(\d+(?:\.\d+)?)\s*{unit}\s*使用\s*{label}",
        query,
    ):
        rules.append((
            float(match.group(1)), True, float(match.group(2)), True,
            _clean(match.group(3)),
        ))
    for match in re.finditer(rf"超过\s*(\d+(?:\.\d+)?)\s*{unit}\s*使用\s*{label}", query):
        rules.append((float(match.group(1)), False, None, False, _clean(match.group(2))))
    return rules


def _deterministic_collection_bucket_answer(
    query: str,
    evidence: list[Evidence],
) -> tuple[str, list[dict[str, object]]]:
    years = list(dict.fromkeys(re.findall(r"(?<!\d)(20\d{2})(?!\d)", query)))
    rules = _range_classification_rules(query)
    if len(years) < 2 or len(rules) < 2:
        return "", []

    facts: dict[str, tuple[int, int, str]] = {}
    for item in evidence:
        if item.result.get("collection_complete") != "true":
            continue
        raw_size = _clean(item.result.get("collection_size"))
        if not raw_size.isdigit():
            continue
        for year in years:
            if year not in item.title:
                continue
            size = int(raw_size)
            priority = (
                2
                if item.result.get("evidence_scope") == "complete_document_tables"
                else 1
            )
            previous = facts.get(year)
            if previous is not None:
                previous_priority, previous_size, _previous_id = previous
                if priority < previous_priority:
                    continue
                if priority == previous_priority and previous_size != size:
                    return "", []
            facts[year] = (priority, size, item.evidence_id)
    if any(year not in facts for year in years):
        return "", []

    rows = []
    claims = []
    for year in years:
        _priority, size, evidence_id = facts[year]
        labels = []
        for lower, lower_inclusive, upper, upper_inclusive, value in rules:
            lower_ok = lower is None or size > lower or (lower_inclusive and size == lower)
            upper_ok = upper is None or size < upper or (upper_inclusive and size == upper)
            if lower_ok and upper_ok:
                labels.append(value)
        if len(labels) != 1:
            return "", []
        statement = f"{year}年使用{labels[0]}"
        rows.append(statement)
        claims.append({
            "statement": f"{year}年完整集合共{size}项，{statement}",
            "evidence_ids": [evidence_id],
            "calculation_ids": [],
        })
    return "；".join(rows) + "。", claims


def _deterministic_complete_table_ratio_answer(
    query: str,
    evidence: list[Evidence],
) -> tuple[str, list[dict[str, object]]]:
    """Materialize an explicit ratio bound returned by the complete-table tool."""

    if not re.search(r"(?:比(?:例)?|率)", query):
        return "", []
    lower = any(marker in query for marker in (
        "最低", "最少", "下界", "至少", "不少于", "不低于",
    ))
    upper = any(marker in query for marker in (
        "最高", "最多", "上界", "至多", "不超过", "不高于",
    ))
    if lower == upper:
        return "", []
    direction = "lower" if lower else "upper"
    years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", query))
    normalized_query = query.replace("学术硕士", "学硕")
    query_terms = {
        term
        for term, _offset in tokenize_with_positions(
            normalized_query,
            mode="search",
        )
        if len(term) >= 2
    }
    matches: list[tuple[int, float, int, str]] = []
    for item in evidence:
        if item.result.get("evidence_scope") != "complete_table_section":
            continue
        if years and not years.intersection(re.findall(r"20\d{2}", item.title)):
            continue
        raw_relations = item.result.get("derived_relations")
        if not isinstance(raw_relations, str):
            continue
        try:
            relations = json.loads(raw_relations)
        except json.JSONDecodeError:
            continue
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if (
                not isinstance(relation, dict)
                or relation.get("kind") != "ratio_bound"
                or relation.get("direction") != direction
            ):
                continue
            numerator = relation.get("numerator")
            denominator = relation.get("denominator")
            if (
                not isinstance(numerator, (int, float))
                or isinstance(numerator, bool)
                or not isinstance(denominator, int)
                or isinstance(denominator, bool)
                or numerator < 0
                or denominator <= 0
                or numerator > denominator
            ):
                continue
            normalized_title = item.title.replace("学术硕士", "学硕")
            relevance = sum(
                len(term) for term in query_terms if term in normalized_title
            )
            matches.append((
                relevance,
                float(numerator),
                denominator,
                item.evidence_id,
            ))
    if not matches:
        return "", []
    best_relevance = max(item[0] for item in matches)
    unique = list(dict.fromkeys(
        item[1:] for item in matches if item[0] == best_relevance
    ))
    if len(unique) != 1:
        return "", []

    numerator, denominator, evidence_id = unique[0]
    rendered_numerator = (
        str(int(numerator)) if numerator.is_integer() else str(numerator)
    )
    percentage = numerator / denominator * 100
    qualifier = "最低" if direction == "lower" else "最高"
    answer = (
        f"{qualifier}比例为{percentage:.2f}%"
        f"（{rendered_numerator}/{denominator}≈{percentage / 100:.4f}）。"
    )
    claim = (
        f"完整表格共{denominator}项，数量{qualifier}边界为"
        f"{rendered_numerator}项，{qualifier}比例为{percentage:.2f}%"
    )
    return answer, [{
        "statement": claim,
        "evidence_ids": [evidence_id],
        "calculation_ids": [],
    }]


def _requires_complete_collection(query: str) -> bool:
    """Require full-set evidence for bounded ratios and cross-group bucketing."""

    has_ratio = re.search(r"(?:比(?:例)?|率)", query) is not None
    has_bound = any(marker in query for marker in (
        "最低", "最少", "下界", "至少", "不少于", "不低于",
        "最高", "最多", "上界", "至多", "不超过", "不高于",
    ))
    if has_ratio and has_bound:
        return True
    years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", query))
    return len(years) >= 2 and len(_range_classification_rules(query)) >= 2


def _prefer_latest_record(query: str) -> bool:
    if re.search(r"(?:多少|几).{0,6}次|次数|累计", query):
        return False
    if any(marker in query for marker in (
        "最早", "首次", "起初", "之前", "以前", "历年", "历任", "曾经", "变化",
    )):
        return False
    if re.search(r"(?:19|20)\d{2}年?", query):
        return False
    return True


def _needs_sorted_object_format(query: str) -> bool:
    return any(marker in query for marker in ("顺序", "排序", "排列"))


def _query_company_entities(query: str) -> list[str]:
    entities = []
    for match in re.finditer("公司", query):
        prefix = query[:match.start()]
        pieces = re.split(
            r"(?:参访|走进|前往|赴|和|与|及)|[，。！？；：、,;:\s]",
            prefix,
        )
        stem = pieces[-1].strip() if pieces else ""
        stem = re.sub(r"^.*?(?:中|为)", "", stem)
        if 1 <= len(stem) <= 12:
            entities.append(stem + "公司")
    return list(dict.fromkeys(entities))


def _parse_ordinal_number(value: str) -> int | None:
    value = value.strip()
    if value.isdigit():
        return int(value)
    digits = {
        "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
        "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000}
    if not value or any(character not in digits | units for character in value):
        return None
    total = 0
    current = 0
    for character in value:
        if character in digits:
            current = digits[character]
        else:
            if current == 0:
                current = 1
            total += current * units[character]
            current = 0
    return total + current


def _deterministic_numbered_event_answer(
    query: str,
    evidence: list[Evidence],
) -> tuple[str, list[dict[str, object]]]:
    """Count unique numbered events for the companies named by the query."""

    if not re.search(r"(?:一共|总共|合计).{0,20}(?:多少|几).{0,6}次", query):
        return "", []
    entities = _query_company_entities(query)
    if not entities:
        return "", []

    events: dict[tuple[str, int], str] = {}
    for item in evidence:
        text = " ".join(filter(None, (item.title, _grounding_text(item))))
        stations = list(re.finditer(
            r"第\s*([0-9零〇一二两三四五六七八九十百千]+)\s*站",
            text,
        ))
        for index, station in enumerate(stations):
            next_start = (
                stations[index + 1].start()
                if index + 1 < len(stations)
                else len(text)
            )
            start = max(0, station.start() - 40)
            end = min(next_start, station.end() + 160)
            window = text[start:end]
            candidates = []
            relative_station = station.start() - start
            for entity in entities:
                for entity_match in re.finditer(re.escape(entity), window):
                    candidates.append((
                        abs(entity_match.start() - relative_station),
                        entity,
                    ))
            if not candidates:
                continue
            distance, entity = min(candidates)
            station_number = _parse_ordinal_number(station.group(1))
            if distance <= 120 and station_number is not None:
                events.setdefault((entity, station_number), item.evidence_id)

    if not events or any(
        not any(entity == event_entity for event_entity, _station in events)
        for entity in entities
    ):
        return "", []
    ordered = sorted(events.items(), key=lambda item: (item[0][1], item[0][0]))
    answer = f"一共{len(ordered)}次。"
    return answer, [{
        "statement": (
            f"{entity}第{station}站参访计为一次独立活动"
        ),
        "evidence_ids": [evidence_id],
        "calculation_ids": [],
    } for (entity, station), evidence_id in ordered]



def _title_entity_keys(
    query: str,
    evidence: list[Evidence],
) -> dict[str, str]:
    terms = list(dict.fromkeys(
        term
        for term, _offset in tokenize_with_positions(query, mode="search")
        if len(term) >= 2
    ))
    document_frequency = {
        term: sum(term in item.title for item in evidence)
        for term in terms
    }
    keys = {}
    for item in evidence:
        candidates = [term for term in terms if term in item.title]
        if candidates:
            keys[item.evidence_id] = min(
                candidates,
                key=lambda term: (document_frequency[term], -len(term), term),
            )
    return keys


def _superseded_evidence_ids(
    query: str,
    evidence: list[Evidence],
) -> set[str]:
    """Find older dated records for the same query entity from page titles."""

    if not _prefer_latest_record(query):
        return set()
    entity_keys = _title_entity_keys(query, evidence)
    groups: dict[str, list[Evidence]] = {}
    for item in evidence:
        if not item.published_at:
            continue
        key = entity_keys.get(item.evidence_id)
        if key is None:
            continue
        groups.setdefault(key, []).append(item)

    excluded = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        latest_date = max(item.published_at for item in group)
        excluded.update(
            item.evidence_id
            for item in group
            if item.published_at < latest_date
        )
    return excluded


def _deterministic_sorted_answer(
    query: str,
    evidence: list[Evidence],
) -> tuple[str, list[dict[str, object]]]:
    if not _needs_sorted_object_format(query):
        return "", []
    entity_keys = _title_entity_keys(query, evidence)
    rows = []
    for item in evidence:
        entity = entity_keys.get(item.evidence_id)
        match = re.search(
            r"第\s*(-?\d+(?:\.\d+)?)\s*(站|项|位|名|届|期|次)",
            item.title,
        )
        if entity is None or match is None:
            continue
        label = f"{entity}公司" if f"{entity}公司" in query else entity
        number, unit = match.groups()
        statement = f"{label}是第{number}{unit}"
        rows.append((float(number), label, statement, item.evidence_id))
    if len(rows) < 2 or len({row[1] for row in rows}) != len(rows):
        return "", []
    rows.sort(key=lambda row: row[0])
    return (
        "、".join(row[2] for row in rows) + "。",
        [
            {
                "statement": row[2],
                "evidence_ids": [row[3]],
                "calculation_ids": [],
            }
            for row in rows
        ],
    )
