"""Structured execution path for exhaustive cross-resource collection QA."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from zeta_engine.collection_ops import (
    Coverage,
    Record,
    RecordCollection,
    ScanResult,
    aggregate,
    declared_record_count,
    field_frequencies,
    filter_records,
    infer_record_kind,
    measure,
    normalize_field_name,
    normalize_value,
    scan,
    segment,
    semantic_map,
    set_op,
)
from zeta_engine.dense import PassageHit
from zeta_engine.rag import call_model
from zeta_engine.rag_types import RagResponse
from zeta_engine.storage import Storage


_COUNT_SIGNALS = re.compile(r"多少|几(?:篇|个|项|条|名|人)|数量|总数|计数")
_GROUP_SIGNALS = re.compile(r"分别|各自|逐年|每年|比较|对比")
_INTERSECTION_SIGNALS = re.compile(r"共同|同时出现|都(?:有|是|出现)|交集|共有")
_RELATIONS = ("包含", "含有", "包括", "等于", "为", "是")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_FIELD_PUNCTUATION = " \t\r\n:：,，。;；?？!！'\"“”‘’"
_TOPIC_STOPWORDS = frozenset({
    "about", "across", "after", "again", "against", "all", "also", "among",
    "and", "another", "are", "based", "before", "better", "between", "beyond",
    "can", "comprehensive", "does", "effective", "efficient", "empowering",
    "enhance", "enhancing", "exploring", "for", "from", "full", "high",
    "improving", "into", "large", "learning", "makes", "models", "more", "not",
    "over", "promote", "scaling", "self", "strong", "study", "the", "through",
    "towards", "toward", "using", "via", "with", "within",
})


@dataclass(frozen=True)
class CollectionPlan:
    """Small domain-neutral plan compiled from collection query signals."""

    group_values: tuple[str, ...]
    needs_count: bool
    needs_intersection: bool


@dataclass(frozen=True)
class PreparedSource:
    """One candidate source after exhaustive scan and segmentation."""

    group_value: str
    rank: int
    document_id: int
    title: str
    url: str
    scanned: ScanResult
    collection: RecordCollection
    coverage: Coverage


@dataclass(frozen=True)
class FieldFilter:
    field: str
    operator: str
    value: str


@dataclass(frozen=True)
class SharedTopic:
    label: str
    evidence: dict[str, tuple[str, ...]]


def plan_collection_query(query: str) -> CollectionPlan | None:
    """Recognize the safe subset that requires exhaustive collection execution."""

    query = normalize_value(query)
    groups = tuple(dict.fromkeys(_YEAR_RE.findall(query)))
    needs_count = _COUNT_SIGNALS.search(query) is not None
    needs_intersection = _INTERSECTION_SIGNALS.search(query) is not None
    if (
        len(groups) < 2
        or not (needs_count or needs_intersection)
        or _GROUP_SIGNALS.search(query) is None
    ):
        return None
    return CollectionPlan(
        group_values=groups,
        needs_count=needs_count,
        needs_intersection=needs_intersection,
    )


def _prepare_candidate(
    storage: Storage,
    document_id: int,
    *,
    rank: int,
    group_value: str,
) -> PreparedSource | None:
    assert storage.documents is not None
    document = storage.documents.get(document_id)
    if document is None:
        return None
    url, title, text, _fetched_at = document
    scanned = scan(
        str(document_id),
        title=title,
        url=url,
        text=text,
        content_html=storage.documents.get_content_html(document_id),
    )
    collection = segment(scanned)
    if not collection.records:
        return None
    kind = infer_record_kind(collection.start_field)
    declared = declared_record_count(scanned, kind=kind)
    coverage = measure(scanned, collection, declared_records=declared)
    return PreparedSource(
        group_value=group_value,
        rank=rank,
        document_id=document_id,
        title=title,
        url=url,
        scanned=scanned,
        collection=collection,
        coverage=coverage,
    )


def resolve_sources(
    storage: Storage,
    hits: Sequence[PassageHit],
    plan: CollectionPlan,
) -> tuple[PreparedSource, ...]:
    """Resolve one exhaustive repeated-record source for every planned group."""

    assert storage.documents is not None
    ranked_document_ids: list[int] = []
    ranks: dict[int, int] = {}
    for rank, hit in enumerate(hits, start=1):
        if hit.document_id in ranks:
            continue
        ranks[hit.document_id] = rank
        ranked_document_ids.append(hit.document_id)

    selected: list[PreparedSource] = []
    used_document_ids: set[int] = set()
    for group_value in plan.group_values:
        candidates: list[PreparedSource] = []
        for document_id in ranked_document_ids:
            if document_id in used_document_ids:
                continue
            document = storage.documents.get(document_id)
            if document is None:
                continue
            _url, title, text, _fetched_at = document
            scope_text = f"{title} {text[:1000]}"
            if group_value not in scope_text:
                continue
            prepared = _prepare_candidate(
                storage,
                document_id,
                rank=ranks[document_id],
                group_value=group_value,
            )
            if prepared is not None:
                candidates.append(prepared)
        if not candidates:
            return ()

        # A collection page with explicit matching cardinality is stronger than
        # an individual detail page, then record count and retrieval rank decide.
        chosen = max(candidates, key=lambda item: (
            item.coverage.record_complete is True,
            len(item.collection.records),
            -item.rank,
        ))
        selected.append(chosen)
        used_document_ids.add(chosen.document_id)
    return tuple(selected)


def _available_fields(sources: Iterable[PreparedSource]) -> Counter[str]:
    fields: Counter[str] = Counter()
    for source in sources:
        fields.update(field_frequencies(source.collection.records))
    return fields


def compile_field_filter(
    query: str,
    sources: Sequence[PreparedSource],
) -> FieldFilter | None:
    """Compile ``runtime field + relation + value`` into a safe predicate."""

    query_norm = normalize_value(query)
    fields = _available_fields(sources)
    kinds = {
        infer_record_kind(source.collection.start_field)
        for source in sources
    }
    kinds.discard("")
    for field, _frequency in sorted(
        fields.items(),
        key=lambda item: (-len(item[0]), -item[1], item[0]),
    ):
        start = 0
        while (offset := query_norm.find(field, start)) >= 0:
            suffix = query_norm[offset + len(field):].lstrip()
            relation = next(
                (item for item in _RELATIONS if suffix.startswith(item)),
                "",
            )
            if not relation:
                start = offset + len(field)
                continue
            value_text = suffix[len(relation):].lstrip(_FIELD_PUNCTUATION)
            stops = [
                value_text.find(f"的{kind}")
                for kind in kinds
                if value_text.find(f"的{kind}") >= 0
            ]
            stop = min(stops, default=-1)
            if stop >= 0:
                value_text = value_text[:stop]
            else:
                value_text = re.split(
                    r"\s*(?:分别|各自|多少|哪些|共同|同时|比较|对比|[，。；;？?])",
                    value_text,
                    maxsplit=1,
                )[0]
            value = value_text.strip(_FIELD_PUNCTUATION)
            if value and len(value) <= 80:
                return FieldFilter(field=field, operator="contains", value=value)
            start = offset + len(field)
    return None


def _topic_text(record: Record, start_field: str) -> str:
    title = " ".join(record.values(start_field))
    other = " ".join(
        value
        for field, values in record.fields.items()
        if field != start_field
        for value in values
        if any(marker in field for marker in ("概述", "介绍", "摘要", "描述"))
    )
    return f"{title}。{other[:800]}".strip("。")


def _normalize_topic_token(token: str) -> str:
    token = token.casefold().strip("-_")
    token = token.replace("multi-modal", "multimodal")
    token = token.replace("cross-modal", "multimodal")
    token = token.replace("omni-modal", "multimodal")
    if token.startswith("retriev"):
        return "retrieval"
    if token in {"agent", "agents", "agentic"}:
        return "agentic"
    if token.startswith("embed"):
        return "embedding"
    if token.startswith("rank"):
        return "ranking"
    if token.startswith("reason"):
        return "reasoning"
    return token


def _record_topic_tokens(record: Record, start_field: str) -> set[str]:
    # Titles are concise author-provided descriptors and provide a conservative
    # model-free fallback when the semantic mapper is unavailable.
    title = " ".join(record.values(start_field)).casefold()
    title = title.replace("multi-modal", "multimodal")
    title = title.replace("cross-modal", "multimodal")
    title = title.replace("omni-modal", "multimodal")
    tokens = {
        _normalize_topic_token(token)
        for token in re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", title)
    }
    return {
        token
        for token in tokens
        if len(token) >= 4 and token not in _TOPIC_STOPWORDS
    }


def lexical_shared_topics(
    records_by_group: dict[str, tuple[Record, ...]],
    start_fields: dict[str, str],
    *,
    limit: int = 6,
) -> tuple[SharedTopic, ...]:
    """Return conservative shared title concepts without an external model."""

    tokens_by_group: dict[str, dict[str, list[str]]] = {}
    for group_value, records in records_by_group.items():
        index: dict[str, list[str]] = defaultdict(list)
        assignments = semantic_map(
            records,
            item_id=lambda record: record.record_id,
            mapper=lambda record: _record_topic_tokens(
                record,
                start_fields[group_value],
            ),
        )
        for record_id, labels in assignments.items():
            for token in labels:
                index[token].append(record_id)
        tokens_by_group[group_value] = index
    common = set_op(
        tokens_by_group[next(iter(records_by_group))],
        set.intersection(*(
            set(values) for values in tokens_by_group.values()
        )),
        operator="intersection",
    )
    # The expression above preserves the first group's order; score below makes
    # the final order depend on cross-group support instead of input ordering.
    scored = sorted(
        common,
        key=lambda token: (
            -min(len(index[token]) for index in tokens_by_group.values()),
            -sum(len(index[token]) for index in tokens_by_group.values()),
            token,
        ),
    )

    topics: list[SharedTopic] = []
    consumed: set[str] = set()
    # Merge only near-equivalent concepts whose supporting records overlap in
    # every group; this keeps the fallback mechanical and inspectable.
    for token in scored:
        if token in consumed:
            continue
        cluster = [token]
        for other in scored:
            if other == token or other in consumed:
                continue
            overlaps = []
            for group_value in records_by_group:
                left = set(tokens_by_group[group_value][token])
                right = set(tokens_by_group[group_value][other])
                overlaps.append(len(left & right) / max(1, min(len(left), len(right))))
            if min(overlaps, default=0.0) >= .5:
                cluster.append(other)
        consumed.update(cluster)
        evidence = {
            group_value: tuple(dict.fromkeys(
                record_id
                for item in cluster
                for record_id in tokens_by_group[group_value][item]
            ))
            for group_value in records_by_group
        }
        topics.append(SharedTopic(" / ".join(cluster), evidence))
        if len(topics) == limit:
            break
    return tuple(topics)


def _semantic_topic_prompt(
    query: str,
    records_by_group: dict[str, tuple[Record, ...]],
    start_fields: dict[str, str],
) -> str:
    groups = []
    for group_value, records in records_by_group.items():
        groups.append({
            "group": group_value,
            "records": [
                {
                    "id": record.record_id,
                    "text": _topic_text(record, start_fields[group_value]),
                }
                for record in records
            ],
        })
    return f"""你是集合语义映射器。请基于每组的全部记录，找出真正同时出现在所有组中的研究主题。

要求：
1. 输出2到5个中等粒度的中文主题，合并同义词，但不要把整个学科当作主题。
2. 每个主题必须在每一组中至少引用一条直接相关记录。
3. 只能引用输入中的记录ID，不得修改记录或计数，不得补充外部知识。
4. 只输出JSON：{{"topics":[{{"label":"主题","evidence":{{"组值":["记录ID"]}}}}]}}

原问题：{query}
完整分组记录：{json.dumps(groups, ensure_ascii=False)}
"""


def semantic_shared_topics(
    query: str,
    records_by_group: dict[str, tuple[Record, ...]],
    start_fields: dict[str, str],
    *,
    mapper: Callable[..., str] = call_model,
) -> tuple[SharedTopic, ...]:
    """Map shared themes and reject any topic lacking cross-group evidence."""

    prompt = _semantic_topic_prompt(query, records_by_group, start_fields)
    response = mapper(prompt, json_output=True)
    payload = json.loads(response)
    raw_topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(raw_topics, list):
        raise ValueError("语义映射器缺少 topics")
    valid_ids = {
        group_value: {record.record_id for record in records}
        for group_value, records in records_by_group.items()
    }
    topics = []
    for raw in raw_topics[:5]:
        if not isinstance(raw, dict):
            continue
        label = " ".join(str(raw.get("label", "")).split())
        evidence = raw.get("evidence")
        if not label or not isinstance(evidence, dict):
            continue
        normalized: dict[str, tuple[str, ...]] = {}
        for group_value, allowed in valid_ids.items():
            raw_ids = evidence.get(group_value)
            if not isinstance(raw_ids, list):
                normalized = {}
                break
            ids = tuple(dict.fromkeys(
                item for item in raw_ids if isinstance(item, str) and item in allowed
            ))
            if not ids:
                normalized = {}
                break
            normalized[group_value] = ids
        if normalized:
            topics.append(SharedTopic(label=label, evidence=normalized))
    if not topics:
        raise ValueError("语义映射器没有返回跨组可验证主题")
    return tuple(topics)


def _source_result(
    source: PreparedSource,
    selected: Sequence[Record],
    *,
    filter_field: str,
) -> dict[str, Any]:
    summary = (
        f"完整扫描 {source.coverage.scanned_blocks} 个内容块，"
        f"解析 {source.coverage.parsed_records} 条记录，"
        f"筛选后 {len(selected)} 条。"
    )
    return {
        "title": source.title,
        "url": source.url,
        "document_id": source.document_id,
        "group": source.group_value,
        "snippet": summary,
        "content": summary,
        "coverage": {
            "scan_complete": source.coverage.scan_complete,
            "parsed_records": source.coverage.parsed_records,
            "declared_records": source.coverage.declared_records,
            "record_complete": source.coverage.record_complete,
        },
        "record_ids": [record.record_id for record in selected],
        "records": [
            {
                "record_id": record.record_id,
                "label": " ".join(record.values(source.collection.start_field)),
                "filter_values": list(record.values(filter_field)),
                "provenance": {
                    "block_start": record.provenance.block_start,
                    "block_end": record.provenance.block_end,
                },
            }
            for record in selected
        ],
    }


def execute_collection_plan(
    query: str,
    plan: CollectionPlan,
    sources: Sequence[PreparedSource],
    *,
    topic_mapper: Callable[..., str] | None = call_model,
    debug: bool = False,
) -> RagResponse | None:
    """Execute scan→segment→filter→aggregate→semantic map with coverage gates."""

    if len(sources) != len(plan.group_values):
        return None
    compiled_filter = compile_field_filter(query, sources)
    if compiled_filter is None:
        return None
    if any(source.coverage.record_complete is False for source in sources):
        return None

    kinds = {
        infer_record_kind(source.collection.start_field)
        for source in sources
    }
    kinds.discard("")
    record_kind = next(iter(kinds)) if len(kinds) == 1 else "记录"
    unit_match = re.search(r"(?:多少|几)\s*([个篇项条名位人次])", query)
    count_unit = unit_match.group(1) if unit_match else "条"

    selected_by_group = {
        source.group_value: filter_records(
            source.collection.records,
            field=compiled_filter.field,
            operator=compiled_filter.operator,
            value=compiled_filter.value,
        )
        for source in sources
    }
    counts = {
        group_value: int(aggregate(records, operator="count"))
        for group_value, records in selected_by_group.items()
    }
    start_fields = {
        source.group_value: source.collection.start_field for source in sources
    }

    topics: tuple[SharedTopic, ...] = ()
    topic_mode = "not_requested"
    if plan.needs_intersection:
        if topic_mapper is not None:
            try:
                topics = semantic_shared_topics(
                    query,
                    selected_by_group,
                    start_fields,
                    mapper=topic_mapper,
                )
                topic_mode = "semantic"
            except (RuntimeError, ValueError, json.JSONDecodeError, TypeError):
                topics = ()
        if not topics:
            topics = lexical_shared_topics(selected_by_group, start_fields)
            topic_mode = "lexical_fallback"

    count_text = "，".join(
        f"{group_value}年{count}{count_unit}"
        for group_value, count in counts.items()
    )
    topic_text = ""
    if plan.needs_intersection:
        topic_text = (
            f"；两组共同主题为：{'、'.join(topic.label for topic in topics)}"
            if topics
            else "；主题交集无法在当前语义映射能力下可靠确定"
        )
    answer = (
        f"{compiled_filter.field}字段包含{compiled_filter.value}的{record_kind}中，"
        f"{count_text}{topic_text}。"
    )
    complete = all(
        source.coverage.scan_complete
        and source.coverage.record_complete is not False
        for source in sources
    ) and (not plan.needs_intersection or bool(topics))
    results = [
        _source_result(
            source,
            selected_by_group[source.group_value],
            filter_field=compiled_filter.field,
        )
        for source in sources
    ]
    claims: list[dict[str, Any]] = [
        {
            "statement": f"{group_value}年共{count}条匹配记录",
            "record_ids": [
                record.record_id for record in selected_by_group[group_value]
            ],
        }
        for group_value, count in counts.items()
    ]
    claims.extend({
        "statement": f"共同主题：{topic.label}",
        "record_ids_by_group": topic.evidence,
    } for topic in topics)
    response: dict[str, Any] = {
        "answer": answer,
        "status": "answered" if complete else "partial",
        "complete": complete,
        "requirements": [
            {
                "id": f"count_{group_value}",
                "question": f"计算{group_value}组的匹配记录数量",
                "status": "answered",
                "record_ids": [record.record_id for record in records],
            }
            for group_value, records in selected_by_group.items()
        ] + ([{
            "id": "shared_topics",
            "question": "找出所有组共同出现的主题",
            "status": "answered" if topics else "missing",
            "record_ids": sorted({
                record_id
                for topic in topics
                for ids in topic.evidence.values()
                for record_id in ids
            }),
        }] if plan.needs_intersection else []),
        "claims": claims,
        "sources": results,
        "results": results,
    }
    if debug:
        response["collection_trace"] = {
            "plan": {
                "group_values": plan.group_values,
                "needs_count": plan.needs_count,
                "needs_intersection": plan.needs_intersection,
            },
            "filter": {
                "field": compiled_filter.field,
                "operator": compiled_filter.operator,
                "value": compiled_filter.value,
            },
            "counts": counts,
            "topic_mode": topic_mode,
            "topics": [
                {"label": topic.label, "evidence": topic.evidence}
                for topic in topics
            ],
        }
    return cast(RagResponse, response)


def answer_collection_question(
    storage: Storage,
    query: str,
    hits: Sequence[PassageHit],
    *,
    topic_mapper: Callable[..., str] | None = call_model,
    debug: bool = False,
) -> RagResponse | None:
    """Try the exhaustive collection path; return ``None`` when inapplicable."""

    plan = plan_collection_query(query)
    if plan is None:
        return None
    sources = resolve_sources(storage, hits, plan)
    if not sources:
        return None
    return execute_collection_plan(
        query,
        plan,
        sources,
        topic_mapper=topic_mapper,
        debug=debug,
    )
