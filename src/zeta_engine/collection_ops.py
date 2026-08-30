"""Composable primitives for exhaustive, provenance-preserving collection QA.

The operators in this module deliberately know nothing about papers, authors,
years, or any other application domain.  Domain words are supplied at runtime
by a query plan or inferred from repeated labels in a resource.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from bs4 import BeautifulSoup, Tag

from zeta_engine.tokenizer import text_normalize


_BLOCK_TAGS = (
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr",
    "blockquote", "pre", "figcaption", "dt", "dd",
)
_NESTED_CONTAINER_TAGS = frozenset({"li", "tr"})
_LABEL_RE = re.compile(
    r"^([^:：\n]{1,32})\s*[:：]\s*(.*)$",
    re.DOTALL,
)
_START_SUFFIXES = (
    "题目", "标题", "名称", "姓名", "编号", "项目", "条目", "事项",
)
_KIND_SUFFIXES = (
    "题目", "标题", "名称", "姓名", "编号", "项目", "条目", "事项",
)


@dataclass(frozen=True)
class Provenance:
    """Stable location of one derived value in a source resource."""

    resource_id: str
    block_start: int
    block_end: int


@dataclass(frozen=True)
class Block:
    """One ordered, independently addressable block returned by ``scan``."""

    index: int
    text: str
    tag: str = "text"


@dataclass(frozen=True)
class ScanResult:
    """Complete bounded traversal of one resource."""

    resource_id: str
    title: str
    url: str
    blocks: tuple[Block, ...]
    complete: bool = True


@dataclass(frozen=True)
class ScanPage:
    """One resumable page of a previously bounded scan."""

    resource_id: str
    blocks: tuple[Block, ...]
    next_cursor: int | None
    complete: bool


@dataclass(frozen=True)
class Record:
    """One repeated unit with fields and immutable source provenance."""

    record_id: str
    fields: Mapping[str, tuple[str, ...]]
    provenance: Provenance

    def values(self, field: str) -> tuple[str, ...]:
        return tuple(self.fields.get(normalize_field_name(field), ()))


@dataclass(frozen=True)
class RecordCollection:
    """Records segmented from a scan, including unassigned source blocks."""

    resource_id: str
    start_field: str
    records: tuple[Record, ...]
    unassigned_blocks: tuple[int, ...]
    scanned_blocks: int


@dataclass(frozen=True)
class Coverage:
    """Separate scan and parse completeness measurements."""

    scan_complete: bool
    scanned_blocks: int
    assigned_blocks: int
    parsed_records: int
    declared_records: int | None

    @property
    def parse_ratio(self) -> float:
        if self.scanned_blocks == 0:
            return 1.0
        return self.assigned_blocks / self.scanned_blocks

    @property
    def record_complete(self) -> bool | None:
        if self.declared_records is None:
            return None
        return self.parsed_records == self.declared_records


@dataclass(frozen=True)
class ValidationIssue:
    record_id: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: tuple[ValidationIssue, ...]


_T = TypeVar("_T")
_U = TypeVar("_U")


def normalize_field_name(value: str) -> str:
    """Normalize a runtime field label without imposing a domain schema."""

    return text_normalize(value).strip(" :：")


def normalize_value(value: object) -> str:
    """Normalize a scalar value used by deterministic relational operators."""

    return text_normalize(str(value or ""))


def _tag_text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def scan(
    resource_id: str,
    *,
    title: str,
    url: str,
    text: str,
    content_html: str = "",
) -> ScanResult:
    """Enumerate every meaningful block in one already-acquired resource."""

    raw_blocks: list[tuple[str, str]] = []
    if content_html.strip():
        soup = BeautifulSoup(content_html, "html.parser")
        for tag in soup.find_all(_BLOCK_TAGS):
            if not isinstance(tag, Tag):
                continue
            if tag.name not in _NESTED_CONTAINER_TAGS and any(
                isinstance(parent, Tag) and parent.name in _NESTED_CONTAINER_TAGS
                for parent in tag.parents
            ):
                continue
            value = _tag_text(tag)
            if value:
                raw_blocks.append((tag.name or "html", value))
    if not raw_blocks:
        lines = [" ".join(item.split()) for item in text.splitlines()]
        values = [item for item in lines if item]
        if not values and text.strip():
            values = [" ".join(text.split())]
        raw_blocks = [("text", item) for item in values]

    return ScanResult(
        resource_id=str(resource_id),
        title=" ".join(title.split()),
        url=url.strip(),
        blocks=tuple(
            Block(index=index, text=value, tag=tag)
            for index, (tag, value) in enumerate(raw_blocks)
        ),
    )


def checkpoint(
    scanned: ScanResult,
    *,
    cursor: int = 0,
    limit: int = 100,
) -> ScanPage:
    """Read a resumable page from a complete scan without losing ordering."""

    if cursor < 0:
        raise ValueError("cursor 不能小于 0")
    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    end = min(len(scanned.blocks), cursor + limit)
    complete = end >= len(scanned.blocks)
    return ScanPage(
        resource_id=scanned.resource_id,
        blocks=scanned.blocks[cursor:end],
        next_cursor=None if complete else end,
        complete=complete,
    )


def locate(
    query: str,
    search: Callable[[str, int], Sequence[_T]],
    *,
    constraints: Sequence[Callable[[_T], bool]] = (),
    limit: int = 20,
) -> tuple[_T, ...]:
    """Locate ranked resources and apply caller-defined runtime constraints."""

    if limit <= 0:
        return ()
    candidates = search(query, limit)
    return tuple(
        candidate
        for candidate in candidates
        if all(predicate(candidate) for predicate in constraints)
    )[:limit]


def parse_labeled_value(text: str) -> tuple[str, str] | None:
    """Project a leading ``label: value`` block into a normalized field."""

    match = _LABEL_RE.match(text.strip())
    if not match:
        return None
    label = normalize_field_name(match.group(1))
    value = " ".join(match.group(2).split())
    if not label:
        return None
    return label, value


def infer_start_field(blocks: Sequence[Block], *, min_records: int = 2) -> str:
    """Infer the label that starts a repeated record schema."""

    positions: dict[str, list[int]] = defaultdict(list)
    for block in blocks:
        labeled = parse_labeled_value(block.text)
        if labeled is not None and labeled[1]:
            positions[labeled[0]].append(block.index)
    candidates = {
        label: indexes
        for label, indexes in positions.items()
        if len(indexes) >= min_records
    }
    if not candidates:
        return ""

    maximum = max(len(indexes) for indexes in candidates.values())
    near_maximum = {
        label: indexes
        for label, indexes in candidates.items()
        if len(indexes) >= max(min_records, maximum - 1)
    }

    def key(item: tuple[str, list[int]]) -> tuple[int, int, int, str]:
        label, indexes = item
        likely_start = int(any(label.endswith(suffix) for suffix in _START_SUFFIXES))
        return (-likely_start, -len(indexes), indexes[0], label)

    return min(near_maximum.items(), key=key)[0]


def segment(
    scanned: ScanResult,
    *,
    start_field: str | None = None,
    min_records: int = 2,
) -> RecordCollection:
    """Split a complete scan into repeated labeled records."""

    normalized_start = normalize_field_name(start_field or "")
    if not normalized_start:
        normalized_start = infer_start_field(
            scanned.blocks,
            min_records=min_records,
        )
    if not normalized_start:
        return RecordCollection(
            resource_id=scanned.resource_id,
            start_field="",
            records=(),
            unassigned_blocks=tuple(block.index for block in scanned.blocks),
            scanned_blocks=len(scanned.blocks),
        )

    starts = [
        index
        for index, block in enumerate(scanned.blocks)
        if (
            (labeled := parse_labeled_value(block.text)) is not None
            and labeled[0] == normalized_start
            and bool(labeled[1])
        )
    ]
    records: list[Record] = []
    assigned: set[int] = set()
    for ordinal, start in enumerate(starts, start=1):
        end = starts[ordinal] if ordinal < len(starts) else len(scanned.blocks)
        fields: dict[str, list[str]] = defaultdict(list)
        last_field = ""
        for block in scanned.blocks[start:end]:
            labeled = parse_labeled_value(block.text)
            if labeled is not None:
                label, value = labeled
                if value:
                    fields[label].append(value)
                    last_field = label
                    assigned.add(block.index)
                continue
            # Preserve continuation blocks inside an already-started record.
            if last_field and block.text:
                fields[last_field][-1] = f"{fields[last_field][-1]} {block.text}"
                assigned.add(block.index)
        if not fields.get(normalized_start):
            continue
        records.append(Record(
            record_id=f"{scanned.resource_id}:record:{ordinal}",
            fields={key: tuple(values) for key, values in fields.items()},
            provenance=Provenance(
                resource_id=scanned.resource_id,
                block_start=scanned.blocks[start].index,
                block_end=scanned.blocks[end - 1].index,
            ),
        ))

    return RecordCollection(
        resource_id=scanned.resource_id,
        start_field=normalized_start,
        records=tuple(records),
        unassigned_blocks=tuple(
            block.index for block in scanned.blocks if block.index not in assigned
        ),
        scanned_blocks=len(scanned.blocks),
    )


def project(record: Record, fields: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Project selected runtime fields from one record."""

    return {field: record.values(field) for field in fields}


def filter_records(
    records: Iterable[Record],
    *,
    field: str,
    operator: str,
    value: object,
) -> tuple[Record, ...]:
    """Apply a deterministic predicate to a collection of records."""

    expected = normalize_value(value)
    if operator not in {"equals", "contains", "exists", "regex"}:
        raise ValueError(f"不支持的过滤操作: {operator}")
    pattern = re.compile(str(value), re.IGNORECASE) if operator == "regex" else None
    selected = []
    for record in records:
        values = tuple(normalize_value(item) for item in record.values(field))
        matched = (
            bool(values)
            if operator == "exists"
            else any(item == expected for item in values)
            if operator == "equals"
            else any(expected in item for item in values)
            if operator == "contains"
            else any(pattern.search(item) is not None for item in values)
        )
        if matched:
            selected.append(record)
    return tuple(selected)


def distinct(records: Iterable[Record], *, field: str) -> tuple[Record, ...]:
    """Keep the first record for each normalized field value."""

    seen: set[tuple[str, ...]] = set()
    output = []
    for record in records:
        key = tuple(normalize_value(value) for value in record.values(field))
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return tuple(output)


def sort_records(
    records: Iterable[Record],
    *,
    key: Callable[[Record], object],
    reverse: bool = False,
) -> tuple[Record, ...]:
    """Return a stable ordering using a caller-supplied runtime key."""

    return tuple(sorted(records, key=key, reverse=reverse))


def group(
    records: Iterable[Record],
    key: Callable[[Record], str],
) -> dict[str, tuple[Record, ...]]:
    """Group records by a caller-supplied, domain-independent key."""

    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[normalize_value(key(record))].append(record)
    return {name: tuple(values) for name, values in grouped.items()}


def join(
    left: Iterable[_T],
    right: Iterable[_U],
    *,
    left_key: Callable[[_T], object],
    right_key: Callable[[_U], object],
    how: str = "inner",
) -> tuple[tuple[_T, _U | None], ...]:
    """Perform a stable inner or left join over normalized runtime keys."""

    if how not in {"inner", "left"}:
        raise ValueError(f"不支持的连接方式: {how}")
    right_index: dict[str, list[_U]] = defaultdict(list)
    for item in right:
        right_index[normalize_value(right_key(item))].append(item)
    output: list[tuple[_T, _U | None]] = []
    for item in left:
        matches = right_index.get(normalize_value(left_key(item)), [])
        if matches:
            output.extend((item, match) for match in matches)
        elif how == "left":
            output.append((item, None))
    return tuple(output)


def aggregate(
    values: Iterable[Any],
    *,
    operator: str,
) -> int | float | Any:
    """Apply one deterministic scalar aggregation."""

    items = list(values)
    if operator == "count":
        return len(items)
    if operator == "count_distinct":
        return len({normalize_value(item) for item in items})
    if operator == "sum":
        return sum(items)
    if operator == "min":
        return min(items)
    if operator == "max":
        return max(items)
    if operator == "average":
        if not items:
            raise ValueError("空集合不能计算平均值")
        return sum(items) / len(items)
    raise ValueError(f"不支持的聚合操作: {operator}")


def calculate(left: float, *, operator: str, right: float) -> float:
    """Evaluate one allow-listed arithmetic operation without expression eval."""

    if operator == "add":
        return left + right
    if operator == "subtract":
        return left - right
    if operator == "multiply":
        return left * right
    if operator in {"divide", "ratio"}:
        if right == 0:
            raise ValueError("除数不能为 0")
        return left / right
    raise ValueError(f"不支持的计算操作: {operator}")


def semantic_map(
    items: Iterable[_T],
    *,
    item_id: Callable[[_T], str],
    mapper: Callable[[_T], Iterable[object]],
) -> dict[str, tuple[str, ...]]:
    """Map items to normalized semantic labels using an injected mapper."""

    assignments: dict[str, tuple[str, ...]] = {}
    for item in items:
        identifier = str(item_id(item))
        if not identifier or identifier in assignments:
            raise ValueError("semantic_map 要求唯一且非空的 item_id")
        labels = tuple(dict.fromkeys(
            normalize_value(label) for label in mapper(item) if normalize_value(label)
        ))
        assignments[identifier] = labels
    return assignments


def set_op(
    left: Iterable[object],
    right: Iterable[object],
    *,
    operator: str,
) -> tuple[str, ...]:
    """Apply a normalized set operation while returning stable ordering."""

    left_values = list(dict.fromkeys(normalize_value(item) for item in left))
    right_values = list(dict.fromkeys(normalize_value(item) for item in right))
    left_set, right_set = set(left_values), set(right_values)
    if operator == "intersection":
        selected = left_set & right_set
        order = left_values
    elif operator == "union":
        selected = left_set | right_set
        order = [*left_values, *right_values]
    elif operator == "difference":
        selected = left_set - right_set
        order = left_values
    else:
        raise ValueError(f"不支持的集合操作: {operator}")
    return tuple(item for item in dict.fromkeys(order) if item in selected)


def infer_record_kind(start_field: str) -> str:
    """Infer a display noun from a runtime start label."""

    value = normalize_field_name(start_field)
    for suffix in _KIND_SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[:-len(suffix)]
    return ""


def declared_record_count(scanned: ScanResult, *, kind: str) -> int | None:
    """Read an explicitly stated collection size from source blocks."""

    kind = normalize_value(kind)
    if not kind:
        return None
    pattern = re.compile(rf"共\s*(\d+)\s*[个篇名项条位]?\s*{re.escape(kind)}")
    for block in scanned.blocks:
        if match := pattern.search(normalize_value(block.text)):
            return int(match.group(1))
    return None


def measure(
    scanned: ScanResult,
    collection: RecordCollection,
    *,
    declared_records: int | None = None,
) -> Coverage:
    """Measure coverage without treating inference confidence as completeness."""

    return Coverage(
        scan_complete=scanned.complete,
        scanned_blocks=len(scanned.blocks),
        assigned_blocks=len(scanned.blocks) - len(collection.unassigned_blocks),
        parsed_records=len(collection.records),
        declared_records=declared_records,
    )


def validate(
    records: Iterable[Record],
    *,
    required_fields: Sequence[str] = (),
    validators: Mapping[str, Callable[[tuple[str, ...]], bool]] | None = None,
) -> ValidationReport:
    """Validate a runtime schema while preserving per-record failure details."""

    normalized_required = tuple(
        normalize_field_name(field) for field in required_fields
    )
    normalized_validators = {
        normalize_field_name(field): validator
        for field, validator in (validators or {}).items()
    }
    issues: list[ValidationIssue] = []
    for record in records:
        for field in normalized_required:
            if not record.values(field):
                issues.append(ValidationIssue(
                    record_id=record.record_id,
                    field=field,
                    message="缺少必填字段",
                ))
        for field, validator in normalized_validators.items():
            values = record.values(field)
            if values and not validator(values):
                issues.append(ValidationIssue(
                    record_id=record.record_id,
                    field=field,
                    message="字段值未通过校验",
                ))
    return ValidationReport(valid=not issues, issues=tuple(issues))


def field_frequencies(records: Iterable[Record]) -> Counter[str]:
    """Return schema coverage counts for runtime planning and validation."""

    frequencies: Counter[str] = Counter()
    for record in records:
        frequencies.update(record.fields.keys())
    return frequencies
