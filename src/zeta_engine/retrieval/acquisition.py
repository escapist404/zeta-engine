"""Recover semantic HTML blocks and complete tables from retrieved documents."""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from zeta_engine.retrieval.dense import PassageHit
from zeta_engine.infrastructure.storage import Storage
from zeta_engine.infrastructure.tokenizer import text_normalize, tokenize_with_positions

RAG_EVIDENCE_MAX_CHARS = 12_000
_STRUCTURAL_PARENT_TAGS = frozenset({"table", "ul", "ol", "dl", "section"})
_CONTENT_BLOCK_TAGS = (
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr",
    "blockquote", "pre", "figcaption", "dt", "dd",
)
_COLLECTION_TITLE_RE = re.compile(r"名单|名录|清单")


@dataclass(frozen=True)
class _AcquiredEvidence:
    """Context recovered from one ranked candidate passage."""

    text: str
    structured_content: str = ""
    scope: str = "passage"
    heading_path: str = ""


def _block_text(tag: Tag) -> str:
    return text_normalize(tag.get_text(" ", strip=True))


def _candidate_blocks(soup: BeautifulSoup) -> list[Tag]:
    """Return semantic blocks without nested list/table duplicates."""

    blocks = []
    for tag in soup.find_all(_CONTENT_BLOCK_TAGS):
        if not isinstance(tag, Tag) or not _block_text(tag):
            continue
        if tag.name in {"p", "blockquote", "pre", "dt", "dd"} and any(
            isinstance(parent, Tag) and parent.name in {"li", "tr"}
            for parent in tag.parents
        ):
            continue
        blocks.append(tag)
    return blocks


def _table_values(table: Tag) -> tuple[str, ...]:
    """Return independently countable data cells from one complete table."""

    values = []
    for row in table.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        cells = [
            cell for cell in row.find_all(["td", "th"], recursive=False)
            if isinstance(cell, Tag)
        ]
        if not cells:
            continue
        if len(cells) == 1 and (
            cells[0].name == "th" or cells[0].has_attr("colspan")
        ):
            continue
        for cell in cells:
            if cell.name == "th":
                continue
            value = _block_text(cell)
            if value and value not in {"-", "—", "–", "/"}:
                values.append(value)
    return tuple(values)


def _table_section_results(
    storage: Storage,
    hits: list[PassageHit],
    query: str,
    *,
    limit: int = 8,
) -> list[dict[str, str]]:
    """Expose complete tables with their nearest source section preserved."""

    assert storage.documents is not None
    query_terms = {
        term
        for term, _offset in tokenize_with_positions(query, mode="search")
        if len(term) >= 2
    }
    ranked: list[tuple[int, int, int, dict[str, str]]] = []
    seen_documents: set[int] = set()
    for rank, hit in enumerate(hits, start=1):
        if hit.document_id in seen_documents:
            continue
        seen_documents.add(hit.document_id)
        document = storage.documents.get(hit.document_id)
        if document is None:
            continue
        url, title, _text, _fetched_at = document
        content_html = storage.documents.get_content_html(hit.document_id)
        if not content_html.strip():
            continue
        soup = BeautifulSoup(content_html, "html.parser")
        document_values: list[str] = []
        for table_index, table in enumerate(soup.find_all("table"), start=1):
            if not isinstance(table, Tag):
                continue
            values = _table_values(table)
            if not values:
                continue
            document_values.extend(values)
            preceding = []
            for tag in table.find_all_previous(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p"],
                limit=12,
            ):
                if not isinstance(tag, Tag) or tag.find_parent("table") is not None:
                    continue
                value = _block_text(tag)
                if value and value not in preceding:
                    preceding.append(value)
                if len(preceding) == 4:
                    break
            label = next(
                (
                    value for value in preceding
                    if _COLLECTION_TITLE_RE.search(value)
                    or re.match(r"^[一二三四五六七八九十]+、", value)
                ),
                preceding[0] if preceding else f"表格{table_index}",
            )
            context = " ".join(reversed(preceding))
            haystack = f"{title} {label} {context}"
            score = sum(len(term) for term in query_terms if term in haystack)
            content = (
                f"确定性工具已完整扫描该表格。章节：{label}。"
                f"完整表格集合共{len(values)}项。表格前文：{context or '（无）'}。"
                f"完整项目：{' '.join(values)}"
            )
            result = {
                "title": f"{title} · {label}",
                "url": url,
                "snippet": content[:500],
                "content": content,
                "passage_id": f"collection-table:{hit.document_id}:{table_index}",
                "chunk_index": f"collection-table:{table_index}",
                "document_id": str(hit.document_id),
                "evidence_scope": "complete_table_section",
                "heading_path": label,
                "collection_size": str(len(values)),
                "collection_complete": "true",
            }
            ranked.append((
                -score,
                rank,
                table_index,
                result,
            ))
        if document_values:
            aggregate_content = (
                "确定性工具已完整扫描该文档的全部表格。"
                f"文档级完整集合共{len(document_values)}项。"
            )
            aggregate_score = sum(
                len(term) for term in query_terms if term in title
            )
            ranked.append((
                -aggregate_score,
                rank,
                0,
                {
                    "title": f"{title} · 全部表格",
                    "url": url,
                    "snippet": aggregate_content,
                    "content": aggregate_content,
                    "passage_id": f"collection-document:{hit.document_id}",
                    "chunk_index": "collection-document",
                    "document_id": str(hit.document_id),
                    "evidence_scope": "complete_document_tables",
                    "heading_path": "全部表格",
                    "collection_size": str(len(document_values)),
                    "collection_complete": "true",
                },
            ))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked[:limit]]


def _anchor_score(anchor: str, block: str) -> tuple[int, int, int]:
    """Score how confidently a semantic block belongs to a hit passage."""

    if not anchor or not block:
        return 0, 0, 0
    if block in anchor:
        return 3, len(block), 0
    if anchor in block:
        return 2, len(anchor), -len(block)
    terms = {
        term
        for term, _offset in tokenize_with_positions(anchor, mode="search")
        if len(term) >= 2
    }
    overlap = sum(len(term) for term in terms if term in block)
    return (1 if overlap >= 4 else 0), overlap, -len(block)


def _query_block_score(query: str, block: str) -> tuple[int, int]:
    """Score a structural block against the current information need."""

    if not query or not block:
        return 0, 0
    terms = {
        term
        for term, _offset in tokenize_with_positions(query, mode="search")
        if len(term) >= 2 or term.isdigit()
    }
    matched = [term for term in terms if term in block]
    return sum(len(term) * block.count(term) for term in matched), len(matched)


def _heading_path(blocks: list[Tag], stop: int) -> str:
    headings: dict[int, str] = {}
    for block in blocks[:stop]:
        if not re.fullmatch(r"h[1-6]", block.name or ""):
            continue
        level = int(block.name[1])
        headings = {
            existing_level: value
            for existing_level, value in headings.items()
            if existing_level < level
        }
        headings[level] = _block_text(block)
    return " > ".join(headings[level] for level in sorted(headings))


def _acquire_structured_evidence(
    content_html: str,
    passage: str,
    *,
    query: str = "",
    max_chars: int = RAG_EVIDENCE_MAX_CHARS,
) -> _AcquiredEvidence:
    """Expand a precise passage to a bounded semantic parent or block window."""

    anchor = text_normalize(passage)
    if not content_html.strip() or not anchor or max_chars <= 0:
        return _AcquiredEvidence(anchor)

    soup = BeautifulSoup(content_html, "html.parser")
    blocks = _candidate_blocks(soup)
    if not blocks:
        return _AcquiredEvidence(anchor)

    scored = [
        (_anchor_score(anchor, _block_text(block)), index)
        for index, block in enumerate(blocks)
    ]
    best_score, best_index = max(scored, default=((0, 0, 0), -1))
    if best_index < 0 or best_score[0] == 0:
        return _AcquiredEvidence(anchor)

    best = blocks[best_index]
    parent = next(
        (
            item
            for item in best.parents
            if isinstance(item, Tag) and item.name in _STRUCTURAL_PARENT_TAGS
        ),
        None,
    )
    heading_path = _heading_path(blocks, best_index)
    structural_parents: list[Tag] = []
    for score, index in scored:
        if score[0] < 2:
            continue
        candidate_parent = next(
            (
                item
                for item in blocks[index].parents
                if isinstance(item, Tag)
                and item.name in _STRUCTURAL_PARENT_TAGS
            ),
            None,
        )
        if (
            candidate_parent is not None
            and candidate_parent not in structural_parents
        ):
            structural_parents.append(candidate_parent)
    if structural_parents:
        parent = max(
            structural_parents,
            key=lambda item: (
                _query_block_score(query, _block_text(item)),
                _anchor_score(anchor, _block_text(item)),
            ),
        )
    if parent is not None:
        parent_text = _block_text(parent)
        if (
            (anchor in parent_text or parent_text in anchor)
            and len(parent_text) <= max_chars
        ):
            parent_indexes = [
                index
                for index, block in enumerate(blocks)
                if block is parent or parent in block.parents
            ]
            if parent_indexes:
                first_parent_index = min(parent_indexes)
                heading_stop = first_parent_index
                if re.fullmatch(
                    r"h[1-6]", blocks[first_parent_index].name or ""
                ):
                    heading_stop += 1
                heading_path = _heading_path(blocks, heading_stop)
            return _AcquiredEvidence(
                text=parent_text,
                structured_content=str(parent),
                scope=parent.name or "section",
                heading_path=heading_path,
            )

    matching = [
        index
        for score, index in scored
        if score[0] >= 2 or (score[0] == 1 and score[1] >= best_score[1])
    ]
    first_match = min(matching, default=best_index)
    last_match = max(matching, default=best_index)
    start = max(0, first_match - 1)
    if re.fullmatch(r"h[1-6]", blocks[start].name or ""):
        start += 1
    end = min(len(blocks), last_match + 2)
    if (
        last_match + 1 < len(blocks)
        and re.fullmatch(r"h[1-6]", blocks[last_match + 1].name or "")
    ):
        end = last_match + 1
    selected = blocks[start:end]
    expanded_text = text_normalize(
        " ".join(_block_text(block) for block in selected)
    )
    if anchor not in expanded_text or len(expanded_text) > max_chars:
        return _AcquiredEvidence(anchor, heading_path=heading_path)
    return _AcquiredEvidence(
        text=expanded_text,
        structured_content="".join(str(block) for block in selected),
        scope="block_window",
        heading_path=heading_path,
    )
