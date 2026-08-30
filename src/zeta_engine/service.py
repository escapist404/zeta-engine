import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from zeta_engine.collection_rag import (
    answer_collection_question,
)
from zeta_engine.dense import DEFAULT_INDEX_DIR, PassageHit, search_dense, warmup_dense
from zeta_engine.rag import AGENT_MAX_CYCLES, agentic_rag_answer
from zeta_engine.rag_types import RagResponse
from zeta_engine.search import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid_with_snippets,
    search_reranked_passages,
    search_phrase,
    search_reranked,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import text_normalize, tokenize_with_positions

_PUBLISHED_DATE_RE = re.compile(
    r"(?:日期|发布时间|更新时间)\s*[:：]?\s*"
    r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})日?"
)
_URL_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})")
RAG_EVIDENCE_MAX_CHARS = 12_000
_STRUCTURAL_PARENT_TAGS = frozenset({"table", "ul", "ol", "dl", "section"})
_CONTENT_BLOCK_TAGS = (
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr",
    "blockquote", "pre", "figcaption", "dt", "dd",
)
_COMPLETE_COLLECTION_QUERY_RE = re.compile(
    r"名单|名录|清单|人数|多少\s*[人名个项条]|几\s*[人名个项条]|数量|总数|计数"
)
_COLLECTION_TITLE_RE = re.compile(r"名单|名录|清单")
_COLLECTION_MARKER_RE = re.compile(r"名单|名录|清单")


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


def _complete_table_values(content_html: str) -> tuple[str, ...]:
    """Return data cells from all tables, excluding structural header rows."""

    if not content_html.strip():
        return ()
    soup = BeautifulSoup(content_html, "html.parser")
    return tuple(
        value
        for table in soup.find_all("table")
        if isinstance(table, Tag)
        for value in _table_values(table)
    )


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
            bound_facts = []
            bound_relations: list[dict[str, object]] = []
            bound_patterns = (
                ("下界", r"(?:不少于|至少|不低于)\s*(\d+(?:\.\d+)?)\s*(?:人|名|个|项|条)"),
                ("上界", r"(?:不超过|至多|不高于)\s*(\d+(?:\.\d+)?)\s*(?:人|名|个|项|条)"),
            )
            for bound_name, pattern in bound_patterns:
                for match in re.finditer(pattern, context):
                    bound = float(match.group(1))
                    if bound < 0 or bound > len(values):
                        continue
                    ratio = bound / len(values)
                    rendered_bound = (
                        str(int(bound)) if bound.is_integer() else str(bound)
                    )
                    bound_facts.append(
                        f"同章节数量{bound_name}为{rendered_bound}项；"
                        f"相对于完整表格{len(values)}项，对应比例{bound_name}为"
                        f"{ratio:.6f}（{ratio * 100:.2f}%）"
                    )
                    bound_relations.append({
                        "kind": "ratio_bound",
                        "direction": "lower" if bound_name == "下界" else "upper",
                        "numerator": bound,
                        "denominator": len(values),
                        "ratio": ratio,
                    })
            derived = (
                f"确定性派生：{'；'.join(dict.fromkeys(bound_facts))}。"
                if bound_facts
                else ""
            )
            content = (
                f"确定性工具已完整扫描该表格。章节：{label}。"
                f"完整表格集合共{len(values)}项。表格前文：{context or '（无）'}。"
                f"{derived}"
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
            if bound_relations:
                result["derived_relations"] = json.dumps(
                    bound_relations,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ranked.append((
                -score,
                rank,
                table_index,
                result,
            ))
        if document_values and _collection_title_matches_query(query, title):
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
    requested_years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", query))
    if len(requested_years) >= 2:
        requested_aggregates = [
            item for item in ranked
            if item[3].get("evidence_scope") == "complete_document_tables"
            and any(year in item[3].get("title", "") for year in requested_years)
        ]
        requested_ids = {id(item) for item in requested_aggregates}
        ranked = requested_aggregates + [
            item for item in ranked if id(item) not in requested_ids
        ]
    return [item[3] for item in ranked[:limit]]


def _collection_response_results(
    response: dict[str, object],
    query: str,
) -> list[dict[str, str]]:
    """Convert a deterministic collection execution into agent evidence."""

    sources = [
        source for source in response.get("sources", [])
        if isinstance(source, dict)
    ]
    trace = response.get("collection_trace", {})
    summary = (
        "确定性完整集合工具执行完成。"
        f"结果：{response.get('answer', '')}。"
        f"执行明细：{json.dumps(trace, ensure_ascii=False)}"
    )
    results = [{
        "title": f"完整集合分析 · {query}",
        "url": str(sources[0].get("url", "")) if sources else "",
        "snippet": summary[:500],
        "content": summary,
        "passage_id": f"collection-analysis:{query}",
        "chunk_index": "collection-analysis",
        "evidence_scope": "complete_collection_analysis",
        "collection_complete": "true",
        "collection_trace": json.dumps(trace, ensure_ascii=False),
    }]
    for source in sources:
        records = source.get("records", [])
        labels = [
            str(record.get("label", ""))
            for record in records
            if isinstance(record, dict) and record.get("label")
        ]
        coverage = source.get("coverage", {})
        content = (
            f"确定性工具完整扫描分组 {source.get('group', '')}。"
            f"覆盖信息：{json.dumps(coverage, ensure_ascii=False)}。"
            f"筛选后记录数：{len(records)}。匹配记录：{'；'.join(labels)}"
        )
        results.append({
            "title": str(source.get("title", "完整集合来源")),
            "url": str(source.get("url", "")),
            "snippet": content[:500],
            "content": content,
            "passage_id": f"collection-source:{source.get('document_id', '')}",
            "chunk_index": "collection-source",
            "document_id": str(source.get("document_id", "")),
            "evidence_scope": "complete_collection_source",
            "collection_complete": "true",
        })
    return results


def _collection_title_matches_query(query: str, title: str) -> bool:
    """Reject a different named collection (for example awards vs campers)."""

    if not _COLLECTION_TITLE_RE.search(title):
        return False

    def qualifiers(text: str) -> set[tuple[str, str]]:
        found = set()
        for match in _COLLECTION_MARKER_RE.finditer(text):
            prefix = re.search(r"[\u3400-\u9fff]{2,}$", text[:match.start()])
            if prefix is not None:
                found.add((match.group(), prefix.group()[-2:]))
        return found

    requested = qualifiers(query)
    if not requested:
        return True
    return bool(requested & qualifiers(title))


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


def _query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(
        term
        for term, _offset in tokenize_with_positions(
            text_normalize(query),
            mode="search",
        )
    ))


def _lexical_score(text: str, terms: list[str]) -> int:
    return sum(len(term) * text.count(term) for term in terms)


def _query_snippet(query: str, text: str, limit: int = 240) -> str:
    text = text_normalize(text)
    if len(text) <= limit:
        return text

    terms = _query_terms(query)
    starts = {0}
    for term in terms:
        offset = 0
        while (position := text.find(term, offset)) >= 0:
            starts.add(max(0, min(position - limit // 3, len(text) - limit)))
            offset = position + len(term)

    def score(start: int) -> tuple[int, int]:
        window = text[start:start + limit]
        return _lexical_score(window, terms), -start

    start = max(starts, key=score)
    return text[start:start + limit]


def _published_at(url: str, text: str) -> str:
    """Extract an ISO publication date from page text or its URL."""

    match = _PUBLISHED_DATE_RE.search(text) or _URL_DATE_RE.search(url)
    if not match:
        return ""
    year, month, day = map(int, match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def search_documents(
    document_db: str | Path,
    index_db: str | Path,
    query: str,
    limit: int = 20,
    *,
    ranking: str = "hybrid",
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    content_limit: int = 0,
) -> list[dict[str, str]]:
    if content_limit < 0:
        raise ValueError("content_limit 不能小于 0")

    with Storage(
        document_db=document_db,
        index_db=index_db if ranking != "dense" else None,
    ) as storage:
        assert storage.documents is not None
        snippets = {}
        if ranking == "dense":
            hits = search_dense(
                query,
                dense_index,
                limit=limit,
                device=device,
            )
            document_ids = [hit.document_id for hit in hits]
            snippets = {hit.document_id: hit.snippet for hit in hits}
        elif ranking == "phrase":
            document_ids = search_phrase(storage, query)[:limit]
        elif ranking == "bm25f":
            document_ids = search_bm25f(storage, query)[:limit]
        elif ranking == "hybrid":
            document_ids, snippets = search_hybrid_with_snippets(
                storage,
                query,
                dense_index,
                limit=limit,
                alpha=alpha,
                device=device,
            )
        elif ranking == "rerank":
            document_ids = search_reranked(
                storage,
                query,
                dense_index,
                reranker_model=reranker_model,
                limit=limit,
                candidate_limit=rerank_candidates,
                batch_size=reranker_batch_size,
                alpha=alpha,
                device=device,
            )
        else:
            raise ValueError(f"不支持的排名方式: {ranking}")

        results = []
        terms = _query_terms(query)
        for document_id in document_ids:
            document = storage.documents.get(document_id)
            if document is None:
                continue
            url, title, text, _fetched_at = document
            content_html = storage.documents.get_content_html(document_id)
            dense_text = text_normalize(snippets.get(document_id, ""))
            dense_snippet = _query_snippet(query, dense_text)
            lexical_snippet = _query_snippet(query, text)
            snippet = max(
                (item for item in (dense_snippet, lexical_snippet) if item),
                key=lambda item: _lexical_score(item, terms),
                default="",
            )
            result = {
                "title": text_normalize(title) or url,
                "url": url,
                "snippet": snippet[:240],
            }
            if published_at := _published_at(url, text):
                result["published_at"] = published_at
            if content_limit:
                lexical_content = _query_snippet(query, text, content_limit)
                dense_content = _query_snippet(query, dense_text, content_limit)
                result["content"] = max(
                    (item for item in (lexical_content, dense_content) if item),
                    key=lambda item: _lexical_score(item, terms),
                    default="",
                )
                if content_html and len(text_normalize(text)) <= content_limit:
                    result["structured_content"] = content_html
            results.append(result)
        return results


def search_passages(
    document_db: str | Path,
    query: str,
    limit: int = 10,
    *,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> list[dict[str, str]]:
    """Return exact, independently citable passages for RAG."""

    hits = search_reranked_passages(
        query,
        dense_index,
        reranker_model=reranker_model,
        limit=limit,
        candidate_limit=rerank_candidates,
        batch_size=reranker_batch_size,
        alpha=alpha,
        device=device,
    )
    with Storage(document_db=document_db) as storage:
        assert storage.documents is not None
        results = []
        emitted_complete_collections: set[int] = set()
        for hit in hits:
            document = storage.documents.get(hit.document_id)
            if document is None:
                continue
            url, document_title, document_text, _fetched_at = document
            title = hit.title or text_normalize(document_title) or url
            passage = text_normalize(hit.text)
            if not passage:
                continue
            content_html = storage.documents.get_content_html(hit.document_id)
            complete_values = ()
            if (
                _COMPLETE_COLLECTION_QUERY_RE.search(query)
                and _collection_title_matches_query(query, title)
            ):
                complete_values = _complete_table_values(content_html)
            if complete_values:
                if hit.document_id in emitted_complete_collections:
                    continue
                emitted_complete_collections.add(hit.document_id)
                complete_text = text_normalize(
                    f"完整表格集合共{len(complete_values)}项 "
                    + " ".join(complete_values)
                )
                result = {
                    "title": title,
                    "url": url,
                    "snippet": _query_snippet(query, complete_text),
                    "content": complete_text,
                    "passage_id": f"{hit.document_id}:complete-tables",
                    "chunk_index": "complete-tables",
                    "candidate_content": passage,
                    "evidence_scope": "complete_tables",
                    "collection_size": str(len(complete_values)),
                    "collection_complete": "true",
                }
                if published_at := _published_at(url, document_text):
                    result["published_at"] = published_at
                results.append(result)
                continue
            evidence = _acquire_structured_evidence(
                content_html,
                passage,
                query=query,
            )
            result = {
                "title": title,
                "url": url,
                "snippet": _query_snippet(query, evidence.text),
                "content": evidence.text,
                "passage_id": hit.passage_id,
                "chunk_index": str(hit.chunk_index),
                "candidate_content": passage,
                "evidence_scope": evidence.scope,
            }
            if evidence.heading_path:
                result["heading_path"] = evidence.heading_path
            if evidence.structured_content:
                result["structured_content"] = evidence.structured_content
            if published_at := _published_at(url, document_text):
                result["published_at"] = published_at
            results.append(result)
        return results


def answer_question(
    document_db: str | Path,
    index_db: str | Path,
    query: str,
    top_k: int = 8,
    *,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    max_cycles: int = AGENT_MAX_CYCLES,
    debug: bool = False,
) -> RagResponse:
    if alpha > 0.:
        warmup_dense(dense_index, device=device)

    def run_collection(collection_query: str) -> list[dict[str, str]]:
        canonical_query = query
        collection_hits = search_reranked_passages(
            canonical_query,
            dense_index,
            reranker_model=reranker_model,
            limit=max(12, top_k * 3),
            candidate_limit=rerank_candidates,
            batch_size=reranker_batch_size,
            alpha=alpha,
            device=device,
        )
        with Storage(document_db=document_db) as storage:
            collection_response = answer_collection_question(
                storage,
                canonical_query,
                collection_hits,
                topic_mapper=None,
                debug=True,
            )
            if collection_response is not None:
                return _collection_response_results(
                    collection_response,
                    canonical_query,
                )
            return _table_section_results(
                storage,
                collection_hits,
                canonical_query,
            )

    def search_fn(search_query: str, limit: int) -> list[dict[str, str]]:
        return search_passages(
            document_db,
            search_query,
            limit,
            dense_index=dense_index,
            reranker_model=reranker_model,
            rerank_candidates=rerank_candidates,
            reranker_batch_size=reranker_batch_size,
            device=device,
            alpha=alpha,
        )

    return agentic_rag_answer(
        query,
        search_fn,
        top_k=top_k,
        max_cycles=max_cycles,
        debug=debug,
        collection_fn=run_collection,
    )
