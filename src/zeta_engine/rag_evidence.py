"""Evidence models, storage, deduplication, and context rendering."""

from dataclasses import dataclass, field
from hashlib import sha256

from zeta_engine.rag_config import (
    AGENT_MAX_CHUNKS_PER_URL,
    AGENT_MAX_RESULTS,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MAX_CONTEXT_TOKENS,
)
from zeta_engine.tokenizer import text_normalize


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _normalize_text(value: object) -> str:
    return text_normalize(str(value or ""))


def _result_text(result: dict[str, str]) -> str:
    primary = _normalize_text(result.get("structured_content")) or _normalize_text(
        result.get("content")
    ) or _normalize_text(
        result.get("snippet")
    )
    candidate = _normalize_text(result.get("candidate_content"))
    if not candidate or candidate == primary:
        return primary
    return "\n".join(filter(None, (primary, candidate)))


def _grounding_text(evidence: "Evidence") -> str:
    """Use tag-free content for deterministic input grounding when available."""

    plain_text = _normalize_text(evidence.result.get("content")) or evidence.text
    candidate = _normalize_text(evidence.result.get("candidate_content"))
    if candidate and candidate != plain_text:
        plain_text = " ".join(filter(None, (plain_text, candidate)))
    return " ".join(filter(None, (
        evidence.evidence_id,
        evidence.title,
        _clean(evidence.result.get("heading_path")),
        evidence.url,
        evidence.published_at,
        plain_text,
    )))


def _estimate_tokens(text: str) -> int:
    """Conservatively estimate mixed Chinese/ASCII tokens without a dependency."""

    wide = sum(ord(character) > 127 for character in text)
    return wide + (len(text) - wide + 3) // 4 + 4


@dataclass
class Evidence:
    """One immutable retrieved passage with stable provenance."""

    evidence_id: str
    title: str
    url: str
    published_at: str
    text: str
    result: dict[str, str]
    retrieved_by: list[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: dict[str, str], query: str = "") -> "Evidence":
        title = _normalize_text(result.get("title"))
        url = _clean(result.get("url"))
        published_at = _clean(result.get("published_at"))
        text = _result_text(result)
        passage_id = _clean(result.get("passage_id"))
        digest = sha256(
            f"{url}\0{passage_id}\0{text}".encode()
        ).hexdigest()[:16]
        return cls(
            evidence_id=f"ev_{digest}",
            title=title,
            url=url,
            published_at=published_at,
            text=text,
            result=dict(result),
            retrieved_by=[query] if query else [],
        )

    def block(self, number: int) -> str:
        date_line = f"发布日期：{self.published_at}\n" if self.published_at else ""
        passage_id = _clean(self.result.get("passage_id"))
        passage_line = f"Passage ID：{passage_id}\n" if passage_id else ""
        heading_path = _clean(self.result.get("heading_path"))
        heading_line = f"章节：{heading_path}\n" if heading_path else ""
        return (
            f"[文档{number}]\n"
            f"相关性优先级：{number}（数字越小越相关）\n"
            f"证据ID：{self.evidence_id}\n"
            f"标题：{self.title}\n"
            f"URL：{self.url}\n"
            f"{date_line}"
            f"{passage_line}"
            f"{heading_line}"
            f"内容：{self.text}"
        )


@dataclass(frozen=True)
class AnswerSlot:
    """One final answer obligation tracked across retrieval cycles."""

    slot_id: str
    question: str
    status: str
    evidence_ids: tuple[str, ...] = ()

    def public(self) -> dict[str, object]:
        return {
            "id": self.slot_id,
            "question": self.question,
            "depends_on": [],
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class AgentDecision:
    """Validated control-plane output from one agent model call."""

    action: str
    answer: str = ""
    queries: list[str] = field(default_factory=list)
    query_slots: dict[str, str] = field(default_factory=dict)
    collection_query: str = ""
    calculations: list[dict[str, object]] = field(default_factory=list)
    claims: list[dict[str, object]] = field(default_factory=list)
    slots: list[AnswerSlot] = field(default_factory=list)


class EvidenceManager:
    """Append-only evidence pool plus token-budgeted context selection."""

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}
        self._pinned: set[str] = set()

    def add(
        self,
        items: list[tuple[dict[str, str], str]],
    ) -> list[Evidence]:
        added: list[Evidence] = []
        for result, query in items:
            evidence = Evidence.from_result(result, query)
            if not evidence.text:
                continue
            existing = self._evidence.get(evidence.evidence_id)
            if existing is not None:
                if query and query not in existing.retrieved_by:
                    existing.retrieved_by.append(query)
                continue
            self._evidence[evidence.evidence_id] = evidence
            added.append(evidence)
        return added

    def pin(self, evidence_ids: list[str]) -> None:
        self._pinned.update(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in self._evidence
        )

    def get(self, evidence_id: str) -> Evidence | None:
        return self._evidence.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._evidence.values())

    def results(self) -> list[dict[str, str]]:
        return [evidence.result for evidence in self._evidence.values()]

    def build_context(
        self,
        *,
        newest_ids: list[str] | None = None,
        only_ids: set[str] | None = None,
        max_tokens: int = RAG_MAX_CONTEXT_TOKENS,
        max_results: int = AGENT_MAX_RESULTS,
    ) -> tuple[str, list[Evidence]]:
        if max_tokens <= 0 or max_results <= 0:
            return "", []

        newest_order = newest_ids or []
        order = list(self._evidence)
        prioritized_ids = list(dict.fromkeys([
            *(item for item in order if item in self._pinned),
            *(item for item in newest_order if item in self._evidence),
            *reversed(order),
        ]))
        selected: list[Evidence] = []
        seen_texts: set[str] = set()
        url_counts: dict[str, int] = {}
        token_count = 0

        for evidence_id in prioritized_ids:
            if only_ids is not None and evidence_id not in only_ids:
                continue
            evidence = self._evidence[evidence_id]
            if (
                evidence.text in seen_texts
                or (
                    evidence.url
                    and url_counts.get(evidence.url, 0) >= AGENT_MAX_CHUNKS_PER_URL
                )
            ):
                continue
            number = len(selected) + 1
            block_tokens = _estimate_tokens(evidence.block(number))
            manifest_tokens = _estimate_tokens(
                f"{evidence.evidence_id} | {evidence.published_at} | "
                f"{evidence.title} | {evidence.url}"
            )
            if token_count + block_tokens + manifest_tokens > max_tokens:
                continue
            selected.append(evidence)
            seen_texts.add(evidence.text)
            if evidence.url:
                url_counts[evidence.url] = url_counts.get(evidence.url, 0) + 1
            token_count += block_tokens + manifest_tokens
            if len(selected) == max_results:
                break

        if not selected:
            return "", []
        manifest = "\n".join(
            f"{item.evidence_id} | {item.published_at or '日期未知'} | "
            f"{item.title} | {item.url}"
            for item in selected
        )
        documents = "\n\n".join(
            item.block(number)
            for number, item in enumerate(selected, start=1)
        )
        return f"<来源清单>\n{manifest}\n</来源清单>\n\n{documents}", selected


def merge_results(
    results: list[dict[str, str]],
    max_chars: int = RAG_MAX_CONTEXT_CHARS,
) -> str:
    """Format complete evidence blocks within a conservative token budget."""

    manager = EvidenceManager()
    manager.add([(result, "") for result in results])
    # Keep this compatibility argument conservative: one non-ASCII char is one
    # estimated token, so complete blocks are selected instead of sliced.
    context, _selected = manager.build_context(
        newest_ids=[evidence.evidence_id for evidence in manager.all()],
        max_tokens=max_chars,
    )
    return context
