"""Shared response contracts for every RAG execution strategy."""

from typing import Literal, NotRequired, TypedDict


RagStatus = Literal["answered", "partial", "missing", "conflicting"]


class RagResponse(TypedDict):
    """Stable response envelope returned by all RAG executors."""

    answer: str
    status: RagStatus
    complete: bool
    requirements: list[dict[str, object]]
    claims: list[dict[str, object]]
    sources: list[dict[str, object]]
    results: list[dict[str, object]]
    answer_slots: NotRequired[list[dict[str, object]]]
    model_call_count: NotRequired[int]
    trace: NotRequired[list[dict[str, object]]]
    collection_trace: NotRequired[dict[str, object]]
