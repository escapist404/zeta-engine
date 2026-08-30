"""Stable compatibility facade for the RAG implementation."""

# Keep the historical module namespace intact while implementation moves to
# focused modules; some callers patch or import these names directly.

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256

from openai import OpenAI

from zeta_engine.rag_config import (
    AGENT_MAX_CHUNKS_PER_URL,
    AGENT_MAX_CYCLES,
    AGENT_MAX_LLM_CALLS,
    AGENT_MAX_QUERIES,
    AGENT_MAX_RESULTS,
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_BASE_URL_ENV,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_MODEL_ENV,
    LLM_TIMEOUT_SECONDS,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MAX_CONTEXT_TOKENS,
    RAG_NO_RESULTS_ANSWER,
)
from zeta_engine.rag_engine import _run_closed_loop as _run_closed_loop_impl
from zeta_engine.rag_evidence import (
    AgentDecision,
    AnswerSlot,
    Evidence,
    EvidenceManager,
    _clean,
    _estimate_tokens,
    _grounding_text,
    _normalize_text,
    _result_text,
    merge_results,
)
from zeta_engine.rag_prompts import (
    build_agent_prompt,
    build_prompt,
    build_verifier_prompt,
)
from zeta_engine.rag_protocol import (
    _parse_agent_action,
    _parse_answer_slots,
    _parse_json_object,
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
    _parse_ordinal_number,
    _prefer_latest_record,
    _query_company_entities,
    _range_classification_rules,
    _requires_complete_collection,
    _superseded_evidence_ids,
    _title_entity_keys,
)
from zeta_engine.tokenizer import text_normalize, tokenize_with_positions


def call_model(prompt: str, *, json_output: bool = False) -> str:
    """Send one prompt to the model and return its text answer."""

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt 不能为空")

    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"请先设置环境变量 {LLM_API_KEY_ENV}")
    base_url = os.environ.get(LLM_BASE_URL_ENV, LLM_BASE_URL).strip()
    model = os.environ.get(LLM_MODEL_ENV, LLM_MODEL).strip()
    if not base_url:
        raise RuntimeError(f"环境变量 {LLM_BASE_URL_ENV} 不能为空")
    if not model:
        raise RuntimeError(f"环境变量 {LLM_MODEL_ENV} 不能为空")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": LLM_MAX_OUTPUT_TOKENS,
        "stream": False,
    }
    if "deepseek.com" in base_url.lower():
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    if json_output:
        request["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        **request,
    )
    if not response.choices:
        raise RuntimeError("大模型未返回答案")

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("大模型返回了空答案")
    return answer.strip()


def _run_closed_loop(
    query: str,
    search_fn: Callable[[str, int], list[dict[str, str]]],
    top_k: int = 8,
    *,
    max_cycles: int = AGENT_MAX_CYCLES,
    max_llm_calls: int = AGENT_MAX_LLM_CALLS,
    include_citations: bool = False,
    debug: bool = False,
    collection_fn: Callable[[str], list[dict[str, str]]] | None = None,
) -> RagResponse:
    """Run the RAG loop through the historical module-level injection point."""

    return _run_closed_loop_impl(
        query,
        search_fn,
        top_k,
        max_cycles=max_cycles,
        max_llm_calls=max_llm_calls,
        include_citations=include_citations,
        debug=debug,
        model_fn=call_model,
        collection_fn=collection_fn,
    )


def agentic_rag_answer(
    query: str,
    search_fn: Callable[[str, int], list[dict[str, str]]],
    top_k: int = 8,
    *,
    max_cycles: int = AGENT_MAX_CYCLES,
    max_llm_calls: int = AGENT_MAX_LLM_CALLS,
    include_citations: bool = False,
    debug: bool = False,
    collection_fn: Callable[[str], list[dict[str, str]]] | None = None,
) -> RagResponse:
    """Answer a question with one bounded, evidence-preserving closed loop."""

    return _run_closed_loop(
        query,
        search_fn,
        top_k,
        max_cycles=max_cycles,
        max_llm_calls=max_llm_calls,
        include_citations=include_citations,
        debug=debug,
        collection_fn=collection_fn,
    )
