"""Search MRR@20 and RAG evaluation clients."""

import ast
import getpass
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense, warmup_dense
from zeta_engine.search import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid,
    search_reranked,
)
from zeta_engine.service import answer_question
from zeta_engine.storage import Storage

DEFAULT_BASE_URL = "http://10.47.253.18:8080/"


def input_idx() -> str:
    return input("idx: ").strip()


def input_passwd() -> str:
    passwd = getpass.getpass(
        "passwd for final submission (None for debug mode): "
    )
    if not passwd:
        print("=== DEBUG MODE ===")
    return passwd


def _parse_response(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    try:
        data = ast.literal_eval(response.text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("评测服务器返回了无法解析的数据") from exc
    if not isinstance(data, dict):
        raise ValueError("评测服务器返回的数据不是字典")
    return data


def login(base_url: str, idx: str, passwd: str) -> list[str]:
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "login"),
        data={"idx": idx, "passwd": passwd},
        timeout=15,
    )
    data = _parse_response(response)
    if data.get("mode") == "illegal":
        raise ValueError("illegal password!")

    queries = data.get("queries")
    if not isinstance(queries, list) or not queries or not all(
        isinstance(query, str) for query in queries
    ):
        raise ValueError("评测服务器未返回查询，请检查题库并重启评测服务")
    print(f"{len(queries)} queries.")
    return queries


def send_answers(
    base_url: str,
    idx: str,
    passwd: str,
    urls: list[list[str]],
    elapsed_seconds: list[float],
) -> tuple[str, float, list[float], float]:
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "mrr"),
        data={
            "idx": idx,
            "passwd": passwd,
            "urls": json.dumps(urls),
            "elapsed_seconds": json.dumps(elapsed_seconds),
        },
        timeout=180,
    )
    data = _parse_response(response)
    if data.get("mode") == "illegal":
        raise ValueError("illegal password!")
    if data.get("mode") == "error":
        raise RuntimeError(data.get("message", "评测服务器错误"))

    mode = data.get("mode")
    mrr = data.get("mrr")
    details = data.get("details", [])
    average_latency = data.get("average_latency_seconds")
    if (
        not isinstance(mode, str)
        or not isinstance(mrr, (int, float))
        or not isinstance(average_latency, (int, float))
    ):
        raise ValueError("评测服务器返回的分数或平均时延格式不正确")
    if mode == "debug":
        if not isinstance(details, list) or not all(
            isinstance(item, (int, float)) for item in details
        ):
            raise ValueError("评测服务器返回的逐题分数格式不正确")
    else:
        details = []
    return (
        mode,
        float(mrr),
        [float(item) for item in details],
        float(average_latency),
    )


def rag_login(base_url: str, idx: str, passwd: str) -> list[str]:
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "rag/login"),
        data={"idx": idx, "passwd": passwd},
        timeout=15,
    )
    data = _parse_response(response)
    if data.get("mode") == "illegal":
        raise ValueError(data.get("message", "illegal password!"))
    if data.get("mode") == "error":
        raise RuntimeError(data.get("message", "RAG 评测服务器错误"))

    queries = data.get("queries")
    if not isinstance(queries, list) or not queries or not all(
        isinstance(query, str) for query in queries
    ):
        raise ValueError("RAG 评测服务器返回的查询格式不正确")
    print(f"{len(queries)} RAG queries, [{data.get('mode')}] mode.")
    return queries


def send_rag_answers(
    base_url: str,
    idx: str,
    passwd: str,
    answers: list[str],
    elapsed_seconds: list[float],
) -> tuple[str, float, list[float], float, list[dict[str, Any]]]:
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "rag/score"),
        data={
            "idx": idx,
            "passwd": passwd,
            "answers": json.dumps(answers, ensure_ascii=False),
            "elapsed_seconds": json.dumps(elapsed_seconds),
        },
        timeout=180,
    )
    data = _parse_response(response)
    if data.get("mode") == "illegal":
        raise ValueError(data.get("message", "illegal password!"))
    if data.get("mode") == "error":
        raise RuntimeError(data.get("message", "RAG 裁判服务错误"))

    mode = data.get("mode")
    score = data.get("score")
    details = data.get("details")
    average_latency = data.get("average_latency_seconds")
    if (
        not isinstance(mode, str)
        or not isinstance(score, (int, float))
        or not isinstance(details, list)
        or not all(isinstance(item, (int, float)) for item in details)
        or not isinstance(average_latency, (int, float))
    ):
        raise ValueError("评测服务器返回的 RAG 分数格式不正确")

    judge_outputs = data.get("judge_outputs", [])
    if mode == "debug":
        if (
            not isinstance(judge_outputs, list)
            or len(judge_outputs) != len(details)
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("score"), (int, float))
                and isinstance(item.get("reason"), str)
                for item in judge_outputs
            )
        ):
            raise ValueError("评测服务器返回的裁判输出格式不正确")
    else:
        judge_outputs = []
    return (
        mode,
        float(score),
        [float(item) for item in details],
        float(average_latency),
        judge_outputs,
    )


def evaluate(
    storage: Storage,
    query: str,
    *,
    limit: int = 20,
    ranking: str = "hybrid",
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> list[str]:
    """Return matching document URLs ordered by the selected ranking."""
    assert storage.documents is not None
    if ranking == "dense":
        document_ids = [
            hit.document_id
            for hit in search_dense(
                query,
                dense_index,
                limit=limit,
                device=device,
            )
        ]
    elif ranking == "bm25f":
        document_ids = search_bm25f(storage, query)[:limit]
    elif ranking == "hybrid":
        document_ids = search_hybrid(
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
        raise ValueError(f"不支持的评测排名方式: {ranking}")

    return [
        document[0]
        for document_id in document_ids
        if (document := storage.documents.get(document_id)) is not None
    ]


def run_evaluation(
    document_db: str | Path,
    index_db: str | Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    ranking: str = "hybrid",
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    reranker_model: str | Path = DEFAULT_RERANKER_MODEL_PATH,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    reranker_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> None:
    idx = input_idx()
    passwd = input_passwd()
    queries = login(base_url, idx, passwd)

    all_urls: list[list[str]] = []
    elapsed_seconds: list[float] = []
    with Storage(
        document_db=document_db,
        index_db=index_db if ranking != "dense" else None,
    ) as storage:
        for query in queries:
            started = time.monotonic()
            all_urls.append(evaluate(
                storage,
                query,
                ranking=ranking,
                dense_index=dense_index,
                reranker_model=reranker_model,
                rerank_candidates=rerank_candidates,
                reranker_batch_size=reranker_batch_size,
                device=device,
                alpha=alpha,
            ))
            elapsed_seconds.append(round(time.monotonic() - started, 3))

    mode, mrr, details, average_latency = send_answers(
        base_url,
        idx,
        passwd,
        all_urls,
        elapsed_seconds,
    )

    print(f"MRR@20: [{mrr}], [{mode}] mode")
    if mode == "debug":
        print(f"Per-query reciprocal ranks: {details}")
    print(f"Average latency: {average_latency:.3f}s/query")


def run_rag_evaluation(
    document_db: str | Path,
    index_db: str | Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    dense_index: str | Path = DEFAULT_INDEX_DIR,
    device: str | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    top_k: int = 5,
) -> None:
    idx = input_idx()
    passwd = input_passwd()
    if alpha > 0.:
        print("Loading BGE...", flush=True)
        warmup_dense(dense_index, device=device)
        print("BGE ready.", flush=True)
    queries = rag_login(base_url, idx, passwd)

    answers: list[str] = []
    elapsed_seconds: list[float] = []
    for number, query in enumerate(queries, start=1):
        started = time.monotonic()
        try:
            response = answer_question(
                document_db,
                index_db,
                query,
                top_k=top_k,
                dense_index=dense_index,
                device=device,
                alpha=alpha,
            )
            answer = response["answer"]
        except Exception as exc:
            print(f"RAG question {number} failed: {type(exc).__name__}")
            answer = ""
        elapsed = time.monotonic() - started

        if not isinstance(answer, str):
            raise TypeError("RAG 回答必须是字符串")
        if elapsed > 60:
            print(f"RAG question {number} exceeded 60 seconds and is invalid.")
            answer = ""

        answers.append(answer.strip())
        elapsed_seconds.append(round(elapsed, 3))
        print(
            f"RAG question {number}/{len(queries)} "
            f"finished in {elapsed:.2f}s"
        )

    mode, score, details, average_latency, judge_outputs = send_rag_answers(
        base_url,
        idx,
        passwd,
        answers,
        elapsed_seconds,
    )
    print(f"RAG score: [{score}], details={details}, [{mode}] mode")
    print(f"Average latency: {average_latency:.3f}s/query")
    if mode == "debug":
        print("=== RAG JUDGE OUTPUTS ===")
        for number, output in enumerate(judge_outputs, start=1):
            print(
                f"Judge {number}: score={float(output['score'])}, "
                f"reason={output['reason']}"
            )
