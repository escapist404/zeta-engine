"""MRR@20 evaluation client."""

import ast
import getpass
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from zeta_engine.dense import DEFAULT_INDEX_DIR, search_dense
from zeta_engine.search import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid,
    search_reranked,
)
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
    alpha: float = .5,
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
    alpha: float = .5,
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
