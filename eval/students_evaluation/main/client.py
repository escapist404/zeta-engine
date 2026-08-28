"""原检索（MRR@20）评测客户端。

本文件只负责原有检索评测；RAG 评测代码位于 ``rag/`` 目录。
正式评测时不得打印、保存或以其他方式泄露服务器下发的查询。
"""

import ast
import getpass
import json
import time
from typing import Any
from urllib.parse import urljoin

import requests

if __package__:
    from .search_engine import evaluate
else:
    from search_engine import evaluate

# 请替换为助教提供的评测服务器地址，并保留末尾斜杠。
base_url = "http://10.47.253.18:8080/"


def input_idx() -> str:
    return input("idx: ").strip()


def input_passwd() -> str:
    passwd = getpass.getpass(
        "passwd for final submission (None for debug mode): "
    )
    if passwd == "":
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


def login(idx: str, passwd: str) -> list[str]:
    response = requests.post(
        urljoin(base_url, "login"),
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
        raise ValueError(
            "评测服务器未返回查询，请检查题库并重启评测服务"
        )
    print(f"{len(queries)} queries.")
    return queries


def send_ans(
    idx: str,
    passwd: str,
    urls: list[list[str]],
    elapsed_seconds: list[float],
) -> tuple[str, float, float]:
    response = requests.post(
        urljoin(base_url, "mrr"),
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

    mode, mrr = data.get("mode"), data.get("mrr")
    average_latency = data.get("average_latency_seconds")
    if (
        not isinstance(mode, str)
        or not isinstance(mrr, (int, float))
        or not isinstance(average_latency, (int, float))
    ):
        raise ValueError("评测服务器返回的分数或平均时延格式不正确")
    return mode, float(mrr), float(average_latency)


def main() -> None:
    idx = input_idx()
    passwd = input_passwd()
    queries = login(idx, passwd)

    # 不要在正式评测时打印 queries 或 passwd。
    all_urls: list[list[str]] = []
    elapsed_seconds: list[float] = []
    for query in queries:
        started = time.monotonic()
        result = evaluate(query)
        elapsed = time.monotonic() - started
        if not isinstance(result, list) or not all(
            isinstance(url, str) for url in result
        ):
            raise TypeError("evaluate(query) 必须返回由 URL 字符串组成的列表")
        all_urls.append(result[:20])
        elapsed_seconds.append(round(elapsed, 3))

    mode, mrr, average_latency = send_ans(
        idx,
        passwd,
        all_urls,
        elapsed_seconds,
    )
    print(f"MRR@20: [{mrr}], [{mode}] mode")
    print(f"Average latency: {average_latency:.3f}s/query")


if __name__ == "__main__":
    main()
