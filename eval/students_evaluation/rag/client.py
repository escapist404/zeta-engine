"""进阶 RAG 评测客户端。

空密码进入可重复提交的 debug 示例；正式密码进入 10 题正式评测。
每道题调用一次 ``search_engine.rag_evaluate``，且只提交一个字符串答案。
"""

import ast
import getpass
import json
import time
from typing import Any
from urllib.parse import urljoin

import requests

from search_engine import rag_evaluate


# 请替换为助教提供的评测服务器地址，并保留末尾斜杠。
base_url = "http://127.0.0.1:8080/"


def _parse_response(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    try:
        data = ast.literal_eval(response.text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("评测服务器返回了无法解析的数据") from exc
    if not isinstance(data, dict):
        raise ValueError("评测服务器返回的数据不是字典")
    return data


def rag_login(idx: str, passwd: str) -> list[str]:
    response = requests.post(
        urljoin(base_url, "rag/login"),
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


def send_answers(
    idx: str,
    passwd: str,
    answers: list[str],
    elapsed_seconds: list[float],
) -> tuple[str, float, list[float], float, list[dict[str, Any]]]:
    response = requests.post(
        urljoin(base_url, "rag/score"),
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

    mode, score, details = (
        data.get("mode"),
        data.get("score"),
        data.get("details"),
    )
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


def main() -> None:
    idx = input("idx: ").strip()
    passwd = getpass.getpass(
        "passwd for RAG final submission (None for debug mode): "
    )
    if passwd == "":
        print("=== RAG DEBUG MODE ===")
    queries = rag_login(idx, passwd)

    # 不要打印、保存 queries、answers 或 passwd。
    answers: list[str] = []
    elapsed_seconds: list[float] = []
    for number, query in enumerate(queries, start=1):
        started = time.monotonic()
        try:
            answer = rag_evaluate(query)
        except Exception as exc:
            print(f"RAG question {number} failed: {type(exc).__name__}")
            answer = ""
        elapsed = time.monotonic() - started

        if not isinstance(answer, str):
            raise TypeError("rag_evaluate(query) 必须返回一个字符串")
        if elapsed > 60:
            print(f"RAG question {number} exceeded 60 seconds and is invalid.")
            answer = ""

        answers.append(answer.strip())
        elapsed_seconds.append(round(elapsed, 3))
        print(
            f"RAG question {number}/{len(queries)} "
            f"finished in {elapsed:.2f}s"
        )

    mode, score, details, average_latency, judge_outputs = send_answers(
        idx, passwd, answers, elapsed_seconds
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


if __name__ == "__main__":
    main()
