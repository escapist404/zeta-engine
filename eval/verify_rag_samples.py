#!/usr/bin/env python3
"""Run the local RAG samples and judge answers against held-out references."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from zeta_engine.dense import DEFAULT_MODEL_PATH, _encode, _load_model
from zeta_engine.rag import call_model
from zeta_engine.service import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval/rag_samples.json"))
    parser.add_argument("--document-db", type=Path, default=Path("data/zeta.db"))
    parser.add_argument("--index-db", type=Path, default=Path("data/index.db"))
    parser.add_argument("--dense-index", type=Path, default=Path("data/dense"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--judge", choices=("local", "model"), default="local")
    parser.add_argument("--answers-from", type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


_QUANTITY_RE = re.compile(
    r"([零〇一二两三四五六七八九十百千万\d.]+)\s*(年|名|人|所|篇|项|%)"
)
_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
           "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _number(value: str) -> float:
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    total = current = 0
    for character in value:
        if character in _DIGITS:
            current = _DIGITS[character]
        elif character in _UNITS:
            total += (current or 1) * _UNITS[character]
            current = 0
    return float(total + current)


def _quantities(text: str) -> set[tuple[float, str]]:
    return {(_number(value), unit) for value, unit in _QUANTITY_RE.findall(text)}


def judge_answer(
    model: object,
    reference: str,
    answer: str,
    threshold: float,
) -> dict[str, object]:
    vectors = _encode(
        model,
        [reference, answer],
        batch_size=2,
        show_progress_bar=False,
    )
    similarity = float(np.dot(vectors[0], vectors[1]))
    expected_quantities = _quantities(reference)
    actual_quantities = _quantities(answer)
    missing_quantities = sorted(expected_quantities - actual_quantities)
    score = int(similarity >= threshold and not missing_quantities)
    reason = f"similarity={similarity:.4f}"
    if missing_quantities:
        reason += f", missing_quantities={missing_quantities}"
    return {"score": score, "reason": reason, "similarity": similarity}


def judge_answer_with_model(
    query: str,
    reference: str,
    answer: str,
) -> dict[str, object]:
    prompt = f"""你是答案正确性评测器。比较候选答案和参考答案，只判断用户明确要求的关键事实。
措辞、语序和额外解释可以不同；关键事实全部正确且完整时 score=1，否则 score=0。
只输出 JSON：{{"score": 0或1, "reason": "简短原因"}}

问题：{query}
参考答案：{reference}
候选答案：{answer}
"""
    payload = json.loads(call_model(prompt, json_output=True))
    score = payload.get("score")
    reason = payload.get("reason")
    if score not in {0, 1} or not isinstance(reason, str):
        raise ValueError("模型 judge 返回格式不正确")
    return {"score": int(score), "reason": reason}


def main() -> int:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("本地 RAG 样例必须是非空数组")
    if args.ids:
        selected_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
        cases = [case for case in cases if case.get("id") in selected_ids]
        if not cases:
            raise ValueError("--ids 未匹配任何本地 RAG 样例")

    judge_model = (
        _load_model(str(DEFAULT_MODEL_PATH), args.device)
        if args.judge == "local"
        else None
    )
    previous_reports = {}
    if args.answers_from:
        previous = json.loads(args.answers_from.read_text(encoding="utf-8"))
        previous_reports = {
            str(report.get("id")): report
            for report in previous.get("reports", [])
            if isinstance(report, dict)
        }
    reports = []
    for number, case in enumerate(cases, start=1):
        started = time.monotonic()
        try:
            previous_report = previous_reports.get(str(case.get("id")))
            if previous_report is None:
                response = answer_question(
                    args.document_db,
                    args.index_db,
                    str(case["query"]),
                    top_k=args.top_k,
                    dense_index=args.dense_index,
                    device=args.device,
                    debug=True,
                )
                answer = str(response.get("answer", ""))
            else:
                response = {"trace": previous_report.get("trace", [])}
                answer = str(previous_report.get("answer", ""))
            if args.judge == "model":
                judgment = judge_answer_with_model(
                    str(case["query"]),
                    str(case["reference_answer"]),
                    answer,
                )
            else:
                assert judge_model is not None
                judgment = judge_answer(
                    judge_model,
                    str(case["reference_answer"]),
                    answer,
                    args.similarity_threshold,
                )
            error = ""
        except Exception as exc:
            response = {}
            answer = ""
            judgment = {"score": 0, "reason": "运行失败"}
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - started, 3)
        reports.append({
            "id": case.get("id", f"case-{number}"),
            "score": judgment["score"],
            "reason": judgment["reason"],
            "similarity": judgment.get("similarity", 0.0),
            "answer": answer,
            "elapsed_seconds": elapsed,
            "error": error,
            "trace": response.get("collection_trace", response.get("trace", [])),
        })
        print(
            f"Local RAG {number}/{len(cases)}: "
            f"score={judgment['score']} elapsed={elapsed:.2f}s",
            flush=True,
        )

    score = sum(int(report["score"]) for report in reports) / len(reports)
    result = {"score": score, "passed": score == 1.0, "reports": reports}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(f"Local RAG score: {score:.3f} ({sum(r['score'] for r in reports)}/{len(reports)})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
