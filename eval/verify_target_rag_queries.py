#!/usr/bin/env python3
"""Run the six target RAG questions as a reproducible end-to-end gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from zeta_engine.service import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("eval/target_rag_cases.json"))
    parser.add_argument("--document-db", type=Path, default=Path("data/zeta.db"))
    parser.add_argument("--index-db", type=Path, default=Path("data/index.db"))
    parser.add_argument("--dense-index", type=Path, default=Path("data/dense"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ids", default="", help="逗号分隔的 case id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def verify_case(case: dict[str, object], response: dict[str, object]) -> list[str]:
    answer = str(response.get("answer", ""))
    errors = []
    for pattern in case.get("expected_regex", []):
        if not re.search(str(pattern), answer, flags=re.IGNORECASE):
            errors.append(f"答案未匹配: {pattern}")

    expected_collection = case.get("expected_collection")
    if isinstance(expected_collection, dict):
        trace = response.get("collection_trace")
        if not isinstance(trace, dict):
            for result in response.get("results", []):
                if not isinstance(result, dict):
                    continue
                raw_trace = result.get("collection_trace")
                if not isinstance(raw_trace, str):
                    continue
                try:
                    candidate = json.loads(raw_trace)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    trace = candidate
                    break
        if not isinstance(trace, dict):
            errors.append("缺少 collection_trace")
        else:
            expected_counts = expected_collection.get("counts", {})
            if trace.get("counts") != expected_counts:
                errors.append(
                    f"集合计数错误: {trace.get('counts')} != {expected_counts}"
                )
            topics = trace.get("topics", [])
            minimum = int(expected_collection.get("minimum_shared_topics", 0))
            if not isinstance(topics, list) or len(topics) < minimum:
                errors.append(f"共同主题不足 {minimum} 个")
            elif any(
                not isinstance(topic, dict)
                or set(topic.get("evidence", {})) != set(expected_counts)
                for topic in topics
            ):
                errors.append("共同主题缺少跨组证据")
    return errors


def main() -> int:
    args = parse_args()
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    selected_ids = {
        item.strip() for item in args.ids.split(",") if item.strip()
    }
    reports = []
    for case in payload["cases"]:
        if selected_ids and case.get("id") not in selected_ids:
            continue
        try:
            response = answer_question(
                args.document_db,
                args.index_db,
                str(case["query"]),
                dense_index=args.dense_index,
                device=args.device,
                debug=True,
            )
            errors = verify_case(case, response)
            reports.append({
                "id": case["id"],
                "passed": not errors,
                "answer": response.get("answer", ""),
                "errors": errors,
                "trace": response.get("collection_trace", response.get("trace", [])),
            })
        except Exception as error:  # keep the gate reporting all cases
            reports.append({
                "id": case.get("id", ""),
                "passed": False,
                "answer": "",
                "errors": [f"{type(error).__name__}: {error}"],
                "trace": [],
            })

    result = {
        "passed": all(report["passed"] for report in reports),
        "reports": reports,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
