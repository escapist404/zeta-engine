#!/usr/bin/env python3
"""Run reproducible end-to-end checks for exhaustive collection questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeta_engine.collection_rag import answer_collection_question  # noqa: E402
from zeta_engine.search import (  # noqa: E402
    DEFAULT_HYBRID_ALPHA,
    search_reranked_passages,
)
from zeta_engine.storage import Storage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "eval/collection_eval_cases.json",
    )
    parser.add_argument(
        "--document-db",
        type=Path,
        default=ROOT / "data/zeta.db",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=ROOT / "data/dense",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    return parser.parse_args()


def verify_case(case: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    query = str(case["query"])
    hits = search_reranked_passages(
        query,
        args.dense_index,
        limit=15,
        alpha=args.alpha,
        device=args.device,
    )
    with Storage(document_db=args.document_db) as storage:
        response = answer_collection_question(
            storage,
            query,
            hits,
            topic_mapper=None,
            debug=True,
        )
    errors = []
    if response is None:
        return {"id": case["id"], "passed": False, "errors": ["未进入集合路径"]}

    expected = case["expected"]
    assert isinstance(expected, dict)
    expected_groups = expected["groups"]
    assert isinstance(expected_groups, dict)
    sources = {
        str(source["group"]): source
        for source in response.get("sources", [])
        if isinstance(source, dict)
    }
    for group_value, raw_group in expected_groups.items():
        assert isinstance(raw_group, dict)
        source = sources.get(group_value)
        if source is None:
            errors.append(f"缺少分组 {group_value}")
            continue
        coverage = source.get("coverage", {})
        checks = {
            "document_id": source.get("document_id"),
            "parsed_records": coverage.get("parsed_records"),
            "matched_records": len(source.get("record_ids", [])),
        }
        for name, actual in checks.items():
            if actual != raw_group.get(name):
                errors.append(
                    f"{group_value}.{name}: expected={raw_group.get(name)} actual={actual}"
                )
        if coverage.get("record_complete") is not True:
            errors.append(f"{group_value} 记录覆盖不完整")

    trace = response.get("collection_trace", {})
    topics = trace.get("topics", []) if isinstance(trace, dict) else []
    minimum_topics = int(expected.get("minimum_shared_topics", 0))
    if len(topics) < minimum_topics:
        errors.append(f"共同主题不足: expected>={minimum_topics} actual={len(topics)}")
    topic_labels = " ".join(
        str(topic.get("label", ""))
        for topic in topics
        if isinstance(topic, dict)
    ).casefold()
    for concept in expected.get("expected_topic_concepts", []):
        if str(concept).casefold() not in topic_labels:
            errors.append(f"缺少预期共同主题概念: {concept}")
    for topic in topics:
        evidence = topic.get("evidence", {}) if isinstance(topic, dict) else {}
        if any(not evidence.get(group_value) for group_value in expected_groups):
            errors.append(f"主题缺少跨组证据: {topic}")

    return {
        "id": case["id"],
        "passed": not errors,
        "answer": response.get("answer"),
        "errors": errors,
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测文件必须包含非空 cases")
    reports = [verify_case(case, args) for case in cases]
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))
    if not all(report["passed"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
