#!/usr/bin/env python3
"""Evaluate local search rankings with random_eval_queries100_v2.json."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeta_engine.eval import evaluate  # noqa: E402
from zeta_engine.search import DEFAULT_HYBRID_ALPHA  # noqa: E402
from zeta_engine.storage import Storage  # noqa: E402


DEFAULT_CASES = ROOT / "eval/random_eval_queries100_v2.json"
DEFAULT_DOCUMENT_DB = ROOT / "data/zeta.db"
DEFAULT_INDEX_DB = ROOT / "data/index.db"
DEFAULT_DENSE_INDEX = ROOT / "data/dense"
DEFAULT_OUTPUT = ROOT / "outputs/random_eval_queries100_v2_report.json"
RANKINGS = ("bm25f", "dense", "hybrid", "rerank")


def canonical_url(url: str) -> str:
    """Ignore document fragments, which do not identify a different page."""
    return urldefrag(url).url


def load_cases(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, list) or not pages:
        raise ValueError("题库必须包含非空的 pages 数组")

    cases = []
    for page_number, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"pages[{page_number - 1}] 不是对象")
        url = page.get("url")
        queries = page.get("queries")
        if not isinstance(url, str) or not url:
            raise ValueError(f"pages[{page_number - 1}] 缺少有效 URL")
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(query, str) and query.strip() for query in queries)
        ):
            raise ValueError(f"pages[{page_number - 1}] 缺少有效 queries")
        for query_number, query in enumerate(queries, start=1):
            cases.append({
                "id": f"p{page_number:03d}-q{query_number}",
                "page_id": f"p{page_number:03d}",
                "docid": page.get("docid"),
                "title": page.get("title"),
                "query": query,
                "target_url": url,
            })
    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def best_of_page(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["page_id"]), []).append(row)

    pages = []
    for page_id, page_rows in grouped.items():
        hits = [row for row in page_rows if row["rank"] is not None]
        best = min(hits, key=lambda row: int(row["rank"])) if hits else None
        pages.append({
            "page_id": page_id,
            "docid": page_rows[0]["docid"],
            "title": page_rows[0]["title"],
            "target_url": page_rows[0]["target_url"],
            "rank": best["rank"] if best else None,
            "best_query_id": best["id"] if best else None,
            "best_query": best["query"] if best else None,
        })
    return pages


def summarize(
    pages: list[dict[str, object]],
    query_rows: list[dict[str, object]],
    limit: int,
) -> dict[str, float | int]:
    ranks = [page["rank"] for page in pages]
    latencies = [float(row["latency_seconds"]) for row in query_rows]
    total = len(pages)
    cutoff = min(20, limit)
    return {
        "pages": total,
        "queries": len(query_rows),
        "queries_per_page": len(query_rows) / total,
        "hits": sum(rank is not None for rank in ranks),
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / total,
        "recall@5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        f"recall@{cutoff}": (
            sum(rank is not None and rank <= cutoff for rank in ranks) / total
        ),
        f"mrr@{cutoff}": (
            sum(1 / rank for rank in ranks if rank is not None and rank <= cutoff)
            / total
        ),
        "latency_mean_seconds": statistics.fmean(latencies),
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
    }


def evaluate_ranking(
    storage: Storage,
    cases: list[dict[str, object]],
    args: argparse.Namespace,
    ranking: str,
) -> dict[str, object]:
    rows = []
    for number, case in enumerate(cases, start=1):
        started = time.monotonic()
        urls = evaluate(
            storage,
            str(case["query"]),
            limit=args.limit,
            ranking=ranking,
            dense_index=args.dense_index,
            device=args.device,
            alpha=args.alpha,
        )
        elapsed = time.monotonic() - started
        target_url = canonical_url(str(case["target_url"]))
        rank = next(
            (
                index
                for index, url in enumerate(urls, start=1)
                if canonical_url(url) == target_url
            ),
            None,
        )
        rows.append({
            **case,
            "rank": rank,
            "reciprocal_rank": 1 / rank if rank is not None else 0.0,
            "latency_seconds": round(elapsed, 6),
            "top_urls": urls[:5],
        })
        if number % 25 == 0 or number == len(cases):
            print(f"{ranking}: {number}/{len(cases)}", flush=True)

    pages = best_of_page(rows)
    return {
        "metrics": summarize(pages, rows, args.limit),
        "pages": pages,
        "details": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--document-db", type=Path, default=DEFAULT_DOCUMENT_DB)
    parser.add_argument("--index-db", type=Path, default=DEFAULT_INDEX_DB)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument(
        "--rankings",
        default="bm25f",
        help="逗号分隔的排名方法：bm25f,dense,hybrid,rerank",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.rankings = [item.strip() for item in args.rankings.split(",") if item.strip()]
    invalid = set(args.rankings) - set(RANKINGS)
    if invalid:
        parser.error("不支持的 ranking: " + ", ".join(sorted(invalid)))
    if not args.rankings:
        parser.error("--rankings 不能为空")
    if args.limit <= 0:
        parser.error("--limit 必须大于 0")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha 必须在 0 到 1 之间")
    return args


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} queries from {args.cases}")

    with Storage(document_db=args.document_db, index_db=args.index_db) as storage:
        results = {
            ranking: evaluate_ranking(storage, cases, args, ranking)
            for ranking in args.rankings
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(args.cases),
        "settings": {
            "rankings": args.rankings,
            "limit": args.limit,
            "alpha": args.alpha,
            "device": args.device,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for ranking, result in results.items():
        metrics = result["metrics"]
        cutoff = min(20, args.limit)
        print(
            f"{ranking}: MRR@{cutoff}={metrics[f'mrr@{cutoff}']:.4f}, "
            f"R@1={metrics['recall@1']:.4f}, "
            f"R@5={metrics['recall@5']:.4f}, "
            f"R@{cutoff}={metrics[f'recall@{cutoff}']:.4f}, "
            f"p95={metrics['latency_p95_seconds']:.3f}s"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
