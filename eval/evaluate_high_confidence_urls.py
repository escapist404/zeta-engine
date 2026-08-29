import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeta_engine.eval import evaluate  # noqa: E402
from zeta_engine.storage import Storage  # noqa: E402


CASES = ROOT / "outputs/day5_query_target_urls_high_confidence.jsonl"
REPORT = ROOT / "outputs/day5_high_confidence_mrr.json"
RANKINGS = ("bm25f", "dense", "hybrid", "rerank")
DEVICE = "mps"


def main() -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines()]
    results = {}

    for ranking in RANKINGS:
        reciprocal_ranks = []
        details = []
        started = time.monotonic()
        with Storage(
            document_db=ROOT / "data/zeta.db",
            index_db=None if ranking == "dense" else ROOT / "data/index.db",
        ) as storage:
            for number, case in enumerate(cases, start=1):
                urls = evaluate(
                    storage,
                    case["query"],
                    ranking=ranking,
                    dense_index=ROOT / "data/dense",
                    reranker_model=ROOT / "models/bge-reranker-base",
                    device=DEVICE,
                )
                relevant = set(case["target_urls"])
                rank = next(
                    (index for index, url in enumerate(urls, 1) if url in relevant),
                    None,
                )
                reciprocal_rank = 1 / rank if rank else 0.0
                reciprocal_ranks.append(reciprocal_rank)
                details.append({
                    "query": case["query"],
                    "relevant_urls": case["target_urls"],
                    "rank": rank,
                    "reciprocal_rank": reciprocal_rank,
                    "matched_url": urls[rank - 1] if rank else None,
                    "top_urls": urls[:5],
                })
                if number % 10 == 0 or number == len(cases):
                    print(f"{ranking}: {number}/{len(cases)}", flush=True)

        elapsed = time.monotonic() - started
        results[ranking] = {
            "avg_mrr@20": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "hits@20": sum(score > 0 for score in reciprocal_ranks),
            "queries": len(cases),
            "elapsed_seconds": elapsed,
            "details": details,
        }
        REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"{ranking}: {results[ranking]}", flush=True)

    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
