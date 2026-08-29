import json
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablate_adapted_alpha import (  # noqa: E402
    METHODS,
    adaptive_alphas,
    confidence,
    is_specific,
    rrf,
    title_coverage,
    urls,
)
from zeta_engine.dense import search_dense  # noqa: E402
from zeta_engine.search import _fuse_scores, _score_bm25f  # noqa: E402
from zeta_engine.storage import Storage  # noqa: E402


CASES = ROOT / "outputs/day5_query_target_urls_high_confidence.jsonl"
REPORT = ROOT / "outputs/adapted_alpha_ablation_72.json"


def main():
    cases = [json.loads(line) for line in CASES.read_text().splitlines()]
    details = {method: [] for method in METHODS}
    alpha_counts = {method: Counter() for method in METHODS if method.startswith("adapt_")}
    started = time.monotonic()

    with Storage(
        document_db=ROOT / "data/zeta.db",
        index_db=ROOT / "data/index.db",
    ) as storage:
        for number, case in enumerate(cases, 1):
            query = case["query"]
            sparse_all = _score_bm25f(storage, query)
            sparse_ids = sorted(
                sparse_all,
                key=lambda item: (-sparse_all[item], item),
            )[:100]
            sparse_scores = {item: sparse_all[item] for item in sparse_ids}
            dense_hits = search_dense(
                query,
                ROOT / "data/dense",
                limit=100,
                device="mps",
            )
            dense_ids = [hit.document_id for hit in dense_hits]
            dense_scores = {hit.document_id: hit.score for hit in dense_hits}

            bm_conf = confidence([sparse_scores[item] for item in sparse_ids])
            dense_conf = confidence([hit.score for hit in dense_hits])
            coverage = title_coverage(storage, query, sparse_ids[0]) if sparse_ids else 0.0
            agreement = len(set(sparse_ids[:10]) & set(dense_ids[:10])) / 10
            title, conf, specific, full = adaptive_alphas(
                query, bm_conf, dense_conf, coverage, agreement
            )
            chosen = {
                "adapt_title": title,
                "adapt_confidence": conf,
                "adapt_specificity": specific,
                "adapt_full": full,
            }
            ranked = {
                "bm25f": sparse_ids[:20],
                "fixed_0.20": _fuse_scores(sparse_scores, dense_scores, alpha=0.20, limit=20),
                "fixed_0.38": _fuse_scores(sparse_scores, dense_scores, alpha=0.38, limit=20),
                "fixed_0.50": _fuse_scores(sparse_scores, dense_scores, alpha=0.50, limit=20),
                "rrf": rrf(sparse_ids, dense_ids),
            }
            for method, alpha in chosen.items():
                ranked[method] = _fuse_scores(
                    sparse_scores, dense_scores, alpha=alpha, limit=20
                )
                alpha_counts[method][alpha] += 1

            relevant = set(case["target_urls"])
            for method in METHODS:
                ranked_urls = urls(storage, ranked[method])
                rank = next(
                    (index for index, url in enumerate(ranked_urls, 1) if url in relevant),
                    None,
                )
                details[method].append(
                    {
                        "query": query,
                        "alpha": chosen.get(method),
                        "rank": rank,
                        "reciprocal_rank": 1 / rank if rank else 0.0,
                        "matched_url": ranked_urls[rank - 1] if rank else None,
                        "top_urls": ranked_urls[:5],
                    }
                )
            if number % 10 == 0 or number == len(cases):
                print(f"queries: {number}/{len(cases)}", flush=True)

    elapsed = time.monotonic() - started
    report = []
    for method in METHODS:
        scores = [item["reciprocal_rank"] for item in details[method]]
        result = {
            "method": method,
            "mrr@20": sum(scores) / len(scores),
            "recall@20": sum(score > 0 for score in scores) / len(scores),
            "top1": sum(score == 1 for score in scores),
            "alpha_counts": dict(sorted(alpha_counts.get(method, {}).items())),
            "shared_retrieval_seconds": elapsed,
            "details": details[method],
        }
        report.append(result)
        print({key: value for key, value in result.items() if key != "details"})

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
