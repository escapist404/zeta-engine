import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeta_engine.eval import (  # noqa: E402
    DEFAULT_BASE_URL,
    evaluate,
    login,
    send_answers,
)
from zeta_engine.storage import Storage  # noqa: E402


REPORT = ROOT / "outputs/hybrid_alpha_sweep.json"
COARSE = [*(index / 20 for index in range(11)), 0.6, 0.75, 1.0]
RANKINGS = ("hybrid", "rerank")
IDX = "1"
PASSWD = ""


def run_alpha(storage, queries, idx, passwd, ranking, alpha):
    answers = []
    latencies = []
    for query in queries:
        started = time.monotonic()
        answers.append(evaluate(
            storage,
            query,
            ranking=ranking,
            dense_index=ROOT / "data/dense",
            device="cuda",
            alpha=alpha,
        ))
        latencies.append(time.monotonic() - started)
    mode, mrr, details, average_latency = send_answers(
        DEFAULT_BASE_URL, idx, passwd, answers, latencies
    )
    result = {
        "ranking": ranking,
        "alpha": alpha,
        "mrr@20": mrr,
        "recall@20": sum(score > 0 for score in details) / len(details),
        "top1": sum(score == 1 for score in details),
        "average_latency_seconds": average_latency,
        "reciprocal_ranks": details,
    }
    print(result, flush=True)
    return result


def main():
    idx = IDX
    passwd = PASSWD
    print("=== DEBUG MODE ===")
    queries = login(DEFAULT_BASE_URL, idx, passwd)
    results = []

    with Storage(
        document_db=ROOT / "data/zeta.db",
        index_db=ROOT / "data/index.db",
    ) as storage:
        for ranking in RANKINGS:
            tested = set()
            for alpha in COARSE:
                results.append(
                    run_alpha(storage, queries, idx, passwd, ranking, alpha)
                )
                tested.add(alpha)

            ranking_results = [
                item for item in results if item["ranking"] == ranking
            ]
            best = max(ranking_results, key=lambda item: item["mrr@20"])[
                "alpha"
            ]
            lower = max(0, best - 0.05)
            upper = min(1, best + 0.05)
            for index in range(round(lower * 100), round(upper * 100) + 1):
                alpha = index / 100
                if alpha not in tested:
                    results.append(
                        run_alpha(
                            storage, queries, idx, passwd, ranking, alpha
                        )
                    )

    results.sort(key=lambda item: (item["ranking"], item["alpha"]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"report: {REPORT}")
    for ranking in RANKINGS:
        best = max(
            (item for item in results if item["ranking"] == ranking),
            key=lambda item: item["mrr@20"],
        )
        print(
            f"best {ranking}: alpha={best['alpha']}, "
            f"mrr@20={best['mrr@20']:.6f}, "
            f"recall@20={best['recall@20']:.6f}, top1={best['top1']}"
        )


if __name__ == "__main__":
    main()
