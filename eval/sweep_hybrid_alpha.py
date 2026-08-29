import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeta_engine.eval import (  # noqa: E402
    DEFAULT_BASE_URL,
    evaluate,
    input_idx,
    input_passwd,
    login,
    send_answers,
)
from zeta_engine.storage import Storage  # noqa: E402


REPORT = ROOT / "outputs/hybrid_alpha_sweep.json"
COARSE = [*(index / 20 for index in range(11)), 0.6, 0.75, 1.0]


def run_alpha(storage, queries, idx, passwd, alpha):
    answers = []
    latencies = []
    for query in queries:
        started = time.monotonic()
        answers.append(evaluate(
            storage,
            query,
            ranking="hybrid",
            dense_index=ROOT / "data/dense",
            device="mps",
            alpha=alpha,
        ))
        latencies.append(time.monotonic() - started)
    mode, mrr, details, average_latency = send_answers(
        DEFAULT_BASE_URL, idx, passwd, answers, latencies
    )
    result = {
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
    idx = input_idx()
    passwd = input_passwd()
    queries = login(DEFAULT_BASE_URL, idx, passwd)
    results = []
    tested = set()

    with Storage(
        document_db=ROOT / "data/zeta.db",
        index_db=ROOT / "data/index.db",
    ) as storage:
        for alpha in COARSE:
            results.append(run_alpha(storage, queries, idx, passwd, alpha))
            tested.add(alpha)

        best = max(results, key=lambda item: item["mrr@20"])["alpha"]
        lower = max(0, best - 0.05)
        upper = min(1, best + 0.05)
        for index in range(round(lower * 100), round(upper * 100) + 1):
            alpha = index / 100
            if alpha not in tested:
                results.append(run_alpha(storage, queries, idx, passwd, alpha))

    results.sort(key=lambda item: item["alpha"])
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
