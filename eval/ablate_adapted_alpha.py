import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeta_engine.dense import search_dense  # noqa: E402
from zeta_engine.eval import (  # noqa: E402
    DEFAULT_BASE_URL,
    input_idx,
    input_passwd,
    login,
    send_answers,
)
from zeta_engine.search import _fuse_scores, _score_bm25f  # noqa: E402
from zeta_engine.storage import Storage  # noqa: E402
from zeta_engine.tokenizer import text_normalize, tokenize_with_positions  # noqa: E402


REPORT = ROOT / "outputs/adapted_alpha_ablation.json"
METHODS = (
    "bm25f",
    "fixed_0.20",
    "fixed_0.38",
    "fixed_0.50",
    "adapt_title",
    "adapt_confidence",
    "adapt_specificity",
    "adapt_full",
    "rrf",
)


def confidence(scores):
    if len(scores) < 2:
        return 1.0 if scores else 0.0
    tail = scores[min(19, len(scores) - 1)]
    scale = scores[0] - tail
    return max(0.0, min(1.0, (scores[0] - scores[1]) / scale)) if scale else 0.0


def title_coverage(storage, query, document_id):
    document = storage.documents.get(document_id)
    if document is None:
        return 0.0
    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    query_terms = {term for term, _ in tokenize_with_positions(query, mode) if term.strip()}
    title_terms = {
        term
        for term, _ in tokenize_with_positions(text_normalize(document[1]), mode)
        if term.strip()
    }
    return len(query_terms & title_terms) / len(query_terms) if query_terms else 0.0


def is_specific(query):
    compact = query.replace(" ", "")
    return (
        len(compact) >= 12
        or bool(re.search(r"20\d{2}|是什么|为什么|如何|哪里|研究方向", query))
        or bool(re.search(r"论文|奖|讲座|报告|论坛|名单|主页", query))
        or bool(re.search(r"[A-Za-z0-9]{2,}", query))
    )


def adaptive_alphas(query, bm_conf, dense_conf, coverage, agreement):
    title = 0.10 if coverage >= 0.8 else 0.20

    if coverage >= 0.8 or bm_conf >= dense_conf + 0.15:
        confidence_alpha = 0.10
    elif dense_conf >= 0.25:
        confidence_alpha = 0.38
    else:
        confidence_alpha = 0.20

    if coverage >= 0.8 or bm_conf >= dense_conf + 0.15:
        specificity = 0.10
    elif is_specific(query) and dense_conf >= 0.25:
        specificity = 0.38
    else:
        specificity = 0.20

    full = specificity
    if (
        full == 0.20
        and agreement >= 0.3
        and dense_conf >= bm_conf
    ):
        full = 0.38
    return title, confidence_alpha, specificity, full


def rrf(sparse_ids, dense_ids, limit=20, k=60):
    scores = Counter()
    for rank, document_id in enumerate(sparse_ids, 1):
        scores[document_id] += 1 / (k + rank)
    for rank, document_id in enumerate(dense_ids, 1):
        scores[document_id] += 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]


def urls(storage, document_ids):
    return [
        document[0]
        for document_id in document_ids
        if (document := storage.documents.get(document_id)) is not None
    ]


def main():
    idx = input_idx()
    passwd = input_passwd()
    queries = login(DEFAULT_BASE_URL, idx, passwd)
    answers = {method: [] for method in METHODS}
    alpha_counts = {method: Counter() for method in METHODS if method.startswith("adapt_")}
    latencies = []
    diagnostics = []

    with Storage(
        document_db=ROOT / "data/zeta.db",
        index_db=ROOT / "data/index.db",
    ) as storage:
        for query in queries:
            started = time.monotonic()
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
            for method in METHODS:
                answers[method].append(urls(storage, ranked[method]))
            diagnostics.append(
                {
                    "query_number": len(diagnostics) + 1,
                    "bm25f_confidence": bm_conf,
                    "dense_confidence": dense_conf,
                    "title_coverage": coverage,
                    "top10_agreement": agreement,
                    "is_specific": is_specific(query),
                    "alphas": chosen,
                }
            )
            latencies.append(time.monotonic() - started)

    report = []
    for method in METHODS:
        mode, mrr, details, average_latency = send_answers(
            DEFAULT_BASE_URL, idx, passwd, answers[method], latencies
        )
        result = {
            "method": method,
            "mrr@20": mrr,
            "recall@20": sum(score > 0 for score in details) / len(details),
            "top1": sum(score == 1 for score in details),
            "alpha_counts": dict(sorted(alpha_counts.get(method, {}).items())),
            "reciprocal_ranks": details,
            "shared_retrieval_latency_seconds": average_latency,
        }
        report.append(result)
        print(result, flush=True)

    for query, scores in zip(
        diagnostics,
        zip(*(result["reciprocal_ranks"] for result in report)),
    ):
        query["reciprocal_ranks"] = dict(zip(METHODS, scores))
    output = {"summary": report, "query_diagnostics": diagnostics}
    REPORT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
