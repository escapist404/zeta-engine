import argparse
import logging
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from zeta_engine.constants import ALLOWED_DOMAINS, SEED_URLS
from zeta_engine.crawler import crawl_urls
from zeta_engine.dense import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL_PATH,
    build_dense_index,
    upgrade_dense_index_v3,
)
from zeta_engine.eval import DEFAULT_BASE_URL, run_evaluation, run_rag_evaluation
from zeta_engine.index import build_index
from zeta_engine.rag import AGENT_MAX_CYCLES
from zeta_engine.search import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
)
from zeta_engine.service import answer_question, search_documents
from zeta_engine.storage import Storage
from zeta_engine.web import serve


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger("jieba").setLevel(logging.INFO)


def run_crawler(args: argparse.Namespace) -> int:
    if args.refresh_after_hours is not None and (
        not isfinite(args.refresh_after_hours) or args.refresh_after_hours < 0
    ):
        raise SystemExit("--refresh-after-hours 必须是有限的非负数")
    args.document_db.parent.mkdir(parents=True, exist_ok=True)
    args.queue_db.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_file)

    with Storage(
        document_db=args.document_db,
        queue_db=args.queue_db,
    ) as storage:
        stats = crawl_urls(
            storage,
            seed_urls=SEED_URLS,
            allowed_domains=ALLOWED_DOMAINS,
            max_pages=args.max_pages,
            download_workers=args.workers,
            per_host_delay=args.delay,
            refresh_after_hours=args.refresh_after_hours,
            retry_failed=args.retry_failed,
        )

    logging.info("本次统计: %s", stats)
    return 0


def show_stats(args: argparse.Namespace) -> int:
    if not args.document_db.is_file():
        raise SystemExit(f"数据库不存在: {args.document_db}")

    hosts = sorted(
        {
            host
            for domain in ALLOWED_DOMAINS
            if (host := urlsplit(domain).hostname)
        }
    )

    with Storage(document_db=args.document_db) as storage:
        assert storage.documents is not None
        results = [
            (
                host,
                storage.documents.count_by_host(host),
            )
            for host in hosts
        ]

    term_count = 0
    if args.index_db.is_file():
        with Storage(index_db=args.index_db) as storage:
            assert storage.index is not None
            term_count = storage.index.count_terms()

    for host, count in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"{host}: {count} documents")
    print(f"terms: {term_count}")

    return 0


def run_indexer(args: argparse.Namespace) -> int:
    if not args.document_db.is_file():
        raise SystemExit(f"数据库不存在: {args.document_db}")

    configure_logging(args.log_file)
    logging.info(
        "开始构造索引: mode=%s, document_db=%s, index_db=%s",
        args.mode,
        args.document_db,
        args.index_db,
    )

    with Storage(
        document_db=args.document_db,
        index_db=args.index_db,
    ) as storage:
        stats = build_index(storage, mode=args.mode)

    logging.info("索引完成: %s", stats)

    return 0


def run_dense_indexer(args: argparse.Namespace) -> int:
    if not args.document_db.is_file():
        raise SystemExit(f"数据库不存在: {args.document_db}")
    if not args.upgrade_v3 and not args.model.is_dir():
        raise SystemExit(f"Dense 模型目录不存在: {args.model}")

    configure_logging(args.log_file)
    with Storage(document_db=args.document_db) as storage:
        if args.upgrade_v3:
            stats = upgrade_dense_index_v3(storage, args.dense_index)
        else:
            stats = build_dense_index(
                storage,
                args.dense_index,
                model_path=args.model,
                batch_size=args.batch_size,
                device=args.device,
            )
    logging.info("Dense 索引完成: %s", stats)
    return 0


def run_search(args: argparse.Namespace) -> int:
    if not args.document_db.is_file():
        raise SystemExit(f"数据库不存在: {args.document_db}")

    if not 0. <= args.alpha <= 1.:
        raise SystemExit("--alpha 需要在 0 到 1 之间")
    use_dense = args.ranking == "dense" and not args.phrase
    needs_dense = args.ranking in {"dense", "hybrid", "rerank"} and not args.phrase
    if not use_dense and not args.index_db.is_file():
        raise SystemExit(f"数据库不存在: {args.index_db}")
    if needs_dense and not (args.dense_index / "metadata.json").is_file():
        raise SystemExit(f"Dense 索引不存在: {args.dense_index}")
    if args.ranking == "rerank" and not args.reranker_model.is_dir():
        raise SystemExit(f"Reranker 模型目录不存在: {args.reranker_model}")
    if args.rerank_candidates <= 0:
        raise SystemExit("--rerank-candidates 必须大于 0")
    if args.reranker_batch_size <= 0:
        raise SystemExit("--reranker-batch-size 必须大于 0")

    results = search_documents(
        args.document_db,
        args.index_db,
        args.query,
        args.limit,
        ranking="phrase" if args.phrase else args.ranking,
        dense_index=args.dense_index,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        reranker_batch_size=args.reranker_batch_size,
        device=args.device,
        alpha=args.alpha,
    )
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['title']}\n"
            f"   {result['url']}\n"
            f"   {result['snippet'][:160]}\n"
        )

    return 0


def format_rag_debug(response: dict[str, object]) -> str:
    """Render the RAG state and trace as a compact human-readable report."""

    action_labels = {
        "search": "继续检索",
        "calculate": "执行计算",
        "verified_answer": "答案通过校验",
        "rejected_answer": "答案未通过校验",
        "deterministic_sort": "确定性排序",
        "fallback_answer": "使用兜底答案",
        "invalid_response": "模型响应无效",
        "invalid_calculation": "计算请求无效",
        "no_results": "没有检索结果",
    }
    status_labels = {
        "answered": "已回答",
        "partial": "部分回答",
        "missing": "材料不足",
        "conflicting": "证据冲突",
    }
    status = str(response.get("status", "unknown"))
    header = f"状态: {status_labels.get(status, status)}"
    if isinstance(response.get("model_call_count"), int):
        header += f" | 模型调用: {response['model_call_count']}"
    lines = [header]

    requirements = response.get("requirements", [])
    if isinstance(requirements, list) and requirements:
        lines.append("答案槽位:")
        for item in requirements:
            if not isinstance(item, dict):
                continue
            marker = "✓" if item.get("status") == "answered" else "✗"
            lines.append(f"  {marker} {item.get('question', '未命名槽位')}")

    collection_trace = response.get("collection_trace")
    if isinstance(collection_trace, dict):
        filter_spec = collection_trace.get("filter", {})
        counts = collection_trace.get("counts", {})
        topics = collection_trace.get("topics", [])
        if isinstance(filter_spec, dict):
            lines.append(
                "集合过滤: "
                f"{filter_spec.get('field', '?')} "
                f"{filter_spec.get('operator', '?')} "
                f"{filter_spec.get('value', '?')}"
            )
        if isinstance(counts, dict):
            lines.append(
                "确定性计数: "
                + "，".join(f"{key}={value}" for key, value in counts.items())
            )
        if isinstance(topics, list):
            labels = [
                str(topic.get("label"))
                for topic in topics
                if isinstance(topic, dict) and topic.get("label")
            ]
            if labels:
                lines.append(
                    f"共同主题 ({collection_trace.get('topic_mode', 'unknown')}): "
                    + "、".join(labels)
                )
        return "\n".join(lines)

    trace = response.get("trace", [])
    if not isinstance(trace, list) or not trace:
        lines.append("没有 trace。")
        return "\n".join(lines)

    for index, item in enumerate(trace, start=1):
        if not isinstance(item, dict):
            continue
        cycle = item.get("cycle", index)
        action = str(item.get("action", "等待决策"))
        lines.append(
            f"\n第 {cycle} 轮 · {action_labels.get(action, action)}"
        )

        queries = item.get("search_queries", [])
        if isinstance(queries, list) and queries:
            lines.append("  搜索: " + " | ".join(map(str, queries)))

        new_results = item.get("new_results", [])
        if isinstance(new_results, list) and new_results:
            titles = [
                str(result.get("title") or result.get("url") or "未命名来源")
                for result in new_results[:3]
                if isinstance(result, dict)
            ]
            suffix = f" 等 {len(new_results)} 条" if len(new_results) > 3 else ""
            lines.append(f"  新证据: {'；'.join(titles)}{suffix}")

        evidence_count = item.get("evidence_count")
        token_count = item.get("context_tokens_estimate")
        if isinstance(evidence_count, int):
            context_summary = f"  State: 累计证据 {evidence_count} 条"
            if isinstance(token_count, int):
                context_summary += f"，上下文约 {token_count} tokens"
            lines.append(context_summary)

        next_queries = item.get("next_queries", [])
        if isinstance(next_queries, list) and next_queries:
            query_slots = item.get("query_slots", {})
            if not isinstance(query_slots, dict):
                query_slots = {}
            formatted_queries = [
                (
                    f"{query} [{query_slots[str(query)]}]"
                    if query_slots.get(str(query))
                    else str(query)
                )
                for query in next_queries
            ]
            lines.append("  下一步搜索: " + " | ".join(formatted_queries))

        answer_slots = item.get("answer_slots", [])
        if isinstance(answer_slots, list) and answer_slots:
            slot_states = [
                f"{slot.get('id', '?')}={slot.get('status', 'unknown')}"
                for slot in answer_slots
                if isinstance(slot, dict)
            ]
            if slot_states:
                lines.append("  槽位状态: " + " | ".join(slot_states))

        calculations = item.get("calculations", [])
        if isinstance(calculations, list):
            for calculation in calculations:
                if not isinstance(calculation, dict):
                    continue
                lines.append(
                    "  计算: "
                    f"{calculation.get('name', '未命名')} = "
                    f"{calculation.get('result')} "
                    f"({calculation.get('operator', 'unknown')}, "
                    f"{calculation.get('input_count', '?')} 项输入)"
                )

        verification = item.get("verification")
        if isinstance(verification, dict):
            issues = verification.get("issues", [])
            missing = [
                requirement.get("description", "未命名槽位")
                for requirement in verification.get("requirements", [])
                if isinstance(requirement, dict)
                and requirement.get("satisfied") is False
            ]
            if isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict):
                        lines.append(
                            f"  校验问题: {issue.get('description', issue)}"
                        )
            if missing:
                lines.append("  尚缺: " + "；".join(map(str, missing)))
            if verification.get("valid") is True:
                lines.append("  校验: 通过")

        if item.get("error"):
            lines.append(f"  错误: {item['error']}")

        for revision_name, label in (
            ("temporal_revision", "时间冲突修订"),
            ("final_revision", "最终修订"),
        ):
            revision = item.get(revision_name)
            if isinstance(revision, dict):
                outcome = "通过" if revision.get("valid") else "未通过"
                lines.append(f"  {label}: {outcome}")

    return "\n".join(lines)


def run_rag(args: argparse.Namespace) -> int:
    if not args.document_db.is_file():
        raise SystemExit(f"数据库不存在: {args.document_db}")
    if not args.index_db.is_file():
        raise SystemExit(f"数据库不存在: {args.index_db}")
    if not (args.dense_index / "metadata.json").is_file():
        raise SystemExit(f"Dense 索引不存在: {args.dense_index}")
    if not args.reranker_model.is_dir():
        raise SystemExit(f"Reranker 模型目录不存在: {args.reranker_model}")
    if not 0. <= args.alpha <= 1.:
        raise SystemExit("--alpha 需要在 0 到 1 之间")
    if args.top_k <= 0:
        raise SystemExit("--top-k 必须大于 0")
    if args.rerank_candidates <= 0:
        raise SystemExit("--rerank-candidates 必须大于 0")
    if args.reranker_batch_size <= 0:
        raise SystemExit("--reranker-batch-size 必须大于 0")
    if not 1 <= args.max_cycles <= 8:
        raise SystemExit("--max-cycles 必须在 1 到 8 之间")

    try:
        response = answer_question(
            args.document_db,
            args.index_db,
            args.query,
            top_k=args.top_k,
            dense_index=args.dense_index,
            device=args.device,
            alpha=args.alpha,
            reranker_model=args.reranker_model,
            rerank_candidates=args.rerank_candidates,
            reranker_batch_size=args.reranker_batch_size,
            max_cycles=args.max_cycles,
            debug=args.debug,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(f"回答\n{response['answer']}\n")
    if args.debug:
        print("调试信息")
        print(format_rag_debug(response))
        print()
    print("搜索结果")
    for rank, result in enumerate(response["results"], start=1):
        print(
            f"{rank}. {result['title']}\n"
            f"   {result['url']}\n"
            f"   {result['snippet']}\n"
        )
    return 0


def run_server(args: argparse.Namespace) -> int:
    try:
        serve(
            args.host,
            args.port,
            args.document_db,
            args.index_db,
            args.frontend,
            dense_index=args.dense_index,
            reranker_model=args.reranker_model,
            rerank_candidates=args.rerank_candidates,
            reranker_batch_size=args.reranker_batch_size,
            device=args.device,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"文件不存在: {error.args[0]}") from error
    except KeyboardInterrupt:
        pass
    return 0


def run_evaluator(args: argparse.Namespace) -> int:
    if not args.document_db.is_file():
        raise SystemExit(f"数据库不存在: {args.document_db}")
    is_rag = args.mode == "rag"
    if is_rag and args.ranking != "hybrid":
        raise SystemExit("RAG 固定使用 hybrid，--ranking 仅用于搜索评测")
    if (is_rag or args.ranking != "dense") and not args.index_db.is_file():
        raise SystemExit(f"数据库不存在: {args.index_db}")
    if (
        (is_rag or args.ranking in {"dense", "hybrid", "rerank"})
        and not (args.dense_index / "metadata.json").is_file()
    ):
        raise SystemExit(f"Dense 索引不存在: {args.dense_index}")
    if (
        (is_rag or args.ranking == "rerank")
        and not args.reranker_model.is_dir()
    ):
        raise SystemExit(f"Reranker 模型目录不存在: {args.reranker_model}")
    if args.rerank_candidates <= 0:
        raise SystemExit("--rerank-candidates 必须大于 0")
    if args.reranker_batch_size <= 0:
        raise SystemExit("--reranker-batch-size 必须大于 0")
    if not 0. <= args.alpha <= 1.:
        raise SystemExit("--alpha 需要在 0 到 1 之间")
    if is_rag and args.top_k <= 0:
        raise SystemExit("--top-k 必须大于 0")
    kwargs = {
        "base_url": args.base_url,
        "dense_index": args.dense_index,
        "device": args.device,
        "alpha": args.alpha,
    }
    if is_rag:
        kwargs["top_k"] = args.top_k
        kwargs["reranker_model"] = args.reranker_model
        kwargs["rerank_candidates"] = args.rerank_candidates
        kwargs["reranker_batch_size"] = args.reranker_batch_size
        evaluator = run_rag_evaluation
    else:
        kwargs.update({
            "ranking": args.ranking,
            "reranker_model": args.reranker_model,
            "rerank_candidates": args.rerank_candidates,
            "reranker_batch_size": args.reranker_batch_size,
        })
        evaluator = run_evaluation
    evaluator(
        args.document_db,
        args.index_db,
        **kwargs,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zeta-engine",
    )
    commands = parser.add_subparsers(required=True)

    crawl = commands.add_parser("crawl", help="运行爬虫")
    crawl.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    crawl.add_argument(
        "--queue-db",
        type=Path,
        default=Path("data/crawl_queue.db"),
    )
    crawl.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/zeta-engine.log"),
    )
    crawl.add_argument("--max-pages", type=int, default=500_000)
    crawl.add_argument("--workers", type=int, default=16)
    crawl.add_argument("--delay", type=float, default=0.2)
    crawl.add_argument(
        "--refresh-after-hours",
        type=float,
        help="重新抓取完成时间早于指定小时数的页面；0 表示全部",
    )
    crawl.add_argument(
        "--retry-failed",
        action="store_true",
        help="重新尝试历史失败任务",
    )
    crawl.set_defaults(handler=run_crawler)

    stats = commands.add_parser("stats", help="查看数据库统计")
    stats.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    stats.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/index.db"),
    )
    stats.set_defaults(handler=show_stats)

    index = commands.add_parser("index", help="构造索引")
    index.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    index.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/index.db"),
    )
    index.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/zeta-engine.log"),
    )
    index.add_argument(
        "--mode",
        choices=("default", "search"),
        default="default",
        help="jieba 分词模式",
    )
    index.set_defaults(handler=run_indexer)

    dense_index = commands.add_parser("dense-index", help="构造 Dense 向量索引")
    dense_index.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    dense_index.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    dense_index.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    dense_index.add_argument("--batch-size", type=int, default=32)
    dense_index.add_argument("--device")
    dense_index.add_argument(
        "--upgrade-v3",
        action="store_true",
        help="复用现有 v3 向量，只新增 passage 元数据和 BM25 索引",
    )
    dense_index.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/zeta-engine.log"),
    )
    dense_index.set_defaults(handler=run_dense_indexer)

    search = commands.add_parser("search", help="查询索引")
    search.add_argument("query", help="查询文本")
    search.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    search.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/index.db"),
    )
    search.add_argument("--phrase", action="store_true", help="精确短语查询")
    search.add_argument(
        "--ranking",
        choices=("bm25f", "dense", "hybrid", "rerank"),
        default="hybrid",
        help="普通查询的排名算法",
    )
    search.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    search.add_argument("--device")
    search.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    search.add_argument(
        "--reranker-model",
        type=Path,
        default=DEFAULT_RERANKER_MODEL_PATH,
    )
    search.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
    )
    search.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANK_BATCH_SIZE,
    )
    search.add_argument("--limit", type=int, default=20)
    search.set_defaults(handler=run_search)

    rag = commands.add_parser("rag", help="检索并生成回答")
    rag.add_argument("query", help="问题文本")
    rag.add_argument("--top-k", type=int, default=8)
    rag.add_argument("--max-cycles", type=int, default=AGENT_MAX_CYCLES)
    rag.add_argument("--debug", action="store_true")
    rag.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    rag.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/index.db"),
    )
    rag.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    rag.add_argument("--device")
    rag.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    rag.add_argument(
        "--reranker-model",
        type=Path,
        default=DEFAULT_RERANKER_MODEL_PATH,
    )
    rag.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
    )
    rag.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANK_BATCH_SIZE,
    )
    rag.set_defaults(handler=run_rag)

    server = commands.add_parser("serve", help="启动 Web 搜索服务")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    server.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/index.db"),
    )
    server.add_argument(
        "--frontend",
        type=Path,
        default=Path("index.html"),
    )
    server.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    server.add_argument("--device")
    server.add_argument(
        "--reranker-model",
        type=Path,
        default=DEFAULT_RERANKER_MODEL_PATH,
    )
    server.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
    )
    server.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANK_BATCH_SIZE,
    )
    server.set_defaults(handler=run_server)

    evaluation = commands.add_parser("eval", help="运行搜索或 RAG 评测")
    evaluation.add_argument(
        "--mode",
        choices=("search", "rag"),
        default="search",
    )
    evaluation.add_argument(
        "--document-db",
        type=Path,
        default=Path("data/zeta.db"),
    )
    evaluation.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/index.db"),
    )
    evaluation.add_argument("--base-url", default=DEFAULT_BASE_URL)
    evaluation.add_argument(
        "--ranking",
        choices=("bm25f", "dense", "hybrid", "rerank"),
        default="hybrid",
        help="搜索评测排名算法；RAG 固定使用 hybrid",
    )
    evaluation.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    evaluation.add_argument("--device")
    evaluation.add_argument("--top-k", type=int, default=8)
    evaluation.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    evaluation.add_argument(
        "--reranker-model",
        type=Path,
        default=DEFAULT_RERANKER_MODEL_PATH,
    )
    evaluation.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
    )
    evaluation.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANK_BATCH_SIZE,
    )
    evaluation.set_defaults(handler=run_evaluator)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)
