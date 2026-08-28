import argparse
import logging
from pathlib import Path
from urllib.parse import urlsplit

from zeta_engine.constants import ALLOWED_DOMAINS, SEED_URLS
from zeta_engine.crawler import crawl_urls
from zeta_engine.dense import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL_PATH,
    build_dense_index,
    search_dense,
)
from zeta_engine.eval import DEFAULT_BASE_URL, run_evaluation
from zeta_engine.index import build_index
from zeta_engine.search import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL_PATH,
    search_bm25f,
    search_hybrid,
    search_phrase,
    search_reranked,
)
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
            extractor=args.extractor,
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
    if not args.model.is_dir():
        raise SystemExit(f"Dense 模型目录不存在: {args.model}")

    configure_logging(args.log_file)
    with Storage(document_db=args.document_db) as storage:
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

    with Storage(
        document_db=args.document_db,
        index_db=None if use_dense else args.index_db,
    ) as storage:
        snippets = {}
        if use_dense:
            hits = search_dense(
                args.query,
                args.dense_index,
                limit=args.limit,
                device=args.device,
            )
            document_ids = [hit.document_id for hit in hits]
            snippets = {hit.document_id: hit.snippet for hit in hits}
        elif args.phrase:
            document_ids = search_phrase(storage, args.query)
        elif args.ranking == "hybrid":
            document_ids = search_hybrid(
                storage,
                args.query,
                args.dense_index,
                limit=args.limit,
                alpha=args.alpha,
                device=args.device,
            )
        elif args.ranking == "rerank":
            document_ids = search_reranked(
                storage,
                args.query,
                args.dense_index,
                reranker_model=args.reranker_model,
                limit=args.limit,
                candidate_limit=args.rerank_candidates,
                batch_size=args.reranker_batch_size,
                alpha=args.alpha,
                device=args.device,
            )
        else:
            document_ids = search_bm25f(storage, args.query)
        for rank, document_id in enumerate(document_ids[:args.limit], start=1):
            document = storage.documents.get(document_id)
            if document is None:
                continue
            url, title, text, _fetched_at = document
            snippet = snippets.get(document_id, " ".join(text.split()))[:160]
            print(f"{rank}. {title}\n   {url}\n   {snippet}\n")

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
    if args.ranking != "dense" and not args.index_db.is_file():
        raise SystemExit(f"数据库不存在: {args.index_db}")
    if (
        args.ranking in {"dense", "hybrid", "rerank"}
        and not (args.dense_index / "metadata.json").is_file()
    ):
        raise SystemExit(f"Dense 索引不存在: {args.dense_index}")
    if args.ranking == "rerank" and not args.reranker_model.is_dir():
        raise SystemExit(f"Reranker 模型目录不存在: {args.reranker_model}")
    if args.rerank_candidates <= 0:
        raise SystemExit("--rerank-candidates 必须大于 0")
    if args.reranker_batch_size <= 0:
        raise SystemExit("--reranker-batch-size 必须大于 0")
    if not 0. <= args.alpha <= 1.:
        raise SystemExit("--alpha 需要在 0 到 1 之间")
    run_evaluation(
        args.document_db,
        args.index_db,
        base_url=args.base_url,
        ranking=args.ranking,
        dense_index=args.dense_index,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        reranker_batch_size=args.reranker_batch_size,
        device=args.device,
        alpha=args.alpha,
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
        "--extractor",
        choices=("beautifulsoup", "resiliparse"),
        default="beautifulsoup",
        help="网页内容抽取器",
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
    search.add_argument("--alpha", type=float, default=.5)
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

    evaluation = commands.add_parser("eval", help="运行 MRR@20 评测")
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
    )
    evaluation.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )
    evaluation.add_argument("--device")
    evaluation.add_argument("--alpha", type=float, default=.5)
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
