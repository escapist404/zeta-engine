import argparse
import logging
from pathlib import Path
from urllib.parse import urlsplit

from zeta_engine.constants import ALLOWED_DOMAINS, SEED_URLS
from zeta_engine.crawler import crawl_urls
from zeta_engine.index import build_index
from zeta_engine.search import search_bm25f, search_query, search_tf_idf
from zeta_engine.storage import Storage


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
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


def run_search(args: argparse.Namespace) -> int:
    for database in (args.document_db, args.index_db):
        if not database.is_file():
            raise SystemExit(f"数据库不存在: {database}")

    with Storage(
        document_db=args.document_db,
        index_db=args.index_db,
    ) as storage:
        if args.phrase:
            document_ids = search_query(storage, args.query, phrase=True)
        elif args.ranking == "tf-idf":
            document_ids = search_tf_idf(storage, args.query)
        elif args.ranking == "bm25f":
            document_ids = search_bm25f(storage, args.query)
        else:
            document_ids = search_query(storage, args.query)
        for rank, document_id in enumerate(document_ids[:args.limit], start=1):
            document = storage.documents.get(document_id)
            if document is None:
                continue
            url, title, text, _fetched_at = document
            snippet = " ".join(text.split())[:160]
            print(f"{rank}. {title}\n   {url}\n   {snippet}")

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
        choices=("simple", "tf-idf", "bm25f"),
        default="simple",
        help="普通查询的排名算法",
    )
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(handler=run_search)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)
