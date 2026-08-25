import argparse
import logging
from pathlib import Path
from urllib.parse import urlsplit

from zeta_engine.constants import ALLOWED_DOMAINS, SEED_URLS
from zeta_engine.crawl_queue import connect_queue, create_queue_tables
from zeta_engine.crawler import crawl_urls
from zeta_engine.storage import connect, create_tables

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

def run_crawler(args: argparse.Namespace) -> int:
    args.document_db.parent.mkdir(parents=True, exist_ok=True)
    args.queue_db.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_file)

    with (
        connect(str(args.document_db)) as document_db,
        connect_queue(str(args.queue_db)) as queue_db,
    ):
        create_tables(document_db)
        create_queue_tables(queue_db)

        stats = crawl_urls(
            document_db,
            queue_db,
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

    with connect(str(args.document_db)) as db:
        results = [
            (
                host,
                db.execute(
                    """
                    SELECT COUNT(*)
                    FROM documents
                    WHERE url LIKE ? OR url LIKE ?
                    """,
                    (f"http://{host}/%", f"https://{host}/%"),
                ).fetchone()[0],
            )
            for host in hosts
        ]

    for host, count in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"{host}: {count} documents")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zeta-engine",
        description="人大站点爬虫",
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
    stats.set_defaults(handler=show_stats)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)
