import logging
from pathlib import Path

from zeta_engine.constants import SEED_URLS
from zeta_engine.crawl_queue import connect_queue, create_queue_tables
from zeta_engine.crawler import crawl_urls
from zeta_engine.storage import connect, create_tables

log_file = Path("logs/zeta-engine.log")
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

with (
    connect("data/zeta.db") as document_db,
    connect_queue("data/crawl_queue.db") as queue_db,
):
    create_tables(document_db)
    create_queue_tables(queue_db)
    crawl_urls(
        document_db,
        queue_db,
        SEED_URLS,
        allowed_domains=SEED_URLS,
        max_pages=500000,
        download_workers=8,
        per_host_delay=0.5,
    )
