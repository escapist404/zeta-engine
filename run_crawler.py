import logging
from pathlib import Path

import zeta_engine

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

db = zeta_engine.storage.connect("data/zeta.db")
zeta_engine.storage.create_tables(db)
zeta_engine.crawler.crawl_url(db, "http://ai.ruc.edu.cn/", allowed_domains=zeta_engine.constants.SEED_URLS)
