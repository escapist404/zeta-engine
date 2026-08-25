import logging
import queue, sqlite3, requests

from datetime import datetime
from zeta_engine.storage import *
from zeta_engine.utils import *
from zeta_engine.constants import *
from bs4 import BeautifulSoup
from time import sleep
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

def get_html(url: str, headers: dict = HEADERS, timeout: int = TIMEOUT) -> str | None:
    try:
        r = requests.get(url=url, headers=headers, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except requests.RequestException as error:
        logger.warning("获取页面失败: %s (%s)", url, error)
        return None

def crawl_url(connection: sqlite3.Connection, seed_url: str, allowed_domains: list[str] = None) -> None:
    q = queue.Queue()
    q.put(normalize_url(seed_url))
    visited = set()
    logger.info("开始爬取: %s", seed_url)

    while not q.empty():
        sleep(5)
        url = q.get()
        if url in visited:
            logger.debug("跳过已访问页面: %s", url)
            continue
        visited.add(url)

        html = get_html(url, HEADERS, TIMEOUT)
        if html is None:
            logger.info("跳过无法获取的页面: %s", url)
            continue

        soup = BeautifulSoup(html, 'html.parser')
        unwanted_tags = ["script", "style", "noscript"]
        for tag in soup.find_all(unwanted_tags, recursive=True):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)

        document_id = save_document(
            connection,
            Document(
                url=url,
                title=soup.title.string if soup.title else "",
                text=text,
                fetched_at=datetime.now().isoformat(),
            ),
        )
        logger.info("页面已保存: id=%s url=%s", document_id, url)

        href = soup.find_all('a', href=True)

        for tag in href:
            tag["href"] = tag["href"].strip()
            target_url = normalize_url(tag["href"] if is_absolute_url(tag["href"]) else urljoin(url, tag["href"]))
            if allowed_domains is None or any(target_url.startswith(domain) for domain in allowed_domains):
                if target_url not in visited and target_url.startswith(seed_url):
                    q.put(target_url)

    logger.info("爬取结束，共访问 %s 个页面", len(visited))
