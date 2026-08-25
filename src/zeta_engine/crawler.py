import logging
import sqlite3
from datetime import datetime
from queue import Queue
from threading import Lock, Thread
from time import monotonic, sleep
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from zeta_engine.constants import HEADERS, TIMEOUT
from zeta_engine.crawl_queue import (
    claim_pending_tasks,
    claim_task,
    complete_task,
    enqueue_tasks,
    fail_task,
    recover_tasks,
    task_counts,
)
from zeta_engine.storage import Document, save_document
from zeta_engine.utils import normalize_url


logger = logging.getLogger(__name__)


def get_html(
    url: str,
    headers: dict = HEADERS,
    timeout: int = TIMEOUT,
    session: requests.Session | None = None,
) -> str | None:
    try:
        client = session or requests
        response = client.get(url=url, headers=headers, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            logger.info("跳过非 HTML 页面: %s", url)
            return None

        response.encoding = response.apparent_encoding
        return response.text
    except requests.RequestException as error:
        logger.warning("获取页面失败: %s (%s)", url, error)
        return None


def extract_page(html: str, url: str) -> tuple[Document, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = [
        normalize_url(urljoin(url, tag["href"].strip()))
        for tag in soup.find_all("a", href=True)
        if tag["href"].strip()
    ]

    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    document = Document(
        url=url,
        title=title,
        text=soup.get_text(" ", strip=True),
        fetched_at=datetime.now().isoformat(),
    )
    return document, links


def crawl_urls(
    document_connection: sqlite3.Connection,
    queue_connection: sqlite3.Connection,
    seed_urls: list[str] | tuple[str, ...],
    allowed_domains: list[str] | tuple[str, ...] | None = None,
    *,
    max_pages: int = 100,
    download_workers: int = 4,
    per_host_delay: float = 1.0,
    max_attempts: int = 3,
) -> None:
    if max_pages <= 0:
        return
    if download_workers <= 0:
        raise ValueError("download_workers 必须大于 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts 必须大于 0")

    allowed_hosts = {
        urlsplit(normalize_url(url)).hostname
        for url in (allowed_domains or seed_urls)
    }
    allowed_hosts.discard(None)

    request_queue: Queue[str | None] = Queue()
    processing_queue: Queue[tuple[str, str | None]] = Queue(
        maxsize=download_workers * 2
    )
    host_next_request: dict[str, float] = {}
    cooldown_lock = Lock()
    scheduled_count = 0

    def schedule(urls: list[str] | tuple[str, ...]) -> None:
        nonlocal scheduled_count
        accepted = []
        for url in urls:
            normalized = normalize_url(url)
            parts = urlsplit(normalized)
            if (
                parts.scheme in {"http", "https"}
                and parts.hostname in allowed_hosts
            ):
                accepted.append(normalized)

        for url in enqueue_tasks(queue_connection, accepted):
            if scheduled_count >= max_pages:
                break
            if claim_task(queue_connection, url):
                scheduled_count += 1
                request_queue.put(url)

    def wait_for_host(url: str) -> None:
        host = urlsplit(url).hostname or ""
        with cooldown_lock:
            now = monotonic()
            ready_at = max(now, host_next_request.get(host, now))
            host_next_request[host] = ready_at + per_host_delay

        # ponytail: a worker waits here; add a scheduler only if hosts starve.
        sleep(max(0.0, ready_at - now))

    def download_worker() -> None:
        with requests.Session() as session:
            while True:
                url = request_queue.get()
                try:
                    if url is None:
                        return

                    wait_for_host(url)
                    processing_queue.put(
                        (url, get_html(url, session=session))
                    )
                except Exception as error:
                    logger.warning("下载页面失败: %s (%s)", url, error)
                    if url is not None:
                        processing_queue.put((url, None))
                finally:
                    request_queue.task_done()

    recovered = recover_tasks(queue_connection)
    if recovered:
        logger.info("恢复 %s 个中断任务", recovered)

    schedule(list(seed_urls))
    for url in claim_pending_tasks(
        queue_connection,
        max_pages - scheduled_count,
    ):
        scheduled_count += 1
        request_queue.put(url)

    if not scheduled_count:
        logger.info("没有待爬任务: %s", task_counts(queue_connection))
        return

    workers = [
        Thread(target=download_worker, name=f"downloader-{index}", daemon=True)
        for index in range(download_workers)
    ]
    for worker in workers:
        worker.start()

    logger.info("开始爬取，本次加载 %s 个任务", scheduled_count)
    processed = 0

    try:
        while processed < scheduled_count:
            url, html = processing_queue.get()
            try:
                if html is None:
                    state = fail_task(
                        queue_connection,
                        url,
                        "获取页面失败",
                        max_attempts,
                    )
                    logger.info("页面未保存: state=%s url=%s", state, url)
                    continue

                document, links = extract_page(html, url)
                document_id = save_document(document_connection, document)
                schedule(links)
                complete_task(queue_connection, url)
                logger.info("页面已保存: id=%s url=%s", document_id, url)
            except Exception as error:
                state = fail_task(
                    queue_connection,
                    url,
                    str(error),
                    max_attempts,
                )
                logger.warning(
                    "处理页面失败: state=%s url=%s (%s)",
                    state,
                    url,
                    error,
                )
            finally:
                processed += 1
                processing_queue.task_done()
    except KeyboardInterrupt:
        recovered = recover_tasks(queue_connection)
        logger.info("爬取已中断，%s 个任务等待下次继续", recovered)
        raise

    for _ in workers:
        request_queue.put(None)
    request_queue.join()
    for worker in workers:
        worker.join()

    logger.info(
        "爬取结束，共处理 %s 个页面，队列状态: %s",
        processed,
        task_counts(queue_connection),
    )


def crawl_url(
    document_connection: sqlite3.Connection,
    queue_connection: sqlite3.Connection,
    seed_url: str,
    allowed_domains: list[str] | tuple[str, ...] | None = None,
    **options,
) -> None:
    crawl_urls(
        document_connection,
        queue_connection,
        [seed_url],
        allowed_domains=allowed_domains,
        **options,
    )
