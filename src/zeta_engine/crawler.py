import logging
from collections import deque
from datetime import datetime, timezone
from queue import Queue
from threading import Condition, Lock, Thread
from time import monotonic, sleep
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree
from url_normalize import url_normalize

from zeta_engine.constants import (
    CONTENT_SELECTORS,
    EXCLUDED_HTML_SELECTORS,
    HEADERS,
    SOCIAL_TITLE_SELECTOR,
    TIMEOUT,
    TITLE_SELECTORS,
)
from zeta_engine.storage import Storage

logger = logging.getLogger(__name__)
SKIPPED_PAGE = object()


def normalize_url(url: str) -> str:
    return urldefrag(url_normalize(url)).url


def get_html(
    url: str,
    allowed_hosts: set[str],
    headers: dict = HEADERS,
    timeout: int = TIMEOUT,
    session: requests.Session | None = None,
) -> tuple[str, str] | None | object:
    try:
        normalized_url = normalize_url(url)
        if urlsplit(normalized_url).hostname not in allowed_hosts:
            logger.info("跳过域名范围外的请求: %s", normalized_url)
            return SKIPPED_PAGE

        client = session or requests
        response = client.get(url=normalized_url, headers=headers, timeout=timeout)
        final_url = normalize_url(response.url)
        if urlsplit(final_url).hostname not in allowed_hosts:
            logger.info("跳过域名范围外的最终页面: %s", final_url)
            return SKIPPED_PAGE

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            logger.info("跳过非 HTML 页面: %s", url)
            return SKIPPED_PAGE

        response.encoding = response.apparent_encoding
        return final_url, response.text
    except requests.RequestException as error:
        logger.warning("获取页面失败: %s (%s)", url, error)
        return None


def extract_page(html: str, url: str) -> tuple[tuple[str, str, str, str], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = [
        normalize_url(urljoin(url, tag["href"].strip()))
        for tag in soup.find_all("a", href=True)
        if tag["href"].strip()
    ]

    headline = next(
        filter(None, (soup.select_one(selector) for selector in TITLE_SELECTORS)),
        None,
    )
    social_title = soup.select_one(SOCIAL_TITLE_SELECTOR)
    title = (
        headline.get_text(" ", strip=True)
        if headline
        else social_title["content"].strip()
        if social_title
        else soup.title.get_text(" ", strip=True)
        if soup.title
        else ""
    )

    for tag in soup.select(EXCLUDED_HTML_SELECTORS):
        tag.decompose()

    content = soup.body or soup
    for selector in CONTENT_SELECTORS:
        if selected := soup.select_one(selector):
            content = selected
            break
    document = (
        url,
        title,
        content.get_text(" ", strip=True),
        datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )
    return document, links


def extract_page_resiliparse(
    html: str,
    url: str,
) -> tuple[tuple[str, str, str, str], list[str]]:
    tree = HTMLTree.parse(html)
    root = tree.document
    links = []
    for link in root.query_selector_all("a[href]"):
        href = link.getattr("href").strip()
        if href:
            links.append(normalize_url(urljoin(url, href)))

    title = ""
    for selector in TITLE_SELECTORS:
        headline = root.query_selector(selector)
        if headline is not None and (title := " ".join(headline.text.split())):
            break

    if not title:
        social_title = root.query_selector(SOCIAL_TITLE_SELECTOR)
        if social_title is not None:
            title = social_title.getattr("content").strip()
    if not title:
        title = " ".join((tree.title or "").split())

    content = None
    for selector in CONTENT_SELECTORS:
        if content := root.query_selector(selector):
            break
    text = extract_plain_text(
        content.html if content is not None else tree,
        main_content=content is None,
        preserve_formatting=False,
    )
    document = (
        url,
        title,
        text,
        datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )
    return document, links


def crawl_urls(
    storage: Storage,
    seed_urls: list[str] | tuple[str, ...],
    allowed_domains: list[str] | tuple[str, ...] | None = None,
    *,
    max_pages: int = 100,
    download_workers: int = 4,
    per_host_delay: float = 1.0,
    max_attempts: int = 3,
    extractor: str = "beautifulsoup",
) -> dict[str, int]:
    stats = {
        "scheduled": 0,
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "retried": 0,
        "failed": 0,
    }
    if max_pages <= 0:
        return stats
    if download_workers <= 0:
        raise ValueError("download_workers 必须大于 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts 必须大于 0")
    extract = {
        "beautifulsoup": extract_page,
        "resiliparse": extract_page_resiliparse,
    }.get(extractor)
    if extract is None:
        raise ValueError(f"不支持的页面抽取器: {extractor}")
    if storage.documents is None or storage.queue is None:
        raise ValueError("爬虫需要 document_db 和 queue_db")

    documents = storage.documents
    queue = storage.queue

    allowed_hosts = {
        urlsplit(normalize_url(url)).hostname
        for url in (allowed_domains or seed_urls)
    }
    allowed_hosts.discard(None)

    request_queue: deque[str | None] = deque()
    request_condition = Condition()
    processing_queue: Queue[
        tuple[str, tuple[str, str] | None | object]
    ] = Queue(
        maxsize=download_workers * 2
    )
    host_next_request: dict[str, float] = {}
    cooldown_lock = Lock()
    scheduled_count = 0
    outstanding = 0

    def push_request(url: str | None, *, priority: bool = False) -> None:
        with request_condition:
            if priority:
                request_queue.appendleft(url)
            else:
                request_queue.append(url)
            request_condition.notify()

    def pop_request() -> str | None:
        with request_condition:
            while not request_queue:
                request_condition.wait()
            return request_queue.popleft()

    def schedule(urls: list[str] | tuple[str, ...]) -> None:
        nonlocal outstanding, scheduled_count
        accepted = []
        for url in urls:
            normalized = normalize_url(url)
            parts = urlsplit(normalized)
            if (
                parts.scheme in {"http", "https"}
                and parts.hostname in allowed_hosts
            ):
                accepted.append(normalized)

        for url in queue.enqueue(accepted):
            if scheduled_count >= max_pages:
                break
            if queue.claim(url):
                scheduled_count += 1
                stats["scheduled"] = scheduled_count
                outstanding += 1
                push_request(url)

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
                url = pop_request()
                try:
                    if url is None:
                        return

                    wait_for_host(url)
                    processing_queue.put(
                        (
                            url,
                            get_html(
                                url,
                                allowed_hosts,
                                session=session,
                            ),
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning("下载页面失败: %s (%s)", url, error)
                    if url is not None:
                        processing_queue.put((url, None))

    recovered = queue.recover()
    if recovered:
        logger.info("恢复 %s 个中断任务", recovered)

    schedule(list(seed_urls))

    for url in queue.claim_pending(max_pages - scheduled_count):
        scheduled_count += 1
        stats["scheduled"] = scheduled_count
        outstanding += 1
        push_request(url)

    if not scheduled_count:
        logger.info("没有待爬任务: %s", queue.count_by_state())
        logger.info("本次统计: %s", stats)
        return stats

    workers = [
        Thread(target=download_worker, name=f"downloader-{index}", daemon=True)
        for index in range(download_workers)
    ]
    for worker in workers:
        worker.start()

    logger.info("开始爬取，本次加载 %s 个任务", scheduled_count)

    def retry_or_fail(url: str, error: str) -> None:
        nonlocal outstanding
        state = queue.fail(
            url,
            error,
            max_attempts,
        )
        if state == "pending":
            queue.claim(url)
            push_request(url, priority=True)
            outstanding += 1
            stats["retried"] += 1
            logger.info("页面重试: %s (重试数=%s)", url, stats["retried"])
        else:
            stats["failed"] += 1
            logger.info("页面未保存: state=%s url=%s", state, url)

    try:
        while outstanding:
            url, result = processing_queue.get()
            try:
                outstanding -= 1
                stats["processed"] += 1
                if result is SKIPPED_PAGE:
                    queue.complete(url)
                    stats["skipped"] += 1
                    logger.info("页面已跳过: %s", url)
                    continue

                if result is None:
                    retry_or_fail(url, "获取页面失败")
                    continue

                final_url, html = result
                document, links = extract(html, final_url)
                document_id = documents.save(
                    url=document[0],
                    title=document[1],
                    text=document[2],
                    fetched_at=document[3],
                )
                schedule(links)
                queue.complete(url)
                stats["saved"] += 1
                logger.info(
                    "页面已保存: id=%s url=%s",
                    document_id,
                    final_url,
                )
            except Exception as error:  # noqa: BLE001
                retry_or_fail(url, str(error))
                logger.warning(
                    "处理页面失败: url=%s (%s)",
                    url,
                    error,
                )
            finally:
                processing_queue.task_done()
                if stats["processed"] % 100 == 0:
                    logger.info("爬取进度: %s", stats)
    except KeyboardInterrupt:
        recovered = queue.recover()
        logger.info("爬取已中断，%s 个任务等待下次继续", recovered)
        raise

    for _ in workers:
        push_request(None)
    for worker in workers:
        worker.join()

    logger.info(
        "爬取结束: %s，队列状态: %s",
        stats,
        queue.count_by_state(),
    )
    return stats
