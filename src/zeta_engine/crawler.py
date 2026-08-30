import logging
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from html import escape
from math import isfinite
from queue import Queue
from threading import Condition, Lock, Thread
from time import monotonic, sleep
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Comment
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
from zeta_engine.extraction_rules import resolve_extraction_rule
from zeta_engine.storage import Storage

logger = logging.getLogger(__name__)
SKIPPED_PAGE = object()
MINIMAL_HTML_TAGS = frozenset({
    "a", "article", "blockquote", "br", "caption", "code", "dd", "dl", "dt",
    "em", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "img", "li", "main", "ol", "p", "pre", "section", "strong",
    "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
})
MINIMAL_HTML_DROP_TAGS = frozenset({
    "button", "canvas", "iframe", "input", "noscript", "object",
    "script", "select", "style", "svg", "template", "textarea",
})
UNIVERSAL_EXCLUDED_HTML_SELECTORS = (
    "script, style, noscript, template, [hidden], [aria-hidden='true']"
)


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


def _minimal_content_html(content: object, url: str) -> str:
    """Return a safe semantic HTML subset without interpreting page meaning."""

    fragment = BeautifulSoup(str(content), "html.parser")
    for comment in fragment.find_all(string=lambda item: isinstance(item, Comment)):
        comment.extract()
    for tag in fragment.find_all(MINIMAL_HTML_DROP_TAGS):
        tag.decompose()

    for tag in list(fragment.find_all(True)):
        name = tag.name.lower()
        if name == "b":
            name = tag.name = "strong"
        elif name == "i":
            name = tag.name = "em"
        if name not in MINIMAL_HTML_TAGS:
            tag.unwrap()
            continue

        attributes = {}
        if name == "a" and (href := str(tag.get("href", "")).strip()):
            absolute = urljoin(url, href)
            if urlsplit(absolute).scheme in {"http", "https"}:
                attributes["href"] = normalize_url(absolute)
        elif name == "img" and (alt := str(tag.get("alt", "")).strip()):
            attributes["alt"] = alt
        elif name in {"td", "th"}:
            for attribute in ("colspan", "rowspan"):
                value = str(tag.get(attribute, "")).strip()
                if value.isdigit() and int(value) > 1:
                    attributes[attribute] = value
        elif name == "ol":
            start = str(tag.get("start", "")).strip()
            if re.fullmatch(r"-?\d+", start):
                attributes["start"] = start
        tag.attrs = attributes

    for text in fragment.find_all(string=True):
        if text.parent and text.parent.name in {"code", "pre"}:
            continue
        text.replace_with(re.sub(r"\s+", " ", str(text)))

    return fragment.decode(formatter="minimal").strip()


def _extract_page_beautifulsoup(
    html: str,
    url: str,
) -> tuple[tuple[str, str, str, str, str], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    rule = resolve_extraction_rule(url)
    links = [
        normalize_url(urljoin(url, tag["href"].strip()))
        for tag in soup.find_all("a", href=True)
        if tag["href"].strip()
    ]

    headline = next(
        filter(None, (
            soup.select_one(selector)
            for selector in (
                (rule.title_selectors if rule is not None else ())
                + TITLE_SELECTORS
            )
        )),
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

    content = soup.body or soup
    used_site_selector = False
    for selector in rule.content_selectors if rule is not None else ():
        if selected := soup.select_one(selector):
            content = selected
            used_site_selector = True
            break
    if not used_site_selector:
        for selector in CONTENT_SELECTORS:
            if selected := soup.select_one(selector):
                content = selected
                break
    excluded_selectors = (
        UNIVERSAL_EXCLUDED_HTML_SELECTORS
        if used_site_selector
        else EXCLUDED_HTML_SELECTORS
    )
    if rule is not None:
        excluded_selectors += ", " + ", ".join(rule.excluded_selectors)
    for tag in content.select(excluded_selectors):
        tag.decompose()
    text = content.get_text(" ", strip=True)
    content_html = _minimal_content_html(content, url)
    document = (
        url,
        title,
        text,
        datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        content_html or (f"<p>{escape(text)}</p>" if text else ""),
    )
    return document, links


def _extract_page_resiliparse(
    html: str,
    url: str,
) -> tuple[tuple[str, str, str, str, str], list[str]]:
    tree = HTMLTree.parse(html)
    root = tree.document
    rule = resolve_extraction_rule(url)
    links = [
        normalize_url(urljoin(url, href))
        for tag in root.query_selector_all("a[href]")
        if (href := tag.getattr("href").strip())
    ]

    title = ""
    for selector in (
        (rule.title_selectors if rule is not None else ())
        + TITLE_SELECTORS
    ):
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
    for selector in (
        (rule.content_selectors if rule is not None else ())
        + CONTENT_SELECTORS
    ):
        if (selected := root.query_selector(selector)) is not None:
            content = selected
            break
    text = extract_plain_text(
        content.html if content is not None else tree,
        main_content=content is None,
        preserve_formatting=False,
    )
    if not text.strip():
        text = extract_plain_text(
            tree,
            main_content=False,
            preserve_formatting=False,
        )

    content_html = _minimal_content_html(
        content.html if content is not None else html,
        url,
    )
    document = (
        url,
        title,
        text,
        datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        content_html or (f"<p>{escape(text)}</p>" if text else ""),
    )
    return document, links


def extract_page(
    html: str,
    url: str,
) -> tuple[tuple[str, str, str, str, str], list[str]]:
    if "/addons/video/" in urlsplit(url).path:
        return _extract_page_resiliparse(html, url)

    document, links = _extract_page_beautifulsoup(html, url)
    if document[2].strip():
        return document, links
    return _extract_page_resiliparse(html, url)


def crawl_urls(
    storage: Storage,
    seed_urls: list[str] | tuple[str, ...],
    allowed_domains: list[str] | tuple[str, ...] | None = None,
    *,
    max_pages: int = 100,
    download_workers: int = 4,
    per_host_delay: float = 1.0,
    max_attempts: int = 3,
    refresh_after_hours: float | None = None,
    retry_failed: bool = False,
) -> dict[str, int]:
    stats = {
        "scheduled": 0,
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "retried": 0,
        "failed": 0,
        "refreshed": 0,
        "requeued_failed": 0,
    }
    if max_pages <= 0:
        return stats
    if download_workers <= 0:
        raise ValueError("download_workers 必须大于 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts 必须大于 0")
    if refresh_after_hours is not None and (
        not isfinite(refresh_after_hours) or refresh_after_hours < 0
    ):
        raise ValueError("refresh_after_hours 必须是有限的非负数")
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

    done_before = (
        (
            datetime.now(timezone.utc)
            - timedelta(hours=refresh_after_hours)
        ).isoformat()
        if refresh_after_hours is not None
        else None
    )
    requeued = queue.requeue(
        done_before=done_before,
        failed=retry_failed,
    )
    stats["refreshed"] = requeued["done"]
    stats["requeued_failed"] = requeued["failed"]
    if any(requeued.values()):
        logger.info("重新排入爬取队列: %s", requeued)

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
                document, links = extract_page(html, final_url)
                document_id = documents.save(
                    url=document[0],
                    title=document[1],
                    text=document[2],
                    fetched_at=document[3],
                    content_html=document[4],
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
