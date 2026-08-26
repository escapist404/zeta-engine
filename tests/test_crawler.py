import unittest
from unittest.mock import patch

from zeta_engine.crawler import SKIPPED_PAGE, crawl_urls, extract_page
from zeta_engine.storage import Storage


PAGES = {
    "https://a.test/": """
        <html><head><title>A</title></head>
        <body><a href="/next#part">下一页</a></body></html>
    """,
    "https://a.test/next": "<html><title>A2</title><body>正文</body></html>",
    "https://b.test/": "<html><title>B</title><body>正文</body></html>",
}


class CrawlerTest(unittest.TestCase):
    def test_extract_page_keeps_links_but_indexes_only_main_content(self) -> None:
        cases = (
            (
                "article",
                '<div id="articleDiv"><a href="/content">详情正文</a></div>',
                "详情正文",
            ),
            (
                "listing",
                '<ul class="notice_list"><li><a href="/content">公告正文</a></li></ul>',
                "公告正文",
            ),
            (
                "fallback",
                '<main><a href="/content">通用正文</a></main>',
                "通用正文",
            ),
        )

        for name, content, expected_text in cases:
            with self.subTest(name=name):
                html = f"""
                    <html><head><title>页面标题</title></head><body>
                    <header class="main-header"><a href="/nav">导航</a></header>
                    {content}
                    <div class="footer"><a href="/footer">页脚</a></div>
                    </body></html>
                """
                document, links = extract_page(html, "https://a.test/")

                self.assertEqual(document[1], "页面标题")
                self.assertEqual(document[2], expected_text)
                self.assertEqual(
                    links,
                    [
                        "https://a.test/nav",
                        "https://a.test/content",
                        "https://a.test/footer",
                    ],
                )

    @patch("zeta_engine.crawler.get_html", side_effect=lambda url, **_: PAGES[url])
    def test_two_queues_crawl_and_save_discovered_pages(self, _get_html) -> None:
        with Storage(
            document_db=":memory:",
            queue_db=":memory:",
        ) as storage:
            stats = crawl_urls(
                storage,
                ["https://a.test/", "https://b.test/"],
                max_pages=3,
                download_workers=2,
                per_host_delay=0,
            )

            assert storage.documents is not None
            assert storage.queue is not None
            self.assertEqual(
                {row[1] for row in storage.documents.iter_all()},
                set(PAGES),
            )
            self.assertEqual(storage.queue.count_by_state(), {"done": 3})
            self.assertEqual(stats["scheduled"], 3)
            self.assertEqual(stats["processed"], 3)
            self.assertEqual(stats["saved"], 3)

    @patch("zeta_engine.crawler.get_html", side_effect=lambda url, **_: PAGES[url])
    def test_restart_continues_pending_tasks(self, get_html) -> None:
        with Storage(
            document_db=":memory:",
            queue_db=":memory:",
        ) as storage:
            crawl_urls(
                storage,
                ["https://a.test/", "https://b.test/"],
                max_pages=1,
                per_host_delay=0,
            )
            assert storage.queue is not None
            self.assertEqual(
                storage.queue.count_by_state(),
                {"done": 1, "pending": 2},
            )

            storage.queue.claim_pending(1)
            self.assertEqual(
                storage.queue.count_by_state(),
                {"done": 1, "pending": 1, "processing": 1},
            )

            crawl_urls(
                storage,
                ["https://a.test/", "https://b.test/"],
                max_pages=10,
                per_host_delay=0,
            )

            self.assertEqual(storage.queue.count_by_state(), {"done": 3})
            self.assertEqual(get_html.call_count, 3)

    @patch(
        "zeta_engine.crawler.get_html",
        side_effect=[
            None,
            "<html><title>A</title><body>正文</body></html>",
        ],
    )
    def test_failed_page_is_retried_and_counted(self, _get_html) -> None:
        with Storage(
            document_db=":memory:",
            queue_db=":memory:",
        ) as storage:
            stats = crawl_urls(
                storage,
                ["https://a.test/"],
                max_pages=1,
                download_workers=1,
                per_host_delay=0,
            )

            assert storage.queue is not None
            self.assertEqual(stats["processed"], 2)
            self.assertEqual(stats["retried"], 1)
            self.assertEqual(stats["saved"], 1)
            self.assertEqual(storage.queue.count_by_state(), {"done": 1})

    @patch("zeta_engine.crawler.get_html", return_value=SKIPPED_PAGE)
    def test_skipped_non_html_is_not_retried(self, _get_html) -> None:
        with Storage(
            document_db=":memory:",
            queue_db=":memory:",
        ) as storage:
            stats = crawl_urls(
                storage,
                ["https://a.test/"],
                max_pages=1,
                download_workers=1,
                per_host_delay=0,
            )

            assert storage.queue is not None
            self.assertEqual(stats["processed"], 1)
            self.assertEqual(stats["skipped"], 1)
            self.assertEqual(stats["retried"], 0)
            self.assertEqual(storage.queue.count_by_state(), {"done": 1})


if __name__ == "__main__":
    unittest.main()
