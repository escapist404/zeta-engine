import unittest
from unittest.mock import Mock, patch

from zeta_engine.ingestion.crawler import SKIPPED_PAGE, crawl_urls, get_html
from zeta_engine.infrastructure.storage import Storage


PAGES = {
    "https://a.test/": """
        <html><head><title>A</title></head>
        <body>
            <a href="/next#part">下一页</a>
            <a href="https://outside.test/ad">站外广告</a>
        </body></html>
    """,
    "https://a.test/next": "<html><title>A2</title><body>正文</body></html>",
    "https://b.test/": "<html><title>B</title><body>正文</body></html>",
}


class CrawlerTest(unittest.TestCase):
    @patch(
        "zeta_engine.ingestion.crawler.get_html",
        side_effect=lambda url, *_args, **_kwargs: (url, PAGES[url]),
    )
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

    @patch(
        "zeta_engine.ingestion.crawler.get_html",
        side_effect=[
            None,
            (
                "https://a.test/",
                "<html><title>A</title><body>正文</body></html>",
            ),
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

    def test_get_html_checks_requested_and_final_hosts(self) -> None:
        external_session = Mock()
        self.assertIs(
            get_html(
                "https://outside.test/start",
                {"a.test"},
                session=external_session,
            ),
            SKIPPED_PAGE,
        )
        external_session.get.assert_not_called()

        allowed_response = Mock(
            url="https://a.test/final",
            headers={"Content-Type": "text/html"},
            apparent_encoding="utf-8",
            text="<html><body>正文</body></html>",
        )
        allowed_session = Mock()
        allowed_session.get.return_value = allowed_response

        self.assertEqual(
            get_html(
                "https://a.test/start",
                {"a.test"},
                session=allowed_session,
            ),
            (
                "https://a.test/final",
                "<html><body>正文</body></html>",
            ),
        )
        allowed_session.get.assert_called_once()

        redirected_session = Mock()
        redirected_session.get.return_value = Mock(
            url="https://outside.test/final",
            headers={"Content-Type": "text/html"},
        )
        self.assertIs(
            get_html(
                "https://a.test/start",
                {"a.test"},
                session=redirected_session,
            ),
            SKIPPED_PAGE,
        )


if __name__ == "__main__":
    unittest.main()
