import unittest
from unittest.mock import Mock, patch

from zeta_engine.crawler import (
    SKIPPED_PAGE,
    crawl_urls,
    extract_page,
    get_html,
)
from zeta_engine.storage import Storage


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
    def test_extract_page_keeps_only_minimal_semantic_html(self) -> None:
        html = """
            <html><body><main class="layout" onclick="alert(1)">
                <h2 data-id="x">申诉处理</h2>
                <div><p>十日内<strong style="color:red">提交</strong>。</p></div>
                <table class="grid"><tr><th colspan="2">期限</th></tr></table>
                <a href="/details" target="_blank">详情</a>
                <a href="javascript:alert(1)">危险链接</a>
                <script>不可信脚本</script>
            </main></body></html>
        """

        document, _links = extract_page(html, "https://a.test/rules")
        content_html = document[4]

        self.assertIn("<main>", content_html)
        self.assertIn("<h2>申诉处理</h2>", content_html)
        self.assertIn("<p>十日内<strong>提交</strong>。</p>", content_html)
        self.assertIn('<th colspan="2">期限</th>', content_html)
        self.assertIn('href="https://a.test/details"', content_html)
        self.assertNotIn("<div", content_html)
        self.assertNotIn("class=", content_html)
        self.assertNotIn("onclick", content_html)
        self.assertNotIn("javascript:", content_html)
        self.assertNotIn("不可信脚本", content_html)

    def test_extract_page_routes_to_resiliparse_only_when_needed(self) -> None:
        html = """
            <html><head><title>页面标题</title></head><body>
                <div class="notice_list"></div>
                <section><a href="/detail">Resiliparse 正文</a></section>
            </body></html>
        """

        with patch("zeta_engine.crawler._extract_page_beautifulsoup") as parser:
            video, video_links = extract_page(
                html,
                "https://gsai.ruc.edu.cn/addons/video/video/play.html?id=1",
            )
            parser.assert_not_called()
        fallback, fallback_links = extract_page(html, "https://a.test/empty")
        with patch("zeta_engine.crawler._extract_page_resiliparse") as parser:
            beautifulsoup, _ = extract_page(
                "<html><body><main>BeautifulSoup 正文</main></body></html>",
                "https://a.test/article",
            )
            parser.assert_not_called()

        self.assertIn("Resiliparse 正文", video[2])
        self.assertIn("Resiliparse 正文", fallback[2])
        self.assertEqual(beautifulsoup[2], "BeautifulSoup 正文")
        self.assertEqual(video_links, ["https://gsai.ruc.edu.cn/detail"])
        self.assertEqual(fallback_links, ["https://a.test/detail"])

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

    def test_extract_page_prefers_content_title_and_semantic_body(self) -> None:
        html = """
            <html>
            <head>
                <title>学院新闻 - 中国人民大学</title>
                <meta property="og:title" content="社交分享标题">
            </head>
            <body>
                <header>站点导航</header>
                <main>
                    <h1>真正的文章标题</h1>
                    <p>文章正文</p>
                </main>
                <footer>版权信息</footer>
            </body>
            </html>
        """

        document, _links = extract_page(html, "https://a.test/article")

        self.assertEqual(document[1], "真正的文章标题")
        self.assertEqual(document[2], "真正的文章标题 文章正文")

    def test_extract_page_uses_social_title_before_html_title(self) -> None:
        html = """
            <html><head>
                <title>网站首页</title>
                <meta property="og:title" content="具体内容标题">
            </head><body><p>正文</p></body></html>
        """

        document, _links = extract_page(html, "https://a.test/article")

        self.assertEqual(document[1], "具体内容标题")

    def test_extract_page_ignores_nonvisible_fallback_content(self) -> None:
        html = """
            <html><head><title>页面标题</title></head><body>
                <script>不应索引</script>
                <style>.hidden { display: none; }</style>
                <p>页面正文</p>
            </body></html>
        """

        document, _links = extract_page(html, "https://a.test/article")

        self.assertEqual(document[2], "页面正文")

    def test_extract_page_uses_guoxue_title_fallback(self) -> None:
        html = """
            <html><head><title>国学院网站</title></head><body>
            <div>
                <div>首栏</div>
                <div><div><div class="right">
                    <div>信息栏</div>
                    <div><div><div>国学院文章标题</div></div></div>
                </div></div></div>
            </div>
            </body></html>
        """

        document, _links = extract_page(html, "https://guoxue.ruc.edu.cn/article")

        self.assertEqual(document[1], "国学院文章标题")

    @patch(
        "zeta_engine.crawler.get_html",
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
        "zeta_engine.crawler.get_html",
        side_effect=lambda url, *_args, **_kwargs: (url, PAGES[url]),
    )
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

    def test_refreshes_done_pages_and_retries_historical_failures(self) -> None:
        old_page = "<html><title>A</title><body>旧正文</body></html>"
        new_page = "<html><title>A</title><body>新正文</body></html>"
        with Storage(
            document_db=":memory:",
            queue_db=":memory:",
        ) as storage:
            with patch(
                "zeta_engine.crawler.get_html",
                side_effect=[
                    ("https://a.test/", old_page),
                    ("https://a.test/", new_page),
                ],
            ) as get_html:
                crawl_urls(storage, ["https://a.test/"], per_host_delay=0)
                unchanged = crawl_urls(
                    storage,
                    ["https://a.test/"],
                    per_host_delay=0,
                )
                refreshed = crawl_urls(
                    storage,
                    ["https://a.test/"],
                    per_host_delay=0,
                    refresh_after_hours=0,
                )

            assert storage.documents is not None
            self.assertEqual(get_html.call_count, 2)
            self.assertEqual(unchanged["scheduled"], 0)
            self.assertEqual(refreshed["refreshed"], 1)
            self.assertEqual(storage.documents.get(1)[2], "新正文")

        with Storage(
            document_db=":memory:",
            queue_db=":memory:",
        ) as storage:
            with patch(
                "zeta_engine.crawler.get_html",
                side_effect=[
                    None,
                    None,
                    None,
                    (
                        "https://a.test/",
                        "<html><title>A</title><body>恢复正文</body></html>",
                    ),
                ],
            ) as get_html:
                crawl_urls(storage, ["https://a.test/"], per_host_delay=0)
                unchanged = crawl_urls(
                    storage,
                    ["https://a.test/"],
                    per_host_delay=0,
                )
                retried = crawl_urls(
                    storage,
                    ["https://a.test/"],
                    per_host_delay=0,
                    retry_failed=True,
                )

            assert storage.queue is not None
            self.assertEqual(get_html.call_count, 4)
            self.assertEqual(unchanged["scheduled"], 0)
            self.assertEqual(retried["requeued_failed"], 1)
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
