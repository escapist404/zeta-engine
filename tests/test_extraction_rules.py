import unittest

from zeta_engine.constants import ALLOWED_DOMAINS
from zeta_engine.crawler import extract_page
from zeta_engine.extraction_rules import resolve_extraction_rule


class ExtractionRulesTest(unittest.TestCase):
    def test_every_configured_domain_has_a_site_rule(self) -> None:
        for url in ALLOWED_DOMAINS:
            with self.subTest(url=url):
                self.assertIsNotNone(resolve_extraction_rule(url))

    def test_site_content_container_excludes_navigation_and_footer(self) -> None:
        html = """
            <html><head><title>学院网站</title></head><body>
              <div class="headerT">首页 学院概况 师资队伍 科学研究</div>
              <div class="main">
                <div class="commonNavLeft">学院简介 领导班子</div>
                <div class="commonRightCont">
                  <div class="location">当前位置：首页</div>
                  <h1 class="articleTitle">招生通知</h1>
                  <p>这里是需要保留的完整通知正文。</p>
                  <div class="articlePage">上一篇：旧通知</div>
                </div>
              </div>
              <div class="footerBg">友情链接 Copyright</div>
            </body></html>
        """

        document, _links = extract_page(
            html,
            "http://sph.ruc.edu.cn/xydt/tzgg/example.htm",
        )
        text, content_html = document[2], document[4]

        self.assertIn("需要保留的完整通知正文", text)
        self.assertNotIn("师资队伍", text)
        self.assertNotIn("上一篇", text)
        self.assertNotIn("Copyright", text)
        self.assertIn("<p>这里是需要保留的完整通知正文。</p>", content_html)

    def test_teacher_rule_keeps_profile_aside_but_removes_navigation(self) -> None:
        html = """
            <html><head><title>教师系统</title></head><body>
              <nav id="mainNav">首页 教师中心</nav>
              <div id="colorlib-page"><div class="container-wrap">
                <aside id="colorlib-aside">
                  <h1>张老师</h1><p>主要研究兴趣为信息检索。</p>
                </aside>
                <div id="colorlib-main">
                  <section><h2>教育经历</h2><p>中国人民大学博士。</p></section>
                </div>
              </div></div>
            </body></html>
        """

        document, _links = extract_page(html, "https://gsai.ruc.edu.cn/teacher_1")

        self.assertEqual(document[1], "张老师")
        self.assertIn("主要研究兴趣", document[2])
        self.assertIn("教育经历", document[2])
        self.assertNotIn("教师中心", document[2])

    def test_form_container_is_unwrapped_without_losing_article_text(self) -> None:
        html = """
            <html><head><title>物理学院</title></head><body>
              <div class="m3nCon"><form>
                <h6>报告标题</h6>
                <div class="v_news_content"><p>报告摘要正文。</p></div>
              </form></div>
            </body></html>
        """

        document, _links = extract_page(
            html,
            "http://www.phys.ruc.edu.cn/info/1053/1.htm",
        )

        self.assertIn("报告摘要正文", document[2])
        self.assertIn("<p>报告摘要正文。</p>", document[4])
        self.assertNotIn("<form", document[4])

    def test_gsai_special_page_families_resolve_separately(self) -> None:
        cases = {
            "https://gsai.ruc.edu.cn/addons/teacher/index.html": "teacher-list",
            "https://gsai.ruc.edu.cn/zhicheng_dou": "teacher-detail",
            "https://gsai.ruc.edu.cn/english/zhicheng_dou": "teacher-detail",
            (
                "https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=1"
            ): "teacher-detail",
            (
                "https://gsai.ruc.edu.cn/addons/video/video/cate.html?cate_id=13"
            ): "video-list",
            "https://gsai.ruc.edu.cn/addons/video/video/play.html?id=31": (
                "video-detail"
            ),
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                rule = resolve_extraction_rule(url)
                self.assertIsNotNone(rule)
                self.assertEqual(rule.name, expected)
