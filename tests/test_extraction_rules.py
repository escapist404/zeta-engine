import unittest

from zeta_engine.ingestion.crawler import extract_page


class ExtractionRulesTest(unittest.TestCase):
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
