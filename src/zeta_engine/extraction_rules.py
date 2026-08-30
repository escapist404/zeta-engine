import re
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class PageExtractionRule:
    """Selectors for one page family within a site."""

    name: str
    path_pattern: str
    content_selectors: tuple[str, ...]
    title_selectors: tuple[str, ...] = ()
    excluded_selectors: tuple[str, ...] = ()

    def matches(self, url: str) -> bool:
        request = urlsplit(url)
        target = request.path + (f"?{request.query}" if request.query else "")
        return re.search(self.path_pattern, target, re.IGNORECASE) is not None


@dataclass(frozen=True)
class SiteExtractionRule:
    """Verified extraction policy shared by pages on one host."""

    content_selectors: tuple[str, ...]
    title_selectors: tuple[str, ...] = ()
    excluded_selectors: tuple[str, ...] = ()
    page_rules: tuple[PageExtractionRule, ...] = ()


COMMON_CONTENT_NOISE = (
    ".bread",
    ".crumbs",
    ".location",
    ".gp-bread",
    ".page_nav",
    ".pageTab",
    ".pageNP",
    ".m3npage",
    ".articlePage",
    ".annex",
    ".bdsharebuttonbox",
)


SITE_EXTRACTION_RULES: dict[str, SiteExtractionRule] = {
    "pd.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(
            "#articleDiv",
            ".gp-article",
            ".page-list23",
            "section.subPage .row > div.layout:last-child",
        ),
        title_selectors=(".gp-articleTitle h1", ".gp-articleTitle",),
        excluded_selectors=(
            ".wrapHeader1", ".footer_block1", ".footer", "#gp-subLeft",
            ".asideList1", ".dtbread2", ".gp-articleAuthor", ".gp-page1",
        ),
    ),
    "sph.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".commonRightCont",),
        title_selectors=(".articleTitle h1", ".articleTitle", ".commonRightTitle h1"),
        excluded_selectors=(
            ".headerT", ".footerBg", ".iphone_nav", ".commonNavLeft",
            ".articlePage",
        ),
    ),
    "clr.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".sub_right",),
        title_selectors=(".articleTitle h1", ".articleTitle h2", ".articleTitle"),
        excluded_selectors=(
            ".wraq_header", ".nav", ".sub_left", ".sbu_leftWrap",
            ".Wrap_footer", ".footer", ".articleAuthor",
        ),
    ),
    "dis.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".subPage_con",),
        title_selectors=(".subArticleTitle h1", ".subArticleTitle",),
        excluded_selectors=(
            ".wraq_header", ".snav", ".sub_left", ".footer_wrap",
            ".footer", ".subTitle",
        ),
    ),
    "dsdj.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".rightCon",),
        title_selectors=(".detailTit h1", ".detailTit h6", ".detailTit"),
        excluded_selectors=(
            ".top", ".wrapper", ".bottom", ".leftNav", ".pageNP",
        ),
    ),
    "scsce.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".sub_right",),
        title_selectors=(".articleTitle h1", ".articleTitle h2", ".articleTitle"),
        excluded_selectors=(
            ".wraq_header", ".aslide", ".fooer_wrap", ".sub_left",
            ".articleAuthor",
        ),
    ),
    "isbd.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".cnytion", ".cn .right"),
        title_selectors=(".hyher h1", ".hyher",),
        excluded_selectors=(
            ".nav", ".phone_nav", ".footer", ".point_out", ".ytoo",
        ),
    ),
    "sis.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".collegueNewsContainer .contentWrapper",),
        title_selectors=(
            ".contentWrapper h1", ".contentWrapper .f32", ".contentWrapper .f28",
        ),
        excluded_selectors=(".menu", ".stickyMenu", ".footer", ".banner"),
    ),
    "info.ruc.edu.cn": SiteExtractionRule(
        content_selectors=("#main .content",),
        title_selectors=("#main .articleTitle", "#main h1"),
        excluded_selectors=(
            "#top", "#footer", ".left_menu", ".navigation", ".extra_info",
        ),
    ),
    "www.phys.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".m3nCon", ".m2gkCon", ".container.cont"),
        title_selectors=(".m3nLx h1", ".m3nLx h6", ".m2gkCon h1"),
        excluded_selectors=(
            ".mMmenuLay", ".header", ".footer", ".m3pos", ".m3nRx",
            ".m3nShare", ".m3npage",
        ),
    ),
    "psy.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".layout.col-md-9",),
        title_selectors=(".gp-articleTitle h1", ".gp-articleTitle",),
        excluded_selectors=(
            ".pid-5341740c-f613-4fb8-8acd-967031d0631d",
            ".pid-9ec3926c-0d43-4cf3-aa6d-ff2305ccf367",
            ".asideList12", ".gp-articleAuthor", ".gp-page1",
        ),
    ),
    "guoxue.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".right",),
        title_selectors=(".right h1", ".right h2"),
        excluded_selectors=(".top", ".footer", ".left", ".r_t"),
    ),
    "envi.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".right.listRight",),
        title_selectors=(".detailTit h1", ".detailTit h6", ".detailTit"),
        excluded_selectors=(
            ".top", ".moblieTop", ".footerTop", ".copyRight", ".left",
            ".pageTab", ".pagination",
        ),
    ),
    "ai.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".page_content > .fr", ".main"),
        title_selectors=(".page_content > .fr .tit h6",),
        excluded_selectors=(
            "header", ".footer", ".point_out", ".mobile-nav__wrapper",
            ".page_content > .fl",
        ),
    ),
    "gsai.ruc.edu.cn": SiteExtractionRule(
        content_selectors=(".page_content > .fr", ".main"),
        title_selectors=(".page_content > .fr .tit h6",),
        excluded_selectors=(
            "header", "#mainNav", ".footer", ".point_out",
            ".mobile-nav__wrapper", ".page_content > .fl",
        ),
        page_rules=(
            PageExtractionRule(
                name="video-detail",
                path_pattern=r"^/addons/video/video/play\.html",
                content_selectors=(".mn-sec.full_wdth_single_video",),
                title_selectors=(".vid-info h1", ".vid-info h2", ".vid-info"),
            ),
            PageExtractionRule(
                name="video-list",
                path_pattern=r"^/addons/video/video/cate\.html",
                content_selectors=(".videso_section",),
            ),
            PageExtractionRule(
                name="teacher-list",
                path_pattern=r"^/addons/teacher/index\.html$",
                content_selectors=(".row.m-t-30",),
            ),
            PageExtractionRule(
                name="teacher-detail",
                path_pattern=(
                    r"^/(?:addons/teacher/index/info\.html|"
                    r"(?:english/)?~?[\w-]+)(?:\?.*)?$"
                ),
                content_selectors=("#colorlib-page",),
                title_selectors=("#colorlib-aside h1",),
                excluded_selectors=("#mainNav",),
            ),
        ),
    ),
}


@dataclass(frozen=True)
class ResolvedExtractionRule:
    name: str
    content_selectors: tuple[str, ...]
    title_selectors: tuple[str, ...]
    excluded_selectors: tuple[str, ...]


def resolve_extraction_rule(url: str) -> ResolvedExtractionRule | None:
    site = SITE_EXTRACTION_RULES.get(urlsplit(url).hostname or "")
    if site is None:
        return None
    page = next((rule for rule in site.page_rules if rule.matches(url)), None)
    if page is None:
        return ResolvedExtractionRule(
            name="site-default",
            content_selectors=site.content_selectors,
            title_selectors=site.title_selectors,
            excluded_selectors=COMMON_CONTENT_NOISE + site.excluded_selectors,
        )
    return ResolvedExtractionRule(
        name=page.name,
        content_selectors=page.content_selectors,
        title_selectors=page.title_selectors + site.title_selectors,
        excluded_selectors=(
            COMMON_CONTENT_NOISE + site.excluded_selectors + page.excluded_selectors
        ),
    )
