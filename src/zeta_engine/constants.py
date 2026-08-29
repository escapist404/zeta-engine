SEED_URLS = (
    "http://pd.ruc.edu.cn/",
    "http://sph.ruc.edu.cn/",
    "https://clr.ruc.edu.cn/zwwz/index.htm",
    "http://dis.ruc.edu.cn/",
    "http://dsdj.ruc.edu.cn/",
    "http://scsce.ruc.edu.cn/",
    "http://isbd.ruc.edu.cn/",
    "http://sis.ruc.edu.cn/",
    "http://info.ruc.edu.cn/",
    "http://www.phys.ruc.edu.cn/",
    "http://psy.ruc.edu.cn/",
    "http://guoxue.ruc.edu.cn/",
    "https://envi.ruc.edu.cn/",
    "http://ai.ruc.edu.cn/", 
    "https://gsai.ruc.edu.cn"
)

ALLOWED_DOMAINS = SEED_URLS

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

TIMEOUT = 30

TITLE_SELECTORS = (
    "#articleDiv h1, #articleDiv [itemprop='headline']",
    "article h1, article [itemprop='headline']",
    "main h1, main [itemprop='headline']",
    "[role='main'] h1, [role='main'] [itemprop='headline']",
    (".article-title, .article_title, .articleTitle, "
    ".news-title, .news_title, .newsTitle"),
    ("body > div > div:nth-child(2) > div > div.right > "
     "div:nth-child(2) > div:nth-child(1) > div:nth-child(1)"),
    ("body > article > div.subPage_con > section "
     "> div > div.articleTitle.article02 > h2"),
    "body > div.subPage > div > div > div > div.article.art > p:nth-child(1)",
    "body > div.cn > div > div.cnytion > div.hyher > h1",
    ("body > div.collegueNewsContainer > div.content > div > div > "
     "div.flex.align-center.justify-center.f32.t1-black"),
    "#main > div.content > h1",
    "body > div.container > div > div.m3nLx > form > h2",
    ("body > div > section.gp-mb-40.pid-cbdec571-3b15-4f63-978c-"
     "865634de0a1d > div > div > div > "
     "div.gp-articleTitle.gp-articleTitle1"),
    ("body > div > div:nth-child(2) > div > div.right > "
     "div:nth-child(2) > div:nth-child(1) > div:nth-child(1)"),
    ("body > main > div > div > div.mainCon > div.right.listRight > "
     "div > div.detailTit > h6"),
    ("body > div.page-wrapper > div.page > "
     "div.page_content.clearfix.common_width_1 > div.fr > "
     "div.tit.text-left > h6"),
)

SOCIAL_TITLE_SELECTOR = (
    "meta[property='og:title'][content], meta[name='twitter:title'][content]"
)

CONTENT_SELECTORS = (
    "#articleDiv",
    "article",
    "main",
    "[role='main']",
    ".notice_list",
)

EXCLUDED_HTML_SELECTORS = (
    "script, style, noscript, template, nav, aside, header, footer, bottom, "
    "[hidden], [aria-hidden='true'], #search_warp, "
    ".stricky-header, .page_content > .fl, .crumbs, .page_nav, "
    ".footer, .point_out, .mobile-nav__wrapper, "
    "section.btm_bar, body.div > section.mn-sec.full_wdth_single_video > div > div.row > div > div > div > div > div, "
    "body > div.header.__web-inspector-hide-shortcut__, body > div.footer, body > div.page-wrapper > div.page > div.page_content.clearfix.common_width_1 > div.fr > div.crumbs, "
    "body > div.page-wrapper > div.page > div.page_content.clearfix.common_width_1 > div.fl, "
    "#top > div.header, #top > div.top_menu_box, "
    "#footer, #main > div.left_menu, #main > div.content > div.navigation, #main > div.content > div.activity_detail > div.extra_info, "
    "body > div.content > div > div.leftNav, body > div.content > div > div.rightCon > div.crumbs, "
    "body > header, #top > div.top_menu_box, #app > header, body > div.top.wow.fadeIn, "
    "body > div.header.wow.fadeIn, body > div.m3pos.wow.fadeIn, body > div.container > div > div.m3nRx, body > div.container > div > div.m3nLx > form > div.m3n_tm, "
    "body > div.footer.wow.fadeIn, body > div.container > div > div.m3nLx > form > div.m3nShare.bdsharebuttonbox.wow.fadeIn.bdshare-button-style0-24, "
    "body > div.moblieTop, body > main > div > div > div.bread, body > main > div > div > div.mainCon > div.left, body > main > div > div > div.mainCon > div.right.listRight > div > div.pageTab, "
    "body > footer, body > div > div.top, body > div > div.footer, body > div > div:nth-child(2) > div > div.left, body > div > div:nth-child(2) > div > div.right > div.r_t, "
    "body > header, #subbanner, body > article > div.subPage_con > div, body > article > div.subPage_con > section > div > div.articleTitle.article02 > div.articleAuthor, "
    "body > article > div.subPage_con > section > div > div.articleTitle.article02 > div.wrapSize, body > footer, body > div.content > div > div.leftNav, body > div.content > div > div.rightCon > div.crumbs.borderNone, "
    "body > div.wraq_header, body > div.aslide, #subbanner, body > div.subPage > div > div > div > div.annex, body > div.fooer_wrap, "
    "body > div.hr_cur, body > div.nav, body > div.phone_nav, body > div.cn > div > div.ytoo, body > div.footer, "
    "body > div.menu, body > div.collegueNewsContainer > div.content > div > div > div.flex.align-center.h20.link_box, "
    "body > div.collegueNewsContainer > div.content > div > div > div.h54.flex.align-center.justify-center, body > div.footer, "
    "#top, #banner, #main > div.left_menu, #main > div.content > div.navigation, #main > div.content > div.extra_info, #main > div.content > div.share, #footer, "
    "body > div > section.gp-clearFix.pid-5341740c-f613-4fb8-8acd-967031d0631d, body > div > section:nth-child(3), body > div > section.gp-mb-40.pid-cbdec571-3b15-4f63-978c-865634de0a1d > div > div > div > div.block-list82.gpFontSize, "
    "body > div > section.gp-mb-40.pid-cbdec571-3b15-4f63-978c-865634de0a1d > div > div > div > div.gp-articleAuthor.gp-articleAuthor1, "
    "body > div > section.gp-clearFix.pid-9ec3926c-0d43-4cf3-aa6d-ff2305ccf367, body > div > div:nth-child(2) > div > div.right > div:nth-child(2) > div:nth-child(1) > div:nth-child(2), "
    "body > div.moblieTop, body > div.adv, body > footer, body > main > div > div > div.bread, body > main > div > div > div.mainCon > div.left, body > main > div > div > div.mainCon > div.right.listRight > div > div.detailTit > p, "
    "body > main > div > div > div.mainCon > div.right.listRight > div > div.pageTab, body > footer, "
    "body > div.page-wrapper > header, body > div.page-wrapper > div.page > div.page_tit, body > div.page-wrapper > div.page > div.page_content.clearfix.common_width_1 > div.fr > div.tit.text-left > span:nth-child(2), "
    "body > div.page-wrapper > div.page > div.page_content.clearfix.common_width_1 > div.fr > div.tit.text-left > span:nth-child(3), body > div.page-wrapper > div.footer.container-fluid, "
    "body > div > div.container.m-t-20"
)

# Keep this deliberately small: BM25 already downweights common terms, while
# aggressive stop-word lists can destroy meaningful dates, ordinals and codes.
STOPWORDS = frozenset({
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "是",
    "为",
    "于",
    "对",
    "把",
    "被",
    "由",
    "从",
    "向",
    "以",
    "而",
    "并",
    "也",
    "都",
})
STOPWORDS_VERSION = "builtin-v1"
