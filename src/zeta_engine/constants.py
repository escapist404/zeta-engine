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

ALLOWED_DOMAINS = (
    "http://pd.ruc.edu.cn/",
    "http://sph.ruc.edu.cn/",
    "https://clr.ruc.edu.cn/",
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

TIMEOUT = 30

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
