from pathlib import Path

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

# Union of the four lists from https://github.com/goto456/stopwords
STOPWORDS_PATH = Path(__file__).with_name("stopwords.txt")
STOPWORDS_VERSION = "goto456-bf8b03b9-union"

K1 = 1.2
TITLE_WEIGHT = 2.0
TITLE_B = 0.3
TEXT_WEIGHT = 1.0
TEXT_B = 0.75
