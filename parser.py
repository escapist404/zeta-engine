import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

URL = "https://news.ruc.edu.cn/"
TIMEOUT = 3

response = requests.get(url=URL, timeout=TIMEOUT)
response.raise_for_status()
response.encoding = response.apparent_encoding

soup = BeautifulSoup(response.text, "html.parser")

for tag in soup(["script", "style", "noscript"]):
    tag.decompose()

for tag in soup.select("a[href]"):
    text = tag.get_text(' ', strip=True)
    link = urljoin(base=URL, url=tag["href"])
    print(text, link)
