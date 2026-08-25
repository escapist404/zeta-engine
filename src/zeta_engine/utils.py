from urllib.parse import urlsplit, urlunsplit
from url_normalize import url_normalize

def is_absolute_url(url: str) -> bool:
    parts = urlsplit(url.strip())
    return bool(parts.scheme and parts.netloc)
