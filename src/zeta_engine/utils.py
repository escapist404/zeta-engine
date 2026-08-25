from urllib.parse import urlsplit, urlunsplit

def is_absolute_url(url: str) -> bool:
    parts = urlsplit(url.strip())
    return bool(parts.scheme and parts.netloc)

def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())

    scheme_ = parts.scheme.lower()
    netloc_ = parts.netloc.lower()

    if scheme_.startswith("https") and netloc_.endswith(":443"):
        netloc_ = netloc_.removesuffix(":443")
    if scheme_.startswith("http") and not scheme_.startswith("https") and netloc_.endswith(":80"):
        netloc_ = netloc_.removesuffix(":80")

    return urlunsplit(
        (
            scheme_, 
            netloc_, 
            parts.path or '/', 
            parts.query, 
            ''
        )
    )