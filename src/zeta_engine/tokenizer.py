import unicodedata
from functools import cache

import jieba

from zeta_engine.constants import STOPWORDS_PATH


def text_normalize(text: str) -> str:
    """使用相容性合成对文本进行标准化。"""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    return " ".join(text.split())


@cache
def load_stopwords() -> frozenset[str]:
    """加载禁用词。返回一个冻结集合。"""

    return frozenset(
        text_normalize(line)
        for line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def tokenize_with_positions(
    text: str,
    mode: str = "default",
) -> list[tuple[str, int]]:
    """使用 `jieba` 库分词，返回单词、起始下标构成元组的列表。"""

    if mode not in {"default", "search"}:
        raise ValueError(f"不支持的分词模式: {mode}")
    stopwords = load_stopwords()
    return [
        (token, start)
        for token, start, _ in jieba.tokenize(text, mode=mode)
        if token.strip() and token.casefold() not in stopwords
    ]


def load_user_dictionary() -> None:
    ...
