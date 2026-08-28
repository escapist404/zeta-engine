import unicodedata

import jieba

from zeta_engine.constants import STOPWORDS


def text_normalize(text: str) -> str:
    """使用相容性合成对文本进行标准化。"""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    return " ".join(text.split())


def load_stopwords() -> frozenset[str]:
    """返回内置停用词集合。"""

    return STOPWORDS


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
        if (
            token.strip()
            and any(character.isalnum() for character in token)
            and token.casefold() not in stopwords
        )
    ]


def load_user_dictionary() -> None:
    ...
