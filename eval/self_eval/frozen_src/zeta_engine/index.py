import logging

from zeta_engine.constants import STOPWORDS_VERSION
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import (
    TEXT_NORMALIZER_VERSION,
    text_normalize,
    tokenize_with_positions,
)

logger = logging.getLogger(__name__)


def build_index(storage: Storage, mode: str = "default") -> dict[str, int]:
    if storage.documents is None or storage.index is None:
        raise ValueError("建立索引需要 document_db 和 index_db")
    if mode not in {"default", "search"}:
        raise ValueError(f"不支持的分词模式: {mode}")

    documents = storage.documents
    index = storage.index
    mode_changed = index.get_metadata("tokenizer_mode") != mode
    stopwords_changed = (
        index.get_metadata("stopwords_version") != STOPWORDS_VERSION
    )
    normalizer_changed = (
        index.get_metadata("text_normalizer_version")
        != TEXT_NORMALIZER_VERSION
    )

    stats = {
        "seen": 0,
        "indexed": 0,
        "skipped": 0,
    }

    for document_id, _url, title, text, fetched_at in documents.iter_all():
        stats["seen"] += 1
        if stats["seen"] % 100 == 0:
            logger.info("索引进度: 已读取 %s 篇", stats["seen"])

        indexed_document = index.get_document_by_index(document_id)

        if (
            not mode_changed
            and not stopwords_changed
            and not normalizer_changed
            and indexed_document is not None
            and indexed_document[0] == fetched_at
        ):
            stats["skipped"] += 1
            continue

        title_norm = text_normalize(title)
        text_norm = text_normalize(text)

        postings: dict[tuple[str, int], list[int]] = {}

        for field, content in (
            (index.TITLE, title_norm),
            (index.TEXT, text_norm),
        ):
            for term, position in tokenize_with_positions(content, mode):
                postings.setdefault((term, field), []).append(position)

        index.replace_document(
            document_id=document_id,
            fetched_at=fetched_at,
            title_norm=title_norm,
            text_norm=text_norm,
            postings=postings,
        )

        stats["indexed"] += 1

    index.set_metadata("tokenizer_mode", mode)
    index.set_metadata("stopwords_version", STOPWORDS_VERSION)
    index.set_metadata("text_normalizer_version", TEXT_NORMALIZER_VERSION)

    return stats
