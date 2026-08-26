import logging
import unicodedata

import jieba

from zeta_engine.storage import Storage

logger = logging.getLogger(__name__)


def text_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    return " ".join(text.split())


def tokenize_with_positions(
    text: str,
    mode: str = "default",
) -> list[tuple[str, int]]:
    if mode not in {"default", "search"}:
        raise ValueError(f"不支持的分词模式: {mode}")
    return [(token, start) for token, start, _ in jieba.tokenize(text, mode=mode)]


def load_user_dictionary() -> None:
    ...


def build_index(storage: Storage, mode: str = "default") -> dict[str, int]:
    if storage.documents is None or storage.index is None:
        raise ValueError("建立索引需要 document_db 和 index_db")
    if mode not in {"default", "search"}:
        raise ValueError(f"不支持的分词模式: {mode}")

    documents = storage.documents
    index = storage.index
    mode_changed = index.get_metadata("tokenizer_mode") != mode

    stats = {
        "seen": 0,
        "indexed": 0,
        "skipped": 0,
    }

    for document_id, _url, title, text, fetched_at in documents.iter_all():
        stats["seen"] += 1
        if stats["seen"] % 100 == 0:
            logger.info("索引进度: 已读取 %s 篇", stats["seen"])

        indexed_document = index.get_indexed_document(document_id)

        if (
            not mode_changed
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
    return stats


def search_term(
    storage: Storage,
    term: str,
) -> set[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    term = text_normalize(term)
    if not term:
        return set()

    return set(storage.index.lookup_posting(term, storage.index.TITLE)) | set(
        storage.index.lookup_posting(term, storage.index.TEXT)
    )


def search_phrase(
    storage: Storage,
    phrase: str,
) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    phrase = text_normalize(phrase)
    if not phrase:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    tokens = tokenize_with_positions(phrase, mode)
    if not tokens:
        return []

    matches: set[int] = set()
    first_offset = tokens[0][1]

    for field in (storage.index.TITLE, storage.index.TEXT):
        postings = [
            storage.index.lookup_posting(term, field)
            for term, _offset in tokens
        ]
        if any(not posting for posting in postings):
            continue

        document_ids = set(postings[0])
        for posting in postings[1:]:
            document_ids &= posting.keys()

        for document_id in document_ids:
            positions = [set(posting[document_id]) for posting in postings]
            if any(
                all(
                    start + offset - first_offset in positions[index]
                    for index, (_term, offset) in enumerate(tokens[1:], start=1)
                )
                for start in positions[0]
            ):
                matches.add(document_id)

    return sorted(matches)


def search_query(storage: Storage, query: str, *, phrase: bool = False) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")
    if phrase:
        return search_phrase(storage, query)

    query = text_normalize(query)
    if not query:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    terms = list(dict.fromkeys(
        term for term, _position in tokenize_with_positions(query, mode)
        if term.strip()
    ))
    if not terms:
        return []

    postings = {
        term: (
            storage.index.lookup_posting(term, storage.index.TITLE),
            storage.index.lookup_posting(term, storage.index.TEXT),
        )
        for term in terms
    }
    matches = set(postings[terms[0]][0]) | set(postings[terms[0]][1])
    for title_posting, text_posting in postings.values():
        matches &= set(title_posting) | set(text_posting)

    scores = {
        document_id: sum(
            2 * len(title_posting.get(document_id, ()))
            + len(text_posting.get(document_id, ()))
            for title_posting, text_posting in postings.values()
        )
        for document_id in matches
    }

    return sorted(matches, key=lambda document_id: (-scores[document_id], document_id))
