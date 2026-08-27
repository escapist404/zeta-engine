from collections import Counter
from math import log

from zeta_engine.storage import Storage
from zeta_engine.tokenizer import (
    load_stopwords,
    text_normalize,
    tokenize_with_positions,
)


def search_term(
    storage: Storage,
    term: str,
) -> set[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    term = text_normalize(term)
    if not term or term in load_stopwords():
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


def search_query(
        storage: Storage, 
        query: str, *, 
        phrase: bool = False
) -> list[int]:
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


def search_tf_idf(
        storage: Storage,
        query: str,
) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    query = text_normalize(query)
    if not query:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    query_terms = [
        term
        for term, _ in tokenize_with_positions(query, mode)
        if term.strip()
    ]

    if not query_terms:
        return []

    terms = list(dict.fromkeys(query_terms))

    document_count = storage.index.count_documents()
    query_tf = Counter(query_terms)

    postings = {
        term: (
            storage.index.lookup_posting(term, storage.index.TITLE),
            storage.index.lookup_posting(term, storage.index.TEXT),
        )
        for term in terms
    }

    idfs = {}
    for term, (title_posting, text_posting) in postings.items():
        df = len(set(title_posting) | set(text_posting))
        idfs[term] = log((document_count + 1) / (df + 1)) + 1.
    query_weights = {
        term: (1 + log(query_tf[term])) * idfs[term]
        for term in terms
    }

    matches = set(postings[terms[0]][0]) | set(postings[terms[0]][1])
    for title_posting, text_posting in postings.values():
        matches &= set(title_posting) | set(text_posting)

    def document_weight(document_id: int, term: str) -> float:
        title_posting, text_posting = postings[term]
        tf = (
            len(title_posting.get(document_id, ()))
            + len(text_posting.get(document_id, ()))
        )
        return (1 + log(tf)) * idfs[term] if tf else 0.

    selected_tf_idf_norm = storage.index.get_tf_idf_norm(matches)

    scores = {
        document_id: sum(
            document_weight(document_id, term) * query_weights[term]
            for term in terms
        ) / selected_tf_idf_norm[document_id]
        for document_id in matches
        if selected_tf_idf_norm.get(document_id, 0.) > 0.
    }

    return sorted(matches, key=lambda document_id: (-scores[document_id], document_id))

def search_bm25f(
        storage: Storage,
        query: str,
) -> list[int]:
    if storage.index is None:
        raise ValueError("查询需要 index_db")

    query = text_normalize(query)
    if not query:
        return []

    mode = storage.index.get_metadata("tokenizer_mode") or "default"
    query_terms = [
        term
        for term, _ in tokenize_with_positions(query, mode)
        if term.strip()
    ]

    if not query_terms:
        return []

    terms = list(dict.fromkeys(query_terms))
    ...
