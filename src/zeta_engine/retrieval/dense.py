import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from zeta_engine.infrastructure.storage import Storage
from zeta_engine.infrastructure.tokenizer import text_normalize, tokenize_with_positions

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/bge-small-zh-v1.5")
DEFAULT_INDEX_DIR = Path("data/dense")
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
SCHEMA_VERSION = 4
EMBEDDING_DIMENSION = 512
MAX_TOKENS = 384
TITLE_MAX_TOKENS = 64
OVERLAP_TOKENS = 64
DEFAULT_BATCH_SIZE = 32

_ENCODE_LOCK = threading.Lock()


@dataclass(frozen=True)
class DenseHit:
    document_id: int
    score: float
    snippet: str


@dataclass(frozen=True)
class PassageHit:
    """One ranked, independently citable passage."""

    document_id: int
    chunk_index: int
    score: float
    text: str
    title: str = ""

    @property
    def passage_id(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"


def _chunk_document(
    tokenizer: Any,
    title: str,
    text: str,
    *,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    title_max_tokens: int = TITLE_MAX_TOKENS,
) -> list[tuple[str, str]]:
    """Return (embedding text, snippet) pairs within the token budget."""

    title = text_normalize(title)
    text = text_normalize(text)
    if not title and not text:
        return []

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    title_encoding = tokenizer(
        title,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    title_offsets = title_encoding["offset_mapping"][:title_max_tokens]
    short_title = title[:title_offsets[-1][1]] if title_offsets else ""

    if not text:
        passage = f"标题：{short_title}"
        passage_ids = tokenizer.encode(passage, add_special_tokens=False)
        if len(passage_ids) + special_tokens > max_tokens:
            raise ValueError("标题超出 token 限制")
        return [(passage, short_title)] if short_title else []

    prefix = f"标题：{short_title}\n正文："
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    body_budget = max_tokens - special_tokens - len(prefix_ids)
    if body_budget <= overlap_tokens:
        raise ValueError("标题和前缀占用的 token 过多，无法分块")

    body_encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    body_ids = body_encoding["input_ids"]
    body_offsets = body_encoding["offset_mapping"]
    step = body_budget - overlap_tokens
    chunks = []

    for start in range(0, len(body_ids), step):
        chunk_ids = body_ids[start:start + body_budget]
        if not chunk_ids:
            break

        chunk_offsets = body_offsets[start:start + len(chunk_ids)]
        snippet = text[chunk_offsets[0][0]:chunk_offsets[-1][1]].strip()
        passage = prefix + snippet
        if passage and snippet:
            chunks.append((passage, snippet))
        if start + body_budget >= len(body_ids):
            break

    return chunks


@lru_cache(maxsize=2)
def _load_model(model_path: str, device: str | None) -> Any:
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dense 模型目录不存在: {path}")

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(path),
        device=device,
        local_files_only=True,
    )


def _encode(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    with _ENCODE_LOCK:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
    return np.asarray(vectors, dtype=np.float32)


def _validate_vectors(vectors: np.ndarray, expected_rows: int) -> None:
    if vectors.ndim != 2:
        raise ValueError("Dense 向量必须是二维矩阵")
    if vectors.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise ValueError(
            "Dense 向量形状不正确: "
            f"期望 ({expected_rows}, {EMBEDDING_DIMENSION})，"
            f"实际 {vectors.shape}"
        )
    if vectors.dtype != np.float32:
        raise ValueError("Dense 向量必须使用 float32")
    if not np.isfinite(vectors).all():
        raise ValueError("Dense 向量包含 NaN 或 Inf")
    if expected_rows:
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Dense 向量未完成 L2 归一化")


def build_dense_index(
    storage: Storage,
    output_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> dict[str, int | float]:
    """Build and atomically publish a complete dense index generation."""

    if storage.documents is None:
        raise ValueError("Dense 建库需要 document_db")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    started = time.monotonic()
    model_path = Path(model_path)
    model = _load_model(str(model_path), device)
    tokenizer = model.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        from transformers import BertTokenizerFast

        tokenizer = BertTokenizerFast.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    dimension = model.get_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Dense 模型维度必须是 {EMBEDDING_DIMENSION}，实际为 {dimension}"
        )

    passages: list[str] = []
    records: list[dict[str, int | str]] = []
    document_count = 0
    indexed_document_count = 0

    for document_id, _url, title, text, _fetched_at in storage.documents.iter_all():
        document_count += 1
        chunks = _chunk_document(tokenizer, title, text)
        if chunks:
            indexed_document_count += 1
        for chunk_index, (passage, snippet) in enumerate(chunks):
            passages.append(passage)
            records.append({
                "document_id": document_id,
                "chunk_index": chunk_index,
                "title": text_normalize(title),
                "text": snippet,
            })

    if passages:
        vectors = _encode(
            model,
            passages,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    else:
        vectors = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    _validate_vectors(vectors, len(records))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_id = uuid.uuid4().hex
    vector_name = f"vectors-{build_id}.npy"
    chunk_name = f"chunks-{build_id}.jsonl"
    passage_index_name = f"passages-{build_id}.db"
    vector_path = output_dir / vector_name
    chunk_path = output_dir / chunk_name
    passage_index_path = output_dir / passage_index_name
    metadata_path = output_dir / "metadata.json"
    metadata_temporary_path = output_dir / f".metadata-{build_id}.json"

    np.save(vector_path, vectors, allow_pickle=False)
    with chunk_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    _build_passage_sparse_index(passage_index_path, records)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(model_path),
        "query_instruction": QUERY_INSTRUCTION,
        "dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "max_tokens": MAX_TOKENS,
        "title_max_tokens": TITLE_MAX_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "document_count": document_count,
        "indexed_document_count": indexed_document_count,
        "chunk_count": len(records),
        "vector_file": vector_name,
        "chunk_file": chunk_name,
        "passage_index_file": passage_index_name,
        "passage_tokenizer_mode": "search",
    }
    metadata_temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_temporary_path.replace(metadata_path)
    _load_dense_index.cache_clear()

    elapsed_seconds = time.monotonic() - started
    logger.info(
        "Dense 索引完成: documents=%s, indexed=%s, chunks=%s, elapsed=%.3fs",
        document_count,
        indexed_document_count,
        len(records),
        elapsed_seconds,
    )
    return {
        "documents": document_count,
        "indexed_documents": indexed_document_count,
        "chunks": len(records),
        "dimension": EMBEDDING_DIMENSION,
        "elapsed_seconds": elapsed_seconds,
    }


def _read_metadata(index_dir: Path) -> dict[str, Any]:
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dense 索引元数据不存在: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Dense 索引元数据损坏: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Dense 索引元数据必须是 JSON 对象")
    return metadata


def upgrade_dense_index_v3(
    storage: Storage,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> dict[str, int]:
    """Add canonical passage metadata and BM25 to an existing v3 index.

    The embedding vectors and chunk boundaries are unchanged between v3 and
    v4, so this migration avoids recomputing embeddings.
    """

    if storage.documents is None:
        raise ValueError("Dense 索引迁移需要 document_db")
    index_dir = Path(index_dir)
    metadata = _read_metadata(index_dir)
    if metadata.get("schema_version") == SCHEMA_VERSION:
        _vectors, records, validated = _load_dense_index(
            str(index_dir.resolve())
        )
        document_count = validated.get("document_count")
        if not isinstance(document_count, int) or document_count < 0:
            raise ValueError("Dense v4 索引 document_count 无效")
        return {
            "documents": document_count,
            "chunks": len(records),
        }
    if metadata.get("schema_version") != 3:
        raise ValueError("只有 Dense v3 索引可以执行此迁移")
    chunk_count = metadata.get("chunk_count")
    if not isinstance(chunk_count, int) or chunk_count < 0:
        raise ValueError("Dense v3 索引 chunk_count 无效")
    vector_file = metadata.get("vector_file")
    chunk_file = metadata.get("chunk_file")
    if not isinstance(vector_file, str) or not isinstance(chunk_file, str):
        raise ValueError("Dense v3 索引文件字段无效")

    vector_path = index_dir / vector_file
    old_chunk_path = index_dir / chunk_file
    try:
        vectors = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Dense 向量文件损坏: {vector_path}") from error
    _validate_vectors(vectors, chunk_count)

    titles = {
        document_id: text_normalize(title)
        for document_id, _url, title, _text, _fetched_at
        in storage.documents.iter_all()
    }
    expected_documents = metadata.get("document_count")
    if isinstance(expected_documents, int) and len(titles) != expected_documents:
        raise ValueError("文档库数量已变化，不能复用旧 Dense 向量")

    records: list[dict[str, int | str]] = []
    try:
        with old_chunk_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                record = json.loads(line)
                document_id = record.get("document_id") if isinstance(record, dict) else None
                if (
                    not isinstance(record, dict)
                    or not isinstance(document_id, int)
                    or document_id not in titles
                    or not isinstance(record.get("chunk_index"), int)
                    or not isinstance(record.get("text"), str)
                ):
                    raise ValueError(f"Dense 分块第 {line_number} 行格式无效")
                records.append({
                    "document_id": document_id,
                    "chunk_index": int(record["chunk_index"]),
                    "title": titles[document_id],
                    "text": text_normalize(str(record["text"])),
                })
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Dense 分块文件损坏: {old_chunk_path}") from error
    if len(records) != chunk_count:
        raise ValueError("Dense v3 分块数量与元数据不一致")

    build_id = uuid.uuid4().hex
    new_chunk_name = f"chunks-{build_id}.jsonl"
    passage_index_name = f"passages-{build_id}.db"
    new_chunk_path = index_dir / new_chunk_name
    passage_index_path = index_dir / passage_index_name
    temporary_metadata_path = index_dir / f".metadata-{build_id}.json"
    with new_chunk_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    _build_passage_sparse_index(passage_index_path, records)

    upgraded = dict(metadata)
    upgraded.update({
        "schema_version": SCHEMA_VERSION,
        "chunk_file": new_chunk_name,
        "passage_index_file": passage_index_name,
        "passage_tokenizer_mode": "search",
    })
    temporary_metadata_path.write_text(
        json.dumps(upgraded, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata_path.replace(index_dir / "metadata.json")
    _load_dense_index.cache_clear()
    return {
        "documents": len(titles),
        "chunks": len(records),
    }


@lru_cache(maxsize=2)
def _load_dense_index(
    index_dir_value: str,
) -> tuple[np.ndarray, tuple[dict[str, int | str], ...], dict[str, Any]]:
    index_dir = Path(index_dir_value)
    metadata = _read_metadata(index_dir)

    required = {
        "schema_version",
        "model_path",
        "query_instruction",
        "dimension",
        "normalized",
        "chunk_count",
        "vector_file",
        "chunk_file",
        "passage_index_file",
        "passage_tokenizer_mode",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(
            "Dense 索引元数据缺少字段: " + ", ".join(sorted(missing))
        )
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的 Dense 索引版本: {metadata['schema_version']}"
        )
    if metadata["dimension"] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Dense 索引维度必须是 {EMBEDDING_DIMENSION}"
        )
    if metadata["normalized"] is not True:
        raise ValueError("Dense 索引向量必须已归一化")
    if not isinstance(metadata["chunk_count"], int) or metadata["chunk_count"] < 0:
        raise ValueError("Dense 索引 chunk_count 无效")
    if not isinstance(metadata["model_path"], str):
        raise ValueError("Dense 索引 model_path 无效")
    if not isinstance(metadata["query_instruction"], str):
        raise ValueError("Dense 索引 query_instruction 无效")
    if not isinstance(metadata["vector_file"], str):
        raise ValueError("Dense 索引 vector_file 无效")
    if not isinstance(metadata["chunk_file"], str):
        raise ValueError("Dense 索引 chunk_file 无效")
    if not isinstance(metadata["passage_index_file"], str):
        raise ValueError("Dense 索引 passage_index_file 无效")
    if metadata["passage_tokenizer_mode"] != "search":
        raise ValueError("Dense 索引 passage_tokenizer_mode 无效")

    vector_path = index_dir / metadata["vector_file"]
    chunk_path = index_dir / metadata["chunk_file"]
    passage_index_path = index_dir / metadata["passage_index_file"]
    if not vector_path.is_file():
        raise FileNotFoundError(f"Dense 向量文件不存在: {vector_path}")
    if not chunk_path.is_file():
        raise FileNotFoundError(f"Dense 分块文件不存在: {chunk_path}")
    if not passage_index_path.is_file():
        raise FileNotFoundError(f"Passage 倒排索引不存在: {passage_index_path}")

    try:
        vectors = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Dense 向量文件损坏: {vector_path}") from error
    _validate_vectors(vectors, metadata["chunk_count"])

    records = []
    try:
        with chunk_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("document_id"), int)
                    or not isinstance(record.get("chunk_index"), int)
                    or not isinstance(record.get("text"), str)
                ):
                    raise ValueError(f"Dense 分块第 {line_number} 行格式无效")
                records.append(record)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Dense 分块文件损坏: {chunk_path}") from error

    if len(records) != metadata["chunk_count"]:
        raise ValueError(
            "Dense 分块数量与元数据不一致: "
            f"期望 {metadata['chunk_count']}，实际 {len(records)}"
        )

    return vectors, tuple(records), metadata


def _build_passage_sparse_index(
    path: Path,
    records: list[dict[str, int | str]],
) -> None:
    """Build an FTS5 BM25 index over the exact Dense passage boundaries."""

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute(
            """
            CREATE VIRTUAL TABLE passages USING fts5(
                title_tokens,
                text_tokens,
                document_id UNINDEXED,
                chunk_index UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )
        batch = []
        for record in records:
            title = text_normalize(str(record.get("title", "")))
            text = text_normalize(str(record["text"]))
            batch.append((
                " ".join(
                    term
                    for term, _offset in tokenize_with_positions(title, mode="search")
                ),
                " ".join(
                    term
                    for term, _offset in tokenize_with_positions(text, mode="search")
                ),
                int(record["document_id"]),
                int(record["chunk_index"]),
            ))
            if len(batch) == 1000:
                connection.executemany(
                    "INSERT INTO passages VALUES (?, ?, ?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO passages VALUES (?, ?, ?, ?)",
                batch,
            )
        connection.execute("INSERT INTO passages(passages) VALUES ('optimize')")


def _passage_from_row(
    records: tuple[dict[str, int | str], ...],
    row_index: int,
    score: float,
) -> PassageHit:
    record = records[row_index]
    return PassageHit(
        document_id=int(record["document_id"]),
        chunk_index=int(record["chunk_index"]),
        score=float(score),
        text=text_normalize(str(record["text"])),
        title=text_normalize(str(record.get("title", ""))),
    )


def search_dense_passages(
    query: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    limit: int = 100,
    device: str | None = None,
) -> list[PassageHit]:
    """Return individual passages ranked by embedding similarity."""

    query = text_normalize(query)
    if not query or limit <= 0:
        return []

    index_dir = Path(index_dir)
    vectors, records, metadata = _load_dense_index(str(index_dir.resolve()))
    if not records:
        return []

    model = _load_model(metadata["model_path"], device)
    query_vectors = _encode(
        model,
        [metadata["query_instruction"] + query],
        batch_size=1,
        show_progress_bar=False,
    )
    _validate_vectors(query_vectors, 1)
    scores = vectors @ query_vectors[0]
    order = np.argsort(-scores, kind="stable")[:limit]
    return [
        _passage_from_row(records, int(row_index), float(scores[row_index]))
        for row_index in order
    ]


def search_sparse_passages(
    query: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    limit: int = 100,
) -> list[PassageHit]:
    """Return individual passages ranked by passage-level BM25."""

    query = text_normalize(query)
    if not query or limit <= 0:
        return []
    terms = list(dict.fromkeys(
        term
        for term, _offset in tokenize_with_positions(query, mode="search")
    ))
    if not terms:
        return []

    index_dir = Path(index_dir)
    _vectors, records, metadata = _load_dense_index(str(index_dir.resolve()))
    passage_index_path = index_dir.resolve() / str(metadata["passage_index_file"])
    match_query = " OR ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"'
        for term in terms
    )
    with sqlite3.connect(passage_index_path) as connection:
        rows = connection.execute(
            """
            SELECT rowid, bm25(passages, 2.0, 1.0)
            FROM passages
            WHERE passages MATCH ?
            ORDER BY bm25(passages, 2.0, 1.0), rowid
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
    return [
        _passage_from_row(records, int(row_id) - 1, -float(rank))
        for row_id, rank in rows
    ]


def search_dense(
    query: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    limit: int = 20,
    device: str | None = None,
) -> list[DenseHit]:
    """Return distinct documents ranked by their best matching chunk."""

    query = text_normalize(query)
    if not query or limit <= 0:
        return []

    index_dir = Path(index_dir)
    vectors, records, metadata = _load_dense_index(str(index_dir.resolve()))
    if not records:
        return []

    model = _load_model(metadata["model_path"], device)
    query_vectors = _encode(
        model,
        [metadata["query_instruction"] + query],
        batch_size=1,
        show_progress_bar=False,
    )
    _validate_vectors(query_vectors, 1)

    scores = vectors @ query_vectors[0]
    order = np.argsort(-scores, kind="stable")
    seen = set()
    hits = []

    for row_index in order:
        record = records[int(row_index)]
        document_id = int(record["document_id"])
        if document_id in seen:
            continue
        seen.add(document_id)
        hits.append(DenseHit(
            document_id=document_id,
            score=float(scores[row_index]),
            snippet=text_normalize(str(record["text"])),
        ))
        if len(hits) == limit:
            break

    return hits


def warmup_dense(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    device: str | None = None,
) -> None:
    """Load the Dense index and model before request processing starts."""

    _vectors, records, metadata = _load_dense_index(
        str(Path(index_dir).resolve())
    )
    if records:
        _load_model(metadata["model_path"], device)
