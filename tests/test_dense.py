import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from zeta_engine import dense
from zeta_engine.storage import Storage


class FakeTokenizer:
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        verbose: bool = True,
    ) -> dict[str, list]:
        del add_special_tokens, return_offsets_mapping, verbose
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1)
                for index in range(len(text))
            ],
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(chr(token_id) for token_id in token_ids)

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        del pair
        return 2


class FakeModel:
    tokenizer = FakeTokenizer()

    def get_embedding_dimension(self) -> int:
        return dense.EMBEDDING_DIMENSION

    def encode(self, texts: list[str], **_kwargs) -> np.ndarray:
        vectors = np.zeros(
            (len(texts), dense.EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        for row, text in enumerate(texts):
            if "甲" in text:
                vectors[row, 0] = 1.0
            elif "乙" in text:
                vectors[row, 1] = 1.0
            else:
                vectors[row, 2] = 1.0
        return vectors


def write_index(
    directory: Path,
    vectors: np.ndarray,
    records: list[dict],
    *,
    chunk_count: int | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", vectors, allow_pickle=False)
    with (directory / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    dense._build_passage_sparse_index(directory / "passages.db", records)
    (directory / "metadata.json").write_text(
        json.dumps({
            "schema_version": dense.SCHEMA_VERSION,
            "model_path": "models/fake",
            "query_instruction": dense.QUERY_INSTRUCTION,
            "dimension": dense.EMBEDDING_DIMENSION,
            "normalized": True,
            "chunk_count": len(records) if chunk_count is None else chunk_count,
            "vector_file": "vectors.npy",
            "chunk_file": "chunks.jsonl",
            "passage_index_file": "passages.db",
            "passage_tokenizer_mode": "search",
        }),
        encoding="utf-8",
    )
    dense._load_dense_index.cache_clear()


class DenseTest(unittest.TestCase):
    def tearDown(self) -> None:
        dense._load_dense_index.cache_clear()
        dense._load_model.cache_clear()

    def test_chunks_with_overlap_and_truncates_title(self) -> None:
        tokenizer = FakeTokenizer()
        chunks = dense._chunk_document(
            tokenizer,
            "abcdef",
            "abcdefghijklmnop",
            max_tokens=20,
            overlap_tokens=3,
            title_max_tokens=3,
        )

        self.assertEqual(
            [snippet for _passage, snippet in chunks],
            ["abcdefgh", "fghijklm", "klmnop"],
        )
        self.assertTrue(all("标题：abc" in passage for passage, _ in chunks))
        self.assertTrue(all(
            len(tokenizer.encode(passage, add_special_tokens=False)) + 2 <= 20
            for passage, _snippet in chunks
        ))

    def test_chunks_short_title_only_and_empty_documents(self) -> None:
        tokenizer = FakeTokenizer()

        self.assertEqual(
            dense._chunk_document(tokenizer, " 标题 ", ""),
            [("标题：标题", "标题")],
        )
        self.assertEqual(dense._chunk_document(tokenizer, "", ""), [])
        self.assertEqual(
            dense._chunk_document(
                tokenizer,
                "标题ＶＣＲ",
                "短正文１０月１４日",
            )[0][1],
            "短正文10月14日",
        )

    def test_builds_and_searches_a_dense_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_db = root / "documents.db"
            index_dir = root / "dense"

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                first_id = storage.documents.save(
                    url="https://example.test/first",
                    title="甲文档",
                    text="甲相关内容",
                    fetched_at="2026-08-28T10:00:00",
                )
                storage.documents.save(
                    url="https://example.test/second",
                    title="乙文档",
                    text="乙相关内容",
                    fetched_at="2026-08-28T10:00:00",
                )

                with patch("zeta_engine.dense._load_model", return_value=FakeModel()):
                    stats = dense.build_dense_index(
                        storage,
                        index_dir,
                        model_path=root / "model",
                    )
                    hits = dense.search_dense("甲问题", index_dir)

            self.assertEqual(stats["documents"], 2)
            self.assertEqual(stats["indexed_documents"], 2)
            self.assertEqual(stats["chunks"], 2)
            self.assertEqual(hits[0].document_id, first_id)
            self.assertEqual(hits[0].snippet, "甲相关内容")

            metadata = json.loads(
                (index_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dimension"], 512)
            self.assertTrue((index_dir / metadata["vector_file"]).is_file())
            self.assertTrue((index_dir / metadata["chunk_file"]).is_file())
            self.assertTrue(
                (index_dir / metadata["passage_index_file"]).is_file()
            )

    def test_searches_individual_dense_and_sparse_passages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            vectors = np.zeros((3, dense.EMBEDDING_DIMENSION), dtype=np.float32)
            vectors[0, 0] = 1.0
            vectors[1, 0] = 1.0
            vectors[2, 1] = 1.0
            records = [
                {
                    "document_id": 1,
                    "chunk_index": 0,
                    "title": "甲文档",
                    "text": "普通内容",
                },
                {
                    "document_id": 1,
                    "chunk_index": 1,
                    "title": "甲文档",
                    "text": "目标证据",
                },
                {
                    "document_id": 2,
                    "chunk_index": 0,
                    "title": "乙文档",
                    "text": "其他材料",
                },
            ]
            write_index(index_dir, vectors, records)

            with patch("zeta_engine.dense._load_model", return_value=FakeModel()):
                dense_hits = dense.search_dense_passages("甲", index_dir, limit=2)
            sparse_hits = dense.search_sparse_passages(
                "目标证据",
                index_dir,
                limit=2,
            )

            self.assertEqual(
                [(hit.document_id, hit.chunk_index) for hit in dense_hits],
                [(1, 0), (1, 1)],
            )
            self.assertEqual(sparse_hits[0].passage_id, "1:1")
            self.assertEqual(sparse_hits[0].text, "目标证据")

    def test_upgrades_v3_index_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_db = root / "documents.db"
            index_dir = root / "dense"
            index_dir.mkdir()
            vector = np.zeros((1, dense.EMBEDDING_DIMENSION), dtype=np.float32)
            vector[0, 0] = 1.0
            np.save(index_dir / "vectors.npy", vector, allow_pickle=False)
            (index_dir / "chunks.jsonl").write_text(
                json.dumps({
                    "document_id": 1,
                    "chunk_index": 0,
                    "text": "目标证据",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (index_dir / "metadata.json").write_text(json.dumps({
                "schema_version": 3,
                "model_path": "models/fake",
                "query_instruction": dense.QUERY_INSTRUCTION,
                "dimension": dense.EMBEDDING_DIMENSION,
                "normalized": True,
                "document_count": 1,
                "chunk_count": 1,
                "vector_file": "vectors.npy",
                "chunk_file": "chunks.jsonl",
            }), encoding="utf-8")

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                storage.documents.save(
                    url="https://example.test/",
                    title="目标标题",
                    text="目标证据",
                    fetched_at="2026-08-29T10:00:00",
                )
                stats = dense.upgrade_dense_index_v3(storage, index_dir)

            metadata = json.loads(
                (index_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stats, {"documents": 1, "chunks": 1})
            self.assertEqual(metadata["schema_version"], dense.SCHEMA_VERSION)
            self.assertTrue(
                (index_dir / metadata["passage_index_file"]).is_file()
            )
            self.assertEqual(
                dense.search_sparse_passages("目标证据", index_dir)[0].title,
                "目标标题",
            )

            first_metadata = metadata
            with Storage(document_db=document_db) as storage:
                repeated_stats = dense.upgrade_dense_index_v3(storage, index_dir)
            repeated_metadata = json.loads(
                (index_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(repeated_stats, stats)
            self.assertEqual(repeated_metadata, first_metadata)

    def test_searches_best_chunks_and_deduplicates_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            vectors = np.zeros((3, dense.EMBEDDING_DIMENSION), dtype=np.float32)
            vectors[0, :2] = (0.8, 0.6)
            vectors[1, 0] = 1.0
            vectors[2, :2] = (0.6, 0.8)
            records = [
                {"document_id": 1, "chunk_index": 0, "text": "次优分块"},
                {"document_id": 1, "chunk_index": 1, "text": "最佳分块１０"},
                {"document_id": 2, "chunk_index": 0, "text": "第二篇"},
            ]
            write_index(index_dir, vectors, records)

            with patch("zeta_engine.dense._load_model", return_value=FakeModel()):
                hits = dense.search_dense("甲", index_dir)
                limited = dense.search_dense("甲", index_dir, limit=1)

            self.assertEqual([hit.document_id for hit in hits], [1, 2])
            self.assertEqual(hits[0].snippet, "最佳分块10")
            self.assertAlmostEqual(hits[0].score, 1.0)
            self.assertEqual(limited, [hits[0]])
            self.assertEqual(dense.search_dense(" ", index_dir), [])
            self.assertEqual(dense.search_dense("甲", index_dir, limit=0), [])

    def test_empty_index_returns_no_results_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            write_index(
                index_dir,
                np.empty((0, dense.EMBEDDING_DIMENSION), dtype=np.float32),
                [],
            )

            with patch("zeta_engine.dense._load_model") as load_model:
                self.assertEqual(dense.search_dense("查询", index_dir), [])
            load_model.assert_not_called()

    def test_rejects_corrupt_metadata_vectors_and_chunk_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            corrupt_metadata = root / "metadata"
            corrupt_metadata.mkdir()
            (corrupt_metadata / "metadata.json").write_text(
                "not json",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "元数据损坏"):
                dense.search_dense("查询", corrupt_metadata)

            wrong_dimension = root / "dimension"
            vector = np.zeros((1, dense.EMBEDDING_DIMENSION - 1), dtype=np.float32)
            vector[0, 0] = 1.0
            write_index(
                wrong_dimension,
                vector,
                [{"document_id": 1, "chunk_index": 0, "text": "正文"}],
            )
            with self.assertRaisesRegex(ValueError, "向量形状不正确"):
                dense.search_dense("查询", wrong_dimension)

            non_finite = root / "non-finite"
            vector = np.zeros((1, dense.EMBEDDING_DIMENSION), dtype=np.float32)
            vector[0, 0] = np.nan
            write_index(
                non_finite,
                vector,
                [{"document_id": 1, "chunk_index": 0, "text": "正文"}],
            )
            with self.assertRaisesRegex(ValueError, "NaN"):
                dense.search_dense("查询", non_finite)

            wrong_count = root / "count"
            vector = np.zeros((2, dense.EMBEDDING_DIMENSION), dtype=np.float32)
            vector[:, 0] = 1.0
            write_index(
                wrong_count,
                vector,
                [{"document_id": 1, "chunk_index": 0, "text": "正文"}],
                chunk_count=2,
            )
            with self.assertRaisesRegex(ValueError, "分块数量"):
                dense.search_dense("查询", wrong_count)

    def test_failed_build_preserves_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_db = root / "documents.db"
            index_dir = root / "dense"
            index_dir.mkdir()
            metadata_path = index_dir / "metadata.json"
            metadata_path.write_text('{"old": true}\n', encoding="utf-8")

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                storage.documents.save(
                    url="https://example.test/",
                    title="甲文档",
                    text="正文",
                    fetched_at="2026-08-28T10:00:00",
                )
                with (
                    patch("zeta_engine.dense._load_model", return_value=FakeModel()),
                    patch("zeta_engine.dense._encode", side_effect=RuntimeError("boom")),
                    self.assertRaisesRegex(RuntimeError, "boom"),
                ):
                    dense.build_dense_index(
                        storage,
                        index_dir,
                        model_path=root / "model",
                    )

            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"old": true}\n',
            )


if __name__ == "__main__":
    unittest.main()
