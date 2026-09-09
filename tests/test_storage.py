import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from zeta_engine.infrastructure.storage import Storage


class StorageTest(unittest.TestCase):
    def test_migrates_and_updates_structured_document_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with closing(sqlite3.connect(document_db)) as connection:
                connection.execute(
                    """
                    CREATE TABLE documents (
                        id INTEGER PRIMARY KEY,
                        url TEXT NOT NULL UNIQUE,
                        title TEXT NOT NULL,
                        text TEXT NOT NULL,
                        fetched_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO documents VALUES (1, ?, ?, ?, ?)",
                    ("https://a.test/", "旧标题", "旧正文", "2026-08-28"),
                )
                connection.commit()

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                self.assertEqual(storage.documents.get_content_html(1), "")
                document_id = storage.documents.save(
                    url="https://a.test/",
                    title="新标题",
                    text="新正文",
                    fetched_at="2026-08-29",
                    content_html="<main><p>新正文</p></main>",
                )
                self.assertEqual(document_id, 1)
                self.assertEqual(
                    storage.documents.get_content_html(1),
                    "<main><p>新正文</p></main>",
                )
                self.assertEqual(
                    storage.documents.get(1),
                    ("https://a.test/", "新标题", "新正文", "2026-08-29"),
                )
                self.assertEqual(
                    list(storage.documents.iter_all_with_content_html()),
                    [(
                        1,
                        "https://a.test/",
                        "新标题",
                        "新正文",
                        "2026-08-29",
                        "<main><p>新正文</p></main>",
                    )],
                )

    def test_replaces_indexed_document(self) -> None:
        with Storage(index_db=":memory:") as storage:
            assert storage.index is not None

            storage.index.replace_document(
                document_id=1,
                fetched_at="2026-08-26T10:00:00",
                title_norm="中国人民大学",
                text_norm="中国人民大学位于北京",
                postings={
                    ("中国人民大学", storage.index.TITLE): [0],
                    ("北京", storage.index.TEXT): [8, 42],
                },
            )

            self.assertEqual(
                storage.index.get_document_by_index(1),
                (
                    "2026-08-26T10:00:00",
                    "中国人民大学",
                    "中国人民大学位于北京",
                ),
            )
            self.assertEqual(
                storage.index.lookup_posting("北京", storage.index.TEXT),
                {1: [8, 42]},
            )
            self.assertEqual(storage.index.count_terms(), 2)

            storage.index.replace_document(
                document_id=1,
                fetched_at="2026-08-26T11:00:00",
                title_norm="中国人民大学",
                text_norm="中国人民大学位于苏州",
                postings={
                    ("中国人民大学", storage.index.TITLE): [0],
                    ("苏州", storage.index.TEXT): [8],
                },
            )

            self.assertEqual(
                storage.index.lookup_posting("北京", storage.index.TEXT),
                {},
            )
            self.assertEqual(
                storage.index.lookup_posting("苏州", storage.index.TEXT),
                {1: [8]},
            )
            self.assertEqual(storage.index.count_terms(), 2)


if __name__ == "__main__":
    unittest.main()
