import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from zeta_engine.storage import Storage


class StorageTest(unittest.TestCase):
    def test_requires_at_least_one_database(self) -> None:
        with self.assertRaises(ValueError):
            Storage()

    def test_opens_only_requested_database(self) -> None:
        with Storage(document_db=":memory:") as storage:
            self.assertIsNotNone(storage.documents)
            self.assertIsNone(storage.queue)
            self.assertIsNone(storage.index)

        with Storage(queue_db=":memory:") as storage:
            self.assertIsNone(storage.documents)
            self.assertIsNotNone(storage.queue)
            self.assertIsNone(storage.index)

        with Storage(index_db=":memory:") as storage:
            self.assertIsNone(storage.documents)
            self.assertIsNone(storage.queue)
            self.assertIsNotNone(storage.index)

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

    def test_migrates_and_requeues_legacy_crawl_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue_db = Path(directory) / "queue.db"
            with closing(sqlite3.connect(queue_db)) as connection:
                connection.execute(
                    """
                    CREATE TABLE crawl_tasks (
                        url TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO crawl_tasks VALUES (?, ?, ?, ?)",
                    (
                        ("https://done.test/", "done", 1, None),
                        ("https://failed.test/", "failed", 3, "timeout"),
                    ),
                )
                connection.commit()

            with Storage(queue_db=queue_db) as storage:
                assert storage.queue is not None
                self.assertEqual(
                    storage.queue.requeue(
                        done_before="9999-12-31T23:59:59+00:00",
                        failed=True,
                    ),
                    {"done": 1, "failed": 1},
                )
                self.assertEqual(
                    storage.queue.count_by_state(),
                    {"pending": 2},
                )

            with closing(sqlite3.connect(queue_db)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(crawl_tasks)"
                    )
                }
            self.assertIn("finished_at", columns)

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
