import unittest

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
                storage.index.get_indexed_document(1),
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
