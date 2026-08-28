import unittest
from unittest.mock import patch

from zeta_engine.index import build_index
from zeta_engine.search import (
    search_bm25f,
    search_phrase,
    search_query,
    search_term,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import load_stopwords, tokenize_with_positions


class IndexTest(unittest.TestCase):
    @patch(
        "zeta_engine.tokenizer.jieba.tokenize",
        return_value=[
            ("人民大学", 0, 4),
            (" ", 4, 5),
            ("的", 5, 6),
            ("招生", 6, 8),
            ("。", 8, 9),
        ],
    )
    def test_tokenizer_filters_stopwords_and_keeps_positions(self, _tokenize) -> None:
        self.assertIn("的", load_stopwords())
        self.assertEqual(
            tokenize_with_positions("人民大学 的招生。"),
            [("人民大学", 0), ("招生", 6)],
        )

    def test_searches_terms_queries_and_phrases(self) -> None:
        with Storage(document_db=":memory:", index_db=":memory:") as storage:
            assert storage.documents is not None
            storage.documents.save(
                url="https://example.test/1",
                title="人民大学招生",
                text="欢迎报考人民大学",
                fetched_at="2026-08-26T10:00:00",
            )
            storage.documents.save(
                url="https://example.test/2",
                title="大学招生",
                text="人民欢迎你",
                fetched_at="2026-08-26T10:00:00",
            )

            tokenize = lambda text, _mode: [
                (term, text.index(term))
                for term in ("人民", "大学", "招生", "欢迎", "报考", "你")
                if term in text
            ]
            with (
                patch("zeta_engine.index.tokenize_with_positions", side_effect=tokenize),
                patch("zeta_engine.search.tokenize_with_positions", side_effect=tokenize),
            ):
                build_index(storage)
                self.assertEqual(search_term(storage, "人民"), {1, 2})
                self.assertEqual(search_query(storage, "人民 招生"), [1, 2])
                self.assertEqual(search_phrase(storage, "人民大学"), [1])

    def test_changing_mode_rebuilds_unchanged_documents(self) -> None:
        with Storage(document_db=":memory:", index_db=":memory:") as storage:
            assert storage.documents is not None
            assert storage.index is not None
            storage.documents.save(
                url="https://example.test/",
                title="标题",
                text="正文",
                fetched_at="2026-08-26T10:00:00",
            )

            with patch(
                "zeta_engine.index.tokenize_with_positions",
                side_effect=lambda _text, mode: [(mode, 0)],
            ):
                self.assertEqual(build_index(storage, "default")["indexed"], 1)
                self.assertEqual(build_index(storage, "default")["skipped"], 1)
                self.assertEqual(build_index(storage, "search")["indexed"], 1)
                storage.index.set_metadata("stopwords_version", "outdated")
                self.assertEqual(build_index(storage, "search")["indexed"], 1)

            self.assertEqual(
                storage.index.lookup_posting("search", storage.index.TITLE),
                {1: [0]},
            )
            self.assertEqual(storage.index.count_terms(), 1)

    def test_bm25f_boosts_title_and_penalizes_long_fields(self) -> None:
        with Storage(document_db=":memory:", index_db=":memory:") as storage:
            assert storage.documents is not None
            for url, title, text in (
                ("https://example.test/title", "x", "pad"),
                ("https://example.test/short", "pad", "x"),
                ("https://example.test/long", "pad", "x pad pad pad"),
            ):
                storage.documents.save(
                    url=url,
                    title=title,
                    text=text,
                    fetched_at="2026-08-26T10:00:00",
                )

            tokenize = lambda text, _mode: [
                (term, position)
                for position, term in enumerate(text.split())
            ]
            with (
                patch("zeta_engine.index.tokenize_with_positions", side_effect=tokenize),
                patch("zeta_engine.search.tokenize_with_positions", side_effect=tokenize),
            ):
                build_index(storage)
                self.assertEqual(search_bm25f(storage, "x"), [1, 2, 3])

    def test_logs_progress_every_100_documents(self) -> None:
        with Storage(document_db=":memory:", index_db=":memory:") as storage:
            assert storage.documents is not None
            for document_id in range(100):
                storage.documents.save(
                    url=f"https://example.test/{document_id}",
                    title="标题",
                    text="正文",
                    fetched_at="2026-08-26T10:00:00",
                )

            with (
                patch(
                    "zeta_engine.index.tokenize_with_positions",
                    return_value=[("term", 0)],
                ),
                patch("zeta_engine.index.logger.info") as log_info,
            ):
                build_index(storage)

            log_info.assert_called_once_with("索引进度: 已读取 %s 篇", 100)


if __name__ == "__main__":
    unittest.main()
