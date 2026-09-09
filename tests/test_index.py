import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from zeta_engine.retrieval.dense import PassageHit
from zeta_engine.retrieval.index import build_index
from zeta_engine.retrieval.search import (
    search_bm25f,
    search_hybrid_passages,
    search_phrase,
    search_reranked_passages,
)
from zeta_engine.infrastructure.storage import Storage


class IndexTest(unittest.TestCase):
    def test_hybrid_fuses_scores_for_the_same_passage_identity(self) -> None:
        sparse = [
            PassageHit(1, 0, 2.0, "稀疏第一", "标题一"),
            PassageHit(2, 0, 1.0, "稀疏第二", "标题二"),
        ]
        dense_hits = [
            PassageHit(2, 0, .9, "稀疏第二", "标题二"),
            PassageHit(3, 1, .5, "向量第三", "标题三"),
        ]
        with (
            patch(
                "zeta_engine.retrieval.search.search_sparse_passages",
                return_value=sparse,
            ),
            patch(
                "zeta_engine.retrieval.search.search_dense_passages",
                return_value=dense_hits,
            ),
        ):
            result = search_hybrid_passages(
                "query",
                Path("dense"),
                limit=3,
                alpha=.5,
                device="cpu",
            )

        self.assertEqual(
            [(hit.document_id, hit.chunk_index) for hit in result],
            [(1, 0), (2, 0), (3, 1)],
        )

    def test_cross_encoder_reranks_exact_passages(self) -> None:
        candidates = [
            PassageHit(1, 0, .9, "第一段", "第一篇"),
            PassageHit(1, 1, .8, "第二段", "第一篇"),
            PassageHit(2, 0, .7, "真正证据", "第二篇"),
        ]
        model = Mock()
        model.predict.return_value = [.1, .2, .9]
        with (
            patch(
                "zeta_engine.retrieval.search.search_hybrid_passages",
                return_value=candidates,
            ),
            patch(
                "zeta_engine.retrieval.search._load_cross_encoder",
                return_value=model,
            ),
        ):
            result = search_reranked_passages(
                "问题",
                Path("dense"),
                reranker_model=Path("reranker"),
                limit=2,
            )

        self.assertEqual([hit.passage_id for hit in result], ["2:0", "1:1"])
        self.assertEqual(
            model.predict.call_args.args[0],
            [
                ("问题", "第一篇\n第一段"),
                ("问题", "第一篇\n第二段"),
                ("问题", "第二篇\n真正证据"),
            ],
        )

    def test_searches_phrases(self) -> None:
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
                patch("zeta_engine.retrieval.index.tokenize_with_positions", side_effect=tokenize),
                patch("zeta_engine.retrieval.search.tokenize_with_positions", side_effect=tokenize),
            ):
                build_index(storage)
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
                "zeta_engine.retrieval.index.tokenize_with_positions",
                side_effect=lambda _text, mode: [(mode, 0)],
            ):
                self.assertEqual(build_index(storage, "default")["indexed"], 1)
                self.assertEqual(build_index(storage, "default")["skipped"], 1)
                self.assertEqual(build_index(storage, "search")["indexed"], 1)
                storage.index.set_metadata("stopwords_version", "outdated")
                self.assertEqual(build_index(storage, "search")["indexed"], 1)
                storage.index.set_metadata(
                    "text_normalizer_version",
                    "outdated",
                )
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
                patch("zeta_engine.retrieval.index.tokenize_with_positions", side_effect=tokenize),
                patch("zeta_engine.retrieval.search.tokenize_with_positions", side_effect=tokenize),
            ):
                build_index(storage)
                self.assertEqual(search_bm25f(storage, "x"), [1, 2, 3])

if __name__ == "__main__":
    unittest.main()
