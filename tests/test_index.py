import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from zeta_engine.dense import DenseHit
from zeta_engine.index import build_index
from zeta_engine.search import (
    _fuse_scores,
    search_bm25f,
    search_hybrid,
    search_hybrid_with_snippets,
    search_phrase,
    search_reranked,
)
from zeta_engine.storage import Storage
from zeta_engine.tokenizer import (
    load_stopwords,
    text_normalize,
    tokenize_with_positions,
)


class CharacterTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        verbose: bool = True,
    ) -> list[int]:
        del add_special_tokens, verbose
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        del pair
        return 3


class IndexTest(unittest.TestCase):
    def test_normalizes_compatibility_characters(self) -> None:
        self.assertEqual(
            text_normalize(" １０月１４日 ＶＣＲ  "),
            "10月14日 vcr",
        )

    def test_hybrid_combines_sparse_and_dense_results(self) -> None:
        with (
            patch("zeta_engine.search._score_bm25f", return_value={1: 2., 2: 1.}),
            patch(
                "zeta_engine.search.search_dense",
                return_value=[DenseHit(2, .9, ""), DenseHit(3, .5, "")],
            ) as search_dense,
        ):
            result = search_hybrid(
                Mock(),
                "query",
                Path("dense"),
                limit=2,
                alpha=.5,
                device="cpu",
            )

        self.assertEqual(result, [1, 2])
        search_dense.assert_called_once_with(
            "query",
            Path("dense"),
            limit=100,
            device="cpu",
        )

    def test_hybrid_preserves_dense_snippets(self) -> None:
        with (
            patch("zeta_engine.search._score_bm25f", return_value={1: 2.}),
            patch(
                "zeta_engine.search.search_dense",
                return_value=[DenseHit(2, .9, "相关片段")],
            ),
        ):
            document_ids, snippets = search_hybrid_with_snippets(
                Mock(),
                "query",
                Path("dense"),
                limit=2,
            )

        self.assertEqual(document_ids, [1, 2])
        self.assertEqual(snippets, {2: "相关片段"})

    def test_cross_encoder_reranks_hybrid_candidates(self) -> None:
        with Storage(document_db=":memory:") as storage:
            assert storage.documents is not None
            first = storage.documents.save(
                url="https://example.test/first",
                title="第一篇ＶＣＲ",
                text="普通正文１０",
                fetched_at="2026-08-28T10:00:00",
            )
            second = storage.documents.save(
                url="https://example.test/second",
                title="第二篇",
                text="真正相关的正文",
                fetched_at="2026-08-28T10:00:00",
            )
            model = Mock()
            model.tokenizer = CharacterTokenizer()
            model.predict.return_value = [0.1, 0.9]

            with (
                patch(
                    "zeta_engine.search.search_hybrid",
                    return_value=[first, second],
                ) as search_hybrid,
                patch(
                    "zeta_engine.search._load_cross_encoder",
                    return_value=model,
                ),
            ):
                result = search_reranked(
                    storage,
                    "相关ＱＵＥＲＹ",
                    Path("dense"),
                    reranker_model=Path("reranker"),
                    limit=2,
                    candidate_limit=10,
                    batch_size=4,
                    alpha=.7,
                    device="mps",
                )

        self.assertEqual(result, [second, first])
        search_hybrid.assert_called_once_with(
            storage,
            "相关query",
            Path("dense"),
            limit=10,
            alpha=.7,
            device="mps",
        )
        self.assertEqual(
            model.predict.call_args.args[0],
            [
                ("相关query", "第一篇vcr\n普通正文10"),
                ("相关query", "第二篇\n真正相关的正文"),
            ],
        )
        self.assertEqual(model.predict.call_args.kwargs["batch_size"], 4)

    def test_cross_encoder_uses_dynamic_query_focused_passages(self) -> None:
        with Storage(document_db=":memory:") as storage:
            assert storage.documents is not None
            long_document = storage.documents.save(
                url="https://example.test/long",
                title="长文档",
                text="目标开头" + "无关内容" * 400 + "末尾目标证据",
                fetched_at="2026-08-28T10:00:00",
            )
            distractor = storage.documents.save(
                url="https://example.test/distractor",
                title="干扰文档",
                text="表面相关",
                fetched_at="2026-08-28T10:00:00",
            )
            model = Mock()
            model.tokenizer = CharacterTokenizer()

            def score_passages(pairs, **_kwargs):
                return [
                    .9 if "末尾目标证据" in passage
                    else .5 if "表面相关" in passage
                    else .1
                    for _query, passage in pairs
                ]

            model.predict.side_effect = score_passages

            with (
                patch(
                    "zeta_engine.search.search_hybrid",
                    return_value=[long_document, distractor],
                ),
                patch(
                    "zeta_engine.search._load_cross_encoder",
                    return_value=model,
                ),
            ):
                result = search_reranked(
                    storage,
                    "目标查询",
                    Path("dense"),
                    reranker_model=Path("reranker"),
                    limit=2,
                )

        self.assertEqual(result, [long_document, distractor])
        pairs = model.predict.call_args.args[0]
        long_passages = [
            passage
            for _query, passage in pairs
            if passage.startswith("长文档\n")
        ]
        self.assertEqual(len(long_passages), 2)
        self.assertIn("目标开头", long_passages[0])
        self.assertIn("末尾目标证据", long_passages[1])

    def test_normalizes_and_linearly_fuses_sparse_and_dense_scores(self) -> None:
        sparse = {1: 100., 2: 80., 3: 0.}
        dense = {3: .9, 2: .72, 4: 0.}

        self.assertEqual(
            _fuse_scores(sparse, dense, alpha=0., limit=3),
            [1, 2, 3],
        )
        self.assertEqual(
            _fuse_scores(sparse, dense, alpha=1., limit=3),
            [3, 2, 4],
        )
        self.assertEqual(
            _fuse_scores(sparse, dense, alpha=.5, limit=4),
            [2, 1, 3, 4],
        )
        self.assertEqual(
            _fuse_scores(
                {document_id: score * 1000 for document_id, score in sparse.items()},
                dense,
                alpha=.5,
                limit=4,
            ),
            [2, 1, 3, 4],
        )

        with self.assertRaisesRegex(ValueError, "alpha"):
            _fuse_scores(sparse, dense, alpha=1.1, limit=3)

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
        self.assertNotIn("10", load_stopwords())
        self.assertEqual(
            tokenize_with_positions("人民大学 的招生。"),
            [("人民大学", 0), ("招生", 6)],
        )

    @patch(
        "zeta_engine.tokenizer.jieba.tokenize",
        return_value=[("10", 0, 2), ("月", 2, 3), ("14", 3, 5), ("日", 5, 6)],
    )
    def test_tokenizer_keeps_date_numbers(self, _tokenize) -> None:
        self.assertEqual(
            tokenize_with_positions("10月14日"),
            [("10", 0), ("月", 2), ("14", 3), ("日", 5)],
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
                patch("zeta_engine.index.tokenize_with_positions", side_effect=tokenize),
                patch("zeta_engine.search.tokenize_with_positions", side_effect=tokenize),
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
                "zeta_engine.index.tokenize_with_positions",
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
