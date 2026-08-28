import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import ANY, patch

from zeta_engine.cli import build_parser
from zeta_engine.dense import DenseHit
from zeta_engine.eval import DEFAULT_BASE_URL
from zeta_engine.storage import Storage


class CliTest(unittest.TestCase):
    def test_crawl_accepts_resiliparse_extractor(self) -> None:
        args = build_parser().parse_args([
            "crawl",
            "--extractor", "resiliparse",
        ])

        self.assertEqual(args.extractor, "resiliparse")

    def test_runs_evaluation_from_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"
            dense_index = Path(directory) / "dense"
            document_db.touch()
            index_db.touch()
            dense_index.mkdir()
            (dense_index / "metadata.json").touch()

            args = build_parser().parse_args([
                "eval",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--dense-index", str(dense_index),
                "--base-url", "http://localhost:8080",
            ])
            with patch("zeta_engine.cli.run_evaluation") as run_evaluation:
                self.assertEqual(args.handler(args), 0)

            run_evaluation.assert_called_once_with(
                document_db,
                index_db,
                base_url="http://localhost:8080",
                ranking="hybrid",
                dense_index=dense_index,
                reranker_model=Path("models/bge-reranker-base"),
                rerank_candidates=50,
                reranker_batch_size=16,
                device=None,
                alpha=.5,
            )

    def test_runs_rag_from_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_db = root / "documents.db"
            dense_index = root / "dense"
            document_db.touch()
            dense_index.mkdir()
            (dense_index / "metadata.json").touch()
            result = {
                "title": "资助政策",
                "url": "https://example.test/policy",
                "snippet": "学生可以申请。",
            }

            args = build_parser().parse_args([
                "rag", "如何申请资助？",
                "--document-db", str(document_db),
                "--dense-index", str(dense_index),
                "--top-k", "3",
                "--device", "mps",
            ])
            with (
                patch(
                    "zeta_engine.cli.rag_answer",
                    return_value={
                        "answer": "可以申请。[文档1]",
                        "results": [result],
                    },
                ) as rag_answer,
                redirect_stdout(output := io.StringIO()),
            ):
                self.assertEqual(args.handler(args), 0)

            self.assertIn("可以申请。[文档1]", output.getvalue())
            self.assertIn("https://example.test/policy", output.getvalue())
            self.assertEqual(rag_answer.call_args.args[0], "如何申请资助？")
            self.assertEqual(rag_answer.call_args.kwargs["top_k"], 3)

            search_fn = rag_answer.call_args.args[1]
            with patch(
                "zeta_engine.cli.search_documents",
                return_value=[result],
            ) as search_documents:
                self.assertEqual(search_fn("查询", 2), [result])
            search_documents.assert_called_once_with(
                document_db,
                Path("data/index.db"),
                "查询",
                2,
                ranking="dense",
                dense_index=dense_index,
                reranker_model=Path("models/bge-reranker-base"),
                rerank_candidates=50,
                reranker_batch_size=16,
                device="mps",
                alpha=.5,
            )

    def test_builds_and_searches_dense_index_without_sparse_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_db = root / "documents.db"
            dense_index = root / "dense"
            model = root / "model"
            model.mkdir()
            dense_index.mkdir()
            (dense_index / "metadata.json").touch()

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://example.test/dense",
                    title="Ｄｅｎｓｅ 结果",
                    text="原始正文",
                    fetched_at="2026-08-28T10:00:00",
                )

            parser = build_parser()
            self.assertEqual(
                parser.parse_args(["search", "查询"]).ranking,
                "hybrid",
            )
            build_args = parser.parse_args([
                "dense-index",
                "--document-db", str(document_db),
                "--dense-index", str(dense_index),
                "--model", str(model),
                "--device", "mps",
                "--log-file", str(root / "dense.log"),
            ])
            with (
                patch("zeta_engine.cli.configure_logging"),
                patch(
                    "zeta_engine.cli.build_dense_index",
                    return_value={"documents": 1},
                ) as build_dense_index,
            ):
                self.assertEqual(build_args.handler(build_args), 0)
            build_dense_index.assert_called_once_with(
                ANY,
                dense_index,
                model_path=model,
                batch_size=32,
                device="mps",
            )

            search_args = parser.parse_args([
                "search", "语义查询",
                "--document-db", str(document_db),
                "--ranking", "dense",
                "--dense-index", str(dense_index),
                "--device", "mps",
            ])
            with patch(
                "zeta_engine.cli.search_dense",
                return_value=[DenseHit(document_id, 0.9, "最佳分块１０")],
            ) as search_dense, redirect_stdout(output := io.StringIO()):
                self.assertEqual(search_args.handler(search_args), 0)

            search_dense.assert_called_once_with(
                "语义查询",
                dense_index,
                limit=20,
                device="mps",
            )
            self.assertIn("dense 结果", output.getvalue())
            self.assertIn("最佳分块10", output.getvalue())

            evaluation_args = parser.parse_args([
                "eval",
                "--document-db", str(document_db),
                "--ranking", "dense",
                "--dense-index", str(dense_index),
                "--device", "mps",
            ])
            with patch("zeta_engine.cli.run_evaluation") as run_evaluation:
                self.assertEqual(evaluation_args.handler(evaluation_args), 0)
            run_evaluation.assert_called_once_with(
                document_db,
                Path("data/index.db"),
                base_url=DEFAULT_BASE_URL,
                ranking="dense",
                dense_index=dense_index,
                reranker_model=Path("models/bge-reranker-base"),
                rerank_candidates=50,
                reranker_batch_size=16,
                device="mps",
                alpha=.5,
            )

            index_db = root / "index.db"
            index_db.touch()
            hybrid_args = parser.parse_args([
                "search", "混合查询",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--ranking", "hybrid",
                "--dense-index", str(dense_index),
                "--device", "mps",
                "--alpha", "0.7",
                "--limit", "7",
            ])
            with patch(
                "zeta_engine.cli.search_hybrid",
                return_value=[document_id],
            ) as search_hybrid, redirect_stdout(io.StringIO()):
                self.assertEqual(hybrid_args.handler(hybrid_args), 0)
            search_hybrid.assert_called_once_with(
                ANY,
                "混合查询",
                dense_index,
                limit=7,
                alpha=.7,
                device="mps",
            )

            reranker_model = root / "reranker"
            reranker_model.mkdir()
            rerank_args = parser.parse_args([
                "search", "重排查询",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--ranking", "rerank",
                "--dense-index", str(dense_index),
                "--reranker-model", str(reranker_model),
                "--rerank-candidates", "40",
                "--reranker-batch-size", "8",
                "--device", "mps",
            ])
            with patch(
                "zeta_engine.cli.search_reranked",
                return_value=[document_id],
            ) as search_reranked, redirect_stdout(io.StringIO()):
                self.assertEqual(rerank_args.handler(rerank_args), 0)
            search_reranked.assert_called_once_with(
                ANY,
                "重排查询",
                dense_index,
                reranker_model=reranker_model,
                limit=20,
                candidate_limit=40,
                batch_size=8,
                alpha=.5,
                device="mps",
            )

    def test_builds_search_index_and_reports_term_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"
            log_file = Path(directory) / "index.log"

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                storage.documents.save(
                    url="https://info.ruc.edu.cn/example",
                    title="中国人民大学",
                    text="信息学院",
                    fetched_at="2026-08-26T10:00:00",
                )

            parser = build_parser()
            index_args = parser.parse_args([
                "index",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--log-file", str(log_file),
                "--mode", "search",
            ])
            with patch("zeta_engine.cli.logging.info") as log_info:
                self.assertEqual(index_args.handler(index_args), 0)
            self.assertTrue(log_file.is_file())
            self.assertEqual(log_info.call_count, 2)

            stats_args = parser.parse_args([
                "stats",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
            ])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(stats_args.handler(stats_args), 0)

            term_count = int(output.getvalue().rsplit("terms: ", 1)[1])
            self.assertGreater(term_count, 0)

            search_args = parser.parse_args([
                "search", "中国人民大学",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--ranking", "bm25f",
            ])
            self.assertEqual(search_args.ranking, "bm25f")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(search_args.handler(search_args), 0)
            self.assertIn("https://info.ruc.edu.cn/example", output.getvalue())

            bm25f_args = parser.parse_args([
                "search", "中国人民大学",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--ranking", "bm25f",
            ])
            with patch(
                "zeta_engine.cli.search_bm25f",
                return_value=[],
            ) as search_bm25f:
                self.assertEqual(bm25f_args.handler(bm25f_args), 0)
            search_bm25f.assert_called_once_with(ANY, "中国人民大学")


if __name__ == "__main__":
    unittest.main()
