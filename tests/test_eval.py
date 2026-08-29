import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from zeta_engine.dense import DenseHit
from zeta_engine.eval import (
    DEFAULT_BASE_URL,
    evaluate,
    run_evaluation,
    run_rag_evaluation,
    send_answers,
    send_rag_answers,
)
from zeta_engine.index import build_index
from zeta_engine.storage import Storage


class EvalTest(unittest.TestCase):
    def test_send_answers_reads_debug_details(self) -> None:
        response = Mock()
        response.text = repr({
            "mode": "debug",
            "mrr": 0.75,
            "details": [1, 0.5],
            "average_latency_seconds": 0.2,
        })

        with patch("zeta_engine.eval.requests.post", return_value=response):
            self.assertEqual(
                send_answers(
                    "http://example.test",
                    "student",
                    "",
                    [[], []],
                    [0.1, 0.3],
                ),
                ("debug", 0.75, [1.0, 0.5], 0.2),
            )

    def test_send_rag_answers_reads_debug_judge_outputs(self) -> None:
        response = Mock()
        response.text = repr({
            "mode": "debug",
            "score": 0.8,
            "details": [0.8],
            "average_latency_seconds": 0.2,
            "judge_outputs": [{"score": 0.8, "reason": "正确"}],
        })

        with patch(
            "zeta_engine.eval.requests.post",
            return_value=response,
        ) as post:
            result = send_rag_answers(
                "http://example.test",
                "student",
                "",
                ["答案"],
                [0.2],
            )

        self.assertEqual(
            result,
            (
                "debug",
                0.8,
                [0.8],
                0.2,
                [{"score": 0.8, "reason": "正确"}],
            ),
        )
        self.assertEqual(post.call_args.args[0], "http://example.test/rag/score")

    def test_runs_rag_evaluation_and_submits_answers(self) -> None:
        result = [{
            "title": "标题",
            "url": "https://example.test",
            "snippet": "材料",
        }]

        def answer(document_db, index_db, query, **kwargs):
            return {"answer": f"回答 {query}", "results": result}

        with (
            patch("zeta_engine.eval.input_idx", return_value="student"),
            patch("zeta_engine.eval.input_passwd", return_value=""),
            patch("zeta_engine.eval.rag_login", return_value=["问题"]),
            patch(
                "zeta_engine.eval.answer_question",
                side_effect=answer,
            ) as answer_question,
            patch(
                "zeta_engine.eval.send_rag_answers",
                return_value=(
                    "debug",
                    1.0,
                    [1.0],
                    0.1,
                    [{"score": 1.0, "reason": "正确"}],
                ),
            ) as send_rag_answers_mock,
            redirect_stdout(output := io.StringIO()),
        ):
            run_rag_evaluation(
                "documents.db",
                "index.db",
                top_k=7,
            )

        answer_question.assert_called_once_with(
            "documents.db",
            "index.db",
            "问题",
            top_k=7,
            ranking="hybrid",
            dense_index=Path("data/dense"),
            reranker_model=Path("models/bge-reranker-base"),
            rerank_candidates=50,
            reranker_batch_size=16,
            device=None,
            alpha=.5,
        )
        self.assertEqual(send_rag_answers_mock.call_args.args[3], ["回答 问题"])
        self.assertIn("Judge 1: score=1.0, reason=正确", output.getvalue())

    def test_debug_evaluation_prints_per_query_scores(self) -> None:
        first_urls = [f"https://a{rank}.test" for rank in range(1, 21)]
        second_urls = [f"https://b{rank}.test" for rank in range(1, 21)]

        with (
            patch("zeta_engine.eval.input_idx", return_value="student"),
            patch("zeta_engine.eval.input_passwd", return_value=""),
            patch("zeta_engine.eval.login", return_value=["q1", "q2"]),
            patch("zeta_engine.eval.Storage"),
            patch(
                "zeta_engine.eval.evaluate",
                side_effect=[first_urls, second_urls],
            ),
            patch(
                "zeta_engine.eval.send_answers",
                return_value=("debug", 2 / 3, [1.0, 1 / 3], 0.1),
            ) as send_answers,
            redirect_stdout(output := io.StringIO()),
        ):
            run_evaluation("documents.db", "index.db")

        send_answers.assert_called_once()
        self.assertEqual(
            send_answers.call_args.args[:4],
            (
                DEFAULT_BASE_URL,
                "student",
                "",
                [first_urls, second_urls],
            ),
        )
        self.assertIn(
            "Per-query reciprocal ranks: [1.0, 0.3333333333333333]",
            output.getvalue(),
        )

    def test_evaluate_returns_ranked_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"

            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                storage.documents.save(
                    url="https://info.ruc.edu.cn/example",
                    title="中国人民大学",
                    text="信息学院",
                    fetched_at="2026-08-28T10:00:00",
                )
            with Storage(
                document_db=document_db,
                index_db=index_db,
            ) as storage:
                build_index(storage, mode="search")
                self.assertEqual(
                    evaluate(storage, "中国人民大学", ranking="bm25f"),
                    ["https://info.ruc.edu.cn/example"],
                )

    def test_evaluate_supports_dense_without_sparse_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            with Storage(document_db=document_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://info.ruc.edu.cn/dense",
                    title="Dense",
                    text="语义检索",
                    fetched_at="2026-08-28T10:00:00",
                )
                with patch(
                    "zeta_engine.eval.search_dense",
                    return_value=[DenseHit(document_id, 0.9, "语义检索")],
                ) as search_dense:
                    urls = evaluate(
                        storage,
                        "查询",
                        ranking="dense",
                        dense_index=Path("dense"),
                        device="mps",
                    )

            self.assertEqual(urls, ["https://info.ruc.edu.cn/dense"])
            search_dense.assert_called_once_with(
                "查询",
                Path("dense"),
                limit=20,
                device="mps",
            )

    def test_evaluate_supports_hybrid_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"
            with Storage(document_db=document_db, index_db=index_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://info.ruc.edu.cn/hybrid",
                    title="Hybrid",
                    text="混合检索",
                    fetched_at="2026-08-28T10:00:00",
                )
                with patch(
                    "zeta_engine.eval.search_hybrid",
                    return_value=[document_id],
                ) as search_hybrid:
                    urls = evaluate(
                        storage,
                        "查询",
                        ranking="hybrid",
                        dense_index=Path("dense"),
                        device="mps",
                        alpha=.7,
                    )

            self.assertEqual(urls, ["https://info.ruc.edu.cn/hybrid"])
            search_hybrid.assert_called_once_with(
                storage,
                "查询",
                Path("dense"),
                limit=20,
                alpha=.7,
                device="mps",
            )

    def test_evaluate_supports_cross_encoder_reranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_db = Path(directory) / "documents.db"
            index_db = Path(directory) / "index.db"
            with Storage(document_db=document_db, index_db=index_db) as storage:
                assert storage.documents is not None
                document_id = storage.documents.save(
                    url="https://info.ruc.edu.cn/reranked",
                    title="Reranked",
                    text="重排结果",
                    fetched_at="2026-08-28T10:00:00",
                )
                with patch(
                    "zeta_engine.eval.search_reranked",
                    return_value=[document_id],
                ) as search_reranked:
                    urls = evaluate(
                        storage,
                        "查询",
                        ranking="rerank",
                        dense_index=Path("dense"),
                        reranker_model=Path("reranker"),
                        rerank_candidates=40,
                        reranker_batch_size=8,
                        device="mps",
                        alpha=.7,
                    )

            self.assertEqual(urls, ["https://info.ruc.edu.cn/reranked"])
            search_reranked.assert_called_once_with(
                storage,
                "查询",
                Path("dense"),
                reranker_model=Path("reranker"),
                limit=20,
                candidate_limit=40,
                batch_size=8,
                alpha=.7,
                device="mps",
            )


if __name__ == "__main__":
    unittest.main()
