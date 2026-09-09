import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from zeta_engine.evaluation.runner import _post_with_retry, run_rag_evaluation


class EvalTest(unittest.TestCase):
    def test_rag_request_retries_transient_gateway_failure(self) -> None:
        gateway_failure = Mock(status_code=502)
        success = Mock(status_code=200)
        with (
            patch(
                "zeta_engine.evaluation.runner.requests.post",
                side_effect=[gateway_failure, success],
            ) as post,
            patch("zeta_engine.evaluation.runner.time.sleep") as sleep,
        ):
            response = _post_with_retry(
                "http://example.test/rag/login",
                data={"idx": "student", "passwd": ""},
                timeout=15,
            )

        self.assertIs(response, success)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_runs_rag_evaluation_and_submits_answers(self) -> None:
        result = [{
            "title": "标题",
            "url": "https://example.test",
            "snippet": "材料",
        }]
        calls = []

        def answer(document_db, index_db, query, **kwargs):
            return {"answer": f"回答 {query}", "results": result}

        with (
            patch(
                "zeta_engine.evaluation.runner.warmup_dense",
                side_effect=lambda *_args, **_kwargs: calls.append("bge"),
            ) as warmup,
            patch(
                "zeta_engine.evaluation.runner.input_idx",
                side_effect=lambda: calls.append("idx") or "student",
            ),
            patch("zeta_engine.evaluation.runner.input_passwd", return_value=""),
            patch(
                "zeta_engine.evaluation.runner.rag_login",
                side_effect=lambda *_args: calls.append("login") or ["问题"],
            ),
            patch(
                "zeta_engine.evaluation.runner.answer_question",
                side_effect=answer,
            ) as answer_question,
            patch(
                "zeta_engine.evaluation.runner.send_rag_answers",
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
            dense_index=Path("data/dense"),
            reranker_model=Path("models/bge-reranker-base"),
            rerank_candidates=50,
            reranker_batch_size=16,
            device=None,
            alpha=.23,
        )
        warmup.assert_called_once_with(Path("data/dense"), device=None)
        self.assertEqual(calls, ["idx", "bge", "login"])
        self.assertEqual(send_rag_answers_mock.call_args.args[3], ["回答 问题"])
        self.assertIn("Judge 1: score=1.0, reason=正确", output.getvalue())

if __name__ == "__main__":
    unittest.main()
