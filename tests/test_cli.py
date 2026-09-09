import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from zeta_engine.interfaces.cli import build_parser


class CliTest(unittest.TestCase):
    def test_runs_rag_from_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_db = root / "documents.db"
            index_db = root / "index.db"
            dense_index = root / "dense"
            reranker_model = root / "reranker"
            document_db.touch()
            index_db.touch()
            dense_index.mkdir()
            reranker_model.mkdir()
            (dense_index / "metadata.json").touch()
            result = {
                "title": "资助政策",
                "url": "https://example.test/policy",
                "snippet": "学生可以申请。",
            }

            args = build_parser().parse_args([
                "rag", "如何申请资助？",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--dense-index", str(dense_index),
                "--reranker-model", str(reranker_model),
                "--top-k", "3",
                "--max-cycles", "6",
                "--debug",
                "--device", "mps",
            ])
            with (
                patch(
                    "zeta_engine.interfaces.cli.answer_question",
                    return_value={
                        "answer": "可以申请。[文档1]",
                        "results": [result],
                        "trace": [{"cycle": 1, "action": "answer"}],
                    },
                ) as answer_question,
                redirect_stdout(output := io.StringIO()),
            ):
                self.assertEqual(args.handler(args), 0)

            self.assertIn("可以申请。[文档1]", output.getvalue())
            self.assertIn("https://example.test/policy", output.getvalue())
            self.assertIn("调试信息", output.getvalue())
            self.assertIn("第 1 轮 · 提交候选答案", output.getvalue())
            self.assertEqual(
                answer_question.call_args.args,
                (document_db, index_db, "如何申请资助？"),
            )
            answer_question.assert_called_once_with(
                document_db,
                index_db,
                "如何申请资助？",
                top_k=3,
                dense_index=dense_index,
                device="mps",
                alpha=.23,
                reranker_model=reranker_model,
                rerank_candidates=50,
                reranker_batch_size=16,
                max_cycles=6,
                debug=True,
            )

if __name__ == "__main__":
    unittest.main()
