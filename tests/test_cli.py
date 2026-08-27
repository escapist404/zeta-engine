import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import ANY, patch

from zeta_engine.cli import build_parser
from zeta_engine.storage import Storage


class CliTest(unittest.TestCase):
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
            ])
            self.assertEqual(search_args.ranking, "simple")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(search_args.handler(search_args), 0)
            self.assertIn("https://info.ruc.edu.cn/example", output.getvalue())

            tf_idf_args = parser.parse_args([
                "search", "中国人民大学",
                "--document-db", str(document_db),
                "--index-db", str(index_db),
                "--ranking", "tf-idf",
            ])
            with patch(
                "zeta_engine.cli.search_tf_idf",
                return_value=[],
            ) as search_tf_idf:
                self.assertEqual(tf_idf_args.handler(tf_idf_args), 0)
            search_tf_idf.assert_called_once_with(ANY, "中国人民大学")


if __name__ == "__main__":
    unittest.main()
