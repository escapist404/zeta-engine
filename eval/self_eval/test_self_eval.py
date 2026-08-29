import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("self_eval.py")
SPEC = importlib.util.spec_from_file_location("self_eval", PATH)
assert SPEC and SPEC.loader
self_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(self_eval)


class SelfEvalTest(unittest.TestCase):
    def test_title_question_mutation_and_multiple_answers(self) -> None:
        title = (
            "中国人民大学物理学院2027年推免生接收工作报名通知"
            "-中国人民大学物理学院"
        )
        self.assertEqual(
            self_eval.canonical_title(title),
            "中国人民大学物理学院2027年推免生接收工作报名通知",
        )
        question, tag = self_eval.natural_question(title)
        self.assertEqual(tag, "admission")
        variants = self_eval.challenge_mutations(question)
        self.assertTrue(any("人大" in query for _kind, query in variants))

        rows = [
            {"rank": 2, "latency_seconds": .1},
            {"rank": None, "latency_seconds": .2},
        ]
        metrics = self_eval._metrics(rows)
        self.assertEqual(metrics["recall@5"], .5)
        self.assertEqual(metrics["mrr@20"], .25)


if __name__ == "__main__":
    unittest.main()
