import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_day09_eval_candidates.py")
SPEC = importlib.util.spec_from_file_location("prepare_day09_eval_candidates", MODULE_PATH)
assert SPEC and SPEC.loader
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class Day09EvalCandidateTest(unittest.TestCase):
    def test_source_file_path_is_repository_relative_when_possible(self):
        source_file = EVAL.REPO_ROOT / "tmp" / "day09-eval-sources" / "fixture.jsonl"
        self.assertEqual(
            EVAL.relative_path(source_file),
            "tmp/day09-eval-sources/fixture.jsonl",
        )

    def test_canonical_candidate_has_stable_id_and_content_hash(self):
        source_config = {
            "source": "openai/grade-school-math",
            "revision": "abc123",
            "split": "test",
            "adapter": "gsm8k_jsonl_v1",
            "license": "MIT",
        }
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "test.jsonl"
            source_file.write_text("fixture\n", encoding="utf-8")
            first = EVAL.canonical_candidate(
                version="v1",
                skill="math",
                source_config=source_config,
                parent_id="test:0001",
                prompt="  What is 1 + 1?  ",
                reference=" 2 ",
                source_file=source_file,
            )
            second = EVAL.canonical_candidate(
                version="v1",
                skill="math",
                source_config=source_config,
                parent_id="test:0001",
                prompt="What is 1 + 1?",
                reference="2",
                source_file=source_file,
            )
        self.assertEqual(first, second)
        self.assertEqual(first["eval_sample_id"], "eval:math:gsm8k:test:0001")
        self.assertEqual(
            first["content_hash"],
            EVAL.object_sha256(
                {"prompt": "What is 1 + 1?", "reference": "2"}
            ),
        )

    def test_stable_take_is_input_order_independent(self):
        rows = [
            {"parent_id": "b", "selection_hash": "2"},
            {"parent_id": "a", "selection_hash": "1"},
            {"parent_id": "c", "selection_hash": "3"},
        ]
        self.assertEqual(
            EVAL.stable_take(rows, 2), EVAL.stable_take(list(reversed(rows)), 2)
        )
        self.assertEqual(
            [row["parent_id"] for row in EVAL.stable_take(rows, 2)], ["a", "b"]
        )

    def test_tatqa_context_orders_paragraphs(self):
        document = {
            "table": {"table": [["Year", "2024"], ["Revenue", "10"]]},
            "paragraphs": [
                {"order": 2, "text": "Second."},
                {"order": 1, "text": "First."},
            ],
        }
        context = EVAL.render_tatqa_context(document)
        self.assertIn("Year\t2024", context)
        self.assertLess(context.index("[1] First."), context.index("[2] Second."))


if __name__ == "__main__":
    unittest.main()
