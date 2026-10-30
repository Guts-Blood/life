import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("find_day09_eval_overlap.py")
SPEC = importlib.util.spec_from_file_location("find_day09_eval_overlap", MODULE_PATH)
assert SPEC and SPEC.loader
OVERLAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERLAP)


def train_row(prompt: str, answer: str, sample_id: str = "train:1") -> dict:
    return {
        "sample_id": sample_id,
        "source": "train/source",
        "skill": "math",
        "content_hash": f"train-hash-{sample_id}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def eval_row(prompt: str, reference: str, eval_id: str = "eval:1") -> dict:
    return {
        "eval_sample_id": eval_id,
        "source": "eval/source",
        "skill": "math",
        "content_hash": f"eval-hash-{eval_id}",
        "prompt": prompt,
        "reference": reference,
    }


def boundary(train_field: str, eval_field: str, minimum: int = 40) -> dict:
    return {
        "name": f"test_{train_field}_vs_{eval_field}",
        "matched_field": f"{train_field}_{eval_field}",
        "train_field": train_field,
        "eval_field": eval_field,
        "near_threshold": 90.0,
        "near_minimum_characters": minimum,
    }


def generate(train, evaluation, boundary_config):
    return OVERLAP.generate_boundary_candidates(
        train,
        evaluation,
        version="test-v1",
        boundary_config=boundary_config,
        matcher_config_sha256="config-hash",
        batch_size=2,
        workers=1,
        score_decimals=6,
    )


class Day09EvalOverlapTest(unittest.TestCase):
    def test_short_exact_answer_is_found_but_not_near_scored(self):
        candidates, coverage = generate(
            [train_row("Different train prompt", "42")],
            [eval_row("Different eval prompt", "42")],
            boundary("answer", "reference", minimum=40),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["match_type"], "exact")
        self.assertEqual(candidates[0]["similarity_score"], 100.0)
        self.assertIsNone(candidates[0]["decision"])
        self.assertEqual(coverage["near_cross_pairs_scored"], 0)

    def test_near_candidate_excludes_non_identical_exact_pair(self):
        base = "Explain the multi-step reasoning for this financial question. " * 2
        candidates, _ = generate(
            [train_row(base + "A", "train answer")],
            [eval_row(base + "B", "eval reference")],
            boundary("prompt", "prompt", minimum=40),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["match_type"], "near")
        self.assertGreaterEqual(candidates[0]["similarity_score"], 90.0)
        self.assertLess(candidates[0]["similarity_score"], 100.0)

    def test_normalized_complete_record_exact_match_is_separate(self):
        candidates, _ = generate(
            [train_row("  SAME Question\n", "Same   Answer")],
            [eval_row("same question", "same answer")],
            boundary("complete_record", "complete_record", minimum=80),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["match_type"], "exact")
        self.assertEqual(candidates[0]["matched_field"], "complete_record_complete_record")


if __name__ == "__main__":
    unittest.main()
