import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("find_day09_near_duplicates.py")
SPEC = importlib.util.spec_from_file_location("find_day09_near_duplicates", MODULE_PATH)
assert SPEC and SPEC.loader
NEAR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NEAR)


def fixture_row(
    sample_id: str,
    prompt: str,
    answer: str,
    *,
    skill: str = "general",
) -> dict:
    return {
        "sample_id": sample_id,
        "source": f"source/{skill}",
        "skill": skill,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "raw_token_count": 10,
        "input_token_count": 12,
        "supervised_token_count": 4,
    }


def config() -> dict:
    return {
        "candidate_version": "test-v1",
        "matcher": "rapidfuzz.fuzz.ratio",
        "version": NEAR.rapidfuzz.__version__,
        "threshold": 90.0,
        "fields": [
            "normalized_prompt",
            "normalized_answer",
            "normalized_complete_record",
        ],
        "minimum_characters": 20,
        "batch_size": 2,
        "workers": 1,
        "score_decimals": 6,
        "policy": "candidate generation only; no automatic deletion",
    }


class Day09NearDuplicateTest(unittest.TestCase):
    def test_prompt_match_is_separate_and_does_not_create_decision(self):
        shared_prompt = "Explain the following financial result in detail."
        rows = [
            fixture_row("general:a", shared_prompt, "First unrelated answer."),
            fixture_row("math:b", shared_prompt, "Second different response.", skill="math"),
        ]
        candidates, coverage = NEAR.generate_candidates(rows, config())
        self.assertEqual(
            [item["matched_field"] for item in candidates],
            ["normalized_prompt"],
        )
        self.assertEqual(candidates[0]["similarity_score"], 100.0)
        self.assertEqual(candidates[0]["candidate_scope"], "cross_source")
        self.assertIsNone(candidates[0]["review_result"])
        self.assertEqual(coverage["normalized_prompt"]["unordered_pairs_scored"], 1)

    def test_short_values_are_excluded_before_pair_scoring(self):
        rows = [
            fixture_row("general:a", "short", "short"),
            fixture_row("general:b", "short", "short"),
        ]
        candidates, coverage = NEAR.generate_candidates(rows, config())
        self.assertEqual(candidates, [])
        self.assertEqual(coverage["normalized_prompt"]["eligible_examples"], 0)
        self.assertEqual(coverage["normalized_answer"]["eligible_examples"], 0)

    def test_generation_is_deterministic_for_reversed_input(self):
        base = "A" * 100
        rows = [
            fixture_row("general:a", base, base),
            fixture_row("general:b", "A" * 99 + "B", "A" * 99 + "B"),
            fixture_row("general:c", "Z" * 100, "Z" * 100),
        ]
        first, _ = NEAR.generate_candidates(rows, config())
        second, _ = NEAR.generate_candidates(list(reversed(rows)), config())
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(item["sample_id_a"] < item["sample_id_b"] for item in first))


if __name__ == "__main__":
    unittest.main()
