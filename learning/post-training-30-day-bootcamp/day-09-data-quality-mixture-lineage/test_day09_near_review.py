import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("sample_day09_near_review.py")
SPEC = importlib.util.spec_from_file_location("sample_day09_near_review", MODULE_PATH)
assert SPEC and SPEC.loader
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


def candidate(candidate_id: str, sample_a: str, sample_b: str, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "sample_id_a": sample_a,
        "sample_id_b": sample_b,
        "skill_a": "code",
        "skill_b": "code",
        "matched_field": "normalized_answer",
        "similarity_score": score,
    }


class Day09NearReviewTest(unittest.TestCase):
    def test_score_boundaries_are_explicit(self):
        stratum = {
            "skill": "code",
            "matched_field": "normalized_answer",
            "minimum_score": 95.0,
            "maximum_score_exclusive": 100.0,
        }
        self.assertTrue(REVIEW.matches_stratum(candidate("a", "a", "b", 95.0), stratum))
        self.assertTrue(REVIEW.matches_stratum(candidate("b", "a", "c", 99.9), stratum))
        self.assertFalse(REVIEW.matches_stratum(candidate("c", "a", "d", 100.0), stratum))

    def test_selection_excludes_pairs_already_used_by_earlier_stratum(self):
        candidates = [
            candidate("a", "code:a", "code:b", 100.0),
            candidate("b", "code:c", "code:d", 100.0),
        ]
        calibration = {
            "version": "test-v1",
            "total_pairs": 2,
            "strata": [
                {
                    "name": "first",
                    "skill": "code",
                    "matched_field": "normalized_answer",
                    "minimum_score": 100.0,
                    "maximum_score": 100.0,
                    "quota": 1,
                },
                {
                    "name": "second",
                    "skill": "code",
                    "matched_field": "normalized_answer",
                    "minimum_score": 100.0,
                    "maximum_score": 100.0,
                    "quota": 1,
                },
            ],
        }
        selected = REVIEW.select_candidates(candidates, calibration)
        pairs = [REVIEW.pair_key(item) for _, item in selected]
        self.assertEqual(len(pairs), len(set(pairs)))


if __name__ == "__main__":
    unittest.main()
