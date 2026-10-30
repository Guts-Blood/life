import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_day09_quality.py")
SPEC = importlib.util.spec_from_file_location("audit_day09_quality", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class Day09AuditTest(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        values = [5, 1, 4, 2, 3]
        self.assertEqual(AUDIT.nearest_rank(values, 0.50), 3)
        self.assertEqual(AUDIT.nearest_rank(values, 0.90), 5)
        self.assertEqual(AUDIT.nearest_rank(values, 1.00), 5)

    def test_bucket_boundaries_are_inclusive(self):
        buckets = [
            {"name": "short", "minimum": 1, "maximum": 64},
            {"name": "long", "minimum": 65, "maximum": None},
        ]
        self.assertEqual(AUDIT.bucket_value(1, buckets), "short")
        self.assertEqual(AUDIT.bucket_value(64, buckets), "short")
        self.assertEqual(AUDIT.bucket_value(65, buckets), "long")

    def test_refusal_match_uses_normalized_text(self):
        phrases = ["i cannot assist"]
        self.assertEqual(
            AUDIT.refusal_match("I   CANNOT\nASSIST with that.", phrases),
            "i cannot assist",
        )
        self.assertIsNone(AUDIT.refusal_match("Here is the answer.", phrases))

    def test_template_signature_normalizes_urls_and_numbers(self):
        left = AUDIT.template_signature(
            "Review https://example.com and calculate 12.5% for 2025.", 16
        )
        right = AUDIT.template_signature(
            "Review https://another.test and calculate 99% for 2030.", 16
        )
        self.assertEqual(left, right)
        self.assertIn("<url>", left)
        self.assertIn("<num>", left)

    def test_dimension_profile_keeps_units_and_evidence_separate(self):
        rows = [
            {
                "sample_id": "a",
                "group": "x",
                "raw_token_count": 10,
                "input_token_count": 12,
                "supervised_token_count": 3,
            },
            {
                "sample_id": "b",
                "group": "x",
                "raw_token_count": 20,
                "input_token_count": 22,
                "supervised_token_count": 7,
            },
            {
                "sample_id": "c",
                "group": "y",
                "raw_token_count": 70,
                "input_token_count": 76,
                "supervised_token_count": 90,
            },
        ]
        totals = AUDIT.row_totals(rows)
        profile, evidence = AUDIT.profile_dimension(
            rows, "group", lambda row: row["group"], totals
        )
        self.assertEqual(sum(row["examples"] for row in profile), 3)
        self.assertAlmostEqual(
            sum(row["example_percentage"] for row in profile), 100.0, places=5
        )
        self.assertAlmostEqual(
            sum(row["raw_token_percentage"] for row in profile), 100.0, places=5
        )
        self.assertAlmostEqual(
            sum(row["supervised_token_percentage"] for row in profile),
            100.0,
            places=5,
        )
        x_evidence = next(row for row in evidence if row["value"] == "x")
        self.assertEqual(x_evidence["sample_ids"], ["a", "b"])
        self.assertEqual(
            x_evidence["sample_ids_sha256"],
            next(row for row in profile if row["value"] == "x")[
                "sample_ids_sha256"
            ],
        )


if __name__ == "__main__":
    unittest.main()
