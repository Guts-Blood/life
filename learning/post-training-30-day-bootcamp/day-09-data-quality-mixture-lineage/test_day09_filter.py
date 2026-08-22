import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("filter_day09_quality.py")
SPEC = importlib.util.spec_from_file_location("filter_day09_quality", MODULE_PATH)
assert SPEC and SPEC.loader
FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILTER)


def fixture_row(index: int, *, skill: str = "general") -> dict:
    messages = [
        {"role": "user", "content": f"question {index}"},
        {"role": "assistant", "content": f"answer {index}"},
    ]
    parent_id = f"r{index}"
    return {
        "sample_id": f"{skill}:{parent_id}",
        "source": "example/source",
        "revision": "a" * 40,
        "parent_id": parent_id,
        "split": "train",
        "license": "test",
        "content_hash": FILTER.object_sha256(messages),
        "transform_chain": ["fixture_v1"],
        "messages": messages,
        "skill": skill,
        "subskill": "fixture",
        "language": "en",
        "raw_token_count": 10 + index,
        "input_token_count": 20 + index,
        "supervised_token_count": index + 1,
        "truncated": False,
    }


class Day09FilterTest(unittest.TestCase):
    def test_review_selection_is_stable_and_risk_stratified(self):
        rows = [fixture_row(index) for index in range(12)]
        hints = {
            row["sample_id"]: {
                "length_bucket": "1-256",
                "refusal_status": (
                    "refusal_phrase_match"
                    if row["sample_id"] == "general:r2"
                    else "no_refusal_phrase_match"
                ),
                "difficulty_proxy": "low_1_64",
                "source_provenance": "declared_synthetic",
                "template_cluster_status": (
                    "repeated_prefix_cluster"
                    if row["sample_id"] == "general:r3"
                    else "not_repeated"
                ),
            }
            for row in rows
        }
        first = FILTER.choose_review_rows(
            rows, hints, version="gate-a-test", per_source=6
        )
        second = FILTER.choose_review_rows(
            list(reversed(rows)), hints, version="gate-a-test", per_source=6
        )
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        selected = {row["sample_id"]: row for row in first}
        self.assertIn("general:r2", selected)
        self.assertIn("general:r3", selected)
        self.assertIn("general:r0", selected)
        self.assertIn("general:r11", selected)
        self.assertIn("refusal_hint", selected["general:r2"]["selection_reasons"])

    def test_guardrails_accept_valid_row_and_find_independent_failures(self):
        row = fixture_row(1)
        self.assertEqual(FILTER.guardrail_reasons(row, 2048), [])
        broken = {**row, "sample_id": "general:wrong", "content_hash": "bad"}
        self.assertEqual(
            FILTER.guardrail_reasons(broken, 2048),
            ["sample_id_parent_mismatch", "content_hash_mismatch"],
        )

    def test_manual_reject_has_priority_but_all_matches_are_recorded(self):
        row = fixture_row(1)
        row["content_hash"] = "bad"
        review = {
            "sample_id": row["sample_id"],
            "review_result": "reject",
            "reason_code": "incorrect_reasoning",
            "review_note": "confirmed fixture error",
        }
        decisions, kept = FILTER.build_decisions([row], [], [review], 2048)
        self.assertEqual(kept, [])
        self.assertEqual(
            decisions[0]["terminal_filter"], "confirmed_manual_quality_v1"
        )
        self.assertEqual(
            decisions[0]["matched_reasons"],
            [
                "confirmed_manual_quality_v1:incorrect_reasoning",
                "deterministic_guardrails_v1:content_hash_mismatch",
            ],
        )

    def test_record_accept_all_is_auditable_and_refuses_conflicting_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            review_path = Path(temp_dir) / "review.jsonl"
            rows = [
                {"sample_id": "general:r1", "review_result": None},
                {"sample_id": "general:r2", "review_result": "accept"},
            ]
            review_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            FILTER.record_gate_a_accept_all(
                review_path,
                reviewer="user",
                reviewed_at="2026-08-05",
                review_note="Manual review confirmed acceptable quality.",
            )
            recorded = FILTER.load_jsonl(review_path)
            self.assertEqual(recorded[0]["review_result"], "accept")
            self.assertEqual(recorded[0]["reviewer"], "user")
            self.assertEqual(recorded[0]["reviewed_at"], "2026-08-05")
            self.assertEqual(recorded[0]["reason_code"], None)
            self.assertEqual(recorded[1], rows[1])

            recorded[1]["review_result"] = "reject"
            review_path.write_text(
                "".join(json.dumps(row) + "\n" for row in recorded),
                encoding="utf-8",
            )
            with self.assertRaises(FILTER.FilterError):
                FILTER.record_gate_a_accept_all(
                    review_path,
                    reviewer="user",
                    reviewed_at="2026-08-05",
                    review_note="Manual review confirmed acceptable quality.",
                )


if __name__ == "__main__":
    unittest.main()
