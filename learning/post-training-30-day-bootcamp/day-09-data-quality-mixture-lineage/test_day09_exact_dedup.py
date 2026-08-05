import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("dedup_day09_exact.py")
SPEC = importlib.util.spec_from_file_location("dedup_day09_exact", MODULE_PATH)
assert SPEC and SPEC.loader
DEDUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEDUP)


def fixture_row(
    sample_id: str,
    skill: str,
    prompt: str,
    answer: str,
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


class Day09ExactDedupTest(unittest.TestCase):
    def test_normalization_is_nfkc_lowercase_and_whitespace_stable(self):
        self.assertEqual(DEDUP.normalize_text("  Ａ,\n  B!  "), "a, b!")

    def test_manual_accept_wins_before_slice_priority(self):
        finance = fixture_row("finance:z", "finance", "Same Q", "Same A")
        general = fixture_row("general:a", "general", "same  q", "same a")
        evidence, removed, _ = DEDUP.build_exact_groups(
            [finance, general],
            match_fields=[
                "normalized_prompt",
                "normalized_answer",
                "normalized_complete_record",
            ],
            terminal_field="normalized_complete_record",
            manual_accepts={"general:a"},
            slice_priority=["finance", "general", "math", "code"],
        )
        self.assertEqual(removed, {"finance:z": "general:a"})
        complete = [
            item
            for item in evidence
            if item["matched_field"] == "normalized_complete_record"
        ]
        self.assertEqual(complete[0]["survivor_sample_id"], "general:a")

    def test_slice_priority_wins_without_manual_accept(self):
        finance = fixture_row("finance:z", "finance", "Same Q", "Same A")
        general = fixture_row("general:a", "general", "same q", "same a")
        _, removed, _ = DEDUP.build_exact_groups(
            [general, finance],
            match_fields=["normalized_complete_record"],
            terminal_field="normalized_complete_record",
            manual_accepts=set(),
            slice_priority=["finance", "general", "math", "code"],
        )
        self.assertEqual(removed, {"general:a": "finance:z"})

    def test_prompt_only_match_is_evidence_not_removal(self):
        first = fixture_row("general:a", "general", "Same Q", "Answer one")
        second = fixture_row("math:b", "math", "same q", "Answer two")
        evidence, removed, memberships = DEDUP.build_exact_groups(
            [first, second],
            match_fields=[
                "normalized_prompt",
                "normalized_answer",
                "normalized_complete_record",
            ],
            terminal_field="normalized_complete_record",
            manual_accepts=set(),
            slice_priority=["finance", "general", "math", "code"],
        )
        self.assertEqual(removed, {})
        self.assertEqual(
            [item["matched_field"] for item in evidence], ["normalized_prompt"]
        )
        decisions, kept = DEDUP.build_decisions(
            [first, second], removed, memberships, set()
        )
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(item["decision"] == "keep" for item in decisions))


if __name__ == "__main__":
    unittest.main()
