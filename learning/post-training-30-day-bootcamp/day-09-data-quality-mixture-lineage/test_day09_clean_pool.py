import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("freeze_day09_clean_pool.py")
SPEC = importlib.util.spec_from_file_location("freeze_day09_clean_pool", MODULE_PATH)
assert SPEC and SPEC.loader
CLEAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEAN)


def near_candidate(candidate_id: str, sample_a: str, sample_b: str, skill: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "sample_id_a": sample_a,
        "sample_id_b": sample_b,
        "skill_a": skill,
        "matched_field": "normalized_prompt",
        "similarity_score": 95.0,
    }


def config() -> dict:
    return {
        "near_review_calibration": {
            "gate_b_decision": {
                "reviewer": "user",
                "reviewed_pairs": 1,
                "result": "20 keep_both, 0 removal, 0 uncertain",
            }
        },
        "clean_pool": {"created_at": "2026-08-05T00:00:00+08:00"},
    }


def manifest_row(sample_id: str) -> dict:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    return {
        "sample_id": sample_id,
        "source": "source",
        "license": "MIT",
        "revision": "revision",
        "parent_id": "parent",
        "split": "train",
        "skill": "general",
        "subskill": "test",
        "language": "en",
        "content_hash": CLEAN.object_sha256(messages),
        "transform_chain": ["adapter"],
        "raw_token_count": 2,
        "input_token_count": 4,
        "supervised_token_count": 1,
        "messages": messages,
    }


class Day09CleanPoolTest(unittest.TestCase):
    def test_candidate_hash_lookup_is_independent_of_work_dir_name(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            for skill in ("general", "math", "code", "finance"):
                path = work_dir / f"{skill}-candidates.jsonl"
                path.write_text(
                    json.dumps({"sample_id": f"{skill}:one"}) + "\n",
                    encoding="utf-8",
                )
            rows, hashes = CLEAN.load_candidate_rows(work_dir)
        self.assertEqual(set(hashes), {"general", "math", "code", "finance"})
        self.assertEqual(set(rows), {f"{skill}:one" for skill in hashes})

    def test_clean_pool_dependency_hash_excludes_future_stage_config(self):
        dependency_keys = (
            "assignment", "preregistered_date", "intended_use",
            "candidate_samples_per_slice", "sources", "selection", "amendments",
            "preprocessing", "identity", "token_count_definitions", "quality_audit",
            "normalization", "exact_dedup", "near_duplicate",
            "near_review_calibration", "eval_candidates", "decontamination",
            "manual_review", "quality_filter", "clean_pool",
        )
        base = {key: {"value": key} for key in dependency_keys}
        base.update({"token_budget": {"value": 1}, "mixtures": {"value": 1}})
        changed = dict(base)
        changed.update({"token_budget": {"value": 2}, "mixtures": {"value": 2}})
        self.assertEqual(
            CLEAN.object_sha256(CLEAN.clean_pool_dependency_config(base)),
            CLEAN.object_sha256(CLEAN.clean_pool_dependency_config(changed)),
        )

    def test_gate_b_distinguishes_manual_and_non_terminal_policy_decisions(self):
        candidates = [
            near_candidate("c1", "code:a", "code:b", "code"),
            near_candidate("c2", "finance:a", "finance:b", "finance"),
        ]
        review = [{"sample_id_a": "code:a", "sample_id_b": "code:b"}]
        ledger, summary = CLEAN.build_gate_b_ledger(candidates, review, [], config())
        self.assertEqual(len(ledger), 2)
        self.assertTrue(all(row["decision"] == "keep_both" for row in ledger))
        self.assertEqual(summary["manual_keep_both_pairs"], 1)
        self.assertEqual(summary["policy_keep_both_pairs"], 1)
        self.assertEqual(summary["confirmed_removed_sample_ids"], [])

    def test_gate_b_refuses_unreviewed_train_eval_candidate(self):
        with self.assertRaises(CLEAN.CleanPoolError):
            CLEAN.build_gate_b_ledger(
                [near_candidate("c1", "code:a", "code:b", "code")],
                [{"sample_id_a": "code:a", "sample_id_b": "code:b"}],
                [{"candidate_id": "contamination"}],
                config(),
            )

    def test_manifest_hash_and_pool_hash_validate(self):
        rows = [manifest_row("general:b"), manifest_row("general:a")]
        rows.sort(key=lambda row: row["sample_id"])
        header = {
            "totals": CLEAN.pool_totals(rows),
            "clean_pool_hash": CLEAN.object_sha256(rows),
        }
        header["manifest_hash"] = CLEAN.object_sha256(
            {"header": header.copy(), "records": rows}
        )
        manifest = {"header": header, "records": rows}
        step6_summary = {"dedup_accounting": {"removed_sample_ids": ["removed"]}}
        checks = CLEAN.validate_manifest(manifest, step6_summary)
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
