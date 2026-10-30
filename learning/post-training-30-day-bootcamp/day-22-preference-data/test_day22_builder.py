from __future__ import annotations

import copy
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_day22_mbpp as builder


SEED_MANIFEST = (
    BOOTCAMP_ROOT / "artifacts/data/day22-qwen35-mbpp-seed-manifest.json"
)


class Day22BuilderTest(unittest.TestCase):
    def test_exact_family_partition(self) -> None:
        split_by_task = builder._assign_exact_splits(range(1, 331))
        self.assertEqual(Counter(split_by_task.values()), builder.SPLIT_COUNTS)
        self.assertEqual(len(builder._smoke_ids(split_by_task)), 80)
        smoke = builder._smoke_ids(split_by_task)
        self.assertEqual(
            Counter(split_by_task[task_id] for task_id in smoke),
            builder.SMOKE_COUNTS,
        )

    def test_mutants_are_distinct_valid_continuations(self) -> None:
        chosen_text = "    if n < 2:\n        return n\n    return n + 1"
        chosen = builder.canonicalize_code_continuation(chosen_text, "def f(n):")
        record = {
            "task_family_id": "mbpp:task:9999",
            "code_prefix": "def f(n):",
            "canonical_chosen": {
                "text": chosen.canonical,
                "sha256": chosen.canonical_sha256,
                "ast_sha256": chosen.ast_sha256,
            },
        }
        mutants = builder.generate_mutants(record, limit=8)
        self.assertGreaterEqual(len(mutants), 4)
        self.assertEqual(len({row["sha256"] for row in mutants}), len(mutants))
        for mutant in mutants:
            validated = builder.validate_raw_code_continuation(
                mutant["text"], record["code_prefix"]
            )
            self.assertEqual(validated.canonical_sha256, mutant["sha256"])
            self.assertNotEqual(validated.ast_sha256, chosen.ast_sha256)

    def test_frozen_seed_manifest_invariants(self) -> None:
        if not SEED_MANIFEST.is_file():
            self.skipTest("generated seed manifest is not present")
        manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest.pop("manifest_sha256")
        self.assertEqual(expected, builder.object_sha256(manifest))
        records = manifest["records"]
        self.assertEqual(len(records), 330)
        self.assertEqual(Counter(row["split"] for row in records), builder.SPLIT_COUNTS)
        self.assertEqual(
            Counter(row["split"] for row in records if row["smoke_selected"]),
            builder.SMOKE_COUNTS,
        )
        self.assertEqual(len({row["task_id"] for row in records}), 330)
        self.assertTrue(all(row["source"]["split"] == "train" for row in records))
        self.assertTrue(
            all(
                row["task_id"]
                not in manifest["header"]["day20_inputs"]["probe_overlap_task_ids"]
                for row in records
            )
        )

    def test_candidate_builder_rejects_tampered_seed_hash(self) -> None:
        if not SEED_MANIFEST.is_file():
            self.skipTest("generated seed manifest is not present")
        manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(manifest)
        tampered["records"][0]["task_id"] = -1
        with self.assertRaisesRegex(builder.Day22BuildError, "self-hash"):
            builder.build_candidate_rows(tampered, scope="smoke", mutation_limit=2)

    def test_replay_rows_select_one_mutant_and_normalize_identity(self) -> None:
        candidate = {
            "task_family_id": "mbpp:task:1",
            "task_id": 1,
            "split": "train",
            "family_keys": {
                "problem": "mbpp:task:1",
                "prompt": "sha256:" + "1" * 64,
                "test": "sha256:" + "a" * 64,
                "source": "mbpp:task:1",
            },
            "source": {"dataset": "google-research-datasets/mbpp"},
            "problem": {"text": "Add one.", "sha256": "2" * 64},
            "prompt": {"text": "Complete f.", "sha256": "3" * 64},
            "code_prefix": "def f(n):",
            "entry_point": "f",
            "tests": {
                "test_list": ["assert f(1) == 2"],
                "challenge_test_list": [],
                "test_setup_code": "",
                "sha256": "a" * 64,
            },
            "chosen": {
                "candidate_id": "chosen",
                "origin": "mbpp_canonical_solution",
                "text": "    return n + 1",
                "sha256": "b" * 64,
                "ast_sha256": "d" * 64,
            },
            "mutants": [
                {
                    "candidate_id": "mutant",
                    "origin": "deterministic_ast_mutation",
                    "text": "    return n - 1",
                    "sha256": "c" * 64,
                    "ast_sha256": "e" * 64,
                    "mutation": {"rule": "Add_to_Sub"},
                }
            ],
        }
        rows = builder.build_replay_rows([candidate])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "1")
        self.assertEqual(rows[0]["chosen"]["origin"], "mbpp_canonical")
        self.assertEqual(rows[0]["rejected"]["origin"], "deterministic_mutation")
        self.assertEqual(rows[0]["prompt"]["text"], "Complete f.")
        self.assertEqual(rows[0]["entry_point"], "f")


if __name__ == "__main__":
    unittest.main()
