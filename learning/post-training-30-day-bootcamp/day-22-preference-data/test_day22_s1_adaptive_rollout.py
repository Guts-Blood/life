#!/usr/bin/env python3
"""CPU contract tests for the Day 22 adaptive S1 rollout."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import rollout_day22_s1 as base
import rollout_day22_s1_adaptive as adaptive


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Day22AdaptiveS1RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed_path = ARTIFACTS / "data/day22-qwen35-mbpp-seed-manifest.json"
        cls.selection_path = ARTIFACTS / "data/day22-qwen35-formal-s1-selections.jsonl"
        cls.summary_path = ARTIFACTS / "data/day22-qwen35-formal-s1-selection-summary.json"
        cls.parent_path = ARTIFACTS / "configs/day22-qwen35-formal-s1-rollout-contract.json"
        cls.seed = load_json(cls.seed_path)
        cls.selections = load_jsonl(cls.selection_path)
        cls.summary = load_json(cls.summary_path)
        cls.parent = load_json(cls.parent_path)

    def test_targets_exactly_cover_all_189_execution_quarantines(self) -> None:
        targets = adaptive.target_rows(self.selections, self.summary, self.seed)
        self.assertEqual(len(targets), 189)
        self.assertEqual(len({row["family_id"] for row in targets}), 189)
        self.assertEqual(
            Counter(row["first_round_reason"] for row in targets),
            Counter(adaptive.TARGET_REASON_COUNTS),
        )
        self.assertEqual({row["profile"] for row in targets}, {"supplemental_t08"})

    def test_contract_freezes_k12_new_seeds_and_balanced_family_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = adaptive.build_contract(
                parent_contract_path=self.parent_path,
                parent_contract=self.parent,
                selections_path=self.selection_path,
                selections=self.selections,
                summary_path=self.summary_path,
                summary=self.summary,
                seed_path=self.seed_path,
                seed_manifest=self.seed,
                output_root=Path(temporary) / "adaptive-output",
                implementation_sha256="a" * 64,
            )
        specs = adaptive.expected_specs(contract, self.seed)
        self.assertEqual(len(specs), 2268)
        self.assertEqual(len({row["candidate_id"] for row in specs}), 2268)
        self.assertEqual(len({row["seed"] for row in specs}), 2268)
        self.assertEqual(Counter(row["shard_id"] for row in specs), {0: 1140, 1: 1128})
        per_family: dict[str, list[dict]] = {}
        for spec in specs:
            per_family.setdefault(spec["record"]["task_family_id"], []).append(spec)
        self.assertEqual(len(per_family), 189)
        self.assertTrue(all(len(rows) == 12 for rows in per_family.values()))
        self.assertTrue(
            all(
                {row["sample_index"] for row in rows} == set(range(6, 18))
                and len({row["shard_id"] for row in rows}) == 1
                for rows in per_family.values()
            )
        )
        self.assertEqual(
            contract["generation_profiles"]["supplemental_t08"]["temperature"],
            0.8,
        )
        self.assertEqual(
            contract["generation_profiles"]["supplemental_t08"]["top_p"],
            0.95,
        )
        self.assertNotEqual(
            adaptive.candidate_seed("mbpp:task:1", "supplemental_t08", 0),
            base.candidate_seed("mbpp:task:1", 0),
        )

    def test_candidate_row_binds_adaptive_contract_profile_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = adaptive.build_contract(
                parent_contract_path=self.parent_path,
                parent_contract=self.parent,
                selections_path=self.selection_path,
                selections=self.selections,
                summary_path=self.summary_path,
                summary=self.summary,
                seed_path=self.seed_path,
                seed_manifest=self.seed,
                output_root=Path(temporary) / "adaptive-output",
                implementation_sha256="b" * 64,
            )
            spec = adaptive.expected_specs(contract, self.seed)[0]
            view = adaptive.contract_view(contract, spec["profile"])
            row = base.build_candidate_row(
                contract=view,
                spec=spec,
                message_content="    return 1",
                prompt_token_ids=[1, 2],
                generated_token_ids=[3],
                generated_text="    return 1",
                finish_reason="stop",
                elapsed_seconds=0.1,
                response_adapter={},
                format_contract={
                    "valid": True,
                    "execution_eligible": True,
                    "validator": "validate_raw_code_continuation",
                    "evidence": {},
                    "error": None,
                },
            )
            adaptive.validate_candidate_row(row, contract=contract, spec=spec)
            tampered = copy.deepcopy(row)
            tampered["generator"]["generation"]["temperature"] = 1.2
            tampered["candidate_sha256"] = base.object_sha256(
                {key: value for key, value in tampered.items() if key != "candidate_sha256"}
            )
            with self.assertRaisesRegex(adaptive.Day22AdaptiveRolloutError, "generation profile"):
                adaptive.validate_candidate_row(tampered, contract=contract, spec=spec)


if __name__ == "__main__":
    unittest.main()
