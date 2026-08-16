#!/usr/bin/env python3
"""Focused standard-library tests for the Day 23 RSI-v0003 campaign builder."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
if str(DAY23_DIR) not in sys.path:
    sys.path.insert(0, str(DAY23_DIR))

import build_day23_rsi_v0003 as builder  # noqa: E402
import day23_contract as contract  # noqa: E402


class Day23RSIV0003Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="day23-rsi-v0003-test-")
        cls.source_root = Path(cls.temporary.name) / "bootcamp"
        required = (
            builder.DATA_MANIFEST_REL,
            builder.TRAIN_REL,
            builder.PROCESSOR_SUMMARY_REL,
            builder.CPU_CONTRACT_REL,
            "day-23-dpo-theory-smoke/build_day23_rsi_v0003.py",
            "day-23-dpo-theory-smoke/run_day23_rsi_candidate.py",
            "day-23-dpo-theory-smoke/eval_day23_rsi_candidate.py",
            "rsi-control/charters/goal-0002-day23-dpo/validate_day23_dpo_charter.py",
        )
        for relative in required:
            source = BOOTCAMP_ROOT / relative
            target = cls.source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        # This is a test-only authority stand-in.  The production builder
        # refuses to build until the separately-owned real charter exists.
        train_rows = contract.load_jsonl(cls.source_root / builder.TRAIN_REL)
        fit_rows, search_rows, _ = builder.split_fit_search(train_rows)
        charter = {
            "schema_name": "rsi.day23_dpo_charter",
            "schema_version": 1,
            "goal_id": builder.GOAL_ID,
            "status": "test_authority_only",
            "dataset_campaign": {
                "optimizer124": {
                    "ordered_pair_ids_sha256": contract.object_sha256(
                        [row["pair_id"] for row in fit_rows]
                    )
                },
                "search30": {
                    "ordered_pair_ids_sha256": contract.object_sha256(
                        [row["pair_id"] for row in search_rows]
                    )
                },
                "full154_refit": {
                    "ordered_pair_ids_sha256": contract.object_sha256(
                        [row["pair_id"] for row in train_rows]
                    )
                },
            },
            "access_leases": {
                "scope_id": "day22-test-scope",
                "ledger_root": "access-ledger/day22-test-scope",
                "dev": {
                    "claim_path": "access-ledger/day22-test-scope/dev-claim.json",
                    "identity": {"records": 17, "file_sha256": "2" * 64},
                    "maximum_global_claims": 1,
                    "prerequisite_state": "candidate_pool_frozen",
                    "raw_content_access_before_claim": False,
                },
                "heldout": {
                    "claim_path": "access-ledger/day22-test-scope/heldout-claim.json",
                    "identity": {"records": 29, "file_sha256": "3" * 64},
                    "current_authorized_claims": 0,
                    "maximum_global_claims_after_append_only_authorization": 1,
                    "authorization": "blocked_until_guardrail_and_heldout_gate_amendment",
                    "prerequisite_state": "guardrails_passed",
                    "raw_content_access_before_claim": False,
                },
            },
        }
        charter["charter_sha256"] = contract.object_sha256(charter)
        charter_path = cls.source_root / builder.CHARTER_REL
        charter_path.parent.mkdir(parents=True, exist_ok=True)
        charter_path.write_bytes(builder._json_bytes(charter))

        # Deliberately do not copy dev, heldout, all-pairs, split IDs, or the
        # 200-row processor audit.  A successful build proves they are not read.
        cls.payloads, cls.campaign = builder.build_campaign_bundle(cls.source_root)
        builder.apply_bundle(cls.payloads, output_root=cls.source_root, mode="build")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _reseal(self, value: dict) -> dict:
        value.pop("campaign_sha256", None)
        value["campaign_sha256"] = contract.object_sha256(value)
        return value

    def test_build_uses_only_train_and_public_metadata(self) -> None:
        self.assertFalse((self.source_root / builder.DEV_REL).exists())
        self.assertFalse(
            (
                self.source_root
                / "artifacts/data/day23-qwen35-coding-dpo-heldout.jsonl"
            ).exists()
        )
        self.assertFalse(
            (
                self.source_root
                / "artifacts/eval/day23-qwen35-coding-dpo-processor-audit.jsonl"
            ).exists()
        )
        self.assertEqual(self.campaign["datasets"]["dev"]["records"], 17)
        self.assertNotIn("path", self.campaign["datasets"]["heldout"])

    def test_deterministic_unstratified_partition(self) -> None:
        train = contract.load_jsonl(self.source_root / builder.TRAIN_REL)
        fit = [json.loads(line) for line in self.payloads[builder.FIT_REL].splitlines()]
        search = [json.loads(line) for line in self.payloads[builder.SEARCH_REL].splitlines()]
        self.assertEqual((len(fit), len(search)), (124, 30))
        old = set(builder.OLD_MECHANISM_PAIR_IDS)
        self.assertTrue(old <= {row["pair_id"] for row in fit})
        self.assertTrue(old.isdisjoint({row["pair_id"] for row in search}))
        eligible = [row["pair_id"] for row in train if row["pair_id"] not in old]
        expected = set(
            sorted(
                eligible,
                key=lambda pair_id: (
                    hashlib.sha256(
                        f"{builder.CAMPAIGN_ID}|{pair_id}".encode("utf-8")
                    ).hexdigest(),
                    pair_id,
                ),
            )[:30]
        )
        self.assertEqual({row["pair_id"] for row in search}, expected)
        partition = self.campaign["data_partition"]
        self.assertEqual(
            partition["authority"],
            "frozen_train_pair_ids_only_no_processor_audit_rows",
        )

    def test_campaign_hashes_and_budgets_are_strict(self) -> None:
        builder.validate_campaign(self.campaign, payloads=self.payloads)
        self.assertEqual(
            contract.object_sha256(self.campaign, "campaign_sha256"),
            self.campaign["campaign_sha256"],
        )
        self.assertEqual(self.campaign["budgets"]["maximum_optimizer_steps"], 90)
        self.assertEqual(self.campaign["leases"], {"search": 4, "dev": 1, "heldout": 0})
        self.assertEqual(
            self.campaign["fixed_recipe"]["runtime"]["min_free_memory_fraction"],
            0.30,
        )
        self.assertEqual(
            set(self.campaign["implementation_sources"]),
            {
                "builder_and_binder",
                "training_runner",
                "candidate_evaluator",
                "charter_validator",
            },
        )

    def test_configs_freeze_single_lever_b16_and_no_dev(self) -> None:
        search_lrs = set()
        for run_id, spec in self.campaign["run_specs"].items():
            config = json.loads(self.payloads[spec["executable_config"]["path"]])
            self.assertFalse(set(config) & builder.FORBIDDEN_EXECUTABLE_KEYS)
            self.assertEqual(config["per_device_train_batch_size"], 16)
            self.assertEqual(config["gradient_accumulation_steps"], 1)
            self.assertEqual(spec["runtime"]["configured_nominal_global_train_batch_size"], 32)
            self.assertEqual(config["learning_rate"], spec["learning_rate"])
            self.assertEqual(config["max_steps"], spec["max_steps"])
            if spec["role"] == "search_train":
                search_lrs.add(config["learning_rate"])
                self.assertTrue(spec["authorized"])
                self.assertEqual(spec["dataset_key"], "fit")
                self.assertEqual(spec["checkpoint_steps"], [15, 30])
            else:
                self.assertFalse(spec["authorized"])
                self.assertEqual(spec["dataset_key"], "full_train")
        self.assertEqual(search_lrs, {1e-6, 5e-6})

    def test_resealed_b24_or_refit_authorization_is_rejected(self) -> None:
        b24 = copy.deepcopy(self.campaign)
        b24["fixed_recipe"]["topology"]["per_device_train_batch_size"] = 24
        with self.assertRaisesRegex(builder.Day23RSICampaignError, "topology"):
            builder.validate_campaign(self._reseal(b24))

        refit = copy.deepcopy(self.campaign)
        refit["run_specs"]["refit_lr_1e_6_step_15"]["authorized"] = True
        with self.assertRaisesRegex(builder.Day23RSICampaignError, "authorization"):
            builder.validate_campaign(self._reseal(refit))

    def test_resealed_heldout_path_or_lease_is_rejected(self) -> None:
        leaked = copy.deepcopy(self.campaign)
        leaked["datasets"]["heldout"]["path"] = "/forbidden/heldout.jsonl"
        leaked["leases"]["heldout"] = 1
        with self.assertRaisesRegex(builder.Day23RSICampaignError, "heldout"):
            builder.validate_campaign(self._reseal(leaked))

    def test_selection_is_locked_and_lexicographic(self) -> None:
        rows = []
        for lr in builder.LEARNING_RATES:
            for step in builder.CHECKPOINT_STEPS:
                rows.append(
                    {
                        "candidate_id": builder._candidate_id(lr, step),
                        "learning_rate": lr,
                        "checkpoint_step": step,
                        "finite_records": 30,
                        "positive_margin_count": 20,
                        "mean_margin": 0.1,
                        "length_matched_mean_margin": 0.05,
                        "runtime_reload_freeze_integrity": "pass",
                    }
                )
        self.assertEqual(
            builder.select_search_candidate(rows),
            "candidate_lr_1e_6_step_15",
        )
        rows[3]["positive_margin_count"] = 21
        self.assertEqual(builder.select_search_candidate(rows), rows[3]["candidate_id"])
        for row in rows:
            row["positive_margin_count"] = 19
        self.assertIsNone(builder.select_search_candidate(rows))
        with self.assertRaisesRegex(builder.Day23RSICampaignError, "four locked"):
            builder.select_search_candidate(rows[:3])

    def test_build_check_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day23-rsi-output-") as temporary:
            output = Path(temporary)
            builder.apply_bundle(self.payloads, output_root=output, mode="build")
            builder.apply_bundle(self.payloads, output_root=output, mode="check")
            target = output / builder.CONFIG_DIR_REL / builder.CONFIG_FILENAMES["search_lr_1e_6"]
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaisesRegex(builder.Day23RSICampaignError, "drifted"):
                builder.apply_bundle(self.payloads, output_root=output, mode="check")

    def test_pure_gpu_binding_replaces_all_placeholders_without_dev_open(self) -> None:
        run_root = Path(self.temporary.name) / "gpu-run"
        run_root.mkdir()
        model_path = Path(self.temporary.name) / "merged-s1"
        model_path.mkdir()
        python_path = Path(self.temporary.name) / "python"
        python_path.write_bytes(b"test-python")
        checkout = Path(self.temporary.name) / "ms-swift"
        checkout.mkdir()
        source_binding = {
            "remote_parent": {"merged_export": {"path": str(model_path)}}
        }
        runtime_parse = {
            "status": "pass",
            "python_executable": {"path": str(python_path)},
            "ms_swift_checkout": str(checkout),
        }
        payloads, gpu = builder.bind_campaign(
            self.campaign,
            bootcamp_root=self.source_root,
            run_root=run_root,
            model_path=model_path,
            python_path=python_path,
            ms_swift_checkout=checkout,
            source_binding=source_binding,
            source_binding_identity={
                "path": "/source/gpu-execution-binding.json",
                "file_sha256": "0" * 64,
                "binding_sha256": "1" * 64,
            },
            runtime_parse=runtime_parse,
        )
        self.assertEqual(gpu["status"], "gpu_execution_bound_optimizer_pending")
        self.assertFalse(builder._contains_placeholder(gpu))
        self.assertEqual(set(gpu["producers"]), {"binder", "runner", "evaluator"})
        self.assertTrue(Path(gpu["access_ledger"]["dev_claim"]).is_absolute())
        self.assertNotIn("claim_path", gpu["access_ledger"]["heldout_lease"])
        for run_id, spec in gpu["run_specs"].items():
            self.assertTrue(Path(spec["executable_config"]["path"]).is_absolute())
            config = json.loads(payloads[f"configs/{builder.CONFIG_FILENAMES[run_id]}"])
            self.assertTrue(Path(config["dataset"][0]).is_absolute())
            self.assertTrue(Path(config["model"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
