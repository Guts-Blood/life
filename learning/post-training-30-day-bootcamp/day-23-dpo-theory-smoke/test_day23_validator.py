#!/usr/bin/env python3
"""Offline happy-path and tamper tests for the independent Day 23 validator."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
SCRIPT_DIR = BOOTCAMP_ROOT / "artifacts/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_day23_qwen35_dpo as validator  # noqa: E402


class Day23IndependentValidatorTests(unittest.TestCase):
    def test_real_bundle_is_cpu_ready_gpu_pending(self) -> None:
        result = validator.validate_bundle()
        self.assertEqual(result["status"], "valid_cpu_ready_gpu_pending")
        self.assertTrue(result["cpu_ready"])
        self.assertFalse(result["gpu_optimizer_ready"])
        self.assertFalse(result["formal_dpo_ready"])
        self.assertEqual(result["pairs"], 200)
        self.assertEqual(result["day22_exact_branches"], 400)

    def test_unsealed_run_contract_tamper_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_RUN_CONTRACT)
        value["claim_boundary"]["gpu_optimizer_ready"] = True
        with self.assertRaisesRegex(validator.Day23ValidationError, "self-hash"):
            validator.verify_self(value, "contract_sha256", "tampered contract")

    def test_resealed_heldout_leak_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_RUN_CONTRACT)
        heldout = value["dataset_usage"]["heldout"]["path"]
        value["intended_ms_swift_args"]["dataset"].append(heldout)
        value["contract_sha256"] = validator.object_sha256(
            value, "contract_sha256"
        )
        processor = validator.load_json(validator.DEFAULT_PROCESSOR_SUMMARY)
        processor_sha = validator.verify_self(
            processor, "summary_sha256", "processor"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-contract.json"
            path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validator.Day23ValidationError,
                "complete intended|train dataset|heldout",
            ):
                validator._validate_run_contract(
                    path,
                    validator.DEFAULT_DATA_MANIFEST,
                    validator.DEFAULT_PROCESSOR_SUMMARY,
                    processor_sha,
                )

    def test_resealed_argument_audit_weight_claim_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_ARGUMENT_AUDIT)
        value["stages"]["mechanism_5step"]["weights_loaded"] = True
        value["audit_sha256"] = validator.object_sha256(value, "audit_sha256")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "argument-audit.json"
            path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )
            run = validator.load_json(validator.DEFAULT_RUN_CONTRACT)
            run_sha = validator.verify_self(run, "contract_sha256", "run")
            with self.assertRaisesRegex(validator.Day23ValidationError, "parsed argument"):
                validator._validate_argument_audit(
                    path, validator.DEFAULT_RUN_CONTRACT, run_sha
                )

    def test_resealed_processor_branch_drift_differs_from_frozen_source(self) -> None:
        audit_path = (
            BOOTCAMP_ROOT
            / "artifacts/eval/day23-qwen35-coding-dpo-processor-audit.jsonl"
        )
        audit = copy.deepcopy(validator.load_jsonl(audit_path)[0])
        source_path = (
            BOOTCAMP_ROOT
            / "artifacts/data/day22-qwen35-formal-s1-preference-pairs.jsonl"
        )
        source = validator.load_jsonl(source_path)[0]
        audit["chosen"]["labels_sha256"] = "0" * 64
        audit["audit_sha256"] = validator.object_sha256(audit, "audit_sha256")
        validator.verify_self(audit, "audit_sha256", "resealed audit")
        self.assertNotEqual(audit["chosen"], source["processor_audit"]["chosen"])

    def test_independent_scalar_oracle_passes(self) -> None:
        validator._validate_scalar_oracle()

    def test_resealed_empty_cpu_gate_inventory_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_RUN_CONTRACT)
        value["cpu_gates"] = {}
        value["contract_sha256"] = validator.object_sha256(
            value, "contract_sha256"
        )
        processor = validator.load_json(validator.DEFAULT_PROCESSOR_SUMMARY)
        processor_sha = validator.verify_self(
            processor, "summary_sha256", "processor"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-contract.json"
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.Day23ValidationError, "CPU gate status"
            ):
                validator._validate_run_contract(
                    path,
                    validator.DEFAULT_DATA_MANIFEST,
                    validator.DEFAULT_PROCESSOR_SUMMARY,
                    processor_sha,
                )

    def test_resealed_dangerous_complete_argument_drift_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_RUN_CONTRACT)
        value["intended_ms_swift_args"].update(
            {
                "enable_thinking": True,
                "learning_rate": 1.0,
                "packing": True,
                "target_regex": ".*",
            }
        )
        value["contract_sha256"] = validator.object_sha256(
            value, "contract_sha256"
        )
        processor = validator.load_json(validator.DEFAULT_PROCESSOR_SUMMARY)
        processor_sha = validator.verify_self(
            processor, "summary_sha256", "processor"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-contract.json"
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.Day23ValidationError, "complete intended"
            ):
                validator._validate_run_contract(
                    path,
                    validator.DEFAULT_DATA_MANIFEST,
                    validator.DEFAULT_PROCESSOR_SUMMARY,
                    processor_sha,
                )

    def test_resealed_data_compiler_source_inventory_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_DATA_MANIFEST)
        value["compiler"]["source_files"] = {}
        parent = validator._validate_day21(value)
        with self.assertRaisesRegex(
            validator.Day23ValidationError, "compiler contract/source"
        ):
            validator._validate_day22_and_compiled(value, parent)

    def test_resealed_empty_processor_snapshot_inventory_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_PROCESSOR_SUMMARY)
        value["processor"]["snapshot_files"] = {}
        value["processor"]["snapshot_files_sha256"] = validator.object_sha256({})
        value["summary_sha256"] = validator.object_sha256(
            value, "summary_sha256"
        )
        data_manifest = validator.DEFAULT_DATA_MANIFEST
        source = validator.load_json(data_manifest)
        parent = validator._validate_day21(source)
        pairs, compiled = validator._validate_day22_and_compiled(source, parent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processor-summary.json"
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.Day23ValidationError, "processor asset inventory"
            ):
                validator._validate_processor(
                    path,
                    data_manifest,
                    pairs,
                    compiled,
                )

    def test_resealed_processor_data_content_binding_is_rejected(self) -> None:
        value = validator.load_json(validator.DEFAULT_PROCESSOR_SUMMARY)
        value["data_manifest"]["manifest_sha256"] = "0" * 64
        value["summary_sha256"] = validator.object_sha256(
            value, "summary_sha256"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processor-summary.json"
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.Day23ValidationError, "processor/data manifest binding"
            ):
                validator._validate_processor(
                    path,
                    validator.DEFAULT_DATA_MANIFEST,
                    {},
                    {},
                )


if __name__ == "__main__":
    unittest.main()
