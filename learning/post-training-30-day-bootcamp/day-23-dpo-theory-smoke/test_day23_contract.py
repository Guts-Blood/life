#!/usr/bin/env python3
"""Offline tests for Day 23 ancestry, compilation, and run contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest

import build_day23_cpu_contract
import day23_contract as contract
import prepare_day23_qwen35_dpo as prepare


class Day23ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiled, cls.manifest = prepare.build_bundle()
        cls.promotion = contract.load_json(prepare.DEFAULT_PROMOTION)
        cls.key = contract.load_json(prepare.DEFAULT_KEY)
        cls.export = contract.load_json(prepare.DEFAULT_EXPORT)
        experimental = contract.load_json(prepare.DEFAULT_EXPERIMENTAL)
        cls.source_pairs = contract.load_jsonl(
            prepare.BOOTCAMP_ROOT / experimental["outputs"]["pairs"]["path"]
        )
        cls.source_by_id = {row["pair_id"]: row for row in cls.source_pairs}

    def parent(self, promotion=None, key=None, export=None):
        return contract.validate_s1_bundle(
            promotion or self.promotion,
            key or self.key,
            export or self.export,
            promotion_file_sha256=contract.file_sha256(prepare.DEFAULT_PROMOTION),
            downstream_key_file_sha256=contract.file_sha256(prepare.DEFAULT_KEY),
            merged_export_file_sha256=contract.file_sha256(prepare.DEFAULT_EXPORT),
        )

    def test_real_bundle_is_154_17_29_and_gpu_pending(self) -> None:
        self.assertEqual(
            self.manifest["counts"]["split_counts"],
            {"train": 154, "dev": 17, "heldout": 29},
        )
        self.assertEqual(self.manifest["status"], "cpu_data_ready")
        self.assertFalse(self.manifest["claim_boundary"]["gpu_optimizer_ready"])
        self.assertEqual(
            self.manifest["claim_boundary"]["remote_payload_gate"],
            contract.REMOTE_PAYLOAD_GATE,
        )

    def test_compilation_is_byte_deterministic(self) -> None:
        second_rows, second_manifest = prepare.build_bundle()
        self.assertEqual(
            prepare.materialized_bytes(self.compiled, self.manifest),
            prepare.materialized_bytes(second_rows, second_manifest),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = __import__("pathlib").Path(directory)
            payloads = prepare.materialized_bytes(self.compiled, self.manifest)
            prepare.apply_bundle(output, payloads, mode="build")
            prepare.apply_bundle(output, payloads, mode="check")

    def test_compilation_preserves_exact_prompt_and_response_bytes(self) -> None:
        row = self.compiled["train"][0]
        source = self.source_by_id[row["pair_id"]]
        self.assertEqual(row["messages"][0]["content"], source["prompt"]["text"])
        self.assertEqual(row["messages"][1]["content"], source["chosen"]["text"])
        self.assertEqual(row["rejected_response"], source["rejected"]["text"])
        self.assertTrue(row["messages"][1]["content"].startswith("    "))

    def test_resealed_non_promoted_parent_is_rejected(self) -> None:
        value = copy.deepcopy(self.promotion)
        value["status"] = "candidate"
        value["promotion_manifest_sha256"] = contract.object_sha256(
            value, "promotion_manifest_sha256"
        )
        with self.assertRaisesRegex(contract.Day23ContractError, "not promoted"):
            self.parent(promotion=value)

    def test_resealed_downstream_key_cross_binding_is_rejected(self) -> None:
        value = copy.deepcopy(self.key)
        value["checkpoint_id"] = "qwen35-4b-base-s0"
        value["key_sha256"] = contract.object_sha256(value, "key_sha256")
        with self.assertRaisesRegex(contract.Day23ContractError, "checkpoint_id"):
            self.parent(key=value)

    def test_resealed_export_lineage_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.export)
        value["source"]["checkpoint_integrity_sha256"] = "0" * 64
        value["manifest_sha256"] = contract.object_sha256(value, "manifest_sha256")
        with self.assertRaises(contract.Day23ContractError):
            self.parent(export=value)

    def test_synthetic_or_off_policy_pair_is_rejected(self) -> None:
        for mutation in (
            lambda row: row["creation"].update({"synthetic": True}),
            lambda row: row["chosen"]["generator"].update({"on_policy": False}),
        ):
            value = copy.deepcopy(self.source_pairs[0])
            mutation(value)
            with self.assertRaises(contract.Day23ContractError):
                contract.validate_pair_lineage(value, self.manifest["parent"])

    def test_label_flip_even_when_resealed_is_rejected(self) -> None:
        source = self.source_pairs[0]
        value = contract.compile_row(source, split=source["split"])
        value["messages"][1]["content"], value["rejected_response"] = (
            value["rejected_response"],
            value["messages"][1]["content"],
        )
        value["row_sha256"] = contract.object_sha256(value, "row_sha256")
        with self.assertRaisesRegex(contract.Day23ContractError, "source projection"):
            contract.validate_compiled_row(value, source_pair=source)

    def test_split_duplicate_and_cross_split_move_fail_closed(self) -> None:
        experimental = contract.load_json(prepare.DEFAULT_EXPERIMENTAL)
        split_path = prepare.BOOTCAMP_ROOT / experimental["outputs"]["split_ids"]["path"]
        split_ids = contract.load_json(split_path)
        split_ids["dev"][0] = split_ids["train"][0]
        split_ids["split_ids_sha256"] = contract.object_sha256(
            split_ids, "split_ids_sha256"
        )
        with self.assertRaises(contract.Day23ContractError):
            contract.validate_split_ids(split_ids, self.source_pairs)

    def test_cpu_run_contract_excludes_heldout_and_keeps_runs_fresh(self) -> None:
        value = build_day23_cpu_contract.build_contract()
        args = json.dumps(value["intended_ms_swift_args"], sort_keys=True)
        heldout = value["dataset_usage"]["heldout"]["path"]
        self.assertNotIn(heldout, args)
        self.assertNotIn("ref_model", value["intended_ms_swift_args"])
        self.assertNotIn("ref_adapters", value["intended_ms_swift_args"])
        self.assertNotIn("adapters", value["intended_ms_swift_args"])
        self.assertEqual(
            value["cli_default_semantics"]["omitted_keys"]["ref_adapters"], []
        )
        self.assertEqual(
            value["intended_ms_swift_args"]["loss_scale"],
            "default+ignore_empty_think",
        )
        self.assertTrue(value["stages"]["mechanism_5step"]["fresh_start_from_parent"])
        self.assertEqual(
            value["stages"]["mechanism_5step"]["remove_common_args"],
            ["val_dataset"],
        )
        self.assertFalse(
            value["stages"]["mechanism_5step"]["train_dataloader_shuffle"]
        )
        self.assertTrue(value["stages"]["bounded_smoke_30step"]["fresh_start_from_parent"])
        self.assertFalse(value["claim_boundary"]["gpu_optimizer_ready"])


if __name__ == "__main__":
    unittest.main()
