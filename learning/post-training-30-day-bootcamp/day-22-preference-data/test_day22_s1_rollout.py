#!/usr/bin/env python3
"""Pure CPU tests for the resumable two-shard Day 22 S1 rollout contract."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import rollout_day22_s1 as rollout


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class Day22S1RolloutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        records = []
        for task_id in range(600, 600 + rollout.EXPECTED_FAMILIES):
            family_id = f"mbpp:task:{task_id}"
            prompt_text = f"Complete this function.\n\ndef f_{task_id}(x):"
            record = {
                "schema_name": "day22.mbpp_seed",
                "schema_version": 1,
                "task_id": task_id,
                "task_family_id": family_id,
                "split": "train" if task_id % 3 else "dev",
                "problem": {"text": f"problem {task_id}", "sha256": "p"},
                "prompt": {
                    "text": prompt_text,
                    "sha256": rollout.text_sha256(prompt_text),
                },
                "code_prefix": f"def f_{task_id}(x):",
                "entry_point": f"f_{task_id}",
                "tests": {"test_list": [f"assert f_{task_id}(1) == 1"], "sha256": "t"},
                "family_keys": {"problem": family_id, "source": family_id},
                "source": {"dataset": "google-research-datasets/mbpp"},
            }
            record["record_sha256"] = rollout.object_sha256(record)
            records.append(record)
        self.seed = {
            "header": {
                "schema_name": "day22.mbpp_seed_manifest",
                "schema_version": 1,
                "selection": {"records": rollout.EXPECTED_FAMILIES},
            },
            "records": records,
        }
        self.seed["manifest_sha256"] = rollout.object_sha256(self.seed)
        self.seed_path = self.root / "seed.json"
        write_json(self.seed_path, self.seed)
        self.output_root = self.root / "output"
        self.runtime = {
            "ms_swift_commit": rollout.MS_SWIFT_COMMIT,
            "platform": "test",
            "python": "test",
            "versions": dict(rollout.EXPECTED_RUNTIME),
        }
        self.runtime["runtime_sha256"] = rollout.object_sha256(self.runtime)
        self.contract = rollout.build_contract(
            seed_path=self.seed_path,
            seed_manifest=self.seed,
            model_path=self.root / "merged-s1",
            model_files={"config.json": {"bytes": 2, "sha256": "a"}, "model.safetensors": {"bytes": 3, "sha256": "b"}},
            lineage_identity={
                "path": "/lineage.json",
                "file_sha256": "c" * 64,
                "promotion_manifest_sha256": "d" * 64,
                "downstream_key": "qwen35-s1-main-s20260809-lr1e-4-final",
                "merged_export_manifest_sha256": "f" * 64,
            },
            output_root=self.output_root,
            runtime=self.runtime,
            implementation_sha256="e" * 64,
        )
        self.contract_path = self.root / "contract.json"
        write_json(self.contract_path, self.contract)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_partition_is_disjoint_balanced_complete_and_deterministic(self) -> None:
        first = rollout.expected_specs(self.seed["records"])
        second = rollout.expected_specs(list(reversed(self.seed["records"])))
        self.assertEqual(
            [(row["candidate_id"], row["shard_id"], row["seed"]) for row in first],
            [(row["candidate_id"], row["shard_id"], row["seed"]) for row in second],
        )
        self.assertEqual(len(first), 1980)
        self.assertEqual(len({row["candidate_id"] for row in first}), 1980)
        self.assertEqual(len({row["seed"] for row in first}), 1980)
        self.assertEqual(sum(row["shard_id"] == 0 for row in first), 990)
        self.assertEqual(sum(row["shard_id"] == 1 for row in first), 990)
        family_shards: dict[str, set[int]] = {}
        for row in first:
            family_shards.setdefault(row["record"]["task_family_id"], set()).add(row["shard_id"])
        self.assertTrue(all(len(shards) == 1 for shards in family_shards.values()))

    def test_contract_and_empty_status_validate(self) -> None:
        manifest = rollout.validate_seed_manifest(self.seed_path)
        self.assertEqual(manifest["manifest_sha256"], self.seed["manifest_sha256"])
        contract, _ = rollout.verify_contract(self.contract_path, live=False)
        self.assertEqual(contract, self.contract)
        status = rollout.rollout_status(self.contract_path)
        self.assertEqual(status["expected"], 1980)
        self.assertEqual(status["present"], 0)
        self.assertFalse(status["complete"])
        self.assertEqual(status["shards"]["0"], {"expected": 990, "present": 0})

    def test_candidate_row_self_hash_and_context(self) -> None:
        spec = rollout.expected_specs(self.seed["records"])[0]
        row = rollout.build_candidate_row(
            contract=self.contract,
            spec=spec,
            message_content="    return x",
            prompt_token_ids=[1, 2],
            generated_token_ids=[3, 4],
            generated_text="    return x",
            finish_reason="stop",
            elapsed_seconds=0.25,
            response_adapter={"boundary_source": "generated_token_ids_decode"},
            format_contract={"valid": True, "execution_eligible": True, "evidence": {}, "error": None},
        )
        rollout.validate_candidate_row(row, contract=self.contract, spec=spec)
        self.assertEqual(row["task_family_id"], spec["record"]["task_family_id"])
        self.assertEqual(row["response"]["generated_token_count"], 2)
        self.assertEqual(
            row["generator"]["promotion_manifest_sha256"], "d" * 64
        )
        self.assertEqual(
            row["generator"]["merged_export_manifest_sha256"], "f" * 64
        )
        self.assertEqual(
            row["generator"]["generation_config_sha256"],
            rollout.object_sha256(self.contract["generation"]),
        )
        tampered = copy.deepcopy(row)
        tampered["response"]["text"] = "    return 0"
        with self.assertRaisesRegex(rollout.Day22RolloutError, "candidate_sha256"):
            rollout.validate_candidate_row(tampered, contract=self.contract, spec=spec)

    def test_atomic_candidate_publish_is_idempotently_verifiable(self) -> None:
        spec = rollout.expected_specs(self.seed["records"])[0]
        row = rollout.build_candidate_row(
            contract=self.contract,
            spec=spec,
            message_content="    return x",
            prompt_token_ids=[1],
            generated_token_ids=[2],
            generated_text="    return x",
            finish_reason="stop",
            elapsed_seconds=0.1,
            response_adapter={},
            format_contract={"valid": True},
        )
        path = rollout.row_path(self.output_root, spec["shard_id"], spec["candidate_id"])
        rollout.write_json_new(path, row)
        existing = rollout.load_or_none(path, contract=self.contract, spec=spec)
        self.assertEqual(existing, row)
        with self.assertRaisesRegex(rollout.Day22RolloutError, "overwrite"):
            rollout.write_json_new(path, row)

    def test_seed_and_contract_tampering_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.seed)
        tampered["records"][0]["prompt"]["text"] = "changed"
        write_json(self.seed_path, tampered)
        with self.assertRaisesRegex(rollout.Day22RolloutError, "manifest_sha256"):
            rollout.validate_seed_manifest(self.seed_path)

    def test_lineage_manifest_binds_downstream_and_export(self) -> None:
        lineage = {
            "schema_name": "day21.qwen35_downstream_ready_s1_promotion",
            "schema_version": 1,
            "status": "promoted",
            "downstream_ready": True,
            "role": "S1",
            "state": {"candidate": rollout.WINNER},
            "training": {"inference_template": "qwen3_5"},
            "downstream_key": "qwen35-s1-main-s20260809-lr1e-4-final",
            "inference_export": {
                "path": str((self.root / "merged-s1").resolve()),
                "manifest_sha256": "a" * 64,
                "fresh_process_exact_token_id_parity_passed": True,
            },
        }
        lineage["promotion_manifest_sha256"] = rollout.object_sha256(lineage)
        path = self.root / "promotion.json"
        write_json(path, lineage)
        identity = rollout.lineage_manifest_identity(path)
        self.assertEqual(
            identity["promotion_manifest_sha256"],
            lineage["promotion_manifest_sha256"],
        )
        self.assertEqual(identity["merged_export_manifest_sha256"], "a" * 64)
        lineage["downstream_key"] = "changed"
        write_json(path, lineage)
        with self.assertRaisesRegex(rollout.Day22RolloutError, "promotion_manifest_sha256"):
            rollout.lineage_manifest_identity(path)


if __name__ == "__main__":
    unittest.main()
