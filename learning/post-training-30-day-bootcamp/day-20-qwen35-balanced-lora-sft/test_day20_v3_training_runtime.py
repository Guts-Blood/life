#!/usr/bin/env python3
"""Offline tests for the layered Day 20 v3 training runtime."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_train_runtime_v3 as runtime
from test_day20_v2_training_runtime import make_input_fixture, write_json


def _rehash(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = {key: item for key, item in value.items() if key != field}
    result[field] = runtime.object_sha256(result)
    return result


def make_v3_fixture(root: Path) -> dict[str, Path]:
    paths = make_input_fixture(root)
    run_root = paths["run_root"]
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["template"] = {
        "model_type": "qwen3_5",
        "template": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "default+ignore_empty_think",
        "padding_free": False,
        "packing": False,
        "max_length": 2304,
        "truncation_strategy": "delete",
    }
    config = _rehash(config, "immutable_sha256")
    write_json(paths["config"], config)

    core = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    identity = core["configs"]["probes"]["1e-4"]
    identity["file_sha256"] = runtime.file_sha256(paths["config"])
    identity["immutable_sha256"] = config["immutable_sha256"]
    core["datasets"]["main"] = json.loads(json.dumps(core["datasets"]["probe"]))
    core["datasets"]["diagnostic"] = json.loads(
        json.dumps(core["datasets"]["probe"])
    )
    core = _rehash(core, "manifest_sha256")
    write_json(paths["manifest"], core)

    contract = {
        "contract_version": runtime.EXPECTED_TARGET_CONTRACT_VERSION,
        "template_alias": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
        "preserve_code_continuation_leading_whitespace": True,
    }
    contract_sha = runtime.object_sha256(contract)
    target_module = root / "day20_target_encoding_v3.py"
    target_module.write_text(
        "\n".join(
            (
                f"TARGET_TEMPLATE_ALIAS = {runtime.EXPECTED_TARGET_TEMPLATE_ALIAS!r}",
                f"TARGET_ENCODING_CONTRACT_VERSION = {runtime.EXPECTED_TARGET_CONTRACT_VERSION!r}",
                f"TARGET_ENCODING_CONTRACT = {contract!r}",
                f"TARGET_ENCODING_CONTRACT_SHA256 = {contract_sha!r}",
                "def register_target_template_v3():",
                "    return TARGET_TEMPLATE_ALIAS",
                "",
            )
        ),
        encoding="utf-8",
    )

    dataset_identity = core["datasets"]["probe"]
    audit_datasets: dict[str, Any] = {}
    records = int(dataset_identity["records"])
    code_records = 1
    non_code_records = records - code_records
    tokens = int(dataset_identity["supervised_tokens"])
    for name in ("probe", "main"):
        module = {
            "schema_name": runtime.TARGET_AUDIT_DOMAIN,
            "schema_version": 3,
            "status": "pass",
            "contract_version": runtime.EXPECTED_TARGET_CONTRACT_VERSION,
            "contract_sha256": contract_sha,
            "template_alias": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
            "records": records,
            "code_records": code_records,
            "non_code_records": non_code_records,
            "native_supervised_tokens": tokens,
            "target_supervised_tokens": tokens,
            "code_exact_four_space_labels": code_records,
            "code_per_row_supervised_count_preserved": code_records,
            "non_code_input_ids_labels_identical": non_code_records,
            "ordered_target_evidence_sha256": "b" * 64,
        }
        module["audit_sha256"] = runtime.object_sha256(module)
        audit_datasets[name] = {
            "local": {
                "status": "pass",
                "records": records,
                "code_records": code_records,
                "non_code_records": non_code_records,
                "supervised_tokens": tokens,
                "ordered_sample_ids_sha256": "a" * 64,
                "ordered_target_evidence_sha256": "b" * 64,
            },
            "module": module,
        }
    audit_datasets["diagnostic"] = {
        "status": "pass",
        "path": dataset_identity["path"],
        "file_sha256": dataset_identity["file_sha256"],
        "records": records,
        "template_alias": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
        "contract_sha256": contract_sha,
        "copy_identical": True,
    }
    audit = {
        "schema_version": 3,
        "domain": runtime.TARGET_AUDIT_DOMAIN,
        "status": "pass",
        "template_alias": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
        "contract_version": runtime.EXPECTED_TARGET_CONTRACT_VERSION,
        "contract_sha256": contract_sha,
        "datasets": audit_datasets,
    }
    audit = _rehash(audit, "audit_sha256")
    audit_path = run_root / "TARGET-ENCODING-AUDIT.json"
    write_json(audit_path, audit)

    outer = {
        "schema_version": 3,
        "domain": runtime.OUTER_MANIFEST_DOMAIN,
        "status": "prepared",
        "run_root": str(run_root.resolve()),
        "core_v2": {
            "manifest": {
                "path": str(paths["manifest"].resolve()),
                "file_sha256": runtime.file_sha256(paths["manifest"]),
                "content_sha256": core["manifest_sha256"],
            }
        },
        "target_encoding": {
            "template_alias": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
            "contract_version": runtime.EXPECTED_TARGET_CONTRACT_VERSION,
            "contract_sha256": contract_sha,
            "module": {
                "path": str(target_module.resolve()),
                "file_sha256": runtime.file_sha256(target_module),
            },
            "audit": {
                "path": str(audit_path.resolve()),
                "file_sha256": runtime.file_sha256(audit_path),
                "content_sha256": audit["audit_sha256"],
            },
        },
        "datasets": core["datasets"],
        "configs": core["configs"],
    }
    outer = _rehash(outer, "manifest_sha256")
    outer_path = run_root / "DAY20-V3-MANIFEST.json"
    write_json(outer_path, outer)
    return {
        **paths,
        "outer": outer_path,
        "audit": audit_path,
        "target_module": target_module,
    }


def validate(paths: dict[str, Path]) -> dict[str, Any]:
    return runtime.validate_prepared_inputs(
        paths["outer"],
        core_manifest_path=paths["manifest"],
        audit_path=paths["audit"],
        target_module_path=paths["target_module"],
        dataset_path=paths["dataset"],
        config_path=paths["config"],
        run_kind="probe",
        seed=runtime.validate_v2_inputs.__globals__["PRIMARY_SEED"],
        learning_rate="1e-4",
    )


class Day20V3TrainingRuntimeTests(unittest.TestCase):
    def test_outer_core_target_and_full_audit_validate_as_distinct_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_v3_fixture(Path(temporary))
            result = validate(paths)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["target_module"]["template_alias"],
                runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
            )
            self.assertFalse(
                result["identity_layers"]["v3_covered_by_v2_runtime_hash"]
            )
            self.assertEqual(result["manifest_sha256"], result["core_manifest"]["content_sha256"])

    def test_outer_config_or_audit_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_v3_fixture(Path(temporary))
            outer = json.loads(paths["outer"].read_text(encoding="utf-8"))
            outer["configs"] = {}
            write_json(paths["outer"], _rehash(outer, "manifest_sha256"))
            with self.assertRaises(runtime.Day20TrainingRuntimeV3Error):
                validate(paths)

        with tempfile.TemporaryDirectory() as temporary:
            paths = make_v3_fixture(Path(temporary))
            audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
            audit["datasets"]["probe"]["module"][
                "code_exact_four_space_labels"
            ] = 0
            write_json(paths["audit"], _rehash(audit, "audit_sha256"))
            with self.assertRaises((runtime.Day20TrainingRuntimeV3Error, Exception)):
                validate(paths)

    def test_missing_target_module_and_wrong_training_alias_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_v3_fixture(Path(temporary))
            paths["target_module"].unlink()
            with self.assertRaises(Exception):
                validate(paths)

        with tempfile.TemporaryDirectory() as temporary:
            paths = make_v3_fixture(Path(temporary))
            config = json.loads(paths["config"].read_text(encoding="utf-8"))
            config["template"]["template"] = "qwen3_5"
            write_json(paths["config"], _rehash(config, "immutable_sha256"))
            with self.assertRaises(Exception):
                validate(paths)

    def test_attestation_is_self_hashed_and_checks_actual_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_v3_fixture(Path(temporary))
            validated = validate(paths)
            binding_path = paths["run_root"] / "evidence" / "attempt" / "binding.json"
            binding_path.parent.mkdir(parents=True)
            binding: dict[str, Any] = {
                "schema_version": 3,
                "domain": runtime.BINDING_CONFIG_DOMAIN,
                "status": "prepared",
                "run_root": validated["run_root"],
                "evidence_dir": str(binding_path.parent),
                "identity_layers": validated["identity_layers"],
                "outer_manifest": validated["outer_manifest"],
                "core_manifest": validated["core_manifest"],
                "dataset": validated["dataset"],
                "training_config": validated["training_config"],
                "target_module": validated["target_module"],
                "target_audit": validated["target_audit"],
                "target_runtime": runtime.target_runtime_identity(
                    paths["target_module"]
                ),
                "v2_callback_config": {
                    "path": str(paths["config"]),
                    "file_sha256": runtime.file_sha256(paths["config"]),
                    "content_sha256": "a" * 64,
                },
                "run_kind": validated["run_kind"],
                "seed": validated["seed"],
                "learning_rate": validated["learning_rate"],
            }
            binding["binding_config_sha256"] = runtime.object_sha256(binding)
            write_json(binding_path, binding)
            # This unit isolates the live-template check from v2 callback plumbing.
            original = runtime.verify_binding_config
            runtime.verify_binding_config = lambda _: binding  # type: ignore[assignment]
            try:
                with self.assertRaises(runtime.Day20TrainingRuntimeV3Error):
                    runtime.build_attestation(
                        binding_path,
                        actual_template_alias="qwen3_5",
                        template_class="FakeTemplate",
                        trainer_arguments={},
                    )
                attestation = runtime.build_attestation(
                    binding_path,
                    actual_template_alias=runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
                    template_class="FakeTemplate",
                    trainer_arguments={"template": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS},
                )
            finally:
                runtime.verify_binding_config = original  # type: ignore[assignment]
            self.assertEqual(
                attestation["attestation_sha256"],
                runtime.object_sha256(
                    {
                        key: value
                        for key, value in attestation.items()
                        if key != "attestation_sha256"
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main()
