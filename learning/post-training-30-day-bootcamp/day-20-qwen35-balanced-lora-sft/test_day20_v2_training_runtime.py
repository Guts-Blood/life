#!/usr/bin/env python3
"""Offline tests for the isolated Day 20 v2 training runtime contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_train_runtime_v2 as runtime


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = runtime.object_sha256(result)
    return result


def dataset_rows(run_kind: str) -> list[dict[str, Any]]:
    if run_kind == "probe":
        count, tokens = 64, 375
    else:
        count, tokens = 80, 4_000
    return [
        {
            "sample_id": f"{run_kind}-{index:04d}",
            "skill": runtime.SKILLS[index % len(runtime.SKILLS)],
            "qwen35_supervised_tokens": tokens,
        }
        for index in range(count)
    ]


def temporal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return runtime.audit_temporal_mix(rows, global_batch_size=runtime.GLOBAL_BATCH_SIZE)


def make_input_fixture(root: Path, run_kind: str = "probe") -> dict[str, Path]:
    run_root = root / "run"
    (run_root / "data").mkdir(parents=True)
    (run_root / "configs").mkdir()
    rows = dataset_rows(run_kind)
    dataset = run_root / "data" / f"{run_kind}.jsonl"
    dataset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = temporal_audit(rows)
    audit_hash = runtime.object_sha256(audit)
    dataset_hash = runtime.file_sha256(dataset)
    lr = "1e-4"
    seed = runtime.PRIMARY_SEED
    configured_lr: str | float = (
        lr if run_kind == "probe" else "__SELECT_FROM_PASSING_PROBE__"
    )
    config = self_hashed(
        {
            "schema_version": 2,
            "domain": runtime.CONFIG_DOMAIN,
            "run_kind": run_kind,
            "data": {
                "path": str(dataset.resolve()),
                "file_sha256": dataset_hash,
                "supervised_tokens": runtime.EXPECTED_TOKENS[run_kind],
                "temporal_mix_sha256": audit_hash,
            },
            "training": {
                "seed": seed,
                "data_seed": seed,
                "global_batch_size": 8,
                "learning_rate": configured_lr,
                "checkpoint_supervised_tokens": list(
                    runtime.CHECKPOINT_TARGETS[run_kind].values()
                ),
                "checkpoint_policy": "all_milestones_resumable_v2",
            },
        },
        "immutable_sha256",
    )
    config_path = run_root / "configs" / (
        "probe-lr-1e-4.json" if run_kind == "probe" else "main-template.json"
    )
    write_json(config_path, config)
    config_identity = {
        "path": str(config_path.resolve()),
        "file_sha256": runtime.file_sha256(config_path),
        "immutable_sha256": config["immutable_sha256"],
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "domain": runtime.MANIFEST_DOMAIN,
        "status": "prepared",
        "run_root": str(run_root.resolve()),
        "contract": {"candidate_factory_version": runtime.CONTRACT_VERSION},
        "datasets": {
            run_kind: {
                "path": str(dataset.resolve()),
                "file_sha256": dataset_hash,
                "records": len(rows),
                "supervised_tokens": sum(
                    row["qwen35_supervised_tokens"] for row in rows
                ),
                "ordered_supervised_tokens": [
                    row["qwen35_supervised_tokens"] for row in rows
                ],
                "temporal_mix_audit": audit,
                "temporal_mix_sha256": audit_hash,
            }
        },
        "configs": (
            {"probes": {lr: config_identity}}
            if run_kind == "probe"
            else {"main_template": config_identity}
        ),
    }
    manifest = self_hashed(manifest, "manifest_sha256")
    manifest_path = run_root / "DAY20-MANIFEST.json"
    write_json(manifest_path, manifest)
    return {
        "run_root": run_root,
        "manifest": manifest_path,
        "dataset": dataset,
        "config": config_path,
    }


def make_checkpoint(
    run_root: Path,
    *,
    run_kind: str,
    step: int,
    cumulative_tokens: int,
    target: str,
    complete: bool = True,
) -> Path:
    output = run_root / "adapters" / run_kind / "attempt-001"
    checkpoint = output / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    files = {
        "adapter_config.json": b"{}\n",
        "adapter_model.safetensors": b"adapter",
        "trainer_state.json": b"{}\n",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state_0.pth": b"rng",
    }
    if not complete:
        files.pop("optimizer.pt")
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
    if complete:
        integrity = runtime.build_checkpoint_integrity(
            checkpoint,
            run_root=run_root,
            run_kind=run_kind,
            seed=runtime.PRIMARY_SEED,
            learning_rate="1e-4",
            global_step=step,
            cumulative_supervised_tokens=cumulative_tokens,
            checkpoint_targets=[target],
            dataset_file_sha256="a" * 64,
            training_config_file_sha256="b" * 64,
            experiment_manifest_sha256="c" * 64,
            temporal_mix_sha256="d" * 64,
            runtime_sha256="e" * 64,
        )
        runtime.write_checkpoint_integrity(checkpoint, integrity)
    return checkpoint


class Day20V2TrainingRuntimeTests(unittest.TestCase):
    def test_checkpoint_steps_use_eight_record_batches_and_all_milestones(self) -> None:
        self.assertEqual(
            runtime.checkpoint_steps([3_000] * 8, run_kind="probe")[
                "checkpoint_steps"
            ],
            {"t6000": 2, "t12000": 4, "t18000": 6, "t24000": 8},
        )
        main = runtime.checkpoint_steps([32_000] * 10, run_kind="main")
        self.assertEqual(
            main["checkpoint_steps"], {"early": 3, "mid": 6, "final": 10}
        )
        self.assertEqual(main["checkpoint_actual_tokens"]["early"], 96_000)
        with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
            runtime.checkpoint_steps([3_000] * 7, run_kind="probe")

    def test_validate_inputs_binds_manifest_data_config_and_temporal_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_input_fixture(Path(temporary))
            result = runtime.validate_inputs(
                paths["manifest"],
                dataset_path=paths["dataset"],
                config_path=paths["config"],
                run_kind="probe",
                seed=runtime.PRIMARY_SEED,
                learning_rate="0.0001",
            )
            self.assertEqual(result["supervised_tokens"], 24_000)
            self.assertEqual(result["checkpoint_steps"]["t24000"], 8)
            self.assertFalse(result["config_requires_resolution"])

            config = json.loads(paths["config"].read_text(encoding="utf-8"))
            config["data"]["temporal_mix_sha256"] = "0" * 64
            config = self_hashed(
                {key: value for key, value in config.items() if key != "immutable_sha256"},
                "immutable_sha256",
            )
            write_json(paths["config"], config)
            with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
                runtime.validate_inputs(
                    paths["manifest"],
                    dataset_path=paths["dataset"],
                    config_path=paths["config"],
                    run_kind="probe",
                    seed=runtime.PRIMARY_SEED,
                    learning_rate="1e-4",
                )

    def test_main_template_is_valid_but_explicitly_requires_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_input_fixture(Path(temporary), "main")
            result = runtime.validate_inputs(
                paths["manifest"],
                dataset_path=paths["dataset"],
                config_path=paths["config"],
                run_kind="main",
                seed=runtime.PRIMARY_SEED,
                learning_rate="1e-4",
            )
            self.assertEqual(result["supervised_tokens"], 320_000)
            self.assertTrue(result["config_requires_resolution"])

    def test_probe_and_main_checkpoint_packages_are_always_resumable(self) -> None:
        cases = (
            ("probe", 2, 6_000, "t6000"),
            ("main", 3, 96_000, "early"),
        )
        for run_kind, step, tokens, target in cases:
            with self.subTest(run_kind=run_kind), tempfile.TemporaryDirectory() as temporary:
                run_root = Path(temporary) / "run"
                checkpoint = make_checkpoint(
                    run_root,
                    run_kind=run_kind,
                    step=step,
                    cumulative_tokens=tokens,
                    target=target,
                )
                verified = runtime.verify_checkpoint_integrity(
                    checkpoint,
                    run_root=run_root,
                    run_kind=run_kind,
                    seed=runtime.PRIMARY_SEED,
                    learning_rate="1e-4",
                )
                self.assertTrue(verified["resumable"])
                self.assertEqual(
                    verified["candidate_ids"][target],
                    runtime.candidate_id(
                        run_kind=run_kind,
                        seed=runtime.PRIMARY_SEED,
                        learning_rate="1e-4",
                        checkpoint=target,
                    ),
                )

    def test_checkpoint_rejects_missing_resume_state_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            incomplete = make_checkpoint(
                run_root,
                run_kind="probe",
                step=2,
                cumulative_tokens=6_000,
                target="t6000",
                complete=False,
            )
            with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
                runtime.build_checkpoint_integrity(
                    incomplete,
                    run_root=run_root,
                    run_kind="probe",
                    seed=runtime.PRIMARY_SEED,
                    learning_rate="1e-4",
                    global_step=2,
                    cumulative_supervised_tokens=6_000,
                    checkpoint_targets=["t6000"],
                    dataset_file_sha256="a" * 64,
                    training_config_file_sha256="b" * 64,
                    experiment_manifest_sha256="c" * 64,
                    temporal_mix_sha256="d" * 64,
                    runtime_sha256="e" * 64,
                )

            complete = make_checkpoint(
                run_root,
                run_kind="main",
                step=3,
                cumulative_tokens=96_000,
                target="early",
            )
            (complete / "optimizer.pt").write_bytes(b"tampered")
            with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
                runtime.verify_checkpoint_integrity(complete, run_root=run_root)

    def test_latest_resumable_is_generic_and_confined_to_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            valid = make_checkpoint(
                run_root,
                run_kind="probe",
                step=2,
                cumulative_tokens=6_000,
                target="t6000",
            )
            output = valid.parent
            partial = output / "checkpoint-3"
            partial.mkdir()
            (partial / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                runtime.latest_resumable_checkpoint(
                    output, run_root=run_root, run_kind="probe"
                ),
                valid.resolve(),
            )
            with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
                runtime.latest_resumable_checkpoint(
                    output,
                    run_root=run_root,
                    run_kind="probe",
                    temporal_mix_sha256="f" * 64,
                )
            with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
                runtime.latest_resumable_checkpoint(
                    output,
                    run_root=run_root,
                    run_kind="probe",
                    runtime_sha256="f" * 64,
                )
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaises(runtime.Day20TrainingRuntimeV2Error):
                runtime.latest_resumable_checkpoint(outside, run_root=run_root)

    def test_cli_exposes_all_four_contract_commands(self) -> None:
        source = (HERE / "day20_train_runtime_v2.py").read_text(encoding="utf-8")
        for command in (
            "validate-inputs",
            "checkpoint-steps",
            "verify-checkpoint",
            "latest-resumable",
        ):
            self.assertIn(f'add_parser("{command}")', source)
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_input_fixture(Path(temporary))
            validated = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "day20_train_runtime_v2.py"),
                    "validate-inputs",
                    "--manifest",
                    str(paths["manifest"]),
                    "--dataset",
                    str(paths["dataset"]),
                    "--config",
                    str(paths["config"]),
                    "--run-kind",
                    "probe",
                    "--seed",
                    str(runtime.PRIMARY_SEED),
                    "--learning-rate",
                    "1e-4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(validated.stdout)["status"], "pass")

            stepped = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "day20_train_runtime_v2.py"),
                    "checkpoint-steps",
                    "--dataset",
                    str(paths["dataset"]),
                    "--run-kind",
                    "probe",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(stepped.stdout)
            self.assertEqual(payload["checkpoint_steps"]["t24000"], 8)

            checkpoint = make_checkpoint(
                paths["run_root"],
                run_kind="probe",
                step=2,
                cumulative_tokens=6_000,
                target="t6000",
            )
            verified = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "day20_train_runtime_v2.py"),
                    "verify-checkpoint",
                    "--checkpoint",
                    str(checkpoint),
                    "--run-root",
                    str(paths["run_root"]),
                    "--run-kind",
                    "probe",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(json.loads(verified.stdout)["resumable"])
            latest = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "day20_train_runtime_v2.py"),
                    "latest-resumable",
                    "--output-dir",
                    str(checkpoint.parent),
                    "--run-root",
                    str(paths["run_root"]),
                    "--run-kind",
                    "probe",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(Path(latest.stdout.strip()), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
