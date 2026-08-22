#!/usr/bin/env python3
"""Stdlib-only tests for the Day 20 AutoDL runner and training plugin."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str) -> ModuleType:
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plugin = load_module("day20_train_plugin.py", "day20_test_train_plugin")
contract = load_module("day20_contract.py", "day20_test_contract_for_runner")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_safetensors(path: Path, tensor_name: str = "model.weight") -> None:
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00")


class Day20RunnerTests(unittest.TestCase):
    def test_runner_shell_is_valid_and_pins_the_training_contract(self) -> None:
        runner = HERE / "run_day20_autodl.sh"
        subprocess.run(["bash", "-n", str(runner)], check=True)
        source = runner.read_text(encoding="utf-8")
        self.assertIn(
            "/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/"
            "1001bb4d826a52d1f399e183466143f4da7b741b",
            source,
        )
        for fragment in (
            "--tuner_type lora",
            "--lora_rank 8",
            "--lora_alpha 16",
            "--lora_dropout 0.05",
            "--per_device_train_batch_size 2",
            "--gradient_accumulation_steps 4",
            "--max_length 2304",
            "--truncation_strategy delete",
            "--packing false",
            "CUDA_VISIBLE_DEVICES=0",
            "--experiment-manifest",
            "day20-e2b.env",
            "day20-e2b-attestation.json",
            "--credential-attestation",
            'MS_SWIFT_ROOT="/root/autodl-tmp/ms-swift"',
            "nvidia/cudnn/lib",
            'LD_LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:',
            "flock -n 9",
            "flock -n 8",
            "stage-resume-evidence",
            "verify-winner",
            "seal-export",
            "verify-export",
            "mv -T --",
            "SOURCE-ADAPTER-MANIFEST.json",
            ".normalized.qwen35.jsonl",
            "prepare-in-progress",
            "retaining abandoned prepare attempt",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("day12-e2b.env", source)
        self.assertNotIn("--tuner_type full", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0,1", source)
        self.assertNotIn("${DATA_ROOT}/ms-swift", source)
        self.assertNotRegex(source, r"\brm\b")

    def test_probe_lr_allowlist_is_numeric_but_canonical(self) -> None:
        self.assertEqual(plugin.canonical_lr("0.00001"), "1e-5")
        self.assertEqual(plugin.canonical_lr("3e-5"), "3e-5")
        self.assertEqual(plugin.canonical_lr("1e-4"), "1e-4")
        with self.assertRaises(plugin.Day20PluginError):
            plugin.canonical_lr("2e-5")

    def test_token_schedule_uses_global_batch_eight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "data.jsonl"
            write_jsonl(
                dataset,
                [
                    {"qwen35_supervised_tokens": value}
                    for value in [1, 2, 3, 4, 5, 6, 7, 8, 9]
                ],
            )
            self.assertEqual(plugin.supervised_token_schedule(dataset), [36, 9])
            with self.assertRaises(plugin.Day20PluginError):
                plugin.supervised_token_schedule(dataset, per_device_batch_size=1)

    def test_live_template_reaudit_rejects_fabricated_token_counts(self) -> None:
        class FakeTemplate:
            def encode(self, value: dict[str, Any], return_length: bool) -> dict[str, Any]:
                self.assert_contract = return_length and bool(value["messages"])
                return {"input_ids": [10, 11, 12], "labels": [-100, 11, 12]}

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "data.jsonl"
            messages = [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
            token_ids = [10, 11, 12]
            labels = [-100, 11, 12]
            valid = {
                "sample_id": "sample-1",
                "messages": messages,
                "content_sha256": plugin.object_sha256(messages),
                "qwen35_input_tokens": 3,
                "qwen35_supervised_tokens": 2,
                "qwen35_render_sha256": plugin.object_sha256(token_ids),
                "qwen35_labels_sha256": plugin.object_sha256(labels),
            }
            write_jsonl(dataset, [valid])
            report = plugin.reaudit_prepared_dataset(dataset, FakeTemplate())
            self.assertEqual(report["supervised_tokens"], 2)

            fabricated = dict(valid)
            fabricated["qwen35_supervised_tokens"] = 2000
            write_jsonl(Path(temporary) / "fabricated.jsonl", [fabricated])
            with self.assertRaises(plugin.Day20PluginError):
                plugin.reaudit_prepared_dataset(
                    Path(temporary) / "fabricated.jsonl", FakeTemplate()
                )

    def test_checkpoint_thresholds_record_first_reachable_step(self) -> None:
        self.assertEqual(
            plugin.checkpoint_steps([7, 7, 7], {"early": 10, "final": 21}),
            {"early": 2, "final": 3},
        )
        with self.assertRaises(plugin.Day20PluginError):
            plugin.checkpoint_steps([7, 7], {"final": 15})

    def test_checkpoint_integrity_and_resume_skip_partial_newest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            output = run_root / "adapters" / "main"
            checkpoint = output / "checkpoint-5"
            checkpoint.mkdir(parents=True)
            runtime = {"backend": "test"}
            runtime["runtime_sha256"] = plugin.object_sha256(runtime)
            for name, content in {
                "adapter_config.json": b"{}\n",
                "adapter_model.safetensors": b"adapter",
                "trainer_state.json": b"{}\n",
                "optimizer.pt": b"optimizer",
                "scheduler.pt": b"scheduler",
                "rng_state.pth": b"rng",
            }.items():
                (checkpoint / name).write_bytes(content)
            integrity = plugin.build_checkpoint_integrity(
                checkpoint,
                run_kind="main",
                global_step=5,
                cumulative_supervised_tokens=64_000,
                targets=["early"],
                dataset_file_sha256="a" * 64,
                training_config_file_sha256="b" * 64,
                runtime_sha256=runtime["runtime_sha256"],
            )
            write_json(
                checkpoint / plugin.CHECKPOINT_INTEGRITY_FILE, integrity
            )
            package = {
                "path": str(checkpoint.resolve()),
                "integrity_file_sha256": plugin.file_sha256(
                    checkpoint / plugin.CHECKPOINT_INTEGRITY_FILE
                ),
                "integrity_sha256": integrity["integrity_sha256"],
                "snapshot_sha256": integrity["snapshot_sha256"],
                "files": integrity["files"],
                "resumable": True,
            }
            summary = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_training_summary",
                "status": "pass",
                "run_kind": "main",
                "runtime_identity": runtime,
                "dataset_file_sha256": "a" * 64,
                "training_config": {"file_sha256": "b" * 64},
                "checkpoints": {"early": str(checkpoint.resolve())},
                "checkpoint_packages": {"early": package},
            }
            summary["summary_sha256"] = plugin.object_sha256(summary)
            summary_path = run_root / "training-summary.json"
            write_json(summary_path, summary)
            self.assertEqual(
                plugin.verified_checkpoint_path(
                    summary_path, name="early", run_root=run_root
                ),
                checkpoint.resolve(),
            )

            partial = output / "checkpoint-6"
            partial.mkdir()
            (partial / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                plugin.latest_resumable_checkpoint(output, run_root=run_root),
                checkpoint.resolve(),
            )
            (checkpoint / "optimizer.pt").write_bytes(b"tampered")
            with self.assertRaises(plugin.Day20PluginError):
                plugin.verified_checkpoint_path(
                    summary_path, name="early", run_root=run_root
                )

    def test_resume_stages_only_evidence_through_verified_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            checkpoint = (
                run_root
                / "adapters/main/attempt-001/checkpoint-5"
            )
            checkpoint.mkdir(parents=True)
            runtime = {"backend": "test"}
            runtime["runtime_sha256"] = plugin.object_sha256(runtime)
            for name, content in {
                "adapter_config.json": b"{}\n",
                "adapter_model.safetensors": b"adapter",
                "trainer_state.json": b"{}\n",
                "optimizer.pt": b"optimizer",
                "scheduler.pt": b"scheduler",
                "rng_state.pth": b"rng",
            }.items():
                (checkpoint / name).write_bytes(content)
            integrity = plugin.build_checkpoint_integrity(
                checkpoint,
                run_kind="main",
                global_step=5,
                cumulative_supervised_tokens=500,
                targets=["early"],
                dataset_file_sha256="a" * 64,
                training_config_file_sha256="b" * 64,
                runtime_sha256=runtime["runtime_sha256"],
            )
            write_json(checkpoint / plugin.CHECKPOINT_INTEGRITY_FILE, integrity)

            source = run_root / "evidence/main/attempt-001"
            destination = run_root / "evidence/main/attempt-002"
            source.mkdir(parents=True)
            destination.mkdir()
            trace = [
                {
                    "global_step": step,
                    "loss": 2.0 / step,
                    "grad_norm": 1.0,
                    "learning_rate": 1e-5,
                    "lora_update_ratio": step * 1e-7,
                    "cumulative_supervised_tokens": step * 100,
                }
                for step in range(1, 8)
            ]
            write_jsonl(source / "step-metrics.jsonl", trace)
            inventory = {"status": "pass"}
            inventory["inventory_sha256"] = plugin.object_sha256(inventory)
            write_json(source / "trainable-inventory.json", inventory)
            reaudit = {"status": "pass"}
            reaudit["audit_sha256"] = plugin.object_sha256(reaudit)
            write_json(source / "trainer-token-reaudit.json", reaudit)
            write_json(source / "runtime-identity.json", runtime)
            write_json(
                source / "five-step-safety.json",
                plugin.validate_five_step_metrics(trace),
            )

            staged = plugin.stage_resume_evidence(
                source,
                destination,
                checkpoint=checkpoint,
                run_root=run_root,
            )
            self.assertEqual(staged["global_step"], 5)
            self.assertEqual(
                [row["global_step"] for row in plugin.load_jsonl(
                    destination / "step-metrics.jsonl"
                )],
                [1, 2, 3, 4, 5],
            )
            events = plugin.load_jsonl(destination / "checkpoint-events.jsonl")
            self.assertEqual(events[0]["targets"], ["early"])
            self.assertEqual(events[0]["checkpoint"], str(checkpoint.resolve()))

    def test_winner_only_export_is_sealed_and_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            (run_root / "exports").mkdir(parents=True)
            (run_root / ".day20-run-root").write_text(
                "day20-qwen35-balanced-lora-sft-v1\n", encoding="utf-8"
            )
            training_config_path = run_root / "configs/main-selected.json"
            training_config_path.parent.mkdir()
            write_json(training_config_path, {"status": "selected"})
            checkpoint = run_root / "adapters/main/attempt-001/checkpoint-5"
            checkpoint.mkdir(parents=True)
            runtime = {"backend": "test"}
            runtime["runtime_sha256"] = plugin.object_sha256(runtime)
            for name, content in {
                "adapter_config.json": b"{}\n",
                "adapter_model.safetensors": b"adapter",
                "trainer_state.json": b"{}\n",
                "optimizer.pt": b"optimizer",
                "scheduler.pt": b"scheduler",
                "rng_state.pth": b"rng",
            }.items():
                (checkpoint / name).write_bytes(content)
            integrity = plugin.build_checkpoint_integrity(
                checkpoint,
                run_kind="main",
                global_step=5,
                cumulative_supervised_tokens=64_000,
                targets=["early"],
                dataset_file_sha256="a" * 64,
                training_config_file_sha256=plugin.file_sha256(
                    training_config_path
                ),
                runtime_sha256=runtime["runtime_sha256"],
            )
            write_json(checkpoint / plugin.CHECKPOINT_INTEGRITY_FILE, integrity)
            package = {
                "path": str(checkpoint.resolve()),
                "integrity_file_sha256": plugin.file_sha256(
                    checkpoint / plugin.CHECKPOINT_INTEGRITY_FILE
                ),
                "integrity_sha256": integrity["integrity_sha256"],
                "snapshot_sha256": integrity["snapshot_sha256"],
                "files": integrity["files"],
                "resumable": True,
            }
            training_summary = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_training_summary",
                "status": "pass",
                "run_kind": "main",
                "runtime_identity": runtime,
                "learning_rate": 1e-5,
                "cumulative_supervised_tokens": 256_000,
                "dataset_file_sha256": "a" * 64,
                "training_config": {
                    "path": str(training_config_path.resolve()),
                    "file_sha256": plugin.file_sha256(training_config_path),
                },
                "checkpoints": {"early": str(checkpoint.resolve())},
                "checkpoint_packages": {"early": package},
            }
            training_summary["summary_sha256"] = plugin.object_sha256(
                training_summary
            )
            training_summary_path = run_root / "evidence/main/training-summary.json"
            training_summary_path.parent.mkdir(parents=True)
            write_json(training_summary_path, training_summary)

            artifact_path = run_root / "eval/early-summary.json"
            artifact_path.parent.mkdir()
            write_json(artifact_path, {"status": "complete"})
            metrics = {
                "general_correct": 10,
                "math_correct": 20,
                "finance_correct": 15,
                "code_correct": 20,
                "total_correct": 65,
                "format_compliant": 100,
                "code_sandbox_execution_eligible": 30,
                "infrastructure_failures": 0,
            }
            checks = {
                field: (
                    metrics[field] == minimum
                    if field == "infrastructure_failures"
                    else metrics[field] >= minimum
                )
                for field, minimum in plugin.PROMOTION_THRESHOLDS.items()
            }
            candidate = {
                "candidate": "early",
                "checkpoint_tokens": 64_000,
                "checkpoint_path": str(checkpoint.resolve()),
                "model_identity_sha256": "c" * 64,
                "metrics": metrics,
                "threshold_checks": checks,
                "failed_thresholds": [],
                "promotion_eligible": True,
                "training_identity": {
                    "path": str(training_summary_path.resolve()),
                    "file_sha256": plugin.file_sha256(training_summary_path),
                    "content_sha256": training_summary["summary_sha256"],
                    "learning_rate": 1e-5,
                    "training_config_path": str(training_config_path.resolve()),
                    "training_config_file_sha256": plugin.file_sha256(
                        training_config_path
                    ),
                },
                "artifacts": {
                    "adapter_summary": {
                        "path": str(artifact_path.resolve()),
                        "file_sha256": plugin.file_sha256(artifact_path),
                    }
                },
            }
            results = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_promotion_results",
                "status": "pass",
                "run_root": str(run_root.resolve()),
                "frozen_test_consumed": False,
                "thresholds": plugin.PROMOTION_THRESHOLDS,
                "experiment_manifest": {"base_model_path": "/model/base"},
                "candidates": {"early": candidate},
                "decision": {
                    "status": "eligible_day20_lora_anchor",
                    "selected_candidate": "early",
                    "selected_checkpoint_path": str(checkpoint.resolve()),
                    "selected_checkpoint_tokens": 64_000,
                    "ranked_eligible_candidates": ["early"],
                },
            }
            results["results_sha256"] = plugin.object_sha256(results)
            results_path = run_root / "DAY20-RESULTS.json"
            write_json(results_path, results)
            pass_marker = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_pass",
                "status": "pass",
                "canonical_run": str(run_root.resolve()),
                "selected_candidate": "early",
                "selected_checkpoint_path": str(checkpoint.resolve()),
                "selected_checkpoint_tokens": 64_000,
                "selected_model_identity_sha256": "c" * 64,
                "selected_metrics": metrics,
                "results": {
                    "path": str(results_path.resolve()),
                    "file_sha256": plugin.file_sha256(results_path),
                },
                "frozen_test_consumed": False,
            }
            pass_marker["marker_sha256"] = plugin.object_sha256(pass_marker)
            pass_path = run_root / "DAY20-PASS.json"
            write_json(pass_path, pass_marker)
            verified = plugin.validate_winner_for_export(run_root)
            self.assertEqual(verified["checkpoint"], str(checkpoint.resolve()))

            self_signed_wrong = dict(pass_marker)
            self_signed_wrong["selected_checkpoint_path"] = "/tmp/wrong"
            self_signed_wrong["marker_sha256"] = plugin.object_sha256(
                {
                    key: value
                    for key, value in self_signed_wrong.items()
                    if key != "marker_sha256"
                }
            )
            write_json(pass_path, self_signed_wrong)
            with self.assertRaises(plugin.Day20PluginError):
                plugin.validate_winner_for_export(run_root)
            write_json(pass_path, pass_marker)

            staging = run_root / "exports/.early-merged-attempt-test"
            staging.mkdir()
            write_json(
                staging / "config.json",
                {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]},
            )
            write_json(
                staging / "tokenizer_config.json",
                {"tokenizer_class": "Qwen2TokenizerFast"},
            )
            write_json(
                staging / "tokenizer.json",
                {"version": "1.0", "model": {"type": "BPE"}},
            )
            first_shard = "model-00001-of-00002.safetensors"
            second_shard = "model-00002-of-00002.safetensors"
            write_safetensors(staging / first_shard, "model.layer.0.weight")
            write_safetensors(staging / second_shard, "model.layer.1.weight")
            write_json(
                staging / "model.safetensors.index.json",
                {
                    "metadata": {"total_size": 4},
                    "weight_map": {
                        "model.layer.0.weight": first_shard,
                        "model.layer.1.weight": second_shard,
                    },
                },
            )
            published = run_root / "exports/early-merged"
            plugin.seal_merged_export(
                staging,
                published_dir=published,
                run_root=run_root,
                candidate="early",
            )
            staging.rename(published)
            plugin.verify_merged_export(
                published, run_root=run_root, expected_candidate="early"
            )
            with (published / first_shard).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(plugin.Day20PluginError):
                plugin.verify_merged_export(
                    published, run_root=run_root, expected_candidate="early"
                )

    def test_five_step_gate_allows_warmup_zero_then_requires_updates(self) -> None:
        rows = [
            {
                "global_step": step,
                "loss": 2.0 / step,
                "grad_norm": 1.0,
                "learning_rate": 1e-5,
                "lora_update_ratio": step * 1e-7,
                "cumulative_supervised_tokens": step * 100,
            }
            for step in range(1, 6)
        ]
        rows[0]["lora_update_ratio"] = 0.0
        result = plugin.validate_five_step_metrics(rows)
        self.assertEqual(result["status"], "pass")
        broken = [dict(row) for row in rows]
        broken[2]["lora_update_ratio"] = 0.0
        with self.assertRaises(plugin.Day20PluginError):
            plugin.validate_five_step_metrics(broken)

    def test_trainable_inventory_is_lora_only_and_language_scoped(self) -> None:
        names = [
            "base_model.model.language_model.layers.0.self_attn.q_proj."
            "lora_A.default.weight",
            "base_model.model.language_model.layers.0.self_attn.q_proj."
            "lora_B.default.weight",
            "base_model.model.language_model.layers.1.linear_attn.in_proj_qkv."
            "lora_A.default.weight",
            "base_model.model.language_model.layers.1.linear_attn.in_proj_qkv."
            "lora_B.default.weight",
        ]
        inventory = plugin.validate_trainable_names(
            names, contract.LORA_TARGET_REGEX
        )
        self.assertEqual(inventory["trainable_parameter_tensors"], 4)
        with self.assertRaises(plugin.Day20PluginError):
            plugin.validate_trainable_names(
                [
                    "base_model.model.visual.layers.0.q_proj."
                    "lora_A.default.weight"
                ],
                contract.LORA_TARGET_REGEX,
            )

    def test_validate_prepared_probe_binds_manifest_and_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "probe.jsonl"
            write_jsonl(
                dataset,
                [
                    {
                        "sample_id": f"sample-{index}",
                        "qwen35_supervised_tokens": 400,
                    }
                    for index in range(40)
                ],
            )
            immutable = contract.immutable_experiment_contract("/model/snapshot")
            probe_config = contract.build_probe_config(
                model_snapshot="/model/snapshot",
                dataset_path=str(dataset),
                dataset_sha256=plugin.file_sha256(dataset),
                learning_rate=3e-5,
            )
            probe_config_path = root / "probe-lr-3e-5.json"
            write_json(probe_config_path, probe_config)
            manifest = {
                "schema_version": 1,
                "contract": {
                    "immutable": immutable,
                    "immutable_sha256": plugin.object_sha256(immutable),
                },
                "datasets": {
                    "probe": {
                        "path": str(dataset),
                        "file_sha256": plugin.file_sha256(dataset),
                        "supervised_tokens": 16000,
                    }
                },
                "configs": {
                    "probes": {
                        "3e-5": {
                            "path": str(probe_config_path),
                            "file_sha256": plugin.file_sha256(probe_config_path),
                        }
                    }
                },
            }
            manifest["manifest_sha256"] = plugin.object_sha256(manifest)
            manifest_path = root / "DAY20-MANIFEST.json"
            write_json(manifest_path, manifest)
            result = plugin.validate_prepared_inputs(
                manifest_path, dataset_kind="probe", learning_rate="3e-5"
            )
            self.assertEqual(result["expected_supervised_tokens"], 16000)
            self.assertEqual(result["checkpoint_steps"], {"probe": 5})

    def test_main_config_rejects_a_self_signed_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "main.jsonl"
            write_jsonl(dataset, [{"qwen35_supervised_tokens": 256000}])
            template = contract.build_main_config(
                model_snapshot="/model/snapshot",
                dataset_path=str(dataset),
                dataset_sha256=plugin.file_sha256(dataset),
            )
            template_path = root / "main-template.json"
            write_json(template_path, template)
            manifest = {
                "configs": {
                    "main_template": {
                        "path": str(template_path),
                        "file_sha256": plugin.file_sha256(template_path),
                    }
                }
            }
            manifest["manifest_sha256"] = plugin.object_sha256(manifest)
            manifest_path = root / "DAY20-MANIFEST.json"
            write_json(manifest_path, manifest)
            selection = {"status": "selected", "selected_lr": "1e-5"}
            selection["selection_sha256"] = plugin.object_sha256(selection)
            selection_path = root / "PROBE-SELECTION.json"
            write_json(selection_path, selection)
            with self.assertRaises(plugin.Day20PluginError):
                plugin.resolve_main_config(manifest_path, selection_path)


class ProbeSelectionTests(unittest.TestCase):
    def _write_candidate(
        self,
        root: Path,
        candidate: str,
        ordered: list[dict[str, Any]],
        scores: tuple[int, int, int, int],
        *,
        code_eligible: int = 7,
    ) -> tuple[Path, Path]:
        general, math, finance, code = scores
        skill_scores = {"general": general, "math": math, "finance": finance}
        comparison = plugin.object_sha256({"candidate": candidate, "kind": "comparison"})
        run_hash = plugin.object_sha256({"candidate": candidate, "kind": "run"})
        model_identity = plugin.object_sha256({"candidate": candidate, "kind": "model"})
        predictions_path = root / f"{candidate}.predictions.jsonl"
        observed = {skill: 0 for skill in plugin.SKILLS}
        prediction_rows: list[dict[str, Any]] = []
        for ordinal, source_row in enumerate(ordered, 1):
            skill = source_row["slice"]
            index = observed[skill]
            observed[skill] += 1
            eligible = skill == "code" and index < code_eligible
            scorer_result = {
                "parse_status": "ok",
                "score_status": (
                    "sandbox_required" if skill == "code" else "scored"
                ),
                "score": (
                    None if skill == "code" else int(index < skill_scores[skill])
                ),
            }
            row = {
                "schema_version": 2,
                "domain": "day20.qwen35_adapter_prediction",
                "ordinal": ordinal,
                "candidate": candidate,
                "recipe": candidate,
                "sample_id": source_row["sample_id"],
                "slice": skill,
                "comparison_key": comparison,
                "run_hash": run_hash,
                "normalized_output": "answer",
                "normalized_scorer_result": scorer_result,
                "code_candidate": (
                    {
                        "candidate_mode": "solution",
                        "execution_eligible": eligible,
                    }
                    if skill == "code"
                    else None
                ),
                "code_static_syntax": (
                    {"valid": True} if skill == "code" else None
                ),
                "sandbox_execution_eligible": (
                    eligible if skill == "code" else None
                ),
                "format_compliant": eligible if skill == "code" else True,
                "anomalies": [],
            }
            row["row_sha256"] = plugin.object_sha256(row)
            prediction_rows.append(row)
        write_jsonl(predictions_path, prediction_rows)
        summary = {
            "schema_version": 1,
            "domain": "day20.qwen35_adapter_summary",
            "status": "complete_with_code_sandbox_required",
            "candidate": candidate,
            "recipe": candidate,
            "comparison_key": comparison,
            "run_hash": run_hash,
            "model_identity_sha256": model_identity,
            "metrics": plugin._recompute_probe_adapter_metrics(
                prediction_rows, candidate=candidate
            ),
            "predictions": {
                "path": str(predictions_path.resolve()),
                "records": 32,
                "file_sha256": plugin.file_sha256(predictions_path),
            },
        }
        summary["summary_sha256"] = plugin.object_sha256(summary)
        summary_path = root / f"{candidate}.json"
        write_json(summary_path, summary)

        result_path = root / f"{candidate}-e2b.jsonl"
        code_ids = [row["sample_id"] for row in ordered if row["slice"] == "code"]
        e2b_comparison = plugin.object_sha256(
            {"candidate": candidate, "kind": "e2b-comparison"}
        )
        complete_comparison = plugin.object_sha256(
            {"candidate": candidate, "kind": "complete-comparison"}
        )
        e2b_run = plugin.object_sha256({"candidate": candidate, "kind": "e2b-run"})
        code_run = plugin.object_sha256({"candidate": candidate, "kind": "code-run"})
        results: list[dict[str, Any]] = []
        for index, sample_id in enumerate(code_ids):
            score = float(index < code)
            execution_status = (
                "rejected_candidate_contract"
                if index >= code_eligible
                else ("passed" if score == 1.0 else "failed")
            )
            row = {
                "schema_version": 1,
                "domain": "day20.qwen35_adapter_code_sandbox_result",
                "code_ordinal": index,
                "prediction_run_ordinal": prediction_rows[
                    next(
                        position
                        for position, prediction in enumerate(prediction_rows)
                        if prediction["sample_id"] == sample_id
                    )
                ]["ordinal"] - 1,
                "sample_id": sample_id,
                "slice": "code",
                "candidate": candidate,
                "recipe": candidate,
                "score_status": "ok",
                "score": score,
                "passed": score == 1.0,
                "execution_status": execution_status,
                "error_type": None,
                "scorer_result": {
                    "score_status": "ok",
                    "score": score,
                    "passed": score == 1.0,
                    "execution_outcome": execution_status,
                },
                "candidate_mode": "solution",
                "candidate_executed_in_sandbox": index < code_eligible,
                "candidate_code_executed_on_host": False,
                "prediction_run_hash": run_hash,
                "prediction_comparison_key": comparison,
                "model_snapshot_hash": model_identity,
                "run_hash": e2b_run,
                "comparison_key": e2b_comparison,
                "complete_comparison_key": complete_comparison,
                "code_run_hash": code_run,
            }
            results.append(row)
        write_jsonl(result_path, results)
        e2b = {
            "schema_version": 1,
            "domain": "day20.qwen35_adapter_code_e2b_summary",
            "status": "complete",
            "candidate": candidate,
            "recipe": candidate,
            "records": 8,
            "passed": code,
            "failed": 8 - code,
            "sandbox_execution_eligible": code_eligible,
            "candidate_modes": {"solution": 8},
            "outcomes": dict(
                sorted(
                    {
                        status: sum(
                            row["execution_status"] == status for row in results
                        )
                        for status in {row["execution_status"] for row in results}
                    }.items()
                )
            ),
            "error_types": {"None": 8},
            "result_path": str(result_path.resolve()),
            "result_file_sha256": plugin.file_sha256(result_path),
            "prediction_path": str(predictions_path.resolve()),
            "prediction_file_sha256": plugin.file_sha256(predictions_path),
            "prediction_run_hash": run_hash,
            "prediction_comparison_key": comparison,
            "model_identity_sha256": model_identity,
            "comparison_key": e2b_comparison,
            "complete_comparison_key": complete_comparison,
            "run_hash": e2b_run,
            "code_run_hash": code_run,
        }
        e2b["summary_sha256"] = plugin.object_sha256(e2b)
        e2b_path = root / f"{candidate}-e2b-summary.json"
        write_json(e2b_path, e2b)
        return summary_path, e2b_path

    def test_probe_selection_consumes_manifest_diagnostic_and_tiebreaks_low_lr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ordered = [
                {"sample_id": f"{skill}-{index}", "slice": skill}
                for index in range(8)
                for skill in plugin.SKILLS
            ]
            diagnostic_path = root / "diagnostic.jsonl"
            write_jsonl(diagnostic_path, ordered)
            main_dataset_path = root / "main.jsonl"
            write_jsonl(main_dataset_path, [{"qwen35_supervised_tokens": 256000}])
            main_template = contract.build_main_config(
                model_snapshot="/model/snapshot",
                dataset_path=str(main_dataset_path),
                dataset_sha256=plugin.file_sha256(main_dataset_path),
            )
            main_template_path = root / "main-template.json"
            write_json(main_template_path, main_template)
            day20_manifest = {
                "datasets": {
                    "diagnostic": {
                        "path": str(diagnostic_path),
                        "file_sha256": plugin.file_sha256(diagnostic_path),
                        "records": 32,
                        "ordered_sample_ids": [row["sample_id"] for row in ordered],
                        "selection_sha256": plugin.object_sha256(
                            [row["sample_id"] for row in ordered]
                        ),
                    }
                },
                "configs": {
                    "main_template": {
                        "path": str(main_template_path),
                        "file_sha256": plugin.file_sha256(main_template_path),
                    }
                },
            }
            day20_manifest["manifest_sha256"] = plugin.object_sha256(
                day20_manifest
            )
            day20_manifest_path = root / "DAY20-MANIFEST.json"
            write_json(day20_manifest_path, day20_manifest)
            eval_manifest = {
                "records": [
                    {
                        **row,
                        "evaluation_split": "dev",
                    }
                    for row in ordered
                ],
                "header": {"evaluation_order": {"dev": [row["sample_id"] for row in ordered] + [f"unused-{i}" for i in range(80)]}},
            }
            eval_manifest_path = root / "eval.json"
            write_json(eval_manifest_path, eval_manifest)
            base = self._write_candidate(
                root, "base-probe", ordered, (2, 6, 4, 5), code_eligible=8
            )
            low = self._write_candidate(
                root, "probe-1e-5", ordered, (5, 5, 4, 5)
            )
            middle = self._write_candidate(
                root, "probe-3e-5", ordered, (5, 5, 4, 5)
            )
            high = self._write_candidate(
                root, "probe-1e-4", ordered, (4, 5, 4, 5)
            )
            result = plugin.select_probe(
                day20_manifest=day20_manifest_path,
                eval_manifest=eval_manifest_path,
                base_summary=base[0],
                base_e2b_summary=base[1],
                probes=[
                    ("1e-5", low[0], low[1]),
                    ("3e-5", middle[0], middle[1]),
                    ("1e-4", high[0], high[1]),
                ],
            )
            self.assertEqual(result["status"], "selected")
            self.assertEqual(result["selected_lr"], "1e-5")
            self.assertEqual(
                result["selection_sha256"],
                plugin.object_sha256(
                    {
                        key: value
                        for key, value in result.items()
                        if key != "selection_sha256"
                    }
                ),
            )
            selection_path = root / "PROBE-SELECTION.json"
            write_json(selection_path, result)
            verified = plugin.validate_probe_selection(
                day20_manifest_path, selection_path
            )
            self.assertEqual(verified, result)
            resolved = plugin.resolve_main_config(
                day20_manifest_path, selection_path
            )
            self.assertEqual(resolved["training"]["learning_rate"], 1e-5)
            self.assertTrue(plugin._self_hash_valid(resolved, "immutable_sha256"))

            tampered = json.loads(selection_path.read_text(encoding="utf-8"))
            tampered["selected_lr"] = "3e-5"
            tampered["selection_sha256"] = plugin.object_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "selection_sha256"
                }
            )
            write_json(root / "tampered-selection.json", tampered)
            with self.assertRaises(plugin.Day20PluginError):
                plugin.validate_probe_selection(
                    day20_manifest_path, root / "tampered-selection.json"
                )

            resigned_adapter = plugin.load_json(low[0])
            resigned_adapter["metrics"]["by_slice"]["general"]["correct"] = 8.0
            resigned_adapter["summary_sha256"] = plugin.object_sha256(
                {
                    key: value
                    for key, value in resigned_adapter.items()
                    if key != "summary_sha256"
                }
            )
            write_json(low[0], resigned_adapter)
            with self.assertRaises(plugin.Day20PluginError):
                plugin.select_probe(
                    day20_manifest=day20_manifest_path,
                    eval_manifest=eval_manifest_path,
                    base_summary=base[0],
                    base_e2b_summary=base[1],
                    probes=[
                        ("1e-5", low[0], low[1]),
                        ("3e-5", middle[0], middle[1]),
                        ("1e-4", high[0], high[1]),
                    ],
                )

            original_e2b = plugin.load_json(middle[1])
            resigned_e2b = json.loads(json.dumps(original_e2b))
            resigned_e2b["passed"] += 1
            resigned_e2b["failed"] -= 1
            resigned_e2b["summary_sha256"] = plugin.object_sha256(
                {
                    key: value
                    for key, value in resigned_e2b.items()
                    if key != "summary_sha256"
                }
            )
            write_json(middle[1], resigned_e2b)
            with self.assertRaises(plugin.Day20PluginError):
                plugin._probe_metrics(
                    candidate="probe-3e-5",
                    summary_path=middle[0],
                    e2b_summary_path=middle[1],
                    expected_ids=[row["sample_id"] for row in ordered],
                    expected_slices=[row["slice"] for row in ordered],
                )


if __name__ == "__main__":
    unittest.main()
