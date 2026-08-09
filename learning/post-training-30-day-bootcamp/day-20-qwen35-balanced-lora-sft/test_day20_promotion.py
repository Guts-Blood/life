#!/usr/bin/env python3
"""Synthetic promotion tests for the Day 20 finalizer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock


HERE = Path(__file__).resolve().parent
LORA_TARGET_REGEX = (
    r"^(?:(?:base_model|model)\.)*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def load_finalizer() -> ModuleType:
    path = HERE / "finalize_day20.py"
    spec = importlib.util.spec_from_file_location("day20_test_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finalizer = load_finalizer()


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def semantic_hash(label: str) -> str:
    return "sha256:" + sha(label)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def split_format_count(total: int) -> list[int]:
    quotient, remainder = divmod(total, 4)
    return [quotient + int(index < remainder) for index in range(4)]


class SyntheticRun:
    def __init__(
        self,
        root: Path,
        metrics: dict[str, tuple[int, int, int, int]],
        *,
        format_compliant: int = 108,
        code_eligible: int = 27,
    ) -> None:
        self.root = root
        self.eval_dir = root / "eval"
        self.eval_dir.mkdir(parents=True)
        (root / ".day20-run-root").write_text("day20\n", encoding="utf-8")
        main_dataset_path = root / "data" / "main.jsonl"
        main_dataset_path.parent.mkdir(parents=True)
        token_counts = []
        for batch_index in range(20):
            first = 1601 if batch_index < 19 else 1581
            token_counts.extend([first, *([1600] * 7)])
        write_jsonl(
            main_dataset_path,
            [
                {"sample_id": f"train-{ordinal}", "qwen35_supervised_tokens": tokens}
                for ordinal, tokens in enumerate(token_counts, 1)
            ],
        )
        checkpoints = self.root / "checkpoints"
        for candidate in finalizer.CANDIDATES:
            checkpoint = checkpoints / candidate
            checkpoint.mkdir(parents=True)
            if candidate == "base":
                (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
                (checkpoint / "model.safetensors").write_bytes(b"base-weights")
            else:
                (checkpoint / "adapter_config.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (checkpoint / "adapter_model.safetensors").write_bytes(
                    f"{candidate}-adapter-weights".encode("utf-8")
                )
                (checkpoint / "optimizer.pt").write_bytes(
                    f"{candidate}-optimizer".encode("utf-8")
                )
        self.checkpoint_steps = {"early": 5, "mid": 12, "final": 20}
        self.checkpoint_actual_tokens = {
            "early": 64_005,
            "mid": 153_612,
            "final": 256_000,
        }
        immutable = {
            "name": "synthetic-day20-contract",
            "model": {
                "snapshot": str((root / "checkpoints" / "base").resolve())
            },
        }
        manifest = {
            "schema_version": 1,
            "domain": "day20.qwen35_balanced_lora.experiment_manifest",
            "status": "prepared_probes_not_started",
            "run_root": str(root.resolve()),
            "contract": {
                "immutable": immutable,
                "immutable_sha256": finalizer.object_sha256(immutable),
            },
            "datasets": {
                "main": {
                    "path": str(main_dataset_path.resolve()),
                    "file_sha256": finalizer.file_sha256(main_dataset_path),
                }
            },
            "base_model_identity": {
                "path": str((checkpoints / "base").resolve()),
                "files": finalizer.directory_file_manifest(checkpoints / "base", {}),
                "snapshot_sha256": finalizer.object_sha256(
                    finalizer.directory_file_manifest(checkpoints / "base", {})
                ),
            },
        }
        manifest["manifest_sha256"] = finalizer.object_sha256(manifest)
        self.manifest_path = root / "DAY20-MANIFEST.json"
        write_json(self.manifest_path, manifest)
        self.manifest_file_sha = finalizer.file_sha256(self.manifest_path)
        self.manifest_content_sha = manifest["manifest_sha256"]
        config_path = self.root / "configs" / "main-selected.json"
        config_path.parent.mkdir(parents=True)
        write_json(
            config_path,
            {
                "training": {
                    "learning_rate": 3e-5,
                    "target_regex": LORA_TARGET_REGEX,
                }
            },
        )
        self.training_config = {
            "path": str(config_path.resolve()),
            "file_sha256": finalizer.file_sha256(config_path),
        }
        checkpoint_packages = {}
        for step, candidate in enumerate(finalizer.PROMOTION_CANDIDATES, 1):
            checkpoint = checkpoints / candidate
            files = finalizer.directory_file_manifest(checkpoint, {})
            integrity = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_checkpoint_integrity",
                "status": "complete",
                "checkpoint": str(checkpoint.resolve()),
                "run_kind": "main",
                "global_step": self.checkpoint_steps[candidate],
                "cumulative_supervised_tokens": self.checkpoint_actual_tokens[
                    candidate
                ],
                "targets": [candidate],
                "resumable": True,
                "dataset_file_sha256": finalizer.file_sha256(main_dataset_path),
                "training_config_file_sha256": self.training_config[
                    "file_sha256"
                ],
                "files": files,
                "snapshot_sha256": finalizer.object_sha256(files),
            }
            integrity["integrity_sha256"] = finalizer.object_sha256(integrity)
            integrity_path = checkpoint / finalizer.CHECKPOINT_INTEGRITY_FILE
            write_json(integrity_path, integrity)
            checkpoint_packages[candidate] = {
                "path": str(checkpoint.resolve()),
                "integrity_file_sha256": finalizer.file_sha256(integrity_path),
                "integrity_sha256": integrity["integrity_sha256"],
                "snapshot_sha256": integrity["snapshot_sha256"],
                "files": files,
                "resumable": True,
            }
        evidence_dir = self.root / "evidence" / "main" / "attempt-001"
        evidence_dir.mkdir(parents=True)
        self.evidence_paths = {
            "trainable_inventory": evidence_dir / "trainable-inventory.json",
            "trainer_token_reaudit": evidence_dir / "trainer-token-reaudit.json",
            "step_metrics": evidence_dir / "step-metrics.jsonl",
            "five_step_safety": evidence_dir / "five-step-safety.json",
        }
        runtime_identity = {
            "schema_version": 1,
            "domain": "day20.qwen35_lora_runtime_identity",
            "backend": "hf_transformers",
            "versions": dict(finalizer.EXPECTED_RUNTIME_VERSIONS),
            "ms_swift_root": "/root/autodl-tmp/ms-swift",
            "ms_swift_commit": finalizer.EXPECTED_MS_SWIFT_COMMIT,
            "ms_swift_worktree_clean": True,
            "implementation_file_sha256": {
                name: finalizer.file_sha256(HERE / name)
                for name in finalizer.RUNTIME_IMPLEMENTATION_FILES
            },
        }
        runtime_identity["runtime_sha256"] = finalizer.object_sha256(
            runtime_identity
        )
        modules = []
        for layer in range(8):
            modules.extend(
                f"base_model.model.language_model.layers.{layer}.self_attn.{leaf}"
                for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        for layer in range(8, 32):
            modules.extend(
                f"base_model.model.language_model.layers.{layer}.linear_attn.{leaf}"
                for leaf in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                )
            )
        for layer in range(32):
            modules.extend(
                f"base_model.model.language_model.layers.{layer}.mlp.{leaf}"
                for leaf in ("gate_proj", "up_proj", "down_proj")
            )
        trainable_names = sorted(
            f"{module}.lora_{side}.default.weight"
            for module in modules
            for side in ("A", "B")
        )
        inventory = {
            "schema_version": 1,
            "domain": "day20.qwen35_lora_trainable_inventory",
            "status": "pass",
            "target_regex": LORA_TARGET_REGEX,
            "trainable_parameter_tensors": 496,
            "target_module_count": 248,
            "target_module_counts": dict(
                sorted(finalizer.EXPECTED_LORA_MODULE_COUNTS.items())
            ),
            "trainable_names": trainable_names,
        }
        inventory["inventory_sha256"] = finalizer.object_sha256(inventory)
        write_json(self.evidence_paths["trainable_inventory"], inventory)
        token_reaudit = {
            "schema_version": 1,
            "domain": "day20.qwen35_trainer_token_reaudit",
            "status": "pass",
            "dataset": str(main_dataset_path.resolve()),
            "dataset_file_sha256": finalizer.file_sha256(main_dataset_path),
            "records": len(token_counts),
            "supervised_tokens": 256_000,
            "ordered_evidence_sha256": sha("ordered-live-token-evidence"),
        }
        token_reaudit["audit_sha256"] = finalizer.object_sha256(token_reaudit)
        write_json(self.evidence_paths["trainer_token_reaudit"], token_reaudit)
        step_schedule = [
            sum(token_counts[offset : offset + 8])
            for offset in range(0, len(token_counts), 8)
        ]
        self.step_rows = []
        cumulative = 0
        for step, window_tokens in enumerate(step_schedule, 1):
            cumulative += window_tokens
            self.step_rows.append(
                {
                    "schema_version": 1,
                    "domain": "day20.qwen35_lora_optimizer_step",
                    "global_step": step,
                    "window_supervised_tokens": window_tokens,
                    "cumulative_supervised_tokens": cumulative,
                    "loss": 2.0 + step / 100,
                    "grad_norm": 1.0 + step / 100,
                    "learning_rate": 3e-5,
                    "lora_update_ratio": step * 1e-6 if step <= 5 else None,
                }
            )
        write_jsonl(self.evidence_paths["step_metrics"], self.step_rows)
        first_five = self.step_rows[:5]
        five_step_safety = {
            "schema_version": 1,
            "domain": "day20.qwen35_lora_five_step_safety",
            "status": "pass",
            "steps": 5,
            "cumulative_supervised_tokens": first_five[-1][
                "cumulative_supervised_tokens"
            ],
            "maximum_loss": max(float(row["loss"]) for row in first_five),
            "maximum_grad_norm": max(
                float(row["grad_norm"]) for row in first_five
            ),
            "final_lora_update_ratio": float(
                first_five[-1]["lora_update_ratio"]
            ),
        }
        five_step_safety["summary_sha256"] = finalizer.object_sha256(
            five_step_safety
        )
        write_json(self.evidence_paths["five_step_safety"], five_step_safety)
        training_summary = {
            "schema_version": 1,
            "domain": "day20.qwen35_lora_training_summary",
            "status": "pass",
            "run_kind": "main",
            "learning_rate": 3e-5,
            "runtime_identity": runtime_identity,
            "training_config": self.training_config,
            "dataset": str(main_dataset_path.resolve()),
            "dataset_file_sha256": finalizer.file_sha256(main_dataset_path),
            "optimizer_steps": len(step_schedule),
            "cumulative_supervised_tokens": 256_000,
            "checkpoint_tokens": {
                candidate: finalizer.CHECKPOINT_TOKENS[candidate]
                for candidate in finalizer.PROMOTION_CANDIDATES
            },
            "checkpoint_actual_tokens": {
                candidate: self.checkpoint_actual_tokens[candidate]
                for candidate in finalizer.PROMOTION_CANDIDATES
            },
            "checkpoint_steps": self.checkpoint_steps,
            "checkpoints": {
                candidate: str((checkpoints / candidate).resolve())
                for candidate in finalizer.PROMOTION_CANDIDATES
            },
            "checkpoint_packages": checkpoint_packages,
            "trainable_inventory": {
                "path": str(self.evidence_paths["trainable_inventory"].resolve()),
                "file_sha256": finalizer.file_sha256(
                    self.evidence_paths["trainable_inventory"]
                ),
            },
            "trainer_token_reaudit": {
                "path": str(
                    self.evidence_paths["trainer_token_reaudit"].resolve()
                ),
                "file_sha256": finalizer.file_sha256(
                    self.evidence_paths["trainer_token_reaudit"]
                ),
                "content_sha256": token_reaudit["audit_sha256"],
            },
            "step_metrics": {
                "path": str(self.evidence_paths["step_metrics"].resolve()),
                "file_sha256": finalizer.file_sha256(
                    self.evidence_paths["step_metrics"]
                ),
                "records": len(self.step_rows),
            },
            "five_step_safety": {
                "path": str(self.evidence_paths["five_step_safety"].resolve()),
                "file_sha256": finalizer.file_sha256(
                    self.evidence_paths["five_step_safety"]
                ),
            },
        }
        training_summary["summary_sha256"] = finalizer.object_sha256(
            training_summary
        )
        self.training_summary = training_summary
        self.training_summary_path = (
            self.root / "evidence" / "main" / "training-summary.json"
        )
        self.training_summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.training_summary_path, training_summary)
        self.training_summary_hash = finalizer.file_sha256(
            self.training_summary_path
        )
        self.training_summary_content_hash = training_summary["summary_sha256"]
        format_counts = split_format_count(format_compliant)
        for candidate in finalizer.CANDIDATES:
            self._write_candidate(
                candidate,
                metrics[candidate],
                format_counts=format_counts,
                code_eligible=code_eligible,
            )

    def resign_training_evidence(self, name: str) -> None:
        identity = self.training_summary[name]
        path = self.evidence_paths[name]
        identity["file_sha256"] = finalizer.file_sha256(path)
        if name == "trainer_token_reaudit":
            identity["content_sha256"] = finalizer.load_json(path)["audit_sha256"]
        self.resign_training_summary()

    def resign_training_summary(self) -> None:
        self.training_summary["summary_sha256"] = finalizer.object_sha256(
            {
                key: value
                for key, value in self.training_summary.items()
                if key != "summary_sha256"
            }
        )
        write_json(self.training_summary_path, self.training_summary)
        self.training_summary_hash = finalizer.file_sha256(
            self.training_summary_path
        )
        self.training_summary_content_hash = self.training_summary[
            "summary_sha256"
        ]
        for candidate in finalizer.PROMOTION_CANDIDATES:
            path = self.eval_dir / f"{candidate}.qwen35-v2.json"
            if not path.is_file():
                continue
            adapter = finalizer.load_json(path)
            adapter["training_identity"][
                "file_sha256"
            ] = self.training_summary_hash
            adapter["training_identity"][
                "content_sha256"
            ] = self.training_summary_content_hash
            adapter["summary_sha256"] = finalizer.object_sha256(
                {
                    key: value
                    for key, value in adapter.items()
                    if key != "summary_sha256"
                }
            )
            write_json(path, adapter)

    def _write_candidate(
        self,
        candidate: str,
        metric_counts: tuple[int, int, int, int],
        *,
        format_counts: list[int],
        code_eligible: int,
    ) -> None:
        general, math, finance, code = metric_counts
        checkpoint = self.root / "checkpoints" / candidate
        base_checkpoint = self.root / "checkpoints" / "base"
        predictions_path = self.eval_dir / f"{candidate}.qwen35-v2.predictions.jsonl"
        adapter_path = self.eval_dir / f"{candidate}.qwen35-v2.json"
        result_path = self.eval_dir / f"{candidate}-code-e2b-qwen35-v2.jsonl"
        e2b_path = self.eval_dir / f"{candidate}-code-e2b-qwen35-v2-summary.json"
        adapter_comparison = sha("shared-adapter-comparison")
        adapter_run = sha(f"{candidate}-adapter-run")
        correct_by_slice = {
            "general": general,
            "math": math,
            "code": 0,
            "finance": finance,
        }
        format_by_slice = dict(
            zip(("general", "math", "code", "finance"), format_counts)
        )
        prediction_rows = []
        ordinal = 0
        for skill in ("general", "math", "code", "finance"):
            for skill_ordinal in range(1, 29):
                ordinal += 1
                row = {
                    "schema_version": 2,
                    "domain": "day20.qwen35_adapter_prediction",
                    "ordinal": ordinal,
                    "candidate": candidate,
                    "recipe": candidate,
                    "sample_id": f"{skill}-{skill_ordinal}",
                    "slice": skill,
                    "comparison_key": adapter_comparison,
                    "run_hash": adapter_run,
                    "normalized_scorer_result": {
                        "score": (
                            None
                            if skill == "code"
                            else float(skill_ordinal <= correct_by_slice[skill])
                        )
                    },
                    "sandbox_execution_eligible": (
                        skill_ordinal <= code_eligible if skill == "code" else None
                    ),
                    "format_compliant": skill_ordinal <= format_by_slice[skill],
                }
                row["row_sha256"] = finalizer.object_sha256(row)
                prediction_rows.append(row)
        write_jsonl(predictions_path, prediction_rows)
        base_files = finalizer.directory_file_manifest(base_checkpoint, {})
        base_snapshot = finalizer.object_sha256(base_files)
        adapter_files = None
        adapter_identity = None
        adapter_snapshot = None
        if candidate != "base":
            package_files = finalizer.directory_file_manifest(checkpoint, {})
            adapter_files = finalizer.adapter_inference_file_manifest(checkpoint, {})
            adapter_snapshot = finalizer.object_sha256(adapter_files)
            adapter_identity = {
                "path": str(checkpoint.resolve()),
                "format": "peft_lora",
                "files": adapter_files,
                "snapshot_sha256": adapter_snapshot,
                "checkpoint_package_files": package_files,
                "checkpoint_package_snapshot_sha256": finalizer.object_sha256(
                    package_files
                ),
            }
        combined_snapshot = finalizer.object_sha256(
            {
                "domain": "day20.qwen35_base_adapter_identity",
                "schema_version": 1,
                "base_snapshot_sha256": base_snapshot,
                "adapter_snapshot_sha256": adapter_snapshot,
            }
        )
        model_identity = {
            "base": {
                "path": str(base_checkpoint.resolve()),
                "model_class": "Qwen3_5ForConditionalGeneration",
                "files": base_files,
                "snapshot_sha256": base_snapshot,
            },
            "adapter": adapter_identity,
            "combined_snapshot_sha256": combined_snapshot,
            "load_and_generation_verified": True,
        }
        training_identity = None
        if candidate != "base":
            training_identity = {
                "path": str(self.training_summary_path.resolve()),
                "file_sha256": self.training_summary_hash,
                "content_sha256": self.training_summary_content_hash,
                "run_kind": "main",
                "checkpoint_label": candidate,
                "checkpoint_target_supervised_tokens": finalizer.CHECKPOINT_TOKENS[
                    candidate
                ],
                "learning_rate": 3e-5,
                "training_config": self.training_config,
                "checkpoint_package": json.loads(
                    json.dumps(
                        self.training_summary["checkpoint_packages"][candidate]
                    )
                ),
            }
        by_slice = {
            "general": {
                "records": 28,
                "scored_records": 28,
                "correct": general,
                "format_compliant": format_counts[0],
            },
            "math": {
                "records": 28,
                "scored_records": 28,
                "correct": math,
                "format_compliant": format_counts[1],
            },
            "code": {
                "records": 28,
                "format_compliant": format_counts[2],
                "sandbox_execution_eligible": code_eligible,
                "executable_score_status": "sandbox_required",
            },
            "finance": {
                "records": 28,
                "scored_records": 28,
                "correct": finance,
                "format_compliant": format_counts[3],
            },
        }
        adapter = {
            "schema_version": 1,
            "domain": "day20.qwen35_adapter_summary",
            "status": "complete_with_code_sandbox_required",
            "candidate": candidate,
            "recipe": candidate,
            "checkpoint": str(checkpoint.resolve()),
            "model_identity": model_identity,
            "model_identity_sha256": combined_snapshot,
            "model_identity_post_run_unchanged": True,
            "training_identity": training_identity,
            "comparison_key": adapter_comparison,
            "run_hash": adapter_run,
            "protocol": {
                "manifest_file_sha256": sha("manifest"),
                "experiment_manifest_file_sha256": self.manifest_file_sha,
                "experiment_manifest_content_sha256": self.manifest_content_sha,
                "scorer_file_sha256": sha("scorer"),
                "response_adapter_file_sha256": sha("response-adapter"),
                "code_adapter_file_sha256": sha("code-adapter"),
                "rescorer_file_sha256": sha("rescorer"),
                "parent_rescorer_file_sha256": sha("parent-rescorer"),
                "legacy_scorer_semantics_preserved": True,
                "candidate_code_executed_on_host": False,
                "frozen_test_consumed": False,
                "source_evaluation_protocol": {
                    "records": 112,
                    "sample_limit": None,
                    "experiment_manifest_file_sha256": self.manifest_file_sha,
                    "experiment_manifest_content_sha256": self.manifest_content_sha,
                    "diagnostic_selection_sha256": None,
                    "template": "qwen3_5",
                    "backend": "hf_transformers",
                    "use_mcore_gdn": False,
                    "enable_thinking": False,
                    "add_non_thinking_prefix": True,
                    "response_capture": "ms-swift RequestConfig(return_details=True)",
                    "generated_token_ids_retained": True,
                },
            },
            "metrics": {
                "by_slice": by_slice,
                "non_code_correct": general + math + finance,
                "format_compliant": sum(format_counts),
            },
            "predictions": {
                "path": str(predictions_path.resolve()),
                "records": 112,
                "file_sha256": finalizer.file_sha256(predictions_path),
            },
        }
        adapter["summary_sha256"] = finalizer.object_sha256(adapter)
        write_json(adapter_path, adapter)

        result_rows = []
        for ordinal in range(1, 29):
            passed = ordinal <= code
            result_rows.append(
                {
                    "schema_version": 1,
                    "domain": "day20.qwen35_adapter_code_sandbox_result",
                    "candidate": candidate,
                    "recipe": candidate,
                    "sample_id": f"code-{ordinal}",
                    "candidate_mode": "completion",
                    "candidate_executed_in_sandbox": ordinal <= code_eligible,
                    "score": 1.0 if passed else 0.0,
                    "score_status": "ok",
                    "execution_status": "passed" if passed else "failed",
                    "error_type": None if passed else "assertion_failure",
                    "prediction_run_hash": adapter_run,
                    "prediction_comparison_key": adapter_comparison,
                    "normalized_prediction_file_sha256": finalizer.file_sha256(
                        predictions_path
                    ),
                    "candidate_code_executed_on_host": False,
                    "sandbox_contract_hash": semantic_hash("effective-protocol"),
                    "code_execution_protocol_hash": semantic_hash("effective-protocol"),
                    "effective_code_protocol_hash": semantic_hash("effective-protocol"),
                    "base_day10_sandbox_contract_hash": semantic_hash(
                        "sandbox-contract"
                    ),
                }
            )
        write_jsonl(result_path, result_rows)
        outcomes = Counter(row["execution_status"] for row in result_rows)
        errors = Counter(str(row["error_type"]) for row in result_rows)
        e2b = {
            "schema_version": 1,
            "domain": "day20.qwen35_adapter_code_e2b_summary",
            "status": "complete",
            "candidate": candidate,
            "recipe": candidate,
            "records": 28,
            "passed": code,
            "failed": 28 - code,
            "sandbox_execution_eligible": code_eligible,
            "candidate_modes": {"completion": 28},
            "outcomes": dict(sorted(outcomes.items())),
            "error_types": dict(sorted(errors.items())),
            "result_path": str(result_path.resolve()),
            "result_file_sha256": finalizer.file_sha256(result_path),
            "prediction_path": str(predictions_path.resolve()),
            "prediction_file_sha256": finalizer.file_sha256(predictions_path),
            "prediction_run_hash": adapter_run,
            "prediction_comparison_key": adapter_comparison,
            "manifest_file_sha256": sha("manifest"),
            "experiment_manifest_file_sha256": self.manifest_file_sha,
            "experiment_manifest_content_sha256": self.manifest_content_sha,
            "frozen_e2b_scorer_file_sha256": sha("frozen-e2b-scorer"),
            "response_adapter_file_sha256": sha("response-adapter"),
            "code_adapter_file_sha256": sha("code-adapter"),
            "sandbox_wrapper_file_sha256": sha("sandbox-wrapper"),
            "base_sandbox_contract_file_sha256": sha("sandbox-contract-file"),
            "base_sandbox_contract_hash": semantic_hash("sandbox-contract"),
            "effective_code_protocol": {
                "parent_day10_sandbox_contract_hash": semantic_hash(
                    "sandbox-contract"
                )
            },
            "effective_code_protocol_hash": semantic_hash("effective-protocol"),
            "comparison_key": semantic_hash("e2b-comparison"),
            "complete_comparison_key": semantic_hash("complete-comparison"),
            "run_hash": semantic_hash(f"{candidate}-e2b-run"),
            "code_run_hash": semantic_hash(f"{candidate}-code-run"),
            "model_identity_sha256": combined_snapshot,
            "candidate_code_executed_on_host": False,
        }
        e2b["summary_sha256"] = finalizer.object_sha256(e2b)
        write_json(e2b_path, e2b)

BASE_METRICS = (1, 24, 15, 19)


class Day20PromotionTests(unittest.TestCase):
    def test_eligible_candidate_is_published_with_integrity_hashes(self) -> None:
        metrics = {
            "base": BASE_METRICS,
            "early": (7, 20, 20, 18),
            "mid": (8, 20, 20, 18),
            "final": (5, 17, 10, 14),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SyntheticRun(root, metrics)
            results = finalizer.finalize(root, created_at_utc="2026-08-09T00:00:00+00:00")
            self.assertEqual(results["decision"]["selected_candidate"], "mid")
            self.assertEqual(results["candidates"]["mid"]["metrics"]["total_correct"], 66)
            self.assertEqual(
                results["results_sha256"],
                finalizer.object_sha256(
                    {key: value for key, value in results.items() if key != "results_sha256"}
                ),
            )
            marker = finalizer.load_json(root / "DAY20-PASS.json")
            self.assertEqual(marker["selected_candidate"], "mid")
            self.assertEqual(
                marker["marker_sha256"],
                finalizer.object_sha256(
                    {key: value for key, value in marker.items() if key != "marker_sha256"}
                ),
            )
            self.assertEqual(finalizer.finalize(root), results)

    def test_eligible_publication_recovers_one_sided_crash_states(self) -> None:
        metrics = {
            "base": BASE_METRICS,
            "early": (7, 20, 20, 18),
            "mid": (8, 20, 20, 18),
            "final": (5, 17, 10, 14),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SyntheticRun(root, metrics)
            original_write = finalizer.atomic_write_json_new

            def interrupt_after_pass(path: Path, value: dict[str, Any]) -> None:
                original_write(path, value)
                if path.name == "DAY20-PASS.json":
                    raise RuntimeError("simulated interruption")

            with mock.patch.object(
                finalizer, "atomic_write_json_new", side_effect=interrupt_after_pass
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    finalizer.finalize(root, created_at_utc="2026-08-09T00:00:00+00:00")
            self.assertTrue((root / "DAY20-PASS.json").is_file())
            self.assertFalse((root / "DAY20-RESULTS.json").exists())
            recovered = finalizer.finalize(root)
            self.assertEqual(recovered["decision"]["selected_candidate"], "mid")
            self.assertTrue((root / "DAY20-RESULTS.json").is_file())

            (root / "DAY20-PASS.json").unlink()
            recovered_again = finalizer.finalize(root)
            self.assertEqual(recovered_again, recovered)
            self.assertTrue((root / "DAY20-PASS.json").is_file())

    def test_resigned_one_sided_finalization_tamper_is_rejected(self) -> None:
        metrics = {
            "base": BASE_METRICS,
            "early": (7, 20, 20, 18),
            "mid": (8, 20, 20, 18),
            "final": (5, 17, 10, 14),
        }
        for side in ("results", "pass"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                SyntheticRun(root, metrics)
                finalizer.finalize(root, created_at_utc="2026-08-09T00:00:00+00:00")
                results_path = root / "DAY20-RESULTS.json"
                pass_path = root / "DAY20-PASS.json"
                if side == "results":
                    pass_path.unlink()
                    value = finalizer.load_json(results_path)
                    value["decision"]["selected_candidate"] = "early"
                    value["results_sha256"] = finalizer.object_sha256(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "results_sha256"
                        }
                    )
                    write_json(results_path, value)
                else:
                    results_path.unlink()
                    value = finalizer.load_json(pass_path)
                    value["selected_candidate"] = "early"
                    value["marker_sha256"] = finalizer.object_sha256(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "marker_sha256"
                        }
                    )
                    write_json(pass_path, value)
                with self.assertRaises(finalizer.Day20FinalizationError):
                    finalizer.finalize(root)

    def test_tie_break_order(self) -> None:
        scenarios = (
            (
                "total",
                {"early": (10, 20, 20, 19), "mid": (5, 23, 25, 17)},
                "mid",
            ),
            (
                "general",
                {"early": (6, 24, 20, 20), "mid": (7, 24, 20, 19)},
                "mid",
            ),
            (
                "code",
                {"early": (7, 24, 21, 18), "mid": (7, 24, 20, 19)},
                "mid",
            ),
            (
                "earlier checkpoint",
                {"early": (7, 24, 20, 19), "mid": (7, 24, 20, 19)},
                "early",
            ),
        )
        for name, candidates, expected in scenarios:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                metrics = {
                    "base": BASE_METRICS,
                    "early": candidates["early"],
                    "mid": candidates["mid"],
                    "final": (5, 17, 10, 14),
                }
                root = Path(temporary)
                SyntheticRun(root, metrics)
                results = finalizer.finalize(root)
                self.assertEqual(results["decision"]["selected_candidate"], expected)

    def test_no_eligible_candidate_emits_results_without_pass(self) -> None:
        metrics = {
            "base": BASE_METRICS,
            "early": (5, 17, 10, 14),
            "mid": (6, 17, 10, 14),
            "final": (5, 18, 10, 14),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SyntheticRun(root, metrics)
            results = finalizer.finalize(root)
            self.assertEqual(
                results["decision"]["status"], "no_eligible_day20_lora_anchor"
            )
            self.assertIsNone(results["decision"]["selected_candidate"])
            self.assertTrue((root / "DAY20-RESULTS.json").is_file())
            self.assertFalse((root / "DAY20-PASS.json").exists())

    def test_resigned_training_evidence_semantic_tamper_is_rejected(self) -> None:
        metrics = {
            "base": BASE_METRICS,
            "early": (7, 20, 20, 18),
            "mid": (8, 20, 20, 18),
            "final": (5, 17, 10, 14),
        }
        for tamper in (
            "runtime_version",
            "runtime_implementation",
            "inventory_distribution",
            "token_reaudit_total",
            "step_schedule",
            "step_nonfinite",
            "five_step_gate",
        ):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = SyntheticRun(root, metrics)
                if tamper.startswith("runtime_"):
                    runtime = run.training_summary["runtime_identity"]
                    if tamper == "runtime_version":
                        runtime["versions"]["torch"] = "2.10.1+cu126"
                    else:
                        runtime["implementation_file_sha256"][
                            "day20_train_plugin.py"
                        ] = sha("replaced-training-plugin")
                    runtime["runtime_sha256"] = finalizer.object_sha256(
                        {
                            key: value
                            for key, value in runtime.items()
                            if key != "runtime_sha256"
                        }
                    )
                    run.resign_training_summary()
                elif tamper == "inventory_distribution":
                    path = run.evidence_paths["trainable_inventory"]
                    value = finalizer.load_json(path)
                    value["target_module_counts"]["q_proj"] = 7
                    value["inventory_sha256"] = finalizer.object_sha256(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "inventory_sha256"
                        }
                    )
                    write_json(path, value)
                    run.resign_training_evidence("trainable_inventory")
                elif tamper == "token_reaudit_total":
                    path = run.evidence_paths["trainer_token_reaudit"]
                    value = finalizer.load_json(path)
                    value["supervised_tokens"] = 255_999
                    value["audit_sha256"] = finalizer.object_sha256(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "audit_sha256"
                        }
                    )
                    write_json(path, value)
                    run.resign_training_evidence("trainer_token_reaudit")
                elif tamper in {"step_schedule", "step_nonfinite"}:
                    path = run.evidence_paths["step_metrics"]
                    rows = finalizer.load_jsonl(path)
                    if tamper == "step_schedule":
                        rows[6]["window_supervised_tokens"] += 1
                    else:
                        rows[6]["loss"] = float("nan")
                    write_jsonl(path, rows)
                    run.resign_training_evidence("step_metrics")
                else:
                    path = run.evidence_paths["five_step_safety"]
                    value = finalizer.load_json(path)
                    value["maximum_loss"] += 1.0
                    value["summary_sha256"] = finalizer.object_sha256(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "summary_sha256"
                        }
                    )
                    write_json(path, value)
                    run.resign_training_evidence("five_step_safety")
                with self.assertRaises(finalizer.Day20FinalizationError):
                    finalizer.finalize(root)
                self.assertFalse((root / "DAY20-RESULTS.json").exists())
                self.assertFalse((root / "DAY20-PASS.json").exists())

    def test_tampered_summary_or_prediction_is_rejected_without_output(self) -> None:
        metrics = {
            "base": BASE_METRICS,
            "early": (7, 20, 20, 18),
            "mid": (8, 20, 20, 18),
            "final": (5, 17, 10, 14),
        }
        for tamper in (
            "summary",
            "summary_rehashed",
            "predictions",
            "e2b_summary",
            "e2b_summary_rehashed",
            "e2b_result",
            "sandbox_hash_rehashed",
            "experiment_manifest",
            "training_summary",
            "base_checkpoint",
            "adapter_checkpoint",
        ):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                SyntheticRun(root, metrics)
                if tamper in ("summary", "summary_rehashed"):
                    path = root / "eval" / "early.qwen35-v2.json"
                    value = finalizer.load_json(path)
                    value["metrics"]["by_slice"]["general"]["correct"] += 1
                    if tamper == "summary_rehashed":
                        value["summary_sha256"] = finalizer.object_sha256(
                            {
                                key: item
                                for key, item in value.items()
                                if key != "summary_sha256"
                            }
                        )
                    write_json(path, value)
                else:
                    if tamper == "predictions":
                        path = root / "eval" / "mid.qwen35-v2.predictions.jsonl"
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write("{}\n")
                    elif tamper in ("e2b_summary", "e2b_summary_rehashed"):
                        path = root / "eval" / "early-code-e2b-qwen35-v2-summary.json"
                        value = finalizer.load_json(path)
                        value["passed"] += 1
                        if tamper == "e2b_summary_rehashed":
                            value["summary_sha256"] = finalizer.object_sha256(
                                {
                                    key: item
                                    for key, item in value.items()
                                    if key != "summary_sha256"
                                }
                            )
                        write_json(path, value)
                    elif tamper == "e2b_result":
                        path = root / "eval" / "mid-code-e2b-qwen35-v2.jsonl"
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write("{}\n")
                    elif tamper == "sandbox_hash_rehashed":
                        result_path = (
                            root / "eval" / "mid-code-e2b-qwen35-v2.jsonl"
                        )
                        rows = finalizer.load_jsonl(result_path)
                        rows[0]["sandbox_contract_hash"] = semantic_hash("wrong")
                        write_jsonl(result_path, rows)
                        summary_path = (
                            root
                            / "eval"
                            / "mid-code-e2b-qwen35-v2-summary.json"
                        )
                        value = finalizer.load_json(summary_path)
                        value["result_file_sha256"] = finalizer.file_sha256(
                            result_path
                        )
                        value["summary_sha256"] = finalizer.object_sha256(
                            {
                                key: item
                                for key, item in value.items()
                                if key != "summary_sha256"
                            }
                        )
                        write_json(summary_path, value)
                    elif tamper == "experiment_manifest":
                        with (root / "DAY20-MANIFEST.json").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write("\n")
                    elif tamper == "training_summary":
                        with (
                            root / "evidence" / "main" / "training-summary.json"
                        ).open("a", encoding="utf-8") as handle:
                            handle.write("\n")
                    elif tamper == "base_checkpoint":
                        with (root / "checkpoints" / "base" / "model.safetensors").open(
                            "ab"
                        ) as handle:
                            handle.write(b"tamper")
                    else:
                        with (
                            root
                            / "checkpoints"
                            / "early"
                            / "adapter_model.safetensors"
                        ).open("ab") as handle:
                            handle.write(b"tamper")
                with self.assertRaises(finalizer.Day20FinalizationError):
                    finalizer.finalize(root)
                self.assertFalse((root / "DAY20-RESULTS.json").exists())
                self.assertFalse((root / "DAY20-PASS.json").exists())


if __name__ == "__main__":
    unittest.main()
