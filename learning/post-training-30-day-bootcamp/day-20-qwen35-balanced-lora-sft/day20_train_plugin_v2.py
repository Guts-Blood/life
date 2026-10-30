#!/usr/bin/env python3
"""ms-swift callback and evidence helpers for Day 20 v2 training.

Normal import is stdlib-only.  The callback is registered only when
``DAY20_V2_REGISTER_SWIFT_CALLBACK=1``; its complete environment is one
self-hashed callback config path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from day20_candidate_factory_v2 import candidate_id
from day20_e2b_preflight_v2 import file_sha256 as e2b_file_sha256
from day20_e2b_preflight_v2 import verify_preflight
from day20_train_runtime_v2 import (
    CHECKPOINT_INTEGRITY_FILE,
    build_checkpoint_integrity,
    file_sha256,
    load_json,
    object_sha256,
    validate_inputs,
    verify_checkpoint_integrity,
    write_checkpoint_integrity,
)


CALLBACK_CONFIG_DOMAIN = "day20.qwen35_lora_callback_config.v2"
TRAINING_SUMMARY_DOMAIN = "day20.qwen35_lora_training_summary.v2"
EXPECTED_RUNTIME_VERSIONS = {
    "ms-swift": "4.5.0.dev0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
    "torch": "2.10.0+cu128",
}
EXPECTED_LORA_MODULE_COUNTS = {
    "q_proj": 8,
    "k_proj": 8,
    "v_proj": 8,
    "o_proj": 8,
    "in_proj_qkv": 24,
    "in_proj_z": 24,
    "in_proj_b": 24,
    "in_proj_a": 24,
    "out_proj": 24,
    "gate_proj": 32,
    "up_proj": 32,
    "down_proj": 32,
}


class Day20TrainPluginV2Error(ValueError):
    """The callback config, live Trainer, or training summary drifted."""


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if not isinstance(expected, str) or expected != actual:
        raise Day20TrainPluginV2Error(f"{field} mismatch")
    return expected


def _within(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise Day20TrainPluginV2Error(f"{label} is outside {root}")
    return resolved


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    from day20_train_plugin import write_json_new

    write_json_new(path, dict(value))


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    from day20_train_plugin import append_jsonl

    append_jsonl(path, dict(value))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    from day20_train_plugin import load_jsonl

    return load_jsonl(path)


def build_callback_config(
    *,
    run_root: Path,
    manifest_path: Path,
    dataset_path: Path,
    training_config_path: Path,
    run_kind: str,
    seed: int,
    learning_rate: str,
    evidence_dir: Path,
    e2b_preflight_path: Path,
    expected_ms_swift_commit: str,
) -> dict[str, Any]:
    root = run_root.resolve()
    evidence = _within(evidence_dir, root / "evidence", "evidence directory")
    validated = validate_inputs(
        manifest_path,
        dataset_path=dataset_path,
        config_path=training_config_path,
        run_kind=run_kind,
        seed=seed,
        learning_rate=learning_rate,
    )
    if validated["config_requires_resolution"]:
        raise Day20TrainPluginV2Error(
            "unresolved main template cannot be used as a callback config"
        )
    preflight = verify_preflight(e2b_preflight_path, run_root=root)
    if (
        not isinstance(expected_ms_swift_commit, str)
        or len(expected_ms_swift_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_ms_swift_commit)
    ):
        raise Day20TrainPluginV2Error("expected ms-swift commit must be a bare SHA-1")
    result: dict[str, Any] = {
        "schema_version": 2,
        "domain": CALLBACK_CONFIG_DOMAIN,
        "run_root": str(root),
        "evidence_dir": str(evidence),
        "run_kind": validated["run_kind"],
        "seed": validated["seed"],
        "learning_rate": validated["learning_rate"],
        "manifest": {
            "path": validated["manifest"],
            "file_sha256": validated["manifest_file_sha256"],
            "content_sha256": validated["manifest_sha256"],
        },
        "dataset": {
            "path": validated["dataset"],
            "file_sha256": validated["dataset_file_sha256"],
            "supervised_tokens": validated["supervised_tokens"],
            "temporal_mix_sha256": validated["temporal_mix_sha256"],
        },
        "training_config": {
            "path": validated["training_config"],
            "file_sha256": validated["training_config_file_sha256"],
            "requires_resolution": validated["config_requires_resolution"],
        },
        "optimizer_steps": validated["optimizer_steps"],
        "checkpoint_tokens": validated["checkpoint_tokens"],
        "checkpoint_steps": validated["checkpoint_steps"],
        "checkpoint_actual_tokens": validated["checkpoint_actual_tokens"],
        "e2b_preflight": {
            "path": str(e2b_preflight_path.resolve()),
            "file_sha256": e2b_file_sha256(e2b_preflight_path.resolve()),
            "content_sha256": preflight["preflight_sha256"],
        },
        "expected_ms_swift_commit": expected_ms_swift_commit,
    }
    result["callback_config_sha256"] = object_sha256(result)
    return result


def resolve_main_config(
    *,
    manifest_path: Path,
    template_path: Path,
    learning_rate: str,
    seed: int,
) -> dict[str, Any]:
    """Resolve the manifest-bound main template for primary or confirmation."""
    from day20_candidate_factory_v2 import (
        CONFIRMATION_SEED,
        PRIMARY_SEED,
        canonical_lr,
    )

    if seed not in {PRIMARY_SEED, CONFIRMATION_SEED} or isinstance(seed, bool):
        raise Day20TrainPluginV2Error("main seed is outside the v2 allowlist")
    lr = canonical_lr(learning_rate)
    manifest = load_json(manifest_path.resolve())
    _self_hash(manifest, "manifest_sha256")
    identity = (manifest.get("configs") or {}).get("main_template")
    if not isinstance(identity, Mapping):
        raise Day20TrainPluginV2Error("main template identity is missing")
    template = load_json(template_path.resolve())
    if (
        str(template_path.resolve()) != identity.get("path")
        or file_sha256(template_path.resolve()) != identity.get("file_sha256")
        or _self_hash(template, "immutable_sha256") != identity.get("immutable_sha256")
        or template.get("run_kind") != "main"
        or template.get("requires_resolution") is not True
        or (template.get("training") or {}).get("learning_rate")
        != "__SELECT_FROM_PASSING_PROBE__"
    ):
        raise Day20TrainPluginV2Error("main template identity drifted")
    result = copy.deepcopy(template)
    result["requires_resolution"] = False
    result["training"]["learning_rate"] = lr
    result["training"]["seed"] = seed
    result["training"]["data_seed"] = seed
    result["parent_main_template"] = {
        "path": str(template_path.resolve()),
        "file_sha256": identity["file_sha256"],
        "immutable_sha256": identity["immutable_sha256"],
    }
    result.pop("immutable_sha256", None)
    result["immutable_sha256"] = object_sha256(result)
    return result


def verify_callback_config(path: Path) -> dict[str, Any]:
    value = load_json(path.resolve())
    _self_hash(value, "callback_config_sha256")
    if value.get("schema_version") != 2 or value.get("domain") != CALLBACK_CONFIG_DOMAIN:
        raise Day20TrainPluginV2Error("callback config identity drifted")
    root = Path(str(value.get("run_root", ""))).resolve()
    evidence = _within(
        Path(str(value.get("evidence_dir", ""))), root / "evidence", "evidence directory"
    )
    manifest = value.get("manifest")
    dataset = value.get("dataset")
    training_config = value.get("training_config")
    e2b = value.get("e2b_preflight")
    if not all(isinstance(item, Mapping) for item in (manifest, dataset, training_config, e2b)):
        raise Day20TrainPluginV2Error("callback input identity is incomplete")
    for label, identity in (
        ("manifest", manifest),
        ("dataset", dataset),
        ("training config", training_config),
        ("E2B preflight", e2b),
    ):
        source = Path(str(identity.get("path", ""))).resolve()
        if identity.get("file_sha256") != file_sha256(source):
            raise Day20TrainPluginV2Error(f"{label} file identity drifted")
    validated = validate_inputs(
        Path(manifest["path"]),
        dataset_path=Path(dataset["path"]),
        config_path=Path(training_config["path"]),
        run_kind=str(value.get("run_kind")),
        seed=value.get("seed"),
        learning_rate=str(value.get("learning_rate")),
    )
    expected = {
        "manifest": {
            "path": validated["manifest"],
            "file_sha256": validated["manifest_file_sha256"],
            "content_sha256": validated["manifest_sha256"],
        },
        "dataset": {
            "path": validated["dataset"],
            "file_sha256": validated["dataset_file_sha256"],
            "supervised_tokens": validated["supervised_tokens"],
            "temporal_mix_sha256": validated["temporal_mix_sha256"],
        },
        "training_config": {
            "path": validated["training_config"],
            "file_sha256": validated["training_config_file_sha256"],
            "requires_resolution": validated["config_requires_resolution"],
        },
        "optimizer_steps": validated["optimizer_steps"],
        "checkpoint_tokens": validated["checkpoint_tokens"],
        "checkpoint_steps": validated["checkpoint_steps"],
        "checkpoint_actual_tokens": validated["checkpoint_actual_tokens"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise Day20TrainPluginV2Error("callback input derivation drifted")
    preflight = verify_preflight(Path(e2b["path"]), run_root=root)
    if e2b.get("content_sha256") != preflight.get("preflight_sha256"):
        raise Day20TrainPluginV2Error("E2B preflight content drifted")
    if evidence.exists() and evidence.is_symlink():
        raise Day20TrainPluginV2Error("evidence directory must not be symbolic")
    return value


def runtime_identity(expected_ms_swift_commit: str) -> dict[str, Any]:
    versions = {
        package: importlib.metadata.version(package)
        for package in EXPECTED_RUNTIME_VERSIONS
    }
    if any(
        versions[package] != expected
        for package, expected in EXPECTED_RUNTIME_VERSIONS.items()
    ):
        raise Day20TrainPluginV2Error(f"training runtime version drifted: {versions}")
    try:
        import swift
    except ImportError as error:
        raise Day20TrainPluginV2Error("ms-swift runtime is unavailable") from error
    swift_root = Path(swift.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(swift_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(swift_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day20TrainPluginV2Error("cannot verify pinned ms-swift checkout") from error
    if commit != expected_ms_swift_commit or dirty:
        raise Day20TrainPluginV2Error("ms-swift commit/worktree identity drifted")
    here = Path(__file__).resolve().parent
    implementation_files = (
        "day20_candidate_factory_v2.py",
        "day20_contract_v2.py",
        "day20_e2b_preflight_v2.py",
        "day20_ordering_v2.py",
        "day20_train_runtime_v2.py",
        "day20_train_plugin_v2.py",
        "run_day20_v2_autodl.sh",
        # Reused, pure helper implementations are explicitly in provenance.
        "day20_train_plugin.py",
    )
    source_hashes = {name: file_sha256(here / name) for name in implementation_files}
    result: dict[str, Any] = {
        "schema_version": 2,
        "domain": "day20.qwen35_lora_runtime_identity.v2",
        "backend": "hf_transformers",
        "versions": versions,
        "ms_swift_root": str(swift_root),
        "ms_swift_commit": commit,
        "ms_swift_worktree_clean": True,
        "implementation_file_sha256": source_hashes,
    }
    result["runtime_sha256"] = object_sha256(result)
    return result


def _five_step_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = [row for row in rows if row.get("global_step") in range(1, 6)]
    if [row.get("global_step") for row in first] != [1, 2, 3, 4, 5]:
        raise Day20TrainPluginV2Error("five-step gate requires steps 1 through 5")
    for row in first:
        for name in ("loss", "grad_norm", "learning_rate"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Day20TrainPluginV2Error(f"step metric {name} is invalid")
        ratio = row.get("lora_update_ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio < 0:
            raise Day20TrainPluginV2Error("LoRA update ratio is invalid")
    if any(float(row["lora_update_ratio"]) <= 0 for row in first[1:]):
        raise Day20TrainPluginV2Error("LoRA parameters did not update on steps 2-5")
    result: dict[str, Any] = {
        "schema_version": 2,
        "domain": "day20.qwen35_lora_five_step_safety.v2",
        "status": "pass",
        "steps": 5,
        "cumulative_supervised_tokens": first[-1]["cumulative_supervised_tokens"],
        "maximum_loss": max(float(row["loss"]) for row in first),
        "maximum_grad_norm": max(float(row["grad_norm"]) for row in first),
        "final_lora_update_ratio": float(first[-1]["lora_update_ratio"]),
    }
    result["summary_sha256"] = object_sha256(result)
    return result


def verify_training_summary(
    summary_path: Path, *, run_root: Path, candidate: str | None = None
) -> dict[str, Any]:
    summary = load_json(summary_path.resolve())
    _self_hash(summary, "summary_sha256")
    if (
        summary.get("schema_version") != 2
        or summary.get("domain") != TRAINING_SUMMARY_DOMAIN
        or summary.get("status") != "pass"
    ):
        raise Day20TrainPluginV2Error("training summary identity drifted")
    manifest = summary.get("experiment_manifest")
    training_config = summary.get("training_config")
    runtime = summary.get("runtime_identity")
    e2b = summary.get("e2b_preflight")
    if not all(
        isinstance(item, Mapping)
        for item in (manifest, training_config, runtime, e2b)
    ):
        raise Day20TrainPluginV2Error("training summary provenance is incomplete")
    _self_hash(runtime, "runtime_sha256")
    if (
        runtime.get("schema_version") != 2
        or runtime.get("domain") != "day20.qwen35_lora_runtime_identity.v2"
    ):
        raise Day20TrainPluginV2Error("training runtime identity drifted")
    validated = validate_inputs(
        Path(str(manifest.get("path", ""))),
        dataset_path=Path(str(summary.get("dataset", ""))),
        config_path=Path(str(training_config.get("path", ""))),
        run_kind=str(summary.get("run_kind")),
        seed=summary.get("seed"),
        learning_rate=str(summary.get("learning_rate")),
    )
    if (
        manifest.get("file_sha256") != validated["manifest_file_sha256"]
        or manifest.get("content_sha256") != validated["manifest_sha256"]
        or training_config.get("file_sha256")
        != validated["training_config_file_sha256"]
        or summary.get("dataset_file_sha256")
        != validated["dataset_file_sha256"]
        or summary.get("temporal_mix_sha256")
        != validated["temporal_mix_sha256"]
        or summary.get("optimizer_steps") != validated["optimizer_steps"]
        or summary.get("cumulative_supervised_tokens")
        != validated["supervised_tokens"]
        or summary.get("checkpoint_tokens") != validated["checkpoint_tokens"]
        or summary.get("checkpoint_steps") != validated["checkpoint_steps"]
        or summary.get("checkpoint_actual_tokens")
        != validated["checkpoint_actual_tokens"]
        or validated["config_requires_resolution"]
    ):
        raise Day20TrainPluginV2Error("training summary input derivation drifted")
    verified_preflight = verify_preflight(
        Path(str(e2b.get("path", ""))),
        run_root=run_root,
        require_live_credential=False,
    )
    if (
        e2b.get("file_sha256") != file_sha256(Path(str(e2b["path"])))
        or e2b.get("content_sha256") != verified_preflight["preflight_sha256"]
    ):
        raise Day20TrainPluginV2Error("training summary E2B preflight drifted")
    packages = summary.get("checkpoint_packages")
    candidates = summary.get("candidate_ids")
    checkpoints = summary.get("checkpoints")
    targets = summary.get("checkpoint_tokens")
    if not all(isinstance(item, Mapping) for item in (packages, candidates, checkpoints, targets)):
        raise Day20TrainPluginV2Error("training checkpoint summary is incomplete")
    if set(packages) != set(targets) or set(checkpoints) != set(targets) or set(candidates) != set(targets):
        raise Day20TrainPluginV2Error("training summary milestone set drifted")
    verified: dict[str, Any] = {}
    for label in targets:
        checkpoint = Path(str(checkpoints[label])).resolve()
        marker = verify_checkpoint_integrity(
            checkpoint,
            run_root=run_root,
            run_kind=str(summary["run_kind"]),
            seed=summary["seed"],
            learning_rate=str(summary["learning_rate"]),
            dataset_file_sha256=str(summary["dataset_file_sha256"]),
            training_config_file_sha256=str(summary["training_config"]["file_sha256"]),
            experiment_manifest_sha256=str(summary["experiment_manifest"]["content_sha256"]),
            temporal_mix_sha256=str(summary["temporal_mix_sha256"]),
        )
        expected_candidate = candidate_id(
            run_kind=str(summary["run_kind"]),
            seed=summary["seed"],
            learning_rate=str(summary["learning_rate"]),
            checkpoint=label,
        )
        expected_package = {
            "path": str(checkpoint),
            "integrity_file_sha256": file_sha256(
                checkpoint / CHECKPOINT_INTEGRITY_FILE
            ),
            "integrity_sha256": marker["integrity_sha256"],
            "snapshot_sha256": marker["snapshot_sha256"],
            "files": marker["files"],
            "resumable": True,
        }
        if candidates[label] != expected_candidate or packages[label] != expected_package:
            raise Day20TrainPluginV2Error(f"checkpoint package drifted: {label}")
        if marker.get("runtime_sha256") != runtime.get("runtime_sha256"):
            raise Day20TrainPluginV2Error(
                f"checkpoint runtime identity drifted: {label}"
            )
        verified[label] = expected_package
    if candidate is not None and candidate not in set(candidates.values()):
        raise Day20TrainPluginV2Error("requested candidate is absent from summary")
    return summary


def checkpoint_path(
    summary_path: Path, *, run_root: Path, candidate: str
) -> Path:
    summary = verify_training_summary(
        summary_path, run_root=run_root, candidate=candidate
    )
    label = next(
        name for name, value in summary["candidate_ids"].items() if value == candidate
    )
    return Path(summary["checkpoints"][label]).resolve()


def _register_swift_callback() -> None:
    import torch
    from swift.callbacks import TrainerCallback, callbacks_map
    from day20_train_plugin import (
        _update_ratio,
        reaudit_prepared_dataset,
        validate_trainable_names,
    )

    class Day20V2EvidenceCallback(TrainerCallback):
        def __init__(self, args: Any, trainer: Any) -> None:
            super().__init__(args, trainer)
            config_path = os.environ.get("DAY20_V2_CALLBACK_CONFIG")
            if not config_path:
                raise RuntimeError("DAY20_V2_CALLBACK_CONFIG is required")
            self.callback_config_path = Path(config_path).resolve()
            self.config = verify_callback_config(self.callback_config_path)
            self.run_root = Path(self.config["run_root"])
            self.evidence_dir = Path(self.config["evidence_dir"])
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.evidence_dir / "step-metrics.jsonl"
            self.inventory_path = self.evidence_dir / "trainable-inventory.json"
            self.token_reaudit_path = self.evidence_dir / "trainer-token-reaudit.json"
            self.gate_path = self.evidence_dir / "five-step-safety.json"
            self.runtime_path = self.evidence_dir / "runtime-identity.json"
            self.summary_path = self.evidence_dir / "training-summary.json"
            if self.summary_path.exists():
                raise RuntimeError("training attempt is already complete")
            self.dataset = Path(self.config["dataset"]["path"])
            self.training_config = Path(self.config["training_config"]["path"])
            self.run_kind = self.config["run_kind"]
            self.seed = int(self.config["seed"])
            self.learning_rate = float(self.config["learning_rate"])
            self.checkpoint_steps = dict(self.config["checkpoint_steps"])
            self.checkpoint_tokens = dict(self.config["checkpoint_tokens"])
            self.optimizer_steps = int(self.config["optimizer_steps"])
            from day20_train_runtime_v2 import supervised_token_schedule

            self.step_tokens = supervised_token_schedule(self.dataset)
            self.runtime_identity = runtime_identity(
                self.config["expected_ms_swift_commit"]
            )
            if self.runtime_path.exists():
                if load_json(self.runtime_path) != self.runtime_identity:
                    raise RuntimeError("runtime identity changed across resume")
            else:
                _write_json_new(self.runtime_path, self.runtime_identity)
            self.recorded_steps = (
                {int(row["global_step"]) for row in _load_jsonl(self.trace_path)}
                if self.trace_path.exists()
                else set()
            )
            self.initial_trainable: dict[str, Any] = {}

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if verify_callback_config(self.callback_config_path) != self.config:
                raise RuntimeError("callback config changed before training")
            model = self.trainer.model
            template = self.trainer.template
            expected = {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "bf16": True,
                "learning_rate": self.learning_rate,
                "num_train_epochs": 1.0,
                "seed": self.seed,
                "data_seed": self.seed,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.05,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_epsilon": 1e-8,
                "save_only_model": False,
                "save_total_limit": len(self.checkpoint_tokens),
                "save_strategy": "no",
                "train_dataloader_shuffle": False,
            }
            actual = {
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "bf16": args.bf16,
                "learning_rate": float(args.learning_rate),
                "num_train_epochs": float(args.num_train_epochs),
                "seed": int(args.seed),
                "data_seed": int(args.data_seed),
                "weight_decay": float(args.weight_decay),
                "max_grad_norm": float(args.max_grad_norm),
                "warmup_ratio": float(args.warmup_ratio),
                "adam_beta1": float(args.adam_beta1),
                "adam_beta2": float(args.adam_beta2),
                "adam_epsilon": float(args.adam_epsilon),
                "save_only_model": args.save_only_model,
                "save_total_limit": int(args.save_total_limit),
                "save_strategy": getattr(args.save_strategy, "value", args.save_strategy),
                "train_dataloader_shuffle": getattr(args, "train_dataloader_shuffle", None),
            }
            drift = {
                key: {"actual": actual[key], "expected": value}
                for key, value in expected.items()
                if actual[key] != value
            }
            if drift:
                raise RuntimeError("pinned Trainer arguments drifted: " + json.dumps(drift, sort_keys=True))
            token_reaudit = reaudit_prepared_dataset(self.dataset, template)
            if token_reaudit["supervised_tokens"] != self.config["dataset"]["supervised_tokens"]:
                raise RuntimeError("live token re-audit budget drifted")
            if self.token_reaudit_path.exists():
                if load_json(self.token_reaudit_path) != token_reaudit:
                    raise RuntimeError("token re-audit changed across resume")
            else:
                _write_json_new(self.token_reaudit_path, token_reaudit)
            training = load_json(self.training_config)["training"]
            inventory = validate_trainable_names(
                [name for name, parameter in model.named_parameters() if parameter.requires_grad],
                training["target_regex"],
                expected_module_counts=EXPECTED_LORA_MODULE_COUNTS,
            )
            if self.inventory_path.exists():
                if load_json(self.inventory_path).get("inventory_sha256") != inventory["inventory_sha256"]:
                    raise RuntimeError("trainable inventory changed across resume")
            else:
                _write_json_new(self.inventory_path, inventory)
            self.initial_trainable = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if int(state.global_step) in self.checkpoint_steps.values():
                control.should_save = True
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            logs = logs or {}
            step = int(state.global_step)
            if step <= 0 or step in self.recorded_steps or "loss" not in logs:
                return control
            ratio = _update_ratio(self.trainer.model, self.initial_trainable) if step <= 5 else None
            row = {
                "schema_version": 2,
                "domain": "day20.qwen35_lora_optimizer_step.v2",
                "global_step": step,
                "window_supervised_tokens": self.step_tokens[step - 1],
                "cumulative_supervised_tokens": sum(self.step_tokens[:step]),
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
                "lora_update_ratio": ratio,
            }
            _append_jsonl(self.trace_path, row)
            self.recorded_steps.add(step)
            if step == 5 and not self.gate_path.exists():
                _write_json_new(self.gate_path, _five_step_gate(_load_jsonl(self.trace_path)))
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(state.global_step)
            names = [name for name, value in self.checkpoint_steps.items() if value == step]
            if not names:
                return control
            checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
            integrity = build_checkpoint_integrity(
                checkpoint,
                run_root=self.run_root,
                run_kind=self.run_kind,
                seed=self.seed,
                learning_rate=str(self.config["learning_rate"]),
                global_step=step,
                cumulative_supervised_tokens=sum(self.step_tokens[:step]),
                checkpoint_targets=names,
                dataset_file_sha256=self.config["dataset"]["file_sha256"],
                training_config_file_sha256=self.config["training_config"]["file_sha256"],
                experiment_manifest_sha256=self.config["manifest"]["content_sha256"],
                temporal_mix_sha256=self.config["dataset"]["temporal_mix_sha256"],
                runtime_sha256=self.runtime_identity["runtime_sha256"],
            )
            marker = checkpoint / CHECKPOINT_INTEGRITY_FILE
            if marker.exists():
                if verify_checkpoint_integrity(checkpoint, run_root=self.run_root) != integrity:
                    raise RuntimeError("checkpoint integrity changed across save")
            else:
                write_checkpoint_integrity(checkpoint, integrity)
            _append_jsonl(
                self.evidence_dir / "checkpoint-events.jsonl",
                {
                    "global_step": step,
                    "cumulative_supervised_tokens": sum(self.step_tokens[:step]),
                    "targets": names,
                    "checkpoint": str(checkpoint),
                    "integrity_sha256": integrity["integrity_sha256"],
                },
            )
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if int(state.global_step) != self.optimizer_steps:
                raise RuntimeError("training ended before the supervised-token budget")
            trace = _load_jsonl(self.trace_path)
            if [row.get("global_step") for row in trace] != list(range(1, self.optimizer_steps + 1)):
                raise RuntimeError("optimizer evidence is incomplete or out of order")
            if load_json(self.gate_path) != _five_step_gate(trace):
                raise RuntimeError("five-step safety evidence drifted")
            checkpoints: dict[str, str] = {}
            packages: dict[str, Any] = {}
            candidate_ids: dict[str, str] = {}
            for label, step in self.checkpoint_steps.items():
                checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
                integrity = verify_checkpoint_integrity(
                    checkpoint,
                    run_root=self.run_root,
                    run_kind=self.run_kind,
                    seed=self.seed,
                    learning_rate=str(self.config["learning_rate"]),
                    dataset_file_sha256=self.config["dataset"]["file_sha256"],
                    training_config_file_sha256=self.config["training_config"]["file_sha256"],
                    experiment_manifest_sha256=self.config["manifest"]["content_sha256"],
                    temporal_mix_sha256=self.config["dataset"]["temporal_mix_sha256"],
                )
                checkpoints[label] = str(checkpoint)
                candidate_ids[label] = integrity["candidate_ids"][label]
                packages[label] = {
                    "path": str(checkpoint),
                    "integrity_file_sha256": file_sha256(checkpoint / CHECKPOINT_INTEGRITY_FILE),
                    "integrity_sha256": integrity["integrity_sha256"],
                    "snapshot_sha256": integrity["snapshot_sha256"],
                    "files": integrity["files"],
                    "resumable": True,
                }
            summary = {
                "schema_version": 2,
                "domain": TRAINING_SUMMARY_DOMAIN,
                "status": "pass",
                "run_kind": self.run_kind,
                "seed": self.seed,
                "learning_rate": self.config["learning_rate"],
                "experiment_manifest": self.config["manifest"],
                "callback_config": {
                    "path": str(self.callback_config_path),
                    "file_sha256": file_sha256(self.callback_config_path),
                    "content_sha256": self.config["callback_config_sha256"],
                },
                "training_config": self.config["training_config"],
                "dataset": str(self.dataset),
                "dataset_file_sha256": self.config["dataset"]["file_sha256"],
                "temporal_mix_sha256": self.config["dataset"]["temporal_mix_sha256"],
                "optimizer_steps": self.optimizer_steps,
                "cumulative_supervised_tokens": self.config["dataset"]["supervised_tokens"],
                "checkpoint_tokens": self.checkpoint_tokens,
                "checkpoint_steps": self.checkpoint_steps,
                "checkpoint_actual_tokens": self.config["checkpoint_actual_tokens"],
                "candidate_ids": candidate_ids,
                "checkpoints": checkpoints,
                "checkpoint_packages": packages,
                "runtime_identity": self.runtime_identity,
                "e2b_preflight": self.config["e2b_preflight"],
                "trainable_inventory": {
                    "path": str(self.inventory_path),
                    "file_sha256": file_sha256(self.inventory_path),
                },
                "trainer_token_reaudit": {
                    "path": str(self.token_reaudit_path),
                    "file_sha256": file_sha256(self.token_reaudit_path),
                },
                "step_metrics": {
                    "path": str(self.trace_path),
                    "file_sha256": file_sha256(self.trace_path),
                    "records": len(trace),
                },
                "five_step_safety": {
                    "path": str(self.gate_path),
                    "file_sha256": file_sha256(self.gate_path),
                },
            }
            summary["summary_sha256"] = object_sha256(summary)
            _write_json_new(self.summary_path, summary)
            return control

    callbacks_map["day20_v2_evidence"] = Day20V2EvidenceCallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-callback-config")
    build.add_argument("--run-root", required=True, type=Path)
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--dataset", required=True, type=Path)
    build.add_argument("--training-config", required=True, type=Path)
    build.add_argument("--run-kind", required=True, choices=("probe", "main"))
    build.add_argument("--seed", required=True, type=int)
    build.add_argument("--learning-rate", required=True)
    build.add_argument("--evidence-dir", required=True, type=Path)
    build.add_argument("--e2b-preflight", required=True, type=Path)
    build.add_argument("--expected-ms-swift-commit", required=True)
    build.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify-summary")
    verify.add_argument("--summary", required=True, type=Path)
    verify.add_argument("--run-root", required=True, type=Path)
    verify.add_argument("--candidate")
    checkpoint = subparsers.add_parser("checkpoint-path")
    checkpoint.add_argument("--summary", required=True, type=Path)
    checkpoint.add_argument("--run-root", required=True, type=Path)
    checkpoint.add_argument("--candidate", required=True)
    resolve = subparsers.add_parser("resolve-main-config")
    resolve.add_argument("--manifest", required=True, type=Path)
    resolve.add_argument("--template", required=True, type=Path)
    resolve.add_argument("--learning-rate", required=True)
    resolve.add_argument("--seed", required=True, type=int)
    resolve.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build-callback-config":
        result = build_callback_config(
            run_root=args.run_root,
            manifest_path=args.manifest,
            dataset_path=args.dataset,
            training_config_path=args.training_config,
            run_kind=args.run_kind,
            seed=args.seed,
            learning_rate=args.learning_rate,
            evidence_dir=args.evidence_dir,
            e2b_preflight_path=args.e2b_preflight,
            expected_ms_swift_commit=args.expected_ms_swift_commit,
        )
        _write_json_new(args.output.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "verify-summary":
        result = verify_training_summary(
            args.summary, run_root=args.run_root, candidate=args.candidate
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "checkpoint-path":
        print(
            checkpoint_path(
                args.summary, run_root=args.run_root, candidate=args.candidate
            )
        )
    else:
        result = resolve_main_config(
            manifest_path=args.manifest,
            template_path=args.template,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        _write_json_new(args.output.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if os.environ.get("DAY20_V2_REGISTER_SWIFT_CALLBACK") == "1":
    _register_swift_callback()


if __name__ == "__main__":
    main()
