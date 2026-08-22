#!/usr/bin/env python3
"""Validate Day 20 evaluation evidence and make the promotion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATES = ("base", "early", "mid", "final")
PROMOTION_CANDIDATES = ("early", "mid", "final")
CHECKPOINT_TOKENS = {
    "base": 0,
    "early": 64_000,
    "mid": 153_600,
    "final": 256_000,
}
THRESHOLDS = {
    "total_correct": 65,
    "general_correct": 5,
    "math_correct": 17,
    "finance_correct": 10,
    "code_correct": 14,
    "format_compliant": 90,
    "code_sandbox_execution_eligible": 26,
    "infrastructure_failures": 0,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CHECKPOINT_INTEGRITY_FILE = "day20-checkpoint-integrity.json"
EXPECTED_MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
EXPECTED_RUNTIME_VERSIONS = {
    "ms-swift": "4.5.0.dev0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
    "torch": "2.10.0+cu126",
}
EXPECTED_LORA_TARGET_REGEX = (
    r"^(?:(?:base_model|model)\.)*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
RUNTIME_IMPLEMENTATION_FILES = (
    "day20_contract.py",
    "day20_source_adapter.py",
    "day20_train_plugin.py",
    "run_day20_autodl.sh",
)
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
FORBIDDEN_TRAINABLE_MARKERS = (
    "visual",
    "vision",
    "aligner",
    "merger",
    "embed_tokens",
    "embedding",
    "lm_head",
)


class Day20FinalizationError(ValueError):
    """A Day 20 evidence or publication invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20FinalizationError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_file_manifest(
    path: Path, cache: dict[Path, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    path = path.resolve()
    if path in cache:
        return cache[path]
    if not path.is_dir():
        raise Day20FinalizationError(f"model directory is missing: {path}")
    manifest = {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    cache[path] = manifest
    return manifest


def adapter_inference_file_manifest(
    path: Path, cache: dict[Path, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    return {
        name: identity
        for name, identity in directory_file_manifest(path, cache).items()
        if name == "adapter_config.json"
        or name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day20FinalizationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day20FinalizationError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise Day20FinalizationError(f"JSONL file is missing: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise Day20FinalizationError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day20FinalizationError(
                f"invalid JSONL {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise Day20FinalizationError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Day20FinalizationError(f"{field} must be a lowercase SHA-256")
    return value


def require_semantic_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SEMANTIC_HASH_RE.fullmatch(value):
        raise Day20FinalizationError(f"{field} must be a sha256: semantic hash")
    return value


def require_absolute_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise Day20FinalizationError(f"{field} must be an absolute path")
    return Path(value).resolve()


def require_count(value: Any, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or int(value) != value
        or not 0 <= int(value) <= maximum
    ):
        raise Day20FinalizationError(
            f"{field} must be an integer count between 0 and {maximum}"
        )
    return int(value)


def verify_self_hash(payload: dict[str, Any], field: str, label: str) -> None:
    actual = require_hash(payload.get(field), f"{label}.{field}")
    expected = object_sha256({key: value for key, value in payload.items() if key != field})
    if actual != expected:
        raise Day20FinalizationError(f"{label} self-hash mismatch")


def validate_experiment_manifest(run_root: Path) -> dict[str, Any]:
    path = (run_root / "DAY20-MANIFEST.json").resolve()
    manifest = load_json(path)
    verify_self_hash(manifest, "manifest_sha256", "Day 20 experiment manifest")
    contract = manifest.get("contract")
    immutable = contract.get("immutable") if isinstance(contract, dict) else None
    main_dataset = manifest.get("datasets", {}).get("main")
    base_model_identity = manifest.get("base_model_identity")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("domain")
        != "day20.qwen35_balanced_lora.experiment_manifest"
        or manifest.get("status") != "prepared_probes_not_started"
        or require_absolute_path(manifest.get("run_root"), "manifest.run_root")
        != run_root
        or not isinstance(immutable, dict)
        or contract.get("immutable_sha256") != object_sha256(immutable)
        or not isinstance(main_dataset, dict)
        or not isinstance(base_model_identity, dict)
    ):
        raise Day20FinalizationError("Day 20 experiment manifest contract mismatch")
    base_model_path = require_absolute_path(
        base_model_identity.get("path"), "manifest.base_model_identity.path"
    )
    immutable_base_path = require_absolute_path(
        immutable.get("model", {}).get("snapshot"),
        "manifest.contract.immutable.model.snapshot",
    )
    base_model_files = base_model_identity.get("files")
    if (
        base_model_path != immutable_base_path
        or not isinstance(base_model_files, dict)
        or not base_model_files
        or base_model_identity.get("snapshot_sha256")
        != object_sha256(base_model_files)
        or directory_file_manifest(base_model_path, {}) != base_model_files
    ):
        raise Day20FinalizationError("Day 20 Base model identity drifted")
    main_dataset_path = require_absolute_path(
        main_dataset.get("path"), "manifest.datasets.main.path"
    )
    if (
        main_dataset_path != (run_root / "data" / "main.jsonl").resolve()
        or file_sha256(main_dataset_path) != main_dataset.get("file_sha256")
    ):
        raise Day20FinalizationError("Day 20 main dataset identity drifted")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": manifest["manifest_sha256"],
        "main_dataset_file_sha256": require_hash(
            main_dataset.get("file_sha256"), "manifest.datasets.main.file_sha256"
        ),
        "main_dataset_path": str(main_dataset_path),
        "base_model_path": str(base_model_path),
        "base_model_files": base_model_files,
        "base_model_snapshot_sha256": base_model_identity["snapshot_sha256"],
    }


def expected_checkpoint_step(dataset_path: Path, target_tokens: int) -> tuple[int, int]:
    rows = load_jsonl(dataset_path)
    token_counts: list[int] = []
    for ordinal, row in enumerate(rows, 1):
        value = row.get("qwen35_supervised_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Day20FinalizationError(
                f"invalid supervised-token evidence at main dataset row {ordinal}"
            )
        token_counts.append(value)
    cumulative = 0
    for offset in range(0, len(token_counts), 8):
        cumulative += sum(token_counts[offset : offset + 8])
        if cumulative >= target_tokens:
            return offset // 8 + 1, cumulative
    raise Day20FinalizationError(
        f"main dataset cannot reach checkpoint target {target_tokens}"
    )


def main_supervised_token_schedule(dataset_path: Path) -> tuple[int, list[int]]:
    rows = load_jsonl(dataset_path)
    token_counts: list[int] = []
    for ordinal, row in enumerate(rows, 1):
        value = row.get("qwen35_supervised_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Day20FinalizationError(
                f"invalid supervised-token evidence at main dataset row {ordinal}"
            )
        token_counts.append(value)
    if not token_counts:
        raise Day20FinalizationError("main training dataset is empty")
    schedule = [
        sum(token_counts[offset : offset + 8])
        for offset in range(0, len(token_counts), 8)
    ]
    if sum(schedule) != 256_000:
        raise Day20FinalizationError("main dataset supervised-token total drifted")
    return len(rows), schedule


def require_finite_number(
    value: Any, field: str, *, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Day20FinalizationError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise Day20FinalizationError(f"{field} must be {qualifier}")
    return numeric


def declared_main_evidence_file(
    identity: Any,
    *,
    run_root: Path,
    filename: str,
    label: str,
    required_fields: set[str],
) -> tuple[Path, str]:
    if not isinstance(identity, dict) or set(identity) != required_fields:
        raise Day20FinalizationError(f"{label} identity is incomplete")
    path = require_absolute_path(identity.get("path"), f"{label}.path")
    evidence_root = (run_root / "evidence" / "main").resolve()
    if path.name != filename or evidence_root not in path.parents:
        raise Day20FinalizationError(f"{label} path is outside main evidence")
    expected_hash = require_hash(identity.get("file_sha256"), f"{label}.file_sha256")
    if file_sha256(path) != expected_hash:
        raise Day20FinalizationError(f"{label} file hash mismatch")
    return path, expected_hash


def validate_runtime_identity(identity: Any) -> str:
    if not isinstance(identity, dict):
        raise Day20FinalizationError("main runtime identity is missing")
    verify_self_hash(identity, "runtime_sha256", "main runtime identity")
    ms_swift_root = require_absolute_path(
        identity.get("ms_swift_root"), "main.runtime_identity.ms_swift_root"
    )
    implementation = {
        name: file_sha256(Path(__file__).resolve().with_name(name))
        for name in RUNTIME_IMPLEMENTATION_FILES
    }
    expected = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_runtime_identity",
        "backend": "hf_transformers",
        "versions": EXPECTED_RUNTIME_VERSIONS,
        "ms_swift_root": str(ms_swift_root),
        "ms_swift_commit": EXPECTED_MS_SWIFT_COMMIT,
        "ms_swift_worktree_clean": True,
        "implementation_file_sha256": implementation,
    }
    expected["runtime_sha256"] = object_sha256(expected)
    if identity != expected:
        raise Day20FinalizationError("main runtime identity drifted")
    return expected["runtime_sha256"]


def target_name_variants(trainable_name: str) -> list[str]:
    stem = trainable_name.split(".lora_", 1)[0]
    variants = [stem]
    prefixes = ("base_model.model.", "base_model.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if variants[-1].startswith(prefix):
                variants.append(variants[-1][len(prefix) :])
                changed = True
                break
    return variants


def validate_trainable_inventory(
    identity: Any, *, run_root: Path, target_regex: str
) -> dict[str, str]:
    path, disk_hash = declared_main_evidence_file(
        identity,
        run_root=run_root,
        filename="trainable-inventory.json",
        label="main.trainable_inventory",
        required_fields={"path", "file_sha256"},
    )
    inventory = load_json(path)
    verify_self_hash(inventory, "inventory_sha256", "main trainable inventory")
    try:
        compiled = re.compile(target_regex)
    except re.error as error:
        raise Day20FinalizationError(f"invalid main LoRA target regex: {error}") from error
    names = inventory.get("trainable_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or names != sorted(names)
        or len(names) != len(set(names))
    ):
        raise Day20FinalizationError("main trainable inventory names are invalid")
    module_sides: dict[str, set[str]] = {}
    parameter_pattern = re.compile(
        r"^(?P<module>.+)\.lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$"
    )
    for name in names:
        match = parameter_pattern.fullmatch(name)
        if (
            match is None
            or any(marker in name.lower() for marker in FORBIDDEN_TRAINABLE_MARKERS)
            or not any(compiled.search(item) for item in target_name_variants(name))
        ):
            raise Day20FinalizationError(
                f"main trainable parameter is outside the exact LoRA scope: {name}"
            )
        module_sides.setdefault(match.group("module"), set()).add(match.group("side"))
    if any(sides != {"A", "B"} for sides in module_sides.values()):
        raise Day20FinalizationError("main trainable inventory has unpaired LoRA modules")
    module_counts = Counter(module.rsplit(".", 1)[-1] for module in module_sides)
    expected_module_count = sum(EXPECTED_LORA_MODULE_COUNTS.values())
    expected_tensor_count = expected_module_count * 2
    expected = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_trainable_inventory",
        "status": "pass",
        "target_regex": target_regex,
        "trainable_parameter_tensors": expected_tensor_count,
        "target_module_count": expected_module_count,
        "target_module_counts": dict(sorted(EXPECTED_LORA_MODULE_COUNTS.items())),
        "trainable_names": names,
    }
    expected["inventory_sha256"] = object_sha256(expected)
    if (
        len(names) != expected_tensor_count
        or len(module_sides) != expected_module_count
        or module_counts != Counter(EXPECTED_LORA_MODULE_COUNTS)
        or inventory != expected
    ):
        raise Day20FinalizationError("main trainable inventory distribution drifted")
    return {
        "path": str(path),
        "file_sha256": disk_hash,
        "content_sha256": expected["inventory_sha256"],
    }


def validate_trainer_token_reaudit(
    identity: Any,
    *,
    run_root: Path,
    dataset_path: Path,
    dataset_file_sha256: str,
    dataset_records: int,
) -> dict[str, str]:
    path, disk_hash = declared_main_evidence_file(
        identity,
        run_root=run_root,
        filename="trainer-token-reaudit.json",
        label="main.trainer_token_reaudit",
        required_fields={"path", "file_sha256", "content_sha256"},
    )
    report = load_json(path)
    verify_self_hash(report, "audit_sha256", "main trainer token re-audit")
    ordered_hash = require_hash(
        report.get("ordered_evidence_sha256"),
        "main.trainer_token_reaudit.ordered_evidence_sha256",
    )
    expected = {
        "schema_version": 1,
        "domain": "day20.qwen35_trainer_token_reaudit",
        "status": "pass",
        "dataset": str(dataset_path),
        "dataset_file_sha256": dataset_file_sha256,
        "records": dataset_records,
        "supervised_tokens": 256_000,
        "ordered_evidence_sha256": ordered_hash,
    }
    expected["audit_sha256"] = object_sha256(expected)
    if (
        report != expected
        or identity.get("content_sha256") != expected["audit_sha256"]
    ):
        raise Day20FinalizationError("main trainer token re-audit drifted")
    return {
        "path": str(path),
        "file_sha256": disk_hash,
        "content_sha256": expected["audit_sha256"],
    }


def validate_step_metrics(
    identity: Any, *, run_root: Path, schedule: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path, disk_hash = declared_main_evidence_file(
        identity,
        run_root=run_root,
        filename="step-metrics.jsonl",
        label="main.step_metrics",
        required_fields={"path", "file_sha256", "records"},
    )
    rows = load_jsonl(path)
    if identity.get("records") != len(schedule) or len(rows) != len(schedule):
        raise Day20FinalizationError("main optimizer step count drifted")
    cumulative = 0
    for step, (row, window_tokens) in enumerate(zip(rows, schedule), 1):
        cumulative += window_tokens
        if (
            set(row)
            != {
                "schema_version",
                "domain",
                "global_step",
                "window_supervised_tokens",
                "cumulative_supervised_tokens",
                "loss",
                "grad_norm",
                "learning_rate",
                "lora_update_ratio",
            }
            or row.get("schema_version") != 1
            or row.get("domain") != "day20.qwen35_lora_optimizer_step"
            or isinstance(row.get("global_step"), bool)
            or row.get("global_step") != step
            or row.get("window_supervised_tokens") != window_tokens
            or row.get("cumulative_supervised_tokens") != cumulative
        ):
            raise Day20FinalizationError(f"main optimizer step {step} evidence drifted")
        positive = step <= 5
        require_finite_number(row.get("loss"), f"main step {step} loss", positive=positive)
        require_finite_number(
            row.get("grad_norm"), f"main step {step} grad_norm", positive=positive
        )
        require_finite_number(
            row.get("learning_rate"),
            f"main step {step} learning_rate",
            positive=positive,
        )
        ratio = row.get("lora_update_ratio")
        if positive:
            update_ratio = require_finite_number(
                ratio,
                f"main step {step} lora_update_ratio",
                positive=step > 1,
            )
            if step == 1 and update_ratio < 0:
                raise Day20FinalizationError(
                    "main step 1 lora_update_ratio must be non-negative"
                )
        elif ratio is not None:
            raise Day20FinalizationError(
                f"main step {step} must not carry a post-gate update ratio"
            )
    return rows, {
        "path": str(path),
        "file_sha256": disk_hash,
        "records": len(rows),
    }


def validate_five_step_safety(
    identity: Any, *, run_root: Path, step_rows: list[dict[str, Any]]
) -> dict[str, str]:
    path, disk_hash = declared_main_evidence_file(
        identity,
        run_root=run_root,
        filename="five-step-safety.json",
        label="main.five_step_safety",
        required_fields={"path", "file_sha256"},
    )
    gate = load_json(path)
    verify_self_hash(gate, "summary_sha256", "main five-step safety")
    first_five = step_rows[:5]
    if len(first_five) != 5:
        raise Day20FinalizationError("main five-step safety evidence is incomplete")
    expected = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_five_step_safety",
        "status": "pass",
        "steps": 5,
        "cumulative_supervised_tokens": first_five[-1][
            "cumulative_supervised_tokens"
        ],
        "maximum_loss": max(float(row["loss"]) for row in first_five),
        "maximum_grad_norm": max(float(row["grad_norm"]) for row in first_five),
        "final_lora_update_ratio": float(first_five[-1]["lora_update_ratio"]),
    }
    expected["summary_sha256"] = object_sha256(expected)
    if gate != expected:
        raise Day20FinalizationError("main five-step safety gate drifted")
    return {
        "path": str(path),
        "file_sha256": disk_hash,
        "content_sha256": expected["summary_sha256"],
    }


def validate_training_identity(
    identity: Any,
    *,
    candidate: str,
    checkpoint: Path,
    run_root: Path,
    experiment_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if candidate == "base":
        if identity is not None:
            raise Day20FinalizationError("base candidate cannot have training identity")
        return None
    if not isinstance(identity, dict):
        raise Day20FinalizationError(f"{candidate} training identity is missing")
    expected_path = (run_root / "evidence" / "main" / "training-summary.json").resolve()
    path = require_absolute_path(identity.get("path"), f"{candidate}.training.path")
    if path != expected_path:
        raise Day20FinalizationError(f"{candidate} training summary path mismatch")
    summary = load_json(path)
    verify_self_hash(summary, "summary_sha256", f"{candidate} training summary")
    expected_tokens = CHECKPOINT_TOKENS[candidate]
    expected_step, expected_actual_tokens = expected_checkpoint_step(
        Path(experiment_manifest["main_dataset_path"]), expected_tokens
    )
    training_config = summary.get("training_config")
    checkpoint_package = (summary.get("checkpoint_packages") or {}).get(candidate)
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_lora_training_summary"
        or summary.get("status") != "pass"
        or summary.get("run_kind") != "main"
        or summary.get("cumulative_supervised_tokens") != 256_000
        or summary.get("dataset_file_sha256")
        != experiment_manifest["main_dataset_file_sha256"]
        or (summary.get("checkpoints") or {}).get(candidate) != str(checkpoint)
        or (summary.get("checkpoint_tokens") or {}).get(candidate) != expected_tokens
        or (summary.get("checkpoint_actual_tokens") or {}).get(candidate)
        != expected_actual_tokens
        or (summary.get("checkpoint_steps") or {}).get(candidate) != expected_step
        or require_absolute_path(summary.get("dataset"), f"{candidate}.training.dataset")
        != Path(experiment_manifest["main_dataset_path"])
        or not isinstance(training_config, dict)
        or not isinstance(checkpoint_package, dict)
    ):
        raise Day20FinalizationError(f"{candidate} training summary provenance mismatch")
    config_path = require_absolute_path(
        training_config.get("path"), f"{candidate}.training_config.path"
    )
    config_hash = require_hash(
        training_config.get("file_sha256"),
        f"{candidate}.training_config.file_sha256",
    )
    if file_sha256(config_path) != config_hash:
        raise Day20FinalizationError(f"{candidate} training config changed after evaluation")
    config = load_json(config_path)
    config_training = config.get("training")
    target_regex = (
        config_training.get("target_regex")
        if isinstance(config_training, dict)
        else None
    )
    if target_regex != EXPECTED_LORA_TARGET_REGEX:
        raise Day20FinalizationError(f"{candidate} training target regex drifted")
    dataset_path = Path(experiment_manifest["main_dataset_path"])
    dataset_records, step_schedule = main_supervised_token_schedule(dataset_path)
    if summary.get("optimizer_steps") != len(step_schedule):
        raise Day20FinalizationError(f"{candidate} optimizer step count drifted")
    runtime_identity_sha256 = validate_runtime_identity(summary.get("runtime_identity"))
    trainable_inventory = validate_trainable_inventory(
        summary.get("trainable_inventory"),
        run_root=run_root,
        target_regex=target_regex,
    )
    trainer_token_reaudit = validate_trainer_token_reaudit(
        summary.get("trainer_token_reaudit"),
        run_root=run_root,
        dataset_path=dataset_path,
        dataset_file_sha256=experiment_manifest["main_dataset_file_sha256"],
        dataset_records=dataset_records,
    )
    step_rows, step_metrics = validate_step_metrics(
        summary.get("step_metrics"), run_root=run_root, schedule=step_schedule
    )
    five_step_safety = validate_five_step_safety(
        summary.get("five_step_safety"), run_root=run_root, step_rows=step_rows
    )
    learning_rate = summary.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or float(learning_rate) not in {1e-5, 3e-5, 1e-4}
        or identity.get("file_sha256") != file_sha256(path)
        or identity.get("content_sha256") != summary["summary_sha256"]
        or identity.get("run_kind") != "main"
        or identity.get("checkpoint_label") != candidate
        or identity.get("checkpoint_target_supervised_tokens") != expected_tokens
        or identity.get("learning_rate") != float(learning_rate)
        or identity.get("training_config") != training_config
        or identity.get("checkpoint_package") != checkpoint_package
    ):
        raise Day20FinalizationError(f"{candidate} training identity drifted")

    integrity_path = checkpoint / CHECKPOINT_INTEGRITY_FILE
    integrity = load_json(integrity_path)
    verify_self_hash(integrity, "integrity_sha256", f"{candidate} checkpoint integrity")
    package_files = directory_file_manifest(checkpoint, {})
    package_files_without_marker = {
        name: file_identity
        for name, file_identity in package_files.items()
        if name != CHECKPOINT_INTEGRITY_FILE
    }
    expected_package = {
        "path": str(checkpoint),
        "integrity_file_sha256": file_sha256(integrity_path),
        "integrity_sha256": integrity["integrity_sha256"],
        "snapshot_sha256": object_sha256(package_files_without_marker),
        "files": package_files_without_marker,
        "resumable": True,
    }
    if (
        checkpoint_package != expected_package
        or integrity.get("checkpoint") != str(checkpoint)
        or integrity.get("run_kind") != "main"
        or integrity.get("status") != "complete"
        or integrity.get("resumable") is not True
        or candidate not in (integrity.get("targets") or [])
        or integrity.get("global_step") != expected_step
        or integrity.get("cumulative_supervised_tokens") != expected_actual_tokens
        or integrity.get("dataset_file_sha256")
        != experiment_manifest["main_dataset_file_sha256"]
        or integrity.get("training_config_file_sha256") != config_hash
        or integrity.get("files") != package_files_without_marker
        or integrity.get("snapshot_sha256")
        != object_sha256(package_files_without_marker)
    ):
        raise Day20FinalizationError(
            f"{candidate} checkpoint integrity differs from training evidence"
        )
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": summary["summary_sha256"],
        "learning_rate": float(learning_rate),
        "training_config_path": str(config_path),
        "training_config_file_sha256": config_hash,
        "runtime_identity_sha256": runtime_identity_sha256,
        "trainable_inventory": trainable_inventory,
        "trainer_token_reaudit": trainer_token_reaudit,
        "step_metrics": step_metrics,
        "five_step_safety": five_step_safety,
    }


def require_declared_file(
    identity: Any, expected_path: Path, field: str
) -> str:
    if not isinstance(identity, dict):
        raise Day20FinalizationError(f"{field} identity is missing")
    declared_path = require_absolute_path(identity.get("path"), f"{field}.path")
    if declared_path != expected_path.resolve():
        raise Day20FinalizationError(f"{field} path mismatch")
    expected_hash = require_hash(identity.get("file_sha256"), f"{field}.file_sha256")
    if file_sha256(expected_path) != expected_hash:
        raise Day20FinalizationError(f"{field} file hash mismatch")
    return expected_hash


def validate_file_manifest(
    files: Any, *, candidate: str, label: str, required_config: str
) -> str:
    if (
        not isinstance(files, dict)
        or required_config not in files
        or not any(isinstance(name, str) and name.endswith(".safetensors") for name in files)
    ):
        raise Day20FinalizationError(f"{candidate} {label} file manifest is incomplete")
    for name, file_identity in files.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(file_identity, dict)
            or isinstance(file_identity.get("bytes"), bool)
            or not isinstance(file_identity.get("bytes"), int)
            or file_identity["bytes"] < 0
        ):
            raise Day20FinalizationError(
                f"{candidate} invalid {label} file identity: {name}"
            )
        require_hash(
            file_identity.get("sha256"), f"{candidate}.{label}.files.{name}"
        )
    return object_sha256(files)


def validate_model_identity(
    identity: Any,
    *,
    candidate: str,
    checkpoint: Path,
    experiment_manifest: dict[str, Any],
    manifest_cache: dict[Path, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not isinstance(identity, dict) or identity.get("load_and_generation_verified") is not True:
        raise Day20FinalizationError(f"{candidate} model identity is incomplete")
    base = identity.get("base")
    if (
        not isinstance(base, dict)
        or base.get("model_class") != "Qwen3_5ForConditionalGeneration"
    ):
        raise Day20FinalizationError(f"{candidate} Base identity is incomplete")
    base_path = require_absolute_path(base.get("path"), f"{candidate}.base.path")
    base_snapshot = validate_file_manifest(
        base.get("files"),
        candidate=candidate,
        label="base",
        required_config="config.json",
    )
    if (
        base.get("snapshot_sha256") != base_snapshot
        or base_path != Path(experiment_manifest["base_model_path"])
        or base.get("files") != experiment_manifest["base_model_files"]
        or base_snapshot != experiment_manifest["base_model_snapshot_sha256"]
    ):
        raise Day20FinalizationError(f"{candidate} Base snapshot identity mismatch")
    if directory_file_manifest(base_path, manifest_cache) != base.get("files"):
        raise Day20FinalizationError(f"{candidate} Base directory changed after evaluation")
    adapter = identity.get("adapter")
    adapter_snapshot = None
    adapter_path = None
    if candidate == "base":
        if adapter is not None or checkpoint != base_path:
            raise Day20FinalizationError("base candidate must reference only the Base model")
    else:
        if not isinstance(adapter, dict) or adapter.get("format") != "peft_lora":
            raise Day20FinalizationError(f"{candidate} LoRA adapter identity is incomplete")
        adapter_path = require_absolute_path(
            adapter.get("path"), f"{candidate}.adapter.path"
        )
        if adapter_path != checkpoint:
            raise Day20FinalizationError(f"{candidate} checkpoint/adapter path mismatch")
        adapter_snapshot = validate_file_manifest(
            adapter.get("files"),
            candidate=candidate,
            label="adapter",
            required_config="adapter_config.json",
        )
        if adapter.get("snapshot_sha256") != adapter_snapshot:
            raise Day20FinalizationError(f"{candidate} adapter snapshot identity mismatch")
        if adapter_inference_file_manifest(adapter_path, manifest_cache) != adapter.get(
            "files"
        ):
            raise Day20FinalizationError(
                f"{candidate} inference adapter files changed after evaluation"
            )
        checkpoint_package_snapshot = validate_file_manifest(
            adapter.get("checkpoint_package_files"),
            candidate=candidate,
            label="checkpoint_package",
            required_config="adapter_config.json",
        )
        if (
            adapter.get("checkpoint_package_snapshot_sha256")
            != checkpoint_package_snapshot
            or directory_file_manifest(adapter_path, manifest_cache)
            != adapter.get("checkpoint_package_files")
        ):
            raise Day20FinalizationError(
                f"{candidate} checkpoint package changed after evaluation"
            )
    combined = object_sha256(
        {
            "domain": "day20.qwen35_base_adapter_identity",
            "schema_version": 1,
            "base_snapshot_sha256": base_snapshot,
            "adapter_snapshot_sha256": adapter_snapshot,
        }
    )
    if identity.get("combined_snapshot_sha256") != combined:
        raise Day20FinalizationError(f"{candidate} combined model identity mismatch")
    return {
        "base_path": str(base_path),
        "base_snapshot_sha256": base_snapshot,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_snapshot_sha256": adapter_snapshot,
        "checkpoint_package_snapshot_sha256": (
            checkpoint_package_snapshot if candidate != "base" else None
        ),
        "combined_snapshot_sha256": combined,
    }


def validate_prediction_rows(
    rows: list[dict[str, Any]], *, candidate: str, summary: dict[str, Any]
) -> dict[str, Any]:
    if len(rows) != 112:
        raise Day20FinalizationError(f"{candidate} predictions must contain 112 rows")
    by_slice = {
        skill: {"correct": 0, "format_compliant": 0}
        for skill in ("general", "math", "code", "finance")
    }
    sample_ids: set[str] = set()
    code_eligible = 0
    for ordinal, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        skill = row.get("slice")
        if (
            row.get("schema_version") != 2
            or row.get("domain") != "day20.qwen35_adapter_prediction"
            or row.get("ordinal") != ordinal
            or row.get("candidate") != candidate
            or row.get("recipe") != candidate
            or row.get("comparison_key") != summary.get("comparison_key")
            or row.get("run_hash") != summary.get("run_hash")
            or skill not in by_slice
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
        ):
            raise Day20FinalizationError(f"{candidate} normalized prediction identity mismatch")
        verify_self_hash(row, "row_sha256", f"{candidate} prediction {ordinal}")
        sample_ids.add(sample_id)
        if not isinstance(row.get("format_compliant"), bool):
            raise Day20FinalizationError(f"{candidate} prediction format flag is invalid")
        by_slice[skill]["format_compliant"] += int(row["format_compliant"])
        if skill == "code":
            eligible = row.get("sandbox_execution_eligible")
            if not isinstance(eligible, bool):
                raise Day20FinalizationError(f"{candidate} code eligibility flag is invalid")
            code_eligible += int(eligible)
        else:
            score_result = row.get("normalized_scorer_result")
            score = score_result.get("score") if isinstance(score_result, dict) else None
            if isinstance(score, bool) or score not in (0, 0.0, 1, 1.0):
                raise Day20FinalizationError(f"{candidate} non-code score is invalid")
            if row.get("sandbox_execution_eligible") is not None:
                raise Day20FinalizationError(
                    f"{candidate} non-code row has sandbox eligibility"
                )
            by_slice[skill]["correct"] += int(float(score) == 1.0)
    if any(sum(1 for row in rows if row["slice"] == skill) != 28 for skill in by_slice):
        raise Day20FinalizationError(f"{candidate} prediction slice distribution drifted")
    return {"by_slice": by_slice, "code_sandbox_execution_eligible": code_eligible}


def validate_adapter_summary(
    *,
    candidate: str,
    run_root: Path,
    path: Path,
    predictions_path: Path,
    experiment_manifest: dict[str, Any],
    manifest_cache: dict[Path, dict[str, dict[str, Any]]],
) -> tuple[dict[str, int], dict[str, str], dict[str, Any]]:
    summary = load_json(path)
    verify_self_hash(summary, "summary_sha256", f"{candidate} adapter summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_adapter_summary"
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("candidate") != candidate
        or summary.get("recipe") != candidate
    ):
        raise Day20FinalizationError(f"{candidate} adapter summary contract mismatch")
    checkpoint = require_absolute_path(
        summary.get("checkpoint"), f"{candidate}.adapter_summary.checkpoint"
    )
    if not checkpoint.exists():
        raise Day20FinalizationError(f"{candidate} checkpoint is missing: {checkpoint}")
    model_identity = validate_model_identity(
        summary.get("model_identity"),
        candidate=candidate,
        checkpoint=checkpoint,
        experiment_manifest=experiment_manifest,
        manifest_cache=manifest_cache,
    )
    if summary.get("model_identity_sha256") != model_identity["combined_snapshot_sha256"]:
        raise Day20FinalizationError(f"{candidate} model identity hash mismatch")
    training_identity = validate_training_identity(
        summary.get("training_identity"),
        candidate=candidate,
        checkpoint=checkpoint,
        run_root=run_root,
        experiment_manifest=experiment_manifest,
    )

    prediction_identity = summary.get("predictions")
    prediction_hash = require_declared_file(
        prediction_identity, predictions_path, f"{candidate}.predictions"
    )
    if prediction_identity.get("records") != 112:
        raise Day20FinalizationError(f"{candidate} predictions must contain 112 rows")
    prediction_metrics = validate_prediction_rows(
        load_jsonl(predictions_path), candidate=candidate, summary=summary
    )

    metrics = summary.get("metrics")
    by_slice = metrics.get("by_slice") if isinstance(metrics, dict) else None
    if not isinstance(by_slice, dict):
        raise Day20FinalizationError(f"{candidate} adapter metrics are missing")
    counts: dict[str, int] = {}
    format_by_slice = 0
    for skill in ("general", "math", "code", "finance"):
        skill_metrics = by_slice.get(skill)
        if not isinstance(skill_metrics, dict) or skill_metrics.get("records") != 28:
            raise Day20FinalizationError(f"{candidate}.{skill} must contain 28 records")
        format_by_slice += require_count(
            skill_metrics.get("format_compliant"),
            f"{candidate}.{skill}.format_compliant",
            28,
        )
        if (
            skill_metrics.get("format_compliant")
            != prediction_metrics["by_slice"][skill]["format_compliant"]
        ):
            raise Day20FinalizationError(f"{candidate}.{skill} format count drifted")
        if skill != "code":
            if skill_metrics.get("scored_records") != 28:
                raise Day20FinalizationError(f"{candidate}.{skill} is not fully scored")
            counts[f"{skill}_correct"] = require_count(
                skill_metrics.get("correct"), f"{candidate}.{skill}.correct", 28
            )
            if counts[f"{skill}_correct"] != prediction_metrics["by_slice"][skill]["correct"]:
                raise Day20FinalizationError(f"{candidate}.{skill} score count drifted")
    code_metrics = by_slice["code"]
    if code_metrics.get("executable_score_status") != "sandbox_required":
        raise Day20FinalizationError(f"{candidate} code summary is not sandbox-ready")
    counts["code_sandbox_execution_eligible"] = require_count(
        code_metrics.get("sandbox_execution_eligible"),
        f"{candidate}.code.sandbox_execution_eligible",
        28,
    )
    if (
        counts["code_sandbox_execution_eligible"]
        != prediction_metrics["code_sandbox_execution_eligible"]
    ):
        raise Day20FinalizationError(f"{candidate} code eligibility count drifted")
    counts["format_compliant"] = require_count(
        metrics.get("format_compliant"), f"{candidate}.format_compliant", 112
    )
    if counts["format_compliant"] != format_by_slice:
        raise Day20FinalizationError(f"{candidate} format count is inconsistent")
    non_code_correct = require_count(
        metrics.get("non_code_correct"), f"{candidate}.non_code_correct", 84
    )
    if non_code_correct != sum(
        counts[f"{skill}_correct"] for skill in ("general", "math", "finance")
    ):
        raise Day20FinalizationError(f"{candidate} non-code count is inconsistent")

    protocol = summary.get("protocol")
    source_protocol = protocol.get("source_evaluation_protocol") if isinstance(protocol, dict) else None
    if (
        not isinstance(source_protocol, dict)
        or protocol.get("legacy_scorer_semantics_preserved") is not True
        or protocol.get("candidate_code_executed_on_host") is not False
        or protocol.get("frozen_test_consumed") is not False
        or source_protocol.get("records") != 112
        or source_protocol.get("sample_limit") is not None
        or source_protocol.get("experiment_manifest_file_sha256")
        != experiment_manifest["file_sha256"]
        or source_protocol.get("experiment_manifest_content_sha256")
        != experiment_manifest["content_sha256"]
        or source_protocol.get("diagnostic_selection_sha256") is not None
        or source_protocol.get("template") != "qwen3_5"
        or source_protocol.get("backend") != "hf_transformers"
        or source_protocol.get("use_mcore_gdn") is not False
        or source_protocol.get("enable_thinking") is not False
        or source_protocol.get("add_non_thinking_prefix") is not True
        or source_protocol.get("response_capture")
        != "ms-swift RequestConfig(return_details=True)"
        or source_protocol.get("generated_token_ids_retained") is not True
    ):
        raise Day20FinalizationError(f"{candidate} Qwen3.5 evaluation protocol drifted")
    comparison = {
        "adapter_comparison_key": require_hash(
            summary.get("comparison_key"), f"{candidate}.adapter_comparison_key"
        ),
        "manifest_file_sha256": require_hash(
            protocol.get("manifest_file_sha256"), f"{candidate}.manifest_file_sha256"
        ),
        "scorer_file_sha256": require_hash(
            protocol.get("scorer_file_sha256"), f"{candidate}.scorer_file_sha256"
        ),
        "experiment_manifest_file_sha256": require_hash(
            protocol.get("experiment_manifest_file_sha256"),
            f"{candidate}.experiment_manifest_file_sha256",
        ),
        "experiment_manifest_content_sha256": require_hash(
            protocol.get("experiment_manifest_content_sha256"),
            f"{candidate}.experiment_manifest_content_sha256",
        ),
        "response_adapter_file_sha256": require_hash(
            protocol.get("response_adapter_file_sha256"),
            f"{candidate}.response_adapter_file_sha256",
        ),
        "code_adapter_file_sha256": require_hash(
            protocol.get("code_adapter_file_sha256"),
            f"{candidate}.code_adapter_file_sha256",
        ),
        "rescorer_file_sha256": require_hash(
            protocol.get("rescorer_file_sha256"), f"{candidate}.rescorer_file_sha256"
        ),
        "parent_rescorer_file_sha256": require_hash(
            protocol.get("parent_rescorer_file_sha256"),
            f"{candidate}.parent_rescorer_file_sha256",
        ),
    }
    if (
        comparison["experiment_manifest_file_sha256"]
        != experiment_manifest["file_sha256"]
        or comparison["experiment_manifest_content_sha256"]
        != experiment_manifest["content_sha256"]
    ):
        raise Day20FinalizationError(f"{candidate} experiment manifest drifted")
    require_hash(summary.get("run_hash"), f"{candidate}.adapter_run_hash")
    return counts, comparison, {
        "summary": summary,
        "prediction_file_sha256": prediction_hash,
        "checkpoint_path": str(checkpoint),
        "model_identity": model_identity,
        "training_identity": training_identity,
    }


def validate_e2b_summary(
    *,
    candidate: str,
    path: Path,
    result_path: Path,
    predictions_path: Path,
    adapter: dict[str, Any],
    adapter_comparison: dict[str, str],
    expected_model_identity: str,
    experiment_manifest: dict[str, Any],
) -> tuple[int, dict[str, str], int]:
    summary = load_json(path)
    verify_self_hash(summary, "summary_sha256", f"{candidate} E2B summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_adapter_code_e2b_summary"
        or summary.get("status") != "complete"
        or summary.get("candidate") != candidate
        or summary.get("recipe") != candidate
        or summary.get("records") != 28
        or summary.get("candidate_code_executed_on_host") is not False
    ):
        raise Day20FinalizationError(f"{candidate} E2B summary contract mismatch")
    declared_result = require_absolute_path(
        summary.get("result_path"), f"{candidate}.e2b.result_path"
    )
    if declared_result != result_path.resolve():
        raise Day20FinalizationError(f"{candidate} E2B result path mismatch")
    if require_hash(
        summary.get("result_file_sha256"), f"{candidate}.e2b.result_file_sha256"
    ) != file_sha256(result_path):
        raise Day20FinalizationError(f"{candidate} E2B result file hash mismatch")
    declared_predictions = require_absolute_path(
        summary.get("prediction_path"), f"{candidate}.e2b.prediction_path"
    )
    if declared_predictions != predictions_path.resolve():
        raise Day20FinalizationError(f"{candidate} E2B prediction path mismatch")
    if summary.get("prediction_file_sha256") != adapter["prediction_file_sha256"]:
        raise Day20FinalizationError(f"{candidate} E2B prediction hash mismatch")
    if summary.get("model_identity_sha256") != expected_model_identity:
        raise Day20FinalizationError(f"{candidate} E2B model identity mismatch")
    if (
        summary.get("experiment_manifest_file_sha256")
        != experiment_manifest["file_sha256"]
        or summary.get("experiment_manifest_content_sha256")
        != experiment_manifest["content_sha256"]
    ):
        raise Day20FinalizationError(f"{candidate} E2B experiment manifest drifted")
    expected_eligible = adapter["summary"]["metrics"]["by_slice"]["code"][
        "sandbox_execution_eligible"
    ]
    if (
        summary.get("sandbox_execution_eligible") != expected_eligible
        or summary.get("prediction_run_hash") != adapter["summary"]["run_hash"]
        or summary.get("prediction_comparison_key")
        != adapter["summary"]["comparison_key"]
    ):
        raise Day20FinalizationError(f"{candidate} E2B prediction identity mismatch")

    rows = load_jsonl(result_path)
    if len(rows) != 28:
        raise Day20FinalizationError(f"{candidate} E2B result must contain 28 rows")
    passed = 0
    outcomes: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    candidate_modes: Counter[str] = Counter()
    sample_ids: set[str] = set()
    execution_eligible = 0
    infrastructure_failures = 0
    effective_protocol_hash = summary.get("effective_code_protocol_hash")
    base_contract_hash = summary.get("base_sandbox_contract_hash")
    effective_protocol = summary.get("effective_code_protocol")
    if (
        not isinstance(effective_protocol, dict)
        or effective_protocol.get("parent_day10_sandbox_contract_hash")
        != base_contract_hash
    ):
        raise Day20FinalizationError(f"{candidate} sandbox parent contract drifted")
    for row in rows:
        score = row.get("score")
        sample_id = row.get("sample_id")
        candidate_mode = row.get("candidate_mode")
        executed = row.get("candidate_executed_in_sandbox")
        if (
            row.get("schema_version") != 1
            or row.get("domain") != "day20.qwen35_adapter_code_sandbox_result"
            or row.get("candidate") != candidate
            or row.get("recipe") != candidate
            or row.get("score_status") != "ok"
            or isinstance(score, bool)
            or score not in (0, 0.0, 1, 1.0)
            or row.get("prediction_run_hash") != adapter["summary"]["run_hash"]
            or row.get("prediction_comparison_key")
            != adapter["summary"]["comparison_key"]
            or row.get("normalized_prediction_file_sha256")
            != adapter["prediction_file_sha256"]
            or row.get("candidate_code_executed_on_host") is not False
            or row.get("sandbox_contract_hash") != effective_protocol_hash
            or row.get("code_execution_protocol_hash") != effective_protocol_hash
            or row.get("effective_code_protocol_hash") != effective_protocol_hash
            or row.get("base_day10_sandbox_contract_hash") != base_contract_hash
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(candidate_mode, str)
            or not candidate_mode
            or not isinstance(executed, bool)
        ):
            raise Day20FinalizationError(f"{candidate} invalid E2B result row")
        sample_ids.add(sample_id)
        candidate_modes[candidate_mode] += 1
        execution_eligible += int(executed)
        execution_status = row.get("execution_status")
        if not isinstance(execution_status, str):
            raise Day20FinalizationError(f"{candidate} E2B execution status is invalid")
        outcomes[execution_status] += 1
        error_types[str(row.get("error_type"))] += 1
        passed += int(float(score) == 1.0)
        infrastructure_failures += int(execution_status == "infrastructure_error")
    if infrastructure_failures:
        raise Day20FinalizationError(f"{candidate} E2B infrastructure failure")
    if (
        summary.get("passed") != passed
        or summary.get("failed") != 28 - passed
        or summary.get("sandbox_execution_eligible") != execution_eligible
        or summary.get("candidate_modes") != dict(sorted(candidate_modes.items()))
        or summary.get("outcomes") != dict(sorted(outcomes.items()))
        or summary.get("error_types") != dict(sorted(error_types.items()))
    ):
        raise Day20FinalizationError(f"{candidate} E2B aggregate mismatch")

    comparison = {
        "e2b_comparison_key": require_semantic_hash(
            summary.get("comparison_key"), f"{candidate}.e2b_comparison_key"
        ),
        "complete_comparison_key": require_semantic_hash(
            summary.get("complete_comparison_key"),
            f"{candidate}.complete_comparison_key",
        ),
        "effective_code_protocol_hash": require_semantic_hash(
            summary.get("effective_code_protocol_hash"),
            f"{candidate}.effective_code_protocol_hash",
        ),
        "frozen_e2b_scorer_file_sha256": require_hash(
            summary.get("frozen_e2b_scorer_file_sha256"),
            f"{candidate}.frozen_e2b_scorer_file_sha256",
        ),
        "sandbox_wrapper_file_sha256": require_hash(
            summary.get("sandbox_wrapper_file_sha256"),
            f"{candidate}.sandbox_wrapper_file_sha256",
        ),
        "base_sandbox_contract_file_sha256": require_hash(
            summary.get("base_sandbox_contract_file_sha256"),
            f"{candidate}.base_sandbox_contract_file_sha256",
        ),
        "base_sandbox_contract_hash": require_semantic_hash(
            summary.get("base_sandbox_contract_hash"),
            f"{candidate}.base_sandbox_contract_hash",
        ),
        "experiment_manifest_file_sha256": require_hash(
            summary.get("experiment_manifest_file_sha256"),
            f"{candidate}.e2b.experiment_manifest_file_sha256",
        ),
        "experiment_manifest_content_sha256": require_hash(
            summary.get("experiment_manifest_content_sha256"),
            f"{candidate}.e2b.experiment_manifest_content_sha256",
        ),
    }
    if (
        summary.get("manifest_file_sha256")
        != adapter_comparison["manifest_file_sha256"]
        or summary.get("response_adapter_file_sha256")
        != adapter_comparison["response_adapter_file_sha256"]
        or summary.get("code_adapter_file_sha256")
        != adapter_comparison["code_adapter_file_sha256"]
    ):
        raise Day20FinalizationError(f"{candidate} adapter/E2B provenance mismatch")
    for field in ("run_hash", "code_run_hash"):
        require_semantic_hash(summary.get(field), f"{candidate}.e2b.{field}")
    return passed, comparison, infrastructure_failures


def candidate_paths(run_root: Path, candidate: str) -> dict[str, Path]:
    eval_dir = run_root / "eval"
    return {
        "adapter_summary": eval_dir / f"{candidate}.qwen35-v2.json",
        "predictions": eval_dir / f"{candidate}.qwen35-v2.predictions.jsonl",
        "e2b_summary": eval_dir / f"{candidate}-code-e2b-qwen35-v2-summary.json",
        "e2b_result": eval_dir / f"{candidate}-code-e2b-qwen35-v2.jsonl",
    }


def validate_candidate(
    run_root: Path,
    candidate: str,
    *,
    experiment_manifest: dict[str, Any],
    manifest_cache: dict[Path, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    paths = candidate_paths(run_root, candidate)
    counts, adapter_comparison, adapter = validate_adapter_summary(
        candidate=candidate,
        run_root=run_root,
        path=paths["adapter_summary"],
        predictions_path=paths["predictions"],
        experiment_manifest=experiment_manifest,
        manifest_cache=manifest_cache,
    )
    snapshot = adapter["model_identity"]["combined_snapshot_sha256"]
    code_correct, e2b_comparison, infrastructure_failures = validate_e2b_summary(
        candidate=candidate,
        path=paths["e2b_summary"],
        result_path=paths["e2b_result"],
        predictions_path=paths["predictions"],
        adapter=adapter,
        adapter_comparison=adapter_comparison,
        expected_model_identity=snapshot,
        experiment_manifest=experiment_manifest,
    )
    counts["code_correct"] = code_correct
    counts["total_correct"] = sum(
        counts[f"{skill}_correct"] for skill in ("general", "math", "finance", "code")
    )
    counts["infrastructure_failures"] = infrastructure_failures
    checks = {field: counts[field] >= minimum for field, minimum in THRESHOLDS.items()}
    checks["infrastructure_failures"] = (
        counts["infrastructure_failures"] == THRESHOLDS["infrastructure_failures"]
    )
    failed = [field for field, passed in checks.items() if not passed]
    return {
        "candidate": candidate,
        "checkpoint_tokens": CHECKPOINT_TOKENS[candidate],
        "checkpoint_path": adapter["checkpoint_path"],
        "model_identity_sha256": snapshot,
        "base_snapshot_sha256": adapter["model_identity"]["base_snapshot_sha256"],
        "adapter_snapshot_sha256": adapter["model_identity"]["adapter_snapshot_sha256"],
        "training_identity": adapter["training_identity"],
        "metrics": counts,
        "threshold_checks": checks,
        "failed_thresholds": failed,
        "promotion_eligible": candidate in PROMOTION_CANDIDATES and not failed,
        "artifacts": {
            name: {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "comparison_identity": {**adapter_comparison, **e2b_comparison},
    }


def atomic_write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise Day20FinalizationError(f"refusing to overwrite finalization artifact: {path}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def serialized_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finalize(run_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    if not (run_root / ".day20-run-root").is_file():
        raise Day20FinalizationError("run root marker is missing")
    results_path = run_root / "DAY20-RESULTS.json"
    pass_path = run_root / "DAY20-PASS.json"
    existing_results = load_json(results_path) if results_path.exists() else None
    existing_marker = load_json(pass_path) if pass_path.exists() else None
    existing_created_at_values = [
        artifact.get("created_at_utc")
        for artifact in (existing_results, existing_marker)
        if artifact is not None
    ]
    if any(
        not isinstance(value, str) or not value
        for value in existing_created_at_values
    ):
        raise Day20FinalizationError("existing finalization timestamps are inconsistent")
    existing_created_at = set(existing_created_at_values)
    if len(existing_created_at) > 1:
        raise Day20FinalizationError("existing finalization timestamps are inconsistent")
    if existing_created_at:
        recovered_created_at = next(iter(existing_created_at))
        if created_at_utc is not None and created_at_utc != recovered_created_at:
            raise Day20FinalizationError("existing finalization timestamp differs")
        created_at_utc = recovered_created_at

    experiment_manifest = validate_experiment_manifest(run_root)
    manifest_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    candidates = {
        candidate: validate_candidate(
            run_root,
            candidate,
            experiment_manifest=experiment_manifest,
            manifest_cache=manifest_cache,
        )
        for candidate in CANDIDATES
    }
    comparison_identities = {
        object_sha256(candidate["comparison_identity"])
        for candidate in candidates.values()
    }
    if len(comparison_identities) != 1:
        raise Day20FinalizationError("comparison identity differs across candidates")
    if len({candidate["checkpoint_path"] for candidate in candidates.values()}) != 4:
        raise Day20FinalizationError("candidate checkpoint paths are not distinct")
    if len({candidate["model_identity_sha256"] for candidate in candidates.values()}) != 4:
        raise Day20FinalizationError("candidate model identities are not distinct")
    if len({candidate["base_snapshot_sha256"] for candidate in candidates.values()}) != 1:
        raise Day20FinalizationError("Base model identity differs across candidates")
    if candidates["base"]["checkpoint_path"] != experiment_manifest["base_model_path"]:
        raise Day20FinalizationError("Base candidate differs from experiment manifest")
    common_training_identities = {
        object_sha256(candidates[candidate]["training_identity"])
        for candidate in PROMOTION_CANDIDATES
    }
    if len(common_training_identities) != 1:
        raise Day20FinalizationError("main training provenance differs across checkpoints")

    eligible = [
        candidates[candidate]
        for candidate in PROMOTION_CANDIDATES
        if candidates[candidate]["promotion_eligible"]
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["metrics"]["total_correct"],
            -item["metrics"]["general_correct"],
            -item["metrics"]["code_correct"],
            item["checkpoint_tokens"],
        ),
    )
    selected = ranked[0] if ranked else None
    created_at = created_at_utc or utc_now()
    decision = {
        "status": (
            "eligible_day20_lora_anchor"
            if selected is not None
            else "no_eligible_day20_lora_anchor"
        ),
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_checkpoint_path": selected["checkpoint_path"] if selected else None,
        "selected_checkpoint_tokens": selected["checkpoint_tokens"] if selected else None,
        "ranked_eligible_candidates": [item["candidate"] for item in ranked],
        "fallback_candidate": "base",
        "tie_break_order": [
            "total_correct_desc",
            "general_correct_desc",
            "code_correct_desc",
            "checkpoint_tokens_asc",
        ],
    }
    results = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_promotion_results",
        "status": "pass" if selected else "no_eligible_day20_lora_anchor",
        "created_at_utc": created_at,
        "run_root": str(run_root),
        "candidate_recipe_contract": "recipe equals candidate",
        "experiment_manifest": experiment_manifest,
        "thresholds": THRESHOLDS,
        "comparison_identity": next(iter(candidates.values()))["comparison_identity"],
        "candidates": candidates,
        "decision": decision,
        "frozen_test_consumed": False,
    }
    results["results_sha256"] = object_sha256(results)
    expected_results_file_sha256 = serialized_json_sha256(results)
    marker = None
    if selected is not None:
        marker = {
            "schema_version": 1,
            "domain": "day20.qwen35_lora_pass",
            "status": "pass",
            "created_at_utc": created_at,
            "canonical_run": str(run_root),
            "selected_candidate": selected["candidate"],
            "selected_checkpoint_path": selected["checkpoint_path"],
            "selected_checkpoint_tokens": selected["checkpoint_tokens"],
            "selected_model_identity_sha256": selected["model_identity_sha256"],
            "selected_metrics": selected["metrics"],
            "results": {
                "path": str(results_path),
                "file_sha256": expected_results_file_sha256,
            },
            "frozen_test_consumed": False,
        }
        marker["marker_sha256"] = object_sha256(marker)

    if existing_results is not None and (
        existing_results != results
        or file_sha256(results_path) != expected_results_file_sha256
    ):
        raise Day20FinalizationError("existing Day 20 results failed strict recovery")
    if existing_marker is not None and (
        marker is None
        or existing_marker != marker
        or file_sha256(pass_path) != serialized_json_sha256(marker)
    ):
        raise Day20FinalizationError("existing Day 20 pass marker failed strict recovery")

    if marker is None:
        if existing_results is None:
            atomic_write_json_new(results_path, results)
        return results

    # Publish PASS first so an interruption can never expose eligible RESULTS
    # without its marker.  Either one-sided crash state is recoverable only
    # after the complete evidence chain is revalidated above.
    if existing_marker is None:
        atomic_write_json_new(pass_path, marker)
    if existing_results is None:
        atomic_write_json_new(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    results = finalize(args.run_root)
    print(json.dumps(results["decision"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
