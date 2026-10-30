#!/usr/bin/env python3
"""Day 20 LoRA training evidence callback and deterministic runner helpers.

The module is stdlib-only during normal import.  The ms-swift callback is
registered lazily only when DAY20_REGISTER_SWIFT_CALLBACK=1 is set by the
remote runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import copy
import importlib.metadata
import subprocess
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


PROBE_LRS = ("1e-5", "3e-5", "1e-4")
SKILLS = ("general", "math", "code", "finance")
MAIN_CHECKPOINTS = {"early": 64_000, "mid": 153_600, "final": 256_000}
FORBIDDEN_TRAINABLE_MARKERS = (
    "visual",
    "vision",
    "aligner",
    "merger",
    "embed_tokens",
    "embedding",
    "lm_head",
)
CHECKPOINT_INTEGRITY_FILE = "day20-checkpoint-integrity.json"
EXPORT_MANIFEST_FILE = "EXPORT-MANIFEST.json"
PROMOTION_CHECKPOINT_TOKENS = {
    "early": 64_000,
    "mid": 153_600,
    "final": 256_000,
}
PROMOTION_THRESHOLDS = {
    "total_correct": 65,
    "general_correct": 5,
    "math_correct": 17,
    "finance_correct": 10,
    "code_correct": 14,
    "format_compliant": 90,
    "code_sandbox_execution_eligible": 26,
    "infrastructure_failures": 0,
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
EXPECTED_RUNTIME_VERSIONS = {
    "ms-swift": "4.5.0.dev0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
}


class Day20PluginError(ValueError):
    """A Day 20 training or evidence invariant failed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20PluginError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_manifest(checkpoint: Path) -> dict[str, dict[str, Any]]:
    if not checkpoint.is_dir():
        raise Day20PluginError(f"checkpoint directory is missing: {checkpoint}")
    return {
        path.relative_to(checkpoint).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != CHECKPOINT_INTEGRITY_FILE
    }


def build_checkpoint_integrity(
    checkpoint: Path,
    *,
    run_kind: str,
    global_step: int,
    cumulative_supervised_tokens: int,
    targets: list[str],
    dataset_file_sha256: str,
    training_config_file_sha256: str,
    runtime_sha256: str,
) -> dict[str, Any]:
    if run_kind not in {"probe", "main"}:
        raise Day20PluginError("checkpoint run kind must be probe or main")
    if not re.fullmatch(r"[0-9a-f]{64}", runtime_sha256):
        raise Day20PluginError("checkpoint runtime identity is invalid")
    files = checkpoint_file_manifest(checkpoint)
    required = {"adapter_config.json", "trainer_state.json"}
    if not required <= set(files) or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise Day20PluginError("checkpoint model/trainer files are incomplete")
    resumable = run_kind == "main"
    if resumable and (
        not {"optimizer.pt", "scheduler.pt"} <= set(files)
        or not any(Path(name).name.startswith("rng_state") and name.endswith(".pth") for name in files)
    ):
        raise Day20PluginError("main checkpoint resumability files are incomplete")
    payload = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_checkpoint_integrity",
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "run_kind": run_kind,
        "global_step": global_step,
        "cumulative_supervised_tokens": cumulative_supervised_tokens,
        "targets": sorted(targets),
        "resumable": resumable,
        "dataset_file_sha256": dataset_file_sha256,
        "training_config_file_sha256": training_config_file_sha256,
        "runtime_sha256": runtime_sha256,
        "files": files,
        "snapshot_sha256": object_sha256(files),
    }
    payload["integrity_sha256"] = object_sha256(payload)
    return payload


def verify_checkpoint_integrity(checkpoint: Path) -> dict[str, Any]:
    marker_path = checkpoint / CHECKPOINT_INTEGRITY_FILE
    marker = load_json(marker_path)
    expected_hash = object_sha256(
        {key: value for key, value in marker.items() if key != "integrity_sha256"}
    )
    files = checkpoint_file_manifest(checkpoint)
    if (
        marker.get("schema_version") != 1
        or marker.get("domain") != "day20.qwen35_lora_checkpoint_integrity"
        or marker.get("status") != "complete"
        or marker.get("checkpoint") != str(checkpoint.resolve())
        or marker.get("integrity_sha256") != expected_hash
        or marker.get("files") != files
        or marker.get("snapshot_sha256") != object_sha256(files)
    ):
        raise Day20PluginError(f"checkpoint integrity drifted: {checkpoint}")
    # Re-run required-file checks rather than trusting the marker's resumable flag.
    rebuilt = build_checkpoint_integrity(
        checkpoint,
        run_kind=str(marker.get("run_kind")),
        global_step=int(marker.get("global_step")),
        cumulative_supervised_tokens=int(marker.get("cumulative_supervised_tokens")),
        targets=list(marker.get("targets") or []),
        dataset_file_sha256=str(marker.get("dataset_file_sha256")),
        training_config_file_sha256=str(
            marker.get("training_config_file_sha256")
        ),
        runtime_sha256=str(marker.get("runtime_sha256")),
    )
    if rebuilt != marker:
        raise Day20PluginError(f"checkpoint contract drifted: {checkpoint}")
    return marker


def require_checkpoint_within_run(checkpoint: Path, run_root: Path) -> Path:
    resolved = checkpoint.resolve()
    adapters = (run_root.resolve() / "adapters").resolve()
    if resolved == adapters or adapters not in resolved.parents:
        raise Day20PluginError("checkpoint is outside the current Day 20 adapters root")
    return resolved


def runtime_identity(expected_ms_swift_commit: str) -> dict[str, Any]:
    versions = {
        package: importlib.metadata.version(package)
        for package in (*EXPECTED_RUNTIME_VERSIONS, "torch")
    }
    if any(
        versions[package] != expected
        for package, expected in EXPECTED_RUNTIME_VERSIONS.items()
    ):
        raise Day20PluginError(f"training runtime version drifted: {versions}")
    try:
        import swift
    except ImportError as error:
        raise Day20PluginError("ms-swift runtime is unavailable") from error
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
        raise Day20PluginError("cannot verify the pinned ms-swift checkout") from error
    if commit != expected_ms_swift_commit or dirty:
        raise Day20PluginError("ms-swift commit/worktree identity drifted")
    source_files = {
        name: file_sha256(Path(__file__).resolve().with_name(name))
        for name in (
            "day20_contract.py",
            "day20_source_adapter.py",
            "day20_train_plugin.py",
            "run_day20_autodl.sh",
        )
    }
    result = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_runtime_identity",
        "backend": "hf_transformers",
        "versions": versions,
        "ms_swift_root": str(swift_root),
        "ms_swift_commit": commit,
        "ms_swift_worktree_clean": True,
        "implementation_file_sha256": source_files,
    }
    result["runtime_sha256"] = object_sha256(result)
    return result


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day20PluginError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day20PluginError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise Day20PluginError(f"JSONL file is missing: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise Day20PluginError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day20PluginError(
                f"invalid JSONL {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise Day20PluginError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise Day20PluginError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonical_lr(value: str) -> str:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise Day20PluginError(f"invalid probe learning rate: {value!r}") from error
    for allowed in PROBE_LRS:
        if parsed == Decimal(allowed):
            return allowed
    raise Day20PluginError(
        f"probe learning rate must be one of {', '.join(PROBE_LRS)}"
    )


def supervised_token_schedule(
    dataset: Path, *, per_device_batch_size: int = 2, accumulation_steps: int = 4
) -> list[int]:
    if per_device_batch_size != 2 or accumulation_steps != 4:
        raise Day20PluginError("Day 20 global batch must remain 2 x 4 = 8")
    token_counts: list[int] = []
    for index, row in enumerate(load_jsonl(dataset), 1):
        value = row.get("qwen35_supervised_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Day20PluginError(
                f"invalid qwen35_supervised_tokens at {dataset}:{index}"
            )
        token_counts.append(value)
    if not token_counts:
        raise Day20PluginError("training dataset is empty")
    window = per_device_batch_size * accumulation_steps
    return [
        sum(token_counts[start : start + window])
        for start in range(0, len(token_counts), window)
    ]


def reaudit_prepared_dataset(dataset: Path, template: Any) -> dict[str, Any]:
    """Re-encode every training row with the live Trainer template."""
    before_sha = file_sha256(dataset)
    evidence: list[dict[str, Any]] = []
    for index, row in enumerate(load_jsonl(dataset), 1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise Day20PluginError(f"training row {index} has no messages")
        try:
            encoded = template.encode({"messages": messages}, return_length=True)
        except Exception as error:
            raise Day20PluginError(
                f"live Qwen3.5 encoding failed for training row {index}: {error}"
            ) from error
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        labels = encoded.get("labels") if isinstance(encoded, dict) else None
        if not isinstance(input_ids, (list, tuple)) or not isinstance(
            labels, (list, tuple)
        ):
            raise Day20PluginError(
                f"live Qwen3.5 encoding omitted tokens for training row {index}"
            )
        try:
            token_ids = [int(value) for value in input_ids]
            label_ids = [int(value) for value in labels]
        except (TypeError, ValueError) as error:
            raise Day20PluginError(
                f"live Qwen3.5 token evidence is non-integral at row {index}"
            ) from error
        supervised = sum(label != -100 for label in label_ids[1:])
        expected = {
            "input_tokens": len(token_ids),
            "supervised_tokens": supervised,
            "render_sha256": object_sha256(token_ids),
            "labels_sha256": object_sha256(label_ids),
            "messages_sha256": object_sha256(messages),
        }
        stored = {
            "input_tokens": row.get("qwen35_input_tokens"),
            "supervised_tokens": row.get("qwen35_supervised_tokens"),
            "render_sha256": row.get("qwen35_render_sha256"),
            "labels_sha256": row.get("qwen35_labels_sha256"),
            "messages_sha256": row.get("content_sha256"),
        }
        if expected != stored or supervised <= 0:
            raise Day20PluginError(
                f"live Qwen3.5 token evidence drifted at training row {index}"
            )
        evidence.append(
            {
                "sample_id": row.get("sample_id"),
                **expected,
            }
        )
    after_sha = file_sha256(dataset)
    if before_sha != after_sha:
        raise Day20PluginError("training dataset changed during live token re-audit")
    report = {
        "schema_version": 1,
        "domain": "day20.qwen35_trainer_token_reaudit",
        "status": "pass",
        "dataset": str(dataset.resolve()),
        "dataset_file_sha256": before_sha,
        "records": len(evidence),
        "supervised_tokens": sum(row["supervised_tokens"] for row in evidence),
        "ordered_evidence_sha256": object_sha256(evidence),
    }
    report["audit_sha256"] = object_sha256(report)
    return report


def checkpoint_steps(
    step_tokens: Iterable[int], targets: dict[str, int]
) -> dict[str, int]:
    ordered_targets = sorted(targets.items(), key=lambda item: item[1])
    if not ordered_targets or any(
        not name
        or isinstance(target, bool)
        or not isinstance(target, int)
        or target <= 0
        for name, target in ordered_targets
    ):
        raise Day20PluginError("checkpoint targets must be named positive integers")
    if len({target for _, target in ordered_targets}) != len(ordered_targets):
        raise Day20PluginError("checkpoint token targets must be unique")
    result: dict[str, int] = {}
    cumulative = 0
    target_index = 0
    steps = list(step_tokens)
    for step, count in enumerate(steps, 1):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise Day20PluginError(f"invalid supervised-token count at step {step}")
        cumulative += count
        while target_index < len(ordered_targets):
            name, target = ordered_targets[target_index]
            if cumulative < target:
                break
            # A Trainer checkpoint can only be taken after an optimizer step.
            # Intermediate token budgets are therefore thresholds; the final
            # budget remains exact.  The callback records both target and
            # actual cumulative tokens so this rounding is never hidden.
            result[name] = step
            target_index += 1
    if target_index != len(ordered_targets):
        missing = [name for name, _ in ordered_targets[target_index:]]
        raise Day20PluginError(f"dataset does not reach checkpoint targets: {missing}")
    if cumulative != ordered_targets[-1][1]:
        raise Day20PluginError(
            f"dataset token total {cumulative} differs from final target "
            f"{ordered_targets[-1][1]}"
        )
    return result


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Day20PluginError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise Day20PluginError(f"{field} must be finite and positive")
    return numeric


def validate_five_step_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_five = [row for row in rows if int(row.get("global_step", -1)) <= 5]
    if len(first_five) != 5 or [row.get("global_step") for row in first_five] != list(
        range(1, 6)
    ):
        raise Day20PluginError("five-step safety gate requires steps 1 through 5")
    cumulative: list[int] = []
    update_ratios: list[float] = []
    for row in first_five:
        step = row["global_step"]
        _finite_positive(row.get("loss"), f"step {step} loss")
        _finite_positive(row.get("grad_norm"), f"step {step} grad_norm")
        _finite_positive(row.get("learning_rate"), f"step {step} learning_rate")
        update_value = row.get("lora_update_ratio")
        if isinstance(update_value, bool) or not isinstance(
            update_value, (int, float)
        ):
            raise Day20PluginError(
                f"step {step} lora_update_ratio must be numeric"
            )
        update_ratio = float(update_value)
        if not math.isfinite(update_ratio) or update_ratio < 0:
            raise Day20PluginError(
                f"step {step} lora_update_ratio must be finite and non-negative"
            )
        update_ratios.append(update_ratio)
        value = row.get("cumulative_supervised_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Day20PluginError(
                f"step {step} cumulative supervised tokens are invalid"
            )
        cumulative.append(value)
    # The pinned cosine scheduler starts with optimizer LR=0; its first logged
    # LR is the value for the following step.  Therefore step 1 may legitimately
    # have a zero parameter delta, while steps 2-5 must prove LoRA updates.
    if any(value <= 0 for value in update_ratios[1:]):
        raise Day20PluginError(
            "LoRA parameters did not update on each of safety steps 2 through 5"
        )
    if cumulative != sorted(set(cumulative)):
        raise Day20PluginError("cumulative supervised tokens are not strictly increasing")
    result = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_five_step_safety",
        "status": "pass",
        "steps": 5,
        "cumulative_supervised_tokens": cumulative[-1],
        "maximum_loss": max(float(row["loss"]) for row in first_five),
        "maximum_grad_norm": max(float(row["grad_norm"]) for row in first_five),
        "final_lora_update_ratio": float(first_five[-1]["lora_update_ratio"]),
    }
    result["summary_sha256"] = object_sha256(result)
    return result


def _target_name_variants(trainable_name: str) -> list[str]:
    stem = trainable_name.split(".lora_", 1)[0]
    variants = [stem]
    prefixes = (
        "base_model.model.",
        "base_model.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if variants[-1].startswith(prefix):
                variants.append(variants[-1][len(prefix) :])
                changed = True
                break
    return variants


def validate_trainable_names(
    trainable_names: Iterable[str],
    target_regex: str,
    *,
    expected_module_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    try:
        compiled = re.compile(target_regex)
    except re.error as error:
        raise Day20PluginError(f"invalid LoRA target regex: {error}") from error
    names = sorted(trainable_names)
    if not names or len(names) != len(set(names)):
        raise Day20PluginError("trainable parameter inventory is empty or duplicated")
    failures: list[str] = []
    module_sides: dict[str, set[str]] = {}
    for name in names:
        lowered = name.lower()
        if ".lora_" not in name:
            failures.append(f"non-LoRA trainable parameter: {name}")
            continue
        if any(marker in lowered for marker in FORBIDDEN_TRAINABLE_MARKERS):
            failures.append(f"forbidden trainable parameter: {name}")
            continue
        if not any(compiled.search(variant) for variant in _target_name_variants(name)):
            failures.append(f"trainable parameter is outside target_regex: {name}")
            continue
        side_match = re.search(r"\.lora_([AB])(?:\.|$)", name)
        if side_match is None:
            failures.append(f"cannot identify LoRA A/B side: {name}")
            continue
        module_sides.setdefault(name.split(".lora_", 1)[0], set()).add(
            side_match.group(1)
        )
    unpaired = sorted(
        module for module, sides in module_sides.items() if sides != {"A", "B"}
    )
    if unpaired:
        failures.append(f"unpaired LoRA modules: {unpaired[:5]}")
    module_counts = Counter(module.rsplit(".", 1)[-1] for module in module_sides)
    if expected_module_counts is not None and module_counts != Counter(
        expected_module_counts
    ):
        failures.append(
            "LoRA target module distribution drifted: "
            f"expected={dict(expected_module_counts)}, actual={dict(module_counts)}"
        )
    if failures:
        raise Day20PluginError("; ".join(failures[:10]))
    payload = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_trainable_inventory",
        "status": "pass",
        "target_regex": target_regex,
        "trainable_parameter_tensors": len(names),
        "target_module_count": len(module_sides),
        "target_module_counts": dict(sorted(module_counts.items())),
        "trainable_names": names,
    }
    payload["inventory_sha256"] = object_sha256(payload)
    return payload


def validate_prepared_inputs(
    manifest_path: Path, *, dataset_kind: str, learning_rate: str
) -> dict[str, Any]:
    if dataset_kind not in {"probe", "main"}:
        raise Day20PluginError("dataset_kind must be probe or main")
    manifest = load_json(manifest_path)
    expected_manifest_hash = object_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise Day20PluginError("DAY20-MANIFEST self-hash mismatch")
    contract = manifest.get("contract")
    immutable = contract.get("immutable") if isinstance(contract, dict) else None
    if not isinstance(immutable, dict) or contract.get("immutable_sha256") != object_sha256(
        immutable
    ):
        raise Day20PluginError("immutable Day 20 contract hash mismatch")
    lora = immutable.get("lora")
    if not isinstance(lora, dict):
        raise Day20PluginError("immutable LoRA contract is missing")
    required_lora = {"rank": 8, "alpha": 16, "dropout": 0.05, "bias": "none"}
    if any(lora.get(key) != value for key, value in required_lora.items()):
        raise Day20PluginError("immutable LoRA hyperparameters drifted")
    target_regex = lora.get("target_regex")
    if not isinstance(target_regex, str) or not target_regex:
        raise Day20PluginError("immutable LoRA target_regex is missing")
    re.compile(target_regex)
    datasets = manifest.get("datasets")
    dataset_identity = datasets.get(dataset_kind) if isinstance(datasets, dict) else None
    expected_tokens = 16_000 if dataset_kind == "probe" else 256_000
    if (
        not isinstance(dataset_identity, dict)
        or dataset_identity.get("supervised_tokens") != expected_tokens
    ):
        raise Day20PluginError(f"{dataset_kind} supervised-token budget drifted")
    dataset_path = Path(str(dataset_identity.get("path", ""))).resolve()
    if not dataset_path.is_absolute() or file_sha256(dataset_path) != dataset_identity.get(
        "file_sha256"
    ):
        raise Day20PluginError(f"{dataset_kind} dataset identity drifted")
    step_tokens = supervised_token_schedule(dataset_path)
    if sum(step_tokens) != expected_tokens:
        raise Day20PluginError(f"{dataset_kind} dataset token total drifted")
    if dataset_kind == "probe" and len(step_tokens) < 5:
        raise Day20PluginError("probe dataset cannot reach the five-step safety gate")
    if dataset_kind == "probe":
        lr = canonical_lr(learning_rate)
        targets = {"probe": 16_000}
    else:
        lr = canonical_lr(learning_rate)
        targets = MAIN_CHECKPOINTS
    steps = checkpoint_steps(step_tokens, targets)
    configs = manifest.get("configs")
    if dataset_kind == "probe":
        config_identity = (
            configs.get("probes", {}).get(lr) if isinstance(configs, dict) else None
        )
        if not isinstance(config_identity, dict):
            raise Day20PluginError(f"prepared probe config is missing: {lr}")
        config_path = Path(str(config_identity.get("path", ""))).resolve()
        if file_sha256(config_path) != config_identity.get("file_sha256"):
            raise Day20PluginError(f"prepared probe config hash drifted: {lr}")
        config = load_json(config_path)
        if not _self_hash_valid(config, "immutable_sha256") or Decimal(
            str(config.get("training", {}).get("learning_rate"))
        ) != Decimal(lr):
            raise Day20PluginError(f"prepared probe config content drifted: {lr}")
    else:
        config_path = None
    return {
        "dataset": str(dataset_path),
        "learning_rate": lr,
        "target_regex": target_regex,
        "expected_supervised_tokens": expected_tokens,
        "optimizer_steps": len(step_tokens),
        "checkpoint_steps": steps,
        "prepared_config": str(config_path) if config_path else None,
    }


def resolve_main_config(
    manifest_path: Path, selection_path: Path
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    selection = validate_probe_selection(manifest_path, selection_path)
    if not _self_hash_valid(manifest, "manifest_sha256"):
        raise Day20PluginError("DAY20-MANIFEST self-hash mismatch")
    learning_rate = canonical_lr(str(selection.get("selected_lr")))
    configs = manifest.get("configs")
    identity = configs.get("main_template") if isinstance(configs, dict) else None
    if not isinstance(identity, dict):
        raise Day20PluginError("main training config template is missing")
    template_path = Path(str(identity.get("path", ""))).resolve()
    if file_sha256(template_path) != identity.get("file_sha256"):
        raise Day20PluginError("main training config template hash drifted")
    template = load_json(template_path)
    if (
        not _self_hash_valid(template, "immutable_sha256")
        or template.get("training", {}).get("learning_rate")
        != "__SELECT_FROM_PASSING_PROBE__"
    ):
        raise Day20PluginError("main training config template content drifted")
    resolved = copy.deepcopy(template)
    resolved["training"]["learning_rate"] = float(Decimal(learning_rate))
    resolved["parent_main_template"] = {
        "path": str(template_path),
        "file_sha256": file_sha256(template_path),
        "immutable_sha256": template["immutable_sha256"],
    }
    resolved["probe_selection"] = {
        "path": str(selection_path.resolve()),
        "file_sha256": file_sha256(selection_path),
        "selection_sha256": selection["selection_sha256"],
    }
    resolved.pop("immutable_sha256", None)
    resolved["immutable_sha256"] = object_sha256(resolved)
    return resolved


def _self_hash_valid(payload: dict[str, Any], field: str) -> bool:
    return payload.get(field) == object_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _diagnostic_ids(
    day20_manifest_path: Path, eval_manifest_path: Path
) -> tuple[list[str], list[str]]:
    day20_manifest = load_json(day20_manifest_path)
    if not _self_hash_valid(day20_manifest, "manifest_sha256"):
        raise Day20PluginError("Day 20 experiment manifest self-hash mismatch")
    datasets = day20_manifest.get("datasets")
    identity = datasets.get("diagnostic") if isinstance(datasets, dict) else None
    if not isinstance(identity, dict) or identity.get("records") != 32:
        raise Day20PluginError("Day 20 diagnostic identity is incomplete")
    path = Path(str(identity.get("path", ""))).resolve()
    if file_sha256(path) != identity.get("file_sha256"):
        raise Day20PluginError("Day 20 diagnostic file hash drifted")
    rows = load_jsonl(path)
    ordered_ids = [row.get("sample_id") for row in rows]
    if (
        len(rows) != 32
        or ordered_ids != identity.get("ordered_sample_ids")
        or object_sha256(ordered_ids) != identity.get("selection_sha256")
        or Counter(row.get("slice") for row in rows)
        != Counter({skill: 8 for skill in SKILLS})
    ):
        raise Day20PluginError("Day 20 diagnostic order/distribution drifted")
    eval_manifest = load_json(eval_manifest_path)
    eval_records = eval_manifest.get("records")
    if not isinstance(eval_records, list):
        raise Day20PluginError("frozen eval manifest is malformed")
    dev_by_id = {
        row.get("sample_id"): row
        for row in eval_records
        if isinstance(row, dict) and row.get("evaluation_split") == "dev"
    }
    if any(
        sample_id not in dev_by_id
        or dev_by_id[sample_id].get("slice") != row.get("slice")
        for sample_id, row in zip(ordered_ids, rows)
    ):
        raise Day20PluginError("Day 20 diagnostic is not bound to frozen dev")
    return ordered_ids, [str(row["slice"]) for row in rows]


def _recompute_probe_adapter_metrics(
    rows: list[dict[str, Any]], *, candidate: str
) -> dict[str, Any]:
    by_slice: dict[str, dict[str, Any]] = {}
    for row in rows:
        score = row.get("normalized_scorer_result")
        code_candidate = row.get("code_candidate")
        code_static = row.get("code_static_syntax")
        output = row.get("normalized_output")
        anomalies = row.get("anomalies")
        if (
            not isinstance(score, dict)
            or not isinstance(output, str)
            or not isinstance(anomalies, list)
            or not isinstance(row.get("format_compliant"), bool)
        ):
            raise Day20PluginError(f"probe row scoring evidence is invalid: {candidate}")
        if row["slice"] == "code":
            if (
                not isinstance(code_candidate, dict)
                or not isinstance(code_candidate.get("candidate_mode"), str)
                or not isinstance(code_candidate.get("execution_eligible"), bool)
                or not isinstance(code_static, dict)
                or not isinstance(code_static.get("valid"), bool)
                or not isinstance(row.get("sandbox_execution_eligible"), bool)
                or score.get("score") is not None
                or score.get("score_status") != "sandbox_required"
            ):
                raise Day20PluginError(
                    f"probe Code acceptance evidence is invalid: {candidate}"
                )
            eligible = code_candidate["execution_eligible"] and code_static["valid"]
            format_compliant = (
                bool(output.strip())
                and score.get("parse_status") == "ok"
                and eligible
                and code_static["valid"]
            )
            if (
                row["sandbox_execution_eligible"] is not eligible
                or row["format_compliant"] is not format_compliant
            ):
                raise Day20PluginError(
                    f"probe Code format/eligibility drifted: {candidate}"
                )
        else:
            numeric = score.get("score")
            if (
                isinstance(numeric, bool)
                or not isinstance(numeric, (int, float))
                or not math.isfinite(float(numeric))
                or float(numeric) not in {0.0, 1.0}
            ):
                raise Day20PluginError(
                    f"probe non-Code score is invalid: {candidate}"
                )

    for skill in SKILLS:
        skill_rows = [row for row in rows if row["slice"] == skill]
        scores = [row["normalized_scorer_result"].get("score") for row in skill_rows]
        numeric_scores = [
            float(score)
            for score in scores
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        ]
        entry: dict[str, Any] = {
            "records": len(skill_rows),
            "scored_records": len(numeric_scores),
            "correct": sum(numeric_scores) if numeric_scores else None,
            "accuracy": (
                sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
            ),
            "format_compliant": sum(
                bool(row["format_compliant"]) for row in skill_rows
            ),
            "format_compliance_rate": sum(
                bool(row["format_compliant"]) for row in skill_rows
            )
            / len(skill_rows),
            "anomalous_records": sum(bool(row["anomalies"]) for row in skill_rows),
        }
        if skill == "code":
            entry.update(
                {
                    "syntax_valid": sum(
                        bool(row["code_static_syntax"]["valid"])
                        for row in skill_rows
                    ),
                    "sandbox_execution_eligible": sum(
                        bool(row["sandbox_execution_eligible"])
                        for row in skill_rows
                    ),
                    "candidate_modes": dict(
                        sorted(
                            Counter(
                                row["code_candidate"]["candidate_mode"]
                                for row in skill_rows
                            ).items()
                        )
                    ),
                    "executable_score_status": "sandbox_required",
                }
            )
        by_slice[skill] = entry
    non_code = [row for row in rows if row["slice"] != "code"]
    non_code_correct = sum(
        float(row["normalized_scorer_result"]["score"]) for row in non_code
    )
    return {
        "by_slice": by_slice,
        "non_code_correct": non_code_correct,
        "non_code_total": len(non_code),
        "non_code_accuracy": non_code_correct / len(non_code),
        "format_compliant": sum(bool(row["format_compliant"]) for row in rows),
        "format_compliance_rate": sum(
            bool(row["format_compliant"]) for row in rows
        )
        / len(rows),
        "anomalous_records": sum(bool(row["anomalies"]) for row in rows),
    }


def _probe_metrics(
    *,
    candidate: str,
    summary_path: Path,
    e2b_summary_path: Path,
    expected_ids: list[str],
    expected_slices: list[str],
) -> dict[str, Any]:
    summary = load_json(summary_path)
    comparison = summary.get("comparison_key")
    run = summary.get("run_hash")
    model_identity = summary.get("model_identity_sha256")
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_adapter_summary"
        or not _self_hash_valid(summary, "summary_sha256")
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("candidate") != candidate
        or summary.get("recipe") != candidate
        or not isinstance(comparison, str)
        or not re.fullmatch(r"[0-9a-f]{64}", comparison)
        or not isinstance(run, str)
        or not re.fullmatch(r"[0-9a-f]{64}", run)
        or not isinstance(model_identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", model_identity)
    ):
        raise Day20PluginError(f"invalid normalized probe summary: {candidate}")
    predictions = summary.get("predictions")
    if not isinstance(predictions, dict) or predictions.get("records") != 32:
        raise Day20PluginError(f"probe {candidate} must contain 32 predictions")
    predictions_path = Path(str(predictions.get("path", ""))).resolve()
    if file_sha256(predictions_path) != predictions.get("file_sha256"):
        raise Day20PluginError(f"probe prediction hash drifted: {candidate}")
    rows = load_jsonl(predictions_path)
    if (
        len(rows) != 32
        or [row.get("ordinal") for row in rows] != list(range(1, 33))
        or [row.get("sample_id") for row in rows] != expected_ids
        or [row.get("slice") for row in rows] != expected_slices
    ):
        raise Day20PluginError(f"probe identity/order drifted: {candidate}")
    for row in rows:
        if (
            row.get("schema_version") != 2
            or row.get("domain") != "day20.qwen35_adapter_prediction"
            or row.get("candidate") != candidate
            or row.get("recipe") != candidate
            or row.get("comparison_key") != comparison
            or row.get("run_hash") != run
            or not _self_hash_valid(row, "row_sha256")
        ):
            raise Day20PluginError(f"probe prediction row drifted: {candidate}")
    recomputed_metrics = _recompute_probe_adapter_metrics(
        rows, candidate=candidate
    )
    if summary.get("metrics") != recomputed_metrics:
        raise Day20PluginError(f"probe aggregate metrics drifted: {candidate}")
    by_slice = recomputed_metrics["by_slice"]
    counts = {
        skill: int(by_slice[skill]["correct"])
        for skill in ("general", "math", "finance")
    }
    code_eligible = int(by_slice["code"]["sandbox_execution_eligible"])

    e2b = load_json(e2b_summary_path)
    result_path = Path(str(e2b.get("result_path", ""))).resolve()
    if (
        e2b.get("schema_version") != 1
        or e2b.get("domain") != "day20.qwen35_adapter_code_e2b_summary"
        or not _self_hash_valid(e2b, "summary_sha256")
        or e2b.get("status") != "complete"
        or e2b.get("candidate") != candidate
        or e2b.get("recipe") != candidate
        or e2b.get("records") != 8
        or e2b.get("result_path") != str(result_path)
        or e2b.get("prediction_path") != str(predictions_path)
        or e2b.get("prediction_file_sha256") != file_sha256(predictions_path)
        or e2b.get("prediction_run_hash") != run
        or e2b.get("prediction_comparison_key") != comparison
        or e2b.get("model_identity_sha256") != model_identity
        or not isinstance(e2b.get("comparison_key"), str)
        or not e2b["comparison_key"]
        or not isinstance(e2b.get("complete_comparison_key"), str)
        or not e2b["complete_comparison_key"]
        or not isinstance(e2b.get("run_hash"), str)
        or not e2b["run_hash"]
        or not isinstance(e2b.get("code_run_hash"), str)
        or not e2b["code_run_hash"]
    ):
        raise Day20PluginError(f"invalid probe E2B summary: {candidate}")
    if file_sha256(result_path) != e2b.get("result_file_sha256"):
        raise Day20PluginError(f"probe E2B result hash drifted: {candidate}")
    result_rows = load_jsonl(result_path)
    expected_code_ids = [
        sample_id for sample_id, skill in zip(expected_ids, expected_slices)
        if skill == "code"
    ]
    if (
        len(result_rows) != 8
        or [row.get("sample_id") for row in result_rows] != expected_code_ids
    ):
        raise Day20PluginError(f"probe E2B identity/order drifted: {candidate}")
    code_rows_by_id = {
        row["sample_id"]: row for row in rows if row["slice"] == "code"
    }
    for ordinal, row in enumerate(result_rows):
        score = row.get("score")
        status = row.get("execution_status")
        scorer_result = row.get("scorer_result")
        prediction = code_rows_by_id[row["sample_id"]]
        if (
            row.get("schema_version") != 1
            or row.get("domain")
            != "day20.qwen35_adapter_code_sandbox_result"
            or row.get("code_ordinal") != ordinal
            or row.get("prediction_run_ordinal")
            != prediction["ordinal"] - 1
            or row.get("candidate") != candidate
            or row.get("recipe") != candidate
            or row.get("slice") != "code"
            or row.get("prediction_run_hash") != run
            or row.get("prediction_comparison_key") != comparison
            or row.get("model_snapshot_hash") != model_identity
            or row.get("run_hash") != e2b["run_hash"]
            or row.get("comparison_key") != e2b["comparison_key"]
            or row.get("complete_comparison_key")
            != e2b["complete_comparison_key"]
            or row.get("code_run_hash") != e2b["code_run_hash"]
            or row.get("score_status") != "ok"
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or float(score) not in {0.0, 1.0}
            or row.get("passed") is not (float(score) == 1.0)
            or not isinstance(status, str)
            or not status
            or not isinstance(scorer_result, dict)
            or scorer_result.get("score_status") != "ok"
            or scorer_result.get("score") != score
            or scorer_result.get("passed") is not row.get("passed")
            or scorer_result.get("execution_outcome") != status
            or row.get("candidate_executed_in_sandbox")
            is not prediction["sandbox_execution_eligible"]
            or (
                not prediction["sandbox_execution_eligible"]
                and (
                    status != "rejected_candidate_contract"
                    or float(score) != 0.0
                )
            )
            or row.get("candidate_code_executed_on_host") is not False
            or not isinstance(row.get("candidate_mode"), str)
            or not row["candidate_mode"]
        ):
            raise Day20PluginError(f"probe E2B result row drifted: {candidate}")
    code_correct = sum(float(row["score"]) == 1.0 for row in result_rows)
    infrastructure_failures = sum(
        row["execution_status"] == "infrastructure_error"
        or row["score_status"] == "infrastructure_error"
        for row in result_rows
    )
    outcomes = dict(sorted(Counter(row["execution_status"] for row in result_rows).items()))
    candidate_modes = dict(
        sorted(Counter(str(row.get("candidate_mode")) for row in result_rows).items())
    )
    error_types = dict(
        sorted(Counter(str(row.get("error_type")) for row in result_rows).items())
    )
    if (
        infrastructure_failures != 0
        or e2b.get("passed") != code_correct
        or e2b.get("failed") != len(result_rows) - code_correct
        or e2b.get("sandbox_execution_eligible") != code_eligible
        or e2b.get("outcomes") != outcomes
        or e2b.get("candidate_modes") != candidate_modes
        or e2b.get("error_types") != error_types
    ):
        raise Day20PluginError(f"probe E2B aggregate metrics drifted: {candidate}")
    counts["code"] = code_correct
    counts["code_sandbox_execution_eligible"] = code_eligible
    counts["total"] = sum(counts[skill] for skill in SKILLS)
    return counts


def select_probe(
    *,
    day20_manifest: Path,
    eval_manifest: Path,
    base_summary: Path,
    base_e2b_summary: Path,
    probes: list[tuple[str, Path, Path]],
) -> dict[str, Any]:
    if len(probes) != 3 or {canonical_lr(item[0]) for item in probes} != set(
        PROBE_LRS
    ):
        raise Day20PluginError("probe selection requires all three allowlisted LRs")
    expected_ids, expected_slices = _diagnostic_ids(
        day20_manifest, eval_manifest
    )
    base = _probe_metrics(
        candidate="base-probe",
        summary_path=base_summary,
        e2b_summary_path=base_e2b_summary,
        expected_ids=expected_ids,
        expected_slices=expected_slices,
    )
    candidates: list[dict[str, Any]] = []
    for raw_lr, summary_path, e2b_path in probes:
        lr = canonical_lr(raw_lr)
        candidate = f"probe-{lr}"
        metrics = _probe_metrics(
            candidate=candidate,
            summary_path=summary_path,
            e2b_summary_path=e2b_path,
            expected_ids=expected_ids,
            expected_slices=expected_slices,
        )
        failures: list[str] = []
        if metrics["total"] < base["total"] + 2:
            failures.append("total_below_base_plus_2")
        for skill in ("math", "finance", "code"):
            if metrics[skill] < base[skill] - 1:
                failures.append(f"{skill}_regression_gt_1")
        if metrics["code_sandbox_execution_eligible"] < 7:
            failures.append("code_sandbox_execution_eligible_below_7")
        candidates.append(
            {
                "learning_rate": lr,
                "candidate": candidate,
                "metrics": metrics,
                "failures": failures,
                "eligible": not failures,
                "adapter_summary": {
                    "path": str(summary_path.resolve()),
                    "file_sha256": file_sha256(summary_path),
                },
                "e2b_summary": {
                    "path": str(e2b_path.resolve()),
                    "file_sha256": file_sha256(e2b_path),
                },
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        status = "no_passing_probe"
        selected = None
    else:
        selected = sorted(
            eligible,
            key=lambda row: (
                -row["metrics"]["total"],
                -row["metrics"]["general"],
                Decimal(row["learning_rate"]),
            ),
        )[0]
        status = "selected"
    result = {
        "schema_version": 1,
        "domain": "day20.qwen35_lora_probe_selection",
        "status": status,
        "eval_manifest": {
            "path": str(eval_manifest.resolve()),
            "file_sha256": file_sha256(eval_manifest),
        },
        "day20_manifest": {
            "path": str(day20_manifest.resolve()),
            "file_sha256": file_sha256(day20_manifest),
        },
        "diagnostic_sample_ids": expected_ids,
        "base_summary": {
            "path": str(base_summary.resolve()),
            "file_sha256": file_sha256(base_summary),
        },
        "base_e2b_summary": {
            "path": str(base_e2b_summary.resolve()),
            "file_sha256": file_sha256(base_e2b_summary),
        },
        "base_metrics": base,
        "candidates": candidates,
        "selected_lr": selected["learning_rate"] if selected else None,
        "selected_candidate": selected["candidate"] if selected else None,
        "tie_break": ["total_desc", "general_desc", "learning_rate_asc"],
    }
    result["selection_sha256"] = object_sha256(result)
    return result


def validate_probe_selection(
    day20_manifest_path: Path, selection_path: Path
) -> dict[str, Any]:
    """Recompute a probe decision from its immutable evaluation artifacts."""
    selection = load_json(selection_path)
    if (
        selection.get("schema_version") != 1
        or selection.get("domain") != "day20.qwen35_lora_probe_selection"
        or selection.get("status") != "selected"
        or not _self_hash_valid(selection, "selection_sha256")
        or selection.get("tie_break")
        != ["total_desc", "general_desc", "learning_rate_asc"]
    ):
        raise Day20PluginError("probe selection is not eligible for main training")

    manifest_identity = selection.get("day20_manifest")
    if (
        not isinstance(manifest_identity, dict)
        or Path(str(manifest_identity.get("path", ""))).resolve()
        != day20_manifest_path.resolve()
        or manifest_identity.get("file_sha256") != file_sha256(day20_manifest_path)
    ):
        raise Day20PluginError("probe selection Day 20 manifest identity drifted")

    eval_identity = selection.get("eval_manifest")
    base_identity = selection.get("base_summary")
    base_e2b_identity = selection.get("base_e2b_summary")
    if not all(
        isinstance(identity, dict)
        and isinstance(identity.get("path"), str)
        and isinstance(identity.get("file_sha256"), str)
        for identity in (eval_identity, base_identity, base_e2b_identity)
    ):
        raise Day20PluginError("probe selection source artifacts are incomplete")

    def verified_path(identity: dict[str, Any], label: str) -> Path:
        path = Path(identity["path"]).resolve()
        if file_sha256(path) != identity["file_sha256"]:
            raise Day20PluginError(f"probe selection {label} identity drifted")
        return path

    eval_manifest = verified_path(eval_identity, "eval manifest")
    base_summary = verified_path(base_identity, "Base summary")
    base_e2b_summary = verified_path(base_e2b_identity, "Base E2B summary")

    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(PROBE_LRS):
        raise Day20PluginError("probe selection requires exactly three candidates")
    probes: list[tuple[str, Path, Path]] = []
    for expected_lr, candidate in zip(PROBE_LRS, candidates):
        if (
            not isinstance(candidate, dict)
            or candidate.get("learning_rate") != expected_lr
            or candidate.get("candidate") != f"probe-{expected_lr}"
        ):
            raise Day20PluginError("probe selection candidate order/identity drifted")
        adapter_identity = candidate.get("adapter_summary")
        e2b_identity = candidate.get("e2b_summary")
        if not isinstance(adapter_identity, dict) or not isinstance(e2b_identity, dict):
            raise Day20PluginError("probe candidate artifact identity is incomplete")
        probes.append(
            (
                expected_lr,
                verified_path(adapter_identity, f"{expected_lr} summary"),
                verified_path(e2b_identity, f"{expected_lr} E2B summary"),
            )
        )

    recomputed = select_probe(
        day20_manifest=day20_manifest_path,
        eval_manifest=eval_manifest,
        base_summary=base_summary,
        base_e2b_summary=base_e2b_summary,
        probes=probes,
    )
    if recomputed != selection:
        raise Day20PluginError("probe selection differs from recomputed evidence")
    return selection


def verified_checkpoint_path(
    summary_path: Path, *, name: str, run_root: Path
) -> Path:
    summary = load_json(summary_path)
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_lora_training_summary"
        or summary.get("status") != "pass"
        or not _self_hash_valid(summary, "summary_sha256")
    ):
        raise Day20PluginError("training summary is not complete")
    checkpoint_value = (summary.get("checkpoints") or {}).get(name)
    package = (summary.get("checkpoint_packages") or {}).get(name)
    if not isinstance(checkpoint_value, str) or not isinstance(package, dict):
        raise Day20PluginError(f"checkpoint is missing from training summary: {name}")
    checkpoint = require_checkpoint_within_run(Path(checkpoint_value), run_root)
    integrity = verify_checkpoint_integrity(checkpoint)
    expected_package = {
        "path": str(checkpoint),
        "integrity_file_sha256": file_sha256(
            checkpoint / CHECKPOINT_INTEGRITY_FILE
        ),
        "integrity_sha256": integrity["integrity_sha256"],
        "snapshot_sha256": integrity["snapshot_sha256"],
        "files": integrity["files"],
        "resumable": integrity["resumable"],
    }
    run_kind = summary.get("run_kind")
    runtime = summary.get("runtime_identity")
    if (
        package != expected_package
        or integrity.get("run_kind") != run_kind
        or name not in integrity.get("targets", [])
        or integrity.get("dataset_file_sha256")
        != summary.get("dataset_file_sha256")
        or integrity.get("training_config_file_sha256")
        != (summary.get("training_config") or {}).get("file_sha256")
        or not isinstance(runtime, dict)
        or not _self_hash_valid(runtime, "runtime_sha256")
        or integrity.get("runtime_sha256") != runtime.get("runtime_sha256")
        or (run_kind == "main") is not bool(integrity.get("resumable"))
    ):
        raise Day20PluginError(f"checkpoint package identity drifted: {name}")
    return checkpoint


def latest_resumable_checkpoint(output_dir: Path, *, run_root: Path) -> Path:
    output = require_checkpoint_within_run(output_dir, run_root)
    candidates: list[tuple[int, Path]] = []
    for path in output.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    for _, candidate in sorted(candidates, reverse=True):
        try:
            integrity = verify_checkpoint_integrity(candidate)
        except (Day20PluginError, ValueError, TypeError):
            continue
        if integrity.get("run_kind") == "main" and integrity.get("resumable") is True:
            return candidate.resolve()
    raise Day20PluginError("no complete resumable main checkpoint was found")


def stage_resume_evidence(
    source_evidence: Path,
    destination_evidence: Path,
    *,
    checkpoint: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Seed a new append-only attempt with evidence through a verified checkpoint."""

    run_root = run_root.resolve()
    evidence_root = (run_root / "evidence/main").resolve()
    source = source_evidence.resolve()
    destination = destination_evidence.resolve()
    if (
        source.parent != evidence_root
        or destination.parent != evidence_root
        or source == destination
        or source_evidence.is_symlink()
        or destination_evidence.is_symlink()
        or not source.is_dir()
        or not destination.is_dir()
        or any(destination.iterdir())
    ):
        raise Day20PluginError("resume evidence attempts are missing or unsafe")
    integrity = verify_checkpoint_integrity(
        require_checkpoint_within_run(checkpoint, run_root)
    )
    if integrity.get("run_kind") != "main" or integrity.get("resumable") is not True:
        raise Day20PluginError("resume checkpoint is not a complete main checkpoint")
    resume_step = integrity["global_step"]
    resume_targets = set(integrity.get("targets") or [])
    if not resume_targets:
        raise Day20PluginError("resume checkpoint has no checkpoint target")
    trace = load_jsonl(source / "step-metrics.jsonl")
    if any(
        isinstance(row.get("global_step"), bool)
        or not isinstance(row.get("global_step"), int)
        for row in trace
    ):
        raise Day20PluginError("optimizer evidence contains an invalid step")
    prefix = [row for row in trace if row["global_step"] <= resume_step]
    if (
        [row.get("global_step") for row in prefix]
        != list(range(1, resume_step + 1))
        or not prefix
        or prefix[-1].get("cumulative_supervised_tokens")
        != integrity.get("cumulative_supervised_tokens")
    ):
        raise Day20PluginError("optimizer evidence does not align with resume checkpoint")
    inventory = load_json(source / "trainable-inventory.json")
    token_reaudit = load_json(source / "trainer-token-reaudit.json")
    safety = load_json(source / "five-step-safety.json")
    runtime = load_json(source / "runtime-identity.json")
    if (
        not _self_hash_valid(inventory, "inventory_sha256")
        or not _self_hash_valid(token_reaudit, "audit_sha256")
        or not _self_hash_valid(runtime, "runtime_sha256")
        or runtime.get("runtime_sha256") != integrity.get("runtime_sha256")
        or safety != validate_five_step_metrics(prefix)
    ):
        raise Day20PluginError("resume evidence self-validation failed")
    for name, value in (
        ("trainable-inventory.json", inventory),
        ("trainer-token-reaudit.json", token_reaudit),
        ("five-step-safety.json", safety),
        ("runtime-identity.json", runtime),
    ):
        write_json_new(destination / name, value)
    for row in prefix:
        append_jsonl(destination / "step-metrics.jsonl", row)

    source_events_path = source / "checkpoint-events.jsonl"
    source_events = load_jsonl(source_events_path) if source_events_path.exists() else []
    events: list[dict[str, Any]] = []
    observed_targets: set[str] = set()
    target_checkpoints: dict[str, str] = {}
    for row in source_events:
        step = row.get("global_step")
        event_checkpoint = row.get("checkpoint")
        targets = row.get("targets")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step > resume_step
        ):
            continue
        if (
            not isinstance(event_checkpoint, str)
            or not isinstance(targets, list)
            or any(not isinstance(name, str) or not name for name in targets)
        ):
            raise Day20PluginError("checkpoint event evidence is malformed")
        event_integrity = verify_checkpoint_integrity(
            require_checkpoint_within_run(Path(event_checkpoint), run_root)
        )
        if (
            event_integrity.get("run_kind") != "main"
            or event_integrity.get("global_step") != step
            or sorted(targets) != event_integrity.get("targets")
            or event_integrity.get("dataset_file_sha256")
            != integrity.get("dataset_file_sha256")
            or event_integrity.get("training_config_file_sha256")
            != integrity.get("training_config_file_sha256")
            or event_integrity.get("runtime_sha256")
            != integrity.get("runtime_sha256")
        ):
            raise Day20PluginError("checkpoint event evidence drifted")
        for name in targets:
            previous = target_checkpoints.setdefault(name, event_checkpoint)
            if previous != event_checkpoint:
                raise Day20PluginError("checkpoint target has conflicting events")
        events.append(row)
        observed_targets.update(targets)
    if not resume_targets <= observed_targets:
        events.append(
            {
                "global_step": resume_step,
                "cumulative_supervised_tokens": integrity[
                    "cumulative_supervised_tokens"
                ],
                "targets": sorted(resume_targets),
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    for row in events:
        append_jsonl(destination / "checkpoint-events.jsonl", row)
    result = {
        "schema_version": 1,
        "domain": "day20.qwen35_resume_evidence_stage",
        "status": "complete",
        "source_evidence": str(source),
        "destination_evidence": str(destination),
        "checkpoint": str(checkpoint.resolve()),
        "global_step": resume_step,
        "cumulative_supervised_tokens": integrity[
            "cumulative_supervised_tokens"
        ],
        "trace_records": len(prefix),
        "checkpoint_events": len(events),
    }
    result["stage_sha256"] = object_sha256(result)
    return result


def _verified_self_hash(
    payload: dict[str, Any], field: str, *, label: str
) -> str:
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-f]{64}", value)
        or value
        != object_sha256({key: item for key, item in payload.items() if key != field})
    ):
        raise Day20PluginError(f"{label} self-hash drifted")
    return value


def _verified_file_identity(
    identity: Any,
    *,
    expected_path: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise Day20PluginError(f"{label} file identity is missing")
    declared = identity.get("path")
    expected_hash = identity.get("file_sha256")
    if (
        not isinstance(declared, str)
        or not Path(declared).is_absolute()
        or Path(declared).resolve() != expected_path.resolve()
        or not isinstance(expected_hash, str)
        or expected_hash != file_sha256(expected_path)
    ):
        raise Day20PluginError(f"{label} file identity drifted")
    return {"path": str(expected_path.resolve()), "file_sha256": expected_hash}


def validate_winner_for_export(run_root: Path) -> dict[str, Any]:
    """Recompute the immutable promotion and checkpoint gate before merging."""

    run_root = run_root.resolve()
    marker = run_root / ".day20-run-root"
    if (
        not marker.is_file()
        or marker.is_symlink()
        or marker.read_text(encoding="utf-8").strip()
        != "day20-qwen35-balanced-lora-sft-v1"
    ):
        raise Day20PluginError("Day 20 run marker is invalid")
    results_path = run_root / "DAY20-RESULTS.json"
    pass_path = run_root / "DAY20-PASS.json"
    results = load_json(results_path)
    pass_marker = load_json(pass_path)
    results_content_hash = _verified_self_hash(
        results, "results_sha256", label="Day 20 results"
    )
    pass_content_hash = _verified_self_hash(
        pass_marker, "marker_sha256", label="Day 20 PASS marker"
    )
    if (
        results.get("schema_version") != 1
        or results.get("domain") != "day20.qwen35_lora_promotion_results"
        or results.get("status") != "pass"
        or results.get("run_root") != str(run_root)
        or results.get("frozen_test_consumed") is not False
        or results.get("thresholds") != PROMOTION_THRESHOLDS
        or pass_marker.get("schema_version") != 1
        or pass_marker.get("domain") != "day20.qwen35_lora_pass"
        or pass_marker.get("status") != "pass"
        or pass_marker.get("canonical_run") != str(run_root)
        or pass_marker.get("frozen_test_consumed") is not False
    ):
        raise Day20PluginError("Day 20 promotion artifacts are not merge-eligible")
    _verified_file_identity(
        pass_marker.get("results"),
        expected_path=results_path,
        label="Day 20 PASS results",
    )

    decision = results.get("decision")
    candidates = results.get("candidates")
    winner = pass_marker.get("selected_candidate")
    if (
        not isinstance(decision, dict)
        or not isinstance(candidates, dict)
        or winner not in PROMOTION_CHECKPOINT_TOKENS
        or decision.get("status") != "eligible_day20_lora_anchor"
        or decision.get("selected_candidate") != winner
        or not isinstance(decision.get("ranked_eligible_candidates"), list)
        or not decision["ranked_eligible_candidates"]
        or decision["ranked_eligible_candidates"][0] != winner
    ):
        raise Day20PluginError("Day 20 selected winner is inconsistent")
    candidate = candidates.get(winner)
    if not isinstance(candidate, dict):
        raise Day20PluginError("Day 20 selected candidate evidence is missing")
    expected_tokens = PROMOTION_CHECKPOINT_TOKENS[winner]
    selected_checkpoint = pass_marker.get("selected_checkpoint_path")
    if (
        candidate.get("candidate") != winner
        or candidate.get("promotion_eligible") is not True
        or candidate.get("failed_thresholds") != []
        or candidate.get("checkpoint_tokens") != expected_tokens
        or decision.get("selected_checkpoint_tokens") != expected_tokens
        or pass_marker.get("selected_checkpoint_tokens") != expected_tokens
        or not isinstance(selected_checkpoint, str)
        or candidate.get("checkpoint_path") != selected_checkpoint
        or decision.get("selected_checkpoint_path") != selected_checkpoint
        or candidate.get("model_identity_sha256")
        != pass_marker.get("selected_model_identity_sha256")
    ):
        raise Day20PluginError("Day 20 winner identity drifted")

    metrics = candidate.get("metrics")
    if not isinstance(metrics, dict):
        raise Day20PluginError("Day 20 winner metrics are missing")
    required_metrics = set(PROMOTION_THRESHOLDS)
    if any(
        isinstance(metrics.get(field), bool)
        or not isinstance(metrics.get(field), int)
        or metrics[field] < 0
        for field in required_metrics
    ):
        raise Day20PluginError("Day 20 winner metrics are invalid")
    checks = {
        field: (
            metrics[field] == minimum
            if field == "infrastructure_failures"
            else metrics[field] >= minimum
        )
        for field, minimum in PROMOTION_THRESHOLDS.items()
    }
    if (
        metrics["total_correct"]
        != sum(
            metrics[f"{skill}_correct"]
            for skill in ("general", "math", "finance", "code")
        )
        or candidate.get("threshold_checks") != checks
        or not all(checks.values())
        or pass_marker.get("selected_metrics") != metrics
    ):
        raise Day20PluginError("Day 20 winner no longer satisfies promotion floors")

    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise Day20PluginError("Day 20 winner artifact identities are missing")
    verified_artifacts: dict[str, dict[str, str]] = {}
    for name, identity in sorted(artifacts.items()):
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise Day20PluginError(f"Day 20 winner artifact is invalid: {name}")
        path = Path(identity["path"])
        if not path.is_absolute() or run_root not in path.resolve().parents:
            raise Day20PluginError(f"Day 20 winner artifact escaped the run: {name}")
        verified_artifacts[name] = _verified_file_identity(
            identity, expected_path=path, label=f"Day 20 winner {name}"
        )

    summary_path = (run_root / "evidence/main/training-summary.json").resolve()
    checkpoint = verified_checkpoint_path(
        summary_path, name=winner, run_root=run_root
    )
    if str(checkpoint) != selected_checkpoint:
        raise Day20PluginError("PASS checkpoint differs from verified training evidence")
    summary = load_json(summary_path)
    training_identity = candidate.get("training_identity")
    training_config = summary.get("training_config")
    if (
        not isinstance(training_identity, dict)
        or not isinstance(training_config, dict)
        or training_identity.get("path") != str(summary_path)
        or training_identity.get("file_sha256") != file_sha256(summary_path)
        or training_identity.get("content_sha256") != summary.get("summary_sha256")
        or training_identity.get("training_config_path")
        != training_config.get("path")
        or training_identity.get("training_config_file_sha256")
        != training_config.get("file_sha256")
        or training_identity.get("learning_rate") != float(summary.get("learning_rate"))
    ):
        raise Day20PluginError("Day 20 winner training identity drifted")
    checkpoint_package = summary["checkpoint_packages"][winner]
    result = {
        "schema_version": 1,
        "domain": "day20.qwen35_merge_source_verification",
        "status": "pass",
        "run_root": str(run_root),
        "candidate": winner,
        "base_model": results.get("experiment_manifest", {}).get(
            "base_model_path"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_package": checkpoint_package,
        "pass_marker": {
            "path": str(pass_path),
            "file_sha256": file_sha256(pass_path),
            "content_sha256": pass_content_hash,
        },
        "results": {
            "path": str(results_path),
            "file_sha256": file_sha256(results_path),
            "content_sha256": results_content_hash,
        },
        "training_summary": {
            "path": str(summary_path),
            "file_sha256": file_sha256(summary_path),
            "content_sha256": summary["summary_sha256"],
        },
        "winner_artifacts": verified_artifacts,
    }
    if not isinstance(result["base_model"], str) or not Path(
        result["base_model"]
    ).is_absolute():
        raise Day20PluginError("Day 20 Base model identity is missing from results")
    result["verification_sha256"] = object_sha256(result)
    return result


def _direct_export_child(path: Path, *, run_root: Path, must_exist: bool) -> Path:
    run_root = run_root.resolve()
    exports = (run_root / "exports").resolve()
    if not exports.is_dir() or exports.is_symlink():
        raise Day20PluginError("Day 20 exports directory is invalid")
    if path.is_symlink():
        raise Day20PluginError("export directory must not be a symlink")
    resolved = path.resolve()
    if resolved.parent != exports:
        raise Day20PluginError("export directory must be a direct child of run exports")
    if must_exist and (not resolved.is_dir() or resolved.is_symlink()):
        raise Day20PluginError(f"export directory is missing or unsafe: {resolved}")
    return resolved


def _export_file_manifest(export_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(export_dir.rglob("*")):
        if path.is_symlink():
            raise Day20PluginError(f"export contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise Day20PluginError(f"export contains a non-regular entry: {path}")
        relative = path.relative_to(export_dir).as_posix()
        if relative == EXPORT_MANIFEST_FILE:
            continue
        if path.stat().st_size <= 0:
            raise Day20PluginError(f"export contains an empty file: {relative}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return files


def _load_export_json(export_dir: Path, relative: str) -> dict[str, Any]:
    path = export_dir / relative
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise Day20PluginError(f"export JSON is missing: {relative}")
    return load_json(path)


def _validate_safetensors(path: Path) -> tuple[dict[str, int | str], set[str]]:
    size = path.stat().st_size
    if size < 10:
        raise Day20PluginError(f"safetensors shard is truncated: {path.name}")
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little", signed=False)
        if header_size <= 1 or header_size > size - 8:
            raise Day20PluginError(f"safetensors header is invalid: {path.name}")
        try:
            header = json.loads(handle.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Day20PluginError(
                f"safetensors header JSON is invalid: {path.name}"
            ) from error
    if not isinstance(header, dict):
        raise Day20PluginError(f"safetensors header is not an object: {path.name}")
    payload_bytes = size - 8 - header_size
    tensors = 0
    tensor_bytes = 0
    tensor_names: set[str] = set()
    for name, value in header.items():
        if name == "__metadata__":
            continue
        offsets = value.get("data_offsets") if isinstance(value, dict) else None
        shape = value.get("shape") if isinstance(value, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value.get("dtype") if isinstance(value, dict) else None, str)
            or not isinstance(shape, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
        ):
            raise Day20PluginError(f"invalid tensor metadata in shard: {path.name}")
        tensors += 1
        tensor_bytes += offsets[1] - offsets[0]
        tensor_names.add(name)
    if tensors == 0:
        raise Day20PluginError(f"safetensors shard contains no tensors: {path.name}")
    return (
        {
            "bytes": size,
            "header_bytes": header_size,
            "tensor_bytes": tensor_bytes,
            "tensors": tensors,
            "tensor_names_sha256": object_sha256(sorted(tensor_names)),
        },
        tensor_names,
    )


def validate_hf_export(export_dir: Path) -> dict[str, Any]:
    files = _export_file_manifest(export_dir)
    config = _load_export_json(export_dir, "config.json")
    tokenizer_config = _load_export_json(export_dir, "tokenizer_config.json")
    architectures = config.get("architectures")
    if (
        config.get("model_type") != "qwen3_5"
        or not isinstance(architectures, list)
        or not architectures
        or any(not isinstance(item, str) or not item for item in architectures)
        or not any("Qwen3_5" in item for item in architectures)
        or not isinstance(tokenizer_config.get("tokenizer_class"), str)
        or not tokenizer_config["tokenizer_class"]
    ):
        raise Day20PluginError("merged Hugging Face config/tokenizer contract is incomplete")
    tokenizer_assets: list[str]
    if "tokenizer.json" in files:
        tokenizer = _load_export_json(export_dir, "tokenizer.json")
        if not isinstance(tokenizer.get("model"), dict) or not tokenizer["model"]:
            raise Day20PluginError("merged fast-tokenizer model is incomplete")
        tokenizer_assets = ["tokenizer.json"]
    elif {"vocab.json", "merges.txt"} <= set(files):
        if not _load_export_json(export_dir, "vocab.json"):
            raise Day20PluginError("merged tokenizer vocabulary is empty")
        tokenizer_assets = ["vocab.json", "merges.txt"]
    elif "tokenizer.model" in files:
        tokenizer_assets = ["tokenizer.model"]
    elif "spiece.model" in files:
        tokenizer_assets = ["spiece.model"]
    else:
        raise Day20PluginError("merged Hugging Face tokenizer assets are incomplete")

    shard_names = sorted(name for name in files if name.endswith(".safetensors"))
    if not shard_names:
        raise Day20PluginError("merged Hugging Face weights are missing")
    validated_shards = {
        name: _validate_safetensors(export_dir / name) for name in shard_names
    }
    shard_layout = {
        name: validation[0] for name, validation in validated_shards.items()
    }
    index_name = "model.safetensors.index.json"
    if index_name in files:
        index = _load_export_json(export_dir, index_name)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise Day20PluginError("merged Hugging Face weight index is empty")
        referenced: set[str] = set()
        for tensor, shard in weight_map.items():
            if (
                not isinstance(tensor, str)
                or not tensor
                or not isinstance(shard, str)
                or not shard
                or Path(shard).name != shard
            ):
                raise Day20PluginError("merged Hugging Face weight index is unsafe")
            referenced.add(shard)
        if referenced != set(shard_names):
            raise Day20PluginError("merged Hugging Face shard set differs from its index")
        indexed_names = {
            shard: {tensor for tensor, declared_shard in weight_map.items() if declared_shard == shard}
            for shard in shard_names
        }
        if any(
            indexed_names[shard] != validated_shards[shard][1]
            for shard in shard_names
        ):
            raise Day20PluginError("merged Hugging Face index tensor map drifted")
        total_size = index.get("metadata", {}).get("total_size")
        expected_total_size = sum(
            int(layout["tensor_bytes"]) for layout in shard_layout.values()
        )
        if (
            isinstance(total_size, bool)
            or not isinstance(total_size, int)
            or total_size != expected_total_size
        ):
            raise Day20PluginError("merged Hugging Face index total_size drifted")
    elif shard_names != ["model.safetensors"]:
        raise Day20PluginError("sharded Hugging Face weights require an index")
    return {
        "files": files,
        "files_sha256": object_sha256(files),
        "model_type": config["model_type"],
        "architectures": architectures,
        "tokenizer_assets": tokenizer_assets,
        "weight_index": index_name if index_name in files else None,
        "weight_shards": shard_layout,
    }


def seal_merged_export(
    export_dir: Path,
    *,
    published_dir: Path,
    run_root: Path,
    candidate: str,
) -> dict[str, Any]:
    actual = _direct_export_child(export_dir, run_root=run_root, must_exist=True)
    published = _direct_export_child(
        published_dir, run_root=run_root, must_exist=False
    )
    if published.name != f"{candidate}-merged" or published.exists():
        raise Day20PluginError("merged export publication target is invalid or occupied")
    source = validate_winner_for_export(run_root)
    if source["candidate"] != candidate:
        raise Day20PluginError("only the verified Day 20 winner may be merged")
    layout = validate_hf_export(actual)
    manifest = {
        "schema_version": 1,
        "domain": "day20.qwen35_merged_export_manifest",
        "status": "complete",
        "run_root": str(run_root.resolve()),
        "candidate": candidate,
        "export_dir": str(published),
        "source": source,
        "files": layout.pop("files"),
        "files_sha256": layout.pop("files_sha256"),
        "hf_layout": layout,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    write_json_new(actual / EXPORT_MANIFEST_FILE, manifest)
    return verify_merged_export(
        actual,
        run_root=run_root,
        published_dir=published,
        expected_candidate=candidate,
    )


def verify_merged_export(
    export_dir: Path,
    *,
    run_root: Path,
    published_dir: Path | None = None,
    expected_candidate: str | None = None,
) -> dict[str, Any]:
    actual = _direct_export_child(export_dir, run_root=run_root, must_exist=True)
    published = _direct_export_child(
        published_dir or actual, run_root=run_root, must_exist=False
    )
    manifest = load_json(actual / EXPORT_MANIFEST_FILE)
    _verified_self_hash(
        manifest, "manifest_sha256", label="merged export manifest"
    )
    source = validate_winner_for_export(run_root)
    candidate = source["candidate"]
    layout = validate_hf_export(actual)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("domain") != "day20.qwen35_merged_export_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("run_root") != str(run_root.resolve())
        or manifest.get("candidate") != candidate
        or (expected_candidate is not None and candidate != expected_candidate)
        or published.name != f"{candidate}-merged"
        or manifest.get("export_dir") != str(published)
        or manifest.get("source") != source
        or manifest.get("files") != layout["files"]
        or manifest.get("files_sha256") != layout["files_sha256"]
        or manifest.get("hf_layout")
        != {
            key: value
            for key, value in layout.items()
            if key not in {"files", "files_sha256"}
        }
    ):
        raise Day20PluginError("merged export package identity drifted")
    return manifest


def _update_ratio(model: Any, initial: dict[str, Any]) -> float:
    numerator = 0.0
    denominator = 0.0
    current = dict(model.named_parameters())
    for name, baseline in initial.items():
        parameter = current[name].detach().float().cpu()
        base_float = baseline.float()
        numerator += float((parameter - base_float).pow(2).sum().item())
        denominator += float(base_float.pow(2).sum().item())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30)


def _register_swift_callback() -> None:
    import torch
    from swift.callbacks import TrainerCallback, callbacks_map

    class Day20EvidenceCallback(TrainerCallback):
        def __init__(self, args: Any, trainer: Any) -> None:
            super().__init__(args, trainer)
            required = (
                "DAY20_PLUGIN_EVIDENCE_DIR",
                "DAY20_DATASET_PATH",
                "DAY20_EXPECTED_SUPERVISED_TOKENS",
                "DAY20_CHECKPOINT_TARGETS_JSON",
                "DAY20_TARGET_REGEX",
                "DAY20_RUN_KIND",
                "DAY20_EXPECTED_LR",
                "DAY20_TRAINING_CONFIG",
                "DAY20_EXPECTED_MS_SWIFT_COMMIT",
            )
            missing = [name for name in required if not os.environ.get(name)]
            if missing:
                raise RuntimeError(f"missing Day 20 callback environment: {missing}")
            self.evidence_dir = Path(os.environ["DAY20_PLUGIN_EVIDENCE_DIR"])
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.evidence_dir / "step-metrics.jsonl"
            self.inventory_path = self.evidence_dir / "trainable-inventory.json"
            self.token_reaudit_path = self.evidence_dir / "trainer-token-reaudit.json"
            self.gate_path = self.evidence_dir / "five-step-safety.json"
            self.runtime_path = self.evidence_dir / "runtime-identity.json"
            self.summary_path = self.evidence_dir / "training-summary.json"
            self.dataset = Path(os.environ["DAY20_DATASET_PATH"]).resolve()
            self.expected_tokens = int(
                os.environ["DAY20_EXPECTED_SUPERVISED_TOKENS"]
            )
            self.targets = json.loads(os.environ["DAY20_CHECKPOINT_TARGETS_JSON"])
            self.target_regex = os.environ["DAY20_TARGET_REGEX"]
            self.run_kind = os.environ["DAY20_RUN_KIND"]
            self.expected_lr = float(os.environ["DAY20_EXPECTED_LR"])
            self.runtime_identity = runtime_identity(
                os.environ["DAY20_EXPECTED_MS_SWIFT_COMMIT"]
            )
            if self.runtime_path.exists():
                if load_json(self.runtime_path) != self.runtime_identity:
                    raise RuntimeError("training runtime identity changed across resume")
            else:
                write_json_new(self.runtime_path, self.runtime_identity)
            self.training_config = Path(
                os.environ["DAY20_TRAINING_CONFIG"]
            ).resolve()
            if not self.training_config.is_file():
                raise RuntimeError("Day 20 training config is missing")
            config = load_json(self.training_config)
            config_training = config.get("training")
            config_data = config.get("data")
            if (
                not _self_hash_valid(config, "immutable_sha256")
                or config.get("run_kind") != self.run_kind
                or not isinstance(config_training, dict)
                or not isinstance(config_data, dict)
                or Path(str(config_data.get("path", ""))).resolve() != self.dataset
                or config_data.get("file_sha256") != file_sha256(self.dataset)
                or config_data.get("supervised_tokens") != self.expected_tokens
                or float(config_training.get("learning_rate")) != self.expected_lr
                or config_training.get("rank") != 8
                or config_training.get("alpha") != 16
                or config_training.get("dropout") != 0.05
                or config_training.get("bias") != "none"
                or config_training.get("target_regex") != self.target_regex
                or config_training.get("per_device_train_batch_size") != 2
                or config_training.get("gradient_accumulation_steps") != 4
                or config_training.get("packing") is not False
                or config_training.get("padding_free") is not False
            ):
                raise RuntimeError("Day 20 training config/environment drifted")
            self.step_tokens = supervised_token_schedule(self.dataset)
            self.target_steps = checkpoint_steps(self.step_tokens, self.targets)
            if sum(self.step_tokens) != self.expected_tokens:
                raise RuntimeError("callback supervised-token total drifted")
            self.recorded_steps = {
                int(row["global_step"])
                for row in load_jsonl(self.trace_path)
            } if self.trace_path.exists() else set()
            self.initial_trainable: dict[str, Any] = {}
            self.saved: dict[str, str] = {}

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            template = self.trainer.template
            model = self.trainer.model
            actual_contract = {
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
                "lr_scheduler_type": enum_value(args.lr_scheduler_type),
                "optim": enum_value(args.optim),
                # ms-swift enables checkpointing on the model, then deliberately
                # clears the HF Trainer flag to avoid applying it a second time.
                "trainer.gradient_checkpointing": args.gradient_checkpointing,
                "model.is_gradient_checkpointing": getattr(
                    model, "is_gradient_checkpointing", None
                ),
                "dataloader_drop_last": args.dataloader_drop_last,
                "group_by_length": args.group_by_length,
                "dataloader_num_workers": int(args.dataloader_num_workers),
                "world_size": int(args.world_size),
                "train_dataloader_shuffle": getattr(
                    args, "train_dataloader_shuffle", None
                ),
                "save_only_model": args.save_only_model,
                "save_total_limit": int(args.save_total_limit),
                "save_strategy": enum_value(args.save_strategy),
                "tuner_type": enum_value(getattr(args, "tuner_type", None)),
                "template.max_length": getattr(template, "max_length", None),
                "template.truncation_strategy": getattr(
                    template, "truncation_strategy", None
                ),
                "template.enable_thinking": getattr(
                    template, "enable_thinking", None
                ),
                "template.add_non_thinking_prefix": getattr(
                    template, "add_non_thinking_prefix", None
                ),
                "template.template_type": getattr(
                    getattr(template, "template_meta", None),
                    "template_type",
                    None,
                ),
            }
            expected_contract = {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "bf16": True,
                "learning_rate": self.expected_lr,
                "num_train_epochs": 1.0,
                "seed": 20260809,
                "data_seed": 20260809,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.05,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_epsilon": 1e-8,
                "lr_scheduler_type": "cosine",
                "optim": "adamw_torch",
                "trainer.gradient_checkpointing": False,
                "model.is_gradient_checkpointing": True,
                "dataloader_drop_last": False,
                "group_by_length": False,
                "dataloader_num_workers": 0,
                "world_size": 1,
                "train_dataloader_shuffle": False,
                "save_only_model": self.run_kind == "probe",
                "save_total_limit": 1 if self.run_kind == "probe" else 3,
                "save_strategy": "no",
                "tuner_type": "lora",
                "template.max_length": 2304,
                "template.truncation_strategy": "raise",
                "template.enable_thinking": False,
                "template.add_non_thinking_prefix": True,
                "template.template_type": "qwen3_5",
            }
            drift = {
                key: {"actual": actual_contract[key], "expected": expected}
                for key, expected in expected_contract.items()
                if actual_contract[key] != expected
            }
            if drift:
                raise RuntimeError(
                    "Day 20 pinned Trainer arguments drifted: "
                    + json.dumps(drift, sort_keys=True)
                )
            if load_json(self.runtime_path) != self.runtime_identity:
                raise RuntimeError("training runtime identity artifact drifted")
            token_reaudit = reaudit_prepared_dataset(self.dataset, template)
            if (
                token_reaudit["supervised_tokens"] != self.expected_tokens
                or token_reaudit["records"] != len(load_jsonl(self.dataset))
            ):
                raise RuntimeError("live Trainer token re-audit budget drifted")
            if self.token_reaudit_path.exists():
                if load_json(self.token_reaudit_path) != token_reaudit:
                    raise RuntimeError("live Trainer token re-audit changed across resume")
            else:
                write_json_new(self.token_reaudit_path, token_reaudit)
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            inventory = validate_trainable_names(
                trainable,
                self.target_regex,
                expected_module_counts=EXPECTED_LORA_MODULE_COUNTS,
            )
            if self.inventory_path.exists():
                previous = load_json(self.inventory_path)
                if previous.get("inventory_sha256") != inventory["inventory_sha256"]:
                    raise RuntimeError("trainable inventory changed across resume")
            else:
                write_json_new(self.inventory_path, inventory)
            self.initial_trainable = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            step = int(state.global_step)
            if step <= 0 or step > len(self.step_tokens):
                raise RuntimeError(f"unexpected optimizer step: {step}")
            if step in self.target_steps.values():
                control.should_save = True
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            logs = logs or {}
            step = int(state.global_step)
            if step <= 0 or step in self.recorded_steps or "loss" not in logs:
                return control
            ratio = None
            if step <= 5:
                ratio = _update_ratio(self.trainer.model, self.initial_trainable)
            row = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_optimizer_step",
                "global_step": step,
                "window_supervised_tokens": self.step_tokens[step - 1],
                "cumulative_supervised_tokens": sum(self.step_tokens[:step]),
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
                "lora_update_ratio": ratio,
            }
            append_jsonl(self.trace_path, row)
            self.recorded_steps.add(step)
            if step == 5 and not self.gate_path.exists():
                gate = validate_five_step_metrics(load_jsonl(self.trace_path))
                write_json_new(self.gate_path, gate)
            return control

        def on_save(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            step = int(state.global_step)
            names = [name for name, target_step in self.target_steps.items() if target_step == step]
            if names:
                checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
                if not checkpoint.is_dir():
                    raise RuntimeError(f"expected checkpoint was not written: {checkpoint}")
                integrity = build_checkpoint_integrity(
                    checkpoint,
                    run_kind=self.run_kind,
                    global_step=step,
                    cumulative_supervised_tokens=sum(self.step_tokens[:step]),
                    targets=names,
                    dataset_file_sha256=file_sha256(self.dataset),
                    training_config_file_sha256=file_sha256(self.training_config),
                    runtime_sha256=self.runtime_identity["runtime_sha256"],
                )
                integrity_path = checkpoint / CHECKPOINT_INTEGRITY_FILE
                if integrity_path.exists():
                    if verify_checkpoint_integrity(checkpoint) != integrity:
                        raise RuntimeError("checkpoint integrity changed across save")
                else:
                    write_json_new(integrity_path, integrity)
                for name in names:
                    self.saved[name] = str(checkpoint)
                append_jsonl(
                    self.evidence_dir / "checkpoint-events.jsonl",
                    {
                        "global_step": step,
                        "cumulative_supervised_tokens": sum(self.step_tokens[:step]),
                        "targets": names,
                        "checkpoint": str(checkpoint),
                    },
                )
            return control

        def on_train_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if int(state.global_step) != len(self.step_tokens):
                raise RuntimeError("training ended before the supervised-token budget")
            if self.run_kind == "probe" and not self.gate_path.is_file():
                raise RuntimeError("probe ended without the five-step safety gate")
            events = (
                load_jsonl(self.evidence_dir / "checkpoint-events.jsonl")
                if (self.evidence_dir / "checkpoint-events.jsonl").exists()
                else []
            )
            saved: dict[str, str] = {}
            for event in events:
                for name in event.get("targets", []):
                    saved[name] = str(event.get("checkpoint"))
            for name, step in self.target_steps.items():
                checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
                if checkpoint.is_dir():
                    saved.setdefault(name, str(checkpoint))
            if set(saved) != set(self.targets):
                raise RuntimeError(
                    f"checkpoint evidence mismatch: expected {sorted(self.targets)}, got {sorted(saved)}"
                )
            checkpoint_packages: dict[str, Any] = {}
            for name, checkpoint_string in saved.items():
                checkpoint = Path(checkpoint_string)
                integrity = verify_checkpoint_integrity(checkpoint)
                expected_checkpoint_targets = sorted(
                    candidate_name
                    for candidate_name, candidate_step in self.target_steps.items()
                    if candidate_step == self.target_steps[name]
                )
                if (
                    integrity.get("run_kind") != self.run_kind
                    or integrity.get("targets") != expected_checkpoint_targets
                    or integrity.get("global_step") != self.target_steps[name]
                    or integrity.get("dataset_file_sha256") != file_sha256(self.dataset)
                    or integrity.get("training_config_file_sha256")
                    != file_sha256(self.training_config)
                    or integrity.get("runtime_sha256")
                    != self.runtime_identity["runtime_sha256"]
                ):
                    raise RuntimeError(f"checkpoint lineage is incomplete: {name}")
                checkpoint_packages[name] = {
                    "path": str(checkpoint.resolve()),
                    "integrity_file_sha256": file_sha256(
                        checkpoint / CHECKPOINT_INTEGRITY_FILE
                    ),
                    "integrity_sha256": integrity["integrity_sha256"],
                    "snapshot_sha256": integrity["snapshot_sha256"],
                    "files": integrity["files"],
                    "resumable": integrity["resumable"],
                }
            trace_rows = load_jsonl(self.trace_path)
            if [row.get("global_step") for row in trace_rows] != list(
                range(1, len(self.step_tokens) + 1)
            ):
                raise RuntimeError("optimizer step metrics are incomplete or out of order")
            summary = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_training_summary",
                "status": "pass",
                "run_kind": self.run_kind,
                "learning_rate": self.expected_lr,
                "runtime_identity": self.runtime_identity,
                "runtime_identity_artifact": {
                    "path": str(self.runtime_path.resolve()),
                    "file_sha256": file_sha256(self.runtime_path),
                    "content_sha256": self.runtime_identity["runtime_sha256"],
                },
                "training_config": {
                    "path": str(self.training_config),
                    "file_sha256": file_sha256(self.training_config),
                },
                "dataset": str(self.dataset),
                "dataset_file_sha256": file_sha256(self.dataset),
                "optimizer_steps": len(self.step_tokens),
                "cumulative_supervised_tokens": self.expected_tokens,
                "checkpoint_tokens": self.targets,
                "checkpoint_steps": self.target_steps,
                "checkpoint_actual_tokens": {
                    name: sum(self.step_tokens[:step])
                    for name, step in self.target_steps.items()
                },
                "checkpoints": saved,
                "checkpoint_packages": checkpoint_packages,
                "trainable_inventory": {
                    "path": str(self.inventory_path.resolve()),
                    "file_sha256": file_sha256(self.inventory_path),
                },
                "trainer_token_reaudit": {
                    "path": str(self.token_reaudit_path.resolve()),
                    "file_sha256": file_sha256(self.token_reaudit_path),
                    "content_sha256": load_json(self.token_reaudit_path)["audit_sha256"],
                },
                "step_metrics": {
                    "path": str(self.trace_path.resolve()),
                    "file_sha256": file_sha256(self.trace_path),
                    "records": len(trace_rows),
                },
                "five_step_safety": (
                    {
                        "path": str(self.gate_path.resolve()),
                        "file_sha256": file_sha256(self.gate_path),
                    }
                    if self.gate_path.exists()
                    else None
                ),
            }
            summary["summary_sha256"] = object_sha256(summary)
            write_json_new(self.summary_path, summary)
            return control

    callbacks_map["day20_evidence"] = Day20EvidenceCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset-kind", required=True, choices=("probe", "main"))
    validate.add_argument("--learning-rate", required=True)

    checkpoint = subparsers.add_parser("checkpoint-path")
    checkpoint.add_argument("--summary", required=True, type=Path)
    checkpoint.add_argument("--name", required=True)
    checkpoint.add_argument("--run-root", required=True, type=Path)

    resume = subparsers.add_parser("resume-checkpoint")
    resume.add_argument("--output-dir", required=True, type=Path)
    resume.add_argument("--run-root", required=True, type=Path)

    stage_resume = subparsers.add_parser("stage-resume-evidence")
    stage_resume.add_argument("--source-evidence", required=True, type=Path)
    stage_resume.add_argument("--destination-evidence", required=True, type=Path)
    stage_resume.add_argument("--checkpoint", required=True, type=Path)
    stage_resume.add_argument("--run-root", required=True, type=Path)

    winner = subparsers.add_parser("verify-winner")
    winner.add_argument("--run-root", required=True, type=Path)

    seal_export = subparsers.add_parser("seal-export")
    seal_export.add_argument("--run-root", required=True, type=Path)
    seal_export.add_argument("--export-dir", required=True, type=Path)
    seal_export.add_argument("--published-dir", required=True, type=Path)
    seal_export.add_argument("--candidate", required=True)

    verify_export = subparsers.add_parser("verify-export")
    verify_export.add_argument("--run-root", required=True, type=Path)
    verify_export.add_argument("--export-dir", required=True, type=Path)
    verify_export.add_argument("--candidate", required=True)

    resolve = subparsers.add_parser("resolve-main-config")
    resolve.add_argument("--manifest", required=True, type=Path)
    resolve.add_argument("--selection", required=True, type=Path)
    resolve.add_argument("--output", required=True, type=Path)

    verify_selection = subparsers.add_parser("verify-selection")
    verify_selection.add_argument("--manifest", required=True, type=Path)
    verify_selection.add_argument("--selection", required=True, type=Path)

    selection = subparsers.add_parser("select-probe")
    selection.add_argument("--day20-manifest", required=True, type=Path)
    selection.add_argument("--eval-manifest", required=True, type=Path)
    selection.add_argument("--base-summary", required=True, type=Path)
    selection.add_argument("--base-e2b-summary", required=True, type=Path)
    selection.add_argument(
        "--probe", action="append", nargs=3, metavar=("LR", "SUMMARY", "E2B_SUMMARY"), required=True
    )
    selection.add_argument("--output", required=True, type=Path)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-inputs":
        result = validate_prepared_inputs(
            args.manifest,
            dataset_kind=args.dataset_kind,
            learning_rate=args.learning_rate,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "checkpoint-path":
        checkpoint = verified_checkpoint_path(
            args.summary, name=args.name, run_root=args.run_root
        )
        print(str(checkpoint))
    elif args.command == "resume-checkpoint":
        checkpoint = latest_resumable_checkpoint(
            args.output_dir, run_root=args.run_root
        )
        print(str(checkpoint))
    elif args.command == "stage-resume-evidence":
        result = stage_resume_evidence(
            args.source_evidence,
            args.destination_evidence,
            checkpoint=args.checkpoint,
            run_root=args.run_root,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "verify-winner":
        result = validate_winner_for_export(args.run_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "seal-export":
        result = seal_merged_export(
            args.export_dir,
            published_dir=args.published_dir,
            run_root=args.run_root,
            candidate=args.candidate,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "verify-export":
        result = verify_merged_export(
            args.export_dir,
            run_root=args.run_root,
            expected_candidate=args.candidate,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "resolve-main-config":
        result = resolve_main_config(args.manifest, args.selection)
        write_json_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "verify-selection":
        result = validate_probe_selection(args.manifest, args.selection)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "select-probe":
        probes = [
            (item[0], Path(item[1]), Path(item[2])) for item in args.probe
        ]
        result = select_probe(
            day20_manifest=args.day20_manifest,
            eval_manifest=args.eval_manifest,
            base_summary=args.base_summary,
            base_e2b_summary=args.base_e2b_summary,
            probes=probes,
        )
        write_json_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if result["status"] != "selected":
            raise SystemExit(2)


if os.environ.get("DAY20_REGISTER_SWIFT_CALLBACK") == "1":
    _register_swift_callback()


if __name__ == "__main__":
    main()
