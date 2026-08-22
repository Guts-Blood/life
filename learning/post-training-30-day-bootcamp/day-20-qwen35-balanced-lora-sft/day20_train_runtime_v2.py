#!/usr/bin/env python3
"""Stdlib-only Day 20 v2 training input and checkpoint contract.

The actual ms-swift callback remains a GPU-side concern.  This module gives
local preparation and the remote runner one fail-closed contract for immutable
inputs, token-based checkpoint steps, candidate identity, and resumability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from day20_candidate_factory_v2 import (
    CONFIRMATION_SEED,
    CONTRACT_VERSION,
    MAIN_CHECKPOINT_TOKENS,
    PRIMARY_SEED,
    PROBE_CHECKPOINT_TOKENS,
    candidate_id,
    canonical_lr,
)
from day20_ordering_v2 import SKILLS, audit_temporal_mix


MANIFEST_DOMAIN = "day20.qwen35_balanced_lora.experiment_manifest.v2"
CONFIG_DOMAIN = "day20.qwen35_balanced_lora.training_config.v2"
CHECKPOINT_DOMAIN = "day20.qwen35_lora_checkpoint_integrity.v2"
CHECKPOINT_INTEGRITY_FILE = "day20-v2-checkpoint-integrity.json"
GLOBAL_BATCH_SIZE = 8
EXPECTED_TOKENS = {"probe": 24_000, "main": 320_000}
CHECKPOINT_TARGETS = {
    "probe": PROBE_CHECKPOINT_TOKENS,
    "main": MAIN_CHECKPOINT_TOKENS,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class Day20TrainingRuntimeV2Error(ValueError):
    """A v2 training input, checkpoint, or path invariant failed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20TrainingRuntimeV2Error(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20TrainingRuntimeV2Error(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day20TrainingRuntimeV2Error(f"JSON root must be an object: {path}")
    return value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    expected = payload.get(field)
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        raise Day20TrainingRuntimeV2Error(f"{field} is missing or invalid")
    actual = object_sha256({key: value for key, value in payload.items() if key != field})
    if actual != expected:
        raise Day20TrainingRuntimeV2Error(f"{field} mismatch")
    return expected


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Day20TrainingRuntimeV2Error(f"{label} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Day20TrainingRuntimeV2Error(f"{label} must be a positive integer")
    return value


def _run_kind(value: str) -> str:
    if value not in EXPECTED_TOKENS:
        raise Day20TrainingRuntimeV2Error("run_kind must be probe or main")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or value not in {PRIMARY_SEED, CONFIRMATION_SEED}:
        raise Day20TrainingRuntimeV2Error("seed is outside the v2 allowlist")
    return value


def _resolved_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Day20TrainingRuntimeV2Error(f"{label} path is missing")
    raw = Path(value)
    if not raw.is_absolute():
        raise Day20TrainingRuntimeV2Error(f"{label} path must be absolute")
    return raw.resolve()


def _learning_rate(value: str) -> str:
    try:
        return canonical_lr(value)
    except ValueError as error:
        raise Day20TrainingRuntimeV2Error(str(error)) from error


def supervised_token_schedule(
    dataset: Path, *, global_batch_size: int = GLOBAL_BATCH_SIZE
) -> list[int]:
    """Sum supervised tokens for each consecutive optimizer batch of 8 rows."""
    if global_batch_size != GLOBAL_BATCH_SIZE:
        raise Day20TrainingRuntimeV2Error("v2 global batch size is frozen at 8")
    rows: list[int] = []
    try:
        with dataset.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day20TrainingRuntimeV2Error(
                        f"blank JSONL row at line {line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Day20TrainingRuntimeV2Error(
                        f"invalid JSONL row at line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise Day20TrainingRuntimeV2Error(
                        f"JSONL row {line_number} must be an object"
                    )
                rows.append(
                    _require_positive_int(
                        row.get("qwen35_supervised_tokens"),
                        f"qwen35_supervised_tokens at line {line_number}",
                    )
                )
    except OSError as error:
        raise Day20TrainingRuntimeV2Error(
            f"cannot read training dataset {dataset}: {error}"
        ) from error
    if not rows:
        raise Day20TrainingRuntimeV2Error("training dataset is empty")
    return [
        sum(rows[start : start + global_batch_size])
        for start in range(0, len(rows), global_batch_size)
    ]


def checkpoint_steps(
    step_tokens: Iterable[int], *, run_kind: str
) -> dict[str, Any]:
    """Map every token milestone to its first reachable optimizer step."""
    kind = _run_kind(run_kind)
    targets = CHECKPOINT_TARGETS[kind]
    result: dict[str, int] = {}
    actual: dict[str, int] = {}
    cumulative = 0
    ordered_targets = list(targets.items())
    target_index = 0
    steps = list(step_tokens)
    for step, count in enumerate(steps, 1):
        cumulative += _require_positive_int(count, f"step {step} supervised tokens")
        while target_index < len(ordered_targets):
            name, target = ordered_targets[target_index]
            if cumulative < target:
                break
            result[name] = step
            actual[name] = cumulative
            target_index += 1
    if target_index != len(ordered_targets):
        missing = [name for name, _ in ordered_targets[target_index:]]
        raise Day20TrainingRuntimeV2Error(
            f"dataset does not reach checkpoint targets: {missing}"
        )
    if cumulative != EXPECTED_TOKENS[kind]:
        raise Day20TrainingRuntimeV2Error(
            f"{kind} dataset token total {cumulative} differs from "
            f"{EXPECTED_TOKENS[kind]}"
        )
    return {
        "run_kind": kind,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "optimizer_steps": len(steps),
        "supervised_tokens": cumulative,
        "checkpoint_tokens": dict(targets),
        "checkpoint_steps": result,
        "checkpoint_actual_tokens": actual,
    }


def _validate_config_common(
    config: Mapping[str, Any],
    *,
    run_kind: str,
    dataset: Path,
    dataset_sha256: str,
    temporal_mix_sha256: str,
    expected_tokens: int,
    seed: int,
    learning_rate: str,
    allow_main_placeholder: bool,
) -> bool:
    _self_hash(config, "immutable_sha256")
    if config.get("schema_version") != 2 or config.get("domain") != CONFIG_DOMAIN:
        raise Day20TrainingRuntimeV2Error("v2 training config identity drifted")
    if config.get("run_kind") != run_kind:
        raise Day20TrainingRuntimeV2Error("training config run_kind drifted")
    data = config.get("data")
    training = config.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise Day20TrainingRuntimeV2Error("training config data/training block is missing")
    if (
        _resolved_path(data.get("path"), "config dataset") != dataset
        or data.get("file_sha256") != dataset_sha256
        or data.get("supervised_tokens") != expected_tokens
        or data.get("temporal_mix_sha256") != temporal_mix_sha256
    ):
        raise Day20TrainingRuntimeV2Error("training config dataset identity drifted")
    if (
        training.get("seed") != seed
        or training.get("data_seed") != seed
        or training.get("global_batch_size") != GLOBAL_BATCH_SIZE
    ):
        raise Day20TrainingRuntimeV2Error("training config seed/batch identity drifted")
    expected_checkpoints = list(CHECKPOINT_TARGETS[run_kind].values())
    if (
        training.get("checkpoint_supervised_tokens") != expected_checkpoints
        or training.get("checkpoint_policy") != "all_milestones_resumable_v2"
    ):
        raise Day20TrainingRuntimeV2Error("training config checkpoint policy drifted")
    configured_lr = training.get("learning_rate")
    placeholder = run_kind == "main" and configured_lr == "__SELECT_FROM_PASSING_PROBE__"
    if placeholder and allow_main_placeholder:
        return True
    try:
        matches = Decimal(str(configured_lr)) == Decimal(learning_rate)
    except Exception as error:
        raise Day20TrainingRuntimeV2Error("training config learning rate is invalid") from error
    if not matches:
        raise Day20TrainingRuntimeV2Error("training config learning rate drifted")
    return False


def validate_inputs(
    manifest_path: Path,
    *,
    dataset_path: Path,
    config_path: Path,
    run_kind: str,
    seed: int,
    learning_rate: str,
) -> dict[str, Any]:
    """Validate one v2 manifest/data/config tuple and derive its milestones."""
    kind = _run_kind(run_kind)
    run_seed = _seed(seed)
    lr = _learning_rate(learning_rate)
    manifest_path = manifest_path.resolve()
    dataset_path = dataset_path.resolve()
    config_path = config_path.resolve()
    manifest = load_json(manifest_path)
    manifest_sha256 = _self_hash(manifest, "manifest_sha256")
    if manifest.get("schema_version") != 2 or manifest.get("domain") != MANIFEST_DOMAIN:
        raise Day20TrainingRuntimeV2Error("v2 experiment manifest identity drifted")
    run_root = _resolved_path(manifest.get("run_root"), "manifest run_root")
    if manifest_path.parent != run_root:
        raise Day20TrainingRuntimeV2Error("manifest path differs from its run_root")
    contract = manifest.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("candidate_factory_version") != CONTRACT_VERSION
    ):
        raise Day20TrainingRuntimeV2Error("candidate-factory contract version drifted")

    datasets = manifest.get("datasets")
    identity = datasets.get(kind) if isinstance(datasets, dict) else None
    expected_tokens = EXPECTED_TOKENS[kind]
    if not isinstance(identity, dict):
        raise Day20TrainingRuntimeV2Error(f"manifest {kind} dataset is missing")
    dataset_hash = file_sha256(dataset_path)
    if (
        _resolved_path(identity.get("path"), f"manifest {kind} dataset") != dataset_path
        or identity.get("file_sha256") != dataset_hash
        or identity.get("supervised_tokens") != expected_tokens
    ):
        raise Day20TrainingRuntimeV2Error(f"manifest {kind} dataset identity drifted")
    schedule = supervised_token_schedule(dataset_path)
    row_tokens: list[int] = []
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
            row_tokens.append(int(row["qwen35_supervised_tokens"]))
    if identity.get("ordered_supervised_tokens") != row_tokens:
        raise Day20TrainingRuntimeV2Error("manifest ordered token evidence drifted")
    if "records" in identity and identity.get("records") != len(row_tokens):
        raise Day20TrainingRuntimeV2Error("manifest dataset record count drifted")
    milestone_schedule = checkpoint_steps(schedule, run_kind=kind)
    temporal_mix = identity.get("temporal_mix_audit")
    temporal_mix_sha256 = _require_hash(
        identity.get("temporal_mix_sha256"), "temporal_mix_sha256"
    )
    try:
        recomputed_temporal_mix = audit_temporal_mix(
            rows, global_batch_size=GLOBAL_BATCH_SIZE
        )
    except ValueError as error:
        raise Day20TrainingRuntimeV2Error(
            f"cannot recompute temporal-mix audit: {error}"
        ) from error
    if (
        not isinstance(temporal_mix, dict)
        or object_sha256(temporal_mix) != temporal_mix_sha256
        or temporal_mix != recomputed_temporal_mix
        or recomputed_temporal_mix.get("status") != "pass"
    ):
        raise Day20TrainingRuntimeV2Error("manifest temporal-mix audit drifted")

    configs = manifest.get("configs")
    if not isinstance(configs, dict):
        raise Day20TrainingRuntimeV2Error("manifest configs block is missing")
    if kind == "probe":
        probes = configs.get("probes")
        config_identity = probes.get(lr) if isinstance(probes, dict) else None
    else:
        config_identity = configs.get("main_template")
    if not isinstance(config_identity, dict):
        raise Day20TrainingRuntimeV2Error(f"manifest {kind} config identity is missing")

    config = load_json(config_path)
    config_file_hash = file_sha256(config_path)
    exact_manifest_config = (
        _resolved_path(config_identity.get("path"), f"manifest {kind} config")
        == config_path
        and config_identity.get("file_sha256") == config_file_hash
    )
    if kind == "probe" and not exact_manifest_config:
        raise Day20TrainingRuntimeV2Error("probe config differs from manifest identity")
    if kind == "main" and not exact_manifest_config:
        parent = config.get("parent_main_template")
        if not isinstance(parent, dict) or any(
            parent.get(field) != config_identity.get(field)
            for field in ("path", "file_sha256", "immutable_sha256")
        ):
            raise Day20TrainingRuntimeV2Error(
                "resolved main config is not bound to the manifest template"
            )
        template_path = _resolved_path(config_identity.get("path"), "main template")
        if file_sha256(template_path) != config_identity.get("file_sha256"):
            raise Day20TrainingRuntimeV2Error("main template file identity drifted")
        template = load_json(template_path)
        if _self_hash(template, "immutable_sha256") != config_identity.get(
            "immutable_sha256"
        ):
            raise Day20TrainingRuntimeV2Error("main template immutable identity drifted")

    requires_resolution = _validate_config_common(
        config,
        run_kind=kind,
        dataset=dataset_path,
        dataset_sha256=dataset_hash,
        temporal_mix_sha256=temporal_mix_sha256,
        expected_tokens=expected_tokens,
        seed=run_seed,
        learning_rate=lr,
        allow_main_placeholder=exact_manifest_config,
    )
    if config_identity.get("immutable_sha256") is not None and exact_manifest_config:
        if config.get("immutable_sha256") != config_identity.get("immutable_sha256"):
            raise Day20TrainingRuntimeV2Error("manifest config immutable identity drifted")
    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": file_sha256(manifest_path),
        "dataset": str(dataset_path),
        "dataset_file_sha256": dataset_hash,
        "temporal_mix_sha256": temporal_mix_sha256,
        "training_config": str(config_path),
        "training_config_file_sha256": config_file_hash,
        "run_kind": kind,
        "seed": run_seed,
        "learning_rate": lr,
        "config_requires_resolution": requires_resolution,
        **milestone_schedule,
    }


def require_within_adapters(path: Path, run_root: Path) -> Path:
    resolved = path.resolve()
    adapters = (run_root.resolve() / "adapters").resolve()
    if resolved == adapters or adapters not in resolved.parents:
        raise Day20TrainingRuntimeV2Error(
            "path is outside the current run_root/adapters tree"
        )
    return resolved


def checkpoint_file_manifest(checkpoint: Path) -> dict[str, dict[str, Any]]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise Day20TrainingRuntimeV2Error(
            f"checkpoint directory is missing or symbolic: {checkpoint}"
        )
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise Day20TrainingRuntimeV2Error(
                f"checkpoint package contains a symbolic link: {path}"
            )
        if path.is_file() and path.name != CHECKPOINT_INTEGRITY_FILE:
            files[path.relative_to(checkpoint).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return files


def _validate_checkpoint_files(files: Mapping[str, Any]) -> None:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    }
    if not required <= set(files) or not any(
        Path(name).name.startswith("rng_state")
        and name.endswith(".pth")
        for name in files
    ):
        raise Day20TrainingRuntimeV2Error(
            "checkpoint is not resumable: adapter/trainer/optimizer/scheduler/RNG files are required"
        )


def build_checkpoint_integrity(
    checkpoint: Path,
    *,
    run_root: Path,
    run_kind: str,
    seed: int,
    learning_rate: str,
    global_step: int,
    cumulative_supervised_tokens: int,
    checkpoint_targets: Iterable[str],
    dataset_file_sha256: str,
    training_config_file_sha256: str,
    experiment_manifest_sha256: str,
    temporal_mix_sha256: str,
    runtime_sha256: str,
) -> dict[str, Any]:
    """Build one immutable, always-resumable v2 checkpoint package identity."""
    checkpoint = require_within_adapters(checkpoint, run_root)
    kind = _run_kind(run_kind)
    run_seed = _seed(seed)
    lr = _learning_rate(learning_rate)
    step = _require_positive_int(global_step, "global_step")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or int(match.group(1)) != step:
        raise Day20TrainingRuntimeV2Error("checkpoint directory/global_step mismatch")
    cumulative = _require_positive_int(
        cumulative_supervised_tokens, "cumulative_supervised_tokens"
    )
    if cumulative > EXPECTED_TOKENS[kind]:
        raise Day20TrainingRuntimeV2Error("checkpoint token count exceeds run budget")
    names = list(checkpoint_targets)
    if not names or len(set(names)) != len(names):
        raise Day20TrainingRuntimeV2Error("checkpoint targets must be non-empty and unique")
    unknown = set(names) - set(CHECKPOINT_TARGETS[kind])
    if unknown:
        raise Day20TrainingRuntimeV2Error(f"unknown checkpoint targets: {sorted(unknown)}")
    names.sort(key=CHECKPOINT_TARGETS[kind].__getitem__)
    if any(CHECKPOINT_TARGETS[kind][name] > cumulative for name in names):
        raise Day20TrainingRuntimeV2Error("checkpoint target exceeds cumulative tokens")
    files = checkpoint_file_manifest(checkpoint)
    _validate_checkpoint_files(files)
    candidate_ids = {
        name: candidate_id(
            run_kind=kind,
            seed=run_seed,
            learning_rate=lr,
            checkpoint=name,
        )
        for name in names
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "domain": CHECKPOINT_DOMAIN,
        "status": "complete",
        "checkpoint": str(checkpoint),
        "run_kind": kind,
        "seed": run_seed,
        "learning_rate": lr,
        "global_step": step,
        "cumulative_supervised_tokens": cumulative,
        "checkpoint_targets": names,
        "candidate_ids": candidate_ids,
        "candidate_identity_sha256": object_sha256(candidate_ids),
        "resumable": True,
        "dataset_file_sha256": _require_hash(
            dataset_file_sha256, "dataset_file_sha256"
        ),
        "training_config_file_sha256": _require_hash(
            training_config_file_sha256, "training_config_file_sha256"
        ),
        "experiment_manifest_sha256": _require_hash(
            experiment_manifest_sha256, "experiment_manifest_sha256"
        ),
        "temporal_mix_sha256": _require_hash(
            temporal_mix_sha256, "temporal_mix_sha256"
        ),
        "runtime_sha256": _require_hash(runtime_sha256, "runtime_sha256"),
        "files": files,
        "snapshot_sha256": object_sha256(files),
    }
    payload["integrity_sha256"] = object_sha256(payload)
    return payload


def write_checkpoint_integrity(checkpoint: Path, payload: Mapping[str, Any]) -> Path:
    marker = checkpoint.resolve() / CHECKPOINT_INTEGRITY_FILE
    if marker.exists():
        raise Day20TrainingRuntimeV2Error(f"refusing to overwrite checkpoint marker: {marker}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{marker.name}.", dir=marker.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return marker


def verify_checkpoint_integrity(
    checkpoint: Path,
    *,
    run_root: Path,
    run_kind: str | None = None,
    seed: int | None = None,
    learning_rate: str | None = None,
    dataset_file_sha256: str | None = None,
    training_config_file_sha256: str | None = None,
    experiment_manifest_sha256: str | None = None,
    temporal_mix_sha256: str | None = None,
    runtime_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = require_within_adapters(checkpoint, run_root)
    marker = load_json(checkpoint / CHECKPOINT_INTEGRITY_FILE)
    _self_hash(marker, "integrity_sha256")
    if (
        marker.get("schema_version") != 2
        or marker.get("domain") != CHECKPOINT_DOMAIN
        or marker.get("status") != "complete"
        or marker.get("checkpoint") != str(checkpoint)
        or marker.get("resumable") is not True
    ):
        raise Day20TrainingRuntimeV2Error("checkpoint marker identity drifted")
    rebuilt = build_checkpoint_integrity(
        checkpoint,
        run_root=run_root,
        run_kind=str(marker.get("run_kind")),
        seed=marker.get("seed"),
        learning_rate=str(marker.get("learning_rate")),
        global_step=marker.get("global_step"),
        cumulative_supervised_tokens=marker.get("cumulative_supervised_tokens"),
        checkpoint_targets=marker.get("checkpoint_targets") or [],
        dataset_file_sha256=str(marker.get("dataset_file_sha256")),
        training_config_file_sha256=str(marker.get("training_config_file_sha256")),
        experiment_manifest_sha256=str(marker.get("experiment_manifest_sha256")),
        temporal_mix_sha256=str(marker.get("temporal_mix_sha256")),
        runtime_sha256=str(marker.get("runtime_sha256")),
    )
    if rebuilt != marker:
        raise Day20TrainingRuntimeV2Error("checkpoint package integrity drifted")
    if run_kind is not None and marker["run_kind"] != _run_kind(run_kind):
        raise Day20TrainingRuntimeV2Error("checkpoint run_kind differs from expectation")
    if seed is not None and marker["seed"] != _seed(seed):
        raise Day20TrainingRuntimeV2Error("checkpoint seed differs from expectation")
    if learning_rate is not None and marker["learning_rate"] != _learning_rate(
        learning_rate
    ):
        raise Day20TrainingRuntimeV2Error(
            "checkpoint learning rate differs from expectation"
        )
    expected_hashes = {
        "dataset_file_sha256": dataset_file_sha256,
        "training_config_file_sha256": training_config_file_sha256,
        "experiment_manifest_sha256": experiment_manifest_sha256,
        "temporal_mix_sha256": temporal_mix_sha256,
        "runtime_sha256": runtime_sha256,
    }
    for field, expected in expected_hashes.items():
        if expected is not None and marker[field] != _require_hash(expected, field):
            raise Day20TrainingRuntimeV2Error(
                f"checkpoint {field} differs from expectation"
            )
    return marker


def latest_resumable_checkpoint(
    output_dir: Path,
    *,
    run_root: Path,
    run_kind: str | None = None,
    seed: int | None = None,
    learning_rate: str | None = None,
    dataset_file_sha256: str | None = None,
    training_config_file_sha256: str | None = None,
    experiment_manifest_sha256: str | None = None,
    temporal_mix_sha256: str | None = None,
    runtime_sha256: str | None = None,
) -> Path:
    """Return the newest verified probe or main checkpoint under adapters."""
    output = require_within_adapters(output_dir, run_root)
    candidates: list[tuple[int, Path]] = []
    if not output.is_dir():
        raise Day20TrainingRuntimeV2Error(f"checkpoint output is missing: {output}")
    for path in output.iterdir():
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is not None and path.is_dir():
            candidates.append((int(match.group(1)), path))
    for _, checkpoint in sorted(candidates, reverse=True):
        try:
            verify_checkpoint_integrity(
                checkpoint,
                run_root=run_root,
                run_kind=run_kind,
                seed=seed,
                learning_rate=learning_rate,
                dataset_file_sha256=dataset_file_sha256,
                training_config_file_sha256=training_config_file_sha256,
                experiment_manifest_sha256=experiment_manifest_sha256,
                temporal_mix_sha256=temporal_mix_sha256,
                runtime_sha256=runtime_sha256,
            )
        except (Day20TrainingRuntimeV2Error, OSError, TypeError, ValueError):
            continue
        return checkpoint.resolve()
    raise Day20TrainingRuntimeV2Error("no complete resumable v2 checkpoint was found")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset", required=True, type=Path)
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--run-kind", required=True, choices=("probe", "main"))
    validate.add_argument("--seed", required=True, type=int)
    validate.add_argument("--learning-rate", required=True)

    steps = subparsers.add_parser("checkpoint-steps")
    steps.add_argument("--dataset", required=True, type=Path)
    steps.add_argument("--run-kind", required=True, choices=("probe", "main"))

    verify = subparsers.add_parser("verify-checkpoint")
    verify.add_argument("--checkpoint", required=True, type=Path)
    verify.add_argument("--run-root", required=True, type=Path)
    verify.add_argument("--run-kind", choices=("probe", "main"))
    verify.add_argument("--seed", type=int)
    verify.add_argument("--learning-rate")
    verify.add_argument("--dataset-sha256")
    verify.add_argument("--config-sha256")
    verify.add_argument("--manifest-sha256")
    verify.add_argument("--temporal-mix-sha256")
    verify.add_argument("--runtime-sha256")

    latest = subparsers.add_parser("latest-resumable")
    latest.add_argument("--output-dir", required=True, type=Path)
    latest.add_argument("--run-root", required=True, type=Path)
    latest.add_argument("--run-kind", choices=("probe", "main"))
    latest.add_argument("--seed", type=int)
    latest.add_argument("--learning-rate")
    latest.add_argument("--dataset-sha256")
    latest.add_argument("--config-sha256")
    latest.add_argument("--manifest-sha256")
    latest.add_argument("--temporal-mix-sha256")
    latest.add_argument("--runtime-sha256")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "validate-inputs":
            result: Any = validate_inputs(
                args.manifest,
                dataset_path=args.dataset,
                config_path=args.config,
                run_kind=args.run_kind,
                seed=args.seed,
                learning_rate=args.learning_rate,
            )
        elif args.command == "checkpoint-steps":
            result = checkpoint_steps(
                supervised_token_schedule(args.dataset), run_kind=args.run_kind
            )
        elif args.command == "verify-checkpoint":
            result = verify_checkpoint_integrity(
                args.checkpoint,
                run_root=args.run_root,
                run_kind=args.run_kind,
                seed=args.seed,
                learning_rate=args.learning_rate,
                dataset_file_sha256=args.dataset_sha256,
                training_config_file_sha256=args.config_sha256,
                experiment_manifest_sha256=args.manifest_sha256,
                temporal_mix_sha256=args.temporal_mix_sha256,
                runtime_sha256=args.runtime_sha256,
            )
        else:
            print(
                latest_resumable_checkpoint(
                    args.output_dir,
                    run_root=args.run_root,
                    run_kind=args.run_kind,
                    seed=args.seed,
                    learning_rate=args.learning_rate,
                    dataset_file_sha256=args.dataset_sha256,
                    training_config_file_sha256=args.config_sha256,
                    experiment_manifest_sha256=args.manifest_sha256,
                    temporal_mix_sha256=args.temporal_mix_sha256,
                    runtime_sha256=args.runtime_sha256,
                )
            )
            return
    except Day20TrainingRuntimeV2Error as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
