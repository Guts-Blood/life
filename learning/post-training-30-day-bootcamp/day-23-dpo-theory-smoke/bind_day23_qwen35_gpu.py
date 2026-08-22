#!/usr/bin/env python3
"""Bind the frozen Day 23 CPU contract to remote GPU execution configs.

This builder deliberately does not load model weights or run an optimizer.  It
verifies the remote S1 payload, projects only the frozen ms-swift arguments,
parses every generated JSON through the pinned ``RLHFArguments`` CLI, and seals
the configs plus the runtime/selection/heldout evidence protocol in one
self-hashed binding manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
DAY21_DIR = BOOTCAMP_ROOT / "day-21-weekend-eval-reading"
for import_root in (DAY23_DIR, DAY21_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import day21_s1_handoff as day21  # noqa: E402
import day23_contract as cpu_contract  # noqa: E402


SCHEMA_NAME = "day23.qwen35_dpo_gpu_execution_binding"
SCHEMA_VERSION = 1
EXPECTED_CPU_CONTRACT_SHA256 = (
    "9d85ff6e149000ab3a4fea160eda5804c96a9c41655848dfb4613e0043c037cc"
)
EXPECTED_MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
EXPECTED_PRIOR_MECHANISM_FAILURE_SHA256 = (
    "600341cb77ee2bd97b2e4a00ae26a89720a9204b8266b86b6571c6c3181adc20"
)
EXPECTED_PRIOR_MECHANISM_BINDING_SHA256 = (
    "5e666283757a3c60c7d8062cb65b7f6e53579088b25f25fd947134970c237674"
)
EXPECTED_PRIOR_MECHANISM_FAILURE_PATH = Path(
    "/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T091033Z/"
    "evidence/mechanism-5step/failure-receipt.json"
)
EXPECTED_GPU_RUNTIME = {
    "ms-swift": "4.5.0.dev0",
    "peft": "0.19.1",
    "torch": "2.10.0+cu128",
    "transformers": "5.12.1",
}
GPU_WORLD_SIZE = 2
GPU_PER_DEVICE_TRAIN_BATCH = 4
GPU_MECHANISM_PER_DEVICE_TRAIN_BATCH = 1
GPU_PER_DEVICE_EVAL_BATCH = 4
GPU_GRADIENT_ACCUMULATION = 1
GPU_MECHANISM_GRADIENT_ACCUMULATION = 4
FORBIDDEN_EXECUTABLE_KEYS = {
    "adapters",
    "ref_adapters",
    "ref_model",
    "resume_from_checkpoint",
}
STAGE_METADATA_KEYS = {
    "checkpoint_candidates",
    "fresh_start_from_parent",
    "mechanism_pair_ids",
    "purpose",
    "remove_common_args",
}
CONFIG_FILENAMES = {
    "g2_one_step": "g2-one-step.json",
    "mechanism_5step": "mechanism-5step.json",
    "bounded_smoke_30step": "bounded-smoke-30step.json",
}
STAGE_SLUGS = {
    "g2_one_step": "g2-one-step",
    "mechanism_5step": "mechanism-5step",
    "bounded_smoke_30step": "bounded-smoke-30step",
}
STAGE_ORDER = (
    "g2_one_step",
    "mechanism_5step",
    "bounded_smoke_30step",
)
DEFAULT_RUN_CONTRACT = (
    BOOTCAMP_ROOT
    / "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json"
)
DEFAULT_PROMOTION = (
    BOOTCAMP_ROOT
    / "artifacts/checkpoints/day21-qwen35-s1-promotion-manifest.json"
)
DEFAULT_DOWNSTREAM_KEY = (
    BOOTCAMP_ROOT / "artifacts/checkpoints/day21-qwen35-s1-downstream-key.json"
)
DEFAULT_EXPORT_MANIFEST = (
    BOOTCAMP_ROOT
    / "artifacts/checkpoints/day21-qwen35-s1-merged-export-manifest.json"
)
DEFAULT_CPU_VALIDATOR = (
    BOOTCAMP_ROOT / "artifacts/scripts/validate_day23_qwen35_dpo.py"
)
DEFAULT_MS_SWIFT_ROOT = REPO_ROOT / "vendor/ms-swift"
DEFAULT_STAGE_RUNNER = DAY23_DIR / "run_day23_qwen35_gpu_stage.py"
DEFAULT_PREFERENCE_EVALUATOR = DAY23_DIR / "eval_day23_qwen35_preferences.py"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class Day23GPUBindingError(ValueError):
    """The remote payload or executable binding failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23GPUBindingError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any, *, self_field: str | None = None) -> str:
    if self_field is not None:
        value = dict(value)
        value.pop(self_field, None)
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    path = path.resolve()
    _require(path.is_file() and not path.is_symlink(), f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day23GPUBindingError(f"cannot load JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(
        isinstance(expected, str) and _HASH_RE.fullmatch(expected) is not None,
        f"{label} {field} is invalid",
    )
    actual = object_sha256(value, self_field=field)
    _require(expected == actual, f"{label} {field} drifted")
    return expected


def _validate_prior_mechanism_failure(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"prior mechanism failure receipt is missing: {resolved}",
    )
    _require(
        resolved == EXPECTED_PRIOR_MECHANISM_FAILURE_PATH,
        "prior mechanism failure receipt is not the canonical sealed attempt-1 path",
    )
    receipt = load_json(resolved)
    receipt_sha = verify_self_hash(
        receipt, "receipt_sha256", "prior mechanism failure receipt"
    )
    error = receipt.get("error")
    _require(
        receipt_sha == EXPECTED_PRIOR_MECHANISM_FAILURE_SHA256
        and receipt.get("status") == "fail"
        and receipt.get("stage") == "mechanism_5step"
        and receipt.get("binding_sha256")
        == EXPECTED_PRIOR_MECHANISM_BINDING_SHA256
        and isinstance(error, Mapping)
        and error.get("message") == "fewer than 3/4 mechanism pairs improved",
        "prior mechanism failure is not the exact sealed attempt-1 gate failure",
    )
    return {
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "receipt_sha256": receipt_sha,
        "binding_sha256": receipt["binding_sha256"],
        "status": "fail",
        "stage": "mechanism_5step",
        "failure_message": error["message"],
        "candidate_or_resume_source": False,
    }


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _file_identity(path: Path, *, content_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if content_sha256 is not None:
        result["content_sha256"] = content_sha256
    return result


def _resolve_frozen_path(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path is missing")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return resolved


def _canonical_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    _require(expanded.is_absolute(), f"{label} must be absolute")
    resolved = expanded.resolve()
    _require(resolved == expanded, f"{label} must already be canonical: {expanded}")
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return resolved


def _assert_no_symlinks(root: Path, label: str) -> None:
    _require(not root.is_symlink(), f"{label} cannot be a symlink: {root}")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"{label} contains a symlink: {path}")


def _file_manifest(
    directory: Path, *, exclude: Iterable[str] = ()
) -> dict[str, dict[str, Any]]:
    root = _canonical_directory(directory, "manifest directory")
    excluded = set(exclude)
    _assert_no_symlinks(root, "manifest directory")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return files


def _git_identity(checkout: Path) -> dict[str, Any]:
    root = _canonical_directory(checkout, "ms-swift checkout")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day23GPUBindingError("cannot inspect pinned ms-swift checkout") from error
    _require(commit == EXPECTED_MS_SWIFT_COMMIT, "ms-swift commit drifted")
    _require(not dirty, "ms-swift checkout must be clean")
    return {"path": str(root), "commit": commit, "clean": True}


def _run_cpu_validator(path: Path) -> dict[str, Any]:
    validator = path.resolve()
    identity = _file_identity(validator)
    try:
        completed = subprocess.run(
            [sys.executable, str(validator)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise Day23GPUBindingError("cannot execute Day 23 independent CPU validator") from error
    _require(
        completed.returncode == 0,
        "Day 23 independent CPU validator failed: " + completed.stderr.strip(),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    _require(lines, "Day 23 independent CPU validator emitted no result")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise Day23GPUBindingError("Day 23 CPU validator result is not JSON") from error
    _require(isinstance(result, dict), "Day 23 CPU validator result is not an object")
    _require(
        result.get("status") == "valid_cpu_ready_gpu_pending"
        and result.get("cpu_ready") is True
        and result.get("gpu_optimizer_ready") is False
        and result.get("run_contract_sha256") == EXPECTED_CPU_CONTRACT_SHA256,
        "Day 23 independent CPU validator boundary drifted",
    )
    return {
        **identity,
        "command": [str(Path(sys.executable).resolve()), str(validator)],
        "status": result["status"],
        "result": result,
        "result_sha256": object_sha256(result),
    }


def _validate_cpu_bundle(
    run_contract_path: Path, validator_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = run_contract_path.resolve()
    contract = load_json(path)
    contract_sha = verify_self_hash(contract, "contract_sha256", "CPU run contract")
    _require(
        contract_sha == EXPECTED_CPU_CONTRACT_SHA256,
        "refusing a CPU contract other than frozen 9d85ff...037cc",
    )
    _require(
        contract.get("schema_name") == "day23.qwen35_coding_dpo_cpu_run_contract"
        and contract.get("schema_version") == 1
        and contract.get("status") == "cpu_ready_gpu_pending",
        "CPU run contract schema/status drifted",
    )
    rule = contract.get("binding_rule")
    _require(isinstance(rule, dict), "CPU binding_rule is missing")
    _require(
        set(rule.get("stage_metadata_keys_to_remove") or []) == STAGE_METADATA_KEYS
        and set(rule.get("executable_config_must_omit_cli_default_keys") or [])
        == FORBIDDEN_EXECUTABLE_KEYS,
        "CPU binding projection rules drifted",
    )
    data_path = _resolve_frozen_path(
        BOOTCAMP_ROOT, contract["inputs"]["data_manifest"]["path"], "data manifest"
    )
    processor_path = _resolve_frozen_path(
        BOOTCAMP_ROOT,
        contract["inputs"]["processor_audit"]["path"],
        "processor audit",
    )
    argument_path = (
        BOOTCAMP_ROOT
        / "artifacts/eval/day23-qwen35-coding-dpo-argument-audit.json"
    ).resolve()
    data = load_json(data_path)
    processor = load_json(processor_path)
    argument = load_json(argument_path)
    data_sha = verify_self_hash(data, "manifest_sha256", "data manifest")
    processor_sha = verify_self_hash(processor, "summary_sha256", "processor audit")
    argument_sha = verify_self_hash(argument, "audit_sha256", "argument audit")
    _require(
        data_sha == contract["inputs"]["data_manifest"]["content_sha256"]
        and file_sha256(data_path)
        == contract["inputs"]["data_manifest"]["file_sha256"],
        "CPU data manifest cross-binding drifted",
    )
    _require(
        processor_sha == contract["inputs"]["processor_audit"]["content_sha256"]
        and file_sha256(processor_path)
        == contract["inputs"]["processor_audit"]["file_sha256"],
        "CPU processor audit cross-binding drifted",
    )
    _require(
        argument.get("run_contract", {}).get("contract_sha256") == contract_sha,
        "argument audit no longer binds the CPU contract",
    )
    independent = _run_cpu_validator(validator_path)
    _require(
        independent["result"]["data_manifest_sha256"] == data_sha
        and independent["result"]["processor_summary_sha256"] == processor_sha
        and independent["result"]["argument_audit_sha256"] == argument_sha,
        "independent CPU validator results do not match frozen inputs",
    )
    identity = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "contract_sha256": contract_sha,
        "status": contract["status"],
        "frozen_inputs": {
            "data_manifest": _file_identity(data_path, content_sha256=data_sha),
            "processor_audit": _file_identity(
                processor_path, content_sha256=processor_sha
            ),
            "argument_audit": _file_identity(
                argument_path, content_sha256=argument_sha
            ),
        },
        "independent_validator": independent,
    }
    return contract, data, processor, identity


def _validate_remote_parent(
    contract: Mapping[str, Any],
    *,
    promotion_path: Path,
    downstream_key_path: Path,
    export_manifest_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    parent = contract.get("parent")
    _require(isinstance(parent, Mapping), "CPU parent identity is missing")
    promotion_path = promotion_path.resolve()
    downstream_key_path = downstream_key_path.resolve()
    export_manifest_path = export_manifest_path.resolve()
    model = _canonical_directory(model_path, "remote merged S1 export")
    _require(
        str(model) == parent.get("remote_merged_export_path"),
        "remote model path differs from the frozen merged S1 export",
    )
    _require(
        file_sha256(promotion_path) == parent.get("promotion_file_sha256")
        and file_sha256(downstream_key_path)
        == parent.get("downstream_key_file_sha256")
        and file_sha256(export_manifest_path)
        == parent.get("merged_export_file_sha256"),
        "compact Day 21 handoff file bytes drifted",
    )

    try:
        promotion = day21.verify_promotion(promotion_path)
        downstream_key = day21.verify_downstream_key(downstream_key_path)
        export_manifest = day21.verify_export_manifest(export_manifest_path)
    except (OSError, ValueError, day21.HandoffError) as error:
        raise Day23GPUBindingError(f"Day 21 remote handoff verification failed: {error}") from error
    promotion_content = promotion.pop("_verified_promotion_manifest_sha256")
    _require(
        promotion_content == parent.get("promotion_manifest_sha256")
        and downstream_key.get("key_sha256") == parent.get("downstream_key_sha256")
        and export_manifest.get("manifest_sha256")
        == parent.get("merged_export_manifest_sha256")
        and export_manifest.get("files_sha256")
        == parent.get("merged_export_artifact_sha256")
        and downstream_key.get("downstream_key") == parent.get("downstream_key")
        and promotion.get("checkpoint_id") == parent.get("checkpoint_id"),
        "Day 21 compact handoff content drifted",
    )

    canonical_promotion_path = Path(
        str(downstream_key["promotion_manifest"]["path"])
    ).resolve()
    canonical_export_manifest_path = Path(
        str(promotion["inference_export"]["manifest"]["path"])
    ).resolve()
    recovery_path = Path(str(promotion["state"]["recovery_evidence"]["path"])).resolve()
    parity_path = Path(str(promotion["inference_export"]["parity_report"]["path"])).resolve()
    try:
        canonical_promotion = day21.verify_promotion(canonical_promotion_path)
        canonical_export = day21.verify_export_manifest(canonical_export_manifest_path)
        recovery = day21.verify_recovery(recovery_path)
        parity = day21.verify_parity(parity_path)
    except (OSError, ValueError, day21.HandoffError) as error:
        raise Day23GPUBindingError(
            f"canonical remote Day 21 payload verification failed: {error}"
        ) from error
    canonical_promotion.pop("_verified_promotion_manifest_sha256", None)
    _require(
        file_sha256(canonical_promotion_path) == file_sha256(promotion_path)
        and canonical_promotion == promotion,
        "canonical remote promotion differs from the compact trust root",
    )
    _require(
        file_sha256(canonical_export_manifest_path) == file_sha256(export_manifest_path)
        and canonical_export == export_manifest,
        "canonical remote export manifest differs from the compact trust root",
    )
    recovery_identity = promotion["state"]["recovery_evidence"]
    parity_identity = promotion["inference_export"]["parity_report"]
    _require(
        file_sha256(recovery_path) == recovery_identity["file_sha256"]
        and recovery["evidence_sha256"] == recovery_identity["content_sha256"],
        "remote checkpoint recovery evidence drifted",
    )
    _require(
        file_sha256(parity_path) == parity_identity["file_sha256"]
        and parity["parity_sha256"] == parity_identity["content_sha256"],
        "remote S1 adapter/merged parity evidence drifted",
    )
    checkpoint = _canonical_directory(
        Path(str(parent["remote_resumable_checkpoint_path"])),
        "remote resumable S1 checkpoint",
    )
    _require(
        Path(str(recovery.get("archive_checkpoint", ""))).resolve() == checkpoint
        and recovery.get("checkpoint", {}).get("integrity_sha256")
        == parent.get("checkpoint_integrity_sha256")
        and recovery.get("checkpoint", {}).get("snapshot_sha256")
        == parent.get("checkpoint_snapshot_sha256")
        and recovery.get("checkpoint", {}).get("adapter_sha256")
        == parent.get("adapter_sha256"),
        "remote resumable checkpoint identity drifted",
    )
    model_files = _file_manifest(model, exclude=("S1-EXPORT-MANIFEST.json",))
    _require(
        model_files == export_manifest.get("files")
        and object_sha256(model_files) == parent.get("merged_export_artifact_sha256"),
        "remote merged S1 file manifest drifted",
    )
    return {
        "status": "verified_remote_s1_payload",
        "role": "S1",
        "checkpoint_id": parent["checkpoint_id"],
        "downstream_key": parent["downstream_key"],
        "promotion_manifest": _file_identity(
            canonical_promotion_path, content_sha256=promotion_content
        ),
        "downstream_key_manifest": _file_identity(
            downstream_key_path, content_sha256=downstream_key["key_sha256"]
        ),
        "merged_export_manifest": _file_identity(
            canonical_export_manifest_path,
            content_sha256=export_manifest["manifest_sha256"],
        ),
        "merged_export": {
            "path": str(model),
            "files": model_files,
            "files_sha256": object_sha256(model_files),
            "file_count": len(model_files),
        },
        "resumable_checkpoint": {
            "path": str(checkpoint),
            "integrity_sha256": parent["checkpoint_integrity_sha256"],
            "snapshot_sha256": parent["checkpoint_snapshot_sha256"],
            "adapter_sha256": parent["adapter_sha256"],
            "recovery_evidence": _file_identity(
                recovery_path, content_sha256=recovery["evidence_sha256"]
            ),
            "not_a_day23_model_input": True,
        },
        "adapter_merged_parity": _file_identity(
            parity_path, content_sha256=parity["parity_sha256"]
        ),
        "policy_initial_state": "merged_export_plus_fresh_lora",
        "reference_initial_state": "same_merged_export_with_fresh_lora_disabled",
        "model_argument": str(model),
        "forbidden_as_model_argument": str(checkpoint),
        "ms_swift_commit": EXPECTED_MS_SWIFT_COMMIT,
    }


def _dataset_value(value: str, root: Path, label: str) -> str:
    path_value, marker, sample = value.rpartition("#")
    if marker and sample.isdigit():
        suffix = f"#{sample}"
    else:
        path_value, suffix = value, ""
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return str(resolved) + suffix


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(key) or _contains_placeholder(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return "__" in value if isinstance(value, str) else False


def _project_stage(
    contract: Mapping[str, Any],
    *,
    source_stage: str,
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    common = dict(contract["intended_ms_swift_args"])
    stage = dict(contract["stages"][source_stage])
    for key in stage.get("remove_common_args", []):
        _require(key in common, f"{source_stage} removes unknown common key: {key}")
        common.pop(str(key))
    runtime_overrides = {
        key: value for key, value in stage.items() if key not in STAGE_METADATA_KEYS
    }
    result = common | runtime_overrides
    result["model"] = str(model_path.resolve())
    result["output_dir"] = str(output_dir)
    result["dataset"] = [
        _dataset_value(str(value), BOOTCAMP_ROOT, f"{source_stage} dataset")
        for value in result["dataset"]
    ]
    if "val_dataset" in result:
        result["val_dataset"] = [
            _dataset_value(str(value), BOOTCAMP_ROOT, f"{source_stage} val_dataset")
            for value in result["val_dataset"]
        ]
    plugins = []
    for value in result["external_plugins"]:
        plugin = Path(str(value))
        resolved = plugin.resolve() if plugin.is_absolute() else (BOOTCAMP_ROOT / plugin).resolve()
        _require(resolved.is_file() and not resolved.is_symlink(), f"external plugin is missing: {resolved}")
        plugins.append(str(resolved))
    result["external_plugins"] = plugins
    _require(not (set(result) & FORBIDDEN_EXECUTABLE_KEYS), f"{source_stage} contains forbidden CLI keys")
    _require(
        not any(value is None or value == [] for value in result.values()),
        f"{source_stage} contains null or empty-list values",
    )
    _require(not _contains_placeholder(result), f"{source_stage} contains an unresolved placeholder")
    return result


def _build_configs(
    contract: Mapping[str, Any], *, model_path: Path, run_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    outputs = {
        stage: (run_root / "outputs" / STAGE_SLUGS[stage]).resolve()
        for stage in STAGE_ORDER
    }
    _require(len(set(outputs.values())) == len(outputs), "stage output directories overlap")
    mechanism = _project_stage(
        contract,
        source_stage="mechanism_5step",
        model_path=model_path,
        output_dir=outputs["mechanism_5step"],
    )
    smoke = _project_stage(
        contract,
        source_stage="bounded_smoke_30step",
        model_path=model_path,
        output_dir=outputs["bounded_smoke_30step"],
    )
    g2 = dict(mechanism)
    g2["max_steps"] = 1
    g2["save_steps"] = 1
    # Any positive frozen warmup ratio rounds up to one warmup step when
    # max_steps=1, making the sole optimizer step use LR=0.  G2 exists to
    # prove a real update, so only this quarantined derivative disables
    # warmup; the 5/30 step scientific stages retain the frozen schedule.
    g2["warmup_ratio"] = 0.0
    g2["output_dir"] = str(outputs["g2_one_step"])
    _require(
        isinstance(g2.get("dataset"), list)
        and len(g2["dataset"]) == 1
        and str(g2["dataset"][0]).endswith("#4"),
        "G2 source dataset is not the frozen mechanism projection",
    )
    g2["dataset"] = [str(g2["dataset"][0])[:-2] + "#8"]
    configs = {
        "g2_one_step": g2,
        "mechanism_5step": mechanism,
        "bounded_smoke_30step": smoke,
    }
    for stage, config in configs.items():
        config["per_device_train_batch_size"] = (
            GPU_MECHANISM_PER_DEVICE_TRAIN_BATCH
            if stage == "mechanism_5step"
            else GPU_PER_DEVICE_TRAIN_BATCH
        )
        config["per_device_eval_batch_size"] = GPU_PER_DEVICE_EVAL_BATCH
        config["gradient_accumulation_steps"] = (
            GPU_MECHANISM_GRADIENT_ACCUMULATION
            if stage == "mechanism_5step"
            else GPU_GRADIENT_ACCUMULATION
        )
    _require(
        GPU_WORLD_SIZE * GPU_PER_DEVICE_TRAIN_BATCH * GPU_GRADIENT_ACCUMULATION
        == 8
        and GPU_WORLD_SIZE
        * GPU_MECHANISM_PER_DEVICE_TRAIN_BATCH
        * GPU_MECHANISM_GRADIENT_ACCUMULATION
        == 8,
        "GPU batch amendment has an invalid nominal global batch",
    )
    mechanism_diff = {
        key: {"g2_one_step": g2.get(key), "mechanism_5step": mechanism.get(key)}
        for key in sorted(set(g2) | set(mechanism))
        if g2.get(key) != mechanism.get(key)
    }
    _require(
        set(mechanism_diff)
        == {
            "dataset",
            "gradient_accumulation_steps",
            "max_steps",
            "output_dir",
            "per_device_train_batch_size",
            "save_steps",
            "warmup_ratio",
        }
        and mechanism_diff["max_steps"] == {"g2_one_step": 1, "mechanism_5step": 5}
        and mechanism_diff["save_steps"] == {"g2_one_step": 1, "mechanism_5step": 5},
        "G2 one-step/high-batch derivation exceeds its authorized seven-field diff",
    )
    heldout = str(
        (BOOTCAMP_ROOT / contract["dataset_usage"]["heldout"]["path"]).resolve()
    )
    for stage, config in configs.items():
        _require(not (set(config) & FORBIDDEN_EXECUTABLE_KEYS), f"{stage} contains a forbidden key")
        encoded = canonical_json(config)
        _require(heldout.encode("utf-8") not in encoded, f"{stage} leaks heldout path")
        for key in ("model", "output_dir"):
            _require(Path(str(config[key])).is_absolute(), f"{stage} {key} is not absolute")
        for key in ("dataset", "val_dataset", "external_plugins"):
            for value in config.get(key, []):
                path_value = str(value).rpartition("#")[0] if key != "external_plugins" and str(value).rpartition("#")[2].isdigit() else str(value)
                _require(Path(path_value).is_absolute(), f"{stage} {key} path is not absolute")
    return configs, outputs


def _parse_configs(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    ms_swift_root: Path,
    mechanism_pair_ids: Sequence[str],
) -> dict[str, Any]:
    checkout = _git_identity(ms_swift_root)
    try:
        import swift
        from swift.arguments import RLHFArguments
        from swift.cli.main import parse_yaml_args
        from swift.template import TEMPLATE_MAPPING
        from swift.utils import parse_args
    except ImportError as error:
        raise Day23GPUBindingError("pinned ms-swift RLHF JSON CLI is unavailable") from error
    imported_root = Path(swift.__file__).resolve().parent
    _require(
        imported_root == (ms_swift_root.resolve() / "swift"),
        "RLHF parse imported swift outside the pinned checkout",
    )
    versions: dict[str, str] = {}
    for package in (*EXPECTED_GPU_RUNTIME, "accelerate", "datasets", "trl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise Day23GPUBindingError(f"required GPU package is missing: {package}") from error
    _require(
        all(versions[name] == expected for name, expected in EXPECTED_GPU_RUNTIME.items()),
        f"remote GPU package versions drifted: {versions}",
    )
    argument_fields = {field.name for field in dataclasses.fields(RLHFArguments)}
    parsed_stages: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="day23-gpu-binding-parse-") as temporary:
        root = Path(temporary)
        for stage in STAGE_ORDER:
            config = dict(configs[stage])
            unknown = sorted(set(config) - argument_fields)
            _require(not unknown, f"{stage} contains unknown RLHFArguments keys: {unknown}")
            path = root / CONFIG_FILENAMES[stage]
            # RLHFArguments creates output_dir during __post_init__ even with
            # add_version=False.  Parse an otherwise byte-equivalent projection
            # into the temporary tree so the binder cannot consume the fresh
            # stage output path before the runner's no-overwrite gate.
            parse_config = dict(config)
            parse_config["output_dir"] = str(root / f"parsed-output-{stage}")
            path.write_bytes(json_bytes(parse_config))
            argv = [str(path)]
            previous_config = os.environ.get("SWIFT_CONFIG_FILE")
            try:
                parse_yaml_args(argv)
                parsed, remaining = parse_args(RLHFArguments, argv)
            except (OSError, SystemExit, ValueError) as error:
                raise Day23GPUBindingError(
                    f"{stage} real remote RLHFArguments JSON parse failed: {error}"
                ) from error
            finally:
                if previous_config is None:
                    os.environ.pop("SWIFT_CONFIG_FILE", None)
                else:
                    os.environ["SWIFT_CONFIG_FILE"] = previous_config
            _require(not remaining, f"{stage} left unparsed CLI arguments: {remaining}")
            _require(
                Path(str(parsed.output_dir)).resolve()
                == Path(parse_config["output_dir"]).resolve(),
                f"{stage} parsed output_dir drifted",
            )
            training_args = parsed.training_args
            _require(
                parsed.rlhf_type == "dpo"
                and parsed.tuner_type == "lora"
                and parsed.template == "day23_qwen3_5_dpo_target_v1"
                and parsed.beta == 0.1
                and parsed.loss_type == "sigmoid"
                and parsed.max_length == 512
                and parsed.freeze_vit is True
                and parsed.freeze_aligner is True
                and parsed.freeze_llm is False
                and parsed.adapters == []
                and parsed.ref_adapters == []
                and parsed.ref_model is None
                and parsed.resume_from_checkpoint is None
                and parsed.sync_ref_model is False
                and getattr(training_args, "reference_free", None) in (None, False)
                and getattr(training_args, "precompute_ref_log_probs", False) is False
                and getattr(training_args, "disable_dropout", False) is True,
                f"{stage} parsed DPO/reference/freeze semantics drifted",
            )
            expected_train_batch = (
                GPU_MECHANISM_PER_DEVICE_TRAIN_BATCH
                if stage == "mechanism_5step"
                else GPU_PER_DEVICE_TRAIN_BATCH
            )
            expected_gradient_accumulation = (
                GPU_MECHANISM_GRADIENT_ACCUMULATION
                if stage == "mechanism_5step"
                else GPU_GRADIENT_ACCUMULATION
            )
            _require(
                training_args.per_device_train_batch_size == expected_train_batch
                and training_args.per_device_eval_batch_size
                == GPU_PER_DEVICE_EVAL_BATCH
                and training_args.gradient_accumulation_steps
                == expected_gradient_accumulation,
                f"{stage} parsed dual-GPU batch amendment drifted",
            )
            train_dataset, val_dataset = parsed.load_dataset()
            expected_train = (
                154
                if stage == "bounded_smoke_30step"
                else (8 if stage == "g2_one_step" else 4)
            )
            expected_dev = 17 if stage == "bounded_smoke_30step" else 0
            _require(len(train_dataset) == expected_train, f"{stage} loaded wrong train count")
            _require(
                (0 if val_dataset is None else len(val_dataset)) == expected_dev,
                f"{stage} loaded wrong dev count",
            )
            if stage != "bounded_smoke_30step":
                frozen_rows = cpu_contract.load_jsonl(
                    Path(str(config["dataset"][0]).rsplit("#", 1)[0])
                )[:expected_train]
                actual_text = [
                    {
                        "messages": row["messages"],
                        "rejected_response": row["rejected_response"],
                    }
                    for row in train_dataset
                ]
                expected_text = [
                    {
                        "messages": row["messages"],
                        "rejected_response": row["rejected_response"],
                    }
                    for row in frozen_rows
                ]
                _require(actual_text == expected_text, f"{stage} mechanism row order/text drifted")
                if stage == "mechanism_5step":
                    _require(
                        list(mechanism_pair_ids)
                        == [str(row["pair_id"]) for row in frozen_rows],
                        "frozen mechanism pair IDs drifted",
                    )
            _require(
                "day23_qwen3_5_dpo_target_v1" in TEMPLATE_MAPPING,
                f"{stage} external template plugin did not register",
            )
            parsed_stages[stage] = {
                "status": "pass",
                "entrypoint": "swift.cli.main.parse_yaml_args_then_swift.utils.parse_args(RLHFArguments)",
                "real_json_cli": True,
                "weights_loaded": False,
                "executable_config_keys": sorted(config),
                "omitted_cli_default_keys": sorted(FORBIDDEN_EXECUTABLE_KEYS),
                "max_steps": parsed.max_steps,
                "save_steps": parsed.save_steps,
                "train_records": expected_train,
                "dev_records": expected_dev,
                "dataset_shuffle": parsed.dataset_shuffle,
                "train_dataloader_shuffle": parsed.train_dataloader_shuffle,
                "reference_free": getattr(training_args, "reference_free", None),
                "precompute_ref_log_probs": getattr(
                    training_args, "precompute_ref_log_probs", False
                ),
                "disable_dropout": getattr(training_args, "disable_dropout", None),
                "world_size_at_execution": GPU_WORLD_SIZE,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "configured_nominal_global_train_batch_size": (
                    GPU_WORLD_SIZE
                    * training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                ),
                "expected_realized_global_train_batch_size_range": (
                    [4, 4]
                    if stage == "mechanism_5step"
                    else ([2, 8] if stage == "bounded_smoke_30step" else [8, 8])
                ),
            }
    python = Path(sys.executable).resolve()
    return {
        "status": "pass",
        "scope": "real_remote_RLHFArguments_JSON_parse_no_model_weights",
        "python_executable": {
            "path": str(python),
            "file_sha256": file_sha256(python),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "ms_swift_checkout": checkout["path"],
        "ms_swift_commit": checkout["commit"],
        "ms_swift_clean": checkout["clean"],
        "ms_swift_import_root": str(imported_root),
        "ms_swift_package_version": versions["ms-swift"],
        "package_versions": dict(sorted(versions.items())),
        "rlhf_argument_field_count": len(argument_fields),
        "stages": parsed_stages,
    }


def _absolute_data_identity(
    data_manifest: Mapping[str, Any], split: str
) -> dict[str, Any]:
    item = data_manifest["outputs"][split]
    path = _resolve_frozen_path(BOOTCAMP_ROOT, item["path"], f"{split} dataset")
    _require(file_sha256(path) == item["file_sha256"], f"{split} dataset bytes drifted")
    return {
        "path": str(path),
        "file_sha256": item["file_sha256"],
        "records": item["records"],
        "ordered_pair_ids_sha256": item["ordered_pair_ids_sha256"],
        "ordered_row_hashes_sha256": item["ordered_row_hashes_sha256"],
    }


def _producer_identity(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return _file_identity(resolved)


def _receipt_paths(run_root: Path, stage: str) -> dict[str, Any]:
    evidence = (run_root / "evidence" / STAGE_SLUGS[stage]).resolve()
    return {
        "evidence_dir": str(evidence),
        "success": str(evidence / "success-receipt.json"),
        "failure": str(evidence / "failure-receipt.json"),
        "self_hash_field": "receipt_sha256",
        "exclusive_create_no_overwrite": True,
    }


def _build_selector_contract(
    *,
    run_root: Path,
    data_manifest: Mapping[str, Any],
    processor_summary: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
    smoke_config_identity: Mapping[str, Any],
    smoke_receipts: Mapping[str, Any],
) -> dict[str, Any]:
    source_pairs = _resolve_frozen_path(
        BOOTCAMP_ROOT,
        data_manifest["inputs"]["day22_pairs"]["path"],
        "Day 22 source pairs",
    )
    audit_rows = _resolve_frozen_path(
        BOOTCAMP_ROOT,
        processor_summary["audit_rows"]["path"],
        "Day 23 processor audit rows",
    )
    _require(
        file_sha256(source_pairs)
        == data_manifest["inputs"]["day22_pairs"]["file_sha256"],
        "Day 22 source pair bytes drifted",
    )
    _require(
        file_sha256(audit_rows) == processor_summary["audit_rows"]["file_sha256"],
        "Day 23 processor audit row bytes drifted",
    )
    selection_root = (run_root / "evidence" / "selection").resolve()
    candidates = []
    for step in (15, 30):
        checkpoint = (
            run_root
            / "outputs"
            / STAGE_SLUGS["bounded_smoke_30step"]
            / f"checkpoint-{step}"
        ).resolve()
        dev_evaluation = (selection_root / f"dev-checkpoint-{step}.json").resolve()
        candidates.append(
            {
                "step": step,
                "checkpoint_path": str(checkpoint),
                "adapter_path": str(checkpoint / "adapter_model.safetensors"),
                "checkpoint_receipt_path": str(dev_evaluation),
                "training_stage_receipt_path": smoke_receipts["success"],
                "training_config_file_sha256": smoke_config_identity["file_sha256"],
                "dev_evaluation_path": str(dev_evaluation),
            }
        )
    metric_rule = {
        "response_logprob_reduction": "sum_over_response_only_tokens",
        "policy_margin": "policy_chosen_logp_sum-policy_rejected_logp_sum",
        "reference_margin": "reference_chosen_logp_sum-reference_rejected_logp_sum",
        "reward_margin": "beta*(policy_margin-reference_margin)",
        "beta": 0.1,
        "pair_correct": "reward_margin>0",
        "dev_pair_accuracy": "count(pair_correct)/17",
        "dev_mean_reward_margin": "arithmetic_mean(reward_margin_over_all_17)",
        "length_matched_eligible": "abs(chosen_response_tokens-rejected_response_tokens)<=8",
        "length_matched_public_length": "min(chosen_response_tokens,rejected_response_tokens)",
        "length_matched_pair_margin": (
            "beta*((sum_first_public_policy_chosen-sum_first_public_policy_rejected)-"
            "(sum_first_public_reference_chosen-sum_first_public_reference_rejected))"
        ),
        "dev_length_matched_margin": (
            "arithmetic_mean(length_matched_pair_margin_over_eligible_pairs)"
        ),
        "zero_length_matched_eligible": "fail_closed_no_selection",
        "nonfinite_value": "fail_closed_no_selection",
    }
    slices = {
        "response_length": {
            "source": "processor_audit_rows.chosen/rejected.response_token_count",
            "required_row_fields": [
                "chosen_response_tokens",
                "rejected_response_tokens",
                "absolute_length_delta_tokens",
                "public_length_tokens",
            ],
            "max_branch_length_buckets": [
                {"name": "<=64", "minimum": 0, "maximum_inclusive": 64},
                {"name": "65-128", "minimum": 65, "maximum_inclusive": 128},
                {"name": ">128", "minimum": 129, "maximum_inclusive": None},
            ],
        },
        "source_family": "compiled_row.family_keys.source",
        "test_family": "compiled_row.family_keys.test",
        "quality_status": (
            "source_pair.pair_status+'|'+','.join(sorted(source_pair.quality_flags))"
        ),
    }
    selection_rule = {
        "eligible_steps": [15, 30],
        "primary": {"metric": "dev_pair_accuracy", "direction": "max"},
        "tie_breakers": [
            {"metric": "dev_mean_reward_margin", "direction": "max"},
            {"metric": "dev_length_matched_margin", "direction": "max"},
            {"metric": "checkpoint_step", "direction": "min"},
        ],
        "trainer_native_eval_is_not_selector": True,
        "selection_write_count": 1,
    }
    return {
        "status": "frozen_pending_dev_evaluation",
        "evaluator": dict(evaluator_identity),
        "datasets": {
            "dev": _absolute_data_identity(data_manifest, "dev"),
            "source_pairs": {
                "path": str(source_pairs),
                "file_sha256": file_sha256(source_pairs),
                "records": 200,
            },
            "processor_audit_rows": {
                "path": str(audit_rows),
                "file_sha256": file_sha256(audit_rows),
                "records": processor_summary["audit_rows"]["records"],
                "ordered_audit_sha256": processor_summary["audit_rows"][
                    "ordered_audit_sha256"
                ],
            },
        },
        "eligible_checkpoints": candidates,
        "forbidden_candidates": [
            "g2_one_step",
            "mechanism_5step",
            "merged_s1_parent",
        ],
        "metric_rule": metric_rule,
        "metric_rule_sha256": object_sha256(metric_rule),
        "preference_slices": slices,
        "selection_rule": selection_rule,
        "selection_rule_sha256": object_sha256(selection_rule),
        "selection_receipt": {
            "path": str(selection_root / "selection-decision.json"),
            "schema_name": "day23.qwen35_dpo_checkpoint_selection",
            "status": "selected",
            "self_hash_field": "selection_sha256",
            "exclusive_create_no_overwrite": True,
            "required_fields": [
                "binding_sha256",
                "cpu_contract_sha256",
                "eligible",
                "selected",
                "selection_rule",
                "selection_sha256",
            ],
        },
    }


def _build_heldout_contract(
    *,
    run_root: Path,
    data_manifest: Mapping[str, Any],
    selector_contract: Mapping[str, Any],
) -> dict[str, Any]:
    root = (run_root / "evidence" / "heldout").resolve()
    heldout = _absolute_data_identity(data_manifest, "heldout")
    return {
        "status": "sealed_until_selection",
        "one_shot": True,
        "dataset": heldout,
        "source_pairs": selector_contract["datasets"]["source_pairs"],
        "processor_audit_rows": selector_contract["datasets"][
            "processor_audit_rows"
        ],
        "selection_receipt_path": selector_contract["selection_receipt"]["path"],
        "winner_only": True,
        "preconditions": [
            "selection receipt exists and independently verifies",
            "selected step is exactly one of 15 or 30",
            "selected checkpoint/reload receipt and adapter hashes verify",
            "heldout path/hash is absent from every executable training config",
            "claim path does not already exist",
        ],
        "claim": {
            "path": str(root / "heldout-claim.json"),
            "schema_name": "day23.qwen35_dpo_heldout_claim",
            "status": "claimed",
            "self_hash_field": "claim_sha256",
            "exclusive_create_no_overwrite": True,
            "required_bindings": [
                "binding_sha256",
                "selection_receipt_file_sha256",
                "selection_sha256",
                "selected_step",
                "selected_checkpoint_path",
                "selected_checkpoint_receipt_sha256",
                "selected_adapter_file_sha256",
                "heldout_file_sha256",
                "heldout_ordered_pair_ids_sha256",
            ],
        },
        "evaluation": {
            "path": str(root / "heldout-winner-evaluation.json"),
            "schema_name": "day23.qwen35_dpo_preference_evaluation",
            "self_hash_field": "evaluation_sha256",
            "exclusive_create_no_overwrite": True,
            "expected_records": 29,
            "expected_unique_pair_ids": 29,
            "exact_order_required": True,
        },
        "final_receipt": {
            "path": str(root / "heldout-consumption.json"),
            "schema_name": "day23.qwen35_dpo_heldout_consumption",
            "status": "consumed",
            "self_hash_field": "consumption_sha256",
            "exclusive_create_no_overwrite": True,
            "required_bindings": [
                "binding_sha256",
                "claim_file_sha256",
                "claim_sha256",
                "selection_sha256",
                "selected_step",
                "evaluation_file_sha256",
                "evaluation_sha256",
                "records",
                "ordered_pair_ids_sha256",
                "attempts",
            ],
            "required_values": {"records": 29, "attempts": 1},
        },
        "failure_after_claim": {
            "path": str(root / "heldout-failure.json"),
            "schema_name": "day23.qwen35_dpo_heldout_failure",
            "status": "failed_closed_no_retry",
            "self_hash_field": "failure_sha256",
            "exclusive_create_no_overwrite": True,
            "retry_allowed": False,
            "manual_adjudication_required": True,
        },
        "state_machine": [
            "sealed_until_selection",
            "claimed_once",
            "consumed_once_or_failed_closed_without_retry",
        ],
        "forbidden": [
            "heldout access before a valid selection receipt",
            "evaluation of both checkpoint 15 and checkpoint 30",
            "second claim after any claim exists",
            "selection changes after heldout claim",
            "using heldout metrics to reselect a checkpoint",
        ],
    }


def _build_stage_contracts(
    *,
    run_root: Path,
    binding_dir: Path,
    configs: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_identities: dict[str, Any] = {}
    receipts = {stage: _receipt_paths(run_root, stage) for stage in STAGE_ORDER}
    roles = {
        "g2_one_step": "memory_save_reload_preflight_quarantined",
        "mechanism_5step": "mechanism_only_never_candidate_never_resume_source",
        "bounded_smoke_30step": "fresh_bounded_smoke_checkpoint_candidates_15_30",
    }
    source_stages = {
        "g2_one_step": "mechanism_5step_authorized_one_step_derivation",
        "mechanism_5step": "cpu_contract.stages.mechanism_5step",
        "bounded_smoke_30step": "cpu_contract.stages.bounded_smoke_30step",
    }
    checkpoint_steps = {
        "g2_one_step": [1],
        "mechanism_5step": [5],
        "bounded_smoke_30step": [15, 30],
    }
    stages: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        path = (binding_dir / CONFIG_FILENAMES[stage]).resolve()
        payload = json_bytes(configs[stage])
        identity = {
            "path": str(path),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "keys": sorted(configs[stage]),
        }
        config_identities[stage] = identity
        stages[stage] = {
            "role": roles[stage],
            "source_projection": source_stages[stage],
            "executable_config": identity,
            "output_dir": str(outputs[stage]),
            "evidence_dir": receipts[stage]["evidence_dir"],
            "fresh_process_required": True,
            "fresh_start_from_remote_parent": True,
            "parent_model_path": configs[stage]["model"],
            "forbidden_config_keys": sorted(FORBIDDEN_EXECUTABLE_KEYS),
            "checkpoint_steps": checkpoint_steps[stage],
            "success_receipt_path": receipts[stage]["success"],
            "failure_receipt_path": receipts[stage]["failure"],
        }
    stages["g2_one_step"]["authorized_derivation"] = {
        "base_stage": "mechanism_5step",
        "authorization": "user-requested dual-GPU high-batch memory preflight",
        "only_changed_config_keys": [
            "dataset",
            "gradient_accumulation_steps",
            "max_steps",
            "output_dir",
            "per_device_train_batch_size",
            "save_steps",
            "warmup_ratio",
        ],
        "dataset_sample_count": {"from": 4, "to": 8},
        "per_device_train_batch_size": {"from": 1, "to": 4},
        "gradient_accumulation_steps": {"from": 4, "to": 1},
        "max_steps": {"from": 5, "to": 1},
        "save_steps": {"from": 5, "to": 1},
        "warmup_ratio": {
            "from": configs["mechanism_5step"]["warmup_ratio"],
            "to": 0.0,
            "reason": "a one-step run otherwise rounds warmup to one zero-LR update",
        },
        "all_other_keys_and_values_byte_equivalent_after_output_normalization": True,
    }
    stages["g2_one_step"]["runtime_gate"] = {
        "g1_required_before_first_forward": [
            "policy is merged S1 plus exactly one fresh LoRA adapter",
            "trainer.ref_model is None and no reference adapter is configured",
            "live reference forward enters null_ref_context and disables the fresh adapter",
            "trainable names match the frozen target_regex and exact Qwen3.5 module distribution",
            "optimizer parameter IDs exactly equal fresh LoRA trainable parameter IDs",
            "vision tower, aligner, embeddings, lm_head, and every non-LoRA parameter are frozen",
            "step-zero policy/reference response logprobs match within the frozen tolerance",
        ],
        "g2_required_for_success": [
            "one real optimizer update completes with finite loss/rewards/logps/logits/grad_norm",
            "policy/reference/optimizer/activation/logit CUDA phases are measured",
            "minimum observed free-memory fraction is at least 0.15",
            "fresh LoRA tensor digest changes and frozen parent tensor digest does not",
            "checkpoint-1 contains adapter/trainer/optimizer/scheduler/RNG files",
            "a different process reloads merged S1 plus checkpoint-1 adapter",
            "fresh-process token IDs and adapter outputs pass frozen numeric parity",
        ],
        "checkpoint_is_candidate": False,
        "checkpoint_may_be_resumed_by_later_stage": False,
    }
    stages["mechanism_5step"]["runtime_gate"] = {
        "exact_pair_ids_source": "cpu_contract.stages.mechanism_5step.mechanism_pair_ids",
        "dev_must_be_absent": True,
        "shuffle_must_be_false": True,
        "finite_steps": [1, 2, 3, 4, 5],
        "step0_and_step5_pair_evaluation_required": True,
        "mean_reward_margin_must_increase": True,
        "minimum_improved_pairs": 3,
        "pair_count": 4,
        "checkpoint_is_candidate": False,
        "checkpoint_may_be_resumed_by_smoke": False,
    }
    stages["bounded_smoke_30step"]["runtime_gate"] = {
        "must_start_in_new_process_from_merged_s1": True,
        "forbid_g2_or_mechanism_checkpoint_input": True,
        "required_checkpoint_reload_steps": [15, 30],
        "candidate_steps": [15, 30],
        "native_trainer_eval_is_not_selector": True,
    }
    return stages, config_identities


def build_binding(
    *,
    run_root: Path,
    prior_mechanism_failure_receipt_path: Path,
    run_contract_path: Path = DEFAULT_RUN_CONTRACT,
    promotion_path: Path = DEFAULT_PROMOTION,
    downstream_key_path: Path = DEFAULT_DOWNSTREAM_KEY,
    export_manifest_path: Path = DEFAULT_EXPORT_MANIFEST,
    cpu_validator_path: Path = DEFAULT_CPU_VALIDATOR,
    model_path: Path | None = None,
    ms_swift_root: Path = DEFAULT_MS_SWIFT_ROOT,
    stage_runner_path: Path = DEFAULT_STAGE_RUNNER,
    preference_evaluator_path: Path = DEFAULT_PREFERENCE_EVALUATOR,
    require_fresh_paths: bool = True,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = _canonical_directory(run_root, "Day 23 run root")
    binding_dir = (root / "binding").resolve()
    _require(binding_dir.parent == root, "binding directory escaped the run root")
    contract, data_manifest, processor_summary, cpu_identity = _validate_cpu_bundle(
        run_contract_path, cpu_validator_path
    )
    parent_model = Path(
        str(model_path or contract["parent"]["remote_merged_export_path"])
    )
    remote_parent = _validate_remote_parent(
        contract,
        promotion_path=promotion_path,
        downstream_key_path=downstream_key_path,
        export_manifest_path=export_manifest_path,
        model_path=parent_model,
    )
    prior_mechanism_failure = _validate_prior_mechanism_failure(
        prior_mechanism_failure_receipt_path
    )
    correction_claim_path = (
        Path(prior_mechanism_failure["path"]).parent
        / "topology-correction-attempt2-claim.json"
    ).resolve()
    _require(
        root not in parent_model.resolve().parents
        and parent_model.resolve() not in root.parents
        and root not in BOOTCAMP_ROOT.resolve().parents
        and BOOTCAMP_ROOT.resolve() not in root.parents,
        "run root must be disjoint from the model and source trees",
    )
    configs, outputs = _build_configs(contract, model_path=parent_model, run_root=root)
    if require_fresh_paths:
        _require(
            not correction_claim_path.exists()
            and not correction_claim_path.is_symlink(),
            "the global mechanism correction attempt has already been claimed",
        )
        _require(not binding_dir.exists(), f"binding directory already exists: {binding_dir}")
        for output in outputs.values():
            _require(not output.exists(), f"stage output already exists: {output}")
        for stage in STAGE_ORDER:
            evidence = root / "evidence" / STAGE_SLUGS[stage]
            _require(not evidence.exists(), f"stage evidence already exists: {evidence}")
        _require(not (root / "evidence" / "selection").exists(), "selection evidence already exists")
        _require(not (root / "evidence" / "heldout").exists(), "heldout evidence already exists")
    runtime_parse = _parse_configs(
        configs,
        ms_swift_root=ms_swift_root,
        mechanism_pair_ids=contract["stages"]["mechanism_5step"]["mechanism_pair_ids"],
    )
    runner = _producer_identity(stage_runner_path, "Day 23 GPU stage runner")
    evaluator = _producer_identity(
        preference_evaluator_path, "Day 23 preference evaluator"
    )
    stages, config_identities = _build_stage_contracts(
        run_root=root,
        binding_dir=binding_dir,
        configs=configs,
        outputs=outputs,
    )
    receipts = {stage: _receipt_paths(root, stage) for stage in STAGE_ORDER}
    selector = _build_selector_contract(
        run_root=root,
        data_manifest=data_manifest,
        processor_summary=processor_summary,
        evaluator_identity=evaluator,
        smoke_config_identity=config_identities["bounded_smoke_30step"],
        smoke_receipts=receipts["bounded_smoke_30step"],
    )
    heldout = _build_heldout_contract(
        run_root=root,
        data_manifest=data_manifest,
        selector_contract=selector,
    )
    binding_manifest_path = (binding_dir / "gpu-execution-binding.json").resolve()
    transition_policy = {
        "stage_order": list(STAGE_ORDER),
        "g2_one_step": {
            "previous_gate": "binding.binding_sha256",
            "success_required_for": "mechanism_5step",
        },
        "mechanism_5step": {
            "previous_gate_receipt_path": receipts["g2_one_step"]["success"],
            "previous_checkpoint_reload_receipt_path": str(
                Path(receipts["g2_one_step"]["evidence_dir"])
                / "checkpoint-1-reload-receipt.json"
            ),
            "success_required_for": "bounded_smoke_30step",
            "output_forbidden_as_later_model_or_resume_input": True,
        },
        "bounded_smoke_30step": {
            "previous_gate_receipt_path": receipts["mechanism_5step"]["success"],
            "previous_checkpoint_reload_receipt_path": str(
                Path(receipts["mechanism_5step"]["evidence_dir"])
                / "checkpoint-5-integrity-receipt.json"
            ),
            "success_required_for": "dev_checkpoint_selection",
            "must_reinitialize_fresh_lora_from_remote_parent": True,
        },
        "dev_checkpoint_selection": {
            "previous_gate_receipt_path": receipts["bounded_smoke_30step"]["success"],
            "success_receipt_path": selector["selection_receipt"]["path"],
            "success_required_for": "heldout_one_shot",
        },
        "heldout_one_shot": {
            "previous_gate_receipt_path": selector["selection_receipt"]["path"],
            "success_receipt_path": heldout["final_receipt"]["path"],
        },
        "failure_policy": "any_missing_invalid_or_nonfinite_predecessor_blocks_all_successors",
    }
    evidence_contract = {
        "status": "frozen_protocol_runtime_receipts_pending",
        "producers": {
            "stage_runner_and_live_runtime_hook": runner,
            "preference_evaluator": evaluator,
        },
        "stage_runner_cli": {
            "environment": {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "NPROC_PER_NODE": "2",
            },
            "python": runtime_parse["python_executable"]["path"],
            "module": "torch.distributed.run",
            "launcher_arguments": [
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=2",
            ],
            "script": runner["path"],
            "arguments": [
                "--binding",
                str(binding_manifest_path),
                "--stage",
                "{g2_one_step|mechanism_5step|bounded_smoke_30step}",
                "--execute",
                "RUN_GPU_OPTIMIZER",
            ],
            "effective_command_shape": (
                "CUDA_VISIBLE_DEVICES=0,1 <python> -m torch.distributed.run "
                "--standalone --nnodes=1 --nproc_per_node=2 <script> <arguments>"
            ),
        },
        "common_receipt": {
            "schema_name": "day23.qwen35_dpo_gpu_stage_receipt",
            "schema_version": 1,
            "self_hash_field": "receipt_sha256",
            "exclusive_create_no_overwrite": True,
            "required_fields": [
                "schema_name",
                "schema_version",
                "status",
                "stage",
                "binding_path",
                "binding_file_sha256",
                "binding_sha256",
                "cpu_contract_sha256",
                "executable_config_path",
                "executable_config_file_sha256",
                "remote_parent_files_sha256",
                "process_identity",
                "runtime_identity",
                "previous_gate_receipt_sha256",
                "mechanism_protocol_correction",
                "mechanism_correction_claim",
                "evidence",
                "receipt_sha256",
            ],
            "process_identity_required_fields": [
                "pid",
                "proc_start_ticks",
                "boot_id",
                "python_executable_file_sha256",
            ],
            "runtime_identity_cross_binding": {
                "python": "path/file_sha256/version equal binding.runtime_parse.python_executable",
                "ms_swift_checkout": "path/commit/clean/import_file agree with binding.runtime_parse",
                "package_versions": "exactly equal binding.runtime_parse.package_versions",
            },
        },
        "receipts": receipts,
        "previous_gate_rules": {
            "g2_one_step": "bind directly to binding_sha256; previous receipt is null",
            "mechanism_5step": (
                "bind exact g2_one_step success receipt plus checkpoint-1 independent "
                "fresh-process reload evaluation"
            ),
            "bounded_smoke_30step": (
                "bind exact mechanism_5step success receipt plus checkpoint-5 independent "
                "fresh-process reload evaluation"
            ),
        },
        "checkpoint_receipts": {
            "g2_one_step": {
                "1": str(
                    Path(receipts["g2_one_step"]["evidence_dir"])
                    / "checkpoint-1-reload-receipt.json"
                )
            },
            "mechanism_5step": {
                "5": str(
                    Path(receipts["mechanism_5step"]["evidence_dir"])
                    / "checkpoint-5-integrity-receipt.json"
                )
            },
            "bounded_smoke_30step": {
                str(step): str(
                    (root / "evidence" / "selection" / f"dev-checkpoint-{step}.json").resolve()
                )
                for step in (15, 30)
            },
        },
        "fail_closed_checks": [
            "recompute binding self-hash and executable config raw file hash before launch",
            "reject forbidden CLI keys, nulls, empty lists, placeholders, relative paths, and unknown args",
            "reject an existing/symlinked stage output or evidence directory",
            "reject any model/adapters/resume input other than the bound merged S1 parent plus fresh LoRA",
            "reject a stage process that is not fresh or whose predecessor receipt is missing/drifted",
            "seal exactly one immutable success or failure receipt; never overwrite either",
            "never authorize a successor from a failure receipt",
        ],
    }
    binding: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "gpu_execution_bound",
        "binding_id": object_sha256(
            {
                "cpu_contract_sha256": EXPECTED_CPU_CONTRACT_SHA256,
                "run_root": str(root),
                "remote_parent_files_sha256": remote_parent["merged_export"][
                    "files_sha256"
                ],
                "prior_mechanism_failure_receipt_sha256": (
                    prior_mechanism_failure["receipt_sha256"]
                ),
            }
        ),
        "implementation": _file_identity(Path(__file__).resolve()),
        "cpu_contract": cpu_identity,
        "runtime_parse": runtime_parse,
        "user_authorized_gpu_amendment": {
            "authorized_at_local": "2026-08-14",
            "request": "尽量高batch和双卡跑训练，当然也要给足余量",
            "scope": (
                "GPU execution topology/microbatch plus quarantined G2 warmup "
                "correction and one append-only mechanism microbatch sensitivity "
                "retry; CPU contract remains immutable"
            ),
            "topology": "two_process_DDP_one_visible_GPU_per_rank",
            "world_size": GPU_WORLD_SIZE,
            "stage_train_batches": {
                "g2_one_step": {
                    "per_device": GPU_PER_DEVICE_TRAIN_BATCH,
                    "gradient_accumulation_steps": GPU_GRADIENT_ACCUMULATION,
                    "nominal_global": 8,
                    "realized_global_range": [8, 8],
                    "dataset_records": 8,
                    "batches_per_rank": 1,
                    "microbatches_per_optimizer_step": 1,
                    "partial_epoch_flush": False,
                    "purpose": "high-batch memory stress only",
                },
                "mechanism_5step": {
                    "per_device": GPU_MECHANISM_PER_DEVICE_TRAIN_BATCH,
                    "gradient_accumulation_steps": (
                        GPU_MECHANISM_GRADIENT_ACCUMULATION
                    ),
                    "nominal_global": 8,
                    "realized_global_range": [4, 4],
                    "dataset_records": 4,
                    "batches_per_rank": 2,
                    "microbatches_per_optimizer_step": 2,
                    "partial_epoch_flush": True,
                    "purpose": (
                        "exact four-pair mechanism gate using the frozen B1 "
                        "microbatch shape; configured GA4 is flushed after two "
                        "microbatches per rank at each epoch boundary"
                    ),
                },
                "bounded_smoke_30step": {
                    "per_device": GPU_PER_DEVICE_TRAIN_BATCH,
                    "gradient_accumulation_steps": GPU_GRADIENT_ACCUMULATION,
                    "nominal_global": 8,
                    "realized_global_range": [2, 8],
                    "dataset_records": 154,
                    "batches_per_rank_per_epoch": 20,
                    "partial_batch_global_records": 2,
                    "partial_batch_optimizer_step": 20,
                },
            },
            "per_device_eval_batch_size": GPU_PER_DEVICE_EVAL_BATCH,
            "bounded_smoke_nominal_global_train_batch_size": 8,
            "frozen_nominal_effective_global_train_batch_size": 8,
            "g2_runtime_correctness_amendment": {
                "warmup_ratio": 0.0,
                "reason": (
                    "max_steps=1 with the frozen positive ratio rounds to one zero-LR "
                    "warmup step and cannot prove a parameter update"
                ),
                "mechanism_and_smoke_schedules_changed": False,
            },
            "sampling_amendment": (
                "G2 uses the first 8 frozen train rows to exercise the B4-per-rank "
                "memory path; mechanism remains the exact first four rows"
            ),
            "bounded_smoke_nominal_batch_preserved": True,
            "distributed_sampler_padding": "must_be_attested_in_runtime_receipt",
            "minimum_actual_free_memory_fraction_per_rank": 0.15,
            "mechanism_protocol_correction": {
                "attempt_name": "mechanism_5step_topology_correction_attempt2",
                "attempt_number": 2,
                "append_only": True,
                "one_shot": True,
                "third_attempt_forbidden": True,
                "execution_claim": {
                    "path": str(correction_claim_path),
                    "schema_name": "day23.qwen35_dpo_mechanism_correction_claim",
                    "schema_version": 1,
                    "self_hash_field": "claim_sha256",
                    "exclusive_create_no_overwrite": True,
                    "producer": "bound_stage_runner_rank0_before_mechanism_model_load",
                },
                "prior_attempt": prior_mechanism_failure,
                "prior_failure_remains_scientific_evidence": True,
                "prior_checkpoint_forbidden_as_candidate_or_resume_source": True,
                "changed_fields": {
                    "per_device_train_batch_size": {"from": 2, "to": 1},
                    "gradient_accumulation_steps": {"from": 1, "to": 4},
                },
                "unchanged_fields": [
                    "dataset",
                    "pair_ids",
                    "seed",
                    "data_seed",
                    "learning_rate",
                    "warmup_ratio",
                    "max_steps",
                    "gate_thresholds",
                    "remote_parent",
                ],
                "interpretation_if_pass": (
                    "topology-sensitive mechanism pass; report attempt-1 failure "
                    "and attempt-2 pass together"
                ),
                "interpretation_if_fail": "terminal mechanism NO-GO; no third attempt",
            },
        },
        "remote_parent": remote_parent,
        "paths": {
            "bootcamp_root": str(BOOTCAMP_ROOT.resolve()),
            "run_root": str(root),
            "binding_dir": str(binding_dir),
            "binding_manifest": str(binding_manifest_path),
            "outputs_root": str((root / "outputs").resolve()),
            "evidence_root": str((root / "evidence").resolve()),
        },
        "stages": stages,
        "transition_policy": transition_policy,
        "selector_contract": selector,
        "heldout_ledger_contract": heldout,
        "evidence_contract": evidence_contract,
        "claim_boundary": {
            "frozen_cpu_contract_modified": False,
            "remote_s1_payload_bytes_verified": True,
            "real_remote_rlhf_arguments_json_parse_passed": True,
            "model_weights_loaded_by_binder": False,
            "optimizer_step_run_by_binder": False,
            "g1_policy_reference_freeze_runtime_proven": False,
            "g2_memory_one_step_save_reload_proven": False,
            "gpu_optimizer_ready_now": False,
            "g2_authorization": (
                "only_stage_runner_after_live_G1_pre_forward_checks_and_explicit_"
                "RUN_GPU_OPTIMIZER_token"
            ),
            "mechanism_authorization": "only_after_verified_g2_success_receipt",
            "bounded_smoke_authorization": "only_after_verified_mechanism_success_receipt",
            "heldout_authorization": "only_after_one_immutable_dev_selection_receipt",
        },
    }
    binding["binding_sha256"] = object_sha256(binding)
    payloads = {
        CONFIG_FILENAMES[stage]: json_bytes(configs[stage]) for stage in STAGE_ORDER
    }
    payloads[binding_manifest_path.name] = json_bytes(binding)
    for stage in STAGE_ORDER:
        identity = binding["stages"][stage]["executable_config"]
        payload = payloads[CONFIG_FILENAMES[stage]]
        _require(
            hashlib.sha256(payload).hexdigest() == identity["file_sha256"]
            and len(payload) == identity["bytes"],
            f"{stage} config identity is internally inconsistent",
        )
    return binding, payloads


def apply_bundle(
    binding_dir: Path, payloads: Mapping[str, bytes], *, mode: str
) -> None:
    target = binding_dir.resolve()
    _require(target.parent.is_dir(), f"binding parent is missing: {target.parent}")
    if mode == "build":
        try:
            target.mkdir(mode=0o755)
        except FileExistsError as error:
            raise Day23GPUBindingError(f"refusing to overwrite binding directory: {target}") from error
        for name, payload in payloads.items():
            path = target / name
            try:
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as error:
                raise Day23GPUBindingError(f"refusing to overwrite binding file: {path}") from error
        evidence_root = target.parent / "evidence"
        evidence_root.mkdir(mode=0o700)
        for name in (*STAGE_SLUGS.values(), "selection", "heldout"):
            (evidence_root / name).mkdir(mode=0o700)
    elif mode == "check":
        _require(target.is_dir() and not target.is_symlink(), f"binding directory is missing: {target}")
        actual_names = sorted(path.name for path in target.iterdir() if path.is_file())
        _require(actual_names == sorted(payloads), "binding directory file inventory drifted")
        for name, payload in payloads.items():
            path = target / name
            _require(path.read_bytes() == payload, f"bound GPU artifact drifted: {path}")
        evidence_root = target.parent / "evidence"
        _require(
            evidence_root.is_dir() and not evidence_root.is_symlink(),
            f"bound evidence root is missing: {evidence_root}",
        )
        for name in (*STAGE_SLUGS.values(), "selection", "heldout"):
            path = evidence_root / name
            _require(
                path.is_dir() and not path.is_symlink(),
                f"bound evidence directory is missing: {path}",
            )
    else:
        raise Day23GPUBindingError(f"unsupported mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "check"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--prior-mechanism-failure-receipt", type=Path, required=True
    )
    parser.add_argument("--run-contract", type=Path, default=DEFAULT_RUN_CONTRACT)
    parser.add_argument("--promotion-manifest", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--downstream-key", type=Path, default=DEFAULT_DOWNSTREAM_KEY)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    parser.add_argument("--cpu-validator", type=Path, default=DEFAULT_CPU_VALIDATOR)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ms-swift-root", type=Path, default=DEFAULT_MS_SWIFT_ROOT)
    parser.add_argument("--stage-runner", type=Path, default=DEFAULT_STAGE_RUNNER)
    parser.add_argument(
        "--preference-evaluator", type=Path, default=DEFAULT_PREFERENCE_EVALUATOR
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binding_dir = (args.run_root.expanduser().resolve() / "binding").resolve()
    try:
        binding, payloads = build_binding(
            run_root=args.run_root,
            prior_mechanism_failure_receipt_path=(
                args.prior_mechanism_failure_receipt
            ),
            run_contract_path=args.run_contract,
            promotion_path=args.promotion_manifest,
            downstream_key_path=args.downstream_key,
            export_manifest_path=args.export_manifest,
            cpu_validator_path=args.cpu_validator,
            model_path=args.model,
            ms_swift_root=args.ms_swift_root,
            stage_runner_path=args.stage_runner,
            preference_evaluator_path=args.preference_evaluator,
            require_fresh_paths=args.mode == "build",
        )
        apply_bundle(binding_dir, payloads, mode=args.mode)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        cpu_contract.Day23ContractError,
        Day23GPUBindingError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": binding["status"],
                "binding_path": str(binding_dir / "gpu-execution-binding.json"),
                "binding_sha256": binding["binding_sha256"],
                "stages": list(STAGE_ORDER),
                "gpu_optimizer_ready_now": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
