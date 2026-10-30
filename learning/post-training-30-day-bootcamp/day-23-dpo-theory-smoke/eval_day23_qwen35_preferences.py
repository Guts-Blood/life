#!/usr/bin/env python3
"""Evaluate frozen Day 23 DPO pairs with one GPU and no generation.

The evaluator is deliberately separate from the trainer.  It loads the
promoted merged S1, attaches exactly one saved policy adapter, and evaluates
the same model twice: once with the policy adapter active and once inside
``PeftModel.disable_adapter()`` for the frozen reference.  It never creates an
optimizer, generates text, executes candidate code, or opens the preference
heldout before an immutable O_EXCL claim exists.

Runtime inputs are fail-closed evidence artifacts:

* a self-hashed Day 23 GPU binding and its raw executable config;
* a self-hashed checkpoint receipt with a complete byte manifest;
* for heldout only, a self-hashed checkpoint-selection receipt plus three
  distinct immutable paths: claim, evaluation, and consumption receipt.

The exact binding schema is owned by ``bind_day23_qwen35_gpu.py``.  This file
does not manufacture a binding or infer missing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import day23_contract as contract
import day23_rlhf_template as dpo_template


SCHEMA_VERSION = 1
BINDING_SCHEMA = "day23.qwen35_dpo_gpu_execution_binding"
BINDING_STATUS = "gpu_execution_bound"
EVALUATION_SCHEMA = "day23.qwen35_dpo_preference_evaluation"
PAIR_SCHEMA = "day23.qwen35_dpo_pair_evaluation"
SELECTION_SCHEMA = "day23.qwen35_dpo_checkpoint_selection"
CLAIM_SCHEMA = "day23.qwen35_dpo_heldout_claim"
CONSUMPTION_SCHEMA = "day23.qwen35_dpo_heldout_consumption"
CORRECTION_CLAIM_SCHEMA = "day23.qwen35_dpo_mechanism_correction_claim"
EXPECTED_PRIOR_MECHANISM_FAILURE_PATH = Path(
    "/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T091033Z/"
    "evidence/mechanism-5step/failure-receipt.json"
)
TEMPLATE_ALIAS = dpo_template.TARGET_TEMPLATE_ALIAS
GENERATION_MAX_NEW_TOKENS = 256
LENGTH_MATCH_TOLERANCE = 8
EXPECTED_LORA_TENSORS = 496
EXPECTED_LORA_PARAMS = 16_232_448
RELOAD_LOGP_ABSOLUTE_TOLERANCE = 0.2
RELOAD_LOGP_RELATIVE_TOLERANCE = 0.001
RELOAD_REWARD_MARGIN_TOLERANCE = 0.05
MECHANISM_PAIR_IDS = (
    "mbpp:task:602:s1pair:37673b32b9385aea",
    "mbpp:task:604:s1pair:388a8daec7e97c5a",
    "mbpp:task:605:s1pair:9ae278fe596ed4d4",
    "mbpp:task:610:s1pair:9e57fb63f7544777",
)
SPLIT_STAGE = {
    "g2": "g2_one_step",
    "mechanism": "mechanism_5step",
    "dev": "bounded_smoke_30step",
    "heldout": "bounded_smoke_30step",
}
EXPECTED_STEPS = {
    "g2": {1},
    "mechanism": {5},
    "dev": {15, 30},
    "heldout": {15, 30},
}
REQUIRED_CONFIG = {
    "rlhf_type": "dpo",
    "tuner_type": "lora",
    "template": TEMPLATE_ALIAS,
    "model_type": "qwen3_5",
    "loss_type": "sigmoid",
    "beta": 0.1,
    "max_length": 512,
    "loss_scale": "default+ignore_empty_think",
    "packing": False,
    "padding_free": False,
    "enable_thinking": False,
    "add_non_thinking_prefix": True,
    "add_version": False,
}
FORBIDDEN_CONFIG_KEYS = {
    "adapters",
    "ref_adapters",
    "ref_model",
    "resume_from_checkpoint",
}


class Day23PreferenceEvaluationError(ValueError):
    """An evaluator input, runtime invariant, or result failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23PreferenceEvaluationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty text")
    _require("\x00" not in value, f"{label} contains a NUL byte")
    return str(value)


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return int(value)


def _sha256(value: Any, label: str) -> str:
    value = _text(value, label)
    _require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase bare SHA-256",
    )
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Mapping[str, Any], self_field: str | None = None) -> str:
    payload = dict(value)
    if self_field is not None:
        payload.pop(self_field, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Day23PreferenceEvaluationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_absolute(), f"{label} path must be absolute: {path}")
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day23PreferenceEvaluationError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_absolute(), f"{label} path must be absolute: {path}")
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day23PreferenceEvaluationError(f"cannot read {label}: {path}") from error
    for line_number, line in enumerate(lines, 1):
        _require(bool(line), f"blank {label} row: {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day23PreferenceEvaluationError(
                f"invalid {label} JSON at row {line_number}"
            ) from error
        _require(isinstance(row, dict), f"non-object {label} row: {line_number}")
        rows.append(row)
    _require(bool(rows), f"{label} is empty")
    return rows


def verify_self_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256(value, field)
    _require(actual == expected, f"{label}.{field} does not bind its contents")
    return expected


def _absolute_path(value: Any, label: str, *, kind: str | None = None) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    path = Path(_text(value, label)).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute")
    resolved = path.resolve()
    if kind == "file":
        _require(resolved.is_file() and not path.is_symlink(), f"{label} file is missing: {resolved}")
    elif kind == "dir":
        _require(resolved.is_dir(), f"{label} directory is missing: {resolved}")
    return resolved


def _validate_mechanism_correction_claim(
    path: Path,
    *,
    binding_sha256: str,
    config_file_sha256: str,
    prior_receipt_sha256: str,
) -> dict[str, Any]:
    value = load_json(path, "mechanism correction claim")
    claim_sha = verify_self_hash(
        value, "claim_sha256", "mechanism correction claim"
    )
    _require(
        value.get("schema_name") == CORRECTION_CLAIM_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "claimed_for_execution"
        and value.get("attempt_name")
        == "mechanism_5step_topology_correction_attempt2"
        and value.get("attempt_number") == 2
        and value.get("binding_sha256") == binding_sha256
        and value.get("executable_config_file_sha256")
        == config_file_sha256
        and value.get("prior_failure_receipt_sha256")
        == prior_receipt_sha256
        and value.get("one_shot") is True
        and value.get("third_attempt_forbidden") is True,
        "mechanism correction claim identity drifted",
    )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "claim_sha256": claim_sha,
        "binding_sha256": binding_sha256,
        "status": "claimed_for_execution",
    }


def _bound_json(
    entry: Mapping[str, Any], label: str, *, self_field: str | None = None
) -> tuple[Path, dict[str, Any], str]:
    path = _absolute_path(entry.get("path"), f"{label}.path", kind="file")
    actual_file_sha = file_sha256(path)
    expected_file_sha = _sha256(entry.get("file_sha256"), f"{label}.file_sha256")
    _require(actual_file_sha == expected_file_sha, f"{label} file hash drifted")
    value = load_json(path, label)
    if self_field is not None:
        actual_self = verify_self_hash(value, self_field, label)
        bound_self = entry.get(self_field)
        if bound_self is not None:
            _require(
                actual_self == _sha256(bound_self, f"bound {label}.{self_field}"),
                f"{label} content hash drifted from binding",
            )
    return path, value, actual_file_sha


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_exclusive_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    self_field: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    _require(path.is_absolute(), f"{label} path must be absolute")
    _require(path.parent.is_dir(), f"{label} parent is missing: {path.parent}")
    _require(self_field not in value, f"{label} already contains {self_field}")
    sealed = dict(value)
    sealed[self_field] = object_sha256(sealed)
    payload = _json_bytes(sealed)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise Day23PreferenceEvaluationError(
            f"refusing to overwrite immutable {label}: {path}"
        ) from error
    except OSError as error:
        raise Day23PreferenceEvaluationError(f"cannot create {label}: {path}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise Day23PreferenceEvaluationError(f"cannot seal {label}: {path}") from error
    return sealed, hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_identity() -> dict[str, Any]:
    stat = Path("/proc/self/stat")
    boot = Path("/proc/sys/kernel/random/boot_id")
    _require(stat.is_file() and boot.is_file(), "Linux process identity files are missing")
    raw = stat.read_text(encoding="utf-8").strip()
    _require(") " in raw, "cannot parse /proc/self/stat")
    fields = raw.split(") ", 1)[1].split()
    _require(len(fields) > 19 and fields[19].isdigit(), "process start ticks are invalid")
    python = Path(sys.executable).resolve()
    return {
        "pid": os.getpid(),
        "proc_start_ticks": int(fields[19]),
        "boot_id": boot.read_text(encoding="utf-8").strip(),
        "python_executable_file_sha256": file_sha256(python),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _mean(values: Sequence[float], label: str) -> float:
    _require(bool(values), f"cannot calculate empty mean: {label}")
    result = math.fsum(float(value) for value in values) / len(values)
    _require(math.isfinite(result), f"non-finite mean: {label}")
    return result


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Day23PreferenceEvaluationError(f"{label} is not numeric") from error
    _require(math.isfinite(result), f"{label} is non-finite")
    return result


def _reload_logp_tolerance(actual: float, expected: float) -> float:
    # The runner evaluates chosen+rejected together (B=2); the independent
    # evaluator uses two B=1 forwards.  BF16 reduction order differs slightly,
    # while exact adapter key/value parity remains a separate hard gate.
    return max(
        RELOAD_LOGP_ABSOLUTE_TOLERANCE,
        RELOAD_LOGP_RELATIVE_TOLERANCE * max(abs(actual), abs(expected)),
    )


def _directory_manifest(path: Path) -> dict[str, dict[str, Any]]:
    _require(path.is_absolute() and path.is_dir(), f"checkpoint directory is missing: {path}")
    files: dict[str, dict[str, Any]] = {}
    for item in sorted(path.rglob("*")):
        _require(not item.is_symlink(), f"checkpoint contains a symlink: {item}")
        if item.is_dir():
            continue
        _require(item.is_file(), f"checkpoint contains a non-regular entry: {item}")
        relative = item.relative_to(path).as_posix()
        files[relative] = {"bytes": item.stat().st_size, "sha256": file_sha256(item)}
    _require(bool(files), "checkpoint directory is empty")
    return files


def _verify_remote_parent_payload(binding: Mapping[str, Any]) -> Path:
    remote = _mapping(binding.get("remote_parent"), "binding.remote_parent")
    merged = _mapping(remote.get("merged_export"), "binding.remote_parent.merged_export")
    root = _absolute_path(merged.get("path"), "remote merged S1 export", kind="dir")
    _require(not root.is_symlink(), "remote merged S1 export cannot be a symlink")
    expected = _mapping(merged.get("files"), "remote merged S1 files")
    actual: dict[str, dict[str, Any]] = {}
    for item in sorted(root.rglob("*")):
        _require(not item.is_symlink(), f"remote merged S1 contains a symlink: {item}")
        if item.is_dir():
            continue
        _require(item.is_file(), f"remote merged S1 contains a non-file: {item}")
        relative = item.relative_to(root).as_posix()
        if relative == "S1-EXPORT-MANIFEST.json":
            continue
        actual[relative] = {
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
    _require(actual == dict(expected), "remote merged S1 payload bytes drifted")
    _require(
        contract.object_sha256(actual) == merged.get("files_sha256"),
        "remote merged S1 aggregate hash drifted",
    )
    _require(
        Path(_text(remote.get("model_argument"), "remote parent model argument")).resolve()
        == root,
        "remote parent model argument drifted",
    )
    return root


def _git_identity(checkout: Path) -> dict[str, Any]:
    _require(checkout.is_dir(), f"ms-swift checkout is missing: {checkout}")
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day23PreferenceEvaluationError("cannot inspect pinned ms-swift checkout") from error
    _require(commit == contract.MS_SWIFT_COMMIT, "ms-swift commit drifted")
    _require(status == "", "ms-swift checkout is dirty")
    return {"path": str(checkout), "commit": commit, "clean": True}


def _length_bucket(chosen_tokens: int, rejected_tokens: int) -> str:
    maximum = max(chosen_tokens, rejected_tokens)
    if maximum <= 64:
        return "<=64"
    if maximum <= 128:
        return "65-128"
    return ">128"


def _summary(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    _require(bool(rows), f"empty metric slice: {label}")
    margins = [_finite(row["reward_margin"], f"{label}.reward_margin") for row in rows]
    wins = sum(margin > 0 for margin in margins)
    ties = sum(margin == 0 for margin in margins)
    return {
        "pairs": len(rows),
        "wins": wins,
        "ties": ties,
        "pair_accuracy": wins / len(rows),
        "mean_reward_margin": _mean(margins, f"{label}.reward_margin"),
        "mean_chosen_response_tokens": _mean(
            [float(row["chosen_response_tokens"]) for row in rows],
            f"{label}.chosen_response_tokens",
        ),
        "mean_rejected_response_tokens": _mean(
            [float(row["rejected_response_tokens"]) for row in rows],
            f"{label}.rejected_response_tokens",
        ),
    }


def aggregate_results(
    rows: Sequence[Mapping[str, Any]], *, split: str, checkpoint_step: int
) -> dict[str, Any]:
    _require(bool(rows), "evaluation produced no pair results")
    matched = [row for row in rows if row["length_matched_eligible"] is True]
    _require(bool(matched), "length-matched slice is empty")
    matched_margins = [
        _finite(row["length_matched_reward_margin"], "length-matched margin")
        for row in matched
    ]
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "response_length": defaultdict(list),
        "source_family": defaultdict(list),
        "test_family": defaultdict(list),
        "quality_status": defaultdict(list),
    }
    for row in rows:
        groups["response_length"][str(row["length_bucket"])].append(row)
        groups["source_family"][str(row["source_family"])].append(row)
        groups["test_family"][str(row["test_family"])].append(row)
        groups["quality_status"][str(row["quality_status"])].append(row)
    slices = {
        dimension: {
            key: _summary(items, f"{dimension}:{key}")
            for key, items in sorted(values.items())
        }
        for dimension, values in groups.items()
    }
    overall = _summary(rows, "overall")
    result: dict[str, Any] = {
        "overall": overall,
        "length_matched": {
            "eligibility": f"abs(chosen_response_tokens-rejected_response_tokens)<={LENGTH_MATCH_TOLERANCE}",
            "eligible_pairs": len(matched),
            "ineligible_pairs": len(rows) - len(matched),
            "prefix_tokens_per_pair": "min(chosen_response_tokens,rejected_response_tokens)",
            "mean_reward_margin": _mean(matched_margins, "length-matched reward margin"),
        },
        "length_bucket_basis": "max(chosen_response_tokens,rejected_response_tokens)",
        "slices": slices,
    }
    if split == "dev":
        result["selector_metrics"] = {
            "dev_pair_accuracy": overall["pair_accuracy"],
            "dev_mean_reward_margin": overall["mean_reward_margin"],
            "dev_length_matched_margin": result["length_matched"]["mean_reward_margin"],
            "earlier_checkpoint": checkpoint_step,
        }
    return result


def _self_test() -> None:
    base = {
        "chosen_response_tokens": 10,
        "rejected_response_tokens": 12,
        "length_matched_eligible": True,
        "length_bucket": "<=64",
        "source_family": "source:a",
        "test_family": "test:a",
        "quality_status": "accepted|pending",
    }
    rows = [
        base | {"reward_margin": 0.5, "length_matched_reward_margin": 0.4},
        base | {"reward_margin": -0.25, "length_matched_reward_margin": -0.2},
    ]
    aggregate = aggregate_results(rows, split="dev", checkpoint_step=15)
    _require(aggregate["overall"]["pair_accuracy"] == 0.5, "self-test accuracy failed")
    _require(
        aggregate["selector_metrics"]["dev_mean_reward_margin"] == 0.125,
        "self-test margin failed",
    )
    value = {"schema_name": "self-test", "status": "pass"}
    sealed = value | {"self_sha256": object_sha256(value)}
    verify_self_hash(sealed, "self_sha256", "self-test")
    path_probe = Path("/tmp/day23-evaluator-pathlike-self-test")
    _require(
        _absolute_path(path_probe, "self-test PathLike") == path_probe.resolve(),
        "self-test PathLike normalization failed",
    )
    _require(
        _reload_logp_tolerance(-13.099804282188416, -12.958833694458008)
        == RELOAD_LOGP_ABSOLUTE_TOLERANCE,
        "self-test reload tolerance failed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--split", choices=tuple(SPLIT_STAGE))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--select-dev", action="store_true")
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--heldout-claim", type=Path)
    parser.add_argument("--heldout-evaluation", type=Path)
    parser.add_argument("--heldout-consumption", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    return parser


# Binding, checkpoint, data, and model execution helpers follow below.  They
# remain lazy with respect to torch/swift/peft so --help and --self-test are CPU
# only and never load model weights.


def _recursive_values(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            if item_key == key:
                result.append(item)
            result.extend(_recursive_values(item, key))
    elif isinstance(value, list):
        for item in value:
            result.extend(_recursive_values(item, key))
    return result


def _unique_recursive(value: Any, key: str, label: str) -> Any:
    matches = _recursive_values(value, key)
    _require(len(matches) == 1, f"{label} must contain exactly one {key}")
    return matches[0]


def _contains_scalar(value: Any, expected: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_scalar(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(item, expected) for item in value)
    return value == expected


def validate_binding(
    binding_path: Path, *, split: str
) -> dict[str, Any]:
    _require(binding_path.is_absolute(), "GPU binding path must be absolute")
    _require(not binding_path.is_symlink(), "GPU binding cannot be a symlink")
    binding_path = binding_path.resolve()
    binding = load_json(binding_path, "GPU binding")
    binding_sha = verify_self_hash(binding, "binding_sha256", "GPU binding")
    required_top = {
        "schema_name",
        "status",
        "binding_id",
        "implementation",
        "cpu_contract",
        "runtime_parse",
        "remote_parent",
        "paths",
        "stages",
        "transition_policy",
        "selector_contract",
        "heldout_ledger_contract",
        "evidence_contract",
        "claim_boundary",
        "binding_sha256",
    }
    _require(required_top <= set(binding), "GPU binding top-level schema is incomplete")
    _require(binding.get("schema_name") == BINDING_SCHEMA, "GPU binding schema drifted")
    _require(binding.get("schema_version") == SCHEMA_VERSION, "GPU binding version drifted")
    _require(binding.get("status") == BINDING_STATUS, "GPU binding is not executable")
    bound_paths = _mapping(binding.get("paths"), "binding.paths")
    _require(
        Path(_text(bound_paths.get("binding_manifest"), "binding.paths.binding_manifest")).resolve()
        == binding_path,
        "GPU binding path differs from its own path binding",
    )
    cpu_entry = _mapping(binding.get("cpu_contract"), "binding.cpu_contract")
    cpu_path, cpu_contract, cpu_file_sha = _bound_json(
        cpu_entry, "CPU contract", self_field="contract_sha256"
    )
    _require(
        cpu_contract.get("status") == "cpu_ready_gpu_pending",
        "CPU contract is not CPU-ready/GPU-pending",
    )
    validator_entry = _mapping(
        cpu_entry.get("independent_validator"),
        "binding CPU independent validator",
    )
    validator_path = _absolute_path(
        validator_entry.get("path"), "CPU independent validator", kind="file"
    )
    _require(
        file_sha256(validator_path)
        == _sha256(validator_entry.get("file_sha256"), "CPU validator file hash"),
        "CPU independent validator source drifted",
    )
    completed = subprocess.run(
        [sys.executable, str(validator_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        completed.returncode == 0,
        "CPU independent validator failed before evaluation: "
        + completed.stderr.strip(),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    _require(lines, "CPU independent validator emitted no result")
    try:
        validator_result = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise Day23PreferenceEvaluationError("CPU validator result is not JSON") from error
    _require(
        isinstance(validator_result, Mapping)
        and validator_result.get("status") == "valid_cpu_ready_gpu_pending"
        and contract.object_sha256(validator_result)
        == validator_entry.get("result_sha256"),
        "CPU independent validator result drifted from binding",
    )
    bootcamp_root = cpu_path.parents[3]
    _require(
        cpu_path
        == bootcamp_root
        / "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json",
        "CPU contract path does not establish the bootcamp root",
    )
    _require(
        Path(_text(bound_paths.get("bootcamp_root"), "binding.paths.bootcamp_root")).resolve()
        == bootcamp_root,
        "binding bootcamp root drifted",
    )
    stage_name = SPLIT_STAGE[split]
    stages = _mapping(binding.get("stages"), "binding.stages")
    stage_entry = _mapping(stages.get(stage_name), f"binding.stages.{stage_name}")
    config_entry = _mapping(
        stage_entry.get("executable_config"),
        f"binding.stages.{stage_name}.executable_config",
    )
    config_path, config, config_file_sha = _bound_json(
        config_entry, f"{stage_name} executable config"
    )
    declared_keys = config_entry.get("keys")
    _require(
        isinstance(declared_keys, list)
        and declared_keys == sorted(config)
        and len(declared_keys) == len(set(declared_keys)),
        f"{stage_name} executable config key inventory drifted",
    )
    for key, expected in REQUIRED_CONFIG.items():
        _require(config.get(key) == expected, f"executable config {key} drifted")
    expected_train_batch = 1 if stage_name == "mechanism_5step" else 4
    expected_gradient_accumulation = (
        4 if stage_name == "mechanism_5step" else 1
    )
    _require(
        config.get("per_device_train_batch_size") == expected_train_batch
        and config.get("per_device_eval_batch_size") == 4
        and config.get("gradient_accumulation_steps")
        == expected_gradient_accumulation,
        "bound dual-GPU batch amendment drifted",
    )
    amendment = _mapping(
        binding.get("user_authorized_gpu_amendment"),
        "binding GPU amendment",
    )
    stage_batch = _mapping(
        _mapping(
            amendment.get("stage_train_batches"),
            "binding stage train batches",
        ).get(stage_name),
        f"binding {stage_name} train batch",
    )
    correction = _mapping(
        amendment.get("mechanism_protocol_correction"),
        "binding mechanism protocol correction",
    )
    prior = _mapping(correction.get("prior_attempt"), "prior mechanism attempt")
    prior_path = _absolute_path(
        prior.get("path"), "prior mechanism failure receipt", kind="file"
    )
    prior_value = load_json(prior_path, "prior mechanism failure receipt")
    _require(
        prior_path == EXPECTED_PRIOR_MECHANISM_FAILURE_PATH
        and correction.get("attempt_number") == 2
        and correction.get("one_shot") is True
        and correction.get("third_attempt_forbidden") is True
        and verify_self_hash(
            prior_value,
            "receipt_sha256",
            "prior mechanism failure receipt",
        )
        == prior.get("receipt_sha256")
        == "600341cb77ee2bd97b2e4a00ae26a89720a9204b8266b86b6571c6c3181adc20"
        and file_sha256(prior_path) == prior.get("file_sha256")
        and prior_value.get("status") == "fail"
        and prior_value.get("stage") == "mechanism_5step",
        "mechanism correction provenance drifted",
    )
    if stage_name == "mechanism_5step":
        _require(
            stage_batch.get("per_device") == 1
            and stage_batch.get("gradient_accumulation_steps") == 4
            and stage_batch.get("nominal_global") == 8
            and stage_batch.get("realized_global_range") == [4, 4]
            and stage_batch.get("batches_per_rank") == 2
            and stage_batch.get("microbatches_per_optimizer_step") == 2
            and stage_batch.get("partial_epoch_flush") is True,
            "mechanism realized batch semantics drifted",
        )
    claim_spec = _mapping(
        correction.get("execution_claim"),
        "binding mechanism correction execution claim",
    )
    claim_path = _absolute_path(
        claim_spec.get("path"), "mechanism correction execution claim"
    )
    _require(
        claim_path
        == (prior_path.parent / "topology-correction-attempt2-claim.json").resolve()
        and claim_spec.get("schema_name") == CORRECTION_CLAIM_SCHEMA
        and claim_spec.get("schema_version") == 1
        and claim_spec.get("self_hash_field") == "claim_sha256"
        and claim_spec.get("exclusive_create_no_overwrite") is True,
        "mechanism correction claim contract drifted",
    )
    correction_claim: dict[str, Any] | None = None
    if split == "g2":
        _require(
            not claim_path.exists() and not claim_path.is_symlink(),
            "G2 evaluation found an already-claimed mechanism retry",
        )
    else:
        _require(
            claim_path.is_file() and not claim_path.is_symlink(),
            "mechanism correction execution claim is missing",
        )
        mechanism_config_sha = _mapping(
            _mapping(
                stages.get("mechanism_5step"), "binding mechanism stage"
            ).get("executable_config"),
            "binding mechanism executable config",
        ).get("file_sha256")
        correction_claim = _validate_mechanism_correction_claim(
            claim_path,
            binding_sha256=binding_sha,
            config_file_sha256=mechanism_config_sha,
            prior_receipt_sha256=prior["receipt_sha256"],
        )
    _require(not (FORBIDDEN_CONFIG_KEYS & set(config)), "forbidden null/default keys entered config")
    _require(
        all(value is not None and value != [] for value in config.values()),
        "executable config contains null or empty-list values",
    )
    _require(
        not any("__BIND_" in str(value) for value in config.values()),
        "executable config contains unresolved placeholders",
    )
    model_path = _absolute_path(config.get("model"), "executable config model", kind="dir")
    verified_model_path = _verify_remote_parent_payload(binding)
    _require(
        model_path == verified_model_path,
        "executable model is not the byte-verified remote parent",
    )
    _require(
        _contains_scalar(binding.get("remote_parent"), contract.MS_SWIFT_COMMIT),
        "binding remote parent does not bind pinned ms-swift commit",
    )
    runtime_parse = _mapping(binding.get("runtime_parse"), "binding.runtime_parse")
    _require(runtime_parse.get("status") == "pass", "runtime parse audit did not pass")
    _require(
        runtime_parse.get("scope") == "real_remote_RLHFArguments_JSON_parse_no_model_weights",
        "runtime parse scope drifted",
    )
    parsed_stage = _mapping(
        _mapping(runtime_parse.get("stages"), "runtime_parse.stages").get(stage_name),
        f"runtime_parse.stages.{stage_name}",
    )
    expected_realized_global_range = (
        [4, 4]
        if stage_name == "mechanism_5step"
        else ([2, 8] if stage_name == "bounded_smoke_30step" else [8, 8])
    )
    _require(
        parsed_stage.get("status") == "pass"
        and parsed_stage.get("per_device_train_batch_size")
        == expected_train_batch
        and parsed_stage.get("gradient_accumulation_steps")
        == expected_gradient_accumulation
        and parsed_stage.get("configured_nominal_global_train_batch_size")
        == 2 * expected_train_batch * expected_gradient_accumulation
        and parsed_stage.get("expected_realized_global_train_batch_size_range")
        == expected_realized_global_range,
        f"{stage_name} runtime parse batch semantics drifted",
    )
    _require(
        parsed_stage.get("executable_config_keys") == sorted(config),
        f"{stage_name} runtime parse key inventory drifted",
    )
    checkout_value = runtime_parse.get("ms_swift_checkout")
    checkout = _absolute_path(checkout_value, "ms_swift checkout", kind="dir")
    _require(
        runtime_parse.get("ms_swift_commit") == contract.MS_SWIFT_COMMIT,
        "bound ms-swift commit drifted",
    )
    _require(
        runtime_parse.get("ms_swift_clean") is True,
        "binding does not require a clean ms-swift checkout",
    )
    import_root = _absolute_path(
        runtime_parse.get("ms_swift_import_root"), "ms-swift import root", kind="dir"
    )
    _require(
        import_root == checkout / "swift",
        "bound ms-swift import root is not checkout/swift",
    )
    python_entry = _mapping(
        runtime_parse.get("python_executable"), "runtime_parse.python_executable"
    )
    python_path = _absolute_path(python_entry.get("path"), "bound Python")
    _require(python_path.is_file(), f"bound Python is missing: {python_path}")
    _require(
        file_sha256(python_path)
        == _sha256(python_entry.get("file_sha256"), "bound Python file hash"),
        "bound Python bytes drifted",
    )
    _require(
        python_path == Path(sys.executable).resolve(),
        f"evaluator Python differs from binding: {sys.executable}",
    )
    git_identity = _git_identity(checkout)
    producers = _mapping(
        _mapping(binding.get("evidence_contract"), "binding.evidence_contract").get(
            "producers"
        ),
        "binding.evidence_contract.producers",
    )
    evaluator_entry = _mapping(
        producers.get("preference_evaluator"), "binding preference evaluator"
    )
    evaluator_path = _absolute_path(
        evaluator_entry.get("path"), "bound preference evaluator", kind="file"
    )
    _require(
        evaluator_path == Path(__file__).resolve(),
        "binding points to a different preference evaluator",
    )
    _require(
        file_sha256(evaluator_path)
        == _sha256(
            evaluator_entry.get("file_sha256"),
            "bound preference evaluator file hash",
        ),
        "preference evaluator bytes drifted from binding",
    )
    return {
        "binding": binding,
        "binding_path": binding_path,
        "binding_file_sha256": file_sha256(binding_path),
        "binding_sha256": binding_sha,
        "cpu_contract": cpu_contract,
        "cpu_contract_path": cpu_path,
        "cpu_contract_file_sha256": cpu_file_sha,
        "bootcamp_root": bootcamp_root,
        "stage_name": stage_name,
        "stage_entry": stage_entry,
        "config": config,
        "config_path": config_path,
        "config_file_sha256": config_file_sha,
        "model_path": model_path,
        "ms_swift": git_identity,
        "runtime_parse": runtime_parse,
        "mechanism_protocol_correction": dict(correction),
        "mechanism_correction_claim": correction_claim,
    }


def validate_checkpoint_receipt(
    receipt_path: Path,
    context: Mapping[str, Any],
    *,
    split: str,
    checkpoint_step: int | None,
) -> dict[str, Any]:
    """Validate a bound training-stage receipt and select one saved checkpoint.

    The evaluator itself is the producer of the independent fresh-process
    reload receipt.  Its input must therefore be the stage runner's immutable
    success receipt, never a purported reload receipt (which would be a
    circular trust dependency).
    """
    _require(receipt_path.is_absolute(), "checkpoint receipt path must be absolute")
    _require(not receipt_path.is_symlink(), "checkpoint receipt cannot be a symlink")
    receipt_path = receipt_path.resolve()
    receipts = _mapping(
        _mapping(context["binding"].get("evidence_contract"), "binding.evidence_contract")
        .get("receipts"),
        "binding.evidence_contract.receipts",
    )
    stage_receipts = _mapping(
        receipts.get(context["stage_name"]),
        f"binding evidence receipts for {context['stage_name']}",
    )
    expected_stage_receipt = _absolute_path(
        stage_receipts.get("success"), "bound stage success receipt", kind="file"
    )
    _require(
        receipt_path == expected_stage_receipt,
        "checkpoint input must be the bound stage success receipt",
    )
    receipt = load_json(receipt_path, "checkpoint receipt")
    receipt_sha = verify_self_hash(receipt, "receipt_sha256", "checkpoint receipt")
    import run_day23_qwen35_gpu_stage as stage_runner

    producers = _mapping(
        _mapping(
            context["binding"].get("evidence_contract"),
            "binding.evidence_contract",
        ).get("producers"),
        "binding.evidence_contract.producers",
    )
    runner_bound = _mapping(
        producers.get("stage_runner_and_live_runtime_hook"),
        "bound stage runner producer",
    )
    runner_path = Path(stage_runner.__file__).resolve()
    _require(
        runner_path
        == _absolute_path(runner_bound.get("path"), "bound stage runner", kind="file")
        and file_sha256(runner_path) == runner_bound.get("file_sha256"),
        "stage receipt validator source drifted from the bound runner",
    )
    try:
        receipt = stage_runner.validate_stage_success_receipt_value(
            context["binding"],
            receipt,
            stage=context["stage_name"],
            binding_sha256=context["binding_sha256"],
        )
    except stage_runner.Day23GPUStageError as error:
        raise Day23PreferenceEvaluationError(
            f"checkpoint stage success receipt failed strict validation: {error}"
        ) from error
    binding_links = [
        receipt.get("binding_sha256"),
        receipt.get("binding", {}).get("binding_sha256")
        if isinstance(receipt.get("binding"), Mapping)
        else None,
    ]
    binding_links = [value for value in binding_links if value is not None]
    _require(
        bool(binding_links)
        and all(value == context["binding_sha256"] for value in binding_links),
        "checkpoint receipt binding hash drifted",
    )
    _require(receipt.get("stage") == context["stage_name"], "checkpoint receipt stage drifted")
    config_links = [receipt.get("executable_config_file_sha256")]
    if isinstance(receipt.get("executable_config"), Mapping):
        config_links.append(receipt["executable_config"].get("file_sha256"))
    config_links = [value for value in config_links if value is not None]
    _require(
        bool(config_links)
        and all(value == context["config_file_sha256"] for value in config_links),
        "checkpoint receipt config hash drifted",
    )
    claim = _mapping(receipt.get("claim_boundary"), "checkpoint receipt claim boundary")
    _require(claim.get("runtime_stage_passed") is True, "training runtime stage did not pass")
    checkpoints = receipt.get("checkpoints")
    _require(
        isinstance(checkpoints, list) and checkpoints,
        "stage receipt checkpoint inventory is absent",
    )
    by_step: dict[int, Mapping[str, Any]] = {}
    for item in checkpoints:
        entry = _mapping(item, "stage receipt checkpoint")
        step_value = _integer(entry.get("global_step"), "checkpoint global_step")
        _require(step_value not in by_step, "stage receipt repeats a checkpoint step")
        by_step[step_value] = entry
    eligible_steps = EXPECTED_STEPS[split]
    if checkpoint_step is None:
        _require(len(eligible_steps) == 1, "--checkpoint-step is required for this split")
        checkpoint_step = next(iter(eligible_steps))
    _require(checkpoint_step in eligible_steps, f"checkpoint step {checkpoint_step} is ineligible for {split}")
    _require(checkpoint_step in by_step, "requested checkpoint is absent from the stage receipt")
    checkpoint = by_step[checkpoint_step]
    checkpoint_path = _absolute_path(checkpoint.get("path"), "checkpoint path", kind="dir")
    step = _integer(checkpoint.get("global_step"), "checkpoint global_step")
    _require(step == checkpoint_step, "selected checkpoint step drifted")
    stage_entry = context["stage_entry"]
    _require(
        step in stage_entry.get("checkpoint_steps", []),
        "checkpoint step is absent from the bound stage inventory",
    )
    expected_checkpoint_path = (
        _absolute_path(stage_entry.get("output_dir"), "bound stage output", kind="dir")
        / f"checkpoint-{step}"
    ).resolve()
    _require(
        checkpoint_path == expected_checkpoint_path,
        "checkpoint path differs from the bound stage output",
    )
    files = _mapping(checkpoint.get("files"), "checkpoint file manifest")
    _require(
        checkpoint.get("files_sha256") == contract.object_sha256(files),
        "checkpoint file-manifest hash drifted",
    )
    actual_files = _directory_manifest(checkpoint_path)
    _require(actual_files == dict(files), "checkpoint bytes differ from receipt")
    _require(
        {"adapter_config.json", "adapter_model.safetensors"} <= set(actual_files),
        "checkpoint is not a complete PEFT adapter",
    )
    return {
        "receipt": receipt,
        "path": receipt_path,
        "file_sha256": file_sha256(receipt_path),
        "receipt_sha256": receipt_sha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": step,
        "checkpoint_files_sha256": checkpoint["files_sha256"],
        "checkpoint_files": dict(files),
        "adapter_file_sha256": actual_files["adapter_model.safetensors"]["sha256"],
    }


def _validated_recorded_process_identity(
    value: Any, label: str, *, python_file_sha256: str
) -> dict[str, Any]:
    identity = _mapping(value, label)
    required = {
        "pid",
        "proc_start_ticks",
        "boot_id",
        "python_executable_file_sha256",
    }
    _require(set(identity) == required, f"{label} field inventory drifted")
    _require(
        isinstance(identity.get("pid"), int)
        and not isinstance(identity.get("pid"), bool)
        and identity["pid"] > 0,
        f"{label} pid is invalid",
    )
    _require(
        isinstance(identity.get("proc_start_ticks"), int)
        and not isinstance(identity.get("proc_start_ticks"), bool)
        and identity["proc_start_ticks"] >= 0,
        f"{label} process start ticks are invalid",
    )
    _require(
        isinstance(identity.get("boot_id"), str)
        and bool(identity["boot_id"].strip()),
        f"{label} boot ID is invalid",
    )
    _require(
        identity.get("python_executable_file_sha256") == python_file_sha256,
        f"{label} Python executable hash drifted",
    )
    return dict(identity)


def validate_bound_evaluation_receipt(
    context: Mapping[str, Any],
    evaluation_path: Path,
    *,
    split: str,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Strictly validate an immutable evaluator output used by another gate."""
    _require(split in {"g2", "mechanism", "dev"}, "unsupported reusable evaluation split")
    _require(checkpoint_step in EXPECTED_STEPS[split], "evaluation checkpoint step is ineligible")
    evidence = _mapping(
        context["binding"].get("evidence_contract"), "binding.evidence_contract"
    )
    receipt_map = _mapping(
        _mapping(
            evidence.get("checkpoint_receipts"),
            "binding.evidence_contract.checkpoint_receipts",
        ).get(context["stage_name"]),
        f"bound checkpoint evaluations for {context['stage_name']}",
    )
    expected_path = _absolute_path(
        receipt_map.get(str(checkpoint_step)),
        f"bound {split} checkpoint-{checkpoint_step} evaluation",
        kind="file",
    )
    actual_path = _absolute_path(
        str(evaluation_path), f"{split} checkpoint-{checkpoint_step} evaluation", kind="file"
    )
    _require(actual_path == expected_path, "evaluation path differs from binding")
    value = load_json(actual_path, f"{split} checkpoint-{checkpoint_step} evaluation")
    evaluation_sha = verify_self_hash(
        value,
        "evaluation_sha256",
        f"{split} checkpoint-{checkpoint_step} evaluation",
    )
    _require(
        value.get("schema_name") == EVALUATION_SCHEMA
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "pass"
        and value.get("split") == split
        and value.get("stage") == context["stage_name"],
        "evaluation schema/status/stage drifted",
    )
    producers = _mapping(evidence.get("producers"), "binding evaluator producers")
    producer_bound = _mapping(
        producers.get("preference_evaluator"), "bound preference evaluator"
    )
    producer = _mapping(value.get("producer"), "evaluation producer")
    _require(
        Path(_text(producer.get("path"), "evaluation producer path")).resolve()
        == _absolute_path(
            producer_bound.get("path"), "bound preference evaluator", kind="file"
        )
        and producer.get("file_sha256") == producer_bound.get("file_sha256")
        and file_sha256(Path(producer["path"])) == producer_bound.get("file_sha256"),
        "evaluation producer identity drifted",
    )
    binding_link = _mapping(value.get("binding"), "evaluation binding")
    _require(
        Path(_text(binding_link.get("path"), "evaluation binding path")).resolve()
        == context["binding_path"]
        and binding_link.get("file_sha256") == context["binding_file_sha256"]
        and binding_link.get("binding_sha256") == context["binding_sha256"],
        "evaluation binding identity drifted",
    )
    _require(
        value.get("cpu_contract_sha256") == context["cpu_contract"]["contract_sha256"],
        "evaluation CPU contract identity drifted",
    )
    config_link = _mapping(value.get("executable_config"), "evaluation config")
    _require(
        Path(_text(config_link.get("path"), "evaluation config path")).resolve()
        == context["config_path"]
        and config_link.get("file_sha256") == context["config_file_sha256"],
        "evaluation executable config identity drifted",
    )
    bound_correction = _mapping(
        _mapping(
            context["binding"].get("user_authorized_gpu_amendment"),
            "binding GPU amendment",
        ).get("mechanism_protocol_correction"),
        "binding mechanism protocol correction",
    )
    _require(
        value.get("mechanism_protocol_correction")
        == (dict(bound_correction) if split == "mechanism" else None),
        "evaluation mechanism correction provenance drifted",
    )
    _require(
        value.get("mechanism_correction_claim")
        == context.get("mechanism_correction_claim"),
        "evaluation mechanism correction claim drifted",
    )

    checkpoint_link = _mapping(value.get("checkpoint"), "evaluation checkpoint")
    stage_receipt_path = _absolute_path(
        checkpoint_link.get("receipt_path"), "evaluation source stage receipt", kind="file"
    )
    checkpoint = validate_checkpoint_receipt(
        stage_receipt_path,
        context,
        split=split,
        checkpoint_step=checkpoint_step,
    )
    _require(
        Path(_text(checkpoint_link.get("path"), "evaluation checkpoint path")).resolve()
        == checkpoint["checkpoint_path"]
        and checkpoint_link.get("global_step") == checkpoint_step
        and checkpoint_link.get("step") == checkpoint_step
        and checkpoint_link.get("files") == checkpoint["checkpoint_files"]
        and checkpoint_link.get("files_sha256") == checkpoint["checkpoint_files_sha256"]
        and checkpoint_link.get("adapter_file_sha256") == checkpoint["adapter_file_sha256"]
        and checkpoint_link.get("receipt_file_sha256") == checkpoint["file_sha256"]
        and checkpoint_link.get("receipt_sha256") == checkpoint["receipt_sha256"],
        "evaluation checkpoint/source receipt identity drifted",
    )

    spec = prepare_data_spec(context, split=split)
    rows, source_by_id = load_evaluation_rows(spec, split=split)
    expected_pair_ids = [row["pair_id"] for row in rows]
    data = _mapping(value.get("data"), "evaluation data")
    _require(
        Path(_text(data.get("manifest_path"), "evaluation data manifest")).resolve()
        == spec["manifest_path"]
        and data.get("manifest_file_sha256") == spec["manifest_file_sha256"]
        and data.get("manifest_sha256") == spec["manifest_sha256"]
        and Path(_text(data.get("dataset_path"), "evaluation dataset path")).resolve()
        == spec["dataset_path"]
        and data.get("dataset_file_sha256") == spec["dataset_entry"]["file_sha256"]
        and data.get("ordered_pair_ids_sha256")
        == contract.object_sha256(expected_pair_ids),
        "evaluation data identity/order drifted",
    )
    pair_results = value.get("pair_results")
    _require(
        isinstance(pair_results, list)
        and len(pair_results) == value.get("records") == len(expected_pair_ids)
        and [item.get("pair_id") for item in pair_results if isinstance(item, Mapping)]
        == expected_pair_ids
        and len(set(expected_pair_ids)) == len(expected_pair_ids),
        "evaluation pair inventory/order drifted",
    )
    beta = float(context["config"]["beta"])
    for row, raw_pair in zip(rows, pair_results):
        pair = _mapping(raw_pair, f"evaluation pair {row['pair_id']}")
        _require(
            pair.get("schema_name") == PAIR_SCHEMA
            and pair.get("schema_version") == SCHEMA_VERSION
            and pair.get("status") == "pass",
            f"{row['pair_id']}: evaluation pair schema/status drifted",
        )
        verify_self_hash(pair, "pair_evaluation_sha256", f"evaluation pair {row['pair_id']}")
        source = source_by_id[row["pair_id"]]
        family = _mapping(row.get("family_keys"), f"{row['pair_id']} family")
        expected_quality = _text(source.get("pair_status"), "source pair status") + "|" + ",".join(
            sorted(source.get("quality_flags") or [])
        )
        _require(
            pair.get("split") == row["split"]
            and pair.get("source_pair_sha256") == row["source_pair_sha256"]
            and pair.get("compiled_row_sha256") == row["row_sha256"]
            and pair.get("source_family") == family.get("source")
            and pair.get("test_family") == family.get("test")
            and pair.get("quality_status") == expected_quality
            and pair.get("generation_performed") is False
            and pair.get("sandbox_execution_performed") is False,
            f"{row['pair_id']}: evaluation pair provenance drifted",
        )
        numeric_keys = (
            "policy_chosen_logps",
            "policy_rejected_logps",
            "reference_chosen_logps",
            "reference_rejected_logps",
            "chosen_reward",
            "rejected_reward",
            "reward_margin",
        )
        numeric = {key: _finite(pair.get(key), f"{row['pair_id']}:{key}") for key in numeric_keys}
        chosen_reward = beta * (
            numeric["policy_chosen_logps"] - numeric["reference_chosen_logps"]
        )
        rejected_reward = beta * (
            numeric["policy_rejected_logps"] - numeric["reference_rejected_logps"]
        )
        margin = chosen_reward - rejected_reward
        tolerance = 1e-9 * max(1.0, abs(chosen_reward), abs(rejected_reward), abs(margin))
        _require(
            abs(numeric["chosen_reward"] - chosen_reward) <= tolerance
            and abs(numeric["rejected_reward"] - rejected_reward) <= tolerance
            and abs(numeric["reward_margin"] - margin) <= tolerance
            and pair.get("pair_accuracy") is (margin > 0),
            f"{row['pair_id']}: evaluation reward arithmetic drifted",
        )
        chosen_tokens = _integer(
            pair.get("chosen_response_tokens"), f"{row['pair_id']} chosen tokens"
        )
        rejected_tokens = _integer(
            pair.get("rejected_response_tokens"), f"{row['pair_id']} rejected tokens"
        )
        eligible = abs(chosen_tokens - rejected_tokens) <= LENGTH_MATCH_TOLERANCE
        _require(
            chosen_tokens > 0
            and rejected_tokens > 0
            and pair.get("length_matched_eligible") is eligible
            and pair.get("length_matched_prefix_tokens")
            == (min(chosen_tokens, rejected_tokens) if eligible else None),
            f"{row['pair_id']}: evaluation length contract drifted",
        )
        matched = pair.get("length_matched_reward_margin")
        _require(
            (eligible and isinstance(matched, (int, float)) and math.isfinite(float(matched)))
            or (not eligible and matched is None),
            f"{row['pair_id']}: length-matched margin drifted",
        )
        messages = row["messages"]
        _require(
            pair.get("generation_request")
            == {
                "pair_id": row["pair_id"],
                "source_pair_sha256": row["source_pair_sha256"],
                "messages": [
                    {"role": "user", "content": messages[0]["content"]}
                ],
                "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
            },
            f"{row['pair_id']}: generation request drifted",
        )
    aggregate = aggregate_results(pair_results, split=split, checkpoint_step=checkpoint_step)
    _require(value.get("aggregate") == aggregate, "evaluation aggregate drifted")

    runtime = _mapping(value.get("runtime"), "evaluation runtime")
    runtime_bound = context["runtime_parse"]
    python_bound = _mapping(runtime_bound.get("python_executable"), "bound Python")
    _require(
        runtime.get("python") == python_bound.get("version")
        and Path(_text(runtime.get("python_executable"), "evaluation Python")).resolve()
        == Path(_text(python_bound.get("path"), "bound Python path")).resolve(),
        "evaluation Python runtime drifted",
    )
    for package in ("torch", "transformers", "ms-swift", "peft", "trl", "datasets", "accelerate"):
        _require(
            runtime.get(package) == runtime_bound["package_versions"].get(package),
            f"evaluation runtime package drifted: {package}",
        )
    inventory = _mapping(runtime.get("adapter_tensor_inventory"), "evaluation adapter inventory")
    _require(
        runtime.get("device") == "cuda:0"
        and runtime.get("visible_cuda_devices") == 1
        and runtime.get("single_device_parameter_placement") is True
        and inventory.get("saved_tensors") == EXPECTED_LORA_TENSORS
        and inventory.get("loaded_tensors") == EXPECTED_LORA_TENSORS
        and inventory.get("saved_params") == EXPECTED_LORA_PARAMS
        and inventory.get("loaded_params") == EXPECTED_LORA_PARAMS
        and inventory.get("key_inventory_exact") is True
        and inventory.get("shape_inventory_exact") is True
        and inventory.get("tensor_content_exact_after_saved_dtype_cast") is True
        and _sha256(
            inventory.get("tensor_evidence_sha256"),
            "evaluation adapter tensor evidence hash",
        )
        == inventory.get("tensor_evidence_sha256"),
        "evaluation adapter reload inventory drifted",
    )
    evaluator_process = _validated_recorded_process_identity(
        value.get("process_identity"),
        "evaluation process identity",
        python_file_sha256=python_bound["file_sha256"],
    )
    runner_process = _validated_recorded_process_identity(
        checkpoint["receipt"].get("process_identity"),
        "evaluation source trainer process identity",
        python_file_sha256=python_bound["file_sha256"],
    )
    evaluator_key = (
        evaluator_process["boot_id"],
        evaluator_process["pid"],
        evaluator_process["proc_start_ticks"],
    )
    runner_key = (
        runner_process["boot_id"],
        runner_process["pid"],
        runner_process["proc_start_ticks"],
    )
    _require(evaluator_key != runner_key, "evaluation did not use a fresh process")
    parity = _mapping(value.get("reload_parity"), "evaluation reload parity")
    _require(
        value.get("fresh_process_reload_proven") is True
        and value.get("adapter_reload_proven") is True
        and parity.get("fresh_process_reload_proven") is True
        and parity.get("runner_process_distinct") is True
        and parity.get("runner_process_identity") == runner_process
        and parity.get("evaluator_process_identity") == evaluator_process
        and parity.get("required") is (split in {"g2", "mechanism"}),
        "evaluation fresh-process reload proof drifted",
    )
    if split in {"g2", "mechanism"}:
        comparisons = parity.get("comparisons")
        _require(
            isinstance(comparisons, list)
            and [item.get("pair_id") for item in comparisons if isinstance(item, Mapping)]
            == expected_pair_ids,
            "evaluation reload comparison inventory drifted",
        )
        post = _mapping(
            checkpoint["receipt"].get("observations"),
            "source stage observations",
        ).get("probes", {}).get("post_train")
        _require(isinstance(post, list), "source stage post-train probes are absent")
        probes_by_id = {
            item.get("pair_id"): item for item in post if isinstance(item, Mapping)
        }
        results_by_id = {item["pair_id"]: item for item in pair_results}
        key_map = {
            "policy_chosen_logp": "policy_chosen_logps",
            "policy_rejected_logp": "policy_rejected_logps",
            "reference_chosen_logp": "reference_chosen_logps",
            "reference_rejected_logp": "reference_rejected_logps",
        }
        expected_comparisons: list[dict[str, Any]] = []
        for pair_id in expected_pair_ids:
            probe = _mapping(probes_by_id.get(pair_id), f"{pair_id} source probe")
            result = results_by_id[pair_id]
            differences: dict[str, float] = {}
            tolerances: dict[str, float] = {}
            for probe_key, result_key in key_map.items():
                expected_logp = _finite(probe.get(probe_key), f"{pair_id} source logp")
                actual_logp = _finite(result.get(result_key), f"{pair_id} reload logp")
                differences[probe_key] = abs(actual_logp - expected_logp)
                tolerances[probe_key] = _reload_logp_tolerance(
                    actual_logp, expected_logp
                )
            runner_margin = _finite(
                probe.get("chosen_reward_margin"), f"{pair_id} source margin"
            )
            reload_margin = _finite(result.get("reward_margin"), f"{pair_id} reload margin")
            expected_comparisons.append(
                {
                    "pair_id": pair_id,
                    "absolute_differences": differences,
                    "tolerances": tolerances,
                    "reward_margin": {
                        "runner": runner_margin,
                        "reload": reload_margin,
                        "absolute_difference": abs(reload_margin - runner_margin),
                        "tolerance": RELOAD_REWARD_MARGIN_TOLERANCE,
                    },
                }
            )
        _require(
            comparisons == expected_comparisons
            and all(
                all(
                    comparison["absolute_differences"][key]
                    <= comparison["tolerances"][key]
                    for key in key_map
                )
                and comparison["reward_margin"]["absolute_difference"]
                <= RELOAD_REWARD_MARGIN_TOLERANCE
                for comparison in expected_comparisons
            ),
            "evaluation reload comparison values/tolerances drifted",
        )
    reference = _mapping(value.get("reference_topology"), "evaluation reference topology")
    _require(
        reference
        == {
            "policy": "selected_checkpoint_PEFT_adapter",
            "reference": "same_frozen_parent_inside_PeftModel.disable_adapter",
            "explicit_reference_model_loaded": False,
            "beta": beta,
            "response_logprob_reduction": "sum_over_response_only_tokens",
        }
        and value.get("generation_performed") is False
        and value.get("sandbox_execution_performed") is False
        and value.get("heldout_claim") is None,
        "evaluation reference/claim boundary drifted",
    )
    return {
        "value": value,
        "evaluation_sha256": evaluation_sha,
        "file_sha256": file_sha256(actual_path),
        "path": actual_path,
        "checkpoint": checkpoint,
        "aggregate": aggregate,
    }


def _load_bound_dev_evaluations(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    selector = _mapping(context["binding"].get("selector_contract"), "binding.selector_contract")
    candidates = selector.get("eligible_checkpoints")
    _require(
        isinstance(candidates, list)
        and [item.get("step") for item in candidates if isinstance(item, Mapping)] == [15, 30],
        "bound dev candidates drifted",
    )
    loaded: list[dict[str, Any]] = []
    for candidate_value in candidates:
        candidate = _mapping(candidate_value, "bound dev candidate")
        step = _integer(candidate.get("step"), "bound dev candidate step")
        path = _absolute_path(
            candidate.get("dev_evaluation_path"),
            f"bound dev checkpoint-{step} evaluation",
            kind="file",
        )
        _require(
            _absolute_path(
                candidate.get("checkpoint_receipt_path"),
                f"bound dev checkpoint-{step} reload receipt",
                kind="file",
            )
            == path,
            f"dev checkpoint-{step} evaluation/reload receipt paths diverged",
        )
        _require(not path.is_symlink(), "dev evaluation cannot be a symlink")
        validated = validate_bound_evaluation_receipt(
            context, path, split="dev", checkpoint_step=step
        )
        value = validated["value"]
        checkpoint = _mapping(value.get("checkpoint"), "dev evaluation checkpoint")
        checkpoint_path = validated["checkpoint"]["checkpoint_path"]
        _require(
            checkpoint_path
            == _absolute_path(
                candidate.get("checkpoint_path"),
                "bound candidate checkpoint",
                kind="dir",
            ),
            f"dev checkpoint-{step} identity drifted",
        )
        adapter_path = _absolute_path(
            candidate.get("adapter_path"), "bound candidate adapter", kind="file"
        )
        _require(
            adapter_path == checkpoint_path / "adapter_model.safetensors"
            and file_sha256(adapter_path) == checkpoint.get("adapter_file_sha256"),
            f"dev checkpoint-{step} adapter identity drifted",
        )
        metrics = _mapping(
            validated["aggregate"].get("selector_metrics"), "dev selector metrics"
        )
        loaded.append(
            {
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "adapter_file_sha256": checkpoint["adapter_file_sha256"],
                "stage_receipt_path": checkpoint["receipt_path"],
                "stage_receipt_sha256": checkpoint["receipt_sha256"],
                "dev_evaluation_path": str(path),
                "dev_evaluation_file_sha256": validated["file_sha256"],
                "dev_eval_sha256": validated["evaluation_sha256"],
                "checkpoint_receipt_sha256": validated["evaluation_sha256"],
                "metrics": dict(metrics),
            }
        )
    return loaded


def _select_dev_candidate(eligible: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    _require(len(eligible) == 2, "selection requires exactly two dev candidates")
    return max(
        eligible,
        key=lambda item: (
            _finite(item["metrics"]["dev_pair_accuracy"], "dev pair accuracy"),
            _finite(item["metrics"]["dev_mean_reward_margin"], "dev mean margin"),
            _finite(item["metrics"]["dev_length_matched_margin"], "dev length-matched margin"),
            -_integer(item["step"], "dev checkpoint step"),
        ),
    )


def run_dev_selection(binding_path: Path, output_path: Path) -> dict[str, Any]:
    context = validate_binding(binding_path, split="dev")
    selector = _mapping(context["binding"].get("selector_contract"), "binding.selector_contract")
    bound_output = _absolute_path(
        _mapping(selector.get("selection_receipt"), "bound selection receipt").get("path"),
        "bound selection receipt path",
    )
    output = _absolute_path(output_path, "selection output")
    _require(output == bound_output, "selection output path differs from binding")
    _require(output.parent.is_dir() and not output.exists(), "selection output is not fresh")
    eligible = _load_bound_dev_evaluations(context)
    winner = dict(_select_dev_candidate(eligible))
    selection_rule = _mapping(selector.get("selection_rule"), "bound selection rule")
    _require(
        selector.get("selection_rule_sha256") == contract.object_sha256(selection_rule),
        "bound selection rule self-hash drifted",
    )
    payload = {
        "schema_name": SELECTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "selected",
        "created_at_utc": _utc_now(),
        "binding_sha256": context["binding_sha256"],
        "cpu_contract_sha256": context["cpu_contract"]["contract_sha256"],
        "selection_rule": dict(selection_rule),
        "eligible": eligible,
        "selected": winner,
        "heldout_opened": False,
    }
    sealed, file_sha = write_exclusive_json(
        output, payload, self_field="selection_sha256", label="dev checkpoint selection"
    )
    return {
        "status": "selected",
        "output": str(output),
        "file_sha256": file_sha,
        "selection_sha256": sealed["selection_sha256"],
        "selected_step": winner["step"],
    }


def validate_selection_receipt(
    selection_path: Path,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    _require(selection_path.is_absolute(), "checkpoint selection receipt path must be absolute")
    _require(not selection_path.is_symlink(), "checkpoint selection receipt cannot be a symlink")
    selection_path = selection_path.resolve()
    selector = _mapping(context["binding"].get("selector_contract"), "binding.selector_contract")
    bound_selection = _mapping(selector.get("selection_receipt"), "bound selection receipt")
    _require(
        _absolute_path(bound_selection.get("path"), "bound selection receipt path")
        == selection_path,
        "selection receipt path differs from binding",
    )
    selection = load_json(selection_path, "checkpoint selection receipt")
    selection_sha = verify_self_hash(
        selection, "selection_sha256", "checkpoint selection receipt"
    )
    _require(selection.get("schema_name") == SELECTION_SCHEMA, "selection schema drifted")
    _require(selection.get("status") == "selected", "checkpoint selection is not final")
    _require(
        selection.get("binding_sha256") == context["binding_sha256"],
        "selection binding hash drifted",
    )
    _require(
        selection.get("cpu_contract_sha256")
        == context["cpu_contract"]["contract_sha256"],
        "selection CPU contract hash drifted",
    )
    _require(
        selection.get("selection_rule") == selector.get("selection_rule"),
        "selection rule differs from binding",
    )
    expected_eligible = _load_bound_dev_evaluations(context)
    _require(
        selection.get("eligible") == expected_eligible,
        "selection eligible evidence differs from the two bound dev evaluations",
    )
    expected_selected = dict(_select_dev_candidate(expected_eligible))
    selected = _mapping(selection.get("selected"), "selected checkpoint")
    _require(dict(selected) == expected_selected, "selection winner was not recomputed correctly")
    _require(
        _integer(selected.get("step"), "selected checkpoint step")
        == checkpoint["checkpoint_step"],
        "heldout checkpoint is not the selected step",
    )
    _require(
        Path(_text(selected.get("checkpoint_path"), "selected checkpoint path")).resolve()
        == checkpoint["checkpoint_path"],
        "heldout checkpoint path differs from selection",
    )
    _require(
        selected.get("stage_receipt_sha256") == checkpoint["receipt_sha256"],
        "heldout training-stage receipt differs from selection",
    )
    _require(
        selected.get("adapter_file_sha256") == checkpoint["adapter_file_sha256"],
        "heldout adapter differs from selected dev candidate",
    )
    return {
        "selection": selection,
        "path": selection_path,
        "file_sha256": file_sha256(selection_path),
        "selection_sha256": selection_sha,
    }


def prepare_data_spec(context: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    cpu_contract = context["cpu_contract"]
    input_entry = _mapping(
        cpu_contract.get("inputs", {}).get("data_manifest"),
        "CPU contract data manifest",
    )
    manifest_path = (context["bootcamp_root"] / _text(input_entry.get("path"), "data manifest path")).resolve()
    _require(
        manifest_path.is_relative_to(context["bootcamp_root"]),
        "data manifest escapes bootcamp root",
    )
    actual_manifest_file_sha = file_sha256(manifest_path)
    _require(
        actual_manifest_file_sha == input_entry.get("file_sha256"),
        "data manifest file hash drifted",
    )
    manifest = load_json(manifest_path, "data manifest")
    manifest_sha = verify_self_hash(manifest, "manifest_sha256", "data manifest")
    _require(
        manifest_sha == input_entry.get("content_sha256"),
        "data manifest content hash drifted",
    )
    source_entry = _mapping(manifest.get("inputs", {}).get("day22_pairs"), "source pairs entry")
    source_path = (context["bootcamp_root"] / _text(source_entry.get("path"), "source pairs path")).resolve()
    _require(source_path.is_relative_to(context["bootcamp_root"]), "source pairs escape bootcamp root")
    output_split = "train" if split in {"g2", "mechanism"} else split
    dataset_entry = _mapping(manifest.get("outputs", {}).get(output_split), f"{output_split} dataset entry")
    dataset_path = (context["bootcamp_root"] / _text(dataset_entry.get("path"), "dataset path")).resolve()
    _require(dataset_path.is_relative_to(context["bootcamp_root"]), "dataset escapes bootcamp root")
    config = context["config"]
    if split in {"g2", "mechanism"}:
        expected_syntax = f"{dataset_path}#{8 if split == 'g2' else 4}"
        _require(config.get("dataset") == [expected_syntax], f"{split} config dataset#4 drifted")
        _require("val_dataset" not in config, f"{split} config must not contain dev")
    elif split == "dev":
        _require(config.get("val_dataset") == [str(dataset_path)], "smoke config dev path drifted")
    else:
        _require("heldout" not in canonical_json(config).decode("utf-8"), "heldout leaked into trainer config")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_file_sha256": actual_manifest_file_sha,
        "manifest_sha256": manifest_sha,
        "source_path": source_path,
        "source_entry": source_entry,
        "dataset_path": dataset_path,
        "dataset_entry": dataset_entry,
        "output_split": output_split,
    }


def load_evaluation_rows(
    spec: Mapping[str, Any], *, split: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source_path = spec["source_path"]
    _require(file_sha256(source_path) == spec["source_entry"].get("file_sha256"), "source pair file hash drifted")
    source_rows = load_jsonl(source_path, "source preference pairs")
    source_by_id = {str(row.get("pair_id")): row for row in source_rows}
    _require(len(source_by_id) == len(source_rows) == 200, "source pair inventory drifted")
    dataset_path = spec["dataset_path"]
    _require(file_sha256(dataset_path) == spec["dataset_entry"].get("file_sha256"), "dataset file hash drifted")
    rows = load_jsonl(dataset_path, f"{split} dataset")
    _require(
        contract.object_sha256([row.get("pair_id") for row in rows])
        == spec["dataset_entry"].get("ordered_pair_ids_sha256"),
        f"{split} ordered pair IDs drifted",
    )
    _require(
        contract.object_sha256([row.get("row_sha256") for row in rows])
        == spec["dataset_entry"].get("ordered_row_hashes_sha256"),
        f"{split} ordered compiled row hashes drifted",
    )
    if split in {"g2", "mechanism"}:
        rows = rows[: (1 if split == "g2" else 4)]
        _require(
            tuple(row.get("pair_id") for row in rows)
            == MECHANISM_PAIR_IDS[: len(rows)],
            f"{split} pair order drifted",
        )
    else:
        _require(len(rows) == spec["dataset_entry"].get("records"), f"{split} row count drifted")
    for row in rows:
        pair_id = _text(row.get("pair_id"), "compiled pair_id")
        _require(row.get("split") == spec["output_split"], f"{pair_id}: split drifted")
        _require(pair_id in source_by_id, f"unknown compiled pair: {pair_id}")
        try:
            contract.validate_compiled_row(row, source_pair=source_by_id[pair_id])
        except contract.Day23ContractError as error:
            raise Day23PreferenceEvaluationError(f"{pair_id}: compiled row failed: {error}") from error
    return rows, source_by_id


def load_policy(
    context: Mapping[str, Any], checkpoint: Mapping[str, Any], *, device_name: str
) -> dict[str, Any]:
    try:
        import torch
        import swift
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors import safe_open
        from swift import get_model_processor, get_template
    except ImportError as error:
        raise Day23PreferenceEvaluationError("pinned GPU evaluation runtime is incomplete") from error
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "CUDA_VISIBLE_DEVICES must be fixed to exactly `0`",
    )
    _require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(device_name)
    _require(
        device.type == "cuda" and device.index == 0,
        "device must be the single visible cuda:0",
    )
    _require(torch.cuda.device_count() == 1, "evaluator must expose exactly one GPU")
    _require(0 <= device.index < torch.cuda.device_count(), "requested CUDA device is unavailable")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "selected GPU does not support BF16")
    imported_swift = Path(swift.__file__).resolve().parent
    expected_swift = Path(context["ms_swift"]["path"]).resolve() / "swift"
    _require(imported_swift == expected_swift, "imported ms-swift differs from binding")
    adapter_config_path = checkpoint["checkpoint_path"] / "adapter_config.json"
    adapter_weights_path = checkpoint["checkpoint_path"] / "adapter_model.safetensors"
    adapter_config = load_json(adapter_config_path, "PEFT adapter config")
    config = context["config"]
    _require(adapter_config.get("peft_type") == "LORA", "checkpoint is not LoRA")
    _require(adapter_config.get("task_type") == "CAUSAL_LM", "checkpoint task type drifted")
    _require(adapter_config.get("r") == config.get("lora_rank"), "checkpoint LoRA rank drifted")
    _require(adapter_config.get("lora_alpha") == config.get("lora_alpha"), "checkpoint LoRA alpha drifted")
    _require(
        float(adapter_config.get("lora_dropout")) == float(config.get("lora_dropout")),
        "checkpoint LoRA dropout drifted",
    )
    _require(adapter_config.get("target_modules") == config.get("target_regex"), "checkpoint target regex drifted")
    _require(not adapter_config.get("modules_to_save"), "checkpoint unexpectedly saves non-LoRA modules")
    saved_state: dict[str, Any] = {}
    saved_numel = 0
    with safe_open(str(adapter_weights_path), framework="pt", device="cpu") as handle:
        saved_keys = sorted(handle.keys())
        for key in saved_keys:
            _require(
                ".lora_A." in key or ".lora_B." in key,
                f"adapter checkpoint contains a non-LoRA tensor: {key}",
            )
            tensor = handle.get_tensor(key).detach().cpu().contiguous()
            saved_state[key] = tensor
            saved_numel += tensor.numel()
    _require(
        len(saved_keys) == EXPECTED_LORA_TENSORS
        and saved_numel == EXPECTED_LORA_PARAMS,
        "adapter checkpoint tensor inventory drifted",
    )
    dpo_template.register_dpo_template()
    model, processor = get_model_processor(
        str(context["model_path"]),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map={"": device.index},
        load_model=True,
        use_hf=True,
        download_model=False,
    )
    _require(model is not None, "get_model_processor did not load model weights")
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    template = get_template(
        processor,
        template_type=TEMPLATE_ALIAS,
        max_length=config["max_length"],
        truncation_strategy="raise",
        padding_free=False,
        loss_scale=config["loss_scale"],
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    template.set_mode("rlhf")
    dpo_template._assert_template_contract(template, max_length=config["max_length"])
    policy = PeftModel.from_pretrained(
        model,
        str(checkpoint["checkpoint_path"]),
        adapter_name="policy",
        is_trainable=False,
    )
    _require(set(policy.peft_config) == {"policy"}, "policy must contain exactly one adapter")
    policy.set_adapter("policy")
    policy.requires_grad_(False)
    policy.eval()
    active = policy.active_adapters
    if isinstance(active, str):
        active = [active]
    _require(list(active) == ["policy"], "policy adapter is not active")
    loaded_state = get_peft_model_state_dict(policy, adapter_name="policy")
    _require(
        set(loaded_state) == set(saved_state),
        "loaded PEFT adapter key inventory differs from the saved checkpoint",
    )
    loaded_numel = 0
    tensor_evidence: dict[str, dict[str, Any]] = {}
    for key in saved_keys:
        saved_tensor = saved_state[key]
        loaded_tensor = loaded_state[key].detach().to(
            device="cpu", dtype=saved_tensor.dtype
        ).contiguous()
        _require(
            tuple(loaded_tensor.shape) == tuple(saved_tensor.shape),
            f"loaded PEFT adapter shape drifted: {key}",
        )
        _require(
            torch.equal(loaded_tensor, saved_tensor),
            f"loaded PEFT adapter tensor content drifted: {key}",
        )
        loaded_numel += loaded_tensor.numel()
        tensor_bytes = loaded_tensor.view(torch.uint8).numpy().tobytes()
        tensor_evidence[key] = {
            "dtype": str(saved_tensor.dtype),
            "shape": list(saved_tensor.shape),
            "sha256": hashlib.sha256(tensor_bytes).hexdigest(),
        }
    _require(
        len(loaded_state) == EXPECTED_LORA_TENSORS
        and loaded_numel == EXPECTED_LORA_PARAMS,
        "loaded PEFT adapter tensor inventory drifted",
    )
    parameter_devices = {str(parameter.device) for parameter in policy.parameters()}
    buffer_devices = {str(buffer.device) for buffer in policy.buffers() if buffer.numel()}
    _require(parameter_devices == {str(device)}, f"policy spans unexpected devices: {parameter_devices}")
    _require(buffer_devices <= {str(device)}, f"policy buffers span unexpected devices: {buffer_devices}")
    properties = torch.cuda.get_device_properties(device)
    runtime = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "ms-swift": _package_version("ms-swift"),
        "peft": _package_version("peft"),
        "trl": _package_version("trl"),
        "datasets": _package_version("datasets"),
        "accelerate": _package_version("accelerate"),
        "device": str(device),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "visible_cuda_devices": torch.cuda.device_count(),
        "single_device_parameter_placement": True,
        "dtype": "bfloat16",
        "adapter_tensor_inventory": {
            "saved_tensors": len(saved_keys),
            "saved_params": saved_numel,
            "loaded_tensors": len(loaded_state),
            "loaded_params": loaded_numel,
            "key_inventory_exact": True,
            "shape_inventory_exact": True,
            "tensor_content_exact_after_saved_dtype_cast": True,
            "tensor_evidence_sha256": contract.object_sha256(tensor_evidence),
        },
    }
    expected_versions = _mapping(
        context["runtime_parse"].get("package_versions"),
        "binding runtime package versions",
    )
    for name in ("torch", "transformers", "ms-swift", "peft", "trl", "datasets", "accelerate"):
        _require(
            runtime[name] == expected_versions.get(name),
            f"evaluation runtime package drifted: {name}",
        )
    python_entry = _mapping(
        context["runtime_parse"].get("python_executable"),
        "binding runtime Python",
    )
    _require(
        runtime["python"] == python_entry.get("version"),
        "evaluation Python version drifted",
    )
    return {"torch": torch, "policy": policy, "template": template, "device": device, "runtime": runtime}


def _branch_logps(
    runtime: Mapping[str, Any], *, input_ids: Sequence[int], labels: Sequence[int], label: str
) -> tuple[float, list[float]]:
    torch = runtime["torch"]
    try:
        from trl.trainer.utils import selective_log_softmax
    except ImportError as error:
        raise Day23PreferenceEvaluationError("TRL selective_log_softmax is unavailable") from error
    _require(bool(input_ids) and len(input_ids) == len(labels), f"{label} token arrays are misaligned")
    _require(len(input_ids) <= REQUIRED_CONFIG["max_length"], f"{label} exceeds max_length")
    device = runtime["device"]
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    target = torch.tensor([list(labels)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    with torch.inference_mode():
        outputs = runtime["policy"](
            input_ids=ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        _require(logits.ndim == 3 and logits.shape[0] == 1, f"{label} logits shape drifted")
        if logits.shape[1] != target.shape[1]:
            logits = logits[:, -target.shape[1] :]
        _require(logits.shape[:2] == target.shape, f"{label} logits/labels length drifted")
        shifted = torch.roll(target, shifts=-1, dims=1)
        mask = shifted != -100
        safe_labels = shifted.clone()
        safe_labels[~mask] = 0
        per_token = selective_log_softmax(logits, safe_labels)
        selected = per_token[mask].float().cpu()
    values = [float(value) for value in selected.tolist()]
    _require(bool(values), f"{label} contains no response labels")
    _require(all(math.isfinite(value) for value in values), f"{label} log-probs are non-finite")
    total = math.fsum(values)
    _require(math.isfinite(total), f"{label} sequence log-prob is non-finite")
    return total, values


def evaluate_pair(
    runtime: Mapping[str, Any],
    row: Mapping[str, Any],
    source_pair: Mapping[str, Any],
    *,
    beta: float,
) -> dict[str, Any]:
    pair_id = _text(row.get("pair_id"), "pair_id")
    encoded = dpo_template.encode_dpo_row(runtime["template"], row)
    for branch in ("chosen", "rejected"):
        _require(
            len(encoded[f"{branch}_input_ids"]) == len(encoded[f"{branch}_labels"]),
            f"{pair_id}:{branch} encoding is misaligned",
        )
    runtime["policy"].set_adapter("policy")
    policy_chosen, policy_chosen_tokens = _branch_logps(
        runtime,
        input_ids=encoded["chosen_input_ids"],
        labels=encoded["chosen_labels"],
        label=f"{pair_id}:policy:chosen",
    )
    policy_rejected, policy_rejected_tokens = _branch_logps(
        runtime,
        input_ids=encoded["rejected_input_ids"],
        labels=encoded["rejected_labels"],
        label=f"{pair_id}:policy:rejected",
    )
    with runtime["policy"].disable_adapter():
        reference_chosen, reference_chosen_tokens = _branch_logps(
            runtime,
            input_ids=encoded["chosen_input_ids"],
            labels=encoded["chosen_labels"],
            label=f"{pair_id}:reference:chosen",
        )
        reference_rejected, reference_rejected_tokens = _branch_logps(
            runtime,
            input_ids=encoded["rejected_input_ids"],
            labels=encoded["rejected_labels"],
            label=f"{pair_id}:reference:rejected",
        )
    active = runtime["policy"].active_adapters
    if isinstance(active, str):
        active = [active]
    _require(list(active) == ["policy"], f"{pair_id}: policy adapter was not restored")
    chosen_count = len(policy_chosen_tokens)
    rejected_count = len(policy_rejected_tokens)
    _require(chosen_count == len(reference_chosen_tokens), f"{pair_id}: chosen response count drifted")
    _require(rejected_count == len(reference_rejected_tokens), f"{pair_id}: rejected response count drifted")
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    reward_margin = chosen_reward - rejected_reward
    eligible = abs(chosen_count - rejected_count) <= LENGTH_MATCH_TOLERANCE
    prefix_tokens = min(chosen_count, rejected_count)
    matched_margin: float | None = None
    if eligible:
        matched_margin = beta * (
            (math.fsum(policy_chosen_tokens[:prefix_tokens]) - math.fsum(policy_rejected_tokens[:prefix_tokens]))
            - (
                math.fsum(reference_chosen_tokens[:prefix_tokens])
                - math.fsum(reference_rejected_tokens[:prefix_tokens])
            )
        )
        _finite(matched_margin, f"{pair_id}: length-matched margin")
    family_keys = _mapping(row.get("family_keys"), f"{pair_id}.family_keys")
    quality_flags = source_pair.get("quality_flags")
    _require(
        isinstance(quality_flags, list)
        and all(isinstance(flag, str) and bool(flag) for flag in quality_flags),
        f"{pair_id}: quality flags are invalid",
    )
    pair_status = _text(source_pair.get("pair_status"), f"{pair_id}.pair_status")
    messages = row.get("messages")
    _require(
        isinstance(messages, list)
        and len(messages) == 2
        and messages[0].get("role") == "user"
        and messages[1].get("role") == "assistant",
        f"{pair_id}: messages are not one user/assistant pair",
    )
    generation_request = {
        "pair_id": pair_id,
        "source_pair_sha256": row["source_pair_sha256"],
        "messages": [{"role": "user", "content": _text(messages[0].get("content"), f"{pair_id}.user")}],
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    }
    result: dict[str, Any] = {
        "schema_name": PAIR_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "pair_id": pair_id,
        "split": row["split"],
        "source_pair_sha256": row["source_pair_sha256"],
        "compiled_row_sha256": row["row_sha256"],
        "policy_chosen_logps": _finite(policy_chosen, f"{pair_id}: policy chosen"),
        "policy_rejected_logps": _finite(policy_rejected, f"{pair_id}: policy rejected"),
        "reference_chosen_logps": _finite(reference_chosen, f"{pair_id}: reference chosen"),
        "reference_rejected_logps": _finite(reference_rejected, f"{pair_id}: reference rejected"),
        "chosen_reward": _finite(chosen_reward, f"{pair_id}: chosen reward"),
        "rejected_reward": _finite(rejected_reward, f"{pair_id}: rejected reward"),
        "reward_margin": _finite(reward_margin, f"{pair_id}: reward margin"),
        "pair_accuracy": reward_margin > 0,
        "chosen_response_tokens": chosen_count,
        "rejected_response_tokens": rejected_count,
        "length_matched_eligible": eligible,
        "length_matched_prefix_tokens": prefix_tokens if eligible else None,
        "length_matched_reward_margin": matched_margin,
        "length_bucket": _length_bucket(chosen_count, rejected_count),
        "source_family": _text(family_keys.get("source"), f"{pair_id}.source family"),
        "test_family": _text(family_keys.get("test"), f"{pair_id}.test family"),
        "quality_status": pair_status + "|" + ",".join(sorted(quality_flags)),
        "generation_request": generation_request,
        "generation_performed": False,
        "sandbox_execution_performed": False,
    }
    result["pair_evaluation_sha256"] = object_sha256(result)
    return result


def evaluate_rows(
    runtime: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_by_id: Mapping[str, Mapping[str, Any]],
    *,
    beta: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        results.append(evaluate_pair(runtime, row, source_by_id[pair_id], beta=beta))
    _require(len(results) == len(rows), "pair evaluation count drifted")
    return results


def validate_distinct_process(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    receipt = _mapping(checkpoint.get("receipt"), "stage receipt")
    runner_process = _mapping(receipt.get("process_identity"), "stage runner process identity")
    evaluator_process = _process_identity()
    _require(
        (runner_process.get("boot_id"), runner_process.get("pid"), runner_process.get("proc_start_ticks"))
        != (
            evaluator_process.get("boot_id"),
            evaluator_process.get("pid"),
            evaluator_process.get("proc_start_ticks"),
        ),
        "checkpoint reload must run in a process distinct from the trainer",
    )
    return runner_process, evaluator_process


def validate_reload_parity(
    checkpoint: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    """Cross-check fresh-process outputs against the runner's in-process probe."""
    runner_process, evaluator_process = validate_distinct_process(checkpoint)
    receipt = _mapping(checkpoint.get("receipt"), "stage receipt")
    if split not in {"g2", "mechanism"}:
        return {
            "required": False,
            "fresh_process_reload_proven": True,
            "proof": "checkpoint adapter bytes and finite pair outputs",
            "runner_process_distinct": True,
            "runner_process_identity": dict(runner_process),
            "evaluator_process_identity": evaluator_process,
        }
    observations = _mapping(receipt.get("observations"), "stage receipt observations")
    probes = _mapping(observations.get("probes"), "stage receipt probes")
    post = probes.get("post_train")
    _require(isinstance(post, list) and post, "stage receipt has no post-train probes")
    by_pair = {
        _text(item.get("pair_id"), "post-train probe pair_id"): item
        for item in post
        if isinstance(item, Mapping)
    }
    keys = (
        "policy_chosen_logp",
        "policy_rejected_logp",
        "reference_chosen_logp",
        "reference_rejected_logp",
    )
    result_keys = {
        "policy_chosen_logp": "policy_chosen_logps",
        "policy_rejected_logp": "policy_rejected_logps",
        "reference_chosen_logp": "reference_chosen_logps",
        "reference_rejected_logp": "reference_rejected_logps",
    }
    comparisons: list[dict[str, Any]] = []
    for result in results:
        pair_id = _text(result.get("pair_id"), "fresh reload pair_id")
        _require(pair_id in by_pair, f"fresh reload pair is absent from runner probes: {pair_id}")
        probe = by_pair[pair_id]
        differences: dict[str, float] = {}
        tolerances: dict[str, float] = {}
        for key in keys:
            expected = _finite(probe.get(key), f"{pair_id}:{key}:runner")
            actual = _finite(result.get(result_keys[key]), f"{pair_id}:{key}:reload")
            difference = abs(actual - expected)
            tolerance = _reload_logp_tolerance(actual, expected)
            _require(
                difference <= tolerance,
                f"fresh-process reload output drifted for {pair_id}:{key}: "
                f"{difference} > {tolerance}",
            )
            differences[key] = difference
            tolerances[key] = tolerance
        runner_margin = _finite(
            probe.get("chosen_reward_margin"), f"{pair_id}:runner reward margin"
        )
        reload_margin = _finite(
            result.get("reward_margin"), f"{pair_id}:reload reward margin"
        )
        margin_difference = abs(reload_margin - runner_margin)
        _require(
            margin_difference <= RELOAD_REWARD_MARGIN_TOLERANCE,
            f"fresh-process reward margin drifted for {pair_id}: "
            f"{margin_difference} > {RELOAD_REWARD_MARGIN_TOLERANCE}",
        )
        comparisons.append(
            {
                "pair_id": pair_id,
                "absolute_differences": differences,
                "tolerances": tolerances,
                "reward_margin": {
                    "runner": runner_margin,
                    "reload": reload_margin,
                    "absolute_difference": margin_difference,
                    "tolerance": RELOAD_REWARD_MARGIN_TOLERANCE,
                },
            }
        )
    _require(len(comparisons) == len(post), "fresh reload probe inventory drifted")
    return {
        "required": True,
        "fresh_process_reload_proven": True,
        "runner_process_distinct": True,
        "runner_process_identity": dict(runner_process),
        "evaluator_process_identity": evaluator_process,
        "comparisons": comparisons,
    }


def validate_output_contract(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Path]:
    _require(args.output is not None, "--output is required")
    output = _absolute_path(args.output, "evaluation output")
    _require(output.parent.is_dir(), f"evaluation output parent is missing: {output.parent}")
    _require(not output.exists() and not output.is_symlink(), f"evaluation output already exists: {output}")
    heldout_args = {
        "selection": args.selection_receipt,
        "claim": args.heldout_claim,
        "evaluation": args.heldout_evaluation,
        "consumption": args.heldout_consumption,
    }
    if args.split != "heldout":
        _require(
            all(value is None for value in heldout_args.values()),
            "heldout/selection arguments are forbidden outside heldout",
        )
        evidence_root = _absolute_path(
            context["binding"]["paths"]["evidence_root"],
            "binding evidence root",
            kind="dir",
        )
        _require(output.is_relative_to(evidence_root), "evaluation output must remain under evidence_root")
        if args.split in {"g2", "mechanism"}:
            receipt_map = _mapping(
                _mapping(
                    context["binding"].get("evidence_contract"),
                    "binding.evidence_contract",
                ).get("checkpoint_receipts"),
                "binding.evidence_contract.checkpoint_receipts",
            )
            stage_map = _mapping(
                receipt_map.get(context["stage_name"]),
                f"bound reload receipts for {context['stage_name']}",
            )
            bound_output = _absolute_path(
                stage_map.get(str(checkpoint["checkpoint_step"])),
                f"bound {args.split} reload receipt",
            )
            _require(
                output == bound_output,
                f"{args.split} output path differs from the bound reload receipt",
            )
        elif args.split == "dev":
            candidates = context["binding"]["selector_contract"].get("eligible_checkpoints")
            _require(isinstance(candidates, list), "bound checkpoint candidates are missing")
            matches = [
                item
                for item in candidates
                if isinstance(item, Mapping)
                and item.get("step") == checkpoint["checkpoint_step"]
            ]
            _require(len(matches) == 1, "dev checkpoint is not a unique bound candidate")
            _require(
                output
                == _absolute_path(matches[0].get("dev_evaluation_path"), "bound dev output"),
                "dev output path differs from the bound checkpoint path",
            )
        return {"output": output}

    _require(all(value is not None for value in heldout_args.values()), "heldout requires selection and all ledger paths")
    ledger = _mapping(
        context["binding"].get("heldout_ledger_contract"),
        "binding.heldout_ledger_contract",
    )
    _require(ledger.get("one_shot") is True and ledger.get("winner_only") is True, "heldout is not bound one-shot/winner-only")
    paths: dict[str, Path] = {"output": output}
    for key, binding_key, expected_hash_field in (
        ("claim", "claim", "claim_sha256"),
        ("evaluation", "evaluation", "evaluation_sha256"),
        ("consumption", "final_receipt", "consumption_sha256"),
    ):
        entry = _mapping(ledger.get(binding_key), f"heldout ledger {binding_key}")
        _require(entry.get("self_hash_field") == expected_hash_field, f"heldout {key} self-hash field drifted")
        supplied = heldout_args[key]
        bound = _absolute_path(entry.get("path"), f"bound heldout {key}")
        actual = _absolute_path(supplied, f"heldout {key}")
        _require(actual == bound, f"heldout {key} path differs from binding")
        _require(actual.parent.is_dir(), f"heldout {key} parent is missing")
        _require(not actual.exists() and not actual.is_symlink(), f"heldout {key} already exists")
        paths[key] = actual
    _require(output == paths["evaluation"], "--output must equal --heldout-evaluation")
    bound_selection = _absolute_path(ledger.get("selection_receipt_path"), "bound heldout selection")
    supplied_selection = _absolute_path(args.selection_receipt, "heldout selection receipt", kind="file")
    _require(supplied_selection == bound_selection, "heldout selection path differs from binding")
    paths["selection"] = supplied_selection
    failure_entry = _mapping(
        ledger.get("failure_after_claim"), "heldout failure-after-claim contract"
    )
    failure = _absolute_path(
        failure_entry.get("path"), "bound heldout failure receipt"
    )
    _require(
        failure.parent.is_dir() and not failure.exists() and not failure.is_symlink(),
        "heldout failure receipt path is not fresh",
    )
    paths["failure"] = failure
    return paths


def build_evaluation_document(
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    spec: Mapping[str, Any],
    runtime: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    split: str,
    claim: Mapping[str, Any] | None,
    reload_parity: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_results(
        results, split=split, checkpoint_step=checkpoint["checkpoint_step"]
    )
    mechanism_correction = None
    if split == "mechanism":
        amendment = _mapping(
            context["binding"].get("user_authorized_gpu_amendment"),
            "binding GPU amendment",
        )
        mechanism_correction = dict(
            _mapping(
                amendment.get("mechanism_protocol_correction"),
                "binding mechanism protocol correction",
            )
        )
    return {
        "schema_name": EVALUATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": _utc_now(),
        "split": split,
        "stage": context["stage_name"],
        "binding": {
            "path": str(context["binding_path"]),
            "file_sha256": context["binding_file_sha256"],
            "binding_sha256": context["binding_sha256"],
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": file_sha256(Path(__file__).resolve()),
        },
        "cpu_contract_sha256": context["cpu_contract"]["contract_sha256"],
        "executable_config": {
            "path": str(context["config_path"]),
            "file_sha256": context["config_file_sha256"],
        },
        "checkpoint": {
            "path": str(checkpoint["checkpoint_path"]),
            "global_step": checkpoint["checkpoint_step"],
            "step": checkpoint["checkpoint_step"],
            "files": checkpoint["checkpoint_files"],
            "files_sha256": checkpoint["checkpoint_files_sha256"],
            "adapter_file_sha256": checkpoint["adapter_file_sha256"],
            "receipt_path": str(checkpoint["path"]),
            "receipt_file_sha256": checkpoint["file_sha256"],
            "receipt_sha256": checkpoint["receipt_sha256"],
        },
        "data": {
            "manifest_path": str(spec["manifest_path"]),
            "manifest_file_sha256": spec["manifest_file_sha256"],
            "manifest_sha256": spec["manifest_sha256"],
            "dataset_path": str(spec["dataset_path"]),
            "dataset_file_sha256": spec["dataset_entry"]["file_sha256"],
            "ordered_pair_ids_sha256": contract.object_sha256(
                [row["pair_id"] for row in results]
            ),
        },
        "reference_topology": {
            "policy": "selected_checkpoint_PEFT_adapter",
            "reference": "same_frozen_parent_inside_PeftModel.disable_adapter",
            "explicit_reference_model_loaded": False,
            "beta": float(context["config"]["beta"]),
            "response_logprob_reduction": "sum_over_response_only_tokens",
        },
        "runtime": runtime["runtime"],
        "process_identity": _process_identity(),
        "reload_parity": dict(reload_parity),
        "fresh_process_reload_proven": True,
        "adapter_reload_proven": True,
        "mechanism_protocol_correction": mechanism_correction,
        "mechanism_correction_claim": context.get(
            "mechanism_correction_claim"
        ),
        "heldout_claim": dict(claim) if claim is not None else None,
        "records": len(results),
        "pair_results": list(results),
        "aggregate": aggregate,
        "generation_performed": False,
        "sandbox_execution_performed": False,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.binding is not None, "--binding is required")
    _require(args.checkpoint_receipt is not None, "--checkpoint-receipt is required")
    _require(args.split is not None, "--split is required")
    context = validate_binding(args.binding, split=args.split)
    checkpoint = validate_checkpoint_receipt(
        args.checkpoint_receipt,
        context,
        split=args.split,
        checkpoint_step=args.checkpoint_step,
    )
    # Cheap, non-heldout process gate: reject an in-process pseudo-reload
    # before the one-shot claim can ever be created.
    validate_distinct_process(checkpoint)
    paths = validate_output_contract(args, context, checkpoint)
    spec = prepare_data_spec(context, split=args.split)
    selection: dict[str, Any] | None = None
    if args.split == "heldout":
        selection = validate_selection_receipt(paths["selection"], context, checkpoint)

    # Weight loading and adapter attachment happen before a heldout claim, but
    # no heldout/source-pair bytes are opened until the immutable claim exists.
    runtime = load_policy(context, checkpoint, device_name=args.device)
    claim: dict[str, Any] | None = None
    claim_file_sha: str | None = None
    if args.split == "heldout":
        assert selection is not None
        claim_payload = {
            "schema_name": CLAIM_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "claimed",
            "claimed_at_utc": _utc_now(),
            "binding_sha256": context["binding_sha256"],
            "selection_receipt_file_sha256": selection["file_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "selected_step": checkpoint["checkpoint_step"],
            "selected_checkpoint_path": str(checkpoint["checkpoint_path"]),
            "selected_checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "selected_adapter_file_sha256": checkpoint["adapter_file_sha256"],
            "heldout_file_sha256": spec["dataset_entry"]["file_sha256"],
            "heldout_ordered_pair_ids_sha256": spec["dataset_entry"]["ordered_pair_ids_sha256"],
        }
        claim, claim_file_sha = write_exclusive_json(
            paths["claim"], claim_payload, self_field="claim_sha256", label="heldout claim"
        )

    try:
        rows, source_by_id = load_evaluation_rows(spec, split=args.split)
        results = evaluate_rows(
            runtime,
            rows,
            source_by_id,
            beta=float(context["config"]["beta"]),
        )
        reload_parity = validate_reload_parity(checkpoint, results, split=args.split)
        evaluation = build_evaluation_document(
            context,
            checkpoint,
            spec,
            runtime,
            results,
            split=args.split,
            claim=claim,
            reload_parity=reload_parity,
        )
        sealed, evaluation_file_sha = write_exclusive_json(
            paths["output"],
            evaluation,
            self_field="evaluation_sha256",
            label=f"{args.split} preference evaluation",
        )
        consumption: dict[str, Any] | None = None
        if args.split == "heldout":
            assert claim is not None and claim_file_sha is not None and selection is not None
            consumption_payload = {
                "schema_name": CONSUMPTION_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "status": "consumed",
                "consumed_at_utc": _utc_now(),
                "binding_sha256": context["binding_sha256"],
                "claim_file_sha256": claim_file_sha,
                "claim_sha256": claim["claim_sha256"],
                "selection_sha256": selection["selection_sha256"],
                "selected_step": checkpoint["checkpoint_step"],
                "evaluation_file_sha256": evaluation_file_sha,
                "evaluation_sha256": sealed["evaluation_sha256"],
                "records": len(results),
                "ordered_pair_ids_sha256": contract.object_sha256(
                    [row["pair_id"] for row in results]
                ),
                "attempts": 1,
            }
            consumption, _ = write_exclusive_json(
                paths["consumption"],
                consumption_payload,
                self_field="consumption_sha256",
                label="heldout consumption receipt",
            )
    except BaseException as error:
        if args.split == "heldout" and claim is not None:
            failure_payload = {
                "schema_name": "day23.qwen35_dpo_heldout_failure",
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed_no_retry",
                "failed_at_utc": _utc_now(),
                "binding_sha256": context["binding_sha256"],
                "claim_sha256": claim["claim_sha256"],
                "selected_step": checkpoint["checkpoint_step"],
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error_message": str(error),
                "traceback": traceback.format_exception(error)[-12:],
                "retry_allowed": False,
                "manual_adjudication_required": True,
            }
            try:
                write_exclusive_json(
                    paths["failure"],
                    failure_payload,
                    self_field="failure_sha256",
                    label="heldout failure receipt",
                )
            except BaseException as seal_error:
                raise Day23PreferenceEvaluationError(
                    f"heldout failed after claim and failure receipt could not be sealed: {seal_error}"
                ) from error
        raise
    return {
        "status": "pass",
        "split": args.split,
        "output": str(paths["output"]),
        "evaluation_sha256": sealed["evaluation_sha256"],
        "records": len(results),
        "consumption_sha256": (
            consumption["consumption_sha256"] if consumption is not None else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _require(
                all(
                    value is None
                    for value in (
                        args.binding,
                        args.checkpoint_receipt,
                        args.checkpoint_step,
                        args.split,
                        args.output,
                        True if args.select_dev else None,
                        args.selection_receipt,
                        args.heldout_claim,
                        args.heldout_evaluation,
                        args.heldout_consumption,
                    )
                ),
                "--self-test cannot be combined with evaluation arguments",
            )
            _self_test()
            print(json.dumps({"status": "pass", "scope": "cpu_in_memory_no_weights"}, sort_keys=True))
            return 0
        if args.select_dev:
            _require(args.binding is not None and args.output is not None, "--select-dev requires --binding and --output")
            _require(
                all(
                    value is None
                    for value in (
                        args.checkpoint_receipt,
                        args.checkpoint_step,
                        args.split,
                        args.selection_receipt,
                        args.heldout_claim,
                        args.heldout_evaluation,
                        args.heldout_consumption,
                    )
                ),
                "--select-dev cannot be combined with evaluation arguments",
            )
            print(
                json.dumps(
                    run_dev_selection(args.binding, args.output),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        print(json.dumps(run_evaluation(args), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "fail_closed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
