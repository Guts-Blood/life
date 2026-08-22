#!/usr/bin/env python3
"""Verify the remote S1/runtime and emit sealed executable Day 25 GRPO configs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging import version

import day25_contract as contract


DAY21_DIR = contract.BOOTCAMP_ROOT / "day-21-weekend-eval-reading"
if str(DAY21_DIR) not in sys.path:
    sys.path.insert(0, str(DAY21_DIR))

import day21_s1_handoff as day21  # noqa: E402


# Updated only after the CPU bundle is final.  This is deliberately outside the
# CPU contract's implementation-source closure, avoiding a self-referential hash.
EXPECTED_CPU_CONTRACT_SHA256 = "59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c"
DEFAULT_CPU_CONTRACT = contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/cpu-run-contract.json"
DEFAULT_ARGUMENT_AUDIT = contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-argument-audit.json"
DEFAULT_PROMOTION = contract.ARTIFACTS / "checkpoints/day21-qwen35-s1-promotion-manifest.json"
DEFAULT_KEY = contract.ARTIFACTS / "checkpoints/day21-qwen35-s1-downstream-key.json"
DEFAULT_EXPORT = contract.ARTIFACTS / "checkpoints/day21-qwen35-s1-merged-export-manifest.json"
STAGE_METADATA = {"policy_version", "purpose", "claim_boundary"}
CONFIG_NAMES = {
    "g1_rollout_only": "g1-rollout-only.json",
    "g2_one_update": "g2-one-update.json",
    "g3_weight_sync_probe": "g3-weight-sync-probe.json",
    "g4_bounded_short_run": "g4-bounded-short-run.json",
}
RUNTIME_EXACT = {
    "ms-swift": "4.5.0.dev0",
    "torch": "2.10.0+cu128",
    "transformers": "5.12.1",
    "peft": "0.19.1",
    "e2b": "2.37.0",
}


class Day25GPUBindingError(ValueError):
    """A paid-runtime binding prerequisite failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25GPUBindingError(message)


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return resolved


def _canonical_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    _require(expanded.is_absolute(), f"{label} must be absolute")
    resolved = expanded.resolve()
    _require(resolved == expanded, f"{label} must already be canonical")
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return resolved


def _file_manifest(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    directory = _canonical_directory(root, "merged S1 export")
    excluded = set(exclude)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), f"merged S1 export contains a symlink: {path}")
        if path.is_file() and path.name not in excluded:
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": contract.file_sha256(path),
            }
    return files


def _runtime_identity(ms_swift_root: Path) -> dict[str, Any]:
    checkout = _canonical_directory(ms_swift_root, "ms-swift checkout")
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(commit == contract.MS_SWIFT_COMMIT and not dirty, "ms-swift checkout drifted or is dirty")
    versions: dict[str, str] = {}
    for package in (*RUNTIME_EXACT, "vllm", "trl", "accelerate", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise Day25GPUBindingError(f"required GPU package is missing: {package}") from error
    _require(
        all(versions[name] == expected for name, expected in RUNTIME_EXACT.items()),
        f"exact GPU runtime versions drifted: {versions}",
    )
    _require(version.parse(versions["vllm"]) >= version.parse("0.17.0"), "vLLM must be >=0.17.0")
    _require(version.parse(versions["trl"]) >= version.parse("0.20.0"), "TRL must be >=0.20")
    try:
        import torch
        import swift
    except ImportError as error:
        raise Day25GPUBindingError("GPU torch/ms-swift import failed") from error
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "binding requires exactly one visible CUDA GPU")
    _require(
        Path(swift.__file__).resolve().parent == (checkout / "swift").resolve(),
        "runtime imported ms-swift outside the pinned checkout",
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "versions": versions,
        "ms_swift_checkout": str(checkout),
        "ms_swift_commit": commit,
        "ms_swift_clean": True,
        "cuda_device_count": 1,
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        },
    }


def _verify_parent(
    cpu: Mapping[str, Any],
    *,
    promotion_path: Path,
    key_path: Path,
    export_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    parent = cpu["parent"]
    promotion_path = _regular_file(promotion_path, "Day 21 promotion")
    key_path = _regular_file(key_path, "Day 21 downstream key")
    export_path = _regular_file(export_path, "Day 21 export manifest")
    model_path = _canonical_directory(model_path, "promoted merged S1")
    _require(str(model_path) == parent["remote_merged_export_path"], "model path is not the promoted merged S1")
    _require(contract.file_sha256(promotion_path) == parent["promotion_file_sha256"], "promotion file drifted")
    _require(contract.file_sha256(key_path) == parent["downstream_key_file_sha256"], "downstream key file drifted")
    _require(contract.file_sha256(export_path) == parent["merged_export_file_sha256"], "export manifest file drifted")
    try:
        promotion = day21.verify_promotion(promotion_path)
        key = day21.verify_downstream_key(key_path)
        export = day21.verify_export_manifest(export_path)
    except (OSError, ValueError, day21.HandoffError) as error:
        raise Day25GPUBindingError(f"Day 21 handoff verification failed: {error}") from error
    promotion_hash = promotion.pop("_verified_promotion_manifest_sha256")
    _require(promotion_hash == parent["promotion_manifest_sha256"], "promotion content drifted")
    _require(key["key_sha256"] == parent["downstream_key_sha256"], "downstream key content drifted")
    _require(export["manifest_sha256"] == parent["merged_export_manifest_sha256"], "export content drifted")
    _require(key["downstream_key"] == parent["downstream_key"], "S1 downstream key drifted")
    files = _file_manifest(model_path, exclude=("S1-EXPORT-MANIFEST.json",))
    _require(files == export["files"], "promoted merged S1 file bytes drifted")
    _require(contract.object_sha256(files) == parent["merged_export_artifact_sha256"], "merged S1 file manifest hash drifted")
    return {
        "status": "verified_remote_promoted_s1",
        "model_path": str(model_path),
        "downstream_key": parent["downstream_key"],
        "checkpoint_id": parent["checkpoint_id"],
        "promotion_manifest_sha256": promotion_hash,
        "export_manifest_sha256": export["manifest_sha256"],
        "merged_export_files_sha256": contract.object_sha256(files),
        "merged_export_file_count": len(files),
        "day23_checkpoint_used": False,
    }


def _project_config(
    cpu: Mapping[str, Any], stage_name: str, model_path: Path, run_root: Path
) -> dict[str, Any]:
    result = dict(cpu["intended_ms_swift_args"])
    result.update(
        {
            key: item
            for key, item in cpu["stages"][stage_name].items()
            if key not in STAGE_METADATA
        }
    )
    result["model"] = str(model_path)
    result["dataset"] = [
        str((contract.BOOTCAMP_ROOT / item).resolve()) for item in result["dataset"]
    ]
    result["external_plugins"] = [
        str((contract.BOOTCAMP_ROOT / item).resolve())
        for item in result["external_plugins"]
    ]
    result["output_dir"] = str((run_root / "outputs" / stage_name).resolve())
    _require(not any(item is None or item == [] for item in result.values()), f"{stage_name}: null/empty CLI value")
    _require("__BIND_" not in json.dumps(result, sort_keys=True), f"{stage_name}: unresolved placeholder")
    for value in result["dataset"] + result["external_plugins"]:
        _require(_regular_file(Path(value), f"{stage_name} input").is_file(), f"{stage_name}: missing input")
    return result


def _parse_configs(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from swift.arguments import RLHFArguments
        from swift.cli.main import parse_yaml_args
        from swift.rewards import orms
        from swift.utils import parse_args
    except ImportError as error:
        raise Day25GPUBindingError("pinned RLHF JSON CLI is unavailable") from error
    fields = {field.name for field in dataclasses.fields(RLHFArguments)}
    result: dict[str, Any] = {}
    env = {
        "DAY25_RUN_ID": "day25-binding-parse",
        "DAY25_POLICY_VERSION": "v0",
        "DAY25_TRAJECTORY_LEDGER": "/tmp/day25-binding-parse-ledger.jsonl",
        "DAY25_SANDBOX_WORKERS": "4",
    }
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        with tempfile.TemporaryDirectory(prefix="day25-gpu-bind-") as temporary:
            root = Path(temporary)
            for stage_name, config in configs.items():
                _require(not sorted(set(config) - fields), f"{stage_name}: unknown RLHF argument key")
                parse_config = dict(config)
                parse_config["output_dir"] = str(root / f"output-{stage_name}")
                path = root / f"{stage_name}.json"
                path.write_bytes(contract.json_bytes(parse_config))
                argv = [str(path)]
                old_swift_config = os.environ.get("SWIFT_CONFIG_FILE")
                try:
                    parse_yaml_args(argv)
                    parsed, remaining = parse_args(RLHFArguments, argv)
                finally:
                    if old_swift_config is None:
                        os.environ.pop("SWIFT_CONFIG_FILE", None)
                    else:
                        os.environ["SWIFT_CONFIG_FILE"] = old_swift_config
                _require(not remaining, f"{stage_name}: unparsed JSON args")
                _require(
                    parsed.rlhf_type == "grpo"
                    and parsed.beta == 0.0
                    and parsed.ref_model is None
                    and parsed.adapters == []
                    and parsed.ref_adapters == []
                    and parsed.resume_from_checkpoint is None
                    and parsed.num_generations == 4
                    and parsed.generation_batch_size == 8
                    and parsed.training_args.steps_per_generation == 8
                    and parsed.vllm_enable_lora
                    and parsed.vllm_mode == "colocate"
                    and "day25_mbpp_tests_only" in orms,
                    f"{stage_name}: parsed GRPO semantics drifted",
                )
                train, val = parsed.load_dataset()
                _require(len(train) == 32 and val is None, f"{stage_name}: data loader drifted")
                result[stage_name] = {
                    "status": "pass",
                    "max_steps": parsed.max_steps,
                    "train_records": len(train),
                    "steps_per_generation": parsed.training_args.steps_per_generation,
                }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result


def bind(
    *,
    cpu_path: Path,
    argument_audit_path: Path,
    promotion_path: Path,
    key_path: Path,
    export_path: Path,
    model_path: Path,
    ms_swift_root: Path,
    run_root: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    cpu_path = _regular_file(cpu_path, "Day 25 CPU contract")
    cpu = contract.load_json(cpu_path)
    cpu_hash = contract.verify_seal(cpu, "contract_sha256", "Day 25 CPU contract")
    _require(
        cpu_hash == EXPECTED_CPU_CONTRACT_SHA256,
        f"CPU contract is not the finalized trust root: {cpu_hash}",
    )
    argument_path = _regular_file(argument_audit_path, "Day 25 argument audit")
    argument = contract.load_json(argument_path)
    contract.verify_seal(argument, "audit_sha256", "Day 25 argument audit")
    _require(argument.get("status") == "pass", "CPU argument audit did not pass")
    _require(argument["run_contract"]["content_sha256"] == cpu_hash, "argument audit binds another CPU contract")
    for label, expected in cpu["implementation_sources"].items():
        if label.startswith("pinned_ms_swift_"):
            continue
        path = (
            contract.DAY25_DIR / label
            if (contract.DAY25_DIR / label).is_file()
            else contract.BOOTCAMP_ROOT / "day-24-online-rl-dataflow-reward" / label
        )
        _require(path.is_file() and contract.file_sha256(path) == expected, f"implementation source drifted: {label}")
    runtime = _runtime_identity(ms_swift_root)
    parent = _verify_parent(
        cpu,
        promotion_path=promotion_path,
        key_path=key_path,
        export_path=export_path,
        model_path=model_path,
    )
    destination = run_root.expanduser()
    _require(destination.is_absolute() and destination == destination.resolve(), "run_root must be canonical absolute")
    _require(not destination.exists(), f"run_root already exists: {destination}")
    configs = {
        stage: _project_config(cpu, stage, model_path.resolve(), destination)
        for stage in CONFIG_NAMES
    }
    parse_results = _parse_configs(configs)
    payloads = {
        f"configs/{CONFIG_NAMES[stage]}": contract.json_bytes(config)
        for stage, config in configs.items()
    }
    binding: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_gpu_binding",
        "schema_version": 1,
        "status": "gpu_config_bound_preflight_pending",
        "run_root": str(destination),
        "cpu_contract": {
            "path": str(cpu_path),
            "file_sha256": contract.file_sha256(cpu_path),
            "content_sha256": cpu_hash,
        },
        "argument_audit": {
            "path": str(argument_path),
            "file_sha256": contract.file_sha256(argument_path),
            "content_sha256": argument["audit_sha256"],
        },
        "runtime": runtime,
        "parent": parent,
        "config_parse": parse_results,
        "configs": {
            stage: {
                "path": f"configs/{CONFIG_NAMES[stage]}",
                "file_sha256": hashlib.sha256(
                    payloads[f"configs/{CONFIG_NAMES[stage]}"]
                ).hexdigest(),
                "max_steps": config["max_steps"],
            }
            for stage, config in configs.items()
        },
        "remaining_gates": [
            "live E2B connectivity",
            "Qwen3.5 vLLM engine load and LoRA attachment",
            "measured peak VRAM with at least 15% free margin",
            "G1 rollout-only ledger offline audit",
        ],
    }
    binding["binding_sha256"] = contract.object_sha256(binding)
    payloads["binding.json"] = contract.json_bytes(binding)
    return payloads, binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-contract", type=Path, default=DEFAULT_CPU_CONTRACT)
    parser.add_argument("--argument-audit", type=Path, default=DEFAULT_ARGUMENT_AUDIT)
    parser.add_argument("--promotion", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--downstream-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--merged-export-manifest", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ms-swift-root", type=Path, default=contract.REPO_ROOT / "vendor/ms-swift")
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payloads, binding = bind(
            cpu_path=args.cpu_contract,
            argument_audit_path=args.argument_audit,
            promotion_path=args.promotion,
            key_path=args.downstream_key,
            export_path=args.merged_export_manifest,
            model_path=args.model,
            ms_swift_root=args.ms_swift_root,
            run_root=args.run_root,
        )
        for relative, payload in payloads.items():
            contract.write_atomic(args.run_root / relative, payload, overwrite=False)
    except (
        OSError,
        subprocess.CalledProcessError,
        contract.Day25ContractError,
        Day25GPUBindingError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": binding["status"],
                "run_root": binding["run_root"],
                "binding_sha256": binding["binding_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
