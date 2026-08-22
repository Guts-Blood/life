#!/usr/bin/env python3
"""Parse every Day 25 GRPO stage with pinned ms-swift without loading weights."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract


DEFAULT_RUN_CONTRACT = contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/cpu-run-contract.json"
DEFAULT_MODEL_PATH = (
    contract.REPO_ROOT
    / "tmp/qwen35-v2/models/Qwen--Qwen3.5-4B-Base"
    / contract.MODEL_REVISION
)
DEFAULT_OUTPUT = contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-argument-audit.json"
STAGE_METADATA = {"policy_version", "purpose", "claim_boundary"}
CLI_DEFAULTS = {
    "adapters": [],
    "ref_adapters": [],
    "ref_model": None,
    "resume_from_checkpoint": None,
}


class Day25ArgumentAuditError(ValueError):
    """The frozen GRPO config is not executable under pinned argument semantics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25ArgumentAuditError(message)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _bound_config(
    value: Mapping[str, Any], stage_name: str, model_path: Path
) -> dict[str, Any]:
    result = dict(value["intended_ms_swift_args"])
    stage = value["stages"][stage_name]
    result.update({key: item for key, item in stage.items() if key not in STAGE_METADATA})
    result["model"] = str(model_path.resolve())
    result["dataset"] = [
        str((contract.BOOTCAMP_ROOT / item).resolve()) for item in result["dataset"]
    ]
    result["external_plugins"] = [
        str((contract.BOOTCAMP_ROOT / item).resolve())
        for item in result["external_plugins"]
    ]
    result["output_dir"] = f"/tmp/day25-{stage_name}-parse-only"
    _require(not any(item is None or item == [] for item in result.values()), "bound config contains null/empty list")
    _require("__BIND_" not in json.dumps(result, sort_keys=True), "bound config contains a placeholder")
    return result


def _parse_json(config: Mapping[str, Any], path: Path) -> Any:
    try:
        from swift.arguments import RLHFArguments
        from swift.cli.main import parse_yaml_args
        from swift.utils import parse_args
    except ImportError as error:
        raise Day25ArgumentAuditError("pinned ms-swift JSON CLI is unavailable") from error
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    argv = [str(path)]
    previous = os.environ.get("SWIFT_CONFIG_FILE")
    try:
        parse_yaml_args(argv)
        parsed, remaining = parse_args(RLHFArguments, argv)
    except (OSError, SystemExit, ValueError, AssertionError) as error:
        raise Day25ArgumentAuditError(
            f"real JSON CLI parse failed: {type(error).__name__}: {error}"
        ) from error
    finally:
        if previous is None:
            os.environ.pop("SWIFT_CONFIG_FILE", None)
        else:
            os.environ["SWIFT_CONFIG_FILE"] = previous
    _require(not remaining, f"unparsed JSON arguments: {remaining}")
    return parsed


def _advantage_parity() -> dict[str, Any]:
    import torch
    from swift.rl_core.advantage import compute_advantages

    rewards = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0 / 3.0, 2.0 / 3.0]
    tensor = torch.tensor(rewards, dtype=torch.float64).unsqueeze(1)
    actual, scalar_rewards = compute_advantages(
        tensor,
        torch.tensor([1.0], dtype=torch.float64),
        num_generations=4,
        advantage_estimator="grpo",
        scale_rewards="group",
        kl_in_reward=False,
        beta=0.0,
    )
    expected: list[float] = []
    for start in range(0, len(rewards), 4):
        group = rewards[start : start + 4]
        mean = sum(group) / 4
        std = math.sqrt(sum((value - mean) ** 2 for value in group) / 3)
        expected.extend(
            [0.0] * 4
            if std == 0.0
            else [(value - mean) / (std + 1e-4) for value in group]
        )
    maximum_error = max(
        abs(observed - reference)
        for observed, reference in zip(actual.tolist(), expected)
    )
    _require(scalar_rewards.tolist() == rewards, "trainer scalar reward aggregation drifted")
    _require(maximum_error <= 1e-12, f"trainer advantage parity failed: {maximum_error}")
    return {
        "status": "pass",
        "fixture_rewards": rewards,
        "trainer_advantages": actual.tolist(),
        "offline_advantages": expected,
        "maximum_absolute_error": maximum_error,
        "ddof": 1,
        "epsilon": 1e-4,
    }


def build_audit(
    *,
    run_contract_path: Path = DEFAULT_RUN_CONTRACT,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    try:
        import swift
        from swift.arguments import RLHFArguments
        from swift.rewards import orms
    except ImportError as error:
        raise Day25ArgumentAuditError("pinned ms-swift runtime is unavailable") from error
    _require(
        Path(swift.__file__).resolve().parent
        == (contract.REPO_ROOT / "vendor/ms-swift/swift").resolve(),
        "argument audit imported ms-swift outside the pinned checkout",
    )
    checkout = contract.REPO_ROOT / "vendor/ms-swift"
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
    _require(commit == contract.MS_SWIFT_COMMIT and not dirty, "pinned ms-swift checkout drifted or is dirty")

    run_contract_path = run_contract_path.resolve()
    model_path = model_path.resolve()
    _require(model_path.is_dir(), f"local parse-only model snapshot is missing: {model_path}")
    value = contract.load_json(run_contract_path)
    run_hash = contract.verify_seal(value, "contract_sha256", "Day 25 CPU run contract")
    _require(value.get("status") == "cpu_ready_gpu_pending", "CPU contract status drifted")
    fields = {field.name for field in dataclasses.fields(RLHFArguments)}
    common = value["intended_ms_swift_args"]
    _require(not sorted(set(common) - fields), f"unknown common args: {sorted(set(common) - fields)}")
    _require(value["binding_rule"]["omit_cli_defaults"] == CLI_DEFAULTS, "CLI default omission contract drifted")

    previous_env = {
        key: os.environ.get(key)
        for key in (
            "DAY25_RUN_ID",
            "DAY25_POLICY_VERSION",
            "DAY25_TRAJECTORY_LEDGER",
            "DAY25_SANDBOX_WORKERS",
        )
    }
    os.environ.update(
        {
            "DAY25_RUN_ID": "day25-parse-only",
            "DAY25_POLICY_VERSION": "v0",
            "DAY25_TRAJECTORY_LEDGER": "/tmp/day25-parse-only-ledger.jsonl",
            "DAY25_SANDBOX_WORKERS": "4",
        }
    )
    stages: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="day25-json-cli-") as temporary:
            root = Path(temporary)
            for stage_name, stage in value["stages"].items():
                overrides = {key: item for key, item in stage.items() if key not in STAGE_METADATA}
                _require(not sorted(set(overrides) - fields), f"unknown {stage_name} args")
                config = _bound_config(value, stage_name, model_path)
                parsed = _parse_json(config, root / f"{stage_name}.json")
                _require(parsed.rlhf_type == "grpo", f"{stage_name}: algorithm drifted")
                _require(parsed.model == str(model_path), f"{stage_name}: parse-only model drifted")
                _require(parsed.adapters == [] and parsed.ref_adapters == [], f"{stage_name}: adapter defaults drifted")
                _require(parsed.ref_model is None, f"{stage_name}: beta=0 must not load a reference")
                _require(parsed.resume_from_checkpoint is None, f"{stage_name}: must start fresh")
                _require(parsed.beta == 0.0 and parsed.kl_in_reward is False, f"{stage_name}: KL contract drifted")
                _require(parsed.num_generations == 4, f"{stage_name}: G drifted")
                _require(parsed.generation_batch_size == 8, f"{stage_name}: generation batch drifted")
                training_args = parsed.training_args
                _require(training_args.steps_per_generation == 8, f"{stage_name}: resolved steps/generation drifted")
                _require(parsed.gradient_accumulation_steps == 8, f"{stage_name}: accumulation drifted")
                _require(parsed.scale_rewards == "group", f"{stage_name}: normalization drifted")
                _require(parsed.max_length == 1024 and parsed.max_completion_length == 512, f"{stage_name}: length drifted")
                _require(parsed.vllm_max_model_len == 1152, f"{stage_name}: vLLM length drifted")
                _require(parsed.use_vllm and parsed.vllm_mode == "colocate", f"{stage_name}: topology drifted")
                _require(parsed.vllm_enable_lora, f"{stage_name}: LoRA sync disabled")
                _require(parsed.freeze_vit and parsed.freeze_aligner and not parsed.freeze_llm, f"{stage_name}: freeze contract drifted")
                _require(parsed.reward_funcs == ["day25_mbpp_tests_only"], f"{stage_name}: reward plugin drifted")
                _require("day25_mbpp_tests_only" in orms, f"{stage_name}: reward plugin not registered")
                _require(parsed.max_steps == stage["max_steps"], f"{stage_name}: max_steps drifted")
                train_dataset, val_dataset = parsed.load_dataset()
                _require(len(train_dataset) == 32 and val_dataset is None, f"{stage_name}: dataset loader drifted")
                first = train_dataset[0]
                for column in (
                    "task_family_id",
                    "code_prefix",
                    "reward_tests",
                    "task_manifest_sha256",
                    "reward_payload_sha256",
                ):
                    _require(column in first, f"{stage_name}: reward column dropped: {column}")
                reward_class = orms["day25_mbpp_tests_only"]
                reward_instance = reward_class(args=parsed)
                _require(hasattr(reward_instance, "adapter"), f"{stage_name}: reward adapter did not initialize")
                stages[stage_name] = {
                    "status": "pass",
                    "max_steps": parsed.max_steps,
                    "dataset_records": len(train_dataset),
                    "validation_records": 0,
                    "reward_plugin_registered_and_initialized": True,
                    "resolved": {
                        "ref_model": parsed.ref_model,
                        "adapters": parsed.adapters,
                        "ref_adapters": parsed.ref_adapters,
                        "resume_from_checkpoint": parsed.resume_from_checkpoint,
                        "steps_per_generation": training_args.steps_per_generation,
                        "generation_batch_size": parsed.generation_batch_size,
                        "num_generations": parsed.num_generations,
                        "beta": parsed.beta,
                        "scale_rewards": parsed.scale_rewards,
                        "vllm_mode": parsed.vllm_mode,
                        "vllm_enable_lora": parsed.vllm_enable_lora,
                    },
                }
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    result: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_argument_audit",
        "schema_version": 1,
        "status": "pass",
        "scope": "real_pinned_json_cli_plugin_loader_and_advantage_no_model_weights",
        "run_contract": {
            "path": contract.relative_to_bootcamp(run_contract_path),
            "file_sha256": contract.file_sha256(run_contract_path),
            "content_sha256": run_hash,
        },
        "pinned_ms_swift": {
            "commit": commit,
            "checkout_clean": True,
            "import_path": str(Path(swift.__file__).resolve()),
            "advantage_source_sha256": contract.file_sha256(
                checkout / "swift/rl_core/advantage.py"
            ),
            "reward_bridge_source_sha256": contract.file_sha256(
                checkout / "swift/rl_core/grpo_algorithm.py"
            ),
        },
        "rlhf_argument_field_count": len(fields),
        "stages": stages,
        "advantage_parity": _advantage_parity(),
        "runtime": {
            "python": platform.python_version(),
            "ms_swift": _package_version("ms-swift"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
            "peft": _package_version("peft"),
            "trl": _package_version("trl"),
            "vllm": _package_version("vllm"),
            "e2b": _package_version("e2b"),
        },
        "claim_boundary": {
            "json_cli_and_dataset_loader": "pass",
            "reward_plugin_import_and_init": "pass",
            "pinned_advantage_math": "pass",
            "model_weights_loaded": False,
            "vllm_engine_created": False,
            "e2b_live_call": False,
            "gpu_ready": False,
        },
    }
    result["audit_sha256"] = contract.object_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-contract", type=Path, default=DEFAULT_RUN_CONTRACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("build", "rebuild", "check"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_audit(run_contract_path=args.run_contract, model_path=args.model)
        payload = contract.json_bytes(result)
        if args.mode == "build":
            contract.write_atomic(args.output, payload, overwrite=False)
        elif args.mode == "rebuild":
            contract.write_atomic(args.output, payload, overwrite=True)
        else:
            _require(args.output.is_file() and args.output.read_bytes() == payload, "argument audit drifted")
    except (OSError, subprocess.CalledProcessError, contract.Day25ContractError, Day25ArgumentAuditError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": args.mode,
                "stages": sorted(result["stages"]),
                "audit_sha256": result["audit_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
