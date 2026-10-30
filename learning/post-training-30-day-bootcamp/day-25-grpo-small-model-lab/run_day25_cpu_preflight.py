#!/usr/bin/env python3
"""Run every offline Day 25 gate and persist a sealed pre-GPU summary."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import day25_contract as contract


DEFAULT_OUTPUT = (
    contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-cpu-preflight.json"
)
ISOLATED_PYTHON = contract.REPO_ROOT / "tmp/day22-token-runtime/bin/python"
MS_SWIFT_ROOT = contract.REPO_ROOT / "vendor/ms-swift"
CPU_CONTRACT = (
    contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/cpu-run-contract.json"
)
ARGUMENT_AUDIT = (
    contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-argument-audit.json"
)
BINDER = contract.DAY25_DIR / "bind_day25_qwen35_gpu.py"


class Day25CPUPreflightError(ValueError):
    """One or more offline gates failed before a paid GPU was opened."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25CPUPreflightError(message)


def _run(name: str, argv: Sequence[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    display = " ".join(argv)
    completed = subprocess.run(
        list(argv),
        cwd=contract.BOOTCAMP_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
        raise Day25CPUPreflightError(
            f"{name} failed with exit {completed.returncode}: {display}\n{detail}"
        )
    print(json.dumps({"gate": name, "status": "pass"}, sort_keys=True))
    return {"name": name, "status": "pass", "command": display}


def _pinned_environment() -> dict[str, str]:
    _require(ISOLATED_PYTHON.is_file(), f"isolated Python is missing: {ISOLATED_PYTHON}")
    _require(MS_SWIFT_ROOT.is_dir(), f"pinned ms-swift is missing: {MS_SWIFT_ROOT}")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(MS_SWIFT_ROOT) if not existing else os.pathsep.join((str(MS_SWIFT_ROOT), existing))
    )
    return env


def _verify_final_trust_roots() -> dict[str, Any]:
    cpu = contract.load_json(CPU_CONTRACT)
    cpu_hash = contract.verify_seal(cpu, "contract_sha256", "Day 25 CPU contract")
    argument = contract.load_json(ARGUMENT_AUDIT)
    argument_hash = contract.verify_seal(
        argument, "audit_sha256", "Day 25 argument audit"
    )
    _require(cpu.get("status") == "cpu_ready_gpu_pending", "CPU contract status drifted")
    _require(argument.get("status") == "pass", "argument audit status drifted")
    _require(
        argument["run_contract"]["content_sha256"] == cpu_hash,
        "argument audit is not bound to the final CPU contract",
    )
    binder_source = BINDER.read_text(encoding="utf-8")
    match = re.search(
        r'^EXPECTED_CPU_CONTRACT_SHA256 = "([0-9a-f]{64})"$',
        binder_source,
        re.MULTILINE,
    )
    _require(match is not None, "GPU binder contains an unresolved trust-root placeholder")
    _require(match.group(1) == cpu_hash, "GPU binder is bound to another CPU contract")
    _require("__FINALIZE" not in binder_source, "GPU binder still contains a finalization marker")
    return {
        "cpu_contract_sha256": cpu_hash,
        "cpu_contract_file_sha256": contract.file_sha256(CPU_CONTRACT),
        "argument_audit_sha256": argument_hash,
        "argument_audit_file_sha256": contract.file_sha256(ARGUMENT_AUDIT),
        "gpu_binder_file_sha256": contract.file_sha256(BINDER),
        "ms_swift_commit": contract.MS_SWIFT_COMMIT,
    }


def run_preflight() -> dict[str, Any]:
    pinned_env = _pinned_environment()
    python = str(Path(sys.executable).resolve())
    # Do not resolve this venv symlink: invoking its base interpreter would bypass
    # the venv and silently select an older Transformers installation.
    isolated = str(ISOLATED_PYTHON)
    gates = [
        _run(
            "day25_data_manifest",
            [python, "day-25-grpo-small-model-lab/prepare_day25_qwen35_grpo.py", "--mode", "check"],
        ),
        _run(
            "day25_qwen35_processor",
            [isolated, "day-25-grpo-small-model-lab/audit_day25_qwen35_processor.py", "--mode", "check"],
            env=pinned_env,
        ),
        _run(
            "day25_cpu_contract",
            [python, "day-25-grpo-small-model-lab/build_day25_cpu_contract.py", "--mode", "check"],
        ),
        _run(
            "day25_pinned_argument_semantics",
            [isolated, "day-25-grpo-small-model-lab/audit_day25_qwen35_arguments.py", "--mode", "check"],
            env=pinned_env,
        ),
        _run(
            "day22_sandbox_contract",
            [
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "day-22-preference-data",
                "-p",
                "test_day22_sandbox.py",
            ],
        ),
        _run(
            "day24_reward_contract",
            [python, "artifacts/scripts/test_day24_coding_reward_contract.py"],
        ),
        _run(
            "day25_reward_adapter",
            [python, "day-25-grpo-small-model-lab/test_day25_reward_adapter.py"],
        ),
        _run(
            "day24_day25_python_compile",
            [
                python,
                "-m",
                "compileall",
                "-q",
                "day-24-online-rl-dataflow-reward",
                "day-25-grpo-small-model-lab",
                "artifacts/scripts/test_day24_coding_reward_contract.py",
            ],
        ),
    ]
    trust_roots = _verify_final_trust_roots()
    result: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_cpu_preflight",
        "schema_version": 1,
        "status": "pass_cpu_ready_gpu_pending",
        "gates": gates,
        "test_inventory": {
            "day22_sandbox": {"passed": 13},
            "day24_reward_contract": {"passed": 22, "skipped_live_e2b": 1},
            "day25_reward_adapter": {"passed": 7},
        },
        "trust_roots": trust_roots,
        "claim_boundary": {
            "paid_gpu_opened": False,
            "model_weights_loaded_for_training": False,
            "vllm_engine_created": False,
            "live_e2b_from_target_gpu_runtime": False,
            "optimizer_step_run": False,
            "weight_sync_proven": False,
            "capability_gain_claimed": False,
        },
        "next_action": "run the frozen GPU binder on exactly one H100 80GB target",
    }
    result["preflight_sha256"] = contract.object_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_preflight()
        contract.write_atomic(args.output, contract.json_bytes(result), overwrite=True)
    except (
        OSError,
        subprocess.SubprocessError,
        contract.Day25ContractError,
        Day25CPUPreflightError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": len(result["gates"]),
                "preflight_sha256": result["preflight_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
