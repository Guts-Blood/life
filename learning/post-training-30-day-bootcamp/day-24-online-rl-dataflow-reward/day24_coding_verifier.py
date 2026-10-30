#!/usr/bin/env python3
"""Day 24 MBPP verifier, reward contract, and CPU mini-pipeline.

This module deliberately leaves the frozen Day 22 scorer unchanged.  It owns
task semantics (per-test evidence), pure reward policies, group reduction, and
the thin E2B adapter needed to replay untrusted promoted-S1 completions.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
BOOTCAMP = HERE.parent
ARTIFACTS = BOOTCAMP / "artifacts"
DAY22_DIR = BOOTCAMP / "day-22-preference-data"
DAY22_SCORER = DAY22_DIR / "score_day22_mbpp_sandbox.py"

DEFAULT_ROLLOUTS = ARTIFACTS / "data/day22-qwen35-formal-s1-rollout-candidates.jsonl"
DEFAULT_SCHEMA = ARTIFACTS / "data/day24-coding-trajectory-schema-v2.json"
DEFAULT_INPUT = ARTIFACTS / "data/day24-coding-mini-pipeline-input.jsonl"
DEFAULT_EVIDENCE = ARTIFACTS / "eval/day24-coding-verifier-evidence.jsonl"
DEFAULT_REWARDS = ARTIFACTS / "eval/day24-coding-group-rewards.jsonl"

SCHEMA_VERSION = 2
VERIFIER_VERSION = "day24.mbpp_per_test_verifier.v1"
GROUP_REDUCER_VERSION = "day24.pinned_ms_swift_group_zscore.v2"
GROUP_REDUCER_DDOF = 1
GROUP_REDUCER_EPSILON = 1e-4
SELECTED_COHORT: dict[str, tuple[int, ...]] = {
    "mbpp:task:602": (0, 3, 1, 2),
    "mbpp:task:604": (0, 1, 2, 4),
}
FROZEN_EVAL_TESTS: dict[str, tuple[str, ...]] = {
    "mbpp:task:602": (
        'assert first_repeated_char("abccba") == "c"',
        'assert first_repeated_char("abcdd") == "d"',
    ),
    "mbpp:task:604": (
        'assert reverse_words("one two three") == "three two one"',
        'assert reverse_words("hello, world!") == "world! hello,"',
    ),
}

PROCESSOR_PROVENANCE = {
    "model_key": "Qwen/Qwen3.5-4B-Base",
    "model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "processor_contract_sha256": "12af5b3df5597fc2de4e6e9eebc9abba2e0f14d27971f39dc7e4af83c66e1083",
    "processor_sha256": "c282b8ca1bacf0627bfaad55ad64c65e8232afb7f0b435b20ff4bf1a0f81dbc6",
    "tokenizer_sha256": "4570b194bd50f1fe92788865224ce75518f64963ad699cb32148655bdd3eed4e",
    "template_revision": "day20.qwen35_code_boundary_target_v3",
    "template_sha256": "7f835342861cc9aabfd6299228033df80437b183227e04db311450437cd07ded",
}

SANDBOX_CONTRACT = {
    "backend": "e2b_firecracker",
    "template_id": "rki5dems9wqfm4r03t7g",
    "sdk": {"package": "e2b", "version": "2.37.0"},
    "secure": True,
    "allow_internet_access": False,
    "allow_public_traffic": False,
    "deny_out": ["0.0.0.0/0"],
    "fresh_sandbox_per_completion": True,
    "testcase_isolation": "fresh_python_process_and_cwd_shared_completion_vm",
    "candidate_bytes_in_shell": False,
    "limits": {
        "cpu_seconds_per_test": 2,
        "address_space_bytes": 536_870_912,
        "file_size_bytes": 1_048_576,
        "open_files": 64,
        "processes": 32,
        "core_bytes": 0,
        "wall_seconds_per_test": 2.0,
        "max_output_bytes": 8192,
    },
}

REWARD_POLICIES = {
    "tests_only": {
        "version": "day24.tests_only.v1",
        "weights": {"correctness": 1.0},
        "safety_hard_gate": True,
        "frozen_eval_in_aggregate": False,
    },
    "tests_format_style": {
        "version": "day24.tests_format_style.v1",
        "weights": {"correctness": 0.90, "format": 0.05, "style": 0.05},
        "safety_hard_gate": True,
        "frozen_eval_in_aggregate": False,
    },
}


class Day24VerifierError(ValueError):
    """A Day 24 contract, input, or evidence invariant failed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = object_sha256(result)
    return result


def verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if expected != actual:
        raise Day24VerifierError(f"{label} self hash mismatch")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day24VerifierError(f"invalid JSON on line {number}: {path}") from error
        if not isinstance(value, dict):
            raise Day24VerifierError(f"line {number} is not an object: {path}")
        rows.append(value)
    if not rows:
        raise Day24VerifierError(f"empty JSONL: {path}")
    return rows


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise Day24VerifierError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _write_atomic(path, payload, overwrite=overwrite)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    _write_atomic(path, payload, overwrite=overwrite)


def verifier_contract() -> dict[str, Any]:
    value = {
        "schema_name": "day24.coding_verifier_contract",
        "schema_version": 1,
        "version": VERIFIER_VERSION,
        "language": "python",
        "test_families": ["visible_example", "reward", "frozen_eval"],
        "statuses": [
            "ok",
            "parse_error",
            "compile_error",
            "wrong_answer",
            "runtime_error",
            "candidate_timeout",
            "infra_error",
            "truncated",
            "aborted",
        ],
        "infra_reward": None,
        "missing_component": "fail_closed",
        "sandbox_contract_sha256": object_sha256(SANDBOX_CONTRACT),
    }
    return seal(value, "contract_sha256")


def policy_contract(name: str) -> dict[str, Any]:
    if name not in REWARD_POLICIES:
        raise Day24VerifierError(f"unknown reward policy: {name}")
    return seal(
        {"name": name, **REWARD_POLICIES[name]},
        "policy_sha256",
    )


def _tests_for_family(family_id: str, source_tests: Sequence[str]) -> dict[str, Any]:
    frozen = FROZEN_EVAL_TESTS.get(family_id)
    if frozen is None:
        raise Day24VerifierError(f"frozen tests missing for {family_id}")
    groups = {
        "visible_examples": [],
        "reward_tests": [
            {"test_id": f"{family_id}:reward:{index:02d}", "source": source}
            for index, source in enumerate(source_tests)
        ],
        "frozen_eval_tests": [
            {"test_id": f"{family_id}:frozen:{index:02d}", "source": source}
            for index, source in enumerate(frozen)
        ],
    }
    groups["manifest_sha256"] = object_sha256(groups)
    return groups


def _replay_key(row: Mapping[str, Any]) -> str:
    material = {
        "task_manifest_sha256": row["task"]["manifest_sha256"],
        "completion_sha256": row["code_artifact"]["completion_sha256"],
        "tests_manifest_sha256": row["tests"]["manifest_sha256"],
        "verifier_contract_sha256": row["contracts"]["verifier_contract_sha256"],
        "sandbox_contract_sha256": row["contracts"]["sandbox_contract_sha256"],
        "reward_policy_sha256s": row["contracts"]["reward_policy_sha256s"],
    }
    return object_sha256(material)


def freeze_cohort(source_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in source_rows:
        family_id = row.get("task_family_id")
        sample_index = row.get("sample_index")
        if isinstance(family_id, str) and isinstance(sample_index, int):
            index[(family_id, sample_index)] = row
    contract = verifier_contract()
    policy_hashes = {
        name: policy_contract(name)["policy_sha256"] for name in REWARD_POLICIES
    }
    output: list[dict[str, Any]] = []
    for family_id, sample_indices in SELECTED_COHORT.items():
        group_id = f"day24:group:{family_id}"
        for group_ordinal, sample_index in enumerate(sample_indices):
            try:
                source = index[(family_id, sample_index)]
            except KeyError as error:
                raise Day24VerifierError(
                    f"source rollout missing: {family_id} sample {sample_index}"
                ) from error
            if source.get("split") != "train":
                raise Day24VerifierError(f"Day 24 cohort escaped train split: {family_id}")
            response = source["response"]
            if not isinstance(response, Mapping):
                raise Day24VerifierError("source response is not an object")
            tests = _tests_for_family(family_id, source["tests"]["test_list"])
            task = {
                "task_id": str(source["task_id"]),
                "family_id": family_id,
                "split": "train",
                "language": "python",
                "prompt": source["prompt"],
                "problem": source["problem"],
                "code_prefix": source["code_prefix"],
                "entry_point": source["entry_point"],
                "test_setup_code": source["tests"].get("test_setup_code", ""),
                "source": source["source"],
            }
            task["manifest_sha256"] = object_sha256(task)
            generator = source["generator"]
            policy = {
                "role": "rollout_old_current_same_promoted_s1",
                "candidate": generator["candidate"],
                "model_role": generator["model_role"],
                "downstream_key": generator["downstream_key"],
                "promotion_manifest_sha256": generator["promotion_manifest_sha256"],
                "source_checkpoint_integrity_sha256": generator[
                    "source_checkpoint_integrity_sha256"
                ],
                "merged_snapshot_sha256": generator["merged_snapshot_sha256"],
                "rollout_contract_sha256": generator["rollout_contract_sha256"],
                "runtime_sha256": generator["runtime_sha256"],
                "generation_config_sha256": generator["generation_config_sha256"],
            }
            trajectory = {
                "schema_name": "day24.coding_trajectory",
                "schema_version": SCHEMA_VERSION,
                "run_id": "day24-cpu-mini-pipeline-v1",
                "trajectory_id": f"day24:{family_id}:sample:{sample_index:02d}",
                "group_id": group_id,
                "group_ordinal": group_ordinal,
                "prompt_id": f"day24:prompt:{family_id}",
                "completion_id": source["candidate_id"],
                "modality": {"value": "text", "image_tokens": 0, "video_tokens": 0},
                "task": task,
                "code_artifact": {
                    "language": "python",
                    "code_prefix": source["code_prefix"],
                    "completion": response["text"],
                    "completion_sha256": response["sha256"],
                    "format_contract": source["format_contract"],
                    "finish_reason": response["finish_reason"],
                },
                "tokens": {
                    "input_token_ids": response["prompt_token_ids"],
                    "input_token_ids_sha256": response["prompt_token_ids_sha256"],
                    "output_token_ids": response["generated_token_ids"],
                    "output_token_ids_sha256": response["generated_token_ids_sha256"],
                    "response_span": [
                        response["prompt_token_count"],
                        response["prompt_token_count"] + response["generated_token_count"],
                    ],
                    "loss_mask": {
                        "value": None,
                        "availability": "not_computed_day24_cpu_contract",
                    },
                },
                "processor": PROCESSOR_PROVENANCE,
                "policy": policy,
                "logprobs": {
                    "old_logprob": None,
                    "current_logprob": None,
                    "reference_logprob": None,
                    "availability": "not_computed_day24_cpu_contract",
                },
                "tests": tests,
                "contracts": {
                    "verifier_contract_sha256": contract["contract_sha256"],
                    "sandbox_contract_sha256": object_sha256(SANDBOX_CONTRACT),
                    "reward_policy_sha256s": policy_hashes,
                },
                "source_parent": {
                    "namespace": "day22.promoted_s1_rollout_candidate",
                    "source_candidate_id": source["candidate_id"],
                    "source_row_sha256": object_sha256(source),
                    "day23_failure_checkpoint_used": False,
                },
            }
            trajectory["deterministic_replay_key"] = _replay_key(trajectory)
            output.append(seal(trajectory, "trajectory_sha256"))
    if len(output) != 8 or {row["group_id"] for row in output} != {
        "day24:group:mbpp:task:602",
        "day24:group:mbpp:task:604",
    }:
        raise Day24VerifierError("frozen cohort is not exactly two G=4 groups")
    return output


def _style_and_safety(
    program: str, completion_start_line: int
) -> tuple[float, float, list[str]]:
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return 0.0, 1.0, []
    completion_nodes = [
        node
        for node in ast.walk(tree)
        if getattr(node, "lineno", 0) >= completion_start_line
    ]
    safety_reasons: list[str] = []
    forbidden_calls = {"eval", "exec", "open", "compile", "__import__"}
    for node in completion_nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            safety_reasons.append("completion_import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                safety_reasons.append(f"forbidden_call:{node.func.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"system", "popen", "spawn", "fork"}:
                safety_reasons.append(f"forbidden_attribute:{node.func.attr}")
    style_score = (
        0.0
        if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in completion_nodes)
        else 1.0
    )
    return style_score, 0.0 if safety_reasons else 1.0, sorted(set(safety_reasons))


def _semantic_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": value["trajectory_id"],
        "deterministic_replay_key": value["deterministic_replay_key"],
        "status": value["status"],
        "eligible_for_reward": value["eligible_for_reward"],
        "parse": value["parse"],
        "compile": value["compile"],
        "test_cases": [
            {
                "test_id": case["test_id"],
                "family": case["family"],
                "status": case["status"],
                "error_type": case.get("error_type"),
                "stdout_sha256": case.get("stdout_sha256"),
                "stderr_sha256": case.get("stderr_sha256"),
            }
            for case in value["test_cases"]
        ],
        "summary": value["summary"],
        "format_score": value["format_score"],
        "style_score": value["style_score"],
        "safety_score": value["safety_score"],
        "safety_reasons": value["safety_reasons"],
        "sandbox_digest": value.get("sandbox_digest"),
    }


def _summary(test_cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("visible_example", "reward", "frozen_eval"):
        cases = [case for case in test_cases if case.get("family") == family]
        result[family] = {
            "passed": sum(case.get("status") == "pass" for case in cases),
            "total": len(cases),
            "status_counts": {
                status: sum(case.get("status") == status for case in cases)
                for status in sorted({str(case.get("status")) for case in cases})
            },
        }
    return result


def verify_trajectory(
    trajectory: Mapping[str, Any],
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    replay_ordinal: int = 1,
) -> dict[str, Any]:
    verify_seal(trajectory, "trajectory_sha256", "trajectory")
    started = datetime.now(timezone.utc).isoformat()
    completion = str(trajectory["code_artifact"]["completion"])
    code_prefix = str(trajectory["task"]["code_prefix"]).rstrip()
    program = f"{code_prefix}\n{completion.rstrip()}\n"
    completion_start_line = code_prefix.count("\n") + 2
    style_score, safety_score, safety_reasons = _style_and_safety(
        program, completion_start_line
    )
    format_score = 1.0 if trajectory["code_artifact"]["format_contract"].get("valid") else 0.0
    evidence: dict[str, Any] = {
        "schema_name": "day24.coding_verifier_evidence",
        "schema_version": 1,
        "event_id": f"day24-event-{uuid.uuid4().hex}",
        "started_at_utc": started,
        "replay_ordinal": replay_ordinal,
        "trajectory_id": trajectory["trajectory_id"],
        "group_id": trajectory["group_id"],
        "completion_id": trajectory["completion_id"],
        "deterministic_replay_key": trajectory["deterministic_replay_key"],
        "verifier_contract_sha256": trajectory["contracts"]["verifier_contract_sha256"],
        "format_score": format_score,
        "style_score": style_score,
        "safety_score": safety_score,
        "safety_reasons": safety_reasons,
    }
    if trajectory["code_artifact"].get("finish_reason") != "stop":
        evidence.update(
            {
                "status": "truncated",
                "eligible_for_reward": True,
                "parse": {"status": "not_run"},
                "compile": {"status": "not_run"},
                "test_cases": [],
                "summary": _summary([]),
                "sandbox_digest": None,
                "sandbox": None,
                "duration_ms": 0.0,
            }
        )
    else:
        try:
            ast.parse(program)
        except SyntaxError as error:
            evidence.update(
                {
                    "status": "parse_error",
                    "eligible_for_reward": True,
                    "parse": {"status": "fail", "error_type": type(error).__name__},
                    "compile": {"status": "not_run"},
                    "test_cases": [],
                    "summary": _summary([]),
                    "sandbox_digest": None,
                    "sandbox": None,
                    "duration_ms": 0.0,
                }
            )
        else:
            request = {
                "program": program,
                "setup_code": trajectory["task"].get("test_setup_code", ""),
                "test_cases": [
                    {**case, "family": family}
                    for family, key in (
                        ("visible_example", "visible_examples"),
                        ("reward", "reward_tests"),
                        ("frozen_eval", "frozen_eval_tests"),
                    )
                    for case in trajectory["tests"][key]
                ],
                "sandbox_contract": SANDBOX_CONTRACT,
            }
            execution = dict(executor(request))
            test_cases = list(execution.get("test_cases") or [])
            compile_result = dict(execution.get("compile") or {"status": "not_run"})
            infra = execution.get("status") == "infra_error" or any(
                case.get("status") == "infra_error" for case in test_cases
            )
            reward_cases = [case for case in test_cases if case.get("family") == "reward"]
            if infra:
                status = "infra_error"
                eligible = False
            elif compile_result.get("status") == "fail":
                status = "compile_error"
                eligible = True
            elif any(case.get("status") == "candidate_timeout" for case in reward_cases):
                status = "candidate_timeout"
                eligible = True
            elif any(case.get("status") == "runtime_error" for case in reward_cases):
                status = "runtime_error"
                eligible = True
            elif any(case.get("status") != "pass" for case in reward_cases):
                status = "wrong_answer"
                eligible = True
            else:
                status = "ok"
                eligible = True
            evidence.update(
                {
                    "status": status,
                    "eligible_for_reward": eligible,
                    "parse": {"status": "pass"},
                    "compile": compile_result,
                    "test_cases": test_cases,
                    "summary": _summary(test_cases),
                    "sandbox_digest": execution.get("sandbox_digest"),
                    "sandbox": execution.get("sandbox"),
                    "duration_ms": execution.get("duration_ms"),
                }
            )
    semantic = _semantic_evidence(evidence)
    evidence["semantic_result_sha256"] = object_sha256(semantic)
    return seal(evidence, "evidence_sha256")


def score_verification(evidence: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    verify_seal(evidence, "evidence_sha256", "verifier evidence")
    policy = policy_contract(policy_name)
    required = set(policy["weights"]) | {"safety"}
    summary = evidence.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("reward"), Mapping):
        raise Day24VerifierError("reward summary component is missing")
    reward_summary = summary["reward"]
    total = reward_summary.get("total")
    passed = reward_summary.get("passed")
    eligible = bool(evidence.get("eligible_for_reward"))
    terminal_zero_statuses = {"parse_error", "compile_error", "truncated"}
    terminal_zero = eligible and evidence.get("status") in terminal_zero_statuses
    if eligible and not terminal_zero and (
        not isinstance(total, int)
        or total <= 0
        or not isinstance(passed, int)
    ):
        raise Day24VerifierError("reward test counts are incomplete")
    components: dict[str, Any] = {
        "correctness": 0.0 if terminal_zero else (passed / total if eligible else None),
        "format": evidence.get("format_score"),
        "style": evidence.get("style_score"),
        "safety": evidence.get("safety_score"),
        "frozen_eval_correctness": (
            evidence["summary"]["frozen_eval"]["passed"]
            / evidence["summary"]["frozen_eval"]["total"]
            if evidence["summary"]["frozen_eval"]["total"]
            else None
        ),
    }
    if eligible and any(components.get(name) is None for name in required):
        raise Day24VerifierError("required reward component is missing")
    if not eligible:
        aggregate: float | None = None
    elif policy["safety_hard_gate"] and components["safety"] != 1.0:
        aggregate = 0.0
    else:
        aggregate = sum(
            float(components[name]) * float(weight)
            for name, weight in policy["weights"].items()
        )
    result = {
        "schema_name": "day24.coding_reward",
        "schema_version": 1,
        "trajectory_id": evidence["trajectory_id"],
        "group_id": evidence["group_id"],
        "semantic_result_sha256": evidence["semantic_result_sha256"],
        "policy": policy,
        "components": components,
        "aggregate_reward": aggregate,
        "eligible_for_group": aggregate is not None,
    }
    return seal(result, "reward_sha256")


def reduce_group(rewards: Sequence[Mapping[str, Any]], policy_name: str) -> dict[str, Any]:
    if len(rewards) != 4:
        raise Day24VerifierError("Day 24 group size must be exactly G=4")
    for reward in rewards:
        verify_seal(reward, "reward_sha256", "reward")
        if reward["policy"]["name"] != policy_name:
            raise Day24VerifierError("group mixed reward policies")
    group_ids = {str(reward["group_id"]) for reward in rewards}
    if len(group_ids) != 1:
        raise Day24VerifierError("group contains multiple group IDs")
    values = [reward.get("aggregate_reward") for reward in rewards]
    valid = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
    if not valid:
        status = "invalid_infra"
        mean = std = None
        members = [
            {
                "trajectory_id": reward["trajectory_id"],
                "aggregate_reward": reward["aggregate_reward"],
                "normalized_reward": None,
                "advantage": None,
            }
            for reward in rewards
        ]
        zero_variance = False
        optimizer_update_eligible = False
    else:
        numeric = [float(value) for value in values]
        mean = sum(numeric) / len(numeric)
        std = math.sqrt(
            sum((value - mean) ** 2 for value in numeric)
            / (len(numeric) - GROUP_REDUCER_DDOF)
        )
        zero_variance = std == 0.0
        normalized = (
            [0.0] * len(numeric)
            if zero_variance
            else [
                (value - mean) / (std + GROUP_REDUCER_EPSILON)
                for value in numeric
            ]
        )
        members = [
            {
                "trajectory_id": reward["trajectory_id"],
                "aggregate_reward": value,
                "normalized_reward": normalized[index],
                "advantage": normalized[index],
            }
            for index, (reward, value) in enumerate(zip(rewards, numeric))
        ]
        status = "zero_variance" if zero_variance else "ok"
        optimizer_update_eligible = not zero_variance
    result = {
        "schema_name": "day24.coding_group_reward",
        "schema_version": 1,
        "group_id": next(iter(group_ids)),
        "policy": policy_contract(policy_name),
        "reducer_version": GROUP_REDUCER_VERSION,
        "ddof": GROUP_REDUCER_DDOF,
        "epsilon": GROUP_REDUCER_EPSILON,
        "status": status,
        "group_size": 4,
        "mean": mean,
        "std": std,
        "zero_variance": zero_variance,
        "optimizer_update_eligible": optimizer_update_eligible,
        "members": members,
    }
    return seal(result, "group_reward_sha256")


def _day22_module() -> Any:
    spec = importlib.util.spec_from_file_location("day24_day22_sandbox", DAY22_SCORER)
    if spec is None or spec.loader is None:
        raise Day24VerifierError("cannot load frozen Day 22 sandbox adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _e2b_runner_source(request: Mapping[str, Any], result_path: str) -> str:
    payload = base64.b64encode(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    child_source = r'''#!/usr/bin/env python3
import base64, io, json, os, resource, sys

payload_path, result_path = sys.argv[1], sys.argv[2]
payload = json.loads(open(payload_path, encoding="utf-8").read())
limits = payload["limits"]
for rid, value in (
    (resource.RLIMIT_CORE, limits["core_bytes"]),
    (resource.RLIMIT_CPU, limits["cpu_seconds_per_test"]),
    (resource.RLIMIT_AS, limits["address_space_bytes"]),
    (resource.RLIMIT_FSIZE, limits["file_size_bytes"]),
    (resource.RLIMIT_NOFILE, limits["open_files"]),
    (resource.RLIMIT_NPROC, limits["processes"]),
):
    soft, hard = resource.getrlimit(rid)
    requested = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(rid, (requested, hard))

class Bounded(io.TextIOBase):
    encoding = "utf-8"
    def __init__(self, limit): self.limit, self.parts, self.total = limit, [], 0
    def writable(self): return True
    def write(self, value):
        encoded = str(value).encode("utf-8", "replace")
        self.total += len(encoded)
        used = sum(len(part.encode("utf-8")) for part in self.parts)
        if used < self.limit:
            self.parts.append(encoded[: self.limit-used].decode("utf-8", "ignore"))
        return len(str(value))
    def value(self): return "".join(self.parts)

out, err = Bounded(limits["max_output_bytes"]), Bounded(limits["max_output_bytes"])
oldout, olderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = out, err
result = {"status": "pass", "error_type": None}
try:
    source = base64.b64decode(payload["source_b64"], validate=True).decode("utf-8")
    exec(compile(source, "<day24-case>", "exec"), {"__name__": "__main__"})
except AssertionError:
    result = {"status": "wrong_answer", "error_type": "AssertionError"}
except BaseException as error:
    result = {"status": "runtime_error", "error_type": type(error).__name__}
finally:
    sys.stdout, sys.stderr = oldout, olderr
result.update({"stdout": out.value(), "stdout_total_bytes": out.total,
               "stderr": err.value(), "stderr_total_bytes": err.total})
data = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try: os.write(fd, data); os.fsync(fd)
finally: os.close(fd)
'''
    child_b64 = base64.b64encode(child_source.encode("utf-8")).decode("ascii")
    return f'''#!/usr/bin/env python3
import base64, hashlib, json, os, signal, subprocess, sys, tempfile, time
PAYLOAD = json.loads(base64.b64decode({payload!r}, validate=True).decode("utf-8"))
RESULT_PATH = {result_path!r}
CHILD = base64.b64decode({child_b64!r}, validate=True).decode("utf-8")
limits = PAYLOAD["sandbox_contract"]["limits"]
started = time.perf_counter()
compile_result = {{"status": "pass", "error_type": None}}
try:
    compile(PAYLOAD["setup_code"] + "\\n" + PAYLOAD["program"], "<day24-program>", "exec")
except SyntaxError as error:
    compile_result = {{"status": "fail", "error_type": type(error).__name__}}
cases = []
if compile_result["status"] == "pass":
    root = tempfile.mkdtemp(prefix="day24-cases-")
    child_path = os.path.join(root, "child.py")
    with open(child_path, "w", encoding="utf-8") as handle: handle.write(CHILD)
    os.chmod(child_path, 0o700)
    for index, case in enumerate(PAYLOAD["test_cases"]):
        case_dir = os.path.join(root, f"case-{{index:02d}}")
        os.mkdir(case_dir)
        payload_path = os.path.join(case_dir, "payload.json")
        case_result_path = os.path.join(case_dir, "result.json")
        source = PAYLOAD["setup_code"] + "\\n" + PAYLOAD["program"] + "\\n" + case["source"] + "\\n"
        child_payload = {{"source_b64": base64.b64encode(source.encode()).decode(), "limits": limits}}
        with open(payload_path, "w", encoding="utf-8") as handle: json.dump(child_payload, handle, sort_keys=True)
        case_started = time.perf_counter()
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", child_path, payload_path, case_result_path],
            cwd=case_dir, env={{"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=limits["wall_seconds_per_test"])
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            raw = {{"status": "candidate_timeout", "error_type": "wall_timeout",
                   "stdout": "", "stdout_total_bytes": 0, "stderr": "", "stderr_total_bytes": 0}}
        else:
            try:
                with open(case_result_path, encoding="utf-8") as handle: raw = json.load(handle)
            except BaseException:
                raw = {{"status": "runtime_error", "error_type": "missing_case_result",
                       "stdout": "", "stdout_total_bytes": 0, "stderr": "", "stderr_total_bytes": 0}}
        for stream in ("stdout", "stderr"):
            text = str(raw.get(stream) or "")
            raw[stream + "_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            raw.pop(stream, None)
        raw.update({{"test_id": case["test_id"], "family": case["family"],
                    "duration_ms": round((time.perf_counter()-case_started)*1000, 3)}})
        cases.append(raw)
result = {{"status": "ok", "compile": compile_result, "test_cases": cases,
          "duration_ms": round((time.perf_counter()-started)*1000, 3)}}
data = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
fd = os.open(RESULT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try: os.write(fd, data); os.fsync(fd)
finally: os.close(fd)
'''


def execute_e2b_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one completion's tests in a fresh attested E2B sandbox."""
    day22 = _day22_module()
    key = day22.read_e2b_api_key(day22.DEFAULT_E2B_CREDENTIAL)
    bindings = day22.load_e2b_bindings()
    sandbox = None
    started = time.perf_counter()
    result: dict[str, Any] = {"status": "infra_error", "compile": {"status": "not_run"}, "test_cases": []}
    runner_path = "/home/user/day24_verifier_runner.py"
    result_path = "/home/user/day24_verifier_result.json"
    try:
        try:
            sandbox = bindings["create"](**day22.e2b_create_kwargs(key))
        except Exception as error:
            result["error_type"] = f"sandbox_create_{type(error).__name__}"
            return result
        try:
            mismatch = day22._e2b_info_mismatch(sandbox.get_info())
        except Exception as error:
            result["error_type"] = f"sandbox_info_{type(error).__name__}"
            return result
        if mismatch is not None:
            result["error_type"] = f"sandbox_attestation_mismatch:{mismatch}"
            return result
        runner = _e2b_runner_source(request, result_path)
        sandbox.files.write(runner_path, runner)
        test_count = len(request["test_cases"])
        outer_timeout = min(50.0, test_count * float(SANDBOX_CONTRACT["limits"]["wall_seconds_per_test"]) + 12.0)
        try:
            command_result = sandbox.commands.run(
                f"python3 {runner_path} >/dev/null 2>/dev/null", timeout=outer_timeout
            )
            exit_code = getattr(command_result, "exit_code", None)
        except bindings["timeout_exception"]:
            result["error_type"] = "sdk_timeout"
            return result
        except bindings["command_exit_exception"] as error:
            exit_code = getattr(error, "exit_code", None)
        except Exception as error:
            result["error_type"] = f"command_{type(error).__name__}"
            return result
        try:
            raw = sandbox.files.read(result_path)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("result is not an object")
        except Exception:
            result["error_type"] = "missing_or_invalid_result"
            result["exit_code"] = exit_code
            return result
        result = parsed
        result["sandbox"] = SANDBOX_CONTRACT
        result["sandbox_digest"] = object_sha256(SANDBOX_CONTRACT)
        result["exit_code"] = exit_code
        return result
    finally:
        if sandbox is not None:
            try:
                killed = sandbox.kill()
                if killed is False:
                    result.clear()
                    result.update(
                        {
                            "status": "infra_error",
                            "error_type": "sandbox_kill_returned_false",
                            "compile": {"status": "not_run"},
                            "test_cases": [],
                        }
                    )
            except Exception as error:
                result.clear()
                result.update(
                    {
                        "status": "infra_error",
                        "error_type": f"sandbox_kill_{type(error).__name__}",
                        "compile": {"status": "not_run"},
                        "test_cases": [],
                    }
                )
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)


def build_trajectory_schema() -> dict[str, Any]:
    required = [
        "run_id", "trajectory_id", "group_id", "prompt_id", "completion_id",
        "deterministic_replay_key", "modality", "task", "code_artifact", "tokens",
        "processor", "policy", "logprobs", "tests", "contracts", "source_parent",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://local/day24-coding-trajectory-schema-v2.json",
        "title": "Day 24 coding online-RL trajectory v2",
        "description": "CPU contract schema; real old/current/reference log-probs remain explicitly unavailable until Day 25.",
        "type": "object",
        "required": required,
        "properties": {
            key: {"type": "object"} if key in {"modality", "task", "code_artifact", "tokens", "processor", "policy", "logprobs", "tests", "contracts", "source_parent"} else {"type": "string"}
            for key in required
        },
        "status_taxonomy": verifier_contract()["statuses"],
        "object_grains": ["prompt", "completion", "code_artifact", "execution_attempt", "trajectory", "group", "training_sample"],
        "reward_grains": ["raw_test_result", "reward_component", "aggregate_reward", "normalized_reward", "advantage"],
        "policy_roles": ["rollout", "old", "current", "reference"],
        "contracts": {
            "verifier": verifier_contract(),
            "sandbox": SANDBOX_CONTRACT,
            "reward_policies": {name: policy_contract(name) for name in REWARD_POLICIES},
            "group_reducer": {
                "version": GROUP_REDUCER_VERSION,
                "ddof": GROUP_REDUCER_DDOF,
                "epsilon": GROUP_REDUCER_EPSILON,
                "zero_variance_advantage": 0.0,
                "trainer_source": "pinned ms-swift swift/rl_core/advantage.py::compute_advantages",
            },
        },
    }


def prepare_outputs(*, rollouts: Path, schema: Path, cohort: Path, overwrite: bool) -> None:
    rows = freeze_cohort(load_jsonl(rollouts))
    write_json(schema, build_trajectory_schema(), overwrite=overwrite)
    write_jsonl(cohort, rows, overwrite=overwrite)


def run_pipeline(
    cohort_rows: Sequence[Mapping[str, Any]],
    *,
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    replays: int = 2,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if replays < 2:
        raise Day24VerifierError("Day 24 requires at least two semantic replays")
    def verify_with_retry(
        row: Mapping[str, Any], replay: int
    ) -> list[dict[str, Any]]:
        first = verify_trajectory(row, executor, replay_ordinal=replay)
        first["infra_retry_ordinal"] = 0
        first = seal(first, "evidence_sha256")
        if first["status"] != "infra_error":
            return [first]
        second = verify_trajectory(row, executor, replay_ordinal=replay)
        second["infra_retry_ordinal"] = 1
        second["retry_of_event_id"] = first["event_id"]
        second = seal(second, "evidence_sha256")
        return [first, second]

    evidence: list[dict[str, Any]] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for replay in range(1, replays + 1):
            for row in cohort_rows:
                future = pool.submit(verify_with_retry, row, replay)
                futures[future] = (row["trajectory_id"], replay)
        for future in as_completed(futures):
            evidence.extend(future.result())
    evidence.sort(
        key=lambda row: (
            row["trajectory_id"],
            row["replay_ordinal"],
            row.get("infra_retry_ordinal", 0),
        )
    )
    by_trajectory: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in evidence:
        by_trajectory.setdefault(row["trajectory_id"], {}).setdefault(
            int(row["replay_ordinal"]), []
        ).append(row)
    if set(by_trajectory) != {row["trajectory_id"] for row in cohort_rows}:
        raise Day24VerifierError("pipeline evidence coverage mismatch")
    canonical: dict[str, dict[str, Any]] = {}
    for trajectory_id, replay_attempts in by_trajectory.items():
        if set(replay_attempts) != set(range(1, replays + 1)):
            raise Day24VerifierError(f"replay ordinal coverage mismatch: {trajectory_id}")
        terminals = [attempts[-1] for _, attempts in sorted(replay_attempts.items())]
        hashes = {row["semantic_result_sha256"] for row in terminals}
        if len(hashes) != 1:
            raise Day24VerifierError(f"semantic replay mismatch: {trajectory_id}")
        canonical[trajectory_id] = terminals[0]
    group_rows: list[dict[str, Any]] = []
    group_ids = sorted({row["group_id"] for row in cohort_rows})
    for policy_name in REWARD_POLICIES:
        rewards = {
            trajectory_id: score_verification(row, policy_name)
            for trajectory_id, row in canonical.items()
        }
        for group_id in group_ids:
            ordered = sorted(
                (row for row in cohort_rows if row["group_id"] == group_id),
                key=lambda row: row["group_ordinal"],
            )
            group_rows.append(
                reduce_group([rewards[row["trajectory_id"]] for row in ordered], policy_name)
            )
    return evidence, group_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    prepare.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    prepare.add_argument("--cohort", type=Path, default=DEFAULT_INPUT)
    prepare.add_argument("--overwrite", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--cohort", type=Path, default=DEFAULT_INPUT)
    run.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    run.add_argument("--rewards", type=Path, default=DEFAULT_REWARDS)
    run.add_argument("--replays", type=int, default=2)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_outputs(
                rollouts=args.rollouts.resolve(),
                schema=args.schema.resolve(),
                cohort=args.cohort.resolve(),
                overwrite=args.overwrite,
            )
            print(json.dumps({"status": "prepared", "schema": str(args.schema), "cohort": str(args.cohort)}, sort_keys=True))
        else:
            rows = load_jsonl(args.cohort.resolve())
            evidence, rewards = run_pipeline(
                rows, executor=execute_e2b_request, replays=args.replays, workers=args.workers
            )
            write_jsonl(args.evidence, evidence, overwrite=args.overwrite)
            write_jsonl(args.rewards, rewards, overwrite=args.overwrite)
            print(json.dumps({"status": "complete", "evidence_rows": len(evidence), "group_reward_rows": len(rewards)}, sort_keys=True))
    except (Day24VerifierError, OSError, KeyError, TypeError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
