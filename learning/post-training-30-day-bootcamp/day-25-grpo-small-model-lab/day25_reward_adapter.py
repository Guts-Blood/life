#!/usr/bin/env python3
"""Thin Day 24 verifier adapter used by the Day 25 ms-swift reward plugin."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import day25_contract as contract


DAY24_PATH = (
    contract.BOOTCAMP_ROOT
    / "day-24-online-rl-dataflow-reward/day24_coding_verifier.py"
)
DAY20_DIR = contract.BOOTCAMP_ROOT / "day-20-qwen35-balanced-lora-sft"
if str(DAY20_DIR) not in sys.path:
    sys.path.insert(0, str(DAY20_DIR))

from day20_contract_v2 import (  # noqa: E402
    Day20V2ContractError,
    validate_raw_code_continuation,
)


def _load_day24() -> Any:
    spec = importlib.util.spec_from_file_location("day25_day24_verifier", DAY24_PATH)
    if spec is None or spec.loader is None:
        raise contract.Day25ContractError("cannot load the Day 24 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


day24 = _load_day24()

NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
ADAPTER_VERSION = "day25.ms_swift_mbpp_tests_only.v1"
LEDGER_SCHEMA = "day25.coding_reward_batch"


class Day25RewardAdapterError(RuntimeError):
    """The reward batch cannot be safely converted into trainer scalars."""


class Day25RolloutOnlyStop(Day25RewardAdapterError):
    """An intentional G1 stop after durable rewards and before optimizer work."""


def _column(kwargs: Mapping[str, Any], name: str, size: int) -> list[Any]:
    value = kwargs.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Day25RewardAdapterError(f"reward column is missing or not batched: {name}")
    result = list(value)
    if len(result) != size:
        raise Day25RewardAdapterError(
            f"reward column length mismatch: {name}={len(result)}, completions={size}"
        )
    return result


def _completion_boundary(message_content: str) -> tuple[str, str]:
    if message_content.startswith(NON_THINKING_PREFIX):
        return message_content[len(NON_THINKING_PREFIX) :], "known_non_thinking_prefix_removed"
    return message_content, "message_content_equals_generated_decode_expected"


def _format_contract(completion: str, code_prefix: str) -> dict[str, Any]:
    try:
        result = validate_raw_code_continuation(completion, code_prefix)
    except Day20V2ContractError as error:
        return {
            "valid": False,
            "execution_eligible": False,
            "validator": "day20.validate_raw_code_continuation",
            "evidence": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    return {
        "valid": True,
        "execution_eligible": True,
        "validator": "day20.validate_raw_code_continuation",
        "evidence": result.as_evidence(),
        "error": None,
    }


def _response_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Day25RewardAdapterError("response_token_ids is not a sequence")
    flat: list[int] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            flat.extend(item)
        else:
            flat.append(item)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in flat):
        raise Day25RewardAdapterError("response_token_ids contains an invalid token ID")
    return flat


def _reward_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "task_family_id",
            "code_prefix",
            "entry_point",
            "test_setup_code",
            "reward_tests",
            "tests_manifest_sha256",
        )
    }


def _trajectory(
    *,
    index: int,
    group_ordinal: int,
    completion_message: str,
    metadata: Mapping[str, Any],
    run_id: str,
    policy_version: str,
) -> dict[str, Any]:
    family_id = str(metadata["task_family_id"])
    code_prefix = str(metadata["code_prefix"])
    reward_tests = metadata["reward_tests"]
    if isinstance(reward_tests, (str, bytes)) or not isinstance(reward_tests, Sequence):
        raise Day25RewardAdapterError(f"{family_id}: reward_tests is invalid")
    reward_tests = [str(item) for item in reward_tests]
    if not reward_tests or any(not item.strip() for item in reward_tests):
        raise Day25RewardAdapterError(f"{family_id}: reward_tests is empty")
    payload = _reward_payload({**metadata, "reward_tests": reward_tests})
    if metadata["reward_payload_sha256"] != contract.object_sha256(payload):
        raise Day25RewardAdapterError(f"{family_id}: reward payload hash drifted")

    completion, boundary = _completion_boundary(completion_message)
    response_ids = _response_token_ids(metadata.get("response_token_ids"))
    prompt_id = str(metadata.get("prompt_id") or f"day25:prompt:{family_id}")
    request_id = str(metadata.get("request_id") or f"day25:request:{index}")
    group_id = f"day25:group:{policy_version}:{prompt_id}"
    tests = {
        "visible_examples": [],
        "reward_tests": [
            {"test_id": f"{family_id}:reward:{ordinal:02d}", "source": source}
            for ordinal, source in enumerate(reward_tests)
        ],
        "frozen_eval_tests": [],
    }
    tests["manifest_sha256"] = day24.object_sha256(tests)
    task = {
        "task_id": str(metadata.get("task_id", family_id.rsplit(":", 1)[-1])),
        "family_id": family_id,
        "split": "train",
        "language": "python",
        "prompt": str(metadata.get("prompt") or ""),
        "problem": str(metadata.get("problem") or ""),
        "code_prefix": code_prefix,
        "entry_point": str(metadata["entry_point"]),
        "test_setup_code": str(metadata.get("test_setup_code") or ""),
        "source": {
            "namespace": "day25.frozen_train32",
            "source_record_sha256": str(metadata.get("source_record_sha256") or ""),
        },
    }
    task["manifest_sha256"] = day24.object_sha256(task)
    format_contract = _format_contract(completion, code_prefix)
    verifier = day24.verifier_contract()
    reward_policy = day24.policy_contract(contract.REWARD_POLICY)
    trajectory: dict[str, Any] = {
        "schema_name": "day24.coding_trajectory",
        "schema_version": day24.SCHEMA_VERSION,
        "run_id": run_id,
        "trajectory_id": f"{run_id}:{policy_version}:{request_id}",
        "group_id": group_id,
        "group_ordinal": group_ordinal,
        "prompt_id": prompt_id,
        "completion_id": f"day25:completion:{contract.text_sha256(completion_message)}",
        "modality": {"value": "text", "image_tokens": 0, "video_tokens": 0},
        "task": task,
        "code_artifact": {
            "language": "python",
            "code_prefix": code_prefix,
            "completion": completion,
            "completion_sha256": day24.text_sha256(completion),
            "message_content": completion_message,
            "message_content_sha256": day24.text_sha256(completion_message),
            "boundary": boundary,
            "format_contract": format_contract,
            "finish_reason": str(metadata.get("finish_reason") or "unknown"),
        },
        "tokens": {
            "response_token_ids": response_ids,
            "response_token_ids_sha256": contract.object_sha256(response_ids),
            "response_token_count": len(response_ids),
            "availability": "provided_by_ms_swift_reward_kwargs",
        },
        "processor": day24.PROCESSOR_PROVENANCE,
        "policy": {
            "role": "rollout_old",
            "policy_version": policy_version,
            "parent_role": contract.POLICY_PARENT_ROLE,
        },
        "logprobs": {
            "old_logprob": None,
            "current_logprob": None,
            "reference_logprob": None,
            "availability": "trainer_owned_not_reward_adapter",
        },
        "tests": tests,
        "contracts": {
            "verifier_contract_sha256": verifier["contract_sha256"],
            "sandbox_contract_sha256": day24.object_sha256(day24.SANDBOX_CONTRACT),
            "reward_policy_sha256s": {
                contract.REWARD_POLICY: reward_policy["policy_sha256"]
            },
        },
        "source_parent": {
            "namespace": "day25.frozen_train32",
            "task_manifest_sha256": str(metadata["task_manifest_sha256"]),
            "reward_payload_sha256": str(metadata["reward_payload_sha256"]),
            "day23_checkpoint_used": False,
        },
    }
    trajectory["deterministic_replay_key"] = day24._replay_key(trajectory)
    return day24.seal(trajectory, "trajectory_sha256")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    import fcntl

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = contract.canonical_json(row) + b"\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class Day25RewardAdapter:
    """Convert a full ms-swift reward batch into sealed Day 24 evidence."""

    def __init__(
        self,
        *,
        executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        run_id: str,
        policy_version: str,
        ledger_path: Path | None,
        group_size: int = contract.GROUP_SIZE,
        workers: int = 4,
        abort_after_complete_batch: bool = False,
    ) -> None:
        if group_size != contract.GROUP_SIZE:
            raise Day25RewardAdapterError("Day 25 requires G=4")
        if not run_id or not policy_version:
            raise Day25RewardAdapterError("run_id and policy_version are required")
        if workers <= 0:
            raise Day25RewardAdapterError("workers must be positive")
        self.executor = executor
        self.run_id = run_id
        self.policy_version = policy_version
        self.ledger_path = ledger_path
        self.group_size = group_size
        self.workers = workers
        self.abort_after_complete_batch = abort_after_complete_batch

    def _verify_with_retry(self, trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        for retry in range(2):
            evidence = day24.verify_trajectory(
                trajectory, self.executor, replay_ordinal=1
            )
            evidence["infra_retry_ordinal"] = retry
            if attempts:
                evidence["retry_of_event_id"] = attempts[0]["event_id"]
            evidence["day25_adapter_version"] = ADAPTER_VERSION
            evidence["policy_version"] = trajectory["policy"]["policy_version"]
            evidence = day24.seal(evidence, "evidence_sha256")
            attempts.append(evidence)
            if evidence["status"] != "infra_error":
                break
        return attempts

    def score_batch(
        self, completions: Sequence[str], **kwargs: Any
    ) -> list[float]:
        if isinstance(completions, (str, bytes)) or not isinstance(completions, Sequence):
            raise Day25RewardAdapterError("completions must be a batched sequence")
        values = list(completions)
        if not values or len(values) % self.group_size:
            raise Day25RewardAdapterError("reward batch size must be a non-zero multiple of G=4")
        if any(not isinstance(item, str) for item in values):
            raise Day25RewardAdapterError("completion is not text")

        trainer_state = kwargs.get("trainer_state")
        global_step = getattr(trainer_state, "global_step", None)
        if global_step is not None and (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise Day25RewardAdapterError("trainer_state.global_step is invalid")
        effective_policy_version = (
            self.policy_version
            if global_step is None
            else f"{self.policy_version}:trainer-step-{global_step}"
        )

        required = (
            "task_id",
            "task_family_id",
            "prompt",
            "problem",
            "code_prefix",
            "entry_point",
            "test_setup_code",
            "reward_tests",
            "tests_manifest_sha256",
            "task_manifest_sha256",
            "reward_payload_sha256",
            "source_record_sha256",
            "prompt_id",
            "request_id",
            "finish_reason",
            "response_token_ids",
        )
        columns = {name: _column(kwargs, name, len(values)) for name in required}
        metadata_rows = [
            {name: column[index] for name, column in columns.items()}
            for index in range(len(values))
        ]
        family_counts: dict[str, int] = {}
        family_prompt_ids: dict[str, set[str]] = {}
        for row in metadata_rows:
            family = str(row["task_family_id"])
            family_counts[family] = family_counts.get(family, 0) + 1
            family_prompt_ids.setdefault(family, set()).add(str(row["prompt_id"]))
        if any(count != self.group_size for count in family_counts.values()):
            raise Day25RewardAdapterError(
                f"reward batch is not exactly one G=4 group per family: {family_counts}"
            )
        if any(len(values) != 1 for values in family_prompt_ids.values()):
            raise Day25RewardAdapterError(
                f"one family mapped to multiple prompt IDs: {family_prompt_ids}"
            )
        if len({str(row["request_id"]) for row in metadata_rows}) != len(metadata_rows):
            raise Day25RewardAdapterError("reward batch contains duplicate request IDs")

        family_ordinals: dict[str, int] = {}
        trajectories: list[dict[str, Any]] = []
        for index, (completion, metadata) in enumerate(zip(values, metadata_rows)):
            family = str(metadata["task_family_id"])
            group_ordinal = family_ordinals.get(family, 0)
            family_ordinals[family] = group_ordinal + 1
            trajectories.append(
                _trajectory(
                    index=index,
                    group_ordinal=group_ordinal,
                    completion_message=completion,
                    metadata=metadata,
                    run_id=self.run_id,
                    policy_version=effective_policy_version,
                )
            )
        attempts_by_index: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(self.workers, len(trajectories))) as pool:
            futures = {
                pool.submit(self._verify_with_retry, trajectory): index
                for index, trajectory in enumerate(trajectories)
            }
            for future in as_completed(futures):
                attempts_by_index[futures[future]] = future.result()

        records: list[dict[str, Any]] = []
        persistent_infra: list[str] = []
        rewards: list[float] = []
        for index, trajectory in enumerate(trajectories):
            attempts = attempts_by_index[index]
            terminal = attempts[-1]
            if terminal["status"] == "infra_error":
                persistent_infra.append(str(trajectory["trajectory_id"]))
                reward = None
            else:
                reward_row = day24.score_verification(terminal, contract.REWARD_POLICY)
                reward = reward_row["aggregate_reward"]
                if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                    raise Day25RewardAdapterError("eligible terminal evidence did not produce a scalar")
                rewards.append(float(reward))
            records.append(
                {
                    "trajectory": copy.deepcopy(trajectory),
                    "attempts": copy.deepcopy(attempts),
                    "terminal_evidence_sha256": terminal["evidence_sha256"],
                    "reward": (
                        None
                        if reward is None
                        else day24.score_verification(terminal, contract.REWARD_POLICY)
                    ),
                }
            )

        status = (
            "aborted_persistent_infra"
            if persistent_infra
            else "complete_rollout_only_stop"
            if self.abort_after_complete_batch
            else "complete"
        )
        batch: dict[str, Any] = {
            "schema_name": LEDGER_SCHEMA,
            "schema_version": 1,
            "adapter_version": ADAPTER_VERSION,
            "run_id": self.run_id,
            "configured_policy_version": self.policy_version,
            "policy_version": effective_policy_version,
            "trainer_global_step": global_step,
            "group_size": self.group_size,
            "status": status,
            "atomic_scalar_returned": status == "complete",
            "persistent_infra_trajectory_ids": persistent_infra,
            "records": records,
        }
        batch = contract.seal(batch, "batch_sha256")
        if self.ledger_path is not None:
            _append_jsonl(self.ledger_path, batch)
        if persistent_infra:
            raise Day25RewardAdapterError(
                "persistent sandbox infra failure; reward batch aborted before returning scalars: "
                + ", ".join(persistent_infra)
            )
        if self.abort_after_complete_batch:
            raise Day25RolloutOnlyStop(
                "intentional G1 rollout-only stop after the sealed reward batch; "
                "no reward scalars were returned to the trainer"
            )
        if len(rewards) != len(values):
            raise Day25RewardAdapterError("reward output lost completion order")
        return rewards
