#!/usr/bin/env python3
"""slime v0.3.1 batch reward adapter for the frozen Day 24 verifier."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Sequence


DAY26_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY26_DIR.parent
DAY25_DIR = BOOTCAMP_ROOT / "day-25-grpo-small-model-lab"
if str(DAY25_DIR) not in sys.path:
    sys.path.insert(0, str(DAY25_DIR))

from day25_reward_adapter import Day25RewardAdapter, Day25RewardAdapterError, day24  # noqa: E402


REQUIRED_METADATA = (
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
)


def _policy_version(samples: Sequence[Any]) -> str:
    versions = [
        str(sample.weight_versions[-1]) if getattr(sample, "weight_versions", None) else None
        for sample in samples
    ]
    if any(version is not None for version in versions) and any(version is None for version in versions):
        raise Day25RewardAdapterError("one reward group mixes reported and missing weight versions")
    observed = {version for version in versions if version is not None}
    if len(observed) > 1:
        raise Day25RewardAdapterError(f"one reward group contains multiple weight versions: {sorted(observed)}")
    if observed:
        return str(observed.pop())
    value = os.environ.get("DAY26_POLICY_VERSION")
    if not value:
        raise Day25RewardAdapterError("DAY26_POLICY_VERSION is required when SGLang did not report weight_version")
    return value


def _columns(samples: Sequence[Any], run_id: str) -> dict[str, list[Any]]:
    columns = {name: [] for name in REQUIRED_METADATA}
    columns.update({"request_id": [], "finish_reason": [], "response_token_ids": []})
    for ordinal, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            raise Day25RewardAdapterError("slime Sample.metadata must be a mapping")
        for name in REQUIRED_METADATA:
            if name not in metadata:
                raise Day25RewardAdapterError(f"slime Sample.metadata is missing {name}")
            columns[name].append(metadata[name])
        response_length = getattr(sample, "response_length", None)
        tokens = list(getattr(sample, "tokens", []))
        if isinstance(response_length, bool) or not isinstance(response_length, int) or response_length < 0:
            raise Day25RewardAdapterError("slime Sample.response_length is invalid")
        if response_length > len(tokens):
            raise Day25RewardAdapterError("response_length exceeds Sample.tokens")
        index = getattr(sample, "index", ordinal)
        status = getattr(getattr(sample, "status", None), "value", "unknown")
        finish_reason = {"completed": "stop", "truncated": "length", "aborted": "abort"}.get(
            str(status), str(status)
        )
        columns["request_id"].append(f"{run_id}:sample:{index}")
        columns["finish_reason"].append(finish_reason)
        columns["response_token_ids"].append(tokens[-response_length:] if response_length else [])
    return columns


async def reward_func(args: Any, samples: list[Any], **_: Any) -> list[float]:
    """Score one complete G=4 group; use with --group-rm only."""

    if not isinstance(samples, list) or len(samples) != 4:
        raise Day25RewardAdapterError("Day 26 slime reward requires --group-rm and exactly one G=4 group")
    if len({getattr(sample, "group_index", None) for sample in samples}) != 1:
        raise Day25RewardAdapterError("reward batch mixes slime group_index values")
    completions = [getattr(sample, "response", None) for sample in samples]
    if any(not isinstance(value, str) for value in completions):
        raise Day25RewardAdapterError("slime Sample.response must be text")

    run_id = os.environ.get("DAY26_RUN_ID")
    ledger = os.environ.get("DAY26_TRAJECTORY_LEDGER")
    if not run_id or not ledger:
        raise Day25RewardAdapterError("DAY26_RUN_ID and DAY26_TRAJECTORY_LEDGER are required")
    adapter = Day25RewardAdapter(
        executor=day24.execute_e2b_request,
        run_id=run_id,
        policy_version=_policy_version(samples),
        ledger_path=Path(ledger),
        group_size=4,
        workers=int(os.environ.get("DAY26_SANDBOX_WORKERS", "4")),
    )
    return await asyncio.to_thread(adapter.score_batch, completions, **_columns(samples, run_id))
