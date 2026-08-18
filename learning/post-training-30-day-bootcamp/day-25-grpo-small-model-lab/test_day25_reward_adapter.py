#!/usr/bin/env python3
"""CPU contract tests for the Day 25 GRPO reward boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import day25_contract as contract
from audit_day25_rewards import audit_path
from day25_reward_adapter import (
    Day25RewardAdapter,
    Day25RewardAdapterError,
    Day25RolloutOnlyStop,
    NON_THINKING_PREFIX,
    day24,
)


TRAIN_PATH = contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-train.jsonl"


def fake_execution(request: Mapping[str, Any]) -> dict[str, Any]:
    program = str(request["program"])
    statuses = ["pass"] * len(request["test_cases"])
    if "return 2" in program and statuses:
        statuses[-1] = "wrong_answer"
    if "return None" in program:
        statuses = ["wrong_answer"] * len(statuses)
    cases = []
    for case, status in zip(request["test_cases"], statuses):
        cases.append(
            {
                "test_id": case["test_id"],
                "family": case["family"],
                "status": status,
                "error_type": None if status == "pass" else "AssertionError",
                "stdout_sha256": day24.text_sha256(""),
                "stderr_sha256": day24.text_sha256(""),
                "duration_ms": 1.0,
            }
        )
    return {
        "status": "ok",
        "compile": {"status": "pass", "error_type": None},
        "test_cases": cases,
        "sandbox": day24.SANDBOX_CONTRACT,
        "sandbox_digest": day24.object_sha256(day24.SANDBOX_CONTRACT),
        "duration_ms": float(len(cases)),
    }


def infra_execution(_: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "infra_error",
        "error_type": "sdk_timeout",
        "compile": {"status": "not_run"},
        "test_cases": [],
        "sandbox": day24.SANDBOX_CONTRACT,
        "sandbox_digest": day24.object_sha256(day24.SANDBOX_CONTRACT),
        "duration_ms": 1.0,
    }


def batch_fixture() -> tuple[list[str], dict[str, list[Any]]]:
    rows = contract.load_jsonl(TRAIN_PATH)[:2]
    expanded = [row for row in rows for _ in range(contract.GROUP_SIZE)]
    completions = [
        NON_THINKING_PREFIX + "    return 1",
        "    return 2",
        "    return None",
        "    return 1",
        "    return 1",
        "    return 2",
        "    return None",
        "    return 1",
    ]
    columns = {
        name: [row[name] for row in expanded]
        for name in (
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
        )
    }
    columns.update(
        {
            "prompt_id": [f"prompt-{index // 4}" for index in range(8)],
            "request_id": [f"request-{index}" for index in range(8)],
            "finish_reason": ["stop"] * 8,
            "response_token_ids": [[100 + index] for index in range(8)],
        }
    )
    return completions, columns


class Day25RewardAdapterTests(unittest.TestCase):
    def test_full_batch_is_ordered_persisted_and_offline_recomputable(self) -> None:
        completions, kwargs = batch_fixture()
        with tempfile.TemporaryDirectory(prefix="day25-reward-test-") as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            adapter = Day25RewardAdapter(
                executor=fake_execution,
                run_id="day25-unit",
                policy_version="v0",
                ledger_path=ledger,
                workers=2,
            )
            rewards = adapter.score_batch(completions, **kwargs)
            self.assertEqual(rewards, [1.0, 2 / 3, 0.0, 1.0, 1.0, 2 / 3, 0.0, 1.0])
            batch = json.loads(ledger.read_text(encoding="utf-8"))
            contract.verify_seal(batch, "batch_sha256", "batch")
            self.assertEqual(batch["status"], "complete")
            first = batch["records"][0]["trajectory"]
            self.assertEqual(first["code_artifact"]["completion"], "    return 1")
            self.assertEqual(
                first["code_artifact"]["boundary"],
                "known_non_thinking_prefix_removed",
            )
            audit = audit_path(ledger)
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["complete_groups"], 2)
            self.assertEqual(audit["complete_trajectories"], 8)

    def test_persistent_infra_retries_once_and_aborts_atomic_batch(self) -> None:
        completions, kwargs = batch_fixture()
        with tempfile.TemporaryDirectory(prefix="day25-infra-test-") as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            adapter = Day25RewardAdapter(
                executor=infra_execution,
                run_id="day25-unit",
                policy_version="v0",
                ledger_path=ledger,
                workers=2,
            )
            with self.assertRaisesRegex(Day25RewardAdapterError, "aborted"):
                adapter.score_batch(completions, **kwargs)
            batch = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(batch["status"], "aborted_persistent_infra")
            self.assertFalse(batch["atomic_scalar_returned"])
            self.assertTrue(all(len(record["attempts"]) == 2 for record in batch["records"]))
            audit = audit_path(ledger)
            self.assertEqual(audit["aborted_persistent_infra_batches"], 1)
            self.assertEqual(audit["complete_trajectories"], 0)

    def test_rollout_only_gate_persists_rewards_then_stops_before_scalars(self) -> None:
        completions, kwargs = batch_fixture()
        with tempfile.TemporaryDirectory(prefix="day25-rollout-only-test-") as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            adapter = Day25RewardAdapter(
                executor=fake_execution,
                run_id="day25-g1-unit",
                policy_version="v0",
                ledger_path=ledger,
                workers=2,
                abort_after_complete_batch=True,
            )
            with self.assertRaisesRegex(Day25RolloutOnlyStop, "intentional G1"):
                adapter.score_batch(completions, **kwargs)
            batch = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(batch["status"], "complete_rollout_only_stop")
            self.assertFalse(batch["atomic_scalar_returned"])
            audit = audit_path(ledger)
            self.assertEqual(audit["complete_rollout_only_stop_batches"], 1)
            self.assertEqual(audit["complete_groups"], 2)

    def test_trainer_step_is_part_of_the_effective_policy_version(self) -> None:
        completions, kwargs = batch_fixture()
        kwargs["trainer_state"] = type("TrainerState", (), {"global_step": 1})()
        with tempfile.TemporaryDirectory(prefix="day25-policy-version-test-") as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            adapter = Day25RewardAdapter(
                executor=fake_execution,
                run_id="day25-g3-unit",
                policy_version="g3",
                ledger_path=ledger,
                workers=2,
            )
            adapter.score_batch(completions, **kwargs)
            batch = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(batch["configured_policy_version"], "g3")
            self.assertEqual(batch["policy_version"], "g3:trainer-step-1")
            self.assertTrue(
                all(
                    record["trajectory"]["policy"]["policy_version"]
                    == "g3:trainer-step-1"
                    for record in batch["records"]
                )
            )

    def test_truncation_is_model_zero_not_infra(self) -> None:
        completions, kwargs = batch_fixture()
        kwargs["finish_reason"][0] = "length"
        adapter = Day25RewardAdapter(
            executor=fake_execution,
            run_id="day25-unit",
            policy_version="v0",
            ledger_path=None,
            workers=2,
        )
        rewards = adapter.score_batch(completions, **kwargs)
        self.assertEqual(rewards[0], 0.0)

    def test_reward_payload_tampering_fails_before_sandbox(self) -> None:
        completions, kwargs = batch_fixture()
        kwargs["reward_tests"][0] = ["assert False"]
        called = False

        def executor(_: Mapping[str, Any]) -> Mapping[str, Any]:
            nonlocal called
            called = True
            return fake_execution(_)

        adapter = Day25RewardAdapter(
            executor=executor,
            run_id="day25-unit",
            policy_version="v0",
            ledger_path=None,
        )
        with self.assertRaisesRegex(Day25RewardAdapterError, "payload hash"):
            adapter.score_batch(completions, **kwargs)
        self.assertFalse(called)

    def test_group_shape_must_be_exact(self) -> None:
        completions, kwargs = batch_fixture()
        for key in kwargs:
            kwargs[key] = kwargs[key][:-1]
        with self.assertRaisesRegex(Day25RewardAdapterError, "multiple of G=4"):
            Day25RewardAdapter(
                executor=fake_execution,
                run_id="day25-unit",
                policy_version="v0",
                ledger_path=None,
            ).score_batch(completions[:-1], **kwargs)


if __name__ == "__main__":
    unittest.main()
