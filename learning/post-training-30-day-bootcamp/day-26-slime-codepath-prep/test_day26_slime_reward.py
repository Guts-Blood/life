#!/usr/bin/env python3
"""CPU tests for the Day 26 slime reward boundary."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import day26_slime_reward as reward


TRAIN_DATA = reward.BOOTCAMP_ROOT / "artifacts/data/day25-qwen35-coding-grpo-train.jsonl"


def fake_execution(request):
    program = str(request["program"])
    status = "wrong_answer" if "return None" in program else "pass"
    cases = [
        {
            "test_id": case["test_id"],
            "family": case["family"],
            "status": status,
            "error_type": None if status == "pass" else "AssertionError",
            "stdout_sha256": reward.day24.text_sha256(""),
            "stderr_sha256": reward.day24.text_sha256(""),
            "duration_ms": 1.0,
        }
        for case in request["test_cases"]
    ]
    return {
        "status": "ok",
        "compile": {"status": "pass", "error_type": None},
        "test_cases": cases,
        "sandbox": reward.day24.SANDBOX_CONTRACT,
        "sandbox_digest": reward.day24.object_sha256(reward.day24.SANDBOX_CONTRACT),
        "duration_ms": float(len(cases)),
    }


def samples():
    row = json.loads(TRAIN_DATA.read_text(encoding="utf-8").splitlines()[0])
    metadata = {name: row[name] for name in reward.REQUIRED_METADATA if name != "prompt_id"}
    metadata["prompt_id"] = f"day26:prompt:{row['task_family_id']}"
    completions = ["    return 1", "    return None", "    return 2", "    return 1"]
    return [
        SimpleNamespace(
            group_index=7,
            index=index,
            response=value,
            response_length=1,
            tokens=[10, 100 + index],
            weight_versions=[],
            status=SimpleNamespace(value="completed"),
            metadata=metadata,
        )
        for index, value in enumerate(completions)
    ]


class Day26SlimeRewardTests(unittest.TestCase):
    def test_group_reward_is_ordered_and_persisted(self):
        with tempfile.TemporaryDirectory(prefix="day26-reward-") as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            env = {
                "DAY26_RUN_ID": "day26-unit",
                "DAY26_POLICY_VERSION": "slime-v000000",
                "DAY26_TRAJECTORY_LEDGER": str(ledger),
                "DAY26_SANDBOX_WORKERS": "2",
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                reward.day24, "execute_e2b_request", fake_execution
            ):
                values = asyncio.run(reward.reward_func(SimpleNamespace(), samples()))
            self.assertEqual(values, [1.0, 0.0, 1.0, 1.0])
            batch = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(batch["status"], "complete")
            self.assertEqual(batch["group_size"], 4)
            self.assertEqual(len(batch["records"]), 4)

    def test_non_group_call_fails_closed(self):
        with self.assertRaisesRegex(Exception, "--group-rm"):
            asyncio.run(reward.reward_func(SimpleNamespace(), samples()[:1]))

    def test_mixed_group_indices_fail_closed(self):
        batch = samples()
        batch[-1].group_index = 8
        with self.assertRaisesRegex(Exception, "group_index"):
            asyncio.run(reward.reward_func(SimpleNamespace(), batch))

    def test_partial_weight_version_evidence_fails_closed(self):
        batch = samples()
        batch[0].weight_versions = ["1"]
        with tempfile.TemporaryDirectory(prefix="day26-reward-") as temporary:
            env = {
                "DAY26_RUN_ID": "day26-unit",
                "DAY26_TRAJECTORY_LEDGER": str(Path(temporary) / "ledger.jsonl"),
            }
            with patch.dict(os.environ, env, clear=False), self.assertRaisesRegex(
                Exception, "reported and missing"
            ):
                asyncio.run(reward.reward_func(SimpleNamespace(), batch))


if __name__ == "__main__":
    unittest.main()
