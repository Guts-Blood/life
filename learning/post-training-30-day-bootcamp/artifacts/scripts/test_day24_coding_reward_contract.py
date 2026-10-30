#!/usr/bin/env python3
"""Contract tests for the Day 24 MBPP verifier and reward pipeline."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import unittest
from pathlib import Path
from typing import Any, Mapping


BOOTCAMP = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    BOOTCAMP
    / "day-24-online-rl-dataflow-reward/day24_coding_verifier.py"
)
SPEC = importlib.util.spec_from_file_location("day24_coding_verifier_for_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def frozen_rows() -> list[dict[str, Any]]:
    return verifier.freeze_cohort(verifier.load_jsonl(verifier.DEFAULT_ROLLOUTS))


def trajectory() -> dict[str, Any]:
    return copy.deepcopy(frozen_rows()[0])


def fake_execution(
    request: Mapping[str, Any],
    *,
    statuses: Mapping[str, str] | None = None,
    compile_status: str = "pass",
    infra: bool = False,
) -> dict[str, Any]:
    if infra:
        return {
            "status": "infra_error",
            "error_type": "sdk_timeout",
            "compile": {"status": "not_run"},
            "test_cases": [],
            "sandbox": verifier.SANDBOX_CONTRACT,
            "sandbox_digest": verifier.object_sha256(verifier.SANDBOX_CONTRACT),
            "duration_ms": 1.0,
        }
    if compile_status == "fail":
        return {
            "status": "ok",
            "compile": {"status": "fail", "error_type": "SyntaxError"},
            "test_cases": [],
            "sandbox": verifier.SANDBOX_CONTRACT,
            "sandbox_digest": verifier.object_sha256(verifier.SANDBOX_CONTRACT),
            "duration_ms": 1.0,
        }
    cases = []
    for case in request["test_cases"]:
        status = (statuses or {}).get(case["test_id"], "pass")
        cases.append(
            {
                "test_id": case["test_id"],
                "family": case["family"],
                "status": status,
                "error_type": None if status == "pass" else status,
                "stdout_sha256": verifier.text_sha256(""),
                "stderr_sha256": verifier.text_sha256(""),
                "duration_ms": 1.0,
            }
        )
    return {
        "status": "ok",
        "compile": {"status": "pass", "error_type": None},
        "test_cases": cases,
        "sandbox": verifier.SANDBOX_CONTRACT,
        "sandbox_digest": verifier.object_sha256(verifier.SANDBOX_CONTRACT),
        "duration_ms": float(len(cases)),
    }


def evidence_with_reward(
    correctness_passed: int,
    *,
    frozen_passed: int = 0,
    eligible: bool = True,
    style_score: float | None = 1.0,
) -> dict[str, Any]:
    row = {
        "schema_name": "day24.coding_verifier_evidence",
        "schema_version": 1,
        "event_id": "fixture",
        "started_at_utc": "2026-08-16T00:00:00+00:00",
        "replay_ordinal": 1,
        "trajectory_id": f"fixture-{correctness_passed}-{frozen_passed}-{eligible}",
        "group_id": "fixture-group",
        "completion_id": "fixture-completion",
        "deterministic_replay_key": "0" * 64,
        "verifier_contract_sha256": verifier.verifier_contract()["contract_sha256"],
        "format_score": 1.0,
        "style_score": style_score,
        "safety_score": 1.0,
        "safety_reasons": [],
        "status": "ok" if eligible else "infra_error",
        "eligible_for_reward": eligible,
        "parse": {"status": "pass"},
        "compile": {"status": "pass"},
        "test_cases": [],
        "summary": {
            "visible_example": {"passed": 0, "total": 0, "status_counts": {}},
            "reward": {
                "passed": correctness_passed,
                "total": 1 if eligible else 0,
                "status_counts": {},
            },
            "frozen_eval": {
                "passed": frozen_passed,
                "total": 1 if eligible else 0,
                "status_counts": {},
            },
        },
        "sandbox_digest": verifier.object_sha256(verifier.SANDBOX_CONTRACT),
        "sandbox": verifier.SANDBOX_CONTRACT,
        "duration_ms": 1.0,
    }
    row["semantic_result_sha256"] = verifier.object_sha256(
        verifier._semantic_evidence(row)
    )
    return verifier.seal(row, "evidence_sha256")


class CohortAndSchemaTests(unittest.TestCase):
    def test_cohort_is_two_train_groups_of_four_with_frozen_tests(self) -> None:
        rows = frozen_rows()
        self.assertEqual(len(rows), 8)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
            self.assertEqual(row["task"]["split"], "train")
            self.assertEqual(row["modality"], {"value": "text", "image_tokens": 0, "video_tokens": 0})
            self.assertTrue(row["tests"]["reward_tests"])
            self.assertTrue(row["tests"]["frozen_eval_tests"])
            self.assertFalse(row["source_parent"]["day23_failure_checkpoint_used"])
            verifier.verify_seal(row, "trajectory_sha256", "trajectory")
        self.assertEqual(sorted(counts.values()), [4, 4])

    def test_schema_contains_required_rl_objects_and_null_logprob_contract(self) -> None:
        schema = verifier.build_trajectory_schema()
        for name in (
            "deterministic_replay_key",
            "modality",
            "tokens",
            "processor",
            "policy",
            "logprobs",
            "tests",
        ):
            self.assertIn(name, schema["required"])
        row = trajectory()
        self.assertEqual(row["logprobs"]["availability"], "not_computed_day24_cpu_contract")
        self.assertIsNone(row["logprobs"]["old_logprob"])


class VerifierTests(unittest.TestCase):
    def test_parse_failure_does_not_call_executor(self) -> None:
        row = trajectory()
        row["code_artifact"]["completion"] = "    return ("
        row = verifier.seal(row, "trajectory_sha256")
        called = False

        def executor(_: Mapping[str, Any]) -> Mapping[str, Any]:
            nonlocal called
            called = True
            raise AssertionError("must not run")

        result = verifier.verify_trajectory(row, executor)
        self.assertEqual(result["status"], "parse_error")
        self.assertFalse(called)

    def test_compile_failure_is_distinct(self) -> None:
        result = verifier.verify_trajectory(
            trajectory(), lambda request: fake_execution(request, compile_status="fail")
        )
        self.assertEqual(result["status"], "compile_error")

    def test_partial_and_full_test_results(self) -> None:
        row = trajectory()
        failed_id = row["tests"]["reward_tests"][0]["test_id"]
        partial = verifier.verify_trajectory(
            row,
            lambda request: fake_execution(request, statuses={failed_id: "wrong_answer"}),
        )
        self.assertEqual(partial["status"], "wrong_answer")
        self.assertEqual(partial["summary"]["reward"]["passed"], 2)
        self.assertEqual(partial["summary"]["reward"]["total"], 3)
        full = verifier.verify_trajectory(row, lambda request: fake_execution(request))
        self.assertEqual(full["status"], "ok")
        self.assertEqual(full["summary"]["reward"]["passed"], 3)

    def test_format_valid_but_wrong_answer_and_visible_only_pass(self) -> None:
        row = trajectory()
        row["tests"]["visible_examples"] = [
            {
                "test_id": "mbpp:task:602:visible:00",
                "source": 'assert first_repeated_char("aa") == "a"',
            }
        ]
        row["tests"]["manifest_sha256"] = verifier.object_sha256(
            {
                key: value
                for key, value in row["tests"].items()
                if key != "manifest_sha256"
            }
        )
        row["deterministic_replay_key"] = verifier._replay_key(row)
        row = verifier.seal(row, "trajectory_sha256")
        statuses = {
            case["test_id"]: "wrong_answer"
            for case in row["tests"]["reward_tests"] + row["tests"]["frozen_eval_tests"]
        }
        result = verifier.verify_trajectory(
            row, lambda request: fake_execution(request, statuses=statuses)
        )
        self.assertEqual(result["format_score"], 1.0)
        self.assertEqual(result["summary"]["visible_example"]["passed"], 1)
        self.assertEqual(result["summary"]["visible_example"]["total"], 1)
        self.assertEqual(result["summary"]["reward"]["passed"], 0)
        self.assertEqual(verifier.score_verification(result, "tests_only")["aggregate_reward"], 0.0)

    def test_candidate_timeout_and_runtime_error_are_model_outcomes(self) -> None:
        row = trajectory()
        first = row["tests"]["reward_tests"][0]["test_id"]
        for status, expected in (
            ("candidate_timeout", "candidate_timeout"),
            ("runtime_error", "runtime_error"),
        ):
            with self.subTest(status=status):
                result = verifier.verify_trajectory(
                    row,
                    lambda request, status=status: fake_execution(
                        request, statuses={first: status}
                    ),
                )
                self.assertEqual(result["status"], expected)
                self.assertTrue(result["eligible_for_reward"])

    def test_infra_error_is_null_reward(self) -> None:
        result = verifier.verify_trajectory(
            trajectory(), lambda request: fake_execution(request, infra=True)
        )
        self.assertEqual(result["status"], "infra_error")
        self.assertFalse(result["eligible_for_reward"])
        reward = verifier.score_verification(result, "tests_only")
        self.assertIsNone(reward["aggregate_reward"])

    def test_repeated_execution_has_same_semantic_hash_but_distinct_event(self) -> None:
        row = trajectory()
        first = verifier.verify_trajectory(row, lambda request: fake_execution(request), replay_ordinal=1)
        second = verifier.verify_trajectory(row, lambda request: fake_execution(request), replay_ordinal=2)
        self.assertEqual(first["semantic_result_sha256"], second["semantic_result_sha256"])
        self.assertNotEqual(first["event_id"], second["event_id"])
        self.assertNotEqual(first["evidence_sha256"], second["evidence_sha256"])

    def test_truncated_completion_is_explicit(self) -> None:
        row = trajectory()
        row["code_artifact"]["finish_reason"] = "length"
        row = verifier.seal(row, "trajectory_sha256")
        result = verifier.verify_trajectory(row, lambda _: self.fail("executor called"))
        self.assertEqual(result["status"], "truncated")
        reward = verifier.score_verification(result, "tests_only")
        self.assertEqual(reward["aggregate_reward"], 0.0)

    def test_parse_and_compile_failures_score_zero(self) -> None:
        row = trajectory()
        row["code_artifact"]["completion"] = "    return ("
        row = verifier.seal(row, "trajectory_sha256")
        parse_result = verifier.verify_trajectory(
            row, lambda _: self.fail("executor called")
        )
        self.assertEqual(
            verifier.score_verification(parse_result, "tests_only")["aggregate_reward"],
            0.0,
        )
        compile_result = verifier.verify_trajectory(
            trajectory(), lambda request: fake_execution(request, compile_status="fail")
        )
        self.assertEqual(
            verifier.score_verification(compile_result, "tests_only")["aggregate_reward"],
            0.0,
        )


class RewardAndGroupTests(unittest.TestCase):
    def test_missing_reward_component_fails_closed(self) -> None:
        evidence = evidence_with_reward(1, style_score=None)
        with self.assertRaisesRegex(verifier.Day24VerifierError, "component"):
            verifier.score_verification(evidence, "tests_format_style")

    def test_policies_share_raw_evidence_and_have_distinct_hashes(self) -> None:
        evidence = evidence_with_reward(0)
        tests = verifier.score_verification(evidence, "tests_only")
        styled = verifier.score_verification(evidence, "tests_format_style")
        self.assertEqual(tests["semantic_result_sha256"], styled["semantic_result_sha256"])
        self.assertNotEqual(tests["policy"]["policy_sha256"], styled["policy"]["policy_sha256"])
        self.assertEqual(tests["aggregate_reward"], 0.0)
        self.assertEqual(styled["aggregate_reward"], 0.1)

    def test_group_advantage_matches_hand_calculation(self) -> None:
        evidences = [
            evidence_with_reward(value) for value in (1, 1, 0, 0)
        ]
        rewards = []
        for index, evidence in enumerate(evidences):
            evidence["trajectory_id"] = f"trajectory-{index}"
            evidence = verifier.seal(evidence, "evidence_sha256")
            rewards.append(verifier.score_verification(evidence, "tests_only"))
        group = verifier.reduce_group(rewards, "tests_only")
        self.assertEqual(group["mean"], 0.5)
        expected_std = math.sqrt(1.0 / 3.0)
        expected_advantage = 0.5 / (expected_std + 1e-4)
        self.assertAlmostEqual(group["std"], expected_std)
        self.assertEqual(group["ddof"], 1)
        self.assertEqual(group["epsilon"], 1e-4)
        for actual, expected in zip(
            [member["advantage"] for member in group["members"]],
            [expected_advantage, expected_advantage, -expected_advantage, -expected_advantage],
        ):
            self.assertAlmostEqual(actual, expected)

    def test_zero_variance_group_has_zero_advantage(self) -> None:
        rewards = []
        for index in range(4):
            evidence = evidence_with_reward(0)
            evidence["trajectory_id"] = f"zero-{index}"
            evidence = verifier.seal(evidence, "evidence_sha256")
            rewards.append(verifier.score_verification(evidence, "tests_only"))
        group = verifier.reduce_group(rewards, "tests_only")
        self.assertTrue(group["zero_variance"])
        self.assertFalse(group["optimizer_update_eligible"])
        self.assertEqual([member["advantage"] for member in group["members"]], [0.0] * 4)

    def test_all_infra_group_is_invalid(self) -> None:
        rewards = []
        for index in range(4):
            evidence = evidence_with_reward(0, eligible=False)
            evidence["trajectory_id"] = f"infra-{index}"
            evidence = verifier.seal(evidence, "evidence_sha256")
            rewards.append(verifier.score_verification(evidence, "tests_only"))
        group = verifier.reduce_group(rewards, "tests_only")
        self.assertEqual(group["status"], "invalid_infra")
        self.assertFalse(group["optimizer_update_eligible"])
        self.assertTrue(all(member["advantage"] is None for member in group["members"]))

    def test_reward_hacking_is_visible_to_frozen_correctness(self) -> None:
        evidence = evidence_with_reward(1, frozen_passed=0)
        reward = verifier.score_verification(evidence, "tests_only")
        self.assertEqual(reward["aggregate_reward"], 1.0)
        self.assertEqual(reward["components"]["frozen_eval_correctness"], 0.0)


class PipelineTests(unittest.TestCase):
    def test_fake_pipeline_has_two_replays_and_four_group_policy_rows(self) -> None:
        rows = frozen_rows()
        evidence, groups = verifier.run_pipeline(
            rows,
            executor=lambda request: fake_execution(request),
            replays=2,
            workers=2,
        )
        self.assertEqual(len(evidence), 16)
        self.assertEqual(len(groups), 4)
        self.assertEqual({group["group_size"] for group in groups}, {4})

    def test_pipeline_retries_infra_once_then_invalidates_groups(self) -> None:
        evidence, groups = verifier.run_pipeline(
            frozen_rows(),
            executor=lambda request: fake_execution(request, infra=True),
            replays=2,
            workers=2,
        )
        self.assertEqual(len(evidence), 32)
        self.assertEqual({row["infra_retry_ordinal"] for row in evidence}, {0, 1})
        self.assertEqual({group["status"] for group in groups}, {"invalid_infra"})
        self.assertTrue(all(not group["optimizer_update_eligible"] for group in groups))


class PersistedArtifactTests(unittest.TestCase):
    def test_persisted_pipeline_is_complete_self_hashed_and_replayable(self) -> None:
        cohort = verifier.load_jsonl(verifier.DEFAULT_INPUT)
        evidence = verifier.load_jsonl(verifier.DEFAULT_EVIDENCE)
        groups = verifier.load_jsonl(verifier.DEFAULT_REWARDS)
        self.assertEqual(len(cohort), 8)
        self.assertEqual(len(evidence), 16)
        self.assertEqual(len(groups), 4)
        for row in cohort:
            verifier.verify_seal(row, "trajectory_sha256", "trajectory")
            self.assertEqual(row["task"]["split"], "train")
        by_trajectory: dict[str, list[dict[str, Any]]] = {}
        for row in evidence:
            verifier.verify_seal(row, "evidence_sha256", "evidence")
            by_trajectory.setdefault(row["trajectory_id"], []).append(row)
            self.assertNotEqual(row["status"], "infra_error")
            self.assertFalse(row["sandbox"]["allow_internet_access"])
        self.assertEqual(set(by_trajectory), {row["trajectory_id"] for row in cohort})
        self.assertTrue(
            all(
                len(rows) == 2
                and len({row["semantic_result_sha256"] for row in rows}) == 1
                for rows in by_trajectory.values()
            )
        )
        for group in groups:
            verifier.verify_seal(group, "group_reward_sha256", "group reward")
            self.assertEqual(group["group_size"], 4)
            self.assertEqual(group["status"], "ok")

    def test_persisted_reward_hacking_case_is_revealed(self) -> None:
        evidence = verifier.load_jsonl(verifier.DEFAULT_EVIDENCE)
        rows = [
            row
            for row in evidence
            if row["trajectory_id"] == "day24:mbpp:task:602:sample:03"
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["summary"]["reward"]["passed"] == 3 for row in rows))
        self.assertTrue(all(row["summary"]["frozen_eval"]["passed"] == 1 for row in rows))


@unittest.skipUnless(
    os.environ.get("DAY24_RUN_E2B_INTEGRATION") == "1",
    "set DAY24_RUN_E2B_INTEGRATION=1 for the live E2B smoke",
)
class LiveE2BIntegrationTests(unittest.TestCase):
    def test_fresh_network_denied_e2b_per_test_verifier(self) -> None:
        request = {
            "program": "def add_one(x):\n    return x + 1\n",
            "setup_code": "",
            "test_cases": [
                {"test_id": "live-pass", "family": "reward", "source": "assert add_one(1) == 2"},
                {"test_id": "live-fail", "family": "frozen_eval", "source": "assert add_one(1) == 3"},
            ],
            "sandbox_contract": verifier.SANDBOX_CONTRACT,
        }
        result = verifier.execute_e2b_request(request)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual([case["status"] for case in result["test_cases"]], ["pass", "wrong_answer"])
        self.assertFalse(result["sandbox"]["allow_internet_access"])


if __name__ == "__main__":
    unittest.main()
