#!/usr/bin/env python3
"""Contract tests for the frozen Day 25 code-eval promotion decision."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import build_day25_promotion_eval as builder
import day25_contract as contract
import generate_day25_promotion_eval as generator
import sandbox_day25_promotion_eval as sandbox
import score_day25_promotion_eval as scorer
from day25_reward_adapter import day24


def fake_execution(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "compile": {"status": "pass", "error_type": None},
        "test_cases": [
            {
                "test_id": item["test_id"],
                "family": item["family"],
                "status": "pass",
                "error_type": None,
                "stdout_sha256": day24.text_sha256(""),
                "stderr_sha256": day24.text_sha256(""),
                "duration_ms": 1.0,
            }
            for item in request["test_cases"]
        ],
        "sandbox": day24.SANDBOX_CONTRACT,
        "sandbox_digest": day24.object_sha256(day24.SANDBOX_CONTRACT),
        "duration_ms": float(len(request["test_cases"])),
    }


def result_package(
    value: Mapping[str, Any],
    suite_id: str,
    role: str,
    correct_indices: set[int],
) -> dict[str, Any]:
    suite = value["suites"][suite_id]
    rows = contract.load_jsonl(contract.BOOTCAMP_ROOT / suite["path"])
    records = []
    for index, task in enumerate(rows):
        total = len(task["reward_tests"])
        correct = index in correct_indices
        row = {
            "schema_name": "day25.coding_eval_result",
            "schema_version": 1,
            "task_family_id": task["task_family_id"],
            "task_row_sha256": task["row_sha256"],
            "status": "ok" if correct else "wrong_answer",
            "finish_reason": "stop",
            "format_valid": True,
            "passed_tests": total if correct else max(0, total - 1),
            "total_tests": total,
            "completion_token_count": 20,
            "completion_sha256": contract.text_sha256(f"completion:{role}:{index}"),
            "evidence_sha256": contract.text_sha256(f"evidence:{role}:{index}"),
        }
        row["result_sha256"] = contract.object_sha256(row)
        records.append(row)
    baseline = value["checkpoint_policy"]["baseline"]
    if role == "s1_parent":
        identity = {
            "checkpoint_id": baseline["checkpoint_id"],
            "downstream_key": baseline["downstream_key"],
            "artifact_sha256": baseline["artifact_sha256"],
        }
    else:
        identity = {
            "parent_downstream_key": baseline["downstream_key"],
            "stage": "g4_bounded_short_run",
            "checkpoint_step": 10,
            "cpu_contract_sha256": value["cpu_contract_sha256"],
            "binding_sha256": "2" * 64,
            "artifact_sha256": "3" * 64,
        }
    package = {
        "schema_name": "day25.coding_eval_results",
        "schema_version": 1,
        "eval_contract_sha256": value["eval_contract_sha256"],
        "suite_id": suite_id,
        "model_role": role,
        "model_identity": identity,
        "generation_contract_sha256": value["generation_contract"]["generation_contract_sha256"],
        "records": records,
    }
    package["results_sha256"] = contract.object_sha256(package)
    return package


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(contract.json_bytes(value))


def completion_package(value: Mapping[str, Any], suite_id: str) -> dict[str, Any]:
    rows = contract.load_jsonl(contract.BOOTCAMP_ROOT / value["suites"][suite_id]["path"])
    records = []
    for ordinal, task in enumerate(rows):
        response_ids = [257, 1000 + ordinal]
        row = {
            "schema_name": "day25.coding_eval_completion",
            "schema_version": 1,
            "ordinal": ordinal,
            "task_family_id": task["task_family_id"],
            "task_row_sha256": task["row_sha256"],
            "message_content": "    return 1",
            "message_content_sha256": contract.text_sha256("    return 1"),
            "generated_text": "    return 1",
            "generated_text_sha256": contract.text_sha256("    return 1"),
            "prompt_token_ids": [1, ordinal + 2],
            "prompt_token_ids_sha256": contract.object_sha256([1, ordinal + 2]),
            "response_token_ids": response_ids,
            "response_token_ids_sha256": contract.object_sha256(response_ids),
            "finish_reason": "stop",
        }
        row["completion_sha256"] = contract.object_sha256(row)
        records.append(row)
    baseline = value["checkpoint_policy"]["baseline"]
    package = {
        "schema_name": "day25.coding_eval_completions",
        "schema_version": 1,
        "eval_contract_sha256": value["eval_contract_sha256"],
        "suite_id": suite_id,
        "model_role": "s1_parent",
        "model_identity": {
            "checkpoint_id": baseline["checkpoint_id"],
            "downstream_key": baseline["downstream_key"],
            "artifact_sha256": baseline["artifact_sha256"],
        },
        "generation_contract_sha256": value["generation_contract"]["generation_contract_sha256"],
        "gpu_binding_sha256": "2" * 64,
        "records": records,
    }
    package["completions_sha256"] = contract.object_sha256(package)
    return package


class PromotionEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = contract.load_json(builder.DEFAULT_OUTPUT)
        contract.verify_seal(cls.value, "eval_contract_sha256", "eval contract")

    def decide_fixture(
        self,
        suite_id: str,
        baseline_correct: set[int],
        candidate_correct: set[int],
        *,
        search_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="day25-promotion-test-") as temporary:
            root = Path(temporary)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            write_json(
                baseline_path,
                result_package(self.value, suite_id, "s1_parent", baseline_correct),
            )
            write_json(
                candidate_path,
                result_package(self.value, suite_id, "grpo_g4_final", candidate_correct),
            )
            search_path = None
            if search_decision is not None:
                search_path = root / "search-decision.json"
                write_json(search_path, search_decision)
            return scorer.decide(
                eval_contract_path=builder.DEFAULT_OUTPUT,
                suite_id=suite_id,
                baseline_results_path=baseline_path,
                candidate_results_path=candidate_path,
                search_decision_path=search_path,
            )

    def test_confirmation24_is_deterministic_and_disjoint(self) -> None:
        confirmation, value = builder.build()
        train_ids = {
            row["task_family_id"] for row in contract.load_jsonl(builder.DEFAULT_TRAIN)
        }
        search_ids = {
            row["task_family_id"] for row in contract.load_jsonl(builder.DEFAULT_SEARCH)
        }
        confirmation_ids = {row["task_family_id"] for row in confirmation}
        self.assertEqual(len(confirmation_ids), 24)
        self.assertFalse(confirmation_ids & train_ids)
        self.assertFalse(confirmation_ids & search_ids)
        self.assertEqual(
            value["suites"]["confirmation24"]["ordered_family_ids"],
            [row["task_family_id"] for row in confirmation],
        )
        self.assertEqual(
            sum(len(items) for items in value["suites"]["confirmation24"]["quarantined_unused_family_ids"].values()),
            2,
        )

    def test_generation_manifest_matches_binding_export_exclusion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day25-model-manifest-test-") as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "S1-EXPORT-MANIFEST.json").write_text("{}", encoding="utf-8")
            files = generator._directory_manifest(
                root, exclude_names=("S1-EXPORT-MANIFEST.json",)
            )
            self.assertEqual(set(files), {"config.json", "model.safetensors"})

    def test_search_requires_three_net_new_correct_tasks(self) -> None:
        baseline = set(range(10))
        passing = self.decide_fixture("search40", baseline, baseline | {10, 11, 12})
        failing = self.decide_fixture("search40", baseline, baseline | {10, 11})
        self.assertTrue(passing["gate_pass"])
        self.assertTrue(passing["confirmation_authorized"])
        self.assertFalse(failing["gate_pass"])
        self.assertEqual(failing["status"], "closed_no_candidate_confirmation_unopened")

    def test_confirmation_is_single_use_gated_by_search(self) -> None:
        baseline = set(range(6))
        with self.assertRaisesRegex(scorer.Day25PromotionScoreError, "sealed search"):
            self.decide_fixture("confirmation24", baseline, baseline | {6, 7})
        search = self.decide_fixture(
            "search40", set(range(10)), set(range(13))
        )
        decision = self.decide_fixture(
            "confirmation24", baseline, baseline | {6, 7}, search_decision=search
        )
        self.assertTrue(decision["gate_pass"])
        self.assertTrue(decision["directional_code_candidate_eligible"])
        self.assertFalse(decision["strong_statistical_gain_claim_allowed"])
        self.assertFalse(decision["broader_s2_promotion_allowed"])

    def test_result_tampering_fails_closed(self) -> None:
        package = result_package(self.value, "search40", "s1_parent", set(range(10)))
        package["records"][0]["passed_tests"] = 0
        with tempfile.TemporaryDirectory(prefix="day25-tamper-test-") as temporary:
            root = Path(temporary)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            write_json(baseline_path, package)
            write_json(
                candidate_path,
                result_package(self.value, "search40", "grpo_g4_final", set(range(13))),
            )
            with self.assertRaises(contract.Day25ContractError):
                scorer.decide(
                    eval_contract_path=builder.DEFAULT_OUTPUT,
                    suite_id="search40",
                    baseline_results_path=baseline_path,
                    candidate_results_path=candidate_path,
                )

    def test_completion_package_reuses_day24_verifier_and_emits_results(self) -> None:
        package = completion_package(self.value, "confirmation24")
        package_hash = package["completions_sha256"]
        tasks = contract.load_jsonl(builder.DEFAULT_CONFIRMATION)
        evidence, results = sandbox.score_completions(
            package=package,
            package_hash=package_hash,
            completion_rows=package["records"],
            tasks=tasks,
            executor=fake_execution,
            workers=3,
        )
        self.assertEqual(evidence["status"], "complete")
        self.assertIsNotNone(results)
        assert results is not None
        self.assertEqual(len(results["records"]), 24)
        self.assertTrue(all(row["status"] == "ok" for row in results["records"]))
        contract.verify_seal(results, "results_sha256", "results")


if __name__ == "__main__":
    unittest.main()
