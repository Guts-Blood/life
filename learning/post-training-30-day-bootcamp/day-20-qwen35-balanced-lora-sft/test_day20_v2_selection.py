#!/usr/bin/env python3
"""Offline tests for the Day 20 v2 complete-evidence probe selector."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import day20_candidate_factory_v2 as factory
import day20_eval_identity_v2 as identity
import select_day20_v2 as selection


BASE_METRICS = {
    "general": 0,
    "math": 7,
    "finance": 3,
    "code": 7,
    "total": 17,
    "code_sandbox_execution_eligible": 8,
    "infrastructure_failures": 0,
}
STRICT_METRICS = {
    "general": 2,
    "math": 7,
    "finance": 3,
    "code": 7,
    "total": 19,
    "code_sandbox_execution_eligible": 8,
    "infrastructure_failures": 0,
}
NEAR_METRICS = {
    "general": 5,
    "math": 5,
    "finance": 2,
    "code": 6,
    "total": 18,
    "code_sandbox_execution_eligible": 6,
    "infrastructure_failures": 0,
}
GROSS_METRICS = {
    "general": 0,
    "math": 3,
    "finance": 1,
    "code": 2,
    "total": 6,
    "code_sandbox_execution_eligible": 4,
    "infrastructure_failures": 0,
}


def comparison_context() -> dict[str, Any]:
    return {
        "contract_version": factory.CONTRACT_VERSION,
        "scope": "probe32",
        "records": 32,
        "diagnostic_selection_sha256": "a" * 64,
        "eval_manifest_file_sha256": "b" * 64,
        "experiment_manifest_file_sha256": "c" * 64,
        "experiment_manifest_content_sha256": "d" * 64,
        "scorer_version": "frozen-scorer-v2",
        "scorer_file_sha256": "e" * 64,
        "response_adapter_version": identity.RESPONSE_ADAPTER_VERSION,
        "response_adapter_file_sha256": "f" * 64,
        "code_parser_version": "strict-code-v2",
        "code_parser_file_sha256": "1" * 64,
        "code_composer_version": "strict-compose-v2",
        "code_composer_file_sha256": "2" * 64,
        "evaluator_version": "day20-evaluator-v2",
        "evaluator_file_sha256": "3" * 64,
        "rescorer_version": "day20-rescorer-v3",
        "rescorer_file_sha256": "4" * 64,
    }


def e2b_context() -> dict[str, Any]:
    normalized = comparison_context()
    code_ids = [f"eval-code-{index}" for index in range(8)]
    runtime = {
        "sandbox_python": "/fixture/e2b/bin/python",
        "e2b_sdk": {"package": "e2b", "version": "2.37.0"},
        "preflight_file_sha256": "5" * 64,
        "preflight_content_sha256": "6" * 64,
    }
    return {
        "normalized_context": normalized,
        "normalized_comparison_key": identity.normalized_comparison_key(normalized),
        "code_sample_ids": code_ids,
        "code_sample_order_sha256": identity.object_sha256(code_ids),
        "sandbox_contract_file_sha256": "7" * 64,
        "sandbox_contract_content_sha256": "8" * 64,
        "sandbox_contract_hash": "9" * 64,
        "e2b_scorer_version": identity.E2B_SCORER_VERSION,
        "e2b_scorer_file_sha256": "a" * 64,
        "humaneval_source_file_sha256": "b" * 64,
        "e2b_runtime_identity": runtime,
        "e2b_runtime_identity_sha256": identity.object_sha256(runtime),
    }


def rows_for_metrics(metrics: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    e2b_rows: list[dict[str, Any]] = []
    counts = {skill: 0 for skill in factory.SKILLS}
    code_eligible = 0
    for index in range(8):
        for skill in factory.SKILLS:
            sample_id = f"eval-{skill}-{index}"
            row: dict[str, Any] = {
                "ordinal": len(rows) + 1,
                "sample_id": sample_id,
                "slice": skill,
            }
            if skill == "code":
                eligible = code_eligible < metrics["code_sandbox_execution_eligible"]
                row["sandbox_execution_eligible"] = eligible
                if eligible:
                    score = 1.0 if counts["code"] < metrics["code"] else 0.0
                    if score == 1.0:
                        counts["code"] += 1
                    e2b_rows.append({"sample_id": sample_id, "score": score})
                    code_eligible += 1
            else:
                score = 1.0 if counts[skill] < metrics[skill] else 0.0
                if score == 1.0:
                    counts[skill] += 1
                row["normalized_scorer_result"] = {"score": score}
            rows.append(row)
    return rows, e2b_rows


class EvidenceRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.values: dict[Path, tuple[Any, ...]] = {}
        self.normalized_context = comparison_context()
        self.e2b_context = e2b_context()
        self.normalized_key = identity.normalized_comparison_key(
            self.normalized_context
        )
        self.e2b_key = identity.e2b_comparison_key(self.e2b_context)
        self.complete_key = identity.complete_comparison_key(self.e2b_context)

    def add(self, candidate: str, metrics: dict[str, int]) -> Path:
        directory = self.root / candidate
        directory.mkdir()
        predictions = directory / f"{candidate}.qwen35-v3.predictions.jsonl"
        normalized_summary_path = directory / f"{candidate}.qwen35-v3.json"
        results_path = directory / f"{candidate}-code-e2b-qwen35-v3.jsonl"
        e2b_summary_path = directory / f"{candidate}-code-e2b-qwen35-v3-summary.json"
        rows, e2b_rows = rows_for_metrics(metrics)
        predictions.write_text("normalized\n", encoding="utf-8")
        normalized_summary_path.write_text("{}\n", encoding="utf-8")
        results_path.write_text("results\n", encoding="utf-8")
        normalized_summary = {
            "candidate": candidate,
            "scope": "probe32",
            "comparison_context": copy.deepcopy(self.normalized_context),
            "normalized_comparison_key": self.normalized_key,
        }
        e2b_summary = {
            "candidate": candidate,
            "scope": "probe32",
            "infrastructure_failures": metrics["infrastructure_failures"],
            "normalized_predictions": {"path": str(predictions.resolve())},
            "normalized_summary": {"path": str(normalized_summary_path.resolve())},
            "result": {"path": str(results_path.resolve())},
            "e2b_comparison_context": copy.deepcopy(self.e2b_context),
            "e2b_comparison_key": self.e2b_key,
            "complete_comparison_key": self.complete_key,
        }
        e2b_summary_path.write_text(
            json.dumps(e2b_summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.values[e2b_summary_path.resolve()] = (
            e2b_rows,
            e2b_summary,
            rows,
            normalized_summary,
        )
        return e2b_summary_path

    def verifier(
        self,
        summary_path: Path,
        *,
        pair_verifier: Any,
        expected_candidate: str | None = None,
        expected_scope: str | None = None,
    ) -> tuple[Any, ...]:
        value = self.values[summary_path.resolve()]
        if expected_candidate is not None and value[1]["candidate"] != expected_candidate:
            raise ValueError("candidate drifted")
        if expected_scope is not None and value[1]["scope"] != expected_scope:
            raise ValueError("scope drifted")
        return copy.deepcopy(value)


def unused_pair_verifier(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    raise AssertionError("selection should delegate normalized verification to E2B")


class Day20V2SelectionTests(unittest.TestCase):
    def build_cohort(
        self,
        root: Path,
        *,
        bracket_completed: bool,
        selected_metrics: dict[str, int] | None,
        selected_candidate: str | None = None,
        default_metrics: dict[str, int] = GROSS_METRICS,
    ) -> tuple[EvidenceRegistry, Path, list[Path], str | None]:
        registry = EvidenceRegistry(root)
        base = registry.add("base-probe", BASE_METRICS)
        expected = selection.expected_probe_cohort(
            bracket_completed=bracket_completed
        )
        target = selected_candidate or (expected[1] if selected_metrics else None)
        paths = [
            registry.add(
                candidate,
                selected_metrics
                if selected_metrics is not None and candidate == target
                else default_metrics,
            )
            for candidate in expected
        ]
        return registry, base, paths, target

    def test_stage_a_strict_pass_selects_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry, base, candidates, target = self.build_cohort(
                Path(temporary),
                bracket_completed=False,
                selected_metrics=STRICT_METRICS,
            )
            result = selection.select_probe_cohort(
                base_e2b_summary=base,
                candidate_e2b_summaries=list(reversed(candidates)),
                bracket_completed=False,
                pair_verifier=unused_pair_verifier,
                e2b_verifier=registry.verifier,
            )
            self.assertEqual(result["status"], "run_main")
            self.assertEqual(result["selected_candidate"], target)
            self.assertEqual(
                result["selection_sha256"],
                identity.object_sha256(
                    {
                        key: value
                        for key, value in result.items()
                        if key != "selection_sha256"
                    }
                ),
            )

    def test_near_miss_requests_bracket_but_gross_fail_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry, base, candidates, _ = self.build_cohort(
                Path(temporary),
                bracket_completed=False,
                selected_metrics=None,
                default_metrics=NEAR_METRICS,
            )
            near = selection.select_probe_cohort(
                base_e2b_summary=base,
                candidate_e2b_summaries=candidates,
                bracket_completed=False,
                pair_verifier=unused_pair_verifier,
                e2b_verifier=registry.verifier,
            )
            self.assertEqual(near["status"], "run_bracket")

        with tempfile.TemporaryDirectory() as temporary:
            registry, base, candidates, _ = self.build_cohort(
                Path(temporary),
                bracket_completed=False,
                selected_metrics=None,
            )
            gross = selection.select_probe_cohort(
                base_e2b_summary=base,
                candidate_e2b_summaries=candidates,
                bracket_completed=False,
                pair_verifier=unused_pair_verifier,
                e2b_verifier=registry.verifier,
            )
            self.assertEqual(gross["status"], "stop_no_passing_probe")

    def test_completed_bracket_requires_exact_cohort_and_can_select_bracket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expected = selection.expected_probe_cohort(bracket_completed=True)
            target = next(candidate for candidate in expected if "lr6e-5" in candidate)
            registry, base, candidates, _ = self.build_cohort(
                Path(temporary),
                bracket_completed=True,
                selected_metrics=STRICT_METRICS,
                selected_candidate=target,
            )
            result = selection.select_probe_cohort(
                base_e2b_summary=base,
                candidate_e2b_summaries=candidates,
                bracket_completed=True,
                pair_verifier=unused_pair_verifier,
                e2b_verifier=registry.verifier,
            )
            self.assertEqual(result["selected_candidate"], target)
            with self.assertRaisesRegex(
                selection.Day20SelectionV2Error, "cohort is incomplete"
            ):
                selection.select_probe_cohort(
                    base_e2b_summary=base,
                    candidate_e2b_summaries=candidates[:-1],
                    bracket_completed=True,
                    pair_verifier=unused_pair_verifier,
                    e2b_verifier=registry.verifier,
                )

    def test_common_key_drift_and_existing_selection_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, base, candidates, _ = self.build_cohort(
                root,
                bracket_completed=False,
                selected_metrics=STRICT_METRICS,
            )
            drifted = registry.values[candidates[-1].resolve()]
            drifted[3]["normalized_comparison_key"] = "0" * 64
            with self.assertRaisesRegex(
                selection.Day20SelectionV2Error, "common comparison drifted"
            ):
                selection.select_probe_cohort(
                    base_e2b_summary=base,
                    candidate_e2b_summaries=candidates,
                    bracket_completed=False,
                    pair_verifier=unused_pair_verifier,
                    e2b_verifier=registry.verifier,
                )
            drifted[3]["normalized_comparison_key"] = registry.normalized_key

            output = root / "PROBE-SELECTION-V2.json"
            value = selection.write_or_verify_selection(
                output,
                base_e2b_summary=base,
                candidate_e2b_summaries=candidates,
                bracket_completed=False,
                pair_verifier=unused_pair_verifier,
                e2b_verifier=registry.verifier,
            )
            self.assertEqual(
                selection.verify_selection(
                    output,
                    pair_verifier=unused_pair_verifier,
                    e2b_verifier=registry.verifier,
                ),
                value,
            )
            tampered = json.loads(output.read_text(encoding="utf-8"))
            tampered["selected_candidate"] = None
            tampered["selection_sha256"] = identity.object_sha256(
                {
                    key: item
                    for key, item in tampered.items()
                    if key != "selection_sha256"
                }
            )
            output.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                selection.Day20SelectionV2Error, "recomputed evidence"
            ):
                selection.verify_selection(
                    output,
                    pair_verifier=unused_pair_verifier,
                    e2b_verifier=registry.verifier,
                )

            malformed = copy.deepcopy(value)
            malformed["base"] = {}
            malformed["selection_sha256"] = identity.object_sha256(
                {
                    key: item
                    for key, item in malformed.items()
                    if key != "selection_sha256"
                }
            )
            output.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(
                selection.Day20SelectionV2Error, "artifact identities"
            ):
                selection.verify_selection(
                    output,
                    pair_verifier=unused_pair_verifier,
                    e2b_verifier=registry.verifier,
                )


if __name__ == "__main__":
    unittest.main()
