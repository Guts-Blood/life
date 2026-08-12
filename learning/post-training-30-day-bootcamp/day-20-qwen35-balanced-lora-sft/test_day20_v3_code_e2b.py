#!/usr/bin/env python3
"""Offline tests for strict-only Day 20 v3 E2B scoring artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import day20_candidate_factory_v2 as factory
import day20_eval_identity_v2 as identity
import score_day20_code_e2b_v3 as scorer
from day20_contract_v2 import V2_CONTRACT_VERSION, validate_raw_code_continuation


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def normalized_context() -> dict[str, Any]:
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


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root / "run"
        self.root.mkdir()
        self.predictions = self.root / "base-probe.qwen35-v3.predictions.jsonl"
        self.normalized_summary = self.root / "base-probe.qwen35-v3.json"
        self.output = self.root / "base-probe-code-e2b-qwen35-v3.jsonl"
        self.e2b_summary = self.root / "base-probe-code-e2b-qwen35-v3-summary.json"
        self.frozen_path = root / "frozen-e2b.py"
        self.frozen_path.write_text("# frozen fixture\n", encoding="utf-8")
        self.sandbox_config = root / "sandbox.json"
        self.source_path = root / "HumanEval.jsonl.gz"
        self.source_path.write_bytes(b"pinned HumanEval fixture")
        self.preflight_path = self.root / "evidence/E2B-PREFLIGHT.json"
        self.preflight_path.parent.mkdir()
        self.preflight_path.write_text("{}\n", encoding="utf-8")
        self.execution_calls: list[str] = []
        self.rows, self.summary, self.sources = self._normalized_pair()
        self.predictions.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.summary["predictions"] = {
            "path": str(self.predictions.resolve()),
            "file_sha256": scorer.file_sha256(self.predictions),
            "content_sha256": identity.object_sha256(self.rows),
            "records": 32,
            "ordered_sample_ids_sha256": identity.object_sha256(
                [row["sample_id"] for row in self.rows]
            ),
        }
        self.summary["summary_sha256"] = identity.object_sha256(self.summary)
        write_json(self.normalized_summary, self.summary)
        self.config = {
            "source": {
                "source": "openai/human-eval",
                "revision": "fixture-revision",
            }
        }
        write_json(self.sandbox_config, self.config)
        self.frozen = self._frozen_module()

    def _normalized_pair(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        sources: dict[str, dict[str, Any]] = {}
        code_index = 0
        for ordinal in range(1, 33):
            is_code = ordinal % 4 == 0
            sample_id = f"eval:{'code' if is_code else 'math'}:{ordinal:03d}"
            row: dict[str, Any] = {
                "schema_version": 2,
                "domain": "day20.v2.normalized_prediction",
                "ordinal": ordinal,
                "candidate": "base-probe",
                "scope": "probe32",
                "sample_id": sample_id,
                "slice": "code" if is_code else "math",
                "reference_hash": hashlib.sha256(
                    f"reference-{ordinal}".encode()
                ).hexdigest(),
                "normalized_scorer_result": {
                    "score_status": "sandbox_required" if is_code else "ok",
                    "score": None if is_code else 1.0,
                },
                "format_compliant": True,
                "anomalies": [],
            }
            if is_code:
                entry = f"solve_{code_index}"
                parent_id = f"HumanEval/{code_index}"
                raw_prompt = f'def {entry}(x):\n    """Return x."""'
                reference = "return x"
                source_test = f"def check(candidate):\n    assert candidate(1) == 1\n"
                sources[parent_id] = {
                    "prompt": raw_prompt,
                    "canonical_solution": reference,
                    "entry_point": entry,
                    "test": source_test,
                }
                eligible = code_index < 6
                candidate = "    return x" if eligible else "return x"
                if eligible:
                    evidence = validate_raw_code_continuation(candidate, raw_prompt)
                    code_candidate = {
                        "candidate_mode": "completion",
                        "candidate": candidate,
                        "execution_eligible": True,
                        "contract_version": V2_CONTRACT_VERSION,
                        "raw_sha256": evidence.raw_sha256,
                        "canonical_sha256": evidence.canonical_sha256,
                        "ast_sha256": evidence.ast_sha256,
                        "error": None,
                    }
                    static = {"valid": True, "error": None}
                else:
                    code_candidate = {
                        "candidate_mode": "completion",
                        "candidate": candidate,
                        "execution_eligible": False,
                        "contract_version": V2_CONTRACT_VERSION,
                        "raw_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                        "canonical_sha256": None,
                        "ast_sha256": None,
                        "error": "raw contract failed",
                    }
                    static = {"valid": False, "error": "raw contract failed"}
                row.update(
                    {
                        "raw_prompt": raw_prompt,
                        "reference": reference,
                        "source_lineage": {
                            "parent_id": parent_id,
                            "source": "openai/human-eval",
                            "revision": "fixture-revision",
                        },
                        "metadata": {
                            "entry_point": entry,
                            "test_sha256": identity.object_sha256(source_test),
                        },
                        "code_candidate": code_candidate,
                        "code_static_syntax": static,
                        "sandbox_execution_eligible": eligible,
                    }
                )
                code_index += 1
            row["row_sha256"] = identity.object_sha256(row)
            rows.append(row)
        comparison_context = normalized_context()
        summary: dict[str, Any] = {
            "schema_version": 2,
            "domain": "day20.v2.normalized_summary",
            "status": "complete_with_code_sandbox_required",
            "candidate": "base-probe",
            "scope": "probe32",
            "comparison_context": comparison_context,
            "normalized_comparison_key": identity.normalized_comparison_key(
                comparison_context
            ),
            "evaluation_run_sha256": "5" * 64,
            "checkpoint_package_sha256": "6" * 64,
            "model_identity_sha256": "7" * 64,
        }
        return rows, summary, sources

    def _frozen_module(self) -> types.SimpleNamespace:
        def execute_task(
            task: dict[str, Any], context: dict[str, Any], bindings: dict[str, Any]
        ) -> dict[str, Any]:
            self.execution_calls.append(task["sample_id"])
            score = 0.0 if task["sample_id"].endswith("024") else 1.0
            return {
                "sample_id": task["sample_id"],
                "execution_status": "passed" if score == 1.0 else "failed",
                "score_status": "ok",
                "score": score,
                "passed": score == 1.0,
                "error_type": None if score == 1.0 else "assertion_error",
                "exit_code": 0 if score == 1.0 else 1,
                "scorer_result": {
                    "score_status": "ok",
                    "score": score,
                    "passed": score == 1.0,
                },
                "sandbox": {"backend": "fixture", "execution_attempted": True},
            }

        return types.SimpleNamespace(
            verify_config=lambda config: None,
            semantic_hash=identity.object_sha256,
            load_humaneval_source=lambda path, config: (
                self.sources,
                scorer.file_sha256(path),
            ),
            object_sha256=identity.object_sha256,
            exact_text_hash=lambda value: hashlib.sha256(value.encode()).hexdigest(),
            execute_task=execute_task,
            load_e2b_bindings=lambda config: {"fixture": True},
        )

    def pair_verifier(
        self,
        predictions_path: Path,
        summary_path: Path,
        *,
        expected_scope: str | None = None,
        expected_candidate: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = scorer.load_jsonl(predictions_path)
        summary = scorer.load_json(summary_path)
        for row in rows:
            scorer._self_hash(row, "row_sha256")
        scorer._self_hash(summary, "summary_sha256")
        if expected_scope is not None and summary["scope"] != expected_scope:
            raise ValueError("scope drifted")
        if expected_candidate is not None and summary["candidate"] != expected_candidate:
            raise ValueError("candidate drifted")
        return rows, summary

    def preflight_verifier(self, path: Path, *, run_root: Path) -> dict[str, Any]:
        return {
            "sandbox_python": "/fixture/e2b/bin/python",
            "e2b_sdk": {"package": "e2b", "version": "2.37.0"},
            "preflight_sha256": "8" * 64,
            "frozen_scorer": {
                "path": str(self.frozen_path.resolve()),
                "file_sha256": scorer.file_sha256(self.frozen_path),
            },
            "sandbox_config": {
                "path": str(self.sandbox_config.resolve()),
                "file_sha256": scorer.file_sha256(self.sandbox_config),
            },
            "humaneval_source": {
                "path": str(self.source_path.resolve()),
                "file_sha256": scorer.file_sha256(self.source_path),
            },
        }

    def score(self) -> dict[str, Any]:
        return scorer.score_code_pair(
            predictions_path=self.predictions,
            normalized_summary_path=self.normalized_summary,
            frozen_scorer_path=self.frozen_path,
            sandbox_config_path=self.sandbox_config,
            humaneval_source_path=self.source_path,
            preflight_path=self.preflight_path,
            run_root=self.root,
            output_path=self.output,
            summary_output_path=self.e2b_summary,
            pair_verifier=self.pair_verifier,
            frozen_module=self.frozen,
            preflight_verifier=self.preflight_verifier,
            bindings={"fixture": True},
        )


class Day20CodeE2BV3Tests(unittest.TestCase):
    def test_only_strict_eligible_rows_are_executed_and_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            summary = fixture.score()
            self.assertEqual(len(fixture.execution_calls), 6)
            self.assertEqual(summary["records"], 6)
            self.assertEqual(summary["sandbox_execution_eligible"], 6)
            self.assertEqual(summary["code_records"], 8)
            results, verified, _, _ = scorer.verify_published_pair(
                fixture.e2b_summary, pair_verifier=fixture.pair_verifier
            )
            self.assertEqual(verified, summary)
            self.assertEqual(len(results), 6)
            self.assertTrue(all(row["strict_contract_eligible"] for row in results))
            self.assertTrue(
                all(row["domain"] == scorer.ROW_DOMAIN for row in results)
            )

    def test_existing_pair_is_fully_reverified_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            first = fixture.score()
            calls = list(fixture.execution_calls)
            second = fixture.score()
            self.assertEqual(first, second)
            self.assertEqual(fixture.execution_calls, calls)

            rows = scorer.load_jsonl(fixture.output)
            rows[0]["score"] = 0.0
            fixture.output.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaises(scorer.Day20CodeE2BV3Error):
                fixture.score()

    def test_invalid_eligible_continuation_fails_before_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            rows = scorer.load_jsonl(fixture.predictions)
            code = next(row for row in rows if row["slice"] == "code")
            code["code_candidate"]["candidate"] = "return x"
            code["row_sha256"] = identity.object_sha256(
                {key: value for key, value in code.items() if key != "row_sha256"}
            )
            fixture.predictions.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            fixture.summary["predictions"]["file_sha256"] = scorer.file_sha256(
                fixture.predictions
            )
            fixture.summary["predictions"]["content_sha256"] = identity.object_sha256(
                rows
            )
            fixture.summary.pop("summary_sha256", None)
            fixture.summary["summary_sha256"] = identity.object_sha256(fixture.summary)
            write_json(fixture.normalized_summary, fixture.summary)
            with self.assertRaisesRegex(
                scorer.Day20CodeE2BV3Error, "eligible Code continuation is invalid"
            ):
                fixture.score()
            self.assertEqual(fixture.execution_calls, [])
            self.assertFalse(fixture.output.exists())

    def test_incomplete_existing_pair_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.output.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                scorer.Day20CodeE2BV3Error, "output pair is incomplete"
            ):
                fixture.score()


if __name__ == "__main__":
    unittest.main()
