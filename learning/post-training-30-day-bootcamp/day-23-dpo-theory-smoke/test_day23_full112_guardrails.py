#!/usr/bin/env python3
"""Focused CPU-only tests for the Day 23 full112 guardrail adapter."""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


DAY23_DIR = Path(__file__).resolve().parent
if str(DAY23_DIR) not in sys.path:
    sys.path.insert(0, str(DAY23_DIR))

import eval_day23_full112_guardrails as guardrail  # noqa: E402
import score_day20_code_e2b_v3 as e2b_scorer  # noqa: E402


class Full112GuardrailTests(unittest.TestCase):
    def test_frozen_component_closure_and_full112_order(self) -> None:
        components = guardrail._frozen_components(guardrail.BOOTCAMP_ROOT)
        self.assertEqual(set(components), set(guardrail.FROZEN_SHA256))
        for name, expected in guardrail.FROZEN_SHA256.items():
            self.assertEqual(components[name]["file_sha256"], expected)
        _, _, selected = guardrail._load_frozen_suite(components)
        self.assertEqual(len(selected), guardrail.RECORDS)
        self.assertEqual(
            {skill: sum(row["slice"] == skill for row in selected) for skill in guardrail.raw_evaluator.SKILLS},
            {skill: guardrail.RECORDS_PER_SKILL for skill in guardrail.raw_evaluator.SKILLS},
        )

    def test_gate_is_pending_only_for_sandbox_scores(self) -> None:
        aggregate = {
            "records": 112,
            "records_by_skill": {skill: 28 for skill in guardrail.raw_evaluator.SKILLS},
            "general_correct": 5,
            "math_correct": 17,
            "finance_correct": 10,
            "non_code_correct": 32,
            "code_correct": None,
            "total_correct": None,
            "format_compliant": 90,
            "code_sandbox_execution_eligible": 26,
            "infrastructure_failures": 0,
        }
        gate = guardrail._gate(aggregate)
        self.assertTrue(gate["passed_pre_sandbox"])
        self.assertEqual(set(gate["pending_sandbox_checks"]), {"code_correct", "total_correct"})
        aggregate["general_correct"] = 4
        self.assertFalse(guardrail._gate(aggregate)["passed_pre_sandbox"])

    def test_cli_and_public_verifier_surfaces_are_fixed(self) -> None:
        destinations = {action.dest for action in guardrail.build_parser()._actions}
        self.assertEqual(
            destinations,
            {"help", "campaign", "training_receipt", "dev_evaluation", "self_test"},
        )
        pair_parameters = inspect.signature(guardrail.verify_normalized_pair).parameters
        self.assertEqual(
            list(pair_parameters),
            ["predictions_path", "summary_path", "expected_scope", "expected_candidate"],
        )
        self.assertEqual(list(inspect.signature(guardrail.validate_receipt).parameters), ["path"])

    def test_gpu_visibility_and_atomic_publish_fail_closed(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            guardrail._pin_physical_gpu_zero()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True):
            with self.assertRaisesRegex(guardrail.Full112GuardrailError, "physical GPU 0"):
                guardrail._pin_physical_gpu_zero()
        with tempfile.TemporaryDirectory(prefix="day23-full112-atomic-") as raw:
            path = Path(raw) / "artifact.json"
            guardrail._atomic_bytes_new(path, b"first\n")
            self.assertEqual(path.read_bytes(), b"first\n")
            with self.assertRaisesRegex(guardrail.Full112GuardrailError, "overwrite"):
                guardrail._atomic_bytes_new(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_offline_raw_to_normalized_pair_revalidates_all_112_rows(self) -> None:
        components = guardrail._frozen_components(guardrail.BOOTCAMP_ROOT)
        _, scorer, records = guardrail._load_frozen_suite(components)
        model = {
            "base": {"fixture": True},
            "adapter": {"files_sha256": "a" * 64},
            "expected_model_class": guardrail.raw_evaluator.EXPECTED_MODEL_CLASS,
        }
        checkpoint = {"files_sha256": "a" * 64}
        provenance = {
            "model_identity": model,
            "model_identity_sha256": guardrail._object_sha256(model),
            "checkpoint_package": checkpoint,
            "checkpoint_package_sha256": guardrail._object_sha256(checkpoint),
            "compact_model_identity": {
                "model_identity_sha256": guardrail._object_sha256(model),
                "base_snapshot_sha256": guardrail._object_sha256(model["base"]),
                "adapter_snapshot_sha256": "a" * 64,
            },
            "compact_checkpoint_package": {
                "checkpoint_package_sha256": guardrail._object_sha256(checkpoint)
            },
        }
        raw_rows = []
        for ordinal, record in enumerate(records, 1):
            output = (
                "    pass"
                if record["slice"] == "code"
                else ("A" if record["slice"] == "general" else "0")
            )
            raw_rows.append(
                guardrail.raw_evaluator.build_prediction_row(
                    ordinal=ordinal,
                    candidate_identity=guardrail.CANDIDATE,
                    eval_scope=guardrail.SCOPE,
                    record=record,
                    message_content=output,
                    prompt_token_ids=[1, ordinal],
                    generated_token_ids=[ordinal],
                    generated_only_text=output,
                    finish_reason="stop",
                    prompt_tokens=2,
                    completion_tokens=1,
                    scorer=scorer,
                    compact_model_identity=provenance["compact_model_identity"],
                    compact_checkpoint_package=provenance[
                        "compact_checkpoint_package"
                    ],
                )
            )
        with tempfile.TemporaryDirectory(prefix="day23-full112-pair-") as raw:
            root = Path(raw)
            raw_predictions = root / guardrail.RAW_PREDICTIONS_NAME
            raw_summary_path = root / guardrail.RAW_SUMMARY_NAME
            predictions = root / guardrail.NORMALIZED_PREDICTIONS_NAME
            summary_path = root / guardrail.NORMALIZED_SUMMARY_NAME
            guardrail._atomic_bytes_new(
                raw_predictions, guardrail._jsonl_bytes(raw_rows)
            )
            raw_summary = guardrail._raw_summary(
                rows=raw_rows,
                path=raw_predictions,
                provenance=provenance,
                components=components,
                context={"campaign_sha256": "b" * 64},
                checkpoint={"receipt_sha256": "c" * 64},
                dev={"identity": {"content_sha256": "d" * 64}},
                runtime={},
                elapsed=1.0,
                peak_memory=1.0,
                process_identity={},
            )
            guardrail._atomic_bytes_new(
                raw_summary_path, guardrail._json_bytes(raw_summary)
            )
            normalized_rows, normalized_summary = guardrail._normalize(
                raw_rows=raw_rows,
                raw_summary=raw_summary,
                records=records,
                scorer=scorer,
                components=components,
                context={
                    "campaign_file_sha256": "e" * 64,
                    "campaign_sha256": "b" * 64,
                },
                raw_predictions_path=raw_predictions,
                raw_summary_path=raw_summary_path,
                normalized_predictions_path=predictions,
            )
            guardrail._atomic_bytes_new(
                predictions, guardrail._jsonl_bytes(normalized_rows)
            )
            guardrail._atomic_bytes_new(
                summary_path, guardrail._json_bytes(normalized_summary)
            )
            verified, _ = guardrail.verify_normalized_pair(
                predictions,
                summary_path,
                expected_scope="full112",
                expected_candidate=guardrail.CANDIDATE_ID,
            )
            self.assertEqual(len(verified), 112)
            self.assertEqual(sum(row["slice"] == "code" for row in verified), 28)
            code_rows, eligible_rows = e2b_scorer._strict_code_rows(
                verified, scope="full112"
            )
            self.assertEqual((len(code_rows), len(eligible_rows)), (28, 28))


if __name__ == "__main__":
    unittest.main()
