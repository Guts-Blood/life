#!/usr/bin/env python3
"""Focused offline tests for the Day 23 E2B guardrail wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_day23_qwen35_gpu_stage as gpu_stage
import score_day23_code_e2b_guardrails as guardrail


def code_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal in range(guardrail.CODE_RECORDS):
        row: dict[str, object] = {
            "sample_id": f"eval:code:{ordinal:03d}",
            "slice": "code",
            "sandbox_execution_eligible": ordinal < 26,
        }
        row["row_sha256"] = gpu_stage.object_sha256(row)
        rows.append(row)
    return rows


def full_aggregate() -> dict[str, int]:
    return {
        "general_correct": 20,
        "math_correct": 17,
        "finance_correct": 14,
        "format_compliant": 100,
    }


def e2b_summary(**updates: int) -> dict[str, int]:
    value = {
        "records": 26,
        "passed": 14,
        "failed": 12,
        "sandbox_execution_eligible": 26,
        "infrastructure_failures": 0,
        "code_records": 28,
    }
    value.update(updates)
    return value


class Day23E2BGuardrailTests(unittest.TestCase):
    def test_exact_promotion_boundaries_pass(self) -> None:
        gate = guardrail.evaluate_gate(full_aggregate(), e2b_summary())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["observed"]["total_correct"], 65)
        self.assertTrue(all(gate["checks"].values()))

    def test_code_and_eligibility_floors_fail_closed(self) -> None:
        gate = guardrail.evaluate_gate(
            full_aggregate(),
            e2b_summary(passed=13, sandbox_execution_eligible=25, failed=12),
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["code_correct"])
        self.assertFalse(gate["checks"]["code_sandbox_execution_eligible"])

    def test_e2b_must_bind_exactly_28_code_rows(self) -> None:
        with self.assertRaisesRegex(
            guardrail.Day23E2BGuardrailError, "exactly 28 Code rows"
        ):
            guardrail.evaluate_gate(
                full_aggregate(), e2b_summary(code_records=27)
            )

    def test_infrastructure_failure_is_a_hard_gate_failure(self) -> None:
        gate = guardrail.evaluate_gate(
            full_aggregate(), e2b_summary(infrastructure_failures=1)
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["infrastructure_failures"])

    def test_code_row_boundary_rejects_noncode_duplicate_and_tamper(self) -> None:
        rows = code_rows()
        self.assertEqual(len(guardrail._validate_code_rows(rows)), 28)

        noncode = [dict(row) for row in rows]
        noncode[0]["slice"] = "math"
        noncode[0]["row_sha256"] = gpu_stage.object_sha256(
            noncode[0], "row_sha256"
        )
        with self.assertRaisesRegex(
            guardrail.Day23E2BGuardrailError, "contains a non-Code row"
        ):
            guardrail._validate_code_rows(noncode)

        duplicate = [dict(row) for row in rows]
        duplicate[-1]["sample_id"] = duplicate[0]["sample_id"]
        duplicate[-1]["row_sha256"] = gpu_stage.object_sha256(
            duplicate[-1], "row_sha256"
        )
        with self.assertRaisesRegex(
            guardrail.Day23E2BGuardrailError, "not unique"
        ):
            guardrail._validate_code_rows(duplicate)

        tampered = [dict(row) for row in rows]
        tampered[0]["sample_id"] = "tampered"
        with self.assertRaisesRegex(
            guardrail.Day23E2BGuardrailError, "self hash drifted"
        ):
            guardrail._validate_code_rows(tampered)

    def test_receipt_is_self_hashed_o_excl_and_has_no_heldout_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            guardrail_root = root / "evidence/guardrails"
            guardrail_root.mkdir(parents=True)
            campaign_path = root / "binding/gpu-campaign.json"
            training_path = root / "evidence/training.json"
            dev_path = root / "evidence/dev.json"
            full_path = guardrail_root / guardrail.FULL112_RECEIPT_NAME
            campaign_path.parent.mkdir()
            training_path.parent.mkdir(exist_ok=True)
            for path in (campaign_path, training_path, dev_path, full_path):
                path.write_text("{}\n", encoding="utf-8")
            context = {
                "run_root": root,
                "campaign_path": campaign_path,
                "campaign_sha256": "a" * 64,
                "training_path": training_path,
                "training_sha256": "b" * 64,
                "checkpoint": {
                    "path": str(root / "outputs/checkpoint-20"),
                    "files_sha256": "c" * 64,
                },
            }
            dev = {"path": dev_path, "evaluation_sha256": "d" * 64}
            full = {
                "path": full_path,
                "receipt_sha256": "e" * 64,
                "aggregate": full_aggregate(),
                "code_ids": [f"eval:code:{ordinal:03d}" for ordinal in range(28)],
            }
            summary = {
                **e2b_summary(),
                "e2b_run_sha256": "f" * 64,
                "complete_comparison_key": "1" * 64,
            }
            execution = {
                "components": {},
                "preflight": {},
                "credential_file": {},
                "credential_attestation": {},
                "results": {},
                "summary": {},
            }
            output = guardrail_root / guardrail.E2B_RECEIPT_NAME
            sealed = guardrail._write_receipt(
                output, context, dev, full, summary, execution
            )
            self.assertEqual(sealed["status"], "pass")
            self.assertEqual(
                gpu_stage.object_sha256(sealed, "receipt_sha256"),
                sealed["receipt_sha256"],
            )
            self.assertFalse(sealed["claim_boundary"]["heldout_interface_exposed"])
            self.assertNotIn("heldout_path", json.dumps(sealed))
            with self.assertRaisesRegex(
                guardrail.Day23E2BGuardrailError, "already exists"
            ):
                guardrail._write_receipt(
                    output, context, dev, full, summary, execution
                )

    def test_cli_exposes_no_heldout_argument(self) -> None:
        destinations = {action.dest for action in guardrail.build_parser()._actions}
        self.assertNotIn("heldout", destinations)
        self.assertNotIn("heldout_path", destinations)

    def test_operational_failure_is_sealed_once_without_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "evidence/guardrails").mkdir(parents=True)
            campaign = root / "campaign.json"
            training = root / "training.json"
            dev = root / "dev.json"
            full = root / "evidence/guardrails" / guardrail.FULL112_RECEIPT_NAME
            for path in (campaign, training, dev, full):
                path.write_text("{}\n", encoding="utf-8")
            context = {
                "run_root": root,
                "campaign_path": campaign,
                "campaign_sha256": "a" * 64,
                "training_path": training,
                "training_sha256": "b" * 64,
            }
            output = root / "evidence/guardrails" / guardrail.E2B_FAILURE_NAME
            sealed = guardrail._seal_failure(
                output,
                phase="frozen_e2b_preflight_and_scoring",
                error=RuntimeError("fixture network failure"),
                context=context,
                dev_path=dev,
                full112_path=full,
            )
            self.assertEqual(sealed["status"], "failed_closed")
            self.assertFalse(sealed["claim_boundary"]["retry_allowed"])
            self.assertNotIn("heldout_path", json.dumps(sealed))
            self.assertEqual(
                sealed["failure_sha256"],
                gpu_stage.object_sha256(sealed, "failure_sha256"),
            )
            with self.assertRaisesRegex(
                guardrail.Day23E2BGuardrailError,
                "cannot seal E2B failure receipt",
            ):
                guardrail._seal_failure(
                    output,
                    phase="frozen_e2b_preflight_and_scoring",
                    error=RuntimeError("second failure"),
                    context=context,
                    dev_path=dev,
                    full112_path=full,
                )


if __name__ == "__main__":
    unittest.main()
