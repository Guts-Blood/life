from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import control_day23_qualification_closeout as closeout
import eval_day23_qualification_heldout as heldout_eval


def _sealed(path: Path, field: str, **value: object) -> dict[str, object]:
    document = closeout.seal(value, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(closeout.json_bytes(document))
    return document


class QualificationCloseoutTests(unittest.TestCase):
    def test_heldout_gate_freezes_first_sign_test_boundary(self) -> None:
        aggregate = {
            "overall": {"pairs": 29, "wins": 21, "mean_reward_margin": 0.01},
            "length_matched": {"mean_reward_margin": 0.001},
        }
        self.assertTrue(closeout.heldout_gate(aggregate)["passed"])
        aggregate["overall"]["wins"] = 20
        self.assertFalse(closeout.heldout_gate(aggregate)["passed"])
        aggregate["overall"]["wins"] = 21
        aggregate["length_matched"]["mean_reward_margin"] = 0.0
        self.assertFalse(closeout.heldout_gate(aggregate)["passed"])

    def test_combined_guardrails_require_full112_plus_e2b(self) -> None:
        full112 = {
            "records": 112,
            "records_by_skill": {"code": 28, "finance": 28, "general": 28, "math": 28},
            "general_correct": 20,
            "math_correct": 17,
            "finance_correct": 14,
            "non_code_correct": 51,
            "format_compliant": 90,
            "code_sandbox_execution_eligible": 26,
            "infrastructure_failures": 0,
        }
        e2b = {
            "total_correct": 65,
            "general_correct": 20,
            "math_correct": 17,
            "finance_correct": 14,
            "code_correct": 14,
            "format_compliant": 90,
            "code_sandbox_execution_eligible": 26,
            "infrastructure_failures": 0,
            "code_records": 28,
            "failed": 12,
        }
        self.assertTrue(closeout.recompute_combined_guardrails(full112, e2b)["passed"])
        e2b["code_correct"] = 13
        e2b["total_correct"] = 64
        e2b["failed"] = 13
        self.assertFalse(closeout.recompute_combined_guardrails(full112, e2b)["passed"])
        e2b["total_correct"] = 65
        with self.assertRaisesRegex(closeout.QualificationCloseoutError, "projection drifted"):
            closeout.recompute_combined_guardrails(full112, e2b)

    def test_appendment_and_authorization_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            bootcamp = tmp_path / "bootcamp"
            goal_root = bootcamp / "rsi-control/charters" / closeout.GOAL_ID
            run_root = tmp_path / "qual-run"
            charter_path = goal_root / "charter.json"
            goal2_path = bootcamp / closeout.GOAL2_REL
            training_path = run_root / "evidence/refit/success-receipt.json"
            dev_path = run_root / "evidence/dev/finalist-dev.json"
            full112_path = run_root / "evidence/guardrails/full112-generation-and-score.json"
            e2b_path = run_root / "evidence/guardrails/e2b-sandbox.json"
            _sealed(charter_path, "charter_sha256", goal_id=closeout.GOAL_ID)
            _sealed(goal2_path, "charter_sha256", goal_id="goal-0002-day23-dpo")
            training = _sealed(training_path, "receipt_sha256", status="pass")
            dev = _sealed(dev_path, "evaluation_sha256", status="pass")
            full112 = _sealed(
                full112_path, "receipt_sha256", status="pass_pending_code_sandbox"
            )
            e2b = _sealed(e2b_path, "receipt_sha256", status="pass")
            event1_path = goal_root / "events/event-000001-charter-frozen.json"
            event2_path = goal_root / "events/event-000002-source-candidate-imported.json"
            event1 = _sealed(event1_path, "event_sha256", sequence=1)
            event2 = _sealed(event2_path, "event_sha256", sequence=2)
            evaluator = Path(heldout_eval.__file__).resolve(strict=True)
            combined = {
                "passed": True,
                "thresholds": closeout.EXPECTED_CODING_GATES,
                "observed": {
                    "total_correct": 65,
                    "general_correct": 20,
                    "math_correct": 17,
                    "finance_correct": 14,
                    "code_correct": 14,
                    "format_compliant": 90,
                    "code_sandbox_execution_eligible": 26,
                    "infrastructure_failures": 0,
                },
            }
            state = {
                "context": {
                    "run_root": run_root,
                    "campaign_path": run_root / "binding/gpu-campaign.json",
                    "campaign_file_sha256": "1" * 64,
                    "campaign_sha256": "2" * 64,
                    "training_receipt_path": training_path,
                    "training_receipt_file_sha256": closeout.gpu_stage.file_sha256(training_path),
                    "training_receipt_sha256": training["receipt_sha256"],
                    "checkpoint": {"path": "/checkpoint-20", "global_step": 20},
                },
                "authority": {
                    "bootcamp": bootcamp,
                    "goal_root": goal_root,
                    "goal4_charter_path": charter_path,
                    "goal2_path": goal2_path,
                    "initial_event_paths": [event1_path, event2_path],
                    "initial_events": [event1, event2],
                    "heldout_identity": {"path": "protected/heldout.jsonl", "records": 29},
                    "heldout_claim_path": bootcamp
                    / "rsi-control/access-ledger/scope/heldout-claim.json",
                },
                "dev": {
                    "path": dev_path,
                    "file_sha256": closeout.gpu_stage.file_sha256(dev_path),
                    "evaluation_sha256": dev["evaluation_sha256"],
                    "value": {"eligibility": {"passed": True}},
                },
                "full112": {
                    "path": full112_path,
                    "file_sha256": closeout.gpu_stage.file_sha256(full112_path),
                    "receipt_sha256": full112["receipt_sha256"],
                },
                "e2b": {
                    "path": e2b_path,
                    "file_sha256": closeout.gpu_stage.file_sha256(e2b_path),
                    "receipt_sha256": e2b["receipt_sha256"],
                    "combined_gate": combined,
                },
                "heldout_evaluator": closeout.file_identity(evaluator),
            }
            documents = closeout._authorize_documents(state, "2026-08-14T12:00:00Z")
            amendment = documents[2][1]
            authorization = documents[-1][1]
            gate = amendment["heldout_gate"]
            self.assertEqual(gate["positive_margin_pairs"]["threshold"], 21)
            self.assertEqual(
                gate["derivation"]["heldout_21_of_29_tail_probability"],
                closeout.HELDOUT_21_SIGN_TEST_P,
            )
            self.assertEqual(
                gate["derivation"]["adjacent_20_of_29_tail_probability"],
                closeout.HELDOUT_20_SIGN_TEST_P,
            )
            self.assertEqual(authorization["maximum_global_claims"], 1)
            self.assertEqual(authorization["maximum_candidate_evaluations"], 1)
            self.assertFalse(authorization["retry_allowed"])
            self.assertFalse(
                authorization["claim_boundary"]["heldout_claimed_at_authorization"]
            )
            state["dev"]["value"]["eligibility"] = {
                "passed": False,
                "observed": {
                    "pairs": 17,
                    "positive_margin_pairs": 8,
                    "mean_reward_margin": -0.0007107867914087636,
                    "length_matched_margin": -0.002253224849700928,
                },
            }
            terminal_documents = closeout._dev_failure_documents(
                state, "2026-08-14T12:00:00Z"
            )
            self.assertEqual(len(terminal_documents), 2)
            terminal = terminal_documents[-1][1]
            self.assertEqual(terminal["state_after"], "terminal_dev_not_qualified")
            self.assertFalse(terminal["payload"]["heldout_authorized"])
            self.assertFalse(terminal["payload"]["full112_guardrail_authorized"])

    def test_heldout_evaluator_has_no_split_selector(self) -> None:
        destinations = {action.dest for action in heldout_eval.build_parser()._actions}
        self.assertEqual(destinations, {"help", "authorization", "device", "self_test"})


if __name__ == "__main__":
    unittest.main()
