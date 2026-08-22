#!/usr/bin/env python3

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import rsi_control


class RSIControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = rsi_control.load_json(rsi_control.ROOT / "goal.json")
        self.gates = self.goal["hard_gates"]

    def test_folder_validates(self) -> None:
        self.assertEqual(rsi_control.validate_control()["status"], "valid")

    def test_exact_threshold_passes(self) -> None:
        values = {name: gate["threshold"] for name, gate in self.gates.items()}
        result = rsi_control.compute_gate_summary(values, self.gates)
        self.assertTrue(result["all_numeric_gates_pass"])
        self.assertEqual(result["worst_gate_margin"], 0)

    def test_one_gate_cannot_be_hidden_by_high_total(self) -> None:
        values = {name: gate["threshold"] for name, gate in self.gates.items()}
        values["total_correct"] = 100
        values["finance_correct"] = 8
        result = rsi_control.compute_gate_summary(values, self.gates)
        self.assertFalse(result["all_numeric_gates_pass"])
        self.assertEqual(result["worst_gate_margin"], -2)
        self.assertEqual(result["total_gate_deficit"], 2)

    def test_infrastructure_is_a_maximum_gate(self) -> None:
        values = {name: gate["threshold"] for name, gate in self.gates.items()}
        values["infrastructure_failures"] = 1
        result = rsi_control.compute_gate_summary(values, self.gates)
        self.assertEqual(result["gate_margins"]["infrastructure_failures"], -1)

    def test_gate_schema_is_exact(self) -> None:
        values = {name: gate["threshold"] for name, gate in self.gates.items()}
        values["average"] = 999
        with self.assertRaises(rsi_control.RSIControlError):
            rsi_control.compute_gate_summary(values, self.gates)


class RSITaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = rsi_control.load_json(rsi_control.ROOT / "taxonomy.json")
        self.failure_taxonomy = rsi_control.load_json(
            rsi_control.ROOT / "failure-taxonomy.json"
        )
        self.intervention_contract = rsi_control.validate_intervention_taxonomy(
            self.taxonomy
        )
        self.failure_contract = rsi_control.validate_failure_taxonomy(
            self.failure_taxonomy, self.intervention_contract["detailed_levers"]
        )

    def test_taxonomies_validate_and_activate_together(self) -> None:
        self.assertEqual(
            self.intervention_contract["effective_from_version"], "rsi-v0003"
        )
        self.assertEqual(
            self.failure_contract["effective_from_version"], "rsi-v0003"
        )
        self.assertIn(
            "failure.training.transfer.catastrophic_forgetting",
            self.failure_contract["failure_modes"],
        )
        self.assertIn(
            "model.data.replay.replay_ratio",
            self.intervention_contract["detailed_levers"],
        )

    def test_lever_family_matching_respects_segment_boundaries(self) -> None:
        for lever, family in (
            ("verifier.coverage.domain_task", "verifier"),
            ("runtime.topology.world_size", "runtime"),
            ("governance.audit.hash_chain", "governance"),
        ):
            self.assertTrue(rsi_control.segment_prefix(lever, family))
        self.assertFalse(rsi_control.segment_prefix("model.database.foo", "model.data"))

    def test_duplicate_detailed_lever_is_rejected(self) -> None:
        taxonomy = copy.deepcopy(self.taxonomy)
        taxonomy["detailed_levers"].append(
            copy.deepcopy(taxonomy["detailed_levers"][0])
        )
        with self.assertRaisesRegex(rsi_control.RSIControlError, "duplicate"):
            rsi_control.validate_intervention_taxonomy(taxonomy)

    def test_unknown_failure_parent_and_alias_target_are_rejected(self) -> None:
        bad_parent = copy.deepcopy(self.failure_taxonomy)
        bad_parent["failure_modes"][0]["parent"] = "failure.data.unknown"
        with self.assertRaisesRegex(rsi_control.RSIControlError, "unknown parent"):
            rsi_control.validate_failure_taxonomy(
                bad_parent, self.intervention_contract["detailed_levers"]
            )

        bad_alias = copy.deepcopy(self.failure_taxonomy)
        bad_alias["legacy_aliases"][0]["maps_to"] = "failure.data.unknown.mode"
        with self.assertRaisesRegex(rsi_control.RSIControlError, "unknown target"):
            rsi_control.validate_failure_taxonomy(
                bad_alias, self.intervention_contract["detailed_levers"]
            )

        bad_crosswalk = copy.deepcopy(self.failure_taxonomy)
        bad_crosswalk["intervention_crosswalk_examples"][0]["candidate_levers"] = [
            "model.target.unknown"
        ]
        with self.assertRaisesRegex(rsi_control.RSIControlError, "unknown intervention"):
            rsi_control.validate_failure_taxonomy(
                bad_crosswalk, self.intervention_contract["detailed_levers"]
            )

    def test_worked_case_may_have_no_dependent_lever(self) -> None:
        taxonomy = copy.deepcopy(self.failure_taxonomy)
        taxonomy["case_examples"][0]["historical_intervention"][
            "dependent_lever"
        ] = None
        taxonomy["case_examples"][0]["prospective_canonical_equivalent"][
            "dependent_lever"
        ] = None
        rsi_control.validate_failure_taxonomy(
            taxonomy, self.intervention_contract["detailed_levers"]
        )

    def diagnosis_version(self) -> tuple[dict, dict]:
        lever = "model.data.replay.replay_ratio"
        intervention = {"primary_lever": lever, "changed_levers": [lever]}
        version = {
            "diagnosis": {
                "diagnosis_id": "diagnosis-0003",
                "observed_symptoms": [
                    {
                        "id": "symptom.training.retention_loss",
                        "scope": "primary checkpoints / retained domain",
                        "evidence_refs": ["evidence/retention-trajectory.json"],
                    }
                ],
                "primary_failure": {
                    "id": "failure.training.transfer.catastrophic_forgetting",
                    "first_broken_invariant": "retained-domain capability must not regress",
                    "pipeline_boundary": "valid checkpoint before transfer -> checkpoint after transfer",
                    "causal_status": "supported",
                    "evidence_refs": ["evidence/retention-trajectory.json"],
                },
                "contributing_failures": [],
                "excluded_alternatives": [
                    {
                        "id": "failure.eval.scorer.metric_semantics",
                        "reason": "the frozen scorer reproduced the baseline",
                        "evidence_refs": ["evidence/scorer-replay.json"],
                    }
                ],
                "intervention": {
                    "primary_lever": lever,
                    "dependent_lever": None,
                    "dependent_lever_reason": None,
                    "expected_mechanism": "replay restores old-domain gradient exposure",
                },
                "prediction": "retained-domain score recovers without losing the new domain",
                "falsifier": "retained-domain regression persists at the preregistered replay ratio",
                "frozen_invariants": ["dataset identities outside replay selection"],
                "guardrails": ["new-domain hard gate"],
            }
        }
        return version, intervention

    def test_structured_future_diagnosis_validates(self) -> None:
        version, intervention = self.diagnosis_version()
        rsi_control.validate_version_diagnosis(
            version,
            intervention,
            self.failure_contract,
            self.intervention_contract["detailed_levers"],
        )

    def test_unknown_future_failure_is_rejected(self) -> None:
        version, intervention = self.diagnosis_version()
        version["diagnosis"]["primary_failure"]["id"] = "failure.training.unknown"
        with self.assertRaisesRegex(rsi_control.RSIControlError, "canonical mode"):
            rsi_control.validate_version_diagnosis(
                version,
                intervention,
                self.failure_contract,
                self.intervention_contract["detailed_levers"],
            )

    def test_future_diagnosis_rejects_generated_evidence(self) -> None:
        version, intervention = self.diagnosis_version()
        version["diagnosis"]["observed_symptoms"][0]["evidence_refs"] = [
            "generated/evidence.json"
        ]
        with self.assertRaisesRegex(
            rsi_control.RSIControlError, "non-generated relative paths"
        ):
            rsi_control.validate_version_diagnosis(
                version,
                intervention,
                self.failure_contract,
                self.intervention_contract["detailed_levers"],
            )
        version["diagnosis"]["observed_symptoms"][0]["evidence_refs"] = [
            "Generated/evidence.json"
        ]
        with self.assertRaisesRegex(
            rsi_control.RSIControlError, "non-generated relative paths"
        ):
            rsi_control.validate_version_diagnosis(
                version,
                intervention,
                self.failure_contract,
                self.intervention_contract["detailed_levers"],
            )


class RSIFutureTaxonomyEnforcementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_root = rsi_control.ROOT
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "rsi-control"
        shutil.copytree(self.old_root, self.root)
        rsi_control.ROOT = self.root
        for name in ("taxonomy.json", "failure-taxonomy.json"):
            path = self.root / name
            value = json.loads(path.read_text(encoding="utf-8"))
            value["effective_from_version"] = "rsi-v0002"
            path.write_text(json.dumps(value), encoding="utf-8")

    def tearDown(self) -> None:
        rsi_control.ROOT = self.old_root
        self.temporary.cleanup()

    def test_new_policy_rejects_coarse_historical_levers(self) -> None:
        with self.assertRaisesRegex(rsi_control.RSIControlError, "canonical lever leaf"):
            rsi_control.validate_control()

    def test_new_policy_requires_structured_diagnosis_after_exact_levers(self) -> None:
        path = self.root / "versions/rsi-v0002/version.json"
        version = json.loads(path.read_text(encoding="utf-8"))
        version["intervention"].update(
            {
                "primary_lever": "model.target.representation.boundary_tokens",
                "changed_levers": [
                    "model.target.representation.boundary_tokens",
                    "model.target.masking.token_eligibility",
                ],
            }
        )
        path.write_text(json.dumps(version), encoding="utf-8")
        with self.assertRaisesRegex(rsi_control.RSIControlError, "version diagnosis"):
            rsi_control.validate_control()

    def test_dependent_lever_must_be_inside_goal_action_space(self) -> None:
        path = self.root / "versions/rsi-v0002/version.json"
        version = json.loads(path.read_text(encoding="utf-8"))
        version["intervention"].update(
            {
                "primary_lever": "model.data.replay.replay_ratio",
                "changed_levers": [
                    "model.data.replay.replay_ratio",
                    "verifier.scoring.normalization",
                ],
                "dependent_lever_reason": "invalid test-only dependency",
            }
        )
        path.write_text(json.dumps(version), encoding="utf-8")
        with self.assertRaisesRegex(rsi_control.RSIControlError, "outside current goal"):
            rsi_control.validate_control()


class RSIRecordingBridgeTests(unittest.TestCase):
    version_id = "rsi-v0001"
    run_id = "run-001-probe-primary"
    attempt_id = "attempt-001"

    def setUp(self) -> None:
        self.old_root = rsi_control.ROOT
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "rsi-control"
        shutil.copytree(self.old_root, self.root)
        artifacts_path = self.root / "versions/rsi-v0001/artifacts.json"
        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        for record in artifacts["implementation"]:
            record["sha256"] = rsi_control.sha256(
                rsi_control.BOOTCAMP_ROOT / record["path"]
            )
        artifacts_path.write_text(json.dumps(artifacts), encoding="utf-8")
        run_dir = (
            self.root
            / "versions"
            / self.version_id
            / "runs"
            / self.run_id
        )
        run_path = run_dir / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run.update(
            {
                "status": "planned",
                "run_contract_sha256": None,
                "remote_run_root": None,
            }
        )
        run.pop("run_contract_path", None)
        run_path.write_text(json.dumps(run), encoding="utf-8")
        (run_dir / "run-contract.json").unlink(missing_ok=True)

        attempt_dir = run_dir / "attempts" / self.attempt_id
        attempt_path = attempt_dir / "attempt.json"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt.update(
            {
                "status": "planned",
                "run_contract_sha256": None,
                "execution_identity": None,
                "started_at_utc": None,
                "ended_at_utc": None,
                "exit_code": None,
                "consumed_gpu_hours": 0.0,
                "last_valid_checkpoint": None,
                "retry_count": 0,
                "error": None,
            }
        )
        attempt.pop("event_log", None)
        attempt.pop("active_retry", None)
        attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
        logs_dir = attempt_dir / "logs"
        (logs_dir / "operations.jsonl").unlink(missing_ok=True)
        (logs_dir / "retries.jsonl").write_text(
            json.dumps(
                {
                    "schema_name": "rsi.retry_event",
                    "schema_version": 1,
                    "event_type": "retry_log_initialized",
                    "version_id": self.version_id,
                    "run_id": self.run_id,
                    "attempt_id": self.attempt_id,
                    "retry_ordinal": 0,
                    "created_at_utc": "2026-08-12T00:00:00Z",
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

        metrics_path = self.root / "versions" / self.version_id / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["observed"]["gpu_hours"] = 0.0
        metrics["observed"]["eval_accesses"] = 0
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        rsi_control.ROOT = self.root
        self.contract_source = self.root / "contract-input.json"
        self.contract = {
            "schema_name": "rsi.run_contract",
            "schema_version": 1,
            "version_id": self.version_id,
            "run_id": self.run_id,
            "phase": "probe",
            "seed": 20260809,
            "implementation_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
        }
        self.contract_source.write_text(
            json.dumps(self.contract), encoding="utf-8"
        )
        self.execution = {"host": "gpu.example", "gpu": "GPU-0"}
        self.error = {
            "layer": "infrastructure",
            "code": "ssh_disconnect",
            "message": "connection lost",
            "retryable": True,
        }

    def tearDown(self) -> None:
        rsi_control.ROOT = self.old_root
        self.temporary.cleanup()

    def bind(self) -> dict:
        return rsi_control.bind_run(
            version_id=self.version_id,
            run_id=self.run_id,
            contract_source=self.contract_source,
            remote_run_root="/root/autodl-tmp/runs/test",
        )

    def start(self) -> dict:
        self.bind()
        return rsi_control.attempt_start(
            version_id=self.version_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_identity=self.execution,
            started_at_utc="2026-08-12T01:00:00Z",
        )

    def record(self, event_id: str, operation_type: str, gpu_hours: float = 0.0):
        return rsi_control.record_operation(
            version_id=self.version_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            event_id=event_id,
            operation_type=operation_type,
            payload={"command": event_id},
            gpu_hours=gpu_hours,
            eval_accesses=1 if operation_type == "eval" else 0,
            created_at_utc="2026-08-12T01:10:00Z",
        )

    def attempt_path(self) -> Path:
        return (
            self.root
            / "versions"
            / self.version_id
            / "runs"
            / self.run_id
            / "attempts"
            / self.attempt_id
            / "attempt.json"
        )

    def test_bind_stores_payload_and_hash_and_is_idempotent(self) -> None:
        first = self.bind()
        second = self.bind()
        self.assertEqual(first, second)
        contract = rsi_control.load_json(self.contract_source.parent / "versions" / self.version_id / "runs" / self.run_id / "run-contract.json")
        self.assertEqual(first["run_contract_sha256"], contract["contract_sha256"])
        self.assertEqual(
            contract["contract_sha256"],
            rsi_control.object_sha256(
                {key: value for key, value in contract.items() if key != "contract_sha256"}
            ),
        )

    def test_rebinding_different_contract_is_rejected(self) -> None:
        self.bind()
        self.contract["seed"] = 7
        self.contract_source.write_text(json.dumps(self.contract), encoding="utf-8")
        with self.assertRaisesRegex(rsi_control.RSIControlError, "rebind"):
            self.bind()

    def test_attempt_requires_contract_and_valid_lifecycle(self) -> None:
        with self.assertRaisesRegex(rsi_control.RSIControlError, "bind-run"):
            rsi_control.attempt_start(
                version_id=self.version_id,
                run_id=self.run_id,
                attempt_id=self.attempt_id,
                execution_identity=self.execution,
            )
        attempt = self.start()
        self.assertEqual(attempt["status"], "running")
        finished = rsi_control.attempt_finish(
            version_id=self.version_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            status="succeeded",
            exit_code=0,
            error=None,
            ended_at_utc="2026-08-12T02:00:00Z",
        )
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(rsi_control.validate_control()["status"], "valid")

    def test_operation_ledger_is_hash_chained_idempotent_and_aggregated(self) -> None:
        self.start()
        first = self.record("train-001", "train", 0.5)
        self.assertEqual(first, self.record("train-001", "train", 0.5))
        second = self.record("eval-001", "eval", 0.25)
        self.assertEqual(second["previous_event_sha256"], first["event_sha256"])
        attempt = rsi_control.load_json(self.attempt_path())
        self.assertEqual(attempt["consumed_gpu_hours"], 0.75)
        metrics = rsi_control.load_json(
            self.root / "versions" / self.version_id / "metrics.json"
        )
        self.assertEqual(metrics["observed"]["gpu_hours"], 0.75)
        self.assertEqual(metrics["observed"]["eval_accesses"], 1)

    def test_operation_event_id_cannot_change_meaning(self) -> None:
        self.start()
        self.record("train-001", "train", 0.5)
        with self.assertRaisesRegex(rsi_control.RSIControlError, "reused"):
            self.record("train-001", "train", 0.6)

    def test_ledger_tampering_and_metric_drift_are_rejected(self) -> None:
        self.start()
        self.record("eval-001", "eval", 0.1)
        ledger = self.attempt_path().parent / "logs/operations.jsonl"
        row = json.loads(ledger.read_text(encoding="utf-8"))
        row["gpu_hours"] = 99
        ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(rsi_control.RSIControlError, "hash drifted"):
            rsi_control.validate_control()

    def test_retry_is_paired_and_preserves_contract(self) -> None:
        self.start()
        rsi_control.attempt_finish(
            version_id=self.version_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            status="infrastructure_failed",
            exit_code=255,
            error=self.error,
            ended_at_utc="2026-08-12T02:00:00Z",
        )
        started = rsi_control.retry_start(
            version_id=self.version_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            remediation="reconnect without semantic changes",
            created_at_utc="2026-08-12T02:10:00Z",
        )
        self.assertEqual(started["run_contract_before"], started["run_contract_after"])
        finished = rsi_control.retry_finish(
            version_id=self.version_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            status="succeeded",
            exit_code=0,
            error=None,
            created_at_utc="2026-08-12T03:00:00Z",
        )
        self.assertEqual(finished["retry_ordinal"], 1)
        self.assertEqual(rsi_control.load_json(self.attempt_path())["retry_count"], 1)
        self.assertEqual(rsi_control.validate_control()["status"], "valid")

    def test_infrastructure_error_cannot_be_labeled_model_failure(self) -> None:
        self.start()
        with self.assertRaisesRegex(rsi_control.RSIControlError, "infrastructure"):
            rsi_control.attempt_finish(
                version_id=self.version_id,
                run_id=self.run_id,
                attempt_id=self.attempt_id,
                status="model_failed",
                exit_code=1,
                error=self.error,
            )


if __name__ == "__main__":
    unittest.main()
