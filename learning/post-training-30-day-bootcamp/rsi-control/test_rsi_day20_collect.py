#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import rsi_day20_collect as collector


class Day20EvidenceCollectorTests(unittest.TestCase):
    candidate = "probe-s20260809-lr1e-4-t6000"
    normalized_key = "a" * 64
    e2b_key = "b" * 64
    complete_key = "c" * 64

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "day20-v2-qwen35-lora-fixture"
        (self.root / "evidence").mkdir(parents=True)
        (self.root / "eval").mkdir()
        (self.root / collector.RUN_MARKER).write_text(
            collector.RUN_MARKER_VALUE + "\n", encoding="utf-8"
        )
        self.calls: list[str] = []
        self.summaries: dict[str, dict] = {}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write(path: Path, value: str = "fixture\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def verifier_bundle(self, *, raw_error: Exception | None = None):
        def training(path, *, run_root):
            self.calls.append("training")
            self.assertEqual(run_root, self.root.resolve())
            return self.summaries["training"]

        def raw(path, summary_path, *, expected_candidate, expected_scope):
            self.calls.append("raw")
            if raw_error is not None:
                raise raw_error
            self.assertEqual(expected_candidate, self.candidate)
            self.assertEqual(expected_scope, "probe32")
            return self.summaries["raw"]

        def normalized(path, summary_path, *, expected_candidate):
            self.calls.append("normalized")
            self.assertEqual(expected_candidate, self.candidate)
            return [], self.summaries["normalized"]

        def e2b(path, *, expected_candidate):
            self.calls.append("e2b")
            self.assertEqual(expected_candidate, self.candidate)
            return [], self.summaries["e2b"], [], self.summaries["normalized"]

        def probe_selection(path):
            self.calls.append("probe_selection")
            return self.summaries["probe_selection"]

        def unused(path):
            raise AssertionError(f"unexpected verifier call for {path}")

        return collector.VerifierBundle(
            training=training,
            raw=raw,
            normalized=normalized,
            e2b=e2b,
            probe_selection=probe_selection,
            primary_selection=unused,
            final_promotion=unused,
        )

    def build_complete_fixture(self) -> None:
        run_dir = self.root / "evidence/probe/s20260809/lr1e-4"
        training_summary = self.write(run_dir / "training-summary.json")
        checkpoint = self.root / "adapters/probe/checkpoint-1"
        integrity = self.write(
            checkpoint / collector.CHECKPOINT_INTEGRITY_FILE, "integrity\n"
        )
        self.summaries["training"] = {
            "run_kind": "probe",
            "candidate_ids": {"t6000": self.candidate},
            "checkpoints": {"t6000": str(checkpoint.resolve())},
            "checkpoint_tokens": {"t6000": 6000},
            "checkpoint_packages": {
                "t6000": {
                    "integrity_file_sha256": collector.file_sha256(integrity),
                    "integrity_sha256": "1" * 64,
                    "snapshot_sha256": "2" * 64,
                }
            },
            "summary_sha256": "3" * 64,
        }

        raw_predictions = self.write(
            self.root
            / f"eval/{self.candidate}-probe32-raw-predictions-v2.jsonl",
            "{}\n",
        )
        self.write(
            self.root / f"eval/{self.candidate}-probe32-raw-summary-v2.json"
        )
        self.summaries["raw"] = {
            "predictions": {
                "file_sha256": collector.file_sha256(raw_predictions),
                "content_sha256": "4" * 64,
            },
            "summary_sha256": "5" * 64,
            "metrics": {"records": 32},
            "runtime_metrics": {"elapsed_seconds": 1.0},
        }

        normalized_predictions = self.write(
            self.root / f"eval/{self.candidate}.qwen35-v3.predictions.jsonl",
            "{}\n",
        )
        self.write(self.root / f"eval/{self.candidate}.qwen35-v3.json")
        self.summaries["normalized"] = {
            "candidate": self.candidate,
            "scope": "probe32",
            "normalized_comparison_key": self.normalized_key,
            "evaluation_run_sha256": "6" * 64,
            "predictions": {
                "file_sha256": collector.file_sha256(normalized_predictions),
                "content_sha256": "7" * 64,
            },
            "summary_sha256": "8" * 64,
            "metrics": {"total": 12},
        }

        e2b_results = self.write(
            self.root / f"eval/{self.candidate}-code-e2b-qwen35-v3.jsonl",
            "{}\n",
        )
        self.write(
            self.root
            / f"eval/{self.candidate}-code-e2b-qwen35-v3-summary.json"
        )
        self.summaries["e2b"] = {
            "candidate": self.candidate,
            "scope": "probe32",
            "normalized_comparison_key": self.normalized_key,
            "e2b_comparison_key": self.e2b_key,
            "complete_comparison_key": self.complete_key,
            "evaluation_run_sha256": "6" * 64,
            "e2b_run_sha256": "9" * 64,
            "result": {
                "file_sha256": collector.file_sha256(e2b_results),
                "content_sha256": "d" * 64,
            },
            "summary_sha256": "e" * 64,
            "records": 8,
            "code_records": 8,
            "sandbox_execution_eligible": 7,
            "passed": 6,
            "failed": 1,
            "infrastructure_failures": 0,
        }

        self.write(self.root / "PROBE-SELECTION-STAGE-A.json")
        self.summaries["probe_selection"] = {
            "status": "selected",
            "selected_candidate": self.candidate,
            "selection_sha256": "f" * 64,
            "common_comparison": {
                "scope": "probe32",
                "normalized_comparison_key": self.normalized_key,
                "e2b_comparison_key": self.e2b_key,
                "complete_comparison_key": self.complete_key,
            },
        }

    @staticmethod
    def tree_snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    def test_collects_only_verified_items_and_is_deterministic_and_read_only(self):
        self.build_complete_fixture()
        before = self.tree_snapshot(self.root)
        first = collector.collect_evidence(
            self.root, verifiers=self.verifier_bundle()
        )
        second = collector.collect_evidence(
            self.root, verifiers=self.verifier_bundle()
        )

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(
            first["counts"],
            {
                "checkpoint": 1,
                "normalized_evaluation": 1,
                "probe_selection": 1,
                "raw_evaluation": 1,
                "sandbox": 1,
            },
        )
        self.assertEqual(first["operational_incomplete"], [])
        self.assertEqual(first["verification_failures"], [])
        self.assertEqual(before, self.tree_snapshot(self.root))
        self.assertEqual(
            first["inventory_sha256"],
            collector.object_sha256(
                {key: value for key, value in first.items() if key != "inventory_sha256"}
            ),
        )
        for item in first["complete_verified_items"]:
            self.assertEqual(
                item["evidence_sha256"],
                collector.object_sha256(
                    {key: value for key, value in item.items() if key != "evidence_sha256"}
                ),
            )
            if item["candidate"] == self.candidate:
                self.assertEqual(
                    item["comparison_keys"],
                    {
                        "normalized": self.normalized_key,
                        "e2b": self.e2b_key,
                        "complete": self.complete_key,
                    },
                )
        self.assertEqual(
            self.calls.count("training"), 2, "each inventory must reverify evidence"
        )

    def test_partial_artifacts_are_operational_incomplete_and_not_collected(self):
        (self.root / "evidence/probe/s20260809/lr1e-4/attempt-001").mkdir(
            parents=True
        )
        self.write(
            self.root
            / f"eval/{self.candidate}-probe32-raw-predictions-v2.jsonl"
        )
        self.write(self.root / f"eval/{self.candidate}.qwen35-v3.json")
        self.write(
            self.root / f"eval/{self.candidate}-code-e2b-qwen35-v3.jsonl"
        )
        self.write(
            self.root
            / f"eval/.{self.candidate}-probe32-raw-predictions-v2.jsonl.123.attempt"
        )

        result = collector.collect_evidence(
            self.root, verifiers=self.verifier_bundle()
        )

        self.assertEqual(result["status"], "operational_incomplete")
        self.assertEqual(result["complete_verified_items"], [])
        self.assertEqual(result["verification_failures"], [])
        self.assertEqual(len(result["operational_incomplete"]), 5)
        self.assertEqual(self.calls, [])

    def test_verifier_rejection_is_an_integrity_failure_not_a_completed_item(self):
        predictions = self.write(
            self.root
            / f"eval/{self.candidate}-probe32-raw-predictions-v2.jsonl"
        )
        summary = self.write(
            self.root / f"eval/{self.candidate}-probe32-raw-summary-v2.json"
        )

        result = collector.collect_evidence(
            self.root,
            verifiers=self.verifier_bundle(
                raw_error=ValueError("summary_sha256 mismatch")
            ),
        )

        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["complete_verified_items"], [])
        self.assertEqual(result["operational_incomplete"], [])
        self.assertEqual(
            result["verification_failures"],
            [
                {
                    "artifact_type": "raw_evaluation",
                    "paths": sorted(
                        [str(predictions.resolve()), str(summary.resolve())]
                    ),
                    "failure_layer": "evidence_integrity",
                    "error_type": "ValueError",
                    "message": "summary_sha256 mismatch",
                }
            ],
        )

    def test_cli_rejects_non_day20_directory_with_structured_failure(self):
        invalid = Path(self.temporary.name) / "not-a-run"
        invalid.mkdir()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = collector.main(["--run-root", str(invalid)])
        result = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["failure_layer"], "collector_input")


if __name__ == "__main__":
    unittest.main()
