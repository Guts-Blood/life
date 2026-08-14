#!/usr/bin/env python3
"""Focused lifecycle tests for the Day 22 human-review finalizer."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import finalize_day22_formal_s1_review as finalizer
from test_validate_day22_formal_s1_bundle import VALIDATOR_PATH, _build_bundle


FINALIZER_PATH = Path(finalizer.__file__).resolve()


def _load_validator():
    spec = importlib.util.spec_from_file_location("ready_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ready validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class FormalS1ReviewFinalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.pending_manifest = _build_bundle(cls.root)
        cls.blank_path = cls.root / "formal-review-worksheet.jsonl"
        cls.key_path = cls.root / "formal-review-key.json"
        cls.blank = [json.loads(line) for line in cls.blank_path.read_text().splitlines()]
        cls.key = json.loads(cls.key_path.read_text())
        cls.key_by_id = {item["review_item_id"]: item for item in cls.key["items"]}
        cls.validator = _load_validator()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def completed_chosen_rows(self) -> list[dict]:
        rows = copy.deepcopy(self.blank)
        for row in rows:
            key = self.key_by_id[row["review_item_id"]]
            row["verdict"] = "A" if key["a_side"] == "chosen" else "B"
            row["confidence"] = "high"
        return rows

    def test_all_position_consistent_chosen_reviews_emit_valid_ready_wrapper(self) -> None:
        completed = self.root / "completed-ready.jsonl"
        audit_path = self.root / "review-ready-audit.json"
        ready_path = self.root / "formal-ready-manifest.json"
        _write_jsonl(completed, self.completed_chosen_rows())
        audit, ready = finalizer.finalize_review(
            self.pending_manifest,
            completed,
            review_audit_output_path=audit_path,
        )
        self.assertTrue(audit["formal_dpo_ready"])
        self.assertEqual(audit["counts"]["excluded_reviewed_pairs"], 0)
        self.assertIsNotNone(ready)
        _write_json(audit_path, audit)
        _write_json(ready_path, ready)
        report = self.validator.validate_bundle(ready_path)
        self.assertTrue(report["formal_dpo_ready"])
        self.assertEqual(report["status"], "valid_formal_s1_bundle_ready")

    def test_one_position_swap_disagreement_stays_blocked(self) -> None:
        rows = self.completed_chosen_rows()
        first_pair = self.key["items"][0]["pair_id"]
        item_ids = [
            item["review_item_id"]
            for item in self.key["items"]
            if item["pair_id"] == first_pair
        ]
        row = next(row for row in rows if row["review_item_id"] == item_ids[0])
        row["verdict"] = "B" if row["verdict"] == "A" else "A"
        completed = self.root / "completed-disagreement.jsonl"
        _write_jsonl(completed, rows)
        audit, ready = finalizer.finalize_review(
            self.pending_manifest,
            completed,
            review_audit_output_path=self.root / "blocked-audit.json",
        )
        self.assertFalse(audit["formal_dpo_ready"])
        self.assertIsNone(ready)
        self.assertEqual(audit["counts"]["formal_pairs_after_review"], 199)
        self.assertEqual(audit["exclusion_reason_counts"]["position_swap_disagreement"], 1)

    def test_incomplete_or_tampered_completed_worksheet_fails_closed(self) -> None:
        rows = self.completed_chosen_rows()
        incomplete = self.root / "completed-incomplete.jsonl"
        _write_jsonl(incomplete, rows[:-1])
        with self.assertRaisesRegex(finalizer.ReviewFinalizationError, "row count"):
            finalizer.finalize_review(self.pending_manifest, incomplete)

        rows[0]["prompt"] += " tampered"
        tampered = self.root / "completed-tampered.jsonl"
        _write_jsonl(tampered, rows)
        with self.assertRaisesRegex(finalizer.ReviewFinalizationError, "prompt drifted"):
            finalizer.finalize_review(self.pending_manifest, tampered)

    def test_non_directional_verdict_requires_notes_and_blocks(self) -> None:
        rows = self.completed_chosen_rows()
        rows[0]["verdict"] = "ambiguous"
        rows[0]["notes"] = ""
        invalid = self.root / "completed-ambiguous-no-notes.jsonl"
        _write_jsonl(invalid, rows)
        with self.assertRaisesRegex(finalizer.ReviewFinalizationError, "notes is required"):
            finalizer.finalize_review(self.pending_manifest, invalid)

    def test_independent_validator_rejects_forged_resealed_audit(self) -> None:
        completed = self.root / "completed-forged-audit.jsonl"
        audit_path = self.root / "forged-review-audit.json"
        ready_path = self.root / "forged-ready-manifest.json"
        _write_jsonl(completed, self.completed_chosen_rows())
        audit, ready = finalizer.finalize_review(
            self.pending_manifest,
            completed,
            review_audit_output_path=audit_path,
        )
        audit["rates"]["position_consistency"] = 0.95
        audit.pop("review_audit_sha256")
        audit["review_audit_sha256"] = self.validator.object_sha256(audit)
        _write_json(audit_path, audit)
        ready["rates"] = copy.deepcopy(audit["rates"])
        ready["review_audit"]["file_sha256"] = self.validator.file_sha256(audit_path)
        ready["review_audit"]["review_audit_sha256"] = audit["review_audit_sha256"]
        ready.pop("ready_manifest_sha256")
        ready["ready_manifest_sha256"] = self.validator.object_sha256(ready)
        _write_json(ready_path, ready)
        with self.assertRaisesRegex(
            self.validator.FormalS1ValidationError, "review audit rates drifted"
        ):
            self.validator.validate_bundle(ready_path)

    def test_independent_validator_rejects_resealed_prompt_tamper(self) -> None:
        completed = self.root / "completed-forged-prompt.jsonl"
        audit_path = self.root / "prompt-review-audit.json"
        ready_path = self.root / "prompt-ready-manifest.json"
        rows = self.completed_chosen_rows()
        _write_jsonl(completed, rows)
        audit, ready = finalizer.finalize_review(
            self.pending_manifest,
            completed,
            review_audit_output_path=audit_path,
        )
        rows[0]["prompt"] += " forged"
        _write_jsonl(completed, rows)
        audit["completed_worksheet_sha256"] = self.validator.file_sha256(completed)
        audit.pop("review_audit_sha256")
        audit["review_audit_sha256"] = self.validator.object_sha256(audit)
        _write_json(audit_path, audit)
        ready["completed_worksheet"]["file_sha256"] = self.validator.file_sha256(completed)
        ready["review_audit"]["file_sha256"] = self.validator.file_sha256(audit_path)
        ready["review_audit"]["review_audit_sha256"] = audit["review_audit_sha256"]
        ready.pop("ready_manifest_sha256")
        ready["ready_manifest_sha256"] = self.validator.object_sha256(ready)
        _write_json(ready_path, ready)
        with self.assertRaisesRegex(
            self.validator.FormalS1ValidationError, "prompt drifted"
        ):
            self.validator.validate_bundle(ready_path)

    def test_cli_preflights_both_outputs_before_writing(self) -> None:
        completed = self.root / "completed-preflight.jsonl"
        audit_path = self.root / "preflight-audit-must-not-appear.json"
        ready_path = self.root / "preflight-ready-already-exists.json"
        _write_jsonl(completed, self.completed_chosen_rows())
        ready_path.write_text("sentinel\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(FINALIZER_PATH),
                "--pending-manifest",
                str(self.pending_manifest),
                "--completed-worksheet",
                str(completed),
                "--review-audit-output",
                str(audit_path),
                "--ready-manifest-output",
                str(ready_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(audit_path.exists())
        self.assertEqual(ready_path.read_text(encoding="utf-8"), "sentinel\n")


if __name__ == "__main__":
    unittest.main()
