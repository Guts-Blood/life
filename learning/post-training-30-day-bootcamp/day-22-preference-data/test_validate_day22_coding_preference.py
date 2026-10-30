#!/usr/bin/env python3
"""Focused end-to-end checks for the standalone Day 22 artifact validator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


DAY22_ROOT = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY22_ROOT.parent
VALIDATOR_PATH = (
    BOOTCAMP_ROOT / "artifacts/scripts/validate_day22_coding_preference.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("day22_artifact_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Day 22 artifact validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Day22ArtifactValidatorTest(unittest.TestCase):
    def test_real_bundle_passes_smoke_gate_and_fails_formal_gate(self) -> None:
        validator = _load_validator()
        processor_contract = json.loads(
            validator.DEFAULT_PROCESSOR_CONTRACT.read_text(encoding="utf-8")
        )
        self.assertFalse(Path(processor_contract["snapshot_path"]).is_absolute())
        report = validator.validate_artifacts()
        self.assertEqual(report["status"], "valid_audited_smoke")
        self.assertFalse(report["formal_dpo_ready"])
        self.assertEqual(report["readiness"], "BLOCKED")
        self.assertEqual(report["records"]["pairs"], 63)
        self.assertEqual(report["records"]["quarantine"], 17)
        self.assertEqual(report["records"]["sandbox_evidence"], 80)
        self.assertEqual(report["records"]["processor_evidence"], 80)
        self.assertEqual(
            sum(report["slices"]["relative_response_token_delta"].values()), 63
        )
        for family_overlaps in report["cross_split_overlaps"].values():
            self.assertFalse(any(family_overlaps.values()))

        completed = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "--require-formal-ready"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 3, completed.stderr)
        cli_report = json.loads(completed.stdout)
        self.assertEqual(
            cli_report["status"], "formal_readiness_required_but_blocked"
        )


if __name__ == "__main__":
    unittest.main()
