#!/usr/bin/env python3
"""Integration audit for the independent formal S1 bundle validator."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import formal_s1_pair_assembler as assembler
from test_formal_s1_pair_assembler import fixture


DAY22_ROOT = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY22_ROOT.parent
VALIDATOR_PATH = (
    BOOTCAMP_ROOT / "artifacts/scripts/validate_day22_formal_s1_bundle.py"
)
SMOKE_MANIFEST = (
    BOOTCAMP_ROOT
    / "artifacts/data/day22-qwen35-coding-preference-manifest.json"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "day22_formal_s1_bundle_validator", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load formal S1 bundle validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(assembler.contract.canonical_json(row) + b"\n" for row in rows)


def _write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _build_bundle(root: Path) -> Path:
    data = fixture(assembler.MIN_FORMAL_PAIRS)
    inputs = {
        "selections": root / "formal-selections.jsonl",
        "replay_input": root / "formal-replay.jsonl",
        "selection_summary": root / "formal-selection-summary.json",
        "candidate_evidence": root / "formal-candidate-e2b-evidence.jsonl",
        "processor_audit": root / "formal-processor-audit.jsonl",
        "processor_contract": root / "formal-processor-contract.json",
    }
    payloads = {
        "selections": _jsonl_bytes(data["selections"]),
        "replay_input": _jsonl_bytes(data["replays"]),
        "selection_summary": _json_bytes(data["summary"]),
        "candidate_evidence": _jsonl_bytes(data["evidence"]),
        "processor_audit": _jsonl_bytes(data["audits"]),
        "processor_contract": _json_bytes(data["processor_contract"]),
    }
    for name, path in inputs.items():
        _write(path, payloads[name])
    input_identities = {
        name: {
            "path": path.name,
            "file_sha256": assembler.file_sha256(path),
        }
        for name, path in inputs.items()
    }
    result = assembler.assemble_formal_s1(
        data["selections"],
        data["replays"],
        data["summary"],
        data["evidence"],
        data["audits"],
        data["processor_contract"],
        input_identities=input_identities,
    )
    outputs = {
        root / "formal-pairs.jsonl": _jsonl_bytes(result["pairs"]),
        root / "formal-split-ids.json": _json_bytes(result["split_ids"]),
        root / "formal-machine-gate-audit.json": _json_bytes(result["audit"]),
        root / "formal-review-worksheet.jsonl": _jsonl_bytes(
            result["review_worksheet"]
        ),
        root / "formal-review-key.json": _json_bytes(result["review_key"]),
    }
    for path, payload in outputs.items():
        _write(path, payload)
    manifest = copy.deepcopy(result["manifest"])
    manifest.pop("manifest_sha256", None)
    manifest["output_files"] = {
        path.name: {
            "path": path.name,
            "file_sha256": assembler.file_sha256(path),
        }
        for path in outputs
    }
    manifest = assembler._seal(manifest, "manifest_sha256")
    manifest_path = root / "formal-manifest.json"
    _write(manifest_path, _json_bytes(manifest))
    return manifest_path


class FormalS1BundleValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.manifest = _build_bundle(cls.root)
        cls.validator = _load_validator()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_machine_gates_pass_and_only_human_review_blocks_formal_ready(self) -> None:
        report = self.validator.validate_formal_bundle(self.manifest)
        self.assertEqual(
            report["status"], "valid_formal_s1_bundle_human_review_pending"
        )
        self.assertEqual(report["machine_gate_status"], "PASS")
        self.assertTrue(report["machine_ready"])
        self.assertFalse(report["formal_dpo_ready"])
        self.assertEqual(report["formal_readiness"], "BLOCKED")
        self.assertEqual(
            report["formal_dpo_blockers"], [assembler.PENDING_BLOCKER]
        )
        self.assertEqual(report["records"]["formal_pairs"], 200)
        self.assertEqual(report["records"]["candidate_e2b_evidence"], 400)
        self.assertEqual(report["records"]["processor_audits"], 200)
        self.assertEqual(report["records"]["review_unique_pairs"], 50)
        self.assertEqual(report["records"]["review_completed_unique_pairs"], 0)
        self.assertFalse(any(report["cross_split_family_overlap"].values()))

        default = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "--manifest", str(self.manifest)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(json.loads(default.stdout)["machine_gate_status"], "PASS")

        formal = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR_PATH),
                "--manifest",
                str(self.manifest),
                "--require-formal-ready",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(formal.returncode, 3, formal.stderr)
        self.assertEqual(
            json.loads(formal.stdout)["status"],
            "formal_readiness_required_but_human_review_pending",
        )

    def test_smoke_manifest_is_not_a_formal_bundle(self) -> None:
        with self.assertRaisesRegex(
            self.validator.FormalS1ValidationError,
            "formal manifest schema drifted",
        ):
            self.validator.validate_formal_bundle(SMOKE_MANIFEST)


if __name__ == "__main__":
    unittest.main()
