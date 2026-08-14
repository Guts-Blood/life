#!/usr/bin/env python3
"""Integration tests for the explicit Day 22 experimental close."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import finalize_day22_experimental_close as closer


BOOTCAMP_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = BOOTCAMP_ROOT / "artifacts"
MANIFEST = ARTIFACTS / "data/day22-qwen35-experimental-ai-assisted-manifest.json"
PACKET = ARTIFACTS / "data/day22-qwen35-experimental-codex-adjudication-packet.jsonl"
SCORES = ARTIFACTS / "data/day22-qwen35-experimental-codex-adjudication-scores.jsonl"


def load_validator():
    path = ARTIFACTS / "scripts/validate_day22_experimental_close.py"
    spec = importlib.util.spec_from_file_location("day22_experimental_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentalCloseTest(unittest.TestCase):
    def command(
        self,
        output_root: Path,
        scores: Path = SCORES,
        pairs_output: Path | None = None,
    ) -> list[str]:
        command = [
            "--formal-manifest",
            str(ARTIFACTS / "data/day22-qwen35-formal-s1-manifest.json"),
            "--formal-pairs",
            str(ARTIFACTS / "data/day22-qwen35-formal-s1-preference-pairs.jsonl"),
            "--formal-split-ids",
            str(ARTIFACTS / "data/day22-qwen35-formal-s1-split-ids.json"),
            "--ai-audit",
            str(ARTIFACTS / "reports/day22-qwen35-formal-s1-ai-assisted-review-audit.json"),
            "--blank-worksheet",
            str(ARTIFACTS / "data/day22-qwen35-formal-s1-blind-review.jsonl"),
            "--concealed-key",
            str(ARTIFACTS / "data/day22-qwen35-formal-s1-blind-review-key.json"),
            "--adjudication-packet",
            str(PACKET),
            "--adjudication-scores",
            str(scores),
            "--split-ids-output",
            str(output_root / "split.json"),
            "--manifest-output",
            str(output_root / "manifest.json"),
            "--report-output",
            str(output_root / "report.md"),
        ]
        if pairs_output is not None:
            command.extend(["--pairs-output", str(pairs_output)])
        return command

    def test_real_scores_and_bundle_validate(self) -> None:
        closer.validate_scores(closer.load_jsonl(PACKET), closer.load_jsonl(SCORES))
        result = load_validator().validate_manifest(MANIFEST)
        self.assertTrue(result["experimental_dpo_ready"])
        self.assertFalse(result["formal_dpo_ready"])
        self.assertEqual(result["experimental_pairs"], 200)
        self.assertEqual(result["adjudication_keep"], 11)
        self.assertEqual(result["adjudication_exclude"], 0)

    def test_reproducible_close_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            root = Path(temporary)
            self.assertEqual(closer.main(self.command(root)), 0)
            generated = closer.load_json(root / "manifest.json")
            frozen = closer.load_json(MANIFEST)
            self.assertEqual(generated["counts"], frozen["counts"])
            self.assertEqual(
                generated["content_identities"], frozen["content_identities"]
            )
            self.assertEqual(
                generated["outputs"]["pairs"]["path"],
                "artifacts/data/day22-qwen35-formal-s1-preference-pairs.jsonl",
            )
            self.assertEqual(
                generated["outputs"]["pairs"]["file_sha256"],
                closer.file_sha256(
                    ARTIFACTS / "data/day22-qwen35-formal-s1-preference-pairs.jsonl"
                ),
            )
            self.assertFalse((root / "pairs.jsonl").exists())

    def test_preference_for_verifier_rejected_excludes_without_flipping(self) -> None:
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            root = Path(temporary)
            scores = closer.load_jsonl(SCORES)
            key = closer.load_json(
                ARTIFACTS / "data/day22-qwen35-formal-s1-blind-review-key.json"
            )
            key_by_id = {row["review_item_id"]: row for row in key["items"]}
            first = scores[0]
            item = key_by_id[first["review_item_id"]]
            rejected_decision = (
                "prefer_A" if item["a_side"] == "rejected" else "prefer_B"
            )
            first["decision"] = rejected_decision
            if rejected_decision == "prefer_A":
                first["scores_a"]["semantic_correctness"] = 5
                first["scores_a"]["public_test_alignment"] = 2
                first["scores_a"]["instruction_compliance"] = 1
                first["scores_a"]["robustness"] = 2
                first["scores_a"]["total"] = 10
                first["scores_b"]["total"] = sum(
                    first["scores_b"][field]
                    for field in closer.SCORE_LIMITS
                )
            else:
                first["scores_b"]["semantic_correctness"] = 5
                first["scores_b"]["public_test_alignment"] = 2
                first["scores_b"]["instruction_compliance"] = 1
                first["scores_b"]["robustness"] = 2
                first["scores_b"]["total"] = 10
                first["scores_a"]["total"] = sum(
                    first["scores_a"][field]
                    for field in closer.SCORE_LIMITS
                )
            score_path = root / "scores.jsonl"
            score_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in scores
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                closer.main(
                    self.command(root, score_path, pairs_output=root / "pairs.jsonl")
                ),
                0,
            )
            result = closer.load_json(root / "manifest.json")
            self.assertEqual(result["counts"]["experimental_pairs"], 199)
            self.assertEqual(result["counts"]["adjudication_exclude"], 1)
            self.assertFalse(result["policy"]["label_flip_allowed"])

    def test_exclusion_without_pairs_output_fails_without_mutating_formal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            root = Path(temporary)
            scores = closer.load_jsonl(SCORES)
            scores[0]["decision"] = "exclude"
            scores[0]["issue_type"] = "ambiguous_spec"
            score_path = root / "scores.jsonl"
            score_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in scores
                ),
                encoding="utf-8",
            )
            formal = ARTIFACTS / "data/day22-qwen35-formal-s1-preference-pairs.jsonl"
            before = closer.file_sha256(formal)
            self.assertEqual(closer.main(self.command(root, score_path)), 2)
            self.assertEqual(closer.file_sha256(formal), before)
            self.assertFalse((root / "manifest.json").exists())

    def test_validator_rejects_resealed_status_tamper(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            path = Path(temporary) / "manifest.json"
            value = closer.load_json(MANIFEST)
            value["status"] = "formal_ready"
            value["manifest_sha256"] = closer.object_sha256(value, "manifest_sha256")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(validator.ExperimentalValidationError):
                validator.validate_manifest(path)

    def test_validator_rejects_zero_exclusion_pairs_copy(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            root = Path(temporary)
            copied_pairs = root / "pairs-copy.jsonl"
            formal = ARTIFACTS / "data/day22-qwen35-formal-s1-preference-pairs.jsonl"
            copied_pairs.write_bytes(formal.read_bytes())
            path = root / "manifest.json"
            value = closer.load_json(MANIFEST)
            value["outputs"]["pairs"]["path"] = closer.relative_path(copied_pairs)
            value["manifest_sha256"] = closer.object_sha256(value, "manifest_sha256")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(validator.ExperimentalValidationError):
                validator.validate_manifest(path)


if __name__ == "__main__":
    unittest.main()
