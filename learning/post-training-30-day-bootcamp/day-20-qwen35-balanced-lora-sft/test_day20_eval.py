#!/usr/bin/env python3
"""Offline regression tests for Day 20 evaluation and sandbox boundaries."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
BOOTCAMP = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_contract
import evaluate_day20
import prepare_day20
import rescore_day20_qwen35_v2
import score_day20_code_e2b_v2


EVAL_MANIFEST = BOOTCAMP / "artifacts/eval/day10-frozen-eval-manifest.json"
SCORERS = BOOTCAMP / "day-10-frozen-eval-baseline/day10_scorers.py"


class Day20EvaluationContractTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, content: bytes = b"fixture") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _model_tree(self, root: Path) -> tuple[Path, Path]:
        base = root / "base"
        adapter = root / "adapter"
        self._write(base / "config.json", b"{}")
        self._write(base / "model-00001-of-00001.safetensors")
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "peft_type": "LORA",
                }
            ),
            encoding="utf-8",
        )
        self._write(adapter / "adapter_model.safetensors")
        self._write(adapter / "optimizer.pt", b"optimizer-a")
        return base, adapter

    def test_inference_identity_is_separate_from_checkpoint_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, adapter = self._model_tree(Path(temporary))
            first = evaluate_day20.build_model_identity(
                base=base,
                adapter=adapter,
                model_class="Qwen3_5ForConditionalGeneration",
            )
            self.assertEqual(
                rescore_day20_qwen35_v2.verify_model_identity(first),
                first["combined_snapshot_sha256"],
            )
            (adapter / "optimizer.pt").write_bytes(b"optimizer-b")
            second = evaluate_day20.build_model_identity(
                base=base,
                adapter=adapter,
                model_class="Qwen3_5ForConditionalGeneration",
            )
            self.assertEqual(
                first["combined_snapshot_sha256"],
                second["combined_snapshot_sha256"],
            )
            self.assertNotEqual(
                first["adapter"]["checkpoint_package_snapshot_sha256"],
                second["adapter"]["checkpoint_package_snapshot_sha256"],
            )
            tampered = json.loads(json.dumps(second))
            tampered["adapter"]["checkpoint_package_snapshot_sha256"] = "0" * 64
            with self.assertRaises(rescore_day20_qwen35_v2.Day20Qwen35RescoreError):
                rescore_day20_qwen35_v2.verify_model_identity(tampered)

    def test_diagnostic_selection_consumes_prepared_stable_ids(self) -> None:
        frozen = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
        dev = [
            row
            for row in frozen["records"]
            if row.get("evaluation_split") == "dev"
        ]
        selected = prepare_day20.select_diagnostics(dev)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic_path = root / "diagnostic.jsonl"
            diagnostic_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in selected
                ),
                encoding="utf-8",
            )
            immutable = day20_contract.immutable_experiment_contract("/models/base")
            base_files = {
                "config.json": {"bytes": 2, "sha256": "a" * 64},
                "model.safetensors": {"bytes": 7, "sha256": "b" * 64},
            }
            experiment = {
                "schema_version": 1,
                "domain": "day20.qwen35_balanced_lora.experiment_manifest",
                "status": "prepared_probes_not_started",
                "contract": {
                    "immutable": immutable,
                    "immutable_sha256": evaluate_day20.object_sha256(immutable),
                },
                "base_model_identity": {
                    "path": "/models/base",
                    "files": base_files,
                    "snapshot_sha256": evaluate_day20.object_sha256(base_files),
                },
                "datasets": {
                    "diagnostic": {
                        "path": str(diagnostic_path),
                        "file_sha256": evaluate_day20.file_sha256(diagnostic_path),
                        "records": 32,
                        "records_by_skill": {
                            skill: 8 for skill in immutable["data"]["skills"]
                        },
                        "ordered_sample_ids": [row["sample_id"] for row in selected],
                        "selection_sha256": evaluate_day20.object_sha256(
                            [row["sample_id"] for row in selected]
                        ),
                        "source_eval_manifest_sha256": evaluate_day20.file_sha256(
                            EVAL_MANIFEST
                        ),
                    }
                },
            }
            experiment["manifest_sha256"] = evaluate_day20.object_sha256(experiment)
            experiment_path = root / "DAY20-MANIFEST.json"
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            verified = evaluate_day20.load_experiment_manifest(
                experiment_path,
                eval_manifest_path=EVAL_MANIFEST,
                verify_diagnostic_file=True,
            )
            actual = evaluate_day20.selected_experiment_records(
                eval_manifest=frozen,
                experiment_manifest=verified,
                sample_limit=32,
            )
            self.assertEqual(
                [row["sample_id"] for row in actual],
                [row["sample_id"] for row in selected],
            )
            day19_first = evaluate_day20.day19_eval.selected_dev_records(frozen, 32)
            self.assertNotEqual(
                [row["sample_id"] for row in actual],
                [row["sample_id"] for row in day19_first],
            )

    def test_finance_training_target_matches_frozen_scorer(self) -> None:
        scorers = evaluate_day20.day19_eval.load_scorers(SCORERS)
        manifest = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
        finance = next(
            row
            for row in manifest["records"]
            if row.get("evaluation_split") == "dev"
            and row.get("slice") == "finance"
            and json.loads(row["reference"])["answer_type"] in {"arithmetic", "count"}
        )
        reference = json.loads(finance["reference"])
        answer = str(reference["answer"])
        scale = reference["scale"] or "none"
        suffix = "%" if scale == "percent" else (f" {scale}" if scale != "none" else "")
        target = day20_contract.normalize_assistant_target(
            "finance",
            "finance_value_scale",
            f"Final answer: {answer}{suffix}",
            finance_scale=scale,
        )
        self.assertEqual(scorers.score_prediction("finance", target, finance["reference"])["score"], 1.0)

    def test_credential_attestation_is_self_hashed_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attestation.json"
            value = {
                "schema_version": 1,
                "domain": "day20.e2b_credential_attestation",
                "status": "rotated",
                "credential_file_sha256": "a" * 64,
                "created_at_utc": "2026-08-09T00:00:00+00:00",
            }
            value["attestation_sha256"] = score_day20_code_e2b_v2.object_sha256(value)
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)
            identity = score_day20_code_e2b_v2.verify_credential_attestation(path)
            self.assertEqual(identity["status"], "rotated")
            path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
            with self.assertRaises(score_day20_code_e2b_v2.Day20CodeSandboxError):
                score_day20_code_e2b_v2.verify_credential_attestation(path)

    def test_comparison_key_binds_experiment_manifest(self) -> None:
        first = rescore_day20_qwen35_v2.comparison_key(
            manifest_sha="a" * 64,
            experiment_manifest_sha="b" * 64,
            scorer_sha="c" * 64,
        )
        second = rescore_day20_qwen35_v2.comparison_key(
            manifest_sha="a" * 64,
            experiment_manifest_sha="d" * 64,
            scorer_sha="c" * 64,
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
