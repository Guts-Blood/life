#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import day20_eval_preflight_v2 as preflight


class EvalPreflightV2Tests(unittest.TestCase):
    @staticmethod
    def _write_pair(eval_dir: Path, candidate: str, suffix: str, records: int) -> None:
        rows_path = eval_dir / f"{candidate}{suffix}.jsonl"
        rows_path.write_text(
            "".join(json.dumps({"ordinal": index}) + "\n" for index in range(records)),
            encoding="utf-8",
        )
        summary_path = eval_dir / f"{candidate}{suffix.removesuffix('.predictions')}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "predictions": {
                        "records": records,
                        "file_sha256": preflight.file_sha256(rows_path),
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_inference_without_e2b_is_not_selection_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            self._write_pair(eval_dir, "probe-1e-4", ".raw.predictions", 32)
            self._write_pair(eval_dir, "probe-1e-4", ".qwen35-v2.predictions", 32)
            result = preflight.audit_run(
                root, candidates=("probe-1e-4",), expected_records=32
            )
            self.assertEqual(
                result["domain"], "day20.legacy_eval_artifact_inventory.v1"
            )
            self.assertFalse(result["usable_for_v2_selection"])
            self.assertTrue(result["inference_complete"])
            self.assertFalse(result["e2b_complete"])
            self.assertFalse(result["selection_ready"])

    def test_repository_canonical_run_has_inference_but_no_e2b(self) -> None:
        run_root = Path(__file__).resolve().parent / "remote-runs" / "day20-qwen35-lora-20260809T042005Z"
        result = preflight.audit_run(run_root)
        self.assertTrue(result["inference_complete"])
        self.assertFalse(result["e2b_complete"])
        self.assertFalse(result["selection"]["complete"])


if __name__ == "__main__":
    unittest.main()
