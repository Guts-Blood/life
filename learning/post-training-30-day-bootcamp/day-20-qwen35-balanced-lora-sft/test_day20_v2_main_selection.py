#!/usr/bin/env python3
"""Offline linked-artifact tests for Day 20 main selection and confirmation."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import day20_candidate_factory_v2 as factory
import day20_eval_identity_v2 as identity
import select_day20_main_v2 as selector


COMMON_KEYS = {
    "normalized_comparison_key": "a" * 64,
    "e2b_comparison_key": "b" * 64,
    "complete_comparison_key": "c" * 64,
}


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bundles: dict[Path, tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}

    @staticmethod
    def candidate(seed: int, checkpoint: str, lr: str = "6e-5") -> str:
        return factory.candidate_id(
            run_kind="main",
            seed=seed,
            learning_rate=lr,
            checkpoint=checkpoint,
        )

    def add(
        self,
        candidate: str,
        *,
        general: int,
        math: int,
        finance: int,
        code: int,
        eligible: int = 28,
        format_compliant: int = 112,
        common_keys: dict[str, str] | None = None,
    ) -> Path:
        keys = dict(COMMON_KEYS if common_keys is None else common_keys)
        rows: list[dict[str, Any]] = []
        code_ids: list[str] = []
        code_index = 0
        limits = {"general": general, "math": math, "finance": finance}
        ordinal = 0
        for skill in factory.SKILLS:
            for index in range(28):
                ordinal += 1
                sample_id = f"eval:{skill}:{index:02d}"
                row: dict[str, Any] = {
                    "ordinal": ordinal,
                    "sample_id": sample_id,
                    "slice": skill,
                    "format_compliant": ordinal <= format_compliant,
                    "normalized_scorer_result": {
                        "score": None if skill == "code" else float(index < limits[skill])
                    },
                    "sandbox_execution_eligible": None,
                }
                if skill == "code":
                    row["sandbox_execution_eligible"] = code_index < eligible
                    code_ids.append(sample_id)
                    code_index += 1
                rows.append(row)
        eligible_ids = code_ids[:eligible]
        results = [
            {
                "sample_id": sample_id,
                "score": float(index < code),
                "execution_status": "passed" if index < code else "failed",
            }
            for index, sample_id in enumerate(eligible_ids)
        ]
        context = {"scope": "full112", "code_sample_ids": code_ids}
        summary: dict[str, Any] = {
            "schema_version": 2,
            "domain": "day20.v2.code_e2b_summary",
            "status": "complete",
            "candidate": candidate,
            "scope": "full112",
            "records": len(results),
            "code_records": 28,
            "sandbox_execution_eligible": eligible,
            "passed": code,
            "failed": len(results) - code,
            "infrastructure_failures": 0,
            "e2b_comparison_context": context,
            **keys,
            "e2b_run_sha256": identity.object_sha256(
                {"candidate": candidate, "run": "fixture"}
            ),
        }
        summary["summary_sha256"] = identity.object_sha256(summary)
        path = self.root / f"{candidate}-e2b-summary.json"
        path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        normalized_summary = {
            "candidate": candidate,
            "scope": "full112",
            "metrics": {"infrastructure_failures": 0},
        }
        self.bundles[path.resolve()] = (
            results,
            summary,
            rows,
            normalized_summary,
        )
        return path

    def verify(
        self,
        path: Path,
        *,
        expected_candidate: str | None = None,
        expected_scope: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        try:
            bundle = self.bundles[path.resolve()]
        except KeyError as error:
            raise ValueError("unknown E2B fixture") from error
        declared = json.loads(path.read_text(encoding="utf-8"))
        if declared != bundle[1]:
            raise ValueError("E2B fixture file drifted")
        if expected_candidate is not None and declared["candidate"] != expected_candidate:
            raise ValueError("candidate drifted")
        if expected_scope is not None and declared["scope"] != expected_scope:
            raise ValueError("scope drifted")
        return copy.deepcopy(bundle)

    def primary_inputs(
        self, *, passing: bool = True
    ) -> tuple[Path, list[Path], str]:
        base = self.add(
            "base-full", general=10, math=20, finance=15, code=19
        )
        if passing:
            counts = {
                "early": (10, 20, 15, 20),
                "mid": (12, 20, 15, 20),
                "final": (12, 20, 15, 20),
            }
        else:
            counts = {
                "early": (5, 16, 10, 13),
                "mid": (5, 16, 10, 13),
                "final": (5, 16, 10, 13),
            }
        candidates = []
        for checkpoint, values in counts.items():
            candidate = self.candidate(factory.PRIMARY_SEED, checkpoint)
            candidates.append(
                self.add(
                    candidate,
                    general=values[0],
                    math=values[1],
                    finance=values[2],
                    code=values[3],
                )
            )
        winner = self.candidate(factory.PRIMARY_SEED, "mid")
        return base, candidates, winner


class MainSelectionV2Tests(unittest.TestCase):
    def test_primary_ranking_and_existing_evidence_reverification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            base, candidates, expected_winner = fixture.primary_inputs()
            output = Path(temporary) / "evidence"
            selection = selector.publish_primary_selection(
                base_summary_path=base,
                primary_summary_paths=list(reversed(candidates)),
                output_dir=output,
                verifier=fixture.verify,
            )
            self.assertEqual(selection["decision"]["primary_winner"], expected_winner)
            self.assertEqual(selection["decision"]["action"], "run_confirmation")
            self.assertTrue(selection["decision"]["confirmation_required"])
            self.assertEqual(
                selector.publish_primary_selection(
                    base_summary_path=base,
                    primary_summary_paths=candidates,
                    output_dir=output,
                    verifier=fixture.verify,
                ),
                selection,
            )
            self.assertFalse(selection["merge_performed"])

    def test_confirmation_pass_promotes_primary_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            base, candidates, expected_winner = fixture.primary_inputs()
            output = Path(temporary) / "evidence"
            selector.publish_primary_selection(
                base_summary_path=base,
                primary_summary_paths=candidates,
                output_dir=output,
                verifier=fixture.verify,
            )
            confirmation_id = fixture.candidate(
                factory.CONFIRMATION_SEED, "mid"
            )
            confirmation = fixture.add(
                confirmation_id,
                general=11,
                math=19,
                finance=15,
                code=20,
            )
            final = selector.publish_final_promotion(
                primary_selection_path=output / selector.PRIMARY_FILENAME,
                confirmation_summary_path=confirmation,
                output_dir=output,
                verifier=fixture.verify,
            )
            self.assertEqual(final["decision"]["action"], "promote_primary_checkpoint")
            self.assertEqual(final["decision"]["selected_candidate"], expected_winner)
            self.assertEqual(
                final["decision"]["confirmation_candidate"], confirmation_id
            )
            self.assertFalse(final["merge_performed"])

    def test_confirmation_floor_failure_falls_back_to_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            base, candidates, _ = fixture.primary_inputs()
            output = Path(temporary) / "evidence"
            selector.publish_primary_selection(
                base_summary_path=base,
                primary_summary_paths=candidates,
                output_dir=output,
                verifier=fixture.verify,
            )
            confirmation = fixture.add(
                fixture.candidate(factory.CONFIRMATION_SEED, "mid"),
                general=5,
                math=16,
                finance=10,
                code=13,
            )
            final = selector.publish_final_promotion(
                primary_selection_path=output / selector.PRIMARY_FILENAME,
                confirmation_summary_path=confirmation,
                output_dir=output,
                verifier=fixture.verify,
            )
            self.assertEqual(final["decision"]["action"], "base_fallback")
            self.assertEqual(final["decision"]["selected_candidate"], "base-full")

    def test_no_primary_candidate_needs_no_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            base, candidates, _ = fixture.primary_inputs(passing=False)
            output = Path(temporary) / "evidence"
            primary = selector.publish_primary_selection(
                base_summary_path=base,
                primary_summary_paths=candidates,
                output_dir=output,
                verifier=fixture.verify,
            )
            self.assertEqual(primary["decision"]["action"], "base_fallback")
            self.assertFalse(primary["decision"]["confirmation_required"])
            final = selector.publish_final_promotion(
                primary_selection_path=output / selector.PRIMARY_FILENAME,
                confirmation_summary_path=None,
                output_dir=output,
                verifier=fixture.verify,
            )
            self.assertEqual(final["decision"]["selected_candidate"], "base-full")

    def test_lr_checkpoint_common_key_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            base, candidates, _ = fixture.primary_inputs()
            wrong_lr = fixture.add(
                fixture.candidate(factory.PRIMARY_SEED, "final", lr="8e-5"),
                general=12,
                math=20,
                finance=15,
                code=20,
            )
            with self.assertRaises(selector.Day20MainSelectionV2Error):
                selector.build_primary_selection(
                    base_summary_path=base,
                    primary_summary_paths=[candidates[0], candidates[1], wrong_lr],
                    verifier=fixture.verify,
                )

            output = Path(temporary) / "evidence"
            selector.publish_primary_selection(
                base_summary_path=base,
                primary_summary_paths=candidates,
                output_dir=output,
                verifier=fixture.verify,
            )
            selection_path = output / selector.PRIMARY_FILENAME
            tampered = json.loads(selection_path.read_text(encoding="utf-8"))
            tampered["decision"]["primary_winner"] = candidates[0].stem
            tampered["selection_sha256"] = identity.object_sha256(
                {key: value for key, value in tampered.items() if key != "selection_sha256"}
            )
            selection_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(selector.Day20MainSelectionV2Error):
                selector.verify_primary_selection(
                    selection_path, verifier=fixture.verify
                )


if __name__ == "__main__":
    unittest.main()
