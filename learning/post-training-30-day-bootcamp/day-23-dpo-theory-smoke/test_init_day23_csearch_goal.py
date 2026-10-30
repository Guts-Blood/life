#!/usr/bin/env python3
"""Focused stdlib tests for the isolated Day 23 candidate-search initializer."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


DAY23_DIR = Path(__file__).resolve().parent
if str(DAY23_DIR) not in sys.path:
    sys.path.insert(0, str(DAY23_DIR))

import day23_contract  # noqa: E402
import init_day23_csearch_goal as initializer  # noqa: E402


def _row(pair_id: str) -> dict:
    value = {"pair_id": pair_id}
    value["row_sha256"] = day23_contract.object_sha256(value)
    return value


def _write_rows(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path),
        "file_sha256": initializer.day23_gpu.file_sha256(path),
        "bytes": path.stat().st_size,
        "records": len(rows),
        "ordered_pair_ids_sha256": day23_contract.object_sha256([row["pair_id"] for row in rows]),
        "ordered_row_hashes_sha256": day23_contract.object_sha256([row["row_sha256"] for row in rows]),
    }


def _fake_identity(path: str, digest: str) -> dict:
    return {"path": path, "file_sha256": digest, "bytes": 1, "content_sha256": digest}


class CSearchInitializerTests(unittest.TestCase):
    def _rotation_fixture(self, root: Path) -> tuple[dict, dict, dict]:
        mechanism = list(initializer.rsi_eval.MECHANISM_PAIR_IDS)
        v4_ids = [f"v4-{index:03d}" for index in range(30)]
        v5_ids = [f"v5-{index:03d}" for index in range(30)]
        eligible = [f"fresh-{index:03d}" for index in range(90)]
        full_rows = [_row(pair_id) for pair_id in mechanism + v4_ids + v5_ids + eligible]
        by_id = {row["pair_id"]: row for row in full_rows}
        full = _write_rows(root / "full.jsonl", full_rows)
        v4 = _write_rows(root / "v4-search.jsonl", [by_id[pair_id] for pair_id in v4_ids])
        v5 = _write_rows(root / "v5-search.jsonl", [by_id[pair_id] for pair_id in v5_ids])
        campaign = {
            "datasets": {
                "full_train": full,
                "search": v5,
                "dev": {"path": str(root / "PROTECTED-dev.jsonl")},
                "heldout": {"path": str(root / "PROTECTED-heldout.jsonl")},
            }
        }
        source = {"datasets": {"search": v4}}
        return campaign, source, {"full": full, "v4": v4, "v5": v5}

    def test_rotation_is_three_disjoint_search30_and_never_reads_protected_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day23-csearch-rotation-") as raw:
            root = Path(raw)
            campaign, source, _ = self._rotation_fixture(root)
            original = Path.read_text

            def guarded(path: Path, *args, **kwargs):
                if "PROTECTED" in str(path):
                    raise AssertionError("protected row file was opened")
                return original(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", guarded):
                first = initializer._rotation_plan(campaign, source, bootcamp=root)
                second = initializer._rotation_plan(campaign, source, bootcamp=root)
            self.assertEqual(first, second)
            self.assertEqual(first["historical_exposure"]["distinct_pair_count"], 64)
            self.assertEqual(first["eligible_pair_count"], 90)
            self.assertEqual([item["records"] for item in first["tranches"]], [30, 30, 30])
            self.assertEqual(
                [item["version_id"] for item in first["tranches"]],
                initializer.ALLOWED_VERSION_IDS,
            )

    def _state(self, root: Path) -> dict:
        campaign, source, _ = self._rotation_fixture(root)
        rotation = initializer._rotation_plan(campaign, source, bootcamp=root)
        digest = "1" * 64
        source_charter = _fake_identity(
            "rsi-control/charters/goal-0002-day23-dpo/charter.json", digest
        )
        return {
            "bootcamp": root,
            "run_root": root / "run",
            "campaign": {"campaign_sha256": "2" * 64},
            "campaign_identity": _fake_identity(str(root / "run/binding/gpu-campaign.json"), "2" * 64),
            "selection": {"selection_sha256": "3" * 64},
            "selection_identity": _fake_identity(str(root / "run/evidence/selection/search-selection.json"), "3" * 64),
            "receipts": [_fake_identity(str(root / f"receipt-{index}.json"), str(index + 4) * 64) for index in range(2)],
            "evaluations": [_fake_identity(str(root / f"eval-{index}.json"), "6" * 64) for index in range(4)],
            "search_claim_identity": _fake_identity(str(root / "search-claim.json"), "7" * 64),
            "event7_identity": _fake_identity(
                "rsi-control/charters/goal-0002-day23-dpo/events/event-000007-execution-binding-corrected.json",
                "8" * 64,
            ),
            "strict_receipt_identity": _fake_identity(str(root / "strict.json"), "9" * 64),
            "strict_amendment_identity": _fake_identity(
                "rsi-control/charters/goal-0002-day23-dpo/amendments/amendment-000004-v0005-strict-preunseal.json",
                "a" * 64,
            ),
            "leases": {
                "source_charter": source_charter,
                "scope_id": "shared-scope",
                "ledger_root": "access-ledger/shared-scope",
                "dev": {"claim_path": "access-ledger/shared-scope/dev-claim.json"},
                "heldout": {"claim_path": "access-ledger/shared-scope/heldout-claim.json"},
            },
            "rotation": rotation,
            "created_at_utc": "2026-08-14T12:00:00+00:00",
        }

    def test_documents_freeze_authority_not_scientific_recipe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day23-csearch-documents-") as raw:
            state = self._state(Path(raw))
            documents = initializer._documents(state)
            self.assertEqual(len(documents), 5)
            charter = next(value for path, value, _ in documents if path.name == "charter.json")
            self.assertFalse(charter["scientific_recipe_authority"]["frozen_by_initializer"])
            self.assertEqual(charter["iteration_budget"]["maximum_search_trajectories_per_version"], 2)
            self.assertEqual(charter["iteration_budget"]["maximum_search_candidates_per_version"], 4)
            self.assertFalse(charter["protected_data"]["raw_row_access_authorized"])
            self.assertFalse(charter["protected_data"]["claim_creation_authorized"])
            self.assertNotIn("learning_rate", json.dumps(charter))

    def test_preflight_build_check_and_o_excl(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day23-csearch-build-") as raw:
            root = Path(raw)
            state = self._state(root)
            args = argparse.Namespace(campaign=root / "unused.json", mode="preflight")
            with mock.patch.object(initializer, "_terminal_state", return_value=state):
                self.assertEqual(initializer.execute(args)["status"], "preflight_pass")
                args.mode = "build"
                self.assertEqual(initializer.execute(args)["status"], "initialized")
                for path, _, field in initializer._documents(state):
                    self.assertEqual(
                        initializer.verify(initializer.load(path, str(path)), field, str(path)),
                        initializer.load(path, str(path))[field],
                    )
                args.mode = "check"
                self.assertEqual(initializer.execute(args)["status"], "check_pass")
                args.mode = "build"
                with self.assertRaisesRegex(initializer.CSearchInitError, "already exists"):
                    initializer.execute(args)


if __name__ == "__main__":
    unittest.main()
