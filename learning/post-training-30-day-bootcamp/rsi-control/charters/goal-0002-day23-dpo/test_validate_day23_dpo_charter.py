#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("validate_day23_dpo_charter.py")
SPEC = importlib.util.spec_from_file_location("validate_day23_dpo_charter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
charter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(charter)


class Day23DPOCharterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        temporary_root = Path(self.temporary.name)
        self.control_root = temporary_root / "rsi-control"
        self.root = (
            self.control_root / "charters" / "goal-0002-day23-dpo"
        )
        self.timestamp = "2026-08-14T10:11:12Z"

    def initialize(self) -> dict:
        return charter.init_control(
            root=self.root,
            control_root=self.control_root,
            bootcamp_root=charter.BOOTCAMP_ROOT,
            created_at_utc=self.timestamp,
        )

    def validate(self, *, check_derived: bool = True) -> dict:
        return charter.validate_control(
            root=self.root,
            control_root=self.control_root,
            bootcamp_root=charter.BOOTCAMP_ROOT,
            check_derived=check_derived,
        )

    def load(self, relative: str) -> dict:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def overwrite(self, relative: str, value: dict) -> None:
        (self.root / relative).write_bytes(charter.pretty_json_bytes(value))

    def test_init_and_validate_exact_precommit(self) -> None:
        initialized = self.initialize()
        result = self.validate()
        self.assertEqual(initialized["status"], "initialized")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["current_state"], "version_precommitted")
        self.assertEqual(result["current_version"], "rsi-v0003")
        self.assertEqual(result["event_count"], 2)
        self.assertFalse(result["dev_lease_present"])
        self.assertFalse(result["heldout_lease_present"])
        self.assertFalse(result["heldout_content_opened_by_validator"])

    def test_init_is_exclusive_and_cannot_rewrite_authority(self) -> None:
        self.initialize()
        original = (self.root / "charter.json").read_bytes()
        with self.assertRaisesRegex(charter.CharterError, "append-only"):
            self.initialize()
        self.assertEqual((self.root / "charter.json").read_bytes(), original)

    def test_resealed_charter_policy_tamper_is_rejected(self) -> None:
        self.initialize()
        document = self.load("charter.json")
        document.pop("charter_sha256")
        document["iteration_budget"]["maximum_semantic_versions"] = 4
        self.overwrite("charter.json", charter.seal(document, "charter_sha256"))
        with self.assertRaisesRegex(charter.CharterError, "dynamically rebound"):
            self.validate()

    def test_resealed_version_grid_tamper_is_rejected(self) -> None:
        self.initialize()
        path = "versions/rsi-v0003/version.json"
        document = self.load(path)
        document.pop("version_sha256")
        document["intervention"]["candidate_values"] = [1e-6, 2e-6, 5e-6]
        self.overwrite(path, charter.seal(document, "version_sha256"))
        with self.assertRaisesRegex(charter.CharterError, "immutable precommit"):
            self.validate()

    def test_resealed_broken_event_chain_is_rejected(self) -> None:
        self.initialize()
        path = "events/event-000002-version-precommitted.json"
        event = self.load(path)
        event.pop("event_sha256")
        event["prior_event"]["event_sha256"] = "0" * 64
        self.overwrite(path, charter.seal(event, "event_sha256"))
        with self.assertRaisesRegex(charter.CharterError, "predecessor"):
            self.validate()

    def test_derived_state_is_explicitly_non_authoritative_and_rebuildable(self) -> None:
        self.initialize()
        derived_path = self.root / "derived" / "state.json"
        derived = json.loads(derived_path.read_text(encoding="utf-8"))
        self.assertIs(derived["authority"], False)
        derived["current_state"] = "qualified"
        derived_path.write_bytes(charter.pretty_json_bytes(derived))

        authority_only = self.validate(check_derived=False)
        self.assertEqual(authority_only["authority_status"], "valid")
        with self.assertRaisesRegex(charter.CharterError, "derived state is stale"):
            self.validate()
        rebuilt = charter.rebuild_derived(
            root=self.root,
            control_root=self.control_root,
            bootcamp_root=charter.BOOTCAMP_ROOT,
        )
        self.assertEqual(rebuilt["current_state"], "version_precommitted")
        self.assertEqual(self.validate()["status"], "valid")

    def test_initializer_and_validator_never_open_protected_rows(self) -> None:
        self.assertIn(
            charter.PROCESSOR_ROWS_PATH, charter.PROTECTED_PRELEASE_PATHS
        )
        protected = {
            (charter.BOOTCAMP_ROOT / relative).resolve()
            for relative in charter.PROTECTED_PRELEASE_PATHS
        }
        observed: list[Path] = []
        original = charter._read_bytes_unchecked

        def audited(path: Path) -> bytes:
            observed.append(path.resolve())
            return original(path)

        with mock.patch.object(charter, "_read_bytes_unchecked", side_effect=audited):
            self.initialize()
            self.validate()
        self.assertTrue(observed)
        self.assertTrue(protected.isdisjoint(observed))
        for relative in charter.PROTECTED_PRELEASE_PATHS:
            with self.assertRaisesRegex(charter.CharterError, "pre-lease access"):
                charter.read_bytes(
                    charter.BOOTCAMP_ROOT / relative,
                    bootcamp_root=charter.BOOTCAMP_ROOT,
                )

    def test_v0003_grid_topology_and_campaign_are_exact(self) -> None:
        self.initialize()
        charter_document = self.load("charter.json")
        version = self.load("versions/rsi-v0003/version.json")
        self.assertEqual(
            version["intervention"]["primary_lever"],
            "model.optimization.learning_rate.base",
        )
        self.assertEqual(version["intervention"]["candidate_values"], [1e-6, 5e-6])
        self.assertIsNone(version["intervention"]["dependent_lever"])
        self.assertEqual(
            version["fixed_operational_constraint"],
            {
                "world_size": 2,
                "parallelism": "DDP",
                "per_device_pair_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "candidate_axis": False,
                "B24_use": "forward_capacity_audit_only",
            },
        )
        campaign = charter_document["dataset_campaign"]
        self.assertEqual(campaign["old4"]["records"], 4)
        self.assertEqual(campaign["fit120"]["records"], 120)
        self.assertEqual(campaign["optimizer124"]["records"], 124)
        self.assertEqual(campaign["search30"]["records"], 30)
        self.assertEqual(campaign["full154_refit"]["records"], 154)
        self.assertEqual(version["campaign"]["refit_checkpoint_candidates"], [15, 30])
        self.assertEqual(len(version["campaign"]["candidate_values"]), 4)
        self.assertEqual(version["frozen_recipe"]["search_seed"], 20260819)
        self.assertEqual(version["frozen_recipe"]["independent_refit_seed"], 20260820)
        self.assertFalse(
            charter_document["global_hard_gates"]["old4_descriptive_diagnostic"][
                "promotion_gate"
            ]
        )
        self.assertEqual(
            charter_document["global_hard_gates"]["search30_gate"][
                "positive_margin_pairs"
            ]["threshold"],
            20,
        )
        self.assertEqual(
            charter_document["global_hard_gates"]["dev17_gate"][
                "positive_margin_pairs"
            ]["threshold"],
            13,
        )

    def test_legacy_failure_outputs_are_negative_provenance_only(self) -> None:
        self.initialize()
        document = self.load("charter.json")
        legacy = document["legacy_day23_terminal_chain"]
        self.assertEqual(legacy["status"], "closed_no_candidate")
        self.assertEqual(legacy["allowed_use"], "negative_provenance_and_diagnosis_only")
        self.assertTrue(
            legacy["permanent_prohibitions"][
                "resume_from_any_legacy_failure_checkpoint"
            ]
        )
        self.assertTrue(
            legacy["permanent_prohibitions"][
                "use_any_legacy_failure_checkpoint_as_candidate"
            ]
        )
        for evidence in legacy["remote_evidence"].values():
            self.assertEqual(
                evidence["availability"], "remote_reported_not_reverified"
            )

    def test_leases_are_global_declared_and_absent_at_precommit(self) -> None:
        self.initialize()
        document = self.load("charter.json")
        leases = document["access_leases"]
        self.assertEqual(leases["dev"]["prerequisite_state"], "candidate_pool_frozen")
        self.assertEqual(leases["heldout"]["prerequisite_state"], "guardrails_passed")
        self.assertEqual(leases["dev"]["maximum_global_claims"], 1)
        self.assertEqual(leases["heldout"]["current_authorized_claims"], 0)
        self.assertEqual(
            leases["heldout"]["maximum_global_claims_after_append_only_authorization"],
            1,
        )
        self.assertEqual(
            leases["heldout"]["authorization"],
            "blocked_until_guardrail_and_heldout_gate_amendment",
        )
        self.assertTrue(leases["dev"]["claim_path"].startswith("access-ledger/"))
        self.assertTrue(leases["heldout"]["claim_path"].startswith("access-ledger/"))
        self.assertFalse((self.control_root / leases["dev"]["claim_path"]).exists())
        self.assertFalse((self.control_root / leases["heldout"]["claim_path"]).exists())
        self.assertFalse(
            document["state_machine"]["runtime_event_append_authorized"]
        )

    def test_authority_symlink_to_heldout_is_rejected_before_read(self) -> None:
        self.initialize()
        derived_path = self.root / "derived" / "state.json"
        derived_path.unlink()
        derived_path.symlink_to(charter.BOOTCAMP_ROOT / charter.HELDOUT_PATH)
        with self.assertRaisesRegex(charter.CharterError, "escaped|symlink"):
            self.validate()

    def test_third_event_is_fail_closed_until_runtime_receipts_are_bound(self) -> None:
        self.initialize()
        prior = self.load("events/event-000002-version-precommitted.json")
        event = charter._make_event(
            sequence=3,
            event_type="lr-grid-search-completed",
            created_at_utc=self.timestamp,
            state_before="version_precommitted",
            state_after="lr_grid_search_completed",
            version_id="rsi-v0003",
            prior_event=charter._event_reference(prior),
            authority_refs=[],
            payload={},
        )
        charter.write_json_exclusive(charter._event_path(self.root, event), event)
        with self.assertRaisesRegex(charter.CharterError, "not authorized"):
            self.validate(check_derived=False)

    def test_partition_and_policy_match_the_campaign_builder(self) -> None:
        builder_path = (
            charter.BOOTCAMP_ROOT
            / "day-23-dpo-theory-smoke"
            / "build_day23_rsi_v0003.py"
        )
        spec = importlib.util.spec_from_file_location("build_day23_rsi_v0003", builder_path)
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = builder
        try:
            spec.loader.exec_module(builder)
            train_rows = builder.contract.load_jsonl(
                charter.BOOTCAMP_ROOT / builder.TRAIN_REL
            )
            fit_rows, search_rows, _ = builder.split_fit_search(train_rows)
        finally:
            sys.modules.pop(spec.name, None)

        expected = charter.build_charter(
            created_at_utc=self.timestamp,
            bootcamp_root=charter.BOOTCAMP_ROOT,
        )
        campaign = expected["dataset_campaign"]
        fit_ids = [row["pair_id"] for row in fit_rows]
        search_ids = [row["pair_id"] for row in search_rows]
        self.assertEqual(campaign["optimizer124"]["ordered_pair_ids"], fit_ids)
        self.assertEqual(campaign["search30"]["ordered_pair_ids"], search_ids)
        self.assertEqual(builder.GOAL_ID, charter.GOAL_ID)
        self.assertEqual(builder.CAMPAIGN_ID, charter.CAMPAIGN_ID)
        self.assertEqual(list(builder.LEARNING_RATES), charter.LEARNING_RATE_GRID)
        self.assertEqual(builder.SEARCH_SEED, charter.SEARCH_SEED)
        self.assertEqual(builder.REFIT_SEED, charter.REFIT_SEED)
        self.assertEqual(list(builder.CHECKPOINT_STEPS), charter.CHECKPOINT_STEPS)

    def test_manifest_declared_eval_identities_are_not_local_verified(self) -> None:
        self.initialize()
        bindings = self.load("charter.json")["artifact_bindings"]
        declared = bindings["manifest_attested_not_opened"]
        self.assertEqual(declared["dev"]["records"], 17)
        self.assertEqual(declared["heldout"]["records"], 29)
        self.assertEqual(
            declared["heldout"]["availability"],
            "manifest_attested_not_opened",
        )
        local_paths = {item["path"] for item in bindings["local_verified"]}
        self.assertNotIn(charter.DEV_PATH.as_posix(), local_paths)
        self.assertNotIn(charter.HELDOUT_PATH.as_posix(), local_paths)


if __name__ == "__main__":
    unittest.main()
