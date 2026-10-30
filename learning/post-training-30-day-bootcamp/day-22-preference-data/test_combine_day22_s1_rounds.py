#!/usr/bin/env python3
"""Pure CPU tests for base/adaptive rollout combination."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import day22_contract as contract
import combine_day22_s1_rounds as combine
import formal_s1_pair_labeler as labeler
from test_formal_s1_pair_labeler import (
    evidence_for_rollout,
    make_generator,
    make_rollout,
)


def alternate_generator() -> dict:
    value = copy.deepcopy(make_generator())
    value.pop("provenance_sha256")
    value["generation_config"]["temperature"] = 0.8
    value["generation_config_sha256"] = contract.object_sha256(
        value["generation_config"]
    )
    return labeler.seal_generator(value)


class Day22S1RoundCombineTests(unittest.TestCase):
    def test_cross_round_dedup_retains_base_and_requests_only_new_candidates(self) -> None:
        base_generator = make_generator()
        supplemental_generator = alternate_generator()
        base_rows = [
            make_rollout(
                1000 + index,
                [("base-pass", "    return x + 1", 8), ("base-wrong", "    return x - 1", 9)],
                generator=base_generator,
            )
            for index in range(330)
        ]
        adaptive_rows = []
        for index in range(189):
            row = make_rollout(
                1000 + index,
                [("duplicate", "    return x + 1", 8), ("new", "    return x + 2", 10)],
                generator=supplemental_generator,
            )
            family_id = row["family_id"]
            rewritten = []
            for sample_index, candidate in enumerate(row["candidates"], 6):
                value = copy.deepcopy(candidate)
                value["candidate_id"] = (
                    f"{family_id}:s1:test:adaptive-r1:supplemental_t08:sample:{sample_index:02d}"
                )
                value["sample_index"] = sample_index
                value["sample_seed"] = 9000 + index * 20 + sample_index
                rewritten.append(labeler.seal_candidate(value))
            row["candidates"] = rewritten
            adaptive_rows.append(labeler.seal_rollout(row))

        merged, counts = combine.combine_normalized_families(
            base_rows, adaptive_rows
        )
        self.assertEqual(len(merged), 330)
        self.assertEqual(counts["cross_round_exact_response_duplicates"], 189)
        self.assertEqual(counts["supplemental_retained_candidates"], 189)
        self.assertEqual(counts["combined_retained_candidates"], 849)
        requests = combine.supplemental_requests(merged)
        self.assertEqual(len(requests), 189)
        self.assertTrue(
            all(":adaptive-r1:" in request["candidate_id"] for request in requests)
        )
        target = next(row for row in merged if row["family_id"] == "mbpp:task:1000")
        self.assertEqual(len(target["generators"]), 2)
        self.assertEqual(
            {candidate["text"] for candidate in target["candidates"]},
            {"    return x + 1", "    return x - 1", "    return x + 2"},
        )
        self.assertIn(
            "cross_round_exact_response_duplicate",
            {item["reason_code"] for item in target["preselection_exclusions"]},
        )

    def test_different_promoted_checkpoint_fails_closed(self) -> None:
        base_rows = [
            make_rollout(
                2000 + index,
                [("a", "    return x + 1", 8), ("b", "    return x - 1", 9)],
            )
            for index in range(330)
        ]
        drifted = make_generator(checkpoint="s1/a-different-checkpoint")
        adaptive_rows = [
            make_rollout(
                2000 + index,
                [("a", "    return x + 2", 8), ("b", "    return x - 2", 9)],
                generator=drifted,
            )
            for index in range(189)
        ]
        with self.assertRaisesRegex(
            labeler.FormalS1LabelerError, "one promoted S1 policy identity"
        ):
            combine.combine_normalized_families(base_rows, adaptive_rows)

    def test_merge_evidence_does_not_revalidate_derived_unsealed_candidates(self) -> None:
        """Real normalized candidates omit the validator-derived AST hash."""

        rollout = make_rollout(
            3000,
            [
                ("base", "    return x + 1", 8),
                ("adaptive-r1:new", "    return x - 1", 9),
            ],
        )
        evidence = evidence_for_rollout(
            rollout,
            [("pass", "pass"), ("wrong_answer", "wrong_answer")],
        )
        base_evidence = [evidence[0]]
        supplemental_evidence = [evidence[1]]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rollouts_path = root / "rollouts.jsonl"
            base_evidence_path = root / "base-evidence.jsonl"
            supplemental_evidence_path = root / "supplemental-evidence.jsonl"
            combined_evidence_path = root / "combined-evidence.jsonl"
            combine_manifest_path = root / "combine-manifest.json"
            evidence_manifest_path = root / "evidence-manifest.json"

            rollout_payload = labeler._jsonl_bytes([rollout])
            rollouts_path.write_bytes(rollout_payload)
            base_evidence_path.write_bytes(labeler._jsonl_bytes(base_evidence))
            supplemental_evidence_path.write_bytes(
                labeler._jsonl_bytes(supplemental_evidence)
            )
            manifest = combine.seal(
                {
                    "schema_name": combine.COMBINE_SCHEMA,
                    "schema_version": combine.SCHEMA_VERSION,
                    "status": "ready_for_supplemental_e2b",
                    "outputs": {
                        "combined_rollouts": {
                            "path": str(rollouts_path),
                            "file_sha256": hashlib.sha256(rollout_payload).hexdigest(),
                        }
                    },
                },
                "manifest_sha256",
            )
            combine_manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )

            result = combine.merge_evidence(
                Namespace(
                    combined_rollouts=rollouts_path,
                    combine_manifest=combine_manifest_path,
                    base_evidence=base_evidence_path,
                    supplemental_evidence=supplemental_evidence_path,
                    output=combined_evidence_path,
                    manifest_output=evidence_manifest_path,
                )
            )

            self.assertEqual(result["counts"]["combined_evidence"], 2)
            self.assertEqual(result["counts"]["eligible_pairs_after_merge"], 1)
            self.assertTrue(combined_evidence_path.is_file())


if __name__ == "__main__":
    unittest.main()
