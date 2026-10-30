#!/usr/bin/env python3
"""Offline tests for the Day 20 v2 temporal ordering contract."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_ordering_v2 as ordering


def records_for_skill(skill: str, count: int, tokens: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{skill}-{index:04d}",
            "skill": skill,
            "qwen35_supervised_tokens": tokens,
            "payload": {"source_index": index},
        }
        for index in range(count)
    ]


def balanced_records(count_per_skill: int = 64) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for skill in ordering.SKILLS:
        rows.extend(records_for_skill(skill, count_per_skill, 10))
    return rows


class Day20OrderingV2Tests(unittest.TestCase):
    def test_order_is_input_order_independent_and_does_not_mutate_records(self) -> None:
        rows = balanced_records()
        original = copy.deepcopy(rows)

        forward = ordering.order_records(rows)
        reverse = ordering.order_records(list(reversed(rows)))

        self.assertEqual(rows, original)
        self.assertEqual(
            [record["sample_id"] for record in forward],
            [record["sample_id"] for record in reverse],
        )
        self.assertEqual(
            {record["sample_id"] for record in forward},
            {record["sample_id"] for record in rows},
        )

    def test_audit_exposes_matrix_and_all_passing_gates(self) -> None:
        ordered = ordering.order_records(balanced_records())
        audit = ordering.audit_temporal_mix(ordered)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["optimizer_steps"], 32)
        self.assertEqual(len(audit["optimizer_step_skill_matrix"]), 32)
        self.assertTrue(audit["gates"]["paced_release"]["passing"])
        self.assertTrue(audit["gates"]["rolling_skill_coverage"]["passing"])
        self.assertTrue(audit["gates"]["cumulative_token_share"]["passing"])
        self.assertTrue(audit["gates"]["last_observed_step"]["passing"])
        self.assertEqual(
            [
                checkpoint["fraction"]
                for checkpoint in audit["gates"]["cumulative_token_share"][
                    "checkpoints"
                ]
            ],
            ["25%", "50%", "75%", "100%"],
        )
        for checkpoint in audit["gates"]["cumulative_token_share"][
            "checkpoints"
        ]:
            self.assertEqual(
                checkpoint["token_share_by_skill"],
                {skill: 0.25 for skill in ordering.SKILLS},
            )

    def test_extreme_long_and_short_records_still_pass_when_mix_is_feasible(self) -> None:
        # Every skill contributes 1,600 tokens, but Math has 10x longer records
        # than General and 10x fewer records.  This exercises both release
        # pacing and token-deficit selection instead of record-count balancing.
        rows: list[dict[str, object]] = []
        rows.extend(records_for_skill("general", 160, 10))
        rows.extend(records_for_skill("math", 16, 100))
        rows.extend(records_for_skill("finance", 80, 20))
        rows.extend(records_for_skill("code", 40, 40))

        ordered = ordering.order_records(rows)
        audit = ordering.audit_temporal_mix(ordered)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["optimizer_steps"], 37)
        for checkpoint in audit["gates"]["cumulative_token_share"][
            "checkpoints"
        ]:
            self.assertTrue(checkpoint["passing"])
            self.assertTrue(
                all(
                    0.20 <= share <= 0.30
                    for share in checkpoint["token_share_by_skill"].values()
                )
            )

    def test_partial_final_batch_does_not_make_release_mathematically_infeasible(self) -> None:
        rows: list[dict[str, object]] = []
        rows.extend(records_for_skill("general", 14, 1))
        rows.extend(records_for_skill("math", 1, 14))
        rows.extend(records_for_skill("finance", 1, 14))
        rows.extend(records_for_skill("code", 1, 14))

        # The older step/ceil(total/batch) denominator under-released these 17
        # rows before the partial final batch.  The slot-progress denominator
        # schedules them; later token-share gates may still reject the mix.
        try:
            ordered = ordering.order_records(rows)
        except ordering.Day20OrderingV2Error as error:
            self.assertNotIn("paced release cannot fill", str(error))
        else:
            self.assertEqual(len(ordered), len(rows))

    def test_order_fails_closed_when_temporal_gates_are_unprovable(self) -> None:
        rows: list[dict[str, object]] = []
        for skill in ordering.SKILLS:
            rows.extend(records_for_skill(skill, 8, 10))

        with self.assertRaisesRegex(
            ordering.Day20OrderingV2Error, "rolling 8-step skill coverage failed"
        ):
            ordering.order_records(rows)

    def test_audit_rejects_clustered_skill_order(self) -> None:
        clustered: list[dict[str, object]] = []
        for skill in ordering.SKILLS:
            clustered.extend(records_for_skill(skill, 64, 10))

        audit = ordering.audit_temporal_mix(clustered)

        self.assertEqual(audit["status"], "fail")
        self.assertFalse(audit["gates"]["paced_release"]["passing"])
        self.assertFalse(audit["gates"]["rolling_skill_coverage"]["passing"])
        self.assertFalse(audit["gates"]["cumulative_token_share"]["passing"])
        self.assertFalse(audit["gates"]["last_observed_step"]["passing"])

    def test_invalid_records_fail_closed(self) -> None:
        cases = {
            "duplicate sample_id": balanced_records() + [balanced_records()[0]],
            "unsupported skill": balanced_records()
            + [
                {
                    "sample_id": "other-1",
                    "skill": "other",
                    "qwen35_supervised_tokens": 1,
                }
            ],
            "positive integer": [
                {
                    **record,
                    "qwen35_supervised_tokens": 0
                    if record["sample_id"] == "code-0000"
                    else record["qwen35_supervised_tokens"],
                }
                for record in balanced_records()
            ],
        }
        for message, rows in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ordering.Day20OrderingV2Error, message):
                    ordering.order_records(rows)

        with self.assertRaisesRegex(
            ordering.Day20OrderingV2Error, "global_batch_size must be"
        ):
            ordering.order_records(balanced_records(), global_batch_size=0)


if __name__ == "__main__":
    unittest.main()
