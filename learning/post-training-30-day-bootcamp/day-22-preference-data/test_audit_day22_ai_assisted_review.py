#!/usr/bin/env python3
"""Focused tests for mixed human/sub-agent review auditing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import audit_day22_ai_assisted_review as audit


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AiAssistedReviewAuditTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> dict[str, Path | list[tuple[Path, Path, str]]]:
        blank_rows = []
        key_items = []
        for index in range(1, 101):
            review_id = f"review:{index:03d}"
            blank_rows.append(
                {
                    "schema_name": "day22.blind_review_item",
                    "schema_version": 1,
                    "review_item_id": review_id,
                    "prompt": f"prompt {index}",
                    "response_a": f"a {index}",
                    "response_b": f"b {index}",
                    "rubric_version": "v1",
                    "allowed_verdicts": ["A", "B", "tie", "ambiguous", "reject"],
                    "verdict": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            key_items.append(
                {
                    "review_item_id": review_id,
                    "pair_id": f"pair:{(index - 1) % 50:02d}",
                    "presentation": "primary" if index <= 50 else "swapped",
                    "a_side": "chosen" if index <= 50 else "rejected",
                    "b_side": "rejected" if index <= 50 else "chosen",
                }
            )
        blank = root / "blank.jsonl"
        draft = root / "draft.jsonl"
        key = root / "key.json"
        write_jsonl(blank, blank_rows)
        draft_rows = [dict(row) for row in blank_rows]
        for row in draft_rows[:10]:
            row.update(verdict="A", confidence="high", notes="human")
        write_jsonl(draft, draft_rows)
        key.write_text(
            json.dumps(
                {
                    "schema_name": "day22.blind_review_concealed_key",
                    "items": key_items,
                }
            ),
            encoding="utf-8",
        )
        reviews: list[tuple[Path, Path, str]] = []
        for shard in range(3):
            start = 11 + shard * 30
            packet_rows = []
            annotation_rows = []
            for row_number in range(start, start + 30):
                source = blank_rows[row_number - 1]
                packet_rows.append(
                    {
                        "row_number": row_number,
                        "review_item_id": source["review_item_id"],
                        "prompt": source["prompt"],
                        "response_a": source["response_a"],
                        "response_b": source["response_b"],
                        "allowed_verdicts": ["A", "B", "tie", "ambiguous", "reject"],
                        "allowed_confidence": ["low", "medium", "high"],
                    }
                )
                annotation_rows.append(
                    {
                        "row_number": row_number,
                        "review_item_id": source["review_item_id"],
                        "verdict": "A" if row_number <= 50 else "B",
                        "confidence": "high",
                        "notes": "subagent",
                    }
                )
            packet = root / f"packet-{shard}.jsonl"
            annotations = root / f"annotations-{shard}.jsonl"
            write_jsonl(packet, packet_rows)
            write_jsonl(annotations, annotation_rows)
            reviews.append((packet, annotations, f"reviewer-{shard}"))
        return {"blank": blank, "draft": draft, "key": key, "reviews": reviews}

    def command(self, root: Path, fixture: dict) -> list[str]:
        command = [
            "--blank",
            str(fixture["blank"]),
            "--human-draft",
            str(fixture["draft"]),
            "--concealed-key",
            str(fixture["key"]),
        ]
        for packet, annotations, reviewer in fixture["reviews"]:
            command.extend(
                ["--ai-review", str(packet), str(annotations), reviewer]
            )
        command.extend(
            [
                "--annotations-output",
                str(root / "mixed.jsonl"),
                "--audit-output",
                str(root / "audit.json"),
                "--report-output",
                str(root / "report.md"),
                "--human-draft-snapshot-output",
                str(root / "snapshot.jsonl"),
            ]
        )
        return command

    def test_complete_mixed_review_remains_nonformal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.build_fixture(root)
            self.assertEqual(audit.main(self.command(root, fixture)), 0)
            result = json.loads((root / "audit.json").read_text())
            self.assertEqual(result["counts"]["human_presentations"], 10)
            self.assertEqual(result["counts"]["subagent_presentations"], 90)
            self.assertEqual(result["counts"]["unique_pairs"], 50)
            self.assertFalse(result["formal_human_review_eligible"])
            self.assertFalse(result["formal_dpo_ready"])
            self.assertEqual(len(audit.load_jsonl(root / "mixed.jsonl")), 100)

    def test_packet_text_drift_is_rejected_before_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.build_fixture(root)
            packet, _, _ = fixture["reviews"][0]
            rows = audit.load_jsonl(packet)
            rows[0]["response_a"] += " drift"
            write_jsonl(packet, rows)
            self.assertEqual(audit.main(self.command(root, fixture)), 2)
            self.assertFalse((root / "mixed.jsonl").exists())
            self.assertFalse((root / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
