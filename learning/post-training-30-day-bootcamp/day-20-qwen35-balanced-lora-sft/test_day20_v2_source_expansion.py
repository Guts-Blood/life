#!/usr/bin/env python3
"""Offline tests for the append-only Day 20 v2 source expansion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import day20_source_expansion_v2 as expansion
from day20_contract import object_sha256


class FakeTemplate:
    def encode(self, data: dict[str, object], return_length: bool = True) -> dict[str, list[int]]:
        messages = data["messages"]
        assert isinstance(messages, list)
        assistant = str(messages[-1]["content"])
        supervised = max(2, min(20, len(assistant.split()) + 2))
        total = supervised + 8
        return {
            "input_ids": list(range(total)),
            "labels": [-100] * (total - supervised) + list(range(supervised)),
        }


def reference(target_format: str, source_content: str, *, code: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "verifier_version": "day20_reference_evidence_v1",
        "target_format": target_format,
        "source_content_sha256": source_content,
        "checks": (
            {"source_rebuilt_match": True, "static_compile": True}
            if code
            else {"source_target_match": True}
        ),
    }
    value["reference_evidence_sha256"] = object_sha256(value)
    return value


def base_row(skill: str) -> dict[str, object]:
    definitions = {
        "general": ("general_mcq", "Choose A.", "Final answer: A"),
        "math": ("math_reasoning", "Compute 1+1.", "1 plus 1 is 2.\nFinal answer: 2"),
        "finance": ("finance_value_scale", "Return the value.", "Final answer: 2 million"),
        "code": ("code_continuation", "Complete solve.", "return x + 1"),
    }
    target_format, prompt, target = definitions[skill]
    source_content = object_sha256({"skill": skill, "source": "fixture"})
    row: dict[str, object] = {
        "sample_id": f"base:{skill}",
        "skill": skill,
        "target_format": target_format,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "source_lineage": {
            "source": f"fixture/{skill}",
            "revision": "a" * 40,
            "split": "train",
            "adapter": f"fixture_{skill}_v1",
            "license": "test",
            "source_file_sha256": "1" * 64,
            "source_content_sha256": source_content,
        },
        "reference_evidence": reference(
            target_format, source_content, code=skill == "code"
        ),
    }
    if skill == "finance":
        row["finance_scale"] = "million"
    if skill == "code":
        row["code_prefix"] = "def solve(x):"
    return row


def day09_row(skill: str) -> dict[str, object]:
    if skill == "finance":
        messages = [
            {"role": "user", "content": "What is the margin?"},
            {"role": "assistant", "content": "Calculation: divide(1, 2)\nFinal answer: 50%"},
        ]
        source = "bevaya/FinQA"
        revision = "3" * 40
    else:
        messages = [
            {"role": "user", "content": "Increment an integer."},
            {"role": "assistant", "content": "def increment(x):\n    return x + 1"},
        ]
        source = "allenai/tulu-3-sft-personas-code"
        revision = "4" * 40
    sample_id = f"{skill}:day09"
    return {
        "sample_id": sample_id,
        "parent_id": f"parent-{skill}",
        "skill": skill,
        "source": source,
        "revision": revision,
        "split": "train",
        "license": "test",
        "messages": messages,
        "content_hash": object_sha256(messages),
    }


class Day20SourceExpansionV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        (self.model / "tokenizer.json").write_text(
            '{"model": {"type": "test"}}\n', encoding="utf-8"
        )
        self.base_sources: dict[str, Path] = {}
        for skill in expansion.SKILLS:
            path = self.root / f"{skill}.jsonl"
            path.write_text(json.dumps(base_row(skill)) + "\n", encoding="utf-8")
            self.base_sources[skill] = path
        records = sorted(
            [day09_row("finance"), day09_row("code")],
            key=lambda row: str(row["sample_id"]),
        )
        header: dict[str, object] = {
            "manifest_name": "day09-dataset-manifest",
            "manifest_version": "day09_clean_parent_pool_v1",
        }
        header["manifest_hash"] = object_sha256(
            {"header": header, "records": records}
        )
        self.day09 = self.root / "day09.json"
        self.day09.write_text(
            json.dumps({"header": header, "records": records}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_adds_pinned_finance_and_code_and_upgrades_code(self) -> None:
        output = self.root / "expanded"
        manifest = expansion.build_expanded_sources(
            base_sources=self.base_sources,
            day09_manifest_path=self.day09,
            output_dir=output,
            template=FakeTemplate(),
            model_path=self.model,
        )
        self.assertEqual(manifest["domain"], "day20.qwen35_source_expansion.v2")
        self.assertEqual(manifest["outputs"]["general"]["records"], 1)
        self.assertEqual(manifest["outputs"]["math"]["records"], 1)
        self.assertEqual(manifest["outputs"]["finance"]["records"], 2)
        self.assertEqual(manifest["outputs"]["code"]["records"], 2)

        code_rows = expansion._load_jsonl(
            output / "code.normalized.qwen35-v2.jsonl"
        )
        self.assertTrue(
            all(row["messages"][-1]["content"].startswith("    ") for row in code_rows)
        )
        self.assertTrue(all("code_continuation_v2" in row for row in code_rows))
        adapters = {row["source_lineage"]["adapter"] for row in code_rows}
        self.assertIn(expansion.TULU_CODE_ADAPTER, adapters)

        finance_rows = expansion._load_jsonl(
            output / "finance.normalized.qwen35-v2.jsonl"
        )
        added = next(
            row for row in finance_rows if row["source_lineage"]["adapter"] == expansion.FINQA_ADAPTER
        )
        self.assertEqual(added["messages"][-1]["content"], "Final answer: 50%")
        self.assertEqual(added["finance_scale"], "percent")

    def test_day09_manifest_tamper_fails_closed(self) -> None:
        value = json.loads(self.day09.read_text(encoding="utf-8"))
        value["records"][0]["parent_id"] = "tampered"
        self.day09.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(
            expansion.Day20SourceExpansionV2Error, "content hash drifted"
        ):
            expansion.verify_day09_manifest(self.day09)

    def test_legacy_code_row_with_invalid_prefix_is_rejected_and_counted(self) -> None:
        invalid = base_row("code")
        invalid["code_prefix"] = "def solve(x):\n        import math"
        invalid["messages"][-1]["content"] = "        return x + 1"
        self.base_sources["code"].write_text(
            json.dumps(invalid) + "\n", encoding="utf-8"
        )

        output = self.root / "expanded-invalid-code"
        manifest = expansion.build_expanded_sources(
            base_sources=self.base_sources,
            day09_manifest_path=self.day09,
            output_dir=output,
            template=FakeTemplate(),
            model_path=self.model,
        )

        self.assertEqual(manifest["outputs"]["code"]["records"], 1)
        self.assertEqual(
            manifest["rejections"]["base_code:v2_contract_rejected"], 1
        )
        code_rows = expansion._load_jsonl(
            output / "code.normalized.qwen35-v2.jsonl"
        )
        self.assertEqual(
            code_rows[0]["source_lineage"]["adapter"], expansion.TULU_CODE_ADAPTER
        )

    def test_raw_code_and_token_identity_are_strict(self) -> None:
        candidate = expansion._upgrade_base_row(base_row("code"))
        normalized = expansion._retokenize(candidate, FakeTemplate())
        bad_code = json.loads(json.dumps(normalized))
        bad_code["messages"][-1]["content"] = "return x"
        with self.assertRaises(expansion.Day20SourceExpansionV2Error):
            expansion.validate_v2_normalized_record(bad_code, expected_skill="code")
        bad_hash = json.loads(json.dumps(normalized))
        bad_hash["qwen35_tokenization"]["messages_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            expansion.Day20SourceExpansionV2Error, "token evidence"
        ):
            expansion.validate_v2_normalized_record(bad_hash, expected_skill="code")


if __name__ == "__main__":
    unittest.main()
