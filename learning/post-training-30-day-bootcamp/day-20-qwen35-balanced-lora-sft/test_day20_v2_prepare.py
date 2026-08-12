#!/usr/bin/env python3
"""Offline end-to-end preparation tests for the Day 20 v2 data contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import day20_train_runtime_v2 as runtime
import day20_train_plugin_v2 as train_plugin
import prepare_day20_v2 as prepare
from day20_contract import object_sha256
from day20_contract_v2 import canonicalize_code_continuation


def reference_evidence(
    target_format: str, source_content_sha256: str, *, code: bool = False
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": 2,
        "verifier_version": "day20_reference_evidence_v2",
        "target_format": target_format,
        "source_content_sha256": source_content_sha256,
        "checks": (
            {"source_rebuilt_ast_match": True, "static_compile": True}
            if code
            else {"source_target_match": True}
        ),
    }
    evidence["reference_evidence_sha256"] = object_sha256(evidence)
    return evidence


def normalized_row(
    *,
    sample_id: str,
    skill: str,
    target_format: str,
    tokens: int,
    adapter: str,
) -> dict[str, object]:
    target_by_format = {
        "general_mcq": "Final answer: A",
        "general_instruction": "A concise factual response.",
        "math_reasoning": "One plus one equals two.\nFinal answer: 2",
        "finance_value_scale": "Final answer: 2 million",
        "code_continuation": "    return x + 1",
    }
    messages = [
        {"role": "user", "content": f"Fixture prompt {sample_id}"},
        {"role": "assistant", "content": target_by_format[target_format]},
    ]
    source_content = object_sha256({"source": sample_id})
    row: dict[str, object] = {
        "sample_id": sample_id,
        "skill": skill,
        "target_format": target_format,
        "messages": messages,
        "qwen35_tokenization": {
            "template": "qwen3_5",
            "enable_thinking": False,
            "add_non_thinking_prefix": True,
            "loss_scale": "default+ignore_empty_think",
            "truncated": False,
            "input_tokens": tokens,
            "supervised_tokens": tokens,
            "messages_sha256": object_sha256(messages),
            "render_sha256": object_sha256([sample_id, "render"]),
            "labels_sha256": object_sha256([sample_id, "labels"]),
        },
        "source_lineage": {
            "source": "fixture/source",
            "revision": "a" * 40,
            "split": "train",
            "adapter": adapter,
            "license": "test",
            "source_file_sha256": object_sha256([sample_id, "file"]),
            "source_content_sha256": source_content,
        },
        "reference_evidence": reference_evidence(
            target_format, source_content, code=target_format == "code_continuation"
        ),
    }
    if target_format == "finance_value_scale":
        row["finance_scale"] = "million"
    if target_format == "code_continuation":
        prefix = "def solve(x):"
        row["code_prefix"] = prefix
        row["code_continuation_v2"] = canonicalize_code_continuation(
            messages[-1]["content"], prefix, require_raw_contract=True
        ).as_evidence()
    return row


def add_rows(
    rows: list[dict[str, object]],
    *,
    skill: str,
    target_format: str,
    count: int,
    tokens: int,
    adapter: str,
    offset: int = 0,
) -> None:
    for index in range(count):
        rows.append(
            normalized_row(
                sample_id=f"{skill}:{target_format}:{adapter}:{offset + index:04d}",
                skill=skill,
                target_format=target_format,
                tokens=tokens,
                adapter=adapter,
            )
        )


class Day20PrepareV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        (self.model / "tokenizer.json").write_text(
            '{"model": {"type": "test"}}\n', encoding="utf-8"
        )
        (self.model / "model.safetensors").write_bytes(b"fixture-weights")

        rows_by_skill: dict[str, list[dict[str, object]]] = {
            skill: [] for skill in prepare.SKILLS
        }
        add_rows(
            rows_by_skill["general"],
            skill="general",
            target_format="general_mcq",
            count=120,
            tokens=250,
            adapter="mmlu_v1",
        )
        add_rows(
            rows_by_skill["general"],
            skill="general",
            target_format="general_mcq",
            count=1,
            tokens=1_998,
            adapter="mmlu_v1",
            offset=120,
        )
        add_rows(
            rows_by_skill["general"],
            skill="general",
            target_format="general_instruction",
            count=188,
            tokens=250,
            adapter="tulu_general_v1",
        )
        add_rows(
            rows_by_skill["general"],
            skill="general",
            target_format="general_instruction",
            count=1,
            tokens=1_002,
            adapter="tulu_general_v1",
            offset=188,
        )
        add_rows(
            rows_by_skill["math"],
            skill="math",
            target_format="math_reasoning",
            count=320,
            tokens=250,
            adapter="gsm8k_v1",
        )
        for skill, target_format, replay_adapter, new_adapter in (
            (
                "finance",
                "finance_value_scale",
                "tatqa_v1",
                prepare.FINQA_ADAPTER,
            ),
            (
                "code",
                "code_continuation",
                "mbpp_v1",
                prepare.TULU_CODE_ADAPTER,
            ),
        ):
            add_rows(
                rows_by_skill[skill],
                skill=skill,
                target_format=target_format,
                count=260,
                tokens=250,
                adapter=replay_adapter,
            )
            add_rows(
                rows_by_skill[skill],
                skill=skill,
                target_format=target_format,
                count=60,
                tokens=250,
                adapter=new_adapter,
            )

        self.sources: dict[str, Path] = {}
        source_dir = self.root / "sources"
        source_dir.mkdir()
        outputs: dict[str, object] = {}
        for skill, rows in rows_by_skill.items():
            path = source_dir / f"{skill}.normalized.qwen35-v2.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.sources[skill] = path
            outputs[skill] = {
                "path": str(path.resolve()),
                "file_sha256": prepare.file_sha256(path),
                "records": len(rows),
                "supervised_tokens": sum(
                    int(row["qwen35_tokenization"]["supervised_tokens"])
                    for row in rows
                ),
            }
        source_manifest: dict[str, object] = {
            "schema_version": 2,
            "domain": "day20.qwen35_source_expansion.v2",
            "status": "pass",
            "adapter_version": prepare.ADAPTER_VERSION,
            "dataset_code_execution": False,
            "model_path": str(self.model.resolve()),
            "tokenizer_files": prepare.tokenizer_file_identity(self.model),
            "template": prepare.QWEN35_TEMPLATE_CONTRACT,
            "outputs": outputs,
            "day09_manifest": {"path": "/fixture/day09", "file_sha256": "2" * 64},
        }
        source_manifest["manifest_sha256"] = object_sha256(source_manifest)
        self.source_manifest = source_dir / "SOURCE-EXPANSION-MANIFEST.json"
        self.source_manifest.write_text(
            json.dumps(source_manifest), encoding="utf-8"
        )

        eval_rows = [
            {
                "sample_id": f"eval:{skill}:{index:02d}",
                "slice": skill,
                "evaluation_split": "dev",
            }
            for skill in prepare.SKILLS
            for index in range(28)
        ]
        self.eval_manifest = self.root / "eval.json"
        self.eval_manifest.write_text(
            json.dumps({"records": eval_rows}), encoding="utf-8"
        )
        self.run_root = self.root / "run"
        self.run_root.mkdir()
        (self.run_root / prepare.RUN_ROOT_MARKER).write_text(
            "day20-qwen35-candidate-factory-v2\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_exact_budgets_quotas_order_and_runtime_contract(self) -> None:
        manifest = prepare.prepare_run(
            run_root=self.run_root,
            model_snapshot=self.model,
            sources=self.sources,
            source_expansion_manifest=self.source_manifest,
            eval_manifest=self.eval_manifest,
        )
        self.assertEqual(manifest["datasets"]["probe"]["supervised_tokens"], 24_000)
        self.assertEqual(manifest["datasets"]["main"]["supervised_tokens"], 320_000)
        self.assertEqual(manifest["datasets"]["probe"]["temporal_mix_audit"]["status"], "pass")
        self.assertEqual(manifest["datasets"]["main"]["temporal_mix_audit"]["status"], "pass")
        for adapter in (prepare.FINQA_ADAPTER, prepare.TULU_CODE_ADAPTER):
            self.assertGreaterEqual(
                manifest["datasets"]["probe"]["cohort_supervised_tokens"][adapter],
                1_000,
            )
            self.assertGreaterEqual(
                manifest["datasets"]["main"]["cohort_supervised_tokens"][adapter],
                15_000,
            )
        probe_ids = set(manifest["datasets"]["probe"]["ordered_sample_ids"])
        main_ids = set(manifest["datasets"]["main"]["ordered_sample_ids"])
        self.assertTrue(probe_ids <= main_ids)

        validation = runtime.validate_inputs(
            self.run_root / "DAY20-V2-MANIFEST.json",
            dataset_path=self.run_root / "data/probe-v2.jsonl",
            config_path=self.run_root / "configs/probe-s20260809-lr1e-4.json",
            run_kind="probe",
            seed=20260809,
            learning_rate="1e-4",
        )
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(
            validation["checkpoint_tokens"],
            {"t6000": 6_000, "t12000": 12_000, "t18000": 18_000, "t24000": 24_000},
        )

        resolved_main = train_plugin.resolve_main_config(
            manifest_path=self.run_root / "DAY20-V2-MANIFEST.json",
            template_path=self.run_root / "configs/main-template-v2.json",
            learning_rate="6e-5",
            seed=20260810,
        )
        resolved_path = self.run_root / "configs/main-s20260810-lr6e-5.json"
        resolved_path.write_text(json.dumps(resolved_main), encoding="utf-8")
        main_validation = runtime.validate_inputs(
            self.run_root / "DAY20-V2-MANIFEST.json",
            dataset_path=self.run_root / "data/main-v2.jsonl",
            config_path=resolved_path,
            run_kind="main",
            seed=20260810,
            learning_rate="6e-5",
        )
        self.assertFalse(main_validation["config_requires_resolution"])
        self.assertEqual(main_validation["checkpoint_tokens"]["final"], 320_000)

    def test_missing_new_source_cohort_fails_closed(self) -> None:
        rows = prepare.load_normalized_sources(self.sources)
        for row in rows:
            if row["source_lineage"]["adapter"] == prepare.FINQA_ADAPTER:
                row["source_lineage"]["adapter"] = "legacy_finance"
        with self.assertRaisesRegex(
            prepare.Day20PreparationV2Error, "cohort quota"
        ):
            prepare.select_nested_datasets(rows)


if __name__ == "__main__":
    unittest.main()
