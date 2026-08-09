#!/usr/bin/env python3
"""Offline unit tests for the Day 20 data and configuration contracts."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_contract as contract
import prepare_day20 as prepare


class Day20ContractTests(unittest.TestCase):
    def test_frozen_budgets(self) -> None:
        self.assertEqual(contract.MAIN_SUPERVISED_TOKENS, 256_000)
        self.assertEqual(contract.MAIN_TOKENS_PER_SKILL, 64_000)
        self.assertEqual(contract.PROBE_SUPERVISED_TOKENS, 16_000)
        self.assertEqual(contract.PROBE_TOKENS_PER_SKILL, 4_000)
        self.assertEqual(sum(contract.MAIN_TOKENS_BY_FORMAT.values()), 256_000)
        self.assertEqual(sum(contract.PROBE_TOKENS_BY_FORMAT.values()), 16_000)

    def test_target_grammars(self) -> None:
        self.assertEqual(
            contract.normalize_assistant_target("general", "general_mcq", "Final answer: b"),
            "Final answer: B",
        )
        self.assertEqual(
            contract.normalize_assistant_target(
                "math", "math_reasoning", "We add 1 and 2.\nFinal answer: 3.0"
            ),
            "We add 1 and 2.\nFinal answer: 3",
        )
        self.assertEqual(
            contract.normalize_assistant_target(
                "finance",
                "finance_value_scale",
                "Final answer: 8.50 million",
                finance_scale="million",
            ),
            "Final answer: 8.5 million",
        )
        self.assertEqual(
            contract.normalize_assistant_target(
                "code",
                "code_continuation",
                "    return x + 1",
                code_prefix='def f(x):\n    """Return x plus one."""',
            ),
            "    return x + 1",
        )

    def test_target_grammars_fail_closed(self) -> None:
        invalid = (
            ("general", "general_mcq", "B", None),
            ("general", "general_instruction", "Reasoning: answer", None),
            ("math", "math_reasoning", "Final answer: 2", None),
            ("finance", "finance_value_scale", "Final answer: 2", None),
            (
                "code",
                "code_continuation",
                "def f(x):\n    return x",
                'def f(x):\n    """Return x."""',
            ),
            (
                "code",
                "code_continuation",
                "    value = x + 1\ndef replacement(y):\n    return y",
                'def f(x):\n    """Return x."""',
            ),
            (
                "code",
                "code_continuation",
                "return x",
                'def f(x):\n    """Return x."""',
            ),
        )
        for skill, target_format, target, prefix in invalid:
            with self.subTest(target_format=target_format):
                with self.assertRaises(contract.Day20ContractError):
                    contract.normalize_assistant_target(
                        skill, target_format, target, code_prefix=prefix
                    )

    def test_exact_and_stable_subsets(self) -> None:
        weights = [2, 3, 5, 7]
        selected = contract.exact_subset_indices(weights, 10)
        self.assertEqual(sum(weights[index] for index in selected), 10)
        with self.assertRaises(contract.Day20ContractError):
            contract.exact_subset_indices([4, 8], 3)

        rows = [{"sample_id": f"id-{index}"} for index in range(20)]
        forward = contract.stable_hash_subset(rows, 5, seed="fixed")
        reverse = contract.stable_hash_subset(list(reversed(rows)), 5, seed="fixed")
        self.assertEqual(
            [row["sample_id"] for row in forward],
            [row["sample_id"] for row in reverse],
        )

        counterexample = [
            12, 17, 11, 1, 4, 15, 23, 15, 12, 10, 18, 13, 11,
            22, 19, 16, 4, 21, 13, 13, 7, 18, 1, 9, 21,
        ]
        probe, main = prepare.select_nested_exact_indices(
            counterexample, probe_target=20, main_target=320
        )
        self.assertTrue(probe <= main)
        self.assertEqual(sum(counterexample[index] for index in probe), 20)
        self.assertEqual(sum(counterexample[index] for index in main), 320)

    def test_leakage_checks_all_three_layers(self) -> None:
        clean = {
            "sample_id": "train-1",
            "prompt_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "source_lineage": {},
        }
        eval_record = {
            "sample_id": "eval-1",
            "messages": [{"role": "user", "content": "question"}],
            "content_hash": "c" * 64,
        }
        self.assertFalse(
            any(contract.assert_no_train_dev_leakage([clean], [eval_record]).values())
        )
        cases = (
            ({**clean, "sample_id": "eval-1"}, eval_record),
            ({**clean, "content_sha256": "c" * 64}, eval_record),
            (
                {
                    **clean,
                    "prompt_sha256": contract.object_sha256(eval_record["messages"]),
                },
                eval_record,
            ),
        )
        for train_record, dev_record in cases:
            with self.assertRaises(contract.Day20ContractError):
                contract.assert_no_train_dev_leakage([train_record], [dev_record])

    def test_trainable_inventory_and_hash(self) -> None:
        names = [
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight",
            "base_model.model.model.language_model.layers.1.linear_attn.in_proj_qkv.lora_A.default.weight",
            "base_model.model.model.language_model.layers.1.linear_attn.in_proj_qkv.lora_B.default.weight",
            "base_model.model.model.language_model.layers.2.mlp.down_proj.lora_A.default.weight",
            "base_model.model.model.language_model.layers.2.mlp.down_proj.lora_B.default.weight",
        ]
        inventory = contract.validate_trainable_inventory(names)
        self.assertEqual(inventory["target_module_count"], 3)
        self.assertEqual(
            contract.validate_trainable_inventory(
                names, expected_sha256=inventory["trainable_names_sha256"]
            )["trainable_names_sha256"],
            inventory["trainable_names_sha256"],
        )
        with self.assertRaises(contract.Day20ContractError):
            contract.validate_trainable_inventory(
                [
                    "model.visual.layers.0.self_attn.q_proj.lora_A.default.weight",
                    "model.visual.layers.0.self_attn.q_proj.lora_B.default.weight",
                ]
            )
        with self.assertRaises(contract.Day20ContractError):
            contract.validate_trainable_inventory(names[:-1])

    def test_config_hash_detects_tampering(self) -> None:
        config = contract.build_probe_config(
            model_snapshot="/models/qwen35/base/revision",
            dataset_path="/run/data/probe.jsonl",
            dataset_sha256="d" * 64,
            learning_rate=3e-5,
        )
        self.assertEqual(contract.verify_immutable_hash(config), config["immutable_sha256"])
        tampered = copy.deepcopy(config)
        tampered["training"]["learning_rate"] = 1e-4
        with self.assertRaises(contract.Day20ContractError):
            contract.verify_immutable_hash(tampered)


class Day20PreparationTests(unittest.TestCase):
    @staticmethod
    def _target(skill: str, target_format: str, index: int) -> tuple[str, str | None]:
        if target_format == "general_mcq":
            return f"Final answer: {'ABCD'[index % 4]}", None
        if target_format == "general_instruction":
            return f"A concise replay answer number {index}.", None
        if target_format == "math_reasoning":
            return f"Adding the quantities gives {index}.\nFinal answer: {index}", None
        if target_format == "finance_value_scale":
            return f"Final answer: {index} million", None
        prefix = f'def solve_{index}(x):\n    """Return a deterministic value."""'
        return f"    return x + {index}", prefix

    @classmethod
    def _row(
        cls,
        skill: str,
        target_format: str,
        index: int,
        supervised_tokens: int = 200,
    ) -> dict[str, object]:
        target, prefix = cls._target(skill, target_format, index)
        messages = [
            {"role": "user", "content": f"Unique {skill} question {target_format} {index}"},
            {"role": "assistant", "content": target},
        ]
        source_content_sha256 = contract.text_sha256(
            f"source-content:{skill}:{target_format}:{index}"
        )
        evidence = {
            "schema_version": 1,
            "verifier_version": "day20_reference_evidence_v1",
            "target_format": target_format,
            "source_content_sha256": source_content_sha256,
            "checks": (
                {"source_rebuilt_match": True, "static_compile": True}
                if target_format == "code_continuation"
                else {"source_target_match": True}
            ),
        }
        evidence["reference_evidence_sha256"] = contract.object_sha256(evidence)
        row: dict[str, object] = {
            "sample_id": f"train:{skill}:{target_format}:{index}",
            "skill": skill,
            "target_format": target_format,
            "messages": messages,
            "qwen35_tokenization": {
                "template": "qwen3_5",
                "enable_thinking": False,
                "add_non_thinking_prefix": True,
                "loss_scale": "default+ignore_empty_think",
                "truncated": False,
                "input_tokens": supervised_tokens + 100,
                "supervised_tokens": supervised_tokens,
                "messages_sha256": contract.object_sha256(messages),
                "render_sha256": contract.text_sha256(f"render:{skill}:{target_format}:{index}"),
                "labels_sha256": contract.text_sha256(f"labels:{skill}:{target_format}:{index}"),
            },
            "source_lineage": {
                "source": f"offline-fixture-{skill}",
                "revision": "fixed-revision",
                "split": "train",
                "adapter": "offline_normalized_fixture_v1",
                "license": "CC0-1.0",
                "source_file_sha256": "e" * 64,
                "source_content_sha256": source_content_sha256,
                "source_id": f"source:{skill}:{target_format}:{index}",
            },
            "reference_evidence": evidence,
        }
        if prefix is not None:
            row["code_prefix"] = prefix
        if target_format == "finance_value_scale":
            row["finance_scale"] = "million"
        return row

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def test_end_to_end_prepare_is_exact_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            run_root.mkdir()
            (run_root / ".day20-run-root").write_text("day20\n", encoding="utf-8")
            source_root = root / "sources"
            source_root.mkdir()

            rows_by_skill: dict[str, list[dict[str, object]]] = {
                skill: [] for skill in contract.SKILLS
            }
            next_index = 0
            for target_format, main_tokens in contract.MAIN_TOKENS_BY_FORMAT.items():
                skill = contract.TARGET_FORMATS[target_format]
                if target_format == "general_mcq":
                    weights = [6] * (main_tokens // 6)
                else:
                    weights = [200] * (main_tokens // 200)
                    if main_tokens % 200:
                        weights.append(main_tokens % 200)
                for supervised_tokens in weights:
                    rows_by_skill[skill].append(
                        self._row(
                            skill,
                            target_format,
                            next_index,
                            supervised_tokens=supervised_tokens,
                        )
                    )
                    next_index += 1
            sources: dict[str, Path] = {}
            for skill, rows in rows_by_skill.items():
                path = source_root / f"{skill}.jsonl"
                self._write_jsonl(path, rows)
                sources[skill] = path

            eval_records = []
            for skill in contract.SKILLS:
                for index in range(28):
                    prompt = f"Frozen {skill} dev question {index}"
                    eval_records.append(
                        {
                            "sample_id": f"eval:{skill}:{index}",
                            "slice": skill,
                            "evaluation_split": "dev",
                            "messages": [{"role": "user", "content": prompt}],
                            "adapted_prompt": prompt,
                            "raw_prompt": prompt,
                            "content_hash": contract.text_sha256(
                                f"eval-content:{skill}:{index}"
                            ),
                            "source_lineage": {
                                "parent_id": f"eval-parent:{skill}:{index}",
                                "candidate_content_hash": contract.text_sha256(
                                    f"eval-candidate:{skill}:{index}"
                                ),
                            },
                        }
                    )
            eval_path = root / "eval.json"
            eval_path.write_text(
                json.dumps({"records": eval_records}, sort_keys=True), encoding="utf-8"
            )

            model_dir = root / "model" / "fixed-revision"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"base-fixture")
            model_snapshot = str(model_dir.resolve())
            audit_sources = {
                skill: {
                    "path": str(path.resolve()),
                    "file_sha256": prepare.file_sha256(path),
                    "records": len(rows_by_skill[skill]),
                    "supervised_tokens": sum(
                        int(row["qwen35_tokenization"]["supervised_tokens"])
                        for row in rows_by_skill[skill]
                    ),
                }
                for skill, path in sources.items()
            }
            source_token_audit = {
                "schema_version": 1,
                "domain": "day20.qwen35_source_token_reaudit",
                "status": "pass",
                "model_path": str(Path(model_snapshot).resolve()),
                "tokenizer_files": {
                    "config.json": "a" * 64,
                    "tokenizer.json": "b" * 64,
                },
                "template": dict(contract.QWEN35_TEMPLATE_CONTRACT),
                "sources": audit_sources,
                "records": sum(len(rows) for rows in rows_by_skill.values()),
                "supervised_tokens": sum(
                    int(row["qwen35_tokenization"]["supervised_tokens"])
                    for rows in rows_by_skill.values()
                    for row in rows
                ),
                "ordered_evidence_sha256": contract.object_sha256(
                    [
                        row["sample_id"]
                        for skill in contract.SKILLS
                        for row in rows_by_skill[skill]
                    ]
                ),
            }
            source_token_audit["audit_sha256"] = contract.object_sha256(
                source_token_audit
            )
            source_adapter_manifest = {
                "schema_version": 1,
                "domain": "day20.qwen35_pinned_source_adapter_manifest",
                "status": "pass",
                "dataset_code_execution": False,
                "implementation": {
                    "source_adapter_file_sha256": prepare.file_sha256(
                        HERE / "day20_source_adapter.py"
                    ),
                    "contract_file_sha256": prepare.file_sha256(
                        HERE / "day20_contract.py"
                    ),
                    "ms_swift_version": "fixture",
                },
                "model_path": str(Path(model_snapshot).resolve()),
                "tokenizer_files": source_token_audit["tokenizer_files"],
                "fixed_sources": {
                    key: {"revision": revision}
                    for key, revision in prepare.PINNED_SOURCE_REVISIONS.items()
                },
                "outputs": {
                    skill: {
                        "path": str(path.resolve()),
                        "file_sha256": prepare.file_sha256(path),
                        "records": len(rows_by_skill[skill]),
                        "supervised_tokens": sum(
                            int(row["qwen35_tokenization"]["supervised_tokens"])
                            for row in rows_by_skill[skill]
                        ),
                    }
                    for skill, path in sources.items()
                },
            }
            source_adapter_manifest["manifest_sha256"] = contract.object_sha256(
                source_adapter_manifest
            )
            source_adapter_manifest_path = (
                source_root / "SOURCE-ADAPTER-MANIFEST.json"
            )
            source_adapter_manifest_path.write_text(
                json.dumps(source_adapter_manifest, sort_keys=True), encoding="utf-8"
            )

            manifest = prepare.prepare_run(
                run_root=run_root,
                model_snapshot=model_snapshot,
                sources=sources,
                eval_manifest_path=eval_path,
                source_token_audit=source_token_audit,
                source_adapter_manifest_path=source_adapter_manifest_path,
            )
            self.assertEqual(
                manifest["datasets"]["probe"]["supervised_tokens"], 16_000
            )
            self.assertEqual(
                manifest["datasets"]["main"]["supervised_tokens"], 256_000
            )
            self.assertEqual(
                manifest["datasets"]["diagnostic"]["records_by_skill"],
                {skill: 8 for skill in contract.SKILLS},
            )
            self.assertTrue(manifest["selection"]["probe_is_subset_of_main"])
            self.assertEqual(
                contract.object_sha256(
                    {key: value for key, value in manifest.items() if key != "manifest_sha256"}
                ),
                manifest["manifest_sha256"],
            )
            first_main = json.loads(
                Path(manifest["datasets"]["main"]["path"])
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("qwen35_supervised_tokens", first_main)
            self.assertEqual(first_main["skill"], "general")
            with self.assertRaises(prepare.Day20PreparationError):
                prepare.prepare_run(
                    run_root=run_root,
                    model_snapshot=model_snapshot,
                    sources=sources,
                    eval_manifest_path=eval_path,
                    source_token_audit=source_token_audit,
                    source_adapter_manifest_path=source_adapter_manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
