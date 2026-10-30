#!/usr/bin/env python3
"""Offline contract tests for the append-only Day 20 v3 materializer."""

from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import prepare_day20_v3 as prepare
from day20_contract_v2 import canonicalize_code_continuation


class _Tokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        pieces = {257: "    ", 101: "return x + 1", 102: "Final answer: A"}
        special = {248046, 198}
        return "".join(
            "" if skip_special_tokens and token in special else pieces.get(token, "")
            for token in token_ids
        )


class _Template:
    def __init__(self, alias: str) -> None:
        self.template_meta = types.SimpleNamespace(template_type=alias)
        self.mode = "train"
        self.max_length = 2304
        self.enable_thinking = False
        self.add_non_thinking_prefix = True
        self.tokenizer = _Tokenizer()

    def encode(self, value: dict[str, object], *, return_length: bool) -> dict[str, object]:
        if return_length is not True:
            raise AssertionError("return_length must be enabled")
        messages = value["messages"]
        target = messages[-1]["content"]
        if target.startswith("    "):
            return {
                "input_ids": [5, 257, 101, 248046, 198],
                "labels": [-100, 257, 101, 248046, -100],
                "length": 5,
            }
        return {
            "input_ids": [5, 102, 248046, 198],
            "labels": [-100, 102, 248046, 198],
            "length": 4,
        }


class Day20PrepareV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parent = self.root / "parent"
        self.parent.mkdir()
        (self.parent / prepare.PARENT_RUN_ROOT_MARKER).write_text(
            prepare.PARENT_RUN_ROOT_MARKER_CONTENT + "\n", encoding="utf-8"
        )
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        self.module_path = self.root / "day20_target_encoding_v3.py"
        self.module_path.write_text("# test identity\n", encoding="utf-8")
        self.template_alias = "day20_qwen3_5_target_v3"
        self.contract = {
            "contract_version": "day20.qwen35_code_boundary_target_v3",
            "template_alias": self.template_alias,
        }
        self.module = self._fake_module()

        self.code = self._code_row()
        self.non_code = self._non_code_row()
        self.rows = [self.non_code, self.code]
        self.diagnostic = [{"sample_id": "eval:fixture", "slice": "code"}]
        data_dir = self.parent / "data"
        data_dir.mkdir()
        self.parent_paths = {
            "probe": data_dir / "probe-v2.jsonl",
            "main": data_dir / "main-v2.jsonl",
            "diagnostic": data_dir / "diagnostic-v2.jsonl",
        }
        self._write_jsonl(self.parent_paths["probe"], self.rows)
        self._write_jsonl(self.parent_paths["main"], self.rows)
        self._write_jsonl(self.parent_paths["diagnostic"], self.diagnostic)
        self.temporal = {"status": "pass", "fixture": True}
        self.dataset_identities = {
            "probe": self._training_dataset_identity("probe"),
            "main": self._training_dataset_identity("main"),
            "diagnostic": self._diagnostic_identity(),
        }
        config_dir = self.parent / "configs"
        config_dir.mkdir()
        self.probe_config = config_dir / "probe-s20260809-lr1e-4.json"
        self.main_config = config_dir / "main-template-v2.json"
        self._write_config(self.probe_config, "probe", self.dataset_identities["probe"])
        self._write_config(self.main_config, "main", self.dataset_identities["main"])
        self.manifest_path = self.parent / prepare.CORE_MANIFEST_NAME
        manifest = {
            "schema_version": 2,
            "domain": prepare.CORE_MANIFEST_DOMAIN,
            "status": "prepared_stage_a_not_started",
            "created_at_utc": "2026-08-12T00:00:00+00:00",
            "run_root": str(self.parent.resolve()),
            "contract": {"candidate_factory_version": "fixture-v2"},
            "base_model_identity": {
                "path": str(self.model.resolve()),
                "snapshot_sha256": "a" * 64,
                "files": {},
            },
            "source_expansion": {"status": "fixture"},
            "datasets": self.dataset_identities,
            "configs": {
                "probes": {"1e-4": self._config_identity(self.probe_config)},
                "main_template": self._config_identity(self.main_config),
            },
            "leakage": {"status": "pass"},
            "selection": {"probe_is_subset_of_main": True},
        }
        manifest["manifest_sha256"] = prepare.object_sha256(manifest)
        self._write_json(self.manifest_path, manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _code_row(self) -> dict[str, object]:
        prefix = "def solve(x):"
        target = "    return x + 1"
        messages = [
            {
                "role": "user",
                "content": "Complete the Python function below. Return only the indented Python continuation: fixture",
            },
            {"role": "assistant", "content": target},
        ]
        return {
            "sample_id": "train:code:fixture",
            "skill": "code",
            "target_format": "code_continuation",
            "messages": messages,
            "code_prefix": prefix,
            "code_continuation_v2": canonicalize_code_continuation(
                target, prefix, require_raw_contract=True
            ).as_evidence(),
            "source_content_sha256": "b" * 64,
            "qwen35_input_tokens": 4,
            "qwen35_supervised_tokens": 3,
            "qwen35_render_sha256": "c" * 64,
            "qwen35_labels_sha256": "d" * 64,
        }

    def _non_code_row(self) -> dict[str, object]:
        input_ids = [5, 102, 248046, 198]
        labels = [-100, 102, 248046, 198]
        return {
            "sample_id": "train:general:fixture",
            "skill": "general",
            "target_format": "general_mcq",
            "messages": [
                {"role": "user", "content": "Fixture question"},
                {"role": "assistant", "content": "Final answer: A"},
            ],
            "source_content_sha256": "e" * 64,
            "qwen35_input_tokens": len(input_ids),
            "qwen35_supervised_tokens": 3,
            "qwen35_render_sha256": prepare.object_sha256(input_ids),
            "qwen35_labels_sha256": prepare.object_sha256(labels),
        }

    def _training_dataset_identity(self, label: str) -> dict[str, object]:
        path = self.parent_paths[label]
        ids = [row["sample_id"] for row in self.rows]
        tokens = [row["qwen35_supervised_tokens"] for row in self.rows]
        return {
            "path": str(path.resolve()),
            "file_sha256": prepare.file_sha256(path),
            "records": 2,
            "supervised_tokens": 6,
            "supervised_tokens_by_skill": {"code": 3, "general": 3},
            "supervised_tokens_by_format": {
                "code_continuation": 3,
                "general_mcq": 3,
            },
            "records_by_format": {"code_continuation": 1, "general_mcq": 1},
            "cohort_supervised_tokens": {},
            "ordered_sample_ids": ids,
            "ordered_supervised_tokens": tokens,
            "selection_sha256": prepare.object_sha256(ids),
            "ordering_version": "fixture-order-v2",
            "temporal_mix_audit": self.temporal,
            "temporal_mix_sha256": prepare.object_sha256(self.temporal),
        }

    def _diagnostic_identity(self) -> dict[str, object]:
        path = self.parent_paths["diagnostic"]
        return {
            "path": str(path.resolve()),
            "file_sha256": prepare.file_sha256(path),
            "records": 1,
            "records_by_skill": {"code": 1},
            "ordered_sample_ids": ["eval:fixture"],
            "selection_sha256": prepare.object_sha256(["eval:fixture"]),
            "eval_manifest_path": str((self.root / "eval.json").resolve()),
            "eval_manifest_file_sha256": "f" * 64,
        }

    def _write_config(
        self, path: Path, run_kind: str, dataset: dict[str, object]
    ) -> None:
        config = {
            "schema_version": 2,
            "domain": prepare.CORE_CONFIG_DOMAIN,
            "run_kind": run_kind,
            "requires_resolution": run_kind == "main",
            "model": {"snapshot": str(self.model.resolve()), "model_type": "qwen3_5"},
            "data": {
                "path": dataset["path"],
                "file_sha256": dataset["file_sha256"],
                "supervised_tokens": dataset["supervised_tokens"],
                "temporal_mix_sha256": dataset["temporal_mix_sha256"],
            },
            "template": {
                "template": prepare.NATIVE_TEMPLATE_TYPE,
                "loss_scale": prepare.LOSS_SCALE,
                "max_length": 2304,
            },
            "training": {
                "learning_rate": "__SELECT_FROM_PASSING_PROBE__"
                if run_kind == "main"
                else "1e-4",
                "checkpoint_supervised_tokens": [3, 6],
            },
        }
        config["immutable_sha256"] = prepare.object_sha256(config)
        self._write_json(path, config)

    @staticmethod
    def _config_identity(path: Path) -> dict[str, object]:
        config = json.loads(path.read_text(encoding="utf-8"))
        return {
            "path": str(path.resolve()),
            "file_sha256": prepare.file_sha256(path),
            "immutable_sha256": config["immutable_sha256"],
            "learning_rate": config["training"]["learning_rate"],
            "seed": 20260809,
        }

    def _fake_module(self) -> types.ModuleType:
        module = types.ModuleType("fake_day20_target_encoding_v3")
        module.__file__ = str(self.module_path)
        module.TARGET_TEMPLATE_TYPE = self.template_alias
        module.TARGET_ENCODING_CONTRACT_VERSION = self.contract["contract_version"]
        module.TARGET_ENCODING_CONTRACT = self.contract
        module.TARGET_ENCODING_CONTRACT_SHA256 = prepare.object_sha256(self.contract)
        module.register_target_template_v3 = lambda: None
        module.build_target_template_v3 = lambda model, max_length=2304: _Template(
            self.template_alias
        )
        module.build_native_template_v3 = lambda model, max_length=2304: _Template(
            prepare.NATIVE_TEMPLATE_TYPE
        )

        def audit(
            rows: list[dict[str, object]],
            *,
            native_template: object,
            target_template: object,
            expected_total_supervised_tokens: int,
        ) -> dict[str, object]:
            del native_template, target_template
            total = sum(int(row["qwen35_supervised_tokens"]) for row in rows)
            if total != expected_total_supervised_tokens:
                raise ValueError("fixture total drifted")
            return {
                "status": "pass",
                "records": len(rows),
                "target_supervised_tokens": total,
                "audit_sha256": "0" * 64,
            }

        module.audit_target_encoding_v3 = audit
        return module

    def _patches(self) -> list[mock._patch]:
        return [
            mock.patch.object(prepare, "_import_target_module", return_value=self.module),
            mock.patch.object(
                prepare,
                "_verify_ms_swift",
                return_value={
                    "root": "/fixture/ms-swift",
                    "commit": "1" * 40,
                    "worktree_clean": True,
                    "qwen_template": {"path": "/fixture/qwen.py", "file_sha256": "2" * 64},
                },
            ),
            mock.patch.object(prepare, "audit_temporal_mix", return_value=self.temporal),
            mock.patch.object(prepare, "supervised_token_schedule", return_value=[6]),
            mock.patch.object(
                prepare,
                "checkpoint_steps",
                side_effect=lambda values, run_kind: {
                    "run_kind": run_kind,
                    "supervised_tokens": sum(values),
                    "checkpoint_steps": {"fixture": 1},
                },
            ),
        ]

    def _prepare(self, output: Path) -> dict[str, object]:
        patches = self._patches()
        for patcher in patches:
            patcher.start()
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patches)])
        return prepare.prepare_run_v3(
            parent_run_root=self.parent,
            run_root=output,
            target_encoding_module=self.module_path,
            expected_ms_swift_commit="1" * 40,
        )

    def test_materializes_v2_core_and_v3_evidence_without_reselection(self) -> None:
        parent_hash = prepare.file_sha256(self.manifest_path)
        output = self.root / "v3"
        result = self._prepare(output)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(
            result["core_v2"]["manifest"]["path"],
            str((output / prepare.CORE_MANIFEST_NAME).resolve()),
        )
        self.assertEqual(result["target_encoding"]["template_alias"], self.template_alias)
        self.assertEqual(prepare.file_sha256(self.manifest_path), parent_hash)
        self.assertEqual(
            (output / prepare.RUN_ROOT_MARKER).read_text(encoding="utf-8").strip(),
            prepare.RUN_ROOT_MARKER_CONTENT,
        )

        probe = prepare._load_jsonl(output / "data" / "probe-v2.jsonl")
        self.assertEqual([row["sample_id"] for row in probe], [row["sample_id"] for row in self.rows])
        self.assertEqual([row["qwen35_supervised_tokens"] for row in probe], [3, 3])
        self.assertEqual(probe[0], self.non_code)
        expected_code = copy.deepcopy(self.code)
        expected_code.update(
            {
                "qwen35_input_tokens": 5,
                "qwen35_render_sha256": prepare.object_sha256([5, 257, 101, 248046, 198]),
                "qwen35_labels_sha256": prepare.object_sha256([-100, 257, 101, 248046, -100]),
            }
        )
        self.assertEqual(probe[1], expected_code)
        self.assertEqual(
            prepare.file_sha256(output / "data" / "diagnostic-v2.jsonl"),
            prepare.file_sha256(self.parent_paths["diagnostic"]),
        )

        core = prepare._load_json(output / prepare.CORE_MANIFEST_NAME)
        self.assertEqual(
            core["manifest_sha256"],
            prepare.object_sha256({key: value for key, value in core.items() if key != "manifest_sha256"}),
        )
        self.assertEqual(core["datasets"]["probe"]["ordered_sample_ids"], self.dataset_identities["probe"]["ordered_sample_ids"])
        probe_config = prepare._load_json(output / "configs" / self.probe_config.name)
        self.assertEqual(probe_config["template"]["template"], self.template_alias)
        self.assertEqual(
            probe_config["data"]["path"],
            str((output / "data" / "probe-v2.jsonl").resolve()),
        )

        audit = prepare._load_json(output / prepare.TARGET_AUDIT_NAME)
        self.assertEqual(audit["audit_sha256"], prepare.object_sha256({key: value for key, value in audit.items() if key != "audit_sha256"}))
        self.assertTrue(audit["datasets"]["diagnostic"]["copy_identical"])
        equivalence = prepare._load_json(output / prepare.SELECTION_EQUIVALENCE_NAME)
        self.assertEqual(equivalence["status"], "pass")
        self.assertEqual(equivalence["datasets"]["probe"]["status"], "equivalent")

    def test_refuses_overwrite_and_nonempty_output(self) -> None:
        output = self.root / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("user data\n", encoding="utf-8")
        with self.assertRaisesRegex(prepare.Day20PreparationV3Error, "must be empty"):
            self._prepare(output)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user data\n")

    def test_refuses_symbolic_output_root(self) -> None:
        real = self.root / "real-output"
        real.mkdir()
        symbolic = self.root / "symbolic-output"
        symbolic.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(prepare.Day20PreparationV3Error, "must not be symbolic"):
            prepare._validate_output_root(symbolic, [])

    def test_non_code_hash_drift_fails_closed(self) -> None:
        bad = copy.deepcopy(self.non_code)
        bad["qwen35_labels_sha256"] = "0" * 64
        with self.assertRaisesRegex(prepare.Day20PreparationV3Error, "non-Code token arrays changed"):
            prepare._reencode_dataset([bad], target_template=_Template(self.template_alias), label="probe")

    def test_parent_manifest_tamper_fails_closed(self) -> None:
        manifest = prepare._load_json(self.manifest_path)
        manifest["status"] = "tampered"
        self._write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(prepare.Day20PreparationV3Error, "manifest_sha256 mismatch"):
            prepare._validate_parent(self.parent)


if __name__ == "__main__":
    unittest.main()
