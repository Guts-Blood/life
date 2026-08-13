#!/usr/bin/env python3
"""Offline tests for the Day 20 v3 external training plugin."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import day20_train_plugin_v3 as plugin
import day20_train_runtime_v3 as runtime
import day20_target_encoding_v3 as target_encoding


class Day20V3TrainPluginTests(unittest.TestCase):
    def test_registration_order_is_alias_then_v2_evidence_then_v3_binding(self) -> None:
        source = (HERE / "day20_train_plugin_v3.py").read_text(encoding="utf-8")
        body = source[
            source.index("def _register_external_plugin()") :
            source.index("def seal_summary(")
        ]
        alias = body.index("register_target_template_v3()")
        v2 = body.index("import day20_train_plugin_v2")
        binding = body.index("_register_target_binding_callback()")
        self.assertLess(alias, v2)
        self.assertLess(v2, binding)
        self.assertIn('DAY20_V2_REGISTER_SWIFT_CALLBACK") == "1"', body)
        self.assertEqual(plugin.TARGET_BINDING_CALLBACK_NAME, "day20_v3_target_binding")

    def test_live_template_alias_is_exact_and_ambiguous_metadata_fails(self) -> None:
        template = SimpleNamespace(
            template_meta=SimpleNamespace(
                template_type=runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
                name=runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
            ),
            template_type=runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
        )
        self.assertEqual(
            plugin._live_template_alias(template),
            runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
        )
        template.template_type = "qwen3_5"
        with self.assertRaises(plugin.Day20TrainPluginV3Error):
            plugin._live_template_alias(template)

    def test_training_arguments_without_sft_fields_pass_and_real_drift_fails(self) -> None:
        values = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "learning_rate": 1e-4,
            "num_train_epochs": 1.0,
            "seed": 20260809,
            "data_seed": 20260809,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.05,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "save_only_model": False,
            "save_total_limit": 4,
            "save_strategy": "no",
            "train_dataloader_shuffle": False,
        }
        args = SimpleNamespace(**values)
        for sft_field in (
            "model_type",
            "template",
            "enable_thinking",
            "loss_scale",
            "max_length",
            "packing",
        ):
            self.assertFalse(hasattr(args, sft_field))
        actual = plugin._trainer_arguments(args)
        self.assertEqual(actual, values)
        plugin._validate_live_trainer_arguments(
            actual, expected=values
        )
        drifted = plugin._trainer_arguments(
            SimpleNamespace(**dict(values, learning_rate=3e-4))
        )
        with self.assertRaises(plugin.Day20TrainPluginV3Error):
            plugin._validate_live_trainer_arguments(
                drifted, expected=values
            )

    def test_live_template_fields_come_from_template_and_bound_config(self) -> None:
        class LiveTargetTemplate:
            day20_target_encoding_contract_sha256 = (
                target_encoding.TARGET_ENCODING_CONTRACT_SHA256
            )

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "training-config.json"
            template_contract = {
                "model_type": "qwen3_5",
                "template": runtime.EXPECTED_TARGET_TEMPLATE_ALIAS,
                "enable_thinking": False,
                "add_non_thinking_prefix": True,
                "loss_scale": "default+ignore_empty_think",
                "max_length": 2304,
                "truncation_strategy": "raise",
                "packing": False,
                "padding_free": False,
            }
            config_path.write_text(
                json.dumps({"template": template_contract}) + "\n",
                encoding="utf-8",
            )
            binding = {
                "training_config": {"path": str(config_path)},
                "target_module": {
                    "path": str(Path(target_encoding.__file__).resolve()),
                    "template_alias": target_encoding.TARGET_TEMPLATE_ALIAS,
                    "contract_version": target_encoding.TARGET_ENCODING_CONTRACT_VERSION,
                    "contract_sha256": target_encoding.TARGET_ENCODING_CONTRACT_SHA256,
                },
            }
            template = LiveTargetTemplate()
            template.template_meta = SimpleNamespace(
                template_type=runtime.EXPECTED_TARGET_TEMPLATE_ALIAS
            )
            template.model_info = SimpleNamespace(model_type="qwen3_5")
            template.enable_thinking = False
            template.add_non_thinking_prefix = True
            template._loss_scale = "default+ignore_empty_think"
            template.max_length = 2304
            template.truncation_strategy = "raise"
            template.packing = False
            template.padding_free = False
            with patch.object(target_encoding, "_assert_template_contract") as check:
                plugin._validate_live_target_template(template, binding=binding)
            check.assert_called_once_with(template)
            template.max_length = 2048
            with self.assertRaises(plugin.Day20TrainPluginV3Error):
                plugin._validate_live_target_template(template, binding=binding)

    def test_summary_sealing_writes_envelopes_outside_checkpoints_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary_path = root / "evidence" / "training-summary.json"
            binding = root / "evidence" / "binding.json"
            attestation = root / "evidence" / "attestation.json"
            output = root / "evidence" / "checkpoint-envelopes"
            checkpoint = root / "adapters" / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            for path in (summary_path, binding, attestation):
                path.write_text("{}\n", encoding="utf-8")
            summary = {
                "candidate_ids": {"t6000": "probe-s20260809-lr1e-4-t6000"},
                "checkpoints": {"t6000": str(checkpoint)},
            }
            envelope = {
                "schema_version": 3,
                "domain": runtime.CHECKPOINT_ENVELOPE_DOMAIN,
                "status": "complete",
                "candidate": "probe-s20260809-lr1e-4-t6000",
                "envelope_sha256": "a" * 64,
            }
            with patch(
                "day20_train_plugin_v2.verify_training_summary",
                return_value=summary,
            ), patch.object(
                plugin, "build_checkpoint_envelope", return_value=envelope
            ):
                first = plugin.seal_summary(
                    summary_path=summary_path,
                    run_root=root,
                    binding_config_path=binding,
                    attestation_path=attestation,
                    output_dir=output,
                )
            envelope_path = Path(first["envelopes"]["t6000"]["path"])
            self.assertTrue(envelope_path.is_file())
            self.assertNotEqual(envelope_path.parent, checkpoint)
            with patch(
                "day20_train_plugin_v2.verify_training_summary",
                return_value=summary,
            ), patch.object(
                plugin, "build_checkpoint_envelope", return_value=envelope
            ), patch.object(
                plugin, "verify_checkpoint_envelope", return_value=envelope
            ):
                second = plugin.seal_summary(
                    summary_path=summary_path,
                    run_root=root,
                    binding_config_path=binding,
                    attestation_path=attestation,
                    output_dir=output,
                )
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
