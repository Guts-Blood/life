#!/usr/bin/env python3
"""Focused stdlib tests for the Day 23 RSI-v0004 capacity correction."""

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

import build_day23_rsi_v0004 as builder  # noqa: E402
import eval_day23_rsi_candidate as evaluator  # noqa: E402
import run_day23_rsi_candidate as runner  # noqa: E402


def _seal(value: dict, field: str) -> dict:
    sealed = dict(value)
    sealed[field] = runner.day23_gpu.object_sha256(sealed)
    return sealed


class StopBeforeModelLoad(RuntimeError):
    pass


class Day23RSIV0004Tests(unittest.TestCase):
    def _campaign(self) -> dict:
        return {
            "version_id": "rsi-v0004",
            "fixed_recipe": {
                "runtime": {
                    "per_device_train_batch_size": 8,
                    "gradient_accumulation_steps": 2,
                    "min_free_memory_fraction": 0.20,
                }
            },
        }

    def test_dynamic_schema_and_v0004_runtime_profile(self) -> None:
        campaign = self._campaign()
        self.assertEqual(builder.GPU_SCHEMA, "day23.rsi_v0004_gpu_campaign")
        self.assertEqual(builder.VERSION_ID, "rsi-v0004")
        self.assertEqual(runner.campaign_schema(campaign["version_id"]), builder.GPU_SCHEMA)
        self.assertEqual(
            runner.receipt_schema(campaign["version_id"]),
            "day23.rsi_v0004_training_receipt",
        )
        self.assertEqual(
            evaluator._schema({"campaign": campaign}, "access_claim"),
            "day23.rsi_v0004_access_claim",
        )
        self.assertEqual(runner._runtime_profile(campaign), (8, 2))

        drifted = json.loads(json.dumps(campaign))
        drifted["fixed_recipe"]["runtime"]["min_free_memory_fraction"] = 0.30
        with self.assertRaisesRegex(runner.RSICandidateError, "runtime profile"):
            runner._runtime_profile(drifted)

    def test_builder_emits_b8_ga2_schema_v1_and_twenty_percent_gate(self) -> None:
        # Keep the fixture below the repository root so the production
        # builder's real producer files remain relative to ``bootcamp``.
        with tempfile.TemporaryDirectory(
            prefix="day23-rsi-v0004-builder-", dir=Path.cwd()
        ) as raw:
            root = Path(raw).resolve()
            bootcamp = Path.cwd().resolve()
            source_run = root / "v0003"
            source_binding = source_run / "binding"
            new_run = root / "v0004"
            for path in (source_binding, new_run):
                path.mkdir(parents=True)

            charter_path = bootcamp / "charter.json"
            charter_path.write_text(
                json.dumps(
                    {
                        "access_leases": {
                            "heldout": {
                                "claim_path": f"ledger/{root.name}/heldout-claim.json"
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            python_path = root / "python"
            python_path.write_bytes(b"test-python")
            checkout = root / "ms-swift"
            checkout.mkdir()

            run_specs = {}
            for run_id, learning_rate in (
                ("search_lr_1e_6", 1e-6),
                ("search_lr_5e_6", 5e-6),
            ):
                config_path = source_binding / f"{run_id}.json"
                config = {
                    "per_device_train_batch_size": 16,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": learning_rate,
                    "output_dir": str(source_run / "outputs" / run_id),
                }
                config_path.write_bytes(builder._bytes(config))
                run_specs[run_id] = {
                    "role": "search_train",
                    "learning_rate": learning_rate,
                    "max_steps": 30,
                    "checkpoint_steps": [15, 30],
                    "executable_config": {
                        "path": str(config_path),
                        "file_sha256": runner.day23_gpu.file_sha256(config_path),
                    },
                    "output_dir": config["output_dir"],
                    "success_receipt": str(source_run / "evidence" / run_id / "success.json"),
                    "failure_receipt": str(source_run / "evidence" / run_id / "failure.json"),
                }

            source = {
                "schema_name": builder.SOURCE_SCHEMA,
                "schema_version": 1,
                "status": "gpu_execution_bound_optimizer_pending",
                "campaign_id": "day23-qwen35-dpo-rsi-v0003",
                "version_id": builder.SOURCE_VERSION,
                "goal_id": builder.GOAL_ID,
                "bootcamp_root": str(bootcamp),
                "remote_run_root": str(source_run),
                "charter": {"path": str(charter_path.relative_to(bootcamp))},
                "run_specs": run_specs,
                "runtime_parse": {
                    "python_executable": {"path": str(python_path)},
                    "ms_swift_checkout": str(checkout),
                    "package_versions": {},
                },
                "fixed_recipe": {
                    "runtime": {
                        "per_device_train_batch_size": 16,
                        "gradient_accumulation_steps": 1,
                        "nominal_global_train_batch_size": 32,
                        "min_free_memory_fraction": 0.30,
                    },
                    "topology": {
                        "per_device_train_batch_size": 16,
                        "gradient_accumulation_steps": 1,
                        "nominal_global_train_batch_size": 32,
                    },
                },
                "implementation_sources": {},
                "access_ledger": {
                    "search_claim": str(source_run / "evidence/search-claim.json"),
                    "dev_claim": str(bootcamp / "rsi-control/ledger/dev-claim.json"),
                },
                "retry_policy": {},
                "budgets": {},
                "claim_boundary": {},
            }
            source = _seal(source, "campaign_sha256")
            source_path = source_binding / "gpu-campaign.json"
            source_path.write_bytes(builder._bytes(source))

            failure_path = Path(run_specs["search_lr_1e_6"]["failure_receipt"])
            failure_path.parent.mkdir(parents=True)
            failure = _seal(
                {
                    "status": "fail",
                    "run_id": "search_lr_1e_6",
                    "campaign": {"campaign_sha256": source["campaign_sha256"]},
                    "observations": {
                        "global_step": 30,
                        "memory": {"minimum_observed_device_free_fraction": 0.162},
                    },
                    "error": {
                        "message": "observed free VRAM fell below the campaign gate"
                    },
                    "claim_boundary": {
                        "scientific_result_available": True,
                        "dev_consumed": False,
                        "heldout_consumed": False,
                    },
                },
                "receipt_sha256",
            )
            failure_path.write_bytes(builder._bytes(failure))

            runtime_parse = {
                "status": "pass",
                "python_executable": {"path": str(python_path)},
                "ms_swift_checkout": str(checkout),
                "package_versions": {},
                "stages": {},
            }
            control = {
                "amendment": {},
                "runtime_failure_event": {},
                "version": {},
                "precommit_event": {},
            }
            with (
                mock.patch.object(builder, "_parse_configs", return_value=runtime_parse),
                mock.patch.object(builder, "_control_artifacts", return_value=control),
            ):
                result = builder.build(
                    argparse.Namespace(
                        source_campaign=source_path,
                        source_failure=failure_path,
                        run_root=new_run,
                        preflight_only=False,
                    )
                )

            campaign_path = Path(result["campaign"])
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            runtime = campaign["fixed_recipe"]["runtime"]
            self.assertEqual(
                (campaign["schema_name"], campaign["schema_version"], campaign["version_id"]),
                (builder.GPU_SCHEMA, 1, builder.VERSION_ID),
            )
            self.assertEqual(
                (
                    runtime["per_device_train_batch_size"],
                    runtime["gradient_accumulation_steps"],
                    runtime["nominal_global_train_batch_size"],
                    runtime["min_free_memory_fraction"],
                ),
                (8, 2, 32, 0.20),
            )
            self.assertEqual(
                runner.day23_gpu.object_sha256(campaign, "campaign_sha256"),
                campaign["campaign_sha256"],
            )
            for spec in campaign["run_specs"].values():
                config = json.loads(
                    Path(spec["executable_config"]["path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    (
                        config["per_device_train_batch_size"],
                        config["gradient_accumulation_steps"],
                    ),
                    (8, 2),
                )

    def test_b8_ga2_realized_profile_includes_epoch_remainder(self) -> None:
        context = {
            "spec": {"max_steps": 5},
            "records": 124,
            "campaign": self._campaign(),
        }
        ranks = []
        microbatches = ((8, 8), (8, 8), (8, 8), (8, 6), (8, 8))
        for rank in (0, 1):
            observations = [
                {"global_step_before": step, "local_pair_batch_size": size}
                for step, sizes in enumerate(microbatches)
                for size in sizes
            ]
            ranks.append(
                {
                    "rank": rank,
                    "observations": {
                        "gradient_accumulation": {"observations": observations}
                    },
                }
            )

        profile = runner._realized_batch_profile(ranks, context)
        self.assertEqual(profile["per_step_global_pair_counts"], [32, 32, 32, 28, 32])
        self.assertEqual(profile["unique"], [28, 32])
        self.assertTrue(profile["partial_batch_present"])
        self.assertEqual(profile["total_pair_exposures"], 156)

    def test_twenty_percent_memory_gate_is_inclusive(self) -> None:
        campaign = self._campaign()
        spec = {"max_steps": 1}
        dataset = {"records": 124}
        observations = {
            "global_step": 1,
            "trainable_inventory": {
                "tensors": runner.EXPECTED_LORA_TENSORS,
                "params": runner.EXPECTED_LORA_PARAMS,
            },
            "optimizer_inventory": {
                "parameter_tensors": runner.EXPECTED_LORA_TENSORS,
                "parameter_numel": runner.EXPECTED_LORA_PARAMS,
                "exactly_trainable_inventory": True,
            },
            "freeze": {"all_versions_unchanged": True, "version_changes": []},
            "lora": {"changed": True, "initial_digest": "a", "final_digest": "b"},
            "reference": {
                "context_count": 1,
                "implementation": "same_peft_model_disable_adapter",
            },
            "forwards": {"counts": {"policy": 1, "reference": 1}},
            "logs": {"nonfinite": [], "seen_numeric_keys": sorted(runner.REQUIRED_LOG_KEYS)},
            "dataloader": {"records": 124, "configured_per_device_batch": 8},
            "gradient_accumulation": {
                "observations": [
                    {
                        "configured_gradient_accumulation_steps": 2,
                        "current_gradient_accumulation_steps": 2,
                        "local_pair_batch_size": 8,
                    }
                ]
            },
            "memory": {
                "minimum_observed_device_free_fraction": 0.20,
                "device_peaks": [{"conservative_peak_free_fraction": 0.20}],
            },
            "violations": [],
        }
        runner._validate_live_observations(
            campaign, spec, dataset, observations, "boundary observation"
        )

        below = json.loads(json.dumps(observations))
        below["memory"]["minimum_observed_device_free_fraction"] = 0.199999
        with self.assertRaisesRegex(runner.RSICandidateError, "free-memory gate"):
            runner._validate_live_observations(
                campaign, spec, dataset, below, "below-gate observation"
            )

    def test_search_grid_receipts_are_strictly_validated_before_claim(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day23-rsi-v0004-order-") as raw:
            root = Path(raw)
            campaign_sha = "c" * 64
            run_specs = {}
            for run_id in ("search_lr_1e_6", "search_lr_5e_6"):
                success = root / run_id / "success.json"
                failure = root / run_id / "failure.json"
                success.parent.mkdir(parents=True)
                receipt = _seal(
                    {
                        "schema_name": runner.receipt_schema("rsi-v0004"),
                        "schema_version": 1,
                        "status": "pass",
                        "run_id": run_id,
                        "campaign": {"campaign_sha256": campaign_sha},
                    },
                    "receipt_sha256",
                )
                success.write_bytes(builder._bytes(receipt))
                run_specs[run_id] = {
                    "role": "search_train",
                    "checkpoint_steps": [15, 30],
                    "success_receipt": str(success),
                    "failure_receipt": str(failure),
                }

            campaign = {
                **self._campaign(),
                "run_specs": run_specs,
            }
            context = {
                "campaign": campaign,
                "campaign_path": root / "gpu-campaign.json",
                "campaign_sha256": campaign_sha,
            }
            output = root / "search-evaluation.json"
            events = []

            def strict_receipt(campaign_arg, campaign_path, receipt, receipt_path):
                del campaign_arg, campaign_path, receipt_path
                run_id = receipt["run_id"]
                events.append(f"strict:{run_id}")
                return {"run_id": run_id, "spec": run_specs[run_id]}

            def claim(context_arg):
                del context_arg
                events.append("claim")
                return {"status": "claimed"}

            with (
                mock.patch.object(evaluator, "validate_campaign", return_value=context),
                mock.patch.object(
                    evaluator,
                    "_checkpoint_from_receipt",
                    return_value={
                        "run_id": "search_lr_1e_6",
                        "spec": run_specs["search_lr_1e_6"],
                    },
                ),
                mock.patch.object(
                    evaluator,
                    "_expected_evaluation_path",
                    return_value=output.resolve(),
                ),
                mock.patch.object(
                    runner,
                    "validate_training_receipt_value",
                    side_effect=strict_receipt,
                ),
                mock.patch.object(
                    evaluator, "_claim_or_validate_search", side_effect=claim
                ),
                mock.patch.object(
                    evaluator,
                    "_dataset_rows",
                    side_effect=StopBeforeModelLoad("stop after claim ordering proof"),
                ),
            ):
                with self.assertRaises(StopBeforeModelLoad):
                    evaluator.run_evaluation(
                        root / "campaign.json",
                        root / "requested-receipt.json",
                        15,
                        "search",
                        output,
                        None,
                        "cuda:0",
                    )

            self.assertEqual(
                events,
                [
                    "strict:search_lr_1e_6",
                    "strict:search_lr_5e_6",
                    "claim",
                ],
            )


if __name__ == "__main__":
    unittest.main()
