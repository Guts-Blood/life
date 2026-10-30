#!/usr/bin/env python3
"""Build the frozen CPU-to-GPU run contract for Day 25 coding GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract


DEFAULT_DATA = contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-data-manifest.json"
DEFAULT_PROCESSOR = contract.ARTIFACTS / "eval/day25-qwen35-coding-grpo-processor-audit-summary.json"
DEFAULT_OUTPUT = contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/cpu-run-contract.json"
DAY24_VERIFIER = contract.BOOTCAMP_ROOT / "day-24-online-rl-dataflow-reward/day24_coding_verifier.py"
DAY24_SCHEMA = contract.ARTIFACTS / "data/day24-coding-trajectory-schema-v2.json"
ADVANTAGE_SOURCE = contract.REPO_ROOT / "vendor/ms-swift/swift/rl_core/advantage.py"
REWARD_BRIDGE_SOURCE = contract.REPO_ROOT / "vendor/ms-swift/swift/rl_core/grpo_algorithm.py"


class Day25CPUContractError(ValueError):
    """The Day 25 CPU run contract cannot be safely frozen."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25CPUContractError(message)


def _entry(path: Path, hash_field: str) -> dict[str, Any]:
    value = contract.load_json(path.resolve())
    content_hash = contract.verify_seal(value, hash_field, path.name)
    return {
        "path": contract.relative_to_bootcamp(path),
        "file_sha256": contract.file_sha256(path),
        "content_sha256": content_hash,
    }


def build_contract(
    *,
    data_manifest_path: Path = DEFAULT_DATA,
    processor_summary_path: Path = DEFAULT_PROCESSOR,
) -> dict[str, Any]:
    data_manifest_path = data_manifest_path.resolve()
    processor_summary_path = processor_summary_path.resolve()
    data_manifest = contract.load_json(data_manifest_path)
    contract.verify_seal(data_manifest, "manifest_sha256", "Day 25 data manifest")
    processor = contract.load_json(processor_summary_path)
    contract.verify_seal(processor, "summary_sha256", "Day 25 processor audit")
    _require(data_manifest.get("status") == "cpu_data_ready", "data manifest is not ready")
    _require(processor.get("status") == "pass" and processor.get("records") == 72, "processor audit did not pass 72 rows")
    _require(processor.get("max_prompt_token_count", 9999) <= 512, "prompt token cap failed")

    source_paths = {
        "day25_contract.py": Path(contract.__file__).resolve(),
        "build_day25_cpu_contract.py": contract.DAY25_DIR / "build_day25_cpu_contract.py",
        "prepare_day25_qwen35_grpo.py": contract.DAY25_DIR / "prepare_day25_qwen35_grpo.py",
        "day25_reward_adapter.py": contract.DAY25_DIR / "day25_reward_adapter.py",
        "day25_ms_swift_plugin.py": contract.DAY25_DIR / "day25_ms_swift_plugin.py",
        "audit_day25_rewards.py": contract.DAY25_DIR / "audit_day25_rewards.py",
        "audit_day25_qwen35_processor.py": contract.DAY25_DIR / "audit_day25_qwen35_processor.py",
        "audit_day25_qwen35_arguments.py": contract.DAY25_DIR / "audit_day25_qwen35_arguments.py",
        "test_day25_reward_adapter.py": contract.DAY25_DIR / "test_day25_reward_adapter.py",
        "day24_coding_verifier.py": DAY24_VERIFIER,
        "pinned_ms_swift_advantage.py": ADVANTAGE_SOURCE,
        "pinned_ms_swift_grpo_algorithm.py": REWARD_BRIDGE_SOURCE,
    }
    for label, path in source_paths.items():
        _require(path.is_file(), f"implementation source is missing: {label}")

    common_args: dict[str, Any] = {
        "rlhf_type": "grpo",
        "model": "__BIND_VERIFIED_REMOTE_S1_MERGED_EXPORT__",
        "model_type": "qwen3_5",
        "template": "qwen3_5",
        "use_hf": True,
        "tuner_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_regex": contract.TARGET_REGEX,
        "freeze_llm": False,
        "freeze_vit": True,
        "freeze_aligner": True,
        "dataset": ["artifacts/data/day25-qwen35-coding-grpo-train.jsonl"],
        "split_dataset_ratio": 0,
        "strict": True,
        "disable_auto_column_mapping": True,
        "remove_unused_columns": False,
        "dataset_shuffle": False,
        "train_dataloader_shuffle": False,
        "load_from_cache_file": False,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "last_round",
        "packing": False,
        "padding_free": False,
        "max_length": 1024,
        "max_completion_length": 512,
        "truncation_strategy": "delete",
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_enable_lora": True,
        "vllm_gpu_memory_utilization": 0.30,
        "vllm_tensor_parallel_size": 1,
        "vllm_max_num_seqs": 8,
        "vllm_max_model_len": 1152,
        "vllm_enable_prefix_caching": True,
        "sleep_level": 1,
        "offload_model": False,
        "offload_optimizer": False,
        "async_generate": False,
        "num_generations": 4,
        "generation_batch_size": 8,
        "num_iterations": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "beta": 0.0,
        "advantage_estimator": "grpo",
        "scale_rewards": "group",
        "kl_in_reward": False,
        "epsilon": 0.2,
        "reward_funcs": ["day25_mbpp_tests_only"],
        "external_plugins": ["day-25-grpo-small-model-lab/day25_ms_swift_plugin.py"],
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "bf16": True,
        "logging_steps": 1,
        "log_completions": True,
        "log_entropy": True,
        "log_rollout_offpolicy_metrics": True,
        "report_to": ["tensorboard"],
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 1,
        "save_total_limit": 2,
        "save_only_model": False,
        "add_version": False,
        "seed": 20260820,
        "data_seed": 20260820,
    }
    stages = {
        "g1_rollout_only": {
            "max_steps": 1,
            "output_dir": "__BIND_DAY25_G1_OUTPUT_DIR__",
            "policy_version": "v0_rollout_only_intentional_stop",
            "purpose": "persist_and_audit_one_live_rollout_batch_before_optimizer_authorization",
            "claim_boundary": "DAY25_ABORT_AFTER_REWARD_BATCH=1 must stop inside reward before scalars return",
        },
        "g2_one_update": {
            "max_steps": 1,
            "output_dir": "__BIND_DAY25_G2_OUTPUT_DIR__",
            "policy_version": "v0_rollout_then_v1_checkpoint",
            "purpose": "one_rollout_batch_one_optimizer_update",
        },
        "g3_weight_sync_probe": {
            "max_steps": 2,
            "output_dir": "__BIND_DAY25_G3_OUTPUT_DIR__",
            "policy_version": "fresh_v0_then_v1_rollout_before_second_update",
            "purpose": "prove_colocate_lora_sync_on_the_second_rollout_batch",
            "claim_boundary": "second optimizer update is incidental and not used for the one-update claim",
        },
        "g4_bounded_short_run": {
            "max_steps": 10,
            "output_dir": "__BIND_DAY25_G4_OUTPUT_DIR__",
            "policy_version": "fresh_bounded_short_run",
            "purpose": "optional_only_after_g0_g1_g2_g3_pass",
        },
    }
    value: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_cpu_run_contract",
        "schema_version": 1,
        "status": "cpu_ready_gpu_pending",
        "parent": data_manifest["parent"],
        "inputs": {
            "data_manifest": _entry(data_manifest_path, "manifest_sha256"),
            "processor_audit": _entry(processor_summary_path, "summary_sha256"),
            "day24_schema": {
                "path": contract.relative_to_bootcamp(DAY24_SCHEMA),
                "file_sha256": contract.file_sha256(DAY24_SCHEMA),
            },
        },
        "implementation_sources": {
            label: contract.file_sha256(path)
            for label, path in sorted(source_paths.items())
        },
        "objective": {
            "algorithm": "grpo",
            "reward_policy": contract.REWARD_POLICY,
            "group_size": contract.GROUP_SIZE,
            "beta": 0.0,
            "reference_model_loaded": False,
            "advantage_estimator": "grpo",
            "scale_rewards": "group",
            "std_ddof": 1,
            "normalization_epsilon": 1e-4,
            "num_iterations": 1,
        },
        "batch_geometry": {
            "world_size": 1,
            "per_device_train_batch_size": 1,
            "generation_batch_size": 8,
            "steps_per_generation_resolved": 8,
            "gradient_accumulation_steps": 8,
            "groups_per_rollout_batch": 2,
            "optimizer_updates_per_generation_batch": 1,
        },
        "length_contract": {
            "audited_prompt_token_cap": 512,
            "audited_max_prompt_tokens": processor["max_prompt_token_count"],
            "trainer_max_length": 1024,
            "max_completion_length": 512,
            "vllm_max_model_len": 1152,
            "vllm_headroom_tokens": 128,
        },
        "topology": {
            "primary": "1xh100_80gb_colocate",
            "tensor_parallel": 1,
            "vllm_gpu_memory_utilization": 0.30,
            "sleep_level": 1,
            "vllm_lora_sync": True,
            "fallback": "new_config_key_required_for_2xh100_server; never mutate this contract in place",
        },
        "intended_ms_swift_args": common_args,
        "stages": stages,
        "environment_contract": {
            "required": {
                "DAY25_RUN_ID": "unique stage run ID",
                "DAY25_POLICY_VERSION": "v0 for G1/G2; stage-specific value for G3/G4",
                "DAY25_TRAJECTORY_LEDGER": "absolute append-only JSONL path",
                "DAY25_SANDBOX_WORKERS": "4",
                "DAY25_ABORT_AFTER_REWARD_BATCH": "1 for G1 only; 0 for G2/G3/G4",
                "E2B_API_KEY": "resolved by frozen Day 22 credential helper",
                "USE_MCORE_GDN": "0",
            },
            "reward_adapter_requires_full_batch": True,
        },
        "runtime_gates": {
            "g0_remote_payload_and_runtime": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "verify every promoted S1 merged-export file size and SHA-256",
                    "verify pinned ms-swift commit and GPU package versions",
                    "prove Qwen3.5 model and vLLM load before creating a paid long run",
                ],
            },
            "g1_rollout_only": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "G=4 prompt grouping and response token evidence",
                    "fresh E2B isolation with retry-once infra semantics",
                    "reward ledger offline audit pass before optimizer authorization",
                ],
            },
            "g2_one_update": {
                "status": "pending_gpu_runtime",
                "minimum_free_memory_fraction": 0.15,
                "requirements": [
                    "one finite optimizer step from fresh LoRA over promoted S1",
                    "old/current logprob, ratio, advantage, entropy, grad norm and peak VRAM finite",
                    "checkpoint-1 adapter and optimizer state durable",
                ],
            },
            "g3_weight_sync": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "second rollout batch occurs after checkpoint-1 update",
                    "vLLM LoRA sync/version evidence identifies v1 weights",
                    "no Day 23 checkpoint or Base fallback is loaded",
                ],
            },
            "g4_short_run": {
                "status": "blocked_until_g0_g1_g2_g3_pass",
                "requirements": [
                    "fresh run, at most 10 updates",
                    "frozen eval40 scored separately from training reward",
                ],
            },
        },
        "cpu_gates": {
            "day21_promoted_s1_lineage": "pass",
            "day24_reward_and_sandbox_contract": "pass",
            "train32_eval40_family_disjoint": "pass",
            "real_qwen35_processor_72_rows": "pass",
            "prompt_cap_and_zero_truncation": "pass",
            "four_space_response_token_supervised": "pass",
            "non_thinking_prefix_masked": "pass",
            "reward_adapter_unit_contract": "pass_via_required_test_script",
            "pinned_advantage_parity": "pass_via_argument_audit",
            "pinned_json_cli_parse": "pass_via_argument_audit",
        },
        "binding_rule": {
            "cpu_contract_is_not_direct_swift_config": True,
            "replace_all_double_underscore_placeholders": True,
            "dataset_plugin_output_and_ledger_paths_must_be_absolute": True,
            "omit_null_and_empty_list_values": True,
            "omit_cli_defaults": {
                "adapters": [],
                "ref_adapters": [],
                "ref_model": None,
                "resume_from_checkpoint": None,
            },
            "stage_metadata_keys_to_remove": [
                "policy_version",
                "purpose",
                "claim_boundary",
            ],
        },
        "claim_boundary": {
            "cpu_contract_ready": True,
            "gpu_model_loaded": False,
            "vllm_qwen35_compatible_on_target_gpu": False,
            "e2b_live_connectivity_on_target_runtime": False,
            "memory_preflight_passed": False,
            "optimizer_step_run": False,
            "weight_sync_proven": False,
            "capability_gain_claimed": False,
        },
    }
    _require(
        "day23" not in json.dumps(value["parent"], sort_keys=True).lower(),
        "Day 23 leaked into policy parent lineage",
    )
    value["contract_sha256"] = contract.object_sha256(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--processor-summary", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("build", "rebuild", "check"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = build_contract(
            data_manifest_path=args.data_manifest,
            processor_summary_path=args.processor_summary,
        )
        payload = contract.json_bytes(value)
        if args.mode == "build":
            contract.write_atomic(args.output, payload, overwrite=False)
        elif args.mode == "rebuild":
            contract.write_atomic(args.output, payload, overwrite=True)
        else:
            _require(args.output.is_file() and args.output.read_bytes() == payload, "CPU run contract drifted")
    except (OSError, contract.Day25ContractError, Day25CPUContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": value["status"],
                "mode": args.mode,
                "contract_sha256": value["contract_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
