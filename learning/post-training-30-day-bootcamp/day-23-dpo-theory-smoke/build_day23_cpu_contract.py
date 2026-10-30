#!/usr/bin/env python3
"""Build/check the self-hashed Day 23 CPU-ready, GPU-pending run contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract as contract
import day23_rlhf_template as dpo_template


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
DAY20_DIR = BOOTCAMP_ROOT / "day-20-qwen35-balanced-lora-sft"
if str(DAY20_DIR) not in sys.path:
    sys.path.insert(0, str(DAY20_DIR))

import day20_contract  # noqa: E402


DEFAULT_DATA_MANIFEST = (
    BOOTCAMP_ROOT / "artifacts/data/day23-qwen35-coding-dpo-data-manifest.json"
)
DEFAULT_PROCESSOR_SUMMARY = (
    BOOTCAMP_ROOT
    / "artifacts/eval/day23-qwen35-coding-dpo-processor-audit-summary.json"
)
DEFAULT_OUTPUT = (
    BOOTCAMP_ROOT
    / "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json"
)


class Day23CPUContractError(ValueError):
    """A frozen input, intended argument, or run-stage invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23CPUContractError(message)


def _entry(path: Path, self_field: str) -> dict[str, str]:
    value = contract.load_json(path)
    digest = contract.verify_self_hash(value, self_field, path.name)
    return {
        "path": path.resolve().relative_to(BOOTCAMP_ROOT.resolve()).as_posix(),
        "file_sha256": contract.file_sha256(path),
        "content_sha256": digest,
    }


def build_contract(
    *,
    data_manifest_path: Path = DEFAULT_DATA_MANIFEST,
    processor_summary_path: Path = DEFAULT_PROCESSOR_SUMMARY,
) -> dict[str, Any]:
    data_manifest_path = data_manifest_path.resolve()
    processor_summary_path = processor_summary_path.resolve()
    data_manifest = contract.load_json(data_manifest_path)
    processor = contract.load_json(processor_summary_path)
    contract.verify_self_hash(data_manifest, "manifest_sha256", "data manifest")
    contract.verify_self_hash(processor, "summary_sha256", "processor audit")
    _require(data_manifest.get("status") == "cpu_data_ready", "data gate is not ready")
    _require(processor.get("status") == "pass", "processor gate did not pass")
    _require(
        processor.get("counts", {}).get("branches") == 400
        and processor.get("counts", {}).get("day22_frozen_exact_branches") == 400,
        "processor branch gate is incomplete",
    )
    _require(
        processor.get("dataset_loader", {}).get("silent_row_deletions") == 0,
        "ms-swift loader deleted rows",
    )
    _require(
        data_manifest.get("parent", {}).get("ms_swift_commit")
        == contract.MS_SWIFT_COMMIT
        == processor.get("vendor_ms_swift", {}).get("commit"),
        "ms-swift identity differs across inputs",
    )
    train_path = data_manifest["outputs"]["train"]["path"]
    dev_path = data_manifest["outputs"]["dev"]["path"]
    heldout_path = data_manifest["outputs"]["heldout"]["path"]
    train_rows = contract.load_jsonl(BOOTCAMP_ROOT / train_path)
    _require(len(train_rows) == 154, "compiled train row count drifted")
    mechanism_pair_ids = [str(row["pair_id"]) for row in train_rows[:4]]
    source_paths = {
        "audit_day23_qwen35_arguments.py": DAY23_DIR
        / "audit_day23_qwen35_arguments.py",
        "audit_day23_qwen35_processor.py": DAY23_DIR
        / "audit_day23_qwen35_processor.py",
        "day23_contract.py": Path(contract.__file__).resolve(),
        "day23_dpo_math.py": DAY23_DIR / "day23_dpo_math.py",
        "day23_rlhf_template.py": Path(dpo_template.__file__).resolve(),
        "day23_ms_swift_plugin.py": DAY23_DIR / "day23_ms_swift_plugin.py",
        "build_day23_cpu_contract.py": Path(__file__).resolve(),
        "day20_target_encoding_v3.py": DAY20_DIR
        / "day20_target_encoding_v3.py",
        "prepare_day23_qwen35_dpo.py": DAY23_DIR
        / "prepare_day23_qwen35_dpo.py",
        "test_day23_qwen35_dpo_loss.py": (
            BOOTCAMP_ROOT / "artifacts/scripts/test_day23_qwen35_dpo_loss.py"
        ),
        "validate_day23_qwen35_dpo.py": (
            BOOTCAMP_ROOT / "artifacts/scripts/validate_day23_qwen35_dpo.py"
        ),
    }
    for label, path in source_paths.items():
        _require(path.is_file(), f"implementation source is missing: {label}")

    common_args: dict[str, Any] = {
        "rlhf_type": "dpo",
        "model": "__BIND_VERIFIED_REMOTE_S1_MERGED_EXPORT__",
        "model_type": "qwen3_5",
        "dataset": [train_path],
        "val_dataset": [dev_path],
        "split_dataset_ratio": 0,
        "strict": True,
        "disable_auto_column_mapping": True,
        "remove_unused_columns": True,
        "template": dpo_template.TARGET_TEMPLATE_ALIAS,
        "external_plugins": [
            "day-23-dpo-theory-smoke/day23_ms_swift_plugin.py"
        ],
        "loss_scale": dpo_template.LOSS_SCALE,
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "max_length": 512,
        "truncation_strategy": "delete",
        "packing": False,
        "padding_free": False,
        "tuner_type": "lora",
        "target_regex": day20_contract.LORA_TARGET_REGEX,
        "freeze_llm": False,
        "freeze_vit": True,
        "freeze_aligner": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "beta": 0.1,
        "loss_type": "sigmoid",
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "bf16": True,
        "seed": 20260818,
        "data_seed": 20260818,
        "logging_steps": 1,
        "report_to": ["tensorboard"],
        "add_version": False,
        "save_strategy": "steps",
        "save_only_model": False,
        "load_best_model_at_end": False,
    }
    stages = {
        "mechanism_5step": {
            "fresh_start_from_parent": True,
            "remove_common_args": ["val_dataset"],
            "dataset": [f"{train_path}#4"],
            "dataset_shuffle": False,
            "train_dataloader_shuffle": False,
            "mechanism_pair_ids": mechanism_pair_ids,
            "max_steps": 5,
            "save_steps": 5,
            "save_total_limit": 1,
            "eval_strategy": "no",
            "output_dir": "__BIND_DAY23_MECHANISM_OUTPUT_DIR__",
            "purpose": "mechanism_only_not_checkpoint_candidate",
        },
        "bounded_smoke_30step": {
            "fresh_start_from_parent": True,
            "dataset_shuffle": True,
            "train_dataloader_shuffle": True,
            "max_steps": 30,
            "save_steps": 15,
            "save_total_limit": 2,
            "eval_strategy": "steps",
            "eval_steps": 15,
            "output_dir": "__BIND_DAY23_SMOKE_OUTPUT_DIR__",
            "checkpoint_candidates": [15, 30],
            "purpose": "bounded_smoke_checkpoint_selection",
        },
    }
    value: dict[str, Any] = {
        "schema_name": "day23.qwen35_coding_dpo_cpu_run_contract",
        "schema_version": 1,
        "status": "cpu_ready_gpu_pending",
        "inputs": {
            "data_manifest": _entry(data_manifest_path, "manifest_sha256"),
            "processor_audit": _entry(processor_summary_path, "summary_sha256"),
        },
        "implementation_sources": {
            label: contract.file_sha256(path)
            for label, path in sorted(source_paths.items())
        },
        "parent": data_manifest["parent"],
        "policy_reference_mapping": data_manifest["policy_reference_mapping"],
        "objective": {
            "type": "sigmoid_dpo",
            "beta": 0.1,
            "sequence_logprob_reduction": "sum_over_response_only_causal_labels",
            "formula": "softplus(-beta*((logpi_c-logpi_r)-(logref_c-logref_r)))",
            "reference_implicit_reward": "beta*(logpi-logref)",
            "length_bias_monitoring_required": True,
        },
        "cli_default_semantics": {
            "omitted_keys": {
                "adapters": [],
                "ref_adapters": [],
                "ref_model": None,
                "resume_from_checkpoint": None,
            },
            "reason": "pinned JSON CLI expands null/empty-list values incorrectly; omission selects these dataclass defaults",
        },
        "intended_ms_swift_args": common_args,
        "stages": stages,
        "dataset_usage": {
            "train": {"path": train_path, "role": "optimizer_input"},
            "dev": {"path": dev_path, "role": "checkpoint_selection_only"},
            "heldout": {
                "path": heldout_path,
                "role": "one_shot_confirmation_after_selection",
                "present_in_intended_ms_swift_args": False,
            },
        },
        "selection": {
            "eligible_checkpoints": [15, 30],
            "primary": "dev_pair_accuracy",
            "tie_breakers": [
                "dev_mean_reward_margin",
                "dev_length_matched_margin",
                "earlier_checkpoint",
            ],
            "preference_slices": [
                "response_length",
                "source_family",
                "test_family",
                "quality_status",
            ],
            "heldout_consumption": "exactly_once_after_checkpoint_selection",
            "trainer_native_eval_is_not_the_selector": True,
            "external_pair_level_evaluator_required": True,
        },
        "runtime_gates": {
            "g0_remote_payload": {
                "status": contract.REMOTE_PAYLOAD_GATE,
                "requirements": [
                    "resolve merged S1 path from promotion manifest",
                    "verify every export shard byte count and SHA-256",
                    "verify remote promotion/key/export manifests and source checkout commit",
                ],
            },
            "g1_policy_reference_and_freeze": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "fresh policy LoRA initialized over merged S1",
                    "no ref_model and no ref_adapter supplied",
                    "reference forward disables the fresh policy adapter",
                    "reference and non-LoRA policy parent hashes remain immutable",
                    "trainable inventory matches target_regex only",
                    "vision tower, aligner, embeddings, and lm_head remain frozen",
                ],
            },
            "g2_one_step_memory_save_reload": {
                "status": "pending_gpu_runtime",
                "minimum_free_memory_fraction": 0.15,
                "requirements": [
                    "measure policy/reference/optimizer/activation/logit peak memory",
                    "one real optimizer update is finite",
                    "save and fresh-process reload reproduce adapter outputs",
                ],
            },
            "g3_mechanism_5step": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "only the four frozen mechanism pair IDs are loaded; dev is absent",
                    "dataset and train dataloader shuffling remain disabled",
                    "all logged losses/rewards/grad norms finite",
                    "mean chosen reward margin increases from step 0",
                    "at least 3 of 4 mechanism pairs improve margin",
                    "mechanism output is never promoted or resumed into smoke",
                ],
            },
            "g4_bounded_smoke": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "fresh 30-step run starts independently from merged S1",
                    "checkpoints 15 and 30 save and reload",
                    "external pair-level dev evaluator selects once; heldout remains concealed until selection",
                ],
            },
            "g5_post_selection_guardrails": {
                "status": "pending_gpu_runtime",
                "requirements": [
                    "one-shot preference heldout",
                    "sandbox coding correctness",
                    "general/math/format guardrails",
                    "report separates mechanism, preference metrics, and coding correctness",
                ],
            },
        },
        "cpu_gates": {
            "day21_compact_ancestry": "pass",
            "day22_independent_experimental_close": "pass",
            "deterministic_split_compilation": "pass",
            "pinned_ms_swift_loader_no_row_deletion": "pass",
            "real_qwen35_rlhf_template_400_branches": "pass",
            "day22_token_label_exact_match_400_branches": "pass",
            "response_mask_and_four_space_boundary": "pass",
            "max_length_512_zero_truncation": "pass",
            "scalar_dpo_oracle": "pass_via_required_pinned_test_script",
        },
        "binding_rule": {
            "cpu_contract_is_not_direct_swift_config": True,
            "gpu_binding_must_replace_all_double_underscore_placeholders": True,
            "stage_metadata_keys_to_remove": [
                "checkpoint_candidates",
                "fresh_start_from_parent",
                "mechanism_pair_ids",
                "purpose",
                "remove_common_args",
            ],
            "relative_path_base": "bootcamp_root",
            "dataset_val_dataset_and_plugin_paths_must_be_absolute": True,
            "executable_config_must_omit_null_and_empty_list_values": True,
            "executable_config_must_omit_cli_default_keys": [
                "adapters",
                "ref_adapters",
                "ref_model",
                "resume_from_checkpoint",
            ],
            "executable_config_must_contain_only_pinned_ms_swift_argument_keys": True,
            "binding_and_runtime_attestation_must_be_self_hashed": True,
        },
        "claim_boundary": {
            "cpu_contract_ready": True,
            "formal_human_reviewed": False,
            "remote_s1_payload_verified_now": False,
            "model_weights_loaded_on_cpu": False,
            "optimizer_step_run": False,
            "reference_immutability_runtime_proven": False,
            "gpu_memory_preflight_passed": False,
            "gpu_optimizer_ready": False,
        },
    }
    _require(heldout_path not in json.dumps(common_args, sort_keys=True), "heldout leaked into trainer args")
    _require(stages["mechanism_5step"]["fresh_start_from_parent"] is True, "mechanism is not fresh")
    _require(stages["bounded_smoke_30step"]["fresh_start_from_parent"] is True, "smoke is not fresh")
    value["contract_sha256"] = contract.object_sha256(value)
    return value


def output_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"


def apply_output(path: Path, payload: bytes, *, mode: str) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"output parent is missing: {path.parent}")
    if mode == "build":
        try:
            with path.open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise Day23CPUContractError(f"refusing to overwrite: {path}") from error
    elif mode == "check":
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise Day23CPUContractError(f"cannot read frozen contract: {path}") from error
        _require(actual == payload, f"frozen CPU run contract drifted: {path}")
    else:
        raise Day23CPUContractError(f"unsupported mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "check"), required=True)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--processor-summary", type=Path, default=DEFAULT_PROCESSOR_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = build_contract(
            data_manifest_path=args.data_manifest,
            processor_summary_path=args.processor_summary,
        )
        apply_output(args.output, output_bytes(value), mode=args.mode)
    except (OSError, contract.Day23ContractError, Day23CPUContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "valid_cpu_ready_gpu_pending",
                "mode": args.mode,
                "contract_sha256": value["contract_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
