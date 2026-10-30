#!/usr/bin/env python3
"""Build/check the Day 23 RSI-v0003 CPU campaign and bind it for GPU use.

The CPU campaign is a preregistration artifact only.  It deterministically
splits the frozen Day 23 train data into fit/search subsets, emits pure
``RLHFArguments`` JSON configs, and freezes the search/refit/dev protocol.
``--mode bind`` replaces every runtime placeholder and seals a separate GPU
campaign; it still does not load model weights or run training/evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
if str(DAY23_DIR) not in sys.path:
    sys.path.insert(0, str(DAY23_DIR))

import day23_contract as contract  # noqa: E402


CAMPAIGN_ID = "day23-qwen35-dpo-rsi-v0003"
GOAL_ID = "goal-0002-day23-dpo"
VERSION_ID = "rsi-v0003"
SCIENTIFIC_LEVER = "model.optimization.learning_rate.base"
LEARNING_RATES = (1e-6, 5e-6)
CHECKPOINT_STEPS = (15, 30)
SEARCH_SEED = 20260819
REFIT_SEED = 20260820
WORLD_SIZE = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
NOMINAL_GLOBAL_BATCH_SIZE = 32
OLD_MECHANISM_PAIR_IDS = (
    "mbpp:task:602:s1pair:37673b32b9385aea",
    "mbpp:task:604:s1pair:388a8daec7e97c5a",
    "mbpp:task:605:s1pair:9ae278fe596ed4d4",
    "mbpp:task:610:s1pair:9e57fb63f7544777",
)
FORBIDDEN_EXECUTABLE_KEYS = {
    "adapters",
    "ref_adapters",
    "ref_model",
    "resume_from_checkpoint",
    "val_dataset",
}

DATA_MANIFEST_REL = "artifacts/data/day23-qwen35-coding-dpo-data-manifest.json"
TRAIN_REL = "artifacts/data/day23-qwen35-coding-dpo-train.jsonl"
DEV_REL = "artifacts/data/day23-qwen35-coding-dpo-dev.jsonl"
PROCESSOR_SUMMARY_REL = (
    "artifacts/eval/day23-qwen35-coding-dpo-processor-audit-summary.json"
)
CPU_CONTRACT_REL = "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json"
FIT_REL = "artifacts/data/day23-qwen35-dpo-rsi-v0003-fit.jsonl"
SEARCH_REL = "artifacts/data/day23-qwen35-dpo-rsi-v0003-search.jsonl"
CONFIG_DIR_REL = "artifacts/configs/day23-qwen35-dpo-rsi-v0003"
CPU_CAMPAIGN_REL = f"{CONFIG_DIR_REL}/cpu-campaign.json"
CHARTER_REL = f"rsi-control/charters/{GOAL_ID}/charter.json"
PREREGISTRATION_REL = f"{CONFIG_DIR_REL}/campaign-preregistration.json"

CONFIG_FILENAMES = {
    "search_lr_1e_6": "search-lr1e-6.json",
    "search_lr_5e_6": "search-lr5e-6.json",
    "refit_lr_1e_6_step_15": "refit-lr1e-6-step15.json",
    "refit_lr_1e_6_step_30": "refit-lr1e-6-step30.json",
    "refit_lr_5e_6_step_15": "refit-lr5e-6-step15.json",
    "refit_lr_5e_6_step_30": "refit-lr5e-6-step30.json",
}


class Day23RSICampaignError(ValueError):
    """A frozen RSI campaign invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23RSICampaignError(message)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"missing regular file: {resolved}")
    return {
        "path": relative_path if relative_path is not None else str(resolved),
        "file_sha256": contract.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _load_self_hashed(path: Path, field: str, label: str) -> dict[str, Any]:
    value = contract.load_json(path)
    contract.verify_self_hash(value, field, label)
    return value


def _dataset_identity(
    *, path: str, payload: bytes, rows: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Any]:
    pair_ids = [str(row["pair_id"]) for row in rows]
    row_hashes = [str(row["row_sha256"]) for row in rows]
    return {
        "path": path,
        "file_sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "records": len(rows),
        "ordered_pair_ids_sha256": contract.object_sha256(pair_ids),
        "ordered_row_hashes_sha256": contract.object_sha256(row_hashes),
        "role": role,
    }


def split_fit_search(
    train_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return the preregistered 124-row fit and 30-row search partitions."""
    _require(len(train_rows) == 154, "frozen train row count drifted")
    pair_ids = [str(row.get("pair_id")) for row in train_rows]
    _require(len(pair_ids) == len(set(pair_ids)), "train pair IDs are not unique")
    _require(tuple(pair_ids[:4]) == OLD_MECHANISM_PAIR_IDS, "historical four-pair prefix drifted")

    old = set(OLD_MECHANISM_PAIR_IDS)
    eligible: list[str] = []
    for row in train_rows:
        pair_id = str(row["pair_id"])
        contract.verify_self_hash(row, "row_sha256", f"train row {pair_id}")
        if pair_id in old:
            continue
        eligible.append(pair_id)

    _require(len(eligible) == 150, "uncontaminated train population drifted")
    salt = f"{CAMPAIGN_ID}|"
    ordered = sorted(
        eligible,
        key=lambda pair_id: (
            hashlib.sha256((salt + pair_id).encode("utf-8")).hexdigest(),
            pair_id,
        ),
    )
    selected = set(ordered[:30])

    fit_rows = [dict(row) for row in train_rows if str(row["pair_id"]) not in selected]
    search_rows = [dict(row) for row in train_rows if str(row["pair_id"]) in selected]
    _require(len(fit_rows) == 124 and len(search_rows) == 30, "fit/search counts drifted")
    _require(old.isdisjoint(selected), "historical four pairs leaked into search")
    _require(old.issubset({str(row["pair_id"]) for row in fit_rows}), "historical four pairs missing from fit")
    split_protocol = {
        "authority": "frozen_train_pair_ids_only_no_processor_audit_rows",
        "selection_salt": salt,
        "order": "sha256(campaign_id + '|' + pair_id), then pair_id",
        "uncontaminated_population": 150,
        "search_prefix_records": 30,
        "fit_complement_records": 120,
        "historical_four_pair_disposition": {
            "pair_ids": list(OLD_MECHANISM_PAIR_IDS),
            "optimizer_input_in_fit": True,
            "eligible_for_search_selection_or_confirmation": False,
        },
    }
    return fit_rows, search_rows, split_protocol


def _config_identity(relative_path: str, payload: bytes, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": relative_path,
        "file_sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "keys": sorted(value),
    }


def _base_config(cpu_contract: Mapping[str, Any]) -> dict[str, Any]:
    common = dict(cpu_contract["intended_ms_swift_args"])
    for key in FORBIDDEN_EXECUTABLE_KEYS:
        common.pop(key, None)
    common.update(
        {
            "model": "__BIND_VERIFIED_REMOTE_S1_MERGED_EXPORT__",
            "external_plugins": ["day-23-dpo-theory-smoke/day23_ms_swift_plugin.py"],
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "dataset_shuffle": True,
            "train_dataloader_shuffle": True,
            "eval_strategy": "no",
            "logging_steps": 1,
            "save_strategy": "steps",
            "add_version": False,
            "load_best_model_at_end": False,
        }
    )
    common.pop("eval_steps", None)
    _require(not (set(common) & FORBIDDEN_EXECUTABLE_KEYS), "forbidden CLI default key leaked")
    return common


def _run_id(lr: float, *, step: int | None = None) -> str:
    lr_name = "1e_6" if lr == 1e-6 else "5e_6"
    return f"search_lr_{lr_name}" if step is None else f"refit_lr_{lr_name}_step_{step}"


def _candidate_id(lr: float, step: int) -> str:
    lr_name = "1e_6" if lr == 1e-6 else "5e_6"
    return f"candidate_lr_{lr_name}_step_{step}"


def _build_configs(cpu_contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    base = _base_config(cpu_contract)
    configs: dict[str, dict[str, Any]] = {}
    for lr in LEARNING_RATES:
        run_id = _run_id(lr)
        value = dict(base)
        value.update(
            {
                "dataset": [FIT_REL],
                "learning_rate": lr,
                "seed": SEARCH_SEED,
                "data_seed": SEARCH_SEED,
                "max_steps": 30,
                "save_steps": 15,
                "save_total_limit": 2,
                "output_dir": f"__BIND_{run_id.upper()}_OUTPUT_DIR__",
            }
        )
        configs[run_id] = value
        for step in CHECKPOINT_STEPS:
            refit_id = _run_id(lr, step=step)
            refit = dict(base)
            refit.update(
                {
                    "dataset": [TRAIN_REL],
                    "learning_rate": lr,
                    "seed": REFIT_SEED,
                    "data_seed": REFIT_SEED,
                    "max_steps": step,
                    "save_steps": step,
                    "save_total_limit": 1,
                    "output_dir": f"__BIND_{refit_id.upper()}_OUTPUT_DIR__",
                }
            )
            configs[refit_id] = refit
    _require(set(configs) == set(CONFIG_FILENAMES), "config registry drifted")
    return configs


def _identity_from_manifest(output: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "path": str(output["path"]),
        "file_sha256": str(output["file_sha256"]),
        "records": int(output["records"]),
        "ordered_pair_ids_sha256": str(output["ordered_pair_ids_sha256"]),
        "ordered_row_hashes_sha256": str(output["ordered_row_hashes_sha256"]),
        "role": role,
    }


def _preregistration_value(data_partition: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_name": "day23.rsi_v0003_campaign_preregistration",
        "schema_version": 1,
        "status": "cpu_preregistered_guardrails_block_heldout",
        "goal_id": GOAL_ID,
        "campaign_id": CAMPAIGN_ID,
        "version_id": VERSION_ID,
        "historical_evidence_boundary": {
            "day23_original_campaign_disposition": "closed_no_candidate",
            "mechanism_attempts_observed": 2,
            "both_attempts_positive_pair_count": "1_of_4",
            "old_four_pairs_contaminated_for_future_selection": True,
            "old_four_pairs_allowed_only_as_optimizer_input_or_descriptive_diagnostic": True,
            "old_three_of_four_direction_gate_retired_for_rsi_candidate_selection": True,
        },
        "diagnosis": {
            "causal_status": "inconclusive",
            "primary_failure_mode": "failure.unknown.localization.unlocalized",
            "candidate_contributors": [
                "very_short_optimization_budget",
                "cosine_schedule_decay",
                "four_pair_mechanism_sample_variance",
            ],
        },
        "hypothesis": {
            "single_scientific_lever": SCIENTIFIC_LEVER,
            "levels": list(LEARNING_RATES),
            "falsifier": "no_locked_search_candidate_is_eligible_or_unique_refit_fails_dev",
            "forbidden_same_version_axes": [
                "beta",
                "loss_type",
                "lora_rank_or_alpha_or_dropout",
                "batch_or_accumulation",
                "seed_grid",
                "extra_checkpoint_steps",
            ],
        },
        "data_preregistration": dict(data_partition),
        "fixed_topology": {
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "nominal_global_train_batch_size": NOMINAL_GLOBAL_BATCH_SIZE,
            "b24_optimizer_use": "forbidden_in_v0003",
        },
        "exposure_budget": {
            "search_training_trajectories": 2,
            "refit_training_trajectories": 1,
            "maximum_optimizer_steps": 90,
            "seed_roles": 2,
            "search_checkpoint_evaluations": 4,
            "dev_finalist_evaluations": 1,
            "heldout_evaluations": 0,
        },
        "claim_boundary": {
            "cpu_preregistered": True,
            "training_run": False,
            "dev_consumed": False,
            "heldout_consumed": False,
            "guardrail_suite_implemented_and_frozen": False,
            "heldout_authorized": False,
            "maximum_future_claim": "small_experimental_preference_improvement_on_frozen_pairs",
        },
    }
    value["preregistration_sha256"] = contract.object_sha256(value)
    return value


def _run_spec(
    *,
    run_id: str,
    role: str,
    dataset_key: str,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    config_identity: Mapping[str, Any],
    learning_rate: float,
    seed: int,
    max_steps: int,
    checkpoint_steps: Sequence[int],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "authorized": role == "search_train",
        "dataset_key": dataset_key,
        "dataset": dict(dataset),
        "executable_config": dict(config_identity),
        "model_argument": str(config["model"]),
        "runtime": {
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "configured_nominal_global_train_batch_size": NOMINAL_GLOBAL_BATCH_SIZE,
        },
        "learning_rate": learning_rate,
        "seed": seed,
        "max_steps": max_steps,
        "checkpoint_steps": list(checkpoint_steps),
        "output_dir": str(config["output_dir"]),
        "success_receipt": f"__BIND_{run_id.upper()}_SUCCESS_RECEIPT__",
        "failure_receipt": f"__BIND_{run_id.upper()}_FAILURE_RECEIPT__",
        "fresh_start_from_parent": True,
        "authorization": dict(authorization),
    }


def build_campaign_bundle(source_root: Path = BOOTCAMP_ROOT) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build all deterministic CPU payload bytes without writing them."""
    root = source_root.resolve()
    data_manifest = _load_self_hashed(root / DATA_MANIFEST_REL, "manifest_sha256", "data manifest")
    processor_summary = _load_self_hashed(
        root / PROCESSOR_SUMMARY_REL, "summary_sha256", "processor summary"
    )
    cpu_contract = _load_self_hashed(root / CPU_CONTRACT_REL, "contract_sha256", "CPU contract")
    _require(data_manifest.get("status") == "cpu_data_ready", "Day 23 data is not CPU-ready")
    _require(processor_summary.get("status") == "pass", "processor audit did not pass")
    _require(cpu_contract.get("status") == "cpu_ready_gpu_pending", "CPU contract status drifted")

    train_path = root / TRAIN_REL
    _require(contract.file_sha256(train_path) == data_manifest["outputs"]["train"]["file_sha256"], "train file drifted")
    train_rows = contract.load_jsonl(train_path)
    fit_rows, search_rows, data_partition = split_fit_search(train_rows)
    fit_payload = contract.jsonl_bytes(fit_rows)
    search_payload = contract.jsonl_bytes(search_rows)
    train_payload = train_path.read_bytes()

    datasets = {
        "fit": _dataset_identity(path=FIT_REL, payload=fit_payload, rows=fit_rows, role="search_optimizer_input"),
        "search": _dataset_identity(path=SEARCH_REL, payload=search_payload, rows=search_rows, role="locked_hyperparameter_ranking_only"),
        "full_train": _dataset_identity(path=TRAIN_REL, payload=train_payload, rows=train_rows, role="unique_finalist_refit_optimizer_input"),
        "dev": _identity_from_manifest(
            data_manifest["outputs"]["dev"],
            "one_shot_finalist_qualification_not_hyperparameter_selection",
        ),
        "heldout": {
            "runtime_path_disclosed": False,
            "file_sha256": data_manifest["outputs"]["heldout"]["file_sha256"],
            "records": data_manifest["outputs"]["heldout"]["records"],
            "ordered_pair_ids_sha256": data_manifest["outputs"]["heldout"]["ordered_pair_ids_sha256"],
            "ordered_row_hashes_sha256": data_manifest["outputs"]["heldout"]["ordered_row_hashes_sha256"],
            "role": "sealed_confirmation_unavailable_in_v0003",
            "maximum_evaluations": 0,
        },
    }
    _require(datasets["fit"]["records"] == 124 and datasets["search"]["records"] == 30, "derived data counts drifted")

    configs = _build_configs(cpu_contract)
    payloads: dict[str, bytes] = {FIT_REL: fit_payload, SEARCH_REL: search_payload}
    config_identities: dict[str, dict[str, Any]] = {}
    for run_id, value in sorted(configs.items()):
        config_rel = f"{CONFIG_DIR_REL}/{CONFIG_FILENAMES[run_id]}"
        payload = _json_bytes(value)
        payloads[config_rel] = payload
        config_identities[run_id] = _config_identity(config_rel, payload, value)

    preregistration = _preregistration_value(data_partition)
    preregistration_payload = _json_bytes(preregistration)
    payloads[PREREGISTRATION_REL] = preregistration_payload
    charter_path = root / CHARTER_REL
    charter = _load_self_hashed(charter_path, "charter_sha256", "authoritative RSI charter")
    _require(
        charter.get("schema_name") == "rsi.day23_dpo_charter"
        and charter.get("goal_id") == GOAL_ID,
        "authoritative RSI charter identity drifted",
    )
    charter_campaign = charter.get("dataset_campaign")
    _require(isinstance(charter_campaign, Mapping), "authoritative charter dataset campaign missing")
    _require(
        charter_campaign.get("optimizer124", {}).get("ordered_pair_ids_sha256")
        == datasets["fit"]["ordered_pair_ids_sha256"]
        and charter_campaign.get("search30", {}).get("ordered_pair_ids_sha256")
        == datasets["search"]["ordered_pair_ids_sha256"]
        and charter_campaign.get("full154_refit", {}).get("ordered_pair_ids_sha256")
        == datasets["full_train"]["ordered_pair_ids_sha256"],
        "campaign partition differs from authoritative charter",
    )
    run_specs: dict[str, Any] = {}
    for lr in LEARNING_RATES:
        search_run = _run_id(lr)
        run_specs[search_run] = _run_spec(
            run_id=search_run,
            role="search_train",
            dataset_key="fit",
            dataset=datasets["fit"],
            config=configs[search_run],
            config_identity=config_identities[search_run],
            learning_rate=lr,
            seed=SEARCH_SEED,
            max_steps=30,
            checkpoint_steps=CHECKPOINT_STEPS,
            authorization={"state": "authorized_after_gpu_binding", "requires_selection_receipt": False},
        )
        for step in CHECKPOINT_STEPS:
            refit_run = _run_id(lr, step=step)
            run_specs[refit_run] = _run_spec(
                run_id=refit_run,
                role="unique_finalist_refit_option",
                dataset_key="full_train",
                dataset=datasets["full_train"],
                config=configs[refit_run],
                config_identity=config_identities[refit_run],
                learning_rate=lr,
                seed=REFIT_SEED,
                max_steps=step,
                checkpoint_steps=[step],
                authorization={
                    "state": "conditional_exactly_one",
                    "requires_selected_candidate_id": _candidate_id(lr, step),
                    "search_checkpoint_itself_is_forbidden_as_refit_initialization": True,
                },
            )

    candidates = [
        {
            "candidate_id": _candidate_id(lr, step),
            "search_run_spec": _run_id(lr),
            "checkpoint_step": step,
            "learning_rate": lr,
            "refit_run_spec": _run_id(lr, step=step),
        }
        for lr in LEARNING_RATES
        for step in CHECKPOINT_STEPS
    ]
    fixed_recipe = {
        "single_scientific_lever": SCIENTIFIC_LEVER,
        "lever_levels": list(LEARNING_RATES),
        "search_seed": SEARCH_SEED,
        "refit_seed": REFIT_SEED,
        "model": {
            "parent_role": "S1",
            "checkpoint_id": data_manifest["parent"]["checkpoint_id"],
            "model_argument": "__BIND_VERIFIED_REMOTE_S1_MERGED_EXPORT__",
            "policy": "fresh_lora_over_merged_s1",
            "reference": "same_model_with_fresh_policy_adapter_disabled",
            "forbidden_inputs": sorted(FORBIDDEN_EXECUTABLE_KEYS - {"val_dataset"}),
        },
        "runtime": {
            "python": "__BIND_PYTHON_EXECUTABLE__",
            "ms_swift_checkout": "__BIND_PINNED_MS_SWIFT_CHECKOUT__",
            "ms_swift_commit": contract.MS_SWIFT_COMMIT,
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "nominal_global_train_batch_size": NOMINAL_GLOBAL_BATCH_SIZE,
            "min_free_memory_fraction": 0.30,
        },
        "topology": {
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "nominal_global_train_batch_size": NOMINAL_GLOBAL_BATCH_SIZE,
            "b24_optimizer_authorized": False,
        },
        "objective": {
            "rlhf_type": "dpo",
            "loss_type": "sigmoid",
            "beta": 0.1,
            "response_logprob_reduction": "sum",
        },
    }
    artifact_inventory = {
        relative: {"file_sha256": _sha256_bytes(payload), "bytes": len(payload)}
        for relative, payload in sorted(payloads.items())
    }
    implementation_paths = {
        "builder_and_binder": root / "day-23-dpo-theory-smoke/build_day23_rsi_v0003.py",
        "training_runner": root / "day-23-dpo-theory-smoke/run_day23_rsi_candidate.py",
        "candidate_evaluator": root / "day-23-dpo-theory-smoke/eval_day23_rsi_candidate.py",
        "charter_validator": root
        / "rsi-control/charters/goal-0002-day23-dpo/validate_day23_dpo_charter.py",
    }
    implementation_sources = {
        name: _file_identity(
            path,
            relative_path=path.resolve().relative_to(root).as_posix(),
        )
        for name, path in implementation_paths.items()
    }
    access_leases = charter.get("access_leases")
    _require(isinstance(access_leases, Mapping), "authoritative charter access leases missing")
    dev_lease = access_leases.get("dev")
    heldout_lease = access_leases.get("heldout")
    _require(isinstance(dev_lease, Mapping) and isinstance(heldout_lease, Mapping), "authoritative split leases missing")
    _require(
        dev_lease.get("maximum_global_claims") == 1
        and heldout_lease.get("current_authorized_claims") == 0,
        "authoritative lease counts drifted",
    )
    dev_ledger_rel = f"rsi-control/{dev_lease['claim_path']}"
    heldout_lease_without_path = {
        key: value for key, value in heldout_lease.items() if key != "claim_path"
    }
    campaign: dict[str, Any] = {
        "schema_name": "day23.rsi_v0003_cpu_campaign",
        "schema_version": 1,
        "status": "cpu_preregistered_gpu_pending_guardrails_block_heldout",
        "campaign_id": CAMPAIGN_ID,
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "bootcamp_root": "__BIND_BOOTCAMP_ROOT__",
        "remote_run_root": "__BIND_REMOTE_RUN_ROOT__",
        "charter": {
            "path": CHARTER_REL,
            "file_sha256": contract.file_sha256(charter_path),
            "content_sha256": charter["charter_sha256"],
        },
        "campaign_preregistration": {
            "path": PREREGISTRATION_REL,
            "file_sha256": _sha256_bytes(preregistration_payload),
            "content_sha256": preregistration["preregistration_sha256"],
        },
        "inputs": {
            "data_manifest": {**_file_identity(root / DATA_MANIFEST_REL, relative_path=DATA_MANIFEST_REL), "content_sha256": data_manifest["manifest_sha256"]},
            "processor_summary": {**_file_identity(root / PROCESSOR_SUMMARY_REL, relative_path=PROCESSOR_SUMMARY_REL), "content_sha256": processor_summary["summary_sha256"]},
            "cpu_contract": {**_file_identity(root / CPU_CONTRACT_REL, relative_path=CPU_CONTRACT_REL), "content_sha256": cpu_contract["contract_sha256"]},
        },
        "implementation_sources": implementation_sources,
        "parent": {**data_manifest["parent"], "model_path": "__BIND_VERIFIED_REMOTE_S1_MERGED_EXPORT__"},
        "fixed_recipe": fixed_recipe,
        "datasets": datasets,
        "data_partition": data_partition,
        "run_specs": dict(sorted(run_specs.items())),
        "candidate_registry": candidates,
        "protocol": {
            "state_order": ["bind", "two_search_runs", "seal_four_candidates", "rank_top1", "fresh_full_train_refit", "one_shot_dev", "guardrails", "heldout"],
            "search_selection": {
                "all_four_candidates_must_be_sealed_before_unblinding": True,
                "dataset_key": "search",
                "maximum_evaluations": 4,
                "eligibility": {
                    "finite_records": "30_of_30",
                    "minimum_positive_margin_count": 20,
                    "mean_margin_strictly_greater_than": 0.0,
                    "length_matched_mean_margin_strictly_greater_than": 0.0,
                    "runtime_reload_freeze_integrity": "pass",
                },
                "ranking": [
                    "positive_margin_count_desc",
                    "mean_margin_desc",
                    "length_matched_mean_margin_desc",
                    "checkpoint_step_asc",
                    "learning_rate_asc",
                ],
                "winner_count": 1,
                "no_eligible_candidate": "close_no_candidate",
            },
            "refit": {
                "maximum_executions": 1,
                "fresh_from_s1": True,
                "dataset_key": "full_train",
                "seed": REFIT_SEED,
                "must_match_selected_learning_rate_and_step": True,
                "search_checkpoint_as_initialization": "forbidden",
            },
            "dev": {
                "dataset_key": "dev",
                "maximum_evaluations": 1,
                "minimum_positive_margin_count": 13,
                "records": 17,
                "exact_one_sided_binomial_p_at_threshold": 0.0245209,
                "mean_margin_strictly_greater_than": 0.0,
                "length_matched_mean_margin_strictly_greater_than": 0.0,
                "failure": "close_no_candidate_no_runner_up_fallback",
            },
            "guardrails": {
                "required_suites": ["sandbox", "general", "math", "format"],
                "implementation_status": "missing",
                "blocks_heldout": True,
            },
            "heldout": {
                "lease_count": 0,
                "maximum_evaluations": 0,
                "runtime_path_disclosed": False,
                "authorization": "blocked_until_separate_append_only_guardrail_amendment",
            },
        },
        "leases": {"search": 4, "dev": 1, "heldout": 0},
        "access_ledger": {
            "authority": "charter.access_leases",
            "access_leases_sha256": contract.object_sha256(access_leases),
            "scope_id": access_leases.get("scope_id"),
            "ledger_root": access_leases.get("ledger_root"),
            "search_claim": "__BIND_REMOTE_RUN_ROOT__/evidence/search-selection/search-claim.json",
            "dev_claim": dev_ledger_rel,
            "dev_lease": dict(dev_lease),
            "heldout_lease": heldout_lease_without_path,
            "heldout_claim_disclosed": False,
        },
        "budgets": {
            "maximum_training_trajectories": 3,
            "maximum_search_training_runs": 2,
            "maximum_refit_training_runs": 1,
            "maximum_seed_roles": 2,
            "maximum_optimizer_steps": 90,
            "maximum_search_checkpoint_evaluations": 4,
            "maximum_dev_evaluations": 1,
            "maximum_heldout_evaluations": 0,
        },
        "retry_policy": {
            "runtime_failure_before_model_load_or_scientific_metric": "same_semantics_retry_allowed_with_append_only_failure_receipt",
            "any_run_that_emitted_scientific_metric": "no_retry",
            "dev_failure": "no_fallback",
            "heldout_retry": "forbidden",
        },
        "artifact_inventory": artifact_inventory,
        "claim_boundary": {
            "frozen_day23_bundle_modified": False,
            "cpu_campaign_preregistered": True,
            "model_weights_loaded": False,
            "optimizer_step_run": False,
            "search_or_dev_consumed": False,
            "heldout_consumed": False,
            "heldout_authorized": False,
        },
    }
    campaign["campaign_sha256"] = contract.object_sha256(campaign)
    validate_campaign(campaign, payloads=payloads)
    payloads[CPU_CAMPAIGN_REL] = _json_bytes(campaign)
    return payloads, campaign


def validate_campaign(value: Mapping[str, Any], *, payloads: Mapping[str, bytes] | None = None) -> None:
    contract.verify_self_hash(value, "campaign_sha256", "RSI campaign")
    _require(value.get("schema_name") in {"day23.rsi_v0003_cpu_campaign", "day23.rsi_v0003_gpu_campaign"}, "campaign schema drifted")
    _require(value.get("campaign_id") == CAMPAIGN_ID and value.get("version_id") == VERSION_ID, "campaign identity drifted")
    implementations = value.get("implementation_sources")
    _require(
        isinstance(implementations, Mapping)
        and set(implementations)
        == {"builder_and_binder", "training_runner", "candidate_evaluator", "charter_validator"},
        "implementation source inventory drifted",
    )
    recipe = value.get("fixed_recipe")
    _require(isinstance(recipe, Mapping), "fixed recipe missing")
    _require(recipe.get("single_scientific_lever") == SCIENTIFIC_LEVER, "scientific lever drifted")
    _require(recipe.get("lever_levels") == list(LEARNING_RATES), "learning-rate grid drifted")
    topology = recipe.get("topology")
    _require(
        isinstance(topology, Mapping)
        and topology.get("world_size") == WORLD_SIZE
        and topology.get("per_device_train_batch_size") == PER_DEVICE_TRAIN_BATCH_SIZE
        and topology.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION_STEPS
        and topology.get("nominal_global_train_batch_size") == NOMINAL_GLOBAL_BATCH_SIZE
        and topology.get("b24_optimizer_authorized") is False,
        "B16/world2/GA1 topology drifted or B24 was authorized",
    )
    runtime_recipe = recipe.get("runtime")
    _require(
        isinstance(runtime_recipe, Mapping)
        and runtime_recipe.get("world_size") == 2
        and runtime_recipe.get("per_device_train_batch_size") == 16
        and runtime_recipe.get("gradient_accumulation_steps") == 1
        and runtime_recipe.get("min_free_memory_fraction") == 0.30,
        "runtime B16/GA1/30-percent-headroom contract drifted",
    )
    datasets = value.get("datasets")
    _require(isinstance(datasets, Mapping), "datasets missing")
    _require(datasets.get("fit", {}).get("records") == 124, "fit count drifted")
    _require(datasets.get("search", {}).get("records") == 30, "search count drifted")
    _require(datasets.get("full_train", {}).get("records") == 154, "train count drifted")
    _require(datasets.get("dev", {}).get("records") == 17, "dev count drifted")
    heldout = datasets.get("heldout")
    _require(isinstance(heldout, Mapping) and heldout.get("runtime_path_disclosed") is False and "path" not in heldout, "heldout path or lease leaked")
    runs = value.get("run_specs")
    _require(isinstance(runs, Mapping) and set(runs) == set(CONFIG_FILENAMES), "run spec registry drifted")
    for run_id, spec in runs.items():
        _require(isinstance(spec, Mapping), f"invalid run spec: {run_id}")
        _require(spec.get("dataset_key") in {"fit", "full_train"}, f"invalid optimizer dataset: {run_id}")
        _require(spec.get("dataset") == datasets[spec["dataset_key"]], f"dataset identity drifted: {run_id}")
        _require(spec.get("authorized") is (spec.get("role") == "search_train"), f"static authorization drifted: {run_id}")
        _require(spec.get("runtime", {}).get("per_device_train_batch_size") == 16, f"B16 drifted: {run_id}")
        _require(spec.get("runtime", {}).get("world_size") == 2, f"world size drifted: {run_id}")
        _require(spec.get("runtime", {}).get("gradient_accumulation_steps") == 1, f"GA drifted: {run_id}")
        _require(spec.get("learning_rate") in LEARNING_RATES, f"LR grid drifted: {run_id}")
        config_identity = spec.get("executable_config")
        _require(isinstance(config_identity, Mapping), f"config identity missing: {run_id}")
        if payloads is not None:
            path = str(config_identity.get("path"))
            payload = payloads.get(path)
            _require(payload is not None and _sha256_bytes(payload) == config_identity.get("file_sha256"), f"config bytes drifted: {run_id}")
            config = json.loads(payload)
            _require(not (set(config) & FORBIDDEN_EXECUTABLE_KEYS), f"forbidden config keys: {run_id}")
            _require(config.get("per_device_train_batch_size") == 16 and config.get("gradient_accumulation_steps") == 1, f"config topology drifted: {run_id}")
            _require(config.get("learning_rate") == spec.get("learning_rate"), f"config LR drifted: {run_id}")
            _require(config.get("max_steps") == spec.get("max_steps"), f"config max_steps drifted: {run_id}")
    _require(value.get("leases") == {"search": 4, "dev": 1, "heldout": 0}, "lease budget drifted")
    ledger = value.get("access_ledger")
    _require(
        isinstance(ledger, Mapping)
        and ledger.get("authority") == "charter.access_leases"
        and ledger.get("dev_lease", {}).get("maximum_global_claims") == 1
        and ledger.get("heldout_lease", {}).get("current_authorized_claims") == 0
        and ledger.get("heldout_claim_disclosed") is False
        and "claim_path" not in ledger.get("heldout_lease", {})
        and isinstance(ledger.get("dev_claim"), str)
        and isinstance(ledger.get("search_claim"), str),
        "global dev or heldout access-ledger boundary drifted",
    )
    budgets = value.get("budgets")
    _require(isinstance(budgets, Mapping) and budgets.get("maximum_optimizer_steps") == 90 and budgets.get("maximum_training_trajectories") == 3, "training budget drifted")
    protocol = value.get("protocol")
    _require(isinstance(protocol, Mapping), "protocol missing")
    _require(protocol.get("dev", {}).get("minimum_positive_margin_count") == 13, "dev gate drifted")
    _require(protocol.get("heldout", {}).get("lease_count") == 0, "heldout lease drifted")


def select_search_candidate(results: Sequence[Mapping[str, Any]]) -> str | None:
    """Apply the frozen eligibility and lexicographic ranking rule."""
    expected = {
        _candidate_id(lr, step) for lr in LEARNING_RATES for step in CHECKPOINT_STEPS
    }
    _require(len(results) == 4 and {str(row.get("candidate_id")) for row in results} == expected, "exactly the four locked candidates are required")
    eligible: list[Mapping[str, Any]] = []
    for row in results:
        finite = row.get("finite_records") == 30
        integrity = row.get("runtime_reload_freeze_integrity") == "pass"
        positive = int(row.get("positive_margin_count", -1))
        mean = float(row.get("mean_margin", float("-inf")))
        length = float(row.get("length_matched_mean_margin", float("-inf")))
        if finite and integrity and positive >= 20 and mean > 0.0 and length > 0.0:
            eligible.append(row)
    if not eligible:
        return None
    winner = min(
        eligible,
        key=lambda row: (
            -int(row["positive_margin_count"]),
            -float(row["mean_margin"]),
            -float(row["length_matched_mean_margin"]),
            int(row["checkpoint_step"]),
            float(row["learning_rate"]),
        ),
    )
    candidate_id = str(winner["candidate_id"])
    expected_metadata = next(
        (lr, step)
        for lr in LEARNING_RATES
        for step in CHECKPOINT_STEPS
        if _candidate_id(lr, step) == candidate_id
    )
    _require(
        (float(winner.get("learning_rate")), int(winner.get("checkpoint_step"))) == expected_metadata,
        "candidate metadata does not match its locked ID",
    )
    return candidate_id


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(k) or _contains_placeholder(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(v) for v in value)
    return isinstance(value, str) and "__BIND_" in value


def _absolute_dataset(identity: Mapping[str, Any], bootcamp_root: Path) -> dict[str, Any]:
    result = dict(identity)
    if "path" in result:
        result["path"] = str((bootcamp_root / str(result["path"])).resolve())
    return result


def bind_campaign(
    cpu_campaign: Mapping[str, Any],
    *,
    bootcamp_root: Path,
    run_root: Path,
    model_path: Path,
    python_path: Path,
    ms_swift_checkout: Path,
    source_binding: Mapping[str, Any],
    source_binding_identity: Mapping[str, Any],
    runtime_parse: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Purely project a verified CPU campaign into absolute GPU payloads."""
    validate_campaign(cpu_campaign)
    bootcamp = bootcamp_root.resolve()
    run = run_root.resolve()
    model = model_path.resolve()
    python = python_path.resolve()
    checkout = ms_swift_checkout.resolve()
    _require(all(path.is_absolute() for path in (bootcamp, run, model, python, checkout)), "binding paths must be absolute")
    _require(source_binding.get("remote_parent") is not None, "source binding remote_parent missing")
    remote_parent = dict(source_binding["remote_parent"])
    _require(Path(str(remote_parent.get("merged_export", {}).get("path"))).resolve() == model, "model path differs from verified remote parent")
    _require(runtime_parse.get("status") == "pass", "new RSI config parse did not pass")
    _require(Path(str(runtime_parse.get("python_executable", {}).get("path"))).resolve() == python, "parsed Python path drifted")
    _require(Path(str(runtime_parse.get("ms_swift_checkout"))).resolve() == checkout, "parsed ms-swift checkout drifted")

    for relative, identity in cpu_campaign["artifact_inventory"].items():
        path = (bootcamp / relative).resolve()
        _require(path.is_file() and not path.is_symlink(), f"CPU campaign artifact missing: {path}")
        _require(
            contract.file_sha256(path) == identity["file_sha256"]
            and path.stat().st_size == identity["bytes"],
            f"CPU campaign artifact drifted: {relative}",
        )
    for name, identity in cpu_campaign["implementation_sources"].items():
        path = (bootcamp / str(identity["path"])).resolve()
        _require(path.is_file() and not path.is_symlink(), f"implementation missing: {name}")
        _require(
            contract.file_sha256(path) == identity["file_sha256"]
            and path.stat().st_size == identity["bytes"],
            f"implementation drifted: {name}",
        )
    charter_identity = cpu_campaign.get("charter")
    _require(isinstance(charter_identity, Mapping), "authoritative charter binding missing")
    charter_path = (bootcamp / str(charter_identity.get("path"))).resolve()
    charter = _load_self_hashed(charter_path, "charter_sha256", "authoritative RSI charter")
    _require(
        contract.file_sha256(charter_path) == charter_identity.get("file_sha256")
        and charter["charter_sha256"] == charter_identity.get("content_sha256"),
        "authoritative RSI charter drifted",
    )
    charter_leases = charter.get("access_leases")
    campaign_ledger = cpu_campaign.get("access_ledger")
    _require(isinstance(charter_leases, Mapping) and isinstance(campaign_ledger, Mapping), "charter lease binding missing")
    _require(
        campaign_ledger.get("access_leases_sha256")
        == contract.object_sha256(charter_leases)
        and campaign_ledger.get("dev_lease") == charter_leases.get("dev")
        and campaign_ledger.get("heldout_lease")
        == {
            key: item
            for key, item in charter_leases.get("heldout", {}).items()
            if key != "claim_path"
        }
        and campaign_ledger.get("dev_claim")
        == f"rsi-control/{charter_leases.get('dev', {}).get('claim_path')}",
        "campaign access ledger does not exactly inherit charter leases",
    )

    datasets = {
        key: (_absolute_dataset(identity, bootcamp) if key != "heldout" else dict(identity))
        for key, identity in cpu_campaign["datasets"].items()
    }
    for key, identity in datasets.items():
        if key in {"dev", "heldout"}:
            continue
        path = Path(str(identity["path"]))
        _require(path.is_file() and not path.is_symlink(), f"bound dataset missing: {key}")
        _require(contract.file_sha256(path) == identity["file_sha256"], f"bound dataset hash drifted: {key}")
        rows = contract.load_jsonl(path)
        _require(len(rows) == identity["records"], f"bound dataset count drifted: {key}")
        _require(
            contract.object_sha256([str(row["pair_id"]) for row in rows])
            == identity["ordered_pair_ids_sha256"],
            f"bound dataset ordered IDs drifted: {key}",
        )
    bound_config_payloads: dict[str, bytes] = {}
    run_specs: dict[str, Any] = {}
    for run_id, cpu_spec in cpu_campaign["run_specs"].items():
        source_config_path = bootcamp / str(cpu_spec["executable_config"]["path"])
        _require(source_config_path.is_file() and not source_config_path.is_symlink(), f"CPU config missing: {source_config_path}")
        source_payload = source_config_path.read_bytes()
        _require(_sha256_bytes(source_payload) == cpu_spec["executable_config"]["file_sha256"], f"CPU config hash drifted: {run_id}")
        config = json.loads(source_payload)
        config["model"] = str(model)
        config["dataset"] = [datasets[cpu_spec["dataset_key"]]["path"]]
        config["external_plugins"] = [str((bootcamp / path).resolve()) for path in config["external_plugins"]]
        output_dir = (run / "outputs" / run_id).resolve()
        config["output_dir"] = str(output_dir)
        _require(not _contains_placeholder(config), f"unresolved executable placeholder: {run_id}")
        _require(not (set(config) & FORBIDDEN_EXECUTABLE_KEYS), f"forbidden executable keys: {run_id}")
        filename = CONFIG_FILENAMES[run_id]
        bound_path = (run / "binding" / "configs" / filename).resolve()
        payload = _json_bytes(config)
        bound_config_payloads[f"configs/{filename}"] = payload
        spec = dict(cpu_spec)
        spec.update(
            {
                "dataset": datasets[cpu_spec["dataset_key"]],
                "model_argument": str(model),
                "executable_config": _config_identity(str(bound_path), payload, config),
                "output_dir": str(output_dir),
                "success_receipt": str((run / "evidence" / run_id / "success-receipt.json").resolve()),
                "failure_receipt": str((run / "evidence" / run_id / "failure-receipt.json").resolve()),
            }
        )
        run_specs[run_id] = spec

    fixed_recipe = json.loads(json.dumps(cpu_campaign["fixed_recipe"]))
    fixed_recipe["model"]["model_argument"] = str(model)
    fixed_recipe["runtime"].update(
        {"python": str(python), "ms_swift_checkout": str(checkout)}
    )
    access_ledger = json.loads(json.dumps(cpu_campaign["access_ledger"]))
    access_ledger["search_claim"] = str(
        (run / "evidence/search-selection/search-claim.json").resolve()
    )
    access_ledger["dev_claim"] = str(
        (bootcamp / str(cpu_campaign["access_ledger"]["dev_claim"])).resolve()
    )
    artifact_inventory = {
        name: {"file_sha256": _sha256_bytes(payload), "bytes": len(payload)}
        for name, payload in sorted(bound_config_payloads.items())
    }
    producer_paths = {
        "binder": (bootcamp / "day-23-dpo-theory-smoke/build_day23_rsi_v0003.py").resolve(),
        "runner": (bootcamp / "day-23-dpo-theory-smoke/run_day23_rsi_candidate.py").resolve(),
        "evaluator": (bootcamp / "day-23-dpo-theory-smoke/eval_day23_rsi_candidate.py").resolve(),
    }
    _require(
        all(path.is_file() and not path.is_symlink() for path in producer_paths.values()),
        "an RSI GPU producer is missing",
    )
    gpu: dict[str, Any] = {
        **{k: json.loads(json.dumps(v)) for k, v in cpu_campaign.items() if k not in {"campaign_sha256", "artifact_inventory", "status", "schema_name", "bootcamp_root", "remote_run_root", "parent", "fixed_recipe", "datasets", "run_specs"}},
        "schema_name": "day23.rsi_v0003_gpu_campaign",
        "schema_version": 1,
        "status": "gpu_execution_bound_optimizer_pending",
        "bootcamp_root": str(bootcamp),
        "remote_run_root": str(run),
        "source_cpu_campaign": {
            "path": str((bootcamp / CPU_CAMPAIGN_REL).resolve()),
            "file_sha256": contract.file_sha256(bootcamp / CPU_CAMPAIGN_REL),
            "campaign_sha256": cpu_campaign["campaign_sha256"],
        },
        "source_binding": dict(source_binding_identity),
        "producers": {
            name: _file_identity(path) for name, path in producer_paths.items()
        },
        "remote_parent": remote_parent,
        "runtime_parse": dict(runtime_parse),
        "parent": {**cpu_campaign["parent"], "model_path": str(model)},
        "fixed_recipe": fixed_recipe,
        "datasets": datasets,
        "run_specs": dict(sorted(run_specs.items())),
        "access_ledger": access_ledger,
        "artifact_inventory": artifact_inventory,
    }
    gpu["claim_boundary"].update(
        {
            "remote_s1_payload_bytes_verified_via_source_binding": True,
            "real_remote_rlhf_arguments_json_parse_passed": True,
            "gpu_campaign_bound": True,
        }
    )
    _require(not _contains_placeholder(gpu), "GPU campaign contains an unresolved binding placeholder")
    gpu["campaign_sha256"] = contract.object_sha256(gpu)
    validate_campaign(gpu, payloads={spec["executable_config"]["path"]: bound_config_payloads[f"configs/{CONFIG_FILENAMES[run_id]}"] for run_id, spec in gpu["run_specs"].items()})
    payloads = dict(bound_config_payloads)
    payloads["gpu-campaign.json"] = _json_bytes(gpu)
    return payloads, gpu


def _git_identity(path: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day23RSICampaignError("cannot inspect ms-swift checkout") from error
    return commit, not bool(dirty)


def _parse_bound_configs(
    configs: Mapping[str, bytes],
    *,
    python_path: Path,
    ms_swift_checkout: Path,
    source_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse every bound JSON through the supplied pinned RLHFArguments runtime."""
    python = python_path.resolve()
    checkout = ms_swift_checkout.resolve()
    _require(python.is_file() and not python.is_symlink(), "Python executable is missing")
    _require(checkout.is_dir() and not checkout.is_symlink(), "ms-swift checkout is missing")
    commit, clean = _git_identity(checkout)
    _require(commit == contract.MS_SWIFT_COMMIT and clean, "ms-swift checkout is not the pinned clean commit")
    expected_packages = source_runtime.get("package_versions")
    _require(isinstance(expected_packages, Mapping), "source binding package versions missing")
    script = r'''
import dataclasses, importlib.metadata, json, os, platform, sys
from pathlib import Path
import swift
from swift.arguments import RLHFArguments
from swift.cli.main import parse_yaml_args
from swift.utils import parse_args
paths = [Path(p) for p in sys.argv[1:]]
fields = {f.name for f in dataclasses.fields(RLHFArguments)}
stages = {}
for path in paths:
    value = json.loads(path.read_text())
    unknown = sorted(set(value) - fields)
    if unknown:
        raise RuntimeError(f"unknown RLHFArguments keys: {unknown}")
    argv = [str(path)]
    old = os.environ.get("SWIFT_CONFIG_FILE")
    try:
        parse_yaml_args(argv)
        parsed, remaining = parse_args(RLHFArguments, argv)
    finally:
        if old is None:
            os.environ.pop("SWIFT_CONFIG_FILE", None)
        else:
            os.environ["SWIFT_CONFIG_FILE"] = old
    if remaining:
        raise RuntimeError(f"unparsed arguments: {remaining}")
    ta = parsed.training_args
    stages[path.stem] = {
        "status": "pass", "real_json_cli": True,
        "executable_config_keys": sorted(value),
        "rlhf_type": parsed.rlhf_type, "tuner_type": parsed.tuner_type,
        "template": parsed.template, "beta": parsed.beta, "loss_type": parsed.loss_type,
        "max_steps": parsed.max_steps, "save_steps": parsed.save_steps,
        "learning_rate": parsed.learning_rate,
        "per_device_train_batch_size": ta.per_device_train_batch_size,
        "per_device_eval_batch_size": ta.per_device_eval_batch_size,
        "gradient_accumulation_steps": ta.gradient_accumulation_steps,
        "adapters": parsed.adapters, "ref_adapters": parsed.ref_adapters,
        "ref_model": parsed.ref_model, "resume_from_checkpoint": parsed.resume_from_checkpoint,
    }
packages = {}
for package in json.loads(os.environ["DAY23_RSI_EXPECTED_PACKAGES"]):
    packages[package] = importlib.metadata.version(package)
print(json.dumps({
    "python_version": platform.python_version(), "platform": platform.platform(),
    "swift_file": str(Path(swift.__file__).resolve()),
    "package_versions": packages, "rlhf_argument_field_count": len(fields), "stages": stages,
}, sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="day23-rsi-v0003-parse-") as temporary:
        temporary_root = Path(temporary)
        paths: list[Path] = []
        for run_id, payload in sorted(configs.items()):
            value = json.loads(payload)
            value["output_dir"] = str(temporary_root / f"output-{run_id}")
            path = temporary_root / f"{run_id}.json"
            path.write_bytes(_json_bytes(value))
            paths.append(path)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(checkout) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["DAY23_RSI_EXPECTED_PACKAGES"] = json.dumps(sorted(expected_packages))
        try:
            completed = subprocess.run([str(python), "-c", script, *map(str, paths)], check=False, capture_output=True, text=True, env=env)
        except OSError as error:
            raise Day23RSICampaignError("cannot run pinned RLHFArguments parser") from error
    _require(completed.returncode == 0, "pinned RLHFArguments parse failed: " + completed.stderr.strip())
    try:
        parsed = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise Day23RSICampaignError("pinned parser did not emit JSON") from error
    _require(parsed["package_versions"] == dict(expected_packages), "GPU package versions drifted from source binding")
    imported_root = Path(parsed["swift_file"]).resolve().parent
    _require(imported_root == checkout / "swift", "swift imported outside pinned checkout")
    _require(set(parsed["stages"]) == set(configs), "not every RSI config was parsed")
    for run_id, result in parsed["stages"].items():
        _require(
            result["status"] == "pass"
            and result["rlhf_type"] == "dpo"
            and result["tuner_type"] == "lora"
            and result["template"] == "day23_qwen3_5_dpo_target_v1"
            and result["beta"] == 0.1
            and result["loss_type"] == "sigmoid"
            and result["per_device_train_batch_size"] == 16
            and result["per_device_eval_batch_size"] == 4
            and result["gradient_accumulation_steps"] == 1
            and result["adapters"] == []
            and result["ref_adapters"] == []
            and result["ref_model"] is None
            and result["resume_from_checkpoint"] is None,
            f"parsed RSI semantics drifted: {run_id}",
        )
    return {
        "status": "pass",
        "scope": "real_remote_RLHFArguments_JSON_parse_no_model_weights",
        "python_executable": {"path": str(python), "file_sha256": contract.file_sha256(python), "version": parsed["python_version"]},
        "platform": parsed["platform"],
        "ms_swift_checkout": str(checkout),
        "ms_swift_commit": commit,
        "ms_swift_clean": clean,
        "ms_swift_import_root": str(imported_root),
        "ms_swift_package_version": parsed["package_versions"]["ms-swift"],
        "package_versions": dict(sorted(parsed["package_versions"].items())),
        "rlhf_argument_field_count": parsed["rlhf_argument_field_count"],
        "stages": parsed["stages"],
    }


def build_bound_bundle(
    *,
    bootcamp_root: Path,
    run_root: Path,
    model_path: Path,
    python_path: Path,
    ms_swift_checkout: Path,
    source_binding_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    bootcamp = bootcamp_root.resolve()
    run = run_root.resolve()
    _require(bootcamp.is_dir() and not bootcamp.is_symlink(), "bootcamp root missing")
    _require(run.is_dir() and not run.is_symlink(), "run root must already exist")
    cpu_path = bootcamp / CPU_CAMPAIGN_REL
    cpu_campaign = _load_self_hashed(cpu_path, "campaign_sha256", "CPU campaign")
    validate_campaign(cpu_campaign)
    source_path = source_binding_path.resolve()
    source_binding = _load_self_hashed(source_path, "binding_sha256", "source Day23 GPU binding")
    _require(source_binding.get("schema_name") == "day23.qwen35_dpo_gpu_execution_binding", "source binding schema drifted")
    _require(source_binding.get("status") == "gpu_execution_bound", "source binding is not GPU-bound")
    source_runtime = source_binding.get("runtime_parse")
    _require(isinstance(source_runtime, Mapping), "source runtime identity missing")
    _require(Path(str(source_runtime.get("python_executable", {}).get("path"))).resolve() == python_path.resolve(), "Python differs from source binding")
    _require(Path(str(source_runtime.get("ms_swift_checkout"))).resolve() == ms_swift_checkout.resolve(), "ms-swift checkout differs from source binding")
    _require(
        contract.file_sha256(python_path.resolve())
        == source_runtime.get("python_executable", {}).get("file_sha256"),
        "Python executable bytes differ from source binding",
    )
    _require(
        source_runtime.get("ms_swift_commit") == contract.MS_SWIFT_COMMIT
        and source_runtime.get("ms_swift_clean") is True,
        "source binding ms-swift identity drifted",
    )

    # First make absolute configs with the source runtime identity, then parse
    # their exact semantic projection and reseal with the new parse evidence.
    provisional_runtime = dict(source_runtime)
    provisional_runtime["status"] = "pass"
    provisional_payloads, _ = bind_campaign(
        cpu_campaign,
        bootcamp_root=bootcamp,
        run_root=run,
        model_path=model_path,
        python_path=python_path,
        ms_swift_checkout=ms_swift_checkout,
        source_binding=source_binding,
        source_binding_identity={"path": str(source_path), "file_sha256": contract.file_sha256(source_path), "binding_sha256": source_binding["binding_sha256"]},
        runtime_parse=provisional_runtime,
    )
    parse_inputs = {
        run_id: provisional_payloads[f"configs/{filename}"]
        for run_id, filename in CONFIG_FILENAMES.items()
    }
    runtime_parse = _parse_bound_configs(
        parse_inputs,
        python_path=python_path,
        ms_swift_checkout=ms_swift_checkout,
        source_runtime=source_runtime,
    )
    return bind_campaign(
        cpu_campaign,
        bootcamp_root=bootcamp,
        run_root=run,
        model_path=model_path,
        python_path=python_path,
        ms_swift_checkout=ms_swift_checkout,
        source_binding=source_binding,
        source_binding_identity={"path": str(source_path), "file_sha256": contract.file_sha256(source_path), "binding_sha256": source_binding["binding_sha256"]},
        runtime_parse=runtime_parse,
    )


def apply_bundle(payloads: Mapping[str, bytes], *, output_root: Path, mode: str) -> None:
    root = output_root.resolve()
    targets = {relative: (root / relative).resolve() for relative in payloads}
    _require(all(root == path or root in path.parents for path in targets.values()), "output escaped root")
    if mode == "build":
        existing = [str(path) for path in targets.values() if path.exists() or path.is_symlink()]
        _require(not existing, "refusing to overwrite: " + ", ".join(existing))
        for path in targets.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for relative, path in targets.items():
            with path.open("xb") as handle:
                handle.write(payloads[relative])
    elif mode == "check":
        for relative, path in targets.items():
            _require(path.is_file() and not path.is_symlink(), f"missing frozen output: {path}")
            _require(path.read_bytes() == payloads[relative], f"frozen output drifted: {path}")
    else:
        raise Day23RSICampaignError(f"unsupported mode: {mode}")


def _apply_gpu_bundle(payloads: Mapping[str, bytes], run_root: Path) -> None:
    binding_root = (run_root.resolve() / "binding").resolve()
    _require(not binding_root.exists() and not binding_root.is_symlink(), f"refusing existing binding directory: {binding_root}")
    binding_root.mkdir(parents=False, exist_ok=False)
    for relative, payload in payloads.items():
        path = (binding_root / relative).resolve()
        _require(binding_root in path.parents, "GPU payload escaped binding root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "check", "bind"), required=True)
    parser.add_argument("--source-root", type=Path, default=BOOTCAMP_ROOT)
    parser.add_argument("--output-root", type=Path, default=BOOTCAMP_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--bootcamp-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--python", dest="python_path", type=Path)
    parser.add_argument("--ms-swift-checkout", type=Path)
    parser.add_argument("--source-binding", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode in {"build", "check"}:
            payloads, campaign = build_campaign_bundle(args.source_root)
            apply_bundle(payloads, output_root=args.output_root, mode=args.mode)
        else:
            required = {
                "--run-root": args.run_root,
                "--bootcamp-root": args.bootcamp_root,
                "--model-path": args.model_path,
                "--python": args.python_path,
                "--ms-swift-checkout": args.ms_swift_checkout,
                "--source-binding": args.source_binding,
            }
            missing = [name for name, value in required.items() if value is None]
            _require(not missing, "bind requires " + ", ".join(missing))
            payloads, campaign = build_bound_bundle(
                bootcamp_root=args.bootcamp_root,
                run_root=args.run_root,
                model_path=args.model_path,
                python_path=args.python_path,
                ms_swift_checkout=args.ms_swift_checkout,
                source_binding_path=args.source_binding,
            )
            _apply_gpu_bundle(payloads, args.run_root)
    except (OSError, ValueError, contract.Day23ContractError, Day23RSICampaignError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": campaign["status"], "mode": args.mode, "campaign_sha256": campaign["campaign_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
