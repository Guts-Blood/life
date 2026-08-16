#!/usr/bin/env python3
"""Strict authority preflight for the goal-0003 csearch-v0001 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_day23_csearch_v0001 as builder
import day23_contract as contract
import run_day23_qwen35_gpu_stage as day23_gpu


class CSearchAuthorityError(RuntimeError):
    """A csearch authority, source-terminal, or execution invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CSearchAuthorityError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value) and "\x00" not in value, f"{label} must be text")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CSearchAuthorityError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def identity(path: Path, field: str | None = None, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if field is not None:
        result["content_sha256"] = verify(load(resolved, str(resolved)), field, str(resolved))
    return result


def _validate_runtime_dependencies(campaign: Mapping[str, Any], bootcamp: Path) -> dict[str, Any]:
    dependencies = mapping(campaign.get("runtime_dependencies"), "csearch runtime dependencies")
    expected = builder._runtime_dependency_paths()
    require(set(dependencies) == set(expected), "csearch runtime dependency registry drifted")
    validated: dict[str, Any] = {}
    for name, expected_path in sorted(expected.items()):
        entry = mapping(dependencies.get(name), f"csearch runtime dependency {name}")
        path = Path(text(entry.get("path"), f"csearch runtime dependency {name}.path")).resolve(strict=True)
        expected_resolved = expected_path.resolve(strict=True)
        require(
            set(entry) == {"path", "file_sha256", "bytes"}
            and path == expected_resolved
            and path.is_relative_to(bootcamp)
            and path.is_file()
            and not path.is_symlink()
            and day23_gpu.file_sha256(path) == entry.get("file_sha256")
            and path.stat().st_size == entry.get("bytes"),
            f"csearch runtime dependency drifted: {name}",
        )
        validated[name] = dict(entry)
    return validated


def resolve_bound(
    bootcamp: Path,
    entry: Any,
    field: str,
    label: str,
    *,
    allow_external: bool = False,
) -> tuple[dict[str, Any], Path]:
    bound = mapping(entry, label)
    raw = Path(text(bound.get("path"), f"{label}.path"))
    path = (raw if raw.is_absolute() else bootcamp / raw).resolve(strict=True)
    if not allow_external:
        require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    value = load(path, label)
    actual = identity(path, field, relative_to=None if raw.is_absolute() else bootcamp)
    require(dict(bound) == actual, f"{label} identity drifted")
    return value, path


def _validate_source(campaign: Mapping[str, Any], bootcamp: Path) -> dict[str, Any]:
    source_terminal = mapping(campaign.get("source_terminal"), "source terminal")
    source, source_path = resolve_bound(
        bootcamp,
        source_terminal.get("campaign"),
        "campaign_sha256",
        "source RSI-v0005 campaign",
        allow_external=True,
    )
    source_sha = source["campaign_sha256"]
    selection, selection_path = resolve_bound(
        bootcamp,
        source_terminal.get("selection"),
        "selection_sha256",
        "source RSI-v0005 selection",
        allow_external=True,
    )
    require(
        source.get("schema_name") == builder.SOURCE_SCHEMA
        and source.get("version_id") == "rsi-v0005"
        and source_path == Path(str(source.get("remote_run_root"))).resolve() / "binding/gpu-campaign.json",
        "source RSI-v0005 campaign drifted",
    )
    require(
        selection.get("schema_name") == builder.SOURCE_SELECTION_SCHEMA
        and selection.get("campaign_sha256") == source_sha
        and selection.get("status") == "closed_no_candidate"
        and selection.get("eligible_count") == 0
        and selection.get("candidate_count") == 4
        and selection.get("selected_candidate") is None
        and source_terminal.get("status") == "closed_no_candidate"
        and source_terminal.get("eligible_count") == 0
        and source_terminal.get("candidate_or_resume") is False,
        "source RSI-v0005 terminal claim drifted",
    )
    require(
        selection_path == Path(str(source.get("remote_run_root"))).resolve() / "evidence/selection/search-selection.json",
        "source selection is not canonical",
    )
    candidates = selection.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 4, "source evaluation registry drifted")
    for index, candidate_value in enumerate(candidates):
        candidate = mapping(candidate_value, f"source candidate {index}")
        evaluation_path = Path(text(candidate.get("path"), f"source candidate {index}.path")).resolve(strict=True)
        require(
            day23_gpu.file_sha256(evaluation_path) == candidate.get("file_sha256"),
            f"source evaluation file drifted: {index}",
        )
        evaluation = load(evaluation_path, f"source evaluation {index}")
        require(
            verify(evaluation, "evaluation_sha256", f"source evaluation {index}") == candidate.get("evaluation_sha256")
            and evaluation.get("campaign_sha256") == source_sha
            and evaluation.get("split") == "search"
            and evaluation.get("eligibility", {}).get("passed") is False
            and candidate.get("eligible") is False,
            f"source evaluation terminal status drifted: {index}",
        )
    source_specs = mapping(source.get("run_specs"), "source run specs")
    search_specs = [mapping(value, f"source run {run_id}") for run_id, value in source_specs.items() if isinstance(value, Mapping) and value.get("role") == "search_train"]
    require(len(search_specs) == 2, "source training trajectory count drifted")
    for spec in search_specs:
        receipt_path = Path(text(spec.get("success_receipt"), "source training receipt path")).resolve(strict=True)
        receipt = load(receipt_path, "source training receipt")
        require(
            verify(receipt, "receipt_sha256", "source training receipt")
            and receipt.get("status") == "pass"
            and receipt.get("campaign", {}).get("campaign_sha256") == source_sha
            and receipt.get("observations", {}).get("global_step") == 30,
            "source training receipt drifted",
        )
    return {
        "campaign": source,
        "campaign_path": source_path,
        "campaign_sha256": source_sha,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": selection["selection_sha256"],
    }


def _validate_partition(campaign: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    source_campaign = source["campaign"]
    source_datasets = mapping(source_campaign.get("datasets"), "source datasets")
    full_rows, full_path = builder._validate_dataset_entry(source_datasets.get("full_train"), "source full154")
    v5_rows, _ = builder._validate_dataset_entry(source_datasets.get("search"), "source v0005 search30")
    prior = mapping(source_campaign.get("supersedes"), "source v0004 predecessor")
    v4_path = Path(text(prior.get("campaign_path"), "source v0004 campaign path")).resolve(strict=True)
    require(day23_gpu.file_sha256(v4_path) == prior.get("campaign_file_sha256"), "source v0004 campaign file drifted")
    v4_campaign = load(v4_path, "source v0004 campaign")
    require(verify(v4_campaign, "campaign_sha256", "source v0004 campaign") == prior.get("campaign_sha256"), "source v0004 campaign binding drifted")
    v4_rows, _ = builder._validate_dataset_entry(mapping(v4_campaign.get("datasets"), "v0004 datasets").get("search"), "source v0004 search30")
    fit_rows, search_rows, expected_partition = builder.rotate_tranches(
        full_rows,
        {str(row["pair_id"]) for row in v4_rows},
        {str(row["pair_id"]) for row in v5_rows},
    )
    require(campaign.get("data_partition") == expected_partition, "csearch tranche plan drifted")
    datasets = mapping(campaign.get("datasets"), "csearch datasets")
    run_root = Path(text(campaign.get("remote_run_root"), "csearch run root")).resolve(strict=True)
    expected: dict[str, Any] = {}
    for key, rows, relative, role in (
        ("fit", fit_rows, "binding/data/day23-qwen35-dpo-csearch-v0001-fit.jsonl", "search_optimizer_input"),
        ("search", search_rows, "binding/data/day23-qwen35-dpo-csearch-v0001-search.jsonl", "tranche1_locked_ranking_only"),
    ):
        path = run_root / relative
        payload = contract.jsonl_bytes(rows)
        entry = builder._dataset_identity(path, payload, rows, role)
        require(datasets.get(key) == entry, f"csearch {key} dataset identity drifted")
        require(
            path.is_file()
            and not path.is_symlink()
            and day23_gpu.file_sha256(path) == entry["file_sha256"]
            and path.stat().st_size == entry["bytes"],
            f"csearch {key} dataset bytes drifted",
        )
        expected[key] = entry
    require(campaign.get("source_training_corpus") == identity(full_path), "source full154 binding drifted")
    return {"datasets": expected, "partition": expected_partition}


def _validate_recipe(campaign: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    recipe = mapping(campaign.get("fixed_recipe"), "csearch fixed recipe")
    require(
        recipe
        == {
            "only_scientific_lever": "model.optimization.learning_rate.base",
            "lever_levels": list(builder.LEARNING_RATES),
            "candidate_checkpoint_step": builder.CANDIDATE_STEP,
            "max_steps": builder.MAX_STEPS,
            "cosine_schedule_horizon": builder.MAX_STEPS,
            "search_seed": builder.SEARCH_SEED,
            "objective": {"rlhf_type": "dpo", "loss_type": "sigmoid", "beta": 0.1},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
            "runtime": {
                "world_size": builder.WORLD_SIZE,
                "per_device_train_batch_size": builder.PER_DEVICE_BATCH,
                "gradient_accumulation_steps": builder.GRADIENT_ACCUMULATION,
                "nominal_global_train_batch_size": builder.GLOBAL_BATCH,
                "min_free_memory_fraction": builder.MIN_FREE_FRACTION,
            },
            "fresh_s1_only": True,
        },
        "csearch fixed recipe drifted",
    )
    require(campaign.get("remote_parent") == source["campaign"].get("remote_parent"), "fresh-S1 remote parent drifted")
    parent = mapping(campaign.get("parent"), "csearch parent")
    require(
        parent.get("role") == "S1"
        and parent.get("policy") == "fresh_lora_over_merged_s1"
        and parent.get("resume_from_source_checkpoint") is False
        and parent.get("model_argument") == source["campaign"]["fixed_recipe"]["model"]["model_argument"],
        "fresh-S1 parent contract drifted",
    )
    fit_entry = mapping(mapping(campaign.get("datasets"), "csearch datasets").get("fit"), "csearch fit dataset")
    partition_path = Path(text(fit_entry.get("path"), "csearch fit dataset path")).resolve(strict=True)
    specs = mapping(campaign.get("run_specs"), "csearch run specs")
    require(set(specs) == set(builder.RUN_IDS), "csearch run registry drifted")
    runtime_parse = mapping(campaign.get("runtime_parse"), "csearch runtime parse")
    runtime_stages = mapping(runtime_parse.get("stages"), "csearch runtime parse stages")
    require(
        runtime_parse.get("output_dir_binding")
        == "temporary_parse_sandbox_rebound_to_frozen_executable_config"
        and set(runtime_stages) == set(specs),
        "csearch runtime parse projection drifted",
    )
    seen_lrs: list[float] = []
    for run_id, spec_value in sorted(specs.items()):
        spec = mapping(spec_value, f"csearch run {run_id}")
        config_entry = mapping(spec.get("executable_config"), f"csearch config {run_id}")
        config_path = Path(text(config_entry.get("path"), f"csearch config {run_id}.path")).resolve(strict=True)
        require(
            config_path.is_file()
            and not config_path.is_symlink()
            and day23_gpu.file_sha256(config_path) == config_entry.get("file_sha256")
            and config_path.stat().st_size == config_entry.get("bytes"),
            f"csearch config bytes drifted: {run_id}",
        )
        config = load(config_path, f"csearch config {run_id}")
        runtime_stage = mapping(runtime_stages.get(run_id), f"csearch runtime stage {run_id}")
        lr = float(spec.get("learning_rate"))
        seen_lrs.append(lr)
        require(
            spec.get("role") == "search_train"
            and spec.get("authorized") is True
            and spec.get("fresh_start_from_parent") is True
            and spec.get("dataset_key") == "fit"
            and spec.get("seed") == builder.SEARCH_SEED
            and spec.get("checkpoint_steps") == [builder.CANDIDATE_STEP]
            and spec.get("checkpoint_capture")
            == {
                "candidate_checkpoint_steps": [builder.CANDIDATE_STEP],
                "expected_retained_checkpoint_steps": [builder.CANDIDATE_STEP, builder.MAX_STEPS],
                "retained_noncandidate_checkpoint_steps": [builder.MAX_STEPS],
                "retained_noncandidate_checkpoints_candidate_eligible": False,
            }
            and spec.get("candidate_checkpoint_only") == builder.CANDIDATE_STEP
            and spec.get("max_steps") == builder.MAX_STEPS
            and spec.get("cosine_schedule_horizon") == builder.MAX_STEPS
            and config.get("learning_rate") == lr
            and config.get("max_steps") == builder.MAX_STEPS
            and config.get("lr_scheduler_type") == "cosine"
            and config.get("save_steps") == builder.CANDIDATE_STEP
            and config.get("save_total_limit") == 2
            and config.get("seed") == config.get("data_seed") == builder.SEARCH_SEED
            and config.get("per_device_train_batch_size") == builder.PER_DEVICE_BATCH
            and config.get("gradient_accumulation_steps") == builder.GRADIENT_ACCUMULATION
            and config.get("beta") == 0.1
            and config.get("loss_type") == "sigmoid"
            and config.get("rlhf_type") == "dpo"
            and config.get("lora_rank") == 8
            and config.get("lora_alpha") == 16
            and config.get("lora_dropout") == 0.05
            and config.get("tuner_type") == "lora"
            and config.get("freeze_llm") is False
            and config.get("freeze_vit") is True
            and config.get("freeze_aligner") is True
            and isinstance(config.get("target_regex"), str)
            and bool(config["target_regex"])
            and config.get("gradient_checkpointing") is True
            and config.get("warmup_ratio") == 0.1
            and config.get("dataset_shuffle") is True
            and config.get("train_dataloader_shuffle") is True
            and config.get("save_strategy") == "steps"
            and config.get("dataset") == [str(partition_path)]
            and config.get("output_dir") == spec.get("output_dir"),
            f"csearch recipe drifted: {run_id}",
        )
        require(
            runtime_stage.get("status") == "pass"
            and runtime_stage.get("output_dir") == config.get("output_dir")
            and runtime_stage.get("max_steps") == builder.MAX_STEPS
            and runtime_stage.get("learning_rate") == lr
            and runtime_stage.get("per_device_train_batch_size") == builder.PER_DEVICE_BATCH
            and runtime_stage.get("gradient_accumulation_steps") == builder.GRADIENT_ACCUMULATION,
            f"csearch normalized runtime parse drifted: {run_id}",
        )
        for forbidden in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset"):
            require(forbidden not in config, f"forbidden fresh-start key present: {forbidden}")
        lowered = json.dumps(config, ensure_ascii=False, sort_keys=True).lower()
        require("heldout" not in lowered and "coding-dpo-dev" not in lowered, "protected path leaked into optimizer config")
    require(sorted(seen_lrs) == list(builder.LEARNING_RATES), "csearch LR grid drifted")
    require(
        campaign.get("candidate_registry")
        == [
            {
                "candidate_id": f"candidate_lr_{builder._lr_token(lr)}_step_20",
                "search_run_spec": f"search_lr_{builder._lr_token(lr)}",
                "learning_rate": lr,
                "checkpoint_step": builder.CANDIDATE_STEP,
            }
            for lr in builder.LEARNING_RATES
        ],
        "csearch candidate registry drifted",
    )


def validate(campaign_path: Path, *, require_search_unopened: bool) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = load(campaign_path, "csearch campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "csearch campaign")
    require(
        campaign.get("schema_name") == builder.CAMPAIGN_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("goal_id") == builder.GOAL_ID
        and campaign.get("version_id") == builder.VERSION_ID
        and campaign.get("campaign_id") == builder.CAMPAIGN_ID,
        "csearch campaign identity drifted",
    )
    run_root = Path(text(campaign.get("remote_run_root"), "csearch run root")).resolve(strict=True)
    bootcamp = Path(text(campaign.get("bootcamp_root"), "csearch bootcamp")).resolve(strict=True)
    require(campaign_path == run_root / "binding/gpu-campaign.json", "csearch campaign is not canonical")

    producers = mapping(campaign.get("producers"), "csearch producers")
    require(set(producers) == {"builder", "runner", "evaluator", "authority_validator"}, "csearch producer registry drifted")
    for name, entry_value in sorted(producers.items()):
        entry = mapping(entry_value, f"csearch producer {name}")
        path = Path(text(entry.get("path"), f"csearch producer {name}.path")).resolve(strict=True)
        require(
            path.is_relative_to(bootcamp)
            and path.is_file()
            and not path.is_symlink()
            and day23_gpu.file_sha256(path) == entry.get("file_sha256")
            and path.stat().st_size == entry.get("bytes"),
            f"csearch producer drifted: {name}",
        )
        if name == "authority_validator":
            require(path == Path(__file__).resolve(strict=True), "campaign bound another authority validator")
    dependencies = _validate_runtime_dependencies(campaign, bootcamp)

    control = mapping(campaign.get("control"), "csearch control")
    charter, _ = resolve_bound(bootcamp, control.get("charter"), "charter_sha256", "goal-0003 charter")
    event2, _ = resolve_bound(bootcamp, control.get("source_terminal_event"), "event_sha256", "goal-0003 source event")
    require(campaign.get("authority") == {"charter": control.get("charter"), "source_terminal_event": control.get("source_terminal_event")}, "csearch authority projection drifted")
    require(
        charter.get("schema_name") == "day23.csearch_goal_charter"
        and charter.get("schema_version") == 1
        and charter.get("status") == "frozen"
        and charter.get("goal_id") == builder.GOAL_ID
        and charter.get("iteration_budget", {}).get("allowed_version_ids")
        == ["csearch-v0001", "csearch-v0002", "csearch-v0003"]
        and charter.get("iteration_budget", {}).get("maximum_search_trajectories_per_version") == 2
        and charter.get("iteration_budget", {}).get("maximum_search_candidates_per_version") == 4
        and charter.get("iteration_budget", {}).get("exactly_one_scientific_lever_per_version") is True
        and charter.get("candidate_gate", {}).get("minimum_positive_pairs") == 20
        and charter.get("candidate_gate", {}).get("mean_reward_margin") == ">0"
        and charter.get("candidate_gate", {}).get("length_matched_mean_reward_margin") == ">0"
        and charter.get("candidate_gate", {}).get("threshold_relaxation_forbidden") is True
        and charter.get("scientific_recipe_authority", {}).get("frozen_by_initializer") is False
        and charter.get("scientific_recipe_authority", {}).get("required_before_each_version_optimizer") is True,
        "goal-0003 charter drifted",
    )
    require(
        event2.get("schema_name") == "day23.csearch_event"
        and event2.get("schema_version") == 1
        and event2.get("event_id") == "event-000002-source-terminal-imported"
        and event2.get("sequence") == 2
        and event2.get("event_type") == "source-terminal-imported"
        and event2.get("goal_id") == builder.GOAL_ID
        and event2.get("version_id") is None
        and event2.get("state_before") == "charter_frozen"
        and event2.get("state_after") == "iteration_open"
        and isinstance(event2.get("authority_refs"), list)
        and len(event2["authority_refs"]) == 3
        and event2["authority_refs"][0] == control.get("charter"),
        "goal-0003 source import event drifted",
    )

    source = _validate_source(campaign, bootcamp)
    partition = _validate_partition(campaign, source)
    event_payload = mapping(event2.get("payload"), "goal-0003 source import payload")
    source_terminal = mapping(campaign.get("source_terminal"), "csearch source terminal")
    require(
        event_payload.get("source_goal_id") == "goal-0002-day23-dpo"
        and event_payload.get("source_terminal_state") == "terminal_no_go"
        and event_payload.get("source_campaign") == source_terminal.get("campaign")
        and event_payload.get("source_selection") == source_terminal.get("selection")
        and event_payload.get("fresh_search_pair_count") == 90
        and event_payload.get("frozen_tranche_count") == 3
        and event_payload.get("next_version_id") == builder.VERSION_ID
        and event_payload.get("scientific_recipe_pending") is True,
        "goal-0003 source import payload drifted",
    )
    ledger = mapping(charter.get("search_exposure_ledger"), "goal-0003 exposure ledger")
    tranches = ledger.get("tranches")
    require(isinstance(tranches, list) and len(tranches) == 3, "goal-0003 tranche registry drifted")
    require(
        campaign.get("tranche_authority") == tranches[0]
        and tranches[0]
        == {
            "version_id": builder.VERSION_ID,
            "membership_rank_start_inclusive": 0,
            "membership_rank_end_exclusive": 30,
            "records": 30,
            "ordered_pair_ids_sha256": partition["datasets"]["search"]["ordered_pair_ids_sha256"],
            "ordered_row_hashes_sha256": partition["datasets"]["search"]["ordered_row_hashes_sha256"],
        }
        and ledger.get("eligible_pair_count") == 90
        and ledger.get("selection_salt") == builder.ROTATION_SALT
        and ledger.get("search_reuse_forbidden") is True,
        "csearch-v0001 tranche authority drifted",
    )
    _validate_recipe(campaign, source)
    gate = mapping(campaign.get("search_gate"), "csearch search gate")
    require(
        gate
        == {
            "pairs": 30,
            "minimum_positive_pairs": 20,
            "mean_reward_margin": ">0",
            "length_matched_mean_reward_margin": ">0",
            "lowering_forbidden": True,
        },
        "csearch search gate drifted",
    )
    budget = mapping(campaign.get("execution_budget"), "csearch execution budget")
    require(budget == {"maximum_training_runs": 2, "maximum_optimizer_steps": 60, "maximum_candidate_evaluations": 2}, "csearch execution budget drifted")

    requirement = mapping(control.get("strict_preunseal_requirement"), "csearch strict requirement")
    expected_receipt = run_root / "evidence/authority/strict-preunseal.json"
    require(
        requirement
        == {
            "required_before_first_optimizer": True,
            "require_search_unopened": True,
            "validator": producers["authority_validator"],
            "receipt_path": str(expected_receipt),
            "receipt_schema_name": "day23.csearch_v0001_strict_authority_preflight",
            "receipt_self_hash_field": "validation_sha256",
        },
        "csearch strict requirement drifted",
    )
    claim_path = Path(text(mapping(campaign.get("access_ledger"), "csearch access ledger").get("search_claim"), "search claim path")).resolve()
    require(claim_path == run_root / "evidence/search-selection/search-claim.json", "csearch search claim escaped run root")
    if require_search_unopened:
        require(not expected_receipt.exists(), "csearch strict receipt already exists")
        require(not claim_path.exists(), "csearch search was already opened")
        for spec in mapping(campaign.get("run_specs"), "csearch run specs").values():
            run = mapping(spec, "csearch run")
            require(not Path(str(run.get("success_receipt"))).exists() and not Path(str(run.get("failure_receipt"))).exists(), "optimizer receipt exists before strict preflight")

    return {
        "schema_name": "day23.csearch_v0001_strict_authority_preflight",
        "schema_version": 1,
        "status": "pass",
        "goal_id": builder.GOAL_ID,
        "version_id": builder.VERSION_ID,
        "campaign_path": str(campaign_path),
        "campaign_file_sha256": day23_gpu.file_sha256(campaign_path),
        "campaign_sha256": campaign_sha,
        "authority": {
            "charter": control.get("charter"),
            "source_terminal_event": control.get("source_terminal_event"),
        },
        "source_terminal": {
            "campaign_sha256": source["campaign_sha256"],
            "selection_sha256": source["selection_sha256"],
            "status": "closed_no_candidate",
            "eligible_count": 0,
        },
        "tranche1": {
            "fit_ordered_pair_ids_sha256": partition["datasets"]["fit"]["ordered_pair_ids_sha256"],
            "search_ordered_pair_ids_sha256": partition["datasets"]["search"]["ordered_pair_ids_sha256"],
            "fit_records": 124,
            "search_records": 30,
        },
        "recipe": {
            "learning_rates": list(builder.LEARNING_RATES),
            "candidate_checkpoint_step": builder.CANDIDATE_STEP,
            "max_steps": builder.MAX_STEPS,
            "runtime_profile": "world2_B8_GA2_global32",
            "minimum_free_memory_fraction": builder.MIN_FREE_FRACTION,
            "fresh_s1_only": True,
        },
        "search_gate": dict(gate),
        "strict_preunseal_requirement": dict(requirement),
        "search_unopened_at_preflight": require_search_unopened,
        "producer": dict(producers["authority_validator"]),
        "runtime_dependencies": dependencies,
    }


def write_exclusive(path: Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(value)
    require(field not in sealed, f"receipt already contains {field}")
    sealed[field] = day23_gpu.object_sha256(sealed)
    payload = (json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise CSearchAuthorityError(f"refusing to overwrite sealed receipt: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return sealed


def _self_test() -> None:
    value = {"schema_name": "self-test", "status": "pass"}
    sealed = dict(value)
    sealed["validation_sha256"] = day23_gpu.object_sha256(sealed)
    require(verify(sealed, "validation_sha256", "self-test") == sealed["validation_sha256"], "self-test hash failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-search-unopened", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {"status": "pass", "scope": "stdlib_self_test"}
            print(json.dumps(result, sort_keys=True))
            return 0
        require(args.campaign is not None, "--campaign is required")
        require(args.output is not None, "--output is required")
        result = validate(args.campaign, require_search_unopened=args.require_search_unopened)
        output = args.output.expanduser().resolve()
        required = Path(result["strict_preunseal_requirement"]["receipt_path"]).resolve()
        require(output == required, "strict receipt output path drifted from campaign")
        sealed = write_exclusive(output, result, "validation_sha256")
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "pass", "output": str(output), "validation_sha256": sealed["validation_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
