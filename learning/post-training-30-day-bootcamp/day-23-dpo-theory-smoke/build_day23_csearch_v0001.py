#!/usr/bin/env python3
"""Build the goal-0003 csearch-v0001 GPU campaign without opening protected data.

The builder imports the sealed RSI-v0005 terminal result, deterministically
rotates tranche 1 from the 90 never-evaluated training pairs, and binds two
fresh-S1 DPO trajectories whose only scientific lever is base learning rate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_day23_rsi_v0004 as v4
import day23_contract as contract
import run_day23_qwen35_gpu_stage as day23_gpu


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
GOAL_ID = "goal-0003-day23-dpo-candidate-search"
VERSION_ID = "csearch-v0001"
CAMPAIGN_ID = "day23-qwen35-dpo-csearch-v0001"
CAMPAIGN_SCHEMA = "day23.csearch_v0001_gpu_campaign"
SOURCE_SCHEMA = "day23.rsi_v0005_gpu_campaign"
SOURCE_SELECTION_SCHEMA = "day23.rsi_v0005_search_selection"
GOAL_ROOT_REL = Path(f"rsi-control/charters/{GOAL_ID}")
CHARTER_REL = GOAL_ROOT_REL / "charter.json"
SOURCE_EVENT_REL = GOAL_ROOT_REL / "events/event-000002-source-terminal-imported.json"

LEARNING_RATES = (4.3e-6, 4.5e-6)
CANDIDATE_STEP = 20
MAX_STEPS = 30
SEARCH_SEED = 20260819
WORLD_SIZE = 2
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
GLOBAL_BATCH = 32
MIN_FREE_FRACTION = 0.20
ROTATION_SALT = f"{GOAL_ID}|"
MECHANISM_PAIR_IDS = (
    "mbpp:task:602:s1pair:37673b32b9385aea",
    "mbpp:task:604:s1pair:388a8daec7e97c5a",
    "mbpp:task:605:s1pair:9ae278fe596ed4d4",
    "mbpp:task:610:s1pair:9e57fb63f7544777",
)
RUN_IDS = ("search_lr_4p3e_6", "search_lr_4p5e_6")


class CSearchBuildError(RuntimeError):
    """A sealed source, deterministic tranche, or campaign invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSearchBuildError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CSearchBuildError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    _require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _identity(
    path: Path,
    *,
    content_field: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    value: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if content_field is not None:
        value["content_sha256"] = _verify(_load(resolved, str(resolved)), content_field, str(resolved))
    return value


def _resolve_bound(
    bootcamp: Path,
    relative: Path,
    self_field: str,
    label: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = (bootcamp / relative).resolve(strict=True)
    _require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    value = _load(path, label)
    identity = _identity(path, content_field=self_field, relative_to=bootcamp)
    return value, path, identity


def _load_rows(path: Path, label: str) -> list[dict[str, Any]]:
    path = path.resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        rows = contract.load_jsonl(path)
    except (OSError, contract.Day23ContractError) as error:
        raise CSearchBuildError(f"cannot validate {label}: {error}") from error
    pair_ids: list[str] = []
    for row in rows:
        pair_id = str(row.get("pair_id"))
        _require(pair_id and pair_id != "None", f"{label} pair_id missing")
        try:
            contract.verify_self_hash(row, "row_sha256", f"{label} row {pair_id}")
        except contract.Day23ContractError as error:
            raise CSearchBuildError(str(error)) from error
        pair_ids.append(pair_id)
    _require(len(pair_ids) == len(set(pair_ids)), f"{label} pair IDs are not unique")
    return rows


def _validate_dataset_entry(entry: Any, label: str) -> tuple[list[dict[str, Any]], Path]:
    bound = _mapping(entry, label)
    path = Path(str(bound.get("path"))).resolve(strict=True)
    _require(day23_gpu.file_sha256(path) == bound.get("file_sha256"), f"{label} file hash drifted")
    _require(path.stat().st_size == bound.get("bytes"), f"{label} byte count drifted")
    rows = _load_rows(path, label)
    ids = [str(row["pair_id"]) for row in rows]
    hashes = [str(row["row_sha256"]) for row in rows]
    _require(len(rows) == bound.get("records"), f"{label} record count drifted")
    _require(contract.object_sha256(ids) == bound.get("ordered_pair_ids_sha256"), f"{label} ID hash drifted")
    _require(contract.object_sha256(hashes) == bound.get("ordered_row_hashes_sha256"), f"{label} row hash drifted")
    return rows, path


def _source_state(source_campaign_path: Path, source_selection_path: Path) -> dict[str, Any]:
    campaign_path = source_campaign_path.expanduser().resolve(strict=True)
    campaign = _load(campaign_path, "source csearch terminal campaign")
    campaign_sha = _verify(campaign, "campaign_sha256", "source csearch terminal campaign")
    _require(
        campaign.get("schema_name") == SOURCE_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("version_id") == "rsi-v0005"
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending",
        "source RSI-v0005 campaign identity drifted",
    )
    source_root = Path(str(campaign.get("remote_run_root"))).resolve(strict=True)
    _require(campaign_path == source_root / "binding/gpu-campaign.json", "source campaign is not canonical")
    bootcamp = Path(str(campaign.get("bootcamp_root"))).resolve(strict=True)

    selection_path = source_selection_path.expanduser().resolve(strict=True)
    selection = _load(selection_path, "source csearch terminal selection")
    selection_sha = _verify(selection, "selection_sha256", "source csearch terminal selection")
    _require(
        selection_path == source_root / "evidence/selection/search-selection.json"
        and selection.get("schema_name") == SOURCE_SELECTION_SCHEMA
        and selection.get("schema_version") == 1
        and selection.get("campaign_sha256") == campaign_sha
        and selection.get("status") == "closed_no_candidate"
        and selection.get("eligible_count") == 0
        and selection.get("candidate_count") == 4
        and selection.get("selected_candidate") is None,
        "source RSI-v0005 terminal selection drifted",
    )

    datasets = _mapping(campaign.get("datasets"), "source datasets")
    full_rows, full_path = _validate_dataset_entry(datasets.get("full_train"), "source full154")
    v5_search_rows, v5_search_path = _validate_dataset_entry(datasets.get("search"), "source v0005 search30")
    prior = _mapping(campaign.get("supersedes"), "source v0004 predecessor")
    v4_campaign_path = Path(str(prior.get("campaign_path"))).resolve(strict=True)
    _require(day23_gpu.file_sha256(v4_campaign_path) == prior.get("campaign_file_sha256"), "v0004 campaign file drifted")
    v4_campaign = _load(v4_campaign_path, "source v0004 campaign")
    _require(_verify(v4_campaign, "campaign_sha256", "source v0004 campaign") == prior.get("campaign_sha256"), "v0004 campaign binding drifted")
    v4_search_rows, v4_search_path = _validate_dataset_entry(
        _mapping(v4_campaign.get("datasets"), "v0004 datasets").get("search"),
        "source v0004 search30",
    )
    _require(len(full_rows) == 154 and len(v4_search_rows) == len(v5_search_rows) == 30, "source split sizes drifted")

    charter, _, charter_identity = _resolve_bound(bootcamp, CHARTER_REL, "charter_sha256", "goal-0003 charter")
    event2, _, event2_identity = _resolve_bound(bootcamp, SOURCE_EVENT_REL, "event_sha256", "goal-0003 source import event")
    _require(
        charter.get("schema_name") == "day23.csearch_goal_charter"
        and charter.get("schema_version") == 1
        and charter.get("status") == "frozen"
        and charter.get("goal_id") == GOAL_ID,
        "goal-0003 charter identity drifted",
    )
    charter_budget = _mapping(charter.get("iteration_budget"), "goal-0003 iteration budget")
    charter_gate = _mapping(charter.get("candidate_gate"), "goal-0003 candidate gate")
    recipe_authority = _mapping(
        charter.get("scientific_recipe_authority"),
        "goal-0003 scientific recipe authority",
    )
    _require(
        charter_budget.get("allowed_version_ids") == [
            "csearch-v0001",
            "csearch-v0002",
            "csearch-v0003",
        ]
        and charter_budget.get("maximum_search_trajectories_per_version") == 2
        and charter_budget.get("maximum_search_candidates_per_version") == 4
        and charter_budget.get("exactly_one_scientific_lever_per_version") is True
        and charter_gate.get("split") == "search"
        and charter_gate.get("pairs") == 30
        and charter_gate.get("minimum_positive_pairs") == 20
        and charter_gate.get("mean_reward_margin") == ">0"
        and charter_gate.get("length_matched_mean_reward_margin") == ">0"
        and charter_gate.get("threshold_relaxation_forbidden") is True
        and recipe_authority.get("frozen_by_initializer") is False
        and recipe_authority.get("required_before_each_version_optimizer") is True,
        "goal-0003 charter authority drifted",
    )
    _require(
        event2.get("schema_name") == "day23.csearch_event"
        and event2.get("schema_version") == 1
        and event2.get("event_id") == "event-000002-source-terminal-imported"
        and event2.get("sequence") == 2
        and event2.get("event_type") == "source-terminal-imported"
        and event2.get("goal_id") == GOAL_ID
        and event2.get("version_id") is None
        and event2.get("state_before") == "charter_frozen"
        and event2.get("state_after") == "iteration_open"
        and isinstance(event2.get("authority_refs"), list)
        and len(event2["authority_refs"]) == 3
        and event2["authority_refs"][0] == charter_identity,
        "goal-0003 source import event drifted",
    )
    event_payload = _mapping(event2.get("payload"), "goal-0003 source import payload")
    _require(
        event_payload.get("source_goal_id") == "goal-0002-day23-dpo"
        and event_payload.get("source_terminal_state") == "terminal_no_go"
        and event_payload.get("source_campaign")
        == _identity(campaign_path, content_field="campaign_sha256")
        and event_payload.get("source_selection")
        == _identity(selection_path, content_field="selection_sha256")
        and event_payload.get("fresh_search_pair_count") == 90
        and event_payload.get("frozen_tranche_count") == 3
        and event_payload.get("next_version_id") == VERSION_ID
        and event_payload.get("scientific_recipe_pending") is True,
        "goal-0003 source import payload drifted",
    )
    return {
        "campaign": campaign,
        "campaign_path": campaign_path,
        "campaign_sha256": campaign_sha,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": selection_sha,
        "bootcamp": bootcamp,
        "full_rows": full_rows,
        "full_path": full_path,
        "v4_search_rows": v4_search_rows,
        "v4_search_path": v4_search_path,
        "v5_search_rows": v5_search_rows,
        "v5_search_path": v5_search_path,
        "charter": charter,
        "charter_identity": charter_identity,
        "event2": event2,
        "event2_identity": event2_identity,
    }


def rotate_tranches(
    full_rows: Sequence[Mapping[str, Any]],
    v4_search_ids: set[str],
    v5_search_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_ids = [str(row["pair_id"]) for row in full_rows]
    full_set = set(full_ids)
    excluded = set(MECHANISM_PAIR_IDS) | v4_search_ids | v5_search_ids
    _require(set(MECHANISM_PAIR_IDS).issubset(full_set), "historical mechanism pairs are absent")
    _require(v4_search_ids.isdisjoint(v5_search_ids), "v4/v5 search tranches overlap")
    _require(excluded.issubset(full_set) and len(excluded) == 64, "consumed search exclusion set drifted")
    ranked = sorted(
        full_set - excluded,
        key=lambda pair_id: (
            hashlib.sha256((ROTATION_SALT + pair_id).encode("utf-8")).hexdigest(),
            pair_id,
        ),
    )
    _require(len(ranked) == 90, "goal-0003 eligible population must contain 90 pairs")
    memberships = [ranked[index : index + 30] for index in range(0, 90, 30)]
    tranche1_ids = set(memberships[0])
    search_rows = [dict(row) for row in full_rows if str(row["pair_id"]) in tranche1_ids]
    fit_rows = [dict(row) for row in full_rows if str(row["pair_id"]) not in tranche1_ids]
    _require(len(search_rows) == 30 and len(fit_rows) == 124, "tranche1 fit/search sizes drifted")
    plan = {
        "authority": "goal_0003_never_evaluated_training_pair_ids_only",
        "selection_salt": ROTATION_SALT,
        "ranking": "sha256(selection_salt + pair_id), then pair_id",
        "emission_order": "frozen_full154_order",
        "excluded": {
            "historical_mechanism_pair_ids": list(MECHANISM_PAIR_IDS),
            "v0004_search_ordered_pair_ids_sha256": contract.object_sha256(
                [str(row["pair_id"]) for row in full_rows if str(row["pair_id"]) in v4_search_ids]
            ),
            "v0005_search_ordered_pair_ids_sha256": contract.object_sha256(
                [str(row["pair_id"]) for row in full_rows if str(row["pair_id"]) in v5_search_ids]
            ),
            "records": 64,
        },
        "eligible_records": 90,
        "tranches": [
            {
                "tranche": index + 1,
                "rank_start": index * 30,
                "rank_stop": (index + 1) * 30,
                "records": 30,
                "ranked_membership_sha256": contract.object_sha256(membership),
                "emitted_ordered_pair_ids_sha256": contract.object_sha256(
                    [pair_id for pair_id in full_ids if pair_id in set(membership)]
                ),
                "status": "active_search" if index == 0 else "frozen_unopened_reserve",
            }
            for index, membership in enumerate(memberships)
        ],
        "active_tranche": 1,
        "fit_records": 124,
        "search_records": 30,
    }
    return fit_rows, search_rows, plan


def _dataset_identity(path: Path, payload: bytes, rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "records": len(rows),
        "ordered_pair_ids_sha256": contract.object_sha256([str(row["pair_id"]) for row in rows]),
        "ordered_row_hashes_sha256": contract.object_sha256([str(row["row_sha256"]) for row in rows]),
        "role": role,
    }


def _lr_token(lr: float) -> str:
    if lr == 4.3e-6:
        return "4p3e_6"
    if lr == 4.5e-6:
        return "4p5e_6"
    raise CSearchBuildError(f"unsupported csearch LR: {lr}")


def _build_configs(
    source: Mapping[str, Any],
    run_root: Path,
    fit: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source_specs = _mapping(source["campaign"].get("run_specs"), "source run specs")
    template_entry = _mapping(source_specs.get("search_lr_4p1e_6"), "source search template spec")
    config_entry = _mapping(template_entry.get("executable_config"), "source search template config")
    template_path = Path(str(config_entry.get("path"))).resolve(strict=True)
    _require(day23_gpu.file_sha256(template_path) == config_entry.get("file_sha256"), "source config template drifted")
    template = _load(template_path, "source search config template")
    _require(
        template.get("beta") == 0.1
        and template.get("loss_type") == "sigmoid"
        and template.get("rlhf_type") == "dpo"
        and template.get("lr_scheduler_type") == "cosine"
        and template.get("max_steps") == MAX_STEPS
        and template.get("seed") == SEARCH_SEED
        and template.get("data_seed") == SEARCH_SEED
        and template.get("lora_rank") == 8
        and template.get("lora_alpha") == 16
        and template.get("lora_dropout") == 0.05
        and template.get("tuner_type") == "lora"
        and template.get("freeze_llm") is False
        and template.get("freeze_vit") is True
        and template.get("freeze_aligner") is True
        and isinstance(template.get("target_regex"), str)
        and bool(template["target_regex"])
        and template.get("gradient_checkpointing") is True
        and template.get("warmup_ratio") == 0.1
        and template.get("dataset_shuffle") is True
        and template.get("train_dataloader_shuffle") is True
        and template.get("save_strategy") == "steps",
        "source recipe template drifted",
    )
    configs: dict[str, bytes] = {}
    specs: dict[str, Any] = {}
    registry: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    for lr in LEARNING_RATES:
        token = _lr_token(lr)
        run_id = f"search_lr_{token}"
        config = copy.deepcopy(template)
        config.update(
            {
                "dataset": [str(fit["path"])],
                "learning_rate": lr,
                "seed": SEARCH_SEED,
                "data_seed": SEARCH_SEED,
                "max_steps": MAX_STEPS,
                "save_steps": CANDIDATE_STEP,
                "save_total_limit": 2,
                "per_device_train_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                "output_dir": str(run_root / "outputs" / run_id),
            }
        )
        for forbidden in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset"):
            _require(forbidden not in config, f"source template contains forbidden fresh-start key: {forbidden}")
        lowered = json.dumps(config, ensure_ascii=False, sort_keys=True).lower()
        _require("heldout" not in lowered and "coding-dpo-dev" not in lowered, "protected path leaked into config")
        payload = _json_bytes(config)
        config_path = run_root / f"binding/configs/search-lr{token.replace('_', '-')}.json"
        identity = {
            "path": str(config_path),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "keys": sorted(config),
        }
        candidate_id = f"candidate_lr_{token}_step_20"
        configs[run_id] = payload
        specs[run_id] = {
            "role": "search_train",
            "authorized": True,
            "dataset_key": "fit",
            "executable_config": identity,
            "learning_rate": lr,
            "seed": SEARCH_SEED,
            "max_steps": MAX_STEPS,
            "cosine_schedule_horizon": MAX_STEPS,
            "checkpoint_steps": [CANDIDATE_STEP],
            "checkpoint_capture": {
                "candidate_checkpoint_steps": [CANDIDATE_STEP],
                "expected_retained_checkpoint_steps": [CANDIDATE_STEP, MAX_STEPS],
                "retained_noncandidate_checkpoint_steps": [MAX_STEPS],
                "retained_noncandidate_checkpoints_candidate_eligible": False,
            },
            "output_dir": str(run_root / "outputs" / run_id),
            "success_receipt": str(run_root / f"evidence/{run_id}/success-receipt.json"),
            "failure_receipt": str(run_root / f"evidence/{run_id}/failure-receipt.json"),
            "fresh_start_from_parent": True,
            "candidate_checkpoint_only": CANDIDATE_STEP,
        }
        registry.append(
            {
                "candidate_id": candidate_id,
                "search_run_spec": run_id,
                "learning_rate": lr,
                "checkpoint_step": CANDIDATE_STEP,
            }
        )
        inventory[str(config_path.relative_to(run_root))] = {
            "file_sha256": identity["file_sha256"],
            "bytes": identity["bytes"],
        }
    _require(tuple(sorted(configs)) == tuple(sorted(RUN_IDS)), "csearch run registry drifted")
    return configs, dict(sorted(specs.items())), registry, dict(sorted(inventory.items()))


def _producer_paths() -> dict[str, Path]:
    return {
        "authority_validator": DAY23_DIR / "validate_day23_csearch_v0001.py",
        "builder": Path(__file__).resolve(),
        "evaluator": DAY23_DIR / "eval_day23_csearch_candidate.py",
        "runner": DAY23_DIR / "run_day23_csearch_candidate.py",
    }


def _runtime_dependency_paths() -> dict[str, Path]:
    """Return the complete local-code closure used after campaign sealing."""

    return {
        "build_day23_rsi_v0004": DAY23_DIR / "build_day23_rsi_v0004.py",
        "day20_target_encoding_v3": (
            BOOTCAMP_ROOT
            / "day-20-qwen35-balanced-lora-sft/day20_target_encoding_v3.py"
        ),
        "day23_contract": DAY23_DIR / "day23_contract.py",
        "day23_ms_swift_plugin": DAY23_DIR / "day23_ms_swift_plugin.py",
        "day23_rlhf_template": DAY23_DIR / "day23_rlhf_template.py",
        "eval_day23_qwen35_preferences": DAY23_DIR / "eval_day23_qwen35_preferences.py",
        "eval_day23_rsi_candidate": DAY23_DIR / "eval_day23_rsi_candidate.py",
        "run_day23_qwen35_gpu_stage": DAY23_DIR / "run_day23_qwen35_gpu_stage.py",
        "run_day23_rsi_candidate": DAY23_DIR / "run_day23_rsi_candidate.py",
    }


def _normalize_runtime_parse(
    value: Mapping[str, Any], configs: Mapping[str, bytes]
) -> dict[str, Any]:
    """Remove v4 parser temp-directory volatility without weakening its proof."""

    normalized = copy.deepcopy(dict(value))
    stages = _mapping(normalized.get("stages"), "csearch runtime parse stages")
    _require(set(stages) == set(configs), "csearch runtime parse stage registry drifted")
    for run_id, payload in sorted(configs.items()):
        stage = _mapping(stages.get(run_id), f"csearch runtime parse stage {run_id}")
        parsed_output = Path(str(stage.get("output_dir")))
        _require(
            parsed_output.is_absolute()
            and parsed_output.name == f"output-{run_id}",
            f"csearch parser sandbox output drifted: {run_id}",
        )
        config = _mapping(json.loads(payload), f"csearch executable config {run_id}")
        rebound = dict(stage)
        rebound["output_dir"] = str(config.get("output_dir"))
        normalized["stages"][run_id] = rebound
    normalized["output_dir_binding"] = (
        "temporary_parse_sandbox_rebound_to_frozen_executable_config"
    )
    return normalized


def _prepared(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_state(args.source_campaign, args.source_selection)
    run_root = args.run_root.expanduser().resolve(strict=True)
    _require(run_root.is_dir() and not run_root.is_symlink(), "csearch run root must be a real directory")
    if args.mode != "check":
        _require(not any(run_root.iterdir()), "new csearch run root must be empty")
    v4_ids = {str(row["pair_id"]) for row in source["v4_search_rows"]}
    v5_ids = {str(row["pair_id"]) for row in source["v5_search_rows"]}
    fit_rows, search_rows, partition = rotate_tranches(source["full_rows"], v4_ids, v5_ids)
    fit_payload = contract.jsonl_bytes(fit_rows)
    search_payload = contract.jsonl_bytes(search_rows)
    fit_path = run_root / "binding/data/day23-qwen35-dpo-csearch-v0001-fit.jsonl"
    search_path = run_root / "binding/data/day23-qwen35-dpo-csearch-v0001-search.jsonl"
    datasets = {
        "fit": _dataset_identity(fit_path, fit_payload, fit_rows, "search_optimizer_input"),
        "search": _dataset_identity(search_path, search_payload, search_rows, "tranche1_locked_ranking_only"),
    }
    configs, specs, registry, inventory = _build_configs(source, run_root, datasets["fit"])
    inventory.update(
        {
            str(fit_path.relative_to(run_root)): {"file_sha256": datasets["fit"]["file_sha256"], "bytes": len(fit_payload)},
            str(search_path.relative_to(run_root)): {"file_sha256": datasets["search"]["file_sha256"], "bytes": len(search_payload)},
        }
    )
    source_runtime = _mapping(source["campaign"].get("runtime_parse"), "source runtime parse")
    python_path = Path(str(_mapping(source_runtime.get("python_executable"), "source python").get("path"))).resolve(strict=True)
    checkout = Path(str(source_runtime.get("ms_swift_checkout"))).resolve(strict=True)
    runtime_parse = _normalize_runtime_parse(
        v4._parse_configs(
            configs,
            python_path,
            checkout,
            _mapping(source_runtime.get("package_versions"), "source packages"),
        ),
        configs,
    )
    producers = _producer_paths()
    dependencies = _runtime_dependency_paths()
    for name, path in {**producers, **dependencies}.items():
        _require(path.resolve().is_file() and not path.resolve().is_symlink(), f"missing csearch producer: {name}")

    ledger = _mapping(source["charter"].get("search_exposure_ledger"), "goal-0003 exposure ledger")
    tranches = ledger.get("tranches")
    _require(isinstance(tranches, list) and len(tranches) == 3, "goal-0003 tranche registry drifted")
    tranche_authority = dict(_mapping(tranches[0], "goal-0003 csearch-v0001 tranche"))
    _require(
        tranche_authority
        == {
            "version_id": VERSION_ID,
            "membership_rank_start_inclusive": 0,
            "membership_rank_end_exclusive": 30,
            "records": 30,
            "ordered_pair_ids_sha256": contract.object_sha256(
                [str(row["pair_id"]) for row in search_rows]
            ),
            "ordered_row_hashes_sha256": contract.object_sha256(
                [str(row["row_sha256"]) for row in search_rows]
            ),
        },
        "goal-0003 csearch-v0001 tranche no longer matches deterministic rotation",
    )
    _require(
        ledger.get("eligible_pair_count") == 90
        and ledger.get("selection_salt") == ROTATION_SALT
        and ledger.get("search_reuse_forbidden") is True,
        "goal-0003 exposure ledger drifted",
    )
    return {
        "source": source,
        "run_root": run_root,
        "fit_payload": fit_payload,
        "search_payload": search_payload,
        "datasets": datasets,
        "partition": partition,
        "configs": configs,
        "specs": specs,
        "registry": registry,
        "inventory": dict(sorted(inventory.items())),
        "runtime_parse": runtime_parse,
        "producers": producers,
        "runtime_dependencies": dependencies,
        "tranche_authority": tranche_authority,
    }


def _campaign(prepared: Mapping[str, Any]) -> dict[str, Any]:
    source = prepared["source"]
    run_root: Path = prepared["run_root"]
    producers = prepared["producers"]
    producer_ids = {name: _identity(path) for name, path in sorted(producers.items())}
    dependency_ids = {
        name: _identity(path)
        for name, path in sorted(prepared["runtime_dependencies"].items())
    }
    campaign = {
        "schema_name": CAMPAIGN_SCHEMA,
        "schema_version": 1,
        "status": "gpu_execution_bound_optimizer_pending",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "campaign_id": CAMPAIGN_ID,
        "bootcamp_root": str(source["bootcamp"]),
        "remote_run_root": str(run_root),
        "authority": {
            "charter": source["charter_identity"],
            "source_terminal_event": source["event2_identity"],
        },
        "control": {
            "charter": source["charter_identity"],
            "source_terminal_event": source["event2_identity"],
            "strict_preunseal_requirement": {
                "required_before_first_optimizer": True,
                "require_search_unopened": True,
                "validator": producer_ids["authority_validator"],
                "receipt_path": str(run_root / "evidence/authority/strict-preunseal.json"),
                "receipt_schema_name": "day23.csearch_v0001_strict_authority_preflight",
                "receipt_self_hash_field": "validation_sha256",
            },
        },
        "source_terminal": {
            "campaign": _identity(source["campaign_path"], content_field="campaign_sha256"),
            "selection": _identity(source["selection_path"], content_field="selection_sha256"),
            "status": "closed_no_candidate",
            "eligible_count": 0,
            "candidate_or_resume": False,
        },
        "source_training_corpus": _identity(source["full_path"]),
        "producers": producer_ids,
        "runtime_dependencies": dependency_ids,
        "remote_parent": copy.deepcopy(source["campaign"]["remote_parent"]),
        "parent": {
            "role": "S1",
            "policy": "fresh_lora_over_merged_s1",
            "model_argument": source["campaign"]["fixed_recipe"]["model"]["model_argument"],
            "resume_from_source_checkpoint": False,
        },
        "fixed_recipe": {
            "only_scientific_lever": "model.optimization.learning_rate.base",
            "lever_levels": list(LEARNING_RATES),
            "candidate_checkpoint_step": CANDIDATE_STEP,
            "max_steps": MAX_STEPS,
            "cosine_schedule_horizon": MAX_STEPS,
            "search_seed": SEARCH_SEED,
            "objective": {"rlhf_type": "dpo", "loss_type": "sigmoid", "beta": 0.1},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
            "runtime": {
                "world_size": WORLD_SIZE,
                "per_device_train_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                "nominal_global_train_batch_size": GLOBAL_BATCH,
                "min_free_memory_fraction": MIN_FREE_FRACTION,
            },
            "fresh_s1_only": True,
        },
        "data_partition": prepared["partition"],
        "tranche_authority": prepared["tranche_authority"],
        "datasets": prepared["datasets"],
        "run_specs": prepared["specs"],
        "candidate_registry": prepared["registry"],
        "runtime_parse": prepared["runtime_parse"],
        "artifact_inventory": prepared["inventory"],
        "access_ledger": {"search_claim": str(run_root / "evidence/search-selection/search-claim.json")},
        "search_gate": {
            "pairs": 30,
            "minimum_positive_pairs": 20,
            "mean_reward_margin": ">0",
            "length_matched_mean_reward_margin": ">0",
            "lowering_forbidden": True,
        },
        "execution_budget": {
            "maximum_training_runs": 2,
            "maximum_optimizer_steps": 60,
            "maximum_candidate_evaluations": 2,
        },
        "claim_boundary": {
            "optimizer_started": False,
            "search_opened": False,
            "both_training_receipts_required_before_search": True,
        },
    }
    campaign["campaign_sha256"] = day23_gpu.object_sha256(campaign)
    return campaign


def _check(prepared: Mapping[str, Any]) -> dict[str, Any]:
    run_root: Path = prepared["run_root"]
    campaign_path = run_root / "binding/gpu-campaign.json"
    campaign = _load(campaign_path, "csearch campaign")
    campaign_sha = _verify(campaign, "campaign_sha256", "csearch campaign")
    _require(campaign == _campaign(prepared), "csearch campaign no longer recomputes exactly")
    for relative, identity in prepared["inventory"].items():
        path = run_root / relative
        _require(
            path.is_file()
            and not path.is_symlink()
            and day23_gpu.file_sha256(path) == identity["file_sha256"]
            and path.stat().st_size == identity["bytes"],
            f"csearch artifact drifted: {relative}",
        )
    return {"status": "pass", "campaign": str(campaign_path), "campaign_sha256": campaign_sha}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepared(args)
    if args.mode == "preflight":
        return {
            "status": "preflight_pass",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "learning_rates": list(LEARNING_RATES),
            "candidate_checkpoint_step": CANDIDATE_STEP,
            "max_steps": MAX_STEPS,
            "fit_records": 124,
            "search_records": 30,
            "eligible_reserve_records": 60,
            "producer_file_sha256": {
                name: day23_gpu.file_sha256(path) for name, path in sorted(prepared["producers"].items())
            },
            "runtime_dependency_file_sha256": {
                name: day23_gpu.file_sha256(path)
                for name, path in sorted(prepared["runtime_dependencies"].items())
            },
        }
    if args.mode == "check":
        return _check(prepared)
    run_root: Path = prepared["run_root"]
    v4._write_exclusive(run_root / "binding/data/day23-qwen35-dpo-csearch-v0001-fit.jsonl", prepared["fit_payload"])
    v4._write_exclusive(run_root / "binding/data/day23-qwen35-dpo-csearch-v0001-search.jsonl", prepared["search_payload"])
    for run_id, payload in prepared["configs"].items():
        v4._write_exclusive(Path(prepared["specs"][run_id]["executable_config"]["path"]), payload)
    campaign = _campaign(prepared)
    campaign_path = run_root / "binding/gpu-campaign.json"
    v4._write_exclusive(campaign_path, _json_bytes(campaign))
    return {
        "status": "gpu_execution_bound_optimizer_pending",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
    }


def _self_test() -> None:
    ids = list(MECHANISM_PAIR_IDS) + [f"pair-{index:03d}" for index in range(150)]
    rows = [{"pair_id": pair_id} for pair_id in ids]
    v4_ids = set(ids[4:34])
    v5_ids = set(ids[34:64])
    fit, search, plan = rotate_tranches(rows, v4_ids, v5_ids)
    _require(len(fit) == 124 and len(search) == 30, "self-test tranche counts failed")
    _require(plan["eligible_records"] == 90 and len(plan["tranches"]) == 3, "self-test tranche plan failed")
    _require({row["pair_id"] for row in search}.isdisjoint(v4_ids | v5_ids | set(MECHANISM_PAIR_IDS)), "self-test leakage guard failed")
    configs = {
        run_id: _json_bytes({"output_dir": f"/frozen/{run_id}"})
        for run_id in RUN_IDS
    }
    def volatile(root: str) -> dict[str, Any]:
        return {
            "stages": {
                run_id: {"status": "pass", "output_dir": f"{root}/output-{run_id}"}
                for run_id in RUN_IDS
            }
        }
    _require(
        _normalize_runtime_parse(volatile("/tmp/first"), configs)
        == _normalize_runtime_parse(volatile("/tmp/second"), configs),
        "self-test runtime parse normalization remained volatile",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--source-selection", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="build")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {"status": "pass", "scope": "stdlib_self_test"}
        else:
            _require(args.source_campaign is not None, "--source-campaign is required")
            _require(args.source_selection is not None, "--source-selection is required")
            _require(args.run_root is not None, "--run-root is required")
            result = execute(args)
    except Exception as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
