#!/usr/bin/env python3
"""Build the goal-0004 qual-v0001 full154 refit campaign.

The campaign contains exactly one two-GPU, fresh-S1 DPO trajectory.  It trains
for the frozen 30-step cosine horizon, retains checkpoint 20 as the only
qualification candidate, and records checkpoint 30 as non-candidate.  Dev is
bound by manifest and global lease only; this builder never opens dev or
heldout rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_day23_csearch_v0001 as csearch_builder
import build_day23_rsi_v0004 as v4
import day23_contract
import eval_day23_csearch_candidate as csearch_eval
import init_day23_qualification_goal as goal_init
import run_day23_qwen35_gpu_stage as day23_gpu


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
GOAL_ID = goal_init.GOAL_ID
VERSION_ID = goal_init.VERSION_ID
CAMPAIGN_ID = "day23-qwen35-dpo-qualification-v0001"
CAMPAIGN_SCHEMA = "day23.qualification_v0001_gpu_campaign"
STRICT_SCHEMA = "day23.qualification_v0001_strict_authority_preflight"
RUN_ID = "full154_refit_lr_4p3e_6_step_20"
RUN_ROLE = "unique_finalist_full154_refit"

LEARNING_RATE = 4.3e-6
REFIT_SEED = 20260820
MAX_STEPS = 30
CANDIDATE_STEP = 20
WORLD_SIZE = 2
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
GLOBAL_BATCH = 32
MIN_FREE_FRACTION = 0.20
CHECKPOINT_CAPTURE = {
    "candidate_checkpoint_steps": [CANDIDATE_STEP],
    "expected_retained_checkpoint_steps": [CANDIDATE_STEP, MAX_STEPS],
    "retained_noncandidate_checkpoint_steps": [MAX_STEPS],
    "retained_noncandidate_checkpoints_candidate_eligible": False,
}

GOAL3_REL = Path("rsi-control/charters") / goal_init.SOURCE_GOAL_ID
GOAL4_REL = Path("rsi-control/charters") / GOAL_ID
GOAL3_EVENT3_REL = GOAL3_REL / f"events/{goal_init.SOURCE_EVENT3_NAME}.json"
GOAL4_CHARTER_REL = GOAL4_REL / "charter.json"
GOAL4_EVENT2_REL = GOAL4_REL / f"events/{goal_init.EVENT2_NAME}.json"


class QualificationBuildError(RuntimeError):
    """A frozen source, recipe, runtime, or data authority failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationBuildError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationBuildError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def identity(
    path: Path,
    *,
    self_field: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if self_field is not None:
        result["content_sha256"] = verify(load(resolved, str(resolved)), self_field, str(resolved))
    return result


def resolve_bootcamp(bootcamp: Path, value: Any, label: str) -> Path:
    raw = Path(str(value))
    path = raw if raw.is_absolute() else bootcamp / raw
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    return resolved


def validate_dataset(path: Path, entry: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"{label} file hash drifted")
    if "bytes" in entry:
        require(path.stat().st_size == entry.get("bytes"), f"{label} byte count drifted")
    rows = day23_contract.load_jsonl(path)
    require(len(rows) == entry.get("records"), f"{label} record count drifted")
    pair_ids: list[str] = []
    row_hashes: list[str] = []
    for row in rows:
        pair_id = str(row.get("pair_id"))
        try:
            day23_contract.verify_self_hash(row, "row_sha256", f"{label} row {pair_id}")
        except day23_contract.Day23ContractError as error:
            raise QualificationBuildError(str(error)) from error
        pair_ids.append(pair_id)
        row_hashes.append(str(row["row_sha256"]))
    require(len(pair_ids) == len(set(pair_ids)), f"{label} pair IDs are not unique")
    require(
        day23_contract.object_sha256(pair_ids) == entry.get("ordered_pair_ids_sha256")
        and day23_contract.object_sha256(row_hashes) == entry.get("ordered_row_hashes_sha256"),
        f"{label} ordered identity drifted",
    )
    return rows


def _load_bound(
    bootcamp: Path, relative: Path, self_field: str, label: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = (bootcamp / relative).resolve(strict=True)
    require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    value = load(path, label)
    verify(value, self_field, label)
    return value, path, identity(path, self_field=self_field, relative_to=bootcamp)


def _source_state(campaign_arg: Path, selection_arg: Path) -> dict[str, Any]:
    campaign_path = campaign_arg.expanduser().resolve(strict=True)
    campaign = load(campaign_path, "source csearch campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "source csearch campaign")
    require(
        campaign.get("schema_name") == "day23.csearch_v0001_gpu_campaign"
        and campaign.get("goal_id") == goal_init.SOURCE_GOAL_ID
        and campaign.get("version_id") == goal_init.SOURCE_VERSION_ID
        and campaign_sha == goal_init.SOURCE_CAMPAIGN_SHA256,
        "source csearch campaign drifted",
    )
    source_root = Path(str(campaign.get("remote_run_root"))).resolve(strict=True)
    bootcamp = Path(str(campaign.get("bootcamp_root"))).resolve(strict=True)
    require(campaign_path == source_root / "binding/gpu-campaign.json", "source campaign is not canonical")

    selection_path = selection_arg.expanduser().resolve(strict=True)
    selection = load(selection_path, "source search selection")
    selection_sha = verify(selection, "selection_sha256", "source search selection")
    require(
        selection_path == source_root / "evidence/selection/search-selection.json"
        and selection_sha == goal_init.SOURCE_SELECTION_SHA256
        and selection.get("campaign_sha256") == campaign_sha
        and selection.get("status") == "selected"
        and selection.get("selected_candidate") == goal_init.SELECTED_CANDIDATE
        and selection.get("selected_run_id") == goal_init.SELECTED_RUN_ID
        and selection.get("selected_checkpoint_step") == CANDIDATE_STEP
        and float(selection.get("selected_learning_rate")) == LEARNING_RATE,
        "source selection drifted",
    )
    context = csearch_eval.validate_campaign(campaign_path)
    require(
        csearch_eval.validate_search_selection(context, selection_path) == selection,
        "source selection failed exact recomputation",
    )

    charter, charter_path, charter_identity = _load_bound(
        bootcamp, GOAL4_CHARTER_REL, "charter_sha256", "goal-0004 charter"
    )
    event2, event2_path, event2_identity = _load_bound(
        bootcamp, GOAL4_EVENT2_REL, "event_sha256", "goal-0004 source candidate event"
    )
    closeout, closeout_path, closeout_identity = _load_bound(
        bootcamp, GOAL3_EVENT3_REL, "event_sha256", "goal-0003 closeout event"
    )
    require(
        charter.get("schema_name") == "day23.qualification_goal_charter"
        and charter.get("goal_id") == GOAL_ID
        and charter.get("status") == "frozen",
        "goal-0004 charter identity drifted",
    )
    source_candidate = mapping(charter.get("source_candidate"), "goal-0004 source candidate")
    require(
        source_candidate.get("source_campaign")
        == identity(campaign_path, self_field="campaign_sha256")
        and source_candidate.get("source_selection")
        == identity(selection_path, self_field="selection_sha256")
        and source_candidate.get("source_closeout_event") == closeout_identity
        and source_candidate.get("selected_candidate") == goal_init.SELECTED_CANDIDATE
        and source_candidate.get("selected_checkpoint_step") == CANDIDATE_STEP
        and float(source_candidate.get("selected_learning_rate")) == LEARNING_RATE
        and source_candidate.get("search_checkpoint_is_initialization") is False,
        "goal-0004 source candidate binding drifted",
    )
    source_charter_entry = mapping(
        source_candidate.get("source_charter"), "goal-0004 source csearch charter"
    )
    source_charter_path = resolve_bootcamp(
        bootcamp, source_charter_entry.get("path"), "source csearch charter path"
    )
    require(
        day23_gpu.file_sha256(source_charter_path)
        == source_charter_entry.get("file_sha256")
        and source_charter_path.stat().st_size == source_charter_entry.get("bytes"),
        "source csearch charter file identity drifted",
    )
    source_charter = load(source_charter_path, "source csearch charter")
    require(
        verify(source_charter, "charter_sha256", "source csearch charter")
        == source_charter_entry.get("content_sha256"),
        "source csearch charter content identity drifted",
    )
    inherited_protected = mapping(
        source_charter.get("protected_data"), "source csearch protected data"
    )
    require(
        closeout.get("event_id") == goal_init.SOURCE_EVENT3_NAME
        and closeout.get("sequence") == 3
        and closeout.get("state_after") == "candidate_frozen_search_only"
        and event2.get("event_id") == goal_init.EVENT2_NAME
        and event2.get("sequence") == 2
        and event2.get("state_after") == "source_candidate_imported",
        "qualification control event chain drifted",
    )

    data_authority = mapping(charter.get("data_authority"), "goal-0004 data authority")
    full_entry = dict(mapping(data_authority.get("full_train"), "goal-0004 full154"))
    full_path = resolve_bootcamp(bootcamp, full_entry.get("path"), "full154 path")
    full_rows = validate_dataset(full_path, full_entry, "full154")
    require(len(full_rows) == 154, "qualification optimizer corpus is not full154")
    full_entry["path"] = str(full_path)
    full_entry["bytes"] = full_path.stat().st_size
    full_entry["role"] = "unique_finalist_full154_refit_optimizer_input"

    dev_authority = mapping(data_authority.get("dev"), "goal-0004 dev authority")
    dev_lease = dict(mapping(dev_authority.get("lease"), "goal-0004 dev lease"))
    dev_identity = dict(mapping(dev_lease.get("identity"), "goal-0004 dev identity"))
    require(
        dev_lease.get("maximum_global_claims") == 1
        and dev_lease.get("raw_content_access_before_claim") is False
        and dev_identity.get("records") == 17,
        "goal-0004 dev lease drifted",
    )
    claim_rel = Path(str(dev_lease.get("claim_path")))
    require(not claim_rel.is_absolute() and ".." not in claim_rel.parts, "dev claim path is not canonical")
    dev_claim = (bootcamp / "rsi-control" / claim_rel).resolve()
    require(not dev_claim.exists(), "global dev lease was already claimed")
    heldout = mapping(data_authority.get("heldout"), "goal-0004 heldout authority")
    require(
        heldout.get("maximum_evaluations") == 0
        and heldout.get("claim_authorized") is False
        and heldout.get("runtime_path_disclosed") is False,
        "goal-0004 heldout boundary drifted",
    )

    selected = [
        item
        for item in selection.get("candidates", [])
        if isinstance(item, Mapping) and item.get("candidate_id") == goal_init.SELECTED_CANDIDATE
    ]
    require(len(selected) == 1 and selected[0].get("eligible") is True, "selected search evidence missing")
    selected_evaluation = load(Path(str(selected[0]["path"])), "selected search evaluation")
    verify(selected_evaluation, "evaluation_sha256", "selected search evaluation")
    search_checkpoint = dict(mapping(selected_evaluation.get("checkpoint"), "selected search checkpoint"))

    return {
        "campaign": campaign,
        "campaign_path": campaign_path,
        "campaign_sha256": campaign_sha,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": selection_sha,
        "source_root": source_root,
        "bootcamp": bootcamp,
        "charter": charter,
        "charter_path": charter_path,
        "charter_identity": charter_identity,
        "event2": event2,
        "event2_path": event2_path,
        "event2_identity": event2_identity,
        "closeout": closeout,
        "closeout_path": closeout_path,
        "closeout_identity": closeout_identity,
        "full_train": full_entry,
        "full_path": full_path,
        "full_rows": full_rows,
        "dev_lease": dev_lease,
        "dev_identity": dev_identity,
        "dev_claim": dev_claim,
        "inherited_scope_id": inherited_protected.get("inherited_scope_id"),
        "inherited_ledger_root": inherited_protected.get("inherited_ledger_root"),
        "search_checkpoint": search_checkpoint,
        "selected_evaluation": selected_evaluation,
        "selected_evaluation_path": Path(str(selected[0]["path"])).resolve(strict=True),
    }


def _producer_paths() -> dict[str, Path]:
    return {
        "authority_validator": DAY23_DIR / "validate_day23_qualification_v0001.py",
        "builder": Path(__file__).resolve(),
        "evaluator": DAY23_DIR / "eval_day23_qualification_candidate.py",
        "runner": DAY23_DIR / "run_day23_qualification_candidate.py",
    }


def _runtime_dependency_paths() -> dict[str, Path]:
    return {
        "build_day23_csearch_v0001": DAY23_DIR / "build_day23_csearch_v0001.py",
        "build_day23_rsi_v0004": DAY23_DIR / "build_day23_rsi_v0004.py",
        "day20_target_encoding_v3": (
            BOOTCAMP_ROOT / "day-20-qwen35-balanced-lora-sft/day20_target_encoding_v3.py"
        ),
        "day23_contract": DAY23_DIR / "day23_contract.py",
        "day23_ms_swift_plugin": DAY23_DIR / "day23_ms_swift_plugin.py",
        "day23_rlhf_template": DAY23_DIR / "day23_rlhf_template.py",
        "eval_day23_csearch_candidate": DAY23_DIR / "eval_day23_csearch_candidate.py",
        "eval_day23_qwen35_preferences": DAY23_DIR / "eval_day23_qwen35_preferences.py",
        "init_day23_qualification_goal": DAY23_DIR / "init_day23_qualification_goal.py",
        "run_day23_csearch_candidate": DAY23_DIR / "run_day23_csearch_candidate.py",
        "run_day23_qwen35_gpu_stage": DAY23_DIR / "run_day23_qwen35_gpu_stage.py",
    }


def _config(source: Mapping[str, Any], run_root: Path) -> tuple[dict[str, Any], bytes, Path]:
    source_spec = mapping(
        mapping(source["campaign"].get("run_specs"), "source run specs").get(goal_init.SELECTED_RUN_ID),
        "selected source run spec",
    )
    source_config_entry = mapping(source_spec.get("executable_config"), "selected source config")
    source_config_path = Path(str(source_config_entry.get("path"))).resolve(strict=True)
    require(
        day23_gpu.file_sha256(source_config_path) == source_config_entry.get("file_sha256"),
        "selected source config drifted",
    )
    value = copy.deepcopy(load(source_config_path, "selected source config"))
    require(
        value.get("learning_rate") == LEARNING_RATE
        and value.get("max_steps") == MAX_STEPS
        and value.get("lr_scheduler_type") == "cosine"
        and value.get("warmup_ratio") == 0.1
        and value.get("per_device_train_batch_size") == PER_DEVICE_BATCH
        and value.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
        and value.get("save_steps") == CANDIDATE_STEP
        and value.get("save_total_limit") == 2,
        "selected source recipe template drifted",
    )
    value.update(
        {
            "dataset": [str(source["full_path"])],
            "learning_rate": LEARNING_RATE,
            "seed": REFIT_SEED,
            "data_seed": REFIT_SEED,
            "max_steps": MAX_STEPS,
            "save_steps": CANDIDATE_STEP,
            "save_total_limit": 2,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "output_dir": str(run_root / "outputs" / RUN_ID),
        }
    )
    for forbidden in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset"):
        require(forbidden not in value, f"forbidden fresh-start config key present: {forbidden}")
    lowered = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    require(
        "heldout" not in lowered
        and "coding-dpo-dev" not in lowered
        and str(source["search_checkpoint"].get("path", "")) not in lowered,
        "protected or source-search checkpoint path leaked into optimizer config",
    )
    payload = json_bytes(value)
    path = run_root / "binding/configs/full154-refit-lr4p3e-6-step20.json"
    return value, payload, path


def _normalize_runtime_parse(value: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    stages = mapping(normalized.get("stages"), "qualification runtime parse stages")
    require(set(stages) == {RUN_ID}, "qualification runtime parse stage registry drifted")
    stage = mapping(stages.get(RUN_ID), "qualification runtime parse stage")
    rebound = dict(stage)
    rebound["output_dir"] = config["output_dir"]
    normalized["stages"][RUN_ID] = rebound
    normalized["output_dir_binding"] = "temporary_parse_sandbox_rebound_to_frozen_executable_config"
    return normalized


def _prepared(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_state(args.source_campaign, args.source_selection)
    run_root = args.run_root.expanduser().resolve(strict=True)
    require(run_root.is_dir() and not run_root.is_symlink(), "qualification run root must be a real directory")
    if args.mode != "check":
        require(not any(run_root.iterdir()), "new qualification run root must be empty")
    config, config_payload, config_path = _config(source, run_root)
    config_identity = {
        "path": str(config_path),
        "file_sha256": hashlib.sha256(config_payload).hexdigest(),
        "bytes": len(config_payload),
        "keys": sorted(config),
    }
    source_runtime = mapping(source["campaign"].get("runtime_parse"), "source runtime parse")
    python_path = Path(str(mapping(source_runtime.get("python_executable"), "source python").get("path"))).resolve(strict=True)
    checkout = Path(str(source_runtime.get("ms_swift_checkout"))).resolve(strict=True)
    parsed = v4._parse_configs(
        {RUN_ID: config_payload},
        python_path,
        checkout,
        mapping(source_runtime.get("package_versions"), "source package versions"),
    )
    runtime_parse = _normalize_runtime_parse(parsed, config)

    producers = _producer_paths()
    dependencies = _runtime_dependency_paths()
    for name, path in {**producers, **dependencies}.items():
        require(path.resolve().is_file() and not path.resolve().is_symlink(), f"missing qualification code: {name}")
    return {
        "source": source,
        "run_root": run_root,
        "config": config,
        "config_payload": config_payload,
        "config_path": config_path,
        "config_identity": config_identity,
        "runtime_parse": runtime_parse,
        "producers": producers,
        "dependencies": dependencies,
    }


def _campaign(prepared: Mapping[str, Any]) -> dict[str, Any]:
    source = prepared["source"]
    run_root: Path = prepared["run_root"]
    producers = {
        name: identity(path) for name, path in sorted(prepared["producers"].items())
    }
    dependencies = {
        name: identity(path) for name, path in sorted(prepared["dependencies"].items())
    }
    success = run_root / f"evidence/{RUN_ID}/success-receipt.json"
    failure = run_root / f"evidence/{RUN_ID}/failure-receipt.json"
    dev_output = run_root / "evidence/dev/finalist-dev.json"
    dev_failure = run_root / "evidence/dev/dev-failure-after-claim.json"
    spec = {
        "role": RUN_ROLE,
        "authorized": True,
        "dataset_key": "full_train",
        "executable_config": prepared["config_identity"],
        "learning_rate": LEARNING_RATE,
        "seed": REFIT_SEED,
        "max_steps": MAX_STEPS,
        "cosine_schedule_horizon": MAX_STEPS,
        "checkpoint_steps": [CANDIDATE_STEP],
        "checkpoint_capture": dict(CHECKPOINT_CAPTURE),
        "output_dir": str(run_root / "outputs" / RUN_ID),
        "success_receipt": str(success),
        "failure_receipt": str(failure),
        "fresh_start_from_parent": True,
        "source_search_checkpoint_as_initialization_forbidden": True,
        "candidate_checkpoint_only": CANDIDATE_STEP,
    }
    source_search_campaign = identity(source["campaign_path"], self_field="campaign_sha256")
    source_search_selection = identity(source["selection_path"], self_field="selection_sha256")
    campaign: dict[str, Any] = {
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
            "source_candidate_event": source["event2_identity"],
            "source_search_closeout_event": source["closeout_identity"],
        },
        "control": {
            "charter": source["charter_identity"],
            "source_candidate_event": source["event2_identity"],
            "source_search_closeout_event": source["closeout_identity"],
            "source_search_campaign": source_search_campaign,
            "source_search_selection": source_search_selection,
            "strict_preunseal_requirement": {
                "required_before_optimizer": True,
                "require_dev_unopened": True,
                "require_heldout_unopened": True,
                "validator": producers["authority_validator"],
                "receipt_path": str(run_root / "evidence/authority/strict-preunseal.json"),
                "receipt_schema_name": STRICT_SCHEMA,
                "receipt_self_hash_field": "validation_sha256",
            },
        },
        "source_candidate": {
            "campaign": source_search_campaign,
            "selection": source_search_selection,
            "selected_evaluation": identity(
                source["selected_evaluation_path"], self_field="evaluation_sha256"
            ),
            "status": "candidate_frozen_search_only",
            "candidate_id": goal_init.SELECTED_CANDIDATE,
            "search_run_id": goal_init.SELECTED_RUN_ID,
            "selected_learning_rate": LEARNING_RATE,
            "selected_checkpoint_step": CANDIDATE_STEP,
            "search_checkpoint_evidence": source["search_checkpoint"],
            "search_checkpoint_as_initialization": False,
            "runner_up_fallback": False,
        },
        "remote_parent": copy.deepcopy(source["campaign"]["remote_parent"]),
        "parent": {
            "role": "S1",
            "policy": "fresh_lora_over_merged_s1",
            "model_argument": source["campaign"]["parent"]["model_argument"],
            "resume_from_source_or_search_checkpoint": False,
        },
        "producers": producers,
        "runtime_dependencies": dependencies,
        "fixed_recipe": {
            "learning_rate": LEARNING_RATE,
            "selected_candidate_checkpoint_step": CANDIDATE_STEP,
            "max_steps": MAX_STEPS,
            "cosine_schedule_horizon": MAX_STEPS,
            "refit_seed": REFIT_SEED,
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
            "source_search_checkpoint_as_initialization_forbidden": True,
        },
        "datasets": {
            "full_train": source["full_train"],
            "dev": {
                **source["dev_identity"],
                "role": "one_shot_finalist_qualification_not_hyperparameter_selection",
                "raw_content_access_before_claim": False,
            },
        },
        "run_specs": {RUN_ID: spec},
        "candidate_registry": [
            {
                "candidate_id": "qualified_refit_lr_4p3e_6_step_20",
                "refit_run_spec": RUN_ID,
                "source_search_candidate_id": goal_init.SELECTED_CANDIDATE,
                "learning_rate": LEARNING_RATE,
                "checkpoint_step": CANDIDATE_STEP,
            }
        ],
        "runtime_parse": prepared["runtime_parse"],
        "artifact_inventory": {
            str(prepared["config_path"].relative_to(run_root)): {
                "file_sha256": prepared["config_identity"]["file_sha256"],
                "bytes": prepared["config_identity"]["bytes"],
            }
        },
        "access_ledger": {
            "scope_id": source["inherited_scope_id"],
            "ledger_root": source["inherited_ledger_root"],
            "dev_claim": str(source["dev_claim"]),
            "dev_lease": source["dev_lease"],
            "dev_evaluation": str(dev_output),
            "dev_failure_after_claim": str(dev_failure),
            "heldout_claim_disclosed": False,
            "heldout_authorized_claims": 0,
        },
        "protocol": {
            "refit": {
                "maximum_executions": 1,
                "dataset_key": "full_train",
                "records": 154,
                "fresh_from_s1": True,
                "seed": REFIT_SEED,
                "learning_rate": LEARNING_RATE,
                "cosine_schedule_horizon": MAX_STEPS,
                "candidate_checkpoint_step": CANDIDATE_STEP,
                "search_checkpoint_initialization": "forbidden",
            },
            "dev": {
                "maximum_claims": 1,
                "maximum_evaluations": 1,
                "records": 17,
                "minimum_positive_pairs": 13,
                "mean_reward_margin": ">0",
                "length_matched_mean_reward_margin": ">0",
                "retry_after_claim": "forbidden",
                "runner_up_fallback": "forbidden",
                "success_state": "dev_qualified_guardrails_pending",
            },
            "heldout": {
                "maximum_claims": 0,
                "maximum_evaluations": 0,
                "runtime_path_disclosed": False,
                "authorized": False,
            },
        },
        "execution_budget": {
            "maximum_training_runs": 1,
            "maximum_optimizer_steps": MAX_STEPS,
            "maximum_candidate_evaluations": 1,
            "maximum_dev_claims": 1,
            "maximum_heldout_claims": 0,
        },
        "claim_boundary": {
            "optimizer_started": False,
            "dev_opened": False,
            "heldout_opened": False,
            "strict_refit_receipt_required_before_dev": True,
        },
    }
    campaign["campaign_sha256"] = day23_gpu.object_sha256(campaign)
    return campaign


def _check(prepared: Mapping[str, Any]) -> dict[str, Any]:
    run_root: Path = prepared["run_root"]
    campaign_path = run_root / "binding/gpu-campaign.json"
    actual = load(campaign_path, "qualification campaign")
    campaign_sha = verify(actual, "campaign_sha256", "qualification campaign")
    require(actual == _campaign(prepared), "qualification campaign no longer recomputes exactly")
    config_path: Path = prepared["config_path"]
    require(
        config_path.is_file()
        and not config_path.is_symlink()
        and day23_gpu.file_sha256(config_path) == prepared["config_identity"]["file_sha256"]
        and config_path.read_bytes() == prepared["config_payload"],
        "qualification executable config drifted",
    )
    return {
        "status": "pass",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign_sha,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepared(args)
    if args.mode == "preflight":
        return {
            "status": "preflight_pass",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "run_id": RUN_ID,
            "full_train_records": 154,
            "learning_rate": LEARNING_RATE,
            "refit_seed": REFIT_SEED,
            "max_steps": MAX_STEPS,
            "candidate_checkpoint_step": CANDIDATE_STEP,
            "world_size": WORLD_SIZE,
            "nominal_global_train_batch_size": GLOBAL_BATCH,
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
            "producer_file_sha256": {
                name: day23_gpu.file_sha256(path)
                for name, path in sorted(prepared["producers"].items())
            },
        }
    if args.mode == "check":
        return _check(prepared)
    v4._write_exclusive(prepared["config_path"], prepared["config_payload"])
    campaign = _campaign(prepared)
    campaign_path = prepared["run_root"] / "binding/gpu-campaign.json"
    v4._write_exclusive(campaign_path, json_bytes(campaign))
    return {
        "status": "gpu_execution_bound_optimizer_pending",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "run_id": RUN_ID,
        "dev_rows_opened": 0,
        "heldout_rows_opened": 0,
    }


def _self_test() -> None:
    require(WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION == GLOBAL_BATCH, "global batch drifted")
    require(MAX_STEPS == 30 and CANDIDATE_STEP == 20, "checkpoint horizon drifted")
    require(CHECKPOINT_CAPTURE["expected_retained_checkpoint_steps"] == [20, 30], "checkpoint inventory drifted")
    sample = {"dataset": ["/tmp/full154.jsonl"], "model": "/tmp/s1", "max_steps": 30}
    lowered = json.dumps(sample, sort_keys=True).lower()
    require("heldout" not in lowered and "coding-dpo-dev" not in lowered, "leakage self-test failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--source-selection", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="preflight")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {"status": "pass", "scope": "stdlib_self_test"}
        else:
            require(args.source_campaign is not None, "--source-campaign is required")
            require(args.source_selection is not None, "--source-selection is required")
            require(args.run_root is not None, "--run-root is required")
            result = execute(args)
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
