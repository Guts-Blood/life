#!/usr/bin/env python3
"""Run the single frozen Day 23 full154 qualification refit on two GPUs.

This is an independent qualification adapter.  It reuses the hash-bound
csearch/RSI observer machinery, but it neither mutates nor resumes a search
checkpoint and it has no dev/heldout execution mode.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_day23_csearch_candidate as csearch_runtime


day23_gpu = csearch_runtime.day23_gpu
rsi_runtime = csearch_runtime.rsi_runtime

GOAL_ID = "goal-0004-day23-dpo-qualification"
VERSION_ID = "qual-v0001"
CAMPAIGN_ID = "day23-qwen35-dpo-qualification-v0001"
CAMPAIGN_SCHEMA = "day23.qualification_v0001_gpu_campaign"
STRICT_SCHEMA = "day23.qualification_v0001_strict_authority_preflight"
RECEIPT_SCHEMA = "day23.qualification_v0001_training_receipt"
RUN_ID = "full154_refit_lr_4p3e_6_step_20"
RUN_ROLE = "unique_finalist_full154_refit"
SOURCE_CANDIDATE_ID = "candidate_lr_4p3e_6_step_20"
SOURCE_SEARCH_RUN_ID = "search_lr_4p3e_6"
EXECUTE_TOKEN = "RUN_GPU_OPTIMIZER"

WORLD_SIZE = 2
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
GLOBAL_BATCH = 32
MIN_FREE_FRACTION = 0.20
LEARNING_RATE = 4.3e-6
SEED = 20260820
MAX_STEPS = 30
CANDIDATE_STEP = 20
FULL_TRAIN_RECORDS = 154
CHECKPOINT_CAPTURE = {
    "candidate_checkpoint_steps": [CANDIDATE_STEP],
    "expected_retained_checkpoint_steps": [CANDIDATE_STEP, MAX_STEPS],
    "retained_noncandidate_checkpoint_steps": [MAX_STEPS],
    "retained_noncandidate_checkpoints_candidate_eligible": False,
}
EXPECTED_LORA = dict(csearch_runtime.EXPECTED_LORA)

# The imported observer/profile code validates by semantic version.  Register
# only this process-local profile; no frozen source file is changed.
rsi_runtime.SUPPORTED_RUNTIME_PROFILES[VERSION_ID] = {
    "per_device_train_batch_size": PER_DEVICE_BATCH,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
    "min_free_memory_fraction": MIN_FREE_FRACTION,
}


class QualificationCandidateError(RuntimeError):
    """A frozen qualification invariant or live optimizer gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationCandidateError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and bool(value) and "\x00" not in value,
        f"{label} must be non-empty text",
    )
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return value


def _absolute(value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(_text(value, label)).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute")
    return path.resolve(strict=must_exist)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationCandidateError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _text(value.get(field), f"{label}.{field}")
    _require(
        len(expected) == 64 and day23_gpu.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def _producer(campaign: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], Path]:
    try:
        return csearch_runtime._producer(campaign, name)
    except BaseException as error:
        raise QualificationCandidateError(str(error)) from error


def _dataset_path(campaign: Mapping[str, Any], entry: Mapping[str, Any]) -> Path:
    try:
        return csearch_runtime._dataset_path(campaign, entry, "full154 dataset")
    except BaseException as error:
        raise QualificationCandidateError(str(error)) from error


def _config_identity(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(spec.get("executable_config", spec.get("config")), "run config")


def _contains_protected_path(value: Any) -> bool:
    return csearch_runtime._contains_protected_path(value)


def _validate_bound_dependencies(campaign: Mapping[str, Any]) -> None:
    """Validate the declared dependency closure, including frozen csearch."""
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    dependencies = _mapping(
        campaign.get("runtime_dependencies"), "campaign.runtime_dependencies"
    )
    required = {
        "run_day23_csearch_candidate": (
            bootcamp / "day-23-dpo-theory-smoke/run_day23_csearch_candidate.py"
        ).resolve(strict=True),
        "run_day23_qwen35_gpu_stage": Path(day23_gpu.__file__).resolve(strict=True),
    }
    _require(required.keys() <= dependencies.keys(), "runtime dependency closure is incomplete")
    for name, raw in dependencies.items():
        entry = _mapping(raw, f"runtime dependency {name}")
        path = _absolute(entry.get("path"), f"runtime dependency {name} path")
        _require(
            path.is_relative_to(bootcamp)
            and path.is_file()
            and not path.is_symlink()
            and day23_gpu.file_sha256(path) == entry.get("file_sha256"),
            f"runtime dependency drifted: {name}",
        )
        if "bytes" in entry:
            _require(path.stat().st_size == entry.get("bytes"), f"dependency bytes drifted: {name}")
        if name in required:
            _require(path == required[name], f"runtime dependency path drifted: {name}")


def _strict_campaign_identity(
    receipt: Mapping[str, Any], campaign: Mapping[str, Any], campaign_path: Path
) -> bool:
    nested = receipt.get("campaign")
    if isinstance(nested, Mapping):
        return (
            Path(str(nested.get("path", ""))).resolve() == campaign_path
            and nested.get("file_sha256") == day23_gpu.file_sha256(campaign_path)
            and nested.get("campaign_sha256") == campaign.get("campaign_sha256")
        )
    return (
        Path(str(receipt.get("campaign_path", ""))).resolve() == campaign_path
        and receipt.get("campaign_file_sha256") == day23_gpu.file_sha256(campaign_path)
        and receipt.get("campaign_sha256") == campaign.get("campaign_sha256")
    )


def _validate_control(
    campaign: Mapping[str, Any],
    campaign_path: Path,
    *,
    require_dev_unopened: bool,
) -> None:
    """Recompute the pre-unseal authority without opening protected rows."""
    run_root = _absolute(campaign.get("remote_run_root"), "campaign.remote_run_root")
    control = _mapping(campaign.get("control"), "campaign.control")
    requirement = _mapping(
        control.get("strict_preunseal_requirement"), "strict pre-unseal requirement"
    )
    validator_bound, validator_path = _producer(campaign, "authority_validator")
    _require(
        _mapping(requirement.get("validator"), "required validator") == validator_bound,
        "strict requirement bound a different validator",
    )
    strict_path = _absolute(requirement.get("receipt_path"), "strict receipt")
    _require(
        strict_path == run_root / "evidence/authority/strict-preunseal.json"
        and requirement.get("required_before_optimizer") is True
        and requirement.get("require_dev_unopened") is True
        and requirement.get("require_heldout_unopened") is True
        and requirement.get("receipt_schema_name") == STRICT_SCHEMA,
        "strict pre-unseal path/ordering drifted",
    )
    field = requirement.get("receipt_self_hash_field", "validation_sha256")
    _require(field == "validation_sha256", "strict receipt self-hash field drifted")
    strict = _load_json(strict_path, "strict authority receipt")
    _verify_self(strict, field, "strict authority receipt")
    _require(
        strict.get("schema_name") == STRICT_SCHEMA
        and strict.get("schema_version") == 1
        and strict.get("status") == "pass"
        and strict.get("goal_id") == GOAL_ID
        and strict.get("version_id") == VERSION_ID
        and _strict_campaign_identity(strict, campaign, campaign_path),
        "strict authority receipt identity drifted",
    )
    producer = _mapping(strict.get("producer"), "strict receipt producer")
    _require(
        Path(_text(producer.get("path"), "strict producer path")).resolve() == validator_path
        and producer.get("file_sha256") == validator_bound.get("file_sha256"),
        "strict receipt producer drifted",
    )
    spec = importlib.util.spec_from_file_location(
        f"_day23_qualification_authority_{validator_bound['file_sha256'][:16]}",
        validator_path,
    )
    _require(spec is not None and spec.loader is not None, "cannot load authority validator")
    authority = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authority)
    validate = getattr(authority, "validate", None)
    _require(callable(validate), "authority validator has no validate function")
    parameters = inspect.signature(validate).parameters
    kwargs: dict[str, Any] = {}
    # The sealed receipt is a historical pre-optimizer fact.  The authority
    # validator's dev-unopened mode also requires optimizer outputs to be
    # absent, so post-training verifiers must recompute it in historical mode
    # and check the live dev lease separately below.
    if "require_protected_unopened" in parameters:
        kwargs["require_protected_unopened"] = False
    if "require_dev_unopened" in parameters:
        kwargs["require_dev_unopened"] = False
    if "require_heldout_unopened" in parameters:
        kwargs["require_heldout_unopened"] = True
    recomputed = validate(campaign_path, **kwargs)
    _require(isinstance(recomputed, Mapping), "authority validator returned no document")
    sealed = dict(strict)
    sealed.pop(field, None)
    _require(dict(recomputed) == sealed, "strict authority receipt no longer recomputes")


def _validate_access_boundary(
    campaign: Mapping[str, Any], *, require_dev_unopened: bool
) -> None:
    access = _mapping(campaign.get("access_ledger"), "campaign.access_ledger")
    _require(
        access.get("heldout_claim_disclosed") is False
        and access.get("heldout_authorized_claims") == 0,
        "heldout access boundary drifted",
    )
    if require_dev_unopened:
        claim = _absolute(access.get("dev_claim"), "global dev claim", must_exist=False)
        bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
        _require(
            claim.is_relative_to(bootcamp / "rsi-control/access-ledger")
            and not claim.exists()
            and not claim.is_symlink(),
            "global dev lease was already claimed",
        )


def _validate_recipe(campaign: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    fixed = _mapping(campaign.get("fixed_recipe"), "campaign.fixed_recipe")
    runtime = _mapping(fixed.get("runtime"), "fixed_recipe.runtime")
    objective = _mapping(fixed.get("objective"), "fixed_recipe.objective")
    lora = _mapping(fixed.get("lora"), "fixed_recipe.lora")
    _require(
        runtime.get("world_size") == WORLD_SIZE
        and runtime.get("per_device_train_batch_size") == PER_DEVICE_BATCH
        and runtime.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
        and runtime.get("nominal_global_train_batch_size") == GLOBAL_BATCH
        and float(runtime.get("min_free_memory_fraction")) == MIN_FREE_FRACTION,
        "fixed runtime recipe drifted",
    )
    _require(
        float(fixed.get("learning_rate")) == LEARNING_RATE
        and fixed.get("refit_seed") == SEED
        and fixed.get("max_steps") == MAX_STEPS
        and fixed.get("selected_candidate_checkpoint_step") == CANDIDATE_STEP
        and fixed.get("cosine_schedule_horizon") == MAX_STEPS
        and fixed.get("fresh_s1_only") is True,
        "fixed qualification recipe drifted",
    )
    _require(
        objective == {"rlhf_type": "dpo", "loss_type": "sigmoid", "beta": 0.1}
        and lora == {"rank": 8, "alpha": 16, "dropout": 0.05}
        and fixed.get("source_search_checkpoint_as_initialization_forbidden") is True,
        "fixed DPO/LoRA/source recipe drifted",
    )
    datasets = _mapping(campaign.get("datasets"), "campaign.datasets")
    full_train = _mapping(datasets.get("full_train"), "campaign.datasets.full_train")
    full_path = _dataset_path(campaign, full_train)
    _require(
        _integer(full_train.get("records"), "full154 records") == FULL_TRAIN_RECORDS
        and day23_gpu.file_sha256(full_path) == full_train.get("file_sha256"),
        "full154 dataset identity drifted",
    )
    _require(
        full_train.get("role") == "unique_finalist_full154_refit_optimizer_input",
        "full154 dataset role drifted",
    )
    runs = _mapping(campaign.get("run_specs"), "campaign.run_specs")
    _require(set(runs) == {RUN_ID}, "qualification must contain exactly one refit run")
    spec = _mapping(runs.get(RUN_ID), f"run spec {RUN_ID}")
    _require(
        spec.get("authorized") is True
        and spec.get("role") == RUN_ROLE
        and spec.get("dataset_key") == "full_train"
        and spec.get("fresh_start_from_parent") is True
        and spec.get("source_search_checkpoint_as_initialization_forbidden") is True
        and spec.get("learning_rate") == LEARNING_RATE
        and spec.get("seed") == SEED
        and spec.get("max_steps") == MAX_STEPS
        and spec.get("cosine_schedule_horizon") == MAX_STEPS
        and spec.get("checkpoint_steps") == [CANDIDATE_STEP]
        and spec.get("candidate_checkpoint_only") == CANDIDATE_STEP
        and spec.get("checkpoint_capture") == CHECKPOINT_CAPTURE,
        "qualification run spec drifted",
    )
    config_entry = _config_identity(spec)
    config_path = _absolute(config_entry.get("path"), "qualification config path")
    _require(
        day23_gpu.file_sha256(config_path) == config_entry.get("file_sha256"),
        "qualification config hash drifted",
    )
    if "bytes" in config_entry:
        _require(config_path.stat().st_size == config_entry.get("bytes"), "config bytes drifted")
    config = _load_json(config_path, "qualification config")
    forbidden = {
        key
        for key in config
        if key in {"adapters", "resume_from_checkpoint", "val_dataset"}
        or key.casefold().startswith("ref")
    }
    _require(not forbidden, f"forbidden fresh-start config keys present: {sorted(forbidden)}")
    _require(not _contains_protected_path(config), "protected-data path leaked into optimizer config")
    source_candidate = _mapping(campaign.get("source_candidate"), "campaign.source_candidate")
    search_checkpoint = _mapping(
        source_candidate.get("search_checkpoint_evidence"), "source search checkpoint"
    )
    serialized_config = json.dumps(config, ensure_ascii=False, sort_keys=True)
    _require(
        source_candidate.get("status") == "candidate_frozen_search_only"
        and source_candidate.get("candidate_id") == SOURCE_CANDIDATE_ID
        and source_candidate.get("search_run_id") == SOURCE_SEARCH_RUN_ID
        and source_candidate.get("selected_learning_rate") == LEARNING_RATE
        and source_candidate.get("selected_checkpoint_step") == CANDIDATE_STEP
        and source_candidate.get("runner_up_fallback") is False
        and source_candidate.get("search_checkpoint_as_initialization") is False
        and _text(search_checkpoint.get("path"), "source search checkpoint path")
        not in serialized_config,
        "source search checkpoint leaked into refit initialization",
    )
    for key, expected in EXPECTED_LORA.items():
        _require(config.get(key) == expected, f"LoRA recipe drifted: {key}")
    _require(
        isinstance(config.get("target_regex"), str)
        and bool(config["target_regex"])
        and config.get("rlhf_type") == "dpo"
        and config.get("loss_type") == "sigmoid"
        and float(config.get("beta")) == 0.1
        and config.get("gradient_checkpointing") is True
        and config.get("lr_scheduler_type") == "cosine"
        and float(config.get("warmup_ratio")) == 0.1
        and config.get("dataset_shuffle") is True
        and config.get("train_dataloader_shuffle") is True
        and config.get("eval_strategy") == "no",
        "frozen DPO config drifted",
    )
    _require(
        config.get("learning_rate") == LEARNING_RATE
        and config.get("seed") == SEED
        and config.get("data_seed") == SEED
        and config.get("max_steps") == MAX_STEPS
        and config.get("save_strategy") == "steps"
        and config.get("save_steps") == CANDIDATE_STEP
        and config.get("save_total_limit") == 2
        and config.get("per_device_train_batch_size") == PER_DEVICE_BATCH
        and config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
        and config.get("dataset") == [str(full_path)]
        and Path(_text(config.get("output_dir"), "config output_dir")).resolve()
        == Path(_text(spec.get("output_dir"), "spec output_dir")).resolve(),
        "qualification executable config drifted",
    )
    registry = campaign.get("candidate_registry")
    _require(
        isinstance(registry, list)
        and len(registry) == 1
        and isinstance(registry[0], Mapping)
        and registry[0].get("refit_run_spec") == RUN_ID
        and registry[0].get("learning_rate") == LEARNING_RATE
        and registry[0].get("checkpoint_step") == CANDIDATE_STEP,
        "qualification candidate registry drifted",
    )
    return spec, config_path


def validate_campaign_document(
    campaign_path: Path, *, require_dev_unopened: bool = False
) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = _load_json(campaign_path, "qualification campaign")
    _require(
        campaign.get("schema_name") == CAMPAIGN_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("campaign_id") == CAMPAIGN_ID
        and campaign.get("goal_id") == GOAL_ID
        and campaign.get("version_id") == VERSION_ID,
        "qualification campaign identity drifted",
    )
    campaign_sha = _verify_self(campaign, "campaign_sha256", "qualification campaign")
    root = _absolute(campaign.get("remote_run_root"), "campaign.remote_run_root")
    _require(
        campaign_path == (root / "binding/gpu-campaign.json").resolve(strict=True),
        "runner did not receive the canonical qualification campaign",
    )
    for name in ("builder", "runner", "evaluator", "authority_validator"):
        _producer(campaign, name)
    _validate_bound_dependencies(campaign)
    spec, config_path = _validate_recipe(campaign)
    _validate_control(
        campaign, campaign_path, require_dev_unopened=require_dev_unopened
    )
    _validate_access_boundary(campaign, require_dev_unopened=require_dev_unopened)
    model_path = day23_gpu._verify_remote_parent_payload(
        _mapping(campaign.get("remote_parent"), "campaign.remote_parent")
    )
    return {
        "campaign": campaign,
        "campaign_path": campaign_path,
        "campaign_file_sha256": day23_gpu.file_sha256(campaign_path),
        "campaign_sha256": campaign_sha,
        "run_root": root,
        "model_path": model_path,
        "spec": spec,
        "config_path": config_path,
    }


def validate_campaign(
    campaign_path: Path, *, require_dev_unopened: bool = False
) -> dict[str, Any]:
    """Public read-only verifier used by downstream qualification stages."""
    return validate_campaign_document(
        campaign_path, require_dev_unopened=require_dev_unopened
    )


def _prepare_run(campaign_path: Path, run_id: str) -> dict[str, Any]:
    _require(run_id == RUN_ID, "only the frozen full154 refit run is executable")
    context = validate_campaign_document(campaign_path, require_dev_unopened=True)
    campaign = context["campaign"]
    spec = context["spec"]
    config = _load_json(context["config_path"], "qualification config")
    dataset = _mapping(campaign["datasets"]["full_train"], "full154 dataset")
    dataset_path = _dataset_path(campaign, dataset)
    output_dir = _absolute(spec.get("output_dir"), "run output_dir", must_exist=False)
    success = _absolute(spec.get("success_receipt"), "success receipt", must_exist=False)
    failure = _absolute(spec.get("failure_receipt"), "failure receipt", must_exist=False)
    expected_output = context["run_root"] / "outputs" / RUN_ID
    expected_receipt_root = context["run_root"] / "evidence" / RUN_ID
    _require(
        output_dir == expected_output
        and success == expected_receipt_root / "success-receipt.json"
        and failure == expected_receipt_root / "failure-receipt.json",
        "qualification output/receipt path drifted",
    )
    access = _mapping(campaign.get("access_ledger"), "campaign.access_ledger")
    dev_claim = _absolute(access.get("dev_claim"), "global dev claim", must_exist=False)
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    _require(
        dev_claim.is_relative_to(bootcamp / "rsi-control/access-ledger")
        and not dev_claim.exists()
        and not dev_claim.is_symlink()
        and access.get("heldout_claim_disclosed") is False
        and access.get("heldout_authorized_claims") == 0,
        "protected lease was opened or qualification access boundary drifted",
    )
    _require(not output_dir.exists(), "fresh qualification output directory already exists")
    _require(not success.exists() and not failure.exists(), "refit already has a terminal receipt")
    _require(success.parent == failure.parent, "success/failure receipt directories differ")
    success.parent.mkdir(parents=True, exist_ok=True)
    runner_bound, runner_path = _producer(campaign, "runner")
    _require(
        runner_path == Path(__file__).resolve(strict=True)
        and runner_bound.get("file_sha256") == day23_gpu.file_sha256(runner_path),
        "campaign bound a different qualification runner",
    )
    _require(
        Path(_text(config.get("model"), "config.model")).resolve() == context["model_path"],
        "qualification config does not start from the verified promoted S1",
    )
    runtime = day23_gpu.validate_runtime(campaign)
    context.update(
        {
            "run_id": RUN_ID,
            "role": RUN_ROLE,
            "config": config,
            "config_file_sha256": _config_identity(spec).get("file_sha256"),
            "dataset_key": "full_train",
            "dataset": dataset,
            "dataset_path": dataset_path,
            "records": FULL_TRAIN_RECORDS,
            "output_dir": output_dir,
            "success_receipt": success,
            "failure_receipt": failure,
            "checkpoint_steps": [CANDIDATE_STEP],
            "runtime": runtime,
            "per_device_batch": PER_DEVICE_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "selection": None,
        }
    )
    return context


def _checkpoint_manifest(path: Path, step: int) -> dict[str, Any]:
    try:
        return day23_gpu.checkpoint_manifest(path, step)
    except BaseException as error:
        raise QualificationCandidateError(str(error)) from error


def _collect_checkpoints(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual: dict[int, Path] = {}
    for path in context["output_dir"].glob("checkpoint-*"):
        try:
            actual[int(path.name.removeprefix("checkpoint-"))] = path
        except ValueError:
            continue
    _require(set(actual) == {CANDIDATE_STEP, MAX_STEPS}, "retained checkpoint inventory drifted")
    return (
        [_checkpoint_manifest(actual[CANDIDATE_STEP], CANDIDATE_STEP)],
        [_checkpoint_manifest(actual[MAX_STEPS], MAX_STEPS)],
    )


def _memory_gate(ranks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [
        float(item["observations"]["memory"]["minimum_observed_device_free_fraction"])
        for item in ranks
    ]
    _require(
        len(observed) == WORLD_SIZE
        and all(math.isfinite(value) and value >= MIN_FREE_FRACTION for value in observed),
        "qualification memory gate failed",
    )
    return {
        "required_minimum_free_fraction": MIN_FREE_FRACTION,
        "observed_minimum_free_fraction_by_rank": observed,
        "minimum_observed_free_fraction": min(observed),
        "passed": True,
    }


def _base_receipt(
    context: Mapping[str, Any], started_at: str, runtime: Mapping[str, Any] | None
) -> dict[str, Any]:
    runner = Path(__file__).resolve(strict=True)
    return {
        "schema_name": RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "run_id": RUN_ID,
        "run_role": RUN_ROLE,
        "started_at_utc": started_at,
        "campaign": {
            "path": str(context.get("campaign_path")),
            "file_sha256": context.get("campaign_file_sha256"),
            "campaign_sha256": context.get("campaign_sha256"),
        },
        "producer": {"path": str(runner), "file_sha256": day23_gpu.file_sha256(runner)},
        "config": {
            "path": str(context.get("config_path")),
            "file_sha256": context.get("config_file_sha256"),
        },
        "dataset": {
            "key": "full_train",
            "path": str(context.get("dataset_path")),
            "file_sha256": context.get("dataset", {}).get("file_sha256"),
            "records": context.get("records"),
        },
        "process_identity": day23_gpu._process_identity(),
        "runtime_identity": runtime,
    }


def validate_training_receipt_value(
    campaign: Mapping[str, Any],
    campaign_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Strictly recompute every candidate-relevant full154 receipt claim."""
    receipt_path = receipt_path.expanduser().resolve()
    receipt_sha = _verify_self(receipt, "receipt_sha256", "qualification receipt")
    _require(
        receipt.get("schema_name") == RECEIPT_SCHEMA
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "pass"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("goal_id") == GOAL_ID
        and receipt.get("version_id") == VERSION_ID
        and receipt.get("run_id") == RUN_ID
        and receipt.get("run_role") == RUN_ROLE,
        "qualification receipt identity drifted",
    )
    spec = _mapping(campaign["run_specs"][RUN_ID], "qualification run spec")
    _require(
        receipt_path == Path(_text(spec.get("success_receipt"), "success receipt")).resolve()
        and not Path(_text(spec.get("failure_receipt"), "failure receipt")).resolve().exists(),
        "qualification terminal receipt path drifted",
    )
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    _require(
        receipt.get("campaign")
        == {
            "path": str(campaign_path),
            "file_sha256": day23_gpu.file_sha256(campaign_path),
            "campaign_sha256": campaign.get("campaign_sha256"),
        },
        "qualification receipt campaign binding drifted",
    )
    runner_bound, runner_path = _producer(campaign, "runner")
    producer = _mapping(receipt.get("producer"), "qualification receipt producer")
    _require(
        Path(_text(producer.get("path"), "receipt producer path")).resolve() == runner_path
        and producer.get("file_sha256") == runner_bound.get("file_sha256"),
        "qualification receipt producer drifted",
    )
    config_bound = _config_identity(spec)
    config_path = _absolute(config_bound.get("path"), "bound config path")
    _require(
        receipt.get("config") == {"path": str(config_path), "file_sha256": config_bound.get("file_sha256")}
        and day23_gpu.file_sha256(config_path) == config_bound.get("file_sha256"),
        "qualification receipt config binding drifted",
    )
    dataset = _mapping(campaign["datasets"]["full_train"], "bound full154 dataset")
    dataset_path = _dataset_path(campaign, dataset)
    _require(
        receipt.get("dataset")
        == {
            "key": "full_train",
            "path": str(dataset_path),
            "file_sha256": dataset.get("file_sha256"),
            "records": FULL_TRAIN_RECORDS,
        }
        and day23_gpu.file_sha256(dataset_path) == dataset.get("file_sha256"),
        "qualification receipt dataset binding drifted",
    )
    runtime = _mapping(receipt.get("runtime_identity"), "qualification runtime")
    bound_runtime = _mapping(campaign.get("runtime_parse"), "campaign.runtime_parse")
    python_bound = _mapping(bound_runtime.get("python_executable"), "bound Python")
    _require(
        runtime.get("python")
        == {
            "path": python_bound.get("path"),
            "file_sha256": python_bound.get("file_sha256"),
            "version": python_bound.get("version"),
        }
        and runtime.get("package_versions") == bound_runtime.get("package_versions"),
        "qualification runtime identity drifted",
    )
    swift_runtime = _mapping(runtime.get("ms_swift_checkout"), "qualification ms-swift runtime")
    _require(
        swift_runtime.get("path") == bound_runtime.get("ms_swift_checkout")
        and swift_runtime.get("commit") == bound_runtime.get("ms_swift_commit")
        and swift_runtime.get("clean") is True,
        "qualification ms-swift runtime drifted",
    )
    python_sha = _text(python_bound.get("file_sha256"), "bound Python hash")
    top_process = day23_gpu._validated_process_identity(
        receipt.get("process_identity"), "qualification process", python_file_sha256=python_sha
    )
    claim = _mapping(receipt.get("claim_boundary"), "qualification claim boundary")
    for key in (
        "runtime_stage_passed",
        "fresh_parent_start",
        "fresh_lora_only",
        "full154_unique_refit",
        "search_checkpoint_not_used_as_initialization",
        "reference_disable_adapter_observed",
        "frozen_tensor_versions_unchanged",
        "two_gpu_ddp_runtime_proven",
    ):
        _require(claim.get(key) is True, f"qualification receipt did not prove {key}")
    _require(
        claim.get("checkpoint_reload_pending_fresh_process") is True
        and claim.get("dev_consumed") is False
        and claim.get("heldout_consumed") is False,
        "qualification receipt claim boundary drifted",
    )
    distributed = _mapping(receipt.get("distributed_evidence"), "distributed evidence")
    ranks = distributed.get("ranks")
    _require(
        distributed.get("world_size") == WORLD_SIZE
        and distributed.get("configured_nominal_global_train_batch_size") == GLOBAL_BATCH
        and isinstance(ranks, list)
        and len(ranks) == WORLD_SIZE
        and [item.get("rank") for item in ranks] == [0, 1]
        and [item.get("local_rank") for item in ranks] == [0, 1],
        "qualification DDP topology drifted",
    )
    processes: list[tuple[Any, Any, Any]] = []
    lora_digests: set[str] = set()
    for rank, item in enumerate(ranks):
        process = day23_gpu._validated_process_identity(
            item.get("process_identity"), f"qualification rank {rank}", python_file_sha256=python_sha
        )
        processes.append((process["boot_id"], process["pid"], process["proc_start_ticks"]))
        topology = _mapping(item.get("topology"), f"rank {rank} topology")
        _require(
            topology.get("world_size") == WORLD_SIZE
            and topology.get("rank") == rank
            and topology.get("local_rank") == rank
            and topology.get("visible_device_count") == WORLD_SIZE,
            f"qualification rank {rank} topology drifted",
        )
        observed = rsi_runtime._validate_live_observations(
            campaign, spec, dataset, item.get("observations"), f"qualification rank {rank}"
        )
        lora_digests.add(observed["lora"]["final_digest"])
    _require(
        len(set(processes)) == WORLD_SIZE
        and processes[0] == (top_process["boot_id"], top_process["pid"], top_process["proc_start_ticks"])
        and len(lora_digests) == 1,
        "qualification rank/process evidence drifted",
    )
    _require(receipt.get("observations") == ranks[0].get("observations"), "rank-zero projection drifted")
    profile = rsi_runtime._realized_batch_profile(
        ranks, {"spec": spec, "records": FULL_TRAIN_RECORDS, "campaign": campaign}
    )
    _require(distributed.get("realized_global_batch_profile") == profile, "batch profile drifted")
    expected_memory = _memory_gate(ranks)
    _require(receipt.get("memory_gate") == expected_memory, "memory gate projection drifted")
    checkpoints = receipt.get("checkpoints")
    noncandidates = receipt.get("retained_noncandidate_checkpoints")
    _require(
        isinstance(checkpoints, list)
        and [item.get("global_step") for item in checkpoints] == [CANDIDATE_STEP]
        and isinstance(noncandidates, list)
        and [item.get("global_step") for item in noncandidates] == [MAX_STEPS]
        and receipt.get("checkpoint_capture") == CHECKPOINT_CAPTURE
        and receipt.get("retained_noncandidate_checkpoint_steps") == [MAX_STEPS]
        and receipt.get("retained_noncandidate_checkpoints_candidate_eligible") is False,
        "qualification checkpoint inventory drifted",
    )
    output_dir = _absolute(spec.get("output_dir"), "qualification output directory")
    retained = {
        int(path.name.removeprefix("checkpoint-")): path
        for path in output_dir.glob("checkpoint-*")
        if path.name.removeprefix("checkpoint-").isdigit()
    }
    _require(set(retained) == {CANDIDATE_STEP, MAX_STEPS}, "retained checkpoint set drifted")
    actual = _checkpoint_manifest(retained[CANDIDATE_STEP], CANDIDATE_STEP)
    actual_noncandidate = _checkpoint_manifest(retained[MAX_STEPS], MAX_STEPS)
    _require(
        checkpoints == [actual]
        and noncandidates == [actual_noncandidate]
        and receipt.get("checkpoint") == actual,
        "qualification checkpoint manifest drifted",
    )
    return {
        "receipt_sha256": receipt_sha,
        "run_id": RUN_ID,
        "spec": spec,
        "dataset": dataset,
        "checkpoints": [actual],
        "retained_noncandidate_checkpoints": [actual_noncandidate],
    }


def _seal_failure(
    context: Mapping[str, Any],
    started_at: str,
    started_monotonic: float,
    runtime: Mapping[str, Any] | None,
    observer: rsi_runtime.CampaignObserver | None,
    error: BaseException,
) -> None:
    failure = context.get("failure_receipt")
    if not isinstance(failure, Path):
        return
    observations: Any = None
    if observer is not None:
        with contextlib.suppress(BaseException):
            observations = observer.finalize(require_success=False)
    value = _base_receipt(context, started_at, runtime)
    value.update(
        {
            "status": "fail",
            "completed_at_utc": day23_gpu.utc_now(),
            "duration_seconds": time.monotonic() - started_monotonic,
            "error": {
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
                "traceback": traceback.format_exception(error)[-12:],
            },
            "observations": observations,
            "claim_boundary": {
                "scientific_result_available": bool(observations and observations.get("global_step", 0)),
                "dev_consumed": False,
                "heldout_consumed": False,
            },
        }
    )
    with contextlib.suppress(FileExistsError, OSError, BaseException):
        day23_gpu.write_sealed_json(failure, value, "receipt_sha256")


def run_candidate(campaign_path: Path, run_id: str) -> tuple[Path | None, str | None]:
    started_at = day23_gpu.utc_now()
    started_monotonic = time.monotonic()
    context: dict[str, Any] = {"campaign_path": campaign_path.expanduser().absolute(), "run_id": run_id}
    runtime: Mapping[str, Any] | None = None
    observer: rsi_runtime.CampaignObserver | None = None
    try:
        ddp = day23_gpu.initialize_ddp()
        context.update(_prepare_run(campaign_path, run_id))
        runtime = context["runtime"]
        import torch

        torch.distributed.barrier()
        from swift.cli.main import parse_yaml_args
        from swift.pipelines import rlhf_main

        argv = [str(context["config_path"])]
        parse_yaml_args(argv)
        topology = day23_gpu.validate_topology(context["config"])
        torch.cuda.reset_peak_memory_stats(ddp["local_rank"])
        observer = rsi_runtime.CampaignObserver(context)
        observer.snapshot_memory("before_qualification_full154_rlhf_main")
        with day23_gpu.install_observers(observer):
            result = rlhf_main(argv)
        observer.result_summary = day23_gpu.json_safe(result)
        observations = observer.finalize(require_success=True)
        local = {
            "rank": ddp["rank"],
            "local_rank": ddp["local_rank"],
            "process_identity": day23_gpu._process_identity(),
            "topology": topology,
            "observations": observations,
        }
        gathered: list[Any] = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(gathered, local)
        _require([item.get("rank") for item in gathered] == [0, 1], "DDP rank inventory drifted")
        _require(
            len({item["observations"]["lora"]["final_digest"] for item in gathered}) == 1,
            "DDP ranks ended with different LoRA tensors",
        )
        _require(
            {item["observations"]["global_step"] for item in gathered} == {MAX_STEPS},
            "DDP ranks ended at different steps",
        )
        profile = rsi_runtime._realized_batch_profile(gathered, context)
        memory_gate = _memory_gate(gathered)
        torch.distributed.barrier()
        if ddp["rank"] != 0:
            return None, None
        checkpoints, noncandidates = _collect_checkpoints(context)
        receipt = _base_receipt(context, started_at, runtime)
        receipt.update(
            {
                "status": "pass",
                "completed_at_utc": day23_gpu.utc_now(),
                "duration_seconds": time.monotonic() - started_monotonic,
                "observations": observations,
                "distributed_evidence": {
                    "world_size": WORLD_SIZE,
                    "configured_nominal_global_train_batch_size": GLOBAL_BATCH,
                    "realized_global_batch_profile": profile,
                    "ranks": gathered,
                },
                "memory_gate": memory_gate,
                "checkpoints": checkpoints,
                "checkpoint": checkpoints[0],
                "checkpoint_capture": dict(CHECKPOINT_CAPTURE),
                "retained_noncandidate_checkpoint_steps": [MAX_STEPS],
                "retained_noncandidate_checkpoints": noncandidates,
                "retained_noncandidate_checkpoints_candidate_eligible": False,
                "claim_boundary": {
                    "runtime_stage_passed": True,
                    "fresh_parent_start": True,
                    "fresh_lora_only": True,
                    "full154_unique_refit": True,
                    "search_checkpoint_not_used_as_initialization": True,
                    "reference_disable_adapter_observed": True,
                    "frozen_tensor_versions_unchanged": True,
                    "two_gpu_ddp_runtime_proven": True,
                    "checkpoint_reload_pending_fresh_process": True,
                    "dev_consumed": False,
                    "heldout_consumed": False,
                },
            }
        )
        sealed = dict(receipt)
        sealed["receipt_sha256"] = day23_gpu.object_sha256(receipt)
        validate_training_receipt_value(
            context["campaign"], context["campaign_path"], sealed, context["success_receipt"]
        )
        sha = day23_gpu.write_sealed_json(context["success_receipt"], receipt, "receipt_sha256")
        return context["success_receipt"], sha
    except BaseException as error:
        _seal_failure(context, started_at, started_monotonic, runtime, observer, error)
        raise


def _self_test() -> None:
    fake = {
        "spec": {"max_steps": 5},
        "records": FULL_TRAIN_RECORDS,
        "campaign": {
            "version_id": VERSION_ID,
            "fixed_recipe": {
                "runtime": {
                    "per_device_train_batch_size": PER_DEVICE_BATCH,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                    "min_free_memory_fraction": MIN_FREE_FRACTION,
                }
            },
        },
    }
    ranks = []
    for rank in (0, 1):
        rows = []
        for step in range(5):
            sizes = (8, 8) if step < 4 else (8, 5)
            rows.extend(
                {
                    "global_step_before": step,
                    "local_pair_batch_size": size,
                    "current_gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                }
                for size in sizes
            )
        ranks.append({"rank": rank, "observations": {"gradient_accumulation": {"observations": rows}}})
    profile = rsi_runtime._realized_batch_profile(ranks, fake)
    _require(
        profile["per_step_global_pair_counts"] == [32, 32, 32, 32, 26],
        "full154 batch profile self-test failed",
    )
    _require(_contains_protected_path({"dataset": ["/tmp/coding-dpo-dev.jsonl"]}), "protected path self-test failed")
    _require(not _contains_protected_path({"dev_consumed": False}), "claim field self-test failed")
    memory = _memory_gate(
        [
            {"observations": {"memory": {"minimum_observed_device_free_fraction": value}}}
            for value in (0.21, 0.25)
        ]
    )
    _require(memory["minimum_observed_free_fraction"] == 0.21, "memory gate self-test failed")
    try:
        _memory_gate(
            [
                {"observations": {"memory": {"minimum_observed_device_free_fraction": value}}}
                for value in (0.19, 0.25)
            ]
        )
    except QualificationCandidateError:
        pass
    else:
        raise QualificationCandidateError("sub-threshold memory self-test did not fail")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--execute", choices=[EXECUTE_TOKEN])
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"status": "pass", "scope": "stdlib_self_test"}, sort_keys=True))
        return 0
    try:
        _require(
            args.campaign is not None and args.run == RUN_ID and args.execute == EXECUTE_TOKEN,
            "optimizer execution requires the frozen campaign, run ID, and execute token",
        )
        path, sha = run_candidate(args.campaign, args.run)
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    if path is not None:
        print(json.dumps({"status": "pass", "receipt": str(path), "receipt_sha256": sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
