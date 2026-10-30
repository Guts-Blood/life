#!/usr/bin/env python3
"""Run one frozen Day 23 csearch-v0001 DPO trajectory on two GPUs.

This adapter reuses the audited RSI runtime observer, but owns an independent
csearch campaign and receipt contract.  It has no refit or protected-data
execution path.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_day23_qwen35_gpu_stage as day23_gpu
import run_day23_rsi_candidate as rsi_runtime


VERSION_ID = "csearch-v0001"
GOAL_ID = "goal-0003-day23-dpo-candidate-search"
CAMPAIGN_ID = "day23-qwen35-dpo-csearch-v0001"
CAMPAIGN_SCHEMA = "day23.csearch_v0001_gpu_campaign"
RECEIPT_SCHEMA = "day23.csearch_v0001_training_receipt"
STRICT_SCHEMA = "day23.csearch_v0001_strict_authority_preflight"
EXECUTE_TOKEN = "RUN_GPU_OPTIMIZER"
WORLD_SIZE = 2
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
GLOBAL_BATCH = 32
MIN_FREE_FRACTION = 0.20
SEARCH_SEED = 20260819
LEARNING_RATES = (4.3e-6, 4.5e-6)
MAX_STEPS = 30
CANDIDATE_STEP = 20
CHECKPOINT_CAPTURE = {
    "candidate_checkpoint_steps": [CANDIDATE_STEP],
    "expected_retained_checkpoint_steps": [CANDIDATE_STEP, MAX_STEPS],
    "retained_noncandidate_checkpoint_steps": [MAX_STEPS],
    "retained_noncandidate_checkpoints_candidate_eligible": False,
}
EXPECTED_LORA = {
    "tuner_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "freeze_llm": False,
    "freeze_vit": True,
    "freeze_aligner": True,
}
RUNTIME_DEPENDENCY_PATHS = {
    "build_day23_rsi_v0004": "day-23-dpo-theory-smoke/build_day23_rsi_v0004.py",
    "day20_target_encoding_v3": "day-20-qwen35-balanced-lora-sft/day20_target_encoding_v3.py",
    "day23_contract": "day-23-dpo-theory-smoke/day23_contract.py",
    "day23_ms_swift_plugin": "day-23-dpo-theory-smoke/day23_ms_swift_plugin.py",
    "day23_rlhf_template": "day-23-dpo-theory-smoke/day23_rlhf_template.py",
    "eval_day23_qwen35_preferences": "day-23-dpo-theory-smoke/eval_day23_qwen35_preferences.py",
    "eval_day23_rsi_candidate": "day-23-dpo-theory-smoke/eval_day23_rsi_candidate.py",
    "run_day23_qwen35_gpu_stage": "day-23-dpo-theory-smoke/run_day23_qwen35_gpu_stage.py",
    "run_day23_rsi_candidate": "day-23-dpo-theory-smoke/run_day23_rsi_candidate.py",
}

# The imported observer/profile helpers are deterministic runtime machinery.
# Registering this process-local profile lets them validate the new adapter
# without weakening or editing any frozen RSI source.
rsi_runtime.SUPPORTED_RUNTIME_PROFILES[VERSION_ID] = {
    "per_device_train_batch_size": PER_DEVICE_BATCH,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
    "min_free_memory_fraction": MIN_FREE_FRACTION,
}


class CSearchCandidateError(RuntimeError):
    """A frozen csearch campaign or live optimizer invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSearchCandidateError(message)


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
        raise CSearchCandidateError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _text(value.get(field), f"{label}.{field}")
    _require(
        len(expected) == 64 and day23_gpu.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def _contains_protected_path(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.casefold()
        protected = "heldout" in lowered or "coding-dpo-dev" in lowered
        path_like = (
            "/" in value
            or "\\" in value
            or lowered.endswith((".json", ".jsonl", ".parquet", ".arrow"))
        )
        return protected and path_like
    if isinstance(value, Mapping):
        return any(_contains_protected_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_protected_path(item) for item in value)
    return False


def _resolve_bootcamp_identity(
    entry: Any,
    bootcamp: Path,
    label: str,
    *,
    self_field: str,
) -> tuple[dict[str, Any], Path]:
    identity = _mapping(entry, label)
    raw = Path(_text(identity.get("path"), f"{label}.path"))
    path = (raw if raw.is_absolute() else bootcamp / raw).resolve(strict=True)
    _require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp root")
    _require(
        day23_gpu.file_sha256(path) == identity.get("file_sha256"),
        f"{label} file hash drifted",
    )
    if "bytes" in identity:
        _require(path.stat().st_size == identity.get("bytes"), f"{label} byte count drifted")
    value = _load_json(path, label)
    content_sha = _verify_self(value, self_field, label)
    if "content_sha256" in identity:
        _require(
            identity.get("content_sha256") == content_sha,
            f"{label} content hash drifted",
        )
    return value, path


def _producer(campaign: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], Path]:
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    entry = _mapping(
        _mapping(campaign.get("producers"), "campaign.producers").get(name),
        f"campaign.producers.{name}",
    )
    path = _absolute(entry.get("path"), f"campaign producer {name} path")
    _require(
        path.is_relative_to(bootcamp)
        and day23_gpu.file_sha256(path) == entry.get("file_sha256"),
        f"campaign producer drifted: {name}",
    )
    if "bytes" in entry:
        _require(path.stat().st_size == entry.get("bytes"), f"producer bytes drifted: {name}")
    return entry, path


def _validate_runtime_dependencies(campaign: Mapping[str, Any]) -> None:
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    dependencies = _mapping(
        campaign.get("runtime_dependencies"), "campaign.runtime_dependencies"
    )
    _require(
        set(dependencies) == set(RUNTIME_DEPENDENCY_PATHS),
        "runtime dependency key inventory drifted",
    )
    for name, relative in sorted(RUNTIME_DEPENDENCY_PATHS.items()):
        entry = _mapping(dependencies.get(name), f"runtime dependency {name}")
        _require(
            set(entry) == {"path", "file_sha256", "bytes"},
            f"runtime dependency field inventory drifted: {name}",
        )
        expected = (bootcamp / relative).resolve(strict=True)
        actual = _absolute(entry.get("path"), f"runtime dependency {name} path")
        _require(
            actual == expected
            and actual.is_file()
            and not actual.is_symlink()
            and day23_gpu.file_sha256(actual) == entry.get("file_sha256")
            and actual.stat().st_size == entry.get("bytes"),
            f"runtime dependency identity drifted: {name}",
        )


def _strict_campaign_identity(
    receipt: Mapping[str, Any], campaign: Mapping[str, Any], campaign_path: Path
) -> bool:
    expected_file = day23_gpu.file_sha256(campaign_path)
    nested = receipt.get("campaign")
    if isinstance(nested, Mapping):
        return (
            Path(str(nested.get("path", ""))).resolve() == campaign_path
            and nested.get("file_sha256") == expected_file
            and nested.get("campaign_sha256") == campaign.get("campaign_sha256")
        )
    return (
        Path(str(receipt.get("campaign_path", ""))).resolve() == campaign_path
        and receipt.get("campaign_file_sha256") == expected_file
        and receipt.get("campaign_sha256") == campaign.get("campaign_sha256")
    )


def _validate_control(
    campaign: Mapping[str, Any], campaign_path: Path, *, require_search_unopened: bool
) -> None:
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    run_root = _absolute(campaign.get("remote_run_root"), "campaign.remote_run_root")
    control = _mapping(campaign.get("control"), "campaign.control")
    charter, _ = _resolve_bootcamp_identity(
        control.get("charter"), bootcamp, "control charter", self_field="charter_sha256"
    )
    _require(charter.get("goal_id") == GOAL_ID, "csearch charter goal drifted")
    _resolve_bootcamp_identity(
        control.get("source_terminal_event"),
        bootcamp,
        "source terminal event",
        self_field="event_sha256",
    )

    requirement = _mapping(
        control.get("strict_preunseal_requirement"),
        "strict pre-unseal requirement",
    )
    validator_bound, validator_path = _producer(campaign, "authority_validator")
    validator_required = _mapping(requirement.get("validator"), "required authority validator")
    _require(
        validator_required == validator_bound,
        "strict requirement bound a different authority validator",
    )
    strict_path = _absolute(
        requirement.get("receipt_path"), "strict authority receipt"
    )
    _require(
        strict_path == run_root / "evidence/authority/strict-preunseal.json"
        and requirement.get("required_before_first_optimizer") is True
        and requirement.get("require_search_unopened") is True,
        "strict pre-unseal requirement drifted",
    )
    self_field = requirement.get("receipt_self_hash_field", "validation_sha256")
    _require(self_field == "validation_sha256", "strict receipt self-hash field drifted")
    strict = _load_json(strict_path, "strict authority receipt")
    _verify_self(strict, self_field, "strict authority receipt")
    expected_schema = requirement.get("receipt_schema_name", STRICT_SCHEMA)
    _require(
        expected_schema == STRICT_SCHEMA
        and strict.get("schema_name") == STRICT_SCHEMA
        and strict.get("schema_version") == 1
        and strict.get("status") == "pass"
        and strict.get("goal_id") == GOAL_ID
        and strict.get("version_id") == VERSION_ID
        and _strict_campaign_identity(strict, campaign, campaign_path),
        "strict authority receipt identity drifted",
    )
    strict_producer = _mapping(strict.get("producer"), "strict receipt producer")
    _require(
        Path(_text(strict_producer.get("path"), "strict producer path")).resolve()
        == validator_path
        and strict_producer.get("file_sha256") == validator_bound.get("file_sha256"),
        "strict authority receipt producer drifted",
    )
    if "search_unopened_at_preflight" in strict:
        _require(
            strict.get("search_unopened_at_preflight") is True,
            "strict receipt did not prove unopened search",
        )

    spec = importlib.util.spec_from_file_location(
        f"_day23_csearch_authority_{validator_bound['file_sha256'][:16]}",
        validator_path,
    )
    _require(spec is not None and spec.loader is not None, "cannot load authority validator")
    authority = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authority)
    _require(
        Path(authority.__file__).resolve(strict=True) == validator_path
        and callable(getattr(authority, "validate", None)),
        "loaded a different authority validator",
    )
    recomputed = authority.validate(
        campaign_path, require_search_unopened=require_search_unopened
    )
    _require(isinstance(recomputed, Mapping), "authority validator returned no document")
    sealed_projection = dict(strict)
    sealed_projection.pop(self_field, None)
    # The sealed receipt proves the preflight state.  Later training/evaluation
    # launches may see a legitimate search claim, so retain that historical
    # fact while exactly recomputing every other field.
    if "search_unopened_at_preflight" in sealed_projection:
        recomputed = dict(recomputed)
        recomputed["search_unopened_at_preflight"] = sealed_projection[
            "search_unopened_at_preflight"
        ]
    _require(recomputed == sealed_projection, "strict authority receipt no longer recomputes exactly")


def _dataset_path(campaign: Mapping[str, Any], entry: Mapping[str, Any], label: str) -> Path:
    raw = Path(_text(entry.get("path"), f"{label}.path"))
    if raw.is_absolute():
        return raw.resolve(strict=True)
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    path = (bootcamp / raw).resolve(strict=True)
    _require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp root")
    return path


def _config_identity(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(
        spec.get("executable_config", spec.get("config")),
        "run executable config",
    )


def _validate_grid(campaign: Mapping[str, Any]) -> None:
    fixed = _mapping(campaign.get("fixed_recipe"), "campaign.fixed_recipe")
    runtime = _mapping(fixed.get("runtime"), "fixed_recipe.runtime")
    _require(
        runtime.get("world_size") == WORLD_SIZE
        and runtime.get("per_device_train_batch_size") == PER_DEVICE_BATCH
        and runtime.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
        and runtime.get("nominal_global_train_batch_size") == GLOBAL_BATCH
        and float(runtime.get("min_free_memory_fraction")) == MIN_FREE_FRACTION,
        "fixed runtime recipe drifted",
    )
    levels = fixed.get("lever_levels", fixed.get("learning_rates"))
    _require(
        isinstance(levels, list)
        and sorted(float(item) for item in levels) == list(LEARNING_RATES),
        "fixed csearch learning-rate grid drifted",
    )
    _require(fixed.get("search_seed") == SEARCH_SEED, "fixed csearch seed drifted")

    datasets = _mapping(campaign.get("datasets"), "campaign.datasets")
    fit = _mapping(datasets.get("fit"), "campaign.datasets.fit")
    fit_path = _dataset_path(campaign, fit, "fit dataset")
    _require(
        day23_gpu.file_sha256(fit_path) == fit.get("file_sha256")
        and _integer(fit.get("records"), "fit records") > 0
        and int(fit["records"]) % WORLD_SIZE == 0,
        "fit dataset identity drifted",
    )

    runs = _mapping(campaign.get("run_specs"), "campaign.run_specs")
    _require(len(runs) == 2, "csearch must contain exactly two trajectories")
    observed_lrs: list[float] = []
    for run_id, raw_spec in sorted(runs.items()):
        spec = _mapping(raw_spec, f"run spec {run_id}")
        _require(
            spec.get("role") == "search_train"
            and spec.get("authorized") is True
            and spec.get("fresh_start_from_parent") is True
            and spec.get("dataset_key") == "fit"
            and spec.get("max_steps") == MAX_STEPS
            and spec.get("checkpoint_steps") == [CANDIDATE_STEP]
            and spec.get("checkpoint_capture") == CHECKPOINT_CAPTURE
            and spec.get("seed") == SEARCH_SEED,
            f"csearch run contract drifted: {run_id}",
        )
        lr = float(spec.get("learning_rate"))
        _require(math.isfinite(lr), f"non-finite learning rate: {run_id}")
        observed_lrs.append(lr)
        config_entry = _config_identity(spec)
        config_path = _absolute(config_entry.get("path"), f"{run_id} config path")
        _require(
            day23_gpu.file_sha256(config_path) == config_entry.get("file_sha256"),
            f"config hash drifted: {run_id}",
        )
        if "bytes" in config_entry:
            _require(config_path.stat().st_size == config_entry.get("bytes"), f"config bytes drifted: {run_id}")
        config = _load_json(config_path, f"{run_id} config")
        _require(not _contains_protected_path(config), f"protected-data path leaked into config: {run_id}")
        for key in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset"):
            _require(key not in config, f"forbidden fresh-start key present: {run_id}.{key}")
        for key, expected in EXPECTED_LORA.items():
            _require(config.get(key) == expected, f"LoRA recipe drifted: {run_id}.{key}")
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
            and config.get("train_dataloader_shuffle") is True,
            f"frozen DPO recipe drifted: {run_id}",
        )
        _require(
            config.get("learning_rate") == spec.get("learning_rate")
            and config.get("seed") == SEARCH_SEED
            and config.get("data_seed") == SEARCH_SEED
            and config.get("max_steps") == MAX_STEPS
            and config.get("save_strategy") == "steps"
            and config.get("save_steps") == CANDIDATE_STEP
            and config.get("save_total_limit") == 2
            and config.get("per_device_train_batch_size") == PER_DEVICE_BATCH
            and config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
            and config.get("dataset") == [str(fit_path)]
            and Path(_text(config.get("output_dir"), f"{run_id} output_dir")).resolve()
            == Path(_text(spec.get("output_dir"), f"{run_id} spec output_dir")).resolve(),
            f"csearch executable config drifted: {run_id}",
        )
    _require(sorted(observed_lrs) == list(LEARNING_RATES), "csearch LR levels drifted")

    registry = campaign.get("candidate_registry")
    _require(isinstance(registry, list) and len(registry) == 2, "candidate registry drifted")
    mappings = {
        (item.get("search_run_spec"), item.get("checkpoint_step"), float(item.get("learning_rate")))
        for item in registry
        if isinstance(item, Mapping) and isinstance(item.get("learning_rate"), (int, float))
    }
    expected_mappings = {
        (run_id, CANDIDATE_STEP, float(spec["learning_rate"]))
        for run_id, spec in runs.items()
    }
    _require(
        mappings == expected_mappings
        and len({_text(item.get("candidate_id"), "candidate ID") for item in registry}) == 2,
        "candidate registry mapping drifted",
    )


def validate_campaign_document(
    campaign_path: Path, *, require_search_unopened: bool = False
) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = _load_json(campaign_path, "GPU campaign")
    _require(
        campaign.get("schema_name") == CAMPAIGN_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("campaign_id") == CAMPAIGN_ID
        and campaign.get("goal_id") == GOAL_ID
        and campaign.get("version_id") == VERSION_ID,
        "csearch campaign identity drifted",
    )
    campaign_sha = _verify_self(campaign, "campaign_sha256", "GPU campaign")
    root = _absolute(campaign.get("remote_run_root"), "campaign.remote_run_root")
    _require(
        campaign_path == (root / "binding/gpu-campaign.json").resolve(strict=True)
        and _load_json(root / "binding/gpu-campaign.json", "canonical campaign") == campaign,
        "runner did not receive the canonical csearch campaign",
    )
    _require(not _contains_protected_path(campaign), "protected-data path leaked into campaign")
    for name in ("builder", "runner", "evaluator", "authority_validator"):
        _producer(campaign, name)
    _validate_runtime_dependencies(campaign)
    _validate_grid(campaign)
    _validate_control(
        campaign, campaign_path, require_search_unopened=require_search_unopened
    )
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
    }


def validate_campaign(campaign_path: Path, run_id: str) -> dict[str, Any]:
    context = validate_campaign_document(campaign_path, require_search_unopened=False)
    campaign = context["campaign"]
    access = _mapping(campaign.get("access_ledger"), "campaign.access_ledger")
    search_claim = _absolute(
        access.get("search_claim"), "campaign search claim", must_exist=False
    )
    _require(
        search_claim
        == context["run_root"] / "evidence/search-selection/search-claim.json"
        and not search_claim.exists(),
        "search must remain unclaimed before every csearch optimizer launch",
    )
    spec = _mapping(
        _mapping(campaign.get("run_specs"), "campaign.run_specs").get(run_id),
        f"campaign run {run_id}",
    )
    config_entry = _config_identity(spec)
    config_path = _absolute(config_entry.get("path"), "run config path")
    config = _load_json(config_path, "run config")
    fit = _mapping(_mapping(campaign.get("datasets"), "campaign.datasets").get("fit"), "fit dataset")
    fit_path = _dataset_path(campaign, fit, "fit dataset")
    output_dir = _absolute(spec.get("output_dir"), "run output_dir", must_exist=False)
    success = _absolute(spec.get("success_receipt"), "success receipt", must_exist=False)
    failure = _absolute(spec.get("failure_receipt"), "failure receipt", must_exist=False)
    _require(not output_dir.exists(), "fresh csearch output directory already exists")
    _require(not success.exists() and not failure.exists(), "run already has a terminal receipt")
    _require(success.parent == failure.parent, "success/failure receipt directories differ")
    success.parent.mkdir(parents=True, exist_ok=True)
    runner_bound, runner_path = _producer(campaign, "runner")
    _require(
        runner_path == Path(__file__).resolve(strict=True)
        and runner_bound.get("file_sha256") == day23_gpu.file_sha256(runner_path),
        "campaign bound a different csearch runner",
    )
    _require(
        Path(_text(config.get("model"), "config.model")).resolve()
        == context["model_path"],
        "config model is not the verified merged S1 parent",
    )
    runtime = day23_gpu.validate_runtime(campaign)
    context.update(
        {
            "run_id": run_id,
            "spec": spec,
            "role": "search_train",
            "config": config,
            "config_path": config_path,
            "config_file_sha256": config_entry.get("file_sha256"),
            "dataset_key": "fit",
            "dataset": fit,
            "dataset_path": fit_path,
            "records": int(fit["records"]),
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
    except day23_gpu.Day23GPUStageError as error:
        raise CSearchCandidateError(str(error)) from error


def _collect_checkpoints(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual: dict[int, Path] = {}
    for path in context["output_dir"].glob("checkpoint-*"):
        try:
            actual[int(path.name.removeprefix("checkpoint-"))] = path
        except ValueError:
            continue
    _require(
        set(actual) == {CANDIDATE_STEP, MAX_STEPS},
        "retained checkpoint inventory drifted",
    )
    return (
        [_checkpoint_manifest(actual[CANDIDATE_STEP], CANDIDATE_STEP)],
        [_checkpoint_manifest(actual[MAX_STEPS], MAX_STEPS)],
    )


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
        "run_id": context.get("run_id"),
        "run_role": "search_train",
        "started_at_utc": started_at,
        "campaign": {
            "path": str(context.get("campaign_path")),
            "file_sha256": context.get("campaign_file_sha256"),
            "campaign_sha256": context.get("campaign_sha256"),
        },
        "producer": {
            "path": str(runner),
            "file_sha256": day23_gpu.file_sha256(runner),
        },
        "config": {
            "path": str(context.get("config_path")),
            "file_sha256": context.get("config_file_sha256"),
        },
        "dataset": {
            "key": "fit",
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
    """Recompute every promotion-relevant csearch training receipt claim."""
    receipt_path = receipt_path.expanduser().resolve()
    receipt_sha = _verify_self(receipt, "receipt_sha256", "training receipt")
    _require(
        receipt.get("schema_name") == RECEIPT_SCHEMA
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "pass"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("goal_id") == GOAL_ID
        and receipt.get("version_id") == VERSION_ID
        and receipt.get("run_role") == "search_train",
        "training receipt identity drifted",
    )
    run_id = _text(receipt.get("run_id"), "training receipt run_id")
    spec = _mapping(_mapping(campaign.get("run_specs"), "campaign.run_specs").get(run_id), f"run {run_id}")
    _require(
        receipt_path == Path(_text(spec.get("success_receipt"), "success receipt path")).resolve()
        and not Path(_text(spec.get("failure_receipt"), "failure receipt path")).resolve().exists(),
        "training receipt terminal path drifted",
    )
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    _require(
        receipt.get("campaign")
        == {
            "path": str(campaign_path),
            "file_sha256": day23_gpu.file_sha256(campaign_path),
            "campaign_sha256": campaign.get("campaign_sha256"),
        },
        "training receipt campaign binding drifted",
    )
    producer = _mapping(receipt.get("producer"), "training producer")
    runner_bound, runner_path = _producer(campaign, "runner")
    _require(
        Path(_text(producer.get("path"), "training producer path")).resolve() == runner_path
        and producer.get("file_sha256") == runner_bound.get("file_sha256"),
        "training receipt producer drifted",
    )
    config_bound = _config_identity(spec)
    config_path = _absolute(config_bound.get("path"), "bound config path")
    _require(
        receipt.get("config")
        == {"path": str(config_path), "file_sha256": config_bound.get("file_sha256")}
        and day23_gpu.file_sha256(config_path) == config_bound.get("file_sha256"),
        "training receipt config binding drifted",
    )
    dataset = _mapping(_mapping(campaign.get("datasets"), "campaign.datasets").get("fit"), "fit dataset")
    dataset_path = _dataset_path(campaign, dataset, "fit dataset")
    _require(
        receipt.get("dataset")
        == {
            "key": "fit",
            "path": str(dataset_path),
            "file_sha256": dataset.get("file_sha256"),
            "records": dataset.get("records"),
        }
        and day23_gpu.file_sha256(dataset_path) == dataset.get("file_sha256"),
        "training receipt dataset binding drifted",
    )
    runtime = _mapping(receipt.get("runtime_identity"), "training runtime")
    runtime_bound = _mapping(campaign.get("runtime_parse"), "campaign.runtime_parse")
    python_bound = _mapping(runtime_bound.get("python_executable"), "bound Python")
    _require(
        runtime.get("python")
        == {
            "path": python_bound.get("path"),
            "file_sha256": python_bound.get("file_sha256"),
            "version": python_bound.get("version"),
        }
        and runtime.get("package_versions") == runtime_bound.get("package_versions"),
        "training receipt runtime drifted",
    )
    swift_runtime = _mapping(runtime.get("ms_swift_checkout"), "training ms-swift runtime")
    _require(
        swift_runtime.get("path") == runtime_bound.get("ms_swift_checkout")
        and swift_runtime.get("commit") == runtime_bound.get("ms_swift_commit")
        and swift_runtime.get("clean") is True,
        "training ms-swift runtime drifted",
    )
    python_sha = _text(python_bound.get("file_sha256"), "bound Python hash")
    top_process = day23_gpu._validated_process_identity(
        receipt.get("process_identity"), "training process", python_file_sha256=python_sha
    )
    claim = _mapping(receipt.get("claim_boundary"), "training claim boundary")
    for key in (
        "runtime_stage_passed",
        "fresh_parent_start",
        "fresh_lora_only",
        "reference_disable_adapter_observed",
        "frozen_tensor_versions_unchanged",
        "two_gpu_ddp_runtime_proven",
    ):
        _require(claim.get(key) is True, f"training receipt did not prove {key}")
    _require(
        claim.get("checkpoint_reload_pending_fresh_process") is True
        and claim.get("dev_consumed") is False
        and claim.get("heldout_consumed") is False,
        "training receipt claim boundary drifted",
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
        "training DDP topology drifted",
    )
    processes: list[tuple[Any, Any, Any]] = []
    lora_digests: set[str] = set()
    for rank, item in enumerate(ranks):
        process = day23_gpu._validated_process_identity(
            item.get("process_identity"),
            f"training rank {rank} process",
            python_file_sha256=python_sha,
        )
        processes.append((process["boot_id"], process["pid"], process["proc_start_ticks"]))
        topology = _mapping(item.get("topology"), f"rank {rank} topology")
        _require(
            topology.get("world_size") == WORLD_SIZE
            and topology.get("rank") == rank
            and topology.get("local_rank") == rank
            and topology.get("visible_device_count") == WORLD_SIZE,
            f"training rank {rank} topology drifted",
        )
        observed = rsi_runtime._validate_live_observations(
            campaign, spec, dataset, item.get("observations"), f"training rank {rank}"
        )
        lora_digests.add(observed["lora"]["final_digest"])
    _require(
        len(set(processes)) == WORLD_SIZE
        and processes[0]
        == (top_process["boot_id"], top_process["pid"], top_process["proc_start_ticks"])
        and len(lora_digests) == 1,
        "training rank/process evidence drifted",
    )
    _require(receipt.get("observations") == ranks[0].get("observations"), "rank-zero projection drifted")
    profile = rsi_runtime._realized_batch_profile(
        ranks, {"spec": spec, "records": dataset.get("records"), "campaign": campaign}
    )
    _require(
        distributed.get("realized_global_batch_profile") == profile,
        "realized batch profile drifted",
    )
    checkpoints = receipt.get("checkpoints")
    noncandidate_checkpoints = receipt.get("retained_noncandidate_checkpoints")
    _require(
        isinstance(checkpoints, list)
        and [item.get("global_step") for item in checkpoints] == [CANDIDATE_STEP]
        and isinstance(noncandidate_checkpoints, list)
        and [item.get("global_step") for item in noncandidate_checkpoints]
        == [MAX_STEPS]
        and receipt.get("checkpoint_capture") == CHECKPOINT_CAPTURE
        and receipt.get("retained_noncandidate_checkpoint_steps") == [MAX_STEPS]
        and receipt.get("retained_noncandidate_checkpoints_candidate_eligible") is False,
        "candidate checkpoint inventory drifted",
    )
    actual = _checkpoint_manifest(
        _absolute(checkpoints[0].get("path"), "checkpoint path"), CANDIDATE_STEP
    )
    output_dir = _absolute(spec.get("output_dir"), "training output directory")
    retained: dict[int, Path] = {}
    for path in output_dir.glob("checkpoint-*"):
        try:
            retained[int(path.name.removeprefix("checkpoint-"))] = path
        except ValueError:
            continue
    _require(
        set(retained) == {CANDIDATE_STEP, MAX_STEPS},
        "post-training retained checkpoint set drifted",
    )
    actual_noncandidate = _checkpoint_manifest(retained[MAX_STEPS], MAX_STEPS)
    _require(
        retained[CANDIDATE_STEP].resolve(strict=True)
        == _absolute(checkpoints[0].get("path"), "candidate checkpoint path")
        and actual == checkpoints[0]
        and receipt.get("checkpoint") == actual,
        "training checkpoint manifest drifted",
    )
    _require(
        noncandidate_checkpoints == [actual_noncandidate],
        "retained non-candidate checkpoint manifest drifted",
    )
    return {
        "receipt_sha256": receipt_sha,
        "run_id": run_id,
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
                "scientific_result_available": bool(
                    observations and observations.get("global_step", 0)
                ),
                "dev_consumed": False,
                "heldout_consumed": False,
            },
        }
    )
    with contextlib.suppress(FileExistsError, OSError, day23_gpu.Day23GPUStageError):
        day23_gpu.write_sealed_json(failure, value, "receipt_sha256")


def run_candidate(campaign_path: Path, run_id: str) -> tuple[Path | None, str | None]:
    started_at = day23_gpu.utc_now()
    started_monotonic = time.monotonic()
    context: dict[str, Any] = {
        "campaign_path": campaign_path.expanduser().absolute(),
        "run_id": run_id,
    }
    runtime: Mapping[str, Any] | None = None
    observer: rsi_runtime.CampaignObserver | None = None
    try:
        ddp = day23_gpu.initialize_ddp()
        context.update(validate_campaign(campaign_path, run_id))
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
        observer.snapshot_memory("before_csearch_candidate_rlhf_main")
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
        torch.distributed.barrier()
        if ddp["rank"] != 0:
            return None, None

        checkpoints, noncandidate_checkpoints = _collect_checkpoints(context)
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
                "checkpoints": checkpoints,
                "checkpoint": checkpoints[0],
                "checkpoint_capture": dict(CHECKPOINT_CAPTURE),
                "retained_noncandidate_checkpoint_steps": [MAX_STEPS],
                "retained_noncandidate_checkpoints": noncandidate_checkpoints,
                "retained_noncandidate_checkpoints_candidate_eligible": False,
                "claim_boundary": {
                    "runtime_stage_passed": True,
                    "fresh_parent_start": True,
                    "fresh_lora_only": True,
                    "reference_disable_adapter_observed": True,
                    "frozen_tensor_versions_unchanged": True,
                    "two_gpu_ddp_runtime_proven": True,
                    "checkpoint_reload_pending_fresh_process": True,
                    "dev_consumed": False,
                    "heldout_consumed": False,
                },
            }
        )
        sealed_for_validation = dict(receipt)
        sealed_for_validation["receipt_sha256"] = day23_gpu.object_sha256(receipt)
        validate_training_receipt_value(
            context["campaign"],
            context["campaign_path"],
            sealed_for_validation,
            context["success_receipt"],
        )
        sha = day23_gpu.write_sealed_json(
            context["success_receipt"], receipt, "receipt_sha256"
        )
        return context["success_receipt"], sha
    except BaseException as error:
        _seal_failure(context, started_at, started_monotonic, runtime, observer, error)
        raise


def _self_test() -> None:
    fake = {
        "spec": {"max_steps": 2},
        "records": 124,
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
        for step in range(2):
            rows.extend(
                {
                    "global_step_before": step,
                    "local_pair_batch_size": size,
                    "current_gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                }
                for size in (8, 8)
            )
        ranks.append({"rank": rank, "observations": {"gradient_accumulation": {"observations": rows}}})
    profile = rsi_runtime._realized_batch_profile(ranks, fake)
    _require(profile["per_step_global_pair_counts"] == [32, 32], "batch-profile self-test failed")
    _require(
        _contains_protected_path({"dataset": ["/tmp/coding-dpo-dev.jsonl"]}),
        "protected-path self-test failed",
    )
    _require(
        not _contains_protected_path({"dev_consumed": False, "heldout_consumed": False}),
        "claim-field self-test failed",
    )


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
            args.campaign is not None
            and args.run is not None
            and args.execute == EXECUTE_TOKEN,
            "optimizer execution requires --campaign, --run, and the execute token",
        )
        path, sha = run_candidate(args.campaign, args.run)
    except BaseException as error:
        print(
            json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    if path is not None:
        print(json.dumps({"status": "pass", "receipt": str(path), "receipt_sha256": sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
