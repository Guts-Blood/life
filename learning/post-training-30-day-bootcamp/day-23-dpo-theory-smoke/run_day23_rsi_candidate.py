#!/usr/bin/env python3
"""Run one preregistered Day 23 RSI DPO trajectory under two-GPU DDP.

This is deliberately a thin campaign adapter over the already-audited Day 23
runtime observer.  It does not select candidates or read dev/heldout data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_day23_qwen35_gpu_stage as day23_gpu


RUN_ROLES = {"capacity_preflight", "search_train", "unique_finalist_refit_option"}
EXECUTE_TOKEN = "RUN_GPU_OPTIMIZER"
WORLD_SIZE = 2
SUPPORTED_RUNTIME_PROFILES = {
    "rsi-v0003": {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 1, "min_free_memory_fraction": 0.30},
    "rsi-v0004": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "min_free_memory_fraction": 0.20},
    "rsi-v0005": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "min_free_memory_fraction": 0.20},
}
EXPECTED_LORA_TENSORS = 496
EXPECTED_LORA_PARAMS = 16_232_448
REQUIRED_LOG_KEYS = {
    "loss",
    "grad_norm",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/margins",
}


class RSICandidateError(RuntimeError):
    """A frozen campaign or live training invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RSICandidateError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value) and "\x00" not in value, f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return value


def _version_token(version_id: Any) -> str:
    version = _text(version_id, "campaign.version_id")
    _require(version in SUPPORTED_RUNTIME_PROFILES, "unsupported RSI campaign version")
    return version.replace("-", "_")


def campaign_schema(version_id: Any) -> str:
    return f"day23.{_version_token(version_id)}_gpu_campaign"


def receipt_schema(version_id: Any) -> str:
    return f"day23.{_version_token(version_id)}_training_receipt"


def _runtime_profile(campaign: Mapping[str, Any]) -> tuple[int, int]:
    version = _text(campaign.get("version_id"), "campaign.version_id")
    expected = _mapping(SUPPORTED_RUNTIME_PROFILES.get(version), "supported runtime profile")
    runtime = _mapping(_mapping(campaign.get("fixed_recipe"), "campaign.fixed_recipe").get("runtime"), "campaign runtime")
    per_device = _integer(runtime.get("per_device_train_batch_size"), "campaign per-device batch")
    accumulation = _integer(runtime.get("gradient_accumulation_steps"), "campaign gradient accumulation")
    _require(
        per_device == expected["per_device_train_batch_size"]
        and accumulation == expected["gradient_accumulation_steps"]
        and float(runtime.get("min_free_memory_fraction")) == expected["min_free_memory_fraction"]
        and WORLD_SIZE * per_device * accumulation == 32,
        "campaign runtime profile drifted",
    )
    return per_device, accumulation


def _absolute_path(value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(_text(value, label)).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute")
    path = path.resolve(strict=must_exist)
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RSICandidateError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} is invalid")
    actual = day23_gpu.object_sha256(value, field)
    _require(actual == expected, f"{label}.{field} does not bind its contents")
    return expected


def _path_entry(value: Any, label: str) -> tuple[Path, str]:
    entry = _mapping(value, label)
    path = _absolute_path(entry.get("path"), f"{label}.path")
    expected = _text(entry.get("file_sha256"), f"{label}.file_sha256")
    _require(day23_gpu.file_sha256(path) == expected, f"{label} file hash drifted")
    return path, expected


def _contains_forbidden_data(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    return "heldout" in text or "coding-dpo-dev" in text


def _validate_charter_binding(campaign: Mapping[str, Any]) -> None:
    binding = _mapping(campaign.get("charter"), "campaign.charter")
    bootcamp = _absolute_path(
        campaign.get("bootcamp_root"), "campaign.bootcamp_root"
    )
    raw_path = Path(_text(binding.get("path"), "campaign.charter.path"))
    path = raw_path if raw_path.is_absolute() else bootcamp / raw_path
    path = path.resolve(strict=True)
    _require(path.is_relative_to(bootcamp), "charter escaped bootcamp root")
    _require(
        day23_gpu.file_sha256(path) == binding.get("file_sha256"),
        "authoritative charter file hash drifted",
    )
    charter = _load_json(path, "authoritative charter")
    charter_sha = _verify_self_hash(
        charter, "charter_sha256", "authoritative charter"
    )
    _require(
        charter_sha == binding.get("content_sha256")
        and charter.get("schema_name") == "rsi.day23_dpo_charter"
        and charter.get("goal_id") == "goal-0002-day23-dpo",
        "authoritative charter binding drifted",
    )


def _validate_control_extension(campaign: Mapping[str, Any]) -> None:
    version_id = campaign.get("version_id")
    if version_id == "rsi-v0003":
        return
    _require(version_id in {"rsi-v0004", "rsi-v0005"}, "unsupported control extension version")
    bootcamp = _absolute_path(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    extension = _mapping(campaign.get("control_extension"), "campaign.control_extension")

    def load_bound(name: str, self_field: str) -> tuple[dict[str, Any], Path]:
        entry = _mapping(extension.get(name), f"control_extension.{name}")
        raw = Path(_text(entry.get("path"), f"control_extension.{name}.path"))
        path = (raw if raw.is_absolute() else bootcamp / raw).resolve(strict=True)
        _require(path.is_relative_to(bootcamp), f"control extension escaped bootcamp: {name}")
        _require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"control extension file hash drifted: {name}")
        value = _load_json(path, f"control extension {name}")
        content = _verify_self_hash(value, self_field, f"control extension {name}")
        _require(content == entry.get("content_sha256"), f"control extension content hash drifted: {name}")
        return value, path

    if version_id == "rsi-v0005":
        requirement = _mapping(
            extension.get("strict_preunseal_requirement"),
            "v0005 strict pre-unseal requirement",
        )
        run_root = _absolute_path(campaign.get("remote_run_root"), "v0005 run root")
        campaign_path = (run_root / "binding/gpu-campaign.json").resolve(strict=True)
        _require(
            _load_json(campaign_path, "canonical v0005 campaign") == campaign,
            "v0005 runner did not receive the canonical campaign bytes",
        )

        producers = _mapping(campaign.get("producers"), "campaign.producers")
        validator_bound = _mapping(
            producers.get("authority_validator"),
            "campaign.producers.authority_validator",
        )
        validator_required = _mapping(
            requirement.get("validator"),
            "v0005 strict pre-unseal validator requirement",
        )
        validator_path = _absolute_path(
            validator_bound.get("path"), "v0005 authority validator path"
        )
        _require(
            validator_path.is_relative_to(bootcamp)
            and validator_required == validator_bound
            and day23_gpu.file_sha256(validator_path)
            == validator_bound.get("file_sha256"),
            "v0005 authority validator binding drifted",
        )

        # Root one is append-only failure evidence and must never be executable.
        # Require the corrected binding marker and both correction identities
        # here; the campaign-bound authority validator owns their rich schema
        # and the full root-one/event-chain audit below.
        _require(
            campaign.get("campaign_id")
            == "day23-qwen35-dpo-rsi-v0005-binding-r0001"
            and campaign.get("binding_revision_id") == "preoptimizer-binding-r0001",
            "v0005 runner requires the corrected pre-optimizer binding",
        )
        load_bound("preoptimizer_correction_amendment", "amendment_sha256")
        load_bound("execution_binding_correction_event", "event_sha256")
        superseded_identity = _mapping(
            extension.get("superseded_campaign"),
            "v0005 superseded pre-optimizer campaign",
        )
        supersedes = _mapping(
            campaign.get("supersedes_preoptimizer"),
            "v0005 pre-optimizer supersession",
        )
        _require(
            supersedes.get("campaign") == superseded_identity,
            "v0005 corrected campaign source binding drifted",
        )

        for producer_name in ("builder", "runner", "evaluator", "authority_validator"):
            producer = _mapping(
                producers.get(producer_name),
                f"campaign.producers.{producer_name}",
            )
            producer_path = _absolute_path(
                producer.get("path"), f"campaign producer {producer_name} path"
            )
            _require(
                producer_path.is_relative_to(bootcamp)
                and day23_gpu.file_sha256(producer_path)
                == producer.get("file_sha256"),
                f"v0005 campaign producer hash drifted: {producer_name}",
            )

        # Import only the exact campaign-bound validator.  Root one's original
        # validator is preserved separately in its immutable source snapshot.
        import importlib.util

        validator_spec = importlib.util.spec_from_file_location(
            f"_day23_v5_authority_{validator_bound['file_sha256'][:16]}",
            validator_path,
        )
        _require(
            validator_spec is not None and validator_spec.loader is not None,
            "cannot load the bound v0005 authority validator",
        )
        authority = importlib.util.module_from_spec(validator_spec)
        validator_spec.loader.exec_module(authority)
        _require(
            Path(authority.__file__).resolve(strict=True) == validator_path
            and callable(getattr(authority, "validate", None)),
            "v0005 loaded a different authority validator",
        )
        strict_path = _absolute_path(
            requirement.get("receipt_path"),
            "v0005 strict pre-unseal receipt",
        )
        amendment4_raw = Path(
            _text(
                requirement.get("amendment_path"),
                "v0005 strict pre-unseal amendment path",
            )
        )
        _require(not amendment4_raw.is_absolute(), "v0005 strict amendment path must be bootcamp-relative")
        amendment4_path = (bootcamp / amendment4_raw).resolve(strict=True)
        amendment_id = _text(
            requirement.get("amendment_id"),
            "v0005 strict pre-unseal amendment id",
        )
        _require(
            amendment4_path.is_relative_to(
                bootcamp / "rsi-control/charters/goal-0002-day23-dpo/amendments"
            )
            and amendment4_raw.name == f"{amendment_id}.json",
            "v0005 strict amendment escaped the charter amendment directory",
        )
        _require(
            requirement.get("required_before_first_optimizer") is True
            and requirement.get("require_search_unopened") is True
            and strict_path == run_root / "evidence/authority/strict-preunseal.json"
            and requirement.get("receipt_schema_name")
            == "day23.rsi_v0005_strict_authority_preflight"
            and requirement.get("receipt_self_hash_field") == "validation_sha256"
            and requirement.get("amendment_self_hash_field") == "amendment_sha256",
            "v0005 strict pre-unseal requirement drifted",
        )

        strict = _load_json(strict_path, "v0005 strict pre-unseal receipt")
        _verify_self_hash(
            strict, "validation_sha256", "v0005 strict pre-unseal receipt"
        )
        _require(
            strict.get("schema_name")
            == "day23.rsi_v0005_strict_authority_preflight"
            and strict.get("schema_version") == 1
            and strict.get("status") == "pass"
            and strict.get("goal_id") == "goal-0002-day23-dpo"
            and strict.get("version_id") == "rsi-v0005"
            and Path(_text(strict.get("campaign_path"), "v0005 strict campaign path")).resolve()
            == campaign_path
            and strict.get("campaign_file_sha256")
            == day23_gpu.file_sha256(campaign_path)
            and strict.get("campaign_sha256") == campaign.get("campaign_sha256")
            and strict.get("search_unopened_at_preflight") is True
            and strict.get("dev_unopened") is True
            and strict.get("heldout_unopened") is True,
            "v0005 strict pre-unseal receipt identity drifted",
        )
        strict_producer = _mapping(
            strict.get("producer"), "v0005 strict pre-unseal producer"
        )
        _require(
            Path(_text(strict_producer.get("path"), "v0005 strict producer path")).resolve()
            == validator_path
            and strict_producer.get("file_sha256")
            == validator_bound.get("file_sha256"),
            "v0005 strict pre-unseal producer drifted",
        )

        amendment4 = _load_json(amendment4_path, "v0005 strict pre-unseal amendment")
        amendment4_sha = _verify_self_hash(
            amendment4,
            "amendment_sha256",
            "v0005 strict pre-unseal amendment",
        )
        amendment4_entry = _mapping(
            strict.get("strict_preunseal_amendment"),
            "v0005 strict pre-unseal amendment binding",
        )
        _require(
            amendment4_entry.get("path") == str(amendment4_raw)
            and amendment4_entry.get("file_sha256")
            == day23_gpu.file_sha256(amendment4_path)
            and amendment4_entry.get("bytes") == amendment4_path.stat().st_size
            and amendment4_entry.get("content_sha256") == amendment4_sha,
            "v0005 strict pre-unseal amendment binding drifted",
        )
        amendment4_campaign = _mapping(
            amendment4.get("campaign"), "v0005 amendment4 campaign"
        )
        _require(
            amendment4.get("schema_name") == "rsi.day23_dpo_append_only_amendment"
            and amendment4.get("schema_version") == 1
            and amendment4.get("status") == "frozen"
            and amendment4.get("goal_id") == "goal-0002-day23-dpo"
            and amendment4.get("version_id") == "rsi-v0005"
            and amendment4.get("amendment_id") == amendment_id
            and amendment4.get("required_before_first_optimizer") is True
            and amendment4.get("prior_search_rotation_amendment")
            == extension.get("amendment")
            and amendment4.get("strict_validator") == strict_producer
            and amendment4.get("adaptive_overfit_risk")
            == strict.get("risk_disclosure")
            and amendment4_campaign
            == {
                "path": str(campaign_path),
                "file_sha256": day23_gpu.file_sha256(campaign_path),
                "campaign_sha256": campaign.get("campaign_sha256"),
            },
            "v0005 strict pre-unseal amendment drifted",
        )

        # The independent validator owns the full v4 terminal-evidence,
        # rotation, event-chain, recipe, producer, and protected-lease audit.
        # Recompute it on every optimizer/refit launch.  Search may legitimately
        # be claimed by the time a refit launches, so preserve the sealed proof
        # that it was unopened at preflight while keeping all other fields exact.
        recomputed = authority.validate(
            campaign_path, require_search_unopened=False
        )
        recomputed["search_unopened_at_preflight"] = True
        sealed_projection = dict(strict)
        sealed_projection.pop("validation_sha256", None)
        sealed_projection.pop("strict_preunseal_amendment", None)
        _require(
            recomputed == sealed_projection,
            "v0005 strict pre-unseal receipt no longer recomputes exactly",
        )
        return

    amendment, _ = load_bound("amendment", "amendment_sha256")
    failure_event, _ = load_bound("runtime_failure_event", "event_sha256")
    version, _ = load_bound("version", "version_sha256")
    precommit, _ = load_bound("precommit_event", "event_sha256")
    _require(amendment.get("schema_name") == "rsi.day23_dpo_append_only_amendment" and amendment.get("version_id") == "rsi-v0004", "v0004 amendment identity drifted")
    change = _mapping(amendment.get("authorized_change"), "v0004 authorized change")
    _require(change.get("per_device_train_batch_size") == [16, 8] and change.get("gradient_accumulation_steps") == [1, 2] and change.get("nominal_global_train_batch_size") == 32 and float(change.get("minimum_free_memory_fraction")) == 0.20, "v0004 authorized capacity correction drifted")
    source_failure = _mapping(amendment.get("source_runtime_failure"), "v0004 source failure")
    failure_path = _absolute_path(source_failure.get("receipt_path"), "v0004 source failure receipt")
    _require(day23_gpu.file_sha256(failure_path) == source_failure.get("receipt_file_sha256"), "v0004 source failure file drifted")
    failure = _load_json(failure_path, "v0004 source failure receipt")
    _require(_verify_self_hash(failure, "receipt_sha256", "v0004 source failure receipt") == source_failure.get("receipt_sha256") and failure.get("status") == "fail", "v0004 source failure receipt drifted")
    _require(failure.get("error", {}).get("message") == "observed free VRAM fell below the campaign gate" and failure.get("claim_boundary", {}).get("dev_consumed") is False and failure.get("claim_boundary", {}).get("heldout_consumed") is False, "v0004 source failure classification drifted")
    _require(failure_event.get("sequence") == 3 and failure_event.get("state_before") == "version_precommitted" and failure_event.get("state_after") == "iteration_open" and failure_event.get("version_id") == "rsi-v0003", "v0004 failure event transition drifted")
    prior4 = _mapping(precommit.get("prior_event"), "v0004 precommit prior event")
    _require(precommit.get("sequence") == 4 and precommit.get("state_before") == "iteration_open" and precommit.get("state_after") == "version_precommitted" and precommit.get("version_id") == "rsi-v0004" and prior4.get("event_sha256") == failure_event.get("event_sha256"), "v0004 precommit event transition drifted")
    operational = _mapping(version.get("fixed_operational_constraint"), "v0004 operational constraint")
    fresh = _mapping(version.get("fresh_start"), "v0004 fresh-start contract")
    _require(version.get("version_id") == "rsi-v0004" and operational.get("world_size") == 2 and operational.get("per_device_train_batch_size") == 8 and operational.get("gradient_accumulation_steps") == 2 and operational.get("nominal_global_train_batch_size") == 32 and float(operational.get("minimum_free_memory_fraction")) == 0.20, "v0004 version runtime drifted")
    _require(fresh.get("from_promoted_s1") is True and fresh.get("resume_from_v0003_checkpoint") is False and fresh.get("source_failure_checkpoint_candidate_eligible") is False, "v0004 fresh-start authority drifted")
    bound_producers = _mapping(campaign.get("producers"), "campaign.producers")
    amendment_producers = _mapping(amendment.get("producers"), "amendment.producers")
    for name in ("builder", "runner", "evaluator"):
        _require(amendment_producers.get(name, {}).get("file_sha256") == bound_producers.get(name, {}).get("file_sha256"), f"v0004 producer amendment drifted: {name}")


def _dataset_identity(campaign: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    dataset_key = spec.get("dataset_key")
    if dataset_key is None and isinstance(spec.get("dataset"), Mapping):
        dataset_key = spec["dataset"].get("key")
    dataset_key = _text(dataset_key, "run_spec.dataset_key")
    datasets = _mapping(campaign.get("datasets"), "campaign.datasets")
    entry = _mapping(datasets.get(dataset_key), f"campaign.datasets.{dataset_key}")
    _require(dataset_key in {"fit", "full_train"}, "optimizer dataset must be fit or full_train")
    return dataset_key, entry


def _config_entry(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    value = spec.get("executable_config", spec.get("config"))
    return _mapping(value, "run_spec.executable_config")


def _receipt_path(spec: Mapping[str, Any], key: str) -> Path:
    return _absolute_path(spec.get(key), f"run_spec.{key}", must_exist=False)


def _validate_refit_selection(
    campaign: Mapping[str, Any],
    campaign_path: Path,
    campaign_sha: str,
    spec: Mapping[str, Any],
    selection_path: Path | None,
) -> dict[str, Any]:
    _require(selection_path is not None, "refit requires the sealed search selection")
    run_root = _absolute_path(campaign.get("remote_run_root"), "campaign.remote_run_root")
    expected_path = run_root / "evidence" / "selection" / "search-selection.json"
    selection_path = selection_path.expanduser().resolve(strict=True)
    _require(selection_path == expected_path, "refit selection path drifted")
    selection = _load_json(selection_path, "search selection")
    selection_sha = _verify_self_hash(selection, "selection_sha256", "search selection")
    _require(selection.get("schema_name") == f"day23.{_version_token(campaign.get('version_id'))}_search_selection", "refit selection schema drifted")
    _require(selection.get("status") == "selected", "search did not select a refit candidate")
    _require(selection.get("campaign_sha256") == campaign_sha, "refit selection campaign drifted")
    selected = _text(selection.get("selected_candidate"), "selected candidate")
    authorization = _mapping(spec.get("authorization"), "refit authorization")
    required_candidate = spec.get(
        "authorized_only_if_selected_candidate",
        authorization.get("requires_selected_candidate_id"),
    )
    _require(required_candidate == selected, "refit option does not implement the selected candidate")
    _require(float(spec.get("learning_rate")) == float(selection.get("selected_learning_rate")), "refit learning rate differs from selection")
    _require(spec.get("max_steps") == selection.get("selected_checkpoint_step"), "refit step budget differs from selection")
    evaluator = _mapping(
        _mapping(campaign.get("producers"), "campaign.producers").get(
            "evaluator"
        ),
        "campaign.producers.evaluator",
    )
    evaluator_path = _absolute_path(
        evaluator.get("path"), "campaign evaluator path"
    )
    _require(
        day23_gpu.file_sha256(evaluator_path) == evaluator.get("file_sha256"),
        "campaign evaluator source hash drifted",
    )
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            str(evaluator_path),
            "--campaign",
            str(campaign_path),
            "--selection-receipt",
            str(selection_path),
            "--verify-selection",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    _require(
        completed.returncode == 0,
        "independent search-selection recomputation failed: "
        + completed.stderr.strip(),
    )
    return {
        "path": str(selection_path),
        "file_sha256": day23_gpu.file_sha256(selection_path),
        "selection_sha256": selection_sha,
        "selected_candidate": selected,
    }


def validate_campaign(
    campaign_path: Path, run_id: str, selection_path: Path | None = None
) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = _load_json(campaign_path, "GPU campaign")
    _require(campaign.get("schema_name") == campaign_schema(campaign.get("version_id")), "runner requires a supported GPU-bound campaign")
    _require(campaign.get("schema_version") == 1, "campaign schema version drifted")
    _require(
        campaign.get("status") == "gpu_execution_bound_optimizer_pending",
        "GPU campaign status drifted",
    )
    campaign_sha = _verify_self_hash(campaign, "campaign_sha256", "GPU campaign")
    _require(campaign.get("goal_id") == "goal-0002-day23-dpo", "campaign goal drifted")
    _version_token(campaign.get("version_id"))
    _validate_charter_binding(campaign)
    _validate_control_extension(campaign)

    run_specs = _mapping(campaign.get("run_specs"), "campaign.run_specs")
    spec = _mapping(run_specs.get(run_id), f"campaign.run_specs.{run_id}")
    role = _text(spec.get("role"), "run_spec.role")
    _require(role in RUN_ROLES, "run role is not executable by this runner")
    _require(spec.get("fresh_start_from_parent") is True, "run is not a fresh parent start")
    selection: Mapping[str, Any] | None = None
    if role == "unique_finalist_refit_option":
        _require(spec.get("authorized") is False, "refit options must remain dormant in the frozen campaign")
        selection = _validate_refit_selection(
            campaign, campaign_path, campaign_sha, spec, selection_path
        )
    else:
        _require(spec.get("authorized") is True, "non-refit run is not explicitly authorized")

    config_path, config_file_sha = _path_entry(_config_entry(spec), "run config")
    config = _load_json(config_path, "run config")
    _require(not any(value is None or value == [] for value in config.values()), "config contains null or empty-list CLI values")
    for key in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint"):
        _require(key not in config, f"forbidden fresh-start key present: {key}")
    _require(not _contains_forbidden_data(config), "dev or heldout leaked into optimizer config")
    _require("val_dataset" not in config, "optimizer config must not mount dev")

    dataset_key, dataset = _dataset_identity(campaign, spec)
    dataset_path = _absolute_path(dataset.get("path"), f"datasets.{dataset_key}.path")
    _require(day23_gpu.file_sha256(dataset_path) == dataset.get("file_sha256"), "optimizer dataset file hash drifted")
    records = _integer(dataset.get("records"), f"datasets.{dataset_key}.records")
    _require(records > 0 and records % WORLD_SIZE == 0, "optimizer dataset must shard evenly across two ranks")
    _require(config.get("dataset") == [str(dataset_path)], "config optimizer dataset binding drifted")

    fixed = _mapping(campaign.get("fixed_recipe"), "campaign.fixed_recipe")
    runtime_recipe = _mapping(fixed.get("runtime"), "fixed_recipe.runtime")
    per_device_batch, gradient_accumulation = _runtime_profile(campaign)
    _require(runtime_recipe.get("world_size") == WORLD_SIZE, "campaign world size drifted")
    _require(config.get("per_device_train_batch_size") == per_device_batch, "config per-device batch drifted")
    _require(config.get("gradient_accumulation_steps") == gradient_accumulation, "config accumulation drifted")
    _require(config.get("gradient_checkpointing") is True, "gradient checkpointing must remain enabled")
    _require(config.get("rlhf_type") == "dpo" and config.get("loss_type") == "sigmoid", "DPO objective drifted")
    _require(float(config.get("beta")) == 0.1, "DPO beta drifted")
    _require(config.get("dataset_shuffle") is True and config.get("train_dataloader_shuffle") is True, "training shuffle contract drifted")
    _require(config.get("max_steps") == spec.get("max_steps"), "max_steps drifted")
    _require(config.get("learning_rate") == spec.get("learning_rate"), "learning-rate lever drifted")
    _require(config.get("seed") == spec.get("seed") and config.get("data_seed") == spec.get("seed"), "run seed drifted")
    checkpoint_steps = spec.get("checkpoint_steps")
    _require(isinstance(checkpoint_steps, list) and checkpoint_steps, "checkpoint_steps are absent")
    _require(all(isinstance(step, int) and 0 < step <= config["max_steps"] for step in checkpoint_steps), "checkpoint steps are invalid")
    if campaign.get("version_id") == "rsi-v0005" and role == "search_train":
        _require(
            checkpoint_steps == [18, 20]
            and config.get("max_steps") == 30
            and config.get("save_steps") == 2
            and config.get("save_total_limit", 0) >= 7,
            "v0005 search checkpoint-retention contract drifted",
        )
    else:
        _require(config.get("save_steps") in checkpoint_steps, "save_steps is not preregistered")
    _require(config.get("save_strategy") == "steps", "save strategy drifted")

    output_dir = _absolute_path(spec.get("output_dir"), "run_spec.output_dir", must_exist=False)
    _require(Path(config.get("output_dir", "")).resolve() == output_dir, "config output_dir drifted")
    _require(not output_dir.exists(), "fresh run output_dir already exists")
    success_receipt = _receipt_path(spec, "success_receipt")
    failure_receipt = _receipt_path(spec, "failure_receipt")
    _require(not success_receipt.exists() and not failure_receipt.exists(), "run already has a terminal receipt")
    success_receipt.parent.mkdir(parents=True, exist_ok=True)
    _require(success_receipt.parent == failure_receipt.parent, "success/failure receipts must share a directory")

    runner_entry = _mapping(_mapping(campaign.get("producers"), "campaign.producers").get("runner"), "campaign.producers.runner")
    runner_path = Path(__file__).resolve(strict=True)
    _require(Path(_text(runner_entry.get("path"), "producer runner path")).resolve(strict=True) == runner_path, "campaign bound a different runner")
    _require(day23_gpu.file_sha256(runner_path) == runner_entry.get("file_sha256"), "runner source hash drifted")

    remote_parent = _mapping(campaign.get("remote_parent"), "campaign.remote_parent")
    model_path = day23_gpu._verify_remote_parent_payload(remote_parent)
    _require(Path(_text(config.get("model"), "config.model")).resolve() == model_path, "config model is not the verified promoted S1")
    runtime = day23_gpu.validate_runtime(campaign)

    return {
        "campaign": campaign,
        "campaign_path": campaign_path,
        "campaign_file_sha256": day23_gpu.file_sha256(campaign_path),
        "campaign_sha256": campaign_sha,
        "run_id": run_id,
        "spec": spec,
        "role": role,
        "config": config,
        "config_path": config_path,
        "config_file_sha256": config_file_sha,
        "dataset_key": dataset_key,
        "dataset": dataset,
        "dataset_path": dataset_path,
        "records": records,
        "output_dir": output_dir,
        "success_receipt": success_receipt,
        "failure_receipt": failure_receipt,
        "checkpoint_steps": checkpoint_steps,
        "runtime": runtime,
        "per_device_batch": per_device_batch,
        "gradient_accumulation": gradient_accumulation,
        "selection": selection,
    }


def _validate_live_observations(
    campaign: Mapping[str, Any],
    spec: Mapping[str, Any],
    dataset: Mapping[str, Any],
    observations: Any,
    label: str,
) -> Mapping[str, Any]:
    value = _mapping(observations, label)
    per_device_batch, gradient_accumulation = _runtime_profile(campaign)
    _require(value.get("global_step") == spec.get("max_steps"), f"{label} global step drifted")
    trainable = _mapping(value.get("trainable_inventory"), f"{label} trainable")
    optimizer = _mapping(value.get("optimizer_inventory"), f"{label} optimizer")
    _require(trainable.get("tensors") == EXPECTED_LORA_TENSORS and trainable.get("params") == EXPECTED_LORA_PARAMS, f"{label} trainable inventory drifted")
    _require(optimizer.get("parameter_tensors") == EXPECTED_LORA_TENSORS and optimizer.get("parameter_numel") == EXPECTED_LORA_PARAMS and optimizer.get("exactly_trainable_inventory") is True, f"{label} optimizer inventory drifted")
    freeze = _mapping(value.get("freeze"), f"{label} freeze")
    _require(freeze.get("all_versions_unchanged") is True and freeze.get("version_changes") == [], f"{label} frozen parent changed")
    lora = _mapping(value.get("lora"), f"{label} LoRA")
    _require(lora.get("changed") is True and lora.get("initial_digest") != lora.get("final_digest"), f"{label} LoRA did not change")
    reference = _mapping(value.get("reference"), f"{label} reference")
    _require(reference.get("context_count", 0) > 0 and reference.get("implementation") == "same_peft_model_disable_adapter", f"{label} reference evidence drifted")
    forwards = _mapping(_mapping(value.get("forwards"), f"{label} forwards").get("counts"), f"{label} forward counts")
    _require(forwards.get("policy", 0) > 0 and forwards.get("reference", 0) > 0, f"{label} forward inventory drifted")
    logs = _mapping(value.get("logs"), f"{label} logs")
    _require(logs.get("nonfinite") == [] and REQUIRED_LOG_KEYS <= set(logs.get("seen_numeric_keys", [])), f"{label} finite log gate drifted")
    dataloader = _mapping(value.get("dataloader"), f"{label} dataloader")
    _require(dataloader.get("records") == dataset.get("records") and dataloader.get("configured_per_device_batch") == per_device_batch, f"{label} dataloader identity drifted")
    accumulation = _mapping(value.get("gradient_accumulation"), f"{label} accumulation")
    records = accumulation.get("observations")
    _require(isinstance(records, list) and records, f"{label} batch observations are absent")
    _require(all(item.get("configured_gradient_accumulation_steps") == gradient_accumulation and item.get("current_gradient_accumulation_steps") == gradient_accumulation and isinstance(item.get("local_pair_batch_size"), int) and 0 < item["local_pair_batch_size"] <= per_device_batch for item in records), f"{label} live batch observations drifted")
    memory = _mapping(value.get("memory"), f"{label} memory")
    minimum_required = float(campaign["fixed_recipe"]["runtime"]["min_free_memory_fraction"])
    observed_minimum = memory.get("minimum_observed_device_free_fraction")
    _require(isinstance(observed_minimum, (int, float)) and math.isfinite(float(observed_minimum)) and float(observed_minimum) >= minimum_required, f"{label} observed free-memory gate failed")
    peaks = memory.get("device_peaks")
    _require(isinstance(peaks, list) and peaks and all(float(item.get("conservative_peak_free_fraction", -1)) >= minimum_required for item in peaks), f"{label} peak free-memory gate failed")
    _require(value.get("violations") == [], f"{label} runtime violations are non-empty")
    return value


def validate_training_receipt_value(
    campaign: Mapping[str, Any],
    campaign_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Strictly recompute all promotion-relevant training receipt claims."""
    receipt_path = receipt_path.expanduser().resolve()
    receipt_sha = _verify_self_hash(receipt, "receipt_sha256", "training receipt")
    version_id = _text(campaign.get("version_id"), "campaign.version_id")
    per_device_batch, gradient_accumulation = _runtime_profile(campaign)
    _require(receipt.get("schema_name") == receipt_schema(version_id) and receipt.get("schema_version") == 1 and receipt.get("status") == "pass", "training success receipt schema/status drifted")
    _require(receipt.get("goal_id") == "goal-0002-day23-dpo" and receipt.get("version_id") == version_id, "training receipt goal/version drifted")
    run_id = _text(receipt.get("run_id"), "training receipt run_id")
    spec = _mapping(_mapping(campaign.get("run_specs"), "campaign.run_specs").get(run_id), f"campaign run {run_id}")
    _require(receipt.get("run_role") == spec.get("role"), "training receipt role drifted")
    _require(receipt_path == Path(_text(spec.get("success_receipt"), "run success receipt")).resolve(), "training receipt path is not bound")
    _require(not Path(_text(spec.get("failure_receipt"), "run failure receipt")).resolve().exists(), "run has both success and failure receipts")
    campaign_entry = _mapping(receipt.get("campaign"), "receipt campaign")
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    _require(Path(_text(campaign_entry.get("path"), "receipt campaign path")).resolve() == campaign_path and campaign_entry.get("file_sha256") == day23_gpu.file_sha256(campaign_path) and campaign_entry.get("campaign_sha256") == campaign.get("campaign_sha256"), "training receipt campaign identity drifted")
    producer = _mapping(receipt.get("producer"), "receipt producer")
    producer_bound = _mapping(_mapping(campaign.get("producers"), "campaign.producers").get("runner"), "campaign runner")
    _require(Path(_text(producer.get("path"), "receipt producer path")).resolve() == Path(_text(producer_bound.get("path"), "bound runner path")).resolve() and producer.get("file_sha256") == producer_bound.get("file_sha256"), "training receipt producer drifted")
    config_bound = _mapping(spec.get("executable_config"), "bound run config")
    config_receipt = _mapping(receipt.get("config"), "receipt config")
    config_path = _absolute_path(config_bound.get("path"), "bound run config path")
    _require(config_receipt == {"path": str(config_path), "file_sha256": config_bound.get("file_sha256")} and day23_gpu.file_sha256(config_path) == config_bound.get("file_sha256"), "training receipt config identity drifted")
    dataset = _mapping(_mapping(campaign.get("datasets"), "campaign.datasets").get(spec.get("dataset_key")), "bound optimizer dataset")
    dataset_receipt = _mapping(receipt.get("dataset"), "receipt dataset")
    dataset_path = _absolute_path(dataset.get("path"), "bound optimizer dataset path")
    _require(dataset_receipt == {"key": spec.get("dataset_key"), "path": str(dataset_path), "file_sha256": dataset.get("file_sha256"), "records": dataset.get("records")} and day23_gpu.file_sha256(dataset_path) == dataset.get("file_sha256"), "training receipt dataset identity drifted")
    runtime = _mapping(receipt.get("runtime_identity"), "training receipt runtime")
    runtime_bound = _mapping(campaign.get("runtime_parse"), "campaign.runtime_parse")
    python_bound = _mapping(runtime_bound.get("python_executable"), "bound Python")
    _require(runtime.get("python") == {"path": python_bound.get("path"), "file_sha256": python_bound.get("file_sha256"), "version": python_bound.get("version")}, "training receipt Python runtime drifted")
    _require(runtime.get("package_versions") == runtime_bound.get("package_versions"), "training receipt package runtime drifted")
    swift_runtime = _mapping(runtime.get("ms_swift_checkout"), "training receipt ms-swift")
    _require(swift_runtime.get("path") == runtime_bound.get("ms_swift_checkout") and swift_runtime.get("commit") == runtime_bound.get("ms_swift_commit") and swift_runtime.get("clean") is True, "training receipt ms-swift runtime drifted")
    python_sha = _text(python_bound.get("file_sha256"), "bound Python hash")
    top_process = day23_gpu._validated_process_identity(receipt.get("process_identity"), "training receipt process", python_file_sha256=python_sha)
    claim = _mapping(receipt.get("claim_boundary"), "training receipt claim")
    for key in ("runtime_stage_passed", "fresh_parent_start", "fresh_lora_only", "reference_disable_adapter_observed", "frozen_tensor_versions_unchanged", "two_gpu_ddp_runtime_proven"):
        _require(claim.get(key) is True, f"training receipt did not prove {key}")
    _require(claim.get("checkpoint_reload_pending_fresh_process") is True and claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, "training receipt claim boundary drifted")
    distributed = _mapping(receipt.get("distributed_evidence"), "training distributed evidence")
    ranks = distributed.get("ranks")
    _require(distributed.get("world_size") == WORLD_SIZE and distributed.get("configured_nominal_global_train_batch_size") == WORLD_SIZE * per_device_batch * gradient_accumulation and isinstance(ranks, list) and len(ranks) == WORLD_SIZE, "training distributed topology drifted")
    _require([item.get("rank") for item in ranks] == [0, 1] and [item.get("local_rank") for item in ranks] == [0, 1], "training rank inventory drifted")
    rank_processes = []
    lora_digests = set()
    for rank, entry in enumerate(ranks):
        process = day23_gpu._validated_process_identity(entry.get("process_identity"), f"training rank {rank} process", python_file_sha256=python_sha)
        rank_processes.append((process["boot_id"], process["pid"], process["proc_start_ticks"]))
        topology = _mapping(entry.get("topology"), f"training rank {rank} topology")
        _require(topology.get("world_size") == WORLD_SIZE and topology.get("rank") == rank and topology.get("local_rank") == rank and topology.get("visible_device_count") == WORLD_SIZE, f"training rank {rank} topology drifted")
        observed = _validate_live_observations(campaign, spec, dataset, entry.get("observations"), f"training rank {rank}")
        lora_digests.add(observed["lora"]["final_digest"])
    _require(len(set(rank_processes)) == WORLD_SIZE and tuple(rank_processes[0]) == (top_process["boot_id"], top_process["pid"], top_process["proc_start_ticks"]), "training process identity inventory drifted")
    _require(len(lora_digests) == 1, "training ranks ended with different LoRA tensors")
    _require(receipt.get("observations") == ranks[0].get("observations"), "top-level observations do not equal rank zero")
    profile_context = {"spec": spec, "records": dataset.get("records"), "campaign": campaign}
    expected_profile = _realized_batch_profile(ranks, profile_context)
    _require(distributed.get("realized_global_batch_profile") == expected_profile, "training realized batch profile drifted")
    checkpoints = receipt.get("checkpoints")
    _require(isinstance(checkpoints, list) and [item.get("global_step") for item in checkpoints] == spec.get("checkpoint_steps"), "training checkpoint step inventory drifted")
    expected_noncandidates = (
        [22, 24, 26, 28, 30]
        if version_id == "rsi-v0005" and spec.get("role") == "search_train"
        else []
    )
    _require(
        receipt.get("retained_noncandidate_checkpoint_steps", [])
        == expected_noncandidates,
        "training non-candidate checkpoint disclosure drifted",
    )
    actual_checkpoints = []
    for checkpoint in checkpoints:
        actual = _checkpoint_manifest(_absolute_path(checkpoint.get("path"), "training checkpoint path"), int(checkpoint["global_step"]))
        _require(actual == checkpoint, "training checkpoint manifest drifted")
        actual_checkpoints.append(actual)
    _require(receipt.get("checkpoint") == actual_checkpoints[-1], "training final checkpoint projection drifted")
    return {"receipt_sha256": receipt_sha, "run_id": run_id, "spec": spec, "dataset": dataset, "checkpoints": actual_checkpoints}


class CampaignObserver(day23_gpu.RuntimeObserver):
    """Use the proven observer hooks with campaign-specific success gates."""

    def __init__(self, context: Mapping[str, Any]) -> None:
        super().__init__("bounded_smoke_30step", context["config"])
        self.context = context

    def finalize(self, *, require_success: bool) -> dict[str, Any]:
        evidence = super().finalize(require_success=False)
        if not require_success:
            return evidence
        _require(evidence.get("global_step") == self.context["spec"].get("max_steps"), "trainer global_step drifted")
        trainable = _mapping(evidence.get("trainable_inventory"), "trainable inventory")
        optimizer = _mapping(evidence.get("optimizer_inventory"), "optimizer inventory")
        _require(trainable.get("tensors") == EXPECTED_LORA_TENSORS and trainable.get("params") == EXPECTED_LORA_PARAMS, "trainable LoRA inventory drifted")
        _require(optimizer.get("parameter_tensors") == EXPECTED_LORA_TENSORS and optimizer.get("parameter_numel") == EXPECTED_LORA_PARAMS, "optimizer inventory drifted")
        freeze = _mapping(evidence.get("freeze"), "freeze evidence")
        _require(freeze.get("all_versions_unchanged") is True and freeze.get("version_changes") == [], "a frozen tensor changed")
        lora = _mapping(evidence.get("lora"), "LoRA evidence")
        _require(lora.get("changed") is True, "LoRA digest did not change")
        reference = _mapping(evidence.get("reference"), "reference evidence")
        _require(reference.get("context_count", 0) > 0 and reference.get("implementation") == "same_peft_model_disable_adapter", "reference adapter-disable path was not observed")
        forwards = _mapping(evidence.get("forwards"), "forward evidence")
        counts = _mapping(forwards.get("counts"), "forward counts")
        _require(counts.get("policy", 0) > 0 and counts.get("reference", 0) > 0, "policy/reference forwards were not observed")
        logs = _mapping(evidence.get("logs"), "log evidence")
        _require(logs.get("nonfinite") == [], "non-finite runtime evidence was observed")
        _require(REQUIRED_LOG_KEYS <= set(logs.get("seen_numeric_keys", [])), "required finite DPO metrics were not logged")
        per_device_batch, gradient_accumulation = _runtime_profile(self.context["campaign"])
        dataloader = _mapping(evidence.get("dataloader"), "dataloader evidence")
        _require(dataloader.get("records") == self.context["records"], "optimizer dataset record count drifted")
        _require(dataloader.get("configured_per_device_batch") == per_device_batch, "live per-device batch drifted")
        accumulation = _mapping(evidence.get("gradient_accumulation"), "gradient accumulation evidence")
        observations = accumulation.get("observations")
        _require(isinstance(observations, list) and observations, "training-step observations are absent")
        _require(all(item.get("configured_gradient_accumulation_steps") == gradient_accumulation and item.get("current_gradient_accumulation_steps") == gradient_accumulation for item in observations), "live gradient accumulation drifted")
        _require(all(0 < item.get("local_pair_batch_size", 0) <= per_device_batch for item in observations), "live local pair batch drifted")
        memory = _mapping(evidence.get("memory"), "memory evidence")
        minimum = memory.get("minimum_observed_device_free_fraction")
        _require(isinstance(minimum, (int, float)) and math.isfinite(float(minimum)) and float(minimum) >= float(self.context["campaign"]["fixed_recipe"]["runtime"]["min_free_memory_fraction"]), "observed free VRAM fell below the campaign gate")
        peaks = memory.get("device_peaks")
        _require(isinstance(peaks, list) and peaks and all(float(item.get("conservative_peak_free_fraction", -1)) >= float(self.context["campaign"]["fixed_recipe"]["runtime"]["min_free_memory_fraction"]) for item in peaks), "peak VRAM headroom gate failed")
        _require(evidence.get("violations") == [], "runtime observer recorded violations")
        return evidence


def _realized_batch_profile(rank_evidence: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    per_device_batch, gradient_accumulation = _runtime_profile(_mapping(context.get("campaign"), "batch-profile campaign"))
    max_steps = _integer(context["spec"].get("max_steps"), "run max_steps")
    by_step: dict[int, dict[int, int]] = {step: {} for step in range(max_steps)}
    for rank_doc in rank_evidence:
        rank = _integer(rank_doc.get("rank"), "rank evidence rank")
        observations = rank_doc["observations"]["gradient_accumulation"]["observations"]
        local: dict[int, int] = {}
        for item in observations:
            step = _integer(item.get("global_step_before"), "batch global step")
            size = _integer(item.get("local_pair_batch_size"), "local pair batch")
            _require(step in by_step, "batch observation step escaped run")
            local[step] = local.get(step, 0) + size
        counts: dict[int, int] = {}
        for item in observations:
            step = _integer(item.get("global_step_before"), "batch global step")
            counts[step] = counts.get(step, 0) + 1
        _require(all(counts.get(step) == gradient_accumulation for step in by_step), f"rank {rank} microbatch count drifted")
        _require(set(local) == set(by_step), f"rank {rank} did not observe every optimizer step")
        for step, size in local.items():
            by_step[step][rank] = size
    _require(all(set(value) == {0, 1} for value in by_step.values()), "batch profile lacks one DDP rank")
    actual = [sum(by_step[step].values()) for step in range(max_steps)]
    per_rank = context["records"] // WORLD_SIZE
    full_batches, remainder = divmod(per_rank, per_device_batch)
    microbatches = [WORLD_SIZE * per_device_batch] * full_batches
    if remainder:
        microbatches.append(WORLD_SIZE * remainder)
    epoch = [
        sum(microbatches[index : index + gradient_accumulation])
        for index in range(0, len(microbatches), gradient_accumulation)
    ]
    _require(epoch, "empty expected batch profile")
    expected = [epoch[index % len(epoch)] for index in range(max_steps)]
    _require(actual == expected, "realized global pair batch profile drifted")
    return {
        "per_step_global_pair_counts": actual,
        "per_step_global_pair_counts_sha256": hashlib.sha256(
            day23_gpu.canonical_json(actual)
        ).hexdigest(),
        "minimum": min(actual),
        "maximum": max(actual),
        "unique": sorted(set(actual)),
        "partial_batch_present": len(set(actual)) > 1,
        "total_pair_exposures": sum(actual),
    }


def _checkpoint_manifest(path: Path, step: int) -> dict[str, Any]:
    try:
        return day23_gpu.checkpoint_manifest(path, step)
    except day23_gpu.Day23GPUStageError as error:
        raise RSICandidateError(str(error)) from error


def _collect_checkpoints(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    expected = list(context["checkpoint_steps"])
    actual: dict[int, Path] = {}
    for path in context["output_dir"].glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        actual[step] = path
    if (
        context.get("campaign", {}).get("version_id") == "rsi-v0005"
        and context.get("role") == "search_train"
    ):
        # The frozen 30-step cosine horizon needs save_steps=2 so that the
        # preregistered observations at steps 18 and 20 both exist.  Keeping
        # the final seven saves retains those two plus five explicitly
        # non-candidate diagnostics; only the two registry steps are sealed
        # into the promotion-capable receipt below.
        retained = {18, 20, 22, 24, 26, 28, 30}
        _require(set(actual) == retained, "v0005 retained checkpoint inventory drifted")
        _require(set(expected) == {18, 20}, "v0005 candidate checkpoint inventory drifted")
    else:
        _require(set(actual) == set(expected), "checkpoint step inventory drifted")
    for step in expected:
        result.append(_checkpoint_manifest(actual[step], step))
    return result


def _base_receipt(context: Mapping[str, Any], started_at: str, runtime: Mapping[str, Any] | None) -> dict[str, Any]:
    version_id = context.get("campaign", {}).get("version_id")
    return {
        "schema_name": receipt_schema(version_id),
        "schema_version": 1,
        "campaign_id": context.get("campaign", {}).get("campaign_id"),
        "goal_id": "goal-0002-day23-dpo",
        "version_id": version_id,
        "run_id": context.get("run_id"),
        "run_role": context.get("role"),
        "started_at_utc": started_at,
        "campaign": {
            "path": str(context.get("campaign_path")),
            "file_sha256": context.get("campaign_file_sha256"),
            "campaign_sha256": context.get("campaign_sha256"),
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": day23_gpu.file_sha256(Path(__file__).resolve()),
        },
        "config": {
            "path": str(context.get("config_path")),
            "file_sha256": context.get("config_file_sha256"),
        },
        "dataset": {
            "key": context.get("dataset_key"),
            "path": str(context.get("dataset_path")),
            "file_sha256": context.get("dataset", {}).get("file_sha256"),
            "records": context.get("records"),
        },
        "process_identity": day23_gpu._process_identity(),
        "runtime_identity": runtime,
        "selection": context.get("selection"),
    }


def _seal_failure(context: Mapping[str, Any], started_at: str, started_monotonic: float, runtime: Mapping[str, Any] | None, observer: CampaignObserver | None, error: BaseException) -> None:
    failure_path = context.get("failure_receipt")
    if not isinstance(failure_path, Path):
        return
    observations: Any = None
    if observer is not None:
        with contextlib.suppress(BaseException):
            observations = observer.finalize(require_success=False)
    value = _base_receipt(context, started_at, runtime)
    value.update({
        "status": "fail",
        "completed_at_utc": day23_gpu.utc_now(),
        "duration_seconds": time.monotonic() - started_monotonic,
        "error": {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": traceback.format_exception(error)[-12:],
        },
        "observations": observations,
        "claim_boundary": {"scientific_result_available": bool(observations and observations.get("global_step", 0)), "dev_consumed": False, "heldout_consumed": False},
    })
    with contextlib.suppress(FileExistsError, OSError, RSICandidateError, day23_gpu.Day23GPUStageError):
        day23_gpu.write_sealed_json(failure_path, value, "receipt_sha256")


def run_candidate(
    campaign_path: Path, run_id: str, selection_path: Path | None = None
) -> tuple[Path | None, str | None]:
    started_at = day23_gpu.utc_now()
    started_monotonic = time.monotonic()
    context: dict[str, Any] = {"campaign_path": campaign_path.expanduser().absolute(), "run_id": run_id}
    runtime: Mapping[str, Any] | None = None
    observer: CampaignObserver | None = None
    ddp: Mapping[str, int] | None = None
    try:
        ddp = day23_gpu.initialize_ddp()
        context.update(validate_campaign(campaign_path, run_id, selection_path))
        runtime = context["runtime"]
        import torch

        torch.distributed.barrier()
        from swift.cli.main import parse_yaml_args
        from swift.pipelines import rlhf_main

        argv = [str(context["config_path"])]
        parse_yaml_args(argv)
        topology = day23_gpu.validate_topology(context["config"])
        torch.cuda.reset_peak_memory_stats(ddp["local_rank"])
        observer = CampaignObserver(context)
        observer.snapshot_memory("before_rsi_candidate_rlhf_main")
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
        _require([item.get("rank") for item in gathered] == [0, 1], "DDP rank evidence inventory drifted")
        _require(len({item["observations"]["lora"]["final_digest"] for item in gathered}) == 1, "DDP ranks ended with different LoRA tensors")
        _require({item["observations"]["global_step"] for item in gathered} == {context["spec"]["max_steps"]}, "DDP ranks ended at different steps")
        profile = _realized_batch_profile(gathered, context)
        torch.distributed.barrier()
        if ddp["rank"] != 0:
            return None, None

        checkpoints = _collect_checkpoints(context)
        receipt = _base_receipt(context, started_at, runtime)
        receipt.update({
            "status": "pass",
            "completed_at_utc": day23_gpu.utc_now(),
            "duration_seconds": time.monotonic() - started_monotonic,
            "observations": observations,
            "distributed_evidence": {
                "world_size": WORLD_SIZE,
                "configured_nominal_global_train_batch_size": WORLD_SIZE * context["per_device_batch"] * context["gradient_accumulation"],
                "realized_global_batch_profile": profile,
                "ranks": gathered,
            },
            "checkpoints": checkpoints,
            "checkpoint": checkpoints[-1],
            "retained_noncandidate_checkpoint_steps": (
                [22, 24, 26, 28, 30]
                if context["campaign"].get("version_id") == "rsi-v0005"
                and context["role"] == "search_train"
                else []
            ),
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
        })
        sealed_for_validation = dict(receipt)
        sealed_for_validation["receipt_sha256"] = day23_gpu.object_sha256(
            receipt
        )
        validate_training_receipt_value(
            context["campaign"],
            context["campaign_path"],
            sealed_for_validation,
            context["success_receipt"],
        )
        sha = day23_gpu.write_sealed_json(context["success_receipt"], receipt, "receipt_sha256")
        return context["success_receipt"], sha
    except BaseException as error:
        _seal_failure(context, started_at, started_monotonic, runtime, observer, error)
        raise


def _self_test() -> None:
    fake = {
        "spec": {"max_steps": 3},
        "records": 124,
        "campaign": {"version_id": "rsi-v0003", "fixed_recipe": {"runtime": {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 1, "min_free_memory_fraction": 0.30}}},
    }
    ranks = []
    for rank in (0, 1):
        observations = []
        for step, size in enumerate((16, 16, 16)):
            observations.append({"global_step_before": step, "local_pair_batch_size": size, "current_gradient_accumulation_steps": 1})
        ranks.append({"rank": rank, "observations": {"gradient_accumulation": {"observations": observations}}})
    profile = _realized_batch_profile(ranks, fake)
    _require(profile["per_step_global_pair_counts"] == [32, 32, 32], "self-test batch profile failed")
    fake_v4 = {
        "spec": {"max_steps": 4},
        "records": 124,
        "campaign": {"version_id": "rsi-v0004", "fixed_recipe": {"runtime": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "min_free_memory_fraction": 0.20}}},
    }
    ranks_v4 = []
    for rank in (0, 1):
        observations = []
        for step, sizes in enumerate(((8, 8), (8, 8), (8, 8), (8, 6))):
            observations.extend({"global_step_before": step, "local_pair_batch_size": size, "current_gradient_accumulation_steps": 2} for size in sizes)
        ranks_v4.append({"rank": rank, "observations": {"gradient_accumulation": {"observations": observations}}})
    profile_v4 = _realized_batch_profile(ranks_v4, fake_v4)
    _require(profile_v4["per_step_global_pair_counts"] == [32, 32, 32, 28], "v0004 batch profile self-test failed")
    _require(_contains_forbidden_data({"dataset": ["x-heldout.jsonl"]}), "self-test leakage guard failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--execute", choices=[EXECUTE_TOKEN], required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"status": "pass", "scope": "stdlib_self_test"}, sort_keys=True))
        return 0
    try:
        path, sha = run_candidate(
            args.campaign, args.run, args.selection_receipt
        )
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    if path is not None:
        print(json.dumps({"status": "pass", "receipt": str(path), "receipt_sha256": sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
