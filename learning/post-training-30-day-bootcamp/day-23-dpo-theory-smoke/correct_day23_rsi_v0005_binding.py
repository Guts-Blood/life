#!/usr/bin/env python3
"""Append-only correction for the pre-optimizer RSI-v0005 binding projection.

The first v0005 GPU campaign is immutable and remains evidence of an invalid
authority binding.  Its amendment3 recorded the genuine v0004 strict
pre-unseal amendment identity with an additional, accurate ``bytes``
attestation, while the strict validator compared that identity to the
three-field projection stored by the v0004 receipt.  No optimizer, search,
dev, or heldout access occurred.

This executable validates that exact bytes-only mismatch, seals amendment5
and event7, and emits a new run-root campaign with byte-identical datasets and
scientifically identical configs.  It never reads dev or heldout rows and
never overwrites the source campaign or any control artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_day23_rsi_v0004 as v4
import day23_contract as contract
import run_day23_qwen35_gpu_stage as day23_gpu


GOAL_ID = "goal-0002-day23-dpo"
VERSION_ID = "rsi-v0005"
CAMPAIGN_SCHEMA = "day23.rsi_v0005_gpu_campaign"
CAMPAIGN_ID = "day23-qwen35-dpo-rsi-v0005-binding-r0001"
BINDING_REVISION_ID = "preoptimizer-binding-r0001"
CONTROL_ROOT_REL = Path("rsi-control/charters/goal-0002-day23-dpo")
AMENDMENT5_NAME = "amendment-000005-v0005-validator-projection-correction"
EVENT7_NAME = "event-000007-execution-binding-corrected"
AMENDMENT4_REL = CONTROL_ROOT_REL / "amendments/amendment-000004-v0005-strict-preunseal.json"
EXPECTED_RUN_IDS = {
    "search_lr_3p9e_6",
    "search_lr_4p1e_6",
    "refit_lr_3p9e_6_step_18",
    "refit_lr_3p9e_6_step_20",
    "refit_lr_4p1e_6_step_18",
    "refit_lr_4p1e_6_step_20",
}


class BindingCorrectionError(RuntimeError):
    """The source-invalid or corrected-binding contract did not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingCorrectionError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BindingCorrectionError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    _require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def _bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _identity(
    path: Path,
    *,
    content_field: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"missing regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if content_field:
        result["content_sha256"] = _verify(_load(resolved, str(resolved)), content_field, str(resolved))
    return result


def _resolve_bound(
    bootcamp: Path,
    entry: Any,
    field: str,
    label: str,
    *,
    allow_external: bool = False,
) -> tuple[dict[str, Any], Path]:
    item = _mapping(entry, label)
    raw = Path(str(item.get("path")))
    path = (raw if raw.is_absolute() else bootcamp / raw).resolve(strict=True)
    _require(allow_external or path.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    _require(day23_gpu.file_sha256(path) == item.get("file_sha256"), f"{label} file hash drifted")
    value = _load(path, label)
    _require(_verify(value, field, label) == item.get("content_sha256"), f"{label} content binding drifted")
    if "bytes" in item:
        _require(path.stat().st_size == item.get("bytes"), f"{label} byte count drifted")
    return value, path


def _validate_file_identity(path: Path, entry: Mapping[str, Any], label: str) -> None:
    path = path.resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    _require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"{label} file hash drifted")
    if "bytes" in entry:
        _require(path.stat().st_size == entry.get("bytes"), f"{label} bytes drifted")


def _source_state(source_path: Path, *, allow_root2_strict: bool) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve(strict=True)
    source = _load(source_path, "source v0005 campaign")
    source_sha = _verify(source, "campaign_sha256", "source v0005 campaign")
    _require(
        source.get("schema_name") == CAMPAIGN_SCHEMA
        and source.get("schema_version") == 1
        and source.get("status") == "gpu_execution_bound_optimizer_pending"
        and source.get("goal_id") == GOAL_ID
        and source.get("version_id") == VERSION_ID,
        "source v0005 campaign identity drifted",
    )
    source_root = Path(str(source.get("remote_run_root"))).resolve(strict=True)
    bootcamp = Path(str(source.get("bootcamp_root"))).resolve(strict=True)
    _require(source_path == source_root / "binding/gpu-campaign.json", "source campaign is not at its canonical path")
    _require(source.get("binding_revision_id") is None, "source is already a binding correction")

    extension = _mapping(source.get("control_extension"), "source control extension")
    amendment3, amendment3_path = _resolve_bound(bootcamp, extension.get("amendment"), "amendment_sha256", "amendment3")
    event6, event6_path = _resolve_bound(bootcamp, extension.get("precommit_event"), "event_sha256", "event6")
    strict_source, strict_source_path = _resolve_bound(
        bootcamp,
        extension.get("source_strict_preunseal"),
        "validation_sha256",
        "v0004 strict pre-unseal receipt",
        allow_external=True,
    )
    _require(
        amendment3.get("amendment_id") == "amendment-000003-v0005-terminal-interpolation"
        and event6.get("event_id") == "event-000006-version-precommitted"
        and event6.get("sequence") == 6
        and event6.get("state_after") == "version_precommitted",
        "source v0005 control chain drifted",
    )

    observed = copy.deepcopy(_mapping(amendment3.get("authority"), "amendment3 authority").get("source_strict_preunseal_amendment"))
    canonical = copy.deepcopy(_mapping(strict_source.get("strict_preunseal_amendment"), "v0004 strict amendment projection"))
    _require(set(observed) == set(canonical) | {"bytes"}, "source mismatch is not bytes-only")
    _require({key: observed[key] for key in canonical} == canonical, "source mismatch changed a semantic identity field")
    amendment2_raw = Path(str(canonical.get("path")))
    amendment2_path = (amendment2_raw if amendment2_raw.is_absolute() else bootcamp / amendment2_raw).resolve(strict=True)
    _validate_file_identity(amendment2_path, observed, "v0004 strict amendment2 augmented identity")
    _require(observed.get("bytes") == amendment2_path.stat().st_size, "source bytes attestation is inaccurate")

    requirement = _mapping(extension.get("strict_preunseal_requirement"), "source strict requirement")
    source_strict_receipt_path = Path(str(requirement.get("receipt_path"))).resolve()
    _require(not source_strict_receipt_path.exists(), "source invalid campaign unexpectedly has a strict receipt")
    amendment4_path = bootcamp / AMENDMENT4_REL
    if amendment4_path.exists():
        _require(allow_root2_strict, "canonical strict amendment4 existed before corrected binding")
        amendment4 = _load(amendment4_path, "corrected strict amendment4")
        _verify(amendment4, "amendment_sha256", "corrected strict amendment4")
        _require(amendment4.get("campaign", {}).get("campaign_sha256") != source_sha, "strict amendment4 incorrectly binds invalid source campaign")

    access = _mapping(source.get("access_ledger"), "source access ledger")
    search_claim = Path(str(access.get("search_claim"))).resolve()
    dev_claim = Path(str(access.get("dev_claim"))).resolve()
    _require(not search_claim.exists(), "source search was opened")
    _require(not dev_claim.exists(), "global dev was opened")
    charter_entry = _mapping(source.get("charter"), "source charter")
    charter_path = (bootcamp / str(charter_entry.get("path"))).resolve(strict=True)
    charter = _load(charter_path, "charter")
    _require(day23_gpu.file_sha256(charter_path) == charter_entry.get("file_sha256"), "charter file drifted")
    _require(_verify(charter, "charter_sha256", "charter") == charter_entry.get("content_sha256"), "charter content drifted")
    heldout_rel = Path(str(charter["access_leases"]["heldout"]["claim_path"]))
    _require(not (bootcamp / "rsi-control" / heldout_rel).exists(), "global heldout was opened")

    outputs = source_root / "outputs"
    _require(not outputs.exists() or not any(outputs.iterdir()), "source optimizer output exists")
    specs = _mapping(source.get("run_specs"), "source run specs")
    _require(set(specs) == EXPECTED_RUN_IDS, "source run registry drifted")
    for run_id, spec_value in specs.items():
        spec = _mapping(spec_value, f"source run {run_id}")
        _require(not Path(str(spec.get("success_receipt"))).exists(), f"source success receipt exists: {run_id}")
        _require(not Path(str(spec.get("failure_receipt"))).exists(), f"source failure receipt exists: {run_id}")
        config_entry = _mapping(spec.get("executable_config"), f"source config {run_id}")
        _validate_file_identity(Path(str(config_entry.get("path"))), config_entry, f"source config {run_id}")

    datasets = _mapping(source.get("datasets"), "source datasets")
    for key in ("fit", "search", "full_train"):
        entry = _mapping(datasets.get(key), f"source dataset {key}")
        _validate_file_identity(Path(str(entry.get("path"))), entry, f"source dataset {key}")

    snapshots_root = source_root / "evidence/source-snapshot"
    source_snapshots: dict[str, Any] = {}
    for name, producer_value in sorted(_mapping(source.get("producers"), "source producers").items()):
        producer = _mapping(producer_value, f"source producer {name}")
        snapshot_path = snapshots_root / Path(str(producer.get("path"))).name
        _require(snapshot_path.is_file(), f"source producer snapshot missing: {name}")
        _require(
            day23_gpu.file_sha256(snapshot_path) == producer.get("file_sha256")
            and snapshot_path.stat().st_size == producer.get("bytes"),
            f"source producer snapshot drifted: {name}",
        )
        source_snapshots[name] = _identity(snapshot_path)

    source_identity = _identity(source_path, content_field="campaign_sha256")
    return {
        "campaign": source,
        "campaign_path": source_path,
        "campaign_sha256": source_sha,
        "campaign_identity": source_identity,
        "source_root": source_root,
        "bootcamp": bootcamp,
        "charter": charter,
        "event6": event6,
        "event6_path": event6_path,
        "amendment3": amendment3,
        "amendment3_path": amendment3_path,
        "strict_source": strict_source,
        "strict_source_path": strict_source_path,
        "observed_projection": observed,
        "canonical_projection": canonical,
        "amendment2_path": amendment2_path,
        "source_snapshots": source_snapshots,
    }


def _producer_paths() -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    result = {
        "builder": Path(__file__).resolve(),
        "runner": script_dir / "run_day23_rsi_candidate.py",
        "evaluator": script_dir / "eval_day23_rsi_candidate.py",
        "authority_validator": script_dir / "validate_day23_rsi_v0005_authority.py",
    }
    for name, path in result.items():
        _require(path.is_file() and not path.is_symlink(), f"corrected producer missing: {name}")
    return result


def _prepare_payloads(source: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    source_campaign = source["campaign"]
    source_root: Path = source["source_root"]
    source_datasets = _mapping(source_campaign.get("datasets"), "source datasets")
    payloads: dict[str, bytes] = {}
    datasets = copy.deepcopy(source_datasets)
    inventory: dict[str, Any] = {}
    for key, filename in (
        ("fit", "day23-qwen35-dpo-rsi-v0005-fit.jsonl"),
        ("search", "day23-qwen35-dpo-rsi-v0005-search.jsonl"),
    ):
        old_entry = _mapping(source_datasets.get(key), f"source dataset {key}")
        old_path = Path(str(old_entry.get("path"))).resolve(strict=True)
        payload = old_path.read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == old_entry.get("file_sha256"), f"source dataset bytes drifted: {key}")
        new_path = run_root / "binding/data" / filename
        new_entry = copy.deepcopy(old_entry)
        new_entry["path"] = str(new_path)
        datasets[key] = new_entry
        relative = f"binding/data/{filename}"
        payloads[relative] = payload
        inventory[relative] = {"file_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}

    source_specs = _mapping(source_campaign.get("run_specs"), "source run specs")
    specs: dict[str, Any] = {}
    configs: dict[str, bytes] = {}
    for run_id, source_spec_value in sorted(source_specs.items()):
        source_spec = _mapping(source_spec_value, f"source run {run_id}")
        old_config_entry = _mapping(source_spec.get("executable_config"), f"source config {run_id}")
        old_config_path = Path(str(old_config_entry.get("path"))).resolve(strict=True)
        _validate_file_identity(old_config_path, old_config_entry, f"source config {run_id}")
        config = _load(old_config_path, f"source config {run_id}")
        if source_spec.get("dataset_key") == "fit":
            config["dataset"] = [str(datasets["fit"]["path"])]
        else:
            _require(source_spec.get("dataset_key") == "full_train", f"unexpected optimizer dataset: {run_id}")
            config["dataset"] = [str(datasets["full_train"]["path"])]
        config["output_dir"] = str(run_root / "outputs" / run_id)
        payload = _bytes(config)
        filename = old_config_path.name
        new_config_path = run_root / "binding/configs" / filename
        new_config_entry = {
            "path": str(new_config_path),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "keys": sorted(config),
        }
        spec = copy.deepcopy(source_spec)
        spec["dataset"] = copy.deepcopy(datasets[str(source_spec["dataset_key"])])
        spec["executable_config"] = new_config_entry
        spec["output_dir"] = str(run_root / "outputs" / run_id)
        spec["success_receipt"] = str(run_root / "evidence" / run_id / "success-receipt.json")
        spec["failure_receipt"] = str(run_root / "evidence" / run_id / "failure-receipt.json")
        specs[run_id] = spec
        configs[run_id] = payload
        relative = f"binding/configs/{filename}"
        payloads[relative] = payload
        inventory[relative] = {"file_sha256": new_config_entry["file_sha256"], "bytes": len(payload)}

    runtime = _mapping(source_campaign.get("runtime_parse"), "source runtime parse")
    python_path = Path(str(runtime["python_executable"]["path"])).resolve(strict=True)
    checkout = Path(str(runtime["ms_swift_checkout"])).resolve(strict=True)
    runtime_parse = v4._parse_configs(configs, python_path, checkout, runtime["package_versions"])
    return {
        "payloads": payloads,
        "datasets": datasets,
        "run_specs": specs,
        "configs": configs,
        "artifact_inventory": dict(sorted(inventory.items())),
        "runtime_parse": runtime_parse,
        "source_dataset_paths": {
            "fit": str((source_root / "binding/data/day23-qwen35-dpo-rsi-v0005-fit.jsonl").resolve(strict=True)),
            "search": str((source_root / "binding/data/day23-qwen35-dpo-rsi-v0005-search.jsonl").resolve(strict=True)),
        },
    }


def _correction_artifacts(
    source: Mapping[str, Any],
    run_root: Path,
    prepared: Mapping[str, Any],
    producers: Mapping[str, Path],
) -> dict[str, Any]:
    bootcamp: Path = source["bootcamp"]
    control_root = bootcamp / CONTROL_ROOT_REL
    producer_ids = {name: _identity(path) for name, path in sorted(producers.items())}
    event6 = source["event6"]
    event6_path: Path = source["event6_path"]
    created_at = str(event6.get("created_at_utc"))
    _require(created_at and created_at != "None", "event6 time anchor missing")

    diagnosed = {
        "classification": "failure.control.validator_projection",
        "check": "source_strict_preunseal_amendment_identity_exact_equality",
        "lhs_with_bytes": source["observed_projection"],
        "rhs_without_bytes": source["canonical_projection"],
        "exact_difference": {
            "field": "bytes",
            "lhs_value": source["observed_projection"]["bytes"],
            "rhs_field_present": False,
        },
        "common_projection_equal": True,
        "error": "v0005 amendment3 predecessor authority drifted",
        "optimizer_started": False,
        "search_claimed": False,
        "dev_claimed": False,
        "heldout_claimed": False,
    }
    frozen_scientific = {
        "version_id": VERSION_ID,
        "learning_rates": [3.9e-6, 4.1e-6],
        "checkpoint_steps": [18, 20],
        "search_max_steps": 30,
        "search_seed": 20260819,
        "refit_seed": 20260820,
        "datasets": {
            key: {
                field: prepared["datasets"][key][field]
                for field in ("file_sha256", "records", "ordered_pair_ids_sha256", "ordered_row_hashes_sha256")
            }
            for key in ("fit", "search", "full_train")
        },
        "runtime": {
            "world_size": 2,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "nominal_global_train_batch_size": 32,
            "minimum_free_memory_fraction": 0.20,
        },
        "fresh_promoted_s1_only": True,
        "candidate_registry": copy.deepcopy(source["campaign"]["candidate_registry"]),
        "search_eligibility": copy.deepcopy(source["campaign"]["protocol"]["search_selection"]["eligibility"]),
    }
    amendment_path = control_root / f"amendments/{AMENDMENT5_NAME}.json"
    amendment = v4._write_or_verify(
        amendment_path,
        {
            "schema_name": "rsi.day23_dpo_append_only_amendment",
            "schema_version": 1,
            "status": "frozen",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "amendment_id": AMENDMENT5_NAME,
            "created_at_utc": created_at,
            "authority": {
                "charter_sha256": source["charter"]["charter_sha256"],
                "prior_event_000006": copy.deepcopy(source["campaign"]["control_extension"]["precommit_event"]),
                "source_campaign": source["campaign_identity"],
                "frozen_validator": copy.deepcopy(source["campaign"]["producers"]["authority_validator"]),
            },
            "diagnosed_failure": diagnosed,
            "authorized_correction": {
                "execution_binding_only": True,
                "semantic_version_unchanged": True,
                "new_semantic_version": False,
                "source_campaign_candidate_or_resume": False,
                "correction_builder": producer_ids["builder"],
                "runner_adapter": producer_ids["runner"],
                "evaluator_adapter": producer_ids["evaluator"],
                "corrected_validator": producer_ids["authority_validator"],
            },
            "frozen_scientific_invariants": frozen_scientific,
        },
        "amendment_sha256",
    )
    amendment_id = _identity(amendment_path, content_field="amendment_sha256", relative_to=bootcamp)

    event_path = control_root / f"events/{EVENT7_NAME}.json"
    event = v4._write_or_verify(
        event_path,
        {
            "schema_name": "rsi.day23_dpo_event",
            "schema_version": 1,
            "event_id": EVENT7_NAME,
            "created_at_utc": created_at,
            "sequence": 7,
            "event_type": "execution-binding-corrected",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "state_before": "version_precommitted",
            "state_after": "version_precommitted",
            "prior_event": {
                "path": str(event6_path.relative_to(control_root)),
                "file_sha256": day23_gpu.file_sha256(event6_path),
                "event_sha256": event6["event_sha256"],
                "sequence": 6,
            },
            "authority_refs": [amendment_id],
            "payload": {
                "classification": "failure.control.validator_projection",
                "source_campaign": source["campaign_identity"],
                "optimizer_steps_started": 0,
                "search_rows_opened": 0,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
                "semantic_version_unchanged": True,
                "scientific_configuration_unchanged": True,
                "corrected_execution_binding_pending": True,
            },
        },
        "event_sha256",
    )
    return {
        "preoptimizer_correction_amendment": amendment_id,
        "execution_binding_correction_event": _identity(event_path, content_field="event_sha256", relative_to=bootcamp),
        "superseded_campaign": source["campaign_identity"],
        "diagnosed_failure": diagnosed,
    }


def _campaign(
    source: Mapping[str, Any],
    run_root: Path,
    prepared: Mapping[str, Any],
    correction: Mapping[str, Any],
    producers: Mapping[str, Path],
) -> dict[str, Any]:
    campaign = copy.deepcopy(source["campaign"])
    campaign.pop("campaign_sha256", None)
    campaign.update({
        "campaign_id": CAMPAIGN_ID,
        "binding_revision_id": BINDING_REVISION_ID,
        "remote_run_root": str(run_root),
        "datasets": prepared["datasets"],
        "run_specs": prepared["run_specs"],
        "runtime_parse": prepared["runtime_parse"],
        "artifact_inventory": prepared["artifact_inventory"],
        "producers": {name: _identity(path) for name, path in sorted(producers.items())},
        "supersedes_preoptimizer": {
            "campaign": correction["superseded_campaign"],
            "status": "preoptimizer_control_failure",
            "classification": "failure.control.validator_projection",
            "optimizer_steps_started": 0,
            "search_claimed": False,
            "dev_claimed": False,
            "heldout_claimed": False,
            "candidate_or_resume": False,
        },
    })
    extension = copy.deepcopy(campaign["control_extension"])
    extension.update({
        "preoptimizer_correction_amendment": correction["preoptimizer_correction_amendment"],
        "execution_binding_correction_event": correction["execution_binding_correction_event"],
        "superseded_campaign": correction["superseded_campaign"],
    })
    extension["strict_preunseal_requirement"] = {
        "required_before_first_optimizer": True,
        "require_search_unopened": True,
        "validator": _identity(producers["authority_validator"]),
        "receipt_path": str(run_root / "evidence/authority/strict-preunseal.json"),
        "receipt_schema_name": "day23.rsi_v0005_strict_authority_preflight",
        "receipt_self_hash_field": "validation_sha256",
        "amendment_path": str(AMENDMENT4_REL),
        "amendment_id": "amendment-000004-v0005-strict-preunseal",
        "amendment_self_hash_field": "amendment_sha256",
    }
    campaign["control_extension"] = extension
    bootcamp: Path = source["bootcamp"]
    campaign["implementation_sources"].update({
        "builder_and_binder": {**_identity(producers["builder"], relative_to=bootcamp), "path": str(producers["builder"].relative_to(bootcamp))},
        "training_runner": {**_identity(producers["runner"], relative_to=bootcamp), "path": str(producers["runner"].relative_to(bootcamp))},
        "candidate_evaluator": {**_identity(producers["evaluator"], relative_to=bootcamp), "path": str(producers["evaluator"].relative_to(bootcamp))},
        "authority_validator": {**_identity(producers["authority_validator"], relative_to=bootcamp), "path": str(producers["authority_validator"].relative_to(bootcamp))},
    })
    campaign["access_ledger"]["search_claim"] = str(run_root / "evidence/search-selection/search-claim.json")
    campaign["claim_boundary"].update({
        "optimizer_step_run": False,
        "search_or_dev_consumed": False,
        "heldout_consumed": False,
        "preoptimizer_binding_revision": True,
        "source_preoptimizer_campaign_executed": False,
    })
    campaign["retry_policy"].update({
        "binding_revision": "same_semantics_preoptimizer_authority_projection_correction",
        "superseded_preoptimizer_campaign": "never_executable_never_candidate",
    })
    campaign["campaign_sha256"] = day23_gpu.object_sha256(campaign)
    return campaign


def _prepared(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_state(args.source_campaign, allow_root2_strict=args.mode == "check")
    run_root = args.run_root.expanduser().resolve(strict=True)
    _require(run_root.is_dir() and not run_root.is_symlink(), "corrected run root must be a real directory")
    if args.mode != "check":
        _require(not any(run_root.iterdir()), "corrected run root must be empty")
    prepared = _prepare_payloads(source, run_root)
    producers = _producer_paths()
    return {"source": source, "run_root": run_root, "prepared": prepared, "producers": producers}


def _check(value: Mapping[str, Any]) -> dict[str, Any]:
    run_root: Path = value["run_root"]
    expected = value["prepared"]
    campaign_path = run_root / "binding/gpu-campaign.json"
    campaign = _load(campaign_path, "corrected campaign")
    campaign_sha = _verify(campaign, "campaign_sha256", "corrected campaign")
    _require(campaign.get("campaign_id") == CAMPAIGN_ID and campaign.get("binding_revision_id") == BINDING_REVISION_ID, "corrected campaign identity drifted")
    for relative, entry in expected["artifact_inventory"].items():
        path = run_root / relative
        _validate_file_identity(path, entry, f"corrected artifact {relative}")
    extension = _mapping(campaign.get("control_extension"), "corrected control extension")
    bootcamp: Path = value["source"]["bootcamp"]
    _resolve_bound(bootcamp, extension.get("preoptimizer_correction_amendment"), "amendment_sha256", "correction amendment5")
    _resolve_bound(bootcamp, extension.get("execution_binding_correction_event"), "event_sha256", "correction event7")
    _require(extension.get("superseded_campaign") == value["source"]["campaign_identity"], "corrected campaign source binding drifted")
    return {"status": "pass", "campaign": str(campaign_path), "campaign_sha256": campaign_sha, "binding_revision_id": BINDING_REVISION_ID}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight_only:
        args.mode = "preflight"
    value = _prepared(args)
    source = value["source"]
    prepared = value["prepared"]
    producers = value["producers"]
    if args.mode == "preflight":
        return {
            "status": "preflight_pass",
            "version_id": VERSION_ID,
            "campaign_id": CAMPAIGN_ID,
            "binding_revision_id": BINDING_REVISION_ID,
            "source_campaign_sha256": source["campaign_sha256"],
            "classification": "failure.control.validator_projection",
            "only_extra_field": "bytes",
            "optimizer_steps": 0,
            "search_rows_opened": 0,
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
            "fit_file_sha256": prepared["datasets"]["fit"]["file_sha256"],
            "search_file_sha256": prepared["datasets"]["search"]["file_sha256"],
            "configs_parsed": len(prepared["configs"]),
            "producer_file_sha256": {name: day23_gpu.file_sha256(path) for name, path in sorted(producers.items())},
        }
    if args.mode == "check":
        return _check(value)

    correction = _correction_artifacts(source, value["run_root"], prepared, producers)
    campaign = _campaign(source, value["run_root"], prepared, correction, producers)
    for relative, payload in prepared["payloads"].items():
        v4._write_exclusive(value["run_root"] / relative, payload)
    campaign_path = value["run_root"] / "binding/gpu-campaign.json"
    v4._write_exclusive(campaign_path, _bytes(campaign))
    return {
        "status": "gpu_execution_bound_optimizer_pending",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "version_id": VERSION_ID,
        "campaign_id": CAMPAIGN_ID,
        "binding_revision_id": BINDING_REVISION_ID,
        "superseded_preoptimizer_campaign_sha256": source["campaign_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="build")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = execute(build_parser().parse_args(argv))
    except Exception as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
