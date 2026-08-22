#!/usr/bin/env python3
"""Seal the Day 23 RSI-v0004 B8/GA2 capacity correction on GPU.

This builder never reads search, dev, or heldout rows.  It preserves the
v0003 fit/search bytes and scientific grid, binds the terminal B16 memory-gate
failure, emits an append-only control extension, and creates fresh executable
configs under a new run root.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract as contract
import run_day23_qwen35_gpu_stage as day23_gpu


VERSION_ID = "rsi-v0004"
CAMPAIGN_ID = "day23-qwen35-dpo-rsi-v0004"
GOAL_ID = "goal-0002-day23-dpo"
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
WORLD_SIZE = 2
MIN_FREE_FRACTION = 0.20
SOURCE_VERSION = "rsi-v0003"
SOURCE_SCHEMA = "day23.rsi_v0003_gpu_campaign"
GPU_SCHEMA = "day23.rsi_v0004_gpu_campaign"
CONTROL_ROOT_REL = Path("rsi-control/charters/goal-0002-day23-dpo")


class V0004BuildError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0004BuildError(message)


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V0004BuildError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(isinstance(expected, str) and len(expected) == 64, f"{label} self hash missing")
    _require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def _bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _identity(path: Path, *, content_field: str | None = None, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"missing regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if content_field:
        value = _load(resolved, str(resolved))
        result["content_sha256"] = _verify(value, content_field, str(resolved))
    return result


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _write_or_verify(path: Path, value: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed[field] = day23_gpu.object_sha256(sealed)
    payload = _bytes(sealed)
    if path.exists():
        _require(path.read_bytes() == payload, f"immutable control artifact already differs: {path}")
    else:
        _write_exclusive(path, payload)
    return sealed


def _control_artifacts(
    *,
    bootcamp: Path,
    source_campaign_path: Path,
    source_campaign: Mapping[str, Any],
    failure_path: Path,
    failure: Mapping[str, Any],
    producer_paths: Mapping[str, Path],
) -> dict[str, Any]:
    root = bootcamp / CONTROL_ROOT_REL
    charter_path = root / "charter.json"
    event2_path = root / "events/event-000002-version-precommitted.json"
    charter = _load(charter_path, "v0003 charter")
    event2 = _load(event2_path, "v0003 event2")
    charter_sha = _verify(charter, "charter_sha256", "v0003 charter")
    event2_sha = _verify(event2, "event_sha256", "v0003 event2")
    producer_ids = {
        name: _identity(path, relative_to=bootcamp)
        for name, path in sorted(producer_paths.items())
    }
    sealed_at = str(failure.get("completed_at_utc"))
    _require(bool(sealed_at) and sealed_at != "None", "source failure completion time missing")
    amendment_path = root / "amendments/amendment-000001-v0004-capacity.json"
    amendment = _write_or_verify(
        amendment_path,
        {
            "schema_name": "rsi.day23_dpo_append_only_amendment",
            "schema_version": 1,
            "status": "frozen",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "amendment_id": "amendment-000001-v0004-capacity",
            "created_at_utc": sealed_at,
            "authority": {"charter_sha256": charter_sha, "event_000002_sha256": event2_sha},
            "source_runtime_failure": {
                "campaign_path": str(source_campaign_path),
                "campaign_file_sha256": day23_gpu.file_sha256(source_campaign_path),
                "campaign_sha256": source_campaign["campaign_sha256"],
                "receipt_path": str(failure_path),
                "receipt_file_sha256": day23_gpu.file_sha256(failure_path),
                "receipt_sha256": failure["receipt_sha256"],
                "classification": "failure.runtime.hardware.capacity",
                "observed_minimum_free_memory_fraction": failure["observations"]["memory"]["minimum_observed_device_free_fraction"],
            },
            "authorized_transition": [
                "version_precommitted",
                "version_runtime_failed",
                "iteration_open",
                "version_precommitted",
            ],
            "authorized_change": {
                "per_device_train_batch_size": [16, PER_DEVICE_BATCH],
                "gradient_accumulation_steps": [1, GRADIENT_ACCUMULATION],
                "world_size": WORLD_SIZE,
                "nominal_global_train_batch_size": WORLD_SIZE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION,
                "minimum_free_memory_fraction": MIN_FREE_FRACTION,
            },
            "frozen_invariants": {
                "datasets_and_order": "byte_exact_from_v0003",
                "learning_rates": [1e-6, 5e-6],
                "search_seed": 20260819,
                "refit_seed": 20260820,
                "checkpoint_steps": [15, 30],
                "fresh_s1_only": True,
                "source_failure_checkpoint_candidate_or_resume": False,
                "dev_and_heldout_leases_unchanged": True,
            },
            "producers": producer_ids,
        },
        "amendment_sha256",
    )
    amendment_id = _identity(amendment_path, content_field="amendment_sha256", relative_to=bootcamp)

    event3_path = root / "events/event-000003-version-runtime-failed.json"
    event3 = _write_or_verify(
        event3_path,
        {
            "schema_name": "rsi.day23_dpo_event",
            "schema_version": 1,
            "event_id": "event-000003-version-runtime-failed",
            "created_at_utc": sealed_at,
            "sequence": 3,
            "event_type": "version-runtime-failed",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": SOURCE_VERSION,
            "state_before": "version_precommitted",
            "state_after": "iteration_open",
            "prior_event": {
                "path": str(event2_path.relative_to(root)),
                "file_sha256": day23_gpu.file_sha256(event2_path),
                "event_sha256": event2_sha,
                "sequence": 2,
            },
            "authority_refs": [amendment_id],
            "payload": {
                "failure_receipt": amendment["source_runtime_failure"],
                "optimizer_steps_completed": 30,
                "candidate_eligible": False,
                "checkpoint_resume_forbidden": True,
                "search_rows_opened": 0,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    event3_id = _identity(event3_path, content_field="event_sha256", relative_to=bootcamp)

    source_version_path = root / "versions/rsi-v0003/version.json"
    source_version = _load(source_version_path, "v0003 version")
    _verify(source_version, "version_sha256", "v0003 version")
    version = copy.deepcopy(source_version)
    version.pop("version_sha256", None)
    version.update(
        {
            "version_id": VERSION_ID,
            "status": "precommitted",
            "created_at_utc": sealed_at,
            "intervention": {
                "primary_lever_id": "model.optimization.batch.per_device",
                "change": "B16/GA1 to B8/GA2 after terminal memory headroom failure",
                "single_semantic_change": True,
                "scientific_lr_grid_inherited_unchanged": True,
            },
            "fixed_operational_constraint": {
                "world_size": WORLD_SIZE,
                "per_device_train_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                "nominal_global_train_batch_size": 32,
                "minimum_free_memory_fraction": MIN_FREE_FRACTION,
            },
            "fresh_start": {
                "from_promoted_s1": True,
                "resume_from_v0003_checkpoint": False,
                "source_failure_checkpoint_candidate_eligible": False,
            },
            "supersedes": amendment["source_runtime_failure"],
            "amendment": amendment_id,
        }
    )
    version_path = root / "versions/rsi-v0004/version.json"
    version = _write_or_verify(version_path, version, "version_sha256")
    version_id = _identity(version_path, content_field="version_sha256", relative_to=bootcamp)

    event4_path = root / "events/event-000004-version-precommitted.json"
    event3_identity = _identity(event3_path, content_field="event_sha256", relative_to=bootcamp)
    event4 = _write_or_verify(
        event4_path,
        {
            "schema_name": "rsi.day23_dpo_event",
            "schema_version": 1,
            "event_id": "event-000004-version-precommitted",
            "created_at_utc": sealed_at,
            "sequence": 4,
            "event_type": "version-precommitted",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "state_before": "iteration_open",
            "state_after": "version_precommitted",
            "prior_event": {
                "path": str(event3_path.relative_to(root)),
                "file_sha256": event3_identity["file_sha256"],
                "event_sha256": event3["event_sha256"],
                "sequence": 3,
            },
            "authority_refs": [amendment_id, version_id],
            "payload": {
                "runtime_profile": "world2_B8_GA2_global32",
                "learning_rate_grid": [1e-6, 5e-6],
                "source_dataset_bytes_reused": True,
                "training_started": False,
                "search_rows_opened": 0,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    return {
        "amendment": amendment_id,
        "runtime_failure_event": event3_id,
        "version": version_id,
        "precommit_event": _identity(event4_path, content_field="event_sha256", relative_to=bootcamp),
    }


def _parse_configs(configs: Mapping[str, bytes], python_path: Path, checkout: Path, expected_packages: Mapping[str, Any]) -> dict[str, Any]:
    script = r'''
import dataclasses, importlib.metadata, json, os, platform, sys
from pathlib import Path
import swift
from swift.arguments import RLHFArguments
from swift.cli.main import parse_yaml_args
from swift.utils import parse_args
fields={f.name for f in dataclasses.fields(RLHFArguments)}
stages={}
for raw in sys.argv[1:]:
 p=Path(raw); value=json.loads(p.read_text()); unknown=sorted(set(value)-fields)
 if unknown: raise RuntimeError(f"unknown keys: {unknown}")
 argv=[str(p)]; parse_yaml_args(argv); parsed,remaining=parse_args(RLHFArguments,argv)
 if remaining: raise RuntimeError(f"remaining: {remaining}")
 ta=parsed.training_args
 stages[p.stem]={"status":"pass","keys":sorted(value),"per_device_train_batch_size":ta.per_device_train_batch_size,"gradient_accumulation_steps":ta.gradient_accumulation_steps,"max_steps":parsed.max_steps,"learning_rate":parsed.learning_rate,"model":parsed.model,"dataset":parsed.dataset,"output_dir":parsed.output_dir,"adapters":parsed.adapters,"ref_adapters":parsed.ref_adapters,"ref_model":parsed.ref_model,"resume_from_checkpoint":parsed.resume_from_checkpoint}
packages={name:importlib.metadata.version(name) for name in json.loads(os.environ["EXPECTED_PACKAGES"])}
print(json.dumps({"python_version":platform.python_version(),"platform":platform.platform(),"swift_file":str(Path(swift.__file__).resolve()),"package_versions":packages,"stages":stages},sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="day23-rsi-v0004-parse-") as temporary:
        root = Path(temporary)
        paths = []
        for run_id, payload in sorted(configs.items()):
            value = json.loads(payload)
            value["output_dir"] = str(root / f"output-{run_id}")
            path = root / f"{run_id}.json"
            path.write_bytes(_bytes(value))
            paths.append(path)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(checkout) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["EXPECTED_PACKAGES"] = json.dumps(sorted(expected_packages))
        completed = subprocess.run([str(python_path), "-c", script, *map(str, paths)], check=False, capture_output=True, text=True, env=env)
    _require(completed.returncode == 0, "pinned v0004 config parse failed: " + completed.stderr.strip())
    parsed = json.loads(completed.stdout.splitlines()[-1])
    _require(parsed["package_versions"] == dict(expected_packages), "package versions drifted")
    _require(Path(parsed["swift_file"]).resolve().parent == checkout / "swift", "swift imported outside pinned checkout")
    for run_id, stage in parsed["stages"].items():
        _require(stage["status"] == "pass" and stage["per_device_train_batch_size"] == 8 and stage["gradient_accumulation_steps"] == 2, f"v0004 parse topology drifted: {run_id}")
        _require(stage["adapters"] == [] and stage["ref_adapters"] == [] and stage["ref_model"] is None and stage["resume_from_checkpoint"] is None, f"v0004 fresh-start parse drifted: {run_id}")
    return {
        "status": "pass",
        "scope": "real_remote_RLHFArguments_JSON_parse_no_model_weights",
        "python_executable": {"path": str(python_path), "file_sha256": day23_gpu.file_sha256(python_path), "version": parsed["python_version"]},
        "platform": parsed["platform"],
        "ms_swift_checkout": str(checkout),
        "ms_swift_commit": contract.MS_SWIFT_COMMIT,
        "ms_swift_clean": True,
        "ms_swift_import_root": str(checkout / "swift"),
        "package_versions": dict(sorted(parsed["package_versions"].items())),
        "stages": parsed["stages"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_campaign.resolve(strict=True)
    failure_path = args.source_failure.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    _require(run_root.is_dir() and not run_root.is_symlink() and not any(run_root.iterdir()), "new v0004 run root must be an empty real directory")
    source = _load(source_path, "v0003 GPU campaign")
    _verify(source, "campaign_sha256", "v0003 GPU campaign")
    _require(source.get("schema_name") == SOURCE_SCHEMA and source.get("version_id") == SOURCE_VERSION, "source campaign identity drifted")
    bootcamp = Path(source["bootcamp_root"]).resolve(strict=True)
    _require(source_path.is_relative_to(Path(source["remote_run_root"]).resolve()), "source campaign escaped its v0003 run root")
    failure = _load(failure_path, "B16 failure receipt")
    _verify(failure, "receipt_sha256", "B16 failure receipt")
    first_spec = source["run_specs"]["search_lr_1e_6"]
    _require(failure_path == Path(first_spec["failure_receipt"]).resolve(), "failure receipt is not the bound first search failure")
    _require(failure.get("status") == "fail" and failure.get("run_id") == "search_lr_1e_6", "source failure status/run drifted")
    _require(failure.get("campaign", {}).get("campaign_sha256") == source["campaign_sha256"], "source failure campaign drifted")
    _require(failure.get("observations", {}).get("global_step") == 30, "source failure did not finish 30 optimizer steps")
    observed = failure.get("observations", {}).get("memory", {}).get("minimum_observed_device_free_fraction")
    _require(isinstance(observed, (int, float)) and 0 < observed < MIN_FREE_FRACTION, "source failure is not the frozen memory-headroom failure")
    _require(failure.get("error", {}).get("message") == "observed free VRAM fell below the campaign gate", "source failure reason drifted")
    claim = failure.get("claim_boundary", {})
    _require(claim.get("scientific_result_available") is True and claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, "source failure claim boundary drifted")
    _require(not Path(source["access_ledger"]["search_claim"]).exists(), "v0003 search lease was already opened")
    _require(not Path(source["access_ledger"]["dev_claim"]).exists(), "global dev lease was already opened")
    charter = _load(bootcamp / source["charter"]["path"], "authoritative charter")
    heldout_relative = Path(charter["access_leases"]["heldout"]["claim_path"])
    _require(not (bootcamp / "rsi-control" / heldout_relative).exists(), "global heldout lease was already opened")
    for path in (Path(first_spec["success_receipt"]), Path(source["run_specs"]["search_lr_5e_6"]["success_receipt"]), Path(source["run_specs"]["search_lr_5e_6"]["failure_receipt"])):
        _require(not path.exists(), f"unexpected v0003 terminal artifact exists: {path}")

    script_dir = Path(__file__).resolve().parent
    producers = {
        "builder": Path(__file__).resolve(),
        "runner": script_dir / "run_day23_rsi_candidate.py",
        "evaluator": script_dir / "eval_day23_rsi_candidate.py",
    }
    config_payloads: dict[str, bytes] = {}
    specs: dict[str, Any] = {}
    artifact_inventory: dict[str, Any] = {}
    for run_id, source_spec in sorted(source["run_specs"].items()):
        source_config_path = Path(source_spec["executable_config"]["path"]).resolve(strict=True)
        _require(day23_gpu.file_sha256(source_config_path) == source_spec["executable_config"]["file_sha256"], f"source config drifted: {run_id}")
        config = _load(source_config_path, f"source config {run_id}")
        _require(config.get("per_device_train_batch_size") == 16 and config.get("gradient_accumulation_steps") == 1, f"source topology drifted: {run_id}")
        _require(not any(key in config for key in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset")), f"source fresh-start keys drifted: {run_id}")
        config["per_device_train_batch_size"] = PER_DEVICE_BATCH
        config["gradient_accumulation_steps"] = GRADIENT_ACCUMULATION
        config["output_dir"] = str(run_root / "outputs" / run_id)
        payload = _bytes(config)
        filename = source_config_path.name
        bound_path = run_root / "binding/configs" / filename
        config_payloads[run_id] = payload
        payload_sha = hashlib.sha256(payload).hexdigest()
        artifact_inventory[f"configs/{filename}"] = {"file_sha256": payload_sha, "bytes": len(payload)}
        spec = copy.deepcopy(source_spec)
        spec["runtime"] = {"world_size": 2, "per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "nominal_global_train_batch_size": 32}
        spec["executable_config"] = {"path": str(bound_path), "file_sha256": payload_sha, "bytes": len(payload), "keys": sorted(config)}
        spec["output_dir"] = str(run_root / "outputs" / run_id)
        spec["success_receipt"] = str(run_root / "evidence" / run_id / "success-receipt.json")
        spec["failure_receipt"] = str(run_root / "evidence" / run_id / "failure-receipt.json")
        spec["fresh_start_from_parent"] = True
        specs[run_id] = spec

    python_path = Path(source["runtime_parse"]["python_executable"]["path"]).resolve(strict=True)
    checkout = Path(source["runtime_parse"]["ms_swift_checkout"]).resolve(strict=True)
    runtime_parse = _parse_configs(config_payloads, python_path, checkout, source["runtime_parse"]["package_versions"])
    if args.preflight_only:
        return {
            "status": "preflight_pass",
            "version_id": VERSION_ID,
            "configs_parsed": len(config_payloads),
            "runtime_profile": "world2_B8_GA2_global32",
            "minimum_free_memory_fraction": MIN_FREE_FRACTION,
        }
    control = _control_artifacts(bootcamp=bootcamp, source_campaign_path=source_path, source_campaign=source, failure_path=failure_path, failure=failure, producer_paths=producers)
    campaign = copy.deepcopy(source)
    campaign.pop("campaign_sha256", None)
    campaign.update({
        "schema_name": GPU_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "version_id": VERSION_ID,
        "remote_run_root": str(run_root),
        "control_extension": control,
        "producers": {name: _identity(path) for name, path in sorted(producers.items())},
        "runtime_parse": runtime_parse,
        "run_specs": specs,
        "artifact_inventory": artifact_inventory,
        "supersedes": {
            "campaign_path": str(source_path),
            "campaign_file_sha256": day23_gpu.file_sha256(source_path),
            "campaign_sha256": source["campaign_sha256"],
            "failure_receipt_path": str(failure_path),
            "failure_receipt_file_sha256": day23_gpu.file_sha256(failure_path),
            "failure_receipt_sha256": failure["receipt_sha256"],
            "old_checkpoint_candidate_or_resume": False,
        },
    })
    campaign["fixed_recipe"]["runtime"].update({"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "nominal_global_train_batch_size": 32, "min_free_memory_fraction": MIN_FREE_FRACTION})
    campaign["fixed_recipe"]["topology"].update({"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "nominal_global_train_batch_size": 32})
    campaign["implementation_sources"].update({
        "builder_and_binder": {**_identity(producers["builder"], relative_to=bootcamp), "path": str(producers["builder"].relative_to(bootcamp))},
        "training_runner": {**_identity(producers["runner"], relative_to=bootcamp), "path": str(producers["runner"].relative_to(bootcamp))},
        "candidate_evaluator": {**_identity(producers["evaluator"], relative_to=bootcamp), "path": str(producers["evaluator"].relative_to(bootcamp))},
    })
    campaign["access_ledger"]["search_claim"] = str(run_root / "evidence/search-selection/search-claim.json")
    campaign["retry_policy"]["supersession_class"] = "new_semantic_version_not_infrastructure_retry"
    campaign["retry_policy"]["v0003_failure_checkpoint"] = "diagnostic_only_never_candidate_or_resume"
    campaign["budgets"]["cumulative_optimizer_steps_including_failed_v0003"] = 120
    campaign["claim_boundary"].update({"optimizer_step_run": False, "search_or_dev_consumed": False, "heldout_consumed": False})
    campaign["campaign_sha256"] = day23_gpu.object_sha256(campaign)

    for run_id, payload in config_payloads.items():
        _write_exclusive(Path(specs[run_id]["executable_config"]["path"]), payload)
    campaign_path = run_root / "binding/gpu-campaign.json"
    _write_exclusive(campaign_path, _bytes(campaign))
    return {"status": "gpu_execution_bound_optimizer_pending", "campaign": str(campaign_path), "campaign_sha256": campaign["campaign_sha256"], "version_id": VERSION_ID}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--source-failure", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = build(build_parser().parse_args(argv))
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
