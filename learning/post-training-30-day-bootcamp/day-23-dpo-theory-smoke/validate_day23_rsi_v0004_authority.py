#!/usr/bin/env python3
"""Strict pre-unseal validation for the Day 23 RSI-v0004 authority chain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_day23_qwen35_gpu_stage as day23_gpu


class AuthorityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def bound_file(bootcamp: Path, entry: Any, field: str, label: str) -> tuple[dict[str, Any], Path]:
    item = mapping(entry, label)
    raw = Path(str(item.get("path")))
    path = (raw if raw.is_absolute() else bootcamp / raw).resolve(strict=True)
    require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    require(day23_gpu.file_sha256(path) == item.get("file_sha256"), f"{label} file hash drifted")
    value = load(path, label)
    require(verify(value, field, label) == item.get("content_sha256"), f"{label} content binding drifted")
    return value, path


def validate(campaign_path: Path, *, require_search_unopened: bool) -> dict[str, Any]:
    campaign_path = campaign_path.resolve(strict=True)
    campaign = load(campaign_path, "v0004 campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "v0004 campaign")
    require(
        campaign.get("schema_name") == "day23.rsi_v0004_gpu_campaign"
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("goal_id") == "goal-0002-day23-dpo"
        and campaign.get("version_id") == "rsi-v0004",
        "v0004 campaign identity drifted",
    )
    bootcamp = Path(str(campaign.get("bootcamp_root"))).resolve(strict=True)
    run_root = Path(str(campaign.get("remote_run_root"))).resolve(strict=True)
    require(campaign_path.is_relative_to(run_root), "campaign escaped run root")

    producers = mapping(campaign.get("producers"), "campaign producers")
    for name in ("builder", "runner", "evaluator"):
        entry = mapping(producers.get(name), f"producer {name}")
        path = Path(str(entry.get("path"))).resolve(strict=True)
        require(path.is_relative_to(bootcamp) and day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"producer drifted: {name}")

    charter_entry = mapping(campaign.get("charter"), "campaign charter")
    charter_path = (bootcamp / str(charter_entry.get("path"))).resolve(strict=True)
    charter = load(charter_path, "charter")
    require(day23_gpu.file_sha256(charter_path) == charter_entry.get("file_sha256"), "charter file drifted")
    require(verify(charter, "charter_sha256", "charter") == charter_entry.get("content_sha256"), "charter content drifted")

    supersedes = mapping(campaign.get("supersedes"), "campaign supersedes")
    source_path = Path(str(supersedes.get("campaign_path"))).resolve(strict=True)
    source = load(source_path, "v0003 campaign")
    require(day23_gpu.file_sha256(source_path) == supersedes.get("campaign_file_sha256"), "v0003 campaign file drifted")
    require(verify(source, "campaign_sha256", "v0003 campaign") == supersedes.get("campaign_sha256"), "v0003 campaign content drifted")
    require(source.get("schema_name") == "day23.rsi_v0003_gpu_campaign" and source.get("schema_version") == 1 and source.get("version_id") == "rsi-v0003", "v0003 campaign identity drifted")
    source_spec = mapping(mapping(source.get("run_specs"), "v0003 runs").get("search_lr_1e_6"), "v0003 failed run")
    failure_path = Path(str(supersedes.get("failure_receipt_path"))).resolve(strict=True)
    require(failure_path == Path(str(source_spec.get("failure_receipt"))).resolve(strict=True), "v0003 failure path is not bound")
    failure = load(failure_path, "v0003 failure")
    require(day23_gpu.file_sha256(failure_path) == supersedes.get("failure_receipt_file_sha256"), "v0003 failure file drifted")
    require(verify(failure, "receipt_sha256", "v0003 failure") == supersedes.get("failure_receipt_sha256"), "v0003 failure content drifted")
    require(failure.get("schema_name") == "day23.rsi_v0003_training_receipt" and failure.get("schema_version") == 1 and failure.get("status") == "fail" and failure.get("run_id") == "search_lr_1e_6", "v0003 failure identity drifted")
    require(failure.get("campaign", {}).get("campaign_sha256") == source.get("campaign_sha256"), "v0003 failure campaign drifted")
    require(failure.get("producer", {}).get("file_sha256") == source.get("producers", {}).get("runner", {}).get("file_sha256"), "v0003 failure producer drifted")
    observations = mapping(failure.get("observations"), "v0003 failure observations")
    memory = mapping(observations.get("memory"), "v0003 failure memory")
    minimum = memory.get("minimum_observed_device_free_fraction")
    source_gate = source.get("fixed_recipe", {}).get("runtime", {}).get("min_free_memory_fraction")
    require(observations.get("global_step") == 30 and isinstance(minimum, (int, float)) and math.isfinite(float(minimum)) and float(minimum) < float(source_gate) == 0.30, "v0003 memory failure was not recomputed")
    require(failure.get("error", {}).get("message") == "observed free VRAM fell below the campaign gate", "v0003 failure reason drifted")
    claim = mapping(failure.get("claim_boundary"), "v0003 failure claim")
    require(claim.get("scientific_result_available") is True and claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, "v0003 failure claim drifted")
    require(failure.get("config") == {"path": source_spec["executable_config"]["path"], "file_sha256": source_spec["executable_config"]["file_sha256"]}, "v0003 failure config drifted")
    require(not Path(str(source_spec.get("success_receipt"))).exists(), "v0003 failed run also has success")
    other = mapping(source["run_specs"]["search_lr_5e_6"], "v0003 other search")
    require(not Path(str(other["success_receipt"])).exists() and not Path(str(other["failure_receipt"])).exists(), "v0003 grid unexpectedly continued")
    require(not Path(str(source["access_ledger"]["search_claim"])).exists(), "v0003 search was unsealed")

    extension = mapping(campaign.get("control_extension"), "control extension")
    amendment, _ = bound_file(bootcamp, extension.get("amendment"), "amendment_sha256", "v0004 amendment")
    event3, event3_path = bound_file(bootcamp, extension.get("runtime_failure_event"), "event_sha256", "v0004 event3")
    version, _ = bound_file(bootcamp, extension.get("version"), "version_sha256", "v0004 version")
    event4, _ = bound_file(bootcamp, extension.get("precommit_event"), "event_sha256", "v0004 event4")
    event2_path = charter_path.parent / "events/event-000002-version-precommitted.json"
    event2 = load(event2_path, "v0003 event2")
    event2_sha = verify(event2, "event_sha256", "v0003 event2")
    authority = mapping(amendment.get("authority"), "amendment authority")
    require(amendment.get("schema_name") == "rsi.day23_dpo_append_only_amendment" and amendment.get("schema_version") == 1 and amendment.get("status") == "frozen" and amendment.get("version_id") == "rsi-v0004", "amendment identity drifted")
    require(authority == {"charter_sha256": charter["charter_sha256"], "event_000002_sha256": event2_sha}, "amendment predecessor drifted")
    require(amendment.get("source_runtime_failure") == {"campaign_path": str(source_path), "campaign_file_sha256": day23_gpu.file_sha256(source_path), "campaign_sha256": source["campaign_sha256"], "receipt_path": str(failure_path), "receipt_file_sha256": day23_gpu.file_sha256(failure_path), "receipt_sha256": failure["receipt_sha256"], "classification": "failure.runtime.hardware.capacity", "observed_minimum_free_memory_fraction": minimum}, "amendment failure binding drifted")
    change = mapping(amendment.get("authorized_change"), "amendment change")
    require(change == {"per_device_train_batch_size": [16, 8], "gradient_accumulation_steps": [1, 2], "world_size": 2, "nominal_global_train_batch_size": 32, "minimum_free_memory_fraction": 0.20}, "amendment authorized change drifted")
    for name in ("builder", "runner", "evaluator"):
        require(amendment.get("producers", {}).get(name, {}).get("file_sha256") == producers[name]["file_sha256"], f"amendment producer drifted: {name}")

    prior3 = mapping(event3.get("prior_event"), "event3 prior")
    require(event3.get("schema_name") == "rsi.day23_dpo_event" and event3.get("schema_version") == 1 and event3.get("event_type") == "version-runtime-failed" and event3.get("sequence") == 3 and event3.get("goal_id") == "goal-0002-day23-dpo" and event3.get("version_id") == "rsi-v0003" and event3.get("state_before") == "version_precommitted" and event3.get("state_after") == "iteration_open", "event3 transition drifted")
    require(prior3 == {"path": str(event2_path.relative_to(charter_path.parent)), "file_sha256": day23_gpu.file_sha256(event2_path), "event_sha256": event2_sha, "sequence": 2}, "event3 predecessor drifted")
    require(event3.get("authority_refs") == [extension["amendment"]], "event3 authority drifted")
    require(event3.get("payload", {}).get("failure_receipt") == amendment["source_runtime_failure"] and event3.get("payload", {}).get("candidate_eligible") is False and event3.get("payload", {}).get("checkpoint_resume_forbidden") is True, "event3 payload drifted")

    prior4 = mapping(event4.get("prior_event"), "event4 prior")
    require(event4.get("schema_name") == "rsi.day23_dpo_event" and event4.get("schema_version") == 1 and event4.get("event_type") == "version-precommitted" and event4.get("sequence") == 4 and event4.get("goal_id") == "goal-0002-day23-dpo" and event4.get("version_id") == "rsi-v0004" and event4.get("state_before") == "iteration_open" and event4.get("state_after") == "version_precommitted", "event4 transition drifted")
    require(prior4 == {"path": str(event3_path.relative_to(charter_path.parent)), "file_sha256": day23_gpu.file_sha256(event3_path), "event_sha256": event3["event_sha256"], "sequence": 3}, "event4 predecessor drifted")
    require(event4.get("authority_refs") == [extension["amendment"], extension["version"]], "event4 authority drifted")
    operational = mapping(version.get("fixed_operational_constraint"), "v0004 operational contract")
    fresh = mapping(version.get("fresh_start"), "v0004 fresh contract")
    require(version.get("schema_version") == 1 and version.get("version_id") == "rsi-v0004" and operational == {"world_size": 2, "per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "nominal_global_train_batch_size": 32, "minimum_free_memory_fraction": 0.20}, "v0004 version topology drifted")
    require(fresh == {"from_promoted_s1": True, "resume_from_v0003_checkpoint": False, "source_failure_checkpoint_candidate_eligible": False}, "v0004 fresh-start contract drifted")

    runtime = mapping(campaign.get("fixed_recipe"), "fixed recipe").get("runtime")
    require(runtime == {**runtime, "world_size": 2, "per_device_train_batch_size": 8, "gradient_accumulation_steps": 2, "nominal_global_train_batch_size": 32, "min_free_memory_fraction": 0.20}, "campaign runtime correction drifted")
    for run_id, spec in mapping(campaign.get("run_specs"), "v0004 runs").items():
        config = load(Path(str(spec["executable_config"]["path"])), f"v0004 config {run_id}")
        require(config.get("per_device_train_batch_size") == 8 and config.get("gradient_accumulation_steps") == 2 and config.get("output_dir") == spec.get("output_dir"), f"v0004 config topology drifted: {run_id}")
        require(not any(key in config for key in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset")), f"v0004 config fresh-start drifted: {run_id}")
        require(spec.get("dataset") == source["run_specs"][run_id].get("dataset") and spec.get("learning_rate") == source["run_specs"][run_id].get("learning_rate") and spec.get("seed") == source["run_specs"][run_id].get("seed") and spec.get("checkpoint_steps") == source["run_specs"][run_id].get("checkpoint_steps"), f"v0004 scientific invariant drifted: {run_id}")

    access = mapping(campaign.get("access_ledger"), "v0004 access ledger")
    search_claim = Path(str(access.get("search_claim")))
    if require_search_unopened:
        require(not search_claim.exists(), "v0004 search was already unsealed")
    dev_claim = Path(str(access.get("dev_claim")))
    require(not dev_claim.exists(), "global dev was already unsealed")
    heldout_relative = Path(str(charter["access_leases"]["heldout"]["claim_path"]))
    require(not (bootcamp / "rsi-control" / heldout_relative).exists(), "global heldout was already unsealed")
    return {
        "schema_name": "day23.rsi_v0004_strict_authority_preflight",
        "schema_version": 1,
        "status": "pass",
        "campaign_path": str(campaign_path),
        "campaign_file_sha256": day23_gpu.file_sha256(campaign_path),
        "campaign_sha256": campaign_sha,
        "authority": {name: extension[name] for name in ("amendment", "runtime_failure_event", "version", "precommit_event")},
        "source_failure_receipt_sha256": failure["receipt_sha256"],
        "runtime_profile": "world2_B8_GA2_global32",
        "minimum_free_memory_fraction": 0.20,
        "search_unopened_at_preflight": require_search_unopened,
        "dev_unopened": True,
        "heldout_unopened": True,
        "producer": {"path": str(Path(__file__).resolve()), "file_sha256": day23_gpu.file_sha256(Path(__file__).resolve())},
    }


def seal_preunseal_amendment(campaign_path: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    campaign = load(campaign_path.resolve(strict=True), "v0004 campaign")
    bootcamp = Path(str(campaign["bootcamp_root"])).resolve(strict=True)
    charter_path = (bootcamp / str(campaign["charter"]["path"])).resolve(strict=True)
    path = charter_path.parent / "amendments/amendment-000002-v0004-strict-preunseal.json"
    value = {
        "schema_name": "rsi.day23_dpo_append_only_amendment",
        "schema_version": 1,
        "status": "frozen",
        "goal_id": "goal-0002-day23-dpo",
        "version_id": "rsi-v0004",
        "amendment_id": "amendment-000002-v0004-strict-preunseal",
        "campaign": {
            "path": str(campaign_path.resolve(strict=True)),
            "file_sha256": day23_gpu.file_sha256(campaign_path.resolve(strict=True)),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "prior_capacity_amendment": campaign["control_extension"]["amendment"],
        "strict_validator": validation["producer"],
        "required_before_first_search_claim": True,
        "requirements": [
            "strict_v0003_source_failure_recomputation",
            "event2_to_event3_to_event4_hash_chain",
            "amendment_and_version_authority_refs",
            "schema_version_one",
            "exact_producer_hashes",
            "search_dev_heldout_unopened",
        ],
    }
    expected_sha = day23_gpu.object_sha256(value)
    if path.exists():
        sealed = load(path, "strict pre-unseal amendment")
        require(verify(sealed, "amendment_sha256", "strict pre-unseal amendment") == expected_sha, "strict pre-unseal amendment drifted")
        actual = dict(sealed)
        actual.pop("amendment_sha256", None)
        require(actual == value, "strict pre-unseal amendment contents drifted")
    else:
        day23_gpu.write_sealed_json(path, value, "amendment_sha256")
        sealed = load(path, "strict pre-unseal amendment")
    return {
        "path": str(path.relative_to(bootcamp)),
        "file_sha256": day23_gpu.file_sha256(path),
        "content_sha256": sealed["amendment_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-search-unopened", action="store_true")
    args = parser.parse_args(argv)
    try:
        value = validate(args.campaign, require_search_unopened=args.require_search_unopened)
        value["strict_preunseal_amendment"] = seal_preunseal_amendment(args.campaign, value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sha = day23_gpu.write_sealed_json(args.output, value, "validation_sha256")
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "pass", "output": str(args.output), "validation_sha256": sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
