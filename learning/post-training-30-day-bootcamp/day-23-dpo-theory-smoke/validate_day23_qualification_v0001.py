#!/usr/bin/env python3
"""Strict pre-unseal authority validator for Day 23 qual-v0001.

The validator independently recomputes the campaign and source csearch
selection, validates the full154 optimizer corpus, and proves that neither the
global dev lease nor heldout lease was claimed.  It never opens dev or heldout
rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_day23_qualification_v0001 as builder
import run_day23_qwen35_gpu_stage as day23_gpu


class QualificationAuthorityError(RuntimeError):
    """The qualification campaign no longer has strict pre-unseal authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationAuthorityError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationAuthorityError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def identity(path: Path, *, self_field: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if self_field is not None:
        result["content_sha256"] = verify(load(resolved, str(resolved)), self_field, str(resolved))
    return result


def _bound_path(entry: Any, label: str, *, bootcamp: Path | None = None) -> Path:
    bound = mapping(entry, label)
    raw = Path(str(bound.get("path")))
    path = raw if raw.is_absolute() else (bootcamp / raw if bootcamp else raw)
    resolved = path.resolve(strict=True)
    if bootcamp is not None:
        require(resolved.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    require(
        day23_gpu.file_sha256(resolved) == bound.get("file_sha256")
        and resolved.stat().st_size == bound.get("bytes"),
        f"{label} file identity drifted",
    )
    content = bound.get("content_sha256")
    if content is not None:
        value = load(resolved, label)
        fields = [
            field
            for field in (
                "campaign_sha256",
                "selection_sha256",
                "charter_sha256",
                "event_sha256",
            )
            if field in value
            and value.get(field) == content
            and day23_gpu.object_sha256(value, field) == content
        ]
        require(len(fields) == 1, f"{label} content identity drifted")
    return resolved


def _producer_identity(campaign: Mapping[str, Any], name: str) -> dict[str, Any]:
    entry = mapping(mapping(campaign.get("producers"), "campaign producers").get(name), f"producer {name}")
    path = _bound_path(entry, f"producer {name}")
    return {"path": str(path), "file_sha256": entry["file_sha256"], "bytes": entry["bytes"]}


def _claims(campaign: Mapping[str, Any]) -> tuple[Path, Path]:
    bootcamp = Path(str(campaign.get("bootcamp_root"))).resolve(strict=True)
    access = mapping(campaign.get("access_ledger"), "campaign access ledger")
    dev_claim = Path(str(access.get("dev_claim"))).resolve()
    dev_lease = mapping(access.get("dev_lease"), "campaign dev lease")
    dev_relative = Path(str(dev_lease.get("claim_path")))
    require(not dev_relative.is_absolute() and ".." not in dev_relative.parts, "dev claim is not canonical")
    require(dev_claim == (bootcamp / "rsi-control" / dev_relative).resolve(), "dev claim escaped global ledger")

    charter_path = _bound_path(
        mapping(campaign.get("control"), "campaign control").get("charter"),
        "qualification charter",
        bootcamp=bootcamp,
    )
    charter = load(charter_path, "qualification charter")
    verify(charter, "charter_sha256", "qualification charter")
    source_charter_entry = mapping(mapping(charter.get("source_candidate"), "charter source candidate").get("source_charter"), "source csearch charter")
    source_charter_path = _bound_path(source_charter_entry, "source csearch charter", bootcamp=bootcamp)
    source_charter = load(source_charter_path, "source csearch charter")
    verify(source_charter, "charter_sha256", "source csearch charter")
    heldout_lease = mapping(
        mapping(source_charter.get("protected_data"), "source protected data").get("heldout"),
        "source heldout lease",
    )
    heldout_relative = Path(str(heldout_lease.get("claim_path")))
    require(not heldout_relative.is_absolute() and ".." not in heldout_relative.parts, "heldout claim is not canonical")
    require(heldout_lease.get("current_authorized_claims") == 0, "heldout gained authorization")
    heldout_claim = (bootcamp / "rsi-control" / heldout_relative).resolve()
    return dev_claim, heldout_claim


def validate(
    campaign_path: Path, *, require_dev_unopened: bool
) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = load(campaign_path, "qualification campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "qualification campaign")
    require(
        campaign.get("schema_name") == builder.CAMPAIGN_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("goal_id") == builder.GOAL_ID
        and campaign.get("version_id") == builder.VERSION_ID
        and campaign.get("campaign_id") == builder.CAMPAIGN_ID,
        "qualification campaign identity drifted",
    )
    run_root = Path(str(campaign.get("remote_run_root"))).resolve(strict=True)
    bootcamp = Path(str(campaign.get("bootcamp_root"))).resolve(strict=True)
    require(campaign_path == run_root / "binding/gpu-campaign.json", "qualification campaign is not canonical")

    control = mapping(campaign.get("control"), "campaign control")
    source_campaign_path = _bound_path(control.get("source_search_campaign"), "source search campaign")
    source_selection_path = _bound_path(control.get("source_search_selection"), "source search selection")
    args = argparse.Namespace(
        source_campaign=source_campaign_path,
        source_selection=source_selection_path,
        run_root=run_root,
        mode="check",
    )
    prepared = builder._prepared(args)
    expected_campaign = builder._campaign(prepared)
    require(campaign == expected_campaign, "qualification campaign failed exact independent recomputation")

    producers = mapping(campaign.get("producers"), "campaign producers")
    require(set(producers) == {"authority_validator", "builder", "evaluator", "runner"}, "producer registry drifted")
    actual_validator = Path(__file__).resolve(strict=True)
    validator_bound = mapping(producers.get("authority_validator"), "bound authority validator")
    require(
        Path(str(validator_bound.get("path"))).resolve(strict=True) == actual_validator
        and day23_gpu.file_sha256(actual_validator) == validator_bound.get("file_sha256"),
        "campaign bound a different authority validator",
    )
    dependencies = mapping(campaign.get("runtime_dependencies"), "runtime dependencies")
    require(set(dependencies) == set(builder._runtime_dependency_paths()), "runtime dependency closure drifted")
    for name, path in builder._runtime_dependency_paths().items():
        entry = mapping(dependencies.get(name), f"runtime dependency {name}")
        require(
            Path(str(entry.get("path"))).resolve(strict=True) == path.resolve(strict=True)
            and day23_gpu.file_sha256(path.resolve(strict=True)) == entry.get("file_sha256")
            and path.stat().st_size == entry.get("bytes"),
            f"runtime dependency drifted: {name}",
        )

    specs = mapping(campaign.get("run_specs"), "campaign run specs")
    require(set(specs) == {builder.RUN_ID}, "qualification run registry is not singleton")
    spec = mapping(specs.get(builder.RUN_ID), "qualification run spec")
    require(
        spec.get("role") == builder.RUN_ROLE
        and spec.get("authorized") is True
        and spec.get("dataset_key") == "full_train"
        and spec.get("learning_rate") == builder.LEARNING_RATE
        and spec.get("seed") == builder.REFIT_SEED
        and spec.get("max_steps") == builder.MAX_STEPS
        and spec.get("cosine_schedule_horizon") == builder.MAX_STEPS
        and spec.get("checkpoint_steps") == [builder.CANDIDATE_STEP]
        and spec.get("checkpoint_capture") == builder.CHECKPOINT_CAPTURE
        and spec.get("fresh_start_from_parent") is True
        and spec.get("source_search_checkpoint_as_initialization_forbidden") is True,
        "qualification run spec drifted",
    )
    config_entry = mapping(spec.get("executable_config"), "qualification config identity")
    config_path = _bound_path(config_entry, "qualification executable config")
    config = load(config_path, "qualification executable config")
    full_train = mapping(mapping(campaign.get("datasets"), "campaign datasets").get("full_train"), "full154 dataset")
    full_path = Path(str(full_train.get("path"))).resolve(strict=True)
    rows = builder.validate_dataset(full_path, full_train, "full154")
    require(
        len(rows) == 154
        and config.get("dataset") == [str(full_path)]
        and config.get("model") == campaign.get("parent", {}).get("model_argument")
        and config.get("learning_rate") == builder.LEARNING_RATE
        and config.get("seed") == builder.REFIT_SEED
        and config.get("data_seed") == builder.REFIT_SEED
        and config.get("max_steps") == builder.MAX_STEPS
        and config.get("lr_scheduler_type") == "cosine"
        and config.get("save_steps") == builder.CANDIDATE_STEP
        and config.get("save_total_limit") == 2
        and config.get("per_device_train_batch_size") == builder.PER_DEVICE_BATCH
        and config.get("gradient_accumulation_steps") == builder.GRADIENT_ACCUMULATION,
        "qualification executable recipe drifted",
    )
    for forbidden in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset"):
        require(forbidden not in config, f"forbidden optimizer key present: {forbidden}")
    search_checkpoint = str(
        mapping(campaign.get("source_candidate"), "campaign source candidate")
        .get("search_checkpoint_evidence", {})
        .get("path", "")
    )
    require(search_checkpoint and search_checkpoint not in json.dumps(config, sort_keys=True), "search checkpoint became optimizer input")
    require(campaign.get("parent", {}).get("resume_from_source_or_search_checkpoint") is False, "parent resume policy drifted")

    runtime = mapping(campaign.get("fixed_recipe"), "fixed recipe")
    runtime_profile = mapping(runtime.get("runtime"), "fixed runtime")
    require(
        runtime.get("fresh_s1_only") is True
        and runtime.get("source_search_checkpoint_as_initialization_forbidden") is True
        and runtime.get("max_steps") == 30
        and runtime.get("cosine_schedule_horizon") == 30
        and runtime_profile
        == {
            "world_size": 2,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "nominal_global_train_batch_size": 32,
            "min_free_memory_fraction": 0.2,
        },
        "fixed qualification recipe drifted",
    )
    protocol = mapping(campaign.get("protocol"), "qualification protocol")
    dev_protocol = mapping(protocol.get("dev"), "dev protocol")
    heldout_protocol = mapping(protocol.get("heldout"), "heldout protocol")
    require(
        dev_protocol.get("maximum_claims") == 1
        and dev_protocol.get("maximum_evaluations") == 1
        and dev_protocol.get("records") == 17
        and dev_protocol.get("minimum_positive_pairs") == 13
        and dev_protocol.get("retry_after_claim") == "forbidden"
        and dev_protocol.get("runner_up_fallback") == "forbidden"
        and dev_protocol.get("success_state") == "dev_qualified_guardrails_pending"
        and heldout_protocol.get("maximum_claims") == 0
        and heldout_protocol.get("maximum_evaluations") == 0
        and heldout_protocol.get("authorized") is False
        and heldout_protocol.get("runtime_path_disclosed") is False,
        "qualification protected-data protocol drifted",
    )
    require("heldout" not in mapping(campaign.get("datasets"), "campaign datasets"), "heldout dataset was disclosed")

    dev_claim, heldout_claim = _claims(campaign)
    require(not heldout_claim.exists(), "global heldout lease was already claimed")
    if require_dev_unopened:
        require(not dev_claim.exists(), "global dev lease was already claimed")
        output_dir = Path(str(spec.get("output_dir"))).resolve()
        success = Path(str(spec.get("success_receipt"))).resolve()
        failure = Path(str(spec.get("failure_receipt"))).resolve()
        require(not output_dir.exists(), "qualification optimizer output already exists")
        require(not success.exists() and not failure.exists(), "qualification run already has a terminal receipt")
        require(not Path(str(campaign["access_ledger"]["dev_evaluation"])).exists(), "dev evaluation already exists")
        require(not Path(str(campaign["access_ledger"]["dev_failure_after_claim"])).exists(), "dev failure receipt already exists")

    requirement = mapping(control.get("strict_preunseal_requirement"), "strict pre-unseal requirement")
    expected_output = Path(str(requirement.get("receipt_path"))).resolve()
    require(
        requirement.get("required_before_optimizer") is True
        and requirement.get("require_dev_unopened") is True
        and requirement.get("require_heldout_unopened") is True
        and requirement.get("receipt_schema_name") == builder.STRICT_SCHEMA
        and requirement.get("receipt_self_hash_field") == "validation_sha256"
        and expected_output == run_root / "evidence/authority/strict-preunseal.json",
        "strict pre-unseal requirement drifted",
    )
    document: dict[str, Any] = {
        "schema_name": builder.STRICT_SCHEMA,
        "schema_version": 1,
        "status": "pass",
        "goal_id": builder.GOAL_ID,
        "version_id": builder.VERSION_ID,
        "campaign_id": builder.CAMPAIGN_ID,
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": day23_gpu.file_sha256(campaign_path),
            "campaign_sha256": campaign_sha,
        },
        "producer": _producer_identity(campaign, "authority_validator"),
        "source_authority": {
            "source_campaign_sha256": builder.goal_init.SOURCE_CAMPAIGN_SHA256,
            "source_selection_sha256": builder.goal_init.SOURCE_SELECTION_SHA256,
            "source_selection_recomputed": True,
            "source_state": "candidate_frozen_search_only",
            "selected_candidate": builder.goal_init.SELECTED_CANDIDATE,
            "selected_learning_rate": builder.LEARNING_RATE,
            "selected_checkpoint_step": builder.CANDIDATE_STEP,
        },
        "campaign_recomputed_exactly": True,
        "producer_hashes_recomputed": len(producers),
        "runtime_dependency_hashes_recomputed": len(dependencies),
        "full_train_rows_validated": len(rows),
        "fresh_s1_refit_only": True,
        "search_checkpoint_initialization_forbidden": True,
        "world_size": builder.WORLD_SIZE,
        "nominal_global_train_batch_size": builder.GLOBAL_BATCH,
        "dev_unopened_at_preflight": True,
        "heldout_unopened_at_preflight": True,
        "dev_rows_opened": 0,
        "heldout_rows_opened": 0,
        "optimizer_unopened_at_preflight": True,
    }
    return document


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    require("validation_sha256" not in sealed, "strict receipt is already sealed")
    sealed["validation_sha256"] = day23_gpu.object_sha256(sealed)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return sealed


def _self_test() -> None:
    document = {
        "schema_name": builder.STRICT_SCHEMA,
        "status": "pass",
        "dev_rows_opened": 0,
        "heldout_rows_opened": 0,
    }
    sealed = dict(document)
    sealed["validation_sha256"] = day23_gpu.object_sha256(sealed)
    require(
        day23_gpu.object_sha256(sealed, "validation_sha256") == sealed["validation_sha256"],
        "strict receipt self hash failed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-dev-unopened", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {"status": "pass", "scope": "stdlib_self_test"}
        else:
            require(args.campaign is not None, "--campaign is required")
            require(args.output is not None, "--output is required")
            require(args.require_dev_unopened, "strict pre-unseal requires --require-dev-unopened")
            document = validate(args.campaign, require_dev_unopened=True)
            campaign_value = load(args.campaign.resolve(strict=True), "campaign")
            control = mapping(campaign_value.get("control"), "campaign control")
            requirement = mapping(
                control.get("strict_preunseal_requirement"),
                "strict pre-unseal requirement",
            )
            expected = Path(str(requirement.get("receipt_path"))).resolve()
            output = args.output.expanduser().resolve()
            require(output == expected, "strict receipt output path drifted")
            sealed = _write_exclusive(output, document)
            result = {
                "status": "pass",
                "output": str(output),
                "validation_sha256": sealed["validation_sha256"],
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            }
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
