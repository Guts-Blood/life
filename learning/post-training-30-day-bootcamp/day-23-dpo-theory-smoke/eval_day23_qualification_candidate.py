#!/usr/bin/env python3
"""One-shot dev17 evaluation for the fresh Day 23 qualification refit.

This executable deliberately has no split selector and no heldout interface.
It validates the campaign and the full154 training receipt through the exact
campaign-bound qualification runner, reloads checkpoint-20 in a fresh
single-GPU process, and only then creates the inherited global dev claim with
O_EXCL.  The dev17 bytes are not opened before that claim exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import day23_contract
import eval_day23_qwen35_preferences as preference_eval
import run_day23_qwen35_gpu_stage as gpu_stage


GOAL_ID = "goal-0004-day23-dpo-qualification"
VERSION_ID = "qual-v0001"
CAMPAIGN_ID = "day23-qwen35-dpo-qualification-v0001"
CAMPAIGN_SCHEMA = "day23.qualification_v0001_gpu_campaign"
RUN_ID = "full154_refit_lr_4p3e_6_step_20"
RUN_ROLE = "unique_finalist_full154_refit"
TRAINING_RECEIPT_SCHEMA = "day23.qualification_v0001_training_receipt"
EVALUATION_SCHEMA = "day23.qualification_v0001_dev_evaluation"
CLAIM_SCHEMA = "day23.qualification_v0001_access_claim"
FAILURE_SCHEMA = "day23.qualification_v0001_dev_failure"
CHECKPOINT_STEP = 20
MAX_STEPS = 30
LEARNING_RATE = 4.3e-6
SOURCE_CANDIDATE_ID = "candidate_lr_4p3e_6_step_20"
SOURCE_RUN_ID = "search_lr_4p3e_6"
DEV_RECORDS = 17
MINIMUM_POSITIVE_PAIRS = 13
INHERITED_SCOPE_ID = (
    "day22-experimental-ai-assisted-"
    "3e812fa76a61cbeccf4f6371de56c75baa2913710ff9da05ec068d39e907f6ce"
)
INHERITED_DEV_CLAIM_RELATIVE = Path(
    f"rsi-control/access-ledger/{INHERITED_SCOPE_ID}/dev-claim.json"
)
DIRECT_DEPENDENCIES = {
    "day23_contract": Path(day23_contract.__file__).resolve(),
    "eval_day23_qwen35_preferences": Path(preference_eval.__file__).resolve(),
    "run_day23_qwen35_gpu_stage": Path(gpu_stage.__file__).resolve(),
}


class QualificationEvaluationError(RuntimeError):
    """A qualification identity, one-shot lease, or dev gate failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationEvaluationError(message)


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
    try:
        return path.resolve(strict=must_exist)
    except OSError as error:
        raise QualificationEvaluationError(f"cannot resolve {label}: {path}") from error


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationEvaluationError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QualificationEvaluationError(f"cannot read {label}: {path}") from error
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        _require(bool(line), f"blank line in {label}:{index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise QualificationEvaluationError(
                f"invalid JSON in {label}:{index}"
            ) from error
        _require(isinstance(value, dict), f"non-object row in {label}:{index}")
        rows.append(value)
    return rows


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _text(value.get(field), f"{label}.{field}")
    _require(
        len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
        and gpu_stage.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def _write_exclusive(path: Path, value: Mapping[str, Any], field: str) -> str:
    _require(path.is_absolute(), f"sealed output path must be absolute: {path}")
    _require(path.parent.is_dir(), f"sealed output parent is missing: {path.parent}")
    _require(not path.exists() and not path.is_symlink(), f"sealed output already exists: {path}")
    try:
        return gpu_stage.write_sealed_json(path, value, field)
    except (OSError, gpu_stage.Day23GPUStageError) as error:
        raise QualificationEvaluationError(f"cannot seal {path}: {error}") from error


def _identity_entry(value: Mapping[str, Any], label: str) -> tuple[Path, str]:
    path = _absolute(value.get("path"), f"{label}.path")
    expected_sha = _text(value.get("file_sha256"), f"{label}.file_sha256")
    _require(
        len(expected_sha) == 64
        and gpu_stage.file_sha256(path) == expected_sha,
        f"{label} source hash drifted",
    )
    if "bytes" in value:
        _require(path.stat().st_size == value.get("bytes"), f"{label} byte count drifted")
    return path, expected_sha


def _bound_dependency(
    campaign: Mapping[str, Any], name: str, actual_path: Path
) -> dict[str, Any]:
    dependencies = _mapping(
        campaign.get("runtime_dependencies"), "campaign.runtime_dependencies"
    )
    entry = _mapping(dependencies.get(name), f"runtime dependency {name}")
    bound_path, bound_sha = _identity_entry(entry, f"runtime dependency {name}")
    _require(
        bound_path == actual_path.resolve(strict=True),
        f"campaign bound a different runtime dependency: {name}",
    )
    return {
        "path": str(bound_path),
        "file_sha256": bound_sha,
        "bytes": bound_path.stat().st_size,
    }


def _load_bound_runner(campaign: Mapping[str, Any]) -> tuple[ModuleType, dict[str, Any]]:
    producers = _mapping(campaign.get("producers"), "campaign.producers")
    entry = _mapping(producers.get("runner"), "campaign.producers.runner")
    runner_path, runner_sha = _identity_entry(entry, "campaign runner")
    bootcamp = _absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    _require(runner_path.is_relative_to(bootcamp), "campaign runner escaped bootcamp root")
    module_name = "_day23_bound_qualification_runner"
    specification = importlib.util.spec_from_file_location(module_name, runner_path)
    _require(
        specification is not None and specification.loader is not None,
        "campaign-bound qualification runner cannot be imported",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    for name, expected in (
        ("GOAL_ID", GOAL_ID),
        ("VERSION_ID", VERSION_ID),
        ("CAMPAIGN_ID", CAMPAIGN_ID),
        ("CAMPAIGN_SCHEMA", CAMPAIGN_SCHEMA),
        ("RUN_ID", RUN_ID),
        ("RUN_ROLE", RUN_ROLE),
        ("RECEIPT_SCHEMA", TRAINING_RECEIPT_SCHEMA),
        ("MAX_STEPS", MAX_STEPS),
        ("CANDIDATE_STEP", CHECKPOINT_STEP),
        ("FULL_TRAIN_RECORDS", 154),
    ):
        _require(getattr(module, name, None) == expected, f"bound runner {name} drifted")
    _require(
        callable(getattr(module, "validate_campaign_document", None))
        and callable(getattr(module, "validate_training_receipt_value", None)),
        "campaign-bound runner verifier API is incomplete",
    )
    return module, {
        "path": str(runner_path),
        "file_sha256": runner_sha,
        "bytes": runner_path.stat().st_size,
    }


def _campaign_context(campaign_path: Path) -> dict[str, Any]:
    path = campaign_path.expanduser().resolve(strict=True)
    preliminary = _load_json(path, "qualification campaign")
    _require(preliminary.get("schema_name") == CAMPAIGN_SCHEMA, "campaign schema drifted")
    _require(preliminary.get("schema_version") == 1, "campaign schema version drifted")
    _require(preliminary.get("goal_id") == GOAL_ID, "campaign goal drifted")
    _require(preliminary.get("version_id") == VERSION_ID, "campaign version drifted")
    _require(preliminary.get("campaign_id") == CAMPAIGN_ID, "campaign ID drifted")
    _verify_self(preliminary, "campaign_sha256", "qualification campaign")
    runner, runner_identity = _load_bound_runner(preliminary)
    try:
        validated = runner.validate_campaign_document(path, require_dev_unopened=True)
    except (TypeError, ValueError, RuntimeError) as error:
        raise QualificationEvaluationError(
            f"campaign-bound runner rejected qualification campaign: {error}"
        ) from error
    _require(isinstance(validated, Mapping), "runner campaign verifier returned no context")
    campaign_value = validated.get("campaign", validated)
    campaign = _mapping(campaign_value, "runner-validated campaign")
    _require(dict(campaign) == preliminary, "runner validated different campaign bytes")
    campaign_sha = _verify_self(campaign, "campaign_sha256", "qualification campaign")
    run_root = _absolute(campaign.get("remote_run_root"), "campaign.remote_run_root")
    _require(path.is_relative_to(run_root), "qualification campaign escaped run root")
    source_candidate = _mapping(
        campaign.get("source_candidate"), "campaign.source_candidate"
    )
    _require(
        source_candidate.get("status") == "candidate_frozen_search_only"
        and source_candidate.get("candidate_id") == SOURCE_CANDIDATE_ID
        and source_candidate.get("search_run_id") == SOURCE_RUN_ID
        and float(source_candidate.get("selected_learning_rate")) == LEARNING_RATE
        and source_candidate.get("selected_checkpoint_step") == CHECKPOINT_STEP
        and source_candidate.get("search_checkpoint_as_initialization") is False
        and source_candidate.get("runner_up_fallback") is False,
        "source csearch finalist binding drifted",
    )
    datasets = _mapping(campaign.get("datasets"), "campaign.datasets")
    _require(
        set(datasets) == {"full_train", "dev"},
        "qualification campaign dataset registry drifted",
    )
    protocol = _mapping(campaign.get("protocol"), "campaign.protocol")
    dev_protocol = _mapping(protocol.get("dev"), "campaign.protocol.dev")
    _require(
        dev_protocol.get("maximum_claims") == 1
        and dev_protocol.get("maximum_evaluations") == 1
        and dev_protocol.get("records") == DEV_RECORDS
        and dev_protocol.get("minimum_positive_pairs") == MINIMUM_POSITIVE_PAIRS
        and dev_protocol.get("mean_reward_margin") == ">0"
        and dev_protocol.get("length_matched_mean_reward_margin") == ">0"
        and dev_protocol.get("retry_after_claim") == "forbidden"
        and dev_protocol.get("runner_up_fallback") == "forbidden",
        "one-shot dev qualification protocol drifted",
    )
    protected_protocol = _mapping(
        protocol.get("heldout"), "campaign.protocol protected split"
    )
    _require(
        protected_protocol.get("maximum_claims") == 0
        and protected_protocol.get("maximum_evaluations") == 0
        and protected_protocol.get("runtime_path_disclosed") is False
        and protected_protocol.get("authorized") is False,
        "post-dev protected boundary drifted",
    )
    model_path_value = validated.get("model_path")
    if model_path_value is None:
        model_path = gpu_stage._verify_remote_parent_payload(
            _mapping(campaign.get("remote_parent"), "campaign.remote_parent")
        )
    else:
        model_path = _absolute(model_path_value, "validated parent model")
    evaluator_entry = _mapping(
        _mapping(campaign.get("producers"), "campaign.producers").get("evaluator"),
        "campaign.producers.evaluator",
    )
    evaluator_path, _ = _identity_entry(evaluator_entry, "campaign evaluator")
    _require(evaluator_path == Path(__file__).resolve(strict=True), "campaign bound a different evaluator")
    bindings = {
        name: _bound_dependency(campaign, name, source)
        for name, source in sorted(DIRECT_DEPENDENCIES.items())
    }
    return {
        "campaign": campaign,
        "campaign_path": path,
        "campaign_file_sha256": gpu_stage.file_sha256(path),
        "campaign_sha256": campaign_sha,
        "run_root": run_root,
        "model_path": model_path,
        "runner": runner,
        "runner_identity": runner_identity,
        "source_bindings": bindings,
    }


def _checkpoint_from_receipt(
    context: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    path = receipt_path.expanduser().resolve(strict=True)
    receipt = _load_json(path, "qualification training receipt")
    receipt_sha = _verify_self(receipt, "receipt_sha256", "qualification training receipt")
    runner = context["runner"]
    try:
        strict = runner.validate_training_receipt_value(
            context["campaign"], context["campaign_path"], receipt, path
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise QualificationEvaluationError(
            f"campaign-bound runner rejected training receipt: {error}"
        ) from error
    _require(isinstance(strict, Mapping), "runner receipt verifier returned no projection")
    specs = _mapping(context["campaign"].get("run_specs"), "campaign.run_specs")
    spec = _mapping(specs.get(RUN_ID), f"campaign.run_specs.{RUN_ID}")
    _require(set(specs) == {RUN_ID}, "qualification campaign must bind exactly one run")
    _require(
        receipt.get("schema_name") == TRAINING_RECEIPT_SCHEMA
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "pass"
        and receipt.get("run_id") == RUN_ID,
        "qualification training receipt schema/status/run drifted",
    )
    _require(
        strict.get("run_id") == RUN_ID and strict.get("spec") == spec,
        "strict training receipt projection drifted",
    )
    campaign_link = _mapping(receipt.get("campaign"), "training receipt campaign")
    _require(
        Path(_text(campaign_link.get("path"), "receipt campaign path")).resolve()
        == context["campaign_path"]
        and campaign_link.get("file_sha256") == context["campaign_file_sha256"]
        and campaign_link.get("campaign_sha256") == context["campaign_sha256"],
        "training receipt campaign identity drifted",
    )
    _require(
        path == _absolute(spec.get("success_receipt"), "bound success receipt"),
        "training receipt path is not campaign-bound",
    )
    failure_path = Path(_text(spec.get("failure_receipt"), "bound failure receipt")).resolve()
    _require(not failure_path.exists() and not failure_path.is_symlink(), "training run has a failure receipt")
    _require(spec.get("role") == RUN_ROLE, "qualification run role drifted")
    _require(spec.get("max_steps") == MAX_STEPS, "qualification max_steps drifted")
    _require(spec.get("checkpoint_steps") == [CHECKPOINT_STEP], "qualification checkpoint inventory drifted")
    claim = _mapping(receipt.get("claim_boundary"), "training receipt claim boundary")
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
        _require(claim.get(key) is True, f"training receipt lacks {key}")
    _require(
        claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False,
        "training consumed protected data",
    )
    checkpoints = strict.get("checkpoints")
    noncandidate_checkpoints = strict.get("retained_noncandidate_checkpoints")
    _require(isinstance(checkpoints, list), "strict checkpoint inventory is absent")
    _require(
        isinstance(noncandidate_checkpoints, list)
        and len(noncandidate_checkpoints) == 1
        and noncandidate_checkpoints[0].get("global_step") == MAX_STEPS,
        "retained checkpoint-30 must remain the unique non-candidate",
    )
    matches = [
        item
        for item in checkpoints
        if isinstance(item, Mapping) and item.get("global_step") == CHECKPOINT_STEP
    ]
    _require(len(matches) == 1, "checkpoint-20 is not the unique candidate checkpoint")
    checkpoint = dict(matches[0])
    checkpoint_path = _absolute(checkpoint.get("path"), "checkpoint-20 path")
    actual = gpu_stage.checkpoint_manifest(checkpoint_path, CHECKPOINT_STEP)
    _require(actual == checkpoint, "checkpoint-20 bytes differ from strict receipt")
    process = _mapping(receipt.get("process_identity"), "training process identity")
    return {
        "receipt": receipt,
        "receipt_path": path,
        "receipt_file_sha256": gpu_stage.file_sha256(path),
        "receipt_sha256": receipt_sha,
        "spec": spec,
        "checkpoint": actual,
        "checkpoint_path": checkpoint_path,
        "runner_process_identity": dict(process),
    }


def _runtime_context(
    context: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_parse = _mapping(
        context["campaign"].get("runtime_parse"), "campaign.runtime_parse"
    )
    config_entry = _mapping(
        checkpoint["spec"].get("executable_config"), "run executable config"
    )
    config_path, _ = _identity_entry(config_entry, "run executable config")
    config = _load_json(config_path, "run executable config")
    _require(config.get("max_steps") == MAX_STEPS, "runtime config max_steps drifted")
    _require(config.get("dataset") == [context["campaign"]["datasets"]["full_train"]["path"]], "runtime config is not full154-only")
    lowered = json.dumps(config, ensure_ascii=False, sort_keys=True).lower()
    _require("heldout" not in lowered and "coding-dpo-dev" not in lowered, "protected path leaked into optimizer config")
    return {
        "ms_swift": {"path": runtime_parse.get("ms_swift_checkout")},
        "model_path": context["model_path"],
        "config": config,
        "runtime_parse": runtime_parse,
    }


def _expected_paths(context: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output = context["run_root"] / "evidence/dev/finalist-dev.json"
    failure = context["run_root"] / "evidence/dev/dev-failure-after-claim.json"
    access = _mapping(context["campaign"].get("access_ledger"), "campaign.access_ledger")
    claim = _absolute(access.get("dev_claim"), "campaign dev claim", must_exist=False)
    bootcamp = _absolute(context["campaign"].get("bootcamp_root"), "campaign.bootcamp_root")
    expected_claim = (bootcamp / INHERITED_DEV_CLAIM_RELATIVE).resolve()
    _require(claim == expected_claim, "dev claim is not the inherited global lease")
    _require(
        Path(_text(access.get("dev_evaluation"), "campaign dev evaluation")).resolve()
        == output
        and Path(
            _text(access.get("dev_failure_after_claim"), "campaign dev failure")
        ).resolve()
        == failure
        and access.get("heldout_claim_disclosed") is False
        and access.get("heldout_authorized_claims") == 0,
        "campaign dev output or protected access boundary drifted",
    )
    for path, label in ((output, "dev output"), (failure, "dev failure"), (claim, "dev claim")):
        _require(path.parent.is_dir(), f"{label} parent is missing")
        _require(not path.exists() and not path.is_symlink(), f"{label} is not fresh")
    return output, failure, claim


def _dev_entry(context: Mapping[str, Any]) -> Mapping[str, Any]:
    datasets = _mapping(context["campaign"].get("datasets"), "campaign.datasets")
    entry = _mapping(datasets.get("dev"), "campaign.datasets.dev")
    _require(entry.get("records") == DEV_RECORDS, "dev manifest record count drifted")
    for field in (
        "file_sha256",
        "ordered_pair_ids_sha256",
        "ordered_row_hashes_sha256",
        "ordered_source_pair_hashes_sha256",
    ):
        value = _text(entry.get(field), f"campaign.datasets.dev.{field}")
        _require(len(value) == 64, f"campaign.datasets.dev.{field} is not SHA-256")
    return entry


def _dev_rows_after_claim(
    context: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Path]:
    # This is the sole function that opens dev bytes.  Its caller invokes it
    # only after the global O_EXCL claim has been durably written.
    bootcamp = _absolute(context["campaign"].get("bootcamp_root"), "campaign.bootcamp_root")
    relative = Path(_text(entry.get("path"), "dev dataset path"))
    _require(not relative.is_absolute() and ".." not in relative.parts, "dev dataset path is not canonical")
    path = (bootcamp / relative).resolve(strict=True)
    _require(path.is_relative_to(bootcamp), "dev dataset escaped bootcamp root")
    _require(gpu_stage.file_sha256(path) == entry.get("file_sha256"), "dev file hash drifted")
    if "bytes" in entry:
        _require(path.stat().st_size == entry.get("bytes"), "dev byte count drifted")
    rows = _load_jsonl(path, "dev17 dataset")
    _require(len(rows) == DEV_RECORDS, "dev17 record count drifted")
    try:
        summary = day23_contract.split_summary(rows)
    except day23_contract.Day23ContractError as error:
        raise QualificationEvaluationError(f"dev17 compiled rows failed: {error}") from error
    expected = {
        "records": DEV_RECORDS,
        "ordered_pair_ids_sha256": entry.get("ordered_pair_ids_sha256"),
        "ordered_source_pair_hashes_sha256": entry.get(
            "ordered_source_pair_hashes_sha256"
        ),
        "ordered_row_hashes_sha256": entry.get("ordered_row_hashes_sha256"),
    }
    _require(summary == expected, "dev17 row/order identity drifted")
    _require(all(row.get("split") == "dev" for row in rows), "dev17 split label drifted")
    source: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = _text(row.get("pair_id"), "dev pair_id")
        source[pair_id] = {
            "pair_status": "inherited_day23_cpu_validated",
            "quality_flags": ["source_metadata_not_reopened_during_qualification"],
        }
    return rows, source, path


def _gate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    overall = _mapping(aggregate.get("overall"), "aggregate.overall")
    matched = _mapping(aggregate.get("length_matched"), "aggregate.length_matched")
    pairs = _integer(overall.get("pairs"), "aggregate pair count")
    wins = _integer(overall.get("wins"), "aggregate positive pairs")
    mean = float(overall.get("mean_reward_margin"))
    matched_mean = float(matched.get("mean_reward_margin"))
    _require(
        all(math.isfinite(value) for value in (mean, matched_mean)),
        "dev aggregate contains non-finite margins",
    )
    thresholds = {
        "pairs": {"operator": "==", "threshold": DEV_RECORDS},
        "positive_margin_pairs": {
            "operator": ">=",
            "threshold": MINIMUM_POSITIVE_PAIRS,
            "denominator": DEV_RECORDS,
        },
        "mean_reward_margin": {"operator": ">", "threshold": 0.0},
        "length_matched_margin": {"operator": ">", "threshold": 0.0},
    }
    observed = {
        "pairs": pairs,
        "positive_margin_pairs": wins,
        "mean_reward_margin": mean,
        "length_matched_margin": matched_mean,
    }
    return {
        "passed": (
            pairs == DEV_RECORDS
            and wins >= MINIMUM_POSITIVE_PAIRS
            and mean > 0.0
            and matched_mean > 0.0
        ),
        "thresholds": thresholds,
        "observed": observed,
    }


def _claim_document(
    context: Mapping[str, Any], checkpoint: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    evaluator = Path(__file__).resolve(strict=True)
    return {
        "schema_name": CLAIM_SCHEMA,
        "schema_version": 1,
        "status": "claimed",
        "split": "dev",
        "campaign": {
            "path": str(context["campaign_path"]),
            "file_sha256": context["campaign_file_sha256"],
            "campaign_sha256": context["campaign_sha256"],
        },
        "run_id": RUN_ID,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "checkpoint_files_sha256": checkpoint["checkpoint"]["files_sha256"],
        "training_receipt_sha256": checkpoint["receipt_sha256"],
        "dev_identity": {
            key: entry.get(key)
            for key in (
                "file_sha256",
                "records",
                "ordered_pair_ids_sha256",
                "ordered_row_hashes_sha256",
                "ordered_source_pair_hashes_sha256",
            )
        },
        "max_global_claims": 1,
        "max_candidate_evaluations": 1,
        "retry_allowed": False,
        "fallback_allowed": False,
        "producer": {
            "path": str(evaluator),
            "file_sha256": gpu_stage.file_sha256(evaluator),
        },
        "heldout_authorized": False,
        "heldout_consumed": False,
    }


def _claim_dev_once(
    path: Path,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    _require(not path.exists() and not path.is_symlink(), "global dev lease was already claimed")
    document = _claim_document(context, checkpoint, entry)
    sealed = dict(document)
    sealed["claim_sha256"] = gpu_stage.object_sha256(sealed)
    payload = (
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    _write_exclusive(path, document, "claim_sha256")
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "claim_sha256": sealed["claim_sha256"],
    }


def _failure_document(
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    claim: Mapping[str, Any],
    error: BaseException,
    phase: str,
    dev_rows_opened: bool,
) -> dict[str, Any]:
    return {
        "schema_name": FAILURE_SCHEMA,
        "schema_version": 1,
        "status": "failed_closed_no_retry",
        "campaign_sha256": context["campaign_sha256"],
        "run_id": RUN_ID,
        "checkpoint_step": CHECKPOINT_STEP,
        "training_receipt_sha256": checkpoint["receipt_sha256"],
        "dev_claim_sha256": claim["claim_sha256"],
        "failure_phase": phase,
        "dev_rows_opened": dev_rows_opened,
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error_message": str(error),
        "traceback": traceback.format_exception(error)[-12:],
        "retry_allowed": False,
        "fallback_allowed": False,
        "heldout_consumed": False,
    }


def run_dev_evaluation(
    campaign_path: Path,
    training_receipt_path: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    # Everything through the successful policy reload is non-protected and
    # must pass before the single global dev lease can be consumed.
    context = _campaign_context(campaign_path)
    checkpoint = _checkpoint_from_receipt(context, training_receipt_path)
    expected_output, failure_path, claim_path = _expected_paths(context)
    output = output_path.expanduser().resolve()
    _require(output == expected_output, "dev evaluation output path drifted")
    entry = _dev_entry(context)
    runtime_context = _runtime_context(context, checkpoint)
    runtime = preference_eval.load_policy(
        runtime_context,
        {"checkpoint_path": checkpoint["checkpoint_path"]},
        device_name=device,
    )
    evaluator_process = gpu_stage._process_identity()
    runner_process = checkpoint["runner_process_identity"]
    _require(
        (
            evaluator_process.get("boot_id"),
            evaluator_process.get("pid"),
            evaluator_process.get("proc_start_ticks"),
        )
        != (
            runner_process.get("boot_id"),
            runner_process.get("pid"),
            runner_process.get("proc_start_ticks"),
        ),
        "dev evaluation must reload in a process distinct from training",
    )
    claim: dict[str, Any] | None = None
    phase = "dev_claim_pending"
    dev_rows_opened = False
    try:
        claim = _claim_dev_once(claim_path, context, checkpoint, entry)
        phase = "dev_claim_sealed"
        rows, source, dev_path = _dev_rows_after_claim(context, entry)
        dev_rows_opened = True
        phase = "dev_rows_validated"
        results = preference_eval.evaluate_rows(runtime, rows, source, beta=0.1)
        phase = "pair_evaluation_completed"
        _require(len(results) == DEV_RECORDS, "dev pair-result count drifted")
        for result in results:
            _require(
                result.get("schema_name") == preference_eval.PAIR_SCHEMA
                and result.get("status") == "pass",
                "dev pair-result schema/status drifted",
            )
            _verify_self(result, "pair_evaluation_sha256", "dev pair result")
            for key in (
                "policy_chosen_logps",
                "policy_rejected_logps",
                "reference_chosen_logps",
                "reference_rejected_logps",
                "chosen_reward",
                "rejected_reward",
                "reward_margin",
            ):
                number = result.get(key)
                _require(
                    isinstance(number, (int, float))
                    and not isinstance(number, bool)
                    and math.isfinite(float(number)),
                    f"dev pair-result contains non-finite {key}",
                )
        aggregate = preference_eval.aggregate_results(
            results, split="dev", checkpoint_step=CHECKPOINT_STEP
        )
        gate = _gate(aggregate)
        evaluator = Path(__file__).resolve(strict=True)
        value = {
            "schema_name": EVALUATION_SCHEMA,
            "schema_version": 1,
            "status": "pass" if gate["passed"] else "gate_fail",
            "campaign": {
                "path": str(context["campaign_path"]),
                "file_sha256": context["campaign_file_sha256"],
                "campaign_sha256": context["campaign_sha256"],
            },
            "run_id": RUN_ID,
            "producer": {
                "path": str(evaluator),
                "file_sha256": gpu_stage.file_sha256(evaluator),
            },
            "source_bindings": context["source_bindings"],
            "campaign_bound_runner": context["runner_identity"],
            "checkpoint": {
                "path": str(checkpoint["checkpoint_path"]),
                "global_step": CHECKPOINT_STEP,
                "files": checkpoint["checkpoint"]["files"],
                "files_sha256": checkpoint["checkpoint"]["files_sha256"],
                "adapter_file_sha256": checkpoint["checkpoint"]["files"]
                ["adapter_model.safetensors"]["sha256"],
                "training_receipt_path": str(checkpoint["receipt_path"]),
                "training_receipt_file_sha256": checkpoint[
                    "receipt_file_sha256"
                ],
                "training_receipt_sha256": checkpoint["receipt_sha256"],
            },
            "split": "dev",
            "dataset": {
                "path": str(dev_path),
                "file_sha256": entry.get("file_sha256"),
                "records": DEV_RECORDS,
                "ordered_pair_ids_sha256": entry.get("ordered_pair_ids_sha256"),
                "ordered_row_hashes_sha256": entry.get("ordered_row_hashes_sha256"),
                "ordered_source_pair_hashes_sha256": entry.get(
                    "ordered_source_pair_hashes_sha256"
                ),
            },
            "access_claim": claim,
            "runtime": runtime["runtime"],
            "process_identity": evaluator_process,
            "runner_process_identity": runner_process,
            "pair_results": results,
            "aggregate": aggregate,
            "eligibility": gate,
            "claim_boundary": {
                "fresh_process_checkpoint_reload_proven_before_dev_claim": True,
                "adapter_tensor_inventory_and_content_exact": True,
                "dev_global_claims": 1,
                "dev_candidate_evaluations": 1,
                "retry_allowed": False,
                "fallback_allowed": False,
                "generation_performed": False,
                "sandbox_execution_performed": False,
                "dev_consumed": True,
                "heldout_consumed": False,
            },
        }
        _write_exclusive(output, value, "evaluation_sha256")
        phase = "dev_evaluation_sealed"
    except BaseException as error:
        if claim is None:
            # write_sealed_json removes an incomplete O_EXCL file.  An O_EXCL
            # collision is somebody else's consumed lease and must not be
            # annotated with this process's failure receipt.
            raise
        try:
            _write_exclusive(
                failure_path,
                _failure_document(
                    context,
                    checkpoint,
                    claim,
                    error,
                    phase,
                    dev_rows_opened,
                ),
                "failure_sha256",
            )
        except BaseException as seal_error:
            raise QualificationEvaluationError(
                f"dev failed after claim and failure receipt could not be sealed: {seal_error}"
            ) from error
        raise
    _require(phase == "dev_evaluation_sealed", "dev evaluation did not seal")
    return _load_json(output, "sealed dev evaluation")


def _self_test() -> None:
    aggregate = {
        "overall": {
            "pairs": DEV_RECORDS,
            "wins": MINIMUM_POSITIVE_PAIRS,
            "mean_reward_margin": 0.001,
        },
        "length_matched": {"mean_reward_margin": 0.0001},
    }
    _require(_gate(aggregate)["passed"], "dev threshold boundary self-test failed")
    aggregate["overall"]["wins"] = MINIMUM_POSITIVE_PAIRS - 1
    _require(not _gate(aggregate)["passed"], "dev win rejection self-test failed")
    aggregate["overall"]["wins"] = MINIMUM_POSITIVE_PAIRS
    aggregate["length_matched"]["mean_reward_margin"] = 0.0
    _require(not _gate(aggregate)["passed"], "dev zero-margin rejection self-test failed")
    destinations = {action.dest for action in build_parser()._actions}
    _require(
        destinations
        == {"help", "campaign", "training_receipt", "output", "device", "self_test"},
        "qualification evaluator CLI surface drifted",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"status": "pass", "scope": "stdlib_self_test"}, sort_keys=True))
        return 0
    _require(args.campaign is not None, "campaign is required")
    _require(args.training_receipt is not None, "training receipt is required")
    _require(args.output is not None, "output is required")
    try:
        result = run_dev_evaluation(
            args.campaign, args.training_receipt, args.output, args.device
        )
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error": str(error),
                    "traceback": traceback.format_exception(error)[-8:],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "output": str(args.output),
                "evaluation_sha256": result.get("evaluation_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
