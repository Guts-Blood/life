#!/usr/bin/env python3
"""Run the Day 23 full112 Code guardrail in frozen Day 20 v3 E2B.

The wrapper has no heldout interface.  It accepts only a PASS qualification
dev result and a sealed full112 generation receipt, delegates sandbox work to
the frozen Day 20 v3 scorer, then publishes one O_EXCL Day 23 guardrail
receipt.  Non-Code full112 responses are never sent to the sandbox.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import run_day23_qwen35_gpu_stage as gpu_stage


GOAL_ID = "goal-0004-day23-dpo-qualification"
VERSION_ID = "qual-v0001"
CAMPAIGN_SCHEMA = "day23.qualification_v0001_gpu_campaign"
TRAINING_SCHEMA = "day23.qualification_v0001_training_receipt"
DEV_SCHEMA = "day23.qualification_v0001_dev_evaluation"
FULL112_SCHEMA = "day23.qualification_v0001_full112_guardrail_receipt"
RECEIPT_SCHEMA = "day23.qualification_v0001_e2b_guardrail_receipt"
RUN_ID = "full154_refit_lr_4p3e_6_step_20"
CANDIDATE = "qual-v0001-checkpoint-20"
CHECKPOINT_STEP = 20
FULL112_RECORDS = 112
CODE_RECORDS = 28

MINIMUM_TOTAL_CORRECT = 65
MINIMUM_GENERAL_CORRECT = 5
MINIMUM_MATH_CORRECT = 17
MINIMUM_FINANCE_CORRECT = 10
MINIMUM_CODE_CORRECT = 14
MINIMUM_FORMAT_COMPLIANT = 90
MINIMUM_SANDBOX_ELIGIBLE = 26
MAXIMUM_INFRASTRUCTURE_FAILURES = 0

CREDENTIAL_FILE = Path("/root/autodl-tmp/secrets/day20-v2-e2b.env")
CREDENTIAL_ATTESTATION = Path(
    "/root/autodl-tmp/secrets/day20-v2-e2b-attestation.json"
)
HUMANEVAL_SOURCE = Path(
    "/root/autodl-tmp/runs/day19-qwen35-20260808T100142Z/"
    "configs/frozen-inputs/HumanEval.jsonl.gz"
)
FULL112_RECEIPT_NAME = "full112-generation-and-score.json"
FULL112_PRODUCER_NAME = "eval_day23_full112_guardrails.py"
E2B_RECEIPT_NAME = "e2b-sandbox.json"
E2B_FAILURE_NAME = "e2b-sandbox-failure.json"
E2B_RESULTS_NAME = "full112-code-e2b-results.jsonl"
E2B_SUMMARY_NAME = "full112-code-e2b-summary.json"
E2B_PREFLIGHT_NAME = "E2B-PREFLIGHT.json"


class Day23E2BGuardrailError(RuntimeError):
    """A qualification, full112, credential, or sandbox invariant drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23E2BGuardrailError(message)


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
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def _absolute(value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(_text(value, label)).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        return path.resolve(strict=must_exist)
    except OSError as error:
        raise Day23E2BGuardrailError(f"cannot resolve {label}: {path}") from error


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day23E2BGuardrailError(f"cannot load {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _text(value.get(field), f"{label}.{field}")
    _require(
        len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
        and gpu_stage.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def _identity(path: Path, *, self_field: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "file_sha256": gpu_stage.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if self_field is not None:
        result["content_sha256"] = _verify_self(
            _load_json(resolved, str(resolved)), self_field, str(resolved)
        )
    return result


def _identity_entry(
    entry: Any, label: str, *, expected_path: Path | None = None
) -> tuple[Path, Mapping[str, Any]]:
    value = _mapping(entry, label)
    path = _absolute(value.get("path"), f"{label}.path")
    if expected_path is not None:
        _require(path == expected_path.resolve(strict=True), f"{label} path drifted")
    _require(
        value.get("file_sha256") == gpu_stage.file_sha256(path),
        f"{label} file hash drifted",
    )
    if "bytes" in value:
        _require(value.get("bytes") == path.stat().st_size, f"{label} byte count drifted")
    return path, value


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    _require(
        specification is not None and specification.loader is not None,
        f"cannot import bound module: {path}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextlib.contextmanager
def _python_path(path: Path):
    value = str(path.resolve(strict=True))
    inserted = value not in sys.path
    if inserted:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(value)


def _campaign_and_training(
    campaign_path: Path, training_receipt_path: Path
) -> dict[str, Any]:
    path = campaign_path.expanduser().resolve(strict=True)
    preliminary = _load_json(path, "qualification campaign")
    campaign_sha = _verify_self(preliminary, "campaign_sha256", "qualification campaign")
    _require(
        preliminary.get("schema_name") == CAMPAIGN_SCHEMA
        and preliminary.get("schema_version") == 1
        and preliminary.get("goal_id") == GOAL_ID
        and preliminary.get("version_id") == VERSION_ID,
        "qualification campaign identity drifted",
    )
    run_root = _absolute(preliminary.get("remote_run_root"), "campaign.remote_run_root")
    _require(path == run_root / "binding/gpu-campaign.json", "campaign path is not canonical")
    bootcamp = _absolute(preliminary.get("bootcamp_root"), "campaign.bootcamp_root")
    producers = _mapping(preliminary.get("producers"), "campaign.producers")
    runner_entry = _mapping(producers.get("runner"), "campaign runner")
    runner_path, _ = _identity_entry(runner_entry, "campaign runner")
    _require(runner_path.is_relative_to(bootcamp), "campaign runner escaped bootcamp")
    runner = _load_module(runner_path, "_day23_e2b_bound_qualification_runner")
    validate_campaign = getattr(runner, "validate_campaign_document", None)
    validate_training = getattr(runner, "validate_training_receipt_value", None)
    _require(
        callable(validate_campaign) and callable(validate_training),
        "campaign-bound runner verifier API is incomplete",
    )
    try:
        context = validate_campaign(path, require_dev_unopened=False)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Day23E2BGuardrailError(f"bound runner rejected campaign: {error}") from error
    campaign = _mapping(context.get("campaign"), "runner-validated campaign")
    _require(dict(campaign) == preliminary, "runner validated different campaign bytes")

    receipt_path = training_receipt_path.expanduser().resolve(strict=True)
    receipt = _load_json(receipt_path, "qualification training receipt")
    receipt_sha = _verify_self(receipt, "receipt_sha256", "qualification training receipt")
    _require(
        receipt.get("schema_name") == TRAINING_SCHEMA
        and receipt.get("status") == "pass"
        and receipt.get("run_id") == RUN_ID,
        "qualification training receipt identity drifted",
    )
    try:
        strict = validate_training(campaign, path, receipt, receipt_path)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Day23E2BGuardrailError(
            f"bound runner rejected training receipt: {error}"
        ) from error
    checkpoints = strict.get("checkpoints") if isinstance(strict, Mapping) else None
    _require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 1
        and checkpoints[0].get("global_step") == CHECKPOINT_STEP,
        "training receipt lacks the unique checkpoint-20 candidate",
    )
    return {
        "campaign": dict(campaign),
        "campaign_path": path,
        "campaign_sha256": campaign_sha,
        "run_root": run_root,
        "bootcamp": bootcamp,
        "training": receipt,
        "training_path": receipt_path,
        "training_sha256": receipt_sha,
        "checkpoint": dict(checkpoints[0]),
    }


def _verify_dev(context: Mapping[str, Any], dev_path: Path) -> dict[str, Any]:
    campaign = context["campaign"]
    access = _mapping(campaign.get("access_ledger"), "campaign.access_ledger")
    expected = _absolute(access.get("dev_evaluation"), "campaign dev evaluation")
    path = dev_path.expanduser().resolve(strict=True)
    _require(path == expected, "dev evaluation path is not campaign-bound")
    failure_path = _absolute(
        access.get("dev_failure_after_claim"), "campaign dev failure", must_exist=False
    )
    _require(not failure_path.exists(), "dev evaluation has a failure receipt")
    value = _load_json(path, "qualification dev evaluation")
    evaluation_sha = _verify_self(value, "evaluation_sha256", "qualification dev evaluation")
    _require(
        value.get("schema_name") == DEV_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "pass"
        and value.get("run_id") == RUN_ID
        and value.get("split") == "dev",
        "qualification dev evaluation did not PASS",
    )
    _require(
        value.get("campaign")
        == {
            "path": str(context["campaign_path"]),
            "file_sha256": gpu_stage.file_sha256(context["campaign_path"]),
            "campaign_sha256": context["campaign_sha256"],
        },
        "dev-to-campaign binding drifted",
    )
    producers = _mapping(campaign.get("producers"), "campaign.producers")
    evaluator_path, evaluator_entry = _identity_entry(
        producers.get("evaluator"), "campaign evaluator"
    )
    producer = _mapping(value.get("producer"), "dev producer")
    _require(
        Path(str(producer.get("path", ""))).resolve() == evaluator_path
        and producer.get("file_sha256") == evaluator_entry.get("file_sha256"),
        "dev producer binding drifted",
    )
    evaluator = _load_module(evaluator_path, "_day23_e2b_bound_dev_evaluator")
    checkpoint = _mapping(value.get("checkpoint"), "dev checkpoint")
    expected_checkpoint = context["checkpoint"]
    _require(
        checkpoint.get("global_step") == CHECKPOINT_STEP
        and Path(str(checkpoint.get("path", ""))).resolve()
        == Path(str(expected_checkpoint["path"])).resolve()
        and checkpoint.get("files_sha256") == expected_checkpoint.get("files_sha256")
        and checkpoint.get("training_receipt_sha256") == context["training_sha256"],
        "dev checkpoint/training binding drifted",
    )
    pair_results = value.get("pair_results")
    _require(isinstance(pair_results, list) and len(pair_results) == 17, "dev pair coverage drifted")
    for row in pair_results:
        _mapping(row, "dev pair result")
        _verify_self(row, "pair_evaluation_sha256", "dev pair result")
    aggregate_results = getattr(evaluator.preference_eval, "aggregate_results", None)
    gate_function = getattr(evaluator, "_gate", None)
    _require(callable(aggregate_results) and callable(gate_function), "dev verifier API drifted")
    recomputed = aggregate_results(pair_results, split="dev", checkpoint_step=CHECKPOINT_STEP)
    gate = gate_function(recomputed)
    _require(
        value.get("aggregate") == recomputed
        and value.get("eligibility") == gate
        and gate.get("passed") is True,
        "dev aggregate/gate no longer recomputes",
    )
    boundary = _mapping(value.get("claim_boundary"), "dev claim boundary")
    _require(
        boundary.get("dev_consumed") is True
        and boundary.get("heldout_consumed") is False
        and boundary.get("sandbox_execution_performed") is False,
        "dev protected-data boundary drifted",
    )
    return {
        "value": value,
        "path": path,
        "evaluation_sha256": evaluation_sha,
        "producer": _identity(evaluator_path),
    }


def _validate_code_rows(value: Any) -> list[str]:
    _require(isinstance(value, list) and len(value) == CODE_RECORDS, "full112 must bind 28 Code rows")
    code_ids: list[str] = []
    for row in value:
        _mapping(row, "full112 Code row")
        _require(row.get("slice") == "code", "full112 sandbox input contains a non-Code row")
        _verify_self(row, "row_sha256", "full112 Code row")
        code_ids.append(_text(row.get("sample_id"), "full112 Code sample_id"))
    _require(len(set(code_ids)) == CODE_RECORDS, "full112 Code sample IDs are not unique")
    return code_ids


def _load_full112_receipt(
    context: Mapping[str, Any], dev: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    expected = context["run_root"] / "evidence/guardrails" / FULL112_RECEIPT_NAME
    path = receipt_path.expanduser().resolve(strict=True)
    _require(path == expected, "full112 receipt path is not canonical")
    value = _load_json(path, "full112 generation receipt")
    receipt_sha = _verify_self(value, "receipt_sha256", "full112 generation receipt")
    _require(
        value.get("schema_name") == FULL112_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "pass_pending_code_sandbox",
        "full112 generation receipt is not sandbox-ready",
    )
    producer_path, producer_entry = _identity_entry(
        value.get("producer"), "full112 receipt producer"
    )
    _require(
        producer_path
        == (Path(__file__).resolve().parent / FULL112_PRODUCER_NAME).resolve(strict=True),
        "full112 producer path drifted",
    )
    producer = _load_module(producer_path, "_day23_e2b_bound_full112_producer")
    verify_receipt = getattr(producer, "validate_receipt", None)
    _require(callable(verify_receipt), "full112 producer has no validate_receipt API")
    try:
        verified = verify_receipt(path)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Day23E2BGuardrailError(
            f"full112 producer rejected sealed receipt: {error}"
        ) from error
    verified_value = verified.get("receipt", verified) if isinstance(verified, Mapping) else None
    _require(verified_value == value, "full112 producer verified different receipt bytes")

    campaign_entry = _mapping(value.get("campaign"), "full112 campaign binding")
    _require(
        Path(str(campaign_entry.get("path", ""))).resolve() == context["campaign_path"]
        and campaign_entry.get("file_sha256") == gpu_stage.file_sha256(context["campaign_path"])
        and campaign_entry.get("campaign_sha256") == context["campaign_sha256"],
        "full112 campaign binding drifted",
    )
    training_entry = _mapping(value.get("training_receipt"), "full112 training binding")
    _require(
        Path(str(training_entry.get("path", ""))).resolve() == context["training_path"]
        and training_entry.get("file_sha256") == gpu_stage.file_sha256(context["training_path"])
        and training_entry.get("content_sha256") == context["training_sha256"],
        "full112 training binding drifted",
    )
    _require(
        value.get("checkpoint") == context["checkpoint"],
        "full112 checkpoint-20 binding drifted",
    )
    dev_entry = _mapping(value.get("dev_evaluation"), "full112 dev binding")
    _require(
        Path(str(dev_entry.get("path", ""))).resolve() == dev["path"]
        and dev_entry.get("file_sha256") == gpu_stage.file_sha256(dev["path"])
        and dev_entry.get("content_sha256") == dev["evaluation_sha256"],
        "full112 dev binding drifted",
    )
    boundary = _mapping(value.get("claim_boundary"), "full112 claim boundary")
    _require(
        boundary.get("fresh_process_checkpoint_reload") is True
        and boundary.get("candidate_code_executed_on_host") is False
        and boundary.get("code_and_total_gates_require_e2b") is True
        and boundary.get("final_qualification_claimed") is False,
        "full112 pre-sandbox boundary drifted",
    )
    aggregate = _mapping(value.get("aggregate"), "full112 aggregate")
    _require(
        _integer(aggregate.get("records"), "full112 records") == FULL112_RECORDS
        and aggregate.get("code_correct") is None
        and aggregate.get("total_correct") is None
        and _integer(
            aggregate.get("code_sandbox_execution_eligible"),
            "full112 sandbox eligibility",
        )
        >= MINIMUM_SANDBOX_ELIGIBLE
        and _integer(aggregate.get("infrastructure_failures"), "full112 infra failures")
        == 0,
        "full112 aggregate is not E2B-ready",
    )
    normalized_predictions = _mapping(
        value.get("normalized_predictions"), "full112 normalized predictions"
    )
    normalized_summary = _mapping(
        value.get("normalized_summary"), "full112 normalized summary"
    )
    predictions_path, _ = _identity_entry(
        normalized_predictions, "full112 normalized predictions"
    )
    summary_path, _ = _identity_entry(normalized_summary, "full112 normalized summary")
    pair_verifier = getattr(producer, "verify_normalized_pair", None)
    _require(callable(pair_verifier), "full112 producer has no verify_normalized_pair API")
    try:
        normalized_rows, normalized_value = pair_verifier(
            predictions_path,
            summary_path,
            expected_scope="full112",
            expected_candidate=CANDIDATE,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise Day23E2BGuardrailError(f"normalized full112 pair was rejected: {error}") from error
    selected_code = [row for row in normalized_rows if row.get("slice") == "code"]
    code_ids = _validate_code_rows(selected_code)
    _require(
        len(normalized_rows) == FULL112_RECORDS
        and len(selected_code) == CODE_RECORDS
        and normalized_value.get("candidate") == CANDIDATE
        and normalized_value.get("scope") == "full112",
        "normalized full112/Code coverage drifted",
    )
    return {
        "value": value,
        "path": path,
        "receipt_sha256": receipt_sha,
        "producer": producer_entry,
        "producer_module": producer,
        "pair_verifier": pair_verifier,
        "predictions_path": predictions_path,
        "summary_path": summary_path,
        "aggregate": dict(aggregate),
        "code_ids": code_ids,
        "code_rows": selected_code,
    }


def evaluate_gate(
    full112_aggregate: Mapping[str, Any], e2b_summary: Mapping[str, Any]
) -> dict[str, Any]:
    general = _integer(full112_aggregate.get("general_correct"), "general_correct")
    math_correct = _integer(full112_aggregate.get("math_correct"), "math_correct")
    finance = _integer(full112_aggregate.get("finance_correct"), "finance_correct")
    format_compliant = _integer(
        full112_aggregate.get("format_compliant"), "format_compliant"
    )
    code = _integer(e2b_summary.get("passed"), "E2B passed")
    eligible = _integer(
        e2b_summary.get("sandbox_execution_eligible"), "E2B eligibility"
    )
    infrastructure = _integer(
        e2b_summary.get("infrastructure_failures"), "E2B infrastructure failures"
    )
    code_records = _integer(e2b_summary.get("code_records"), "E2B Code records")
    _require(code_records == CODE_RECORDS, "E2B scorer did not bind exactly 28 Code rows")
    total = general + math_correct + finance + code
    checks = {
        "total_correct": total >= MINIMUM_TOTAL_CORRECT,
        "general_correct": general >= MINIMUM_GENERAL_CORRECT,
        "math_correct": math_correct >= MINIMUM_MATH_CORRECT,
        "finance_correct": finance >= MINIMUM_FINANCE_CORRECT,
        "code_correct": code >= MINIMUM_CODE_CORRECT,
        "format_compliant": format_compliant >= MINIMUM_FORMAT_COMPLIANT,
        "code_sandbox_execution_eligible": eligible >= MINIMUM_SANDBOX_ELIGIBLE,
        "infrastructure_failures": infrastructure <= MAXIMUM_INFRASTRUCTURE_FAILURES,
    }
    return {
        "passed": all(checks.values()),
        "thresholds": {
            "total_correct": MINIMUM_TOTAL_CORRECT,
            "general_correct": MINIMUM_GENERAL_CORRECT,
            "math_correct": MINIMUM_MATH_CORRECT,
            "finance_correct": MINIMUM_FINANCE_CORRECT,
            "code_correct": MINIMUM_CODE_CORRECT,
            "format_compliant": MINIMUM_FORMAT_COMPLIANT,
            "code_sandbox_execution_eligible": MINIMUM_SANDBOX_ELIGIBLE,
            "maximum_infrastructure_failures": MAXIMUM_INFRASTRUCTURE_FAILURES,
        },
        "observed": {
            "total_correct": total,
            "general_correct": general,
            "math_correct": math_correct,
            "finance_correct": finance,
            "code_correct": code,
            "format_compliant": format_compliant,
            "code_sandbox_execution_eligible": eligible,
            "infrastructure_failures": infrastructure,
        },
        "checks": checks,
    }


def _frozen_paths(bootcamp: Path) -> dict[str, Path]:
    day20 = bootcamp / "day-20-qwen35-balanced-lora-sft"
    day10 = bootcamp / "day-10-frozen-eval-baseline"
    return {
        "day20_e2b_wrapper": day20 / "score_day20_code_e2b_v3.py",
        "day20_e2b_preflight": day20 / "day20_e2b_preflight_v2.py",
        "frozen_e2b_scorer": day10 / "score_day10_code_e2b.py",
        "sandbox_config": day10 / "day10_e2b_sandbox_config.json",
        # This is the exact source snapshot used by the frozen Day 20 v3
        # AutoDL runbook.  The staged bootcamp checkout intentionally does not
        # carry the large historical ``tmp/`` source tree.
        "humaneval_source": HUMANEVAL_SOURCE,
    }


def _score(
    context: Mapping[str, Any],
    full112: Mapping[str, Any],
    *,
    credential_file: Path,
    credential_attestation: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {name: path.resolve(strict=True) for name, path in _frozen_paths(context["bootcamp"]).items()}
    _require(
        credential_file.resolve(strict=True) == CREDENTIAL_FILE
        and credential_attestation.resolve(strict=True) == CREDENTIAL_ATTESTATION,
        "E2B credential/attestation path drifted",
    )
    guardrail_root = context["run_root"] / "evidence/guardrails"
    _require(guardrail_root.is_dir(), "guardrail evidence directory is missing")
    preflight_path = context["run_root"] / "evidence" / E2B_PREFLIGHT_NAME
    result_path = guardrail_root / E2B_RESULTS_NAME
    summary_path = guardrail_root / E2B_SUMMARY_NAME
    day20_dir = paths["day20_e2b_wrapper"].parent
    with _python_path(day20_dir):
        preflight = _load_module(paths["day20_e2b_preflight"], "_day23_bound_day20_e2b_preflight")
        scorer = _load_module(paths["day20_e2b_wrapper"], "_day23_bound_day20_e2b_wrapper")
        try:
            if preflight_path.exists():
                preflight_value = preflight.verify_preflight(
                    preflight_path, run_root=context["run_root"]
                )
            else:
                preflight_value = preflight.run_preflight(
                    run_root=context["run_root"],
                    frozen_scorer=paths["frozen_e2b_scorer"],
                    sandbox_config=paths["sandbox_config"],
                    humaneval_source=paths["humaneval_source"],
                    credential_file=credential_file,
                    credential_attestation=credential_attestation,
                    marker_path=preflight_path,
                )
            secret, attestation_identity = preflight.verify_credential(
                credential_file, credential_attestation
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise Day23E2BGuardrailError(f"frozen E2B preflight failed: {error}") from error
        previous = os.environ.get("E2B_API_KEY")
        os.environ["E2B_API_KEY"] = secret
        try:
            e2b_summary = scorer.score_code_pair(
                predictions_path=full112["predictions_path"],
                normalized_summary_path=full112["summary_path"],
                frozen_scorer_path=paths["frozen_e2b_scorer"],
                sandbox_config_path=paths["sandbox_config"],
                humaneval_source_path=paths["humaneval_source"],
                preflight_path=preflight_path,
                run_root=context["run_root"],
                output_path=result_path,
                summary_output_path=summary_path,
                pair_verifier=full112["pair_verifier"],
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise Day23E2BGuardrailError(f"frozen E2B scorer failed: {error}") from error
        finally:
            if previous is None:
                os.environ.pop("E2B_API_KEY", None)
            else:
                os.environ["E2B_API_KEY"] = previous
    _require(
        e2b_summary.get("candidate") == CANDIDATE
        and e2b_summary.get("scope") == "full112"
        and e2b_summary.get("status") == "complete"
        and e2b_summary.get("code_records") == CODE_RECORDS
        and e2b_summary.get("eligible_code_sample_ids")
        == [
            sample_id
            for sample_id, row in zip(full112["code_ids"], full112["code_rows"])
            if row.get("sandbox_execution_eligible") is True
        ],
        "frozen E2B summary coverage/identity drifted",
    )
    return e2b_summary, {
        "components": {name: _identity(path) for name, path in sorted(paths.items())},
        "preflight": _identity(preflight_path, self_field="preflight_sha256"),
        "preflight_value": preflight_value,
        "credential_file": {
            "path": str(credential_file),
            "file_sha256": gpu_stage.file_sha256(credential_file),
        },
        "credential_attestation": attestation_identity,
        "results": _identity(result_path),
        "summary": _identity(summary_path, self_field="summary_sha256"),
    }


def _write_receipt(
    path: Path,
    context: Mapping[str, Any],
    dev: Mapping[str, Any],
    full112: Mapping[str, Any],
    e2b_summary: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    expected = context["run_root"] / "evidence/guardrails" / E2B_RECEIPT_NAME
    output = path.expanduser().resolve()
    _require(output == expected, "E2B guardrail receipt path is not canonical")
    _require(not output.exists() and not output.is_symlink(), "E2B guardrail receipt already exists")
    gate = evaluate_gate(full112["aggregate"], e2b_summary)
    producer = Path(__file__).resolve(strict=True)
    receipt = {
        "schema_name": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "pass" if gate["passed"] else "gate_fail",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "run_id": RUN_ID,
        "candidate": CANDIDATE,
        "campaign": {
            "path": str(context["campaign_path"]),
            "file_sha256": gpu_stage.file_sha256(context["campaign_path"]),
            "campaign_sha256": context["campaign_sha256"],
        },
        "training_receipt": {
            "path": str(context["training_path"]),
            "file_sha256": gpu_stage.file_sha256(context["training_path"]),
            "receipt_sha256": context["training_sha256"],
        },
        "checkpoint": {
            "path": context["checkpoint"]["path"],
            "global_step": CHECKPOINT_STEP,
            "files_sha256": context["checkpoint"]["files_sha256"],
        },
        "dev_evaluation": {
            "path": str(dev["path"]),
            "file_sha256": gpu_stage.file_sha256(dev["path"]),
            "evaluation_sha256": dev["evaluation_sha256"],
            "status": "pass",
        },
        "full112_receipt": {
            "path": str(full112["path"]),
            "file_sha256": gpu_stage.file_sha256(full112["path"]),
            "receipt_sha256": full112["receipt_sha256"],
            "status": "pass_pending_code_sandbox",
        },
        "producer": _identity(producer),
        "frozen_e2b": {
            "components": execution["components"],
            "preflight": execution["preflight"],
            "credential_file": execution["credential_file"],
            "credential_attestation": execution["credential_attestation"],
            "results": execution["results"],
            "summary": execution["summary"],
            "e2b_run_sha256": e2b_summary["e2b_run_sha256"],
            "complete_comparison_key": e2b_summary["complete_comparison_key"],
        },
        "sandbox_input_boundary": {
            "full112_records_verified": FULL112_RECORDS,
            "code_records_selected": CODE_RECORDS,
            "non_code_records_executed": 0,
            "eligible_code_records_executed": e2b_summary[
                "sandbox_execution_eligible"
            ],
            "ordered_code_sample_ids": full112["code_ids"],
            "candidate_code_executed_on_host": False,
            "heldout_consumed": False,
        },
        "metrics": {
            **gate["observed"],
            "code_records": CODE_RECORDS,
            "records": e2b_summary["records"],
            "passed": e2b_summary["passed"],
            "failed": e2b_summary["failed"],
            "sandbox_execution_eligible": e2b_summary[
                "sandbox_execution_eligible"
            ],
        },
        "gate": gate,
        "claim_boundary": {
            "dev_pass_required": True,
            "generation_performed_by_this_wrapper": False,
            "sandbox_execution_performed": True,
            "heldout_interface_exposed": False,
            "heldout_consumed": False,
        },
    }
    try:
        gpu_stage.write_sealed_json(output, receipt, "receipt_sha256")
    except (OSError, gpu_stage.Day23GPUStageError) as error:
        raise Day23E2BGuardrailError(f"cannot seal E2B receipt: {error}") from error
    return _load_json(output, "sealed E2B guardrail receipt")


def _optional_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        return {"path": str(resolved), "present": False}
    return {**_identity(resolved), "present": True}


def _seal_failure(
    path: Path,
    *,
    phase: str,
    error: BaseException,
    context: Mapping[str, Any],
    dev_path: Path,
    full112_path: Path,
) -> dict[str, Any]:
    expected = context["run_root"] / "evidence/guardrails" / E2B_FAILURE_NAME
    failure_path = path.expanduser().resolve()
    success_path = context["run_root"] / "evidence/guardrails" / E2B_RECEIPT_NAME
    _require(failure_path == expected, "E2B failure receipt path is not canonical")
    _require(
        not success_path.exists() and not success_path.is_symlink(),
        "cannot publish E2B failure after a terminal success/gate receipt",
    )
    guardrail_root = context["run_root"] / "evidence/guardrails"
    value = {
        "schema_name": "day23.qualification_v0001_e2b_guardrail_failure",
        "schema_version": 1,
        "status": "failed_closed",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "run_id": RUN_ID,
        "candidate": CANDIDATE,
        "failure_phase": phase,
        "error": {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": traceback.format_exception(
                type(error), error, error.__traceback__
            )[-12:],
        },
        "campaign": {
            "path": str(context["campaign_path"]),
            "file_sha256": gpu_stage.file_sha256(context["campaign_path"]),
            "campaign_sha256": context["campaign_sha256"],
        },
        "training_receipt": {
            "path": str(context["training_path"]),
            "file_sha256": gpu_stage.file_sha256(context["training_path"]),
            "receipt_sha256": context["training_sha256"],
        },
        "inputs": {
            "dev_evaluation": _optional_identity(dev_path),
            "full112_receipt": _optional_identity(full112_path),
        },
        "partial_e2b_artifacts": {
            "preflight": _optional_identity(
                context["run_root"] / "evidence" / E2B_PREFLIGHT_NAME
            ),
            "results": _optional_identity(guardrail_root / E2B_RESULTS_NAME),
            "summary": _optional_identity(guardrail_root / E2B_SUMMARY_NAME),
        },
        "claim_boundary": {
            "scientific_guardrail_result_available": False,
            "retry_allowed": False,
            "fallback_allowed": False,
            "candidate_code_executed_on_host": False,
            "heldout_interface_exposed": False,
            "heldout_consumed": False,
        },
        "producer": _identity(Path(__file__).resolve(strict=True)),
    }
    try:
        gpu_stage.write_sealed_json(failure_path, value, "failure_sha256")
    except (OSError, gpu_stage.Day23GPUStageError) as seal_error:
        raise Day23E2BGuardrailError(
            f"cannot seal E2B failure receipt: {seal_error}"
        ) from error
    return _load_json(failure_path, "sealed E2B failure receipt")


def run_guardrail(
    campaign_path: Path,
    training_receipt_path: Path,
    dev_evaluation_path: Path,
    full112_receipt_path: Path,
    output_path: Path,
    *,
    credential_file: Path = CREDENTIAL_FILE,
    credential_attestation: Path = CREDENTIAL_ATTESTATION,
) -> dict[str, Any]:
    context = _campaign_and_training(campaign_path, training_receipt_path)
    output = output_path.expanduser().resolve()
    expected_output = context["run_root"] / "evidence/guardrails" / E2B_RECEIPT_NAME
    failure_path = context["run_root"] / "evidence/guardrails" / E2B_FAILURE_NAME
    _require(output == expected_output, "E2B guardrail receipt path is not canonical")
    _require(
        not output.exists()
        and not output.is_symlink()
        and not failure_path.exists()
        and not failure_path.is_symlink(),
        "E2B guardrail already has a terminal receipt",
    )
    phase = "dev_pass_verification"
    try:
        dev = _verify_dev(context, dev_evaluation_path)
        phase = "full112_receipt_verification"
        full112 = _load_full112_receipt(context, dev, full112_receipt_path)
        phase = "frozen_e2b_preflight_and_scoring"
        e2b_summary, execution = _score(
            context,
            full112,
            credential_file=credential_file,
            credential_attestation=credential_attestation,
        )
        phase = "terminal_guardrail_receipt_publication"
        return _write_receipt(
            output, context, dev, full112, e2b_summary, execution
        )
    except BaseException as error:
        _seal_failure(
            failure_path,
            phase=phase,
            error=error,
            context=context,
            dev_path=dev_evaluation_path,
            full112_path=full112_receipt_path,
        )
        raise


def _self_test() -> None:
    gate = evaluate_gate(
        {
            "general_correct": 20,
            "math_correct": 17,
            "finance_correct": 14,
            "format_compliant": 100,
        },
        {
            "passed": 14,
            "sandbox_execution_eligible": 26,
            "infrastructure_failures": 0,
            "code_records": 28,
        },
    )
    _require(gate["passed"] and gate["observed"]["total_correct"] == 65, "gate self-test failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--dev-evaluation", type=Path)
    parser.add_argument("--full112-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {"status": "pass", "scope": "stdlib_self_test"}
        else:
            for name in (
                "campaign",
                "training_receipt",
                "dev_evaluation",
                "full112_receipt",
                "output",
            ):
                _require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
            result = run_guardrail(
                args.campaign,
                args.training_receipt,
                args.dev_evaluation,
                args.full112_receipt,
                args.output,
            )
    except BaseException as error:
        print(
            json.dumps(
                {"status": "fail", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
