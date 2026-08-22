#!/usr/bin/env python3
"""Append-only guardrail authorization and heldout closeout for qual-v0001.

This control-plane executable never opens dev17 or heldout29 dataset rows.  It
validates sealed evidence, freezes the previously-missing heldout29 numerical
gate, binds one immutable heldout evaluator, and authorizes at most one global
heldout claim/evaluation.  A second phase validates the sealed heldout result
and appends the unique terminal event.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import eval_day23_qualification_candidate as dev_eval
import run_day23_qualification_candidate as qualification_runner
import run_day23_qwen35_gpu_stage as gpu_stage


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
GOAL_ID = "goal-0004-day23-dpo-qualification"
VERSION_ID = "qual-v0001"
CAMPAIGN_ID = "day23-qwen35-dpo-qualification-v0001"
RUN_ID = "full154_refit_lr_4p3e_6_step_20"
CHECKPOINT_STEP = 20

DEV_SCHEMA = "day23.qualification_v0001_dev_evaluation"
FULL112_SCHEMA = "day23.qualification_v0001_full112_guardrail_receipt"
E2B_SCHEMA = "day23.qualification_v0001_e2b_guardrail_receipt"
AMENDMENT_SCHEMA = "day23.qualification_v0001_append_only_amendment"
AUTHORIZATION_SCHEMA = "day23.qualification_v0001_heldout_authorization"
HELDOUT_CLAIM_SCHEMA = "day23.qualification_v0001_heldout_access_claim"
HELDOUT_EVALUATION_SCHEMA = "day23.qualification_v0001_heldout_evaluation"
HELDOUT_FAILURE_SCHEMA = "day23.qualification_v0001_heldout_failure"
EVENT_SCHEMA = "day23.qualification_event"

HELDOUT_RECORDS = 29
HELDOUT_MINIMUM_POSITIVE_PAIRS = 21
DEV_SIGN_TEST_P = 0.0245208740234375
HELDOUT_20_SIGN_TEST_P = 0.03071417286992073
HELDOUT_21_SIGN_TEST_P = 0.01205977238714695

GOAL2_REL = Path("rsi-control/charters/goal-0002-day23-dpo/charter.json")
GOAL4_REL = Path("rsi-control/charters") / GOAL_ID
EVENT3_NAME = "event-000003-full154-refit-completed"
EVENT4_NAME = "event-000004-dev-qualified"
EVENT4_FAILURE_NAME = "event-000004-dev-terminal-no-go"
EVENT5_NAME = "event-000005-guardrails-passed-heldout-authorized"
EVENT6_NAME = "event-000006-heldout-terminal"
AMENDMENT_NAME = "amendment-000001-guardrails-and-heldout-gate"

EXPECTED_CODING_GATES = {
    "total_correct": {"operator": ">=", "threshold": 65},
    "general_correct": {"operator": ">=", "threshold": 5},
    "math_correct": {"operator": ">=", "threshold": 17},
    "finance_correct": {"operator": ">=", "threshold": 10},
    "code_correct": {"operator": ">=", "threshold": 14},
    "format_compliant": {"operator": ">=", "threshold": 90},
    "code_sandbox_execution_eligible": {"operator": ">=", "threshold": 26},
    "infrastructure_failures": {"operator": "<=", "threshold": 0},
}
EXPECTED_DEV_GATE = {
    "positive_margin_pairs": {"denominator": 17, "operator": ">=", "threshold": 13},
    "mean_reward_margin": {"operator": ">", "threshold": 0.0},
    "length_matched_margin": {"operator": ">", "threshold": 0.0},
}


class QualificationCloseoutError(RuntimeError):
    """A sealed-evidence, gate, lease, or append-only invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationCloseoutError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def text(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and bool(value) and "\x00" not in value,
        f"{label} must be non-empty text",
    )
    return value


def integer(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return value


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def absolute(value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(text(value, label)).expanduser()
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        return path.resolve(strict=must_exist)
    except OSError as error:
        raise QualificationCloseoutError(f"cannot resolve {label}: {path}") from error


def load(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationCloseoutError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = text(value.get(field), f"{label}.{field}")
    require(
        len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
        and gpu_stage.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = gpu_stage.object_sha256(result)
    return result


def file_identity(
    path: Path,
    *,
    self_field: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to.resolve())) if relative_to else str(resolved),
        "file_sha256": gpu_stage.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if self_field is not None:
        result["content_sha256"] = verify_self(load(resolved, str(resolved)), self_field, str(resolved))
    return result


def payload_identity(
    path: Path,
    value: Mapping[str, Any],
    self_field: str,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    payload = json_bytes(value)
    return {
        "path": str(path.resolve().relative_to(relative_to.resolve())),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "content_sha256": value[self_field],
    }


def event_identity(path: Path, value: Mapping[str, Any], goal_root: Path) -> dict[str, Any]:
    full = payload_identity(path, value, "event_sha256", relative_to=goal_root)
    return {
        "path": full["path"],
        "file_sha256": full["file_sha256"],
        "event_sha256": full["content_sha256"],
        "sequence": value["sequence"],
    }


def write_exclusive_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        require(
            path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
            f"append-only output collision: {path}",
        )
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def validate_created_at(value: str) -> str:
    text(value, "created_at_utc")
    require(value.endswith("Z"), "created_at_utc must be UTC with Z suffix")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise QualificationCloseoutError("created_at_utc is not ISO-8601") from error
    return value


def heldout_gate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    overall = mapping(aggregate.get("overall"), "heldout aggregate.overall")
    length_matched = mapping(
        aggregate.get("length_matched"), "heldout aggregate.length_matched"
    )
    pairs = integer(overall.get("pairs"), "heldout pairs")
    wins = integer(overall.get("wins"), "heldout positive pairs")
    mean = finite_number(overall.get("mean_reward_margin"), "heldout mean margin")
    matched_mean = finite_number(
        length_matched.get("mean_reward_margin"), "heldout length-matched mean margin"
    )
    observed = {
        "finite_records": pairs,
        "positive_margin_pairs": wins,
        "mean_reward_margin": mean,
        "length_matched_mean_reward_margin": matched_mean,
    }
    thresholds = {
        "finite_records": {"operator": "==", "threshold": HELDOUT_RECORDS},
        "positive_margin_pairs": {
            "operator": ">=",
            "threshold": HELDOUT_MINIMUM_POSITIVE_PAIRS,
            "denominator": HELDOUT_RECORDS,
        },
        "mean_reward_margin": {"operator": ">", "threshold": 0.0},
        "length_matched_mean_reward_margin": {"operator": ">", "threshold": 0.0},
    }
    return {
        "passed": (
            pairs == HELDOUT_RECORDS
            and wins >= HELDOUT_MINIMUM_POSITIVE_PAIRS
            and mean > 0.0
            and matched_mean > 0.0
        ),
        "thresholds": thresholds,
        "observed": observed,
    }


def recompute_combined_guardrails(
    full112_metrics: Mapping[str, Any], e2b_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    records = integer(full112_metrics.get("records"), "full112 records")
    general = integer(full112_metrics.get("general_correct"), "general correct")
    math_correct = integer(full112_metrics.get("math_correct"), "math correct")
    finance = integer(full112_metrics.get("finance_correct"), "finance correct")
    non_code = integer(full112_metrics.get("non_code_correct"), "non-code correct")
    formatted = integer(full112_metrics.get("format_compliant"), "format compliant")
    generated_eligible = integer(
        full112_metrics.get("code_sandbox_execution_eligible"),
        "generated code sandbox eligible",
    )
    generation_infra = integer(
        full112_metrics.get("infrastructure_failures"), "generation infrastructure failures"
    )
    code_records = integer(e2b_metrics.get("code_records"), "E2B code records")
    eligible = integer(
        e2b_metrics.get("code_sandbox_execution_eligible"), "E2B eligible"
    )
    code_correct = integer(e2b_metrics.get("code_correct"), "E2B passed")
    code_failed = integer(e2b_metrics.get("failed"), "E2B failed")
    e2b_infra = integer(e2b_metrics.get("infrastructure_failures"), "E2B infrastructure failures")
    require(
        records == 112
        and full112_metrics.get("records_by_skill")
        == {"code": 28, "finance": 28, "general": 28, "math": 28},
        "full112 record/slice inventory drifted",
    )
    require(
        non_code == general + math_correct + finance,
        "full112 non-code total is not the exact three-slice sum",
    )
    require(
        code_records == 28
        and 0 <= eligible <= code_records
        and code_correct + code_failed == eligible,
        "E2B code accounting drifted",
    )
    require(
        generated_eligible == eligible,
        "generation and E2B eligible inventories disagree",
    )
    observed = {
        "total_correct": non_code + code_correct,
        "general_correct": general,
        "math_correct": math_correct,
        "finance_correct": finance,
        "code_correct": code_correct,
        "format_compliant": formatted,
        "code_sandbox_execution_eligible": eligible,
        "infrastructure_failures": generation_infra + e2b_infra,
    }
    require(
        e2b_metrics.get("total_correct") == observed["total_correct"]
        and e2b_metrics.get("general_correct") == general
        and e2b_metrics.get("math_correct") == math_correct
        and e2b_metrics.get("finance_correct") == finance
        and e2b_metrics.get("format_compliant") == formatted,
        "E2B combined metric projection drifted",
    )
    passed = all(
        observed[key] >= gate["threshold"] if gate["operator"] == ">=" else observed[key] <= gate["threshold"]
        for key, gate in EXPECTED_CODING_GATES.items()
    )
    return {"passed": passed, "thresholds": EXPECTED_CODING_GATES, "observed": observed}


def _identity_matches(
    value: Mapping[str, Any],
    path: Path,
    *,
    self_field: str,
    content_key: str,
    label: str,
) -> None:
    expected = file_identity(path, self_field=self_field)
    require(
        Path(text(value.get("path"), f"{label}.path")).resolve() == path.resolve()
        and value.get("file_sha256") == expected["file_sha256"]
        and value.get(content_key) == expected["content_sha256"],
        f"{label} identity drifted",
    )


def _campaign_training_context(
    campaign_path: Path, training_receipt_path: Path
) -> dict[str, Any]:
    try:
        context = qualification_runner.validate_campaign(
            campaign_path, require_dev_unopened=False
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise QualificationCloseoutError(
            f"frozen qualification runner rejected campaign: {error}"
        ) from error
    require(isinstance(context, Mapping), "qualification runner returned no campaign context")
    campaign = mapping(context.get("campaign"), "validated qualification campaign")
    producers = mapping(campaign.get("producers"), "qualification campaign producers")
    for name, module in (("runner", qualification_runner), ("evaluator", dev_eval)):
        entry = mapping(producers.get(name), f"qualification {name} producer")
        actual_path = Path(module.__file__).resolve(strict=True)
        require(
            Path(text(entry.get("path"), f"qualification {name} path")).resolve()
            == actual_path
            and entry.get("file_sha256") == gpu_stage.file_sha256(actual_path),
            f"controller loaded a non-campaign-bound qualification {name}",
        )
    path = Path(context["campaign_path"]).resolve(strict=True)
    receipt_path = training_receipt_path.expanduser().resolve(strict=True)
    receipt = load(receipt_path, "qualification training receipt")
    try:
        projection = qualification_runner.validate_training_receipt_value(
            campaign, path, receipt, receipt_path
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise QualificationCloseoutError(
            f"frozen qualification runner rejected training receipt: {error}"
        ) from error
    require(isinstance(projection, Mapping), "runner returned no receipt projection")
    checkpoints = projection.get("checkpoints")
    require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 1
        and checkpoints[0].get("global_step") == CHECKPOINT_STEP,
        "training receipt did not project the unique checkpoint-20 candidate",
    )
    return {
        **dict(context),
        "campaign": campaign,
        "campaign_path": path,
        "campaign_file_sha256": gpu_stage.file_sha256(path),
        "campaign_sha256": verify_self(campaign, "campaign_sha256", "qualification campaign"),
        "training_receipt": receipt,
        "training_receipt_path": receipt_path,
        "training_receipt_file_sha256": gpu_stage.file_sha256(receipt_path),
        "training_receipt_sha256": verify_self(
            receipt, "receipt_sha256", "qualification training receipt"
        ),
        "training_projection": projection,
        "checkpoint": dict(checkpoints[0]),
    }


def _control_authority(context: Mapping[str, Any]) -> dict[str, Any]:
    campaign = mapping(context["campaign"], "qualification campaign")
    bootcamp = absolute(campaign.get("bootcamp_root"), "campaign.bootcamp_root")
    require(bootcamp == BOOTCAMP_ROOT.resolve(strict=True), "bootcamp root binding drifted")
    goal2_path = bootcamp / GOAL2_REL
    goal2 = load(goal2_path, "goal-0002 charter")
    goal2_sha = verify_self(goal2, "charter_sha256", "goal-0002 charter")
    require(
        goal2.get("goal_id") == "goal-0002-day23-dpo"
        and goal2.get("status") == "frozen",
        "goal-0002 authority drifted",
    )
    hard_gates = mapping(goal2.get("global_hard_gates"), "goal-0002 hard gates")
    coding = dict(mapping(hard_gates.get("coding_guardrails"), "goal-0002 coding gates"))
    implementation_status = coding.pop("implementation_status", None)
    require(
        coding == EXPECTED_CODING_GATES
        and implementation_status == "not_yet_bound_blocks_heldout",
        "inherited coding hard gates drifted",
    )
    require(
        mapping(hard_gates.get("dev17_gate"), "goal-0002 dev gate") == EXPECTED_DEV_GATE,
        "inherited dev17 gate drifted",
    )
    heldout_gate_source = mapping(
        hard_gates.get("heldout29_gate"), "goal-0002 heldout gate"
    )
    require(
        heldout_gate_source.get("status") == "not_yet_frozen"
        and heldout_gate_source.get("authorization")
        == "blocked_until_append_only_guardrail_amendment"
        and heldout_gate_source.get("global_maximum_claims_after_authorization") == 1
        and not any(
            key in heldout_gate_source
            for key in (
                "positive_margin_pairs",
                "minimum_positive_pairs",
                "mean_reward_margin",
                "length_matched_margin",
            )
        ),
        "goal-0002 no longer has the expected missing heldout numerical gate",
    )
    leases = mapping(goal2.get("access_leases"), "goal-0002 access leases")
    heldout_lease = mapping(leases.get("heldout"), "goal-0002 heldout lease")
    heldout_identity = dict(mapping(heldout_lease.get("identity"), "heldout identity"))
    require(
        heldout_identity.get("records") == HELDOUT_RECORDS
        and heldout_lease.get("current_authorized_claims") == 0
        and heldout_lease.get("maximum_global_claims_after_append_only_authorization") == 1
        and heldout_lease.get("raw_content_access_before_claim") is False
        and heldout_lease.get("prerequisite_state") == "guardrails_passed",
        "inherited heldout lease drifted",
    )
    claim_relative = Path(text(heldout_lease.get("claim_path"), "heldout claim path"))
    require(
        not claim_relative.is_absolute() and ".." not in claim_relative.parts,
        "heldout claim path is not canonical",
    )
    claim_path = (bootcamp / "rsi-control" / claim_relative).resolve()
    require(
        claim_path.is_relative_to(bootcamp / "rsi-control/access-ledger"),
        "heldout claim escaped the inherited ledger",
    )
    goal_root = bootcamp / GOAL4_REL
    charter_path = goal_root / "charter.json"
    charter = load(charter_path, "goal-0004 charter")
    verify_self(charter, "charter_sha256", "goal-0004 charter")
    require(
        charter.get("goal_id") == GOAL_ID
        and charter.get("status") == "frozen"
        and mapping(mapping(charter.get("data_authority"), "goal4 data authority").get("heldout"), "goal4 heldout authority").get("claim_authorized") is False,
        "goal-0004 base charter boundary drifted",
    )
    event_paths = [
        goal_root / "events/event-000001-charter-frozen.json",
        goal_root / "events/event-000002-source-candidate-imported.json",
    ]
    events = [load(path, f"goal-0004 event {index}") for index, path in enumerate(event_paths, 1)]
    for index, event in enumerate(events, 1):
        verify_self(event, "event_sha256", f"goal-0004 event {index}")
        require(event.get("sequence") == index, "goal-0004 initial event sequence drifted")
    require(
        events[1].get("state_after") == "source_candidate_imported",
        "goal-0004 initial state drifted",
    )
    return {
        "bootcamp": bootcamp,
        "goal_root": goal_root,
        "goal2": goal2,
        "goal2_path": goal2_path,
        "goal2_sha256": goal2_sha,
        "goal4_charter": charter,
        "goal4_charter_path": charter_path,
        "initial_event_paths": event_paths,
        "initial_events": events,
        "heldout_identity": heldout_identity,
        "heldout_claim_path": claim_path,
    }


def _validate_dev(
    context: Mapping[str, Any],
    authority: Mapping[str, Any],
    dev_path: Path,
    *,
    expected_pass: bool = True,
) -> dict[str, Any]:
    campaign = mapping(context["campaign"], "qualification campaign")
    access = mapping(campaign.get("access_ledger"), "campaign access ledger")
    expected_path = absolute(access.get("dev_evaluation"), "campaign dev evaluation")
    path = dev_path.expanduser().resolve(strict=True)
    require(path == expected_path, "dev evaluation path is not canonical")
    failure_path = absolute(
        access.get("dev_failure_after_claim"), "campaign dev failure", must_exist=False
    )
    require(not failure_path.exists() and not failure_path.is_symlink(), "dev has an after-claim failure receipt")
    value = load(path, "qualification dev evaluation")
    dev_sha = verify_self(value, "evaluation_sha256", "qualification dev evaluation")
    require(
        value.get("schema_name") == DEV_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == ("pass" if expected_pass else "gate_fail")
        and value.get("run_id") == RUN_ID
        and value.get("split") == "dev",
        "dev evaluation identity/status drifted",
    )
    require(
        value.get("campaign")
        == {
            "path": str(context["campaign_path"]),
            "file_sha256": context["campaign_file_sha256"],
            "campaign_sha256": context["campaign_sha256"],
        },
        "dev campaign binding drifted",
    )
    bound_evaluator = mapping(
        mapping(campaign.get("producers"), "campaign producers").get("evaluator"),
        "campaign dev evaluator",
    )
    dev_producer = mapping(value.get("producer"), "dev producer")
    require(
        Path(text(dev_producer.get("path"), "dev producer path")).resolve()
        == Path(text(bound_evaluator.get("path"), "bound dev evaluator path")).resolve()
        and dev_producer.get("file_sha256") == bound_evaluator.get("file_sha256")
        and dev_producer.get("file_sha256")
        == gpu_stage.file_sha256(Path(dev_producer["path"]).resolve(strict=True)),
        "dev producer binding drifted",
    )
    checkpoint = mapping(value.get("checkpoint"), "dev checkpoint")
    require(
        checkpoint.get("global_step") == CHECKPOINT_STEP
        and checkpoint.get("training_receipt_path") == str(context["training_receipt_path"])
        and checkpoint.get("training_receipt_file_sha256")
        == context["training_receipt_file_sha256"]
        and checkpoint.get("training_receipt_sha256") == context["training_receipt_sha256"],
        "dev training/checkpoint binding drifted",
    )
    campaign_dev = mapping(
        mapping(campaign.get("datasets"), "campaign datasets").get("dev"),
        "campaign dev dataset",
    )
    receipt_dev = mapping(value.get("dataset"), "dev evaluation dataset")
    for key in (
        "file_sha256",
        "records",
        "ordered_pair_ids_sha256",
        "ordered_row_hashes_sha256",
        "ordered_source_pair_hashes_sha256",
    ):
        require(
            receipt_dev.get(key) == campaign_dev.get(key),
            f"dev dataset {key} binding drifted",
        )
    pair_results = value.get("pair_results")
    require(isinstance(pair_results, list) and len(pair_results) == 17, "dev pair result count drifted")
    for result in pair_results:
        row = mapping(result, "dev pair result")
        require(
            row.get("schema_name") == dev_eval.preference_eval.PAIR_SCHEMA
            and row.get("schema_version") == 1
            and row.get("status") == "pass"
            and row.get("split") == "dev",
            "dev pair result identity drifted",
        )
        verify_self(row, "pair_evaluation_sha256", "dev pair result")
        policy_chosen = finite_number(row.get("policy_chosen_logps"), "dev policy chosen")
        policy_rejected = finite_number(row.get("policy_rejected_logps"), "dev policy rejected")
        reference_chosen = finite_number(
            row.get("reference_chosen_logps"), "dev reference chosen"
        )
        reference_rejected = finite_number(
            row.get("reference_rejected_logps"), "dev reference rejected"
        )
        chosen_reward = finite_number(row.get("chosen_reward"), "dev chosen reward")
        rejected_reward = finite_number(row.get("rejected_reward"), "dev rejected reward")
        reward_margin = finite_number(row.get("reward_margin"), "dev reward margin")
        require(
            math.isclose(
                chosen_reward,
                0.1 * (policy_chosen - reference_chosen),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                rejected_reward,
                0.1 * (policy_rejected - reference_rejected),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                reward_margin,
                chosen_reward - rejected_reward,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "dev reward arithmetic drifted",
        )
    require(
        dev_eval.day23_contract.object_sha256([row["pair_id"] for row in pair_results])
        == receipt_dev.get("ordered_pair_ids_sha256")
        and dev_eval.day23_contract.object_sha256(
            [row["compiled_row_sha256"] for row in pair_results]
        )
        == receipt_dev.get("ordered_row_hashes_sha256")
        and dev_eval.day23_contract.object_sha256(
            [row["source_pair_sha256"] for row in pair_results]
        )
        == receipt_dev.get("ordered_source_pair_hashes_sha256"),
        "dev pair-result ordered identity drifted",
    )
    aggregate = dev_eval.preference_eval.aggregate_results(
        pair_results, split="dev", checkpoint_step=CHECKPOINT_STEP
    )
    gate = dev_eval._gate(aggregate)
    require(
        value.get("aggregate") == aggregate
        and value.get("eligibility") == gate
        and gate["passed"] is expected_pass,
        "dev gate outcome did not recompute exactly",
    )
    boundary = mapping(value.get("claim_boundary"), "dev claim boundary")
    require(
        boundary.get("fresh_process_checkpoint_reload_proven_before_dev_claim") is True
        and boundary.get("dev_global_claims") == 1
        and boundary.get("dev_candidate_evaluations") == 1
        and boundary.get("retry_allowed") is False
        and boundary.get("fallback_allowed") is False
        and boundary.get("dev_consumed") is True
        and boundary.get("heldout_consumed") is False,
        "dev one-shot boundary drifted",
    )
    claim_identity = mapping(value.get("access_claim"), "dev access claim")
    claim_path = absolute(access.get("dev_claim"), "global dev claim")
    _identity_matches(
        claim_identity,
        claim_path,
        self_field="claim_sha256",
        content_key="claim_sha256",
        label="dev access claim",
    )
    claim_value = load(claim_path, "global dev claim")
    expected_dev_identity = {
        key: campaign_dev.get(key)
        for key in (
            "file_sha256",
            "records",
            "ordered_pair_ids_sha256",
            "ordered_row_hashes_sha256",
            "ordered_source_pair_hashes_sha256",
        )
    }
    require(
        claim_value.get("schema_name") == dev_eval.CLAIM_SCHEMA
        and claim_value.get("schema_version") == 1
        and claim_value.get("status") == "claimed"
        and claim_value.get("split") == "dev"
        and claim_value.get("campaign") == value.get("campaign")
        and claim_value.get("training_receipt_sha256") == context["training_receipt_sha256"]
        and claim_value.get("dev_identity") == expected_dev_identity
        and claim_value.get("max_global_claims") == 1
        and claim_value.get("max_candidate_evaluations") == 1
        and claim_value.get("retry_allowed") is False
        and claim_value.get("fallback_allowed") is False
        and claim_value.get("heldout_authorized") is False
        and claim_value.get("heldout_consumed") is False,
        "global dev claim contract drifted",
    )
    training_process = mapping(
        context["training_receipt"].get("process_identity"), "training process identity"
    )
    dev_process = mapping(value.get("process_identity"), "dev process identity")
    require(
        (
            training_process.get("boot_id"),
            training_process.get("pid"),
            training_process.get("proc_start_ticks"),
        )
        != (
            dev_process.get("boot_id"),
            dev_process.get("pid"),
            dev_process.get("proc_start_ticks"),
        ),
        "dev evaluation was not a fresh-process checkpoint reload",
    )
    return {
        "value": value,
        "path": path,
        "file_sha256": gpu_stage.file_sha256(path),
        "evaluation_sha256": dev_sha,
        "claim_path": claim_path,
        "claim": claim_value,
    }


def _load_bound_module(
    entry: Mapping[str, Any], expected_name: str, module_name: str
) -> tuple[Any, Path]:
    path = absolute(entry.get("path"), f"{expected_name} producer")
    require(
        path.name == expected_name
        and path.parent == DAY23_DIR.resolve(strict=True)
        and entry.get("file_sha256") == gpu_stage.file_sha256(path),
        f"{expected_name} producer identity drifted",
    )
    specification = importlib.util.spec_from_file_location(module_name, path)
    require(
        specification is not None and specification.loader is not None,
        f"cannot import bound producer: {path}",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, path


def _validate_file_identity_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "path" in value and "file_sha256" in value:
            path = absolute(value.get("path"), f"{label}.path")
            require(
                value.get("file_sha256") == gpu_stage.file_sha256(path),
                f"{label} file hash drifted",
            )
            if "bytes" in value:
                require(value.get("bytes") == path.stat().st_size, f"{label} byte count drifted")
        for key, nested in value.items():
            _validate_file_identity_tree(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_file_identity_tree(nested, f"{label}[{index}]")


def _validate_evidence_binding(
    entry: Mapping[str, Any],
    expected_path: Path,
    expected_file_sha: str,
    expected_content_sha: str,
    content_key: str,
    label: str,
) -> None:
    require(
        Path(text(entry.get("path"), f"{label}.path")).resolve() == expected_path.resolve()
        and entry.get("file_sha256") == expected_file_sha
        and entry.get(content_key) == expected_content_sha,
        f"{label} binding drifted",
    )


def _validate_full112(
    context: Mapping[str, Any], dev: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    expected = Path(context["run_root"]) / "evidence/guardrails/full112-generation-and-score.json"
    failure = Path(context["run_root"]) / "evidence/guardrails/full112-generation-and-score-failure.json"
    path = receipt_path.expanduser().resolve(strict=True)
    require(path == expected, "full112 receipt path is not canonical")
    require(not failure.exists() and not failure.is_symlink(), "full112 has a failure receipt")
    value = load(path, "full112 guardrail receipt")
    receipt_sha = verify_self(value, "receipt_sha256", "full112 guardrail receipt")
    require(
        value.get("schema_name") == FULL112_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "pass_pending_code_sandbox",
        "full112 receipt is not sandbox-ready",
    )
    producer, _ = _load_bound_module(
        mapping(value.get("producer"), "full112 producer"),
        "eval_day23_full112_guardrails.py",
        "_day23_closeout_bound_full112",
    )
    validator = getattr(producer, "validate_receipt", None)
    require(callable(validator), "full112 producer has no receipt validator")
    try:
        validated = validator(path)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise QualificationCloseoutError(
            f"bound full112 producer rejected receipt: {error}"
        ) from error
    require(validated == value, "full112 producer validated different receipt bytes")
    require(
        value.get("campaign")
        == {
            "path": str(context["campaign_path"]),
            "file_sha256": context["campaign_file_sha256"],
            "campaign_sha256": context["campaign_sha256"],
        },
        "full112 campaign binding drifted",
    )
    _validate_evidence_binding(
        mapping(value.get("training_receipt"), "full112 training receipt"),
        context["training_receipt_path"],
        context["training_receipt_file_sha256"],
        context["training_receipt_sha256"],
        "content_sha256",
        "full112 training receipt",
    )
    _validate_evidence_binding(
        mapping(value.get("dev_evaluation"), "full112 dev evaluation"),
        dev["path"],
        dev["file_sha256"],
        dev["evaluation_sha256"],
        "content_sha256",
        "full112 dev evaluation",
    )
    aggregate = mapping(value.get("aggregate"), "full112 aggregate")
    gate = mapping(value.get("gate"), "full112 gate")
    require(
        aggregate.get("records") == 112
        and aggregate.get("records_by_skill")
        == {"code": 28, "finance": 28, "general": 28, "math": 28}
        and aggregate.get("code_correct") is None
        and aggregate.get("total_correct") is None
        and gate.get("passed_pre_sandbox") is True,
        "full112 pre-sandbox gate drifted",
    )
    boundary = mapping(value.get("claim_boundary"), "full112 claim boundary")
    require(
        boundary.get("fresh_process_checkpoint_reload") is True
        and boundary.get("candidate_code_executed_on_host") is False
        and boundary.get("code_and_total_gates_require_e2b") is True
        and boundary.get("final_qualification_claimed") is False,
        "full112 claim boundary drifted",
    )
    return {
        "value": value,
        "path": path,
        "file_sha256": gpu_stage.file_sha256(path),
        "receipt_sha256": receipt_sha,
        "aggregate": dict(aggregate),
    }


def _validate_e2b(
    context: Mapping[str, Any],
    dev: Mapping[str, Any],
    full112: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    expected = Path(context["run_root"]) / "evidence/guardrails/e2b-sandbox.json"
    failure = Path(context["run_root"]) / "evidence/guardrails/e2b-sandbox-failure.json"
    path = receipt_path.expanduser().resolve(strict=True)
    require(path == expected, "E2B receipt path is not canonical")
    require(not failure.exists() and not failure.is_symlink(), "E2B has a failure receipt")
    value = load(path, "E2B guardrail receipt")
    receipt_sha = verify_self(value, "receipt_sha256", "E2B guardrail receipt")
    require(
        value.get("schema_name") == E2B_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "pass"
        and value.get("goal_id") == GOAL_ID
        and value.get("version_id") == VERSION_ID
        and value.get("run_id") == RUN_ID,
        "E2B receipt identity/status drifted",
    )
    producer, _ = _load_bound_module(
        mapping(value.get("producer"), "E2B producer"),
        "score_day23_code_e2b_guardrails.py",
        "_day23_closeout_bound_e2b",
    )
    require(
        value.get("campaign")
        == {
            "path": str(context["campaign_path"]),
            "file_sha256": context["campaign_file_sha256"],
            "campaign_sha256": context["campaign_sha256"],
        },
        "E2B campaign binding drifted",
    )
    _validate_evidence_binding(
        mapping(value.get("training_receipt"), "E2B training receipt"),
        context["training_receipt_path"],
        context["training_receipt_file_sha256"],
        context["training_receipt_sha256"],
        "receipt_sha256",
        "E2B training receipt",
    )
    dev_binding = mapping(value.get("dev_evaluation"), "E2B dev evaluation")
    _validate_evidence_binding(
        dev_binding,
        dev["path"],
        dev["file_sha256"],
        dev["evaluation_sha256"],
        "evaluation_sha256",
        "E2B dev evaluation",
    )
    require(dev_binding.get("status") == "pass", "E2B bound a non-PASS dev result")
    full_binding = mapping(value.get("full112_receipt"), "E2B full112 receipt")
    _validate_evidence_binding(
        full_binding,
        full112["path"],
        full112["file_sha256"],
        full112["receipt_sha256"],
        "receipt_sha256",
        "E2B full112 receipt",
    )
    require(
        full_binding.get("status") == "pass_pending_code_sandbox",
        "E2B bound a non-ready full112 receipt",
    )
    metrics = mapping(value.get("metrics"), "E2B metrics")
    require(
        metrics.get("records") == metrics.get("sandbox_execution_eligible")
        and metrics.get("passed") == metrics.get("code_correct")
        and metrics.get("sandbox_execution_eligible")
        == metrics.get("code_sandbox_execution_eligible"),
        "E2B metric aliases/accounting drifted",
    )
    combined = recompute_combined_guardrails(full112["aggregate"], metrics)
    gate_function = getattr(producer, "evaluate_gate", None)
    require(callable(gate_function), "E2B producer has no gate recomputation API")
    producer_gate = gate_function(full112["aggregate"], metrics)
    require(
        value.get("gate") == producer_gate
        and producer_gate.get("passed") is True
        and combined["passed"] is True
        and producer_gate.get("observed") == combined["observed"],
        "combined inherited coding hard gates did not pass exactly",
    )
    _validate_file_identity_tree(value.get("frozen_e2b"), "E2B frozen evidence")
    sandbox_boundary = mapping(value.get("sandbox_input_boundary"), "E2B sandbox boundary")
    require(
        sandbox_boundary.get("full112_records_verified") == 112
        and sandbox_boundary.get("code_records_selected") == 28
        and sandbox_boundary.get("non_code_records_executed") == 0
        and sandbox_boundary.get("candidate_code_executed_on_host") is False
        and sandbox_boundary.get("heldout_consumed") is False,
        "E2B sandbox input boundary drifted",
    )
    claim = mapping(value.get("claim_boundary"), "E2B claim boundary")
    require(
        claim.get("dev_pass_required") is True
        and claim.get("sandbox_execution_performed") is True
        and claim.get("heldout_interface_exposed") is False
        and claim.get("heldout_consumed") is False,
        "E2B protected-data boundary drifted",
    )
    return {
        "value": value,
        "path": path,
        "file_sha256": gpu_stage.file_sha256(path),
        "receipt_sha256": receipt_sha,
        "combined_gate": combined,
    }


def _event(
    *,
    event_id: str,
    sequence: int,
    created_at: str,
    event_type: str,
    state_before: str,
    state_after: str,
    prior_event: Mapping[str, Any],
    authority_refs: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return seal(
        {
            "schema_name": EVENT_SCHEMA,
            "schema_version": 1,
            "event_id": event_id,
            "created_at_utc": created_at,
            "sequence": sequence,
            "event_type": event_type,
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "state_before": state_before,
            "state_after": state_after,
            "prior_event": dict(prior_event),
            "authority_refs": [dict(item) for item in authority_refs],
            "payload": dict(payload),
        },
        "event_sha256",
    )


def _authorize_gate_contract() -> dict[str, Any]:
    return {
        "records": HELDOUT_RECORDS,
        "finite_records": {"operator": "==", "threshold": HELDOUT_RECORDS},
        "positive_margin_pairs": {
            "operator": ">=",
            "threshold": HELDOUT_MINIMUM_POSITIVE_PAIRS,
            "denominator": HELDOUT_RECORDS,
        },
        "mean_reward_margin": {"operator": ">", "threshold": 0.0},
        "length_matched_mean_reward_margin": {"operator": ">", "threshold": 0.0},
        "derivation": {
            "method": "one_sided_exact_sign_test_under_p_equals_0p5",
            "alpha_ceiling": 0.025,
            "source_dev_gate": "13_of_17",
            "source_dev_tail_probability": DEV_SIGN_TEST_P,
            "heldout_first_passing_integer": 21,
            "heldout_21_of_29_tail_probability": HELDOUT_21_SIGN_TEST_P,
            "adjacent_20_of_29_tail_probability": HELDOUT_20_SIGN_TEST_P,
            "data_independent": True,
            "heldout_rows_inspected_for_derivation": 0,
        },
        "threshold_relaxation_forbidden": True,
        "retry_after_claim": "forbidden",
        "runner_up_fallback": "forbidden",
    }


def _dev_failure_documents(
    state: Mapping[str, Any], created_at: str
) -> list[tuple[Path, dict[str, Any], str]]:
    context = state["context"]
    authority = state["authority"]
    dev = state["dev"]
    goal_root: Path = authority["goal_root"]
    event3_path = goal_root / f"events/{EVENT3_NAME}.json"
    event4_path = goal_root / f"events/{EVENT4_FAILURE_NAME}.json"
    charter_ref = file_identity(
        authority["goal4_charter_path"],
        self_field="charter_sha256",
        relative_to=authority["bootcamp"],
    )
    training_ref = file_identity(
        context["training_receipt_path"], self_field="receipt_sha256"
    )
    dev_ref = file_identity(dev["path"], self_field="evaluation_sha256")
    initial_prior = event_identity(
        authority["initial_event_paths"][1], authority["initial_events"][1], goal_root
    )
    event3 = _event(
        event_id=EVENT3_NAME,
        sequence=3,
        created_at=created_at,
        event_type="full154-refit-completed",
        state_before="source_candidate_imported",
        state_after="full154_refit_completed",
        prior_event=initial_prior,
        authority_refs=[charter_ref, training_ref],
        payload={
            "run_id": RUN_ID,
            "candidate_checkpoint_step": CHECKPOINT_STEP,
            "fresh_s1_full154_refit": True,
            "training_receipt": training_ref,
            "dev_consumed": False,
            "heldout_consumed": False,
        },
    )
    event4 = _event(
        event_id=EVENT4_FAILURE_NAME,
        sequence=4,
        created_at=created_at,
        event_type="dev-terminal-no-go",
        state_before="full154_refit_completed",
        state_after="terminal_dev_not_qualified",
        prior_event=event_identity(event3_path, event3, goal_root),
        authority_refs=[charter_ref, training_ref, dev_ref],
        payload={
            "status": "terminal_no_go",
            "reason": "one_shot_dev_gate_failed",
            "dev_evaluation": dev_ref,
            "dev_gate": dev["value"]["eligibility"],
            "global_dev_claims": 1,
            "dev_candidate_evaluations": 1,
            "scientific_retry_allowed": False,
            "runner_up_fallback_allowed": False,
            "full112_guardrail_authorized": False,
            "e2b_guardrail_authorized": False,
            "heldout_authorized": False,
            "heldout_claims": 0,
            "heldout_rows_opened": 0,
            "qualification_scope": "fixed_experimental_suite_only",
            "deployment_ready_claim": False,
            "general_improvement_claim": False,
        },
    )
    return [
        (event3_path, event3, "event_sha256"),
        (event4_path, event4, "event_sha256"),
    ]


def _prepare_dev_failure(args: argparse.Namespace) -> dict[str, Any]:
    context = _campaign_training_context(args.campaign, args.training_receipt)
    authority = _control_authority(context)
    require(
        not authority["heldout_claim_path"].exists()
        and not authority["heldout_claim_path"].is_symlink(),
        "heldout was touched despite terminal dev failure",
    )
    dev = _validate_dev(
        context, authority, args.dev_evaluation, expected_pass=False
    )
    run_root = Path(context["run_root"])
    forbidden_downstream = (
        run_root / "evidence/guardrails/full112-generation-and-score.json",
        run_root / "evidence/guardrails/full112-generation-and-score-failure.json",
        run_root / "evidence/guardrails/e2b-sandbox.json",
        run_root / "evidence/guardrails/e2b-sandbox-failure.json",
        run_root / "evidence/authority/heldout-authorization.json",
        run_root / "evidence/heldout/finalist-heldout.json",
        run_root / "evidence/heldout/heldout-failure-after-claim.json",
        authority["goal_root"] / f"amendments/{AMENDMENT_NAME}.json",
        authority["goal_root"] / f"events/{EVENT5_NAME}.json",
        authority["goal_root"] / f"events/{EVENT6_NAME}.json",
    )
    for path in forbidden_downstream:
        require(
            not path.exists() and not path.is_symlink(),
            f"downstream evidence exists after terminal dev failure: {path}",
        )
    return {"context": context, "authority": authority, "dev": dev}


def execute_dev_failure(args: argparse.Namespace) -> dict[str, Any]:
    created_at = validate_created_at(args.created_at_utc)
    state = _prepare_dev_failure(args)
    documents = _dev_failure_documents(state, created_at)
    conflicting_event = (
        state["authority"]["goal_root"] / f"events/{EVENT4_NAME}.json"
    )
    require(
        not conflicting_event.exists() and not conflicting_event.is_symlink(),
        "PASS dev event conflicts with terminal dev result",
    )
    if args.mode == "preflight":
        _documents_match(documents, require_present=False)
        status = "preflight_pass"
    elif args.mode == "check":
        _documents_match(documents, require_present=True)
        status = "terminal_dev_not_qualified"
    else:
        for path, value, _ in documents:
            write_exclusive_or_verify(path, value)
        _documents_match(documents, require_present=True)
        status = "terminal_dev_not_qualified"
    return {
        "status": status,
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "terminal_state": "terminal_dev_not_qualified",
        "training_receipt_sha256": state["context"]["training_receipt_sha256"],
        "dev_evaluation_sha256": state["dev"]["evaluation_sha256"],
        "dev_gate": state["dev"]["value"]["eligibility"],
        "terminal_event": str(documents[-1][0]),
        "event_sha256": documents[-1][1]["event_sha256"],
        "full112_guardrail_authorized": False,
        "e2b_guardrail_authorized": False,
        "heldout_authorized": False,
        "heldout_claims": 0,
        "scientific_retry_allowed": False,
    }


def _authorize_documents(
    state: Mapping[str, Any], created_at: str
) -> list[tuple[Path, dict[str, Any], str]]:
    context = state["context"]
    authority = state["authority"]
    dev = state["dev"]
    full112 = state["full112"]
    e2b = state["e2b"]
    evaluator_identity = state["heldout_evaluator"]
    goal_root: Path = authority["goal_root"]
    run_root = Path(context["run_root"])
    event3_path = goal_root / f"events/{EVENT3_NAME}.json"
    event4_path = goal_root / f"events/{EVENT4_NAME}.json"
    event5_path = goal_root / f"events/{EVENT5_NAME}.json"
    amendment_path = goal_root / f"amendments/{AMENDMENT_NAME}.json"
    authorization_path = run_root / "evidence/authority/heldout-authorization.json"
    heldout_output = run_root / "evidence/heldout/finalist-heldout.json"
    heldout_failure = run_root / "evidence/heldout/heldout-failure-after-claim.json"

    charter_ref = file_identity(
        authority["goal4_charter_path"], self_field="charter_sha256", relative_to=authority["bootcamp"]
    )
    goal2_ref = file_identity(
        authority["goal2_path"], self_field="charter_sha256", relative_to=authority["bootcamp"]
    )
    training_ref = file_identity(context["training_receipt_path"], self_field="receipt_sha256")
    dev_ref = file_identity(dev["path"], self_field="evaluation_sha256")
    full112_ref = file_identity(full112["path"], self_field="receipt_sha256")
    e2b_ref = file_identity(e2b["path"], self_field="receipt_sha256")
    initial_prior = event_identity(
        authority["initial_event_paths"][1], authority["initial_events"][1], goal_root
    )
    event3 = _event(
        event_id=EVENT3_NAME,
        sequence=3,
        created_at=created_at,
        event_type="full154-refit-completed",
        state_before="source_candidate_imported",
        state_after="full154_refit_completed",
        prior_event=initial_prior,
        authority_refs=[charter_ref, training_ref],
        payload={
            "run_id": RUN_ID,
            "candidate_checkpoint_step": CHECKPOINT_STEP,
            "fresh_s1_full154_refit": True,
            "training_receipt": training_ref,
            "dev_consumed": False,
            "heldout_consumed": False,
        },
    )
    event3_ref = event_identity(event3_path, event3, goal_root)
    event4 = _event(
        event_id=EVENT4_NAME,
        sequence=4,
        created_at=created_at,
        event_type="dev-qualified",
        state_before="full154_refit_completed",
        state_after="dev_qualified_guardrails_pending",
        prior_event=event3_ref,
        authority_refs=[charter_ref, training_ref, dev_ref],
        payload={
            "dev_evaluation": dev_ref,
            "gate": dev["value"]["eligibility"],
            "global_dev_claims": 1,
            "retry_allowed": False,
            "fallback_allowed": False,
            "heldout_consumed": False,
        },
    )
    event4_ref = event_identity(event4_path, event4, goal_root)

    heldout_gate_contract = _authorize_gate_contract()
    amendment = seal(
        {
            "schema_name": AMENDMENT_SCHEMA,
            "schema_version": 1,
            "status": "frozen",
            "amendment_id": AMENDMENT_NAME,
            "created_at_utc": created_at,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "campaign_id": CAMPAIGN_ID,
            "append_only": True,
            "base_charter": charter_ref,
            "inherited_goal2_charter": goal2_ref,
            "prior_event": event4_ref,
            "reason": "goal-0002 reserved one future claim but contained no heldout29 numerical pass gate",
            "preconditions": {
                "training_receipt": training_ref,
                "dev_evaluation": dev_ref,
                "full112_guardrail": full112_ref,
                "e2b_guardrail": e2b_ref,
                "combined_coding_hard_gate": e2b["combined_gate"],
            },
            "heldout_gate": heldout_gate_contract,
            "heldout_authority": {
                "dataset_identity": authority["heldout_identity"],
                "global_claim_path": str(authority["heldout_claim_path"]),
                "maximum_global_claims": 1,
                "maximum_candidate_evaluations": 1,
                "raw_rows_before_claim": "forbidden",
                "heldout_evaluator": evaluator_identity,
                "evaluation_path": str(heldout_output),
                "failure_after_claim_path": str(heldout_failure),
                "evaluation_schema_name": HELDOUT_EVALUATION_SCHEMA,
                "failure_schema_name": HELDOUT_FAILURE_SCHEMA,
                "claim_schema_name": HELDOUT_CLAIM_SCHEMA,
                "retry_allowed": False,
                "fallback_allowed": False,
            },
            "producer": file_identity(Path(__file__).resolve(strict=True)),
            "claim_boundary": {
                "dev_consumed": True,
                "guardrails_passed": True,
                "heldout_claimed": False,
                "heldout_rows_opened": 0,
            },
        },
        "amendment_sha256",
    )
    amendment_ref = payload_identity(
        amendment_path, amendment, "amendment_sha256", relative_to=authority["bootcamp"]
    )
    event5 = _event(
        event_id=EVENT5_NAME,
        sequence=5,
        created_at=created_at,
        event_type="guardrails-passed-heldout-authorized",
        state_before="dev_qualified_guardrails_pending",
        state_after="guardrails_passed_heldout_authorized",
        prior_event=event4_ref,
        authority_refs=[charter_ref, amendment_ref, full112_ref, e2b_ref],
        payload={
            "combined_coding_hard_gate": e2b["combined_gate"],
            "heldout_gate": heldout_gate_contract,
            "heldout_evaluator": evaluator_identity,
            "maximum_global_claims": 1,
            "maximum_candidate_evaluations": 1,
            "claim_created": False,
            "heldout_rows_opened": 0,
            "retry_allowed": False,
            "fallback_allowed": False,
        },
    )
    event5_ref = event_identity(event5_path, event5, goal_root)
    event5_authority_ref = payload_identity(
        event5_path, event5, "event_sha256", relative_to=authority["bootcamp"]
    )
    event5_authority_ref["event_sha256"] = event5_authority_ref.pop("content_sha256")
    authorization = seal(
        {
            "schema_name": AUTHORIZATION_SCHEMA,
            "schema_version": 1,
            "status": "authorized_unclaimed",
            "created_at_utc": created_at,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_id": RUN_ID,
            "campaign": {
                "path": str(context["campaign_path"]),
                "file_sha256": context["campaign_file_sha256"],
                "campaign_sha256": context["campaign_sha256"],
            },
            "training_receipt": training_ref,
            "checkpoint": context["checkpoint"],
            "dev_evaluation": dev_ref,
            "full112_guardrail": full112_ref,
            "e2b_guardrail": e2b_ref,
            "append_only_amendment": amendment_ref,
            "authorization_event": event5_authority_ref,
            "heldout_evaluator": evaluator_identity,
            "heldout_dataset": authority["heldout_identity"],
            "heldout_gate": heldout_gate_contract,
            "global_claim_path": str(authority["heldout_claim_path"]),
            "evaluation_path": str(heldout_output),
            "failure_after_claim_path": str(heldout_failure),
            "maximum_global_claims": 1,
            "maximum_candidate_evaluations": 1,
            "retry_allowed": False,
            "fallback_allowed": False,
            "producer": file_identity(Path(__file__).resolve(strict=True)),
            "claim_boundary": {
                "all_prerequisites_passed": True,
                "claim_must_use_o_excl": True,
                "fresh_process_checkpoint_reload_required_before_claim": True,
                "raw_rows_before_claim": "forbidden",
                "heldout_claimed_at_authorization": False,
                "heldout_rows_opened_at_authorization": 0,
            },
        },
        "authorization_sha256",
    )
    return [
        (event3_path, event3, "event_sha256"),
        (event4_path, event4, "event_sha256"),
        (amendment_path, amendment, "amendment_sha256"),
        (event5_path, event5, "event_sha256"),
        (authorization_path, authorization, "authorization_sha256"),
    ]


def _prepare_authorize(args: argparse.Namespace) -> dict[str, Any]:
    context = _campaign_training_context(args.campaign, args.training_receipt)
    authority = _control_authority(context)
    require(
        not authority["heldout_claim_path"].exists()
        and not authority["heldout_claim_path"].is_symlink(),
        "global heldout lease was already claimed",
    )
    dev = _validate_dev(context, authority, args.dev_evaluation)
    full112 = _validate_full112(context, dev, args.full112_receipt)
    e2b = _validate_e2b(context, dev, full112, args.e2b_receipt)
    evaluator_path = args.heldout_evaluator.expanduser().resolve(strict=True)
    require(
        evaluator_path == DAY23_DIR / "eval_day23_qualification_heldout.py"
        and evaluator_path.is_file()
        and not evaluator_path.is_symlink(),
        "heldout evaluator path is not the independent canonical executable",
    )
    evaluator_identity = file_identity(evaluator_path)
    run_root = Path(context["run_root"])
    for path, label in (
        (run_root / "evidence/heldout/finalist-heldout.json", "heldout evaluation"),
        (run_root / "evidence/heldout/heldout-failure-after-claim.json", "heldout failure"),
    ):
        require(not path.exists() and not path.is_symlink(), f"{label} already exists")
    return {
        "context": context,
        "authority": authority,
        "dev": dev,
        "full112": full112,
        "e2b": e2b,
        "heldout_evaluator": evaluator_identity,
    }


def _documents_match(
    documents: Sequence[tuple[Path, Mapping[str, Any], str]], *, require_present: bool
) -> None:
    for path, expected, field in documents:
        if not require_present:
            require(not path.exists() and not path.is_symlink(), f"append-only output already exists: {path}")
            continue
        actual = load(path, f"sealed {path.name}")
        verify_self(actual, field, f"sealed {path.name}")
        require(actual == expected, f"sealed output no longer recomputes: {path}")


def execute_authorize(args: argparse.Namespace) -> dict[str, Any]:
    created_at = validate_created_at(args.created_at_utc)
    state = _prepare_authorize(args)
    documents = _authorize_documents(state, created_at)
    if args.mode == "preflight":
        _documents_match(documents, require_present=False)
        return {
            "status": "preflight_pass",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "combined_coding_hard_gate": state["e2b"]["combined_gate"],
            "heldout_minimum_positive_pairs": HELDOUT_MINIMUM_POSITIVE_PAIRS,
            "heldout_global_claims_authorized": 1,
            "heldout_claim_created": False,
            "heldout_rows_opened": 0,
        }
    if args.mode == "check":
        _documents_match(documents, require_present=True)
    else:
        for path, value, _ in documents:
            write_exclusive_or_verify(path, value)
        _documents_match(documents, require_present=True)
    authorization_path = documents[-1][0]
    return {
        "status": "authorized_unclaimed",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "authorization": str(authorization_path),
        "authorization_sha256": documents[-1][1]["authorization_sha256"],
        "heldout_evaluator_file_sha256": state["heldout_evaluator"]["file_sha256"],
        "heldout_minimum_positive_pairs": HELDOUT_MINIMUM_POSITIVE_PAIRS,
        "heldout_global_claims_authorized": 1,
        "heldout_claim_created": False,
        "heldout_rows_opened": 0,
    }


def validate_authorization_for_evaluator(
    authorization_path: Path, *, require_claim_unopened: bool = True
) -> dict[str, Any]:
    """Recompute authorization before the evaluator creates the global claim."""
    path = authorization_path.expanduser().resolve(strict=True)
    authorization = load(path, "heldout authorization")
    verify_self(authorization, "authorization_sha256", "heldout authorization")
    require(
        authorization.get("schema_name") == AUTHORIZATION_SCHEMA
        and authorization.get("schema_version") == 1
        and authorization.get("status") == "authorized_unclaimed",
        "heldout authorization identity/status drifted",
    )
    arguments = argparse.Namespace(
        campaign=Path(text(mapping(authorization.get("campaign"), "authorization campaign").get("path"), "authorization campaign path")),
        training_receipt=Path(text(mapping(authorization.get("training_receipt"), "authorization training receipt").get("path"), "authorization training receipt path")),
        dev_evaluation=Path(text(mapping(authorization.get("dev_evaluation"), "authorization dev evaluation").get("path"), "authorization dev evaluation path")),
        full112_receipt=Path(text(mapping(authorization.get("full112_guardrail"), "authorization full112 receipt").get("path"), "authorization full112 path")),
        e2b_receipt=Path(text(mapping(authorization.get("e2b_guardrail"), "authorization E2B receipt").get("path"), "authorization E2B path")),
        heldout_evaluator=Path(text(mapping(authorization.get("heldout_evaluator"), "authorization evaluator").get("path"), "authorization evaluator path")),
    )
    state = _prepare_authorize(arguments)
    documents = _authorize_documents(
        state, text(authorization.get("created_at_utc"), "authorization created_at_utc")
    )
    _documents_match(documents, require_present=True)
    require(path == documents[-1][0] and authorization == documents[-1][1], "authorization is not canonical")
    if require_claim_unopened:
        claim_path: Path = state["authority"]["heldout_claim_path"]
        require(
            not claim_path.exists() and not claim_path.is_symlink(),
            "global heldout lease was already claimed",
        )
    return {
        **state,
        "authorization": authorization,
        "authorization_path": path,
        "authorization_file_sha256": gpu_stage.file_sha256(path),
        "authorization_sha256": authorization["authorization_sha256"],
    }


def _static_authorization(authorization_path: Path) -> dict[str, Any]:
    """Validate immutable authorization after heldout has consumed its claim."""
    path = authorization_path.expanduser().resolve(strict=True)
    value = load(path, "heldout authorization")
    authorization_sha = verify_self(value, "authorization_sha256", "heldout authorization")
    require(
        value.get("schema_name") == AUTHORIZATION_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "authorized_unclaimed"
        and value.get("goal_id") == GOAL_ID
        and value.get("version_id") == VERSION_ID
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("run_id") == RUN_ID,
        "heldout authorization identity drifted",
    )
    run_root = path.parent.parent.parent
    require(path == run_root / "evidence/authority/heldout-authorization.json", "authorization path is not canonical")
    producer = mapping(value.get("producer"), "authorization producer")
    require(
        Path(text(producer.get("path"), "authorization producer path")).resolve()
        == Path(__file__).resolve(strict=True)
        and producer.get("file_sha256") == gpu_stage.file_sha256(Path(__file__).resolve(strict=True)),
        "authorization producer drifted",
    )
    evaluator = mapping(value.get("heldout_evaluator"), "authorized heldout evaluator")
    evaluator_path = absolute(evaluator.get("path"), "authorized heldout evaluator")
    require(
        evaluator_path == DAY23_DIR / "eval_day23_qualification_heldout.py"
        and evaluator.get("file_sha256") == gpu_stage.file_sha256(evaluator_path),
        "authorized evaluator drifted",
    )
    campaign_entry = mapping(value.get("campaign"), "authorization campaign")
    campaign_path = absolute(campaign_entry.get("path"), "authorization campaign path")
    campaign = load(campaign_path, "qualification campaign")
    require(
        verify_self(campaign, "campaign_sha256", "qualification campaign")
        == campaign_entry.get("campaign_sha256")
        and campaign_entry.get("file_sha256") == gpu_stage.file_sha256(campaign_path)
        and Path(text(campaign.get("remote_run_root"), "campaign run root")).resolve() == run_root,
        "authorization campaign bytes drifted",
    )
    references = (
        ("training_receipt", "receipt_sha256", "content_sha256"),
        ("dev_evaluation", "evaluation_sha256", "content_sha256"),
        ("full112_guardrail", "receipt_sha256", "content_sha256"),
        ("e2b_guardrail", "receipt_sha256", "content_sha256"),
        ("append_only_amendment", "amendment_sha256", "content_sha256"),
        ("authorization_event", "event_sha256", "event_sha256"),
    )
    resolved_refs: dict[str, dict[str, Any]] = {}
    for key, self_field, content_key in references:
        entry = mapping(value.get(key), f"authorization {key}")
        raw_path = Path(text(entry.get("path"), f"authorization {key} path"))
        referenced = (
            raw_path.resolve(strict=True)
            if raw_path.is_absolute()
            else (BOOTCAMP_ROOT / raw_path).resolve(strict=True)
        )
        document = load(referenced, f"authorization {key}")
        content_sha = verify_self(document, self_field, f"authorization {key}")
        require(
            entry.get("file_sha256") == gpu_stage.file_sha256(referenced)
            and entry.get(content_key) == content_sha,
            f"authorization {key} identity drifted",
        )
        resolved_refs[key] = {"path": referenced, "value": document}
    expected_gate = _authorize_gate_contract()
    require(
        value.get("heldout_gate") == expected_gate
        and value.get("maximum_global_claims") == 1
        and value.get("maximum_candidate_evaluations") == 1
        and value.get("retry_allowed") is False
        and value.get("fallback_allowed") is False,
        "authorization heldout gate/budget drifted",
    )
    claim_path = absolute(value.get("global_claim_path"), "authorized global claim", must_exist=False)
    heldout_output = absolute(value.get("evaluation_path"), "authorized heldout output", must_exist=False)
    failure_path = absolute(value.get("failure_after_claim_path"), "authorized heldout failure", must_exist=False)
    require(
        heldout_output == run_root / "evidence/heldout/finalist-heldout.json"
        and failure_path == run_root / "evidence/heldout/heldout-failure-after-claim.json",
        "authorization heldout output paths drifted",
    )
    return {
        "authorization": value,
        "authorization_path": path,
        "authorization_file_sha256": gpu_stage.file_sha256(path),
        "authorization_sha256": authorization_sha,
        "run_root": run_root,
        "campaign": campaign,
        "campaign_path": campaign_path,
        "evaluator_path": evaluator_path,
        "claim_path": claim_path,
        "heldout_output": heldout_output,
        "failure_path": failure_path,
        "references": resolved_refs,
    }


def _validate_heldout_evaluation(
    static: Mapping[str, Any], evaluation_path: Path
) -> dict[str, Any]:
    path = evaluation_path.expanduser().resolve(strict=True)
    require(path == static["heldout_output"], "heldout evaluation path is not canonical")
    require(
        not static["failure_path"].exists() and not static["failure_path"].is_symlink(),
        "heldout evaluation coexists with an after-claim failure receipt",
    )
    value = load(path, "heldout evaluation")
    evaluation_sha = verify_self(value, "evaluation_sha256", "heldout evaluation")
    require(
        value.get("schema_name") == HELDOUT_EVALUATION_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") in {"pass", "gate_fail"}
        and value.get("goal_id") == GOAL_ID
        and value.get("version_id") == VERSION_ID
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("run_id") == RUN_ID
        and value.get("split") == "heldout",
        "heldout evaluation identity/status drifted",
    )
    authorization = static["authorization"]
    require(
        value.get("authorization")
        == {
            "path": str(static["authorization_path"]),
            "file_sha256": static["authorization_file_sha256"],
            "authorization_sha256": static["authorization_sha256"],
        }
        and value.get("campaign") == authorization.get("campaign")
        and value.get("training_receipt") == authorization.get("training_receipt")
        and value.get("checkpoint") == authorization.get("checkpoint")
        and value.get("dev_evaluation") == authorization.get("dev_evaluation")
        and value.get("full112_guardrail") == authorization.get("full112_guardrail")
        and value.get("e2b_guardrail") == authorization.get("e2b_guardrail")
        and value.get("dataset") == authorization.get("heldout_dataset")
        and value.get("producer") == authorization.get("heldout_evaluator"),
        "heldout evaluation authority binding drifted",
    )
    claim_identity = mapping(value.get("access_claim"), "heldout access claim identity")
    _identity_matches(
        claim_identity,
        static["claim_path"],
        self_field="claim_sha256",
        content_key="claim_sha256",
        label="heldout access claim",
    )
    claim = load(static["claim_path"], "global heldout access claim")
    require(
        claim.get("schema_name") == HELDOUT_CLAIM_SCHEMA
        and claim.get("schema_version") == 1
        and claim.get("status") == "claimed"
        and claim.get("split") == "heldout"
        and claim.get("authorization_sha256") == static["authorization_sha256"]
        and claim.get("max_global_claims") == 1
        and claim.get("max_candidate_evaluations") == 1
        and claim.get("retry_allowed") is False
        and claim.get("fallback_allowed") is False
        and claim.get("producer") == authorization.get("heldout_evaluator"),
        "global heldout claim contract drifted",
    )
    pair_results = value.get("pair_results")
    require(
        isinstance(pair_results, list) and len(pair_results) == HELDOUT_RECORDS,
        "heldout pair result count drifted",
    )
    for result in pair_results:
        row = mapping(result, "heldout pair result")
        require(
            row.get("schema_name") == dev_eval.preference_eval.PAIR_SCHEMA
            and row.get("schema_version") == 1
            and row.get("status") == "pass"
            and row.get("split") == "heldout",
            "heldout pair result identity drifted",
        )
        verify_self(row, "pair_evaluation_sha256", "heldout pair result")
        for key in (
            "policy_chosen_logps",
            "policy_rejected_logps",
            "reference_chosen_logps",
            "reference_rejected_logps",
            "chosen_reward",
            "rejected_reward",
            "reward_margin",
        ):
            finite_number(row.get(key), f"heldout pair result {key}")
    aggregate = dev_eval.preference_eval.aggregate_results(
        pair_results, split="heldout", checkpoint_step=CHECKPOINT_STEP
    )
    gate = heldout_gate(aggregate)
    require(
        value.get("aggregate") == aggregate
        and value.get("eligibility") == gate
        and value.get("status") == ("pass" if gate["passed"] else "gate_fail"),
        "heldout aggregate/gate no longer recomputes",
    )
    boundary = mapping(value.get("claim_boundary"), "heldout claim boundary")
    require(
        boundary.get("fresh_process_checkpoint_reload_proven_before_claim") is True
        and boundary.get("global_heldout_claims") == 1
        and boundary.get("candidate_heldout_evaluations") == 1
        and boundary.get("retry_allowed") is False
        and boundary.get("fallback_allowed") is False
        and boundary.get("heldout_consumed") is True
        and boundary.get("generation_performed") is False
        and boundary.get("sandbox_execution_performed") is False,
        "heldout one-shot claim boundary drifted",
    )
    return {
        "value": value,
        "path": path,
        "file_sha256": gpu_stage.file_sha256(path),
        "evaluation_sha256": evaluation_sha,
        "claim": claim,
        "gate": gate,
    }


def _validate_heldout_failure(
    static: Mapping[str, Any], failure_path: Path
) -> dict[str, Any]:
    path = failure_path.expanduser().resolve(strict=True)
    require(path == static["failure_path"], "heldout failure path is not canonical")
    require(
        not static["heldout_output"].exists() and not static["heldout_output"].is_symlink(),
        "heldout failure coexists with an evaluation receipt",
    )
    value = load(path, "heldout after-claim failure")
    failure_sha = verify_self(value, "failure_sha256", "heldout after-claim failure")
    require(
        value.get("schema_name") == HELDOUT_FAILURE_SCHEMA
        and value.get("schema_version") == 1
        and value.get("status") == "failed_closed_no_retry"
        and value.get("goal_id") == GOAL_ID
        and value.get("version_id") == VERSION_ID
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("run_id") == RUN_ID
        and value.get("authorization_sha256") == static["authorization_sha256"]
        and isinstance(value.get("heldout_rows_opened"), bool)
        and value.get("retry_allowed") is False
        and value.get("fallback_allowed") is False
        and value.get("heldout_consumed") is True,
        "heldout after-claim failure contract drifted",
    )
    claim = load(static["claim_path"], "global heldout access claim")
    claim_sha = verify_self(claim, "claim_sha256", "global heldout access claim")
    require(
        value.get("heldout_claim_sha256") == claim_sha
        and claim.get("schema_name") == HELDOUT_CLAIM_SCHEMA
        and claim.get("authorization_sha256") == static["authorization_sha256"]
        and claim.get("max_global_claims") == 1
        and claim.get("max_candidate_evaluations") == 1
        and claim.get("retry_allowed") is False
        and claim.get("fallback_allowed") is False
        and claim.get("producer") == static["authorization"].get("heldout_evaluator"),
        "failed heldout attempt claim drifted",
    )
    return {
        "value": value,
        "path": path,
        "file_sha256": gpu_stage.file_sha256(path),
        "failure_sha256": failure_sha,
        "claim": claim,
    }


def _closeout_document(
    static: Mapping[str, Any], heldout: Mapping[str, Any], created_at: str
) -> tuple[Path, dict[str, Any], str]:
    authorization = static["authorization"]
    goal_root = BOOTCAMP_ROOT / GOAL4_REL
    event5_entry = mapping(authorization.get("authorization_event"), "authorization event")
    event5_path_raw = Path(text(event5_entry.get("path"), "authorization event path"))
    event5_path = (
        event5_path_raw.resolve(strict=True)
        if event5_path_raw.is_absolute()
        else (BOOTCAMP_ROOT / event5_path_raw).resolve(strict=True)
    )
    event5 = load(event5_path, "heldout authorization event")
    event5_ref = event_identity(event5_path, event5, goal_root)
    authorization_ref = file_identity(
        static["authorization_path"], self_field="authorization_sha256"
    )
    evaluation_ref = file_identity(heldout["path"], self_field="evaluation_sha256")
    claim_ref = file_identity(static["claim_path"], self_field="claim_sha256")
    passed = heldout["gate"]["passed"] is True
    event = _event(
        event_id=EVENT6_NAME,
        sequence=6,
        created_at=created_at,
        event_type="heldout-terminal",
        state_before="guardrails_passed_heldout_authorized",
        state_after="qualified" if passed else "terminal_no_go",
        prior_event=event5_ref,
        authority_refs=[authorization_ref, evaluation_ref, claim_ref],
        payload={
            "status": "qualified" if passed else "terminal_no_go",
            "heldout_evaluation": evaluation_ref,
            "heldout_gate": heldout["gate"],
            "global_heldout_claims": 1,
            "candidate_heldout_evaluations": 1,
            "retry_allowed": False,
            "fallback_allowed": False,
            "qualification_scope": "fixed_experimental_suite_only",
            "deployment_ready_claim": False,
            "general_improvement_claim": False,
        },
    )
    return goal_root / f"events/{EVENT6_NAME}.json", event, "event_sha256"


def _failure_closeout_document(
    static: Mapping[str, Any], failure: Mapping[str, Any], created_at: str
) -> tuple[Path, dict[str, Any], str]:
    authorization = static["authorization"]
    goal_root = BOOTCAMP_ROOT / GOAL4_REL
    event5_entry = mapping(authorization.get("authorization_event"), "authorization event")
    event5_path_raw = Path(text(event5_entry.get("path"), "authorization event path"))
    event5_path = (
        event5_path_raw.resolve(strict=True)
        if event5_path_raw.is_absolute()
        else (BOOTCAMP_ROOT / event5_path_raw).resolve(strict=True)
    )
    event5 = load(event5_path, "heldout authorization event")
    failure_ref = file_identity(failure["path"], self_field="failure_sha256")
    event = _event(
        event_id=EVENT6_NAME,
        sequence=6,
        created_at=created_at,
        event_type="heldout-terminal",
        state_before="guardrails_passed_heldout_authorized",
        state_after="terminal_no_go",
        prior_event=event_identity(event5_path, event5, goal_root),
        authority_refs=[
            file_identity(static["authorization_path"], self_field="authorization_sha256"),
            failure_ref,
            file_identity(static["claim_path"], self_field="claim_sha256"),
        ],
        payload={
            "status": "terminal_no_go",
            "reason": "heldout_failed_after_global_claim",
            "heldout_failure": failure_ref,
            "failure_phase": failure["value"].get("failure_phase"),
            "heldout_rows_opened": failure["value"].get("heldout_rows_opened"),
            "global_heldout_claims": 1,
            "heldout_attempts": 1,
            "scientific_evaluations_completed": 0,
            "retry_allowed": False,
            "fallback_allowed": False,
            "qualification_scope": "fixed_experimental_suite_only",
            "deployment_ready_claim": False,
            "general_improvement_claim": False,
        },
    )
    return goal_root / f"events/{EVENT6_NAME}.json", event, "event_sha256"


def execute_closeout(args: argparse.Namespace) -> dict[str, Any]:
    created_at = validate_created_at(args.created_at_utc)
    static = _static_authorization(args.authorization)
    require(
        (args.heldout_evaluation is None) != (args.heldout_failure is None),
        "exactly one of --heldout-evaluation or --heldout-failure is required",
    )
    heldout = (
        _validate_heldout_evaluation(static, args.heldout_evaluation)
        if args.heldout_evaluation is not None
        else _validate_heldout_failure(static, args.heldout_failure)
    )
    completed_evaluation = args.heldout_evaluation is not None
    document = (
        _closeout_document(static, heldout, created_at)
        if completed_evaluation
        else _failure_closeout_document(static, heldout, created_at)
    )
    if args.mode == "preflight":
        _documents_match([document], require_present=False)
        status = "preflight_pass"
    elif args.mode == "check":
        _documents_match([document], require_present=True)
        status = heldout["value"]["status"]
    else:
        write_exclusive_or_verify(document[0], document[1])
        _documents_match([document], require_present=True)
        status = heldout["value"]["status"]
    qualified = completed_evaluation and heldout["gate"]["passed"] is True
    return {
        "status": status,
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "terminal_state": "qualified" if qualified else "terminal_no_go",
        "terminal_event": str(document[0]),
        "event_sha256": document[1]["event_sha256"],
        "global_heldout_claims": 1,
        "candidate_heldout_evaluations": 1 if completed_evaluation else 0,
        "retry_allowed": False,
    }


def _self_test() -> None:
    require(
        math.fsum(math.comb(17, index) for index in range(13, 18)) / 2**17
        == DEV_SIGN_TEST_P,
        "dev sign-test p-value drifted",
    )
    require(
        math.fsum(math.comb(29, index) for index in range(21, 30)) / 2**29
        == HELDOUT_21_SIGN_TEST_P
        and math.fsum(math.comb(29, index) for index in range(20, 30)) / 2**29
        == HELDOUT_20_SIGN_TEST_P
        and HELDOUT_21_SIGN_TEST_P <= 0.025 < HELDOUT_20_SIGN_TEST_P,
        "heldout first-passing sign-test boundary drifted",
    )
    aggregate = {
        "overall": {
            "pairs": HELDOUT_RECORDS,
            "wins": HELDOUT_MINIMUM_POSITIVE_PAIRS,
            "mean_reward_margin": 0.001,
        },
        "length_matched": {"mean_reward_margin": 0.0001},
    }
    require(heldout_gate(aggregate)["passed"], "heldout threshold boundary failed")
    aggregate["overall"]["wins"] = HELDOUT_MINIMUM_POSITIVE_PAIRS - 1
    require(not heldout_gate(aggregate)["passed"], "heldout 20/29 was accepted")
    full112_metrics = {
        "records": 112,
        "records_by_skill": {"code": 28, "finance": 28, "general": 28, "math": 28},
        "general_correct": 20,
        "math_correct": 17,
        "finance_correct": 14,
        "non_code_correct": 51,
        "format_compliant": 90,
        "code_sandbox_execution_eligible": 26,
        "infrastructure_failures": 0,
    }
    e2b_metrics = {
        "total_correct": 65,
        "general_correct": 20,
        "math_correct": 17,
        "finance_correct": 14,
        "code_correct": 14,
        "format_compliant": 90,
        "code_sandbox_execution_eligible": 26,
        "infrastructure_failures": 0,
        "code_records": 28,
        "failed": 12,
    }
    require(
        recompute_combined_guardrails(full112_metrics, e2b_metrics)["passed"],
        "combined coding threshold boundary failed",
    )
    e2b_metrics["code_correct"] = 13
    e2b_metrics["total_correct"] = 64
    e2b_metrics["failed"] = 13
    require(
        not recompute_combined_guardrails(full112_metrics, e2b_metrics)["passed"],
        "combined coding rejection boundary failed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("closeout-dev-fail", "authorize-heldout", "closeout")
    )
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="preflight")
    parser.add_argument("--created-at-utc")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--dev-evaluation", type=Path)
    parser.add_argument("--full112-receipt", type=Path)
    parser.add_argument("--e2b-receipt", type=Path)
    parser.add_argument("--heldout-evaluator", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--heldout-evaluation", type=Path)
    parser.add_argument("--heldout-failure", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _require_arguments(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {
                "status": "pass",
                "scope": "stdlib_self_test",
                "heldout_minimum_positive_pairs": HELDOUT_MINIMUM_POSITIVE_PAIRS,
            }
        else:
            _require_arguments(args, ("phase", "created_at_utc"))
            if args.phase == "closeout-dev-fail":
                _require_arguments(
                    args, ("campaign", "training_receipt", "dev_evaluation")
                )
                result = execute_dev_failure(args)
            elif args.phase == "authorize-heldout":
                _require_arguments(
                    args,
                    (
                        "campaign",
                        "training_receipt",
                        "dev_evaluation",
                        "full112_receipt",
                        "e2b_receipt",
                        "heldout_evaluator",
                    ),
                )
                result = execute_authorize(args)
            else:
                _require_arguments(args, ("authorization",))
                result = execute_closeout(args)
    except BaseException as error:
        print(
            json.dumps(
                {"status": "fail_closed", "error": str(error)},
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
