#!/usr/bin/env python3
"""Decision contract for the versioned Day 20 Qwen3.5 candidate funnel.

This module is deliberately stdlib-only.  It contains the immutable budgets,
candidate identities, probe gates, and main promotion floors that local
preflight and the remote runner must share.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "day20_qwen35_candidate_factory_v2"
PRIMARY_SEED = 20260809
CONFIRMATION_SEED = 20260810
SKILLS = ("general", "math", "code", "finance")
PROBE_LRS = ("1e-4", "3e-5", "6e-5", "8e-5")
STAGE_A_LR = "1e-4"
BRACKET_LRS = ("3e-5", "6e-5", "8e-5")
PROBE_CHECKPOINT_TOKENS = {
    "t6000": 6_000,
    "t12000": 12_000,
    "t18000": 18_000,
    "t24000": 24_000,
}
MAIN_CHECKPOINT_TOKENS = {
    "early": 80_000,
    "mid": 192_000,
    "final": 320_000,
}
MAIN_PROMOTION_THRESHOLDS = {
    "total_correct": 65,
    "general_correct": 5,
    "math_correct": 17,
    "finance_correct": 10,
    "code_correct": 14,
    "format_compliant": 90,
    "code_sandbox_execution_eligible": 26,
    "infrastructure_failures": 0,
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROBE_ID = re.compile(
    r"^probe-s(?P<seed>\d+)-lr(?P<lr>1e-4|3e-5|6e-5|8e-5)-"
    r"(?P<checkpoint>t6000|t12000|t18000|t24000)$"
)
_MAIN_ID = re.compile(
    r"^main-s(?P<seed>\d+)-lr(?P<lr>1e-4|3e-5|6e-5|8e-5)-"
    r"(?P<checkpoint>early|mid|final)$"
)


class CandidateFactoryV2Error(ValueError):
    """A candidate-funnel input violates the frozen v2 contract."""


@dataclass(frozen=True)
class CandidateIdentity:
    """The complete logical identity shared by train and evaluation code."""

    id: str
    scope: str
    model_role: str
    run_kind: str | None
    seed: int | None
    learning_rate: str | None
    checkpoint_label: str | None
    target_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_lr(value: str | float | Decimal) -> str:
    """Return one allowlisted canonical scientific-notation learning rate."""
    try:
        numeric = Decimal(str(value))
    except InvalidOperation as error:
        raise CandidateFactoryV2Error(f"invalid learning rate: {value!r}") from error
    for allowed in PROBE_LRS:
        if numeric == Decimal(allowed):
            return allowed
    raise CandidateFactoryV2Error(
        f"learning rate must be one of {', '.join(PROBE_LRS)}"
    )


def candidate_id(
    *, run_kind: str, seed: int, learning_rate: str | float, checkpoint: str
) -> str:
    """Build the sole accepted logical candidate identifier."""
    if isinstance(seed, bool) or seed not in {PRIMARY_SEED, CONFIRMATION_SEED}:
        raise CandidateFactoryV2Error("candidate seed is not allowlisted")
    lr = canonical_lr(learning_rate)
    if run_kind == "probe":
        if seed != PRIMARY_SEED:
            raise CandidateFactoryV2Error(
                "probe candidates are restricted to the primary seed"
            )
        if checkpoint not in PROBE_CHECKPOINT_TOKENS:
            raise CandidateFactoryV2Error("unknown probe checkpoint")
        value = f"probe-s{seed}-lr{lr}-{checkpoint}"
    elif run_kind == "main":
        if checkpoint not in MAIN_CHECKPOINT_TOKENS:
            raise CandidateFactoryV2Error("unknown main checkpoint")
        value = f"main-s{seed}-lr{lr}-{checkpoint}"
    else:
        raise CandidateFactoryV2Error("run_kind must be probe or main")
    if not _SAFE_NAME.fullmatch(value):
        raise CandidateFactoryV2Error("candidate identifier is unsafe")
    return value


def base_candidate_id(scope: str) -> str:
    """Return the only Base identity for an explicit evaluation scope."""
    if scope == "probe32":
        return "base-probe"
    if scope == "full112":
        return "base-full"
    raise CandidateFactoryV2Error("scope must be probe32 or full112")


def parse_candidate_id(value: str) -> CandidateIdentity:
    """Parse an allowlisted ID without inferring scope from row counts."""
    if value in {"base-probe", "base-full"}:
        return CandidateIdentity(
            id=value,
            scope="probe32" if value == "base-probe" else "full112",
            model_role="base",
            run_kind=None,
            seed=None,
            learning_rate=None,
            checkpoint_label=None,
            target_tokens=0,
        )
    match = _PROBE_ID.fullmatch(value)
    if match:
        seed = int(match.group("seed"))
        if seed != PRIMARY_SEED:
            raise CandidateFactoryV2Error(
                "probe candidates are restricted to the primary seed"
            )
        checkpoint = match.group("checkpoint")
        lr = canonical_lr(match.group("lr"))
        return CandidateIdentity(
            id=value,
            scope="probe32",
            model_role="lora",
            run_kind="probe",
            seed=seed,
            learning_rate=lr,
            checkpoint_label=checkpoint,
            target_tokens=PROBE_CHECKPOINT_TOKENS[checkpoint],
        )
    match = _MAIN_ID.fullmatch(value)
    if match:
        seed = int(match.group("seed"))
        if seed not in {PRIMARY_SEED, CONFIRMATION_SEED}:
            raise CandidateFactoryV2Error("main candidate seed is not allowlisted")
        checkpoint = match.group("checkpoint")
        lr = canonical_lr(match.group("lr"))
        return CandidateIdentity(
            id=value,
            scope="full112",
            model_role="lora",
            run_kind="main",
            seed=seed,
            learning_rate=lr,
            checkpoint_label=checkpoint,
            target_tokens=MAIN_CHECKPOINT_TOKENS[checkpoint],
        )
    raise CandidateFactoryV2Error("candidate identifier is not in the v2 grammar")


def _integer_metric(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateFactoryV2Error(f"metric {name!r} must be a non-negative int")
    return value


def validate_probe_metrics(metrics: Mapping[str, Any]) -> dict[str, int]:
    required = (
        "total",
        "general",
        "math",
        "finance",
        "code",
        "code_sandbox_execution_eligible",
        "infrastructure_failures",
    )
    normalized = {name: _integer_metric(metrics, name) for name in required}
    if normalized["total"] != sum(normalized[skill] for skill in SKILLS):
        raise CandidateFactoryV2Error("probe total differs from four-skill sum")
    if any(normalized[skill] > 8 for skill in SKILLS):
        raise CandidateFactoryV2Error("probe skill metric exceeds denominator 8")
    if normalized["code_sandbox_execution_eligible"] > 8:
        raise CandidateFactoryV2Error("probe Code eligibility exceeds denominator 8")
    return normalized


def classify_probe(
    *, base_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify one 32-case checkpoint as strict-pass, near-miss, or gross-fail."""
    base = validate_probe_metrics(base_metrics)
    candidate = validate_probe_metrics(candidate_metrics)
    strict_failures: list[str] = []
    if candidate["total"] < base["total"] + 2:
        strict_failures.append("total_below_base_plus_2")
    for skill in ("math", "finance", "code"):
        if candidate[skill] < base[skill] - 1:
            strict_failures.append(f"{skill}_regression_gt_1")
    if candidate["code_sandbox_execution_eligible"] < 7:
        strict_failures.append("code_sandbox_execution_eligible_below_7")
    if candidate["infrastructure_failures"] != 0:
        strict_failures.append("infrastructure_failures_nonzero")

    if not strict_failures:
        classification = "strict_pass"
    else:
        near_miss = (
            candidate["total"] >= base["total"]
            and all(
                candidate[skill] >= base[skill] - 2
                for skill in ("math", "finance", "code")
            )
            and candidate["code_sandbox_execution_eligible"] >= 6
            and candidate["infrastructure_failures"] == 0
        )
        classification = "near_miss" if near_miss else "gross_fail"
    return {
        "classification": classification,
        "strict_failures": strict_failures,
        "metrics": candidate,
    }


def probe_rank_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank already-eligible probe checkpoints without post-hoc discretion."""
    metrics = validate_probe_metrics(candidate.get("metrics", {}))
    lr = canonical_lr(str(candidate.get("learning_rate")))
    checkpoint = str(candidate.get("checkpoint"))
    if checkpoint not in PROBE_CHECKPOINT_TOKENS:
        raise CandidateFactoryV2Error("ranked probe has an invalid checkpoint")
    return (
        -metrics["total"],
        -metrics["code"],
        -metrics["math"],
        -metrics["general"],
        Decimal(lr),
        PROBE_CHECKPOINT_TOKENS[checkpoint],
    )


def select_probe_action(
    *,
    base_metrics: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    bracket_completed: bool,
) -> dict[str, Any]:
    """Return the next deterministic action in the staged probe funnel."""
    classified: list[dict[str, Any]] = []
    for raw in candidates:
        decision = classify_probe(
            base_metrics=base_metrics,
            candidate_metrics=raw.get("metrics", {}),
        )
        row = dict(raw)
        row.update(decision)
        classified.append(row)
    passing = [row for row in classified if row["classification"] == "strict_pass"]
    if passing:
        selected = sorted(passing, key=probe_rank_key)[0]
        action = "run_main"
    elif not bracket_completed and any(
        row["classification"] == "near_miss" for row in classified
    ):
        selected = None
        action = "run_bracket"
    else:
        selected = None
        action = "stop_no_passing_probe"
    return {
        "action": action,
        "selected": selected,
        "candidates": classified,
    }


def classify_main(metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        name: _integer_metric(metrics, name) for name in MAIN_PROMOTION_THRESHOLDS
    }
    failures = [
        name
        for name, threshold in MAIN_PROMOTION_THRESHOLDS.items()
        if (
            normalized[name] != threshold
            if name == "infrastructure_failures"
            else normalized[name] < threshold
        )
    ]
    if normalized["total_correct"] != sum(
        normalized[f"{skill}_correct"] for skill in SKILLS
    ):
        raise CandidateFactoryV2Error("main total differs from four-skill sum")
    return {
        "promotion_eligible": not failures,
        "failed_thresholds": failures,
        "metrics": normalized,
    }


__all__ = [
    "BRACKET_LRS",
    "CONFIRMATION_SEED",
    "CONTRACT_VERSION",
    "CandidateIdentity",
    "CandidateFactoryV2Error",
    "MAIN_CHECKPOINT_TOKENS",
    "MAIN_PROMOTION_THRESHOLDS",
    "PRIMARY_SEED",
    "PROBE_CHECKPOINT_TOKENS",
    "PROBE_LRS",
    "SKILLS",
    "STAGE_A_LR",
    "base_candidate_id",
    "candidate_id",
    "canonical_lr",
    "classify_main",
    "classify_probe",
    "probe_rank_key",
    "parse_candidate_id",
    "select_probe_action",
    "validate_probe_metrics",
]
