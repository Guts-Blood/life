#!/usr/bin/env python3
"""Deterministic token-aware temporal ordering for the Day 20 v2 datasets.

The scheduler is deliberately independent from the v1 preparation code.  It
paces records from every skill across the complete optimizer-step horizon, then
chooses the skill with the largest cumulative deficit from the 25% token
target.  The public ordering entrypoint fails closed unless the resulting
sequence satisfies every temporal-mix gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence


SKILLS = ("general", "math", "finance", "code")
ORDERING_VERSION = "token_aware_temporal_interleave_v2"
ROLLING_WINDOW_STEPS = 8
MIN_TOKEN_SHARE_PERCENT = 20
MAX_TOKEN_SHARE_PERCENT = 30
MIN_LAST_OBSERVED_PERCENT = 90


class Day20OrderingV2Error(ValueError):
    """The v2 temporal ordering contract could not be satisfied."""


def _object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_records(
    records: Sequence[Mapping[str, Any]], global_batch_size: int
) -> list[dict[str, Any]]:
    if (
        isinstance(global_batch_size, bool)
        or not isinstance(global_batch_size, int)
        or global_batch_size <= 0
    ):
        raise Day20OrderingV2Error("global_batch_size must be a positive integer")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise Day20OrderingV2Error("records must be a sequence of mappings")
    if not records:
        raise Day20OrderingV2Error("records must not be empty")

    normalized: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    seen_skills: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Day20OrderingV2Error(f"record {index} must be a mapping")
        skill = record.get("skill")
        if skill not in SKILLS:
            raise Day20OrderingV2Error(
                f"record {index} has unsupported skill {skill!r}"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise Day20OrderingV2Error(
                f"record {index} sample_id must be non-empty text"
            )
        if sample_id in sample_ids:
            raise Day20OrderingV2Error(f"duplicate sample_id: {sample_id}")
        tokens = record.get("qwen35_supervised_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise Day20OrderingV2Error(
                f"record {sample_id} qwen35_supervised_tokens must be a positive integer"
            )
        sample_ids.add(sample_id)
        seen_skills.add(skill)
        normalized.append(dict(record))

    missing = sorted(set(SKILLS) - seen_skills)
    if missing:
        raise Day20OrderingV2Error(f"records are missing required skills: {missing}")
    return normalized


def _stable_skill_order(
    records: Sequence[Mapping[str, Any]], skill: str
) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records if record["skill"] == skill]
    selected.sort(
        key=lambda record: (
            hashlib.sha256(
                (
                    f"{ORDERING_VERSION}\0{skill}\0{record['sample_id']}"
                ).encode("utf-8")
            ).hexdigest(),
            record["sample_id"],
        )
    )
    return selected


def optimizer_step_skill_matrix(
    records: Sequence[Mapping[str, Any]], *, global_batch_size: int = 8
) -> list[dict[str, Any]]:
    """Return the deterministic per-optimizer-step skill/token matrix."""
    rows = _validate_records(records, global_batch_size)
    matrix: list[dict[str, Any]] = []
    for start in range(0, len(rows), global_batch_size):
        window = rows[start : start + global_batch_size]
        tokens = {
            skill: sum(
                int(record["qwen35_supervised_tokens"])
                for record in window
                if record["skill"] == skill
            )
            for skill in SKILLS
        }
        matrix.append(
            {
                "optimizer_step": len(matrix) + 1,
                "records": len(window),
                "supervised_tokens": sum(tokens.values()),
                "supervised_tokens_by_skill": tokens,
            }
        )
    return matrix


def _released_record_cap(
    skill_records: int,
    step: int,
    total_records: int,
    global_batch_size: int,
) -> int:
    """Release by consumed-record progress, including a partial final batch.

    Using ``step / ceil(records / batch)`` under-releases records whenever the
    final optimizer step is partial.  The cumulative slot count is the exact
    progress denominator that keeps the paced schedule feasible.
    """
    consumed_slots = min(step * global_batch_size, total_records)
    return (
        skill_records * consumed_slots + total_records - 1
    ) // total_records


def _schedule_records(
    records: Sequence[Mapping[str, Any]], global_batch_size: int
) -> list[dict[str, Any]]:
    rows = _validate_records(records, global_batch_size)
    queues = {skill: _stable_skill_order(rows, skill) for skill in SKILLS}
    skill_record_counts = {skill: len(queues[skill]) for skill in SKILLS}
    consumed = {skill: 0 for skill in SKILLS}
    scheduled_tokens = {skill: 0 for skill in SKILLS}
    scheduled_total_tokens = 0
    ordered: list[dict[str, Any]] = []
    total_steps = (len(rows) + global_batch_size - 1) // global_batch_size

    for step in range(1, total_steps + 1):
        step_size = min(global_batch_size, len(rows) - len(ordered))
        for _ in range(step_size):
            eligible = [
                skill
                for skill in SKILLS
                if consumed[skill]
                < _released_record_cap(
                    skill_record_counts[skill], step, len(rows), global_batch_size
                )
            ]
            if not eligible:
                raise Day20OrderingV2Error(
                    "paced release cannot fill optimizer step "
                    f"{step}; dataset record counts are incompatible with "
                    f"global_batch_size={global_batch_size}"
                )

            # Four times the absolute deficit from the shared 25% token target.
            # SKILLS order is the deterministic tie breaker.
            skill = max(
                eligible,
                key=lambda candidate: (
                    scheduled_total_tokens - 4 * scheduled_tokens[candidate]
                ),
            )
            record = queues[skill][consumed[skill]]
            ordered.append(record)
            consumed[skill] += 1
            tokens = int(record["qwen35_supervised_tokens"])
            scheduled_tokens[skill] += tokens
            scheduled_total_tokens += tokens
    return ordered


def audit_temporal_mix(
    records: Sequence[Mapping[str, Any]], *, global_batch_size: int = 8
) -> dict[str, Any]:
    """Audit an existing sequence against all Day 20 v2 temporal gates.

    Invalid row shapes raise immediately.  A structurally valid sequence that
    misses a temporal gate returns ``status == "fail"`` with deterministic
    violation details so callers can inspect rejected preparations.
    """
    rows = _validate_records(records, global_batch_size)
    matrix = optimizer_step_skill_matrix(rows, global_batch_size=global_batch_size)
    total_steps = len(matrix)
    skill_record_counts = Counter(record["skill"] for record in rows)
    cumulative_records = {skill: 0 for skill in SKILLS}
    pace_failures: list[dict[str, Any]] = []
    for step in matrix:
        start = (step["optimizer_step"] - 1) * global_batch_size
        for record in rows[start : start + step["records"]]:
            cumulative_records[record["skill"]] += 1
        for skill in SKILLS:
            cap = _released_record_cap(
                skill_record_counts[skill],
                step["optimizer_step"],
                len(rows),
                global_batch_size,
            )
            if cumulative_records[skill] > cap:
                pace_failures.append(
                    {
                        "optimizer_step": step["optimizer_step"],
                        "skill": skill,
                        "observed_records": cumulative_records[skill],
                        "released_record_cap": cap,
                    }
                )

    rolling_failures: list[dict[str, Any]] = []
    rolling_window_count = max(0, total_steps - ROLLING_WINDOW_STEPS + 1)
    if not rolling_window_count:
        rolling_failures.append(
            {
                "reason": "insufficient_optimizer_steps",
                "optimizer_steps": total_steps,
                "required_steps": ROLLING_WINDOW_STEPS,
            }
        )
    else:
        for start in range(rolling_window_count):
            window = matrix[start : start + ROLLING_WINDOW_STEPS]
            missing = [
                skill
                for skill in SKILLS
                if sum(
                    step["supervised_tokens_by_skill"][skill] for step in window
                )
                == 0
            ]
            if missing:
                rolling_failures.append(
                    {
                        "start_step": start + 1,
                        "end_step": start + ROLLING_WINDOW_STEPS,
                        "missing_skills": missing,
                    }
                )

    checkpoint_audits: list[dict[str, Any]] = []
    for numerator, label in ((1, "25%"), (2, "50%"), (3, "75%"), (4, "100%")):
        checkpoint_step = (total_steps * numerator + 3) // 4
        cumulative_tokens = {
            skill: sum(
                step["supervised_tokens_by_skill"][skill]
                for step in matrix[:checkpoint_step]
            )
            for skill in SKILLS
        }
        total_tokens = sum(cumulative_tokens.values())
        shares = {
            skill: round(cumulative_tokens[skill] / total_tokens, 6)
            for skill in SKILLS
        }
        failing_skills = [
            skill
            for skill in SKILLS
            if cumulative_tokens[skill] * 100
            < total_tokens * MIN_TOKEN_SHARE_PERCENT
            or cumulative_tokens[skill] * 100
            > total_tokens * MAX_TOKEN_SHARE_PERCENT
        ]
        checkpoint_audits.append(
            {
                "fraction": label,
                "optimizer_step": checkpoint_step,
                "cumulative_supervised_tokens": total_tokens,
                "cumulative_supervised_tokens_by_skill": cumulative_tokens,
                "token_share_by_skill": shares,
                "passing": not failing_skills,
                "failing_skills": failing_skills,
            }
        )

    minimum_last_step = (total_steps * MIN_LAST_OBSERVED_PERCENT + 99) // 100
    last_observed_steps = {
        skill: max(
            (
                step["optimizer_step"]
                for step in matrix
                if step["supervised_tokens_by_skill"][skill] > 0
            ),
            default=0,
        )
        for skill in SKILLS
    }
    late_skill_failures = [
        skill for skill in SKILLS if last_observed_steps[skill] < minimum_last_step
    ]

    violations: list[str] = []
    if pace_failures:
        violations.append("paced release cap exceeded")
    if rolling_failures:
        violations.append("rolling 8-step skill coverage failed")
    if any(not checkpoint["passing"] for checkpoint in checkpoint_audits):
        violations.append("cumulative skill token share left the 20%-30% band")
    if late_skill_failures:
        violations.append("one or more skills ended before 90% of optimizer steps")

    return {
        "domain": "day20.qwen35_temporal_mix_audit",
        "version": ORDERING_VERSION,
        "status": "pass" if not violations else "fail",
        "config": {
            "global_batch_size": global_batch_size,
            "rolling_window_steps": ROLLING_WINDOW_STEPS,
            "target_token_share": 0.25,
            "minimum_token_share": MIN_TOKEN_SHARE_PERCENT / 100,
            "maximum_token_share": MAX_TOKEN_SHARE_PERCENT / 100,
            "minimum_last_observed_fraction": MIN_LAST_OBSERVED_PERCENT / 100,
        },
        "records": len(rows),
        "optimizer_steps": total_steps,
        "ordered_sample_ids_sha256": _object_sha256(
            [record["sample_id"] for record in rows]
        ),
        "optimizer_step_skill_matrix": matrix,
        "optimizer_step_skill_matrix_sha256": _object_sha256(matrix),
        "gates": {
            "paced_release": {
                "passing": not pace_failures,
                "failures": pace_failures,
            },
            "rolling_skill_coverage": {
                "passing": not rolling_failures,
                "window_count": rolling_window_count,
                "failures": rolling_failures,
            },
            "cumulative_token_share": {
                "passing": all(
                    checkpoint["passing"] for checkpoint in checkpoint_audits
                ),
                "checkpoints": checkpoint_audits,
            },
            "last_observed_step": {
                "passing": not late_skill_failures,
                "minimum_optimizer_step": minimum_last_step,
                "last_observed_step_by_skill": last_observed_steps,
                "failing_skills": late_skill_failures,
            },
        },
        "violations": violations,
    }


def order_records(
    records: Sequence[Mapping[str, Any]], *, global_batch_size: int = 8
) -> list[dict[str, Any]]:
    """Produce a deterministic v2 order, raising unless every gate passes."""
    ordered = _schedule_records(records, global_batch_size)
    audit = audit_temporal_mix(ordered, global_batch_size=global_batch_size)
    if audit["status"] != "pass":
        raise Day20OrderingV2Error(
            "temporal mix gates failed: " + "; ".join(audit["violations"])
        )
    return ordered
