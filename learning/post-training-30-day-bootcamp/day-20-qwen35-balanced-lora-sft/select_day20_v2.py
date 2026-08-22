#!/usr/bin/env python3
"""Select a Day 20 v2 probe only from complete normalized-plus-E2B evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import day20_candidate_factory_v2 as factory
import day20_eval_identity_v2 as eval_identity
import score_day20_code_e2b_v3 as e2b_scorer


SCHEMA_VERSION = 2
SELECTION_DOMAIN = "day20.v2.probe_selection"


class Day20SelectionV2Error(ValueError):
    """A cohort artifact, common comparison identity, or decision drifted."""


E2BVerifier = Callable[..., tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]]


def expected_probe_cohort(*, bracket_completed: bool) -> list[str]:
    learning_rates = (factory.STAGE_A_LR,) + (
        factory.BRACKET_LRS if bracket_completed else ()
    )
    return [
        factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate=learning_rate,
            checkpoint=checkpoint,
        )
        for learning_rate in learning_rates
        for checkpoint in factory.PROBE_CHECKPOINT_TOKENS
    ]


def _artifact_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "file_sha256": e2b_scorer.file_sha256(resolved),
    }


def _binary_score(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        raise Day20SelectionV2Error(f"{label} is not a binary score")
    return int(float(value))


def recompute_probe_metrics(
    normalized_rows: Sequence[Mapping[str, Any]],
    e2b_rows: Sequence[Mapping[str, Any]],
    e2b_summary: Mapping[str, Any],
) -> dict[str, int]:
    if len(normalized_rows) != 32 or Counter(
        row.get("slice") for row in normalized_rows
    ) != Counter({skill: 8 for skill in factory.SKILLS}):
        raise Day20SelectionV2Error("probe slice distribution drifted")
    code_results = {row.get("sample_id"): row for row in e2b_rows}
    if len(code_results) != len(e2b_rows):
        raise Day20SelectionV2Error("duplicate E2B result sample ID")
    counts = {skill: 0 for skill in factory.SKILLS}
    eligible = 0
    expected_eligible_ids: list[str] = []
    for row in normalized_rows:
        skill = row.get("slice")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise Day20SelectionV2Error("normalized sample ID is invalid")
        if skill == "code":
            is_eligible = row.get("sandbox_execution_eligible")
            if not isinstance(is_eligible, bool):
                raise Day20SelectionV2Error("Code eligibility evidence is invalid")
            if is_eligible:
                eligible += 1
                expected_eligible_ids.append(sample_id)
                result = code_results.get(sample_id)
                if not isinstance(result, Mapping):
                    raise Day20SelectionV2Error(
                        f"eligible Code result is missing: {sample_id}"
                    )
                counts["code"] += _binary_score(
                    result.get("score"), f"Code score {sample_id}"
                )
            elif sample_id in code_results:
                raise Day20SelectionV2Error(
                    f"ineligible Code row was executed: {sample_id}"
                )
        else:
            score = row.get("normalized_scorer_result")
            if not isinstance(score, Mapping):
                raise Day20SelectionV2Error("non-Code scorer evidence is missing")
            counts[str(skill)] += _binary_score(
                score.get("score"), f"{skill} score {sample_id}"
            )
    if [row.get("sample_id") for row in e2b_rows] != expected_eligible_ids:
        raise Day20SelectionV2Error("eligible E2B coverage/order drifted")
    infrastructure = e2b_summary.get("infrastructure_failures")
    if isinstance(infrastructure, bool) or not isinstance(infrastructure, int):
        raise Day20SelectionV2Error("E2B infrastructure count is invalid")
    metrics = {
        **counts,
        "total": sum(counts.values()),
        "code_sandbox_execution_eligible": eligible,
        "infrastructure_failures": infrastructure,
    }
    try:
        return factory.validate_probe_metrics(metrics)
    except factory.CandidateFactoryV2Error as error:
        raise Day20SelectionV2Error(str(error)) from error


def _verified_artifact(
    summary_path: Path,
    *,
    expected_candidate: str,
    e2b_verifier: E2BVerifier,
    pair_verifier: e2b_scorer.PairVerifier,
) -> dict[str, Any]:
    try:
        e2b_rows, e2b_summary, normalized_rows, normalized_summary = e2b_verifier(
            summary_path.resolve(),
            pair_verifier=pair_verifier,
            expected_candidate=expected_candidate,
            expected_scope="probe32",
        )
    except ValueError as error:
        raise Day20SelectionV2Error(str(error)) from error
    metrics = recompute_probe_metrics(normalized_rows, e2b_rows, e2b_summary)
    normalized_summary_path = Path(
        str(e2b_summary["normalized_summary"]["path"])
    ).resolve()
    predictions_path = Path(
        str(e2b_summary["normalized_predictions"]["path"])
    ).resolve()
    result_path = Path(str(e2b_summary["result"]["path"])).resolve()
    return {
        "candidate": expected_candidate,
        "metrics": metrics,
        "normalized_order": [row["sample_id"] for row in normalized_rows],
        "comparison_context": normalized_summary["comparison_context"],
        "normalized_comparison_key": normalized_summary[
            "normalized_comparison_key"
        ],
        "e2b_comparison_context": e2b_summary["e2b_comparison_context"],
        "e2b_comparison_key": e2b_summary["e2b_comparison_key"],
        "complete_comparison_key": e2b_summary["complete_comparison_key"],
        "artifacts": {
            "normalized_predictions": _artifact_identity(predictions_path),
            "normalized_summary": _artifact_identity(normalized_summary_path),
            "e2b_results": _artifact_identity(result_path),
            "e2b_summary": _artifact_identity(summary_path),
        },
    }


def _common_comparison(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not artifacts:
        raise Day20SelectionV2Error("selection cohort is empty")
    first = artifacts[0]
    fields = (
        "normalized_order",
        "comparison_context",
        "normalized_comparison_key",
        "e2b_comparison_context",
        "e2b_comparison_key",
        "complete_comparison_key",
    )
    for artifact in artifacts[1:]:
        for field in fields:
            if artifact.get(field) != first.get(field):
                raise Day20SelectionV2Error(
                    f"cohort common comparison drifted: {field}"
                )
    try:
        normalized_context = eval_identity.normalize_comparison_context(
            first["comparison_context"]
        )
        normalized_key = eval_identity.normalized_comparison_key(
            normalized_context, expected=first["normalized_comparison_key"]
        )
        e2b_context = eval_identity.normalize_e2b_comparison_context(
            first["e2b_comparison_context"]
        )
        e2b_key = eval_identity.e2b_comparison_key(
            e2b_context, expected=first["e2b_comparison_key"]
        )
        complete = eval_identity.complete_comparison_key(
            e2b_context, expected=first["complete_comparison_key"]
        )
    except ValueError as error:
        raise Day20SelectionV2Error(str(error)) from error
    if e2b_context["normalized_comparison_key"] != normalized_key:
        raise Day20SelectionV2Error("normalized and E2B contexts are not joined")
    return {
        "scope": "probe32",
        "normalized_comparison_key": normalized_key,
        "e2b_comparison_key": e2b_key,
        "complete_comparison_key": complete,
        "ordered_sample_ids": first["normalized_order"],
        "ordered_sample_ids_sha256": eval_identity.object_sha256(
            first["normalized_order"]
        ),
        "code_sample_ids": e2b_context["code_sample_ids"],
        "code_sample_order_sha256": e2b_context["code_sample_order_sha256"],
    }


def select_probe_cohort(
    *,
    base_e2b_summary: Path,
    candidate_e2b_summaries: Sequence[Path],
    bracket_completed: bool,
    pair_verifier: e2b_scorer.PairVerifier = e2b_scorer._default_pair_verifier,
    e2b_verifier: E2BVerifier = e2b_scorer.verify_published_pair,
) -> dict[str, Any]:
    if not isinstance(bracket_completed, bool):
        raise Day20SelectionV2Error("bracket_completed must be boolean")
    base = _verified_artifact(
        base_e2b_summary,
        expected_candidate=factory.base_candidate_id("probe32"),
        e2b_verifier=e2b_verifier,
        pair_verifier=pair_verifier,
    )
    expected_candidates = expected_probe_cohort(
        bracket_completed=bracket_completed
    )
    by_candidate: dict[str, dict[str, Any]] = {}
    for path in candidate_e2b_summaries:
        summary = e2b_scorer.load_json(path.resolve())
        candidate = summary.get("candidate")
        if not isinstance(candidate, str) or candidate in by_candidate:
            raise Day20SelectionV2Error("candidate summary identity is invalid or duplicate")
        by_candidate[candidate] = _verified_artifact(
            path,
            expected_candidate=candidate,
            e2b_verifier=e2b_verifier,
            pair_verifier=pair_verifier,
        )
    if set(by_candidate) != set(expected_candidates):
        missing = sorted(set(expected_candidates) - set(by_candidate))
        extra = sorted(set(by_candidate) - set(expected_candidates))
        raise Day20SelectionV2Error(
            f"probe cohort is incomplete; missing={missing}, extra={extra}"
        )
    ordered = [by_candidate[candidate] for candidate in expected_candidates]
    common = _common_comparison([base, *ordered])
    factory_candidates: list[dict[str, Any]] = []
    for artifact in ordered:
        try:
            parsed = factory.parse_candidate_id(artifact["candidate"])
        except factory.CandidateFactoryV2Error as error:
            raise Day20SelectionV2Error(str(error)) from error
        factory_candidates.append(
            {
                "candidate": artifact["candidate"],
                "learning_rate": parsed.learning_rate,
                "checkpoint": parsed.checkpoint_label,
                "metrics": artifact["metrics"],
                "artifacts": artifact["artifacts"],
            }
        )
    try:
        decision = factory.select_probe_action(
            base_metrics=base["metrics"],
            candidates=factory_candidates,
            bracket_completed=bracket_completed,
        )
    except factory.CandidateFactoryV2Error as error:
        raise Day20SelectionV2Error(str(error)) from error
    selected = decision["selected"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": SELECTION_DOMAIN,
        "status": decision["action"],
        "bracket_completed": bracket_completed,
        "expected_candidate_ids": expected_candidates,
        "common_comparison": common,
        "base": {
            "candidate": base["candidate"],
            "metrics": base["metrics"],
            "artifacts": base["artifacts"],
        },
        "candidates": decision["candidates"],
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_learning_rate": (
            selected["learning_rate"] if selected else None
        ),
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "tie_break": [
            "total_desc",
            "code_desc",
            "math_desc",
            "general_desc",
            "learning_rate_asc",
            "checkpoint_tokens_asc",
        ],
    }
    result["selection_sha256"] = eval_identity.object_sha256(result)
    return result


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Day20SelectionV2Error(f"refusing to overwrite selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as error:
        raise Day20SelectionV2Error("selection appeared during write") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _embedded_e2b_summary_path(value: Any, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise Day20SelectionV2Error(f"{label} selection entry is invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Day20SelectionV2Error(f"{label} artifact identities are missing")
    summary = artifacts.get("e2b_summary")
    if not isinstance(summary, Mapping):
        raise Day20SelectionV2Error(f"{label} E2B summary identity is missing")
    path = summary.get("path")
    if not isinstance(path, str) or not path:
        raise Day20SelectionV2Error(f"{label} E2B summary path is invalid")
    return Path(path)


def verify_selection(
    selection_path: Path,
    *,
    pair_verifier: e2b_scorer.PairVerifier = e2b_scorer._default_pair_verifier,
    e2b_verifier: E2BVerifier = e2b_scorer.verify_published_pair,
) -> dict[str, Any]:
    selection = e2b_scorer.load_json(selection_path.resolve())
    expected_hash = eval_identity.object_sha256(
        {key: value for key, value in selection.items() if key != "selection_sha256"}
    )
    if (
        selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("domain") != SELECTION_DOMAIN
        or selection.get("selection_sha256") != expected_hash
    ):
        raise Day20SelectionV2Error("selection identity drifted")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise Day20SelectionV2Error("selection candidate list is invalid")
    base_path = _embedded_e2b_summary_path(selection.get("base"), "base")
    candidate_paths = [
        _embedded_e2b_summary_path(candidate, f"candidate[{index}]")
        for index, candidate in enumerate(candidates)
    ]
    recomputed = select_probe_cohort(
        base_e2b_summary=base_path,
        candidate_e2b_summaries=candidate_paths,
        bracket_completed=selection.get("bracket_completed"),
        pair_verifier=pair_verifier,
        e2b_verifier=e2b_verifier,
    )
    if recomputed != selection:
        raise Day20SelectionV2Error("selection differs from recomputed evidence")
    return selection


def write_or_verify_selection(
    output: Path,
    *,
    base_e2b_summary: Path,
    candidate_e2b_summaries: Sequence[Path],
    bracket_completed: bool,
    pair_verifier: e2b_scorer.PairVerifier = e2b_scorer._default_pair_verifier,
    e2b_verifier: E2BVerifier = e2b_scorer.verify_published_pair,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        return verify_selection(
            output, pair_verifier=pair_verifier, e2b_verifier=e2b_verifier
        )
    result = select_probe_cohort(
        base_e2b_summary=base_e2b_summary,
        candidate_e2b_summaries=candidate_e2b_summaries,
        bracket_completed=bracket_completed,
        pair_verifier=pair_verifier,
        e2b_verifier=e2b_verifier,
    )
    _atomic_json_new(output, result)
    return verify_selection(
        output, pair_verifier=pair_verifier, e2b_verifier=e2b_verifier
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-e2b-summary", required=True, type=Path)
    parser.add_argument(
        "--candidate-e2b-summary", action="append", default=[], type=Path
    )
    parser.add_argument("--bracket-completed", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = write_or_verify_selection(
            args.output,
            base_e2b_summary=args.base_e2b_summary,
            candidate_e2b_summaries=args.candidate_e2b_summary,
            bracket_completed=args.bracket_completed,
        )
    except Day20SelectionV2Error as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
