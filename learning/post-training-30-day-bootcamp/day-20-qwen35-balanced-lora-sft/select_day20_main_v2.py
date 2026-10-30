#!/usr/bin/env python3
"""Select and confirm Day 20 v2 main checkpoints without merging adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import day20_candidate_factory_v2 as factory
import day20_eval_identity_v2 as eval_identity
import score_day20_code_e2b_v3 as e2b


SCHEMA_VERSION = 2
SELECTOR_VERSION = "day20-main-promotion-selector-v2"
PRIMARY_DOMAIN = "day20.v2.primary_main_selection"
FINAL_DOMAIN = "day20.v2.final_promotion"
PRIMARY_FILENAME = "PRIMARY-MAIN-SELECTION.json"
FINAL_FILENAME = "FINAL-PROMOTION.json"
_SHA256_CHARS = frozenset("0123456789abcdef")


class Day20MainSelectionV2Error(ValueError):
    """A main evaluation, selection, or confirmation invariant failed."""


E2BVerifier = Callable[..., tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]]


def object_sha256(value: Any) -> str:
    return eval_identity.object_sha256(value)


def file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise Day20MainSelectionV2Error(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20MainSelectionV2Error(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day20MainSelectionV2Error(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise Day20MainSelectionV2Error(f"{label} must be a lowercase bare SHA-256")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = _require_sha256(value.get(field), field)
    actual = object_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if expected != actual:
        raise Day20MainSelectionV2Error(f"{field} drifted")
    return expected


def _score_bit(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Day20MainSelectionV2Error(f"{label} must be an exact 0/1 score")
    numeric = float(value)
    if numeric not in {0.0, 1.0}:
        raise Day20MainSelectionV2Error(f"{label} must be an exact 0/1 score")
    return int(numeric)


def _verified_bundle(
    summary_path: Path,
    *,
    verifier: E2BVerifier | None,
) -> dict[str, Any]:
    path = summary_path.expanduser().resolve()
    declared = _load_json(path)
    candidate = declared.get("candidate")
    if not isinstance(candidate, str):
        raise Day20MainSelectionV2Error("E2B candidate is missing")
    try:
        identity = factory.parse_candidate_id(candidate)
    except factory.CandidateFactoryV2Error as error:
        raise Day20MainSelectionV2Error("E2B candidate is outside v2 grammar") from error
    if identity.scope != "full112":
        raise Day20MainSelectionV2Error("main selection requires full112 artifacts")
    active_verifier = verifier or e2b.verify_published_pair
    try:
        results, summary, normalized_rows, normalized_summary = active_verifier(
            path,
            expected_candidate=candidate,
            expected_scope="full112",
        )
    except ValueError as error:
        raise Day20MainSelectionV2Error(
            f"E2B verifier rejected {candidate}: {error}"
        ) from error
    if summary != declared:
        raise Day20MainSelectionV2Error("E2B verifier returned a different summary")
    if (
        not isinstance(results, list)
        or not isinstance(normalized_rows, list)
        or not isinstance(normalized_summary, dict)
        or len(normalized_rows) != 112
    ):
        raise Day20MainSelectionV2Error("E2B verifier returned an incomplete bundle")
    metrics = _recompute_metrics(
        results=results,
        e2b_summary=summary,
        normalized_rows=normalized_rows,
        normalized_summary=normalized_summary,
    )
    try:
        classification = factory.classify_main(metrics)
    except factory.CandidateFactoryV2Error as error:
        raise Day20MainSelectionV2Error(str(error)) from error
    order = [row.get("sample_id") for row in normalized_rows]
    if (
        any(not isinstance(sample_id, str) or not sample_id for sample_id in order)
        or len(set(order)) != 112
    ):
        raise Day20MainSelectionV2Error("normalized full112 order is invalid")
    common_identity = {
        "scope": "full112",
        "records": 112,
        "normalized_sample_order_sha256": object_sha256(order),
        "normalized_comparison_key": _require_sha256(
            summary.get("normalized_comparison_key"),
            "normalized_comparison_key",
        ),
        "e2b_comparison_key": _require_sha256(
            summary.get("e2b_comparison_key"), "e2b_comparison_key"
        ),
        "complete_comparison_key": _require_sha256(
            summary.get("complete_comparison_key"), "complete_comparison_key"
        ),
        "e2b_comparison_context_sha256": object_sha256(
            summary.get("e2b_comparison_context")
        ),
    }
    artifact = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": _require_sha256(
            summary.get("summary_sha256"), "E2B summary_sha256"
        ),
        "e2b_run_sha256": _require_sha256(
            summary.get("e2b_run_sha256"), "e2b_run_sha256"
        ),
    }
    return {
        "candidate": candidate,
        "candidate_identity": identity.as_dict(),
        "artifact": artifact,
        "metrics": metrics,
        "classification": classification,
        "common_evaluation_identity": common_identity,
    }


def _recompute_metrics(
    *,
    results: Sequence[Mapping[str, Any]],
    e2b_summary: Mapping[str, Any],
    normalized_rows: Sequence[Mapping[str, Any]],
    normalized_summary: Mapping[str, Any],
) -> dict[str, int]:
    skill_counts = Counter(row.get("slice") for row in normalized_rows)
    if skill_counts != Counter({skill: 28 for skill in factory.SKILLS}):
        raise Day20MainSelectionV2Error("full112 slice distribution drifted")
    correct: dict[str, int] = {}
    for skill in ("general", "math", "finance"):
        total = 0
        for row in normalized_rows:
            if row.get("slice") != skill:
                continue
            score = row.get("normalized_scorer_result")
            if not isinstance(score, Mapping):
                raise Day20MainSelectionV2Error(
                    f"normalized {skill} scorer result is missing"
                )
            total += _score_bit(score.get("score"), f"{skill} score")
        correct[skill] = total

    code_ids = [row.get("sample_id") for row in normalized_rows if row.get("slice") == "code"]
    eligible_ids = [
        row.get("sample_id")
        for row in normalized_rows
        if row.get("slice") == "code"
        and row.get("sandbox_execution_eligible") is True
    ]
    result_ids = [row.get("sample_id") for row in results]
    if (
        len(code_ids) != 28
        or any(not isinstance(value, str) for value in code_ids)
        or any(not isinstance(value, str) for value in eligible_ids)
        or result_ids != eligible_ids
    ):
        raise Day20MainSelectionV2Error("Code eligibility/result order drifted")
    code_correct = sum(
        _score_bit(row.get("score"), "Code E2B score") for row in results
    )
    if (
        e2b_summary.get("code_records") != 28
        or e2b_summary.get("sandbox_execution_eligible") != len(eligible_ids)
        or e2b_summary.get("passed") != code_correct
        or e2b_summary.get("records") != len(results)
    ):
        raise Day20MainSelectionV2Error("Code E2B aggregates drifted")
    e2b_infrastructure = sum(
        row.get("execution_status") == "infrastructure_error" for row in results
    )
    if e2b_summary.get("infrastructure_failures") != e2b_infrastructure:
        raise Day20MainSelectionV2Error("E2B infrastructure aggregate drifted")
    normalized_infrastructure = normalized_summary.get("metrics", {}).get(
        "infrastructure_failures"
    )
    if (
        isinstance(normalized_infrastructure, bool)
        or not isinstance(normalized_infrastructure, int)
        or normalized_infrastructure < 0
    ):
        raise Day20MainSelectionV2Error("normalized infrastructure metric drifted")
    format_compliant = sum(
        row.get("format_compliant") is True for row in normalized_rows
    )
    metrics = {
        "total_correct": correct["general"]
        + correct["math"]
        + correct["finance"]
        + code_correct,
        "general_correct": correct["general"],
        "math_correct": correct["math"],
        "finance_correct": correct["finance"],
        "code_correct": code_correct,
        "format_compliant": format_compliant,
        "code_sandbox_execution_eligible": len(eligible_ids),
        "infrastructure_failures": normalized_infrastructure + e2b_infrastructure,
    }
    return metrics


def _require_common_identity(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not bundles:
        raise Day20MainSelectionV2Error("at least one verified bundle is required")
    common = bundles[0].get("common_evaluation_identity")
    if not isinstance(common, dict) or any(
        bundle.get("common_evaluation_identity") != common for bundle in bundles[1:]
    ):
        raise Day20MainSelectionV2Error(
            "scope/order or normalized/E2B/complete comparison key drifted"
        )
    return dict(common)


def _primary_rank(bundle: Mapping[str, Any]) -> tuple[int, int, int, int]:
    metrics = bundle["metrics"]
    identity = bundle["candidate_identity"]
    return (
        -int(metrics["total_correct"]),
        -int(metrics["general_correct"]),
        -int(metrics["code_correct"]),
        int(identity["target_tokens"]),
    )


def build_primary_selection(
    *,
    base_summary_path: Path,
    primary_summary_paths: Sequence[Path],
    verifier: E2BVerifier | None = None,
) -> dict[str, Any]:
    """Verify Base plus early/mid/final primary artifacts and select one."""
    if len(primary_summary_paths) != 3:
        raise Day20MainSelectionV2Error(
            "primary selection requires exactly early/mid/final summaries"
        )
    base = _verified_bundle(base_summary_path, verifier=verifier)
    if base["candidate"] != factory.base_candidate_id("full112"):
        raise Day20MainSelectionV2Error("Base artifact must use base-full")
    candidates = [
        _verified_bundle(path, verifier=verifier) for path in primary_summary_paths
    ]
    identities = [bundle["candidate_identity"] for bundle in candidates]
    if any(
        identity["run_kind"] != "main"
        or identity["seed"] != factory.PRIMARY_SEED
        for identity in identities
    ):
        raise Day20MainSelectionV2Error(
            "primary artifacts must be primary-seed main candidates"
        )
    labels = {identity["checkpoint_label"] for identity in identities}
    learning_rates = {identity["learning_rate"] for identity in identities}
    if labels != set(factory.MAIN_CHECKPOINT_TOKENS) or len(learning_rates) != 1:
        raise Day20MainSelectionV2Error(
            "primary artifacts must be same-LR early/mid/final"
        )
    candidates.sort(key=lambda bundle: bundle["candidate_identity"]["target_tokens"])
    common = _require_common_identity([base, *candidates])
    eligible = [
        bundle
        for bundle in candidates
        if bundle["classification"]["promotion_eligible"] is True
    ]
    winner = sorted(eligible, key=_primary_rank)[0] if eligible else None
    if winner is None:
        decision = {
            "action": "base_fallback",
            "selected_candidate": base["candidate"],
            "primary_winner": None,
            "confirmation_required": False,
            "reason": "no_primary_checkpoint_passed_main_floors",
        }
    else:
        decision = {
            "action": "run_confirmation",
            "selected_candidate": None,
            "primary_winner": winner["candidate"],
            "confirmation_required": True,
            "reason": "primary_checkpoint_passed_main_floors",
        }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": PRIMARY_DOMAIN,
        "status": "complete",
        "selector_version": SELECTOR_VERSION,
        "base": base,
        "primary_learning_rate": next(iter(learning_rates)),
        "primary_candidates": candidates,
        "common_evaluation_identity": common,
        "selection_policy": {
            "promotion_floors": dict(factory.MAIN_PROMOTION_THRESHOLDS),
            "ranking": [
                "total_correct_desc",
                "general_correct_desc",
                "code_correct_desc",
                "checkpoint_target_tokens_asc",
            ],
            "base_is_fallback_not_ranked_candidate": True,
        },
        "decision": decision,
        "merge_performed": False,
    }
    evidence["selection_sha256"] = object_sha256(evidence)
    return evidence


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish(path: Path, evidence: Mapping[str, Any]) -> None:
    payload = _json_bytes(evidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise Day20MainSelectionV2Error(f"existing evidence drifted: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def publish_primary_selection(
    *,
    base_summary_path: Path,
    primary_summary_paths: Sequence[Path],
    output_dir: Path,
    verifier: E2BVerifier | None = None,
) -> dict[str, Any]:
    evidence = build_primary_selection(
        base_summary_path=base_summary_path,
        primary_summary_paths=primary_summary_paths,
        verifier=verifier,
    )
    path = output_dir.expanduser().resolve() / PRIMARY_FILENAME
    _publish(path, evidence)
    verified = verify_primary_selection(path, verifier=verifier)
    if verified != evidence:
        raise Day20MainSelectionV2Error("published primary evidence changed")
    return verified


def verify_primary_selection(
    path: Path, *, verifier: E2BVerifier | None = None
) -> dict[str, Any]:
    selection_path = path.expanduser().resolve()
    evidence = _load_json(selection_path)
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("domain") != PRIMARY_DOMAIN
        or evidence.get("status") != "complete"
        or evidence.get("selector_version") != SELECTOR_VERSION
        or evidence.get("merge_performed") is not False
    ):
        raise Day20MainSelectionV2Error("primary selection schema drifted")
    _verify_self_hash(evidence, "selection_sha256")
    base = evidence.get("base")
    candidates = evidence.get("primary_candidates")
    if not isinstance(base, Mapping) or not isinstance(candidates, list):
        raise Day20MainSelectionV2Error("primary input identities are missing")
    rebuilt = build_primary_selection(
        base_summary_path=Path(str(base.get("artifact", {}).get("path", ""))),
        primary_summary_paths=[
            Path(str(candidate.get("artifact", {}).get("path", "")))
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ],
        verifier=verifier,
    )
    if evidence != rebuilt or selection_path.read_bytes() != _json_bytes(rebuilt):
        raise Day20MainSelectionV2Error("primary selection no longer matches inputs")
    return evidence


def build_final_promotion(
    *,
    primary_selection_path: Path,
    confirmation_summary_path: Path | None,
    verifier: E2BVerifier | None = None,
) -> dict[str, Any]:
    """Confirm the primary recipe or record a deterministic Base fallback."""
    primary_path = primary_selection_path.expanduser().resolve()
    primary = verify_primary_selection(primary_path, verifier=verifier)
    primary_decision = primary["decision"]
    base = primary["base"]
    winner_id = primary_decision["primary_winner"]
    confirmation: dict[str, Any] | None = None
    if winner_id is None:
        if confirmation_summary_path is not None:
            raise Day20MainSelectionV2Error(
                "primary Base fallback must not consume confirmation"
            )
        decision = {
            "action": "base_fallback",
            "selected_candidate": base["candidate"],
            "validated_primary_candidate": None,
            "confirmation_candidate": None,
            "reason": "no_primary_checkpoint_passed_main_floors",
        }
    else:
        if confirmation_summary_path is None:
            raise Day20MainSelectionV2Error(
                "a primary winner requires a confirmation summary"
            )
        winner = next(
            (
                candidate
                for candidate in primary["primary_candidates"]
                if candidate["candidate"] == winner_id
            ),
            None,
        )
        if winner is None:
            raise Day20MainSelectionV2Error("primary winner evidence is missing")
        confirmation = _verified_bundle(
            confirmation_summary_path, verifier=verifier
        )
        expected = winner["candidate_identity"]
        observed = confirmation["candidate_identity"]
        if (
            observed["run_kind"] != "main"
            or observed["seed"] != factory.CONFIRMATION_SEED
            or observed["learning_rate"] != expected["learning_rate"]
            or observed["checkpoint_label"] != expected["checkpoint_label"]
            or confirmation["common_evaluation_identity"]
            != primary["common_evaluation_identity"]
        ):
            raise Day20MainSelectionV2Error(
                "confirmation must match the primary LR/checkpoint and common eval"
            )
        if confirmation["classification"]["promotion_eligible"] is True:
            decision = {
                "action": "promote_primary_checkpoint",
                "selected_candidate": winner_id,
                "validated_primary_candidate": winner_id,
                "confirmation_candidate": confirmation["candidate"],
                "reason": "primary_and_confirmation_passed_main_floors",
            }
        else:
            decision = {
                "action": "base_fallback",
                "selected_candidate": base["candidate"],
                "validated_primary_candidate": winner_id,
                "confirmation_candidate": confirmation["candidate"],
                "reason": "confirmation_failed_main_floors",
            }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": FINAL_DOMAIN,
        "status": "complete",
        "selector_version": SELECTOR_VERSION,
        "primary_selection": {
            "path": str(primary_path),
            "file_sha256": file_sha256(primary_path),
            "content_sha256": primary["selection_sha256"],
        },
        "base": base,
        "common_evaluation_identity": primary["common_evaluation_identity"],
        "primary_winner": winner_id,
        "confirmation": confirmation,
        "decision": decision,
        "merge_performed": False,
    }
    evidence["promotion_sha256"] = object_sha256(evidence)
    return evidence


def publish_final_promotion(
    *,
    primary_selection_path: Path,
    confirmation_summary_path: Path | None,
    output_dir: Path,
    verifier: E2BVerifier | None = None,
) -> dict[str, Any]:
    evidence = build_final_promotion(
        primary_selection_path=primary_selection_path,
        confirmation_summary_path=confirmation_summary_path,
        verifier=verifier,
    )
    path = output_dir.expanduser().resolve() / FINAL_FILENAME
    _publish(path, evidence)
    verified = verify_final_promotion(path, verifier=verifier)
    if verified != evidence:
        raise Day20MainSelectionV2Error("published final evidence changed")
    return verified


def verify_final_promotion(
    path: Path, *, verifier: E2BVerifier | None = None
) -> dict[str, Any]:
    promotion_path = path.expanduser().resolve()
    evidence = _load_json(promotion_path)
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("domain") != FINAL_DOMAIN
        or evidence.get("status") != "complete"
        or evidence.get("selector_version") != SELECTOR_VERSION
        or evidence.get("merge_performed") is not False
    ):
        raise Day20MainSelectionV2Error("final promotion schema drifted")
    _verify_self_hash(evidence, "promotion_sha256")
    primary = evidence.get("primary_selection")
    confirmation = evidence.get("confirmation")
    if not isinstance(primary, Mapping):
        raise Day20MainSelectionV2Error("final primary-selection identity is missing")
    confirmation_path = None
    if confirmation is not None:
        if not isinstance(confirmation, Mapping):
            raise Day20MainSelectionV2Error("final confirmation identity is malformed")
        confirmation_path = Path(str(confirmation.get("artifact", {}).get("path", "")))
    rebuilt = build_final_promotion(
        primary_selection_path=Path(str(primary.get("path", ""))),
        confirmation_summary_path=confirmation_path,
        verifier=verifier,
    )
    if evidence != rebuilt or promotion_path.read_bytes() != _json_bytes(rebuilt):
        raise Day20MainSelectionV2Error("final promotion no longer matches inputs")
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    primary = subparsers.add_parser("primary")
    primary.add_argument("--base-summary", required=True, type=Path)
    primary.add_argument(
        "--primary-summary", action="append", required=True, type=Path
    )
    primary.add_argument("--output-dir", required=True, type=Path)
    confirmation = subparsers.add_parser("confirmation")
    confirmation.add_argument("--primary-selection", required=True, type=Path)
    confirmation.add_argument("--confirmation-summary", type=Path)
    confirmation.add_argument("--output-dir", required=True, type=Path)
    verify_primary = subparsers.add_parser("verify-primary")
    verify_primary.add_argument("--selection", required=True, type=Path)
    verify_final = subparsers.add_parser("verify-final")
    verify_final.add_argument("--promotion", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "primary":
            result = publish_primary_selection(
                base_summary_path=args.base_summary,
                primary_summary_paths=args.primary_summary,
                output_dir=args.output_dir,
            )
        elif args.command == "confirmation":
            result = publish_final_promotion(
                primary_selection_path=args.primary_selection,
                confirmation_summary_path=args.confirmation_summary,
                output_dir=args.output_dir,
            )
        elif args.command == "verify-primary":
            result = verify_primary_selection(args.selection)
        else:
            result = verify_final_promotion(args.promotion)
    except Day20MainSelectionV2Error as error:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


__all__ = [
    "FINAL_DOMAIN",
    "FINAL_FILENAME",
    "PRIMARY_DOMAIN",
    "PRIMARY_FILENAME",
    "SCHEMA_VERSION",
    "SELECTOR_VERSION",
    "Day20MainSelectionV2Error",
    "build_final_promotion",
    "build_primary_selection",
    "publish_final_promotion",
    "publish_primary_selection",
    "verify_final_promotion",
    "verify_primary_selection",
]


if __name__ == "__main__":
    main()
