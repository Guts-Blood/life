#!/usr/bin/env python3
"""Create a deterministic, stratified Step 7 near-duplicate review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_WORK = HERE.parents[2] / "tmp" / "day09-work"
DEFAULT_INPUT = DEFAULT_WORK / "step6-exact-dedup"
DEFAULT_CANDIDATES = DEFAULT_WORK / "step7-near-candidates"
DEFAULT_OUTPUT = (
    HERE.parent / "artifacts" / "reports" / "day09-step7-near-review.jsonl"
)
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP7_NEAR_REVIEW_START -->"
REPORT_END = "<!-- STEP7_NEAR_REVIEW_END -->"


class ReviewPackageError(ValueError):
    """The review configuration or an upstream artifact is inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def pair_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return candidate["sample_id_a"], candidate["sample_id_b"]


def load_upstream(
    input_dir: Path, candidate_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    near_summary_path = candidate_dir / "near-duplicate-summary.json"
    candidate_path = candidate_dir / "near-duplicate-candidates.jsonl"
    near_summary = load_json(near_summary_path)
    if near_summary.get("status") != "complete_candidates_only":
        raise ReviewPackageError("Step 7 candidate generation is not complete")
    expected_hash = near_summary["output_hashes"][candidate_path.name]
    if file_sha256(candidate_path) != expected_hash:
        raise ReviewPackageError("Step 7 candidate artifact hash mismatch")
    candidates = load_jsonl(candidate_path)

    step6_summary = load_json(input_dir / "exact-dedup-summary.json")
    if step6_summary.get("status") != "complete":
        raise ReviewPackageError("Step 6 exact-dedup pool is not complete")
    rows: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(step6_summary["output_hashes"].items()):
        if not name.endswith("-deduped.jsonl"):
            continue
        path = input_dir / name
        if file_sha256(path) != expected:
            raise ReviewPackageError(f"Step 6 output hash mismatch: {name}")
        for row in load_jsonl(path):
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ReviewPackageError(f"duplicate Step 6 sample_id: {sample_id}")
            rows[sample_id] = row
    missing = {
        sample_id
        for candidate in candidates
        for sample_id in pair_key(candidate)
        if sample_id not in rows
    }
    if missing:
        raise ReviewPackageError(f"Step 7 references missing Step 6 rows: {missing}")
    return candidates, rows, near_summary


def matches_stratum(candidate: dict[str, Any], stratum: dict[str, Any]) -> bool:
    score = float(candidate["similarity_score"])
    if candidate["skill_a"] != stratum["skill"]:
        return False
    if candidate["skill_b"] != stratum["skill"]:
        return False
    if candidate["matched_field"] != stratum["matched_field"]:
        return False
    if score < float(stratum["minimum_score"]):
        return False
    if "maximum_score" in stratum and score > float(stratum["maximum_score"]):
        return False
    if "maximum_score_exclusive" in stratum and score >= float(
        stratum["maximum_score_exclusive"]
    ):
        return False
    return True


def select_candidates(
    candidates: list[dict[str, Any]], calibration: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    selected_pairs: set[tuple[str, str]] = set()
    version = calibration["version"]
    for stratum in calibration["strata"]:
        eligible = [
            candidate
            for candidate in candidates
            if matches_stratum(candidate, stratum)
            and pair_key(candidate) not in selected_pairs
        ]
        eligible.sort(
            key=lambda item: (
                stable_hash(
                    version,
                    stratum["name"],
                    item["candidate_id"],
                ),
                item["candidate_id"],
            )
        )
        quota = int(stratum["quota"])
        if len(eligible) < quota:
            raise ReviewPackageError(
                f"stratum {stratum['name']} has {len(eligible)} eligible "
                f"unique pairs but requires {quota}"
            )
        for candidate in eligible[:quota]:
            selected.append((stratum, candidate))
            selected_pairs.add(pair_key(candidate))
    expected = int(calibration["total_pairs"])
    if len(selected) != expected or len(selected_pairs) != expected:
        raise ReviewPackageError(
            f"selected {len(selected)} records/{len(selected_pairs)} pairs, "
            f"expected {expected}"
        )
    return selected


def sample_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "source": row["source"],
        "revision": row["revision"],
        "split": row["split"],
        "parent_id": row["parent_id"],
        "skill": row["skill"],
        "subskill": row["subskill"],
        "messages": row["messages"],
        "content_hash": row["content_hash"],
        "input_token_count": row["input_token_count"],
        "supervised_token_count": row["supervised_token_count"],
    }


def build_review_records(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    candidates: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_pair[pair_key(candidate)].append(candidate)

    records: list[dict[str, Any]] = []
    for display_index, (stratum, trigger) in enumerate(selected, start=1):
        sample_a, sample_b = pair_key(trigger)
        evidence = sorted(
            by_pair[(sample_a, sample_b)],
            key=lambda item: (item["matched_field"], item["candidate_id"]),
        )
        review_id = "near-review-" + stable_hash(
            calibration["version"], sample_a, sample_b
        )[:16]
        records.append(
            {
                "review_id": review_id,
                "display_index": display_index,
                "calibration_version": calibration["version"],
                "selection_stratum": stratum["name"],
                "selection_candidate_id": trigger["candidate_id"],
                "sample_id_a": sample_a,
                "sample_id_b": sample_b,
                "candidate_evidence": [
                    {
                        "candidate_id": item["candidate_id"],
                        "matched_field": item["matched_field"],
                        "similarity_score": item["similarity_score"],
                        "matcher": item["matcher"],
                        "matcher_version": item["matcher_version"],
                        "matcher_config_sha256": item[
                            "matcher_config_sha256"
                        ],
                        "threshold": item["threshold"],
                    }
                    for item in evidence
                ],
                "sample_a": sample_payload(rows[sample_a]),
                "sample_b": sample_payload(rows[sample_b]),
                "applicable_review_results": calibration[
                    "review_result_values"
                ],
                "review_result": None,
                "reason": None,
                "reviewer": None,
                "reviewed_at": None,
            }
        )
    return records


def write_outputs(
    output_path: Path,
    records: list[dict[str, Any]],
    calibration: dict[str, Any],
    near_summary: dict[str, Any],
    config_path: Path,
    input_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        reported_output_path = output_path.resolve().relative_to(
            HERE.parents[2].resolve()
        ).as_posix()
    except ValueError:
        reported_output_path = output_path.as_posix()
    summary = {
        "status": "complete_review_package_pending_decisions",
        "calibration_version": calibration["version"],
        "pair_count": len(records),
        "selection_method": calibration["selection_method"],
        "stratum_counts": dict(
            sorted(Counter(row["selection_stratum"] for row in records).items())
        ),
        "source_pair_counts": dict(
            sorted(
                Counter(
                    f"{row['sample_a']['skill']} / {row['sample_b']['skill']}"
                    for row in records
                ).items()
            )
        ),
        "decision_counts": {"pending": len(records)},
        "training_pool_changed": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "step6_summary_sha256": file_sha256(
                input_dir / "exact-dedup-summary.json"
            ),
            "near_candidate_summary_sha256": file_sha256(
                candidate_dir / "near-duplicate-summary.json"
            ),
            "near_candidate_jsonl_sha256": near_summary["output_hashes"][
                "near-duplicate-candidates.jsonl"
            ],
        },
        "output": {
            "path": reported_output_path,
            "sha256": file_sha256(output_path),
        },
        "validation_checks": {
            "pair_count_passed": len(records) == calibration["total_pairs"],
            "review_ids_unique": len({row["review_id"] for row in records})
            == len(records),
            "pairs_unique": len(
                {(row["sample_id_a"], row["sample_id_b"]) for row in records}
            )
            == len(records),
            "all_decisions_pending": all(
                row["review_result"] is None for row in records
            ),
            "training_pool_unchanged": True,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def render_report_section(summary: dict[str, Any]) -> str:
    source_counts = summary["source_pair_counts"]
    return "\n".join(
        [
            REPORT_START,
            "",
            "## Step 7 calibration review samples 20 unique near-match pairs",
            "",
            "A deterministic stratified package samples 10 code/code pairs and "
            "10 finance/finance pairs across prompt, answer, and complete-record "
            "matches and across the configured score bands. It is a calibration "
            "set for human/model review, not an automatic filter.",
            "",
            f"- Code/code pairs: {source_counts.get('code / code', 0)}",
            f"- Finance/finance pairs: {source_counts.get('finance / finance', 0)}",
            f"- Pending decisions: {summary['decision_counts']['pending']}",
            "- Training pool changed: false",
            "- Review package: `artifacts/reports/day09-step7-near-review.jsonl`",
            "",
            REPORT_END,
            "",
        ]
    )


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    if not report_path.is_file():
        raise ReviewPackageError(f"missing report: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise ReviewPackageError("Step 7 review report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    calibration = config["near_review_calibration"]
    if sum(int(item["quota"]) for item in calibration["strata"]) != int(
        calibration["total_pairs"]
    ):
        raise ReviewPackageError("stratum quotas do not equal total_pairs")
    candidates, rows, near_summary = load_upstream(
        args.input_dir, args.candidate_dir
    )
    selected = select_candidates(candidates, calibration)
    records = build_review_records(
        selected, candidates, rows, calibration
    )
    summary = write_outputs(
        args.output,
        records,
        calibration,
        near_summary,
        args.config,
        args.input_dir,
        args.candidate_dir,
    )
    if not all(summary["validation_checks"].values()):
        raise ReviewPackageError("review package validation failed")
    update_report(args.report, summary)
    print(
        f"review_pairs={len(records)} pending={len(records)} "
        f"sha256={summary['output']['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
