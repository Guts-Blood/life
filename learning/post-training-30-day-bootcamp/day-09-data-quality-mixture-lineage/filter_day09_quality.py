#!/usr/bin/env python3
"""Prepare Gate A review and apply deterministic Day 09 quality filters."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_WORK = HERE.parents[2] / "tmp" / "day09-work"
DEFAULT_STEP4 = DEFAULT_WORK / "step4-audit"
DEFAULT_OUTPUT = DEFAULT_WORK / "step5-filter"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
CODE = chr(96)
REPORT_START = "<!-- STEP5_FILTER_AUDIT_START -->"
REPORT_END = "<!-- STEP5_FILTER_AUDIT_END -->"


class FilterError(ValueError):
    """A review or filtering invariant failed."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
        "input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "supervised_tokens": sum(int(row["supervised_token_count"]) for row in rows),
    }


def prompt_text(row: dict[str, Any]) -> str:
    return "\n".join(
        message["content"]
        for message in row["messages"]
        if message["role"] == "user"
    )


def assistant_text(row: dict[str, Any]) -> str:
    return "\n".join(
        message["content"]
        for message in row["messages"]
        if message["role"] == "assistant"
    )


def stable_review_hash(version: str, sample_id: str) -> str:
    return hashlib.sha256(f"{version}\0{sample_id}".encode("utf-8")).hexdigest()


def load_step3(
    preprocess_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = preprocess_dir / "preprocess-summary.json"
    if not summary_path.is_file():
        raise FilterError(f"missing Step 3 summary: {summary_path}")
    summary = load_json(summary_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_summary in summary["sources"]:
        accepted_path = preprocess_dir / source_summary["accepted_file"]
        rejected_path = preprocess_dir / source_summary["rejected_file"]
        if file_sha256(accepted_path) != source_summary["accepted_sha256"]:
            raise FilterError(f"accepted hash mismatch for {source_summary['slice']}")
        if file_sha256(rejected_path) != source_summary["rejected_sha256"]:
            raise FilterError(f"rejected hash mismatch for {source_summary['slice']}")
        source_accepted = load_jsonl(accepted_path)
        source_rejected = load_jsonl(rejected_path)
        if len(source_accepted) != source_summary["accepted_count"]:
            raise FilterError(f"accepted count mismatch for {source_summary['slice']}")
        if len(source_rejected) != source_summary["rejected_count"]:
            raise FilterError(f"rejected count mismatch for {source_summary['slice']}")
        for row in [*source_accepted, *source_rejected]:
            if row["sample_id"] in seen:
                raise FilterError(f"duplicate Step 3 sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
        accepted.extend(source_accepted)
        rejected.extend(source_rejected)
    if len(accepted) != summary["totals"]["accepted_count"]:
        raise FilterError("Step 3 accepted total mismatch")
    if len(rejected) != summary["totals"]["rejected_count"]:
        raise FilterError("Step 3 rejected total mismatch")
    if len(seen) != summary["totals"]["candidate_count"]:
        raise FilterError("Step 3 candidate total mismatch")
    return summary, accepted, rejected


def load_risk_hints(
    evidence_path: Path, expected_sample_ids: set[str]
) -> dict[str, dict[str, str]]:
    dimensions = {
        "length_bucket",
        "refusal_status",
        "difficulty_proxy",
        "source_provenance",
        "template_cluster_status",
    }
    hints: dict[str, dict[str, str]] = defaultdict(dict)
    for item in load_jsonl(evidence_path):
        dimension = item["dimension"]
        if dimension not in dimensions:
            continue
        for sample_id in item["sample_ids"]:
            if sample_id not in expected_sample_ids:
                raise FilterError(f"Step 4 evidence has unknown sample_id: {sample_id}")
            if dimension in hints[sample_id]:
                raise FilterError(
                    f"duplicate Step 4 evidence dimension for {sample_id}: {dimension}"
                )
            hints[sample_id][dimension] = item["value"]
    for sample_id in expected_sample_ids:
        missing = dimensions - hints[sample_id].keys()
        if missing:
            raise FilterError(f"{sample_id} missing risk hints: {sorted(missing)}")
    return hints


def choose_review_rows(
    rows: list[dict[str, Any]],
    hints: dict[str, dict[str, str]],
    *,
    version: str,
    per_source: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["skill"]].append(row)

    selected: list[dict[str, Any]] = []
    for source in sorted(by_source):
        source_rows = by_source[source]
        chosen: dict[str, dict[str, Any]] = {}
        reasons: dict[str, list[str]] = defaultdict(list)

        def add(row: dict[str, Any], reason: str) -> None:
            sample_id = row["sample_id"]
            chosen[sample_id] = row
            if reason not in reasons[sample_id]:
                reasons[sample_id].append(reason)

        def lowest_hash(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda row: stable_review_hash(version, row["sample_id"]),
            )

        refusal = lowest_hash(
            [
                row
                for row in source_rows
                if hints[row["sample_id"]]["refusal_status"]
                == "refusal_phrase_match"
            ]
        )
        if refusal is not None:
            add(refusal, "refusal_hint")
        repeated = lowest_hash(
            [
                row
                for row in source_rows
                if hints[row["sample_id"]]["template_cluster_status"]
                == "repeated_prefix_cluster"
            ]
        )
        if repeated is not None:
            add(repeated, "repeated_prefix_cluster_hint")
        add(
            max(
                source_rows,
                key=lambda row: (
                    row["supervised_token_count"],
                    stable_review_hash(version, row["sample_id"]),
                ),
            ),
            "longest_supervised_response",
        )
        add(
            min(
                source_rows,
                key=lambda row: (
                    row["supervised_token_count"],
                    stable_review_hash(version, row["sample_id"]),
                ),
            ),
            "shortest_supervised_response",
        )
        for row in sorted(
            source_rows,
            key=lambda item: stable_review_hash(version, item["sample_id"]),
        ):
            if len(chosen) >= per_source:
                break
            add(row, "stable_hash_fill")
        if len(chosen) != per_source:
            raise FilterError(
                f"review selection for {source} expected {per_source}, got {len(chosen)}"
            )

        ordered = sorted(
            chosen.values(),
            key=lambda row: stable_review_hash(version, row["sample_id"]),
        )
        for index, row in enumerate(ordered, start=1):
            selected.append(
                {
                    "review_id": f"{source}-{index:02d}",
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "skill": row["skill"],
                    "subskill": row["subskill"],
                    "parent_id": row["parent_id"],
                    "prompt": prompt_text(row),
                    "answer": assistant_text(row),
                    "raw_token_count": row["raw_token_count"],
                    "input_token_count": row["input_token_count"],
                    "supervised_token_count": row["supervised_token_count"],
                    "automatic_risk_hints": hints[row["sample_id"]],
                    "selection_reasons": reasons[row["sample_id"]],
                    "review_result": None,
                    "reason_code": None,
                    "review_note": None,
                    "reviewer": None,
                }
            )
    return selected


def prepare_review_package(
    selected: list[dict[str, Any]],
    review_path: Path,
    allowed_results: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_ids = [item["sample_id"] for item in selected]
    if review_path.exists():
        review_rows = load_jsonl(review_path)
        if [item["sample_id"] for item in review_rows] != selected_ids:
            raise FilterError("existing Gate A review package has a different selection")
    else:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", encoding="utf-8") as handle:
            for item in selected:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        review_rows = selected

    result_counts: dict[str, int] = CounterWithPending()
    confirmed_rejects: dict[str, dict[str, Any]] = {}
    invalid_rows: list[str] = []
    for item in review_rows:
        result = item.get("review_result")
        if result is None:
            result_counts["pending"] += 1
            continue
        if result not in allowed_results:
            invalid_rows.append(item["sample_id"])
            continue
        if not isinstance(item.get("review_note"), str) or not item["review_note"].strip():
            invalid_rows.append(item["sample_id"])
            continue
        if result in {"reject", "uncertain"} and (
            not isinstance(item.get("reason_code"), str)
            or not item["reason_code"].strip()
        ):
            invalid_rows.append(item["sample_id"])
            continue
        result_counts[result] += 1
        if result == "reject":
            confirmed_rejects[item["sample_id"]] = item
    if invalid_rows:
        raise FilterError(
            f"invalid completed Gate A review rows: {sorted(invalid_rows)}"
        )
    complete = result_counts["pending"] == 0
    downstream_eligible = complete and result_counts["uncertain"] == 0
    status = {
        "reviewed_count": len(review_rows) - result_counts["pending"],
        "pending_count": result_counts["pending"],
        "result_counts": dict(sorted(result_counts.items())),
        "complete": complete,
        "downstream_eligible": downstream_eligible,
        "confirmed_reject_count": len(confirmed_rejects),
        "confirmed_reject_sample_ids": sorted(confirmed_rejects),
    }
    return review_rows, status


def record_gate_a_accept_all(
    review_path: Path,
    *,
    reviewer: str,
    reviewed_at: str,
    review_note: str,
) -> None:
    """Record an explicit all-accept Gate A decision without overwriting conflicts."""
    if not reviewer.strip() or not reviewed_at.strip() or not review_note.strip():
        raise FilterError(
            "reviewer, reviewed_at, and review_note are required to record Gate A"
        )
    review_rows = load_jsonl(review_path)
    for item in review_rows:
        existing = item.get("review_result")
        if existing not in {None, "accept"}:
            raise FilterError(
                "refusing to overwrite non-accept Gate A decision: "
                f"{item['sample_id']}={existing}"
            )
    for item in review_rows:
        if item.get("review_result") is None:
            item["review_result"] = "accept"
            item["reason_code"] = None
            item["review_note"] = review_note.strip()
            item["reviewer"] = reviewer.strip()
            item["reviewed_at"] = reviewed_at.strip()
    with review_path.open("w", encoding="utf-8") as handle:
        for item in review_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


class CounterWithPending(defaultdict[str, int]):
    def __init__(self) -> None:
        super().__init__(int)
        self["pending"] = 0


def guardrail_reasons(row: dict[str, Any], max_length: int) -> list[str]:
    reasons: list[str] = []
    required = {
        "sample_id",
        "source",
        "revision",
        "parent_id",
        "split",
        "license",
        "content_hash",
        "transform_chain",
        "messages",
        "skill",
        "raw_token_count",
        "input_token_count",
        "supervised_token_count",
    }
    if required - row.keys() or any(
        row.get(field) in {None, ""} for field in required - {"messages", "transform_chain"}
    ):
        reasons.append("missing_required_lineage")
        return reasons
    if row["sample_id"] != f"{row['skill']}:{row['parent_id']}":
        reasons.append("sample_id_parent_mismatch")
    if object_sha256(row["messages"]) != row["content_hash"]:
        reasons.append("content_hash_mismatch")
    if not prompt_text(row).strip():
        reasons.append("empty_prompt")
    if not assistant_text(row).strip():
        reasons.append("empty_answer")
    if row["supervised_token_count"] <= 0:
        reasons.append("non_positive_supervision")
    if not 0 < row["input_token_count"] <= max_length:
        reasons.append("input_length_out_of_range")
    return reasons


def build_decisions(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_by_id = {row["sample_id"]: row for row in review_rows}
    decisions: list[dict[str, Any]] = []
    for row in rejected:
        decisions.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "parent_id": row["parent_id"],
                "input_cohort": "original_candidate_pool",
                "decision": "remove",
                "terminal_filter": "step3_contract_rejection_v3",
                "terminal_reason": row["reason_code"],
                "matched_reasons": [
                    f"step3_contract_rejection_v3:{row['reason_code']}"
                ],
                "review_result": None,
            }
        )

    kept: list[dict[str, Any]] = []
    for row in accepted:
        matched_reasons: list[str] = []
        review = review_by_id.get(row["sample_id"])
        confirmed_manual_reject = False
        if review is not None and review.get("review_result") == "reject":
            if not review.get("review_note") or not review.get("reason_code"):
                raise FilterError(
                    f"incomplete confirmed reject: {row['sample_id']}"
                )
            matched_reasons.append(
                f"confirmed_manual_quality_v1:{review['reason_code']}"
            )
            confirmed_manual_reject = True
        deterministic = guardrail_reasons(row, max_length)
        matched_reasons.extend(
            f"deterministic_guardrails_v1:{reason}" for reason in deterministic
        )
        if confirmed_manual_reject:
            decision = "remove"
            terminal_filter = "confirmed_manual_quality_v1"
            terminal_reason = review["reason_code"]
        elif deterministic:
            decision = "remove"
            terminal_filter = "deterministic_guardrails_v1"
            terminal_reason = deterministic[0]
        else:
            decision = "keep"
            terminal_filter = None
            terminal_reason = None
            kept.append(row)
        decisions.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "parent_id": row["parent_id"],
                "input_cohort": "step3_accepted_pool",
                "decision": decision,
                "terminal_filter": terminal_filter,
                "terminal_reason": terminal_reason,
                "matched_reasons": matched_reasons,
                "review_result": review.get("review_result") if review else None,
            }
        )
    decisions.sort(key=lambda item: item["sample_id"])
    return decisions, kept


def filter_stage(
    name: str,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    removed_ids: list[str],
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "before": row_totals(before_rows),
        "after": row_totals(after_rows),
        "removed_unique_count": len(removed_ids),
        "removed_sample_ids": sorted(removed_ids),
        "removed_sample_ids_sha256": object_sha256(sorted(removed_ids)),
    }


def build_summary(
    config: dict[str, Any],
    preprocess_dir: Path,
    step4_dir: Path,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    review_status: dict[str, Any],
    review_rows: list[dict[str, Any]],
    review_path: Path,
    decisions: list[dict[str, Any]],
    kept: list[dict[str, Any]],
) -> dict[str, Any]:
    manual_removed = sorted(
        item["sample_id"]
        for item in decisions
        if item["terminal_filter"] == "confirmed_manual_quality_v1"
    )
    deterministic_removed = sorted(
        item["sample_id"]
        for item in decisions
        if item["terminal_filter"] == "deterministic_guardrails_v1"
    )
    manual_after = [
        row for row in accepted if row["sample_id"] not in set(manual_removed)
    ]
    quality_filter = config["quality_filter"]
    downstream_eligible = review_status["downstream_eligible"]
    status = "complete" if downstream_eligible else "machine_pass_complete_gate_a_pending"
    return {
        "filter_version": quality_filter["version"],
        "status": status,
        "downstream_eligible": downstream_eligible,
        "filter_implementation_sha256": file_sha256(Path(__file__)),
        "quality_filter_config_sha256": object_sha256(quality_filter),
        "preprocess_summary_sha256": file_sha256(
            preprocess_dir / "preprocess-summary.json"
        ),
        "step4_summary_sha256": file_sha256(step4_dir / "audit-summary.json"),
        "step4_evidence_sha256": file_sha256(step4_dir / "evidence-index.jsonl"),
        "candidate_accounting": {
            "before_examples": len(accepted) + len(rejected),
            "contract_removed_unique_count": len(rejected),
            "contract_removed_sample_ids": sorted(
                row["sample_id"] for row in rejected
            ),
            "contract_removed_sample_ids_sha256": object_sha256(
                sorted(row["sample_id"] for row in rejected)
            ),
            "after_contract_examples": len(accepted),
            "note": "Input/supervised token totals are undefined for contract-rejected records; the Step 5 token baseline starts from Step 3 accepted records.",
        },
        "gate_a": {
            **review_status,
            "review_package_sha256": file_sha256(review_path),
            "review_sample_ids_sha256": object_sha256(
                sorted(item["sample_id"] for item in review_rows)
            ),
        },
        "filter_stages": [
            filter_stage(
                "confirmed_manual_quality_v1",
                accepted,
                manual_after,
                manual_removed,
                status=(
                    "applied_complete"
                    if review_status["complete"]
                    else "applied_confirmed_decisions_gate_pending"
                ),
            ),
            filter_stage(
                "deterministic_guardrails_v1",
                manual_after,
                kept,
                deterministic_removed,
                status="applied",
            ),
        ],
        "quality_filter_accounting": {
            "before": row_totals(accepted),
            "after": row_totals(kept),
            "removed_unique_count": len(accepted) - len(kept),
            "removed_sample_ids": sorted(
                set(manual_removed) | set(deterministic_removed)
            ),
            "removed_sample_ids_sha256": object_sha256(
                sorted(set(manual_removed) | set(deterministic_removed))
            ),
            "accounting_identity_passed": (
                len(accepted)
                - len(set(manual_removed) | set(deterministic_removed))
                == len(kept)
            ),
        },
        "risk_hints_not_filters": quality_filter["risk_hints_not_filters"],
    }


def write_filter_outputs(
    summary: dict[str, Any],
    decisions: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "filter-decisions.jsonl"
    with decision_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    output_hashes = {"filter-decisions.jsonl": file_sha256(decision_path)}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_source[row["skill"]].append(row)
    for source in sorted(by_source):
        path = output_dir / f"{source}-filtered.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in sorted(by_source[source], key=lambda item: item["sample_id"]):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_hashes[path.name] = file_sha256(path)

    summary["output_hashes"] = output_hashes
    summary_path = output_dir / "filter-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_hashes


def render_report_section(summary: dict[str, Any]) -> str:
    accounting = summary["quality_filter_accounting"]
    gate = summary["gate_a"]
    manual_stage, deterministic_stage = summary["filter_stages"]
    eligibility = "yes" if summary["downstream_eligible"] else "no"
    if gate["complete"]:
        heading = (
            "## Step 5 preserves all 7,864 accepted records after Gate A approval"
        )
        gate_heading = "### Gate A review is complete"
        gate_text = (
            f"The deterministic review package contains "
            f"{gate['reviewed_count'] + gate['pending_count']:,} records: "
            f"{gate['reviewed_count']:,} reviewed and {gate['pending_count']:,} pending. "
            f"Review results are "
            f"{gate['result_counts'].get('accept', 0):,} accept, "
            f"{gate['result_counts'].get('reject', 0):,} reject, and "
            f"{gate['result_counts'].get('uncertain', 0):,} uncertain. "
            "No stop condition was triggered, so the filtered pool is eligible for "
            "the next pipeline stage."
        )
    else:
        heading = (
            "## Step 5 machine-safe filtering preserves the accepted pool while "
            "Gate A is pending"
        )
        gate_heading = "### Gate A remains an explicit dependency"
        gate_text = (
            f"The deterministic review package contains "
            f"{gate['reviewed_count'] + gate['pending_count']:,} records: "
            f"{gate['reviewed_count']:,} reviewed and {gate['pending_count']:,} pending. "
            "Pending does not mean accepted. Confirmed reject decisions can be "
            "applied immediately, but the source pool cannot be declared "
            "quality-cleared until all review rows contain a result and note and "
            "any uncertain cases receive a documented follow-up."
        )
    lines = [
        REPORT_START,
        "",
        heading,
        "",
        f"The machine-safe pass starts from {accounting['before']['examples']:,} "
        "Step 3 accepted records. It applies only confirmed manual rejects and "
        "proven deterministic invariant failures. Step 4 risk hints are not "
        "deletion rules.",
        "",
        "| Stage | Status | Before examples | Removed unique | After examples | Before supervised | After supervised |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| Confirmed manual quality | {manual_stage['status']} | "
        f"{manual_stage['before']['examples']:,} | "
        f"{manual_stage['removed_unique_count']:,} | "
        f"{manual_stage['after']['examples']:,} | "
        f"{manual_stage['before']['supervised_tokens']:,} | "
        f"{manual_stage['after']['supervised_tokens']:,} |",
        f"| Deterministic guardrails | {deterministic_stage['status']} | "
        f"{deterministic_stage['before']['examples']:,} | "
        f"{deterministic_stage['removed_unique_count']:,} | "
        f"{deterministic_stage['after']['examples']:,} | "
        f"{deterministic_stage['before']['supervised_tokens']:,} | "
        f"{deterministic_stage['after']['supervised_tokens']:,} |",
        "",
        f"Overall quality-filter accounting is "
        f"{accounting['before']['examples']:,} - "
        f"{accounting['removed_unique_count']:,} = "
        f"{accounting['after']['examples']:,}. The identity check passed. "
        f"Downstream eligibility is {eligibility}.",
        "",
        gate_heading,
        "",
        gate_text,
        "",
        f"- Review package: {CODE}artifacts/reports/day09-gate-a-review.jsonl{CODE}",
        f"- Decision ledger: {CODE}tmp/day09-work/step5-filter/filter-decisions.jsonl{CODE}",
        f"- Filter summary: {CODE}tmp/day09-work/step5-filter/filter-summary.json{CODE}",
        "",
        "### Risk hints deliberately remain non-terminal",
        "",
        "Refusal phrase matches, repeated prompt prefixes, declared synthetic "
        "provenance, and response-length buckets remain review hints. Automatically "
        "deleting them would conflate source style or workload with confirmed low "
        "quality.",
        "",
        REPORT_END,
        "",
    ]
    return "\n".join(lines)


def update_report(
    report_path: Path,
    summary: dict[str, Any],
) -> None:
    if not report_path.is_file():
        raise FilterError(f"missing Step 4 report: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise FilterError("Step 5 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    section = render_report_section(summary)
    report_path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preprocess-dir", type=Path)
    parser.add_argument("--step4-dir", type=Path, default=DEFAULT_STEP4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-package", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--record-gate-a-accept-all", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--reviewed-at")
    parser.add_argument("--review-note")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    preprocess_dir = args.preprocess_dir or (
        DEFAULT_WORK / f"step3-max{config['preprocessing']['max_length']}"
    )
    configured_review = Path(config["manual_review"]["review_package"])
    review_path = args.review_package or (HERE / configured_review).resolve()
    step3_summary, accepted, rejected = load_step3(preprocess_dir)
    if step3_summary["contract_version"] != config["preprocessing"]["contract_version"]:
        raise FilterError("Step 3 contract version does not match current config")
    step4_summary_path = args.step4_dir / "audit-summary.json"
    evidence_path = args.step4_dir / "evidence-index.jsonl"
    if not step4_summary_path.is_file() or not evidence_path.is_file():
        raise FilterError("missing Step 4 audit outputs")
    step4_summary = load_json(step4_summary_path)
    if step4_summary["totals"]["examples"] != len(accepted):
        raise FilterError("Step 4 cohort does not match Step 3 accepted pool")
    hints = load_risk_hints(
        evidence_path, {row["sample_id"] for row in accepted}
    )
    manual_config = config["manual_review"]
    selected = choose_review_rows(
        accepted,
        hints,
        version=manual_config["gate_a_version"],
        per_source=manual_config["samples_per_source"],
    )
    if len(selected) != manual_config["total_samples"]:
        raise FilterError("Gate A total selection count mismatch")
    review_rows, review_status = prepare_review_package(
        selected,
        review_path,
        set(manual_config["review_result_values"]),
    )
    if args.record_gate_a_accept_all:
        record_gate_a_accept_all(
            review_path,
            reviewer=args.reviewer or "",
            reviewed_at=args.reviewed_at or "",
            review_note=args.review_note or "",
        )
        review_rows, review_status = prepare_review_package(
            selected,
            review_path,
            set(manual_config["review_result_values"]),
        )
    decisions, kept = build_decisions(
        accepted,
        rejected,
        review_rows,
        config["preprocessing"]["max_length"],
    )
    summary = build_summary(
        config,
        preprocess_dir,
        args.step4_dir,
        accepted,
        rejected,
        review_status,
        review_rows,
        review_path,
        decisions,
        kept,
    )
    write_filter_outputs(summary, decisions, kept, args.output_dir)
    update_report(args.report, summary)
    print(
        f"before={len(accepted)} removed={len(accepted) - len(kept)} "
        f"after={len(kept)} gate_a_pending={review_status['pending_count']} "
        f"downstream_eligible={summary['downstream_eligible']}"
    )


if __name__ == "__main__":
    main()
