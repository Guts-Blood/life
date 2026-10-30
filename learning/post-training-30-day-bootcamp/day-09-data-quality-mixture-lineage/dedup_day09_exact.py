#!/usr/bin/env python3
"""Apply deterministic exact deduplication to the Step 5 Day 09 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_WORK = HERE.parents[2] / "tmp" / "day09-work"
DEFAULT_INPUT = DEFAULT_WORK / "step5-filter"
DEFAULT_OUTPUT = DEFAULT_WORK / "step6-exact-dedup"
DEFAULT_REVIEW = HERE.parent / "artifacts" / "reports" / "day09-gate-a-review.jsonl"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP6_EXACT_DEDUP_START -->"
REPORT_END = "<!-- STEP6_EXACT_DEDUP_END -->"
CODE = chr(96)
EXPECTED_NORMALIZATION_STEPS = [
    "Unicode NFKC",
    "lowercase",
    "collapse consecutive whitespace",
    "strip leading and trailing whitespace",
    "keep punctuation",
]


class DedupError(ValueError):
    """An exact-dedup input or accounting invariant failed."""


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


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_role_text(row: dict[str, Any], role: str) -> str:
    return normalize_text(
        "\n".join(
            message["content"]
            for message in row["messages"]
            if message["role"] == role
        )
    )


def normalized_complete_record(row: dict[str, Any]) -> str:
    content = [
        [message["role"], normalize_text(message["content"])]
        for message in row["messages"]
    ]
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def normalized_fields(row: dict[str, Any]) -> dict[str, str]:
    return {
        "normalized_prompt": normalized_role_text(row, "user"),
        "normalized_answer": normalized_role_text(row, "assistant"),
        "normalized_complete_record": normalized_complete_record(row),
    }


def row_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
        "input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "supervised_tokens": sum(
            int(row["supervised_token_count"]) for row in rows
        ),
    }


def load_step5(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = input_dir / "filter-summary.json"
    if not summary_path.is_file():
        raise DedupError(f"missing Step 5 summary: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("status") != "complete" or not summary.get(
        "downstream_eligible"
    ):
        raise DedupError("Step 5 pool is not downstream eligible")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    filtered_files = sorted(
        name
        for name in summary["output_hashes"]
        if name.endswith("-filtered.jsonl")
    )
    if not filtered_files:
        raise DedupError("Step 5 summary has no filtered pool files")
    for name in filtered_files:
        path = input_dir / name
        if file_sha256(path) != summary["output_hashes"][name]:
            raise DedupError(f"Step 5 output hash mismatch: {name}")
        for row in load_jsonl(path):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise DedupError(f"duplicate input sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    if row_totals(rows) != summary["quality_filter_accounting"]["after"]:
        raise DedupError("Step 5 row/token totals do not match its summary")
    return summary, rows


def load_manual_accepts(
    review_path: Path, expected_sha256: str
) -> set[str]:
    if file_sha256(review_path) != expected_sha256:
        raise DedupError("Gate A review package hash does not match Step 5")
    rows = load_jsonl(review_path)
    if any(row.get("review_result") not in {"accept", "reject"} for row in rows):
        raise DedupError("Gate A review package is incomplete or uncertain")
    return {
        row["sample_id"]
        for row in rows
        if row.get("review_result") == "accept"
    }


def survivor_key(
    row: dict[str, Any],
    manual_accepts: set[str],
    slice_priority: list[str],
) -> tuple[int, int, str]:
    try:
        priority = slice_priority.index(row["skill"])
    except ValueError as exc:
        raise DedupError(f"missing slice priority for {row['skill']}") from exc
    return (
        0 if row["sample_id"] in manual_accepts else 1,
        priority,
        row["sample_id"],
    )


def build_exact_groups(
    rows: list[dict[str, Any]],
    *,
    match_fields: list[str],
    terminal_field: str,
    manual_accepts: set[str],
    slice_priority: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    values: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: defaultdict(list) for field in match_fields
    }
    for row in rows:
        fields = normalized_fields(row)
        for field in match_fields:
            values[field][fields[field]].append(row)

    evidence: list[dict[str, Any]] = []
    removed_to_survivor: dict[str, str] = {}
    memberships: dict[str, list[str]] = defaultdict(list)
    for field in match_fields:
        groups = [members for members in values[field].values() if len(members) > 1]
        groups.sort(key=lambda members: min(row["sample_id"] for row in members))
        for members in groups:
            normalized_value = normalized_fields(members[0])[field]
            normalized_hash = text_sha256(normalized_value)
            group_id = f"{field}:{normalized_hash}"
            members = sorted(members, key=lambda row: row["sample_id"])
            for row in members:
                memberships[row["sample_id"]].append(group_id)
            survivor = None
            removed_ids: list[str] = []
            if field == terminal_field:
                survivor = min(
                    members,
                    key=lambda row: survivor_key(
                        row, manual_accepts, slice_priority
                    ),
                )
                for row in members:
                    if row["sample_id"] != survivor["sample_id"]:
                        removed_to_survivor[row["sample_id"]] = survivor["sample_id"]
                        removed_ids.append(row["sample_id"])
            sources = sorted({row["source"] for row in members})
            evidence.append(
                {
                    "match_group_id": group_id,
                    "matched_field": field,
                    "normalized_sha256": normalized_hash,
                    "group_size": len(members),
                    "scope": "cross_source" if len(sources) > 1 else "within_source",
                    "member_sample_ids": [row["sample_id"] for row in members],
                    "member_sources": sources,
                    "member_slices": sorted({row["skill"] for row in members}),
                    "manual_accept_sample_ids": sorted(
                        row["sample_id"]
                        for row in members
                        if row["sample_id"] in manual_accepts
                    ),
                    "survivor_sample_id": (
                        survivor["sample_id"] if survivor is not None else None
                    ),
                    "removed_sample_ids": sorted(removed_ids),
                }
            )
    evidence.sort(key=lambda item: (item["matched_field"], item["match_group_id"]))
    return evidence, removed_to_survivor, memberships


def build_decisions(
    rows: list[dict[str, Any]],
    removed_to_survivor: dict[str, str],
    memberships: dict[str, list[str]],
    manual_accepts: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        survivor = removed_to_survivor.get(row["sample_id"])
        if survivor is None:
            kept.append(row)
        decisions.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "skill": row["skill"],
                "decision": "remove" if survivor is not None else "keep",
                "terminal_reason": (
                    "exact_normalized_complete_record_duplicate"
                    if survivor is not None
                    else None
                ),
                "survivor_sample_id": survivor,
                "matched_exact_group_ids": sorted(
                    memberships.get(row["sample_id"], [])
                ),
                "manual_gate_a_accept": row["sample_id"] in manual_accepts,
            }
        )
    return decisions, kept


def match_statistics(
    evidence: list[dict[str, Any]], match_fields: list[str]
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for field in match_fields:
        groups = [item for item in evidence if item["matched_field"] == field]
        stats[field] = {
            "duplicate_groups": len(groups),
            "affected_examples": sum(item["group_size"] for item in groups),
            "duplicate_excess": sum(item["group_size"] - 1 for item in groups),
            "within_source_groups": sum(
                item["scope"] == "within_source" for item in groups
            ),
            "cross_source_groups": sum(
                item["scope"] == "cross_source" for item in groups
            ),
            "maximum_group_size": max(
                (item["group_size"] for item in groups), default=0
            ),
        }
    return stats


def source_distribution(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    before_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    after_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in before:
        before_by_skill[row["skill"]].append(row)
    for row in after:
        after_by_skill[row["skill"]].append(row)
    return [
        {
            "skill": skill,
            "before": row_totals(before_by_skill[skill]),
            "after": row_totals(after_by_skill[skill]),
            "removed_unique_count": (
                len(before_by_skill[skill]) - len(after_by_skill[skill])
            ),
        }
        for skill in sorted(before_by_skill)
    ]


def build_summary(
    config: dict[str, Any],
    input_dir: Path,
    step5_summary: dict[str, Any],
    review_path: Path,
    evidence: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    removed_to_survivor: dict[str, str],
) -> dict[str, Any]:
    exact_config = config["exact_dedup"]
    normalization = config["normalization"]
    match_fields = exact_config["match_fields"]
    before = row_totals(rows)
    after = row_totals(kept)
    removed_ids = sorted(removed_to_survivor)
    removed_id_set = set(removed_ids)
    removed = row_totals(
        [row for row in rows if row["sample_id"] in removed_id_set]
    )
    kept_ids = {row["sample_id"] for row in kept}
    accounting_identity_passed = all(
        before[key] - removed[key] == after[key] for key in before
    )
    survivor_integrity_passed = all(
        survivor_id in kept_ids and removed_id not in kept_ids
        for removed_id, survivor_id in removed_to_survivor.items()
    )
    terminal_uniqueness_passed = len(
        {normalized_complete_record(row) for row in kept}
    ) == len(kept)
    return {
        "dedup_version": exact_config["version"],
        "status": "complete",
        "downstream_eligible": True,
        "implementation_sha256": file_sha256(Path(__file__)),
        "exact_dedup_config_sha256": object_sha256(exact_config),
        "normalization": {
            **normalization,
            "config_sha256": object_sha256(normalization),
        },
        "input": {
            "cohort": exact_config["input_cohort"],
            "step5_summary_sha256": file_sha256(input_dir / "filter-summary.json"),
            "gate_a_review_sha256": file_sha256(review_path),
            "examples": before["examples"],
        },
        "match_fields": match_fields,
        "terminal_match_field": exact_config["terminal_match_field"],
        "survivor_rule": exact_config["survivor_rule"],
        "match_statistics": match_statistics(evidence, match_fields),
        "dedup_accounting": {
            "before": before,
            "removed": removed,
            "after": after,
            "removed_unique_count": len(removed_ids),
            "removed_sample_ids": removed_ids,
            "removed_sample_ids_sha256": object_sha256(removed_ids),
            "removed_to_survivor": dict(sorted(removed_to_survivor.items())),
            "accounting_identity_passed": accounting_identity_passed,
            "survivor_integrity_passed": survivor_integrity_passed,
            "terminal_uniqueness_passed": terminal_uniqueness_passed,
        },
        "source_distribution": source_distribution(rows, kept),
        "step5_gate_a": step5_summary["gate_a"],
    }


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    evidence: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    kept: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "exact-match-groups.jsonl"
    with evidence_path.open("w", encoding="utf-8") as handle:
        for item in evidence:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    decisions_path = output_dir / "exact-dedup-decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    output_hashes = {
        evidence_path.name: file_sha256(evidence_path),
        decisions_path.name: file_sha256(decisions_path),
    }
    before_skills = sorted({row["skill"] for row in rows})
    kept_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        kept_by_skill[row["skill"]].append(row)
    for skill in before_skills:
        path = output_dir / f"{skill}-deduped.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in sorted(
                kept_by_skill[skill], key=lambda item: item["sample_id"]
            ):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_hashes[path.name] = file_sha256(path)

    summary["output_hashes"] = dict(sorted(output_hashes.items()))
    with (output_dir / "exact-dedup-summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def render_report_section(summary: dict[str, Any]) -> str:
    accounting = summary["dedup_accounting"]
    stats = summary["match_statistics"]
    normalization = summary["normalization"]
    lines = [
        REPORT_START,
        "",
        f"## Step 6 exact dedup removes {accounting['removed_unique_count']:,} "
        "normalized complete-record duplicates",
        "",
        f"The downstream-eligible Step 5 cohort contains "
        f"{accounting['before']['examples']:,} records. Exact dedup retains "
        f"{accounting['after']['examples']:,} records after removing "
        f"{accounting['removed_unique_count']:,} non-survivors. Prompt-only and "
        "answer-only matches remain evidence and do not trigger deletion.",
        "",
        "| Match field | Duplicate groups | Affected examples | Duplicate excess | Within source | Cross source | Max group |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field in summary["match_fields"]:
        item = stats[field]
        lines.append(
            f"| {field} | {item['duplicate_groups']:,} | "
            f"{item['affected_examples']:,} | {item['duplicate_excess']:,} | "
            f"{item['within_source_groups']:,} | "
            f"{item['cross_source_groups']:,} | "
            f"{item['maximum_group_size']:,} |"
        )
    lines.extend(
        [
            "",
            "### The survivor rule is deterministic and order-independent",
            "",
            "Complete-record groups select one survivor using: manual Gate A accept "
            "first, then slice priority finance/general/math/code, then the lowest "
            "canonical sample_id. Every removed ID points to that survivor in the "
            "decision ledger.",
            "",
            f"Normalization is {CODE}{normalization['name']}{CODE} with config hash "
            f"{CODE}{normalization['config_sha256']}{CODE}: Unicode NFKC, lowercase, "
            "collapse whitespace, trim boundaries, and retain punctuation. Complete "
            "records are canonical JSON arrays of role and normalized-content pairs.",
            "",
            "| Slice | Before | Removed | After | Before supervised | After supervised |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["source_distribution"]:
        lines.append(
            f"| {item['skill']} | {item['before']['examples']:,} | "
            f"{item['removed_unique_count']:,} | {item['after']['examples']:,} | "
            f"{item['before']['supervised_tokens']:,} | "
            f"{item['after']['supervised_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            f"Overall accounting is {accounting['before']['examples']:,} - "
            f"{accounting['removed_unique_count']:,} = "
            f"{accounting['after']['examples']:,}; example, raw-token, input-token, "
            "and supervised-token identities all passed. Survivor pointers resolve "
            "to kept records, and the output complete-record hashes are unique.",
            "",
            "### Scope and limitation",
            "",
            "This stage proves equality only under the frozen exact normalization. "
            "It does not claim that prompt-only, answer-only, paraphrased, or "
            "template-similar records are duplicates. Those remain candidates for "
            "Step 7 near-duplicate review.",
            "",
            f"- Match evidence: {CODE}tmp/day09-work/step6-exact-dedup/exact-match-groups.jsonl{CODE}",
            f"- Decision ledger: {CODE}tmp/day09-work/step6-exact-dedup/exact-dedup-decisions.jsonl{CODE}",
            f"- Summary: {CODE}tmp/day09-work/step6-exact-dedup/exact-dedup-summary.json{CODE}",
            "",
            REPORT_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    if not report_path.is_file():
        raise DedupError(f"missing quality report: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise DedupError("Step 6 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary),
        encoding="utf-8",
    )


def validate_config(config: dict[str, Any]) -> None:
    if config["normalization"]["steps"] != EXPECTED_NORMALIZATION_STEPS:
        raise DedupError("normalization config does not match implementation")
    exact = config["exact_dedup"]
    expected_fields = {
        "normalized_prompt",
        "normalized_answer",
        "normalized_complete_record",
    }
    if set(exact["match_fields"]) != expected_fields:
        raise DedupError("exact-dedup match fields do not match implementation")
    if exact["terminal_match_field"] != "normalized_complete_record":
        raise DedupError("only normalized complete records may trigger deletion")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-package", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    validate_config(config)
    step5_summary, rows = load_step5(args.input_dir)
    manual_accepts = load_manual_accepts(
        args.review_package, step5_summary["gate_a"]["review_package_sha256"]
    )
    exact = config["exact_dedup"]
    evidence, removed_to_survivor, memberships = build_exact_groups(
        rows,
        match_fields=exact["match_fields"],
        terminal_field=exact["terminal_match_field"],
        manual_accepts=manual_accepts,
        slice_priority=exact["slice_priority"],
    )
    decisions, kept = build_decisions(
        rows, removed_to_survivor, memberships, manual_accepts
    )
    summary = build_summary(
        config,
        args.input_dir,
        step5_summary,
        args.review_package,
        evidence,
        rows,
        kept,
        removed_to_survivor,
    )
    accounting = summary["dedup_accounting"]
    if not accounting["accounting_identity_passed"]:
        raise DedupError("exact-dedup example/token accounting failed")
    if not accounting["survivor_integrity_passed"]:
        raise DedupError("exact-dedup survivor integrity failed")
    if not accounting["terminal_uniqueness_passed"]:
        raise DedupError("exact-dedup output still has complete-record duplicates")
    write_outputs(args.output_dir, summary, evidence, decisions, rows, kept)
    update_report(args.report, summary)
    print(
        f"before={len(rows)} removed={len(removed_to_survivor)} "
        f"after={len(kept)} complete_groups="
        f"{summary['match_statistics']['normalized_complete_record']['duplicate_groups']}"
    )


if __name__ == "__main__":
    main()
