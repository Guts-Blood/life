#!/usr/bin/env python3
"""Apply confirmed Day 09 decisions and freeze the clean parent manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BOOTCAMP = HERE.parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_WORK = REPO_ROOT / "tmp" / "day09-work"
DEFAULT_STEP6 = DEFAULT_WORK / "step6-exact-dedup"
DEFAULT_STEP7 = DEFAULT_WORK / "step7-near-candidates"
DEFAULT_STEP9 = DEFAULT_WORK / "step9-decontamination"
DEFAULT_NEAR_REVIEW = BOOTCAMP / "artifacts" / "reports" / "day09-step7-near-review.jsonl"
DEFAULT_MANIFEST = BOOTCAMP / "artifacts" / "data" / "day09-dataset-manifest.json"
DEFAULT_GATE_B = BOOTCAMP / "artifacts" / "reports" / "day09-gate-b-review.jsonl"
DEFAULT_REBUILD = BOOTCAMP / "artifacts" / "reports" / "day09-manifest-rebuild-evidence.jsonl"
DEFAULT_OUTPUT = DEFAULT_WORK / "step10-clean-pool"
DEFAULT_REPORT = BOOTCAMP / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP10_CLEAN_POOL_START -->"
REPORT_END = "<!-- STEP10_CLEAN_POOL_END -->"
REQUIRED_RECORD_FIELDS = {
    "sample_id",
    "source",
    "license",
    "revision",
    "parent_id",
    "split",
    "skill",
    "subskill",
    "language",
    "content_hash",
    "transform_chain",
    "raw_token_count",
    "input_token_count",
    "supervised_token_count",
    "messages",
}


class CleanPoolError(ValueError):
    """A Gate B decision, lineage input, or manifest invariant failed."""


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


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def pool_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
        "input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "supervised_tokens": sum(
            int(row["supervised_token_count"]) for row in rows
        ),
    }


def load_step6(step6_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = step6_dir / "exact-dedup-summary.json"
    summary = load_json(summary_path)
    accounting = summary.get("dedup_accounting", {})
    if summary.get("status") != "complete" or not summary.get(
        "downstream_eligible"
    ):
        raise CleanPoolError("Step 6 pool is not downstream eligible")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, expected_hash in sorted(summary["output_hashes"].items()):
        if not name.endswith("-deduped.jsonl"):
            continue
        path = step6_dir / name
        if file_sha256(path) != expected_hash:
            raise CleanPoolError(f"Step 6 hash mismatch: {name}")
        for row in load_jsonl(path):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise CleanPoolError(f"duplicate Step 6 sample_id: {sample_id}")
            missing = REQUIRED_RECORD_FIELDS - set(row)
            if missing:
                raise CleanPoolError(f"{sample_id} missing manifest fields: {missing}")
            seen.add(sample_id)
            rows.append(row)
    if pool_totals(rows) != accounting["after"]:
        raise CleanPoolError("Step 6 totals do not match its summary")
    return summary, sorted(rows, key=lambda item: item["sample_id"])


def load_step7(
    step7_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = step7_dir / "near-duplicate-summary.json"
    candidate_path = step7_dir / "near-duplicate-candidates.jsonl"
    summary = load_json(summary_path)
    if summary.get("status") != "complete_candidates_only":
        raise CleanPoolError("Step 7 candidate generation is incomplete")
    if file_sha256(candidate_path) != summary["output_hashes"][candidate_path.name]:
        raise CleanPoolError("Step 7 candidate hash mismatch")
    return summary, load_jsonl(candidate_path)


def load_step9(
    step9_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = step9_dir / "decontamination-summary.json"
    candidate_path = step9_dir / "train-eval-overlap-candidates.jsonl"
    summary = load_json(summary_path)
    if summary.get("status") != "complete_candidates_only":
        raise CleanPoolError("Step 9 candidate generation is incomplete")
    if file_sha256(candidate_path) != summary["output_hashes"][candidate_path.name]:
        raise CleanPoolError("Step 9 candidate hash mismatch")
    candidates = load_jsonl(candidate_path)
    if len(candidates) != summary["candidate_statistics"]["candidate_records"]:
        raise CleanPoolError("Step 9 candidate count mismatch")
    return summary, candidates


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["sample_id_a"], row["sample_id_b"]


def build_gate_b_ledger(
    near_candidates: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    overlap_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decision_config = config["near_review_calibration"]["gate_b_decision"]
    if decision_config["result"] != "20 keep_both, 0 removal, 0 uncertain":
        raise CleanPoolError("unsupported user Gate B decision")
    if len(review_rows) != int(decision_config["reviewed_pairs"]):
        raise CleanPoolError("manual review pair count does not match config")
    manually_reviewed = {
        (row["sample_id_a"], row["sample_id_b"]) for row in review_rows
    }
    if len(manually_reviewed) != len(review_rows):
        raise CleanPoolError("manual review package contains duplicate pairs")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in near_candidates:
        grouped[pair_key(candidate)].append(candidate)
    if not manually_reviewed <= set(grouped):
        raise CleanPoolError("manual review contains pairs absent from Step 7")

    ledger: list[dict[str, Any]] = []
    for pair in sorted(grouped):
        evidence = sorted(
            grouped[pair],
            key=lambda item: (item["matched_field"], item["candidate_id"]),
        )
        skill = evidence[0]["skill_a"]
        manual = pair in manually_reviewed
        reason_code = (
            "intentional_parameterized_variant"
            if skill == "finance"
            else "intentional_paraphrase_or_template_variant"
        )
        ledger.append(
            {
                "gate_b_decision_id": object_sha256(
                    ["day09_gate_b_v1", pair[0], pair[1]]
                ),
                "candidate_kind": "train_train_near_duplicate",
                "sample_id_a": pair[0],
                "sample_id_b": pair[1],
                "candidate_ids": [item["candidate_id"] for item in evidence],
                "matched_fields": [item["matched_field"] for item in evidence],
                "similarity_scores": [
                    item["similarity_score"] for item in evidence
                ],
                "decision": "keep_both",
                "decision_reason": reason_code,
                "decision_basis": (
                    "user_manual_stratified_review"
                    if manual
                    else "candidate_is_non_terminal_no_confirmed_removal"
                ),
                "reviewer": (
                    decision_config["reviewer"]
                    if manual
                    else "policy:confirmed-removals-only"
                ),
                "reviewed_at": config["clean_pool"]["created_at"],
            }
        )

    if overlap_candidates:
        raise CleanPoolError(
            "Step 9 has contamination candidates but no explicit Gate B ledger"
        )
    statistics = {
        "status": "complete_for_clean_pool",
        "unique_near_pairs": len(grouped),
        "manual_keep_both_pairs": len(manually_reviewed),
        "policy_keep_both_pairs": len(grouped) - len(manually_reviewed),
        "train_eval_candidate_count": 0,
        "confirmed_near_removals": 0,
        "confirmed_contamination_removals": 0,
        "uncertain_count": 0,
        "confirmed_removed_sample_ids": [],
        "decision_counts": dict(
            sorted(Counter(row["decision"] for row in ledger).items())
        ),
        "scope_note": (
            "Twenty stratified pairs were manually reviewed by the user. Remaining "
            "within-source Step 7 candidates are retained under the frozen "
            "candidate-is-not-removal policy; they are not relabeled as exact "
            "duplicates. Step 9 produced no train/eval candidates."
        ),
    }
    return ledger, statistics


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def distribution_by_skill(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_skill[row["skill"]].append(row)
    return {skill: pool_totals(items) for skill, items in sorted(by_skill.items())}


def source_lineage(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "slice": source["slice"],
            "source": source["source_id"],
            "revision": source["revision"],
            "split": source["split"],
            "license": source["license"],
            "license_note": source["license_note"],
        }
        for source in config["sources"]
    ]


def clean_pool_dependency_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "assignment",
        "preregistered_date",
        "intended_use",
        "candidate_samples_per_slice",
        "sources",
        "selection",
        "amendments",
        "preprocessing",
        "identity",
        "token_count_definitions",
        "quality_audit",
        "normalization",
        "exact_dedup",
        "near_duplicate",
        "near_review_calibration",
        "eval_candidates",
        "decontamination",
        "manual_review",
        "quality_filter",
        "clean_pool",
    )
    return {key: config[key] for key in keys}


def manifest_payload(
    config: dict[str, Any],
    step6_dir: Path,
    step7_dir: Path,
    step9_dir: Path,
    step6_summary: dict[str, Any],
    step7_summary: dict[str, Any],
    step9_summary: dict[str, Any],
    gate_b_path: Path,
    gate_b_summary_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean = config["clean_pool"]
    records = sorted(rows, key=lambda item: item["sample_id"])
    totals = pool_totals(records)
    exact_removed = int(
        step6_summary["dedup_accounting"]["removed_unique_count"]
    )
    pre_filter_accepted = int(
        step6_summary["dedup_accounting"]["before"]["examples"]
    )
    if pre_filter_accepted - exact_removed != totals["examples"]:
        raise CleanPoolError("clean-pool accounting identity failed")
    header = {
        "manifest_name": "day09-dataset-manifest",
        "manifest_version": clean["version"],
        "created_at": clean["created_at"],
        "intended_use": config["intended_use"],
        "source_lineage": source_lineage(config),
        "stable_record_order": clean["stable_record_order"],
        "hash_definitions": {
            "clean_pool_hash": clean["clean_pool_hash_definition"],
            "manifest_hash": clean["manifest_hash_definition"],
            "record_content_hash": config["identity"]["content_hash"],
        },
        "config_hashes": {
            "clean_pool_dependency_config_sha256": object_sha256(
                clean_pool_dependency_config(config)
            ),
            "selection_config_sha256": object_sha256(config["selection"]),
            "preprocessing_config_sha256": object_sha256(config["preprocessing"]),
            "quality_filter_config_sha256": object_sha256(config["quality_filter"]),
            "exact_dedup_config_sha256": object_sha256(config["exact_dedup"]),
            "near_duplicate_config_sha256": object_sha256(config["near_duplicate"]),
            "decontamination_config_sha256": object_sha256(
                config["decontamination"]
            ),
            "clean_pool_config_sha256": object_sha256(clean),
        },
        "input_artifact_hashes": {
            "step6_summary_sha256": file_sha256(
                step6_dir / "exact-dedup-summary.json"
            ),
            "step7_summary_sha256": file_sha256(
                step7_dir / "near-duplicate-summary.json"
            ),
            "step9_decontamination_report_sha256": file_sha256(
                step9_dir / "decontamination-summary.json"
            ),
            "gate_b_ledger_sha256": file_sha256(gate_b_path),
            "gate_b_summary_sha256": file_sha256(gate_b_summary_path),
        },
        "totals": totals,
        "distribution_by_skill": distribution_by_skill(records),
        "removal_accounting": {
            "pre_filter_accepted_examples": pre_filter_accepted,
            "quality_removals": 0,
            "exact_dedup_removals": exact_removed,
            "confirmed_near_duplicate_removals": 0,
            "confirmed_contamination_removals": 0,
            "total_confirmed_unique_removals": exact_removed,
            "clean_examples": totals["examples"],
            "identity": f"{pre_filter_accepted} - {exact_removed} = {totals['examples']}",
        },
        "gate_b": {
            "status": "complete_for_clean_pool",
            "confirmed_removed_sample_ids": [],
            "step7_candidate_records": step7_summary["candidate_statistics"][
                "candidate_records"
            ],
            "step9_candidate_records": step9_summary["candidate_statistics"][
                "candidate_records"
            ],
        },
        "clean_pool_hash": object_sha256(records),
    }
    manifest_without_hash = {"header": header, "records": records}
    header["manifest_hash"] = object_sha256(manifest_without_hash)
    return {"header": header, "records": records}


def load_candidate_rows(work_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for skill in ("general", "math", "code", "finance"):
        path = work_dir / f"{skill}-candidates.jsonl"
        hashes[skill] = file_sha256(path)
        for row in load_jsonl(path):
            rows[row["sample_id"]] = row
    return rows, hashes


def build_rebuild_evidence(
    config: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    candidate_hashes: dict[str, str],
    step6_summary: dict[str, Any],
    step6_dir: Path,
) -> list[dict[str, Any]]:
    version = config["clean_pool"]["version"]
    count = int(config["clean_pool"]["rebuild_sample_count"])
    selected = sorted(
        manifest_rows,
        key=lambda row: (
            hashlib.sha256(
                f"{version}\0rebuild\0{row['sample_id']}".encode("utf-8")
            ).hexdigest(),
            row["sample_id"],
        ),
    )[:count]
    evidence: list[dict[str, Any]] = []
    for row in selected:
        candidate = candidate_rows.get(row["sample_id"])
        if candidate is None:
            raise CleanPoolError(f"missing original candidate: {row['sample_id']}")
        candidate_content_hash = object_sha256(candidate["messages"])
        step6_file = f"{row['skill']}-deduped.jsonl"
        checks = {
            "source_matches": candidate["source"] == row["source"],
            "revision_matches": candidate["revision"] == row["revision"],
            "parent_id_matches": candidate["parent_id"] == row["parent_id"],
            "content_hash_rebuilt": candidate_content_hash == row["content_hash"],
            "transform_chain_extends_candidate": row["transform_chain"][: len(candidate["transform_chain"])]
            == candidate["transform_chain"],
            "positive_supervision": int(row["supervised_token_count"]) > 0,
            "step6_file_hash_matches": step6_summary["output_hashes"][step6_file]
            == file_sha256(step6_dir / step6_file),
        }
        if not all(checks.values()):
            raise CleanPoolError(f"rebuild evidence failed: {row['sample_id']}")
        evidence.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "revision": row["revision"],
                "parent_id": row["parent_id"],
                "candidate_artifact": relative_path(
                    step6_dir.parent / f"{row['skill']}-candidates.jsonl"
                ),
                "candidate_artifact_sha256": candidate_hashes[row["skill"]],
                "step6_artifact": relative_path(step6_dir / step6_file),
                "step6_artifact_sha256": step6_summary["output_hashes"][step6_file],
                "transform_chain": row["transform_chain"],
                "content_hash": row["content_hash"],
                "record_sha256": object_sha256(row),
                "checks": checks,
            }
        )
    return evidence


def validate_manifest(
    manifest: dict[str, Any],
    step6_summary: dict[str, Any],
) -> dict[str, bool]:
    rows = manifest["records"]
    ids = [row["sample_id"] for row in rows]
    content_hashes = [row["content_hash"] for row in rows]
    removed = set(step6_summary["dedup_accounting"]["removed_sample_ids"])
    stored_manifest_hash = manifest["header"]["manifest_hash"]
    header_without_hash = {
        key: value
        for key, value in manifest["header"].items()
        if key != "manifest_hash"
    }
    return {
        "record_order_passed": ids == sorted(ids),
        "sample_ids_unique": len(ids) == len(set(ids)),
        "content_hashes_valid": all(
            row["content_hash"] == object_sha256(row["messages"]) for row in rows
        ),
        "positive_supervision": all(
            int(row["supervised_token_count"]) > 0 for row in rows
        ),
        "confirmed_removals_absent": not (set(ids) & removed),
        "clean_pool_hash_valid": manifest["header"]["clean_pool_hash"]
        == object_sha256(sorted(rows, key=lambda item: item["sample_id"])),
        "manifest_hash_valid": stored_manifest_hash
        == object_sha256({"header": header_without_hash, "records": rows}),
        "totals_valid": manifest["header"]["totals"] == pool_totals(rows),
        "required_fields_present": all(
            REQUIRED_RECORD_FIELDS <= set(row) for row in rows
        ),
        "content_hashes_recorded": len(content_hashes) == len(rows),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def render_report_section(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    lines = [
        REPORT_START,
        "",
        "## Step 10 freezes one clean parent pool with 7,860 canonical records",
        "",
        "Confirmed quality, exact-dedup, near-review, and contamination decisions "
        "have been reconciled into one parent pool shared by future Mix A and Mix B. "
        "No new removal occurs after Step 6, so accounting remains 7,864 accepted "
        "records minus 4 exact complete-record duplicates equals 7,860 clean records.",
        "",
        "| Slice | Examples | Raw tokens | Input tokens | Supervised tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for skill, values in summary["distribution_by_skill"].items():
        lines.append(
            f"| {skill} | {values['examples']:,} | {values['raw_tokens']:,} | "
            f"{values['input_tokens']:,} | {values['supervised_tokens']:,} |"
        )
    lines.extend(
        [
            f"| **Total** | **{totals['examples']:,}** | "
            f"**{totals['raw_tokens']:,}** | **{totals['input_tokens']:,}** | "
            f"**{totals['supervised_tokens']:,}** |",
            "",
            "### Freeze and rebuild evidence",
            "",
            f"The order-independent clean-pool hash is `{summary['clean_pool_hash']}`. "
            f"The manifest hash is `{summary['manifest_hash']}`. Five deterministic "
            "rebuild samples passed source/revision/parent, content-hash, transform-chain, "
            "positive-supervision, and Step 6 artifact checks.",
            "",
            "- Manifest: `artifacts/data/day09-dataset-manifest.json`",
            "- Gate B ledger: `artifacts/reports/day09-gate-b-review.jsonl`",
            "- Rebuild evidence: `artifacts/reports/day09-manifest-rebuild-evidence.jsonl`",
            "- Summary: `tmp/day09-work/step10-clean-pool/clean-pool-summary.json`",
            "",
            REPORT_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise CleanPoolError("Step 10 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--step6-dir", type=Path, default=DEFAULT_STEP6)
    parser.add_argument("--step7-dir", type=Path, default=DEFAULT_STEP7)
    parser.add_argument("--step9-dir", type=Path, default=DEFAULT_STEP9)
    parser.add_argument("--near-review", type=Path, default=DEFAULT_NEAR_REVIEW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gate-b-ledger", type=Path, default=DEFAULT_GATE_B)
    parser.add_argument("--rebuild-evidence", type=Path, default=DEFAULT_REBUILD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    step6_summary, rows = load_step6(args.step6_dir)
    step7_summary, near_candidates = load_step7(args.step7_dir)
    step9_summary, overlap_candidates = load_step9(args.step9_dir)
    review_rows = load_jsonl(args.near_review)

    gate_b_ledger, gate_b_summary = build_gate_b_ledger(
        near_candidates, review_rows, overlap_candidates, config
    )
    write_jsonl(args.gate_b_ledger, gate_b_ledger)
    gate_b_summary.update(
        {
            "ledger_sha256": file_sha256(args.gate_b_ledger),
            "step7_summary_sha256": file_sha256(
                args.step7_dir / "near-duplicate-summary.json"
            ),
            "step9_summary_sha256": file_sha256(
                args.step9_dir / "decontamination-summary.json"
            ),
        }
    )
    gate_b_summary_path = args.gate_b_ledger.with_suffix(".summary.json")
    with gate_b_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(gate_b_summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = manifest_payload(
        config,
        args.step6_dir,
        args.step7_dir,
        args.step9_dir,
        step6_summary,
        step7_summary,
        step9_summary,
        args.gate_b_ledger,
        gate_b_summary_path,
        rows,
    )
    validation = validate_manifest(manifest, step6_summary)
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise CleanPoolError(f"manifest validation failed: {failed}")
    write_manifest(args.manifest, manifest)

    candidate_rows, candidate_hashes = load_candidate_rows(args.work_dir)
    rebuild_evidence = build_rebuild_evidence(
        config,
        manifest["records"],
        candidate_rows,
        candidate_hashes,
        step6_summary,
        args.step6_dir,
    )
    write_jsonl(args.rebuild_evidence, rebuild_evidence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete_frozen",
        "clean_pool_version": config["clean_pool"]["version"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "clean_pool_dependency_config_sha256": object_sha256(
            clean_pool_dependency_config(config)
        ),
        "manifest_path": relative_path(args.manifest),
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_hash": manifest["header"]["manifest_hash"],
        "clean_pool_hash": manifest["header"]["clean_pool_hash"],
        "totals": manifest["header"]["totals"],
        "distribution_by_skill": manifest["header"]["distribution_by_skill"],
        "removal_accounting": manifest["header"]["removal_accounting"],
        "gate_b": gate_b_summary,
        "gate_b_ledger_path": relative_path(args.gate_b_ledger),
        "gate_b_ledger_sha256": file_sha256(args.gate_b_ledger),
        "gate_b_summary_sha256": file_sha256(gate_b_summary_path),
        "rebuild_evidence_path": relative_path(args.rebuild_evidence),
        "rebuild_evidence_sha256": file_sha256(args.rebuild_evidence),
        "rebuild_sample_count": len(rebuild_evidence),
        "validation_checks": validation,
        "downstream_eligible": True,
    }
    summary_path = args.output_dir / "clean-pool-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    update_report(args.report, summary)
    print(
        f"clean_examples={summary['totals']['examples']} "
        f"supervised_tokens={summary['totals']['supervised_tokens']} "
        f"manifest_hash={summary['manifest_hash']} downstream_eligible=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
