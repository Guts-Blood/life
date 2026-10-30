#!/usr/bin/env python3
"""Generate auditable Day 09 near-duplicate candidates without deleting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import rapidfuzz
from rapidfuzz import fuzz, process


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_WORK = HERE.parents[2] / "tmp" / "day09-work"
DEFAULT_INPUT = DEFAULT_WORK / "step6-exact-dedup"
DEFAULT_OUTPUT = DEFAULT_WORK / "step7-near-candidates"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP7_NEAR_CANDIDATES_START -->"
REPORT_END = "<!-- STEP7_NEAR_CANDIDATES_END -->"
CODE = chr(96)
EXPECTED_FIELDS = {
    "normalized_prompt",
    "normalized_answer",
    "normalized_complete_record",
}


class NearDuplicateError(ValueError):
    """A near-duplicate input, configuration, or evidence invariant failed."""


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


def normalized_character_count(row: dict[str, Any], field: str) -> int:
    if field == "normalized_complete_record":
        return sum(
            len(normalize_text(message["content"])) for message in row["messages"]
        )
    return len(normalized_fields(row)[field])


def row_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
        "input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "supervised_tokens": sum(
            int(row["supervised_token_count"]) for row in rows
        ),
    }


def load_step6(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = input_dir / "exact-dedup-summary.json"
    if not summary_path.is_file():
        raise NearDuplicateError(f"missing Step 6 summary: {summary_path}")
    summary = load_json(summary_path)
    accounting = summary.get("dedup_accounting", {})
    if summary.get("status") != "complete" or not summary.get(
        "downstream_eligible"
    ):
        raise NearDuplicateError("Step 6 pool is not downstream eligible")
    if not all(
        accounting.get(check)
        for check in (
            "accounting_identity_passed",
            "survivor_integrity_passed",
            "terminal_uniqueness_passed",
        )
    ):
        raise NearDuplicateError("Step 6 validation checks are incomplete")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    pool_files = sorted(
        name
        for name in summary["output_hashes"]
        if name.endswith("-deduped.jsonl")
    )
    if not pool_files:
        raise NearDuplicateError("Step 6 summary has no deduped pool files")
    for name in pool_files:
        path = input_dir / name
        if file_sha256(path) != summary["output_hashes"][name]:
            raise NearDuplicateError(f"Step 6 output hash mismatch: {name}")
        for row in load_jsonl(path):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise NearDuplicateError(f"duplicate Step 6 sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    if row_totals(rows) != accounting["after"]:
        raise NearDuplicateError("Step 6 row/token totals do not match its summary")
    return summary, rows


def candidate_id(version: str, field: str, sample_a: str, sample_b: str) -> str:
    payload = f"{version}\0{field}\0{sample_a}\0{sample_b}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_field_candidates(
    rows: list[dict[str, Any]],
    *,
    field: str,
    candidate_version: str,
    matcher_config_sha256: str,
    threshold: float,
    minimum_characters: int,
    batch_size: int,
    workers: int,
    score_decimals: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[tuple[dict[str, Any], str, int]] = []
    excluded_too_short = 0
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        value = normalized_fields(row)[field]
        character_count = normalized_character_count(row, field)
        if character_count < minimum_characters:
            excluded_too_short += 1
            continue
        eligible.append((row, value, character_count))

    candidates: list[dict[str, Any]] = []
    values = [value for _, value, _ in eligible]
    for start in range(0, len(eligible), batch_size):
        end = min(start + batch_size, len(eligible))
        choice_start = start + 1
        if choice_start >= len(eligible):
            break
        scores = process.cdist(
            values[start:end],
            values[choice_start:],
            scorer=fuzz.ratio,
            score_cutoff=threshold,
            score_hint=threshold,
            workers=workers,
        )
        for local_query, global_query in enumerate(range(start, end)):
            first_valid_column = global_query + 1 - choice_start
            row_scores = scores[local_query, first_valid_column:]
            for relative_column in row_scores.nonzero()[0].tolist():
                global_choice = global_query + 1 + relative_column
                score = round(float(row_scores[relative_column]), score_decimals)
                if score < threshold:
                    continue
                row_a, _, characters_a = eligible[global_query]
                row_b, _, characters_b = eligible[global_choice]
                sample_a = row_a["sample_id"]
                sample_b = row_b["sample_id"]
                sources = {row_a["source"], row_b["source"]}
                candidates.append(
                    {
                        "candidate_id": candidate_id(
                            candidate_version, field, sample_a, sample_b
                        ),
                        "sample_id_a": sample_a,
                        "sample_id_b": sample_b,
                        "source_a": row_a["source"],
                        "source_b": row_b["source"],
                        "skill_a": row_a["skill"],
                        "skill_b": row_b["skill"],
                        "matched_field": field,
                        "matcher": "rapidfuzz.fuzz.ratio",
                        "matcher_version": rapidfuzz.__version__,
                        "matcher_config_sha256": matcher_config_sha256,
                        "similarity_score": score,
                        "threshold": threshold,
                        "minimum_characters": minimum_characters,
                        "normalized_characters_a": characters_a,
                        "normalized_characters_b": characters_b,
                        "candidate_scope": (
                            "cross_source" if len(sources) > 1 else "within_source"
                        ),
                        "combined_supervised_tokens": int(
                            row_a["supervised_token_count"]
                        )
                        + int(row_b["supervised_token_count"]),
                        "review_result": None,
                        "reason_code": None,
                        "review_note": None,
                    }
                )
    candidates.sort(
        key=lambda item: (
            item["matched_field"],
            item["sample_id_a"],
            item["sample_id_b"],
        )
    )
    return candidates, {
        "input_examples": len(rows),
        "eligible_examples": len(eligible),
        "excluded_too_short": excluded_too_short,
        "unordered_pairs_scored": len(eligible) * (len(eligible) - 1) // 2,
    }


def generate_candidates(
    rows: list[dict[str, Any]], near_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    config_sha256 = object_sha256(near_config)
    candidates: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = {}
    for field in near_config["fields"]:
        print(f"scoring field={field}", flush=True)
        field_candidates, field_coverage = generate_field_candidates(
            rows,
            field=field,
            candidate_version=near_config["candidate_version"],
            matcher_config_sha256=config_sha256,
            threshold=float(near_config["threshold"]),
            minimum_characters=int(near_config["minimum_characters"]),
            batch_size=int(near_config["batch_size"]),
            workers=int(near_config["workers"]),
            score_decimals=int(near_config["score_decimals"]),
        )
        candidates.extend(field_candidates)
        coverage[field] = field_coverage
        print(
            f"field={field} candidates={len(field_candidates)} "
            f"pairs={field_coverage['unordered_pairs_scored']}",
            flush=True,
        )
    candidates.sort(
        key=lambda item: (
            item["matched_field"],
            item["sample_id_a"],
            item["sample_id_b"],
        )
    )
    return candidates, coverage


def score_bucket(score: float) -> str:
    if score == 100.0:
        return "100"
    if score >= 95.0:
        return "95-<100"
    return "90-<95"


def candidate_statistics(
    candidates: list[dict[str, Any]], fields: list[str]
) -> dict[str, Any]:
    by_field: dict[str, dict[str, Any]] = {}
    for field in fields:
        items = [item for item in candidates if item["matched_field"] == field]
        affected = {
            sample_id
            for item in items
            for sample_id in (item["sample_id_a"], item["sample_id_b"])
        }
        buckets = Counter(score_bucket(item["similarity_score"]) for item in items)
        by_field[field] = {
            "candidate_count": len(items),
            "affected_unique_samples": len(affected),
            "within_source_count": sum(
                item["candidate_scope"] == "within_source" for item in items
            ),
            "cross_source_count": sum(
                item["candidate_scope"] == "cross_source" for item in items
            ),
            "minimum_score": min(
                (item["similarity_score"] for item in items), default=None
            ),
            "maximum_score": max(
                (item["similarity_score"] for item in items), default=None
            ),
            "score_buckets": {
                name: buckets.get(name, 0)
                for name in ("90-<95", "95-<100", "100")
            },
        }
    unique_pairs = {
        (item["sample_id_a"], item["sample_id_b"]) for item in candidates
    }
    affected_samples = {
        sample_id
        for item in candidates
        for sample_id in (item["sample_id_a"], item["sample_id_b"])
    }
    skill_pairs = Counter(
        " / ".join(sorted((item["skill_a"], item["skill_b"])))
        for item in candidates
    )
    return {
        "candidate_records": len(candidates),
        "unique_sample_pairs": len(unique_pairs),
        "affected_unique_samples": len(affected_samples),
        "confirmed_removal_count": 0,
        "by_field": by_field,
        "by_skill_pair": dict(sorted(skill_pairs.items())),
    }


def validate_candidates(
    candidates: list[dict[str, Any]], near_config: dict[str, Any]
) -> dict[str, bool]:
    ids = [item["candidate_id"] for item in candidates]
    threshold = float(near_config["threshold"])
    minimum = int(near_config["minimum_characters"])
    return {
        "candidate_ids_unique": len(ids) == len(set(ids)),
        "pair_order_passed": all(
            item["sample_id_a"] < item["sample_id_b"] for item in candidates
        ),
        "threshold_passed": all(
            item["similarity_score"] >= threshold for item in candidates
        ),
        "minimum_characters_passed": all(
            item["normalized_characters_a"] >= minimum
            and item["normalized_characters_b"] >= minimum
            for item in candidates
        ),
        "no_confirmed_removals": all(
            item["review_result"] is None for item in candidates
        ),
    }


def build_summary(
    config: dict[str, Any],
    input_dir: Path,
    step6_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    coverage: dict[str, dict[str, int]],
) -> dict[str, Any]:
    near_config = config["near_duplicate"]
    totals = row_totals(rows)
    return {
        "candidate_version": near_config["candidate_version"],
        "status": "complete_candidates_only",
        "training_pool_changed": False,
        "downstream_examples": totals["examples"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "near_duplicate_config_sha256": object_sha256(near_config),
        "normalization": step6_summary["normalization"],
        "matcher": {
            "name": near_config["matcher"],
            "version": rapidfuzz.__version__,
            "threshold": near_config["threshold"],
            "minimum_characters": near_config["minimum_characters"],
            "policy": near_config["policy"],
        },
        "input": {
            "cohort": "Step 6 exact-deduped pool",
            "step6_summary_sha256": file_sha256(
                input_dir / "exact-dedup-summary.json"
            ),
            "totals": totals,
        },
        "pair_coverage": coverage,
        "candidate_statistics": candidate_statistics(
            candidates, near_config["fields"]
        ),
        "validation_checks": validate_candidates(candidates, near_config),
        "confirmed_decisions": {
            "accept": 0,
            "reject": 0,
            "uncertain": 0,
            "pending": len(candidates),
        },
    }


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "near-duplicate-candidates.jsonl"
    with candidate_path.open("w", encoding="utf-8") as handle:
        for item in candidates:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary["output_hashes"] = {
        candidate_path.name: file_sha256(candidate_path),
    }
    with (output_dir / "near-duplicate-summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def render_report_section(summary: dict[str, Any]) -> str:
    stats = summary["candidate_statistics"]
    matcher = summary["matcher"]
    lines = [
        REPORT_START,
        "",
        f"## Step 7 finds {stats['candidate_records']:,} near-match records "
        "without changing the training pool",
        "",
        f"An exhaustive unordered-pair scan of the {summary['downstream_examples']:,} "
        "Step 6 records produced "
        f"{stats['candidate_records']:,} field-specific candidate records covering "
        f"{stats['unique_sample_pairs']:,} unique sample pairs. Confirmed removals "
        "remain zero: the similarity threshold creates review candidates, not "
        "deletion labels.",
        "",
        "| Field | Candidates | Affected samples | Within source | Cross source | 90-<95 | 95-<100 | 100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field, item in stats["by_field"].items():
        buckets = item["score_buckets"]
        lines.append(
            f"| {field} | {item['candidate_count']:,} | "
            f"{item['affected_unique_samples']:,} | "
            f"{item['within_source_count']:,} | "
            f"{item['cross_source_count']:,} | "
            f"{buckets['90-<95']:,} | {buckets['95-<100']:,} | "
            f"{buckets['100']:,} |"
        )
    lines.extend(
        [
            "",
            "### Matcher scope and evidence contract",
            "",
            f"The matcher is {CODE}{matcher['name']}{CODE} version "
            f"{CODE}{matcher['version']}{CODE}, with score >= "
            f"{matcher['threshold']} and both normalized values at least "
            f"{matcher['minimum_characters']} characters. Prompt, answer, and "
            "complete-record scores are emitted separately. Each candidate stores "
            "both sample IDs, sources, slices, score, threshold, matcher config hash, "
            "scope, and pending review fields.",
            "",
            "### Similarity is not a removal decision",
            "",
            "Character-level ratio can flag legitimate shared instructions, finance "
            "evidence templates, or mathematical scaffolding. It does not establish "
            "semantic duplication. The Step 6 pool therefore remains unchanged, and "
            "any future removal requires a Gate B review decision or another frozen "
            "deterministic rule.",
            "",
            f"- Candidates: {CODE}tmp/day09-work/step7-near-candidates/near-duplicate-candidates.jsonl{CODE}",
            f"- Summary: {CODE}tmp/day09-work/step7-near-candidates/near-duplicate-summary.json{CODE}",
            "",
            REPORT_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    if not report_path.is_file():
        raise NearDuplicateError(f"missing quality report: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise NearDuplicateError("Step 7 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary),
        encoding="utf-8",
    )


def validate_config(config: dict[str, Any]) -> None:
    near = config["near_duplicate"]
    if near["matcher"] != "rapidfuzz.fuzz.ratio":
        raise NearDuplicateError("unsupported near-duplicate matcher")
    if near["version"] != rapidfuzz.__version__:
        raise NearDuplicateError("RapidFuzz version does not match frozen config")
    if set(near["fields"]) != EXPECTED_FIELDS:
        raise NearDuplicateError("near-duplicate fields do not match assignment")
    if float(near["threshold"]) <= 0 or float(near["threshold"]) > 100:
        raise NearDuplicateError("near-duplicate threshold must be in (0, 100]")
    if int(near["minimum_characters"]) <= 0:
        raise NearDuplicateError("minimum_characters must be positive")
    if near["policy"] != "candidate generation only; no automatic deletion":
        raise NearDuplicateError("near-duplicate policy must not delete records")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    validate_config(config)
    step6_summary, rows = load_step6(args.input_dir)
    candidates, coverage = generate_candidates(rows, config["near_duplicate"])
    summary = build_summary(
        config, args.input_dir, step6_summary, rows, candidates, coverage
    )
    if not all(summary["validation_checks"].values()):
        raise NearDuplicateError("near-duplicate candidate validation failed")
    write_outputs(args.output_dir, summary, candidates)
    update_report(args.report, summary)
    print(
        f"candidates={len(candidates)} unique_pairs="
        f"{summary['candidate_statistics']['unique_sample_pairs']} "
        "confirmed_removals=0 pool_changed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
