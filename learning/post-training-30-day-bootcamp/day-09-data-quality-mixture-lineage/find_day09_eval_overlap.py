#!/usr/bin/env python3
"""Generate auditable Day 09 train/eval contamination candidates."""

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
DEFAULT_TRAIN = DEFAULT_WORK / "step6-exact-dedup"
DEFAULT_EVAL = DEFAULT_WORK / "step8-eval-candidates"
DEFAULT_OUTPUT = DEFAULT_WORK / "step9-decontamination"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP9_DECONTAMINATION_START -->"
REPORT_END = "<!-- STEP9_DECONTAMINATION_END -->"
EXPECTED_BOUNDARIES = {
    "train_prompt_vs_eval_prompt": ("prompt", "prompt"),
    "train_answer_vs_eval_reference": ("answer", "reference"),
    "train_complete_record_vs_eval_record": ("complete_record", "complete_record"),
}


class DecontaminationError(ValueError):
    """A Step 9 input, config, or overlap invariant failed."""


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


def normalized_train_value(row: dict[str, Any], field: str) -> str:
    if field == "prompt":
        return normalize_text(
            "\n".join(
                item["content"] for item in row["messages"] if item["role"] == "user"
            )
        )
    if field == "answer":
        return normalize_text(
            "\n".join(
                item["content"]
                for item in row["messages"]
                if item["role"] == "assistant"
            )
        )
    if field == "complete_record":
        return json.dumps(
            [
                [item["role"], normalize_text(item["content"])]
                for item in row["messages"]
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise DecontaminationError(f"unsupported train field: {field}")


def normalized_eval_value(row: dict[str, Any], field: str) -> str:
    if field == "prompt":
        return normalize_text(row["prompt"])
    if field == "reference":
        return normalize_text(row["reference"])
    if field == "complete_record":
        return json.dumps(
            [
                ["user", normalize_text(row["prompt"])],
                ["assistant", normalize_text(row["reference"])],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise DecontaminationError(f"unsupported eval field: {field}")


def train_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
        "input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "supervised_tokens": sum(
            int(row["supervised_token_count"]) for row in rows
        ),
    }


def load_train_pool(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = input_dir / "exact-dedup-summary.json"
    if not summary_path.is_file():
        raise DecontaminationError(f"missing Step 6 summary: {summary_path}")
    summary = load_json(summary_path)
    accounting = summary.get("dedup_accounting", {})
    if summary.get("status") != "complete" or not summary.get(
        "downstream_eligible"
    ):
        raise DecontaminationError("Step 6 train pool is not downstream eligible")
    if not all(
        accounting.get(name)
        for name in (
            "accounting_identity_passed",
            "survivor_integrity_passed",
            "terminal_uniqueness_passed",
        )
    ):
        raise DecontaminationError("Step 6 accounting checks are incomplete")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, expected_hash in sorted(summary["output_hashes"].items()):
        if not name.endswith("-deduped.jsonl"):
            continue
        path = input_dir / name
        if file_sha256(path) != expected_hash:
            raise DecontaminationError(f"Step 6 output hash mismatch: {name}")
        for row in load_jsonl(path):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise DecontaminationError(f"duplicate train sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    if train_totals(rows) != accounting["after"]:
        raise DecontaminationError("Step 6 train totals do not match its summary")
    return summary, rows


def load_eval_pool(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = input_dir / "eval-candidates-summary.json"
    if not summary_path.is_file():
        raise DecontaminationError(f"missing Step 8 summary: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("status") != "complete_candidates_only":
        raise DecontaminationError("Step 8 eval candidate pool is not complete")
    if not all(summary.get("validation_checks", {}).values()):
        raise DecontaminationError("Step 8 validation checks are incomplete")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, expected_hash in sorted(summary["output_hashes"].items()):
        path = input_dir / name
        if file_sha256(path) != expected_hash:
            raise DecontaminationError(f"Step 8 output hash mismatch: {name}")
        for row in load_jsonl(path):
            eval_id = row["eval_sample_id"]
            if eval_id in seen:
                raise DecontaminationError(f"duplicate eval_sample_id: {eval_id}")
            seen.add(eval_id)
            rows.append(row)
    if len(rows) != int(summary["total_candidates"]):
        raise DecontaminationError("Step 8 row count does not match its summary")
    return summary, rows


def candidate_id(
    version: str,
    match_type: str,
    boundary: str,
    train_sample_id: str,
    eval_sample_id: str,
) -> str:
    payload = "\0".join(
        (version, match_type, boundary, train_sample_id, eval_sample_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidate(
    *,
    version: str,
    match_type: str,
    boundary_config: dict[str, Any],
    matcher_config_sha256: str,
    train_row: dict[str, Any],
    eval_row: dict[str, Any],
    train_value: str,
    eval_value: str,
    similarity_score: float,
) -> dict[str, Any]:
    train_id = train_row["sample_id"]
    eval_id = eval_row["eval_sample_id"]
    same_skill = train_row["skill"] == eval_row["skill"]
    return {
        "candidate_id": candidate_id(
            version,
            match_type,
            boundary_config["name"],
            train_id,
            eval_id,
        ),
        "train_sample_id": train_id,
        "eval_sample_id": eval_id,
        "train_source": train_row["source"],
        "eval_source": eval_row["source"],
        "train_skill": train_row["skill"],
        "eval_skill": eval_row["skill"],
        "skill_scope": "same_skill" if same_skill else "cross_skill",
        "matched_field": boundary_config["matched_field"],
        "boundary": boundary_config["name"],
        "match_type": match_type,
        "similarity_score": similarity_score,
        "matcher": (
            "normalized_exact_equality"
            if match_type == "exact"
            else "rapidfuzz.fuzz.ratio"
        ),
        "matcher_version": (
            "exact_normalization_v1"
            if match_type == "exact"
            else rapidfuzz.__version__
        ),
        "matcher_config_sha256": matcher_config_sha256,
        "near_threshold": boundary_config["near_threshold"],
        "near_minimum_characters": boundary_config[
            "near_minimum_characters"
        ],
        "normalized_characters_train": len(train_value),
        "normalized_characters_eval": len(eval_value),
        "normalized_sha256_train": text_sha256(train_value),
        "normalized_sha256_eval": text_sha256(eval_value),
        "train_content_hash": train_row["content_hash"],
        "eval_content_hash": eval_row["content_hash"],
        "decision": None,
        "decision_reason": None,
        "reviewer": None,
        "reviewed_at": None,
    }


def generate_boundary_candidates(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    version: str,
    boundary_config: dict[str, Any],
    matcher_config_sha256: str,
    batch_size: int,
    workers: int,
    score_decimals: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_values = [
        normalized_train_value(row, boundary_config["train_field"])
        for row in train_rows
    ]
    eval_values = [
        normalized_eval_value(row, boundary_config["eval_field"])
        for row in eval_rows
    ]

    eval_exact: dict[str, list[int]] = defaultdict(list)
    for eval_index, value in enumerate(eval_values):
        eval_exact[value].append(eval_index)
    exact_candidates: list[dict[str, Any]] = []
    for train_index, value in enumerate(train_values):
        for eval_index in eval_exact.get(value, []):
            exact_candidates.append(
                build_candidate(
                    version=version,
                    match_type="exact",
                    boundary_config=boundary_config,
                    matcher_config_sha256=matcher_config_sha256,
                    train_row=train_rows[train_index],
                    eval_row=eval_rows[eval_index],
                    train_value=value,
                    eval_value=eval_values[eval_index],
                    similarity_score=100.0,
                )
            )

    minimum = int(boundary_config["near_minimum_characters"])
    threshold = float(boundary_config["near_threshold"])
    eligible_train = [
        (index, value)
        for index, value in enumerate(train_values)
        if len(value) >= minimum
    ]
    eligible_eval = [
        (index, value)
        for index, value in enumerate(eval_values)
        if len(value) >= minimum
    ]
    near_candidates: list[dict[str, Any]] = []
    near_eval_values = [value for _, value in eligible_eval]
    maximum_score: float | None = None
    maximum_pair: tuple[str, str] | None = None
    for start in range(0, len(eligible_train), batch_size):
        train_batch = eligible_train[start : start + batch_size]
        scores = process.cdist(
            [value for _, value in train_batch],
            near_eval_values,
            scorer=fuzz.ratio,
            score_hint=threshold,
            workers=workers,
        )
        if scores.size:
            flat_index = int(scores.argmax())
            local_train_max, local_eval_max = divmod(
                flat_index, scores.shape[1]
            )
            score_max = round(
                float(scores[local_train_max, local_eval_max]), score_decimals
            )
            train_index_max, _ = train_batch[local_train_max]
            eval_index_max, _ = eligible_eval[local_eval_max]
            pair_max = (
                train_rows[train_index_max]["sample_id"],
                eval_rows[eval_index_max]["eval_sample_id"],
            )
            if (
                maximum_score is None
                or score_max > maximum_score
                or (score_max == maximum_score and pair_max < maximum_pair)
            ):
                maximum_score = score_max
                maximum_pair = pair_max
        for local_train, local_eval in zip(*(scores >= threshold).nonzero()):
            train_index, train_value = train_batch[int(local_train)]
            eval_index, eval_value = eligible_eval[int(local_eval)]
            if train_value == eval_value:
                continue
            score = round(float(scores[local_train, local_eval]), score_decimals)
            if score < threshold:
                continue
            near_candidates.append(
                build_candidate(
                    version=version,
                    match_type="near",
                    boundary_config=boundary_config,
                    matcher_config_sha256=matcher_config_sha256,
                    train_row=train_rows[train_index],
                    eval_row=eval_rows[eval_index],
                    train_value=train_value,
                    eval_value=eval_value,
                    similarity_score=score,
                )
            )

    candidates = exact_candidates + near_candidates
    candidates.sort(
        key=lambda item: (
            item["match_type"],
            item["train_sample_id"],
            item["eval_sample_id"],
        )
    )
    return candidates, {
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "all_cross_pairs_checked_for_exact": len(train_rows) * len(eval_rows),
        "near_eligible_train_examples": len(eligible_train),
        "near_excluded_short_train_examples": len(train_rows) - len(eligible_train),
        "near_eligible_eval_examples": len(eligible_eval),
        "near_excluded_short_eval_examples": len(eval_rows) - len(eligible_eval),
        "near_cross_pairs_scored": len(eligible_train) * len(eligible_eval),
        "maximum_near_eligible_similarity_score": maximum_score,
        "maximum_near_eligible_train_sample_id": (
            maximum_pair[0] if maximum_pair is not None else None
        ),
        "maximum_near_eligible_eval_sample_id": (
            maximum_pair[1] if maximum_pair is not None else None
        ),
        "exact_candidates": len(exact_candidates),
        "near_candidates": len(near_candidates),
    }


def generate_candidates(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    decontamination = config["decontamination"]
    matcher_config_sha256 = object_sha256(decontamination)
    candidates: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {}
    for boundary in decontamination["boundaries"]:
        print(f"scanning boundary={boundary['name']}", flush=True)
        boundary_candidates, boundary_coverage = generate_boundary_candidates(
            train_rows,
            eval_rows,
            version=decontamination["version"],
            boundary_config=boundary,
            matcher_config_sha256=matcher_config_sha256,
            batch_size=int(decontamination["batch_size"]),
            workers=int(decontamination["workers"]),
            score_decimals=int(decontamination["score_decimals"]),
        )
        candidates.extend(boundary_candidates)
        coverage[boundary["name"]] = boundary_coverage
        print(
            f"boundary={boundary['name']} "
            f"exact={boundary_coverage['exact_candidates']} "
            f"near={boundary_coverage['near_candidates']}",
            flush=True,
        )
    candidates.sort(
        key=lambda item: (
            item["match_type"],
            item["boundary"],
            item["train_sample_id"],
            item["eval_sample_id"],
        )
    )
    return candidates, coverage


def candidate_statistics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_boundary: dict[str, dict[str, int]] = {}
    for boundary in EXPECTED_BOUNDARIES:
        rows = [item for item in candidates if item["boundary"] == boundary]
        by_boundary[boundary] = {
            "exact": sum(item["match_type"] == "exact" for item in rows),
            "near": sum(item["match_type"] == "near" for item in rows),
            "total": len(rows),
        }
    skill_pairs = Counter(
        f"{item['train_skill']} -> {item['eval_skill']}" for item in candidates
    )
    return {
        "candidate_records": len(candidates),
        "exact_candidates": sum(
            item["match_type"] == "exact" for item in candidates
        ),
        "near_candidates": sum(
            item["match_type"] == "near" for item in candidates
        ),
        "affected_train_samples": len(
            {item["train_sample_id"] for item in candidates}
        ),
        "affected_eval_samples": len(
            {item["eval_sample_id"] for item in candidates}
        ),
        "same_skill_candidates": sum(
            item["skill_scope"] == "same_skill" for item in candidates
        ),
        "cross_skill_candidates": sum(
            item["skill_scope"] == "cross_skill" for item in candidates
        ),
        "by_boundary": by_boundary,
        "by_skill_pair": dict(sorted(skill_pairs.items())),
        "confirmed_contamination": 0,
        "pending_review": len(candidates),
    }


def validate_candidates(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, bool]:
    boundaries = {
        item["name"]: item for item in config["decontamination"]["boundaries"]
    }
    ids = [item["candidate_id"] for item in candidates]
    return {
        "candidate_ids_unique": len(ids) == len(set(ids)),
        "boundaries_valid": all(
            item["boundary"] in EXPECTED_BOUNDARIES for item in candidates
        ),
        "exact_scores_valid": all(
            item["similarity_score"] == 100.0
            for item in candidates
            if item["match_type"] == "exact"
        ),
        "near_scores_valid": all(
            item["similarity_score"]
            >= float(boundaries[item["boundary"]]["near_threshold"])
            and item["similarity_score"] < 100.0
            for item in candidates
            if item["match_type"] == "near"
        ),
        "near_lengths_valid": all(
            item["normalized_characters_train"]
            >= int(boundaries[item["boundary"]]["near_minimum_characters"])
            and item["normalized_characters_eval"]
            >= int(boundaries[item["boundary"]]["near_minimum_characters"])
            for item in candidates
            if item["match_type"] == "near"
        ),
        "exact_and_near_disjoint": len(
            {
                (item["boundary"], item["train_sample_id"], item["eval_sample_id"])
                for item in candidates
            }
        )
        == len(candidates),
        "all_decisions_pending": all(
            item["decision"] is None for item in candidates
        ),
        "no_automatic_removal": True,
    }


def build_summary(
    config: dict[str, Any],
    config_path: Path,
    train_dir: Path,
    eval_dir: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    coverage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decontamination = config["decontamination"]
    statistics = candidate_statistics(candidates)
    validation = validate_candidates(candidates, config)
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise DecontaminationError(f"candidate validation failed: {failed}")
    return {
        "status": "complete_candidates_only",
        "candidate_version": decontamination["version"],
        "policy": decontamination["policy"],
        "normalization": {
            "name": config["normalization"]["name"],
            "config_sha256": object_sha256(config["normalization"]),
        },
        "matcher": {
            "name": decontamination["matcher"],
            "version": rapidfuzz.__version__,
            "near_excludes_exact": decontamination["near_excludes_exact"],
        },
        "config_sha256": file_sha256(config_path),
        "decontamination_config_sha256": object_sha256(decontamination),
        "implementation_sha256": file_sha256(Path(__file__)),
        "inputs": {
            "train_cohort": decontamination["train_cohort"],
            "train_examples": len(train_rows),
            "train_summary_sha256": file_sha256(
                train_dir / "exact-dedup-summary.json"
            ),
            "eval_cohort": decontamination["eval_cohort"],
            "eval_examples": len(eval_rows),
            "eval_summary_sha256": file_sha256(
                eval_dir / "eval-candidates-summary.json"
            ),
        },
        "coverage": coverage,
        "candidate_statistics": statistics,
        "validation_checks": validation,
        "training_pool_changed": False,
        "eval_pool_changed": False,
        "gate_b_complete": False,
    }


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "train-eval-overlap-candidates.jsonl"
    with candidate_path.open("w", encoding="utf-8") as handle:
        for item in candidates:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary["output_hashes"] = {
        candidate_path.name: file_sha256(candidate_path)
    }
    with (output_dir / "decontamination-summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def render_report_section(summary: dict[str, Any]) -> str:
    stats = summary["candidate_statistics"]
    lines = [
        REPORT_START,
        "",
        "## Step 9 scans all train/eval boundaries without automatic deletion",
        "",
        f"The scan compares {summary['inputs']['train_examples']:,} Step 6 train "
        f"records with {summary['inputs']['eval_examples']:,} Step 8 eval candidates. "
        f"It emitted {stats['candidate_records']:,} overlap candidates: "
        f"{stats['exact_candidates']:,} exact and {stats['near_candidates']:,} near. "
        "Candidate status is not a contamination decision.",
        "",
        "| Boundary | Exact | Near | Total | Near pairs scored | Max eligible score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for boundary in EXPECTED_BOUNDARIES:
        counts = stats["by_boundary"][boundary]
        coverage = summary["coverage"][boundary]
        lines.append(
            f"| {boundary} | {counts['exact']:,} | {counts['near']:,} | "
            f"{counts['total']:,} | {coverage['near_cross_pairs_scored']:,} | "
            f"{coverage['maximum_near_eligible_similarity_score']} |"
        )
    lines.extend(
        [
            "",
            "### Evidence contract",
            "",
            "Exact equality uses the frozen NFKC/lowercase/whitespace "
            "normalization without a length threshold. Near matching uses "
            f"`{summary['matcher']['name']}` version "
            f"`{summary['matcher']['version']}`; exact pairs are excluded from near "
            "counts. Prompt, answer/reference, and complete-record boundaries remain "
            "separate in both the ledger and summary.",
            "",
            "No train or eval record was removed. Gate B must review every emitted "
            "candidate before a contamination decision can change either pool.",
            "",
            "- Candidates: `tmp/day09-work/step9-decontamination/train-eval-overlap-candidates.jsonl`",
            "- Summary: `tmp/day09-work/step9-decontamination/decontamination-summary.json`",
            "",
            REPORT_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    if not report_path.is_file():
        raise DecontaminationError(f"missing audit report: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise DecontaminationError("Step 9 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary), encoding="utf-8"
    )


def validate_config(config: dict[str, Any]) -> None:
    decontamination = config["decontamination"]
    if decontamination["matcher"] != "rapidfuzz.fuzz.ratio":
        raise DecontaminationError("unsupported near matcher")
    if decontamination["matcher_version"] != rapidfuzz.__version__:
        raise DecontaminationError("RapidFuzz version does not match config")
    if not decontamination["near_excludes_exact"]:
        raise DecontaminationError("exact candidates must be excluded from near")
    if decontamination["policy"] != (
        "candidate generation only; no automatic train or eval deletion"
    ):
        raise DecontaminationError("Step 9 policy must not delete records")
    boundaries = {item["name"]: item for item in decontamination["boundaries"]}
    if set(boundaries) != set(EXPECTED_BOUNDARIES):
        raise DecontaminationError("Step 9 boundaries do not match assignment")
    for name, (train_field, eval_field) in EXPECTED_BOUNDARIES.items():
        boundary = boundaries[name]
        if (boundary["train_field"], boundary["eval_field"]) != (
            train_field,
            eval_field,
        ):
            raise DecontaminationError(f"field mismatch for boundary: {name}")
        threshold = float(boundary["near_threshold"])
        if threshold <= 0 or threshold >= 100:
            raise DecontaminationError("near threshold must be in (0, 100)")
        if int(boundary["near_minimum_characters"]) <= 0:
            raise DecontaminationError("near minimum characters must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    validate_config(config)
    _, train_rows = load_train_pool(args.train_dir)
    _, eval_rows = load_eval_pool(args.eval_dir)
    candidates, coverage = generate_candidates(train_rows, eval_rows, config)
    summary = build_summary(
        config,
        args.config,
        args.train_dir,
        args.eval_dir,
        train_rows,
        eval_rows,
        candidates,
        coverage,
    )
    write_outputs(args.output_dir, summary, candidates)
    update_report(args.report, summary)
    print(
        f"overlap_candidates={len(candidates)} "
        f"exact={summary['candidate_statistics']['exact_candidates']} "
        f"near={summary['candidate_statistics']['near_candidates']} "
        "pool_changed=false gate_b_complete=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
