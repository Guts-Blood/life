#!/usr/bin/env python3
"""Prepare deterministic Day 09 eval candidate pools for Day 10."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_SOURCE_ROOT = HERE.parents[2] / "tmp" / "day09-eval-sources"
DEFAULT_OUTPUT = HERE.parents[2] / "tmp" / "day09-work" / "step8-eval-candidates"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP8_EVAL_CANDIDATES_START -->"
REPORT_END = "<!-- STEP8_EVAL_CANDIDATES_END -->"
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class EvalCandidateError(ValueError):
    """An eval source, adapter, or candidate invariant failed."""


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_json(path: Path) -> Any:
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


def stable_selection_hash(
    version: str, source: str, revision: str, split: str, parent_id: str
) -> str:
    payload = "\0".join((version, source, revision, split, parent_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def canonical_candidate(
    *,
    version: str,
    skill: str,
    source_config: dict[str, Any],
    parent_id: str,
    prompt: str,
    reference: str,
    source_file: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = source_config["source"]
    revision = source_config["revision"]
    split = source_config["split"]
    source_slug = {
        "cais/mmlu": "mmlu",
        "openai/grade-school-math": "gsm8k",
        "openai/human-eval": "human-eval",
        "NExTplusplus/TAT-QA": "tat-qa",
    }[source]
    prompt = prompt.strip()
    reference = reference.strip()
    if not prompt or not reference:
        raise EvalCandidateError(f"empty prompt/reference for {source}:{parent_id}")
    selection_hash = stable_selection_hash(
        version, source, revision, split, parent_id
    )
    record = {
        "eval_sample_id": f"eval:{skill}:{source_slug}:{parent_id}",
        "source": source,
        "revision": revision,
        "split": split,
        "parent_id": parent_id,
        "prompt": prompt,
        "reference": reference,
        "skill": skill,
        "content_hash": object_sha256(
            {"prompt": prompt, "reference": reference}
        ),
        "selection_hash": selection_hash,
        "adapter": source_config["adapter"],
        "transform_chain": [
            source_config["adapter"],
            "day09_eval_candidate_schema_v1",
        ],
        "license": source_config["license"],
        "source_file": relative_path(source_file),
        "source_file_sha256": file_sha256(source_file),
    }
    if metadata:
        record["metadata"] = metadata
    return record


def stable_take(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (row["selection_hash"], row["parent_id"]))
    if len(ranked) < count:
        raise EvalCandidateError(f"source has {len(ranked)} rows but requires {count}")
    return ranked[:count]


def load_mmlu(
    source_root: Path, source_config: dict[str, Any], version: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    repo = source_root / "mmlu-data-mirror"
    if git_revision(repo) != source_config["revision"]:
        raise EvalCandidateError("MMLU mirror revision mismatch")
    records: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    per_subject = int(source_config["samples_per_subject"])
    for subject in source_config["subjects"]:
        path = repo / subject / "test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        input_hashes[path.relative_to(source_root).as_posix()] = file_sha256(path)
        subject_records: list[dict[str, Any]] = []
        for row_index, row in frame.iterrows():
            choices = [str(choice) for choice in row["choices"]]
            answer_index = int(row["answer"])
            if answer_index < 0 or answer_index >= len(choices):
                raise EvalCandidateError(f"invalid MMLU answer index: {answer_index}")
            prompt_lines = [str(row["question"]).strip(), "", "Choices:"]
            prompt_lines.extend(
                f"{CHOICE_LABELS[index]}. {choice}"
                for index, choice in enumerate(choices)
            )
            parent_id = f"{subject}:test:{int(row_index):04d}"
            subject_records.append(
                canonical_candidate(
                    version=version,
                    skill="general",
                    source_config=source_config,
                    parent_id=parent_id,
                    prompt="\n".join(prompt_lines),
                    reference=(
                        f"{CHOICE_LABELS[answer_index]}. {choices[answer_index]}"
                    ),
                    source_file=path,
                    metadata={
                        "subject": subject,
                        "answer_index": answer_index,
                        "parent_id_policy": "subject + split + zero-based parquet row index",
                    },
                )
            )
        records.extend(stable_take(subject_records, per_subject))
    return records, input_hashes


def load_gsm8k(
    source_root: Path,
    source_config: dict[str, Any],
    version: str,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    repo = source_root / "gsm8k"
    if git_revision(repo) != source_config["revision"]:
        raise EvalCandidateError("GSM8K revision mismatch")
    path = repo / "grade_school_math" / "data" / "test.jsonl"
    records = [
        canonical_candidate(
            version=version,
            skill="math",
            source_config=source_config,
            parent_id=f"test:{index:04d}",
            prompt=row["question"],
            reference=row["answer"],
            source_file=path,
            metadata={
                "parent_id_policy": "split + zero-based JSONL row index",
                "reference_format": "GSM8K rationale with final answer after ####",
            },
        )
        for index, row in enumerate(load_jsonl(path))
    ]
    return stable_take(records, count), {
        path.relative_to(source_root).as_posix(): file_sha256(path)
    }


def load_human_eval(
    source_root: Path,
    source_config: dict[str, Any],
    version: str,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    repo = source_root / "human-eval"
    if git_revision(repo) != source_config["revision"]:
        raise EvalCandidateError("HumanEval revision mismatch")
    path = repo / "data" / "HumanEval.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    records = [
        canonical_candidate(
            version=version,
            skill="code",
            source_config=source_config,
            parent_id=row["task_id"],
            prompt=row["prompt"],
            reference=row["canonical_solution"],
            source_file=path,
            metadata={
                "entry_point": row["entry_point"],
                "test_sha256": object_sha256(row["test"]),
                "execution_policy": "candidate preparation only; do not execute",
            },
        )
        for row in source_rows
    ]
    return stable_take(records, count), {
        path.relative_to(source_root).as_posix(): file_sha256(path)
    }


def render_tatqa_context(document: dict[str, Any]) -> str:
    table_rows = document["table"]["table"]
    table_lines = ["\t".join(str(cell).strip() for cell in row) for row in table_rows]
    paragraphs = sorted(document["paragraphs"], key=lambda item: int(item["order"]))
    paragraph_lines = [
        f"[{item['order']}] {str(item['text']).strip()}" for item in paragraphs
    ]
    return "Table:\n" + "\n".join(table_lines) + "\n\nParagraphs:\n" + "\n".join(
        paragraph_lines
    )


def load_tatqa(
    source_root: Path,
    source_config: dict[str, Any],
    version: str,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    repo = source_root / "tat-qa"
    if git_revision(repo) != source_config["revision"]:
        raise EvalCandidateError("TAT-QA revision mismatch")
    path = repo / "dataset_raw" / "tatqa_dataset_dev.json"
    records: list[dict[str, Any]] = []
    for document in load_json(path):
        context = render_tatqa_context(document)
        table_uid = document["table"]["uid"]
        for question in document["questions"]:
            reference = json.dumps(
                {
                    "answer": question["answer"],
                    "answer_type": question["answer_type"],
                    "derivation": question["derivation"],
                    "scale": question["scale"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            prompt = context + "\n\nQuestion:\n" + question["question"].strip()
            records.append(
                canonical_candidate(
                    version=version,
                    skill="finance",
                    source_config=source_config,
                    parent_id=question["uid"],
                    prompt=prompt,
                    reference=reference,
                    source_file=path,
                    metadata={
                        "table_uid": table_uid,
                        "answer_from": question["answer_from"],
                        "question_order": question["order"],
                    },
                )
            )
    return stable_take(records, count), {
        path.relative_to(source_root).as_posix(): file_sha256(path)
    }


def prepare_candidates(
    config: dict[str, Any], source_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    eval_config = config["eval_candidates"]
    version = eval_config["version"]
    count = int(eval_config["samples_per_slice"])
    sources = {item["slice"]: item for item in eval_config["sources"]}
    if set(sources) != {"general", "math", "code", "finance"}:
        raise EvalCandidateError("eval sources must cover four assignment slices")

    general, general_hashes = load_mmlu(source_root, sources["general"], version)
    if len(general) != count:
        raise EvalCandidateError("MMLU subject quotas do not equal samples_per_slice")
    math, math_hashes = load_gsm8k(
        source_root, sources["math"], version, count
    )
    code, code_hashes = load_human_eval(
        source_root, sources["code"], version, count
    )
    finance, finance_hashes = load_tatqa(
        source_root, sources["finance"], version, count
    )
    input_hashes = {
        **general_hashes,
        **math_hashes,
        **code_hashes,
        **finance_hashes,
    }
    return {
        "general": general,
        "math": math,
        "code": code,
        "finance": finance,
    }, dict(sorted(input_hashes.items()))


def validate_candidates(
    config: dict[str, Any], pools: dict[str, list[dict[str, Any]]]
) -> dict[str, bool]:
    expected = int(config["eval_candidates"]["samples_per_slice"])
    records = [record for rows in pools.values() for record in rows]
    ids = [record["eval_sample_id"] for record in records]
    content_hashes = [record["content_hash"] for record in records]
    training_sources = {
        source["source_id"] for source in config["sources"]
    }
    eval_sources = {record["source"] for record in records}
    return {
        "four_slices_present": set(pools) == {"general", "math", "code", "finance"},
        "counts_passed": all(len(rows) == expected for rows in pools.values()),
        "eval_ids_unique": len(ids) == len(set(ids)),
        "content_hashes_unique": len(content_hashes) == len(set(content_hashes)),
        "content_hashes_valid": all(
            record["content_hash"]
            == object_sha256(
                {"prompt": record["prompt"], "reference": record["reference"]}
            )
            for record in records
        ),
        "required_fields_present": all(
            all(
                field in record
                for field in (
                    "eval_sample_id",
                    "source",
                    "revision",
                    "split",
                    "parent_id",
                    "prompt",
                    "reference",
                    "skill",
                    "content_hash",
                )
            )
            for record in records
        ),
        "eval_sources_independent_from_training_sources": not (
            training_sources & eval_sources
        ),
    }


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    pools: dict[str, list[dict[str, Any]]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for skill, rows in pools.items():
        path = output_dir / f"{skill}-eval-candidates.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_hashes[path.name] = file_sha256(path)

    eval_config = config["eval_candidates"]
    source_details = {
        item["slice"]: {
            key: item[key]
            for key in (
                "source",
                "revision",
                "split",
                "adapter",
                "license",
                "license_note",
            )
        }
        for item in eval_config["sources"]
    }
    validation = validate_candidates(config, pools)
    summary = {
        "status": "complete_candidates_only",
        "candidate_version": eval_config["version"],
        "candidate_policy": eval_config["candidate_policy"],
        "selection_method": eval_config["selection_method"],
        "samples_per_slice": eval_config["samples_per_slice"],
        "total_candidates": sum(len(rows) for rows in pools.values()),
        "counts_by_slice": {
            skill: len(rows) for skill, rows in sorted(pools.items())
        },
        "source_details": source_details,
        "subject_counts": dict(
            sorted(
                Counter(
                    row.get("metadata", {}).get("subject")
                    for row in pools["general"]
                ).items()
            )
        ),
        "config_sha256": file_sha256(config_path),
        "eval_candidate_config_sha256": object_sha256(eval_config),
        "implementation_sha256": file_sha256(Path(__file__)),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "validation_checks": validation,
        "training_pool_changed": False,
        "day10_protocol_frozen": False,
    }
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise EvalCandidateError(f"eval candidate validation failed: {failed}")
    summary_path = output_dir / "eval-candidates-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def render_report_section(summary: dict[str, Any]) -> str:
    lines = [
        REPORT_START,
        "",
        "## Step 8 fixes 160 independent eval candidates, not the Day 10 protocol",
        "",
        "Four independently sourced candidate pools now contain 40 records each. "
        "Their IDs, prompts, references, revisions, and content hashes are stable; "
        "decoder settings, scoring, baselines, and the final evaluation protocol "
        "remain intentionally unfrozen until Day 10.",
        "",
        "| Slice | Source | Revision | Split | Candidates | License |",
        "|---|---|---|---|---:|---|",
    ]
    for skill in ("general", "math", "code", "finance"):
        source = summary["source_details"][skill]
        lines.append(
            f"| {skill} | `{source['source']}` | `{source['revision']}` | "
            f"{source['split']} | {summary['counts_by_slice'][skill]} | "
            f"{source['license']} |"
        )
    lines.extend(
        [
            "",
            "### Stability and scope",
            "",
            "Every candidate includes `eval_sample_id`, source/revision/split/parent "
            "ID, prompt, reference, skill, content hash, selection hash, adapter, "
            "license, and source-file hash. MMLU is stratified across four general "
            "subjects at 10 candidates each; the other slices use stable-hash "
            "sampling from their pinned split.",
            "",
            "These pools are inputs to Step 9 train/eval overlap detection. They "
            "must not yet be described as decontaminated or as the frozen Day 10 "
            "evaluation protocol.",
            "",
            "- Candidates: `tmp/day09-work/step8-eval-candidates/*-eval-candidates.jsonl`",
            "- Summary: `tmp/day09-work/step8-eval-candidates/eval-candidates-summary.json`",
            "",
            REPORT_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    if not report_path.is_file():
        raise EvalCandidateError(f"missing report: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise EvalCandidateError("Step 8 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    pools, input_hashes = prepare_candidates(config, args.source_root)
    summary = write_outputs(
        args.output_dir, config, args.config, pools, input_hashes
    )
    update_report(args.report, summary)
    print(
        f"eval_candidates={summary['total_candidates']} "
        f"counts={summary['counts_by_slice']} day10_protocol_frozen=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
