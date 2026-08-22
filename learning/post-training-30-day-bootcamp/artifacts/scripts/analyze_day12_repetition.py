#!/usr/bin/env python3
"""Quantify Day 12 raw-generation repetition independent of answer correctness."""

from __future__ import annotations

import json
import re
import statistics
import zlib
from collections import Counter
from pathlib import Path
from typing import Any


ARTIFACTS = Path(__file__).resolve().parents[1]
EVAL_DIR = ARTIFACTS / "eval"
OUTPUT_PATH = ARTIFACTS / "reports" / "day12-repetition-diagnostic.json"
MODELS = ("Base", "E", "H")
SLICES = ("math", "code")
MAX_NEW_TOKENS = 512
CONTAMINATION_PATTERNS = {
    "role_or_header_leak": r"Human:|Assistant:|You are a helpful assistant|<\|im_",
    "replacement_character": r"�",
    "baojiahu_loop": r"保驾护",
    "larinizi_loop": r"larınızı",
    "ostoyannya_loop": r"остояння",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prediction_path(model: str) -> Path:
    if model == "Base":
        return EVAL_DIR / "day10-qwen3-0.6b-base-predictions.jsonl"
    return EVAL_DIR / f"{model}-25_percent-dev-predictions.jsonl"


def repeated_ngram_fraction(token_ids: list[int], n: int = 4) -> float:
    count = len(token_ids) - n + 1
    if count <= 0:
        return 0.0
    unique = len({tuple(token_ids[index : index + n]) for index in range(count)})
    return (count - unique) / count


def longest_exact_cycle_tokens(token_ids: list[int], max_block: int = 32) -> int:
    """Return the longest consecutive span made from one exactly repeated token block."""
    best = 0
    length = len(token_ids)
    for block_size in range(1, min(max_block, length // 2) + 1):
        for start in range(0, length - 2 * block_size + 1):
            block = token_ids[start : start + block_size]
            if block != token_ids[start + block_size : start + 2 * block_size]:
                continue
            repeats = 2
            while (
                start + (repeats + 1) * block_size <= length
                and token_ids[
                    start + repeats * block_size : start + (repeats + 1) * block_size
                ]
                == block
            ):
                repeats += 1
            best = max(best, repeats * block_size)
    return best


def safe_round(value: float) -> float:
    return round(value, 6)


def row_metrics(model: str, row: dict[str, Any]) -> dict[str, Any]:
    token_ids = row["output_token_ids"]
    raw_output = row["raw_output"]
    raw_bytes = raw_output.encode("utf-8")
    cycle_tokens = longest_exact_cycle_tokens(token_ids)
    contamination = {
        name: bool(re.search(pattern, raw_output))
        for name, pattern in CONTAMINATION_PATTERNS.items()
    }
    parsed = row.get("scorer_result", {}).get("parsed_answer")
    raw_parsed_mismatch = (
        row["slice"] == "code" and str(raw_output).strip() != str(parsed).strip()
    )
    repeat4 = repeated_ngram_fraction(token_ids, 4)
    compression_ratio = len(zlib.compress(raw_bytes, 9)) / max(1, len(raw_bytes))
    return {
        "model": model,
        "slice": row["slice"],
        "sample_id": row["sample_id"],
        "output_tokens": len(token_ids),
        "ceiling_hit": len(token_ids) == MAX_NEW_TOKENS,
        "repeat4_fraction": safe_round(repeat4),
        "compression_ratio": safe_round(compression_ratio),
        "exact_cycle_tokens": cycle_tokens,
        "exact_cycle_fraction": safe_round(cycle_tokens / max(1, len(token_ids))),
        "high_repeat4_posthoc": repeat4 >= 0.20,
        "high_compressibility_posthoc": compression_ratio <= 0.40,
        "long_exact_cycle_posthoc": cycle_tokens >= 16,
        "contamination_flag": any(contamination.values()),
        "contamination_categories": [name for name, hit in contamination.items() if hit],
        "raw_parsed_mismatch": raw_parsed_mismatch,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contamination_counts = Counter(
        category for row in rows for category in row["contamination_categories"]
    )
    return {
        "n": len(rows),
        "ceiling_hits": sum(row["ceiling_hit"] for row in rows),
        "median_output_tokens": statistics.median(row["output_tokens"] for row in rows),
        "mean_output_tokens": safe_round(
            statistics.mean(row["output_tokens"] for row in rows)
        ),
        "median_repeat4_fraction": safe_round(
            statistics.median(row["repeat4_fraction"] for row in rows)
        ),
        "mean_repeat4_fraction": safe_round(
            statistics.mean(row["repeat4_fraction"] for row in rows)
        ),
        "median_compression_ratio": safe_round(
            statistics.median(row["compression_ratio"] for row in rows)
        ),
        "median_exact_cycle_tokens": statistics.median(
            row["exact_cycle_tokens"] for row in rows
        ),
        "high_repeat4_posthoc_count": sum(row["high_repeat4_posthoc"] for row in rows),
        "high_compressibility_posthoc_count": sum(
            row["high_compressibility_posthoc"] for row in rows
        ),
        "long_exact_cycle_posthoc_count": sum(
            row["long_exact_cycle_posthoc"] for row in rows
        ),
        "contamination_count": sum(row["contamination_flag"] for row in rows),
        "contamination_category_counts": dict(sorted(contamination_counts.items())),
        "raw_parsed_mismatch_count": sum(row["raw_parsed_mismatch"] for row in rows),
    }


def paired_comparison(
    rows: list[dict[str, Any]], focal: str, comparator: str, slice_name: str
) -> dict[str, Any]:
    focal_rows = {
        row["sample_id"]: row
        for row in rows
        if row["model"] == focal and row["slice"] == slice_name
    }
    comparator_rows = {
        row["sample_id"]: row
        for row in rows
        if row["model"] == comparator and row["slice"] == slice_name
    }
    sample_ids = sorted(focal_rows)
    return {
        "n_paired": len(sample_ids),
        "focal_shorter_output_count": sum(
            focal_rows[sample_id]["output_tokens"]
            < comparator_rows[sample_id]["output_tokens"]
            for sample_id in sample_ids
        ),
        "focal_lower_repeat4_count": sum(
            focal_rows[sample_id]["repeat4_fraction"]
            < comparator_rows[sample_id]["repeat4_fraction"]
            for sample_id in sample_ids
        ),
        "focal_less_compressible_count": sum(
            focal_rows[sample_id]["compression_ratio"]
            > comparator_rows[sample_id]["compression_ratio"]
            for sample_id in sample_ids
        ),
        "focal_shorter_exact_cycle_count": sum(
            focal_rows[sample_id]["exact_cycle_tokens"]
            < comparator_rows[sample_id]["exact_cycle_tokens"]
            for sample_id in sample_ids
        ),
    }


def main() -> None:
    case_rows = [
        row_metrics(model, row)
        for model in MODELS
        for row in read_jsonl(prediction_path(model))
        if row["evaluation_split"] == "dev" and row["slice"] in SLICES
    ]
    aggregates = {
        f"{model}.{slice_name}": aggregate(
            [
                row
                for row in case_rows
                if row["model"] == model and row["slice"] == slice_name
            ]
        )
        for model in MODELS
        for slice_name in SLICES
    }
    paired = {
        f"E_vs_{comparator}.{slice_name}": paired_comparison(
            case_rows, "E", comparator, slice_name
        )
        for comparator in ("Base", "H")
        for slice_name in SLICES
    }
    output = {
        "schema_version": 1,
        "domain": "day12.raw_generation_repetition_diagnostic",
        "status": "complete_descriptive_posthoc",
        "question": "Ignoring answer correctness, do Base and H repeat or drift more than E?",
        "scope": {
            "models": list(MODELS),
            "slices": list(SLICES),
            "cases_per_model_slice": 28,
            "total_case_outputs": len(case_rows),
            "evaluation_split": "dev",
            "answer_correctness_used": False,
        },
        "metric_definitions": {
            "ceiling_hit": "output token count equals max_new_tokens=512",
            "repeat4_fraction": "1 - unique contiguous 4-grams / all contiguous 4-grams over output token ids",
            "compression_ratio": "zlib(level=9) bytes / raw UTF-8 bytes; lower is more compressible and usually more repetitive",
            "exact_cycle_tokens": "longest consecutive token span formed by exactly repeating one block of 1-32 tokens",
            "contamination_flag": "raw text contains a reviewed role/header, replacement-character, or observed multilingual loop marker",
            "raw_parsed_mismatch": "for Code, raw output differs from the completion executed after extractor parsing",
        },
        "posthoc_thresholds": {
            "high_repeat4": ">=0.20",
            "high_compressibility": "compression_ratio<=0.40",
            "long_exact_cycle": ">=16 consecutive cycle tokens",
            "interpretation": "Sensitivity aids selected after inspection; conclusions should rest on the continuous metrics and ceiling counts, not one threshold.",
        },
        "aggregates": aggregates,
        "paired_comparisons": paired,
        "case_metrics": case_rows,
        "claim_boundary": "Descriptive evidence for these 168 dev outputs under one generation protocol; not a causal estimate of training-data effects or a population benchmark.",
        "source_files": [str(prediction_path(model).relative_to(ARTIFACTS.parent)) for model in MODELS],
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
