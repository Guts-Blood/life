#!/usr/bin/env python3
"""Build the deterministic Day 09 pre-filter data-quality profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_WORK = HERE.parents[2] / "tmp" / "day09-work"
DEFAULT_OUTPUT = DEFAULT_WORK / "step4-audit"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
CODE = chr(96)


class AuditError(ValueError):
    """A Step 4 input or accounting invariant failed."""


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


def percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 6)


def bucket_value(value: int, buckets: list[dict[str, Any]]) -> str:
    for bucket in buckets:
        minimum = bucket["minimum"]
        maximum = bucket["maximum"]
        if value >= minimum and (maximum is None or value <= maximum):
            return bucket["name"]
    raise AuditError(f"value {value} does not match a configured bucket")


def nearest_rank(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise AuditError("cannot compute a percentile for an empty collection")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def length_statistics(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [int(row[field]) for row in rows]
    return {
        "min": min(values),
        "p50": nearest_rank(values, 0.50),
        "p90": nearest_rank(values, 0.90),
        "p99": nearest_rank(values, 0.99),
        "max": max(values),
    }


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def refusal_match(text: str, phrases: list[str]) -> str | None:
    normalized = normalized_text(text)
    return next((phrase for phrase in phrases if phrase in normalized), None)


def template_signature(text: str, prefix_token_count: int) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"https?://\S+", " <url> ", normalized)
    normalized = re.sub(r"(?<!\w)[+-]?(?:\d[\d,]*)(?:\.\d+)?%?", " <num> ", normalized)
    tokens = re.findall(r"<url>|<num>|[a-z]+(?:'[a-z]+)?", normalized)
    return " ".join(tokens[:prefix_token_count]) or "<empty>"


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


def row_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
        "input_tokens": sum(int(row["input_token_count"]) for row in rows),
        "supervised_tokens": sum(int(row["supervised_token_count"]) for row in rows),
    }


def profile_dimension(
    rows: list[dict[str, Any]],
    dimension: str,
    value_for: Callable[[dict[str, Any]], str],
    totals: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[value_for(row)].append(row)

    profile: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for value in sorted(groups):
        group_rows = groups[value]
        group_totals = row_totals(group_rows)
        sample_ids = sorted(row["sample_id"] for row in group_rows)
        evidence_key = f"{dimension}={value}"
        sample_ids_sha256 = object_sha256(sample_ids)
        profile.append(
            {
                "value": value,
                **group_totals,
                "example_percentage": percentage(
                    group_totals["examples"], totals["examples"]
                ),
                "raw_token_percentage": percentage(
                    group_totals["raw_tokens"], totals["raw_tokens"]
                ),
                "input_token_percentage": percentage(
                    group_totals["input_tokens"], totals["input_tokens"]
                ),
                "supervised_token_percentage": percentage(
                    group_totals["supervised_tokens"], totals["supervised_tokens"]
                ),
                "evidence_key": evidence_key,
                "sample_ids_sha256": sample_ids_sha256,
            }
        )
        evidence.append(
            {
                "dimension": dimension,
                "value": value,
                "evidence_key": evidence_key,
                **group_totals,
                "sample_ids": sample_ids,
                "sample_ids_sha256": sample_ids_sha256,
            }
        )
    return profile, evidence


def validate_and_load_inputs(
    config: dict[str, Any], preprocess_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = preprocess_dir / "preprocess-summary.json"
    if not summary_path.is_file():
        raise AuditError(f"missing Step 3 summary: {summary_path}")
    preprocess_summary = load_json(summary_path)
    preprocessing = config["preprocessing"]
    if preprocess_summary["max_length"] != preprocessing["max_length"]:
        raise AuditError("Step 3 max_length does not match current config")
    if preprocess_summary["contract_version"] != preprocessing["contract_version"]:
        raise AuditError("Step 3 contract version does not match current config")

    required_accepted = {
        "sample_id",
        "source",
        "revision",
        "parent_id",
        "messages",
        "skill",
        "subskill",
        "language",
        "raw_token_count",
        "input_token_count",
        "supervised_token_count",
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = [
        {
            "role": "step3_summary",
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        }
    ]
    seen_ids: set[str] = set()

    for source_summary in preprocess_summary["sources"]:
        slice_name = source_summary["slice"]
        accepted_path = preprocess_dir / source_summary["accepted_file"]
        rejected_path = preprocess_dir / source_summary["rejected_file"]
        if file_sha256(accepted_path) != source_summary["accepted_sha256"]:
            raise AuditError(f"accepted file hash mismatch for {slice_name}")
        if file_sha256(rejected_path) != source_summary["rejected_sha256"]:
            raise AuditError(f"rejected file hash mismatch for {slice_name}")
        source_accepted = load_jsonl(accepted_path)
        source_rejected = load_jsonl(rejected_path)
        if len(source_accepted) != source_summary["accepted_count"]:
            raise AuditError(f"accepted count mismatch for {slice_name}")
        if len(source_rejected) != source_summary["rejected_count"]:
            raise AuditError(f"rejected count mismatch for {slice_name}")
        if len(source_accepted) + len(source_rejected) != source_summary["candidate_count"]:
            raise AuditError(f"candidate accounting mismatch for {slice_name}")

        for row in source_accepted:
            missing = required_accepted - row.keys()
            if missing:
                raise AuditError(
                    f"{row.get('sample_id')} missing accepted fields: {sorted(missing)}"
                )
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise AuditError(f"duplicate sample_id across Step 3 outputs: {sample_id}")
            if row["skill"] != slice_name:
                raise AuditError(f"skill/slice mismatch for {sample_id}")
            if not 0 < row["supervised_token_count"] <= row["input_token_count"]:
                raise AuditError(f"invalid token counts for {sample_id}")
            seen_ids.add(sample_id)
        for row in source_rejected:
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise AuditError(f"duplicate sample_id across Step 3 outputs: {sample_id}")
            seen_ids.add(sample_id)

        accepted.extend(source_accepted)
        rejected.extend(source_rejected)
        input_files.extend(
            [
                {
                    "role": f"{slice_name}_accepted",
                    "path": str(accepted_path),
                    "sha256": source_summary["accepted_sha256"],
                },
                {
                    "role": f"{slice_name}_rejected",
                    "path": str(rejected_path),
                    "sha256": source_summary["rejected_sha256"],
                },
            ]
        )

    expected = preprocess_summary["totals"]
    if len(accepted) != expected["accepted_count"]:
        raise AuditError("total accepted count does not match Step 3 summary")
    if len(rejected) != expected["rejected_count"]:
        raise AuditError("total rejected count does not match Step 3 summary")
    if len(seen_ids) != expected["candidate_count"]:
        raise AuditError("total unique sample IDs do not match Step 3 summary")
    return preprocess_summary, accepted, rejected, input_files


def rejection_profile(
    rejected: list[dict[str, Any]], dimension: str, field: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rejected:
        value = row[field] if field != "slice" else row["sample_id"].split(":", 1)[0]
        groups[str(value)].append(row)
    profile = []
    evidence = []
    for value in sorted(groups):
        sample_ids = sorted(row["sample_id"] for row in groups[value])
        evidence_key = f"{dimension}={value}"
        sample_ids_sha256 = object_sha256(sample_ids)
        profile.append(
            {
                "value": value,
                "examples": len(sample_ids),
                "percentage": percentage(len(sample_ids), len(rejected)),
                "evidence_key": evidence_key,
                "sample_ids_sha256": sample_ids_sha256,
            }
        )
        evidence.append(
            {
                "dimension": dimension,
                "value": value,
                "evidence_key": evidence_key,
                "examples": len(sample_ids),
                "sample_ids": sample_ids,
                "sample_ids_sha256": sample_ids_sha256,
            }
        )
    return profile, evidence


def build_audit(
    config: dict[str, Any],
    preprocess_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preprocess_summary, rows, rejected, input_files = validate_and_load_inputs(
        config, preprocess_dir
    )
    audit_config = config["quality_audit"]
    totals = row_totals(rows)

    template_rule = audit_config["template_cluster"]
    refusal_rule = audit_config["refusal_heuristic"]
    difficulty_rule = audit_config["difficulty_proxy"]
    signatures = {
        row["sample_id"]: template_signature(
            prompt_text(row), template_rule["prefix_token_count"]
        )
        for row in rows
    }
    signature_counts = Counter(signatures.values())
    refusal_matches = {
        row["sample_id"]: refusal_match(
            assistant_text(row), refusal_rule["phrases"]
        )
        for row in rows
    }

    dimensions: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    dimension_rules: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("source", lambda row: row["source"]),
        ("skill", lambda row: row["skill"]),
        ("subskill", lambda row: row["subskill"]),
        ("language", lambda row: row["language"]),
        (
            "length_bucket",
            lambda row: bucket_value(
                row[audit_config["length_field"]],
                audit_config["length_buckets"],
            ),
        ),
        (
            "refusal_status",
            lambda row: (
                "refusal_phrase_match"
                if refusal_matches[row["sample_id"]] is not None
                else "no_refusal_phrase_match"
            ),
        ),
        (
            "difficulty_proxy",
            lambda row: bucket_value(
                row[difficulty_rule["field"]],
                difficulty_rule["buckets"],
            ),
        ),
        (
            "source_provenance",
            lambda row: audit_config["source_provenance"][row["skill"]],
        ),
        (
            "template_cluster_status",
            lambda row: (
                "repeated_prefix_cluster"
                if signature_counts[signatures[row["sample_id"]]]
                >= template_rule["minimum_cluster_size"]
                else "not_repeated"
            ),
        ),
    ]
    for dimension, value_for in dimension_rules:
        profile, dimension_evidence = profile_dimension(
            rows, dimension, value_for, totals
        )
        dimensions[dimension] = profile
        evidence.extend(dimension_evidence)

    by_slice: dict[str, dict[str, dict[str, int]]] = {}
    for slice_name in sorted({row["skill"] for row in rows}):
        slice_rows = [row for row in rows if row["skill"] == slice_name]
        by_slice[slice_name] = {
            "input_token_count": length_statistics(slice_rows, "input_token_count"),
            "supervised_token_count": length_statistics(
                slice_rows, "supervised_token_count"
            ),
        }

    cluster_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster_groups[signatures[row["sample_id"]]].append(row)
    repeated_clusters = []
    for signature, cluster_rows in cluster_groups.items():
        if len(cluster_rows) < template_rule["minimum_cluster_size"]:
            continue
        cluster_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
        sample_ids = sorted(row["sample_id"] for row in cluster_rows)
        evidence_key = f"template_cluster={cluster_id}"
        sample_ids_sha256 = object_sha256(sample_ids)
        repeated_clusters.append(
            {
                "cluster_id": cluster_id,
                "signature_preview": signature[:160],
                "examples": len(cluster_rows),
                "example_percentage": percentage(len(cluster_rows), len(rows)),
                "source_counts": dict(
                    sorted(Counter(row["skill"] for row in cluster_rows).items())
                ),
                "evidence_key": evidence_key,
                "sample_ids_sha256": sample_ids_sha256,
            }
        )
        evidence.append(
            {
                "dimension": "template_cluster",
                "value": cluster_id,
                "evidence_key": evidence_key,
                "examples": len(cluster_rows),
                "sample_ids": sample_ids,
                "sample_ids_sha256": sample_ids_sha256,
            }
        )
    repeated_clusters.sort(key=lambda item: (-item["examples"], item["cluster_id"]))

    rejection_by_reason, rejection_reason_evidence = rejection_profile(
        rejected, "contract_rejection_reason", "reason_code"
    )
    rejection_by_slice, rejection_slice_evidence = rejection_profile(
        rejected, "contract_rejection_slice", "slice"
    )
    evidence.extend(rejection_reason_evidence)
    evidence.extend(rejection_slice_evidence)

    matched_refusal_phrases = Counter(
        match for match in refusal_matches.values() if match is not None
    )
    summary = {
        "audit_version": audit_config["version"],
        "audit_date": config["preregistered_date"],
        "cohort_definition": audit_config["cohort"],
        "quality_audit_config_sha256": object_sha256(audit_config),
        "preprocessing_config_sha256": object_sha256(config["preprocessing"]),
        "preprocess_summary_sha256": file_sha256(
            preprocess_dir / "preprocess-summary.json"
        ),
        "input_files": input_files,
        "totals": {
            **totals,
            "contract_rejected_examples": len(rejected),
            "candidate_examples": len(rows) + len(rejected),
            "contract_acceptance_rate": percentage(
                len(rows), len(rows) + len(rejected)
            ),
            "accepted_truncated_examples": sum(bool(row["truncated"]) for row in rows),
        },
        "dimensions": dimensions,
        "length_statistics": {
            "percentile_method": audit_config["percentile_method"],
            "overall": {
                "input_token_count": length_statistics(rows, "input_token_count"),
                "supervised_token_count": length_statistics(
                    rows, "supervised_token_count"
                ),
            },
            "by_slice": by_slice,
        },
        "refusal_heuristic": {
            "matched_phrase_counts": dict(sorted(matched_refusal_phrases.items())),
            "rule": refusal_rule,
        },
        "template_clusters": {
            "rule": template_rule,
            "repeated_cluster_count": len(repeated_clusters),
            "top_clusters": repeated_clusters[:20],
        },
        "difficulty_proxy": difficulty_rule,
        "contract_rejections": {
            "by_reason": rejection_by_reason,
            "by_slice": rejection_by_slice,
        },
        "method_notes": {
            "claim_scope": "descriptive pre-filter profile; no causal or quality-removal claim",
            "chart_omission": "Exact audit tables are used because this is a four-slice static snapshot and sample-ID lookup matters more than a graphical trend.",
        },
    }
    evidence.sort(key=lambda item: (item["dimension"], item["value"]))
    return summary, evidence


def markdown_dimension(
    title: str, rows: list[dict[str, Any]]
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Value | Examples | Example % | Raw tokens | Raw % | Supervised tokens | Supervised % | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['value']} | {row['examples']:,} | "
            f"{row['example_percentage']:.2f}% | {row['raw_tokens']:,} | "
            f"{row['raw_token_percentage']:.2f}% | "
            f"{row['supervised_tokens']:,} | "
            f"{row['supervised_token_percentage']:.2f}% | "
            f"{CODE}{row['evidence_key']}{CODE} |"
        )
    lines.append("")
    return lines


def build_report(
    summary: dict[str, Any],
    evidence_sha256: str,
    summary_sha256: str,
) -> str:
    totals = summary["totals"]
    dimensions = summary["dimensions"]
    source_rows = dimensions["skill"]
    largest_supervised = max(
        source_rows, key=lambda row: row["supervised_token_percentage"]
    )
    synthetic = next(
        row
        for row in dimensions["source_provenance"]
        if row["value"] == "declared_synthetic"
    )
    refusal = next(
        row
        for row in dimensions["refusal_status"]
        if row["value"] == "refusal_phrase_match"
    )
    repeated = next(
        row
        for row in dimensions["template_cluster_status"]
        if row["value"] == "repeated_prefix_cluster"
    )
    overall_lengths = summary["length_statistics"]["overall"]

    lines = [
        "# Day 09 Pre-filter Data Quality Audit",
        "",
        "## Technical summary",
        "",
        f"- Step 3 produced {totals['examples']:,} contract-accepted records from "
        f"{totals['candidate_examples']:,} candidates "
        f"({totals['contract_acceptance_rate']:.2f}%); "
        f"{totals['contract_rejected_examples']:,} records remain outside this "
        "pre-filter cohort.",
        f"- {largest_supervised['value']} contributes "
        f"{largest_supervised['supervised_token_percentage']:.2f}% of supervised "
        f"tokens but {largest_supervised['example_percentage']:.2f}% of examples. "
        "Example share therefore does not represent training-signal share.",
        f"- Sources that explicitly declare synthetic construction contribute "
        f"{synthetic['example_percentage']:.2f}% of examples and "
        f"{synthetic['supervised_token_percentage']:.2f}% of supervised tokens. "
        "Synthetic provenance is the largest pre-filter composition risk.",
        f"- The deterministic refusal heuristic flags {refusal['examples']:,} "
        f"records ({refusal['example_percentage']:.2f}%), while the prompt-prefix "
        f"heuristic places {repeated['examples']:,} records "
        f"({repeated['example_percentage']:.2f}%) in repeated clusters. Both are "
        "review hints, not automatic quality failures.",
        "",
        "## The accepted pool is token-heavy in a different mix than its example counts",
        "",
        "All percentages below use the 7,864 Step 3 accepted records as the "
        "denominator. Raw-token and supervised-token percentages are calculated "
        "independently. Every Evidence value resolves to the complete sample-ID "
        "list in the evidence index.",
        "",
    ]
    lines.extend(markdown_dimension("Source dataset", dimensions["source"]))
    lines.extend(markdown_dimension("Skill", dimensions["skill"]))
    lines.extend(markdown_dimension("Subskill", dimensions["subskill"]))
    lines.extend(markdown_dimension("Language", dimensions["language"]))

    lines.extend(
        [
            "## Most accepted sequences fit below 2048, but length and response workload are source-dependent",
            "",
            f"Overall input length is p50={overall_lengths['input_token_count']['p50']:,}, "
            f"p90={overall_lengths['input_token_count']['p90']:,}, and "
            f"p99={overall_lengths['input_token_count']['p99']:,} tokens. "
            f"Supervised response length is "
            f"p50={overall_lengths['supervised_token_count']['p50']:,}, "
            f"p90={overall_lengths['supervised_token_count']['p90']:,}, and "
            f"p99={overall_lengths['supervised_token_count']['p99']:,}. "
            "Longer responses are not treated as higher quality.",
            "",
        ]
    )
    lines.extend(markdown_dimension("Input-token length buckets", dimensions["length_bucket"]))
    lines.extend(
        markdown_dimension(
            "Supervised-response length proxy", dimensions["difficulty_proxy"]
        )
    )
    lines.extend(
        [
            "### Length percentiles by slice",
            "",
            "| Slice | Input min | p50 | p90 | p99 | max | Supervised min | p50 | p90 | p99 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slice_name, stats in summary["length_statistics"]["by_slice"].items():
        input_stats = stats["input_token_count"]
        supervised_stats = stats["supervised_token_count"]
        lines.append(
            f"| {slice_name} | {input_stats['min']:,} | {input_stats['p50']:,} | "
            f"{input_stats['p90']:,} | {input_stats['p99']:,} | "
            f"{input_stats['max']:,} | {supervised_stats['min']:,} | "
            f"{supervised_stats['p50']:,} | {supervised_stats['p90']:,} | "
            f"{supervised_stats['p99']:,} | {supervised_stats['max']:,} |"
        )
    lines.extend(
        [
            "",
            "The difficulty label above is only a fixed supervised-response-length "
            "proxy: low=1–64, medium=65–256, high=257+. It does not establish "
            "correctness, reasoning depth, or semantic difficulty.",
            "",
            "## Synthetic provenance and repeated prompt prefixes need manual review",
            "",
            "Synthetic provenance comes from the pinned source README declarations. "
            "Template clusters use the first 16 normalized lexical prompt tokens "
            "and require at least five records; they are deliberately candidate "
            "signals rather than dedup decisions.",
            "",
        ]
    )
    lines.extend(
        markdown_dimension("Source provenance", dimensions["source_provenance"])
    )
    lines.extend(
        markdown_dimension(
            "Template-cluster status", dimensions["template_cluster_status"]
        )
    )
    lines.extend(markdown_dimension("Refusal heuristic", dimensions["refusal_status"]))
    lines.extend(
        [
            "### Largest repeated prompt-prefix clusters",
            "",
            "| Cluster | Examples | Example % | Source counts | Signature preview | Evidence |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for cluster in summary["template_clusters"]["top_clusters"][:10]:
        source_counts = ", ".join(
            f"{key}:{value}" for key, value in cluster["source_counts"].items()
        )
        preview = cluster["signature_preview"].replace("|", "\\|")
        lines.append(
            f"| {cluster['cluster_id']} | {cluster['examples']:,} | "
            f"{cluster['example_percentage']:.2f}% | {source_counts} | "
            f"{preview} | {CODE}{cluster['evidence_key']}{CODE} |"
        )
    lines.extend(
        [
            "",
            "## Contract rejections are isolated from the pre-filter baseline",
            "",
            f"Step 3 rejected {totals['contract_rejected_examples']:,} records. "
            "They do not contribute to any accepted-pool example or token "
            "percentage above.",
            "",
            "| Rejection reason | Examples | Rejected % | Evidence |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["contract_rejections"]["by_reason"]:
        lines.append(
            f"| {row['value']} | {row['examples']:,} | {row['percentage']:.2f}% | "
            f"{CODE}{row['evidence_key']}{CODE} |"
        )
    lines.extend(
        [
            "",
            "## Scope, definitions, and method",
            "",
            "- Grain: one canonical sample_id per accepted record.",
            "- Cohort: Step 3 accepted records before manual quality decisions, "
            "quality filters, exact/near dedup, and decontamination.",
            "- Token units: raw tokens exclude chat-template overhead; input tokens "
            "include the frozen template after truncation; supervised tokens are "
            "post-truncation labels not equal to -100.",
            "- Percentiles use the nearest-rank method.",
            "- Refusal, response-length difficulty, and prompt-prefix clustering are "
            "deterministic risk hints only.",
            "- Exact audit tables are used instead of charts because this is a "
            "four-slice static snapshot and sample-ID lookup is the primary need.",
            "",
            "## Limitations and robustness checks",
            "",
            "- Source-level skill, subskill, language, and provenance labels are "
            "pipeline metadata, not independently verified per-example labels.",
            "- The refusal phrase list can miss paraphrases and can produce false positives.",
            "- Response length is confounded with source and answer style; it is not "
            "a semantic difficulty measurement.",
            "- Prefix clustering can group structurally similar but substantively "
            "different prompts. Step 7 near-duplicate review must not treat it as "
            "confirmed duplication.",
            "- Input hashes, aggregate sums, percentage closures, and sample-ID "
            "evidence hashes are validated by the audit script.",
            "",
            "## Recommended next step",
            "",
            "Proceed to Step 8 only after fixing independent eval sources and revisions "
            "for general, math, code, and finance. Do not derive evaluation candidates "
            "from the training sources or treat the Step 7 similarity threshold as a "
            "removal decision.",
            "",
            "## Further questions",
            "",
            "- Are the source-level language labels correct for every accepted sample?",
            "- Do the largest synthetic prefix clusters contain real task diversity "
            "or mostly surface-level persona variation?",
            "- Are long math responses correct and non-redundant, especially in the "
            "high response-length proxy bucket?",
            "",
            "Audit support hashes:",
            "",
            f"- summary: {CODE}{summary_sha256}{CODE}",
            f"- evidence index: {CODE}{evidence_sha256}{CODE}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    report_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "audit-summary.json"
    evidence_path = output_dir / "evidence-index.jsonl"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with evidence_path.open("w", encoding="utf-8") as handle:
        for item in evidence:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = build_report(
        summary,
        evidence_sha256=file_sha256(evidence_path),
        summary_sha256=file_sha256(summary_path),
    )
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preprocess-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    preprocess_dir = args.preprocess_dir or (
        DEFAULT_WORK / f"step3-max{config['preprocessing']['max_length']}"
    )
    summary, evidence = build_audit(config, preprocess_dir)
    write_outputs(summary, evidence, args.output_dir, args.report)
    totals = summary["totals"]
    print(
        f"accepted={totals['examples']} rejected={totals['contract_rejected_examples']} "
        f"raw_tokens={totals['raw_tokens']} "
        f"supervised_tokens={totals['supervised_tokens']}"
    )


if __name__ == "__main__":
    main()
