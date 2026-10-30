#!/usr/bin/env python3
"""Materialize frozen Day 09 Mix A/B manifests from the Step 11 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BOOTCAMP = HERE.parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_CLEAN_POOL = BOOTCAMP / "artifacts" / "data" / "day09-dataset-manifest.json"
DEFAULT_PLAN = REPO_ROOT / "tmp" / "day09-work" / "step11-token-budget" / "token-budget-plan.json"
DEFAULT_OUTPUT = BOOTCAMP / "artifacts" / "data"
DEFAULT_WORK = REPO_ROOT / "tmp" / "day09-work" / "step12-mixtures"
DEFAULT_REPORT = BOOTCAMP / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP12_MIXTURES_START -->"
REPORT_END = "<!-- STEP12_MIXTURES_END -->"
SKILLS = ("general", "math", "code", "finance")
MIX_FILENAMES = {
    "mix_A_balanced": "day09-mix-A-balanced.json",
    "mix_B_targeted": "day09-mix-B-targeted.json",
}


class MixtureBuildError(ValueError):
    """A frozen input or materialized mixture violates the Day 09 contract."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_clean_pool(clean_pool: dict[str, Any]) -> None:
    header = clean_pool["header"]
    records = clean_pool["records"]
    header_without_hash = {
        key: value for key, value in header.items() if key != "manifest_hash"
    }
    if header["manifest_hash"] != object_sha256(
        {"header": header_without_hash, "records": records}
    ):
        raise MixtureBuildError("clean-pool manifest hash is invalid")
    if header["clean_pool_hash"] != object_sha256(
        sorted(records, key=lambda row: row["sample_id"])
    ):
        raise MixtureBuildError("clean-pool hash is invalid")
    sample_ids = [row["sample_id"] for row in records]
    if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise MixtureBuildError("clean-pool sample IDs must be unique and sorted")


def validate_plan(
    plan: dict[str, Any],
    plan_path: Path,
    clean_pool: dict[str, Any],
    clean_pool_path: Path,
    config: dict[str, Any],
) -> None:
    plan_without_hash = {key: value for key, value in plan.items() if key != "plan_hash"}
    if plan["plan_hash"] != object_sha256(plan_without_hash):
        raise MixtureBuildError("Step 11 plan hash is invalid")
    source = plan["source_manifest"]
    clean_header = clean_pool["header"]
    source_checks = {
        "clean-pool file hash": source["file_sha256"] == file_sha256(clean_pool_path),
        "clean-pool manifest hash": source["manifest_hash"] == clean_header["manifest_hash"],
        "parent-pool hash": source["clean_pool_hash"] == clean_header["clean_pool_hash"],
        "clean supervised-token total": source["total_clean_supervised_tokens"]
        == clean_header["totals"]["supervised_tokens"],
        "configured plan path": relative_path(plan_path)
        == config["token_budget"]["budget_plan_path"],
        "materialization source plan path": relative_path(plan_path)
        == config["mixture_materialization"]["source_plan_path"],
        "token-budget version": plan["token_budget_version"]
        == config["token_budget"]["version"],
        "unit": plan["unit"] == config["mixtures"]["unit"],
        "plan status": plan["status"] == "complete_frozen",
    }
    failed = [name for name, passed in source_checks.items() if not passed]
    if failed:
        raise MixtureBuildError(f"Step 11 source linkage failed: {failed}")
    if not all(plan["validation_checks"].values()):
        raise MixtureBuildError("Step 11 plan contains a failed validation check")
    for mix_name in MIX_FILENAMES:
        configured = config["mixtures"][mix_name]
        planned = plan["ratios"][mix_name]
        if any(Fraction(configured[skill]) != Fraction(planned[skill]) for skill in SKILLS):
            raise MixtureBuildError(f"{mix_name} ratios differ between config and plan")


def occurrence_sort_key(
    row: dict[str, Any], seed: int, mix_name: str
) -> tuple[str, str]:
    payload = "\0".join(
        (
            str(seed),
            mix_name,
            row["canonical_sample_id"],
            str(row["occurrence_index"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), row["occurrence_id"]


def build_occurrences(
    mix_name: str,
    proof: dict[str, Any],
    parent_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for skill in SKILLS:
        slice_proof = proof["slices"][skill]
        slice_ids: list[str] = []
        slice_tokens = 0
        slice_occurrences = 0
        for selection in slice_proof["selection"]:
            sample_id = selection["sample_id"]
            if sample_id in selected_ids:
                raise MixtureBuildError(f"{mix_name} selects {sample_id} more than once")
            selected_ids.add(sample_id)
            slice_ids.append(sample_id)
            if sample_id not in parent_by_id:
                raise MixtureBuildError(f"{mix_name} selection is missing from clean pool: {sample_id}")
            parent = parent_by_id[sample_id]
            supervised = int(selection["supervised_token_count"])
            count = int(selection["occurrence_count"])
            if parent["skill"] != skill:
                raise MixtureBuildError(f"{sample_id} is assigned to the wrong slice")
            if supervised != int(parent["supervised_token_count"]):
                raise MixtureBuildError(f"{sample_id} supervised-token count changed")
            if count <= 0:
                raise MixtureBuildError(f"{sample_id} has a non-positive occurrence count")
            slice_tokens += supervised * count
            slice_occurrences += count
            for occurrence_index in range(1, count + 1):
                occurrence = dict(parent)
                occurrence.update(
                    {
                        "canonical_sample_id": sample_id,
                        "occurrence_id": (
                            f"{mix_name}|{sample_id}|occurrence={occurrence_index}"
                        ),
                        "occurrence_index": occurrence_index,
                        "sampling_count": count,
                        "supervised_tokens_per_occurrence": supervised,
                    }
                )
                occurrences.append(occurrence)
        if object_sha256(slice_ids) != slice_proof["selection_sample_ids_sha256"]:
            raise MixtureBuildError(f"{mix_name}/{skill} selection ID hash changed")
        if slice_tokens != int(slice_proof["actual_supervised_tokens"]):
            raise MixtureBuildError(f"{mix_name}/{skill} token total changed")
        if slice_occurrences != int(slice_proof["total_occurrences"]):
            raise MixtureBuildError(f"{mix_name}/{skill} occurrence total changed")
    return sorted(occurrences, key=lambda row: occurrence_sort_key(row, seed, mix_name))


def distribution_by_skill(
    records: list[dict[str, Any]], target_ratios: dict[str, str]
) -> dict[str, dict[str, Any]]:
    total = sum(int(row["supervised_tokens_per_occurrence"]) for row in records)
    distribution: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        rows = [row for row in records if row["skill"] == skill]
        supervised = sum(int(row["supervised_tokens_per_occurrence"]) for row in rows)
        distribution[skill] = {
            "target_ratio": target_ratios[skill],
            "actual_ratio": str(Fraction(supervised, total)),
            "actual_ratio_decimal": supervised / total,
            "unique_canonical_examples": len({row["canonical_sample_id"] for row in rows}),
            "occurrences": len(rows),
            "raw_tokens": sum(int(row["raw_token_count"]) for row in rows),
            "input_tokens": sum(int(row["input_token_count"]) for row in rows),
            "supervised_tokens": supervised,
        }
    return distribution


def materialize_manifest(
    *,
    mix_name: str,
    clean_pool: dict[str, Any],
    clean_pool_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    parent_by_id = {row["sample_id"]: row for row in clean_pool["records"]}
    seed = int(config["mixtures"]["occurrence_order_seed"])
    proof = plan["mix_feasibility_proofs"][mix_name]
    records = build_occurrences(mix_name, proof, parent_by_id, seed)
    target_ratios = plan["ratios"][mix_name]
    distribution = distribution_by_skill(records, target_ratios)
    parent_header = clean_pool["header"]
    materialization = config["mixture_materialization"]
    gate_c = config["gate_c"]
    gate_c_status = gate_c["status"]
    downstream_eligible = bool(
        gate_c_status == "passed"
        or (
            gate_c_status == "waived_by_user"
            and gate_c["downstream_training_eligible_under_waiver"]
        )
    )
    artifact_status = (
        "complete"
        if gate_c_status == "passed"
        else (
            "complete_with_gate_c_waiver"
            if gate_c_status == "waived_by_user"
            else "complete_pending_gate_c"
        )
    )
    common_invariants = {
        "parent_pool_hash": parent_header["clean_pool_hash"],
        "parent_manifest_hash": parent_header["manifest_hash"],
        "parent_manifest_file_sha256": file_sha256(clean_pool_path),
        "token_budget_plan_hash": plan["plan_hash"],
        "token_budget_plan_file_sha256": file_sha256(plan_path),
        "token_budget_version": plan["token_budget_version"],
        "unit": plan["unit"],
        "common_total_supervised_tokens": plan["common_total_supervised_tokens"],
        "config_hashes": parent_header["config_hashes"],
        "source_lineage": parent_header["source_lineage"],
        "occurrence_order_seed": seed,
        "occurrence_order": materialization["occurrence_order"],
        "gate_c_decision_sha256": object_sha256(gate_c),
    }
    header = {
        "status": artifact_status,
        "manifest_name": MIX_FILENAMES[mix_name].removesuffix(".json"),
        "manifest_version": materialization["version"],
        "mix_name": mix_name,
        "source_manifest_path": relative_path(clean_pool_path),
        "source_plan_path": relative_path(plan_path),
        "common_invariants": common_invariants,
        "mixture_config_sha256": object_sha256(
            {
                "mixtures": config["mixtures"],
                "token_budget": config["token_budget"],
                "mixture_materialization": materialization,
                "gate_c": gate_c,
            }
        ),
        "target_ratios": target_ratios,
        "ratio_tolerance_absolute": plan["ratio_tolerance_absolute"],
        "total_token_tolerance_absolute": plan["total_token_tolerance_absolute"],
        "sampling_mode": plan["sampling_mode"],
        "fallback_with_replacement_used": plan["fallback_with_replacement_used"],
        "maximum_sampling_count": max(row["sampling_count"] for row in records),
        "totals": {
            "unique_canonical_examples": len(
                {row["canonical_sample_id"] for row in records}
            ),
            "occurrences": len(records),
            "raw_tokens": sum(int(row["raw_token_count"]) for row in records),
            "input_tokens": sum(int(row["input_token_count"]) for row in records),
            "supervised_tokens": sum(
                int(row["supervised_tokens_per_occurrence"]) for row in records
            ),
        },
        "distribution_by_skill": distribution,
        "occurrence_ids_sha256": object_sha256(
            [row["occurrence_id"] for row in records]
        ),
        "gate_c_status": gate_c_status,
        "gate_c_claim_boundary": gate_c["claim_boundary"],
        "downstream_training_eligible": downstream_eligible,
        "downstream_eligibility_basis": (
            "gate_c_passed"
            if gate_c_status == "passed"
            else "automated_checks_passed_and_gate_c_waived_by_user"
        ),
        "hash_definition": (
            "sha256 of canonical JSON object containing header without "
            "manifest_hash and occurrence records in frozen occurrence order"
        ),
    }
    manifest = {"header": header, "records": records}
    header["manifest_hash"] = object_sha256(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, bool]:
    header = manifest["header"]
    records = manifest["records"]
    header_without_hash = {
        key: value for key, value in header.items() if key != "manifest_hash"
    }
    total = int(header["totals"]["supervised_tokens"])
    return {
        "manifest_hash_valid": header["manifest_hash"]
        == object_sha256({"header": header_without_hash, "records": records}),
        "occurrence_ids_unique": len({row["occurrence_id"] for row in records})
        == len(records),
        "canonical_ids_preserved": all(
            row["sample_id"] == row["canonical_sample_id"] for row in records
        ),
        "occurrence_counts_recorded": all(
            int(row["sampling_count"]) >= int(row["occurrence_index"]) >= 1
            for row in records
        ),
        "supervised_total_exact": sum(
            int(row["supervised_tokens_per_occurrence"]) for row in records
        )
        == total,
        "slice_ratios_exact": all(
            Fraction(header["distribution_by_skill"][skill]["supervised_tokens"], total)
            == Fraction(header["target_ratios"][skill])
            for skill in SKILLS
        ),
        "occurrence_order_hash_valid": header["occurrence_ids_sha256"]
        == object_sha256([row["occurrence_id"] for row in records]),
    }


def validate_pair(
    manifest_a: dict[str, Any], manifest_b: dict[str, Any]
) -> dict[str, bool]:
    checks_a = validate_manifest(manifest_a)
    checks_b = validate_manifest(manifest_b)
    header_a = manifest_a["header"]
    header_b = manifest_b["header"]
    checks = {
        "mix_A_manifest_valid": all(checks_a.values()),
        "mix_B_manifest_valid": all(checks_b.values()),
        "actual_total_supervised_tokens_equal": header_a["totals"]["supervised_tokens"]
        == header_b["totals"]["supervised_tokens"],
        "parent_pool_hash_equal": header_a["common_invariants"]["parent_pool_hash"]
        == header_b["common_invariants"]["parent_pool_hash"],
        "preprocessing_hash_equal": header_a["common_invariants"]["config_hashes"][
            "preprocessing_config_sha256"
        ]
        == header_b["common_invariants"]["config_hashes"][
            "preprocessing_config_sha256"
        ],
        "filter_hash_equal": header_a["common_invariants"]["config_hashes"][
            "quality_filter_config_sha256"
        ]
        == header_b["common_invariants"]["config_hashes"][
            "quality_filter_config_sha256"
        ],
        "decontamination_hash_equal": header_a["common_invariants"]["config_hashes"][
            "decontamination_config_sha256"
        ]
        == header_b["common_invariants"]["config_hashes"][
            "decontamination_config_sha256"
        ],
        "all_common_invariants_equal": header_a["common_invariants"]
        == header_b["common_invariants"],
        "target_ratios_differ": header_a["target_ratios"] != header_b["target_ratios"],
        "manifest_hashes_differ": header_a["manifest_hash"] != header_b["manifest_hash"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise MixtureBuildError(f"A/B mixture assertions failed: {failed}")
    return checks


def render_report_section(summary: dict[str, Any]) -> str:
    a = summary["manifests"]["mix_A_balanced"]
    b = summary["manifests"]["mix_B_targeted"]
    lines = [
        REPORT_START,
        "",
        "## Step 12 materializes two equal-token mixture manifests",
        "",
        f"Mix A and Mix B each contain {a['supervised_tokens']:,} supervised tokens. "
        "Both are direct child manifests of the same clean parent pool and Step 11 "
        "plan. All frozen preprocessing, filtering, deduplication, decontamination, "
        "source-lineage, ordering, and total-token invariants match; only the target "
        "slice ratios and resulting occurrences differ.",
        "",
        "| Manifest | Unique examples | Occurrences | Supervised tokens | Manifest hash |",
        "|---|---:|---:|---:|---|",
        f"| Mix A balanced | {a['unique_canonical_examples']:,} | {a['occurrences']:,} | "
        f"{a['supervised_tokens']:,} | `{a['manifest_hash']}` |",
        f"| Mix B targeted | {b['unique_canonical_examples']:,} | {b['occurrences']:,} | "
        f"{b['supervised_tokens']:,} | `{b['manifest_hash']}` |",
        "",
        "No replacement was needed: maximum sampling_count is 1 in both manifests. "
        "Canonical sample_id values are preserved and occurrence_id is stored separately. "
        "Gate C was explicitly waived by the user rather than passed: 0 of 60 planned "
        "occurrences were manually reviewed. The manifests are downstream-eligible only "
        "under that documented risk acceptance.",
        "",
        f"- Mix A file SHA-256: `{a['file_sha256']}`",
        f"- Mix B file SHA-256: `{b['file_sha256']}`",
        f"- Step 12 summary hash: `{summary['summary_hash']}`",
        "",
        REPORT_END,
        "",
    ]
    return "\n".join(lines)


def update_report(report_path: Path, summary: dict[str, Any]) -> None:
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise MixtureBuildError("Step 12 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pool", type=Path, default=DEFAULT_CLEAN_POOL)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--allow-verification-output",
        action="store_true",
        help="allow a non-frozen output directory for Step 13 hash verification",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    clean_pool = load_json(args.clean_pool)
    plan = load_json(args.plan)
    validate_clean_pool(clean_pool)
    validate_plan(plan, args.plan, clean_pool, args.clean_pool, config)

    manifests = {
        mix_name: materialize_manifest(
            mix_name=mix_name,
            clean_pool=clean_pool,
            clean_pool_path=args.clean_pool,
            plan=plan,
            plan_path=args.plan,
            config=config,
        )
        for mix_name in MIX_FILENAMES
    }
    assertions = validate_pair(
        manifests["mix_A_balanced"], manifests["mix_B_targeted"]
    )
    output_paths: dict[str, Path] = {}
    for mix_name, manifest in manifests.items():
        output_path = args.output_dir / MIX_FILENAMES[mix_name]
        configured_path_key = (
            "mix_A_output_path" if mix_name == "mix_A_balanced" else "mix_B_output_path"
        )
        if not args.allow_verification_output and relative_path(output_path) != config["mixture_materialization"][
            configured_path_key
        ]:
            raise MixtureBuildError(f"{mix_name} output path differs from frozen config")
        write_json(output_path, manifest)
        output_paths[mix_name] = output_path

    summary = {
        "status": manifests["mix_A_balanced"]["header"]["status"],
        "gate_c": config["gate_c"],
        "parent_pool_hash": clean_pool["header"]["clean_pool_hash"],
        "parent_manifest_hash": clean_pool["header"]["manifest_hash"],
        "token_budget_plan_hash": plan["plan_hash"],
        "common_total_supervised_tokens": plan["common_total_supervised_tokens"],
        "manifests": {
            mix_name: {
                "path": relative_path(output_paths[mix_name]),
                "file_sha256": file_sha256(output_paths[mix_name]),
                "manifest_hash": manifest["header"]["manifest_hash"],
                **manifest["header"]["totals"],
                "target_ratios": manifest["header"]["target_ratios"],
                "maximum_sampling_count": manifest["header"]["maximum_sampling_count"],
                "downstream_training_eligible": manifest["header"][
                    "downstream_training_eligible"
                ],
            }
            for mix_name, manifest in manifests.items()
        },
        "assertions": assertions,
    }
    summary["summary_hash"] = object_sha256(summary)
    summary_path = args.work_dir / "mixture-summary.json"
    write_json(summary_path, summary)
    update_report(args.report, summary)
    print(
        f"mix_A_tokens={summary['manifests']['mix_A_balanced']['supervised_tokens']} "
        f"mix_B_tokens={summary['manifests']['mix_B_targeted']['supervised_tokens']} "
        f"assertions={sum(assertions.values())}/{len(assertions)} "
        f"summary_hash={summary['summary_hash']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
