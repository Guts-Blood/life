#!/usr/bin/env python3
"""Find and prove a common supervised-token budget for Day 09 Mix A and B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_MANIFEST = HERE.parent / "artifacts" / "data" / "day09-dataset-manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "tmp" / "day09-work" / "step11-token-budget"
DEFAULT_REPORT = HERE.parent / "artifacts" / "reports" / "day09-data-quality-audit.md"
REPORT_START = "<!-- STEP11_TOKEN_BUDGET_START -->"
REPORT_END = "<!-- STEP11_TOKEN_BUDGET_END -->"
SKILLS = ("general", "math", "code", "finance")


class TokenBudgetError(ValueError):
    """The clean manifest or common-budget feasibility proof is invalid."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_ratios(config: dict[str, Any]) -> dict[str, dict[str, Fraction]]:
    mixtures = config["mixtures"]
    parsed = {
        "mix_A_balanced": {
            skill: Fraction(mixtures["mix_A_balanced"][skill]) for skill in SKILLS
        },
        "mix_B_targeted": {
            skill: Fraction(mixtures["mix_B_targeted"][skill]) for skill in SKILLS
        },
    }
    for name, ratios in parsed.items():
        if sum(ratios.values(), Fraction(0, 1)) != Fraction(1, 1):
            raise TokenBudgetError(f"{name} ratios do not sum to one")
        if any(value <= 0 for value in ratios.values()):
            raise TokenBudgetError(f"{name} ratios must be positive")
    return parsed


def validate_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    header = manifest["header"]
    records = manifest["records"]
    stored_hash = header["manifest_hash"]
    header_without_hash = {
        key: value for key, value in header.items() if key != "manifest_hash"
    }
    if stored_hash != object_sha256(
        {"header": header_without_hash, "records": records}
    ):
        raise TokenBudgetError("clean manifest hash is invalid")
    if header["clean_pool_hash"] != object_sha256(
        sorted(records, key=lambda item: item["sample_id"])
    ):
        raise TokenBudgetError("clean pool hash is invalid")
    if header["manifest_version"] != config["clean_pool"]["version"]:
        raise TokenBudgetError("clean manifest version does not match config")
    ids = [row["sample_id"] for row in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise TokenBudgetError("clean manifest IDs are not unique and sorted")
    if any(int(row["supervised_token_count"]) <= 0 for row in records):
        raise TokenBudgetError("clean manifest contains non-positive supervision")


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["skill"]].append(row)
    if set(grouped) != set(SKILLS):
        raise TokenBudgetError("clean manifest does not contain all four slices")
    return {skill: grouped[skill] for skill in SKILLS}


def reachable_bitset(weights: list[int], maximum: int) -> int:
    if maximum < 0:
        raise TokenBudgetError("maximum subset target must be non-negative")
    bits = 1
    mask = (1 << (maximum + 1)) - 1
    for weight in weights:
        if weight <= 0:
            raise TokenBudgetError("subset weights must be positive")
        if weight <= maximum:
            bits = (bits | (bits << weight)) & mask
    return bits


def is_reachable(bits: int, target: int) -> bool:
    return target >= 0 and bool((bits >> target) & 1)


def stable_order(
    rows: list[dict[str, Any]], version: str, mix: str, skill: str
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        payload = "\0".join((version, mix, skill, row["sample_id"]))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), row["sample_id"]

    return sorted(rows, key=key)


def reconstruct_subset(
    rows: list[dict[str, Any]], target: int
) -> list[dict[str, Any]]:
    if target < 0:
        raise TokenBudgetError("subset target must be non-negative")
    eligible = [
        row for row in rows if int(row["supervised_token_count"]) <= target
    ]
    mask = (1 << (target + 1)) - 1
    prefixes = [1]
    for row in eligible:
        weight = int(row["supervised_token_count"])
        prefixes.append((prefixes[-1] | (prefixes[-1] << weight)) & mask)
    if not is_reachable(prefixes[-1], target):
        raise TokenBudgetError(f"subset target is unreachable: {target}")

    selected: list[dict[str, Any]] = []
    remaining = target
    for index in range(len(eligible), 0, -1):
        if is_reachable(prefixes[index - 1], remaining):
            continue
        row = eligible[index - 1]
        selected.append(row)
        remaining -= int(row["supervised_token_count"])
    if remaining != 0:
        raise TokenBudgetError("subset reconstruction did not reach zero")
    selected.reverse()
    return selected


def ratio_denominator(ratios: dict[str, dict[str, Fraction]]) -> int:
    denominator = 1
    for mix in ratios.values():
        for value in mix.values():
            denominator = math.lcm(denominator, value.denominator)
    return denominator


def exact_targets(
    total: int, ratios: dict[str, dict[str, Fraction]]
) -> dict[str, dict[str, int]] | None:
    targets: dict[str, dict[str, int]] = {}
    for mix_name, mix_ratios in ratios.items():
        targets[mix_name] = {}
        for skill, ratio in mix_ratios.items():
            value = total * ratio
            if value.denominator != 1:
                return None
            targets[mix_name][skill] = value.numerator
    return targets


def theoretical_upper_total(
    available: dict[str, int], ratios: dict[str, dict[str, Fraction]], denominator: int
) -> tuple[int, list[dict[str, Any]]]:
    constraints: list[dict[str, Any]] = []
    for mix_name, mix_ratios in ratios.items():
        for skill, ratio in mix_ratios.items():
            upper = int(Fraction(available[skill], 1) / ratio)
            constraints.append(
                {
                    "mix": mix_name,
                    "skill": skill,
                    "available_supervised_tokens": available[skill],
                    "target_ratio": str(ratio),
                    "implied_total_upper_bound": upper,
                }
            )
    raw_upper = min(item["implied_total_upper_bound"] for item in constraints)
    aligned_upper = raw_upper - (raw_upper % denominator)
    return aligned_upper, constraints


def find_common_budget(
    available: dict[str, int],
    reachability: dict[str, int],
    ratios: dict[str, dict[str, Fraction]],
    denominator: int,
    upper: int,
) -> tuple[int, dict[str, dict[str, int]]]:
    for total in range(upper, 0, -denominator):
        targets = exact_targets(total, ratios)
        if targets is None:
            continue
        if all(
            is_reachable(reachability[skill], target)
            for mix_targets in targets.values()
            for skill, target in mix_targets.items()
        ):
            return total, targets
    raise TokenBudgetError(
        "no positive exact without-replacement common budget was found"
    )


def build_mix_proof(
    *,
    mix_name: str,
    targets: dict[str, int],
    grouped: dict[str, list[dict[str, Any]]],
    version: str,
) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    total = 0
    for skill in SKILLS:
        ordered = stable_order(grouped[skill], version, mix_name, skill)
        selected = reconstruct_subset(ordered, targets[skill])
        actual = sum(int(row["supervised_token_count"]) for row in selected)
        if actual != targets[skill]:
            raise TokenBudgetError(f"{mix_name}/{skill} reconstruction mismatch")
        total += actual
        slices[skill] = {
            "target_supervised_tokens": targets[skill],
            "actual_supervised_tokens": actual,
            "difference_tokens": actual - targets[skill],
            "selected_unique_examples": len(selected),
            "total_occurrences": len(selected),
            "maximum_occurrences_per_sample": 1,
            "with_replacement": False,
            "selection_sample_ids_sha256": object_sha256(
                [row["sample_id"] for row in selected]
            ),
            "selection": [
                {
                    "sample_id": row["sample_id"],
                    "supervised_token_count": int(row["supervised_token_count"]),
                    "occurrence_count": 1,
                }
                for row in selected
            ],
        }
    return {
        "actual_total_supervised_tokens": total,
        "slices": slices,
    }


def validate_plan(
    plan: dict[str, Any], ratios: dict[str, dict[str, Fraction]]
) -> dict[str, bool]:
    total = int(plan["common_total_supervised_tokens"])
    mix_proofs = plan["mix_feasibility_proofs"]
    all_selected = [
        item
        for proof in mix_proofs.values()
        for slice_proof in proof["slices"].values()
        for item in slice_proof["selection"]
    ]
    return {
        "common_totals_equal": all(
            proof["actual_total_supervised_tokens"] == total
            for proof in mix_proofs.values()
        ),
        "slice_targets_exact": all(
            slice_proof["difference_tokens"] == 0
            for proof in mix_proofs.values()
            for slice_proof in proof["slices"].values()
        ),
        "ratios_exact": all(
            Fraction(proof["slices"][skill]["actual_supervised_tokens"], total)
            == ratios[mix_name][skill]
            for mix_name, proof in mix_proofs.items()
            for skill in SKILLS
        ),
        "whole_examples_only": all(
            item["supervised_token_count"] > 0 for item in all_selected
        ),
        "without_replacement": all(
            slice_proof["maximum_occurrences_per_sample"] == 1
            and not slice_proof["with_replacement"]
            for proof in mix_proofs.values()
            for slice_proof in proof["slices"].values()
        ),
        "occurrences_recorded": all(
            item["occurrence_count"] == 1 for item in all_selected
        ),
        "positive_budget": total > 0,
    }


def build_plan(
    config: dict[str, Any], manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    validate_manifest(manifest, config)
    budget_config = config["token_budget"]
    ratios = parse_ratios(config)
    denominator = ratio_denominator(ratios)
    if denominator != int(budget_config["base_budget_denominator"]):
        raise TokenBudgetError("configured budget denominator is inconsistent")
    grouped = group_records(manifest["records"])
    available = {
        skill: sum(int(row["supervised_token_count"]) for row in rows)
        for skill, rows in grouped.items()
    }
    upper, constraints = theoretical_upper_total(available, ratios, denominator)
    upper_targets = exact_targets(upper, ratios)
    if upper_targets is None:
        raise TokenBudgetError("aligned upper budget does not have integer targets")
    maximum_targets = {
        skill: max(targets[skill] for targets in upper_targets.values())
        for skill in SKILLS
    }
    reachability = {
        skill: reachable_bitset(
            [int(row["supervised_token_count"]) for row in grouped[skill]],
            maximum_targets[skill],
        )
        for skill in SKILLS
    }
    total, targets = find_common_budget(
        available, reachability, ratios, denominator, upper
    )
    proofs = {
        mix_name: build_mix_proof(
            mix_name=mix_name,
            targets=mix_targets,
            grouped=grouped,
            version=budget_config["version"],
        )
        for mix_name, mix_targets in targets.items()
    }
    plan = {
        "status": "complete_frozen",
        "token_budget_version": budget_config["version"],
        "unit": budget_config["unit"],
        "objective": budget_config["objective"],
        "source_manifest": {
            "path": relative_path(manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "manifest_hash": manifest["header"]["manifest_hash"],
            "clean_pool_hash": manifest["header"]["clean_pool_hash"],
            "total_clean_supervised_tokens": manifest["header"]["totals"][
                "supervised_tokens"
            ],
        },
        "available_supervised_tokens_by_slice": available,
        "ratios": {
            mix_name: {skill: str(value) for skill, value in mix.items()}
            for mix_name, mix in ratios.items()
        },
        "ratio_tolerance_absolute": budget_config[
            "ratio_tolerance_absolute"
        ],
        "total_token_tolerance_absolute": budget_config[
            "total_token_tolerance_absolute"
        ],
        "budget_denominator": denominator,
        "theoretical_without_replacement_upper_total": upper,
        "common_total_supervised_tokens": total,
        "upper_bound_reduction_tokens": upper - total,
        "common_budget_share_of_clean_pool": total
        / int(manifest["header"]["totals"]["supervised_tokens"]),
        "bottleneck_constraints": constraints,
        "sampling_mode": "without_replacement",
        "fallback_with_replacement_used": False,
        "stable_candidate_order": budget_config["stable_candidate_order"],
        "subset_solver": budget_config["subset_solver"],
        "mix_feasibility_proofs": proofs,
    }
    validation = validate_plan(plan, ratios)
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise TokenBudgetError(f"token budget validation failed: {failed}")
    plan["validation_checks"] = validation
    plan["plan_hash"] = object_sha256(plan)
    return plan


def write_plan(output_dir: Path, plan: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "token-budget-plan.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def render_report_section(plan: dict[str, Any]) -> str:
    total = plan["common_total_supervised_tokens"]
    finance_tokens = plan["available_supervised_tokens_by_slice"]["finance"]
    lines = [
        REPORT_START,
        "",
        f"## Step 11 fixes an exact common budget of {total:,} supervised tokens",
        "",
        "The maximum whole-example, without-replacement budget jointly feasible "
        f"for Mix A and Mix B is {total:,} supervised tokens. Finance is the binding "
        f"slice: its {finance_tokens:,} available tokens must represent 25% in both mixtures. "
        "Deterministic subset-sum reconstruction reaches every slice target exactly, "
        "so no replacement or answer truncation is required.",
        "",
        "| Mix | General | Math | Code | Finance | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mix_name in ("mix_A_balanced", "mix_B_targeted"):
        proof = plan["mix_feasibility_proofs"][mix_name]
        values = [proof["slices"][skill]["actual_supervised_tokens"] for skill in SKILLS]
        lines.append(
            f"| {mix_name} | {values[0]:,} | {values[1]:,} | {values[2]:,} | "
            f"{values[3]:,} | {proof['actual_total_supervised_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            f"The common budget is {plan['common_budget_share_of_clean_pool']:.2%} "
            "of the clean pool's supervised tokens. Ratio tolerance and total-token "
            "tolerance are both zero. Occurrence counts are recorded and every "
            "selected sample has occurrence_count=1.",
            "",
            "Step 11 freezes feasibility and budget only. Step 12 will materialize "
            "the final Mix A/B manifests from these proofs.",
            "",
            "- Plan: `tmp/day09-work/step11-token-budget/token-budget-plan.json`",
            f"- Plan hash: `{plan['plan_hash']}`",
            "",
            REPORT_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_report(report_path: Path, plan: dict[str, Any]) -> None:
    text = report_path.read_text(encoding="utf-8")
    if REPORT_START in text:
        before, remainder = text.split(REPORT_START, 1)
        if REPORT_END not in remainder:
            raise TokenBudgetError("Step 11 report marker is incomplete")
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    report_path.write_text(
        text.rstrip() + "\n\n" + render_report_section(plan), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    plan = build_plan(config, manifest, args.manifest)
    path = write_plan(args.output_dir, plan)
    update_report(args.report, plan)
    print(
        f"common_supervised_token_budget={plan['common_total_supervised_tokens']} "
        f"sampling_mode={plan['sampling_mode']} plan_hash={plan['plan_hash']} "
        f"file_sha256={file_sha256(path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
