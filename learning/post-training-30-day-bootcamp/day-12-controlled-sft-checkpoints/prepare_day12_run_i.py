#!/usr/bin/env python3
"""Build the math/code-focused, format-repaired Run I mixture and schedule."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
CONFIG_DIR = BOOTCAMP_ROOT / "artifacts" / "configs"
DATA_DIR = BOOTCAMP_ROOT / "artifacts" / "data"
REPORT_DIR = BOOTCAMP_ROOT / "artifacts" / "reports"
DEFAULT_SPEC = CONFIG_DIR / "day12-rollout-I-spec.json"
DEFAULT_BASE_CONFIG = CONFIG_DIR / "day12-controlled-sft-H-base.yaml"
DEFAULT_PLAN = DATA_DIR / "day12-mix-I-math-code-focused-plan.json"
DEFAULT_MANIFEST = DATA_DIR / "day12-mix-I-math-code-focused.json"
DEFAULT_CONFIG = CONFIG_DIR / "day12-controlled-sft-I.yaml"
DEFAULT_SCHEDULE = DATA_DIR / "day12-training-schedule-I.json"
DEFAULT_SUMMARY = REPORT_DIR / "day12-rollout-I-preparation.json"
SKILLS = ("general", "math", "code", "finance")
CHECKPOINTS = ("25_percent", "60_percent", "100_percent")


class RunIPreparationError(ValueError):
    """Run I cannot satisfy its single-variable mixture contract."""


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunIPreparationError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_D = load_module("run_d_helpers_for_run_i", HERE / "prepare_day12_run_d.py")
DAY12 = RUN_D.DAY12
DAY09_MIX = RUN_D.DAY09_MIX


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunIPreparationError(f"JSON root must be an object: {path}")
    return value


def configured_path(path: Path) -> str:
    return DAY12.canonical_project_path(path)


def validate_spec(spec: dict[str, Any]) -> dict[str, Fraction]:
    if spec.get("run_id") != "I" or spec.get("parent_run_id") != "H":
        raise RunIPreparationError("Run I identity changed")
    if spec.get("mix_name") != "mix_I_math_code_focused":
        raise RunIPreparationError("Run I mix identity changed")
    ratios = {skill: Fraction(spec["target_ratios"][skill]) for skill in SKILLS}
    if any(value <= 0 for value in ratios.values()) or sum(ratios.values(), Fraction()) != 1:
        raise RunIPreparationError("target ratios must be positive and sum to one")
    total = int(spec["total_supervised_tokens"])
    if any((total * value).denominator != 1 for value in ratios.values()):
        raise RunIPreparationError("full token budget must divide exactly by target ratios")
    if set(spec["routing_suffixes"]) != set(SKILLS):
        raise RunIPreparationError("routing suffixes must cover all skills")
    return ratios


def transform_pools(
    clean_pool: dict[str, Any], tokenizer: Any, spec: dict[str, Any], max_length: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    pools = {skill: [] for skill in SKILLS}
    skipped = {skill: 0 for skill in SKILLS}
    for row in clean_pool["records"]:
        try:
            transformed = RUN_D.transform_row(
                row, tokenizer, spec["routing_suffixes"], max_length
            )
        except ValueError as error:
            if "truncation" not in str(error) and "truncated" not in str(error):
                raise
            skipped[row["skill"]] += 1
            continue
        pools[row["skill"]].append(transformed)
    return pools, skipped


def build_manifest(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
    parent: dict[str, Any],
    parent_path: Path,
    selected: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    manifest = RUN_D.build_manifest(
        spec=spec,
        spec_path=spec_path,
        plan=plan,
        plan_path=plan_path,
        parent=parent,
        parent_path=parent_path,
        selected=selected,
    )
    header = manifest["header"]
    header["manifest_version"] = "day12_math_code_focused_manifest_v1"
    header["downstream_eligibility_basis"] = (
        "parent_gate_c_waiver_plus_same_format_repair_plus_exact_ratio_rebalance"
    )
    header["format_repair"]["nonfinance_canonical_selection_preserved"] = False
    header["format_repair"]["selection_rebalanced_across_all_skills"] = True
    header["mixture_rebalance"] = {
        "version": spec["selection_version"],
        "parent_manifest_hash": parent["header"]["manifest_hash"],
        "parent_target_ratios": parent["header"]["target_ratios"],
        "target_ratios": spec["target_ratios"],
    }
    header.pop("manifest_hash", None)
    header["manifest_hash"] = DAY09_MIX.object_sha256(
        {"header": header, "records": manifest["records"]}
    )
    return manifest


def validate_manifest(
    spec: dict[str, Any],
    manifest: dict[str, Any],
    selected: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, bool]:
    header = manifest["header"]
    records = manifest["records"]
    header_without_hash = {key: value for key, value in header.items() if key != "manifest_hash"}
    finance_outputs = [
        row["messages"][-1]["content"] for row in records if row["skill"] == "finance"
    ]
    contract = spec["finance_response_contract"]
    planned_ids = {
        f"{spec['mix_name']}|{row['sample_id']}|occurrence=1"
        for checkpoint in CHECKPOINTS
        for skill in SKILLS
        for row in selected[checkpoint][skill]
    }
    total = int(spec["total_supervised_tokens"])
    return {
        "manifest_hash_valid": header["manifest_hash"]
        == DAY09_MIX.object_sha256({"header": header_without_hash, "records": records}),
        "occurrence_ids_unique": len({row["occurrence_id"] for row in records}) == len(records),
        "canonical_ids_unique": len({row["canonical_sample_id"] for row in records}) == len(records),
        "planned_ids_cover_manifest": planned_ids == {row["occurrence_id"] for row in records},
        "supervised_total_exact": header["totals"]["supervised_tokens"] == total,
        "slice_ratios_exact": all(
            Fraction(header["distribution_by_skill"][skill]["supervised_tokens"], total)
            == Fraction(spec["target_ratios"][skill])
            for skill in SKILLS
        ),
        "routing_suffixes_present": all(
            any(
                message["role"] == "user"
                and message["content"].endswith(spec["routing_suffixes"][row["skill"]])
                for message in row["messages"]
            )
            for row in records
        ),
        "finance_natural_reasoning_prefix": all(
            output.startswith(contract["prefix"] + "\n") for output in finance_outputs
        ),
        "finance_final_answer_present": all(
            "\n" + contract["final_answer_prefix"] + " " in output
            for output in finance_outputs
        ),
        "finance_dsl_removed": all(
            fragment not in output
            for output in finance_outputs
            for fragment in contract["forbidden_fragments"]
        ),
        "no_truncation": not any(row["truncated"] for row in records),
        "maximum_length_respected": max(row["input_token_count"] for row in records) <= 2048,
    }


def render_run_config(
    base_config_path: Path,
    config_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> str:
    extends = os.path.relpath(base_config_path.resolve(), config_path.resolve().parent)
    header = manifest["header"]
    return (
        f"extends: {extends}\n"
        "run:\n  id: I\n"
        "data:\n"
        f"  manifest_path: {configured_path(manifest_path)}\n"
        f"  manifest_file_sha256: {DAY09_MIX.file_sha256(manifest_path)}\n"
        f"  manifest_hash: {header['manifest_hash']}\n"
        f"  mix_name: {header['mix_name']}\n"
    )


def prepare(
    *,
    spec_path: Path = DEFAULT_SPEC,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    plan_path: Path = DEFAULT_PLAN,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    schedule_path: Path = DEFAULT_SCHEDULE,
    summary_path: Path = DEFAULT_SUMMARY,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    ratios = validate_spec(spec)
    parent_path = DAY12.resolve_project_path(spec["parent_manifest_path"])
    clean_pool_path = DAY12.resolve_project_path(spec["clean_pool_path"])
    parent = load_json(parent_path)
    clean_pool = load_json(clean_pool_path)
    DAY09_MIX.validate_clean_pool(clean_pool)
    tokenizer = RUN_D.load_tokenizer(tokenizer_path or RUN_D.default_tokenizer_path())
    base_config = DAY12.load_yaml(base_config_path)
    pools, skipped = transform_pools(
        clean_pool, tokenizer, spec, int(base_config["data"]["max_length"])
    )
    full_targets = {
        skill: int(int(spec["total_supervised_tokens"]) * ratios[skill])
        for skill in SKILLS
    }
    available = {
        skill: sum(int(row["supervised_token_count"]) for row in pools[skill])
        for skill in SKILLS
    }
    for skill in SKILLS:
        if available[skill] < full_targets[skill]:
            raise RunIPreparationError(
                f"format-repaired {skill} pool has {available[skill]} tokens, below {full_targets[skill]}"
            )

    budgets = {
        key: int(value) for key, value in base_config["training"]["checkpoint_budgets"].items()
    }
    increments = RUN_D.checkpoint_skill_targets(budgets, spec["target_ratios"])
    selected = RUN_D.select_stage_subsets(pools, increments, spec)
    plan = RUN_D.build_plan(
        spec=spec,
        spec_path=spec_path,
        parent=parent,
        parent_path=parent_path,
        clean_pool=clean_pool,
        clean_pool_path=clean_pool_path,
        selected=selected,
        increments=increments,
    )
    RUN_D.write_json_stable(plan_path, plan)
    manifest = build_manifest(
        spec=spec,
        spec_path=spec_path,
        plan=plan,
        plan_path=plan_path,
        parent=parent,
        parent_path=parent_path,
        selected=selected,
    )
    validation = validate_manifest(spec, manifest, selected)
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise RunIPreparationError(f"Run I manifest validation failed: {failed}")
    RUN_D.write_json_stable(manifest_path, manifest)
    RUN_D.write_text_stable(
        config_path,
        render_run_config(base_config_path, config_path, manifest_path, manifest),
    )
    resolved = DAY12.load_resolved_config(config_path)
    verified = DAY12.verify_mixture(resolved)
    schedule = DAY12.build_schedule(resolved, verified)
    DAY12.verify_schedule(schedule)
    RUN_D.write_json_stable(schedule_path, schedule)

    summary: dict[str, Any] = {
        "status": "day12_run_i_preparation_pass",
        "ordinal": spec["ordinal"],
        "run_id": spec["run_id"],
        "parent_run_id": spec["parent_run_id"],
        "hypothesis": spec["hypothesis"],
        "primary_change": spec["primary_change"],
        "target_ratios": spec["target_ratios"],
        "target_supervised_tokens": full_targets,
        "available_supervised_tokens": available,
        "truncation_filtered_records": skipped,
        "distribution_by_skill": manifest["header"]["distribution_by_skill"],
        "total_supervised_tokens": manifest["header"]["totals"]["supervised_tokens"],
        "maximum_input_tokens": max(row["input_token_count"] for row in manifest["records"]),
        "plan_path": configured_path(plan_path),
        "plan_file_sha256": DAY09_MIX.file_sha256(plan_path),
        "plan_hash": plan["plan_hash"],
        "manifest_path": configured_path(manifest_path),
        "manifest_file_sha256": DAY09_MIX.file_sha256(manifest_path),
        "manifest_hash": manifest["header"]["manifest_hash"],
        "config_path": configured_path(config_path),
        "config_file_sha256": DAY09_MIX.file_sha256(config_path),
        "config_hash": DAY12.object_sha256(resolved),
        "schedule_path": configured_path(schedule_path),
        "schedule_file_sha256": DAY09_MIX.file_sha256(schedule_path),
        "schedule_hash": schedule["header"]["schedule_hash"],
        "validation_checks": {
            **validation,
            "optimizer_reused_from_H": resolved["optimizer"]
            == DAY12.load_yaml(DEFAULT_BASE_CONFIG)["optimizer"],
            "schedule_total_exact": schedule["segments"]["100_percent"]
            ["cumulative_supervised_tokens"]
            == int(spec["total_supervised_tokens"]),
        },
    }
    if not all(summary["validation_checks"].values()):
        raise RunIPreparationError("Run I summary validation failed")
    summary["summary_hash"] = DAY09_MIX.object_sha256(summary)
    RUN_D.write_json_stable(summary_path, summary)
    return summary


def main() -> None:
    try:
        summary = prepare()
    except (RunIPreparationError, RUN_D.RunDPreparationError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
