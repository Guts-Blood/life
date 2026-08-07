#!/usr/bin/env python3
"""Materialize the Day 12 finance-reduced recovery mixture and Run C schedule."""

from __future__ import annotations

import argparse
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
DAY09_DIR = BOOTCAMP_ROOT / "day-09-data-quality-mixture-lineage"
CONFIG_DIR = BOOTCAMP_ROOT / "artifacts" / "configs"
DATA_DIR = BOOTCAMP_ROOT / "artifacts" / "data"
REPORT_DIR = BOOTCAMP_ROOT / "artifacts" / "reports"

DEFAULT_SPEC = CONFIG_DIR / "day12-mix-C-finance-reduced-spec.json"
DEFAULT_ASSIGNMENT = DAY09_DIR / "day09_assignment.json"
DEFAULT_BASE_CONFIG = CONFIG_DIR / "day12-controlled-sft-base.yaml"
DEFAULT_PLAN = DATA_DIR / "day12-mix-C-finance-reduced-plan.json"
DEFAULT_MANIFEST = DATA_DIR / "day12-mix-C-finance-reduced.json"
DEFAULT_CONFIG = CONFIG_DIR / "day12-controlled-sft-C.yaml"
DEFAULT_SCHEDULE = DATA_DIR / "day12-training-schedule-C.json"
DEFAULT_SUMMARY = REPORT_DIR / "day12-mix-C-preparation.json"
SKILLS = ("general", "math", "code", "finance")


class MixCPreparationError(ValueError):
    """The recovery mix cannot be materialized under the frozen data contract."""


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MixCPreparationError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DAY09_BUDGET = load_module(
    "day09_budget_for_mix_c", DAY09_DIR / "plan_day09_token_budget.py"
)
DAY09_MIX = load_module(
    "day09_materializer_for_mix_c", DAY09_DIR / "build_day09_mixtures.py"
)
DAY12 = load_module("day12_preparation_for_mix_c", HERE / "prepare_day12_experiment.py")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise MixCPreparationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MixCPreparationError(f"JSON root must be an object: {path}")
    return value


def write_json_stable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise MixCPreparationError(f"refusing to overwrite different artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def write_text_stable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise MixCPreparationError(f"refusing to overwrite different artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def resolve_repo_path(value: str) -> Path:
    return DAY12.resolve_project_path(value)


def configured_path(path: Path) -> str:
    return DAY12.canonical_project_path(path)


def configure_day09_canonical_paths() -> None:
    # Day 09's helper derives provenance from its physical repository root.  The
    # AutoDL bundle is intentionally stripped, so normalize only this imported
    # helper to Day 12's stable logical project path before materialization.
    DAY09_MIX.relative_path = configured_path


def validate_spec(spec: dict[str, Any]) -> dict[str, Fraction]:
    if spec.get("run_id") != "C" or spec.get("mix_name") != "mix_C_finance_reduced":
        raise MixCPreparationError("Run C identity changed")
    ratios = {skill: Fraction(spec["target_ratios"][skill]) for skill in SKILLS}
    if sum(ratios.values(), Fraction()) != 1 or any(value <= 0 for value in ratios.values()):
        raise MixCPreparationError("target ratios must be positive and sum to one")
    total = int(spec["total_supervised_tokens"])
    if any((total * value).denominator != 1 for value in ratios.values()):
        raise MixCPreparationError("token budget does not exactly divide by target ratios")
    return ratios


def build_plan(
    spec: dict[str, Any], clean_pool: dict[str, Any], clean_pool_path: Path
) -> dict[str, Any]:
    DAY09_MIX.validate_clean_pool(clean_pool)
    ratios = validate_spec(spec)
    total = int(spec["total_supervised_tokens"])
    grouped = DAY09_BUDGET.group_records(clean_pool["records"])
    targets = {skill: int(total * ratios[skill]) for skill in SKILLS}
    proof = DAY09_BUDGET.build_mix_proof(
        mix_name=spec["mix_name"],
        targets=targets,
        grouped=grouped,
        version=spec["selection_version"],
    )
    if int(proof["actual_total_supervised_tokens"]) != total:
        raise MixCPreparationError("materialized proof does not preserve total tokens")

    plan: dict[str, Any] = {
        "status": "complete_frozen",
        "token_budget_version": spec["selection_version"],
        "unit": "post-template/post-mask/post-truncation supervised tokens",
        "objective": "fixed Day 12 total with finance halved and released tokens distributed equally across general, math, and code",
        "source_manifest": {
            "path": configured_path(clean_pool_path),
            "file_sha256": DAY09_MIX.file_sha256(clean_pool_path),
            "manifest_hash": clean_pool["header"]["manifest_hash"],
            "clean_pool_hash": clean_pool["header"]["clean_pool_hash"],
            "total_clean_supervised_tokens": clean_pool["header"]["totals"][
                "supervised_tokens"
            ],
        },
        "available_supervised_tokens_by_slice": {
            skill: sum(int(row["supervised_token_count"]) for row in rows)
            for skill, rows in grouped.items()
        },
        "ratios": {
            spec["mix_name"]: {skill: str(ratios[skill]) for skill in SKILLS}
        },
        "ratio_tolerance_absolute": 0.0,
        "total_token_tolerance_absolute": 0,
        "common_total_supervised_tokens": total,
        "sampling_mode": "without_replacement",
        "fallback_with_replacement_used": False,
        "mix_feasibility_proofs": {spec["mix_name"]: proof},
        "validation_checks": {
            "total_exact": proof["actual_total_supervised_tokens"] == total,
            "slice_targets_exact": all(
                proof["slices"][skill]["actual_supervised_tokens"] == targets[skill]
                for skill in SKILLS
            ),
            "ratios_exact": all(
                Fraction(proof["slices"][skill]["actual_supervised_tokens"], total)
                == ratios[skill]
                for skill in SKILLS
            ),
            "without_replacement": all(
                proof["slices"][skill]["maximum_occurrences_per_sample"] == 1
                and not proof["slices"][skill]["with_replacement"]
                for skill in SKILLS
            ),
        },
    }
    if not all(plan["validation_checks"].values()):
        raise MixCPreparationError("one or more plan validation checks failed")
    plan["plan_hash"] = DAY09_MIX.object_sha256(plan)
    return plan


def materialization_config(
    assignment: dict[str, Any], spec: dict[str, Any], plan_path: Path
) -> dict[str, Any]:
    config = copy.deepcopy(assignment)
    config["mixtures"][spec["mix_name"]] = copy.deepcopy(spec["target_ratios"])
    config["token_budget"]["version"] = spec["selection_version"]
    config["token_budget"]["budget_plan_path"] = configured_path(plan_path)
    config["mixture_materialization"]["version"] = "day12_recovery_mix_manifest_v1"
    config["mixture_materialization"]["source_plan_path"] = configured_path(plan_path)
    return config


def render_run_config(
    *, base_config_path: Path, config_path: Path, manifest_path: Path, manifest: dict[str, Any]
) -> str:
    extends = os.path.relpath(base_config_path.resolve(), config_path.resolve().parent)
    header = manifest["header"]
    return (
        f"extends: {extends}\n"
        "run:\n"
        "  id: C\n"
        "data:\n"
        f"  manifest_path: {configured_path(manifest_path)}\n"
        f"  manifest_file_sha256: {DAY09_MIX.file_sha256(manifest_path)}\n"
        f"  manifest_hash: {header['manifest_hash']}\n"
        f"  mix_name: {header['mix_name']}\n"
    )


def prepare(
    *,
    spec_path: Path = DEFAULT_SPEC,
    assignment_path: Path = DEFAULT_ASSIGNMENT,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    plan_path: Path = DEFAULT_PLAN,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    schedule_path: Path = DEFAULT_SCHEDULE,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    configure_day09_canonical_paths()
    spec = load_json(spec_path)
    clean_pool_path = resolve_repo_path(spec["parent_manifest_path"])
    clean_pool = load_json(clean_pool_path)
    plan = build_plan(spec, clean_pool, clean_pool_path)
    write_json_stable(plan_path, plan)

    assignment = load_json(assignment_path)
    config = materialization_config(assignment, spec, plan_path)
    DAY09_MIX.MIX_FILENAMES[spec["mix_name"]] = manifest_path.name
    manifest = DAY09_MIX.materialize_manifest(
        mix_name=spec["mix_name"],
        clean_pool=clean_pool,
        clean_pool_path=clean_pool_path,
        plan=plan,
        plan_path=plan_path,
        config=config,
    )
    manifest_checks = DAY09_MIX.validate_manifest(manifest)
    if not all(manifest_checks.values()):
        failed = [name for name, passed in manifest_checks.items() if not passed]
        raise MixCPreparationError(f"manifest validation failed: {failed}")
    write_json_stable(manifest_path, manifest)

    write_text_stable(
        config_path,
        render_run_config(
            base_config_path=base_config_path,
            config_path=config_path,
            manifest_path=manifest_path,
            manifest=manifest,
        ),
    )
    resolved_config = DAY12.load_resolved_config(config_path)
    verified_manifest = DAY12.verify_mixture(resolved_config)
    schedule = DAY12.build_schedule(resolved_config, verified_manifest)
    DAY12.verify_schedule(schedule)
    write_json_stable(schedule_path, schedule)

    summary = {
        "status": "day12_mix_c_preparation_pass",
        "comparison_boundary": spec["comparison_boundary"],
        "hypothesis": spec["hypothesis"],
        "first_gate": spec["first_gate"],
        "target_ratios": manifest["header"]["target_ratios"],
        "distribution_by_skill": manifest["header"]["distribution_by_skill"],
        "total_supervised_tokens": manifest["header"]["totals"]["supervised_tokens"],
        "manifest_path": configured_path(manifest_path),
        "manifest_file_sha256": DAY09_MIX.file_sha256(manifest_path),
        "manifest_hash": manifest["header"]["manifest_hash"],
        "config_path": configured_path(config_path),
        "config_file_sha256": DAY09_MIX.file_sha256(config_path),
        "schedule_path": configured_path(schedule_path),
        "schedule_file_sha256": DAY09_MIX.file_sha256(schedule_path),
        "schedule_hash": schedule["header"]["schedule_hash"],
        "validation_checks": {
            **plan["validation_checks"],
            **manifest_checks,
            "schedule_total_exact": schedule["segments"]["100_percent"][
                "cumulative_supervised_tokens"
            ]
            == int(spec["total_supervised_tokens"]),
        },
    }
    summary["summary_hash"] = DAY09_MIX.object_sha256(summary)
    write_json_stable(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    try:
        summary = prepare(
            spec_path=args.spec,
            assignment_path=args.assignment,
            base_config_path=args.base_config,
            plan_path=args.plan,
            manifest_path=args.manifest,
            config_path=args.config,
            schedule_path=args.schedule,
            summary_path=args.summary,
        )
    except (MixCPreparationError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
