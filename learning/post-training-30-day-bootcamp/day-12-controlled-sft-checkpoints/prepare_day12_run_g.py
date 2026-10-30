#!/usr/bin/env python3
"""Verify the single-variable full-parameter-to-LoRA Run G contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
CONFIG_DIR = BOOTCAMP_ROOT / "artifacts" / "configs"
DATA_DIR = BOOTCAMP_ROOT / "artifacts" / "data"
REPORT_DIR = BOOTCAMP_ROOT / "artifacts" / "reports"
DEFAULT_SPEC = CONFIG_DIR / "day12-rollout-G-spec.json"
DEFAULT_PARENT_CONFIG = CONFIG_DIR / "day12-controlled-sft-E.yaml"
DEFAULT_CONFIG = CONFIG_DIR / "day12-controlled-sft-G.yaml"
DEFAULT_SCHEDULE = DATA_DIR / "day12-training-schedule-G.json"
DEFAULT_SUMMARY = REPORT_DIR / "day12-rollout-G-preparation.json"
AUTHORIZED_DIFFERENCES = {
    "run.id",
    "adapter.type",
    "adapter.peft_version",
    "adapter.task_type",
    "adapter.rank",
    "adapter.alpha",
    "adapter.dropout",
    "adapter.target_modules",
    "adapter.bias",
}


class RunGPreparationError(ValueError):
    """Run G differs from Run E outside the LoRA parameterization."""


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunGPreparationError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DAY12 = load_module("day12_preparation_for_run_g", HERE / "prepare_day12_experiment.py")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunGPreparationError(f"JSON root must be an object: {path}")
    return value


def behavioral_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("config_source", None)
    return result


def changed_paths(parent: dict[str, Any], rollout: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    parent_flat = DAY12.flatten(behavioral_config(parent))
    rollout_flat = DAY12.flatten(behavioral_config(rollout))
    paths = sorted(set(parent_flat) | set(rollout_flat))
    return {
        path: (parent_flat.get(path), rollout_flat.get(path))
        for path in paths
        if parent_flat.get(path) != rollout_flat.get(path)
    }


def prepare(
    *,
    spec_path: Path = DEFAULT_SPEC,
    parent_config_path: Path = DEFAULT_PARENT_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    schedule_path: Path = DEFAULT_SCHEDULE,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    parent = DAY12.load_resolved_config(parent_config_path)
    rollout = DAY12.load_resolved_config(config_path)
    if parent.get("run", {}).get("id") != spec.get("parent_run_id"):
        raise RunGPreparationError("resolved parent identity differs from spec")
    if rollout.get("run", {}).get("id") != spec.get("run_id"):
        raise RunGPreparationError("resolved rollout identity differs from spec")

    differences = changed_paths(parent, rollout)
    if set(differences) != AUTHORIZED_DIFFERENCES:
        raise RunGPreparationError(
            f"Run G unauthorized differences: {sorted(set(differences) - AUTHORIZED_DIFFERENCES)}; "
            f"missing: {sorted(AUTHORIZED_DIFFERENCES - set(differences))}"
        )
    implementation = spec["primary_change"]["implementation"]
    adapter = rollout["adapter"]
    for key in ("peft_version", "target_modules", "rank", "alpha", "dropout", "bias"):
        if adapter[key] != implementation[key]:
            raise RunGPreparationError(f"adapter.{key} differs from rollout spec")
    if parent["optimizer"] != rollout["optimizer"]:
        raise RunGPreparationError("Run G must reuse the complete Run E optimizer config")

    manifest = DAY12.verify_mixture(rollout)
    schedule = DAY12.build_schedule(rollout, manifest)
    DAY12.verify_schedule(schedule)
    DAY12.write_json_stable(schedule_path, schedule)
    parent_manifest = DAY12.verify_mixture(parent)
    parent_schedule = DAY12.build_schedule(parent, parent_manifest)
    occurrence_order_reused = all(
        schedule["segments"][name]["occurrence_ids"]
        == parent_schedule["segments"][name]["occurrence_ids"]
        for name in rollout["training"]["checkpoint_order"]
    )
    summary: dict[str, Any] = {
        "status": "day12_run_g_preparation_pass",
        "ordinal": spec["ordinal"],
        "run_id": spec["run_id"],
        "parent_run_id": spec["parent_run_id"],
        "hypothesis": spec["hypothesis"],
        "primary_change": spec["primary_change"],
        "authorized_differences": {
            path: {"parent": values[0], "rollout": values[1]}
            for path, values in sorted(differences.items())
        },
        "manifest_hash": manifest["header"]["manifest_hash"],
        "config_hash": DAY12.object_sha256(rollout),
        "schedule_path": DAY12.canonical_project_path(schedule_path),
        "schedule_file_sha256": DAY12.file_sha256(schedule_path),
        "schedule_hash": schedule["header"]["schedule_hash"],
        "checkpoint_budgets": rollout["training"]["checkpoint_budgets"],
        "first_gate": spec["first_gate"],
        "validation_checks": {
            "only_identity_and_parameterization_changed": True,
            "optimizer_reused_exactly": parent["optimizer"] == rollout["optimizer"],
            "manifest_reused_exactly": parent["data"]["manifest_hash"]
            == rollout["data"]["manifest_hash"],
            "occurrence_order_reused_exactly": occurrence_order_reused,
            "schedule_total_exact": schedule["segments"]["100_percent"]
            ["cumulative_supervised_tokens"]
            == 246936,
        },
    }
    if not all(summary["validation_checks"].values()):
        raise RunGPreparationError("Run G validation failed")
    summary["summary_hash"] = DAY12.object_sha256(summary)
    DAY12.write_json_stable(summary_path, summary)
    return summary


def main() -> None:
    try:
        summary = prepare()
    except (RunGPreparationError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
