#!/usr/bin/env python3
"""Verify Run K's single-variable output-projection LoRA coverage contract."""

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
DEFAULT_SPEC = CONFIG_DIR / "day12-rollout-K-spec.json"
DEFAULT_PARENT_CONFIG = CONFIG_DIR / "day12-controlled-sft-J.yaml"
DEFAULT_CONFIG = CONFIG_DIR / "day12-controlled-sft-K.yaml"
DEFAULT_SCHEDULE = DATA_DIR / "day12-training-schedule-K.json"
DEFAULT_SUMMARY = REPORT_DIR / "day12-rollout-K-preparation.json"
AUTHORIZED_DIFFERENCES = {"run.id", "adapter.target_modules"}


class RunKPreparationError(ValueError):
    """Run K differs from Run J outside output-projection LoRA coverage."""


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunKPreparationError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_G = load_module("run_g_helpers_for_run_k", HERE / "prepare_day12_run_g.py")
DAY12 = RUN_G.DAY12


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunKPreparationError(f"JSON root must be an object: {path}")
    return value


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
        raise RunKPreparationError("resolved parent identity differs from spec")
    if rollout.get("run", {}).get("id") != spec.get("run_id"):
        raise RunKPreparationError("resolved rollout identity differs from spec")

    differences = RUN_G.changed_paths(parent, rollout)
    if set(differences) != AUTHORIZED_DIFFERENCES:
        raise RunKPreparationError(
            f"Run K unauthorized differences: {sorted(set(differences) - AUTHORIZED_DIFFERENCES)}; "
            f"missing: {sorted(AUTHORIZED_DIFFERENCES - set(differences))}"
        )
    change = spec["primary_change"]
    if differences[change["path"]] != (
        change["parent_value"],
        change["rollout_value"],
    ):
        raise RunKPreparationError("Run K target-module values differ from its spec")

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
        "status": "day12_run_k_preparation_pass",
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
            "only_identity_and_target_modules_changed": True,
            "optimizer_reused_exactly": parent["optimizer"] == rollout["optimizer"],
            "adapter_shape_reused_exactly": all(
                parent["adapter"][key] == rollout["adapter"][key]
                for key in ("type", "peft_version", "task_type", "rank", "alpha", "dropout", "bias")
            ),
            "manifest_reused_exactly": parent["data"]["manifest_hash"]
            == rollout["data"]["manifest_hash"],
            "occurrence_order_reused_exactly": occurrence_order_reused,
            "schedule_total_exact": schedule["segments"]["100_percent"]
            ["cumulative_supervised_tokens"]
            == 246936,
        },
    }
    if not all(summary["validation_checks"].values()):
        raise RunKPreparationError("Run K validation failed")
    summary["summary_hash"] = DAY12.object_sha256(summary)
    DAY12.write_json_stable(summary_path, summary)
    return summary


def main() -> None:
    try:
        summary = prepare()
    except (RunKPreparationError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
