#!/usr/bin/env python3
"""Verify the single-variable low-LR Run E contract and exact schedule."""

from __future__ import annotations

import argparse
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
DEFAULT_SPEC = CONFIG_DIR / "day12-rollout-E-spec.json"
DEFAULT_PARENT_CONFIG = CONFIG_DIR / "day12-controlled-sft-D.yaml"
DEFAULT_CONFIG = CONFIG_DIR / "day12-controlled-sft-E.yaml"
DEFAULT_SCHEDULE = DATA_DIR / "day12-training-schedule-E.json"
DEFAULT_SUMMARY = REPORT_DIR / "day12-rollout-E-preparation.json"
AUTHORIZED_DIFFERENCES = {"run.id", "optimizer.learning_rate"}


class RunEPreparationError(ValueError):
    """Run E differs from Run D outside the authorized low-LR ablation."""


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunEPreparationError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DAY12 = load_module("day12_preparation_for_run_e", HERE / "prepare_day12_experiment.py")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunEPreparationError(f"JSON root must be an object: {path}")
    return value


def behavioral_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("config_source", None)
    return result


def verify_difference(
    parent: dict[str, Any], rollout: dict[str, Any], spec: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    differences = {
        path: values
        for path, values in {
            path: (DAY12.flatten(behavioral_config(parent)).get(path), value)
            for path, value in DAY12.flatten(behavioral_config(rollout)).items()
            if DAY12.flatten(behavioral_config(parent)).get(path) != value
        }.items()
    }
    if set(differences) != AUTHORIZED_DIFFERENCES:
        raise RunEPreparationError(
            f"Run E unauthorized differences: {sorted(set(differences) - AUTHORIZED_DIFFERENCES)}; "
            f"missing: {sorted(AUTHORIZED_DIFFERENCES - set(differences))}"
        )
    change = spec["primary_change"]
    if differences[change["path"]] != (
        float(change["parent_value"]),
        float(change["rollout_value"]),
    ):
        raise RunEPreparationError("Run E learning-rate values differ from its spec")
    return differences


def prepare(
    *,
    spec_path: Path = DEFAULT_SPEC,
    parent_config_path: Path = DEFAULT_PARENT_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    schedule_path: Path = DEFAULT_SCHEDULE,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    run_id = spec.get("run_id")
    parent_run_id = spec.get("parent_run_id")
    if not isinstance(run_id, str) or not isinstance(parent_run_id, str):
        raise RunEPreparationError("rollout and parent identities must be strings")
    parent = DAY12.load_resolved_config(parent_config_path)
    rollout = DAY12.load_resolved_config(config_path)
    if parent.get("run", {}).get("id") != parent_run_id:
        raise RunEPreparationError("resolved parent run identity differs from spec")
    if rollout.get("run", {}).get("id") != run_id:
        raise RunEPreparationError("resolved rollout identity differs from spec")
    differences = verify_difference(parent, rollout, spec)
    manifest = DAY12.verify_mixture(rollout)
    schedule = DAY12.build_schedule(rollout, manifest)
    DAY12.verify_schedule(schedule)
    DAY12.write_json_stable(schedule_path, schedule)
    summary: dict[str, Any] = {
        "status": f"day12_run_{run_id.lower()}_preparation_pass",
        "ordinal": spec["ordinal"],
        "run_id": spec["run_id"],
        "parent_run_id": spec["parent_run_id"],
        "hypothesis": spec["hypothesis"],
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
            "only_run_id_and_learning_rate_changed": True,
            "manifest_reused_exactly": rollout["data"]["manifest_hash"]
            == parent["data"]["manifest_hash"],
            "schedule_total_exact": schedule["segments"]["100_percent"][
                "cumulative_supervised_tokens"
            ]
            == 246936,
        },
    }
    if not all(summary["validation_checks"].values()):
        raise RunEPreparationError("Run E validation failed")
    summary["summary_hash"] = DAY12.object_sha256(summary)
    DAY12.write_json_stable(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    try:
        summary = prepare(
            spec_path=args.spec,
            parent_config_path=args.parent_config,
            config_path=args.config,
            schedule_path=args.schedule,
            summary_path=args.summary,
        )
    except (RunEPreparationError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
