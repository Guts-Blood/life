#!/usr/bin/env python3
"""Validate the Day 12 A/B contract and materialize exact token schedules."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
CONFIG_DIR = BOOTCAMP_ROOT / "artifacts" / "configs"
DATA_DIR = BOOTCAMP_ROOT / "artifacts" / "data"
REPORT_DIR = BOOTCAMP_ROOT / "artifacts" / "reports"
BOOTCAMP_PREFIX = Path("learning/post-training-30-day-bootcamp")
DEFAULT_A_CONFIG = CONFIG_DIR / "day12-controlled-sft-A.yaml"
DEFAULT_B_CONFIG = CONFIG_DIR / "day12-controlled-sft-B.yaml"
DEFAULT_A_SCHEDULE = DATA_DIR / "day12-training-schedule-A.json"
DEFAULT_B_SCHEDULE = DATA_DIR / "day12-training-schedule-B.json"
DEFAULT_REPORT = REPORT_DIR / "day12-config-diff.md"
SKILLS = ("general", "math", "code", "finance")
AUTHORIZED_DIFFS = {
    "run.id",
    "data.manifest_path",
    "data.manifest_file_sha256",
    "data.manifest_hash",
    "data.mix_name",
}


class Day12PreparationError(ValueError):
    """A frozen input or A/B invariant is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day12PreparationError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day12PreparationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day12PreparationError(f"JSON root must be an object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as error:
        raise Day12PreparationError(f"cannot load YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day12PreparationError(f"YAML root must be an object: {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def canonical_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(BOOTCAMP_ROOT.resolve())
    except ValueError:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    return (BOOTCAMP_PREFIX / relative).as_posix()


def load_resolved_config(path: Path) -> dict[str, Any]:
    override = load_yaml(path.resolve())
    extends = override.pop("extends", None)
    if not isinstance(extends, str) or not extends:
        raise Day12PreparationError(f"config must extend a base file: {path}")
    unexpected = set(override) - {"run", "data"}
    if unexpected:
        raise Day12PreparationError(f"override has unauthorized sections: {sorted(unexpected)}")
    base_path = path.resolve().parent / extends
    resolved = deep_merge(load_yaml(base_path), override)
    resolved["config_source"] = {
        "base_path": canonical_project_path(base_path),
        "override_path": canonical_project_path(path.resolve()),
        "base_file_sha256": file_sha256(base_path),
        "override_file_sha256": file_sha256(path.resolve()),
    }
    return resolved


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        result.update(flatten(child, path))
    return result


def verify_ab_diff(config_a: dict[str, Any], config_b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    comparable_a = copy.deepcopy(config_a)
    comparable_b = copy.deepcopy(config_b)
    comparable_a.pop("config_source", None)
    comparable_b.pop("config_source", None)
    flat_a = flatten(comparable_a)
    flat_b = flatten(comparable_b)
    paths = sorted(set(flat_a) | set(flat_b))
    differences = {
        path: (flat_a.get(path), flat_b.get(path))
        for path in paths
        if flat_a.get(path) != flat_b.get(path)
    }
    unauthorized = set(differences) - AUTHORIZED_DIFFS
    missing = AUTHORIZED_DIFFS - set(differences)
    if unauthorized or missing:
        raise Day12PreparationError(
            f"A/B diff contract failed; unauthorized={sorted(unauthorized)}, missing={sorted(missing)}"
        )
    return differences


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    repository_path = (REPO_ROOT / path).resolve()
    if repository_path.exists():
        return repository_path
    try:
        bootcamp_relative = path.relative_to(BOOTCAMP_PREFIX)
    except ValueError:
        return repository_path
    return (BOOTCAMP_ROOT / bootcamp_relative).resolve()


def verify_mixture(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    path = resolve_project_path(data["manifest_path"])
    if file_sha256(path) != data["manifest_file_sha256"]:
        raise Day12PreparationError(f"mixture file hash mismatch: {path}")
    manifest = load_json(path)
    header = manifest.get("header")
    records = manifest.get("records")
    if not isinstance(header, dict) or not isinstance(records, list):
        raise Day12PreparationError("mixture manifest has no header/records")
    if header.get("status") != "complete_with_gate_c_waiver":
        raise Day12PreparationError("mixture is not frozen for downstream training")
    if header.get("manifest_hash") != data["manifest_hash"]:
        raise Day12PreparationError("configured mixture manifest hash mismatch")
    if header.get("mix_name") != data["mix_name"]:
        raise Day12PreparationError("configured mixture name mismatch")
    candidate = {"header": {k: v for k, v in header.items() if k != "manifest_hash"}, "records": records}
    if object_sha256(candidate) != header["manifest_hash"]:
        raise Day12PreparationError("mixture semantic hash is invalid")
    ids = [record.get("occurrence_id") for record in records]
    if not all(isinstance(value, str) and value for value in ids) or len(ids) != len(set(ids)):
        raise Day12PreparationError("mixture occurrence IDs are missing or duplicated")
    expected_total = config["training"]["checkpoint_budgets"]["100_percent"]
    if header.get("totals", {}).get("supervised_tokens") != expected_total:
        raise Day12PreparationError("mixture total differs from the full training budget")
    return manifest


def allocate_skill_targets(total: int, ratios: dict[str, str]) -> dict[str, int]:
    exact = {skill: Fraction(total) * Fraction(ratios[skill]) for skill in SKILLS}
    allocated = {skill: value.numerator // value.denominator for skill, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        SKILLS,
        key=lambda skill: (-(exact[skill] - allocated[skill]), SKILLS.index(skill)),
    )
    for skill in order[:remaining]:
        allocated[skill] += 1
    if sum(allocated.values()) != total:
        raise Day12PreparationError("skill target allocation did not preserve the token budget")
    return allocated


def exact_subset_indices(weights: list[int], target: int) -> set[int]:
    if target < 0:
        raise Day12PreparationError("subset target must be non-negative")
    reachable = 1
    snapshots: list[int] = []
    mask = (1 << (target + 1)) - 1
    for weight in weights:
        if not isinstance(weight, int) or weight <= 0:
            raise Day12PreparationError("supervised-token weights must be positive integers")
        snapshots.append(reachable)
        reachable = (reachable | (reachable << weight)) & mask
    if not (reachable >> target) & 1:
        raise Day12PreparationError(f"cannot construct an exact supervised-token subset: {target}")
    chosen: set[int] = set()
    cursor = target
    for index in range(len(weights) - 1, -1, -1):
        if (snapshots[index] >> cursor) & 1:
            continue
        chosen.add(index)
        cursor -= weights[index]
    if cursor != 0 or sum(weights[index] for index in chosen) != target:
        raise Day12PreparationError("exact subset reconstruction failed")
    return chosen


def build_schedule(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest["records"]
    header = manifest["header"]
    ratios = header["target_ratios"]
    budgets = config["training"]["checkpoint_budgets"]
    early_targets = allocate_skill_targets(int(budgets["25_percent"]), ratios)
    mid_targets = allocate_skill_targets(int(budgets["60_percent"]), ratios)
    selected_by_stage: dict[str, set[str]] = {name: set() for name in budgets}

    for skill in SKILLS:
        skill_records = [record for record in records if record.get("skill") == skill]
        weights = [int(record["supervised_tokens_per_occurrence"]) for record in skill_records]
        early_indices = exact_subset_indices(weights, early_targets[skill])
        for index in early_indices:
            selected_by_stage["25_percent"].add(skill_records[index]["occurrence_id"])

        remaining_records = [
            record for index, record in enumerate(skill_records) if index not in early_indices
        ]
        remaining_weights = [int(record["supervised_tokens_per_occurrence"]) for record in remaining_records]
        mid_increment = mid_targets[skill] - early_targets[skill]
        mid_indices = exact_subset_indices(remaining_weights, mid_increment)
        for index in mid_indices:
            selected_by_stage["60_percent"].add(remaining_records[index]["occurrence_id"])

    selected_by_stage["100_percent"] = {
        record["occurrence_id"] for record in records
    } - selected_by_stage["25_percent"] - selected_by_stage["60_percent"]

    by_id = {record["occurrence_id"]: record for record in records}
    segments: dict[str, Any] = {}
    cumulative_tokens = 0
    cumulative_records = 0
    covered: set[str] = set()
    for name in config["training"]["checkpoint_order"]:
        ids = [
            record["occurrence_id"]
            for record in records
            if record["occurrence_id"] in selected_by_stage[name]
        ]
        if covered.intersection(ids):
            raise Day12PreparationError("schedule segments overlap")
        covered.update(ids)
        tokens_by_skill = {
            skill: sum(
                int(by_id[occurrence_id]["supervised_tokens_per_occurrence"])
                for occurrence_id in ids
                if by_id[occurrence_id]["skill"] == skill
            )
            for skill in SKILLS
        }
        segment_tokens = sum(tokens_by_skill.values())
        cumulative_tokens += segment_tokens
        cumulative_records += len(ids)
        if cumulative_tokens != int(budgets[name]):
            raise Day12PreparationError(
                f"{name} cumulative token budget mismatch: {cumulative_tokens}"
            )
        segments[name] = {
            "occurrence_ids": ids,
            "record_count": len(ids),
            "supervised_tokens": segment_tokens,
            "tokens_by_skill": tokens_by_skill,
            "cumulative_record_count": cumulative_records,
            "cumulative_supervised_tokens": cumulative_tokens,
        }
    if covered != set(by_id):
        raise Day12PreparationError("schedule does not cover every mixture occurrence exactly once")

    schedule = {
        "header": {
            "domain": "day12.exact_supervised_token_schedule",
            "schema_version": 1,
            "run_id": config["run"]["id"],
            "seed": config["seed"],
            "mixture_manifest_hash": header["manifest_hash"],
            "mixture_file_sha256": config["data"]["manifest_file_sha256"],
            "algorithm": "per_skill_exact_subset_bitset_v1",
            "skill_order": list(SKILLS),
            "checkpoint_order": config["training"]["checkpoint_order"],
            "checkpoint_budgets": budgets,
            "target_ratios": ratios,
            "config_hash": object_sha256(config),
        },
        "segments": segments,
    }
    schedule["header"]["schedule_hash"] = object_sha256(schedule)
    return schedule


def verify_schedule(schedule: dict[str, Any]) -> None:
    header = schedule.get("header", {})
    expected = header.get("schedule_hash")
    candidate = copy.deepcopy(schedule)
    candidate.get("header", {}).pop("schedule_hash", None)
    if not isinstance(expected, str) or object_sha256(candidate) != expected:
        raise Day12PreparationError("schedule hash is invalid")


def write_json_stable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise Day12PreparationError(f"refusing to overwrite different frozen artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def write_diff_report(
    path: Path,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    differences: dict[str, tuple[Any, Any]],
    schedule_a: dict[str, Any],
    schedule_b: dict[str, Any],
) -> None:
    lines = [
        "# Day 12 A/B Config Diff",
        "",
        "Status: `pass` — the only behavioral difference is the frozen Day 09 mixture.",
        "",
        "| Path | A | B | Classification |",
        "|---|---|---|---|",
    ]
    for field, (value_a, value_b) in differences.items():
        classification = "run identity" if field == "run.id" else "authorized mixture identity"
        lines.append(
            f"| `{field}` | `{value_a}` | `{value_b}` | {classification} |"
        )
    lines.extend(
        [
            "",
            "## Common training contract",
            "",
            f"- Base config SHA-256: `{config_a['config_source']['base_file_sha256']}`",
            f"- A resolved config hash: `{object_sha256(config_a)}`",
            f"- B resolved config hash: `{object_sha256(config_b)}`",
            "- Checkpoints use exact cumulative supervised-token budgets: `61,734 / 148,162 / 246,936`.",
            f"- A schedule hash: `{schedule_a['header']['schedule_hash']}`",
            f"- B schedule hash: `{schedule_b['header']['schedule_hash']}`",
            "- The exact subset schedule preserves each run's preregistered skill ratios at every boundary up to deterministic integer rounding.",
            "",
        ]
    )
    payload = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise Day12PreparationError(f"refusing to overwrite different diff report: {path}")
    path.write_text(payload, encoding="utf-8")


def prepare(
    config_a_path: Path,
    config_b_path: Path,
    schedule_a_path: Path,
    schedule_b_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    config_a = load_resolved_config(config_a_path)
    config_b = load_resolved_config(config_b_path)
    differences = verify_ab_diff(config_a, config_b)
    manifest_a = verify_mixture(config_a)
    manifest_b = verify_mixture(config_b)
    common_a = manifest_a["header"]["common_invariants"]
    common_b = manifest_b["header"]["common_invariants"]
    if common_a != common_b:
        raise Day12PreparationError("A/B common Day 09 invariants differ")
    schedule_a = build_schedule(config_a, manifest_a)
    schedule_b = build_schedule(config_b, manifest_b)
    verify_schedule(schedule_a)
    verify_schedule(schedule_b)
    write_json_stable(schedule_a_path, schedule_a)
    write_json_stable(schedule_b_path, schedule_b)
    write_diff_report(report_path, config_a, config_b, differences, schedule_a, schedule_b)
    return {
        "status": "day12_preparation_pass",
        "authorized_differences": sorted(differences),
        "schedule_a": str(schedule_a_path.resolve()),
        "schedule_b": str(schedule_b_path.resolve()),
        "schedule_a_hash": schedule_a["header"]["schedule_hash"],
        "schedule_b_hash": schedule_b["header"]["schedule_hash"],
        "report": str(report_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-a", type=Path, default=DEFAULT_A_CONFIG)
    parser.add_argument("--config-b", type=Path, default=DEFAULT_B_CONFIG)
    parser.add_argument("--schedule-a", type=Path, default=DEFAULT_A_SCHEDULE)
    parser.add_argument("--schedule-b", type=Path, default=DEFAULT_B_SCHEDULE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        result = prepare(
            args.config_a,
            args.config_b,
            args.schedule_a,
            args.schedule_b,
            args.report,
        )
    except Day12PreparationError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
