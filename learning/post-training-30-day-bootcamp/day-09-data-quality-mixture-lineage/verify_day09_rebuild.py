#!/usr/bin/env python3
"""Verify a fresh Day 09 rebuild and write the final acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BOOTCAMP = HERE.parent
DEFAULT_CONFIG = HERE / "day09_assignment.json"
DEFAULT_OFFICIAL_WORK = REPO_ROOT / "tmp" / "day09-work"
DEFAULT_OFFICIAL_DATA = BOOTCAMP / "artifacts" / "data"
DEFAULT_OUTPUT = BOOTCAMP / "artifacts" / "reports" / "day09-rebuild-acceptance.json"
SKILLS = ("general", "math", "code", "finance")


class RebuildVerificationError(ValueError):
    """The fresh rebuild does not reproduce the frozen Day 09 evidence."""


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


def compare_file(official: Path, rebuilt: Path) -> dict[str, Any]:
    official_hash = file_sha256(official)
    rebuilt_hash = file_sha256(rebuilt)
    return {
        "official_sha256": official_hash,
        "rebuilt_sha256": rebuilt_hash,
        "byte_identical": official_hash == rebuilt_hash,
    }


def exact_stage_files() -> list[str]:
    files = [f"{skill}-candidates.jsonl" for skill in SKILLS]
    files.extend(
        f"step3-max2048/{skill}-{decision}.jsonl"
        for skill in SKILLS
        for decision in ("accepted", "rejected")
    )
    files.extend(
        ["step4-audit/evidence-index.jsonl", "step5-filter/filter-decisions.jsonl"]
    )
    files.extend(f"step5-filter/{skill}-filtered.jsonl" for skill in SKILLS)
    files.extend(
        [
            "step6-exact-dedup/exact-dedup-decisions.jsonl",
            "step6-exact-dedup/exact-match-groups.jsonl",
        ]
    )
    files.extend(f"step6-exact-dedup/{skill}-deduped.jsonl" for skill in SKILLS)
    files.append("step7-near-candidates/near-duplicate-candidates.jsonl")
    files.extend(
        f"step8-eval-candidates/{skill}-eval-candidates.jsonl" for skill in SKILLS
    )
    files.append("step9-decontamination/train-eval-overlap-candidates.jsonl")
    return files


def run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(HERE),
        "-p",
        "test_day09_*.py",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    matched = re.search(r"Ran (\d+) tests?", output)
    return {
        "command": " ".join(command),
        "exit_code": result.returncode,
        "tests_run": int(matched.group(1)) if matched else None,
        "passed": result.returncode == 0 and "OK" in output,
    }


def build_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    official_work = args.official_work
    full_work = args.rebuild_root / "work"
    frozen_root = args.rebuild_root / "frozen-replay"
    official_data = args.official_data

    stage_files = {
        relative: compare_file(official_work / relative, full_work / relative)
        for relative in exact_stage_files()
    }
    official_clean = load_json(official_data / "day09-dataset-manifest.json")
    rebuilt_clean = load_json(
        args.rebuild_root / "artifacts" / "data" / "day09-dataset-manifest.json"
    )
    official_plan = load_json(
        official_work / "step11-token-budget" / "token-budget-plan.json"
    )
    rebuilt_plan = load_json(
        full_work / "step11-token-budget" / "token-budget-plan.json"
    )

    frozen_files = {
        "clean_parent_manifest": compare_file(
            official_data / "day09-dataset-manifest.json",
            frozen_root / "artifacts" / "data" / "day09-dataset-manifest.json",
        ),
        "token_budget_plan": compare_file(
            official_work / "step11-token-budget" / "token-budget-plan.json",
            frozen_root / "work" / "step11-token-budget" / "token-budget-plan.json",
        ),
        "mix_A_manifest": compare_file(
            official_data / "day09-mix-A-balanced.json",
            frozen_root / "artifacts" / "data" / "day09-mix-A-balanced.json",
        ),
        "mix_B_manifest": compare_file(
            official_data / "day09-mix-B-targeted.json",
            frozen_root / "artifacts" / "data" / "day09-mix-B-targeted.json",
        ),
    }
    config = load_json(args.config)
    gate_c = config["gate_c"]
    required_artifacts = {
        "clean_parent_manifest": official_data / "day09-dataset-manifest.json",
        "data_quality_audit": BOOTCAMP / "artifacts" / "reports" / "day09-data-quality-audit.md",
        "decontamination_report": BOOTCAMP / "artifacts" / "reports" / "day09-decontamination-report.md",
        "mix_A_manifest": official_data / "day09-mix-A-balanced.json",
        "mix_B_manifest": official_data / "day09-mix-B-targeted.json",
    }
    tests = run_tests()
    checks = {
        "steps_2_to_9_data_files_byte_identical": all(
            item["byte_identical"] for item in stage_files.values()
        ),
        "full_rebuild_clean_pool_hash_equal": official_clean["header"]["clean_pool_hash"]
        == rebuilt_clean["header"]["clean_pool_hash"],
        "full_rebuild_clean_records_equal": object_sha256(official_clean["records"])
        == object_sha256(rebuilt_clean["records"]),
        "full_rebuild_budget_equal": official_plan["common_total_supervised_tokens"]
        == rebuilt_plan["common_total_supervised_tokens"],
        "full_rebuild_ratios_equal": official_plan["ratios"] == rebuilt_plan["ratios"],
        "full_rebuild_selections_equal": object_sha256(
            official_plan["mix_feasibility_proofs"]
        )
        == object_sha256(rebuilt_plan["mix_feasibility_proofs"]),
        "frozen_manifest_plan_and_mixes_byte_identical": all(
            item["byte_identical"] for item in frozen_files.values()
        ),
        "common_supervised_tokens_equal": load_json(
            official_data / "day09-mix-A-balanced.json"
        )["header"]["totals"]["supervised_tokens"]
        == load_json(official_data / "day09-mix-B-targeted.json")["header"][
            "totals"
        ]["supervised_tokens"],
        "gate_c_waiver_explicit": gate_c["status"] == "waived_by_user"
        and gate_c["reviewed_occurrences_mix_A"] == 0
        and gate_c["reviewed_occurrences_mix_B"] == 0,
        "required_artifacts_present": all(path.is_file() for path in required_artifacts.values()),
        "tests_passed": tests["passed"],
    }
    acceptance = {
        "status": "complete_with_gate_c_waiver",
        "scope": {
            "full_rebuild": "Steps 2-11 semantic/data outputs from cached pinned sources in a fresh directory",
            "frozen_replay": "Steps 10-12 exact artifact replay from frozen reviewed inputs",
        },
        "gate_c": gate_c,
        "checks": checks,
        "full_rebuild_stage_file_comparisons": stage_files,
        "full_rebuild_semantic_hashes": {
            "official_clean_pool_hash": official_clean["header"]["clean_pool_hash"],
            "rebuilt_clean_pool_hash": rebuilt_clean["header"]["clean_pool_hash"],
            "official_clean_records_sha256": object_sha256(official_clean["records"]),
            "rebuilt_clean_records_sha256": object_sha256(rebuilt_clean["records"]),
            "official_selection_proofs_sha256": object_sha256(
                official_plan["mix_feasibility_proofs"]
            ),
            "rebuilt_selection_proofs_sha256": object_sha256(
                rebuilt_plan["mix_feasibility_proofs"]
            ),
        },
        "frozen_artifact_comparisons": frozen_files,
        "required_artifacts": {
            name: {
                "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
                "file_sha256": file_sha256(path),
            }
            for name, path in required_artifacts.items()
        },
        "tests": tests,
        "claim_boundary": (
            "Automated reproducibility passed. Gate C was waived by the user and "
            "must not be described as a passed human occurrence review."
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        mismatched_stage_files = [
            name for name, item in stage_files.items() if not item["byte_identical"]
        ]
        raise RebuildVerificationError(
            f"Step 13 verification failed: {failed}; "
            f"mismatched_stage_files={mismatched_stage_files}"
        )
    acceptance["acceptance_hash"] = object_sha256(acceptance)
    return acceptance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--official-work", type=Path, default=DEFAULT_OFFICIAL_WORK)
    parser.add_argument("--official-data", type=Path, default=DEFAULT_OFFICIAL_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    acceptance = build_acceptance(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(acceptance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"checks={sum(acceptance['checks'].values())}/{len(acceptance['checks'])} "
        f"tests={acceptance['tests']['tests_run']} "
        f"acceptance_hash={acceptance['acceptance_hash']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
