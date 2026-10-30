#!/usr/bin/env python3
"""Append problem records and emit an immutable Day18 pass manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_GATES = (
    "host-inventory",
    "bootstrap",
    "model-preflight",
    "runtime-preflight",
    "topology-preflight",
    "c0-c1",
    "c2-dp2",
    "c3-tp2",
    "c4-resume-export",
    "c5-tiny-overfit",
)

REQUIRED_JSON_EVIDENCE: Dict[str, Optional[str]] = {
    "model-snapshot-verified.json": "downloaded_verified",
    "runtime-preflight.json": "pass",
    "topology/summary.json": "pass",
    "c1-hf-reference.json": None,
    "c1-conversion-parity.json": "pass",
    "c1-single-rank-loss-parity.json": "pass",
    "c1-mtp-roundtrip.json": "pass",
    "c2-dp2.json": "pass",
    "c2-checkpoint-audit.json": "pass",
    "c2-ranks-summary.json": "pass",
    "c2-loadcheck-ranks-summary.json": "pass",
    "c1-hf-vs-c2-loss.json": "pass",
    "c3-tp2.json": "pass",
    "c3-checkpoint-audit.json": "pass",
    "c3-ranks-summary.json": "pass",
    "c2-vs-c3-loss.json": "pass",
    "c4-resume.json": "pass",
    "c4-checkpoint-audit.json": "pass",
    "c4-ranks-summary.json": "pass",
    "c4-export-parity.json": "pass",
    "c4-hf-reference.json": None,
    "c4-mtp-trained-export.json": "pass",
    "c5-checkpoint-audit.json": "pass",
    "c5-ranks-summary.json": "pass",
    "c5-export-parity.json": "pass",
    "c5-hf-reference.json": None,
    "c5-parameter-scopes.json": "pass",
    "c5-tiny-overfit.json": "pass",
    "codepath-runtime-evidence.json": "pass",
}

REQUIRED_FILES = (
    "run-contract.preregistered.json",
    "parity-thresholds.preregistered.json",
    "pip-freeze.txt",
    "pip-inspect.json",
    "ms-swift-commit.txt",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_problems(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if row.get("status") not in {"open", "resolved", "observation"}:
            raise ValueError(f"{path}:{line_number}: invalid problem status {row.get('status')}")
        rows.append(row)
    return rows


def record_problem(args: argparse.Namespace) -> None:
    evidence_dir = args.run_root / "evidence"
    if not (args.run_root / ".day18-run-root").is_file():
        raise SystemExit(f"invalid Day18 run root: {args.run_root}")
    if (args.run_root / "DAY18-PASS.json").exists():
        raise SystemExit("refusing to mutate problem history after immutable Day18 pass")
    problem_path = evidence_dir / "problems.jsonl"
    default_problem_id = hashlib.sha256(
        f"{args.gate}\0{args.category}\0{args.summary}".encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "schema_version": 1,
        "recorded_at_utc": now(),
        "problem_id": args.problem_id or default_problem_id,
        "gate": args.gate,
        "category": args.category,
        "status": args.status,
        "summary": args.summary,
        "evidence": args.evidence,
    }
    problem_path.parent.mkdir(parents=True, exist_ok=True)
    with problem_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def inventory(paths: Iterable[Path], root: Path) -> List[Dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(set(paths))
    ]


def validate_problem_evidence(problems: Iterable[Dict[str, Any]], run_root: Path) -> tuple[List[str], List[Path]]:
    failures: List[str] = []
    paths: List[Path] = []
    for index, problem in enumerate(problems, 1):
        references = problem.get("evidence")
        if not isinstance(references, list):
            failures.append(f"problem record {index} evidence is not a list")
            continue
        for reference in references:
            if not isinstance(reference, str) or not reference.strip():
                failures.append(f"problem record {index} contains an invalid evidence reference")
                continue
            candidate = Path(reference)
            if not candidate.is_absolute():
                candidate = run_root / candidate
            resolved = candidate.resolve()
            if not resolved.is_relative_to(run_root):
                failures.append(f"problem record {index} evidence escapes run root: {reference}")
            elif not resolved.is_file():
                failures.append(f"problem record {index} evidence does not exist: {reference}")
            else:
                paths.append(resolved)
    return failures, paths


def finalize(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    evidence_dir = run_root / "evidence"
    pass_path = run_root / "DAY18-PASS.json"
    summary_path = evidence_dir / "day18-final-summary.json"
    if pass_path.exists():
        raise SystemExit(f"refusing to overwrite immutable pass manifest: {pass_path}")
    failures: List[str] = []
    evidence_paths: List[Path] = []

    for gate in REQUIRED_GATES:
        path = evidence_dir / "gates" / f"{gate}.pass.json"
        if not path.is_file():
            failures.append(f"missing pass marker: {gate}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("gate") != gate or payload.get("status") != "pass":
            failures.append(f"invalid pass marker: {gate}")
        evidence_paths.append(path)

    for relative, expected_status in REQUIRED_JSON_EVIDENCE.items():
        path = evidence_dir / relative
        if not path.is_file():
            failures.append(f"missing evidence: {relative}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if expected_status is not None and payload.get("status") != expected_status:
            failures.append(f"{relative} status is {payload.get('status')}, expected {expected_status}")
        evidence_paths.append(path)

    for relative in REQUIRED_FILES:
        path = evidence_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty evidence: {relative}")
            continue
        evidence_paths.append(path)

    problem_path = evidence_dir / "problems.jsonl"
    if not problem_path.is_file() or problem_path.stat().st_size == 0:
        failures.append("missing or empty evidence: problems.jsonl")
    try:
        problems = load_problems(problem_path)
    except ValueError as exc:
        failures.append(str(exc))
        problems = []
    if not problems:
        failures.append("problems.jsonl has no problem records")
    problem_evidence_failures, problem_evidence_paths = validate_problem_evidence(problems, run_root)
    failures.extend(problem_evidence_failures)
    evidence_paths.extend(problem_evidence_paths)
    effective_problems: Dict[str, Dict[str, Any]] = {}
    for index, problem in enumerate(problems):
        problem_id = str(problem.get("problem_id") or f"legacy-{index}")
        effective_problems[problem_id] = problem
    open_problems = [problem for problem in effective_problems.values() if problem["status"] == "open"]
    if open_problems:
        failures.append(f"{len(open_problems)} problem records remain open")
    if problem_path.is_file():
        evidence_paths.append(problem_path)

    payload = {
        "schema_version": 1,
        "created_at_utc": now(),
        "status": "day18_pass" if not failures else "day18_fail",
        "failures": failures,
        "claim": (
            "Day 18 E2E compatibility and same-entrypoint tiny-overfit learnability passed within the frozen "
            "Qwen3.5-4B text-only BF16 H800/H100 TP/DP envelope; this is not a generalization, exact-resume, "
            "Day15-completion, or global no-bug claim."
        ),
        "required_gates": list(REQUIRED_GATES),
        "problems": problems,
        "effective_problems": effective_problems,
        "open_problem_count": len(open_problems),
        "evidence_inventory": inventory(evidence_paths, run_root),
    }
    dump(summary_path, payload)
    if failures:
        raise SystemExit(1)
    dump(pass_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    problem_parser = subparsers.add_parser("problem")
    problem_parser.add_argument("--run-root", required=True, type=Path)
    problem_parser.add_argument("--gate", required=True)
    problem_parser.add_argument("--category", required=True)
    problem_parser.add_argument("--problem-id")
    problem_parser.add_argument("--status", required=True, choices=("open", "resolved", "observation"))
    problem_parser.add_argument("--summary", required=True)
    problem_parser.add_argument("--evidence", action="append", default=[])
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    {"problem": record_problem, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
