#!/usr/bin/env python3
"""Inventory legacy Day 20 v1 eval artifacts before a fresh v2 GPU run.

This is deliberately *not* a v2 selection validator.  It only establishes
which historical ``qwen35-v2`` files are present and internally linked, so a
new candidate factory cannot accidentally inherit old candidate identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_CANDIDATES = (
    "base-probe",
    "probe-1e-5",
    "probe-3e-5",
    "probe-1e-4",
)


class EvalPreflightV2Error(ValueError):
    """An evaluation artifact is malformed or internally inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalPreflightV2Error(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise EvalPreflightV2Error(f"JSON artifact is not an object: {path}")
    return value


def jsonl_rows(path: Path) -> int:
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise EvalPreflightV2Error(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EvalPreflightV2Error(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows += 1
    except (OSError, json.JSONDecodeError) as error:
        raise EvalPreflightV2Error(f"cannot read JSONL artifact: {path}") from error
    return rows


def _pair_status(
    *, summary_path: Path, rows_path: Path, expected_records: int | None
) -> dict[str, Any]:
    missing = [str(path) for path in (summary_path, rows_path) if not path.is_file()]
    if missing:
        return {"complete": False, "missing": missing}
    summary = load_json(summary_path)
    actual_rows = jsonl_rows(rows_path)
    declared: int | None = None
    predictions = summary.get("predictions")
    if isinstance(predictions, Mapping):
        value = predictions.get("records")
        if isinstance(value, int) and not isinstance(value, bool):
            declared = value
        expected_hash = predictions.get("file_sha256")
        if isinstance(expected_hash, str) and file_sha256(rows_path) != expected_hash:
            raise EvalPreflightV2Error(
                f"prediction hash differs from summary: {rows_path}"
            )
    elif isinstance(summary.get("records"), int):
        declared = int(summary["records"])
    if declared is not None and declared != actual_rows:
        raise EvalPreflightV2Error(
            f"prediction row count differs from summary: {rows_path}"
        )
    if expected_records is not None and actual_rows != expected_records:
        raise EvalPreflightV2Error(
            f"expected {expected_records} prediction rows, found {actual_rows}: {rows_path}"
        )
    return {
        "complete": True,
        "summary": str(summary_path.resolve()),
        "rows": str(rows_path.resolve()),
        "records": actual_rows,
        "summary_sha256": file_sha256(summary_path),
        "rows_sha256": file_sha256(rows_path),
    }


def audit_candidate(
    eval_dir: Path, candidate: str, *, expected_records: int | None = 32
) -> dict[str, Any]:
    raw = _pair_status(
        summary_path=eval_dir / f"{candidate}.raw.json",
        rows_path=eval_dir / f"{candidate}.raw.predictions.jsonl",
        expected_records=expected_records,
    )
    normalized = _pair_status(
        summary_path=eval_dir / f"{candidate}.qwen35-v2.json",
        rows_path=eval_dir / f"{candidate}.qwen35-v2.predictions.jsonl",
        expected_records=expected_records,
    )
    e2b = _pair_status(
        summary_path=eval_dir / f"{candidate}-code-e2b-qwen35-v2-summary.json",
        rows_path=eval_dir / f"{candidate}-code-e2b-qwen35-v2.jsonl",
        expected_records=(expected_records // 4 if expected_records is not None else None),
    )
    return {
        "candidate": candidate,
        "raw_inference": raw,
        "normalized_inference": normalized,
        "e2b": e2b,
        "inference_complete": bool(raw["complete"] and normalized["complete"]),
        "selection_ready": bool(
            raw["complete"] and normalized["complete"] and e2b["complete"]
        ),
    }


def audit_run(
    run_root: Path,
    *,
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    expected_records: int | None = 32,
) -> dict[str, Any]:
    root = run_root.resolve()
    eval_dir = root / "eval"
    if not eval_dir.is_dir():
        raise EvalPreflightV2Error(f"eval directory is missing: {eval_dir}")
    rows = [
        audit_candidate(eval_dir, candidate, expected_records=expected_records)
        for candidate in candidates
    ]
    selection_path = root / "PROBE-SELECTION.json"
    return {
        "schema_version": 1,
        "domain": "day20.legacy_eval_artifact_inventory.v1",
        "artifact_generation": "legacy_day20_v1_qwen35_v2",
        "usable_for_v2_selection": False,
        "run_root": str(root),
        "candidates": rows,
        "inference_complete": all(row["inference_complete"] for row in rows),
        "e2b_complete": all(row["e2b"]["complete"] for row in rows),
        "selection": {
            "path": str(selection_path),
            "complete": selection_path.is_file(),
        },
        "selection_ready": bool(
            rows
            and all(row["selection_ready"] for row in rows)
            and selection_path.is_file()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--expected-records", type=int, default=32)
    parser.add_argument("--require-selection-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_run(
        args.run_root,
        candidates=args.candidates or DEFAULT_CANDIDATES,
        expected_records=args.expected_records,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_selection_ready and not result["selection_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
