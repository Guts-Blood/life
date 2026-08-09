#!/usr/bin/env python3
"""Prove C5 changed language/MTP weights while frozen multimodal weights stayed exact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from compare_day18_mtp_tensors import EXPECTED_MTP_TENSORS, get_tensor, weight_map


FROZEN_MARKERS = ("visual", "vision", "aligner", "merger", "multi_modal_projector")


def parameter_scope(name: str) -> str:
    if name in EXPECTED_MTP_TENSORS:
        return "mtp"
    if any(marker in name.lower() for marker in FROZEN_MARKERS):
        return "visual_or_aligner"
    return "main_language_model"


def names_sha256(names: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--changed-sample-limit", type=int, default=20)
    args = parser.parse_args()

    import torch

    base_map = weight_map(args.base)
    trained_map = weight_map(args.trained)
    base_names = set(base_map)
    trained_names = set(trained_map)
    failures: List[str] = []
    missing_from_trained = sorted(base_names - trained_names)
    added_in_trained = sorted(trained_names - base_names)
    if missing_from_trained:
        failures.append(f"trained export is missing {len(missing_from_trained)} base tensors")
    if added_in_trained:
        failures.append(f"trained export added {len(added_in_trained)} tensors")

    scope_rows: Dict[str, Dict[str, Any]] = {}
    for scope in ("main_language_model", "mtp", "visual_or_aligner"):
        names = sorted(name for name in base_names if parameter_scope(name) == scope)
        scope_rows[scope] = {
            "tensor_count": len(names),
            "tensor_names_sha256": names_sha256(names),
            "exact_tensor_count": 0,
            "changed_tensor_count": 0,
            "shape_or_dtype_mismatch_count": 0,
            "changed_tensor_sample": [],
        }

    for name in sorted(base_names & trained_names):
        scope = parameter_scope(name)
        row = scope_rows[scope]
        base_tensor = get_tensor(base_map, name)
        trained_tensor = get_tensor(trained_map, name)
        same_shape = tuple(base_tensor.shape) == tuple(trained_tensor.shape)
        same_dtype = base_tensor.dtype == trained_tensor.dtype
        if not same_shape or not same_dtype:
            row["shape_or_dtype_mismatch_count"] += 1
            failures.append(f"{name}: shape or dtype drift")
            continue
        if torch.equal(base_tensor, trained_tensor):
            row["exact_tensor_count"] += 1
            continue

        row["changed_tensor_count"] += 1
        if len(row["changed_tensor_sample"]) < args.changed_sample_limit:
            max_abs_difference = float((base_tensor.float() - trained_tensor.float()).abs().max().item())
            if not math.isfinite(max_abs_difference):
                failures.append(f"{name}: non-finite difference")
            row["changed_tensor_sample"].append(
                {
                    "name": name,
                    "shape": list(base_tensor.shape),
                    "dtype": str(base_tensor.dtype),
                    "max_absolute_difference": max_abs_difference,
                }
            )

    main_scope = scope_rows["main_language_model"]
    mtp_scope = scope_rows["mtp"]
    frozen_scope = scope_rows["visual_or_aligner"]
    if main_scope["tensor_count"] == 0 or main_scope["changed_tensor_count"] == 0:
        failures.append("no main language-model tensor changed during C5")
    if mtp_scope["tensor_count"] != len(EXPECTED_MTP_TENSORS):
        failures.append(
            f"expected {len(EXPECTED_MTP_TENSORS)} MTP tensors, found {mtp_scope['tensor_count']}"
        )
    if mtp_scope["changed_tensor_count"] == 0:
        failures.append("no MTP tensor changed during C5")
    if frozen_scope["tensor_count"] == 0:
        failures.append("no visual/aligner tensors were found in the full conditional model")
    if frozen_scope["changed_tensor_count"]:
        failures.append(
            f"{frozen_scope['changed_tensor_count']} frozen visual/aligner tensors changed during C5"
        )

    payload = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "base": str(args.base.resolve()),
        "trained": str(args.trained.resolve()),
        "base_tensor_count": len(base_names),
        "trained_tensor_count": len(trained_names),
        "missing_from_trained": missing_from_trained,
        "added_in_trained": added_in_trained,
        "scopes": scope_rows,
        "claim_scope": "parameter ownership only; exact means bitwise equality after HF export",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
