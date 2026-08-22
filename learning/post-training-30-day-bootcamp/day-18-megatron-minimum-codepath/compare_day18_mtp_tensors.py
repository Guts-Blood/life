#!/usr/bin/env python3
"""Compare the exact 15 Qwen3.5 MTP tensors across HF snapshots/exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


EXPECTED_MTP_TENSORS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def weight_map(root: Path) -> Dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        mapping = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        return {name: root / filename for name, filename in mapping.items()}
    from safetensors import safe_open

    mapping: Dict[str, Path] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                mapping[name] = path
    return mapping


def get_tensor(mapping: Dict[str, Path], name: str):
    from safetensors import safe_open

    with safe_open(mapping[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def tensor_sha256(tensor) -> str:
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("exact", "changed"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    left_map = weight_map(args.left)
    right_map = weight_map(args.right)
    failures: List[str] = []
    for side, mapping in (("left", left_map), ("right", right_map)):
        missing = sorted(EXPECTED_MTP_TENSORS - set(mapping))
        if missing:
            failures.append(f"{side} is missing MTP tensors: {missing}")

    rows = []
    changed_count = 0
    if not failures:
        for name in sorted(EXPECTED_MTP_TENSORS):
            left = get_tensor(left_map, name)
            right = get_tensor(right_map, name)
            same_shape = tuple(left.shape) == tuple(right.shape)
            same_dtype = left.dtype == right.dtype
            exact_equal = same_shape and same_dtype and torch.equal(left, right)
            if not exact_equal:
                changed_count += 1
            max_abs_difference = None
            if same_shape:
                max_abs_difference = float((left.float() - right.float()).abs().max().item())
                if not math.isfinite(max_abs_difference):
                    failures.append(f"{name}: non-finite difference")
            rows.append(
                {
                    "name": name,
                    "shape_left": list(left.shape),
                    "shape_right": list(right.shape),
                    "dtype_left": str(left.dtype),
                    "dtype_right": str(right.dtype),
                    "sha256_left": tensor_sha256(left),
                    "sha256_right": tensor_sha256(right),
                    "exact_equal": exact_equal,
                    "max_absolute_difference": max_abs_difference,
                }
            )
            if not same_shape or not same_dtype:
                failures.append(f"{name}: shape or dtype drift")
            if args.mode == "exact" and not exact_equal:
                failures.append(f"{name}: C0 conversion round-trip is not exact")
    if args.mode == "changed" and changed_count == 0:
        failures.append("none of the 15 MTP tensors changed after five optimizer steps")

    payload = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "mode": args.mode,
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "expected_tensor_count": len(EXPECTED_MTP_TENSORS),
        "changed_tensor_count": changed_count,
        "tensors": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
