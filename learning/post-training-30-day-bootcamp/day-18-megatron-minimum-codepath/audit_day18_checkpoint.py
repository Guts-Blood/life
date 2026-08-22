#!/usr/bin/env python3
"""Fail-closed inventory for Day18 full-state or model-only checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


MODEL_STATE_PREFIXES = ("model.", "model0", "language_model.", "visual.", "output_layer.")


def is_model_state_key(key: str) -> bool:
    return key == "model" or key.startswith(MODEL_STATE_PREFIXES)


def flatten_keys(value: Any, prefix: str = "", depth: int = 0) -> Iterable[str]:
    if depth > 5:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name
            yield from flatten_keys(child, name, depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value[:8]):
            yield from flatten_keys(child, f"{prefix}[{index}]", depth + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--minimum-bytes", required=True, type=int)
    parser.add_argument("--checkpoint-kind", choices=("full-state", "model-only"), default="full-state")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    failures: List[str] = []
    checkpoint = args.checkpoint
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    iteration_dir = checkpoint / f"iter_{args.expected_iteration:07d}"
    args_path = checkpoint / "args.json"
    checkpoint_args: Dict[str, Any] = {}

    if not checkpoint.is_dir():
        failures.append(f"missing checkpoint directory: {checkpoint}")
    if not tracker.is_file():
        failures.append(f"missing tracker: {tracker}")
    else:
        try:
            tracked = int(tracker.read_text(encoding="utf-8").strip())
            if tracked != args.expected_iteration:
                failures.append(f"tracker expected {args.expected_iteration}, got {tracked}")
        except (OSError, ValueError) as exc:
            failures.append(f"invalid tracker: {exc}")
    if not args_path.is_file():
        failures.append(f"missing checkpoint args: {args_path}")
    else:
        checkpoint_args = json.loads(args_path.read_text(encoding="utf-8"))
        expected_no_save = args.checkpoint_kind == "model-only"
        if checkpoint_args.get("no_save_optim") is not expected_no_save:
            failures.append(f"no_save_optim was not {expected_no_save}")
        if checkpoint_args.get("no_save_rng") is not expected_no_save:
            failures.append(f"no_save_rng was not {expected_no_save}")
        if checkpoint_args.get("async_save") is not False:
            failures.append("async_save must be false for an immediate audit")
    if not iteration_dir.is_dir():
        failures.append(f"missing iteration directory: {iteration_dir}")

    files = sorted(path for path in iteration_dir.rglob("*") if path.is_file()) if iteration_dir.is_dir() else []
    inventory = [{"path": str(path.relative_to(iteration_dir)), "size": path.stat().st_size} for path in files]
    total_bytes = sum(item["size"] for item in inventory)
    if total_bytes < args.minimum_bytes:
        failures.append(f"checkpoint payload is only {total_bytes} bytes; expected at least {args.minimum_bytes}")
    common_path = iteration_dir / "common.pt"
    if not common_path.is_file():
        failures.append(f"missing common state: {common_path}")

    state_keys: List[str] = []
    dcp_state_keys: List[str] = []
    metadata_errors: List[str] = []
    if common_path.is_file():
        try:
            import torch

            common = torch.load(common_path, map_location="cpu", weights_only=False)
            state_keys.extend(flatten_keys(common))
        except Exception as exc:  # noqa: BLE001 - the precise checkpoint reader failure is evidence.
            metadata_errors.append(f"common.pt: {type(exc).__name__}: {exc}")
    try:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(iteration_dir)).read_metadata()
        # MCore TorchDist checkpoints legitimately set optional planner_data
        # to None. state_dict_metadata is the authoritative payload inventory.
        dcp_state_keys = [str(key) for key in metadata.state_dict_metadata]
        state_keys.extend(dcp_state_keys)
    except Exception as exc:  # noqa: BLE001
        metadata_errors.append(f"torch.distributed.checkpoint metadata: {type(exc).__name__}: {exc}")
    failures.extend(metadata_errors)

    unique_keys = sorted(set(state_keys))
    # Presence/absence gates must be based on the distributed payload itself.
    # common.pt may contain top-level optimizer metadata even when the actual
    # optimizer shards are missing, which would otherwise create a false pass.
    optimizer_keys = [
        key for key in dcp_state_keys if "optimizer" in key.lower() or "exp_avg" in key.lower()
    ]
    rng_keys = [
        key for key in dcp_state_keys if "rng_state" in key.lower() or "rng_tracker" in key.lower()
    ]
    model_keys = sorted(key for key in dcp_state_keys if is_model_state_key(key))
    language_model_keys = [key for key in model_keys if key.startswith("language_model.")]
    mtp_keys = [key for key in model_keys if ".mtp." in key or key.startswith("mtp.")]
    visual_keys = [key for key in model_keys if key.startswith("visual.") or ".visual." in key]
    if not model_keys:
        failures.append("checkpoint metadata has no model state keys")
    if checkpoint_args.get("model_type") == "qwen3_5" and not language_model_keys:
        failures.append("Qwen3.5 checkpoint metadata has no language-model state keys")
    if checkpoint_args.get("mtp_num_layers") == 1 and not mtp_keys:
        failures.append("checkpoint metadata has no MTP state keys")
    if checkpoint_args.get("is_multimodal") is True and not visual_keys:
        failures.append("checkpoint metadata has no visual state keys")
    if args.checkpoint_kind == "full-state":
        if not optimizer_keys:
            failures.append("checkpoint metadata has no optimizer state keys")
        if not rng_keys:
            failures.append("checkpoint metadata has no RNG state keys")
    else:
        if optimizer_keys:
            failures.append("model-only checkpoint unexpectedly contains optimizer state keys")
        if rng_keys:
            failures.append("model-only checkpoint unexpectedly contains RNG state keys")

    inventory_digest = hashlib.sha256(
        json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    key_digest = hashlib.sha256("\n".join(unique_keys).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "expected_iteration": args.expected_iteration,
        "total_payload_bytes": total_bytes,
        "minimum_payload_bytes": args.minimum_bytes,
        "file_count": len(inventory),
        "file_inventory_sha256": inventory_digest,
        "file_inventory_sample": inventory[:100],
        "state_key_count": len(unique_keys),
        "state_keys_sha256": key_digest,
        "optimizer_key_sample": optimizer_keys[:50],
        "rng_key_sample": rng_keys[:50],
        "model_key_sample": model_keys[:50],
        "language_model_key_sample": language_model_keys[:50],
        "mtp_key_sample": mtp_keys[:50],
        "visual_key_sample": visual_keys[:50],
        "metadata_reader_errors": metadata_errors,
        "checkpoint_args": checkpoint_args,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
