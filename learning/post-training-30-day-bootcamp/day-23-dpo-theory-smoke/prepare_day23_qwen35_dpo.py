#!/usr/bin/env python3
"""Compile the frozen Day 22 experimental pairs into deterministic ms-swift DPO JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract as contract


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
DAY21_DIR = BOOTCAMP_ROOT / "day-21-weekend-eval-reading"
DAY22_DIR = BOOTCAMP_ROOT / "day-22-preference-data"
SCRIPT_DIR = BOOTCAMP_ROOT / "artifacts" / "scripts"
for import_dir in (DAY21_DIR, DAY22_DIR, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import day22_contract  # noqa: E402
import validate_day22_experimental_close  # noqa: E402


DEFAULT_PROMOTION = BOOTCAMP_ROOT / "artifacts/checkpoints/day21-qwen35-s1-promotion-manifest.json"
DEFAULT_KEY = BOOTCAMP_ROOT / "artifacts/checkpoints/day21-qwen35-s1-downstream-key.json"
DEFAULT_EXPORT = BOOTCAMP_ROOT / "artifacts/checkpoints/day21-qwen35-s1-merged-export-manifest.json"
DEFAULT_EXPERIMENTAL = BOOTCAMP_ROOT / "artifacts/data/day22-qwen35-experimental-ai-assisted-manifest.json"
DEFAULT_OUTPUT_DIR = BOOTCAMP_ROOT / "artifacts/data"
OUTPUT_NAMES = {
    "train": "day23-qwen35-coding-dpo-train.jsonl",
    "dev": "day23-qwen35-coding-dpo-dev.jsonl",
    "heldout": "day23-qwen35-coding-dpo-heldout.jsonl",
}
MANIFEST_NAME = "day23-qwen35-coding-dpo-data-manifest.json"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(BOOTCAMP_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise contract.Day23ContractError(
            f"frozen input/output must be inside bootcamp root: {path}"
        ) from error


def _resolve_entry(entry: Mapping[str, Any], label: str) -> Path:
    value = entry.get("path")
    if not isinstance(value, str) or not value:
        raise contract.Day23ContractError(f"{label} path is invalid")
    path = (BOOTCAMP_ROOT / value).resolve()
    try:
        path.relative_to(BOOTCAMP_ROOT.resolve())
    except ValueError as error:
        raise contract.Day23ContractError(f"{label} path escapes bootcamp root") from error
    if not path.is_file():
        raise contract.Day23ContractError(f"{label} is missing: {path}")
    if entry.get("file_sha256") != contract.file_sha256(path):
        raise contract.Day23ContractError(f"{label} file hash drifted")
    return path


def build_bundle(
    *,
    promotion_path: Path = DEFAULT_PROMOTION,
    key_path: Path = DEFAULT_KEY,
    export_path: Path = DEFAULT_EXPORT,
    experimental_manifest_path: Path = DEFAULT_EXPERIMENTAL,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    promotion_path = promotion_path.resolve()
    key_path = key_path.resolve()
    export_path = export_path.resolve()
    experimental_manifest_path = experimental_manifest_path.resolve()

    promotion = contract.load_json(promotion_path)
    downstream_key = contract.load_json(key_path)
    merged_export = contract.load_json(export_path)
    parent = contract.validate_s1_bundle(
        promotion,
        downstream_key,
        merged_export,
        promotion_file_sha256=contract.file_sha256(promotion_path),
        downstream_key_file_sha256=contract.file_sha256(key_path),
        merged_export_file_sha256=contract.file_sha256(export_path),
    )

    # Keep the existing Day 22 close validator as the first authority gate.
    close_result = validate_day22_experimental_close.validate_manifest(
        experimental_manifest_path
    )
    experimental = contract.load_json(experimental_manifest_path)
    data_source = contract.validate_experimental_manifest(
        experimental,
        manifest_file_sha256=contract.file_sha256(experimental_manifest_path),
    )
    pairs_path = _resolve_entry(experimental["outputs"]["pairs"], "experimental pairs")
    split_path = _resolve_entry(experimental["outputs"]["split_ids"], "experimental split IDs")
    pairs = contract.load_jsonl(pairs_path)
    summary = day22_contract.validate_manifest(pairs, mode="final")
    if summary["records"] != 200:
        raise contract.Day23ContractError("Day 22 experimental pair count drifted")
    if summary["ordered_pair_hashes_sha256"] != data_source["ordered_pair_hashes_sha256"]:
        raise contract.Day23ContractError("Day 22 ordered pair identity drifted")
    if day22_contract.object_sha256([row["pair_id"] for row in pairs]) != data_source[
        "accepted_pair_ids_sha256"
    ]:
        raise contract.Day23ContractError("Day 22 accepted pair IDs drifted")
    for pair in pairs:
        contract.validate_pair_lineage(pair, parent)

    split_ids = contract.load_json(split_path)
    ordered_ids = contract.validate_split_ids(split_ids, pairs)
    if split_ids["split_ids_sha256"] != data_source["split_ids_sha256"]:
        raise contract.Day23ContractError("Day 22 split self-hash drifted")
    by_id = {str(row["pair_id"]): row for row in pairs}
    compiled = {
        split: [contract.compile_row(by_id[pair_id], split=split) for pair_id in ordered_ids[split]]
        for split in contract.SPLITS
    }

    outputs: dict[str, Any] = {}
    for split in contract.SPLITS:
        split_summary = contract.split_summary(compiled[split])
        payload = contract.jsonl_bytes(compiled[split])
        outputs[split] = {
            "path": f"artifacts/data/{OUTPUT_NAMES[split]}",
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            **split_summary,
        }
    manifest: dict[str, Any] = {
        "schema_name": contract.DATA_MANIFEST_SCHEMA_NAME,
        "schema_version": contract.SCHEMA_VERSION,
        "status": "cpu_data_ready",
        "dataset_track": contract.DATASET_TRACK,
        "claim_boundary": {
            "formal_human_reviewed": False,
            "gpu_optimizer_ready": False,
            "remote_payload_gate": contract.REMOTE_PAYLOAD_GATE,
            "gpu_runtime_gate": contract.GPU_RUNTIME_GATE,
        },
        "parent": parent,
        "policy_reference_mapping": {
            "policy_initial_state": "same_promoted_merged_s1",
            "reference_initial_state": "same_promoted_merged_s1",
            "policy_dpo_adapter": "fresh_lora_pending_gpu_runtime",
            "reference_implementation": "same_model_with_policy_adapter_disabled_pending_gpu_runtime",
            "ref_model_argument": None,
            "cpu_scope": "logical_identity_only_no_model_weights_loaded",
        },
        "inputs": {
            "day21_promotion": {
                "path": _relative(promotion_path),
                "file_sha256": contract.file_sha256(promotion_path),
                "content_sha256": parent["promotion_manifest_sha256"],
            },
            "day21_downstream_key": {
                "path": _relative(key_path),
                "file_sha256": contract.file_sha256(key_path),
                "content_sha256": parent["downstream_key_sha256"],
            },
            "day21_merged_export_manifest": {
                "path": _relative(export_path),
                "file_sha256": contract.file_sha256(export_path),
                "content_sha256": parent["merged_export_manifest_sha256"],
            },
            "day22_experimental_manifest": {
                "path": _relative(experimental_manifest_path),
                **data_source,
            },
            "day22_pairs": {
                "path": _relative(pairs_path),
                "file_sha256": contract.file_sha256(pairs_path),
                "records": len(pairs),
            },
            "day22_split_ids": {
                "path": _relative(split_path),
                "file_sha256": contract.file_sha256(split_path),
                "content_sha256": split_ids["split_ids_sha256"],
            },
        },
        "compiler": {
            "row_schema": contract.ROW_SCHEMA_NAME,
            "projection": "messages(user,chosen_assistant)+rejected_response",
            "preserve_text_bytes": True,
            "canonical_jsonl": "utf8_sort_keys_compact_lf",
            "split_order_authority": "day22.experimental_ai_assisted_split_ids",
            "source_files": {
                "day23_contract.py": contract.file_sha256(
                    Path(contract.__file__).resolve()
                ),
                "prepare_day23_qwen35_dpo.py": contract.file_sha256(
                    Path(__file__).resolve()
                ),
            },
            "ms_swift_loader_contract": {
                "strict": True,
                "disable_auto_column_mapping": True,
                "remove_unused_columns": True,
                "split_dataset_ratio": 0,
            },
        },
        "counts": {
            "pairs": sum(len(rows) for rows in compiled.values()),
            "branches": 2 * sum(len(rows) for rows in compiled.values()),
            "split_counts": {split: len(compiled[split]) for split in contract.SPLITS},
        },
        "outputs": outputs,
        "upstream_validator": close_result,
    }
    manifest["manifest_sha256"] = contract.object_sha256(manifest)
    return compiled, manifest


def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def materialized_bytes(
    compiled: Mapping[str, Sequence[Mapping[str, Any]]], manifest: Mapping[str, Any]
) -> dict[str, bytes]:
    result = {
        OUTPUT_NAMES[split]: contract.jsonl_bytes(compiled[split])
        for split in contract.SPLITS
    }
    result[MANIFEST_NAME] = _manifest_bytes(manifest)
    return result


def apply_bundle(output_dir: Path, payloads: Mapping[str, bytes], *, mode: str) -> None:
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise contract.Day23ContractError(f"output directory is missing: {output_dir}")
    for name, payload in payloads.items():
        path = output_dir / name
        if mode == "build":
            try:
                with path.open("xb") as handle:
                    handle.write(payload)
            except FileExistsError as error:
                raise contract.Day23ContractError(f"refusing to overwrite: {path}") from error
        elif mode == "check":
            try:
                actual = path.read_bytes()
            except OSError as error:
                raise contract.Day23ContractError(f"cannot read frozen output: {path}") from error
            if actual != payload:
                raise contract.Day23ContractError(f"frozen output is not reproducible: {path}")
        else:
            raise contract.Day23ContractError(f"unsupported mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "check"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--promotion", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--downstream-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--merged-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--experimental-manifest", type=Path, default=DEFAULT_EXPERIMENTAL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        compiled, manifest = build_bundle(
            promotion_path=args.promotion,
            key_path=args.downstream_key,
            export_path=args.merged_export,
            experimental_manifest_path=args.experimental_manifest,
        )
        payloads = materialized_bytes(compiled, manifest)
        apply_bundle(args.output_dir, payloads, mode=args.mode)
    except (
        OSError,
        contract.Day23ContractError,
        day22_contract.Day22ContractError,
        validate_day22_experimental_close.ExperimentalValidationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "valid_day23_dpo_data_bundle",
                "mode": args.mode,
                "manifest_sha256": manifest["manifest_sha256"],
                "split_counts": manifest["counts"]["split_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
