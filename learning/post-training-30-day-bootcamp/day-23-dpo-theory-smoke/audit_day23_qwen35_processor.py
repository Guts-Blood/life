#!/usr/bin/env python3
"""Run the real pinned Qwen3.5 processor through the Day 23 RLHF template."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract as contract
import day23_rlhf_template as dpo_template


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
DAY22_DIR = BOOTCAMP_ROOT / "day-22-preference-data"
if str(DAY22_DIR) not in sys.path:
    sys.path.insert(0, str(DAY22_DIR))

import audit_day22_qwen35_processor as day22_processor  # noqa: E402


DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "tmp/qwen35-v2/models/Qwen--Qwen3.5-4B-Base"
    / contract.MODEL_REVISION
)
DEFAULT_VENDOR = REPO_ROOT / "vendor/ms-swift"
DEFAULT_DATA_MANIFEST = (
    BOOTCAMP_ROOT / "artifacts/data/day23-qwen35-coding-dpo-data-manifest.json"
)
DEFAULT_SOURCE_PAIRS = (
    BOOTCAMP_ROOT / "artifacts/data/day22-qwen35-formal-s1-preference-pairs.jsonl"
)
DEFAULT_OUTPUT_DIR = BOOTCAMP_ROOT / "artifacts/eval"
AUDIT_ROWS_NAME = "day23-qwen35-coding-dpo-processor-audit.jsonl"
AUDIT_SUMMARY_NAME = "day23-qwen35-coding-dpo-processor-audit-summary.json"


class Day23ProcessorAuditError(ValueError):
    """The processor snapshot, RLHF encoding, or audit output drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23ProcessorAuditError(message)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_identity(vendor: Path) -> dict[str, Any]:
    resolved = vendor.resolve()
    _require(resolved.is_dir(), f"ms-swift checkout is missing: {resolved}")
    try:
        import swift
    except ImportError as error:
        raise Day23ProcessorAuditError("ms-swift import is unavailable") from error
    imported_root = Path(swift.__file__).resolve().parent
    _require(
        imported_root == resolved / "swift",
        "imported ms-swift does not come from the pinned vendor checkout",
    )
    try:
        commit = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(resolved), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day23ProcessorAuditError("cannot inspect pinned ms-swift checkout") from error
    _require(commit == contract.MS_SWIFT_COMMIT, "ms-swift checkout commit drifted")
    _require(status == "", "ms-swift checkout is dirty")
    critical = (
        "swift/template/base.py",
        "swift/template/template_inputs.py",
        "swift/template/templates/qwen.py",
        "swift/pipelines/train/rlhf.py",
        "swift/cli/main.py",
        "swift/arguments/rlhf_args.py",
        "swift/rlhf_trainers/dpo_trainer.py",
        "swift/rlhf_trainers/rlhf_mixin.py",
    )
    files: dict[str, str] = {}
    for relative in critical:
        path = resolved / relative
        _require(path.is_file(), f"pinned ms-swift file is missing: {relative}")
        files[relative] = contract.file_sha256(path)
    return {
        "commit": commit,
        "clean": True,
        "imported_from_pinned_checkout": True,
        "package_init_sha256": contract.file_sha256(imported_root / "__init__.py"),
        "critical_files_sha256": files,
        "critical_files_identity_sha256": contract.object_sha256(files),
    }


def _processor_identity(model_path: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    _require(resolved.is_dir(), f"model snapshot is missing: {resolved}")
    _require(resolved.name == contract.MODEL_REVISION, "model snapshot revision drifted")
    files: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(day22_processor.EXPECTED_SNAPSHOT_SHA256.items()):
        path = resolved / relative
        _require(path.is_file(), f"processor asset is missing: {relative}")
        actual = contract.file_sha256(path)
        _require(actual == expected, f"processor asset hash drifted: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": actual}
    return {
        "scope": "processor_tokenizer_template_only_no_model_weights",
        "model_id": contract.MODEL_ID,
        "model_revision": contract.MODEL_REVISION,
        "snapshot_files": files,
        "snapshot_files_sha256": contract.object_sha256(files),
    }


def _resolve_entry(entry: Mapping[str, Any], label: str) -> Path:
    value = entry.get("path")
    _require(isinstance(value, str) and bool(value), f"{label} path is invalid")
    path = (BOOTCAMP_ROOT / str(value)).resolve()
    try:
        path.relative_to(BOOTCAMP_ROOT.resolve())
    except ValueError as error:
        raise Day23ProcessorAuditError(f"{label} escapes bootcamp root") from error
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(entry.get("file_sha256") == contract.file_sha256(path), f"{label} hash drifted")
    return path


def _branch(
    encoded: Mapping[str, Any], prefix: str, pair_id: str
) -> dict[str, Any]:
    try:
        return {
            "input_ids": encoded[f"{prefix}_input_ids"],
            "labels": encoded[f"{prefix}_labels"],
        }
    except KeyError as error:
        raise Day23ProcessorAuditError(
            f"{pair_id}:{prefix} encoded branch is incomplete"
        ) from error


def _load_with_pinned_swift(
    split_paths: Mapping[str, Path],
    expected_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    try:
        from swift.dataset import load_dataset
    except ImportError as error:
        raise Day23ProcessorAuditError("pinned ms-swift dataset loader is unavailable") from error
    loaded: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    columns: dict[str, list[str]] = {}
    for split in contract.SPLITS:
        dataset, validation = load_dataset(
            str(split_paths[split]),
            split_dataset_ratio=0,
            seed=20260818,
            num_proc=1,
            load_from_cache_file=False,
            shuffle=False,
            streaming=False,
            strict=True,
            remove_unused_columns=True,
            disable_auto_column_mapping=True,
        )
        _require(validation is None, f"ms-swift unexpectedly split {split} data")
        _require(len(dataset) == len(expected_rows[split]), f"ms-swift deleted {split} rows")
        _require(
            set(dataset.column_names) == {"messages", "rejected_response", "dataset"},
            f"ms-swift {split} standard columns drifted: {dataset.column_names}",
        )
        loaded_rows: list[dict[str, Any]] = []
        for index, (actual, expected) in enumerate(zip(dataset, expected_rows[split])):
            standard = {
                "messages": actual.get("messages"),
                "rejected_response": actual.get("rejected_response"),
            }
            expected_standard = {
                "messages": expected["messages"],
                "rejected_response": expected["rejected_response"],
            }
            _require(
                standard == expected_standard,
                f"ms-swift loader changed {split} row {index} text/labels",
            )
            loaded_rows.append(standard)
        loaded[split] = loaded_rows
        counts[split] = len(loaded_rows)
        columns[split] = sorted(dataset.column_names)
    evidence = {
        "implementation": "swift.dataset.load_dataset",
        "strict": True,
        "disable_auto_column_mapping": True,
        "remove_unused_columns": True,
        "split_dataset_ratio": 0,
        "shuffle": False,
        "input_counts": dict(counts),
        "output_counts": dict(counts),
        "silent_row_deletions": 0,
        "standard_text_fields_exact": True,
        "columns_after_preprocess": columns,
    }
    evidence["loader_evidence_sha256"] = contract.object_sha256(evidence)
    return loaded, evidence


def build_audit(
    *,
    data_manifest_path: Path = DEFAULT_DATA_MANIFEST,
    source_pairs_path: Path = DEFAULT_SOURCE_PAIRS,
    model_path: Path = DEFAULT_MODEL_PATH,
    vendor_path: Path = DEFAULT_VENDOR,
    max_length: int = dpo_template.DEFAULT_MAX_LENGTH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_manifest_path = data_manifest_path.resolve()
    source_pairs_path = source_pairs_path.resolve()
    data_manifest = contract.load_json(data_manifest_path)
    contract.verify_self_hash(data_manifest, "manifest_sha256", "Day 23 data manifest")
    _require(
        data_manifest.get("schema_name") == contract.DATA_MANIFEST_SCHEMA_NAME
        and data_manifest.get("status") == "cpu_data_ready",
        "Day 23 data manifest is not CPU-ready",
    )
    _require(
        data_manifest.get("counts", {}).get("split_counts")
        == contract.EXPECTED_SPLIT_COUNTS,
        "Day 23 data split counts drifted",
    )
    source_pairs = contract.load_jsonl(source_pairs_path)
    source_by_id = {str(pair["pair_id"]): pair for pair in source_pairs}
    _require(len(source_by_id) == 200, "source pair inventory drifted")

    runtime_rows: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_paths: dict[str, Path] = {}
    for split in contract.SPLITS:
        path = _resolve_entry(data_manifest["outputs"][split], f"Day 23 {split}")
        split_paths[split] = path
        rows = contract.load_jsonl(path)
        _require(len(rows) == contract.EXPECTED_SPLIT_COUNTS[split], f"{split} rows drifted")
        for row in rows:
            pair_id = str(row.get("pair_id"))
            _require(pair_id in source_by_id, f"unknown compiled pair: {pair_id}")
            contract.validate_compiled_row(row, source_pair=source_by_id[pair_id])
            runtime_rows.append(row)
        rows_by_split[split] = rows
    _require(len(runtime_rows) == 200, "compiled runtime row inventory drifted")

    vendor_identity = _git_identity(vendor_path)
    processor_identity = _processor_identity(model_path)
    loaded_by_split, loader_evidence = _load_with_pinned_swift(
        split_paths, rows_by_split
    )
    template = dpo_template.build_dpo_template(model_path, max_length=max_length)
    audits: list[dict[str, Any]] = []
    lengths: list[int] = []
    response_lengths: list[int] = []
    split_counts: Counter[str] = Counter()
    loaded_indexes: Counter[str] = Counter()
    for row in runtime_rows:
        pair_id = str(row["pair_id"])
        source = source_by_id[pair_id]
        split = str(row["split"])
        loader_row = loaded_by_split[split][loaded_indexes[split]]
        loaded_indexes[split] += 1
        encoded = dpo_template.encode_dpo_row(template, loader_row)
        evidence: dict[str, dict[str, Any]] = {}
        prefixes: dict[str, tuple[int, ...]] = {}
        for branch in ("chosen", "rejected"):
            response = (
                str(row["messages"][1]["content"])
                if branch == "chosen"
                else str(row["rejected_response"])
            )
            branch_evidence, prefix = day22_processor.analyze_branch_encoding(
                _branch(encoded, branch, pair_id),
                response_text=response,
                decode_supervised=lambda ids: dpo_template.decode_supervised(
                    template, ids
                ),
                max_length=max_length,
                label=f"{pair_id}:{branch}",
            )
            expected = source["processor_audit"][branch]
            _require(
                branch_evidence == expected,
                f"{pair_id}:{branch} Day 23 encoding differs from frozen Day 22 evidence",
            )
            evidence[branch] = branch_evidence
            prefixes[branch] = prefix
            lengths.append(int(branch_evidence["input_token_count"]))
            response_lengths.append(int(branch_evidence["response_token_count"]))
        _require(
            prefixes["chosen"] == prefixes["rejected"],
            f"{pair_id}: chosen/rejected prompt prefixes differ",
        )
        _require(
            evidence["chosen"]["response_span"][0]
            == evidence["rejected"]["response_span"][0],
            f"{pair_id}: chosen/rejected response boundaries differ",
        )
        audit: dict[str, Any] = {
            "schema_name": "day23.qwen35_dpo_pair_processor_audit",
            "schema_version": 1,
            "status": "pass",
            "pair_id": pair_id,
            "split": row["split"],
            "source_pair_sha256": row["source_pair_sha256"],
            "compiled_row_sha256": row["row_sha256"],
            "prompt_prefix_exact_between_branches": True,
            "day22_frozen_branch_evidence_exact": True,
            "chosen": evidence["chosen"],
            "rejected": evidence["rejected"],
        }
        audit["audit_sha256"] = contract.object_sha256(audit)
        audits.append(audit)
        split_counts[str(row["split"])] += 1

    runtime = {
        "python": platform.python_version(),
        "transformers": _package_version("transformers"),
        "torch": _package_version("torch"),
        "peft": _package_version("peft"),
        "trl": _package_version("trl"),
        "accelerate": _package_version("accelerate"),
        "datasets": _package_version("datasets"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "ms_swift": _package_version("ms-swift"),
    }
    audit_rows_bytes = contract.jsonl_bytes(audits)
    summary: dict[str, Any] = {
        "schema_name": "day23.qwen35_dpo_processor_audit_summary",
        "schema_version": 1,
        "status": "pass",
        "scope": "real_processor_rlhf_encode_no_model_weights_no_optimizer",
        "data_manifest": {
            "path": "artifacts/data/day23-qwen35-coding-dpo-data-manifest.json",
            "file_sha256": contract.file_sha256(data_manifest_path),
            "manifest_sha256": data_manifest["manifest_sha256"],
        },
        "source_pairs": {
            "path": "artifacts/data/day22-qwen35-formal-s1-preference-pairs.jsonl",
            "file_sha256": contract.file_sha256(source_pairs_path),
        },
        "processor": processor_identity,
        "template": {
            "alias": dpo_template.TARGET_TEMPLATE_ALIAS,
            "mode": "rlhf",
            "contract": dpo_template.TARGET_ENCODING_CONTRACT,
            "contract_sha256": dpo_template.TARGET_ENCODING_CONTRACT_SHA256,
            "module_sha256": contract.file_sha256(Path(dpo_template.__file__).resolve()),
            "day20_source_contract_sha256": dpo_template.EXPECTED_DAY20_CONTRACT_SHA256,
            "max_length": max_length,
            "truncation_strategy": "raise",
        },
        "implementation_sources": {
            "audit_day23_qwen35_processor.py": contract.file_sha256(
                Path(__file__).resolve()
            ),
            "day20_target_encoding_v3.py": contract.file_sha256(
                Path(dpo_template.day20.__file__).resolve()
            ),
            "day23_contract.py": contract.file_sha256(
                Path(contract.__file__).resolve()
            ),
            "day23_rlhf_template.py": contract.file_sha256(
                Path(dpo_template.__file__).resolve()
            ),
        },
        "vendor_ms_swift": vendor_identity,
        "dataset_loader": loader_evidence,
        "runtime": runtime,
        "counts": {
            "pairs": len(audits),
            "branches": 2 * len(audits),
            "split_counts": {split: split_counts[split] for split in contract.SPLITS},
            "prompt_prefix_exact_pairs": sum(
                audit["prompt_prefix_exact_between_branches"] for audit in audits
            ),
            "day22_frozen_exact_branches": 2
            * sum(audit["day22_frozen_branch_evidence_exact"] for audit in audits),
            "response_only_mask_branches": sum(
                audit[branch]["response_only_mask"]
                for audit in audits
                for branch in ("chosen", "rejected")
            ),
            "four_space_boundary_branches": sum(
                audit[branch]["four_space_boundary"]
                for audit in audits
                for branch in ("chosen", "rejected")
            ),
            "zero_truncation_branches": sum(
                not audit[branch]["truncation"]
                for audit in audits
                for branch in ("chosen", "rejected")
            ),
        },
        "lengths": {
            "min_input_tokens": min(lengths),
            "max_input_tokens": max(lengths),
            "min_response_tokens": min(response_lengths),
            "max_response_tokens": max(response_lengths),
        },
        "audit_rows": {
            "path": f"artifacts/eval/{AUDIT_ROWS_NAME}",
            "file_sha256": hashlib.sha256(audit_rows_bytes).hexdigest(),
            "ordered_audit_sha256": contract.object_sha256(
                [audit["audit_sha256"] for audit in audits]
            ),
            "records": len(audits),
        },
        "claim_boundary": {
            "cpu_processor_gate": "pass",
            "model_weights_loaded": False,
            "gpu_optimizer_ready": False,
            "remote_payload_gate": contract.REMOTE_PAYLOAD_GATE,
            "gpu_runtime_gate": contract.GPU_RUNTIME_GATE,
        },
    }
    expected_counts = {
        "pairs": 200,
        "branches": 400,
        "split_counts": contract.EXPECTED_SPLIT_COUNTS,
        "prompt_prefix_exact_pairs": 200,
        "day22_frozen_exact_branches": 400,
        "response_only_mask_branches": 400,
        "four_space_boundary_branches": 400,
        "zero_truncation_branches": 400,
    }
    _require(summary["counts"] == expected_counts, "aggregate processor gate failed")
    _require(max(lengths) <= max_length, "processor audit contains truncation")
    summary["summary_sha256"] = contract.object_sha256(summary)
    return audits, summary


def materialized_bytes(
    audits: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, bytes]:
    return {
        AUDIT_ROWS_NAME: contract.jsonl_bytes(audits),
        AUDIT_SUMMARY_NAME: json.dumps(
            summary, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n",
    }


def apply_outputs(output_dir: Path, payloads: Mapping[str, bytes], *, mode: str) -> None:
    output_dir = output_dir.resolve()
    _require(output_dir.is_dir(), f"output directory is missing: {output_dir}")
    for name, payload in payloads.items():
        path = output_dir / name
        if mode == "build":
            try:
                with path.open("xb") as handle:
                    handle.write(payload)
            except FileExistsError as error:
                raise Day23ProcessorAuditError(f"refusing to overwrite: {path}") from error
        elif mode == "check":
            try:
                actual = path.read_bytes()
            except OSError as error:
                raise Day23ProcessorAuditError(f"cannot read frozen audit: {path}") from error
            _require(actual == payload, f"frozen processor audit drifted: {path}")
        else:
            raise Day23ProcessorAuditError(f"unsupported mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "check"), required=True)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--source-pairs", type=Path, default=DEFAULT_SOURCE_PAIRS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=dpo_template.DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audits, summary = build_audit(
            data_manifest_path=args.data_manifest,
            source_pairs_path=args.source_pairs,
            model_path=args.model,
            vendor_path=args.vendor,
            max_length=args.max_length,
        )
        apply_outputs(args.output_dir, materialized_bytes(audits, summary), mode=args.mode)
    except (
        OSError,
        contract.Day23ContractError,
        dpo_template.Day23RLHFTemplateError,
        day22_processor.Day22ProcessorAuditError,
        Day23ProcessorAuditError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "valid_day23_processor_cpu_gate",
                "mode": args.mode,
                "summary_sha256": summary["summary_sha256"],
                "counts": summary["counts"],
                "lengths": summary["lengths"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
