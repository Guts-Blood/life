#!/usr/bin/env python3
"""Independently validate the complete Day 23 CPU-ready/GPU-pending bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
BOOTCAMP_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
DAY22_DIR = BOOTCAMP_ROOT / "day-22-preference-data"
if str(DAY22_DIR) not in sys.path:
    sys.path.insert(0, str(DAY22_DIR))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import day22_contract  # noqa: E402
import validate_day22_experimental_close  # noqa: E402


SPLITS = ("train", "dev", "heldout")
EXPECTED_COUNTS = {"train": 154, "dev": 17, "heldout": 29}
MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
TEMPLATE_ALIAS = "day23_qwen3_5_dpo_target_v1"
TEMPLATE_CONTRACT_SHA256 = (
    "29f5d0550cc0d80536916940949d9deba7c2c2010dc891ca453f71b2ad6c5969"
)
DAY20_TEMPLATE_SHA256 = "7f835342861cc9aabfd6299228033df80437b183227e04db311450437cd07ded"
REMOTE_GATE = "pending_remote_s1_payload_verification"
GPU_GATE = "pending_gpu_reference_lora_memory_save_reload"
EXPECTED_UPSTREAM_FILE_SHA256 = {
    "day21_promotion": "40c76f690dcb73805250e0036f8e124ef07b8d2d32b1723637656ae4304d94f8",
    "day21_downstream_key": "979e599aa6c9c05bec9a14144b6ebae689dbdf8315a9112caaec09b5469bd545",
    "day21_merged_export": "9c196e43f633116d7fc804870dcfe961db5ae0d216a15f5e21d679d8c1f14b5a",
    "day22_experimental": "386c26c2f548f4b57c56ecb8265f36e83a89278bf1d7cf66bee6b0dfac819bcc",
}
EXPECTED_PROCESSOR_FILES = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "video_preprocessor_config.json": "d039cd7d88b3502a99edd455496b75b44ecf7dd3b3669748bafac37e6cecc085",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}
EXPECTED_VENDOR_FILES = {
    "swift/arguments/rlhf_args.py",
    "swift/cli/main.py",
    "swift/pipelines/train/rlhf.py",
    "swift/rlhf_trainers/dpo_trainer.py",
    "swift/rlhf_trainers/rlhf_mixin.py",
    "swift/template/base.py",
    "swift/template/template_inputs.py",
    "swift/template/templates/qwen.py",
}
DEFAULT_DATA_MANIFEST = (
    BOOTCAMP_ROOT / "artifacts/data/day23-qwen35-coding-dpo-data-manifest.json"
)
DEFAULT_PROCESSOR_SUMMARY = (
    BOOTCAMP_ROOT
    / "artifacts/eval/day23-qwen35-coding-dpo-processor-audit-summary.json"
)
DEFAULT_RUN_CONTRACT = (
    BOOTCAMP_ROOT
    / "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json"
)
DEFAULT_ARGUMENT_AUDIT = (
    BOOTCAMP_ROOT
    / "artifacts/eval/day23-qwen35-coding-dpo-argument-audit.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day23ValidationError(ValueError):
    """The independently reconstructed CPU bundle failed validation."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any, self_field: str | None = None) -> str:
    if self_field is not None:
        value = dict(value)
        value.pop(self_field, None)
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Day23ValidationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23ValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase bare SHA-256",
    )
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day23ValidationError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day23ValidationError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        _require(bool(line), f"blank JSONL row: {path}:{number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day23ValidationError(f"invalid JSONL row: {path}:{number}") from error
        _require(isinstance(row, dict), f"non-object JSONL row: {path}:{number}")
        rows.append(row)
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha(value.get(field), f"{label}.{field}")
    _require(expected == object_sha256(value, field), f"{label} self-hash mismatch")
    return expected


def resolve(entry: Mapping[str, Any], label: str) -> Path:
    value = entry.get("path")
    _require(isinstance(value, str) and bool(value), f"{label} path is invalid")
    path = (BOOTCAMP_ROOT / str(value)).resolve()
    try:
        path.relative_to(BOOTCAMP_ROOT.resolve())
    except ValueError as error:
        raise Day23ValidationError(f"{label} path escapes bootcamp root") from error
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(entry.get("file_sha256") == file_sha256(path), f"{label} file hash mismatch")
    return path


def _validate_day21(data_manifest: Mapping[str, Any]) -> dict[str, Any]:
    inputs = _mapping(data_manifest.get("inputs"), "data inputs")
    promotion_path = resolve(inputs["day21_promotion"], "Day 21 promotion")
    key_path = resolve(inputs["day21_downstream_key"], "Day 21 downstream key")
    export_path = resolve(
        inputs["day21_merged_export_manifest"], "Day 21 merged export"
    )
    _require(
        file_sha256(promotion_path) == EXPECTED_UPSTREAM_FILE_SHA256["day21_promotion"]
        and file_sha256(key_path)
        == EXPECTED_UPSTREAM_FILE_SHA256["day21_downstream_key"]
        and file_sha256(export_path)
        == EXPECTED_UPSTREAM_FILE_SHA256["day21_merged_export"],
        "Day 21 evidence differs from the frozen Day 23 trust root",
    )
    promotion = load_json(promotion_path)
    key = load_json(key_path)
    export = load_json(export_path)
    promotion_sha = verify_self(promotion, "promotion_manifest_sha256", "promotion")
    key_sha = verify_self(key, "key_sha256", "downstream key")
    export_sha = verify_self(export, "manifest_sha256", "merged export")
    _require(
        promotion.get("schema_name") == "day21.qwen35_downstream_ready_s1_promotion"
        and promotion.get("status") == "promoted"
        and promotion.get("downstream_ready") is True
        and promotion.get("role") == "S1",
        "Day 21 promotion is not downstream-ready S1",
    )
    _require(
        promotion.get("training", {}).get("framework_sha") == MS_SWIFT_COMMIT,
        "Day 21 framework commit drifted",
    )
    _require(
        promotion.get("inference_export", {}).get(
            "fresh_process_exact_token_id_parity_passed"
        )
        is True,
        "Day 21 fresh-process parity is absent",
    )
    _require(
        key.get("checkpoint_id") == promotion.get("checkpoint_id")
        and key.get("downstream_key") == promotion.get("downstream_key")
        and key.get("role") == "S1"
        and key.get("status") == "active",
        "Day 21 downstream key cross-binding drifted",
    )
    binding = _mapping(key.get("promotion_manifest"), "key promotion binding")
    _require(binding.get("content_sha256") == promotion_sha, "key content binding drifted")
    _require(binding.get("file_sha256") == file_sha256(promotion_path), "key file binding drifted")
    inference = _mapping(promotion.get("inference_export"), "promotion export")
    state = _mapping(promotion.get("state"), "promotion state")
    _require(
        inference.get("manifest", {}).get("content_sha256") == export_sha
        and inference.get("manifest", {}).get("file_sha256") == file_sha256(export_path)
        and inference.get("artifact_hash") == export.get("files_sha256"),
        "promotion/merged export binding drifted",
    )
    _require(
        export.get("source", {}).get("checkpoint_snapshot_sha256")
        == state.get("resumable_checkpoint_snapshot_sha256")
        and export.get("source", {}).get("checkpoint_integrity_sha256")
        == state.get("resumable_checkpoint_integrity_sha256")
        and export.get("source", {}).get("adapter_sha256") == state.get("adapter_sha256"),
        "promotion/merged state binding drifted",
    )
    parent = _mapping(data_manifest.get("parent"), "Day 23 parent")
    expected_parent = {
        "checkpoint_id": promotion["checkpoint_id"],
        "role": "S1",
        "downstream_key": promotion["downstream_key"],
        "promotion_file_sha256": file_sha256(promotion_path),
        "promotion_manifest_sha256": promotion_sha,
        "downstream_key_sha256": key_sha,
        "downstream_key_file_sha256": file_sha256(key_path),
        "checkpoint_snapshot_sha256": state["resumable_checkpoint_snapshot_sha256"],
        "checkpoint_integrity_sha256": state[
            "resumable_checkpoint_integrity_sha256"
        ],
        "adapter_sha256": state["adapter_sha256"],
        "merged_export_artifact_sha256": inference["artifact_hash"],
        "merged_export_file_sha256": file_sha256(export_path),
        "merged_export_manifest_sha256": export_sha,
        "ms_swift_commit": MS_SWIFT_COMMIT,
        "remote_merged_export_path": inference["path"],
        "remote_resumable_checkpoint_path": state["resumable_checkpoint_path"],
    }
    _require(dict(parent) == expected_parent, "Day 23 parent projection drifted")
    return expected_parent


def _expected_row(pair: Mapping[str, Any], split: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_name": "day23.ms_swift_dpo_row",
        "schema_version": 1,
        "dataset_track": "experimental_ai_assisted",
        "pair_id": pair["pair_id"],
        "split": split,
        "source_pair_sha256": pair["pair_sha256"],
        "family_keys": pair["family_keys"],
        "messages": [
            {"role": "user", "content": pair["prompt"]["text"]},
            {"role": "assistant", "content": pair["chosen"]["text"]},
        ],
        "rejected_response": pair["rejected"]["text"],
    }
    row["row_sha256"] = object_sha256(row)
    return row


def _validate_day22_and_compiled(
    data_manifest: Mapping[str, Any], parent: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    inputs = data_manifest["inputs"]
    experimental_path = resolve(
        inputs["day22_experimental_manifest"], "Day 22 experimental manifest"
    )
    _require(
        file_sha256(experimental_path)
        == EXPECTED_UPSTREAM_FILE_SHA256["day22_experimental"],
        "Day 22 experimental manifest differs from the frozen Day 23 trust root",
    )
    close = validate_day22_experimental_close.validate_manifest(experimental_path)
    _require(close.get("experimental_pairs") == 200, "Day 22 validator count drifted")
    experimental = load_json(experimental_path)
    _require(
        experimental.get("status") == "completed_experimental_ai_assisted"
        and experimental.get("experimental_dpo_ready") is True
        and experimental.get("formal_dpo_ready") is False,
        "Day 22 experimental/formal boundary drifted",
    )
    pairs_path = resolve(inputs["day22_pairs"], "Day 22 pairs")
    split_path = resolve(inputs["day22_split_ids"], "Day 22 split IDs")
    pairs = load_jsonl(pairs_path)
    pair_summary = day22_contract.validate_manifest(pairs, mode="final")
    _require(pair_summary["records"] == 200, "Day 22 pair count drifted")
    by_id = {str(pair["pair_id"]): pair for pair in pairs}
    _require(len(by_id) == 200, "Day 22 pair IDs are not unique")
    split_ids = load_json(split_path)
    split_sha = verify_self(split_ids, "split_ids_sha256", "Day 22 split IDs")
    _require(
        split_sha == experimental["content_identities"]["split_ids_sha256"],
        "experimental split identity drifted",
    )
    _require(
        object_sha256([pair["pair_id"] for pair in pairs])
        == experimental["content_identities"]["accepted_pair_ids_sha256"],
        "experimental accepted pair identity drifted",
    )
    _require(
        pair_summary["ordered_pair_hashes_sha256"]
        == experimental["content_identities"]["ordered_pair_hashes_sha256"],
        "experimental ordered pair identity drifted",
    )

    compiled_by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    family_split: dict[str, str] = {}
    for split in SPLITS:
        ids = split_ids.get(split)
        _require(isinstance(ids, list) and len(ids) == EXPECTED_COUNTS[split], f"{split} IDs drifted")
        output_path = resolve(data_manifest["outputs"][split], f"Day 23 {split}")
        rows = load_jsonl(output_path)
        _require(len(rows) == len(ids), f"Day 23 {split} count drifted")
        _require(
            data_manifest["outputs"][split].get("records") == len(ids),
            f"Day 23 {split} manifest record count drifted",
        )
        expected_rows: list[dict[str, Any]] = []
        for index, pair_id in enumerate(ids):
            _require(pair_id in by_id and pair_id not in seen, f"bad/duplicate pair ID: {pair_id}")
            pair = by_id[pair_id]
            _require(pair.get("split") == split, f"{pair_id}: source split drifted")
            creation = pair.get("creation", {})
            _require(
                creation.get("promoted_s1_on_policy") is True
                and creation.get("synthetic") is False,
                f"{pair_id}: off-policy or synthetic source",
            )
            for branch in ("chosen", "rejected"):
                generator = pair[branch]["generator"]
                expected_generator = {
                    "downstream_key": parent["downstream_key"],
                    "promotion_manifest_sha256": parent["promotion_manifest_sha256"],
                    "merged_export_manifest_sha256": parent[
                        "merged_export_manifest_sha256"
                    ],
                    "source_adapter_sha256": parent["adapter_sha256"],
                    "source_checkpoint_integrity_sha256": parent[
                        "checkpoint_integrity_sha256"
                    ],
                    "ms_swift_commit": MS_SWIFT_COMMIT,
                    "on_policy": True,
                    "promoted_s1": True,
                    "model_role": "merged_s1",
                }
                for field, value in expected_generator.items():
                    _require(generator.get(field) == value, f"{pair_id}:{branch}:{field} drifted")
            expected = _expected_row(pair, split)
            _require(rows[index] == expected, f"{pair_id}: compiled row projection drifted")
            verify_self(rows[index], "row_sha256", f"compiled {pair_id}")
            expected_rows.append(expected)
            compiled_by_id[pair_id] = expected
            seen.add(pair_id)
            family = str(pair["family_keys"]["problem"])
            prior = family_split.get(family)
            _require(prior is None or prior == split, f"family leaks across {prior}/{split}")
            family_split[family] = split
        output = data_manifest["outputs"][split]
        _require(
            output.get("ordered_pair_ids_sha256")
            == object_sha256([row["pair_id"] for row in expected_rows])
            and output.get("ordered_source_pair_hashes_sha256")
            == object_sha256([row["source_pair_sha256"] for row in expected_rows])
            and output.get("ordered_row_hashes_sha256")
            == object_sha256([row["row_sha256"] for row in expected_rows]),
            f"Day 23 {split} ordered identities drifted",
        )
    _require(seen == set(by_id), "compiled splits do not cover exactly Day 22 pairs")
    _require(
        data_manifest.get("counts")
        == {"pairs": 200, "branches": 400, "split_counts": EXPECTED_COUNTS},
        "Day 23 data aggregate counts drifted",
    )
    _require(
        data_manifest.get("claim_boundary")
        == {
            "formal_human_reviewed": False,
            "gpu_optimizer_ready": False,
            "remote_payload_gate": REMOTE_GATE,
            "gpu_runtime_gate": GPU_GATE,
        },
        "Day 23 data claim boundary drifted",
    )
    expected_compiler_sources = {
        "day23_contract.py": file_sha256(
            BOOTCAMP_ROOT / "day-23-dpo-theory-smoke/day23_contract.py"
        ),
        "prepare_day23_qwen35_dpo.py": file_sha256(
            BOOTCAMP_ROOT
            / "day-23-dpo-theory-smoke/prepare_day23_qwen35_dpo.py"
        ),
    }
    compiler = _mapping(data_manifest.get("compiler"), "Day 23 compiler")
    _require(
        compiler.get("source_files") == expected_compiler_sources
        and compiler.get("row_schema") == "day23.ms_swift_dpo_row"
        and compiler.get("projection")
        == "messages(user,chosen_assistant)+rejected_response"
        and compiler.get("preserve_text_bytes") is True
        and compiler.get("canonical_jsonl") == "utf8_sort_keys_compact_lf"
        and compiler.get("split_order_authority")
        == "day22.experimental_ai_assisted_split_ids"
        and compiler.get("ms_swift_loader_contract")
        == {
            "strict": True,
            "disable_auto_column_mapping": True,
            "remove_unused_columns": True,
            "split_dataset_ratio": 0,
        },
        "Day 23 compiler contract/source binding drifted",
    )
    return by_id, compiled_by_id


def _validate_processor(
    summary_path: Path,
    data_manifest_path: Path,
    pairs_by_id: Mapping[str, Mapping[str, Any]],
    compiled_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = load_json(summary_path)
    summary_sha = verify_self(summary, "summary_sha256", "processor summary")
    _require(
        summary.get("schema_name") == "day23.qwen35_dpo_processor_audit_summary"
        and summary.get("status") == "pass",
        "processor summary status drifted",
    )
    _require(
        summary.get("data_manifest", {}).get("file_sha256")
        == file_sha256(data_manifest_path)
        and summary.get("data_manifest", {}).get("manifest_sha256")
        == verify_self(
            load_json(data_manifest_path),
            "manifest_sha256",
            "processor input data manifest",
        ),
        "processor/data manifest binding drifted",
    )
    audit_path = resolve(summary["audit_rows"], "Day 23 processor audit rows")
    audits = load_jsonl(audit_path)
    _require(len(audits) == 200, "processor audit row count drifted")
    seen: set[str] = set()
    lengths: list[int] = []
    response_lengths: list[int] = []
    split_counts: Counter[str] = Counter()
    for audit in audits:
        pair_id = str(audit.get("pair_id"))
        _require(pair_id in pairs_by_id and pair_id not in seen, f"bad audit pair: {pair_id}")
        verify_self(audit, "audit_sha256", f"audit {pair_id}")
        _require(
            audit.get("compiled_row_sha256") == compiled_by_id[pair_id]["row_sha256"]
            and audit.get("source_pair_sha256") == pairs_by_id[pair_id]["pair_sha256"]
            and audit.get("split") == pairs_by_id[pair_id]["split"],
            f"{pair_id}: processor/source binding drifted",
        )
        _require(
            audit.get("prompt_prefix_exact_between_branches") is True
            and audit.get("day22_frozen_branch_evidence_exact") is True,
            f"{pair_id}: processor exactness flag drifted",
        )
        for branch in ("chosen", "rejected"):
            expected = pairs_by_id[pair_id]["processor_audit"][branch]
            _require(audit.get(branch) == expected, f"{pair_id}:{branch} token evidence drifted")
            _require(
                expected.get("response_only_mask") is True
                and expected.get("four_space_boundary") is True
                and expected.get("truncation") is False,
                f"{pair_id}:{branch} mask/truncation gate failed",
            )
            lengths.append(int(expected["input_token_count"]))
            response_lengths.append(int(expected["response_token_count"]))
        _require(
            audit["chosen"]["prompt_prefix_token_ids_sha256"]
            == audit["rejected"]["prompt_prefix_token_ids_sha256"],
            f"{pair_id}: chosen/rejected prompt prefixes differ",
        )
        seen.add(pair_id)
        split_counts[str(audit["split"])] += 1
    _require(seen == set(pairs_by_id), "processor audit coverage drifted")
    _require(
        summary["audit_rows"].get("ordered_audit_sha256")
        == object_sha256([audit["audit_sha256"] for audit in audits]),
        "ordered processor audit identity drifted",
    )
    expected_counts = {
        "pairs": 200,
        "branches": 400,
        "split_counts": EXPECTED_COUNTS,
        "prompt_prefix_exact_pairs": 200,
        "day22_frozen_exact_branches": 400,
        "response_only_mask_branches": 400,
        "four_space_boundary_branches": 400,
        "zero_truncation_branches": 400,
    }
    _require(summary.get("counts") == expected_counts, "processor aggregate counts drifted")
    _require(
        summary.get("lengths")
        == {
            "min_input_tokens": min(lengths),
            "max_input_tokens": max(lengths),
            "min_response_tokens": min(response_lengths),
            "max_response_tokens": max(response_lengths),
        }
        and max(lengths) == 381
        and max(lengths) <= 512,
        "processor length envelope drifted",
    )
    loader = _mapping(summary.get("dataset_loader"), "dataset loader")
    verify_self(loader, "loader_evidence_sha256", "dataset loader")
    _require(
        loader.get("input_counts") == EXPECTED_COUNTS
        and loader.get("output_counts") == EXPECTED_COUNTS
        and loader.get("silent_row_deletions") == 0
        and loader.get("strict") is True
        and loader.get("disable_auto_column_mapping") is True,
        "pinned ms-swift loader gate drifted",
    )
    _require(
        loader.get("implementation") == "swift.dataset.load_dataset"
        and loader.get("remove_unused_columns") is True
        and loader.get("split_dataset_ratio") == 0
        and loader.get("shuffle") is False
        and loader.get("standard_text_fields_exact") is True,
        "pinned ms-swift loader execution contract drifted",
    )
    template = _mapping(summary.get("template"), "processor template")
    _require(
        template.get("alias") == TEMPLATE_ALIAS
        and template.get("mode") == "rlhf"
        and template.get("contract_sha256") == TEMPLATE_CONTRACT_SHA256
        and object_sha256(template.get("contract")) == TEMPLATE_CONTRACT_SHA256
        and template.get("day20_source_contract_sha256") == DAY20_TEMPLATE_SHA256,
        "Day 23 RLHF template contract drifted",
    )
    _require(
        template.get("module_sha256")
        == file_sha256(
            BOOTCAMP_ROOT / "day-23-dpo-theory-smoke/day23_rlhf_template.py"
        ),
        "Day 23 RLHF template module binding drifted",
    )
    expected_runtime = {
        "python": "3.11.15",
        "transformers": "5.12.1",
        "torch": "2.9.1",
        "peft": "0.18.0",
        "trl": "0.29.1",
        "accelerate": "1.12.0",
        "datasets": "4.8.4",
        "huggingface_hub": "1.27.0",
        "ms_swift": "4.5.0.dev0",
    }
    _require(summary.get("runtime") == expected_runtime, "CPU processor runtime drifted")
    vendor = _mapping(summary.get("vendor_ms_swift"), "vendor ms-swift")
    _require(
        vendor.get("commit") == MS_SWIFT_COMMIT
        and vendor.get("clean") is True
        and vendor.get("imported_from_pinned_checkout") is True,
        "vendor identity drifted",
    )
    vendor_root = REPO_ROOT / "vendor/ms-swift"
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        actual_status = subprocess.run(
            ["git", "-C", str(vendor_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Day23ValidationError("cannot verify pinned ms-swift checkout") from error
    _require(
        actual_commit == MS_SWIFT_COMMIT and actual_status == "",
        "actual pinned ms-swift checkout commit/cleanliness drifted",
    )
    _require(
        set(vendor.get("critical_files_sha256", {})) == EXPECTED_VENDOR_FILES,
        "vendor critical file inventory drifted",
    )
    actual_files: dict[str, str] = {}
    for relative, expected in vendor.get("critical_files_sha256", {}).items():
        path = vendor_root / relative
        actual = file_sha256(path)
        _require(actual == expected, f"pinned ms-swift file drifted: {relative}")
        actual_files[str(relative)] = actual
    _require(
        object_sha256(actual_files) == vendor.get("critical_files_identity_sha256"),
        "vendor critical file aggregate drifted",
    )
    _require(
        vendor.get("package_init_sha256")
        == file_sha256(vendor_root / "swift/__init__.py"),
        "imported vendor package identity drifted",
    )
    rlhf_args = (vendor_root / "swift/arguments/rlhf_args.py").read_text(encoding="utf-8")
    mixin = (vendor_root / "swift/rlhf_trainers/rlhf_mixin.py").read_text(encoding="utf-8")
    pipeline = (vendor_root / "swift/pipelines/train/rlhf.py").read_text(encoding="utf-8")
    _require(
        "LoRA training does not require a ref_model" in rlhf_args
        and ".disable_adapter() if is_peft_model(" in mixin
        and "and not self.ref_adapter_name else nullcontext()" in mixin
        and "self.template.set_mode(mode_mapping.get(args.rlhf_type, 'rlhf'))" in pipeline,
        "pinned LoRA reference/RLHF-mode semantics drifted",
    )
    processor = _mapping(summary.get("processor"), "processor identity")
    snapshot = (
        REPO_ROOT
        / "tmp/qwen35-v2/models/Qwen--Qwen3.5-4B-Base"
        / "1001bb4d826a52d1f399e183466143f4da7b741b"
    )
    snapshot_entries = processor.get("snapshot_files", {})
    _require(
        isinstance(snapshot_entries, Mapping)
        and set(snapshot_entries) == set(EXPECTED_PROCESSOR_FILES),
        "processor asset inventory drifted",
    )
    actual_snapshot_entries: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in sorted(EXPECTED_PROCESSOR_FILES.items()):
        path = snapshot / relative
        entry = snapshot_entries[relative]
        actual = file_sha256(path)
        _require(
            actual == expected_sha
            and entry.get("sha256") == expected_sha
            and entry.get("bytes") == path.stat().st_size,
            f"processor asset drifted: {relative}",
        )
        actual_snapshot_entries[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    _require(
        processor.get("snapshot_files_sha256")
        == object_sha256(actual_snapshot_entries),
        "processor asset aggregate drifted",
    )
    expected_implementation_sources = {
        "audit_day23_qwen35_processor.py": file_sha256(
            BOOTCAMP_ROOT
            / "day-23-dpo-theory-smoke/audit_day23_qwen35_processor.py"
        ),
        "day20_target_encoding_v3.py": file_sha256(
            BOOTCAMP_ROOT
            / "day-20-qwen35-balanced-lora-sft/day20_target_encoding_v3.py"
        ),
        "day23_contract.py": file_sha256(
            BOOTCAMP_ROOT / "day-23-dpo-theory-smoke/day23_contract.py"
        ),
        "day23_rlhf_template.py": file_sha256(
            BOOTCAMP_ROOT / "day-23-dpo-theory-smoke/day23_rlhf_template.py"
        ),
    }
    _require(
        summary.get("implementation_sources") == expected_implementation_sources,
        "processor auditor implementation source binding drifted",
    )
    claim = summary.get("claim_boundary", {})
    _require(
        claim
        == {
            "cpu_processor_gate": "pass",
            "gpu_optimizer_ready": False,
            "gpu_runtime_gate": GPU_GATE,
            "model_weights_loaded": False,
            "remote_payload_gate": REMOTE_GATE,
        },
        "processor claim boundary drifted",
    )
    return {"summary_sha256": summary_sha, "summary": summary}


def _validate_scalar_oracle() -> None:
    def softplus(value: float) -> float:
        return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

    def loss(pc: float, pr: float, rc: float, rr: float, beta: float) -> float:
        return softplus(-beta * ((pc - pr) - (rc - rr)))

    _require(abs(loss(-3, -5, -2, -4, 0.1) - math.log(2)) < 1e-14, "zero-margin DPO oracle failed")
    _require(loss(-3, -5, -4, -5, 0.1) < loss(-4, -5, -4, -5, 0.1), "chosen direction failed")
    _require(loss(-5, -2, -4, -3, 0.1) > loss(-2, -5, -3, -4, 0.1), "swap direction failed")
    labels = [-100, -100, 257, 91, 248046, -100]
    first = sum(x for x, label in zip([-100, -200, -0.2, -0.3, -0.4, -300], labels) if label != -100)
    second = sum(x for x, label in zip([100, 200, -0.2, -0.3, -0.4, 300], labels) if label != -100)
    _require(abs(first + 0.9) < 1e-14 and first == second, "response-only sum mask failed")


def _validate_run_contract(
    run_contract_path: Path,
    data_manifest_path: Path,
    processor_summary_path: Path,
    processor_sha: str,
) -> dict[str, Any]:
    value = load_json(run_contract_path)
    digest = verify_self(value, "contract_sha256", "CPU run contract")
    _require(
        value.get("schema_name") == "day23.qwen35_coding_dpo_cpu_run_contract"
        and value.get("status") == "cpu_ready_gpu_pending",
        "CPU run contract status drifted",
    )
    _require(
        value["inputs"]["data_manifest"].get("file_sha256") == file_sha256(data_manifest_path)
        and value["inputs"]["data_manifest"].get("content_sha256")
        == verify_self(
            load_json(data_manifest_path), "manifest_sha256", "run input data manifest"
        )
        and value["inputs"]["processor_audit"].get("file_sha256")
        == file_sha256(processor_summary_path)
        and value["inputs"]["processor_audit"].get("content_sha256") == processor_sha,
        "run contract input binding drifted",
    )
    implementation_paths = {
        "audit_day23_qwen35_arguments.py": BOOTCAMP_ROOT
        / "day-23-dpo-theory-smoke/audit_day23_qwen35_arguments.py",
        "audit_day23_qwen35_processor.py": BOOTCAMP_ROOT
        / "day-23-dpo-theory-smoke/audit_day23_qwen35_processor.py",
        "build_day23_cpu_contract.py": BOOTCAMP_ROOT
        / "day-23-dpo-theory-smoke/build_day23_cpu_contract.py",
        "day20_target_encoding_v3.py": BOOTCAMP_ROOT
        / "day-20-qwen35-balanced-lora-sft/day20_target_encoding_v3.py",
        "day23_contract.py": BOOTCAMP_ROOT / "day-23-dpo-theory-smoke/day23_contract.py",
        "day23_dpo_math.py": BOOTCAMP_ROOT / "day-23-dpo-theory-smoke/day23_dpo_math.py",
        "day23_ms_swift_plugin.py": BOOTCAMP_ROOT
        / "day-23-dpo-theory-smoke/day23_ms_swift_plugin.py",
        "day23_rlhf_template.py": BOOTCAMP_ROOT
        / "day-23-dpo-theory-smoke/day23_rlhf_template.py",
        "prepare_day23_qwen35_dpo.py": BOOTCAMP_ROOT
        / "day-23-dpo-theory-smoke/prepare_day23_qwen35_dpo.py",
        "test_day23_qwen35_dpo_loss.py": BOOTCAMP_ROOT
        / "artifacts/scripts/test_day23_qwen35_dpo_loss.py",
        "validate_day23_qwen35_dpo.py": BOOTCAMP_ROOT
        / "artifacts/scripts/validate_day23_qwen35_dpo.py",
    }
    _require(
        value.get("implementation_sources")
        == {name: file_sha256(path) for name, path in sorted(implementation_paths.items())},
        "run contract implementation source binding drifted",
    )
    args = _mapping(value.get("intended_ms_swift_args"), "intended ms-swift args")
    expected_args = {
        "add_non_thinking_prefix": True,
        "add_version": False,
        "beta": 0.1,
        "bf16": True,
        "data_seed": 20260818,
        "dataset": ["artifacts/data/day23-qwen35-coding-dpo-train.jsonl"],
        "disable_auto_column_mapping": True,
        "enable_thinking": False,
        "external_plugins": [
            "day-23-dpo-theory-smoke/day23_ms_swift_plugin.py"
        ],
        "freeze_aligner": True,
        "freeze_llm": False,
        "freeze_vit": True,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 5e-6,
        "load_best_model_at_end": False,
        "logging_steps": 1,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_rank": 8,
        "loss_scale": "default+ignore_empty_think",
        "loss_type": "sigmoid",
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "max_length": 512,
        "model": "__BIND_VERIFIED_REMOTE_S1_MERGED_EXPORT__",
        "model_type": "qwen3_5",
        "packing": False,
        "padding_free": False,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "remove_unused_columns": True,
        "report_to": ["tensorboard"],
        "rlhf_type": "dpo",
        "save_only_model": False,
        "save_strategy": "steps",
        "seed": 20260818,
        "split_dataset_ratio": 0,
        "strict": True,
        "target_regex": (
            r"^(?:(?:base_model|model)\.)*language_model\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))$"
        ),
        "template": TEMPLATE_ALIAS,
        "truncation_strategy": "delete",
        "tuner_type": "lora",
        "val_dataset": ["artifacts/data/day23-qwen35-coding-dpo-dev.jsonl"],
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
    }
    _require(dict(args) == expected_args, "complete intended ms-swift args drifted")
    _require(
        all(key not in args for key in ("adapters", "ref_adapters", "ref_model")),
        "null/empty reference defaults leaked into JSON arguments",
    )
    _require(
        value.get("cli_default_semantics", {}).get("omitted_keys")
        == {
            "adapters": [],
            "ref_adapters": [],
            "ref_model": None,
            "resume_from_checkpoint": None,
        },
        "JSON CLI default omission contract drifted",
    )
    usage = value["dataset_usage"]
    _require(
        usage
        == {
            "train": {
                "path": "artifacts/data/day23-qwen35-coding-dpo-train.jsonl",
                "role": "optimizer_input",
            },
            "dev": {
                "path": "artifacts/data/day23-qwen35-coding-dpo-dev.jsonl",
                "role": "checkpoint_selection_only",
            },
            "heldout": {
                "path": "artifacts/data/day23-qwen35-coding-dpo-heldout.jsonl",
                "role": "one_shot_confirmation_after_selection",
                "present_in_intended_ms_swift_args": False,
            },
        },
        "dataset usage contract drifted",
    )
    _require(args.get("dataset") == [usage["train"]["path"]], "train dataset binding drifted")
    _require(args.get("val_dataset") == [usage["dev"]["path"]], "dev dataset binding drifted")
    _require(
        usage["heldout"]["path"] not in json.dumps(args, sort_keys=True)
        and usage["heldout"]["present_in_intended_ms_swift_args"] is False,
        "heldout leaked into trainer/checkpoint selection",
    )
    mechanism = value["stages"]["mechanism_5step"]
    smoke = value["stages"]["bounded_smoke_30step"]
    expected_mechanism_ids = [
        str(row["pair_id"])
        for row in load_jsonl(BOOTCAMP_ROOT / usage["train"]["path"])[:4]
    ]
    expected_stages = {
        "mechanism_5step": {
            "dataset": [
                "artifacts/data/day23-qwen35-coding-dpo-train.jsonl#4"
            ],
            "dataset_shuffle": False,
            "eval_strategy": "no",
            "fresh_start_from_parent": True,
            "max_steps": 5,
            "mechanism_pair_ids": expected_mechanism_ids,
            "output_dir": "__BIND_DAY23_MECHANISM_OUTPUT_DIR__",
            "purpose": "mechanism_only_not_checkpoint_candidate",
            "remove_common_args": ["val_dataset"],
            "save_steps": 5,
            "save_total_limit": 1,
            "train_dataloader_shuffle": False,
        },
        "bounded_smoke_30step": {
            "checkpoint_candidates": [15, 30],
            "dataset_shuffle": True,
            "eval_steps": 15,
            "eval_strategy": "steps",
            "fresh_start_from_parent": True,
            "max_steps": 30,
            "output_dir": "__BIND_DAY23_SMOKE_OUTPUT_DIR__",
            "purpose": "bounded_smoke_checkpoint_selection",
            "save_steps": 15,
            "save_total_limit": 2,
            "train_dataloader_shuffle": True,
        },
    }
    _require(value.get("stages") == expected_stages, "complete run stages drifted")
    _require(
        mechanism.get("fresh_start_from_parent") is True
        and "resume_from_checkpoint" not in mechanism
        and mechanism.get("remove_common_args") == ["val_dataset"]
        and mechanism.get("dataset") == [f"{usage['train']['path']}#4"]
        and mechanism.get("mechanism_pair_ids") == expected_mechanism_ids
        and mechanism.get("dataset_shuffle") is False
        and mechanism.get("train_dataloader_shuffle") is False
        and mechanism.get("max_steps") == 5
        and mechanism.get("purpose") == "mechanism_only_not_checkpoint_candidate",
        "5-step mechanism stage drifted",
    )
    _require(
        smoke.get("fresh_start_from_parent") is True
        and "resume_from_checkpoint" not in smoke
        and smoke.get("dataset_shuffle") is True
        and smoke.get("train_dataloader_shuffle") is True
        and smoke.get("max_steps") == 30
        and smoke.get("checkpoint_candidates") == [15, 30],
        "30-step bounded smoke stage drifted",
    )
    claim = value.get("claim_boundary", {})
    _require(
        claim
        == {
            "cpu_contract_ready": True,
            "formal_human_reviewed": False,
            "gpu_memory_preflight_passed": False,
            "gpu_optimizer_ready": False,
            "model_weights_loaded_on_cpu": False,
            "optimizer_step_run": False,
            "reference_immutability_runtime_proven": False,
            "remote_s1_payload_verified_now": False,
        },
        "CPU/GPU claim boundary drifted",
    )
    expected_cpu_gates = {
        "day21_compact_ancestry": "pass",
        "day22_independent_experimental_close": "pass",
        "deterministic_split_compilation": "pass",
        "pinned_ms_swift_loader_no_row_deletion": "pass",
        "real_qwen35_rlhf_template_400_branches": "pass",
        "day22_token_label_exact_match_400_branches": "pass",
        "response_mask_and_four_space_boundary": "pass",
        "max_length_512_zero_truncation": "pass",
        "scalar_dpo_oracle": "pass_via_required_pinned_test_script",
    }
    _require(
        value.get("cpu_gates") == expected_cpu_gates,
        "CPU gate status drifted",
    )
    _require(
        value.get("binding_rule")
        == {
            "binding_and_runtime_attestation_must_be_self_hashed": True,
            "cpu_contract_is_not_direct_swift_config": True,
            "dataset_val_dataset_and_plugin_paths_must_be_absolute": True,
            "executable_config_must_contain_only_pinned_ms_swift_argument_keys": True,
            "executable_config_must_omit_cli_default_keys": [
                "adapters",
                "ref_adapters",
                "ref_model",
                "resume_from_checkpoint",
            ],
            "executable_config_must_omit_null_and_empty_list_values": True,
            "gpu_binding_must_replace_all_double_underscore_placeholders": True,
            "relative_path_base": "bootcamp_root",
            "stage_metadata_keys_to_remove": [
                "checkpoint_candidates",
                "fresh_start_from_parent",
                "mechanism_pair_ids",
                "purpose",
                "remove_common_args",
            ],
        },
        "GPU config binding rule drifted",
    )
    _validate_scalar_oracle()
    return {"contract_sha256": digest, "run_contract": value}


def _validate_argument_audit(
    argument_audit_path: Path,
    run_contract_path: Path,
    run_contract_sha: str,
) -> str:
    value = load_json(argument_audit_path)
    run_value = load_json(run_contract_path)
    digest = verify_self(value, "audit_sha256", "argument parse audit")
    _require(
        value.get("schema_name") == "day23.qwen35_dpo_argument_parse_audit"
        and value.get("status") == "pass"
        and value.get("pinned_ms_swift_commit") == MS_SWIFT_COMMIT,
        "argument parse audit status drifted",
    )
    _require(
        value.get("scope")
        == "pinned_real_JSON_CLI_and_external_plugin_parse_no_model_weights"
        and value.get("imported_ms_swift_from_pinned_vendor") is True,
        "argument audit did not use the pinned real JSON CLI",
    )
    _require(
        value.get("run_contract", {}).get("file_sha256")
        == file_sha256(run_contract_path)
        and value.get("run_contract", {}).get("contract_sha256")
        == run_contract_sha,
        "argument audit/run contract binding drifted",
    )
    _require(
        value.get("rlhf_argument_field_count") == 474
        and value.get("unknown_common_argument_keys") == []
        and value.get("unknown_stage_argument_keys")
        == {"mechanism_5step": [], "bounded_smoke_30step": []},
        "pinned RLHFArguments schema audit drifted",
    )
    for stage_name, max_steps in (
        ("mechanism_5step", 5),
        ("bounded_smoke_30step", 30),
    ):
        stage = value.get("stages", {}).get(stage_name, {})
        _require(
            stage.get("status") == "pass"
            and stage.get("entrypoint")
            == "swift.cli.main.parse_yaml_args_then_swift.utils.parse_args"
            and stage.get("max_steps") == max_steps
            and stage.get("template") == TEMPLATE_ALIAS
            and stage.get("rlhf_type") == "dpo"
            and stage.get("tuner_type") == "lora"
            and stage.get("adapters") == []
            and stage.get("ref_model") is None
            and stage.get("ref_adapters") == []
            and stage.get("resume_from_checkpoint") is None
            and stage.get("omitted_cli_default_keys")
            == ["adapters", "ref_adapters", "ref_model", "resume_from_checkpoint"]
            and stage.get("external_plugin_registered") is True
            and stage.get("add_version") is False
            and stage.get("save_strategy") == "steps"
            and stage.get("save_only_model") is False
            and stage.get("load_best_model_at_end") is False
            and stage.get("sync_ref_model") is False
            and stage.get("reference_free") in (None, False)
            and stage.get("precompute_ref_log_probs") is False
            and stage.get("disable_dropout") is True
            and stage.get("weights_loaded") is False,
            f"{stage_name} parsed argument evidence drifted",
        )
        _require(
            not any(
                key in stage.get("executable_config_keys", [])
                for key in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint")
            ),
            f"{stage_name} executable JSON contains omitted-default keys",
        )
    mechanism = value["stages"]["mechanism_5step"]
    _require(
        mechanism.get("val_dataset_syntax") == []
        and mechanism.get("dataset_shuffle") is False
        and mechanism.get("train_dataloader_shuffle") is False
        and mechanism.get("mechanism_loader", {}).get("status") == "pass"
        and mechanism.get("mechanism_loader", {}).get("records") == 4
        and mechanism.get("mechanism_loader", {}).get("validation_records") == 0
        and mechanism.get("mechanism_loader", {}).get("standard_text_fields_exact")
        is True
        and mechanism.get("mechanism_loader", {}).get("pair_ids")
        == run_value["stages"]["mechanism_5step"]["mechanism_pair_ids"],
        "mechanism dataset#4/dev/shuffle audit drifted",
    )
    smoke = value["stages"]["bounded_smoke_30step"]
    _require(
        smoke.get("val_dataset_syntax")
        == ["artifacts/data/day23-qwen35-coding-dpo-dev.jsonl"]
        and smoke.get("dataset_shuffle") is True
        and smoke.get("train_dataloader_shuffle") is True,
        "bounded smoke train/dev/shuffle audit drifted",
    )
    _require(
        value.get("claim_boundary")
        == {
            "argument_schema_and_plugin_parse": "pass",
            "model_weights_loaded": False,
            "local_base_used_as_policy_parent": False,
            "gpu_optimizer_ready": False,
            "remaining_gates": [REMOTE_GATE, GPU_GATE],
        },
        "argument parse claim boundary drifted",
    )
    return digest


def validate_bundle(
    *,
    data_manifest_path: Path = DEFAULT_DATA_MANIFEST,
    processor_summary_path: Path = DEFAULT_PROCESSOR_SUMMARY,
    run_contract_path: Path = DEFAULT_RUN_CONTRACT,
    argument_audit_path: Path = DEFAULT_ARGUMENT_AUDIT,
) -> dict[str, Any]:
    data_manifest_path = data_manifest_path.resolve()
    processor_summary_path = processor_summary_path.resolve()
    run_contract_path = run_contract_path.resolve()
    argument_audit_path = argument_audit_path.resolve()
    data_manifest = load_json(data_manifest_path)
    data_sha = verify_self(data_manifest, "manifest_sha256", "Day 23 data manifest")
    _require(
        data_manifest.get("schema_name") == "day23.qwen35_coding_dpo_data_manifest"
        and data_manifest.get("status") == "cpu_data_ready",
        "Day 23 data manifest status drifted",
    )
    parent = _validate_day21(data_manifest)
    pairs_by_id, compiled_by_id = _validate_day22_and_compiled(data_manifest, parent)
    processor = _validate_processor(
        processor_summary_path,
        data_manifest_path,
        pairs_by_id,
        compiled_by_id,
    )
    run = _validate_run_contract(
        run_contract_path,
        data_manifest_path,
        processor_summary_path,
        processor["summary_sha256"],
    )
    argument_audit_sha = _validate_argument_audit(
        argument_audit_path,
        run_contract_path,
        run["contract_sha256"],
    )
    claim = data_manifest.get("claim_boundary", {})
    _require(
        claim.get("gpu_optimizer_ready") is False
        and claim.get("remote_payload_gate") == REMOTE_GATE
        and claim.get("gpu_runtime_gate") == GPU_GATE,
        "data manifest CPU/GPU claim boundary drifted",
    )
    return {
        "status": "valid_cpu_ready_gpu_pending",
        "cpu_ready": True,
        "gpu_optimizer_ready": False,
        "formal_dpo_ready": False,
        "pairs": 200,
        "branches": 400,
        "split_counts": EXPECTED_COUNTS,
        "max_input_tokens": 381,
        "day22_exact_branches": 400,
        "data_manifest_sha256": data_sha,
        "processor_summary_sha256": processor["summary_sha256"],
        "run_contract_sha256": run["contract_sha256"],
        "argument_audit_sha256": argument_audit_sha,
        "remaining_gates": [REMOTE_GATE, GPU_GATE],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--processor-summary", type=Path, default=DEFAULT_PROCESSOR_SUMMARY)
    parser.add_argument("--run-contract", type=Path, default=DEFAULT_RUN_CONTRACT)
    parser.add_argument("--argument-audit", type=Path, default=DEFAULT_ARGUMENT_AUDIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_bundle(
            data_manifest_path=args.data_manifest,
            processor_summary_path=args.processor_summary,
            run_contract_path=args.run_contract,
            argument_audit_path=args.argument_audit,
        )
    except (
        OSError,
        Day23ValidationError,
        day22_contract.Day22ContractError,
        validate_day22_experimental_close.ExperimentalValidationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
