#!/usr/bin/env python3
"""Standard-library contracts for the Day 23 Qwen3.5 coding DPO CPU gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ROW_SCHEMA_NAME = "day23.ms_swift_dpo_row"
DATA_MANIFEST_SCHEMA_NAME = "day23.qwen35_coding_dpo_data_manifest"
DATASET_TRACK = "experimental_ai_assisted"
SPLITS = ("train", "dev", "heldout")
EXPECTED_SPLIT_COUNTS = {"train": 154, "dev": 17, "heldout": 29}
MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
EXPECTED_UPSTREAM_FILE_SHA256 = {
    "day21_promotion": "40c76f690dcb73805250e0036f8e124ef07b8d2d32b1723637656ae4304d94f8",
    "day21_downstream_key": "979e599aa6c9c05bec9a14144b6ebae689dbdf8315a9112caaec09b5469bd545",
    "day21_merged_export": "9c196e43f633116d7fc804870dcfe961db5ae0d216a15f5e21d679d8c1f14b5a",
    "day22_experimental": "386c26c2f548f4b57c56ecb8265f36e83a89278bf1d7cf66bee6b0dfac819bcc",
}
REMOTE_PAYLOAD_GATE = "pending_remote_s1_payload_verification"
GPU_RUNTIME_GATE = "pending_gpu_reference_lora_memory_save_reload"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day23ContractError(ValueError):
    """A Day 23 ancestry, dataset, or output invariant failed."""


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
        raise Day23ContractError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise Day23ContractError("text hash input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day23ContractError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty text")
    _require("\x00" not in value, f"{label} contains a NUL byte")
    return str(value)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase bare SHA-256",
    )
    return str(value)


def verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256(value, field)
    _require(expected == actual, f"{label}.{field} does not bind its contents")
    return expected


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day23ContractError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day23ContractError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(bool(line), f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day23ContractError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from error
        _require(isinstance(row, dict), f"non-object JSONL row: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def validate_s1_bundle(
    promotion: Mapping[str, Any],
    downstream_key: Mapping[str, Any],
    merged_export: Mapping[str, Any],
    *,
    promotion_file_sha256: str,
    downstream_key_file_sha256: str,
    merged_export_file_sha256: str,
) -> dict[str, Any]:
    """Validate the local compact S1 evidence without resolving remote paths."""
    promotion_sha = verify_self_hash(
        promotion, "promotion_manifest_sha256", "Day 21 promotion"
    )
    key_sha = verify_self_hash(downstream_key, "key_sha256", "Day 21 downstream key")
    export_sha = verify_self_hash(
        merged_export, "manifest_sha256", "Day 21 merged export"
    )
    promotion_file_sha256 = _sha256(
        promotion_file_sha256, "promotion_file_sha256"
    )
    downstream_key_file_sha256 = _sha256(
        downstream_key_file_sha256, "downstream_key_file_sha256"
    )
    merged_export_file_sha256 = _sha256(
        merged_export_file_sha256, "merged_export_file_sha256"
    )
    _require(
        promotion_file_sha256 == EXPECTED_UPSTREAM_FILE_SHA256["day21_promotion"],
        "Day 21 promotion does not match the frozen Day 23 trust root",
    )
    _require(
        merged_export_file_sha256
        == EXPECTED_UPSTREAM_FILE_SHA256["day21_merged_export"],
        "Day 21 merged export does not match the frozen Day 23 trust root",
    )
    _require(
        downstream_key_file_sha256
        == EXPECTED_UPSTREAM_FILE_SHA256["day21_downstream_key"],
        "Day 21 downstream key does not match the frozen Day 23 trust root",
    )

    _require(
        promotion.get("schema_name")
        == "day21.qwen35_downstream_ready_s1_promotion",
        "Day 21 promotion schema drifted",
    )
    _require(promotion.get("status") == "promoted", "Day 21 S1 is not promoted")
    _require(promotion.get("downstream_ready") is True, "Day 21 S1 is not downstream-ready")
    _require(promotion.get("role") == "S1", "DPO parent must be the promoted S1 role")

    lineage = _mapping(promotion.get("lineage"), "promotion.lineage")
    state = _mapping(promotion.get("state"), "promotion.state")
    training = _mapping(promotion.get("training"), "promotion.training")
    inference = _mapping(promotion.get("inference_export"), "promotion.inference_export")
    selection = _mapping(promotion.get("selection"), "promotion.selection")
    _require(lineage.get("model_id") == MODEL_ID, "S1 model ID drifted")
    _require(lineage.get("model_revision") == MODEL_REVISION, "S1 model revision drifted")
    _require(lineage.get("objective") == "coding_sft", "DPO parent is not coding SFT")
    _require(
        lineage.get("training_scope")
        == "text_only_coding_lora_vision_and_aligner_frozen",
        "S1 training scope drifted",
    )
    _require(training.get("framework_sha") == MS_SWIFT_COMMIT, "S1 ms-swift commit drifted")
    _require(training.get("packing") is False, "S1 packing identity drifted")
    _require(
        state.get("candidate") == selection.get("primary_candidate"),
        "S1 selected candidate and resumable state differ",
    )
    _require(
        all(
            state.get(field) is True
            for field in (
                "optimizer_state_present",
                "rng_state_present",
                "scheduler_state_present",
                "trainer_state_present",
            )
        ),
        "S1 resumable state evidence is incomplete",
    )
    _require(
        inference.get("fresh_process_exact_token_id_parity_passed") is True
        and inference.get("text_only_modality_parity_passed") is True
        and inference.get("processor_assets_present") is True,
        "S1 merged-export parity evidence is incomplete",
    )

    _require(
        downstream_key.get("schema_name") == "day21.qwen35_s1_downstream_key"
        and downstream_key.get("status") == "active"
        and downstream_key.get("role") == "S1",
        "Day 21 downstream key status drifted",
    )
    for field in ("checkpoint_id", "downstream_key"):
        _require(
            downstream_key.get(field) == promotion.get(field),
            f"Day 21 downstream key {field} is not bound to promotion",
        )
    key_binding = _mapping(
        downstream_key.get("promotion_manifest"), "downstream_key.promotion_manifest"
    )
    _require(
        key_binding.get("content_sha256") == promotion_sha,
        "downstream key promotion content hash drifted",
    )
    _require(
        key_binding.get("file_sha256") == promotion_file_sha256,
        "downstream key promotion file hash drifted",
    )
    # key_binding.path is deliberately historical provenance. It is remote and
    # is not resolved during the CPU-only gate.

    _require(
        merged_export.get("schema_name") == "day21.qwen35_s1_merged_export_manifest"
        and merged_export.get("status") == "pass",
        "Day 21 merged export status drifted",
    )
    conversion = _mapping(merged_export.get("conversion"), "merged_export.conversion")
    export_source = _mapping(merged_export.get("source"), "merged_export.source")
    _require(conversion.get("merge_lora") is True, "S1 export is not a merged LoRA export")
    _require(conversion.get("ms_swift_commit") == MS_SWIFT_COMMIT, "export ms-swift commit drifted")
    _require(merged_export.get("processor_assets_present") is True, "export processor assets missing")
    inference_manifest = _mapping(inference.get("manifest"), "inference_export.manifest")
    _require(inference_manifest.get("content_sha256") == export_sha, "promotion/export content hash drifted")
    _require(
        inference_manifest.get("file_sha256") == merged_export_file_sha256,
        "promotion/export file hash drifted",
    )
    _require(
        inference.get("artifact_hash") == merged_export.get("files_sha256"),
        "promotion/export artifact hash drifted",
    )
    bindings = {
        "adapter_sha256": "adapter_sha256",
        "resumable_checkpoint_integrity_sha256": "checkpoint_integrity_sha256",
        "resumable_checkpoint_snapshot_sha256": "checkpoint_snapshot_sha256",
    }
    for state_field, export_field in bindings.items():
        _require(
            state.get(state_field) == export_source.get(export_field),
            f"promotion/export {state_field} binding drifted",
        )

    return {
        "checkpoint_id": promotion["checkpoint_id"],
        "role": "S1",
        "downstream_key": promotion["downstream_key"],
        "promotion_file_sha256": promotion_file_sha256,
        "promotion_manifest_sha256": promotion_sha,
        "downstream_key_sha256": key_sha,
        "downstream_key_file_sha256": downstream_key_file_sha256,
        "checkpoint_snapshot_sha256": state["resumable_checkpoint_snapshot_sha256"],
        "checkpoint_integrity_sha256": state[
            "resumable_checkpoint_integrity_sha256"
        ],
        "adapter_sha256": state["adapter_sha256"],
        "merged_export_artifact_sha256": inference["artifact_hash"],
        "merged_export_file_sha256": merged_export_file_sha256,
        "merged_export_manifest_sha256": export_sha,
        "ms_swift_commit": MS_SWIFT_COMMIT,
        "remote_merged_export_path": inference["path"],
        "remote_resumable_checkpoint_path": state["resumable_checkpoint_path"],
    }


def validate_experimental_manifest(
    manifest: Mapping[str, Any], *, manifest_file_sha256: str
) -> dict[str, Any]:
    manifest_sha = verify_self_hash(
        manifest, "manifest_sha256", "Day 22 experimental manifest"
    )
    _require(
        _sha256(manifest_file_sha256, "experimental manifest file hash")
        == EXPECTED_UPSTREAM_FILE_SHA256["day22_experimental"],
        "Day 22 experimental manifest does not match the frozen Day 23 trust root",
    )
    _require(
        manifest.get("schema_name")
        == "day22.experimental_ai_assisted_close_manifest",
        "Day 22 experimental schema drifted",
    )
    _require(
        manifest.get("status") == "completed_experimental_ai_assisted",
        "Day 22 experimental close is incomplete",
    )
    _require(
        manifest.get("dataset_role") == "experimental_ai_assisted_dpo_input",
        "Day 22 experimental role drifted",
    )
    _require(manifest.get("experimental_dpo_ready") is True, "experimental DPO readiness is false")
    _require(manifest.get("formal_dpo_ready") is False, "formal DPO must remain unclaimed")
    _require(
        manifest.get("formal_human_review_ready") is False,
        "formal human review must remain unclaimed",
    )
    _require(
        manifest.get("policy", {}).get("formal_gate_waived_for_experimental_track")
        is True,
        "experimental authorization is absent",
    )
    _require(
        manifest.get("counts", {}).get("split_counts") == EXPECTED_SPLIT_COUNTS,
        "Day 22 split counts drifted",
    )
    return {
        "file_sha256": str(manifest_file_sha256),
        "manifest_sha256": manifest_sha,
        "accepted_pair_ids_sha256": manifest["content_identities"][
            "accepted_pair_ids_sha256"
        ],
        "ordered_pair_hashes_sha256": manifest["content_identities"][
            "ordered_pair_hashes_sha256"
        ],
        "split_ids_sha256": manifest["content_identities"]["split_ids_sha256"],
        "pair_file_sha256": manifest["outputs"]["pairs"]["file_sha256"],
        "split_file_sha256": manifest["outputs"]["split_ids"]["file_sha256"],
    }


def validate_pair_lineage(pair: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    pair_id = _text(pair.get("pair_id"), "pair_id")
    _require(pair.get("pair_status") == "accepted", f"{pair_id}: pair is not accepted")
    creation = _mapping(pair.get("creation"), f"{pair_id}.creation")
    _require(
        creation.get("method") == "promoted_s1_on_policy_rollout_pair"
        and creation.get("promoted_s1_on_policy") is True
        and creation.get("synthetic") is False,
        f"{pair_id}: pair is not non-synthetic promoted-S1 on-policy data",
    )
    for branch in ("chosen", "rejected"):
        response = _mapping(pair.get(branch), f"{pair_id}.{branch}")
        _require(
            response.get("origin") == "promoted_s1_rollout",
            f"{pair_id}:{branch} origin drifted",
        )
        generator = _mapping(response.get("generator"), f"{pair_id}.{branch}.generator")
        expected = {
            "downstream_key": parent["downstream_key"],
            "lineage_manifest_file_sha256": parent["promotion_file_sha256"],
            "promotion_manifest_sha256": parent["promotion_manifest_sha256"],
            "merged_export_manifest_sha256": parent[
                "merged_export_manifest_sha256"
            ],
            "source_adapter_sha256": parent["adapter_sha256"],
            "source_checkpoint_integrity_sha256": parent[
                "checkpoint_integrity_sha256"
            ],
            "ms_swift_commit": parent["ms_swift_commit"],
            "model_role": "merged_s1",
            "on_policy": True,
            "promoted_s1": True,
        }
        for field, value in expected.items():
            _require(
                generator.get(field) == value,
                f"{pair_id}:{branch} generator {field} drifted",
            )


def validate_split_ids(
    split_ids: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    verify_self_hash(split_ids, "split_ids_sha256", "Day 22 split IDs")
    _require(
        split_ids.get("schema_name") == "day22.experimental_ai_assisted_split_ids",
        "experimental split schema drifted",
    )
    _require(
        split_ids.get("dataset_role") == "experimental_ai_assisted_dpo_input",
        "experimental split role drifted",
    )
    by_id = {str(pair.get("pair_id")): pair for pair in pairs}
    _require(len(by_id) == len(pairs), "source pair IDs are not unique")
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    family_splits: dict[str, str] = {}
    for split in SPLITS:
        ids = split_ids.get(split)
        _require(isinstance(ids, list), f"{split} IDs must be a list")
        _require(len(ids) == EXPECTED_SPLIT_COUNTS[split], f"{split} count drifted")
        _require(len(ids) == len(set(ids)), f"{split} IDs contain duplicates")
        result[split] = []
        expected_families: list[str] = []
        for pair_id in ids:
            _require(isinstance(pair_id, str) and pair_id in by_id, f"unknown {split} pair ID")
            _require(pair_id not in seen, f"pair ID appears in multiple splits: {pair_id}")
            seen.add(pair_id)
            pair = by_id[pair_id]
            _require(pair.get("split") == split, f"{pair_id}: source split drifted")
            family = str(pair.get("family_keys", {}).get("problem"))
            prior = family_splits.get(family)
            _require(prior is None or prior == split, f"family leaks across {prior}/{split}: {family}")
            family_splits[family] = split
            expected_families.append(family)
            result[split].append(pair_id)
        _require(
            split_ids.get("family_ids", {}).get(split) == expected_families,
            f"{split} family IDs drifted",
        )
    _require(seen == set(by_id), "split IDs do not cover exactly the source pairs")
    return result


def compile_row(pair: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    pair_id = _text(pair.get("pair_id"), "pair_id")
    _require(split in SPLITS and pair.get("split") == split, f"{pair_id}: split drifted")
    prompt = _mapping(pair.get("prompt"), f"{pair_id}.prompt")
    chosen = _mapping(pair.get("chosen"), f"{pair_id}.chosen")
    rejected = _mapping(pair.get("rejected"), f"{pair_id}.rejected")
    prompt_text = _text(prompt.get("text"), f"{pair_id}.prompt.text")
    chosen_text = _text(chosen.get("text"), f"{pair_id}.chosen.text")
    rejected_text = _text(rejected.get("text"), f"{pair_id}.rejected.text")
    _require(prompt.get("sha256") == text_sha256(prompt_text), f"{pair_id}: prompt hash drifted")
    _require(chosen.get("sha256") == text_sha256(chosen_text), f"{pair_id}: chosen hash drifted")
    _require(rejected.get("sha256") == text_sha256(rejected_text), f"{pair_id}: rejected hash drifted")
    _require(chosen_text != rejected_text, f"{pair_id}: chosen/rejected are identical")
    row: dict[str, Any] = {
        "schema_name": ROW_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dataset_track": DATASET_TRACK,
        "pair_id": pair_id,
        "split": split,
        "source_pair_sha256": _sha256(pair.get("pair_sha256"), f"{pair_id}.pair_sha256"),
        "family_keys": dict(_mapping(pair.get("family_keys"), f"{pair_id}.family_keys")),
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_text},
        ],
        "rejected_response": rejected_text,
    }
    row["row_sha256"] = object_sha256(row)
    return row


def validate_compiled_row(
    row: Mapping[str, Any], *, source_pair: Mapping[str, Any] | None = None
) -> None:
    row_id = _text(row.get("pair_id"), "compiled pair_id")
    verify_self_hash(row, "row_sha256", f"compiled row {row_id}")
    _require(row.get("schema_name") == ROW_SCHEMA_NAME, f"{row_id}: row schema drifted")
    _require(row.get("schema_version") == SCHEMA_VERSION, f"{row_id}: row version drifted")
    _require(row.get("dataset_track") == DATASET_TRACK, f"{row_id}: dataset track drifted")
    _require(row.get("split") in SPLITS, f"{row_id}: invalid split")
    messages = row.get("messages")
    _require(isinstance(messages, list) and len(messages) == 2, f"{row_id}: messages drifted")
    _require(
        [message.get("role") for message in messages if isinstance(message, Mapping)]
        == ["user", "assistant"],
        f"{row_id}: roles drifted",
    )
    _text(messages[0].get("content"), f"{row_id}.user")
    chosen = _text(messages[1].get("content"), f"{row_id}.assistant")
    rejected = _text(row.get("rejected_response"), f"{row_id}.rejected_response")
    _require(chosen != rejected, f"{row_id}: chosen/rejected are identical")
    if source_pair is not None:
        expected = compile_row(source_pair, split=str(source_pair.get("split")))
        _require(dict(row) == expected, f"{row_id}: compiled row is not byte-semantic source projection")


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "compiled split is empty")
    ids: list[str] = []
    source_hashes: list[str] = []
    row_hashes: list[str] = []
    counts: Counter[str] = Counter()
    for row in rows:
        validate_compiled_row(row)
        ids.append(str(row["pair_id"]))
        source_hashes.append(str(row["source_pair_sha256"]))
        row_hashes.append(str(row["row_sha256"]))
        counts[str(row["split"])] += 1
    _require(len(ids) == len(set(ids)), "compiled split has duplicate pair IDs")
    _require(len(counts) == 1, "compiled file mixes splits")
    return {
        "records": len(rows),
        "ordered_pair_ids_sha256": object_sha256(ids),
        "ordered_source_pair_hashes_sha256": object_sha256(source_hashes),
        "ordered_row_hashes_sha256": object_sha256(row_hashes),
    }
