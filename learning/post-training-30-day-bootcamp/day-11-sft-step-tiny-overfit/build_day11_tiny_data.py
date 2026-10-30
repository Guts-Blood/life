#!/usr/bin/env python3
"""Freeze the reviewed Day 11 tiny-overfit dataset and its token audit manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
DAY08_CONTRACT_PATH = BOOTCAMP_ROOT / "day-08-sft-data-contract/inspect_sft_sample.py"
DAY08_CASES_PATH = BOOTCAMP_ROOT / "artifacts/data/day08-template-golden-cases.jsonl"
DAY09_MANIFEST_PATH = BOOTCAMP_ROOT / "artifacts/data/day09-dataset-manifest.json"
DAY09_ASSIGNMENT_PATH = BOOTCAMP_ROOT / "day-09-data-quality-mixture-lineage/day09_assignment.json"
DEFAULT_DATA_PATH = BOOTCAMP_ROOT / "artifacts/data/day11-tiny-overfit.jsonl"
DEFAULT_MANIFEST_PATH = BOOTCAMP_ROOT / "artifacts/data/day11-tiny-overfit-manifest.json"

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "ddc928429ed09d9ad603fd762053d0434c15e865"
MAX_LENGTH = 1024

# These records were individually inspected on 2026-08-05. The two long math
# responses are intentional: the set should expose incorrect equal-per-sample
# loss averaging, not consist only of equally short labels.
DAY09_SELECTED_IDS = (
    "general:personas_IF_2vrdloiscoo2gpir02aj5nrj",
    "general:personas_IF_w2wu9i0s32hxesdafbk3kddp",
    "code:personas_code_8ggy5ckja3p5ery5tv9c9qwl",
    "code:personas_code_3e9973s4899ziqz37veadliq",
    "finance:ADBE/2009/page_81.pdf-2",
    "finance:ADBE/2018/page_66.pdf-4",
    "math:personas_math_c4tl5ey1hrcqijnuyr8t5a2e",
    "math:personas_math_am1x9qgi41gpfq9afvoq9f16",
)


class Day11DataError(ValueError):
    """A frozen-input or generated-manifest invariant failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day11DataError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise Day11DataError(f"required JSON is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise Day11DataError(f"invalid JSON in {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError as error:
        raise Day11DataError(f"required JSONL is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise Day11DataError(f"invalid JSONL in {path}: {error}") from error


def load_day08_contract() -> Any:
    spec = importlib.util.spec_from_file_location("day08_sft_contract", DAY08_CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise Day11DataError(f"cannot import Day 08 contract: {DAY08_CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    day08_cases = load_jsonl(DAY08_CASES_PATH)
    day08_accept = [case for case in day08_cases if case.get("expected_stage") == "accept"]
    if len(day08_accept) != 10:
        raise Day11DataError(f"expected 10 accepted Day 08 golden cases, got {len(day08_accept)}")

    day09_manifest = load_json(DAY09_MANIFEST_PATH)
    day09_assignment = load_json(DAY09_ASSIGNMENT_PATH)
    header = day09_manifest.get("header", {})
    records = day09_manifest.get("records")
    if not isinstance(records, list):
        raise Day11DataError("Day 09 manifest records are missing")
    if day09_assignment.get("status") != "day09_complete_with_gate_c_waiver":
        raise Day11DataError(f"unexpected Day 09 status: {day09_assignment.get('status')!r}")
    gate_c = day09_assignment.get("gate_c", {})
    if gate_c.get("status") != "waived_by_user":
        raise Day11DataError(f"unexpected Day 09 Gate C status: {gate_c.get('status')!r}")
    day09_by_id = {record.get("sample_id"): record for record in records}
    missing = [sample_id for sample_id in DAY09_SELECTED_IDS if sample_id not in day09_by_id]
    if missing:
        raise Day11DataError(f"selected Day 09 IDs are missing: {missing}")

    selected: list[dict[str, Any]] = []
    for case in day08_accept:
        selected.append(
            {
                "id": f"day08:{case['id']}",
                "messages": copy.deepcopy(case["messages"]),
                "metadata": {
                    "origin": "day08_template_golden_case",
                    "upstream_id": case["id"],
                    "upstream_metadata": copy.deepcopy(case.get("metadata", {})),
                    "content_review": "handwritten_golden_case",
                },
            }
        )

    for sample_id in DAY09_SELECTED_IDS:
        record = day09_by_id[sample_id]
        if record.get("truncated"):
            raise Day11DataError(f"selected record is truncated: {sample_id}")
        selected.append(
            {
                "id": sample_id,
                "messages": copy.deepcopy(record["messages"]),
                "metadata": {
                    "origin": "day09_clean_pool_reviewed_subset",
                    "upstream_sample_id": sample_id,
                    "source": record.get("source"),
                    "revision": record.get("revision"),
                    "split": record.get("split"),
                    "license": record.get("license"),
                    "skill": record.get("skill"),
                    "subskill": record.get("subskill"),
                    "upstream_content_hash": record.get("content_hash"),
                    "content_review": "passed_by_codex_2026-08-05",
                },
            }
        )

    ids = [record["id"] for record in selected]
    if len(ids) != len(set(ids)):
        raise Day11DataError("selected IDs are not unique")
    lineage = {
        "day08_file_sha256": file_sha256(DAY08_CASES_PATH),
        "day09_file_sha256": file_sha256(DAY09_MANIFEST_PATH),
        "day09_assignment_file_sha256": file_sha256(DAY09_ASSIGNMENT_PATH),
        "day09_manifest_hash": header.get("manifest_hash"),
        "day09_clean_pool_hash": header.get("clean_pool_hash"),
        "day09_gate_c_status": gate_c.get("status"),
        "day09_claim_boundary": gate_c.get("claim_boundary"),
    }
    return selected, lineage


def build_outputs(tokenizer: Any, model_path: Path) -> tuple[bytes, dict[str, Any]]:
    contract = load_day08_contract()
    selected, lineage = select_records()
    audit_records: list[dict[str, Any]] = []
    for record in selected:
        audit = contract.encode_sample(tokenizer, record, MAX_LENGTH)
        audit_records.append(
            {
                "id": record["id"],
                "origin": record["metadata"]["origin"],
                "skill": record["metadata"].get("skill", "golden_contract"),
                "encoded_tokens": audit["encoded_length"],
                "supervised_tokens": audit["effective_label_tokens"],
                "truncated": audit["truncated"],
                "record_hash": semantic_hash(
                    {"id": record["id"], "messages": record["messages"], "metadata": record["metadata"]}
                ),
            }
        )
    if any(record["truncated"] for record in audit_records):
        raise Day11DataError("tiny-set construction unexpectedly truncated a selected record")

    data_bytes = (
        "".join(canonical_json(record) + "\n" for record in selected).encode("utf-8")
    )
    template = tokenizer.chat_template or ""
    required_model_files = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    manifest: dict[str, Any] = {
        "header": {
            "domain": "day11.tiny_overfit_manifest",
            "schema_version": 1,
            "status": "frozen_for_day11_execution",
            "created_date": "2026-08-05",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_files": {
                relative_path: file_sha256(model_path / relative_path)
                for relative_path in required_model_files
            },
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "max_length": MAX_LENGTH,
            "loss_mask": "all assistant bodies plus each assistant <|im_end|>; shifted causal targets",
            "selection_policy": {
                "day08": "all 10 expected_stage=accept handwritten golden cases",
                "day09": "8 explicit IDs reviewed for answer correctness; 2 per general/code/finance/math",
                "reviewer": "Codex",
                "review_date": "2026-08-05",
            },
            "claim_boundary": (
                "A pipeline-correctness memorization set. It does not establish generalization, "
                "dataset quality, or recipe quality. Day 09 Gate C was waived, not passed."
            ),
            "lineage": lineage,
            "record_count": len(selected),
            "total_encoded_tokens": sum(record["encoded_tokens"] for record in audit_records),
            "total_supervised_tokens": sum(record["supervised_tokens"] for record in audit_records),
            "dataset_file_sha256": hashlib.sha256(data_bytes).hexdigest(),
        },
        "records": audit_records,
    }
    manifest["header"]["manifest_hash"] = semantic_hash(manifest)
    return data_bytes, manifest


def write_or_check(
    *, data_bytes: bytes, manifest: dict[str, Any], data_path: Path, manifest_path: Path, check: bool
) -> None:
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if check:
        if data_path.read_bytes() != data_bytes:
            raise Day11DataError(f"frozen dataset differs from deterministic rebuild: {data_path}")
        if manifest_path.read_bytes() != manifest_bytes:
            raise Day11DataError(f"frozen manifest differs from deterministic rebuild: {manifest_path}")
        print(f"PASS: {data_path}")
        print(f"PASS: {manifest_path}")
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(data_bytes)
    manifest_path.write_bytes(manifest_bytes)
    print(f"wrote {data_path}")
    print(f"wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    data_bytes, manifest = build_outputs(tokenizer, args.model_path)
    write_or_check(
        data_bytes=data_bytes,
        manifest=manifest,
        data_path=args.data_path,
        manifest_path=args.manifest_path,
        check=args.check,
    )


if __name__ == "__main__":
    main()
