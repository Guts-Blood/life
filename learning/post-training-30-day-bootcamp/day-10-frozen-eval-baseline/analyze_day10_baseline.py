#!/usr/bin/env python3
"""Validate and summarize a complete Day 10 development-set baseline run.

The analyzer is intentionally stricter than a generic metrics script: it only
accepts the complete frozen 112-record development run (28 records per slice),
reconstructs every run-level semantic hash, and refuses frozen-test records.
HumanEval remains unscored unless a complete, separately pinned sandbox sidecar
is supplied and independently verified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import rfc8785


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-frozen-eval-manifest.json"
)
DEFAULT_CODE_SANDBOX_CONFIG = HERE / "day10_e2b_sandbox_config.json"
SLICES = ("general", "math", "code", "finance")
SCORED_SLICES = ("general", "math", "finance")
EXPECTED_DEV_RECORDS = 112
EXPECTED_RECORDS_PER_SLICE = 28
WILSON_Z_95 = 1.959963984540054
CODE_RESULT_FIELDS = frozenset(
    {
        "code_execution_protocol",
        "code_execution_protocol_hash",
        "code_ordinal",
        "code_run_hash",
        "comparison_key",
        "complete_comparison_key",
        "completion_hash",
        "composed_program_hash",
        "domain",
        "entry_point",
        "error_type",
        "evaluation_split",
        "evaluator_source_sha256",
        "execution_status",
        "exit_code",
        "extractor_version",
        "failure_message",
        "harness_version",
        "latency_seconds",
        "manifest_file_sha256",
        "manifest_hash",
        "model_id",
        "model_revision",
        "model_snapshot_hash",
        "observed",
        "output_token_ids_hash",
        "parsed_completion_hash",
        "passed",
        "prediction_comparison_key",
        "prediction_run_hash",
        "prediction_run_ordinal",
        "predictions_file_sha256",
        "raw_output_hash",
        "raw_prompt_hash",
        "reference_hash",
        "run_hash",
        "sample_id",
        "sandbox",
        "sandbox_contract_file_sha256",
        "sandbox_contract_hash",
        "schema_version",
        "score",
        "score_status",
        "scorer_result",
        "scorer_version",
        "slice",
        "source_file_sha256",
        "source_revision",
        "stderr",
        "stdout",
        "task_id",
        "test_sha256",
    }
)


class Day10AnalysisError(ValueError):
    """A frozen manifest, prediction, or output invariant failed."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Day10AnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Day10AnalysisError(f"non-finite JSON value in {context}: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise Day10AnalysisError(f"invalid JSON in {context}: {error}") from error


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.resolve().read_bytes()
    except FileNotFoundError as error:
        raise Day10AnalysisError(f"{label} is missing: {path}") from error


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_bytes(path, "manifest")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Day10AnalysisError(f"manifest is not UTF-8: {path}") from error
    manifest = _parse_json(text, str(path))
    if not isinstance(manifest, dict):
        raise Day10AnalysisError("manifest root must be an object")
    return manifest, hashlib.sha256(payload).hexdigest()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    payload = _read_bytes(path, "predictions JSONL")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Day10AnalysisError(f"predictions JSONL is not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise Day10AnalysisError(f"blank JSONL row at {path}:{line_number}")
        row = _parse_json(line, f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise Day10AnalysisError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise Day10AnalysisError("predictions JSONL is empty")
    return rows


def load_code_results(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = _read_bytes(path, "code sandbox results JSONL")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Day10AnalysisError(
            f"code sandbox results JSONL is not UTF-8: {path}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise Day10AnalysisError(f"blank JSONL row at {path}:{line_number}")
        row = _parse_json(line, f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise Day10AnalysisError(
                f"code result row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise Day10AnalysisError("code sandbox results JSONL is empty")
    return rows, hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError) as error:
        raise Day10AnalysisError(f"RFC 8785 canonicalization failed: {error}") from error


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_text_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any, *, prefixed: bool) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value[7:] if prefixed and value.startswith("sha256:") else value
    if prefixed and not value.startswith("sha256:"):
        return False
    return len(candidate) == 64 and all(
        character in "0123456789abcdef" for character in candidate
    )


def _require_sha256(value: Any, context: str, *, prefixed: bool) -> str:
    if not _is_sha256(value, prefixed=prefixed):
        label = "sha256:-prefixed SHA-256" if prefixed else "SHA-256"
        raise Day10AnalysisError(f"{context} must be a lowercase {label}")
    return value


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise Day10AnalysisError(f"{context}.{key} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Day10AnalysisError(f"{context} must be a non-negative integer")
    return value


def _require_positive_number(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise Day10AnalysisError(f"{context} must be a finite positive number")
    return float(value)


def _verify_manifest_record(
    record: dict[str, Any], maximum_input_tokens: int
) -> None:
    sample_id = _require_string(record, "sample_id", "manifest record")
    if record.get("evaluation_split") not in {"dev", "frozen_test"}:
        raise Day10AnalysisError(f"invalid manifest split: {sample_id}")
    if record.get("slice") not in SLICES:
        raise Day10AnalysisError(f"invalid manifest slice: {sample_id}")

    rendered_prompt = record.get("rendered_prompt")
    reference = record.get("reference")
    input_ids = record.get("input_ids")
    if not isinstance(rendered_prompt, str) or not isinstance(reference, str):
        raise Day10AnalysisError(f"manifest text field is invalid: {sample_id}")
    if record.get("rendered_prompt_hash") != exact_text_hash(rendered_prompt):
        raise Day10AnalysisError(f"manifest rendered prompt hash mismatch: {sample_id}")
    if record.get("reference_hash") != exact_text_hash(reference):
        raise Day10AnalysisError(f"manifest reference hash mismatch: {sample_id}")
    if not isinstance(input_ids, list) or not input_ids or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in input_ids
    ):
        raise Day10AnalysisError(f"manifest input IDs are invalid: {sample_id}")
    expected_input_hash = semantic_hash(
        {
            "domain": "day10.input_ids",
            "schema_version": 1,
            "input_ids": input_ids,
        }
    )
    if record.get("input_ids_hash") != expected_input_hash:
        raise Day10AnalysisError(f"manifest input ID hash mismatch: {sample_id}")
    if record.get("input_token_count") != len(input_ids):
        raise Day10AnalysisError(f"manifest input token count mismatch: {sample_id}")
    if len(input_ids) > maximum_input_tokens:
        raise Day10AnalysisError(f"manifest input exceeds token limit: {sample_id}")


def verify_manifest(
    manifest: dict[str, Any], manifest_file_sha256: str
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    header = manifest.get("header")
    records = manifest.get("records")
    if not isinstance(header, dict) or not isinstance(records, list):
        raise Day10AnalysisError("manifest must contain an object header and records array")
    if header.get("domain") != "day10.frozen_eval_manifest":
        raise Day10AnalysisError("unexpected manifest domain")
    if header.get("schema_version") != 1:
        raise Day10AnalysisError("unsupported manifest schema version")
    if header.get("status") != "frozen_manifest_pre_baseline":
        raise Day10AnalysisError("manifest is not frozen for baseline evaluation")

    expected_manifest_hash = _require_string(header, "manifest_hash", "header")
    candidate = copy.deepcopy(manifest)
    candidate["header"].pop("manifest_hash", None)
    if semantic_hash(candidate) != expected_manifest_hash:
        raise Day10AnalysisError("manifest semantic hash mismatch")
    for field in (
        "config_file_sha256",
        "dataset_context_hash",
        "eval_suite_hash",
        "protocol_hash",
        "scorer_registry_hash",
        "environment_contract_sha256",
        "environment_snapshot_sha256",
    ):
        _require_string(header, field, "header")

    rendering = header.get("model_and_rendering")
    generation = header.get("generation")
    orders = header.get("evaluation_order")
    if not isinstance(rendering, dict) or not isinstance(generation, dict):
        raise Day10AnalysisError("manifest rendering/generation contract is missing")
    if not isinstance(orders, dict):
        raise Day10AnalysisError("manifest evaluation order is missing")
    maximum_input_tokens = rendering.get("maximum_input_tokens")
    if (
        isinstance(maximum_input_tokens, bool)
        or not isinstance(maximum_input_tokens, int)
        or maximum_input_tokens <= 0
    ):
        raise Day10AnalysisError("manifest maximum_input_tokens is invalid")
    if generation.get("do_sample") is not False or generation.get("num_beams") != 1:
        raise Day10AnalysisError("manifest generation is not frozen to greedy decoding")
    if any(generation.get(field) is not None for field in ("temperature", "top_p", "top_k")):
        raise Day10AnalysisError("manifest greedy generation has sampling parameters")
    eos_token_id = generation.get("eos_token_id")
    pad_token_id = generation.get("pad_token_id")
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or generation.get("stop_token_ids") != [eos_token_id]
    ):
        raise Day10AnalysisError("manifest stop-token contract is invalid")
    repetition_penalty = generation.get("repetition_penalty")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(repetition_penalty)
        or repetition_penalty <= 0
    ):
        raise Day10AnalysisError("manifest repetition penalty is invalid")
    max_new_tokens_by_slice = generation.get("max_new_tokens_by_slice")
    if not isinstance(max_new_tokens_by_slice, dict) or set(max_new_tokens_by_slice) != set(
        SLICES
    ) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in max_new_tokens_by_slice.values()
    ):
        raise Day10AnalysisError("manifest per-slice generation limits are invalid")

    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise Day10AnalysisError("manifest record is not an object")
        _verify_manifest_record(record, maximum_input_tokens)
        sample_id = record["sample_id"]
        if sample_id in by_id:
            raise Day10AnalysisError(f"duplicate manifest sample ID: {sample_id}")
        for field in ("extractor_version", "scorer_version"):
            _require_string(record, field, f"manifest record {sample_id}")
        if record.get("generation_max_new_tokens") != max_new_tokens_by_slice[
            record["slice"]
        ]:
            raise Day10AnalysisError(
                f"manifest generation limit differs from slice contract: {sample_id}"
            )
        by_id[sample_id] = record

    dev_order = orders.get("dev")
    frozen_order = orders.get("frozen_test")
    if not isinstance(dev_order, list) or not all(isinstance(item, str) for item in dev_order):
        raise Day10AnalysisError("manifest dev evaluation order is invalid")
    if not isinstance(frozen_order, list) or not all(
        isinstance(item, str) for item in frozen_order
    ):
        raise Day10AnalysisError("manifest frozen_test evaluation order is invalid")
    if len(dev_order) != len(set(dev_order)) or len(frozen_order) != len(set(frozen_order)):
        raise Day10AnalysisError("manifest evaluation order contains duplicate IDs")
    if set(dev_order) & set(frozen_order):
        raise Day10AnalysisError("manifest dev/frozen_test orders overlap")
    for split, order in (("dev", dev_order), ("frozen_test", frozen_order)):
        actual_ids = {
            sample_id
            for sample_id, record in by_id.items()
            if record["evaluation_split"] == split
        }
        if set(order) != actual_ids:
            raise Day10AnalysisError(f"manifest {split} order does not cover its records")

    if len(dev_order) != EXPECTED_DEV_RECORDS:
        raise Day10AnalysisError(
            f"expected {EXPECTED_DEV_RECORDS} dev records, found {len(dev_order)}"
        )
    slice_counts = {slice_name: 0 for slice_name in SLICES}
    for sample_id in dev_order:
        slice_counts[by_id[sample_id]["slice"]] += 1
    if any(count != EXPECTED_RECORDS_PER_SLICE for count in slice_counts.values()):
        raise Day10AnalysisError(
            f"dev slice counts must each be {EXPECTED_RECORDS_PER_SLICE}: {slice_counts}"
        )

    counts = header.get("counts")
    if not isinstance(counts, dict):
        raise Day10AnalysisError("manifest header counts are missing")
    by_split = counts.get("by_split")
    by_slice_and_split = counts.get("by_slice_and_split")
    if not isinstance(by_split, dict) or by_split.get("dev") != EXPECTED_DEV_RECORDS:
        raise Day10AnalysisError("manifest header dev count is inconsistent")
    if not isinstance(by_slice_and_split, dict) or any(
        not isinstance(by_slice_and_split.get(slice_name), dict)
        or by_slice_and_split[slice_name].get("dev") != EXPECTED_RECORDS_PER_SLICE
        for slice_name in SLICES
    ):
        raise Day10AnalysisError("manifest header slice counts are inconsistent")

    return by_id, dev_order, {
        "manifest_hash": expected_manifest_hash,
        "manifest_file_sha256": manifest_file_sha256,
        "maximum_input_tokens": maximum_input_tokens,
        "header": header,
    }


def _verify_model_identity(
    record: dict[str, Any], manifest_header: dict[str, Any]
) -> None:
    for field in (
        "model_id",
        "model_revision",
        "model_snapshot_path",
        "model_snapshot_source",
        "model_snapshot_hash",
    ):
        _require_string(record, field, "prediction")
    if record["model_snapshot_source"] not in {
        "manifest",
        "cli_override",
        "cli_required",
    }:
        raise Day10AnalysisError("prediction model snapshot source is invalid")
    model_files = record.get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise Day10AnalysisError("prediction model_files must be a non-empty object")
    for name, spec in model_files.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise Day10AnalysisError("prediction model_files entry is invalid")
        sha256 = spec.get("sha256")
        byte_count = spec.get("bytes")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise Day10AnalysisError(f"invalid model file hash: {name}")
        _require_nonnegative_int(byte_count, f"model_files[{name!r}].bytes")
    expected_snapshot_hash = semantic_hash(
        {
            "domain": "day10.model_snapshot",
            "schema_version": 1,
            "files": model_files,
        }
    )
    if record.get("model_snapshot_hash") != expected_snapshot_hash:
        raise Day10AnalysisError("model snapshot hash mismatch")

    if record.get("model_snapshot_source") == "manifest":
        rendering = manifest_header["model_and_rendering"]
        if record.get("model_id") != rendering.get("model_id") or record.get(
            "model_revision"
        ) != rendering.get("model_revision"):
            raise Day10AnalysisError("manifest model identity was relabeled")
        required_files = rendering.get("model_files")
        if not isinstance(required_files, dict) or any(
            name not in model_files or model_files[name].get("sha256") != sha256
            for name, sha256 in required_files.items()
        ):
            raise Day10AnalysisError("runtime model files differ from the manifest")


def _expected_generation(
    header_generation: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": header_generation.get("eos_token_id"),
        "pad_token_id": header_generation.get("pad_token_id"),
        "repetition_penalty": header_generation.get("repetition_penalty"),
    }


def _verify_prediction_payload(
    prediction: dict[str, Any], manifest_record: dict[str, Any], ordinal: int,
) -> None:
    sample_id = manifest_record["sample_id"]
    if prediction.get("schema_version") != 1:
        raise Day10AnalysisError(f"unsupported prediction schema: {sample_id}")
    if prediction.get("run_ordinal") != ordinal:
        raise Day10AnalysisError(f"run ordinal mismatch: {sample_id}")
    if prediction.get("evaluation_split") != "dev":
        raise Day10AnalysisError(f"non-dev prediction is prohibited: {sample_id}")
    if prediction.get("slice") != manifest_record["slice"]:
        raise Day10AnalysisError(f"prediction slice mismatch: {sample_id}")
    for field in (
        "rendered_prompt_hash",
        "input_ids_hash",
        "reference_hash",
        "input_token_count",
        "extractor_version",
        "scorer_version",
    ):
        if prediction.get(field) != manifest_record.get(field):
            raise Day10AnalysisError(f"prediction {field} differs from manifest: {sample_id}")
    _require_nonnegative_int(
        prediction.get("input_token_count"), f"input_token_count for {sample_id}"
    )

    raw_output = prediction.get("raw_output")
    output_token_ids = prediction.get("output_token_ids")
    if not isinstance(raw_output, str):
        raise Day10AnalysisError(f"raw output is not text: {sample_id}")
    if prediction.get("raw_output_hash") != exact_text_hash(raw_output):
        raise Day10AnalysisError(f"raw output hash mismatch: {sample_id}")
    if not isinstance(output_token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in output_token_ids
    ):
        raise Day10AnalysisError(f"output token IDs are invalid: {sample_id}")
    expected_output_hash = semantic_hash(
        {
            "domain": "day10.output_token_ids",
            "schema_version": 1,
            "output_token_ids": output_token_ids,
        }
    )
    if prediction.get("output_token_ids_hash") != expected_output_hash:
        raise Day10AnalysisError(f"output token ID hash mismatch: {sample_id}")
    output_count = _require_nonnegative_int(
        prediction.get("output_token_count"), f"output_token_count for {sample_id}"
    )
    if output_count != len(output_token_ids):
        raise Day10AnalysisError(f"output token count mismatch: {sample_id}")
    limit = manifest_record.get("generation_max_new_tokens")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise Day10AnalysisError(f"manifest generation limit is invalid: {sample_id}")
    if output_count > limit:
        raise Day10AnalysisError(f"output exceeds generation token limit: {sample_id}")
    total_count = _require_nonnegative_int(
        prediction.get("total_token_count"), f"total_token_count for {sample_id}"
    )
    if total_count != manifest_record["input_token_count"] + output_count:
        raise Day10AnalysisError(f"total token count mismatch: {sample_id}")
    _require_positive_number(prediction.get("latency_seconds"), f"latency for {sample_id}")


def _verify_execution_protocol(
    execution_protocol: dict[str, Any], runtime: dict[str, Any], header: dict[str, Any]
) -> None:
    if execution_protocol.get("domain") != "day10.execution_protocol" or execution_protocol.get(
        "schema_version"
    ) != 1:
        raise Day10AnalysisError("execution protocol identity is invalid")
    if execution_protocol.get("local_files_only") is not True or execution_protocol.get(
        "trust_remote_code"
    ) is not False:
        raise Day10AnalysisError("execution protocol is not local/fail-closed")
    for field in ("environment_contract_sha256", "environment_snapshot_sha256"):
        if execution_protocol.get(field) != header.get(field):
            raise Day10AnalysisError(f"execution protocol {field} differs from manifest")
    runtime_fields = (
        "device",
        "model_dtype",
        "torch_version",
        "transformers_version",
        "python_version",
        "python_implementation",
        "platform",
        "torch_num_threads",
        "torch_num_interop_threads",
    )
    if any(execution_protocol.get(field) != runtime.get(field) for field in runtime_fields):
        raise Day10AnalysisError("execution protocol and runtime differ")


def _score_classification(slice_name: str, scorer_result: Any) -> tuple[int, int, int]:
    if not isinstance(scorer_result, dict):
        raise Day10AnalysisError("scorer_result must be an object")
    score = scorer_result.get("score")
    parse_status = scorer_result.get("parse_status")
    score_status = scorer_result.get("score_status")
    error_type = scorer_result.get("error_type")
    if slice_name == "code":
        if not (
            parse_status == "ok"
            and score_status == "sandbox_required"
            and score is None
            and error_type == "sandbox_required"
        ):
            raise Day10AnalysisError("code result escaped sandbox-pending state")
        return 0, 0, 1

    if score_status != "ok" or parse_status not in {"ok", "parse_error"}:
        raise Day10AnalysisError(f"invalid scored result state for {slice_name}")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in {0, 1}:
        raise Day10AnalysisError(f"invalid binary score for {slice_name}")
    if parse_status == "parse_error":
        if score != 0 or error_type != "parse_error":
            raise Day10AnalysisError(f"inconsistent parse error for {slice_name}")
        return 1, 0, 0
    if score == 1:
        if error_type is not None:
            raise Day10AnalysisError(f"correct result has an error type for {slice_name}")
        return 0, 0, 0
    if error_type != "wrong_answer":
        raise Day10AnalysisError(f"incorrect result is not wrong_answer for {slice_name}")
    return 0, 1, 0


def wilson_interval(correct: int, total: int) -> dict[str, float] | None:
    if total == 0:
        return None
    probability = correct / total
    z_squared = WILSON_Z_95 * WILSON_Z_95
    denominator = 1 + z_squared / total
    center = (probability + z_squared / (2 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(
            (probability * (1 - probability) + z_squared / (4 * total)) / total
        )
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 12),
        "upper": round(min(1.0, center + margin), 12),
    }


def _uniform_value(predictions: list[dict[str, Any]], field: str) -> Any:
    value = predictions[0].get(field)
    if any(prediction.get(field) != value for prediction in predictions[1:]):
        raise Day10AnalysisError(f"prediction field is not constant across run: {field}")
    return value


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise Day10AnalysisError(f"{context} does not match the frozen schema")
    return value


def _verify_code_execution_protocol(protocol: Any) -> dict[str, Any]:
    protocol = _require_exact_keys(
        protocol,
        {
            "backend",
            "composition",
            "domain",
            "evaluator_source_sha256",
            "execution",
            "expected_dev_code_records",
            "isolation",
            "lifecycle",
            "network",
            "resource_limits",
            "schema_version",
            "sdk",
            "source",
            "template_id",
            "template_runtime",
        },
        "code execution protocol",
    )
    if (
        protocol.get("domain") != "day10.humaneval_e2b_sandbox_contract"
        or protocol.get("schema_version") != 1
        or protocol.get("backend") != "e2b_firecracker"
        or protocol.get("expected_dev_code_records") != EXPECTED_RECORDS_PER_SLICE
        or protocol.get("composition")
        != (
            "source_prompt + parsed_completion + newline + source_test + newline + "
            "check(entry_point)"
        )
    ):
        raise Day10AnalysisError("code execution protocol identity changed")
    _require_sha256(
        protocol.get("evaluator_source_sha256"),
        "code execution protocol evaluator source",
        prefixed=False,
    )
    _require_string(protocol, "template_id", "code execution protocol")

    sdk = _require_exact_keys(protocol.get("sdk"), {"package", "version"}, "sandbox SDK")
    if sdk != {"package": "e2b", "version": "2.37.0"}:
        raise Day10AnalysisError("sandbox SDK contract changed")
    network = _require_exact_keys(
        protocol.get("network"),
        {"allow_internet_access", "allow_public_traffic", "deny_out", "secure"},
        "sandbox network",
    )
    if network != {
        "allow_internet_access": False,
        "allow_public_traffic": False,
        "deny_out": ["0.0.0.0/0"],
        "secure": True,
    }:
        raise Day10AnalysisError("sandbox network isolation changed")
    isolation = _require_exact_keys(
        protocol.get("isolation"),
        {"envs", "fresh_sandbox_per_sample", "mounts", "mcp_servers"},
        "sandbox isolation",
    )
    if isolation != {
        "envs": {},
        "fresh_sandbox_per_sample": True,
        "mounts": [],
        "mcp_servers": [],
    }:
        raise Day10AnalysisError("sandbox per-sample isolation changed")
    lifecycle = _require_exact_keys(
        protocol.get("lifecycle"),
        {"auto_resume", "on_timeout", "sandbox_timeout_seconds"},
        "sandbox lifecycle",
    )
    if lifecycle.get("auto_resume") is not False or lifecycle.get("on_timeout") != "kill":
        raise Day10AnalysisError("sandbox lifecycle changed")
    _require_nonnegative_int(
        lifecycle.get("sandbox_timeout_seconds"), "sandbox timeout seconds"
    )
    if lifecycle["sandbox_timeout_seconds"] == 0:
        raise Day10AnalysisError("sandbox timeout seconds must be positive")

    execution = _require_exact_keys(
        protocol.get("execution"),
        {
            "candidate_timeout_seconds",
            "kill_after_seconds",
            "max_captured_output_bytes",
            "max_result_bytes",
            "python_command",
            "result_path",
            "runner_path",
            "sdk_outer_timeout_seconds",
            "timeout_semantics",
        },
        "sandbox execution",
    )
    if (
        execution.get("python_command") != "python3"
        or execution.get("runner_path") != "/home/user/day10_humaneval_runner.py"
        or execution.get("result_path") != "/home/user/day10_humaneval_result.json"
        or execution.get("timeout_semantics")
        != {
            "candidate_gnu_timeout_exit_124": "valid_score_zero",
            "sdk_timeout_exception": "infrastructure_error_null",
        }
    ):
        raise Day10AnalysisError("sandbox command/timeout contract changed")
    for field in (
        "candidate_timeout_seconds",
        "kill_after_seconds",
        "max_captured_output_bytes",
        "max_result_bytes",
        "sdk_outer_timeout_seconds",
    ):
        _require_nonnegative_int(execution.get(field), f"sandbox execution {field}")
        if execution[field] == 0:
            raise Day10AnalysisError(f"sandbox execution {field} must be positive")
    if execution["sdk_outer_timeout_seconds"] <= execution["candidate_timeout_seconds"]:
        raise Day10AnalysisError("sandbox outer timeout must exceed candidate timeout")

    limits = _require_exact_keys(
        protocol.get("resource_limits"),
        {
            "address_space_bytes",
            "core_bytes",
            "cpu_seconds",
            "file_size_bytes",
            "open_files",
            "processes",
        },
        "sandbox resource limits",
    )
    for field in (
        "address_space_bytes",
        "cpu_seconds",
        "file_size_bytes",
        "open_files",
        "processes",
    ):
        _require_nonnegative_int(limits.get(field), f"sandbox resource limit {field}")
        if limits[field] == 0:
            raise Day10AnalysisError(f"sandbox resource limit {field} must be positive")
    if limits.get("core_bytes") != 0:
        raise Day10AnalysisError("sandbox core dump limit changed")

    source = _require_exact_keys(
        protocol.get("source"),
        {"file", "file_sha256", "revision", "source", "split"},
        "HumanEval source",
    )
    if source.get("source") != "openai/human-eval" or source.get("split") != "test":
        raise Day10AnalysisError("HumanEval source identity changed")
    source_path = Path(_require_string(source, "file", "HumanEval source"))
    if source_path.is_absolute() or ".." in source_path.parts:
        raise Day10AnalysisError("HumanEval source path is not repository-relative")
    _require_string(source, "revision", "HumanEval source")
    _require_sha256(source.get("file_sha256"), "HumanEval source file", prefixed=False)

    runtime = _require_exact_keys(
        protocol.get("template_runtime"),
        {"envd_version", "memory_mib", "python_version", "vcpus"},
        "sandbox template runtime",
    )
    _require_string(runtime, "envd_version", "sandbox template runtime")
    _require_string(runtime, "python_version", "sandbox template runtime")
    for field in ("memory_mib", "vcpus"):
        _require_nonnegative_int(runtime.get(field), f"sandbox template {field}")
        if runtime[field] == 0:
            raise Day10AnalysisError(f"sandbox template {field} must be positive")
    return protocol


def _verify_bounded_evidence(value: Any, context: str) -> dict[str, Any]:
    evidence = _require_exact_keys(value, {"bytes", "excerpt", "sha256"}, context)
    byte_count = _require_nonnegative_int(evidence.get("bytes"), f"{context}.bytes")
    excerpt = evidence.get("excerpt")
    if not isinstance(excerpt, str):
        raise Day10AnalysisError(f"{context}.excerpt must be text")
    excerpt_bytes = excerpt.encode("utf-8")
    if len(excerpt_bytes) > 1024 or len(excerpt_bytes) > byte_count:
        raise Day10AnalysisError(f"{context}.excerpt exceeds its frozen bound")
    digest = _require_sha256(evidence.get("sha256"), f"{context}.sha256", prefixed=False)
    if byte_count <= 1024 and (
        len(excerpt_bytes) != byte_count
        or hashlib.sha256(excerpt_bytes).hexdigest() != digest
    ):
        raise Day10AnalysisError(f"{context} hash/length mismatch")
    return evidence


def _verify_code_result_state(row: dict[str, Any], sample_id: str) -> None:
    if row.get("score_status") != "ok":
        raise Day10AnalysisError(f"code result is not a valid score: {sample_id}")
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in {0, 1}:
        raise Day10AnalysisError(f"code result is not binary: {sample_id}")
    passed = row.get("passed")
    if not isinstance(passed, bool) or passed != (score == 1):
        raise Day10AnalysisError(f"code result passed/score mismatch: {sample_id}")
    execution_status = row.get("execution_status")
    error_type = row.get("error_type")
    exit_code = row.get("exit_code")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise Day10AnalysisError(f"invalid code result exit code: {sample_id}")
    if execution_status == "passed":
        valid_state = score == 1 and error_type is None and exit_code == 0
    elif execution_status == "timeout":
        valid_state = score == 0 and (exit_code, error_type) in {
            (124, "wall_timeout"),
            (152, "cpu_timeout"),
        }
    elif execution_status == "failed":
        if exit_code == 137:
            valid_state = score == 0 and error_type == "resource_limit"
        elif exit_code == 153:
            valid_state = score == 0 and error_type == "file_size_limit"
        else:
            valid_state = score == 0 and exit_code == 0 and error_type in {
                "assertion_error",
                "memory_limit",
                "runtime_error",
                "syntax_error",
            }
    else:
        valid_state = False
    if not valid_state:
        raise Day10AnalysisError(f"invalid code execution state: {sample_id}")

    failure_message = row.get("failure_message")
    if failure_message is not None and (
        not isinstance(failure_message, str)
        or len(failure_message.encode("utf-8")) > 4096
    ):
        raise Day10AnalysisError(f"invalid code failure message: {sample_id}")
    stdout = _verify_bounded_evidence(row.get("stdout"), f"stdout for {sample_id}")
    stderr = _verify_bounded_evidence(row.get("stderr"), f"stderr for {sample_id}")
    scorer_result = _require_exact_keys(
        row.get("scorer_result"),
        {
            "error_type",
            "execution_outcome",
            "executor_result_sha256",
            "parse_status",
            "passed",
            "score",
            "score_status",
        },
        f"scorer result for {sample_id}",
    )
    if (
        scorer_result.get("parse_status") != "ok"
        or scorer_result.get("score_status") != row["score_status"]
        or scorer_result.get("score") != score
        or scorer_result.get("passed") != passed
        or scorer_result.get("execution_outcome") != execution_status
        or scorer_result.get("error_type") != error_type
    ):
        raise Day10AnalysisError(f"nested code scorer result mismatch: {sample_id}")
    expected_executor_hash = semantic_hash(
        {
            "domain": "day10.code_executor_result",
            "schema_version": 1,
            "execution_status": execution_status,
            "score_status": row["score_status"],
            "score": score,
            "error_type": error_type,
            "exit_code": exit_code,
            "stdout_sha256": stdout["sha256"],
            "stderr_sha256": stderr["sha256"],
            "failure_message": failure_message,
        }
    )
    if scorer_result.get("executor_result_sha256") != expected_executor_hash:
        raise Day10AnalysisError(f"code executor result hash mismatch: {sample_id}")


def _code_results_semantic_hash(
    rows: list[dict[str, Any]], code_run_hash: str, complete_comparison_key: str
) -> str:
    return semantic_hash(
        {
            "domain": "day10.code_sandbox_results",
            "schema_version": 1,
            "code_run_hash": code_run_hash,
            "complete_comparison_key": complete_comparison_key,
            "results": [
                {
                    "code_ordinal": row["code_ordinal"],
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "test_sha256": row["test_sha256"],
                    "raw_output_hash": row["raw_output_hash"],
                    "completion_hash": row["completion_hash"],
                    "composed_program_hash": row["composed_program_hash"],
                    "execution_status": row["execution_status"],
                    "score": row["score"],
                    "error_type": row["error_type"],
                    "executor_result_sha256": row["scorer_result"][
                        "executor_result_sha256"
                    ],
                }
                for row in rows
            ],
        }
    )


def _verify_code_results(
    *,
    code_results_path: Path,
    manifest_by_id: dict[str, dict[str, Any]],
    dev_order: list[str],
    predictions: list[dict[str, Any]],
    predictions_path: Path,
    manifest_hash: str,
    manifest_file_sha256: str,
    run_hash: str,
    comparison_key: str,
    model_id: str,
    model_revision: str,
    model_snapshot_hash: str,
) -> dict[str, Any]:
    rows, results_file_sha256 = load_code_results(code_results_path)
    code_order = [
        sample_id
        for sample_id in dev_order
        if manifest_by_id[sample_id]["slice"] == "code"
    ]
    if len(rows) != EXPECTED_RECORDS_PER_SLICE:
        raise Day10AnalysisError(
            f"expected {EXPECTED_RECORDS_PER_SLICE} code results, found {len(rows)}"
        )
    result_ids = [row.get("sample_id") for row in rows]
    if result_ids != code_order:
        raise Day10AnalysisError(
            "code result sample IDs/order differ from frozen dev code order"
        )
    if len(set(result_ids)) != len(result_ids):
        raise Day10AnalysisError("code result sample IDs are duplicated")

    config_payload = _read_bytes(DEFAULT_CODE_SANDBOX_CONFIG, "code sandbox config")
    try:
        frozen_protocol = _parse_json(
            config_payload.decode("utf-8"), str(DEFAULT_CODE_SANDBOX_CONFIG)
        )
    except UnicodeDecodeError as error:
        raise Day10AnalysisError("code sandbox config is not UTF-8") from error
    frozen_protocol = _verify_code_execution_protocol(frozen_protocol)
    code_execution_protocol_hash = semantic_hash(frozen_protocol)
    sandbox_contract_file_sha256 = hashlib.sha256(config_payload).hexdigest()
    predictions_file_sha256 = hashlib.sha256(
        _read_bytes(predictions_path, "predictions JSONL")
    ).hexdigest()
    complete_comparison_key = semantic_hash(
        {
            "domain": "day10.complete_checkpoint_comparison",
            "schema_version": 1,
            "prediction_comparison_key": comparison_key,
            "code_execution_protocol_hash": code_execution_protocol_hash,
        }
    )
    code_run_hash = semantic_hash(
        {
            "domain": "day10.code_sandbox_run",
            "schema_version": 1,
            "manifest_hash": manifest_hash,
            "manifest_file_sha256": manifest_file_sha256,
            "predictions_file_sha256": predictions_file_sha256,
            "prediction_run_hash": run_hash,
            "code_execution_protocol_hash": code_execution_protocol_hash,
            "sample_ids": code_order,
        }
    )
    predictions_by_id = {row["sample_id"]: row for row in predictions}
    prediction_ordinals = {
        sample_id: ordinal for ordinal, sample_id in enumerate(dev_order)
    }

    for code_ordinal, row in enumerate(rows):
        sample_id = code_order[code_ordinal]
        if set(row) != CODE_RESULT_FIELDS:
            raise Day10AnalysisError(
                f"code result row does not match the frozen schema: {sample_id}"
            )
        record = manifest_by_id[sample_id]
        prediction = predictions_by_id[sample_id]
        lineage = record.get("source_lineage")
        metadata = record.get("metadata")
        if not isinstance(lineage, dict) or not isinstance(metadata, dict):
            raise Day10AnalysisError(f"code manifest metadata is missing: {sample_id}")
        raw_prompt = record.get("raw_prompt")
        if not isinstance(raw_prompt, str) or record.get("raw_prompt_hash") != exact_text_hash(
            raw_prompt
        ):
            raise Day10AnalysisError(f"code manifest raw prompt hash mismatch: {sample_id}")
        completion = prediction.get("scorer_result", {}).get("parsed_answer")
        if not isinstance(completion, str):
            raise Day10AnalysisError(f"code prediction completion is missing: {sample_id}")
        completion_hash = exact_text_hash(completion)
        task_id = _require_string(lineage, "parent_id", sample_id)
        entry_point = _require_string(metadata, "entry_point", sample_id)
        test_sha256 = _require_sha256(
            metadata.get("test_sha256"), f"test hash for {sample_id}", prefixed=False
        )
        expected_values = {
            "domain": "day10.code_sandbox_result",
            "schema_version": 1,
            "sample_id": sample_id,
            "code_ordinal": code_ordinal,
            "prediction_run_ordinal": prediction_ordinals[sample_id],
            "evaluation_split": "dev",
            "slice": "code",
            "task_id": task_id,
            "entry_point": entry_point,
            "test_sha256": test_sha256,
            "manifest_hash": manifest_hash,
            "manifest_file_sha256": manifest_file_sha256,
            "predictions_file_sha256": predictions_file_sha256,
            "run_hash": run_hash,
            "prediction_run_hash": run_hash,
            "comparison_key": comparison_key,
            "prediction_comparison_key": comparison_key,
            "complete_comparison_key": complete_comparison_key,
            "code_run_hash": code_run_hash,
            "model_id": model_id,
            "model_revision": model_revision,
            "model_snapshot_hash": model_snapshot_hash,
            "extractor_version": prediction["extractor_version"],
            "scorer_version": prediction["scorer_version"],
            "harness_version": "day10_humaneval_e2b_v1",
            "raw_output_hash": prediction["raw_output_hash"],
            "raw_prompt_hash": record["raw_prompt_hash"],
            "reference_hash": record["reference_hash"],
            "output_token_ids_hash": prediction["output_token_ids_hash"],
            "parsed_completion_hash": completion_hash,
            "completion_hash": completion_hash,
            "source_file_sha256": frozen_protocol["source"]["file_sha256"],
            "source_revision": frozen_protocol["source"]["revision"],
            "sandbox_contract_hash": code_execution_protocol_hash,
            "code_execution_protocol": frozen_protocol,
            "code_execution_protocol_hash": code_execution_protocol_hash,
            "sandbox_contract_file_sha256": sandbox_contract_file_sha256,
            "evaluator_source_sha256": frozen_protocol["evaluator_source_sha256"],
            "sandbox": {
                "backend": frozen_protocol["backend"],
                "sdk": frozen_protocol["sdk"],
                "template_id": frozen_protocol["template_id"],
                "template_runtime": frozen_protocol["template_runtime"],
                "fresh_sandbox_per_sample": True,
                "secure": True,
                "allow_internet_access": False,
                "allow_public_traffic": False,
                "deny_out": ["0.0.0.0/0"],
            },
        }
        for field, expected in expected_values.items():
            if row.get(field) != expected:
                raise Day10AnalysisError(
                    f"code result {field} differs from frozen evidence: {sample_id}"
                )
        if (
            lineage.get("source") != frozen_protocol["source"]["source"]
            or lineage.get("revision") != frozen_protocol["source"]["revision"]
            or lineage.get("source_split") != frozen_protocol["source"]["split"]
            or lineage.get("source_file") != frozen_protocol["source"]["file"]
            or lineage.get("source_file_sha256")
            != frozen_protocol["source"]["file_sha256"]
        ):
            raise Day10AnalysisError(f"code source lineage mismatch: {sample_id}")
        _require_sha256(
            row.get("composed_program_hash"),
            f"composed program hash for {sample_id}",
            prefixed=True,
        )
        latency = _require_positive_number(
            row.get("latency_seconds"), f"sandbox latency for {sample_id}"
        )
        observed = _require_exact_keys(
            row.get("observed"), {"elapsed_seconds"}, f"observed result for {sample_id}"
        )
        if observed.get("elapsed_seconds") != latency:
            raise Day10AnalysisError(f"sandbox latency observation mismatch: {sample_id}")
        _verify_code_result_state(row, sample_id)

    correct = sum(int(row["score"]) for row in rows)
    return {
        "records": len(rows),
        "scored": len(rows),
        "correct": correct,
        "wrong_answer": len(rows) - correct,
        "results_file_sha256": results_file_sha256,
        "results_semantic_hash": _code_results_semantic_hash(
            rows, code_run_hash, complete_comparison_key
        ),
        "code_run_hash": code_run_hash,
        "code_execution_protocol_hash": code_execution_protocol_hash,
        "complete_comparison_key": complete_comparison_key,
        "sandbox_contract_file_sha256": sandbox_contract_file_sha256,
        "evaluator_source_sha256": frozen_protocol["evaluator_source_sha256"],
        "source_file_sha256": frozen_protocol["source"]["file_sha256"],
        "source_revision": frozen_protocol["source"]["revision"],
        "backend": frozen_protocol["backend"],
        "template_id": frozen_protocol["template_id"],
        "sdk": frozen_protocol["sdk"],
    }


def analyze_baseline(
    manifest_path: Path,
    predictions_path: Path,
    code_results_path: Path | None = None,
) -> dict[str, Any]:
    manifest, manifest_file_sha256 = load_manifest(manifest_path)
    manifest_by_id, dev_order, contract = verify_manifest(
        manifest, manifest_file_sha256
    )
    predictions = load_predictions(predictions_path)
    if len(predictions) != EXPECTED_DEV_RECORDS:
        raise Day10AnalysisError(
            f"expected {EXPECTED_DEV_RECORDS} predictions, found {len(predictions)}"
        )
    prediction_ids = [prediction.get("sample_id") for prediction in predictions]
    if prediction_ids != dev_order:
        raise Day10AnalysisError("prediction sample IDs/order differ from frozen dev order")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise Day10AnalysisError("prediction sample IDs are duplicated")
    if any(prediction.get("evaluation_split") == "frozen_test" for prediction in predictions):
        raise Day10AnalysisError("frozen_test prediction is prohibited")

    header = contract["header"]
    repeated_fields = (
        "run_hash",
        "run_started_at_utc",
        "manifest_hash",
        "manifest_file_sha256",
        "config_file_sha256",
        "protocol_hash",
        "scorer_registry_hash",
        "execution_protocol",
        "execution_protocol_hash",
        "comparison_key",
        "model_id",
        "model_revision",
        "model_snapshot_path",
        "model_snapshot_source",
        "model_snapshot_hash",
        "model_files",
        "runtime",
        "selection_policy",
    )
    uniform = {field: _uniform_value(predictions, field) for field in repeated_fields}
    for field in repeated_fields:
        if uniform[field] is None:
            raise Day10AnalysisError(f"required prediction field is missing: {field}")
    _require_string(predictions[0], "run_started_at_utc", "prediction")

    if uniform["manifest_hash"] != contract["manifest_hash"]:
        raise Day10AnalysisError("prediction manifest hash differs from manifest")
    if uniform["manifest_file_sha256"] != contract["manifest_file_sha256"]:
        raise Day10AnalysisError("prediction manifest file hash differs from manifest")
    for field in ("config_file_sha256", "protocol_hash", "scorer_registry_hash"):
        if uniform[field] != header[field]:
            raise Day10AnalysisError(f"prediction {field} differs from manifest")

    runtime = uniform["runtime"]
    execution_protocol = uniform["execution_protocol"]
    if not isinstance(runtime, dict) or not isinstance(execution_protocol, dict):
        raise Day10AnalysisError("runtime/execution protocol must be objects")
    _verify_execution_protocol(execution_protocol, runtime, header)
    calculated_execution_hash = semantic_hash(execution_protocol)
    if uniform["execution_protocol_hash"] != calculated_execution_hash:
        raise Day10AnalysisError("execution protocol hash mismatch")
    calculated_comparison_key = semantic_hash(
        {
            "domain": "day10.checkpoint_comparison",
            "schema_version": 1,
            "dataset_context_hash": header["dataset_context_hash"],
            "eval_suite_hash": header["eval_suite_hash"],
            "protocol_hash": header["protocol_hash"],
            "scorer_registry_hash": header["scorer_registry_hash"],
            "execution_protocol_hash": calculated_execution_hash,
        }
    )
    if uniform["comparison_key"] != calculated_comparison_key:
        raise Day10AnalysisError("checkpoint comparison key mismatch")
    _verify_model_identity(predictions[0], header)

    expected_selection_policy = {
        "version": "day10_frozen_dev_evaluation_order_v1",
        "evaluation_split": "dev",
        "sample_limit": None,
    }
    if uniform["selection_policy"] != expected_selection_policy:
        raise Day10AnalysisError("run is not the complete frozen dev selection")
    run_contract = {
        "domain": "day10.base_inference_run",
        "schema_version": 1,
        "manifest_hash": header["manifest_hash"],
        "config_file_sha256": header["config_file_sha256"],
        "protocol_hash": header["protocol_hash"],
        "scorer_registry_hash": header["scorer_registry_hash"],
        "execution_protocol_hash": calculated_execution_hash,
        "comparison_key": calculated_comparison_key,
        "model_id": uniform["model_id"],
        "model_revision": uniform["model_revision"],
        "model_snapshot_source": uniform["model_snapshot_source"],
        "model_snapshot_hash": uniform["model_snapshot_hash"],
        "evaluation_split": "dev",
        "selection_policy": expected_selection_policy,
        "sample_ids": dev_order,
        "generation": header["generation"],
        "runtime": runtime,
    }
    calculated_run_hash = semantic_hash(run_contract)
    if uniform["run_hash"] != calculated_run_hash:
        raise Day10AnalysisError("run hash mismatch")

    metrics = {
        slice_name: {
            "n": 0,
            "scored": 0,
            "correct": 0,
            "parse_error": 0,
            "wrong_answer": 0,
            "sandbox_pending": 0,
            "ceiling_hits": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
        }
        for slice_name in SLICES
    }
    for ordinal, prediction in enumerate(predictions):
        manifest_record = manifest_by_id[dev_order[ordinal]]
        _verify_prediction_payload(prediction, manifest_record, ordinal)
        if prediction.get("generation") != _expected_generation(
            header["generation"], manifest_record["generation_max_new_tokens"]
        ):
            raise Day10AnalysisError(
                f"generation contract mismatch: {manifest_record['sample_id']}"
            )
        _verify_model_identity(prediction, header)
        parse_error, wrong_answer, sandbox_pending = _score_classification(
            manifest_record["slice"], prediction.get("scorer_result")
        )
        slice_metrics = metrics[manifest_record["slice"]]
        slice_metrics["n"] += 1
        slice_metrics["parse_error"] += parse_error
        slice_metrics["wrong_answer"] += wrong_answer
        slice_metrics["sandbox_pending"] += sandbox_pending
        score = prediction["scorer_result"].get("score")
        if score is not None:
            slice_metrics["scored"] += 1
            slice_metrics["correct"] += int(score)
        output_count = prediction["output_token_count"]
        if output_count == manifest_record["generation_max_new_tokens"]:
            slice_metrics["ceiling_hits"] += 1
        slice_metrics["input_tokens"] += prediction["input_token_count"]
        slice_metrics["output_tokens"] += output_count
        slice_metrics["latency_seconds"] += float(prediction["latency_seconds"])

    code_evidence: dict[str, Any] | None = None
    if code_results_path is not None:
        code_evidence = _verify_code_results(
            code_results_path=code_results_path,
            manifest_by_id=manifest_by_id,
            dev_order=dev_order,
            predictions=predictions,
            predictions_path=predictions_path,
            manifest_hash=header["manifest_hash"],
            manifest_file_sha256=manifest_file_sha256,
            run_hash=calculated_run_hash,
            comparison_key=calculated_comparison_key,
            model_id=uniform["model_id"],
            model_revision=uniform["model_revision"],
            model_snapshot_hash=uniform["model_snapshot_hash"],
        )
        code_metrics = metrics["code"]
        code_metrics["scored"] = code_evidence["scored"]
        code_metrics["correct"] = code_evidence["correct"]
        code_metrics["wrong_answer"] = code_evidence["wrong_answer"]
        code_metrics["sandbox_pending"] = 0

    per_slice: dict[str, dict[str, Any]] = {}
    for slice_name in SLICES:
        values = metrics[slice_name]
        if values["n"] != EXPECTED_RECORDS_PER_SLICE:
            raise Day10AnalysisError(f"prediction slice count mismatch: {slice_name}")
        accuracy = (
            values["correct"] / values["scored"] if values["scored"] else None
        )
        per_slice[slice_name] = {
            **{key: values[key] for key in (
                "n",
                "scored",
                "correct",
                "parse_error",
                "wrong_answer",
                "sandbox_pending",
                "ceiling_hits",
                "input_tokens",
                "output_tokens",
            )},
            "accuracy": round(accuracy, 12) if accuracy is not None else None,
            "wilson_95": wilson_interval(values["correct"], values["scored"]),
            "latency_seconds": round(values["latency_seconds"], 6),
            "tokens_per_second": round(
                values["output_tokens"] / values["latency_seconds"], 6
            ),
        }

    provisional_accuracy = sum(
        per_slice[slice_name]["accuracy"] for slice_name in SCORED_SLICES
    ) / len(SCORED_SLICES)
    provisional_correct = sum(per_slice[name]["correct"] for name in SCORED_SLICES)
    provisional_scored = sum(per_slice[name]["scored"] for name in SCORED_SLICES)
    summary: dict[str, Any] = {
        "domain": "day10.baseline_summary",
        "schema_version": 1,
        "status": "provisional_code_sandbox_pending",
        "provenance": {
            "manifest_hash": header["manifest_hash"],
            "manifest_file_sha256": manifest_file_sha256,
            "protocol_hash": header["protocol_hash"],
            "scorer_registry_hash": header["scorer_registry_hash"],
            "execution_protocol_hash": calculated_execution_hash,
            "comparison_key": calculated_comparison_key,
            "run_hash": calculated_run_hash,
            "model_id": uniform["model_id"],
            "model_revision": uniform["model_revision"],
            "model_snapshot_hash": uniform["model_snapshot_hash"],
            "observed_run_started_at_utc": uniform["run_started_at_utc"],
        },
        "counts": {
            "predictions": len(predictions),
            "dev": len(predictions),
            "frozen_test": 0,
        },
        "per_slice": per_slice,
        "aggregates": {
            "provisional_three_slice_macro": {
                "status": "provisional_excludes_code_sandbox",
                "included_slices": list(SCORED_SLICES),
                "scored": provisional_scored,
                "correct": provisional_correct,
                "accuracy": round(provisional_accuracy, 12),
                "pooled_wilson_95": wilson_interval(
                    provisional_correct, provisional_scored
                ),
            },
            "four_slice_macro": {
                "status": "incomplete_due_to_code_sandbox",
                "included_slices": list(SLICES),
                "accuracy": None,
                "wilson_95": None,
            },
        },
        "uncertainty": {
            "interval": "wilson_score_interval_95_v1",
            "z": WILSON_Z_95,
            "interpretation": (
                "Pilot-set sensitivity only; these intervals do not establish "
                "population coverage or benchmark-level confidence."
            ),
        },
        "summary_hash_definition": (
            "RFC 8785 over this summary excluding summary_hash and "
            "provenance.observed_run_started_at_utc; no analysis wall-clock "
            "timestamp is recorded."
        ),
    }
    if code_evidence is not None:
        four_slice_correct = sum(
            per_slice[slice_name]["correct"] for slice_name in SLICES
        )
        four_slice_scored = sum(
            per_slice[slice_name]["scored"] for slice_name in SLICES
        )
        four_slice_accuracy = sum(
            per_slice[slice_name]["accuracy"] for slice_name in SLICES
        ) / len(SLICES)
        summary["status"] = "complete_code_sandbox_scored"
        summary["provenance"]["code_sandbox"] = {
            "results_file_sha256": code_evidence["results_file_sha256"],
            "results_semantic_hash": code_evidence["results_semantic_hash"],
            "code_run_hash": code_evidence["code_run_hash"],
            "code_execution_protocol_hash": code_evidence[
                "code_execution_protocol_hash"
            ],
            "complete_comparison_key": code_evidence["complete_comparison_key"],
            "sandbox_contract_file_sha256": code_evidence[
                "sandbox_contract_file_sha256"
            ],
            "evaluator_source_sha256": code_evidence["evaluator_source_sha256"],
            "source_file_sha256": code_evidence["source_file_sha256"],
            "source_revision": code_evidence["source_revision"],
            "backend": code_evidence["backend"],
            "template_id": code_evidence["template_id"],
            "sdk": code_evidence["sdk"],
        }
        summary["counts"]["code_results"] = code_evidence["records"]
        summary["aggregates"]["four_slice_macro"] = {
            "status": "complete",
            "included_slices": list(SLICES),
            "scored": four_slice_scored,
            "correct": four_slice_correct,
            "accuracy": round(four_slice_accuracy, 12),
            "wilson_95": None,
            "uncertainty_status": (
                "preregistered_stratified_bootstrap_not_computed"
            ),
        }
        summary["summary_hash_definition"] = (
            "RFC 8785 over this summary excluding summary_hash, "
            "provenance.observed_run_started_at_utc, and "
            "provenance.code_sandbox.results_file_sha256; exact sidecar bytes and "
            "analysis wall-clock timestamps are not semantic result identity."
        )
    hash_material = copy.deepcopy(summary)
    hash_material["provenance"].pop("observed_run_started_at_utc")
    if code_evidence is not None:
        hash_material["provenance"]["code_sandbox"].pop("results_file_sha256")
    summary["summary_hash"] = semantic_hash(hash_material)
    return summary


def write_summary_atomic(
    summary: dict[str, Any], output_path: Path, *, overwrite: bool
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise Day10AnalysisError(
            f"output already exists (pass --overwrite to replace it): {output_path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise Day10AnalysisError(
                    f"output appeared during analysis and was not replaced: {output_path}"
                ) from error
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--code-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        summary = analyze_baseline(
            args.manifest, args.predictions, code_results_path=args.code_results
        )
        write_summary_atomic(summary, args.output, overwrite=args.overwrite)
    except Day10AnalysisError as error:
        parser.error(str(error))
    print(
        f"summary=written path={args.output.resolve()} "
        f"run_hash={summary['provenance']['run_hash']} "
        f"summary_hash={summary['summary_hash']}"
    )


if __name__ == "__main__":
    main()
