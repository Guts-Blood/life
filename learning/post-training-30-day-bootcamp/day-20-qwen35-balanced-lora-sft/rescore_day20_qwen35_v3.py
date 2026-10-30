#!/usr/bin/env python3
"""Build a deterministic Qwen3.5-v3 normalized sidecar from raw v2 inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import day20_candidate_factory_v2 as candidate_factory
import day20_eval_identity_v2 as eval_identity
from day20_contract_v2 import (
    V2_CONTRACT_VERSION,
    Day20V2ContractError,
    validate_raw_code_continuation,
)


SCHEMA_VERSION = 2
RESCORER_VERSION = "day20-qwen35-normalized-rescore-v3"
SUMMARY_DOMAIN = "day20.v2.normalized_summary"
ROW_DOMAIN = "day20.v2.normalized_prediction"
RAW_SUMMARY_DOMAIN = "day20.v2.raw_evaluation_summary"
RAW_ROW_DOMAIN = "day20.v2.raw_evaluation_prediction"
CODE_PARSER_VERSION = "day20-code-continuation-parser-v2"
CODE_COMPOSER_VERSION = "day20-code-prefix-continuation-composer-v2"
TARGET_FORMAT_BY_SKILL = {
    "general": "general_mcq",
    "math": "math_reasoning",
    "finance": "finance_value_scale",
    "code": "code_continuation",
}

HERE = Path(__file__).resolve().parent
DEFAULT_EVALUATOR_PATH = HERE / "evaluate_day20_v2.py"
DEFAULT_RESPONSE_ADAPTER_PATH = HERE / "qwen35_response_adapter_v3.py"
DEFAULT_CODE_CONTRACT_PATH = HERE / "day20_contract_v2.py"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREFIXED_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZED_ROW_FIELDS = {
    "schema_version",
    "domain",
    "ordinal",
    "candidate",
    "scope",
    "sample_id",
    "slice",
    "target_format",
    "reference_hash",
    "prompt_messages_hash",
    "source_raw_row_sha256",
    "raw_output_sha256",
    "response_adapter",
    "normalized_output",
    "normalized_output_sha256",
    "normalized_scorer_result",
    "format_compliant",
    "anomalies",
    "normalized_comparison_key",
    "evaluation_run_sha256",
    "code_candidate",
    "code_static_syntax",
    "sandbox_execution_eligible",
    "source_lineage",
    "metadata",
    "raw_prompt",
    "reference",
    "row_sha256",
}
_NORMALIZED_SUMMARY_FIELDS = {
    "schema_version",
    "domain",
    "status",
    "candidate",
    "candidate_identity",
    "scope",
    "comparison_context",
    "normalized_comparison_key",
    "evaluation_run_sha256",
    "checkpoint_package_sha256",
    "model_identity_sha256",
    "source_raw",
    "protocol",
    "metrics",
    "predictions",
    "summary_sha256",
}


class Day20Qwen35RescoreV3Error(ValueError):
    """A raw or normalized evaluation identity failed closed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise Day20Qwen35RescoreV3Error("hashed text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20Qwen35RescoreV3Error(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Day20Qwen35RescoreV3Error(
            f"{label} must be a lowercase bare SHA-256"
        )
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20Qwen35RescoreV3Error(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day20Qwen35RescoreV3Error(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day20Qwen35RescoreV3Error(f"cannot load JSONL: {path}") from error
    if not lines:
        raise Day20Qwen35RescoreV3Error(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, 1):
        if not line.strip():
            raise Day20Qwen35RescoreV3Error(
                f"blank JSONL row: {path}:{ordinal}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day20Qwen35RescoreV3Error(
                f"invalid JSONL: {path}:{ordinal}"
            ) from error
        if not isinstance(row, dict):
            raise Day20Qwen35RescoreV3Error(
                f"JSONL row is not an object: {path}:{ordinal}"
            )
        rows.append(row)
    return rows


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise Day20Qwen35RescoreV3Error(f"implementation file is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Day20Qwen35RescoreV3Error(f"cannot import implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day20Qwen35RescoreV3Error(
            f"cannot import implementation: {path}"
        ) from error
    return module


def _normalized_json(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise Day20Qwen35RescoreV3Error("component returned non-JSON evidence") from error


def _self_hash_valid(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == object_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _selected_records(
    *,
    eval_manifest: Mapping[str, Any],
    experiment_manifest: Mapping[str, Any],
    scope: str,
) -> list[dict[str, Any]]:
    records = eval_manifest.get("records")
    order = eval_manifest.get("header", {}).get("evaluation_order", {}).get("dev")
    if not isinstance(records, list) or not isinstance(order, list):
        raise Day20Qwen35RescoreV3Error("frozen eval manifest is malformed")
    by_id = {
        row.get("sample_id"): row
        for row in records
        if isinstance(row, dict)
        and row.get("evaluation_split") == "dev"
        and isinstance(row.get("sample_id"), str)
    }
    if len(order) != 112 or len(set(order)) != 112 or set(order) != set(by_id):
        raise Day20Qwen35RescoreV3Error("frozen dev order drifted")
    if scope == "full112":
        selected_ids = order
    else:
        diagnostic = experiment_manifest.get("datasets", {}).get("diagnostic")
        if not isinstance(diagnostic, Mapping):
            raise Day20Qwen35RescoreV3Error("diagnostic identity is missing")
        selected_ids = diagnostic.get("ordered_sample_ids")
        if (
            not isinstance(selected_ids, list)
            or len(selected_ids) != 32
            or len(set(selected_ids)) != 32
            or diagnostic.get("selection_sha256") != object_sha256(selected_ids)
        ):
            raise Day20Qwen35RescoreV3Error("diagnostic selection drifted")
    try:
        selected = [dict(by_id[sample_id]) for sample_id in selected_ids]
    except KeyError as error:
        raise Day20Qwen35RescoreV3Error(
            "evaluation selection references an unknown sample"
        ) from error
    expected_per_skill = eval_identity.EXPECTED_RECORDS_BY_SCOPE[scope] // 4
    if Counter(row.get("slice") for row in selected) != Counter(
        {skill: expected_per_skill for skill in TARGET_FORMAT_BY_SKILL}
    ):
        raise Day20Qwen35RescoreV3Error("evaluation slice distribution drifted")
    return selected


def _adapt_raw_row(row: Mapping[str, Any], adapter: ModuleType) -> dict[str, Any]:
    if getattr(adapter, "ADAPTER_VERSION", None) != eval_identity.RESPONSE_ADAPTER_VERSION:
        raise Day20Qwen35RescoreV3Error("response adapter version drifted")
    function = getattr(adapter, "adapt_generated_token_decode", None)
    if not callable(function):
        raise Day20Qwen35RescoreV3Error("response adapter v3 API is incomplete")
    capture = row.get("response_capture")
    if not isinstance(capture, Mapping):
        raise Day20Qwen35RescoreV3Error("raw response capture is missing")
    validate = getattr(adapter, "validate_response_capture", None)
    if not callable(validate):
        raise Day20Qwen35RescoreV3Error("response adapter validator is missing")
    try:
        validated_text = validate(
            capture,
            raw_output=row.get("raw_output"),
            generation=row.get("generation"),
        )
        adapted = function(
            capture.get("message_content"),
            generated_token_ids=capture.get("generated_token_ids"),
            generated_only_text=capture.get("generated_only_text"),
        )
    except (TypeError, ValueError) as error:
        raise Day20Qwen35RescoreV3Error("response adapter rejected raw evidence") from error
    normalized = _normalized_json(adapted)
    if not isinstance(normalized, dict):
        raise Day20Qwen35RescoreV3Error("response adapter evidence is not an object")
    final_text = normalized.get("final_text")
    if (
        not isinstance(final_text, str)
        or final_text != validated_text
        or normalized.get("final_text_sha256") != text_sha256(final_text)
        or capture.get("response_adapter") != normalized
    ):
        raise Day20Qwen35RescoreV3Error("response adapter evidence drifted")
    return normalized


def _code_evidence(raw_output: str, raw_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        result = validate_raw_code_continuation(raw_output, raw_prompt)
    except Day20V2ContractError as error:
        message = str(error)
        candidate = {
            "candidate_mode": "completion",
            "candidate": raw_output,
            "execution_eligible": False,
            "contract_version": V2_CONTRACT_VERSION,
            "raw_sha256": text_sha256(raw_output),
            "canonical_sha256": None,
            "ast_sha256": None,
            "error": message,
        }
        return candidate, {"valid": False, "error": message}
    candidate = {
        "candidate_mode": "completion",
        "candidate": raw_output,
        "execution_eligible": True,
        "contract_version": V2_CONTRACT_VERSION,
        "raw_sha256": result.raw_sha256,
        "canonical_sha256": result.canonical_sha256,
        "ast_sha256": result.ast_sha256,
        "error": None,
    }
    return candidate, {"valid": True, "error": None}


def _normalized_code_score(candidate: Mapping[str, Any]) -> dict[str, Any]:
    eligible = candidate["execution_eligible"] is True
    return {
        "parsed_answer": candidate["candidate"],
        "canonical_answer": None,
        "canonical_reference": None,
        "parse_status": "ok" if eligible else "parse_error",
        "score_status": "sandbox_required" if eligible else "not_eligible",
        "score": None,
        "error_type": None if eligible else "code_contract_invalid",
    }


def _build_row(
    *,
    raw: Mapping[str, Any],
    record: Mapping[str, Any],
    adapter_evidence: Mapping[str, Any],
    scorer: ModuleType,
    candidate: str,
    scope: str,
    comparison_key: str,
    run_hash: str,
) -> dict[str, Any]:
    skill = record["slice"]
    final_text = adapter_evidence["final_text"]
    code_candidate = None
    code_static_syntax = None
    sandbox_eligible = None
    source_lineage = None
    metadata = None
    raw_prompt = None
    reference = None
    if skill == "code":
        raw_prompt = record.get("raw_prompt")
        reference = record.get("reference")
        if not isinstance(raw_prompt, str) or not isinstance(reference, str):
            raise Day20Qwen35RescoreV3Error("HumanEval source fields are incomplete")
        code_candidate, code_static_syntax = _code_evidence(final_text, raw_prompt)
        sandbox_eligible = code_candidate["execution_eligible"]
        normalized_score = _normalized_code_score(code_candidate)
        source_lineage = _normalized_json(record.get("source_lineage"))
        raw_metadata = record.get("metadata")
        if not isinstance(source_lineage, dict) or not isinstance(raw_metadata, Mapping):
            raise Day20Qwen35RescoreV3Error("HumanEval lineage is incomplete")
        metadata = {
            "entry_point": raw_metadata.get("entry_point"),
            "test_sha256": raw_metadata.get("test_sha256"),
        }
        if not all(isinstance(value, str) and value for value in metadata.values()):
            raise Day20Qwen35RescoreV3Error("HumanEval metadata is incomplete")
    else:
        score_function = getattr(scorer, "score_prediction", None)
        if not callable(score_function):
            raise Day20Qwen35RescoreV3Error("frozen scorer API is incomplete")
        try:
            normalized_score = _normalized_json(
                score_function(skill, final_text, record.get("reference"))
            )
        except Exception as error:
            raise Day20Qwen35RescoreV3Error(
                f"frozen scorer rejected {record.get('sample_id')}"
            ) from error
        if not isinstance(normalized_score, dict):
            raise Day20Qwen35RescoreV3Error("frozen scorer result is not an object")

    anomalies = list(raw.get("anomalies", []))
    if not all(isinstance(item, str) and item for item in anomalies):
        raise Day20Qwen35RescoreV3Error("raw anomalies are malformed")
    if skill == "code" and not sandbox_eligible:
        anomalies.append("code_contract_invalid")
    anomalies = sorted(set(anomalies))
    format_compliant = bool(final_text) and (
        bool(sandbox_eligible)
        if skill == "code"
        else normalized_score.get("parse_status") == "ok"
    )
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": ROW_DOMAIN,
        "ordinal": raw["ordinal"],
        "candidate": candidate,
        "scope": scope,
        "sample_id": record["sample_id"],
        "slice": skill,
        "target_format": TARGET_FORMAT_BY_SKILL[skill],
        "reference_hash": record["reference_hash"],
        "prompt_messages_hash": record["messages_hash"],
        "source_raw_row_sha256": raw["row_sha256"],
        "raw_output_sha256": raw["raw_output_sha256"],
        "response_adapter": dict(adapter_evidence),
        "normalized_output": final_text,
        "normalized_output_sha256": text_sha256(final_text),
        "normalized_scorer_result": normalized_score,
        "format_compliant": format_compliant,
        "anomalies": anomalies,
        "normalized_comparison_key": comparison_key,
        "evaluation_run_sha256": run_hash,
        "code_candidate": code_candidate,
        "code_static_syntax": code_static_syntax,
        "sandbox_execution_eligible": sandbox_eligible,
        "source_lineage": source_lineage,
        "metadata": metadata,
        "raw_prompt": raw_prompt,
        "reference": reference,
    }
    row["row_sha256"] = object_sha256(row)
    return row


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_skill: dict[str, dict[str, Any]] = {}
    for skill in TARGET_FORMAT_BY_SKILL:
        selected = [row for row in rows if row.get("slice") == skill]
        if skill == "code":
            by_skill[skill] = {
                "records": len(selected),
                "scored_records": 0,
                "correct": None,
                "accuracy": None,
                "format_compliant": sum(
                    row.get("format_compliant") is True for row in selected
                ),
                "sandbox_execution_eligible": sum(
                    row.get("sandbox_execution_eligible") is True for row in selected
                ),
                "score_status": "sandbox_required",
            }
            continue
        scores = [row["normalized_scorer_result"].get("score") for row in selected]
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            for score in scores
        ):
            raise Day20Qwen35RescoreV3Error("non-Code scorer returned a non-finite score")
        correct = sum(float(score) for score in scores)
        by_skill[skill] = {
            "records": len(selected),
            "scored_records": len(scores),
            "correct": correct,
            "accuracy": correct / len(scores) if scores else None,
            "format_compliant": sum(
                row.get("format_compliant") is True for row in selected
            ),
        }
    return {
        "records": len(rows),
        "non_code_correct": sum(
            by_skill[skill]["correct"] for skill in ("general", "math", "finance")
        ),
        "format_compliant": sum(row.get("format_compliant") is True for row in rows),
        "code_sandbox_execution_eligible": by_skill["code"][
            "sandbox_execution_eligible"
        ],
        "infrastructure_failures": 0,
        "by_skill": by_skill,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _verify_raw_pair(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    predictions_path: Path,
    selected: Sequence[Mapping[str, Any]],
    candidate: str,
    scope: str,
    adapter: ModuleType,
) -> list[dict[str, Any]]:
    expected_records = eval_identity.EXPECTED_RECORDS_BY_SCOPE[scope]
    identity = candidate_factory.parse_candidate_id(candidate)
    predictions = summary.get("predictions")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("domain") != RAW_SUMMARY_DOMAIN
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("candidate") != candidate
        or summary.get("candidate_identity") != identity.as_dict()
        or summary.get("eval_scope") != scope
        or not _self_hash_valid(summary, "summary_sha256")
        or not isinstance(predictions, Mapping)
        or predictions.get("records") != expected_records
        or Path(str(predictions.get("path", ""))).resolve()
        != predictions_path.resolve()
        or predictions.get("file_sha256") != file_sha256(predictions_path)
        or predictions.get("content_sha256") != object_sha256(rows)
        or predictions.get("ordered_sample_ids_sha256")
        != object_sha256([row.get("sample_id") for row in rows])
        or len(rows) != expected_records
        or len(selected) != expected_records
    ):
        raise Day20Qwen35RescoreV3Error("raw evaluation summary drifted")
    adapted_rows: list[dict[str, Any]] = []
    for ordinal, (row, record) in enumerate(zip(rows, selected), 1):
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("domain") != RAW_ROW_DOMAIN
            or row.get("ordinal") != ordinal
            or row.get("candidate") != candidate
            or row.get("candidate_identity") != identity.as_dict()
            or row.get("eval_scope") != scope
            or row.get("sample_id") != record.get("sample_id")
            or row.get("skill") != record.get("slice")
            or row.get("target_format") != TARGET_FORMAT_BY_SKILL[record["slice"]]
            or row.get("raw_prompt") != record.get("raw_prompt")
            or row.get("raw_prompt_sha256") != text_sha256(record["raw_prompt"])
            or row.get("reference") != record.get("reference")
            or row.get("reference_sha256") != text_sha256(record["reference"])
            or row.get("raw_output_sha256") != text_sha256(row.get("raw_output"))
            or row.get("model_identity", {}).get("model_identity_sha256")
            != summary.get("model_identity_sha256")
            or row.get("checkpoint_package", {}).get(
                "checkpoint_package_sha256"
            )
            != summary.get("checkpoint_package_sha256")
            or not _self_hash_valid(row, "row_sha256")
        ):
            raise Day20Qwen35RescoreV3Error(
                f"raw prediction identity drifted at ordinal {ordinal}"
            )
        adapted_rows.append(_adapt_raw_row(row, adapter))
    return adapted_rows


def _validate_model_and_checkpoint(summary: Mapping[str, Any]) -> tuple[str, str]:
    model_sha = _require_sha256(
        summary.get("model_identity_sha256"), "model_identity_sha256"
    )
    checkpoint_sha = _require_sha256(
        summary.get("checkpoint_package_sha256"), "checkpoint_package_sha256"
    )
    model = summary.get("model_identity")
    checkpoint = summary.get("checkpoint_package")
    if not isinstance(model, Mapping) or not isinstance(checkpoint, Mapping):
        raise Day20Qwen35RescoreV3Error("raw model/checkpoint provenance is missing")
    if object_sha256(model) != model_sha or object_sha256(checkpoint) != checkpoint_sha:
        raise Day20Qwen35RescoreV3Error("raw model/checkpoint provenance drifted")
    return model_sha, checkpoint_sha


def _verify_raw_protocol(
    *,
    summary: Mapping[str, Any],
    scope: str,
    eval_manifest: Mapping[str, Any],
    eval_manifest_path: Path,
    experiment_manifest: Mapping[str, Any],
    experiment_manifest_path: Path,
    scorers_path: Path,
    scorer: ModuleType,
    response_adapter_path: Path,
    adapter: ModuleType,
    evaluator_path: Path,
    evaluator_version: str,
) -> None:
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping):
        raise Day20Qwen35RescoreV3Error("raw evaluation protocol is missing")

    component_expectations = {
        "candidate_factory": (
            Path(candidate_factory.__file__).resolve(),
            candidate_factory.CONTRACT_VERSION,
        ),
        "data_contract": (DEFAULT_CODE_CONTRACT_PATH.resolve(), V2_CONTRACT_VERSION),
        "scorer": (
            scorers_path,
            getattr(scorer, "SCORER_REGISTRY_VERSION", None),
        ),
        "response_adapter": (
            response_adapter_path,
            getattr(adapter, "ADAPTER_VERSION", None),
        ),
        "evaluator": (evaluator_path, evaluator_version),
    }
    for name, (path, version) in component_expectations.items():
        identity = protocol.get(name)
        if (
            not isinstance(identity, Mapping)
            or Path(str(identity.get("path", ""))).resolve() != path
            or identity.get("file_sha256") != file_sha256(path)
            or identity.get("version") != version
        ):
            raise Day20Qwen35RescoreV3Error(
                f"raw {name} implementation provenance drifted"
            )

    eval_identity_block = protocol.get("eval_manifest")
    header = eval_manifest.get("header")
    manifest_hash = header.get("manifest_hash") if isinstance(header, Mapping) else None
    if isinstance(manifest_hash, str) and manifest_hash.startswith("sha256:"):
        manifest_hash = manifest_hash[len("sha256:") :]
    if (
        not isinstance(eval_identity_block, Mapping)
        or Path(str(eval_identity_block.get("path", ""))).resolve()
        != eval_manifest_path
        or eval_identity_block.get("file_sha256") != file_sha256(eval_manifest_path)
        or eval_identity_block.get("content_sha256") != manifest_hash
        or eval_identity_block.get("manifest_version")
        != (header.get("manifest_version") if isinstance(header, Mapping) else None)
    ):
        raise Day20Qwen35RescoreV3Error("raw eval-manifest provenance drifted")

    experiment_identity = protocol.get("experiment_manifest")
    if (
        not isinstance(experiment_identity, Mapping)
        or Path(str(experiment_identity.get("path", ""))).resolve()
        != experiment_manifest_path
        or experiment_identity.get("file_sha256")
        != file_sha256(experiment_manifest_path)
        or experiment_identity.get("content_sha256")
        != experiment_manifest.get("manifest_sha256")
    ):
        raise Day20Qwen35RescoreV3Error("raw experiment provenance drifted")

    diagnostic = experiment_manifest.get("datasets", {}).get("diagnostic")
    recorded_diagnostic = protocol.get("diagnostic")
    if not isinstance(diagnostic, Mapping) or not isinstance(
        recorded_diagnostic, Mapping
    ):
        raise Day20Qwen35RescoreV3Error("raw diagnostic provenance is missing")
    for name in ("path", "file_sha256", "records", "selection_sha256"):
        if recorded_diagnostic.get(name) != diagnostic.get(name):
            raise Day20Qwen35RescoreV3Error("raw diagnostic provenance drifted")

    generation = protocol.get("generation")
    parsed = candidate_factory.parse_candidate_id(summary["candidate"])
    expected_records = eval_identity.EXPECTED_RECORDS_BY_SCOPE[scope]
    if (
        not isinstance(generation, Mapping)
        or generation.get("split") != "dev"
        or generation.get("eval_scope") != scope
        or generation.get("records") != expected_records
        or generation.get("template") != "qwen3_5"
        or generation.get("backend") != "hf_transformers"
        or generation.get("use_mcore_gdn") is not False
        or generation.get("enable_thinking") is not False
        or generation.get("add_non_thinking_prefix") is not True
        or generation.get("greedy") is not True
        or generation.get("seed") != candidate_factory.PRIMARY_SEED
        or generation.get("response_capture")
        != "ms-swift RequestConfig(return_details=True)"
        or generation.get("generated_token_ids_retained") is not True
        or generation.get("adapter_loaded_unmerged")
        is not (parsed.model_role == "lora")
        or generation.get("code_execution") is not False
    ):
        raise Day20Qwen35RescoreV3Error("raw generation protocol drifted")


def _comparison_context(
    *,
    scope: str,
    eval_manifest_path: Path,
    experiment_manifest_path: Path,
    experiment_manifest: Mapping[str, Any],
    scorers_path: Path,
    scorer: ModuleType,
    response_adapter_path: Path,
    adapter: ModuleType,
    evaluator_path: Path,
    evaluator_version: str,
) -> dict[str, Any]:
    diagnostic = experiment_manifest.get("datasets", {}).get("diagnostic")
    selection = (
        diagnostic.get("selection_sha256")
        if scope == "probe32" and isinstance(diagnostic, Mapping)
        else None
    )
    context = {
        "contract_version": candidate_factory.CONTRACT_VERSION,
        "scope": scope,
        "records": eval_identity.EXPECTED_RECORDS_BY_SCOPE[scope],
        "diagnostic_selection_sha256": selection,
        "eval_manifest_file_sha256": file_sha256(eval_manifest_path),
        "experiment_manifest_file_sha256": file_sha256(experiment_manifest_path),
        "experiment_manifest_content_sha256": experiment_manifest.get(
            "manifest_sha256"
        ),
        "scorer_version": getattr(scorer, "SCORER_REGISTRY_VERSION", None),
        "scorer_file_sha256": file_sha256(scorers_path),
        "response_adapter_version": getattr(adapter, "ADAPTER_VERSION", None),
        "response_adapter_file_sha256": file_sha256(response_adapter_path),
        "code_parser_version": CODE_PARSER_VERSION,
        "code_parser_file_sha256": file_sha256(DEFAULT_CODE_CONTRACT_PATH),
        "code_composer_version": CODE_COMPOSER_VERSION,
        "code_composer_file_sha256": file_sha256(DEFAULT_CODE_CONTRACT_PATH),
        "evaluator_version": evaluator_version,
        "evaluator_file_sha256": file_sha256(evaluator_path),
        "rescorer_version": RESCORER_VERSION,
        "rescorer_file_sha256": file_sha256(Path(__file__).resolve()),
    }
    try:
        return eval_identity.normalize_comparison_context(context)
    except ValueError as error:
        raise Day20Qwen35RescoreV3Error(str(error)) from error


def verify_normalized_pair(
    predictions_path: Path,
    summary_path: Path,
    *,
    expected_scope: str | None = None,
    expected_candidate: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute all self-contained normalized sidecar identities."""
    predictions_path = predictions_path.resolve()
    summary_path = summary_path.resolve()
    rows = _load_jsonl(predictions_path)
    summary = _load_json(summary_path)
    candidate = summary.get("candidate")
    scope = summary.get("scope")
    if not isinstance(candidate, str):
        raise Day20Qwen35RescoreV3Error("normalized candidate is missing")
    try:
        parsed = candidate_factory.parse_candidate_id(candidate)
    except candidate_factory.CandidateFactoryV2Error as error:
        raise Day20Qwen35RescoreV3Error("normalized candidate is invalid") from error
    if (
        scope != parsed.scope
        or (expected_scope is not None and scope != expected_scope)
        or (expected_candidate is not None and candidate != expected_candidate)
    ):
        raise Day20Qwen35RescoreV3Error("normalized candidate scope drifted")
    context = summary.get("comparison_context")
    try:
        normalized_context = eval_identity.normalize_comparison_context(context)
        comparison = eval_identity.normalized_comparison_key(
            normalized_context, expected=summary.get("normalized_comparison_key")
        )
        run = eval_identity.evaluation_run_hash(
            normalized_context,
            candidate=candidate,
            checkpoint_package_sha256=summary.get("checkpoint_package_sha256"),
            model_identity_sha256=summary.get("model_identity_sha256"),
            raw_predictions_file_sha256=summary.get("source_raw", {}).get(
                "predictions_file_sha256"
            ),
            raw_summary_file_sha256=summary.get("source_raw", {}).get(
                "summary_file_sha256"
            ),
            raw_summary_content_sha256=summary.get("source_raw", {}).get(
                "summary_content_sha256"
            ),
            expected=summary.get("evaluation_run_sha256"),
        )
    except (AttributeError, ValueError) as error:
        raise Day20Qwen35RescoreV3Error("normalized comparison identity drifted") from error
    expected_records = eval_identity.EXPECTED_RECORDS_BY_SCOPE[scope]
    if len(rows) != expected_records:
        raise Day20Qwen35RescoreV3Error("normalized prediction count drifted")
    sample_ids: list[str] = []
    for ordinal, row in enumerate(rows, 1):
        normalized_output = row.get("normalized_output")
        response_adapter = row.get("response_adapter")
        scorer_result = row.get("normalized_scorer_result")
        anomalies = row.get("anomalies")
        if (
            set(row) != _NORMALIZED_ROW_FIELDS
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("domain") != ROW_DOMAIN
            or row.get("ordinal") != ordinal
            or row.get("candidate") != candidate
            or row.get("scope") != scope
            or row.get("normalized_comparison_key") != comparison
            or row.get("evaluation_run_sha256") != run
            or not isinstance(row.get("sample_id"), str)
            or row.get("slice") not in TARGET_FORMAT_BY_SKILL
            or row.get("target_format")
            != TARGET_FORMAT_BY_SKILL.get(row.get("slice"))
            or not isinstance(normalized_output, str)
            or row.get("normalized_output_sha256")
            != text_sha256(normalized_output)
            or not isinstance(response_adapter, Mapping)
            or response_adapter.get("adapter_version")
            != eval_identity.RESPONSE_ADAPTER_VERSION
            or response_adapter.get("final_text") != normalized_output
            or response_adapter.get("final_text_sha256")
            != row.get("normalized_output_sha256")
            or not isinstance(scorer_result, Mapping)
            or not isinstance(anomalies, list)
            or not all(isinstance(item, str) and item for item in anomalies)
            or not _require_sha256(
                row.get("source_raw_row_sha256"), "source_raw_row_sha256"
            )
            or not _require_sha256(
                row.get("raw_output_sha256"), "raw_output_sha256"
            )
            or not isinstance(row.get("reference_hash"), str)
            or not _PREFIXED_SHA256_RE.fullmatch(row["reference_hash"])
            or not isinstance(row.get("prompt_messages_hash"), str)
            or not _PREFIXED_SHA256_RE.fullmatch(row["prompt_messages_hash"])
            or not _self_hash_valid(row, "row_sha256")
        ):
            raise Day20Qwen35RescoreV3Error(
                f"normalized prediction drifted at ordinal {ordinal}"
            )
        sample_ids.append(row["sample_id"])
        if row.get("slice") == "code":
            source_lineage = row.get("source_lineage")
            metadata = row.get("metadata")
            if (
                not isinstance(row.get("raw_prompt"), str)
                or not isinstance(row.get("reference"), str)
                or not isinstance(source_lineage, Mapping)
                or not all(
                    isinstance(source_lineage.get(name), str)
                    and source_lineage[name]
                    for name in ("source", "revision", "parent_id")
                )
                or not isinstance(metadata, Mapping)
                or not isinstance(metadata.get("entry_point"), str)
                or not metadata["entry_point"]
                or not isinstance(metadata.get("test_sha256"), str)
                or not _SHA256_RE.fullmatch(metadata["test_sha256"])
            ):
                raise Day20Qwen35RescoreV3Error("normalized HumanEval lineage drifted")
            candidate_evidence, syntax = _code_evidence(
                row.get("normalized_output"), row.get("raw_prompt")
            )
            eligible = candidate_evidence["execution_eligible"]
            if (
                row.get("code_candidate") != candidate_evidence
                or row.get("code_static_syntax") != syntax
                or row.get("sandbox_execution_eligible") is not eligible
                or row.get("normalized_scorer_result")
                != _normalized_code_score(candidate_evidence)
            ):
                raise Day20Qwen35RescoreV3Error("normalized Code evidence drifted")
        elif any(
            row.get(key) is not None
            for key in (
                "code_candidate",
                "code_static_syntax",
                "sandbox_execution_eligible",
                "source_lineage",
                "metadata",
                "raw_prompt",
                "reference",
            )
        ):
            raise Day20Qwen35RescoreV3Error("non-Code row carries Code evidence")

    expected_per_skill = expected_records // 4
    if len(set(sample_ids)) != expected_records or Counter(
        row["slice"] for row in rows
    ) != Counter({skill: expected_per_skill for skill in TARGET_FORMAT_BY_SKILL}):
        raise Day20Qwen35RescoreV3Error("normalized sample order/distribution drifted")

    predictions = summary.get("predictions")
    source_raw = summary.get("source_raw")
    protocol = summary.get("protocol")
    expected_metrics = _metrics(rows)
    if (
        set(summary) != _NORMALIZED_SUMMARY_FIELDS
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("domain") != SUMMARY_DOMAIN
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("candidate_identity") != parsed.as_dict()
        or summary.get("comparison_context") != normalized_context
        or not isinstance(source_raw, Mapping)
        or set(source_raw)
        != {
            "predictions_path",
            "predictions_file_sha256",
            "predictions_content_sha256",
            "summary_path",
            "summary_file_sha256",
            "summary_content_sha256",
        }
        or not all(
            isinstance(source_raw.get(name), str) and source_raw[name]
            for name in ("predictions_path", "summary_path")
        )
        or not all(
            isinstance(source_raw.get(name), str)
            and _SHA256_RE.fullmatch(source_raw[name])
            for name in (
                "predictions_file_sha256",
                "predictions_content_sha256",
                "summary_file_sha256",
                "summary_content_sha256",
            )
        )
        or protocol
        != {
            "rescorer_version": RESCORER_VERSION,
            "candidate_code_executed_on_host": False,
            "code_candidate_mode": "completion",
            "indentation_repair_applied": False,
            "frozen_test_consumed": False,
        }
        or not isinstance(predictions, Mapping)
        or Path(str(predictions.get("path", ""))).resolve() != predictions_path
        or predictions.get("records") != expected_records
        or predictions.get("file_sha256") != file_sha256(predictions_path)
        or predictions.get("content_sha256") != object_sha256(rows)
        or predictions.get("ordered_sample_ids_sha256")
        != object_sha256([row["sample_id"] for row in rows])
        or summary.get("metrics") != expected_metrics
        or not _self_hash_valid(summary, "summary_sha256")
    ):
        raise Day20Qwen35RescoreV3Error("normalized summary drifted")
    return rows, summary


def rescore_artifacts(
    *,
    raw_predictions_path: Path,
    raw_summary_path: Path,
    eval_manifest_path: Path,
    experiment_manifest_path: Path,
    scorers_path: Path,
    output_dir: Path,
    scope: str,
    evaluator_path: Path = DEFAULT_EVALUATOR_PATH,
    response_adapter_path: Path = DEFAULT_RESPONSE_ADAPTER_PATH,
) -> dict[str, Any]:
    """Validate raw inference, rescore it, and publish deterministic sidecars."""
    if scope not in eval_identity.EXPECTED_RECORDS_BY_SCOPE:
        raise Day20Qwen35RescoreV3Error("scope must be probe32 or full112")
    paths = [
        raw_predictions_path,
        raw_summary_path,
        eval_manifest_path,
        experiment_manifest_path,
        scorers_path,
        evaluator_path,
        response_adapter_path,
    ]
    (
        raw_predictions_path,
        raw_summary_path,
        eval_manifest_path,
        experiment_manifest_path,
        scorers_path,
        evaluator_path,
        response_adapter_path,
    ) = [path.resolve() for path in paths]
    raw_summary = _load_json(raw_summary_path)
    raw_rows = _load_jsonl(raw_predictions_path)
    candidate = raw_summary.get("candidate")
    if not isinstance(candidate, str):
        raise Day20Qwen35RescoreV3Error("raw candidate is missing")
    try:
        parsed = candidate_factory.parse_candidate_id(candidate)
    except candidate_factory.CandidateFactoryV2Error as error:
        raise Day20Qwen35RescoreV3Error("raw candidate is outside v2 grammar") from error
    if parsed.scope != scope:
        raise Day20Qwen35RescoreV3Error("raw candidate scope differs from --scope")

    eval_manifest = _load_json(eval_manifest_path)
    experiment_manifest = _load_json(experiment_manifest_path)
    if (
        experiment_manifest.get("schema_version") != 2
        or experiment_manifest.get("contract", {}).get("candidate_factory_version")
        != candidate_factory.CONTRACT_VERSION
        or not _self_hash_valid(experiment_manifest, "manifest_sha256")
    ):
        raise Day20Qwen35RescoreV3Error("experiment manifest identity drifted")
    selected = _selected_records(
        eval_manifest=eval_manifest,
        experiment_manifest=experiment_manifest,
        scope=scope,
    )
    scorer = _load_module(scorers_path, "day20_v3_frozen_scorer")
    adapter = _load_module(response_adapter_path, "day20_response_adapter_v3")
    evaluator = _load_module(evaluator_path, "day20_raw_evaluator_v2")
    evaluator_version = getattr(evaluator, "EVALUATOR_VERSION", None)
    if not isinstance(evaluator_version, str) or not evaluator_version:
        raise Day20Qwen35RescoreV3Error("raw evaluator version is missing")
    verify_raw = getattr(evaluator, "verify_published_pair", None)
    if not callable(verify_raw):
        raise Day20Qwen35RescoreV3Error("raw evaluator verifier is missing")
    try:
        verified_raw_summary = verify_raw(
            raw_predictions_path,
            raw_summary_path,
            expected_scope=scope,
            expected_candidate=candidate,
        )
    except Exception as error:
        raise Day20Qwen35RescoreV3Error(
            "raw evaluator rejected the published pair"
        ) from error
    if verified_raw_summary != raw_summary:
        raise Day20Qwen35RescoreV3Error("raw evaluator verifier returned drifted data")
    expected_scorer_sha = eval_manifest.get("header", {}).get("scorer_source_sha256")
    if expected_scorer_sha != file_sha256(scorers_path):
        raise Day20Qwen35RescoreV3Error("frozen scorer source identity drifted")
    _verify_raw_protocol(
        summary=raw_summary,
        scope=scope,
        eval_manifest=eval_manifest,
        eval_manifest_path=eval_manifest_path,
        experiment_manifest=experiment_manifest,
        experiment_manifest_path=experiment_manifest_path,
        scorers_path=scorers_path,
        scorer=scorer,
        response_adapter_path=response_adapter_path,
        adapter=adapter,
        evaluator_path=evaluator_path,
        evaluator_version=evaluator_version,
    )
    adapted_rows = _verify_raw_pair(
        rows=raw_rows,
        summary=raw_summary,
        predictions_path=raw_predictions_path,
        selected=selected,
        candidate=candidate,
        scope=scope,
        adapter=adapter,
    )
    model_sha, checkpoint_sha = _validate_model_and_checkpoint(raw_summary)
    comparison_context = _comparison_context(
        scope=scope,
        eval_manifest_path=eval_manifest_path,
        experiment_manifest_path=experiment_manifest_path,
        experiment_manifest=experiment_manifest,
        scorers_path=scorers_path,
        scorer=scorer,
        response_adapter_path=response_adapter_path,
        adapter=adapter,
        evaluator_path=evaluator_path,
        evaluator_version=evaluator_version,
    )
    comparison_key = eval_identity.normalized_comparison_key(comparison_context)
    raw_predictions_sha = file_sha256(raw_predictions_path)
    raw_summary_file_sha = file_sha256(raw_summary_path)
    run_hash = eval_identity.evaluation_run_hash(
        comparison_context,
        candidate=candidate,
        checkpoint_package_sha256=checkpoint_sha,
        model_identity_sha256=model_sha,
        raw_predictions_file_sha256=raw_predictions_sha,
        raw_summary_file_sha256=raw_summary_file_sha,
        raw_summary_content_sha256=raw_summary["summary_sha256"],
    )
    rows = [
        _build_row(
            raw=raw,
            record=record,
            adapter_evidence=adapted,
            scorer=scorer,
            candidate=candidate,
            scope=scope,
            comparison_key=comparison_key,
            run_hash=run_hash,
        )
        for raw, record, adapted in zip(raw_rows, selected, adapted_rows)
    ]
    output_dir = output_dir.resolve()
    predictions_path = output_dir / f"{candidate}.qwen35-v3.predictions.jsonl"
    summary_path = output_dir / f"{candidate}.qwen35-v3.json"
    source_raw = {
        "predictions_path": str(raw_predictions_path),
        "predictions_file_sha256": raw_predictions_sha,
        "predictions_content_sha256": object_sha256(raw_rows),
        "summary_path": str(raw_summary_path),
        "summary_file_sha256": raw_summary_file_sha,
        "summary_content_sha256": raw_summary["summary_sha256"],
    }
    predictions_bytes = _jsonl_bytes(rows)
    predictions_sha = hashlib.sha256(predictions_bytes).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": SUMMARY_DOMAIN,
        "status": "complete_with_code_sandbox_required",
        "candidate": candidate,
        "candidate_identity": parsed.as_dict(),
        "scope": scope,
        "comparison_context": comparison_context,
        "normalized_comparison_key": comparison_key,
        "evaluation_run_sha256": run_hash,
        "checkpoint_package_sha256": checkpoint_sha,
        "model_identity_sha256": model_sha,
        "source_raw": source_raw,
        "protocol": {
            "rescorer_version": RESCORER_VERSION,
            "candidate_code_executed_on_host": False,
            "code_candidate_mode": "completion",
            "indentation_repair_applied": False,
            "frozen_test_consumed": False,
        },
        "metrics": _metrics(rows),
        "predictions": {
            "path": str(predictions_path),
            "records": len(rows),
            "file_sha256": predictions_sha,
            "content_sha256": object_sha256(rows),
            "ordered_sample_ids_sha256": object_sha256(
                [row["sample_id"] for row in rows]
            ),
        },
    }
    summary["summary_sha256"] = object_sha256(summary)
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        if predictions_path.read_bytes() != predictions_bytes:
            raise Day20Qwen35RescoreV3Error("existing normalized predictions drifted")
    else:
        with predictions_path.open("xb") as handle:
            handle.write(predictions_bytes)
    if summary_path.exists():
        if summary_path.read_bytes() != summary_bytes:
            raise Day20Qwen35RescoreV3Error("existing normalized summary drifted")
    else:
        with summary_path.open("xb") as handle:
            handle.write(summary_bytes)
    _, verified = verify_normalized_pair(
        predictions_path,
        summary_path,
        expected_scope=scope,
        expected_candidate=candidate,
    )
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--raw-summary", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--scorers", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scope", required=True, choices=("probe32", "full112"))
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR_PATH)
    parser.add_argument(
        "--response-adapter", type=Path, default=DEFAULT_RESPONSE_ADAPTER_PATH
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rescore_artifacts(
        raw_predictions_path=args.raw_predictions,
        raw_summary_path=args.raw_summary,
        eval_manifest_path=args.eval_manifest,
        experiment_manifest_path=args.experiment_manifest,
        scorers_path=args.scorers,
        output_dir=args.output_dir,
        scope=args.scope,
        evaluator_path=args.evaluator,
        response_adapter_path=args.response_adapter,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


__all__ = [
    "RESCORER_VERSION",
    "ROW_DOMAIN",
    "SCHEMA_VERSION",
    "SUMMARY_DOMAIN",
    "Day20Qwen35RescoreV3Error",
    "rescore_artifacts",
    "verify_normalized_pair",
]


if __name__ == "__main__":
    main()
