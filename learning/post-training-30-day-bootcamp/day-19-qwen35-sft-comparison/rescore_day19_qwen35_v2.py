#!/usr/bin/env python3
"""Create a non-destructive Qwen3.5-adapted sidecar for Day 19 predictions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from day19_humaneval_adapter import (
    COMPOSER_VERSION,
    PARSER_VERSION,
    classify_code_candidate,
    compose_humaneval_program,
    composed_syntax_status,
)
from qwen35_response_adapter import (
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    adapt_qwen35_text,
    text_sha256,
    validate_ms_swift_response_capture,
)


SKILLS = ("general", "math", "code", "finance")
HERE = Path(__file__).resolve().parent
RESPONSE_ADAPTER_PATH = HERE / "qwen35_response_adapter.py"
CODE_ADAPTER_PATH = HERE / "day19_humaneval_adapter.py"
RESCORER_PATH = Path(__file__).resolve()
LEGACY_FALLBACK_RECIPES = frozenset(("baseline-a", "baseline-b", "best-e"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day19Qwen35RescoreError(ValueError):
    """A source prediction or sidecar invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day19Qwen35RescoreError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day19Qwen35RescoreError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day19Qwen35RescoreError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise Day19Qwen35RescoreError(f"JSONL file is missing: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise Day19Qwen35RescoreError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day19Qwen35RescoreError(
                f"invalid JSONL {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise Day19Qwen35RescoreError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Day19Qwen35RescoreError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.resolve()
    if path.exists():
        raise Day19Qwen35RescoreError(f"refusing to overwrite sidecar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise Day19Qwen35RescoreError(f"refusing to overwrite sidecar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _generated_only_text(row: dict[str, Any], sample_id: str) -> str | None:
    capture = row.get("response_capture")
    if capture is None:
        return None
    try:
        return validate_ms_swift_response_capture(
            capture,
            message_content=row.get("raw_output"),
            generation=row.get("generation"),
        )
    except (TypeError, ValueError) as error:
        raise Day19Qwen35RescoreError(
            f"invalid response capture for {sample_id}: {error}"
        ) from error


def _code_static_status(
    raw_prompt: str, entry_point: str, candidate: dict[str, Any]
) -> dict[str, Any]:
    source_prompt = raw_prompt + ("" if raw_prompt.endswith("\n") else "\n")
    try:
        program = compose_humaneval_program(
            source_prompt=source_prompt,
            source_test="",
            entry_point=entry_point,
            candidate=candidate["candidate"],
            candidate_mode=candidate["candidate_mode"],
        )
    except (TypeError, ValueError) as error:
        return {"valid": False, "error": type(error).__name__}
    return composed_syntax_status(program)


def adapter_comparison_key(
    *, manifest_file_sha256: str, scorer_file_sha256: str
) -> str:
    return object_sha256(
        {
            "domain": "day19.qwen35_adapter_comparison",
            "schema_version": 2,
            "manifest_file_sha256": manifest_file_sha256,
            "scorer_file_sha256": scorer_file_sha256,
            "response_adapter_version": ADAPTER_VERSION,
            "response_adapter_file_sha256": file_sha256(RESPONSE_ADAPTER_PATH),
            "code_parser_version": PARSER_VERSION,
            "code_adapter_file_sha256": file_sha256(CODE_ADAPTER_PATH),
            "code_composer_version": COMPOSER_VERSION,
            "rescorer_file_sha256": file_sha256(RESCORER_PATH),
        }
    )


def adapter_run_hash(
    *,
    recipe: str,
    comparison_key: str,
    checkpoint: str,
    model_snapshot_sha256: str,
    source_prediction_file_sha256: str,
    source_evaluation_summary_file_sha256: str,
    source_evaluation_summary_content_sha256: str,
) -> str:
    return object_sha256(
        {
            "domain": "day19.qwen35_adapter_run",
            "schema_version": 2,
            "recipe": recipe,
            "comparison_key": comparison_key,
            "checkpoint": checkpoint,
            "model_snapshot_sha256": model_snapshot_sha256,
            "source_prediction_file_sha256": source_prediction_file_sha256,
            "source_evaluation_summary_file_sha256": (
                source_evaluation_summary_file_sha256
            ),
            "source_evaluation_summary_content_sha256": (
                source_evaluation_summary_content_sha256
            ),
        }
    )


def verify_model_export_identity(model_export: Any) -> str:
    if (
        not isinstance(model_export, dict)
        or model_export.get("model_class") != "Qwen3_5ForConditionalGeneration"
        or model_export.get("load_and_generation_verified") is not True
    ):
        raise Day19Qwen35RescoreError("source model identity is incomplete")
    files = model_export.get("files")
    if (
        not isinstance(files, dict)
        or "config.json" not in files
        or not any(name.endswith(".safetensors") for name in files)
    ):
        raise Day19Qwen35RescoreError("source model file manifest is incomplete")
    for name, identity in files.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(identity, dict)
            or isinstance(identity.get("bytes"), bool)
            or not isinstance(identity.get("bytes"), int)
            or identity["bytes"] < 0
            or not isinstance(identity.get("sha256"), str)
            or not _SHA256_RE.fullmatch(identity["sha256"])
        ):
            raise Day19Qwen35RescoreError(
                f"invalid source model file identity: {name!r}"
            )
    snapshot = model_export.get("snapshot_sha256")
    if (
        not isinstance(snapshot, str)
        or not _SHA256_RE.fullmatch(snapshot)
        or snapshot != object_sha256(files)
    ):
        raise Day19Qwen35RescoreError("source model snapshot identity drifted")
    return snapshot


def verify_source_summary(
    *,
    summary: dict[str, Any],
    predictions_path: Path,
    manifest_path: Path,
    scorers_path: Path,
    manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    expected_summary_hash = object_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day19.qwen35_eval_summary"
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("summary_sha256") != expected_summary_hash
    ):
        raise Day19Qwen35RescoreError("source evaluation summary identity drifted")
    recipe = summary.get("recipe")
    if recipe not in (*sorted(LEGACY_FALLBACK_RECIPES), "untouched-c0"):
        raise Day19Qwen35RescoreError("evaluation summary has no known recipe identity")
    checkpoint = summary.get("checkpoint")
    model_export = summary.get("model_export")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise Day19Qwen35RescoreError("source model identity is incomplete")
    verify_model_export_identity(model_export)
    predictions = summary.get("predictions")
    source_prediction_sha = file_sha256(predictions_path)
    if (
        not isinstance(predictions, dict)
        or predictions.get("records") != 112
        or predictions.get("file_sha256") != source_prediction_sha
    ):
        raise Day19Qwen35RescoreError("source prediction identity differs from summary")
    protocol = summary.get("protocol")
    manifest_sha = file_sha256(manifest_path)
    scorer_sha = file_sha256(scorers_path)
    if not isinstance(protocol, dict):
        raise Day19Qwen35RescoreError("source evaluation protocol is missing")
    if (
        protocol.get("eval_manifest_file_sha256") != manifest_sha
        or protocol.get("scorer_source_sha256") != scorer_sha
        or manifest.get("header", {}).get("scorer_source_sha256") != scorer_sha
        or protocol.get("split") != "dev"
        or protocol.get("frozen_test_consumed") is not False
        or protocol.get("records") != 112
        or protocol.get("sample_limit") is not None
        or protocol.get("template") != "qwen3_5"
        or protocol.get("enable_thinking") is not False
        or protocol.get("add_non_thinking_prefix") is not True
    ):
        raise Day19Qwen35RescoreError("source evaluation protocol drifted")
    source_protocol = {
        "template": protocol["template"],
        "enable_thinking": protocol["enable_thinking"],
        "add_non_thinking_prefix": protocol["add_non_thinking_prefix"],
        "response_capture": protocol.get("response_capture"),
        "response_boundary_adapter": protocol.get("response_boundary_adapter"),
        "generated_token_ids_retained": protocol.get("generated_token_ids_retained"),
        "ms_swift_version": protocol.get("ms_swift_version"),
        "non_thinking_prefix_sha256": protocol.get("non_thinking_prefix_sha256"),
    }
    return recipe, source_protocol


def verify_source_predictions(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    scorers: ModuleType,
    predictions_path: Path,
    manifest_path: Path,
    scorers_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    order = manifest.get("header", {}).get("evaluation_order", {}).get("dev")
    records = manifest.get("records")
    if not isinstance(order, list) or not isinstance(records, list):
        raise Day19Qwen35RescoreError("frozen eval manifest is malformed")
    if len(order) != 112 or len(rows) != 112:
        raise Day19Qwen35RescoreError("v2 rescore requires all 112 frozen dev rows")
    by_id = {row.get("sample_id"): row for row in records if isinstance(row, dict)}
    if Counter(by_id[sample_id].get("slice") for sample_id in order) != Counter(
        {skill: 28 for skill in SKILLS}
    ):
        raise Day19Qwen35RescoreError("frozen dev slice distribution drifted")
    recipe, source_protocol = verify_source_summary(
        summary=summary,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        scorers_path=scorers_path,
        manifest=manifest,
    )

    verified_records: list[dict[str, Any]] = []
    for ordinal, (row, sample_id) in enumerate(zip(rows, order), 1):
        record = by_id.get(sample_id)
        if record is None:
            raise Day19Qwen35RescoreError(f"manifest record is missing: {sample_id}")
        if (
            row.get("schema_version") != 1
            or row.get("ordinal") != ordinal
            or row.get("recipe") != recipe
            or row.get("sample_id") != sample_id
            or row.get("slice") != record.get("slice")
            or row.get("reference_hash") != record.get("reference_hash")
            or row.get("prompt_messages_hash") != record.get("messages_hash")
        ):
            raise Day19Qwen35RescoreError(
                f"source prediction identity mismatch: {sample_id}"
            )
        raw_output = row.get("raw_output")
        if not isinstance(raw_output, str):
            raise Day19Qwen35RescoreError(f"source prediction is not text: {sample_id}")
        if row.get("raw_output_sha256") != text_sha256(raw_output):
            raise Day19Qwen35RescoreError(f"source prediction hash mismatch: {sample_id}")
        expected_legacy = normalized_json(
            scorers.score_prediction(record["slice"], raw_output, record["reference"])
        )
        if row.get("scorer_result") != expected_legacy:
            raise Day19Qwen35RescoreError(f"legacy scorer result drifted: {sample_id}")
        _generated_only_text(row, sample_id)
        verified_records.append(record)
    rows_with_capture = sum(row.get("response_capture") is not None for row in rows)
    capture_declared = (
        source_protocol.get("response_capture")
        == "ms-swift RequestConfig(return_details=True)"
        and source_protocol.get("generated_token_ids_retained") is True
    )
    if rows_with_capture not in (0, 112) or capture_declared != (rows_with_capture == 112):
        raise Day19Qwen35RescoreError("source response-capture coverage drifted")
    if rows_with_capture == 112 and (
        source_protocol.get("response_boundary_adapter") != ADAPTER_VERSION
        or source_protocol.get("non_thinking_prefix_sha256")
        != text_sha256(NON_THINKING_PREFIX)
        or not isinstance(source_protocol.get("ms_swift_version"), str)
        or not source_protocol["ms_swift_version"]
    ):
        raise Day19Qwen35RescoreError("token-backed response protocol drifted")
    if rows_with_capture == 0:
        modern_fields = (
            "response_capture",
            "response_boundary_adapter",
            "generated_token_ids_retained",
            "ms_swift_version",
            "non_thinking_prefix_sha256",
        )
        if recipe not in LEGACY_FALLBACK_RECIPES or any(
            source_protocol.get(field) is not None for field in modern_fields
        ):
            raise Day19Qwen35RescoreError(
                "prefix fallback is restricted to explicitly tokenless legacy A/B/E"
            )
    return recipe, verified_records


def build_sidecar_rows(
    *,
    source_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    recipe: str,
    source_prediction_sha256: str,
    source_summary_sha256: str,
    scorers: ModuleType,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_row, record in zip(source_rows, records):
        raw_output = source_row["raw_output"]
        adapted = adapt_qwen35_text(
            raw_output,
            generated_only_text=_generated_only_text(source_row, record["sample_id"]),
        )
        final_text = adapted["final_text"]
        normalized_score = normalized_json(
            scorers.score_prediction(record["slice"], final_text, record["reference"])
        )
        code_candidate = None
        code_static = None
        sandbox_execution_eligible = None
        anomalies: list[str] = []
        if adapted["suspicious_thinking_prefix"]:
            anomalies.append("unexpected_generated_thinking_prefix")
        if record["slice"] == "code":
            code_candidate = classify_code_candidate(
                final_text,
                entry_point=record["metadata"]["entry_point"],
                extract_code_completion=scorers.extract_code_completion,
            )
            if normalized_score.get("parsed_answer") != code_candidate["candidate"]:
                raise Day19Qwen35RescoreError(
                    f"normalized code extractor mismatch: {record['sample_id']}"
                )
            code_static = _code_static_status(
                record.get("raw_prompt", ""),
                record["metadata"]["entry_point"],
                code_candidate,
            )
            sandbox_execution_eligible = (
                code_candidate["execution_eligible"] and code_static["valid"]
            )
            anomalies.extend(code_candidate["anomalies"])
            if not code_static["valid"]:
                anomalies.append("normalized_code_syntax_invalid")
        format_compliant = (
            bool(final_text.strip())
            and normalized_score.get("parse_status") == "ok"
            and (
                sandbox_execution_eligible is None or sandbox_execution_eligible
            )
            and (code_static is None or code_static["valid"])
        )
        row = {
            "schema_version": 2,
            "domain": "day19.qwen35_adapter_prediction",
            "ordinal": source_row["ordinal"],
            "recipe": recipe,
            "sample_id": record["sample_id"],
            "slice": record["slice"],
            "reference_hash": record["reference_hash"],
            "prompt_messages_hash": record["messages_hash"],
            "raw_output": raw_output,
            "raw_output_sha256": source_row["raw_output_sha256"],
            "source_prediction": {
                "file_sha256": source_prediction_sha256,
                "row_schema_version": source_row["schema_version"],
                "legacy_scorer_result": source_row["scorer_result"],
                "legacy_format_compliant": source_row.get("format_compliant"),
                "legacy_anomalies": source_row.get("anomalies", []),
            },
            "source_evaluation_summary_sha256": source_summary_sha256,
            "response_adapter": adapted,
            "normalized_output": final_text,
            "normalized_output_sha256": adapted["final_text_sha256"],
            "normalized_scorer_result": normalized_score,
            "code_candidate": code_candidate,
            "code_static_syntax": code_static,
            "sandbox_execution_eligible": sandbox_execution_eligible,
            "format_compliant": format_compliant,
            "anomalies": list(dict.fromkeys(anomalies)),
            "generation": source_row.get("generation"),
            "response_capture": source_row.get("response_capture"),
        }
        row["row_sha256"] = object_sha256(row)
        result.append(row)
    return result


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_slice: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["slice"] == skill]
        scores = [
            row["normalized_scorer_result"].get("score") for row in skill_rows
        ]
        numeric_scores = [
            float(score)
            for score in scores
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        ]
        entry: dict[str, Any] = {
            "records": len(skill_rows),
            "scored_records": len(numeric_scores),
            "correct": sum(numeric_scores) if numeric_scores else None,
            "accuracy": (
                sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
            ),
            "format_compliant": sum(bool(row["format_compliant"]) for row in skill_rows),
            "format_compliance_rate": sum(
                bool(row["format_compliant"]) for row in skill_rows
            )
            / len(skill_rows),
            "anomalous_records": sum(bool(row["anomalies"]) for row in skill_rows),
        }
        if skill == "code":
            entry.update(
                {
                    "syntax_valid": sum(
                        bool(row["code_static_syntax"]["valid"]) for row in skill_rows
                    ),
                    "sandbox_execution_eligible": sum(
                        bool(row["sandbox_execution_eligible"])
                        for row in skill_rows
                    ),
                    "candidate_modes": dict(
                        sorted(
                            Counter(
                                row["code_candidate"]["candidate_mode"]
                                for row in skill_rows
                            ).items()
                        )
                    ),
                    "executable_score_status": "sandbox_required",
                }
            )
        by_slice[skill] = entry
    non_code = [row for row in rows if row["slice"] != "code"]
    non_code_correct = sum(
        float(row["normalized_scorer_result"]["score"]) for row in non_code
    )
    return {
        "by_slice": by_slice,
        "non_code_correct": non_code_correct,
        "non_code_total": len(non_code),
        "non_code_accuracy": non_code_correct / len(non_code),
        "format_compliant": sum(bool(row["format_compliant"]) for row in rows),
        "format_compliance_rate": sum(bool(row["format_compliant"]) for row in rows)
        / len(rows),
        "anomalous_records": sum(bool(row["anomalies"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scorers", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()
    resolved_inputs = {
        args.manifest.resolve(),
        args.scorers.resolve(),
        args.predictions.resolve(),
        args.eval_summary.resolve(),
    }
    if args.output.resolve() in resolved_inputs or args.summary_output.resolve() in resolved_inputs:
        raise Day19Qwen35RescoreError("sidecar output must not replace an input artifact")
    if args.output.resolve() == args.summary_output.resolve():
        raise Day19Qwen35RescoreError("prediction and summary sidecars must be distinct")
    if args.output.exists() or args.summary_output.exists():
        raise Day19Qwen35RescoreError("refusing to overwrite an existing v2 sidecar")

    manifest = load_json(args.manifest)
    summary = load_json(args.eval_summary)
    source_rows = load_jsonl(args.predictions)
    scorers = load_module(args.scorers.resolve(), "day19_qwen35_v2_frozen_scorers")
    if not callable(getattr(scorers, "score_prediction", None)) or not callable(
        getattr(scorers, "extract_code_completion", None)
    ):
        raise Day19Qwen35RescoreError("frozen scorer API is incomplete")
    recipe, records = verify_source_predictions(
        rows=source_rows,
        summary=summary,
        manifest=manifest,
        scorers=scorers,
        predictions_path=args.predictions,
        manifest_path=args.manifest,
        scorers_path=args.scorers,
    )
    source_prediction_sha = file_sha256(args.predictions)
    source_summary_sha = file_sha256(args.eval_summary)
    rows = build_sidecar_rows(
        source_rows=source_rows,
        records=records,
        recipe=recipe,
        source_prediction_sha256=source_prediction_sha,
        source_summary_sha256=source_summary_sha,
        scorers=scorers,
    )
    manifest_file_sha = file_sha256(args.manifest)
    scorer_file_sha = file_sha256(args.scorers)
    comparison_key = adapter_comparison_key(
        manifest_file_sha256=manifest_file_sha,
        scorer_file_sha256=scorer_file_sha,
    )
    run_hash = adapter_run_hash(
        recipe=recipe,
        comparison_key=comparison_key,
        checkpoint=summary["checkpoint"],
        model_snapshot_sha256=summary["model_export"]["snapshot_sha256"],
        source_prediction_file_sha256=source_prediction_sha,
        source_evaluation_summary_file_sha256=source_summary_sha,
        source_evaluation_summary_content_sha256=summary["summary_sha256"],
    )
    for row in rows:
        row["comparison_key"] = comparison_key
        row["run_hash"] = run_hash
        row["row_sha256"] = object_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    write_jsonl_new(args.output, rows)
    metrics = metrics_for_rows(rows)
    if not math.isfinite(float(metrics["non_code_accuracy"])):
        raise Day19Qwen35RescoreError("non-finite normalized metric")
    sidecar_summary = {
        "schema_version": 2,
        "domain": "day19.qwen35_adapter_summary",
        "status": "complete_with_code_sandbox_required",
        "created_at_utc": utc_now(),
        "recipe": recipe,
        "checkpoint": summary.get("checkpoint"),
        "model_export": summary.get("model_export"),
        "comparison_key": comparison_key,
        "run_hash": run_hash,
        "protocol": {
            "response_adapter_version": ADAPTER_VERSION,
            "response_adapter_path": str(RESPONSE_ADAPTER_PATH),
            "response_adapter_file_sha256": file_sha256(RESPONSE_ADAPTER_PATH),
            "code_parser_version": PARSER_VERSION,
            "code_composer_version": COMPOSER_VERSION,
            "code_adapter_path": str(CODE_ADAPTER_PATH),
            "code_adapter_file_sha256": file_sha256(CODE_ADAPTER_PATH),
            "rescorer_path": str(RESCORER_PATH),
            "rescorer_file_sha256": file_sha256(RESCORER_PATH),
            "legacy_scorer_semantics_preserved": True,
            "candidate_code_executed_on_host": False,
            "frozen_test_consumed": False,
            "source_prediction_path": str(args.predictions.resolve()),
            "source_prediction_file_sha256": source_prediction_sha,
            "source_evaluation_summary_path": str(args.eval_summary.resolve()),
            "source_evaluation_summary_sha256": source_summary_sha,
            "source_evaluation_summary_content_sha256": summary["summary_sha256"],
            "source_evaluation_protocol": {
                key: summary["protocol"].get(key)
                for key in (
                    "template",
                    "enable_thinking",
                    "add_non_thinking_prefix",
                    "response_capture",
                    "response_boundary_adapter",
                    "generated_token_ids_retained",
                    "ms_swift_version",
                    "non_thinking_prefix_sha256",
                )
            },
            "manifest_path": str(args.manifest.resolve()),
            "manifest_file_sha256": manifest_file_sha,
            "scorer_path": str(args.scorers.resolve()),
            "scorer_file_sha256": scorer_file_sha,
        },
        "metrics": metrics,
        "predictions": {
            "path": str(args.output.resolve()),
            "records": len(rows),
            "file_sha256": file_sha256(args.output),
        },
    }
    sidecar_summary["summary_sha256"] = object_sha256(sidecar_summary)
    write_json_new(args.summary_output, sidecar_summary)
    print(json.dumps(sidecar_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
