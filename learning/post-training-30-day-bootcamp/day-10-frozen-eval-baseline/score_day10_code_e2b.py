#!/usr/bin/env python3
"""Score the frozen Day 10 HumanEval development slice in fresh E2B sandboxes.

The script treats the frozen manifest, Base predictions, and pinned HumanEval
source as read-only inputs.  Candidate code is only ever sent to a newly
created, network-denied sandbox and is never executed on the host.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import rfc8785


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "day10_e2b_sandbox_config.json"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-frozen-eval-manifest.json"
)
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-qwen3-0.6b-base-predictions.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl"
)
SCORER_PATH = HERE / "day10_scorers.py"
ANALYZER_PATH = HERE / "analyze_day10_baseline.py"
EXPECTED_SDK_PACKAGE = "e2b"
EXPECTED_SDK_VERSION = "2.37.0"
EXPECTED_TEMPLATE_ID = "rki5dems9wqfm4r03t7g"
EXPECTED_DENY_OUT = ["0.0.0.0/0"]
CANDIDATE_LIMIT_RESULTS = {
    124: ("timeout", "wall_timeout"),
    137: ("failed", "resource_limit"),
    152: ("timeout", "cpu_timeout"),
    153: ("failed", "file_size_limit"),
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Day10CodeScorerError(ValueError):
    """A frozen input, sandbox contract, or publication invariant failed."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Day10CodeScorerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Day10CodeScorerError(
                    f"non-finite JSON value in {context}: {value}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise Day10CodeScorerError(f"invalid JSON in {context}: {error}") from error


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.resolve().read_bytes()
    except FileNotFoundError as error:
        raise Day10CodeScorerError(f"{label} is missing: {path}") from error
    try:
        value = _parse_json(payload.decode("utf-8"), str(path))
    except UnicodeDecodeError as error:
        raise Day10CodeScorerError(f"{label} is not UTF-8: {path}") from error
    if not isinstance(value, dict):
        raise Day10CodeScorerError(f"{label} root must be an object")
    return value, payload


def load_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.resolve().read_bytes()
    except FileNotFoundError as error:
        raise Day10CodeScorerError(f"{label} is missing: {path}") from error
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Day10CodeScorerError(f"{label} is not UTF-8: {path}") from error

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise Day10CodeScorerError(f"blank JSONL row at {path}:{line_number}")
        row = _parse_json(line, f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise Day10CodeScorerError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise Day10CodeScorerError(f"{label} is empty: {path}")
    return rows, payload


def canonical_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError) as error:
        raise Day10CodeScorerError(
            f"RFC 8785 canonicalization failed: {error}"
        ) from error


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_text_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_object(mapping: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise Day10CodeScorerError(f"{context}.{key} must be an object")
    return value


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise Day10CodeScorerError(f"{context}.{key} must be a non-empty string")
    return value


def _require_positive_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Day10CodeScorerError(f"{context}.{key} must be a positive integer")
    return value


def verify_config(config: dict[str, Any]) -> None:
    if config.get("domain") != "day10.humaneval_e2b_sandbox_contract":
        raise Day10CodeScorerError("unexpected sandbox contract domain")
    if config.get("schema_version") != 1:
        raise Day10CodeScorerError("unsupported sandbox contract schema")
    evaluator_source_sha256 = config.get("evaluator_source_sha256")
    if (
        not isinstance(evaluator_source_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", evaluator_source_sha256)
    ):
        raise Day10CodeScorerError("evaluator source SHA-256 is not frozen")
    if bytes_sha256(Path(__file__).resolve().read_bytes()) != evaluator_source_sha256:
        raise Day10CodeScorerError("evaluator source SHA-256 mismatch")
    if (
        config.get("backend") != "e2b_firecracker"
        or config.get("template_id") != EXPECTED_TEMPLATE_ID
    ):
        raise Day10CodeScorerError("sandbox backend/template ID is not frozen")
    if config.get("template_runtime") != {
        "envd_version": "0.6.10",
        "memory_mib": 512,
        "python_version": "3.11.6",
        "vcpus": 2,
    }:
        raise Day10CodeScorerError("sandbox template runtime is not frozen")
    if config.get("expected_dev_code_records") != 28:
        raise Day10CodeScorerError("sandbox contract must require 28 dev code records")
    if config.get("composition") != (
        "source_prompt + parsed_completion + newline + source_test + newline + "
        "check(entry_point)"
    ):
        raise Day10CodeScorerError("HumanEval composition is not frozen")

    sdk = _require_object(config, "sdk", "contract")
    if sdk != {"package": EXPECTED_SDK_PACKAGE, "version": EXPECTED_SDK_VERSION}:
        raise Day10CodeScorerError(
            f"sandbox SDK must be {EXPECTED_SDK_PACKAGE}=={EXPECTED_SDK_VERSION}"
        )

    network = _require_object(config, "network", "contract")
    if network != {
        "allow_internet_access": False,
        "allow_public_traffic": False,
        "deny_out": EXPECTED_DENY_OUT,
        "secure": True,
    }:
        raise Day10CodeScorerError("sandbox network deny policy is not frozen")

    isolation = _require_object(config, "isolation", "contract")
    if isolation != {
        "envs": {},
        "fresh_sandbox_per_sample": True,
        "mounts": [],
        "mcp_servers": [],
    }:
        raise Day10CodeScorerError("sandbox isolation policy is not frozen")

    lifecycle = _require_object(config, "lifecycle", "contract")
    if lifecycle.get("on_timeout") != "kill" or lifecycle.get("auto_resume") is not False:
        raise Day10CodeScorerError("sandbox lifecycle policy is not frozen")
    _require_positive_int(lifecycle, "sandbox_timeout_seconds", "lifecycle")

    execution = _require_object(config, "execution", "contract")
    if execution.get("python_command") != "python3":
        raise Day10CodeScorerError("sandbox Python command is not frozen")
    if execution.get("runner_path") != "/home/user/day10_humaneval_runner.py":
        raise Day10CodeScorerError("sandbox runner path is not frozen")
    if execution.get("result_path") != "/home/user/day10_humaneval_result.json":
        raise Day10CodeScorerError("sandbox result path is not frozen")
    for key in (
        "kill_after_seconds",
        "max_captured_output_bytes",
        "max_result_bytes",
        "candidate_timeout_seconds",
        "sdk_outer_timeout_seconds",
    ):
        _require_positive_int(execution, key, "execution")
    if execution["sdk_outer_timeout_seconds"] <= execution["candidate_timeout_seconds"]:
        raise Day10CodeScorerError(
            "SDK outer timeout must exceed the candidate GNU timeout"
        )
    if execution.get("timeout_semantics") != {
        "candidate_gnu_timeout_exit_124": "valid_score_zero",
        "sdk_timeout_exception": "infrastructure_error_null",
    }:
        raise Day10CodeScorerError("timeout classification policy is not frozen")

    limits = _require_object(config, "resource_limits", "contract")
    for key in (
        "address_space_bytes",
        "cpu_seconds",
        "file_size_bytes",
        "open_files",
        "processes",
    ):
        _require_positive_int(limits, key, "resource_limits")
    if limits.get("core_bytes") != 0:
        raise Day10CodeScorerError("core dump limit must be zero")

    source = _require_object(config, "source", "contract")
    expected_source = {
        "file": "tmp/day09-eval-sources/human-eval/data/HumanEval.jsonl.gz",
        "file_sha256": (
            "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
        ),
        "revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
        "source": "openai/human-eval",
        "split": "test",
    }
    if source != expected_source:
        raise Day10CodeScorerError("pinned HumanEval source contract changed")


def verify_manifest(
    manifest: dict[str, Any], manifest_payload: bytes, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    header = manifest.get("header")
    records = manifest.get("records")
    if not isinstance(header, dict) or not isinstance(records, list):
        raise Day10CodeScorerError("manifest must contain header and records")
    if header.get("domain") != "day10.frozen_eval_manifest":
        raise Day10CodeScorerError("unexpected manifest domain")
    if header.get("schema_version") != 1:
        raise Day10CodeScorerError("unsupported manifest schema")
    if header.get("status") != "frozen_manifest_pre_baseline":
        raise Day10CodeScorerError("manifest is not frozen for Base evaluation")
    expected_manifest_hash = _require_string(header, "manifest_hash", "header")
    candidate = copy.deepcopy(manifest)
    candidate["header"].pop("manifest_hash", None)
    if semantic_hash(candidate) != expected_manifest_hash:
        raise Day10CodeScorerError("manifest semantic hash mismatch")

    source_contract = config["source"]
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise Day10CodeScorerError("manifest record is not an object")
        sample_id = _require_string(record, "sample_id", "manifest record")
        if sample_id in by_id:
            raise Day10CodeScorerError(f"duplicate manifest sample ID: {sample_id}")
        by_id[sample_id] = record
        if record.get("slice") != "code":
            continue
        if record.get("raw_prompt_hash") != exact_text_hash(
            _require_string(record, "raw_prompt", sample_id)
        ):
            raise Day10CodeScorerError(f"raw prompt hash mismatch: {sample_id}")
        if record.get("reference_hash") != exact_text_hash(
            _require_string(record, "reference", sample_id)
        ):
            raise Day10CodeScorerError(f"reference hash mismatch: {sample_id}")
        lineage = _require_object(record, "source_lineage", sample_id)
        if (
            lineage.get("source") != source_contract["source"]
            or lineage.get("revision") != source_contract["revision"]
            or lineage.get("source_split") != source_contract["split"]
            or lineage.get("source_file") != source_contract["file"]
            or lineage.get("source_file_sha256") != source_contract["file_sha256"]
        ):
            raise Day10CodeScorerError(f"HumanEval lineage mismatch: {sample_id}")
        metadata = _require_object(record, "metadata", sample_id)
        entry_point = _require_string(metadata, "entry_point", sample_id)
        if not IDENTIFIER_RE.fullmatch(entry_point):
            raise Day10CodeScorerError(f"invalid HumanEval entry point: {sample_id}")
        test_hash = _require_string(metadata, "test_sha256", sample_id)
        if not re.fullmatch(r"[0-9a-f]{64}", test_hash):
            raise Day10CodeScorerError(f"invalid HumanEval test hash: {sample_id}")

    orders = header.get("evaluation_order")
    if not isinstance(orders, dict) or not isinstance(orders.get("dev"), list):
        raise Day10CodeScorerError("manifest dev evaluation order is missing")
    dev_order = orders["dev"]
    if len(dev_order) != len(set(dev_order)):
        raise Day10CodeScorerError("manifest dev order contains duplicate IDs")
    if any(sample_id not in by_id for sample_id in dev_order):
        raise Day10CodeScorerError("manifest dev order references an unknown sample")
    code_order = [
        sample_id for sample_id in dev_order if by_id[sample_id].get("slice") == "code"
    ]
    if len(code_order) != config["expected_dev_code_records"]:
        raise Day10CodeScorerError("manifest does not contain exactly 28 dev code records")
    if any(by_id[sample_id].get("evaluation_split") != "dev" for sample_id in code_order):
        raise Day10CodeScorerError("manifest code order includes a non-dev record")

    if any(
        by_id[sample_id]["source_lineage"].get("source_file_sha256")
        != source_contract["file_sha256"]
        for sample_id in code_order
    ):
        raise Day10CodeScorerError("dev code records do not share the pinned source")
    _require_string(header, "scorer_source_sha256", "header")
    _require_string(header, "scorer_registry_hash", "header")
    return by_id, code_order


def load_frozen_scorers(manifest_header: dict[str, Any]) -> Any:
    expected_hash = _require_string(
        manifest_header, "scorer_source_sha256", "manifest header"
    )
    try:
        source_payload = SCORER_PATH.read_bytes()
    except FileNotFoundError as error:
        raise Day10CodeScorerError(f"frozen scorer is missing: {SCORER_PATH}") from error
    if bytes_sha256(source_payload) != expected_hash:
        raise Day10CodeScorerError("frozen scorer source hash mismatch")
    spec = importlib.util.spec_from_file_location("day10_code_frozen_scorers", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise Day10CodeScorerError("cannot load the frozen scorer module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day10CodeScorerError(
            f"frozen scorer import failed: {type(error).__name__}"
        ) from error
    if not callable(getattr(module, "extract_code_completion", None)):
        raise Day10CodeScorerError("frozen code extractor is missing")
    return module


def verify_complete_base_run(manifest_path: Path, predictions_path: Path) -> None:
    """Delegate full 112-row run validation to the existing strict analyzer."""

    spec = importlib.util.spec_from_file_location(
        "day10_baseline_analyzer_for_code", ANALYZER_PATH
    )
    if spec is None or spec.loader is None:
        raise Day10CodeScorerError("cannot load the frozen Base-run analyzer")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        summary = module.analyze_baseline(manifest_path, predictions_path)
    except Exception as error:
        raise Day10CodeScorerError(
            f"complete Base-run validation failed: {type(error).__name__}: {error}"
        ) from error
    per_slice = summary.get("per_slice") if isinstance(summary, dict) else None
    code = per_slice.get("code") if isinstance(per_slice, dict) else None
    if not isinstance(code, dict) or code.get("sandbox_pending") != 28:
        raise Day10CodeScorerError(
            "Base-run analyzer did not confirm 28 sandbox-pending code rows"
        )


def load_humaneval_source(
    config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], str]:
    source_spec = config["source"]
    source_path = (REPO_ROOT / source_spec["file"]).resolve()
    try:
        source_path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise Day10CodeScorerError("HumanEval source escapes the repository") from error
    try:
        payload = source_path.read_bytes()
    except FileNotFoundError as error:
        raise Day10CodeScorerError(
            f"pinned HumanEval source is missing: {source_path}"
        ) from error
    actual_sha = bytes_sha256(payload)
    if actual_sha != source_spec["file_sha256"]:
        raise Day10CodeScorerError("pinned HumanEval source file hash mismatch")

    rows: dict[str, dict[str, Any]] = {}
    try:
        with gzip.open(source_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day10CodeScorerError(
                        f"blank HumanEval source row: {line_number}"
                    )
                row = _parse_json(line, f"{source_path}:{line_number}")
                if not isinstance(row, dict):
                    raise Day10CodeScorerError(
                        f"HumanEval row is not an object: {line_number}"
                    )
                task_id = _require_string(row, "task_id", f"HumanEval:{line_number}")
                if task_id in rows:
                    raise Day10CodeScorerError(
                        f"duplicate HumanEval task ID: {task_id}"
                    )
                for key in ("prompt", "canonical_solution", "entry_point", "test"):
                    _require_string(row, key, task_id)
                if not row["prompt"].endswith("\n"):
                    raise Day10CodeScorerError(
                        f"HumanEval prompt lacks its canonical newline: {task_id}"
                    )
                if not IDENTIFIER_RE.fullmatch(row["entry_point"]):
                    raise Day10CodeScorerError(
                        f"invalid source entry point: {task_id}"
                    )
                rows[task_id] = row
    except (OSError, UnicodeDecodeError) as error:
        raise Day10CodeScorerError("cannot decode pinned HumanEval source") from error
    return rows, actual_sha


def verify_predictions_and_build_tasks(
    *,
    manifest: dict[str, Any],
    manifest_payload: bytes,
    manifest_by_id: dict[str, dict[str, Any]],
    code_order: list[str],
    predictions: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    frozen_scorers: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    header = manifest["header"]
    dev_order = header["evaluation_order"]["dev"]
    if len(predictions) != len(dev_order):
        raise Day10CodeScorerError("predictions do not cover the complete dev order")
    manifest_file_sha = bytes_sha256(manifest_payload)

    run_identity_fields = (
        "run_hash",
        "comparison_key",
        "model_id",
        "model_revision",
        "model_snapshot_hash",
    )
    uniform: dict[str, str] = {}
    code_predictions: dict[str, dict[str, Any]] = {}
    for ordinal, (row, expected_id) in enumerate(zip(predictions, dev_order)):
        sample_id = _require_string(row, "sample_id", f"prediction:{ordinal}")
        if sample_id != expected_id or row.get("run_ordinal") != ordinal:
            raise Day10CodeScorerError(
                f"prediction order/ordinal mismatch at row {ordinal}"
            )
        manifest_record = manifest_by_id[sample_id]
        if row.get("schema_version") != 1:
            raise Day10CodeScorerError(f"unsupported prediction schema: {sample_id}")
        if row.get("evaluation_split") != "dev":
            raise Day10CodeScorerError(f"prediction is not dev-only: {sample_id}")
        if row.get("slice") != manifest_record.get("slice"):
            raise Day10CodeScorerError(f"prediction slice mismatch: {sample_id}")
        for field in (
            "rendered_prompt_hash",
            "input_ids_hash",
            "reference_hash",
            "input_token_count",
            "extractor_version",
            "scorer_version",
        ):
            if row.get(field) != manifest_record.get(field):
                raise Day10CodeScorerError(
                    f"prediction {field} differs from manifest: {sample_id}"
                )
        if row.get("manifest_hash") != header["manifest_hash"]:
            raise Day10CodeScorerError(f"prediction manifest hash mismatch: {sample_id}")
        if row.get("manifest_file_sha256") != manifest_file_sha:
            raise Day10CodeScorerError(
                f"prediction manifest file hash mismatch: {sample_id}"
            )
        raw_output = row.get("raw_output")
        if not isinstance(raw_output, str):
            raise Day10CodeScorerError(f"prediction output is not text: {sample_id}")
        if row.get("raw_output_hash") != exact_text_hash(raw_output):
            raise Day10CodeScorerError(f"prediction output hash mismatch: {sample_id}")
        output_token_ids = row.get("output_token_ids")
        if not isinstance(output_token_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in output_token_ids
        ):
            raise Day10CodeScorerError(f"invalid output token IDs: {sample_id}")
        expected_token_hash = semantic_hash(
            {
                "domain": "day10.output_token_ids",
                "schema_version": 1,
                "output_token_ids": output_token_ids,
            }
        )
        if row.get("output_token_ids_hash") != expected_token_hash:
            raise Day10CodeScorerError(f"output token hash mismatch: {sample_id}")
        if row.get("output_token_count") != len(output_token_ids):
            raise Day10CodeScorerError(f"output token count mismatch: {sample_id}")
        generation_limit = manifest_record.get("generation_max_new_tokens")
        if (
            isinstance(generation_limit, bool)
            or not isinstance(generation_limit, int)
            or len(output_token_ids) > generation_limit
        ):
            raise Day10CodeScorerError(f"output exceeds generation limit: {sample_id}")
        if row.get("total_token_count") != (
            manifest_record["input_token_count"] + len(output_token_ids)
        ):
            raise Day10CodeScorerError(f"total token count mismatch: {sample_id}")
        latency = row.get("latency_seconds")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or latency <= 0
        ):
            raise Day10CodeScorerError(f"invalid generation latency: {sample_id}")
        for field in run_identity_fields:
            value = _require_string(row, field, sample_id)
            if field in uniform and uniform[field] != value:
                raise Day10CodeScorerError(f"prediction {field} is not uniform")
            uniform[field] = value
        if sample_id in code_order:
            code_predictions[sample_id] = row

    tasks: list[dict[str, Any]] = []
    for sample_id in code_order:
        record = manifest_by_id[sample_id]
        prediction = code_predictions.get(sample_id)
        if prediction is None:
            raise Day10CodeScorerError(f"missing code prediction: {sample_id}")
        if prediction.get("scorer_version") != record.get("scorer_version"):
            raise Day10CodeScorerError(f"code scorer version mismatch: {sample_id}")
        if prediction.get("extractor_version") != record.get("extractor_version"):
            raise Day10CodeScorerError(f"code extractor version mismatch: {sample_id}")
        scorer_result = prediction.get("scorer_result")
        if not isinstance(scorer_result, dict):
            raise Day10CodeScorerError(f"code scorer handoff is missing: {sample_id}")
        completion = frozen_scorers.extract_code_completion(prediction["raw_output"])
        if (
            scorer_result.get("parsed_answer") != completion
            or scorer_result.get("score_status") != "sandbox_required"
            or scorer_result.get("score") is not None
            or scorer_result.get("error_type") != "sandbox_required"
        ):
            raise Day10CodeScorerError(f"code scorer handoff mismatch: {sample_id}")

        lineage = record["source_lineage"]
        task_id = _require_string(lineage, "parent_id", sample_id)
        source = source_rows.get(task_id)
        if source is None:
            raise Day10CodeScorerError(f"HumanEval source task is missing: {task_id}")
        metadata = record["metadata"]
        if source["prompt"].strip() != record["raw_prompt"]:
            raise Day10CodeScorerError(f"source prompt mismatch: {sample_id}")
        if source["canonical_solution"].strip() != record["reference"]:
            raise Day10CodeScorerError(f"source reference mismatch: {sample_id}")
        if source["entry_point"] != metadata["entry_point"]:
            raise Day10CodeScorerError(f"source entry point mismatch: {sample_id}")
        if object_sha256(source["test"]) != metadata["test_sha256"]:
            raise Day10CodeScorerError(f"source test hash mismatch: {sample_id}")
        tasks.append(
            {
                "code_ordinal": len(tasks),
                "sample_id": sample_id,
                "task_id": task_id,
                "entry_point": source["entry_point"],
                "test_sha256": metadata["test_sha256"],
                "source_prompt": source["prompt"],
                "source_test": source["test"],
                "completion": completion,
                "manifest_record": record,
                "prediction": prediction,
            }
        )
    return tasks, uniform


def prepare_tasks(
    manifest_path: Path = DEFAULT_MANIFEST,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config, config_payload = load_json(config_path, "sandbox contract")
    verify_config(config)
    verify_complete_base_run(manifest_path, predictions_path)
    manifest, manifest_payload = load_json(manifest_path, "manifest")
    manifest_by_id, code_order = verify_manifest(manifest, manifest_payload, config)
    predictions, predictions_payload = load_jsonl(predictions_path, "predictions")
    source_rows, source_file_sha = load_humaneval_source(config)
    frozen_scorers = load_frozen_scorers(manifest["header"])
    tasks, uniform = verify_predictions_and_build_tasks(
        manifest=manifest,
        manifest_payload=manifest_payload,
        manifest_by_id=manifest_by_id,
        code_order=code_order,
        predictions=predictions,
        source_rows=source_rows,
        frozen_scorers=frozen_scorers,
    )
    sandbox_contract_hash = semantic_hash(config)
    complete_comparison_key = semantic_hash(
        {
            "domain": "day10.complete_checkpoint_comparison",
            "schema_version": 1,
            "prediction_comparison_key": uniform["comparison_key"],
            "code_execution_protocol_hash": sandbox_contract_hash,
        }
    )
    predictions_file_sha256 = bytes_sha256(predictions_payload)
    code_run_hash = semantic_hash(
        {
            "domain": "day10.code_sandbox_run",
            "schema_version": 1,
            "manifest_hash": manifest["header"]["manifest_hash"],
            "manifest_file_sha256": bytes_sha256(manifest_payload),
            "predictions_file_sha256": predictions_file_sha256,
            "prediction_run_hash": uniform["run_hash"],
            "code_execution_protocol_hash": sandbox_contract_hash,
            "sample_ids": [task["sample_id"] for task in tasks],
        }
    )
    context = {
        "config": config,
        "sandbox_contract_hash": sandbox_contract_hash,
        "sandbox_contract_file_sha256": bytes_sha256(config_payload),
        "evaluator_source_sha256": config["evaluator_source_sha256"],
        "manifest_hash": manifest["header"]["manifest_hash"],
        "manifest_file_sha256": bytes_sha256(manifest_payload),
        "predictions_file_sha256": predictions_file_sha256,
        "source_file_sha256": source_file_sha,
        "complete_comparison_key": complete_comparison_key,
        "code_run_hash": code_run_hash,
        **uniform,
    }
    return tasks, context


def compose_humaneval_program(task: dict[str, Any]) -> str:
    """Apply the canonical HumanEval prompt/completion/test/check composition."""

    prompt = task["source_prompt"]
    completion = task["completion"]
    test = task["source_test"]
    entry_point = task["entry_point"]
    if not prompt.endswith("\n") or not IDENTIFIER_RE.fullmatch(entry_point):
        raise Day10CodeScorerError("invalid task supplied to HumanEval composition")
    return prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"


def build_runner_source(task: dict[str, Any], config: dict[str, Any]) -> str:
    """Build a controlled runner; candidate bytes never enter the shell command."""

    program_b64 = base64.b64encode(
        compose_humaneval_program(task).encode("utf-8")
    ).decode("ascii")
    execution = config["execution"]
    limits = config["resource_limits"]
    result_path_literal = json.dumps(execution["result_path"])
    return f'''#!/usr/bin/env python3
import base64
import io
import json
import os
import resource
import sys

PROGRAM_B64 = {json.dumps(program_b64)}
RESULT_PATH = {result_path_literal}
MAX_OUTPUT_BYTES = {execution["max_captured_output_bytes"]}


class BoundedText(io.TextIOBase):
    encoding = "utf-8"

    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.size = 0

    def writable(self):
        return True

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        encoded = value.encode("utf-8", "replace")
        remaining = max(0, self.limit - self.size)
        if remaining:
            chunk = encoded[:remaining]
            self.parts.append(chunk.decode("utf-8", "ignore"))
            self.size += len(chunk)
        return len(value)

    def flush(self):
        return None

    def value(self):
        return "".join(self.parts)


def write_result(payload):
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    descriptor = os.open(RESULT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def set_limits():
    resource.setrlimit(resource.RLIMIT_CORE, ({limits["core_bytes"]}, {limits["core_bytes"]}))
    resource.setrlimit(resource.RLIMIT_CPU, ({limits["cpu_seconds"]}, {limits["cpu_seconds"] + 1}))
    resource.setrlimit(
        resource.RLIMIT_AS,
        ({limits["address_space_bytes"]}, {limits["address_space_bytes"]}),
    )
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        ({limits["file_size_bytes"]}, {limits["file_size_bytes"]}),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, ({limits["open_files"]}, {limits["open_files"]}))
    resource.setrlimit(resource.RLIMIT_NPROC, ({limits["processes"]}, {limits["processes"]}))


try:
    set_limits()
except BaseException as error:
    write_result({{
        "status": "runner_error",
        "failure_type": type(error).__name__,
        "failure_message": "resource policy setup failed",
        "stdout": "",
        "stderr": "",
    }})
    raise SystemExit(70)

captured_stdout = BoundedText(MAX_OUTPUT_BYTES)
captured_stderr = BoundedText(MAX_OUTPUT_BYTES)
sys.stdout = captured_stdout
sys.stderr = captured_stderr
result = {{
    "status": "passed",
    "failure_type": None,
    "failure_message": None,
}}
try:
    program = base64.b64decode(PROGRAM_B64, validate=True).decode("utf-8")
    namespace = {{"__name__": "__main__"}}
    exec(compile(program, "<day10-humaneval>", "exec"), namespace, namespace)
except BaseException as error:
    if isinstance(error, SyntaxError):
        failure_type = "syntax_error"
    elif isinstance(error, AssertionError):
        failure_type = "assertion_error"
    elif isinstance(error, MemoryError):
        failure_type = "memory_limit"
    else:
        failure_type = "runtime_error"
    try:
        failure_message = str(error)[:4096]
    except BaseException:
        failure_message = "exception stringification failed"
    result.update({{
        "status": "failed",
        "failure_type": failure_type,
        "failure_message": failure_message,
    }})
result["stdout"] = captured_stdout.value()
result["stderr"] = captured_stderr.value()
write_result(result)
'''


def build_command(config: dict[str, Any]) -> str:
    execution = config["execution"]
    return (
        f"timeout -s TERM -k {execution['kill_after_seconds']}s "
        f"{execution['candidate_timeout_seconds']}s {execution['python_command']} "
        f"{execution['runner_path']} >/dev/null 2>/dev/null"
    )


def load_e2b_bindings(config: dict[str, Any]) -> dict[str, Any]:
    sdk = config["sdk"]
    try:
        installed_version = importlib.metadata.version(sdk["package"])
    except importlib.metadata.PackageNotFoundError as error:
        raise Day10CodeScorerError(
            f"required sandbox SDK is not installed: {sdk['package']}"
        ) from error
    if installed_version != sdk["version"]:
        raise Day10CodeScorerError(
            f"sandbox SDK mismatch: expected {sdk['version']}, got {installed_version}"
        )
    try:
        from e2b import CommandExitException, Sandbox, TimeoutException
    except Exception as error:
        raise Day10CodeScorerError(
            f"cannot import pinned E2B SDK: {type(error).__name__}"
        ) from error
    return {
        "create": Sandbox.create,
        "command_exit_exception": CommandExitException,
        "timeout_exception": TimeoutException,
    }


def sandbox_create_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    network = config["network"]
    lifecycle = config["lifecycle"]
    return {
        "template": config["template_id"],
        "timeout": lifecycle["sandbox_timeout_seconds"],
        "secure": network["secure"],
        "allow_internet_access": network["allow_internet_access"],
        "network": {
            "allow_public_traffic": network["allow_public_traffic"],
            "deny_out": list(network["deny_out"]),
        },
        "lifecycle": {
            "on_timeout": lifecycle["on_timeout"],
            "auto_resume": lifecycle["auto_resume"],
        },
        "envs": {},
        "mcp": None,
        "volume_mounts": {},
    }


def sandbox_info_mismatch(info: Any, config: dict[str, Any]) -> str | None:
    """Return the first runtime-attestation mismatch without exposing metadata."""

    runtime = config["template_runtime"]
    expected_attributes = {
        "template_id": config["template_id"],
        "envd_version": runtime["envd_version"],
        "cpu_count": runtime["vcpus"],
        "memory_mb": runtime["memory_mib"],
        "allow_internet_access": False,
        "volume_mounts": [],
    }
    for field, expected in expected_attributes.items():
        if getattr(info, field, None) != expected:
            return field
    network = getattr(info, "network", None)
    if not isinstance(network, dict):
        return "network"
    if network.get("allow_public_traffic") is not False:
        return "network.allow_public_traffic"
    if network.get("deny_out") != EXPECTED_DENY_OUT:
        return "network.deny_out"
    lifecycle = getattr(info, "lifecycle", None)
    if lifecycle != {"on_timeout": "kill", "auto_resume": False}:
        return "lifecycle"
    return None


def _command_exit_code(command_result: Any) -> int | None:
    value = getattr(command_result, "exit_code", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bounded_evidence(value: str, excerpt_bytes: int = 1024) -> dict[str, Any]:
    encoded = value.encode("utf-8", "replace")
    excerpt = encoded[:excerpt_bytes].decode("utf-8", "ignore")
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "excerpt": excerpt,
    }


def _base_result_row(
    task: dict[str, Any], context: dict[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    prediction = task["prediction"]
    return {
        "domain": "day10.code_sandbox_result",
        "schema_version": 1,
        "sample_id": task["sample_id"],
        "code_ordinal": task["code_ordinal"],
        "prediction_run_ordinal": prediction["run_ordinal"],
        "evaluation_split": "dev",
        "slice": "code",
        "task_id": task["task_id"],
        "entry_point": task["entry_point"],
        "test_sha256": task["test_sha256"],
        "manifest_hash": context["manifest_hash"],
        "manifest_file_sha256": context["manifest_file_sha256"],
        "predictions_file_sha256": context["predictions_file_sha256"],
        "run_hash": context["run_hash"],
        "prediction_run_hash": context["run_hash"],
        "comparison_key": context["comparison_key"],
        "prediction_comparison_key": context["comparison_key"],
        "complete_comparison_key": context["complete_comparison_key"],
        "code_run_hash": context["code_run_hash"],
        "model_id": prediction["model_id"],
        "model_revision": prediction["model_revision"],
        "model_snapshot_hash": context["model_snapshot_hash"],
        "extractor_version": prediction["extractor_version"],
        "scorer_version": prediction["scorer_version"],
        "harness_version": "day10_humaneval_e2b_v1",
        "raw_output_hash": prediction["raw_output_hash"],
        "raw_prompt_hash": task["manifest_record"]["raw_prompt_hash"],
        "reference_hash": task["manifest_record"]["reference_hash"],
        "output_token_ids_hash": prediction["output_token_ids_hash"],
        "parsed_completion_hash": exact_text_hash(task["completion"]),
        "completion_hash": exact_text_hash(task["completion"]),
        "composed_program_hash": exact_text_hash(compose_humaneval_program(task)),
        "source_file_sha256": context["source_file_sha256"],
        "source_revision": context["config"]["source"]["revision"],
        "sandbox_contract_hash": context["sandbox_contract_hash"],
        "code_execution_protocol": context["config"],
        "code_execution_protocol_hash": context["sandbox_contract_hash"],
        "sandbox_contract_file_sha256": context[
            "sandbox_contract_file_sha256"
        ],
        "evaluator_source_sha256": context["evaluator_source_sha256"],
        "sandbox": {
            "backend": context["config"]["backend"],
            "sdk": context["config"]["sdk"],
            "template_id": context["config"]["template_id"],
            "template_runtime": context["config"]["template_runtime"],
            "fresh_sandbox_per_sample": True,
            "secure": True,
            "allow_internet_access": False,
            "allow_public_traffic": False,
            "deny_out": list(EXPECTED_DENY_OUT),
        },
        "latency_seconds": elapsed_seconds,
    }


def _finalize_result_row(
    task: dict[str, Any],
    context: dict[str, Any],
    *,
    started: float,
    execution_status: str,
    score_status: str,
    score: float | None,
    error_type: str | None,
    exit_code: int | None,
    stdout: str = "",
    stderr: str = "",
    failure_message: str | None = None,
) -> dict[str, Any]:
    row = _base_result_row(task, context, time.perf_counter() - started)
    stdout_evidence = _bounded_evidence(stdout)
    stderr_evidence = _bounded_evidence(stderr)
    executor_result_sha256 = semantic_hash(
        {
            "domain": "day10.code_executor_result",
            "schema_version": 1,
            "execution_status": execution_status,
            "score_status": score_status,
            "score": score,
            "error_type": error_type,
            "exit_code": exit_code,
            "stdout_sha256": stdout_evidence["sha256"],
            "stderr_sha256": stderr_evidence["sha256"],
            "failure_message": failure_message,
        }
    )
    passed = score == 1.0 if score_status == "ok" else None
    row.update(
        {
            "execution_status": execution_status,
            "score_status": score_status,
            "score": score,
            "passed": passed,
            "error_type": error_type,
            "exit_code": exit_code,
            "stdout": stdout_evidence,
            "stderr": stderr_evidence,
            "failure_message": failure_message,
            "scorer_result": {
                "parse_status": "ok",
                "score_status": score_status,
                "score": score,
                "passed": passed,
                "execution_outcome": execution_status,
                "executor_result_sha256": executor_result_sha256,
                "error_type": error_type,
            },
            "observed": {"elapsed_seconds": row["latency_seconds"]},
        }
    )
    return row


def _invalidate_result_row(row: dict[str, Any], error_type: str) -> None:
    row.update(
        {
            "execution_status": "infrastructure_error",
            "score_status": "infrastructure_error",
            "score": None,
            "passed": None,
            "error_type": error_type,
        }
    )
    scorer_result = row["scorer_result"]
    scorer_result.update(
        {
            "score_status": "infrastructure_error",
            "score": None,
            "passed": None,
            "execution_outcome": "infrastructure_error",
            "error_type": error_type,
        }
    )
    scorer_result["executor_result_sha256"] = semantic_hash(
        {
            "domain": "day10.code_executor_result",
            "schema_version": 1,
            "execution_status": "infrastructure_error",
            "score_status": "infrastructure_error",
            "score": None,
            "error_type": error_type,
            "exit_code": row["exit_code"],
            "stdout_sha256": row["stdout"]["sha256"],
            "stderr_sha256": row["stderr"]["sha256"],
            "failure_message": row["failure_message"],
        }
    )


def execute_task(
    task: dict[str, Any], context: dict[str, Any], bindings: dict[str, Any]
) -> dict[str, Any]:
    """Execute one task in one new sandbox and always attempt to kill it."""

    config = context["config"]
    started = time.perf_counter()
    sandbox = None
    row: dict[str, Any] | None = None
    try:
        try:
            sandbox = bindings["create"](**sandbox_create_kwargs(config))
        except Exception as error:
            return _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type=type(error).__name__,
                exit_code=None,
            )

        try:
            info = sandbox.get_info()
        except Exception as error:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type=f"sandbox_info_{type(error).__name__}",
                exit_code=None,
            )
            return row
        mismatch = sandbox_info_mismatch(info, config)
        if mismatch is not None:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="sandbox_attestation_mismatch",
                exit_code=None,
                failure_message=mismatch,
            )
            return row

        try:
            sandbox.files.write(
                config["execution"]["runner_path"], build_runner_source(task, config)
            )
        except Exception as error:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type=type(error).__name__,
                exit_code=None,
            )
            return row

        try:
            command_result = sandbox.commands.run(
                build_command(config),
                timeout=config["execution"]["sdk_outer_timeout_seconds"],
            )
        except bindings["timeout_exception"]:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="sdk_timeout",
                exit_code=None,
            )
            return row
        except bindings["command_exit_exception"] as error:
            command_result = error
        except Exception as error:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type=type(error).__name__,
                exit_code=None,
            )
            return row

        exit_code = _command_exit_code(command_result)
        if exit_code in CANDIDATE_LIMIT_RESULTS:
            execution_status, error_type = CANDIDATE_LIMIT_RESULTS[exit_code]
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status=execution_status,
                score_status="ok",
                score=0.0,
                error_type=error_type,
                exit_code=exit_code,
            )
            return row

        try:
            result_text = sandbox.files.read(config["execution"]["result_path"])
        except Exception:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="missing_result",
                exit_code=exit_code,
            )
            return row
        if not isinstance(result_text, str):
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="invalid_result_type",
                exit_code=exit_code,
            )
            return row
        if len(result_text.encode("utf-8")) > config["execution"]["max_result_bytes"]:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="oversized_result",
                exit_code=exit_code,
            )
            return row
        try:
            result = _parse_json(result_text, f"sandbox result:{task['sample_id']}")
        except Day10CodeScorerError:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="invalid_result_json",
                exit_code=exit_code,
            )
            return row
        if not isinstance(result, dict):
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="invalid_result_json",
                exit_code=exit_code,
            )
            return row

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="invalid_result_output",
                exit_code=exit_code,
            )
            return row
        result_status = result.get("status")
        if result_status not in {"passed", "failed", "runner_error"}:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="invalid_result_status",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
            return row
        if result_status == "passed" and exit_code == 0:
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="passed",
                score_status="ok",
                score=1.0,
                error_type=None,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
            return row
        if result_status == "passed":
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="inconsistent_pass_result",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
            return row
        if result_status == "runner_error":
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="runner_error",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
            return row
        failure_type = result.get("failure_type")
        failure_message = result.get("failure_message")
        if failure_type not in {
            "syntax_error",
            "assertion_error",
            "memory_limit",
            "runtime_error",
        } or not isinstance(failure_message, str):
            row = _finalize_result_row(
                task,
                context,
                started=started,
                execution_status="infrastructure_error",
                score_status="infrastructure_error",
                score=None,
                error_type="invalid_failure_result",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
            return row
        row = _finalize_result_row(
            task,
            context,
            started=started,
            execution_status="failed",
            score_status="ok",
            score=0.0,
            error_type=failure_type,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            failure_message=(failure_message[:4096] if failure_message else None),
        )
        return row
    finally:
        if sandbox is not None:
            try:
                killed = sandbox.kill()
                if killed is False and row is not None:
                    _invalidate_result_row(row, "sandbox_kill_returned_false")
            except Exception as error:
                if row is not None:
                    _invalidate_result_row(
                        row, f"sandbox_kill_{type(error).__name__}"
                    )


def score_tasks(
    tasks: list[dict[str, Any]],
    context: dict[str, Any],
    bindings: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        row = execute_task(task, context, bindings)
        results.append(row)
        if row["execution_status"] == "infrastructure_error":
            break
    sample_ids = [row["sample_id"] for row in results]
    if sample_ids != [task["sample_id"] for task in tasks[: len(results)]]:
        raise Day10CodeScorerError("sandbox results changed the frozen task order")
    return results


def write_jsonl_atomic(
    rows: list[dict[str, Any]], output_path: Path, *, overwrite: bool
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise Day10CodeScorerError(
            f"output already exists (pass --overwrite to replace it): {output_path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise Day10CodeScorerError(
                    f"output appeared during scoring: {output_path}"
                ) from error
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--sandbox-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise Day10CodeScorerError(
            f"output already exists (pass --overwrite to replace it): {args.output}"
        )
    tasks, context = prepare_tasks(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        config_path=args.sandbox_config,
    )
    bindings = load_e2b_bindings(context["config"])
    results = score_tasks(tasks, context, bindings)
    infrastructure_errors = [
        row["sample_id"]
        for row in results
        if row["execution_status"] == "infrastructure_error"
    ]
    if infrastructure_errors:
        raise Day10CodeScorerError(
            "sandbox infrastructure failed; no sidecar was published for: "
            + ", ".join(infrastructure_errors)
        )
    write_jsonl_atomic(results, args.output, overwrite=args.overwrite)
    summary = {
        "output": str(args.output.resolve()),
        "artifact_sha256": bytes_sha256(args.output.resolve().read_bytes()),
        "records": len(results),
        "passed": sum(row["score"] == 1.0 for row in results),
        "failed": sum(row["score"] == 0.0 for row in results),
        "sandbox_contract_hash": context["sandbox_contract_hash"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Day10CodeScorerError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
