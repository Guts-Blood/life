#!/usr/bin/env python3
"""Execute strictly eligible Day 20 v2 Code continuations in frozen E2B."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import day20_eval_identity_v2 as eval_identity
from day20_contract_v2 import (
    V2_CONTRACT_VERSION,
    Day20V2ContractError,
    validate_raw_code_continuation,
)
from day20_e2b_preflight_v2 import verify_preflight


SCHEMA_VERSION = 2
ROW_DOMAIN = "day20.v2.code_e2b_result"
SUMMARY_DOMAIN = "day20.v2.code_e2b_summary"
WRAPPER_PATH = Path(__file__).resolve()


class Day20CodeE2BV3Error(ValueError):
    """The normalized pair, strict Code task, sandbox, or artifact drifted."""


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20CodeE2BV3Error(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20CodeE2BV3Error(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day20CodeE2BV3Error(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day20CodeE2BV3Error(
                        f"blank JSONL row at line {line_number}: {path}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Day20CodeE2BV3Error(
                        f"JSONL row {line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Day20CodeE2BV3Error(f"cannot load JSONL {path}: {error}") from error
    return rows


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = eval_identity.object_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if not isinstance(expected, str) or expected != actual:
        raise Day20CodeE2BV3Error(f"{field} mismatch")
    return expected


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Day20CodeE2BV3Error(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day20CodeE2BV3Error(
            f"cannot import {path}: {type(error).__name__}"
        ) from error
    return module


def _default_pair_verifier(
    predictions_path: Path,
    summary_path: Path,
    *,
    expected_scope: str | None = None,
    expected_candidate: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        rescorer = importlib.import_module("rescore_day20_qwen35_v3")
    except ImportError as error:
        raise Day20CodeE2BV3Error(
            "rescore_day20_qwen35_v3 verifier is unavailable"
        ) from error
    verifier = getattr(rescorer, "verify_normalized_pair", None)
    if not callable(verifier):
        raise Day20CodeE2BV3Error("normalized pair verifier API is unavailable")
    try:
        return verifier(
            predictions_path,
            summary_path,
            expected_scope=expected_scope,
            expected_candidate=expected_candidate,
        )
    except ValueError as error:
        raise Day20CodeE2BV3Error(str(error)) from error


PairVerifier = Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]


def _strict_code_rows(
    rows: list[dict[str, Any]], *, scope: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code_rows = [row for row in rows if row.get("slice") == "code"]
    expected = eval_identity.EXPECTED_CODE_RECORDS_BY_SCOPE[scope]
    if len(code_rows) != expected:
        raise Day20CodeE2BV3Error(
            f"normalized pair must contain exactly {expected} Code rows"
        )
    eligible: list[dict[str, Any]] = []
    for row in code_rows:
        code_candidate = row.get("code_candidate")
        static = row.get("code_static_syntax")
        is_eligible = row.get("sandbox_execution_eligible")
        if (
            not isinstance(code_candidate, dict)
            or code_candidate.get("candidate_mode") != "completion"
            or not isinstance(code_candidate.get("execution_eligible"), bool)
            or code_candidate.get("contract_version") != V2_CONTRACT_VERSION
            or not isinstance(static, dict)
            or not isinstance(static.get("valid"), bool)
            or not isinstance(is_eligible, bool)
            or is_eligible
            is not (code_candidate["execution_eligible"] and static["valid"])
        ):
            raise Day20CodeE2BV3Error(
                f"strict Code eligibility drifted: {row.get('sample_id')}"
            )
        if is_eligible:
            candidate = code_candidate.get("candidate")
            try:
                evidence = validate_raw_code_continuation(
                    candidate, row.get("raw_prompt")
                )
            except Day20V2ContractError as error:
                raise Day20CodeE2BV3Error(
                    f"eligible Code continuation is invalid: {row.get('sample_id')}"
                ) from error
            expected_evidence = {
                "candidate_mode": "completion",
                "candidate": candidate,
                "execution_eligible": True,
                "contract_version": V2_CONTRACT_VERSION,
                "raw_sha256": evidence.raw_sha256,
                "canonical_sha256": evidence.canonical_sha256,
                "ast_sha256": evidence.ast_sha256,
                "error": None,
            }
            if code_candidate != expected_evidence or static != {
                "valid": True,
                "error": None,
            }:
                raise Day20CodeE2BV3Error(
                    f"eligible Code evidence drifted: {row.get('sample_id')}"
                )
            eligible.append(row)
    return code_rows, eligible


def _preflight_runtime_identity(
    preflight: Mapping[str, Any], preflight_path: Path
) -> dict[str, Any]:
    return {
        "sandbox_python": preflight.get("sandbox_python"),
        "e2b_sdk": preflight.get("e2b_sdk"),
        "preflight_file_sha256": file_sha256(preflight_path),
        "preflight_content_sha256": preflight.get("preflight_sha256"),
    }


def _validate_preflight_bindings(
    preflight: Mapping[str, Any],
    *,
    frozen_scorer_path: Path,
    sandbox_config_path: Path,
    humaneval_source_path: Path,
) -> None:
    expected = {
        "frozen_scorer": (frozen_scorer_path, "file_sha256"),
        "sandbox_config": (sandbox_config_path, "file_sha256"),
        "humaneval_source": (humaneval_source_path, "file_sha256"),
    }
    for field, (path, hash_field) in expected.items():
        value = preflight.get(field)
        if (
            not isinstance(value, Mapping)
            or Path(str(value.get("path", ""))).resolve() != path
            or value.get(hash_field) != file_sha256(path)
        ):
            raise Day20CodeE2BV3Error(f"preflight {field} identity drifted")


def _source_tasks(
    *,
    eligible_rows: list[dict[str, Any]],
    code_rows: list[dict[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    frozen: ModuleType,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    code_ordinals = {
        row["sample_id"]: index for index, row in enumerate(code_rows, 1)
    }
    tasks: list[dict[str, Any]] = []
    for row in eligible_rows:
        lineage = row.get("source_lineage")
        metadata = row.get("metadata")
        if not isinstance(lineage, Mapping) or not isinstance(metadata, Mapping):
            raise Day20CodeE2BV3Error("Code source lineage is incomplete")
        parent_id = lineage.get("parent_id")
        source = source_rows.get(parent_id) if isinstance(parent_id, str) else None
        if not isinstance(source, Mapping):
            raise Day20CodeE2BV3Error(
                f"HumanEval source task is missing: {row.get('sample_id')}"
            )
        source_contract = config.get("source")
        if (
            not isinstance(source_contract, Mapping)
            or lineage.get("source") != source_contract.get("source")
            or lineage.get("revision") != source_contract.get("revision")
            or source.get("prompt", "").strip() != row.get("raw_prompt")
            or source.get("canonical_solution", "").strip() != row.get("reference")
            or source.get("entry_point") != metadata.get("entry_point")
            or frozen.object_sha256(source.get("test"))
            != metadata.get("test_sha256")
        ):
            raise Day20CodeE2BV3Error(
                f"HumanEval lineage drifted: {row.get('sample_id')}"
            )
        candidate = row["code_candidate"]["candidate"]
        tasks.append(
            {
                "code_ordinal": code_ordinals[row["sample_id"]] - 1,
                "sample_id": row["sample_id"],
                "task_id": parent_id,
                "entry_point": source["entry_point"],
                "test_sha256": metadata["test_sha256"],
                "source_prompt": source["prompt"],
                "source_test": source["test"],
                "completion": candidate,
                "manifest_record": {
                    "raw_prompt_hash": frozen.exact_text_hash(row["raw_prompt"]),
                    "reference_hash": row["reference_hash"],
                },
                "normalized_row": row,
                "prediction": {
                    "run_ordinal": row["ordinal"] - 1,
                    "model_id": f"day20-v2/Qwen3.5-4B/{summary['candidate']}",
                    "model_revision": summary["checkpoint_package_sha256"],
                    "extractor_version": V2_CONTRACT_VERSION,
                    "scorer_version": eval_identity.E2B_SCORER_VERSION,
                    "raw_output_hash": frozen.exact_text_hash(candidate),
                    "output_token_ids_hash": None,
                },
            }
        )
    return tasks


def prepare_scoring(
    *,
    predictions_path: Path,
    normalized_summary_path: Path,
    frozen_scorer_path: Path,
    sandbox_config_path: Path,
    humaneval_source_path: Path,
    preflight_path: Path,
    run_root: Path,
    pair_verifier: PairVerifier = _default_pair_verifier,
    frozen_module: ModuleType | None = None,
    preflight_verifier: Callable[..., dict[str, Any]] = verify_preflight,
) -> dict[str, Any]:
    predictions_path = predictions_path.resolve()
    normalized_summary_path = normalized_summary_path.resolve()
    frozen_scorer_path = frozen_scorer_path.resolve()
    sandbox_config_path = sandbox_config_path.resolve()
    humaneval_source_path = humaneval_source_path.resolve()
    preflight_path = preflight_path.resolve()
    try:
        rows, summary = pair_verifier(predictions_path, normalized_summary_path)
    except ValueError as error:
        raise Day20CodeE2BV3Error(str(error)) from error
    scope = summary.get("scope")
    if scope not in eval_identity.EXPECTED_RECORDS_BY_SCOPE:
        raise Day20CodeE2BV3Error("normalized scope is invalid")
    code_rows, eligible_rows = _strict_code_rows(rows, scope=scope)
    frozen = frozen_module or _load_module(frozen_scorer_path, "day20_v3_frozen_e2b")
    config = load_json(sandbox_config_path)
    try:
        frozen.verify_config(config)
        sandbox_contract_hash = frozen.semantic_hash(config)
        source_rows, source_file_hash = frozen.load_humaneval_source(
            humaneval_source_path, config
        )
    except Exception as error:
        raise Day20CodeE2BV3Error(
            f"frozen E2B contract verification failed: {type(error).__name__}"
        ) from error
    if source_file_hash != file_sha256(humaneval_source_path):
        raise Day20CodeE2BV3Error("HumanEval source file identity drifted")
    try:
        preflight = preflight_verifier(preflight_path, run_root=run_root.resolve())
    except ValueError as error:
        raise Day20CodeE2BV3Error(str(error)) from error
    _validate_preflight_bindings(
        preflight,
        frozen_scorer_path=frozen_scorer_path,
        sandbox_config_path=sandbox_config_path,
        humaneval_source_path=humaneval_source_path,
    )
    runtime = _preflight_runtime_identity(preflight, preflight_path)
    code_ids = [row["sample_id"] for row in code_rows]
    e2b_context = eval_identity.normalize_e2b_comparison_context(
        {
            "normalized_context": summary["comparison_context"],
            "normalized_comparison_key": summary["normalized_comparison_key"],
            "code_sample_ids": code_ids,
            "code_sample_order_sha256": eval_identity.object_sha256(code_ids),
            "sandbox_contract_file_sha256": file_sha256(sandbox_config_path),
            "sandbox_contract_content_sha256": eval_identity.object_sha256(config),
            "sandbox_contract_hash": sandbox_contract_hash,
            "e2b_scorer_version": eval_identity.E2B_SCORER_VERSION,
            "e2b_scorer_file_sha256": file_sha256(WRAPPER_PATH),
            "humaneval_source_file_sha256": source_file_hash,
            "e2b_runtime_identity": runtime,
            "e2b_runtime_identity_sha256": eval_identity.object_sha256(runtime),
        }
    )
    comparison = eval_identity.e2b_comparison_key(e2b_context)
    complete = eval_identity.complete_comparison_key(e2b_context)
    eligible_ids = [row["sample_id"] for row in eligible_rows]
    execution_key = eval_identity.object_sha256(
        {
            "domain": "day20.v2.code_e2b_execution",
            "schema_version": SCHEMA_VERSION,
            "candidate": summary["candidate"],
            "evaluation_run_sha256": summary["evaluation_run_sha256"],
            "complete_comparison_key": complete,
            "eligible_code_sample_ids": eligible_ids,
        }
    )
    tasks = _source_tasks(
        eligible_rows=eligible_rows,
        code_rows=code_rows,
        source_rows=source_rows,
        config=config,
        frozen=frozen,
        summary=summary,
    )
    execution_context = {
        "config": config,
        "sandbox_contract_hash": sandbox_contract_hash,
        "sandbox_contract_file_sha256": file_sha256(sandbox_config_path),
        "evaluator_source_sha256": file_sha256(frozen_scorer_path),
        "manifest_hash": summary["comparison_context"][
            "eval_manifest_file_sha256"
        ],
        "manifest_file_sha256": summary["comparison_context"][
            "eval_manifest_file_sha256"
        ],
        "predictions_file_sha256": file_sha256(predictions_path),
        "source_file_sha256": source_file_hash,
        "complete_comparison_key": complete,
        "code_run_hash": execution_key,
        "run_hash": summary["evaluation_run_sha256"],
        "comparison_key": summary["normalized_comparison_key"],
        "model_snapshot_hash": summary["model_identity_sha256"],
    }
    return {
        "rows": rows,
        "summary": summary,
        "code_rows": code_rows,
        "eligible_rows": eligible_rows,
        "eligible_ids": eligible_ids,
        "tasks": tasks,
        "frozen": frozen,
        "config": config,
        "e2b_context": e2b_context,
        "e2b_comparison_key": comparison,
        "complete_comparison_key": complete,
        "execution_key": execution_key,
        "execution_context": execution_context,
        "paths": {
            "predictions": predictions_path,
            "normalized_summary": normalized_summary_path,
            "frozen_scorer": frozen_scorer_path,
            "sandbox_config": sandbox_config_path,
            "humaneval_source": humaneval_source_path,
            "preflight": preflight_path,
        },
    }


def _result_row(
    *,
    raw: Mapping[str, Any],
    task: Mapping[str, Any],
    prepared: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    score = raw.get("score")
    if (
        raw.get("sample_id") != task["sample_id"]
        or raw.get("execution_status") == "infrastructure_error"
        or raw.get("score_status") != "ok"
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or float(score) not in {0.0, 1.0}
        or raw.get("passed") is not (float(score) == 1.0)
    ):
        raise Day20CodeE2BV3Error(
            f"sandbox result is incomplete: {task['sample_id']}"
        )
    normalized = task["normalized_row"]
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": ROW_DOMAIN,
        "ordinal": ordinal,
        "code_ordinal": task["code_ordinal"] + 1,
        "normalized_ordinal": normalized["ordinal"],
        "candidate": prepared["summary"]["candidate"],
        "scope": prepared["summary"]["scope"],
        "sample_id": task["sample_id"],
        "normalized_row_sha256": normalized["row_sha256"],
        "normalized_comparison_key": prepared["summary"][
            "normalized_comparison_key"
        ],
        "evaluation_run_sha256": prepared["summary"]["evaluation_run_sha256"],
        "e2b_comparison_key": prepared["e2b_comparison_key"],
        "complete_comparison_key": prepared["complete_comparison_key"],
        "execution_key": prepared["execution_key"],
        "score_status": "ok",
        "execution_status": raw["execution_status"],
        "score": float(score),
        "passed": raw["passed"],
        "error_type": raw.get("error_type"),
        "exit_code": raw.get("exit_code"),
        "scorer_result": raw.get("scorer_result"),
        "execution_evidence": dict(raw),
        "strict_contract_eligible": True,
        "candidate_code_executed_on_host": False,
    }
    row["row_sha256"] = eval_identity.object_sha256(row)
    return row


def execute_prepared(
    prepared: Mapping[str, Any], *, bindings: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    frozen = prepared["frozen"]
    active_bindings = (
        bindings
        if bindings is not None
        else frozen.load_e2b_bindings(prepared["config"])
    )
    results: list[dict[str, Any]] = []
    for ordinal, task in enumerate(prepared["tasks"], 1):
        raw = frozen.execute_task(
            task, prepared["execution_context"], active_bindings
        )
        results.append(
            _result_row(
                raw=raw,
                task=task,
                prepared=prepared,
                ordinal=ordinal,
            )
        )
    return results


def _atomic_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise Day20CodeE2BV3Error(f"refusing to overwrite artifact: {path}")
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
    except FileExistsError as error:
        raise Day20CodeE2BV3Error(f"artifact appeared during write: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Day20CodeE2BV3Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as error:
        raise Day20CodeE2BV3Error(f"artifact appeared during write: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_summary(
    prepared: Mapping[str, Any], results: list[dict[str, Any]], result_path: Path
) -> dict[str, Any]:
    normalized = prepared["summary"]
    predictions = prepared["paths"]["predictions"]
    normalized_summary_path = prepared["paths"]["normalized_summary"]
    results_content_sha = eval_identity.object_sha256(results)
    run = eval_identity.e2b_run_hash(
        prepared["e2b_context"],
        candidate=normalized["candidate"],
        evaluation_run_sha256=normalized["evaluation_run_sha256"],
        checkpoint_package_sha256=normalized["checkpoint_package_sha256"],
        model_identity_sha256=normalized["model_identity_sha256"],
        normalized_predictions_file_sha256=file_sha256(predictions),
        normalized_predictions_content_sha256=normalized["predictions"][
            "content_sha256"
        ],
        eligible_code_sample_ids=prepared["eligible_ids"],
        raw_e2b_results_file_sha256=file_sha256(result_path),
        raw_e2b_results_content_sha256=results_content_sha,
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": SUMMARY_DOMAIN,
        "status": "complete",
        "candidate": normalized["candidate"],
        "scope": normalized["scope"],
        "records": len(results),
        "code_records": len(prepared["code_rows"]),
        "sandbox_execution_eligible": len(prepared["eligible_ids"]),
        "eligible_code_sample_ids": prepared["eligible_ids"],
        "passed": sum(row["score"] == 1.0 for row in results),
        "failed": sum(row["score"] == 0.0 for row in results),
        "infrastructure_failures": 0,
        "outcomes": dict(
            sorted(Counter(row["execution_status"] for row in results).items())
        ),
        "result": {
            "path": str(result_path.resolve()),
            "file_sha256": file_sha256(result_path),
            "content_sha256": results_content_sha,
            "records": len(results),
            "ordered_sample_ids_sha256": eval_identity.object_sha256(
                [row["sample_id"] for row in results]
            ),
        },
        "normalized_predictions": {
            "path": str(predictions),
            "file_sha256": file_sha256(predictions),
            "content_sha256": normalized["predictions"]["content_sha256"],
        },
        "normalized_summary": {
            "path": str(normalized_summary_path),
            "file_sha256": file_sha256(normalized_summary_path),
            "content_sha256": normalized["summary_sha256"],
        },
        "normalized_comparison_key": normalized["normalized_comparison_key"],
        "evaluation_run_sha256": normalized["evaluation_run_sha256"],
        "e2b_comparison_context": prepared["e2b_context"],
        "e2b_comparison_key": prepared["e2b_comparison_key"],
        "complete_comparison_key": prepared["complete_comparison_key"],
        "execution_key": prepared["execution_key"],
        "e2b_run_sha256": run,
        "candidate_code_executed_on_host": False,
    }
    summary["summary_sha256"] = eval_identity.object_sha256(summary)
    return summary


def verify_published_pair(
    summary_path: Path,
    *,
    pair_verifier: PairVerifier = _default_pair_verifier,
    expected_candidate: str | None = None,
    expected_scope: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary_path = summary_path.resolve()
    summary = load_json(summary_path)
    _self_hash(summary, "summary_sha256")
    candidate = summary.get("candidate")
    scope = summary.get("scope")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("domain") != SUMMARY_DOMAIN
        or summary.get("status") != "complete"
        or not isinstance(candidate, str)
        or scope not in eval_identity.EXPECTED_RECORDS_BY_SCOPE
        or (expected_candidate is not None and candidate != expected_candidate)
        or (expected_scope is not None and scope != expected_scope)
        or summary.get("candidate_code_executed_on_host") is not False
    ):
        raise Day20CodeE2BV3Error("E2B summary identity drifted")
    normalized_predictions = summary.get("normalized_predictions")
    normalized_summary = summary.get("normalized_summary")
    result_identity = summary.get("result")
    if not all(
        isinstance(value, Mapping)
        for value in (normalized_predictions, normalized_summary, result_identity)
    ):
        raise Day20CodeE2BV3Error("E2B artifact identities are incomplete")
    predictions_path = Path(str(normalized_predictions.get("path", ""))).resolve()
    source_summary_path = Path(str(normalized_summary.get("path", ""))).resolve()
    result_path = Path(str(result_identity.get("path", ""))).resolve()
    if (
        normalized_predictions.get("file_sha256") != file_sha256(predictions_path)
        or normalized_summary.get("file_sha256") != file_sha256(source_summary_path)
        or result_identity.get("file_sha256") != file_sha256(result_path)
    ):
        raise Day20CodeE2BV3Error("E2B artifact file identity drifted")
    try:
        normalized_rows, source_summary = pair_verifier(
            predictions_path,
            source_summary_path,
            expected_scope=scope,
            expected_candidate=candidate,
        )
    except ValueError as error:
        raise Day20CodeE2BV3Error(str(error)) from error
    if (
        normalized_predictions.get("content_sha256")
        != source_summary["predictions"]["content_sha256"]
        or normalized_summary.get("content_sha256") != source_summary["summary_sha256"]
        or summary.get("normalized_comparison_key")
        != source_summary["normalized_comparison_key"]
        or summary.get("evaluation_run_sha256")
        != source_summary["evaluation_run_sha256"]
    ):
        raise Day20CodeE2BV3Error("normalized-to-E2B lineage drifted")
    code_rows, eligible_rows = _strict_code_rows(normalized_rows, scope=scope)
    eligible_ids = [row["sample_id"] for row in eligible_rows]
    results = load_jsonl(result_path)
    if (
        len(results) != len(eligible_ids)
        or [row.get("sample_id") for row in results] != eligible_ids
        or result_identity.get("records") != len(results)
        or result_identity.get("content_sha256")
        != eval_identity.object_sha256(results)
        or result_identity.get("ordered_sample_ids_sha256")
        != eval_identity.object_sha256(eligible_ids)
    ):
        raise Day20CodeE2BV3Error("E2B result coverage/order drifted")
    normalized_by_id = {row["sample_id"]: row for row in normalized_rows}
    context = eval_identity.normalize_e2b_comparison_context(
        summary.get("e2b_comparison_context")
    )
    e2b_comparison = eval_identity.e2b_comparison_key(context)
    complete = eval_identity.complete_comparison_key(context)
    execution_key = eval_identity.object_sha256(
        {
            "domain": "day20.v2.code_e2b_execution",
            "schema_version": SCHEMA_VERSION,
            "candidate": candidate,
            "evaluation_run_sha256": source_summary["evaluation_run_sha256"],
            "complete_comparison_key": complete,
            "eligible_code_sample_ids": eligible_ids,
        }
    )
    code_ordinals = {
        row["sample_id"]: index for index, row in enumerate(code_rows, 1)
    }
    for ordinal, row in enumerate(results, 1):
        normalized = normalized_by_id[row["sample_id"]]
        raw = row.get("execution_evidence")
        score = row.get("score")
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("domain") != ROW_DOMAIN
            or row.get("ordinal") != ordinal
            or row.get("code_ordinal") != code_ordinals[row["sample_id"]]
            or row.get("normalized_ordinal") != normalized["ordinal"]
            or row.get("candidate") != candidate
            or row.get("scope") != scope
            or row.get("normalized_row_sha256") != normalized["row_sha256"]
            or row.get("normalized_comparison_key")
            != source_summary["normalized_comparison_key"]
            or row.get("evaluation_run_sha256")
            != source_summary["evaluation_run_sha256"]
            or row.get("e2b_comparison_key") != e2b_comparison
            or row.get("complete_comparison_key") != complete
            or row.get("execution_key") != execution_key
            or row.get("strict_contract_eligible") is not True
            or row.get("candidate_code_executed_on_host") is not False
            or row.get("score_status") != "ok"
            or row.get("execution_status") == "infrastructure_error"
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or float(score) not in {0.0, 1.0}
            or row.get("passed") is not (float(score) == 1.0)
            or not isinstance(raw, Mapping)
            or raw.get("sample_id") != row["sample_id"]
            or raw.get("score") != score
            or raw.get("passed") is not row["passed"]
            or raw.get("score_status") != "ok"
            or raw.get("execution_status") != row["execution_status"]
        ):
            raise Day20CodeE2BV3Error(
                f"E2B result row drifted: {row.get('sample_id')}"
            )
        _self_hash(row, "row_sha256")
    run = eval_identity.e2b_run_hash(
        context,
        candidate=candidate,
        evaluation_run_sha256=source_summary["evaluation_run_sha256"],
        checkpoint_package_sha256=source_summary["checkpoint_package_sha256"],
        model_identity_sha256=source_summary["model_identity_sha256"],
        normalized_predictions_file_sha256=file_sha256(predictions_path),
        normalized_predictions_content_sha256=source_summary["predictions"][
            "content_sha256"
        ],
        eligible_code_sample_ids=eligible_ids,
        raw_e2b_results_file_sha256=file_sha256(result_path),
        raw_e2b_results_content_sha256=eval_identity.object_sha256(results),
    )
    outcomes = dict(
        sorted(Counter(row["execution_status"] for row in results).items())
    )
    if (
        summary.get("records") != len(results)
        or summary.get("code_records") != len(code_rows)
        or summary.get("sandbox_execution_eligible") != len(eligible_ids)
        or summary.get("eligible_code_sample_ids") != eligible_ids
        or summary.get("passed") != sum(row["score"] == 1.0 for row in results)
        or summary.get("failed") != sum(row["score"] == 0.0 for row in results)
        or summary.get("infrastructure_failures") != 0
        or summary.get("outcomes") != outcomes
        or summary.get("e2b_comparison_key") != e2b_comparison
        or summary.get("complete_comparison_key") != complete
        or summary.get("execution_key") != execution_key
        or summary.get("e2b_run_sha256") != run
    ):
        raise Day20CodeE2BV3Error("E2B summary aggregates or keys drifted")
    return results, summary, normalized_rows, source_summary


def score_code_pair(
    *,
    predictions_path: Path,
    normalized_summary_path: Path,
    frozen_scorer_path: Path,
    sandbox_config_path: Path,
    humaneval_source_path: Path,
    preflight_path: Path,
    run_root: Path,
    output_path: Path,
    summary_output_path: Path,
    pair_verifier: PairVerifier = _default_pair_verifier,
    frozen_module: ModuleType | None = None,
    preflight_verifier: Callable[..., dict[str, Any]] = verify_preflight,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = output_path.resolve()
    summary_output = summary_output_path.resolve()
    if output.exists() or summary_output.exists():
        if not output.is_file() or not summary_output.is_file():
            raise Day20CodeE2BV3Error("existing E2B output pair is incomplete")
        results, summary, _, _ = verify_published_pair(
            summary_output, pair_verifier=pair_verifier
        )
        if Path(summary["result"]["path"]).resolve() != output:
            raise Day20CodeE2BV3Error("existing E2B output path drifted")
        prepared = prepare_scoring(
            predictions_path=predictions_path,
            normalized_summary_path=normalized_summary_path,
            frozen_scorer_path=frozen_scorer_path,
            sandbox_config_path=sandbox_config_path,
            humaneval_source_path=humaneval_source_path,
            preflight_path=preflight_path,
            run_root=run_root,
            pair_verifier=pair_verifier,
            frozen_module=frozen_module,
            preflight_verifier=preflight_verifier,
        )
        if (
            summary["e2b_comparison_context"] != prepared["e2b_context"]
            or summary["eligible_code_sample_ids"] != prepared["eligible_ids"]
            or len(results) != len(prepared["tasks"])
        ):
            raise Day20CodeE2BV3Error("existing E2B output no longer matches inputs")
        return summary

    prepared = prepare_scoring(
        predictions_path=predictions_path,
        normalized_summary_path=normalized_summary_path,
        frozen_scorer_path=frozen_scorer_path,
        sandbox_config_path=sandbox_config_path,
        humaneval_source_path=humaneval_source_path,
        preflight_path=preflight_path,
        run_root=run_root,
        pair_verifier=pair_verifier,
        frozen_module=frozen_module,
        preflight_verifier=preflight_verifier,
    )
    results = execute_prepared(prepared, bindings=bindings)
    _atomic_jsonl_new(output, results)
    summary = _build_summary(prepared, results, output)
    _atomic_json_new(summary_output, summary)
    verified_results, verified_summary, _, _ = verify_published_pair(
        summary_output, pair_verifier=pair_verifier
    )
    if verified_results != results or verified_summary != summary:
        raise Day20CodeE2BV3Error("E2B output changed during publication")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--frozen-e2b-scorer", required=True, type=Path)
    parser.add_argument("--sandbox-config", required=True, type=Path)
    parser.add_argument("--humaneval-source", required=True, type=Path)
    parser.add_argument("--e2b-preflight", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = score_code_pair(
            predictions_path=args.predictions,
            normalized_summary_path=args.eval_summary,
            frozen_scorer_path=args.frozen_e2b_scorer,
            sandbox_config_path=args.sandbox_config,
            humaneval_source_path=args.humaneval_source,
            preflight_path=args.e2b_preflight,
            run_root=args.run_root,
            output_path=args.output,
            summary_output_path=args.summary_output,
        )
    except Day20CodeE2BV3Error as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
