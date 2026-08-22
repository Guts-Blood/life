#!/usr/bin/env python3
"""Execute retained Day 19 HumanEval predictions with the frozen Day 10 E2B scorer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


RECIPES = ("baseline-a", "baseline-b", "best-e")


class Day19CodeSandboxError(ValueError):
    """A Day 19 sandbox input, execution, or publication invariant failed."""


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day19CodeSandboxError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day19CodeSandboxError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day19CodeSandboxError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise Day19CodeSandboxError(f"JSONL file is missing: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise Day19CodeSandboxError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day19CodeSandboxError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise Day19CodeSandboxError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("day19_frozen_e2b_scorer", path)
    if spec is None or spec.loader is None:
        raise Day19CodeSandboxError(f"cannot import frozen E2B scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_humaneval_source(
    path: Path, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], str]:
    expected = config["source"]["file_sha256"]
    actual = file_sha256(path)
    if actual != expected:
        raise Day19CodeSandboxError(
            f"pinned HumanEval source hash mismatch: expected {expected}, got {actual}"
        )
    rows: dict[str, dict[str, Any]] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Day19CodeSandboxError(
                        f"HumanEval row is not an object: {line_number}"
                    )
                task_id = row.get("task_id")
                if not isinstance(task_id, str) or not task_id or task_id in rows:
                    raise Day19CodeSandboxError(f"invalid HumanEval task ID: {task_id!r}")
                if any(not isinstance(row.get(key), str) for key in (
                    "prompt", "canonical_solution", "entry_point", "test"
                )):
                    raise Day19CodeSandboxError(f"incomplete HumanEval source row: {task_id}")
                rows[task_id] = row
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Day19CodeSandboxError(f"cannot decode HumanEval source: {error}") from error
    return rows, actual


def verify_day19_predictions(
    *,
    recipe: str,
    predictions_path: Path,
    summary_path: Path,
    manifest: dict[str, Any],
    frozen_scorers: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(predictions_path)
    summary = load_json(summary_path)
    order = manifest["header"]["evaluation_order"]["dev"]
    records = {row["sample_id"]: row for row in manifest["records"]}
    if len(rows) != 112 or len(order) != 112:
        raise Day19CodeSandboxError("Day 19 sandbox scoring requires all 112 dev predictions")
    if summary.get("recipe") != recipe or summary.get("status") != "complete_with_code_sandbox_required":
        raise Day19CodeSandboxError("Day 19 evaluation summary is not sandbox-ready")
    if summary.get("predictions", {}).get("file_sha256") != file_sha256(predictions_path):
        raise Day19CodeSandboxError("prediction file hash differs from the evaluation summary")
    for index, (row, sample_id) in enumerate(zip(rows, order), 1):
        record = records[sample_id]
        if (
            row.get("schema_version") != 1
            or row.get("ordinal") != index
            or row.get("recipe") != recipe
            or row.get("sample_id") != sample_id
            or row.get("slice") != record.get("slice")
        ):
            raise Day19CodeSandboxError(f"prediction order or identity mismatch: {sample_id}")
        raw_output = row.get("raw_output")
        if not isinstance(raw_output, str):
            raise Day19CodeSandboxError(f"prediction is not text: {sample_id}")
        if row.get("raw_output_sha256") != hashlib.sha256(raw_output.encode("utf-8")).hexdigest():
            raise Day19CodeSandboxError(f"prediction text hash mismatch: {sample_id}")
        if row.get("reference_hash") != record.get("reference_hash"):
            raise Day19CodeSandboxError(f"prediction reference hash mismatch: {sample_id}")
        expected_score = frozen_scorers.score_prediction(
            record["slice"], raw_output, record["reference"]
        )
        normalized_expected_score = json.loads(
            json.dumps(expected_score, ensure_ascii=False, sort_keys=True)
        )
        if row.get("scorer_result") != normalized_expected_score:
            raise Day19CodeSandboxError(f"frozen scorer handoff mismatch: {sample_id}")
    return rows, summary


def build_tasks(
    *,
    recipe: str,
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    code_order: list[str],
    source_rows: dict[str, dict[str, Any]],
    frozen_scorers: ModuleType,
    frozen: ModuleType,
) -> list[dict[str, Any]]:
    predictions_by_id = {row["sample_id"]: row for row in predictions}
    tasks = []
    for sample_id in code_order:
        record = manifest_by_id[sample_id]
        row = predictions_by_id[sample_id]
        lineage = record["source_lineage"]
        task_id = lineage["parent_id"]
        source = source_rows.get(task_id)
        if source is None:
            raise Day19CodeSandboxError(f"HumanEval source task is missing: {task_id}")
        metadata = record["metadata"]
        if source["prompt"].strip() != record["raw_prompt"]:
            raise Day19CodeSandboxError(f"source prompt mismatch: {sample_id}")
        if source["canonical_solution"].strip() != record["reference"]:
            raise Day19CodeSandboxError(f"source reference mismatch: {sample_id}")
        if source["entry_point"] != metadata["entry_point"]:
            raise Day19CodeSandboxError(f"source entry point mismatch: {sample_id}")
        if frozen.object_sha256(source["test"]) != metadata["test_sha256"]:
            raise Day19CodeSandboxError(f"source test hash mismatch: {sample_id}")
        completion = frozen_scorers.extract_code_completion(row["raw_output"])
        if row["scorer_result"].get("parsed_answer") != completion:
            raise Day19CodeSandboxError(f"completion extraction mismatch: {sample_id}")
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
                "prediction": {
                    "run_ordinal": row["ordinal"] - 1,
                    "model_id": f"day19/Qwen3.5-4B/{recipe}",
                    "model_revision": summary["checkpoint"],
                    "extractor_version": record["extractor_version"],
                    "scorer_version": record["scorer_version"],
                    "raw_output_hash": frozen.exact_text_hash(row["raw_output"]),
                    "output_token_ids_hash": None,
                },
            }
        )
    return tasks


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise Day19CodeSandboxError(f"refusing to overwrite sandbox artifact: {path}")
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
    finally:
        if temporary.exists():
            temporary.unlink()


def live_smoke(frozen: ModuleType, config: dict[str, Any], bindings: dict[str, Any]) -> None:
    sandbox = bindings["create"](**frozen.sandbox_create_kwargs(config))
    mismatch = None
    try:
        mismatch = frozen.sandbox_info_mismatch(sandbox.get_info(), config)
    finally:
        killed = sandbox.kill()
    if mismatch is not None:
        raise Day19CodeSandboxError(f"live sandbox attestation mismatch: {mismatch}")
    if killed is False:
        raise Day19CodeSandboxError("live sandbox kill returned false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--frozen-e2b-scorer", required=True, type=Path)
    parser.add_argument("--sandbox-config", required=True, type=Path)
    parser.add_argument("--humaneval-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--live-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.summary_output.exists():
        raise Day19CodeSandboxError("refusing to overwrite existing sandbox output")

    frozen = load_module(args.frozen_e2b_scorer.resolve())
    config, config_payload = frozen.load_json(args.sandbox_config, "sandbox contract")
    frozen.verify_config(config)
    manifest, manifest_payload = frozen.load_json(args.manifest, "manifest")
    manifest_by_id, code_order = frozen.verify_manifest(manifest, manifest_payload, config)
    frozen_scorers = frozen.load_frozen_scorers(manifest["header"])
    source_rows, source_file_sha256 = load_humaneval_source(args.humaneval_source, config)
    predictions, eval_summary = verify_day19_predictions(
        recipe=args.recipe,
        predictions_path=args.predictions,
        summary_path=args.eval_summary,
        manifest=manifest,
        frozen_scorers=frozen_scorers,
    )
    tasks = build_tasks(
        recipe=args.recipe,
        predictions=predictions,
        summary=eval_summary,
        manifest=manifest,
        manifest_by_id=manifest_by_id,
        code_order=code_order,
        source_rows=source_rows,
        frozen_scorers=frozen_scorers,
        frozen=frozen,
    )
    if len(tasks) != 28:
        raise Day19CodeSandboxError(f"expected 28 code tasks, got {len(tasks)}")

    manifest_file_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    predictions_file_sha256 = file_sha256(args.predictions)
    contract_hash = frozen.semantic_hash(config)
    comparison_key = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_frozen_dev_comparison",
            "schema_version": 1,
            "manifest_hash": manifest["header"]["manifest_hash"],
            "manifest_file_sha256": manifest_file_sha256,
            "protocol": eval_summary["protocol"],
        }
    )
    run_hash = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_prediction_run",
            "schema_version": 1,
            "recipe": args.recipe,
            "checkpoint": eval_summary["checkpoint"],
            "model_snapshot_hash": eval_summary["model_export"]["snapshot_sha256"],
            "predictions_file_sha256": predictions_file_sha256,
        }
    )
    complete_comparison_key = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_complete_comparison",
            "schema_version": 1,
            "prediction_comparison_key": comparison_key,
            "code_execution_protocol_hash": contract_hash,
        }
    )
    code_run_hash = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_code_sandbox_run",
            "schema_version": 1,
            "run_hash": run_hash,
            "predictions_file_sha256": predictions_file_sha256,
            "code_execution_protocol_hash": contract_hash,
            "sample_ids": code_order,
        }
    )
    context = {
        "config": config,
        "sandbox_contract_hash": contract_hash,
        "sandbox_contract_file_sha256": hashlib.sha256(config_payload).hexdigest(),
        "evaluator_source_sha256": config["evaluator_source_sha256"],
        "manifest_hash": manifest["header"]["manifest_hash"],
        "manifest_file_sha256": manifest_file_sha256,
        "predictions_file_sha256": predictions_file_sha256,
        "source_file_sha256": source_file_sha256,
        "complete_comparison_key": complete_comparison_key,
        "code_run_hash": code_run_hash,
        "run_hash": run_hash,
        "comparison_key": comparison_key,
        "model_snapshot_hash": eval_summary["model_export"]["snapshot_sha256"],
    }
    bindings = frozen.load_e2b_bindings(config)
    if args.live_smoke:
        live_smoke(frozen, config, bindings)
    results = frozen.score_tasks(tasks, context, bindings)
    if len(results) != 28 or any(row.get("score_status") != "ok" for row in results):
        raise Day19CodeSandboxError("sandbox scoring did not return 28 valid results")
    for row in results:
        row["recipe"] = args.recipe
        row["output_token_ids_hash"] = None
        row["output_token_ids_hash_status"] = (
            "not_recorded_by_day19_transformers_engine; raw output text and exact hash retained"
        )
        row["day19_prediction_file_sha256"] = predictions_file_sha256
    frozen.write_jsonl_atomic(results, args.output, overwrite=False)
    result_summary = {
        "schema_version": 1,
        "domain": "day19.qwen35_code_e2b_summary",
        "status": "complete",
        "recipe": args.recipe,
        "records": len(results),
        "passed": sum(row["score"] == 1.0 for row in results),
        "failed": sum(row["score"] == 0.0 for row in results),
        "outcomes": dict(sorted(Counter(row["execution_status"] for row in results).items())),
        "error_types": dict(sorted(Counter(str(row["error_type"]) for row in results).items())),
        "result_path": str(args.output.resolve()),
        "result_file_sha256": file_sha256(args.output),
        "prediction_path": str(args.predictions.resolve()),
        "prediction_file_sha256": predictions_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "humaneval_source_file_sha256": source_file_sha256,
        "frozen_e2b_scorer_file_sha256": file_sha256(args.frozen_e2b_scorer),
        "sandbox_contract_file_sha256": file_sha256(args.sandbox_config),
        "sandbox_contract_hash": contract_hash,
        "comparison_key": comparison_key,
        "complete_comparison_key": complete_comparison_key,
        "live_smoke_attested_and_killed": args.live_smoke,
        "candidate_code_executed_on_host": False,
        "credential_value_logged": False,
    }
    write_json_new(args.summary_output, result_summary)
    print(json.dumps(result_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
