#!/usr/bin/env python3
"""Execute Qwen3.5-adapted Day 19 HumanEval candidates in pinned E2B sandboxes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import score_day19_code_e2b as legacy
from day19_humaneval_adapter import (
    COMPOSER_VERSION,
    PARSER_VERSION,
    classify_code_candidate,
    compose_humaneval_program,
)
from qwen35_response_adapter import (
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    adapt_qwen35_text,
    text_sha256,
)
from qwen35_response_adapter import validate_ms_swift_response_capture
from rescore_day19_qwen35_v2 import (
    LEGACY_FALLBACK_RECIPES,
    RESCORER_PATH,
    _code_static_status,
    adapter_comparison_key,
    adapter_run_hash,
    verify_model_export_identity,
)


HERE = Path(__file__).resolve().parent
RESPONSE_ADAPTER_PATH = HERE / "qwen35_response_adapter.py"
CODE_ADAPTER_PATH = HERE / "day19_humaneval_adapter.py"


class Day19CodeSandboxV2Error(ValueError):
    """A normalized-v2 sandbox input or publication invariant failed."""


def normalized_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        raise Day19CodeSandboxV2Error(
            f"invalid response capture for {sample_id}: {error}"
        ) from error


def verify_v2_predictions(
    *,
    predictions_path: Path,
    summary_path: Path,
    manifest: dict[str, Any],
    manifest_file_sha256: str,
    frozen_scorers: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = legacy.load_jsonl(predictions_path)
    summary = legacy.load_json(summary_path)
    order = manifest["header"]["evaluation_order"]["dev"]
    records = {row["sample_id"]: row for row in manifest["records"]}
    if len(rows) != 112 or len(order) != 112:
        raise Day19CodeSandboxV2Error("normalized-v2 scoring requires all 112 dev rows")
    expected_summary_sha = object_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    if (
        summary.get("schema_version") != 2
        or summary.get("domain") != "day19.qwen35_adapter_summary"
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("summary_sha256") != expected_summary_sha
    ):
        raise Day19CodeSandboxV2Error("normalized-v2 summary is not sandbox-ready")
    prediction_identity = summary.get("predictions")
    if (
        not isinstance(prediction_identity, dict)
        or prediction_identity.get("records") != 112
        or prediction_identity.get("file_sha256")
        != legacy.file_sha256(predictions_path)
    ):
        raise Day19CodeSandboxV2Error("normalized prediction hash differs from summary")
    recipe = summary.get("recipe")
    checkpoint = summary.get("checkpoint")
    model_export = summary.get("model_export")
    protocol = summary.get("protocol")
    scorer_sha = manifest.get("header", {}).get("scorer_source_sha256")
    if (
        recipe not in (*sorted(LEGACY_FALLBACK_RECIPES), "untouched-c0")
        or not isinstance(checkpoint, str)
        or not checkpoint
        or not isinstance(protocol, dict)
        or not isinstance(scorer_sha, str)
    ):
        raise Day19CodeSandboxV2Error("normalized model/protocol identity is incomplete")
    try:
        model_snapshot_sha = verify_model_export_identity(model_export)
    except ValueError as error:
        raise Day19CodeSandboxV2Error(str(error)) from error
    expected_protocol = {
        "response_adapter_version": ADAPTER_VERSION,
        "response_adapter_file_sha256": legacy.file_sha256(RESPONSE_ADAPTER_PATH),
        "code_parser_version": PARSER_VERSION,
        "code_composer_version": COMPOSER_VERSION,
        "code_adapter_file_sha256": legacy.file_sha256(CODE_ADAPTER_PATH),
        "rescorer_file_sha256": legacy.file_sha256(RESCORER_PATH),
        "manifest_file_sha256": manifest_file_sha256,
        "scorer_file_sha256": scorer_sha,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise Day19CodeSandboxV2Error("normalized implementation provenance drifted")
    expected_comparison_key = adapter_comparison_key(
        manifest_file_sha256=manifest_file_sha256,
        scorer_file_sha256=scorer_sha,
    )
    if summary.get("comparison_key") != expected_comparison_key:
        raise Day19CodeSandboxV2Error("normalized comparison key drifted")
    source_prediction_sha = protocol.get("source_prediction_file_sha256")
    source_summary_file_sha = protocol.get("source_evaluation_summary_sha256")
    source_summary_content_sha = protocol.get(
        "source_evaluation_summary_content_sha256"
    )
    if not all(
        isinstance(value, str) and value
        for value in (
            source_prediction_sha,
            source_summary_file_sha,
            source_summary_content_sha,
        )
    ):
        raise Day19CodeSandboxV2Error("normalized source provenance is incomplete")
    expected_run_hash = adapter_run_hash(
        recipe=recipe,
        comparison_key=expected_comparison_key,
        checkpoint=checkpoint,
        model_snapshot_sha256=model_snapshot_sha,
        source_prediction_file_sha256=source_prediction_sha,
        source_evaluation_summary_file_sha256=source_summary_file_sha,
        source_evaluation_summary_content_sha256=source_summary_content_sha,
    )
    if summary.get("run_hash") != expected_run_hash:
        raise Day19CodeSandboxV2Error("normalized run hash drifted")
    source_eval_protocol = protocol.get("source_evaluation_protocol")
    if (
        not isinstance(source_eval_protocol, dict)
        or source_eval_protocol.get("template") != "qwen3_5"
        or source_eval_protocol.get("enable_thinking") is not False
        or source_eval_protocol.get("add_non_thinking_prefix") is not True
    ):
        raise Day19CodeSandboxV2Error("source Qwen3.5 template protocol drifted")
    rows_with_capture = sum(row.get("response_capture") is not None for row in rows)
    capture_declared = (
        source_eval_protocol.get("response_capture")
        == "ms-swift RequestConfig(return_details=True)"
        and source_eval_protocol.get("generated_token_ids_retained") is True
    )
    if rows_with_capture not in (0, 112) or capture_declared != (rows_with_capture == 112):
        raise Day19CodeSandboxV2Error("normalized response-capture coverage drifted")
    if rows_with_capture == 112 and (
        source_eval_protocol.get("response_boundary_adapter") != ADAPTER_VERSION
        or source_eval_protocol.get("non_thinking_prefix_sha256")
        != text_sha256(NON_THINKING_PREFIX)
        or not isinstance(source_eval_protocol.get("ms_swift_version"), str)
        or not source_eval_protocol["ms_swift_version"]
    ):
        raise Day19CodeSandboxV2Error("token-backed response protocol drifted")
    if rows_with_capture == 0:
        modern_fields = (
            "response_capture",
            "response_boundary_adapter",
            "generated_token_ids_retained",
            "ms_swift_version",
            "non_thinking_prefix_sha256",
        )
        if recipe not in LEGACY_FALLBACK_RECIPES or any(
            source_eval_protocol.get(field) is not None for field in modern_fields
        ):
            raise Day19CodeSandboxV2Error(
                "prefix fallback requires explicitly tokenless legacy A/B/E"
            )
    for ordinal, (row, sample_id) in enumerate(zip(rows, order), 1):
        record = records[sample_id]
        if (
            row.get("schema_version") != 2
            or row.get("domain") != "day19.qwen35_adapter_prediction"
            or row.get("ordinal") != ordinal
            or row.get("recipe") != recipe
            or row.get("sample_id") != sample_id
            or row.get("slice") != record.get("slice")
            or row.get("reference_hash") != record.get("reference_hash")
            or row.get("prompt_messages_hash") != record.get("messages_hash")
            or row.get("comparison_key") != summary.get("comparison_key")
            or row.get("run_hash") != summary.get("run_hash")
            or row.get("source_prediction", {}).get("file_sha256")
            != source_prediction_sha
            or row.get("source_evaluation_summary_sha256")
            != source_summary_file_sha
        ):
            raise Day19CodeSandboxV2Error(f"normalized row identity mismatch: {sample_id}")
        expected_row_sha = object_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        if row.get("row_sha256") != expected_row_sha:
            raise Day19CodeSandboxV2Error(f"normalized row hash mismatch: {sample_id}")
        raw_output = row.get("raw_output")
        if not isinstance(raw_output, str) or row.get("raw_output_sha256") != text_sha256(
            raw_output
        ):
            raise Day19CodeSandboxV2Error(f"raw output evidence mismatch: {sample_id}")
        adapted = adapt_qwen35_text(
            raw_output,
            generated_only_text=_generated_only_text(row, sample_id),
        )
        if row.get("response_adapter") != normalized_json(adapted):
            raise Day19CodeSandboxV2Error(f"response adapter drifted: {sample_id}")
        if (
            row.get("normalized_output") != adapted["final_text"]
            or row.get("normalized_output_sha256") != adapted["final_text_sha256"]
        ):
            raise Day19CodeSandboxV2Error(f"normalized output drifted: {sample_id}")
        expected_score = normalized_json(
            frozen_scorers.score_prediction(
                record["slice"], adapted["final_text"], record["reference"]
            )
        )
        if row.get("normalized_scorer_result") != expected_score:
            raise Day19CodeSandboxV2Error(f"normalized scorer drifted: {sample_id}")
        if record["slice"] == "code":
            expected_candidate = normalized_json(
                classify_code_candidate(
                    adapted["final_text"],
                    entry_point=record["metadata"]["entry_point"],
                    extract_code_completion=frozen_scorers.extract_code_completion,
                )
            )
            if row.get("code_candidate") != expected_candidate:
                raise Day19CodeSandboxV2Error(f"code parser drifted: {sample_id}")
            expected_static = _code_static_status(
                record.get("raw_prompt", ""),
                record["metadata"]["entry_point"],
                expected_candidate,
            )
            expected_execution_eligible = (
                expected_candidate["execution_eligible"] and expected_static["valid"]
            )
            if (
                row.get("code_static_syntax") != expected_static
                or row.get("sandbox_execution_eligible")
                is not expected_execution_eligible
            ):
                raise Day19CodeSandboxV2Error(
                    f"code execution eligibility drifted: {sample_id}"
                )
    return rows, summary


def build_v2_tasks(
    *,
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    code_order: list[str],
    source_rows: dict[str, dict[str, Any]],
    frozen: ModuleType,
) -> list[dict[str, Any]]:
    predictions_by_id = {row["sample_id"]: row for row in predictions}
    tasks: list[dict[str, Any]] = []
    for sample_id in code_order:
        record = manifest_by_id[sample_id]
        row = predictions_by_id[sample_id]
        source = source_rows.get(record["source_lineage"]["parent_id"])
        if source is None:
            raise Day19CodeSandboxV2Error(f"HumanEval source task is missing: {sample_id}")
        metadata = record["metadata"]
        if (
            source["prompt"].strip() != record["raw_prompt"]
            or source["canonical_solution"].strip() != record["reference"]
            or source["entry_point"] != metadata["entry_point"]
            or frozen.object_sha256(source["test"]) != metadata["test_sha256"]
        ):
            raise Day19CodeSandboxV2Error(f"HumanEval source lineage mismatch: {sample_id}")
        candidate = row["code_candidate"]
        capture = row.get("response_capture")
        generated_token_hash = None
        if capture is not None:
            generated_token_hash = frozen.semantic_hash(
                {
                    "domain": "day10.output_token_ids",
                    "schema_version": 1,
                    "output_token_ids": capture["generated_token_ids"],
                }
            )
        tasks.append(
            {
                "code_ordinal": len(tasks),
                "sample_id": sample_id,
                "task_id": record["source_lineage"]["parent_id"],
                "entry_point": source["entry_point"],
                "test_sha256": metadata["test_sha256"],
                "source_prompt": source["prompt"],
                "source_test": source["test"],
                "completion": candidate["candidate"],
                "candidate_mode": candidate["candidate_mode"],
                "candidate_metadata": candidate,
                "execution_eligible": row["sandbox_execution_eligible"],
                "manifest_record": record,
                "prediction": {
                    "run_ordinal": row["ordinal"] - 1,
                    "model_id": f"day19/Qwen3.5-4B/{summary['recipe']}",
                    "model_revision": str(summary.get("checkpoint")),
                    "extractor_version": PARSER_VERSION,
                    "scorer_version": "day19-humaneval-pass-at-1-e2b-v3",
                    "raw_output_hash": frozen.exact_text_hash(row["raw_output"]),
                    "output_token_ids_hash": generated_token_hash,
                },
            }
        )
    return tasks


def effective_contract(
    base_config: dict[str, Any], base_contract_hash: str
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.update(
        {
            "domain": "day19.qwen35_humaneval_e2b_contract",
            "schema_version": 3,
            "composition": (
                "solution: source preamble before entry_point + candidate + "
                "source_test + check(entry_point); "
                "completion: source_prompt + candidate + source_test + check(entry_point)"
            ),
            "response_adapter_version": ADAPTER_VERSION,
            "candidate_parser_version": PARSER_VERSION,
            "candidate_composer_version": COMPOSER_VERSION,
            "candidate_acceptance": {
                "solution": (
                    "standalone-valid module with exactly one top-level definition "
                    "of the expected entry point"
                ),
                "completion": (
                    "indented function body contained wholly within a synthetic "
                    "wrapper, with no module-level escape"
                ),
                "composed_program": (
                    "the exact source-preamble/solution or prompt/completion program "
                    "must compile before sandbox submission"
                ),
            },
            "contract_violation_score_policy": (
                "execution_eligible=false is not sent to E2B and is deterministically "
                "published as rejected_candidate_contract with score 0"
            ),
            "implementation_sources": {
                "response_adapter_file_sha256": legacy.file_sha256(
                    RESPONSE_ADAPTER_PATH
                ),
                "code_adapter_file_sha256": legacy.file_sha256(CODE_ADAPTER_PATH),
                "sandbox_wrapper_file_sha256": legacy.file_sha256(
                    Path(__file__).resolve()
                ),
            },
            "parent_day10_sandbox_contract_hash": base_contract_hash,
            "parent_day10_evaluator_source_sha256": base_config[
                "evaluator_source_sha256"
            ],
        }
    )
    return config


def score_v2_tasks(
    *,
    tasks: list[dict[str, Any]],
    context: dict[str, Any],
    bindings: dict[str, Any],
    frozen: ModuleType,
) -> list[dict[str, Any]]:
    """Score eligible tasks and construct deterministic rows for rejections."""

    results: list[dict[str, Any]] = []
    for task in tasks:
        if task["execution_eligible"]:
            row = frozen.execute_task(task, context, bindings)
            if row.get("execution_status") == "infrastructure_error":
                raise Day19CodeSandboxV2Error(
                    f"sandbox infrastructure failed: {task['sample_id']}"
                )
        else:
            started = time.perf_counter()
            row = frozen._finalize_result_row(
                task,
                context,
                started=started,
                execution_status="rejected_candidate_contract",
                score_status="ok",
                score=0.0,
                error_type="candidate_acceptance_contract_violation",
                exit_code=None,
                failure_message="candidate failed the v3 acceptance contract",
            )
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise Day19CodeSandboxV2Error("refusing to overwrite normalized sandbox output")

    frozen = legacy.load_module(args.frozen_e2b_scorer.resolve())
    base_config, base_config_payload = frozen.load_json(
        args.sandbox_config, "sandbox contract"
    )
    frozen.verify_config(base_config)
    base_contract_hash = frozen.semantic_hash(base_config)
    config = effective_contract(base_config, base_contract_hash)
    effective_contract_hash = frozen.semantic_hash(config)
    manifest, manifest_payload = frozen.load_json(args.manifest, "manifest")
    manifest_file_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_by_id, code_order = frozen.verify_manifest(
        manifest, manifest_payload, base_config
    )
    frozen_scorers = frozen.load_frozen_scorers(manifest["header"])
    source_rows, source_file_sha256 = legacy.load_humaneval_source(
        args.humaneval_source, base_config
    )
    predictions, eval_summary = verify_v2_predictions(
        predictions_path=args.predictions,
        summary_path=args.eval_summary,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        frozen_scorers=frozen_scorers,
    )
    tasks = build_v2_tasks(
        predictions=predictions,
        summary=eval_summary,
        manifest_by_id=manifest_by_id,
        code_order=code_order,
        source_rows=source_rows,
        frozen=frozen,
    )
    if len(tasks) != 28:
        raise Day19CodeSandboxV2Error(f"expected 28 code tasks, got {len(tasks)}")

    predictions_file_sha256 = legacy.file_sha256(args.predictions)
    comparison_key = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_adapter_code_comparison",
            "schema_version": 2,
            "manifest_hash": manifest["header"]["manifest_hash"],
            "manifest_file_sha256": manifest_file_sha256,
            "adapter_comparison_key": eval_summary["comparison_key"],
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    run_hash = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_adapter_code_run",
            "schema_version": 2,
            "recipe": eval_summary["recipe"],
            "adapter_run_hash": eval_summary["run_hash"],
            "predictions_file_sha256": predictions_file_sha256,
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    code_run_hash = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_adapter_e2b_run",
            "schema_version": 2,
            "run_hash": run_hash,
            "sample_ids": code_order,
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    complete_comparison_key = frozen.semantic_hash(
        {
            "domain": "day19.qwen35_adapter_complete_comparison",
            "schema_version": 2,
            "comparison_key": comparison_key,
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    model_snapshot_hash = (eval_summary.get("model_export") or {}).get(
        "snapshot_sha256"
    )
    if not isinstance(model_snapshot_hash, str) or not model_snapshot_hash:
        raise Day19CodeSandboxV2Error("normalized summary lacks model snapshot identity")
    context = {
        "config": config,
        "sandbox_contract_hash": effective_contract_hash,
        "sandbox_contract_file_sha256": hashlib.sha256(base_config_payload).hexdigest(),
        "evaluator_source_sha256": base_config["evaluator_source_sha256"],
        "manifest_hash": manifest["header"]["manifest_hash"],
        "manifest_file_sha256": manifest_file_sha256,
        "predictions_file_sha256": predictions_file_sha256,
        "source_file_sha256": source_file_sha256,
        "complete_comparison_key": complete_comparison_key,
        "code_run_hash": code_run_hash,
        "run_hash": run_hash,
        "comparison_key": comparison_key,
        "model_snapshot_hash": model_snapshot_hash,
    }

    def compose_v2(task: dict[str, Any]) -> str:
        return compose_humaneval_program(
            source_prompt=task["source_prompt"],
            source_test=task["source_test"],
            entry_point=task["entry_point"],
            candidate=task["completion"],
            candidate_mode=task["candidate_mode"],
        )

    # Reuse only the pinned E2B isolation/runner machinery.  The effective
    # protocol and all hashes above explicitly record this v2 composer.
    frozen.compose_humaneval_program = compose_v2
    bindings = frozen.load_e2b_bindings(config)
    if args.live_smoke:
        legacy.live_smoke(frozen, config, bindings)
    results = score_v2_tasks(
        tasks=tasks, context=context, bindings=bindings, frozen=frozen
    )
    if len(results) != 28 or any(row.get("score_status") != "ok" for row in results):
        raise Day19CodeSandboxV2Error("sandbox scoring did not return 28 valid results")
    tasks_by_id = {task["sample_id"]: task for task in tasks}
    for row in results:
        task = tasks_by_id[row["sample_id"]]
        row.update(
            {
                "domain": "day19.qwen35_adapter_code_sandbox_result",
                "schema_version": 2,
                "recipe": eval_summary["recipe"],
                "harness_version": "day19-qwen35-humaneval-e2b-v3",
                "response_adapter_version": ADAPTER_VERSION,
                "candidate_parser_version": PARSER_VERSION,
                "candidate_composer_version": COMPOSER_VERSION,
                "candidate_mode": task["candidate_mode"],
                "candidate_metadata": task["candidate_metadata"],
                "prediction_run_hash": eval_summary["run_hash"],
                "prediction_comparison_key": eval_summary["comparison_key"],
                "candidate_executed_in_sandbox": task["execution_eligible"],
                "base_day10_sandbox_contract_hash": base_contract_hash,
                "base_day10_sandbox_contract_file_sha256": hashlib.sha256(
                    base_config_payload
                ).hexdigest(),
                "base_day10_evaluator_source_sha256": base_config[
                    "evaluator_source_sha256"
                ],
                "effective_code_protocol_hash": effective_contract_hash,
                "normalized_prediction_file_sha256": predictions_file_sha256,
                "candidate_code_executed_on_host": False,
            }
        )
        if task["execution_eligible"]:
            row["sandbox"]["execution_attempted"] = True
        else:
            row["sandbox"] = {
                "backend": "not_created",
                "execution_attempted": False,
                "reason": "candidate_acceptance_contract_violation",
            }
        # The inherited frozen fields use one name for both the parent sandbox
        # file and the in-memory v2 protocol.  Publish them as two explicit,
        # internally consistent identities in the v2 schema.
        row["sandbox_contract_hash"] = base_contract_hash
        row["sandbox_contract_file_sha256"] = hashlib.sha256(
            base_config_payload
        ).hexdigest()
        row["code_execution_protocol_hash"] = effective_contract_hash
        row.pop("evaluator_source_sha256", None)
    frozen.write_jsonl_atomic(results, args.output, overwrite=False)
    result_summary = {
        "schema_version": 2,
        "domain": "day19.qwen35_adapter_code_e2b_summary",
        "status": "complete",
        "recipe": eval_summary["recipe"],
        "records": len(results),
        "passed": sum(row["score"] == 1.0 for row in results),
        "failed": sum(row["score"] == 0.0 for row in results),
        "candidate_modes": dict(
            sorted(Counter(row["candidate_mode"] for row in results).items())
        ),
        "outcomes": dict(
            sorted(Counter(row["execution_status"] for row in results).items())
        ),
        "error_types": dict(
            sorted(Counter(str(row["error_type"]) for row in results).items())
        ),
        "result_path": str(args.output.resolve()),
        "result_file_sha256": legacy.file_sha256(args.output),
        "prediction_path": str(args.predictions.resolve()),
        "prediction_file_sha256": predictions_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "humaneval_source_file_sha256": source_file_sha256,
        "frozen_e2b_scorer_file_sha256": legacy.file_sha256(
            args.frozen_e2b_scorer
        ),
        "response_adapter_file_sha256": legacy.file_sha256(RESPONSE_ADAPTER_PATH),
        "code_adapter_file_sha256": legacy.file_sha256(CODE_ADAPTER_PATH),
        "sandbox_wrapper_file_sha256": legacy.file_sha256(Path(__file__).resolve()),
        "base_sandbox_contract_file_sha256": legacy.file_sha256(args.sandbox_config),
        "base_sandbox_contract_hash": base_contract_hash,
        "effective_code_protocol": config,
        "effective_code_protocol_hash": effective_contract_hash,
        "comparison_key": comparison_key,
        "complete_comparison_key": complete_comparison_key,
        "run_hash": run_hash,
        "code_run_hash": code_run_hash,
        "model_snapshot_hash": model_snapshot_hash,
        "live_smoke_attested_and_killed": args.live_smoke,
        "candidate_code_executed_on_host": False,
        "credential_handling": "credential value is not intentionally emitted",
    }
    result_summary["summary_sha256"] = object_sha256(result_summary)
    legacy.write_json_new(args.summary_output, result_summary)
    print(json.dumps(result_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
