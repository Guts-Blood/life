#!/usr/bin/env python3
"""Execute Day 20 normalized HumanEval candidates in pinned E2B sandboxes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
DAY19_DIR = HERE.parent / "day-19-qwen35-sft-comparison"
if str(DAY19_DIR) not in sys.path:
    sys.path.insert(0, str(DAY19_DIR))

import rescore_day19_qwen35_v2 as day19_rescore  # noqa: E402
import score_day19_code_e2b_v2 as day19_e2b  # noqa: E402
from day19_humaneval_adapter import (  # noqa: E402
    COMPOSER_VERSION,
    PARSER_VERSION,
    classify_code_candidate,
    compose_humaneval_program,
)
from qwen35_response_adapter import (  # noqa: E402
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    adapt_qwen35_text,
    text_sha256,
    validate_ms_swift_response_capture,
)
from rescore_day20_qwen35_v2 import (  # noqa: E402
    CODE_ADAPTER_PATH,
    PARENT_RESCORER_PATH,
    RESCORER_PATH,
    RESPONSE_ADAPTER_PATH,
    comparison_key as normalized_comparison_key,
    run_hash as normalized_run_hash,
    verify_model_identity,
)
from evaluate_day20 import (  # noqa: E402
    load_experiment_manifest,
    selected_experiment_records,
)


SUMMARY_DOMAIN = "day20.qwen35_adapter_code_e2b_summary"
ROW_DOMAIN = "day20.qwen35_adapter_code_sandbox_result"
WRAPPER_PATH = Path(__file__).resolve()


class Day20CodeSandboxError(ValueError):
    """A normalized sandbox input or publication invariant failed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_credential_attestation(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day20CodeSandboxError(
            f"cannot load credential attestation: {error}"
        ) from error
    expected_self_hash = object_sha256(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    ) if isinstance(value, dict) else None
    credential_file_sha = value.get("credential_file_sha256") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("domain") != "day20.e2b_credential_attestation"
        or value.get("status") != "rotated"
        or not isinstance(value.get("created_at_utc"), str)
        or not value["created_at_utc"]
        or not isinstance(credential_file_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", credential_file_sha)
        or value.get("attestation_sha256") != expected_self_hash
    ):
        raise Day20CodeSandboxError("credential rotation attestation drifted")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_uid != os.getuid():
        raise Day20CodeSandboxError(
            "credential attestation must be mode 0600 and owned by the scorer user"
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": day19_e2b.legacy.file_sha256(path),
        "content_sha256": value["attestation_sha256"],
        "status": "rotated",
    }


def verify_v2_predictions(
    *,
    predictions_path: Path,
    summary_path: Path,
    manifest: dict[str, Any],
    experiment_manifest: dict[str, Any],
    experiment_manifest_path: Path,
    manifest_file_sha256: str,
    frozen_scorers: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    legacy = day19_e2b.legacy
    rows = legacy.load_jsonl(predictions_path)
    summary = legacy.load_json(summary_path)
    expected_summary_sha = object_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    candidate = summary.get("candidate")
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_adapter_summary"
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("summary_sha256") != expected_summary_sha
        or not isinstance(candidate, str)
        or not candidate
        or summary.get("recipe") != candidate
    ):
        raise Day20CodeSandboxError("normalized summary is not sandbox-ready")
    protocol = summary.get("protocol")
    source_protocol = (
        protocol.get("source_evaluation_protocol")
        if isinstance(protocol, dict)
        else None
    )
    if not isinstance(protocol, dict) or not isinstance(source_protocol, dict):
        raise Day20CodeSandboxError("normalized protocol is incomplete")
    sample_limit = source_protocol.get("sample_limit")
    if sample_limit not in (None, 32):
        raise Day20CodeSandboxError("only the 32-row probe or full dev run is valid")
    expected_records = 32 if sample_limit == 32 else 112
    try:
        selected = selected_experiment_records(
            eval_manifest=manifest,
            experiment_manifest=experiment_manifest,
            sample_limit=sample_limit,
            verify_diagnostic_rows=False,
        )
    except ValueError as error:
        raise Day20CodeSandboxError(str(error)) from error
    if len(rows) != expected_records or len(selected) != expected_records:
        raise Day20CodeSandboxError("normalized row count differs from protocol")
    prediction_identity = summary.get("predictions")
    if (
        not isinstance(prediction_identity, dict)
        or prediction_identity.get("records") != expected_records
        or prediction_identity.get("file_sha256")
        != legacy.file_sha256(predictions_path)
    ):
        raise Day20CodeSandboxError("normalized prediction hash differs from summary")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise Day20CodeSandboxError("normalized checkpoint identity is incomplete")
    try:
        model_identity_sha = verify_model_identity(summary.get("model_identity"))
    except ValueError as error:
        raise Day20CodeSandboxError(str(error)) from error
    if summary.get("model_identity_sha256") != model_identity_sha:
        raise Day20CodeSandboxError("normalized model identity hash drifted")
    scorer_sha = manifest.get("header", {}).get("scorer_source_sha256")
    if not isinstance(scorer_sha, str):
        raise Day20CodeSandboxError("manifest scorer identity is missing")
    expected_protocol = {
        "response_adapter_version": ADAPTER_VERSION,
        "response_adapter_file_sha256": legacy.file_sha256(RESPONSE_ADAPTER_PATH),
        "code_parser_version": PARSER_VERSION,
        "code_composer_version": COMPOSER_VERSION,
        "code_adapter_file_sha256": legacy.file_sha256(CODE_ADAPTER_PATH),
        "parent_rescorer_file_sha256": legacy.file_sha256(PARENT_RESCORER_PATH),
        "rescorer_file_sha256": legacy.file_sha256(RESCORER_PATH),
        "manifest_file_sha256": manifest_file_sha256,
        "experiment_manifest_file_sha256": legacy.file_sha256(
            experiment_manifest_path
        ),
        "scorer_file_sha256": scorer_sha,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise Day20CodeSandboxError("normalized implementation provenance drifted")
    expected_comparison = normalized_comparison_key(
        manifest_sha=manifest_file_sha256,
        experiment_manifest_sha=legacy.file_sha256(experiment_manifest_path),
        scorer_sha=scorer_sha,
    )
    if summary.get("comparison_key") != expected_comparison:
        raise Day20CodeSandboxError("normalized comparison key drifted")
    source_predictions_sha = protocol.get("source_prediction_file_sha256")
    source_summary_file_sha = protocol.get("source_evaluation_summary_sha256")
    source_summary_content_sha = protocol.get(
        "source_evaluation_summary_content_sha256"
    )
    if not all(
        isinstance(value, str) and value
        for value in (
            source_predictions_sha,
            source_summary_file_sha,
            source_summary_content_sha,
        )
    ):
        raise Day20CodeSandboxError("normalized source provenance is incomplete")
    expected_run = normalized_run_hash(
        candidate=candidate,
        comparison=expected_comparison,
        checkpoint=checkpoint,
        model_identity_sha=model_identity_sha,
        source_predictions_sha=source_predictions_sha,
        source_summary_file_sha=source_summary_file_sha,
        source_summary_content_sha=source_summary_content_sha,
    )
    if summary.get("run_hash") != expected_run:
        raise Day20CodeSandboxError("normalized run hash drifted")
    if (
        source_protocol.get("records") != expected_records
        or source_protocol.get("experiment_manifest_file_sha256")
        != legacy.file_sha256(experiment_manifest_path)
        or source_protocol.get("experiment_manifest_content_sha256")
        != experiment_manifest.get("manifest_sha256")
        or source_protocol.get("diagnostic_selection_sha256")
        != (
            experiment_manifest["datasets"]["diagnostic"]["selection_sha256"]
            if sample_limit == 32
            else None
        )
        or source_protocol.get("template") != "qwen3_5"
        or source_protocol.get("enable_thinking") is not False
        or source_protocol.get("add_non_thinking_prefix") is not True
        or source_protocol.get("response_capture")
        != "ms-swift RequestConfig(return_details=True)"
        or source_protocol.get("generated_token_ids_retained") is not True
        or source_protocol.get("response_boundary_adapter") != ADAPTER_VERSION
        or source_protocol.get("non_thinking_prefix_sha256")
        != text_sha256(NON_THINKING_PREFIX)
        or not isinstance(source_protocol.get("ms_swift_version"), str)
        or not source_protocol.get("ms_swift_version")
    ):
        raise Day20CodeSandboxError("source Qwen3.5 response protocol drifted")

    for ordinal, (row, record) in enumerate(zip(rows, selected), 1):
        sample_id = record["sample_id"]
        if (
            row.get("schema_version") != 2
            or row.get("domain") != "day20.qwen35_adapter_prediction"
            or row.get("ordinal") != ordinal
            or row.get("candidate") != candidate
            or row.get("recipe") != candidate
            or row.get("sample_id") != sample_id
            or row.get("slice") != record.get("slice")
            or row.get("reference_hash") != record.get("reference_hash")
            or row.get("prompt_messages_hash") != record.get("messages_hash")
            or row.get("comparison_key") != expected_comparison
            or row.get("run_hash") != expected_run
            or row.get("source_prediction", {}).get("file_sha256")
            != source_predictions_sha
            or row.get("source_evaluation_summary_sha256")
            != source_summary_file_sha
        ):
            raise Day20CodeSandboxError(f"normalized row identity mismatch: {sample_id}")
        expected_row_sha = object_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        if row.get("row_sha256") != expected_row_sha:
            raise Day20CodeSandboxError(f"normalized row hash mismatch: {sample_id}")
        raw_output = row.get("raw_output")
        if not isinstance(raw_output, str) or row.get("raw_output_sha256") != text_sha256(
            raw_output
        ):
            raise Day20CodeSandboxError(f"raw output evidence mismatch: {sample_id}")
        try:
            generated_text = validate_ms_swift_response_capture(
                row.get("response_capture"),
                message_content=raw_output,
                generation=row.get("generation"),
            )
        except (TypeError, ValueError) as error:
            raise Day20CodeSandboxError(
                f"invalid response capture for {sample_id}: {error}"
            ) from error
        adapted = adapt_qwen35_text(
            raw_output, generated_only_text=generated_text
        )
        if row.get("response_adapter") != day19_e2b.normalized_json(adapted):
            raise Day20CodeSandboxError(f"response adapter drifted: {sample_id}")
        if (
            row.get("normalized_output") != adapted["final_text"]
            or row.get("normalized_output_sha256") != adapted["final_text_sha256"]
        ):
            raise Day20CodeSandboxError(f"normalized output drifted: {sample_id}")
        expected_score = day19_e2b.normalized_json(
            frozen_scorers.score_prediction(
                record["slice"], adapted["final_text"], record["reference"]
            )
        )
        if row.get("normalized_scorer_result") != expected_score:
            raise Day20CodeSandboxError(f"normalized scorer drifted: {sample_id}")
        if record["slice"] == "code":
            expected_candidate = day19_e2b.normalized_json(
                classify_code_candidate(
                    adapted["final_text"],
                    entry_point=record["metadata"]["entry_point"],
                    extract_code_completion=frozen_scorers.extract_code_completion,
                )
            )
            expected_static = day19_rescore._code_static_status(
                record.get("raw_prompt", ""),
                record["metadata"]["entry_point"],
                expected_candidate,
            )
            expected_eligible = (
                expected_candidate["execution_eligible"] and expected_static["valid"]
            )
            if (
                row.get("code_candidate") != expected_candidate
                or row.get("code_static_syntax") != expected_static
                or row.get("sandbox_execution_eligible") is not expected_eligible
            ):
                raise Day20CodeSandboxError(
                    f"code execution eligibility drifted: {sample_id}"
                )
    return rows, summary, selected


def effective_contract(
    base_config: dict[str, Any], base_contract_hash: str
) -> dict[str, Any]:
    legacy = day19_e2b.legacy
    config = copy.deepcopy(base_config)
    config.update(
        {
            "domain": "day20.qwen35_humaneval_e2b_contract",
            "schema_version": 1,
            "composition": (
                "solution: source preamble before entry_point + candidate + source_test + "
                "check(entry_point); completion: source_prompt + candidate + source_test + "
                "check(entry_point)"
            ),
            "response_adapter_version": ADAPTER_VERSION,
            "candidate_parser_version": PARSER_VERSION,
            "candidate_composer_version": COMPOSER_VERSION,
            "candidate_acceptance": {
                "solution": "exactly one top-level expected entry-point definition",
                "completion": "indented body contained wholly in a synthetic wrapper",
                "composed_program": "exact composed program must compile",
            },
            "contract_violation_score_policy": (
                "ineligible candidates are not sent to E2B and receive deterministic 0"
            ),
            "implementation_sources": {
                "response_adapter_file_sha256": legacy.file_sha256(
                    RESPONSE_ADAPTER_PATH
                ),
                "code_adapter_file_sha256": legacy.file_sha256(CODE_ADAPTER_PATH),
                "sandbox_wrapper_file_sha256": legacy.file_sha256(WRAPPER_PATH),
            },
            "parent_day10_sandbox_contract_hash": base_contract_hash,
            "parent_day10_evaluator_source_sha256": base_config[
                "evaluator_source_sha256"
            ],
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--frozen-e2b-scorer", required=True, type=Path)
    parser.add_argument("--sandbox-config", required=True, type=Path)
    parser.add_argument("--humaneval-source", required=True, type=Path)
    parser.add_argument("--credential-attestation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--live-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.summary_output.exists():
        raise Day20CodeSandboxError("refusing to overwrite sandbox output")

    legacy = day19_e2b.legacy
    try:
        untrusted_config = json.loads(args.sandbox_config.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day20CodeSandboxError(f"cannot load sandbox contract: {error}") from error
    if (
        not isinstance(untrusted_config, dict)
        or untrusted_config.get("evaluator_source_sha256")
        != legacy.file_sha256(args.frozen_e2b_scorer)
    ):
        raise Day20CodeSandboxError(
            "frozen E2B scorer differs from the sandbox contract"
        )
    frozen = legacy.load_module(args.frozen_e2b_scorer.resolve())
    base_config, base_config_payload = frozen.load_json(
        args.sandbox_config, "sandbox contract"
    )
    frozen.verify_config(base_config)
    base_contract_hash = frozen.semantic_hash(base_config)
    config = effective_contract(base_config, base_contract_hash)
    effective_contract_hash = frozen.semantic_hash(config)
    manifest, manifest_payload = frozen.load_json(args.manifest, "manifest")
    manifest_file_sha = hashlib.sha256(manifest_payload).hexdigest()
    experiment_manifest = load_experiment_manifest(
        args.experiment_manifest,
        eval_manifest_path=args.manifest,
        verify_diagnostic_file=False,
    )
    manifest_by_id, full_code_order = frozen.verify_manifest(
        manifest, manifest_payload, base_config
    )
    frozen_scorers = frozen.load_frozen_scorers(manifest["header"])
    source_rows, source_file_sha = legacy.load_humaneval_source(
        args.humaneval_source, base_config
    )
    predictions, eval_summary, selected = verify_v2_predictions(
        predictions_path=args.predictions,
        summary_path=args.eval_summary,
        manifest=manifest,
        experiment_manifest=experiment_manifest,
        experiment_manifest_path=args.experiment_manifest,
        manifest_file_sha256=manifest_file_sha,
        frozen_scorers=frozen_scorers,
    )
    selected_ids = {record["sample_id"] for record in selected}
    code_order = [
        record["sample_id"] for record in selected if record.get("slice") == "code"
    ]
    if set(code_order) != (set(full_code_order) & selected_ids):
        raise Day20CodeSandboxError("selected Code membership drifted")
    expected_code_records = 8 if len(selected) == 32 else 28
    if len(code_order) != expected_code_records:
        raise Day20CodeSandboxError("selected Code subset count drifted")
    tasks = day19_e2b.build_v2_tasks(
        predictions=predictions,
        summary=eval_summary,
        manifest_by_id=manifest_by_id,
        code_order=code_order,
        source_rows=source_rows,
        frozen=frozen,
    )
    for task in tasks:
        task["prediction"]["model_id"] = (
            f"day20/Qwen3.5-4B-Base+LoRA/{eval_summary['candidate']}"
        )

    predictions_file_sha = legacy.file_sha256(args.predictions)
    comparison = frozen.semantic_hash(
        {
            "domain": "day20.qwen35_adapter_code_comparison",
            "schema_version": 1,
            "manifest_hash": manifest["header"]["manifest_hash"],
            "manifest_file_sha256": manifest_file_sha,
            "experiment_manifest_file_sha256": legacy.file_sha256(
                args.experiment_manifest
            ),
            "adapter_comparison_key": eval_summary["comparison_key"],
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    run = frozen.semantic_hash(
        {
            "domain": "day20.qwen35_adapter_code_run",
            "schema_version": 1,
            "candidate": eval_summary["candidate"],
            "adapter_run_hash": eval_summary["run_hash"],
            "predictions_file_sha256": predictions_file_sha,
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    code_run = frozen.semantic_hash(
        {
            "domain": "day20.qwen35_adapter_e2b_run",
            "schema_version": 1,
            "run_hash": run,
            "sample_ids": code_order,
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    complete_comparison = frozen.semantic_hash(
        {
            "domain": "day20.qwen35_adapter_complete_comparison",
            "schema_version": 1,
            "comparison_key": comparison,
            "effective_code_protocol_hash": effective_contract_hash,
        }
    )
    model_identity_sha = eval_summary["model_identity_sha256"]
    context = {
        "config": config,
        "sandbox_contract_hash": effective_contract_hash,
        "sandbox_contract_file_sha256": hashlib.sha256(base_config_payload).hexdigest(),
        "evaluator_source_sha256": base_config["evaluator_source_sha256"],
        "manifest_hash": manifest["header"]["manifest_hash"],
        "manifest_file_sha256": manifest_file_sha,
        "predictions_file_sha256": predictions_file_sha,
        "source_file_sha256": source_file_sha,
        "complete_comparison_key": complete_comparison,
        "code_run_hash": code_run,
        "run_hash": run,
        "comparison_key": comparison,
        "model_snapshot_hash": model_identity_sha,
    }

    def compose_v2(task: dict[str, Any]) -> str:
        return compose_humaneval_program(
            source_prompt=task["source_prompt"],
            source_test=task["source_test"],
            entry_point=task["entry_point"],
            candidate=task["completion"],
            candidate_mode=task["candidate_mode"],
        )

    frozen.compose_humaneval_program = compose_v2
    credential_attestation = verify_credential_attestation(
        args.credential_attestation
    )
    bindings = frozen.load_e2b_bindings(config)
    if args.live_smoke:
        legacy.live_smoke(frozen, config, bindings)
    results = day19_e2b.score_v2_tasks(
        tasks=tasks, context=context, bindings=bindings, frozen=frozen
    )
    if len(results) != expected_code_records or any(
        row.get("score_status") != "ok" for row in results
    ):
        raise Day20CodeSandboxError("sandbox scoring returned an incomplete result")
    tasks_by_id = {task["sample_id"]: task for task in tasks}
    for row in results:
        task = tasks_by_id[row["sample_id"]]
        row.update(
            {
                "domain": ROW_DOMAIN,
                "schema_version": 1,
                "candidate": eval_summary["candidate"],
                "recipe": eval_summary["candidate"],
                "harness_version": "day20-qwen35-humaneval-e2b-v1",
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
                "effective_code_protocol_hash": effective_contract_hash,
                "normalized_prediction_file_sha256": predictions_file_sha,
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
        row["sandbox_contract_hash"] = effective_contract_hash
        row["code_execution_protocol_hash"] = effective_contract_hash
        row.pop("evaluator_source_sha256", None)
    frozen.write_jsonl_atomic(results, args.output, overwrite=False)
    result_summary = {
        "schema_version": 1,
        "domain": SUMMARY_DOMAIN,
        "status": "complete",
        "candidate": eval_summary["candidate"],
        "recipe": eval_summary["candidate"],
        "records": len(results),
        "passed": sum(row["score"] == 1.0 for row in results),
        "failed": sum(row["score"] == 0.0 for row in results),
        "sandbox_execution_eligible": sum(
            bool(task["execution_eligible"]) for task in tasks
        ),
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
        "prediction_file_sha256": predictions_file_sha,
        "prediction_run_hash": eval_summary["run_hash"],
        "prediction_comparison_key": eval_summary["comparison_key"],
        "manifest_file_sha256": manifest_file_sha,
        "experiment_manifest_file_sha256": legacy.file_sha256(
            args.experiment_manifest
        ),
        "experiment_manifest_content_sha256": experiment_manifest[
            "manifest_sha256"
        ],
        "humaneval_source_file_sha256": source_file_sha,
        "frozen_e2b_scorer_file_sha256": legacy.file_sha256(
            args.frozen_e2b_scorer
        ),
        "response_adapter_file_sha256": legacy.file_sha256(RESPONSE_ADAPTER_PATH),
        "code_adapter_file_sha256": legacy.file_sha256(CODE_ADAPTER_PATH),
        "sandbox_wrapper_file_sha256": legacy.file_sha256(WRAPPER_PATH),
        "base_sandbox_contract_file_sha256": legacy.file_sha256(args.sandbox_config),
        "base_sandbox_contract_hash": base_contract_hash,
        "effective_code_protocol": config,
        "effective_code_protocol_hash": effective_contract_hash,
        "comparison_key": comparison,
        "complete_comparison_key": complete_comparison,
        "run_hash": run,
        "code_run_hash": code_run,
        "model_identity_sha256": model_identity_sha,
        "live_smoke_attested_and_killed": args.live_smoke,
        "candidate_code_executed_on_host": False,
        "credential_handling": "credential value is not intentionally emitted",
        "credential_rotation_attestation": credential_attestation,
    }
    result_summary["summary_sha256"] = object_sha256(result_summary)
    legacy.write_json_new(args.summary_output, result_summary)
    print(json.dumps(result_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
