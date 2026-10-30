#!/usr/bin/env python3
"""Publish a non-destructive Qwen3.5-normalized sidecar for a Day 20 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
DAY19_DIR = HERE.parent / "day-19-qwen35-sft-comparison"
if str(DAY19_DIR) not in sys.path:
    sys.path.insert(0, str(DAY19_DIR))

import rescore_day19_qwen35_v2 as day19_rescore  # noqa: E402
from evaluate_day20 import (  # noqa: E402
    load_experiment_manifest,
    selected_experiment_records,
)
from day20_train_plugin import verified_checkpoint_path  # noqa: E402
from day19_humaneval_adapter import COMPOSER_VERSION, PARSER_VERSION  # noqa: E402
from qwen35_response_adapter import (  # noqa: E402
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    text_sha256,
    validate_ms_swift_response_capture,
)


SKILLS = ("general", "math", "code", "finance")
SUMMARY_DOMAIN = "day20.qwen35_adapter_summary"
ROW_DOMAIN = "day20.qwen35_adapter_prediction"
RESPONSE_ADAPTER_PATH = DAY19_DIR / "qwen35_response_adapter.py"
CODE_ADAPTER_PATH = DAY19_DIR / "day19_humaneval_adapter.py"
PARENT_RESCORER_PATH = DAY19_DIR / "rescore_day19_qwen35_v2.py"
RESCORER_PATH = Path(__file__).resolve()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Day20Qwen35RescoreError(ValueError):
    """A source prediction or normalized sidecar invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_files(
    files: Any, *, required_config: str, require_safetensors: bool = True
) -> str:
    if not isinstance(files, dict) or required_config not in files:
        raise Day20Qwen35RescoreError("model file manifest is incomplete")
    if require_safetensors and not any(
        isinstance(name, str) and name.endswith(".safetensors") for name in files
    ):
        raise Day20Qwen35RescoreError("model file manifest lacks safetensors")
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
            raise Day20Qwen35RescoreError(f"invalid model file identity: {name!r}")
    return object_sha256(files)


def verify_model_identity(identity: Any) -> str:
    if not isinstance(identity, dict) or identity.get("load_and_generation_verified") is not True:
        raise Day20Qwen35RescoreError("source model identity is incomplete")
    base = identity.get("base")
    if (
        not isinstance(base, dict)
        or base.get("model_class") != "Qwen3_5ForConditionalGeneration"
    ):
        raise Day20Qwen35RescoreError("source Base identity is incomplete")
    base_snapshot = _verify_files(base.get("files"), required_config="config.json")
    if base.get("snapshot_sha256") != base_snapshot:
        raise Day20Qwen35RescoreError("source Base snapshot drifted")
    adapter = identity.get("adapter")
    adapter_snapshot = None
    if adapter is not None:
        if not isinstance(adapter, dict) or adapter.get("format") != "peft_lora":
            raise Day20Qwen35RescoreError("source adapter identity is incomplete")
        adapter_snapshot = _verify_files(
            adapter.get("files"), required_config="adapter_config.json"
        )
        if adapter.get("snapshot_sha256") != adapter_snapshot:
            raise Day20Qwen35RescoreError("source adapter snapshot drifted")
        checkpoint_snapshot = _verify_files(
            adapter.get("checkpoint_package_files"),
            required_config="adapter_config.json",
        )
        if adapter.get("checkpoint_package_snapshot_sha256") != checkpoint_snapshot:
            raise Day20Qwen35RescoreError("adapter checkpoint package drifted")
    combined = object_sha256(
        {
            "domain": "day20.qwen35_base_adapter_identity",
            "schema_version": 1,
            "base_snapshot_sha256": base_snapshot,
            "adapter_snapshot_sha256": adapter_snapshot,
        }
    )
    if identity.get("combined_snapshot_sha256") != combined:
        raise Day20Qwen35RescoreError("combined Base+Adapter identity drifted")
    return combined


def comparison_key(
    *, manifest_sha: str, experiment_manifest_sha: str, scorer_sha: str
) -> str:
    return object_sha256(
        {
            "domain": "day20.qwen35_adapter_comparison",
            "schema_version": 1,
            "manifest_file_sha256": manifest_sha,
            "experiment_manifest_file_sha256": experiment_manifest_sha,
            "scorer_file_sha256": scorer_sha,
            "response_adapter_version": ADAPTER_VERSION,
            "response_adapter_file_sha256": day19_rescore.file_sha256(
                RESPONSE_ADAPTER_PATH
            ),
            "code_parser_version": PARSER_VERSION,
            "code_composer_version": COMPOSER_VERSION,
            "code_adapter_file_sha256": day19_rescore.file_sha256(CODE_ADAPTER_PATH),
            "parent_rescorer_file_sha256": day19_rescore.file_sha256(
                PARENT_RESCORER_PATH
            ),
            "rescorer_file_sha256": day19_rescore.file_sha256(RESCORER_PATH),
        }
    )


def run_hash(
    *,
    candidate: str,
    comparison: str,
    checkpoint: str,
    model_identity_sha: str,
    source_predictions_sha: str,
    source_summary_file_sha: str,
    source_summary_content_sha: str,
) -> str:
    return object_sha256(
        {
            "domain": "day20.qwen35_adapter_run",
            "schema_version": 1,
            "candidate": candidate,
            "comparison_key": comparison,
            "checkpoint": checkpoint,
            "model_identity_sha256": model_identity_sha,
            "source_prediction_file_sha256": source_predictions_sha,
            "source_evaluation_summary_file_sha256": source_summary_file_sha,
            "source_evaluation_summary_content_sha256": source_summary_content_sha,
        }
    )


def verify_source(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    experiment_manifest: dict[str, Any],
    scorers: ModuleType,
    predictions_path: Path,
    manifest_path: Path,
    scorers_path: Path,
    experiment_manifest_path: Path,
) -> tuple[str, list[dict[str, Any]], str]:
    expected_summary_sha = object_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    candidate = summary.get("candidate")
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_lora_eval_summary"
        or summary.get("status") != "complete_with_code_sandbox_required"
        or summary.get("summary_sha256") != expected_summary_sha
        or not isinstance(candidate, str)
        or not _CANDIDATE_RE.fullmatch(candidate)
        or summary.get("recipe") != candidate
    ):
        raise Day20Qwen35RescoreError("source evaluation summary identity drifted")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise Day20Qwen35RescoreError("source checkpoint identity is incomplete")
    model_identity = summary.get("model_identity")
    model_identity_sha = verify_model_identity(model_identity)
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise Day20Qwen35RescoreError("source evaluation protocol is missing")
    sample_limit = protocol.get("sample_limit")
    if sample_limit not in (None, 32):
        raise Day20Qwen35RescoreError("only the 32-row probe or full dev run is valid")
    expected_records = 32 if sample_limit == 32 else 112
    is_base = candidate in {"base", "base-probe"}
    expected_candidates = (
        {"base-probe", "probe-1e-5", "probe-3e-5", "probe-1e-4"}
        if sample_limit == 32
        else {"base", "early", "mid", "final"}
    )
    adapter_identity = model_identity.get("adapter")
    training_identity = summary.get("training_identity")
    if (
        candidate not in expected_candidates
        or summary.get("model_identity_post_run_unchanged") is not True
        or (adapter_identity is None) is not is_base
        or (training_identity is None) is not is_base
        or protocol.get("adapter_loaded_unmerged") is not (not is_base)
    ):
        raise Day20Qwen35RescoreError("candidate/model/training identity drifted")
    if not is_base:
        if not isinstance(training_identity, dict):
            raise Day20Qwen35RescoreError("training identity is missing")
        training_summary_path = Path(str(training_identity.get("path", ""))).resolve()
        checkpoint_label = "probe" if sample_limit == 32 else candidate
        training_summary = day19_rescore.load_json(training_summary_path)
        if (
            day19_rescore.file_sha256(training_summary_path)
            != training_identity.get("file_sha256")
            or training_identity.get("checkpoint_label")
            != checkpoint_label
            or training_identity.get("checkpoint_package")
            != (training_summary.get("checkpoint_packages") or {}).get(
                checkpoint_label
            )
        ):
            raise Day20Qwen35RescoreError("training summary provenance drifted")
        verified_checkpoint = verified_checkpoint_path(
            training_summary_path,
            name=checkpoint_label,
            run_root=Path(str(experiment_manifest.get("run_root", ""))),
        )
        adapter_path = Path(str(adapter_identity.get("path", ""))).resolve()
        if (
            verified_checkpoint != Path(checkpoint).resolve()
            or verified_checkpoint != adapter_path
        ):
            raise Day20Qwen35RescoreError(
                "training checkpoint package differs from evaluated adapter"
            )
    predictions = summary.get("predictions")
    source_predictions_sha = day19_rescore.file_sha256(predictions_path)
    if (
        not isinstance(predictions, dict)
        or predictions.get("records") != expected_records
        or predictions.get("file_sha256") != source_predictions_sha
    ):
        raise Day20Qwen35RescoreError("source prediction identity differs from summary")
    manifest_sha = day19_rescore.file_sha256(manifest_path)
    scorer_sha = day19_rescore.file_sha256(scorers_path)
    if not isinstance(protocol, dict) or any(
        (
            protocol.get("eval_manifest_file_sha256") != manifest_sha,
            protocol.get("experiment_manifest_file_sha256")
            != day19_rescore.file_sha256(experiment_manifest_path),
            protocol.get("experiment_manifest_content_sha256")
            != experiment_manifest.get("manifest_sha256"),
            protocol.get("scorer_source_sha256") != scorer_sha,
            manifest.get("header", {}).get("scorer_source_sha256") != scorer_sha,
            protocol.get("split") != "dev",
            protocol.get("frozen_test_consumed") is not False,
            protocol.get("records") != expected_records,
            protocol.get("template") != "qwen3_5",
            protocol.get("enable_thinking") is not False,
            protocol.get("add_non_thinking_prefix") is not True,
            protocol.get("response_capture")
            != "ms-swift RequestConfig(return_details=True)",
            protocol.get("response_boundary_adapter") != ADAPTER_VERSION,
            protocol.get("generated_token_ids_retained") is not True,
            protocol.get("non_thinking_prefix_sha256")
            != text_sha256(NON_THINKING_PREFIX),
            not isinstance(protocol.get("ms_swift_version"), str),
            not protocol.get("ms_swift_version"),
        )
    ):
        raise Day20Qwen35RescoreError("source evaluation protocol drifted")
    order = manifest.get("header", {}).get("evaluation_order", {}).get("dev")
    records = manifest.get("records")
    if not isinstance(order, list) or not isinstance(records, list):
        raise Day20Qwen35RescoreError("frozen eval manifest is malformed")
    if len(order) != 112 or len(set(order)) != 112:
        raise Day20Qwen35RescoreError("frozen manifest requires 112 unique dev rows")
    try:
        verified_records = selected_experiment_records(
            eval_manifest=manifest,
            experiment_manifest=experiment_manifest,
            sample_limit=sample_limit,
        )
    except ValueError as error:
        raise Day20Qwen35RescoreError(str(error)) from error
    expected_diagnostic_sha = (
        experiment_manifest["datasets"]["diagnostic"]["selection_sha256"]
        if sample_limit == 32
        else None
    )
    if protocol.get("diagnostic_selection_sha256") != expected_diagnostic_sha:
        raise Day20Qwen35RescoreError("source diagnostic selection drifted")
    if len(rows) != len(verified_records) or Counter(
        record.get("slice") for record in verified_records
    ) != Counter({skill: expected_records // len(SKILLS) for skill in SKILLS}):
        raise Day20Qwen35RescoreError("frozen dev slice distribution drifted")
    for ordinal, (row, record) in enumerate(zip(rows, verified_records), 1):
        sample_id = record["sample_id"]
        if (
            row.get("schema_version") != 1
            or row.get("domain") != "day20.qwen35_lora_eval_prediction"
            or row.get("ordinal") != ordinal
            or row.get("candidate") != candidate
            or row.get("recipe") != candidate
            or row.get("sample_id") != sample_id
            or row.get("slice") != record.get("slice")
            or row.get("reference_hash") != record.get("reference_hash")
            or row.get("prompt_messages_hash") != record.get("messages_hash")
        ):
            raise Day20Qwen35RescoreError(
                f"source prediction identity mismatch: {sample_id}"
            )
        raw_output = row.get("raw_output")
        if not isinstance(raw_output, str) or row.get("raw_output_sha256") != text_sha256(
            raw_output
        ):
            raise Day20Qwen35RescoreError(
                f"source output evidence mismatch: {sample_id}"
            )
        expected_score = day19_rescore.normalized_json(
            scorers.score_prediction(record["slice"], raw_output, record["reference"])
        )
        if row.get("scorer_result") != expected_score:
            raise Day20Qwen35RescoreError(f"source scorer drifted: {sample_id}")
        capture = row.get("response_capture")
        try:
            validate_ms_swift_response_capture(
                capture,
                message_content=raw_output,
                generation=row.get("generation"),
            )
        except (TypeError, ValueError) as error:
            raise Day20Qwen35RescoreError(
                f"invalid token-backed response capture for {sample_id}: {error}"
            ) from error
    return candidate, verified_records, model_identity_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--scorers", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {
        args.manifest.resolve(),
        args.experiment_manifest.resolve(),
        args.scorers.resolve(),
        args.predictions.resolve(),
        args.eval_summary.resolve(),
    }
    if (
        args.output.resolve() in inputs
        or args.summary_output.resolve() in inputs
        or args.output.resolve() == args.summary_output.resolve()
    ):
        raise Day20Qwen35RescoreError("sidecars must be distinct from all inputs")
    if args.output.exists() or args.summary_output.exists():
        raise Day20Qwen35RescoreError("refusing to overwrite an existing sidecar")

    manifest = day19_rescore.load_json(args.manifest)
    experiment_manifest = load_experiment_manifest(
        args.experiment_manifest,
        eval_manifest_path=args.manifest,
        verify_diagnostic_file=True,
    )
    summary = day19_rescore.load_json(args.eval_summary)
    source_rows = day19_rescore.load_jsonl(args.predictions)
    expected_scorer_sha = manifest.get("header", {}).get("scorer_source_sha256")
    if (
        not isinstance(expected_scorer_sha, str)
        or day19_rescore.file_sha256(args.scorers) != expected_scorer_sha
    ):
        raise Day20Qwen35RescoreError("frozen scorer source identity drifted")
    scorers = day19_rescore.load_module(
        args.scorers.resolve(), "day20_qwen35_frozen_scorers"
    )
    if not callable(getattr(scorers, "score_prediction", None)) or not callable(
        getattr(scorers, "extract_code_completion", None)
    ):
        raise Day20Qwen35RescoreError("frozen scorer API is incomplete")
    candidate, records, model_identity_sha = verify_source(
        rows=source_rows,
        summary=summary,
        manifest=manifest,
        experiment_manifest=experiment_manifest,
        scorers=scorers,
        predictions_path=args.predictions,
        manifest_path=args.manifest,
        scorers_path=args.scorers,
        experiment_manifest_path=args.experiment_manifest,
    )
    source_predictions_sha = day19_rescore.file_sha256(args.predictions)
    source_summary_file_sha = day19_rescore.file_sha256(args.eval_summary)
    rows = day19_rescore.build_sidecar_rows(
        source_rows=source_rows,
        records=records,
        recipe=candidate,
        source_prediction_sha256=source_predictions_sha,
        source_summary_sha256=source_summary_file_sha,
        scorers=scorers,
    )
    manifest_sha = day19_rescore.file_sha256(args.manifest)
    scorer_sha = day19_rescore.file_sha256(args.scorers)
    experiment_manifest_file_sha = day19_rescore.file_sha256(
        args.experiment_manifest
    )
    comparison = comparison_key(
        manifest_sha=manifest_sha,
        experiment_manifest_sha=experiment_manifest_file_sha,
        scorer_sha=scorer_sha,
    )
    run = run_hash(
        candidate=candidate,
        comparison=comparison,
        checkpoint=summary["checkpoint"],
        model_identity_sha=model_identity_sha,
        source_predictions_sha=source_predictions_sha,
        source_summary_file_sha=source_summary_file_sha,
        source_summary_content_sha=summary["summary_sha256"],
    )
    for row in rows:
        row["domain"] = ROW_DOMAIN
        row["candidate"] = candidate
        row["comparison_key"] = comparison
        row["run_hash"] = run
        row["row_sha256"] = object_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    day19_rescore.write_jsonl_new(args.output, rows)
    metrics = day19_rescore.metrics_for_rows(rows)
    if not math.isfinite(float(metrics["non_code_accuracy"])):
        raise Day20Qwen35RescoreError("non-finite normalized metric")
    sidecar_summary = {
        "schema_version": 1,
        "domain": SUMMARY_DOMAIN,
        "status": "complete_with_code_sandbox_required",
        "created_at_utc": utc_now(),
        "candidate": candidate,
        "recipe": candidate,
        "checkpoint": summary["checkpoint"],
        "model_identity": summary["model_identity"],
        "model_identity_sha256": model_identity_sha,
        "training_identity": summary.get("training_identity"),
        "comparison_key": comparison,
        "run_hash": run,
        "protocol": {
            "response_adapter_version": ADAPTER_VERSION,
            "response_adapter_file_sha256": day19_rescore.file_sha256(
                RESPONSE_ADAPTER_PATH
            ),
            "code_parser_version": PARSER_VERSION,
            "code_composer_version": COMPOSER_VERSION,
            "code_adapter_file_sha256": day19_rescore.file_sha256(CODE_ADAPTER_PATH),
            "parent_rescorer_file_sha256": day19_rescore.file_sha256(
                PARENT_RESCORER_PATH
            ),
            "rescorer_file_sha256": day19_rescore.file_sha256(RESCORER_PATH),
            "legacy_scorer_semantics_preserved": True,
            "candidate_code_executed_on_host": False,
            "frozen_test_consumed": False,
            "source_prediction_file_sha256": source_predictions_sha,
            "source_evaluation_summary_sha256": source_summary_file_sha,
            "source_evaluation_summary_content_sha256": summary["summary_sha256"],
            "source_evaluation_protocol": {
                key: summary["protocol"].get(key)
                for key in (
                    "records",
                    "sample_limit",
                    "experiment_manifest_file_sha256",
                    "experiment_manifest_content_sha256",
                    "diagnostic_selection_sha256",
                    "template",
                    "backend",
                    "use_mcore_gdn",
                    "enable_thinking",
                    "add_non_thinking_prefix",
                    "response_capture",
                    "response_boundary_adapter",
                    "generated_token_ids_retained",
                    "ms_swift_version",
                    "non_thinking_prefix_sha256",
                )
            },
            "manifest_file_sha256": manifest_sha,
            "experiment_manifest_file_sha256": experiment_manifest_file_sha,
            "experiment_manifest_content_sha256": experiment_manifest[
                "manifest_sha256"
            ],
            "scorer_file_sha256": scorer_sha,
        },
        "metrics": metrics,
        "predictions": {
            "path": str(args.output.resolve()),
            "records": len(rows),
            "file_sha256": day19_rescore.file_sha256(args.output),
        },
    }
    sidecar_summary["summary_sha256"] = object_sha256(sidecar_summary)
    day19_rescore.write_json_new(args.summary_output, sidecar_summary)
    print(json.dumps(sidecar_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
