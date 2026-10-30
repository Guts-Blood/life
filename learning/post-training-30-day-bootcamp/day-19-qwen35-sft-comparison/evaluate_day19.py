#!/usr/bin/env python3
"""Evaluate one Day 19 HF export on the frozen Day 10 development suite."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen35_response_adapter import (
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    adapt_qwen35_text,
    text_sha256,
    token_ids_sha256,
)


SKILLS = ("general", "math", "code", "finance")


class Day19EvaluationError(ValueError):
    """An evaluation input or runtime invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day19EvaluationError(f"required file is missing: {path}")
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
        raise Day19EvaluationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day19EvaluationError(f"JSON root must be an object: {path}")
    return value


def load_scorers(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("day19_frozen_scorers", path)
    if spec is None or spec.loader is None:
        raise Day19EvaluationError(f"cannot import scorers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "score_prediction", None)):
        raise Day19EvaluationError("frozen scorer has no score_prediction")
    return module


def model_file_manifest(model_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            result[path.relative_to(model_dir).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    if "config.json" not in result or not any(name.endswith(".safetensors") for name in result):
        raise Day19EvaluationError("HF export is incomplete")
    return result


def selected_dev_records(manifest: dict[str, Any], sample_limit: int | None) -> list[dict[str, Any]]:
    records = manifest.get("records")
    order = manifest.get("header", {}).get("evaluation_order", {}).get("dev")
    if not isinstance(records, list) or not isinstance(order, list):
        raise Day19EvaluationError("frozen eval manifest is malformed")
    by_id = {row.get("sample_id"): row for row in records if isinstance(row, dict)}
    if len(order) != 112 or len(set(order)) != 112:
        raise Day19EvaluationError("frozen dev order must contain 112 unique records")
    selected = [by_id[sample_id] for sample_id in order]
    if any(row.get("evaluation_split") != "dev" for row in selected):
        raise Day19EvaluationError("non-dev record reached Day 19 evaluation")
    if Counter(row.get("slice") for row in selected) != Counter({skill: 28 for skill in SKILLS}):
        raise Day19EvaluationError("frozen dev slice distribution drifted")
    if sample_limit is None:
        return selected
    if sample_limit <= 0 or sample_limit > len(selected):
        raise Day19EvaluationError("sample_limit is outside the dev suite")
    buckets = {skill: [row for row in selected if row["slice"] == skill] for skill in SKILLS}
    result: list[dict[str, Any]] = []
    offsets = {skill: 0 for skill in SKILLS}
    while len(result) < sample_limit:
        for skill in SKILLS:
            if offsets[skill] < len(buckets[skill]):
                result.append(buckets[skill][offsets[skill]])
                offsets[skill] += 1
                if len(result) == sample_limit:
                    break
    return result


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def code_syntax_status(raw_prompt: str, completion: str) -> dict[str, Any]:
    try:
        ast.parse(raw_prompt + completion)
    except (SyntaxError, ValueError, TypeError) as error:
        return {"valid": False, "error": type(error).__name__}
    return {"valid": True, "error": None}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--scorers", required=True, type=Path)
    parser.add_argument(
        "--recipe",
        required=True,
        choices=("baseline-a", "baseline-b", "best-e", "untouched-c0"),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(".predictions.jsonl").exists():
        raise Day19EvaluationError("refusing to overwrite existing evaluation output")
    manifest = load_json(args.eval_manifest)
    selected = selected_dev_records(manifest, args.sample_limit)
    scorers = load_scorers(args.scorers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    predictions_attempt = predictions_path.with_name(
        f".{predictions_path.name}.{os.getpid()}.attempt"
    )
    if predictions_attempt.exists():
        raise Day19EvaluationError(
            f"refusing to overwrite evaluation attempt: {predictions_attempt}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    os.environ.setdefault("USE_MCORE_GDN", "1")
    import torch
    from swift import get_model_processor, get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    torch.cuda.reset_peak_memory_stats()
    model, processor = get_model_processor(
        str(args.model),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise Day19EvaluationError(f"wrong Qwen3.5 loader: {model.__class__.__name__}")
    template = get_template(
        processor,
        max_length=4096,
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    template_meta = getattr(template, "template_meta", None)
    if getattr(template_meta, "template_type", None) != "qwen3_5":
        raise Day19EvaluationError("ms-swift did not construct the qwen3_5 template")
    if getattr(template_meta, "non_thinking_prefix", None) != NON_THINKING_PREFIX:
        raise Day19EvaluationError("Qwen3.5 non-thinking response prefix drifted")
    if getattr(template, "enable_thinking", None) is not False:
        raise Day19EvaluationError("Qwen3.5 template did not disable thinking")
    if getattr(template, "add_non_thinking_prefix", None) is not True:
        raise Day19EvaluationError("Qwen3.5 template disabled the non-thinking prefix")
    try:
        ms_swift_version = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError as error:
        raise Day19EvaluationError("cannot identify the installed ms-swift version") from error
    engine = TransformersEngine(model, template=template, max_batch_size=1)

    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for ordinal, record in enumerate(selected, 1):
        slice_name = record["slice"]
        maximum = int(record["generation_max_new_tokens"])
        response = engine.infer(
            [InferRequest(messages=record["messages"])],
            RequestConfig(
                max_tokens=maximum,
                temperature=0,
                num_beams=1,
                repetition_penalty=1.0,
                seed=20260807,
                return_details=True,
            ),
            use_tqdm=False,
        )[0]
        choice = response.choices[0]
        raw_output = choice.message.content or ""
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        generated_token_ids = getattr(choice, "token_ids", None)
        if not isinstance(prompt_token_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in prompt_token_ids
        ):
            raise Day19EvaluationError(
                f"ms-swift did not return prompt token IDs: {record['sample_id']}"
            )
        if not isinstance(generated_token_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in generated_token_ids
        ):
            raise Day19EvaluationError(
                f"ms-swift did not return generated token IDs: {record['sample_id']}"
            )
        generated_only_output = template.decode_generate_ids(
            generated_token_ids, first_token=False
        )
        if not isinstance(generated_only_output, str):
            raise Day19EvaluationError(
                f"generated-only decode is not text: {record['sample_id']}"
            )
        boundary = adapt_qwen35_text(
            raw_output, generated_only_text=generated_only_output
        )
        scorer_result = scorers.score_prediction(slice_name, raw_output, record["reference"])
        completion_tokens = int(response.usage.completion_tokens)
        if completion_tokens != len(generated_token_ids):
            raise Day19EvaluationError(
                f"completion token count mismatch: {record['sample_id']}"
            )
        if int(response.usage.prompt_tokens) != len(prompt_token_ids):
            raise Day19EvaluationError(
                f"prompt token count mismatch: {record['sample_id']}"
            )
        anomalies = []
        if not raw_output.strip():
            anomalies.append("empty_output")
        if choice.finish_reason == "length" or completion_tokens >= maximum:
            anomalies.append("generation_ceiling")
        repetition_ratio = repeated_ngram_ratio(raw_output)
        if repetition_ratio >= 0.5:
            anomalies.append("high_4gram_repetition")
        syntax = None
        if slice_name == "code":
            syntax = code_syntax_status(record.get("raw_prompt", ""), scorer_result["parsed_answer"])
            if not syntax["valid"]:
                anomalies.append("code_syntax_invalid")
        format_compliant = (
            bool(raw_output.strip())
            and scorer_result.get("parse_status") == "ok"
            and (syntax is None or syntax["valid"])
        )
        row = {
            "schema_version": 1,
            "ordinal": ordinal,
            "recipe": args.recipe,
            "sample_id": record["sample_id"],
            "slice": slice_name,
            "raw_output": raw_output,
            "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            "response_capture": {
                "method": "ms_swift_return_details",
                "message_content_semantics": (
                    "ms-swift template decode; the configured response prefix may be "
                    "reconstructed ahead of generated token text"
                ),
                "response_adapter_verification": {
                    "adapter_version": boundary["adapter_version"],
                    "prefix_status": boundary["prefix_status"],
                    "final_text_sha256": boundary["final_text_sha256"],
                },
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_ids_sha256": token_ids_sha256(prompt_token_ids),
                "generated_token_ids": generated_token_ids,
                "generated_token_ids_sha256": token_ids_sha256(generated_token_ids),
                "generated_only_text": generated_only_output,
                "generated_only_text_sha256": text_sha256(generated_only_output),
            },
            "reference_hash": record["reference_hash"],
            "prompt_messages_hash": record["messages_hash"],
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": maximum,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": int(response.usage.prompt_tokens),
                "completion_tokens": completion_tokens,
            },
            "scorer_result": scorer_result,
            "format_compliant": format_compliant,
            "code_syntax": syntax,
            "anomalies": anomalies,
            "repeated_4gram_ratio": repetition_ratio,
        }
        rows.append(row)
        append_jsonl(predictions_attempt, row)
        print(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "sample_id": record["sample_id"],
                    "slice": slice_name,
                    "score": scorer_result.get("score"),
                    "format_compliant": format_compliant,
                    "anomalies": anomalies,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    elapsed = time.monotonic() - start
    by_slice: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["slice"] == skill]
        scored = [row for row in skill_rows if isinstance(row["scorer_result"].get("score"), (int, float))]
        correct = sum(float(row["scorer_result"]["score"]) for row in scored)
        by_slice[skill] = {
            "records": len(skill_rows),
            "scored_records": len(scored),
            "correct": correct if scored else None,
            "accuracy": correct / len(scored) if scored else None,
            "format_compliant": sum(bool(row["format_compliant"]) for row in skill_rows),
            "format_compliance_rate": sum(bool(row["format_compliant"]) for row in skill_rows) / len(skill_rows),
            "anomalous_records": sum(bool(row["anomalies"]) for row in skill_rows),
            "generation_ceiling_records": sum("generation_ceiling" in row["anomalies"] for row in skill_rows),
            "output_tokens": sum(row["generation"]["completion_tokens"] for row in skill_rows),
        }
        if skill == "code":
            by_slice[skill]["syntax_valid"] = sum(bool(row["code_syntax"]["valid"]) for row in skill_rows)
            by_slice[skill]["executable_score_status"] = "sandbox_required"

    non_code = [row for row in rows if row["slice"] != "code"]
    non_code_correct = sum(float(row["scorer_result"]["score"]) for row in non_code)
    model_files = model_file_manifest(args.model)
    summary = {
        "schema_version": 1,
        "domain": "day19.qwen35_eval_summary",
        "status": "complete_with_code_sandbox_required",
        "created_at_utc": utc_now(),
        "recipe": args.recipe,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_export": {
            "path": str(args.model.resolve()),
            "model_class": model.__class__.__name__,
            "files": model_files,
            "snapshot_sha256": object_sha256(model_files),
            "load_and_generation_verified": True,
        },
        "protocol": {
            "eval_manifest": str(args.eval_manifest.resolve()),
            "eval_manifest_file_sha256": file_sha256(args.eval_manifest),
            "scorer_source": str(args.scorers.resolve()),
            "scorer_source_sha256": file_sha256(args.scorers),
            "split": "dev",
            "frozen_test_consumed": False,
            "records": len(rows),
            "sample_limit": args.sample_limit,
            "template": "qwen3_5",
            "enable_thinking": False,
            "add_non_thinking_prefix": True,
            "non_thinking_prefix_sha256": text_sha256(NON_THINKING_PREFIX),
            "ms_swift_version": ms_swift_version,
            "greedy": True,
            "seed": 20260807,
            "response_capture": "ms-swift RequestConfig(return_details=True)",
            "response_boundary_adapter": ADAPTER_VERSION,
            "generated_token_ids_retained": True,
            "code_execution": "not executed on host; frozen scorer remains sandbox_required because no approved E2B runtime is present",
        },
        "metrics": {
            "by_slice": by_slice,
            "non_code_correct": non_code_correct,
            "non_code_total": len(non_code),
            "non_code_accuracy": non_code_correct / len(non_code),
            "format_compliant": sum(bool(row["format_compliant"]) for row in rows),
            "format_compliance_rate": sum(bool(row["format_compliant"]) for row in rows) / len(rows),
            "anomalous_records": sum(bool(row["anomalies"]) for row in rows),
            "anomaly_rate": sum(bool(row["anomalies"]) for row in rows) / len(rows),
            "output_tokens": sum(row["generation"]["completion_tokens"] for row in rows),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "predictions": {
            "path": str(predictions_path),
            "records": len(rows),
            "file_sha256": file_sha256(predictions_attempt),
        },
    }
    if not math.isfinite(summary["metrics"]["peak_cuda_memory_gib"]):
        raise Day19EvaluationError("non-finite peak CUDA memory")
    summary["summary_sha256"] = object_sha256(summary)
    try:
        os.link(predictions_attempt, predictions_path)
    except FileExistsError as error:
        raise Day19EvaluationError(
            f"refusing to overwrite prediction artifact: {predictions_path}"
        ) from error
    predictions_attempt.unlink()
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
