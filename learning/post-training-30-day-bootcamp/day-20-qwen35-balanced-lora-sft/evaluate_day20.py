#!/usr/bin/env python3
"""Evaluate a Qwen3.5 Base or Base+LoRA candidate on the frozen dev suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DAY19_DIR = HERE.parent / "day-19-qwen35-sft-comparison"
if str(DAY19_DIR) not in sys.path:
    sys.path.insert(0, str(DAY19_DIR))

import evaluate_day19 as day19_eval  # noqa: E402
from qwen35_response_adapter import (  # noqa: E402
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    adapt_qwen35_text,
    text_sha256,
    token_ids_sha256,
)
from day20_train_plugin import verified_checkpoint_path  # noqa: E402


SKILLS = ("general", "math", "code", "finance")
DOMAIN = "day20.qwen35_lora_eval_summary"
_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PROBE_CANDIDATES = {
    "probe-1e-5": 1e-5,
    "probe-3e-5": 3e-5,
    "probe-1e-4": 1e-4,
}
MAIN_CHECKPOINT_TOKENS = {"early": 64_000, "mid": 153_600, "final": 256_000}


class Day20EvaluationError(ValueError):
    """An evaluation input or runtime invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20EvaluationError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_evaluation_pair(
    *,
    predictions_attempt: Path,
    predictions_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> None:
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{summary_path.name}.", suffix=".attempt", dir=summary_path.parent
    )
    temporary_summary = Path(temporary_name)
    prediction_published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(predictions_attempt, predictions_path)
        prediction_published = True
        try:
            os.link(temporary_summary, summary_path)
        except BaseException:
            predictions_path.unlink(missing_ok=False)
            prediction_published = False
            raise
        predictions_attempt.unlink()
        temporary_summary.unlink()
    except FileExistsError as error:
        raise Day20EvaluationError("refusing to overwrite evaluation artifact") from error
    finally:
        if temporary_summary.exists():
            temporary_summary.unlink()
        if prediction_published and not summary_path.exists():
            predictions_path.unlink(missing_ok=True)


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_experiment_manifest(
    path: Path,
    *,
    eval_manifest_path: Path,
    verify_diagnostic_file: bool,
) -> dict[str, Any]:
    value = day19_eval.load_json(path)
    expected_hash = object_sha256(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )
    contract = value.get("contract")
    immutable = contract.get("immutable") if isinstance(contract, dict) else None
    diagnostic = value.get("datasets", {}).get("diagnostic")
    base_identity = value.get("base_model_identity")
    base_files = base_identity.get("files") if isinstance(base_identity, dict) else None
    if (
        value.get("schema_version") != 1
        or value.get("domain")
        != "day20.qwen35_balanced_lora.experiment_manifest"
        or value.get("status") != "prepared_probes_not_started"
        or value.get("manifest_sha256") != expected_hash
        or not isinstance(immutable, dict)
        or contract.get("immutable_sha256") != object_sha256(immutable)
        or not isinstance(diagnostic, dict)
        or diagnostic.get("records") != 32
        or diagnostic.get("records_by_skill") != {skill: 8 for skill in immutable["data"]["skills"]}
        or diagnostic.get("source_eval_manifest_sha256")
        != file_sha256(eval_manifest_path)
        or not isinstance(base_identity, dict)
        or base_identity.get("path") != immutable["model"]["snapshot"]
        or not isinstance(base_files, dict)
        or "config.json" not in base_files
        or not any(
            isinstance(name, str) and name.endswith(".safetensors")
            for name in base_files
        )
        or base_identity.get("snapshot_sha256") != object_sha256(base_files)
    ):
        raise Day20EvaluationError("Day 20 experiment manifest identity drifted")
    sample_ids = diagnostic.get("ordered_sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != 32
        or len(set(sample_ids)) != 32
        or diagnostic.get("selection_sha256") != object_sha256(sample_ids)
    ):
        raise Day20EvaluationError("Day 20 diagnostic selection identity drifted")
    if verify_diagnostic_file:
        diagnostic_path = Path(str(diagnostic.get("path", ""))).resolve()
        if file_sha256(diagnostic_path) != diagnostic.get("file_sha256"):
            raise Day20EvaluationError("Day 20 diagnostic file identity drifted")
    return value


def selected_experiment_records(
    *,
    eval_manifest: dict[str, Any],
    experiment_manifest: dict[str, Any],
    sample_limit: int | None,
    verify_diagnostic_rows: bool = True,
) -> list[dict[str, Any]]:
    if sample_limit not in (None, 32):
        raise Day20EvaluationError("only the frozen 32-row probe or full dev run is valid")
    full = day19_eval.selected_dev_records(eval_manifest, None)
    if sample_limit is None:
        return full
    diagnostic = experiment_manifest["datasets"]["diagnostic"]
    by_id = {record["sample_id"]: record for record in full}
    try:
        selected = [by_id[sample_id] for sample_id in diagnostic["ordered_sample_ids"]]
    except KeyError as error:
        raise Day20EvaluationError(
            f"diagnostic sample is outside frozen dev: {error}"
        ) from error
    if Counter(record["slice"] for record in selected) != Counter(
        {skill: 8 for skill in SKILLS}
    ):
        raise Day20EvaluationError("diagnostic slice distribution drifted")
    if not verify_diagnostic_rows:
        return selected
    diagnostic_path = Path(diagnostic["path"]).resolve()
    diagnostic_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        diagnostic_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise Day20EvaluationError(
                f"blank diagnostic row: {diagnostic_path}:{line_number}"
            )
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Day20EvaluationError("diagnostic row is not an object")
        diagnostic_rows.append(value)
    if diagnostic_rows != selected:
        raise Day20EvaluationError("diagnostic records differ from frozen eval manifest")
    return selected


def file_manifest(
    directory: Path, *, kind: str, inference_only: bool = False
) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise Day20EvaluationError(f"{kind} directory is missing: {directory}")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if inference_only and not (
                relative == "adapter_config.json"
                or relative.endswith(".safetensors")
                or relative.endswith(".safetensors.index.json")
            ):
                continue
            result[relative] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    if kind == "base" and (
        "config.json" not in result
        or not any(name.endswith(".safetensors") for name in result)
    ):
        raise Day20EvaluationError("Base model file manifest is incomplete")
    if kind == "adapter" and (
        "adapter_config.json" not in result
        or not any(name.endswith(".safetensors") for name in result)
    ):
        raise Day20EvaluationError("LoRA adapter file manifest is incomplete")
    return result


def build_model_identity(
    *, base: Path, adapter: Path | None, model_class: str
) -> dict[str, Any]:
    base_files = file_manifest(base, kind="base")
    base_identity = {
        "path": str(base.resolve()),
        "model_class": model_class,
        "files": base_files,
        "snapshot_sha256": object_sha256(base_files),
    }
    adapter_identity = None
    if adapter is not None:
        adapter_files = file_manifest(adapter, kind="adapter", inference_only=True)
        checkpoint_files = file_manifest(adapter, kind="adapter")
        adapter_identity = {
            "path": str(adapter.resolve()),
            "format": "peft_lora",
            "files": adapter_files,
            "snapshot_sha256": object_sha256(adapter_files),
            "checkpoint_package_files": checkpoint_files,
            "checkpoint_package_snapshot_sha256": object_sha256(checkpoint_files),
        }
    combined = object_sha256(
        {
            "domain": "day20.qwen35_base_adapter_identity",
            "schema_version": 1,
            "base_snapshot_sha256": base_identity["snapshot_sha256"],
            "adapter_snapshot_sha256": (
                adapter_identity["snapshot_sha256"] if adapter_identity else None
            ),
        }
    )
    return {
        "base": base_identity,
        "adapter": adapter_identity,
        "combined_snapshot_sha256": combined,
        "load_and_generation_verified": True,
    }


def verify_adapter_config(adapter: Path) -> dict[str, Any]:
    config = day19_eval.load_json(adapter / "adapter_config.json")
    required = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
    }
    if any(config.get(key) != expected for key, expected in required.items()):
        raise Day20EvaluationError("LoRA adapter hyperparameters drifted")
    peft_type = str(config.get("peft_type", "")).upper()
    if peft_type and peft_type != "LORA":
        raise Day20EvaluationError("checkpoint is not a LoRA adapter")
    return config


def verify_training_summary(
    *,
    path: Path,
    candidate: str,
    adapter: Path,
    experiment_manifest: dict[str, Any],
) -> dict[str, Any]:
    summary = day19_eval.load_json(path)
    expected_self_hash = object_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    is_probe = candidate in PROBE_CANDIDATES
    expected_kind = "probe" if is_probe else "main"
    expected_label = "probe" if is_probe else candidate
    expected_tokens = 16_000 if is_probe else 256_000
    dataset_identity = experiment_manifest["datasets"][expected_kind]
    if (
        summary.get("schema_version") != 1
        or summary.get("domain") != "day20.qwen35_lora_training_summary"
        or summary.get("status") != "pass"
        or summary.get("summary_sha256") != expected_self_hash
        or summary.get("run_kind") != expected_kind
        or summary.get("cumulative_supervised_tokens") != expected_tokens
        or summary.get("dataset_file_sha256") != dataset_identity["file_sha256"]
    ):
        raise Day20EvaluationError("training summary identity drifted")
    learning_rate = summary.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or float(learning_rate) not in {1e-5, 3e-5, 1e-4}
        or (is_probe and float(learning_rate) != PROBE_CANDIDATES[candidate])
    ):
        raise Day20EvaluationError("training learning rate differs from candidate")
    checkpoint = (summary.get("checkpoints") or {}).get(expected_label)
    verified_checkpoint = verified_checkpoint_path(
        path,
        name=expected_label,
        run_root=Path(str(experiment_manifest.get("run_root", ""))),
    )
    checkpoint_tokens = (summary.get("checkpoint_tokens") or {}).get(expected_label)
    expected_checkpoint_tokens = (
        16_000 if is_probe else MAIN_CHECKPOINT_TOKENS[expected_label]
    )
    if (
        not isinstance(checkpoint, str)
        or Path(checkpoint).resolve() != adapter.resolve()
        or verified_checkpoint != adapter.resolve()
        or checkpoint_tokens != expected_checkpoint_tokens
    ):
        raise Day20EvaluationError("training checkpoint lineage drifted")
    training_config = summary.get("training_config")
    if not isinstance(training_config, dict):
        raise Day20EvaluationError("training config provenance is missing")
    config_path = Path(str(training_config.get("path", ""))).resolve()
    if file_sha256(config_path) != training_config.get("file_sha256"):
        raise Day20EvaluationError("training config file identity drifted")
    config = day19_eval.load_json(config_path)
    config_self_hash = object_sha256(
        {key: value for key, value in config.items() if key != "immutable_sha256"}
    )
    immutable_lora = experiment_manifest["contract"]["immutable"]["lora"]
    config_training = config.get("training")
    config_data = config.get("data")
    if (
        config.get("immutable_sha256") != config_self_hash
        or config.get("run_kind") != expected_kind
        or not isinstance(config_training, dict)
        or not isinstance(config_data, dict)
        or float(config_training.get("learning_rate", -1)) != float(learning_rate)
        or config_training.get("rank") != immutable_lora["rank"]
        or config_training.get("alpha") != immutable_lora["alpha"]
        or config_training.get("dropout") != immutable_lora["dropout"]
        or config_training.get("target_regex") != immutable_lora["target_regex"]
        or config_data.get("file_sha256") != dataset_identity["file_sha256"]
    ):
        raise Day20EvaluationError("resolved training config drifted")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "content_sha256": summary["summary_sha256"],
        "run_kind": expected_kind,
        "checkpoint_label": expected_label,
        "checkpoint_target_supervised_tokens": expected_checkpoint_tokens,
        "learning_rate": float(learning_rate),
        "training_config": training_config,
        "checkpoint_package": summary["checkpoint_packages"][expected_label],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Frozen Base model")
    parser.add_argument("--adapter", type=Path, help="Optional LoRA checkpoint")
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--scorers", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int)
    args = parser.parse_args()
    if not _CANDIDATE_RE.fullmatch(args.candidate):
        raise Day20EvaluationError("candidate has an unsafe or empty identifier")
    is_probe = args.sample_limit == 32
    allowed = (
        {"base-probe", *PROBE_CANDIDATES}
        if is_probe
        else {"base", *MAIN_CHECKPOINT_TOKENS}
    )
    if args.candidate not in allowed:
        raise Day20EvaluationError("candidate is incompatible with evaluation scope")
    is_base = args.candidate in {"base", "base-probe"}
    if is_base:
        if args.adapter is not None or args.training_summary is not None:
            raise Day20EvaluationError("Base candidates cannot load training artifacts")
        if args.checkpoint.resolve() != args.model.resolve():
            raise Day20EvaluationError("Base checkpoint must be the Base model")
    else:
        if args.adapter is None or args.training_summary is None:
            raise Day20EvaluationError(
                "LoRA candidates require adapter and training-summary"
            )
        if args.checkpoint.resolve() != args.adapter.resolve():
            raise Day20EvaluationError("LoRA checkpoint must equal adapter path")
    return args


def main() -> None:
    args = parse_args()
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    if args.output.exists() or predictions_path.exists():
        raise Day20EvaluationError("refusing to overwrite existing evaluation output")
    manifest = day19_eval.load_json(args.eval_manifest)
    expected_scorer_sha = manifest.get("header", {}).get("scorer_source_sha256")
    if (
        not isinstance(expected_scorer_sha, str)
        or file_sha256(args.scorers) != expected_scorer_sha
    ):
        raise Day20EvaluationError("frozen scorer source identity drifted")
    experiment = load_experiment_manifest(
        args.experiment_manifest,
        eval_manifest_path=args.eval_manifest,
        verify_diagnostic_file=True,
    )
    if experiment["contract"]["immutable"]["model"]["snapshot"] != str(
        args.model.resolve()
    ):
        raise Day20EvaluationError("evaluation Base differs from experiment contract")
    selected = selected_experiment_records(
        eval_manifest=manifest,
        experiment_manifest=experiment,
        sample_limit=args.sample_limit,
    )
    scorers = day19_eval.load_scorers(args.scorers)
    training_identity = None
    if args.adapter is not None:
        verify_adapter_config(args.adapter)
        training_identity = verify_training_summary(
            path=args.training_summary,
            candidate=args.candidate,
            adapter=args.adapter,
            experiment_manifest=experiment,
        )
    model_identity = build_model_identity(
        base=args.model,
        adapter=args.adapter,
        model_class="Qwen3_5ForConditionalGeneration",
    )
    prepared_base_identity = experiment["base_model_identity"]
    if any(
        (
            model_identity["base"].get("path")
            != prepared_base_identity.get("path"),
            model_identity["base"].get("files")
            != prepared_base_identity.get("files"),
            model_identity["base"].get("snapshot_sha256")
            != prepared_base_identity.get("snapshot_sha256"),
        )
    ):
        raise Day20EvaluationError("Base model differs from prepared snapshot identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_attempt = predictions_path.with_name(
        f".{predictions_path.name}.{os.getpid()}.attempt"
    )
    if predictions_attempt.exists():
        raise Day20EvaluationError(
            f"refusing to overwrite evaluation attempt: {predictions_attempt}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    if os.environ.get("USE_MCORE_GDN") not in (None, "0"):
        raise Day20EvaluationError("USE_MCORE_GDN conflicts with the HF backend contract")
    os.environ["USE_MCORE_GDN"] = "0"
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
        raise Day20EvaluationError(f"wrong Qwen3.5 loader: {model.__class__.__name__}")
    template = get_template(
        processor,
        max_length=4096,
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    template_meta = getattr(template, "template_meta", None)
    if getattr(template_meta, "template_type", None) != "qwen3_5":
        raise Day20EvaluationError("ms-swift did not construct the qwen3_5 template")
    if getattr(template_meta, "non_thinking_prefix", None) != NON_THINKING_PREFIX:
        raise Day20EvaluationError("Qwen3.5 non-thinking response prefix drifted")
    if getattr(template, "enable_thinking", None) is not False:
        raise Day20EvaluationError("Qwen3.5 template did not disable thinking")
    if getattr(template, "add_non_thinking_prefix", None) is not True:
        raise Day20EvaluationError("Qwen3.5 template disabled non-thinking prefix")
    try:
        ms_swift_version = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError as error:
        raise Day20EvaluationError("cannot identify installed ms-swift") from error

    engine_kwargs: dict[str, Any] = {
        "template": template,
        "max_batch_size": 1,
    }
    if args.adapter is not None:
        engine_kwargs["adapters"] = [str(args.adapter.resolve())]
    engine = TransformersEngine(model, **engine_kwargs)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for ordinal, record in enumerate(selected, 1):
            maximum = int(record["generation_max_new_tokens"])
            response = engine.infer(
                [InferRequest(messages=record["messages"])],
                RequestConfig(
                    max_tokens=maximum,
                    temperature=0,
                    num_beams=1,
                    repetition_penalty=1.0,
                    seed=20260809,
                    return_details=True,
                ),
                use_tqdm=False,
            )[0]
            choice = response.choices[0]
            raw_output = choice.message.content or ""
            prompt_ids = getattr(response, "prompt_token_ids", None)
            generated_ids = getattr(choice, "token_ids", None)
            if not isinstance(prompt_ids, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in prompt_ids
            ):
                raise Day20EvaluationError(
                    f"missing prompt token evidence: {record['sample_id']}"
                )
            if not isinstance(generated_ids, list) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in generated_ids
            ):
                raise Day20EvaluationError(
                    f"missing generated token evidence: {record['sample_id']}"
                )
            generated_text = template.decode_generate_ids(
                generated_ids, first_token=False
            )
            if not isinstance(generated_text, str):
                raise Day20EvaluationError(
                    f"generated-only decode is not text: {record['sample_id']}"
                )
            boundary = adapt_qwen35_text(
                raw_output, generated_only_text=generated_text
            )
            score = scorers.score_prediction(
                record["slice"], raw_output, record["reference"]
            )
            completion_tokens = int(response.usage.completion_tokens)
            prompt_tokens = int(response.usage.prompt_tokens)
            if completion_tokens != len(generated_ids) or prompt_tokens != len(prompt_ids):
                raise Day20EvaluationError(
                    f"token accounting mismatch: {record['sample_id']}"
                )
            anomalies: list[str] = []
            if not raw_output.strip():
                anomalies.append("empty_output")
            if choice.finish_reason == "length" or completion_tokens >= maximum:
                anomalies.append("generation_ceiling")
            repeated_ratio = day19_eval.repeated_ngram_ratio(raw_output)
            if repeated_ratio >= 0.5:
                anomalies.append("high_4gram_repetition")
            syntax = None
            if record["slice"] == "code":
                syntax = day19_eval.code_syntax_status(
                    record.get("raw_prompt", ""), score["parsed_answer"]
                )
                if not syntax["valid"]:
                    anomalies.append("code_syntax_invalid")
            format_compliant = (
                bool(raw_output.strip())
                and score.get("parse_status") == "ok"
                and (syntax is None or syntax["valid"])
            )
            row = {
                "schema_version": 1,
                "domain": "day20.qwen35_lora_eval_prediction",
                "ordinal": ordinal,
                "candidate": args.candidate,
                "recipe": args.candidate,
                "sample_id": record["sample_id"],
                "slice": record["slice"],
                "raw_output": raw_output,
                "raw_output_sha256": text_sha256(raw_output),
                "response_capture": {
                    "method": "ms_swift_return_details",
                    "message_content_semantics": (
                        "ms-swift template decode may reconstruct the configured "
                        "response prefix ahead of generated token text"
                    ),
                    "response_adapter_verification": {
                        "adapter_version": boundary["adapter_version"],
                        "prefix_status": boundary["prefix_status"],
                        "final_text_sha256": boundary["final_text_sha256"],
                    },
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
                    "generated_token_ids": generated_ids,
                    "generated_token_ids_sha256": token_ids_sha256(generated_ids),
                    "generated_only_text": generated_text,
                    "generated_only_text_sha256": text_sha256(generated_text),
                },
                "reference_hash": record["reference_hash"],
                "prompt_messages_hash": record["messages_hash"],
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": maximum,
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "scorer_result": score,
                "format_compliant": format_compliant,
                "code_syntax": syntax,
                "anomalies": anomalies,
                "repeated_4gram_ratio": repeated_ratio,
            }
            rows.append(row)
            day19_eval.append_jsonl(predictions_attempt, row)
            print(
                json.dumps(
                    {
                        "ordinal": ordinal,
                        "sample_id": record["sample_id"],
                        "slice": record["slice"],
                        "score": score.get("score"),
                        "format_compliant": format_compliant,
                        "anomalies": anomalies,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException:
        # Keep the PID-scoped attempt for diagnosis; it is never a published artifact.
        raise

    by_slice: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["slice"] == skill]
        scored = [
            row
            for row in skill_rows
            if isinstance(row["scorer_result"].get("score"), (int, float))
            and not isinstance(row["scorer_result"].get("score"), bool)
        ]
        correct = sum(float(row["scorer_result"]["score"]) for row in scored)
        entry: dict[str, Any] = {
            "records": len(skill_rows),
            "scored_records": len(scored),
            "correct": correct if scored else None,
            "accuracy": correct / len(scored) if scored else None,
            "format_compliant": sum(bool(row["format_compliant"]) for row in skill_rows),
            "anomalous_records": sum(bool(row["anomalies"]) for row in skill_rows),
            "generation_ceiling_records": sum(
                "generation_ceiling" in row["anomalies"] for row in skill_rows
            ),
            "output_tokens": sum(
                row["generation"]["completion_tokens"] for row in skill_rows
            ),
        }
        if skill == "code":
            entry["syntax_valid"] = sum(
                bool(row["code_syntax"]["valid"]) for row in skill_rows
            )
            entry["executable_score_status"] = "sandbox_required"
        by_slice[skill] = entry
    non_code = [row for row in rows if row["slice"] != "code"]
    post_run_model_identity = build_model_identity(
        base=args.model,
        adapter=args.adapter,
        model_class=model.__class__.__name__,
    )
    if post_run_model_identity != model_identity:
        raise Day20EvaluationError("Base or adapter files changed during evaluation")
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": 1,
        "domain": DOMAIN,
        "status": "complete_with_code_sandbox_required",
        "created_at_utc": utc_now(),
        "candidate": args.candidate,
        "recipe": args.candidate,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_identity": model_identity,
        "model_identity_post_run_unchanged": True,
        "training_identity": training_identity,
        "protocol": {
            "eval_manifest": str(args.eval_manifest.resolve()),
            "eval_manifest_file_sha256": file_sha256(args.eval_manifest),
            "experiment_manifest": str(args.experiment_manifest.resolve()),
            "experiment_manifest_file_sha256": file_sha256(
                args.experiment_manifest
            ),
            "experiment_manifest_content_sha256": experiment["manifest_sha256"],
            "diagnostic_selection_sha256": (
                experiment["datasets"]["diagnostic"]["selection_sha256"]
                if args.sample_limit == 32
                else None
            ),
            "scorer_source": str(args.scorers.resolve()),
            "scorer_source_sha256": file_sha256(args.scorers),
            "split": "dev",
            "frozen_test_consumed": False,
            "records": len(rows),
            "sample_limit": args.sample_limit,
            "template": "qwen3_5",
            "backend": "hf_transformers",
            "use_mcore_gdn": False,
            "enable_thinking": False,
            "add_non_thinking_prefix": True,
            "non_thinking_prefix_sha256": text_sha256(NON_THINKING_PREFIX),
            "ms_swift_version": ms_swift_version,
            "greedy": True,
            "seed": 20260809,
            "response_capture": "ms-swift RequestConfig(return_details=True)",
            "response_boundary_adapter": ADAPTER_VERSION,
            "generated_token_ids_retained": True,
            "adapter_loaded_unmerged": args.adapter is not None,
            "code_execution": "not executed on host; sandbox score required",
        },
        "metrics": {
            "by_slice": by_slice,
            "non_code_correct": sum(
                float(row["scorer_result"]["score"]) for row in non_code
            ),
            "non_code_total": len(non_code),
            "format_compliant": sum(bool(row["format_compliant"]) for row in rows),
            "anomalous_records": sum(bool(row["anomalies"]) for row in rows),
            "output_tokens": sum(
                row["generation"]["completion_tokens"] for row in rows
            ),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "predictions": {
            "path": str(predictions_path.resolve()),
            "records": len(rows),
            "file_sha256": file_sha256(predictions_attempt),
        },
    }
    if len(rows) == 112 and Counter(row["slice"] for row in rows) != Counter(
        {skill: 28 for skill in SKILLS}
    ):
        raise Day20EvaluationError("published full evaluation slice distribution drifted")
    if not math.isfinite(float(summary["metrics"]["peak_cuda_memory_gib"])):
        raise Day20EvaluationError("non-finite peak CUDA memory")
    summary["summary_sha256"] = object_sha256(summary)
    publish_evaluation_pair(
        predictions_attempt=predictions_attempt,
        predictions_path=predictions_path,
        summary_path=args.output,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
