#!/usr/bin/env python3
"""Run the frozen Day 10 Base-model evaluation on the development split.

The runner consumes the already-rendered token IDs from the frozen manifest.
It never evaluates ``frozen_test`` and never executes HumanEval completions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rfc8785


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-frozen-eval-manifest.json"
)
SCORER_PATH = HERE / "day10_scorers.py"


class Day10RunnerError(ValueError):
    """The frozen run contract or a local runtime invariant failed."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Day10RunnerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError) as error:
        raise Day10RunnerError(f"RFC 8785 canonicalization failed: {error}") from error


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day10RunnerError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_text_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Day10RunnerError(f"non-finite JSON value: {value}")
            ),
        )
    except FileNotFoundError as error:
        raise Day10RunnerError(f"manifest is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise Day10RunnerError(f"invalid JSON in {path}: {error}") from error


def _load_scorers(expected_source_sha256: str) -> Any:
    actual = file_sha256(SCORER_PATH)
    if actual != expected_source_sha256:
        raise Day10RunnerError(
            "scorer source hash differs from the frozen manifest: "
            f"expected {expected_source_sha256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("day10_baseline_scorers", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise Day10RunnerError(f"cannot load scorer module: {SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day10RunnerError(f"scorer module import failed: {error}") from error
    if not callable(getattr(module, "score_prediction", None)):
        raise Day10RunnerError("frozen scorer has no score_prediction function")
    return module


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise Day10RunnerError(f"{context}.{key} must be a non-empty string")
    return value


def _verify_record(record: dict[str, Any], scorer_registry: dict[str, Any]) -> None:
    sample_id = _require_string(record, "sample_id", "record")
    evaluation_split = record.get("evaluation_split")
    if evaluation_split not in {"dev", "frozen_test"}:
        raise Day10RunnerError(f"invalid evaluation split for {sample_id}")
    slice_name = record.get("slice")
    registry_slices = scorer_registry.get("slices", {})
    if slice_name not in registry_slices:
        raise Day10RunnerError(f"unknown scorer slice for {sample_id}: {slice_name!r}")

    rendered_prompt = record.get("rendered_prompt")
    if not isinstance(rendered_prompt, str):
        raise Day10RunnerError(f"rendered prompt is not text: {sample_id}")
    if exact_text_hash(rendered_prompt) != record.get("rendered_prompt_hash"):
        raise Day10RunnerError(f"rendered prompt hash mismatch: {sample_id}")

    input_ids = record.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in input_ids
    ):
        raise Day10RunnerError(f"invalid input IDs: {sample_id}")
    expected_input_hash = semantic_hash(
        {
            "domain": "day10.input_ids",
            "schema_version": 1,
            "input_ids": input_ids,
        }
    )
    if expected_input_hash != record.get("input_ids_hash"):
        raise Day10RunnerError(f"input ID hash mismatch: {sample_id}")
    if record.get("input_token_count") != len(input_ids):
        raise Day10RunnerError(f"input token count mismatch: {sample_id}")
    if not isinstance(record.get("reference"), str):
        raise Day10RunnerError(f"reference is not text: {sample_id}")
    if record.get("reference_hash") != exact_text_hash(record["reference"]):
        raise Day10RunnerError(f"reference hash mismatch: {sample_id}")
    if not isinstance(record.get("generation_max_new_tokens"), int) or record[
        "generation_max_new_tokens"
    ] <= 0:
        raise Day10RunnerError(f"invalid generation limit: {sample_id}")

    scorer_spec = registry_slices[slice_name]
    if record.get("extractor_version") != scorer_spec.get("extractor_version"):
        raise Day10RunnerError(f"extractor version mismatch: {sample_id}")
    if record.get("scorer_version") != scorer_spec.get("scorer_version"):
        raise Day10RunnerError(f"scorer version mismatch: {sample_id}")
    if slice_name == "code" and scorer_spec.get("execution_policy") != "sandbox_required":
        raise Day10RunnerError("HumanEval scorer is not frozen to sandbox_required")


def verify_manifest(manifest: Any) -> Any:
    if not isinstance(manifest, dict):
        raise Day10RunnerError("manifest root must be an object")
    header = manifest.get("header")
    records = manifest.get("records")
    if not isinstance(header, dict) or not isinstance(records, list):
        raise Day10RunnerError("manifest must contain an object header and records array")
    if header.get("domain") != "day10.frozen_eval_manifest":
        raise Day10RunnerError("unexpected manifest domain")
    if header.get("schema_version") != 1:
        raise Day10RunnerError("unsupported manifest schema version")
    if header.get("status") != "frozen_manifest_pre_baseline":
        raise Day10RunnerError("manifest is not frozen for the Base baseline")

    expected_hash = _require_string(header, "manifest_hash", "header")
    candidate = copy.deepcopy(manifest)
    candidate["header"].pop("manifest_hash", None)
    actual_hash = semantic_hash(candidate)
    if actual_hash != expected_hash:
        raise Day10RunnerError(
            f"manifest semantic hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    for field in (
        "config_file_sha256",
        "dataset_context_hash",
        "eval_suite_hash",
        "protocol_hash",
        "scorer_source_sha256",
        "scorer_registry_hash",
        "environment_contract_sha256",
        "environment_snapshot_sha256",
    ):
        _require_string(header, field, "header")
    generation = header.get("generation")
    rendering = header.get("model_and_rendering")
    if not isinstance(generation, dict) or not isinstance(rendering, dict):
        raise Day10RunnerError("manifest has no frozen generation/rendering config")

    scorers = _load_scorers(header["scorer_source_sha256"])
    registry = getattr(scorers, "SCORER_REGISTRY", None)
    if not isinstance(registry, dict):
        raise Day10RunnerError("frozen scorer registry is missing")

    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise Day10RunnerError("manifest record is not an object")
        _verify_record(record, registry)
        sample_id = record["sample_id"]
        if sample_id in by_id:
            raise Day10RunnerError(f"duplicate sample ID: {sample_id}")
        by_id[sample_id] = record

    orders = header.get("evaluation_order")
    if not isinstance(orders, dict):
        raise Day10RunnerError("manifest evaluation order is missing")
    for split in ("dev", "frozen_test"):
        ordered_ids = orders.get(split)
        if not isinstance(ordered_ids, list) or not all(
            isinstance(sample_id, str) for sample_id in ordered_ids
        ):
            raise Day10RunnerError(f"invalid evaluation order for {split}")
        if len(ordered_ids) != len(set(ordered_ids)):
            raise Day10RunnerError(f"duplicate IDs in {split} evaluation order")
        expected_ids = {
            sample_id
            for sample_id, record in by_id.items()
            if record["evaluation_split"] == split
        }
        if set(ordered_ids) != expected_ids:
            raise Day10RunnerError(f"{split} evaluation order does not cover its records")
    return scorers


def load_manifest(path: Path) -> tuple[dict[str, Any], Any]:
    manifest = load_json(path.resolve())
    scorers = verify_manifest(manifest)
    return manifest, scorers


def select_records(
    manifest: dict[str, Any], *, split: str = "dev", sample_limit: int | None = None
) -> list[dict[str, Any]]:
    if split != "dev":
        raise Day10RunnerError(
            "frozen_test access is prohibited; the Day 10 runner only permits dev"
        )
    if sample_limit is not None and (
        isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit <= 0
    ):
        raise Day10RunnerError("sample_limit must be a positive integer")

    records_by_id = {record["sample_id"]: record for record in manifest["records"]}
    ordered_ids = manifest["header"]["evaluation_order"]["dev"]
    if sample_limit is not None and sample_limit > len(ordered_ids):
        raise Day10RunnerError(
            f"sample_limit {sample_limit} exceeds the {len(ordered_ids)} dev records"
        )
    if sample_limit is None:
        selected_ids = ordered_ids
    else:
        canonical_slice_order = ("general", "math", "code", "finance")
        buckets = {
            slice_name: [
                sample_id
                for sample_id in ordered_ids
                if records_by_id[sample_id]["slice"] == slice_name
            ]
            for slice_name in canonical_slice_order
        }
        slice_order = [slice_name for slice_name in canonical_slice_order if buckets[slice_name]]
        selected_ids = []
        offsets = {slice_name: 0 for slice_name in slice_order}
        while len(selected_ids) < sample_limit:
            progressed = False
            for slice_name in slice_order:
                offset = offsets[slice_name]
                if offset >= len(buckets[slice_name]):
                    continue
                selected_ids.append(buckets[slice_name][offset])
                offsets[slice_name] += 1
                progressed = True
                if len(selected_ids) == sample_limit:
                    break
            if not progressed:
                raise Day10RunnerError("cannot satisfy the requested stratified sample limit")
    selected = [records_by_id[sample_id] for sample_id in selected_ids]
    if any(record["evaluation_split"] != "dev" for record in selected):
        raise Day10RunnerError("non-dev record reached the generation selection")
    return selected


def selection_policy(
    selected: list[dict[str, Any]], sample_limit: int | None
) -> dict[str, Any]:
    if sample_limit is None:
        return {
            "version": "day10_frozen_dev_evaluation_order_v1",
            "evaluation_split": "dev",
            "sample_limit": None,
        }
    slice_order = [
        slice_name
        for slice_name in ("general", "math", "code", "finance")
        if any(record["slice"] == slice_name for record in selected)
    ]
    return {
        "version": "day10_dev_stratified_round_robin_v1",
        "evaluation_split": "dev",
        "sample_limit": sample_limit,
        "slice_order": slice_order,
        "within_slice_order": "frozen_dev_evaluation_order",
    }


def _resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_model_path(
    manifest: dict[str, Any], cli_model_path: Path | None
) -> tuple[Path, str]:
    rendering = manifest["header"]["model_and_rendering"]
    configured = rendering.get("model_snapshot_path")
    if cli_model_path is not None:
        path = cli_model_path.expanduser().resolve()
        source = "cli_override" if configured is not None else "cli_required"
    elif configured is None:
        raise Day10RunnerError(
            "manifest has no model_snapshot_path; pass an explicit --model-path"
        )
    else:
        if not isinstance(configured, str) or not configured:
            raise Day10RunnerError("model_snapshot_path must be a non-empty string")
        path = _resolve_manifest_path(configured)
        source = "manifest"
    if not path.is_dir():
        raise Day10RunnerError(f"model snapshot directory is missing: {path}")
    return path, source


def model_snapshot_identity(
    manifest: dict[str, Any],
    model_path: Path,
    snapshot_source: str,
    *,
    cli_model_id: str | None = None,
    cli_model_revision: str | None = None,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(model_path).as_posix()
        files[relative_path] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    if "config.json" not in files:
        raise Day10RunnerError("model snapshot has no config.json")
    if not any(
        name.endswith(".safetensors") or name.startswith("pytorch_model") and name.endswith(".bin")
        for name in files
    ):
        raise Day10RunnerError("model snapshot has no local model weights")

    rendering = manifest["header"]["model_and_rendering"]
    expected_files = rendering.get("model_files", rendering.get("model_snapshot_files"))
    if expected_files is not None and snapshot_source == "manifest":
        if not isinstance(expected_files, dict) or not expected_files:
            raise Day10RunnerError("manifest model_files must be a non-empty object")
        for name, expected_hash in expected_files.items():
            actual_spec = files.get(name)
            if not isinstance(expected_hash, str) or actual_spec is None:
                raise Day10RunnerError(f"required model snapshot file is missing: {name}")
            if actual_spec["sha256"] != expected_hash:
                raise Day10RunnerError(f"model snapshot file hash mismatch: {name}")

    snapshot_hash = semantic_hash(
        {
            "domain": "day10.model_snapshot",
            "schema_version": 1,
            "files": files,
        }
    )
    expected_snapshot_hash = rendering.get("model_snapshot_hash")
    if (
        expected_snapshot_hash is not None
        and snapshot_source == "manifest"
        and snapshot_hash != expected_snapshot_hash
    ):
        raise Day10RunnerError("model snapshot hash differs from the manifest")

    if snapshot_source.startswith("cli_"):
        if not cli_model_id or not cli_model_revision:
            raise Day10RunnerError(
                "--model-path requires explicit --model-id and --model-revision so "
                "the override is not mislabeled as the manifest Base checkpoint"
            )
        model_id = cli_model_id
        model_revision = cli_model_revision
    else:
        if cli_model_id is not None or cli_model_revision is not None:
            raise Day10RunnerError(
                "--model-id/--model-revision are only valid with --model-path"
            )
        model_id = _require_string(rendering, "model_id", "model_and_rendering")
        model_revision = _require_string(
            rendering, "model_revision", "model_and_rendering"
        )
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_snapshot_path": str(model_path),
        "model_snapshot_source": snapshot_source,
        "model_snapshot_hash": snapshot_hash,
        "model_files": files,
    }


def choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise Day10RunnerError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise Day10RunnerError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise Day10RunnerError(f"unsupported device: {requested}")
    return requested


def load_runtime(
    manifest: dict[str, Any], model_path: Path, device: str
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rendering = manifest["header"]["model_and_rendering"]
    tokenizer_path_value = _require_string(
        rendering, "tokenizer_snapshot_path", "model_and_rendering"
    )
    tokenizer_path = _resolve_manifest_path(tokenizer_path_value)
    if not tokenizer_path.is_dir():
        raise Day10RunnerError(f"tokenizer snapshot directory is missing: {tokenizer_path}")
    tokenizer_files = rendering.get("tokenizer_files")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise Day10RunnerError("frozen tokenizer file hashes are missing")
    for filename, expected_hash in tokenizer_files.items():
        if file_sha256(tokenizer_path / filename) != expected_hash:
            raise Day10RunnerError(f"tokenizer file hash mismatch: {filename}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype="auto",
        )
    except Exception as error:
        raise Day10RunnerError(f"local model/tokenizer loading failed: {error}") from error

    if type(tokenizer).__name__ != rendering.get("tokenizer_class"):
        raise Day10RunnerError("loaded tokenizer class differs from the manifest")
    template = tokenizer.chat_template or ""
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != rendering.get(
        "chat_template_sha256"
    ):
        raise Day10RunnerError("loaded tokenizer chat template differs from the manifest")
    generation = manifest["header"]["generation"]
    if tokenizer.eos_token_id != generation.get("eos_token_id"):
        raise Day10RunnerError("loaded tokenizer EOS ID differs from the manifest")
    if tokenizer.pad_token_id != generation.get("pad_token_id"):
        raise Day10RunnerError("loaded tokenizer padding ID differs from the manifest")

    model.to(device)
    model.eval()
    return model, tokenizer


def frozen_generation_kwargs(
    generation: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    if generation.get("do_sample") is not False or generation.get("num_beams") != 1:
        raise Day10RunnerError("Day 10 generation is not frozen to greedy decoding")
    for disabled in ("temperature", "top_p", "top_k"):
        if generation.get(disabled) is not None:
            raise Day10RunnerError(f"greedy generation requires {disabled}=null")
    eos_token_id = generation.get("eos_token_id")
    pad_token_id = generation.get("pad_token_id")
    stop_token_ids = generation.get("stop_token_ids")
    if not isinstance(eos_token_id, int) or not isinstance(pad_token_id, int):
        raise Day10RunnerError("generation EOS/padding IDs are invalid")
    if stop_token_ids != [eos_token_id]:
        raise Day10RunnerError("only the frozen EOS stop token is supported")
    repetition_penalty = generation.get("repetition_penalty")
    if not isinstance(repetition_penalty, (int, float)):
        raise Day10RunnerError("repetition_penalty is invalid")
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "repetition_penalty": repetition_penalty,
    }


def _synchronize_device(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def generate_prediction(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    generation_kwargs: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    import torch

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    _synchronize_device(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    _synchronize_device(device)
    latency_seconds = time.perf_counter() - started

    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        raise Day10RunnerError("model.generate did not return a rank-2 tensor")
    if generated.shape[0] != 1 or generated.shape[1] < len(input_ids):
        raise Day10RunnerError("model.generate returned an invalid sequence shape")
    if generated[0, : len(input_ids)].detach().cpu().tolist() != input_ids:
        raise Day10RunnerError("model.generate did not preserve the frozen prompt IDs")
    output_token_ids = generated[0, len(input_ids) :].detach().cpu().tolist()
    if not all(isinstance(token_id, int) for token_id in output_token_ids):
        raise Day10RunnerError("generated token IDs are invalid")
    raw_output = tokenizer.decode(
        output_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(raw_output, str):
        raise Day10RunnerError("tokenizer.decode did not return text")
    return {
        "raw_output": raw_output,
        "raw_output_hash": exact_text_hash(raw_output),
        "output_token_ids": output_token_ids,
        "output_token_ids_hash": semantic_hash(
            {
                "domain": "day10.output_token_ids",
                "schema_version": 1,
                "output_token_ids": output_token_ids,
            }
        ),
        "output_token_count": len(output_token_ids),
        "latency_seconds": latency_seconds,
    }


def _model_dtype(model: Any) -> str:
    dtype = getattr(model, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def build_execution_protocol(
    manifest: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    header = manifest["header"]
    return {
        "domain": "day10.execution_protocol",
        "schema_version": 1,
        "runner_source_sha256": file_sha256(Path(__file__).resolve()),
        "environment_contract_sha256": header["environment_contract_sha256"],
        "environment_snapshot_sha256": header["environment_snapshot_sha256"],
        "local_files_only": True,
        "trust_remote_code": False,
        "device": runtime["device"],
        "model_dtype": runtime["model_dtype"],
        "torch_version": runtime["torch_version"],
        "transformers_version": runtime["transformers_version"],
        "python_version": runtime["python_version"],
        "python_implementation": runtime["python_implementation"],
        "platform": runtime["platform"],
        "torch_num_threads": runtime["torch_num_threads"],
        "torch_num_interop_threads": runtime["torch_num_interop_threads"],
    }


def build_comparison_key(
    manifest: dict[str, Any], execution_protocol_hash: str
) -> str:
    header = manifest["header"]
    return semantic_hash(
        {
            "domain": "day10.checkpoint_comparison",
            "schema_version": 1,
            "dataset_context_hash": header["dataset_context_hash"],
            "eval_suite_hash": header["eval_suite_hash"],
            "protocol_hash": header["protocol_hash"],
            "scorer_registry_hash": header["scorer_registry_hash"],
            "execution_protocol_hash": execution_protocol_hash,
        }
    )


def build_run_contract(
    manifest: dict[str, Any],
    model_identity: dict[str, Any],
    selected: list[dict[str, Any]],
    frozen_selection_policy: dict[str, Any],
    runtime: dict[str, Any],
    execution_protocol_hash: str,
    comparison_key: str,
) -> dict[str, Any]:
    header = manifest["header"]
    return {
        "domain": "day10.base_inference_run",
        "schema_version": 1,
        "manifest_hash": header["manifest_hash"],
        "config_file_sha256": header["config_file_sha256"],
        "protocol_hash": header["protocol_hash"],
        "scorer_registry_hash": header["scorer_registry_hash"],
        "execution_protocol_hash": execution_protocol_hash,
        "comparison_key": comparison_key,
        "model_id": model_identity["model_id"],
        "model_revision": model_identity["model_revision"],
        "model_snapshot_source": model_identity["model_snapshot_source"],
        "model_snapshot_hash": model_identity["model_snapshot_hash"],
        "evaluation_split": "dev",
        "selection_policy": frozen_selection_policy,
        "sample_ids": [record["sample_id"] for record in selected],
        "generation": header["generation"],
        "runtime": runtime,
    }


def write_jsonl_atomic(
    records: list[dict[str, Any]], output_path: Path, *, overwrite: bool
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise Day10RunnerError(
            f"output already exists (pass --overwrite to replace it): {output_path}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise Day10RunnerError(
                    f"output appeared during the run and was not replaced: {output_path}"
                ) from error
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_baseline(
    manifest_path: Path,
    output_path: Path,
    *,
    model_path: Path | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    split: str = "dev",
    sample_limit: int | None = None,
    overwrite: bool = False,
    requested_device: str = "auto",
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise Day10RunnerError(
            f"output already exists (pass --overwrite to replace it): {output_path}"
        )
    manifest_path = manifest_path.resolve()
    manifest_file_sha256 = file_sha256(manifest_path)
    manifest, scorers = load_manifest(manifest_path)
    if file_sha256(manifest_path) != manifest_file_sha256:
        raise Day10RunnerError("manifest changed while it was being loaded")
    selected = select_records(manifest, split=split, sample_limit=sample_limit)
    resolved_model_path, snapshot_source = resolve_model_path(manifest, model_path)
    model_identity = model_snapshot_identity(
        manifest,
        resolved_model_path,
        snapshot_source,
        cli_model_id=model_id,
        cli_model_revision=model_revision,
    )
    device = choose_device(requested_device)

    if (model is None) != (tokenizer is None):
        raise Day10RunnerError("model and tokenizer must be injected together")
    if model is None:
        model, tokenizer = load_runtime(manifest, resolved_model_path, device)
    else:
        model.eval()

    import torch
    import transformers

    generation = manifest["header"]["generation"]
    seed = generation.get("seed")
    if not isinstance(seed, int):
        raise Day10RunnerError("generation seed is invalid")
    transformers.set_seed(seed)
    run_started_at = datetime.now(timezone.utc).isoformat()
    runtime = {
        "device": device,
        "model_dtype": _model_dtype(model),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    frozen_selection_policy = selection_policy(selected, sample_limit)
    execution_protocol = build_execution_protocol(manifest, runtime)
    execution_protocol_hash = semantic_hash(execution_protocol)
    comparison_key = build_comparison_key(manifest, execution_protocol_hash)
    run_contract = build_run_contract(
        manifest,
        model_identity,
        selected,
        frozen_selection_policy,
        runtime,
        execution_protocol_hash,
        comparison_key,
    )
    run_hash = semantic_hash(run_contract)

    results: list[dict[str, Any]] = []
    for ordinal, record in enumerate(selected):
        if record["evaluation_split"] != "dev":
            raise Day10RunnerError("frozen_test record reached the generation loop")
        actual_generation = frozen_generation_kwargs(
            generation, record["generation_max_new_tokens"]
        )
        prediction = generate_prediction(
            model,
            tokenizer,
            record["input_ids"],
            actual_generation,
            device=device,
        )
        try:
            scorer_result = scorers.score_prediction(
                record["slice"], prediction["raw_output"], record["reference"]
            )
        except Exception as error:
            raise Day10RunnerError(
                f"scoring failed for {record['sample_id']}: {error}"
            ) from error
        if record["slice"] == "code" and not (
            scorer_result.get("score_status") == "sandbox_required"
            and scorer_result.get("score") is None
            and scorer_result.get("error_type") == "sandbox_required"
        ):
            raise Day10RunnerError("HumanEval result left the sandbox_required state")

        results.append(
            {
                "schema_version": 1,
                "run_hash": run_hash,
                "run_started_at_utc": run_started_at,
                "run_ordinal": ordinal,
                "manifest_hash": manifest["header"]["manifest_hash"],
                "manifest_file_sha256": manifest_file_sha256,
                "config_file_sha256": manifest["header"]["config_file_sha256"],
                "protocol_hash": manifest["header"]["protocol_hash"],
                "scorer_registry_hash": manifest["header"]["scorer_registry_hash"],
                "execution_protocol": execution_protocol,
                "execution_protocol_hash": execution_protocol_hash,
                "comparison_key": comparison_key,
                "model_id": model_identity["model_id"],
                "model_revision": model_identity["model_revision"],
                "model_snapshot_path": model_identity["model_snapshot_path"],
                "model_snapshot_source": model_identity["model_snapshot_source"],
                "model_snapshot_hash": model_identity["model_snapshot_hash"],
                "model_files": model_identity["model_files"],
                "runtime": runtime,
                "sample_id": record["sample_id"],
                "evaluation_split": "dev",
                "selection_policy": frozen_selection_policy,
                "slice": record["slice"],
                "rendered_prompt_hash": record["rendered_prompt_hash"],
                "input_ids_hash": record["input_ids_hash"],
                "reference_hash": record["reference_hash"],
                "input_token_count": record["input_token_count"],
                "generation": actual_generation,
                **prediction,
                "total_token_count": record["input_token_count"]
                + prediction["output_token_count"],
                "extractor_version": record["extractor_version"],
                "scorer_version": record["scorer_version"],
                "scorer_result": scorer_result,
            }
        )

    write_jsonl_atomic(results, output_path, overwrite=overwrite)
    return {
        "run_hash": run_hash,
        "output_path": str(output_path),
        "record_count": len(results),
        "model_snapshot_hash": model_identity["model_snapshot_hash"],
        "execution_protocol_hash": execution_protocol_hash,
        "comparison_key": comparison_key,
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = run_baseline(
            args.manifest,
            args.output,
            model_path=args.model_path,
            model_id=args.model_id,
            model_revision=args.model_revision,
            split=args.split,
            sample_limit=args.sample_limit,
            overwrite=args.overwrite,
            requested_device=args.device,
        )
    except Day10RunnerError as error:
        parser.error(str(error))
    print(
        f"baseline=written path={result['output_path']} "
        f"records={result['record_count']} run_hash={result['run_hash']} "
        f"model_hash={result['model_snapshot_hash']}"
    )


if __name__ == "__main__":
    main()
