#!/usr/bin/env python3
"""Run or fully revalidate one fail-closed Day 20 v2 raw evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import day20_candidate_factory_v2 as candidate_factory
from day20_contract_v2 import (
    Day20V2ContractError,
    V2_CONTRACT_VERSION,
    validate_raw_code_continuation,
)
from day20_train_plugin_v2 import (
    EXPECTED_RUNTIME_VERSIONS as TRAINING_RUNTIME_VERSIONS,
    checkpoint_path,
    runtime_identity as training_runtime_identity,
    verify_training_summary,
)
from qwen35_response_adapter_v3 import (
    ADAPTER_VERSION,
    NON_THINKING_PREFIX,
    Qwen35ResponseAdapterV3Error,
    adapt_generated_token_decode,
    text_sha256,
    token_ids_sha256,
    validate_response_capture,
)


EVALUATOR_VERSION = "day20-v2-raw-evaluator-v2"
PREDICTION_DOMAIN = "day20.v2.raw_evaluation_prediction"
SUMMARY_DOMAIN = "day20.v2.raw_evaluation_summary"
EXPERIMENT_MANIFEST_DOMAIN = "day20.qwen35_balanced_lora.experiment_manifest.v2"
SOURCE_EXPANSION_DOMAIN = "day20.qwen35_source_expansion.v2"
SCHEMA_VERSION = 2
SKILLS = ("general", "math", "code", "finance")
EXPECTED_RECORDS = {"probe32": 32, "full112": 112}
TARGET_FORMAT_BY_SKILL = {
    "general": "general_mcq",
    "math": "math_reasoning",
    "code": "code_continuation",
    "finance": "finance_value_scale",
}
EXPECTED_RUNTIME_VERSIONS = dict(TRAINING_RUNTIME_VERSIONS)
EXPECTED_MS_SWIFT_COMMIT = "565a1ad586a21d24b23931c52d2c62b49c39bee8"
EXPECTED_MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
GENERATION_SEED = candidate_factory.PRIMARY_SEED


class Day20EvaluationV2Error(ValueError):
    """A v2 input, runtime, prediction, or provenance invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20EvaluationV2Error(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20EvaluationV2Error(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day20EvaluationV2Error(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise Day20EvaluationV2Error(f"required JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Day20EvaluationV2Error(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Day20EvaluationV2Error(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Day20EvaluationV2Error(f"cannot load JSONL: {path}") from error
    return rows


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    if not isinstance(expected, str) or expected != actual:
        raise Day20EvaluationV2Error(f"{field} mismatch")
    return expected


def _sha256_prefixed(value: Any) -> str:
    return f"sha256:{object_sha256(value)}"


def _text_sha256_prefixed(value: str) -> str:
    return f"sha256:{text_sha256(value)}"


def _require_within(path: Path, parent: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(parent.expanduser().resolve())
    except ValueError as error:
        raise Day20EvaluationV2Error(f"{label} escapes its run directory") from error
    return resolved


def _require_regular_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise Day20EvaluationV2Error(f"{label} directory is missing or symbolic")
    return resolved


def _verify_eval_record(record: Mapping[str, Any]) -> None:
    skill = record.get("slice")
    if skill not in SKILLS:
        raise Day20EvaluationV2Error("frozen eval record has an unknown skill")
    for field, hash_field in (
        ("raw_prompt", "raw_prompt_hash"),
        ("reference", "reference_hash"),
        ("adapted_prompt", "adapted_prompt_hash"),
        ("rendered_prompt", "rendered_prompt_hash"),
    ):
        value = record.get(field)
        if not isinstance(value, str) or record.get(hash_field) != _text_sha256_prefixed(
            value
        ):
            raise Day20EvaluationV2Error(
                f"frozen eval {hash_field} drifted: {record.get('sample_id')}"
            )
    messages = record.get("messages")
    if not isinstance(messages, list) or record.get("messages_hash") != _sha256_prefixed(
        {"domain": "day10.messages", "schema_version": 1, "messages": messages}
    ):
        raise Day20EvaluationV2Error(
            f"frozen eval messages drifted: {record.get('sample_id')}"
        )
    input_ids = record.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in input_ids
        )
        or record.get("input_ids_hash")
        != _sha256_prefixed(
            {"domain": "day10.input_ids", "schema_version": 1, "input_ids": input_ids}
        )
        or record.get("input_token_count") != len(input_ids)
    ):
        raise Day20EvaluationV2Error(
            f"frozen eval token evidence drifted: {record.get('sample_id')}"
        )
    maximum = record.get("generation_max_new_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise Day20EvaluationV2Error("invalid frozen generation token ceiling")


def load_frozen_eval_manifest(
    path: Path, *, scorer_path: Path
) -> tuple[dict[str, Any], Any]:
    """Validate the frozen manifest, source hash, registry, and all row hashes."""

    resolved = path.expanduser().resolve()
    scorer = scorer_path.expanduser().resolve()
    manifest = load_json(resolved)
    header = manifest.get("header")
    records = manifest.get("records")
    if not isinstance(header, dict) or not isinstance(records, list):
        raise Day20EvaluationV2Error("frozen eval manifest is malformed")
    header_without_hash = dict(header)
    expected_manifest_hash = header_without_hash.pop("manifest_hash", None)
    canonical_manifest = {
        "header": header_without_hash,
        "records": sorted(records, key=lambda row: row.get("sample_id", "")),
    }
    if (
        header.get("schema_version") != 1
        or header.get("domain") != "day10.frozen_eval_manifest"
        or header.get("manifest_version") != "day10_frozen_eval_manifest_v3"
        or expected_manifest_hash != _sha256_prefixed(canonical_manifest)
        or header.get("scorer_source_sha256") != file_sha256(scorer)
    ):
        raise Day20EvaluationV2Error("frozen eval manifest identity drifted")
    order = (header.get("evaluation_order") or {}).get("dev")
    if (
        not isinstance(order, list)
        or len(order) != EXPECTED_RECORDS["full112"]
        or len(set(order)) != EXPECTED_RECORDS["full112"]
        or len(records) != 160
    ):
        raise Day20EvaluationV2Error("frozen dev order/count drifted")
    by_id: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise Day20EvaluationV2Error("frozen eval record is not an object")
        sample_id = raw_record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
            raise Day20EvaluationV2Error("frozen eval sample IDs are invalid")
        _verify_eval_record(raw_record)
        by_id[sample_id] = raw_record
    try:
        selected = [by_id[sample_id] for sample_id in order]
    except KeyError as error:
        raise Day20EvaluationV2Error("frozen dev order references a missing row") from error
    if (
        any(record.get("evaluation_split") != "dev" for record in selected)
        or Counter(record["slice"] for record in selected)
        != Counter({skill: 28 for skill in SKILLS})
    ):
        raise Day20EvaluationV2Error("frozen dev split distribution drifted")

    spec = importlib.util.spec_from_file_location("day20_v2_frozen_scorers", scorer)
    if spec is None or spec.loader is None:
        raise Day20EvaluationV2Error("cannot import the frozen scorer")
    scorer_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(scorer_module)
    except Exception as error:
        raise Day20EvaluationV2Error("cannot import the frozen scorer") from error
    if not callable(getattr(scorer_module, "score_prediction", None)):
        raise Day20EvaluationV2Error("frozen scorer has no score_prediction")
    registry = getattr(scorer_module, "SCORER_REGISTRY", None)
    registry_version = getattr(scorer_module, "SCORER_REGISTRY_VERSION", None)
    if (
        not isinstance(registry, dict)
        or not isinstance(registry_version, str)
        or registry.get("registry_version") != registry_version
    ):
        raise Day20EvaluationV2Error("frozen scorer registry is missing")
    registry_slices = registry.get("slices")
    if not isinstance(registry_slices, dict):
        raise Day20EvaluationV2Error("frozen scorer slice registry is missing")
    for record in records:
        skill = record["slice"]
        registered = registry_slices.get(skill)
        if (
            not isinstance(registered, dict)
            or record.get("scorer_version") != registered.get("scorer_version")
            or record.get("extractor_version") != registered.get("extractor_version")
        ):
            raise Day20EvaluationV2Error("frozen scorer/record version drifted")
    return manifest, scorer_module


def _verify_prepared_dataset(
    identity: Mapping[str, Any], *, run_root: Path, name: str
) -> list[dict[str, Any]]:
    path = _require_within(
        Path(str(identity.get("path", ""))), run_root / "data", f"{name} dataset"
    )
    rows = load_jsonl(path)
    sample_ids = [row.get("sample_id") for row in rows]
    tokens = [row.get("qwen35_supervised_tokens") for row in rows]
    if (
        file_sha256(path) != identity.get("file_sha256")
        or identity.get("records") != len(rows)
        or not all(isinstance(item, str) and item for item in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or identity.get("ordered_sample_ids") != sample_ids
        or identity.get("selection_sha256") != object_sha256(sample_ids)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in tokens
        )
        or identity.get("ordered_supervised_tokens") != tokens
        or identity.get("supervised_tokens") != sum(tokens)
    ):
        raise Day20EvaluationV2Error(f"prepared {name} dataset identity drifted")
    by_skill = {
        skill: sum(
            int(row["qwen35_supervised_tokens"])
            for row in rows
            if row.get("skill") == skill
        )
        for skill in SKILLS
    }
    by_format: dict[str, int] = {}
    records_by_format: Counter[str] = Counter()
    for row in rows:
        target_format = row.get("target_format")
        if not isinstance(target_format, str):
            raise Day20EvaluationV2Error("prepared target format is missing")
        by_format[target_format] = by_format.get(target_format, 0) + int(
            row["qwen35_supervised_tokens"]
        )
        records_by_format[target_format] += 1
    temporal_mix = identity.get("temporal_mix_audit")
    if (
        identity.get("supervised_tokens_by_skill") != by_skill
        or identity.get("supervised_tokens_by_format") != by_format
        or identity.get("records_by_format") != dict(sorted(records_by_format.items()))
        or not isinstance(temporal_mix, dict)
        or temporal_mix.get("status") != "pass"
        or identity.get("temporal_mix_sha256") != object_sha256(temporal_mix)
    ):
        raise Day20EvaluationV2Error(f"prepared {name} dataset audit drifted")
    return rows


def verify_experiment_manifest(
    path: Path, *, eval_manifest_path: Path, frozen_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the v2 experiment, source, training data, and diagnostic lineage."""

    resolved = path.expanduser().resolve()
    manifest = load_json(resolved)
    _self_hash(manifest, "manifest_sha256")
    contract = manifest.get("contract")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("domain") != EXPERIMENT_MANIFEST_DOMAIN
        or manifest.get("status") != "prepared_stage_a_not_started"
        or not isinstance(contract, dict)
        or contract.get("candidate_factory_version")
        != candidate_factory.CONTRACT_VERSION
        or contract.get("data_contract_version") != V2_CONTRACT_VERSION
    ):
        raise Day20EvaluationV2Error("v2 experiment manifest identity drifted")
    run_root = Path(str(manifest.get("run_root", ""))).expanduser().resolve()
    if resolved != run_root / "DAY20-V2-MANIFEST.json":
        raise Day20EvaluationV2Error("experiment manifest path differs from run_root")
    marker = run_root / ".day20-v2-run-root"
    if (
        not marker.is_file()
        or marker.is_symlink()
        or marker.read_text(encoding="utf-8").strip()
        != "day20-qwen35-candidate-factory-v2"
    ):
        raise Day20EvaluationV2Error("Day 20 v2 run-root marker drifted")

    source = manifest.get("source_expansion")
    if not isinstance(source, dict):
        raise Day20EvaluationV2Error("source expansion provenance is missing")
    source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    source_manifest = load_json(source_path)
    prepared_base = manifest.get("base_model_identity")
    if (
        file_sha256(source_path) != source.get("file_sha256")
        or _self_hash(source_manifest, "manifest_sha256")
        != source.get("content_sha256")
        or source_manifest.get("schema_version") != 2
        or source_manifest.get("domain") != SOURCE_EXPANSION_DOMAIN
        or source_manifest.get("status") != "pass"
        or source_manifest.get("dataset_code_execution") is not False
        or source_manifest.get("adapter_version") != source.get("adapter_version")
        or not isinstance(prepared_base, dict)
        or source_manifest.get("model_path") != prepared_base.get("path")
    ):
        raise Day20EvaluationV2Error("source expansion provenance drifted")
    model_path = Path(str(prepared_base["path"])).resolve()
    tokenizer_names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    tokenizer_files = {
        name: file_sha256(model_path / name)
        for name in tokenizer_names
        if (model_path / name).is_file()
    }
    if source_manifest.get("tokenizer_files") != dict(sorted(tokenizer_files.items())):
        raise Day20EvaluationV2Error("source tokenizer provenance drifted")
    for block_name in ("day09_manifest",):
        block = source_manifest.get(block_name)
        if (
            not isinstance(block, dict)
            or file_sha256(Path(str(block.get("path", ""))).resolve())
            != block.get("file_sha256")
        ):
            raise Day20EvaluationV2Error(f"source {block_name} provenance drifted")
    base_sources = source_manifest.get("base_sources")
    outputs = source_manifest.get("outputs")
    if (
        not isinstance(base_sources, dict)
        or set(base_sources) != set(SKILLS)
        or not isinstance(outputs, dict)
        or set(outputs) != set(SKILLS)
    ):
        raise Day20EvaluationV2Error("source expansion input/output inventory drifted")
    for skill in SKILLS:
        base_source = base_sources[skill]
        output = outputs[skill]
        if not isinstance(base_source, dict) or not isinstance(output, dict):
            raise Day20EvaluationV2Error("source expansion identity is malformed")
        if file_sha256(Path(str(base_source.get("path", ""))).resolve()) != base_source.get(
            "file_sha256"
        ):
            raise Day20EvaluationV2Error(f"base source provenance drifted: {skill}")
        output_path = Path(str(output.get("path", ""))).resolve()
        output_rows = load_jsonl(output_path)
        supervised_tokens = 0
        adapters: Counter[str] = Counter()
        for row in output_rows:
            tokenization = row.get("qwen35_tokenization")
            lineage = row.get("source_lineage")
            if not isinstance(tokenization, dict) or not isinstance(lineage, dict):
                raise Day20EvaluationV2Error(
                    f"expanded source row provenance is missing: {skill}"
                )
            tokens = tokenization.get("supervised_tokens")
            adapter_name = lineage.get("adapter")
            if (
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens <= 0
                or not isinstance(adapter_name, str)
                or not adapter_name
            ):
                raise Day20EvaluationV2Error(
                    f"expanded source row identity drifted: {skill}"
                )
            supervised_tokens += tokens
            adapters[adapter_name] += 1
        if (
            file_sha256(output_path) != output.get("file_sha256")
            or output.get("records") != len(output_rows)
            or output.get("supervised_tokens") != supervised_tokens
            or output.get("records_by_adapter") != dict(sorted(adapters.items()))
        ):
            raise Day20EvaluationV2Error(f"expanded source provenance drifted: {skill}")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise Day20EvaluationV2Error("experiment datasets block is missing")
    probe_identity = datasets.get("probe")
    main_identity = datasets.get("main")
    diagnostic = datasets.get("diagnostic")
    if not all(isinstance(item, dict) for item in (probe_identity, main_identity, diagnostic)):
        raise Day20EvaluationV2Error("experiment dataset provenance is incomplete")
    probe_rows = _verify_prepared_dataset(
        probe_identity, run_root=run_root, name="probe"  # type: ignore[arg-type]
    )
    main_rows = _verify_prepared_dataset(
        main_identity, run_root=run_root, name="main"  # type: ignore[arg-type]
    )
    if not {row["sample_id"] for row in probe_rows} <= {
        row["sample_id"] for row in main_rows
    }:
        raise Day20EvaluationV2Error("prepared probe is not nested in main")

    diagnostic_path = _require_within(
        Path(str(diagnostic.get("path", ""))),  # type: ignore[union-attr]
        run_root / "data",
        "diagnostic dataset",
    )
    diagnostic_rows = load_jsonl(diagnostic_path)
    diagnostic_ids = [row.get("sample_id") for row in diagnostic_rows]
    eval_path = eval_manifest_path.expanduser().resolve()
    eval_header = frozen_manifest.get("header")
    eval_records = frozen_manifest.get("records")
    if not isinstance(eval_header, dict) or not isinstance(eval_records, list):
        raise Day20EvaluationV2Error("frozen manifest input is malformed")
    by_id = {row["sample_id"]: row for row in eval_records if isinstance(row, dict)}
    try:
        expected_diagnostic_rows = [by_id[sample_id] for sample_id in diagnostic_ids]
    except KeyError as error:
        raise Day20EvaluationV2Error("diagnostic sample is outside frozen dev") from error
    if (
        file_sha256(diagnostic_path) != diagnostic.get("file_sha256")  # type: ignore[union-attr]
        or diagnostic.get("records") != EXPECTED_RECORDS["probe32"]  # type: ignore[union-attr]
        or diagnostic.get("records_by_skill") != {skill: 8 for skill in SKILLS}  # type: ignore[union-attr]
        or diagnostic.get("ordered_sample_ids") != diagnostic_ids  # type: ignore[union-attr]
        or diagnostic.get("selection_sha256") != object_sha256(diagnostic_ids)  # type: ignore[union-attr]
        or Path(str(diagnostic.get("eval_manifest_path", ""))).resolve() != eval_path  # type: ignore[union-attr]
        or diagnostic.get("eval_manifest_file_sha256") != file_sha256(eval_path)  # type: ignore[union-attr]
        or diagnostic_rows != expected_diagnostic_rows
        or Counter(row.get("slice") for row in diagnostic_rows)
        != Counter({skill: 8 for skill in SKILLS})
        or any(row.get("evaluation_split") != "dev" for row in diagnostic_rows)
    ):
        raise Day20EvaluationV2Error("diagnostic dataset provenance drifted")
    result = dict(manifest)
    result["_verified_run_root"] = str(run_root)
    result["_diagnostic_rows"] = diagnostic_rows
    return result


def validate_candidate_scope(
    candidate: str, eval_scope: str
) -> candidate_factory.CandidateIdentity:
    if eval_scope not in EXPECTED_RECORDS:
        raise Day20EvaluationV2Error("eval scope must be probe32 or full112")
    try:
        identity = candidate_factory.parse_candidate_id(candidate)
    except candidate_factory.CandidateFactoryV2Error as error:
        raise Day20EvaluationV2Error("candidate is outside the v2 allowlist") from error
    if identity.scope != eval_scope:
        raise Day20EvaluationV2Error("candidate scope differs from --eval-scope")
    if identity.model_role == "base" and candidate != candidate_factory.base_candidate_id(
        eval_scope
    ):
        raise Day20EvaluationV2Error("Base candidate identity drifted")
    return identity


def select_evaluation_records(
    *,
    frozen_manifest: Mapping[str, Any],
    experiment_manifest: Mapping[str, Any],
    eval_scope: str,
) -> list[dict[str, Any]]:
    records = frozen_manifest.get("records")
    header = frozen_manifest.get("header")
    if not isinstance(records, list) or not isinstance(header, dict):
        raise Day20EvaluationV2Error("frozen manifest is malformed")
    if eval_scope == "probe32":
        selected = experiment_manifest.get("_diagnostic_rows")
        if not isinstance(selected, list):
            raise Day20EvaluationV2Error("verified diagnostic rows are missing")
        result = [dict(row) for row in selected]
    elif eval_scope == "full112":
        order = (header.get("evaluation_order") or {}).get("dev")
        by_id = {row["sample_id"]: row for row in records if isinstance(row, dict)}
        if not isinstance(order, list):
            raise Day20EvaluationV2Error("frozen dev order is missing")
        result = [dict(by_id[sample_id]) for sample_id in order]
    else:
        raise Day20EvaluationV2Error("eval scope must be probe32 or full112")
    expected = EXPECTED_RECORDS[eval_scope]
    per_skill = expected // len(SKILLS)
    if (
        len(result) != expected
        or len({row.get("sample_id") for row in result}) != expected
        or Counter(row.get("slice") for row in result)
        != Counter({skill: per_skill for skill in SKILLS})
        or any(row.get("evaluation_split") != "dev" for row in result)
    ):
        raise Day20EvaluationV2Error("selected evaluation scope drifted")
    return result


def file_manifest(directory: Path, *, inference_only: bool = False) -> dict[str, dict[str, Any]]:
    root = _require_regular_directory(directory, "model/checkpoint")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
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
    return result


def verify_base_model_identity(
    model_path: Path, prepared_identity: Mapping[str, Any]
) -> dict[str, Any]:
    model = _require_regular_directory(model_path, "Base model")
    files = file_manifest(model)
    rebuilt = {
        "path": str(model),
        "files": files,
        "snapshot_sha256": object_sha256(files),
    }
    if (
        "config.json" not in files
        or not any(name.endswith(".safetensors") for name in files)
        or rebuilt != prepared_identity
    ):
        raise Day20EvaluationV2Error("Base model differs from prepared identity")
    return rebuilt


def _verify_adapter_config(adapter: Path) -> None:
    config = load_json(adapter / "adapter_config.json")
    required = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
    }
    if any(config.get(name) != expected for name, expected in required.items()):
        raise Day20EvaluationV2Error("LoRA adapter hyperparameters drifted")
    peft_type = str(config.get("peft_type", "")).upper()
    if peft_type and peft_type != "LORA":
        raise Day20EvaluationV2Error("checkpoint is not a LoRA adapter")


def resolve_candidate_provenance(
    *,
    candidate_identity: candidate_factory.CandidateIdentity,
    model_path: Path,
    adapter_path: Path | None,
    training_summary_path: Path | None,
    experiment_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = Path(str(experiment_manifest["_verified_run_root"])).resolve()
    prepared_base = experiment_manifest.get("base_model_identity")
    if not isinstance(prepared_base, dict):
        raise Day20EvaluationV2Error("prepared Base identity is missing")
    base_identity = verify_base_model_identity(model_path, prepared_base)

    training_identity: dict[str, Any] | None = None
    adapter_identity: dict[str, Any] | None = None
    if candidate_identity.model_role == "base":
        if adapter_path is not None or training_summary_path is not None:
            raise Day20EvaluationV2Error(
                "Base evaluation cannot load adapter/training artifacts"
            )
        checkpoint = Path(base_identity["path"])
        checkpoint_package: dict[str, Any] = {
            "kind": "base_snapshot",
            "path": str(checkpoint),
            "snapshot_sha256": base_identity["snapshot_sha256"],
        }
    else:
        if adapter_path is None or training_summary_path is None:
            raise Day20EvaluationV2Error(
                "LoRA evaluation requires --adapter and --training-summary"
            )
        summary_path = training_summary_path.expanduser().resolve()
        summary = verify_training_summary(
            summary_path, run_root=run_root, candidate=candidate_identity.id
        )
        expected_checkpoint = checkpoint_path(
            summary_path, run_root=run_root, candidate=candidate_identity.id
        )
        adapter = adapter_path.expanduser().resolve()
        if adapter != expected_checkpoint:
            raise Day20EvaluationV2Error(
                "--adapter differs from the verified candidate checkpoint"
            )
        label = next(
            name
            for name, value in summary["candidate_ids"].items()
            if value == candidate_identity.id
        )
        package = summary["checkpoint_packages"][label]
        if (
            summary.get("run_kind") != candidate_identity.run_kind
            or summary.get("seed") != candidate_identity.seed
            or str(summary.get("learning_rate")) != candidate_identity.learning_rate
            or summary.get("checkpoint_tokens", {}).get(label)
            != candidate_identity.target_tokens
            or summary.get("experiment_manifest", {}).get("content_sha256")
            != experiment_manifest.get("manifest_sha256")
            or summary.get("experiment_manifest", {}).get("file_sha256")
            != file_sha256(
                Path(str(summary["experiment_manifest"].get("path", ""))).resolve()
            )
        ):
            raise Day20EvaluationV2Error("training summary candidate lineage drifted")
        expected_dataset = experiment_manifest["datasets"][candidate_identity.run_kind]
        if (
            summary.get("dataset_file_sha256") != expected_dataset.get("file_sha256")
            or summary.get("temporal_mix_sha256")
            != expected_dataset.get("temporal_mix_sha256")
        ):
            raise Day20EvaluationV2Error("training dataset lineage drifted")
        _verify_adapter_config(adapter)
        inference_files = file_manifest(adapter, inference_only=True)
        package_files = package.get("files")
        if (
            not isinstance(package_files, dict)
            or not inference_files
            or any(package_files.get(name) != identity for name, identity in inference_files.items())
            or "adapter_config.json" not in inference_files
            or not any(name.endswith(".safetensors") for name in inference_files)
        ):
            raise Day20EvaluationV2Error("adapter inference package drifted")
        adapter_identity = {
            "path": str(adapter),
            "format": "peft_lora",
            "files": inference_files,
            "snapshot_sha256": object_sha256(inference_files),
        }
        checkpoint = adapter
        checkpoint_package = dict(package)
        training_identity = {
            "path": str(summary_path),
            "file_sha256": file_sha256(summary_path),
            "content_sha256": summary["summary_sha256"],
            "run_kind": summary["run_kind"],
            "seed": summary["seed"],
            "learning_rate": summary["learning_rate"],
            "checkpoint_label": label,
        }

    model_identity = {
        "base": base_identity,
        "adapter": adapter_identity,
        "expected_model_class": EXPECTED_MODEL_CLASS,
    }
    checkpoint_package_sha256 = object_sha256(checkpoint_package)
    model_identity_sha256 = object_sha256(model_identity)
    compact_model = {
        "model_identity_sha256": model_identity_sha256,
        "base_snapshot_sha256": base_identity["snapshot_sha256"],
        "adapter_snapshot_sha256": (
            adapter_identity["snapshot_sha256"] if adapter_identity else None
        ),
    }
    compact_checkpoint = {
        "checkpoint_package_sha256": checkpoint_package_sha256,
        "kind": checkpoint_package["kind"]
        if "kind" in checkpoint_package
        else "resumable_lora_checkpoint",
        "snapshot_sha256": checkpoint_package["snapshot_sha256"],
        "integrity_sha256": checkpoint_package.get("integrity_sha256"),
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_package": checkpoint_package,
        "checkpoint_package_sha256": checkpoint_package_sha256,
        "compact_checkpoint_package": compact_checkpoint,
        "model_identity": model_identity,
        "model_identity_sha256": model_identity_sha256,
        "compact_model_identity": compact_model,
        "training_identity": training_identity,
    }


def live_runtime_identity() -> dict[str, Any]:
    """Return the same pinned runtime identity required by v2 training."""

    try:
        identity = training_runtime_identity(EXPECTED_MS_SWIFT_COMMIT)
    except Exception as error:
        raise Day20EvaluationV2Error(
            "evaluation runtime/check-out identity drifted"
        ) from error
    versions = identity.get("versions")
    if versions != EXPECTED_RUNTIME_VERSIONS:
        raise Day20EvaluationV2Error("evaluation runtime versions drifted")
    runtime_sha256 = identity.get("runtime_sha256")
    if runtime_sha256 != object_sha256(
        {key: value for key, value in identity.items() if key != "runtime_sha256"}
    ):
        raise Day20EvaluationV2Error("evaluation runtime identity hash drifted")
    return identity


def _validate_qwen35_template(template: Any) -> None:
    template_meta = getattr(template, "template_meta", None)
    if (
        getattr(template_meta, "template_type", None) != "qwen3_5"
        or getattr(template_meta, "non_thinking_prefix", None)
        != NON_THINKING_PREFIX
        or getattr(template, "enable_thinking", None) is not False
        or getattr(template, "add_non_thinking_prefix", None) is not True
    ):
        raise Day20EvaluationV2Error("Qwen3.5 inference template contract drifted")


def build_live_generated_token_decoder(
    model_path: Path,
) -> Callable[[Sequence[int]], str]:
    """Load only the pinned processor/template and return its exact decoder."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    try:
        from swift import get_model_processor, get_template
    except ImportError as error:
        raise Day20EvaluationV2Error("ms-swift decoder runtime is unavailable") from error
    try:
        _, processor = get_model_processor(
            str(model_path.expanduser().resolve()),
            model_type="qwen3_5",
            load_model=False,
            use_hf=True,
            download_model=False,
        )
        template = get_template(
            processor,
            max_length=4096,
            padding_free=False,
            enable_thinking=False,
            add_non_thinking_prefix=True,
        )
    except Exception as error:
        raise Day20EvaluationV2Error(
            "cannot construct the pinned Qwen3.5 token decoder"
        ) from error
    _validate_qwen35_template(template)

    def decode_generated_token_ids(token_ids: Sequence[int]) -> str:
        try:
            # token_ids_sha256 performs the strict integer/range validation.
            token_ids_sha256(token_ids)
            decoded = template.decode_generate_ids(
                list(token_ids), first_token=False
            )
        except Exception as error:
            raise Day20EvaluationV2Error(
                "pinned template rejected generated token IDs"
            ) from error
        if not isinstance(decoded, str):
            raise Day20EvaluationV2Error("pinned generated-token decode is not text")
        return decoded

    return decode_generated_token_ids


def build_protocol(
    *,
    eval_scope: str,
    records: int,
    eval_manifest_path: Path,
    frozen_manifest: Mapping[str, Any],
    experiment_manifest_path: Path,
    experiment_manifest: Mapping[str, Any],
    scorer_path: Path,
    scorer_version: str,
    adapter_loaded: bool,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    diagnostic = experiment_manifest["datasets"]["diagnostic"]
    eval_header = frozen_manifest["header"]
    return {
        "candidate_factory": {
            "version": candidate_factory.CONTRACT_VERSION,
            "path": str((here / "day20_candidate_factory_v2.py").resolve()),
            "file_sha256": file_sha256(here / "day20_candidate_factory_v2.py"),
        },
        "data_contract": {
            "version": V2_CONTRACT_VERSION,
            "path": str((here / "day20_contract_v2.py").resolve()),
            "file_sha256": file_sha256(here / "day20_contract_v2.py"),
        },
        "eval_manifest": {
            "path": str(eval_manifest_path.resolve()),
            "file_sha256": file_sha256(eval_manifest_path.resolve()),
            "content_sha256": str(eval_header["manifest_hash"]).removeprefix(
                "sha256:"
            ),
            "manifest_version": eval_header["manifest_version"],
        },
        "experiment_manifest": {
            "path": str(experiment_manifest_path.resolve()),
            "file_sha256": file_sha256(experiment_manifest_path.resolve()),
            "content_sha256": experiment_manifest["manifest_sha256"],
        },
        "diagnostic": {
            "path": diagnostic["path"],
            "file_sha256": diagnostic["file_sha256"],
            "records": diagnostic["records"],
            "selection_sha256": diagnostic["selection_sha256"],
        },
        "scorer": {
            "path": str(scorer_path.resolve()),
            "file_sha256": file_sha256(scorer_path.resolve()),
            "version": scorer_version,
        },
        "response_adapter": {
            "path": str((here / "qwen35_response_adapter_v3.py").resolve()),
            "file_sha256": file_sha256(here / "qwen35_response_adapter_v3.py"),
            "version": ADAPTER_VERSION,
        },
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": file_sha256(Path(__file__).resolve()),
            "version": EVALUATOR_VERSION,
        },
        "runtime": dict(runtime_identity),
        "generation": {
            "split": "dev",
            "eval_scope": eval_scope,
            "records": records,
            "template": "qwen3_5",
            "backend": "hf_transformers",
            "use_mcore_gdn": False,
            "enable_thinking": False,
            "add_non_thinking_prefix": True,
            "non_thinking_prefix_sha256": text_sha256(NON_THINKING_PREFIX),
            "greedy": True,
            "seed": GENERATION_SEED,
            "response_capture": "ms-swift RequestConfig(return_details=True)",
            "generated_token_ids_retained": True,
            "generated_token_decode": {
                "method": "Template.decode_generate_ids",
                "first_token": False,
                "runtime_sha256": runtime_identity["runtime_sha256"],
            },
            "adapter_loaded_unmerged": adapter_loaded,
            "code_execution": False,
        },
    }


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def build_code_contract(raw_output: str, raw_prompt: str) -> dict[str, Any]:
    try:
        validated = validate_raw_code_continuation(raw_output, raw_prompt)
    except Day20V2ContractError as error:
        return {
            "version": V2_CONTRACT_VERSION,
            "validator": "validate_raw_code_continuation",
            "valid": False,
            "execution_eligible": False,
            "execution_performed": False,
            "evidence": None,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    return {
        "version": V2_CONTRACT_VERSION,
        "validator": "validate_raw_code_continuation",
        "valid": True,
        "execution_eligible": True,
        "execution_performed": False,
        "evidence": validated.as_evidence(),
        "error": None,
    }


def build_response_capture(
    *,
    message_content: str,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    generated_only_text: str,
) -> dict[str, Any]:
    adapter = adapt_generated_token_decode(
        message_content,
        generated_token_ids=generated_token_ids,
        generated_only_text=generated_only_text,
    )
    prompt_ids = list(prompt_token_ids)
    generated_ids = list(generated_token_ids)
    return {
        "method": "ms_swift_return_details",
        "boundary_source": "generated_token_ids_decode",
        "message_content": message_content,
        "message_content_sha256": text_sha256(message_content),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": token_ids_sha256(generated_ids),
        "generated_only_text": generated_only_text,
        "generated_only_text_sha256": text_sha256(generated_only_text),
        "response_adapter": adapter,
    }


def build_prediction_row(
    *,
    ordinal: int,
    candidate_identity: candidate_factory.CandidateIdentity,
    eval_scope: str,
    record: Mapping[str, Any],
    message_content: str,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    generated_only_text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    scorer: Any,
    compact_model_identity: Mapping[str, Any],
    compact_checkpoint_package: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one self-hashed raw row; no code is executed here."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise Day20EvaluationV2Error("prediction ordinal must be a positive int")
    if not isinstance(message_content, str) or not isinstance(generated_only_text, str):
        raise Day20EvaluationV2Error("response text evidence must be text")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise Day20EvaluationV2Error("finish reason is missing")
    skill = record.get("slice")
    if skill not in SKILLS:
        raise Day20EvaluationV2Error("prediction record has an unknown skill")
    maximum = record.get("generation_max_new_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise Day20EvaluationV2Error("prediction token ceiling is invalid")
    prompt_ids = list(prompt_token_ids)
    generated_ids = list(generated_token_ids)
    try:
        response_capture = build_response_capture(
            message_content=message_content,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            generated_only_text=generated_only_text,
        )
    except (TypeError, Qwen35ResponseAdapterV3Error) as error:
        raise Day20EvaluationV2Error("invalid response token evidence") from error
    if prompt_tokens != len(prompt_ids) or completion_tokens != len(generated_ids):
        raise Day20EvaluationV2Error("response token accounting mismatch")
    raw_prompt = record.get("raw_prompt")
    reference = record.get("reference")
    if not isinstance(raw_prompt, str) or not isinstance(reference, str):
        raise Day20EvaluationV2Error("raw prompt/reference must be text")
    raw_output = generated_only_text
    try:
        scorer_result = scorer.score_prediction(skill, raw_output, reference)
    except Exception as error:
        raise Day20EvaluationV2Error(
            f"frozen scorer failed: {record.get('sample_id')}"
        ) from error
    if not isinstance(scorer_result, dict):
        raise Day20EvaluationV2Error("frozen scorer result is not an object")
    code_contract = (
        build_code_contract(raw_output, raw_prompt) if skill == "code" else None
    )
    anomalies: list[str] = []
    if raw_output == "":
        anomalies.append("empty_output")
    elif not raw_output.strip():
        anomalies.append("whitespace_only_output")
    if finish_reason == "length" or completion_tokens >= maximum:
        anomalies.append("generation_ceiling")
    repetition = repeated_ngram_ratio(raw_output)
    if repetition >= 0.5:
        anomalies.append("high_4gram_repetition")
    relation = response_capture["response_adapter"]["message_content_relation"]
    if relation == "divergent_diagnostic_only":
        anomalies.append("message_content_diverges_from_generated_decode")
    if code_contract is not None and not code_contract["valid"]:
        anomalies.append("strict_code_continuation_invalid")
    if skill == "code":
        format_compliant = bool(code_contract and code_contract["valid"])
    else:
        format_compliant = bool(raw_output.strip()) and scorer_result.get(
            "parse_status"
        ) == "ok"
    generation = {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "seed": GENERATION_SEED,
        "max_new_tokens": maximum,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": PREDICTION_DOMAIN,
        "ordinal": ordinal,
        "candidate": candidate_identity.id,
        "candidate_identity": candidate_identity.as_dict(),
        "eval_scope": eval_scope,
        "sample_id": record["sample_id"],
        "skill": skill,
        "target_format": TARGET_FORMAT_BY_SKILL[skill],
        "raw_prompt": raw_prompt,
        "raw_prompt_sha256": text_sha256(raw_prompt),
        "reference": reference,
        "reference_sha256": text_sha256(reference),
        "prompt_messages_sha256": str(record["messages_hash"]).removeprefix(
            "sha256:"
        ),
        "raw_output": raw_output,
        "raw_output_sha256": text_sha256(raw_output),
        "response_capture": response_capture,
        "model_identity": dict(compact_model_identity),
        "checkpoint_package": dict(compact_checkpoint_package),
        "generation": generation,
        "scorer_result": scorer_result,
        "code_contract": code_contract,
        "format_compliant": format_compliant,
        "anomalies": anomalies,
        "repeated_4gram_ratio": repetition,
    }
    row["row_sha256"] = object_sha256(row)
    return row


def _numeric_score(row: Mapping[str, Any]) -> float | None:
    scorer_result = row.get("scorer_result")
    if not isinstance(scorer_result, Mapping):
        return None
    score = scorer_result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    value = float(score)
    if not math.isfinite(value):
        raise Day20EvaluationV2Error("non-finite frozen score")
    return value


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_skill: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        skill_rows = [row for row in rows if row.get("skill") == skill]
        scores = [score for row in skill_rows if (score := _numeric_score(row)) is not None]
        correct = sum(scores) if scores else None
        entry: dict[str, Any] = {
            "records": len(skill_rows),
            "scored_records": len(scores),
            "correct": correct,
            "accuracy": correct / len(scores) if correct is not None else None,
            "format_compliant": sum(
                row.get("format_compliant") is True for row in skill_rows
            ),
            "anomalous_records": sum(bool(row.get("anomalies")) for row in skill_rows),
            "generation_ceiling_records": sum(
                "generation_ceiling" in (row.get("anomalies") or [])
                for row in skill_rows
            ),
            "output_tokens": sum(
                int(row["generation"]["completion_tokens"]) for row in skill_rows
            ),
        }
        if skill == "code":
            entry["strict_continuation_valid"] = sum(
                bool(row.get("code_contract", {}).get("valid"))
                for row in skill_rows
            )
            entry["code_sandbox_execution_eligible"] = sum(
                bool(row.get("code_contract", {}).get("execution_eligible"))
                for row in skill_rows
            )
            entry["executable_score_status"] = "sandbox_required"
        by_skill[skill] = entry
    non_code_rows = [row for row in rows if row.get("skill") != "code"]
    non_code_scores = [
        score for row in non_code_rows if (score := _numeric_score(row)) is not None
    ]
    return {
        "by_skill": by_skill,
        "non_code_correct": sum(non_code_scores),
        "non_code_total": len(non_code_scores),
        "format_compliant": sum(row.get("format_compliant") is True for row in rows),
        "anomalous_records": sum(bool(row.get("anomalies")) for row in rows),
        "output_tokens": sum(
            int(row["generation"]["completion_tokens"]) for row in rows
        ),
        "code_sandbox_execution_eligible": by_skill["code"][
            "code_sandbox_execution_eligible"
        ],
        "code_execution_performed": False,
    }


def build_summary(
    *,
    candidate_identity: candidate_factory.CandidateIdentity,
    eval_scope: str,
    provenance: Mapping[str, Any],
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    predictions_path: Path,
    predictions_file_sha256: str,
    elapsed_seconds: float,
    peak_cuda_memory_gib: float,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != EXPECTED_RECORDS[eval_scope]:
        raise Day20EvaluationV2Error("cannot summarize an incomplete evaluation")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0
        for value in (elapsed_seconds, peak_cuda_memory_gib)
    ):
        raise Day20EvaluationV2Error("runtime metrics are invalid")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": SUMMARY_DOMAIN,
        "status": "complete_with_code_sandbox_required",
        "created_at_utc": utc_now(),
        "candidate": candidate_identity.id,
        "candidate_identity": candidate_identity.as_dict(),
        "eval_scope": eval_scope,
        "checkpoint": provenance["checkpoint"],
        "checkpoint_package": provenance["checkpoint_package"],
        "checkpoint_package_sha256": provenance["checkpoint_package_sha256"],
        "model_identity": provenance["model_identity"],
        "model_identity_sha256": provenance["model_identity_sha256"],
        "training_identity": provenance["training_identity"],
        "protocol": dict(protocol),
        "metrics": aggregate_metrics(rows),
        "runtime_metrics": {
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_gib": peak_cuda_memory_gib,
            "versions": dict(runtime_identity["versions"]),
            "runtime_identity": dict(runtime_identity),
        },
        "predictions": {
            "path": str(predictions_path.resolve()),
            "records": len(rows),
            "file_sha256": predictions_file_sha256,
            "content_sha256": object_sha256(list(rows)),
            "ordered_sample_ids_sha256": object_sha256(
                [row["sample_id"] for row in rows]
            ),
        },
    }
    summary["summary_sha256"] = object_sha256(summary)
    return summary


def verify_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_records: Sequence[Mapping[str, Any]],
    candidate_identity: candidate_factory.CandidateIdentity,
    eval_scope: str,
    scorer: Any,
    compact_model_identity: Mapping[str, Any],
    compact_checkpoint_package: Mapping[str, Any],
    generated_token_decoder: Callable[[Sequence[int]], str],
) -> None:
    if len(rows) != EXPECTED_RECORDS[eval_scope] or len(rows) != len(
        selected_records
    ):
        raise Day20EvaluationV2Error("prediction row count differs from scope")
    for ordinal, (row, record) in enumerate(zip(rows, selected_records), 1):
        if not isinstance(row, Mapping):
            raise Day20EvaluationV2Error("prediction row is not an object")
        capture = row.get("response_capture")
        generation = row.get("generation")
        if not isinstance(capture, Mapping) or not isinstance(generation, Mapping):
            raise Day20EvaluationV2Error("prediction response evidence is missing")
        generated_ids = capture.get("generated_token_ids")
        generated_text = capture.get("generated_only_text")
        try:
            live_decoded_text = generated_token_decoder(generated_ids)  # type: ignore[arg-type]
        except Exception as error:
            raise Day20EvaluationV2Error(
                f"cannot re-decode generated token IDs at ordinal {ordinal}"
            ) from error
        if not isinstance(generated_text, str) or live_decoded_text != generated_text:
            raise Day20EvaluationV2Error(
                f"generated token/text boundary drifted at ordinal {ordinal}"
            )
        try:
            validate_response_capture(
                capture,
                raw_output=row.get("raw_output"),  # type: ignore[arg-type]
                generation=generation,
            )
        except (TypeError, Qwen35ResponseAdapterV3Error) as error:
            raise Day20EvaluationV2Error(
                f"response capture drifted at ordinal {ordinal}"
            ) from error
        rebuilt = build_prediction_row(
            ordinal=ordinal,
            candidate_identity=candidate_identity,
            eval_scope=eval_scope,
            record=record,
            message_content=capture["message_content"],
            prompt_token_ids=capture["prompt_token_ids"],
            generated_token_ids=capture["generated_token_ids"],
            generated_only_text=capture["generated_only_text"],
            finish_reason=generation["finish_reason"],
            prompt_tokens=generation["prompt_tokens"],
            completion_tokens=generation["completion_tokens"],
            scorer=scorer,
            compact_model_identity=compact_model_identity,
            compact_checkpoint_package=compact_checkpoint_package,
        )
        if dict(row) != rebuilt:
            raise Day20EvaluationV2Error(
                f"prediction row drifted at ordinal {ordinal}"
            )


def _path_from_protocol(
    summary: Mapping[str, Any], component: str
) -> Path:
    protocol = summary.get("protocol")
    identity = protocol.get(component) if isinstance(protocol, Mapping) else None
    if not isinstance(identity, Mapping) or not isinstance(identity.get("path"), str):
        raise Day20EvaluationV2Error(f"summary {component} provenance is missing")
    return Path(identity["path"]).expanduser().resolve()


def _validate_runtime_metrics(
    runtime: Any, *, expected_runtime_identity: Mapping[str, Any]
) -> None:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "elapsed_seconds",
        "peak_cuda_memory_gib",
        "versions",
        "runtime_identity",
    }:
        raise Day20EvaluationV2Error("runtime metrics are malformed")
    for name in ("elapsed_seconds", "peak_cuda_memory_gib"):
        value = runtime.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise Day20EvaluationV2Error(f"runtime metric {name} is invalid")
    versions = runtime.get("versions")
    if versions != EXPECTED_RUNTIME_VERSIONS:
        raise Day20EvaluationV2Error("evaluation runtime versions drifted")
    if runtime.get("runtime_identity") != expected_runtime_identity:
        raise Day20EvaluationV2Error("evaluation runtime/check-out provenance drifted")


def verify_published_pair(
    predictions_path: Path,
    summary_path: Path,
    *,
    expected_scope: str | None = None,
    expected_candidate: str | None = None,
) -> dict[str, Any]:
    """Fully revalidate a published raw pair and all of its live provenance."""

    predictions = predictions_path.expanduser().resolve()
    summary_file = summary_path.expanduser().resolve()
    summary = load_json(summary_file)
    _self_hash(summary, "summary_sha256")
    required_summary_fields = {
        "schema_version",
        "domain",
        "status",
        "created_at_utc",
        "candidate",
        "candidate_identity",
        "eval_scope",
        "checkpoint",
        "checkpoint_package",
        "checkpoint_package_sha256",
        "model_identity",
        "model_identity_sha256",
        "training_identity",
        "protocol",
        "metrics",
        "runtime_metrics",
        "predictions",
        "summary_sha256",
    }
    if set(summary) != required_summary_fields or (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("domain") != SUMMARY_DOMAIN
        or summary.get("status") != "complete_with_code_sandbox_required"
    ):
        raise Day20EvaluationV2Error("raw evaluation summary schema drifted")
    candidate = summary.get("candidate")
    eval_scope = summary.get("eval_scope")
    if not isinstance(candidate, str) or not isinstance(eval_scope, str):
        raise Day20EvaluationV2Error("summary candidate/scope is missing")
    if expected_candidate is not None and candidate != expected_candidate:
        raise Day20EvaluationV2Error("published candidate differs from expectation")
    if expected_scope is not None and eval_scope != expected_scope:
        raise Day20EvaluationV2Error("published scope differs from expectation")
    candidate_identity = validate_candidate_scope(candidate, eval_scope)
    if summary.get("candidate_identity") != candidate_identity.as_dict():
        raise Day20EvaluationV2Error("summary candidate identity drifted")
    created_at = summary.get("created_at_utc")
    try:
        parsed_created_at = datetime.fromisoformat(str(created_at))
    except ValueError as error:
        raise Day20EvaluationV2Error("summary creation timestamp is invalid") from error
    if parsed_created_at.tzinfo is None:
        raise Day20EvaluationV2Error("summary creation timestamp lacks timezone")

    eval_manifest_path = _path_from_protocol(summary, "eval_manifest")
    scorer_path = _path_from_protocol(summary, "scorer")
    experiment_manifest_path = _path_from_protocol(summary, "experiment_manifest")
    frozen_manifest, scorer = load_frozen_eval_manifest(
        eval_manifest_path, scorer_path=scorer_path
    )
    experiment = verify_experiment_manifest(
        experiment_manifest_path,
        eval_manifest_path=eval_manifest_path,
        frozen_manifest=frozen_manifest,
    )
    run_root = Path(str(experiment["_verified_run_root"])).resolve()
    _require_within(predictions, run_root / "eval", "published predictions")
    _require_within(summary_file, run_root / "eval", "published summary")
    expected_predictions, expected_summary = artifact_paths(
        predictions.parent, candidate=candidate, eval_scope=eval_scope
    )
    if predictions != expected_predictions or summary_file != expected_summary:
        raise Day20EvaluationV2Error("published raw artifact name/location drifted")
    selected = select_evaluation_records(
        frozen_manifest=frozen_manifest,
        experiment_manifest=experiment,
        eval_scope=eval_scope,
    )
    summary_model = summary.get("model_identity")
    if not isinstance(summary_model, Mapping):
        raise Day20EvaluationV2Error("summary model identity is missing")
    base = summary_model.get("base")
    if not isinstance(base, Mapping) or not isinstance(base.get("path"), str):
        raise Day20EvaluationV2Error("summary Base identity is missing")
    adapter_path: Path | None = None
    training_summary_path: Path | None = None
    if candidate_identity.model_role == "lora":
        adapter = summary_model.get("adapter")
        training = summary.get("training_identity")
        if not isinstance(adapter, Mapping) or not isinstance(adapter.get("path"), str):
            raise Day20EvaluationV2Error("summary adapter identity is missing")
        if not isinstance(training, Mapping) or not isinstance(training.get("path"), str):
            raise Day20EvaluationV2Error("summary training identity is missing")
        adapter_path = Path(adapter["path"])
        training_summary_path = Path(training["path"])
    provenance = resolve_candidate_provenance(
        candidate_identity=candidate_identity,
        model_path=Path(base["path"]),
        adapter_path=adapter_path,
        training_summary_path=training_summary_path,
        experiment_manifest=experiment,
    )
    for field in (
        "checkpoint",
        "checkpoint_package",
        "checkpoint_package_sha256",
        "model_identity",
        "model_identity_sha256",
        "training_identity",
    ):
        if summary.get(field) != provenance[field]:
            raise Day20EvaluationV2Error(f"summary {field} provenance drifted")

    scorer_version = getattr(scorer, "SCORER_REGISTRY_VERSION")
    protocol = build_protocol(
        eval_scope=eval_scope,
        records=len(selected),
        eval_manifest_path=eval_manifest_path,
        frozen_manifest=frozen_manifest,
        experiment_manifest_path=experiment_manifest_path,
        experiment_manifest=experiment,
        scorer_path=scorer_path,
        scorer_version=scorer_version,
        adapter_loaded=candidate_identity.model_role == "lora",
        runtime_identity=live_runtime_identity(),
    )
    if summary.get("protocol") != protocol:
        raise Day20EvaluationV2Error("raw evaluation protocol drifted")
    runtime = protocol["runtime"]
    generated_token_decoder = build_live_generated_token_decoder(
        Path(base["path"])
    )
    rows = load_jsonl(predictions)
    verify_prediction_rows(
        rows,
        selected_records=selected,
        candidate_identity=candidate_identity,
        eval_scope=eval_scope,
        scorer=scorer,
        compact_model_identity=provenance["compact_model_identity"],
        compact_checkpoint_package=provenance["compact_checkpoint_package"],
        generated_token_decoder=generated_token_decoder,
    )
    prediction_identity = summary.get("predictions")
    expected_prediction_identity = {
        "path": str(predictions),
        "records": len(rows),
        "file_sha256": file_sha256(predictions),
        "content_sha256": object_sha256(rows),
        "ordered_sample_ids_sha256": object_sha256(
            [row["sample_id"] for row in rows]
        ),
    }
    if prediction_identity != expected_prediction_identity:
        raise Day20EvaluationV2Error("prediction file/content identity drifted")
    if summary.get("metrics") != aggregate_metrics(rows):
        raise Day20EvaluationV2Error("raw evaluation metrics drifted")
    _validate_runtime_metrics(
        summary.get("runtime_metrics"), expected_runtime_identity=runtime
    )
    return summary


def artifact_paths(
    output_dir: Path, *, candidate: str, eval_scope: str
) -> tuple[Path, Path]:
    directory = output_dir.expanduser().resolve()
    stem = f"{candidate}-{eval_scope}-raw"
    return (
        directory / f"{stem}-predictions-v2.jsonl",
        directory / f"{stem}-summary-v2.json",
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def publish_pair(
    *,
    predictions_attempt: Path,
    predictions_path: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    """Publish a pair with no overwrite; roll back our link on partial failure."""

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
        raise Day20EvaluationV2Error(
            "refusing to overwrite a raw evaluation artifact"
        ) from error
    finally:
        if temporary_summary.exists():
            temporary_summary.unlink()
        if prediction_published and not summary_path.exists():
            predictions_path.unlink(missing_ok=True)


def run_inference(
    *,
    model_path: Path,
    adapter_path: Path | None,
    candidate_identity: candidate_factory.CandidateIdentity,
    eval_scope: str,
    selected_records: Sequence[Mapping[str, Any]],
    scorer: Any,
    provenance: Mapping[str, Any],
    predictions_attempt: Path,
    expected_runtime_identity: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    float,
    float,
    Callable[[Sequence[int]], str],
]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    if os.environ.get("USE_MCORE_GDN") not in (None, "0"):
        raise Day20EvaluationV2Error(
            "USE_MCORE_GDN conflicts with the v2 HF inference contract"
        )
    os.environ["USE_MCORE_GDN"] = "0"
    try:
        import torch
        from swift import get_model_processor, get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    except ImportError as error:
        raise Day20EvaluationV2Error("GPU inference runtime is unavailable") from error
    runtime = live_runtime_identity()
    if runtime != expected_runtime_identity:
        raise Day20EvaluationV2Error(
            "evaluation runtime changed before inference"
        )
    if not torch.cuda.is_available():
        raise Day20EvaluationV2Error("CUDA is required for Day 20 v2 inference")
    torch.cuda.reset_peak_memory_stats()
    model, processor = get_model_processor(
        str(model_path.resolve()),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    if model.__class__.__name__ != EXPECTED_MODEL_CLASS:
        raise Day20EvaluationV2Error(
            f"wrong Qwen3.5 loader: {model.__class__.__name__}"
        )
    template = get_template(
        processor,
        max_length=4096,
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    _validate_qwen35_template(template)

    def generated_token_decoder(token_ids: Sequence[int]) -> str:
        try:
            token_ids_sha256(token_ids)
            decoded = template.decode_generate_ids(
                list(token_ids), first_token=False
            )
        except Exception as error:
            raise Day20EvaluationV2Error(
                "pinned template rejected generated token IDs"
            ) from error
        if not isinstance(decoded, str):
            raise Day20EvaluationV2Error("generated-token decode is not text")
        return decoded

    engine_kwargs: dict[str, Any] = {"template": template, "max_batch_size": 1}
    if adapter_path is not None:
        engine_kwargs["adapters"] = [str(adapter_path.resolve())]
    engine = TransformersEngine(model, **engine_kwargs)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for ordinal, record in enumerate(selected_records, 1):
        maximum = int(record["generation_max_new_tokens"])
        response = engine.infer(
            [InferRequest(messages=record["messages"])],
            RequestConfig(
                max_tokens=maximum,
                temperature=0,
                num_beams=1,
                repetition_penalty=1.0,
                seed=GENERATION_SEED,
                return_details=True,
            ),
            use_tqdm=False,
        )[0]
        if not getattr(response, "choices", None):
            raise Day20EvaluationV2Error(
                f"ms-swift returned no choice: {record['sample_id']}"
            )
        choice = response.choices[0]
        message_content = getattr(getattr(choice, "message", None), "content", None)
        if message_content is None:
            message_content = ""
        if not isinstance(message_content, str):
            raise Day20EvaluationV2Error("ms-swift message content is not text")
        prompt_ids = getattr(response, "prompt_token_ids", None)
        generated_ids = getattr(choice, "token_ids", None)
        if not isinstance(prompt_ids, list) or not isinstance(generated_ids, list):
            raise Day20EvaluationV2Error(
                f"ms-swift token evidence is missing: {record['sample_id']}"
            )
        generated_text = generated_token_decoder(generated_ids)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
        ):
            raise Day20EvaluationV2Error("ms-swift token usage is malformed")
        finish_reason = getattr(choice, "finish_reason", None)
        row = build_prediction_row(
            ordinal=ordinal,
            candidate_identity=candidate_identity,
            eval_scope=eval_scope,
            record=record,
            message_content=message_content,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            generated_only_text=generated_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            scorer=scorer,
            compact_model_identity=provenance["compact_model_identity"],
            compact_checkpoint_package=provenance["compact_checkpoint_package"],
        )
        rows.append(row)
        _append_jsonl(predictions_attempt, row)
        print(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "sample_id": row["sample_id"],
                    "skill": row["skill"],
                    "score": row["scorer_result"].get("score"),
                    "format_compliant": row["format_compliant"],
                    "anomalies": row["anomalies"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.monotonic() - started
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    return rows, runtime, elapsed, peak_memory, generated_token_decoder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Frozen Base model")
    parser.add_argument("--adapter", type=Path, help="Verified v2 LoRA checkpoint")
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--scorers", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--eval-scope", required=True, choices=("probe32", "full112")
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require and fully revalidate an existing raw pair without inference",
    )
    args = parser.parse_args(argv)
    try:
        identity = validate_candidate_scope(args.candidate, args.eval_scope)
    except Day20EvaluationV2Error as error:
        parser.error(str(error))
    if identity.model_role == "base":
        if args.adapter is not None or args.training_summary is not None:
            parser.error("Base candidate cannot use --adapter/--training-summary")
    elif args.adapter is None or args.training_summary is None:
        parser.error("LoRA candidate requires --adapter and --training-summary")
    return args


def _assert_existing_cli_paths(summary: Mapping[str, Any], args: argparse.Namespace) -> None:
    protocol = summary["protocol"]
    expected_paths = {
        "eval_manifest": args.eval_manifest,
        "experiment_manifest": args.experiment_manifest,
        "scorer": args.scorers,
    }
    for component, expected in expected_paths.items():
        if Path(protocol[component]["path"]).resolve() != expected.expanduser().resolve():
            raise Day20EvaluationV2Error(
                f"existing {component} differs from the requested path"
            )
    if Path(summary["model_identity"]["base"]["path"]).resolve() != args.model.expanduser().resolve():
        raise Day20EvaluationV2Error("existing Base model differs from --model")
    if args.adapter is not None and Path(summary["checkpoint"]).resolve() != args.adapter.expanduser().resolve():
        raise Day20EvaluationV2Error("existing checkpoint differs from --adapter")
    if args.training_summary is not None and Path(summary["training_identity"]["path"]).resolve() != args.training_summary.expanduser().resolve():
        raise Day20EvaluationV2Error(
            "existing training identity differs from --training-summary"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_identity = validate_candidate_scope(args.candidate, args.eval_scope)
    predictions_path, summary_path = artifact_paths(
        args.output_dir, candidate=args.candidate, eval_scope=args.eval_scope
    )
    existing = (predictions_path.exists(), summary_path.exists())
    if any(existing):
        if not all(existing):
            raise Day20EvaluationV2Error(
                "existing raw evaluation pair is incomplete; refusing regeneration"
            )
        verified = verify_published_pair(
            predictions_path,
            summary_path,
            expected_scope=args.eval_scope,
            expected_candidate=args.candidate,
        )
        _assert_existing_cli_paths(verified, args)
        print(json.dumps(verified, ensure_ascii=False, sort_keys=True))
        return
    if args.verify_only:
        raise Day20EvaluationV2Error("--verify-only requires an existing raw pair")

    frozen_manifest, scorer = load_frozen_eval_manifest(
        args.eval_manifest, scorer_path=args.scorers
    )
    experiment = verify_experiment_manifest(
        args.experiment_manifest,
        eval_manifest_path=args.eval_manifest,
        frozen_manifest=frozen_manifest,
    )
    selected = select_evaluation_records(
        frozen_manifest=frozen_manifest,
        experiment_manifest=experiment,
        eval_scope=args.eval_scope,
    )
    provenance = resolve_candidate_provenance(
        candidate_identity=candidate_identity,
        model_path=args.model,
        adapter_path=args.adapter,
        training_summary_path=args.training_summary,
        experiment_manifest=experiment,
    )
    run_root = Path(experiment["_verified_run_root"])
    output_dir = _require_within(args.output_dir, run_root / "eval", "eval output")
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_regular_directory(output_dir, "eval output")
    predictions_path, summary_path = artifact_paths(
        output_dir, candidate=args.candidate, eval_scope=args.eval_scope
    )
    predictions_attempt = predictions_path.with_name(
        f".{predictions_path.name}.{os.getpid()}.attempt"
    )
    if predictions_attempt.exists():
        raise Day20EvaluationV2Error(
            f"refusing to overwrite evaluation attempt: {predictions_attempt}"
        )
    pre_runtime_identity = live_runtime_identity()
    protocol = build_protocol(
        eval_scope=args.eval_scope,
        records=len(selected),
        eval_manifest_path=args.eval_manifest,
        frozen_manifest=frozen_manifest,
        experiment_manifest_path=args.experiment_manifest,
        experiment_manifest=experiment,
        scorer_path=args.scorers,
        scorer_version=getattr(scorer, "SCORER_REGISTRY_VERSION"),
        adapter_loaded=candidate_identity.model_role == "lora",
        runtime_identity=pre_runtime_identity,
    )
    (
        rows,
        inference_runtime_identity,
        elapsed,
        peak_memory,
        generated_token_decoder,
    ) = run_inference(
        model_path=args.model,
        adapter_path=args.adapter,
        candidate_identity=candidate_identity,
        eval_scope=args.eval_scope,
        selected_records=selected,
        scorer=scorer,
        provenance=provenance,
        predictions_attempt=predictions_attempt,
        expected_runtime_identity=pre_runtime_identity,
    )
    if inference_runtime_identity != pre_runtime_identity:
        raise Day20EvaluationV2Error("evaluation runtime changed during startup")
    verify_prediction_rows(
        rows,
        selected_records=selected,
        candidate_identity=candidate_identity,
        eval_scope=args.eval_scope,
        scorer=scorer,
        compact_model_identity=provenance["compact_model_identity"],
        compact_checkpoint_package=provenance["compact_checkpoint_package"],
        generated_token_decoder=generated_token_decoder,
    )
    post_frozen_manifest, post_scorer = load_frozen_eval_manifest(
        args.eval_manifest, scorer_path=args.scorers
    )
    if post_frozen_manifest != frozen_manifest:
        raise Day20EvaluationV2Error("frozen eval manifest changed during inference")
    post_experiment = verify_experiment_manifest(
        args.experiment_manifest,
        eval_manifest_path=args.eval_manifest,
        frozen_manifest=post_frozen_manifest,
    )
    if post_experiment != experiment:
        raise Day20EvaluationV2Error(
            "experiment/data/diagnostic provenance changed during inference"
        )
    post_runtime_identity = live_runtime_identity()
    if post_runtime_identity != pre_runtime_identity:
        raise Day20EvaluationV2Error(
            "evaluation runtime/check-out changed during inference"
        )
    post_protocol = build_protocol(
        eval_scope=args.eval_scope,
        records=len(selected),
        eval_manifest_path=args.eval_manifest,
        frozen_manifest=post_frozen_manifest,
        experiment_manifest_path=args.experiment_manifest,
        experiment_manifest=post_experiment,
        scorer_path=args.scorers,
        scorer_version=getattr(post_scorer, "SCORER_REGISTRY_VERSION"),
        adapter_loaded=candidate_identity.model_role == "lora",
        runtime_identity=post_runtime_identity,
    )
    if post_protocol != protocol:
        raise Day20EvaluationV2Error(
            "evaluation implementation/protocol changed during inference"
        )
    post_run_provenance = resolve_candidate_provenance(
        candidate_identity=candidate_identity,
        model_path=args.model,
        adapter_path=args.adapter,
        training_summary_path=args.training_summary,
        experiment_manifest=post_experiment,
    )
    if provenance != post_run_provenance:
        raise Day20EvaluationV2Error(
            "model/checkpoint provenance changed during inference"
        )
    written_rows = load_jsonl(predictions_attempt)
    if written_rows != rows:
        raise Day20EvaluationV2Error("prediction attempt differs from in-memory rows")
    predictions_attempt_sha256 = file_sha256(predictions_attempt)
    summary = build_summary(
        candidate_identity=candidate_identity,
        eval_scope=args.eval_scope,
        provenance=provenance,
        protocol=protocol,
        rows=rows,
        predictions_path=predictions_path,
        predictions_file_sha256=predictions_attempt_sha256,
        elapsed_seconds=elapsed,
        peak_cuda_memory_gib=peak_memory,
        runtime_identity=pre_runtime_identity,
    )
    publish_pair(
        predictions_attempt=predictions_attempt,
        predictions_path=predictions_path,
        summary_path=summary_path,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
