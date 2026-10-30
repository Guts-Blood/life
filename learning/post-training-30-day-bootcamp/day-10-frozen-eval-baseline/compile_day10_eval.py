#!/usr/bin/env python3
"""Compile Day 09 eval candidates into the immutable Day 10 eval manifest.

This module deliberately stops before model generation. It freezes sample
identity, dev/frozen assignment, prompt adaptation, chat rendering, token IDs,
scorer identity, and semantic hashes. Generated HumanEval code is never run by
this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import rfc8785


HERE = Path(__file__).resolve().parent
BOOTCAMP = HERE.parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "day10_eval_config.json"
REQUIRED_CANDIDATE_FIELDS = {
    "eval_sample_id",
    "source",
    "revision",
    "split",
    "parent_id",
    "prompt",
    "reference",
    "skill",
    "content_hash",
    "selection_hash",
    "adapter",
    "transform_chain",
    "license",
    "source_file",
    "source_file_sha256",
}


class Day10CompileError(ValueError):
    """A frozen input or manifest invariant failed."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Day10CompileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Day10CompileError(f"non-finite JSON value: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise Day10CompileError(f"invalid JSON in {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise Day10CompileError(
                f"invalid JSONL row {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise Day10CompileError(f"JSONL row {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise Day10CompileError(f"path escapes repository root: {value}") from error
    return path


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day10CompileError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_text_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError) as error:
        raise Day10CompileError(f"RFC 8785 canonicalization failed: {error}") from error


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def day09_object_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_file_hash(spec: dict[str, Any]) -> Path:
    path = repo_path(spec["path"])
    actual = file_sha256(path)
    if actual != spec["file_sha256"]:
        raise Day10CompileError(
            f"file hash mismatch for {spec['path']}: expected "
            f"{spec['file_sha256']}, got {actual}"
        )
    return path


def verify_environment(config: dict[str, Any]) -> dict[str, str]:
    environment = config["environment"]
    verified: dict[str, str] = {}
    for label, path_key, hash_key in (
        ("direct_contract", "direct_contract_path", "direct_contract_sha256"),
        ("resolved_snapshot", "resolved_snapshot_path", "resolved_snapshot_sha256"),
    ):
        path = repo_path(environment[path_key])
        actual = file_sha256(path)
        expected = environment[hash_key]
        if actual != expected:
            raise Day10CompileError(
                f"environment {label} hash mismatch: expected {expected}, got {actual}"
            )
        verified[label] = actual
    return verified


def load_scorer_registry(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    scoring = config["scoring"]
    scorer_path = repo_path(scoring["source_path"])
    scorer_source_sha256 = file_sha256(scorer_path)
    spec = importlib.util.spec_from_file_location("day10_frozen_scorers", scorer_path)
    if spec is None or spec.loader is None:
        raise Day10CompileError(f"cannot load scorer module: {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day10CompileError(f"scorer module import failed: {error}") from error

    registry = getattr(module, "SCORER_REGISTRY", None)
    if not isinstance(registry, dict):
        raise Day10CompileError("scorer module has no JSON-object SCORER_REGISTRY")
    if registry.get("registry_version") != scoring["registry_version"]:
        raise Day10CompileError("scorer registry version differs from config")
    registry_slices = registry.get("slices")
    expected_slices = set(config["split_assignment"]["slices"])
    if not isinstance(registry_slices, dict) or set(registry_slices) != expected_slices:
        raise Day10CompileError("scorer registry slice set differs from config")
    for slice_name in sorted(expected_slices):
        slice_registry = registry_slices[slice_name]
        if slice_registry.get("extractor_version") != scoring[
            "extractor_by_slice"
        ][slice_name]:
            raise Day10CompileError(
                f"extractor version mismatch for scorer slice {slice_name}"
            )
        if slice_registry.get("scorer_version") != scoring["scorer_by_slice"][
            slice_name
        ]:
            raise Day10CompileError(
                f"scorer version mismatch for scorer slice {slice_name}"
            )
    if registry_slices["code"].get("execution_policy") != "sandbox_required":
        raise Day10CompileError("code scorer must require sandbox execution")
    canonical_bytes(registry)
    return registry, scorer_source_sha256


def verify_manifest_identity(spec: dict[str, Any], payload: dict[str, Any]) -> None:
    header = payload.get("header")
    if not isinstance(header, dict):
        raise Day10CompileError(f"manifest has no header: {spec['path']}")
    if header.get("manifest_hash") != spec["manifest_hash"]:
        raise Day10CompileError(f"semantic manifest hash mismatch: {spec['path']}")


def verify_upstream(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream"]

    assignment_path = require_file_hash(upstream["day09_assignment"])
    assignment = load_json(assignment_path)
    if assignment.get("status") != upstream["day09_assignment"]["required_status"]:
        raise Day10CompileError("Day 09 assignment is not in the required terminal state")
    gate_c = assignment.get("gate_c", {})
    if gate_c.get("status") != upstream["gate_c"]["status"]:
        raise Day10CompileError("Day 09 Gate C disposition changed")
    reviewed = gate_c.get("reviewed_occurrences_mix_A", 0) + gate_c.get(
        "reviewed_occurrences_mix_B", 0
    )
    if reviewed != upstream["gate_c"]["reviewed_occurrences"]:
        raise Day10CompileError("Day 09 Gate C review count changed")

    clean_path = require_file_hash(upstream["clean_parent_manifest"])
    clean_manifest = load_json(clean_path)
    verify_manifest_identity(upstream["clean_parent_manifest"], clean_manifest)
    if clean_manifest["header"].get("clean_pool_hash") != upstream[
        "clean_parent_manifest"
    ]["clean_pool_hash"]:
        raise Day10CompileError("Day 09 clean pool hash changed")

    train_manifest_hashes: list[str] = []
    for spec in upstream["train_manifests"]:
        path = require_file_hash(spec)
        payload = load_json(path)
        verify_manifest_identity(spec, payload)
        train_manifest_hashes.append(spec["manifest_hash"])

    summary_path = require_file_hash(upstream["eval_candidate_summary"])
    candidate_summary = load_json(summary_path)
    if candidate_summary.get("status") != upstream["eval_candidate_summary"][
        "required_status"
    ]:
        raise Day10CompileError("Day 09 eval candidate summary status changed")
    if candidate_summary.get("total_candidates") != upstream[
        "eval_candidate_summary"
    ]["required_total"]:
        raise Day10CompileError("Day 09 eval candidate total changed")

    decontamination_path = require_file_hash(upstream["decontamination_summary"])
    decontamination = load_json(decontamination_path)
    if decontamination.get("candidate_version") != upstream[
        "decontamination_summary"
    ]["candidate_version"]:
        raise Day10CompileError("Day 09 decontamination version changed")
    candidate_records = decontamination.get("candidate_statistics", {}).get(
        "candidate_records"
    )
    if candidate_records != upstream["decontamination_summary"][
        "required_candidate_records"
    ]:
        raise Day10CompileError("Day 09 has unresolved train/eval overlap candidates")
    gate_b_decision = assignment.get("decontamination", {}).get(
        "gate_b_decision", {}
    )
    if gate_b_decision.get("candidate_count") != 0 or gate_b_decision.get(
        "confirmed_contamination"
    ) != 0:
        raise Day10CompileError("Day 09 final Gate B decision is not clean")

    report_path = require_file_hash(upstream["decontamination_report"])

    return {
        "assignment": assignment,
        "candidate_summary": candidate_summary,
        "decontamination": decontamination,
        "train_manifest_hashes": train_manifest_hashes,
        "decontamination_report_sha256": file_sha256(report_path),
    }


def validate_candidate(row: dict[str, Any], expected_slice: str) -> None:
    missing = sorted(REQUIRED_CANDIDATE_FIELDS - row.keys())
    if missing:
        raise Day10CompileError(
            f"candidate {row.get('eval_sample_id')} missing fields: {missing}"
        )
    if row["skill"] != expected_slice:
        raise Day10CompileError(
            f"candidate {row['eval_sample_id']} is {row['skill']}, expected {expected_slice}"
        )
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise Day10CompileError(f"empty prompt: {row['eval_sample_id']}")
    if not isinstance(row["reference"], str) or not row["reference"].strip():
        raise Day10CompileError(f"empty reference: {row['eval_sample_id']}")
    content_hash = day09_object_hash(
        {"prompt": row["prompt"], "reference": row["reference"]}
    )
    if content_hash != row["content_hash"]:
        raise Day10CompileError(f"candidate content hash mismatch: {row['eval_sample_id']}")


def load_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected_count = config["split_assignment"]["candidates_per_slice"]
    rows: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}
    for slice_name in config["split_assignment"]["slices"]:
        spec = config["upstream"]["candidate_files"][slice_name]
        path = require_file_hash(spec)
        slice_rows = load_jsonl(path)
        if len(slice_rows) != expected_count:
            raise Day10CompileError(
                f"{slice_name} has {len(slice_rows)} candidates, expected {expected_count}"
            )
        for row in slice_rows:
            validate_candidate(row, slice_name)
            source_files.setdefault(row["source_file"], row["source_file_sha256"])
        rows.extend(slice_rows)

    ids = [row["eval_sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise Day10CompileError("eval candidate IDs are not globally unique")

    for relative_path, expected_hash in source_files.items():
        source_path = repo_path(relative_path)
        actual_hash = file_sha256(source_path)
        if actual_hash != expected_hash:
            raise Day10CompileError(
                f"candidate source hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return rows


def split_stratum(row: dict[str, Any]) -> str:
    if row["skill"] == "general":
        subject = row.get("metadata", {}).get("subject")
        if not isinstance(subject, str) or not subject:
            raise Day10CompileError(
                f"general candidate has no MMLU subject: {row['eval_sample_id']}"
            )
        return f"general:{subject}"
    return row["skill"]


def assignment_rank(version: str, stratum: str, sample_id: str) -> str:
    payload = "\0".join((version, stratum, sample_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_evaluation_splits(
    rows: list[dict[str, Any]], split_config: dict[str, Any]
) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[split_stratum(row)].append(row)

    general_groups = sorted(key for key in groups if key.startswith("general:"))
    if len(general_groups) != 4 or any(len(groups[key]) != 10 for key in general_groups):
        raise Day10CompileError("general slice is not four MMLU subjects with ten samples each")
    for slice_name in ("math", "code", "finance"):
        if len(groups.get(slice_name, [])) != split_config["candidates_per_slice"]:
            raise Day10CompileError(f"unexpected candidate count for {slice_name}")

    assignments: dict[str, str] = {}
    version = split_config["version"]
    for stratum, stratum_rows in groups.items():
        frozen_count = 3 if stratum.startswith("general:") else split_config[
            "frozen_test_per_slice"
        ]
        ranked = sorted(
            stratum_rows,
            key=lambda row: (
                assignment_rank(version, stratum, row["eval_sample_id"]),
                row["eval_sample_id"],
            ),
        )
        for index, row in enumerate(ranked):
            assignments[row["eval_sample_id"]] = (
                "frozen_test" if index < frozen_count else "dev"
            )

    counts = Counter(
        (row["skill"], assignments[row["eval_sample_id"]]) for row in rows
    )
    for slice_name in split_config["slices"]:
        if counts[(slice_name, "dev")] != split_config["dev_per_slice"]:
            raise Day10CompileError(f"dev count mismatch for {slice_name}")
        if counts[(slice_name, "frozen_test")] != split_config[
            "frozen_test_per_slice"
        ]:
            raise Day10CompileError(f"frozen_test count mismatch for {slice_name}")
    return assignments


def adapt_prompt(slice_name: str, raw_prompt: str) -> str:
    raw_prompt = raw_prompt.strip()
    if slice_name == "general":
        return (
            raw_prompt
            + "\n\nChoose the correct answer. End with exactly: Final answer: <A|B|C|D>"
        )
    if slice_name == "math":
        return raw_prompt + "\n\nSolve the problem. End with exactly: Final answer: <answer>"
    if slice_name == "finance":
        return (
            raw_prompt
            + "\n\nAnswer from the table and paragraphs. End with exactly: "
            "Final answer: <answer>, including any required scale."
        )
    if slice_name == "code":
        return (
            "Complete the Python function below. Return only the code that should "
            "continue after the prompt. Do not repeat the prompt and do not use "
            "Markdown fences.\n\n"
            + raw_prompt
        )
    raise Day10CompileError(f"unknown slice: {slice_name}")


def load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    render_config = config["model_and_rendering"]
    snapshot_path = repo_path(render_config["tokenizer_snapshot_path"])
    if not snapshot_path.is_dir():
        raise Day10CompileError(f"tokenizer snapshot is missing: {snapshot_path}")
    for filename, expected_hash in render_config["tokenizer_files"].items():
        actual_hash = file_sha256(snapshot_path / filename)
        if actual_hash != expected_hash:
            raise Day10CompileError(f"tokenizer file hash mismatch: {filename}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    if type(tokenizer).__name__ != render_config["tokenizer_class"]:
        raise Day10CompileError(
            f"tokenizer class mismatch: {type(tokenizer).__name__}"
        )
    template = tokenizer.chat_template or ""
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != render_config[
        "chat_template_sha256"
    ]:
        raise Day10CompileError("chat template hash mismatch")
    if tokenizer.eos_token_id != config["generation"]["eos_token_id"]:
        raise Day10CompileError("EOS token ID changed")
    if tokenizer.pad_token_id != config["generation"]["pad_token_id"]:
        raise Day10CompileError("padding token ID changed")
    return tokenizer


def verify_model_snapshot(config: dict[str, Any]) -> Path:
    render_config = config["model_and_rendering"]
    snapshot_path = repo_path(render_config["model_snapshot_path"])
    if not snapshot_path.is_dir():
        raise Day10CompileError(f"model snapshot is missing: {snapshot_path}")
    model_files = render_config.get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise Day10CompileError("model_files must freeze at least one model file")
    for relative_path, expected_hash in model_files.items():
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise Day10CompileError(f"unsafe model file path: {relative_path}")
        actual_hash = file_sha256(snapshot_path / relative_path)
        if actual_hash != expected_hash:
            raise Day10CompileError(
                f"model file hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return snapshot_path


def render_candidate(
    tokenizer: Any, row: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    slice_name = row["skill"]
    adapted_prompt = adapt_prompt(slice_name, row["prompt"])
    messages = [{"role": "user", "content": adapted_prompt}]
    flags = config["model_and_rendering"]["render_flags"]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=flags["add_generation_prompt"],
        enable_thinking=flags["enable_thinking"],
    )
    input_ids = tokenizer(
        rendered,
        add_special_tokens=flags["add_special_tokens_after_render"],
        return_attention_mask=False,
    )["input_ids"]
    if not isinstance(input_ids, list) or not all(isinstance(item, int) for item in input_ids):
        raise Day10CompileError(f"invalid input IDs for {row['eval_sample_id']}")
    maximum = config["model_and_rendering"]["maximum_input_tokens"]
    if len(input_ids) > maximum:
        raise Day10CompileError(
            f"{row['eval_sample_id']} has {len(input_ids)} input tokens, limit {maximum}"
        )
    return {
        "adapted_prompt": adapted_prompt,
        "messages": messages,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
    }


def checkpoint_independent_rendering_contract(
    model_and_rendering: dict[str, Any],
) -> dict[str, Any]:
    """Return rendering identity without the checkpoint or local cache path."""

    required_keys = (
        "tokenizer_files",
        "tokenizer_class",
        "chat_template_sha256",
        "render_flags",
        "maximum_input_tokens",
        "prompt_adapters",
    )
    missing = [key for key in required_keys if key not in model_and_rendering]
    if missing:
        raise Day10CompileError(f"rendering contract missing keys: {missing}")
    return {key: model_and_rendering[key] for key in required_keys}


def evaluation_order(
    sample_ids: list[str], evaluation_split: str, version: str
) -> list[str]:
    def rank(sample_id: str) -> tuple[str, str]:
        payload = "\0".join((version, evaluation_split, sample_id)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), sample_id

    return sorted(sample_ids, key=rank)


def build_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    upstream_evidence = verify_upstream(config)
    verify_environment(config)
    rows = load_candidates(config)
    assignments = assign_evaluation_splits(rows, config["split_assignment"])
    verify_model_snapshot(config)
    tokenizer = load_tokenizer(config)

    scorer_registry, scorer_source_sha256 = load_scorer_registry(config)
    compiler_source_sha256 = file_sha256(Path(__file__).resolve())
    config_source_sha256 = file_sha256(config_path)

    overlap_check = {
        "status": "no_unhandled_matches_under_frozen_day09_matcher",
        "candidate_version": config["upstream"]["decontamination_summary"][
            "candidate_version"
        ],
        "checked_clean_pool_hash": config["upstream"]["clean_parent_manifest"][
            "clean_pool_hash"
        ],
        "derived_train_manifest_hashes": upstream_evidence["train_manifest_hashes"],
        "decontamination_summary_file_sha256": config["upstream"][
            "decontamination_summary"
        ]["file_sha256"],
        "decontamination_report_file_sha256": upstream_evidence[
            "decontamination_report_sha256"
        ],
        "scope_limit": "Does not establish absence from model pretraining or unscanned benchmark derivatives.",
    }

    records: list[dict[str, Any]] = []
    for row in rows:
        rendered = render_candidate(tokenizer, row, config)
        slice_name = row["skill"]
        record = {
            "sample_id": row["eval_sample_id"],
            "evaluation_split": assignments[row["eval_sample_id"]],
            "slice": slice_name,
            "source_lineage": {
                "source": row["source"],
                "revision": row["revision"],
                "source_split": row["split"],
                "parent_id": row["parent_id"],
                "adapter": row["adapter"],
                "transform_chain": row["transform_chain"],
                "license": row["license"],
                "source_file": row["source_file"],
                "source_file_sha256": row["source_file_sha256"],
                "candidate_content_hash": row["content_hash"],
                "candidate_selection_hash": row["selection_hash"],
            },
            "raw_prompt": row["prompt"],
            "raw_prompt_hash": exact_text_hash(row["prompt"]),
            "adapted_prompt": rendered["adapted_prompt"],
            "adapted_prompt_hash": exact_text_hash(rendered["adapted_prompt"]),
            "messages": rendered["messages"],
            "messages_hash": semantic_hash(
                {
                    "domain": "day10.messages",
                    "schema_version": 1,
                    "messages": rendered["messages"],
                }
            ),
            "rendered_prompt": rendered["rendered_prompt"],
            "rendered_prompt_hash": exact_text_hash(rendered["rendered_prompt"]),
            "input_ids": rendered["input_ids"],
            "input_ids_hash": semantic_hash(
                {
                    "domain": "day10.input_ids",
                    "schema_version": 1,
                    "input_ids": rendered["input_ids"],
                }
            ),
            "input_token_count": len(rendered["input_ids"]),
            "reference": row["reference"],
            "reference_hash": exact_text_hash(row["reference"]),
            "prompt_adapter_version": config["model_and_rendering"][
                "prompt_adapters"
            ][slice_name],
            "extractor_version": config["scoring"]["extractor_by_slice"][slice_name],
            "scorer_version": config["scoring"]["scorer_by_slice"][slice_name],
            "generation_max_new_tokens": config["generation"][
                "max_new_tokens_by_slice"
            ][slice_name],
            "overlap_check": overlap_check,
            "metadata": row.get("metadata", {}),
        }
        records.append(record)

    records.sort(key=lambda item: item["sample_id"])
    task_identity = [
        {
            "sample_id": record["sample_id"],
            "evaluation_split": record["evaluation_split"],
            "slice": record["slice"],
            "source_lineage": record["source_lineage"],
            "raw_prompt_hash": record["raw_prompt_hash"],
            "reference_hash": record["reference_hash"],
        }
        for record in records
    ]
    dataset_context_hash = semantic_hash(
        {
            "domain": "day10.eval_dataset_context",
            "schema_version": 1,
            "records": task_identity,
            "overlap_check": overlap_check,
        }
    )
    rendered_inputs_hash = semantic_hash(
        {
            "domain": "day10.rendered_inputs",
            "schema_version": 1,
            "records": [
                {
                    "sample_id": record["sample_id"],
                    "prompt_adapter_version": record["prompt_adapter_version"],
                    "messages_hash": record["messages_hash"],
                    "rendered_prompt_hash": record["rendered_prompt_hash"],
                    "input_ids_hash": record["input_ids_hash"],
                }
                for record in records
            ],
        }
    )
    scorer_registry_hash = semantic_hash(
        {
            "domain": "day10.scorer_registry",
            "schema_version": 1,
            "registry": scorer_registry,
            "source_sha256": scorer_source_sha256,
        }
    )
    aggregation_hash = semantic_hash(
        {
            "domain": "day10.aggregation_uncertainty",
            "schema_version": 1,
            **config["aggregation"],
        }
    )
    reference_environment_hash = semantic_hash(
        {
            "domain": "day10.references_and_tool_environment",
            "schema_version": 1,
            "references": [
                [record["sample_id"], record["reference_hash"]] for record in records
            ],
            "code_execution_policy": config["scoring"]["code_execution_policy"],
        }
    )
    eval_suite_hash = semantic_hash(
        {
            "domain": "day10.eval_suite",
            "schema_version": 1,
            "raw_task_manifest_hash": dataset_context_hash,
            "references_and_tool_environment_hash": reference_environment_hash,
            "scorer_registry_hash": scorer_registry_hash,
            "aggregation_uncertainty_hash": aggregation_hash,
        }
    )
    rendering_contract = checkpoint_independent_rendering_contract(
        config["model_and_rendering"]
    )
    protocol_hash = semantic_hash(
        {
            "domain": "day10.measurement_protocol",
            "schema_version": 1,
            "dataset_context_hash": dataset_context_hash,
            "rendered_inputs_hash": rendered_inputs_hash,
            "rendering_contract": rendering_contract,
            "generation": config["generation"],
            "scorer_registry_hash": scorer_registry_hash,
            "aggregation_uncertainty_hash": aggregation_hash,
        }
    )

    counts_by_split = Counter(record["evaluation_split"] for record in records)
    counts_by_slice_split = Counter(
        (record["slice"], record["evaluation_split"]) for record in records
    )
    input_lengths = [record["input_token_count"] for record in records]
    ids_by_split = {
        split: [record["sample_id"] for record in records if record["evaluation_split"] == split]
        for split in ("dev", "frozen_test")
    }
    header = {
        "domain": "day10.frozen_eval_manifest",
        "schema_version": 1,
        "manifest_name": config["identity"]["manifest_name"],
        "manifest_version": config["identity"]["manifest_version"],
        "status": "frozen_manifest_pre_baseline",
        "frozen_date": config["identity"]["frozen_date"],
        "intended_use": config["identity"]["intended_use"],
        "claim_boundary": config["identity"]["claim_boundary"],
        "config_file_sha256": config_source_sha256,
        "compiler_source_sha256": compiler_source_sha256,
        "scorer_source_sha256": scorer_source_sha256,
        "environment_contract_sha256": config["environment"][
            "direct_contract_sha256"
        ],
        "environment_snapshot_sha256": config["environment"][
            "resolved_snapshot_sha256"
        ],
        "hash_contract": "sha256 over RFC 8785 JCS bytes; exact text/file hashes use raw UTF-8/file bytes",
        "dataset_context_hash": dataset_context_hash,
        "rendered_inputs_hash": rendered_inputs_hash,
        "references_and_tool_environment_hash": reference_environment_hash,
        "scorer_registry_hash": scorer_registry_hash,
        "aggregation_uncertainty_hash": aggregation_hash,
        "eval_suite_hash": eval_suite_hash,
        "protocol_hash": protocol_hash,
        "model_and_rendering": config["model_and_rendering"],
        "rendering_contract": rendering_contract,
        "generation": config["generation"],
        "aggregation": config["aggregation"],
        "overlap_check": overlap_check,
        "gate_c_claim_boundary": assignment_gate_c_boundary(
            upstream_evidence["assignment"]
        ),
        "counts": {
            "total": len(records),
            "by_split": dict(sorted(counts_by_split.items())),
            "by_slice_and_split": {
                slice_name: {
                    split: counts_by_slice_split[(slice_name, split)]
                    for split in ("dev", "frozen_test")
                }
                for slice_name in config["split_assignment"]["slices"]
            },
        },
        "input_token_lengths": {
            "minimum": min(input_lengths),
            "maximum": max(input_lengths),
            "total": sum(input_lengths),
        },
        "evaluation_order": {
            split: evaluation_order(
                ids_by_split[split],
                split,
                config["split_assignment"]["evaluation_order_version"],
            )
            for split in ("dev", "frozen_test")
        },
        "frozen_test_access_policy": (
            "Do not generate, inspect, score, or audit frozen_test outputs before "
            "Day 12 selects one checkpoint within each A/B run."
        ),
        "manifest_hash_definition": (
            "sha256 over RFC 8785 canonical bytes of this object with "
            "header.manifest_hash omitted and records sorted by sample_id"
        ),
    }
    manifest = {"header": header, "records": records}
    manifest["header"]["manifest_hash"] = semantic_hash(manifest)
    return manifest


def assignment_gate_c_boundary(assignment: dict[str, Any]) -> str:
    boundary = assignment.get("gate_c", {}).get("claim_boundary")
    if not isinstance(boundary, str) or not boundary:
        raise Day10CompileError("Day 09 Gate C claim boundary is missing")
    return boundary


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def verify_manifest_hash(manifest: dict[str, Any]) -> None:
    expected = manifest.get("header", {}).get("manifest_hash")
    if not isinstance(expected, str):
        raise Day10CompileError("manifest_hash is missing")
    candidate = json.loads(json.dumps(manifest))
    candidate["header"].pop("manifest_hash", None)
    actual = semantic_hash(candidate)
    if actual != expected:
        raise Day10CompileError(
            f"manifest semantic hash mismatch: expected {expected}, got {actual}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-upstream", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    if args.check_upstream:
        evidence = verify_upstream(config)
        verify_environment(config)
        candidates = load_candidates(config)
        verify_model_snapshot(config)
        load_tokenizer(config)
        scorer_registry, _ = load_scorer_registry(config)
        print(
            "upstream=ok "
            f"candidates={len(candidates)} "
            f"train_manifests={len(evidence['train_manifest_hashes'])} "
            f"scorer_registry={scorer_registry['registry_version']}"
        )
        return

    manifest = build_manifest(args.config)
    verify_manifest_hash(manifest)
    output = args.output or repo_path(config["outputs"]["manifest_path"])
    payload = manifest_bytes(manifest)

    if args.verify_only:
        if not output.is_file():
            raise Day10CompileError(f"manifest does not exist: {output}")
        if output.read_bytes() != payload:
            raise Day10CompileError("on-disk manifest differs from deterministic rebuild")
        print(
            f"manifest=verified records={len(manifest['records'])} "
            f"hash={manifest['header']['manifest_hash']}"
        )
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        f"manifest=written path={output} records={len(manifest['records'])} "
        f"hash={manifest['header']['manifest_hash']} "
        f"file_sha256={hashlib.sha256(payload).hexdigest()}"
    )


if __name__ == "__main__":
    main()
