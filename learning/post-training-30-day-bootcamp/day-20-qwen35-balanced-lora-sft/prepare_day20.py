#!/usr/bin/env python3
"""Prepare the frozen Day 20 balanced Qwen3.5 LoRA datasets and configs.

Core input is four tokenizer-audited normalized JSONL files, supplied as
``--source skill=path``.  A caller may instead explicitly opt into a separate
fetch adapter; this script deliberately hard-codes no dataset IDs or revisions.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from day20_contract import (
    DIAGNOSTIC_RECORDS_PER_SKILL,
    MAIN_SUPERVISED_TOKENS,
    MAIN_TOKENS_BY_FORMAT,
    MAIN_TOKENS_PER_SKILL,
    MIN_MAIN_RECORDS_BY_FORMAT,
    MIN_PROBE_RECORDS_BY_FORMAT,
    PROBE_LEARNING_RATES,
    PROBE_SUPERVISED_TOKENS,
    PROBE_TOKENS_BY_FORMAT,
    PROBE_TOKENS_PER_SKILL,
    QWEN35_TEMPLATE_CONTRACT,
    SKILLS,
    TARGET_FORMATS,
    Day20ContractError,
    assert_no_train_dev_leakage,
    build_main_config,
    build_probe_config,
    exact_subset_indices,
    immutable_experiment_contract,
    normalize_and_validate_record,
    object_sha256,
    stable_hash_subset,
    text_sha256,
    validate_unique_training_records,
)


class Day20PreparationError(Day20ContractError):
    """A normalized source or Day 20 output invariant failed."""


PINNED_SOURCE_REVISIONS = {
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "tulu": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "tatqa": "c96247f5077eac447f63527fd3dcfdc58bb56d6a",
    "mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "tulu_code": "1412abe88dd2976af977260788e033013449f7b2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day20PreparationError(f"required file is missing: {path}")
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_base_model_identity(model_snapshot: str | Path) -> dict[str, Any]:
    model_path = Path(model_snapshot).expanduser().resolve()
    if not model_path.is_dir():
        raise Day20PreparationError(f"Base model directory is missing: {model_path}")
    files = {
        path.relative_to(model_path).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(model_path.rglob("*"))
        if path.is_file()
    }
    if "config.json" not in files or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise Day20PreparationError("Base model file manifest is incomplete")
    return {
        "path": str(model_path),
        "files": files,
        "snapshot_sha256": object_sha256(files),
    }


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise Day20PreparationError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Day20PreparationError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def parse_source_specs(specs: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for spec in specs:
        skill, separator, raw_path = spec.partition("=")
        if not separator or skill not in SKILLS or not raw_path:
            raise Day20PreparationError(
                "each --source must be one of general=PATH, math=PATH, finance=PATH, code=PATH"
            )
        if skill in sources:
            raise Day20PreparationError(f"duplicate source for skill {skill!r}")
        sources[skill] = Path(raw_path).expanduser().resolve()
    if set(sources) != set(SKILLS):
        missing = sorted(set(SKILLS) - set(sources))
        raise Day20PreparationError(f"missing normalized sources: {missing}")
    return sources


def _load_fetch_adapter(spec: str) -> Callable[..., Mapping[str, str | Path]]:
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise Day20PreparationError("--fetch-adapter must be MODULE:FUNCTION")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise Day20PreparationError(f"fetch adapter is not callable: {spec}")
    return function


def resolve_sources(
    *,
    source_specs: Sequence[str],
    fetch_adapter: str | None,
    fetch_config_path: Path | None,
    run_root: Path,
) -> dict[str, Path]:
    if source_specs and fetch_adapter:
        raise Day20PreparationError("--source and --fetch-adapter are mutually exclusive")
    if source_specs:
        if fetch_config_path is not None:
            raise Day20PreparationError("--fetch-config requires --fetch-adapter")
        return parse_source_specs(source_specs)
    if not fetch_adapter:
        raise Day20PreparationError("provide four --source arguments or --fetch-adapter")

    config: dict[str, Any] = {}
    if fetch_config_path is not None:
        try:
            loaded = json.loads(fetch_config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise Day20PreparationError(f"cannot load fetch config: {error}") from error
        if not isinstance(loaded, dict):
            raise Day20PreparationError("fetch config root must be an object")
        config = loaded
    adapter = _load_fetch_adapter(fetch_adapter)
    result = adapter(config=config, output_dir=run_root / "adapter-sources")
    if not isinstance(result, Mapping):
        raise Day20PreparationError("fetch adapter must return a skill-to-path mapping")
    return parse_source_specs([f"{skill}={path}" for skill, path in result.items()])


def load_normalized_jsonl(
    path: Path,
    *,
    skill: str,
    max_length: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Day20PreparationError(f"normalized source is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise Day20PreparationError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(raw, dict):
                raise Day20PreparationError(f"JSONL row is not an object: {path}:{line_number}")
            try:
                records.append(
                    normalize_and_validate_record(
                        raw, expected_skill=skill, max_length=max_length
                    )
                )
            except Day20ContractError as error:
                raise Day20PreparationError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise Day20PreparationError(f"normalized source has no records: {path}")
    return records


def verify_source_token_audit(
    audit: Mapping[str, Any],
    *,
    model_snapshot: str,
    sources: Mapping[str, Path],
    records_by_skill: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Bind the real Qwen3.5 re-encoding report to the exact source bytes."""
    payload = dict(audit)
    expected_hash = payload.pop("audit_sha256", None)
    tokenizer_files = audit.get("tokenizer_files")
    audited_sources = audit.get("sources")
    expected_records = sum(len(records_by_skill[skill]) for skill in SKILLS)
    expected_tokens = sum(
        int(row["qwen35_supervised_tokens"])
        for skill in SKILLS
        for row in records_by_skill[skill]
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("domain") != "day20.qwen35_source_token_reaudit"
        or audit.get("status") != "pass"
        or expected_hash != object_sha256(payload)
        or audit.get("model_path") != str(Path(model_snapshot).expanduser().resolve())
        or audit.get("template") != QWEN35_TEMPLATE_CONTRACT
        or audit.get("records") != expected_records
        or audit.get("supervised_tokens") != expected_tokens
        or not isinstance(audit.get("ordered_evidence_sha256"), str)
        or len(audit["ordered_evidence_sha256"]) != 64
        or not isinstance(tokenizer_files, Mapping)
        or "config.json" not in tokenizer_files
        or not any(name in tokenizer_files for name in ("tokenizer.json", "vocab.json"))
        or not isinstance(audited_sources, Mapping)
        or set(audited_sources) != set(SKILLS)
    ):
        raise Day20PreparationError("Qwen3.5 source token re-audit identity drifted")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in tokenizer_files.values()
    ):
        raise Day20PreparationError("tokenizer file identity is malformed")
    for skill in SKILLS:
        identity = audited_sources[skill]
        path = Path(sources[skill]).resolve()
        rows = records_by_skill[skill]
        if (
            not isinstance(identity, Mapping)
            or identity.get("path") != str(path)
            or identity.get("file_sha256") != file_sha256(path)
            or identity.get("records") != len(rows)
            or identity.get("supervised_tokens")
            != sum(int(row["qwen35_supervised_tokens"]) for row in rows)
        ):
            raise Day20PreparationError(
                f"Qwen3.5 token re-audit source identity drifted: {skill}"
            )
    return dict(audit)


def verify_source_adapter_manifest(
    path: Path,
    *,
    model_snapshot: str,
    sources: Mapping[str, Path],
    records_by_skill: Mapping[str, Sequence[Mapping[str, Any]]],
    source_token_audit: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day20PreparationError(
            f"cannot load pinned source-adapter manifest: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise Day20PreparationError("source-adapter manifest root must be an object")
    payload = dict(manifest)
    content_sha = payload.pop("manifest_sha256", None)
    implementation = manifest.get("implementation")
    fixed_sources = manifest.get("fixed_sources")
    outputs = manifest.get("outputs")
    expected_adapter_sha = file_sha256(Path(__file__).with_name("day20_source_adapter.py"))
    expected_contract_sha = file_sha256(Path(__file__).with_name("day20_contract.py"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("domain") != "day20.qwen35_pinned_source_adapter_manifest"
        or manifest.get("status") != "pass"
        or manifest.get("dataset_code_execution") is not False
        or content_sha != object_sha256(payload)
        or manifest.get("model_path")
        != str(Path(model_snapshot).expanduser().resolve())
        or manifest.get("tokenizer_files") != source_token_audit.get("tokenizer_files")
        or not isinstance(implementation, Mapping)
        or implementation.get("source_adapter_file_sha256") != expected_adapter_sha
        or implementation.get("contract_file_sha256") != expected_contract_sha
        or not isinstance(implementation.get("ms_swift_version"), str)
        or not implementation["ms_swift_version"]
        or not isinstance(fixed_sources, Mapping)
        or set(fixed_sources) != set(PINNED_SOURCE_REVISIONS)
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(SKILLS)
    ):
        raise Day20PreparationError("pinned source-adapter manifest identity drifted")
    for key, revision in PINNED_SOURCE_REVISIONS.items():
        item = fixed_sources[key]
        if not isinstance(item, Mapping) or item.get("revision") != revision:
            raise Day20PreparationError(f"pinned source revision drifted: {key}")
    for skill in SKILLS:
        identity = outputs[skill]
        source = Path(sources[skill]).resolve()
        rows = records_by_skill[skill]
        if (
            not isinstance(identity, Mapping)
            or Path(str(identity.get("path", ""))).resolve() != source
            or identity.get("file_sha256") != file_sha256(source)
            or identity.get("records") != len(rows)
            or identity.get("supervised_tokens")
            != sum(int(row["qwen35_supervised_tokens"]) for row in rows)
        ):
            raise Day20PreparationError(
                f"source-adapter output identity drifted: {skill}"
            )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "content_sha256": content_sha,
        "implementation": dict(implementation),
        "fixed_revisions": dict(PINNED_SOURCE_REVISIONS),
    }


def load_frozen_dev_records(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day20PreparationError(f"cannot load eval manifest {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise Day20PreparationError("eval manifest must contain a records array")
    dev = [
        record
        for record in value["records"]
        if isinstance(record, dict) and record.get("evaluation_split") == "dev"
    ]
    counts = Counter(record.get("slice") for record in dev)
    if len(dev) != 112 or counts != Counter({skill: 28 for skill in SKILLS}):
        raise Day20PreparationError("frozen dev must contain exactly 28 records per skill")
    sample_ids = [record.get("sample_id") for record in dev]
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise Day20PreparationError("frozen dev record is missing sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise Day20PreparationError("frozen dev has duplicate sample IDs")
    return dev


def _stable_order(records: Sequence[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (text_sha256(f"{seed}\0{record['sample_id']}"), record["sample_id"]),
    )


def _deterministic_index_orders(count: int, *, attempts: int = 32) -> list[list[int]]:
    if count <= 0:
        return []
    base = list(range(count))
    orders = [base, list(reversed(base))]
    offsets = range(1, count) if count <= attempts else (
        max(1, (index * count) // attempts) for index in range(1, attempts)
    )
    for offset in offsets:
        rotated = base[offset:] + base[:offset]
        if rotated not in orders:
            orders.append(rotated)
    return orders


def select_nested_exact_indices(
    weights: Sequence[int], *, probe_target: int, main_target: int
) -> tuple[set[int], set[int]]:
    """Jointly search disjoint probe and extension assignments.

    Exact subset DP supplies each candidate assignment.  Deterministic order
    backtracking avoids committing to the first exact probe (or main) when that
    choice blocks the nested counterpart.
    """
    if not 0 < probe_target < main_target:
        raise Day20PreparationError("nested targets must satisfy 0 < probe < main")

    def exact_from_order(order: list[int], target: int) -> set[int]:
        local = exact_subset_indices([weights[index] for index in order], target)
        return {order[index] for index in local}

    orders = _deterministic_index_orders(len(weights))
    # Main-first search: find an exact main set that contains an exact probe.
    for main_order in orders:
        try:
            main = exact_from_order(main_order, main_target)
        except Day20ContractError:
            continue
        main_members = sorted(main)
        for local_order in _deterministic_index_orders(len(main_members), attempts=16):
            probe_order = [main_members[index] for index in local_order]
            try:
                probe = exact_from_order(probe_order, probe_target)
            except Day20ContractError:
                continue
            return probe, main

    # Probe-first search covers pools where main-first tie-breaking is unlucky.
    extension_target = main_target - probe_target
    all_indices = set(range(len(weights)))
    for probe_order in orders:
        try:
            probe = exact_from_order(probe_order, probe_target)
        except Day20ContractError:
            continue
        remaining = sorted(all_indices - probe)
        for local_order in _deterministic_index_orders(len(remaining), attempts=16):
            extension_order = [remaining[index] for index in local_order]
            try:
                extension = exact_from_order(extension_order, extension_target)
            except Day20ContractError:
                continue
            return probe, probe | extension
    raise Day20PreparationError(
        "cannot construct a deterministic joint nested exact subset"
    )


def select_nested_datasets(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a 16k probe nested inside a 256k main set, exactly by format."""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pools[record["target_format"]].append(record)
    if set(pools) != set(TARGET_FORMATS):
        missing = sorted(set(TARGET_FORMATS) - set(pools))
        raise Day20PreparationError(f"normalized pools are missing target formats: {missing}")

    probe_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    for target_format in TARGET_FORMATS:
        ordered = _stable_order(pools[target_format], f"day20:{target_format}:v1")
        probe_target = PROBE_TOKENS_BY_FORMAT[target_format]
        main_target = MAIN_TOKENS_BY_FORMAT[target_format]
        try:
            probe_indices, main_indices = select_nested_exact_indices(
                [row["qwen35_supervised_tokens"] for row in ordered],
                probe_target=probe_target,
                main_target=main_target,
            )
        except (Day20ContractError, Day20PreparationError) as error:
            raise Day20PreparationError(
                f"cannot satisfy nested exact budget for {target_format}: {error}"
            ) from error
        probe = [row for index, row in enumerate(ordered) if index in probe_indices]
        main = [row for index, row in enumerate(ordered) if index in main_indices]
        probe_rows.extend(probe)
        main_rows.extend(main)

    return interleave_skills(probe_rows), interleave_skills(main_rows)


def interleave_skills(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    queues = {
        skill: deque(_stable_order([row for row in records if row["skill"] == skill], "day20:data-order:v1"))
        for skill in SKILLS
    }
    output: list[dict[str, Any]] = []
    while any(queues.values()):
        for skill in SKILLS:
            if queues[skill]:
                output.append(queues[skill].popleft())
    return output


def optimizer_step_skill_matrix(
    records: Sequence[Mapping[str, Any]], *, global_batch_size: int = 8
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for start in range(0, len(records), global_batch_size):
        window = records[start : start + global_batch_size]
        tokens = {
            skill: sum(
                int(row["qwen35_supervised_tokens"])
                for row in window
                if row.get("skill") == skill
            )
            for skill in SKILLS
        }
        matrix.append(
            {
                "optimizer_step": len(matrix) + 1,
                "records": len(window),
                "supervised_tokens": sum(tokens.values()),
                "supervised_tokens_by_skill": tokens,
            }
        )
    return matrix


def select_diagnostics(dev_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for skill in SKILLS:
        pool = [record for record in dev_records if record.get("slice") == skill]
        selected.extend(
            stable_hash_subset(
                pool,
                DIAGNOSTIC_RECORDS_PER_SKILL,
                seed=f"day20:diagnostic:{skill}:v1",
            )
        )
    queues = {
        skill: deque(record for record in selected if record.get("slice") == skill)
        for skill in SKILLS
    }
    output: list[dict[str, Any]] = []
    while any(queues.values()):
        for skill in SKILLS:
            if queues[skill]:
                output.append(queues[skill].popleft())
    return output


def _dataset_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens_by_skill = {
        skill: sum(
            int(row["qwen35_supervised_tokens"])
            for row in rows
            if row.get("skill") == skill
        )
        for skill in SKILLS
    }
    tokens_by_format = {
        target_format: sum(
            int(row["qwen35_supervised_tokens"])
            for row in rows
            if row.get("target_format") == target_format
        )
        for target_format in TARGET_FORMATS
    }
    records_by_format = dict(Counter(row.get("target_format") for row in rows))
    step_matrix = optimizer_step_skill_matrix(rows)
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "records": len(rows),
        "supervised_tokens": sum(tokens_by_skill.values()),
        "supervised_tokens_by_skill": tokens_by_skill,
        "supervised_tokens_by_format": tokens_by_format,
        "records_by_format": records_by_format,
        "ordered_sample_ids": [row["sample_id"] for row in rows],
        "ordered_supervised_tokens": [row["qwen35_supervised_tokens"] for row in rows],
        "selection_sha256": object_sha256([row["sample_id"] for row in rows]),
        "interleave": "round_robin_skill_v1",
        "optimizer_step_skill_matrix": step_matrix,
        "optimizer_step_skill_matrix_sha256": object_sha256(step_matrix),
    }


def prepare_run(
    *,
    run_root: Path,
    model_snapshot: str,
    sources: Mapping[str, Path],
    eval_manifest_path: Path,
    source_token_audit: Mapping[str, Any],
    source_adapter_manifest_path: Path,
    max_length: int = 2304,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    if not (run_root / ".day20-run-root").is_file():
        raise Day20PreparationError("run root marker .day20-run-root is missing")
    if max_length != QWEN35_TEMPLATE_CONTRACT["max_length"]:
        raise Day20PreparationError("Day 20 max_length is frozen at 2304")
    if set(sources) != set(SKILLS):
        raise Day20PreparationError("prepare_run requires exactly four skill sources")
    base_model_identity = build_base_model_identity(model_snapshot)

    all_records: list[dict[str, Any]] = []
    records_by_skill: dict[str, list[dict[str, Any]]] = {}
    source_evidence: dict[str, Any] = {}
    for skill in SKILLS:
        path = Path(sources[skill]).resolve()
        rows = load_normalized_jsonl(path, skill=skill, max_length=max_length)
        records_by_skill[skill] = rows
        all_records.extend(rows)
        source_evidence[skill] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "records": len(rows),
            "admission": "explicit_normalized_jsonl_adapter_output",
        }
    verified_token_audit = verify_source_token_audit(
        source_token_audit,
        model_snapshot=model_snapshot,
        sources=sources,
        records_by_skill=records_by_skill,
    )
    verified_source_adapter = verify_source_adapter_manifest(
        source_adapter_manifest_path.resolve(),
        model_snapshot=model_snapshot,
        sources=sources,
        records_by_skill=records_by_skill,
        source_token_audit=verified_token_audit,
    )
    validate_unique_training_records(all_records)

    dev_records = load_frozen_dev_records(eval_manifest_path.resolve())
    leakage = assert_no_train_dev_leakage(all_records, dev_records)
    probe_rows, main_rows = select_nested_datasets(all_records)
    diagnostics = select_diagnostics(dev_records)

    if sum(row["qwen35_supervised_tokens"] for row in probe_rows) != PROBE_SUPERVISED_TOKENS:
        raise Day20PreparationError("probe token budget drift")
    if sum(row["qwen35_supervised_tokens"] for row in main_rows) != MAIN_SUPERVISED_TOKENS:
        raise Day20PreparationError("main token budget drift")
    for rows, expected in (
        (probe_rows, PROBE_TOKENS_PER_SKILL),
        (main_rows, MAIN_TOKENS_PER_SKILL),
    ):
        actual = Counter()
        for row in rows:
            actual[row["skill"]] += row["qwen35_supervised_tokens"]
        if actual != Counter({skill: expected for skill in SKILLS}):
            raise Day20PreparationError(f"per-skill token budget drift: {dict(actual)}")
    for rows, minimums, label in (
        (probe_rows, MIN_PROBE_RECORDS_BY_FORMAT, "probe"),
        (main_rows, MIN_MAIN_RECORDS_BY_FORMAT, "main"),
    ):
        counts = Counter(row["target_format"] for row in rows)
        failures = {
            target_format: {"actual": counts[target_format], "minimum": minimum}
            for target_format, minimum in minimums.items()
            if counts[target_format] < minimum
        }
        if failures:
            raise Day20PreparationError(
                f"{label} diversity/minimum-record contract failed: {failures}"
            )
    if len(optimizer_step_skill_matrix(probe_rows)) < 5:
        raise Day20PreparationError("probe dataset cannot reach the five-step safety gate")

    data_paths = {
        "probe": run_root / "data" / "probe.jsonl",
        "main": run_root / "data" / "main.jsonl",
        "diagnostic": run_root / "data" / "diagnostic.jsonl",
    }
    config_paths = {
        f"{learning_rate:.0e}".replace("e-0", "e-"): run_root
        / "configs"
        / f"probe-lr-{f'{learning_rate:.0e}'.replace('e-0', 'e-')}.json"
        for learning_rate in PROBE_LEARNING_RATES
    }
    main_template_path = run_root / "configs" / "main-template.json"
    manifest_path = run_root / "DAY20-MANIFEST.json"
    planned = [*data_paths.values(), *config_paths.values(), main_template_path, manifest_path]
    collisions = [str(path) for path in planned if path.exists()]
    if collisions:
        raise Day20PreparationError(f"refusing to overwrite existing artifacts: {collisions}")

    _write_jsonl(data_paths["probe"], probe_rows)
    _write_jsonl(data_paths["main"], main_rows)
    _write_jsonl(data_paths["diagnostic"], diagnostics)
    probe_summary = _dataset_summary(data_paths["probe"], probe_rows)
    main_summary = _dataset_summary(data_paths["main"], main_rows)
    diagnostic_summary = {
        "path": str(data_paths["diagnostic"]),
        "file_sha256": file_sha256(data_paths["diagnostic"]),
        "records": len(diagnostics),
        "records_by_skill": dict(Counter(row["slice"] for row in diagnostics)),
        "ordered_sample_ids": [row["sample_id"] for row in diagnostics],
        "selection_sha256": object_sha256([row["sample_id"] for row in diagnostics]),
        "source_eval_manifest": str(eval_manifest_path.resolve()),
        "source_eval_manifest_sha256": file_sha256(eval_manifest_path.resolve()),
    }

    probe_configs: dict[str, Any] = {}
    for learning_rate in PROBE_LEARNING_RATES:
        key = f"{learning_rate:.0e}".replace("e-0", "e-")
        config = build_probe_config(
            model_snapshot=model_snapshot,
            dataset_path=str(data_paths["probe"]),
            dataset_sha256=probe_summary["file_sha256"],
            learning_rate=learning_rate,
        )
        _write_json(config_paths[key], config)
        probe_configs[key] = {
            "path": str(config_paths[key]),
            "file_sha256": file_sha256(config_paths[key]),
            "immutable_sha256": config["immutable_sha256"],
            "learning_rate": learning_rate,
        }
    main_template = build_main_config(
        model_snapshot=model_snapshot,
        dataset_path=str(data_paths["main"]),
        dataset_sha256=main_summary["file_sha256"],
    )
    _write_json(main_template_path, main_template)

    immutable = immutable_experiment_contract(model_snapshot)
    if build_base_model_identity(model_snapshot) != base_model_identity:
        raise Day20PreparationError("Base model changed during Day 20 preparation")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "domain": "day20.qwen35_balanced_lora.experiment_manifest",
        "status": "prepared_probes_not_started",
        "created_at_utc": utc_now(),
        "run_root": str(run_root),
        "contract": {
            "immutable": immutable,
            "immutable_sha256": object_sha256(immutable),
        },
        "base_model_identity": base_model_identity,
        "sources": source_evidence,
        "source_adapter": verified_source_adapter,
        "source_token_reaudit": verified_token_audit,
        "datasets": {
            "probe": probe_summary,
            "main": main_summary,
            "diagnostic": diagnostic_summary,
        },
        "configs": {
            "probes": probe_configs,
            "main_template": {
                "path": str(main_template_path),
                "file_sha256": file_sha256(main_template_path),
                "immutable_sha256": main_template["immutable_sha256"],
                "learning_rate": "__SELECT_FROM_PASSING_PROBE__",
            },
        },
        "leakage": {
            **leakage,
            "status": "pass",
            "checked_records": len(all_records),
        },
        "selection": {
            "algorithm": "probe_first_nested_exact_bitset_stable_hash_v1",
            "diagnostic_algorithm": "per_skill_stable_hash_v1",
            "probe_is_subset_of_main": set(row["sample_id"] for row in probe_rows)
            <= set(row["sample_id"] for row in main_rows),
        },
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bootcamp-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--fetch-adapter")
    parser.add_argument("--fetch-config", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--target-tokens", type=int, default=MAIN_SUPERVISED_TOKENS)
    parser.add_argument("--probe-tokens", type=int, default=PROBE_SUPERVISED_TOKENS)
    parser.add_argument("--max-length", type=int, default=2304)
    args = parser.parse_args()

    if args.target_tokens != MAIN_SUPERVISED_TOKENS:
        raise Day20PreparationError("Day 20 main budget is frozen at 256,000 tokens")
    if args.probe_tokens != PROBE_SUPERVISED_TOKENS:
        raise Day20PreparationError("Day 20 probe budget is frozen at 16,000 tokens")
    run_root = args.run_root.resolve()
    sources = resolve_sources(
        source_specs=args.source,
        fetch_adapter=args.fetch_adapter,
        fetch_config_path=args.fetch_config,
        run_root=run_root,
    )
    try:
        from day20_source_adapter import audit_normalized_sources
    except ImportError as error:
        raise Day20PreparationError(
            "the pinned Day 20 Qwen3.5 token auditor is unavailable"
        ) from error
    source_token_audit = audit_normalized_sources(
        model_path=args.model,
        sources=sources,
        max_length=args.max_length,
    )
    source_parents = {Path(path).resolve().parent for path in sources.values()}
    if len(source_parents) != 1:
        raise Day20PreparationError(
            "all normalized sources must share one pinned adapter directory"
        )
    source_adapter_manifest_path = (
        next(iter(source_parents)) / "SOURCE-ADAPTER-MANIFEST.json"
    )
    eval_manifest = (
        args.eval_manifest.resolve()
        if args.eval_manifest
        else (args.bootcamp_root.resolve() / "artifacts/eval/day10-frozen-eval-manifest.json")
    )
    manifest = prepare_run(
        run_root=run_root,
        model_snapshot=args.model,
        sources=sources,
        eval_manifest_path=eval_manifest,
        source_token_audit=source_token_audit,
        source_adapter_manifest_path=source_adapter_manifest_path,
        max_length=args.max_length,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
