#!/usr/bin/env python3
"""Prepare append-only Day 20 v2 probe/main datasets and training configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from day20_candidate_factory_v2 import (
    CONFIRMATION_SEED,
    CONTRACT_VERSION,
    MAIN_CHECKPOINT_TOKENS,
    PRIMARY_SEED,
    PROBE_CHECKPOINT_TOKENS,
    PROBE_LRS,
)
from day20_contract import (
    DIAGNOSTIC_RECORDS_PER_SKILL,
    LORA_TARGET_LEAVES,
    LORA_TARGET_REGEX,
    MIN_MAIN_RECORDS_BY_FORMAT,
    MIN_PROBE_RECORDS_BY_FORMAT,
    QWEN35_TEMPLATE_CONTRACT,
    SKILLS,
    TARGET_FORMATS,
    assert_no_train_dev_leakage,
    exact_subset_indices,
    object_sha256,
    stable_hash_subset,
    text_sha256,
    validate_unique_training_records,
)
from day20_contract_v2 import (
    MAIN_SUPERVISED_TOKENS,
    MAIN_TOKENS_BY_FORMAT,
    MAIN_TOKENS_PER_SKILL,
    PROBE_SUPERVISED_TOKENS,
    PROBE_TOKENS_BY_FORMAT,
    PROBE_TOKENS_PER_SKILL,
    V2_CONTRACT_VERSION,
)
from day20_ordering_v2 import (
    ORDERING_VERSION,
    audit_temporal_mix,
    order_records,
)
from day20_source_expansion_v2 import (
    ADAPTER_VERSION,
    FINQA_ADAPTER,
    TULU_CODE_ADAPTER,
    file_sha256,
    tokenizer_file_identity,
    validate_v2_normalized_record,
)


MANIFEST_DOMAIN = "day20.qwen35_balanced_lora.experiment_manifest.v2"
CONFIG_DOMAIN = "day20.qwen35_balanced_lora.training_config.v2"
RUN_ROOT_MARKER = ".day20-v2-run-root"
PROBE_MIN_COHORT_TOKENS = {
    "finance_value_scale": {FINQA_ADAPTER: 1_000},
    "code_continuation": {TULU_CODE_ADAPTER: 1_000},
}
MAIN_MIN_COHORT_TOKENS = {
    "finance_value_scale": {FINQA_ADAPTER: 15_000},
    "code_continuation": {TULU_CODE_ADAPTER: 15_000},
}


class Day20PreparationV2Error(ValueError):
    """A source, selection, ordering, or output invariant failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Day20PreparationV2Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Day20PreparationV2Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_source_specs(specs: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        skill, separator, raw_path = spec.partition("=")
        if not separator or skill not in SKILLS or not raw_path or skill in result:
            raise Day20PreparationV2Error(
                "--source must provide each of general/math/finance/code exactly once"
            )
        result[skill] = Path(raw_path).expanduser().resolve()
    if set(result) != set(SKILLS):
        raise Day20PreparationV2Error("four --source inputs are required")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day20PreparationV2Error(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day20PreparationV2Error(f"JSON root must be an object: {path}")
    return value


def _self_hash_valid(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == object_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def build_model_identity(model_snapshot: Path) -> dict[str, Any]:
    model = model_snapshot.expanduser().resolve()
    if not model.is_dir():
        raise Day20PreparationV2Error(f"Base model directory is missing: {model}")
    files = {
        path.relative_to(model).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(model.rglob("*"))
        if path.is_file()
    }
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise Day20PreparationV2Error("Base model file manifest is incomplete")
    return {
        "path": str(model),
        "files": files,
        "snapshot_sha256": object_sha256(files),
    }


def verify_source_expansion_manifest(
    manifest_path: Path,
    *,
    model_snapshot: Path,
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path.resolve())
    outputs = manifest.get("outputs")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("domain") != "day20.qwen35_source_expansion.v2"
        or manifest.get("status") != "pass"
        or manifest.get("adapter_version") != ADAPTER_VERSION
        or manifest.get("dataset_code_execution") is not False
        or not _self_hash_valid(manifest, "manifest_sha256")
        or manifest.get("model_path") != str(model_snapshot.expanduser().resolve())
        or manifest.get("tokenizer_files")
        != tokenizer_file_identity(model_snapshot.expanduser().resolve())
        or manifest.get("template") != QWEN35_TEMPLATE_CONTRACT
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(SKILLS)
    ):
        raise Day20PreparationV2Error("source-expansion manifest identity drifted")
    for skill in SKILLS:
        identity = outputs[skill]
        source = Path(sources[skill]).resolve()
        if (
            not isinstance(identity, Mapping)
            or Path(str(identity.get("path", ""))).resolve() != source
            or identity.get("file_sha256") != file_sha256(source)
        ):
            raise Day20PreparationV2Error(f"expanded source identity drifted: {skill}")
    return {
        "path": str(manifest_path.resolve()),
        "file_sha256": file_sha256(manifest_path.resolve()),
        "content_sha256": manifest["manifest_sha256"],
        "adapter_version": ADAPTER_VERSION,
        "day09_manifest": manifest.get("day09_manifest"),
    }


def load_normalized_sources(sources: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        path = Path(sources[skill]).resolve()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise Day20PreparationV2Error(f"cannot read normalized source: {path}") from error
        if not lines:
            raise Day20PreparationV2Error(f"normalized source is empty: {path}")
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise Day20PreparationV2Error(f"blank JSONL row: {path}:{line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise Day20PreparationV2Error(
                    f"invalid JSONL: {path}:{line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise Day20PreparationV2Error(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            try:
                validated = validate_v2_normalized_record(raw, expected_skill=skill)
            except ValueError as error:
                raise Day20PreparationV2Error(
                    f"invalid normalized row {path}:{line_number}: {error}"
                ) from error
            tokenization = validated.pop("qwen35_tokenization")
            messages = validated["messages"]
            validated.update(
                {
                    "qwen35_input_tokens": tokenization["input_tokens"],
                    "qwen35_supervised_tokens": tokenization["supervised_tokens"],
                    "qwen35_render_sha256": tokenization["render_sha256"],
                    "qwen35_labels_sha256": tokenization["labels_sha256"],
                    "prompt_sha256": object_sha256(messages[:-1]),
                    "content_sha256": object_sha256(messages),
                    "source_content_sha256": validated["source_lineage"][
                        "source_content_sha256"
                    ],
                }
            )
            rows.append(validated)
    validate_unique_training_records(rows)
    return rows


def _stable_order(
    rows: Sequence[dict[str, Any]], seed: str
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (text_sha256(f"{seed}\0{row['sample_id']}"), row["sample_id"]),
    )


def _orders(count: int, attempts: int = 16) -> list[list[int]]:
    if count <= 0:
        return []
    base = list(range(count))
    result = [base]
    if count > 1:
        result.append(list(reversed(base)))
    for attempt in range(1, min(attempts, count)):
        offset = max(1, attempt * count // attempts)
        rotated = base[offset:] + base[:offset]
        if rotated not in result:
            result.append(rotated)
    return result


def _reachable_bits(weights: Sequence[int], target: int) -> int:
    bits = 1
    mask = (1 << (target + 1)) - 1
    for weight in weights:
        bits = (bits | (bits << weight)) & mask
    return bits


def _partition_contributions(
    cohort_weights: Sequence[int],
    other_weights: Sequence[int],
    *,
    target: int,
    minimum_cohort: int,
) -> list[int]:
    cohort_bits = _reachable_bits(cohort_weights, target)
    other_bits = _reachable_bits(other_weights, target)
    return [
        contribution
        for contribution in range(minimum_cohort, target + 1)
        if ((cohort_bits >> contribution) & 1)
        and ((other_bits >> (target - contribution)) & 1)
    ]


def _exact_partition(
    cohort: Sequence[dict[str, Any]],
    other: Sequence[dict[str, Any]],
    *,
    target: int,
    cohort_tokens: int,
    cohort_order: Sequence[int],
    other_order: Sequence[int],
) -> tuple[set[int], set[int]]:
    ordered_cohort_weights = [
        int(cohort[index]["qwen35_supervised_tokens"]) for index in cohort_order
    ]
    ordered_other_weights = [
        int(other[index]["qwen35_supervised_tokens"]) for index in other_order
    ]
    cohort_local = exact_subset_indices(ordered_cohort_weights, cohort_tokens)
    other_local = exact_subset_indices(ordered_other_weights, target - cohort_tokens)
    return (
        {cohort_order[index] for index in cohort_local},
        {other_order[index] for index in other_local},
    )


def select_nested_with_cohort_quota(
    rows: Sequence[dict[str, Any]],
    *,
    probe_target: int,
    main_target: int,
    cohort_predicate: Callable[[Mapping[str, Any]], bool],
    probe_minimum: int,
    main_minimum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Find exact nested subsets while enforcing new-source token floors."""
    cohort = [row for row in rows if cohort_predicate(row)]
    other = [row for row in rows if not cohort_predicate(row)]
    if not cohort or not other:
        raise Day20PreparationV2Error("cohort quota requires cohort and replay rows")
    cohort_weights = [int(row["qwen35_supervised_tokens"]) for row in cohort]
    other_weights = [int(row["qwen35_supervised_tokens"]) for row in other]
    probe_contributions = _partition_contributions(
        cohort_weights,
        other_weights,
        target=probe_target,
        minimum_cohort=probe_minimum,
    )
    if not probe_contributions:
        raise Day20PreparationV2Error("probe new-source quota is not exactly reachable")

    for attempt in range(max(len(_orders(len(cohort))), len(_orders(len(other))))):
        cohort_order = _orders(len(cohort))[attempt % len(_orders(len(cohort)))]
        other_order = _orders(len(other))[attempt % len(_orders(len(other)))]
        for probe_cohort_tokens in probe_contributions[:64]:
            try:
                probe_cohort, probe_other = _exact_partition(
                    cohort,
                    other,
                    target=probe_target,
                    cohort_tokens=probe_cohort_tokens,
                    cohort_order=cohort_order,
                    other_order=other_order,
                )
            except ValueError:
                continue
            remaining_cohort = [
                row for index, row in enumerate(cohort) if index not in probe_cohort
            ]
            remaining_other = [
                row for index, row in enumerate(other) if index not in probe_other
            ]
            extension_target = main_target - probe_target
            needed = max(0, main_minimum - probe_cohort_tokens)
            extension_contributions = _partition_contributions(
                [int(row["qwen35_supervised_tokens"]) for row in remaining_cohort],
                [int(row["qwen35_supervised_tokens"]) for row in remaining_other],
                target=extension_target,
                minimum_cohort=needed,
            )
            if not extension_contributions:
                continue
            remaining_cohort_order = list(range(len(remaining_cohort)))
            remaining_other_order = list(range(len(remaining_other)))
            for extension_cohort_tokens in extension_contributions[:64]:
                try:
                    extension_cohort, extension_other = _exact_partition(
                        remaining_cohort,
                        remaining_other,
                        target=extension_target,
                        cohort_tokens=extension_cohort_tokens,
                        cohort_order=remaining_cohort_order,
                        other_order=remaining_other_order,
                    )
                except ValueError:
                    continue
                probe = [cohort[index] for index in sorted(probe_cohort)] + [
                    other[index] for index in sorted(probe_other)
                ]
                main = probe + [
                    remaining_cohort[index] for index in sorted(extension_cohort)
                ] + [remaining_other[index] for index in sorted(extension_other)]
                return probe, main
    raise Day20PreparationV2Error(
        "cannot construct nested exact subsets under the new-source quota"
    )


def _select_nested_unconstrained(
    rows: Sequence[dict[str, Any]], *, probe_target: int, main_target: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = [int(row["qwen35_supervised_tokens"]) for row in rows]
    for order in _orders(len(rows), attempts=32):
        try:
            ordered_weights = [weights[index] for index in order]
            main_local = exact_subset_indices(ordered_weights, main_target)
            main_indices = {order[index] for index in main_local}
            main_members = sorted(main_indices)
            for probe_order in _orders(len(main_members), attempts=16):
                local_weights = [weights[main_members[index]] for index in probe_order]
                probe_local = exact_subset_indices(local_weights, probe_target)
                probe_indices = {
                    main_members[probe_order[index]] for index in probe_local
                }
                return (
                    [rows[index] for index in sorted(probe_indices)],
                    [rows[index] for index in sorted(main_indices)],
                )
        except ValueError:
            continue
    raise Day20PreparationV2Error("nested exact token subset is unreachable")


def _adapter_is(name: str) -> Callable[[Mapping[str, Any]], bool]:
    def predicate(row: Mapping[str, Any]) -> bool:
        lineage = row.get("source_lineage")
        return isinstance(lineage, Mapping) and lineage.get("adapter") == name

    return predicate


def select_nested_datasets(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pools[str(row["target_format"])].append(row)
    if set(pools) != set(TARGET_FORMATS):
        raise Day20PreparationV2Error("normalized pools do not cover all target formats")
    probe: list[dict[str, Any]] = []
    main: list[dict[str, Any]] = []
    for target_format in TARGET_FORMATS:
        ordered = _stable_order(
            pools[target_format], f"day20:v2:selection:{target_format}"
        )
        if target_format == "finance_value_scale":
            selected_probe, selected_main = select_nested_with_cohort_quota(
                ordered,
                probe_target=PROBE_TOKENS_BY_FORMAT[target_format],
                main_target=MAIN_TOKENS_BY_FORMAT[target_format],
                cohort_predicate=_adapter_is(FINQA_ADAPTER),
                probe_minimum=PROBE_MIN_COHORT_TOKENS[target_format][FINQA_ADAPTER],
                main_minimum=MAIN_MIN_COHORT_TOKENS[target_format][FINQA_ADAPTER],
            )
        elif target_format == "code_continuation":
            selected_probe, selected_main = select_nested_with_cohort_quota(
                ordered,
                probe_target=PROBE_TOKENS_BY_FORMAT[target_format],
                main_target=MAIN_TOKENS_BY_FORMAT[target_format],
                cohort_predicate=_adapter_is(TULU_CODE_ADAPTER),
                probe_minimum=PROBE_MIN_COHORT_TOKENS[target_format][TULU_CODE_ADAPTER],
                main_minimum=MAIN_MIN_COHORT_TOKENS[target_format][TULU_CODE_ADAPTER],
            )
        else:
            selected_probe, selected_main = _select_nested_unconstrained(
                ordered,
                probe_target=PROBE_TOKENS_BY_FORMAT[target_format],
                main_target=MAIN_TOKENS_BY_FORMAT[target_format],
            )
        probe.extend(selected_probe)
        main.extend(selected_main)
    try:
        return order_records(probe), order_records(main)
    except ValueError as error:
        raise Day20PreparationV2Error(f"temporal ordering failed: {error}") from error


def load_frozen_dev(path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(path.resolve())
    records = manifest.get("records")
    if not isinstance(records, list):
        raise Day20PreparationV2Error("eval manifest records are missing")
    dev = [
        dict(row)
        for row in records
        if isinstance(row, dict) and row.get("evaluation_split") == "dev"
    ]
    if len(dev) != 112 or Counter(row.get("slice") for row in dev) != Counter(
        {skill: 28 for skill in SKILLS}
    ):
        raise Day20PreparationV2Error("frozen dev must have 28 rows per skill")
    ids = [row.get("sample_id") for row in dev]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise Day20PreparationV2Error("frozen dev sample IDs are invalid")
    return dev


def select_diagnostics(dev: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for skill in SKILLS:
        selected.extend(
            stable_hash_subset(
                [row for row in dev if row.get("slice") == skill],
                DIAGNOSTIC_RECORDS_PER_SKILL,
                seed=f"day20:v2:diagnostic:{skill}",
            )
        )
    queues = {
        skill: deque(row for row in selected if row.get("slice") == skill)
        for skill in SKILLS
    }
    ordered: list[dict[str, Any]] = []
    while any(queues.values()):
        for skill in SKILLS:
            if queues[skill]:
                ordered.append(queues[skill].popleft())
    return ordered


def _cohort_tokens(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        adapter: sum(
            int(row["qwen35_supervised_tokens"])
            for row in rows
            if isinstance(row.get("source_lineage"), Mapping)
            and row["source_lineage"].get("adapter") == adapter
        )
        for adapter in (FINQA_ADAPTER, TULU_CODE_ADAPTER)
    }


def dataset_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    temporal_mix = audit_temporal_mix(rows)
    if temporal_mix.get("status") != "pass":
        raise Day20PreparationV2Error("prepared dataset failed temporal mix audit")
    by_skill = {
        skill: sum(
            int(row["qwen35_supervised_tokens"])
            for row in rows
            if row.get("skill") == skill
        )
        for skill in SKILLS
    }
    by_format = {
        target_format: sum(
            int(row["qwen35_supervised_tokens"])
            for row in rows
            if row.get("target_format") == target_format
        )
        for target_format in TARGET_FORMATS
    }
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path.resolve()),
        "records": len(rows),
        "supervised_tokens": sum(by_skill.values()),
        "supervised_tokens_by_skill": by_skill,
        "supervised_tokens_by_format": by_format,
        "records_by_format": dict(sorted(Counter(row["target_format"] for row in rows).items())),
        "cohort_supervised_tokens": _cohort_tokens(rows),
        "ordered_sample_ids": [row["sample_id"] for row in rows],
        "ordered_supervised_tokens": [row["qwen35_supervised_tokens"] for row in rows],
        "selection_sha256": object_sha256([row["sample_id"] for row in rows]),
        "ordering_version": ORDERING_VERSION,
        "temporal_mix_audit": temporal_mix,
        "temporal_mix_sha256": object_sha256(temporal_mix),
    }


def build_training_config(
    *,
    run_kind: str,
    model_snapshot: Path,
    dataset: Mapping[str, Any],
    learning_rate: str,
    seed: int,
    requires_resolution: bool = False,
) -> dict[str, Any]:
    checkpoints = (
        dict(PROBE_CHECKPOINT_TOKENS)
        if run_kind == "probe"
        else dict(MAIN_CHECKPOINT_TOKENS)
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "domain": CONFIG_DOMAIN,
        "run_kind": run_kind,
        "requires_resolution": requires_resolution,
        "model": {
            "snapshot": str(model_snapshot.expanduser().resolve()),
            "model_type": "qwen3_5",
            "parent": "untouched_base",
        },
        "data": {
            "path": dataset["path"],
            "file_sha256": dataset["file_sha256"],
            "supervised_tokens": dataset["supervised_tokens"],
            "temporal_mix_sha256": dataset["temporal_mix_sha256"],
            "ordering_version": ORDERING_VERSION,
        },
        "template": dict(QWEN35_TEMPLATE_CONTRACT),
        "training": {
            "tuner_type": "lora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_regex": LORA_TARGET_REGEX,
            "target_leaves": list(LORA_TARGET_LEAVES),
            "freeze_vit": True,
            "freeze_aligner": True,
            "freeze_embeddings": True,
            "freeze_lm_head": True,
            "optimizer": "adamw_torch",
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 8,
            "seed": seed,
            "data_seed": seed,
            "learning_rate": learning_rate,
            "checkpoint_supervised_tokens": list(checkpoints.values()),
            "checkpoint_policy": "all_milestones_resumable_v2",
            "save_only_model": False,
            "save_total_limit": len(checkpoints),
            "num_train_epochs": 1.0,
        },
    }
    result["immutable_sha256"] = object_sha256(result)
    return result


def _verify_budgets(
    probe: Sequence[Mapping[str, Any]], main: Sequence[Mapping[str, Any]]
) -> None:
    for label, rows, total, per_skill, by_format, minimums in (
        (
            "probe",
            probe,
            PROBE_SUPERVISED_TOKENS,
            PROBE_TOKENS_PER_SKILL,
            PROBE_TOKENS_BY_FORMAT,
            MIN_PROBE_RECORDS_BY_FORMAT,
        ),
        (
            "main",
            main,
            MAIN_SUPERVISED_TOKENS,
            MAIN_TOKENS_PER_SKILL,
            MAIN_TOKENS_BY_FORMAT,
            MIN_MAIN_RECORDS_BY_FORMAT,
        ),
    ):
        actual_total = sum(int(row["qwen35_supervised_tokens"]) for row in rows)
        skill_tokens = Counter()
        format_tokens = Counter()
        format_records = Counter()
        for row in rows:
            tokens = int(row["qwen35_supervised_tokens"])
            skill_tokens[row["skill"]] += tokens
            format_tokens[row["target_format"]] += tokens
            format_records[row["target_format"]] += 1
        if actual_total != total:
            raise Day20PreparationV2Error(f"{label} total token budget drifted")
        if skill_tokens != Counter({skill: per_skill for skill in SKILLS}):
            raise Day20PreparationV2Error(f"{label} per-skill token budget drifted")
        if format_tokens != Counter(by_format):
            raise Day20PreparationV2Error(f"{label} per-format token budget drifted")
        failures = {
            name: (format_records[name], minimum)
            for name, minimum in minimums.items()
            if format_records[name] < minimum
        }
        if failures:
            raise Day20PreparationV2Error(f"{label} record diversity failed: {failures}")
    probe_ids = {row["sample_id"] for row in probe}
    main_ids = {row["sample_id"] for row in main}
    if not probe_ids <= main_ids:
        raise Day20PreparationV2Error("probe is not nested inside main")
    for label, rows, required in (
        ("probe", probe, PROBE_MIN_COHORT_TOKENS),
        ("main", main, MAIN_MIN_COHORT_TOKENS),
    ):
        cohorts = _cohort_tokens(rows)
        for target_format, adapters in required.items():
            del target_format
            for adapter, minimum in adapters.items():
                if cohorts[adapter] < minimum:
                    raise Day20PreparationV2Error(
                        f"{label} {adapter} token floor failed"
                    )


def prepare_run(
    *,
    run_root: Path,
    model_snapshot: Path,
    sources: Mapping[str, Path],
    source_expansion_manifest: Path,
    eval_manifest: Path,
) -> dict[str, Any]:
    root = run_root.resolve()
    marker = root / RUN_ROOT_MARKER
    if not marker.is_file() or marker.is_symlink() or marker.read_text(encoding="utf-8").strip() != "day20-qwen35-candidate-factory-v2":
        raise Day20PreparationV2Error(f"valid {RUN_ROOT_MARKER} is required")
    model_identity = build_model_identity(model_snapshot)
    source_identity = verify_source_expansion_manifest(
        source_expansion_manifest,
        model_snapshot=model_snapshot,
        sources=sources,
    )
    all_rows = load_normalized_sources(sources)
    dev = load_frozen_dev(eval_manifest)
    leakage = assert_no_train_dev_leakage(all_rows, dev)
    probe, main = select_nested_datasets(all_rows)
    _verify_budgets(probe, main)
    diagnostics = select_diagnostics(dev)

    paths = {
        "probe": root / "data" / "probe-v2.jsonl",
        "main": root / "data" / "main-v2.jsonl",
        "diagnostic": root / "data" / "diagnostic-v2.jsonl",
    }
    probe_config_paths = {
        lr: root / "configs" / f"probe-s{PRIMARY_SEED}-lr{lr}.json"
        for lr in PROBE_LRS
    }
    main_template_path = root / "configs" / "main-template-v2.json"
    manifest_path = root / "DAY20-V2-MANIFEST.json"
    planned = [*paths.values(), *probe_config_paths.values(), main_template_path, manifest_path]
    collisions = [str(path) for path in planned if path.exists()]
    if collisions:
        raise Day20PreparationV2Error(f"refusing to overwrite artifacts: {collisions}")

    _write_jsonl_new(paths["probe"], probe)
    _write_jsonl_new(paths["main"], main)
    _write_jsonl_new(paths["diagnostic"], diagnostics)
    probe_summary = dataset_summary(paths["probe"], probe)
    main_summary = dataset_summary(paths["main"], main)
    diagnostic_summary = {
        "path": str(paths["diagnostic"].resolve()),
        "file_sha256": file_sha256(paths["diagnostic"].resolve()),
        "records": len(diagnostics),
        "records_by_skill": dict(sorted(Counter(row["slice"] for row in diagnostics).items())),
        "ordered_sample_ids": [row["sample_id"] for row in diagnostics],
        "selection_sha256": object_sha256([row["sample_id"] for row in diagnostics]),
        "eval_manifest_path": str(eval_manifest.resolve()),
        "eval_manifest_file_sha256": file_sha256(eval_manifest.resolve()),
    }

    probe_configs: dict[str, Any] = {}
    for lr in PROBE_LRS:
        config = build_training_config(
            run_kind="probe",
            model_snapshot=model_snapshot,
            dataset=probe_summary,
            learning_rate=lr,
            seed=PRIMARY_SEED,
        )
        _write_json_new(probe_config_paths[lr], config)
        probe_configs[lr] = {
            "path": str(probe_config_paths[lr].resolve()),
            "file_sha256": file_sha256(probe_config_paths[lr].resolve()),
            "immutable_sha256": config["immutable_sha256"],
            "learning_rate": lr,
            "seed": PRIMARY_SEED,
        }
    main_template = build_training_config(
        run_kind="main",
        model_snapshot=model_snapshot,
        dataset=main_summary,
        learning_rate="__SELECT_FROM_PASSING_PROBE__",
        seed=PRIMARY_SEED,
        requires_resolution=True,
    )
    _write_json_new(main_template_path, main_template)

    if build_model_identity(model_snapshot) != model_identity:
        raise Day20PreparationV2Error("Base model changed during preparation")
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "domain": MANIFEST_DOMAIN,
        "status": "prepared_stage_a_not_started",
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "contract": {
            "candidate_factory_version": CONTRACT_VERSION,
            "data_contract_version": V2_CONTRACT_VERSION,
            "source_adapter_version": ADAPTER_VERSION,
            "ordering_version": ORDERING_VERSION,
            "probe_supervised_tokens": PROBE_SUPERVISED_TOKENS,
            "main_supervised_tokens": MAIN_SUPERVISED_TOKENS,
            "probe_tokens_by_format": dict(PROBE_TOKENS_BY_FORMAT),
            "main_tokens_by_format": dict(MAIN_TOKENS_BY_FORMAT),
            "probe_min_cohort_tokens": PROBE_MIN_COHORT_TOKENS,
            "main_min_cohort_tokens": MAIN_MIN_COHORT_TOKENS,
            "primary_seed": PRIMARY_SEED,
            "confirmation_seed": CONFIRMATION_SEED,
            "probe_learning_rates": list(PROBE_LRS),
            "probe_checkpoint_tokens": dict(PROBE_CHECKPOINT_TOKENS),
            "main_checkpoint_tokens": dict(MAIN_CHECKPOINT_TOKENS),
        },
        "base_model_identity": model_identity,
        "source_expansion": source_identity,
        "datasets": {
            "probe": probe_summary,
            "main": main_summary,
            "diagnostic": diagnostic_summary,
        },
        "configs": {
            "probes": probe_configs,
            "main_template": {
                "path": str(main_template_path.resolve()),
                "file_sha256": file_sha256(main_template_path.resolve()),
                "immutable_sha256": main_template["immutable_sha256"],
                "learning_rate": "__SELECT_FROM_PASSING_PROBE__",
                "seed": PRIMARY_SEED,
            },
        },
        "leakage": {**leakage, "status": "pass", "checked_records": len(all_rows)},
        "selection": {
            "algorithm": "nested_exact_token_with_new_source_quota_v2",
            "probe_is_subset_of_main": {row["sample_id"] for row in probe}
            <= {row["sample_id"] for row in main},
        },
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    _write_json_new(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--source-expansion-manifest", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_run(
        run_root=args.run_root,
        model_snapshot=args.model,
        sources=parse_source_specs(args.source),
        source_expansion_manifest=args.source_expansion_manifest,
        eval_manifest=args.eval_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
