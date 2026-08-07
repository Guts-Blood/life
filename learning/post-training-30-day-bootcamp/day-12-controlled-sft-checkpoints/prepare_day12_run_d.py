#!/usr/bin/env python3
"""Build the task-routed, FinQA-format-repaired Day 12 Run D artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
DAY08_DIR = BOOTCAMP_ROOT / "day-08-sft-data-contract"
DAY09_DIR = BOOTCAMP_ROOT / "day-09-data-quality-mixture-lineage"
CONFIG_DIR = BOOTCAMP_ROOT / "artifacts" / "configs"
DATA_DIR = BOOTCAMP_ROOT / "artifacts" / "data"
REPORT_DIR = BOOTCAMP_ROOT / "artifacts" / "reports"

DEFAULT_SPEC = CONFIG_DIR / "day12-format-repair-D-spec.json"
DEFAULT_BASE_CONFIG = CONFIG_DIR / "day12-controlled-sft-base.yaml"
DEFAULT_PLAN = DATA_DIR / "day12-format-repair-D-plan.json"
DEFAULT_MANIFEST = DATA_DIR / "day12-mix-D-format-repaired.json"
DEFAULT_CONFIG = CONFIG_DIR / "day12-controlled-sft-D.yaml"
DEFAULT_SCHEDULE = DATA_DIR / "day12-training-schedule-D.json"
DEFAULT_SUMMARY = REPORT_DIR / "day12-format-repair-D-preparation.json"
SKILLS = ("general", "math", "code", "finance")
CHECKPOINTS = ("25_percent", "60_percent", "100_percent")
OCCURRENCE_FIELDS = {
    "canonical_sample_id",
    "occurrence_id",
    "occurrence_index",
    "sampling_count",
    "supervised_tokens_per_occurrence",
}


class RunDPreparationError(ValueError):
    """The format-repaired run cannot satisfy its frozen data contract."""


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunDPreparationError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DAY08 = load_module("day08_contract_for_run_d", DAY08_DIR / "inspect_sft_sample.py")
DAY09_BUDGET = load_module(
    "day09_budget_for_run_d", DAY09_DIR / "plan_day09_token_budget.py"
)
DAY09_MIX = load_module(
    "day09_mixture_for_run_d", DAY09_DIR / "build_day09_mixtures.py"
)
DAY12 = load_module("day12_preparation_for_run_d", HERE / "prepare_day12_experiment.py")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RunDPreparationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunDPreparationError(f"JSON root must be an object: {path}")
    return value


def write_json_stable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RunDPreparationError(f"refusing to overwrite different artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def write_text_stable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RunDPreparationError(f"refusing to overwrite different artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def configured_path(path: Path) -> str:
    return DAY12.canonical_project_path(path)


def resolve_project_path(value: str) -> Path:
    return DAY12.resolve_project_path(value)


def default_tokenizer_path() -> Path:
    candidates = (
        REPO_ROOT / "tmp" / "day11-ready" / "models" / "Qwen3-0.6B-Base",
        REPO_ROOT / "day11-ready" / "models" / "Qwen3-0.6B-Base",
    )
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file():
            return candidate
    raise RunDPreparationError(
        "cannot find Qwen3-0.6B-Base tokenizer under the local or stripped bundle"
    )


def load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path.resolve(), local_files_only=True, use_fast=True
    )
    if type(tokenizer).__name__ != "Qwen2TokenizerFast":
        raise RunDPreparationError(f"unexpected tokenizer class: {type(tokenizer).__name__}")
    return tokenizer


def validate_spec(spec: dict[str, Any]) -> dict[str, Fraction]:
    if spec.get("run_id") != "D" or spec.get("mix_name") != "mix_D_format_repaired":
        raise RunDPreparationError("Run D identity changed")
    ratios = {skill: Fraction(spec["target_ratios"][skill]) for skill in SKILLS}
    if sum(ratios.values(), Fraction()) != 1:
        raise RunDPreparationError("target ratios must sum to one")
    total = int(spec["total_supervised_tokens"])
    if any((total * ratio).denominator != 1 for ratio in ratios.values()):
        raise RunDPreparationError("full token budget must divide exactly by every ratio")
    if set(spec["routing_suffixes"]) != set(SKILLS):
        raise RunDPreparationError("routing suffixes must cover exactly four skills")
    return ratios


def normalize_program_argument(value: str) -> str:
    value = value.strip()
    if value == "const_m1":
        return "-1"
    if value.startswith("const_"):
        return value.removeprefix("const_")
    match = re.fullmatch(r"#(\d+)", value)
    if match:
        return f"the result from step {int(match.group(1)) + 1}"
    return value


def render_program_step(operation: str, arguments: list[str]) -> str:
    args = [normalize_program_argument(value) for value in arguments]
    if operation == "subtract" and len(args) == 2:
        return f"Subtract {args[1]} from {args[0]}."
    if operation == "divide" and len(args) == 2:
        return f"Divide {args[0]} by {args[1]}."
    if operation == "add" and len(args) == 2:
        return f"Add {args[0]} and {args[1]}."
    if operation == "multiply" and len(args) == 2:
        return f"Multiply {args[0]} by {args[1]}."
    if operation == "exp" and len(args) == 2:
        return f"Raise {args[0]} to the power of {args[1]}."
    if operation == "greater" and len(args) == 2:
        return f"Compare {args[0]} with {args[1]} to determine which is greater."
    table_operations = {
        "table_average": "Take the average of",
        "table_sum": "Add",
        "table_max": "Take the maximum of",
        "table_min": "Take the minimum of",
    }
    if operation in table_operations and args:
        return f"{table_operations[operation]} {', '.join(args)}."
    raise RunDPreparationError(
        f"unsupported FinQA operation signature: {operation}({', '.join(arguments)})"
    )


def rewrite_finance_response(content: str) -> str:
    prefix = "Calculation: "
    separator = "\nFinal answer: "
    if not content.startswith(prefix) or separator not in content:
        raise RunDPreparationError("finance assistant response violates the parent format")
    program, answer = content[len(prefix) :].split(separator, 1)
    matches = list(re.finditer(r"([a-z_]+)\(([^()]*)\)", program))
    if not matches:
        raise RunDPreparationError(f"cannot parse FinQA program: {program}")
    residue = re.sub(r"([a-z_]+)\(([^()]*)\)", "", program)
    if residue.replace(",", "").strip():
        raise RunDPreparationError(f"FinQA program has unparsed residue: {program}")
    steps = []
    for index, match in enumerate(matches, start=1):
        arguments = [value.strip() for value in match.group(2).split(",")]
        steps.append(f"{index}. {render_program_step(match.group(1), arguments)}")
    return "Reasoning:\n" + "\n".join(steps) + f"\nFinal answer: {answer.strip()}"


def routed_messages(
    row: dict[str, Any], suffixes: dict[str, str]
) -> list[dict[str, str]]:
    skill = row["skill"]
    messages = copy.deepcopy(row["messages"])
    user_index = next(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        None,
    )
    if user_index is None:
        raise RunDPreparationError(f"record has no user message: {row['sample_id']}")
    messages[user_index]["content"] = (
        messages[user_index]["content"].rstrip() + suffixes[skill]
    )
    if skill == "finance":
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ]
        if len(assistant_indices) != 1:
            raise RunDPreparationError("FinQA format repair expects one assistant turn")
        index = assistant_indices[0]
        messages[index]["content"] = rewrite_finance_response(
            messages[index]["content"]
        )
    return messages


def raw_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    return sum(
        len(tokenizer(message["content"], add_special_tokens=False)["input_ids"])
        for message in messages
    )


def transform_row(
    row: dict[str, Any], tokenizer: Any, suffixes: dict[str, str], max_length: int
) -> dict[str, Any]:
    transformed = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in OCCURRENCE_FIELDS
    }
    messages = routed_messages(row, suffixes)
    audit = DAY08.encode_sample(
        tokenizer,
        {"id": row["sample_id"], "messages": messages, "metadata": {}},
        max_length,
    )
    if audit["truncated"]:
        raise RunDPreparationError(f"format repair caused truncation: {row['sample_id']}")
    transform_chain = list(row["transform_chain"])
    transform_chain.append("day12_task_routing_v1")
    if row["skill"] == "finance":
        transform_chain.append("day12_finqa_natural_reasoning_v1")
    transformed.update(
        {
            "messages": messages,
            "content_hash": DAY09_MIX.object_sha256(messages),
            "transform_chain": transform_chain,
            "raw_token_count": raw_token_count(tokenizer, messages),
            "rendered_token_count": audit["original_length"],
            "input_token_count": audit["encoded_length"],
            "supervised_token_count": audit["effective_label_tokens"],
            "truncated": False,
        }
    )
    return transformed


def checkpoint_skill_targets(
    budgets: dict[str, int], ratios: dict[str, str]
) -> dict[str, dict[str, int]]:
    cumulative = {
        checkpoint: DAY12.allocate_skill_targets(int(budgets[checkpoint]), ratios)
        for checkpoint in CHECKPOINTS
    }
    increments: dict[str, dict[str, int]] = {}
    previous = {skill: 0 for skill in SKILLS}
    for checkpoint in CHECKPOINTS:
        increments[checkpoint] = {
            skill: cumulative[checkpoint][skill] - previous[skill]
            for skill in SKILLS
        }
        previous = cumulative[checkpoint]
    return increments


def select_stage_subsets(
    pools: dict[str, list[dict[str, Any]]],
    increments: dict[str, dict[str, int]],
    spec: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    selected = {
        checkpoint: {skill: [] for skill in SKILLS} for checkpoint in CHECKPOINTS
    }
    for skill in SKILLS:
        remaining = DAY09_BUDGET.stable_order(
            pools[skill], spec["selection_version"], spec["mix_name"], skill
        )
        for checkpoint in CHECKPOINTS:
            target = increments[checkpoint][skill]
            subset = DAY09_BUDGET.reconstruct_subset(remaining, target)
            selected[checkpoint][skill] = subset
            chosen = {row["sample_id"] for row in subset}
            remaining = [row for row in remaining if row["sample_id"] not in chosen]
    return selected


def make_occurrence(row: dict[str, Any], mix_name: str) -> dict[str, Any]:
    occurrence = copy.deepcopy(row)
    occurrence.update(
        {
            "canonical_sample_id": row["sample_id"],
            "occurrence_id": f"{mix_name}|{row['sample_id']}|occurrence=1",
            "occurrence_index": 1,
            "sampling_count": 1,
            "supervised_tokens_per_occurrence": int(row["supervised_token_count"]),
        }
    )
    return occurrence


def stage_order_key(
    row: dict[str, Any], selection_version: str, checkpoint: str
) -> tuple[str, str]:
    payload = "\0".join((selection_version, checkpoint, row["sample_id"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), row["sample_id"]


def build_plan(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    parent: dict[str, Any],
    parent_path: Path,
    clean_pool: dict[str, Any],
    clean_pool_path: Path,
    selected: dict[str, dict[str, list[dict[str, Any]]]],
    increments: dict[str, dict[str, int]],
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        stages[checkpoint] = {
            "targets_by_skill": increments[checkpoint],
            "selection_by_skill": {
                skill: {
                    "records": len(selected[checkpoint][skill]),
                    "supervised_tokens": sum(
                        int(row["supervised_token_count"])
                        for row in selected[checkpoint][skill]
                    ),
                    "sample_ids": [
                        row["sample_id"] for row in selected[checkpoint][skill]
                    ],
                }
                for skill in SKILLS
            },
        }
    plan: dict[str, Any] = {
        "schema_version": 1,
        "domain": "day12.format_repair_selection_plan",
        "status": "complete_frozen",
        "run_id": spec["run_id"],
        "mix_name": spec["mix_name"],
        "selection_version": spec["selection_version"],
        "format_repair_version": spec["format_repair_version"],
        "spec": {
            "path": configured_path(spec_path),
            "file_sha256": DAY09_MIX.file_sha256(spec_path),
        },
        "parent_manifest": {
            "path": configured_path(parent_path),
            "file_sha256": DAY09_MIX.file_sha256(parent_path),
            "manifest_hash": parent["header"]["manifest_hash"],
        },
        "clean_pool": {
            "path": configured_path(clean_pool_path),
            "file_sha256": DAY09_MIX.file_sha256(clean_pool_path),
            "manifest_hash": clean_pool["header"]["manifest_hash"],
        },
        "sampling_mode": "without_replacement",
        "stages": stages,
    }
    plan["plan_hash"] = DAY09_MIX.object_sha256(plan)
    return plan


def build_manifest(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
    parent: dict[str, Any],
    parent_path: Path,
    selected: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        stage_rows = [
            row
            for skill in SKILLS
            for row in selected[checkpoint][skill]
        ]
        stage_rows.sort(
            key=lambda row: stage_order_key(
                row, spec["selection_version"], checkpoint
            )
        )
        records.extend(make_occurrence(row, spec["mix_name"]) for row in stage_rows)

    target_ratios = spec["target_ratios"]
    distribution = DAY09_MIX.distribution_by_skill(records, target_ratios)
    parent_header = parent["header"]
    common_invariants = copy.deepcopy(parent_header["common_invariants"])
    common_invariants.update(
        {
            "common_total_supervised_tokens": int(spec["total_supervised_tokens"]),
            "token_budget_plan_hash": plan["plan_hash"],
            "token_budget_plan_file_sha256": DAY09_MIX.file_sha256(plan_path),
            "token_budget_version": spec["selection_version"],
            "format_parent_manifest_hash": parent_header["manifest_hash"],
            "format_parent_manifest_file_sha256": DAY09_MIX.file_sha256(parent_path),
            "format_repair_spec_file_sha256": DAY09_MIX.file_sha256(spec_path),
            "format_repair_version": spec["format_repair_version"],
        }
    )
    header: dict[str, Any] = {
        "status": "complete_with_gate_c_waiver",
        "manifest_name": spec["manifest_name"],
        "manifest_version": "day12_format_repaired_manifest_v1",
        "mix_name": spec["mix_name"],
        "source_manifest_path": configured_path(parent_path),
        "source_plan_path": configured_path(plan_path),
        "common_invariants": common_invariants,
        "mixture_config_sha256": DAY09_MIX.object_sha256(spec),
        "target_ratios": target_ratios,
        "ratio_tolerance_absolute": 0.0,
        "total_token_tolerance_absolute": 0,
        "sampling_mode": "without_replacement",
        "fallback_with_replacement_used": False,
        "maximum_sampling_count": 1,
        "totals": {
            "unique_canonical_examples": len(records),
            "occurrences": len(records),
            "raw_tokens": sum(int(row["raw_token_count"]) for row in records),
            "input_tokens": sum(int(row["input_token_count"]) for row in records),
            "supervised_tokens": sum(
                int(row["supervised_tokens_per_occurrence"]) for row in records
            ),
        },
        "distribution_by_skill": distribution,
        "occurrence_ids_sha256": DAY09_MIX.object_sha256(
            [row["occurrence_id"] for row in records]
        ),
        "gate_c_status": parent_header["gate_c_status"],
        "gate_c_claim_boundary": parent_header["gate_c_claim_boundary"],
        "downstream_training_eligible": True,
        "downstream_eligibility_basis": "parent_gate_c_waiver_plus_deterministic_format_repair",
        "format_repair": {
            "version": spec["format_repair_version"],
            "routing_suffixes_sha256": DAY09_MIX.object_sha256(
                spec["routing_suffixes"]
            ),
            "finance_response_contract": spec["finance_response_contract"],
            "nonfinance_canonical_selection_preserved": True,
            "finance_selection_rematerialized_for_exact_rewritten_token_budget": True,
        },
        "hash_definition": (
            "sha256 of canonical JSON object containing header without "
            "manifest_hash and occurrence records in frozen occurrence order"
        ),
    }
    manifest = {"header": header, "records": records}
    header["manifest_hash"] = DAY09_MIX.object_sha256(manifest)
    return manifest


def render_run_config(
    *, base_config_path: Path, config_path: Path, manifest_path: Path, manifest: dict[str, Any]
) -> str:
    extends = os.path.relpath(base_config_path.resolve(), config_path.resolve().parent)
    header = manifest["header"]
    return (
        f"extends: {extends}\n"
        "run:\n"
        "  id: D\n"
        "data:\n"
        f"  manifest_path: {configured_path(manifest_path)}\n"
        f"  manifest_file_sha256: {DAY09_MIX.file_sha256(manifest_path)}\n"
        f"  manifest_hash: {header['manifest_hash']}\n"
        f"  mix_name: {header['mix_name']}\n"
    )


def validate_manifest(
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    parent: dict[str, Any],
    selected: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, bool]:
    header = manifest["header"]
    records = manifest["records"]
    header_without_hash = {
        key: value for key, value in header.items() if key != "manifest_hash"
    }
    parent_nonfinance = {
        row["canonical_sample_id"]
        for row in parent["records"]
        if row["skill"] != "finance"
    }
    output_nonfinance = {
        row["canonical_sample_id"] for row in records if row["skill"] != "finance"
    }
    finance_contract = spec["finance_response_contract"]
    finance_outputs = [
        row["messages"][-1]["content"]
        for row in records
        if row["skill"] == "finance"
    ]
    planned_ids = {
        checkpoint: {
            make_occurrence(row, spec["mix_name"])["occurrence_id"]
            for skill in SKILLS
            for row in selected[checkpoint][skill]
        }
        for checkpoint in CHECKPOINTS
    }
    total = int(spec["total_supervised_tokens"])
    checks = {
        "manifest_hash_valid": header["manifest_hash"]
        == DAY09_MIX.object_sha256(
            {"header": header_without_hash, "records": records}
        ),
        "occurrence_ids_unique": len({row["occurrence_id"] for row in records})
        == len(records),
        "canonical_ids_unique": len({row["canonical_sample_id"] for row in records})
        == len(records),
        "nonfinance_selection_preserved": output_nonfinance == parent_nonfinance,
        "supervised_total_exact": header["totals"]["supervised_tokens"] == total,
        "slice_ratios_exact": all(
            Fraction(header["distribution_by_skill"][skill]["supervised_tokens"], total)
            == Fraction(spec["target_ratios"][skill])
            for skill in SKILLS
        ),
        "routing_suffixes_present": all(
            any(
                message["role"] == "user"
                and message["content"].endswith(spec["routing_suffixes"][row["skill"]])
                for message in row["messages"]
            )
            for row in records
        ),
        "finance_natural_reasoning_prefix": all(
            output.startswith(finance_contract["prefix"] + "\n")
            for output in finance_outputs
        ),
        "finance_final_answer_present": all(
            "\n" + finance_contract["final_answer_prefix"] + " " in output
            for output in finance_outputs
        ),
        "finance_dsl_removed": all(
            fragment not in output
            for output in finance_outputs
            for fragment in finance_contract["forbidden_fragments"]
        ),
        "no_truncation": not any(row["truncated"] for row in records),
        "maximum_length_respected": max(row["input_token_count"] for row in records)
        <= 2048,
        "planned_ids_cover_manifest": set().union(*planned_ids.values())
        == {row["occurrence_id"] for row in records},
    }
    return checks


def prepare(
    *,
    spec_path: Path = DEFAULT_SPEC,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    plan_path: Path = DEFAULT_PLAN,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    schedule_path: Path = DEFAULT_SCHEDULE,
    summary_path: Path = DEFAULT_SUMMARY,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    ratios = validate_spec(spec)
    parent_path = resolve_project_path(spec["parent_manifest_path"])
    clean_pool_path = resolve_project_path(spec["clean_pool_path"])
    parent = load_json(parent_path)
    clean_pool = load_json(clean_pool_path)
    DAY09_MIX.validate_clean_pool(clean_pool)
    tokenizer = load_tokenizer(tokenizer_path or default_tokenizer_path())
    base_config = DAY12.load_yaml(base_config_path)
    max_length = int(base_config["data"]["max_length"])
    suffixes = spec["routing_suffixes"]

    clean_by_id = {row["sample_id"]: row for row in clean_pool["records"]}
    parent_nonfinance_ids = {
        skill: [
            row["canonical_sample_id"]
            for row in parent["records"]
            if row["skill"] == skill
        ]
        for skill in SKILLS
        if skill != "finance"
    }
    pools: dict[str, list[dict[str, Any]]] = {}
    for skill in SKILLS:
        source_rows = (
            [clean_by_id[sample_id] for sample_id in parent_nonfinance_ids[skill]]
            if skill != "finance"
            else [row for row in clean_pool["records"] if row["skill"] == "finance"]
        )
        pools[skill] = [
            transform_row(row, tokenizer, suffixes, max_length) for row in source_rows
        ]

    full_targets = {
        skill: int(int(spec["total_supervised_tokens"]) * ratios[skill])
        for skill in SKILLS
    }
    for skill in SKILLS:
        available = sum(int(row["supervised_token_count"]) for row in pools[skill])
        if available < full_targets[skill]:
            raise RunDPreparationError(
                f"format-repaired {skill} pool has {available} tokens, below {full_targets[skill]}"
            )
        if skill != "finance" and available != full_targets[skill]:
            raise RunDPreparationError(
                f"routing changed nonfinance supervised-token total for {skill}: {available}"
            )

    budgets = {
        key: int(value)
        for key, value in base_config["training"]["checkpoint_budgets"].items()
    }
    increments = checkpoint_skill_targets(budgets, spec["target_ratios"])
    selected = select_stage_subsets(pools, increments, spec)
    plan = build_plan(
        spec=spec,
        spec_path=spec_path,
        parent=parent,
        parent_path=parent_path,
        clean_pool=clean_pool,
        clean_pool_path=clean_pool_path,
        selected=selected,
        increments=increments,
    )
    write_json_stable(plan_path, plan)
    manifest = build_manifest(
        spec=spec,
        spec_path=spec_path,
        plan=plan,
        plan_path=plan_path,
        parent=parent,
        parent_path=parent_path,
        selected=selected,
    )
    validation = validate_manifest(
        spec=spec, manifest=manifest, parent=parent, selected=selected
    )
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise RunDPreparationError(f"Run D manifest validation failed: {failed}")
    write_json_stable(manifest_path, manifest)

    write_text_stable(
        config_path,
        render_run_config(
            base_config_path=base_config_path,
            config_path=config_path,
            manifest_path=manifest_path,
            manifest=manifest,
        ),
    )
    resolved_config = DAY12.load_resolved_config(config_path)
    verified_manifest = DAY12.verify_mixture(resolved_config)
    schedule = DAY12.build_schedule(resolved_config, verified_manifest)
    DAY12.verify_schedule(schedule)
    planned_stage_ids = {
        checkpoint: {
            f"{spec['mix_name']}|{row['sample_id']}|occurrence=1"
            for skill in SKILLS
            for row in selected[checkpoint][skill]
        }
        for checkpoint in CHECKPOINTS
    }
    for checkpoint in CHECKPOINTS:
        actual_ids = set(schedule["segments"][checkpoint]["occurrence_ids"])
        if actual_ids != planned_stage_ids[checkpoint]:
            raise RunDPreparationError(
                f"generated schedule changed the planned {checkpoint} subset"
            )
    write_json_stable(schedule_path, schedule)

    distribution = manifest["header"]["distribution_by_skill"]
    summary: dict[str, Any] = {
        "status": "day12_run_d_preparation_pass",
        "comparison_boundary": spec["comparison_boundary"],
        "first_gate": spec["first_gate"],
        "target_ratios": spec["target_ratios"],
        "distribution_by_skill": distribution,
        "parent_finance_occurrences": sum(
            row["skill"] == "finance" for row in parent["records"]
        ),
        "repaired_finance_occurrences": distribution["finance"]["occurrences"],
        "total_supervised_tokens": manifest["header"]["totals"][
            "supervised_tokens"
        ],
        "maximum_input_tokens": max(
            row["input_token_count"] for row in manifest["records"]
        ),
        "plan_path": configured_path(plan_path),
        "plan_file_sha256": DAY09_MIX.file_sha256(plan_path),
        "plan_hash": plan["plan_hash"],
        "manifest_path": configured_path(manifest_path),
        "manifest_file_sha256": DAY09_MIX.file_sha256(manifest_path),
        "manifest_hash": manifest["header"]["manifest_hash"],
        "config_path": configured_path(config_path),
        "config_file_sha256": DAY09_MIX.file_sha256(config_path),
        "schedule_path": configured_path(schedule_path),
        "schedule_file_sha256": DAY09_MIX.file_sha256(schedule_path),
        "schedule_hash": schedule["header"]["schedule_hash"],
        "validation_checks": {
            **validation,
            "schedule_total_exact": schedule["segments"]["100_percent"][
                "cumulative_supervised_tokens"
            ]
            == int(spec["total_supervised_tokens"]),
            "schedule_stage_subsets_preserved": True,
        },
    }
    summary["summary_hash"] = DAY09_MIX.object_sha256(summary)
    write_json_stable(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    try:
        summary = prepare(
            spec_path=args.spec,
            base_config_path=args.base_config,
            plan_path=args.plan,
            manifest_path=args.manifest,
            config_path=args.config,
            schedule_path=args.schedule,
            summary_path=args.summary,
            tokenizer_path=args.tokenizer_path,
        )
    except (RunDPreparationError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
