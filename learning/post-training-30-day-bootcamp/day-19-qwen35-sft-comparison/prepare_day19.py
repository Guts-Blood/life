#!/usr/bin/env python3
"""Prepare the frozen Day 19 Qwen3.5 SFT comparison datasets and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILLS = ("general", "math", "code", "finance")
DEFAULT_MODEL = Path(
    "/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/"
    "1001bb4d826a52d1f399e183466143f4da7b741b"
)
DEFAULT_DAY18_RUN = Path(
    "/root/autodl-tmp/runs/day18-qwen35-20260808T073811Z"
)
RECIPES = {
    "baseline-a": {
        "historical_run_id": "A",
        "name": "mix_A_balanced",
        "source": "artifacts/data/day09-mix-A-balanced.json",
        "source_sha256": "cb0d4819d383e4a22fd2c389c27c9cf058a20287c08e0fa5d924703f0af77492",
        "historical_config": "artifacts/configs/day12-controlled-sft-A.yaml",
        "ratios": {"general": 1 / 4, "math": 1 / 4, "code": 1 / 4, "finance": 1 / 4},
        "ratio_labels": {skill: "1/4" for skill in SKILLS},
    },
    "baseline-b": {
        "historical_run_id": "B",
        "name": "mix_B_targeted",
        "source": "artifacts/data/day09-mix-B-targeted.json",
        "source_sha256": "d87e32b7925080c4e1abc6676354cb35e60af3f2311b969d6441a8aacc319837",
        "historical_config": "artifacts/configs/day12-controlled-sft-B.yaml",
        "ratios": {"general": 1 / 8, "math": 1 / 8, "code": 1 / 2, "finance": 1 / 4},
        "ratio_labels": {"general": "1/8", "math": "1/8", "code": "1/2", "finance": "1/4"},
    },
    "best-e": {
        "historical_run_id": "E",
        "name": "mix_D_format_repaired",
        "source": "artifacts/data/day12-mix-D-format-repaired.json",
        "source_sha256": "e7d7c522c240aa218d7a86e0466bf48dfd2f63473023d25e1e249fed168430ba",
        "historical_config": "artifacts/configs/day12-controlled-sft-E.yaml",
        "ratios": {"general": 7 / 24, "math": 7 / 24, "code": 7 / 24, "finance": 1 / 8},
        "ratio_labels": {"general": "7/24", "math": "7/24", "code": "7/24", "finance": "1/8"},
        "format_repair_spec": "artifacts/configs/day12-format-repair-D-spec.json",
        "format_repair_plan": "artifacts/data/day12-format-repair-D-plan.json",
    },
}


class Day19PreparationError(ValueError):
    """A frozen input or Day 19 preparation invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day19PreparationError(f"required file is missing: {path}")
    return sha256_bytes(path.read_bytes())


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day19PreparationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day19PreparationError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        raise Day19PreparationError(f"refusing to overwrite existing artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Day19PreparationError(f"refusing to overwrite existing artifact: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def exact_subset_indices(weights: list[int], target: int) -> set[int]:
    """Return a deterministic exact subset using the audited Day 12 bitset method."""
    if target < 0:
        raise Day19PreparationError("subset target must be non-negative")
    reachable = 1
    snapshots: list[int] = []
    mask = (1 << (target + 1)) - 1
    for weight in weights:
        if not isinstance(weight, int) or weight <= 0:
            raise Day19PreparationError("supervised-token weights must be positive integers")
        snapshots.append(reachable)
        reachable = (reachable | (reachable << weight)) & mask
    if not (reachable >> target) & 1:
        raise Day19PreparationError(f"cannot construct exact Qwen3.5 token subset: {target}")
    chosen: set[int] = set()
    cursor = target
    for index in range(len(weights) - 1, -1, -1):
        if (snapshots[index] >> cursor) & 1:
            continue
        chosen.add(index)
        cursor -= weights[index]
    if cursor != 0 or sum(weights[index] for index in chosen) != target:
        raise Day19PreparationError("exact Qwen3.5 subset reconstruction failed")
    return chosen


def validate_messages(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise Day19PreparationError(f"invalid messages: {record.get('occurrence_id')}")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise Day19PreparationError(f"invalid message role: {record.get('occurrence_id')}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise Day19PreparationError(f"empty message content: {record.get('occurrence_id')}")
    if messages[-1].get("role") != "assistant":
        raise Day19PreparationError(f"last message is not assistant: {record.get('occurrence_id')}")
    rendered = json.dumps(messages, ensure_ascii=False).lower()
    if any(marker in rendered for marker in ("<image>", "<video>", "<audio>")):
        raise Day19PreparationError(f"non-text marker in record: {record.get('occurrence_id')}")


def eval_identity_sets(eval_manifest: dict[str, Any]) -> dict[str, set[str]]:
    records = eval_manifest.get("records")
    if not isinstance(records, list):
        raise Day19PreparationError("Day 10 eval manifest has no records")
    dev = [row for row in records if row.get("evaluation_split") == "dev"]
    if len(dev) != 112 or Counter(row.get("slice") for row in dev) != Counter({skill: 28 for skill in SKILLS}):
        raise Day19PreparationError("Day 10 dev split is not the frozen 28x4 suite")
    return {
        "sample_ids": {str(row.get("sample_id")) for row in dev},
        "parent_ids": {str(row.get("source_lineage", {}).get("parent_id")) for row in dev},
        "content_hashes": {str(row.get("source_lineage", {}).get("candidate_content_hash")) for row in dev},
        "prompt_hashes": {
            sha256_bytes(str(row.get(field, "")).encode("utf-8"))
            for row in dev
            for field in ("adapted_prompt", "raw_prompt")
        },
    }


def target_by_skill(recipe: dict[str, Any], target_tokens: int) -> dict[str, int]:
    targets = {skill: round(target_tokens * recipe["ratios"][skill]) for skill in SKILLS}
    if sum(targets.values()) != target_tokens:
        raise Day19PreparationError("target token count is incompatible with recipe ratios")
    return targets


def prepare_recipe(
    *,
    slug: str,
    recipe: dict[str, Any],
    bootcamp_root: Path,
    run_root: Path,
    template: Any,
    target_tokens: int,
    max_length: int,
    eval_ids: dict[str, set[str]],
) -> dict[str, Any]:
    source = bootcamp_root / recipe["source"]
    if file_sha256(source) != recipe["source_sha256"]:
        raise Day19PreparationError(f"historical source hash drift: {source}")
    source_manifest = load_json(source)
    records = source_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise Day19PreparationError(f"source manifest has no records: {source}")

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        validate_messages(record)
        skill = record.get("skill")
        if skill not in SKILLS:
            raise Day19PreparationError(f"unknown skill in {source}: {skill!r}")
        encoded = template.encode(
            __import__("swift.template", fromlist=["TemplateInputs"]).TemplateInputs.from_dict(
                {"messages": record["messages"]}
            ),
            return_length=True,
        )
        input_ids = encoded.get("input_ids") or []
        labels = encoded.get("labels") or []
        if not input_ids or len(input_ids) != len(labels):
            raise Day19PreparationError(f"invalid Qwen3.5 encoding: {record.get('occurrence_id')}")
        supervised_tokens = sum(label != -100 for label in labels[1:])
        if supervised_tokens <= 0:
            raise Day19PreparationError(f"zero supervised tokens: {record.get('occurrence_id')}")
        if len(input_ids) > max_length:
            raise Day19PreparationError(f"unexpected truncation candidate: {record.get('occurrence_id')}")
        pools[skill].append(
            {
                "record": record,
                "qwen35_input_tokens": len(input_ids),
                "qwen35_supervised_tokens": supervised_tokens,
            }
        )

    targets = target_by_skill(recipe, target_tokens)
    selected_ids: set[str] = set()
    for skill in SKILLS:
        indices = exact_subset_indices(
            [item["qwen35_supervised_tokens"] for item in pools[skill]], targets[skill]
        )
        selected_ids.update(pools[skill][index]["record"]["occurrence_id"] for index in indices)

    encoded_by_id = {
        item["record"]["occurrence_id"]: item
        for skill in SKILLS
        for item in pools[skill]
    }
    selected = [
        encoded_by_id[record["occurrence_id"]]
        for record in records
        if record["occurrence_id"] in selected_ids
    ]
    if len(selected) != len(selected_ids):
        raise Day19PreparationError(f"selected occurrence ID mismatch for {slug}")
    if sum(item["qwen35_supervised_tokens"] for item in selected) != target_tokens:
        raise Day19PreparationError(f"Qwen3.5 token budget mismatch for {slug}")

    occurrence_ids = [item["record"]["occurrence_id"] for item in selected]
    content_hashes = [item["record"]["content_hash"] for item in selected]
    canonical_ids = [item["record"].get("canonical_sample_id", "") for item in selected]
    prompt_hashes = [
        sha256_bytes(
            "\n".join(
                message["content"]
                for message in item["record"]["messages"]
                if message["role"] == "user"
            ).encode("utf-8")
        )
        for item in selected
    ]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise Day19PreparationError(f"duplicate occurrences in {slug}")
    if len(content_hashes) != len(set(content_hashes)):
        raise Day19PreparationError(f"exact duplicate content in {slug}")
    leakage = {
        "canonical_id_matches": sorted(set(canonical_ids) & (eval_ids["sample_ids"] | eval_ids["parent_ids"])),
        "content_hash_matches": sorted(set(content_hashes) & eval_ids["content_hashes"]),
        "prompt_hash_matches": sorted(set(prompt_hashes) & eval_ids["prompt_hashes"]),
    }
    if any(leakage.values()):
        raise Day19PreparationError(f"train/dev leakage found for {slug}: {leakage}")

    data_rows = [{"messages": item["record"]["messages"]} for item in selected]
    dataset_path = run_root / "data" / f"{slug}.jsonl"
    write_jsonl(dataset_path, data_rows)
    tokens_by_skill = {
        skill: sum(
            item["qwen35_supervised_tokens"]
            for item in selected
            if item["record"]["skill"] == skill
        )
        for skill in SKILLS
    }
    records_by_skill = Counter(item["record"]["skill"] for item in selected)
    lengths = [item["qwen35_input_tokens"] for item in selected]
    supervised = [item["qwen35_supervised_tokens"] for item in selected]
    manifest = {
        "schema_version": 1,
        "domain": "day19.qwen35_data_manifest",
        "status": "pass",
        "created_at_utc": utc_now(),
        "recipe": slug,
        "historical_identity": {
            key: recipe[key]
            for key in recipe
            if key not in {"ratios"}
        },
        "source": {
            "path": str(source),
            "file_sha256": file_sha256(source),
            "source_manifest_hash": source_manifest.get("header", {}).get("manifest_hash"),
        },
        "selection": {
            "algorithm": "per_skill_exact_subset_bitset_v1_retokenized_qwen35",
            "target_supervised_tokens": target_tokens,
            "target_ratios": recipe["ratio_labels"],
            "target_tokens_by_skill": targets,
            "ordered_occurrence_ids": occurrence_ids,
            "selection_sha256": object_sha256(occurrence_ids),
        },
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": file_sha256(dataset_path),
            "records": len(selected),
            "records_by_skill": dict(records_by_skill),
            "supervised_tokens": sum(supervised),
            "supervised_tokens_by_skill": tokens_by_skill,
            "input_tokens": sum(lengths),
            "length": {"minimum": min(lengths), "maximum": max(lengths), "mean": sum(lengths) / len(lengths)},
            "supervised_length": {
                "minimum": min(supervised),
                "maximum": max(supervised),
                "mean": sum(supervised) / len(supervised),
            },
        },
        "quality": {
            "format_valid_records": len(selected),
            "empty_records": 0,
            "duplicate_occurrences": 0,
            "duplicate_content": 0,
            "truncated_records": 0,
            "max_length": max_length,
            "train_dev_leakage": leakage,
            "historical_overlap_status": "no_unhandled_matches_under_frozen_day09_matcher",
        },
        "transform_chains": dict(
            Counter(
                " -> ".join(item["record"].get("transform_chain", []))
                for item in selected
            )
        ),
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    manifest_path = run_root / "data-manifests" / f"{slug}.json"
    write_json(manifest_path, manifest)
    return {
        "recipe": slug,
        "data_manifest": str(manifest_path),
        "data_manifest_file_sha256": file_sha256(manifest_path),
        "dataset": str(dataset_path),
        "dataset_file_sha256": file_sha256(dataset_path),
        "records": len(selected),
        "train_iters": len(selected),
        "supervised_tokens": target_tokens,
        "tokens_by_skill": tokens_by_skill,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bootcamp-root", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--day18-run", type=Path, default=DEFAULT_DAY18_RUN)
    parser.add_argument("--target-tokens", type=int, default=64608)
    parser.add_argument("--max-length", type=int, default=2304)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    bootcamp_root = args.bootcamp_root.resolve()
    if not (run_root / ".day19-run-root").is_file():
        raise Day19PreparationError("run root marker is missing")
    if (run_root / "DAY19-MANIFEST.json").exists():
        raise Day19PreparationError("Day 19 manifest already exists")
    if args.target_tokens != 64608:
        raise Day19PreparationError("Day 19 target token budget is frozen at 64,608")
    if args.max_length < 2080:
        raise Day19PreparationError("max_length would truncate the audited Best-E selection")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    os.environ.setdefault("USE_MCORE_GDN", "1")
    from swift import get_model_processor, get_template

    _, processor = get_model_processor(
        str(args.model),
        model_type="qwen3_5",
        load_model=False,
        use_hf=True,
        download_model=False,
    )
    template = get_template(
        processor,
        max_length=args.max_length,
        truncation_strategy="raise",
        padding_free=False,
        loss_scale="default+ignore_empty_think",
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    template.set_mode("train")

    eval_manifest_path = bootcamp_root / "artifacts/eval/day10-frozen-eval-manifest.json"
    eval_manifest = load_json(eval_manifest_path)
    eval_ids = eval_identity_sets(eval_manifest)
    matrix = []
    for slug, recipe in RECIPES.items():
        prepared = prepare_recipe(
            slug=slug,
            recipe=recipe,
            bootcamp_root=bootcamp_root,
            run_root=run_root,
            template=template,
            target_tokens=args.target_tokens,
            max_length=args.max_length,
            eval_ids=eval_ids,
        )
        config = {
            "schema_version": 1,
            "domain": "day19.qwen35_training_config",
            "recipe": slug,
            "model": {
                "hf_snapshot": str(args.model.resolve()),
                "hf_snapshot_revision": args.model.name,
                "mcore_start": str((args.day18_run / "c0-mcore-tp1").resolve()),
                "day18_pass": str((args.day18_run / "DAY18-PASS.json").resolve()),
            },
            "data": prepared,
            "training": {
                "seed": 20260807,
                "data_seed": 20260807,
                "tuner_type": "full",
                "torch_dtype": "bfloat16",
                "tensor_model_parallel_size": 2,
                "data_parallel_size": 1,
                "micro_batch_size": 1,
                "global_batch_size": 1,
                "train_iters": prepared["train_iters"],
                "max_length": args.max_length,
                "packing": False,
                "padding_free": False,
                "learning_rate": 1e-5,
                "lr_decay_style": "constant",
                "lr_warmup_fraction": 0.05,
                "weight_decay": 0.1,
                "adam_betas": [0.9, 0.95],
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0,
                "mtp_num_layers": 1,
                "mtp_loss_scaling_factor": 0.1,
                "freeze_vit": True,
                "freeze_aligner": True,
                "checkpoint_policy": "one_final_model_only_no_optimizer_no_rng",
            },
        }
        config["config_sha256"] = object_sha256(config)
        config_path = run_root / "configs" / f"{slug}.json"
        write_json(config_path, config)
        prepared["config"] = str(config_path)
        prepared["config_file_sha256"] = file_sha256(config_path)
        matrix.append(prepared)

    history_files = [
        "artifacts/reports/day12-recovery-final-summary.json",
        "artifacts/reports/day12-recovery-final-retrospective.md",
        "artifacts/reports/day12-recovery-rollout-trace.json",
        "artifacts/reports/day12-rollout-E-outcome.json",
        "artifacts/reports/day12-cloud-dev-selection-outcome.json",
    ]
    manifest = {
        "schema_version": 1,
        "domain": "day19.qwen35_experiment_manifest",
        "status": "prepared_training_not_started",
        "created_at_utc": utc_now(),
        "run_root": str(run_root),
        "objective": "Migrate the historical A/B SFT comparison and best E data processing to a controlled Qwen3.5-4B comparison.",
        "history_selection": {
            "historical_model": "Qwen/Qwen3-0.6B-Base",
            "rollouts": ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
            "accepted_rollout": None,
            "best_code_rollout": "E",
            "best_code_score": 9,
            "best_e_total": 19,
            "best_e_math": 4,
            "claim_boundary": "E was the retained best-code rollout but failed the historical math gate; Day 19 is a new migration experiment, not retroactive promotion.",
            "evidence": [
                {
                    "path": str((bootcamp_root / relative).resolve()),
                    "file_sha256": file_sha256(bootcamp_root / relative),
                }
                for relative in history_files
            ],
        },
        "shared_contract": {
            "same_start_model": str((args.day18_run / "c0-mcore-tp1").resolve()),
            "same_actual_supervised_tokens": args.target_tokens,
            "same_seed": 20260807,
            "same_learning_rate": 1e-5,
            "same_context_length": args.max_length,
            "same_training_entrypoint": "day18_megatron_sft.py",
            "same_eval_manifest": str(eval_manifest_path.resolve()),
            "eval_manifest_file_sha256": file_sha256(eval_manifest_path),
            "dev_only": True,
            "frozen_test_consumed": False,
        },
        "matrix": matrix,
        "checkpoint_retention": {
            "final_count": 3,
            "recipes": list(RECIPES),
            "format": "Megatron distributed model-only",
            "intermediate_checkpoints": 0,
            "temporary_hf_exports": "delete only after per-run load/eval evidence is durable",
        },
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    write_json(run_root / "DAY19-MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
