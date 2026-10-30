#!/usr/bin/env python3
"""Freeze the 32-train/40-eval MBPP family bundle for Day 25 GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract


DAY23_DIR = contract.BOOTCAMP_ROOT / "day-23-dpo-theory-smoke"
if str(DAY23_DIR) not in sys.path:
    sys.path.insert(0, str(DAY23_DIR))

import day23_contract  # noqa: E402


DEFAULT_SEED = contract.ARTIFACTS / "data/day22-qwen35-mbpp-seed-manifest.json"
DEFAULT_PROMOTION = contract.ARTIFACTS / "checkpoints/day21-qwen35-s1-promotion-manifest.json"
DEFAULT_KEY = contract.ARTIFACTS / "checkpoints/day21-qwen35-s1-downstream-key.json"
DEFAULT_EXPORT = contract.ARTIFACTS / "checkpoints/day21-qwen35-s1-merged-export-manifest.json"
DEFAULT_OUTPUT_DIR = contract.ARTIFACTS / "data"
TRAIN_NAME = "day25-qwen35-coding-grpo-train.jsonl"
EVAL_NAME = "day25-qwen35-coding-grpo-eval.jsonl"
MANIFEST_NAME = "day25-qwen35-coding-grpo-data-manifest.json"


def _validate_seed(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = contract.load_json(path)
    contract.verify_seal(value, "manifest_sha256", "Day 22 seed manifest")
    header = value.get("header")
    records = value.get("records")
    contract.require(isinstance(header, dict), "Day 22 seed header is missing")
    contract.require(isinstance(records, list) and len(records) == 330, "Day 22 seed must contain 330 families")
    result: list[dict[str, Any]] = []
    for row in records:
        contract.require(isinstance(row, dict), "Day 22 seed row is not an object")
        contract.verify_seal(row, "record_sha256", "Day 22 seed row")
        family_id = row.get("task_family_id")
        prompt = row.get("prompt")
        tests = row.get("tests")
        contract.require(isinstance(family_id, str) and family_id, "seed family ID is invalid")
        contract.require(
            isinstance(prompt, dict)
            and isinstance(prompt.get("text"), str)
            and prompt.get("sha256") == contract.text_sha256(prompt["text"]),
            f"seed prompt hash drifted: {family_id}",
        )
        contract.require(
            isinstance(tests, dict)
            and isinstance(tests.get("test_list"), list)
            and bool(tests["test_list"])
            and all(isinstance(item, str) and item.strip() for item in tests["test_list"]),
            f"seed reward tests are invalid: {family_id}",
        )
        result.append(dict(row))
    contract.require(
        len({row["task_family_id"] for row in result}) == 330,
        "Day 22 seed contains duplicate family IDs",
    )
    return value, result


def _parent(
    promotion_path: Path, key_path: Path, export_path: Path
) -> dict[str, Any]:
    promotion = contract.load_json(promotion_path)
    downstream_key = contract.load_json(key_path)
    merged_export = contract.load_json(export_path)
    return day23_contract.validate_s1_bundle(
        promotion,
        downstream_key,
        merged_export,
        promotion_file_sha256=contract.file_sha256(promotion_path),
        downstream_key_file_sha256=contract.file_sha256(key_path),
        merged_export_file_sha256=contract.file_sha256(export_path),
    )


def _select(
    records: Sequence[Mapping[str, Any]], split: str, count: int, domain: str
) -> list[Mapping[str, Any]]:
    eligible = [row for row in records if row.get("split") == split]
    eligible.sort(
        key=lambda row: contract.stable_rank(domain, str(row["task_family_id"]))
    )
    contract.require(len(eligible) >= count, f"insufficient {split} families")
    return eligible[:count]


def _compile_row(source: Mapping[str, Any], *, day25_split: str) -> dict[str, Any]:
    prompt = source["prompt"]
    tests = source["tests"]
    task = {
        "task_id": int(source["task_id"]),
        "task_family_id": str(source["task_family_id"]),
        "day22_split": str(source["split"]),
        "day25_split": day25_split,
        "problem": str(source["problem"]["text"]),
        "prompt": str(prompt["text"]),
        "prompt_sha256": str(prompt["sha256"]),
        "code_prefix": str(source["code_prefix"]),
        "entry_point": str(source["entry_point"]),
        "test_setup_code": str(tests.get("test_setup_code") or ""),
        "reward_tests": [str(item) for item in tests["test_list"]],
        "tests_manifest_sha256": str(tests["sha256"]),
        "source_record_sha256": str(source["record_sha256"]),
    }
    task["reward_payload_sha256"] = contract.object_sha256(
        {
            key: task[key]
            for key in (
                "task_family_id",
                "code_prefix",
                "entry_point",
                "test_setup_code",
                "reward_tests",
                "tests_manifest_sha256",
            )
        }
    )
    task["task_manifest_sha256"] = contract.object_sha256(task)
    row: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_prompt",
        "schema_version": 1,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "chat_template_kwargs": {"enable_thinking": False},
        **task,
    }
    row["row_sha256"] = contract.object_sha256(row)
    return row


def build_bundle(
    *,
    seed_path: Path = DEFAULT_SEED,
    promotion_path: Path = DEFAULT_PROMOTION,
    key_path: Path = DEFAULT_KEY,
    export_path: Path = DEFAULT_EXPORT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed_path = seed_path.resolve()
    promotion_path = promotion_path.resolve()
    key_path = key_path.resolve()
    export_path = export_path.resolve()
    seed, records = _validate_seed(seed_path)
    parent = _parent(promotion_path, key_path, export_path)

    train_source = _select(
        records, "train", contract.TRAIN_FAMILIES, contract.TRAIN_SELECTION_DOMAIN
    )
    eval_source = _select(
        records, "dev", 20, contract.EVAL_DEV_SELECTION_DOMAIN
    ) + _select(
        records, "heldout", 20, contract.EVAL_HELDOUT_SELECTION_DOMAIN
    )
    train = [_compile_row(row, day25_split="train") for row in train_source]
    evaluation = [_compile_row(row, day25_split="eval") for row in eval_source]
    train_ids = {row["task_family_id"] for row in train}
    eval_ids = {row["task_family_id"] for row in evaluation}
    contract.require(len(train) == 32 and len(train_ids) == 32, "train32 selection drifted")
    contract.require(len(evaluation) == 40 and len(eval_ids) == 40, "eval40 selection drifted")
    contract.require(train_ids.isdisjoint(eval_ids), "train/eval family leakage")

    train_payload = contract.jsonl_bytes(train)
    eval_payload = contract.jsonl_bytes(evaluation)
    manifest: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_data_manifest",
        "schema_version": 1,
        "status": "cpu_data_ready",
        "parent": parent,
        "selection": {
            "train": {
                "source_split": "day22.train",
                "families": len(train),
                "domain": contract.TRAIN_SELECTION_DOMAIN,
                "ordered_family_ids": [row["task_family_id"] for row in train],
            },
            "eval": {
                "source_splits": {"day22.dev": 20, "day22.heldout": 20},
                "families": len(evaluation),
                "domains": [
                    contract.EVAL_DEV_SELECTION_DOMAIN,
                    contract.EVAL_HELDOUT_SELECTION_DOMAIN,
                ],
                "ordered_family_ids": [row["task_family_id"] for row in evaluation],
            },
            "train_eval_family_overlap": [],
        },
        "reward_contract": {
            "policy": contract.REWARD_POLICY,
            "reward_tests_are_public_mbpp_tests": True,
            "challenge_tests_in_training_reward": False,
            "sandbox_contract_owner": "day24_coding_verifier.py",
        },
        "inputs": {
            "day22_seed_manifest": {
                "path": contract.relative_to_bootcamp(seed_path),
                "file_sha256": contract.file_sha256(seed_path),
                "content_sha256": seed["manifest_sha256"],
            },
            "day21_promotion": {
                "path": contract.relative_to_bootcamp(promotion_path),
                "file_sha256": contract.file_sha256(promotion_path),
            },
            "day21_downstream_key": {
                "path": contract.relative_to_bootcamp(key_path),
                "file_sha256": contract.file_sha256(key_path),
            },
            "day21_merged_export_manifest": {
                "path": contract.relative_to_bootcamp(export_path),
                "file_sha256": contract.file_sha256(export_path),
            },
        },
        "outputs": {
            "train": {
                "path": f"artifacts/data/{TRAIN_NAME}",
                "records": len(train),
                "file_sha256": hashlib.sha256(train_payload).hexdigest(),
                "ordered_row_hashes_sha256": contract.object_sha256(
                    [row["row_sha256"] for row in train]
                ),
            },
            "eval": {
                "path": f"artifacts/data/{EVAL_NAME}",
                "records": len(evaluation),
                "file_sha256": hashlib.sha256(eval_payload).hexdigest(),
                "ordered_row_hashes_sha256": contract.object_sha256(
                    [row["row_sha256"] for row in evaluation]
                ),
            },
        },
        "claim_boundary": {
            "eval_independent_of_day25_optimizer": True,
            "eval_is_new_virgin_blind_pool": False,
            "reason": "Day 22 frozen dev/heldout families are reused; they are disjoint from Day 25 train but not claimed as globally unseen.",
            "day23_checkpoint_used_as_parent": False,
        },
    }
    manifest["manifest_sha256"] = contract.object_sha256(manifest)
    return train, evaluation, manifest


def materialized_bytes(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    return {
        TRAIN_NAME: contract.jsonl_bytes(train),
        EVAL_NAME: contract.jsonl_bytes(evaluation),
        MANIFEST_NAME: contract.json_bytes(manifest),
    }


def apply_bundle(output_dir: Path, payloads: Mapping[str, bytes], *, mode: str) -> None:
    destination = output_dir.resolve()
    contract.require(destination.is_dir(), f"output directory is missing: {destination}")
    for name, payload in payloads.items():
        path = destination / name
        if mode == "build":
            contract.write_atomic(path, payload, overwrite=False)
        elif mode == "rebuild":
            contract.write_atomic(path, payload, overwrite=True)
        else:
            contract.require(path.is_file(), f"frozen output is missing: {path}")
            contract.require(path.read_bytes() == payload, f"frozen output drifted: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "rebuild", "check"), required=True)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--promotion", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--downstream-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--merged-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train, evaluation, manifest = build_bundle(
            seed_path=args.seed,
            promotion_path=args.promotion,
            key_path=args.downstream_key,
            export_path=args.merged_export,
        )
        apply_bundle(
            args.output_dir,
            materialized_bytes(train, evaluation, manifest),
            mode=args.mode,
        )
    except (OSError, contract.Day25ContractError, day23_contract.Day23ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "cpu_data_ready",
                "mode": args.mode,
                "train_families": len(train),
                "eval_families": len(evaluation),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
