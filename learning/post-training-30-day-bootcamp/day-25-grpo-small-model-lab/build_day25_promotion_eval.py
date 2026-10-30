#!/usr/bin/env python3
"""Freeze the Day 25 paired code-eval promotion and confirmation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract
import prepare_day25_qwen35_grpo as data_builder


DEFAULT_SEED = contract.ARTIFACTS / "data/day22-qwen35-mbpp-seed-manifest.json"
DEFAULT_TRAIN = contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-train.jsonl"
DEFAULT_SEARCH = contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-eval.jsonl"
DEFAULT_DATA_MANIFEST = (
    contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-data-manifest.json"
)
DEFAULT_CPU_CONTRACT = (
    contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/cpu-run-contract.json"
)
DEFAULT_CONFIRMATION = (
    contract.ARTIFACTS / "data/day25-qwen35-coding-grpo-confirmation24.jsonl"
)
DEFAULT_OUTPUT = (
    contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/promotion-eval-contract.json"
)
DAY24_VERIFIER = (
    contract.BOOTCAMP_ROOT
    / "day-24-online-rl-dataflow-reward/day24_coding_verifier.py"
)
CONFIRMATION_DOMAINS = {
    "dev": "day25.grpo.promotion.confirmation12.dev.v1",
    "heldout": "day25.grpo.promotion.confirmation12.heldout.v1",
}
CONFIRMATION_PER_SPLIT = 12


class Day25PromotionEvalError(ValueError):
    """The promotion eval cannot be frozen without changing its meaning."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25PromotionEvalError(message)


def _input(path: Path, *, content_field: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": contract.relative_to_bootcamp(path),
        "file_sha256": contract.file_sha256(path),
    }
    if content_field is not None:
        value = contract.load_json(path)
        result["content_sha256"] = contract.verify_seal(value, content_field, path.name)
    return result


def _validate_frozen_rows(path: Path, expected: int, label: str) -> list[dict[str, Any]]:
    rows = contract.load_jsonl(path)
    _require(len(rows) == expected, f"{label} must contain {expected} rows")
    ids: set[str] = set()
    for row in rows:
        contract.verify_seal(row, "row_sha256", f"{label} row")
        family_id = row.get("task_family_id")
        _require(isinstance(family_id, str) and family_id not in ids, f"{label} family drifted")
        ids.add(family_id)
    return rows


def build(
    *,
    seed_path: Path = DEFAULT_SEED,
    train_path: Path = DEFAULT_TRAIN,
    search_path: Path = DEFAULT_SEARCH,
    data_manifest_path: Path = DEFAULT_DATA_MANIFEST,
    cpu_contract_path: Path = DEFAULT_CPU_CONTRACT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = [seed_path, train_path, search_path, data_manifest_path, cpu_contract_path]
    seed_path, train_path, search_path, data_manifest_path, cpu_contract_path = [
        path.resolve() for path in paths
    ]
    seed, seed_rows = data_builder._validate_seed(seed_path)
    train = _validate_frozen_rows(train_path, 32, "Day 25 train")
    search = _validate_frozen_rows(search_path, 40, "Day 25 search eval")
    data_manifest = contract.load_json(data_manifest_path)
    data_hash = contract.verify_seal(
        data_manifest, "manifest_sha256", "Day 25 data manifest"
    )
    cpu = contract.load_json(cpu_contract_path)
    cpu_hash = contract.verify_seal(cpu, "contract_sha256", "Day 25 CPU contract")
    _require(cpu.get("status") == "cpu_ready_gpu_pending", "CPU contract is not ready")
    _require(
        cpu["inputs"]["data_manifest"]["content_sha256"] == data_hash,
        "CPU contract is bound to another data manifest",
    )

    excluded = {
        row["task_family_id"]
        for row in [*train, *search]
    }
    selected: list[Mapping[str, Any]] = []
    reserves: dict[str, list[str]] = {}
    for split in ("dev", "heldout"):
        eligible = [
            row
            for row in seed_rows
            if row.get("split") == split and row["task_family_id"] not in excluded
        ]
        eligible.sort(
            key=lambda row: contract.stable_rank(
                CONFIRMATION_DOMAINS[split], str(row["task_family_id"])
            )
        )
        _require(
            len(eligible) == 13,
            f"expected exactly 13 unused {split} families, got {len(eligible)}",
        )
        selected.extend(eligible[:CONFIRMATION_PER_SPLIT])
        reserves[split] = [str(row["task_family_id"]) for row in eligible[CONFIRMATION_PER_SPLIT:]]
    confirmation = [
        data_builder._compile_row(row, day25_split="confirmation")
        for row in selected
    ]
    confirmation_ids = {row["task_family_id"] for row in confirmation}
    _require(len(confirmation) == 24 and len(confirmation_ids) == 24, "confirmation24 drifted")
    _require(confirmation_ids.isdisjoint(excluded), "confirmation leaked into train/search")

    confirmation_payload = contract.jsonl_bytes(confirmation)
    source_paths = {
        "build_day25_promotion_eval.py": Path(__file__).resolve(),
        "generate_day25_promotion_eval.py": contract.DAY25_DIR / "generate_day25_promotion_eval.py",
        "sandbox_day25_promotion_eval.py": contract.DAY25_DIR / "sandbox_day25_promotion_eval.py",
        "score_day25_promotion_eval.py": contract.DAY25_DIR / "score_day25_promotion_eval.py",
        "test_day25_promotion_eval.py": contract.DAY25_DIR / "test_day25_promotion_eval.py",
        "day24_coding_verifier.py": DAY24_VERIFIER,
    }
    for label, path in source_paths.items():
        _require(path.is_file(), f"promotion source is missing: {label}")

    generation = {
        "mode": "matched_greedy_pass_at_1",
        "do_sample": False,
        "seed": 20260820,
        "max_new_tokens": 512,
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "return_only_indented_continuation": True,
        "same_runtime_processor_template_and_batching_for_both_models": True,
    }
    result_record_schema = {
        "required": [
            "schema_name",
            "schema_version",
            "task_family_id",
            "task_row_sha256",
            "status",
            "finish_reason",
            "format_valid",
            "passed_tests",
            "total_tests",
            "completion_token_count",
            "completion_sha256",
            "evidence_sha256",
            "result_sha256",
        ],
        "status_values": [
            "ok",
            "wrong_answer",
            "runtime_error",
            "compile_error",
            "parse_error",
            "candidate_timeout",
            "truncated",
            "aborted",
            "infra_error",
        ],
        "correct_definition": "status=ok, format_valid=true, passed_tests=total_tests, finish_reason!=length",
    }
    gates = {
        "search40": {
            "minimum_candidate_correct_gain": 3,
            "minimum_paired_net_wins": 3,
            "maximum_regressions": 2,
            "maximum_format_valid_count_loss": 1,
            "maximum_truncation_count_increase": 0,
            "maximum_mean_completion_token_multiplier": 1.50,
            "require_zero_infra_error": True,
        },
        "confirmation24": {
            "minimum_candidate_correct_gain": 2,
            "minimum_paired_net_wins": 2,
            "maximum_regressions": 1,
            "maximum_format_valid_count_loss": 1,
            "maximum_truncation_count_increase": 0,
            "maximum_mean_completion_token_multiplier": 1.50,
            "require_zero_infra_error": True,
        },
    }
    value: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_promotion_eval_contract",
        "schema_version": 1,
        "status": "frozen_before_gpu_eval_outputs",
        "training_trust_root_unchanged": True,
        "cpu_contract_sha256": cpu_hash,
        "implementation_sources": {
            label: contract.file_sha256(path)
            for label, path in sorted(source_paths.items())
        },
        "inputs": {
            "seed_manifest": {
                **_input(seed_path),
                "content_sha256": seed["manifest_sha256"],
            },
            "data_manifest": _input(data_manifest_path, content_field="manifest_sha256"),
            "cpu_contract": _input(cpu_contract_path, content_field="contract_sha256"),
            "train32": _input(train_path),
            "search40": _input(search_path),
        },
        "suites": {
            "search40": {
                "role": "single_use_candidate_screen",
                "path": contract.relative_to_bootcamp(search_path),
                "records": 40,
                "file_sha256": contract.file_sha256(search_path),
                "ordered_family_ids": [row["task_family_id"] for row in search],
                "ordered_row_hashes_sha256": contract.object_sha256(
                    [row["row_sha256"] for row in search]
                ),
            },
            "confirmation24": {
                "role": "single_use_model_output_blind_confirmation",
                "path": contract.relative_to_bootcamp(DEFAULT_CONFIRMATION),
                "records": 24,
                "source_splits": {"day22.dev": 12, "day22.heldout": 12},
                "selection_domains": CONFIRMATION_DOMAINS,
                "file_sha256": hashlib.sha256(confirmation_payload).hexdigest(),
                "ordered_family_ids": [row["task_family_id"] for row in confirmation],
                "ordered_row_hashes_sha256": contract.object_sha256(
                    [row["row_sha256"] for row in confirmation]
                ),
                "quarantined_unused_family_ids": reserves,
            },
        },
        "overlap": {
            "train_search": [],
            "train_confirmation": [],
            "search_confirmation": [],
        },
        "checkpoint_policy": {
            "baseline": {
                "role": "day21_promoted_s1",
                "downstream_key": cpu["parent"]["downstream_key"],
                "checkpoint_id": cpu["parent"]["checkpoint_id"],
                "artifact_sha256": cpu["parent"]["merged_export_artifact_sha256"],
            },
            "candidate": {
                "only_eligible_stage": "g4_bounded_short_run",
                "only_eligible_checkpoint_step": 10,
                "must_start_fresh_from_same_s1": True,
                "g2_and_g3_checkpoints_are_never_candidates": True,
                "intermediate_g4_checkpoint_selection_forbidden": True,
                "early_stopped_or_incomplete_g4_has_no_candidate": True,
            },
        },
        "access_policy": {
            "search40": "open exactly once only after G0-G3 and complete G4 checkpoint-10 pass runtime gates",
            "confirmation24": "open exactly once only after a sealed search40 decision passes every search gate",
            "after_confirmation": "no retraining, checkpoint reselection, prompt change, decoder change, scorer change, or threshold change",
        },
        "generation_contract": {
            **generation,
            "generation_contract_sha256": contract.object_sha256(generation),
        },
        "execution_contract": {
            "sandbox_owner": "day24_coding_verifier.py",
            "day24_verifier_file_sha256": contract.file_sha256(DAY24_VERIFIER),
            "fresh_sandbox_per_completion": True,
            "public_mbpp_test_count_per_task": "as_frozen_in_each_suite_row",
            "challenge_tests_available": False,
            "persistent_infra_invalidates_the_entire_model_suite": True,
        },
        "result_schema": result_record_schema,
        "completion_package_schema": {
            "schema_name": "day25.coding_eval_completions",
            "one_record_per_suite_task_in_frozen_order": True,
            "required_evidence": [
                "message_content",
                "generated_text",
                "prompt_token_ids",
                "response_token_ids",
                "finish_reason",
                "model_identity",
                "GPU binding SHA-256",
            ],
            "sealed_before_sandbox": True,
        },
        "promotion_gates": gates,
        "statistical_diagnostics": {
            "paired_bootstrap_seed": 20260820,
            "paired_bootstrap_resamples": 10000,
            "exact_sign_test": "one_sided_binomial_on_wins_vs_regressions",
            "diagnostics_do_not_override_conjunctive_engineering_gates": True,
        },
        "decision_policy": {
            "search_fail": "closed_no_candidate_confirmation_unopened",
            "search_pass_confirmation_fail": "closed_no_candidate_confirmation_failed",
            "both_pass": "directional_code_candidate_eligible",
            "strong_statistical_gain_claim_allowed": False,
            "reason": "confirmation24 is a small engineering gate and is not globally contamination-free",
            "broader_s2_promotion_allowed": False,
        },
        "required_report_metrics": [
            "baseline_correct",
            "candidate_correct",
            "correct_gain",
            "wins",
            "regressions",
            "ties_pass",
            "ties_fail",
            "paired_net_wins",
            "one_sided_exact_sign_p",
            "paired_delta_bootstrap_95pct",
            "format_valid_counts",
            "truncation_counts",
            "mean_completion_tokens",
            "status_counts",
        ],
        "claim_boundary": {
            "independent_of_day25_optimizer": True,
            "model_outputs_seen_at_freeze_time": False,
            "globally_virgin_or_pretraining_contamination_free": False,
            "promotion_scope": "Day25 directional coding candidate only",
            "capability_gain_claimed_at_freeze_time": False,
        },
    }
    value["eval_contract_sha256"] = contract.object_sha256(value)
    return confirmation, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-output", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("build", "rebuild", "check"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        confirmation, value = build()
        payloads = {
            args.confirmation_output: contract.jsonl_bytes(confirmation),
            args.contract_output: contract.json_bytes(value),
        }
        for path, payload in payloads.items():
            if args.mode == "build":
                contract.write_atomic(path, payload, overwrite=False)
            elif args.mode == "rebuild":
                contract.write_atomic(path, payload, overwrite=True)
            else:
                _require(path.is_file() and path.read_bytes() == payload, f"artifact drifted: {path}")
    except (OSError, contract.Day25ContractError, Day25PromotionEvalError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": value["status"],
                "mode": args.mode,
                "confirmation_records": len(confirmation),
                "eval_contract_sha256": value["eval_contract_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
