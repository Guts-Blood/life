#!/usr/bin/env python3
"""Replay a frozen Day 25 completion package through the Day 24 E2B verifier."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import day25_contract as contract
import score_day25_promotion_eval as promotion
from day25_reward_adapter import _trajectory, day24


DEFAULT_CONTRACT = promotion.DEFAULT_CONTRACT


class Day25PromotionSandboxError(ValueError):
    """A completion suite cannot produce a valid atomic eval result package."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25PromotionSandboxError(message)


def _validate_completions(
    path: Path,
    *,
    value: Mapping[str, Any],
    eval_hash: str,
    suite_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    package = contract.load_json(path.resolve())
    package_hash = contract.verify_seal(package, "completions_sha256", "eval completions")
    _require(package.get("schema_name") == "day25.coding_eval_completions", "wrong completion schema")
    _require(package.get("schema_version") == 1, "wrong completion schema version")
    _require(package.get("eval_contract_sha256") == eval_hash, "completion eval contract drifted")
    _require(package.get("suite_id") == suite_id, "completion suite drifted")
    role = package.get("model_role")
    _require(role in {"s1_parent", "grpo_g4_final"}, "completion model role drifted")
    _require(
        package.get("generation_contract_sha256")
        == value["generation_contract"]["generation_contract_sha256"],
        "completion generation contract drifted",
    )
    promotion._validate_identity(package, str(role), value)
    records = package.get("records")
    _require(isinstance(records, list) and len(records) == len(rows), "completion count drifted")
    result: list[dict[str, Any]] = []
    for ordinal, (task, item) in enumerate(zip(rows, records)):
        _require(isinstance(item, Mapping), "completion row is not an object")
        contract.verify_seal(item, "completion_sha256", "eval completion")
        _require(item.get("schema_name") == "day25.coding_eval_completion", "completion row schema drifted")
        _require(item.get("schema_version") == 1 and item.get("ordinal") == ordinal, "completion ordinal drifted")
        _require(item.get("task_family_id") == task["task_family_id"], "completion task drifted")
        _require(item.get("task_row_sha256") == task["row_sha256"], "completion task hash drifted")
        message = item.get("message_content")
        generated = item.get("generated_text")
        _require(
            isinstance(message, str)
            and item.get("message_content_sha256") == contract.text_sha256(message)
            and isinstance(generated, str)
            and item.get("generated_text_sha256") == contract.text_sha256(generated),
            "completion text hash drifted",
        )
        for name in ("prompt_token_ids", "response_token_ids"):
            tokens = item.get(name)
            _require(
                isinstance(tokens, list)
                and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in tokens)
                and item.get(f"{name}_sha256") == contract.object_sha256(tokens),
                f"{name} drifted",
            )
        _require(isinstance(item.get("finish_reason"), str), "finish reason drifted")
        result.append(dict(item))
    return package, package_hash, result


def _attempts(
    trajectory: Mapping[str, Any],
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for retry in range(2):
        evidence = day24.verify_trajectory(trajectory, executor, replay_ordinal=1)
        evidence["infra_retry_ordinal"] = retry
        if attempts:
            evidence["retry_of_event_id"] = attempts[0]["event_id"]
        evidence["day25_eval_role"] = "promotion_eval"
        evidence = day24.seal(evidence, "evidence_sha256")
        attempts.append(evidence)
        if evidence["status"] != "infra_error":
            break
    return attempts


def score_completions(
    *,
    package: Mapping[str, Any],
    package_hash: str,
    completion_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _require(workers > 0, "workers must be positive")
    trajectories: list[dict[str, Any]] = []
    for index, (task, completion) in enumerate(zip(tasks, completion_rows)):
        metadata = {
            **task,
            "prompt_id": f"day25-eval:{package['suite_id']}:{task['task_family_id']}",
            "request_id": f"day25-eval:{package['model_role']}:{index}",
            "finish_reason": completion["finish_reason"],
            "response_token_ids": completion["response_token_ids"],
        }
        trajectories.append(
            _trajectory(
                index=index,
                group_ordinal=0,
                completion_message=completion["message_content"],
                metadata=metadata,
                run_id=f"day25-eval-{package['suite_id']}-{package['model_role']}",
                policy_version=str(package["model_role"]),
            )
        )
    attempts_by_index: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(trajectories))) as pool:
        futures = {
            pool.submit(_attempts, trajectory, executor): index
            for index, trajectory in enumerate(trajectories)
        }
        for future in as_completed(futures):
            attempts_by_index[futures[future]] = future.result()
    evidence_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    persistent_infra: list[str] = []
    for index, (task, completion, trajectory) in enumerate(
        zip(tasks, completion_rows, trajectories)
    ):
        attempts = attempts_by_index[index]
        terminal = attempts[-1]
        if terminal["status"] == "infra_error":
            persistent_infra.append(str(task["task_family_id"]))
        reward_summary = terminal.get("summary", {}).get("reward", {})
        passed = int(reward_summary.get("passed", 0))
        expected_total = len(task["reward_tests"])
        if terminal["status"] not in {"parse_error", "compile_error", "truncated", "aborted", "infra_error"}:
            _require(reward_summary.get("total") == expected_total, "executed test total drifted")
        evidence_records.append(
            {
                "trajectory": copy.deepcopy(trajectory),
                "attempts": copy.deepcopy(attempts),
                "terminal_evidence_sha256": terminal["evidence_sha256"],
            }
        )
        result: dict[str, Any] = {
            "schema_name": "day25.coding_eval_result",
            "schema_version": 1,
            "task_family_id": task["task_family_id"],
            "task_row_sha256": task["row_sha256"],
            "status": terminal["status"],
            "finish_reason": completion["finish_reason"],
            "format_valid": bool(trajectory["code_artifact"]["format_contract"]["valid"]),
            "passed_tests": passed,
            "total_tests": expected_total,
            "completion_token_count": len(completion["response_token_ids"]),
            "completion_sha256": trajectory["code_artifact"]["completion_sha256"],
            "evidence_sha256": terminal["evidence_sha256"],
        }
        result["result_sha256"] = contract.object_sha256(result)
        result_records.append(result)
    evidence_package: dict[str, Any] = {
        "schema_name": "day25.coding_eval_evidence_package",
        "schema_version": 1,
        "status": "aborted_persistent_infra" if persistent_infra else "complete",
        "eval_contract_sha256": package["eval_contract_sha256"],
        "suite_id": package["suite_id"],
        "model_role": package["model_role"],
        "source_completions_sha256": package_hash,
        "persistent_infra_family_ids": persistent_infra,
        "records": evidence_records,
    }
    evidence_package["evidence_package_sha256"] = contract.object_sha256(evidence_package)
    if persistent_infra:
        return evidence_package, None
    results: dict[str, Any] = {
        "schema_name": "day25.coding_eval_results",
        "schema_version": 1,
        "eval_contract_sha256": package["eval_contract_sha256"],
        "suite_id": package["suite_id"],
        "model_role": package["model_role"],
        "model_identity": copy.deepcopy(package["model_identity"]),
        "generation_contract_sha256": package["generation_contract_sha256"],
        "source_completions_sha256": package_hash,
        "records": result_records,
    }
    results["results_sha256"] = contract.object_sha256(results)
    return evidence_package, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--suite", choices=("search40", "confirmation24"), required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value, eval_hash = promotion._load_eval_contract(args.contract)
        tasks = promotion._suite_rows(value, args.suite)
        package, package_hash, rows = _validate_completions(
            args.completions,
            value=value,
            eval_hash=eval_hash,
            suite_id=args.suite,
            rows=tasks,
        )
        evidence, results = score_completions(
            package=package,
            package_hash=package_hash,
            completion_rows=rows,
            tasks=tasks,
            executor=day24.execute_e2b_request,
            workers=args.workers,
        )
        contract.write_atomic(
            args.evidence_output, contract.json_bytes(evidence), overwrite=False
        )
        if results is None:
            raise Day25PromotionSandboxError(
                "persistent E2B infra invalidated the model suite; no result package was emitted"
            )
        contract.write_atomic(
            args.results_output, contract.json_bytes(results), overwrite=False
        )
    except (
        OSError,
        contract.Day25ContractError,
        promotion.Day25PromotionScoreError,
        Day25PromotionSandboxError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "suite": args.suite,
                "model_role": results["model_role"],
                "records": len(results["records"]),
                "results_sha256": results["results_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
