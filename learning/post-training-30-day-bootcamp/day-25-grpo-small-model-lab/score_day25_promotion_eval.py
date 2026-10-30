#!/usr/bin/env python3
"""Make the sealed paired Day 25 code-eval promotion decision."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract


DEFAULT_CONTRACT = (
    contract.ARTIFACTS / "configs/day25-qwen35-coding-grpo/promotion-eval-contract.json"
)
HEX = set("0123456789abcdef")


class Day25PromotionScoreError(ValueError):
    """A result package is not comparable under the frozen promotion contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25PromotionScoreError(message)


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and set(value) <= HEX,
        f"{label} is not a SHA-256",
    )
    return value


def _load_eval_contract(path: Path) -> tuple[dict[str, Any], str]:
    value = contract.load_json(path.resolve())
    digest = contract.verify_seal(
        value, "eval_contract_sha256", "Day 25 promotion eval contract"
    )
    _require(value.get("status") == "frozen_before_gpu_eval_outputs", "eval contract status drifted")
    return value, digest


def _suite_rows(value: Mapping[str, Any], suite_id: str) -> list[dict[str, Any]]:
    suites = value.get("suites")
    _require(isinstance(suites, Mapping) and suite_id in suites, f"unknown suite: {suite_id}")
    suite = suites[suite_id]
    path = (contract.BOOTCAMP_ROOT / str(suite["path"])).resolve()
    rows = contract.load_jsonl(path)
    _require(len(rows) == suite["records"], f"{suite_id}: row count drifted")
    _require(contract.file_sha256(path) == suite["file_sha256"], f"{suite_id}: file drifted")
    _require(
        [row["task_family_id"] for row in rows] == suite["ordered_family_ids"],
        f"{suite_id}: family order drifted",
    )
    for row in rows:
        contract.verify_seal(row, "row_sha256", f"{suite_id} row")
    return rows


def _validate_identity(
    package: Mapping[str, Any], role: str, value: Mapping[str, Any]
) -> None:
    identity = package.get("model_identity")
    _require(isinstance(identity, Mapping), f"{role}: model identity is missing")
    _sha(identity.get("artifact_sha256"), f"{role}: model artifact")
    policy = value["checkpoint_policy"]
    if role == "s1_parent":
        expected = policy["baseline"]
        _require(identity.get("downstream_key") == expected["downstream_key"], "S1 downstream key drifted")
        _require(identity.get("checkpoint_id") == expected["checkpoint_id"], "S1 checkpoint drifted")
        _require(identity.get("artifact_sha256") == expected["artifact_sha256"], "S1 artifact drifted")
    else:
        expected = policy["candidate"]
        _require(identity.get("parent_downstream_key") == policy["baseline"]["downstream_key"], "candidate parent drifted")
        _require(identity.get("stage") == expected["only_eligible_stage"], "candidate stage is ineligible")
        _require(
            identity.get("checkpoint_step") == expected["only_eligible_checkpoint_step"],
            "candidate checkpoint step is ineligible",
        )
        _require(identity.get("cpu_contract_sha256") == value["cpu_contract_sha256"], "candidate CPU trust root drifted")
        _sha(identity.get("binding_sha256"), "candidate GPU binding")


def _validate_results(
    path: Path,
    *,
    role: str,
    suite_id: str,
    rows: Sequence[Mapping[str, Any]],
    value: Mapping[str, Any],
    eval_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    package = contract.load_json(path.resolve())
    package_hash = contract.verify_seal(package, "results_sha256", f"{role} results")
    _require(package.get("schema_name") == "day25.coding_eval_results", f"{role}: wrong schema")
    _require(package.get("schema_version") == 1, f"{role}: wrong schema version")
    _require(package.get("eval_contract_sha256") == eval_hash, f"{role}: eval contract drifted")
    _require(package.get("suite_id") == suite_id, f"{role}: suite drifted")
    _require(package.get("model_role") == role, f"{role}: role drifted")
    _require(
        package.get("generation_contract_sha256")
        == value["generation_contract"]["generation_contract_sha256"],
        f"{role}: generation contract drifted",
    )
    _validate_identity(package, role, value)
    records = package.get("records")
    _require(isinstance(records, list) and len(records) == len(rows), f"{role}: result count drifted")
    allowed = set(value["result_schema"]["status_values"])
    validated: list[dict[str, Any]] = []
    for task, result in zip(rows, records):
        _require(isinstance(result, Mapping), f"{role}: non-object result")
        contract.verify_seal(result, "result_sha256", f"{role} result")
        _require(result.get("schema_name") == "day25.coding_eval_result", f"{role}: row schema drifted")
        _require(result.get("schema_version") == 1, f"{role}: row schema version drifted")
        _require(result.get("task_family_id") == task["task_family_id"], f"{role}: task order drifted")
        _require(result.get("task_row_sha256") == task["row_sha256"], f"{role}: task hash drifted")
        _require(result.get("status") in allowed, f"{role}: status is invalid")
        _require(isinstance(result.get("finish_reason"), str), f"{role}: finish reason is invalid")
        _require(isinstance(result.get("format_valid"), bool), f"{role}: format flag is invalid")
        passed = result.get("passed_tests")
        total = result.get("total_tests")
        expected_total = len(task["reward_tests"])
        _require(
            isinstance(passed, int)
            and not isinstance(passed, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and total == expected_total
            and 0 <= passed <= total,
            f"{role}: test counts drifted",
        )
        tokens = result.get("completion_token_count")
        _require(isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0, f"{role}: token count invalid")
        _sha(result.get("completion_sha256"), f"{role}: completion")
        _sha(result.get("evidence_sha256"), f"{role}: evidence")
        validated.append(dict(result))
    return {
        "path": str(path.resolve()),
        "file_sha256": contract.file_sha256(path),
        "content_sha256": package_hash,
        "model_identity": dict(package["model_identity"]),
    }, validated


def _correct(row: Mapping[str, Any]) -> bool:
    return bool(
        row["status"] == "ok"
        and row["format_valid"]
        and row["passed_tests"] == row["total_tests"]
        and row["finish_reason"] != "length"
    )


def _one_sided_sign_p(wins: int, regressions: int) -> float:
    discordant = wins + regressions
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2**discordant)


def _bootstrap_delta(deltas: Sequence[int], *, seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    size = len(deltas)
    values = sorted(
        sum(deltas[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return [values[int(0.025 * (samples - 1))], values[int(0.975 * (samples - 1))]]


def paired_metrics(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    _require(len(baseline) == len(candidate) and baseline, "paired results are empty or misaligned")
    base_correct = [_correct(row) for row in baseline]
    candidate_correct = [_correct(row) for row in candidate]
    wins = sum(not left and right for left, right in zip(base_correct, candidate_correct))
    regressions = sum(left and not right for left, right in zip(base_correct, candidate_correct))
    ties_pass = sum(left and right for left, right in zip(base_correct, candidate_correct))
    ties_fail = len(baseline) - wins - regressions - ties_pass
    deltas = [int(right) - int(left) for left, right in zip(base_correct, candidate_correct)]
    base_tokens = sum(int(row["completion_token_count"]) for row in baseline) / len(baseline)
    candidate_tokens = sum(int(row["completion_token_count"]) for row in candidate) / len(candidate)
    multiplier = (
        candidate_tokens / base_tokens
        if base_tokens
        else 1.0
        if candidate_tokens == 0
        else None
    )
    return {
        "records": len(baseline),
        "baseline_correct": sum(base_correct),
        "candidate_correct": sum(candidate_correct),
        "correct_gain": sum(candidate_correct) - sum(base_correct),
        "wins": wins,
        "regressions": regressions,
        "ties_pass": ties_pass,
        "ties_fail": ties_fail,
        "paired_net_wins": wins - regressions,
        "one_sided_exact_sign_p": _one_sided_sign_p(wins, regressions),
        "paired_delta_bootstrap_95pct": _bootstrap_delta(
            deltas, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "format_valid_counts": {
            "baseline": sum(bool(row["format_valid"]) for row in baseline),
            "candidate": sum(bool(row["format_valid"]) for row in candidate),
        },
        "truncation_counts": {
            "baseline": sum(row["finish_reason"] == "length" or row["status"] == "truncated" for row in baseline),
            "candidate": sum(row["finish_reason"] == "length" or row["status"] == "truncated" for row in candidate),
        },
        "mean_completion_tokens": {
            "baseline": base_tokens,
            "candidate": candidate_tokens,
            "candidate_to_baseline_multiplier": multiplier,
        },
        "status_counts": {
            "baseline": dict(sorted(Counter(str(row["status"]) for row in baseline).items())),
            "candidate": dict(sorted(Counter(str(row["status"]) for row in candidate).items())),
        },
        "infra_error_counts": {
            "baseline": sum(row["status"] == "infra_error" for row in baseline),
            "candidate": sum(row["status"] == "infra_error" for row in candidate),
        },
    }


def _apply_gate(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    checks = {
        "candidate_correct_gain": metrics["correct_gain"] >= gate["minimum_candidate_correct_gain"],
        "paired_net_wins": metrics["paired_net_wins"] >= gate["minimum_paired_net_wins"],
        "regressions": metrics["regressions"] <= gate["maximum_regressions"],
        "format_valid_retention": (
            metrics["format_valid_counts"]["baseline"]
            - metrics["format_valid_counts"]["candidate"]
            <= gate["maximum_format_valid_count_loss"]
        ),
        "truncation_non_regression": (
            metrics["truncation_counts"]["candidate"]
            - metrics["truncation_counts"]["baseline"]
            <= gate["maximum_truncation_count_increase"]
        ),
        "completion_length": (
            isinstance(
                metrics["mean_completion_tokens"]["candidate_to_baseline_multiplier"],
                (int, float),
            )
            and metrics["mean_completion_tokens"]["candidate_to_baseline_multiplier"]
            <= gate["maximum_mean_completion_token_multiplier"]
        ),
        "zero_infra_error": (
            not gate["require_zero_infra_error"]
            or sum(metrics["infra_error_counts"].values()) == 0
        ),
    }
    return checks


def decide(
    *,
    eval_contract_path: Path,
    suite_id: str,
    baseline_results_path: Path,
    candidate_results_path: Path,
    search_decision_path: Path | None = None,
) -> dict[str, Any]:
    value, eval_hash = _load_eval_contract(eval_contract_path)
    _require(suite_id in {"search40", "confirmation24"}, "suite must be search40 or confirmation24")
    if suite_id == "confirmation24":
        _require(search_decision_path is not None, "confirmation requires a sealed search decision")
        search = contract.load_json(search_decision_path.resolve())
        contract.verify_seal(search, "decision_sha256", "search decision")
        _require(
            search.get("eval_contract_sha256") == eval_hash
            and search.get("suite_id") == "search40"
            and search.get("gate_pass") is True
            and search.get("confirmation_authorized") is True,
            "search decision does not authorize confirmation",
        )
    else:
        _require(search_decision_path is None, "search40 must not consume a prior decision")
    rows = _suite_rows(value, suite_id)
    baseline_meta, baseline = _validate_results(
        baseline_results_path,
        role="s1_parent",
        suite_id=suite_id,
        rows=rows,
        value=value,
        eval_hash=eval_hash,
    )
    candidate_meta, candidate = _validate_results(
        candidate_results_path,
        role="grpo_g4_final",
        suite_id=suite_id,
        rows=rows,
        value=value,
        eval_hash=eval_hash,
    )
    diagnostics = value["statistical_diagnostics"]
    metrics = paired_metrics(
        baseline,
        candidate,
        bootstrap_seed=diagnostics["paired_bootstrap_seed"],
        bootstrap_samples=diagnostics["paired_bootstrap_resamples"],
    )
    checks = _apply_gate(metrics, value["promotion_gates"][suite_id])
    gate_pass = all(checks.values())
    if suite_id == "search40":
        status = "search_gate_pass_confirmation_authorized" if gate_pass else "closed_no_candidate_confirmation_unopened"
    else:
        status = "directional_code_candidate_eligible" if gate_pass else "closed_no_candidate_confirmation_failed"
    result: dict[str, Any] = {
        "schema_name": "day25.qwen35_coding_grpo_promotion_decision",
        "schema_version": 1,
        "status": status,
        "suite_id": suite_id,
        "eval_contract_sha256": eval_hash,
        "source_results": {
            "baseline": baseline_meta,
            "candidate": candidate_meta,
        },
        "metrics": metrics,
        "gate_thresholds": value["promotion_gates"][suite_id],
        "gate_checks": checks,
        "gate_pass": gate_pass,
        "confirmation_authorized": suite_id == "search40" and gate_pass,
        "directional_code_candidate_eligible": suite_id == "confirmation24" and gate_pass,
        "strong_statistical_gain_claim_allowed": False,
        "broader_s2_promotion_allowed": False,
    }
    if search_decision_path is not None:
        result["search_decision"] = {
            "path": str(search_decision_path.resolve()),
            "file_sha256": contract.file_sha256(search_decision_path),
        }
    result["decision_sha256"] = contract.object_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--suite", choices=("search40", "confirmation24"), required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--search-decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = decide(
            eval_contract_path=args.contract,
            suite_id=args.suite,
            baseline_results_path=args.baseline_results,
            candidate_results_path=args.candidate_results,
            search_decision_path=args.search_decision,
        )
        contract.write_atomic(args.output, contract.json_bytes(result), overwrite=False)
    except (OSError, contract.Day25ContractError, Day25PromotionScoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "suite": result["suite_id"],
                "gate_pass": result["gate_pass"],
                "decision_sha256": result["decision_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
