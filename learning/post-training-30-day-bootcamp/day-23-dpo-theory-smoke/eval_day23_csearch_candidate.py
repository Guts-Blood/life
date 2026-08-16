#!/usr/bin/env python3
"""Evaluate and rank the two frozen Day 23 csearch-v0001 candidates.

Only the search30 split is reachable.  The evaluator opens it after both
two-GPU training receipts strictly validate, and it never exposes a dev or
heldout command-line mode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract
import eval_day23_qwen35_preferences as day23_eval
import eval_day23_rsi_candidate as rsi_eval
import run_day23_csearch_candidate as csearch_runner
import run_day23_qwen35_gpu_stage as day23_gpu


EVALUATION_SCHEMA = "day23.csearch_v0001_preference_evaluation"
SELECTION_SCHEMA = "day23.csearch_v0001_search_selection"
CLAIM_SCHEMA = "day23.csearch_v0001_access_claim"


class CSearchEvaluationError(RuntimeError):
    """A csearch checkpoint, access claim, evaluation, or gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSearchEvaluationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and bool(value) and "\x00" not in value,
        f"{label} must be non-empty text",
    )
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return value


def _absolute(value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(_text(value, label)).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute")
    return path.resolve(strict=must_exist)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CSearchEvaluationError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _text(value.get(field), f"{label}.{field}")
    _require(
        len(expected) == 64 and day23_gpu.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def _write_exclusive(path: Path, value: Mapping[str, Any], field: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return day23_gpu.write_sealed_json(path, value, field)
    except (OSError, day23_gpu.Day23GPUStageError) as error:
        raise CSearchEvaluationError(f"cannot seal {path}: {error}") from error


def _producer(campaign: Mapping[str, Any], name: str, actual_path: Path) -> Mapping[str, Any]:
    entry, bound_path = csearch_runner._producer(campaign, name)
    _require(
        bound_path == actual_path.resolve(strict=True),
        f"campaign bound a different {name}",
    )
    return entry


def validate_campaign(path: Path) -> dict[str, Any]:
    context = csearch_runner.validate_campaign_document(
        path, require_search_unopened=False
    )
    _producer(context["campaign"], "evaluator", Path(__file__))
    context["runtime"] = day23_gpu.validate_runtime(context["campaign"])
    return context


def _expected_evaluation_path(
    context: Mapping[str, Any], run_id: str, step: int
) -> Path:
    return (
        context["run_root"]
        / "evidence/evaluations"
        / f"{run_id}-checkpoint-{step}-search.json"
    )


def _selection_path(context: Mapping[str, Any]) -> Path:
    return context["run_root"] / "evidence/selection/search-selection.json"


def _claim_path(context: Mapping[str, Any]) -> Path:
    access = _mapping(context["campaign"].get("access_ledger"), "campaign.access_ledger")
    path = _absolute(access.get("search_claim"), "campaign search claim", must_exist=False)
    expected = context["run_root"] / "evidence/search-selection/search-claim.json"
    _require(path == expected, "search claim escaped its frozen ledger")
    return path


def _candidate_id(campaign: Mapping[str, Any], run_id: str, step: int) -> str:
    registry = campaign.get("candidate_registry")
    _require(isinstance(registry, list), "candidate registry is absent")
    matches = [
        item
        for item in registry
        if isinstance(item, Mapping)
        and item.get("search_run_spec") == run_id
        and item.get("checkpoint_step") == step
    ]
    _require(len(matches) == 1, "candidate registry mapping drifted")
    return _text(matches[0].get("candidate_id"), "candidate ID")


def _checkpoint_from_receipt(
    context: Mapping[str, Any], receipt_path: Path, step: int
) -> dict[str, Any]:
    _require(step == csearch_runner.CANDIDATE_STEP, "only checkpoint step 20 is a candidate")
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt = _load_json(receipt_path, "training receipt")
    receipt_sha = _verify_self(receipt, "receipt_sha256", "training receipt")
    strict = csearch_runner.validate_training_receipt_value(
        context["campaign"], context["campaign_path"], receipt, receipt_path
    )
    run_id = _text(receipt.get("run_id"), "training receipt run_id")
    spec = _mapping(
        _mapping(context["campaign"].get("run_specs"), "campaign.run_specs").get(run_id),
        f"campaign run {run_id}",
    )
    _require(
        strict.get("run_id") == run_id
        and strict.get("spec") == spec
        and receipt.get("status") == "pass"
        and receipt.get("campaign", {}).get("campaign_sha256") == context["campaign_sha256"],
        "strict training receipt projection drifted",
    )
    checkpoint = strict["checkpoints"][0]
    checkpoint_path = _absolute(checkpoint.get("path"), "candidate checkpoint path")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_file_sha256": day23_gpu.file_sha256(receipt_path),
        "receipt_sha256": receipt_sha,
        "run_id": run_id,
        "spec": spec,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
    }


def _search_specs(context: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    runs = _mapping(context["campaign"].get("run_specs"), "campaign.run_specs")
    result = [
        (run_id, _mapping(spec, f"run spec {run_id}"))
        for run_id, spec in sorted(runs.items())
        if isinstance(spec, Mapping) and spec.get("role") == "search_train"
    ]
    _require(len(result) == 2, "campaign search trajectory count drifted")
    return result


def _require_complete_search_grid(context: Mapping[str, Any]) -> None:
    for run_id, spec in _search_specs(context):
        success = _absolute(spec.get("success_receipt"), f"{run_id} success receipt")
        failure = Path(_text(spec.get("failure_receipt"), f"{run_id} failure receipt")).resolve()
        _require(not failure.exists(), f"search grid contains a failed trajectory: {run_id}")
        receipt = _load_json(success, f"{run_id} success receipt")
        _verify_self(receipt, "receipt_sha256", f"{run_id} success receipt")
        strict = csearch_runner.validate_training_receipt_value(
            context["campaign"], context["campaign_path"], receipt, success
        )
        _require(
            receipt.get("schema_name") == csearch_runner.RECEIPT_SCHEMA
            and receipt.get("status") == "pass"
            and receipt.get("run_id") == run_id
            and receipt.get("campaign", {}).get("campaign_sha256") == context["campaign_sha256"]
            and strict.get("run_id") == run_id
            and strict.get("spec") == spec,
            f"search trajectory receipt drifted: {run_id}",
        )


def _claim_document(context: Mapping[str, Any]) -> dict[str, Any]:
    evaluator = Path(__file__).resolve(strict=True)
    candidates = sorted(
        _candidate_id(context["campaign"], run_id, csearch_runner.CANDIDATE_STEP)
        for run_id, _ in _search_specs(context)
    )
    return {
        "schema_name": CLAIM_SCHEMA,
        "schema_version": 1,
        "status": "claimed",
        "split": "search",
        "campaign_sha256": context["campaign_sha256"],
        "authorized_candidates": candidates,
        "max_unseal_count": 1,
        "producer": {
            "path": str(evaluator),
            "file_sha256": day23_gpu.file_sha256(evaluator),
        },
        "dev_consumed": False,
        "heldout_consumed": False,
    }


def _validate_search_claim(context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical claim without opening any search rows."""
    path = _claim_path(context)
    value = _load_json(path, "search access claim")
    claim_sha = _verify_self(value, "claim_sha256", "search access claim")
    projection = dict(value)
    projection.pop("claim_sha256", None)
    _require(projection == _claim_document(context), "search access claim drifted")
    return {
        "path": str(path),
        "file_sha256": day23_gpu.file_sha256(path),
        "claim_sha256": claim_sha,
    }


def _claim_or_validate_search(context: Mapping[str, Any]) -> dict[str, Any]:
    # This check must precede even the first O_EXCL attempt: a partial grid may
    # never consume the one-shot search access lease.
    _require_complete_search_grid(context)
    path = _claim_path(context)
    expected = _claim_document(context)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            day23_gpu.write_sealed_json(path, expected, "claim_sha256")
        except FileExistsError:
            # Two one-GPU evaluators may race after the same complete grid.
            # The losing process validates the winner's exact bytes below.
            pass
        except (OSError, day23_gpu.Day23GPUStageError) as error:
            raise CSearchEvaluationError(f"cannot seal search claim: {error}") from error
    return _validate_search_claim(context)


def _eligibility(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    overall = _mapping(aggregate.get("overall"), "aggregate.overall")
    matched = _mapping(aggregate.get("length_matched"), "aggregate.length_matched")
    pairs = _integer(overall.get("pairs"), "aggregate pair count")
    wins = _integer(overall.get("wins"), "aggregate win count")
    mean = float(overall.get("mean_reward_margin"))
    matched_mean = float(matched.get("mean_reward_margin"))
    _require(
        all(math.isfinite(value) for value in (mean, matched_mean)),
        "aggregate contains non-finite margins",
    )
    return {
        "passed": pairs == 30 and wins >= 20 and mean > 0 and matched_mean > 0,
        "thresholds": {
            "pairs": 30,
            "minimum_positive_pairs": 20,
            "mean_reward_margin": ">0",
            "length_matched_mean_reward_margin": ">0",
        },
        "observed": {
            "pairs": pairs,
            "positive_pairs": wins,
            "mean_reward_margin": mean,
            "length_matched_mean_reward_margin": matched_mean,
        },
    }


def _search_rows(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Mapping[str, Any]]:
    # The reused helper opens only campaign.datasets.search for this literal
    # split; it does not mount the source corpus that contains protected rows.
    return rsi_eval._dataset_rows(context, "search")


def _runtime_context(
    context: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    return rsi_eval._evaluation_runtime_context(context, checkpoint)


def run_evaluation(
    campaign_path: Path,
    training_receipt: Path,
    checkpoint_step: int,
    output: Path,
    device: str,
) -> dict[str, Any]:
    context = validate_campaign(campaign_path)
    checkpoint = _checkpoint_from_receipt(context, training_receipt, checkpoint_step)
    output = output.expanduser().resolve()
    _require(
        output
        == _expected_evaluation_path(context, checkpoint["run_id"], checkpoint_step),
        "evaluation output path drifted",
    )
    _require(not output.exists(), "evaluation output already exists")
    access = _claim_or_validate_search(context)
    rows, source, dataset_entry = _search_rows(context)
    runtime_context = _runtime_context(context, checkpoint)
    runtime = day23_eval.load_policy(
        runtime_context,
        {"checkpoint_path": checkpoint["checkpoint_path"]},
        device_name=device,
    )
    results = day23_eval.evaluate_rows(runtime, rows, source, beta=0.1)
    aggregate = day23_eval.aggregate_results(
        results, split="search", checkpoint_step=checkpoint_step
    )
    gate = _eligibility(aggregate)
    evaluator_process = day23_gpu._process_identity()
    runner_process = _mapping(
        checkpoint["receipt"].get("process_identity"), "training runner process"
    )
    _require(
        (
            evaluator_process.get("boot_id"),
            evaluator_process.get("pid"),
            evaluator_process.get("proc_start_ticks"),
        )
        != (
            runner_process.get("boot_id"),
            runner_process.get("pid"),
            runner_process.get("proc_start_ticks"),
        ),
        "checkpoint evaluation must run in a process distinct from training",
    )
    evaluator = Path(__file__).resolve(strict=True)
    value = {
        "schema_name": EVALUATION_SCHEMA,
        "schema_version": 1,
        "status": "pass" if gate["passed"] else "gate_fail",
        "campaign_sha256": context["campaign_sha256"],
        "run_id": checkpoint["run_id"],
        "run_role": "search_train",
        "producer": {
            "path": str(evaluator),
            "file_sha256": day23_gpu.file_sha256(evaluator),
        },
        "checkpoint_step": checkpoint_step,
        "checkpoint": {
            "path": str(checkpoint["checkpoint_path"]),
            "files_sha256": checkpoint["checkpoint"]["files_sha256"],
            "training_receipt_path": str(checkpoint["receipt_path"]),
            "training_receipt_file_sha256": checkpoint["receipt_file_sha256"],
            "training_receipt_sha256": checkpoint["receipt_sha256"],
        },
        "split": "search",
        "dataset": {
            "file_sha256": dataset_entry.get("file_sha256"),
            "records": len(rows),
            "ordered_pair_ids_sha256": day23_contract.object_sha256(
                [row["pair_id"] for row in rows]
            ),
            "ordered_row_hashes_sha256": day23_contract.object_sha256(
                [row["row_sha256"] for row in rows]
            ),
        },
        "access_claim": access,
        "runtime": runtime["runtime"],
        "process_identity": evaluator_process,
        "runner_process_identity": dict(runner_process),
        "pair_results": results,
        "aggregate": aggregate,
        "eligibility": gate,
        "claim_boundary": {
            "fresh_process_checkpoint_reload_proven": True,
            "adapter_tensor_inventory_and_content_exact": True,
            "generation_performed": False,
            "sandbox_execution_performed": False,
            "dev_consumed": False,
            "heldout_consumed": False,
        },
    }
    _write_exclusive(output, value, "evaluation_sha256")
    return _load_json(output, "sealed evaluation")


def _expected_candidates(
    context: Mapping[str, Any],
) -> list[tuple[str, int, Mapping[str, Any], Path]]:
    result = [
        (
            run_id,
            csearch_runner.CANDIDATE_STEP,
            spec,
            _expected_evaluation_path(context, run_id, csearch_runner.CANDIDATE_STEP),
        )
        for run_id, spec in _search_specs(context)
    ]
    _require(len(result) == 2, "campaign must preregister exactly two candidates")
    return result


def _validate_search_evaluation(
    context: Mapping[str, Any],
    run_id: str,
    step: int,
    spec: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    _require(path == _expected_evaluation_path(context, run_id, step), "evaluation path drifted")
    value = _load_json(path, "search evaluation")
    evaluation_sha = _verify_self(value, "evaluation_sha256", "search evaluation")
    _require(
        value.get("schema_name") == EVALUATION_SCHEMA
        and value.get("schema_version") == 1
        and value.get("campaign_sha256") == context["campaign_sha256"]
        and value.get("run_id") == run_id
        and value.get("run_role") == "search_train"
        and value.get("checkpoint_step") == step
        and value.get("split") == "search"
        and value.get("status") in {"pass", "gate_fail"},
        "search evaluation identity drifted",
    )
    producer = _mapping(value.get("producer"), "search evaluation producer")
    evaluator_bound = _producer(context["campaign"], "evaluator", Path(__file__))
    _require(
        Path(_text(producer.get("path"), "evaluation producer path")).resolve()
        == Path(_text(evaluator_bound.get("path"), "bound evaluator path")).resolve()
        and producer.get("file_sha256") == evaluator_bound.get("file_sha256"),
        "search evaluation producer drifted",
    )
    checkpoint_value = _mapping(value.get("checkpoint"), "search checkpoint")
    strict_checkpoint = _checkpoint_from_receipt(
        context,
        Path(_text(checkpoint_value.get("training_receipt_path"), "training receipt path")),
        step,
    )
    _require(
        strict_checkpoint["run_id"] == run_id
        and checkpoint_value
        == {
            "path": str(strict_checkpoint["checkpoint_path"]),
            "files_sha256": strict_checkpoint["checkpoint"]["files_sha256"],
            "training_receipt_path": str(strict_checkpoint["receipt_path"]),
            "training_receipt_file_sha256": strict_checkpoint["receipt_file_sha256"],
            "training_receipt_sha256": strict_checkpoint["receipt_sha256"],
        },
        "search checkpoint/training binding drifted",
    )
    # Selection and verification enter here without the claim-creation path.
    # Validate the canonical lease before the first search dataset read.
    claim_identity = _validate_search_claim(context)
    rows, _, dataset_entry = _search_rows(context)
    expected_ids = [row["pair_id"] for row in rows]
    expected_dataset = {
        "file_sha256": dataset_entry.get("file_sha256"),
        "records": len(rows),
        "ordered_pair_ids_sha256": day23_contract.object_sha256(expected_ids),
        "ordered_row_hashes_sha256": day23_contract.object_sha256(
            [row["row_sha256"] for row in rows]
        ),
    }
    _require(value.get("dataset") == expected_dataset, "search dataset identity drifted")
    pair_results = value.get("pair_results")
    _require(
        isinstance(pair_results, list)
        and len(pair_results) == 30
        and [item.get("pair_id") for item in pair_results] == expected_ids
        and len(set(expected_ids)) == 30,
        "search pair-result inventory drifted",
    )
    for item in pair_results:
        _require(
            item.get("schema_name") == day23_eval.PAIR_SCHEMA
            and item.get("status") == "pass",
            "search pair-result schema/status drifted",
        )
        _verify_self(item, "pair_evaluation_sha256", "search pair result")
        for key in (
            "policy_chosen_logps",
            "policy_rejected_logps",
            "reference_chosen_logps",
            "reference_rejected_logps",
            "chosen_reward",
            "rejected_reward",
            "reward_margin",
        ):
            number = item.get(key)
            _require(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(float(number)),
                f"search pair-result non-finite {key}",
            )
        _require(
            item.get("pair_accuracy") is (float(item["reward_margin"]) > 0),
            "search pair accuracy drifted",
        )
    aggregate = day23_eval.aggregate_results(
        pair_results, split="search", checkpoint_step=step
    )
    gate = _eligibility(aggregate)
    _require(
        value.get("aggregate") == aggregate
        and value.get("eligibility") == gate
        and value.get("status") == ("pass" if gate["passed"] else "gate_fail"),
        "search aggregate/gate was not exactly recomputed",
    )
    claim_value = _mapping(value.get("access_claim"), "evaluation access claim")
    _require(
        claim_value == claim_identity,
        "evaluation access-claim binding drifted",
    )
    claim_boundary = _mapping(value.get("claim_boundary"), "evaluation claim boundary")
    _require(
        claim_boundary.get("fresh_process_checkpoint_reload_proven") is True
        and claim_boundary.get("adapter_tensor_inventory_and_content_exact") is True
        and claim_boundary.get("generation_performed") is False
        and claim_boundary.get("sandbox_execution_performed") is False
        and claim_boundary.get("dev_consumed") is False
        and claim_boundary.get("heldout_consumed") is False,
        "search evaluation claim boundary drifted",
    )
    runtime_expected = _mapping(
        _mapping(context["campaign"].get("runtime_parse"), "campaign.runtime_parse").get("package_versions"),
        "bound package versions",
    )
    runtime_actual = _mapping(value.get("runtime"), "search evaluation runtime")
    for package, version in runtime_expected.items():
        key = package if package != "ms_swift" else "ms-swift"
        _require(runtime_actual.get(key) == version, f"evaluation runtime drifted: {package}")
    process = _mapping(value.get("process_identity"), "evaluator process")
    runner_process = _mapping(value.get("runner_process_identity"), "runner process")
    _require(
        (process.get("boot_id"), process.get("pid"), process.get("proc_start_ticks"))
        != (
            runner_process.get("boot_id"),
            runner_process.get("pid"),
            runner_process.get("proc_start_ticks"),
        )
        and runner_process == strict_checkpoint["receipt"].get("process_identity"),
        "evaluation fresh-process evidence drifted",
    )
    return {
        "value": value,
        "path": str(path),
        "file_sha256": day23_gpu.file_sha256(path),
        "evaluation_sha256": evaluation_sha,
        "candidate_id": _candidate_id(context["campaign"], run_id, step),
        "run_id": run_id,
        "checkpoint_step": step,
        "learning_rate": float(spec.get("learning_rate")),
        "eligible": bool(gate["passed"]),
        "positive_pairs": int(gate["observed"]["positive_pairs"]),
        "mean_reward_margin": float(gate["observed"]["mean_reward_margin"]),
        "length_matched_mean_reward_margin": float(
            gate["observed"]["length_matched_mean_reward_margin"]
        ),
    }


def _selection_document(
    context: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    eligible = [dict(item) for item in candidates if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            -item["positive_pairs"],
            -item["mean_reward_margin"],
            -item["length_matched_mean_reward_margin"],
            item["checkpoint_step"],
            item["learning_rate"],
            item["candidate_id"],
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "schema_name": SELECTION_SCHEMA,
        "schema_version": 1,
        "status": "selected" if selected else "closed_no_candidate",
        "campaign_sha256": context["campaign_sha256"],
        "candidate_pool_frozen": True,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "ranking_rule": [
            "positive_pairs_desc",
            "mean_reward_margin_desc",
            "length_matched_mean_reward_margin_desc",
            "earlier_checkpoint",
            "lower_learning_rate",
            "candidate_id",
        ],
        "candidates": [
            {key: value for key, value in item.items() if key != "value"}
            for item in candidates
        ],
        "selected_candidate": selected["candidate_id"] if selected else None,
        "selected_run_id": selected["run_id"] if selected else None,
        "selected_checkpoint_step": selected["checkpoint_step"] if selected else None,
        "selected_learning_rate": selected["learning_rate"] if selected else None,
        "selected_evaluation_sha256": selected["evaluation_sha256"] if selected else None,
        "claim_boundary": {
            "dev_consumed": False,
            "heldout_consumed": False,
            "runner_up_fallback_forbidden": True,
        },
    }


def validate_search_selection(
    context: Mapping[str, Any], selection_path: Path
) -> dict[str, Any]:
    path = selection_path.expanduser().resolve(strict=True)
    _require(path == _selection_path(context), "selection receipt path drifted")
    value = _load_json(path, "search selection")
    _verify_self(value, "selection_sha256", "search selection")
    candidates = [
        _validate_search_evaluation(context, run_id, step, spec, eval_path)
        for run_id, step, spec, eval_path in _expected_candidates(context)
    ]
    expected = _selection_document(context, candidates)
    actual = dict(value)
    actual.pop("selection_sha256", None)
    _require(actual == expected, "selection was not exactly recomputed from both evaluations")
    return value


def select_search(campaign_path: Path, output: Path) -> dict[str, Any]:
    context = validate_campaign(campaign_path)
    output = output.expanduser().resolve()
    _require(output == _selection_path(context), "selection output path drifted")
    _require(not output.exists(), "search selection already exists")
    candidates = [
        _validate_search_evaluation(context, run_id, step, spec, eval_path)
        for run_id, step, spec, eval_path in _expected_candidates(context)
    ]
    value = _selection_document(context, candidates)
    _write_exclusive(output, value, "selection_sha256")
    return _load_json(output, "sealed search selection")


def _self_test() -> None:
    aggregate = {
        "overall": {"pairs": 30, "wins": 20, "mean_reward_margin": 0.01},
        "length_matched": {"mean_reward_margin": 0.001},
    }
    _require(_eligibility(aggregate)["passed"], "search threshold self-test failed")
    aggregate["overall"]["wins"] = 19
    _require(not _eligibility(aggregate)["passed"], "19/30 rejection self-test failed")
    aggregate["overall"]["wins"] = 20
    aggregate["length_matched"]["mean_reward_margin"] = 0.0
    _require(not _eligibility(aggregate)["passed"], "zero-margin rejection self-test failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--checkpoint-step", type=int, choices=[csearch_runner.CANDIDATE_STEP])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--select-search", action="store_true")
    parser.add_argument("--verify-selection", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"status": "pass", "scope": "stdlib_self_test"}, sort_keys=True))
        return 0
    _require(args.campaign is not None, "campaign is required")
    try:
        if args.verify_selection:
            _require(args.selection_receipt is not None, "selection receipt is required")
            context = validate_campaign(args.campaign)
            result = validate_search_selection(context, args.selection_receipt)
        elif args.select_search:
            _require(args.output is not None, "selection output is required")
            result = select_search(args.campaign, args.output)
        else:
            _require(
                args.output is not None
                and args.training_receipt is not None
                and args.checkpoint_step == csearch_runner.CANDIDATE_STEP,
                "evaluation requires a training receipt, checkpoint step 20, and output",
            )
            result = run_evaluation(
                args.campaign,
                args.training_receipt,
                args.checkpoint_step,
                args.output,
                args.device,
            )
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error": str(error),
                    "traceback": traceback.format_exception(error)[-6:],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "output": str(args.output) if args.output else str(args.selection_receipt),
                "self_hash": result.get(
                    "evaluation_sha256", result.get("selection_sha256")
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
