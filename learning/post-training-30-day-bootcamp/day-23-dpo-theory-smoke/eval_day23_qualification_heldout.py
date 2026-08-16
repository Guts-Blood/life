#!/usr/bin/env python3
"""One-shot heldout29 evaluator for the authorized qual-v0001 finalist.

There is no split selector.  The executable validates the append-only
authorization and every training/dev/full112/E2B dependency, fresh-reloads
checkpoint-20, creates the inherited global heldout lease with O_EXCL, and
only then opens heldout29.  Any error after the claim is a sealed terminal
failure with no retry or fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import control_day23_qualification_closeout as closeout
import day23_contract
import eval_day23_qwen35_preferences as preference_eval
import run_day23_qwen35_gpu_stage as gpu_stage


GOAL_ID = closeout.GOAL_ID
VERSION_ID = closeout.VERSION_ID
CAMPAIGN_ID = closeout.CAMPAIGN_ID
RUN_ID = closeout.RUN_ID
CHECKPOINT_STEP = closeout.CHECKPOINT_STEP
HELDOUT_RECORDS = closeout.HELDOUT_RECORDS
MINIMUM_POSITIVE_PAIRS = closeout.HELDOUT_MINIMUM_POSITIVE_PAIRS
AUTHORIZATION_SCHEMA = closeout.AUTHORIZATION_SCHEMA
CLAIM_SCHEMA = closeout.HELDOUT_CLAIM_SCHEMA
EVALUATION_SCHEMA = closeout.HELDOUT_EVALUATION_SCHEMA
FAILURE_SCHEMA = closeout.HELDOUT_FAILURE_SCHEMA


class QualificationHeldoutError(RuntimeError):
    """An authorization, immutable evidence, lease, or gate invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationHeldoutError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def text(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and bool(value) and "\x00" not in value,
        f"{label} must be non-empty text",
    )
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationHeldoutError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def write_sealed(path: Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    require(path.is_absolute(), f"sealed output path must be absolute: {path}")
    require(path.parent.is_dir(), f"sealed output parent is missing: {path.parent}")
    require(not path.exists() and not path.is_symlink(), f"sealed output already exists: {path}")
    try:
        gpu_stage.write_sealed_json(path, value, field)
    except (OSError, gpu_stage.Day23GPUStageError) as error:
        raise QualificationHeldoutError(f"cannot seal {path}: {error}") from error
    return load_json(path, f"sealed {path.name}")


def _process_key(value: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return value.get("boot_id"), value.get("pid"), value.get("proc_start_ticks")


def _runtime_and_checkpoint(state: Mapping[str, Any], device: str) -> dict[str, Any]:
    context = mapping(state.get("context"), "authorization campaign context")
    checkpoint = mapping(context.get("checkpoint"), "authorization checkpoint")
    checkpoint_path = Path(text(checkpoint.get("path"), "checkpoint path")).resolve(strict=True)
    actual = gpu_stage.checkpoint_manifest(checkpoint_path, CHECKPOINT_STEP)
    require(actual == checkpoint, "checkpoint-20 bytes drifted after authorization")
    spec = mapping(context.get("spec"), "qualification run spec")
    runtime_context = closeout.dev_eval._runtime_context(
        context,
        {
            "spec": spec,
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
        },
    )
    runtime = preference_eval.load_policy(
        runtime_context, {"checkpoint_path": checkpoint_path}, device_name=device
    )
    evaluator_process = gpu_stage._process_identity()
    training_receipt = mapping(context.get("training_receipt"), "training receipt")
    training_process = mapping(
        training_receipt.get("process_identity"), "training process identity"
    )
    dev_value = mapping(mapping(state.get("dev"), "dev evidence").get("value"), "dev evaluation")
    dev_process = mapping(dev_value.get("process_identity"), "dev evaluator process identity")
    require(
        len({_process_key(evaluator_process), _process_key(training_process), _process_key(dev_process)}) == 3,
        "heldout checkpoint reload must be distinct from training and dev processes",
    )
    return {
        "runtime": runtime,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "evaluator_process": evaluator_process,
        "training_process": dict(training_process),
        "dev_process": dict(dev_process),
    }


def _claim_document(state: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    authorization = mapping(state.get("authorization"), "heldout authorization")
    return {
        "schema_name": CLAIM_SCHEMA,
        "schema_version": 1,
        "status": "claimed",
        "split": "heldout",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "authorization": {
            "path": str(state["authorization_path"]),
            "file_sha256": state["authorization_file_sha256"],
            "authorization_sha256": state["authorization_sha256"],
        },
        "authorization_sha256": state["authorization_sha256"],
        "campaign": authorization["campaign"],
        "training_receipt": authorization["training_receipt"],
        "checkpoint": authorization["checkpoint"],
        "dev_evaluation": authorization["dev_evaluation"],
        "full112_guardrail": authorization["full112_guardrail"],
        "e2b_guardrail": authorization["e2b_guardrail"],
        "heldout_dataset": authorization["heldout_dataset"],
        "heldout_gate": authorization["heldout_gate"],
        "fresh_process_identity": runtime["evaluator_process"],
        "max_global_claims": 1,
        "max_candidate_evaluations": 1,
        "retry_allowed": False,
        "fallback_allowed": False,
        "producer": authorization["heldout_evaluator"],
    }


def _claim_once(state: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    claim_path: Path = state["authority"]["heldout_claim_path"]
    require(
        claim_path.parent.is_dir()
        and not claim_path.exists()
        and not claim_path.is_symlink(),
        "global heldout lease is not fresh",
    )
    claim = write_sealed(claim_path, _claim_document(state, runtime), "claim_sha256")
    return {
        "value": claim,
        "identity": {
            "path": str(claim_path),
            "file_sha256": gpu_stage.file_sha256(claim_path),
            "claim_sha256": claim["claim_sha256"],
        },
    }


def _rows_after_claim(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Path]:
    context = mapping(state.get("context"), "authorization campaign context")
    campaign = mapping(context.get("campaign"), "qualification campaign")
    bootcamp = Path(text(campaign.get("bootcamp_root"), "campaign bootcamp root")).resolve(strict=True)
    identity = mapping(state["authorization"].get("heldout_dataset"), "heldout identity")
    relative = Path(text(identity.get("path"), "heldout dataset path"))
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "heldout dataset path is not canonical",
    )
    path = (bootcamp / relative).resolve(strict=True)
    require(path.is_relative_to(bootcamp), "heldout dataset escaped bootcamp root")
    require(gpu_stage.file_sha256(path) == identity.get("file_sha256"), "heldout file hash drifted")
    if "bytes" in identity:
        require(path.stat().st_size == identity.get("bytes"), "heldout byte count drifted")
    rows = closeout.dev_eval._load_jsonl(path, "heldout29 dataset")
    require(len(rows) == HELDOUT_RECORDS, "heldout29 record count drifted")
    try:
        summary = day23_contract.split_summary(rows)
    except day23_contract.Day23ContractError as error:
        raise QualificationHeldoutError(f"heldout29 compiled rows failed: {error}") from error
    expected = {
        "records": HELDOUT_RECORDS,
        "ordered_pair_ids_sha256": identity.get("ordered_pair_ids_sha256"),
        "ordered_source_pair_hashes_sha256": identity.get("ordered_source_pair_hashes_sha256"),
        "ordered_row_hashes_sha256": identity.get("ordered_row_hashes_sha256"),
    }
    require(summary == expected, "heldout29 row/order identity drifted")
    require(all(row.get("split") == "heldout" for row in rows), "heldout29 split label drifted")
    source: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = text(row.get("pair_id"), "heldout pair_id")
        source[pair_id] = {
            "pair_status": "inherited_day23_cpu_validated",
            "quality_flags": ["source_metadata_not_reopened_during_qualification"],
        }
    return rows, source, path


def _failure_document(
    state: Mapping[str, Any],
    claim: Mapping[str, Any],
    error: BaseException,
    phase: str,
    rows_opened: bool,
) -> dict[str, Any]:
    return {
        "schema_name": FAILURE_SCHEMA,
        "schema_version": 1,
        "status": "failed_closed_no_retry",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "authorization_sha256": state["authorization_sha256"],
        "heldout_claim_sha256": claim["identity"]["claim_sha256"],
        "failure_phase": phase,
        "heldout_rows_opened": rows_opened,
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error_message": str(error),
        "traceback": traceback.format_exception(error)[-12:],
        "retry_allowed": False,
        "fallback_allowed": False,
        "heldout_consumed": True,
    }


def run_evaluation(authorization_path: Path, device: str) -> dict[str, Any]:
    state = closeout.validate_authorization_for_evaluator(
        authorization_path, require_claim_unopened=True
    )
    require(
        state["authorization"].get("schema_name") == AUTHORIZATION_SCHEMA,
        "authorization schema drifted",
    )
    output = Path(state["authorization"]["evaluation_path"]).resolve()
    failure = Path(state["authorization"]["failure_after_claim_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    require(
        not output.exists()
        and not output.is_symlink()
        and not failure.exists()
        and not failure.is_symlink(),
        "heldout terminal evidence is not fresh",
    )
    runtime = _runtime_and_checkpoint(state, device)
    claim: dict[str, Any] | None = None
    phase = "heldout_claim_pending"
    rows_opened = False
    try:
        claim = _claim_once(state, runtime)
        phase = "heldout_claim_sealed"
        rows, source, dataset_path = _rows_after_claim(state)
        rows_opened = True
        phase = "heldout_rows_validated"
        pair_results = preference_eval.evaluate_rows(
            runtime["runtime"], rows, source, beta=0.1
        )
        require(len(pair_results) == HELDOUT_RECORDS, "heldout pair-result count drifted")
        for result in pair_results:
            require(
                result.get("schema_name") == preference_eval.PAIR_SCHEMA
                and result.get("status") == "pass"
                and result.get("split") == "heldout",
                "heldout pair-result schema/status drifted",
            )
            closeout.verify_self(result, "pair_evaluation_sha256", "heldout pair result")
            for key in (
                "policy_chosen_logps",
                "policy_rejected_logps",
                "reference_chosen_logps",
                "reference_rejected_logps",
                "chosen_reward",
                "rejected_reward",
                "reward_margin",
            ):
                number = result.get(key)
                require(
                    isinstance(number, (int, float))
                    and not isinstance(number, bool)
                    and math.isfinite(float(number)),
                    f"heldout pair-result contains non-finite {key}",
                )
        aggregate = preference_eval.aggregate_results(
            pair_results, split="heldout", checkpoint_step=CHECKPOINT_STEP
        )
        gate = closeout.heldout_gate(aggregate)
        authorization = state["authorization"]
        value = {
            "schema_name": EVALUATION_SCHEMA,
            "schema_version": 1,
            "status": "pass" if gate["passed"] else "gate_fail",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_id": RUN_ID,
            "split": "heldout",
            "authorization": {
                "path": str(state["authorization_path"]),
                "file_sha256": state["authorization_file_sha256"],
                "authorization_sha256": state["authorization_sha256"],
            },
            "campaign": authorization["campaign"],
            "training_receipt": authorization["training_receipt"],
            "checkpoint": authorization["checkpoint"],
            "dev_evaluation": authorization["dev_evaluation"],
            "full112_guardrail": authorization["full112_guardrail"],
            "e2b_guardrail": authorization["e2b_guardrail"],
            "dataset": authorization["heldout_dataset"],
            "dataset_runtime_path": str(dataset_path),
            "producer": authorization["heldout_evaluator"],
            "access_claim": claim["identity"],
            "runtime": runtime["runtime"]["runtime"],
            "process_identity": runtime["evaluator_process"],
            "training_process_identity": runtime["training_process"],
            "dev_process_identity": runtime["dev_process"],
            "pair_results": pair_results,
            "aggregate": aggregate,
            "eligibility": gate,
            "claim_boundary": {
                "fresh_process_checkpoint_reload_proven_before_claim": True,
                "adapter_tensor_inventory_and_content_exact": True,
                "global_heldout_claims": 1,
                "candidate_heldout_evaluations": 1,
                "retry_allowed": False,
                "fallback_allowed": False,
                "generation_performed": False,
                "sandbox_execution_performed": False,
                "heldout_consumed": True,
            },
        }
        phase = "heldout_pair_evaluation_completed"
        sealed = write_sealed(output, value, "evaluation_sha256")
        phase = "heldout_evaluation_sealed"
        return sealed
    except BaseException as error:
        if claim is None:
            raise
        try:
            write_sealed(
                failure,
                _failure_document(state, claim, error, phase, rows_opened),
                "failure_sha256",
            )
        except BaseException as seal_error:
            raise QualificationHeldoutError(
                f"heldout failed after claim and failure receipt could not be sealed: {seal_error}"
            ) from error
        raise


def _self_test() -> None:
    aggregate = {
        "overall": {
            "pairs": HELDOUT_RECORDS,
            "wins": MINIMUM_POSITIVE_PAIRS,
            "mean_reward_margin": 0.001,
        },
        "length_matched": {"mean_reward_margin": 0.0001},
    }
    require(closeout.heldout_gate(aggregate)["passed"], "heldout boundary self-test failed")
    aggregate["overall"]["wins"] = MINIMUM_POSITIVE_PAIRS - 1
    require(not closeout.heldout_gate(aggregate)["passed"], "20/29 rejection self-test failed")
    destinations = {action.dest for action in build_parser()._actions}
    require(
        destinations == {"help", "authorization", "device", "self_test"},
        "heldout evaluator CLI surface drifted",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {
                "status": "pass",
                "scope": "stdlib_self_test",
                "heldout_minimum_positive_pairs": MINIMUM_POSITIVE_PAIRS,
            }
        else:
            require(args.authorization is not None, "--authorization is required")
            result = run_evaluation(args.authorization, args.device)
    except BaseException as error:
        print(
            json.dumps(
                {"status": "fail_closed", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
