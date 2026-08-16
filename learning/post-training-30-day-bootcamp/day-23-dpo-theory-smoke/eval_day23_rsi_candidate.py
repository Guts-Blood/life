#!/usr/bin/env python3
"""Fresh-process preference evaluation and selection for Day 23 RSI.

The executable has no heldout mode.  Search is opened only after every frozen
search trajectory has a terminal receipt; dev is opened only for the unique
fresh refit selected by the frozen search rule.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract as day23_contract
import eval_day23_qwen35_preferences as day23_eval
import run_day23_qwen35_gpu_stage as day23_gpu
import run_day23_rsi_candidate as rsi_runner


SPLITS = ("mechanism", "search", "dev")
MECHANISM_PAIR_IDS = tuple(day23_eval.MECHANISM_PAIR_IDS)


class RSIEvaluationError(RuntimeError):
    """A campaign, checkpoint, access lease, or preference gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RSIEvaluationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value) and "\x00" not in value, f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return value


def _schema(context: Mapping[str, Any], suffix: str) -> str:
    version = _text(context["campaign"].get("version_id"), "campaign.version_id")
    _require(version in rsi_runner.SUPPORTED_RUNTIME_PROFILES, "unsupported evaluator campaign version")
    return f"day23.{version.replace('-', '_')}_{suffix}"


def _absolute(value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(_text(value, label)).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute")
    return path.resolve(strict=must_exist)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RSIEvaluationError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RSIEvaluationError(f"cannot read {label}: {path}") from error
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        _require(bool(line), f"blank line in {label}:{index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RSIEvaluationError(f"invalid JSON in {label}:{index}") from error
        _require(isinstance(value, dict), f"non-object row in {label}:{index}")
        rows.append(value)
    return rows


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _text(value.get(field), f"{label}.{field}")
    _require(len(expected) == 64 and day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def _write_exclusive(path: Path, value: Mapping[str, Any], field: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return day23_gpu.write_sealed_json(path, value, field)
    except (OSError, day23_gpu.Day23GPUStageError) as error:
        raise RSIEvaluationError(f"cannot seal {path}: {error}") from error


def _producer(campaign: Mapping[str, Any], name: str, actual_path: Path) -> None:
    entry = _mapping(_mapping(campaign.get("producers"), "campaign.producers").get(name), f"campaign.producers.{name}")
    _require(Path(_text(entry.get("path"), f"{name}.path")).resolve(strict=True) == actual_path.resolve(strict=True), f"campaign bound a different {name}")
    _require(day23_gpu.file_sha256(actual_path) == entry.get("file_sha256"), f"{name} source hash drifted")


def validate_campaign(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    campaign = _load_json(path, "GPU campaign")
    _require(campaign.get("schema_name") == rsi_runner.campaign_schema(campaign.get("version_id")), "evaluator requires a supported GPU-bound campaign")
    _require(campaign.get("schema_version") == 1, "GPU campaign schema version drifted")
    _require(campaign.get("status") == "gpu_execution_bound_optimizer_pending", "GPU campaign status drifted")
    sha = _verify_self(campaign, "campaign_sha256", "GPU campaign")
    _require(campaign.get("goal_id") == "goal-0002-day23-dpo", "campaign goal drifted")
    rsi_runner._version_token(campaign.get("version_id"))
    rsi_runner._validate_charter_binding(campaign)
    rsi_runner._validate_control_extension(campaign)
    _producer(campaign, "evaluator", Path(__file__))
    day23_gpu.validate_runtime(campaign)
    model_path = day23_gpu._verify_remote_parent_payload(_mapping(campaign.get("remote_parent"), "campaign.remote_parent"))
    root = _absolute(campaign.get("remote_run_root"), "campaign.remote_run_root")
    _require(path.is_relative_to(root), "GPU campaign escaped its run root")
    return {
        "campaign": campaign,
        "campaign_path": path,
        "campaign_file_sha256": day23_gpu.file_sha256(path),
        "campaign_sha256": sha,
        "run_root": root,
        "model_path": model_path,
    }


def _expected_evaluation_path(context: Mapping[str, Any], run_id: str, step: int, split: str) -> Path:
    if split == "dev":
        return context["run_root"] / "evidence" / "dev" / "finalist-dev.json"
    return context["run_root"] / "evidence" / "evaluations" / f"{run_id}-checkpoint-{step}-{split}.json"


def _expected_selection_path(context: Mapping[str, Any]) -> Path:
    return context["run_root"] / "evidence" / "selection" / "search-selection.json"


def _checkpoint_from_receipt(context: Mapping[str, Any], receipt_path: Path, step: int) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt = _load_json(receipt_path, "training receipt")
    receipt_sha = _verify_self(receipt, "receipt_sha256", "training receipt")
    strict = rsi_runner.validate_training_receipt_value(
        context["campaign"],
        context["campaign_path"],
        receipt,
        receipt_path,
    )
    _require(receipt.get("schema_name") == rsi_runner.receipt_schema(context["campaign"].get("version_id")) and receipt.get("status") == "pass", "training receipt is not a pass")
    campaign_ref = _mapping(receipt.get("campaign"), "training receipt campaign")
    _require(Path(_text(campaign_ref.get("path"), "receipt campaign path")).resolve() == context["campaign_path"], "receipt campaign path drifted")
    _require(campaign_ref.get("campaign_sha256") == context["campaign_sha256"], "receipt campaign hash drifted")
    run_id = _text(receipt.get("run_id"), "training receipt run_id")
    specs = _mapping(context["campaign"].get("run_specs"), "campaign.run_specs")
    spec = _mapping(specs.get(run_id), f"campaign run {run_id}")
    _require(strict.get("run_id") == run_id and strict.get("spec") == spec, "strict training receipt projection drifted")
    _require(receipt.get("run_role") == spec.get("role"), "training receipt role drifted")
    _require(receipt_path == Path(_text(spec.get("success_receipt"), "spec.success_receipt")).resolve(), "training receipt path is not bound")
    producer = _mapping(receipt.get("producer"), "training receipt producer")
    runner_entry = _mapping(_mapping(context["campaign"].get("producers"), "campaign.producers").get("runner"), "runner producer")
    _require(producer.get("file_sha256") == runner_entry.get("file_sha256"), "training receipt runner hash drifted")
    claim = _mapping(receipt.get("claim_boundary"), "training receipt claim boundary")
    for key in ("runtime_stage_passed", "fresh_parent_start", "fresh_lora_only", "reference_disable_adapter_observed", "frozen_tensor_versions_unchanged", "two_gpu_ddp_runtime_proven"):
        _require(claim.get(key) is True, f"training receipt lacks {key}")
    _require(claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, "training consumed protected data")
    distributed = _mapping(receipt.get("distributed_evidence"), "distributed evidence")
    _require(distributed.get("world_size") == 2 and len(distributed.get("ranks", [])) == 2, "two-rank evidence drifted")
    checkpoints = receipt.get("checkpoints")
    _require(isinstance(checkpoints, list), "checkpoint inventory is absent")
    matches = [item for item in checkpoints if item.get("global_step") == step]
    _require(len(matches) == 1 and step in spec.get("checkpoint_steps", []), "requested checkpoint is not preregistered")
    recorded = matches[0]
    checkpoint_path = _absolute(recorded.get("path"), "checkpoint path")
    actual = day23_gpu.checkpoint_manifest(checkpoint_path, step)
    _require(actual == recorded, "checkpoint bytes differ from training receipt")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_file_sha256": day23_gpu.file_sha256(receipt_path),
        "receipt_sha256": receipt_sha,
        "run_id": run_id,
        "spec": spec,
        "checkpoint": actual,
        "checkpoint_path": checkpoint_path,
    }


def _bootcamp_path(context: Mapping[str, Any], value: Any, label: str) -> Path:
    path = Path(_text(value, label))
    if path.is_absolute():
        return path.resolve(strict=True)
    root = _absolute(context["campaign"].get("bootcamp_root"), "campaign.bootcamp_root")
    resolved = (root / path).resolve(strict=True)
    _require(resolved.is_relative_to(root), f"{label} escaped bootcamp root")
    return resolved


def _dataset_rows(context: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Mapping[str, Any]]:
    datasets = _mapping(context["campaign"].get("datasets"), "campaign.datasets")
    key = "full_train" if split == "mechanism" else split
    entry = _mapping(datasets.get(key), f"campaign.datasets.{key}")
    path = _bootcamp_path(context, entry.get("path"), f"{key} dataset path")
    _require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"{key} dataset hash drifted")
    rows = _load_jsonl(path, f"{key} dataset")
    _require(len(rows) == entry.get("records"), f"{key} record count drifted")
    _require(day23_contract.object_sha256([row.get("pair_id") for row in rows]) == entry.get("ordered_pair_ids_sha256"), f"{key} ordered IDs drifted")
    source: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = _text(row.get("pair_id"), "compiled pair_id")
        try:
            day23_contract.verify_self_hash(row, "row_sha256", f"compiled row {pair_id}")
        except day23_contract.Day23ContractError as error:
            raise RSIEvaluationError(f"compiled row failed: {pair_id}: {error}") from error
        _require(
            row.get("schema_name") == day23_contract.ROW_SCHEMA_NAME
            and row.get("schema_version") == day23_contract.SCHEMA_VERSION,
            f"compiled row schema drifted: {pair_id}",
        )
        # Quality metadata is not used by any eligibility or ranking gate.  Do
        # not mount/open the 200-row Day 22 source file, because it also holds
        # the sealed heldout split.  The compiled row and its source-pair hash
        # were already validated by the frozen Day 23 CPU contract.
        source[pair_id] = {
            "pair_status": "inherited_day23_cpu_validated",
            "quality_flags": ["source_metadata_not_reopened_in_rsi_runtime"],
        }
    if split == "mechanism":
        by_id = {row["pair_id"]: row for row in rows}
        _require(all(pair_id in by_id for pair_id in MECHANISM_PAIR_IDS), "mechanism pair is absent from full_train")
        rows = [by_id[pair_id] for pair_id in MECHANISM_PAIR_IDS]
    return rows, source, entry


def _access_claim_path(context: Mapping[str, Any], split: str) -> Path:
    access = _mapping(
        context["campaign"].get("access_ledger"), "campaign.access_ledger"
    )
    path = _absolute(
        access.get(f"{split}_claim"), f"campaign {split} claim", must_exist=False
    )
    if split == "search":
        expected = (
            context["run_root"]
            / "evidence"
            / "search-selection"
            / "search-claim.json"
        )
    else:
        charter_binding = _mapping(
            context["campaign"].get("charter"), "campaign.charter"
        )
        bootcamp = _absolute(
            context["campaign"].get("bootcamp_root"),
            "campaign.bootcamp_root",
        )
        charter_rel = Path(
            _text(charter_binding.get("path"), "campaign charter path")
        )
        charter_path = (
            charter_rel if charter_rel.is_absolute() else bootcamp / charter_rel
        ).resolve(strict=True)
        charter = _load_json(charter_path, "authoritative charter")
        dev_lease = _mapping(
            _mapping(charter.get("access_leases"), "charter.access_leases").get(
                "dev"
            ),
            "charter dev lease",
        )
        relative = Path(_text(dev_lease.get("claim_path"), "charter dev claim"))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "charter dev claim path is not canonical",
        )
        expected = (bootcamp / "rsi-control" / relative).resolve()
        _require(
            dev_lease.get("maximum_global_claims") == 1,
            "charter dev global claim budget drifted",
        )
    _require(path == expected, f"{split} claim path escaped its frozen ledger")
    return path


def _claim_or_validate_search(context: Mapping[str, Any]) -> dict[str, Any]:
    path = _access_claim_path(context, "search")
    runs = _mapping(context["campaign"].get("run_specs"), "campaign.run_specs")
    candidates = sorted(
        _candidate_id(context["campaign"], run_id, int(step))
        for run_id, spec in runs.items()
        if isinstance(spec, Mapping) and spec.get("role") == "search_train"
        for step in spec.get("checkpoint_steps", [])
    )
    expected = {
        "schema_name": _schema(context, "access_claim"),
        "schema_version": 1,
        "status": "claimed",
        "split": "search",
        "campaign_sha256": context["campaign_sha256"],
        "authorized_candidates": candidates,
        "max_unseal_count": 1,
        "heldout_consumed": False,
    }
    if not path.exists():
        _write_exclusive(path, expected, "claim_sha256")
    value = _load_json(path, "search access claim")
    _verify_self(value, "claim_sha256", "search access claim")
    for key, item in expected.items():
        _require(value.get(key) == item, f"search access claim drifted: {key}")
    return {"path": str(path), "file_sha256": day23_gpu.file_sha256(path), "claim_sha256": value["claim_sha256"]}


def _require_complete_search_grid(context: Mapping[str, Any]) -> None:
    runs = _mapping(context["campaign"].get("run_specs"), "campaign.run_specs")
    search_specs = [
        (run_id, spec)
        for run_id, spec in sorted(runs.items())
        if isinstance(spec, Mapping) and spec.get("role") == "search_train"
    ]
    _require(len(search_specs) == 2, "campaign search trajectory count drifted")
    for run_id, spec in search_specs:
        success_path = _absolute(
            spec.get("success_receipt"), f"{run_id} success receipt"
        )
        failure_path = Path(
            _text(spec.get("failure_receipt"), f"{run_id} failure receipt")
        ).resolve()
        _require(not failure_path.exists(), f"search grid contains a failed trajectory: {run_id}")
        receipt = _load_json(success_path, f"{run_id} success receipt")
        _verify_self(receipt, "receipt_sha256", f"{run_id} success receipt")
        strict = rsi_runner.validate_training_receipt_value(
            context["campaign"],
            context["campaign_path"],
            receipt,
            success_path,
        )
        _require(
            receipt.get("schema_name") == rsi_runner.receipt_schema(context["campaign"].get("version_id"))
            and receipt.get("status") == "pass"
            and receipt.get("run_id") == run_id
            and receipt.get("campaign", {}).get("campaign_sha256")
            == context["campaign_sha256"],
            f"search trajectory receipt drifted: {run_id}",
        )
        _require(strict.get("run_id") == run_id and strict.get("spec") == spec, f"search trajectory strict projection drifted: {run_id}")


def _claim_dev(context: Mapping[str, Any], checkpoint: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    path = _access_claim_path(context, "dev")
    _require(not path.exists(), "dev was already claimed")
    value = {
        "schema_name": _schema(context, "access_claim"),
        "schema_version": 1,
        "status": "claimed",
        "split": "dev",
        "campaign_sha256": context["campaign_sha256"],
        "selected_candidate": selection.get("selected_candidate"),
        "refit_run_id": checkpoint["run_id"],
        "refit_training_receipt_sha256": checkpoint["receipt_sha256"],
        "max_unseal_count": 1,
        "max_candidate_evaluations": 1,
        "heldout_consumed": False,
    }
    sha = _write_exclusive(path, value, "claim_sha256")
    return {"path": str(path), "file_sha256": day23_gpu.file_sha256(path), "claim_sha256": sha}


def _load_selection(context: Mapping[str, Any], path: Path) -> dict[str, Any]:
    value = validate_search_selection(context, path)
    _require(value.get("status") == "selected", "search did not select a finalist")
    return value


def _candidate_id(
    campaign: Mapping[str, Any], run_id: str, step: int
) -> str:
    registry = campaign.get("candidate_registry")
    _require(isinstance(registry, list), "campaign candidate registry is absent")
    matches = [
        item
        for item in registry
        if isinstance(item, Mapping)
        and item.get("search_run_spec") == run_id
        and item.get("checkpoint_step") == step
    ]
    _require(len(matches) == 1, "candidate registry mapping drifted")
    return _text(matches[0].get("candidate_id"), "candidate registry ID")


def _evaluation_runtime_context(context: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(context["campaign"].get("runtime_parse"), "campaign.runtime_parse")
    config_identity = _mapping(
        checkpoint["spec"].get("executable_config"), "checkpoint config identity"
    )
    config_path = _absolute(config_identity.get("path"), "config path")
    _require(
        day23_gpu.file_sha256(config_path)
        == config_identity.get("file_sha256"),
        "evaluation config file hash drifted",
    )
    config = _load_json(config_path, "training config")
    return {
        "ms_swift": {"path": runtime.get("ms_swift_checkout")},
        "model_path": context["model_path"],
        "config": config,
        "runtime_parse": runtime,
    }


def _eligibility(aggregate: Mapping[str, Any], split: str) -> dict[str, Any]:
    overall = _mapping(aggregate.get("overall"), "aggregate.overall")
    matched = _mapping(aggregate.get("length_matched"), "aggregate.length_matched")
    pairs = _integer(overall.get("pairs"), "aggregate pair count")
    wins = _integer(overall.get("wins"), "aggregate wins")
    mean = float(overall.get("mean_reward_margin"))
    matched_mean = float(matched.get("mean_reward_margin"))
    _require(all(math.isfinite(value) for value in (mean, matched_mean)), "aggregate contains non-finite margins")
    if split == "search":
        passed = pairs == 30 and wins >= 20 and mean > 0 and matched_mean > 0
        thresholds = {"pairs": 30, "minimum_positive_pairs": 20, "mean_reward_margin": ">0", "length_matched_mean_reward_margin": ">0"}
    elif split == "dev":
        passed = pairs == 17 and wins >= 13 and mean > 0 and matched_mean > 0
        thresholds = {"pairs": 17, "minimum_positive_pairs": 13, "mean_reward_margin": ">0", "length_matched_mean_reward_margin": ">0"}
    else:
        passed = True
        thresholds = {"role": "diagnostic_only_not_a_candidate_gate"}
    return {"passed": passed, "thresholds": thresholds, "observed": {"pairs": pairs, "positive_pairs": wins, "mean_reward_margin": mean, "length_matched_mean_reward_margin": matched_mean}}


def run_evaluation(
    campaign_path: Path,
    training_receipt: Path,
    checkpoint_step: int,
    split: str,
    output: Path,
    selection_path: Path | None,
    device: str,
) -> dict[str, Any]:
    context = validate_campaign(campaign_path)
    checkpoint = _checkpoint_from_receipt(context, training_receipt, checkpoint_step)
    output = output.expanduser().resolve()
    _require(output == _expected_evaluation_path(context, checkpoint["run_id"], checkpoint_step, split), "evaluation output path drifted")
    _require(not output.exists(), "evaluation output already exists")
    role = checkpoint["spec"].get("role")
    if split in {"search", "mechanism"}:
        _require(role == "search_train", f"{split} may evaluate only a search run")
    else:
        _require(role == "unique_finalist_refit_option", "dev may evaluate only the selected fresh refit")

    selection: Mapping[str, Any] | None = None
    access: Mapping[str, Any] | None = None
    if split == "search":
        _require_complete_search_grid(context)
        access = _claim_or_validate_search(context)
    elif split == "dev":
        _require(selection_path is not None, "dev requires the sealed search selection")
        selection = _load_selection(context, selection_path)
        selected = _text(selection.get("selected_candidate"), "selected candidate")
        authorization = _mapping(
            checkpoint["spec"].get("authorization"), "refit authorization"
        )
        required_candidate = checkpoint["spec"].get(
            "authorized_only_if_selected_candidate",
            authorization.get("requires_selected_candidate_id"),
        )
        _require(required_candidate == selected, "refit run does not implement the selected candidate")
        access = _claim_dev(context, checkpoint, selection)

    rows, source, dataset_entry = _dataset_rows(context, split)
    runtime_context = _evaluation_runtime_context(context, checkpoint)
    runtime = day23_eval.load_policy(runtime_context, {"checkpoint_path": checkpoint["checkpoint_path"]}, device_name=device)
    results = day23_eval.evaluate_rows(runtime, rows, source, beta=0.1)
    aggregate = day23_eval.aggregate_results(results, split="dev" if split == "dev" else split, checkpoint_step=checkpoint_step)
    gate = _eligibility(aggregate, split)
    evaluator_process = day23_gpu._process_identity()
    runner_process = _mapping(
        checkpoint["receipt"].get("process_identity"),
        "training runner process identity",
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
    value = {
        "schema_name": _schema(context, "preference_evaluation"),
        "schema_version": 1,
        "status": "pass" if gate["passed"] else "gate_fail",
        "campaign_sha256": context["campaign_sha256"],
        "run_id": checkpoint["run_id"],
        "run_role": role,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": day23_gpu.file_sha256(Path(__file__).resolve()),
        },
        "checkpoint_step": checkpoint_step,
        "checkpoint": {
            "path": str(checkpoint["checkpoint_path"]),
            "files_sha256": checkpoint["checkpoint"]["files_sha256"],
            "training_receipt_path": str(checkpoint["receipt_path"]),
            "training_receipt_file_sha256": checkpoint[
                "receipt_file_sha256"
            ],
            "training_receipt_sha256": checkpoint["receipt_sha256"],
        },
        "split": split,
        "dataset": {
            "file_sha256": dataset_entry.get("file_sha256"),
            "records": len(rows),
            "ordered_pair_ids_sha256": day23_contract.object_sha256([row["pair_id"] for row in rows]),
            "ordered_row_hashes_sha256": day23_contract.object_sha256(
                [row["row_sha256"] for row in rows]
            ),
        },
        "access_claim": access,
        "selection_sha256": selection.get("selection_sha256") if selection else None,
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
            "dev_consumed": split == "dev",
            "heldout_consumed": False,
        },
    }
    _write_exclusive(output, value, "evaluation_sha256")
    return _load_json(output, "sealed evaluation")


def _expected_search_candidates(context: Mapping[str, Any]) -> list[tuple[str, int, Mapping[str, Any], Path]]:
    result: list[tuple[str, int, Mapping[str, Any], Path]] = []
    for run_id, spec in sorted(_mapping(context["campaign"].get("run_specs"), "campaign.run_specs").items()):
        if isinstance(spec, Mapping) and spec.get("role") == "search_train":
            for step in spec.get("checkpoint_steps", []):
                result.append((run_id, int(step), spec, _expected_evaluation_path(context, run_id, int(step), "search")))
    _require(len(result) == 4, "campaign must preregister exactly four search candidates")
    return result


def _validate_search_evaluation(context: Mapping[str, Any], run_id: str, step: int, spec: Mapping[str, Any], path: Path) -> dict[str, Any]:
    _require(path == _expected_evaluation_path(context, run_id, step, "search"), "search evaluation path drifted")
    value = _load_json(path, "search evaluation")
    sha = _verify_self(value, "evaluation_sha256", "search evaluation")
    _require(value.get("schema_name") == _schema(context, "preference_evaluation") and value.get("campaign_sha256") == context["campaign_sha256"], "search evaluation binding drifted")
    _require(value.get("run_id") == run_id and value.get("checkpoint_step") == step and value.get("split") == "search", "search evaluation candidate drifted")
    _require(value.get("status") in {"pass", "gate_fail"}, "search evaluation status drifted")
    producer = _mapping(value.get("producer"), "search evaluation producer")
    evaluator_bound = _mapping(
        _mapping(context["campaign"].get("producers"), "campaign.producers").get(
            "evaluator"
        ),
        "bound evaluator",
    )
    _require(
        Path(_text(producer.get("path"), "search evaluator path")).resolve()
        == Path(_text(evaluator_bound.get("path"), "bound evaluator path")).resolve()
        and producer.get("file_sha256") == evaluator_bound.get("file_sha256"),
        "search evaluation producer drifted",
    )
    checkpoint_value = _mapping(value.get("checkpoint"), "search checkpoint")
    strict_checkpoint = _checkpoint_from_receipt(
        context,
        Path(
            _text(
                checkpoint_value.get("training_receipt_path"),
                "search training receipt path",
            )
        ),
        step,
    )
    _require(
        strict_checkpoint["run_id"] == run_id
        and checkpoint_value
        == {
            "path": str(strict_checkpoint["checkpoint_path"]),
            "files_sha256": strict_checkpoint["checkpoint"]["files_sha256"],
            "training_receipt_path": str(strict_checkpoint["receipt_path"]),
            "training_receipt_file_sha256": strict_checkpoint[
                "receipt_file_sha256"
            ],
            "training_receipt_sha256": strict_checkpoint["receipt_sha256"],
        },
        "search checkpoint/training receipt binding drifted",
    )
    rows, _, dataset_entry = _dataset_rows(context, "search")
    expected_pair_ids = [row["pair_id"] for row in rows]
    expected_dataset = {
        "file_sha256": dataset_entry.get("file_sha256"),
        "records": len(rows),
        "ordered_pair_ids_sha256": day23_contract.object_sha256(expected_pair_ids),
        "ordered_row_hashes_sha256": day23_contract.object_sha256(
            [row["row_sha256"] for row in rows]
        ),
    }
    _require(value.get("dataset") == expected_dataset, "search evaluation dataset identity drifted")
    pair_results = value.get("pair_results")
    _require(isinstance(pair_results, list) and len(pair_results) == 30, "search pair-result inventory drifted")
    _require([item.get("pair_id") for item in pair_results] == expected_pair_ids and len(set(expected_pair_ids)) == 30, "search pair-result IDs/order drifted")
    for item in pair_results:
        _require(item.get("schema_name") == day23_eval.PAIR_SCHEMA and item.get("status") == "pass", "search pair-result schema/status drifted")
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
            _require(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(float(number)), f"search pair-result non-finite {key}")
        _require(item.get("pair_accuracy") is (float(item["reward_margin"]) > 0), "search pair accuracy drifted")
    expected_aggregate = day23_eval.aggregate_results(
        pair_results, split="search", checkpoint_step=step
    )
    _require(value.get("aggregate") == expected_aggregate, "search aggregate was not recomputed from pair results")
    gate = _mapping(value.get("eligibility"), "search eligibility")
    recomputed = _eligibility(expected_aggregate, "search")
    _require(gate == recomputed, "search eligibility was not recomputed from aggregate")
    claim = _mapping(value.get("claim_boundary"), "search evaluation claim")
    _require(
        claim.get("fresh_process_checkpoint_reload_proven") is True
        and claim.get("adapter_tensor_inventory_and_content_exact") is True
        and claim.get("dev_consumed") is False
        and claim.get("heldout_consumed") is False,
        "search evaluation claim boundary drifted",
    )
    runtime_expected = _mapping(
        context["campaign"].get("runtime_parse"), "campaign.runtime_parse"
    ).get("package_versions")
    runtime_actual = _mapping(value.get("runtime"), "search evaluation runtime")
    for package, version in _mapping(runtime_expected, "bound package versions").items():
        key = package if package != "ms_swift" else "ms-swift"
        _require(runtime_actual.get(key) == version, f"search evaluation runtime drifted: {package}")
    process = _mapping(value.get("process_identity"), "search evaluator process")
    runner_process = _mapping(value.get("runner_process_identity"), "search runner process")
    _require(
        (process.get("boot_id"), process.get("pid"), process.get("proc_start_ticks"))
        != (
            runner_process.get("boot_id"),
            runner_process.get("pid"),
            runner_process.get("proc_start_ticks"),
        )
        and runner_process == strict_checkpoint["receipt"].get("process_identity"),
        "search evaluation fresh-process identity drifted",
    )
    return {
        "value": value,
        "path": str(path),
        "file_sha256": day23_gpu.file_sha256(path),
        "evaluation_sha256": sha,
        "candidate_id": _candidate_id(context["campaign"], run_id, step),
        "run_id": run_id,
        "checkpoint_step": step,
        "learning_rate": float(spec.get("learning_rate")),
        "eligible": bool(gate["passed"]),
        "positive_pairs": int(gate["observed"]["positive_pairs"]),
        "mean_reward_margin": float(gate["observed"]["mean_reward_margin"]),
        "length_matched_mean_reward_margin": float(gate["observed"]["length_matched_mean_reward_margin"]),
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
        "schema_name": _schema(context, "search_selection"),
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
        "selected_checkpoint_step": (
            selected["checkpoint_step"] if selected else None
        ),
        "selected_learning_rate": selected["learning_rate"] if selected else None,
        "selected_evaluation_sha256": (
            selected["evaluation_sha256"] if selected else None
        ),
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
    _require(path == _expected_selection_path(context), "selection receipt path drifted")
    value = _load_json(path, "search selection")
    _verify_self(value, "selection_sha256", "search selection")
    candidates = [
        _validate_search_evaluation(context, run_id, step, spec, eval_path)
        for run_id, step, spec, eval_path in _expected_search_candidates(context)
    ]
    expected = _selection_document(context, candidates)
    actual = dict(value)
    actual.pop("selection_sha256", None)
    _require(actual == expected, "search selection was not exactly recomputed from all four evaluations")
    return value


def select_search(campaign_path: Path, output: Path) -> dict[str, Any]:
    context = validate_campaign(campaign_path)
    output = output.expanduser().resolve()
    _require(output == _expected_selection_path(context), "selection output path drifted")
    _require(not output.exists(), "search selection already exists")
    candidates = [
        _validate_search_evaluation(context, run_id, step, spec, path)
        for run_id, step, spec, path in _expected_search_candidates(context)
    ]
    value = _selection_document(context, candidates)
    _write_exclusive(output, value, "selection_sha256")
    return _load_json(output, "sealed search selection")


def _self_test() -> None:
    aggregate = {
        "overall": {"pairs": 30, "wins": 20, "mean_reward_margin": 0.01},
        "length_matched": {"mean_reward_margin": 0.001},
    }
    _require(_eligibility(aggregate, "search")["passed"], "search threshold self-test failed")
    aggregate["overall"]["wins"] = 19
    _require(not _eligibility(aggregate, "search")["passed"], "search rejection self-test failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--split", choices=SPLITS)
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
            _require(
                args.selection_receipt is not None,
                "selection receipt is required for verification",
            )
            context = validate_campaign(args.campaign)
            result = validate_search_selection(context, args.selection_receipt)
        elif args.select_search:
            _require(args.output is not None, "selection output is required")
            result = select_search(args.campaign, args.output)
        else:
            _require(args.output is not None, "evaluation output is required")
            _require(args.training_receipt is not None and args.checkpoint_step is not None and args.split is not None, "evaluation requires training receipt, checkpoint step, and split")
            result = run_evaluation(args.campaign, args.training_receipt, args.checkpoint_step, args.split, args.output, args.selection_receipt, args.device)
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error), "traceback": traceback.format_exception(error)[-6:]}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result.get("status"), "output": str(args.output) if args.output else None, "self_hash": result.get("evaluation_sha256", result.get("selection_sha256"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
