#!/usr/bin/env python3
"""Preflight, seal, and check the terminal Day 23 RSI-v0005 campaign.

v0005 is authorized only after a genuine v0004 ``closed_no_candidate``
selection.  It rotates the already evaluated v0004 search rows back into the
optimizer-only fit split, chooses a new search30 exclusively from v0004's
previously unevaluated fit rows, and starts every trajectory from the promoted
S1 parent.  This builder deliberately has no code path that opens dev or
heldout data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_day23_rsi_v0004 as v4
import day23_contract as contract
import run_day23_qwen35_gpu_stage as day23_gpu


VERSION_ID = "rsi-v0005"
SOURCE_VERSION = "rsi-v0004"
CAMPAIGN_ID = "day23-qwen35-dpo-rsi-v0005"
GOAL_ID = "goal-0002-day23-dpo"
GPU_SCHEMA = "day23.rsi_v0005_gpu_campaign"
SOURCE_SCHEMA = "day23.rsi_v0004_gpu_campaign"
SOURCE_SELECTION_SCHEMA = "day23.rsi_v0004_search_selection"
CONTROL_ROOT_REL = Path("rsi-control/charters/goal-0002-day23-dpo")

SCIENTIFIC_LEVER = "model.optimization.learning_rate.base"
# Frozen only when build mode writes the append-only amendment.  These values
# implement the preregistered interpolation around the v0004 response island.
LEARNING_RATES = (3.9e-6, 4.1e-6)
CHECKPOINT_STEPS = (18, 20)
SEARCH_MAX_STEPS = 30
SEARCH_SAVE_STEPS = 2
SEARCH_SAVE_TOTAL_LIMIT = 7
SEARCH_SEED = 20260819
REFIT_SEED = 20260820
WORLD_SIZE = 2
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
GLOBAL_BATCH = 32
MIN_FREE_FRACTION = 0.20
SELECTION_SALT = f"{CAMPAIGN_ID}|"

OLD_MECHANISM_PAIR_IDS = (
    "mbpp:task:602:s1pair:37673b32b9385aea",
    "mbpp:task:604:s1pair:388a8daec7e97c5a",
    "mbpp:task:605:s1pair:9ae278fe596ed4d4",
    "mbpp:task:610:s1pair:9e57fb63f7544777",
)
SEARCH_RUN_IDS = ("search_lr_3p9e_6", "search_lr_4p1e_6")
CONFIG_FILENAMES = {
    "search_lr_3p9e_6": "search-lr3p9e-6.json",
    "search_lr_4p1e_6": "search-lr4p1e-6.json",
    "refit_lr_3p9e_6_step_18": "refit-lr3p9e-6-step18.json",
    "refit_lr_3p9e_6_step_20": "refit-lr3p9e-6-step20.json",
    "refit_lr_4p1e_6_step_18": "refit-lr4p1e-6-step18.json",
    "refit_lr_4p1e_6_step_20": "refit-lr4p1e-6-step20.json",
}


class V0005BuildError(RuntimeError):
    """A source-terminal, split, precommit, or built-artifact invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0005BuildError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V0005BuildError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    _require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def _bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _identity(
    path: Path,
    *,
    content_field: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"missing regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if content_field:
        result["content_sha256"] = _verify(_load(resolved, str(resolved)), content_field, str(resolved))
    return result


def _bound_file(
    bootcamp: Path,
    entry: Any,
    self_field: str,
    label: str,
) -> tuple[dict[str, Any], Path]:
    identity = _mapping(entry, label)
    raw = Path(str(identity.get("path")))
    path = (raw if raw.is_absolute() else bootcamp / raw).resolve(strict=True)
    _require(path.is_relative_to(bootcamp), f"{label} escaped bootcamp")
    _require(day23_gpu.file_sha256(path) == identity.get("file_sha256"), f"{label} file hash drifted")
    value = _load(path, label)
    _require(_verify(value, self_field, label) == identity.get("content_sha256"), f"{label} content binding drifted")
    return value, path


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = contract.load_jsonl(path)
    except (OSError, contract.Day23ContractError) as error:
        raise V0005BuildError(f"cannot validate {label}: {path}: {error}") from error
    return rows


def _dataset_identity(path: Path, payload: bytes, rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "records": len(rows),
        "ordered_pair_ids_sha256": contract.object_sha256([str(row["pair_id"]) for row in rows]),
        "ordered_row_hashes_sha256": contract.object_sha256([str(row["row_sha256"]) for row in rows]),
        "role": role,
    }


def _validate_dataset(path: Path, entry: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    path = path.resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    _require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"{label} file hash drifted")
    _require(path.stat().st_size == entry.get("bytes"), f"{label} byte count drifted")
    rows = _load_jsonl(path, label)
    _require(len(rows) == entry.get("records"), f"{label} record count drifted")
    pair_ids: list[str] = []
    row_hashes: list[str] = []
    for row in rows:
        pair_id = str(row.get("pair_id"))
        _require(pair_id and pair_id != "None", f"{label} pair_id missing")
        try:
            contract.verify_self_hash(row, "row_sha256", f"{label} row {pair_id}")
        except contract.Day23ContractError as error:
            raise V0005BuildError(str(error)) from error
        pair_ids.append(pair_id)
        row_hashes.append(str(row["row_sha256"]))
    _require(len(pair_ids) == len(set(pair_ids)), f"{label} pair IDs are not unique")
    _require(contract.object_sha256(pair_ids) == entry.get("ordered_pair_ids_sha256"), f"{label} ordered IDs drifted")
    _require(contract.object_sha256(row_hashes) == entry.get("ordered_row_hashes_sha256"), f"{label} ordered row hashes drifted")
    return rows


def _source_terminal(
    source_campaign_path: Path,
    source_selection_path: Path,
) -> dict[str, Any]:
    """Validate v0004 terminal evidence without opening dev or heldout rows."""
    source_campaign_path = source_campaign_path.resolve(strict=True)
    source = _load(source_campaign_path, "v0004 campaign")
    source_sha = _verify(source, "campaign_sha256", "v0004 campaign")
    _require(
        source.get("schema_name") == SOURCE_SCHEMA
        and source.get("schema_version") == 1
        and source.get("status") == "gpu_execution_bound_optimizer_pending"
        and source.get("goal_id") == GOAL_ID
        and source.get("version_id") == SOURCE_VERSION,
        "v0004 campaign identity drifted",
    )
    bootcamp = Path(str(source.get("bootcamp_root"))).resolve(strict=True)
    source_root = Path(str(source.get("remote_run_root"))).resolve(strict=True)
    _require(source_campaign_path.is_relative_to(source_root), "v0004 campaign escaped its run root")
    _require(
        source_selection_path.resolve(strict=True)
        == source_root / "evidence/selection/search-selection.json",
        "v0004 selection is not the canonical bound selection",
    )

    charter, charter_path = _bound_file(bootcamp, source.get("charter"), "charter_sha256", "charter")
    _require(charter.get("goal_id") == GOAL_ID, "charter goal drifted")
    extension = _mapping(source.get("control_extension"), "v0004 control extension")
    event4, event4_path = _bound_file(bootcamp, extension.get("precommit_event"), "event_sha256", "v0004 event4")
    _require(
        event4.get("sequence") == 4
        and event4.get("event_type") == "version-precommitted"
        and event4.get("version_id") == SOURCE_VERSION
        and event4.get("state_after") == "version_precommitted",
        "v0004 event4 transition drifted",
    )

    strict_path = source_root / "evidence/authority/strict-preunseal.json"
    strict = _load(strict_path, "v0004 strict pre-unseal receipt")
    strict_sha = _verify(strict, "validation_sha256", "v0004 strict pre-unseal receipt")
    _require(
        strict.get("schema_name") == "day23.rsi_v0004_strict_authority_preflight"
        and strict.get("schema_version") == 1
        and strict.get("status") == "pass"
        and strict.get("campaign_sha256") == source_sha
        and strict.get("campaign_file_sha256") == day23_gpu.file_sha256(source_campaign_path)
        and strict.get("search_unopened_at_preflight") is True
        and strict.get("dev_unopened") is True
        and strict.get("heldout_unopened") is True,
        "v0004 strict pre-unseal authority drifted",
    )
    strict_amendment, strict_amendment_path = _bound_file(
        bootcamp,
        strict.get("strict_preunseal_amendment"),
        "amendment_sha256",
        "v0004 strict pre-unseal amendment",
    )
    _require(
        strict_amendment.get("amendment_id") == "amendment-000002-v0004-strict-preunseal"
        and strict_amendment.get("campaign", {}).get("campaign_sha256") == source_sha
        and strict_amendment.get("required_before_first_search_claim") is True,
        "v0004 strict pre-unseal amendment drifted",
    )

    search_claim_path = Path(str(_mapping(source.get("access_ledger"), "v0004 access ledger").get("search_claim"))).resolve(strict=True)
    _require(search_claim_path == source_root / "evidence/search-selection/search-claim.json", "v0004 search claim path drifted")
    search_claim = _load(search_claim_path, "v0004 search claim")
    _verify(search_claim, "claim_sha256", "v0004 search claim")
    _require(
        search_claim.get("schema_name") == "day23.rsi_v0004_access_claim"
        and search_claim.get("status") == "claimed"
        and search_claim.get("split") == "search"
        and search_claim.get("campaign_sha256") == source_sha
        and search_claim.get("heldout_consumed") is False,
        "v0004 search claim drifted",
    )

    dev_claim = Path(str(source["access_ledger"]["dev_claim"]))
    _require(not dev_claim.exists(), "global dev lease has already been opened")
    heldout_relative = Path(str(charter["access_leases"]["heldout"]["claim_path"]))
    _require(not (bootcamp / "rsi-control" / heldout_relative).exists(), "global heldout lease has already been opened")

    selection = _load(source_selection_path.resolve(strict=True), "v0004 selection")
    selection_sha = _verify(selection, "selection_sha256", "v0004 selection")
    _require(
        selection.get("schema_name") == SOURCE_SELECTION_SCHEMA
        and selection.get("schema_version") == 1
        and selection.get("status") == "closed_no_candidate"
        and selection.get("campaign_sha256") == source_sha
        and selection.get("candidate_pool_frozen") is True
        and selection.get("candidate_count") == 4
        and selection.get("eligible_count") == 0
        and selection.get("selected_candidate") is None
        and selection.get("selected_run_id") is None
        and selection.get("selected_checkpoint_step") is None
        and selection.get("selected_learning_rate") is None
        and selection.get("selected_evaluation_sha256") is None,
        "v0004 terminal selection drifted",
    )
    selection_claim = _mapping(selection.get("claim_boundary"), "v0004 selection claim boundary")
    _require(selection_claim.get("dev_consumed") is False and selection_claim.get("heldout_consumed") is False, "v0004 selection crossed a protected-data boundary")

    run_specs = _mapping(source.get("run_specs"), "v0004 run specs")
    source_receipts: list[dict[str, Any]] = []
    receipts_by_run: dict[str, dict[str, Any]] = {}
    for run_id in ("search_lr_1e_6", "search_lr_5e_6"):
        spec = _mapping(run_specs.get(run_id), f"v0004 run {run_id}")
        receipt_path = Path(str(spec.get("success_receipt"))).resolve(strict=True)
        failure_path = Path(str(spec.get("failure_receipt"))).resolve()
        _require(not failure_path.exists(), f"v0004 search trajectory failed: {run_id}")
        receipt = _load(receipt_path, f"v0004 receipt {run_id}")
        receipt_sha = _verify(receipt, "receipt_sha256", f"v0004 receipt {run_id}")
        claim = _mapping(receipt.get("claim_boundary"), f"v0004 receipt claim {run_id}")
        observations = _mapping(receipt.get("observations"), f"v0004 receipt observations {run_id}")
        minimum = _mapping(observations.get("memory"), f"v0004 receipt memory {run_id}").get("minimum_observed_device_free_fraction")
        _require(
            receipt.get("schema_name") == "day23.rsi_v0004_training_receipt"
            and receipt.get("schema_version") == 1
            and receipt.get("status") == "pass"
            and receipt.get("run_id") == run_id
            and receipt.get("campaign", {}).get("campaign_sha256") == source_sha
            and receipt.get("campaign", {}).get("file_sha256") == day23_gpu.file_sha256(source_campaign_path)
            and receipt.get("config", {}).get("file_sha256") == spec.get("executable_config", {}).get("file_sha256")
            and receipt.get("producer", {}).get("file_sha256") == source.get("producers", {}).get("runner", {}).get("file_sha256")
            and observations.get("global_step") == 30
            and isinstance(minimum, (int, float))
            and math.isfinite(float(minimum))
            and float(minimum) >= MIN_FREE_FRACTION,
            f"v0004 success receipt terminal/runtime drifted: {run_id}",
        )
        for flag in (
            "runtime_stage_passed",
            "fresh_parent_start",
            "fresh_lora_only",
            "reference_disable_adapter_observed",
            "frozen_tensor_versions_unchanged",
            "two_gpu_ddp_runtime_proven",
        ):
            _require(claim.get(flag) is True, f"v0004 receipt lacks {flag}: {run_id}")
        _require(claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, f"v0004 receipt consumed protected data: {run_id}")
        item = {
            **_identity(receipt_path, content_field="receipt_sha256"),
            "run_id": run_id,
            "status": "pass",
            "global_step": 30,
            "minimum_observed_device_free_fraction": float(minimum),
        }
        source_receipts.append(item)
        receipts_by_run[run_id] = {"value": receipt, "identity": item, "sha": receipt_sha}

    candidates = selection.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == 4, "v0004 selection candidate list drifted")
    registry = source.get("candidate_registry")
    _require(isinstance(registry, list) and len(registry) == 4, "v0004 candidate registry drifted")
    registry_ids = {str(item.get("candidate_id")) for item in registry if isinstance(item, Mapping)}
    source_evaluations: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        item = _mapping(candidate, "v0004 selection candidate")
        run_id = str(item.get("run_id"))
        step = item.get("checkpoint_step")
        _require(run_id in receipts_by_run and isinstance(step, int) and step in (15, 30), "v0004 candidate run/step drifted")
        _require((run_id, step) not in seen, "duplicate v0004 candidate")
        seen.add((run_id, step))
        _require(item.get("candidate_id") in registry_ids and item.get("eligible") is False, "v0004 candidate eligibility/registry drifted")
        evaluation_path = Path(str(item.get("path"))).resolve(strict=True)
        expected_path = source_root / "evidence/evaluations" / f"{run_id}-checkpoint-{step}-search.json"
        _require(evaluation_path == expected_path, "v0004 evaluation path drifted")
        _require(day23_gpu.file_sha256(evaluation_path) == item.get("file_sha256"), "v0004 evaluation file hash drifted")
        evaluation = _load(evaluation_path, "v0004 search evaluation")
        evaluation_sha = _verify(evaluation, "evaluation_sha256", "v0004 search evaluation")
        gate = _mapping(evaluation.get("eligibility"), "v0004 search evaluation gate")
        observed = _mapping(gate.get("observed"), "v0004 search evaluation observed")
        pair_results = evaluation.get("pair_results")
        _require(
            evaluation_sha == item.get("evaluation_sha256")
            and evaluation.get("schema_name") == "day23.rsi_v0004_preference_evaluation"
            and evaluation.get("campaign_sha256") == source_sha
            and evaluation.get("run_id") == run_id
            and evaluation.get("checkpoint_step") == step
            and evaluation.get("split") == "search"
            and evaluation.get("status") == "gate_fail"
            and evaluation.get("producer", {}).get("file_sha256") == source.get("producers", {}).get("evaluator", {}).get("file_sha256")
            and evaluation.get("access_claim", {}).get("claim_sha256") == search_claim.get("claim_sha256")
            and gate.get("passed") is False
            and isinstance(pair_results, list)
            and len(pair_results) == 30,
            "v0004 search evaluation terminal binding drifted",
        )
        for pair_result in pair_results:
            pair = _mapping(pair_result, "v0004 pair result")
            _verify(pair, "pair_evaluation_sha256", "v0004 pair result")
            margin = pair.get("reward_margin")
            _require(
                pair.get("status") == "pass"
                and isinstance(margin, (int, float))
                and math.isfinite(float(margin))
                and pair.get("pair_accuracy") is (float(margin) > 0),
                "v0004 pair result drifted",
            )
        _require(
            observed.get("positive_pairs") == item.get("positive_pairs")
            and observed.get("mean_reward_margin") == item.get("mean_reward_margin")
            and observed.get("length_matched_mean_reward_margin") == item.get("length_matched_mean_reward_margin"),
            "v0004 selection did not reproduce evaluation gate",
        )
        checkpoint = _mapping(evaluation.get("checkpoint"), "v0004 evaluation checkpoint")
        _require(
            checkpoint.get("training_receipt_sha256") == receipts_by_run[run_id]["sha"]
            and checkpoint.get("training_receipt_file_sha256") == receipts_by_run[run_id]["identity"]["file_sha256"],
            "v0004 evaluation checkpoint receipt drifted",
        )
        eval_claim = _mapping(evaluation.get("claim_boundary"), "v0004 evaluation claim")
        _require(eval_claim.get("dev_consumed") is False and eval_claim.get("heldout_consumed") is False, "v0004 evaluation crossed protected-data boundary")
        source_evaluations.append({
            **_identity(evaluation_path, content_field="evaluation_sha256"),
            "candidate_id": item["candidate_id"],
            "run_id": run_id,
            "checkpoint_step": step,
            "eligible": False,
        })
    _require(len(seen) == 4, "v0004 terminal grid is incomplete")

    datasets = _mapping(source.get("datasets"), "v0004 datasets")
    full_entry = _mapping(datasets.get("full_train"), "v0004 full_train")
    fit_entry = _mapping(datasets.get("fit"), "v0004 fit")
    search_entry = _mapping(datasets.get("search"), "v0004 search")
    full_path = Path(str(full_entry.get("path")))
    fit_path = Path(str(fit_entry.get("path")))
    old_search_path = Path(str(search_entry.get("path")))
    full_rows = _validate_dataset(full_path, full_entry, "v0004 full_train")
    fit_rows = _validate_dataset(fit_path, fit_entry, "v0004 fit")
    old_search_rows = _validate_dataset(old_search_path, search_entry, "v0004 search")
    full_ids = [str(row["pair_id"]) for row in full_rows]
    fit_ids = [str(row["pair_id"]) for row in fit_rows]
    old_search_ids = [str(row["pair_id"]) for row in old_search_rows]
    _require(len(full_rows) == 154 and len(fit_rows) == 124 and len(old_search_rows) == 30, "v0004 dataset counts drifted")
    _require(set(fit_ids).isdisjoint(old_search_ids) and set(fit_ids) | set(old_search_ids) == set(full_ids), "v0004 fit/search are not a full partition")
    _require([pair_id for pair_id in full_ids if pair_id in set(fit_ids)] == fit_ids, "v0004 fit order drifted from full train")
    _require([pair_id for pair_id in full_ids if pair_id in set(old_search_ids)] == old_search_ids, "v0004 search order drifted from full train")
    _require(set(OLD_MECHANISM_PAIR_IDS).issubset(fit_ids) and set(OLD_MECHANISM_PAIR_IDS).isdisjoint(old_search_ids), "historical mechanism rows drifted")
    for evaluation in source_evaluations:
        value = _load(Path(evaluation["path"]), "v0004 evaluation dataset check")
        _require(value.get("dataset", {}).get("ordered_pair_ids_sha256") == search_entry.get("ordered_pair_ids_sha256"), "v0004 evaluation did not use frozen search30")
        _require([row.get("pair_id") for row in value["pair_results"]] == old_search_ids, "v0004 evaluation pair order drifted")

    return {
        "campaign": source,
        "campaign_path": source_campaign_path,
        "campaign_sha256": source_sha,
        "selection": selection,
        "selection_path": source_selection_path.resolve(strict=True),
        "selection_sha256": selection_sha,
        "bootcamp": bootcamp,
        "charter": charter,
        "charter_path": charter_path,
        "event4": event4,
        "event4_path": event4_path,
        "strict": strict,
        "strict_path": strict_path,
        "strict_sha256": strict_sha,
        "strict_amendment": strict_amendment,
        "strict_amendment_path": strict_amendment_path,
        "search_claim": search_claim,
        "search_claim_path": search_claim_path,
        "receipts": source_receipts,
        "evaluations": source_evaluations,
        "full_rows": full_rows,
        "fit_rows": fit_rows,
        "old_search_rows": old_search_rows,
    }


def _rotate_rows(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_rows = source["full_rows"]
    prior_fit_ids = {str(row["pair_id"]) for row in source["fit_rows"]}
    old_search_ids = {str(row["pair_id"]) for row in source["old_search_rows"]}
    old = set(OLD_MECHANISM_PAIR_IDS)
    eligible = sorted(
        prior_fit_ids - old,
        key=lambda pair_id: (
            hashlib.sha256((SELECTION_SALT + pair_id).encode("utf-8")).hexdigest(),
            pair_id,
        ),
    )
    _require(len(eligible) == 120, "v0005 rotation population must be the 120 previously unevaluated non-mechanism rows")
    new_search_ids = set(eligible[:30])
    _require(new_search_ids.isdisjoint(old_search_ids | old), "v0005 search leaked previously evaluated or historical rows")
    new_search = [dict(row) for row in full_rows if str(row["pair_id"]) in new_search_ids]
    new_fit = [dict(row) for row in full_rows if str(row["pair_id"]) not in new_search_ids]
    _require(len(new_fit) == 124 and len(new_search) == 30, "v0005 rotated split counts drifted")
    new_fit_ids = {str(row["pair_id"]) for row in new_fit}
    _require(old_search_ids.issubset(new_fit_ids), "v0004 evaluated search rows were not moved into optimizer-only fit")
    _require(old.issubset(new_fit_ids), "historical mechanism rows were not retained in optimizer-only fit")
    protocol = {
        "authority": "v0004_fit_pair_ids_only_no_dev_or_heldout_rows",
        "selection_salt": SELECTION_SALT,
        "order": "sha256(campaign_id + '|' + pair_id), then pair_id; emitted in frozen full_train order",
        "rotation_population_records": 120,
        "search_prefix_records": 30,
        "fit_records": 124,
        "v0004_search_disposition": {
            "records": 30,
            "moved_to_optimizer_fit": True,
            "eligible_for_v0005_search_evaluation": False,
            "ordered_pair_ids_sha256": contract.object_sha256([str(row["pair_id"]) for row in source["old_search_rows"]]),
        },
        "historical_four_pair_disposition": {
            "pair_ids": list(OLD_MECHANISM_PAIR_IDS),
            "optimizer_input_in_fit": True,
            "eligible_for_search_selection_or_confirmation": False,
        },
        "v0005_search_prior_exposure": {
            "evaluated_before_v0005": False,
            "present_in_v0004_optimizer_fit": True,
            "v0005_training_starts_fresh_from_promoted_s1": True,
        },
    }
    return new_fit, new_search, protocol


def _lr_token(lr: float) -> str:
    if lr == 3.9e-6:
        return "3p9e_6"
    if lr == 4.1e-6:
        return "4p1e_6"
    raise V0005BuildError(f"unsupported frozen LR: {lr}")


def _run_id(lr: float, step: int | None = None) -> str:
    token = _lr_token(lr)
    return f"search_lr_{token}" if step is None else f"refit_lr_{token}_step_{step}"


def _candidate_id(lr: float, step: int) -> str:
    return f"candidate_lr_{_lr_token(lr)}_step_{step}"


def _build_configs_and_specs(
    source: Mapping[str, Any],
    run_root: Path,
    datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source_campaign = source["campaign"]
    source_specs = _mapping(source_campaign.get("run_specs"), "v0004 run specs")
    search_template_entry = _mapping(source_specs["search_lr_1e_6"]["executable_config"], "v0004 search template identity")
    search_template_path = Path(str(search_template_entry["path"])).resolve(strict=True)
    _require(day23_gpu.file_sha256(search_template_path) == search_template_entry.get("file_sha256"), "v0004 search template drifted")
    search_template = _load(search_template_path, "v0004 search config template")
    configs: dict[str, bytes] = {}
    specs: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    registry: list[dict[str, Any]] = []

    for lr in LEARNING_RATES:
        search_id = _run_id(lr)
        config = copy.deepcopy(search_template)
        config.update({
            "dataset": [str(datasets["fit"]["path"])],
            "learning_rate": lr,
            "seed": SEARCH_SEED,
            "data_seed": SEARCH_SEED,
            "max_steps": SEARCH_MAX_STEPS,
            "save_steps": SEARCH_SAVE_STEPS,
            "save_total_limit": SEARCH_SAVE_TOTAL_LIMIT,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "output_dir": str(run_root / "outputs" / search_id),
        })
        payload = _bytes(config)
        filename = CONFIG_FILENAMES[search_id]
        path = run_root / "binding/configs" / filename
        identity = {"path": str(path), "file_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "keys": sorted(config)}
        configs[search_id] = payload
        inventory[f"binding/configs/{filename}"] = {key: identity[key] for key in ("file_sha256", "bytes")}
        specs[search_id] = {
            "role": "search_train",
            "authorized": True,
            "dataset_key": "fit",
            "dataset": dict(datasets["fit"]),
            "executable_config": identity,
            "model_argument": str(config["model"]),
            "runtime": {
                "world_size": WORLD_SIZE,
                "per_device_train_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                "nominal_global_train_batch_size": GLOBAL_BATCH,
            },
            "learning_rate": lr,
            "seed": SEARCH_SEED,
            "max_steps": SEARCH_MAX_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "output_dir": str(run_root / "outputs" / search_id),
            "success_receipt": str(run_root / "evidence" / search_id / "success-receipt.json"),
            "failure_receipt": str(run_root / "evidence" / search_id / "failure-receipt.json"),
            "fresh_start_from_parent": True,
            "authorization": {"state": "authorized_after_gpu_binding", "requires_selection_receipt": False},
            "checkpoint_capture": {
                "mode": "periodic_save_with_terminal_retention",
                "save_steps": SEARCH_SAVE_STEPS,
                "save_total_limit": SEARCH_SAVE_TOTAL_LIMIT,
                "candidate_checkpoint_steps": list(CHECKPOINT_STEPS),
                "expected_retained_checkpoint_steps": [18, 20, 22, 24, 26, 28, 30],
                "retained_noncandidate_checkpoint_steps": [22, 24, 26, 28, 30],
                "retained_noncandidate_checkpoints_candidate_eligible": False,
                "training_continues_to_max_steps_for_v0004_cosine_horizon_parity": True,
            },
        }

        for step in CHECKPOINT_STEPS:
            refit_id = _run_id(lr, step)
            source_refit_id = f"refit_lr_1e_6_step_{15 if step == 18 else 30}"
            source_refit_entry = _mapping(source_specs[source_refit_id]["executable_config"], f"v0004 refit template identity {source_refit_id}")
            source_refit_path = Path(str(source_refit_entry["path"])).resolve(strict=True)
            _require(day23_gpu.file_sha256(source_refit_path) == source_refit_entry.get("file_sha256"), f"v0004 refit template drifted: {source_refit_id}")
            refit = _load(source_refit_path, f"v0004 refit template {source_refit_id}")
            refit.update({
                "dataset": [str(datasets["full_train"]["path"])],
                "learning_rate": lr,
                "seed": REFIT_SEED,
                "data_seed": REFIT_SEED,
                "max_steps": step,
                "save_steps": step,
                "save_total_limit": 1,
                "per_device_train_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                "output_dir": str(run_root / "outputs" / refit_id),
            })
            refit_payload = _bytes(refit)
            refit_filename = CONFIG_FILENAMES[refit_id]
            refit_path = run_root / "binding/configs" / refit_filename
            refit_identity = {"path": str(refit_path), "file_sha256": hashlib.sha256(refit_payload).hexdigest(), "bytes": len(refit_payload), "keys": sorted(refit)}
            configs[refit_id] = refit_payload
            inventory[f"binding/configs/{refit_filename}"] = {key: refit_identity[key] for key in ("file_sha256", "bytes")}
            candidate_id = _candidate_id(lr, step)
            specs[refit_id] = {
                "role": "unique_finalist_refit_option",
                "authorized": False,
                "dataset_key": "full_train",
                "dataset": dict(datasets["full_train"]),
                "executable_config": refit_identity,
                "model_argument": str(refit["model"]),
                "runtime": {
                    "world_size": WORLD_SIZE,
                    "per_device_train_batch_size": PER_DEVICE_BATCH,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                    "nominal_global_train_batch_size": GLOBAL_BATCH,
                },
                "learning_rate": lr,
                "seed": REFIT_SEED,
                "max_steps": step,
                "checkpoint_steps": [step],
                "output_dir": str(run_root / "outputs" / refit_id),
                "success_receipt": str(run_root / "evidence" / refit_id / "success-receipt.json"),
                "failure_receipt": str(run_root / "evidence" / refit_id / "failure-receipt.json"),
                "fresh_start_from_parent": True,
                "authorization": {
                    "state": "conditional_exactly_one",
                    "requires_selected_candidate_id": candidate_id,
                    "search_checkpoint_itself_is_forbidden_as_refit_initialization": True,
                },
            }
            registry.append({
                "candidate_id": candidate_id,
                "search_run_spec": search_id,
                "checkpoint_step": step,
                "learning_rate": lr,
                "refit_run_spec": refit_id,
            })
    _require(set(configs) == set(CONFIG_FILENAMES), "v0005 config registry drifted")
    return configs, dict(sorted(specs.items())), registry, inventory


def _control_artifacts(
    *,
    source: Mapping[str, Any],
    run_root: Path,
    data_partition: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    producers: Mapping[str, Path],
) -> dict[str, Any]:
    bootcamp: Path = source["bootcamp"]
    root = bootcamp / CONTROL_ROOT_REL
    charter = source["charter"]
    event4 = source["event4"]
    event4_path: Path = source["event4_path"]
    producer_ids = {name: _identity(path, relative_to=bootcamp) for name, path in sorted(producers.items())}
    created_at = str(source["receipts"][-1] and _load(Path(source["receipts"][-1]["path"]), "source completion anchor").get("completed_at_utc"))
    _require(created_at and created_at != "None", "source terminal time anchor missing")

    source_terminal = {
        "campaign_path": str(source["campaign_path"]),
        "campaign_file_sha256": day23_gpu.file_sha256(source["campaign_path"]),
        "campaign_sha256": source["campaign_sha256"],
        "selection_path": str(source["selection_path"]),
        "selection_file_sha256": day23_gpu.file_sha256(source["selection_path"]),
        "selection_sha256": source["selection_sha256"],
        "status": "closed_no_candidate",
        "eligible_count": 0,
        "training_receipts": source["receipts"],
        "evaluations": source["evaluations"],
        "search_claim": _identity(source["search_claim_path"], content_field="claim_sha256"),
    }
    strict_identity = _identity(source["strict_path"], content_field="validation_sha256")
    amendment2_identity = _identity(source["strict_amendment_path"], content_field="amendment_sha256", relative_to=bootcamp)
    amendment_path = root / "amendments/amendment-000003-v0005-terminal-interpolation.json"
    amendment = v4._write_or_verify(
        amendment_path,
        {
            "schema_name": "rsi.day23_dpo_append_only_amendment",
            "schema_version": 1,
            "status": "frozen",
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "amendment_id": "amendment-000003-v0005-terminal-interpolation",
            "created_at_utc": created_at,
            "authority": {
                "charter_sha256": charter["charter_sha256"],
                "prior_event_000004_sha256": event4["event_sha256"],
                "source_strict_preunseal": strict_identity,
                "source_strict_preunseal_amendment": amendment2_identity,
            },
            "source_terminal": source_terminal,
            "authorized_change": {
                "primary_lever_id": SCIENTIFIC_LEVER,
                "dependent_lever_id": "candidate_checkpoint_observation_schedule",
                "dependent_lever_is_inseparable_from_primary_interpolation": True,
                "learning_rates": {"from": [1e-6, 5e-6], "to": list(LEARNING_RATES)},
                "checkpoint_steps": {"from": [15, 30], "to": list(CHECKPOINT_STEPS)},
                "search_max_steps": SEARCH_MAX_STEPS,
                "search_save_steps": SEARCH_SAVE_STEPS,
                "search_save_total_limit": SEARCH_SAVE_TOTAL_LIMIT,
                "cosine_schedule_horizon_inherited_from_v0004": True,
                "split_rotation": {
                    "selection_salt": SELECTION_SALT,
                    "new_fit": dict(datasets["fit"]),
                    "new_search": dict(datasets["search"]),
                    "old_search_disposition": data_partition["v0004_search_disposition"],
                },
            },
            "scientific_basis": {
                "method": "bilinear_interpolation_of_four_v0004_search_aggregates",
                "prediction": "threshold island near lr 4e-6 and optimizer step 20",
                "new_search_rows_observed_before_precommit": False,
                "adaptive_overfit_risk": {
                    "acknowledged": True,
                    "source": "candidate_grid_was_derived_from_consumed_v0004_search_results",
                    "mitigation": "mitigated_only_by_rotated_previously_unobserved_search30",
                    "risk_eliminated": False,
                },
            },
            "frozen_invariants": {
                "world_size": WORLD_SIZE,
                "per_device_train_batch_size": PER_DEVICE_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
                "nominal_global_train_batch_size": GLOBAL_BATCH,
                "minimum_free_memory_fraction": MIN_FREE_FRACTION,
                "search_seed": SEARCH_SEED,
                "refit_seed": REFIT_SEED,
                "fresh_s1_only": True,
                "resume_from_v0004_checkpoint": False,
                "dev_and_heldout_leases_unchanged": True,
            },
            "producers": producer_ids,
        },
        "amendment_sha256",
    )
    amendment_id = _identity(amendment_path, content_field="amendment_sha256", relative_to=bootcamp)

    event5_path = root / "events/event-000005-version-closed-no-candidate.json"
    event5 = v4._write_or_verify(
        event5_path,
        {
            "schema_name": "rsi.day23_dpo_event",
            "schema_version": 1,
            "event_id": "event-000005-version-closed-no-candidate",
            "created_at_utc": created_at,
            "sequence": 5,
            "event_type": "version-closed-no-candidate",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": SOURCE_VERSION,
            "state_before": "version_precommitted",
            "state_after": "iteration_open",
            "prior_event": {
                "path": str(event4_path.relative_to(root)),
                "file_sha256": day23_gpu.file_sha256(event4_path),
                "event_sha256": event4["event_sha256"],
                "sequence": 4,
            },
            "authority_refs": [amendment_id],
            "payload": {
                "source_terminal": source_terminal,
                "candidate_eligible": False,
                "checkpoint_resume_forbidden": True,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    event5_id = _identity(event5_path, content_field="event_sha256", relative_to=bootcamp)

    source_version_path = root / "versions/rsi-v0004/version.json"
    source_version = _load(source_version_path, "v0004 version")
    _verify(source_version, "version_sha256", "v0004 version")
    version = copy.deepcopy(source_version)
    version.pop("version_sha256", None)
    version.update({
        "version_id": VERSION_ID,
        "status": "precommitted",
        "created_at_utc": created_at,
        "intervention": {
            "primary_lever": SCIENTIFIC_LEVER,
            "changed_levers": [
                SCIENTIFIC_LEVER,
                "candidate_checkpoint_observation_schedule",
            ],
            "dependent_lever": "candidate_checkpoint_observation_schedule",
            "dependent_lever_is_inseparable_from_primary_interpolation": True,
            "single_scientific_change_claimed": False,
            "candidate_values": list(LEARNING_RATES),
            "checkpoint_measurements": list(CHECKPOINT_STEPS),
            "prediction": "v0004 interpolation predicts a narrow response island around lr 4e-6 at step 20",
            "falsifier": "none of the four rotated-search candidates passes the frozen search30 gate",
            "adaptive_overfit_risk": {
                "acknowledged": True,
                "source": "candidate_grid_was_derived_from_consumed_v0004_search_results",
                "mitigation": "mitigated_only_by_rotated_previously_unobserved_search30",
                "risk_eliminated": False,
            },
        },
        "campaign": {
            "lr_grid_optimizer_train": {
                "records": datasets["fit"]["records"],
                "ordered_pair_ids_sha256": datasets["fit"]["ordered_pair_ids_sha256"],
                "role": "lr_grid_optimizer_train_rotated_unseen_search_excluded",
            },
            "lr_grid_search_evaluation": {
                "records": datasets["search"]["records"],
                "ordered_pair_ids_sha256": datasets["search"]["ordered_pair_ids_sha256"],
                "role": "new_unobserved_search30_evaluation_only",
            },
            "candidate_values": [
                {"candidate_id": _candidate_id(lr, step), "learning_rate": lr, "checkpoint_step": step}
                for lr in LEARNING_RATES
                for step in CHECKPOINT_STEPS
            ],
            "refit_fresh_from_S1": True,
            "refit_source": "exactly_one_frozen_lr_and_checkpoint_step_finalist",
            "independent_seed_refit": {
                "records": datasets["full_train"]["records"],
                "ordered_pair_ids_sha256": datasets["full_train"]["ordered_pair_ids_sha256"],
                "role": "unique_finalist_independent_seed_optimizer_train",
            },
        },
        "fixed_operational_constraint": {
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "nominal_global_train_batch_size": GLOBAL_BATCH,
            "minimum_free_memory_fraction": MIN_FREE_FRACTION,
        },
        "fresh_start": {
            "from_promoted_s1": True,
            "resume_from_v0004_checkpoint": False,
            "source_v0004_checkpoints_candidate_eligible": False,
        },
        "supersedes": source_terminal,
        "amendment": amendment_id,
        "stop_rules": [
            "reject_v0005_if_no_rotated_search_candidate_passes_search30",
            "stop_entire_charter_on_any_failure_after_dev_claim",
            "stop_entire_charter_on_guardrail_or_heldout_failure",
            "stop_entire_charter_when_rsi_v0005_is_exhausted",
        ],
    })
    version["frozen_recipe"]["max_optimizer_steps"] = SEARCH_MAX_STEPS
    version["frozen_recipe"]["search_seed"] = SEARCH_SEED
    version["frozen_recipe"]["independent_refit_seed"] = REFIT_SEED
    version_path = root / "versions/rsi-v0005/version.json"
    version = v4._write_or_verify(version_path, version, "version_sha256")
    version_id = _identity(version_path, content_field="version_sha256", relative_to=bootcamp)

    event6_path = root / "events/event-000006-version-precommitted.json"
    event6 = v4._write_or_verify(
        event6_path,
        {
            "schema_name": "rsi.day23_dpo_event",
            "schema_version": 1,
            "event_id": "event-000006-version-precommitted",
            "created_at_utc": created_at,
            "sequence": 6,
            "event_type": "version-precommitted",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "state_before": "iteration_open",
            "state_after": "version_precommitted",
            "prior_event": {
                "path": str(event5_path.relative_to(root)),
                "file_sha256": day23_gpu.file_sha256(event5_path),
                "event_sha256": event5["event_sha256"],
                "sequence": 5,
            },
            "authority_refs": [amendment_id, version_id],
            "payload": {
                "runtime_profile": "world2_B8_GA2_global32",
                "learning_rate_grid": list(LEARNING_RATES),
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "search_split_rotated": True,
                "training_started": False,
                "search_rows_opened": 0,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    return {
        "amendment": amendment_id,
        "source_closed_event": event5_id,
        "version": version_id,
        "precommit_event": _identity(event6_path, content_field="event_sha256", relative_to=bootcamp),
        "source_strict_preunseal": strict_identity,
        "strict_preunseal_requirement": {
            "required_before_first_optimizer": True,
            "require_search_unopened": True,
            "validator": _identity(producers["authority_validator"]),
            "receipt_path": str(run_root / "evidence/authority/strict-preunseal.json"),
            "receipt_schema_name": "day23.rsi_v0005_strict_authority_preflight",
            "receipt_self_hash_field": "validation_sha256",
            "amendment_path": str(CONTROL_ROOT_REL / "amendments/amendment-000004-v0005-strict-preunseal.json"),
            "amendment_id": "amendment-000004-v0005-strict-preunseal",
            "amendment_self_hash_field": "amendment_sha256",
        },
    }


def _prepared(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_terminal(args.source_campaign, args.source_selection)
    run_root = args.run_root.expanduser().resolve(strict=True)
    _require(run_root.is_dir() and not run_root.is_symlink(), "v0005 run root must be a real directory")
    if args.mode != "check":
        _require(not any(run_root.iterdir()), "new v0005 run root must be empty")
    fit_rows, search_rows, data_partition = _rotate_rows(source)
    fit_payload = contract.jsonl_bytes(fit_rows)
    search_payload = contract.jsonl_bytes(search_rows)
    fit_path = run_root / "binding/data/day23-qwen35-dpo-rsi-v0005-fit.jsonl"
    search_path = run_root / "binding/data/day23-qwen35-dpo-rsi-v0005-search.jsonl"
    source_datasets = _mapping(source["campaign"].get("datasets"), "v0004 datasets")
    datasets = {
        "fit": _dataset_identity(fit_path, fit_payload, fit_rows, "search_optimizer_input"),
        "search": _dataset_identity(search_path, search_payload, search_rows, "locked_rotated_hyperparameter_ranking_only"),
        "full_train": copy.deepcopy(source_datasets["full_train"]),
        "dev": copy.deepcopy(source_datasets["dev"]),
        "heldout": copy.deepcopy(source_datasets["heldout"]),
    }
    configs, specs, registry, inventory = _build_configs_and_specs(source, run_root, datasets)
    inventory.update({
        "binding/data/day23-qwen35-dpo-rsi-v0005-fit.jsonl": {"file_sha256": datasets["fit"]["file_sha256"], "bytes": len(fit_payload)},
        "binding/data/day23-qwen35-dpo-rsi-v0005-search.jsonl": {"file_sha256": datasets["search"]["file_sha256"], "bytes": len(search_payload)},
    })
    python_path = Path(str(source["campaign"]["runtime_parse"]["python_executable"]["path"])).resolve(strict=True)
    checkout = Path(str(source["campaign"]["runtime_parse"]["ms_swift_checkout"])).resolve(strict=True)
    runtime_parse = v4._parse_configs(configs, python_path, checkout, source["campaign"]["runtime_parse"]["package_versions"])
    script_dir = Path(__file__).resolve().parent
    producers = {
        "builder": Path(__file__).resolve(),
        "runner": script_dir / "run_day23_rsi_candidate.py",
        "evaluator": script_dir / "eval_day23_rsi_candidate.py",
        "authority_validator": script_dir / "validate_day23_rsi_v0005_authority.py",
    }
    for name, path in producers.items():
        _require(path.is_file() and not path.is_symlink(), f"missing producer: {name}")
    return {
        "source": source,
        "run_root": run_root,
        "fit_payload": fit_payload,
        "search_payload": search_payload,
        "datasets": datasets,
        "data_partition": data_partition,
        "configs": configs,
        "specs": specs,
        "registry": registry,
        "inventory": dict(sorted(inventory.items())),
        "runtime_parse": runtime_parse,
        "producers": producers,
    }


def _campaign(prepared: Mapping[str, Any], control: Mapping[str, Any], producers: Mapping[str, Path]) -> dict[str, Any]:
    source = prepared["source"]
    source_campaign = source["campaign"]
    run_root: Path = prepared["run_root"]
    campaign = copy.deepcopy(source_campaign)
    campaign.pop("campaign_sha256", None)
    campaign.update({
        "schema_name": GPU_SCHEMA,
        "schema_version": 1,
        "status": "gpu_execution_bound_optimizer_pending",
        "campaign_id": CAMPAIGN_ID,
        "version_id": VERSION_ID,
        "remote_run_root": str(run_root),
        "control_extension": dict(control),
        "producers": {name: _identity(path) for name, path in sorted(producers.items())},
        "runtime_parse": prepared["runtime_parse"],
        "datasets": prepared["datasets"],
        "data_partition": prepared["data_partition"],
        "run_specs": prepared["specs"],
        "candidate_registry": prepared["registry"],
        "artifact_inventory": prepared["inventory"],
        "supersedes": {
            "campaign_path": str(source["campaign_path"]),
            "campaign_file_sha256": day23_gpu.file_sha256(source["campaign_path"]),
            "campaign_sha256": source["campaign_sha256"],
            "selection_path": str(source["selection_path"]),
            "selection_file_sha256": day23_gpu.file_sha256(source["selection_path"]),
            "selection_sha256": source["selection_sha256"],
            "status": "closed_no_candidate",
            "eligible_count": 0,
            "v0004_checkpoint_candidate_or_resume": False,
        },
    })
    campaign["fixed_recipe"].pop("single_scientific_lever", None)
    campaign["fixed_recipe"].update({
        "primary_scientific_lever": SCIENTIFIC_LEVER,
        "dependent_lever": "candidate_checkpoint_observation_schedule",
        "dependent_lever_is_inseparable_from_primary_interpolation": True,
        "single_scientific_change_claimed": False,
        "lever_levels": list(LEARNING_RATES),
        "search_seed": SEARCH_SEED,
        "refit_seed": REFIT_SEED,
    })
    campaign["fixed_recipe"]["runtime"].update({
        "world_size": WORLD_SIZE,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "nominal_global_train_batch_size": GLOBAL_BATCH,
        "min_free_memory_fraction": MIN_FREE_FRACTION,
    })
    campaign["fixed_recipe"]["topology"].update({
        "world_size": WORLD_SIZE,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "nominal_global_train_batch_size": GLOBAL_BATCH,
    })
    bootcamp: Path = source["bootcamp"]
    campaign["implementation_sources"].update({
        "builder_and_binder": {**_identity(producers["builder"], relative_to=bootcamp), "path": str(producers["builder"].relative_to(bootcamp))},
        "training_runner": {**_identity(producers["runner"], relative_to=bootcamp), "path": str(producers["runner"].relative_to(bootcamp))},
        "candidate_evaluator": {**_identity(producers["evaluator"], relative_to=bootcamp), "path": str(producers["evaluator"].relative_to(bootcamp))},
        "authority_validator": {**_identity(producers["authority_validator"], relative_to=bootcamp), "path": str(producers["authority_validator"].relative_to(bootcamp))},
    })
    campaign["access_ledger"]["search_claim"] = str(run_root / "evidence/search-selection/search-claim.json")
    campaign["retry_policy"].update({
        "supersession_class": "new_terminal_semantic_version_not_infrastructure_retry",
        "v0004_checkpoints": "diagnostic_only_never_candidate_or_resume",
    })
    campaign["budgets"].update({
        "maximum_optimizer_steps": 90,
        "maximum_search_training_runs": 2,
        "maximum_refit_training_runs": 1,
        "maximum_search_checkpoint_evaluations": 4,
        "prior_optimizer_steps_observed_v0003_v0004": 90,
        "maximum_cumulative_optimizer_steps_if_v0005_refit_runs": 170,
        "charter_cumulative_authorized_optimizer_step_ceiling": 180,
    })
    campaign["claim_boundary"].update({
        "optimizer_step_run": False,
        "search_or_dev_consumed": False,
        "heldout_consumed": False,
        "source_v0004_search_consumed": True,
        "v0005_rotated_search_consumed": False,
    })
    campaign["campaign_sha256"] = day23_gpu.object_sha256(campaign)
    return campaign


def _check_built(prepared: Mapping[str, Any]) -> dict[str, Any]:
    run_root: Path = prepared["run_root"]
    campaign_path = run_root / "binding/gpu-campaign.json"
    campaign = _load(campaign_path, "v0005 campaign")
    campaign_sha = _verify(campaign, "campaign_sha256", "v0005 campaign")
    _require(campaign.get("schema_name") == GPU_SCHEMA and campaign.get("version_id") == VERSION_ID, "v0005 campaign identity drifted")
    _require(campaign.get("remote_run_root") == str(run_root), "v0005 campaign run root drifted")
    for key, entry in prepared["inventory"].items():
        path = run_root / key
        _require(path.is_file() and day23_gpu.file_sha256(path) == entry["file_sha256"] and path.stat().st_size == entry["bytes"], f"built artifact drifted: {key}")
    control = _mapping(campaign.get("control_extension"), "v0005 control extension")
    for name, field in (("amendment", "amendment_sha256"), ("source_closed_event", "event_sha256"), ("version", "version_sha256"), ("precommit_event", "event_sha256")):
        _bound_file(prepared["source"]["bootcamp"], control.get(name), field, f"v0005 {name}")
    _require(campaign.get("datasets") == prepared["datasets"], "v0005 dataset bindings drifted")
    _require(campaign.get("run_specs") == prepared["specs"], "v0005 run specs drifted")
    return {"status": "pass", "campaign": str(campaign_path), "campaign_sha256": campaign_sha, "version_id": VERSION_ID}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight_only:
        args.mode = "preflight"
    prepared = _prepared(args)
    if args.mode == "preflight":
        return {
            "status": "preflight_pass",
            "version_id": VERSION_ID,
            "source_status": "closed_no_candidate",
            "configs_parsed": len(prepared["configs"]),
            "learning_rates": list(LEARNING_RATES),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "search_max_steps": SEARCH_MAX_STEPS,
            "rotated_fit_records": prepared["datasets"]["fit"]["records"],
            "unopened_search_records": prepared["datasets"]["search"]["records"],
            "fit_file_sha256": prepared["datasets"]["fit"]["file_sha256"],
            "fit_ordered_pair_ids_sha256": prepared["datasets"]["fit"]["ordered_pair_ids_sha256"],
            "search_file_sha256": prepared["datasets"]["search"]["file_sha256"],
            "search_ordered_pair_ids_sha256": prepared["datasets"]["search"]["ordered_pair_ids_sha256"],
            "runtime_profile": "world2_B8_GA2_global32",
            "minimum_free_memory_fraction": MIN_FREE_FRACTION,
            "dev_opened": False,
            "heldout_opened": False,
            "producer_file_sha256": {
                name: day23_gpu.file_sha256(path)
                for name, path in sorted(prepared["producers"].items())
            },
        }
    if args.mode == "check":
        return _check_built(prepared)

    producers = prepared["producers"]
    control = _control_artifacts(
        source=prepared["source"],
        run_root=prepared["run_root"],
        data_partition=prepared["data_partition"],
        datasets=prepared["datasets"],
        producers=producers,
    )
    campaign = _campaign(prepared, control, producers)
    run_root: Path = prepared["run_root"]
    v4._write_exclusive(run_root / "binding/data/day23-qwen35-dpo-rsi-v0005-fit.jsonl", prepared["fit_payload"])
    v4._write_exclusive(run_root / "binding/data/day23-qwen35-dpo-rsi-v0005-search.jsonl", prepared["search_payload"])
    for run_id, payload in prepared["configs"].items():
        v4._write_exclusive(Path(prepared["specs"][run_id]["executable_config"]["path"]), payload)
    campaign_path = run_root / "binding/gpu-campaign.json"
    v4._write_exclusive(campaign_path, _bytes(campaign))
    return {
        "status": "gpu_execution_bound_optimizer_pending",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "version_id": VERSION_ID,
        "learning_rates": list(LEARNING_RATES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="build")
    parser.add_argument("--preflight-only", action="store_true", help="compatibility alias for --mode preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = execute(build_parser().parse_args(argv))
    except Exception as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
