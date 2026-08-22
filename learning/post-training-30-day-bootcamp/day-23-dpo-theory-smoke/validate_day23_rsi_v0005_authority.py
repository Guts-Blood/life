#!/usr/bin/env python3
"""Strict pre-unseal authority validation for Day 23 RSI-v0005.

This validator is deliberately independent from the v0005 builder.  Before an
optimizer is allowed to run it recomputes the terminal v0004 search result, the
rotated fit/search partition, the append-only event chain, and the protected
lease state.  A passing invocation seals both an append-only validator
amendment and a run-local pre-unseal receipt with exclusive creation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract
import eval_day23_qwen35_preferences as preference_eval
import run_day23_qwen35_gpu_stage as day23_gpu
import run_day23_rsi_candidate as rsi_runner


GOAL_ID = "goal-0002-day23-dpo"
SOURCE_VERSION = "rsi-v0004"
VERSION_ID = "rsi-v0005"
SOURCE_SCHEMA = "day23.rsi_v0004_gpu_campaign"
CAMPAIGN_SCHEMA = "day23.rsi_v0005_gpu_campaign"
CONTROL_ROOT_REL = Path("rsi-control/charters/goal-0002-day23-dpo")
AMENDMENT4_REL = CONTROL_ROOT_REL / "amendments/amendment-000004-v0005-strict-preunseal.json"
LEARNING_RATES = [3.9e-6, 4.1e-6]
CHECKPOINT_STEPS = [18, 20]
MAX_STEPS = 30
WORLD_SIZE = 2
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION = 2
GLOBAL_BATCH = 32
MIN_FREE_FRACTION = 0.20
SEARCH_RECORDS = 30
FIT_RECORDS = 124
FULL_RECORDS = 154
ROTATION_ELIGIBLE_RECORDS = 120
ROTATION_SALT = "day23-qwen35-dpo-rsi-v0005|"
OLD_PAIR_IDS = tuple(preference_eval.MECHANISM_PAIR_IDS)


class AuthorityError(RuntimeError):
    """The v0005 authority chain or one of its bound inputs is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value) and "\x00" not in value, f"{label} must be non-empty text")
    return value


def integer(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AuthorityError(f"cannot read {label}: {path}") from error
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        require(bool(line), f"blank line in {label}:{index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuthorityError(f"invalid JSON in {label}:{index}") from error
        require(isinstance(value, dict), f"non-object row in {label}:{index}")
        rows.append(value)
    return rows


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label}.{field} drifted")
    return expected


def identity(path: Path, field: str | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    result: dict[str, Any] = {
        "path": str(path),
        "file_sha256": day23_gpu.file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if field is not None:
        result["content_sha256"] = verify(load(path, str(path)), field, str(path))
    return result


def resolve_path(value: Any, label: str, *, base: Path | None = None, must_exist: bool = True) -> Path:
    raw = Path(text(value, label)).expanduser()
    if not raw.is_absolute():
        require(base is not None, f"{label} is relative without a base")
        raw = base / raw
    return raw.resolve(strict=must_exist)


def resolve_bound(
    entry: Any,
    label: str,
    *,
    base: Path | None = None,
    field: str | None = None,
) -> tuple[dict[str, Any] | None, Path]:
    item = mapping(entry, label)
    path = resolve_path(item.get("path"), f"{label}.path", base=base)
    require(day23_gpu.file_sha256(path) == item.get("file_sha256"), f"{label} file hash drifted")
    if "bytes" in item:
        require(path.stat().st_size == item.get("bytes"), f"{label} byte count drifted")
    if field is None:
        return None, path
    value = load(path, label)
    require(verify(value, field, label) == item.get("content_sha256"), f"{label} content hash drifted")
    return value, path


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_exclusive(path: Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed[field] = day23_gpu.object_sha256(sealed)
    payload = _json_bytes(sealed)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
    return sealed


def _dataset(
    entry: Any,
    label: str,
    *,
    bootcamp: Path,
    allowed_roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], Path]:
    item = mapping(entry, label)
    path = resolve_path(item.get("path"), f"{label}.path", base=bootcamp)
    require(any(path.is_relative_to(root) for root in allowed_roots), f"{label} escaped allowed roots")
    require(day23_gpu.file_sha256(path) == item.get("file_sha256"), f"{label} file hash drifted")
    if "bytes" in item:
        require(path.stat().st_size == item.get("bytes"), f"{label} bytes drifted")
    rows = load_jsonl(path, label)
    require(len(rows) == item.get("records"), f"{label} records drifted")
    ids: list[str] = []
    hashes: list[str] = []
    for row in rows:
        pair_id = text(row.get("pair_id"), f"{label} pair_id")
        try:
            day23_contract.verify_self_hash(row, "row_sha256", f"{label} row {pair_id}")
        except day23_contract.Day23ContractError as error:
            raise AuthorityError(str(error)) from error
        ids.append(pair_id)
        hashes.append(text(row.get("row_sha256"), f"{label} row hash"))
    require(len(ids) == len(set(ids)), f"{label} pair IDs are not unique")
    require(day23_contract.object_sha256(ids) == item.get("ordered_pair_ids_sha256"), f"{label} ordered IDs drifted")
    require(day23_contract.object_sha256(hashes) == item.get("ordered_row_hashes_sha256"), f"{label} ordered row hashes drifted")
    return rows, path


def _expected_gate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    overall = mapping(aggregate.get("overall"), "source evaluation aggregate.overall")
    matched = mapping(aggregate.get("length_matched"), "source evaluation aggregate.length_matched")
    pairs = integer(overall.get("pairs"), "source evaluation pairs")
    wins = integer(overall.get("wins"), "source evaluation wins")
    mean = float(overall.get("mean_reward_margin"))
    matched_mean = float(matched.get("mean_reward_margin"))
    require(all(math.isfinite(number) for number in (mean, matched_mean)), "source evaluation aggregate is non-finite")
    return {
        "passed": pairs == SEARCH_RECORDS and wins >= 20 and mean > 0 and matched_mean > 0,
        "thresholds": {
            "pairs": SEARCH_RECORDS,
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


def _validate_source_receipt(
    source: Mapping[str, Any],
    source_path: Path,
    run_id: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    success = resolve_path(spec.get("success_receipt"), f"source {run_id} success receipt")
    failure = resolve_path(spec.get("failure_receipt"), f"source {run_id} failure receipt", must_exist=False)
    require(not failure.exists(), f"source search run has a failure receipt: {run_id}")
    receipt = load(success, f"source {run_id} success receipt")
    verify(receipt, "receipt_sha256", f"source {run_id} success receipt")
    strict = rsi_runner.validate_training_receipt_value(source, source_path, receipt, success)
    require(strict.get("run_id") == run_id and strict.get("spec") == spec, f"source {run_id} strict receipt projection drifted")
    require(receipt.get("schema_name") == "day23.rsi_v0004_training_receipt" and receipt.get("status") == "pass", f"source {run_id} receipt identity drifted")
    require(receipt.get("observations", {}).get("global_step") == MAX_STEPS, f"source {run_id} did not complete 30 steps")
    claim = mapping(receipt.get("claim_boundary"), f"source {run_id} claim")
    require(claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, f"source {run_id} consumed protected data")
    return receipt, success


def _validate_source_terminal(
    source: Mapping[str, Any],
    source_path: Path,
    source_selection_path: Path,
    *,
    bootcamp: Path,
) -> dict[str, Any]:
    source_sha = verify(source, "campaign_sha256", "v0004 campaign")
    require(
        source.get("schema_name") == SOURCE_SCHEMA
        and source.get("schema_version") == 1
        and source.get("version_id") == SOURCE_VERSION
        and source.get("goal_id") == GOAL_ID,
        "v0004 campaign identity drifted",
    )
    source_root = resolve_path(source.get("remote_run_root"), "v0004 run root")
    require(source_path.is_relative_to(source_root), "v0004 campaign escaped its run root")
    source_producers = mapping(source.get("producers"), "v0004 producers")

    runs = mapping(source.get("run_specs"), "v0004 run specs")
    search_specs = [
        (run_id, mapping(spec, f"v0004 run {run_id}"))
        for run_id, spec in sorted(runs.items())
        if isinstance(spec, Mapping) and spec.get("role") == "search_train"
    ]
    require(len(search_specs) == 2, "v0004 search trajectory count drifted")
    receipts: dict[str, tuple[dict[str, Any], Path]] = {}
    for run_id, spec in search_specs:
        receipts[run_id] = _validate_source_receipt(source, source_path, run_id, spec)

    datasets = mapping(source.get("datasets"), "v0004 datasets")
    source_search, source_search_path = _dataset(
        datasets.get("search"),
        "v0004 search",
        bootcamp=bootcamp,
        allowed_roots=(bootcamp, source_root),
    )
    require(len(source_search) == SEARCH_RECORDS, "v0004 search count drifted")
    expected_ids = [row["pair_id"] for row in source_search]

    expected_selection_path = source_root / "evidence/selection/search-selection.json"
    source_selection_path = source_selection_path.resolve(strict=True)
    require(source_selection_path == expected_selection_path, "v0004 selection path drifted")
    selection = load(source_selection_path, "v0004 search selection")
    selection_sha = verify(selection, "selection_sha256", "v0004 search selection")
    require(
        selection.get("schema_name") == "day23.rsi_v0004_search_selection"
        and selection.get("schema_version") == 1
        and selection.get("campaign_sha256") == source_sha,
        "v0004 selection campaign binding drifted",
    )
    candidates = selection.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 4, "v0004 candidate inventory drifted")

    recomputed_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        candidate = mapping(candidate, "v0004 selection candidate")
        run_id = text(candidate.get("run_id"), "v0004 candidate run_id")
        step = integer(candidate.get("checkpoint_step"), "v0004 candidate checkpoint")
        require(run_id in receipts and step in [15, 30], "v0004 candidate escaped frozen grid")
        require((run_id, step) not in seen, "duplicate v0004 candidate")
        seen.add((run_id, step))
        evaluation_path = resolve_path(candidate.get("path"), "v0004 evaluation path")
        expected_path = source_root / "evidence/evaluations" / f"{run_id}-checkpoint-{step}-search.json"
        require(evaluation_path == expected_path, "v0004 evaluation path drifted")
        require(day23_gpu.file_sha256(evaluation_path) == candidate.get("file_sha256"), "v0004 evaluation file hash drifted")
        evaluation = load(evaluation_path, "v0004 search evaluation")
        evaluation_sha = verify(evaluation, "evaluation_sha256", "v0004 search evaluation")
        require(evaluation_sha == candidate.get("evaluation_sha256"), "v0004 evaluation content binding drifted")
        require(
            evaluation.get("schema_name") == "day23.rsi_v0004_preference_evaluation"
            and evaluation.get("campaign_sha256") == source_sha
            and evaluation.get("run_id") == run_id
            and evaluation.get("checkpoint_step") == step
            and evaluation.get("split") == "search",
            "v0004 evaluation identity drifted",
        )
        producer = mapping(evaluation.get("producer"), "v0004 evaluation producer")
        require(producer.get("file_sha256") == source_producers.get("evaluator", {}).get("file_sha256"), "v0004 evaluation producer drifted")
        pair_results = evaluation.get("pair_results")
        require(isinstance(pair_results, list) and len(pair_results) == SEARCH_RECORDS, "v0004 pair result inventory drifted")
        require([item.get("pair_id") for item in pair_results] == expected_ids, "v0004 evaluation pair order drifted")
        for item in pair_results:
            item = mapping(item, "v0004 pair result")
            verify(item, "pair_evaluation_sha256", "v0004 pair result")
            require(item.get("status") == "pass", "v0004 pair evaluation did not pass")
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
                require(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(float(number)), f"v0004 pair result non-finite: {key}")
            require(item.get("pair_accuracy") is (float(item["reward_margin"]) > 0), "v0004 pair accuracy drifted")
        aggregate = preference_eval.aggregate_results(pair_results, split="search", checkpoint_step=step)
        require(evaluation.get("aggregate") == aggregate, "v0004 aggregate was not recomputed from pairs")
        gate = _expected_gate(aggregate)
        require(evaluation.get("eligibility") == gate, "v0004 eligibility was not recomputed")
        receipt, receipt_path = receipts[run_id]
        checkpoint = mapping(evaluation.get("checkpoint"), "v0004 evaluation checkpoint")
        require(
            checkpoint.get("training_receipt_path") == str(receipt_path)
            and checkpoint.get("training_receipt_file_sha256") == day23_gpu.file_sha256(receipt_path)
            and checkpoint.get("training_receipt_sha256") == receipt.get("receipt_sha256"),
            "v0004 evaluation training receipt binding drifted",
        )
        projected = {
            "candidate_id": text(candidate.get("candidate_id"), "v0004 candidate_id"),
            "run_id": run_id,
            "checkpoint_step": step,
            "learning_rate": float(runs[run_id].get("learning_rate")),
            "eligible": bool(gate["passed"]),
            "positive_pairs": int(gate["observed"]["positive_pairs"]),
            "mean_reward_margin": float(gate["observed"]["mean_reward_margin"]),
            "length_matched_mean_reward_margin": float(gate["observed"]["length_matched_mean_reward_margin"]),
            "path": str(evaluation_path),
            "file_sha256": day23_gpu.file_sha256(evaluation_path),
            "evaluation_sha256": evaluation_sha,
        }
        require(dict(candidate) == projected, "v0004 selection candidate projection drifted")
        recomputed_candidates.append(projected)

    eligible = [item for item in recomputed_candidates if item["eligible"]]
    require(
        selection.get("status") == "closed_no_candidate"
        and selection.get("candidate_pool_frozen") is True
        and selection.get("candidate_count") == 4
        and selection.get("eligible_count") == 0
        and not eligible
        and selection.get("selected_candidate") is None
        and selection.get("selected_run_id") is None
        and selection.get("selected_checkpoint_step") is None
        and selection.get("selected_learning_rate") is None
        and selection.get("selected_evaluation_sha256") is None,
        "v0004 was not terminal closed_no_candidate",
    )
    claim = mapping(selection.get("claim_boundary"), "v0004 selection claim")
    require(claim.get("dev_consumed") is False and claim.get("heldout_consumed") is False, "v0004 selection consumed protected data")

    search_claim_path = resolve_path(source.get("access_ledger", {}).get("search_claim"), "v0004 search claim")
    search_claim = load(search_claim_path, "v0004 search claim")
    verify(search_claim, "claim_sha256", "v0004 search claim")
    require(search_claim.get("split") == "search" and search_claim.get("campaign_sha256") == source_sha, "v0004 search claim drifted")
    return {
        "campaign": source,
        "campaign_path": source_path,
        "campaign_sha256": source_sha,
        "run_root": source_root,
        "selection": selection,
        "selection_path": source_selection_path,
        "selection_sha256": selection_sha,
        "receipts": receipts,
        "evaluations": recomputed_candidates,
        "search_claim": search_claim,
        "search_claim_path": search_claim_path,
        "search_rows": source_search,
        "search_path": source_search_path,
    }


def _same_identity(entry: Any, path: Path, content_sha256: str | None = None) -> bool:
    item = mapping(entry, "bound identity")
    if resolve_path(item.get("path"), "bound identity path") != path.resolve(strict=True):
        return False
    if item.get("file_sha256") != day23_gpu.file_sha256(path):
        return False
    if content_sha256 is not None and item.get("content_sha256") != content_sha256:
        return False
    return True


def _validate_partition(
    campaign: Mapping[str, Any],
    source_terminal: Mapping[str, Any],
    *,
    bootcamp: Path,
    run_root: Path,
) -> dict[str, Any]:
    datasets = mapping(campaign.get("datasets"), "v0005 datasets")
    source_datasets = mapping(source_terminal["campaign"].get("datasets"), "v0004 datasets")
    full_rows, full_path = _dataset(datasets.get("full_train"), "v0005 full_train", bootcamp=bootcamp, allowed_roots=(bootcamp, run_root))
    fit_rows, fit_path = _dataset(datasets.get("fit"), "v0005 fit", bootcamp=bootcamp, allowed_roots=(bootcamp, run_root))
    search_rows, search_path = _dataset(datasets.get("search"), "v0005 search", bootcamp=bootcamp, allowed_roots=(bootcamp, run_root))
    source_fit_rows, _ = _dataset(source_datasets.get("fit"), "v0004 fit", bootcamp=bootcamp, allowed_roots=(bootcamp, source_terminal["run_root"]))
    require(len(full_rows) == FULL_RECORDS and len(fit_rows) == FIT_RECORDS and len(search_rows) == SEARCH_RECORDS, "v0005 split counts drifted")

    full_ids = [row["pair_id"] for row in full_rows]
    fit_ids = [row["pair_id"] for row in fit_rows]
    search_ids = [row["pair_id"] for row in search_rows]
    source_fit_ids = [row["pair_id"] for row in source_fit_rows]
    old_search_ids = [row["pair_id"] for row in source_terminal["search_rows"]]
    old = set(OLD_PAIR_IDS)
    require(tuple(full_ids[: len(OLD_PAIR_IDS)]) == OLD_PAIR_IDS, "historical old4 prefix drifted")
    require(len(source_fit_ids) == FIT_RECORDS and old.issubset(source_fit_ids), "v0004 fit population drifted")
    eligible = [pair_id for pair_id in source_fit_ids if pair_id not in old]
    require(len(eligible) == ROTATION_ELIGIBLE_RECORDS, "v0005 rotation eligible population drifted")
    ranked = sorted(
        eligible,
        key=lambda pair_id: (
            hashlib.sha256((ROTATION_SALT + pair_id).encode("utf-8")).hexdigest(),
            pair_id,
        ),
    )
    selected = set(ranked[:SEARCH_RECORDS])
    expected_search_ids = [pair_id for pair_id in full_ids if pair_id in selected]
    expected_fit_ids = [pair_id for pair_id in full_ids if pair_id not in selected]
    require(search_ids == expected_search_ids and fit_ids == expected_fit_ids, "v0005 rotated membership/order was not exactly recomputed")
    require(set(search_ids).issubset(set(source_fit_ids) - old), "v0005 search was not drawn only from unobserved v0004 fit rows")
    require(set(search_ids).isdisjoint(old_search_ids) and set(search_ids).isdisjoint(old), "v0005 search reused observed rows")
    require(set(old_search_ids).issubset(fit_ids), "v0004 search rows did not move into v0005 fit")
    require(old.issubset(fit_ids), "old4 did not remain in v0005 fit")
    require(set(fit_ids).isdisjoint(search_ids) and set(fit_ids) | set(search_ids) == set(full_ids), "v0005 split is not a complete disjoint partition")

    by_id = {row["pair_id"]: row for row in full_rows}
    require(all(row == by_id[row["pair_id"]] for row in fit_rows + search_rows), "v0005 split row bytes/content drifted from full154")
    partition = mapping(campaign.get("data_partition"), "v0005 data_partition")
    expected_partition = {
        "authority": "v0004_fit_pair_ids_only_no_dev_or_heldout_rows",
        "selection_salt": ROTATION_SALT,
        "order": "sha256(campaign_id + '|' + pair_id), then pair_id; emitted in frozen full_train order",
        "rotation_population_records": ROTATION_ELIGIBLE_RECORDS,
        "search_prefix_records": SEARCH_RECORDS,
        "fit_records": FIT_RECORDS,
        "v0004_search_disposition": {
            "records": SEARCH_RECORDS,
            "moved_to_optimizer_fit": True,
            "eligible_for_v0005_search_evaluation": False,
            "ordered_pair_ids_sha256": day23_contract.object_sha256(old_search_ids),
        },
        "historical_four_pair_disposition": {
            "pair_ids": list(OLD_PAIR_IDS),
            "optimizer_input_in_fit": True,
            "eligible_for_search_selection_or_confirmation": False,
        },
        "v0005_search_prior_exposure": {
            "evaluated_before_v0005": False,
            "present_in_v0004_optimizer_fit": True,
            "v0005_training_starts_fresh_from_promoted_s1": True,
        },
    }
    require(dict(partition) == expected_partition, "v0005 data_partition contract drifted")
    require(datasets.get("fit", {}).get("ordered_pair_ids_sha256") == day23_contract.object_sha256(fit_ids), "v0005 fit identity drifted")
    require(datasets.get("search", {}).get("ordered_pair_ids_sha256") == day23_contract.object_sha256(search_ids), "v0005 search identity drifted")
    return {
        "full_path": full_path,
        "fit_path": fit_path,
        "search_path": search_path,
        "full_ids": full_ids,
        "fit_ids": fit_ids,
        "search_ids": search_ids,
        "old_search_ids": old_search_ids,
        "selection_salt": ROTATION_SALT,
    }


def _validate_source_campaign(
    campaign_path: Path,
    *,
    require_search_unopened: bool,
    producer_snapshots_root: Path | None = None,
    allow_existing_amendment4: bool = False,
) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = load(campaign_path, "v0005 campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "v0005 campaign")
    require(
        campaign.get("schema_name") == CAMPAIGN_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("goal_id") == GOAL_ID
        and campaign.get("version_id") == VERSION_ID,
        "v0005 campaign identity drifted",
    )
    bootcamp = resolve_path(campaign.get("bootcamp_root"), "v0005 bootcamp root")
    run_root = resolve_path(campaign.get("remote_run_root"), "v0005 run root")
    require(campaign_path.is_relative_to(run_root), "v0005 campaign escaped its run root")

    producers = mapping(campaign.get("producers"), "v0005 producers")
    for name in ("builder", "runner", "evaluator", "authority_validator"):
        entry = mapping(producers.get(name), f"v0005 producer {name}")
        path = resolve_path(entry.get("path"), f"v0005 producer {name} path")
        require(path.is_relative_to(bootcamp), f"v0005 producer escaped bootcamp: {name}")
        if producer_snapshots_root is None:
            require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"v0005 producer drifted: {name}")
        else:
            snapshot = (producer_snapshots_root / path.name).resolve(strict=True)
            require(
                snapshot.is_file()
                and not snapshot.is_symlink()
                and day23_gpu.file_sha256(snapshot) == entry.get("file_sha256")
                and snapshot.stat().st_size == entry.get("bytes"),
                f"v0005 frozen producer snapshot drifted: {name}",
            )
        if name == "authority_validator" and producer_snapshots_root is None:
            require(path == Path(__file__).resolve(strict=True), "campaign bound a different v0005 authority validator")

    charter_entry = mapping(campaign.get("charter"), "v0005 charter binding")
    charter_path = resolve_path(charter_entry.get("path"), "v0005 charter path", base=bootcamp)
    charter = load(charter_path, "authoritative charter")
    charter_sha = verify(charter, "charter_sha256", "authoritative charter")
    require(day23_gpu.file_sha256(charter_path) == charter_entry.get("file_sha256") and charter_sha == charter_entry.get("content_sha256"), "authoritative charter binding drifted")
    require(VERSION_ID in charter.get("iteration_budget", {}).get("allowed_version_ids", []) and charter.get("iteration_budget", {}).get("maximum_semantic_versions") == 3, "v0005 is not the final authorized semantic version")

    supersedes = mapping(campaign.get("supersedes"), "v0005 supersedes")
    source_path = resolve_path(supersedes.get("campaign_path"), "v0004 campaign path")
    source = load(source_path, "v0004 campaign")
    require(day23_gpu.file_sha256(source_path) == supersedes.get("campaign_file_sha256") and verify(source, "campaign_sha256", "v0004 campaign") == supersedes.get("campaign_sha256"), "v0004 source campaign binding drifted")
    selection_path = resolve_path(supersedes.get("selection_path"), "v0004 selection path")
    source_terminal = _validate_source_terminal(source, source_path, selection_path, bootcamp=bootcamp)
    require(day23_gpu.file_sha256(selection_path) == supersedes.get("selection_file_sha256") and source_terminal["selection_sha256"] == supersedes.get("selection_sha256"), "v0004 source selection binding drifted")

    partition = _validate_partition(campaign, source_terminal, bootcamp=bootcamp, run_root=run_root)
    require(campaign.get("remote_parent") == source.get("remote_parent"), "v0005 promoted-S1 parent binding drifted")
    require(campaign.get("fixed_recipe", {}).get("model") == source.get("fixed_recipe", {}).get("model"), "v0005 parent/reference recipe drifted")

    fixed = mapping(campaign.get("fixed_recipe"), "v0005 fixed recipe")
    runtime = mapping(fixed.get("runtime"), "v0005 runtime")
    require(
        runtime.get("world_size") == WORLD_SIZE
        and runtime.get("per_device_train_batch_size") == PER_DEVICE_BATCH
        and runtime.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
        and runtime.get("nominal_global_train_batch_size") == GLOBAL_BATCH
        and float(runtime.get("min_free_memory_fraction")) == MIN_FREE_FRACTION,
        "v0005 B8/GA2/20% runtime drifted",
    )
    require([float(value) for value in fixed.get("lever_levels", [])] == LEARNING_RATES, "v0005 learning-rate grid drifted")
    require(
        fixed.get("primary_scientific_lever") == "model.optimization.learning_rate.base"
        and fixed.get("dependent_lever") == "candidate_checkpoint_observation_schedule"
        and fixed.get("dependent_lever_is_inseparable_from_primary_interpolation") is True
        and fixed.get("single_scientific_change_claimed") is False,
        "v0005 primary/dependent lever disclosure drifted",
    )
    require(fixed.get("search_seed") == 20260819 and fixed.get("refit_seed") == 20260820, "v0005 seeds drifted")

    run_specs = mapping(campaign.get("run_specs"), "v0005 run specs")
    search_specs = [mapping(spec, f"v0005 run {run_id}") for run_id, spec in sorted(run_specs.items()) if isinstance(spec, Mapping) and spec.get("role") == "search_train"]
    require(len(search_specs) == 2, "v0005 search trajectory count drifted")
    require(sorted(float(spec.get("learning_rate")) for spec in search_specs) == LEARNING_RATES, "v0005 search LR grid drifted")
    for spec in search_specs:
        require(spec.get("max_steps") == MAX_STEPS and spec.get("checkpoint_steps") == CHECKPOINT_STEPS and spec.get("dataset_key") == "fit" and spec.get("authorized") is True and spec.get("fresh_start_from_parent") is True, "v0005 search spec drifted")
        require(
            spec.get("checkpoint_capture")
            == {
                "mode": "periodic_save_with_terminal_retention",
                "save_steps": 2,
                "save_total_limit": 7,
                "candidate_checkpoint_steps": CHECKPOINT_STEPS,
                "expected_retained_checkpoint_steps": [18, 20, 22, 24, 26, 28, 30],
                "retained_noncandidate_checkpoint_steps": [22, 24, 26, 28, 30],
                "retained_noncandidate_checkpoints_candidate_eligible": False,
                "training_continues_to_max_steps_for_v0004_cosine_horizon_parity": True,
            },
            "v0005 checkpoint capture/disclosure drifted",
        )
        config_entry = mapping(spec.get("executable_config"), "v0005 search config")
        config_path = resolve_path(config_entry.get("path"), "v0005 search config path")
        require(day23_gpu.file_sha256(config_path) == config_entry.get("file_sha256"), "v0005 search config hash drifted")
        config = load(config_path, "v0005 search config")
        require(
            config.get("per_device_train_batch_size") == PER_DEVICE_BATCH
            and config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION
            and config.get("max_steps") == MAX_STEPS
            and config.get("save_steps") == 2
            and config.get("save_total_limit") == 7
            and float(config.get("learning_rate")) == float(spec.get("learning_rate"))
            and config.get("dataset") == [str(partition["fit_path"])],
            "v0005 search config recipe drifted",
        )
        require(not any(key in config for key in ("adapters", "ref_adapters", "ref_model", "resume_from_checkpoint", "val_dataset")), "v0005 search config is not a fresh protected-data-free start")

    extension = mapping(campaign.get("control_extension"), "v0005 control extension")
    strict_requirement = mapping(extension.get("strict_preunseal_requirement"), "v0005 strict preunseal requirement")
    expected_receipt_path = run_root / "evidence/authority/strict-preunseal.json"
    expected_amendment_path = bootcamp / AMENDMENT4_REL
    require(
        dict(strict_requirement)
        == {
            "required_before_first_optimizer": True,
            "require_search_unopened": True,
            "validator": dict(producers["authority_validator"]),
            "receipt_path": str(expected_receipt_path),
            "receipt_schema_name": "day23.rsi_v0005_strict_authority_preflight",
            "receipt_self_hash_field": "validation_sha256",
            "amendment_path": str(AMENDMENT4_REL),
            "amendment_id": "amendment-000004-v0005-strict-preunseal",
            "amendment_self_hash_field": "amendment_sha256",
        },
        "v0005 strict preunseal requirement drifted",
    )
    require(not expected_receipt_path.exists(), "v0005 strict preunseal receipt already exists")
    if not allow_existing_amendment4:
        require(not expected_amendment_path.exists(), "v0005 strict preunseal amendment4 already exists")
    amendment, amendment_path = resolve_bound(extension.get("amendment"), "v0005 amendment3", base=bootcamp, field="amendment_sha256")
    event5, event5_path = resolve_bound(extension.get("source_closed_event"), "v0005 event5", base=bootcamp, field="event_sha256")
    version, version_path = resolve_bound(extension.get("version"), "v0005 version", base=bootcamp, field="version_sha256")
    event6, event6_path = resolve_bound(extension.get("precommit_event"), "v0005 event6", base=bootcamp, field="event_sha256")
    strict_source, strict_source_path = resolve_bound(extension.get("source_strict_preunseal"), "v0004 strict preunseal", base=bootcamp, field="validation_sha256")
    assert amendment is not None and event5 is not None and version is not None and event6 is not None and strict_source is not None
    require(strict_source.get("status") == "pass" and strict_source.get("campaign_sha256") == source_terminal["campaign_sha256"], "v0004 strict preunseal source drifted")

    event4_path = bootcamp / CONTROL_ROOT_REL / "events/event-000004-version-precommitted.json"
    event4 = load(event4_path, "event4")
    event4_sha = verify(event4, "event_sha256", "event4")
    source_event4 = mapping(source.get("control_extension", {}).get("precommit_event"), "v0004 event4 binding")
    require(
        resolve_path(source_event4.get("path"), "v0004 event4 path", base=bootcamp) == event4_path
        and source_event4.get("file_sha256") == day23_gpu.file_sha256(event4_path)
        and source_event4.get("content_sha256") == event4_sha,
        "v0004 event4 source binding drifted",
    )
    authority = mapping(amendment.get("authority"), "v0005 amendment authority")
    require(
        amendment.get("schema_name") == "rsi.day23_dpo_append_only_amendment"
        and amendment.get("schema_version") == 1
        and amendment.get("status") == "frozen"
        and amendment.get("goal_id") == GOAL_ID
        and amendment.get("version_id") == VERSION_ID
        and amendment.get("amendment_id") == "amendment-000003-v0005-terminal-interpolation",
        "v0005 amendment3 identity drifted",
    )
    require(
        authority.get("charter_sha256") == charter_sha
        and authority.get("prior_event_000004_sha256") == event4_sha
        and authority.get("source_strict_preunseal") == extension.get("source_strict_preunseal")
        and {
            key: value
            for key, value in mapping(
                authority.get("source_strict_preunseal_amendment"),
                "v0005 amendment3 strict-source amendment identity",
            ).items()
            if key != "bytes"
        }
        == strict_source.get("strict_preunseal_amendment")
        and mapping(
            authority.get("source_strict_preunseal_amendment"),
            "v0005 amendment3 strict-source amendment identity",
        ).get("bytes")
        > 0,
        "v0005 amendment3 predecessor authority drifted",
    )
    source_bound = mapping(amendment.get("source_terminal"), "v0005 amendment source terminal")
    expected_source_bound = {
        "campaign_path": str(source_path),
        "campaign_file_sha256": day23_gpu.file_sha256(source_path),
        "campaign_sha256": source_terminal["campaign_sha256"],
        "selection_path": str(selection_path),
        "selection_file_sha256": day23_gpu.file_sha256(selection_path),
        "selection_sha256": source_terminal["selection_sha256"],
        "status": "closed_no_candidate",
        "eligible_count": 0,
        "training_receipts": [
            {
                **identity(path, "receipt_sha256"),
                "run_id": run_id,
                "status": "pass",
                "global_step": MAX_STEPS,
                "minimum_observed_device_free_fraction": float(
                    receipt["observations"]["memory"]["minimum_observed_device_free_fraction"]
                ),
            }
            for run_id, (receipt, path) in source_terminal["receipts"].items()
        ],
        "evaluations": [
            {
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "bytes": Path(item["path"]).stat().st_size,
                "content_sha256": item["evaluation_sha256"],
                "candidate_id": item["candidate_id"],
                "run_id": item["run_id"],
                "checkpoint_step": item["checkpoint_step"],
                "eligible": False,
            }
            for item in source_terminal["evaluations"]
        ],
        "search_claim": identity(source_terminal["search_claim_path"], "claim_sha256"),
    }
    require(dict(source_bound) == expected_source_bound, "v0005 amendment source terminal was not exactly recomputed")
    change = mapping(amendment.get("authorized_change"), "v0005 authorized change")
    require(
        change.get("primary_lever_id") == "model.optimization.learning_rate.base"
        and change.get("dependent_lever_id") == "candidate_checkpoint_observation_schedule"
        and change.get("dependent_lever_is_inseparable_from_primary_interpolation") is True
        and change.get("learning_rates") == {"from": [1e-6, 5e-6], "to": LEARNING_RATES}
        and change.get("checkpoint_steps") == {"from": [15, 30], "to": CHECKPOINT_STEPS}
        and change.get("search_max_steps") == MAX_STEPS
        and change.get("search_save_steps") == 2
        and change.get("search_save_total_limit") == 7
        and change.get("cosine_schedule_horizon_inherited_from_v0004") is True,
        "v0005 amendment authorized interpolation drifted",
    )
    split_change = mapping(change.get("split_rotation"), "v0005 authorized split rotation")
    require(
        split_change.get("selection_salt") == ROTATION_SALT
        and split_change.get("new_fit") == campaign["datasets"]["fit"]
        and split_change.get("new_search") == campaign["datasets"]["search"]
        and split_change.get("old_search_disposition") == campaign["data_partition"]["v0004_search_disposition"],
        "v0005 amendment split rotation drifted",
    )
    basis = mapping(amendment.get("scientific_basis"), "v0005 scientific basis")
    risk = mapping(basis.get("adaptive_overfit_risk"), "v0005 adaptive overfit risk")
    require(
        basis.get("method") == "bilinear_interpolation_of_four_v0004_search_aggregates"
        and basis.get("new_search_rows_observed_before_precommit") is False
        and dict(risk)
        == {
            "acknowledged": True,
            "source": "candidate_grid_was_derived_from_consumed_v0004_search_results",
            "mitigation": "mitigated_only_by_rotated_previously_unobserved_search30",
            "risk_eliminated": False,
        },
        "v0005 adaptive-overfit disclosure drifted",
    )
    invariants = mapping(amendment.get("frozen_invariants"), "v0005 frozen invariants")
    require(
        dict(invariants)
        == {
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "nominal_global_train_batch_size": GLOBAL_BATCH,
            "minimum_free_memory_fraction": MIN_FREE_FRACTION,
            "search_seed": 20260819,
            "refit_seed": 20260820,
            "fresh_s1_only": True,
            "resume_from_v0004_checkpoint": False,
            "dev_and_heldout_leases_unchanged": True,
        },
        "v0005 frozen invariants drifted",
    )
    for name in ("builder", "runner", "evaluator", "authority_validator"):
        require(amendment.get("producers", {}).get(name, {}).get("file_sha256") == producers.get(name, {}).get("file_sha256"), f"v0005 amendment producer drifted: {name}")

    prior5 = mapping(event5.get("prior_event"), "event5 prior")
    require(
        event5.get("schema_name") == "rsi.day23_dpo_event"
        and event5.get("schema_version") == 1
        and event5.get("sequence") == 5
        and event5.get("event_type") == "version-closed-no-candidate"
        and event5.get("version_id") == SOURCE_VERSION
        and event5.get("state_before") == "version_precommitted"
        and event5.get("state_after") == "iteration_open",
        "event5 transition drifted",
    )
    require(prior5.get("path") == str(event4_path.relative_to(event4_path.parent.parent)) and prior5.get("file_sha256") == day23_gpu.file_sha256(event4_path) and prior5.get("event_sha256") == event4_sha and prior5.get("sequence") == 4, "event5 predecessor drifted")
    require(event5.get("authority_refs") == [extension.get("amendment")], "event5 authority refs drifted")
    payload5 = mapping(event5.get("payload"), "event5 payload")
    require(payload5.get("source_terminal") == source_bound and payload5.get("candidate_eligible") is False and payload5.get("checkpoint_resume_forbidden") is True and payload5.get("dev_rows_opened") == 0 and payload5.get("heldout_rows_opened") == 0, "event5 terminal claim drifted")

    require(
        version.get("schema_name") == "rsi.day23_dpo_version_precommit"
        and version.get("schema_version") == 1
        and version.get("version_id") == VERSION_ID
        and version.get("status") == "precommitted",
        "v0005 version identity drifted",
    )
    operational = mapping(version.get("fixed_operational_constraint"), "v0005 version operational constraint")
    require(operational.get("world_size") == WORLD_SIZE and operational.get("per_device_train_batch_size") == PER_DEVICE_BATCH and operational.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION and operational.get("nominal_global_train_batch_size") == GLOBAL_BATCH and float(operational.get("minimum_free_memory_fraction")) == MIN_FREE_FRACTION, "v0005 version runtime drifted")
    intervention = mapping(version.get("intervention"), "v0005 version intervention")
    require(
        intervention.get("primary_lever") == "model.optimization.learning_rate.base"
        and intervention.get("dependent_lever") == "candidate_checkpoint_observation_schedule"
        and intervention.get("dependent_lever_is_inseparable_from_primary_interpolation") is True
        and intervention.get("single_scientific_change_claimed") is False
        and intervention.get("candidate_values") == LEARNING_RATES
        and intervention.get("checkpoint_measurements") == CHECKPOINT_STEPS
        and intervention.get("adaptive_overfit_risk") == risk,
        "v0005 version intervention/risk drifted",
    )
    fresh = mapping(version.get("fresh_start"), "v0005 version fresh start")
    require(fresh == {"from_promoted_s1": True, "resume_from_v0004_checkpoint": False, "source_v0004_checkpoints_candidate_eligible": False}, "v0005 fresh-S1 authority drifted")
    require(version.get("supersedes") == source_bound and version.get("amendment") == extension.get("amendment"), "v0005 version predecessor binding drifted")

    prior6 = mapping(event6.get("prior_event"), "event6 prior")
    require(
        event6.get("schema_name") == "rsi.day23_dpo_event"
        and event6.get("schema_version") == 1
        and event6.get("sequence") == 6
        and event6.get("event_type") == "version-precommitted"
        and event6.get("version_id") == VERSION_ID
        and event6.get("state_before") == "iteration_open"
        and event6.get("state_after") == "version_precommitted",
        "event6 transition drifted",
    )
    require(prior6.get("path") == str(event5_path.relative_to(event5_path.parent.parent)) and prior6.get("file_sha256") == day23_gpu.file_sha256(event5_path) and prior6.get("event_sha256") == event5.get("event_sha256") and prior6.get("sequence") == 5, "event6 predecessor drifted")
    require(event6.get("authority_refs") == [extension.get("amendment"), extension.get("version")], "event6 authority refs drifted")
    payload6 = mapping(event6.get("payload"), "event6 payload")
    require(
        dict(payload6)
        == {
            "runtime_profile": "world2_B8_GA2_global32",
            "learning_rate_grid": LEARNING_RATES,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "search_split_rotated": True,
            "training_started": False,
            "search_rows_opened": 0,
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
        },
        "event6 precommit claim drifted",
    )

    access = mapping(campaign.get("access_ledger"), "v0005 access ledger")
    search_claim = resolve_path(access.get("search_claim"), "v0005 search claim", must_exist=False)
    require(search_claim == run_root / "evidence/search-selection/search-claim.json", "v0005 search claim escaped run root")
    if require_search_unopened:
        require(not search_claim.exists(), "v0005 search was already unsealed")
    dev_claim = resolve_path(access.get("dev_claim"), "global dev claim", must_exist=False)
    require(not dev_claim.exists(), "global dev was already unsealed")
    heldout_lease = mapping(mapping(charter.get("access_leases"), "charter leases").get("heldout"), "charter heldout lease")
    heldout_rel = Path(text(heldout_lease.get("claim_path"), "charter heldout claim path"))
    require(not heldout_rel.is_absolute() and ".." not in heldout_rel.parts, "heldout claim path is not canonical")
    heldout_claim = bootcamp / "rsi-control" / heldout_rel
    require(not heldout_claim.exists(), "global heldout was already unsealed")

    return {
        "schema_name": "day23.rsi_v0005_strict_authority_preflight",
        "schema_version": 1,
        "status": "pass",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "campaign_path": str(campaign_path),
        "campaign_file_sha256": day23_gpu.file_sha256(campaign_path),
        "campaign_sha256": campaign_sha,
        "source_terminal": {
            "campaign_sha256": source_terminal["campaign_sha256"],
            "selection_sha256": source_terminal["selection_sha256"],
            "status": "closed_no_candidate",
            "eligible_count": 0,
            "training_receipts_recomputed": len(source_terminal["receipts"]),
            "evaluations_recomputed": len(source_terminal["evaluations"]),
        },
        "rotation": {
            "selection_salt": ROTATION_SALT,
            "eligible_v0004_fit_rows": ROTATION_ELIGIBLE_RECORDS,
            "new_search_rows": len(partition["search_ids"]),
            "new_fit_rows": len(partition["fit_ids"]),
            "old_search_rows_moved_to_fit": len(partition["old_search_ids"]),
            "new_search_ordered_pair_ids_sha256": day23_contract.object_sha256(partition["search_ids"]),
            "new_fit_ordered_pair_ids_sha256": day23_contract.object_sha256(partition["fit_ids"]),
        },
        "recipe": {
            "learning_rates": LEARNING_RATES,
            "dependent_observation_checkpoints": CHECKPOINT_STEPS,
            "max_steps": MAX_STEPS,
            "runtime_profile": "world2_B8_GA2_global32",
            "minimum_free_memory_fraction": MIN_FREE_FRACTION,
        },
        "risk_disclosure": {
            "adaptive_overfit_risk": "acknowledged",
            "reason": "learning rates and dependent observation checkpoints were adapted from consumed v0004 search evidence",
            "only_frozen_mitigation": "rotated previously-unobserved search30",
            "single_independent_change_claimed": False,
        },
        "authority": {
            "amendment3": extension.get("amendment"),
            "event5": extension.get("source_closed_event"),
            "version5": extension.get("version"),
            "event6": extension.get("precommit_event"),
            "source_strict_preunseal": extension.get("source_strict_preunseal"),
        },
        "strict_preunseal_requirement": dict(strict_requirement),
        "search_unopened_at_preflight": require_search_unopened,
        "dev_unopened": True,
        "heldout_unopened": True,
        "producer": dict(producers["authority_validator"]),
    }


def _canonical_identity(entry: Any, path: Path, field: str, label: str) -> dict[str, Any]:
    item = mapping(entry, label)
    actual = identity(path, field)
    require(dict(item) == actual, f"{label} identity drifted")
    return actual


def _without_path(entry: Any, label: str) -> dict[str, Any]:
    value = dict(mapping(entry, label))
    value.pop("path", None)
    return value


def _normalized(value: Any, run_root: Path) -> Any:
    """Normalize only run-local paths; scientific values stay byte-exact."""
    marker = "<V0005_RUN_ROOT>"
    if isinstance(value, str):
        return value.replace(str(run_root), marker)
    if isinstance(value, Mapping):
        return {key: _normalized(item, run_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item, run_root) for item in value]
    return value


def _validate_scientific_rebind(
    campaign: Mapping[str, Any],
    source: Mapping[str, Any],
    run_root: Path,
    source_root: Path,
) -> None:
    for key in (
        "budgets",
        "campaign_preregistration",
        "candidate_registry",
        "charter",
        "data_partition",
        "fixed_recipe",
        "inputs",
        "leases",
        "parent",
        "protocol",
        "remote_parent",
        "source_binding",
        "source_cpu_campaign",
        "supersedes",
    ):
        require(campaign.get(key) == source.get(key), f"corrected campaign changed scientific field: {key}")

    corrected_datasets = mapping(campaign.get("datasets"), "corrected datasets")
    source_datasets = mapping(source.get("datasets"), "source datasets")
    require(set(corrected_datasets) == set(source_datasets), "corrected dataset registry drifted")
    for key in sorted(corrected_datasets):
        corrected_entry = mapping(corrected_datasets[key], f"corrected dataset {key}")
        source_entry = mapping(source_datasets[key], f"source dataset {key}")
        if key not in {"fit", "search", "full_train"}:
            require(
                corrected_entry == source_entry,
                f"corrected protected dataset identity changed: {key}",
            )
            continue
        corrected_path = resolve_path(corrected_entry.get("path"), f"corrected dataset {key} path")
        require(
            day23_gpu.file_sha256(corrected_path) == corrected_entry.get("file_sha256")
            and corrected_path.stat().st_size == corrected_entry.get("bytes"),
            f"corrected dataset bytes drifted: {key}",
        )
        require(
            _without_path(corrected_entry, f"corrected dataset {key}")
            == _without_path(source_entry, f"source dataset {key}"),
            f"corrected dataset content identity changed: {key}",
        )
        if key in {"fit", "search"}:
            require(corrected_path.is_relative_to(run_root), f"corrected {key} escaped run root")

    corrected_specs = mapping(campaign.get("run_specs"), "corrected run specs")
    source_specs = mapping(source.get("run_specs"), "source run specs")
    require(set(corrected_specs) == set(source_specs), "corrected run registry drifted")
    for run_id in sorted(corrected_specs):
        corrected_spec = dict(mapping(corrected_specs[run_id], f"corrected run {run_id}"))
        source_spec = dict(mapping(source_specs[run_id], f"source run {run_id}"))
        corrected_config_entry = mapping(corrected_spec.pop("executable_config"), f"corrected config {run_id}")
        source_config_entry = mapping(source_spec.pop("executable_config"), f"source config {run_id}")
        require(
            _normalized(corrected_spec, run_root) == _normalized(source_spec, source_root),
            f"corrected run semantics changed: {run_id}",
        )
        corrected_config_path = resolve_path(corrected_config_entry.get("path"), f"corrected config {run_id} path")
        source_config_path = resolve_path(source_config_entry.get("path"), f"source config {run_id} path")
        require(
            corrected_config_path.is_relative_to(run_root)
            and day23_gpu.file_sha256(corrected_config_path) == corrected_config_entry.get("file_sha256")
            and corrected_config_path.stat().st_size == corrected_config_entry.get("bytes"),
            f"corrected executable config drifted: {run_id}",
        )
        corrected_config = load(corrected_config_path, f"corrected config {run_id}")
        source_config = load(source_config_path, f"source config {run_id}")
        require(
            _normalized(corrected_config, run_root) == _normalized(source_config, source_root),
            f"corrected executable recipe changed: {run_id}",
        )

    corrected_runtime = mapping(campaign.get("runtime_parse"), "corrected runtime parse")
    source_runtime = mapping(source.get("runtime_parse"), "source runtime parse")
    for key in (
        "scope",
        "status",
        "python_executable",
        "package_versions",
        "platform",
        "ms_swift_checkout",
        "ms_swift_import_root",
        "ms_swift_commit",
        "ms_swift_clean",
    ):
        require(corrected_runtime.get(key) == source_runtime.get(key), f"corrected runtime authority changed: {key}")
    corrected_stages = mapping(corrected_runtime.get("stages"), "corrected runtime stages")
    source_stages = mapping(source_runtime.get("stages"), "source runtime stages")
    require(set(corrected_stages) == set(source_stages), "corrected parsed stage registry drifted")
    for run_id in corrected_stages:
        corrected_stage = dict(mapping(corrected_stages[run_id], f"corrected parsed stage {run_id}"))
        source_stage = dict(mapping(source_stages[run_id], f"source parsed stage {run_id}"))
        corrected_stage.pop("output_dir", None)
        source_stage.pop("output_dir", None)
        require(
            _normalized(corrected_stage, run_root) == _normalized(source_stage, source_root),
            f"corrected parsed recipe changed: {run_id}",
        )


def validate(campaign_path: Path, *, require_search_unopened: bool) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    campaign = load(campaign_path, "corrected v0005 campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "corrected v0005 campaign")
    require(
        campaign.get("schema_name") == CAMPAIGN_SCHEMA
        and campaign.get("schema_version") == 1
        and campaign.get("status") == "gpu_execution_bound_optimizer_pending"
        and campaign.get("goal_id") == GOAL_ID
        and campaign.get("version_id") == VERSION_ID
        and campaign.get("campaign_id") == "day23-qwen35-dpo-rsi-v0005-binding-r0001"
        and campaign.get("binding_revision_id") == "preoptimizer-binding-r0001",
        "corrected v0005 campaign identity drifted",
    )
    run_root = resolve_path(campaign.get("remote_run_root"), "corrected v0005 run root")
    bootcamp = resolve_path(campaign.get("bootcamp_root"), "corrected v0005 bootcamp root")
    require(campaign_path == run_root / "binding/gpu-campaign.json", "corrected campaign is not canonical")

    producers = mapping(campaign.get("producers"), "corrected producers")
    expected_names = {"builder", "runner", "evaluator", "authority_validator"}
    require(set(producers) == expected_names, "corrected producer registry drifted")
    for name in sorted(expected_names):
        entry = mapping(producers[name], f"corrected producer {name}")
        path = resolve_path(entry.get("path"), f"corrected producer {name} path")
        require(
            path.is_relative_to(bootcamp)
            and path.is_file()
            and not path.is_symlink()
            and day23_gpu.file_sha256(path) == entry.get("file_sha256")
            and path.stat().st_size == entry.get("bytes"),
            f"corrected producer drifted: {name}",
        )
        if name == "authority_validator":
            require(path == Path(__file__).resolve(strict=True), "campaign bound another corrected validator")

    extension = mapping(campaign.get("control_extension"), "corrected control extension")
    superseded_entry = mapping(extension.get("superseded_campaign"), "superseded campaign identity")
    source_path = resolve_path(superseded_entry.get("path"), "superseded campaign path")
    source = load(source_path, "superseded v0005 campaign")
    source_sha = verify(source, "campaign_sha256", "superseded v0005 campaign")
    require(
        dict(superseded_entry) == identity(source_path, "campaign_sha256")
        and source.get("campaign_id") == "day23-qwen35-dpo-rsi-v0005"
        and source.get("binding_revision_id") is None
        and source.get("version_id") == VERSION_ID,
        "superseded campaign identity drifted",
    )
    source_root = resolve_path(source.get("remote_run_root"), "superseded run root")
    source_snapshots = source_root / "evidence/source-snapshot"
    source_validation = _validate_source_campaign(
        source_path,
        require_search_unopened=True,
        producer_snapshots_root=source_snapshots,
        allow_existing_amendment4=True,
    )

    supersedes = mapping(campaign.get("supersedes_preoptimizer"), "preoptimizer supersession")
    require(
        dict(supersedes)
        == {
            "campaign": dict(superseded_entry),
            "status": "preoptimizer_control_failure",
            "classification": "failure.control.validator_projection",
            "optimizer_steps_started": 0,
            "search_claimed": False,
            "dev_claimed": False,
            "heldout_claimed": False,
            "candidate_or_resume": False,
        },
        "preoptimizer supersession drifted",
    )
    require(not (source_root / "outputs").exists() or not any((source_root / "outputs").iterdir()), "superseded campaign has optimizer output")
    for run_id, spec_value in mapping(source.get("run_specs"), "superseded run specs").items():
        spec = mapping(spec_value, f"superseded run {run_id}")
        require(not Path(str(spec.get("success_receipt"))).exists(), f"superseded success receipt exists: {run_id}")
        require(not Path(str(spec.get("failure_receipt"))).exists(), f"superseded failure receipt exists: {run_id}")

    amendment5, amendment5_path = resolve_bound(
        extension.get("preoptimizer_correction_amendment"),
        "correction amendment5",
        base=bootcamp,
        field="amendment_sha256",
    )
    event7, event7_path = resolve_bound(
        extension.get("execution_binding_correction_event"),
        "correction event7",
        base=bootcamp,
        field="event_sha256",
    )
    assert amendment5 is not None and event7 is not None
    amendment5_identity = identity(amendment5_path, "amendment_sha256")
    amendment5_identity["path"] = str(amendment5_path.relative_to(bootcamp))
    event7_identity = identity(event7_path, "event_sha256")
    event7_identity["path"] = str(event7_path.relative_to(bootcamp))
    require(extension.get("preoptimizer_correction_amendment") == amendment5_identity, "amendment5 binding drifted")
    require(extension.get("execution_binding_correction_event") == event7_identity, "event7 binding drifted")

    source_extension = mapping(source.get("control_extension"), "superseded control extension")
    strict_source, _ = resolve_bound(
        source_extension.get("source_strict_preunseal"),
        "v0004 strict preunseal",
        base=bootcamp,
        field="validation_sha256",
    )
    amendment3, _ = resolve_bound(source_extension.get("amendment"), "amendment3", base=bootcamp, field="amendment_sha256")
    event6, event6_path = resolve_bound(source_extension.get("precommit_event"), "event6", base=bootcamp, field="event_sha256")
    assert strict_source is not None and amendment3 is not None and event6 is not None
    lhs = dict(mapping(amendment3.get("authority"), "amendment3 authority").get("source_strict_preunseal_amendment"))
    rhs = dict(mapping(strict_source.get("strict_preunseal_amendment"), "v0004 strict amendment identity"))
    require(set(lhs) == set(rhs) | {"bytes"} and {key: lhs[key] for key in rhs} == rhs, "diagnosed projection is not bytes-only")
    amendment2_raw = Path(text(rhs.get("path"), "v0004 strict amendment path"))
    amendment2_path = (amendment2_raw if amendment2_raw.is_absolute() else bootcamp / amendment2_raw).resolve(strict=True)
    require(lhs["bytes"] == amendment2_path.stat().st_size and day23_gpu.file_sha256(amendment2_path) == lhs["file_sha256"], "diagnosed bytes attestation is false")
    diagnosed = {
        "classification": "failure.control.validator_projection",
        "check": "source_strict_preunseal_amendment_identity_exact_equality",
        "lhs_with_bytes": lhs,
        "rhs_without_bytes": rhs,
        "exact_difference": {"field": "bytes", "lhs_value": lhs["bytes"], "rhs_field_present": False},
        "common_projection_equal": True,
        "error": "v0005 amendment3 predecessor authority drifted",
        "optimizer_started": False,
        "search_claimed": False,
        "dev_claimed": False,
        "heldout_claimed": False,
    }
    authority = mapping(amendment5.get("authority"), "amendment5 authority")
    require(
        amendment5.get("schema_name") == "rsi.day23_dpo_append_only_amendment"
        and amendment5.get("schema_version") == 1
        and amendment5.get("status") == "frozen"
        and amendment5.get("goal_id") == GOAL_ID
        and amendment5.get("version_id") == VERSION_ID
        and amendment5.get("amendment_id") == "amendment-000005-v0005-validator-projection-correction",
        "amendment5 identity drifted",
    )
    charter_entry = mapping(campaign.get("charter"), "corrected charter")
    charter_path = resolve_path(charter_entry.get("path"), "corrected charter path", base=bootcamp)
    charter = load(charter_path, "corrected charter")
    charter_sha = verify(charter, "charter_sha256", "corrected charter")
    require(
        authority
        == {
            "charter_sha256": charter_sha,
            "prior_event_000006": source_extension.get("precommit_event"),
            "source_campaign": dict(superseded_entry),
            "frozen_validator": source.get("producers", {}).get("authority_validator"),
        }
        and amendment5.get("diagnosed_failure") == diagnosed,
        "amendment5 authority/diagnosis drifted",
    )
    authorized = mapping(amendment5.get("authorized_correction"), "authorized correction")
    require(
        dict(authorized)
        == {
            "execution_binding_only": True,
            "semantic_version_unchanged": True,
            "new_semantic_version": False,
            "source_campaign_candidate_or_resume": False,
            "correction_builder": producers["builder"],
            "runner_adapter": producers["runner"],
            "evaluator_adapter": producers["evaluator"],
            "corrected_validator": producers["authority_validator"],
        },
        "authorized correction drifted",
    )
    expected_frozen = {
        "version_id": VERSION_ID,
        "learning_rates": LEARNING_RATES,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "search_max_steps": MAX_STEPS,
        "search_seed": 20260819,
        "refit_seed": 20260820,
        "datasets": {
            key: {field: campaign["datasets"][key][field] for field in ("file_sha256", "records", "ordered_pair_ids_sha256", "ordered_row_hashes_sha256")}
            for key in ("fit", "search", "full_train")
        },
        "runtime": {
            "world_size": WORLD_SIZE,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "nominal_global_train_batch_size": GLOBAL_BATCH,
            "minimum_free_memory_fraction": MIN_FREE_FRACTION,
        },
        "fresh_promoted_s1_only": True,
        "candidate_registry": campaign.get("candidate_registry"),
        "search_eligibility": campaign.get("protocol", {}).get("search_selection", {}).get("eligibility"),
    }
    require(amendment5.get("frozen_scientific_invariants") == expected_frozen, "frozen scientific invariants drifted")

    control_root = bootcamp / CONTROL_ROOT_REL
    prior7 = {
        "path": str(event6_path.relative_to(control_root)),
        "file_sha256": day23_gpu.file_sha256(event6_path),
        "event_sha256": event6["event_sha256"],
        "sequence": 6,
    }
    payload7 = {
        "classification": "failure.control.validator_projection",
        "source_campaign": dict(superseded_entry),
        "optimizer_steps_started": 0,
        "search_rows_opened": 0,
        "dev_rows_opened": 0,
        "heldout_rows_opened": 0,
        "semantic_version_unchanged": True,
        "scientific_configuration_unchanged": True,
        "corrected_execution_binding_pending": True,
    }
    require(
        event7.get("schema_name") == "rsi.day23_dpo_event"
        and event7.get("schema_version") == 1
        and event7.get("event_id") == "event-000007-execution-binding-corrected"
        and event7.get("sequence") == 7
        and event7.get("event_type") == "execution-binding-corrected"
        and event7.get("goal_id") == GOAL_ID
        and event7.get("version_id") == VERSION_ID
        and event7.get("state_before") == "version_precommitted"
        and event7.get("state_after") == "version_precommitted"
        and event7.get("prior_event") == prior7
        and event7.get("authority_refs") == [amendment5_identity]
        and event7.get("payload") == payload7,
        "event7 correction chain drifted",
    )

    _validate_scientific_rebind(campaign, source, run_root, source_root)
    strict_requirement = mapping(extension.get("strict_preunseal_requirement"), "corrected strict requirement")
    expected_receipt = run_root / "evidence/authority/strict-preunseal.json"
    expected_amendment = bootcamp / AMENDMENT4_REL
    require(
        dict(strict_requirement)
        == {
            "required_before_first_optimizer": True,
            "require_search_unopened": True,
            "validator": dict(producers["authority_validator"]),
            "receipt_path": str(expected_receipt),
            "receipt_schema_name": "day23.rsi_v0005_strict_authority_preflight",
            "receipt_self_hash_field": "validation_sha256",
            "amendment_path": str(AMENDMENT4_REL),
            "amendment_id": "amendment-000004-v0005-strict-preunseal",
            "amendment_self_hash_field": "amendment_sha256",
        },
        "corrected strict requirement drifted",
    )
    if require_search_unopened:
        require(not expected_receipt.exists() and not expected_amendment.exists(), "corrected strict artifacts already exist")
    elif expected_amendment.exists():
        sealed_amendment = load(expected_amendment, "existing amendment4")
        require(verify(sealed_amendment, "amendment_sha256", "existing amendment4") and sealed_amendment.get("campaign", {}).get("campaign_sha256") == campaign_sha, "existing amendment4 binds another campaign")

    access = mapping(campaign.get("access_ledger"), "corrected access ledger")
    search_claim = resolve_path(access.get("search_claim"), "corrected search claim", must_exist=False)
    require(search_claim == run_root / "evidence/search-selection/search-claim.json", "corrected search claim escaped run root")
    if require_search_unopened:
        require(not search_claim.exists(), "corrected search was already opened")
    dev_claim = resolve_path(access.get("dev_claim"), "global dev claim", must_exist=False)
    require(not dev_claim.exists(), "global dev was already opened")
    heldout_rel = Path(text(charter.get("access_leases", {}).get("heldout", {}).get("claim_path"), "heldout claim path"))
    require(not (bootcamp / "rsi-control" / heldout_rel).exists(), "global heldout was already opened")

    result = dict(source_validation)
    result.update(
        {
            "campaign_path": str(campaign_path),
            "campaign_file_sha256": day23_gpu.file_sha256(campaign_path),
            "campaign_sha256": campaign_sha,
            "binding_revision_id": "preoptimizer-binding-r0001",
            "preoptimizer_correction": {
                "classification": "failure.control.validator_projection",
                "superseded_campaign": dict(superseded_entry),
                "optimizer_steps_started": 0,
                "candidate_or_resume": False,
                "diagnosed_failure": diagnosed,
                "scientific_configuration_unchanged": True,
            },
            "authority": {
                **dict(mapping(source_validation.get("authority"), "source authority receipt")),
                "amendment5": amendment5_identity,
                "event7": event7_identity,
            },
            "strict_preunseal_requirement": dict(strict_requirement),
            "search_unopened_at_preflight": require_search_unopened,
            "dev_unopened": True,
            "heldout_unopened": True,
            "producer": dict(producers["authority_validator"]),
        }
    )
    return result


def seal_amendment4(campaign_path: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    campaign = load(campaign_path.resolve(strict=True), "v0005 campaign")
    bootcamp = resolve_path(campaign.get("bootcamp_root"), "v0005 bootcamp root")
    path = bootcamp / AMENDMENT4_REL
    value = {
        "schema_name": "rsi.day23_dpo_append_only_amendment",
        "schema_version": 1,
        "status": "frozen",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "amendment_id": "amendment-000004-v0005-strict-preunseal",
        "campaign": {
            "path": str(campaign_path.resolve(strict=True)),
            "file_sha256": day23_gpu.file_sha256(campaign_path.resolve(strict=True)),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "prior_search_rotation_amendment": campaign["control_extension"]["amendment"],
        "strict_validator": validation["producer"],
        "required_before_first_optimizer": True,
        "adaptive_overfit_risk": validation["risk_disclosure"],
        "requirements": [
            "v0004_two_training_receipts_and_four_search_evaluations_recomputed",
            "v0004_closed_no_candidate_selection_recomputed",
            "v0004_event4_to_v0005_event5_to_event6_hash_chain",
            "v0005_rotation_exactly_from_unobserved_v0004_fit_minus_old4",
            "v0004_search_moved_to_v0005_fit",
            "v0005_lr_and_dependent_checkpoint_adaptation_disclosed",
            "world2_B8_GA2_global32_twenty_percent_headroom",
            "fresh_promoted_S1_only",
            "search_dev_heldout_unopened",
            "exact_bound_producer_hashes",
        ],
    }
    sealed = write_exclusive(path, value, "amendment_sha256")
    return {
        "path": str(path.relative_to(bootcamp)),
        "file_sha256": day23_gpu.file_sha256(path),
        "bytes": path.stat().st_size,
        "content_sha256": sealed["amendment_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-search-unopened", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate(args.campaign, require_search_unopened=args.require_search_unopened)
        output = args.output.expanduser().resolve()
        required_output = Path(result["strict_preunseal_requirement"]["receipt_path"]).resolve()
        require(output == required_output, "strict preunseal receipt output path drifted from campaign")
        result["strict_preunseal_amendment"] = seal_amendment4(args.campaign, result)
        sealed = write_exclusive(output, result, "validation_sha256")
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "pass", "output": str(output), "validation_sha256": sealed["validation_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
