#!/usr/bin/env python3
"""Close RSI-v0005 and initialize the isolated Day 23 candidate-search goal.

The initializer never reads dev or heldout rows.  It independently recomputes
the completed v0005 search evidence, closes goal-0002, and freezes only the
new goal's authority, exposure ledger, and three deterministic search
tranches.  Scientific recipes belong to later per-version builders.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day23_contract
import eval_day23_rsi_candidate as rsi_eval
import run_day23_qwen35_gpu_stage as day23_gpu
import run_day23_rsi_candidate as rsi_runner


SOURCE_GOAL_ID = "goal-0002-day23-dpo"
SOURCE_VERSION_ID = "rsi-v0005"
GOAL_ID = "goal-0003-day23-dpo-candidate-search"
ALLOWED_VERSION_IDS = ["csearch-v0001", "csearch-v0002", "csearch-v0003"]
ROTATION_SALT = f"{GOAL_ID}|"
SOURCE_CONTROL_REL = Path("rsi-control/charters/goal-0002-day23-dpo")
CONTROL_REL = Path("rsi-control/charters") / GOAL_ID
AMENDMENT6_NAME = "amendment-000006-v0005-terminal-no-candidate"
EVENT8_NAME = "event-000008-version-closed-no-candidate"
EVENT1_NAME = "event-000001-charter-frozen"
EVENT2_NAME = "event-000002-source-terminal-imported"


class CSearchInitError(RuntimeError):
    """Terminal evidence or the isolated control-plane contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CSearchInitError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and value and "\x00" not in value, f"{label} must be non-empty text")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CSearchInitError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label}.{field} drifted")
    return expected


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = day23_gpu.object_sha256(result)
    return result


def identity(path: Path, field: str | None = None, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"identity is not a regular file: {path}")
    result: dict[str, Any] = {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "file_sha256": day23_gpu.file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if field:
        result["content_sha256"] = verify(load(path, str(path)), field, str(path))
    return result


def payload_identity(
    path: Path,
    value: Mapping[str, Any],
    field: str,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    payload = json_bytes(value)
    return {
        "path": str(path.relative_to(relative_to)),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "content_sha256": text(value.get(field), f"payload {field}"),
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json_bytes(value)
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


def resolve(value: Any, label: str, *, base: Path | None = None, must_exist: bool = True) -> Path:
    path = Path(text(value, label)).expanduser()
    if not path.is_absolute():
        require(base is not None, f"{label} is relative without a base")
        path = base / path
    return path.resolve(strict=must_exist)


def load_rows(path: Path, entry: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    require(day23_gpu.file_sha256(path) == entry.get("file_sha256"), f"{label} file hash drifted")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CSearchInitError(f"cannot read {label}: {path}") from error
    for index, line in enumerate(lines, 1):
        require(bool(line), f"blank line in {label}:{index}")
        value = json.loads(line)
        require(isinstance(value, dict), f"non-object row in {label}:{index}")
        try:
            day23_contract.verify_self_hash(value, "row_sha256", f"{label}:{index}")
        except day23_contract.Day23ContractError as error:
            raise CSearchInitError(str(error)) from error
        rows.append(value)
    require(len(rows) == entry.get("records"), f"{label} record count drifted")
    pair_ids = [text(row.get("pair_id"), f"{label} pair_id") for row in rows]
    row_hashes = [text(row.get("row_sha256"), f"{label} row_sha256") for row in rows]
    require(len(pair_ids) == len(set(pair_ids)), f"{label} pair IDs are not unique")
    require(day23_contract.object_sha256(pair_ids) == entry.get("ordered_pair_ids_sha256"), f"{label} ordered pair IDs drifted")
    require(day23_contract.object_sha256(row_hashes) == entry.get("ordered_row_hashes_sha256"), f"{label} ordered row hashes drifted")
    return rows


def load_bound(entry: Any, base: Path, field: str, label: str) -> tuple[dict[str, Any], Path]:
    item = mapping(entry, label)
    path = resolve(item.get("path"), f"{label}.path", base=base)
    require(day23_gpu.file_sha256(path) == item.get("file_sha256"), f"{label} file hash drifted")
    require(path.stat().st_size == item.get("bytes"), f"{label} byte count drifted")
    value = load(path, label)
    require(verify(value, field, label) == item.get("content_sha256"), f"{label} content hash drifted")
    return value, path


def _protected_leases(campaign: Mapping[str, Any], bootcamp: Path) -> dict[str, Any]:
    charter_entry = mapping(campaign.get("charter"), "v0005 charter binding")
    charter_path = resolve(charter_entry.get("path"), "v0005 charter path", base=bootcamp)
    charter = load(charter_path, "goal-0002 charter")
    charter_sha = verify(charter, "charter_sha256", "goal-0002 charter")
    require(
        day23_gpu.file_sha256(charter_path) == charter_entry.get("file_sha256")
        and charter_sha == charter_entry.get("content_sha256"),
        "goal-0002 charter binding drifted",
    )
    leases = mapping(charter.get("access_leases"), "goal-0002 access leases")
    for split in ("dev", "heldout"):
        lease = mapping(leases.get(split), f"goal-0002 {split} lease")
        claim_rel = Path(text(lease.get("claim_path"), f"{split} claim path"))
        require(not claim_rel.is_absolute() and ".." not in claim_rel.parts, f"{split} claim path is not canonical")
        claim_path = (bootcamp / "rsi-control" / claim_rel).resolve()
        require(not claim_path.exists(), f"global {split} lease was already claimed")
    return {
        "source_charter": identity(charter_path, "charter_sha256", relative_to=bootcamp),
        "scope_id": text(leases.get("scope_id"), "lease scope_id"),
        "ledger_root": text(leases.get("ledger_root"), "lease ledger_root"),
        "dev": dict(mapping(leases.get("dev"), "dev lease")),
        "heldout": dict(mapping(leases.get("heldout"), "heldout lease")),
    }


def _rotation_plan(
    campaign: Mapping[str, Any],
    source_v4: Mapping[str, Any],
    *,
    bootcamp: Path,
) -> dict[str, Any]:
    datasets = mapping(campaign.get("datasets"), "v0005 datasets")
    source_datasets = mapping(source_v4.get("datasets"), "v0004 datasets")
    full_entry = mapping(datasets.get("full_train"), "full154 dataset")
    v4_entry = mapping(source_datasets.get("search"), "v0004 search dataset")
    v5_entry = mapping(datasets.get("search"), "v0005 search dataset")
    full_path = resolve(full_entry.get("path"), "full154 path", base=bootcamp)
    v4_path = resolve(v4_entry.get("path"), "v0004 search path", base=bootcamp)
    v5_path = resolve(v5_entry.get("path"), "v0005 search path", base=bootcamp)
    full_rows = load_rows(full_path, full_entry, "full154")
    v4_rows = load_rows(v4_path, v4_entry, "v0004 search30")
    v5_rows = load_rows(v5_path, v5_entry, "v0005 search30")
    require(len(full_rows) == 154 and len(v4_rows) == len(v5_rows) == 30, "search rotation source counts drifted")

    full_ids = [row["pair_id"] for row in full_rows]
    v4_ids = [row["pair_id"] for row in v4_rows]
    v5_ids = [row["pair_id"] for row in v5_rows]
    mechanism_ids = list(rsi_eval.MECHANISM_PAIR_IDS)
    require(
        len(mechanism_ids) == 4
        and set(mechanism_ids).issubset(full_ids)
        and set(v4_ids).isdisjoint(v5_ids)
        and set(mechanism_ids).isdisjoint(v4_ids)
        and set(mechanism_ids).isdisjoint(v5_ids),
        "historical exposure sets overlap or escaped full154",
    )
    exposed = set(mechanism_ids) | set(v4_ids) | set(v5_ids)
    require(len(exposed) == 64 and exposed.issubset(full_ids), "historical exposed-pair union drifted")
    eligible = [pair_id for pair_id in full_ids if pair_id not in exposed]
    require(len(eligible) == 90, "candidate-search eligible population is not exactly 90")
    ranked = sorted(
        eligible,
        key=lambda pair_id: (
            hashlib.sha256((ROTATION_SALT + pair_id).encode("utf-8")).hexdigest(),
            pair_id,
        ),
    )
    by_id = {row["pair_id"]: row for row in full_rows}
    tranches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, version_id in enumerate(ALLOWED_VERSION_IDS):
        members = set(ranked[index * 30 : (index + 1) * 30])
        ordered = [pair_id for pair_id in full_ids if pair_id in members]
        require(len(ordered) == 30 and not seen.intersection(ordered), "candidate-search tranche overlap")
        seen.update(ordered)
        tranches.append(
            {
                "version_id": version_id,
                "membership_rank_start_inclusive": index * 30,
                "membership_rank_end_exclusive": (index + 1) * 30,
                "records": 30,
                "ordered_pair_ids_sha256": day23_contract.object_sha256(ordered),
                "ordered_row_hashes_sha256": day23_contract.object_sha256(
                    [by_id[pair_id]["row_sha256"] for pair_id in ordered]
                ),
            }
        )
    require(seen == set(eligible), "candidate-search tranches do not exhaust eligible90")
    return {
        "full_train": dict(full_entry),
        "historical_exposure": {
            "mechanism4": {
                "records": 4,
                "pair_ids": mechanism_ids,
                "ordered_pair_ids_sha256": day23_contract.object_sha256(mechanism_ids),
            },
            "v0004_search30": dict(v4_entry),
            "v0005_search30": dict(v5_entry),
            "distinct_pair_count": 64,
            "sorted_pair_ids_sha256": day23_contract.object_sha256(sorted(exposed)),
        },
        "eligible_pair_count": 90,
        "eligible_full_train_order_sha256": day23_contract.object_sha256(eligible),
        "selection_salt": ROTATION_SALT,
        "membership_rule": "sort eligible by (sha256(selection_salt + pair_id), pair_id); take consecutive rank30 blocks",
        "emission_order": "frozen full154 order within each membership block",
        "tranches": tranches,
        "search_reuse_forbidden": True,
        "consumed_tranches_may_enter_later_fit": True,
    }


def _terminal_state(campaign_path: Path) -> dict[str, Any]:
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    context = rsi_eval.validate_campaign(campaign_path)
    campaign = context["campaign"]
    campaign_sha = context["campaign_sha256"]
    require(
        campaign.get("goal_id") == SOURCE_GOAL_ID
        and campaign.get("version_id") == SOURCE_VERSION_ID
        and campaign.get("campaign_id") == "day23-qwen35-dpo-rsi-v0005-binding-r0001"
        and campaign.get("binding_revision_id") == "preoptimizer-binding-r0001",
        "source is not the corrected root2 v0005 campaign",
    )
    bootcamp = resolve(campaign.get("bootcamp_root"), "bootcamp root")
    run_root = resolve(campaign.get("remote_run_root"), "v0005 run root")
    require(campaign_path == run_root / "binding/gpu-campaign.json", "v0005 campaign path is not canonical")

    search_specs = [
        (run_id, mapping(spec, f"search spec {run_id}"))
        for run_id, spec in sorted(mapping(campaign.get("run_specs"), "v0005 run specs").items())
        if isinstance(spec, Mapping) and spec.get("role") == "search_train"
    ]
    require(len(search_specs) == 2, "v0005 terminal must contain exactly two search trajectories")
    receipts: list[dict[str, Any]] = []
    completion_anchors: list[str] = []
    for run_id, spec in search_specs:
        receipt_path = resolve(spec.get("success_receipt"), f"{run_id} success receipt")
        failure_path = resolve(spec.get("failure_receipt"), f"{run_id} failure receipt", must_exist=False)
        require(not failure_path.exists(), f"v0005 search trajectory failed: {run_id}")
        receipt = load(receipt_path, f"{run_id} success receipt")
        strict = rsi_runner.validate_training_receipt_value(campaign, campaign_path, receipt, receipt_path)
        require(strict.get("run_id") == run_id and strict.get("spec") == spec, f"strict training receipt projection drifted: {run_id}")
        require(
            receipt.get("status") == "pass"
            and receipt.get("observations", {}).get("global_step") == 30
            and receipt.get("claim_boundary", {}).get("dev_consumed") is False
            and receipt.get("claim_boundary", {}).get("heldout_consumed") is False,
            f"v0005 search receipt is not terminal/protected: {run_id}",
        )
        completed = text(receipt.get("completed_at_utc"), f"{run_id} completed_at_utc")
        completion_anchors.append(completed)
        receipts.append({**identity(receipt_path, "receipt_sha256"), "run_id": run_id, "global_step": 30})

    selection_path = run_root / "evidence/selection/search-selection.json"
    selection = rsi_eval.validate_search_selection(context, selection_path)
    require(
        selection.get("status") == "closed_no_candidate"
        and selection.get("candidate_pool_frozen") is True
        and selection.get("candidate_count") == 4
        and selection.get("eligible_count") == 0
        and selection.get("selected_candidate") is None
        and selection.get("selected_run_id") is None
        and selection.get("selected_checkpoint_step") is None
        and selection.get("selected_learning_rate") is None
        and selection.get("selected_evaluation_sha256") is None
        and selection.get("claim_boundary", {}).get("dev_consumed") is False
        and selection.get("claim_boundary", {}).get("heldout_consumed") is False,
        "v0005 selection is not closed_no_candidate",
    )
    evaluations: list[dict[str, Any]] = []
    for candidate in selection.get("candidates", []):
        candidate = mapping(candidate, "v0005 selection candidate")
        path = resolve(candidate.get("path"), "v0005 evaluation path")
        value = load(path, "v0005 search evaluation")
        evaluation_sha = verify(value, "evaluation_sha256", "v0005 search evaluation")
        require(
            day23_gpu.file_sha256(path) == candidate.get("file_sha256")
            and evaluation_sha == candidate.get("evaluation_sha256")
            and candidate.get("eligible") is False
            and value.get("claim_boundary", {}).get("dev_consumed") is False
            and value.get("claim_boundary", {}).get("heldout_consumed") is False,
            "v0005 evaluation terminal binding drifted",
        )
        evaluations.append(
            {
                **identity(path, "evaluation_sha256"),
                "candidate_id": candidate.get("candidate_id"),
                "run_id": candidate.get("run_id"),
                "checkpoint_step": candidate.get("checkpoint_step"),
                "eligible": False,
            }
        )
    require(len(evaluations) == 4, "v0005 evaluation count drifted")

    claim_path = resolve(campaign.get("access_ledger", {}).get("search_claim"), "v0005 search claim")
    require(claim_path == run_root / "evidence/search-selection/search-claim.json", "v0005 search claim path drifted")
    claim = load(claim_path, "v0005 search claim")
    claim_sha = verify(claim, "claim_sha256", "v0005 search claim")
    expected_candidates = sorted(item.get("candidate_id") for item in campaign.get("candidate_registry", []))
    require(
        claim.get("schema_name") == "day23.rsi_v0005_access_claim"
        and claim.get("status") == "claimed"
        and claim.get("split") == "search"
        and claim.get("campaign_sha256") == campaign_sha
        and claim.get("authorized_candidates") == expected_candidates
        and claim.get("max_unseal_count") == 1
        and claim.get("heldout_consumed") is False,
        "v0005 search claim drifted",
    )

    extension = mapping(campaign.get("control_extension"), "v0005 control extension")
    event7, event7_path = load_bound(extension.get("execution_binding_correction_event"), bootcamp, "event_sha256", "event7")
    require(event7.get("sequence") == 7 and event7.get("state_after") == "version_precommitted", "event7 transition drifted")
    strict_path = resolve(extension.get("strict_preunseal_requirement", {}).get("receipt_path"), "v0005 strict receipt")
    strict_receipt = load(strict_path, "v0005 strict receipt")
    require(
        verify(strict_receipt, "validation_sha256", "v0005 strict receipt")
        and strict_receipt.get("status") == "pass"
        and strict_receipt.get("campaign_sha256") == campaign_sha,
        "v0005 strict receipt drifted",
    )
    strict_amendment_entry = mapping(strict_receipt.get("strict_preunseal_amendment"), "v0005 strict amendment binding")
    strict_amendment_path = resolve(strict_amendment_entry.get("path"), "v0005 strict amendment path", base=bootcamp)
    strict_amendment = load(strict_amendment_path, "v0005 strict amendment")
    require(
        day23_gpu.file_sha256(strict_amendment_path) == strict_amendment_entry.get("file_sha256")
        and strict_amendment_path.stat().st_size == strict_amendment_entry.get("bytes")
        and verify(strict_amendment, "amendment_sha256", "v0005 strict amendment") == strict_amendment_entry.get("content_sha256"),
        "v0005 strict amendment identity drifted",
    )

    supersedes = mapping(campaign.get("supersedes"), "v0005 source v0004")
    source_v4_path = resolve(supersedes.get("campaign_path"), "v0004 campaign path")
    source_v4 = load(source_v4_path, "v0004 campaign")
    require(
        verify(source_v4, "campaign_sha256", "v0004 campaign") == supersedes.get("campaign_sha256")
        and day23_gpu.file_sha256(source_v4_path) == supersedes.get("campaign_file_sha256"),
        "v0004 source binding drifted",
    )
    leases = _protected_leases(campaign, bootcamp)
    rotation = _rotation_plan(campaign, source_v4, bootcamp=bootcamp)
    return {
        "bootcamp": bootcamp,
        "run_root": run_root,
        "campaign": campaign,
        "campaign_identity": identity(campaign_path, "campaign_sha256"),
        "selection": selection,
        "selection_identity": identity(selection_path, "selection_sha256"),
        "receipts": receipts,
        "evaluations": evaluations,
        "search_claim_identity": identity(claim_path, "claim_sha256"),
        "event7_identity": identity(event7_path, "event_sha256", relative_to=bootcamp),
        "strict_receipt_identity": identity(strict_path, "validation_sha256"),
        "strict_amendment_identity": identity(strict_amendment_path, "amendment_sha256", relative_to=bootcamp),
        "leases": leases,
        "rotation": rotation,
        "created_at_utc": max(completion_anchors),
    }


def _documents(state: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any], str]]:
    bootcamp: Path = state["bootcamp"]
    source_root = bootcamp / SOURCE_CONTROL_REL
    goal_root = bootcamp / CONTROL_REL
    producer = identity(Path(__file__).resolve(strict=True))
    created_at = state["created_at_utc"]

    amendment_path = source_root / f"amendments/{AMENDMENT6_NAME}.json"
    amendment = seal(
        {
            "schema_name": "rsi.day23_dpo_append_only_amendment",
            "schema_version": 1,
            "status": "frozen",
            "goal_id": SOURCE_GOAL_ID,
            "version_id": SOURCE_VERSION_ID,
            "amendment_id": AMENDMENT6_NAME,
            "created_at_utc": created_at,
            "authority": {
                "charter": state["leases"]["source_charter"],
                "prior_event_000007": state["event7_identity"],
                "strict_preunseal_receipt": state["strict_receipt_identity"],
                "strict_preunseal_amendment": state["strict_amendment_identity"],
            },
            "source_campaign": state["campaign_identity"],
            "terminal_evidence": {
                "training_receipts": state["receipts"],
                "search_evaluations": state["evaluations"],
                "search_claim": state["search_claim_identity"],
                "selection": state["selection_identity"],
                "selection_status": "closed_no_candidate",
                "candidate_count": 4,
                "eligible_count": 0,
            },
            "recomputed_claim": {
                "two_training_receipts_passed": True,
                "four_search_evaluations_recomputed": True,
                "selection_recomputed_from_all_evaluations": True,
                "optimizer_trajectories_completed": 2,
                "search_rows_opened": 30,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
                "candidate_or_resume": False,
            },
            "authorized_transition": {
                "state_before": "version_precommitted",
                "state_after": "terminal_no_go",
                "reason": "closed_no_candidate",
            },
            "producer": producer,
        },
        "amendment_sha256",
    )
    amendment_identity = payload_identity(
        amendment_path,
        amendment,
        "amendment_sha256",
        relative_to=bootcamp,
    )

    event8_path = source_root / f"events/{EVENT8_NAME}.json"
    prior7 = dict(state["event7_identity"])
    prior7["path"] = str((bootcamp / prior7["path"]).relative_to(source_root))
    prior7["event_sha256"] = prior7.pop("content_sha256")
    prior7.pop("bytes")
    prior7["sequence"] = 7
    event8 = seal(
        {
            "schema_name": "rsi.day23_dpo_event",
            "schema_version": 1,
            "event_id": EVENT8_NAME,
            "created_at_utc": created_at,
            "sequence": 8,
            "event_type": "version-closed-no-candidate",
            "charter_id": SOURCE_GOAL_ID,
            "goal_id": SOURCE_GOAL_ID,
            "version_id": SOURCE_VERSION_ID,
            "state_before": "version_precommitted",
            "state_after": "terminal_no_go",
            "prior_event": prior7,
            "authority_refs": [amendment_identity],
            "payload": {
                "campaign": state["campaign_identity"],
                "selection": state["selection_identity"],
                "status": "closed_no_candidate",
                "candidate_count": 4,
                "eligible_count": 0,
                "candidate_or_resume": False,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    event8_identity = payload_identity(event8_path, event8, "event_sha256", relative_to=bootcamp)

    charter_path = goal_root / "charter.json"
    charter = seal(
        {
            "schema_name": "day23.csearch_goal_charter",
            "schema_version": 1,
            "status": "frozen",
            "goal_id": GOAL_ID,
            "created_at_utc": created_at,
            "purpose": "continue fresh-search candidate discovery without extending or renaming the closed RSI goal",
            "source_lineage": {
                "source_goal_id": SOURCE_GOAL_ID,
                "source_terminal_state": "terminal_no_go",
                "source_charter": state["leases"]["source_charter"],
                "source_terminal_amendment": amendment_identity,
                "source_terminal_event": event8_identity,
                "source_campaign": state["campaign_identity"],
                "source_selection": state["selection_identity"],
            },
            "iteration_budget": {
                "allowed_version_ids": ALLOWED_VERSION_IDS,
                "maximum_semantic_versions": 3,
                "maximum_search_trajectories_per_version": 2,
                "maximum_search_candidates_per_version": 4,
                "exactly_one_scientific_lever_per_version": True,
                "stop_on_first_candidate": True,
                "stop_when_fresh_search_exhausted": True,
            },
            "candidate_gate": {
                "split": "search",
                "pairs": 30,
                "minimum_positive_pairs": 20,
                "mean_reward_margin": ">0",
                "length_matched_mean_reward_margin": ">0",
                "runner_up_fallback_forbidden": True,
                "threshold_relaxation_forbidden": True,
            },
            "scientific_recipe_authority": {
                "frozen_by_initializer": False,
                "required_before_each_version_optimizer": True,
                "owner": "separate campaign-bound version builder and strict pre-unseal validator",
            },
            "search_exposure_ledger": state["rotation"],
            "protected_data": {
                "inherited_scope_id": state["leases"]["scope_id"],
                "inherited_ledger_root": state["leases"]["ledger_root"],
                "dev": state["leases"]["dev"],
                "heldout": state["leases"]["heldout"],
                "raw_row_access_authorized": False,
                "claim_creation_authorized": False,
                "qualification_claim_allowed": False,
                "initializer_reads_protected_rows": False,
                "lease_reset_under_new_goal_forbidden": True,
            },
            "state_machine": {
                "initial_state": "uninitialized",
                "current_authorized_path": ["charter_frozen", "iteration_open"],
                "per_version_no_candidate_transition": ["version_precommitted", "iteration_open"],
                "success_terminal_state": "candidate_frozen_search_only",
                "exhaustion_terminal_state": "fresh_search_exhausted_no_candidate",
                "qualified_state_authorized": False,
            },
            "forbidden_actions": [
                "use_rsi_v0006_or_any_rsi_version_id",
                "reuse_any_historical_or_consumed_search_pair_for_search",
                "open_or_claim_dev",
                "open_or_claim_heldout",
                "claim_qualified",
                "start_optimizer_without_version_recipe_and_strict_preunseal",
            ],
            "producer": producer,
        },
        "charter_sha256",
    )
    charter_identity = payload_identity(charter_path, charter, "charter_sha256", relative_to=bootcamp)

    event1_path = goal_root / f"events/{EVENT1_NAME}.json"
    event1 = seal(
        {
            "schema_name": "day23.csearch_event",
            "schema_version": 1,
            "event_id": EVENT1_NAME,
            "created_at_utc": created_at,
            "sequence": 1,
            "event_type": "charter-frozen",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": None,
            "state_before": "uninitialized",
            "state_after": "charter_frozen",
            "prior_event": None,
            "authority_refs": [charter_identity],
            "payload": {
                "authority_write_mode": "O_EXCL",
                "scientific_recipe_frozen": False,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    event1_identity = payload_identity(event1_path, event1, "event_sha256", relative_to=bootcamp)
    event1_prior = dict(event1_identity)
    event1_prior["path"] = str(event1_path.relative_to(goal_root))
    event1_prior["event_sha256"] = event1_prior.pop("content_sha256")
    event1_prior.pop("bytes")
    event1_prior["sequence"] = 1

    event2_path = goal_root / f"events/{EVENT2_NAME}.json"
    event2 = seal(
        {
            "schema_name": "day23.csearch_event",
            "schema_version": 1,
            "event_id": EVENT2_NAME,
            "created_at_utc": created_at,
            "sequence": 2,
            "event_type": "source-terminal-imported",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": None,
            "state_before": "charter_frozen",
            "state_after": "iteration_open",
            "prior_event": event1_prior,
            "authority_refs": [charter_identity, amendment_identity, event8_identity],
            "payload": {
                "source_goal_id": SOURCE_GOAL_ID,
                "source_terminal_state": "terminal_no_go",
                "source_campaign": state["campaign_identity"],
                "source_selection": state["selection_identity"],
                "fresh_search_pair_count": 90,
                "frozen_tranche_count": 3,
                "next_version_id": "csearch-v0001",
                "scientific_recipe_pending": True,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    return [
        (amendment_path, amendment, "amendment_sha256"),
        (event8_path, event8, "event_sha256"),
        (charter_path, charter, "charter_sha256"),
        (event1_path, event1, "event_sha256"),
        (event2_path, event2, "event_sha256"),
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    state = _terminal_state(args.campaign)
    documents = _documents(state)
    if args.mode in {"preflight", "build"}:
        for path, _, _ in documents:
            require(not path.exists(), f"append-only output already exists: {path}")
    if args.mode == "preflight":
        return {
            "status": "preflight_pass",
            "goal_id": GOAL_ID,
            "source_campaign_sha256": state["campaign"]["campaign_sha256"],
            "source_selection_sha256": state["selection"]["selection_sha256"],
            "training_receipts_recomputed": len(state["receipts"]),
            "search_evaluations_recomputed": len(state["evaluations"]),
            "historical_exposed_pairs": 64,
            "fresh_search_pairs": 90,
            "tranches": 3,
            "protected_rows_opened": 0,
            "outputs": [str(path) for path, _, _ in documents],
        }
    if args.mode == "build":
        for path, value, _ in documents:
            write_exclusive(path, value)
        status = "initialized"
    else:
        for path, expected, field in documents:
            actual = load(path.resolve(strict=True), str(path))
            verify(actual, field, str(path))
            require(actual == expected and path.read_bytes() == json_bytes(expected), f"sealed control artifact drifted: {path}")
        status = "check_pass"
    return {
        "status": status,
        "goal_id": GOAL_ID,
        "charter": str(state["bootcamp"] / CONTROL_REL / "charter.json"),
        "source_terminal_event": str(state["bootcamp"] / SOURCE_CONTROL_REL / f"events/{EVENT8_NAME}.json"),
        "event2": str(state["bootcamp"] / CONTROL_REL / f"events/{EVENT2_NAME}.json"),
        "allowed_version_ids": ALLOWED_VERSION_IDS,
        "fresh_search_pairs": 90,
        "protected_rows_opened": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="Corrected root2 v0005 campaign")
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="preflight")
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
