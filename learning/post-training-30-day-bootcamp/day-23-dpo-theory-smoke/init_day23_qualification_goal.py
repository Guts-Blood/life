#!/usr/bin/env python3
"""Close goal-0003 and initialize the independent Day 23 qualification goal.

The initializer validates the selected csearch-v0001 candidate from its sealed
search evidence, emits one append-only closeout event for goal-0003, and then
freezes goal-0004 plus its two initial events.  It never opens dev or heldout
rows and it never changes an existing control artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import eval_day23_csearch_candidate as csearch_eval
import run_day23_qwen35_gpu_stage as day23_gpu


DAY23_DIR = Path(__file__).resolve().parent
BOOTCAMP_ROOT = DAY23_DIR.parent
SOURCE_GOAL_ID = "goal-0003-day23-dpo-candidate-search"
SOURCE_VERSION_ID = "csearch-v0001"
GOAL_ID = "goal-0004-day23-dpo-qualification"
VERSION_ID = "qual-v0001"
SOURCE_CAMPAIGN_SHA256 = (
    "850b73b440c412125be47ccbe1d892a537ea8e34d7b5a23de58ab50708a3ff45"
)
SOURCE_SELECTION_SHA256 = (
    "c4dee82793390ac54adbe720adeee7d6f5b7e4750ac563d9d17e3004ef15d438"
)
SELECTED_CANDIDATE = "candidate_lr_4p3e_6_step_20"
SELECTED_RUN_ID = "search_lr_4p3e_6"
SELECTED_LR = 4.3e-6
SELECTED_STEP = 20

SOURCE_CONTROL_REL = Path("rsi-control/charters") / SOURCE_GOAL_ID
CONTROL_REL = Path("rsi-control/charters") / GOAL_ID
SOURCE_EVENT2_REL = SOURCE_CONTROL_REL / "events/event-000002-source-terminal-imported.json"
SOURCE_EVENT3_NAME = "event-000003-version-closed-candidate-selected"
EVENT1_NAME = "event-000001-charter-frozen"
EVENT2_NAME = "event-000002-source-candidate-imported"


class QualificationInitError(RuntimeError):
    """A source-evidence or append-only qualification invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationInitError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationInitError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def verify(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"{label}.{field} missing")
    require(day23_gpu.object_sha256(value, field) == expected, f"{label} self hash drifted")
    return expected


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = day23_gpu.object_sha256(result)
    return result


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def identity(
    path: Path,
    *,
    self_field: str | None = None,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved.relative_to(relative_to)) if relative_to else str(resolved),
        "file_sha256": day23_gpu.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if self_field is not None:
        result["content_sha256"] = verify(load(resolved, str(resolved)), self_field, str(resolved))
    return result


def payload_identity(
    path: Path,
    value: Mapping[str, Any],
    self_field: str,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    payload = json_bytes(value)
    return {
        "path": str(path.resolve().relative_to(relative_to.resolve())),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "content_sha256": value[self_field],
    }


def _relative_event_identity(
    path: Path, value: Mapping[str, Any], *, goal_root: Path
) -> dict[str, Any]:
    full = payload_identity(path, value, "event_sha256", relative_to=goal_root)
    return {
        "path": full["path"],
        "file_sha256": full["file_sha256"],
        "event_sha256": full["content_sha256"],
        "sequence": value["sequence"],
    }


def _source_state(campaign_arg: Path, selection_arg: Path) -> dict[str, Any]:
    campaign_path = campaign_arg.expanduser().resolve(strict=True)
    campaign = load(campaign_path, "source csearch campaign")
    campaign_sha = verify(campaign, "campaign_sha256", "source csearch campaign")
    require(
        campaign.get("schema_name") == "day23.csearch_v0001_gpu_campaign"
        and campaign.get("schema_version") == 1
        and campaign.get("goal_id") == SOURCE_GOAL_ID
        and campaign.get("version_id") == SOURCE_VERSION_ID
        and campaign_sha == SOURCE_CAMPAIGN_SHA256,
        "source csearch campaign identity drifted",
    )
    run_root = Path(str(campaign.get("remote_run_root"))).resolve(strict=True)
    bootcamp = Path(str(campaign.get("bootcamp_root"))).resolve(strict=True)
    require(bootcamp == BOOTCAMP_ROOT.resolve(strict=True), "initializer bootcamp binding drifted")
    require(campaign_path == run_root / "binding/gpu-campaign.json", "source campaign path is not canonical")

    selection_path = selection_arg.expanduser().resolve(strict=True)
    selection = load(selection_path, "source csearch selection")
    selection_sha = verify(selection, "selection_sha256", "source csearch selection")
    require(selection_path == run_root / "evidence/selection/search-selection.json", "source selection path is not canonical")
    require(
        selection.get("schema_name") == "day23.csearch_v0001_search_selection"
        and selection.get("schema_version") == 1
        and selection.get("campaign_sha256") == campaign_sha
        and selection_sha == SOURCE_SELECTION_SHA256
        and selection.get("status") == "selected"
        and selection.get("candidate_pool_frozen") is True
        and selection.get("candidate_count") == 2
        and selection.get("eligible_count") == 1
        and selection.get("selected_candidate") == SELECTED_CANDIDATE
        and selection.get("selected_run_id") == SELECTED_RUN_ID
        and selection.get("selected_checkpoint_step") == SELECTED_STEP
        and float(selection.get("selected_learning_rate")) == SELECTED_LR,
        "source selected candidate drifted",
    )
    boundary = mapping(selection.get("claim_boundary"), "source selection boundary")
    require(
        boundary.get("dev_consumed") is False
        and boundary.get("heldout_consumed") is False
        and boundary.get("runner_up_fallback_forbidden") is True,
        "source selection crossed a protected boundary",
    )

    # This is the authoritative independent recomputation of both search
    # evaluations and the selected candidate.  It opens only the already-used
    # csearch search30 tranche.
    context = csearch_eval.validate_campaign(campaign_path)
    recomputed = csearch_eval.validate_search_selection(context, selection_path)
    require(recomputed == selection, "source selection failed exact recomputation")

    source_goal_root = bootcamp / SOURCE_CONTROL_REL
    source_charter_path = source_goal_root / "charter.json"
    source_charter = load(source_charter_path, "goal-0003 charter")
    source_charter_sha = verify(source_charter, "charter_sha256", "goal-0003 charter")
    require(
        source_charter.get("schema_name") == "day23.csearch_goal_charter"
        and source_charter.get("goal_id") == SOURCE_GOAL_ID
        and source_charter.get("status") == "frozen",
        "goal-0003 charter drifted",
    )
    state_machine = mapping(source_charter.get("state_machine"), "goal-0003 state machine")
    require(
        state_machine.get("success_terminal_state") == "candidate_frozen_search_only"
        and state_machine.get("qualified_state_authorized") is False,
        "goal-0003 success boundary drifted",
    )
    protected = mapping(source_charter.get("protected_data"), "goal-0003 protected data")
    require(
        protected.get("raw_row_access_authorized") is False
        and protected.get("claim_creation_authorized") is False
        and protected.get("qualification_claim_allowed") is False
        and protected.get("lease_reset_under_new_goal_forbidden") is True,
        "goal-0003 protected-data authority drifted",
    )
    dev_lease = dict(mapping(protected.get("dev"), "inherited dev lease"))
    heldout_lease = dict(mapping(protected.get("heldout"), "inherited heldout lease"))
    require(
        dev_lease.get("maximum_global_claims") == 1
        and heldout_lease.get("current_authorized_claims") == 0,
        "inherited protected-data lease budget drifted",
    )
    ledger_root = Path(str(protected.get("inherited_ledger_root")))
    require(not ledger_root.is_absolute() and ".." not in ledger_root.parts, "ledger root is not canonical")
    dev_claim_rel = Path(str(dev_lease.get("claim_path")))
    heldout_claim_rel = Path(str(heldout_lease.get("claim_path")))
    for relative, label in ((dev_claim_rel, "dev"), (heldout_claim_rel, "heldout")):
        require(not relative.is_absolute() and ".." not in relative.parts, f"{label} claim path is not canonical")
        require(not (bootcamp / "rsi-control" / relative).exists(), f"global {label} lease was already claimed")

    event2_path = bootcamp / SOURCE_EVENT2_REL
    event2 = load(event2_path, "goal-0003 event2")
    verify(event2, "event_sha256", "goal-0003 event2")
    require(
        event2.get("event_id") == "event-000002-source-terminal-imported"
        and event2.get("sequence") == 2
        and event2.get("state_after") == "iteration_open",
        "goal-0003 event2 drifted",
    )

    source_specs = mapping(campaign.get("run_specs"), "source run specs")
    receipt_identities: list[dict[str, Any]] = []
    completion_times: list[str] = []
    for run_id in ("search_lr_4p3e_6", "search_lr_4p5e_6"):
        spec = mapping(source_specs.get(run_id), f"source run {run_id}")
        receipt_path = Path(str(spec.get("success_receipt"))).resolve(strict=True)
        receipt = load(receipt_path, f"{run_id} receipt")
        verify(receipt, "receipt_sha256", f"{run_id} receipt")
        require(receipt.get("status") == "pass" and receipt.get("run_id") == run_id, f"{run_id} receipt drifted")
        receipt_identities.append(identity(receipt_path, self_field="receipt_sha256"))
        completion_times.append(str(receipt.get("completed_at_utc")))
    require(all(value and value != "None" for value in completion_times), "source completion time missing")

    evaluation_identities = [
        identity(Path(str(item["path"])), self_field="evaluation_sha256")
        for item in selection["candidates"]
    ]
    strict_path = run_root / "evidence/authority/strict-preunseal.json"
    strict = load(strict_path, "source strict pre-unseal receipt")
    verify(strict, "validation_sha256", "source strict pre-unseal receipt")
    require(strict.get("status") == "pass", "source strict authority receipt did not pass")
    search_claim_path = run_root / "evidence/search-selection/search-claim.json"
    search_claim = load(search_claim_path, "source search claim")
    verify(search_claim, "claim_sha256", "source search claim")

    full_train = dict(
        mapping(
            mapping(source_charter.get("search_exposure_ledger"), "source exposure ledger").get("full_train"),
            "source full154",
        )
    )
    require(
        full_train.get("records") == 154
        and full_train.get("file_sha256") == "2d75622077a0d2cb685f2314fdd4cc3959c0ed7f88a4aa3b3c88d6c988a871d9",
        "source full154 identity drifted",
    )
    return {
        "bootcamp": bootcamp,
        "run_root": run_root,
        "campaign": campaign,
        "campaign_path": campaign_path,
        "selection": selection,
        "selection_path": selection_path,
        "source_charter": source_charter,
        "source_charter_path": source_charter_path,
        "source_charter_sha": source_charter_sha,
        "event2": event2,
        "event2_path": event2_path,
        "dev_lease": dev_lease,
        "heldout_lease": heldout_lease,
        "protected": protected,
        "full_train": full_train,
        "receipts": receipt_identities,
        "evaluations": evaluation_identities,
        "strict": identity(strict_path, self_field="validation_sha256"),
        "search_claim": identity(search_claim_path, self_field="claim_sha256"),
        "created_at_utc": max(completion_times),
    }


def _documents(state: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any], str]]:
    bootcamp: Path = state["bootcamp"]
    source_goal_root = bootcamp / SOURCE_CONTROL_REL
    goal_root = bootcamp / CONTROL_REL
    producer = identity(Path(__file__).resolve(strict=True))
    source_charter_identity = identity(
        state["source_charter_path"], self_field="charter_sha256", relative_to=bootcamp
    )
    source_event2_identity = identity(
        state["event2_path"], self_field="event_sha256", relative_to=bootcamp
    )
    campaign_identity = identity(state["campaign_path"], self_field="campaign_sha256")
    selection_identity = identity(state["selection_path"], self_field="selection_sha256")
    created_at = state["created_at_utc"]

    source_event3_path = source_goal_root / f"events/{SOURCE_EVENT3_NAME}.json"
    event3 = seal(
        {
            "schema_name": "day23.csearch_event",
            "schema_version": 1,
            "event_id": SOURCE_EVENT3_NAME,
            "created_at_utc": created_at,
            "sequence": 3,
            "event_type": "version-closed-candidate-selected",
            "charter_id": SOURCE_GOAL_ID,
            "goal_id": SOURCE_GOAL_ID,
            "version_id": SOURCE_VERSION_ID,
            "state_before": "iteration_open",
            "state_after": "candidate_frozen_search_only",
            "prior_event": _relative_event_identity(
                state["event2_path"], state["event2"], goal_root=source_goal_root
            ),
            "authority_refs": [
                source_charter_identity,
                campaign_identity,
                selection_identity,
                state["strict"],
            ],
            "payload": {
                "campaign": campaign_identity,
                "selection": selection_identity,
                "training_receipts": state["receipts"],
                "search_evaluations": state["evaluations"],
                "search_claim": state["search_claim"],
                "status": "candidate_frozen_search_only",
                "selected_candidate": SELECTED_CANDIDATE,
                "selected_run_id": SELECTED_RUN_ID,
                "selected_learning_rate": SELECTED_LR,
                "selected_checkpoint_step": SELECTED_STEP,
                "source_search_checkpoint_is_evidence_only": True,
                "fresh_full154_refit_required": True,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    event3_identity = payload_identity(
        source_event3_path, event3, "event_sha256", relative_to=bootcamp
    )

    heldout = {
        key: value
        for key, value in state["heldout_lease"].items()
        if key != "claim_path"
    }
    if isinstance(heldout.get("identity"), Mapping):
        heldout["identity"] = {
            key: value
            for key, value in heldout["identity"].items()
            if key != "path"
        }
    charter_path = goal_root / "charter.json"
    charter = seal(
        {
            "schema_name": "day23.qualification_goal_charter",
            "schema_version": 1,
            "status": "frozen",
            "goal_id": GOAL_ID,
            "created_at_utc": created_at,
            "purpose": "fresh-S1 full154 refit and one-shot dev qualification of the frozen csearch winner",
            "source_candidate": {
                "source_goal_id": SOURCE_GOAL_ID,
                "source_version_id": SOURCE_VERSION_ID,
                "source_charter": source_charter_identity,
                "source_closeout_event": event3_identity,
                "source_campaign": campaign_identity,
                "source_selection": selection_identity,
                "selected_candidate": SELECTED_CANDIDATE,
                "selected_run_id": SELECTED_RUN_ID,
                "selected_learning_rate": SELECTED_LR,
                "selected_checkpoint_step": SELECTED_STEP,
                "search_checkpoint_is_initialization": False,
            },
            "allowed_version_ids": [VERSION_ID],
            "fixed_recipe": {
                "learning_rate": SELECTED_LR,
                "selected_candidate_checkpoint_step": SELECTED_STEP,
                "cosine_schedule_horizon": 30,
                "refit_seed": 20260820,
                "full154_fresh_s1_refit": True,
                "world_size": 2,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "nominal_global_train_batch_size": 32,
                "minimum_free_memory_fraction": 0.20,
            },
            "data_authority": {
                "full_train": state["full_train"],
                "dev": {
                    "lease": state["dev_lease"],
                    "role": "one_shot_finalist_qualification_not_hyperparameter_selection",
                    "raw_content_access_before_claim": False,
                },
                "heldout": {
                    "lease_without_claim_or_data_path": heldout,
                    "maximum_evaluations": 0,
                    "claim_authorized": False,
                    "runtime_path_disclosed": False,
                },
                "lease_reset_forbidden": True,
            },
            "execution_budget": {
                "maximum_refit_training_runs": 1,
                "maximum_optimizer_steps": 30,
                "maximum_dev_claims": 1,
                "maximum_dev_evaluations": 1,
                "maximum_heldout_claims": 0,
                "maximum_heldout_evaluations": 0,
                "runner_up_fallback": 0,
            },
            "dev_gate": {
                "pairs": 17,
                "minimum_positive_pairs": 13,
                "mean_reward_margin": ">0",
                "length_matched_mean_reward_margin": ">0",
                "threshold_relaxation_forbidden": True,
                "retry_after_claim": "forbidden",
            },
            "state_machine": {
                "initial_state": "uninitialized",
                "current_authorized_path": [
                    "charter_frozen",
                    "source_candidate_imported",
                    "refit_pending",
                    "one_shot_dev_pending",
                ],
                "success_state": "dev_qualified_guardrails_pending",
                "failure_state": "terminal_dev_not_qualified",
                "heldout_state_authorized": False,
            },
            "forbidden_actions": [
                "resume_or_initialize_from_any_search_checkpoint",
                "train_on_less_or_more_than_frozen_full154",
                "change_selected_learning_rate_or_checkpoint_step",
                "open_or_claim_dev_before_a_strict_refit_success_receipt",
                "evaluate_dev_more_than_once_or_try_a_runner_up",
                "open_or_claim_heldout",
                "claim_guardrails_or_final_heldout_qualification",
            ],
            "producer": producer,
        },
        "charter_sha256",
    )
    charter_identity = payload_identity(
        charter_path, charter, "charter_sha256", relative_to=bootcamp
    )

    event1_path = goal_root / f"events/{EVENT1_NAME}.json"
    event1 = seal(
        {
            "schema_name": "day23.qualification_event",
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
                "recipe_frozen": True,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    event1_identity = payload_identity(
        event1_path, event1, "event_sha256", relative_to=bootcamp
    )
    event2_path = goal_root / f"events/{EVENT2_NAME}.json"
    event2 = seal(
        {
            "schema_name": "day23.qualification_event",
            "schema_version": 1,
            "event_id": EVENT2_NAME,
            "created_at_utc": created_at,
            "sequence": 2,
            "event_type": "source-candidate-imported",
            "charter_id": GOAL_ID,
            "goal_id": GOAL_ID,
            "version_id": None,
            "state_before": "charter_frozen",
            "state_after": "source_candidate_imported",
            "prior_event": {
                "path": str(event1_path.relative_to(goal_root)),
                "file_sha256": event1_identity["file_sha256"],
                "event_sha256": event1["event_sha256"],
                "sequence": 1,
            },
            "authority_refs": [charter_identity, event3_identity, campaign_identity, selection_identity],
            "payload": {
                "source_goal_id": SOURCE_GOAL_ID,
                "source_terminal_state": "candidate_frozen_search_only",
                "source_closeout_event": event3_identity,
                "source_campaign": campaign_identity,
                "source_selection": selection_identity,
                "selected_candidate": SELECTED_CANDIDATE,
                "selected_learning_rate": SELECTED_LR,
                "selected_checkpoint_step": SELECTED_STEP,
                "next_version_id": VERSION_ID,
                "fresh_full154_refit_required": True,
                "dev_rows_opened": 0,
                "heldout_rows_opened": 0,
            },
        },
        "event_sha256",
    )
    return [
        (source_event3_path, event3, "event_sha256"),
        (charter_path, charter, "charter_sha256"),
        (event1_path, event1, "event_sha256"),
        (event2_path, event2, "event_sha256"),
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    state = _source_state(args.source_campaign, args.source_selection)
    documents = _documents(state)
    if args.mode in {"preflight", "build"}:
        for path, _, _ in documents:
            require(not path.exists(), f"append-only output already exists: {path}")
    if args.mode == "preflight":
        return {
            "status": "preflight_pass",
            "source_goal_id": SOURCE_GOAL_ID,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_selection_sha256": SOURCE_SELECTION_SHA256,
            "selected_candidate": SELECTED_CANDIDATE,
            "goal_id": GOAL_ID,
            "version_id": VERSION_ID,
            "source_selection_recomputed": True,
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
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
            require(
                actual == expected and path.read_bytes() == json_bytes(expected),
                f"sealed control artifact drifted: {path}",
            )
        status = "check_pass"
    return {
        "status": status,
        "source_goal_state": "candidate_frozen_search_only",
        "goal_id": GOAL_ID,
        "version_id": VERSION_ID,
        "charter": str(state["bootcamp"] / CONTROL_REL / "charter.json"),
        "source_closeout_event": str(
            state["bootcamp"] / SOURCE_CONTROL_REL / f"events/{SOURCE_EVENT3_NAME}.json"
        ),
        "dev_rows_opened": 0,
        "heldout_rows_opened": 0,
    }


def _self_test() -> None:
    value = seal({"schema_name": "test", "status": "pass"}, "sha256")
    require(day23_gpu.object_sha256(value, "sha256") == value["sha256"], "self hash failed")
    require(SELECTED_LR == 4.3e-6 and SELECTED_STEP == 20, "selected recipe drifted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--source-selection", type=Path)
    parser.add_argument("--mode", choices=("preflight", "build", "check"), default="preflight")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            result = {"status": "pass", "scope": "stdlib_self_test"}
        else:
            require(args.source_campaign is not None, "--source-campaign is required")
            require(args.source_selection is not None, "--source-selection is required")
            result = execute(args)
    except BaseException as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
