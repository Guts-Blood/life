#!/usr/bin/env python3
"""Offline integrity and reward/advantage audit for Day 25 trajectory ledgers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import day25_contract as contract
from day25_reward_adapter import day24


class Day25RewardAuditError(ValueError):
    """A persisted Day 25 reward batch cannot be reproduced exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day25RewardAuditError(message)


def _same_float(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def audit_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(batches), "reward ledger is empty")
    complete = 0
    rollout_only = 0
    aborted = 0
    evidence_attempts = 0
    terminal_records: list[dict[str, Any]] = []
    policy_versions: set[str] = set()
    run_ids: set[str] = set()
    batch_hashes: list[str] = []
    for batch in batches:
        contract.verify_seal(batch, "batch_sha256", "Day 25 reward batch")
        _require(batch.get("schema_name") == "day25.coding_reward_batch", "wrong ledger schema")
        _require(batch.get("group_size") == contract.GROUP_SIZE, "ledger group size drifted")
        status = batch.get("status")
        _require(
            status
            in {"complete", "complete_rollout_only_stop", "aborted_persistent_infra"},
            "unknown batch status",
        )
        complete += status == "complete"
        rollout_only += status == "complete_rollout_only_stop"
        aborted += status == "aborted_persistent_infra"
        _require(
            bool(batch.get("atomic_scalar_returned")) == (status == "complete"),
            "atomic return status drifted",
        )
        policy_versions.add(str(batch["policy_version"]))
        run_ids.add(str(batch["run_id"]))
        batch_hashes.append(str(batch["batch_sha256"]))
        records = batch.get("records")
        _require(isinstance(records, list) and records, "batch records are missing")
        for record in records:
            trajectory = record["trajectory"]
            attempts = record["attempts"]
            day24.verify_seal(trajectory, "trajectory_sha256", "trajectory")
            _require(isinstance(attempts, list) and 1 <= len(attempts) <= 2, "retry count drifted")
            for evidence in attempts:
                day24.verify_seal(evidence, "evidence_sha256", "verifier evidence")
                evidence_attempts += 1
            terminal = attempts[-1]
            _require(
                terminal["evidence_sha256"] == record["terminal_evidence_sha256"],
                "terminal evidence pointer drifted",
            )
            reward = record.get("reward")
            if terminal["status"] == "infra_error":
                _require(status == "aborted_persistent_infra", "infra terminal escaped abort")
                _require(reward is None, "infra terminal received a reward")
            else:
                _require(isinstance(reward, Mapping), "eligible terminal reward is missing")
                day24.verify_seal(reward, "reward_sha256", "reward")
                expected = day24.score_verification(terminal, contract.REWARD_POLICY)
                _require(reward == expected, "offline reward recomputation mismatch")
                terminal_records.append(
                    {"trajectory": trajectory, "reward": reward, "batch": batch}
                )

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in terminal_records:
        group_id = str(record["trajectory"]["group_id"])
        groups.setdefault(group_id, []).append(record)
    reduced: list[dict[str, Any]] = []
    for group_id, records in sorted(groups.items()):
        _require(len(records) == contract.GROUP_SIZE, f"incomplete complete group: {group_id}")
        ordered = sorted(records, key=lambda record: int(record["trajectory"]["group_ordinal"]))
        _require(
            [int(record["trajectory"]["group_ordinal"]) for record in ordered]
            == list(range(contract.GROUP_SIZE)),
            f"group ordinals drifted: {group_id}",
        )
        reduced.append(
            day24.reduce_group(
                [record["reward"] for record in ordered], contract.REWARD_POLICY
            )
        )
    result: dict[str, Any] = {
        "schema_name": "day25.coding_reward_offline_audit",
        "schema_version": 1,
        "status": "pass",
        "batches": len(batches),
        "complete_batches": complete,
        "complete_rollout_only_stop_batches": rollout_only,
        "aborted_persistent_infra_batches": aborted,
        "evidence_attempts": evidence_attempts,
        "complete_trajectories": len(terminal_records),
        "complete_groups": len(reduced),
        "run_ids": sorted(run_ids),
        "policy_versions": sorted(policy_versions),
        "ordered_batch_hashes_sha256": contract.object_sha256(batch_hashes),
        "group_reducer": {
            "version": day24.GROUP_REDUCER_VERSION,
            "ddof": day24.GROUP_REDUCER_DDOF,
            "epsilon": day24.GROUP_REDUCER_EPSILON,
        },
        "groups": reduced,
    }
    result["audit_sha256"] = contract.object_sha256(result)
    return result


def audit_path(path: Path) -> dict[str, Any]:
    return audit_batches(contract.load_jsonl(path.resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_path(args.ledger)
        if args.output is not None:
            contract.write_atomic(args.output, contract.json_bytes(result), overwrite=True)
    except (OSError, contract.Day25ContractError, Day25RewardAuditError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
