#!/usr/bin/env python3
"""Join Day 22 smoke evidence and fail closed before formal DPO use.

This assembler emits pair-level audited records only when execution and
processor evidence pass.  The resulting dataset remains an audited smoke
artifact: formal readiness is independently gated on scale, source mix, human
review, secure execution, and promoted-S1/on-policy provenance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import day22_contract as contract


ASSEMBLY_SCHEMA_VERSION = 1
REVIEW_SEED = "day22-blind-review-v1"
REVIEW_LIMIT = 50
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Day22AssemblyError(ValueError):
    """An upstream identity, join, eligibility, or output invariant failed."""


def object_sha256(value: Any) -> str:
    return contract.object_sha256(value)


def text_sha256(value: str) -> str:
    return contract.text_sha256(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_artifact_path(path: Path) -> str:
    """Return a checkout-independent artifact identity for committed manifests."""
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "post-training-30-day-bootcamp":
            return resolved.relative_to(parent).as_posix()
    return resolved.name


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22AssemblyError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Day22AssemblyError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Day22AssemblyError(f"{label} must be non-empty text without NUL")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Day22AssemblyError(f"{label} must be a lowercase bare SHA-256")
    return value


def _task_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise Day22AssemblyError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise Day22AssemblyError(f"{label} must be a non-negative integer") from error
    _require(result >= 0 and str(value).strip() == str(result), f"{label} is not canonical")
    return result


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    _require(expected == actual, f"{label}.{field} does not bind its contents")
    return expected


def _equal(left: Any, right: Any, label: str) -> None:
    _require(left == right, f"{label} drifted across inputs")


def _index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = _text(row.get(key), f"{label}.{key}")
        _require(value not in result, f"duplicate {label}.{key}: {value}")
        result[value] = row
    return result


def _seed_index(seed_manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    _verify_self_hash(seed_manifest, "manifest_sha256", "seed_manifest")
    header = _mapping(seed_manifest.get("header"), "seed_manifest.header")
    _require(
        header.get("schema_name") == "day22.mbpp_seed_manifest",
        "seed manifest schema drifted",
    )
    records = seed_manifest.get("records")
    _require(isinstance(records, list) and records, "seed manifest records are empty")
    result: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        row = _mapping(raw, f"seed record {index}")
        _verify_self_hash(row, "record_sha256", f"seed record {index}")
        task_id = _task_id(row.get("task_id"), f"seed record {index}.task_id")
        _require(task_id not in result, f"duplicate seed task_id: {task_id}")
        _equal(row.get("task_family_id"), f"mbpp:task:{task_id}", "seed family")
        result[task_id] = row
    return result


def _candidate_index(
    candidate_rows: Sequence[Mapping[str, Any]],
    seeds: Mapping[int, Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    inherited_fields = (
        "task_family_id",
        "split",
        "family_keys",
        "source",
        "problem",
        "prompt",
        "code_prefix",
        "entry_point",
        "tests",
    )
    for index, raw in enumerate(candidate_rows):
        row = _mapping(raw, f"candidate row {index}")
        _verify_self_hash(row, "row_sha256", f"candidate row {index}")
        _require(
            row.get("schema_name") == "day22.mbpp_replay_candidates",
            f"candidate row {index} schema drifted",
        )
        task_id = _task_id(row.get("task_id"), f"candidate row {index}.task_id")
        _require(task_id in seeds, f"candidate task {task_id} is absent from seed")
        _require(task_id not in result, f"duplicate candidate task_id: {task_id}")
        seed = seeds[task_id]
        for field in inherited_fields:
            _equal(row.get(field), seed.get(field), f"task {task_id} {field}")

        chosen = _mapping(row.get("chosen"), f"task {task_id} chosen")
        _verify_self_hash(chosen, "candidate_sha256", f"task {task_id} chosen")
        for field in ("text", "sha256", "ast_sha256"):
            _equal(
                chosen.get(field),
                seed.get("canonical_chosen", {}).get(field),
                f"task {task_id} chosen.{field}",
            )
        candidate_ids = {_text(chosen.get("candidate_id"), "chosen.candidate_id")}
        mutants = row.get("mutants")
        _require(isinstance(mutants, list), f"task {task_id} mutants must be a list")
        for mutant_index, raw_mutant in enumerate(mutants):
            mutant = _mapping(raw_mutant, f"task {task_id} mutant {mutant_index}")
            _verify_self_hash(
                mutant, "candidate_sha256", f"task {task_id} mutant {mutant_index}"
            )
            candidate_id = _text(mutant.get("candidate_id"), "mutant.candidate_id")
            _require(candidate_id not in candidate_ids, f"duplicate candidate_id: {candidate_id}")
            candidate_ids.add(candidate_id)
        result[task_id] = row
    _require(bool(result), "candidate input is empty")
    return result


def _candidate_by_id(row: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    candidates = [row.get("chosen"), *(row.get("mutants") or [])]
    matches = [item for item in candidates if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id]
    _require(len(matches) == 1, f"candidate_id {candidate_id!r} does not resolve uniquely")
    return matches[0]


def _normalized_origin(value: Any) -> str:
    origin = _text(value, "candidate origin")
    aliases = {
        "mbpp_canonical_solution": "mbpp_canonical",
        "canonical": "mbpp_canonical",
        "deterministic_ast_mutation": "deterministic_mutation",
    }
    return aliases.get(origin, origin)


def _replay_index(
    replay_rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[int, Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    result: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = {}
    used_problems: set[str] = set()
    for index, raw in enumerate(replay_rows):
        row = _mapping(raw, f"replay row {index}")
        _verify_self_hash(row, "row_sha256", f"replay row {index}")
        _require(
            row.get("schema_name") == "day22.mbpp_replay_pair",
            f"replay row {index} schema drifted",
        )
        pair_id = _text(row.get("pair_id"), f"replay row {index}.pair_id")
        _require(pair_id not in result, f"duplicate replay pair_id: {pair_id}")
        task_id = _task_id(row.get("task_id"), f"replay row {index}.task_id")
        _require(task_id in candidates, f"replay task {task_id} is absent from candidates")
        candidate_row = candidates[task_id]
        family_id = _text(row.get("family_id"), "replay.family_id")
        _equal(family_id, candidate_row.get("task_family_id"), f"pair {pair_id} family")
        problem_family = str(candidate_row["family_keys"]["problem"])
        _require(problem_family not in used_problems, f"multiple replay pairs for {problem_family}")
        used_problems.add(problem_family)
        for field in ("split", "code_prefix", "tests"):
            _equal(row.get(field), candidate_row.get(field), f"pair {pair_id} {field}")
        _equal(row.get("tests_sha256"), candidate_row["tests"]["sha256"], f"pair {pair_id} tests hash")

        resolved: dict[str, Mapping[str, Any]] = {}
        for side in ("chosen", "rejected"):
            replay_candidate = _mapping(row.get(side), f"pair {pair_id} {side}")
            candidate_id = _text(replay_candidate.get("candidate_id"), f"pair {pair_id} {side}.candidate_id")
            source_candidate = _candidate_by_id(candidate_row, candidate_id)
            for field in ("text", "sha256"):
                _equal(
                    replay_candidate.get(field),
                    source_candidate.get(field),
                    f"pair {pair_id} {side}.{field}",
                )
            _equal(
                replay_candidate.get("origin"),
                _normalized_origin(source_candidate.get("origin")),
                f"pair {pair_id} {side}.origin",
            )
            if "mutation" in replay_candidate or "mutation" in source_candidate:
                _equal(
                    replay_candidate.get("mutation"),
                    source_candidate.get("mutation"),
                    f"pair {pair_id} {side}.mutation",
                )
            resolved[side] = source_candidate
        result[pair_id] = (row, resolved["chosen"], resolved["rejected"])
    _require(bool(result), "replay input is empty")
    return result


def _validate_sandbox_evidence(
    evidence: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> tuple[list[str], bool]:
    pair_id = str(replay["pair_id"])
    _verify_self_hash(evidence, "evidence_sha256", f"sandbox evidence {pair_id}")
    _require(
        evidence.get("domain") == "day22.mbpp_sandbox_pair_evidence",
        f"sandbox evidence {pair_id} domain drifted",
    )
    for field, expected in (
        ("pair_id", pair_id),
        ("task_id", str(replay["task_id"])),
        ("family_id", replay["family_id"]),
        ("source_tests_sha256", replay["tests_sha256"]),
    ):
        _equal(evidence.get(field), expected, f"sandbox evidence {pair_id} {field}")

    reasons: list[str] = []
    if evidence.get("source_tests_sha256") != evidence.get("test_sha256"):
        reasons.append("partial_test_manifest_execution")
    sandbox_digest = _sha256(evidence.get("sandbox_digest"), "sandbox_digest")
    execution = _mapping(evidence.get("execution"), f"sandbox evidence {pair_id}.execution")
    all_runs: list[Mapping[str, Any]] = []
    branch_statuses: dict[str, list[str]] = {}
    for side in ("chosen", "rejected"):
        branch = _mapping(execution.get(side), f"sandbox evidence {pair_id}.{side}")
        replay_candidate = _mapping(replay.get(side), f"replay {pair_id}.{side}")
        for field in ("candidate_id", "origin"):
            _equal(branch.get(field), replay_candidate.get(field), f"sandbox {pair_id} {side}.{field}")
        _equal(branch.get("response_sha256"), replay_candidate.get("sha256"), f"sandbox {pair_id} {side} response")
        runs = branch.get("runs")
        _require(isinstance(runs, list) and len(runs) == 2, f"sandbox {pair_id} {side} needs two runs")
        statuses: list[str] = []
        for attempt, raw_run in enumerate(runs, 1):
            run = _mapping(raw_run, f"sandbox {pair_id} {side} run {attempt}")
            _verify_self_hash(run, "run_sha256", f"sandbox {pair_id} {side} run {attempt}")
            for field, expected in (
                ("pair_id", pair_id),
                ("task_id", str(replay["task_id"])),
                ("family_id", replay["family_id"]),
                ("side", side),
                ("attempt", attempt),
                ("response_sha256", replay_candidate["sha256"]),
                ("source_tests_sha256", replay["tests_sha256"]),
                ("test_sha256", evidence.get("test_sha256")),
                ("sandbox_digest", sandbox_digest),
            ):
                _equal(run.get(field), expected, f"sandbox {pair_id} {side} run {attempt} {field}")
            sandbox = _mapping(run.get("sandbox"), f"sandbox {pair_id} {side} run {attempt}.sandbox")
            _equal(object_sha256(sandbox), sandbox_digest, f"sandbox {pair_id} identity digest")
            statuses.append(str(run.get("status")))
            all_runs.append(run)
        branch_statuses[side] = statuses
    _require(len({str(run.get("run_id")) for run in all_runs}) == 4, f"sandbox {pair_id} run IDs are not unique")

    eligible = (
        branch_statuses["chosen"] == ["pass", "pass"]
        and branch_statuses["rejected"] == ["wrong_answer", "wrong_answer"]
    )
    claimed = "eligible" if eligible else "quarantine"
    _equal(evidence.get("pair_execution_status"), claimed, f"sandbox {pair_id} eligibility")
    if not eligible:
        reasons.append("execution_not_pass_vs_stable_wrong_answer")

    secure = evidence.get("backend") == "e2b" and all(
        run["sandbox"].get("isolation") == "fresh_security_sandbox"
        and run["sandbox"].get("secure") is True
        and run["sandbox"].get("allow_internet_access") is False
        and run["sandbox"].get("fresh_sandbox_per_run") is True
        for run in all_runs
    )
    return reasons, secure


def _processor_join(
    audit: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> list[str]:
    pair_id = str(replay["pair_id"])
    _verify_self_hash(audit, "audit_sha256", f"processor audit {pair_id}")
    for field, expected in (
        ("pair_id", pair_id),
        ("task_id", str(replay["task_id"])),
        ("family_id", replay["family_id"]),
        ("chosen_candidate_id", replay["chosen"]["candidate_id"]),
        ("rejected_candidate_id", replay["rejected"]["candidate_id"]),
        ("chosen_response_sha256", replay["chosen"]["sha256"]),
        ("rejected_response_sha256", replay["rejected"]["sha256"]),
    ):
        _equal(audit.get(field), expected, f"processor audit {pair_id} {field}")
    return [] if audit.get("status") == "pass" else ["processor_audit_not_pass"]


def _adapt_execution(evidence: Mapping[str, Any]) -> dict[str, Any]:
    first_run = evidence["execution"]["chosen"]["runs"][0]
    timeout = first_run["sandbox"].get("wall_timeout_seconds")
    _require(
        isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0,
        "sandbox wall timeout is missing",
    )
    execution: dict[str, Any] = {
        "test_sha256": evidence["test_sha256"],
        "sandbox_digest": evidence["sandbox_digest"],
        "timeout_seconds": timeout,
        "verifier": {
            "name": "score_day22_mbpp_sandbox",
            "version": "day22.mbpp_sandbox_pair_evidence.v1",
        },
        "source_sandbox_evidence_sha256": evidence["evidence_sha256"],
        "source_backend": evidence["backend"],
        "chosen": copy.deepcopy(evidence["execution"]["chosen"]),
        "rejected": copy.deepcopy(evidence["execution"]["rejected"]),
    }
    execution["evidence_sha256"] = object_sha256(execution)
    return execution


def _is_synthetic(candidate: Mapping[str, Any]) -> bool:
    return str(candidate.get("origin")) in {
        "deterministic_ast_mutation",
        "deterministic_mutation",
        "synthetic_mutation",
    }


def _is_promoted_s1_on_policy(candidate: Mapping[str, Any]) -> bool:
    generator = candidate.get("generator")
    return (
        isinstance(generator, Mapping)
        and generator.get("promoted_s1") is True
        and generator.get("on_policy") is True
        and isinstance(generator.get("checkpoint"), str)
        and bool(generator["checkpoint"].strip())
    )


def _make_pair(
    candidate_row: Mapping[str, Any],
    replay: Mapping[str, Any],
    chosen_source: Mapping[str, Any],
    rejected_source: Mapping[str, Any],
    evidence: Mapping[str, Any],
    processor_audit: Mapping[str, Any],
    *,
    secure_sandbox: bool,
) -> dict[str, Any]:
    synthetic = _is_synthetic(rejected_source)
    on_policy = _is_promoted_s1_on_policy(chosen_source) and _is_promoted_s1_on_policy(rejected_source)
    mutation = rejected_source.get("mutation")
    version = mutation.get("version") if isinstance(mutation, Mapping) else "model_pair_v1"
    pair: dict[str, Any] = {
        "schema_name": contract.PAIR_SCHEMA_NAME,
        "schema_version": contract.PAIR_SCHEMA_VERSION,
        "pair_id": replay["pair_id"],
        "pair_status": contract.FINAL_PAIR_STATUS,
        "dataset_role": "audited_smoke_not_formal_dpo_data",
        "split": candidate_row["split"],
        "task_id": int(candidate_row["task_id"]),
        "language": "python",
        "family_keys": copy.deepcopy(candidate_row["family_keys"]),
        "source": copy.deepcopy(candidate_row["source"]),
        "problem": copy.deepcopy(candidate_row["problem"]),
        "prompt": copy.deepcopy(candidate_row["prompt"]),
        "code_prefix": candidate_row["code_prefix"],
        "entry_point": candidate_row["entry_point"],
        "chosen": {
            "candidate_id": chosen_source["candidate_id"],
            "origin": chosen_source["origin"],
            "text": chosen_source["text"],
            "sha256": chosen_source["sha256"],
            "ast_sha256": chosen_source["ast_sha256"],
        },
        "rejected": {
            "candidate_id": rejected_source["candidate_id"],
            "origin": rejected_source["origin"],
            "text": rejected_source["text"],
            "sha256": rejected_source["sha256"],
            "ast_sha256": rejected_source["ast_sha256"],
        },
        "creation": {
            "method": "canonical_vs_deterministic_ast_mutation" if synthetic else "on_policy_model_pair",
            "version": str(version),
            "synthetic": synthetic,
            "promoted_s1_on_policy": on_policy,
        },
        "quality_flags": [
            flag
            for flag, present in (
                ("synthetic_mutation", synthetic),
                ("insecure_smoke_backend", not secure_sandbox),
                ("human_blind_review_pending", True),
                ("not_promoted_s1_on_policy", not on_policy),
            )
            if present
        ],
        "tests": copy.deepcopy(candidate_row["tests"]),
        "execution": _adapt_execution(evidence),
        "processor_audit": copy.deepcopy(processor_audit),
    }
    pair = contract.seal_pair(pair)
    try:
        contract.validate_pair(pair, mode="final")
    except contract.Day22ContractError as error:
        raise Day22AssemblyError(f"assembled pair {pair['pair_id']} violates contract: {error}") from error
    return pair


def build_blind_review(
    pairs: Sequence[Mapping[str, Any]],
    *,
    limit: int = REVIEW_LIMIT,
    seed: str = REVIEW_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(isinstance(limit, int) and not isinstance(limit, bool) and limit > 0, "review limit must be positive")
    seed = _text(seed, "review seed")
    ranked = sorted(
        pairs,
        key=lambda pair: (text_sha256(f"{seed}\0sample\0{pair['pair_id']}"), str(pair["pair_id"])),
    )[:limit]
    worksheet: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for pair in ranked:
        flip = int(text_sha256(f"{seed}\0flip\0{pair['pair_id']}"), 16) % 2
        primary = ("rejected", "chosen") if flip else ("chosen", "rejected")
        for presentation, sides in (("primary", primary), ("swapped", primary[::-1])):
            opaque = "review:" + text_sha256(
                f"{seed}\0item\0{pair['pair_id']}\0{presentation}"
            )[:24]
            worksheet.append(
                {
                    "schema_name": "day22.blind_review_item",
                    "schema_version": 1,
                    "review_item_id": opaque,
                    "prompt": pair["prompt"]["text"],
                    "response_a": pair[sides[0]]["text"],
                    "response_b": pair[sides[1]]["text"],
                    "rubric_version": "day22.code_pair_blind_v1",
                    "allowed_verdicts": ["A", "B", "tie", "ambiguous", "reject"],
                    "verdict": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            key_items.append(
                {
                    "review_item_id": opaque,
                    "pair_id": pair["pair_id"],
                    "presentation": presentation,
                    "a_side": sides[0],
                    "b_side": sides[1],
                }
            )
    worksheet.sort(key=lambda item: text_sha256(f"{seed}\0order\0{item['review_item_id']}"))
    key: dict[str, Any] = {
        "schema_name": "day22.blind_review_concealed_key",
        "schema_version": 1,
        "handling": "keep_separate_from_reviewer_worksheet",
        "review_seed_sha256": text_sha256(seed),
        "unique_pairs": len(ranked),
        "presentations": len(worksheet),
        "items": sorted(key_items, key=lambda item: item["review_item_id"]),
    }
    key["key_sha256"] = object_sha256(key)
    return worksheet, key


def assemble_smoke(
    seed_manifest: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    sandbox_rows: Sequence[Mapping[str, Any]],
    processor_rows: Sequence[Mapping[str, Any]],
    *,
    review_limit: int = REVIEW_LIMIT,
    review_seed: str = REVIEW_SEED,
    input_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seeds = _seed_index(seed_manifest)
    candidates = _candidate_index(candidate_rows, seeds)
    replays = _replay_index(replay_rows, candidates)
    sandbox_by_pair = _index_unique(sandbox_rows, "pair_id", "sandbox evidence")
    processor_by_pair = _index_unique(processor_rows, "pair_id", "processor audit")
    expected_ids = set(replays)
    _require(set(sandbox_by_pair) == expected_ids, "sandbox evidence pair IDs do not exactly match replay input")
    _require(set(processor_by_pair) == expected_ids, "processor audit pair IDs do not exactly match replay input")

    accepted: list[dict[str, Any]] = []
    quarantine: Counter[str] = Counter()
    secure_by_pair: dict[str, bool] = {}
    for pair_id in sorted(replays):
        replay, chosen_source, rejected_source = replays[pair_id]
        evidence = sandbox_by_pair[pair_id]
        audit = processor_by_pair[pair_id]
        evidence_reasons, secure = _validate_sandbox_evidence(evidence, replay)
        audit_reasons = _processor_join(audit, replay)
        reasons = evidence_reasons + audit_reasons
        if reasons:
            quarantine.update(set(reasons))
            continue
        task_id = _task_id(replay["task_id"], f"pair {pair_id}.task_id")
        secure_by_pair[pair_id] = secure
        accepted.append(
            _make_pair(
                candidates[task_id],
                replay,
                chosen_source,
                rejected_source,
                evidence,
                audit,
                secure_sandbox=secure,
            )
        )
    accepted.sort(key=lambda pair: str(pair["pair_id"]))
    if accepted:
        contract_summary = contract.validate_manifest(accepted, mode="final")
    else:
        contract_summary = {
            "records": 0,
            "split_counts": {split: 0 for split in contract.SPLITS},
            "pair_status_counts": {},
            "ordered_pair_hashes_sha256": object_sha256([]),
        }

    split_ids: dict[str, Any] = {
        "schema_name": "day22.audited_smoke_split_ids",
        "schema_version": 1,
        "dataset_role": "audited_smoke_not_formal_dpo_data",
        "train": sorted(str(pair["pair_id"]) for pair in accepted if pair["split"] == "train"),
        "dev": sorted(str(pair["pair_id"]) for pair in accepted if pair["split"] == "dev"),
        "heldout": sorted(str(pair["pair_id"]) for pair in accepted if pair["split"] == "heldout"),
    }
    split_ids["split_ids_sha256"] = object_sha256(split_ids)
    worksheet, review_key = build_blind_review(
        accepted, limit=review_limit, seed=review_seed
    )

    count = len(accepted)
    synthetic_count = sum(pair["creation"]["synthetic"] is True for pair in accepted)
    synthetic_fraction = synthetic_count / count if count else None
    secure_count = sum(secure_by_pair.get(str(pair["pair_id"]), False) for pair in accepted)
    non_synthetic = [pair for pair in accepted if not pair["creation"]["synthetic"]]
    on_policy_count = sum(
        pair["creation"]["promoted_s1_on_policy"] is True for pair in non_synthetic
    )
    gates: dict[str, Any] = {
        "minimum_accepted_pairs": {
            "required": 200,
            "observed": count,
            "passed": count >= 200,
        },
        "synthetic_pair_fraction": {
            "maximum": 0.20,
            "observed": synthetic_fraction,
            "synthetic_pairs": synthetic_count,
            "passed": synthetic_fraction is not None and synthetic_fraction <= 0.20,
        },
        "human_blind_review": {
            "required_unique_pairs": 50,
            "observed_completed_unique_pairs": 0,
            "worksheet_unique_pairs": review_key["unique_pairs"],
            "passed": False,
            "reason": "worksheet generated; adjudicated review evidence is not an assembler input",
        },
        "secure_sandbox": {
            "required": "all accepted pairs replayed twice per side in attested E2B",
            "observed_secure_pairs": secure_count,
            "passed": count > 0 and secure_count == count,
        },
        "promoted_s1_on_policy_source": {
            "required": "all non-synthetic pairs carry promoted_s1=true and on_policy=true generator evidence",
            "observed_non_synthetic_pairs": len(non_synthetic),
            "observed_promoted_s1_on_policy_pairs": on_policy_count,
            "passed": bool(non_synthetic) and on_policy_count == len(non_synthetic),
        },
    }
    readiness = "READY" if all(gate["passed"] is True for gate in gates.values()) else "BLOCKED"
    manifest: dict[str, Any] = {
        "schema_name": "day22.audited_smoke_assembly_manifest",
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "dataset_role": "audited_smoke_not_formal_dpo_data",
        "readiness": readiness,
        "formal_dpo_ready": readiness == "READY",
        "warning": "Synthetic smoke validates the evidence pipeline; it is not formal DPO training data.",
        "inputs": dict(input_identities or {}),
        "join_counts": {
            "seed_records": len(seeds),
            "candidate_records": len(candidates),
            "replay_records": len(replays),
            "sandbox_records": len(sandbox_rows),
            "processor_records": len(processor_rows),
            "accepted_pairs": count,
            "quarantined_pairs": len(replays) - count,
            "quarantine_reason_counts": dict(sorted(quarantine.items())),
        },
        "contract_summary": contract_summary,
        "readiness_gates": gates,
        "content_identities": {
            "ordered_pair_hashes_sha256": contract_summary["ordered_pair_hashes_sha256"],
            "split_ids_sha256": split_ids["split_ids_sha256"],
            "worksheet_sha256": object_sha256(worksheet),
            "concealed_review_key_sha256": review_key["key_sha256"],
        },
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return {
        "pairs": accepted,
        "split_ids": split_ids,
        "manifest": manifest,
        "review_worksheet": worksheet,
        "review_key": review_key,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Day22AssemblyError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise Day22AssemblyError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day22AssemblyError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day22AssemblyError(f"invalid JSONL: {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise Day22AssemblyError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise Day22AssemblyError(f"JSONL is empty: {path}")
    return rows


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(contract.canonical_json(row) + b"\n" for row in rows)


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise Day22AssemblyError(f"refusing to overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise Day22AssemblyError(f"output appeared while writing: {path}") from error
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--sandbox-evidence", type=Path, required=True)
    parser.add_argument("--processor-audit", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--split-ids-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--review-worksheet-output", type=Path, required=True)
    parser.add_argument("--review-key-output", type=Path, required=True)
    parser.add_argument("--review-limit", type=int, default=REVIEW_LIMIT)
    parser.add_argument("--review-seed", default=REVIEW_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = {
            "seed": args.seed.resolve(),
            "candidates": args.candidates.resolve(),
            "replay_input": args.replay_input.resolve(),
            "sandbox_evidence": args.sandbox_evidence.resolve(),
            "processor_audit": args.processor_audit.resolve(),
        }
        identities = {
            name: {"path": portable_artifact_path(path), "file_sha256": file_sha256(path)}
            for name, path in inputs.items()
        }
        result = assemble_smoke(
            _load_json(inputs["seed"]),
            _load_jsonl(inputs["candidates"]),
            _load_jsonl(inputs["replay_input"]),
            _load_jsonl(inputs["sandbox_evidence"]),
            _load_jsonl(inputs["processor_audit"]),
            review_limit=args.review_limit,
            review_seed=args.review_seed,
            input_identities=identities,
        )
        outputs = {
            args.pairs_output: _jsonl_bytes(result["pairs"]),
            args.split_ids_output: _json_bytes(result["split_ids"]),
            args.review_worksheet_output: _jsonl_bytes(result["review_worksheet"]),
            args.review_key_output: _json_bytes(result["review_key"]),
        }
        resolved = [path.resolve() for path in [*outputs, args.manifest_output]]
        _require(len(set(resolved)) == len(resolved), "output paths must be distinct")
        if not args.overwrite:
            for path in resolved:
                _require(not path.exists(), f"refusing to overwrite: {path}")
        manifest = dict(result["manifest"])
        manifest.pop("manifest_sha256", None)
        manifest["output_files"] = {
            path.name: {
                "path": portable_artifact_path(path),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in outputs.items()
        }
        manifest["manifest_sha256"] = object_sha256(manifest)
        outputs[args.manifest_output] = _json_bytes(manifest)
        for path, payload in outputs.items():
            _write_atomic(path, payload, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "accepted_pairs": len(result["pairs"]),
                    "readiness": manifest["readiness"],
                    "manifest": str(args.manifest_output.resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (Day22AssemblyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
