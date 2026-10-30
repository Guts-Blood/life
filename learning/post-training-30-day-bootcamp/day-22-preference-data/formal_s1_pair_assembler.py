#!/usr/bin/env python3
"""Assemble verifier-backed promoted-S1 pairs after every machine gate.

The formal labeler records execution evidence per candidate, whereas the
training-pair contract records one execution object per chosen/rejected pair.
This module is the narrow, fail-closed adapter between those two contracts. It
does not perform or infer a human review.  A successful assembly therefore has
exactly one remaining blocker: ``human_blind_review_pending``.
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
import formal_s1_pair_labeler as labeler
from assemble_day22_smoke import build_blind_review


SCHEMA_VERSION = 1
MIN_FORMAL_PAIRS = 200
MAX_FORMAL_PAIRS = 500
REVIEW_UNIQUE_PAIRS = 50
REVIEW_SEED = "day22-formal-s1-blind-review-v1"
PENDING_BLOCKER = "human_blind_review_pending"
DATASET_ROLE = "formal_s1_dpo_pairs_pending_human_blind_review"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FormalS1AssemblyError(ValueError):
    """A formal input, machine gate, join, or output invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalS1AssemblyError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalS1AssemblyError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FormalS1AssemblyError(f"{label} must be non-empty text without NUL")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FormalS1AssemblyError(f"{label} must be a lowercase bare SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise FormalS1AssemblyError(f"{label} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise FormalS1AssemblyError(f"{label} must be an integer >= {minimum}") from error
    _require(result >= minimum and str(value).strip() == str(result), f"{label} is not canonical")
    return result


def _equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label} drifted")


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = contract.object_sha256({key: item for key, item in value.items() if key != field})
    _require(actual == expected, f"{label}.{field} does not bind its contents")
    return expected


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    result[field] = contract.object_sha256(result)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_artifact_path(path: Path) -> str:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "post-training-30-day-bootcamp":
            return resolved.relative_to(parent).as_posix()
    return resolved.name


def _index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = _text(row.get(key), f"{label}[{index}].{key}")
        _require(identity not in result, f"duplicate {label} {key}: {identity}")
        result[identity] = row
    return result


def _validate_summary(
    summary: Mapping[str, Any],
    selections: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
) -> None:
    _verify_self_hash(summary, "summary_sha256", "selection summary")
    _equal(summary.get("schema_name"), labeler.SUMMARY_SCHEMA, "selection summary schema")
    _equal(summary.get("schema_version"), labeler.SCHEMA_VERSION, "selection summary version")
    _equal(summary.get("status"), "pass", "selection summary status")
    counts = _mapping(summary.get("counts"), "selection summary.counts")
    _equal(counts.get("eligible_pairs"), len(replays), "selection summary eligible count")
    _equal(counts.get("candidate_evidence"), counts.get("rollout_candidates"), "selection evidence coverage")
    identities = _mapping(summary.get("content_identities"), "selection summary.content_identities")
    _equal(
        identities.get("ordered_selection_sha256"),
        contract.object_sha256([row.get("selection_sha256") for row in selections]),
        "ordered selection identity",
    )
    _equal(
        identities.get("ordered_replay_sha256"),
        contract.object_sha256([row.get("row_sha256") for row in replays]),
        "ordered replay identity",
    )
    _summary_generator_hashes(summary)
    if "common_policy_identity" in summary:
        policy = _mapping(summary.get("common_policy_identity"), "common policy identity")
        _equal(
            summary.get("common_policy_identity_sha256"),
            contract.object_sha256(policy),
            "common policy identity hash",
        )
    _sha256(summary.get("common_sandbox_digest"), "common sandbox digest")


def _summary_generator_hashes(summary: Mapping[str, Any]) -> set[str]:
    values = summary.get("generator_provenance_sha256s")
    if values is None:
        return {
            _sha256(
                summary.get("common_generator_provenance_sha256"),
                "common generator provenance",
            )
        }
    _require(isinstance(values, list) and bool(values), "generator provenance set is empty")
    hashes = {
        _sha256(value, f"generator_provenance_sha256s[{index}]")
        for index, value in enumerate(values)
    }
    _equal(len(hashes), len(values), "generator provenance set uniqueness")
    common = summary.get("common_generator_provenance_sha256")
    if common is not None:
        _require(len(hashes) == 1 and common in hashes, "legacy common generator field drifted")
    return hashes


def _validate_selection_rows(
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    _require(bool(selections), "selection input is empty")
    eligible: dict[str, Mapping[str, Any]] = {}
    families: set[str] = set()
    for index, raw in enumerate(selections):
        row = _mapping(raw, f"selection[{index}]")
        _verify_self_hash(row, "selection_sha256", f"selection[{index}]")
        _equal(row.get("schema_name"), labeler.SELECTION_SCHEMA, f"selection[{index}] schema")
        _equal(row.get("schema_version"), labeler.SCHEMA_VERSION, f"selection[{index}] version")
        family = _text(row.get("family_id"), f"selection[{index}].family_id")
        _require(family not in families, f"duplicate selected family record: {family}")
        families.add(family)
        status = row.get("selection_status")
        _require(status in {"eligible", labeler.QUARANTINE}, f"selection[{index}] status is invalid")
        if status == labeler.QUARANTINE:
            _require(row.get("pair_id") is None, f"selection[{index}] quarantine carries pair_id")
            continue
        pair_id = _text(row.get("pair_id"), f"selection[{index}].pair_id")
        _require(pair_id not in eligible, f"duplicate eligible pair_id: {pair_id}")
        _equal(
            row.get("dataset_role"),
            "formal_s1_pair_pending_processor_and_human_review",
            f"selection[{index}] dataset role",
        )
        decision = _mapping(row.get("labeler"), f"selection[{index}].labeler")
        _verify_self_hash(decision, "labeler_sha256", f"selection[{index}].labeler")
        _equal(decision.get("rubric_sha256"), labeler.RUBRIC_SHA256, "labeler rubric")
        _equal(
            decision.get("reason_code"),
            "stable_pass_vs_stable_wrong_answer",
            f"selection[{index}] preference reason",
        )
        eligible[pair_id] = row
    return eligible


def _selection_evidence_identities(
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Return every retained candidate evidence identity sealed by selections."""

    identities: dict[str, str] = {}

    def add(value: Mapping[str, Any], label: str) -> None:
        candidate_id = _text(value.get("candidate_id"), f"{label}.candidate_id")
        evidence_sha256 = _sha256(
            value.get("evidence_sha256"), f"{label}.evidence_sha256"
        )
        _require(
            candidate_id not in identities,
            f"duplicate selection evidence candidate_id: {candidate_id}",
        )
        identities[candidate_id] = evidence_sha256

    for index, selection in enumerate(selections):
        status = selection.get("selection_status")
        if status == "eligible":
            source = _mapping(
                selection.get("source_evidence"),
                f"selection[{index}].source_evidence",
            )
            for side in ("chosen", "rejected"):
                candidate = _mapping(
                    selection.get(side), f"selection[{index}].{side}"
                )
                add(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "evidence_sha256": source.get(f"{side}_evidence_sha256"),
                    },
                    f"selection[{index}].{side}",
                )
            excluded = selection.get("excluded_candidates")
            _require(
                isinstance(excluded, list),
                f"selection[{index}].excluded_candidates must be a list",
            )
            for excluded_index, raw in enumerate(excluded):
                item = _mapping(
                    raw,
                    f"selection[{index}].excluded_candidates[{excluded_index}]",
                )
                # Preselection exclusions never reached E2B and therefore do not
                # carry an evidence hash. Retained but unselected candidates do.
                if "evidence_sha256" in item:
                    add(
                        item,
                        f"selection[{index}].excluded_candidates[{excluded_index}]",
                    )
        elif status == labeler.QUARANTINE:
            candidate_evidence = selection.get("candidate_evidence")
            _require(
                isinstance(candidate_evidence, list),
                f"selection[{index}].candidate_evidence must be a list",
            )
            for evidence_index, raw in enumerate(candidate_evidence):
                add(
                    _mapping(
                        raw,
                        f"selection[{index}].candidate_evidence[{evidence_index}]",
                    ),
                    f"selection[{index}].candidate_evidence[{evidence_index}]",
                )
    return identities


def _validated_generator(value: Any, label: str) -> dict[str, Any]:
    try:
        generator = labeler._validate_generator(value, label)
    except labeler.FormalS1LabelerError as error:
        raise FormalS1AssemblyError(str(error)) from error
    _require(generator.get("promoted_s1") is True, f"{label} is not promoted S1")
    _require(generator.get("on_policy") is True, f"{label} is not on-policy")
    return generator


def _validate_replay_rows(
    replays: Sequence[Mapping[str, Any]],
    eligible: Mapping[str, Mapping[str, Any]],
    allowed_generator_sha256s: set[str],
    common_policy_identity_sha256: str | None,
) -> dict[str, Mapping[str, Any]]:
    _require(bool(replays), "replay input is empty")
    replay_by_pair = _index_unique(replays, "pair_id", "replay")
    _equal(set(replay_by_pair), set(eligible), "eligible selection/replay pair IDs")
    families: set[str] = set()
    for index, replay in enumerate(replays):
        pair_id = str(replay["pair_id"])
        selection = eligible[pair_id]
        _verify_self_hash(replay, "row_sha256", f"replay[{index}]")
        _equal(replay.get("schema_name"), "day22.mbpp_replay_pair", f"replay[{index}] schema")
        _equal(replay.get("schema_version"), 1, f"replay[{index}] version")
        _equal(selection.get("replay_row_sha256"), replay.get("row_sha256"), f"pair {pair_id} replay identity")
        for field in ("task_id", "family_id", "split"):
            _equal(str(selection.get(field)), str(replay.get(field)), f"pair {pair_id} {field}")
        family_id = _text(replay.get("family_id"), f"pair {pair_id}.family_id")
        task_id = _integer(replay.get("task_id"), f"pair {pair_id}.task_id")
        _equal(family_id, f"mbpp:task:{task_id}", f"pair {pair_id} native family")
        _require(family_id not in families, f"more than one pair for family {family_id}")
        families.add(family_id)
        _require(replay.get("split") in contract.SPLITS, f"pair {pair_id} split is invalid")
        _equal(replay.get("tests_sha256"), replay.get("tests", {}).get("sha256"), f"pair {pair_id} tests")
        pair_generators: list[Mapping[str, Any]] = []
        for side in ("chosen", "rejected"):
            replay_candidate = _mapping(replay.get(side), f"pair {pair_id}.{side}")
            selected_candidate = _mapping(selection.get(side), f"selection {pair_id}.{side}")
            for field in ("candidate_id", "origin", "text", "sha256", "ast_sha256"):
                _equal(replay_candidate.get(field), selected_candidate.get(field), f"pair {pair_id} {side}.{field}")
            _equal(replay_candidate.get("origin"), labeler.ORIGIN, f"pair {pair_id} {side} origin")
            _equal(replay_candidate.get("sha256"), contract.text_sha256(str(replay_candidate.get("text"))), f"pair {pair_id} {side} response")
            generator = _validated_generator(replay_candidate.get("generator"), f"pair {pair_id}.{side}.generator")
            _require(
                generator.get("provenance_sha256") in allowed_generator_sha256s,
                f"pair {pair_id} generator is not in the selection summary set",
            )
            pair_generators.append(generator)
            _equal(
                selected_candidate.get("generator"),
                replay_candidate.get("generator"),
                f"pair {pair_id} {side} generator object",
            )
        identities = {
            contract.object_sha256(labeler.generator_policy_identity(generator))
            for generator in pair_generators
        }
        _equal(len(identities), 1, f"pair {pair_id} promoted S1 policy identity")
        if common_policy_identity_sha256 is not None:
            _equal(
                next(iter(identities)),
                common_policy_identity_sha256,
                f"pair {pair_id} common policy identity",
            )
        source_evidence = _mapping(selection.get("source_evidence"), f"selection {pair_id}.source_evidence")
        _equal(source_evidence.get("test_sha256"), replay.get("tests_sha256"), f"pair {pair_id} source tests")
    return replay_by_pair


def _validate_all_candidate_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_identities: Mapping[str, str],
    common_sandbox_digest: str,
) -> dict[str, Mapping[str, Any]]:
    _equal(len(rows), len(expected_identities), "candidate evidence row count")
    evidence_by_id = _index_unique(rows, "candidate_id", "candidate evidence")
    _equal(
        set(evidence_by_id),
        set(expected_identities),
        "selection/candidate evidence candidate IDs",
    )
    run_ids: set[str] = set()
    for index, evidence in enumerate(rows):
        label = f"candidate evidence[{index}]"
        evidence_sha256 = _verify_self_hash(evidence, "evidence_sha256", label)
        candidate_id = str(evidence["candidate_id"])
        _equal(
            evidence_sha256,
            expected_identities[candidate_id],
            f"{label} selection evidence identity",
        )
        _equal(evidence.get("schema_name"), labeler.EVIDENCE_SCHEMA, f"{label} schema")
        _equal(evidence.get("schema_version"), labeler.SCHEMA_VERSION, f"{label} version")
        _equal(evidence.get("backend"), "e2b", f"{label} backend")
        _equal(evidence.get("sandbox_digest"), common_sandbox_digest, f"{label} sandbox")
        _equal(evidence.get("source_tests_sha256"), evidence.get("test_sha256"), f"{label} full tests")
        runs = evidence.get("runs")
        _require(isinstance(runs, list) and len(runs) == 2, f"{label} needs two fresh runs")
        for attempt, raw_run in enumerate(runs, 1):
            run = _mapping(raw_run, f"{label}.runs[{attempt - 1}]")
            _verify_self_hash(run, "run_sha256", f"{label}.runs[{attempt - 1}]")
            _equal(run.get("attempt"), attempt, f"{label} attempt")
            _equal(run.get("candidate_id"), evidence.get("candidate_id"), f"{label} run candidate")
            _equal(run.get("response_sha256"), evidence.get("response_sha256"), f"{label} run response")
            _equal(run.get("test_sha256"), evidence.get("test_sha256"), f"{label} run tests")
            _equal(run.get("sandbox_digest"), common_sandbox_digest, f"{label} run sandbox")
            try:
                labeler._validate_secure_sandbox(run.get("sandbox"), common_sandbox_digest, f"{label}.sandbox")
            except labeler.FormalS1LabelerError as error:
                raise FormalS1AssemblyError(str(error)) from error
            run_id = _text(run.get("run_id"), f"{label}.run_id")
            _require(run_id not in run_ids, f"duplicate E2B run_id: {run_id}")
            run_ids.add(run_id)
    return evidence_by_id


def _validate_candidate_evidence_input_identity(
    summary: Mapping[str, Any], input_identities: Mapping[str, Any] | None
) -> None:
    summary_inputs = _mapping(summary.get("inputs"), "selection summary.inputs")
    if "candidate_evidence" not in summary_inputs:
        return
    expected = _mapping(
        summary_inputs["candidate_evidence"],
        "selection summary.inputs.candidate_evidence",
    )
    expected_sha256 = _sha256(
        expected.get("file_sha256"),
        "selection summary.inputs.candidate_evidence.file_sha256",
    )
    actual_inputs = _mapping(input_identities, "assembly input identities")
    actual = _mapping(
        actual_inputs.get("candidate_evidence"),
        "assembly input identities.candidate_evidence",
    )
    actual_sha256 = _sha256(
        actual.get("file_sha256"),
        "assembly input identities.candidate_evidence.file_sha256",
    )
    _equal(actual_sha256, expected_sha256, "candidate evidence input file identity")


def _validate_selected_evidence(
    replay: Mapping[str, Any],
    selection: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    index: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    pair_id = str(replay["pair_id"])
    validated: dict[str, Mapping[str, Any]] = {}
    source = _mapping(selection.get("source_evidence"), f"selection {pair_id}.source_evidence")
    generator = replay["chosen"]["generator"]
    rollout = {
        "task_id": str(replay["task_id"]),
        "family_id": replay["family_id"],
        "generator": generator,
        "tests": replay["tests"],
    }
    expectations = {"chosen": labeler.STABLE_PASS, "rejected": labeler.STABLE_WRONG}
    for side in ("chosen", "rejected"):
        candidate = _mapping(replay.get(side), f"pair {pair_id}.{side}")
        candidate_id = str(candidate["candidate_id"])
        _require(candidate_id in evidence_by_id, f"pair {pair_id} {side} evidence is missing")
        evidence = evidence_by_id[candidate_id]
        expected_hash_field = f"{side}_evidence_sha256"
        _equal(source.get(expected_hash_field), evidence.get("evidence_sha256"), f"pair {pair_id} {side} evidence identity")
        try:
            result = labeler._validate_evidence(
                evidence,
                candidate=candidate,
                rollout=rollout,
                index=index * 2 + (0 if side == "chosen" else 1),
            )
        except labeler.FormalS1LabelerError as error:
            raise FormalS1AssemblyError(str(error)) from error
        _equal(result.get("status_class"), expectations[side], f"pair {pair_id} {side} execution class")
        validated[side] = evidence
    _equal(
        validated["chosen"].get("sandbox_digest"),
        validated["rejected"].get("sandbox_digest"),
        f"pair {pair_id} sandbox identity",
    )
    return validated["chosen"], validated["rejected"]


def _validate_processor_contract(value: Mapping[str, Any]) -> str:
    digest = _verify_self_hash(value, "contract_sha256", "processor contract")
    _equal(value.get("schema_name"), "day22.qwen35_processor_contract", "processor contract schema")
    _equal(value.get("schema_version"), 1, "processor contract version")
    _equal(value.get("status"), "frozen", "processor contract status")
    _require(value.get("max_length", 0) > 0, "processor contract max_length must be positive")
    for field in ("processor_sha256", "tokenizer_sha256", "template_sha256"):
        _sha256(value.get(field), f"processor contract.{field}")
    return digest


def _validate_processor_audit(
    audit: Mapping[str, Any], replay: Mapping[str, Any], processor_contract_sha256: str
) -> None:
    pair_id = str(replay["pair_id"])
    _verify_self_hash(audit, "audit_sha256", f"processor audit {pair_id}")
    _equal(audit.get("schema_name"), "day22.qwen35_pair_processor_audit", f"processor audit {pair_id} schema")
    _equal(audit.get("schema_version"), 1, f"processor audit {pair_id} version")
    _equal(audit.get("status"), "pass", f"processor audit {pair_id} status")
    for field, expected in (
        ("pair_id", pair_id),
        ("task_id", str(replay["task_id"])),
        ("family_id", replay["family_id"]),
        ("chosen_candidate_id", replay["chosen"]["candidate_id"]),
        ("rejected_candidate_id", replay["rejected"]["candidate_id"]),
        ("chosen_response_sha256", replay["chosen"]["sha256"]),
        ("rejected_response_sha256", replay["rejected"]["sha256"]),
        ("replay_row_sha256", replay["row_sha256"]),
        ("processor_contract_sha256", processor_contract_sha256),
    ):
        _equal(audit.get(field), expected, f"processor audit {pair_id} {field}")


def _adapt_execution(
    chosen_evidence: Mapping[str, Any], rejected_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    first_run = _mapping(chosen_evidence["runs"][0], "chosen evidence first run")
    sandbox = _mapping(first_run.get("sandbox"), "chosen evidence sandbox")
    timeout = sandbox.get("wall_timeout_seconds")
    _require(
        isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0,
        "E2B wall timeout is missing",
    )
    execution = {
        "test_sha256": chosen_evidence["test_sha256"],
        "sandbox_digest": chosen_evidence["sandbox_digest"],
        "timeout_seconds": timeout,
        "verifier": {
            "name": "score_day22_s1_candidates",
            "version": "day22.s1_candidate_e2b_evidence.v1",
        },
        "source_backend": "e2b",
        "source_candidate_evidence_sha256": {
            "chosen": chosen_evidence["evidence_sha256"],
            "rejected": rejected_evidence["evidence_sha256"],
        },
        "chosen": {
            "candidate_id": chosen_evidence["candidate_id"],
            "origin": labeler.ORIGIN,
            "response_sha256": chosen_evidence["response_sha256"],
            "runs": copy.deepcopy(chosen_evidence["runs"]),
        },
        "rejected": {
            "candidate_id": rejected_evidence["candidate_id"],
            "origin": labeler.ORIGIN,
            "response_sha256": rejected_evidence["response_sha256"],
            "runs": copy.deepcopy(rejected_evidence["runs"]),
        },
    }
    return _seal(execution, "evidence_sha256")


def _make_pair(
    replay: Mapping[str, Any],
    selection: Mapping[str, Any],
    chosen_evidence: Mapping[str, Any],
    rejected_evidence: Mapping[str, Any],
    processor_audit: Mapping[str, Any],
) -> dict[str, Any]:
    chosen_generator = replay["chosen"]["generator"]
    rejected_generator = replay["rejected"]["generator"]
    generator_hashes = sorted(
        {
            str(chosen_generator["provenance_sha256"]),
            str(rejected_generator["provenance_sha256"]),
        }
    )
    pair = {
        "schema_name": contract.PAIR_SCHEMA_NAME,
        "schema_version": contract.PAIR_SCHEMA_VERSION,
        "pair_id": replay["pair_id"],
        "pair_status": contract.FINAL_PAIR_STATUS,
        "dataset_role": DATASET_ROLE,
        "split": replay["split"],
        "task_id": _integer(replay["task_id"], f"pair {replay['pair_id']}.task_id"),
        "language": "python",
        "family_keys": copy.deepcopy(replay["family_keys"]),
        "source": copy.deepcopy(replay["source"]),
        "problem": copy.deepcopy(replay["problem"]),
        "prompt": copy.deepcopy(replay["prompt"]),
        "code_prefix": replay["code_prefix"],
        "entry_point": replay["entry_point"],
        "chosen": {
            key: copy.deepcopy(replay["chosen"][key])
            for key in ("candidate_id", "origin", "text", "sha256", "ast_sha256", "generator")
        },
        "rejected": {
            key: copy.deepcopy(replay["rejected"][key])
            for key in ("candidate_id", "origin", "text", "sha256", "ast_sha256", "generator")
        },
        "creation": {
            "method": "promoted_s1_on_policy_rollout_pair",
            "version": labeler.RUBRIC_VERSION,
            "synthetic": False,
            "promoted_s1_on_policy": True,
            "generator_provenance_sha256s": generator_hashes,
            "source_selection_sha256": selection["selection_sha256"],
            "source_replay_row_sha256": replay["row_sha256"],
        },
        "quality_flags": [PENDING_BLOCKER],
        "tests": copy.deepcopy(replay["tests"]),
        "execution": _adapt_execution(chosen_evidence, rejected_evidence),
        "processor_audit": copy.deepcopy(processor_audit),
    }
    if len(generator_hashes) == 1:
        pair["creation"]["generator_provenance_sha256"] = generator_hashes[0]
    sealed = contract.seal_pair(pair)
    try:
        contract.validate_pair(sealed, mode="final")
    except contract.Day22ContractError as error:
        raise FormalS1AssemblyError(
            f"assembled pair {replay['pair_id']} violates final contract: {error}"
        ) from error
    return sealed


def _selected_formal_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Deterministically enforce the formal upper bound without weakening it."""
    if len(pairs) <= MAX_FORMAL_PAIRS:
        return list(pairs)
    return sorted(
        pairs,
        key=lambda pair: (
            contract.text_sha256(f"day22-formal-cap-v1\0{pair['pair_id']}"),
            str(pair["pair_id"]),
        ),
    )[:MAX_FORMAL_PAIRS]


def assemble_formal_s1(
    selection_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
    candidate_evidence_rows: Sequence[Mapping[str, Any]],
    processor_rows: Sequence[Mapping[str, Any]],
    processor_contract: Mapping[str, Any],
    *,
    input_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a formal bundle whose only remaining gate is human blind review."""

    eligible = _validate_selection_rows(selection_rows)
    _validate_summary(selection_summary, selection_rows, replay_rows)
    _validate_candidate_evidence_input_identity(selection_summary, input_identities)
    evidence_identities = _selection_evidence_identities(selection_rows)
    _equal(
        len(evidence_identities),
        int(selection_summary["counts"]["candidate_evidence"]),
        "selection evidence identity count",
    )
    allowed_generators = _summary_generator_hashes(selection_summary)
    policy_identity_sha256 = selection_summary.get("common_policy_identity_sha256")
    if policy_identity_sha256 is not None:
        policy_identity_sha256 = _sha256(
            policy_identity_sha256, "common policy identity sha256"
        )
    replays = _validate_replay_rows(
        replay_rows,
        eligible,
        allowed_generators,
        policy_identity_sha256,
    )
    evidence = _validate_all_candidate_evidence(
        candidate_evidence_rows,
        expected_identities=evidence_identities,
        common_sandbox_digest=str(selection_summary["common_sandbox_digest"]),
    )
    processor_contract_sha256 = _validate_processor_contract(processor_contract)
    processor_by_pair = _index_unique(processor_rows, "pair_id", "processor audit")
    _equal(set(processor_by_pair), set(replays), "processor audit/replay pair IDs")

    pairs: list[dict[str, Any]] = []
    for index, pair_id in enumerate(sorted(replays)):
        replay = replays[pair_id]
        selection = eligible[pair_id]
        audit = processor_by_pair[pair_id]
        _validate_processor_audit(audit, replay, processor_contract_sha256)
        chosen_evidence, rejected_evidence = _validate_selected_evidence(
            replay, selection, evidence, index=index
        )
        pairs.append(
            _make_pair(
                replay,
                selection,
                chosen_evidence,
                rejected_evidence,
                audit,
            )
        )

    pairs = [dict(pair) for pair in _selected_formal_pairs(pairs)]
    if len(pairs) < MIN_FORMAL_PAIRS:
        raise FormalS1AssemblyError(
            "BLOCKED[minimum_accepted_pairs]: "
            f"observed {len(pairs)}, require {MIN_FORMAL_PAIRS}; formal outputs were not assembled"
        )
    _require(len(pairs) <= MAX_FORMAL_PAIRS, "formal pair upper bound was not enforced")
    pairs.sort(key=lambda pair: str(pair["pair_id"]))
    try:
        contract_summary = contract.validate_manifest(pairs, mode="final")
    except contract.Day22ContractError as error:
        raise FormalS1AssemblyError(f"formal pair manifest violates contract: {error}") from error

    synthetic_count = sum(pair["creation"].get("synthetic") is True for pair in pairs)
    on_policy_count = sum(
        pair["creation"].get("promoted_s1_on_policy") is True for pair in pairs
    )
    _require(synthetic_count == 0, "formal S1 assembly contains synthetic pairs")
    _require(on_policy_count == len(pairs), "formal S1 assembly contains off-policy pairs")

    split_ids: dict[str, Any] = {
        "schema_name": "day22.formal_s1_family_split_ids",
        "schema_version": SCHEMA_VERSION,
        "dataset_role": DATASET_ROLE,
        **{
            split: sorted(str(pair["pair_id"]) for pair in pairs if pair["split"] == split)
            for split in contract.SPLITS
        },
        "family_ids": {
            split: sorted(
                str(pair["family_keys"]["problem"])
                for pair in pairs
                if pair["split"] == split
            )
            for split in contract.SPLITS
        },
    }
    split_ids = _seal(split_ids, "split_ids_sha256")
    worksheet, review_key = build_blind_review(
        pairs, limit=REVIEW_UNIQUE_PAIRS, seed=REVIEW_SEED
    )
    _equal(review_key.get("unique_pairs"), REVIEW_UNIQUE_PAIRS, "blind review unique pairs")
    _equal(len(worksheet), REVIEW_UNIQUE_PAIRS * 2, "blind review presentations")

    machine_gates = {
        "formal_pair_count_200_to_500": {
            "passed": MIN_FORMAL_PAIRS <= len(pairs) <= MAX_FORMAL_PAIRS,
            "observed": len(pairs),
            "minimum": MIN_FORMAL_PAIRS,
            "maximum": MAX_FORMAL_PAIRS,
        },
        "one_pair_per_family": {
            "passed": len({pair["family_keys"]["problem"] for pair in pairs}) == len(pairs),
            "observed_unique_families": len({pair["family_keys"]["problem"] for pair in pairs}),
        },
        "cross_split_family_overlap": {
            "passed": True,
            "observed_overlap": 0,
            "family_dimensions": list(contract.FAMILY_KEY_NAMES),
        },
        "promoted_s1_on_policy": {
            "passed": on_policy_count == len(pairs),
            "observed": on_policy_count,
        },
        "synthetic_pairs": {
            "passed": synthetic_count == 0,
            "observed": synthetic_count,
            "fraction": 0.0,
        },
        "secure_e2b_two_fresh_runs_per_side": {
            "passed": True,
            "observed_pairs": len(pairs),
        },
        "qwen35_processor_audit": {
            "passed": True,
            "observed_pairs": len(pairs),
            "processor_contract_sha256": processor_contract_sha256,
        },
        "blind_review_worksheet_generation": {
            "passed": True,
            "unique_pairs": review_key["unique_pairs"],
            "presentations": review_key["presentations"],
            "human_verdicts_filled": 0,
        },
    }
    _require(all(gate["passed"] is True for gate in machine_gates.values()), "a machine gate did not pass")
    audit: dict[str, Any] = {
        "schema_name": "day22.formal_s1_machine_gate_audit",
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "dataset_role": DATASET_ROLE,
        "formal_dpo_ready": False,
        "machine_ready": True,
        "pending_blockers": [PENDING_BLOCKER],
        "machine_gates": machine_gates,
        "counts": {
            "input_selections": len(selection_rows),
            "eligible_selections": len(eligible),
            "formal_pairs": len(pairs),
            "synthetic_pairs": synthetic_count,
            "promoted_s1_on_policy_pairs": on_policy_count,
            "review_unique_pairs": review_key["unique_pairs"],
            "review_presentations": review_key["presentations"],
        },
        "contract_summary": contract_summary,
    }
    audit = _seal(audit, "audit_sha256")
    manifest: dict[str, Any] = {
        "schema_name": "day22.formal_s1_assembly_manifest",
        "schema_version": SCHEMA_VERSION,
        "status": "machine_gates_passed_human_review_pending",
        "dataset_role": DATASET_ROLE,
        "readiness": "BLOCKED",
        "formal_dpo_ready": False,
        "machine_ready": True,
        "formal_dpo_blockers": [PENDING_BLOCKER],
        "human_review": {
            "status": "pending",
            "required_unique_pairs": REVIEW_UNIQUE_PAIRS,
            "completed_unique_pairs": 0,
            "conclusions": None,
        },
        "inputs": copy.deepcopy(dict(input_identities or {})),
        "counts": copy.deepcopy(audit["counts"]),
        "contract_summary": contract_summary,
        "machine_gate_audit_sha256": audit["audit_sha256"],
        "content_identities": {
            "ordered_pair_hashes_sha256": contract_summary["ordered_pair_hashes_sha256"],
            "split_ids_sha256": split_ids["split_ids_sha256"],
            "worksheet_sha256": contract.object_sha256(worksheet),
            "concealed_review_key_sha256": review_key["key_sha256"],
            "selection_summary_sha256": selection_summary["summary_sha256"],
            "processor_contract_sha256": processor_contract_sha256,
        },
    }
    manifest = _seal(manifest, "manifest_sha256")
    return {
        "pairs": pairs,
        "split_ids": split_ids,
        "review_worksheet": worksheet,
        "review_key": review_key,
        "audit": audit,
        "manifest": manifest,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalS1AssemblyError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalS1AssemblyError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FormalS1AssemblyError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise FormalS1AssemblyError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise FormalS1AssemblyError(f"invalid JSONL: {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise FormalS1AssemblyError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise FormalS1AssemblyError(f"JSONL is empty: {path}")
    return rows


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(contract.canonical_json(row) + b"\n" for row in rows)


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FormalS1AssemblyError(f"refusing to overwrite: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, resolved)
        else:
            try:
                os.link(temporary, resolved)
            except FileExistsError as error:
                raise FormalS1AssemblyError(f"output appeared while writing: {resolved}") from error
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--processor-audit", type=Path, required=True)
    parser.add_argument("--processor-contract", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--split-ids-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--assembly-audit-output", type=Path, required=True)
    parser.add_argument("--review-worksheet-output", type=Path, required=True)
    parser.add_argument("--review-key-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = {
            "selections": args.selections.resolve(),
            "replay_input": args.replay_input.resolve(),
            "selection_summary": args.selection_summary.resolve(),
            "candidate_evidence": args.candidate_evidence.resolve(),
            "processor_audit": args.processor_audit.resolve(),
            "processor_contract": args.processor_contract.resolve(),
        }
        identities = {
            name: {"path": portable_artifact_path(path), "file_sha256": file_sha256(path)}
            for name, path in inputs.items()
        }
        result = assemble_formal_s1(
            _load_jsonl(inputs["selections"]),
            _load_jsonl(inputs["replay_input"]),
            _load_json(inputs["selection_summary"]),
            _load_jsonl(inputs["candidate_evidence"]),
            _load_jsonl(inputs["processor_audit"]),
            _load_json(inputs["processor_contract"]),
            input_identities=identities,
        )
        outputs = {
            args.pairs_output: _jsonl_bytes(result["pairs"]),
            args.split_ids_output: _json_bytes(result["split_ids"]),
            args.assembly_audit_output: _json_bytes(result["audit"]),
            args.review_worksheet_output: _jsonl_bytes(result["review_worksheet"]),
            args.review_key_output: _json_bytes(result["review_key"]),
        }
        resolved_outputs = [path.resolve() for path in [*outputs, args.manifest_output]]
        _require(len(resolved_outputs) == len(set(resolved_outputs)), "output paths must be distinct")
        if not args.overwrite:
            for path in resolved_outputs:
                _require(not path.exists(), f"refusing to overwrite: {path}")
        manifest = copy.deepcopy(result["manifest"])
        manifest.pop("manifest_sha256", None)
        manifest["output_files"] = {
            path.name: {
                "path": portable_artifact_path(path),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in outputs.items()
        }
        manifest = _seal(manifest, "manifest_sha256")
        outputs[args.manifest_output] = _json_bytes(manifest)
        for path, payload in outputs.items():
            _write_atomic(path, payload, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "formal_pairs": len(result["pairs"]),
                    "machine_ready": True,
                    "formal_dpo_blockers": [PENDING_BLOCKER],
                    "manifest": str(args.manifest_output.resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (FormalS1AssemblyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
