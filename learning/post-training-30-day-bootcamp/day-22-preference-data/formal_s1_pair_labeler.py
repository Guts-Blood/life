#!/usr/bin/env python3
"""Select verifier-backed Day 22 pairs from promoted-S1 rollouts.

This module does not execute candidate code and does not make a subjective
quality judgement.  It joins one family-level rollout row with one E2B
evidence row per candidate, then emits at most one stable pass-vs-wrong-answer
pair per family.  Its replay output is directly consumable by
``audit_day22_qwen35_processor.py`` and ``score_day22_mbpp_sandbox.py``.

The common ``generator`` object deliberately excludes worker/card/run IDs.
Those belong on individual candidates.  This lets two GPU shards prove that
they used the exact same promoted checkpoint, tokenizer, and generation
configuration while retaining per-sample provenance.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import day22_contract as contract


ROLLOUT_SCHEMA = "day22.s1_rollout_family"
CANDIDATE_ROLLOUT_SCHEMA = "day22.s1_rollout_candidate"
EVIDENCE_SCHEMA = "day22.s1_candidate_e2b_evidence"
SELECTION_SCHEMA = "day22.s1_pair_selection"
SUMMARY_SCHEMA = "day22.s1_pair_selection_summary"
CANDIDATE_REPLAY_SCHEMA = "day22.s1_candidate_replay_request"
SCHEMA_VERSION = 1
RUBRIC_NAME = "day22.promoted_s1_verifier_pair_selection"
RUBRIC_VERSION = "day22.promoted_s1_verifier_pair_selection.v1"
ORIGIN = "promoted_s1_rollout"
STABLE_PASS = "stable_pass"
STABLE_WRONG = "stable_wrong_answer"
QUARANTINE = "quarantine"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

RUBRIC = {
    "name": RUBRIC_NAME,
    "version": RUBRIC_VERSION,
    "actor_type": "codex_subagent",
    "agent_role": "formal_s1_preference_labeler",
    "decision_basis": "execution_evidence_only",
    "chosen_gate": "two fresh E2B runs both pass",
    "rejected_gate": "two fresh E2B runs both verifier-classified wrong_answer",
    "primary_rank": "minimum symmetric relative response-token length delta",
    "secondary_rank": "maximum attested test pass ratio when available",
    "final_tiebreak": "lexicographic candidate IDs",
    "one_pair_per_family": True,
    "quarantine": [
        "identical_or_ast_equivalent_responses",
        "both_pass",
        "both_fail",
        "unstable_execution",
        "syntax_error",
        "runtime_error",
        "timeout",
        "infra_error",
    ],
}
RUBRIC_SHA256 = contract.object_sha256(RUBRIC)

GENERATOR_TEXT_FIELDS = (
    "checkpoint",
    "downstream_key",
)
POLICY_IDENTITY_FIELDS = (
    "checkpoint",
    "downstream_key",
    "candidate",
    "model_role",
    "promotion_manifest_sha256",
    "merged_snapshot_sha256",
    "merged_export_manifest_sha256",
    "source_checkpoint_integrity_sha256",
    "source_adapter_sha256",
    "lineage_manifest_file_sha256",
    "checkpoint_manifest_sha256",
    "tokenizer_sha256",
)
EXECUTION_STATUSES = {
    "pass",
    "wrong_answer",
    "syntax_error",
    "runtime_error",
    "timeout",
    "infra_error",
}


class FormalS1LabelerError(ValueError):
    """A provenance, evidence, selection, or output invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalS1LabelerError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalS1LabelerError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FormalS1LabelerError(f"{label} must be non-empty text without NUL")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FormalS1LabelerError(f"{label} must be a lowercase bare SHA-256")
    return value


def _git_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise FormalS1LabelerError(f"{label} must be a lowercase 40-hex Git commit")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise FormalS1LabelerError(f"{label} must be an integer >= {minimum}")
    return value


def _task_id(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise FormalS1LabelerError(f"{label} must be a canonical non-negative integer")
    try:
        normalized = str(int(value))
    except (TypeError, ValueError) as error:
        raise FormalS1LabelerError(
            f"{label} must be a canonical non-negative integer"
        ) from error
    _require(normalized == str(value).strip() and int(normalized) >= 0, f"{label} is not canonical")
    return normalized


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    payload = {key: item for key, item in value.items() if key != field}
    _require(expected == contract.object_sha256(payload), f"{label}.{field} does not bind its contents")
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


def seal_generator(generator: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(generator, "provenance_sha256")


def seal_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(candidate, "candidate_sha256")


def seal_rollout(row: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(row, "row_sha256")


def seal_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(evidence, "evidence_sha256")


def seal_run(run: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(run, "run_sha256")


def seal_pass_ratio_basis(basis: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(basis, "basis_sha256")


def _validate_generator(value: Any, label: str) -> dict[str, Any]:
    generator = copy.deepcopy(dict(_mapping(value, label)))
    _verify_self_hash(generator, "provenance_sha256", label)
    _require(generator.get("promoted_s1") is True, f"{label}.promoted_s1 must be true")
    _require(generator.get("on_policy") is True, f"{label}.on_policy must be true")
    for field in GENERATOR_TEXT_FIELDS:
        _text(generator.get(field), f"{label}.{field}")
    for field in (
        "checkpoint_manifest_sha256",
        "promotion_manifest_sha256",
        "tokenizer_sha256",
        "generation_config_sha256",
    ):
        if field in generator:
            _sha256(generator.get(field), f"{label}.{field}")
    if "generation_config" in generator:
        config = _mapping(generator.get("generation_config"), f"{label}.generation_config")
        _require(bool(config), f"{label}.generation_config must not be empty")
        _require(
            generator.get("generation_config_sha256") == contract.object_sha256(config),
            f"{label}.generation_config_sha256 does not bind generation_config",
        )
    return generator


def generator_policy_identity(generator: Mapping[str, Any]) -> dict[str, Any]:
    """Return promoted-policy lineage, deliberately excluding sampling config.

    A supplemental rollout remains on-policy when it uses the same promoted
    checkpoint/export but changes temperature, top-p, seed domain, or rollout
    contract.  Those sampling details stay sealed in each candidate generator;
    they are not part of the model-policy identity.
    """

    identity = {
        field: copy.deepcopy(generator[field])
        for field in POLICY_IDENTITY_FIELDS
        if field in generator
    }
    _require(
        all(field in identity for field in GENERATOR_TEXT_FIELDS),
        "generator policy identity is missing checkpoint/downstream_key",
    )
    identity["promoted_s1"] = generator.get("promoted_s1")
    identity["on_policy"] = generator.get("on_policy")
    return identity


def _validate_generator_set(values: Sequence[Any], label: str) -> list[dict[str, Any]]:
    generators = [
        _validate_generator(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    _require(bool(generators), f"{label} must not be empty")
    hashes = [str(item["provenance_sha256"]) for item in generators]
    _require(len(hashes) == len(set(hashes)), f"{label} has duplicate provenance hashes")
    identities = {contract.object_sha256(generator_policy_identity(item)) for item in generators}
    _require(
        len(identities) == 1,
        f"{label} does not share one promoted S1 policy identity",
    )
    return sorted(generators, key=lambda item: str(item["provenance_sha256"]))


def _validate_hashed_text(value: Any, label: str) -> dict[str, str]:
    record = _mapping(value, label)
    text = _text(record.get("text"), f"{label}.text")
    digest = _sha256(record.get("sha256"), f"{label}.sha256")
    _require(digest == contract.text_sha256(text), f"{label}.sha256 does not match text")
    return {"text": text, "sha256": digest}


def _response_ast_sha256(response: str, code_prefix: str, label: str) -> str:
    try:
        wrapper = ast.parse(f"def __day22_response__():\n{response}\n")
        combined = ast.parse(f"{code_prefix}\n{response}\n")
        compile(combined, "<day22-formal-s1-pair>", "exec")
    except (IndentationError, SyntaxError) as error:
        raise FormalS1LabelerError(f"{label} is not a valid code continuation: {error.msg}") from error
    function = wrapper.body[0]
    _require(isinstance(function, ast.FunctionDef) and bool(function.body), f"{label} has no executable body")
    body = ast.Module(body=function.body, type_ignores=[])
    identity = ast.dump(body, annotate_fields=True, include_attributes=False)
    return contract.text_sha256(identity)


def _validate_tests(value: Any, label: str) -> dict[str, Any]:
    tests = copy.deepcopy(dict(_mapping(value, label)))
    normalized: dict[str, Any] = {}
    for key in ("test_list", "challenge_test_list"):
        items = tests.get(key)
        _require(isinstance(items, list), f"{label}.{key} must be a list")
        _require(
            all(isinstance(item, str) and item.strip() and "\x00" not in item for item in items),
            f"{label}.{key} contains invalid tests",
        )
        normalized[key] = [item.strip() for item in items]
    _require(bool(normalized["test_list"]), f"{label}.test_list must not be empty")
    setup = tests.get("test_setup_code")
    _require(isinstance(setup, str) and "\x00" not in setup, f"{label}.test_setup_code must be text")
    normalized["test_setup_code"] = setup.strip()
    digest = _sha256(tests.get("sha256"), f"{label}.sha256")
    _require(digest == contract.object_sha256(normalized), f"{label}.sha256 does not bind canonical tests")
    _require(tests.get("count") == len(normalized["test_list"]), f"{label}.count drifted")
    _require(
        tests.get("challenge_count") == len(normalized["challenge_test_list"]),
        f"{label}.challenge_count drifted",
    )
    return tests


def _validate_candidate(
    value: Any,
    *,
    family_id: str,
    code_prefix: str,
    generator_sha256s: set[str],
    label: str,
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(_mapping(value, label)))
    _verify_self_hash(candidate, "candidate_sha256", label)
    candidate_id = _text(candidate.get("candidate_id"), f"{label}.candidate_id")
    _require(candidate.get("origin") == ORIGIN, f"{label}.origin must be {ORIGIN}")
    _require(candidate.get("family_id") == family_id, f"{label}.family_id drifted")
    _require(
        candidate.get("generator_provenance_sha256") in generator_sha256s,
        f"{label}.generator_provenance_sha256 is not in the family generator set",
    )
    text = _text(candidate.get("text"), f"{label}.text")
    _require("\r" not in text and "```" not in text, f"{label}.text is not a raw LF code continuation")
    response_sha256 = _sha256(candidate.get("sha256"), f"{label}.sha256")
    _require(response_sha256 == contract.text_sha256(text), f"{label}.sha256 does not match text")
    _integer(candidate.get("response_token_count"), f"{label}.response_token_count", minimum=1)
    _integer(candidate.get("sample_index"), f"{label}.sample_index")
    _integer(candidate.get("sample_seed"), f"{label}.sample_seed")
    candidate["ast_sha256"] = _response_ast_sha256(text, code_prefix, label)
    candidate["candidate_id"] = candidate_id
    return candidate


def _candidate_generator_core(value: Any, label: str) -> dict[str, Any]:
    """Return the cross-shard generator identity from one rollout row."""

    generator = copy.deepcopy(dict(_mapping(value, label)))
    expected_keys = {
        "rollout_contract_sha256",
        "candidate",
        "model_role",
        "merged_snapshot_sha256",
        "source_checkpoint_integrity_sha256",
        "source_adapter_sha256",
        "lineage_manifest_file_sha256",
        "runtime_sha256",
        "ms_swift_commit",
        "downstream_key",
        "promotion_manifest_sha256",
        "merged_export_manifest_sha256",
        "generation_config_sha256",
        "generation",
        "shard_id",
        "shard_count",
        "elapsed_seconds",
    }
    _require(set(generator) == expected_keys, f"{label} keys drifted")
    for field in (
        "rollout_contract_sha256",
        "merged_snapshot_sha256",
        "source_checkpoint_integrity_sha256",
        "source_adapter_sha256",
        "lineage_manifest_file_sha256",
        "runtime_sha256",
        "promotion_manifest_sha256",
        "merged_export_manifest_sha256",
        "generation_config_sha256",
    ):
        _sha256(generator.get(field), f"{label}.{field}")
    _git_commit(generator.get("ms_swift_commit"), f"{label}.ms_swift_commit")
    checkpoint = _text(generator.get("candidate"), f"{label}.candidate")
    downstream_key = _text(generator.get("downstream_key"), f"{label}.downstream_key")
    _require(generator.get("model_role") == "merged_s1", f"{label}.model_role must be merged_s1")
    _integer(generator.get("shard_id"), f"{label}.shard_id")
    _integer(generator.get("shard_count"), f"{label}.shard_count", minimum=1)
    elapsed = generator.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0,
        f"{label}.elapsed_seconds must be non-negative",
    )
    generation = copy.deepcopy(dict(_mapping(generator.get("generation"), f"{label}.generation")))
    _require(bool(generation), f"{label}.generation must not be empty")
    _require(
        generator["generation_config_sha256"] == contract.object_sha256(generation),
        f"{label}.generation_config_sha256 does not bind generation",
    )
    core = {key: value for key, value in generator.items() if key not in {"shard_id", "elapsed_seconds"}}
    normalized: dict[str, Any] = {
        "promoted_s1": True,
        "on_policy": True,
        "checkpoint": checkpoint,
        "downstream_key": downstream_key,
        **core,
    }
    return seal_generator(normalized)


def _validate_format_contract(value: Any, label: str) -> bool:
    format_contract = _mapping(value, label)
    _require(
        set(format_contract) == {"valid", "execution_eligible", "validator", "evidence", "error"},
        f"{label} keys drifted",
    )
    valid = format_contract.get("valid")
    eligible = format_contract.get("execution_eligible")
    _require(isinstance(valid, bool) and isinstance(eligible, bool), f"{label} flags must be booleans")
    _require(valid == eligible, f"{label} valid/execution_eligible disagree")
    _require(
        format_contract.get("validator") == "validate_raw_code_continuation",
        f"{label}.validator drifted",
    )
    if eligible:
        _require(isinstance(format_contract.get("evidence"), Mapping), f"{label}.evidence is missing")
        _require(format_contract.get("error") is None, f"{label}.error must be null when valid")
    else:
        _require(format_contract.get("evidence") is None, f"{label}.evidence must be null when invalid")
        error = _mapping(format_contract.get("error"), f"{label}.error")
        _text(error.get("type"), f"{label}.error.type")
        _text(error.get("message"), f"{label}.error.message")
    return eligible


def _normalized_rollout_candidate(
    row: Mapping[str, Any],
    *,
    family_id: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    response = _mapping(row.get("response"), f"{row.get('candidate_id')}.response")
    candidate = {
        "candidate_id": row["candidate_id"],
        "family_id": family_id,
        "origin": ORIGIN,
        "text": response["text"],
        "sha256": response["sha256"],
        "response_token_count": response["generated_token_count"],
        "sample_index": row["sample_index"],
        "sample_seed": row["generation_seed"],
        "generation_run_id": f"rollout-shard-{row['generator']['shard_id']}",
        "generator_provenance_sha256": generator["provenance_sha256"],
        "source_candidate_sha256": row["candidate_sha256"],
    }
    return seal_candidate(candidate)


def _rollout_generators(rollout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = rollout.get("generators")
    if isinstance(values, list):
        return list(values)
    return [rollout["generator"]]


def _candidate_generator(
    rollout: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Any]:
    embedded = candidate.get("generator")
    if isinstance(embedded, Mapping):
        return embedded
    expected = str(candidate["generator_provenance_sha256"])
    matches = [
        generator
        for generator in _rollout_generators(rollout)
        if str(generator.get("provenance_sha256")) == expected
    ]
    _require(len(matches) == 1, f"candidate {candidate.get('candidate_id')} generator is missing")
    return matches[0]


def _normalize_candidate_rollouts(
    rows: Sequence[Mapping[str, Any]],
    *,
    rollout_contract: Mapping[str, Any],
    seed_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate candidate-per-row rollout output and group retained uniques."""

    try:
        import rollout_day22_s1 as rollout
    except ImportError as error:
        raise FormalS1LabelerError("cannot import rollout_day22_s1 contract") from error
    try:
        rollout.verify_self_hash(rollout_contract, "contract_sha256")
        rollout.verify_self_hash(seed_manifest, "manifest_sha256")
        _require(
            rollout_contract.get("schema_name") == rollout.CONTRACT_SCHEMA,
            "rollout contract schema drifted",
        )
        _require(
            rollout_contract.get("seed_pool", {}).get("manifest_sha256")
            == seed_manifest.get("manifest_sha256"),
            "rollout contract/seed manifest identity drifted",
        )
        specs = rollout.expected_specs(seed_manifest.get("records", []))
    except (rollout.Day22RolloutError, TypeError) as error:
        raise FormalS1LabelerError(f"invalid rollout contract inputs: {error}") from error
    spec_by_id = {str(spec["candidate_id"]): spec for spec in specs}
    raw_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = copy.deepcopy(dict(_mapping(raw, f"rollout candidate {index}")))
        candidate_id = _text(row.get("candidate_id"), f"rollout candidate {index}.candidate_id")
        _require(candidate_id in spec_by_id, f"unexpected rollout candidate_id: {candidate_id}")
        _require(candidate_id not in raw_by_id, f"duplicate rollout candidate_id: {candidate_id}")
        try:
            rollout.validate_candidate_row(
                row,
                contract=rollout_contract,
                spec=spec_by_id[candidate_id],
            )
        except rollout.Day22RolloutError as error:
            raise FormalS1LabelerError(f"rollout candidate {candidate_id} violates source contract: {error}") from error
        raw_by_id[candidate_id] = row
    _require(
        set(raw_by_id) == set(spec_by_id),
        "rollout input must exactly cover all candidates frozen by the rollout contract",
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_by_id.values():
        family_id = _text(row.get("task_family_id"), f"{row['candidate_id']}.task_family_id")
        task_id = _task_id(row.get("task_id"), f"{row['candidate_id']}.task_id")
        _require(family_id == f"mbpp:task:{task_id}", f"{row['candidate_id']} family/task drifted")
        grouped.setdefault(family_id, []).append(row)

    normalized_families: list[dict[str, Any]] = []
    duplicate_count = 0
    format_invalid_count = 0
    core_hashes: set[str] = set()
    context_fields = (
        "task_id",
        "task_family_id",
        "split",
        "problem",
        "prompt",
        "code_prefix",
        "entry_point",
        "tests",
        "family_keys",
        "source",
    )
    for family_id, family_rows in sorted(grouped.items()):
        family_rows.sort(key=lambda row: (int(row["sample_index"]), str(row["candidate_id"])))
        _require(
            len(family_rows) == rollout.EXPECTED_K,
            f"{family_id} does not contain the frozen K={rollout.EXPECTED_K} samples",
        )
        reference = family_rows[0]
        for row in family_rows[1:]:
            for field in context_fields:
                _require(row.get(field) == reference.get(field), f"{family_id}.{field} drifted across samples")
        generators = [
            _candidate_generator_core(row.get("generator"), f"{row['candidate_id']}.generator")
            for row in family_rows
        ]
        generator_hashes = {str(item["provenance_sha256"]) for item in generators}
        _require(len(generator_hashes) == 1, f"{family_id} generator provenance drifted across samples")
        generator = generators[0]
        core_hashes.add(str(generator["provenance_sha256"]))

        by_response: dict[str, list[tuple[dict[str, Any], bool]]] = {}
        for row in family_rows:
            eligible = _validate_format_contract(
                row.get("format_contract"), f"{row['candidate_id']}.format_contract"
            )
            response = _mapping(row.get("response"), f"{row['candidate_id']}.response")
            by_response.setdefault(str(response["sha256"]), []).append((row, eligible))

        retained: list[dict[str, Any]] = []
        pre_exclusions: list[dict[str, Any]] = []
        for response_sha256, duplicates in sorted(by_response.items()):
            eligibility = {eligible for _, eligible in duplicates}
            _require(
                len(eligibility) == 1,
                f"{family_id} identical response bytes have contradictory format contracts",
            )
            duplicates.sort(key=lambda item: (int(item[0]["sample_index"]), str(item[0]["candidate_id"])))
            representative, eligible = duplicates[0]
            if not eligible:
                for row, _ in duplicates:
                    format_invalid_count += 1
                    pre_exclusions.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "reason_code": "format_contract_not_execution_eligible",
                            "source_candidate_sha256": row["candidate_sha256"],
                        }
                    )
                continue
            normalized = _normalized_rollout_candidate(
                representative,
                family_id=family_id,
                generator=generator,
            )
            # Valid rows must also satisfy the final pair's syntax/AST contract.
            _response_ast_sha256(
                normalized["text"],
                str(reference["code_prefix"]),
                str(normalized["candidate_id"]),
            )
            retained.append(normalized)
            for row, _ in duplicates[1:]:
                duplicate_count += 1
                pre_exclusions.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "reason_code": "exact_response_duplicate",
                        "duplicate_of_candidate_id": representative["candidate_id"],
                        "response_sha256": response_sha256,
                        "source_candidate_sha256": row["candidate_sha256"],
                    }
                )

        internal = {
            "schema_name": ROLLOUT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "task_id": _task_id(reference["task_id"], f"{family_id}.task_id"),
            "family_id": family_id,
            "split": reference["split"],
            "family_keys": copy.deepcopy(reference["family_keys"]),
            "source": copy.deepcopy(reference["source"]),
            "problem": copy.deepcopy(reference["problem"]),
            "prompt": copy.deepcopy(reference["prompt"]),
            "code_prefix": reference["code_prefix"],
            "entry_point": reference["entry_point"],
            "tests": copy.deepcopy(reference["tests"]),
            "generator": generator,
            "candidates": retained,
            "preselection_exclusions": sorted(
                pre_exclusions, key=lambda item: str(item["candidate_id"])
            ),
        }
        normalized_families.append(seal_rollout(internal))
    _require(
        len(core_hashes) == 1,
        "rollout candidates do not share one exact promoted-S1 generator provenance",
    )
    return normalized_families, {
        "input_candidate_rows": len(rows),
        "retained_execution_unique_candidates": sum(
            len(row["candidates"]) for row in normalized_families
        ),
        "exact_response_duplicates": duplicate_count,
        "format_ineligible_candidates": format_invalid_count,
    }


def prepare_candidate_replay_requests(
    rollout_rows: Sequence[Mapping[str, Any]],
    *,
    rollout_contract: Mapping[str, Any] | None = None,
    seed_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate/deduplicate formal rollout rows before any E2B execution.

    Evidence coverage is defined over the emitted requests, not all raw K
    samples: format-ineligible samples are excluded, and byte-identical valid
    responses retain only their lowest sample-index representative.
    """

    _require(bool(rollout_rows), "rollout input is empty")
    if rollout_rows[0].get("schema_name") == CANDIDATE_ROLLOUT_SCHEMA:
        _require(rollout_contract is not None, "candidate rows require rollout_contract")
        _require(seed_manifest is not None, "candidate rows require seed_manifest")
        families, counts = _normalize_candidate_rollouts(
            rollout_rows,
            rollout_contract=rollout_contract,
            seed_manifest=seed_manifest,
        )
    else:
        _require(
            all(row.get("schema_name") == ROLLOUT_SCHEMA for row in rollout_rows),
            "rollout input mixes or uses an unknown schema",
        )
        families = [_validate_rollout(row, index) for index, row in enumerate(rollout_rows)]
        counts = {
            "input_candidate_rows": sum(len(row["candidates"]) for row in families),
            "retained_execution_unique_candidates": sum(
                len(row["candidates"]) for row in families
            ),
            "exact_response_duplicates": 0,
            "format_ineligible_candidates": 0,
        }
    requests: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for family in sorted(families, key=lambda item: str(item["family_id"])):
        exclusions.extend(
            {
                "family_id": family["family_id"],
                **copy.deepcopy(item),
            }
            for item in family.get("preselection_exclusions", [])
        )
        for candidate in sorted(
            family["candidates"], key=lambda item: str(item["candidate_id"])
        ):
            candidate_generator = _candidate_generator(family, candidate)
            request = {
                "schema_name": CANDIDATE_REPLAY_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate["candidate_id"],
                "task_id": family["task_id"],
                "family_id": family["family_id"],
                "split": family["split"],
                "code_prefix": family["code_prefix"],
                "entry_point": family["entry_point"],
                "tests": copy.deepcopy(family["tests"]),
                "tests_sha256": family["tests"]["sha256"],
                "candidate": {
                    "candidate_id": candidate["candidate_id"],
                    "origin": ORIGIN,
                    "text": candidate["text"],
                    "sha256": candidate["sha256"],
                },
                "generator_provenance_sha256": candidate_generator["provenance_sha256"],
                "source_rollout_candidate_sha256": candidate.get(
                    "source_candidate_sha256", candidate["candidate_sha256"]
                ),
            }
            requests.append(_seal(request, "request_sha256"))
    summary = _seal(
        {
            "schema_name": "day22.s1_candidate_replay_coverage",
            "schema_version": SCHEMA_VERSION,
            "status": "ready_for_e2b",
            "coverage_rule": (
                "one request per format-valid unique response; retain the lowest "
                "sample_index representative; every emitted request requires exactly "
                "two fresh E2B runs"
            ),
            "counts": {**counts, "replay_requests": len(requests)},
            "excluded_candidates": sorted(
                exclusions,
                key=lambda item: (str(item["family_id"]), str(item["candidate_id"])),
            ),
            "ordered_request_sha256": contract.object_sha256(
                [request["request_sha256"] for request in requests]
            ),
        },
        "summary_sha256",
    )
    return {"requests": requests, "summary": summary}


def _validate_rollout(value: Any, index: int) -> dict[str, Any]:
    label = f"rollout row {index}"
    row = copy.deepcopy(dict(_mapping(value, label)))
    _verify_self_hash(row, "row_sha256", label)
    _require(row.get("schema_name") == ROLLOUT_SCHEMA, f"{label} schema_name drifted")
    _require(row.get("schema_version") == SCHEMA_VERSION, f"{label} schema_version drifted")
    task_id = _task_id(row.get("task_id"), f"{label}.task_id")
    family_id = _text(row.get("family_id"), f"{label}.family_id")
    _require(family_id == f"mbpp:task:{task_id}", f"{label}.family_id is not the native MBPP family")
    _require(row.get("split") in contract.SPLITS, f"{label}.split is invalid")
    generator = _validate_generator(row.get("generator"), f"{label}.generator")
    raw_generators = row.get("generators")
    if raw_generators is None:
        generators = [generator]
    else:
        _require(isinstance(raw_generators, list), f"{label}.generators must be a list")
        generators = _validate_generator_set(raw_generators, f"{label}.generators")
        _require(
            str(generator["provenance_sha256"])
            in {str(item["provenance_sha256"]) for item in generators},
            f"{label}.generator is not included in generators",
        )
    generator_sha256s = {str(item["provenance_sha256"]) for item in generators}
    _validate_hashed_text(row.get("problem"), f"{label}.problem")
    _validate_hashed_text(row.get("prompt"), f"{label}.prompt")
    tests = _validate_tests(row.get("tests"), f"{label}.tests")
    _text(row.get("entry_point"), f"{label}.entry_point")
    code_prefix = _text(row.get("code_prefix"), f"{label}.code_prefix")
    _mapping(row.get("family_keys"), f"{label}.family_keys")
    _mapping(row.get("source"), f"{label}.source")
    raw_candidates = row.get("candidates")
    minimum_candidates = 0 if "preselection_exclusions" in row else 2
    _require(
        isinstance(raw_candidates, list) and len(raw_candidates) >= minimum_candidates,
        f"{label}.candidates needs at least {minimum_candidates} samples",
    )
    candidates = [
        _validate_candidate(
            candidate,
            family_id=family_id,
            code_prefix=code_prefix,
            generator_sha256s=generator_sha256s,
            label=f"{label}.candidates[{candidate_index}]",
        )
        for candidate_index, candidate in enumerate(raw_candidates)
    ]
    ids = [candidate["candidate_id"] for candidate in candidates]
    _require(len(ids) == len(set(ids)), f"{label} has duplicate candidate IDs")
    indices = [candidate["sample_index"] for candidate in candidates]
    _require(len(indices) == len(set(indices)), f"{label} has duplicate sample indices")
    row["task_id"] = task_id
    row["family_id"] = family_id
    row["generator"] = generator
    if raw_generators is not None:
        row["generators"] = generators
    row["tests"] = tests
    row["candidates"] = candidates
    return row


def _validate_secure_sandbox(sandbox: Any, digest: str, label: str) -> None:
    value = _mapping(sandbox, label)
    _require(contract.object_sha256(value) == digest, f"{label} does not match sandbox_digest")
    required = {
        "backend": "e2b_firecracker",
        "isolation": "fresh_security_sandbox",
        "secure": True,
        "allow_internet_access": False,
        "fresh_sandbox_per_run": True,
    }
    for field, expected in required.items():
        _require(value.get(field) == expected, f"{label}.{field} is not securely attested")


def _validate_pass_ratio_basis(value: Any, label: str, status_class: str) -> tuple[Fraction | None, str | None]:
    if value is None:
        return None, None
    basis = _mapping(value, label)
    digest = _verify_self_hash(basis, "basis_sha256", label)
    _require(basis.get("method") == "isolated_test_replay", f"{label}.method is not attested")
    passed = _integer(basis.get("tests_passed"), f"{label}.tests_passed")
    total = _integer(basis.get("tests_total"), f"{label}.tests_total", minimum=1)
    _require(passed <= total, f"{label}.tests_passed exceeds total")
    if status_class == STABLE_PASS:
        _require(passed == total, f"{label} contradicts stable pass")
    if status_class == STABLE_WRONG:
        _require(passed < total, f"{label} contradicts stable wrong_answer")
    return Fraction(passed, total), digest


def _classify_statuses(statuses: Sequence[str]) -> tuple[str, str]:
    if list(statuses) == ["pass", "pass"]:
        return STABLE_PASS, "two_e2b_runs_pass"
    if list(statuses) == ["wrong_answer", "wrong_answer"]:
        return STABLE_WRONG, "two_e2b_runs_wrong_answer"
    if len(set(statuses)) > 1:
        return QUARANTINE, "unstable_execution"
    return QUARANTINE, f"{statuses[0]}_not_preference_eligible"


def _validate_evidence(
    value: Any,
    *,
    candidate: Mapping[str, Any],
    rollout: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    label = f"candidate evidence {index}"
    evidence = copy.deepcopy(dict(_mapping(value, label)))
    evidence_sha256 = _verify_self_hash(evidence, "evidence_sha256", label)
    candidate_generator = _candidate_generator(rollout, candidate)
    _require(evidence.get("schema_name") == EVIDENCE_SCHEMA, f"{label} schema_name drifted")
    _require(evidence.get("schema_version") == SCHEMA_VERSION, f"{label} schema_version drifted")
    _require(evidence.get("backend") == "e2b", f"{label} must use E2B")
    for field, expected in (
        ("candidate_id", candidate["candidate_id"]),
        ("family_id", rollout["family_id"]),
        ("task_id", rollout["task_id"]),
        ("response_sha256", candidate["sha256"]),
        ("generator_provenance_sha256", candidate_generator["provenance_sha256"]),
        ("source_tests_sha256", rollout["tests"]["sha256"]),
        ("test_sha256", rollout["tests"]["sha256"]),
    ):
        actual = str(evidence.get(field)) if field == "task_id" else evidence.get(field)
        _require(actual == expected, f"{label}.{field} drifted")
    if "source_candidate_sha256" in candidate:
        _require(
            evidence.get("source_rollout_candidate_sha256")
            == candidate["source_candidate_sha256"],
            f"{label}.source_rollout_candidate_sha256 drifted",
        )
    sandbox_digest = _sha256(evidence.get("sandbox_digest"), f"{label}.sandbox_digest")
    runs = evidence.get("runs")
    _require(isinstance(runs, list) and len(runs) == 2, f"{label}.runs must contain exactly two fresh runs")
    statuses: list[str] = []
    run_ids: list[str] = []
    for run_index, raw_run in enumerate(runs, 1):
        run_label = f"{label}.runs[{run_index - 1}]"
        run = _mapping(raw_run, run_label)
        _verify_self_hash(run, "run_sha256", run_label)
        _require(run.get("attempt") == run_index, f"{run_label}.attempt drifted")
        status = run.get("status")
        _require(status in EXECUTION_STATUSES, f"{run_label}.status is invalid")
        for field, expected in (
            ("candidate_id", candidate["candidate_id"]),
            ("family_id", rollout["family_id"]),
            ("task_id", rollout["task_id"]),
            ("response_sha256", candidate["sha256"]),
            ("source_tests_sha256", rollout["tests"]["sha256"]),
            ("test_sha256", rollout["tests"]["sha256"]),
            ("sandbox_digest", sandbox_digest),
        ):
            actual = str(run.get(field)) if field == "task_id" else run.get(field)
            _require(actual == expected, f"{run_label}.{field} drifted")
        _validate_secure_sandbox(run.get("sandbox"), sandbox_digest, f"{run_label}.sandbox")
        if status == "wrong_answer":
            _require(run.get("error_type") == "assertion_error", f"{run_label} wrong_answer is not verifier-classified")
        statuses.append(str(status))
        run_ids.append(_text(run.get("run_id"), f"{run_label}.run_id"))
    _require(len(set(run_ids)) == 2, f"{label} run IDs are not fresh/unique")
    status_class, reason_code = _classify_statuses(statuses)
    ratio, ratio_basis_sha256 = _validate_pass_ratio_basis(
        evidence.get("test_pass_ratio_basis"),
        f"{label}.test_pass_ratio_basis",
        status_class,
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "evidence_sha256": evidence_sha256,
        "sandbox_digest": sandbox_digest,
        "statuses": statuses,
        "status_class": status_class,
        "reason_code": reason_code,
        "test_pass_ratio": ratio,
        "test_pass_ratio_basis_sha256": ratio_basis_sha256,
    }


def _labeler_decision(reason_code: str, reason: str) -> dict[str, Any]:
    return _seal(
        {
            **RUBRIC,
            "rubric_sha256": RUBRIC_SHA256,
            "reason_code": reason_code,
            "reason": reason,
        },
        "labeler_sha256",
    )


def _candidate_output(candidate: Mapping[str, Any], generator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "origin": ORIGIN,
        "text": candidate["text"],
        "sha256": candidate["sha256"],
        "ast_sha256": candidate["ast_sha256"],
        "response_token_count": candidate["response_token_count"],
        "sample_index": candidate["sample_index"],
        "sample_seed": candidate["sample_seed"],
        "source_candidate_sha256": candidate.get(
            "source_candidate_sha256", candidate["candidate_sha256"]
        ),
        "generator": copy.deepcopy(generator),
    }


def _relative_length_delta(chosen: Mapping[str, Any], rejected: Mapping[str, Any]) -> Fraction:
    chosen_count = int(chosen["response_token_count"])
    rejected_count = int(rejected["response_token_count"])
    return Fraction(abs(chosen_count - rejected_count), max(chosen_count, rejected_count))


def _decision_reason_without_pair(evidence: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    classes = Counter(str(item["status_class"]) for item in evidence)
    quarantine_reasons = Counter(str(item["reason_code"]) for item in evidence if item["status_class"] == QUARANTINE)
    if classes[STABLE_PASS] and not classes[STABLE_WRONG]:
        return "both_pass", "No stable wrong_answer exists; passing candidates cannot be assigned a preference direction."
    if classes[STABLE_WRONG] and not classes[STABLE_PASS]:
        return "both_fail", "No stable pass exists; failing candidates cannot supply a chosen response."
    if quarantine_reasons:
        priority = {
            "unstable_execution": 0,
            "infra_error_not_preference_eligible": 1,
            "timeout_not_preference_eligible": 2,
            "runtime_error_not_preference_eligible": 3,
            "syntax_error_not_preference_eligible": 4,
        }
        reason = sorted(
            quarantine_reasons,
            key=lambda item: (priority.get(item, 100), -quarantine_reasons[item], item),
        )[0]
        return reason, "No stable pass-vs-wrong_answer pair remains after execution quarantine."
    return "tie_no_execution_separation", "The verifier evidence does not separate chosen from rejected."


def _make_replay(
    rollout: Mapping[str, Any],
    *,
    pair_id: str,
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "schema_name": "day22.mbpp_replay_pair",
        "schema_version": 1,
        "pair_id": pair_id,
        "task_id": rollout["task_id"],
        "family_id": rollout["family_id"],
        "split": rollout["split"],
        "family_keys": copy.deepcopy(rollout["family_keys"]),
        "source": copy.deepcopy(rollout["source"]),
        "problem": copy.deepcopy(rollout["problem"]),
        "prompt": copy.deepcopy(rollout["prompt"]),
        "code_prefix": rollout["code_prefix"],
        "entry_point": rollout["entry_point"],
        "tests": copy.deepcopy(rollout["tests"]),
        "tests_sha256": rollout["tests"]["sha256"],
        "chosen": _candidate_output(chosen, _candidate_generator(rollout, chosen)),
        "rejected": _candidate_output(rejected, _candidate_generator(rollout, rejected)),
    }
    return _seal(row, "row_sha256")


def _select_family(
    rollout: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidates = {str(item["candidate_id"]): item for item in rollout["candidates"]}
    by_id = {str(item["candidate_id"]): item for item in evidence}
    pre_exclusions = copy.deepcopy(list(rollout.get("preselection_exclusions", [])))
    stable_pass = [candidates[candidate_id] for candidate_id, item in by_id.items() if item["status_class"] == STABLE_PASS]
    stable_wrong = [candidates[candidate_id] for candidate_id, item in by_id.items() if item["status_class"] == STABLE_WRONG]

    duplicate_statuses: dict[str, set[str]] = {}
    for candidate_id, candidate in candidates.items():
        duplicate_statuses.setdefault(str(candidate["sha256"]), set()).add(str(by_id[candidate_id]["status_class"]))
    if any(STABLE_PASS in classes and STABLE_WRONG in classes for classes in duplicate_statuses.values()):
        reason_code = "conflicting_duplicate_evidence"
        reason = "Identical response bytes received contradictory stable execution labels."
        combinations: list[tuple[Any, ...]] = []
    else:
        combinations = []
        for chosen in stable_pass:
            for rejected in stable_wrong:
                if chosen["sha256"] == rejected["sha256"] or chosen["ast_sha256"] == rejected["ast_sha256"]:
                    continue
                delta = _relative_length_delta(chosen, rejected)
                ratio = by_id[str(rejected["candidate_id"])]["test_pass_ratio"]
                combinations.append(
                    (
                        delta,
                        0 if ratio is not None else 1,
                        -ratio if ratio is not None else Fraction(0, 1),
                        str(chosen["candidate_id"]),
                        str(rejected["candidate_id"]),
                        chosen,
                        rejected,
                    )
                )
        if combinations:
            combinations.sort(key=lambda item: item[:5])
            selected = combinations[0]
            delta, _, _, _, _, chosen, rejected = selected
            rejected_evidence = by_id[str(rejected["candidate_id"])]
            chosen_generator = _candidate_generator(rollout, chosen)
            rejected_generator = _candidate_generator(rollout, rejected)
            pair_identity = contract.object_sha256(
                {
                    "family_id": rollout["family_id"],
                    "chosen_candidate_id": chosen["candidate_id"],
                    "rejected_candidate_id": rejected["candidate_id"],
                    "chosen_generator_provenance_sha256": chosen_generator["provenance_sha256"],
                    "rejected_generator_provenance_sha256": rejected_generator["provenance_sha256"],
                    "rubric_sha256": RUBRIC_SHA256,
                }
            )
            pair_id = f"{rollout['family_id']}:s1pair:{pair_identity[:16]}"
            replay = _make_replay(
                rollout,
                pair_id=pair_id,
                chosen=chosen,
                rejected=rejected,
            )
            ratio = rejected_evidence["test_pass_ratio"]
            selection = {
                "schema_name": SELECTION_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "selection_status": "eligible",
                "dataset_role": "formal_s1_pair_pending_processor_and_human_review",
                "pair_id": pair_id,
                "task_id": rollout["task_id"],
                "family_id": rollout["family_id"],
                "split": rollout["split"],
                "generator_provenance_sha256s": sorted(
                    {
                        str(chosen_generator["provenance_sha256"]),
                        str(rejected_generator["provenance_sha256"]),
                    }
                ),
                "chosen": _candidate_output(chosen, chosen_generator),
                "rejected": _candidate_output(rejected, rejected_generator),
                "selection_metrics": {
                    "relative_token_length_delta": float(delta),
                    "relative_token_length_delta_fraction": [delta.numerator, delta.denominator],
                    "rejected_test_pass_ratio": float(ratio) if ratio is not None else None,
                    "rejected_test_pass_ratio_basis_sha256": rejected_evidence["test_pass_ratio_basis_sha256"],
                },
                "source_evidence": {
                    "chosen_evidence_sha256": by_id[str(chosen["candidate_id"])]["evidence_sha256"],
                    "rejected_evidence_sha256": rejected_evidence["evidence_sha256"],
                    "sandbox_digest": rejected_evidence["sandbox_digest"],
                    "test_sha256": rollout["tests"]["sha256"],
                },
                "excluded_candidates": pre_exclusions + [
                    {
                        "candidate_id": candidate_id,
                        "status_class": item["status_class"],
                        "reason_code": item["reason_code"],
                        "evidence_sha256": item["evidence_sha256"],
                    }
                    for candidate_id, item in sorted(by_id.items())
                    if candidate_id not in {chosen["candidate_id"], rejected["candidate_id"]}
                ],
                "labeler": _labeler_decision(
                    "stable_pass_vs_stable_wrong_answer",
                    "Chosen passed twice and rejected failed the verifier twice; selection minimizes relative token-length delta, then prefers the hardest attested failing candidate.",
                ),
                "replay_row_sha256": replay["row_sha256"],
            }
            if len(selection["generator_provenance_sha256s"]) == 1:
                selection["generator_provenance_sha256"] = selection[
                    "generator_provenance_sha256s"
                ][0]
            return _seal(selection, "selection_sha256"), replay
        if stable_pass and stable_wrong:
            reason_code = "ast_equivalent_tie"
            reason = "All stable pass-vs-wrong_answer combinations are byte-identical or AST-equivalent."
        elif not candidates and pre_exclusions:
            reason_code = "no_execution_eligible_unique_candidate"
            reason = "All rollout samples were format-ineligible or removed as exact response duplicates."
        else:
            reason_code, reason = _decision_reason_without_pair(evidence)

    selection = {
        "schema_name": SELECTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "selection_status": QUARANTINE,
        "dataset_role": "formal_s1_pair_quarantine",
        "pair_id": None,
        "task_id": rollout["task_id"],
        "family_id": rollout["family_id"],
        "split": rollout["split"],
        "generator_provenance_sha256s": sorted(
            str(item["provenance_sha256"]) for item in _rollout_generators(rollout)
        ),
        "candidate_evidence": [
            {
                "candidate_id": candidate_id,
                "status_class": item["status_class"],
                "statuses": item["statuses"],
                "reason_code": item["reason_code"],
                "evidence_sha256": item["evidence_sha256"],
            }
            for candidate_id, item in sorted(by_id.items())
        ],
        "preselection_exclusions": pre_exclusions,
        "labeler": _labeler_decision(reason_code, reason),
    }
    if len(selection["generator_provenance_sha256s"]) == 1:
        selection["generator_provenance_sha256"] = selection[
            "generator_provenance_sha256s"
        ][0]
    return _seal(selection, "selection_sha256"), None


def label_rollouts(
    rollout_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    rollout_contract: Mapping[str, Any] | None = None,
    seed_manifest: Mapping[str, Any] | None = None,
    input_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(bool(rollout_rows), "rollout input is empty")
    source_schema = rollout_rows[0].get("schema_name")
    normalization_counts: dict[str, int] = {}
    if source_schema == CANDIDATE_ROLLOUT_SCHEMA:
        _require(rollout_contract is not None, "candidate-per-row input requires the frozen rollout contract")
        _require(seed_manifest is not None, "candidate-per-row input requires the frozen seed manifest")
        normalized, normalization_counts = _normalize_candidate_rollouts(
            rollout_rows,
            rollout_contract=rollout_contract,
            seed_manifest=seed_manifest,
        )
        rollouts = [_validate_rollout(row, index) for index, row in enumerate(normalized)]
    else:
        _require(
            all(row.get("schema_name") == ROLLOUT_SCHEMA for row in rollout_rows),
            "rollout input mixes or uses an unknown schema",
        )
        rollouts = [_validate_rollout(row, index) for index, row in enumerate(rollout_rows)]
    families = [str(row["family_id"]) for row in rollouts]
    _require(len(families) == len(set(families)), "rollout input has duplicate families")
    generator_by_hash: dict[str, Mapping[str, Any]] = {}
    for rollout in rollouts:
        for generator in _rollout_generators(rollout):
            digest = str(generator["provenance_sha256"])
            prior = generator_by_hash.get(digest)
            _require(
                prior is None or prior == generator,
                "generator objects drift despite identical provenance hashes",
            )
            generator_by_hash[digest] = generator
    generator_hashes = set(generator_by_hash)
    policy_identities = {
        contract.object_sha256(generator_policy_identity(generator))
        for generator in generator_by_hash.values()
    }
    _require(
        len(policy_identities) == 1,
        "rollout families do not share one exact promoted-S1 policy identity",
    )
    common_policy_identity = generator_policy_identity(next(iter(generator_by_hash.values())))

    candidate_index: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for rollout in rollouts:
        for candidate in rollout["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            _require(candidate_id not in candidate_index, f"duplicate global candidate_id: {candidate_id}")
            candidate_index[candidate_id] = (rollout, candidate)

    raw_evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(evidence_rows):
        row = _mapping(raw, f"candidate evidence {index}")
        candidate_id = _text(row.get("candidate_id"), f"candidate evidence {index}.candidate_id")
        _require(candidate_id in candidate_index, f"evidence references unknown candidate_id: {candidate_id}")
        _require(candidate_id not in raw_evidence_by_id, f"duplicate evidence candidate_id: {candidate_id}")
        raw_evidence_by_id[candidate_id] = row
    _require(set(raw_evidence_by_id) == set(candidate_index), "candidate evidence does not exactly cover rollout candidates")

    validated_evidence: dict[str, dict[str, Any]] = {}
    run_ids: set[str] = set()
    for index, candidate_id in enumerate(sorted(raw_evidence_by_id)):
        rollout, candidate = candidate_index[candidate_id]
        evidence = _validate_evidence(
            raw_evidence_by_id[candidate_id],
            candidate=candidate,
            rollout=rollout,
            index=index,
        )
        raw_runs = raw_evidence_by_id[candidate_id].get("runs", [])
        for run in raw_runs:
            run_id = str(run.get("run_id"))
            _require(run_id not in run_ids, f"duplicate E2B run_id across candidates: {run_id}")
            run_ids.add(run_id)
        validated_evidence[candidate_id] = evidence
    sandbox_digests = {item["sandbox_digest"] for item in validated_evidence.values()}
    _require(
        len(sandbox_digests) == (1 if candidate_index else 0),
        "candidate evidence does not share one pinned E2B sandbox identity",
    )

    selections: list[dict[str, Any]] = []
    replays: list[dict[str, Any]] = []
    for rollout in sorted(rollouts, key=lambda item: str(item["family_id"])):
        evidence = [validated_evidence[str(candidate["candidate_id"])] for candidate in rollout["candidates"]]
        selection, replay = _select_family(rollout, evidence)
        selections.append(selection)
        if replay is not None:
            replays.append(replay)
    pair_ids = [str(row["pair_id"]) for row in replays]
    _require(len(pair_ids) == len(set(pair_ids)), "selected pair IDs are not unique")
    _require(len(replays) <= len(rollouts), "more than one pair was emitted per family")

    decision_counts = Counter(str(row["labeler"]["reason_code"]) for row in selections)
    class_counts = Counter(str(item["status_class"]) for item in validated_evidence.values())
    summary = {
        "schema_name": SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "dataset_role": "formal_s1_pair_selection_pending_processor_and_human_review",
        "formal_dpo_ready": False,
        "formal_dpo_blockers": ["processor_audit_pending", "human_blind_review_pending", "final_assembly_pending"],
        "rubric": {**RUBRIC, "rubric_sha256": RUBRIC_SHA256},
        "generator_provenance_sha256s": sorted(generator_hashes),
        "common_policy_identity": common_policy_identity,
        "common_policy_identity_sha256": contract.object_sha256(common_policy_identity),
        "common_sandbox_digest": next(iter(sandbox_digests)) if sandbox_digests else None,
        "inputs": dict(input_identities or {}),
        "counts": {
            "rollout_families": len(rollouts),
            "rollout_candidates": len(candidate_index),
            "candidate_evidence": len(validated_evidence),
            "eligible_pairs": len(replays),
            "quarantined_families": len(rollouts) - len(replays),
            "candidate_execution_classes": dict(sorted(class_counts.items())),
            "decision_reasons": dict(sorted(decision_counts.items())),
            **normalization_counts,
        },
        "content_identities": {
            "ordered_selection_sha256": contract.object_sha256([row["selection_sha256"] for row in selections]),
            "ordered_replay_sha256": contract.object_sha256([row["row_sha256"] for row in replays]),
        },
    }
    if len(generator_hashes) == 1:
        summary["common_generator_provenance_sha256"] = next(iter(generator_hashes))
    summary = _seal(summary, "summary_sha256")
    return {"selections": selections, "replays": replays, "summary": summary}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FormalS1LabelerError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise FormalS1LabelerError(f"invalid JSONL: {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise FormalS1LabelerError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalS1LabelerError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(contract.canonical_json(row) + b"\n" for row in rows)


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FormalS1LabelerError(f"refusing to overwrite: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
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
                raise FormalS1LabelerError(f"output appeared while writing: {resolved}") from error
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _preflight_output_paths(paths: Sequence[Path], *, overwrite: bool) -> list[Path]:
    resolved = [path.resolve() for path in paths]
    _require(len(resolved) == len(set(resolved)), "output paths must be distinct")
    if not overwrite:
        existing = next((path for path in resolved if path.exists()), None)
        _require(existing is None, f"refusing to overwrite: {existing}")
    return resolved


def run_cli(
    *,
    rollouts_path: Path,
    evidence_path: Path,
    selections_output: Path,
    replay_output: Path,
    summary_output: Path,
    overwrite: bool,
    rollout_contract_path: Path | None = None,
    seed_manifest_path: Path | None = None,
) -> dict[str, Any]:
    paths = _preflight_output_paths(
        [selections_output, replay_output, summary_output],
        overwrite=overwrite,
    )
    rollout_rows = load_jsonl(rollouts_path.resolve())
    contract_value = load_json(rollout_contract_path.resolve()) if rollout_contract_path else None
    seed_value = load_json(seed_manifest_path.resolve()) if seed_manifest_path else None
    identities: dict[str, Any] = {
        "rollouts": {"path": str(rollouts_path), "file_sha256": file_sha256(rollouts_path.resolve())},
        "candidate_evidence": {"path": str(evidence_path), "file_sha256": file_sha256(evidence_path.resolve())},
    }
    if rollout_contract_path is not None:
        identities["rollout_contract"] = {
            "path": str(rollout_contract_path),
            "file_sha256": file_sha256(rollout_contract_path.resolve()),
        }
    if seed_manifest_path is not None:
        identities["seed_manifest"] = {
            "path": str(seed_manifest_path),
            "file_sha256": file_sha256(seed_manifest_path.resolve()),
        }
    result = label_rollouts(
        rollout_rows,
        load_jsonl(evidence_path.resolve()),
        rollout_contract=contract_value,
        seed_manifest=seed_value,
        input_identities=identities,
    )
    _write_atomic(paths[0], _jsonl_bytes(result["selections"]), overwrite=overwrite)
    _write_atomic(paths[1], _jsonl_bytes(result["replays"]), overwrite=overwrite)
    _write_atomic(paths[2], _json_bytes(result["summary"]), overwrite=overwrite)
    return {
        "status": "pass",
        "eligible_pairs": len(result["replays"]),
        "quarantined_families": len(result["selections"]) - len(result["replays"]),
        "selections_output": str(paths[0]),
        "selections_file_sha256": file_sha256(paths[0]),
        "replay_output": str(paths[1]),
        "replay_file_sha256": file_sha256(paths[1]),
        "summary_output": str(paths[2]),
        "summary_file_sha256": file_sha256(paths[2]),
        "summary_sha256": result["summary"]["summary_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument(
        "--rollout-contract",
        type=Path,
        help="required for candidate-per-row rollout input",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        help="required for candidate-per-row rollout input",
    )
    parser.add_argument("--selections-output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_cli(
            rollouts_path=args.rollouts,
            evidence_path=args.candidate_evidence,
            selections_output=args.selections_output,
            replay_output=args.replay_output,
            summary_output=args.summary_output,
            overwrite=args.overwrite,
            rollout_contract_path=args.rollout_contract,
            seed_manifest_path=args.seed_manifest,
        )
    except FormalS1LabelerError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
