#!/usr/bin/env python3
"""Independently validate a Day 22 formal promoted-S1 preference bundle.

This validator consumes only the formal assembly manifest and its content-bound
files.  It does not import the formal assembler or labeler, and it never falls
back to the audited-smoke dataset.  A valid pending-review bundle exits zero:
all machine gates have passed, while formal DPO readiness remains blocked only
on the explicitly unfinished human blind review.  ``--require-formal-ready``
turns that expected pending state into exit code 3 for downstream training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
BOOTCAMP_ROOT = SCRIPT_PATH.parents[2]
PAIR_CONTRACT_PATH = BOOTCAMP_ROOT / "day-22-preference-data/day22_contract.py"
REVIEW_PROTOCOL_PATH = (
    BOOTCAMP_ROOT / "day-22-preference-data/FORMAL-S1-BLIND-REVIEW-PROTOCOL.md"
)

FORMAL_ROLE = "formal_s1_dpo_pairs_pending_human_blind_review"
PENDING_BLOCKER = "human_blind_review_pending"
READY_MANIFEST_SCHEMA = "day22.formal_s1_ready_manifest"
REVIEW_AUDIT_SCHEMA = "day22.formal_s1_human_review_audit"
MIN_PAIRS = 200
MAX_PAIRS = 500
REVIEW_PAIRS = 50
SPLITS = ("train", "dev", "heldout")
FAMILY_KEYS = ("problem", "prompt", "test", "source")
ORIGIN = "promoted_s1_rollout"
EXPECTED_INPUTS = {
    "selections",
    "replay_input",
    "selection_summary",
    "candidate_evidence",
    "processor_audit",
    "processor_contract",
}
OUTPUT_SCHEMAS = {
    "day22.coding_preference_pair": "pairs",
    "day22.formal_s1_family_split_ids": "split_ids",
    "day22.formal_s1_machine_gate_audit": "assembly_audit",
    "day22.blind_review_item": "review_worksheet",
    "day22.blind_review_concealed_key": "review_key",
}
MACHINE_GATE_NAMES = {
    "formal_pair_count_200_to_500",
    "one_pair_per_family",
    "cross_split_family_overlap",
    "promoted_s1_on_policy",
    "synthetic_pairs",
    "secure_e2b_two_fresh_runs_per_side",
    "qwen35_processor_audit",
    "blind_review_worksheet_generation",
}
EXECUTION_STATUSES = {
    "pass",
    "wrong_answer",
    "syntax_error",
    "runtime_error",
    "timeout",
    "infra_error",
}
REVIEW_KEYS = {
    "schema_name",
    "schema_version",
    "review_item_id",
    "prompt",
    "response_a",
    "response_b",
    "rubric_version",
    "allowed_verdicts",
    "verdict",
    "confidence",
    "notes",
}
REVIEW_VERDICTS = ["A", "B", "tie", "ambiguous", "reject"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


class FormalS1ValidationError(ValueError):
    """A formal bundle identity, join, or machine-gate invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise FormalS1ValidationError("text hash input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FormalS1ValidationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalS1ValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _text(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()) and "\x00" not in value,
        f"{label} must be non-empty text without NUL",
    )
    return str(value)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase bare SHA-256",
    )
    return str(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def _canonical_task_id(value: Any, label: str) -> str:
    _require(
        isinstance(value, (str, int)) and not isinstance(value, bool),
        f"{label} must be a canonical non-negative integer",
    )
    try:
        result = str(int(value))
    except (TypeError, ValueError) as error:
        raise FormalS1ValidationError(
            f"{label} must be a canonical non-negative integer"
        ) from error
    _require(result == str(value).strip() and int(result) >= 0, f"{label} is not canonical")
    return result


def _equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label} drifted")


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    _require(actual == expected, f"{label}.{field} does not bind its contents")
    return expected


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalS1ValidationError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FormalS1ValidationError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise FormalS1ValidationError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from error
        _require(isinstance(row, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def _load_pair_contract() -> ModuleType:
    _require(PAIR_CONTRACT_PATH.is_file(), "Day 22 pair contract module is missing")
    spec = importlib.util.spec_from_file_location(
        "_day22_contract_for_formal_validator", PAIR_CONTRACT_PATH
    )
    _require(spec is not None and spec.loader is not None, "cannot load pair contract")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise FormalS1ValidationError("cannot import Day 22 pair contract") from error
    return module


def _index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = _text(row.get(key), f"{label}[{index}].{key}")
        _require(identity not in result, f"duplicate {label} {key}: {identity}")
        result[identity] = row
    return result


def _resolve_identity(
    identity: Mapping[str, Any], *, manifest_path: Path, label: str
) -> Path:
    recorded = Path(_text(identity.get("path"), f"{label}.path")).expanduser()
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.extend([manifest_path.parent / recorded, BOOTCAMP_ROOT / recorded])
    existing = {path.resolve() for path in candidates if path.is_file()}
    _require(bool(existing), f"recorded file is missing: {recorded}")
    expected = _sha256(identity.get("file_sha256"), f"{label}.file_sha256")
    matching = [path for path in sorted(existing) if file_sha256(path) == expected]
    _require(len(matching) == 1, f"{label} has no unique file matching its hash")
    return matching[0]


def _manifest_files(
    manifest: Mapping[str, Any], manifest_path: Path
) -> tuple[dict[str, Path], dict[str, Path]]:
    inputs = _mapping(manifest.get("inputs"), "manifest.inputs")
    _equal(set(inputs), EXPECTED_INPUTS, "formal manifest input set")
    input_paths = {
        name: _resolve_identity(
            _mapping(inputs[name], f"manifest.inputs.{name}"),
            manifest_path=manifest_path,
            label=f"manifest.inputs.{name}",
        )
        for name in sorted(EXPECTED_INPUTS)
    }

    raw_outputs = _mapping(manifest.get("output_files"), "manifest.output_files")
    _require(len(raw_outputs) == len(OUTPUT_SCHEMAS), "formal manifest must bind five output files")
    output_paths: dict[str, Path] = {}
    for name, raw_identity in raw_outputs.items():
        identity = _mapping(raw_identity, f"manifest.output_files.{name}")
        path = _resolve_identity(
            identity, manifest_path=manifest_path, label=f"manifest.output_files.{name}"
        )
        recorded_name = Path(_text(identity.get("path"), f"output {name}.path")).name
        _equal(recorded_name, name, f"manifest output {name} basename")
        try:
            content = path.read_text(encoding="utf-8")
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                value = json.loads(content.splitlines()[0])
        except (OSError, IndexError, json.JSONDecodeError) as error:
            raise FormalS1ValidationError(f"cannot identify formal output: {path}") from error
        schema = _text(_mapping(value, f"output {name} first record").get("schema_name"), f"output {name} schema")
        _require(schema in OUTPUT_SCHEMAS, f"unexpected formal output schema: {schema}")
        role = OUTPUT_SCHEMAS[schema]
        _require(role not in output_paths, f"duplicate formal output role: {role}")
        output_paths[role] = path
    _equal(set(output_paths), set(OUTPUT_SCHEMAS.values()), "formal output schema set")
    _require(
        len(set(input_paths.values()) | set(output_paths.values()))
        == len(input_paths) + len(output_paths),
        "formal manifest aliases input and output files",
    )
    return input_paths, output_paths


def _validate_generator(value: Any, label: str) -> str:
    generator = _mapping(value, label)
    digest = _verify_self_hash(generator, "provenance_sha256", label)
    _equal(generator.get("promoted_s1"), True, f"{label}.promoted_s1")
    _equal(generator.get("on_policy"), True, f"{label}.on_policy")
    _text(generator.get("checkpoint"), f"{label}.checkpoint")
    _text(generator.get("downstream_key"), f"{label}.downstream_key")
    if "generation_config" in generator:
        config = _mapping(generator.get("generation_config"), f"{label}.generation_config")
        _require(bool(config), f"{label}.generation_config is empty")
        _equal(
            generator.get("generation_config_sha256"),
            object_sha256(config),
            f"{label}.generation_config_sha256",
        )
    return digest


def _generator_policy_identity(value: Any, label: str) -> dict[str, Any]:
    generator = _mapping(value, label)
    _validate_generator(generator, label)
    identity = {
        field: generator[field]
        for field in POLICY_IDENTITY_FIELDS
        if field in generator
    }
    identity["promoted_s1"] = generator.get("promoted_s1")
    identity["on_policy"] = generator.get("on_policy")
    return identity


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
    result = {
        _sha256(value, f"generator provenance[{index}]")
        for index, value in enumerate(values)
    }
    _equal(len(result), len(values), "generator provenance set uniqueness")
    common = summary.get("common_generator_provenance_sha256")
    if common is not None:
        _require(len(result) == 1 and common in result, "legacy common generator drifted")
    return result


def _validate_processor_contract(contract: Mapping[str, Any]) -> str:
    digest = _verify_self_hash(contract, "contract_sha256", "processor contract")
    _equal(contract.get("schema_name"), "day22.qwen35_processor_contract", "processor contract schema")
    _equal(contract.get("schema_version"), 1, "processor contract version")
    _equal(contract.get("status"), "frozen", "processor contract status")
    _integer(contract.get("max_length"), "processor contract max_length", minimum=1)
    for field in ("processor_sha256", "tokenizer_sha256", "template_sha256"):
        _sha256(contract.get(field), f"processor contract.{field}")
    return digest


def _validate_selection_inputs(
    selections: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], set[str], str]:
    _verify_self_hash(summary, "summary_sha256", "selection summary")
    _equal(summary.get("schema_name"), "day22.s1_pair_selection_summary", "selection summary schema")
    _equal(summary.get("schema_version"), 1, "selection summary version")
    _equal(summary.get("status"), "pass", "selection summary status")
    allowed_generators = _summary_generator_hashes(summary)
    common_policy_sha = summary.get("common_policy_identity_sha256")
    if common_policy_sha is not None:
        common_policy = _mapping(summary.get("common_policy_identity"), "common policy identity")
        _equal(common_policy_sha, object_sha256(common_policy), "common policy identity hash")
    common_sandbox = _sha256(summary.get("common_sandbox_digest"), "common sandbox")

    eligible: dict[str, Mapping[str, Any]] = {}
    families: set[str] = set()
    for index, selection in enumerate(selections):
        label = f"selection[{index}]"
        _verify_self_hash(selection, "selection_sha256", label)
        _equal(selection.get("schema_name"), "day22.s1_pair_selection", f"{label} schema")
        _equal(selection.get("schema_version"), 1, f"{label} version")
        family = _text(selection.get("family_id"), f"{label}.family_id")
        _require(family not in families, f"duplicate selected family: {family}")
        families.add(family)
        status = selection.get("selection_status")
        _require(status in {"eligible", "quarantine"}, f"{label} status is invalid")
        if status == "quarantine":
            _equal(selection.get("pair_id"), None, f"{label} quarantined pair_id")
            continue
        pair_id = _text(selection.get("pair_id"), f"{label}.pair_id")
        _require(pair_id not in eligible, f"duplicate eligible pair_id: {pair_id}")
        _equal(
            selection.get("dataset_role"),
            "formal_s1_pair_pending_processor_and_human_review",
            f"{label} dataset role",
        )
        selection_generators = selection.get("generator_provenance_sha256s")
        if selection_generators is None:
            selection_generators = [selection.get("generator_provenance_sha256")]
        _require(
            isinstance(selection_generators, list)
            and bool(selection_generators)
            and set(selection_generators) <= allowed_generators,
            f"{label} generator set drifted",
        )
        decision = _mapping(selection.get("labeler"), f"{label}.labeler")
        _verify_self_hash(decision, "labeler_sha256", f"{label}.labeler")
        _equal(decision.get("reason_code"), "stable_pass_vs_stable_wrong_answer", f"{label} reason")
        _sha256(decision.get("rubric_sha256"), f"{label}.rubric_sha256")
        pair_policy_identities: set[str] = set()
        for side in ("chosen", "rejected"):
            candidate = _mapping(selection.get(side), f"{label}.{side}")
            _equal(candidate.get("origin"), ORIGIN, f"{label}.{side}.origin")
            response = _text(candidate.get("text"), f"{label}.{side}.text")
            _equal(candidate.get("sha256"), text_sha256(response), f"{label}.{side}.sha256")
            generator = candidate.get("generator")
            _require(
                _validate_generator(generator, f"{label}.{side}.generator")
                in allowed_generators,
                f"{label}.{side} generator identity",
            )
            pair_policy_identities.add(
                object_sha256(
                    _generator_policy_identity(generator, f"{label}.{side}.generator")
                )
            )
        _equal(len(pair_policy_identities), 1, f"{label} promoted S1 policy identity")
        if common_policy_sha is not None:
            _equal(next(iter(pair_policy_identities)), common_policy_sha, f"{label} common policy")
        eligible[pair_id] = selection

    replay_by_pair = _index_unique(replays, "pair_id", "formal replay")
    _equal(set(replay_by_pair), set(eligible), "eligible selection/replay membership")
    replay_families: set[str] = set()
    for index, replay in enumerate(replays):
        pair_id = str(replay["pair_id"])
        selection = eligible[pair_id]
        label = f"replay[{index}]"
        _verify_self_hash(replay, "row_sha256", label)
        _equal(replay.get("schema_name"), "day22.mbpp_replay_pair", f"{label} schema")
        _equal(replay.get("schema_version"), 1, f"{label} version")
        _equal(selection.get("replay_row_sha256"), replay.get("row_sha256"), f"{label} selection join")
        task_id = _canonical_task_id(replay.get("task_id"), f"{label}.task_id")
        family_id = _text(replay.get("family_id"), f"{label}.family_id")
        _equal(family_id, f"mbpp:task:{task_id}", f"{label} native family")
        _require(family_id not in replay_families, f"multiple replay pairs for {family_id}")
        replay_families.add(family_id)
        _require(replay.get("split") in SPLITS, f"{label} split is invalid")
        _equal(replay.get("tests_sha256"), replay.get("tests", {}).get("sha256"), f"{label} tests")
        for field in ("task_id", "family_id", "split"):
            _equal(str(selection.get(field)), str(replay.get(field)), f"{label}.{field}")
        for side in ("chosen", "rejected"):
            selected = _mapping(selection.get(side), f"selection {pair_id}.{side}")
            candidate = _mapping(replay.get(side), f"replay {pair_id}.{side}")
            for field in ("candidate_id", "origin", "text", "sha256", "ast_sha256", "generator"):
                _equal(candidate.get(field), selected.get(field), f"replay {pair_id}.{side}.{field}")
            _require(
                _validate_generator(candidate.get("generator"), f"replay {pair_id}.{side}.generator")
                in allowed_generators,
                f"replay {pair_id}.{side} generator",
            )

    identities = _mapping(summary.get("content_identities"), "selection summary identities")
    _equal(
        identities.get("ordered_selection_sha256"),
        object_sha256([row.get("selection_sha256") for row in selections]),
        "ordered selection identity",
    )
    _equal(
        identities.get("ordered_replay_sha256"),
        object_sha256([row.get("row_sha256") for row in replays]),
        "ordered replay identity",
    )
    counts = _mapping(summary.get("counts"), "selection summary counts")
    _equal(counts.get("eligible_pairs"), len(replays), "selection summary eligible count")
    return eligible, replay_by_pair, allowed_generators, common_sandbox


def _validate_candidate_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    allowed_generators: set[str],
    common_sandbox: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    counts = _mapping(summary.get("counts"), "selection summary counts")
    _equal(len(rows), counts.get("candidate_evidence"), "candidate evidence count")
    _equal(counts.get("candidate_evidence"), counts.get("rollout_candidates"), "candidate evidence coverage")
    by_id = _index_unique(rows, "candidate_id", "candidate evidence")
    classes: dict[str, str] = {}
    global_run_ids: set[str] = set()
    for index, evidence in enumerate(rows):
        label = f"candidate evidence[{index}]"
        _verify_self_hash(evidence, "evidence_sha256", label)
        _equal(evidence.get("schema_name"), "day22.s1_candidate_e2b_evidence", f"{label} schema")
        _equal(evidence.get("schema_version"), 1, f"{label} version")
        _equal(evidence.get("backend"), "e2b", f"{label} backend")
        _require(
            evidence.get("generator_provenance_sha256") in allowed_generators,
            f"{label} generator is not in the selection summary set",
        )
        _equal(evidence.get("sandbox_digest"), common_sandbox, f"{label} sandbox")
        _equal(evidence.get("source_tests_sha256"), evidence.get("test_sha256"), f"{label} full tests")
        candidate_id = str(evidence["candidate_id"])
        statuses: list[str] = []
        runs = _list(evidence.get("runs"), f"{label}.runs")
        _require(len(runs) == 2, f"{label} must contain two fresh runs")
        for attempt, raw_run in enumerate(runs, 1):
            run = _mapping(raw_run, f"{label}.runs[{attempt - 1}]")
            run_label = f"{label}.runs[{attempt - 1}]"
            _verify_self_hash(run, "run_sha256", run_label)
            for field, expected in (
                ("attempt", attempt),
                ("candidate_id", candidate_id),
                ("task_id", evidence.get("task_id")),
                ("family_id", evidence.get("family_id")),
                ("response_sha256", evidence.get("response_sha256")),
                ("source_tests_sha256", evidence.get("source_tests_sha256")),
                ("test_sha256", evidence.get("test_sha256")),
                ("sandbox_digest", common_sandbox),
            ):
                _equal(str(run.get(field)) if field == "task_id" else run.get(field), str(expected) if field == "task_id" else expected, f"{run_label}.{field}")
            status = run.get("status")
            _require(status in EXECUTION_STATUSES, f"{run_label}.status is invalid")
            if status == "wrong_answer":
                _equal(run.get("error_type"), "assertion_error", f"{run_label} wrong_answer classifier")
            sandbox = _mapping(run.get("sandbox"), f"{run_label}.sandbox")
            _equal(object_sha256(sandbox), common_sandbox, f"{run_label} sandbox digest")
            for field, expected in (
                ("backend", "e2b_firecracker"),
                ("isolation", "fresh_security_sandbox"),
                ("secure", True),
                ("allow_internet_access", False),
                ("fresh_sandbox_per_run", True),
            ):
                _equal(sandbox.get(field), expected, f"{run_label}.sandbox.{field}")
            for stream in ("stdout", "stderr"):
                value = run.get(stream)
                _require(isinstance(value, str), f"{run_label}.{stream} must be text")
                _equal(run.get(f"{stream}_sha256"), text_sha256(value), f"{run_label}.{stream} hash")
            run_id = _text(run.get("run_id"), f"{run_label}.run_id")
            _require(run_id not in global_run_ids, f"duplicate E2B run_id: {run_id}")
            global_run_ids.add(run_id)
            statuses.append(str(status))
        if statuses == ["pass", "pass"]:
            classes[candidate_id] = "stable_pass"
        elif statuses == ["wrong_answer", "wrong_answer"]:
            classes[candidate_id] = "stable_wrong_answer"
        elif len(set(statuses)) > 1:
            classes[candidate_id] = "unstable_execution"
        else:
            classes[candidate_id] = f"{statuses[0]}_not_preference_eligible"
    return by_id, classes


def _validate_selected_evidence(
    replays: Mapping[str, Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    classes: Mapping[str, str],
) -> None:
    for pair_id, replay in replays.items():
        selection = selections[pair_id]
        source = _mapping(selection.get("source_evidence"), f"selection {pair_id}.source_evidence")
        for side, expected_class in (
            ("chosen", "stable_pass"),
            ("rejected", "stable_wrong_answer"),
        ):
            candidate = replay[side]
            candidate_id = str(candidate["candidate_id"])
            _require(candidate_id in evidence, f"pair {pair_id}.{side} evidence is missing")
            row = evidence[candidate_id]
            _equal(row.get("evidence_sha256"), source.get(f"{side}_evidence_sha256"), f"pair {pair_id}.{side} evidence identity")
            for field, expected in (
                ("task_id", replay["task_id"]),
                ("family_id", replay["family_id"]),
                ("response_sha256", candidate["sha256"]),
                ("generator_provenance_sha256", candidate["generator"]["provenance_sha256"]),
                ("source_tests_sha256", replay["tests_sha256"]),
                ("test_sha256", replay["tests_sha256"]),
            ):
                _equal(str(row.get(field)) if field == "task_id" else row.get(field), str(expected) if field == "task_id" else expected, f"pair {pair_id}.{side} evidence {field}")
            _equal(classes.get(candidate_id), expected_class, f"pair {pair_id}.{side} execution class")


def _validate_processor_audits(
    rows: Sequence[Mapping[str, Any]],
    replays: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    by_pair = _index_unique(rows, "pair_id", "processor audit")
    _equal(set(by_pair), set(replays), "processor audit/replay membership")
    contract_sha = str(contract["contract_sha256"])
    for pair_id, audit in by_pair.items():
        replay = replays[pair_id]
        _verify_self_hash(audit, "audit_sha256", f"processor audit {pair_id}")
        _equal(audit.get("schema_name"), "day22.qwen35_pair_processor_audit", f"processor audit {pair_id} schema")
        _equal(audit.get("schema_version"), 1, f"processor audit {pair_id} version")
        _equal(audit.get("status"), "pass", f"processor audit {pair_id} status")
        for field, expected in (
            ("task_id", replay["task_id"]),
            ("family_id", replay["family_id"]),
            ("chosen_candidate_id", replay["chosen"]["candidate_id"]),
            ("rejected_candidate_id", replay["rejected"]["candidate_id"]),
            ("chosen_response_sha256", replay["chosen"]["sha256"]),
            ("rejected_response_sha256", replay["rejected"]["sha256"]),
            ("replay_row_sha256", replay["row_sha256"]),
            ("prompt_text_sha256", replay["prompt"]["sha256"]),
            ("processor_contract_sha256", contract_sha),
        ):
            _equal(str(audit.get(field)) if field == "task_id" else audit.get(field), str(expected) if field == "task_id" else expected, f"processor audit {pair_id}.{field}")
        for field in ("processor_sha256", "tokenizer_sha256", "template_sha256"):
            _equal(audit.get(field), contract.get(field), f"processor audit {pair_id}.{field}")
        prompt_hashes: list[str] = []
        starts: list[int] = []
        for side in ("chosen", "rejected"):
            branch = _mapping(audit.get(side), f"processor audit {pair_id}.{side}")
            if "branch_sha256" in branch:
                _verify_self_hash(branch, "branch_sha256", f"processor audit {pair_id}.{side}")
            _equal(branch.get("status"), "pass", f"processor audit {pair_id}.{side} status")
            _equal(branch.get("truncation"), False, f"processor audit {pair_id}.{side} truncation")
            _equal(branch.get("response_only_mask"), True, f"processor audit {pair_id}.{side} mask")
            _equal(branch.get("causal_shift_status"), "pass", f"processor audit {pair_id}.{side} shift")
            span = _list(branch.get("response_span"), f"processor audit {pair_id}.{side} span")
            _require(len(span) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in span), f"processor audit {pair_id}.{side} span is invalid")
            start, end = span
            count = _integer(branch.get("response_token_count"), f"processor audit {pair_id}.{side} response count", minimum=1)
            input_count = _integer(branch.get("input_token_count"), f"processor audit {pair_id}.{side} input count", minimum=1)
            _require(0 < start < end <= input_count and end - start == count, f"processor audit {pair_id}.{side} response span drifted")
            prompt_hashes.append(_sha256(branch.get("prompt_prefix_token_ids_sha256"), f"processor audit {pair_id}.{side} prompt hash"))
            starts.append(start)
        _require(len(set(prompt_hashes)) == 1, f"processor audit {pair_id} prompt prefixes differ")
        _require(len(set(starts)) == 1, f"processor audit {pair_id} response starts differ")
        _equal(audit.get("rendered_prompt_sha256"), prompt_hashes[0], f"processor audit {pair_id} rendered prompt")
    return by_pair


def _expected_formal_pair_ids(replays: Mapping[str, Mapping[str, Any]]) -> list[str]:
    pair_ids = list(replays)
    if len(pair_ids) > MAX_PAIRS:
        pair_ids = sorted(
            pair_ids,
            key=lambda pair_id: (
                text_sha256(f"day22-formal-cap-v1\0{pair_id}"),
                pair_id,
            ),
        )[:MAX_PAIRS]
    return sorted(pair_ids)


def _validate_pairs(
    path: Path,
    *,
    pair_contract: ModuleType,
    replays: Mapping[str, Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    processors: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, Any]]:
    pairs = _load_jsonl(path)
    _require(MIN_PAIRS <= len(pairs) <= MAX_PAIRS, "formal pair count is outside 200..500")
    _require([str(pair.get("pair_id")) for pair in pairs] == sorted(str(pair.get("pair_id")) for pair in pairs), "formal pair rows are not sorted by pair_id")
    try:
        contract_summary = pair_contract.validate_manifest(pairs, mode="final")
    except Exception as error:
        raise FormalS1ValidationError(f"formal pair contract failed: {error}") from error
    by_pair = _index_unique(pairs, "pair_id", "formal pair")
    _equal(sorted(by_pair), _expected_formal_pair_ids(replays), "formal pair/replay capped membership")
    for pair_id, pair in by_pair.items():
        replay = replays[pair_id]
        selection = selections[pair_id]
        _equal(pair.get("dataset_role"), FORMAL_ROLE, f"pair {pair_id} dataset role")
        _equal(pair.get("quality_flags"), [PENDING_BLOCKER], f"pair {pair_id} quality flags")
        for field in ("split", "family_keys", "source", "problem", "prompt", "code_prefix", "entry_point", "tests"):
            _equal(pair.get(field), replay.get(field), f"pair {pair_id}.{field}")
        _equal(str(pair.get("task_id")), str(replay.get("task_id")), f"pair {pair_id}.task_id")
        for side in ("chosen", "rejected"):
            candidate = _mapping(pair.get(side), f"pair {pair_id}.{side}")
            replay_candidate = replay[side]
            for field in ("candidate_id", "origin", "text", "sha256", "ast_sha256", "generator"):
                _equal(candidate.get(field), replay_candidate.get(field), f"pair {pair_id}.{side}.{field}")
            _equal(candidate.get("origin"), ORIGIN, f"pair {pair_id}.{side}.origin")
        creation = _mapping(pair.get("creation"), f"pair {pair_id}.creation")
        _equal(creation.get("method"), "promoted_s1_on_policy_rollout_pair", f"pair {pair_id} creation method")
        _equal(creation.get("version"), selection["labeler"].get("version"), f"pair {pair_id} creation version")
        _equal(creation.get("synthetic"), False, f"pair {pair_id} synthetic")
        _equal(creation.get("promoted_s1_on_policy"), True, f"pair {pair_id} on-policy")
        _equal(creation.get("source_selection_sha256"), selection.get("selection_sha256"), f"pair {pair_id} selection identity")
        _equal(creation.get("source_replay_row_sha256"), replay.get("row_sha256"), f"pair {pair_id} replay identity")
        execution = _mapping(pair.get("execution"), f"pair {pair_id}.execution")
        _equal(execution.get("source_backend"), "e2b", f"pair {pair_id} execution backend")
        source_hashes = _mapping(execution.get("source_candidate_evidence_sha256"), f"pair {pair_id} source evidence")
        for side in ("chosen", "rejected"):
            candidate_id = str(replay[side]["candidate_id"])
            source_evidence = evidence[candidate_id]
            _equal(source_hashes.get(side), source_evidence.get("evidence_sha256"), f"pair {pair_id}.{side} source evidence hash")
            branch = _mapping(execution.get(side), f"pair {pair_id}.execution.{side}")
            _equal(branch.get("candidate_id"), candidate_id, f"pair {pair_id}.{side} candidate")
            _equal(branch.get("origin"), ORIGIN, f"pair {pair_id}.{side} execution origin")
            _equal(branch.get("response_sha256"), source_evidence.get("response_sha256"), f"pair {pair_id}.{side} execution response")
            _equal(branch.get("runs"), source_evidence.get("runs"), f"pair {pair_id}.{side} execution runs")
        _equal(pair.get("processor_audit"), processors[pair_id], f"pair {pair_id} processor audit")
    return pairs, by_pair, contract_summary


def _validate_split_ids(
    path: Path,
    pairs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    split_ids = _load_json(path)
    _verify_self_hash(split_ids, "split_ids_sha256", "formal split IDs")
    _equal(split_ids.get("schema_name"), "day22.formal_s1_family_split_ids", "formal split schema")
    _equal(split_ids.get("schema_version"), 1, "formal split version")
    _equal(split_ids.get("dataset_role"), FORMAL_ROLE, "formal split dataset role")
    family_ids = _mapping(split_ids.get("family_ids"), "formal split family_ids")
    pair_union: set[str] = set()
    family_union: set[str] = set()
    for split in SPLITS:
        actual_pairs = _list(split_ids.get(split), f"formal split {split}")
        actual_families = _list(family_ids.get(split), f"formal family split {split}")
        expected_pairs = sorted(pair_id for pair_id, pair in pairs.items() if pair.get("split") == split)
        expected_families = sorted(str(pair["family_keys"]["problem"]) for pair in pairs.values() if pair.get("split") == split)
        _equal(actual_pairs, expected_pairs, f"formal split {split} membership")
        _equal(actual_families, expected_families, f"formal family split {split} membership")
        _require(not pair_union.intersection(actual_pairs), f"pair ID overlaps into {split}")
        _require(not family_union.intersection(actual_families), f"family ID overlaps into {split}")
        pair_union.update(actual_pairs)
        family_union.update(actual_families)
    _equal(pair_union, set(pairs), "formal split pair union")
    return split_ids


def _validate_review(
    worksheet_path: Path,
    key_path: Path,
    pairs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worksheet = _load_jsonl(worksheet_path)
    key = _load_json(key_path)
    _verify_self_hash(key, "key_sha256", "formal blind review key")
    _equal(key.get("schema_name"), "day22.blind_review_concealed_key", "formal review key schema")
    _equal(key.get("handling"), "keep_separate_from_reviewer_worksheet", "formal review key handling")
    _equal(key.get("unique_pairs"), REVIEW_PAIRS, "formal review unique pairs")
    _equal(key.get("presentations"), REVIEW_PAIRS * 2, "formal review presentation count")
    _equal(len(worksheet), REVIEW_PAIRS * 2, "formal review worksheet rows")
    worksheet_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(worksheet):
        _equal(set(row), REVIEW_KEYS, f"formal review row {index} concealed fields")
        _equal(row.get("schema_name"), "day22.blind_review_item", f"formal review row {index} schema")
        _equal(row.get("allowed_verdicts"), REVIEW_VERDICTS, f"formal review row {index} verdicts")
        _equal(row.get("verdict"), "", f"formal review row {index} verdict")
        _equal(row.get("confidence"), "", f"formal review row {index} confidence")
        _equal(row.get("notes"), "", f"formal review row {index} notes")
        item_id = _text(row.get("review_item_id"), f"formal review row {index}.review_item_id")
        _require(item_id not in worksheet_by_id, f"duplicate formal review_item_id: {item_id}")
        worksheet_by_id[item_id] = row
    key_items = [_mapping(item, "formal review key item") for item in _list(key.get("items"), "formal review key items")]
    key_by_id = _index_unique(key_items, "review_item_id", "formal review key item")
    _equal(set(key_by_id), set(worksheet_by_id), "formal review worksheet/key membership")
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item_id, item in key_by_id.items():
        pair_id = _text(item.get("pair_id"), f"formal review key {item_id}.pair_id")
        _require(pair_id in pairs, f"formal review references unknown pair: {pair_id}")
        presentation = item.get("presentation")
        _require(presentation in {"primary", "swapped"}, f"formal review {item_id} presentation is invalid")
        a_side, b_side = item.get("a_side"), item.get("b_side")
        _require({a_side, b_side} == {"chosen", "rejected"}, f"formal review {item_id} side map is invalid")
        pair = pairs[pair_id]
        row = worksheet_by_id[item_id]
        _equal(row.get("prompt"), pair["prompt"]["text"], f"formal review {item_id} prompt")
        _equal(row.get("response_a"), pair[str(a_side)]["text"], f"formal review {item_id} response A")
        _equal(row.get("response_b"), pair[str(b_side)]["text"], f"formal review {item_id} response B")
        _require(pair_id not in canonical_json(row).decode("utf-8"), f"formal worksheet leaks pair_id {pair_id}")
        _require(str(presentation) not in by_pair[pair_id], f"duplicate {presentation} review for {pair_id}")
        by_pair[pair_id][str(presentation)] = item
    _equal(len(by_pair), REVIEW_PAIRS, "formal review pair coverage")
    for pair_id, presentations in by_pair.items():
        _equal(set(presentations), {"primary", "swapped"}, f"formal review {pair_id} presentations")
        primary, swapped = presentations["primary"], presentations["swapped"]
        _equal(primary.get("a_side"), swapped.get("b_side"), f"formal review {pair_id} swapped A")
        _equal(primary.get("b_side"), swapped.get("a_side"), f"formal review {pair_id} swapped B")
    return worksheet, key


def _cross_split_overlap(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for family_name in FAMILY_KEYS:
        by_split = {
            split: {str(pair["family_keys"][family_name]) for pair in pairs if pair.get("split") == split}
            for split in SPLITS
        }
        overlap = sum(
            len(by_split[left] & by_split[right])
            for left, right in (("train", "dev"), ("train", "heldout"), ("dev", "heldout"))
        )
        _equal(overlap, 0, f"formal {family_name} family cross-split overlap")
        result[family_name] = overlap
    return result


def _validate_machine_audit(
    audit: Mapping[str, Any],
    *,
    selections_count: int,
    eligible_count: int,
    pairs: Sequence[Mapping[str, Any]],
    contract_summary: Mapping[str, Any],
    processor_contract_sha256: str,
    review_key: Mapping[str, Any],
) -> None:
    _verify_self_hash(audit, "audit_sha256", "formal assembly audit")
    _equal(audit.get("schema_name"), "day22.formal_s1_machine_gate_audit", "formal assembly audit schema")
    _equal(audit.get("schema_version"), 1, "formal assembly audit version")
    _equal(audit.get("status"), "pass", "formal assembly audit status")
    _equal(audit.get("dataset_role"), FORMAL_ROLE, "formal assembly audit role")
    _equal(audit.get("machine_ready"), True, "formal assembly machine_ready")
    _equal(audit.get("formal_dpo_ready"), False, "formal assembly formal_dpo_ready")
    _equal(audit.get("pending_blockers"), [PENDING_BLOCKER], "formal assembly blockers")
    _equal(audit.get("contract_summary"), contract_summary, "formal audit contract summary")
    on_policy = sum(pair["creation"].get("promoted_s1_on_policy") is True for pair in pairs)
    synthetic = sum(pair["creation"].get("synthetic") is True for pair in pairs)
    expected_counts = {
        "input_selections": selections_count,
        "eligible_selections": eligible_count,
        "formal_pairs": len(pairs),
        "synthetic_pairs": synthetic,
        "promoted_s1_on_policy_pairs": on_policy,
        "review_unique_pairs": review_key["unique_pairs"],
        "review_presentations": review_key["presentations"],
    }
    _equal(audit.get("counts"), expected_counts, "formal audit counts")
    gates = _mapping(audit.get("machine_gates"), "formal machine gates")
    _equal(set(gates), MACHINE_GATE_NAMES, "formal machine gate set")
    _require(all(_mapping(gate, f"machine gate {name}").get("passed") is True for name, gate in gates.items()), "not all formal machine gates pass")
    expected_gates = {
        "formal_pair_count_200_to_500": {
            "passed": True,
            "observed": len(pairs),
            "minimum": MIN_PAIRS,
            "maximum": MAX_PAIRS,
        },
        "one_pair_per_family": {
            "passed": True,
            "observed_unique_families": len({pair["family_keys"]["problem"] for pair in pairs}),
        },
        "cross_split_family_overlap": {
            "passed": True,
            "observed_overlap": 0,
            "family_dimensions": list(FAMILY_KEYS),
        },
        "promoted_s1_on_policy": {"passed": True, "observed": on_policy},
        "synthetic_pairs": {
            "passed": True,
            "observed": synthetic,
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
            "unique_pairs": REVIEW_PAIRS,
            "presentations": REVIEW_PAIRS * 2,
            "human_verdicts_filled": 0,
        },
    }
    _equal(dict(gates), expected_gates, "formal machine gate evidence")


def _validate_manifest_semantics(
    manifest: Mapping[str, Any],
    *,
    audit: Mapping[str, Any],
    contract_summary: Mapping[str, Any],
    split_ids: Mapping[str, Any],
    worksheet: Sequence[Mapping[str, Any]],
    review_key: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    processor_contract: Mapping[str, Any],
) -> None:
    _equal(manifest.get("schema_name"), "day22.formal_s1_assembly_manifest", "formal manifest schema")
    _equal(manifest.get("schema_version"), 1, "formal manifest version")
    _equal(manifest.get("status"), "machine_gates_passed_human_review_pending", "formal manifest status")
    _equal(manifest.get("dataset_role"), FORMAL_ROLE, "formal manifest dataset role")
    _equal(manifest.get("readiness"), "BLOCKED", "formal manifest readiness")
    _equal(manifest.get("formal_dpo_ready"), False, "formal manifest formal_dpo_ready")
    _equal(manifest.get("machine_ready"), True, "formal manifest machine_ready")
    _equal(manifest.get("formal_dpo_blockers"), [PENDING_BLOCKER], "formal manifest blockers")
    human = _mapping(manifest.get("human_review"), "formal manifest human_review")
    _equal(
        dict(human),
        {
            "status": "pending",
            "required_unique_pairs": REVIEW_PAIRS,
            "completed_unique_pairs": 0,
            "conclusions": None,
        },
        "formal manifest pending human review",
    )
    _equal(manifest.get("machine_gate_audit_sha256"), audit.get("audit_sha256"), "formal manifest audit identity")
    _equal(manifest.get("counts"), audit.get("counts"), "formal manifest counts")
    _equal(manifest.get("contract_summary"), contract_summary, "formal manifest contract summary")
    identities = _mapping(manifest.get("content_identities"), "formal manifest content identities")
    expected = {
        "ordered_pair_hashes_sha256": contract_summary["ordered_pair_hashes_sha256"],
        "split_ids_sha256": split_ids["split_ids_sha256"],
        "worksheet_sha256": object_sha256(list(worksheet)),
        "concealed_review_key_sha256": review_key["key_sha256"],
        "selection_summary_sha256": selection_summary["summary_sha256"],
        "processor_contract_sha256": processor_contract["contract_sha256"],
    }
    _equal(dict(identities), expected, "formal manifest content identities")


def validate_formal_bundle(manifest_path: Path) -> dict[str, Any]:
    """Validate one formal S1 bundle without consulting smoke artifacts."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    _verify_self_hash(manifest, "manifest_sha256", "formal assembly manifest")
    # Reject smoke or any future lifecycle before resolving its referenced files.
    _equal(manifest.get("schema_name"), "day22.formal_s1_assembly_manifest", "formal manifest schema")
    input_paths, output_paths = _manifest_files(manifest, manifest_path)

    selections = _load_jsonl(input_paths["selections"])
    replays = _load_jsonl(input_paths["replay_input"])
    selection_summary = _load_json(input_paths["selection_summary"])
    candidate_evidence_rows = _load_jsonl(input_paths["candidate_evidence"])
    processor_rows = _load_jsonl(input_paths["processor_audit"])
    processor_contract = _load_json(input_paths["processor_contract"])
    processor_contract_sha = _validate_processor_contract(processor_contract)
    eligible, replay_by_pair, allowed_generators, common_sandbox = _validate_selection_inputs(
        selections, replays, selection_summary
    )
    evidence, evidence_classes = _validate_candidate_evidence(
        candidate_evidence_rows,
        summary=selection_summary,
        allowed_generators=allowed_generators,
        common_sandbox=common_sandbox,
    )
    _validate_selected_evidence(replay_by_pair, eligible, evidence, evidence_classes)
    processors = _validate_processor_audits(
        processor_rows, replay_by_pair, processor_contract
    )

    pair_contract = _load_pair_contract()
    pairs, pair_by_id, contract_summary = _validate_pairs(
        output_paths["pairs"],
        pair_contract=pair_contract,
        replays=replay_by_pair,
        selections=eligible,
        evidence=evidence,
        processors=processors,
    )
    split_ids = _validate_split_ids(output_paths["split_ids"], pair_by_id)
    worksheet, review_key = _validate_review(
        output_paths["review_worksheet"], output_paths["review_key"], pair_by_id
    )
    overlaps = _cross_split_overlap(pairs)
    audit = _load_json(output_paths["assembly_audit"])
    _validate_machine_audit(
        audit,
        selections_count=len(selections),
        eligible_count=len(eligible),
        pairs=pairs,
        contract_summary=contract_summary,
        processor_contract_sha256=processor_contract_sha,
        review_key=review_key,
    )
    _validate_manifest_semantics(
        manifest,
        audit=audit,
        contract_summary=contract_summary,
        split_ids=split_ids,
        worksheet=worksheet,
        review_key=review_key,
        selection_summary=selection_summary,
        processor_contract=processor_contract,
    )
    split_counts = Counter(str(pair["split"]) for pair in pairs)
    return {
        "status": "valid_formal_s1_bundle_human_review_pending",
        "machine_gate_status": "PASS",
        "machine_ready": True,
        "formal_readiness": "BLOCKED",
        "formal_dpo_ready": False,
        "formal_dpo_blockers": [PENDING_BLOCKER],
        "records": {
            "formal_pairs": len(pairs),
            "eligible_selections": len(eligible),
            "candidate_e2b_evidence": len(evidence),
            "processor_audits": len(processors),
            "review_unique_pairs": review_key["unique_pairs"],
            "review_presentations": review_key["presentations"],
            "review_completed_unique_pairs": 0,
        },
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "cross_split_family_overlap": overlaps,
        "identities": {
            "manifest_sha256": manifest["manifest_sha256"],
            "machine_gate_audit_sha256": audit["audit_sha256"],
            "processor_contract_sha256": processor_contract_sha,
        },
    }


def validate_ready_bundle(manifest_path: Path) -> dict[str, Any]:
    """Independently validate a human-review-approved wrapper and its parent."""

    manifest_path = manifest_path.expanduser().resolve()
    ready = _load_json(manifest_path)
    _verify_self_hash(ready, "ready_manifest_sha256", "ready manifest")
    _equal(ready.get("schema_name"), READY_MANIFEST_SCHEMA, "ready manifest schema")
    _equal(ready.get("schema_version"), 1, "ready manifest version")
    _equal(ready.get("status"), "formal_dpo_ready", "ready manifest status")
    _equal(ready.get("dataset_role"), "formal_s1_dpo_pairs_human_review_approved", "ready manifest role")
    _equal(ready.get("readiness"), "READY", "ready manifest readiness")
    _equal(ready.get("formal_dpo_ready"), True, "ready manifest formal_dpo_ready")
    _equal(ready.get("formal_dpo_blockers"), [], "ready manifest blockers")

    pending_identity = _mapping(ready.get("pending_manifest"), "ready pending manifest")
    pending_path = _resolve_identity(
        pending_identity,
        manifest_path=manifest_path,
        label="ready pending manifest",
    )
    pending = _load_json(pending_path)
    pending_sha = _verify_self_hash(pending, "manifest_sha256", "pending manifest")
    _equal(
        pending_identity.get("manifest_sha256"), pending_sha, "ready pending manifest identity"
    )
    pending_report = validate_formal_bundle(pending_path)
    _, pending_output_paths = _manifest_files(pending, pending_path)

    audit_identity = _mapping(ready.get("review_audit"), "ready review audit")
    audit_path = _resolve_identity(
        audit_identity,
        manifest_path=manifest_path,
        label="ready review audit",
    )
    audit = _load_json(audit_path)
    audit_sha = _verify_self_hash(audit, "review_audit_sha256", "review audit")
    _equal(audit.get("schema_name"), REVIEW_AUDIT_SCHEMA, "review audit schema")
    _equal(audit.get("schema_version"), 1, "review audit version")
    _equal(audit.get("status"), "pass", "review audit status")
    _equal(audit.get("formal_dpo_ready"), True, "review audit readiness")
    _equal(audit.get("pending_manifest_sha256"), pending_sha, "review audit parent")
    _equal(audit_identity.get("review_audit_sha256"), audit_sha, "ready review audit identity")
    _equal(ready.get("review_audit_sha256"), None, "legacy unbound review audit field")

    gates = _mapping(audit.get("gates"), "review audit gates")
    _equal(
        set(gates),
        {
            "all_100_presentations_completed",
            "position_consistency_at_least_90_percent",
            "verifier_chosen_agreement_at_least_90_percent",
            "remaining_pairs_at_least_200",
            "bound_pair_file_has_no_exclusions",
        },
        "review audit gate set",
    )
    _require(
        all(_mapping(gate, f"review gate {name}").get("passed") is True for name, gate in gates.items()),
        "not all review gates pass",
    )
    counts = _mapping(audit.get("counts"), "review audit counts")
    _equal(counts.get("completed_presentations"), 100, "review completed presentations")
    _equal(counts.get("reviewed_unique_pairs"), 50, "review unique pairs")
    _equal(counts.get("excluded_reviewed_pairs"), 0, "ready review exclusions")
    _equal(
        counts.get("formal_pairs_before_review"),
        pending_report["records"]["formal_pairs"],
        "review pair count before review",
    )
    _equal(
        counts.get("formal_pairs_after_review"),
        pending_report["records"]["formal_pairs"],
        "ready pair membership",
    )
    _equal(ready.get("counts"), counts, "ready manifest counts")
    _equal(ready.get("rates"), audit.get("rates"), "ready manifest rates")
    _equal(ready.get("review_protocol_sha256"), audit.get("protocol", {}).get("file_sha256"), "ready review protocol")

    completed_identity = _mapping(ready.get("completed_worksheet"), "ready completed worksheet")
    completed_path = _resolve_identity(
        completed_identity,
        manifest_path=manifest_path,
        label="ready completed worksheet",
    )
    _equal(
        completed_identity.get("file_sha256"),
        audit.get("completed_worksheet_sha256"),
        "ready completed worksheet identity",
    )
    completed_rows = _load_jsonl(completed_path)
    _recompute_ready_review(
        audit,
        blank_path=pending_output_paths["review_worksheet"],
        key_path=pending_output_paths["review_key"],
        completed_path=completed_path,
        completed_rows=completed_rows,
        total_pairs=pending_report["records"]["formal_pairs"],
    )

    pair_identity = _mapping(ready.get("preference_pairs"), "ready preference pairs")
    ready_pair_path = _resolve_identity(
        pair_identity, manifest_path=pending_path, label="ready preference pairs"
    )
    _equal(ready_pair_path, pending_output_paths["pairs"], "ready preference pair identity")

    return {
        "status": "valid_formal_s1_bundle_ready",
        "machine_gate_status": "PASS",
        "human_review_status": "PASS",
        "machine_ready": True,
        "formal_readiness": "READY",
        "formal_dpo_ready": True,
        "formal_dpo_blockers": [],
        "records": {
            "formal_pairs": counts["formal_pairs_after_review"],
            "review_unique_pairs": counts["reviewed_unique_pairs"],
            "review_presentations": counts["completed_presentations"],
            "review_excluded_pairs": counts["excluded_reviewed_pairs"],
        },
        "rates": copy.deepcopy(dict(audit["rates"])),
        "identities": {
            "ready_manifest_sha256": ready["ready_manifest_sha256"],
            "pending_manifest_sha256": pending_sha,
            "review_audit_sha256": audit_sha,
        },
    }


def _recompute_ready_review(
    audit: Mapping[str, Any],
    *,
    blank_path: Path,
    key_path: Path,
    completed_path: Path,
    completed_rows: Sequence[Mapping[str, Any]],
    total_pairs: int,
) -> None:
    """Recompute every human-review conclusion without trusting the finalizer."""

    blank_rows = _load_jsonl(blank_path)
    key = _load_json(key_path)
    key_sha = _verify_self_hash(key, "key_sha256", "ready concealed review key")
    _equal(key.get("schema_name"), "day22.blind_review_concealed_key", "ready review key schema")
    _equal(len(blank_rows), 100, "ready blank worksheet rows")
    _equal(len(completed_rows), len(blank_rows), "ready completed worksheet rows")
    blank_by_id = _index_unique(blank_rows, "review_item_id", "ready blank review")
    completed_by_id = _index_unique(completed_rows, "review_item_id", "ready completed review")
    _equal(set(completed_by_id), set(blank_by_id), "ready completed review membership")
    for item_id, row in completed_by_id.items():
        blank = blank_by_id[item_id]
        _equal(set(row), set(blank), f"ready completed review {item_id} fields")
        for field in set(blank) - {"verdict", "confidence", "notes"}:
            _equal(row.get(field), blank.get(field), f"ready completed review {item_id}.{field}")
        _require(row.get("verdict") in REVIEW_VERDICTS, f"ready review {item_id} verdict")
        _require(row.get("confidence") in {"low", "medium", "high"}, f"ready review {item_id} confidence")
        notes = row.get("notes")
        _require(isinstance(notes, str) and "\x00" not in notes, f"ready review {item_id} notes")
        if row.get("verdict") in {"tie", "ambiguous", "reject"}:
            _require(bool(notes.strip()), f"ready review {item_id} notes are required")

    key_items = [
        _mapping(item, "ready review key item")
        for item in _list(key.get("items"), "ready review key items")
    ]
    key_by_id = _index_unique(key_items, "review_item_id", "ready review key item")
    _equal(set(key_by_id), set(completed_by_id), "ready review key membership")
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item_id, item in key_by_id.items():
        pair_id = _text(item.get("pair_id"), f"ready review key {item_id}.pair_id")
        presentation = item.get("presentation")
        _require(presentation in {"primary", "swapped"}, f"ready review key {item_id} presentation")
        _require({item.get("a_side"), item.get("b_side")} == {"chosen", "rejected"}, f"ready review key {item_id} sides")
        row = completed_by_id[item_id]
        verdict = str(row["verdict"])
        semantic = item["a_side"] if verdict == "A" else item["b_side"] if verdict == "B" else verdict
        by_pair[pair_id][str(presentation)] = {
            "review_item_id": item_id,
            "display_verdict": verdict,
            "semantic_conclusion": semantic,
            "confidence": row["confidence"],
            "notes": row["notes"],
        }
    _equal(len(by_pair), 50, "ready review unique pair coverage")

    decisions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    position_consistent = 0
    directional = 0
    chosen_agreements = 0
    for pair_id in sorted(by_pair):
        presentations = by_pair[pair_id]
        _equal(set(presentations), {"primary", "swapped"}, f"ready review pair {pair_id} presentations")
        primary, swapped = presentations["primary"], presentations["swapped"]
        semantic = [primary["semantic_conclusion"], swapped["semantic_conclusion"]]
        reasons: list[str] = []
        if any(value in {"tie", "ambiguous", "reject"} for value in semantic):
            reasons.append("non_directional_or_rejected_judgment")
        if semantic[0] != semantic[1]:
            reasons.append("position_swap_disagreement")
        else:
            position_consistent += 1
        if not reasons:
            directional += 1
            if semantic[0] == "chosen":
                chosen_agreements += 1
            else:
                reasons.append("verifier_rejected_preferred")
        reason_counts.update(reasons)
        decision = {
            "pair_id": pair_id,
            "primary": primary,
            "swapped": swapped,
            "position_consistent": semantic[0] == semantic[1],
            "verifier_chosen_agreement": semantic[0] == semantic[1] == "chosen",
            "included": not reasons,
            "exclusion_reasons": reasons,
        }
        decision["decision_sha256"] = object_sha256(decision)
        decisions.append(decision)

    excluded_ids = sorted(item["pair_id"] for item in decisions if not item["included"])
    remaining = total_pairs - len(excluded_ids)
    position_rate = position_consistent / 50
    agreement_rate = chosen_agreements / directional if directional else 0.0
    expected_counts = {
        "completed_presentations": 100,
        "reviewed_unique_pairs": 50,
        "position_consistent_pairs": position_consistent,
        "directionally_eligible_reviewed_pairs": directional,
        "verifier_chosen_agreements": chosen_agreements,
        "excluded_reviewed_pairs": len(excluded_ids),
        "formal_pairs_before_review": total_pairs,
        "formal_pairs_after_review": remaining,
    }
    expected_rates = {
        "position_consistency": position_rate,
        "verifier_chosen_agreement": agreement_rate,
    }
    expected_gates = {
        "all_100_presentations_completed": {"passed": True, "observed": 100, "required": 100},
        "position_consistency_at_least_90_percent": {"passed": position_rate >= 0.90, "observed": position_rate, "required": 0.90},
        "verifier_chosen_agreement_at_least_90_percent": {"passed": agreement_rate >= 0.90, "observed": agreement_rate, "denominator": directional, "required": 0.90},
        "remaining_pairs_at_least_200": {"passed": remaining >= 200, "observed": remaining, "required": 200},
        "bound_pair_file_has_no_exclusions": {"passed": not excluded_ids, "observed_exclusions": len(excluded_ids), "reason": "ready manifest binds the unchanged pending pair file"},
    }
    _equal(audit.get("blank_worksheet_sha256"), file_sha256(blank_path), "review audit blank worksheet")
    _equal(audit.get("completed_worksheet_sha256"), file_sha256(completed_path), "review audit completed worksheet")
    _equal(audit.get("concealed_key_sha256"), key_sha, "review audit concealed key")
    _require(REVIEW_PROTOCOL_PATH.is_file(), "frozen review protocol is missing")
    _equal(
        audit.get("protocol"),
        {"name": "day22.formal_s1_blind_review_protocol.v1", "file_sha256": file_sha256(REVIEW_PROTOCOL_PATH)},
        "review audit protocol",
    )
    _equal(
        audit.get("policy"),
        {"required_presentations": 100, "required_unique_pairs": 50, "minimum_position_consistency": 0.90, "minimum_verifier_chosen_agreement": 0.90, "minimum_remaining_pairs": 200},
        "review audit policy",
    )
    _equal(audit.get("counts"), expected_counts, "review audit counts")
    _equal(audit.get("rates"), expected_rates, "review audit rates")
    _equal(audit.get("exclusion_reason_counts"), dict(sorted(reason_counts.items())), "review audit exclusion reasons")
    _equal(audit.get("excluded_pair_ids"), excluded_ids, "review audit excluded pair IDs")
    _equal(audit.get("gates"), expected_gates, "review audit gates")
    _equal(audit.get("pair_decisions"), decisions, "review audit pair decisions")


def validate_bundle(manifest_path: Path) -> dict[str, Any]:
    """Dispatch by lifecycle schema without weakening either validator."""

    value = _load_json(manifest_path.expanduser().resolve())
    if value.get("schema_name") == READY_MANIFEST_SCHEMA:
        return validate_ready_bundle(manifest_path)
    return validate_formal_bundle(manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="formal_s1_pair_assembler.py output manifest",
    )
    parser.add_argument(
        "--require-formal-ready",
        action="store_true",
        help="return nonzero while the required human blind review is pending",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_bundle(args.manifest)
        if args.require_formal_ready and report.get("formal_dpo_ready") is not True:
            report = dict(report)
            report["status"] = "formal_readiness_required_but_human_review_pending"
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 3
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except FormalS1ValidationError as error:
        print(
            json.dumps(
                {"status": "invalid_formal_s1_bundle", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
