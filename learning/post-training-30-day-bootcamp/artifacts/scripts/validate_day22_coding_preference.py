#!/usr/bin/env python3
"""Validate the complete Day 22 audited-smoke preference artifact bundle.

The default gate proves that the committed smoke bundle is internally
consistent.  It deliberately does not turn a BLOCKED smoke manifest into
formal DPO data.  Pass ``--require-formal-ready`` when a downstream training
job must reject anything that has not passed every formal-readiness gate.
"""

from __future__ import annotations

import argparse
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
REPO_ROOT = BOOTCAMP_ROOT.parents[1]
ARTIFACT_ROOT = BOOTCAMP_ROOT / "artifacts"
DEFAULT_MANIFEST = (
    ARTIFACT_ROOT / "data/day22-qwen35-coding-preference-manifest.json"
)
DEFAULT_PROCESSOR_CONTRACT = (
    ARTIFACT_ROOT / "configs/day22-qwen35-processor-template-contract.json"
)
DEFAULT_TARGET_ENCODING_MODULE = (
    BOOTCAMP_ROOT
    / "day-20-qwen35-balanced-lora-sft/day20_target_encoding_v3.py"
)
PAIR_CONTRACT_MODULE = BOOTCAMP_ROOT / "day-22-preference-data/day22_contract.py"

EXPECTED_INPUTS = {
    "seed",
    "candidates",
    "replay_input",
    "sandbox_evidence",
    "processor_audit",
}
EXPECTED_OUTPUTS = {
    "day22-qwen35-coding-preference-pairs.jsonl",
    "day22-qwen35-coding-preference-split-ids.json",
    "day22-qwen35-coding-preference-blind-review.jsonl",
    "day22-qwen35-coding-preference-blind-review-key.json",
}
SPLITS = ("train", "dev", "heldout")
FAMILY_KEYS = ("problem", "prompt", "test", "source")
PROCESSOR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
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


class Day22ValidationError(ValueError):
    """A content identity, evidence join, or readiness invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise Day22ValidationError("text hash input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Day22ValidationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22ValidationError(message)


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
        raise Day22ValidationError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day22ValidationError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day22ValidationError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from error
        _require(isinstance(row, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def _load_pair_contract() -> ModuleType:
    _require(PAIR_CONTRACT_MODULE.is_file(), "Day 22 pair contract module is missing")
    spec = importlib.util.spec_from_file_location(
        "_day22_contract_for_artifact_validator", PAIR_CONTRACT_MODULE
    )
    _require(spec is not None and spec.loader is not None, "cannot load pair contract")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Day22ValidationError("cannot import Day 22 pair contract") from error
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


def _resolve_recorded_file(
    identity: Mapping[str, Any], label: str, *, fallback_name: str | None = None
) -> Path:
    recorded = Path(_text(identity.get("path"), f"{label}.path")).expanduser()
    candidates = [recorded]
    if not recorded.is_absolute():
        # Committed manifests use paths relative to the bootcamp root so they
        # remain valid after the repository moves to another machine.
        candidates.append(BOOTCAMP_ROOT / recorded)
    if fallback_name:
        candidates.extend(ARTIFACT_ROOT.rglob(fallback_name))
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    _require(bool(existing), f"recorded file is missing: {recorded}")
    expected = _sha256(identity.get("file_sha256"), f"{label}.file_sha256")
    matching = [path for path in sorted(existing) if file_sha256(path) == expected]
    _require(len(matching) == 1, f"{label} has no unique file matching its recorded hash")
    return matching[0]


def _validate_manifest_files(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path]]:
    inputs = _mapping(manifest.get("inputs"), "manifest.inputs")
    outputs = _mapping(manifest.get("output_files"), "manifest.output_files")
    _equal(set(inputs), EXPECTED_INPUTS, "manifest input set")
    _equal(set(outputs), EXPECTED_OUTPUTS, "manifest output set")

    input_paths: dict[str, Path] = {}
    for name in sorted(EXPECTED_INPUTS):
        identity = _mapping(inputs[name], f"manifest.inputs.{name}")
        input_paths[name] = _resolve_recorded_file(
            identity,
            f"manifest.inputs.{name}",
            fallback_name=Path(str(identity.get("path", ""))).name,
        )

    output_paths: dict[str, Path] = {}
    for name in sorted(EXPECTED_OUTPUTS):
        identity = _mapping(outputs[name], f"manifest.output_files.{name}")
        recorded_name = Path(_text(identity.get("path"), f"output {name}.path")).name
        _equal(recorded_name, name, f"manifest output {name} basename")
        output_paths[name] = _resolve_recorded_file(
            identity, f"manifest.output_files.{name}", fallback_name=name
        )

    _require(
        len(set(input_paths.values()) | set(output_paths.values()))
        == len(input_paths) + len(output_paths),
        "manifest records the same file under multiple identities",
    )
    return input_paths, output_paths


def _validate_seed_and_candidates(
    seed_path: Path, candidate_path: Path
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    seed = _load_json(seed_path)
    _verify_self_hash(seed, "manifest_sha256", "seed manifest")
    _equal(
        _mapping(seed.get("header"), "seed header").get("schema_name"),
        "day22.mbpp_seed_manifest",
        "seed schema",
    )
    seed_rows = _list(seed.get("records"), "seed records")
    seeds: dict[int, Mapping[str, Any]] = {}
    for index, row_value in enumerate(seed_rows):
        row = _mapping(row_value, f"seed row {index}")
        _verify_self_hash(row, "record_sha256", f"seed row {index}")
        task_id = _integer(row.get("task_id"), f"seed row {index}.task_id")
        _require(task_id not in seeds, f"duplicate seed task_id: {task_id}")
        seeds[task_id] = row

    candidates: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(_load_jsonl(candidate_path)):
        _verify_self_hash(row, "row_sha256", f"candidate row {index}")
        _equal(row.get("schema_name"), "day22.mbpp_replay_candidates", "candidate schema")
        task_id = _integer(row.get("task_id"), f"candidate row {index}.task_id")
        _require(task_id in seeds, f"candidate task {task_id} is absent from seed")
        _require(task_id not in candidates, f"duplicate candidate task_id: {task_id}")
        candidate_ids: set[str] = set()
        chosen = _mapping(row.get("chosen"), f"candidate task {task_id}.chosen")
        choices = [chosen, *_list(row.get("mutants"), f"candidate task {task_id}.mutants")]
        for choice_index, raw_choice in enumerate(choices):
            choice = _mapping(raw_choice, f"candidate task {task_id} choice {choice_index}")
            _verify_self_hash(
                choice,
                "candidate_sha256",
                f"candidate task {task_id} choice {choice_index}",
            )
            candidate_id = _text(choice.get("candidate_id"), "candidate_id")
            _require(candidate_id not in candidate_ids, f"duplicate candidate_id: {candidate_id}")
            candidate_ids.add(candidate_id)
            response = _text(choice.get("text"), f"candidate {candidate_id}.text")
            _equal(choice.get("sha256"), text_sha256(response), f"candidate {candidate_id} text hash")
        candidates[task_id] = row
    return seeds, candidates


def _candidate_index(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    choices = [row["chosen"], *row["mutants"]]
    return {str(choice["candidate_id"]): choice for choice in choices}


def _validate_replays(
    path: Path, candidates: Mapping[int, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    rows = _load_jsonl(path)
    by_pair: dict[str, Mapping[str, Any]] = {}
    task_ids: set[int] = set()
    for index, row in enumerate(rows):
        _verify_self_hash(row, "row_sha256", f"replay row {index}")
        _equal(row.get("schema_name"), "day22.mbpp_replay_pair", "replay schema")
        pair_id = _text(row.get("pair_id"), f"replay row {index}.pair_id")
        _require(pair_id not in by_pair, f"duplicate replay pair_id: {pair_id}")
        try:
            task_id = int(_text(row.get("task_id"), f"replay {pair_id}.task_id"))
        except ValueError as error:
            raise Day22ValidationError(f"replay {pair_id}.task_id is not an integer") from error
        _equal(str(task_id), row.get("task_id"), f"replay {pair_id} canonical task_id")
        _require(task_id in candidates, f"replay task {task_id} is absent from candidates")
        _require(task_id not in task_ids, f"multiple replay pairs for task {task_id}")
        task_ids.add(task_id)
        candidate_row = candidates[task_id]
        _equal(row.get("family_id"), candidate_row.get("task_family_id"), f"replay {pair_id} family")
        for field in ("split", "code_prefix", "entry_point", "family_keys", "source", "problem", "prompt", "tests"):
            _equal(row.get(field), candidate_row.get(field), f"replay {pair_id} {field}")
        _equal(row.get("tests_sha256"), row["tests"].get("sha256"), f"replay {pair_id} test hash")
        choices = _candidate_index(candidate_row)
        for side in ("chosen", "rejected"):
            replay_choice = _mapping(row.get(side), f"replay {pair_id}.{side}")
            candidate_id = _text(replay_choice.get("candidate_id"), f"replay {pair_id}.{side}.candidate_id")
            _require(candidate_id in choices, f"replay {pair_id} references unknown {candidate_id}")
            source_choice = choices[candidate_id]
            for field in ("text", "sha256", "ast_sha256"):
                _equal(
                    replay_choice.get(field),
                    source_choice.get(field),
                    f"replay {pair_id}.{side}.{field}",
                )
        by_pair[pair_id] = row
    _equal(task_ids, set(candidates), "candidate/replay task membership")
    return rows, by_pair


def _run_status_pattern(evidence: Mapping[str, Any]) -> str:
    execution = _mapping(evidence.get("execution"), "sandbox execution")
    branch_patterns: list[str] = []
    for side in ("chosen", "rejected"):
        branch = _mapping(execution.get(side), f"sandbox execution.{side}")
        runs = _list(branch.get("runs"), f"sandbox execution.{side}.runs")
        branch_patterns.append(f"{side}=" + ",".join(str(run.get("status")) for run in runs))
    return ";".join(branch_patterns)


def _validate_sandbox_rows(
    path: Path, replays: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    rows = _load_jsonl(path)
    by_pair = _index_unique(rows, "pair_id", "sandbox evidence")
    _equal(set(by_pair), set(replays), "sandbox/replay pair membership")
    reasons_by_pair: dict[str, list[str]] = {}
    global_run_ids: set[str] = set()

    for pair_id, evidence in by_pair.items():
        replay = replays[pair_id]
        _verify_self_hash(evidence, "evidence_sha256", f"sandbox evidence {pair_id}")
        _equal(evidence.get("domain"), "day22.mbpp_sandbox_pair_evidence", f"sandbox {pair_id} domain")
        for field, expected in (
            ("task_id", replay["task_id"]),
            ("family_id", replay["family_id"]),
            ("source_tests_sha256", replay["tests_sha256"]),
        ):
            _equal(evidence.get(field), expected, f"sandbox {pair_id}.{field}")
        sandbox_digest = _sha256(evidence.get("sandbox_digest"), f"sandbox {pair_id}.sandbox_digest")
        execution = _mapping(evidence.get("execution"), f"sandbox {pair_id}.execution")
        branch_statuses: dict[str, list[str]] = {}
        local_run_ids: set[str] = set()
        for side in ("chosen", "rejected"):
            branch = _mapping(execution.get(side), f"sandbox {pair_id}.{side}")
            replay_side = _mapping(replay.get(side), f"replay {pair_id}.{side}")
            for field in ("candidate_id", "origin"):
                _equal(branch.get(field), replay_side.get(field), f"sandbox {pair_id}.{side}.{field}")
            _equal(branch.get("response_sha256"), replay_side.get("sha256"), f"sandbox {pair_id}.{side} response")
            runs = _list(branch.get("runs"), f"sandbox {pair_id}.{side}.runs")
            _require(len(runs) == 2, f"sandbox {pair_id}.{side} must contain two runs")
            statuses: list[str] = []
            for attempt, raw_run in enumerate(runs, 1):
                run = _mapping(raw_run, f"sandbox {pair_id}.{side}.run[{attempt}]")
                _verify_self_hash(run, "run_sha256", f"sandbox {pair_id}.{side}.run[{attempt}]")
                for field, expected in (
                    ("pair_id", pair_id),
                    ("task_id", replay["task_id"]),
                    ("family_id", replay["family_id"]),
                    ("side", side),
                    ("attempt", attempt),
                    ("response_sha256", replay_side["sha256"]),
                    ("source_tests_sha256", replay["tests_sha256"]),
                    ("test_sha256", evidence.get("test_sha256")),
                    ("sandbox_digest", sandbox_digest),
                ):
                    _equal(run.get(field), expected, f"sandbox {pair_id}.{side}.run[{attempt}].{field}")
                stdout = run.get("stdout")
                stderr = run.get("stderr")
                _require(isinstance(stdout, str) and isinstance(stderr, str), f"sandbox {pair_id} output is not text")
                _equal(run.get("stdout_sha256"), text_sha256(stdout), f"sandbox {pair_id} stdout hash")
                _equal(run.get("stderr_sha256"), text_sha256(stderr), f"sandbox {pair_id} stderr hash")
                sandbox = _mapping(run.get("sandbox"), f"sandbox {pair_id} run environment")
                _equal(object_sha256(sandbox), sandbox_digest, f"sandbox {pair_id} environment digest")
                run_id = _text(run.get("run_id"), f"sandbox {pair_id} run_id")
                _require(run_id not in local_run_ids, f"duplicate run_id inside pair {pair_id}")
                _require(run_id not in global_run_ids, f"duplicate global run_id: {run_id}")
                local_run_ids.add(run_id)
                global_run_ids.add(run_id)
                statuses.append(_text(run.get("status"), f"sandbox {pair_id} status"))
            branch_statuses[side] = statuses

        eligible = (
            branch_statuses["chosen"] == ["pass", "pass"]
            and branch_statuses["rejected"] == ["wrong_answer", "wrong_answer"]
        )
        _equal(
            evidence.get("pair_execution_status"),
            "eligible" if eligible else "quarantine",
            f"sandbox {pair_id} pair_execution_status",
        )
        reasons: list[str] = []
        if evidence.get("source_tests_sha256") != evidence.get("test_sha256"):
            reasons.append("partial_test_manifest_execution")
        if not eligible:
            reasons.append("execution_not_pass_vs_stable_wrong_answer")
        reasons_by_pair[pair_id] = reasons
    return by_pair, reasons_by_pair


def _execution_status_slices(
    evidence_by_pair: Mapping[str, Mapping[str, Any]], accepted_ids: set[str]
) -> dict[str, Any]:
    slices: dict[str, Any] = {
        "accepted": {
            "pair_patterns": Counter(),
            "chosen_runs": Counter(),
            "rejected_runs": Counter(),
        },
        "quarantine": {
            "pair_patterns": Counter(),
            "chosen_runs": Counter(),
            "rejected_runs": Counter(),
        },
    }
    for pair_id, evidence in evidence_by_pair.items():
        bucket = "accepted" if pair_id in accepted_ids else "quarantine"
        slices[bucket]["pair_patterns"][_run_status_pattern(evidence)] += 1
        execution = evidence["execution"]
        for side in ("chosen", "rejected"):
            slices[bucket][f"{side}_runs"].update(
                str(run["status"]) for run in execution[side]["runs"]
            )
    return {
        bucket: {
            name: dict(sorted(counter.items())) for name, counter in values.items()
        }
        for bucket, values in slices.items()
    }


def _validate_processor_rows(
    path: Path,
    replays: Mapping[str, Mapping[str, Any]],
    processor_contract: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    rows = _load_jsonl(path)
    by_pair = _index_unique(rows, "pair_id", "processor audit")
    _equal(set(by_pair), set(replays), "processor/replay pair membership")
    contract_sha = _sha256(processor_contract.get("contract_sha256"), "processor contract hash")
    contract_fields = (
        "model_key",
        "model_revision",
        "processor_revision",
        "processor_sha256",
        "tokenizer_revision",
        "tokenizer_sha256",
        "template_revision",
        "template_sha256",
    )
    reasons: dict[str, list[str]] = {}
    for pair_id, audit in by_pair.items():
        replay = replays[pair_id]
        _verify_self_hash(audit, "audit_sha256", f"processor audit {pair_id}")
        _equal(audit.get("schema_name"), "day22.qwen35_pair_processor_audit", f"processor {pair_id} schema")
        _equal(audit.get("processor_contract_sha256"), contract_sha, f"processor {pair_id} contract")
        for field in contract_fields:
            _equal(audit.get(field), processor_contract.get(field), f"processor {pair_id}.{field}")
        for field, expected in (
            ("task_id", replay["task_id"]),
            ("family_id", replay["family_id"]),
            ("replay_row_sha256", replay["row_sha256"]),
            ("chosen_candidate_id", replay["chosen"]["candidate_id"]),
            ("rejected_candidate_id", replay["rejected"]["candidate_id"]),
            ("chosen_response_sha256", replay["chosen"]["sha256"]),
            ("rejected_response_sha256", replay["rejected"]["sha256"]),
            ("prompt_text_sha256", replay["prompt"]["sha256"]),
        ):
            _equal(audit.get(field), expected, f"processor {pair_id}.{field}")
        prompt_prefixes: list[str] = []
        response_starts: list[int] = []
        for side in ("chosen", "rejected"):
            branch = _mapping(audit.get(side), f"processor {pair_id}.{side}")
            _verify_self_hash(branch, "branch_sha256", f"processor {pair_id}.{side}")
            _equal(branch.get("response_text_sha256"), replay[side]["sha256"], f"processor {pair_id}.{side} response")
            _equal(branch.get("status"), "pass", f"processor {pair_id}.{side} status")
            _equal(branch.get("truncation"), False, f"processor {pair_id}.{side} truncation")
            _equal(branch.get("response_only_mask"), True, f"processor {pair_id}.{side} mask")
            span = _list(branch.get("response_span"), f"processor {pair_id}.{side}.response_span")
            _require(len(span) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in span), f"processor {pair_id}.{side} response_span is invalid")
            start, end = span
            response_count = _integer(branch.get("response_token_count"), f"processor {pair_id}.{side}.response_token_count", minimum=1)
            input_count = _integer(branch.get("input_token_count"), f"processor {pair_id}.{side}.input_token_count", minimum=1)
            _require(0 < start < end <= input_count and end - start == response_count, f"processor {pair_id}.{side} response span drifted")
            prompt_prefixes.append(_sha256(branch.get("prompt_prefix_token_ids_sha256"), f"processor {pair_id}.{side} prompt prefix"))
            response_starts.append(start)
        _require(len(set(prompt_prefixes)) == 1, f"processor {pair_id} chosen/rejected prompt prefixes differ")
        _require(len(set(response_starts)) == 1, f"processor {pair_id} chosen/rejected boundaries differ")
        _equal(audit.get("rendered_prompt_sha256"), prompt_prefixes[0], f"processor {pair_id} rendered prompt")
        pair_reasons = [] if audit.get("status") == "pass" else ["processor_audit_not_pass"]
        reasons[pair_id] = pair_reasons
    return by_pair, reasons


def _resolve_snapshot_path(
    recorded: str,
    contract_path: Path,
    override: Path | None,
    *,
    model_key: str,
    model_revision: str,
) -> Path:
    if override is not None:
        candidates = [override.expanduser()]
    else:
        recorded_path = Path(recorded).expanduser()
        portable_identity = Path(model_key.replace("/", "--")) / model_revision
        _require(
            recorded_path.parts[-2:] == portable_identity.parts,
            "processor snapshot_path does not match model key/revision",
        )
        checkpoint_sibling = REPO_ROOT.with_name(f"{REPO_ROOT.name}-checkpoints")
        candidates = [recorded_path]
        if not recorded_path.is_absolute():
            candidates.extend(
                [
                    contract_path.parent / recorded_path,
                    BOOTCAMP_ROOT / recorded_path,
                    REPO_ROOT / recorded_path,
                    REPO_ROOT.parent / recorded_path,
                    checkpoint_sibling / recorded_path,
                    checkpoint_sibling / portable_identity,
                ]
            )
    existing = {candidate.resolve() for candidate in candidates if candidate.is_dir()}
    _require(len(existing) == 1, "processor snapshot path is missing or ambiguous")
    return next(iter(existing))


def _validate_processor_contract(
    path: Path, target_module_path: Path, snapshot_override: Path | None
) -> dict[str, Any]:
    contract = _load_json(path)
    _verify_self_hash(contract, "contract_sha256", "processor contract")
    _equal(contract.get("schema_name"), "day22.qwen35_processor_contract", "processor contract schema")
    _equal(contract.get("status"), "frozen", "processor contract status")
    snapshot_files = _mapping(contract.get("snapshot_files"), "processor contract snapshot_files")
    expected_names = set(PROCESSOR_FILES) | set(TOKENIZER_FILES)
    _equal(set(snapshot_files), expected_names, "processor contract snapshot file set")
    snapshot = _resolve_snapshot_path(
        _text(contract.get("snapshot_path"), "processor contract snapshot_path"),
        path,
        snapshot_override,
        model_key=_text(contract.get("model_key"), "processor contract model_key"),
        model_revision=_text(
            contract.get("model_revision"), "processor contract model_revision"
        ),
    )
    for name in sorted(expected_names):
        _equal(Path(name).name, name, f"unsafe snapshot filename {name}")
        identity = _mapping(snapshot_files[name], f"snapshot file {name}")
        file_path = snapshot / name
        _require(file_path.is_file(), f"processor snapshot file is missing: {name}")
        _equal(file_path.stat().st_size, _integer(identity.get("size_bytes"), f"snapshot {name}.size_bytes"), f"snapshot {name} size")
        _equal(file_sha256(file_path), _sha256(identity.get("sha256"), f"snapshot {name}.sha256"), f"snapshot {name} hash")
    processor_identity = {name: snapshot_files[name] for name in PROCESSOR_FILES}
    tokenizer_identity = {name: snapshot_files[name] for name in TOKENIZER_FILES}
    _equal(contract.get("processor_sha256"), object_sha256(processor_identity), "processor composite hash")
    _equal(contract.get("tokenizer_sha256"), object_sha256(tokenizer_identity), "tokenizer composite hash")
    target_module = target_module_path.expanduser().resolve()
    _require(target_module.is_file(), "Day 20 target encoding module is missing")
    _equal(
        file_sha256(target_module),
        _sha256(contract.get("day20_target_encoding_module_sha256"), "target encoding module hash"),
        "target encoding module hash",
    )
    return contract


def _adapted_execution_matches(
    pair: Mapping[str, Any], raw: Mapping[str, Any], pair_id: str
) -> None:
    execution = _mapping(pair.get("execution"), f"pair {pair_id}.execution")
    _equal(execution.get("source_sandbox_evidence_sha256"), raw.get("evidence_sha256"), f"pair {pair_id} raw sandbox reference")
    _equal(execution.get("source_backend"), raw.get("backend"), f"pair {pair_id} sandbox backend")
    for field in ("test_sha256", "sandbox_digest"):
        _equal(execution.get(field), raw.get(field), f"pair {pair_id} execution {field}")
    for side in ("chosen", "rejected"):
        _equal(execution.get(side), raw["execution"].get(side), f"pair {pair_id} execution {side}")


def _validate_pairs_and_splits(
    pair_path: Path,
    split_path: Path,
    manifest: Mapping[str, Any],
    pair_contract: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, Any]]:
    pairs = _load_jsonl(pair_path)
    try:
        summary = pair_contract.validate_manifest(pairs, mode="final")
    except Exception as error:
        raise Day22ValidationError(f"pair contract failed: {error}") from error
    _equal(summary, manifest.get("contract_summary"), "manifest contract summary")
    pair_by_id = _index_unique(pairs, "pair_id", "preference pair")
    for pair_id, pair in pair_by_id.items():
        _equal(pair.get("dataset_role"), manifest.get("dataset_role"), f"pair {pair_id} dataset_role")

    split_ids = _load_json(split_path)
    _verify_self_hash(split_ids, "split_ids_sha256", "split IDs")
    _equal(split_ids.get("schema_name"), "day22.audited_smoke_split_ids", "split ID schema")
    _equal(split_ids.get("dataset_role"), manifest.get("dataset_role"), "split ID dataset role")
    expected_union: set[str] = set()
    for split in SPLITS:
        actual = _list(split_ids.get(split), f"split IDs {split}")
        _require(actual == sorted(set(actual)), f"split IDs {split} are not sorted and unique")
        expected = sorted(pair_id for pair_id, pair in pair_by_id.items() if pair.get("split") == split)
        _equal(actual, expected, f"split IDs {split} exact membership")
        _require(not expected_union.intersection(actual), f"pair IDs overlap into split {split}")
        expected_union.update(actual)
    _equal(expected_union, set(pair_by_id), "split ID union")
    identities = _mapping(manifest.get("content_identities"), "manifest content identities")
    _equal(identities.get("split_ids_sha256"), split_ids.get("split_ids_sha256"), "manifest split ID identity")
    _equal(identities.get("ordered_pair_hashes_sha256"), summary.get("ordered_pair_hashes_sha256"), "manifest ordered pair identity")
    return pairs, pair_by_id, split_ids


def _validate_review(
    worksheet_path: Path,
    key_path: Path,
    pairs: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    worksheet = _load_jsonl(worksheet_path)
    key = _load_json(key_path)
    _verify_self_hash(key, "key_sha256", "blind review key")
    _equal(key.get("schema_name"), "day22.blind_review_concealed_key", "blind review key schema")
    _equal(key.get("handling"), "keep_separate_from_reviewer_worksheet", "blind review key handling")
    identities = _mapping(manifest.get("content_identities"), "manifest content identities")
    _equal(identities.get("worksheet_sha256"), object_sha256(worksheet), "review worksheet content hash")
    _equal(identities.get("concealed_review_key_sha256"), key.get("key_sha256"), "review key content hash")

    worksheet_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(worksheet):
        _equal(set(item), REVIEW_KEYS, f"review worksheet row {index} concealed field set")
        _equal(item.get("schema_name"), "day22.blind_review_item", f"review row {index} schema")
        item_id = _text(item.get("review_item_id"), f"review row {index}.review_item_id")
        _require(item_id not in worksheet_by_id, f"duplicate review_item_id: {item_id}")
        _equal(item.get("allowed_verdicts"), REVIEW_VERDICTS, f"review row {index} verdict choices")
        verdict = item.get("verdict")
        _require(verdict == "" or verdict in REVIEW_VERDICTS, f"review row {index} verdict is invalid")
        _require(isinstance(item.get("confidence"), str), f"review row {index} confidence must be text")
        _require(isinstance(item.get("notes"), str), f"review row {index} notes must be text")
        worksheet_by_id[item_id] = item

    key_items = _list(key.get("items"), "blind review key items")
    _equal(_integer(key.get("presentations"), "blind review presentations"), len(worksheet), "review presentation count")
    key_by_id = _index_unique(
        [_mapping(item, "blind review key item") for item in key_items],
        "review_item_id",
        "blind review key item",
    )
    _equal(set(key_by_id), set(worksheet_by_id), "review worksheet/key item membership")
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    completed_presentations: dict[str, int] = Counter()
    for item_id, key_item in key_by_id.items():
        pair_id = _text(key_item.get("pair_id"), f"review key {item_id}.pair_id")
        _require(pair_id in pairs, f"review key references unknown pair: {pair_id}")
        presentation = key_item.get("presentation")
        _require(presentation in {"primary", "swapped"}, f"review key {item_id} presentation is invalid")
        _require(presentation not in by_pair[pair_id], f"duplicate {presentation} review for {pair_id}")
        a_side = key_item.get("a_side")
        b_side = key_item.get("b_side")
        _require({a_side, b_side} == {"chosen", "rejected"}, f"review key {item_id} side mapping is invalid")
        pair = pairs[pair_id]
        worksheet_item = worksheet_by_id[item_id]
        _equal(worksheet_item.get("prompt"), pair["prompt"]["text"], f"review item {item_id} prompt")
        _equal(worksheet_item.get("response_a"), pair[str(a_side)]["text"], f"review item {item_id} response A")
        _equal(worksheet_item.get("response_b"), pair[str(b_side)]["text"], f"review item {item_id} response B")
        _require(pair_id not in canonical_json(worksheet_item).decode("utf-8"), f"review worksheet leaks pair_id {pair_id}")
        by_pair[pair_id][str(presentation)] = key_item
        if worksheet_item.get("verdict") in REVIEW_VERDICTS:
            completed_presentations[pair_id] += 1

    for pair_id, presentations in by_pair.items():
        _equal(set(presentations), {"primary", "swapped"}, f"review pair {pair_id} presentations")
        primary = presentations["primary"]
        swapped = presentations["swapped"]
        _equal(primary.get("a_side"), swapped.get("b_side"), f"review pair {pair_id} swapped A")
        _equal(primary.get("b_side"), swapped.get("a_side"), f"review pair {pair_id} swapped B")
    unique_pairs = len(by_pair)
    _equal(_integer(key.get("unique_pairs"), "blind review unique_pairs"), unique_pairs, "blind review unique pair count")
    review_gate = _mapping(
        _mapping(manifest.get("readiness_gates"), "readiness gates").get("human_blind_review"),
        "human blind review gate",
    )
    _equal(review_gate.get("worksheet_unique_pairs"), unique_pairs, "review gate worksheet count")
    completed_unique = sum(count == 2 for count in completed_presentations.values())
    _equal(review_gate.get("observed_completed_unique_pairs"), completed_unique, "review gate completed count")
    return {
        "unique_pairs": unique_pairs,
        "presentations": len(worksheet),
        "completed_unique_pairs": completed_unique,
    }


def _cross_split_overlaps(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family_name in FAMILY_KEYS:
        by_split = {
            split: {
                str(pair["family_keys"][family_name])
                for pair in pairs
                if pair.get("split") == split
            }
            for split in SPLITS
        }
        intersections = {
            "train_dev": len(by_split["train"] & by_split["dev"]),
            "train_heldout": len(by_split["train"] & by_split["heldout"]),
            "dev_heldout": len(by_split["dev"] & by_split["heldout"]),
        }
        _require(not any(intersections.values()), f"{family_name} family leaks across splits")
        result[family_name] = intersections
    return result


def _length_bucket(chosen_tokens: int, rejected_tokens: int) -> str:
    relative_delta = abs(chosen_tokens - rejected_tokens) / max(
        chosen_tokens, rejected_tokens
    )
    if relative_delta <= 0.10:
        return "matched"
    if relative_delta <= 0.50:
        return "mid"
    return "large"


def _audit_slices(
    pairs: Sequence[Mapping[str, Any]], execution_statuses: Mapping[str, Any]
) -> dict[str, Any]:
    length_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for pair in pairs:
        audit = pair["processor_audit"]
        chosen_tokens = int(audit["chosen"]["response_token_count"])
        rejected_tokens = int(audit["rejected"]["response_token_count"])
        length_counts[_length_bucket(chosen_tokens, rejected_tokens)] += 1
        test_counts[str(pair["tests"]["count"])] += 1
        variants[str(pair["source"]["variant"])] += 1
        split_counts[str(pair["split"])] += 1
    return {
        "relative_response_token_delta": {
            name: length_counts[name] for name in ("matched", "mid", "large")
        },
        "test_count": dict(sorted(test_counts.items(), key=lambda item: int(item[0]))),
        "source_variant": dict(sorted(variants.items())),
        "split": {split: split_counts[split] for split in SPLITS},
        "execution_statuses": execution_statuses,
    }


def _validate_readiness(manifest: Mapping[str, Any]) -> bool:
    gates = _mapping(manifest.get("readiness_gates"), "manifest readiness_gates")
    _require(bool(gates), "manifest readiness gates are empty")
    passed: list[bool] = []
    for name, raw_gate in gates.items():
        gate = _mapping(raw_gate, f"readiness gate {name}")
        _require(isinstance(gate.get("passed"), bool), f"readiness gate {name}.passed must be boolean")
        passed.append(bool(gate["passed"]))
    formal_ready = all(passed)
    _equal(manifest.get("readiness"), "READY" if formal_ready else "BLOCKED", "manifest readiness")
    _equal(manifest.get("formal_dpo_ready"), formal_ready, "manifest formal_dpo_ready")
    return formal_ready


def validate_artifacts(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    processor_contract_path: Path = DEFAULT_PROCESSOR_CONTRACT,
    target_encoding_module: Path = DEFAULT_TARGET_ENCODING_MODULE,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the artifact bundle and return a concise machine-readable report."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    _verify_self_hash(manifest, "manifest_sha256", "assembly manifest")
    _equal(manifest.get("schema_name"), "day22.audited_smoke_assembly_manifest", "assembly manifest schema")
    _equal(manifest.get("dataset_role"), "audited_smoke_not_formal_dpo_data", "assembly dataset role")
    input_paths, output_paths = _validate_manifest_files(manifest)

    processor_contract_path = processor_contract_path.expanduser().resolve()
    processor_contract = _validate_processor_contract(
        processor_contract_path, target_encoding_module, snapshot_path
    )
    pair_contract = _load_pair_contract()
    pairs, pair_by_id, _ = _validate_pairs_and_splits(
        output_paths["day22-qwen35-coding-preference-pairs.jsonl"],
        output_paths["day22-qwen35-coding-preference-split-ids.json"],
        manifest,
        pair_contract,
    )

    seeds, candidates = _validate_seed_and_candidates(
        input_paths["seed"], input_paths["candidates"]
    )
    replay_rows, replays = _validate_replays(input_paths["replay_input"], candidates)
    sandbox_by_pair, sandbox_reasons = _validate_sandbox_rows(
        input_paths["sandbox_evidence"], replays
    )
    processor_by_pair, processor_reasons = _validate_processor_rows(
        input_paths["processor_audit"], replays, processor_contract
    )

    accepted_ids: set[str] = set()
    quarantine_reasons: Counter[str] = Counter()
    for pair_id in replays:
        reasons = list(sandbox_reasons[pair_id]) + list(processor_reasons[pair_id])
        if reasons:
            quarantine_reasons.update(set(reasons))
        else:
            accepted_ids.add(pair_id)
    _equal(set(pair_by_id), accepted_ids, "accepted pair/evidence eligibility membership")
    for pair_id, pair in pair_by_id.items():
        raw_sandbox = sandbox_by_pair[pair_id]
        raw_processor = processor_by_pair[pair_id]
        _adapted_execution_matches(pair, raw_sandbox, pair_id)
        _equal(pair.get("processor_audit"), raw_processor, f"pair {pair_id} processor evidence")

    execution_statuses = _execution_status_slices(sandbox_by_pair, accepted_ids)

    join_counts = _mapping(manifest.get("join_counts"), "manifest join_counts")
    expected_counts = {
        "seed_records": len(seeds),
        "candidate_records": len(candidates),
        "replay_records": len(replay_rows),
        "sandbox_records": len(sandbox_by_pair),
        "processor_records": len(processor_by_pair),
        "accepted_pairs": len(accepted_ids),
        "quarantined_pairs": len(replays) - len(accepted_ids),
        "quarantine_reason_counts": dict(sorted(quarantine_reasons.items())),
    }
    _equal(dict(join_counts), expected_counts, "manifest join counts")

    review = _validate_review(
        output_paths["day22-qwen35-coding-preference-blind-review.jsonl"],
        output_paths["day22-qwen35-coding-preference-blind-review-key.json"],
        pair_by_id,
        manifest,
    )
    overlaps = _cross_split_overlaps(pairs)
    formal_ready = _validate_readiness(manifest)
    slices = _audit_slices(pairs, execution_statuses)
    return {
        "status": "valid_audited_smoke",
        "formal_dpo_ready": formal_ready,
        "readiness": manifest["readiness"],
        "records": {
            "pairs": len(pairs),
            "quarantine": len(replays) - len(pairs),
            "replay": len(replays),
            "sandbox_evidence": len(sandbox_by_pair),
            "processor_evidence": len(processor_by_pair),
            "review_presentations": review["presentations"],
            "review_completed_unique_pairs": review["completed_unique_pairs"],
        },
        "slice_definitions": {
            "relative_response_token_delta": {
                "matched": "<=0.10",
                "mid": ">0.10 and <=0.50",
                "large": ">0.50",
            }
        },
        "slices": slices,
        "cross_split_overlaps": overlaps,
        "identities": {
            "manifest_sha256": manifest["manifest_sha256"],
            "processor_contract_sha256": processor_contract["contract_sha256"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--processor-contract", type=Path, default=DEFAULT_PROCESSOR_CONTRACT
    )
    parser.add_argument(
        "--target-encoding-module", type=Path, default=DEFAULT_TARGET_ENCODING_MODULE
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        help="override the processor snapshot path recorded by the contract",
    )
    parser.add_argument(
        "--require-formal-ready",
        action="store_true",
        help="return nonzero unless all formal DPO readiness gates pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_artifacts(
            manifest_path=args.manifest,
            processor_contract_path=args.processor_contract,
            target_encoding_module=args.target_encoding_module,
            snapshot_path=args.snapshot_path,
        )
        if args.require_formal_ready and not report["formal_dpo_ready"]:
            report = dict(report)
            report["status"] = "formal_readiness_required_but_blocked"
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 3
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Day22ValidationError as error:
        print(
            json.dumps(
                {"status": "invalid", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
