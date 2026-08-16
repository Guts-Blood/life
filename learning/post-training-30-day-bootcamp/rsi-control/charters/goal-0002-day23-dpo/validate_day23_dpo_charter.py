#!/usr/bin/env python3
"""Build and validate the isolated, append-only Day 23 DPO RSI charter.

This module intentionally does not import or modify the legacy ``rsi_control``
module.  It owns a separate goal ledger whose authority records are written once
with O_EXCL.  The JSON file under ``derived/`` is only a rebuildable cache.

The initializer reads the frozen training split and public metadata manifests.
It must never open the dev/heldout rows (or the all-pairs source containing
them); those identities are inherited from the already sealed data manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


CHARTER_ROOT = Path(__file__).resolve().parent
CONTROL_ROOT = CHARTER_ROOT.parent.parent
BOOTCAMP_ROOT = CONTROL_ROOT.parent

CHARTER_ID = "goal-0002-day23-dpo"
GOAL_ID = "goal-0002-day23-dpo"
CAMPAIGN_ID = "day23-qwen35-dpo-rsi-v0003"
INITIAL_VERSION_ID = "rsi-v0003"
PRIMARY_LEVER = "model.optimization.learning_rate.base"
LEARNING_RATE_GRID = [1e-6, 5e-6]
MAX_SEMANTIC_VERSIONS = 3

CHARTER_SCHEMA = "rsi.day23_dpo_charter"
VERSION_SCHEMA = "rsi.day23_dpo_version_precommit"
EVENT_SCHEMA = "rsi.day23_dpo_event"
LEASE_SCHEMA = "rsi.day23_dpo_access_lease"
DERIVED_SCHEMA = "rsi.day23_dpo_derived_state"
SCHEMA_VERSION = 1

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^rsi-v(?P<ordinal>\d{4})$")
EVENT_FILE_RE = re.compile(
    r"^event-(?P<sequence>\d{6})-(?P<event_type>[a-z0-9-]+)\.json$"
)

DATA_MANIFEST_PATH = Path(
    "artifacts/data/day23-qwen35-coding-dpo-data-manifest.json"
)
CPU_CONTRACT_PATH = Path(
    "artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json"
)
TRAIN_PATH = Path("artifacts/data/day23-qwen35-coding-dpo-train.jsonl")
DEV_PATH = Path("artifacts/data/day23-qwen35-coding-dpo-dev.jsonl")
HELDOUT_PATH = Path("artifacts/data/day23-qwen35-coding-dpo-heldout.jsonl")
DAY22_ALL_PAIRS_PATH = Path(
    "artifacts/data/day22-qwen35-formal-s1-preference-pairs.jsonl"
)
DAY22_SPLIT_IDS_PATH = Path(
    "artifacts/data/day22-qwen35-experimental-ai-assisted-split-ids.json"
)
PROCESSOR_ROWS_PATH = Path(
    "artifacts/eval/day23-qwen35-coding-dpo-processor-audit.jsonl"
)
REPORT_PATH = Path("artifacts/reports/day23-qwen35-coding-dpo-smoke.md")

# Opening any of these before the corresponding lease would expose evaluation
# membership or content.  Metadata manifests may be read; these files may not.
PROTECTED_PRELEASE_PATHS = frozenset(
    {
        DEV_PATH,
        HELDOUT_PATH,
        DAY22_ALL_PAIRS_PATH,
        DAY22_SPLIT_IDS_PATH,
        PROCESSOR_ROWS_PATH,
    }
)

SEARCH_SEED = 20260819
REFIT_SEED = 20260820
CHECKPOINT_STEPS = [15, 30]

LOCAL_JSON_BINDINGS: tuple[tuple[Path, str | None], ...] = (
    (Path("rsi-control/goal.json"), None),
    (Path("rsi-control/state.json"), None),
    (Path("rsi-control/metrics.json"), None),
    (Path("rsi-control/taxonomy.json"), None),
    (Path("rsi-control/failure-taxonomy.json"), None),
    (Path("rsi-control/versions/rsi-v0001/version.json"), None),
    (Path("rsi-control/versions/rsi-v0001/artifacts.json"), None),
    (Path("rsi-control/versions/rsi-v0001/metrics.json"), None),
    (Path("rsi-control/versions/rsi-v0002/version.json"), None),
    (Path("rsi-control/versions/rsi-v0002/artifacts.json"), None),
    (Path("rsi-control/versions/rsi-v0002/metrics.json"), None),
    (DATA_MANIFEST_PATH, "manifest_sha256"),
    (CPU_CONTRACT_PATH, "contract_sha256"),
    (
        Path("artifacts/data/day22-qwen35-experimental-ai-assisted-manifest.json"),
        "manifest_sha256",
    ),
    (
        Path("artifacts/checkpoints/day21-qwen35-s1-promotion-manifest.json"),
        "promotion_manifest_sha256",
    ),
    (
        Path("artifacts/checkpoints/day21-qwen35-s1-merged-export-manifest.json"),
        "manifest_sha256",
    ),
    (Path("artifacts/checkpoints/day21-qwen35-s1-downstream-key.json"), None),
    (
        Path("artifacts/eval/day23-qwen35-coding-dpo-processor-audit-summary.json"),
        "summary_sha256",
    ),
    (
        Path("artifacts/eval/day23-qwen35-coding-dpo-argument-audit.json"),
        "audit_sha256",
    ),
)

LOCAL_TEXT_BINDINGS: tuple[Path, ...] = (
    REPORT_PATH,
    Path("day-23-dpo-theory-smoke/README.md"),
    Path("day-23-dpo-theory-smoke/bind_day23_qwen35_gpu.py"),
    Path("day-23-dpo-theory-smoke/run_day23_qwen35_gpu_stage.py"),
    Path("day-23-dpo-theory-smoke/eval_day23_qwen35_preferences.py"),
    Path("day-23-dpo-theory-smoke/day23_contract.py"),
    Path("day-23-dpo-theory-smoke/day23_rlhf_template.py"),
    Path("day-23-dpo-theory-smoke/day23_ms_swift_plugin.py"),
    Path("day-23-dpo-theory-smoke/build_day23_rsi_v0003.py"),
    Path("day-23-dpo-theory-smoke/run_day23_rsi_candidate.py"),
    Path("day-23-dpo-theory-smoke/eval_day23_rsi_candidate.py"),
    Path("rsi-control/charters/goal-0002-day23-dpo/validate_day23_dpo_charter.py"),
)


class CharterError(ValueError):
    """The charter or one of its authority records is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in document:
        raise CharterError(f"cannot seal document that already has {field}")
    result = dict(document)
    result[field] = object_sha256(result)
    return result


def validate_seal(document: Mapping[str, Any], field: str, label: str) -> None:
    claimed = document.get(field)
    if not isinstance(claimed, str) or SHA_RE.fullmatch(claimed) is None:
        raise CharterError(f"{label}.{field} must be a lowercase SHA-256")
    payload = dict(document)
    payload.pop(field)
    if object_sha256(payload) != claimed:
        raise CharterError(f"{label}.{field} does not match canonical content")


def utc_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CharterError("created_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise CharterError("created_at_utc is invalid") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise CharterError("created_at_utc must be UTC")
    return value


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _relative_to_bootcamp(path: Path, bootcamp_root: Path) -> Path:
    try:
        return path.resolve().relative_to(bootcamp_root.resolve())
    except ValueError as error:
        raise CharterError(f"path is outside bootcamp root: {path}") from error


def _read_bytes_unchecked(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CharterError(f"cannot read required input: {path}") from error


def read_bytes(path: Path, *, bootcamp_root: Path) -> bytes:
    relative = _relative_to_bootcamp(path, bootcamp_root)
    if relative in PROTECTED_PRELEASE_PATHS:
        raise CharterError(
            f"pre-lease access to protected evaluation content is forbidden: {relative}"
        )
    return _read_bytes_unchecked(path)


def load_json(path: Path, *, bootcamp_root: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(read_bytes(path, bootcamp_root=bootcamp_root))
    except json.JSONDecodeError as error:
        raise CharterError(f"invalid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise CharterError(f"JSON authority must be an object: {path}")
    return value


def local_path(relative: Path, bootcamp_root: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise CharterError(f"non-canonical bootcamp-relative path: {relative}")
    return bootcamp_root / relative


def bind_json(
    relative: Path,
    self_hash_field: str | None,
    *,
    bootcamp_root: Path,
) -> dict[str, Any]:
    path = local_path(relative, bootcamp_root)
    raw = read_bytes(path, bootcamp_root=bootcamp_root)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CharterError(f"invalid JSON binding: {relative}") from error
    if not isinstance(document, Mapping):
        raise CharterError(f"JSON binding must be an object: {relative}")
    binding: dict[str, Any] = {
        "path": relative.as_posix(),
        "availability": "local_verified",
        "file_sha256": sha256_bytes(raw),
        "schema_name": document.get("schema_name"),
        "schema_version": document.get("schema_version"),
    }
    if self_hash_field is not None:
        validate_seal(document, self_hash_field, relative.as_posix())
        binding["content_hash_field"] = self_hash_field
        binding["content_sha256"] = document[self_hash_field]
    return binding


def bind_text(relative: Path, *, bootcamp_root: Path) -> dict[str, Any]:
    raw = read_bytes(local_path(relative, bootcamp_root), bootcamp_root=bootcamp_root)
    return {
        "path": relative.as_posix(),
        "availability": "local_verified",
        "file_sha256": sha256_bytes(raw),
    }


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CharterError(f"{label} must be an object")
    return value


def _expect_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise CharterError(f"{label} must be a lowercase SHA-256")
    return value


def _extract_reported_failure_chain(report: str) -> dict[str, Any]:
    patterns = {
        "attempt1_failure": (
            r"Attempt 1（保留，不覆盖）：\s*\n\s*- path：`(?P<path>[^`]+)`"
            r"\s*\n\s*- receipt content SHA-256：`(?P<content>[0-9a-f]{64})`"
            r"\s*\n\s*- file SHA-256：`(?P<file>[0-9a-f]{64})`"
        ),
        "attempt2_claim": (
            r"Attempt 2 one-shot claim：\s*\n\s*- path：`(?P<path>[^`]+)`"
            r"\s*\n\s*- claim content SHA-256：`(?P<content>[0-9a-f]{64})`"
            r"\s*\n\s*- file SHA-256：`(?P<file>[0-9a-f]{64})`"
        ),
        "attempt2_failure": (
            r"Attempt 2 failure：\s*\n\s*- path：`(?P<path>[^`]+)`"
            r"\s*\n\s*- receipt content SHA-256：`(?P<content>[0-9a-f]{64})`"
            r"\s*\n\s*- file SHA-256：`(?P<file>[0-9a-f]{64})`"
        ),
    }
    result: dict[str, Any] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, report)
        if match is None:
            raise CharterError(f"closeout report no longer binds {name}")
        path = match.group("path")
        if not path.startswith("/root/autodl-tmp/runs/day23-qwen35-dpo-"):
            raise CharterError(f"unexpected remote evidence path for {name}")
        result[name] = {
            "path": path,
            "content_sha256": match.group("content"),
            "file_sha256": match.group("file"),
            "availability": "remote_reported_not_reverified",
        }
    if "CLOSED_NO_CANDIDATE" not in report or "禁止第三次" not in report:
        raise CharterError("closeout report lost the terminal failure boundary")
    return result


def _validate_and_partition_train(
    manifest: Mapping[str, Any],
    cpu_contract: Mapping[str, Any],
    *,
    bootcamp_root: Path,
) -> dict[str, Any]:
    raw = read_bytes(local_path(TRAIN_PATH, bootcamp_root), bootcamp_root=bootcamp_root)
    outputs = _expect_mapping(manifest.get("outputs"), "data manifest outputs")
    train_manifest = _expect_mapping(outputs.get("train"), "data manifest train")
    if train_manifest.get("path") != TRAIN_PATH.as_posix():
        raise CharterError("train path drifted from the sealed data manifest")
    if sha256_bytes(raw) != train_manifest.get("file_sha256"):
        raise CharterError("train bytes drifted from the sealed data manifest")

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CharterError(f"invalid train JSONL row {line_number}") from error
        if not isinstance(row, Mapping):
            raise CharterError(f"train row {line_number} must be an object")
        if row.get("schema_name") != "day23.ms_swift_dpo_row":
            raise CharterError(f"train row {line_number} schema drifted")
        validate_seal(row, "row_sha256", f"train row {line_number}")
        rows.append(row)

    if len(rows) != 154 or train_manifest.get("records") != 154:
        raise CharterError("the frozen train split must contain exactly 154 rows")
    pair_ids = [row.get("pair_id") for row in rows]
    if any(not isinstance(item, str) or not item for item in pair_ids):
        raise CharterError("every train row must have a pair_id")
    if len(set(pair_ids)) != 154:
        raise CharterError("train pair IDs must be unique")

    stages = _expect_mapping(cpu_contract.get("stages"), "CPU contract stages")
    mechanism = _expect_mapping(stages.get("mechanism_5step"), "mechanism stage")
    old4 = mechanism.get("mechanism_pair_ids")
    if not isinstance(old4, list) or len(old4) != 4 or pair_ids[:4] != old4:
        raise CharterError("the old4 diagnostic identities no longer match train order")
    old4_set = set(old4)
    selection_salt = f"{CAMPAIGN_ID}|"
    eligible = [pair_id for pair_id in pair_ids if pair_id not in old4_set]
    ordered = sorted(
        eligible,
        key=lambda pair_id: (
            sha256_bytes((selection_salt + pair_id).encode("utf-8")),
            pair_id,
        ),
    )
    selected = set(ordered[:30])
    if len(selected) != 30 or old4_set & selected:
        raise CharterError("hash-only search selection violated count or old4 exclusion")

    optimizer124 = [pair_id for pair_id in pair_ids if pair_id not in selected]
    search30 = [pair_id for pair_id in pair_ids if pair_id in selected]
    fit120 = [pair_id for pair_id in optimizer124 if pair_id not in old4_set]
    if len(optimizer124) != 124 or len(fit120) != 120 or len(search30) != 30:
        raise CharterError("the frozen old4+fit120/search30 partition is incomplete")

    return {
        "partition_rule": (
            "old4 are excluded from search; rank the other 150 by "
            "sha256(campaign_id| + pair_id), then pair_id; first 30 are search"
        ),
        "hash_selection": {
            "authority": "sealed_train_pair_ids_only",
            "selection_salt": selection_salt,
            "ranking": "sha256(selection_salt + pair_id), then pair_id",
            "eligible_records": 150,
            "search_records": 30,
            "processor_audit_opened": False,
            "dev_rows_opened": False,
            "heldout_rows_opened": False,
        },
        "train_file": {
            "path": TRAIN_PATH.as_posix(),
            "file_sha256": train_manifest["file_sha256"],
            "records": 154,
        },
        "old4": {
            "role": "optimizer_train_and_descriptive_diagnostic_only",
            "ordered_pair_ids": old4,
            "ordered_pair_ids_sha256": object_sha256(old4),
            "records": 4,
        },
        "fit120": {
            "role": "optimizer_train",
            "ordered_pair_ids": fit120,
            "ordered_pair_ids_sha256": object_sha256(fit120),
            "records": 120,
        },
        "optimizer124": {
            "role": "lr_grid_optimizer_train",
            "ordered_pair_ids": optimizer124,
            "ordered_pair_ids_sha256": object_sha256(optimizer124),
            "records": 124,
        },
        "search30": {
            "role": "frozen_train_split_search_evaluation_no_optimizer",
            "ordered_pair_ids": search30,
            "ordered_pair_ids_sha256": object_sha256(search30),
            "records": 30,
        },
        "full154_refit": {
            "role": "unique_finalist_independent_seed_optimizer_train",
            "ordered_pair_ids_sha256": object_sha256(pair_ids),
            "records": 154,
        },
        "b24_forward_capacity_audit": {
            "role": "forward_only_no_optimizer_no_candidate",
            "ordered_pair_ids_sha256": object_sha256(optimizer124[:48]),
            "records": 48,
        },
    }


def _manifest_declared_inputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    inputs = _expect_mapping(manifest.get("inputs"), "data manifest inputs")
    outputs = _expect_mapping(manifest.get("outputs"), "data manifest outputs")
    declared: dict[str, Any] = {}
    for name, source_key in (
        ("day22_pairs", "day22_pairs"),
        ("day22_split_ids", "day22_split_ids"),
    ):
        source = dict(_expect_mapping(inputs.get(source_key), source_key))
        for key, value in source.items():
            if key.endswith("sha256"):
                _expect_sha(value, f"{source_key}.{key}")
        source["availability"] = "manifest_attested_not_opened"
        declared[name] = source
    for name in ("dev", "heldout"):
        source = dict(_expect_mapping(outputs.get(name), f"outputs.{name}"))
        if source.get("path") != (
            DEV_PATH.as_posix() if name == "dev" else HELDOUT_PATH.as_posix()
        ):
            raise CharterError(f"{name} path drifted")
        expected_count = 17 if name == "dev" else 29
        if source.get("records") != expected_count:
            raise CharterError(f"{name} count drifted")
        for key, value in source.items():
            if key.endswith("sha256"):
                _expect_sha(value, f"outputs.{name}.{key}")
        source["availability"] = "manifest_attested_not_opened"
        declared[name] = source
    return declared


def _taxonomy_levers(*, bootcamp_root: Path) -> set[str]:
    taxonomy = load_json(
        local_path(Path("rsi-control/taxonomy.json"), bootcamp_root),
        bootcamp_root=bootcamp_root,
    )
    if taxonomy.get("effective_from_version") != "rsi-v0003":
        raise CharterError("taxonomy activation boundary drifted")
    rows = taxonomy.get("detailed_levers")
    if not isinstance(rows, list):
        raise CharterError("taxonomy detailed_levers must be a list")
    return {
        item["id"]
        for item in rows
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }


def _failure_taxonomy_ids(*, bootcamp_root: Path) -> set[str]:
    taxonomy = load_json(
        local_path(Path("rsi-control/failure-taxonomy.json"), bootcamp_root),
        bootcamp_root=bootcamp_root,
    )
    result: set[str] = set()
    for section in ("failure_modes", "symptoms"):
        rows = taxonomy.get(section)
        if not isinstance(rows, list):
            raise CharterError(f"failure taxonomy {section} must be a list")
        for item in rows:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                result.add(item["id"])
    return result


def _runtime_identity_from_sources(
    manifest: Mapping[str, Any], report: str
) -> dict[str, Any]:
    parent = _expect_mapping(manifest.get("parent"), "data manifest parent")
    package_pattern = (
        r"learner runtime：Python `(?P<python>[^`]+)`、torch `(?P<torch>[^`]+)`、"
        r"Transformers `(?P<transformers>[^`]+)`、PEFT `(?P<peft>[^`]+)`、TRL `(?P<trl>[^`]+)`"
    )
    match = re.search(package_pattern, report)
    if match is None:
        raise CharterError("closeout report runtime package identity drifted")
    gpu_match = re.search(r"- GPU：2× (?P<gpu>[^\n]+)", report)
    if gpu_match is None:
        raise CharterError("closeout report GPU identity drifted")
    return {
        "ms_swift_commit": parent.get("ms_swift_commit"),
        "packages": match.groupdict(),
        "prior_proven_gpu_model": gpu_match.group("gpu").strip(),
        "required_world_size": 2,
        "runtime_must_be_rebound_per_run": True,
    }


def _lease_scope(
    manifest: Mapping[str, Any], declared: Mapping[str, Any]
) -> dict[str, Any]:
    inputs = _expect_mapping(manifest.get("inputs"), "data manifest inputs")
    day22 = _expect_mapping(
        inputs.get("day22_experimental_manifest"), "day22 experimental manifest"
    )
    dataset_sha = _expect_sha(day22.get("manifest_sha256"), "day22 manifest hash")
    scope_id = f"day22-experimental-ai-assisted-{dataset_sha}"
    return {
        "scope_id": scope_id,
        "ledger_root": f"access-ledger/{scope_id}",
        "dev": {
            "claim_path": f"access-ledger/{scope_id}/dev-claim.json",
            "identity": declared["dev"],
            "maximum_global_claims": 1,
            "prerequisite_state": "candidate_pool_frozen",
            "raw_content_access_before_claim": False,
        },
        "heldout": {
            "claim_path": f"access-ledger/{scope_id}/heldout-claim.json",
            "identity": declared["heldout"],
            "current_authorized_claims": 0,
            "maximum_global_claims_after_append_only_authorization": 1,
            "authorization": "blocked_until_guardrail_and_heldout_gate_amendment",
            "prerequisite_state": "guardrails_passed",
            "raw_content_access_before_claim": False,
        },
    }


def build_charter(
    *,
    created_at_utc: str,
    bootcamp_root: Path = BOOTCAMP_ROOT,
) -> dict[str, Any]:
    created_at_utc = utc_timestamp(created_at_utc)
    manifest = load_json(
        local_path(DATA_MANIFEST_PATH, bootcamp_root), bootcamp_root=bootcamp_root
    )
    cpu_contract = load_json(
        local_path(CPU_CONTRACT_PATH, bootcamp_root), bootcamp_root=bootcamp_root
    )
    if manifest.get("schema_name") != "day23.qwen35_coding_dpo_data_manifest":
        raise CharterError("Day23 data manifest schema drifted")
    if cpu_contract.get("schema_name") != "day23.qwen35_coding_dpo_cpu_run_contract":
        raise CharterError("Day23 CPU contract schema drifted")
    validate_seal(manifest, "manifest_sha256", "Day23 data manifest")
    validate_seal(cpu_contract, "contract_sha256", "Day23 CPU contract")

    report_bytes = read_bytes(
        local_path(REPORT_PATH, bootcamp_root), bootcamp_root=bootcamp_root
    )
    report = report_bytes.decode("utf-8")
    failure_chain = _extract_reported_failure_chain(report)
    partition = _validate_and_partition_train(
        manifest, cpu_contract, bootcamp_root=bootcamp_root
    )
    declared = _manifest_declared_inputs(manifest)

    if PRIMARY_LEVER not in _taxonomy_levers(bootcamp_root=bootcamp_root):
        raise CharterError(f"primary lever is absent from the active taxonomy: {PRIMARY_LEVER}")
    required_diagnosis_ids = {
        "symptom.gate.hard_failure",
        "symptom.capability.no_lift",
        "failure.unknown.localization.unlocalized",
        "failure.training.distributed.global_batch_drift",
        "failure.training.distributed.global_order_drift",
        "failure.training.distributed.sampler_padding",
        "failure.training.distributed.sampler_duplication",
        "failure.runtime.distributed.topology_visibility",
        "failure.training.dynamics.nan_inf",
        "failure.model.initialization.base_checkpoint_mismatch",
    }
    missing_diagnosis = required_diagnosis_ids - _failure_taxonomy_ids(
        bootcamp_root=bootcamp_root
    )
    if missing_diagnosis:
        raise CharterError(f"diagnosis taxonomy IDs disappeared: {sorted(missing_diagnosis)}")

    local_bindings = [
        bind_json(path, field, bootcamp_root=bootcamp_root)
        for path, field in LOCAL_JSON_BINDINGS
    ]
    local_bindings.extend(
        bind_text(path, bootcamp_root=bootcamp_root) for path in LOCAL_TEXT_BINDINGS
    )
    local_bindings.sort(key=lambda item: item["path"])

    parent = _expect_mapping(manifest.get("parent"), "data manifest parent")
    optimizer124 = partition["optimizer124"]
    full154 = partition["full154_refit"]
    document = {
        "schema_name": CHARTER_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "charter_id": CHARTER_ID,
        "goal_id": GOAL_ID,
        "created_at_utc": created_at_utc,
        "status": "immutable_precommit",
        "parent_lineage": {
            "legacy_goal_id": "goal-0001",
            "legacy_terminal_version": "rsi-v0002",
            "legacy_goal_status": "achieved",
            "fresh_parent_role": "S1",
            "fresh_parent_checkpoint_id": parent.get("checkpoint_id"),
            "fresh_parent_downstream_key": parent.get("downstream_key"),
            "fresh_parent_merged_export_path": parent.get("remote_merged_export_path"),
            "fresh_lora_required": True,
            "resume_from_checkpoint": None,
        },
        "claim_boundary": {
            "objective": (
                "Produce at most one experimental Day23 DPO candidate that passes the "
                "frozen preference and coding guardrails under this charter."
            ),
            "success_means": "qualification_on_this_fixed_experimental_suite_only",
            "formal_human_reviewed": False,
            "general_improvement_claim": False,
            "deployment_ready_claim": False,
        },
        "legacy_day23_terminal_chain": {
            "status": "closed_no_candidate",
            "source_report": {
                "path": REPORT_PATH.as_posix(),
                "file_sha256": sha256_bytes(report_bytes),
            },
            "remote_evidence": failure_chain,
            "permanent_prohibitions": {
                "third_attempt_in_legacy_charter": True,
                "resume_from_any_legacy_failure_checkpoint": True,
                "use_any_legacy_failure_checkpoint_as_candidate": True,
                "rewrite_or_supersede_legacy_failure_receipts": True,
            },
            "allowed_use": "negative_provenance_and_diagnosis_only",
        },
        "artifact_bindings": {
            "local_verified": local_bindings,
            "manifest_attested_not_opened": declared,
        },
        "dataset_campaign": partition,
        "runtime_policy": {
            "identity": _runtime_identity_from_sources(manifest, report),
            "optimizer_topology": {
                "world_size": 2,
                "parallelism": "DDP",
                "per_device_pair_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "nominal_global_pair_batch_size": 32,
                "optimizer124_rank_rows": 62,
                "optimizer124_final_global_batch": 28,
                "full154_rank_rows": 77,
                "full154_final_global_batch": 26,
                "sampler_padding": False,
                "sampler_duplication": False,
                "fixed_operational_constraint_not_candidate_axis": True,
            },
            "capacity_audit": {
                "world_size": 2,
                "per_device_pair_batch_size": 24,
                "gradient_accumulation_steps": 1,
                "forward_only": True,
                "optimizer_steps": 0,
                "candidate_eligible": False,
                "required": False,
                "blocking_gate": False,
            },
        },
        "iteration_budget": {
            "maximum_semantic_versions": MAX_SEMANTIC_VERSIONS,
            "allowed_version_ids": ["rsi-v0003", "rsi-v0004", "rsi-v0005"],
            "versions_after_dev_claim": 0,
            "v0003_lr_grid_train_runs": 2,
            "v0003_full154_refits": 1,
            "v0003_training_trajectories": 3,
            "v0003_maximum_optimizer_steps": 90,
            "v0003_search_checkpoint_evaluations": 4,
            "dev_global_claims": 1,
            "heldout_global_claims_currently_authorized": 0,
            "heldout_global_claims_future_cap": 1,
            "scientific_retry_after_result": 0,
            "infra_retry_rule": (
                "same version/run/seed/data/config/runtime hashes only; no scientific "
                "result and no dev or heldout claim may exist"
            ),
        },
        "access_leases": _lease_scope(manifest, declared),
        "state_machine": {
            "initial_state": "uninitialized",
            "terminal_states": ["qualified", "terminal_no_go"],
            "current_authorized_path": [
                "charter_frozen",
                "version_precommitted",
            ],
            "target_path_after_strict_runtime_and_guardrail_amendments": [
                "charter_frozen",
                "version_precommitted",
                "lr_grid_search_completed",
                "finalist_frozen",
                "full154_refit_completed",
                "candidate_pool_frozen",
                "dev_lease_claimed",
                "dev_gate_passed",
                "guardrails_passed",
                "heldout_lease_claimed",
                "qualified",
            ],
            "no_backward_edge_after": "dev_lease_claimed",
            "failure_is_valid_terminal_output": True,
            "runtime_event_append_authorized": False,
            "runtime_event_authorization_requirement": (
                "a separately reviewed append-only amendment must bind strict run-contract, "
                "receipt, gate-recomputation, event, and lease validators before event 3"
            ),
        },
        "execution_plane": {
            "gpu_campaign_build_and_bind_allowed_after_precommit": True,
            "optimizer_execution_allowed_after_strict_gpu_campaign_bind": True,
            "gpu_campaign_run_and_evaluation_receipts_are_independent_authority": True,
            "charter_event_ingestion_before_runtime_amendment": False,
            "promotion_before_runtime_amendment": False,
            "required_bound_producers": [
                "day-23-dpo-theory-smoke/build_day23_rsi_v0003.py",
                "day-23-dpo-theory-smoke/run_day23_rsi_candidate.py",
                "day-23-dpo-theory-smoke/eval_day23_rsi_candidate.py",
            ],
        },
        "global_hard_gates": {
            "all_optimizer_runs": [
                "fresh_from_exact_promoted_S1",
                "world_size_2_rank_inventory_0_1",
                "no_sampler_padding_or_duplication",
                "finite_loss_logps_rewards_gradients",
                "exact_LoRA_trainable_and_optimizer_inventory",
                "frozen_parent_and_disabled_adapter_reference_unchanged",
                "minimum_free_fraction_at_least_0.30",
                "checkpoint_manifest_and_fresh_reload_exact",
            ],
            "old4_descriptive_diagnostic": {
                "promotion_gate": False,
                "reason": "the four rows were already inspected in two legacy attempts",
                "optimizer_input_allowed": True,
            },
            "search30_gate": {
                "finite_records": {"operator": "==", "threshold": 30},
                "positive_margin_pairs": {"operator": ">=", "threshold": 20, "denominator": 30},
                "mean_reward_margin": {"operator": ">", "threshold": 0.0},
                "length_matched_margin": {"operator": ">", "threshold": 0.0},
                "runtime_reload_freeze_integrity": "pass",
            },
            "dev17_gate": {
                "positive_margin_pairs": {"operator": ">=", "threshold": 13, "denominator": 17},
                "mean_reward_margin": {"operator": ">", "threshold": 0.0},
                "length_matched_margin": {"operator": ">", "threshold": 0.0},
            },
            "heldout29_gate": {
                "status": "not_yet_frozen",
                "authorization": "blocked_until_append_only_guardrail_amendment",
                "global_maximum_claims_after_authorization": 1,
            },
            "coding_guardrails": {
                "implementation_status": "not_yet_bound_blocks_heldout",
                "total_correct": {"operator": ">=", "threshold": 65},
                "general_correct": {"operator": ">=", "threshold": 5},
                "math_correct": {"operator": ">=", "threshold": 17},
                "finance_correct": {"operator": ">=", "threshold": 10},
                "code_correct": {"operator": ">=", "threshold": 14},
                "format_compliant": {"operator": ">=", "threshold": 90},
                "code_sandbox_execution_eligible": {"operator": ">=", "threshold": 26},
                "infrastructure_failures": {"operator": "<=", "threshold": 0},
            },
        },
        "forbidden": [
            "open_dev_rows_before_global_dev_lease",
            "open_heldout_rows_before_global_heldout_lease",
            "train_on_search30_dev17_or_heldout29",
            "add_candidate_or_change_hyperparameters_after_candidate_pool_freeze",
            "create_new_version_after_dev_claim",
            "change_evaluator_selector_gates_or_split_membership_after_precommit",
            "hide_failed_run_attempt_or_lease",
            "use_B24_capacity_audit_for_optimizer_or_candidate_evidence",
            "treat_B16_world2_GA1_as_a_candidate_axis",
            "use_old4_descriptive_results_as_candidate_selection_or_confirmation",
        ],
        "derived_state_policy": {
            "path": "derived/state.json",
            "authority": False,
            "rebuild_from": "validated immutable event chain and global leases",
        },
        "campaign_identity": {
            "campaign_id": CAMPAIGN_ID,
            "optimizer124_pair_ids_sha256": optimizer124["ordered_pair_ids_sha256"],
            "full154_pair_ids_sha256": full154["ordered_pair_ids_sha256"],
            "learning_rate_grid": LEARNING_RATE_GRID,
        },
    }
    return seal(document, "charter_sha256")


def build_v0003(
    charter: Mapping[str, Any], *, created_at_utc: str
) -> dict[str, Any]:
    created_at_utc = utc_timestamp(created_at_utc)
    if charter.get("charter_id") != CHARTER_ID:
        raise CharterError("cannot build v0003 for a different charter")
    campaign = _expect_mapping(charter.get("dataset_campaign"), "dataset campaign")
    parent = _expect_mapping(charter.get("parent_lineage"), "parent lineage")
    document = {
        "schema_name": VERSION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "charter_id": CHARTER_ID,
        "charter_sha256": charter["charter_sha256"],
        "goal_id": GOAL_ID,
        "version_id": INITIAL_VERSION_ID,
        "created_at_utc": created_at_utc,
        "status": "immutable_precommit_not_executed",
        "diagnosis": {
            "diagnosis_id": "day23-dpo-diagnosis-0001",
            "observed_symptoms": [
                "symptom.gate.hard_failure",
                "symptom.capability.no_lift",
            ],
            "primary_failure": "failure.unknown.localization.unlocalized",
            "causal_status": "inconclusive",
            "first_broken_invariant": (
                "after finite DPO updates, only 1/4 old mechanism-pair reward margins "
                "improved in each of two sealed topology mappings"
            ),
            "excluded_alternatives": [
                "failure.training.distributed.global_batch_drift",
                "failure.training.distributed.global_order_drift",
                "failure.training.distributed.sampler_padding",
                "failure.training.distributed.sampler_duplication",
                "failure.runtime.distributed.topology_visibility",
                "failure.training.dynamics.nan_inf",
                "failure.model.initialization.base_checkpoint_mismatch",
            ],
            "evidence_authority": charter["legacy_day23_terminal_chain"],
        },
        "intervention": {
            "primary_lever": PRIMARY_LEVER,
            "dependent_lever": None,
            "changed_levers": [PRIMARY_LEVER],
            "candidate_values": LEARNING_RATE_GRID,
            "prediction": (
                "one preregistered lower learning-rate scale may preserve positive "
                "preference direction across the heterogeneous fit/search campaign"
            ),
            "falsifier": (
                "none of the four locked LR/checkpoint candidates passes search30, "
                "or the independent-seed full154 refit fails any downstream gate"
            ),
        },
        "fixed_operational_constraint": {
            "world_size": 2,
            "parallelism": "DDP",
            "per_device_pair_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "candidate_axis": False,
            "B24_use": "forward_capacity_audit_only",
        },
        "fresh_start": {
            "checkpoint_id": parent["fresh_parent_checkpoint_id"],
            "downstream_key": parent["fresh_parent_downstream_key"],
            "merged_export_path": parent["fresh_parent_merged_export_path"],
            "fresh_lora": True,
            "resume": False,
            "legacy_failure_checkpoint_allowed": False,
        },
        "frozen_recipe": {
            "loss_type": "sigmoid",
            "beta": 0.1,
            "max_length": 512,
            "bf16": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "max_optimizer_steps": 30,
            "search_seed": SEARCH_SEED,
            "independent_refit_seed": REFIT_SEED,
        },
        "campaign": {
            "optional_capacity_audit": campaign["b24_forward_capacity_audit"],
            "lr_grid_optimizer_train": campaign["optimizer124"],
            "lr_grid_search_evaluation": campaign["search30"],
            "candidate_values": [
                {
                    "candidate_id": "candidate_lr_1e_6_step_15",
                    "learning_rate": 1e-6,
                    "checkpoint_step": 15,
                },
                {
                    "candidate_id": "candidate_lr_1e_6_step_30",
                    "learning_rate": 1e-6,
                    "checkpoint_step": 30,
                },
                {
                    "candidate_id": "candidate_lr_5e_6_step_15",
                    "learning_rate": 5e-6,
                    "checkpoint_step": 15,
                },
                {
                    "candidate_id": "candidate_lr_5e_6_step_30",
                    "learning_rate": 5e-6,
                    "checkpoint_step": 30,
                },
            ],
            "selection": {
                "all_four_candidates_sealed_before_unblinding": True,
                "eligible_only_if": ["all_optimizer_runs", "search30_gate"],
                "primary": "search30_positive_margin_pairs",
                "tie_breakers": [
                    "search30_mean_reward_margin",
                    "search30_length_matched_margin",
                    "earlier_checkpoint_step",
                    "lower_learning_rate",
                ],
                "exactly_one_finalist": True,
                "candidate_pool_freeze_before_dev": True,
            },
            "independent_seed_refit": campaign["full154_refit"],
            "refit_source": "exactly_one_frozen_lr_and_checkpoint_step_finalist",
            "refit_fresh_from_S1": True,
            "refit_checkpoint_candidates": CHECKPOINT_STEPS,
            "refit_max_steps": "selected_checkpoint_step",
        },
        "access_budget_at_precommit": {
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
            "dev_claims": 0,
            "heldout_claims": 0,
        },
        "stop_rules": [
            "reject_v0003_if_no_locked_candidate_passes_search30",
            "stop_entire_charter_on_any_failure_after_dev_claim",
            "stop_entire_charter_on_guardrail_or_heldout_failure",
            "stop_entire_charter_when_three_semantic_versions_are_exhausted",
        ],
        "execution_authorization": {
            "training_started": False,
            "requires_fresh_run_contract_and_runtime_binding": True,
            "requires_append_only_claim_and_receipts": True,
            "optimizer_execution_after_gpu_campaign_bind": True,
            "gpu_campaign_receipts_are_independent_authority": True,
            "charter_event_ingestion_and_promotion_pending_runtime_amendment": True,
        },
    }
    return seal(document, "version_sha256")


def _authority_ref(relative: str, document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    return {
        "path": relative,
        "content_sha256": document[hash_field],
        "file_sha256": sha256_bytes(pretty_json_bytes(document)),
    }


def _make_event(
    *,
    sequence: int,
    event_type: str,
    created_at_utc: str,
    state_before: str,
    state_after: str,
    version_id: str | None,
    prior_event: Mapping[str, Any] | None,
    authority_refs: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9-]+", event_type):
        raise CharterError(f"invalid event type: {event_type}")
    document = {
        "schema_name": EVENT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "event_id": f"event-{sequence:06d}-{event_type}",
        "sequence": sequence,
        "event_type": event_type,
        "created_at_utc": utc_timestamp(created_at_utc),
        "charter_id": CHARTER_ID,
        "goal_id": GOAL_ID,
        "version_id": version_id,
        "state_before": state_before,
        "state_after": state_after,
        "prior_event": prior_event,
        "authority_refs": list(authority_refs),
        "payload": dict(payload),
    }
    return seal(document, "event_sha256")


def _event_reference(event: Mapping[str, Any]) -> dict[str, Any]:
    event_type = event["event_type"]
    sequence = event["sequence"]
    relative = f"events/event-{sequence:06d}-{event_type}.json"
    return {
        "path": relative,
        "sequence": sequence,
        "event_sha256": event["event_sha256"],
        "file_sha256": sha256_bytes(pretty_json_bytes(event)),
    }


def build_initial_events(
    charter: Mapping[str, Any],
    version: Mapping[str, Any],
    *,
    created_at_utc: str,
) -> list[dict[str, Any]]:
    charter_event = _make_event(
        sequence=1,
        event_type="charter-frozen",
        created_at_utc=created_at_utc,
        state_before="uninitialized",
        state_after="charter_frozen",
        version_id=None,
        prior_event=None,
        authority_refs=[_authority_ref("charter.json", charter, "charter_sha256")],
        payload={
            "authority_write_mode": "O_EXCL",
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
            "legacy_failures_preserved": True,
        },
    )
    version_event = _make_event(
        sequence=2,
        event_type="version-precommitted",
        created_at_utc=created_at_utc,
        state_before="charter_frozen",
        state_after="version_precommitted",
        version_id=INITIAL_VERSION_ID,
        prior_event=_event_reference(charter_event),
        authority_refs=[
            _authority_ref(
                f"versions/{INITIAL_VERSION_ID}/version.json",
                version,
                "version_sha256",
            )
        ],
        payload={
            "primary_lever": PRIMARY_LEVER,
            "learning_rate_grid": LEARNING_RATE_GRID,
            "training_started": False,
            "dev_rows_opened": 0,
            "heldout_rows_opened": 0,
        },
    )
    return [charter_event, version_event]


TRANSITIONS: dict[tuple[str, str], str] = {
    ("uninitialized", "charter-frozen"): "charter_frozen",
    ("charter_frozen", "version-precommitted"): "version_precommitted",
    ("iteration_open", "version-precommitted"): "version_precommitted",
    ("version_precommitted", "lr-grid-search-completed"): "lr_grid_search_completed",
    ("lr_grid_search_completed", "finalist-frozen"): "finalist_frozen",
    ("lr_grid_search_completed", "version-rejected"): "iteration_open",
    ("finalist_frozen", "full154-refit-completed"): "full154_refit_completed",
    ("full154_refit_completed", "candidate-pool-frozen"): "candidate_pool_frozen",
    ("candidate_pool_frozen", "dev-lease-claimed"): "dev_lease_claimed",
    ("dev_lease_claimed", "dev-gate-passed"): "dev_gate_passed",
    ("dev_gate_passed", "guardrails-passed"): "guardrails_passed",
    ("guardrails_passed", "heldout-lease-claimed"): "heldout_lease_claimed",
    ("heldout_lease_claimed", "qualified"): "qualified",
}


def _allowed_transition(state: str, event_type: str) -> str:
    target = TRANSITIONS.get((state, event_type))
    if target is not None:
        return target
    if event_type == "terminal-no-go" and state not in {"qualified", "terminal_no_go"}:
        return "terminal_no_go"
    raise CharterError(f"event {event_type} is not allowed from state {state}")


def _event_path(root: Path, event: Mapping[str, Any]) -> Path:
    return root / "events" / f"event-{event['sequence']:06d}-{event['event_type']}.json"


def write_json_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise CharterError(f"immutable authority already exists: {path}") from error
    try:
        payload = pretty_json_bytes(document)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial O_EXCL authority is deliberately not overwritten.  Its
        # presence forces explicit forensic recovery instead of silent repair.
        raise


def write_derived(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".new", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(pretty_json_bytes(document))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_authority_bytes(
    path: Path,
    *,
    allowed_root: Path,
    bootcamp_root: Path = BOOTCAMP_ROOT,
) -> bytes:
    try:
        resolved = path.resolve(strict=True)
        resolved_root = allowed_root.resolve(strict=True)
    except OSError as error:
        raise CharterError(f"cannot resolve authority path: {path}") from error
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CharterError(f"authority path escaped its expected root: {path}")
    if path.is_symlink():
        raise CharterError(f"authority path must not be a symlink: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CharterError(f"cannot securely open authority: {path}") from error
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise CharterError(f"authority must be a regular file: {path}")
        for protected_relative in PROTECTED_PRELEASE_PATHS:
            protected = local_path(protected_relative, bootcamp_root)
            try:
                protected_stat = os.stat(protected, follow_symlinks=False)
            except OSError as error:
                raise CharterError(
                    f"cannot stat protected dataset identity: {protected}"
                ) from error
            if (
                opened_stat.st_dev == protected_stat.st_dev
                and opened_stat.st_ino == protected_stat.st_ino
            ):
                raise CharterError(
                    f"authority is a hard link to protected evaluation content: {path}"
                )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_authority_json(
    path: Path,
    *,
    allowed_root: Path,
    bootcamp_root: Path = BOOTCAMP_ROOT,
) -> Mapping[str, Any]:
    try:
        raw = read_authority_bytes(
            path, allowed_root=allowed_root, bootcamp_root=bootcamp_root
        )
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CharterError(f"cannot load authority JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise CharterError(f"authority JSON must be an object: {path}")
    return value


def _load_events(
    root: Path, *, bootcamp_root: Path = BOOTCAMP_ROOT
) -> list[Mapping[str, Any]]:
    events_root = root / "events"
    if not events_root.is_dir():
        raise CharterError("events directory is absent")
    paths = sorted(events_root.glob("*.json"))
    if not paths:
        raise CharterError("event chain is empty")
    events: list[Mapping[str, Any]] = []
    for expected_sequence, path in enumerate(paths, start=1):
        match = EVENT_FILE_RE.fullmatch(path.name)
        if match is None:
            raise CharterError(f"invalid event filename: {path.name}")
        if int(match.group("sequence")) != expected_sequence:
            raise CharterError("event sequence is not contiguous")
        event = load_authority_json(
            path, allowed_root=root, bootcamp_root=bootcamp_root
        )
        if event.get("sequence") != expected_sequence:
            raise CharterError(f"event sequence mismatch: {path.name}")
        if event.get("event_type") != match.group("event_type").replace("-", "-"):
            raise CharterError(f"event type mismatch: {path.name}")
        validate_seal(event, "event_sha256", path.name)
        raw = read_authority_bytes(
            path, allowed_root=root, bootcamp_root=bootcamp_root
        )
        if sha256_bytes(pretty_json_bytes(event)) != sha256_bytes(raw):
            raise CharterError(f"event file encoding is non-canonical: {path.name}")
        events.append(event)
    return events


def _validate_event_chain(
    events: Sequence[Mapping[str, Any]],
    charter: Mapping[str, Any],
    version: Mapping[str, Any],
) -> str:
    if len(events) != 2:
        raise CharterError(
            "runtime event append is not authorized by this precommit-only validator"
        )
    state = "uninitialized"
    previous: Mapping[str, Any] | None = None
    committed_versions: list[str] = []
    for event in events:
        if event.get("schema_name") != EVENT_SCHEMA or event.get("schema_version") != 1:
            raise CharterError("event schema drifted")
        if event.get("charter_id") != CHARTER_ID or event.get("goal_id") != GOAL_ID:
            raise CharterError("event charter identity drifted")
        utc_timestamp(event.get("created_at_utc"))
        if event.get("state_before") != state:
            raise CharterError("event state_before does not match the chain")
        expected_after = _allowed_transition(state, event.get("event_type"))
        if event.get("state_after") != expected_after:
            raise CharterError("event state_after does not match the state machine")
        if previous is None:
            if event.get("prior_event") is not None:
                raise CharterError("first event must not have a predecessor")
        elif event.get("prior_event") != _event_reference(previous):
            raise CharterError("event predecessor file/content hash drifted")
        if event.get("event_type") == "version-precommitted":
            version_id = event.get("version_id")
            match = VERSION_RE.fullmatch(str(version_id))
            if match is None or int(match.group("ordinal")) not in {3, 4, 5}:
                raise CharterError("committed version exceeds the charter budget")
            if version_id in committed_versions:
                raise CharterError("a semantic version was committed twice")
            if committed_versions and int(version_id[-4:]) != int(committed_versions[-1][-4:]) + 1:
                raise CharterError("semantic version ordinals are not contiguous")
            committed_versions.append(version_id)
        previous = event
        state = expected_after

    expected_initial = build_initial_events(
        charter,
        version,
        created_at_utc=events[0]["created_at_utc"],
    )
    if list(events[:2]) != expected_initial:
        raise CharterError("initial charter/version events drifted from the precommit")
    if len(committed_versions) > MAX_SEMANTIC_VERSIONS:
        raise CharterError("semantic-version budget exceeded")
    return state


def _lease_path(
    kind: str, charter: Mapping[str, Any], *, control_root: Path
) -> Path:
    leases = _expect_mapping(charter.get("access_leases"), "access leases")
    spec = _expect_mapping(leases.get(kind), f"{kind} lease")
    relative = Path(str(spec.get("claim_path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise CharterError(f"invalid {kind} lease path")
    return control_root / relative


def _validate_lease(
    kind: str,
    lease: Mapping[str, Any],
    charter: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> None:
    if lease.get("schema_name") != LEASE_SCHEMA or lease.get("schema_version") != 1:
        raise CharterError(f"{kind} lease schema drifted")
    validate_seal(lease, "lease_sha256", f"{kind} lease")
    spec = _expect_mapping(charter["access_leases"][kind], f"{kind} lease spec")
    if lease.get("access_kind") != kind or lease.get("dataset_identity") != spec["identity"]:
        raise CharterError(f"{kind} lease dataset identity drifted")
    if lease.get("charter_id") != CHARTER_ID or lease.get("goal_id") != GOAL_ID:
        raise CharterError(f"{kind} lease charter identity drifted")
    event_type = f"{kind}-lease-claimed"
    matches = [event for event in events if event.get("event_type") == event_type]
    if len(matches) != 1:
        raise CharterError(f"{kind} lease must have exactly one chain event")
    if matches[0].get("payload", {}).get("lease_sha256") != lease["lease_sha256"]:
        raise CharterError(f"{kind} lease event does not bind the lease")


def derive_state(
    charter: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    control_root: Path,
) -> dict[str, Any]:
    last = events[-1]
    committed = [
        event["version_id"]
        for event in events
        if event.get("event_type") == "version-precommitted"
    ]
    lease_presence = {
        kind: _lease_path(kind, charter, control_root=control_root).exists()
        for kind in ("dev", "heldout")
    }
    return {
        "schema_name": DERIVED_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "authority": False,
        "charter_id": CHARTER_ID,
        "goal_id": GOAL_ID,
        "current_state": last["state_after"],
        "current_version": committed[-1],
        "semantic_versions_committed": committed,
        "dev_lease_present": lease_presence["dev"],
        "heldout_lease_present": lease_presence["heldout"],
        "last_event": _event_reference(last),
        "rebuild_rule": "immutable event chain plus fixed-path global leases",
    }


def init_control(
    *,
    root: Path = CHARTER_ROOT,
    control_root: Path = CONTROL_ROOT,
    bootcamp_root: Path = BOOTCAMP_ROOT,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = utc_timestamp(created_at_utc or now_utc())
    charter_path = root / "charter.json"
    version_path = root / "versions" / INITIAL_VERSION_ID / "version.json"
    derived_path = root / "derived" / "state.json"
    planned = [charter_path, version_path, derived_path]
    planned.extend(root / "events" / f"event-{index:06d}-{name}.json" for index, name in ((1, "charter-frozen"), (2, "version-precommitted")))
    existing = [path for path in planned if path.exists()]
    if existing:
        raise CharterError(f"initialization is append-only; target already exists: {existing[0]}")

    charter = build_charter(created_at_utc=timestamp, bootcamp_root=bootcamp_root)
    version = build_v0003(charter, created_at_utc=timestamp)
    events = build_initial_events(charter, version, created_at_utc=timestamp)

    write_json_exclusive(charter_path, charter)
    write_json_exclusive(version_path, version)
    for event in events:
        write_json_exclusive(_event_path(root, event), event)
    write_derived(
        derived_path,
        derive_state(charter, events, control_root=control_root),
    )
    return {
        "status": "initialized",
        "charter_sha256": charter["charter_sha256"],
        "version_sha256": version["version_sha256"],
        "last_event_sha256": events[-1]["event_sha256"],
    }


def validate_control(
    *,
    root: Path = CHARTER_ROOT,
    control_root: Path = CONTROL_ROOT,
    bootcamp_root: Path = BOOTCAMP_ROOT,
    check_derived: bool = True,
) -> dict[str, Any]:
    charter_path = root / "charter.json"
    version_path = root / "versions" / INITIAL_VERSION_ID / "version.json"
    charter = load_authority_json(
        charter_path, allowed_root=root, bootcamp_root=bootcamp_root
    )
    version = load_authority_json(
        version_path, allowed_root=root, bootcamp_root=bootcamp_root
    )
    if charter.get("schema_name") != CHARTER_SCHEMA or charter.get("schema_version") != 1:
        raise CharterError("charter schema drifted")
    validate_seal(charter, "charter_sha256", "charter")
    expected_charter = build_charter(
        created_at_utc=charter.get("created_at_utc"), bootcamp_root=bootcamp_root
    )
    if charter != expected_charter:
        raise CharterError("charter differs from dynamically rebound immutable precommit")
    charter_raw = read_authority_bytes(
        charter_path, allowed_root=root, bootcamp_root=bootcamp_root
    )
    if sha256_bytes(charter_raw) != sha256_bytes(pretty_json_bytes(charter)):
        raise CharterError("charter file encoding is non-canonical")

    if version.get("schema_name") != VERSION_SCHEMA or version.get("schema_version") != 1:
        raise CharterError("version precommit schema drifted")
    validate_seal(version, "version_sha256", "v0003")
    expected_version = build_v0003(
        charter, created_at_utc=version.get("created_at_utc")
    )
    if version != expected_version:
        raise CharterError("v0003 differs from the immutable precommit")
    version_raw = read_authority_bytes(
        version_path, allowed_root=root, bootcamp_root=bootcamp_root
    )
    if sha256_bytes(version_raw) != sha256_bytes(pretty_json_bytes(version)):
        raise CharterError("version file encoding is non-canonical")

    events = _load_events(root, bootcamp_root=bootcamp_root)
    state = _validate_event_chain(events, charter, version)
    for kind in ("dev", "heldout"):
        path = _lease_path(kind, charter, control_root=control_root)
        if path.exists():
            _validate_lease(
                kind,
                load_authority_json(
                    path,
                    allowed_root=control_root,
                    bootcamp_root=bootcamp_root,
                ),
                charter,
                events,
            )
        elif any(event.get("event_type") == f"{kind}-lease-claimed" for event in events):
            raise CharterError(f"{kind} lease event exists without its global lease")

    expected_derived = derive_state(charter, events, control_root=control_root)
    derived_status = "unchecked"
    if check_derived:
        derived_path = root / "derived" / "state.json"
        actual_derived = load_authority_json(
            derived_path, allowed_root=root, bootcamp_root=bootcamp_root
        )
        if actual_derived != expected_derived:
            raise CharterError("derived state is stale; rebuild it from authority records")
        derived_status = "current_non_authority"
    return {
        "status": "valid",
        "authority_status": "valid",
        "derived_status": derived_status,
        "charter_id": CHARTER_ID,
        "current_state": state,
        "current_version": expected_derived["current_version"],
        "event_count": len(events),
        "dev_lease_present": expected_derived["dev_lease_present"],
        "heldout_lease_present": expected_derived["heldout_lease_present"],
        "heldout_content_opened_by_validator": False,
    }


def rebuild_derived(
    *,
    root: Path = CHARTER_ROOT,
    control_root: Path = CONTROL_ROOT,
    bootcamp_root: Path = BOOTCAMP_ROOT,
) -> dict[str, Any]:
    validate_control(
        root=root,
        control_root=control_root,
        bootcamp_root=bootcamp_root,
        check_derived=False,
    )
    charter = load_authority_json(
        root / "charter.json", allowed_root=root, bootcamp_root=bootcamp_root
    )
    events = _load_events(root, bootcamp_root=bootcamp_root)
    state = derive_state(charter, events, control_root=control_root)
    write_derived(root / "derived" / "state.json", state)
    return state


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="O_EXCL-create charter and v0003")
    init_parser.add_argument("--created-at", dest="created_at_utc")
    subparsers.add_parser("validate", help="validate all authority and derived state")
    subparsers.add_parser("status", help="print the validated current state")
    subparsers.add_parser(
        "rebuild-derived", help="rebuild the non-authoritative derived state cache"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = init_control(created_at_utc=args.created_at_utc)
        elif args.command in {"validate", "status"}:
            result = validate_control()
        elif args.command == "rebuild-derived":
            result = rebuild_derived()
        else:  # pragma: no cover - argparse makes this unreachable.
            raise CharterError(f"unsupported command: {args.command}")
    except CharterError as error:
        parser.exit(2, f"error: {error}\n")
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
