#!/usr/bin/env python3
"""Validate the compact, domain-neutral RSI control folder."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
BOOTCAMP_ROOT = ROOT.parent
VERSION_RE = re.compile(r"^rsi-v(\d{4})$")
RUN_RE = re.compile(r"^run-(\d{3})-[a-z0-9-]+$")
ATTEMPT_RE = re.compile(r"^attempt-(\d{3})$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TAXONOMY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
MAIN_RE = re.compile(
    r"^main-s(?P<seed>\d+)-lr(?P<lr>1e-4|3e-5|6e-5|8e-5)-"
    r"(?P<label>early|mid|final)$"
)
ATTEMPT_STATUSES = {
    "planned",
    "running",
    "succeeded",
    "infrastructure_failed",
    "model_failed",
    "interrupted",
}
TERMINAL_ATTEMPT_STATUSES = ATTEMPT_STATUSES - {"planned", "running"}
OPERATION_TYPES = {"train", "eval", "sandbox", "checkpoint", "artifact", "cost"}
CAUSAL_STATUSES = {"suspected", "supported", "causally_confirmed", "inconclusive"}


class RSIControlError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RSIControlError(f"cannot load JSON: {path}") from error


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RSIControlError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RSIControlError(f"{label} is invalid") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise RSIControlError(f"{label} must be UTC")
    return parsed


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".new", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".new", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RSIControlError(f"{label} must be an object")
    return value


def schema(value: Mapping[str, Any], name: str, label: str) -> None:
    if value.get("schema_name") != name or value.get("schema_version") != 1:
        raise RSIControlError(f"{label} schema drifted")


def segment_prefix(value: str, prefix: str) -> bool:
    """Return whether prefix names value or one of its dotted ancestors."""

    return value == prefix or value.startswith(prefix + ".")


def taxonomy_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or TAXONOMY_ID_RE.fullmatch(value) is None:
        raise RSIControlError(f"{label} is not a valid taxonomy ID")
    return value


def version_ordinal(value: Any, label: str) -> int:
    match = VERSION_RE.fullmatch(str(value))
    if match is None:
        raise RSIControlError(f"{label} is not a valid version ID")
    return int(match.group(1))


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RSIControlError(f"{label} must be a non-empty string")
    return value


def string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise RSIControlError(f"{label} must be a {qualifier}list")
    result: list[str] = []
    for item in value:
        result.append(nonempty_string(item, f"{label} item"))
    return result


def evidence_ref_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    refs = string_list(value, label, nonempty=nonempty)
    for ref in refs:
        path = Path(ref)
        if (
            path.is_absolute()
            or "\\" in ref
            or ".." in path.parts
            or path.as_posix() != ref
            or not path.parts
            or path.parts[0].casefold() == "generated"
        ):
            raise RSIControlError(
                f"{label} must contain canonical non-generated relative paths"
            )
    return refs


def nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RSIControlError(f"{label} must be a non-negative integer")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RSIControlError(f"cannot hash: {path}") from error
    return digest.hexdigest()


def validate_file_ref(record: Mapping[str, Any]) -> None:
    relative, expected = record.get("path"), record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
        raise RSIControlError("invalid artifact reference")
    path = (BOOTCAMP_ROOT / relative).resolve()
    try:
        path.relative_to(BOOTCAMP_ROOT.resolve())
    except ValueError as error:
        raise RSIControlError(f"artifact escapes bootcamp root: {relative}") from error
    if not path.is_file() or sha256(path) != expected:
        raise RSIControlError(f"artifact missing or drifted: {relative}")


def compute_gate_summary(
    values: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    if set(values) != set(gates):
        raise RSIControlError("gate values must exactly match the current goal")
    margins: dict[str, int] = {}
    for name, raw_gate in gates.items():
        gate = mapping(raw_gate, f"gate {name}")
        actual = nonnegative_int(values[name], name)
        threshold = nonnegative_int(gate.get("threshold"), f"threshold {name}")
        if gate.get("operator") == ">=":
            margins[name] = actual - threshold
        elif gate.get("operator") == "<=":
            margins[name] = threshold - actual
        else:
            raise RSIControlError(f"invalid gate operator: {name}")
    worst = min(margins.values())
    return {
        "gate_margins": margins,
        "all_numeric_gates_pass": worst >= 0,
        "worst_gate_margin": worst,
        "total_gate_deficit": sum(max(0, -value) for value in margins.values()),
    }


def load_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RSIControlError(f"cannot read {label}: {path}") from error
    rows: list[Mapping[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RSIControlError(f"blank {label} event: {path}:{number}")
        try:
            rows.append(mapping(json.loads(line), f"{label} event"))
        except json.JSONDecodeError as error:
            raise RSIControlError(f"invalid {label} JSONL: {path}:{number}") from error
    return rows


def validate_run_contract(path: Path, version_id: str, run_id: str) -> str:
    contract = mapping(load_json(path), "run contract")
    schema(contract, "rsi.run_contract", "run contract")
    if contract.get("version_id") != version_id or contract.get("run_id") != run_id:
        raise RSIControlError("run contract identity drifted")
    expected = contract.get("contract_sha256")
    actual = object_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    if expected != actual:
        raise RSIControlError("run contract hash drifted")
    return actual


def validate_operation_ledger(
    path: Path,
    *,
    version_id: str,
    run_id: str,
    attempt_id: str,
    run_contract: str,
) -> dict[str, Any]:
    rows = load_jsonl(path, "operation ledger")
    previous: str | None = None
    event_ids: set[str] = set()
    gpu_hours = 0.0
    eval_accesses = 0
    for sequence, row in enumerate(rows, start=1):
        schema(row, "rsi.operation_event", "operation event")
        if (
            row.get("sequence") != sequence
            or row.get("previous_event_sha256") != previous
            or row.get("version_id") != version_id
            or row.get("run_id") != run_id
            or row.get("attempt_id") != attempt_id
            or row.get("run_contract_sha256") != run_contract
        ):
            raise RSIControlError(f"operation event identity drifted: {path}:{sequence}")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise RSIControlError(f"operation event ID is invalid: {path}:{sequence}")
        event_ids.add(event_id)
        if row.get("operation_type") not in OPERATION_TYPES:
            raise RSIControlError(f"operation type is invalid: {path}:{sequence}")
        utc_timestamp(row.get("created_at_utc"), "operation created_at_utc")
        mapping(row.get("payload"), "operation payload")
        event_gpu_hours = row.get("gpu_hours")
        if (
            isinstance(event_gpu_hours, bool)
            or not isinstance(event_gpu_hours, (int, float))
            or not math.isfinite(float(event_gpu_hours))
            or float(event_gpu_hours) < 0
        ):
            raise RSIControlError("operation gpu_hours is invalid")
        event_eval_accesses = nonnegative_int(
            row.get("eval_accesses"), "operation eval_accesses"
        )
        if row.get("operation_type") == "eval" and event_eval_accesses != 1:
            raise RSIControlError("each eval operation must record exactly one access")
        if row.get("operation_type") != "eval" and event_eval_accesses != 0:
            raise RSIControlError("only eval operations may increment eval_accesses")
        expected = row.get("event_sha256")
        actual = object_sha256(
            {key: value for key, value in row.items() if key != "event_sha256"}
        )
        if expected != actual:
            raise RSIControlError(f"operation event hash drifted: {path}:{sequence}")
        previous = actual
        gpu_hours += float(event_gpu_hours)
        eval_accesses += event_eval_accesses
    return {
        "records": len(rows),
        "head_sha256": previous,
        "gpu_hours": round(gpu_hours, 9),
        "eval_accesses": eval_accesses,
    }


def validate_retry_log(
    path: Path,
    version_id: str,
    run_id: str,
    attempt_id: str,
    run_contract: Any,
) -> tuple[int, bool]:
    lines = load_jsonl(path, "retry log")
    if not lines:
        raise RSIControlError(f"retry log is empty: {path}")
    retries = 0
    open_retry = False
    for number, event in enumerate(lines, start=1):
        schema(event, "rsi.retry_event", "retry event")
        if (
            event.get("version_id") != version_id
            or event.get("run_id") != run_id
            or event.get("attempt_id") != attempt_id
        ):
            raise RSIControlError(f"retry identity drifted: {path}:{number}")
        if number == 1 and event.get("event_type") != "retry_log_initialized":
            raise RSIControlError(f"retry log lacks initialization: {path}")
        if event.get("event_type") == "retry_started":
            if open_retry:
                raise RSIControlError("a retry is already open")
            retries += 1
            if event.get("retry_ordinal") != retries:
                raise RSIControlError(f"retry ordinal drifted: {path}:{number}")
            if event.get("training_semantics_changed") is not False:
                raise RSIControlError("a semantic change is a new version, not a retry")
            if (
                run_contract is None
                or event.get("run_contract_before") != run_contract
                or event.get("run_contract_after") != run_contract
            ):
                raise RSIControlError("retry changed its run contract")
            if not all(
                isinstance(event.get(field), str) and event.get(field)
                for field in ("failure_layer", "failure_code", "remediation", "created_at_utc")
            ):
                raise RSIControlError("retry start evidence is incomplete")
            utc_timestamp(event["created_at_utc"], "retry created_at_utc")
            open_retry = True
        elif event.get("event_type") == "retry_finished":
            if not open_retry or event.get("retry_ordinal") != retries:
                raise RSIControlError("retry finish lacks a matching start")
            if event.get("run_contract_sha256") != run_contract:
                raise RSIControlError("retry finish changed its run contract")
            if event.get("outcome") not in TERMINAL_ATTEMPT_STATUSES:
                raise RSIControlError("retry outcome is invalid")
            utc_timestamp(event.get("created_at_utc"), "retry created_at_utc")
            open_retry = False
        elif number != 1 or event.get("event_type") != "retry_log_initialized":
            raise RSIControlError(f"unknown retry event: {path}:{number}")
    return retries, open_retry


def validate_attempt(
    path: Path, version_id: str, run_id: str, run_contract: Any
) -> dict[str, Any]:
    attempt_id = path.name
    match = ATTEMPT_RE.fullmatch(attempt_id)
    if match is None:
        raise RSIControlError(f"invalid attempt ID: {attempt_id}")
    attempt = mapping(load_json(path / "attempt.json"), "attempt")
    schema(attempt, "rsi.attempt", "attempt")
    if (
        attempt.get("version_id") != version_id
        or attempt.get("run_id") != run_id
        or attempt.get("attempt_id") != attempt_id
        or attempt.get("ordinal") != int(match.group(1))
    ):
        raise RSIControlError(f"attempt identity drifted: {path}")
    attempt_contract = attempt.get("run_contract_sha256")
    if attempt_contract is not None and attempt_contract != run_contract:
        raise RSIControlError(f"attempt/run contract drifted: {path}")
    status = attempt.get("status")
    if status not in ATTEMPT_STATUSES:
        raise RSIControlError(f"attempt status is invalid: {path}")
    retries, open_retry = validate_retry_log(
        path / "logs/retries.jsonl", version_id, run_id, attempt_id, run_contract
    )
    if attempt.get("retry_count") != retries:
        raise RSIControlError(f"retry count drifted: {path}")
    started = attempt.get("started_at_utc")
    ended = attempt.get("ended_at_utc")
    if status == "planned":
        if any(
            attempt.get(field) is not None
            for field in (
                "execution_identity",
                "started_at_utc",
                "ended_at_utc",
                "exit_code",
                "error",
            )
        ):
            raise RSIControlError("planned attempt contains execution evidence")
        if attempt.get("consumed_gpu_hours") != 0.0:
            raise RSIControlError("planned attempt has GPU cost")
        return {"gpu_hours": 0.0, "eval_accesses": 0}
    if run_contract is None or attempt_contract != run_contract:
        raise RSIControlError("running/terminal attempt lacks a bound run contract")
    mapping(attempt.get("execution_identity"), "execution identity")
    started_time = utc_timestamp(started, "attempt started_at_utc")
    event_log = mapping(attempt.get("event_log"), "attempt event_log")
    event_path = path / str(event_log.get("path"))
    try:
        event_path.resolve().relative_to(path.resolve())
    except ValueError as error:
        raise RSIControlError("attempt event log escapes its attempt") from error
    ledger = validate_operation_ledger(
        event_path,
        version_id=version_id,
        run_id=run_id,
        attempt_id=attempt_id,
        run_contract=run_contract,
    )
    if (
        event_log.get("records") != ledger["records"]
        or event_log.get("head_sha256") != ledger["head_sha256"]
        or attempt.get("consumed_gpu_hours") != ledger["gpu_hours"]
    ):
        raise RSIControlError("attempt operation aggregation drifted")
    if status == "running":
        if ended is not None or attempt.get("exit_code") is not None or attempt.get("error") is not None:
            raise RSIControlError("running attempt contains terminal evidence")
        expected_active = retries if open_retry else None
        if attempt.get("active_retry") != expected_active:
            raise RSIControlError("attempt active_retry drifted")
    else:
        ended_time = utc_timestamp(ended, "attempt ended_at_utc")
        if ended_time < started_time:
            raise RSIControlError("attempt ended before it started")
        exit_code = attempt.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise RSIControlError("terminal attempt exit_code is invalid")
        if status == "succeeded":
            if exit_code != 0 or attempt.get("error") is not None:
                raise RSIControlError("successful attempt contains failure evidence")
        else:
            error = mapping(attempt.get("error"), "attempt error")
            if exit_code == 0 or not all(
                isinstance(error.get(field), str) and error.get(field)
                for field in ("layer", "code", "message")
            ) or not isinstance(error.get("retryable"), bool):
                raise RSIControlError("failed attempt evidence is incomplete")
            if status == "model_failed" and error.get("layer") == "infrastructure":
                raise RSIControlError("infrastructure failure was labeled as model failure")
        if open_retry:
            raise RSIControlError("terminal attempt has an unfinished retry")
    return {"gpu_hours": ledger["gpu_hours"], "eval_accesses": ledger["eval_accesses"]}


def validate_result(
    version_metrics: Mapping[str, Any], goal: Mapping[str, Any]
) -> None:
    observed = mapping(version_metrics.get("observed"), "observed metrics")
    primary = observed.get("primary")
    confirmation = observed.get("confirmation")
    gates = mapping(goal.get("hard_gates"), "hard gates")
    summaries: list[dict[str, Any]] = []
    for label, value in (("primary", primary), ("confirmation", confirmation)):
        if value is None:
            continue
        result = mapping(value, label)
        computed = compute_gate_summary(mapping(result.get("gate_values"), "gate values"), gates)
        for field in ("worst_gate_margin", "total_gate_deficit", "all_numeric_gates_pass"):
            if result.get(field) != computed[field]:
                raise RSIControlError(f"{label} {field} is not recomputed truth")
        summaries.append(computed)
    if summaries:
        expected_worst = min(item["worst_gate_margin"] for item in summaries)
        expected_deficit = sum(item["total_gate_deficit"] for item in summaries)
        if observed.get("worst_gate_margin") != expected_worst or observed.get("total_gate_deficit") != expected_deficit:
            raise RSIControlError("version gate aggregation drifted")
    if version_metrics.get("confirmed_qualified") is not True:
        return
    if version_metrics.get("complete_evidence") is not True or primary is None or confirmation is None:
        raise RSIControlError("qualification lacks complete primary/confirmation evidence")
    primary_record, confirmation_record = mapping(primary, "primary"), mapping(confirmation, "confirmation")
    if not all(
        item.get("all_numeric_gates_pass") is True and item.get("evidence_gates_pass") is True
        for item in (primary_record, confirmation_record)
    ):
        raise RSIControlError("qualified result contains a failed gate")
    first = MAIN_RE.fullmatch(str(primary_record.get("candidate")))
    second = MAIN_RE.fullmatch(str(confirmation_record.get("candidate")))
    policy = mapping(goal.get("confirmation"), "confirmation policy")
    if (
        first is None
        or second is None
        or int(first.group("seed")) != policy.get("primary_seed")
        or int(second.group("seed")) != policy.get("confirmation_seed")
        or first.group("lr") != second.group("lr")
        or first.group("label") != second.group("label")
    ):
        raise RSIControlError("primary/confirmation identity mismatch")


def validate_intervention_taxonomy(
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the compatibility families and prospective canonical leaves."""

    if taxonomy.get("taxonomy_revision") != 2:
        raise RSIControlError("intervention taxonomy revision drifted")
    nonempty_string(
        taxonomy.get("compatibility_policy"), "intervention taxonomy compatibility policy"
    )
    effective_from = str(taxonomy.get("effective_from_version"))
    version_ordinal(effective_from, "intervention taxonomy effective version")

    raw_families = taxonomy.get("levers")
    if not isinstance(raw_families, list) or not raw_families:
        raise RSIControlError("intervention taxonomy has no lever families")
    families: dict[str, set[str]] = {}
    for raw_family in raw_families:
        family = mapping(raw_family, "lever family")
        family_id = taxonomy_id(family.get("id"), "lever family ID")
        if family_id in families:
            raise RSIControlError(f"duplicate lever family: {family_id}")
        raw_children = family.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise RSIControlError(f"lever family has no children: {family_id}")
        children: set[str] = set()
        for raw_child in raw_children:
            child = taxonomy_id(raw_child, f"lever child under {family_id}")
            if "." in child or child in children:
                raise RSIControlError(f"invalid or duplicate lever child: {family_id}.{child}")
            children.add(child)
        families[family_id] = children

    raw_details = taxonomy.get("detailed_levers")
    if not isinstance(raw_details, list) or not raw_details:
        raise RSIControlError("intervention taxonomy has no detailed levers")
    detailed: set[str] = set()
    for raw_detail in raw_details:
        detail = mapping(raw_detail, "detailed lever")
        lever_id = taxonomy_id(detail.get("id"), "detailed lever ID")
        family_id = taxonomy_id(detail.get("family"), f"family for {lever_id}")
        description = detail.get("description")
        if lever_id in detailed:
            raise RSIControlError(f"duplicate detailed lever: {lever_id}")
        if family_id not in families or not segment_prefix(lever_id, family_id):
            raise RSIControlError(f"detailed lever has unknown family: {lever_id}")
        relative = lever_id[len(family_id) + 1 :]
        if not relative or relative.split(".", 1)[0] not in families[family_id]:
            raise RSIControlError(f"detailed lever has unknown child: {lever_id}")
        if not isinstance(description, str) or not description.strip():
            raise RSIControlError(f"detailed lever lacks a description: {lever_id}")
        detailed.add(lever_id)

    for raw_priority in taxonomy.get("current_sft_priority", []):
        priority = taxonomy_id(raw_priority, "current SFT priority")
        if not any(
            priority == lever_id or lever_id.startswith(priority + ".")
            for lever_id in detailed
        ):
            raise RSIControlError(f"current SFT priority is unknown: {priority}")

    return {
        "effective_from_version": effective_from,
        "families": set(families),
        "detailed_levers": detailed,
    }


def validate_failure_taxonomy(
    failure_taxonomy: Mapping[str, Any], detailed_levers: set[str]
) -> dict[str, Any]:
    """Validate the prospective failure, symptom, alias, and example graph."""

    if failure_taxonomy.get("taxonomy_revision") != 1:
        raise RSIControlError("failure taxonomy revision drifted")
    nonempty_string(
        failure_taxonomy.get("compatibility_policy"),
        "failure taxonomy compatibility policy",
    )
    effective_from = str(failure_taxonomy.get("effective_from_version"))
    version_ordinal(effective_from, "failure taxonomy effective version")

    def indexed_records(
        field: str, *, prefix: str, parent_ids: set[str] | None = None
    ) -> dict[str, Mapping[str, Any]]:
        raw_records = failure_taxonomy.get(field)
        if not isinstance(raw_records, list) or not raw_records:
            raise RSIControlError(f"failure taxonomy has no {field}")
        records: dict[str, Mapping[str, Any]] = {}
        for raw_record in raw_records:
            record = mapping(raw_record, field)
            record_id = taxonomy_id(record.get("id"), f"{field} ID")
            if not record_id.startswith(prefix) or record_id in records:
                raise RSIControlError(f"invalid or duplicate {field} ID: {record_id}")
            nonempty_string(record.get("definition"), f"definition for {record_id}")
            if parent_ids is not None:
                parent = taxonomy_id(record.get("parent"), f"parent for {record_id}")
                if parent not in parent_ids or not segment_prefix(record_id, parent):
                    raise RSIControlError(f"unknown parent for {record_id}: {parent}")
            records[record_id] = record
        return records

    families = indexed_records("failure_families", prefix="failure.")
    topics = indexed_records(
        "failure_topics", prefix="failure.", parent_ids=set(families)
    )
    modes = indexed_records(
        "failure_modes", prefix="failure.", parent_ids=set(topics)
    )
    symptoms = indexed_records("symptoms", prefix="symptom.")
    all_failure_ids = set(families) | set(topics) | set(modes)
    if len(all_failure_ids) != len(families) + len(topics) + len(modes):
        raise RSIControlError("failure family, topic, and mode IDs must be globally unique")

    raw_aliases = failure_taxonomy.get("legacy_aliases")
    if not isinstance(raw_aliases, list):
        raise RSIControlError("failure taxonomy aliases must be a list")
    aliases: set[tuple[str, str]] = set()
    for raw_alias in raw_aliases:
        alias = mapping(raw_alias, "legacy failure alias")
        alias_name = nonempty_string(alias.get("alias"), "legacy failure alias")
        alias_type = nonempty_string(alias.get("alias_type"), "legacy alias type")
        key = (alias_type, alias_name)
        if key in aliases:
            raise RSIControlError(f"duplicate legacy failure alias: {alias_type}/{alias_name}")
        target = taxonomy_id(alias.get("maps_to"), f"target for legacy alias {alias_name}")
        if target not in all_failure_ids:
            raise RSIControlError(f"legacy failure alias has unknown target: {alias_name}")
        aliases.add(key)

    raw_crosswalk = failure_taxonomy.get("intervention_crosswalk_examples")
    if not isinstance(raw_crosswalk, list) or not raw_crosswalk:
        raise RSIControlError("failure taxonomy has no intervention crosswalk examples")
    crosswalk_failures: set[str] = set()
    for raw_example in raw_crosswalk:
        example = mapping(raw_example, "failure/intervention crosswalk example")
        failure_id = taxonomy_id(example.get("failure_mode"), "crosswalk failure mode")
        if failure_id not in modes or failure_id in crosswalk_failures:
            raise RSIControlError(f"invalid or duplicate crosswalk failure: {failure_id}")
        for lever_id in string_list(
            example.get("candidate_levers"),
            f"candidate levers for {failure_id}",
            nonempty=True,
        ):
            if lever_id not in detailed_levers:
                raise RSIControlError(f"crosswalk has unknown intervention lever: {lever_id}")
        nonempty_string(example.get("selection_note"), f"selection note for {failure_id}")
        crosswalk_failures.add(failure_id)

    raw_cases = failure_taxonomy.get("case_examples")
    if not isinstance(raw_cases, list):
        raise RSIControlError("failure taxonomy case examples must be a list")
    case_ids: set[str] = set()
    for raw_case in raw_cases:
        case = mapping(raw_case, "failure taxonomy case")
        case_id = nonempty_string(case.get("id"), "failure taxonomy case ID")
        if case_id in case_ids:
            raise RSIControlError(f"duplicate failure taxonomy case: {case_id}")
        version_ordinal(case.get("diagnosed_in_version"), f"diagnosed version for {case_id}")
        version_ordinal(case.get("remediated_in_version"), f"remediated version for {case_id}")
        for symptom_id in string_list(
            case.get("symptom_ids"), f"symptoms for {case_id}", nonempty=True
        ):
            if symptom_id not in symptoms:
                raise RSIControlError(f"case has unknown symptom: {case_id}/{symptom_id}")
        primary = taxonomy_id(
            case.get("primary_failure_mode"), f"primary failure for {case_id}"
        )
        if primary not in modes:
            raise RSIControlError(f"case primary failure is not a mode: {case_id}/{primary}")
        case_roles: dict[str, set[str]] = {}
        for field in ("contributing_failure_modes", "excluded_failure_modes"):
            role_ids = string_list(case.get(field), f"{field} for {case_id}")
            if len(set(role_ids)) != len(role_ids):
                raise RSIControlError(f"case repeats a failure mode: {case_id}/{field}")
            for failure_id in role_ids:
                if failure_id not in modes:
                    raise RSIControlError(f"case has unknown failure mode: {case_id}/{failure_id}")
            case_roles[field] = set(role_ids)
        if (
            primary in case_roles["contributing_failure_modes"]
            or primary in case_roles["excluded_failure_modes"]
            or case_roles["contributing_failure_modes"]
            & case_roles["excluded_failure_modes"]
        ):
            raise RSIControlError(f"case failure roles overlap: {case_id}")
        if case.get("confidence") not in CAUSAL_STATUSES:
            raise RSIControlError(f"case has invalid causal status: {case_id}")
        for field in ("last_good_stage", "first_bad_stage", "broken_invariant"):
            nonempty_string(case.get(field), f"{field} for {case_id}")
        string_list(case.get("causal_chain"), f"causal chain for {case_id}", nonempty=True)
        for field in ("historical_intervention", "prospective_canonical_equivalent"):
            case_intervention = mapping(case.get(field), f"{field} for {case_id}")
            primary_lever = taxonomy_id(
                case_intervention.get("primary_lever"), f"{field} primary lever"
            )
            dependent_raw = case_intervention.get("dependent_lever")
            dependent_lever = (
                taxonomy_id(dependent_raw, f"{field} dependent lever")
                if dependent_raw is not None
                else None
            )
            if field == "prospective_canonical_equivalent" and (
                primary_lever not in detailed_levers
                or (
                    dependent_lever is not None
                    and dependent_lever not in detailed_levers
                )
            ):
                raise RSIControlError(f"case has unknown canonical intervention: {case_id}")
        evidence_ref_list(
            case.get("evidence_refs"),
            f"evidence refs for {case_id}",
            nonempty=True,
        )
        case_ids.add(case_id)

    return {
        "effective_from_version": effective_from,
        "families": set(families),
        "topics": set(topics),
        "failure_modes": set(modes),
        "symptoms": set(symptoms),
    }


def validate_version_diagnosis(
    version: Mapping[str, Any],
    intervention: Mapping[str, Any],
    failure_contract: Mapping[str, Any],
    detailed_levers: set[str],
) -> None:
    """Require a precommitted, evidence-localized diagnosis for new versions."""

    diagnosis = mapping(version.get("diagnosis"), "version diagnosis")
    nonempty_string(diagnosis.get("diagnosis_id"), "diagnosis ID")
    raw_symptoms = diagnosis.get("observed_symptoms")
    if not isinstance(raw_symptoms, list) or not raw_symptoms:
        raise RSIControlError("version diagnosis has no observed symptoms")
    for raw_symptom in raw_symptoms:
        symptom = mapping(raw_symptom, "observed symptom")
        symptom_id = taxonomy_id(symptom.get("id"), "observed symptom ID")
        if symptom_id not in failure_contract["symptoms"]:
            raise RSIControlError(f"unknown observed symptom: {symptom_id}")
        nonempty_string(symptom.get("scope"), f"scope for {symptom_id}")
        evidence_ref_list(
            symptom.get("evidence_refs"),
            f"evidence refs for {symptom_id}",
            nonempty=True,
        )

    primary_failure = mapping(diagnosis.get("primary_failure"), "primary failure")
    primary_failure_id = taxonomy_id(primary_failure.get("id"), "primary failure ID")
    if primary_failure_id not in failure_contract["failure_modes"]:
        raise RSIControlError(f"primary failure is not a canonical mode: {primary_failure_id}")
    for field in ("first_broken_invariant", "pipeline_boundary"):
        nonempty_string(primary_failure.get(field), f"primary failure {field}")
    if primary_failure.get("causal_status") not in CAUSAL_STATUSES:
        raise RSIControlError("primary failure causal status is invalid")
    evidence_ref_list(
        primary_failure.get("evidence_refs"),
        "primary failure evidence refs",
        nonempty=True,
    )

    contributing = string_list(
        diagnosis.get("contributing_failures"), "contributing failures"
    )
    if len(set(contributing)) != len(contributing) or primary_failure_id in contributing:
        raise RSIControlError("contributing failures repeat the primary or each other")
    for failure_id in contributing:
        if failure_id not in failure_contract["failure_modes"]:
            raise RSIControlError(f"unknown contributing failure: {failure_id}")
    raw_excluded = diagnosis.get("excluded_alternatives")
    if not isinstance(raw_excluded, list):
        raise RSIControlError("excluded alternatives must be a list")
    excluded_ids: set[str] = set()
    for raw_alternative in raw_excluded:
        alternative = mapping(raw_alternative, "excluded alternative")
        failure_id = taxonomy_id(alternative.get("id"), "excluded failure ID")
        if failure_id not in failure_contract["failure_modes"]:
            raise RSIControlError(f"unknown excluded failure: {failure_id}")
        if failure_id in excluded_ids or failure_id == primary_failure_id or failure_id in contributing:
            raise RSIControlError(f"excluded failure overlaps another diagnosis role: {failure_id}")
        nonempty_string(alternative.get("reason"), f"exclusion reason for {failure_id}")
        evidence_ref_list(
            alternative.get("evidence_refs"),
            f"exclusion evidence for {failure_id}",
            nonempty=True,
        )
        excluded_ids.add(failure_id)

    diagnosed_intervention = mapping(
        diagnosis.get("intervention"), "diagnosed intervention"
    )
    diagnosed_primary = taxonomy_id(
        diagnosed_intervention.get("primary_lever"), "diagnosed primary lever"
    )
    if diagnosed_primary not in detailed_levers or diagnosed_primary != intervention.get("primary_lever"):
        raise RSIControlError("diagnosed primary lever differs from the version intervention")
    diagnosed_dependent = diagnosed_intervention.get("dependent_lever")
    changed = list(intervention.get("changed_levers", []))
    expected_dependent = next(
        (lever for lever in changed if lever != intervention.get("primary_lever")), None
    )
    if diagnosed_dependent != expected_dependent:
        raise RSIControlError("diagnosed dependent lever differs from the version intervention")
    if diagnosed_dependent is not None:
        if diagnosed_dependent not in detailed_levers:
            raise RSIControlError("diagnosed dependent lever is not canonical")
        nonempty_string(
            diagnosed_intervention.get("dependent_lever_reason"),
            "diagnosed dependent lever reason",
        )
    nonempty_string(
        diagnosed_intervention.get("expected_mechanism"),
        "diagnosed intervention mechanism",
    )
    for field in ("prediction", "falsifier"):
        nonempty_string(diagnosis.get(field), f"diagnosis {field}")
    for field in ("frozen_invariants", "guardrails"):
        string_list(diagnosis.get(field), f"diagnosis {field}", nonempty=True)


def validate_control() -> dict[str, Any]:
    state = mapping(load_json(ROOT / "state.json"), "state")
    goal = mapping(load_json(ROOT / "goal.json"), "goal")
    taxonomy = mapping(load_json(ROOT / "taxonomy.json"), "taxonomy")
    failure_taxonomy = mapping(
        load_json(ROOT / "failure-taxonomy.json"), "failure taxonomy"
    )
    metric_contract = mapping(load_json(ROOT / "metrics.json"), "metrics")
    history = mapping(load_json(ROOT / "history/legacy.json"), "history")
    schema(state, "rsi.state", "state")
    schema(goal, "rsi.goal", "goal")
    schema(taxonomy, "rsi.taxonomy", "taxonomy")
    schema(failure_taxonomy, "rsi.failure_taxonomy", "failure taxonomy")
    schema(metric_contract, "rsi.metrics", "metrics")
    schema(history, "rsi.history", "history")
    taxonomy_contract = validate_intervention_taxonomy(taxonomy)
    failure_contract = validate_failure_taxonomy(
        failure_taxonomy, taxonomy_contract["detailed_levers"]
    )
    if (
        taxonomy_contract["effective_from_version"]
        != failure_contract["effective_from_version"]
    ):
        raise RSIControlError("failure and intervention taxonomies activate in different versions")
    del metric_contract
    if state.get("current_goal") != goal.get("goal_id"):
        raise RSIControlError("state and goal differ")
    for record in goal.get("frozen_files", []):
        validate_file_ref(mapping(record, "frozen file"))
    for record in history.get("artifact_refs", []):
        validate_file_ref(mapping(record, "history artifact"))

    raw_prefixes = goal.get("allowed_lever_prefixes", [])
    if not isinstance(raw_prefixes, list) or not raw_prefixes:
        raise RSIControlError("goal has no allowed lever prefixes")
    prefixes = [taxonomy_id(value, "allowed lever prefix") for value in raw_prefixes]
    known_levers = taxonomy_contract["families"]
    detailed_levers = taxonomy_contract["detailed_levers"]
    strict_from = version_ordinal(
        taxonomy_contract["effective_from_version"],
        "intervention taxonomy effective version",
    )
    version_entries = state.get("versions")
    if not isinstance(version_entries, list):
        raise RSIControlError("state versions must be a list")
    version_dirs = sorted(path for path in (ROOT / "versions").iterdir() if path.is_dir())
    ids = [path.name for path in version_dirs]
    if ids != [mapping(item, "version entry").get("version_id") for item in version_entries]:
        raise RSIControlError("state version index drifted")
    if [int(VERSION_RE.fullmatch(value).group(1)) if VERSION_RE.fullmatch(value) else -1 for value in ids] != list(range(1, len(ids) + 1)):
        raise RSIControlError("version IDs must be contiguous")

    for version_dir, entry_raw in zip(version_dirs, version_entries):
        entry = mapping(entry_raw, "version entry")
        version_id = version_dir.name
        version = mapping(load_json(version_dir / "version.json"), "version")
        schema(version, "rsi.version", "version")
        if version.get("version_id") != version_id or version.get("goal_id") != goal.get("goal_id"):
            raise RSIControlError(f"version identity drifted: {version_id}")
        if version.get("status") != entry.get("status"):
            raise RSIControlError(f"version status drifted: {version_id}")
        intervention = mapping(version.get("intervention"), "intervention")
        changed = intervention.get("changed_levers")
        if not isinstance(changed, list) or not changed:
            raise RSIControlError(f"version has no declared lever: {version_id}")
        changed_ids = [taxonomy_id(lever, "changed lever") for lever in changed]
        if len(set(changed_ids)) != len(changed_ids):
            raise RSIControlError(f"version repeats a changed lever: {version_id}")
        for lever in changed_ids:
            if not any(segment_prefix(lever, family) for family in known_levers):
                raise RSIControlError(f"unknown lever: {lever}")
            if version.get("kind") != "baseline_reset" and not any(
                segment_prefix(lever, prefix) for prefix in prefixes
            ):
                raise RSIControlError(
                    f"changed lever is outside current goal: {version_id}/{lever}"
                )
            if version_ordinal(version_id, "version ID") >= strict_from and lever not in detailed_levers:
                raise RSIControlError(f"future version must use a canonical lever leaf: {lever}")
        if version.get("kind") != "baseline_reset":
            primary = intervention.get("primary_lever")
            if not isinstance(primary, str) or not any(
                segment_prefix(primary, prefix) for prefix in prefixes
            ):
                raise RSIControlError(f"primary lever is outside current goal: {version_id}")
            if len(changed_ids) > 2 or primary not in changed_ids:
                raise RSIControlError(f"version changes too many levers: {version_id}")
            if version_ordinal(version_id, "version ID") >= strict_from and primary not in detailed_levers:
                raise RSIControlError(f"future primary lever must be a canonical leaf: {version_id}")
            if len(changed_ids) == 2:
                nonempty_string(
                    intervention.get("dependent_lever_reason"),
                    f"dependent lever reason for {version_id}",
                )
            if version_ordinal(version_id, "version ID") >= strict_from:
                validate_version_diagnosis(
                    version,
                    intervention,
                    failure_contract,
                    detailed_levers,
                )
        elif version_id != "rsi-v0001" or intervention.get("causal_attribution") is not False:
            raise RSIControlError("only rsi-v0001 may be a baseline reset")

        artifacts = mapping(load_json(version_dir / "artifacts.json"), "artifacts")
        schema(artifacts, "rsi.version_artifacts", "version artifacts")
        if artifacts.get("version_id") != version_id:
            raise RSIControlError(f"artifact identity drifted: {version_id}")
        for record in artifacts.get("implementation", []):
            validate_file_ref(mapping(record, "implementation artifact"))

        run_ids = version.get("runs")
        version_gpu_hours = 0.0
        version_eval_accesses = 0
        run_dirs = sorted(path for path in (version_dir / "runs").iterdir() if path.is_dir())
        if run_ids != [path.name for path in run_dirs]:
            raise RSIControlError(f"run index drifted: {version_id}")
        for run_dir in run_dirs:
            run_id = run_dir.name
            if RUN_RE.fullmatch(run_id) is None:
                raise RSIControlError(f"invalid run ID: {run_id}")
            run = mapping(load_json(run_dir / "run.json"), "run")
            schema(run, "rsi.run", "run")
            if run.get("version_id") != version_id or run.get("run_id") != run_id:
                raise RSIControlError(f"run identity drifted: {run_id}")
            contract = run.get("run_contract_sha256")
            if contract is not None and (not isinstance(contract, str) or not SHA_RE.fullmatch(contract)):
                raise RSIControlError(f"invalid run contract: {run_id}")
            contract_path = run.get("run_contract_path")
            if contract is not None:
                if contract_path != "run-contract.json":
                    raise RSIControlError(f"run contract path is invalid: {run_id}")
                if validate_run_contract(run_dir / contract_path, version_id, run_id) != contract:
                    raise RSIControlError(f"run contract reference drifted: {run_id}")
                if not isinstance(run.get("remote_run_root"), str) or not run.get("remote_run_root"):
                    raise RSIControlError(f"bound run lacks remote_run_root: {run_id}")
            elif contract_path is not None:
                raise RSIControlError(f"unbound run contains a contract path: {run_id}")
            attempt_dirs = sorted(path for path in (run_dir / "attempts").iterdir() if path.is_dir())
            if run.get("attempts") != [path.name for path in attempt_dirs]:
                raise RSIControlError(f"attempt index drifted: {run_id}")
            for attempt_dir in attempt_dirs:
                totals = validate_attempt(attempt_dir, version_id, run_id, contract)
                version_gpu_hours += totals["gpu_hours"]
                version_eval_accesses += totals["eval_accesses"]

        version_metrics = mapping(load_json(version_dir / "metrics.json"), "version metrics")
        schema(version_metrics, "rsi.version_metrics", "version metrics")
        if version_metrics.get("version_id") != version_id:
            raise RSIControlError(f"metric identity drifted: {version_id}")
        validate_result(version_metrics, goal)
        observed = mapping(version_metrics.get("observed"), "observed metrics")
        if (
            observed.get("gpu_hours") != round(version_gpu_hours, 9)
            or observed.get("eval_accesses") != version_eval_accesses
        ):
            raise RSIControlError(f"version operation aggregation drifted: {version_id}")
        if entry.get("confirmed_qualified") != version_metrics.get("confirmed_qualified"):
            raise RSIControlError(f"state/result qualification drifted: {version_id}")

    if state.get("current_version") not in ids:
        raise RSIControlError("current version is missing")
    current_attempt = (
        ROOT / "versions" / str(state.get("current_version")) / "runs"
        / str(state.get("current_run")) / "attempts" / str(state.get("current_attempt"))
    )
    if not current_attempt.is_dir():
        raise RSIControlError("current run/attempt is missing")
    if state.get("goal_status") == "achieved":
        current_metrics = mapping(
            load_json(ROOT / "versions" / str(state["current_version"]) / "metrics.json"),
            "current metrics",
        )
        if current_metrics.get("confirmed_qualified") is not True:
            raise RSIControlError("goal cannot be achieved without confirmation")
    return {
        "status": "valid",
        "goal_status": state.get("goal_status"),
        "current_goal": state.get("current_goal"),
        "current_version": state.get("current_version"),
        "current_run": state.get("current_run"),
        "versions": len(ids),
    }


def run_paths(version_id: str, run_id: str) -> tuple[Path, Path]:
    if VERSION_RE.fullmatch(version_id) is None or RUN_RE.fullmatch(run_id) is None:
        raise RSIControlError("invalid version/run identity")
    run_dir = ROOT / "versions" / version_id / "runs" / run_id
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise RSIControlError("run does not exist")
    return run_dir, run_path


def attempt_paths(
    version_id: str, run_id: str, attempt_id: str
) -> tuple[Path, Path, Path]:
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise RSIControlError("invalid attempt identity")
    run_dir, run_path = run_paths(version_id, run_id)
    attempt_dir = run_dir / "attempts" / attempt_id
    attempt_path = attempt_dir / "attempt.json"
    if not attempt_path.is_file():
        raise RSIControlError("attempt does not exist")
    return run_path, attempt_dir, attempt_path


def bind_run(
    *,
    version_id: str,
    run_id: str,
    contract_source: Path,
    remote_run_root: str,
) -> dict[str, Any]:
    if not remote_run_root:
        raise RSIControlError("remote_run_root is required")
    run_dir, run_path = run_paths(version_id, run_id)
    run = dict(mapping(load_json(run_path), "run"))
    source = dict(mapping(load_json(contract_source), "run contract"))
    if source.get("schema_name") != "rsi.run_contract" or source.get("schema_version") != 1:
        raise RSIControlError("run contract schema drifted")
    if source.get("version_id") != version_id or source.get("run_id") != run_id:
        raise RSIControlError("run contract identity drifted")
    supplied_hash = source.pop("contract_sha256", None)
    contract_hash = object_sha256(source)
    if supplied_hash is not None and supplied_hash != contract_hash:
        raise RSIControlError("supplied run contract hash drifted")
    source["contract_sha256"] = contract_hash
    existing_hash = run.get("run_contract_sha256")
    if existing_hash is not None:
        if existing_hash != contract_hash or run.get("remote_run_root") != remote_run_root:
            raise RSIControlError("refusing to rebind a run to different semantics")
        validate_run_contract(run_dir / "run-contract.json", version_id, run_id)
        validate_control()
        return run
    contract_path = run_dir / "run-contract.json"
    if contract_path.exists():
        existing = mapping(load_json(contract_path), "run contract")
        if existing != source:
            raise RSIControlError("orphan run contract differs from requested semantics")
    else:
        atomic_write_json(contract_path, source)
    run["run_contract_path"] = "run-contract.json"
    run["run_contract_sha256"] = contract_hash
    run["remote_run_root"] = remote_run_root
    atomic_write_json(run_path, run)
    validate_control()
    return run


def _ledger_identity(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path, "operation ledger")
    return {
        "path": "logs/operations.jsonl",
        "records": len(rows),
        "head_sha256": rows[-1]["event_sha256"] if rows else None,
    }


def attempt_start(
    *,
    version_id: str,
    run_id: str,
    attempt_id: str,
    execution_identity: Mapping[str, Any],
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    run_path, attempt_dir, attempt_path = attempt_paths(version_id, run_id, attempt_id)
    run = mapping(load_json(run_path), "run")
    contract = run.get("run_contract_sha256")
    if not isinstance(contract, str) or not SHA_RE.fullmatch(contract):
        raise RSIControlError("attempt cannot start before bind-run")
    if not execution_identity:
        raise RSIControlError("execution_identity is required")
    attempt = dict(mapping(load_json(attempt_path), "attempt"))
    if attempt.get("status") == "running":
        if (
            attempt.get("run_contract_sha256") == contract
            and attempt.get("execution_identity") == execution_identity
        ):
            validate_control()
            return attempt
        raise RSIControlError("running attempt identity differs")
    if attempt.get("status") != "planned":
        raise RSIControlError("use retry-start for a terminal attempt")
    started = started_at_utc or now_utc()
    utc_timestamp(started, "attempt started_at_utc")
    ledger_path = attempt_dir / "logs/operations.jsonl"
    if ledger_path.exists() and ledger_path.read_text(encoding="utf-8"):
        raise RSIControlError("planned attempt has a non-empty operation ledger")
    if not ledger_path.exists():
        atomic_write_jsonl(ledger_path, [])
    attempt.update(
        {
            "status": "running",
            "run_contract_sha256": contract,
            "execution_identity": dict(execution_identity),
            "started_at_utc": started,
            "ended_at_utc": None,
            "exit_code": None,
            "consumed_gpu_hours": 0.0,
            "error": None,
            "event_log": _ledger_identity(ledger_path),
            "active_retry": None,
        }
    )
    atomic_write_json(attempt_path, attempt)
    validate_control()
    return attempt


def _validate_terminal_fields(
    status: str, exit_code: int, error: Mapping[str, Any] | None
) -> None:
    if status not in TERMINAL_ATTEMPT_STATUSES:
        raise RSIControlError("attempt finish status is invalid")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise RSIControlError("attempt exit_code is invalid")
    if status == "succeeded":
        if exit_code != 0 or error is not None:
            raise RSIControlError("successful attempt cannot contain an error")
        return
    if exit_code == 0 or error is None:
        raise RSIControlError("failed attempt requires non-zero exit and error")
    if not all(
        isinstance(error.get(field), str) and error.get(field)
        for field in ("layer", "code", "message")
    ) or not isinstance(error.get("retryable"), bool):
        raise RSIControlError("attempt error is incomplete")
    if status == "model_failed" and error.get("layer") == "infrastructure":
        raise RSIControlError("infrastructure failure cannot be a model failure")


def attempt_finish(
    *,
    version_id: str,
    run_id: str,
    attempt_id: str,
    status: str,
    exit_code: int,
    error: Mapping[str, Any] | None,
    ended_at_utc: str | None = None,
) -> dict[str, Any]:
    _run_path, _attempt_dir, attempt_path = attempt_paths(
        version_id, run_id, attempt_id
    )
    _validate_terminal_fields(status, exit_code, error)
    attempt = dict(mapping(load_json(attempt_path), "attempt"))
    if attempt.get("status") in TERMINAL_ATTEMPT_STATUSES:
        if (
            attempt.get("status") == status
            and attempt.get("exit_code") == exit_code
            and attempt.get("error") == error
        ):
            validate_control()
            return attempt
        raise RSIControlError("terminal attempt cannot be rewritten")
    if attempt.get("status") != "running" or attempt.get("active_retry") is not None:
        raise RSIControlError("attempt is not finishable")
    ended = ended_at_utc or now_utc()
    if utc_timestamp(ended, "attempt ended_at_utc") < utc_timestamp(
        attempt.get("started_at_utc"), "attempt started_at_utc"
    ):
        raise RSIControlError("attempt ended before it started")
    attempt.update(
        {
            "status": status,
            "ended_at_utc": ended,
            "exit_code": exit_code,
            "error": dict(error) if error is not None else None,
        }
    )
    atomic_write_json(attempt_path, attempt)
    validate_control()
    return attempt


def _update_version_operation_metrics(version_id: str) -> None:
    version_dir = ROOT / "versions" / version_id
    gpu_hours = 0.0
    eval_accesses = 0
    for run_dir in sorted((version_dir / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        run = mapping(load_json(run_dir / "run.json"), "run")
        for attempt_dir in sorted((run_dir / "attempts").iterdir()):
            if not attempt_dir.is_dir():
                continue
            totals = validate_attempt(
                attempt_dir, version_id, run_dir.name, run.get("run_contract_sha256")
            )
            gpu_hours += totals["gpu_hours"]
            eval_accesses += totals["eval_accesses"]
    metrics_path = version_dir / "metrics.json"
    metrics = dict(mapping(load_json(metrics_path), "version metrics"))
    observed = dict(mapping(metrics.get("observed"), "observed metrics"))
    observed["gpu_hours"] = round(gpu_hours, 9)
    observed["eval_accesses"] = eval_accesses
    metrics["observed"] = observed
    atomic_write_json(metrics_path, metrics)


def record_operation(
    *,
    version_id: str,
    run_id: str,
    attempt_id: str,
    event_id: str,
    operation_type: str,
    payload: Mapping[str, Any],
    gpu_hours: float,
    eval_accesses: int,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    run_path, attempt_dir, attempt_path = attempt_paths(version_id, run_id, attempt_id)
    run = mapping(load_json(run_path), "run")
    attempt = dict(mapping(load_json(attempt_path), "attempt"))
    if attempt.get("status") != "running":
        raise RSIControlError("operations may only be recorded on a running attempt")
    if operation_type not in OPERATION_TYPES or not event_id:
        raise RSIControlError("operation identity is invalid")
    if isinstance(gpu_hours, bool) or not isinstance(gpu_hours, (int, float)) or not math.isfinite(float(gpu_hours)) or gpu_hours < 0:
        raise RSIControlError("operation gpu_hours is invalid")
    nonnegative_int(eval_accesses, "operation eval_accesses")
    if (operation_type == "eval" and eval_accesses != 1) or (
        operation_type != "eval" and eval_accesses != 0
    ):
        raise RSIControlError("operation eval_accesses does not match its type")
    ledger_path = attempt_dir / "logs/operations.jsonl"
    rows = load_jsonl(ledger_path, "operation ledger")
    semantic = {
        "event_id": event_id,
        "operation_type": operation_type,
        "payload": dict(payload),
        "gpu_hours": round(float(gpu_hours), 9),
        "eval_accesses": eval_accesses,
    }
    for row in rows:
        if row.get("event_id") == event_id:
            if all(row.get(key) == value for key, value in semantic.items()):
                attempt["event_log"] = _ledger_identity(ledger_path)
                attempt["consumed_gpu_hours"] = round(
                    sum(float(item["gpu_hours"]) for item in rows), 9
                )
                atomic_write_json(attempt_path, attempt)
                _update_version_operation_metrics(version_id)
                validate_control()
                return row
            raise RSIControlError("operation event ID was reused with different evidence")
    created = created_at_utc or now_utc()
    utc_timestamp(created, "operation created_at_utc")
    event: dict[str, Any] = {
        "schema_name": "rsi.operation_event",
        "schema_version": 1,
        "sequence": len(rows) + 1,
        "previous_event_sha256": rows[-1]["event_sha256"] if rows else None,
        "version_id": version_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_contract_sha256": run.get("run_contract_sha256"),
        "created_at_utc": created,
        **semantic,
    }
    event["event_sha256"] = object_sha256(event)
    rows.append(event)
    atomic_write_jsonl(ledger_path, rows)
    attempt["event_log"] = _ledger_identity(ledger_path)
    attempt["consumed_gpu_hours"] = round(
        sum(float(row["gpu_hours"]) for row in rows), 9
    )
    atomic_write_json(attempt_path, attempt)
    _update_version_operation_metrics(version_id)
    validate_control()
    return event


def retry_start(
    *,
    version_id: str,
    run_id: str,
    attempt_id: str,
    remediation: str,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    run_path, attempt_dir, attempt_path = attempt_paths(version_id, run_id, attempt_id)
    run = mapping(load_json(run_path), "run")
    attempt = dict(mapping(load_json(attempt_path), "attempt"))
    if attempt.get("status") not in {
        "infrastructure_failed",
        "model_failed",
        "interrupted",
    }:
        raise RSIControlError("only a failed/interrupted attempt may be retried")
    if not remediation:
        raise RSIControlError("retry remediation is required")
    error = mapping(attempt.get("error"), "attempt error")
    retry_path = attempt_dir / "logs/retries.jsonl"
    rows = load_jsonl(retry_path, "retry log")
    ordinal = int(attempt.get("retry_count", 0)) + 1
    created = created_at_utc or now_utc()
    utc_timestamp(created, "retry created_at_utc")
    event = {
        "schema_name": "rsi.retry_event",
        "schema_version": 1,
        "event_type": "retry_started",
        "version_id": version_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "retry_ordinal": ordinal,
        "created_at_utc": created,
        "failure_layer": error["layer"],
        "failure_code": error["code"],
        "remediation": remediation,
        "training_semantics_changed": False,
        "run_contract_before": run.get("run_contract_sha256"),
        "run_contract_after": run.get("run_contract_sha256"),
    }
    rows.append(event)
    atomic_write_jsonl(retry_path, rows)
    attempt.update(
        {
            "status": "running",
            "ended_at_utc": None,
            "exit_code": None,
            "error": None,
            "retry_count": ordinal,
            "active_retry": ordinal,
        }
    )
    atomic_write_json(attempt_path, attempt)
    validate_control()
    return event


def retry_finish(
    *,
    version_id: str,
    run_id: str,
    attempt_id: str,
    status: str,
    exit_code: int,
    error: Mapping[str, Any] | None,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    run_path, attempt_dir, attempt_path = attempt_paths(version_id, run_id, attempt_id)
    run = mapping(load_json(run_path), "run")
    attempt = dict(mapping(load_json(attempt_path), "attempt"))
    ordinal = attempt.get("active_retry")
    if attempt.get("status") != "running" or not isinstance(ordinal, int):
        raise RSIControlError("no retry is active")
    _validate_terminal_fields(status, exit_code, error)
    created = created_at_utc or now_utc()
    if utc_timestamp(created, "retry created_at_utc") < utc_timestamp(
        attempt.get("started_at_utc"), "attempt started_at_utc"
    ):
        raise RSIControlError("retry ended before the attempt started")
    retry_path = attempt_dir / "logs/retries.jsonl"
    rows = load_jsonl(retry_path, "retry log")
    event = {
        "schema_name": "rsi.retry_event",
        "schema_version": 1,
        "event_type": "retry_finished",
        "version_id": version_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "retry_ordinal": ordinal,
        "created_at_utc": created,
        "run_contract_sha256": run.get("run_contract_sha256"),
        "outcome": status,
        "exit_code": exit_code,
    }
    rows.append(event)
    atomic_write_jsonl(retry_path, rows)
    attempt.update(
        {
            "status": status,
            "ended_at_utc": created,
            "exit_code": exit_code,
            "error": dict(error) if error is not None else None,
            "active_retry": None,
        }
    )
    atomic_write_json(attempt_path, attempt)
    validate_control()
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("status")
    score = subparsers.add_parser("score")
    score.add_argument("--values", type=Path, required=True)
    bind = subparsers.add_parser("bind-run")
    bind.add_argument("--version-id", required=True)
    bind.add_argument("--run-id", required=True)
    bind.add_argument("--contract", type=Path, required=True)
    bind.add_argument("--remote-run-root", required=True)
    start = subparsers.add_parser("attempt-start")
    start.add_argument("--version-id", required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--attempt-id", required=True)
    start.add_argument("--execution-identity", type=Path, required=True)
    start.add_argument("--started-at-utc")
    finish = subparsers.add_parser("attempt-finish")
    finish.add_argument("--version-id", required=True)
    finish.add_argument("--run-id", required=True)
    finish.add_argument("--attempt-id", required=True)
    finish.add_argument("--status", choices=sorted(TERMINAL_ATTEMPT_STATUSES), required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--error", type=Path)
    finish.add_argument("--ended-at-utc")
    retry_begin = subparsers.add_parser("retry-start")
    retry_begin.add_argument("--version-id", required=True)
    retry_begin.add_argument("--run-id", required=True)
    retry_begin.add_argument("--attempt-id", required=True)
    retry_begin.add_argument("--remediation", required=True)
    retry_begin.add_argument("--created-at-utc")
    retry_end = subparsers.add_parser("retry-finish")
    retry_end.add_argument("--version-id", required=True)
    retry_end.add_argument("--run-id", required=True)
    retry_end.add_argument("--attempt-id", required=True)
    retry_end.add_argument("--status", choices=sorted(TERMINAL_ATTEMPT_STATUSES), required=True)
    retry_end.add_argument("--exit-code", type=int, required=True)
    retry_end.add_argument("--error", type=Path)
    retry_end.add_argument("--created-at-utc")
    operation = subparsers.add_parser("record-operation")
    operation.add_argument("--version-id", required=True)
    operation.add_argument("--run-id", required=True)
    operation.add_argument("--attempt-id", required=True)
    operation.add_argument("--event-id", required=True)
    operation.add_argument("--operation-type", choices=sorted(OPERATION_TYPES), required=True)
    operation.add_argument("--payload", type=Path, required=True)
    operation.add_argument("--gpu-hours", type=float, default=0.0)
    operation.add_argument("--eval-accesses", type=int)
    operation.add_argument("--created-at-utc")
    args = parser.parse_args()
    if args.command in {"validate", "status"}:
        print(json.dumps(validate_control(), indent=2, sort_keys=True))
        return
    if args.command == "score":
        goal = mapping(load_json(ROOT / "goal.json"), "goal")
        values = mapping(load_json(args.values), "gate values")
        result: Any = compute_gate_summary(
            values, mapping(goal["hard_gates"], "hard gates")
        )
    elif args.command == "bind-run":
        result = bind_run(
            version_id=args.version_id,
            run_id=args.run_id,
            contract_source=args.contract,
            remote_run_root=args.remote_run_root,
        )
    elif args.command == "attempt-start":
        result = attempt_start(
            version_id=args.version_id,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            execution_identity=mapping(
                load_json(args.execution_identity), "execution identity"
            ),
            started_at_utc=args.started_at_utc,
        )
    elif args.command == "attempt-finish":
        result = attempt_finish(
            version_id=args.version_id,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            status=args.status,
            exit_code=args.exit_code,
            error=(mapping(load_json(args.error), "attempt error") if args.error else None),
            ended_at_utc=args.ended_at_utc,
        )
    elif args.command == "retry-start":
        result = retry_start(
            version_id=args.version_id,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            remediation=args.remediation,
            created_at_utc=args.created_at_utc,
        )
    elif args.command == "retry-finish":
        result = retry_finish(
            version_id=args.version_id,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            status=args.status,
            exit_code=args.exit_code,
            error=(mapping(load_json(args.error), "attempt error") if args.error else None),
            created_at_utc=args.created_at_utc,
        )
    else:
        eval_accesses = args.eval_accesses
        if eval_accesses is None:
            eval_accesses = 1 if args.operation_type == "eval" else 0
        result = record_operation(
            version_id=args.version_id,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            event_id=args.event_id,
            operation_type=args.operation_type,
            payload=mapping(load_json(args.payload), "operation payload"),
            gpu_hours=args.gpu_hours,
            eval_accesses=eval_accesses,
            created_at_utc=args.created_at_utc,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
