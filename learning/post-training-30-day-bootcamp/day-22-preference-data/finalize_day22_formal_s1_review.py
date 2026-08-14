#!/usr/bin/env python3
"""Finalize a completed Day 22 blind review without mutating frozen inputs.

The completed worksheet is an answer-bearing copy of the blank worksheet bound
by the pending formal manifest.  The concealed key is consulted only here,
after annotation.  The script always emits a content-bound review audit.  It
emits a separate ready manifest only when every fail-closed readiness gate
passes; the pending manifest and blank worksheet are never overwritten.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
BOOTCAMP_ROOT = SCRIPT_PATH.parent.parent
PROTOCOL_PATH = SCRIPT_PATH.parent / "FORMAL-S1-BLIND-REVIEW-PROTOCOL.md"

PENDING_MANIFEST_SCHEMA = "day22.formal_s1_assembly_manifest"
READY_MANIFEST_SCHEMA = "day22.formal_s1_ready_manifest"
AUDIT_SCHEMA = "day22.formal_s1_human_review_audit"
REVIEW_SCHEMA = "day22.blind_review_item"
KEY_SCHEMA = "day22.blind_review_concealed_key"
REVIEW_VERDICTS = {"A", "B", "tie", "ambiguous", "reject"}
CONFIDENCES = {"low", "medium", "high"}
ANSWER_FIELDS = {"verdict", "confidence", "notes"}
REQUIRED_UNIQUE_PAIRS = 50
REQUIRED_PRESENTATIONS = REQUIRED_UNIQUE_PAIRS * 2
MIN_POSITION_CONSISTENCY = 0.90
MIN_VERIFIER_AGREEMENT = 0.90
MIN_REMAINING_PAIRS = 200
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReviewFinalizationError(ValueError):
    """A review input, identity, or lifecycle invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReviewFinalizationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def portable_artifact_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(BOOTCAMP_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewFinalizationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str) and "\x00" not in value, f"{label} must be text without NUL")
    if not allow_empty:
        _require(bool(value.strip()), f"{label} must be non-empty text")
    return str(value)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase bare SHA-256",
    )
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewFinalizationError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReviewFinalizationError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReviewFinalizationError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from error
        _require(isinstance(row, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"JSONL is empty: {path}")
    return rows


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = _sha256(value.get(field), f"{label}.{field}")
    actual = object_sha256({key: item for key, item in value.items() if key != field})
    _require(actual == expected, f"{label}.{field} does not bind its contents")
    return expected


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    result[field] = object_sha256(result)
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


def _pending_review_paths(
    manifest: Mapping[str, Any], manifest_path: Path
) -> tuple[Path, Path, Path]:
    outputs = _mapping(manifest.get("output_files"), "pending manifest.output_files")
    found: dict[str, Path] = {}
    for name, raw_identity in outputs.items():
        identity = _mapping(raw_identity, f"pending manifest.output_files.{name}")
        path = _resolve_identity(
            identity,
            manifest_path=manifest_path,
            label=f"pending manifest.output_files.{name}",
        )
        try:
            content = path.read_text(encoding="utf-8")
            try:
                first = json.loads(content)
            except json.JSONDecodeError:
                first = json.loads(content.splitlines()[0])
        except (OSError, IndexError, json.JSONDecodeError) as error:
            raise ReviewFinalizationError(f"cannot identify pending output: {path}") from error
        schema = _mapping(first, f"pending output {name}").get("schema_name")
        if schema == REVIEW_SCHEMA:
            found["blank"] = path
        elif schema == KEY_SCHEMA:
            found["key"] = path
        elif schema == "day22.coding_preference_pair":
            found["pairs"] = path
    _require(set(found) == {"blank", "key", "pairs"}, "pending review outputs are incomplete")
    return found["blank"], found["key"], found["pairs"]


def _index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = _text(row.get(key), f"{label}[{index}].{key}")
        _require(identity not in result, f"duplicate {label} {key}: {identity}")
        result[identity] = row
    return result


def _validate_completed_rows(
    blank_rows: Sequence[Mapping[str, Any]],
    completed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    _require(len(blank_rows) == REQUIRED_PRESENTATIONS, "blank worksheet must have 100 rows")
    _require(len(completed_rows) == len(blank_rows), "completed worksheet row count drifted")
    blank = _index_unique(blank_rows, "review_item_id", "blank review row")
    completed = _index_unique(completed_rows, "review_item_id", "completed review row")
    _require(set(completed) == set(blank), "completed worksheet membership drifted")
    for item_id, row in completed.items():
        source = blank[item_id]
        _require(set(row) == set(source), f"completed review {item_id} fields drifted")
        for field in set(source) - ANSWER_FIELDS:
            _require(row.get(field) == source.get(field), f"completed review {item_id}.{field} drifted")
        verdict = row.get("verdict")
        confidence = row.get("confidence")
        _require(verdict in REVIEW_VERDICTS, f"completed review {item_id}.verdict is invalid")
        _require(confidence in CONFIDENCES, f"completed review {item_id}.confidence is invalid")
        notes = _text(row.get("notes"), f"completed review {item_id}.notes", allow_empty=True)
        if verdict in {"tie", "ambiguous", "reject"}:
            _require(bool(notes.strip()), f"completed review {item_id}.notes is required for {verdict}")
    return completed


def _semantic_side(verdict: str, key_item: Mapping[str, Any]) -> str:
    if verdict == "A":
        return _text(key_item.get("a_side"), "review key a_side")
    if verdict == "B":
        return _text(key_item.get("b_side"), "review key b_side")
    return verdict


def finalize_review(
    pending_manifest_path: Path,
    completed_worksheet_path: Path,
    *,
    protocol_path: Path = PROTOCOL_PATH,
    review_audit_output_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a sealed audit and, only on success, a separate ready manifest."""

    pending_manifest_path = pending_manifest_path.expanduser().resolve()
    completed_worksheet_path = completed_worksheet_path.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    pending = _load_json(pending_manifest_path)
    pending_sha = _verify_self_hash(pending, "manifest_sha256", "pending manifest")
    _require(pending.get("schema_name") == PENDING_MANIFEST_SCHEMA, "pending manifest schema drifted")
    _require(pending.get("formal_dpo_ready") is False, "input manifest is not pending review")
    _require(
        pending.get("formal_dpo_blockers") == ["human_blind_review_pending"],
        "input manifest has unexpected blockers",
    )
    blank_path, key_path, pairs_path = _pending_review_paths(pending, pending_manifest_path)
    blank_rows = _load_jsonl(blank_path)
    completed_rows = _load_jsonl(completed_worksheet_path)
    completed = _validate_completed_rows(blank_rows, completed_rows)

    key = _load_json(key_path)
    key_sha = _verify_self_hash(key, "key_sha256", "concealed review key")
    _require(key.get("schema_name") == KEY_SCHEMA, "concealed review key schema drifted")
    _require(key.get("handling") == "keep_separate_from_reviewer_worksheet", "concealed review handling drifted")
    _require(key.get("unique_pairs") == REQUIRED_UNIQUE_PAIRS, "review key unique pair count drifted")
    _require(key.get("presentations") == REQUIRED_PRESENTATIONS, "review key presentation count drifted")
    key_items = [_mapping(item, "review key item") for item in key.get("items", [])]
    by_item = _index_unique(key_items, "review_item_id", "review key item")
    _require(set(by_item) == set(completed), "review key/worksheet membership drifted")

    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item_id, item in by_item.items():
        pair_id = _text(item.get("pair_id"), f"review key {item_id}.pair_id")
        presentation = item.get("presentation")
        _require(presentation in {"primary", "swapped"}, f"review key {item_id} presentation is invalid")
        _require({item.get("a_side"), item.get("b_side")} == {"chosen", "rejected"}, f"review key {item_id} side map is invalid")
        row = completed[item_id]
        by_pair[pair_id][str(presentation)] = {
            "review_item_id": item_id,
            "display_verdict": row["verdict"],
            "semantic_conclusion": _semantic_side(str(row["verdict"]), item),
            "confidence": row["confidence"],
            "notes": row["notes"],
        }
    _require(len(by_pair) == REQUIRED_UNIQUE_PAIRS, "review unique pair coverage drifted")

    decisions: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    position_consistent = 0
    verifier_chosen_agreements = 0
    eligible_reviewed = 0
    for pair_id in sorted(by_pair):
        presentations = by_pair[pair_id]
        _require(set(presentations) == {"primary", "swapped"}, f"review pair {pair_id} needs primary and swapped judgments")
        primary = presentations["primary"]
        swapped = presentations["swapped"]
        semantic = [primary["semantic_conclusion"], swapped["semantic_conclusion"]]
        reasons: list[str] = []
        if any(value in {"tie", "ambiguous", "reject"} for value in semantic):
            reasons.append("non_directional_or_rejected_judgment")
        if semantic[0] != semantic[1]:
            reasons.append("position_swap_disagreement")
        else:
            position_consistent += 1
        if not reasons:
            eligible_reviewed += 1
            if semantic[0] == "chosen":
                verifier_chosen_agreements += 1
            else:
                reasons.append("verifier_rejected_preferred")
        for reason in reasons:
            excluded[reason] += 1
        decision = {
            "pair_id": pair_id,
            "primary": primary,
            "swapped": swapped,
            "position_consistent": semantic[0] == semantic[1],
            "verifier_chosen_agreement": semantic[0] == semantic[1] == "chosen",
            "included": not reasons,
            "exclusion_reasons": reasons,
        }
        decisions.append(_seal(decision, "decision_sha256"))

    position_rate = position_consistent / REQUIRED_UNIQUE_PAIRS
    agreement_rate = (
        verifier_chosen_agreements / eligible_reviewed if eligible_reviewed else 0.0
    )
    total_pairs = int(_mapping(pending.get("counts"), "pending manifest counts").get("formal_pairs", 0))
    excluded_pair_ids = sorted(
        decision["pair_id"] for decision in decisions if not decision["included"]
    )
    remaining_pairs = total_pairs - len(excluded_pair_ids)
    gates = {
        "all_100_presentations_completed": {
            "passed": len(completed) == REQUIRED_PRESENTATIONS,
            "observed": len(completed),
            "required": REQUIRED_PRESENTATIONS,
        },
        "position_consistency_at_least_90_percent": {
            "passed": position_rate >= MIN_POSITION_CONSISTENCY,
            "observed": position_rate,
            "required": MIN_POSITION_CONSISTENCY,
        },
        "verifier_chosen_agreement_at_least_90_percent": {
            "passed": agreement_rate >= MIN_VERIFIER_AGREEMENT,
            "observed": agreement_rate,
            "denominator": eligible_reviewed,
            "required": MIN_VERIFIER_AGREEMENT,
        },
        "remaining_pairs_at_least_200": {
            "passed": remaining_pairs >= MIN_REMAINING_PAIRS,
            "observed": remaining_pairs,
            "required": MIN_REMAINING_PAIRS,
        },
        "bound_pair_file_has_no_exclusions": {
            "passed": not excluded_pair_ids,
            "observed_exclusions": len(excluded_pair_ids),
            "reason": "ready manifest binds the unchanged pending pair file",
        },
    }
    ready = all(gate["passed"] is True for gate in gates.values())
    _require(protocol_path.is_file(), f"review protocol is missing: {protocol_path}")
    audit: dict[str, Any] = {
        "schema_name": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "pass" if ready else "blocked",
        "formal_dpo_ready": ready,
        "pending_manifest_sha256": pending_sha,
        "blank_worksheet_sha256": file_sha256(blank_path),
        "completed_worksheet_sha256": file_sha256(completed_worksheet_path),
        "concealed_key_sha256": key_sha,
        "protocol": {
            "name": "day22.formal_s1_blind_review_protocol.v1",
            "file_sha256": file_sha256(protocol_path),
        },
        "policy": {
            "required_presentations": REQUIRED_PRESENTATIONS,
            "required_unique_pairs": REQUIRED_UNIQUE_PAIRS,
            "minimum_position_consistency": MIN_POSITION_CONSISTENCY,
            "minimum_verifier_chosen_agreement": MIN_VERIFIER_AGREEMENT,
            "minimum_remaining_pairs": MIN_REMAINING_PAIRS,
        },
        "counts": {
            "completed_presentations": len(completed),
            "reviewed_unique_pairs": len(decisions),
            "position_consistent_pairs": position_consistent,
            "directionally_eligible_reviewed_pairs": eligible_reviewed,
            "verifier_chosen_agreements": verifier_chosen_agreements,
            "excluded_reviewed_pairs": len(excluded_pair_ids),
            "formal_pairs_before_review": total_pairs,
            "formal_pairs_after_review": remaining_pairs,
        },
        "rates": {
            "position_consistency": position_rate,
            "verifier_chosen_agreement": agreement_rate,
        },
        "exclusion_reason_counts": dict(sorted(excluded.items())),
        "excluded_pair_ids": excluded_pair_ids,
        "gates": gates,
        "pair_decisions": decisions,
    }
    audit = _seal(audit, "review_audit_sha256")
    if not ready:
        return audit, None

    _require(review_audit_output_path is not None, "ready review needs an audit output path")
    review_audit_output_path = review_audit_output_path.expanduser().resolve()

    outputs = _mapping(pending.get("output_files"), "pending manifest outputs")
    pair_output_identity = next(
        copy.deepcopy(dict(_mapping(identity, f"pending output {name}")))
        for name, identity in outputs.items()
        if _resolve_identity(
            _mapping(identity, f"pending output {name}"),
            manifest_path=pending_manifest_path,
            label=f"pending output {name}",
        )
        == pairs_path
    )
    ready_manifest: dict[str, Any] = {
        "schema_name": READY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "formal_dpo_ready",
        "dataset_role": "formal_s1_dpo_pairs_human_review_approved",
        "readiness": "READY",
        "formal_dpo_ready": True,
        "formal_dpo_blockers": [],
        "pending_manifest": {
            "path": portable_artifact_path(pending_manifest_path),
            "file_sha256": file_sha256(pending_manifest_path),
            "manifest_sha256": pending_sha,
        },
        "review_audit": {
            "path": portable_artifact_path(review_audit_output_path),
            "file_sha256": hashlib.sha256(_json_bytes(audit)).hexdigest(),
            "review_audit_sha256": audit["review_audit_sha256"],
        },
        "completed_worksheet": {
            "path": portable_artifact_path(completed_worksheet_path),
            "file_sha256": file_sha256(completed_worksheet_path),
        },
        "review_protocol_sha256": file_sha256(protocol_path),
        "preference_pairs": pair_output_identity,
        "counts": copy.deepcopy(audit["counts"]),
        "rates": copy.deepcopy(audit["rates"]),
    }
    ready_manifest = _seal(ready_manifest, "ready_manifest_sha256")
    return audit, ready_manifest


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_atomic(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _require(not resolved.exists(), f"refusing to overwrite: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise ReviewFinalizationError(f"output appeared while writing: {resolved}") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending-manifest", type=Path, required=True)
    parser.add_argument("--completed-worksheet", type=Path, required=True)
    parser.add_argument("--review-audit-output", type=Path, required=True)
    parser.add_argument("--ready-manifest-output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit_output = args.review_audit_output.expanduser().resolve()
        ready_output = args.ready_manifest_output.expanduser().resolve()
        _require(audit_output != ready_output, "review audit and ready manifest outputs must differ")
        _require(not audit_output.exists(), f"refusing to overwrite: {audit_output}")
        _require(not ready_output.exists(), f"refusing to overwrite: {ready_output}")
        audit, ready_manifest = finalize_review(
            args.pending_manifest,
            args.completed_worksheet,
            protocol_path=args.protocol,
            review_audit_output_path=audit_output,
        )
        _write_atomic(audit_output, _json_bytes(audit))
        if ready_manifest is None:
            print(
                json.dumps(
                    {
                        "status": "human_review_complete_readiness_blocked",
                        "formal_dpo_ready": False,
                        "review_audit": str(audit_output),
                        "failed_gates": sorted(
                            name
                            for name, gate in audit["gates"].items()
                            if gate["passed"] is not True
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 3
        _write_atomic(ready_output, _json_bytes(ready_manifest))
        print(
            json.dumps(
                {
                    "status": "formal_dpo_ready",
                    "formal_dpo_ready": True,
                    "review_audit": str(audit_output),
                    "ready_manifest": str(ready_output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (ReviewFinalizationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
