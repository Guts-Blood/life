#!/usr/bin/env python3
"""Independently validate the Day 22 experimental AI-assisted close bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
BOOTCAMP_ROOT = SCRIPT_PATH.parents[2]
DAY22_DIR = BOOTCAMP_ROOT / "day-22-preference-data"
if str(DAY22_DIR) not in sys.path:
    sys.path.insert(0, str(DAY22_DIR))

import day22_contract  # noqa: E402


class ExperimentalValidationError(ValueError):
    """The experimental-close bundle failed independent validation."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any, self_field: str | None = None) -> str:
    if self_field is not None:
        value = dict(value)
        value.pop(self_field, None)
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentalValidationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExperimentalValidationError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExperimentalValidationError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ExperimentalValidationError(f"blank row: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExperimentalValidationError(
                f"invalid row: {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ExperimentalValidationError(
                f"non-object row: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def verify_self(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or expected != object_sha256(value, field):
        raise ExperimentalValidationError(f"{label} self-hash mismatch")


def resolve(entry: Mapping[str, Any], label: str) -> Path:
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ExperimentalValidationError(f"{label} path is invalid")
    resolved = (BOOTCAMP_ROOT / path).resolve()
    try:
        resolved.relative_to(BOOTCAMP_ROOT)
    except ValueError as error:
        raise ExperimentalValidationError(f"{label} escapes bootcamp root") from error
    if not resolved.is_file():
        raise ExperimentalValidationError(f"{label} file is missing")
    if entry.get("file_sha256") != file_sha256(resolved):
        raise ExperimentalValidationError(f"{label} file hash mismatch")
    return resolved


def validate_score_totals(score: Mapping[str, Any], label: str) -> None:
    limits = {
        "semantic_correctness": (0, 5),
        "public_test_alignment": (0, 2),
        "instruction_compliance": (0, 1),
        "robustness": (0, 2),
    }
    if set(score) != {*limits, "total"}:
        raise ExperimentalValidationError(f"{label} fields drifted")
    total = 0
    for field, (minimum, maximum) in limits.items():
        value = score.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ExperimentalValidationError(f"{label}.{field} is not an integer")
        if not minimum <= value <= maximum:
            raise ExperimentalValidationError(f"{label}.{field} is out of range")
        total += value
    if score.get("total") != total:
        raise ExperimentalValidationError(f"{label}.total drifted")


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    verify_self(manifest, "manifest_sha256", "experimental manifest")
    if manifest.get("schema_name") != "day22.experimental_ai_assisted_close_manifest":
        raise ExperimentalValidationError("experimental manifest schema drifted")
    if manifest.get("status") != "completed_experimental_ai_assisted":
        raise ExperimentalValidationError("experimental close status drifted")
    if manifest.get("dataset_role") != "experimental_ai_assisted_dpo_input":
        raise ExperimentalValidationError("experimental dataset role drifted")
    if manifest.get("experimental_dpo_ready") is not True:
        raise ExperimentalValidationError("experimental ready flag is not true")
    if manifest.get("formal_human_review_ready") is not False:
        raise ExperimentalValidationError("formal human flag must remain false")
    if manifest.get("formal_dpo_ready") is not False:
        raise ExperimentalValidationError("formal DPO flag must remain false")
    if manifest.get("formal_blockers") != ["human_blind_review_not_completed"]:
        raise ExperimentalValidationError("formal blocker drifted")
    policy = manifest.get("policy", {})
    if policy.get("label_flip_allowed") is not False:
        raise ExperimentalValidationError("label flipping must remain forbidden")
    if policy.get("formal_gate_waived_for_experimental_track") is not True:
        raise ExperimentalValidationError("experimental waiver is absent")

    inputs = manifest.get("inputs", {})
    outputs = manifest.get("outputs", {})
    formal_manifest_path = resolve(inputs.get("formal_manifest", {}), "formal manifest")
    formal_pairs_path = resolve(inputs.get("formal_pairs", {}), "formal pairs")
    formal_split_path = resolve(inputs.get("formal_split_ids", {}), "formal split ids")
    audit_path = resolve(inputs.get("ai_assisted_audit", {}), "AI-assisted audit")
    blank_path = resolve(inputs.get("blank_worksheet", {}), "blank worksheet")
    key_path = resolve(inputs.get("concealed_key", {}), "concealed key")
    packet_path = resolve(inputs.get("adjudication_packet", {}), "adjudication packet")
    scores_path = resolve(inputs.get("adjudication_scores", {}), "adjudication scores")
    output_pairs_path = resolve(outputs.get("pairs", {}), "experimental pairs")
    output_split_path = resolve(outputs.get("split_ids", {}), "experimental split ids")
    resolve(outputs.get("report", {}), "experimental report")

    formal_manifest = load_json(formal_manifest_path)
    verify_self(formal_manifest, "manifest_sha256", "source formal manifest")
    if inputs["formal_manifest"].get("manifest_sha256") != formal_manifest.get(
        "manifest_sha256"
    ):
        raise ExperimentalValidationError("source formal manifest identity drifted")
    if formal_manifest.get("machine_ready") is not True:
        raise ExperimentalValidationError("source formal bundle is not machine-ready")
    if formal_manifest.get("formal_dpo_blockers") != ["human_blind_review_pending"]:
        raise ExperimentalValidationError("source formal blocker drifted")

    source_pairs = load_jsonl(formal_pairs_path)
    source_summary = day22_contract.validate_manifest(source_pairs, mode="final")
    if source_summary["records"] != 200:
        raise ExperimentalValidationError("source pair count drifted")
    source_by_id = {row["pair_id"]: row for row in source_pairs}
    if len(source_by_id) != 200:
        raise ExperimentalValidationError("source pair IDs are not unique")

    audit = load_json(audit_path)
    verify_self(audit, "audit_sha256", "AI-assisted audit")
    if inputs["ai_assisted_audit"].get("audit_sha256") != audit.get(
        "audit_sha256"
    ):
        raise ExperimentalValidationError("AI-assisted audit identity drifted")
    if audit.get("status") != "audit_only_not_formal_human_review":
        raise ExperimentalValidationError("AI-assisted audit status drifted")
    contested = {
        row["pair_id"]
        for row in audit.get("pair_results", [])
        if not row.get("both_directional") or not row.get("position_consistent")
    }
    if len(contested) != 11:
        raise ExperimentalValidationError("contested pair count drifted")

    blank_by_id = {
        row["review_item_id"]: row for row in load_jsonl(blank_path)
    }
    key = load_json(key_path)
    verify_self(key, "key_sha256", "concealed key")
    if inputs["concealed_key"].get("key_sha256") != key.get("key_sha256"):
        raise ExperimentalValidationError("concealed key identity drifted")
    key_by_id = {row["review_item_id"]: row for row in key.get("items", [])}
    packet = load_jsonl(packet_path)
    scores = load_jsonl(scores_path)
    if len(packet) != len(scores) or len(packet) != 11:
        raise ExperimentalValidationError("packet/score row count drifted")

    expected_decisions: list[dict[str, Any]] = []
    excluded_ids: set[str] = set()
    seen_pairs: set[str] = set()
    for index, (case, score) in enumerate(zip(packet, scores)):
        verify_self(case, "case_sha256", f"packet[{index}]")
        review_id = case.get("review_item_id")
        if review_id not in key_by_id or review_id not in blank_by_id:
            raise ExperimentalValidationError(f"packet[{index}] review id is unknown")
        item = key_by_id[review_id]
        pair_id = item.get("pair_id")
        if item.get("presentation") != "primary" or pair_id not in contested:
            raise ExperimentalValidationError(f"packet[{index}] is not contested primary")
        if pair_id in seen_pairs:
            raise ExperimentalValidationError(f"duplicate packet pair: {pair_id}")
        seen_pairs.add(pair_id)
        source_case = blank_by_id[review_id]
        for field in ("prompt", "response_a", "response_b"):
            if case.get(field) != source_case.get(field):
                raise ExperimentalValidationError(f"packet[{index}] text drifted")
        source_tests = source_by_id[pair_id]["tests"]
        if case.get("public_tests", {}).get("test_list") != source_tests.get(
            "test_list"
        ):
            raise ExperimentalValidationError(f"packet[{index}] tests drifted")
        if score.get("adjudication_id") != case.get("adjudication_id"):
            raise ExperimentalValidationError(f"score[{index}] adjudication id drifted")
        if score.get("review_item_id") != review_id:
            raise ExperimentalValidationError(f"score[{index}] review id drifted")
        validate_score_totals(score.get("scores_a", {}), f"score[{index}].A")
        validate_score_totals(score.get("scores_b", {}), f"score[{index}].B")
        decision = score.get("decision")
        if decision == "prefer_A":
            if score["scores_a"]["total"] <= score["scores_b"]["total"]:
                raise ExperimentalValidationError(f"score[{index}] prefer_A unsupported")
            semantic = item["a_side"]
        elif decision == "prefer_B":
            if score["scores_b"]["total"] <= score["scores_a"]["total"]:
                raise ExperimentalValidationError(f"score[{index}] prefer_B unsupported")
            semantic = item["b_side"]
        elif decision == "exclude":
            semantic = "exclude"
        else:
            raise ExperimentalValidationError(f"score[{index}] decision is invalid")
        action = "keep" if semantic == "chosen" else "exclude"
        if action == "exclude":
            excluded_ids.add(pair_id)
        expected_decisions.append(
            {
                "adjudication_id": case["adjudication_id"],
                "review_item_id": review_id,
                "pair_id": pair_id,
                "decision": decision,
                "semantic_decision": semantic,
                "action": action,
                "score_a_total": score["scores_a"]["total"],
                "score_b_total": score["scores_b"]["total"],
                "confidence": score["confidence"],
                "issue_type": score["issue_type"],
                "rationale_zh": score["rationale_zh"],
            }
        )
    if seen_pairs != contested:
        raise ExperimentalValidationError("packet contested coverage drifted")
    if manifest.get("adjudication_decisions") != expected_decisions:
        raise ExperimentalValidationError("manifest adjudication decisions drifted")
    if excluded_ids:
        if output_pairs_path == formal_pairs_path:
            raise ExperimentalValidationError(
                "filtered experimental pairs must not alias formal pairs"
            )
    elif output_pairs_path != formal_pairs_path:
        raise ExperimentalValidationError(
            "zero-exclusion experimental close must reuse formal pairs"
        )

    expected_pairs = [
        row for row in source_pairs if row["pair_id"] not in excluded_ids
    ]
    output_pairs = load_jsonl(output_pairs_path)
    if output_pairs != expected_pairs:
        raise ExperimentalValidationError("experimental pairs are not the exact source subset")
    output_summary = day22_contract.validate_manifest(output_pairs, mode="final")
    if outputs["pairs"].get("records") != len(output_pairs):
        raise ExperimentalValidationError("experimental output pair count drifted")

    source_split = load_json(formal_split_path)
    verify_self(source_split, "split_ids_sha256", "source formal split ids")
    output_split = load_json(output_split_path)
    verify_self(output_split, "split_ids_sha256", "experimental split ids")
    accepted_ids = {row["pair_id"] for row in output_pairs}
    family_by_pair = {
        row["pair_id"]: row["family_keys"]["problem"] for row in output_pairs
    }
    for split in ("train", "dev", "heldout"):
        expected_ids = [
            pair_id for pair_id in source_split[split] if pair_id in accepted_ids
        ]
        if output_split.get(split) != expected_ids:
            raise ExperimentalValidationError(f"experimental {split} IDs drifted")
        expected_families = [family_by_pair[pair_id] for pair_id in expected_ids]
        if output_split.get("family_ids", {}).get(split) != expected_families:
            raise ExperimentalValidationError(f"experimental {split} families drifted")

    counts = manifest.get("counts", {})
    action_counts = {
        "keep": sum(row["action"] == "keep" for row in expected_decisions),
        "exclude": sum(row["action"] == "exclude" for row in expected_decisions),
    }
    expected_counts = {
        "source_pairs": 200,
        "contested_pairs": 11,
        "adjudication_keep": action_counts["keep"],
        "adjudication_exclude": action_counts["exclude"],
        "experimental_pairs": len(output_pairs),
        "split_counts": output_summary["split_counts"],
    }
    if counts != expected_counts:
        raise ExperimentalValidationError("experimental manifest counts drifted")
    identities = manifest.get("content_identities", {})
    if identities.get("ordered_pair_hashes_sha256") != output_summary.get(
        "ordered_pair_hashes_sha256"
    ):
        raise ExperimentalValidationError("ordered pair identity drifted")
    if identities.get("split_ids_sha256") != output_split.get("split_ids_sha256"):
        raise ExperimentalValidationError("split identity drifted")
    expected_ids_hash = object_sha256([row["pair_id"] for row in output_pairs])
    if identities.get("accepted_pair_ids_sha256") != expected_ids_hash:
        raise ExperimentalValidationError("accepted pair ID identity drifted")

    return {
        "status": "valid_completed_experimental_ai_assisted",
        "experimental_dpo_ready": True,
        "formal_dpo_ready": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_pairs": 200,
        "experimental_pairs": len(output_pairs),
        "adjudication_keep": action_counts["keep"],
        "adjudication_exclude": action_counts["exclude"],
        "split_counts": output_summary["split_counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_manifest(args.manifest.expanduser().resolve())
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ExperimentalValidationError, day22_contract.Day22ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
