#!/usr/bin/env python3
"""Close Day 22 on an explicit AI-assisted experimental DPO track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import day22_contract


SCRIPT_PATH = Path(__file__).resolve()
BOOTCAMP_ROOT = SCRIPT_PATH.parent.parent
SCORE_KEYS = {
    "semantic_correctness",
    "public_test_alignment",
    "instruction_compliance",
    "robustness",
    "total",
}
SCORE_LIMITS = {
    "semantic_correctness": (0, 5),
    "public_test_alignment": (0, 2),
    "instruction_compliance": (0, 1),
    "robustness": (0, 2),
}
SCORE_ROW_KEYS = {
    "schema_name",
    "schema_version",
    "adjudication_id",
    "review_item_id",
    "scores_a",
    "scores_b",
    "decision",
    "confidence",
    "issue_type",
    "rationale_zh",
}
ISSUE_TYPES = {
    "none",
    "prompt_test_mismatch",
    "ambiguous_spec",
    "both_incorrect",
    "boundary_or_type_risk",
}


class ExperimentalCloseError(ValueError):
    """An experimental-close input or invariant failed validation."""


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
        raise ExperimentalCloseError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExperimentalCloseError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExperimentalCloseError(f"cannot read JSONL: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ExperimentalCloseError(f"blank JSONL line: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExperimentalCloseError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ExperimentalCloseError(
                f"non-object JSONL row: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or expected != object_sha256(value, field):
        raise ExperimentalCloseError(f"{label} self-hash mismatch")


def relative_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(BOOTCAMP_ROOT))
    except ValueError as error:
        raise ExperimentalCloseError(
            f"artifact must be inside bootcamp root: {path}"
        ) from error


def validate_score(score: Mapping[str, Any], label: str) -> None:
    if set(score) != SCORE_KEYS:
        raise ExperimentalCloseError(f"{label} score fields drifted")
    subtotal = 0
    for key, (minimum, maximum) in SCORE_LIMITS.items():
        value = score.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ExperimentalCloseError(f"{label}.{key} is not an integer")
        if not minimum <= value <= maximum:
            raise ExperimentalCloseError(f"{label}.{key} is out of range")
        subtotal += value
    if score.get("total") != subtotal:
        raise ExperimentalCloseError(f"{label}.total does not equal its dimensions")


def validate_scores(
    packet: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]]
) -> None:
    if len(packet) != 11 or len(scores) != len(packet):
        raise ExperimentalCloseError("packet and scores must contain exactly 11 rows")
    for index, (case, score) in enumerate(zip(packet, scores)):
        if case.get("schema_name") != "day22.codex_adjudication_case":
            raise ExperimentalCloseError(f"packet[{index}] schema drifted")
        verify_self_hash(case, "case_sha256", f"packet[{index}]")
        if set(score) != SCORE_ROW_KEYS:
            raise ExperimentalCloseError(f"score[{index}] fields drifted")
        if score.get("schema_name") != "day22.codex_adjudication_score":
            raise ExperimentalCloseError(f"score[{index}] schema drifted")
        if score.get("schema_version") != 1:
            raise ExperimentalCloseError(f"score[{index}] version drifted")
        for key in ("adjudication_id", "review_item_id"):
            if score.get(key) != case.get(key):
                raise ExperimentalCloseError(f"score[{index}] {key} drifted")
        validate_score(score.get("scores_a", {}), f"score[{index}].scores_a")
        validate_score(score.get("scores_b", {}), f"score[{index}].scores_b")
        decision = score.get("decision")
        if decision not in {"prefer_A", "prefer_B", "exclude"}:
            raise ExperimentalCloseError(f"score[{index}] decision is invalid")
        total_a = score["scores_a"]["total"]
        total_b = score["scores_b"]["total"]
        if decision == "prefer_A" and total_a <= total_b:
            raise ExperimentalCloseError(f"score[{index}] prefer_A lacks score support")
        if decision == "prefer_B" and total_b <= total_a:
            raise ExperimentalCloseError(f"score[{index}] prefer_B lacks score support")
        if decision == "exclude" and score.get("issue_type") == "none":
            raise ExperimentalCloseError(f"score[{index}] exclusion lacks issue type")
        if score.get("confidence") not in {"low", "medium", "high"}:
            raise ExperimentalCloseError(f"score[{index}] confidence is invalid")
        if score.get("issue_type") not in ISSUE_TYPES:
            raise ExperimentalCloseError(f"score[{index}] issue type is invalid")
        if not isinstance(score.get("rationale_zh"), str) or not score["rationale_zh"].strip():
            raise ExperimentalCloseError(f"score[{index}] rationale is empty")


def render_report(
    *,
    decisions: Sequence[Mapping[str, Any]],
    source_count: int,
    accepted_count: int,
    split_counts: Mapping[str, int],
) -> str:
    excluded = [row for row in decisions if row["action"] == "exclude"]
    lines = [
        "# Day 22 experimental AI-assisted close",
        "",
        "状态：`completed_experimental_ai_assisted`。这是用户明确授权的实验关闭路径，不是 formal human-reviewed readiness。",
        "",
        "## 关闭结果",
        "",
        f"- 原 machine-verified pairs：{source_count}",
        f"- Codex 复核争议 pairs：{len(decisions)}",
        f"- 保留为实验性 DPO 输入：{accepted_count}",
        f"- 裁决后剔除：{len(excluded)}",
        f"- split：train/dev/heldout = {split_counts['train']}/{split_counts['dev']}/{split_counts['heldout']}",
        "- Formal human-review readiness：`false`（没有伪装成人工审阅）。",
        "",
        "## Codex 争议裁决",
        "",
        "| case | pair | A/B 分数 | decision | 映射 | action | issue | rationale |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for row in decisions:
        rationale = str(row["rationale_zh"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{row['adjudication_id']}` | `{row['pair_id']}` | "
            f"{row['score_a_total']}/{row['score_b_total']} | {row['decision']} | "
            f"{row['semantic_decision']} | {row['action']} | {row['issue_type']} | "
            f"{rationale} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 原 formal manifest、pair 文件与 human-review blocker 保持不变。",
            "- 实验输入只做子集过滤；每个保留 pair 的文本、执行证据、processor audit 与 pair hash 均与 machine-verified source 完全相同。",
            "- Codex 若偏好 verifier-rejected 一侧，只会导致该 pair 被剔除，不会翻转 preference label。",
            "",
        ]
    )
    return "\n".join(lines)


def write_atomic(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--formal-pairs", type=Path, required=True)
    parser.add_argument("--formal-split-ids", type=Path, required=True)
    parser.add_argument("--ai-audit", type=Path, required=True)
    parser.add_argument("--blank-worksheet", type=Path, required=True)
    parser.add_argument("--concealed-key", type=Path, required=True)
    parser.add_argument("--adjudication-packet", type=Path, required=True)
    parser.add_argument("--adjudication-scores", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path)
    parser.add_argument("--split-ids-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = [
        args.split_ids_output,
        args.manifest_output,
        args.report_output,
    ]
    if args.pairs_output is not None:
        outputs.append(args.pairs_output)
    resolved_outputs = [path.expanduser().resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        print("error: output paths must be distinct", file=os.sys.stderr)
        return 2
    if not args.overwrite and any(path.exists() for path in resolved_outputs):
        print("error: refusing to overwrite an output", file=os.sys.stderr)
        return 2
    try:
        formal_manifest = load_json(args.formal_manifest)
        verify_self_hash(formal_manifest, "manifest_sha256", "formal manifest")
        if not formal_manifest.get("machine_ready"):
            raise ExperimentalCloseError("formal source is not machine-ready")
        if formal_manifest.get("formal_dpo_blockers") != [
            "human_blind_review_pending"
        ]:
            raise ExperimentalCloseError("formal source blocker drifted")
        expected_files = formal_manifest.get("output_files", {})
        for path, filename in (
            (args.formal_pairs, "day22-qwen35-formal-s1-preference-pairs.jsonl"),
            (args.formal_split_ids, "day22-qwen35-formal-s1-split-ids.json"),
            (args.blank_worksheet, "day22-qwen35-formal-s1-blind-review.jsonl"),
            (args.concealed_key, "day22-qwen35-formal-s1-blind-review-key.json"),
        ):
            expected = expected_files.get(filename, {}).get("file_sha256")
            if expected != file_sha256(path):
                raise ExperimentalCloseError(f"formal source file hash drifted: {filename}")

        audit = load_json(args.ai_audit)
        verify_self_hash(audit, "audit_sha256", "AI-assisted audit")
        if audit.get("status") != "audit_only_not_formal_human_review":
            raise ExperimentalCloseError("AI-assisted audit status drifted")
        if audit.get("counts", {}).get("human_presentations") != 10:
            raise ExperimentalCloseError("expected exactly 10 human presentations")
        if audit.get("counts", {}).get("subagent_presentations") != 90:
            raise ExperimentalCloseError("expected exactly 90 subagent presentations")
        contested = {
            row["pair_id"]
            for row in audit.get("pair_results", [])
            if not row.get("both_directional") or not row.get("position_consistent")
        }
        if len(contested) != 11:
            raise ExperimentalCloseError("expected exactly 11 contested pairs")

        blank_rows = load_jsonl(args.blank_worksheet)
        blank_by_id = {row["review_item_id"]: row for row in blank_rows}
        key = load_json(args.concealed_key)
        verify_self_hash(key, "key_sha256", "concealed key")
        key_by_review_id = {row["review_item_id"]: row for row in key.get("items", [])}
        primary_by_pair = {
            row["pair_id"]: row
            for row in key.get("items", [])
            if row.get("presentation") == "primary" and row.get("pair_id") in contested
        }
        if set(primary_by_pair) != contested:
            raise ExperimentalCloseError("primary adjudication membership drifted")

        source_pairs = load_jsonl(args.formal_pairs)
        source_summary = day22_contract.validate_manifest(source_pairs, mode="final")
        if source_summary["records"] != 200:
            raise ExperimentalCloseError("formal source pair count drifted")
        source_by_id = {row["pair_id"]: row for row in source_pairs}
        if contested - set(source_by_id):
            raise ExperimentalCloseError("contested pair is absent from formal source")

        packet = load_jsonl(args.adjudication_packet)
        scores = load_jsonl(args.adjudication_scores)
        validate_scores(packet, scores)
        packet_pairs: dict[str, str] = {}
        for index, case in enumerate(packet):
            review_id = case["review_item_id"]
            if review_id not in key_by_review_id or review_id not in blank_by_id:
                raise ExperimentalCloseError(f"packet[{index}] review id is unknown")
            item = key_by_review_id[review_id]
            pair_id = item["pair_id"]
            if item.get("presentation") != "primary" or pair_id not in contested:
                raise ExperimentalCloseError(f"packet[{index}] is not a contested primary")
            source = blank_by_id[review_id]
            for field in ("prompt", "response_a", "response_b"):
                if case.get(field) != source.get(field):
                    raise ExperimentalCloseError(f"packet[{index}] case text drifted")
            expected_tests = source_by_id[pair_id]["tests"]
            packet_tests = case.get("public_tests", {})
            if packet_tests.get("test_setup_code") != expected_tests.get("test_setup_code"):
                raise ExperimentalCloseError(f"packet[{index}] test setup drifted")
            if packet_tests.get("test_list") != expected_tests.get("test_list"):
                raise ExperimentalCloseError(f"packet[{index}] tests drifted")
            packet_pairs[case["adjudication_id"]] = pair_id
        if set(packet_pairs.values()) != contested:
            raise ExperimentalCloseError("adjudication packet pair coverage drifted")

        decisions: list[dict[str, Any]] = []
        excluded_ids: set[str] = set()
        for case, score in zip(packet, scores):
            pair_id = packet_pairs[case["adjudication_id"]]
            item = key_by_review_id[case["review_item_id"]]
            decision = score["decision"]
            if decision == "prefer_A":
                semantic = item["a_side"]
            elif decision == "prefer_B":
                semantic = item["b_side"]
            else:
                semantic = "exclude"
            action = "keep" if semantic == "chosen" else "exclude"
            if action == "exclude":
                excluded_ids.add(pair_id)
            decisions.append(
                {
                    "adjudication_id": case["adjudication_id"],
                    "review_item_id": case["review_item_id"],
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

        if excluded_ids:
            if args.pairs_output is None:
                raise ExperimentalCloseError(
                    "pairs output is required when adjudication excludes pairs"
                )
            if (
                args.pairs_output.expanduser().resolve()
                == args.formal_pairs.expanduser().resolve()
            ):
                raise ExperimentalCloseError(
                    "filtered pairs output must not overwrite formal pairs"
                )
            pairs_artifact_path = args.pairs_output
            write_pairs = True
        else:
            if args.pairs_output is not None:
                raise ExperimentalCloseError(
                    "zero-exclusion close must reuse formal pairs without a pairs output"
                )
            pairs_artifact_path = args.formal_pairs
            write_pairs = False

        accepted_pairs = [
            row for row in source_pairs if row["pair_id"] not in excluded_ids
        ]
        accepted_summary = day22_contract.validate_manifest(
            accepted_pairs, mode="final"
        )
        accepted_ids = {row["pair_id"] for row in accepted_pairs}
        pair_payload = b"".join(canonical_json(row) + b"\n" for row in accepted_pairs)

        source_split = load_json(args.formal_split_ids)
        verify_self_hash(source_split, "split_ids_sha256", "formal split ids")
        split_output: dict[str, Any] = {
            "schema_name": "day22.experimental_ai_assisted_split_ids",
            "schema_version": 1,
            "dataset_role": "experimental_ai_assisted_dpo_input",
        }
        for split in ("train", "dev", "heldout"):
            split_output[split] = [
                pair_id for pair_id in source_split[split] if pair_id in accepted_ids
            ]
            family_by_pair = {
                row["pair_id"]: row["family_keys"]["problem"]
                for row in accepted_pairs
            }
            split_output.setdefault("family_ids", {})[split] = [
                family_by_pair[pair_id] for pair_id in split_output[split]
            ]
        split_output["split_ids_sha256"] = object_sha256(split_output)
        split_payload = (
            json.dumps(split_output, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")

        report_text = render_report(
            decisions=decisions,
            source_count=len(source_pairs),
            accepted_count=len(accepted_pairs),
            split_counts=accepted_summary["split_counts"],
        )
        report_payload = report_text.encode("utf-8")
        action_counts = Counter(row["action"] for row in decisions)
        manifest: dict[str, Any] = {
            "schema_name": "day22.experimental_ai_assisted_close_manifest",
            "schema_version": 1,
            "status": "completed_experimental_ai_assisted",
            "dataset_role": "experimental_ai_assisted_dpo_input",
            "experimental_dpo_ready": True,
            "formal_human_review_ready": False,
            "formal_dpo_ready": False,
            "formal_blockers": ["human_blind_review_not_completed"],
            "authorization": {
                "actor": "user",
                "date": "2026-08-14",
                "decision": "close_day22_with_codex_subagent_adjudication",
            },
            "policy": {
                "contested_pair_rule": "non_directional_or_position_inconsistent_in_mixed_review",
                "keep_rule": "Codex preferred side maps to verifier chosen",
                "exclude_rule": "Codex exclusion or preference for verifier rejected",
                "label_flip_allowed": False,
                "source_pair_bytes_changed": False,
                "formal_gate_waived_for_experimental_track": True,
            },
            "inputs": {
                "formal_manifest": {
                    "path": relative_path(args.formal_manifest),
                    "file_sha256": file_sha256(args.formal_manifest),
                    "manifest_sha256": formal_manifest["manifest_sha256"],
                },
                "formal_pairs": {
                    "path": relative_path(args.formal_pairs),
                    "file_sha256": file_sha256(args.formal_pairs),
                },
                "formal_split_ids": {
                    "path": relative_path(args.formal_split_ids),
                    "file_sha256": file_sha256(args.formal_split_ids),
                },
                "ai_assisted_audit": {
                    "path": relative_path(args.ai_audit),
                    "file_sha256": file_sha256(args.ai_audit),
                    "audit_sha256": audit["audit_sha256"],
                },
                "blank_worksheet": {
                    "path": relative_path(args.blank_worksheet),
                    "file_sha256": file_sha256(args.blank_worksheet),
                },
                "concealed_key": {
                    "path": relative_path(args.concealed_key),
                    "file_sha256": file_sha256(args.concealed_key),
                    "key_sha256": key["key_sha256"],
                },
                "adjudication_packet": {
                    "path": relative_path(args.adjudication_packet),
                    "file_sha256": file_sha256(args.adjudication_packet),
                },
                "adjudication_scores": {
                    "path": relative_path(args.adjudication_scores),
                    "file_sha256": file_sha256(args.adjudication_scores),
                },
            },
            "counts": {
                "source_pairs": len(source_pairs),
                "contested_pairs": len(contested),
                "adjudication_keep": action_counts["keep"],
                "adjudication_exclude": action_counts["exclude"],
                "experimental_pairs": len(accepted_pairs),
                "split_counts": accepted_summary["split_counts"],
            },
            "content_identities": {
                "ordered_pair_hashes_sha256": accepted_summary[
                    "ordered_pair_hashes_sha256"
                ],
                "split_ids_sha256": split_output["split_ids_sha256"],
                "accepted_pair_ids_sha256": object_sha256(
                    [row["pair_id"] for row in accepted_pairs]
                ),
            },
            "adjudication_decisions": decisions,
            "outputs": {
                "pairs": {
                    "path": relative_path(pairs_artifact_path),
                    "file_sha256": (
                        hashlib.sha256(pair_payload).hexdigest()
                        if write_pairs
                        else file_sha256(args.formal_pairs)
                    ),
                    "records": len(accepted_pairs),
                },
                "split_ids": {
                    "path": relative_path(args.split_ids_output),
                    "file_sha256": hashlib.sha256(split_payload).hexdigest(),
                },
                "report": {
                    "path": relative_path(args.report_output),
                    "file_sha256": hashlib.sha256(report_payload).hexdigest(),
                },
            },
        }
        manifest["manifest_sha256"] = object_sha256(manifest)
        manifest_payload = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        output_payloads = [
            (args.split_ids_output, split_payload),
            (args.report_output, report_payload),
            (args.manifest_output, manifest_payload),
        ]
        if write_pairs:
            output_payloads.insert(0, (pairs_artifact_path, pair_payload))
        for path, payload in output_payloads:
            write_atomic(path, payload)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "experimental_dpo_ready": True,
                    "formal_dpo_ready": False,
                    "counts": manifest["counts"],
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ExperimentalCloseError, day22_contract.Day22ContractError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
