#!/usr/bin/env python3
"""Assemble a mixed human/AI blind-review audit without promoting the human gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ANSWER_FIELDS = {"verdict", "confidence", "notes"}
VERDICTS = {"A", "B", "tie", "ambiguous", "reject"}
CONFIDENCES = {"low", "medium", "high"}
ANNOTATION_FIELDS = {
    "row_number",
    "review_item_id",
    "verdict",
    "confidence",
    "notes",
}
PACKET_FIELDS = {
    "row_number",
    "review_item_id",
    "prompt",
    "response_a",
    "response_b",
    "allowed_verdicts",
    "allowed_confidence",
}


class ReviewAuditError(ValueError):
    """An input or review annotation failed a fail-closed validation."""


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReviewAuditError(f"cannot read {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ReviewAuditError(f"blank line in {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReviewAuditError(
                f"invalid JSON in {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ReviewAuditError(f"non-object row in {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_answer(row: Mapping[str, Any], *, allow_blank: bool) -> bool:
    verdict = row.get("verdict")
    confidence = row.get("confidence")
    notes = row.get("notes")
    if verdict == confidence == notes == "" and allow_blank:
        return False
    if verdict not in VERDICTS:
        raise ReviewAuditError(f"invalid verdict for {row.get('review_item_id')}")
    if confidence not in CONFIDENCES:
        raise ReviewAuditError(f"invalid confidence for {row.get('review_item_id')}")
    if not isinstance(notes, str):
        raise ReviewAuditError(f"invalid notes for {row.get('review_item_id')}")
    if verdict in {"tie", "ambiguous", "reject"} and not notes.strip():
        raise ReviewAuditError(
            f"non-directional verdict requires notes: {row.get('review_item_id')}"
        )
    return True


def validate_human_draft(
    blank: Sequence[Mapping[str, Any]], draft: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    if len(blank) != 100 or len(draft) != len(blank):
        raise ReviewAuditError("blank/draft must each contain exactly 100 rows")
    answers: dict[str, dict[str, str]] = {}
    for row_number, (source, reviewed) in enumerate(zip(blank, draft), 1):
        if set(source) != set(reviewed):
            raise ReviewAuditError(f"human draft fields drifted at row {row_number}")
        source_fixed = {key: value for key, value in source.items() if key not in ANSWER_FIELDS}
        reviewed_fixed = {
            key: value for key, value in reviewed.items() if key not in ANSWER_FIELDS
        }
        if source_fixed != reviewed_fixed:
            raise ReviewAuditError(f"human draft content drifted at row {row_number}")
        if _validate_answer(reviewed, allow_blank=True):
            answers[reviewed["review_item_id"]] = {
                "verdict": reviewed["verdict"],
                "confidence": reviewed["confidence"],
                "notes": reviewed["notes"],
            }
    return answers


def validate_ai_review(
    packet_path: Path,
    annotation_path: Path,
    blank_by_id: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    packets = load_jsonl(packet_path)
    annotations = load_jsonl(annotation_path)
    if len(packets) != 30 or len(annotations) != len(packets):
        raise ReviewAuditError(
            f"AI packet/annotation must contain 30 rows: {packet_path}"
        )
    output: list[dict[str, Any]] = []
    for packet, annotation in zip(packets, annotations):
        if set(packet) != PACKET_FIELDS:
            raise ReviewAuditError(f"packet fields drifted: {packet_path}")
        if set(annotation) != ANNOTATION_FIELDS:
            raise ReviewAuditError(f"annotation fields drifted: {annotation_path}")
        review_id = packet.get("review_item_id")
        if review_id not in blank_by_id:
            raise ReviewAuditError(f"unknown packet review id: {review_id}")
        expected_number, source = blank_by_id[review_id]
        if packet.get("row_number") != expected_number:
            raise ReviewAuditError(f"packet row number drifted: {review_id}")
        for field in ("prompt", "response_a", "response_b"):
            if packet.get(field) != source.get(field):
                raise ReviewAuditError(f"packet case text drifted: {review_id}")
        if annotation.get("review_item_id") != review_id:
            raise ReviewAuditError(f"annotation order/id drifted: {review_id}")
        if annotation.get("row_number") != expected_number:
            raise ReviewAuditError(f"annotation row number drifted: {review_id}")
        _validate_answer(annotation, allow_blank=False)
        if not annotation["notes"].strip():
            raise ReviewAuditError(f"AI notes must be non-empty: {review_id}")
        output.append(annotation)
    return output


def load_key(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    try:
        key = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewAuditError(f"cannot read concealed key: {path}") from error
    if key.get("schema_name") != "day22.blind_review_concealed_key":
        raise ReviewAuditError("concealed key schema drifted")
    items = key.get("items")
    if not isinstance(items, list) or len(items) != 100:
        raise ReviewAuditError("concealed key must contain 100 items")
    by_id = {item.get("review_item_id"): item for item in items}
    if set(by_id) != expected_ids or None in by_id:
        raise ReviewAuditError("concealed key membership drifted")
    for review_id, item in by_id.items():
        if item.get("a_side") not in {"chosen", "rejected"}:
            raise ReviewAuditError(f"invalid key a_side: {review_id}")
        if item.get("b_side") not in {"chosen", "rejected"}:
            raise ReviewAuditError(f"invalid key b_side: {review_id}")
        if item["a_side"] == item["b_side"]:
            raise ReviewAuditError(f"invalid key side mapping: {review_id}")
    return by_id


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def build_audit(
    annotations: Sequence[Mapping[str, Any]],
    key_by_id: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    presentation_rows: list[dict[str, Any]] = []
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        key = key_by_id[annotation["review_item_id"]]
        verdict = annotation["verdict"]
        semantic = key[f"{verdict.lower()}_side"] if verdict in {"A", "B"} else verdict
        verifier_visible = "A" if key["a_side"] == "chosen" else "B"
        agrees = verdict == verifier_visible if verdict in {"A", "B"} else False
        row = {
            "row_number": annotation["row_number"],
            "review_item_id": annotation["review_item_id"],
            "pair_id": key["pair_id"],
            "presentation": key["presentation"],
            "reviewer_type": annotation["reviewer_type"],
            "reviewer_id": annotation["reviewer_id"],
            "verdict": verdict,
            "confidence": annotation["confidence"],
            "notes": annotation["notes"],
            "semantic_decision": semantic,
            "verifier_visible_preference": verifier_visible,
            "verifier_direction_agreement": agrees,
        }
        presentation_rows.append(row)
        pair_rows[key["pair_id"]].append(row)
        by_type[annotation["reviewer_type"]].append(row)

    source_metrics: dict[str, Any] = {}
    for reviewer_type, rows in sorted(by_type.items()):
        directional = [row for row in rows if row["verdict"] in {"A", "B"}]
        agreements = sum(row["verifier_direction_agreement"] for row in directional)
        source_metrics[reviewer_type] = {
            "presentations": len(rows),
            "directional_presentations": len(directional),
            "verifier_direction_agreements": agreements,
            "verifier_direction_agreement_rate": _rate(agreements, len(directional)),
            "verdict_counts": dict(sorted(Counter(row["verdict"] for row in rows).items())),
            "confidence_counts": dict(
                sorted(Counter(row["confidence"] for row in rows).items())
            ),
        }

    pair_metrics = Counter()
    pair_decisions: list[dict[str, Any]] = []
    for pair_id, rows in sorted(pair_rows.items()):
        if len(rows) != 2:
            raise ReviewAuditError(f"pair does not have two presentations: {pair_id}")
        semantics = [row["semantic_decision"] for row in rows]
        reviewer_types = sorted({row["reviewer_type"] for row in rows})
        both_directional = all(value in {"chosen", "rejected"} for value in semantics)
        consistent = semantics[0] == semantics[1]
        if both_directional:
            pair_metrics["both_directional"] += 1
            if consistent:
                pair_metrics["position_consistent_directional"] += 1
                pair_metrics[f"consistent_{semantics[0]}"] += 1
            else:
                pair_metrics["position_disagreement_directional"] += 1
        else:
            pair_metrics["contains_nondirectional"] += 1
        if reviewer_types == ["human", "subagent"]:
            pair_metrics["mixed_human_subagent"] += 1
        pair_decisions.append(
            {
                "pair_id": pair_id,
                "semantic_decisions": semantics,
                "both_directional": both_directional,
                "position_consistent": consistent,
                "reviewer_types": reviewer_types,
            }
        )

    cohort_metrics: dict[str, Any] = {}
    for cohort, expected_types in (
        ("subagent_only", ["subagent"]),
        ("mixed_human_subagent", ["human", "subagent"]),
    ):
        rows_by_pair = [
            rows
            for rows in pair_rows.values()
            if sorted({row["reviewer_type"] for row in rows}) == expected_types
        ]
        directional_pairs = [
            rows
            for rows in rows_by_pair
            if all(row["verdict"] in {"A", "B"} for row in rows)
        ]
        consistent_pairs = [
            rows
            for rows in directional_pairs
            if len({row["semantic_decision"] for row in rows}) == 1
        ]
        cohort_metrics[cohort] = {
            "pairs": len(rows_by_pair),
            "both_directional_pairs": len(directional_pairs),
            "position_consistent_directional_pairs": len(consistent_pairs),
            "directional_position_consistency_rate": _rate(
                len(consistent_pairs), len(directional_pairs)
            ),
            "contains_nondirectional_pairs": len(rows_by_pair)
            - len(directional_pairs),
        }

    directional_all = [
        row for row in presentation_rows if row["verdict"] in {"A", "B"}
    ]
    agreements_all = sum(
        row["verifier_direction_agreement"] for row in directional_all
    )
    audit: dict[str, Any] = {
        "schema_name": "day22.ai_assisted_blind_review_audit",
        "schema_version": 1,
        "status": "audit_only_not_formal_human_review",
        "formal_human_review_eligible": False,
        "formal_dpo_ready": False,
        "formal_blockers": [
            "ninety_presentations_reviewed_by_subagents_not_humans"
        ],
        "inputs": dict(inputs),
        "counts": {
            "presentations": len(presentation_rows),
            "unique_pairs": len(pair_rows),
            "human_presentations": len(by_type.get("human", [])),
            "subagent_presentations": len(by_type.get("subagent", [])),
            "directional_presentations": len(directional_all),
            "verifier_direction_agreements": agreements_all,
            "verifier_direction_agreement_rate": _rate(
                agreements_all, len(directional_all)
            ),
        },
        "reviewer_source_metrics": source_metrics,
        "pair_metrics": {
            **dict(sorted(pair_metrics.items())),
            "directional_position_consistency_rate": _rate(
                pair_metrics["position_consistent_directional"],
                pair_metrics["both_directional"],
            ),
        },
        "pair_cohort_metrics": cohort_metrics,
        "presentation_results": presentation_rows,
        "pair_results": pair_decisions,
    }
    audit["audit_sha256"] = object_sha256(audit, "audit_sha256")
    return audit


def render_report(audit: Mapping[str, Any]) -> str:
    counts = audit["counts"]
    human = audit["reviewer_source_metrics"]["human"]
    subagent = audit["reviewer_source_metrics"]["subagent"]
    pair = audit["pair_metrics"]
    cohorts = audit["pair_cohort_metrics"]
    disagreements = [
        row
        for row in audit["presentation_results"]
        if row["verdict"] in {"A", "B"}
        and not row["verifier_direction_agreement"]
    ]
    nondirectional = [
        row
        for row in audit["presentation_results"]
        if row["verdict"] not in {"A", "B"}
    ]

    def rows_table(rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return "（无）"
        lines = [
            "| 行 | reviewer | pair | 判断 | 置信度 | 备注 |",
            "|---:|---|---|---|---|---|",
        ]
        for row in rows:
            note = str(row["notes"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['row_number']} | {row['reviewer_type']} | "
                f"`{row['pair_id']}` | {row['verdict']} | "
                f"{row['confidence']} | {note} |"
            )
        return "\n".join(lines)

    return f"""# Day 22 混合盲审审计（10 人工 + 90 AI-assisted）

结论：这是一份校准与风险审计，不是 100 条人工盲审。Formal human-review gate 保持 `BLOCKED`，不得把本报告或 annotations 文件作为 ready manifest 的替代品。

## 结果

- 展示数：{counts['presentations']}（人工 {counts['human_presentations']}，sub-agent {counts['subagent_presentations']}）
- unique pairs：{counts['unique_pairs']}
- 全部 directional 展示与执行 verifier 方向一致：{counts['verifier_direction_agreements']}/{counts['directional_presentations']}（{counts['verifier_direction_agreement_rate']:.1%}）
- 人工 10 条一致：{human['verifier_direction_agreements']}/{human['directional_presentations']}（{human['verifier_direction_agreement_rate']:.1%}）
- sub-agent 90 条一致：{subagent['verifier_direction_agreements']}/{subagent['directional_presentations']}（{subagent['verifier_direction_agreement_rate']:.1%}）
- 两次展示都给出方向的 pair：{pair.get('both_directional', 0)}
- 方向 position-consistent：{pair.get('position_consistent_directional', 0)}（{pair['directional_position_consistency_rate']:.1%}）
- human/sub-agent 混合判断的 pair：{pair.get('mixed_human_subagent', 0)}；这些不能解释为单一 reviewer 的 position-bias 测试。
- 40 个纯 sub-agent pair：{cohorts['subagent_only']['both_directional_pairs']} 个双向明确，其中 {cohorts['subagent_only']['position_consistent_directional_pairs']} 个换位后一致；{cohorts['subagent_only']['contains_nondirectional_pairs']} 个含非定向判断。
- 10 个 human/sub-agent 混合 pair：{cohorts['mixed_human_subagent']['both_directional_pairs']} 个双向明确，其中 {cohorts['mixed_human_subagent']['position_consistent_directional_pairs']} 个语义一致；由于 reviewer 不同，不能把差异直接归因于 position bias。

## 与执行 verifier 方向不一致的展示

{rows_table(disagreements)}

## 非定向判断

{rows_table(nondirectional)}

## 使用边界

- verifier agreement 不是 ground-truth accuracy；MBPP prompt/test 可能存在口径歧义，公开测试也较弱。
- preference 的首要判断标准是功能正确性，其次是格式遵循；代码风格只在功能等价时作为次要因素。
- 高度 verifier agreement 也不能证明 sub-agent 可以替代人类：reviewer 与生成模型可能共享代码先验，而且当前 pair 本来就是按执行 verifier 选出的。
- 本产物可以发现歧义、弱测试和明显偏好捷径，但不完成原协议要求的人工盲审。
"""


def write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise ReviewAuditError(f"refusing to overwrite: {resolved}")
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
    parser.add_argument("--blank", type=Path, required=True)
    parser.add_argument("--human-draft", type=Path, required=True)
    parser.add_argument("--concealed-key", type=Path, required=True)
    parser.add_argument(
        "--ai-review",
        action="append",
        nargs=3,
        metavar=("PACKET", "ANNOTATIONS", "REVIEWER_ID"),
        required=True,
    )
    parser.add_argument("--annotations-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--human-draft-snapshot-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = [
        args.annotations_output,
        args.audit_output,
        args.report_output,
        args.human_draft_snapshot_output,
    ]
    resolved_outputs = [path.expanduser().resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        print("error: output paths must be distinct", file=os.sys.stderr)
        return 2
    if not args.overwrite and any(path.exists() for path in resolved_outputs):
        print("error: refusing to overwrite an output", file=os.sys.stderr)
        return 2
    try:
        blank = load_jsonl(args.blank)
        draft = load_jsonl(args.human_draft)
        human_answers = validate_human_draft(blank, draft)
        blank_by_id = {
            row["review_item_id"]: (row_number, row)
            for row_number, row in enumerate(blank, 1)
        }
        if len(blank_by_id) != 100:
            raise ReviewAuditError("blank review ids are not unique")
        merged: dict[str, dict[str, Any]] = {}
        for review_id, answer in human_answers.items():
            row_number, _ = blank_by_id[review_id]
            merged[review_id] = {
                "row_number": row_number,
                "review_item_id": review_id,
                "reviewer_type": "human",
                "reviewer_id": "user",
                **answer,
            }
        ai_inputs: list[dict[str, Any]] = []
        for packet_text, annotation_text, reviewer_id in args.ai_review:
            packet_path = Path(packet_text).expanduser().resolve()
            annotation_path = Path(annotation_text).expanduser().resolve()
            rows = validate_ai_review(packet_path, annotation_path, blank_by_id)
            for row in rows:
                review_id = row["review_item_id"]
                if review_id in merged:
                    raise ReviewAuditError(f"duplicate review annotation: {review_id}")
                merged[review_id] = {
                    **row,
                    "reviewer_type": "subagent",
                    "reviewer_id": reviewer_id,
                }
            ai_inputs.append(
                {
                    "packet": str(packet_path),
                    "packet_sha256": file_sha256(packet_path),
                    "annotations": str(annotation_path),
                    "annotations_sha256": file_sha256(annotation_path),
                    "reviewer_id": reviewer_id,
                }
            )
        if set(merged) != set(blank_by_id):
            missing = sorted(set(blank_by_id) - set(merged))
            raise ReviewAuditError(f"review coverage is not exact; missing={missing[:3]}")
        annotations: list[dict[str, Any]] = []
        for row_number, source in enumerate(blank, 1):
            annotation = {
                "schema_name": "day22.ai_assisted_blind_review_annotation",
                "schema_version": 1,
                **merged[source["review_item_id"]],
                "source_blank_sha256": file_sha256(args.blank),
            }
            annotation["annotation_sha256"] = object_sha256(
                annotation, "annotation_sha256"
            )
            annotations.append(annotation)
        key_by_id = load_key(args.concealed_key, set(blank_by_id))
        inputs = {
            "blank": str(args.blank.expanduser().resolve()),
            "blank_sha256": file_sha256(args.blank),
            "human_draft": str(args.human_draft.expanduser().resolve()),
            "human_draft_sha256": file_sha256(args.human_draft),
            "concealed_key": str(args.concealed_key.expanduser().resolve()),
            "concealed_key_file_sha256": file_sha256(args.concealed_key),
            "ai_reviews": ai_inputs,
        }
        audit = build_audit(annotations, key_by_id, inputs)
        annotation_payload = b"".join(
            canonical_json(row) + b"\n" for row in annotations
        )
        write_atomic(
            args.annotations_output, annotation_payload, overwrite=args.overwrite
        )
        write_atomic(
            args.audit_output,
            json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2).encode(
                "utf-8"
            )
            + b"\n",
            overwrite=args.overwrite,
        )
        write_atomic(
            args.report_output,
            render_report(audit).encode("utf-8"),
            overwrite=args.overwrite,
        )
        write_atomic(
            args.human_draft_snapshot_output,
            args.human_draft.read_bytes(),
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "counts": audit["counts"],
                    "audit_sha256": audit["audit_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ReviewAuditError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
