#!/usr/bin/env python3
"""Turn C1/C2/C3/C4 logs into fail-closed JSON gate records."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MTP_FALLBACK = re.compile(
    r"mtp.{0,160}(?:weights?.{0,40}(?:not found|missing)|random(?:ly)?[ _-]*initializ)",
    re.IGNORECASE,
)


def dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def conversion(args: argparse.Namespace) -> None:
    text = args.log.read_text(encoding="utf-8", errors="replace")
    if MTP_FALLBACK.search(text):
        raise SystemExit("conversion log reports missing or randomly initialized MTP weights")
    matches = re.findall(
        rf"mean_diff \(with loss\):\s*({FLOAT}),\s*max_diff \(with loss\):\s*({FLOAT})",
        text,
    )
    token_matches = re.findall(r"token_diff \(with loss\):\s*(\d+)", text)
    if len(matches) != args.expected_records or len(token_matches) != args.expected_records:
        raise SystemExit("conversion log is missing text loss-token parity metrics")
    records = [
        {
            "row": index,
            "mean_absolute_difference_on_loss_tokens": float(match[0]),
            "max_absolute_difference_on_loss_tokens": float(match[1]),
            "argmax_token_mismatches_on_loss_tokens": int(token_matches[index]),
        }
        for index, match in enumerate(matches)
    ]
    mean_diff = max(record["mean_absolute_difference_on_loss_tokens"] for record in records)
    max_diff = max(record["max_absolute_difference_on_loss_tokens"] for record in records)
    token_diff = max(record["argmax_token_mismatches_on_loss_tokens"] for record in records)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))[args.threshold_key]
    failures: List[str] = []
    if mean_diff > thresholds["mean_absolute_difference_on_loss_tokens_max"]:
        failures.append(f"mean loss-token logit diff {mean_diff} exceeds threshold")
    if max_diff > thresholds["max_absolute_difference_on_loss_tokens_max"]:
        failures.append(f"max loss-token logit diff {max_diff} exceeds threshold")
    if token_diff > thresholds["argmax_token_mismatches_on_loss_tokens_max"]:
        failures.append(f"loss-token argmax mismatch count {token_diff} exceeds threshold")
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "log": str(args.log.resolve()),
        "threshold_key": args.threshold_key,
        "mean_absolute_difference_on_loss_tokens": mean_diff,
        "max_absolute_difference_on_loss_tokens": max_diff,
        "argmax_token_mismatches_on_loss_tokens": token_diff,
        "records": records,
    }
    dump(args.output, payload)
    if failures:
        raise SystemExit(1)


def train(args: argparse.Namespace) -> None:
    text = args.log.read_text(encoding="utf-8", errors="replace")
    if MTP_FALLBACK.search(text):
        raise SystemExit("training log reports missing or randomly initialized MTP weights")
    loss_matches = re.findall(rf"[\"']loss[\"']\s*:\s*({FLOAT})", text)
    if not loss_matches:
        loss_matches = re.findall(rf"\bloss\s*[=:]\s*({FLOAT})", text)
    losses = [float(value) for value in loss_matches]
    grad_norm_matches = re.findall(rf"[\"']grad_norm[\"']\s*:\s*({FLOAT})", text)
    if not grad_norm_matches:
        grad_norm_matches = re.findall(rf"\bgrad_norm\s*[=:]\s*({FLOAT})", text)
    grad_norms = [float(value) for value in grad_norm_matches]
    mtp_metric_matches = re.findall(
        rf"[\"']([^\"']*mtp[^\"']*loss[^\"']*)[\"']\s*:\s*({FLOAT})",
        text,
        flags=re.IGNORECASE,
    )
    mtp_metrics = [{"name": name, "value": float(value)} for name, value in mtp_metric_matches]
    failures: List[str] = []
    if len(losses) < args.expected_steps:
        failures.append(f"expected at least {args.expected_steps} logged losses, got {len(losses)}")
    if any(not math.isfinite(value) for value in losses):
        failures.append("training log contains non-finite loss")
    if len(grad_norms) < args.expected_steps:
        failures.append(f"expected at least {args.expected_steps} logged grad norms, got {len(grad_norms)}")
    if any(not math.isfinite(value) for value in grad_norms):
        failures.append("training log contains non-finite gradient norm")
    if len(mtp_metrics) < args.expected_steps:
        failures.append(f"expected at least {args.expected_steps} logged MTP loss metrics, got {len(mtp_metrics)}")
    if any(not math.isfinite(metric["value"]) for metric in mtp_metrics):
        failures.append("training log contains non-finite MTP loss")
    checkpoint_exists = args.checkpoint.is_dir()
    if not checkpoint_exists:
        failures.append(f"missing checkpoint directory: {args.checkpoint}")
    args_json = args.checkpoint / "args.json"
    checkpoint_args: Dict[str, Any] = {}
    if args_json.is_file():
        checkpoint_args = json.loads(args_json.read_text(encoding="utf-8"))
        if checkpoint_args.get("no_save_optim") is not False:
            failures.append("checkpoint args do not prove optimizer state was saved")
        if checkpoint_args.get("no_save_rng") is not False:
            failures.append("checkpoint args do not prove RNG state was saved")
    else:
        failures.append(f"missing checkpoint args.json: {args_json}")
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "log": str(args.log.resolve()),
        "expected_steps": args.expected_steps,
        "logged_loss_count": len(losses),
        "losses": losses,
        "grad_norms": grad_norms,
        "mtp_loss_metrics": mtp_metrics,
        "first_loss": losses[0] if losses else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_args": checkpoint_args,
    }
    dump(args.output, payload)
    if failures:
        raise SystemExit(1)


def compare(args: argparse.Namespace) -> None:
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    left_value = float(left[args.left_field])
    right_value = float(right[args.right_field])
    difference = abs(left_value - right_value)
    failures = [] if difference <= args.max_abs else [f"absolute difference {difference} exceeds {args.max_abs}"]
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "left": {"path": str(args.left.resolve()), "field": args.left_field, "value": left_value},
        "right": {"path": str(args.right.resolve()), "field": args.right_field, "value": right_value},
        "absolute_difference": difference,
        "maximum_allowed": args.max_abs,
    }
    dump(args.output, payload)
    if failures:
        raise SystemExit(1)


def mcore_loss(args: argparse.Namespace) -> None:
    text = args.log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        rf"DAY18_MCORE_MAIN_LOSS\s+row=(\d+)\s+numerator=({FLOAT})\s+denominator=({FLOAT})\s+loss=({FLOAT})",
        text,
    )
    failures: List[str] = []
    if len(matches) != args.expected_records:
        failures.append(f"expected {args.expected_records} MCore loss records, got {len(matches)}")
    rows = [
        {
            "row": int(row),
            "numerator": float(numerator),
            "denominator": float(denominator),
            "loss": float(loss),
        }
        for row, numerator, denominator, loss in matches
    ]
    if rows and {row["row"] for row in rows} != set(range(args.expected_records)):
        failures.append("MCore loss records do not cover the preregistered rows")
    if any(
        not all(math.isfinite(row[key]) for key in ("numerator", "denominator", "loss"))
        or row["denominator"] <= 0
        for row in rows
    ):
        failures.append("MCore loss records contain non-finite values or zero denominator")
    mcore_value = sum(row["numerator"] for row in rows) / sum(row["denominator"] for row in rows) if rows else None
    hf = json.loads(args.hf_reference.read_text(encoding="utf-8"))
    hf_value = float(hf["reference_loss"])
    threshold = float(json.loads(args.thresholds.read_text(encoding="utf-8"))[args.threshold_key]["absolute_difference_max"])
    difference = abs(mcore_value - hf_value) if mcore_value is not None else None
    if difference is not None and difference > threshold:
        failures.append(f"single-rank HF/MCore loss difference {difference} exceeds {threshold}")
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "log": str(args.log.resolve()),
        "hf_reference": str(args.hf_reference.resolve()),
        "rows": rows,
        "mcore_loss": mcore_value,
        "hf_reference_loss": hf_value,
        "absolute_difference": difference,
        "maximum_allowed": threshold,
    }
    dump(args.output, payload)
    if failures:
        raise SystemExit(1)


def teacher_forced_metrics(payload: Dict[str, Any], source: Path) -> Dict[str, Any]:
    metrics = payload.get("teacher_forced")
    if not isinstance(metrics, dict):
        raise SystemExit(f"{source} has no structured teacher_forced metrics")
    required = ("supervised_tokens", "correct_tokens", "token_accuracy", "loss_weight_sum", "mean_loss", "rows")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise SystemExit(f"{source} teacher_forced metrics are missing: {missing}")
    return metrics


def tiny_overfit(args: argparse.Namespace) -> None:
    base_payload = json.loads(args.base.read_text(encoding="utf-8"))
    final_payload = json.loads(args.final.read_text(encoding="utf-8"))
    base = teacher_forced_metrics(base_payload, args.base)
    final = teacher_forced_metrics(final_payload, args.final)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))[args.threshold_key]
    text = args.log.read_text(encoding="utf-8", errors="replace")
    loss_matches = re.findall(rf"[\"']loss[\"']\s*:\s*({FLOAT})", text)
    if not loss_matches:
        loss_matches = re.findall(rf"\bloss\s*[=:]\s*({FLOAT})", text)
    logged_losses = [float(value) for value in loss_matches]
    grad_norm_matches = re.findall(rf"[\"']grad_norm[\"']\s*:\s*({FLOAT})", text)
    if not grad_norm_matches:
        grad_norm_matches = re.findall(rf"\bgrad_norm\s*[=:]\s*({FLOAT})", text)
    grad_norms = [float(value) for value in grad_norm_matches]
    mtp_metric_matches = re.findall(
        rf"[\"']([^\"']*mtp[^\"']*loss[^\"']*)[\"']\s*:\s*({FLOAT})",
        text,
        flags=re.IGNORECASE,
    )
    mtp_metrics = [{"name": name, "value": float(value)} for name, value in mtp_metric_matches]
    nonfinite_logged_metric = re.search(
        r"(?:[\"']?(?:loss|grad_norm|mtp[^\"'\s:=]*loss)[\"']?\s*[:=]\s*)"
        r"(?:nan|[-+]?inf(?:inity)?)\b",
        text,
        flags=re.IGNORECASE,
    )

    failures: List[str] = []
    minimum_steps = int(thresholds["minimum_optimizer_steps"])
    maximum_steps = int(thresholds["maximum_optimizer_steps"])
    if not minimum_steps <= args.expected_steps <= maximum_steps:
        failures.append(
            f"expected optimizer steps {args.expected_steps} are outside preregistered range "
            f"[{minimum_steps}, {maximum_steps}]"
        )
    if len(logged_losses) < args.expected_steps:
        failures.append(f"expected at least {args.expected_steps} logged losses, got {len(logged_losses)}")
    if len(grad_norms) < args.expected_steps:
        failures.append(f"expected at least {args.expected_steps} logged gradient norms, got {len(grad_norms)}")
    if len(mtp_metrics) < args.expected_steps:
        failures.append(f"expected at least {args.expected_steps} logged MTP loss metrics, got {len(mtp_metrics)}")
    if any(not math.isfinite(value) for value in logged_losses) or nonfinite_logged_metric:
        failures.append("training log contains a non-finite loss, MTP loss, or gradient norm")
    if any(not math.isfinite(value) for value in grad_norms):
        failures.append("training log contains a non-finite gradient norm")
    if any(not math.isfinite(metric["value"]) for metric in mtp_metrics):
        failures.append("training log contains a non-finite MTP loss")
    if base_payload.get("fixture_sha256") != final_payload.get("fixture_sha256"):
        failures.append("base and final evaluations use different fixture hashes")
    if base_payload.get("model_class") != final_payload.get("model_class"):
        failures.append("base and final evaluations use different model classes")
    if final_payload.get("model_class") != "Qwen3_5ForConditionalGeneration":
        failures.append(f"wrong final conditional model class: {final_payload.get('model_class')}")
    if base_payload.get("model_path") == final_payload.get("model_path"):
        failures.append("final evaluation did not load the exported checkpoint path")
    for stage, payload in (("base", base_payload), ("final", final_payload)):
        if payload.get("forbidden_tensor_keys"):
            failures.append(f"{stage} evaluation contains multimodal tensors")
        if payload.get("forbidden_visual_token_ids"):
            failures.append(f"{stage} evaluation contains visual special tokens")

    base_tokens = int(base["supervised_tokens"])
    final_tokens = int(final["supervised_tokens"])
    if base_tokens <= 0 or final_tokens != base_tokens:
        failures.append(f"supervised token budget drifted: base={base_tokens}, final={final_tokens}")
    if not math.isclose(float(base["loss_weight_sum"]), float(final["loss_weight_sum"]), rel_tol=0, abs_tol=1e-6):
        failures.append("teacher-forced loss-weight budget drifted")
    base_rows = base.get("rows") or []
    final_rows = final.get("rows") or []
    base_row_ids = [row.get("row") for row in base_rows]
    final_row_ids = [row.get("row") for row in final_rows]
    if base_row_ids != [0, 1] or final_row_ids != base_row_ids:
        failures.append("teacher-forced row coverage drifted")
    if [row.get("supervised_tokens") for row in base_rows] != [
        row.get("supervised_tokens") for row in final_rows
    ]:
        failures.append("per-row supervised-token budgets drifted")
    if sum(int(row.get("supervised_tokens", 0)) for row in final_rows) != final_tokens:
        failures.append("final per-row supervised-token counts do not sum to the aggregate")

    base_loss = float(base["mean_loss"])
    final_loss = float(final["mean_loss"])
    final_accuracy = float(final["token_accuracy"])
    final_correct = int(final["correct_tokens"])
    if (
        not all(math.isfinite(value) for value in (base_loss, final_loss, final_accuracy))
        or base_loss <= 0
        or final_loss < 0
        or not 0 <= final_correct <= final_tokens
        or not 0 <= final_accuracy <= 1
    ):
        failures.append("teacher-forced metrics are non-finite or outside their valid ranges")
        relative_loss_reduction = None
    else:
        relative_loss_reduction = (base_loss - final_loss) / base_loss
        if relative_loss_reduction < float(thresholds["minimum_relative_loss_reduction"]):
            failures.append(
                f"relative teacher-forced loss reduction {relative_loss_reduction} is below "
                f"{thresholds['minimum_relative_loss_reduction']}"
            )
    if final_loss > float(thresholds["final_mean_loss_max"]):
        failures.append(f"final teacher-forced mean loss {final_loss} exceeds {thresholds['final_mean_loss_max']}")
    if final_accuracy < float(thresholds["teacher_forced_token_accuracy_min"]):
        failures.append(
            f"final teacher-forced token accuracy {final_accuracy} is below "
            f"{thresholds['teacher_forced_token_accuracy_min']}"
        )
    if final_tokens > 0 and not math.isclose(
        final_accuracy,
        final_correct / final_tokens,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        failures.append("final token accuracy is inconsistent with correct/supervised token counts")

    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "claim_scope": "same-fixture pipeline learnability only; not generalization or exact-resume evidence",
        "base": {"path": str(args.base.resolve()), "teacher_forced": base},
        "final": {"path": str(args.final.resolve()), "teacher_forced": final},
        "training_log": str(args.log.resolve()),
        "optimizer_steps": args.expected_steps,
        "logged_loss_count": len(logged_losses),
        "logged_gradient_norm_count": len(grad_norms),
        "logged_mtp_loss_count": len(mtp_metrics),
        "first_logged_loss": logged_losses[0] if logged_losses else None,
        "last_logged_loss": logged_losses[-1] if logged_losses else None,
        "relative_teacher_forced_loss_reduction": relative_loss_reduction,
        "threshold_key": args.threshold_key,
        "thresholds": thresholds,
    }
    dump(args.output, payload)
    if failures:
        raise SystemExit(1)


def ranks(args: argparse.Namespace) -> None:
    paths = sorted(args.evidence_dir.glob("rank-*.json"))
    failures: List[str] = []
    expected_iterations = list(range(args.expected_start + 1, args.expected_final + 1))
    if len(expected_iterations) != args.expected_steps:
        failures.append(
            f"expected range {expected_iterations} contains {len(expected_iterations)} iterations, "
            f"not {args.expected_steps}"
        )
    if len(paths) != args.expected_ranks:
        failures.append(f"expected {args.expected_ranks} rank records, got {len(paths)}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {path}: {exc}")
            continue
        rows.append(row)
        if row.get("status") != "pass":
            failures.append(f"{path.name} status is {row.get('status')}: {row.get('failures')}")
        for key, expected in (
            ("world_size", args.expected_ranks),
            ("tp_world_size", args.expected_tp),
            ("dp_world_size", args.expected_dp),
            ("start_iteration", args.expected_start),
            ("final_iteration", args.expected_final),
        ):
            if row.get(key) != expected:
                failures.append(f"{path.name} {key}: expected {expected}, got {row.get(key)}")
        steps = row.get("steps") or []
        if len(steps) != args.expected_steps:
            failures.append(f"{path.name}: expected {args.expected_steps} step records, got {len(steps)}")
        observed_iterations = [step.get("iteration") for step in steps]
        if observed_iterations != expected_iterations:
            failures.append(
                f"{path.name}: expected step iterations {expected_iterations}, got {observed_iterations}"
            )
        if not row.get("expected_mcore_bridge_gdn_modules"):
            failures.append(f"{path.name}: missing pinned mcore-bridge GatedDeltaNet class/source evidence")
        if row.get("trainable_visual_parameter_tensors_local") != 0:
            failures.append(f"{path.name}: visual or aligner parameters were trainable")
        if not row.get("visual_parameter_tensors_local"):
            failures.append(f"{path.name}: full conditional model visual parameters were not found")
        if not row.get("mtp_parameter_tensors_local"):
            failures.append(f"{path.name}: no MCore MTP parameters were found")
        if row.get("trainable_mtp_parameter_tensors_local") != row.get("mtp_parameter_tensors_local"):
            failures.append(f"{path.name}: not every MTP parameter tensor was trainable")
        for step in steps:
            if step.get("update_successful") is not True:
                failures.append(
                    f"{path.name} iteration {step.get('iteration')}: optimizer update was not successful"
                )
            if not step.get("trainable_tensors_with_gradient"):
                failures.append(f"{path.name} iteration {step.get('iteration')}: no trainable gradient")
            if step.get("visual_tensors_with_gradient") != 0:
                failures.append(f"{path.name} iteration {step.get('iteration')}: visual gradient detected")
            if float(step.get("cuda_free_fraction", 0.0)) < 0.10:
                failures.append(f"{path.name} iteration {step.get('iteration')}: VRAM headroom below 10%")
            mtp_stats = step.get("mtp_gradient_stats") or {}
            if mtp_stats.get("missing_gradient_tensors") != 0:
                failures.append(f"{path.name} iteration {step.get('iteration')}: missing MTP gradients")
            if mtp_stats.get("nonfinite_gradient_tensors") != 0:
                failures.append(f"{path.name} iteration {step.get('iteration')}: non-finite MTP gradients")
            if not mtp_stats.get("nonzero_gradient_tensors"):
                failures.append(f"{path.name} iteration {step.get('iteration')}: zero MTP gradients")
    if rows and {row.get("rank") for row in rows} != set(range(args.expected_ranks)):
        failures.append("rank records do not cover the expected global ranks")
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "evidence_dir": str(args.evidence_dir.resolve()),
        "expected": {
            "ranks": args.expected_ranks,
            "tp": args.expected_tp,
            "dp": args.expected_dp,
            "start_iteration": args.expected_start,
            "final_iteration": args.expected_final,
            "steps": args.expected_steps,
            "iterations": expected_iterations,
        },
        "ranks": rows,
    }
    dump(args.output, payload)
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("conversion")
    convert_parser.add_argument("--log", required=True, type=Path)
    convert_parser.add_argument("--thresholds", required=True, type=Path)
    convert_parser.add_argument("--threshold-key", required=True)
    convert_parser.add_argument("--expected-records", type=int, default=1)
    convert_parser.add_argument("--output", required=True, type=Path)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--log", required=True, type=Path)
    train_parser.add_argument("--checkpoint", required=True, type=Path)
    train_parser.add_argument("--expected-steps", required=True, type=int)
    train_parser.add_argument("--output", required=True, type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True, type=Path)
    compare_parser.add_argument("--left-field", required=True)
    compare_parser.add_argument("--right", required=True, type=Path)
    compare_parser.add_argument("--right-field", required=True)
    compare_parser.add_argument("--max-abs", required=True, type=float)
    compare_parser.add_argument("--output", required=True, type=Path)
    ranks_parser = subparsers.add_parser("ranks")
    ranks_parser.add_argument("--evidence-dir", required=True, type=Path)
    ranks_parser.add_argument("--expected-ranks", required=True, type=int)
    ranks_parser.add_argument("--expected-tp", required=True, type=int)
    ranks_parser.add_argument("--expected-dp", required=True, type=int)
    ranks_parser.add_argument("--expected-start", required=True, type=int)
    ranks_parser.add_argument("--expected-final", required=True, type=int)
    ranks_parser.add_argument("--expected-steps", required=True, type=int)
    ranks_parser.add_argument("--output", required=True, type=Path)
    loss_parser = subparsers.add_parser("mcore-loss")
    loss_parser.add_argument("--log", required=True, type=Path)
    loss_parser.add_argument("--hf-reference", required=True, type=Path)
    loss_parser.add_argument("--thresholds", required=True, type=Path)
    loss_parser.add_argument("--threshold-key", required=True)
    loss_parser.add_argument("--expected-records", type=int, default=2)
    loss_parser.add_argument("--output", required=True, type=Path)
    overfit_parser = subparsers.add_parser("tiny-overfit")
    overfit_parser.add_argument("--base", required=True, type=Path)
    overfit_parser.add_argument("--final", required=True, type=Path)
    overfit_parser.add_argument("--log", required=True, type=Path)
    overfit_parser.add_argument("--thresholds", required=True, type=Path)
    overfit_parser.add_argument("--threshold-key", default="c5_tiny_overfit")
    overfit_parser.add_argument("--expected-steps", required=True, type=int)
    overfit_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {
        "conversion": conversion,
        "train": train,
        "compare": compare,
        "ranks": ranks,
        "mcore-loss": mcore_loss,
        "tiny-overfit": tiny_overfit,
    }[args.command](args)


if __name__ == "__main__":
    main()
