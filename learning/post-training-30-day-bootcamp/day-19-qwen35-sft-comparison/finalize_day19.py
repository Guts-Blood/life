#!/usr/bin/env python3
"""Audit Day 19 outputs and emit a pending report or a complete PASS manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECIPES = ("baseline-a", "baseline-b", "best-e")


class Day19FinalizationError(ValueError):
    """A Day 19 completion invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise Day19FinalizationError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Day19FinalizationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Day19FinalizationError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise Day19FinalizationError(f"JSONL file is missing: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day19FinalizationError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if isinstance(row, dict):
            rows.append(row)
    return rows


def code_sandbox_metrics(
    run_root: Path, slug: str, evaluation: dict[str, Any]
) -> dict[str, Any]:
    result_path = run_root / "eval" / f"{slug}-code-e2b.jsonl"
    summary_path = run_root / "eval" / f"{slug}-code-e2b-summary.json"
    rows = load_jsonl(result_path)
    summary = load_json(summary_path)
    if len(rows) != 28 or summary.get("records") != 28 or summary.get("status") != "complete":
        raise Day19FinalizationError(f"incomplete E2B result for {slug}")
    if summary.get("recipe") != slug or summary.get("result_file_sha256") != file_sha256(result_path):
        raise Day19FinalizationError(f"E2B summary identity mismatch for {slug}")
    prediction_path = Path(evaluation["predictions"]["path"])
    prediction_rows = load_jsonl(prediction_path)
    code_order = [row["sample_id"] for row in prediction_rows if row.get("slice") == "code"]
    if [row.get("sample_id") for row in rows] != code_order:
        raise Day19FinalizationError(f"E2B sample order mismatch for {slug}")
    prediction_sha = file_sha256(prediction_path)
    if (
        prediction_sha != evaluation["predictions"]["file_sha256"]
        or summary.get("prediction_file_sha256") != prediction_sha
    ):
        raise Day19FinalizationError(f"E2B prediction hash mismatch for {slug}")
    for row in rows:
        sandbox = row.get("sandbox", {})
        if (
            row.get("recipe") != slug
            or row.get("score_status") != "ok"
            or row.get("score") not in (0.0, 1.0)
            or row.get("execution_status") == "infrastructure_error"
            or row.get("day19_prediction_file_sha256") != prediction_sha
            or sandbox.get("backend") != "e2b_firecracker"
            or sandbox.get("fresh_sandbox_per_sample") is not True
            or sandbox.get("secure") is not True
            or sandbox.get("allow_internet_access") is not False
            or sandbox.get("allow_public_traffic") is not False
            or sandbox.get("deny_out") != ["0.0.0.0/0"]
        ):
            raise Day19FinalizationError(f"invalid E2B result row for {slug}")
    correct = sum(float(row["score"]) for row in rows)
    return {
        "status": "complete",
        "records": 28,
        "correct": correct,
        "accuracy": correct / 28,
        "execution_outcomes": summary["outcomes"],
        "error_types": summary["error_types"],
        "sandbox_contract_hash": summary["sandbox_contract_hash"],
        "comparison_key": summary["comparison_key"],
        "complete_comparison_key": summary["complete_comparison_key"],
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "summary_path": str(summary_path),
        "summary_file_sha256": file_sha256(summary_path),
        "candidate_code_executed_on_host": False,
    }


def write_new(path: Path, payload: str) -> None:
    if path.exists():
        raise Day19FinalizationError(f"refusing to overwrite finalization artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def training_metrics(output_dir: Path, expected_iters: int) -> dict[str, Any]:
    rows = [row for row in load_jsonl(output_dir / "logging.jsonl") if "loss" in row]
    if len(rows) != expected_iters:
        raise Day19FinalizationError(
            f"{output_dir.name} logged {len(rows)} losses, expected {expected_iters}"
        )
    for row in rows:
        for key in ("loss", "grad_norm", "mtp_1_loss", "memory(GiB)"):
            if not math.isfinite(float(row[key])):
                raise Day19FinalizationError(f"non-finite {key} in {output_dir}")
    return {
        "optimizer_steps": len(rows),
        "first_loss": float(rows[0]["loss"]),
        "last_loss": float(rows[-1]["loss"]),
        "minimum_loss": min(float(row["loss"]) for row in rows),
        "mean_loss": sum(float(row["loss"]) for row in rows) / len(rows),
        "first_mtp_loss": float(rows[0]["mtp_1_loss"]),
        "last_mtp_loss": float(rows[-1]["mtp_1_loss"]),
        "maximum_grad_norm": max(float(row["grad_norm"]) for row in rows),
        "peak_logged_memory_gib": max(float(row["memory(GiB)"]) for row in rows),
        "final_elapsed_time": rows[-1].get("elapsed_time"),
        "final_train_speed_seconds_per_iteration": float(rows[-1]["train_speed(s/it)"]),
    }


def percent_delta(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline


def comparison(best: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    best_metrics = best["evaluation"]["metrics"]
    base_metrics = baseline["evaluation"]["metrics"]
    result = {}
    for field in ("non_code_accuracy", "format_compliance_rate", "anomaly_rate"):
        best_value = float(best_metrics[field])
        base_value = float(base_metrics[field])
        result[field] = {
            "best_e": best_value,
            "baseline": base_value,
            "absolute_delta": best_value - base_value,
            "relative_delta": percent_delta(best_value, base_value),
        }
    for skill in ("general", "math", "finance"):
        best_value = float(best_metrics["by_slice"][skill]["accuracy"])
        base_value = float(base_metrics["by_slice"][skill]["accuracy"])
        result[f"{skill}_accuracy"] = {
            "best_e": best_value,
            "baseline": base_value,
            "absolute_delta": best_value - base_value,
            "relative_delta": percent_delta(best_value, base_value),
        }
    best_code = best_metrics["by_slice"]["code"]
    base_code = base_metrics["by_slice"]["code"]
    result["code_syntax_valid"] = {
        "best_e": int(best_code["syntax_valid"]),
        "baseline": int(base_code["syntax_valid"]),
        "absolute_delta": int(best_code["syntax_valid"]) - int(base_code["syntax_valid"]),
        "status": best_code["executable_score_status"],
    }
    if best_code["executable_score_status"] == "complete":
        best_value = float(best_code["executable_accuracy"])
        base_value = float(base_code["executable_accuracy"])
        result["code_executable_accuracy"] = {
            "best_e": best_value,
            "baseline": base_value,
            "absolute_delta": best_value - base_value,
            "relative_delta": percent_delta(best_value, base_value),
        }
    return result


def render_report(results: dict[str, Any], complete: bool) -> str:
    rows = results["runs"]
    lines = [
        "# Day 19 — Qwen3.5 SFT A/B/Best-E Comparison",
        "",
        f"Status: **{'PASS' if complete else 'PENDING — code sandbox required'}**",
        "",
        "## Answer first",
        "",
    ]
    if complete:
        lines.append(results["decision"]["summary"])
    else:
        lines.append(
            "All three controlled SFT runs, model-only checkpoints, HF reloads, and non-code evaluations completed. "
            "A promotion decision is intentionally withheld because the frozen HumanEval scorer still requires an approved sandbox; generated code was not executed on the host."
        )
    lines.extend(
        [
            "",
            "## Runtime amendments",
            "",
            f"{len(results['runtime_amendments'])} pre-training runtime amendment(s) are retained with hashes. "
            "Both failed attempts stopped before optimizer step 1 and produced no checkpoint; the data matrix was unchanged.",
            "",
            "## Frozen experiment matrix",
            "",
            "| Run | Records | Supervised tokens | Steps | First loss | Last loss | Peak train GiB | Checkpoint GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slug in RECIPES:
        row = rows[slug]
        train = row["training"]
        lines.append(
            f"| {slug} | {row['data']['records']} | {row['data']['supervised_tokens']} | "
            f"{train['optimizer_steps']} | {train['first_loss']:.6f} | {train['last_loss']:.6f} | "
            f"{train['peak_logged_memory_gib']:.2f} | {row['checkpoint']['bytes'] / (1024**3):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen dev evaluation",
            "",
            "| Run | General | Math | Finance | Non-code total | Code executable | Code syntax | Format | Anomaly | Output tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slug in RECIPES:
        metrics = rows[slug]["evaluation"]["metrics"]
        by = metrics["by_slice"]
        lines.append(
            f"| {slug} | {by['general']['correct']:.0f}/{by['general']['records']} | "
            f"{by['math']['correct']:.0f}/{by['math']['records']} | "
            f"{by['finance']['correct']:.0f}/{by['finance']['records']} | "
            f"{metrics['non_code_correct']:.0f}/{metrics['non_code_total']} | "
            f"{by['code']['executable_correct'] if by['code']['executable_score_status'] == 'complete' else 'pending'}/{by['code']['records']} | "
            f"{by['code']['syntax_valid']}/{by['code']['records']} | "
            f"{metrics['format_compliance_rate']:.2%} | {metrics['anomaly_rate']:.2%} | {metrics['output_tokens']} |"
        )
    lines.extend(
        [
            "",
            (
                "HumanEval executable accuracy comes from one fresh, network-denied E2B Firecracker sandbox per sample; syntax validity remains a separate diagnostic."
                if complete
                else "HumanEval executable accuracy is not substituted with syntax validity. The frozen scorer reports `sandbox_required`; this is the only remaining completion gate."
            ),
            "",
            "## Best-E deltas",
            "",
        ]
    )
    for baseline in ("baseline-a", "baseline-b"):
        lines.append(f"### Versus {baseline}")
        lines.append("")
        for metric, values in results["comparisons"][f"best-e_vs_{baseline}"].items():
            if metric == "code_syntax_valid":
                lines.append(
                    f"- {metric}: {values['absolute_delta']:+d} cases."
                )
            else:
                lines.append(f"- {metric}: {values['absolute_delta']:+.4f} absolute.")
        lines.append("")
    lines.extend(
        [
            "## Day 20 recommendation",
            "",
            results["decision"]["recommendation"],
            "",
            "## Claim boundary",
            "",
            "This is a single-seed, 28-example-per-slice migration experiment. It does not establish population-level superiority. "
            "The historical frozen test remains unconsumed, and no generated code was executed on the AutoDL host.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not (run_root / ".day19-run-root").is_file():
        raise Day19FinalizationError("run root marker is missing")
    manifest = load_json(run_root / "DAY19-MANIFEST.json")
    if manifest.get("status") != "prepared_training_not_started":
        raise Day19FinalizationError("unexpected initial manifest status")
    if list(item["recipe"] for item in manifest["matrix"]) != list(RECIPES):
        raise Day19FinalizationError("experiment matrix drifted")

    runtime_amendments = []
    for amendment_path in sorted((run_root / "configs").glob("runtime-amendment-*.json")):
        amendment = load_json(amendment_path)
        if amendment.get("data_or_matrix_changed") is not False:
            raise Day19FinalizationError(f"runtime amendment changed the matrix: {amendment_path}")
        if amendment.get("failed_optimizer_steps") != 0 or amendment.get("checkpoint_created") is not False:
            raise Day19FinalizationError(f"invalid failed-attempt boundary: {amendment_path}")
        runtime_amendments.append(
            {
                "path": str(amendment_path),
                "file_sha256": file_sha256(amendment_path),
                "amendment": amendment,
            }
        )

    checkpoint_dirs = sorted(run_root.glob("checkpoints/*/checkpoint-*"))
    if len(checkpoint_dirs) != 3:
        raise Day19FinalizationError(f"expected exactly three final checkpoints, got {len(checkpoint_dirs)}")
    if any((run_root / "tmp").glob("hf-*")):
        raise Day19FinalizationError("temporary HF export remains under the run root")

    sandbox_sidecars = [run_root / "eval" / f"{slug}-code-e2b.jsonl" for slug in RECIPES]
    sandbox_summaries = [
        run_root / "eval" / f"{slug}-code-e2b-summary.json" for slug in RECIPES
    ]
    sandbox_artifacts_present = sum(path.exists() for path in sandbox_sidecars + sandbox_summaries)
    if sandbox_artifacts_present not in (0, 6):
        raise Day19FinalizationError("partial E2B result set exists")
    have_complete_sandbox = sandbox_artifacts_present == 6

    runs: dict[str, Any] = {}
    code_complete = have_complete_sandbox
    for slug in RECIPES:
        config = load_json(run_root / "configs" / f"{slug}.json")
        data = load_json(run_root / "data-manifests" / f"{slug}.json")
        expected_iters = int(config["training"]["train_iters"])
        output_dir = run_root / "checkpoints" / slug
        checkpoint = output_dir / f"checkpoint-{expected_iters}"
        if checkpoint not in checkpoint_dirs:
            raise Day19FinalizationError(f"final checkpoint path mismatch for {slug}")
        audit = load_json(run_root / "evidence" / f"{slug}-checkpoint-audit.json")
        if audit.get("status") != "pass" or audit.get("checkpoint_kind") != "model-only":
            raise Day19FinalizationError(f"checkpoint audit failed for {slug}")
        evaluation = load_json(run_root / "eval" / f"{slug}.json")
        cleanup = load_json(run_root / "eval" / f"{slug}-export-validation.json")
        if cleanup.get("temporary_export_deleted") is not True:
            raise Day19FinalizationError(f"temporary export cleanup not verified for {slug}")
        if Path(cleanup["temporary_export_path"]).exists():
            raise Day19FinalizationError(f"temporary export still exists for {slug}")
        code_sandbox = None
        if have_complete_sandbox:
            code_sandbox = code_sandbox_metrics(run_root, slug, evaluation)
            code_metrics = evaluation["metrics"]["by_slice"]["code"]
            code_metrics.update(
                {
                    "scored_records": 28,
                    "correct": code_sandbox["correct"],
                    "accuracy": code_sandbox["accuracy"],
                    "executable_correct": int(code_sandbox["correct"]),
                    "executable_accuracy": code_sandbox["accuracy"],
                    "executable_score_status": "complete",
                    "sandbox": code_sandbox,
                }
            )
        runs[slug] = {
            "config": {
                "path": str(run_root / "configs" / f"{slug}.json"),
                "file_sha256": file_sha256(run_root / "configs" / f"{slug}.json"),
            },
            "data": {
                "records": data["dataset"]["records"],
                "supervised_tokens": data["dataset"]["supervised_tokens"],
                "tokens_by_skill": data["dataset"]["supervised_tokens_by_skill"],
                "manifest_path": str(run_root / "data-manifests" / f"{slug}.json"),
                "manifest_file_sha256": file_sha256(run_root / "data-manifests" / f"{slug}.json"),
            },
            "training": training_metrics(output_dir, expected_iters),
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": tree_bytes(checkpoint),
                "audit_path": str(run_root / "evidence" / f"{slug}-checkpoint-audit.json"),
                "audit_file_sha256": file_sha256(run_root / "evidence" / f"{slug}-checkpoint-audit.json"),
                "load_and_generation_verified": evaluation["model_export"]["load_and_generation_verified"],
            },
            "evaluation": evaluation,
            "export_validation": cleanup,
            "code_sandbox": code_sandbox,
        }

    if have_complete_sandbox:
        if len({runs[slug]["code_sandbox"]["sandbox_contract_hash"] for slug in RECIPES}) != 1:
            raise Day19FinalizationError("sandbox contract differs across recipes")
        if len({runs[slug]["code_sandbox"]["comparison_key"] for slug in RECIPES}) != 1:
            raise Day19FinalizationError("prediction comparison key differs across recipes")
        if len({runs[slug]["code_sandbox"]["complete_comparison_key"] for slug in RECIPES}) != 1:
            raise Day19FinalizationError("complete comparison key differs across recipes")

    comparisons = {
        "best-e_vs_baseline-a": comparison(runs["best-e"], runs["baseline-a"]),
        "best-e_vs_baseline-b": comparison(runs["best-e"], runs["baseline-b"]),
    }
    if code_complete:
        total_correct = {
            slug: float(runs[slug]["evaluation"]["metrics"]["non_code_correct"])
            + float(runs[slug]["evaluation"]["metrics"]["by_slice"]["code"]["executable_correct"])
            for slug in RECIPES
        }
        best_total = max(total_correct.values())
        leaders = [slug for slug, value in total_correct.items() if value == best_total]
        if leaders == ["best-e"]:
            recommendation = (
                "Best-E has the highest total frozen-dev correct count in this single-seed run. "
                "Use it as the Day 20 data-recipe candidate, but retain baseline-B as a control and repair any zero-accuracy slice before broader promotion."
            )
        elif len(leaders) == 1:
            recommendation = (
                f"Do not promote Best-E for Day 20. Use {leaders[0]} as the provisional anchor because it has the highest total frozen-dev correct count "
                "under the matched protocol; separately repair the weak math, finance, and executable-code slices before scaling."
            )
        else:
            recommendation = (
                "Do not declare a unique Day 20 recipe winner: the top total frozen-dev correct count is tied across "
                + ", ".join(leaders)
                + ". Run a second seed or a preregistered diagnostic slice without consuming the frozen test."
            )
        decision = {
            "status": "complete",
            "summary": (
                "The three-way comparison is complete under the frozen protocol. Total correct counts across the 112-example dev suite are: "
                + ", ".join(f"{slug}={int(total_correct[slug])}" for slug in RECIPES)
                + "."
            ),
            "total_correct": total_correct,
            "leaders": leaders,
            "recommendation": recommendation,
        }
    else:
        decision = {
            "status": "inconclusive_code_sandbox_required",
            "summary": "The comparison is not eligible for a final promotion decision because executable code accuracy is unavailable.",
            "recommendation": "Do not promote Best-E for Day 20 yet. Re-score the already-generated 28 code predictions per run in the approved frozen E2B sandbox, then finalize without retraining.",
        }
    results = {
        "schema_version": 1,
        "domain": "day19.qwen35_results",
        "status": "complete" if code_complete else "pending_code_sandbox",
        "created_at_utc": utc_now(),
        "run_root": str(run_root),
        "runs": runs,
        "comparisons": comparisons,
        "decision": decision,
        "runtime_amendments": runtime_amendments,
        "checkpoint_count": 3,
        "frozen_test_consumed": False,
    }
    results["results_sha256"] = object_sha256(results)
    if code_complete:
        results_path = run_root / "DAY19-RESULTS.json"
        report_path = run_root / "DAY19-REPORT.md"
        marker_path = run_root / "DAY19-PASS.json"
    else:
        results_path = run_root / "DAY19-RESULTS.pending.json"
        report_path = run_root / "DAY19-REPORT.pending.md"
        marker_path = run_root / "DAY19-PENDING.json"
    write_new(results_path, json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(report_path, render_report(results, code_complete))
    marker = {
        "schema_version": 1,
        "domain": "day19.qwen35_pass" if code_complete else "day19.qwen35_pending",
        "status": "pass" if code_complete else "pending_code_sandbox",
        "created_at_utc": utc_now(),
        "canonical_run": str(run_root),
        "results": {"path": str(results_path), "file_sha256": file_sha256(results_path)},
        "report": {"path": str(report_path), "file_sha256": file_sha256(report_path)},
        "checkpoint_count": 3,
        "remaining_gate": None if code_complete else "approved frozen E2B execution of 84 retained code predictions",
        "retraining_required": False,
    }
    marker["marker_sha256"] = object_sha256(marker)
    write_new(marker_path, json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(marker, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
