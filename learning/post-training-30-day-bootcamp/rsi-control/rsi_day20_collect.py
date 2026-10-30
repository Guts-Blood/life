#!/usr/bin/env python3
"""Read-only inventory of verified Day 20 v2 run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_NAME = "rsi.day20_evidence_inventory"
SCHEMA_VERSION = 1
RUN_MARKER = ".day20-v2-run-root"
RUN_MARKER_VALUE = "day20-qwen35-candidate-factory-v2"
CHECKPOINT_INTEGRITY_FILE = "day20-v2-checkpoint-integrity.json"
PROBE_SELECTIONS = ("PROBE-SELECTION-STAGE-A.json", "PROBE-SELECTION.json")
PRIMARY_SELECTION = "PRIMARY-MAIN-SELECTION.json"
FINAL_PROMOTION = "FINAL-PROMOTION.json"


class Day20CollectionError(ValueError):
    """The requested run root is unsafe or not a Day 20 v2 run."""


@dataclass(frozen=True)
class VerifierBundle:
    training: Callable[..., dict[str, Any]]
    raw: Callable[..., dict[str, Any]]
    normalized: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]
    e2b: Callable[
        ...,
        tuple[
            list[dict[str, Any]],
            dict[str, Any],
            list[dict[str, Any]],
            dict[str, Any],
        ],
    ]
    probe_selection: Callable[..., dict[str, Any]]
    primary_selection: Callable[..., dict[str, Any]]
    final_promotion: Callable[..., dict[str, Any]]


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise Day20CollectionError(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verifiers() -> VerifierBundle:
    day20 = Path(__file__).resolve().parent.parent / "day-20-qwen35-balanced-lora-sft"
    if not day20.is_dir():
        raise Day20CollectionError(f"Day 20 v2 implementation is missing: {day20}")
    day20_text = str(day20)
    if day20_text not in sys.path:
        sys.path.insert(0, day20_text)

    import day20_train_plugin_v2 as training
    import evaluate_day20_v2 as raw
    import rescore_day20_qwen35_v3 as normalized
    import score_day20_code_e2b_v3 as e2b
    import select_day20_main_v2 as main_selection
    import select_day20_v2 as probe_selection

    return VerifierBundle(
        training=training.verify_training_summary,
        raw=raw.verify_published_pair,
        normalized=normalized.verify_normalized_pair,
        e2b=e2b.verify_published_pair,
        probe_selection=probe_selection.verify_selection,
        primary_selection=main_selection.verify_primary_selection,
        final_promotion=main_selection.verify_final_promotion,
    )


def _validate_run_root(run_root: Path) -> Path:
    requested = run_root.expanduser()
    if requested.is_symlink():
        raise Day20CollectionError(f"run root must not be symbolic: {requested}")
    root = requested.resolve()
    if not root.is_dir():
        raise Day20CollectionError(f"run root is not a regular directory: {root}")
    marker = root / RUN_MARKER
    if (
        not marker.is_file()
        or marker.is_symlink()
        or marker.read_text(encoding="utf-8").strip() != RUN_MARKER_VALUE
    ):
        raise Day20CollectionError(f"Day 20 v2 run marker is missing or drifted: {marker}")
    return root


def _artifact(
    role: str, path: Path, *, content_sha256: str, expected_file_sha256: str | None = None
) -> dict[str, str]:
    if path.expanduser().is_symlink():
        raise Day20CollectionError(f"artifact must not be symbolic: {path}")
    resolved = path.resolve()
    actual_file_sha256 = file_sha256(resolved)
    if expected_file_sha256 is not None and actual_file_sha256 != expected_file_sha256:
        raise Day20CollectionError(f"declared file SHA-256 drifted: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "file_sha256": actual_file_sha256,
        "content_sha256": content_sha256,
    }


def _comparison_keys(
    *, normalized: Any = None, e2b: Any = None, complete: Any = None
) -> dict[str, Any]:
    return {"normalized": normalized, "e2b": e2b, "complete": complete}


def _incomplete(
    artifact_type: str, paths: Sequence[Path], reason: str
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "paths": sorted(str(path.resolve()) for path in paths),
        "reason": reason,
    }


def _failure(
    artifact_type: str, paths: Sequence[Path], error: BaseException
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "paths": sorted(str(path.resolve()) for path in paths),
        "failure_layer": "evidence_integrity",
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _collect_training(
    root: Path,
    verifier: Callable[..., dict[str, Any]],
    items: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    for run_kind, scope in (("probe", "probe32"), ("main", "full112")):
        base = root / "evidence" / run_kind
        run_dirs = sorted(base.glob("s*/lr*")) if base.is_dir() else []
        for run_dir in run_dirs:
            summary_path = run_dir / "training-summary.json"
            if not summary_path.exists() and not summary_path.is_symlink():
                incomplete.append(
                    _incomplete(
                        "training",
                        [run_dir],
                        "training evidence directory exists without canonical summary",
                    )
                )
                continue
            try:
                summary = verifier(summary_path, run_root=root)
                if summary.get("run_kind") != run_kind:
                    raise Day20CollectionError("training run kind differs from its path")
                candidates = summary["candidate_ids"]
                checkpoints = summary["checkpoints"]
                packages = summary["checkpoint_packages"]
                targets = summary["checkpoint_tokens"]
                for label in sorted(targets, key=lambda value: int(targets[value])):
                    candidate = candidates[label]
                    checkpoint = Path(checkpoints[label]).resolve()
                    package = packages[label]
                    item = {
                        "kind": "checkpoint",
                        "candidate": candidate,
                        "scope": scope,
                        "comparison_keys": _comparison_keys(),
                        "checkpoint_label": label,
                        "checkpoint_path": str(checkpoint),
                        "target_supervised_tokens": targets[label],
                        "snapshot_sha256": package["snapshot_sha256"],
                        "artifacts": [
                            _artifact(
                                "training_summary",
                                summary_path,
                                content_sha256=summary["summary_sha256"],
                            ),
                            _artifact(
                                "checkpoint_integrity",
                                checkpoint / CHECKPOINT_INTEGRITY_FILE,
                                content_sha256=package["integrity_sha256"],
                                expected_file_sha256=package[
                                    "integrity_file_sha256"
                                ],
                            ),
                        ],
                    }
                    items.append(item)
            except Exception as error:
                failures.append(_failure("training", [summary_path], error))


def _eval_groups(eval_dir: Path) -> dict[str, dict[tuple[str, str], dict[str, Path]]]:
    groups: dict[str, dict[tuple[str, str], dict[str, Path]]] = {
        "raw_evaluation": {},
        "normalized_evaluation": {},
        "sandbox": {},
    }
    if not eval_dir.is_dir():
        return groups
    for path in sorted(eval_dir.iterdir()):
        name = path.name
        for scope in ("probe32", "full112"):
            predictions_suffix = f"-{scope}-raw-predictions-v2.jsonl"
            summary_suffix = f"-{scope}-raw-summary-v2.json"
            if name.endswith(predictions_suffix):
                candidate = name[: -len(predictions_suffix)]
                groups["raw_evaluation"].setdefault((candidate, scope), {})[
                    "predictions"
                ] = path
                break
            if name.endswith(summary_suffix):
                candidate = name[: -len(summary_suffix)]
                groups["raw_evaluation"].setdefault((candidate, scope), {})[
                    "summary"
                ] = path
                break
        else:
            normalized_predictions = ".qwen35-v3.predictions.jsonl"
            normalized_summary = ".qwen35-v3.json"
            e2b_results = "-code-e2b-qwen35-v3.jsonl"
            e2b_summary = "-code-e2b-qwen35-v3-summary.json"
            if name.endswith(e2b_results):
                candidate = name[: -len(e2b_results)]
                groups["sandbox"].setdefault((candidate, "unknown"), {})[
                    "results"
                ] = path
            elif name.endswith(e2b_summary):
                candidate = name[: -len(e2b_summary)]
                groups["sandbox"].setdefault((candidate, "unknown"), {})[
                    "summary"
                ] = path
            elif name.endswith(normalized_predictions):
                candidate = name[: -len(normalized_predictions)]
                groups["normalized_evaluation"].setdefault(
                    (candidate, "unknown"), {}
                )["predictions"] = path
            elif name.endswith(normalized_summary):
                candidate = name[: -len(normalized_summary)]
                groups["normalized_evaluation"].setdefault(
                    (candidate, "unknown"), {}
                )["summary"] = path
    return groups


def _collect_raw(
    groups: Mapping[tuple[str, str], Mapping[str, Path]],
    verifier: Callable[..., dict[str, Any]],
    items: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    for (candidate, scope), parts in sorted(groups.items()):
        paths = list(parts.values())
        if set(parts) != {"predictions", "summary"}:
            incomplete.append(
                _incomplete("raw_evaluation", paths, "published raw pair is incomplete")
            )
            continue
        try:
            summary = verifier(
                parts["predictions"],
                parts["summary"],
                expected_candidate=candidate,
                expected_scope=scope,
            )
            prediction = summary["predictions"]
            items.append(
                {
                    "kind": "raw_evaluation",
                    "candidate": candidate,
                    "scope": scope,
                    "comparison_keys": _comparison_keys(),
                    "metrics": summary["metrics"],
                    "runtime_metrics": summary["runtime_metrics"],
                    "artifacts": [
                        _artifact(
                            "raw_predictions",
                            parts["predictions"],
                            content_sha256=prediction["content_sha256"],
                            expected_file_sha256=prediction["file_sha256"],
                        ),
                        _artifact(
                            "raw_summary",
                            parts["summary"],
                            content_sha256=summary["summary_sha256"],
                        ),
                    ],
                }
            )
        except Exception as error:
            failures.append(_failure("raw_evaluation", paths, error))


def _collect_normalized(
    groups: Mapping[tuple[str, str], Mapping[str, Path]],
    verifier: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]],
    items: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    for (candidate, _), parts in sorted(groups.items()):
        paths = list(parts.values())
        if set(parts) != {"predictions", "summary"}:
            incomplete.append(
                _incomplete(
                    "normalized_evaluation",
                    paths,
                    "published normalized pair is incomplete",
                )
            )
            continue
        try:
            _, summary = verifier(
                parts["predictions"],
                parts["summary"],
                expected_candidate=candidate,
            )
            prediction = summary["predictions"]
            items.append(
                {
                    "kind": "normalized_evaluation",
                    "candidate": candidate,
                    "scope": summary["scope"],
                    "comparison_keys": _comparison_keys(
                        normalized=summary["normalized_comparison_key"]
                    ),
                    "evaluation_run_sha256": summary["evaluation_run_sha256"],
                    "metrics": summary["metrics"],
                    "artifacts": [
                        _artifact(
                            "normalized_predictions",
                            parts["predictions"],
                            content_sha256=prediction["content_sha256"],
                            expected_file_sha256=prediction["file_sha256"],
                        ),
                        _artifact(
                            "normalized_summary",
                            parts["summary"],
                            content_sha256=summary["summary_sha256"],
                        ),
                    ],
                }
            )
        except Exception as error:
            failures.append(_failure("normalized_evaluation", paths, error))


def _collect_e2b(
    groups: Mapping[tuple[str, str], Mapping[str, Path]],
    verifier: Callable[..., tuple[Any, ...]],
    items: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    for (candidate, _), parts in sorted(groups.items()):
        paths = list(parts.values())
        if set(parts) != {"results", "summary"}:
            incomplete.append(
                _incomplete("sandbox", paths, "published E2B pair is incomplete")
            )
            continue
        try:
            _, summary, _, _ = verifier(
                parts["summary"], expected_candidate=candidate
            )
            result = summary["result"]
            items.append(
                {
                    "kind": "sandbox",
                    "candidate": candidate,
                    "scope": summary["scope"],
                    "comparison_keys": _comparison_keys(
                        normalized=summary["normalized_comparison_key"],
                        e2b=summary["e2b_comparison_key"],
                        complete=summary["complete_comparison_key"],
                    ),
                    "evaluation_run_sha256": summary["evaluation_run_sha256"],
                    "e2b_run_sha256": summary["e2b_run_sha256"],
                    "metrics": {
                        key: summary[key]
                        for key in (
                            "records",
                            "code_records",
                            "sandbox_execution_eligible",
                            "passed",
                            "failed",
                            "infrastructure_failures",
                        )
                    },
                    "artifacts": [
                        _artifact(
                            "e2b_results",
                            parts["results"],
                            content_sha256=result["content_sha256"],
                            expected_file_sha256=result["file_sha256"],
                        ),
                        _artifact(
                            "e2b_summary",
                            parts["summary"],
                            content_sha256=summary["summary_sha256"],
                        ),
                    ],
                }
            )
        except Exception as error:
            failures.append(_failure("sandbox", paths, error))


def _selection_item(path: Path, kind: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "probe_selection":
        common = value["common_comparison"]
        candidate = value.get("selected_candidate")
        content_sha = value["selection_sha256"]
        decision = {"status": value["status"], "selected_candidate": candidate}
    elif kind == "primary_selection":
        common = value["common_evaluation_identity"]
        candidate = value["decision"].get("primary_winner")
        content_sha = value["selection_sha256"]
        decision = value["decision"]
    else:
        common = value["common_evaluation_identity"]
        candidate = value["decision"].get("selected_candidate")
        content_sha = value["promotion_sha256"]
        decision = value["decision"]
    return {
        "kind": kind,
        "candidate": candidate,
        "scope": common["scope"],
        "comparison_keys": _comparison_keys(
            normalized=common["normalized_comparison_key"],
            e2b=common["e2b_comparison_key"],
            complete=common["complete_comparison_key"],
        ),
        "decision": decision,
        "artifacts": [
            _artifact("selection", path, content_sha256=content_sha)
        ],
    }


def _collect_selections(
    root: Path,
    verifiers: VerifierBundle,
    items: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    specifications = [
        *((name, "probe_selection", verifiers.probe_selection) for name in PROBE_SELECTIONS),
        (PRIMARY_SELECTION, "primary_selection", verifiers.primary_selection),
        (FINAL_PROMOTION, "final_promotion", verifiers.final_promotion),
    ]
    for name, kind, verifier in specifications:
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        try:
            items.append(_selection_item(path, kind, verifier(path)))
        except Exception as error:
            failures.append(_failure(kind, [path], error))
    for path in sorted(root.glob(".*SELECTION*.tmp")) + sorted(
        root.glob(".*PROMOTION*.tmp")
    ):
        incomplete.append(
            _incomplete("selection", [path], "temporary selection artifact remains")
        )


def _collect_eval_attempts(eval_dir: Path, incomplete: list[dict[str, Any]]) -> None:
    if not eval_dir.is_dir():
        return
    for path in sorted(eval_dir.glob(".*-raw-predictions-v2.jsonl.*.attempt")):
        incomplete.append(
            _incomplete("raw_evaluation", [path], "raw inference attempt is unpublished")
        )


def _link_comparison_keys(items: list[dict[str, Any]]) -> None:
    by_candidate: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["kind"] not in {"normalized_evaluation", "sandbox"}:
            continue
        candidate = item.get("candidate")
        if isinstance(candidate, str):
            current = by_candidate.setdefault(candidate, _comparison_keys())
            for key, value in item["comparison_keys"].items():
                if value is not None:
                    current[key] = value
    for item in items:
        candidate = item.get("candidate")
        if candidate in by_candidate:
            item["comparison_keys"] = dict(by_candidate[candidate])


def collect_evidence(
    run_root: Path, *, verifiers: VerifierBundle | None = None
) -> dict[str, Any]:
    """Return a deterministic inventory without writing to the run or RSI state."""

    root = _validate_run_root(run_root)
    active = verifiers or _load_verifiers()
    items: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    _collect_training(root, active.training, items, incomplete, failures)
    groups = _eval_groups(root / "eval")
    _collect_raw(groups["raw_evaluation"], active.raw, items, incomplete, failures)
    _collect_normalized(
        groups["normalized_evaluation"],
        active.normalized,
        items,
        incomplete,
        failures,
    )
    _collect_e2b(groups["sandbox"], active.e2b, items, incomplete, failures)
    _collect_selections(root, active, items, incomplete, failures)
    _collect_eval_attempts(root / "eval", incomplete)
    _link_comparison_keys(items)

    for item in items:
        item["evidence_sha256"] = object_sha256(item)
    items.sort(key=lambda item: (item["kind"], str(item.get("candidate")), item["evidence_sha256"]))
    incomplete.sort(key=lambda item: (item["artifact_type"], item["paths"], item["reason"]))
    failures.sort(key=lambda item: (item["artifact_type"], item["paths"], item["message"]))
    counts: dict[str, int] = {}
    for item in items:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    status = (
        "verification_failed"
        if failures
        else "operational_incomplete"
        if incomplete
        else "pass"
    )
    result: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_root": str(root),
        "counts": dict(sorted(counts.items())),
        "complete_verified_items": items,
        "operational_incomplete": incomplete,
        "verification_failures": failures,
    }
    result["inventory_sha256"] = object_sha256(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = collect_evidence(args.run_root)
    except Exception as error:
        result = {
            "schema_name": f"{SCHEMA_NAME}.error",
            "schema_version": SCHEMA_VERSION,
            "status": "verification_failed",
            "failure_layer": "collector_input",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if result["status"] == "verification_failed":
        return 2
    if result["status"] == "operational_incomplete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
