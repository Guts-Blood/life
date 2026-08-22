#!/usr/bin/env python3
"""Build the deterministic 30-sample Day 10 human-review packet.

The builder accepts only the complete frozen 112-record development run.  It
delegates the run-contract checks to ``analyze_day10_baseline.py`` and then
selects records with a frozen, SHA-256-ranked policy.  Human review fields are
created as pending; this program never claims that a person reviewed a row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import rfc8785


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ANALYZER_PATH = HERE / "analyze_day10_baseline.py"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-frozen-eval-manifest.json"
)
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-qwen3-0.6b-base-predictions.jsonl"
)
DEFAULT_CODE_RESULTS = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "learning/post-training-30-day-bootcamp/artifacts/eval/"
    "day10-base-human-review-packet-30.jsonl"
)
SLICES = ("general", "math", "code", "finance")
ERROR_CATEGORIES = (
    "parse_error",
    "wrong_answer",
    "syntax_error",
    "assertion_error",
    "memory_limit",
    "runtime_error",
    "timeout",
)
PACKET_SIZE = 30
MINIMUM_PER_SLICE = 4
SELECTION_VERSION = "day10_human_review_sha256_v2"
RANK_DOMAIN = "day10.human_review.selection_rank.v1"


class ReviewPacketError(ValueError):
    """A source, selection, or output invariant failed."""


def file_sha256(path: Path) -> str:
    try:
        payload = path.resolve().read_bytes()
    except FileNotFoundError as error:
        raise ReviewPacketError(f"required file is missing: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError) as error:
        raise ReviewPacketError(f"RFC 8785 canonicalization failed: {error}") from error


def semantic_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def selection_rank(sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise ReviewPacketError("sample_id must be a non-empty string")
    material = f"{RANK_DOMAIN}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ReviewPacketError(f"source must be inside the repository: {path}") from error


def _load_analyzer() -> Any:
    if not ANALYZER_PATH.is_file():
        raise ReviewPacketError(f"baseline analyzer is missing: {ANALYZER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "day10_review_packet_analyzer", ANALYZER_PATH
    )
    if spec is None or spec.loader is None:
        raise ReviewPacketError(f"cannot load baseline analyzer: {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ReviewPacketError(f"baseline analyzer import failed: {error}") from error
    required = (
        "analyze_baseline",
        "load_manifest",
        "verify_manifest",
        "load_predictions",
        "load_code_results",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise ReviewPacketError("baseline analyzer API is incomplete")
    return module


def automated_category(
    manifest_record: dict[str, Any],
    prediction: dict[str, Any],
    code_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the frozen scorer class plus review-oriented diagnostic labels."""

    slice_name = manifest_record.get("slice")
    scorer_result = prediction.get("scorer_result")
    if slice_name not in SLICES or not isinstance(scorer_result, dict):
        raise ReviewPacketError("record has no valid slice/scorer result")

    score = scorer_result.get("score")
    parse_status = scorer_result.get("parse_status")
    score_status = scorer_result.get("score_status")
    error_type = scorer_result.get("error_type")
    if slice_name == "code":
        if not (
            score is None
            and parse_status == "ok"
            and score_status == "sandbox_required"
            and error_type == "sandbox_required"
        ):
            raise ReviewPacketError("code generation handoff is invalid")
        if not isinstance(code_result, dict):
            raise ReviewPacketError("code row has no validated sandbox result")
        execution_status = code_result.get("execution_status")
        code_score_status = code_result.get("score_status")
        code_score = code_result.get("score")
        code_error_type = code_result.get("error_type")
        if execution_status == "passed":
            if not (
                code_score_status == "ok"
                and code_score == 1.0
                and code_error_type is None
            ):
                raise ReviewPacketError("passing code result is inconsistent")
            primary = "correct"
        elif execution_status in {"failed", "timeout"}:
            if not (
                code_score_status == "ok"
                and code_score == 0.0
                and code_error_type in ERROR_CATEGORIES
                and code_error_type not in {"parse_error", "wrong_answer"}
            ):
                raise ReviewPacketError("failing code result is inconsistent")
            primary = code_error_type
        else:
            raise ReviewPacketError("code result is not a scored outcome")
        category_basis = {
            "score": code_score,
            "score_status": code_score_status,
            "execution_status": execution_status,
            "error_type": code_error_type,
        }
    elif score == 1 and parse_status == "ok" and score_status == "ok" and error_type is None:
        primary = "correct"
        category_basis = {
            "score": score,
            "parse_status": parse_status,
            "score_status": score_status,
            "error_type": error_type,
        }
    elif (
        score == 0
        and parse_status == "parse_error"
        and score_status == "ok"
        and error_type == "parse_error"
    ):
        primary = "parse_error"
        category_basis = {
            "score": score,
            "parse_status": parse_status,
            "score_status": score_status,
            "error_type": error_type,
        }
    elif (
        score == 0
        and parse_status == "ok"
        and score_status == "ok"
        and error_type == "wrong_answer"
    ):
        primary = "wrong_answer"
        category_basis = {
            "score": score,
            "parse_status": parse_status,
            "score_status": score_status,
            "error_type": error_type,
        }
    else:
        raise ReviewPacketError(
            f"unsupported scorer state for {prediction.get('sample_id')!r}"
        )

    output_tokens = prediction.get("output_token_count")
    generation_limit = manifest_record.get("generation_max_new_tokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or isinstance(generation_limit, bool)
        or not isinstance(generation_limit, int)
        or output_tokens < 0
        or generation_limit <= 0
        or output_tokens > generation_limit
    ):
        raise ReviewPacketError("invalid output/generation token count")

    labels = [primary]
    if primary == "correct":
        labels.append("rare_correct")
    ceiling_hit = output_tokens == generation_limit
    if ceiling_hit:
        # This is a triage candidate, not a human judgment that degeneration occurred.
        labels.append("ceiling_degeneration_candidate")
    return {
        "primary": primary,
        "labels": labels,
        "generation_ceiling_hit": ceiling_hit,
        "basis": category_basis,
    }


def _prepare_candidates(
    manifest_by_id: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    code_results_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        manifest_record = manifest_by_id.get(sample_id)
        if manifest_record is None:
            raise ReviewPacketError(f"prediction is absent from manifest: {sample_id}")
        if prediction.get("evaluation_split") != "dev" or manifest_record.get(
            "evaluation_split"
        ) != "dev":
            raise ReviewPacketError(f"review packet may only contain dev rows: {sample_id}")
        code_result = code_results_by_id.get(sample_id)
        if manifest_record["slice"] == "code" and code_result is None:
            raise ReviewPacketError(f"code result is missing: {sample_id}")
        if manifest_record["slice"] != "code" and code_result is not None:
            raise ReviewPacketError(f"non-code row has a code result: {sample_id}")
        candidates.append(
            {
                "sample_id": sample_id,
                "slice": manifest_record["slice"],
                "manifest_record": manifest_record,
                "prediction": prediction,
                "code_result": code_result,
                "automated_category": automated_category(
                    manifest_record, prediction, code_result
                ),
                "selection_rank_sha256": selection_rank(sample_id),
            }
        )
    if len(candidates) != len({item["sample_id"] for item in candidates}):
        raise ReviewPacketError("candidate sample IDs are duplicated")
    return candidates


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    packet_size: int = PACKET_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Apply mandatory coverage, then fill the packet by SHA-256 rank."""

    if isinstance(packet_size, bool) or not isinstance(packet_size, int) or packet_size <= 0:
        raise ReviewPacketError("packet_size must be a positive integer")
    if len(candidates) < packet_size:
        raise ReviewPacketError(
            f"need at least {packet_size} candidates, found {len(candidates)}"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        sample_id = candidate.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ReviewPacketError("candidate sample_id is invalid")
        if sample_id in by_id:
            raise ReviewPacketError(f"duplicate candidate sample_id: {sample_id}")
        if candidate.get("slice") not in SLICES:
            raise ReviewPacketError(f"candidate slice is invalid: {sample_id}")
        category = candidate.get("automated_category")
        if not isinstance(category, dict) or category.get("primary") not in {
            "correct",
            *ERROR_CATEGORIES,
        }:
            raise ReviewPacketError(f"candidate category is invalid: {sample_id}")
        expected_rank = selection_rank(sample_id)
        if candidate.get("selection_rank_sha256") != expected_rank:
            raise ReviewPacketError(f"candidate selection rank is invalid: {sample_id}")
        by_id[sample_id] = candidate

    ranked = sorted(candidates, key=lambda item: (item["selection_rank_sha256"], item["sample_id"]))
    selected_ids: set[str] = set()
    reasons: dict[str, list[str]] = {}

    def add(candidate: dict[str, Any], reason: str) -> None:
        sample_id = candidate["sample_id"]
        if sample_id not in selected_ids:
            if len(selected_ids) >= packet_size:
                raise ReviewPacketError(
                    "mandatory selection exceeds packet size; policy cannot include all priorities"
                )
            selected_ids.add(sample_id)
            reasons[sample_id] = []
        if reason not in reasons[sample_id]:
            reasons[sample_id].append(reason)

    # Preserve every automated correct result for manual confirmation.
    for candidate in ranked:
        if candidate["automated_category"]["primary"] == "correct":
            add(candidate, "all_rare_correct")

    # Cover every observed error class within every slice, not only globally.
    for slice_name in SLICES:
        for category_name in ERROR_CATEGORIES:
            bucket = [
                item
                for item in ranked
                if item["slice"] == slice_name
                and item["automated_category"]["primary"] == category_name
            ]
            if bucket:
                add(bucket[0], f"error_class_coverage:{slice_name}:{category_name}")

    # Make slice coverage meaningful even when one slice has few/no correct rows.
    for slice_name in SLICES:
        while sum(by_id[item_id]["slice"] == slice_name for item_id in selected_ids) < min(
            MINIMUM_PER_SLICE, packet_size
        ):
            candidate = next(
                (
                    item
                    for item in ranked
                    if item["slice"] == slice_name and item["sample_id"] not in selected_ids
                ),
                None,
            )
            if candidate is None:
                raise ReviewPacketError(
                    f"cannot satisfy minimum slice coverage for {slice_name}"
                )
            add(candidate, f"minimum_slice_coverage:{slice_name}")

    # A ceiling hit is a degeneration candidate only; human review decides the label.
    candidate = next(
        (
            item
            for item in ranked
            if item["sample_id"] in selected_ids
            and item["automated_category"]["generation_ceiling_hit"]
        ),
        None,
    )
    if candidate is None:
        candidate = next(
            (
                item
                for item in ranked
                if item["automated_category"]["generation_ceiling_hit"]
            ),
            None,
        )
        if candidate is None:
            raise ReviewPacketError("no generation-ceiling candidate is available")
    add(candidate, "ceiling_degeneration_candidate_coverage")

    for candidate in ranked:
        if len(selected_ids) == packet_size:
            break
        if candidate["sample_id"] not in selected_ids:
            add(candidate, "sha256_rank_fill")

    if len(selected_ids) != packet_size:
        raise ReviewPacketError(
            f"selection produced {len(selected_ids)} rows, expected {packet_size}"
        )
    selected = [item for item in ranked if item["sample_id"] in selected_ids]
    return selected, reasons


def _source_contract(
    manifest_path: Path,
    predictions_path: Path,
    code_results_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ReviewPacketError("baseline summary has no provenance")
    required = (
        "manifest_hash",
        "protocol_hash",
        "scorer_registry_hash",
        "execution_protocol_hash",
        "comparison_key",
        "run_hash",
        "model_snapshot_hash",
    )
    if any(not isinstance(provenance.get(field), str) for field in required):
        raise ReviewPacketError("baseline provenance is incomplete")
    code_sandbox = provenance.get("code_sandbox")
    if not isinstance(code_sandbox, dict):
        raise ReviewPacketError("baseline provenance has no code sandbox evidence")
    required_code_fields = (
        "results_file_sha256",
        "results_semantic_hash",
        "code_run_hash",
        "code_execution_protocol_hash",
        "complete_comparison_key",
    )
    if any(
        not isinstance(code_sandbox.get(field), str)
        for field in required_code_fields
    ):
        raise ReviewPacketError("code sandbox provenance is incomplete")
    return {
        "manifest": {
            "path": _repo_relative(manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "semantic_hash": provenance["manifest_hash"],
        },
        "predictions": {
            "path": _repo_relative(predictions_path),
            "file_sha256": file_sha256(predictions_path),
            "run_hash": provenance["run_hash"],
        },
        "code_results": {
            "path": _repo_relative(code_results_path),
            "file_sha256": file_sha256(code_results_path),
            "semantic_hash": code_sandbox["results_semantic_hash"],
            "code_run_hash": code_sandbox["code_run_hash"],
        },
        "validation": {
            "path": _repo_relative(ANALYZER_PATH),
            "source_sha256": file_sha256(ANALYZER_PATH),
        },
        "identities": {
            "protocol_hash": provenance["protocol_hash"],
            "scorer_registry_hash": provenance["scorer_registry_hash"],
            "execution_protocol_hash": provenance["execution_protocol_hash"],
            "comparison_key": provenance["comparison_key"],
            "complete_comparison_key": code_sandbox[
                "complete_comparison_key"
            ],
            "code_execution_protocol_hash": code_sandbox[
                "code_execution_protocol_hash"
            ],
            "model_snapshot_hash": provenance["model_snapshot_hash"],
        },
    }


def build_review_rows(
    manifest_path: Path = DEFAULT_MANIFEST,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    code_results_path: Path = DEFAULT_CODE_RESULTS,
) -> list[dict[str, Any]]:
    """Validate formal sources and return exactly 30 pending review rows."""

    analyzer = _load_analyzer()
    try:
        summary = analyzer.analyze_baseline(
            manifest_path,
            predictions_path,
            code_results_path=code_results_path,
        )
        manifest, manifest_file_sha256 = analyzer.load_manifest(manifest_path)
        manifest_by_id, dev_order, _ = analyzer.verify_manifest(
            manifest, manifest_file_sha256
        )
        predictions = analyzer.load_predictions(predictions_path)
        code_results, _ = analyzer.load_code_results(code_results_path)
    except analyzer.Day10AnalysisError as error:
        raise ReviewPacketError(f"baseline source validation failed: {error}") from error

    if [row.get("sample_id") for row in predictions] != dev_order:
        raise ReviewPacketError("predictions are not the frozen dev order")
    code_results_by_id = {row["sample_id"]: row for row in code_results}
    expected_code_ids = {
        sample_id
        for sample_id in dev_order
        if manifest_by_id[sample_id]["slice"] == "code"
    }
    if set(code_results_by_id) != expected_code_ids:
        raise ReviewPacketError("code results do not cover the frozen dev code rows")
    candidates = _prepare_candidates(
        manifest_by_id, predictions, code_results_by_id
    )
    selected, reasons = select_candidates(candidates)
    source = _source_contract(
        manifest_path, predictions_path, code_results_path, summary
    )
    selected_ids = [item["sample_id"] for item in selected]
    policy_material = {
        "domain": "day10.human_review_selection",
        "schema_version": 1,
        "version": SELECTION_VERSION,
        "evaluation_split": "dev",
        "packet_size": PACKET_SIZE,
        "rank_definition": f"sha256({RANK_DOMAIN} + NUL + sample_id)",
        "priorities": [
            "all_automated_correct_as_rare_correct",
            "each_observed_error_class_within_each_slice",
            f"minimum_{MINIMUM_PER_SLICE}_records_per_slice",
            "at_least_one_generation_ceiling_degeneration_candidate",
            "sha256_rank_fill",
        ],
        "selected_sample_ids_in_rank_order": selected_ids,
        "source": source,
    }
    policy = {
        **policy_material,
        "selection_hash": semantic_hash(policy_material),
    }

    rows: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(selected):
        manifest_record = candidate["manifest_record"]
        prediction = candidate["prediction"]
        code_result = candidate["code_result"]
        code_execution_result = None
        if code_result is not None:
            code_execution_result = {
                "execution_status": code_result["execution_status"],
                "score_status": code_result["score_status"],
                "score": code_result["score"],
                "error_type": code_result["error_type"],
                "failure_message": code_result["failure_message"],
                "scorer_result": code_result["scorer_result"],
                "code_run_hash": code_result["code_run_hash"],
                "complete_comparison_key": code_result[
                    "complete_comparison_key"
                ],
                "code_execution_protocol_hash": code_result[
                    "code_execution_protocol_hash"
                ],
            }
        rows.append(
            {
                "schema_version": 1,
                "packet_ordinal": ordinal,
                "sample_id": candidate["sample_id"],
                "evaluation_split": "dev",
                "slice": candidate["slice"],
                "automated_category": candidate["automated_category"]["primary"],
                "automated_flags": candidate["automated_category"]["labels"][1:],
                "automated_category_basis": candidate["automated_category"]["basis"],
                "generation_ceiling_hit": candidate["automated_category"][
                    "generation_ceiling_hit"
                ],
                "raw_prompt": manifest_record["raw_prompt"],
                "reference": manifest_record["reference"],
                "raw_output": prediction["raw_output"],
                "scorer_result": prediction["scorer_result"],
                "code_execution_result": code_execution_result,
                "token_counts": {
                    "input": prediction["input_token_count"],
                    "output": prediction["output_token_count"],
                    "total": prediction["total_token_count"],
                    "generation_max_new_tokens": manifest_record[
                        "generation_max_new_tokens"
                    ],
                },
                "selection": {
                    "rank_sha256": candidate["selection_rank_sha256"],
                    "reasons": reasons[candidate["sample_id"]],
                },
                "source": source,
                "selection_policy": policy,
                "human_review": {
                    "status": "pending",
                    "judgment": None,
                    "notes": None,
                },
            }
        )
    return rows


def serialize_rows(rows: list[dict[str, Any]]) -> bytes:
    if len(rows) != PACKET_SIZE:
        raise ReviewPacketError(f"packet must contain exactly {PACKET_SIZE} rows")
    if any(
        row.get("evaluation_split") != "dev"
        or row.get("human_review")
        != {"status": "pending", "judgment": None, "notes": None}
        for row in rows
    ):
        raise ReviewPacketError("packet contains a non-dev or non-pending row")
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def write_packet_atomic(
    rows: list[dict[str, Any]], output_path: Path, *, overwrite: bool
) -> None:
    payload = serialize_rows(rows)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise ReviewPacketError(
            f"output already exists (pass --overwrite to replace it): {output_path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise ReviewPacketError(
                    f"output appeared while writing and was not replaced: {output_path}"
                ) from error
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_packet(rows: list[dict[str, Any]], output_path: Path) -> None:
    expected = serialize_rows(rows)
    try:
        actual = output_path.resolve().read_bytes()
    except FileNotFoundError as error:
        raise ReviewPacketError(f"review packet is missing: {output_path}") from error
    if actual != expected:
        raise ReviewPacketError(
            "review packet differs from deterministic rebuild: "
            f"expected sha256={hashlib.sha256(expected).hexdigest()}, "
            f"got sha256={hashlib.sha256(actual).hexdigest()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--code-results", type=Path, default=DEFAULT_CODE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        rows = build_review_rows(
            args.manifest, args.predictions, args.code_results
        )
        if args.verify_only:
            verify_packet(rows, args.output)
            action = "verified"
        else:
            write_packet_atomic(rows, args.output, overwrite=args.overwrite)
            action = "written"
    except ReviewPacketError as error:
        parser.error(str(error))
    payload_sha256 = hashlib.sha256(serialize_rows(rows)).hexdigest()
    print(
        f"review_packet={action} rows={len(rows)} "
        f"sha256={payload_sha256} path={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
