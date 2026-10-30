#!/usr/bin/env python3
"""Replay retained promoted-S1 candidates twice in fresh pinned E2B sandboxes.

Input rows are produced by
``formal_s1_pair_labeler.prepare_candidate_replay_requests``.  The output is
the exact per-candidate evidence contract consumed by the formal labeler.
Each worker may process a deterministic hash shard and use bounded thread
concurrency; every individual attempt still creates and destroys a fresh,
network-denied Firecracker sandbox.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import formal_s1_pair_labeler as labeler
import score_day22_mbpp_sandbox as sandbox


class Day22S1CandidateScorerError(ValueError):
    """A candidate request, E2B execution, or shard invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22S1CandidateScorerError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Day22S1CandidateScorerError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Day22S1CandidateScorerError(f"{name} must be non-empty text without NUL")
    return value


def validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(dict(_mapping(value, "request")))
    try:
        labeler._verify_self_hash(request, "request_sha256", "request")
    except labeler.FormalS1LabelerError as error:
        raise Day22S1CandidateScorerError(str(error)) from error
    _require(
        request.get("schema_name") == labeler.CANDIDATE_REPLAY_SCHEMA
        and request.get("schema_version") == labeler.SCHEMA_VERSION,
        "candidate replay request schema drifted",
    )
    candidate_id = _text(request.get("candidate_id"), "candidate_id")
    family_id = _text(request.get("family_id"), "family_id")
    task_id = _text(str(request.get("task_id")), "task_id")
    _require(family_id == f"mbpp:task:{task_id}", "family/task identity drifted")
    candidate = _mapping(request.get("candidate"), "candidate")
    _require(candidate.get("candidate_id") == candidate_id, "candidate ID drifted")
    _require(candidate.get("origin") == labeler.ORIGIN, "candidate origin drifted")
    response = _text(candidate.get("text"), "candidate.text")
    _require(
        candidate.get("sha256") == labeler.contract.text_sha256(response),
        "candidate response hash drifted",
    )
    _text(request.get("code_prefix"), "code_prefix")
    _text(request.get("entry_point"), "entry_point")
    labeler._validate_tests(request.get("tests"), "tests")
    _require(
        request.get("tests_sha256") == request["tests"]["sha256"],
        "tests_sha256 drifted",
    )
    labeler._sha256(
        request.get("generator_provenance_sha256"),
        "generator_provenance_sha256",
    )
    labeler._sha256(
        request.get("source_rollout_candidate_sha256"),
        "source_rollout_candidate_sha256",
    )
    return request


def _finalize_candidate_run(
    request: Mapping[str, Any],
    *,
    attempt: int,
    response: str,
    program: str,
    source_tests_sha256: str,
    test_sha256: str,
    sandbox_identity: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "pair_id": f"{request['candidate_id']}:candidate-replay",
        "task_id": str(request["task_id"]),
        "family_id": str(request["family_id"]),
    }
    run = sandbox._finalize_run(
        identity=identity,
        side="candidate",
        attempt=attempt,
        response=response,
        program=program,
        source_tests_sha256=source_tests_sha256,
        test_sha256=test_sha256,
        sandbox_identity=sandbox_identity,
        outcome=outcome,
    )
    run.pop("run_sha256", None)
    run["candidate_id"] = request["candidate_id"]
    return labeler.seal_run(run)


def score_candidate_request(
    value: Mapping[str, Any],
    *,
    e2b_api_key: str,
    e2b_bindings: Mapping[str, Any],
    wall_timeout_seconds: float = 4.0,
    max_output_bytes: int = 8192,
    execute_program: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one candidate exactly twice and return self-hashed evidence."""

    request = validate_request(value)
    _require(bool(e2b_api_key), "E2B API key is required")
    _require(isinstance(e2b_bindings, Mapping), "pinned E2B bindings are required")
    _require(wall_timeout_seconds > 0, "wall timeout must be positive")
    _require(max_output_bytes > 0, "max output bytes must be positive")
    executor = execute_program or sandbox.execute_e2b_program
    response = str(request["candidate"]["text"])
    # Formal scoring always executes the full source test bundle, including
    # challenge tests when present, so source_tests_sha256 == test_sha256.
    program = sandbox.compose_mbpp_program(
        request,
        response,
        include_challenge_tests=True,
    )
    _, source_tests_sha256, test_sha256 = sandbox.execution_test_bundle(
        request,
        include_challenge_tests=True,
    )
    _require(
        source_tests_sha256 == test_sha256 == request["tests_sha256"],
        "formal candidate scorer did not execute the full frozen test bundle",
    )
    sandbox_identity = sandbox.e2b_sandbox_identity(
        wall_timeout_seconds=wall_timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    runs: list[dict[str, Any]] = []
    for attempt in (1, 2):
        outcome = executor(
            program,
            api_key=e2b_api_key,
            bindings=e2b_bindings,
            wall_timeout_seconds=wall_timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        runs.append(
            _finalize_candidate_run(
                request,
                attempt=attempt,
                response=response,
                program=program,
                source_tests_sha256=source_tests_sha256,
                test_sha256=test_sha256,
                sandbox_identity=sandbox_identity,
                outcome=outcome,
            )
        )
    evidence = {
        "schema_name": labeler.EVIDENCE_SCHEMA,
        "schema_version": labeler.SCHEMA_VERSION,
        "backend": "e2b",
        "candidate_id": request["candidate_id"],
        "task_id": str(request["task_id"]),
        "family_id": request["family_id"],
        "response_sha256": request["candidate"]["sha256"],
        "generator_provenance_sha256": request["generator_provenance_sha256"],
        "source_rollout_candidate_sha256": request[
            "source_rollout_candidate_sha256"
        ],
        "source_request_sha256": request["request_sha256"],
        "source_tests_sha256": source_tests_sha256,
        "test_sha256": test_sha256,
        "sandbox_digest": labeler.contract.object_sha256(sandbox_identity),
        "runs": runs,
    }
    return labeler.seal_evidence(evidence)


def request_shard(candidate_id: str, shard_count: int) -> int:
    _require(isinstance(shard_count, int) and not isinstance(shard_count, bool) and shard_count > 0, "shard_count must be positive")
    return int(labeler.contract.text_sha256(candidate_id), 16) % shard_count


def score_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    e2b_api_key: str,
    e2b_bindings: Mapping[str, Any],
    workers: int,
    shard_id: int,
    shard_count: int,
    wall_timeout_seconds: float,
    max_output_bytes: int,
) -> list[dict[str, Any]]:
    _require(isinstance(workers, int) and not isinstance(workers, bool) and workers > 0, "workers must be positive")
    _require(0 <= shard_id < shard_count, "shard_id must be within shard_count")
    requests = [validate_request(row) for row in rows]
    ids = [str(row["candidate_id"]) for row in requests]
    _require(len(ids) == len(set(ids)), "candidate replay request IDs are not unique")
    selected = sorted(
        (
            row
            for row in requests
            if request_shard(str(row["candidate_id"]), shard_count) == shard_id
        ),
        key=lambda row: str(row["candidate_id"]),
    )
    _require(bool(selected), "selected replay shard is empty")
    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                score_candidate_request,
                row,
                e2b_api_key=e2b_api_key,
                e2b_bindings=e2b_bindings,
                wall_timeout_seconds=wall_timeout_seconds,
                max_output_bytes=max_output_bytes,
            ): str(row["candidate_id"])
            for row in selected
        }
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                completed[candidate_id] = future.result()
            except (sandbox.Day22SandboxError, Day22S1CandidateScorerError) as error:
                raise Day22S1CandidateScorerError(
                    f"candidate {candidate_id} scoring failed: {error}"
                ) from error
    return [completed[str(row["candidate_id"])] for row in selected]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--e2b-credential",
        type=Path,
        default=sandbox.DEFAULT_E2B_CREDENTIAL,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=4.0)
    parser.add_argument("--max-output-bytes", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        api_key = sandbox.read_e2b_api_key(args.e2b_credential.resolve())
        bindings = sandbox.load_e2b_bindings()
        rows = labeler.load_jsonl(args.input.resolve())
        evidence = score_requests(
            rows,
            e2b_api_key=api_key,
            e2b_bindings=bindings,
            workers=args.workers,
            shard_id=args.shard_id,
            shard_count=args.shard_count,
            wall_timeout_seconds=args.timeout_seconds,
            max_output_bytes=args.max_output_bytes,
        )
        sandbox.write_jsonl_atomic(
            evidence,
            args.output.resolve(),
            overwrite=args.overwrite,
        )
    except (
        Day22S1CandidateScorerError,
        sandbox.Day22SandboxError,
        labeler.FormalS1LabelerError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "records": len(evidence),
                "shard_id": args.shard_id,
                "shard_count": args.shard_count,
                "output": str(args.output.resolve()),
                "output_file_sha256": labeler.file_sha256(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
