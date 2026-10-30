#!/usr/bin/env python3
"""Combine frozen base/adaptive S1 rollout rounds without weakening gates.

``combine-rollouts`` validates each round against its own contract, performs
byte-exact cross-round response deduplication (base representatives win), and
emits a 330-family normalized rollout plus E2B requests only for retained new
adaptive candidates.  ``merge-evidence`` then joins the immutable base E2B
file with the new E2B file and requires exact candidate coverage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import day22_contract as contract
import formal_s1_pair_labeler as labeler
import rollout_day22_s1 as base
import rollout_day22_s1_adaptive as adaptive


SCHEMA_VERSION = 1
COMBINE_SCHEMA = "day22.s1_multi_round_rollout_combine_manifest"
EVIDENCE_MERGE_SCHEMA = "day22.s1_multi_round_evidence_merge_manifest"


class Day22RoundCombineError(ValueError):
    """A round, merge membership, or output identity failed validation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22RoundCombineError(message)


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    result[field] = contract.object_sha256(result)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return labeler.load_json(path.resolve())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return labeler.load_jsonl(path.resolve())


def _adaptive_normalized_families(
    rows: Sequence[Mapping[str, Any]],
    *,
    rollout_contract: Mapping[str, Any],
    seed_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate the exact adaptive candidate cohort and group retained uniques."""

    try:
        base.verify_self_hash(rollout_contract, "contract_sha256")
        base.verify_self_hash(seed_manifest, "manifest_sha256")
    except base.Day22RolloutError as error:
        raise Day22RoundCombineError(str(error)) from error
    require(
        rollout_contract.get("schema_name") == adaptive.CONTRACT_SCHEMA
        and rollout_contract.get("schema_version") == adaptive.SCHEMA_VERSION
        and rollout_contract.get("status") == "frozen",
        "adaptive contract schema/status drifted",
    )
    require(
        rollout_contract.get("seed_pool", {}).get("manifest_sha256")
        == seed_manifest.get("manifest_sha256"),
        "adaptive contract/seed identity drifted",
    )
    specs = adaptive.expected_specs(rollout_contract, seed_manifest)
    spec_by_id = {str(spec["candidate_id"]): spec for spec in specs}
    raw_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = copy.deepcopy(dict(raw))
        candidate_id = row.get("candidate_id")
        require(isinstance(candidate_id, str) and candidate_id in spec_by_id, f"unexpected adaptive candidate at row {index}")
        require(candidate_id not in raw_by_id, f"duplicate adaptive candidate_id: {candidate_id}")
        try:
            adaptive.validate_candidate_row(
                row,
                contract=rollout_contract,
                spec=spec_by_id[candidate_id],
            )
        except adaptive.Day22AdaptiveRolloutError as error:
            raise Day22RoundCombineError(
                f"adaptive candidate {candidate_id} violates source contract: {error}"
            ) from error
        raw_by_id[candidate_id] = row
    require(set(raw_by_id) == set(spec_by_id), "adaptive input does not exactly cover its frozen candidates")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_by_id.values():
        grouped.setdefault(str(row["task_family_id"]), []).append(row)
    require(len(grouped) == 189, "adaptive rows must cover exactly 189 families")
    normalized: list[dict[str, Any]] = []
    duplicates = 0
    format_invalid = 0
    context_fields = (
        "task_id",
        "task_family_id",
        "split",
        "problem",
        "prompt",
        "code_prefix",
        "entry_point",
        "tests",
        "family_keys",
        "source",
    )
    for family_id, family_rows in sorted(grouped.items()):
        family_rows.sort(key=lambda row: (int(row["sample_index"]), str(row["candidate_id"])))
        require(len(family_rows) == adaptive.SAMPLES_PER_FAMILY, f"{family_id} adaptive K drifted")
        reference = family_rows[0]
        for row in family_rows[1:]:
            for field in context_fields:
                require(row.get(field) == reference.get(field), f"{family_id}.{field} drifted across adaptive samples")
        generators = [
            labeler._candidate_generator_core(row["generator"], f"{row['candidate_id']}.generator")
            for row in family_rows
        ]
        generator_by_hash = {
            str(generator["provenance_sha256"]): generator for generator in generators
        }
        require(len(generator_by_hash) == 1, f"{family_id} adaptive generator drifted")
        generator = next(iter(generator_by_hash.values()))

        by_response: dict[str, list[tuple[dict[str, Any], bool]]] = {}
        for row in family_rows:
            eligible = labeler._validate_format_contract(
                row["format_contract"], f"{row['candidate_id']}.format_contract"
            )
            by_response.setdefault(str(row["response"]["sha256"]), []).append((row, eligible))
        retained: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for response_sha256, matches in sorted(by_response.items()):
            eligible_values = {eligible for _, eligible in matches}
            require(len(eligible_values) == 1, f"{family_id} duplicate response eligibility drifted")
            matches.sort(key=lambda item: (int(item[0]["sample_index"]), str(item[0]["candidate_id"])))
            representative, eligible = matches[0]
            if not eligible:
                for row, _ in matches:
                    format_invalid += 1
                    exclusions.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "reason_code": "format_contract_not_execution_eligible",
                            "source_candidate_sha256": row["candidate_sha256"],
                        }
                    )
                continue
            candidate = labeler._normalized_rollout_candidate(
                representative,
                family_id=family_id,
                generator=generator,
            )
            labeler._response_ast_sha256(
                candidate["text"],
                str(reference["code_prefix"]),
                str(candidate["candidate_id"]),
            )
            retained.append(candidate)
            for row, _ in matches[1:]:
                duplicates += 1
                exclusions.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "reason_code": "exact_response_duplicate",
                        "duplicate_of_candidate_id": representative["candidate_id"],
                        "response_sha256": response_sha256,
                        "source_candidate_sha256": row["candidate_sha256"],
                    }
                )
        family = {
            "schema_name": labeler.ROLLOUT_SCHEMA,
            "schema_version": labeler.SCHEMA_VERSION,
            "task_id": str(reference["task_id"]),
            "family_id": family_id,
            "split": reference["split"],
            "family_keys": copy.deepcopy(reference["family_keys"]),
            "source": copy.deepcopy(reference["source"]),
            "problem": copy.deepcopy(reference["problem"]),
            "prompt": copy.deepcopy(reference["prompt"]),
            "code_prefix": reference["code_prefix"],
            "entry_point": reference["entry_point"],
            "tests": copy.deepcopy(reference["tests"]),
            "generator": generator,
            "candidates": retained,
            "preselection_exclusions": sorted(exclusions, key=lambda item: str(item["candidate_id"])),
        }
        normalized.append(labeler.seal_rollout(family))
    return normalized, {
        "adaptive_input_candidate_rows": len(rows),
        "adaptive_retained_execution_unique_candidates": sum(len(row["candidates"]) for row in normalized),
        "adaptive_exact_response_duplicates": duplicates,
        "adaptive_format_ineligible_candidates": format_invalid,
    }


def combine_normalized_families(
    base_families: Sequence[Mapping[str, Any]],
    adaptive_families: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge family rows, retaining base candidates on exact cross-round ties."""

    base_by_family = {str(row["family_id"]): copy.deepcopy(dict(row)) for row in base_families}
    adaptive_by_family = {str(row["family_id"]): copy.deepcopy(dict(row)) for row in adaptive_families}
    require(len(base_by_family) == 330 and len(adaptive_by_family) == 189, "round family counts drifted")
    require(set(adaptive_by_family) < set(base_by_family), "adaptive families are not a strict base subset")
    context_fields = (
        "task_id",
        "family_id",
        "split",
        "family_keys",
        "source",
        "problem",
        "prompt",
        "code_prefix",
        "entry_point",
        "tests",
    )
    combined: list[dict[str, Any]] = []
    cross_round_duplicates = 0
    supplemental_retained = 0
    all_generators: dict[str, Mapping[str, Any]] = {}
    for family_id in sorted(base_by_family):
        base_family = base_by_family[family_id]
        family_generators = {
            str(item["provenance_sha256"]): item
            for item in labeler._rollout_generators(base_family)
        }
        candidates = list(base_family["candidates"])
        exclusions = list(base_family.get("preselection_exclusions", []))
        response_to_candidate = {str(item["sha256"]): item for item in candidates}
        if family_id in adaptive_by_family:
            supplement = adaptive_by_family[family_id]
            for field in context_fields:
                require(base_family.get(field) == supplement.get(field), f"{family_id}.{field} drifted across rounds")
            for generator in labeler._rollout_generators(supplement):
                digest = str(generator["provenance_sha256"])
                prior = family_generators.get(digest)
                require(prior is None or prior == generator, f"{family_id} generator hash collision")
                family_generators[digest] = generator
            exclusions.extend(supplement.get("preselection_exclusions", []))
            for candidate in sorted(supplement["candidates"], key=lambda item: str(item["candidate_id"])):
                response_sha256 = str(candidate["sha256"])
                if response_sha256 in response_to_candidate:
                    cross_round_duplicates += 1
                    exclusions.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "reason_code": "cross_round_exact_response_duplicate",
                            "duplicate_of_candidate_id": response_to_candidate[response_sha256]["candidate_id"],
                            "response_sha256": response_sha256,
                            "source_candidate_sha256": candidate.get(
                                "source_candidate_sha256", candidate["candidate_sha256"]
                            ),
                        }
                    )
                    continue
                candidates.append(candidate)
                response_to_candidate[response_sha256] = candidate
                supplemental_retained += 1
        generators = labeler._validate_generator_set(
            list(family_generators.values()), f"{family_id}.combined_generators"
        )
        all_generators.update(
            {str(generator["provenance_sha256"]): generator for generator in generators}
        )
        row = {
            key: copy.deepcopy(value)
            for key, value in base_family.items()
            if key not in {"row_sha256", "generator", "generators", "candidates", "preselection_exclusions"}
        }
        row["generator"] = copy.deepcopy(generators[0])
        row["generators"] = copy.deepcopy(generators)
        row["candidates"] = sorted(candidates, key=lambda item: str(item["candidate_id"]))
        row["preselection_exclusions"] = sorted(
            exclusions, key=lambda item: str(item["candidate_id"])
        )
        combined.append(labeler.seal_rollout(row))
    labeler._validate_generator_set(
        list(all_generators.values()), "combined_rollout_generators"
    )
    return combined, {
        "combined_families": len(combined),
        "combined_retained_candidates": sum(len(row["candidates"]) for row in combined),
        "cross_round_exact_response_duplicates": cross_round_duplicates,
        "supplemental_retained_candidates": supplemental_retained,
    }


def supplemental_requests(combined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = labeler.prepare_candidate_replay_requests(combined)
    rows = [
        row
        for row in prepared["requests"]
        if f":{adaptive.ROUND_ID}:" in str(row["candidate_id"])
    ]
    require(bool(rows), "no retained adaptive candidates remain for E2B")
    return rows


def combine_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    outputs = labeler._preflight_output_paths(
        [args.combined_rollouts_output, args.supplemental_requests_output, args.manifest_output],
        overwrite=False,
    )
    seed = _load_json(args.seed_manifest)
    base_contract = _load_json(args.base_contract)
    adaptive_contract = _load_json(args.adaptive_contract)
    require(
        adaptive_contract.get("parent_round", {}).get("rollout_contract_sha256")
        == base_contract.get("contract_sha256"),
        "adaptive contract does not descend from the supplied base contract",
    )
    base_families, base_counts = labeler._normalize_candidate_rollouts(
        _load_jsonl(args.base_rollouts),
        rollout_contract=base_contract,
        seed_manifest=seed,
    )
    adaptive_families, adaptive_counts = _adaptive_normalized_families(
        _load_jsonl(args.adaptive_rollouts),
        rollout_contract=adaptive_contract,
        seed_manifest=seed,
    )
    combined, combine_counts = combine_normalized_families(base_families, adaptive_families)
    requests = supplemental_requests(combined)
    require(len(requests) == combine_counts["supplemental_retained_candidates"], "supplemental request coverage drifted")
    rollout_payload = labeler._jsonl_bytes(combined)
    request_payload = labeler._jsonl_bytes(requests)
    manifest = seal(
        {
            "schema_name": COMBINE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "ready_for_supplemental_e2b",
            "dedup_policy": "exact response SHA-256; retain base-round representative before adaptive",
            "e2b_gate": "every retained adaptive candidate requires exactly two fresh pinned E2B runs",
            "inputs": {
                "base_rollouts": {"file_sha256": labeler.file_sha256(args.base_rollouts.resolve())},
                "base_contract": {"file_sha256": labeler.file_sha256(args.base_contract.resolve()), "contract_sha256": base_contract["contract_sha256"]},
                "adaptive_rollouts": {"file_sha256": labeler.file_sha256(args.adaptive_rollouts.resolve())},
                "adaptive_contract": {"file_sha256": labeler.file_sha256(args.adaptive_contract.resolve()), "contract_sha256": adaptive_contract["contract_sha256"]},
                "seed_manifest": {"file_sha256": labeler.file_sha256(args.seed_manifest.resolve()), "manifest_sha256": seed["manifest_sha256"]},
            },
            "counts": {**base_counts, **adaptive_counts, **combine_counts, "supplemental_e2b_requests": len(requests)},
            "outputs": {
                "combined_rollouts": {"path": str(outputs[0]), "file_sha256": hashlib.sha256(rollout_payload).hexdigest()},
                "supplemental_requests": {"path": str(outputs[1]), "file_sha256": hashlib.sha256(request_payload).hexdigest()},
            },
            "content_identities": {
                "ordered_combined_rows_sha256": contract.object_sha256([row["row_sha256"] for row in combined]),
                "ordered_supplemental_requests_sha256": contract.object_sha256([row["request_sha256"] for row in requests]),
            },
        },
        "manifest_sha256",
    )
    labeler._write_atomic(outputs[0], rollout_payload, overwrite=False)
    try:
        labeler._write_atomic(outputs[1], request_payload, overwrite=False)
        labeler._write_atomic(outputs[2], labeler._json_bytes(manifest), overwrite=False)
    except BaseException:
        for path in outputs:
            path.unlink(missing_ok=True)
        raise
    return manifest


def merge_evidence(args: argparse.Namespace) -> dict[str, Any]:
    outputs = labeler._preflight_output_paths(
        [args.output, args.manifest_output], overwrite=False
    )
    combine_manifest = _load_json(args.combine_manifest)
    labeler._verify_self_hash(combine_manifest, "manifest_sha256", "combine manifest")
    require(combine_manifest.get("schema_name") == COMBINE_SCHEMA, "wrong combine manifest schema")
    combined_path = args.combined_rollouts.resolve()
    require(
        labeler.file_sha256(combined_path)
        == combine_manifest["outputs"]["combined_rollouts"]["file_sha256"],
        "combined rollout file drifted",
    )
    # Keep the sealed rows byte-faithful for the full label pass below.  The
    # rollout validator derives ``ast_sha256`` on its returned candidate
    # copies, so passing those already-validated (but intentionally unsealed)
    # copies back through ``label_rollouts`` would make the second self-hash
    # check fail for real candidate-per-row inputs.  Validate a copy here for
    # membership discovery, then let ``label_rollouts`` validate the original
    # sealed rows exactly once.
    combined_rows = _load_jsonl(combined_path)
    validated_combined = [
        labeler._validate_rollout(row, index)
        for index, row in enumerate(combined_rows)
    ]
    expected = {
        str(candidate["candidate_id"])
        for family in validated_combined
        for candidate in family["candidates"]
    }
    adaptive_ids = {candidate_id for candidate_id in expected if f":{adaptive.ROUND_ID}:" in candidate_id}
    base_ids = expected - adaptive_ids
    base_evidence = _load_jsonl(args.base_evidence)
    adaptive_evidence = _load_jsonl(args.supplemental_evidence)
    base_by_id = {str(row.get("candidate_id")): row for row in base_evidence}
    adaptive_by_id = {str(row.get("candidate_id")): row for row in adaptive_evidence}
    require(len(base_by_id) == len(base_evidence), "duplicate base evidence candidate ID")
    require(len(adaptive_by_id) == len(adaptive_evidence), "duplicate adaptive evidence candidate ID")
    require(set(base_by_id) == base_ids, "base evidence does not exactly cover retained base candidates")
    require(set(adaptive_by_id) == adaptive_ids, "adaptive evidence does not exactly cover retained adaptive candidates")
    ordered = [
        (base_by_id | adaptive_by_id)[candidate_id]
        for candidate_id in sorted(expected)
    ]
    # Full label execution is the strictest available join validation.  It
    # proves candidate/generator/tests hashes, two fresh runs, global run-ID
    # uniqueness, and the common pinned sandbox before publication.
    result = labeler.label_rollouts(combined_rows, ordered)
    payload = labeler._jsonl_bytes(ordered)
    manifest = seal(
        {
            "schema_name": EVIDENCE_MERGE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "combine_manifest_sha256": combine_manifest["manifest_sha256"],
            "inputs": {
                "base_evidence": {"file_sha256": labeler.file_sha256(args.base_evidence.resolve()), "rows": len(base_evidence)},
                "supplemental_evidence": {"file_sha256": labeler.file_sha256(args.supplemental_evidence.resolve()), "rows": len(adaptive_evidence)},
            },
            "counts": {
                "combined_evidence": len(ordered),
                "eligible_pairs_after_merge": len(result["replays"]),
                "quarantined_families_after_merge": len(result["selections"]) - len(result["replays"]),
            },
            "output": {"path": str(outputs[0]), "file_sha256": hashlib.sha256(payload).hexdigest()},
            "ordered_evidence_sha256": contract.object_sha256([row["evidence_sha256"] for row in ordered]),
        },
        "manifest_sha256",
    )
    labeler._write_atomic(outputs[0], payload, overwrite=False)
    try:
        labeler._write_atomic(outputs[1], labeler._json_bytes(manifest), overwrite=False)
    except BaseException:
        outputs[0].unlink(missing_ok=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    combine = commands.add_parser("combine-rollouts")
    combine.add_argument("--base-rollouts", type=Path, required=True)
    combine.add_argument("--base-contract", type=Path, required=True)
    combine.add_argument("--adaptive-rollouts", type=Path, required=True)
    combine.add_argument("--adaptive-contract", type=Path, required=True)
    combine.add_argument("--seed-manifest", type=Path, required=True)
    combine.add_argument("--combined-rollouts-output", type=Path, required=True)
    combine.add_argument("--supplemental-requests-output", type=Path, required=True)
    combine.add_argument("--manifest-output", type=Path, required=True)
    evidence = commands.add_parser("merge-evidence")
    evidence.add_argument("--combined-rollouts", type=Path, required=True)
    evidence.add_argument("--combine-manifest", type=Path, required=True)
    evidence.add_argument("--base-evidence", type=Path, required=True)
    evidence.add_argument("--supplemental-evidence", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    evidence.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = combine_rollouts(args) if args.command == "combine-rollouts" else merge_evidence(args)
    except (
        Day22RoundCombineError,
        labeler.FormalS1LabelerError,
        base.Day22RolloutError,
        adaptive.Day22AdaptiveRolloutError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
