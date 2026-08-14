#!/usr/bin/env python3
"""Adaptive, no-overwrite Day 22 rollout for first-round S1 quarantines.

The frozen first-round selections decide *which* families are sampled and
which of three fixed generation profiles is used.  The profile never changes
the model: every candidate is still sampled on-policy from the exact promoted
S1 merged export bound by the parent rollout contract.

Two ordinary processes run disjoint family shards.  Candidate rows are
published atomically and can be resumed; neither this module nor its merge
command writes to the first-round output namespace.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import formal_s1_pair_labeler as labeler
import rollout_day22_s1 as base


SCHEMA_VERSION = 1
CONTRACT_SCHEMA = "day22.s1_adaptive_rollout_contract"
SHARD_SCHEMA = "day22.s1_adaptive_rollout_shard_summary"
MERGE_SCHEMA = "day22.s1_adaptive_rollout_merge_manifest"
ROUND_ID = "adaptive-r1"
EXPECTED_SHARDS = 2
SAMPLES_PER_FAMILY = 12
FIRST_SUPPLEMENTAL_SAMPLE_INDEX = base.EXPECTED_K
TARGET_REASON_COUNTS = {
    "both_fail": 85,
    "both_pass": 101,
    "runtime_error_not_preference_eligible": 3,
}
REASON_TO_PROFILE = {
    "both_pass": "supplemental_t08",
    "both_fail": "supplemental_t08",
    "runtime_error_not_preference_eligible": "supplemental_t08",
}
GENERATION_PROFILES = {
    "supplemental_t08": {
        "max_new_tokens": 512,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "response_boundary": "exact_generated_token_ids_decode",
        "response_repair": False,
        "samples_per_family": SAMPLES_PER_FAMILY,
        "adaptive_objective": "opposite_class_discovery_for_all_execution_quarantines",
    },
}
PARTITION_DOMAIN = "day22.s1_adaptive_rollout.family_partition.v1"
SEED_DOMAIN = "day22.s1_adaptive_rollout.candidate_seed.v1"


class Day22AdaptiveRolloutError(ValueError):
    """An adaptive target, provenance binding, or durable output drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Day22AdaptiveRolloutError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day22AdaptiveRolloutError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day22AdaptiveRolloutError(
                f"invalid JSONL: {path}:{line_number}"
            ) from error
        require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    require(bool(rows), f"JSONL is empty: {path}")
    return rows


def target_rows(
    selections: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    seed_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the first-round decision file and freeze the 189 targets."""

    try:
        labeler._verify_self_hash(summary, "summary_sha256", "selection summary")
    except labeler.FormalS1LabelerError as error:
        raise Day22AdaptiveRolloutError(str(error)) from error
    require(
        summary.get("schema_name") == labeler.SUMMARY_SCHEMA
        and summary.get("schema_version") == labeler.SCHEMA_VERSION
        and summary.get("status") == "pass",
        "first-round selection summary schema/status drifted",
    )
    identities = summary.get("content_identities")
    require(isinstance(identities, Mapping), "selection content identities are missing")
    require(
        identities.get("ordered_selection_sha256")
        == base.object_sha256([row.get("selection_sha256") for row in selections]),
        "selection rows do not match the first-round summary",
    )
    counts = summary.get("counts")
    require(isinstance(counts, Mapping), "selection summary counts are missing")
    require(
        counts.get("rollout_families") == base.EXPECTED_FAMILIES
        and counts.get("eligible_pairs") == 141
        and counts.get("quarantined_families") == 189,
        "first-round 141/189 family outcome drifted",
    )
    seed_by_family = {
        str(row["task_family_id"]): row for row in seed_manifest.get("records", [])
    }
    require(len(seed_by_family) == base.EXPECTED_FAMILIES, "seed family set drifted")
    seen: set[str] = set()
    targets: list[dict[str, Any]] = []
    for index, raw in enumerate(selections):
        row = copy.deepcopy(dict(raw))
        try:
            labeler._verify_self_hash(row, "selection_sha256", f"selection[{index}]")
        except labeler.FormalS1LabelerError as error:
            raise Day22AdaptiveRolloutError(str(error)) from error
        require(
            row.get("schema_name") == labeler.SELECTION_SCHEMA
            and row.get("schema_version") == labeler.SCHEMA_VERSION,
            f"selection[{index}] schema drifted",
        )
        family_id = row.get("family_id")
        require(isinstance(family_id, str) and family_id in seed_by_family, "selection family is unknown")
        require(family_id not in seen, f"duplicate first-round selection family: {family_id}")
        seen.add(family_id)
        if row.get("selection_status") != labeler.QUARANTINE:
            continue
        decision = row.get("labeler")
        require(isinstance(decision, Mapping), f"{family_id} labeler decision is missing")
        reason = decision.get("reason_code")
        require(reason in REASON_TO_PROFILE, f"unsupported supplemental reason: {reason}")
        targets.append(
            {
                "family_id": family_id,
                "task_id": str(row["task_id"]),
                "split": row["split"],
                "first_round_reason": reason,
                "profile": REASON_TO_PROFILE[str(reason)],
                "source_selection_sha256": row["selection_sha256"],
                "targeting_policy": "all_first_round_execution_quarantines",
            }
        )
    require(len(seen) == base.EXPECTED_FAMILIES, "selection file must cover all 330 families")
    reasons = Counter(str(item["first_round_reason"]) for item in targets)
    require(dict(sorted(reasons.items())) == TARGET_REASON_COUNTS, "target reason counts drifted")
    require(len(targets) == 189, "adaptive stage 1 must target all 189 quarantined families")
    return sorted(
        targets,
        key=lambda item: (
            base.text_sha256(f"{PARTITION_DOMAIN}\0{item['family_id']}"),
            str(item["family_id"]),
        ),
    )


def candidate_seed(family_id: str, profile: str, supplemental_index: int) -> int:
    payload = (
        f"{SEED_DOMAIN}\0{ROUND_ID}\0{family_id}\0{profile}\0{supplemental_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def candidate_id(family_id: str, profile: str, sample_index: int) -> str:
    return (
        f"{family_id}:s1:{base.WINNER}:{ROUND_ID}:{profile}:sample:{sample_index:02d}"
    )


def expected_specs(
    contract: Mapping[str, Any], seed_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = {
        str(row["task_family_id"]): row for row in seed_manifest.get("records", [])
    }
    targets = contract.get("targeting", {}).get("families")
    require(isinstance(targets, list), "adaptive target families are missing")
    specs: list[dict[str, Any]] = []
    for family_ordinal, target in enumerate(targets):
        require(isinstance(target, Mapping), "adaptive target is not an object")
        family_id = str(target.get("family_id"))
        profile = str(target.get("profile"))
        require(family_id in records, f"adaptive family missing from seed: {family_id}")
        require(profile in GENERATION_PROFILES, f"unknown generation profile: {profile}")
        shard_id = family_ordinal % EXPECTED_SHARDS
        for supplemental_index in range(SAMPLES_PER_FAMILY):
            sample_index = FIRST_SUPPLEMENTAL_SAMPLE_INDEX + supplemental_index
            specs.append(
                {
                    "candidate_id": candidate_id(family_id, profile, sample_index),
                    "family_ordinal": family_ordinal,
                    "profile": profile,
                    "record": records[family_id],
                    "sample_index": sample_index,
                    "supplemental_index": supplemental_index,
                    "seed": candidate_seed(family_id, profile, supplemental_index),
                    "shard_id": shard_id,
                    "target": dict(target),
                }
            )
    return specs


def build_contract(
    *,
    parent_contract_path: Path,
    parent_contract: Mapping[str, Any],
    selections_path: Path,
    selections: Sequence[Mapping[str, Any]],
    summary_path: Path,
    summary: Mapping[str, Any],
    seed_path: Path,
    seed_manifest: Mapping[str, Any],
    output_root: Path,
    implementation_sha256: str,
) -> dict[str, Any]:
    targets = target_rows(selections, summary, seed_manifest)
    contract: dict[str, Any] = {
        "schema_name": CONTRACT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "round_id": ROUND_ID,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": implementation_sha256,
        },
        "parent_round": {
            "rollout_contract_path": str(parent_contract_path.resolve()),
            "rollout_contract_file_sha256": base.file_sha256(parent_contract_path.resolve()),
            "rollout_contract_sha256": parent_contract["contract_sha256"],
            "selection_path": str(selections_path.resolve()),
            "selection_file_sha256": base.file_sha256(selections_path.resolve()),
            "ordered_selection_sha256": summary["content_identities"]["ordered_selection_sha256"],
            "selection_summary_path": str(summary_path.resolve()),
            "selection_summary_file_sha256": base.file_sha256(summary_path.resolve()),
            "selection_summary_sha256": summary["summary_sha256"],
            "eligible_pairs": 141,
            "quarantined_families": 189,
        },
        "seed_pool": {
            "path": str(seed_path.resolve()),
            "file_sha256": base.file_sha256(seed_path.resolve()),
            "manifest_sha256": seed_manifest["manifest_sha256"],
            "families": base.EXPECTED_FAMILIES,
        },
        "s1": copy.deepcopy(parent_contract["s1"]),
        "runtime": copy.deepcopy(parent_contract["runtime"]),
        "targeting": {
            "policy": "first_round_execution_quarantine_only",
            "reason_to_profile": dict(sorted(REASON_TO_PROFILE.items())),
            "reason_counts": dict(sorted(TARGET_REASON_COUNTS.items())),
            "families": targets,
            "families_sha256": base.object_sha256(targets),
        },
        "generation_profiles": copy.deepcopy(GENERATION_PROFILES),
        "partition": {
            "domain": PARTITION_DOMAIN,
            "family_unit": "native_mbpp_task_id",
            "shards": EXPECTED_SHARDS,
            "families_per_shard": {"0": 95, "1": 94},
            "candidates_per_shard": {"0": 1140, "1": 1128},
            "total_families": 189,
            "total_candidates": 189 * SAMPLES_PER_FAMILY,
        },
        "output_root": str(output_root.resolve()),
        "non_overwrite": {
            "parent_round_is_read_only": True,
            "separate_output_namespace_required": True,
        },
    }
    require(
        Path(parent_contract["output_root"]).resolve() != output_root.resolve(),
        "adaptive output root must differ from the first-round output root",
    )
    contract["contract_sha256"] = base.object_sha256(contract)
    specs = expected_specs(contract, seed_manifest)
    require(len(specs) == 2268, "adaptive candidate count drifted")
    require(sum(item["shard_id"] == 0 for item in specs) == 1140, "shard 0 imbalance")
    require(sum(item["shard_id"] == 1 for item in specs) == 1128, "shard 1 imbalance")
    return contract


def verify_contract(
    path: Path, *, live: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = base.load_json(path.resolve())
    base.verify_self_hash(contract, "contract_sha256")
    require(
        contract.get("schema_name") == CONTRACT_SCHEMA
        and contract.get("schema_version") == SCHEMA_VERSION
        and contract.get("status") == "frozen"
        and contract.get("round_id") == ROUND_ID,
        "wrong adaptive rollout contract schema/status",
    )
    require(contract.get("generation_profiles") == GENERATION_PROFILES, "generation profiles drifted")
    require(contract.get("partition", {}).get("total_candidates") == 2268, "candidate count drifted")
    seed_path = Path(str(contract["seed_pool"]["path"]))
    seed = base.validate_seed_manifest(seed_path)
    require(
        base.file_sha256(seed_path) == contract["seed_pool"]["file_sha256"]
        and seed["manifest_sha256"] == contract["seed_pool"]["manifest_sha256"],
        "adaptive seed manifest drifted",
    )
    parent_path = Path(str(contract["parent_round"]["rollout_contract_path"]))
    parent, parent_seed = base.verify_contract(parent_path, live=live)
    require(parent_seed["manifest_sha256"] == seed["manifest_sha256"], "parent/adaptive seed drifted")
    require(
        base.file_sha256(parent_path) == contract["parent_round"]["rollout_contract_file_sha256"]
        and parent["contract_sha256"] == contract["parent_round"]["rollout_contract_sha256"],
        "parent rollout contract drifted",
    )
    selections_path = Path(str(contract["parent_round"]["selection_path"]))
    summary_path = Path(str(contract["parent_round"]["selection_summary_path"]))
    require(
        base.file_sha256(selections_path) == contract["parent_round"]["selection_file_sha256"]
        and base.file_sha256(summary_path)
        == contract["parent_round"]["selection_summary_file_sha256"],
        "first-round selection artifacts drifted",
    )
    selections = load_jsonl(selections_path)
    summary = base.load_json(summary_path)
    targets = target_rows(selections, summary, seed)
    require(targets == contract["targeting"]["families"], "adaptive target set drifted")
    require(base.object_sha256(targets) == contract["targeting"]["families_sha256"], "target hash drifted")
    require(contract["s1"] == parent["s1"] and contract["runtime"] == parent["runtime"], "promoted S1/runtime drifted")
    require(
        Path(str(contract["output_root"])).resolve() != Path(parent["output_root"]).resolve(),
        "adaptive output overlaps parent output",
    )
    if live:
        require(
            base.file_sha256(Path(__file__).resolve()) == contract["implementation"]["sha256"],
            "adaptive rollout implementation drifted",
        )
    return contract, seed


def contract_view(contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    """Return the candidate-row view expected by the shared base row builder."""

    view = copy.deepcopy(dict(contract))
    view["generation"] = copy.deepcopy(contract["generation_profiles"][profile])
    return view


def row_path(output_root: Path, shard_id: int, candidate: str) -> Path:
    return output_root / "shards" / f"shard-{shard_id}" / "rows" / f"{base.text_sha256(candidate)}.json"


def validate_candidate_row(
    row: Mapping[str, Any], *, contract: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    view = contract_view(contract, str(spec["profile"]))
    try:
        base.validate_candidate_row(row, contract=view, spec=spec)
    except base.Day22RolloutError as error:
        raise Day22AdaptiveRolloutError(str(error)) from error
    generator = row.get("generator")
    require(isinstance(generator, Mapping), "adaptive candidate generator is missing")
    require(
        generator.get("generation") == contract["generation_profiles"][spec["profile"]]
        and generator.get("generation_config_sha256")
        == base.object_sha256(contract["generation_profiles"][spec["profile"]]),
        "adaptive candidate generation profile drifted",
    )
    require(
        row.get("sample_index") == spec["sample_index"]
        and row.get("generation_seed") == spec["seed"],
        "adaptive sample identity drifted",
    )


def load_or_none(
    path: Path, *, contract: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = base.load_json(path)
    validate_candidate_row(row, contract=contract, spec=spec)
    return row


def run_shard(contract_path: Path, shard_id: int) -> None:
    require(shard_id in range(EXPECTED_SHARDS), "shard ID must be 0 or 1")
    contract, seed = verify_contract(contract_path, live=True)
    ms_swift_root = Path(os.environ.get("DAY22_MS_SWIFT_ROOT", "/root/autodl-tmp/ms-swift"))
    require(base.runtime_identity(ms_swift_root) == contract["runtime"], "live runtime drifted")
    specs = [item for item in expected_specs(contract, seed) if item["shard_id"] == shard_id]
    output_root = Path(contract["output_root"])
    pending = [
        spec
        for spec in specs
        if load_or_none(
            row_path(output_root, shard_id, spec["candidate_id"]),
            contract=contract,
            spec=spec,
        )
        is None
    ]
    if not pending:
        publish_shard_summary(contract, specs, shard_id)
        print(json.dumps({"shard_id": shard_id, "status": "already_complete", "rows": len(specs)}))
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("USE_HF", "1")
    os.environ["USE_MCORE_GDN"] = "0"
    try:
        import torch
        from swift import get_model_processor, get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
        from day20_contract_v2 import Day20V2ContractError, validate_raw_code_continuation
        from qwen35_response_adapter_v3 import adapt_generated_token_decode
    except ImportError as error:
        raise Day22AdaptiveRolloutError("Qwen3.5 rollout runtime is unavailable") from error
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "worker must see exactly one CUDA GPU")
    model_path = Path(contract["s1"]["merged_export"]["path"])
    model, processor = get_model_processor(
        str(model_path),
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_hf=True,
        download_model=False,
    )
    require(model.__class__.__name__ == "Qwen3_5ForConditionalGeneration", "wrong model class")
    template = get_template(
        processor,
        max_length=4096,
        template_type="qwen3_5",
        padding_free=False,
        enable_thinking=False,
        add_non_thinking_prefix=True,
    )
    meta = getattr(template, "template_meta", None)
    require(
        getattr(meta, "template_type", None) == "qwen3_5"
        and getattr(template, "enable_thinking", None) is False
        and getattr(template, "add_non_thinking_prefix", None) is True,
        "Qwen3.5 inference template drifted",
    )
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    for ordinal, spec in enumerate(pending, 1):
        seed_value = spec["seed"]
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        record = spec["record"]
        profile = contract["generation_profiles"][spec["profile"]]
        started = time.monotonic()
        response = engine.infer(
            [InferRequest(messages=[{"role": "user", "content": record["prompt"]["text"]}])],
            RequestConfig(
                max_tokens=profile["max_new_tokens"],
                temperature=profile["temperature"],
                top_p=profile["top_p"],
                num_beams=profile["num_beams"],
                repetition_penalty=profile["repetition_penalty"],
                seed=seed_value,
                return_details=True,
            ),
            use_tqdm=False,
        )[0]
        require(bool(getattr(response, "choices", None)), "ms-swift returned no choice")
        choice = response.choices[0]
        prompt_ids = getattr(response, "prompt_token_ids", None)
        generated_ids = getattr(choice, "token_ids", None)
        require(isinstance(prompt_ids, list) and isinstance(generated_ids, list), "token evidence missing")
        generated_text = template.decode_generate_ids(generated_ids, first_token=False)
        message_content = getattr(getattr(choice, "message", None), "content", "") or ""
        require(isinstance(generated_text, str) and isinstance(message_content, str), "response text missing")
        adapter = adapt_generated_token_decode(
            message_content,
            generated_token_ids=generated_ids,
            generated_only_text=generated_text,
        )
        try:
            validated = validate_raw_code_continuation(generated_text, record["code_prefix"])
            format_contract: dict[str, Any] = {
                "valid": True,
                "execution_eligible": True,
                "validator": "validate_raw_code_continuation",
                "evidence": validated.as_evidence(),
                "error": None,
            }
        except Day20V2ContractError as error:
            format_contract = {
                "valid": False,
                "execution_eligible": False,
                "validator": "validate_raw_code_continuation",
                "evidence": None,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        view = contract_view(contract, str(spec["profile"]))
        row = base.build_candidate_row(
            contract=view,
            spec=spec,
            message_content=message_content,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            generated_text=generated_text,
            finish_reason=str(getattr(choice, "finish_reason", None) or "unknown"),
            elapsed_seconds=time.monotonic() - started,
            response_adapter=adapter,
            format_contract=format_contract,
        )
        validate_candidate_row(row, contract=contract, spec=spec)
        base.write_json_new(row_path(output_root, shard_id, spec["candidate_id"]), row)
        print(
            json.dumps(
                {
                    "shard_id": shard_id,
                    "completed_this_run": ordinal,
                    "pending_at_start": len(pending),
                    "candidate_id": spec["candidate_id"],
                    "profile": spec["profile"],
                    "tokens": len(generated_ids),
                    "format_valid": format_contract["valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    publish_shard_summary(contract, specs, shard_id)


def publish_shard_summary(
    contract: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], shard_id: int
) -> None:
    rows = []
    output_root = Path(contract["output_root"])
    for spec in specs:
        row = load_or_none(
            row_path(output_root, shard_id, spec["candidate_id"]),
            contract=contract,
            spec=spec,
        )
        require(row is not None, f"adaptive shard {shard_id} is incomplete")
        rows.append(row)
    summary: dict[str, Any] = {
        "schema_name": SHARD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "round_id": ROUND_ID,
        "shard_id": shard_id,
        "shard_count": EXPECTED_SHARDS,
        "rollout_contract_sha256": contract["contract_sha256"],
        "rows": len(rows),
        "candidate_ids_sha256": base.object_sha256([row["candidate_id"] for row in rows]),
        "candidate_rows_sha256": base.object_sha256([row["candidate_sha256"] for row in rows]),
    }
    summary["summary_sha256"] = base.object_sha256(summary)
    base.publish_or_verify_json(
        output_root / "shards" / f"shard-{shard_id}" / "SHARD-COMPLETE.json",
        summary,
        "summary_sha256",
    )


def rollout_status(contract_path: Path) -> dict[str, Any]:
    contract, seed = verify_contract(contract_path, live=False)
    output_root = Path(contract["output_root"])
    result: dict[str, Any] = {
        "rollout_contract_sha256": contract["contract_sha256"],
        "expected": contract["partition"]["total_candidates"],
        "shards": {},
    }
    total = 0
    specs = expected_specs(contract, seed)
    for shard_id in range(EXPECTED_SHARDS):
        shard_specs = [item for item in specs if item["shard_id"] == shard_id]
        present = sum(
            load_or_none(
                row_path(output_root, shard_id, spec["candidate_id"]),
                contract=contract,
                spec=spec,
            )
            is not None
            for spec in shard_specs
        )
        result["shards"][str(shard_id)] = {"expected": len(shard_specs), "present": present}
        total += present
    result["present"] = total
    result["complete"] = total == result["expected"]
    return result


def merge_shards(contract_path: Path, output_jsonl: Path, output_manifest: Path) -> None:
    contract, seed = verify_contract(contract_path, live=False)
    output_root = Path(contract["output_root"])
    rows = []
    for spec in expected_specs(contract, seed):
        row = load_or_none(
            row_path(output_root, spec["shard_id"], spec["candidate_id"]),
            contract=contract,
            spec=spec,
        )
        require(row is not None, f"missing adaptive candidate: {spec['candidate_id']}")
        rows.append(row)
    payload = b"".join(base.canonical_json(row) + b"\n" for row in rows)
    manifest: dict[str, Any] = {
        "schema_name": MERGE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "round_id": ROUND_ID,
        "rollout_contract_sha256": contract["contract_sha256"],
        "parent_rollout_contract_sha256": contract["parent_round"]["rollout_contract_sha256"],
        "source_selection_summary_sha256": contract["parent_round"]["selection_summary_sha256"],
        "families": 189,
        "samples_per_family": SAMPLES_PER_FAMILY,
        "rows": len(rows),
        "profile_counts": dict(
            sorted(Counter(spec["profile"] for spec in expected_specs(contract, seed)).items())
        ),
        "candidate_ids_sha256": base.object_sha256([row["candidate_id"] for row in rows]),
        "candidate_rows_sha256": base.object_sha256([row["candidate_sha256"] for row in rows]),
        "generation_seeds_sha256": base.object_sha256([row["generation_seed"] for row in rows]),
        "output": {
            "path": str(output_jsonl.resolve()),
            "bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    manifest["manifest_sha256"] = base.object_sha256(manifest)
    if output_jsonl.exists() or output_manifest.exists():
        require(output_jsonl.is_file() and output_manifest.is_file(), "existing merge pair is incomplete")
        require(base.file_sha256(output_jsonl) == manifest["output"]["file_sha256"], "merged JSONL drifted")
        existing = base.load_json(output_manifest)
        base.verify_self_hash(existing, "manifest_sha256")
        require(existing == manifest, "existing adaptive merge manifest drifted")
        print(json.dumps(manifest, sort_keys=True))
        return
    base.write_bytes_new(output_jsonl.resolve(), payload)
    try:
        base.write_json_new(output_manifest.resolve(), manifest)
    except BaseException:
        output_jsonl.resolve().unlink(missing_ok=True)
        raise
    print(json.dumps(manifest, sort_keys=True))


def prepare_contract(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    path = output_root / "adaptive-rollout-contract.json"
    if path.exists():
        contract, _ = verify_contract(path, live=True)
        print(json.dumps({"contract": str(path), "sha256": contract["contract_sha256"], "status": "verified"}, sort_keys=True))
        return
    parent_path = args.base_rollout_contract.resolve()
    parent, seed = base.verify_contract(parent_path, live=True)
    seed_path = args.seed_manifest.resolve()
    supplied_seed = base.validate_seed_manifest(seed_path)
    require(seed["manifest_sha256"] == supplied_seed["manifest_sha256"], "seed differs from parent round")
    selections_path = args.base_selections.resolve()
    summary_path = args.base_selection_summary.resolve()
    contract = build_contract(
        parent_contract_path=parent_path,
        parent_contract=parent,
        selections_path=selections_path,
        selections=load_jsonl(selections_path),
        summary_path=summary_path,
        summary=base.load_json(summary_path),
        seed_path=seed_path,
        seed_manifest=supplied_seed,
        output_root=output_root,
        implementation_sha256=base.file_sha256(Path(__file__).resolve()),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base.write_json_new(path, contract)
    print(json.dumps({"contract": str(path), "sha256": contract["contract_sha256"]}, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base-rollout-contract", type=Path, required=True)
    prepare.add_argument("--base-selections", type=Path, required=True)
    prepare.add_argument("--base-selection-summary", type=Path, required=True)
    prepare.add_argument("--seed-manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--shard-id", type=int, choices=(0, 1), required=True)
    status = commands.add_parser("status")
    status.add_argument("--contract", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--contract", type=Path, required=True)
    merge.add_argument("--output-jsonl", type=Path, required=True)
    merge.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_contract(args)
    elif args.command == "run":
        run_shard(args.contract, args.shard_id)
    elif args.command == "status":
        print(json.dumps(rollout_status(args.contract), sort_keys=True))
    elif args.command == "merge":
        merge_shards(args.contract, args.output_jsonl, args.output_manifest)
    else:  # pragma: no cover
        raise Day22AdaptiveRolloutError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
