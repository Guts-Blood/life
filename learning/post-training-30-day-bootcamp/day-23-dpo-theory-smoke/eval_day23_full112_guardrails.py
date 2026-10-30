#!/usr/bin/env python3
"""Generate and deterministically score the Day 23 finalist on frozen full112.

The executable has one scientific mode: load the fresh-S1 qualification
checkpoint-20 in a new process, generate the frozen Day 10 full112 suite on
physical GPU 0 with greedy decoding, and rescore through the frozen Day 20 v3
response/code boundary.  Code is only prepared for the separate E2B stage; it
is never executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
BOOTCAMP_ROOT = HERE.parent
DAY20_DIR = BOOTCAMP_ROOT / "day-20-qwen35-balanced-lora-sft"
if str(DAY20_DIR) not in sys.path:
    sys.path.insert(0, str(DAY20_DIR))

import day20_candidate_factory_v2 as candidate_factory  # noqa: E402
import day20_contract_v2 as code_contract  # noqa: E402
import day20_eval_identity_v2 as eval_identity  # noqa: E402
import eval_day23_qualification_candidate as qualification_eval  # noqa: E402
import evaluate_day20_v2 as raw_evaluator  # noqa: E402
import qwen35_response_adapter_v3 as response_adapter  # noqa: E402
import rescore_day20_qwen35_v3 as normalized_rescorer  # noqa: E402
import run_day23_qwen35_gpu_stage as gpu_stage  # noqa: E402


SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "day23.qualification_v0001_full112_guardrail_receipt"
FAILURE_SCHEMA = "day23.qualification_v0001_full112_guardrail_failure"
RAW_SUMMARY_SCHEMA = "day23.qualification_v0001_full112_raw_summary"
NORMALIZED_SUMMARY_DOMAIN = "day23.qualification_v0001_full112_normalized_summary"
ADAPTER_VERSION = "day23-qualification-full112-guardrails-v1"
CANDIDATE_ID = "qual-v0001-checkpoint-20"
SCOPE = "full112"
RECORDS = 112
RECORDS_PER_SKILL = 28
CHECKPOINT_STEP = 20
RUN_ID = "full154_refit_lr_4p3e_6_step_20"

SUCCESS_NAME = "full112-generation-and-score.json"
FAILURE_NAME = "full112-generation-and-score-failure.json"
RAW_PREDICTIONS_NAME = "full112-raw.predictions.jsonl"
RAW_SUMMARY_NAME = "full112-raw.summary.json"
NORMALIZED_PREDICTIONS_NAME = "full112-normalized.predictions.jsonl"
NORMALIZED_SUMMARY_NAME = "full112-normalized.summary.json"

HARD_GATES = {
    "total_correct": 65,
    "general_correct": 5,
    "math_correct": 17,
    "finance_correct": 10,
    "code_correct": 14,
    "format_compliant": 90,
    "code_sandbox_execution_eligible": 26,
    "infrastructure_failures": 0,
}

FROZEN_SHA256 = {
    "eval_manifest": "be4b79f93d8c850360f3f2d38c223fcd347517796daaa9a1a4d3bc1a5eab7491",
    "scorers": "18be9a87b857ba80baf8718bf3c75276f58c96f29359b07400e56e09ecb11d54",
    "candidate_factory": "92b9b575b5ac80d6c896924de4e930badd26be98633dcea133347263dda144de",
    "code_contract": "ad67937b1deeb9cf5cbc044aaa6b9d6315df7486f8ae6a48b9252becb6cad5d7",
    "eval_identity": "a060803ce0fd982ca7585fca544ee15909f5ec5e3c189bdda618810fbcc8ce89",
    "response_adapter": "a00795df99067032c7b2a659c8dba33bf0572a88f9b6451b3a0e97db3d12dde8",
    "raw_evaluator": "4da1e7bed95a628e6914711842001b491b25e821ba6711c7bafac8c08e837963",
    "normalized_rescorer": "0f8cffad3265fa85af920d5e0a8c2e88554b8c49c91d85f9a3885e899c7b06f2",
}


class Full112GuardrailError(ValueError):
    """A qualification, frozen-eval, generation, or evidence invariant failed."""


@dataclass(frozen=True)
class Day23CandidateIdentity:
    id: str = CANDIDATE_ID
    scope: str = SCOPE
    model_role: str = "lora"
    run_kind: str = "day23_dpo_qualification"
    seed: int = 20260820
    learning_rate: str = "4.3e-6"
    checkpoint_label: str = "step20"
    target_steps: int = CHECKPOINT_STEP

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CANDIDATE = Day23CandidateIdentity()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Full112GuardrailError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Full112GuardrailError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symbolic")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise Full112GuardrailError(f"cannot read {label}: {path}") from error
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, 1):
        _require(bool(line), f"blank line in {label}:{ordinal}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Full112GuardrailError(f"invalid JSON in {label}:{ordinal}") from error
        _require(isinstance(row, dict), f"non-object row in {label}:{ordinal}")
        rows.append(row)
    return rows


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> str:
    expected = value.get(field)
    _require(
        isinstance(expected, str)
        and len(expected) == 64
        and gpu_stage.object_sha256(value, field) == expected,
        f"{label} self hash drifted",
    )
    return expected


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    _require(path.parent.is_dir() and not path.parent.is_symlink(), f"output parent is invalid: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
    except FileExistsError as error:
        raise Full112GuardrailError(f"refusing to overwrite artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    _require(published, f"artifact was not published: {path}")


def _identity(path: Path, *, content_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value: dict[str, Any] = {
        "path": str(resolved),
        "file_sha256": gpu_stage.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if content_sha256 is not None:
        value["content_sha256"] = content_sha256
    return value


def _frozen_components(bootcamp: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "eval_manifest": bootcamp / "artifacts/eval/day10-frozen-eval-manifest.json",
        "scorers": bootcamp / "day-10-frozen-eval-baseline/day10_scorers.py",
        "candidate_factory": DAY20_DIR / "day20_candidate_factory_v2.py",
        "code_contract": DAY20_DIR / "day20_contract_v2.py",
        "eval_identity": DAY20_DIR / "day20_eval_identity_v2.py",
        "response_adapter": DAY20_DIR / "qwen35_response_adapter_v3.py",
        "raw_evaluator": DAY20_DIR / "evaluate_day20_v2.py",
        "normalized_rescorer": DAY20_DIR / "rescore_day20_qwen35_v3.py",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        _require(resolved.is_relative_to(bootcamp), f"frozen component escaped bootcamp: {name}")
        actual = gpu_stage.file_sha256(resolved)
        _require(actual == FROZEN_SHA256[name], f"frozen component drifted: {name}")
        result[name] = _identity(resolved)
    result["scorers"]["version"] = getattr(
        _load_frozen_suite(result)[1], "SCORER_REGISTRY_VERSION", None
    )
    result["candidate_factory"]["version"] = candidate_factory.CONTRACT_VERSION
    result["code_contract"]["version"] = code_contract.V2_CONTRACT_VERSION
    result["eval_identity"]["version"] = f"day20-eval-identity-v{eval_identity.SCHEMA_VERSION}"
    result["response_adapter"]["version"] = response_adapter.ADAPTER_VERSION
    result["raw_evaluator"]["version"] = raw_evaluator.EVALUATOR_VERSION
    result["normalized_rescorer"]["version"] = normalized_rescorer.RESCORER_VERSION
    return result


def _load_frozen_suite(
    components: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    manifest_path = Path(str(components["eval_manifest"]["path"]))
    scorers_path = Path(str(components["scorers"]["path"]))
    try:
        manifest, scorer = raw_evaluator.load_frozen_eval_manifest(
            manifest_path, scorer_path=scorers_path
        )
        selected = raw_evaluator.select_evaluation_records(
            frozen_manifest=manifest, experiment_manifest={}, eval_scope=SCOPE
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise Full112GuardrailError(f"frozen full112 suite failed validation: {error}") from error
    _require(
        len(selected) == RECORDS
        and Counter(row.get("slice") for row in selected)
        == Counter({skill: RECORDS_PER_SKILL for skill in raw_evaluator.SKILLS}),
        "frozen full112 selection drifted",
    )
    return manifest, scorer, selected


def _context_after_dev(campaign_path: Path) -> dict[str, Any]:
    path = campaign_path.expanduser().resolve(strict=True)
    preliminary = qualification_eval._load_json(path, "qualification campaign")
    _require(
        preliminary.get("schema_name") == qualification_eval.CAMPAIGN_SCHEMA
        and preliminary.get("goal_id") == qualification_eval.GOAL_ID
        and preliminary.get("version_id") == qualification_eval.VERSION_ID,
        "qualification campaign identity drifted",
    )
    qualification_eval._verify_self(preliminary, "campaign_sha256", "qualification campaign")
    runner, runner_identity = qualification_eval._load_bound_runner(preliminary)
    try:
        validated = runner.validate_campaign_document(path, require_dev_unopened=False)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Full112GuardrailError(f"campaign-bound runner rejected campaign: {error}") from error
    campaign = _mapping(validated.get("campaign", validated), "runner-validated campaign")
    _require(dict(campaign) == preliminary, "runner validated different campaign bytes")
    run_root = qualification_eval._absolute(
        campaign.get("remote_run_root"), "campaign.remote_run_root"
    )
    _require(path == run_root / "binding/gpu-campaign.json", "campaign path is not canonical")
    model_value = validated.get("model_path")
    model_path = (
        qualification_eval._absolute(model_value, "validated parent model")
        if model_value is not None
        else gpu_stage._verify_remote_parent_payload(
            _mapping(campaign.get("remote_parent"), "campaign.remote_parent")
        )
    )
    return {
        "campaign": campaign,
        "campaign_path": path,
        "campaign_file_sha256": gpu_stage.file_sha256(path),
        "campaign_sha256": preliminary["campaign_sha256"],
        "run_root": run_root,
        "model_path": model_path,
        "runner": runner,
        "runner_identity": runner_identity,
    }


def _validate_dev_evaluation(
    context: Mapping[str, Any], checkpoint: Mapping[str, Any], dev_path: Path
) -> dict[str, Any]:
    path = dev_path.expanduser().resolve(strict=True)
    expected = context["run_root"] / "evidence/dev/finalist-dev.json"
    _require(path == expected, "dev evaluation path is not campaign-canonical")
    value = _load_json(path, "qualification dev evaluation")
    evaluation_sha = _verify_self(value, "evaluation_sha256", "qualification dev evaluation")
    campaign = _mapping(value.get("campaign"), "dev evaluation campaign")
    checkpoint_entry = _mapping(value.get("checkpoint"), "dev evaluation checkpoint")
    eligibility = _mapping(value.get("eligibility"), "dev evaluation eligibility")
    claim = _mapping(value.get("access_claim"), "dev access claim")
    producer = _mapping(value.get("producer"), "dev evaluation producer")
    bound_producer = _mapping(
        _mapping(context["campaign"].get("producers"), "campaign producers").get(
            "evaluator"
        ),
        "campaign-bound dev evaluator",
    )
    producer_path = Path(str(producer.get("path", ""))).resolve(strict=True)
    bound_producer_path = Path(str(bound_producer.get("path", ""))).resolve(strict=True)
    results = value.get("pair_results")
    _require(isinstance(results, list), "dev pair results are missing")
    for result in results:
        _require(isinstance(result, Mapping), "dev pair result is not an object")
        _verify_self(result, "pair_evaluation_sha256", "dev pair result")
    aggregate = qualification_eval.preference_eval.aggregate_results(
        results, split="dev", checkpoint_step=CHECKPOINT_STEP
    )
    recomputed_gate = qualification_eval._gate(aggregate)
    _require(
        value.get("schema_name") == qualification_eval.EVALUATION_SCHEMA
        and value.get("status") == "pass"
        and value.get("run_id") == RUN_ID
        and campaign.get("campaign_sha256") == context["campaign_sha256"]
        and campaign.get("file_sha256") == context["campaign_file_sha256"]
        and checkpoint_entry.get("global_step") == CHECKPOINT_STEP
        and checkpoint_entry.get("training_receipt_sha256") == checkpoint["receipt_sha256"]
        and checkpoint_entry.get("files_sha256") == checkpoint["checkpoint"]["files_sha256"]
        and value.get("aggregate") == aggregate
        and eligibility == recomputed_gate
        and recomputed_gate.get("passed") is True
        and producer_path == bound_producer_path
        and producer.get("file_sha256") == bound_producer.get("file_sha256")
        and gpu_stage.file_sha256(producer_path) == producer.get("file_sha256"),
        "qualification dev PASS lineage drifted",
    )
    claim_path = Path(str(claim.get("path", ""))).resolve(strict=True)
    claim_value = _load_json(claim_path, "dev access claim")
    _require(
        gpu_stage.file_sha256(claim_path) == claim.get("file_sha256")
        and _verify_self(claim_value, "claim_sha256", "dev access claim")
        == claim.get("claim_sha256")
        and claim_value.get("campaign", {}).get("campaign_sha256")
        == context["campaign_sha256"]
        and claim_value.get("training_receipt_sha256") == checkpoint["receipt_sha256"],
        "dev access claim drifted",
    )
    failure = context["run_root"] / "evidence/dev/dev-failure-after-claim.json"
    _require(not failure.exists() and not failure.is_symlink(), "dev failure receipt exists")
    return {
        "value": value,
        "identity": _identity(path, content_sha256=evaluation_sha),
    }


def _guardrail_paths(run_root: Path) -> dict[str, Path]:
    directory = run_root / "evidence/guardrails"
    directory.mkdir(parents=True, exist_ok=True)
    _require(directory.is_dir() and not directory.is_symlink(), "guardrail directory is invalid")
    return {
        "directory": directory,
        "receipt": directory / SUCCESS_NAME,
        "failure": directory / FAILURE_NAME,
        "raw_predictions": directory / RAW_PREDICTIONS_NAME,
        "raw_summary": directory / RAW_SUMMARY_NAME,
        "normalized_predictions": directory / NORMALIZED_PREDICTIONS_NAME,
        "normalized_summary": directory / NORMALIZED_SUMMARY_NAME,
    }


def _pin_physical_gpu_zero() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(visible in (None, "0"), "full112 generation must expose only physical GPU 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _process_tuple(value: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return value.get("boot_id"), value.get("pid"), value.get("proc_start_ticks")


def _model_provenance(
    context: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    model_identity = {
        "base": dict(_mapping(context["campaign"].get("remote_parent"), "remote parent")),
        "adapter": checkpoint["checkpoint"],
        "expected_model_class": raw_evaluator.EXPECTED_MODEL_CLASS,
    }
    checkpoint_package = checkpoint["checkpoint"]
    return {
        "model_identity": model_identity,
        "model_identity_sha256": _object_sha256(model_identity),
        "checkpoint_package": checkpoint_package,
        "checkpoint_package_sha256": _object_sha256(checkpoint_package),
        "compact_model_identity": {
            "model_identity_sha256": _object_sha256(model_identity),
            "base_snapshot_sha256": _object_sha256(model_identity["base"]),
            "adapter_snapshot_sha256": checkpoint_package["files_sha256"],
        },
        "compact_checkpoint_package": {
            "checkpoint_package_sha256": _object_sha256(checkpoint_package)
        },
    }


def _raw_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_skill: dict[str, dict[str, Any]] = {}
    for skill in raw_evaluator.SKILLS:
        selected = [row for row in rows if row.get("skill") == skill]
        scores = [row.get("scorer_result", {}).get("score") for row in selected]
        correct = None if skill == "code" else sum(float(score) for score in scores)
        by_skill[skill] = {
            "records": len(selected),
            "correct": correct,
            "format_compliant": sum(row.get("format_compliant") is True for row in selected),
            "code_sandbox_execution_eligible": (
                sum(
                    row.get("code_contract", {}).get("execution_eligible") is True
                    for row in selected
                )
                if skill == "code"
                else None
            ),
        }
    return {
        "records": len(rows),
        "non_code_correct": sum(by_skill[skill]["correct"] for skill in ("general", "math", "finance")),
        "format_compliant": sum(row.get("format_compliant") is True for row in rows),
        "code_sandbox_execution_eligible": by_skill["code"]["code_sandbox_execution_eligible"],
        "infrastructure_failures": 0,
        "by_skill": by_skill,
    }


def _comparison_context(
    components: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any]
) -> dict[str, Any]:
    value = {
        "contract_version": candidate_factory.CONTRACT_VERSION,
        "scope": SCOPE,
        "records": RECORDS,
        "diagnostic_selection_sha256": None,
        "eval_manifest_file_sha256": components["eval_manifest"]["file_sha256"],
        "experiment_manifest_file_sha256": context["campaign_file_sha256"],
        "experiment_manifest_content_sha256": context["campaign_sha256"],
        "scorer_version": components["scorers"]["version"],
        "scorer_file_sha256": components["scorers"]["file_sha256"],
        "response_adapter_version": components["response_adapter"]["version"],
        "response_adapter_file_sha256": components["response_adapter"]["file_sha256"],
        "code_parser_version": normalized_rescorer.CODE_PARSER_VERSION,
        "code_parser_file_sha256": components["code_contract"]["file_sha256"],
        "code_composer_version": normalized_rescorer.CODE_COMPOSER_VERSION,
        "code_composer_file_sha256": components["code_contract"]["file_sha256"],
        "evaluator_version": ADAPTER_VERSION,
        "evaluator_file_sha256": gpu_stage.file_sha256(Path(__file__).resolve()),
        "rescorer_version": components["normalized_rescorer"]["version"],
        "rescorer_file_sha256": components["normalized_rescorer"]["file_sha256"],
    }
    try:
        return eval_identity.normalize_comparison_context(value)
    except ValueError as error:
        raise Full112GuardrailError(str(error)) from error


def _aggregate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    by_skill = _mapping(metrics.get("by_skill"), "normalized metrics.by_skill")
    general = float(_mapping(by_skill.get("general"), "general metrics").get("correct"))
    math_score = float(_mapping(by_skill.get("math"), "math metrics").get("correct"))
    finance = float(_mapping(by_skill.get("finance"), "finance metrics").get("correct"))
    non_code = float(metrics.get("non_code_correct"))
    _require(
        all(math.isfinite(value) and value.is_integer() for value in (general, math_score, finance, non_code)),
        "deterministic scorer totals must be finite integers",
    )
    return {
        "records": RECORDS,
        "records_by_skill": {skill: RECORDS_PER_SKILL for skill in raw_evaluator.SKILLS},
        "general_correct": int(general),
        "math_correct": int(math_score),
        "finance_correct": int(finance),
        "non_code_correct": int(non_code),
        "code_correct": None,
        "total_correct": None,
        "format_compliant": int(metrics.get("format_compliant")),
        "code_sandbox_execution_eligible": int(
            metrics.get("code_sandbox_execution_eligible")
        ),
        "infrastructure_failures": int(metrics.get("infrastructure_failures")),
    }


def _gate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        key: aggregate.get(key)
        for key in (
            "total_correct",
            "general_correct",
            "math_correct",
            "finance_correct",
            "code_correct",
            "format_compliant",
            "code_sandbox_execution_eligible",
            "infrastructure_failures",
        )
    }
    passed = {
        "general_correct": observed["general_correct"] >= HARD_GATES["general_correct"],
        "math_correct": observed["math_correct"] >= HARD_GATES["math_correct"],
        "finance_correct": observed["finance_correct"] >= HARD_GATES["finance_correct"],
        "format_compliant": observed["format_compliant"] >= HARD_GATES["format_compliant"],
        "code_sandbox_execution_eligible": observed["code_sandbox_execution_eligible"]
        >= HARD_GATES["code_sandbox_execution_eligible"],
        "infrastructure_failures": observed["infrastructure_failures"]
        <= HARD_GATES["infrastructure_failures"],
    }
    pending = {
        "code_correct": {"operator": ">=", "threshold": HARD_GATES["code_correct"]},
        "total_correct": {"operator": ">=", "threshold": HARD_GATES["total_correct"]},
    }
    return {
        "passed_pre_sandbox": all(passed.values()),
        "thresholds": {
            "total_correct": {"operator": ">=", "threshold": HARD_GATES["total_correct"]},
            "general_correct": {"operator": ">=", "threshold": HARD_GATES["general_correct"]},
            "math_correct": {"operator": ">=", "threshold": HARD_GATES["math_correct"]},
            "finance_correct": {"operator": ">=", "threshold": HARD_GATES["finance_correct"]},
            "code_correct": {"operator": ">=", "threshold": HARD_GATES["code_correct"]},
            "format_compliant": {"operator": ">=", "threshold": HARD_GATES["format_compliant"]},
            "code_sandbox_execution_eligible": {
                "operator": ">=",
                "threshold": HARD_GATES["code_sandbox_execution_eligible"],
            },
            "infrastructure_failures": {
                "operator": "<=",
                "threshold": HARD_GATES["infrastructure_failures"],
            },
        },
        "observed": observed,
        "pre_sandbox_checks": passed,
        "pending_sandbox_checks": pending,
    }


def _raw_summary(
    *,
    rows: list[dict[str, Any]],
    path: Path,
    provenance: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dev: Mapping[str, Any],
    runtime: Mapping[str, Any],
    elapsed: float,
    peak_memory: float,
    process_identity: Mapping[str, Any],
) -> dict[str, Any]:
    predictions_bytes = _jsonl_bytes(rows)
    value: dict[str, Any] = {
        "schema_name": RAW_SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_with_code_sandbox_required",
        "candidate": CANDIDATE_ID,
        "candidate_identity": CANDIDATE.as_dict(),
        "scope": SCOPE,
        "campaign_sha256": context["campaign_sha256"],
        "training_receipt_sha256": checkpoint["receipt_sha256"],
        "dev_evaluation_sha256": dev["identity"]["content_sha256"],
        "model_identity": provenance["model_identity"],
        "model_identity_sha256": provenance["model_identity_sha256"],
        "checkpoint_package": provenance["checkpoint_package"],
        "checkpoint_package_sha256": provenance["checkpoint_package_sha256"],
        "frozen_components": components,
        "protocol": {
            "adapter_version": ADAPTER_VERSION,
            "device": "physical_gpu_0_only",
            "greedy": True,
            "do_sample": False,
            "num_beams": 1,
            "generation_seed": raw_evaluator.GENERATION_SEED,
            "response_capture": "generated_token_ids_decode",
            "adapter_loaded_unmerged": True,
            "candidate_code_executed_on_host": False,
        },
        "runtime": dict(runtime),
        "runtime_metrics": {
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": peak_memory,
        },
        "process_identity": dict(process_identity),
        "fresh_reload": {
            "distinct_from_training_process": True,
            "distinct_from_dev_process": True,
            "checkpoint_manifest_reverified_before_load": True,
            "checkpoint_manifest_reverified_after_generation": True,
        },
        "metrics": _raw_metrics(rows),
        "predictions": {
            "path": str(path.resolve()),
            "records": len(rows),
            "file_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
            "content_sha256": _object_sha256(rows),
            "ordered_sample_ids_sha256": _object_sha256(
                [row["sample_id"] for row in rows]
            ),
        },
    }
    value["summary_sha256"] = _object_sha256(value)
    return value


def _normalize(
    *,
    raw_rows: list[dict[str, Any]],
    raw_summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    scorer: Any,
    components: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
    raw_predictions_path: Path,
    raw_summary_path: Path,
    normalized_predictions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapted = [
        normalized_rescorer._adapt_raw_row(row, response_adapter) for row in raw_rows
    ]
    comparison_context = _comparison_context(components, context)
    comparison_key = eval_identity.normalized_comparison_key(comparison_context)
    run_hash = _object_sha256(
        {
            "domain": "day23.qualification_v0001_full112_evaluation_run",
            "schema_version": SCHEMA_VERSION,
            "candidate_identity": CANDIDATE.as_dict(),
            "normalized_comparison_key": comparison_key,
            "checkpoint_package_sha256": raw_summary["checkpoint_package_sha256"],
            "model_identity_sha256": raw_summary["model_identity_sha256"],
            "raw_predictions_file_sha256": gpu_stage.file_sha256(raw_predictions_path),
            "raw_summary_file_sha256": gpu_stage.file_sha256(raw_summary_path),
            "raw_summary_content_sha256": raw_summary["summary_sha256"],
        }
    )
    rows = [
        normalized_rescorer._build_row(
            raw=raw,
            record=record,
            adapter_evidence=adapter,
            scorer=scorer,
            candidate=CANDIDATE_ID,
            scope=SCOPE,
            comparison_key=comparison_key,
            run_hash=run_hash,
        )
        for raw, record, adapter in zip(raw_rows, records, adapted)
    ]
    metrics = normalized_rescorer._metrics(rows)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": NORMALIZED_SUMMARY_DOMAIN,
        "status": "complete_with_code_sandbox_required",
        "candidate": CANDIDATE_ID,
        "candidate_identity": CANDIDATE.as_dict(),
        "scope": SCOPE,
        "comparison_context": comparison_context,
        "normalized_comparison_key": comparison_key,
        "evaluation_run_sha256": run_hash,
        "checkpoint_package_sha256": raw_summary["checkpoint_package_sha256"],
        "model_identity_sha256": raw_summary["model_identity_sha256"],
        "source_raw": {
            "predictions_path": str(raw_predictions_path.resolve()),
            "predictions_file_sha256": gpu_stage.file_sha256(raw_predictions_path),
            "predictions_content_sha256": _object_sha256(raw_rows),
            "summary_path": str(raw_summary_path.resolve()),
            "summary_file_sha256": gpu_stage.file_sha256(raw_summary_path),
            "summary_content_sha256": raw_summary["summary_sha256"],
        },
        "frozen_components": components,
        "protocol": {
            "adapter_version": ADAPTER_VERSION,
            "rescorer_version": normalized_rescorer.RESCORER_VERSION,
            "candidate_code_executed_on_host": False,
            "code_candidate_mode": "completion",
            "indentation_repair_applied": False,
            "frozen_test_consumed": False,
        },
        "metrics": metrics,
        "predictions": {
            "path": str(normalized_predictions_path.resolve()),
            "records": len(rows),
            "file_sha256": hashlib.sha256(_jsonl_bytes(rows)).hexdigest(),
            "content_sha256": _object_sha256(rows),
            "ordered_sample_ids_sha256": _object_sha256(
                [row["sample_id"] for row in rows]
            ),
        },
    }
    summary["summary_sha256"] = _object_sha256(summary)
    return rows, summary


def _verify_raw_pair(
    predictions_path: Path, summary_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], Any]:
    rows = _load_jsonl(predictions_path.resolve(), "full112 raw predictions")
    summary = _load_json(summary_path.resolve(), "full112 raw summary")
    _verify_self(summary, "summary_sha256", "full112 raw summary")
    components = _mapping(summary.get("frozen_components"), "raw frozen components")
    _require(
        dict(components) == _frozen_components(BOOTCAMP_ROOT),
        "raw frozen component closure drifted",
    )
    _, scorer, selected = _load_frozen_suite(components)
    predictions = _mapping(summary.get("predictions"), "raw predictions identity")
    _require(
        summary.get("schema_name") == RAW_SUMMARY_SCHEMA
        and summary.get("status") == "complete_with_code_sandbox_required"
        and summary.get("candidate") == CANDIDATE_ID
        and summary.get("candidate_identity") == CANDIDATE.as_dict()
        and summary.get("scope") == SCOPE
        and summary.get("model_identity_sha256")
        == _object_sha256(summary.get("model_identity"))
        and summary.get("checkpoint_package_sha256")
        == _object_sha256(summary.get("checkpoint_package"))
        and len(rows) == RECORDS
        and predictions.get("records") == RECORDS
        and Path(str(predictions.get("path", ""))).resolve() == predictions_path.resolve()
        and predictions.get("file_sha256") == gpu_stage.file_sha256(predictions_path)
        and predictions.get("content_sha256") == _object_sha256(rows)
        and predictions.get("ordered_sample_ids_sha256")
        == _object_sha256([row.get("sample_id") for row in rows]),
        "raw full112 summary/predictions identity drifted",
    )
    provenance = {
        "compact_model_identity": {
            "model_identity_sha256": summary["model_identity_sha256"],
            "base_snapshot_sha256": _object_sha256(summary["model_identity"]["base"]),
            "adapter_snapshot_sha256": summary["checkpoint_package"]["files_sha256"],
        },
        "compact_checkpoint_package": {
            "checkpoint_package_sha256": summary["checkpoint_package_sha256"]
        },
    }
    for ordinal, (row, record) in enumerate(zip(rows, selected), 1):
        capture = _mapping(row.get("response_capture"), f"raw response capture {ordinal}")
        generation = _mapping(row.get("generation"), f"raw generation {ordinal}")
        rebuilt = raw_evaluator.build_prediction_row(
            ordinal=ordinal,
            candidate_identity=CANDIDATE,
            eval_scope=SCOPE,
            record=record,
            message_content=capture.get("message_content"),
            prompt_token_ids=capture.get("prompt_token_ids"),
            generated_token_ids=capture.get("generated_token_ids"),
            generated_only_text=capture.get("generated_only_text"),
            finish_reason=generation.get("finish_reason"),
            prompt_tokens=generation.get("prompt_tokens"),
            completion_tokens=generation.get("completion_tokens"),
            scorer=scorer,
            compact_model_identity=provenance["compact_model_identity"],
            compact_checkpoint_package=provenance["compact_checkpoint_package"],
        )
        _require(rebuilt == row, f"raw prediction drifted at ordinal {ordinal}")
    _require(summary.get("metrics") == _raw_metrics(rows), "raw metrics drifted")
    return rows, summary, selected, scorer


def verify_normalized_pair(
    predictions_path: Path,
    summary_path: Path,
    *,
    expected_scope: str | None = None,
    expected_candidate: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly recompute the Day23-normalized pair without Day20 ID parsing."""
    predictions_path = predictions_path.resolve()
    summary_path = summary_path.resolve()
    rows = _load_jsonl(predictions_path, "full112 normalized predictions")
    summary = _load_json(summary_path, "full112 normalized summary")
    _verify_self(summary, "summary_sha256", "full112 normalized summary")
    _require(
        summary.get("schema_version") == SCHEMA_VERSION
        and summary.get("domain") == NORMALIZED_SUMMARY_DOMAIN
        and summary.get("status") == "complete_with_code_sandbox_required"
        and summary.get("candidate") == CANDIDATE_ID
        and summary.get("candidate_identity") == CANDIDATE.as_dict()
        and summary.get("scope") == SCOPE
        and (expected_scope is None or expected_scope == SCOPE)
        and (expected_candidate is None or expected_candidate == CANDIDATE_ID),
        "normalized candidate/scope identity drifted",
    )
    source = _mapping(summary.get("source_raw"), "normalized source_raw")
    raw_path = Path(str(source.get("predictions_path", ""))).resolve()
    raw_summary_path = Path(str(source.get("summary_path", ""))).resolve()
    raw_rows, raw_summary, records, scorer = _verify_raw_pair(raw_path, raw_summary_path)
    _require(
        source.get("predictions_file_sha256") == gpu_stage.file_sha256(raw_path)
        and source.get("predictions_content_sha256") == _object_sha256(raw_rows)
        and source.get("summary_file_sha256") == gpu_stage.file_sha256(raw_summary_path)
        and source.get("summary_content_sha256") == raw_summary["summary_sha256"],
        "normalized-to-raw lineage drifted",
    )
    components = _mapping(summary.get("frozen_components"), "normalized frozen components")
    context = {
        "campaign_file_sha256": summary["comparison_context"]["experiment_manifest_file_sha256"],
        "campaign_sha256": summary["comparison_context"]["experiment_manifest_content_sha256"],
    }
    expected_context = _comparison_context(components, context)
    comparison_key = eval_identity.normalized_comparison_key(expected_context)
    run_hash = _object_sha256(
        {
            "domain": "day23.qualification_v0001_full112_evaluation_run",
            "schema_version": SCHEMA_VERSION,
            "candidate_identity": CANDIDATE.as_dict(),
            "normalized_comparison_key": comparison_key,
            "checkpoint_package_sha256": raw_summary["checkpoint_package_sha256"],
            "model_identity_sha256": raw_summary["model_identity_sha256"],
            "raw_predictions_file_sha256": gpu_stage.file_sha256(raw_path),
            "raw_summary_file_sha256": gpu_stage.file_sha256(raw_summary_path),
            "raw_summary_content_sha256": raw_summary["summary_sha256"],
        }
    )
    adapted = [
        normalized_rescorer._adapt_raw_row(row, response_adapter) for row in raw_rows
    ]
    rebuilt = [
        normalized_rescorer._build_row(
            raw=raw,
            record=record,
            adapter_evidence=adapter,
            scorer=scorer,
            candidate=CANDIDATE_ID,
            scope=SCOPE,
            comparison_key=comparison_key,
            run_hash=run_hash,
        )
        for raw, record, adapter in zip(raw_rows, records, adapted)
    ]
    predictions = _mapping(summary.get("predictions"), "normalized predictions identity")
    metrics = normalized_rescorer._metrics(rebuilt)
    checks = {
        "rows": rows == rebuilt,
        "comparison_context": summary.get("comparison_context") == expected_context,
        "comparison_key": summary.get("normalized_comparison_key") == comparison_key,
        "run_hash": summary.get("evaluation_run_sha256") == run_hash,
        "checkpoint": summary.get("checkpoint_package_sha256")
        == raw_summary["checkpoint_package_sha256"],
        "model": summary.get("model_identity_sha256")
        == raw_summary["model_identity_sha256"],
        "metrics": summary.get("metrics") == metrics,
        "prediction_path": predictions.get("path") == str(predictions_path),
        "prediction_records": predictions.get("records") == RECORDS,
        "prediction_file": predictions.get("file_sha256")
        == gpu_stage.file_sha256(predictions_path),
        "prediction_content": predictions.get("content_sha256")
        == _object_sha256(rows),
        "prediction_order": predictions.get("ordered_sample_ids_sha256")
        == _object_sha256([row["sample_id"] for row in rows]),
    }
    _require(
        all(checks.values()),
        f"normalized full112 pair drifted: {[name for name, passed in checks.items() if not passed]}",
    )
    return rows, summary


def _receipt(
    *,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dev: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    raw_predictions_path: Path,
    raw_summary_path: Path,
    normalized_predictions_path: Path,
    normalized_summary_path: Path,
    normalized_summary: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = _aggregate(_mapping(normalized_summary.get("metrics"), "normalized metrics"))
    gate = _gate(aggregate)
    normalized_rows = _load_jsonl(normalized_predictions_path, "normalized predictions")
    code_rows = [row for row in normalized_rows if row.get("slice") == "code"]
    _require(len(code_rows) == RECORDS_PER_SKILL, "normalized Code row count drifted")
    producer = Path(__file__).resolve(strict=True)
    return {
        "schema_name": RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "pass_pending_code_sandbox" if gate["passed_pre_sandbox"] else "gate_fail",
        "candidate": CANDIDATE_ID,
        "scope": SCOPE,
        "campaign": {
            "path": str(context["campaign_path"]),
            "file_sha256": context["campaign_file_sha256"],
            "campaign_sha256": context["campaign_sha256"],
        },
        "training_receipt": _identity(
            checkpoint["receipt_path"], content_sha256=checkpoint["receipt_sha256"]
        ),
        "checkpoint": checkpoint["checkpoint"],
        "dev_evaluation": dev["identity"],
        "producer": _identity(producer),
        "frozen_components": components,
        "raw_predictions": _identity(
            raw_predictions_path,
            content_sha256=_object_sha256(
                _load_jsonl(raw_predictions_path, "raw predictions")
            ),
        ),
        "raw_summary": _identity(
            raw_summary_path,
            content_sha256=_load_json(raw_summary_path, "raw summary")["summary_sha256"],
        ),
        "normalized_predictions": _identity(
            normalized_predictions_path,
            content_sha256=normalized_summary["predictions"]["content_sha256"],
        ),
        "normalized_summary": _identity(
            normalized_summary_path,
            content_sha256=normalized_summary["summary_sha256"],
        ),
        "ordered_sample_ids_sha256": normalized_summary["predictions"]
        ["ordered_sample_ids_sha256"],
        "code_rows": {
            "records": len(code_rows),
            "ordered_sample_ids_sha256": _object_sha256(
                [row["sample_id"] for row in code_rows]
            ),
            "ordered_row_hashes_sha256": _object_sha256(
                [row["row_sha256"] for row in code_rows]
            ),
            "normalized_outputs_sha256": _object_sha256(
                [row["normalized_output_sha256"] for row in code_rows]
            ),
        },
        "aggregate": aggregate,
        "gate": gate,
        "claim_boundary": {
            "fresh_process_checkpoint_reload": True,
            "single_physical_gpu_zero": True,
            "greedy_generation": True,
            "candidate_code_executed_on_host": False,
            "code_and_total_gates_require_e2b": True,
            "final_qualification_claimed": False,
        },
    }


def validate_receipt(path: Path) -> dict[str, Any]:
    """Revalidate the canonical receipt and all raw/normalized evidence."""
    resolved = path.expanduser().resolve(strict=True)
    value = _load_json(resolved, "full112 guardrail receipt")
    receipt_sha = _verify_self(value, "receipt_sha256", "full112 guardrail receipt")
    campaign_entry = _mapping(value.get("campaign"), "receipt campaign")
    training_entry = _mapping(value.get("training_receipt"), "receipt training receipt")
    dev_entry = _mapping(value.get("dev_evaluation"), "receipt dev evaluation")
    context = _context_after_dev(Path(str(campaign_entry.get("path", ""))))
    checkpoint = qualification_eval._checkpoint_from_receipt(
        context, Path(str(training_entry.get("path", "")))
    )
    dev = _validate_dev_evaluation(
        context, checkpoint, Path(str(dev_entry.get("path", "")))
    )
    producer = _mapping(value.get("producer"), "receipt producer")
    producer_path = Path(str(producer.get("path", ""))).resolve(strict=True)
    normalized = _mapping(value.get("normalized_predictions"), "normalized predictions")
    normalized_summary = _mapping(value.get("normalized_summary"), "normalized summary")
    rows, summary = verify_normalized_pair(
        Path(str(normalized.get("path", ""))),
        Path(str(normalized_summary.get("path", ""))),
        expected_scope=SCOPE,
        expected_candidate=CANDIDATE_ID,
    )
    aggregate = _aggregate(_mapping(summary.get("metrics"), "normalized metrics"))
    gate = _gate(aggregate)
    code_rows = [row for row in rows if row.get("slice") == "code"]
    expected_status = "pass_pending_code_sandbox" if gate["passed_pre_sandbox"] else "gate_fail"
    source_raw = _mapping(summary.get("source_raw"), "normalized source_raw")
    raw_predictions_path = Path(str(source_raw.get("predictions_path", ""))).resolve()
    raw_summary_path = Path(str(source_raw.get("summary_path", ""))).resolve()
    raw_rows = _load_jsonl(raw_predictions_path, "receipt raw predictions")
    raw_summary_value = _load_json(raw_summary_path, "receipt raw summary")
    expected_claim_boundary = {
        "fresh_process_checkpoint_reload": True,
        "single_physical_gpu_zero": True,
        "greedy_generation": True,
        "candidate_code_executed_on_host": False,
        "code_and_total_gates_require_e2b": True,
        "final_qualification_claimed": False,
    }
    expected_fields = {
        "schema_name",
        "schema_version",
        "status",
        "candidate",
        "scope",
        "campaign",
        "training_receipt",
        "checkpoint",
        "dev_evaluation",
        "producer",
        "frozen_components",
        "raw_predictions",
        "raw_summary",
        "normalized_predictions",
        "normalized_summary",
        "ordered_sample_ids_sha256",
        "code_rows",
        "aggregate",
        "gate",
        "claim_boundary",
        "receipt_sha256",
    }
    _require(
        set(value) == expected_fields
        and value.get("schema_name") == RECEIPT_SCHEMA
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == expected_status
        and value.get("candidate") == CANDIDATE_ID
        and value.get("scope") == SCOPE
        and resolved == context["run_root"] / "evidence/guardrails" / SUCCESS_NAME
        and campaign_entry
        == {
            "path": str(context["campaign_path"]),
            "file_sha256": context["campaign_file_sha256"],
            "campaign_sha256": context["campaign_sha256"],
        }
        and training_entry
        == _identity(checkpoint["receipt_path"], content_sha256=checkpoint["receipt_sha256"])
        and value.get("checkpoint") == checkpoint["checkpoint"]
        and dev_entry == dev["identity"]
        and producer_path == Path(__file__).resolve(strict=True)
        and producer == _identity(producer_path)
        and value.get("frozen_components") == summary["frozen_components"]
        and value.get("raw_predictions")
        == _identity(raw_predictions_path, content_sha256=_object_sha256(raw_rows))
        and value.get("raw_summary")
        == _identity(
            raw_summary_path, content_sha256=raw_summary_value["summary_sha256"]
        )
        and value.get("aggregate") == aggregate
        and value.get("gate") == gate
        and value.get("claim_boundary") == expected_claim_boundary
        and normalized.get("file_sha256")
        == gpu_stage.file_sha256(Path(str(normalized["path"])))
        and normalized.get("content_sha256") == summary["predictions"]["content_sha256"]
        and normalized_summary.get("file_sha256")
        == gpu_stage.file_sha256(Path(str(normalized_summary["path"])))
        and normalized_summary.get("content_sha256") == summary["summary_sha256"]
        and value.get("ordered_sample_ids_sha256")
        == _object_sha256([row["sample_id"] for row in rows])
        and value.get("code_rows")
        == {
            "records": RECORDS_PER_SKILL,
            "ordered_sample_ids_sha256": _object_sha256(
                [row["sample_id"] for row in code_rows]
            ),
            "ordered_row_hashes_sha256": _object_sha256(
                [row["row_sha256"] for row in code_rows]
            ),
            "normalized_outputs_sha256": _object_sha256(
                [row["normalized_output_sha256"] for row in code_rows]
            ),
        },
        "full112 guardrail receipt drifted",
    )
    result = dict(value)
    result["receipt_sha256"] = receipt_sha
    return result


def run_guardrail(
    campaign_path: Path, training_receipt_path: Path, dev_evaluation_path: Path
) -> dict[str, Any]:
    context = _context_after_dev(campaign_path)
    paths = _guardrail_paths(context["run_root"])
    for name, path in paths.items():
        if name != "directory":
            _require(not path.exists() and not path.is_symlink(), f"guardrail output is not fresh: {path}")
    phase = "qualification_validation"
    attempt = paths["directory"] / f".{RAW_PREDICTIONS_NAME}.{os.getpid()}.attempt"
    _require(not attempt.exists(), "raw generation attempt path already exists")
    checkpoint: dict[str, Any] | None = None
    try:
        checkpoint = qualification_eval._checkpoint_from_receipt(
            context, training_receipt_path
        )
        dev = _validate_dev_evaluation(context, checkpoint, dev_evaluation_path)
        bootcamp = qualification_eval._absolute(
            context["campaign"].get("bootcamp_root"), "campaign.bootcamp_root"
        )
        components = _frozen_components(bootcamp)
        _, scorer, records = _load_frozen_suite(components)
        process_identity = gpu_stage._process_identity()
        _require(
            _process_tuple(process_identity)
            != _process_tuple(_mapping(checkpoint["receipt"].get("process_identity"), "training process"))
            and _process_tuple(process_identity)
            != _process_tuple(_mapping(dev["value"].get("process_identity"), "dev process")),
            "full112 generation must fresh-reload outside training and dev processes",
        )
        _pin_physical_gpu_zero()
        provenance = _model_provenance(context, checkpoint)
        runtime = raw_evaluator.live_runtime_identity()
        phase = "fresh_checkpoint_reload_and_generation"
        rows, live_runtime, elapsed, peak_memory, _ = raw_evaluator.run_inference(
            model_path=context["model_path"],
            adapter_path=checkpoint["checkpoint_path"],
            candidate_identity=CANDIDATE,
            eval_scope=SCOPE,
            selected_records=records,
            scorer=scorer,
            provenance=provenance,
            predictions_attempt=attempt,
            expected_runtime_identity=runtime,
        )
        _require(live_runtime == runtime and len(rows) == RECORDS, "full112 inference coverage drifted")
        _require(
            gpu_stage.checkpoint_manifest(checkpoint["checkpoint_path"], CHECKPOINT_STEP)
            == checkpoint["checkpoint"],
            "checkpoint-20 changed during generation",
        )
        raw_bytes = _jsonl_bytes(rows)
        _require(attempt.read_bytes() == raw_bytes, "raw attempt differs from in-memory rows")
        _atomic_bytes_new(paths["raw_predictions"], raw_bytes)
        attempt.unlink(missing_ok=True)
        raw_summary = _raw_summary(
            rows=rows,
            path=paths["raw_predictions"],
            provenance=provenance,
            components=components,
            context=context,
            checkpoint=checkpoint,
            dev=dev,
            runtime=runtime,
            elapsed=elapsed,
            peak_memory=peak_memory,
            process_identity=process_identity,
        )
        _atomic_bytes_new(paths["raw_summary"], _json_bytes(raw_summary))
        phase = "deterministic_v3_rescore"
        normalized_rows, normalized_summary = _normalize(
            raw_rows=rows,
            raw_summary=raw_summary,
            records=records,
            scorer=scorer,
            components=components,
            context=context,
            raw_predictions_path=paths["raw_predictions"],
            raw_summary_path=paths["raw_summary"],
            normalized_predictions_path=paths["normalized_predictions"],
        )
        _atomic_bytes_new(paths["normalized_predictions"], _jsonl_bytes(normalized_rows))
        _atomic_bytes_new(paths["normalized_summary"], _json_bytes(normalized_summary))
        verify_normalized_pair(
            paths["normalized_predictions"], paths["normalized_summary"],
            expected_scope=SCOPE, expected_candidate=CANDIDATE_ID,
        )
        phase = "receipt_publish"
        receipt = _receipt(
            context=context,
            checkpoint=checkpoint,
            dev=dev,
            components=components,
            raw_predictions_path=paths["raw_predictions"],
            raw_summary_path=paths["raw_summary"],
            normalized_predictions_path=paths["normalized_predictions"],
            normalized_summary_path=paths["normalized_summary"],
            normalized_summary=normalized_summary,
        )
        gpu_stage.write_sealed_json(paths["receipt"], receipt, "receipt_sha256")
        return validate_receipt(paths["receipt"])
    except BaseException as error:
        attempt.unlink(missing_ok=True)
        if not paths["receipt"].exists() and not paths["failure"].exists():
            failure = {
                "schema_name": FAILURE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "phase": phase,
                "campaign_path": str(campaign_path.expanduser().resolve()),
                "training_receipt_path": str(training_receipt_path.expanduser().resolve()),
                "dev_evaluation_path": str(dev_evaluation_path.expanduser().resolve()),
                "checkpoint_step": CHECKPOINT_STEP if checkpoint is not None else None,
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error_message": str(error),
                "traceback": traceback.format_exception(error)[-12:],
                "candidate_code_executed_on_host": False,
            }
            gpu_stage.write_sealed_json(paths["failure"], failure, "failure_sha256")
        raise


def _self_test() -> None:
    aggregate = {
        "records": RECORDS,
        "records_by_skill": {skill: RECORDS_PER_SKILL for skill in raw_evaluator.SKILLS},
        "general_correct": 5,
        "math_correct": 17,
        "finance_correct": 10,
        "non_code_correct": 32,
        "code_correct": None,
        "total_correct": None,
        "format_compliant": 90,
        "code_sandbox_execution_eligible": 26,
        "infrastructure_failures": 0,
    }
    _require(_gate(aggregate)["passed_pre_sandbox"], "threshold boundary self-test failed")
    aggregate["math_correct"] = 16
    _require(not _gate(aggregate)["passed_pre_sandbox"], "threshold rejection self-test failed")
    destinations = {action.dest for action in build_parser()._actions}
    _require(
        destinations == {"help", "campaign", "training_receipt", "dev_evaluation", "self_test"},
        "full112 CLI surface drifted",
    )
    _require("split" not in destinations and "device" not in destinations, "forbidden CLI selector appeared")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--dev-evaluation", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"status": "pass", "scope": "stdlib_self_test"}, sort_keys=True))
        return 0
    _require(args.campaign is not None, "campaign is required")
    _require(args.training_receipt is not None, "training receipt is required")
    _require(args.dev_evaluation is not None, "dev evaluation is required")
    try:
        result = run_guardrail(args.campaign, args.training_receipt, args.dev_evaluation)
    except BaseException as error:
        print(
            json.dumps(
                {"status": "fail", "error": str(error), "traceback": traceback.format_exception(error)[-8:]},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_sha256": result["receipt_sha256"],
                "aggregate": result["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
