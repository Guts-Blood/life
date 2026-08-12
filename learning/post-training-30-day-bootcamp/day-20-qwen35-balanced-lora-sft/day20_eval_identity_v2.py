#!/usr/bin/env python3
"""Fail-closed comparison and run identities for Day 20 v2 evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

import day20_candidate_factory_v2 as candidate_factory


SCHEMA_VERSION = 1
RESPONSE_ADAPTER_VERSION = "qwen35-response-boundary-v3"
E2B_SCORER_VERSION = "day20-code-e2b-v3"
EXPECTED_RECORDS_BY_SCOPE = {"probe32": 32, "full112": 112}
EXPECTED_CODE_RECORDS_BY_SCOPE = {"probe32": 8, "full112": 28}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTEXT_FIELDS = {
    "contract_version",
    "scope",
    "records",
    "diagnostic_selection_sha256",
    "eval_manifest_file_sha256",
    "experiment_manifest_file_sha256",
    "experiment_manifest_content_sha256",
    "scorer_version",
    "scorer_file_sha256",
    "response_adapter_version",
    "response_adapter_file_sha256",
    "code_parser_version",
    "code_parser_file_sha256",
    "code_composer_version",
    "code_composer_file_sha256",
    "evaluator_version",
    "evaluator_file_sha256",
    "rescorer_version",
    "rescorer_file_sha256",
}
_E2B_CONTEXT_FIELDS = {
    "normalized_context",
    "normalized_comparison_key",
    "code_sample_ids",
    "code_sample_order_sha256",
    "sandbox_contract_file_sha256",
    "sandbox_contract_content_sha256",
    "sandbox_contract_hash",
    "e2b_scorer_version",
    "e2b_scorer_file_sha256",
    "humaneval_source_file_sha256",
    "e2b_runtime_identity",
    "e2b_runtime_identity_sha256",
}
_E2B_RUNTIME_FIELDS = {
    "sandbox_python",
    "e2b_sdk",
    "preflight_file_sha256",
    "preflight_content_sha256",
}


class EvalIdentityV2Error(ValueError):
    """An evaluation identity input is incomplete, malformed, or drifted."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def object_sha256(value: Any) -> str:
    """Return the canonical JSON identity used by all v2 evaluation artifacts."""
    return _canonical_sha256(value)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvalIdentityV2Error(f"{label} must be a lowercase bare SHA-256")
    return value


def _require_version(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise EvalIdentityV2Error(f"{label} is not a canonical version")
    return value


def _return_or_validate(value: str, expected: Any, label: str) -> str:
    if expected is not None:
        _require_sha256(expected, label)
        if expected != value:
            raise EvalIdentityV2Error(f"{label} drifted")
    return value


def normalize_comparison_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize candidate-independent evaluation provenance."""
    if not isinstance(context, Mapping):
        raise EvalIdentityV2Error("comparison context must be an object")
    actual_fields = set(context)
    if actual_fields != _CONTEXT_FIELDS:
        missing = sorted(_CONTEXT_FIELDS - actual_fields)
        extra = sorted(actual_fields - _CONTEXT_FIELDS)
        raise EvalIdentityV2Error(
            f"comparison context fields drifted; missing={missing}, extra={extra}"
        )

    if context["contract_version"] != candidate_factory.CONTRACT_VERSION:
        raise EvalIdentityV2Error("candidate-factory contract version drifted")
    scope = context["scope"]
    if scope not in EXPECTED_RECORDS_BY_SCOPE:
        raise EvalIdentityV2Error("scope must be probe32 or full112")
    records = context["records"]
    if isinstance(records, bool) or records != EXPECTED_RECORDS_BY_SCOPE[scope]:
        raise EvalIdentityV2Error("scope and expected record count differ")

    diagnostic_selection = context["diagnostic_selection_sha256"]
    if scope == "probe32":
        diagnostic_selection = _require_sha256(
            diagnostic_selection, "diagnostic_selection_sha256"
        )
    elif diagnostic_selection is not None:
        raise EvalIdentityV2Error(
            "full112 comparison must not bind a diagnostic selection"
        )

    normalized: dict[str, Any] = {
        "contract_version": candidate_factory.CONTRACT_VERSION,
        "scope": scope,
        "records": records,
        "diagnostic_selection_sha256": diagnostic_selection,
    }
    for name in (
        "eval_manifest_file_sha256",
        "experiment_manifest_file_sha256",
        "experiment_manifest_content_sha256",
    ):
        normalized[name] = _require_sha256(context[name], name)
    for component in (
        "scorer",
        "response_adapter",
        "code_parser",
        "code_composer",
        "evaluator",
        "rescorer",
    ):
        version_name = f"{component}_version"
        file_name = f"{component}_file_sha256"
        normalized[version_name] = _require_version(
            context[version_name], version_name
        )
        normalized[file_name] = _require_sha256(context[file_name], file_name)
    if normalized["response_adapter_version"] != RESPONSE_ADAPTER_VERSION:
        raise EvalIdentityV2Error("response adapter must use the frozen v3 contract")
    return normalized


def normalized_comparison_key(
    context: Mapping[str, Any], *, expected: str | None = None
) -> str:
    """Build or validate the candidate-independent normalized comparison key."""
    value = _canonical_sha256(
        {
            "domain": "day20.v2.normalized_comparison",
            "schema_version": SCHEMA_VERSION,
            "context": normalize_comparison_context(context),
        }
    )
    return _return_or_validate(value, expected, "normalized_comparison_key")


def _candidate_identity(candidate: Any, scope: str) -> dict[str, Any]:
    if not isinstance(candidate, str):
        raise EvalIdentityV2Error("candidate must be text")
    try:
        identity = candidate_factory.parse_candidate_id(candidate)
    except candidate_factory.CandidateFactoryV2Error as error:
        raise EvalIdentityV2Error("candidate is outside the v2 grammar") from error
    if identity.scope != scope:
        raise EvalIdentityV2Error("candidate scope differs from evaluation scope")
    return identity.as_dict()


def evaluation_run_hash(
    context: Mapping[str, Any],
    *,
    candidate: str,
    checkpoint_package_sha256: str,
    model_identity_sha256: str,
    raw_predictions_file_sha256: str,
    raw_summary_file_sha256: str,
    raw_summary_content_sha256: str,
    expected: str | None = None,
) -> str:
    """Build or validate one candidate's normalized-evaluation run identity."""
    normalized = normalize_comparison_context(context)
    value = _canonical_sha256(
        {
            "domain": "day20.v2.evaluation_run",
            "schema_version": SCHEMA_VERSION,
            "candidate_identity": _candidate_identity(candidate, normalized["scope"]),
            "normalized_comparison_key": normalized_comparison_key(normalized),
            "checkpoint_package_sha256": _require_sha256(
                checkpoint_package_sha256, "checkpoint_package_sha256"
            ),
            "model_identity_sha256": _require_sha256(
                model_identity_sha256, "model_identity_sha256"
            ),
            "raw_artifacts": {
                "predictions_file_sha256": _require_sha256(
                    raw_predictions_file_sha256, "raw_predictions_file_sha256"
                ),
                "summary_file_sha256": _require_sha256(
                    raw_summary_file_sha256, "raw_summary_file_sha256"
                ),
                "summary_content_sha256": _require_sha256(
                    raw_summary_content_sha256, "raw_summary_content_sha256"
                ),
            },
        }
    )
    return _return_or_validate(value, expected, "evaluation_run_hash")


def e2b_comparison_key(
    context: Mapping[str, Any], *, expected: str | None = None
) -> str:
    """Build or validate the candidate-independent E2B comparison identity."""
    normalized = normalize_e2b_comparison_context(context)
    value = _canonical_sha256(
        {
            "domain": "day20.v2.e2b_comparison",
            "schema_version": SCHEMA_VERSION,
            "context": normalized,
        }
    )
    return _return_or_validate(value, expected, "e2b_comparison_key")


def complete_comparison_key(
    context: Mapping[str, Any], *, expected: str | None = None
) -> str:
    """Build or validate the joined normalized-plus-E2B comparison identity."""
    normalized = normalize_e2b_comparison_context(context)
    value = _canonical_sha256(
        {
            "domain": "day20.v2.complete_comparison",
            "schema_version": SCHEMA_VERSION,
            "normalized_comparison_key": normalized["normalized_comparison_key"],
            "e2b_comparison_key": e2b_comparison_key(normalized),
        }
    )
    return _return_or_validate(value, expected, "complete_comparison_key")


def e2b_run_hash(
    context: Mapping[str, Any],
    *,
    candidate: str,
    evaluation_run_sha256: str,
    checkpoint_package_sha256: str,
    model_identity_sha256: str,
    normalized_predictions_file_sha256: str,
    normalized_predictions_content_sha256: str,
    eligible_code_sample_ids: list[str],
    raw_e2b_results_file_sha256: str,
    raw_e2b_results_content_sha256: str,
    expected: str | None = None,
) -> str:
    """Build or validate one candidate's complete E2B run identity."""
    normalized = normalize_e2b_comparison_context(context)
    normalized_context = normalized["normalized_context"]
    eligible_ids = _eligible_code_order(
        eligible_code_sample_ids, normalized["code_sample_ids"]
    )
    value = _canonical_sha256(
        {
            "domain": "day20.v2.e2b_run",
            "schema_version": SCHEMA_VERSION,
            "candidate_identity": _candidate_identity(
                candidate, normalized_context["scope"]
            ),
            "complete_comparison_key": complete_comparison_key(normalized),
            "evaluation_run_sha256": _require_sha256(
                evaluation_run_sha256, "evaluation_run_sha256"
            ),
            "checkpoint_package_sha256": _require_sha256(
                checkpoint_package_sha256, "checkpoint_package_sha256"
            ),
            "model_identity_sha256": _require_sha256(
                model_identity_sha256, "model_identity_sha256"
            ),
            "raw_artifacts": {
                "normalized_predictions_file_sha256": _require_sha256(
                    normalized_predictions_file_sha256,
                    "normalized_predictions_file_sha256",
                ),
                "normalized_predictions_content_sha256": _require_sha256(
                    normalized_predictions_content_sha256,
                    "normalized_predictions_content_sha256",
                ),
                "eligible_code_sample_ids": eligible_ids,
                "eligible_code_sample_order_sha256": _canonical_sha256(
                    eligible_ids
                ),
                "e2b_results_file_sha256": _require_sha256(
                    raw_e2b_results_file_sha256,
                    "raw_e2b_results_file_sha256",
                ),
                "e2b_results_content_sha256": _require_sha256(
                    raw_e2b_results_content_sha256,
                    "raw_e2b_results_content_sha256",
                ),
            },
        }
    )
    return _return_or_validate(value, expected, "e2b_run_hash")


def _sample_ids(value: Any, *, expected_records: int, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != expected_records:
        raise EvalIdentityV2Error(
            f"{label} must contain exactly {expected_records} sample IDs"
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise EvalIdentityV2Error(f"{label} contains an invalid sample ID")
    if len(set(value)) != len(value):
        raise EvalIdentityV2Error(f"{label} contains duplicate sample IDs")
    return list(value)


def _eligible_code_order(value: Any, code_order: list[str]) -> list[str]:
    if not isinstance(value, list):
        raise EvalIdentityV2Error("eligible_code_sample_ids must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise EvalIdentityV2Error("eligible Code sample ID is invalid")
    if len(set(value)) != len(value):
        raise EvalIdentityV2Error("eligible Code sample IDs contain duplicates")
    selected = set(value)
    if not selected <= set(code_order) or value != [
        sample_id for sample_id in code_order if sample_id in selected
    ]:
        raise EvalIdentityV2Error(
            "eligible Code sample IDs are not an ordered subset of Code scope"
        )
    return list(value)


def normalize_e2b_comparison_context(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate candidate-independent normalized, sandbox, and runtime bindings."""
    if not isinstance(context, Mapping):
        raise EvalIdentityV2Error("E2B comparison context must be an object")
    actual_fields = set(context)
    if actual_fields != _E2B_CONTEXT_FIELDS:
        missing = sorted(_E2B_CONTEXT_FIELDS - actual_fields)
        extra = sorted(actual_fields - _E2B_CONTEXT_FIELDS)
        raise EvalIdentityV2Error(
            f"E2B comparison context fields drifted; missing={missing}, extra={extra}"
        )
    normalized_context = normalize_comparison_context(context["normalized_context"])
    comparison = normalized_comparison_key(normalized_context)
    if context["normalized_comparison_key"] != comparison:
        raise EvalIdentityV2Error("normalized_comparison_key drifted in E2B context")
    scope = normalized_context["scope"]
    code_ids = _sample_ids(
        context["code_sample_ids"],
        expected_records=EXPECTED_CODE_RECORDS_BY_SCOPE[scope],
        label="code_sample_ids",
    )
    code_order_sha = _canonical_sha256(code_ids)
    if context["code_sample_order_sha256"] != code_order_sha:
        raise EvalIdentityV2Error("code_sample_order_sha256 drifted")

    runtime = context["e2b_runtime_identity"]
    if not isinstance(runtime, Mapping) or set(runtime) != _E2B_RUNTIME_FIELDS:
        raise EvalIdentityV2Error("E2B runtime identity fields drifted")
    sandbox_python = runtime["sandbox_python"]
    sdk = runtime["e2b_sdk"]
    if not isinstance(sandbox_python, str) or not sandbox_python.startswith("/"):
        raise EvalIdentityV2Error("E2B sandbox Python identity is invalid")
    if (
        not isinstance(sdk, Mapping)
        or set(sdk) != {"package", "version"}
        or sdk.get("package") != "e2b"
    ):
        raise EvalIdentityV2Error("E2B SDK runtime identity is invalid")
    runtime_normalized = {
        "sandbox_python": sandbox_python,
        "e2b_sdk": {
            "package": "e2b",
            "version": _require_version(sdk.get("version"), "e2b_sdk.version"),
        },
        "preflight_file_sha256": _require_sha256(
            runtime["preflight_file_sha256"], "preflight_file_sha256"
        ),
        "preflight_content_sha256": _require_sha256(
            runtime["preflight_content_sha256"], "preflight_content_sha256"
        ),
    }
    runtime_sha = _canonical_sha256(runtime_normalized)
    if context["e2b_runtime_identity_sha256"] != runtime_sha:
        raise EvalIdentityV2Error("e2b_runtime_identity_sha256 drifted")
    scorer_version = _require_version(
        context["e2b_scorer_version"], "e2b_scorer_version"
    )
    if scorer_version != E2B_SCORER_VERSION:
        raise EvalIdentityV2Error("E2B scorer version drifted")
    normalized = {
        "normalized_context": normalized_context,
        "normalized_comparison_key": comparison,
        "code_sample_ids": code_ids,
        "code_sample_order_sha256": code_order_sha,
        "sandbox_contract_file_sha256": _require_sha256(
            context["sandbox_contract_file_sha256"],
            "sandbox_contract_file_sha256",
        ),
        "sandbox_contract_content_sha256": _require_sha256(
            context["sandbox_contract_content_sha256"],
            "sandbox_contract_content_sha256",
        ),
        "sandbox_contract_hash": _require_sha256(
            context["sandbox_contract_hash"], "sandbox_contract_hash"
        ),
        "e2b_scorer_version": scorer_version,
        "e2b_scorer_file_sha256": _require_sha256(
            context["e2b_scorer_file_sha256"], "e2b_scorer_file_sha256"
        ),
        "humaneval_source_file_sha256": _require_sha256(
            context["humaneval_source_file_sha256"],
            "humaneval_source_file_sha256",
        ),
        "e2b_runtime_identity": runtime_normalized,
        "e2b_runtime_identity_sha256": runtime_sha,
    }
    return normalized


__all__ = [
    "EXPECTED_RECORDS_BY_SCOPE",
    "EXPECTED_CODE_RECORDS_BY_SCOPE",
    "E2B_SCORER_VERSION",
    "EvalIdentityV2Error",
    "RESPONSE_ADAPTER_VERSION",
    "SCHEMA_VERSION",
    "complete_comparison_key",
    "e2b_comparison_key",
    "e2b_run_hash",
    "evaluation_run_hash",
    "normalize_comparison_context",
    "normalize_e2b_comparison_context",
    "normalized_comparison_key",
    "object_sha256",
]
