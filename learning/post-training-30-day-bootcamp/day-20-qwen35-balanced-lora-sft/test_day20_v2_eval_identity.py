#!/usr/bin/env python3
"""Tests for Day 20 v2 candidate-independent comparison identities."""

from __future__ import annotations

import copy
import unittest

import day20_candidate_factory_v2 as factory
import day20_eval_identity_v2 as identity


def context(scope: str = "probe32") -> dict[str, object]:
    return {
        "contract_version": factory.CONTRACT_VERSION,
        "scope": scope,
        "records": 32 if scope == "probe32" else 112,
        "diagnostic_selection_sha256": "a" * 64 if scope == "probe32" else None,
        "eval_manifest_file_sha256": "b" * 64,
        "experiment_manifest_file_sha256": "c" * 64,
        "experiment_manifest_content_sha256": "d" * 64,
        "scorer_version": "frozen-scorer-v2",
        "scorer_file_sha256": "e" * 64,
        "response_adapter_version": identity.RESPONSE_ADAPTER_VERSION,
        "response_adapter_file_sha256": "f" * 64,
        "code_parser_version": "code-parser-v3",
        "code_parser_file_sha256": "1" * 64,
        "code_composer_version": "code-composer-v3",
        "code_composer_file_sha256": "2" * 64,
        "evaluator_version": "day20-evaluator-v2",
        "evaluator_file_sha256": "3" * 64,
        "rescorer_version": "day20-rescorer-v2",
        "rescorer_file_sha256": "4" * 64,
    }


def run_kwargs(candidate: str) -> dict[str, str]:
    return {
        "candidate": candidate,
        "checkpoint_package_sha256": "5" * 64,
        "model_identity_sha256": "6" * 64,
        "raw_predictions_file_sha256": "7" * 64,
        "raw_summary_file_sha256": "8" * 64,
        "raw_summary_content_sha256": "9" * 64,
    }


def e2b_context(scope: str = "probe32") -> dict[str, object]:
    normalized = context(scope)
    code_ids = [
        f"eval:code:HumanEval/{index}"
        for index in range(8 if scope == "probe32" else 28)
    ]
    runtime = {
        "sandbox_python": "/root/autodl-tmp/envs/day12-e2b/bin/python",
        "e2b_sdk": {"package": "e2b", "version": "2.37.0"},
        "preflight_file_sha256": "a" * 64,
        "preflight_content_sha256": "b" * 64,
    }
    return {
        "normalized_context": normalized,
        "normalized_comparison_key": identity.normalized_comparison_key(normalized),
        "code_sample_ids": code_ids,
        "code_sample_order_sha256": identity.object_sha256(code_ids),
        "sandbox_contract_file_sha256": "c" * 64,
        "sandbox_contract_content_sha256": "d" * 64,
        "sandbox_contract_hash": "e" * 64,
        "e2b_scorer_version": identity.E2B_SCORER_VERSION,
        "e2b_scorer_file_sha256": "f" * 64,
        "humaneval_source_file_sha256": "1" * 64,
        "e2b_runtime_identity": runtime,
        "e2b_runtime_identity_sha256": identity.object_sha256(runtime),
    }


class EvalIdentityV2Tests(unittest.TestCase):
    def test_round_trip_build_and_validation(self) -> None:
        comparison_context = context()
        candidate = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="1e-4",
            checkpoint="t6000",
        )
        comparison = identity.normalized_comparison_key(comparison_context)
        evaluation_run = identity.evaluation_run_hash(
            comparison_context, **run_kwargs(candidate)
        )
        e2b_comparison_context = e2b_context()
        e2b_comparison = identity.e2b_comparison_key(e2b_comparison_context)
        complete = identity.complete_comparison_key(e2b_comparison_context)
        e2b_run = identity.e2b_run_hash(
            e2b_comparison_context,
            candidate=candidate,
            evaluation_run_sha256=evaluation_run,
            checkpoint_package_sha256="5" * 64,
            model_identity_sha256="6" * 64,
            normalized_predictions_file_sha256="a" * 64,
            normalized_predictions_content_sha256="e" * 64,
            eligible_code_sample_ids=e2b_comparison_context["code_sample_ids"],
            raw_e2b_results_file_sha256="b" * 64,
            raw_e2b_results_content_sha256="f" * 64,
        )

        self.assertEqual(
            identity.normalized_comparison_key(
                comparison_context, expected=comparison
            ),
            comparison,
        )
        self.assertEqual(
            identity.evaluation_run_hash(
                comparison_context, expected=evaluation_run, **run_kwargs(candidate)
            ),
            evaluation_run,
        )
        self.assertEqual(
            identity.e2b_comparison_key(
                e2b_comparison_context, expected=e2b_comparison
            ),
            e2b_comparison,
        )
        self.assertEqual(
            identity.complete_comparison_key(
                e2b_comparison_context, expected=complete
            ),
            complete,
        )
        self.assertEqual(
            identity.e2b_run_hash(
                e2b_comparison_context,
                candidate=candidate,
                evaluation_run_sha256=evaluation_run,
                checkpoint_package_sha256="5" * 64,
                model_identity_sha256="6" * 64,
                normalized_predictions_file_sha256="a" * 64,
                normalized_predictions_content_sha256="e" * 64,
                eligible_code_sample_ids=e2b_comparison_context["code_sample_ids"],
                raw_e2b_results_file_sha256="b" * 64,
                raw_e2b_results_content_sha256="f" * 64,
                expected=e2b_run,
            ),
            e2b_run,
        )

    def test_scope_and_record_contracts_fail_closed(self) -> None:
        probe = context()
        probe["records"] = 112
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.normalized_comparison_key(probe)

        probe = context()
        probe["diagnostic_selection_sha256"] = None
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.normalized_comparison_key(probe)

        full = context("full112")
        full["diagnostic_selection_sha256"] = "a" * 64
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.normalized_comparison_key(full)

        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.evaluation_run_hash(
                context("full112"), **run_kwargs("base-probe")
            )

    def test_comparison_keys_are_common_but_run_hashes_are_candidate_specific(self) -> None:
        comparison_context = context()
        base = "base-probe"
        lora = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="6e-5",
            checkpoint="t12000",
        )
        common = identity.normalized_comparison_key(comparison_context)
        self.assertEqual(common, identity.normalized_comparison_key(context()))
        self.assertEqual(
            identity.complete_comparison_key(e2b_context()),
            identity.complete_comparison_key(e2b_context()),
        )
        self.assertNotEqual(
            identity.evaluation_run_hash(comparison_context, **run_kwargs(base)),
            identity.evaluation_run_hash(comparison_context, **run_kwargs(lora)),
        )

    def test_context_and_run_tampering_is_detected(self) -> None:
        comparison_context = context()
        comparison = identity.normalized_comparison_key(comparison_context)
        tampered = copy.deepcopy(comparison_context)
        tampered["scorer_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(identity.EvalIdentityV2Error, "drifted"):
            identity.normalized_comparison_key(tampered, expected=comparison)

        candidate = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="8e-5",
            checkpoint="t18000",
        )
        evaluation_run = identity.evaluation_run_hash(
            comparison_context, **run_kwargs(candidate)
        )
        changed = run_kwargs(candidate)
        changed["raw_predictions_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(identity.EvalIdentityV2Error, "drifted"):
            identity.evaluation_run_hash(
                comparison_context, expected=evaluation_run, **changed
            )

    def test_malformed_inputs_and_unknown_fields_fail_closed(self) -> None:
        malformed = context()
        malformed["eval_manifest_file_sha256"] = "A" * 64
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.normalized_comparison_key(malformed)

        extra = context()
        extra["candidate"] = "base-probe"
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.normalized_comparison_key(extra)

        wrong_adapter = context()
        wrong_adapter["response_adapter_version"] = "qwen35-response-boundary-v2"
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.normalized_comparison_key(wrong_adapter)

        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.evaluation_run_hash(
                context(), **run_kwargs("probe-1e-4")
            )

    def test_e2b_context_binds_order_sandbox_scorer_source_and_runtime(self) -> None:
        value = e2b_context()
        common = identity.e2b_comparison_key(value)
        for field, replacement in (
            ("code_sample_ids", list(reversed(value["code_sample_ids"]))),
            ("sandbox_contract_hash", "0" * 64),
            ("e2b_scorer_file_sha256", "0" * 64),
            ("humaneval_source_file_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(value)
                changed[field] = replacement
                with self.assertRaises(identity.EvalIdentityV2Error):
                    identity.e2b_comparison_key(changed, expected=common)

        changed = copy.deepcopy(value)
        changed["e2b_runtime_identity"]["sandbox_python"] = "/different/python"
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.e2b_comparison_key(changed, expected=common)

    def test_e2b_run_is_candidate_specific_and_eligible_order_is_strict(self) -> None:
        value = e2b_context()
        base_kwargs = {
            "evaluation_run_sha256": "2" * 64,
            "checkpoint_package_sha256": "3" * 64,
            "model_identity_sha256": "4" * 64,
            "normalized_predictions_file_sha256": "5" * 64,
            "normalized_predictions_content_sha256": "6" * 64,
            "eligible_code_sample_ids": value["code_sample_ids"][:6],
            "raw_e2b_results_file_sha256": "7" * 64,
            "raw_e2b_results_content_sha256": "8" * 64,
        }
        probe = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="1e-4",
            checkpoint="t6000",
        )
        other = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="3e-5",
            checkpoint="t6000",
        )
        self.assertNotEqual(
            identity.e2b_run_hash(value, candidate=probe, **base_kwargs),
            identity.e2b_run_hash(value, candidate=other, **base_kwargs),
        )
        invalid = dict(base_kwargs)
        invalid["eligible_code_sample_ids"] = list(
            reversed(base_kwargs["eligible_code_sample_ids"])
        )
        with self.assertRaises(identity.EvalIdentityV2Error):
            identity.e2b_run_hash(value, candidate=probe, **invalid)


if __name__ == "__main__":
    unittest.main()
