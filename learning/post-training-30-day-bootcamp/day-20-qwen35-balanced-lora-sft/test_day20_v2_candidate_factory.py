#!/usr/bin/env python3

from __future__ import annotations

import unittest

import day20_candidate_factory_v2 as factory


BASE = {
    "general": 0,
    "math": 7,
    "finance": 3,
    "code": 7,
    "total": 17,
    "code_sandbox_execution_eligible": 8,
    "infrastructure_failures": 0,
}


class CandidateFactoryV2Tests(unittest.TestCase):
    def test_candidate_ids_bind_seed_lr_and_milestone(self) -> None:
        self.assertEqual(
            factory.candidate_id(
                run_kind="probe",
                seed=20260809,
                learning_rate="0.0001",
                checkpoint="t6000",
            ),
            "probe-s20260809-lr1e-4-t6000",
        )
        self.assertEqual(
            factory.candidate_id(
                run_kind="main",
                seed=20260810,
                learning_rate="6e-5",
                checkpoint="mid",
            ),
            "main-s20260810-lr6e-5-mid",
        )
        with self.assertRaises(factory.CandidateFactoryV2Error):
            factory.candidate_id(
                run_kind="probe", seed=3, learning_rate="1e-4", checkpoint="t6000"
            )
        with self.assertRaisesRegex(factory.CandidateFactoryV2Error, "primary seed"):
            factory.candidate_id(
                run_kind="probe",
                seed=factory.CONFIRMATION_SEED,
                learning_rate="1e-4",
                checkpoint="t24000",
            )

    def test_candidate_identity_round_trip_and_explicit_scope(self) -> None:
        probe_id = factory.candidate_id(
            run_kind="probe",
            seed=factory.PRIMARY_SEED,
            learning_rate="6e-5",
            checkpoint="t18000",
        )
        probe = factory.parse_candidate_id(probe_id)
        self.assertEqual(probe.scope, "probe32")
        self.assertEqual(probe.target_tokens, 18_000)
        self.assertEqual(probe.learning_rate, "6e-5")

        main_id = factory.candidate_id(
            run_kind="main",
            seed=factory.CONFIRMATION_SEED,
            learning_rate="8e-5",
            checkpoint="final",
        )
        main = factory.parse_candidate_id(main_id)
        self.assertEqual(main.scope, "full112")
        self.assertEqual(main.target_tokens, 320_000)
        self.assertEqual(factory.base_candidate_id("probe32"), "base-probe")
        self.assertEqual(factory.base_candidate_id("full112"), "base-full")

        with self.assertRaises(factory.CandidateFactoryV2Error):
            factory.parse_candidate_id(
                f"probe-s{factory.CONFIRMATION_SEED}-lr1e-4-t24000"
            )

    def test_probe_classification_is_fail_closed(self) -> None:
        passing = dict(BASE, total=19, general=2)
        self.assertEqual(
            factory.classify_probe(
                base_metrics=BASE, candidate_metrics=passing
            )["classification"],
            "strict_pass",
        )
        near = {
            "general": 5,
            "math": 5,
            "finance": 4,
            "code": 5,
            "total": 19,
            "code_sandbox_execution_eligible": 6,
            "infrastructure_failures": 0,
        }
        self.assertEqual(
            factory.classify_probe(base_metrics=BASE, candidate_metrics=near)[
                "classification"
            ],
            "near_miss",
        )
        gross = dict(near, code_sandbox_execution_eligible=0)
        self.assertEqual(
            factory.classify_probe(base_metrics=BASE, candidate_metrics=gross)[
                "classification"
            ],
            "gross_fail",
        )

    def test_funnel_only_brackets_near_misses(self) -> None:
        near = {
            "learning_rate": "1e-4",
            "checkpoint": "t24000",
            "metrics": {
                "general": 5,
                "math": 5,
                "finance": 4,
                "code": 5,
                "total": 19,
                "code_sandbox_execution_eligible": 6,
                "infrastructure_failures": 0,
            },
        }
        self.assertEqual(
            factory.select_probe_action(
                base_metrics=BASE, candidates=[near], bracket_completed=False
            )["action"],
            "run_bracket",
        )
        self.assertEqual(
            factory.select_probe_action(
                base_metrics=BASE, candidates=[near], bracket_completed=True
            )["action"],
            "stop_no_passing_probe",
        )

    def test_main_thresholds_remain_frozen(self) -> None:
        metrics = {
            "general_correct": 10,
            "math_correct": 20,
            "finance_correct": 15,
            "code_correct": 20,
            "total_correct": 65,
            "format_compliant": 90,
            "code_sandbox_execution_eligible": 26,
            "infrastructure_failures": 0,
        }
        self.assertTrue(factory.classify_main(metrics)["promotion_eligible"])
        self.assertFalse(
            factory.classify_main(dict(metrics, code_correct=13, total_correct=58))[
                "promotion_eligible"
            ]
        )


if __name__ == "__main__":
    unittest.main()
