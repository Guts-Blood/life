#!/usr/bin/env python3
"""Focused unit tests for deterministic Day 19 preparation and reporting."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load("day19_prepare_for_test", "prepare_day19.py")
FINAL = load("day19_finalize_for_test", "finalize_day19.py")


class Day19Tests(unittest.TestCase):
    def test_exact_subset(self) -> None:
        weights = [2, 3, 5, 7]
        chosen = PREP.exact_subset_indices(weights, 10)
        self.assertEqual(sum(weights[index] for index in chosen), 10)

    def test_exact_subset_rejects_unreachable(self) -> None:
        with self.assertRaises(PREP.Day19PreparationError):
            PREP.exact_subset_indices([4, 8], 3)

    def test_frozen_targets(self) -> None:
        expected = {
            "baseline-a": {skill: 16152 for skill in PREP.SKILLS},
            "baseline-b": {"general": 8076, "math": 8076, "code": 32304, "finance": 16152},
            "best-e": {"general": 18844, "math": 18844, "code": 18844, "finance": 8076},
        }
        for slug, targets in expected.items():
            self.assertEqual(PREP.target_by_skill(PREP.RECIPES[slug], 64608), targets)

    def test_message_contract(self) -> None:
        PREP.validate_messages(
            {
                "occurrence_id": "ok",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        )
        with self.assertRaises(PREP.Day19PreparationError):
            PREP.validate_messages(
                {
                    "occurrence_id": "bad",
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "   "},
                    ],
                }
            )

    def test_relative_delta(self) -> None:
        self.assertEqual(FINAL.percent_delta(0.75, 0.5), 0.5)
        self.assertIsNone(FINAL.percent_delta(1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
