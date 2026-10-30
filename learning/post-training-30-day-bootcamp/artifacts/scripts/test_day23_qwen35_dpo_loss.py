#!/usr/bin/env python3
"""Independent scalar and response-mask tests for the Day 23 sigmoid DPO loss."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve()
DAY23_DIR = SCRIPT_PATH.parents[2] / "day-23-dpo-theory-smoke"
if str(DAY23_DIR) not in sys.path:
    sys.path.insert(0, str(DAY23_DIR))

import day23_dpo_math as dpo  # noqa: E402


class Day23DPOLossTests(unittest.TestCase):
    def test_zero_relative_margin_is_log_two(self) -> None:
        for beta in (0.01, 0.1, 1.0, 10.0):
            result = dpo.sigmoid_dpo(-3.0, -5.0, -2.0, -4.0, beta=beta)
            self.assertAlmostEqual(result.loss, math.log(2.0), places=14)
            self.assertAlmostEqual(result.reward_margin, 0.0, places=14)

    def test_better_policy_chosen_margin_reduces_loss(self) -> None:
        baseline = dpo.sigmoid_dpo(-4.0, -5.0, -4.0, -5.0, beta=0.1)
        improved = dpo.sigmoid_dpo(-3.0, -5.0, -4.0, -5.0, beta=0.1)
        self.assertLess(improved.loss, baseline.loss)
        self.assertGreater(improved.reward_margin, baseline.reward_margin)
        self.assertLess(improved.dloss_dpolicy_chosen, 0.0)
        self.assertGreater(improved.dloss_dpolicy_rejected, 0.0)

    def test_swapping_chosen_and_rejected_reverses_direction(self) -> None:
        forward = dpo.sigmoid_dpo(-2.0, -5.0, -3.0, -4.0, beta=0.2)
        swapped = dpo.sigmoid_dpo(-5.0, -2.0, -4.0, -3.0, beta=0.2)
        self.assertAlmostEqual(swapped.logit, -forward.logit)
        self.assertAlmostEqual(swapped.reward_margin, -forward.reward_margin)
        self.assertGreater(swapped.loss, forward.loss)

    def test_reference_margin_changes_the_objective(self) -> None:
        weak_reference = dpo.sigmoid_dpo(-2.0, -4.0, -3.0, -4.0, beta=0.1)
        strong_reference = dpo.sigmoid_dpo(-2.0, -4.0, -1.0, -4.0, beta=0.1)
        self.assertLess(weak_reference.loss, strong_reference.loss)
        self.assertGreater(weak_reference.reward_margin, strong_reference.reward_margin)

    def test_beta_controls_confidence_in_the_correct_direction(self) -> None:
        small = dpo.sigmoid_dpo(-2.0, -5.0, -3.0, -4.0, beta=0.05)
        large = dpo.sigmoid_dpo(-2.0, -5.0, -3.0, -4.0, beta=0.5)
        self.assertLess(large.loss, small.loss)
        wrong_small = dpo.sigmoid_dpo(-5.0, -2.0, -4.0, -3.0, beta=0.05)
        wrong_large = dpo.sigmoid_dpo(-5.0, -2.0, -4.0, -3.0, beta=0.5)
        self.assertGreater(wrong_large.loss, wrong_small.loss)

    def test_response_logprob_ignores_prompt_positions_and_sums_tokens(self) -> None:
        labels = [-100, -100, 257, 91, 248046, -100]
        first = dpo.response_sequence_logprob(
            [-100.0, -200.0, -0.2, -0.3, -0.4, -300.0], labels
        )
        second = dpo.response_sequence_logprob(
            [100.0, 200.0, -0.2, -0.3, -0.4, 300.0], labels
        )
        self.assertAlmostEqual(first, -0.9)
        self.assertAlmostEqual(second, -0.9)
        self.assertEqual(
            dpo.response_mask(labels), (False, False, True, True, True, False)
        )

    def test_analytic_gradient_matches_finite_difference(self) -> None:
        args = (-2.3, -4.7, -2.8, -4.0)
        beta = 0.1
        result = dpo.sigmoid_dpo(*args, beta=beta)
        epsilon = 1e-6
        plus = dpo.sigmoid_dpo(args[0] + epsilon, *args[1:], beta=beta).loss
        minus = dpo.sigmoid_dpo(args[0] - epsilon, *args[1:], beta=beta).loss
        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(numerical, result.dloss_dpolicy_chosen, places=8)

    def test_extreme_logits_remain_finite(self) -> None:
        for values in ((1000.0, -1000.0), (-1000.0, 1000.0)):
            result = dpo.sigmoid_dpo(*values, 0.0, 0.0, beta=1.0)
            self.assertTrue(math.isfinite(result.loss))

    def test_invalid_beta_mask_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(dpo.Day23DPOError, "beta"):
            dpo.sigmoid_dpo(0, 0, 0, 0, beta=0)
        with self.assertRaisesRegex(dpo.Day23DPOError, "equal length"):
            dpo.response_sequence_logprob([-1.0], [-100, 1])
        with self.assertRaisesRegex(dpo.Day23DPOError, "no supervised"):
            dpo.response_sequence_logprob([-1.0], [-100])
        with self.assertRaisesRegex(dpo.Day23DPOError, "finite"):
            dpo.sigmoid_dpo(float("nan"), 0, 0, 0, beta=0.1)

    def test_torch_logsigmoid_crosscheck_when_available(self) -> None:
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("torch is not installed in this CPU test environment")
        values = (-2.1, -4.3, -2.7, -3.8)
        beta = 0.1
        expected = dpo.sigmoid_dpo(*values, beta=beta).loss
        tensor = torch.tensor(values, dtype=torch.float64)
        delta = (tensor[0] - tensor[1]) - (tensor[2] - tensor[3])
        actual = -functional.logsigmoid(beta * delta)
        self.assertAlmostEqual(actual.item(), expected, places=14)

    def test_pinned_ms_swift_dpo_trainer_crosscheck_when_available(self) -> None:
        try:
            import torch
            from swift.rlhf_trainers.dpo_trainer import DPOTrainer
        except ImportError:
            self.skipTest("pinned ms-swift is not available in this CPU environment")

        values = (-2.1, -4.3, -2.7, -3.8)
        beta = 0.1
        expected = dpo.sigmoid_dpo(*values, beta=beta)
        trainer = object.__new__(DPOTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.reference_free = False
        trainer.f_divergence_type = "reverse_kl"
        trainer.f_divergence_params = {}
        trainer.beta = beta
        trainer.label_smoothing = 0.0
        tensors = tuple(torch.tensor([value], dtype=torch.float64) for value in values)
        losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
            *tensors, loss_type="sigmoid"
        )
        self.assertAlmostEqual(losses.item(), expected.loss, places=14)
        self.assertAlmostEqual(
            chosen_rewards.item(), expected.chosen_reward, places=14
        )
        self.assertAlmostEqual(
            rejected_rewards.item(), expected.rejected_reward, places=14
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
