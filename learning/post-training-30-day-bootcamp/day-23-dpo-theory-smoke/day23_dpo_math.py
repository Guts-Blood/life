#!/usr/bin/env python3
"""Small, dependency-free scalar oracle for the canonical sigmoid DPO loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


class Day23DPOError(ValueError):
    """A DPO scalar or response-mask invariant failed."""


@dataclass(frozen=True)
class DPOResult:
    loss: float
    logit: float
    policy_margin: float
    reference_margin: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    dloss_dpolicy_chosen: float
    dloss_dpolicy_rejected: float


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Day23DPOError(f"{label} must be finite")
    return result


def _softplus(value: float) -> float:
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def sigmoid_dpo(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    *,
    beta: float,
) -> DPOResult:
    """Return loss, rewards, margins, and analytic policy-logprob gradients."""
    pc = _finite(policy_chosen, "policy_chosen")
    pr = _finite(policy_rejected, "policy_rejected")
    rc = _finite(reference_chosen, "reference_chosen")
    rr = _finite(reference_rejected, "reference_rejected")
    beta = _finite(beta, "beta")
    if beta <= 0:
        raise Day23DPOError("beta must be positive")
    policy_margin = pc - pr
    reference_margin = rc - rr
    logit = beta * (policy_margin - reference_margin)
    loss = _softplus(-logit)
    # d softplus(-x) / dx == -sigmoid(-x), with x=beta*delta.
    if logit >= 0:
        exp_negative = math.exp(-logit)
        sigmoid_negative = exp_negative / (1.0 + exp_negative)
    else:
        sigmoid_negative = 1.0 / (1.0 + math.exp(logit))
    chosen_gradient = -beta * sigmoid_negative
    rejected_gradient = beta * sigmoid_negative
    chosen_reward = beta * (pc - rc)
    rejected_reward = beta * (pr - rr)
    return DPOResult(
        loss=loss,
        logit=logit,
        policy_margin=policy_margin,
        reference_margin=reference_margin,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        reward_margin=chosen_reward - rejected_reward,
        dloss_dpolicy_chosen=chosen_gradient,
        dloss_dpolicy_rejected=rejected_gradient,
    )


def response_sequence_logprob(
    token_logprobs: Sequence[float], labels: Sequence[int], *, ignore_index: int = -100
) -> float:
    """Sum causal token log-probs only where the response label is supervised."""
    if len(token_logprobs) != len(labels):
        raise Day23DPOError("token_logprobs and labels must have equal length")
    if not labels:
        raise Day23DPOError("token arrays must not be empty")
    total = 0.0
    supervised = 0
    for index, (logprob, label) in enumerate(zip(token_logprobs, labels)):
        value = _finite(logprob, f"token_logprobs[{index}]")
        if int(label) != ignore_index:
            total += value
            supervised += 1
    if supervised == 0:
        raise Day23DPOError("response mask has no supervised tokens")
    return total


def response_mask(labels: Iterable[int], *, ignore_index: int = -100) -> tuple[bool, ...]:
    mask = tuple(int(label) != ignore_index for label in labels)
    if not mask or not any(mask):
        raise Day23DPOError("response mask must supervise at least one token")
    return mask
