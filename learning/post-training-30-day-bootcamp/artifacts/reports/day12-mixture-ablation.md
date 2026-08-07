# Day 12 — A/B mixture ablation

## Conclusion

The experiment does **not** support a B-over-A mixture claim under the frozen
decision rule. B has higher HumanEval counts at 25% and 100%, but the paired net
wins are only +2 at both budgets, below the preregistered +3 threshold. At 60%,
B is behind A by one net case. Every A and B checkpoint also fails the math
guardrail, which independently makes the matched-budget claims inconclusive.

| Budget | A code | B code | Both correct | B-only | A-only | Net B−A | +3 threshold |
|---|---:|---:|---:|---:|---:|---:|---|
| 25% | 7 | 9 | 6 | 3 | 1 | +2 | Not met |
| 60% | 5 | 4 | 3 | 1 | 2 | −1 | Not met |
| 100% | 3 | 5 | 1 | 4 | 2 | +2 | Not met |

At the most promising 25% budget, B improves code by two cases and general by
two cases relative to A, while losing one finance case; both have catastrophic
math regression versus Base (A: −12, B: −11). This is useful recipe evidence,
but not a promotable result.

The next controlled experiment should preserve or upweight math-capability data
and retest an earlier/smaller token budget. It should not use the frozen test to
tune that change; the current frozen set remains clean for a future eligible
selection.
