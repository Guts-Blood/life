# Day 12 Recovery Rollouts C–L: Final Retrospective

## Executive conclusion

The preregistered recovery program completed all ten rollouts, C through L. No single checkpoint passed the frozen dev acceptance gate of `math >= 10/28`, `code >= 5/28`, and `total >= 15/112`. The stopping rule was therefore reached without a promotion candidate, and the frozen test remained unconsumed.

The experiment did not produce an acceptable monolithic model, but it isolated a stable trade-off:

- Full-parameter Run E learned code strongly (`9/28`) but catastrophically forgot Base math (`4/28`).
- Low-update LoRA Run G retained Base math (`12/28`) but learned no executable code (`0/28`).
- Higher-update LoRA Runs H and I moved toward code (`3/28`) while math fell to `8/28`.
- Interpolating LR, emphasizing math/code data, adding direct `lm_head` LoRA, and doubling rank did not cross the gap.

The strongest next direction is therefore not another unstructured mixture tweak. It is an explicitly preregistered modular or retention-regularized system: route math to Base and general/code/finance to the still-retained Run E export, or train an E-like code-capable model with Base KL/replay on math.

## Frozen protocol and stopping decision

- Recovery rollout IDs: `C,D,E,F,G,H,I,J,K,L`.
- Early checkpoint: exactly `61,734` supervised tokens (`25_percent`).
- Dev set: 112 records, 28 per slice.
- Comparison key: `sha256:fcae58c2fe207b28bd91037672b4282d23078ae94fcdb7b5152ec56d2a56cfbf`.
- Complete E2B comparison key: `sha256:f33b389071b844f8c2db221c09e4e0f0a39970718e7d4ba76fa639b90f921f8e`.
- Code backend: fresh no-network E2B Firecracker sandbox per sample.
- Acceptance: all three gates required simultaneously.
- Result: 10/10 rollouts completed, 0 accepted, frozen test consumed: `false`.

## Score trace

| Model | General /28 | Math /28 | Code /28 | Finance /28 | Total /112 | Gate result |
|---|---:|---:|---:|---:|---:|---|
| Base | 2 | 12 | 0 | 1 | 15 | code fail |
| C | 0 | 1 | 3 | 1 | 5 | all fail |
| D | 3 | 2 | 5 | 2 | 12 | math, total fail |
| E | 4 | 4 | 9 | 2 | 19 | math fail |
| F | 9 | 3 | 0 | 0 | 12 | all fail |
| G | 2 | 12 | 0 | 0 | 14 | code, total fail |
| H | 8 | 8 | 3 | 0 | 19 | math, code fail |
| I | 5 | 8 | 3 | 0 | 16 | math, code fail |
| J | 2 | 9 | 1 | 0 | 12 | all fail |
| K | 2 | 9 | 0 | 0 | 11 | all fail |
| L | 4 | 8 | 1 | 0 | 13 | all fail |

The useful Pareto frontier is Base `(math 12, code 0)`, H/I `(math 8, code 3)`, and E `(math 4, code 9)`. No observed single model occupies the required quadrant `(math >= 10, code >= 5)`.

## Causal rollout trace

### C — reduce finance exposure

- Change: finance supervised-token share `25% -> 12.5%`; released tokens split over general/math/code.
- Result: `0/1/3/1`, total 5.
- Conclusion: finance share alone was not the primary cause of the regression.

### D — repair prompt/target format

- Change: natural task-routing suffixes; FinQA DSL rewritten as natural arithmetic ending in `Final answer:`.
- Result: `3/2/5/2`, total 12, +7 over C.
- Conclusion: format leakage was real and repairable, but math retention remained poor.

### E — lower full-parameter LR

- Change: LR `2e-5 -> 1e-5`, every data/order invariant retained.
- Result: `4/4/9/2`, total 19.
- Conclusion: best executable code checkpoint, but full-parameter SFT still overwrote math.

### F — continue lowering full-parameter LR

- Change: LR `1e-5 -> 5e-6`.
- Result: `9/3/0/0`, total 12; all samples hit generation ceilings.
- Conclusion: retention did not improve monotonically; code capability collapsed.

### G — freeze Base and use rank-16 LoRA

- Change: full-parameter training -> rank-16, alpha-32, all-linear LoRA at E's `1e-5`.
- Result: `2/12/0/0`, total 14.
- Conclusion: freezing Base recovered math exactly, but update magnitude was insufficient for code.

### H — increase LoRA LR

- Change: LoRA LR `1e-5 -> 5e-5`.
- Result: `8/8/3/0`, total 19.
- Conclusion: code/general acquisition increased as math retention decreased; both constrained gates missed by two.

### I — emphasize math and code data

- Change: ratios `7/24,7/24,7/24,1/8 -> 1/8,5/12,5/12,1/24` for general/math/code/finance.
- Result: `5/8/3/0`, total 16.
- Conclusion: much more math/code exposure changed neither math nor code; raw task scarcity was not the bottleneck.

### J — interpolate LoRA LR

- Change: I LR `5e-5 -> 3e-5`.
- Result: `2/9/1/0`, total 12.
- Conclusion: one math case was recovered, but two code cases were lost. The LR trade-off is real but does not intersect the gate.

### K — directly adapt output projection

- Change: J's decoder linear LoRA targets -> the same projections plus tied `lm_head`.
- Result: `2/9/0/0`, total 11; 112/112 ceilings unchanged.
- Engineering result: PEFT warning was handled; safe merge, standalone BF16 load, tied weights, and long generation all verified.
- Conclusion: output-head coverage was not the main quality or termination bottleneck.

### L — double adapter rank

- Change: return to I and change rank `16 -> 32`, alpha fixed at 32 (`alpha/r: 2 -> 1`).
- Result: `4/8/1/0`, total 13.
- Conclusion: additional capacity plus lower scaling did not improve retention or code; maximum rollout count reached.

## What the ten runs established

1. **The user's finance-interference hypothesis was partly useful but insufficient.** Reducing finance and removing its DSL improved the experiment only after format repair; finance share itself did not explain the large math drop.
2. **The central problem is conflicting adaptation, not simply too little training.** Full-parameter updates can learn code, while frozen-Base LoRA can retain math, but the tested shared adapter could not do both.
3. **The update frontier is non-monotonic.** Lower full-parameter LR erased code without restoring math; lower LoRA LR restored math gradually while also erasing code.
4. **More targeted data did not move the frontier.** I had substantially more math/code tokens than H but identical constrained scores.
5. **Termination is a cross-cutting failure.** Base and every recovery model hit all 112 slice-specific generation ceilings. The runner explicitly used tokenizer EOS `<|im_end|>`; this was not a generation-config mismatch.
6. **Direct output adaptation did not fix termination.** K rules out the simplest `lm_head` explanation under this recipe.
7. **Capacity was not the missing variable at this scale.** L doubled LoRA rank and trainable parameters but regressed code.

## Post-hoc modular opportunity (not an accepted result)

Run E's BF16 export is still complete on AutoDL at:

`/root/autodl-tmp/runs/day12-20260806/inference-exports/E-25_percent`

A post-hoc per-task composition using Base for math and E for general/code/finance has the arithmetic dev upper bound:

- general: E `4`
- math: Base `12`
- code: E `9`
- finance: E `2`
- total: `27/112`

This would pass all three numerical gates (`math 12`, `code 9`, `total 27`), but it is **not** a valid Day 12 acceptance result because the router was not preregistered and the same dev outcomes suggested the policy. It should be evaluated only as a new experiment with a frozen routing rule and a fresh holdout.

## Recommended next phase

### Priority 1 — preregister a routed Base + E system

- Define a deterministic task classifier before looking at a new holdout.
- Route math to Base; route code, general, and finance to E.
- Measure router errors as part of end-to-end evaluation; do not use oracle slice labels at scoring time.
- Freeze a new dev/test split because the current dev has now supported ten sequential choices.

This is the lowest-cost path because Base and the complete E export already exist.

### Priority 2 — E-like full-parameter SFT with explicit retention

- Start from E's code-capable recipe.
- Add one preregistered retention mechanism: Base KL on math/general replay or frozen-Base logit distillation.
- Track per-domain gradient cosine and KL drift at the same 25% token budget.
- Keep code data/format fixed so the only primary variable is retention regularization strength.

### Priority 3 — fix sequence termination in a fresh protocol

- Build a concise-completion curriculum with verified assistant `<|im_end|>` supervision.
- Consider explicit EOS loss weighting as a single-variable ablation.
- Pre-register termination rate and mean generated tokens as guardrail metrics, not merely accuracy diagnostics.

### Priority 4 — separate adapters rather than one mixed adapter

- Train a code adapter and use Base for non-code tasks, or use domain-specific adapters behind a router.
- This matches the observed Pareto frontier better than forcing a single shared adapter through conflicting gradients.

## Evidence and reproducibility

For every rollout, the repository retains:

- rollout spec and resolved config;
- exact data manifest and supervised-token schedule;
- preflight, smoke summary, stage summary, checkpoint/export manifest hashes, and step trace;
- 112 raw dev predictions;
- 28 E2B code results;
- deterministic eval summary and RFC-8785 semantic hash;
- outcome report and deletion/recoverability record.

Failed full-state checkpoints and storage-constraining exports were deleted only after their evidence chains were synchronized. Deleted artifacts are recoverable by retraining only. The final machine-readable index is `artifacts/reports/day12-recovery-rollout-trace.json`.

## Limitations

- Only one training seed was used.
- Each slice contains 28 dev examples; score changes of one or two cases have wide uncertainty.
- Ten sequential decisions create dev-selection pressure even though the frozen test was protected.
- All models hit generation ceilings, so accuracy is entangled with response termination and trailing-text behavior.
- L's rank change also changed `alpha/r` by construction; this was explicit and audited, but capacity and scaling cannot be separated from that single result.

## Final status

- Recovery rollouts completed: `10/10`.
- Accepted checkpoint: `none`.
- Frozen test consumed: `false`.
- Failed L checkpoint/export deleted after evidence preservation.
- Recommended operational artifact to retain: Run E BF16 export plus Base, pending a fresh preregistered routed-system evaluation.
