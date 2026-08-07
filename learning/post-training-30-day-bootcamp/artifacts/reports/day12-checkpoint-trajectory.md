# Day 12 — Cloud dev checkpoint trajectory

Date: 2026-08-06

Status: **dev eval complete; no eligible checkpoint; frozen test unconsumed**

## Answer first

SFT fixed the Base model's code-generation failure mode, but every checkpoint
caused an unacceptable math regression. The preregistered selection policy
therefore returns `inconclusive` for both A and B. A-25 and B-25 are descriptive
code peaks only; neither is promoted and `frozen_test` remains untouched.

All seven runs used the same AutoDL CUDA/BF16 execution protocol, frozen 112-row
dev order, tokenizer/input IDs, greedy decoding, scorers, and E2B sandbox
contract. Their shared keys are:

- prediction comparison key: `sha256:fcae58c2fe207b28bd91037672b4282d23078ae94fcdb7b5152ec56d2a56cfbf`
- complete comparison key: `sha256:f33b389071b844f8c2db221c09e4e0f0a39970718e7d4ba76fa639b90f921f8e`
- code execution protocol: `sha256:9571c64301029e230ab2d01e1b8daf8a6dcf7c4a5c8b0696e21d9b073a730c87`

E2B infrastructure failures: **0**.

## Dev results

Each slice contains 28 examples. `Total` is the equal-weight four-slice result
because all slices have the same size.

| Model | General | Math | Code/E2B | Finance | Total | Output tokens | Eligible? |
|---|---:|---:|---:|---:|---:|---:|---|
| Cloud Base | 2 | 12 | 0 | 1 | 15/112 (13.39%) | 43,904 | Baseline |
| A-25% | 1 | 0 | 7 | 1 | 9/112 (8.04%) | 3,994 | No: math −12 |
| A-60% | 0 | 4 | 5 | 1 | 10/112 (8.93%) | 8,313 | No: math −8 |
| A-100% | 1 | 2 | 3 | 1 | 7/112 (6.25%) | 5,129 | No: math −10 |
| B-25% | 3 | 1 | 9 | 0 | 13/112 (11.61%) | 4,946 | No: math −11 |
| B-60% | 0 | 0 | 4 | 0 | 4/112 (3.57%) | 3,968 | No: math −12 |
| B-100% | 0 | 0 | 5 | 0 | 5/112 (4.46%) | 5,851 | No: math −12 |

The guardrail permits at most two lost correct cases versus Base in each of
general, math, and finance. Every candidate loses at least eight math cases, so
the code tie-band step is never reached for formal selection.

## Trajectory interpretation

- Run A code follows `7 → 5 → 3`; Run B follows `9 → 4 → 5`. Both peak at 25%
  and then regress. Training loss or the final checkpoint would have selected
  the wrong point for the target slice.
- A-60 partially recovers math to 4/28, but that is still eight cases below
  Base and remains ineligible.
- B-25 is the strongest descriptive compromise: 9/28 code, 3/28 general, and
  13/112 overall. It still loses 11/28 math cases and cannot be promoted.
- SFT reduces output tokens by 81–91% versus Base and removes the universal
  max-token-ceiling behavior. Shorter, better-formed answers helped code, but
  brevity is not itself quality: math accuracy collapsed.

## Selection decision

- Run A: `inconclusive`, selected checkpoint `null`.
- Run B: `inconclusive`, selected checkpoint `null`.
- Frozen confirmation: not run; frozen set remains unconsumed.

The durable machine-readable decision is
`day12-cloud-dev-selection-outcome.json`. Raw predictions, E2B sidecars, and
strict per-model summaries remain under
`/root/autodl-tmp/runs/day12-20260806/eval/` on AutoDL.

## Claim boundary

This is a 28-example-per-slice learning experiment. The most defensible claim
is that this SFT recipe trades Base math ability for an early code gain on this
specific model, seed, data, and token budget. It does not show that targeted
mixture B is generally superior, nor that later checkpoints are generally
worse.
