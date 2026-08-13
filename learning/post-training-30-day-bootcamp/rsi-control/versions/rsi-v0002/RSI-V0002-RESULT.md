# RSI v0002 result

## Outcome

`rsi-v0002` is `complete_qualified`. The frozen final selector promoted:

`main-s20260809-lr1e-4-final`

Remote checkpoint:

`/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/adapters/main/s20260809/lr1e-4/attempt-001/checkpoint-1904`

The claim is limited to qualification on the frozen full112 suite. It does not
by itself establish general financial-agent improvement.

## Frozen full112 evidence

| Run | Candidate | G | M | F | Code | Total | Eligible | Format | Infra |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary, seed 20260809 | `main-s20260809-lr1e-4-final` | 21 | 21 | 19 | 20 | 81 | 28 | 112 | 0 |
| Confirmation, seed 20260810 | `main-s20260810-lr1e-4-final` | 21 | 21 | 14 | 17 | 73 | 28 | 112 | 0 |

Both runs used LR `1e-4`, checkpoint label `final`, the same Base-start recipe,
the same sample order and comparison identity, and independent seeds. Both pass
every frozen floor. `FINAL-PROMOTION.json` has file SHA-256
`c5ce7f522706040ac1628659ccc1ee53c2ac61eb820861a092a5dcd7bf8049e4`
and content SHA-256
`de72a8f43da23e955e9525bccb365b49edcba7976018d6b58a9b34f903c2caa7`.

The final read-only collector verified 44 items with no incomplete or rejected
evidence; inventory SHA-256 is
`9d6a78e9954e37e74fae9698ad6729dc253cb3c8156f3008d86ad24ed7ccab5b`.

## Intervention result

The v0001 failure was caused by the native Qwen3.5 training template stripping
or masking the required first four-space Code continuation token. v0002 changed
only target representation plus its necessary balanced loss-mask exchange.
Data rows, order, per-row supervised-token counts, LoRA recipe, inference
template, eval cases, scorers and gates remained frozen.

Stage-A recovered Code eligibility from `0/8` at v0001 t12000+ to `8/8` at all
four v0002 checkpoints. The selected Probe checkpoint scored 23/32 with Code
5/8, and the full Main runs independently cleared all promotion floors.

## Cost and topology

- v0002: `1.840072835 GPU-hours`, 10 eval accesses.
- Prior v0001 diagnostic version: `0.251139846 GPU-hours`.
- Cumulative through first qualification: `2.091212681 GPU-hours`.

Training used one RTX PRO 6000 Blackwell GPU because the 4B LoRA workload fit
comfortably at about 25 GiB and the run contract froze global batch, sampler,
sample order, checkpoint-token schedule and topology. Switching Primary or
Confirmation to DDP would have introduced a new runtime/optimization variable;
it could reduce wall time but was not expected to reduce GPU-hours. A future
version can benchmark two-GPU DDP separately after proving exact batch/order and
checkpoint equivalence.
