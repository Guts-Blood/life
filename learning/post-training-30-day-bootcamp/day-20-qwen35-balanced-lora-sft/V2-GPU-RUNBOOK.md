# Day 20 v2 — candidate factory GPU runbook

This runbook is the executable handoff for the v2 experiment. It does not
reinterpret or overwrite any Day 20 v1 artifact.

## What is already confirmed locally

The canonical historical run is
`remote-runs/day20-qwen35-lora-20260809T042005Z`.

- `base-probe`, `probe-1e-5`, `probe-3e-5`, and `probe-1e-4` each have a
  complete 32-record raw inference pair and a complete `qwen35-v2` normalized
  pair.
- All four Day 20 Code E2B pairs are absent.
- `PROBE-SELECTION.json` is absent.
- Day 20 full-112 inference and E2B are absent.
- The old driver stopped because
  `/root/autodl-tmp/envs/day12-e2b/bin/python` did not exist.

Those files are historical diagnostics only. Their candidate grammar,
experiment manifest, response adapter, and comparison key are v1 identities,
so they are not valid inputs to v2 selection.

The local workspace has the frozen Qwen3.5-4B Base snapshot and the repository
Day 09/Day 10 manifests and scorers. It does not have the four full normalized
Day 20 source pools, the pinned ms-swift environment, or the E2B environment.
Therefore local work stops at deterministic offline tests; source
materialization, live template tokenization, inference, and sandbox execution
belong on the GPU host.

## Changes and intended effects

| Layer | v2 change | Expected effect |
| --- | --- | --- |
| Data | Probe `24,000`, Main `320,000` supervised tokens; exactly 25% per skill | More learning signal without changing the four-skill objective |
| Coverage | Probe includes at least 1k FinQA and 1k new Tulu Code tokens; Main includes at least 15k each | Reduce Finance template memorization and Code source narrowness |
| Target | Code is an AST-validated function-body continuation; inference receives no trimming or fence repair | Make train/eval semantics identical and expose invalid generations honestly |
| Order | Deterministic token-deficit interleave, batch 8, rolling eight-step coverage, 20–30% cumulative share gates | Prevent long runs of one skill and late-stage forgetting |
| Optimization | Resumable Probe checkpoints at 6k/12k/18k/24k; Main at 80k/192k/320k | Observe the trajectory and recover without restarting a recipe |
| Search | Stage A `1e-4`; only a near miss opens `3e-5`, `6e-5`, `8e-5` | Spend GPU on the bracket only when evidence justifies it |
| Infrastructure | Live E2B create/info/kill smoke before any training | Fail before GPU spend when sandbox runtime or credentials are broken |
| Evaluation | Explicit `probe32`/`full112`, v3 response boundary, common comparison identities, candidate-specific run hashes | Prevent cross-run or cross-candidate artifact reuse |

Probe strict pass remains: total at least Base +2; Math, Finance, and Code no
worse than Base -1; Code sandbox eligibility at least 7/8; infrastructure
failures zero. A near miss is total at least Base, the same three skills no
worse than Base -2, Code eligibility at least 6/8, and infrastructure failures
zero. Anything worse stops without opening the bracket.

Main promotion floors remain: total 65/112, General 5/28, Math 17/28,
Finance 10/28, Code 14/28, format compliance 90/112, Code eligibility 26/28,
and infrastructure failures zero.

## Remote prerequisites

The defaults expect:

- Base model:
  `/root/autodl-tmp/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b`
- Python environment: `/root/autodl-tmp/qwen35-v2/venv/bin/python`
- clean ms-swift checkout at commit
  `565a1ad586a21d24b23931c52d2c62b49c39bee8`
- `ms-swift==4.5.0.dev0`, `transformers==5.12.1`, `peft==0.19.1`, and
  `torch==2.10.0+cu126`
- GPU 0 matching `H800`, BF16 support, and at least 80 GiB free under
  `/root/autodl-tmp`
- E2B Python `/root/autodl-tmp/envs/day12-e2b/bin/python` with
  `e2b==2.37.0`
- four pre-existing normalized source pools under
  `/root/autodl-tmp/qwen35-v2/day20-sources`

Overrides are explicit `DAY20_V2_*` environment variables in
`run_day20_v2_autodl.sh`; changing one becomes part of the resulting evidence.

The rotated credential must be transferred out of band to
`/root/autodl-tmp/secrets/day20-v2-e2b.env` as a root-owned mode-0600 file with
exactly one line, `E2B_API_KEY=<secret>`. Do not place the secret in a command
argument, repository file, or log. Create its non-secret attestation with:

```bash
cd /root/autodl-tmp/post-training-30-day-bootcamp/day-20-qwen35-balanced-lora-sft
/root/autodl-tmp/qwen35-v2/venv/bin/python day20_e2b_preflight_v2.py \
  attest-credential \
  --credential-file /root/autodl-tmp/secrets/day20-v2-e2b.env \
  --output /root/autodl-tmp/secrets/day20-v2-e2b-attestation.json
```

The helper stores only file hashes and timestamps, never the secret.

## Canonical execution

Run each command in the v2 directory. Every publication is no-overwrite;
completed train/eval/selection artifacts are revalidated before reuse. An
existing source expansion is revalidated by the following `prepare` command.

```bash
bash run_day20_v2_autodl.sh init
bash run_day20_v2_autodl.sh source-expand
bash run_day20_v2_autodl.sh prepare
bash run_day20_v2_autodl.sh preflight
bash run_day20_v2_autodl.sh probe-funnel
```

`probe-funnel` evaluates Base, trains and evaluates all four Stage A
milestones, then opens the three-LR bracket only when Stage A is classified as
a near miss. A strict pass proceeds to Main; a gross fail or a completed
bracket without a strict pass keeps Base active.

For diagnosis or recovery, the lower-level idempotent commands are:

```bash
bash run_day20_v2_autodl.sh train-probe 1e-4
bash run_day20_v2_autodl.sh eval probe-s20260809-lr1e-4-t12000
bash run_day20_v2_autodl.sh sandbox probe-s20260809-lr1e-4-t12000
bash run_day20_v2_autodl.sh select-probe stage-a
bash run_day20_v2_autodl.sh status
```

After a passing probe, continue with the evidence-selected learning rate:

```bash
bash run_day20_v2_autodl.sh main
bash run_day20_v2_autodl.sh confirm
bash run_day20_v2_autodl.sh finalize
```

`main` evaluates Base full-112 plus the primary seed's early/mid/final
checkpoints. `confirm` trains seed `20260810` at the selected learning rate and
evaluates the exact primary winning checkpoint. `finalize` requires that
confirmation to pass the unchanged Main floors; otherwise it records Base
fallback. Merge/export is deliberately outside this local/GPU training
handoff.

## Stop conditions

Stop without manual JSON edits when any of the following occurs:

- source quotas or exact token budgets are infeasible;
- temporal-mix audit fails;
- model, data, config, runtime, or checkpoint identity drifts;
- any optimizer/scheduler/RNG state is missing from a milestone;
- E2B preflight cannot create, attest, and kill a live sandbox;
- an eval pair is partial or fails its row/file/content hashes;
- comparison keys or sample order differ across a cohort;
- no probe strict pass exists, or Main/confirmation misses a promotion floor.

These are experiment outcomes or infrastructure blockers, not reasons to
relax thresholds or reuse v1 evidence.
