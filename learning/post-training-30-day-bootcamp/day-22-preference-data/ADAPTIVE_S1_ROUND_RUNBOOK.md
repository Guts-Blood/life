# Day 22 adaptive S1 round 1

This round is additive. It never overwrites the frozen first-round rollout,
selection, or evidence artifacts.

## Frozen scope

- targets: all 189 first-round execution quarantines;
- composition: 101 `both_pass`, 85 `both_fail`, 3 runtime-only;
- promoted model: the exact parent S1 downstream-ready merged export;
- generation: `temperature=0.8`, `top_p=0.95`, `K=12`;
- identity: supplemental sample indexes 6–17 and a new seed domain;
- partition: family-disjoint GPU shards, 1,140 and 1,128 candidates;
- execution gate: every retained new unique response needs exactly two fresh
  pinned E2B runs. Runtime, timeout, syntax, and infrastructure failures remain
  quarantined.

## AutoDL rollout

Place these files in one importable remote directory without changing bytes:

```text
rollout_day22_s1.py
rollout_day22_s1_adaptive.py
formal_s1_pair_labeler.py
day22_contract.py
```

The parent `rollout_day22_s1.py` must retain SHA-256
`fc089b3d530f28270aaeb6e0628c756d84b68ae80c43e9a657b64ca3c3f2fd7c`.
The adaptive script SHA-256 is
`595c7fdba4dae30e268198f18bc264a1535b0e7bb9b79906cce2dd8349effd90`.

Use the same `DAY22_BOOTCAMP_ROOT`, `DAY22_MS_SWIFT_ROOT`, offline Hugging Face
settings, Python environment, and model export used by the parent rollout.

```bash
PY=/root/autodl-tmp/qwen35-v2/venv/bin/python
CODE=/root/autodl-tmp/day22-preference-20260813/adaptive-inputs
BASE=/root/autodl-tmp/day22-preference-20260813
OUT=/root/autodl-tmp/day22-preference-20260813/adaptive-r1-t08-k12

PYTHONPATH="$CODE" "$PY" "$CODE/rollout_day22_s1_adaptive.py" prepare \
  --base-rollout-contract "$BASE/rollout/rollout-contract.json" \
  --base-selections "$CODE/day22-qwen35-formal-s1-selections.jsonl" \
  --base-selection-summary "$CODE/day22-qwen35-formal-s1-selection-summary.json" \
  --seed-manifest "$BASE/inputs/day22-qwen35-mbpp-seed-manifest.json" \
  --output-root "$OUT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE" "$PY" \
  "$CODE/rollout_day22_s1_adaptive.py" run \
  --contract "$OUT/adaptive-rollout-contract.json" --shard-id 0

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$CODE" "$PY" \
  "$CODE/rollout_day22_s1_adaptive.py" run \
  --contract "$OUT/adaptive-rollout-contract.json" --shard-id 1

PYTHONPATH="$CODE" "$PY" "$CODE/rollout_day22_s1_adaptive.py" status \
  --contract "$OUT/adaptive-rollout-contract.json"

PYTHONPATH="$CODE" "$PY" "$CODE/rollout_day22_s1_adaptive.py" merge \
  --contract "$OUT/adaptive-rollout-contract.json" \
  --output-jsonl "$OUT/day22-s1-adaptive-r1-candidates.jsonl" \
  --output-manifest "$OUT/day22-s1-adaptive-r1-candidates-manifest.json"
```

Run the two `run` commands concurrently (normally under `nohup` with separate
logs). A rerun resumes verified candidate rows. `prepare` and `merge` verify an
existing identical result and refuse a differing output.

## Local round combination and scoring

```bash
DAY22=learning/post-training-30-day-bootcamp/day-22-preference-data
ART=learning/post-training-30-day-bootcamp/artifacts

python3 "$DAY22/combine_day22_s1_rounds.py" combine-rollouts \
  --base-rollouts "$ART/data/day22-qwen35-formal-s1-rollout-candidates.jsonl" \
  --base-contract "$ART/configs/day22-qwen35-formal-s1-rollout-contract.json" \
  --adaptive-rollouts <downloaded-adaptive-candidates.jsonl> \
  --adaptive-contract <downloaded-adaptive-contract.json> \
  --seed-manifest "$ART/data/day22-qwen35-mbpp-seed-manifest.json" \
  --combined-rollouts-output <combined-normalized-rollouts.jsonl> \
  --supplemental-requests-output <supplemental-e2b-requests.jsonl> \
  --manifest-output <round-combine-manifest.json>

/Users/jiaweiqian/miniforge3/envs/post_training_sandbox/bin/python \
  "$DAY22/score_day22_s1_candidates.py" \
  --input <supplemental-e2b-requests.jsonl> \
  --output <supplemental-e2b-evidence.jsonl> \
  --workers 8

python3 "$DAY22/combine_day22_s1_rounds.py" merge-evidence \
  --combined-rollouts <combined-normalized-rollouts.jsonl> \
  --combine-manifest <round-combine-manifest.json> \
  --base-evidence "$ART/eval/day22-qwen35-formal-s1-candidate-e2b-evidence.jsonl" \
  --supplemental-evidence <supplemental-e2b-evidence.jsonl> \
  --output <combined-e2b-evidence.jsonl> \
  --manifest-output <evidence-merge-manifest.json>

python3 "$DAY22/formal_s1_pair_labeler.py" \
  --rollouts <combined-normalized-rollouts.jsonl> \
  --candidate-evidence <combined-e2b-evidence.jsonl> \
  --selections-output <combined-selections.jsonl> \
  --replay-output <combined-replay.jsonl> \
  --summary-output <combined-selection-summary.json>
```

For E2B sharding, run `score_day22_s1_candidates.py` with disjoint
`--shard-id/--shard-count` outputs, concatenate in shard order, and use that
complete file as `--supplemental-evidence`. The merge command rejects missing,
extra, or duplicate candidate evidence.

The existing processor auditor, formal assembler, and independent validator
consume the new labeler outputs unchanged. Multi-config summaries carry
`generator_provenance_sha256s` plus a single content-bound
`common_policy_identity`; the legacy singular generator field remains for
single-config inputs.
