# Day 12 AutoDL Runbook

Day 12 reuses the already verified Day 11 Base model directory but never resumes from a Day 11 checkpoint.

## Remote layout

Upload the local Day 12 directory to:

```text
/root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/day-12-controlled-sft-checkpoints/
```

Upload the three Day 12 configs to:

```text
/root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/artifacts/configs/
```

Upload the two exact schedules to:

```text
/root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/artifacts/data/
```

The Base model, Day 08 contract, Day 09 mixture manifests and Day 10 eval manifest are already present and their hashes were verified before upload.

## Tmux

```bash
tmux new-session -A -s day12
cd /root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/day-12-controlled-sft-checkpoints
export PYTHONUNBUFFERED=1
```

Run each gate separately:

```bash
bash run_day12_autodl.sh bootstrap
bash run_day12_autodl.sh prepare
bash run_day12_autodl.sh preflight
bash run_day12_autodl.sh smoke
```

Do not start full training unless both preflights and both 10-step smokes pass.

Run A first:

```bash
bash run_day12_autodl.sh run-a
```

After A has all three `COMPLETE` markers and BF16 exports, run B:

```bash
bash run_day12_autodl.sh run-b
```

## AutoDL dev eval and E2B binding

The Day 12 dev comparison runs on the same AutoDL RTX 4090 for Base and all
six A/B checkpoints. This supersedes the earlier local-CPU execution choice,
but does not change the frozen manifest, rendering, generation, scorer, E2B
sandbox contract, checkpoint-selection rules, or frozen-test policy. The
preregistered change is recorded in:

```text
artifacts/configs/day12-autodl-eval-protocol-amendment.json
```

Generation and sandbox scoring use separate Python environments:

```text
/root/miniconda3/bin/python
/root/autodl-tmp/envs/day12-e2b/bin/python
```

The E2B environment is created with:

```bash
/root/miniconda3/bin/python -m venv /root/autodl-tmp/envs/day12-e2b
/root/autodl-tmp/envs/day12-e2b/bin/python -m pip install -r requirements-day12-e2b.txt
```

Store the E2B key outside the repository in this mode-0600 file:

```text
/root/autodl-tmp/secrets/day12-e2b.env
```

Its only line is `E2B_API_KEY='...'` (an optional `export` prefix is also
accepted). Never print, commit, upload back, or pass the key into the candidate
sandbox environment. Source it only in the E2B scoring shell:

```bash
set -a
source /root/autodl-tmp/secrets/day12-e2b.env
set +a
```

The filesystem paths and frozen hashes are recorded in:

```text
artifacts/configs/day12-autodl-e2b-binding.json
```

After sourcing the key, verify the complete binding and one disposable live
sandbox before running any code scoring:

```bash
/root/autodl-tmp/envs/day12-e2b/bin/python preflight_day12_e2b.py --live-smoke
```

The smoke creates no candidate-code result; it only checks the frozen template
attestation and immediately kills the sandbox.

First regenerate and score Base on AutoDL. Its cloud `comparison_key` and
`complete_comparison_key` become the required keys for all six checkpoints.
The Day 10 local Base numbers remain historical evidence only.

After the three-model one-sample dry run confirms a single comparison key,
launch the complete resumable dev batch in its own tmux session:

```bash
tmux new-session -d -s day12-eval \
  'cd /root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/day-12-controlled-sft-checkpoints && bash run_day12_cloud_dev_eval.sh 2>&1 | tee /root/autodl-tmp/runs/day12-20260806/day12-cloud-dev-eval.log'
```

This evaluates only the 112-record `dev` order. It generates all seven model
outputs first, verifies a common cloud comparison key, then runs the frozen E2B
code scorer, verifies a common complete comparison key, and writes one strict
summary per model. Existing complete outputs are validated and skipped, so the
batch can safely resume after an interruption.

## Outputs to download by folder

```text
/root/autodl-tmp/runs/day12-20260806/
/root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/artifacts/logs/day12-*
/root/autodl-tmp/day11-ready/post-training-30-day-bootcamp/artifacts/reports/day12-config-diff.md
```

The seven AutoDL dev evaluations must use the unchanged Day 10 runner and
frozen E2B scorer. The remote training scripts never access `frozen_test`.
