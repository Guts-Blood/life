# Post-training environments

## Local Mac environment

`post_training_lab` is the reproducible common environment for data preparation,
tokenization, scoring, CPU inference, framework imports, and tiny training dry
runs.

Create it from the direct dependency contract:

```bash
conda env create -f learning/post-training-30-day-bootcamp/environments/post-training-lab-macos.yml
conda activate post_training_lab
python -m pip install --no-deps ./vendor/ms-swift
```

The local ms-swift source is pinned separately because its identity includes the
repository commit, not only the package version:

- source: `vendor/ms-swift`
- commit: `565a1ad586a21d24b23931c52d2c62b49c39bee8`
- package version: `4.5.0.dev0`

The complete resolved snapshot is
[`../artifacts/logs/post-training-lab-pip-freeze.txt`](../artifacts/logs/post-training-lab-pip-freeze.txt).

## Linux/H100 overlay

The Mac environment is not a CUDA environment. On the eventual H100 host, keep
the common Python/model/data versions above, then create a host-specific lock
after checking the installed driver and CUDA runtime. The GPU overlay owns:

- a CUDA-matched PyTorch wheel;
- `cpm-kernels`, FlashAttention and DeepSpeed;
- vLLM/SGLang when the selected inference or rollout path uses them;
- `mcore-bridge` and `megatron-core` for Megatron/TP work;
- NCCL, CUDA, driver, container digest and topology evidence;
- slime and its rollout stack only on the days that consume it.

Do not install those CUDA packages into the Mac environment merely to make
`pip check` quiet. A GPU run is comparable only after its own execution protocol
and dependency lock are frozen.

## Sandbox control-plane environment

`post_training_sandbox` is deliberately separate from `post_training_lab`.
It contains only the pinned E2B client and canonical-hash dependency used to
submit already-generated code completions to isolated sandboxes. Keeping the
client separate prevents an E2B SDK install from mutating the frozen dependency
snapshot used by the Day 10 Base generation run.

```bash
conda env create -f learning/post-training-30-day-bootcamp/environments/post-training-sandbox-macos.yml
conda activate post_training_sandbox
```

The resolved client snapshot is
[`../artifacts/logs/post-training-sandbox-pip-freeze.txt`](../artifacts/logs/post-training-sandbox-pip-freeze.txt),
with verification evidence in
[`../artifacts/logs/post-training-sandbox-environment.md`](../artifacts/logs/post-training-sandbox-environment.md).

The local credential file is `.env.e2b`; it is ignored by Git and must remain
mode `0600`. Never copy its value into logs, reports, prediction artifacts, or
sandbox environment variables.

## Current platform exception

The pinned ms-swift package declares `cpm-kernels` for every platform. It is not
installed on macOS because it is a CUDA extension. `pip check` therefore reports
exactly that one known exception; SFT, DPO, GRPO and GKD trainer imports and the
local CLI smoke tests pass.
