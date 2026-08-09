# Post-training environments

## Day 1–12 historical environment

`post_training_lab` is the reproducible common environment for data preparation,
tokenization, scoring, CPU inference, framework imports, and tiny training dry
runs.

This environment, its Python 3.11 runtime, Transformers 4.57.3 dependency, and
the Qwen3-0.6B artifacts are the immutable Day 1–12 reproduction chain. Do not
upgrade it in place for Qwen3.5. Historical checkpoints must continue to load
under this lock even after the v2 environment exists.

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

## Qwen3.5-4B v2 environment contract

Qwen3.5 work uses a separate Python 3.12 environment and artifact namespace.
This README records the contract only; Day 15 freezes the exact environment
file, lock, container digest, and smoke evidence after testing on the selected
CUDA host.

The minimum candidate line is an exact Transformers 5.2.0-or-newer release that
passes import, processor/template, text-only forward/backward, save/reload, and
generation parity checks. The current official
[ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)
recommends `transformers>=5.9`, so Day 15 should normally select and pin one
exact release at or above 5.9 unless runtime evidence documents why a different
5.2+ release is required. A floating lower bound is never a finished run lock.

Day 15 freezes these identities together rather than upgrading them
independently:

- Python 3.12 patch release and Linux/container digest;
- Transformers exact version;
- ms-swift package version and repository commit;
- `qwen_vl_utils` and `decord` exact versions;
- PyTorch, CUDA, NCCL, FlashAttention and any GatedDeltaNet/FLA or Liger kernel
  dependency actually used;
- PEFT, TRL, Accelerate, vLLM/SGLang or slime versions when the selected path
  imports them;
- the Qwen3.5-4B-Base model revision, processor/tokenizer/template files, and
  their hashes.

The full
[Qwen3.5-4B-Base checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B-Base) is a
vision-language model. Load it with `AutoProcessor` and
`AutoModelForMultimodalLM` or the explicit
`Qwen3_5ForConditionalGeneration` class described in the
[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
It is not an `AutoModelForCausalLM` drop-in. A separately converted text-only
backbone would be a new artifact with its own conversion command, hashes, and
parity report.

The initial v2 training scope is text-only coding. Loading the full VLM does not
authorize vision training: before optimizer creation, runtime checks must prove
that the vision tower and aligner have no trainable parameters or adapter
targets; after backward and optimizer creation, their gradients and optimizer
state must also be empty. The assertions and trainable-module manifest are
promotion evidence, not optional log messages.

Qwen3.5 has a native 262,144-token context, but the first RL lock caps total
prompt plus completion length at 8K–12K. A larger context requires a new KV,
activation, throughput, and peak-memory gate. Packing/padding-free and custom
GDN kernels remain disabled unless their exact versioned path passes an
independent correctness smoke.

## Linux/H100 overlays

The Mac environment is not a CUDA environment. On the eventual H100 host, keep
the corresponding historical or v2 Python/model/data versions above, then
create a host-specific lock after checking the installed driver and CUDA
runtime. Never layer v2 packages over the Day 1–12 environment. The GPU overlay
owns:

- a CUDA-matched PyTorch wheel;
- `cpm-kernels`, FlashAttention and DeepSpeed;
- vLLM/SGLang when the selected inference or rollout path uses them;
- `mcore-bridge` and `megatron-core` for Megatron/TP work;
- NCCL, CUDA, driver, container digest and topology evidence;
- slime and its rollout stack only on the days that consume it.

Do not install those CUDA packages into the Mac environment merely to make
`pip check` quiet. A GPU run is comparable only after its own execution protocol
and dependency lock are frozen.

There is no official NVIDIA peak-memory number for the exact v2 combination of
Qwen3.5-4B-Base, text-only data, frozen vision/aligner, the selected adapter or
full-parameter objective, sequence cap, and learner/rollout placement. Official
examples and Ascend/NPU PR or one-step results are codepath evidence only. Each
NVIDIA topology must complete a full optimizer-step peak smoke and retain
10%–15% free headroom before it is approved.

Disk readiness is dynamic rather than a fixed GiB threshold. Account for the
base snapshot, optimizer/distributed shards, every retained checkpoint, merged
export, rollout/log artifacts, and one staging copy, then retain at least 15%
filesystem headroom. Recompute the gate from actual save/reload bytes whenever
checkpoint retention or export format changes.

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

Deterministic reward functions, formatting checks, code execution and verifier
tests for v2 RL also run in a CPU sandbox/control-plane environment. They do not
share the learner/rollout Python environment or consume GPU merely because the
policy does. A model-based judge is a separately declared model role with its
own resource and provenance contract.

## Current platform exception

The pinned ms-swift package declares `cpm-kernels` for every platform. It is not
installed on macOS because it is a CUDA extension. `pip check` therefore reports
exactly that one known exception; SFT, DPO, GRPO and GKD trainer imports and the
local CLI smoke tests pass.
