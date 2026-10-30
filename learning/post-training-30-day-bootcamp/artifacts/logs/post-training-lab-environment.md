# `post_training_lab` environment evidence

Date: `2026-08-05`

Status: `local_common_layer_passed_with_cuda_platform_exception`

## Identity

| Field | Frozen value |
|---|---|
| Conda environment | `post_training_lab` |
| Prefix | `/Users/jiaweiqian/miniforge3/envs/post_training_lab` |
| Platform | `macOS-26.5.2-arm64-arm-64bit` |
| Python | `3.11.15` |
| PyTorch | `2.9.1` |
| Transformers | `4.57.3` |
| Datasets | `4.1.1` |
| Accelerate | `1.12.0` |
| PEFT | `0.18.0` |
| TRL | `0.29.1` |
| ms-swift | `4.5.0.dev0` |
| ms-swift commit | `565a1ad586a21d24b23931c52d2c62b49c39bee8` |
| MPS | built `true`; available to the current process `false` |
| Direct dependency contract SHA-256 | `b5a41b3111dcce0a0ddc8e1cf1e8dac3abd7cf1a2e16835948dbe276af77d7e3` |
| Resolved pip snapshot SHA-256 | `40aac26b5f4672db9d1415b816148107dc05563db116d6b5c53bcfabd8336e12` |

## Verification evidence

- `swift sft --help`: passed.
- `swift infer --help`: passed.
- `Seq2SeqTrainer`, `DPOTrainer`, `GRPOTrainer`, and `GKDTrainer` imports: passed.
- Day 09 unit tests in this environment: `44/44` passed.
- Tiny causal-LM CPU lifecycle: forward, backward, gradient accumulation,
  clipping, two optimizer steps, checkpoint and exact resume all passed.
- RFC 8785 implementation: `rfc8785==0.1.4` installed for Day 10 semantic
  protocol hashes.

## Known exception

`python -m pip check` reports only `ms-swift requires cpm-kernels`. This is an
intentional macOS exception: `cpm-kernels` is CUDA-specific and belongs to the
future Linux/H100 overlay. The environment must not be described as a CUDA/TP
runtime.

The current Day 10 Base baseline therefore defaults to CPU execution. A later
GPU evaluation must create a different `execution_protocol_hash`; for strict
Base/SFT comparison, Base must be rerun under that same GPU execution protocol.
