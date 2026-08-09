# Day 18 — Qwen3.5 × Megatron Compatibility and Learnability Report

Status: `done_standalone_day18_day15_not_completed`

Run ID: `day18-qwen35-20260808T073811Z`
Gated run window: `2026-08-08 15:38–17:02 CST`
Active model: `Qwen/Qwen3.5-4B-Base`
Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`
ms-swift source: `565a1ad586a21d24b23931c52d2c62b49c39bee8` (`4.5.0.dev0`)

## Verdict

Day 18 的 C0–C5 全部通过。Fail-closed finalizer 验证了 10 个 required gates、8 个带 runtime source 的 codepath nodes、65 个 hashed evidence entries，并确认 17 类问题均已 resolved 或记录为 observation，`open_problem_count=0`。

这次结果支持以下有边界的结论：在固定 revision、固定两条 text-only fixture、BF16、同机 2×H800、Megatron TP/DP、MCore GDN、MTP、distributed checkpoint 和 HF export/reload 的冻结 envelope 内，真实训练入口完成了全部 optimizer updates，能够把同一 fixture overfit 到 100% teacher-forced token accuracy；未发现该路径中的训练代码缺陷。

它不证明所有数据、视觉输入、长序列、多节点、其它并行组合、吞吐 scaling、泛化或全局“无 Bug”。C4 证明 fresh-process full-state continuation，但没有 uninterrupted 5-step 数值对照，因此不声称 exact-resume equivalence。Day 15 仍是 `not_started`；本次 standalone run 不回填 Day 15 M1–M5，也不产生可晋级的 S1 checkpoint。

## Frozen runtime envelope

- Hardware: same-host `2×NVIDIA H800 PCIe`, 81,105 MiB each, Hopper CC 9.0, MIG disabled；两卡为 `SYS` cross-NUMA、无 NVLink，2-rank NCCL all-reduce/P2P preflight 通过。
- Storage: 400 GiB AutoDL data disk；完成并保留 C2/C3/C4 full-state checkpoints、C5 model-only checkpoint 和 HF exports 后仍有约 162 GiB 可用。
- Runtime: Python `3.12.13`, PyTorch `2.10.0+cu126`, CUDA runtime `12.6`, NCCL `2.27.5`。
- Core stack: Transformers `5.12.1`, Megatron Core `0.18.0`, mcore-bridge `1.6.0`, Transformer Engine `2.16.0`, flash-linear-attention `0.4.2`, flash-attn `2.8.3`, causal-conv1d `1.6.2.post1`。
- Model path: `Qwen3_5ForConditionalGeneration` + `Qwen3VLProcessor`; `USE_MCORE_GDN=1`; one MTP layer and all 15 expected `mtp.*` tensors preserved.
- Fixture: exactly 2 rows, batch shape `[2, 80]`, 72 supervised tokens, no visual tensors/tokens; ViT and aligner frozen.

## C0–C5 results

| Gate | Result | Key runtime evidence |
|---|---|---|
| C0 HF → MCore conversion | `pass` | 738 HF tensor mappings accounted for; all 15 MTP tensors survived exact HF→MCore→HF round-trip. |
| C1 single-rank logits/loss parity | `pass` | 2/2 rows had 0 loss-token argmax mismatch; max loss-token logit difference `0.0122156`; HF loss `2.40772414` vs MCore loss `2.40289942`, absolute difference `0.00482472`. |
| C2 `TP=1, DP=2` | `pass` | 3/3 successful optimizer updates; main loss `2.40337467 → 0.75684273 → 0.26305130`; MTP loss `2.62292004 → 0.88203031 → 0.45968446`; peak reserved `52.34 GiB/GPU`; 61.24 GB full-state checkpoint and fresh-process model/Adam/RNG loadcheck passed. |
| C3 `TP=2, DP=1` | `pass` | 3/3 successful optimizer updates; main loss `2.40006685 → 0.76450086 → 0.25180891`; C2/C3 first-loss difference `0.00330782`; peak reserved `38.91 GiB/GPU`; full-state checkpoint passed. |
| C4 fresh resume `3→5` + export | `pass` | Fresh process loaded model/optimizer/RNG and completed updates 4–5; loss `0.02164458 → 0.00051258`, MTP loss `0.05578668 → 0.00202528`; peak reserved `53.79 GiB/GPU`; export/reload had 0 argmax mismatch and max difference `0.00941098`; HF teacher-forced loss `6.1829e-05`, accuracy `1.0`. |
| C5 150-step tiny overfit | `pass` | Same `megatron sft` TP2 path completed 150/150 verified successful updates; logged loss `2.40006685 → 2e-08`; fresh HF evaluation loss `2.40772414 → 3.31137e-08`, accuracy `0.541667 → 1.0`, relative loss reduction `0.999999986`; peak reserved `39.36 GiB/GPU`. |

C5 ownership audit compared the final HF export against the C1 conversion round-trip baseline: 347/426 main-language-model tensors and 11/15 MTP tensors changed at BF16 resolution; all 297 visual/aligner tensors were bitwise unchanged; no tensor was added, missing, or shape/dtype mismatched. The final model-only checkpoint contained model/language/MTP/visual state and correctly omitted optimizer/RNG state.

## Runtime codepath evidence

The generated `codepath-runtime-evidence.json` records pinned source SHA, runtime class/source, shapes, ranks, state ownership and evidence paths for all eight nodes:

| Node | Pinned runtime edge |
|---|---|
| Conditional loader | `swift/model/models/qwen.py:Qwen3_5Loader.get_model` |
| Processor/template | `swift/template/templates/qwen.py:Qwen3_5Template._swift_prepare_inputs` |
| Conversion/provider | `mcore_bridge/model/gpts/qwen3_next_gdn.py:Qwen3NextLoader.build_model` |
| GDN | `mcore_bridge/model/modules/gated_delta_net.py:GatedDeltaNet.forward` |
| Batch/labels/loss | `swift/megatron/trainers/trainer.py:MegatronTrainer.forward_step` |
| Backward/optimizer | `swift/megatron/trainers/base.py:BaseMegatronTrainer.train_step` |
| Distributed save/load | `swift/megatron/utils/megatron_lm_utils.py:save_mcore_checkpoint` plus recorded load path |
| HF export | `swift/megatron/pipelines/export/export.py:MegatronExport.convert_mcore2hf` |

## Problems encountered and disposition

The immutable `problems.jsonl` contains 24 lifecycle records for 17 unique issues. The material differences from the original plan were:

| Area | Observed problem/difference | Disposition |
|---|---|---|
| Hardware | Planned H100 was unavailable; actual pair was 2×H800 PCIe. | Accepted as a Hopper CC 9.0 hardware amendment; all numerical gates remained unchanged and passed. |
| Topology | The allocated pair was `SYS` cross-NUMA with no NVLink. | Recorded as an observation; NCCL/P2P smoke and real TP2 training passed. |
| Boot image | AutoDL booted PyTorch 2.5.1/CUDA 12.4 rather than the frozen runtime. | Built and audited an isolated Python 3.12/PyTorch 2.10/cu126 runtime; boot image was not misrepresented as the training runtime. |
| Transfer | FileZilla stripped 14 executable bits and used a different model directory layout. | Restored Git-declared modes, verified source integrity, and pinned the actual hash-verified model path. |
| Dependency acquisition | NVIDIA conda fetch stalled; causal-conv GitHub HTTP/2 clone failed; no exact flash-attn wheel existed. | Used hash-verified exact packages/source, built for SM90, and passed CUDA forward/backward kernel smokes. |
| CUDA extension builds | Transformer Engine initially lacked CRT headers; causal-conv initially lacked `cicc`. | Added the exact CUDA 12.6 development components and rebuilt successfully. |
| Text-only dependency | `decord 0.6.0` carried an incompatible CPython 3.6 wheel tag. | Removed from this text-only runtime; no video path was exercised. |
| Norm backend | Apex was absent. | Recorded Megatron Core's supported Torch Norm fallback as the tested backend. |
| Local model loading | Local path ambiguously matched model types; template initially lacked the concrete model. | Pinned `qwen3_5` explicitly and attached the real model; clean HF reference passed. |
| Argument parser | Transformers 5.12.1 attempted `isinstance` on `typing.List[str]` for `cp_comm_type`. | Added a version/field/type-guarded local shim; no vendor package was modified. |
| Parity probe | Template was not in train mode, so labels were absent; repeated row-two inventory hit an autograd restriction. | Restored train-mode collation and retained one complete row-independent inventory; both row forwards passed. |
| Checkpoint auditor | Optional `planner_data=None` and MCore `language_model.*`/`visual.*` FQNs caused a false negative. | Fixed the auditor to use authoritative DCP metadata and DCP-only model/optimizer/RNG gates; the existing checkpoint then passed metadata audit and fresh loadcheck without retraining. |

## Evidence and integrity

- Local evidence bundle: `tmp/day18-qwen35-results/day18-qwen35-20260808T073811Z-evidence.tar`
- Bundle SHA-256: `2afc7ad1b39bdde3aad40832ceb78ab1e01116ee4f7bbc19ccf85beb4570c9c0`
- Remote canonical run: `/root/autodl-tmp/runs/day18-qwen35-20260808T073811Z/`
- Final verdict: `DAY18-PASS.json` with `status=day18_pass`, 10 required gates, 65 hashed evidence entries and zero open problems.
- Evidence includes preregistered contract/thresholds, runtime/topology inventories, per-rank step records, all gate summaries, source-chain manifest, problems ledger and logs. Full model/checkpoint payloads remain on the AutoDL data disk and are not duplicated inside the 6.9 MiB evidence bundle.

## Decision

`Qwen/Qwen3.5-4B-Base@1001bb4d…` is compatible and learnable inside the exact Day 18 envelope above. The safe downstream use is as runtime/implementation evidence for later Megatron diagnostics; it is not a promoted SFT candidate, a vision-path result, a performance benchmark, an exact-resume proof, or a substitute for the pending Day 15 migration acceptance.
