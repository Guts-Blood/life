# Day 18 — Megatron × Qwen3.5 Compatibility and Learnability

计划日期：`2026-08-13`；独立提前执行：`2026-08-08`
状态：`closed_pass`（standalone pull-forward；`2026-08-10` 完成状态收尾）
强度：4–5 小时人工工作；双卡窗口由 compatibility preflight 决定

Close 结论：C0–C5 原协议全部通过，本地 evidence/PASS inventory 与 16 项测试重新验证通过，不补跑 GPU；详见 [`Day 18 Close`](../artifacts/reports/day18-close.md) 和机器可读 [`day18-closeout.json`](../artifacts/reports/day18-closeout.json)。

## 主要目标

把 Megatron 学习落到 active model：验证 pinned ms-swift/Megatron stack 能否正确解析 `Qwen/Qwen3.5-4B-Base` 的 conditional-generation wrapper、GatedDeltaNet（GDN）、text-only/vision freeze policy、batch/loss、TP/DP state 与 distributed checkpoint。C0–C4 是 compatibility/correctness gate；C5 追加同入口 150-step tiny-overfit learnability gate，但不外推泛化或 scaling speedup。Day 01–12 的旧 Megatron/模型结果保留为历史背景，不能代替 Qwen3.5 evidence。

## 版本与支持 Gate（45 分钟）

- 本次作为 standalone pull-forward，使用本次运行独立冻结并记录的 runtime；Day 15 runtime lock 仍未完成。对照 [Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md) 与 pinned checkout 的 examples/`--help`，记录 ms-swift、Megatron Core、mcore-bridge、Transformers、CUDA/NCCL 和 GDN implementation。
- 用 pinned code 搜索并记录 model conversion/provider、conditional loader、processor/template、GDN、loss、optimizer、distributed save/load 的实际 `file:function`。
- 验证 TP degree 对 attention/KV/GDN layout 的约束；未经运行验证的组合标成 `UNKNOWN`，不凭文档示例宣称支持。
- 明确 text-only sample path 不训练或插入视觉 token；ViT/aligner freeze 与参数 ownership 要由 runtime inventory 证明。

## 最小源码链与插桩（75 分钟）

```text
HF Base revision + processor/template
  -> conversion/model provider
  -> Qwen3.5 conditional wrapper + language model + GDN
  -> text-only batch / labels / loss
  -> backward / optimizer step
  -> distributed save/load
  -> HF/inference export + parity
```

每个节点记录 `file:function`、输入/输出 shape、运行 ranks、拥有的 state 和 evidence path。插桩记录 rank、TP/DP group、trainable/frozen params、batch/label shape、loss、optimizer step、GDN backend、checkpoint save/load 与 export。

## Compatibility Gates（120–150 分钟）

严格按顺序执行：

1. **C0 config/conversion**：exact Base revision 可转换或由官方 bridge 加载；参数数、关键 tensor names 与 hash inventory 可解释。
2. **C1 single-rank parity**：对 Day 18 独立冻结的两行 text-only fixture 比较 Transformers reference 与 Megatron path 的 logits/loss；预先写 tolerance。
3. **C2 DP smoke**：同机 2 ranks，`TP=1, DP=2`，完成 3–5 optimizer steps并保存 per-rank state。
4. **C3 TP smoke**：只在 pinned stack 正式支持时运行 `TP=2, DP=1` 的 3–5 steps；保持 global label-token budget 与 C2 一致。
5. **C4 distributed checkpoint**：退出进程、reload、继续 2 steps，并做 inference export/reload parity。
6. **C5 tiny overfit**：从 C0 Base 独立启动同一 `megatron sft`/TP2 recipe，固定 150 updates；fresh-process HF export/reload 后，teacher-forced token accuracy ≥95%、loss 相对 Base 至少下降 80%，且 main/MTP changed、vision/aligner bitwise unchanged。

任一 gate 失败即保存最小 reproducer、resolved config 和错误证据，状态写 `compatibility_blocked_at_Cx`。不得退回 synthetic GPT 或另一 Qwen 型号后把结果记作 Qwen3.5 通过；synthetic recipe只能用于区分环境损坏与 model-adapter 问题。

## 资源与租卡

- 先完成本地 CPU/config gate；租用同机 2×H800 80GB 或 2×H100 80GB（均为 Hopper CC 9.0）后，C0/C1 只暴露 GPU 0，二者通过后才让 C2–C5 使用两张卡。
- 不跨节点、不做四卡、不做长训练。双卡窗口默认 2–4 小时，实际以 smoke 预算和停止条件为准。
- import/conversion/topology gate 未通过时不开 optimizer run；证据同步后立即关机。

## 实际结果（standalone pull-forward）

- Run `day18-qwen35-20260808T073811Z`：C0–C5 与 fail-closed final gate 全部通过；完整数值、版本、问题记录和 evidence SHA 见 [兼容性报告](../artifacts/reports/day18-qwen35-megatron-compatibility.md)。
- 同一 Megatron SFT 入口覆盖 HF↔MCore parity、`TP=1/DP=2`、`TP=2/DP=1`、fresh-process full-state resume/export，以及从 Base 独立启动的 150-step two-row tiny overfit；vision/aligner ownership 保持冻结。
- 结论只限于此次冻结的 Qwen3.5-4B Base、两行 text-only fixture、BF16、2×H800、TP/DP/MTP/checkpoint/export 路径：未检出训练代码 bug 且能完成同 fixture overfit；不证明泛化、长训、视觉路径、任意拓扑、exact-resume 等价或“全仓无 bug”。
- 在本次 standalone run 收束时，Day 15 仍为 `not_started`，本结果当时不单独补齐 M1–M5 或解锁 Day 16/S1。`2026-08-09` 的后续决定将 Day 18–20 合并证据用于 [`Day 15 Close`](../artifacts/reports/day15-close.md)；该关闭仍不补造原 M2–M5 artifacts，也不解锁 Day 16/17 或 S1。

## Evidence-first 产物

- `../artifacts/reports/day18-qwen35-megatron-compatibility.md`
- `../artifacts/configs/day18-qwen35-megatron/`
- C0–C5 per-rank logs、parameter/layout manifest 与 C5 teacher-forced Base/final evidence
- `problems.jsonl`、fail-closed final summary 与 immutable `DAY18-PASS.json`
- distributed checkpoint、HF export 与 reload parity evidence
- 本地准备、上传与逐 Gate 命令见 [AUTODL-RUNBOOK.md](AUTODL-RUNBOOK.md)；脚本准备完成不等于 GPU compatibility 已通过。

## 验收

- [x] conditional loader、processor/template、GDN 和 conversion 每条边有 pinned `file:function` 与 runtime evidence。
- [x] C1 reference parity 有预注册 tolerance 和逐 tensor/loss 结果。
- [x] C2 完成真实 optimizer step；C3 若不支持则有明确官方/运行证据，而非静默跳过。
- [x] distributed checkpoint 在新进程恢复并完成 export/reload parity，或准确记录阻塞 gate。
- [x] C5 同 fixture learnability 阈值、参数 ownership 与 fresh export/reload gate 全部通过；结果未被写成泛化或 exact-resume 证明。
- [x] synthetic control 未被冒充为 active-model success。

## Daily Log

### Pinned environment / GDN backend

2×H800 PCIe、Python 3.12.13、PyTorch 2.10.0+cu126、Megatron Core 0.18.0、mcore-bridge 1.6.0；`USE_MCORE_GDN=1`，实际类为 `mcore_bridge.model.modules.gated_delta_net.GatedDeltaNet`。

### Conditional loader / conversion chain

8-node runtime codepath manifest 通过；见兼容性报告与 evidence bundle 中的 `codepath-runtime-evidence.json`。

### C0–C5 gate results

10/10 required gates 通过；C5 fresh HF reload 达到 72/72 teacher-forced tokens，loss `3.31137e-08`。

### DP/TP ownership and parity

C2 `TP1/DP2` 与 C3 `TP2/DP1` 均完成 3 个成功 update；first-loss difference `0.00330782`。

### Compatibility blocker or supported envelope

无开放 blocker。支持范围和 17 类已解决/观察问题见 [最终报告](../artifacts/reports/day18-qwen35-megatron-compatibility.md)。
