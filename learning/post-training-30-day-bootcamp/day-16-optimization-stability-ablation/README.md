# Day 16 — Qwen3.5-4B Controlled LoRA Coding SFT 与 Anchor Selection

日期：`2026-08-11`
状态：`not_started`
强度：4–5 小时

## 主要目标

从 Day 15 冻结的 `Qwen/Qwen3.5-4B-Base` v2 lineage 运行一次受控、可审计的 **text-only coding LoRA SFT**，只做有限的 packing correctness/throughput parity，保存 early/mid/final candidates，并按预注册 dev policy 选择一个可供 Day 17、23、25 使用的 SFT anchor。

如果 Day 15 的 revision、processor/template、Base baseline、tiny-overfit/resume 或显存 gate 任一未通过，今天不启动正式 SFT。Day 01–12 的 0.6B checkpoints 不是本日权重起点。

## 理论（60 分钟）

- LoRA adapter、冻结 Base/ViT/aligner、optimizer state 与 merged inference export 分别保存什么。
- Coding SFT 的 assistant-token objective、有效 label-token budget 与 sequence/example batch 口径。
- Packing 只能在 boundary/mask/position correctness 通过后讨论吞吐；Qwen3.5 支持情况以 Day 15 pinned stack 的实际文档、`--help` 和 smoke 为准。
- SFT anchor 是通过 dev metric 与 guardrails 晋级的 checkpoint，不是 final step、最低 train loss 或“能加载”的同义词。

## Coding（75 分钟）

- 从 Day 15 v2 manifest 生成一份 coding-focused train/dev plan，保存 raw IDs、render/token/label hashes 与 supervised-token budget。
- 用一份 base config 加最小 overrides；自动 diff 并拒绝 model revision、processor/template、LoRA target、data order、optimizer、seed、dtype 或 hardware 的意外漂移。
- 每 step 记录 sample IDs、有效 label tokens、loss、LR、grad norm、clip、adapter update/weight ratio、tokens/s、peak memory 和 trainable/frozen parameter inventory。
- 为 unpacked/packed batch 写至少五个 decode/boundary audits；若 pinned stack 不支持正确 packing，直接冻结 `packing=false`，不自写未经验证的替代实现。
- 定义 SFT promotion manifest：Base hash、adapter/resumable checkpoint、merged inference export、dev evidence、processor/template/data/config hashes 与转换 parity。

## 实验 A：有限 Packing Parity（30–45 分钟 GPU）

仅在 Day 15 packing contract tests 通过时运行两条 5–10 steady-state-step probe：

| Run | Packing | 其他条件 | 目的 |
|---|---|---|---|
| P0 | false | frozen length/microbatch/label-token budget | correctness baseline |
| P1 | true | 与 P0 相同 | boundary correctness 与有效 label tokens/s |

比较 decoded boundaries、labels/position/visibility、loss、有效 label tokens/s、step time 和 peak memory。任何 correctness 差异都使 P1 作废；今天不扩展到四组 length×packing 网格。

## 实验 B：Controlled LoRA Coding SFT（120–150 分钟 GPU）

- Parent：Day 15 冻结的 exact Qwen3.5 Base revision；只加载 v2 数据与 processor/template。
- Scope：LoRA 仅覆盖 Day 15 验证的 language modules；ViT/aligner 保持冻结；记录实际 trainable parameter names/count。
- 使用 parity 选出的 packing setting、固定 seed/data order、optimizer/schedule、global supervised-token budget 与 hardware。
- 先过 5-step safety gate；再运行预注册的短 budget，按累计 supervised tokens 保存 25%/60%/100% candidates。
- 每个 candidate 使用同一 v2 dev suite，报告 coding primary metric、general/math/format guardrails、长度/截断和逐样本 bad cases。
- 在看结果前冻结 eligibility、tie 与 `inconclusive` 条件；新 v2 confirmation 继续封存，旧 v1 frozen test 保持只读。
- 将满足条件的 candidate 写为 **provisional SFT anchor**；Day 21 只做 v2 candidate audit/final promotion，不引入 v1 checkpoint。

若没有 candidate 合格，结论必须是 `no_eligible_qwen35_sft_anchor`；Day 23/25 随之 blocked，不能退回 0.6B 或直接从 Base 做 DPO/GRPO。

## 资源与租卡

- 使用 Day 15 实测通过的单卡/多卡 topology；默认 planning 上限仍为 1×H100 80GB LoRA，不宣称这是最低显存。
- 任何 topology、offload、quantization 或 checkpointing 变化都先成为新 preflight，不能在正式 run 中临时加入。
- configs、adapter/resume state、merged export、dev predictions 和 promotion manifest 同步后关机。

## Evidence-first 产物

- `../artifacts/configs/day16-qwen35-lora-sft/`
- `../artifacts/data/day16-qwen35-coding-sft-manifest.json`
- `../artifacts/logs/day16-qwen35-sft-step-metrics.jsonl`
- `../artifacts/reports/day16-packing-parity.md`
- `../artifacts/reports/day16-qwen35-sft-trajectory.md`
- `../artifacts/reports/day16-qwen35-provisional-anchor.json`

## 验收

- [ ] 所有 run 从 exact Qwen3.5 Base v2 lineage 启动，v1 权重未参与。
- [ ] Packing 只在 correctness parity 后采用；不支持时有明确 `packing=false` 结论。
- [ ] LoRA/ViT/aligner trainable/frozen inventory 与配置一致。
- [ ] early/mid/final 按累计 supervised tokens 保存并做同协议 v2 dev eval。
- [ ] 产生一个有完整 promotion evidence 的 provisional SFT anchor，或诚实记录 no-eligible。
- [ ] 新 v2 confirmation 未消费；旧 v1 frozen test 仍 sealed/unconsumed。

## Daily Log

### Pinned v2 parent / runtime

### Packing parity

### SFT trajectory / candidates

### Provisional anchor or blocker

### Day 17 第一动作
