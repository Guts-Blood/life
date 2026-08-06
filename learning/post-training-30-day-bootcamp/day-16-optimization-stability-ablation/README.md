# Day 16 — Optimizer、LR、Warmup、Batch 与梯度稳定性

日期：`2026-08-11`
状态：`not_started`
强度：4–5 小时

## 主要目标

用受控 ablation 理解 AdamW 更新、learning-rate schedule、warmup、effective batch 和 gradient clipping 如何共同决定稳定性；不是寻找一次性的“最优超参”。

## 理论（75 分钟）

精读清单：[Day 16 — Optimization stability](../SCALING-BOOK-READING-GUIDE.md#day-16)。

- 手写 AdamW 的一阶/二阶矩、bias correction、decoupled weight decay 和参数更新式。
- `global batch = micro batch × accumulation × data-parallel world size`；sequence 数和 label-token 数分别如何进入 batch 口径。
- Warmup 解决的早期更新风险；constant、cosine/linear decay 的区别。
- Global grad norm、clip 前/后 norm、update-to-weight ratio、loss scale/overflow。
- 固定 step、固定样本和固定 label-token budget 会回答不同问题。

## Coding（75 分钟）

- 统一 launcher，保存完整 optimizer/scheduler config，禁止手工覆盖未记录参数。
- 每 step 记录 LR、loss、有效 label tokens、global grad norm、clip coefficient、overflow/skip、step time 和 peak memory。
- 实现 `batch/accumulation/world-size -> sequences/tokens per update` 计算与断言。
- 生成参数更新摘要：选定层的 weight norm、update norm 和非有限值计数。

## 训练 / 实验（150 分钟）

使用 Day 15 的唯一推荐 packing/length，固定 model、data order、seed 和总 label-token budget：

| Run | optimizer | LR | warmup | effective batch | 目的 |
|---|---|---:|---:|---:|---|
| A | AdamW 固定 β/ε/WD | `η` | 5% | `B` | baseline |
| B | 同 A | `2η` | 5% | `B` | LR 敏感性 |
| C | 同 A | `η` | 0 | `B` | warmup |
| D | 同 A | `η` | 5% | `2B` | batch/噪声 |

每组先过 10-step safety gate，再跑到相同 label-token budget。若 B 出现非有限值或 grad norm 持续越界，立即停止并保留现场。Optimizer 类型替换、LR scaling rule 和 checkpointing/Flash Attention 只作为 Stretch，不能与 Core 混成组合爆炸。

## 资源与租卡

- 1×H100 80GB，预计 4–6 小时。
- 使用同一卡、同镜像和同数据顺序；A 完成前不启动 B–D。
- 发现 NaN/Inf、连续 step skip 或异常更新比时停止该 run，不用降低 LR 覆盖原证据。

## Evidence-first 产物

- `../artifacts/configs/day16-optimization/`
- `../artifacts/logs/day16-step-metrics.jsonl`
- `../artifacts/reports/day16-stability-ablation.md`

## 验收

- [ ] 能从公式解释 AdamW state、weight decay 和 warmup，而非只会调参数。
- [ ] 四组使用相同数据顺序与 label-token budget。
- [ ] 报告包含 LR、grad norm、clip、update/weight、loss 和吞吐时间线。
- [ ] 给出“稳定默认值”和至少两个不应外推的边界。

## Optional Capstone Handoff

输出 optimizer/scheduler/batch assertion 的 resolved default 与稳定边界。Capstone 可把它作为 first smoke config，但 8B/4B full-parameter run 必须重新做 short LR/grad/update safety gate；不得把小模型的最佳 LR、microbatch 或 warmup 当作 scale-invariant 结论。

## Daily Log

### 预注册假设

### 稳定性结果

### 反直觉现象

### Day 17 第一动作
