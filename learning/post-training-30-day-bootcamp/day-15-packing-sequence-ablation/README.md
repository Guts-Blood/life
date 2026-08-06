# Day 15 — Packing、Length 与有效 Label Token Ablation

日期：`2026-08-10`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

把“训练跑得快”拆成可解释的量：padding 浪费、packing 正确性、有效 label token 比例、显存和 step time。今天只改变 sequence/packing，不同时调整 optimizer。

## 理论（60 分钟）

精读清单：[Day 15 — Sequence length、attention 与 packing](../SCALING-BOOK-READING-GUIDE.md#day-15)。

- Padding、packing boundary、position ID、causal/segment attention mask。
- `raw tokens`、`non-padding tokens`、`label tokens` 与 `effective label tokens`。
- Attention 计算和 activation 随 sequence length 的变化。
- 为什么训练 loss 的分母、梯度累积的 token 数与吞吐口径必须对齐。

必须先写预测：在哪种长度分布下 packing 收益最大；什么错误会让 tokens/s 上升但训练信号变差。

## Coding（90 分钟）

- 写/完善长度审计：prompt、response、rendered、non-padding、label token 的直方图和分位数。
- 为 batch 输出 padding ratio、packing utilization、label-token ratio、effective label tokens/s。
- 抽样 decode packed batch，并检查边界两侧的 labels、position IDs 和 attention 可见性。
- 写至少三个失败测试：跨样本 attention 泄漏、边界 token 被错误计 loss、空 label sample。

## 训练 / 实验（120–150 分钟）

固定 model revision、dataset manifest、seed、optimizer、global **label-token budget**，各跑 20–30 个 steady-state steps：

| Run | max length | packing | 唯一变量 |
|---|---:|---|---|
| A | 1024 | false | baseline |
| B | 1024 | true | packing |
| C | 2048 | false | max length |
| D | 2048 | true | max length + packing |

前 3–5 steps 只做 warmup，不计入吞吐。比较 raw/non-padding/label tokens/s、padding ratio、peak memory、step time、loss 和截断率；按相同 label-token budget 比 loss，不按相同步数草率比较。

## 资源与租卡

- 1×H100 80GB，预计 4–6 小时；四组使用同一实例和软件环境。
- OOM 是有效边界，记录后缩短 sequence；不要顺手再改 batch 或 checkpointing。
- configs、逐 step 指标和五个 batch audit 落盘后关机。

## Evidence-first 产物

- `../artifacts/data/day15-length-audit.json`
- `../artifacts/reports/day15-packing-ablation.md`
- 四份完整 config、逐 step metrics、五个 packed-batch decode

## 验收

- [ ] 至少五个 packed batch 通过 boundary/mask 人工检查。
- [ ] 每组同时报告 raw tokens/s 与 effective label tokens/s。
- [ ] 所有 loss 比较使用相同 label-token budget。
- [ ] 选出后续默认 max length/packing，并写清适用长度分布和反例。

## Optional Capstone Handoff

输出可继承的 length/packing default、适用长度分布和 batch-audit tests。它们只是 8B/4B 的起始假设；Day 32–33 必须用新模型和 TP topology 重测 memory/throughput/correctness，不能直接迁移最大长度或 microbatch。

## Daily Log

### 预注册假设

### 四组结果

### 推荐配置与边界

### Day 16 第一动作
