# Day 15 — Packing 与 Sequence Length Ablation

日期：`2026-08-10`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

量化 padding、packing 和 max sequence length 对有效 token 吞吐、显存与训练正确性的影响。

## 理论（60 分钟）

精读清单：[Day 15 — Sequence length、attention 与 packing](../SCALING-BOOK-READING-GUIDE.md#day-15)。精读 Part 4 Attention、context length cost 与 KV cache，并用真实长度分布计算一次。

- Padding waste、packing boundaries、position IDs、attention mask。
- Attention activation 随 sequence length 的增长。
- Raw tokens/s 与 effective label tokens/s 的差别。

## Coding（90 分钟）

- 写 length histogram/padding ratio/label-token ratio 统计。
- 自动汇总四组 config 和 steady-state 指标。
- 为 packed batch 添加抽样 decode 与 boundary 检查。

## 训练 / 实验（120–150 分钟）

固定 model、data、seed、effective batch，各跑 20–30 steady-state steps：

| Run | max length | packing |
|---|---:|---|
| A | 1024 | false |
| B | 1024 | true |
| C | 2048 | false |
| D | 2048 | true |

比较 padding ratio、label tokens/s、peak memory、step time、loss。

## 资源与租卡

- 1×H100 80GB，预计 4–6 小时；全部 run 使用同一实例。
- 前 3–5 steps 作为 warmup，不计入 steady-state。
- 四组数据落盘后关机。

## 验收

- [ ] 人工检查至少 5 个 packed batch。
- [ ] 结论包含具体提升和数据长度分布。
- [ ] 选出后续默认 max_length/packing，并说明适用边界。

## Daily Log

### 四组结果

### 推荐配置

### Day 16 第一动作
