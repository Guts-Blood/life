# Day 02 — Transformer 参数、FLOPs 与显存

日期：`2026-07-28`  
状态：`done`（用户确认 Day 02 已完成；保留原清单，不补写未记录的 artifact）

强度：工作日 4–5 小时

## 今日结果

完成一个可以用于 0.6B、1.7B、4B、32B 和 35B-A3B 的模型 accounting 表。目标是看到 config 就能做一阶容量判断。

## 时间安排

- 90 分钟：读 [All the Transformer Math You Need to Know](https://jax-ml.github.io/scaling-book/transformers/)。
- 60 分钟：从 Qwen3-1.7B config 列出每个大矩阵的 shape。
- 90 分钟：手算参数、BF16 权重、Adam 状态、训练 FLOPs。
- 60 分钟：实现或起草 `memory_estimator.py`，与模型实际参数量对照。
- 20 分钟：记录误差来源。

## 理论

精读清单：[Day 02 — Transformer 参数与训练 FLOPs](../SCALING-BOOK-READING-GUIDE.md#day-02)。按 `Counting Dots -> Forward/reverse -> Transformer Accounting -> Global FLOPs` 阅读，并先做题再看答案。

今日与 Day 01 合并进行互动阅读时，使用 [`../day-01-environment-baseline/DAY-01-02-READING-QUIZ.md`](../day-01-environment-baseline/DAY-01-02-READING-QUIZ.md) 跟踪原始回答、点评和修正版结论。

- [x] Day 02 Reading Quiz：completed `2026-07-26`。

参数、activation 与 training-state 的电子版参考：

- [`../artifacts/reports/training-memory-visual-guide/dense-transformer-training-memory.svg`](../artifacts/reports/training-memory-visual-guide/dense-transformer-training-memory.svg)
- [`../artifacts/reports/training-memory-visual-guide/qwen3-30b-a3b-moe-training-memory.svg`](../artifacts/reports/training-memory-visual-guide/qwen3-30b-a3b-moe-training-memory.svg)
- [`../artifacts/reports/training-memory-visual-guide/distributed-training-memory-checkpointing.svg`](../artifacts/reports/training-memory-visual-guide/distributed-training-memory-checkpointing.svg)
- [`../artifacts/reports/training-memory-visual-guide/README.md`](../artifacts/reports/training-memory-visual-guide/README.md)

- Transformer 各矩阵的参数公式、forward/backward FLOPs、Adam 状态和 activation 来源。
- 区分 dense 的 total/active parameters 与 MoE 的 total/activated parameters。

## Coding

- 实现 `memory_estimator.py` 初版，输入 config 和训练精度，输出模型状态估算。
- 添加至少 1.7B、4B、32B 三个测试样例。

## 训练 / 实验

- 不训练。加载 config 或在 CPU/meta device 上统计参数，与手算结果对照。
- 成功标准：权重/参数估算误差小于 10%，并写明未计入项。

## 资源与租卡

- CPU only；不要租 GPU。
- 可使用 AutoDL 无卡模式读取已缓存 config，但更推荐本地完成。

## Core

- [ ] 列出 embedding、Q/K/V/O、gate/up/down、norm、lm_head 的参数公式。
- [ ] 区分 total parameters、non-embedding parameters、active parameters。
- [ ] 计算 BF16 权重：约 `2 × P` bytes。
- [ ] 计算全参训练模型状态的常见范围：约 `12–16 × P` bytes，明确假设。
- [ ] 计算 LoRA trainable parameter 数，不把它误当成总显存。
- [ ] 为 1.7B、4B、32B 生成同格式表格。

## 必须推导

```text
Q projection: hidden_size × num_attention_heads × head_dim
K/V projection: hidden_size × num_kv_heads × head_dim
O projection: q_output_size × hidden_size
MLP: gate + up + down
```

训练 FLOPs 先使用一阶近似，再说明 attention、embedding、MoE、重计算导致的偏差。不要只背 `6ND`，要知道 N、D 和近似边界。

## Stretch

- [ ] 读取模型 config 自动估算参数，而不是硬编码 1.7B。
- [ ] 用 `sum(p.numel())` 与估算对比，误差小于 10%。
- [ ] 增加 optimizer master weights 是否存在的开关。

## 产物

- `../artifacts/reports/model-accounting.md`
- `../artifacts/scripts/memory_estimator.py` 初版
- 1.7B/4B/32B 三张容量表

## Definition of Done

- 不加载权重，只看 config，能够给出模型权重和全参训练状态量级。
- 能解释为什么“32B BF16 权重约 64GB”不代表能在一张 80GB 卡上全参训练。
- 写清楚 activation 不仅取决于参数量，还取决于 layers、hidden、sequence、micro-batch 和 checkpointing。

## Daily Log

### 手算结果

### 与实际参数量的误差

### 最大误解被纠正了什么

### Day 03 第一动作
