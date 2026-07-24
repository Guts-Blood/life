# Day 10 — Frozen Eval 与 Base Baseline

日期：`2026-08-05`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

在主 SFT 之前冻结一套 task eval，并保存 Qwen3-1.7B-Base 的逐样本结果。

## 理论（60 分钟）

精读清单：[Day 10 — Eval generation 与 inference](../SCALING-BOOK-READING-GUIDE.md#day-10)。读 Part 7 的 prefill/generation、offline throughput 与 linear bottleneck，再接 Eval Guidebook。

- Train/validation/dev/frozen test 的职责。
- Leakage、selection bias、slice、deterministic vs sampling eval。
- 先定义 success metric 再训练。

## Coding（120 分钟）

- 完成 `run_generation.py`：sample ID、category、prompt hash、model revision、output、latency。
- 建立 150–300 条 frozen eval manifest、hash 和 5+ slices。
- 检查 exact/near duplicate 与 train overlap。

## 训练 / 实验（90–120 分钟）

- 不训练。
- 用 `Qwen/Qwen3-1.7B-Base` 生成 deterministic baseline。
- 人工审查至少 30 条，记录 error taxonomy 初版。

## 资源与租卡

- 推荐：1×H100 80GB，预计 2–4 小时。
- 开卡前 CPU 完成数据、manifest 和 5-sample dry run。
- Predictions 落盘并同步后关机，不在 GPU 上做标注分析。

## 验收

- [ ] Frozen eval 有版本/hash，后续不可静默修改。
- [ ] 同命令重复运行的 deterministic 子集一致。
- [ ] 保存逐样本预测，不只保存平均分。
- [ ] `day10-baseline.md` 说明覆盖和盲区。

## Daily Log

### Eval slices

### Base 主要失败

### Day 11 第一动作
