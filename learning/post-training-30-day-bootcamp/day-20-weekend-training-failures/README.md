# Day 20 — 周末 Reading：Training Failure Signatures

日期：`2026-08-15`
状态：`not_started`
强度：1 小时，仅阅读/复盘

## 主要目标

把 Day 16–19 的 Qwen3.5 v2 优化、resume、兼容性与故障证据压缩成看到指标后可以迅速展开的 signature map，而不是再跑一轮实验。

## 理论与复盘（60 分钟）

精读清单：[Day 20 — Training failure signatures](../SCALING-BOOK-READING-GUIDE.md#day-20)。

- 15 分钟：重看 Day 19 A–F step metrics，只描述观察，不先看 run 名。
- 15 分钟：重看 decoded samples、processor/template hash、label-token ratio、modality/freeze inventory 和数据分布。
- 15 分钟：重看 baseline/throughput profiler 与 Day 18 compatibility evidence，区分 data wait、compute/GDN、communication 和 OOM。
- 15 分钟：完成下表，并为每类写一个“最便宜的下一步 probe”。

| Signature | 可能原因 | 区分性指标 | 最小 probe | 停止条件 |
|---|---|---|---|---|
| loss/grad spike | | | | |
| loss 平稳但能力退化 | | | | |
| loader/GDN/processor 不一致 | | | | |
| 吞吐骤降 | | | | |
| resume 后分叉 | | | | |
| 只在某个 data slice 退化 | | | | |

## Coding

无。

## 训练 / 实验

无；只能使用 Day 15–19 已保存的 v2 证据，不重跑 GPU，也不引用 v1 指标填空。

## 资源与租卡

CPU only；严格 60 分钟，不租 GPU。

## 验收

- [ ] 六类 signature 都有可证伪的替代解释。
- [ ] 每类都有低成本 probe 和明确停止条件。
- [ ] 写出一个曾经会误诊、现在能区分的例子。

### 六条 takeaway

1.
2.
3.
4.
5.
6.
