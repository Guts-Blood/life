# Day 28 — 周末 Reading：slime Debug、Replay、Repro 与 Observability

日期：`2026-08-23`  
状态：`not_started`  
强度：1 小时，仅阅读/复盘

## 主要目标

把 Day 29 的调试顺序冻结：先验证 trajectory/reward，再 train-only replay，最后验证 weight sync 和下一轮 policy version。所有 runtime 名称、参数和文件路径只允许来自 Day 26 对 Qwen3.5 的兼容性验证。

## 理论与复盘（60 分钟）

精读清单：[Day 28 — slime debug/replay/observability](../SCALING-BOOK-READING-GUIDE.md#day-28)。

- 15 分钟：若 Day 26 为 `go_day29`，读其冻结 release/tag/SHA 的 Debug 文档，确认 rollout-only、train-only 和 debug dump 的实际参数；若 blocked，只列文档候选与缺失证据。
- 15 分钟：读 Trace/Profiling 文档，列出 sample、reward、latency、actor/GPU 和 weight-sync 可观测点。
- 15 分钟：读 Reproducibility/Fault-tolerance 文档，区分框架能恢复的 server failure 与完整 job/preemption resume。
- 15 分钟：完成 Day 29 gate 表。

| Gate | 输入 | 必须看到的证据 | 失败时下一步 |
|---|---|---|---|
| rollout-only | | | |
| reward replay | | | |
| train-only replay | | | |
| weight sync | | | |
| next-version rollout | | | |

参数和文件名只引用 Day 26 runtime-verified checkout，不使用 main 分支截图，也不把 `v0.3.0` 历史阅读基线自动升级为 active runtime。若 Day 26 blocked，完成的表必须保留 `BLOCKED/UNKNOWN`，Day 29 不得换模型或换 recipe 继续。

## Coding

无。

## 训练 / 实验

无；不启动 GPU，不在周末临时修环境。

## 资源与租卡

CPU only；严格 60 分钟。

## 验收

- [ ] Day 29 每个 gate 都有输入、证据、停止条件和下一步。
- [ ] 知道哪些状态可 replay，哪些 nondeterminism 仍可能存在。
- [ ] 知道 slime server restart 不等于完整训练 job resume。
- [ ] Day 29 的入口结论与 Day 26 唯一 go/no-go 一致，没有 v1 或其他模型 fallback。

### Day 29 gate 表 / 剩余 UNKNOWN
