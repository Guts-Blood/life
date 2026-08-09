# Day 27 — 周末 Reading：GRPO 与 On-policy 边界

日期：`2026-08-22`  
状态：`not_started`  
强度：1 小时，仅阅读/复盘

## 主要目标

用 Day 25 的 Qwen3.5 v2 trajectories 和 Day 26 已验证的 slime objects，解释何时数据仍是 on-policy，何时需要拒绝、重采样或 importance correction。Day 01–12 的 v1 trajectories/checkpoints 只保留为历史证据，不进入本日 active policy timeline。

## 理论与复盘（60 分钟）

精读清单：[Day 27 — GRPO/on-policy](../SCALING-BOOK-READING-GUIDE.md#day-27)。

- 25 分钟：重读 [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) 的 objective/method，只看 group advantage、policy ratio 和 KL。
- 20 分钟：从 Day 25 Qwen3.5 coding GRPO 抽一组 completions，标出 reward、advantage、rollout/old/current/ref log-prob 和 Day 21 SFT parent。
- 15 分钟：只有 Day 26 输出 `go_day29` 时，才对其 runtime-verified weight-version timeline 回答：
  1. rollout 由哪个 policy version 生成？
  2. 训练时 current policy 是否已变化？
  3. generation、buffer、training、weight sync 的哪些延迟会产生 stale data？

写出一个同步 baseline 和一个 async/stale timeline。Async 是理解题，不在周末实现。若 Day 26 输出 `slime_qwen35_compatibility_blocked`，则只根据 Day 25 evidence 和官方 slime 文档画“预期 timeline”，所有未验证边必须标为 `UNKNOWN`，不得伪造 runtime 对象或命令。

## Coding

无。

## 训练 / 实验

无；不启动 ms-swift、slime、SGLang 或 GPU。

## 资源与租卡

CPU only；严格 60 分钟。

## 验收

- [ ] 能区分 rollout/old/current/reference 四个 policy/log-prob 角色。
- [ ] 能指出 stale data 首次进入的位置。
- [ ] 不把 importance correction 说成可以无限修复任意陈旧 trajectory。
- [ ] active timeline 只引用 Qwen3.5 v2 parent/trajectory/version，v1 仅作为历史索引。

### 三个答案

1. 
2. 
3. 
