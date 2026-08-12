# Day 19 — Qwen3.5 Optimization Stability 与 Failure Injection

日期：`2026-08-14`
状态：`blocked_no_stable_day16_17_baseline`
强度：4–5 小时

## 当前阻塞

Day 16 已以 [`no eligible candidate`](../artifacts/reports/day16-gap-audit.md) 收束，Day 17 也没有 selected/resumable config 可建立 continuity baseline。Day 20 的 probe 可作为既有 LR 与数据边界诊断证据，但不能冒充本日 Run A 的 stable baseline。

原 failure-injection 计划保留；新 SFT charter 先产生可复用 baseline 后再解锁，不为诊断故障而从一个已失败门禁的 adapter 开始扩大 GPU 实验。

## 主要目标

在 Day 16/17 的稳定 Qwen3.5 LoRA coding SFT baseline 上，用短、单变量 probes 理解 AdamW/LR/warmup/batch 对更新稳定性的影响，并通过 mask/data/system 故障验证诊断顺序。今天不寻找“最优超参”，也不把多个故障叠加；Day 01–12 的曲线只保留为历史诊断案例，不作为 v2 baseline。

## 理论（60 分钟）

精读清单：[Day 19 — Training failure diagnosis](../SCALING-BOOK-READING-GUIDE.md#day-19)。

- AdamW moments、bias correction、decoupled weight decay、warmup 与 adapter update/weight ratio。
- `global batch = microbatch × accumulation × DP world size`；sequence 数与 supervised-token 数的不同口径。
- Optimization failure：loss spike、grad norm、clip saturation、NaN/Inf、update ratio。
- Data/label failure：processor/template、assistant span、label-token ratio、source/length drift。
- Systems failure：GPU utilization、data wait、GDN/kernel backend、step time、OOM、communication/idle gap。

## Coding（75 分钟）

- 建立 v2 diagnostic bundle：full config diff、runtime/model/processor hashes、trainable/frozen inventory、step metrics、decoded samples、data histogram、VRAM/throughput。
- 每 step 记录 LR、loss、有效 label tokens、global grad norm、clip coefficient、overflow/skip、adapter update/weight、step/data wait 和 peak memory。
- 断言 batch/accumulation/world-size、assistant mask、text-only modality、ViT/aligner freeze 与 GDN backend 未静默变化。
- 对 baseline 和 throughput fault 各抓 3–5 active steps 的短 profiler trace。

## 短实验（150 分钟）

所有 runs 使用 exact Qwen3.5 Base/adapter lineage、同一数据顺序和独立 output 目录；每条只跑到出现 signature 的最少 steps：

| Run | 单一变量/注入 | 预期主要证据 |
|---|---|---|
| A | Day 16 stable baseline | 正常 LR/loss/grad/update/data/throughput |
| B | LR 提高到预注册危险倍数 | grad、clip、update/loss 异常；触发 stop 即结束 |
| C | warmup=0，其他同 A | early-step update 与稳定性差异 |
| D | assistant-only mask 改错或制造 label shift | label ratio、decode、slice loss 异常 |
| E | coding source/length 分布偏置 | distribution 与 v2 dev slices 退化 |
| F | 关闭已采用 packing 或注入 dataloader delay | step/data wait 变化但优化指标大致稳定 |

A 完成前不启动 B–F。每个故障后恢复正确配置做 2–3 step rerun。NaN/Inf、异常 update ratio、持续 step skip 或显存越过 Day 15 hard cap 时立即停止；不用临时降 LR/offload 覆盖现场。

## 资源与租卡

- 使用 Day 16 同一 topology、镜像、processor/template、worker 数与 GDN backend；预计 3–5 小时。
- Failure runs 独立保存，不能污染 provisional SFT anchor 或未来 Day 20 Optional R 使用的正确 resume checkpoint。
- profiler 只抓短窗口；证据完整后关机。

## Evidence-first 产物

- `../artifacts/reports/day19-qwen35-optimization-failure-matrix.md`
- A–F config diffs、step metrics、sample/data audits
- 两份短 profiler traces 与恢复 rerun evidence

## 验收

- [ ] 能从 optimizer state 和 update/weight 解释 LR/warmup 现象，而非只看 train loss。
- [ ] 高 LR、mask、distribution、throughput 至少四类故障有单变量证据。
- [ ] 每类都有预注册 signature、最小 probe、停止条件、修复与恢复验证。
- [ ] 能区分“loss 正常但 processor/data 错”和“系统慢但优化正常”。
- [ ] 输出可复用的五分钟/三十分钟诊断顺序。

## Daily Log

### Baseline optimizer/update state

### Failure matrix

### 最容易误诊的症状

### 五分钟诊断顺序

### Day 20 Reading 问题
