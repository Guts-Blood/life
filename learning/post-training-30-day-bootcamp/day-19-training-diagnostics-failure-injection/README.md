# Day 19 — Training Diagnostics 与 Failure Injection

日期：`2026-08-14`
状态：`not_started`
强度：4–5 小时

## 主要目标

建立从 symptom 到 root cause 的训练诊断方法。通过高 LR、错误 loss mask/数据分布和吞吐退化的受控故障，证明指标、样本审计和 profiler 能区分不同问题。

## 理论（60 分钟）

精读清单：[Day 19 — Training failure diagnosis](../SCALING-BOOK-READING-GUIDE.md#day-19)。

- Optimization failure：loss spike、grad norm、clip saturation、NaN/Inf、update ratio。
- Data/label failure：label-token ratio、样本 decode、slice loss、重复/分布漂移。
- Systems failure：GPU utilization、data wait、step time、OOM、communication/idle gap。
- 相关性不等于归因；每次只注入一个故障并保留正确 baseline。

## Coding（90 分钟）

- 建立统一 diagnostic bundle：config diff、step metrics、五个 decoded samples、data histogram、GPU/memory/throughput。
- 实现 `baseline -> injection -> symptom -> probe -> root cause -> fix -> rerun` 记录模板。
- 为 mask 检查、数据分布 drift 和非有限梯度各加一个自动断言。
- 对 baseline 和吞吐故障各抓 3–5 个 active steps 的短 profiler trace。

## 训练 / 实验（120–150 分钟）

使用 Day 16/17 的稳定小模型 baseline，每个故障只跑够出现 signature 的最少 steps：

| Run | 单一注入 | 预期主要证据 |
|---|---|---|
| A | 正确 baseline | 正常 loss/grad/data/throughput |
| B | LR 提高到预注册危险倍数 | grad/update/loss 异常 |
| C | assistant-only mask 改成错误范围，或制造 label shift | label ratio、decode、slice loss 异常 |
| D | 训练数据人为偏向单一来源/长度桶 | distribution 与 held-out slice 退化 |
| E | 关闭 packing 或注入可控 dataloader delay | step/data wait 变慢但优化指标基本稳定 |

每个注入后恢复正确配置并做 3-step rerun。不要把多个故障叠加；高 LR run 遇到 NaN/Inf 或异常 update ratio 立即停止。

## 资源与租卡

- 1×H100 80GB，预计 3–5 小时；可复用 Day 17 环境。
- 故障 run 使用独立 output/checkpoint 目录，不污染 baseline。
- profiler 只抓短窗口；证据完整后关机。

## Evidence-first 产物

- `../artifacts/reports/day19-failure-matrix.md`
- baseline 与四类 injection 的 config diff/metrics/sample audit
- 两份短 profiler traces

## 验收

- [ ] 至少完成高 LR、mask/data、distribution、throughput 四类故障。
- [ ] 每类都有预注册 signature、观察证据、最小 probe、修复与恢复验证。
- [ ] 能区分“loss 正常但数据错”和“系统慢但优化正常”。
- [ ] 输出可复用的五分钟/三十分钟诊断顺序。

## Optional Capstone Failure Backlog

将 Day 19 diagnostic bundle 扩展模板预留给：TP rank hang/shape mismatch、teacher OOM/timeout、teacher/student tokenizer or token-ID mismatch、teacher log-prob alignment、stale student rollout/weight sync 与 checkpoint conversion drift。今天不注入这些昂贵故障，只定义未来 evidence slots 和停止顺序。

## Daily Log

### Failure matrix

### 最容易误诊的症状

### 五分钟诊断顺序

### Day 20 Reading 问题
