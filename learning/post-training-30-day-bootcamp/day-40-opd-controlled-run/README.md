# Day 40 — Deferred Controlled OPD Run 与 S3 Selection

状态：`deferred_unselected`
日期：`unscheduled_after_day30`
强度：当前 0 GPU；仅独立 teacher-extension charter v2 激活后按 Day 39 实测预算

## 当前执行状态

当前没有 T2、OPD replay gate 或 S3。本页不得执行，且 S3 必须保持 `deferred_unselected`；活动 capstone policy charter v1 不把“未运行 OPD”视为失败。

## 主要目标

若未来 charter v2 激活，从 exact `S1` 和 frozen T2 运行受控 OPD trajectory，在预注册 extension budget 内选择 `S3`。Extension 保持一个 OPD recipe；若 Day 39 只能通过 S1d recovery，则结果改名 `S3d` 并降级因果声明。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 40 — Controlled OPD](../SCALING-BOOK-READING-GUIDE.md#day-40)。

## 训练 / 实验

- 使用 Day 39 已通过的 resolved config、placement 和 replay schema；
- 保存 student-generated rollouts、teacher scoring、token diagnostics、loss/grad、throughput 和 GPU-hours；
- early/mid/final candidates 使用相同 capstone dev protocol；
- 监控 teacher/student divergence、overlap、entropy、response length、format/error 和 E2E success；
- 抽查 student 最常访问但 teacher/student 分歧最大的 prefixes；
- 使用预注册规则选择 S3，允许 `inconclusive` 或“OPD failure”。

不得根据 dev bad cases 修改 T2、teacher prompt 或 scorer。协议错误需要新 OPD version；训练问题只能在当前 budget/stop rules 内处理。

## Evidence-first 产物

- `../artifacts/logs/capstone/day40-opd/`
- `../artifacts/reports/capstone/day40-opd-selection.md`
- `../artifacts/checkpoints/capstone/S3-opd-manifest.json`
- candidate predictions、token-divergence slices、cost ledger

## 验收

- [ ] 独立 teacher-extension charter v2、用户 teacher decision 和 Day 39 gate 已通过；否则本页保持 deferred 且不生成 S3。
- [ ] 激活后的 candidates 继承同一 S1/T2/data/protocol hashes；recovery candidates 单独继承 S1d 并命名 S3d。
- [ ] Student update budget 不超过 Day 37 前冻结的对照边界。
- [ ] S3 selection 同时看 E2E primary、guardrails、error rate 与 uncertainty。
- [ ] Teacher scoring GPU-hours 和传输/等待时间没有隐藏在 student throughput 中。
- [ ] `capstone_frozen` 未揭盲。
