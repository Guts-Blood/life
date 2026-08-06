# Day 41 — Matched Eval、Frozen Confirmation 与 Cost Accounting

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；inference/eval GPU

## 主要目标

在 T2、S2、S3 全部选定且 manifests 锁定后，一次性运行相同 capstone frozen suite，比较 teacher、direct RL student 和 OPD student 的能力、失败模式与成本。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 41 — Matched Eval 与 Cost](../SCALING-BOOK-READING-GUIDE.md#day-41)。

## 比较规则

先确认三个 candidate manifests 和 selection records 均不可变，再揭盲 frozen：

```text
T0/T1/T2 8B trajectory
S0/S1/S2/S3 <=4B trajectory
  -> same eval_suite_hash
  -> model-specific render/execution comparison keys
  -> paired sample-ID analysis
```

`eval_suite_hash` 表示相同 raw task IDs、references、environment/scorer 和 aggregation；每个模型仍保留自己的 model/render/execution hashes。不得因为跨模型而伪造相同 `comparison_key`。

## 报告

- E2E primary、所有 slices、paired deltas 和 uncertainty；
- call/no-call、tool、args、execution、multi-turn recovery、hallucination/error taxonomy；
- T0→T1→T2、S0→S1→S2/S3 的阶段增量，T2→S3 retention 与 S2↔S3 direct comparison；
- train/rollout/teacher-scoring/eval GPU-hours；
- generated/trained/teacher-scored tokens、peak memory、throughput、checkpoint size 与推理成本；
- teacher preparation one-off 与 amortized 两种账本；
- capability-per-cost 仅在口径完整时计算。

第一次读取 frozen score 时写 consumption record。无论结果如何，不重新选 S2/S3/T2。

## Evidence-first 产物

- `../artifacts/eval/capstone/frozen-{T0,T1,T2,S0,S1,S2,S3}-predictions.<run_id>.jsonl`
- `../artifacts/eval/capstone/frozen-run-manifest.json`
- `../artifacts/eval/capstone/frozen-consumption-record.json`
- `../artifacts/reports/capstone/day41-matched-eval-and-cost.md`

## 验收

- [ ] 全部预注册 stage checkpoints 使用相同 eval suite 和 scorer/environment；模型特定 keys 未被混淆。
- [ ] Frozen 只揭盲一次，selection 未改变。
- [ ] 逐样本 paired evidence、bad cases 和 uncertainty 均保存。
- [ ] Direct RL 与 OPD 同时报告 student budgets 和总 GPU-hours。
- [ ] 结论区分 measured、inferred、unknown；小差异允许 `inconclusive`。
