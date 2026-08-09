# Day 41 — S0/S1/S2 Matched Eval、Frozen Confirmation 与 Cost

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；inference/eval GPU

## 主要目标

在 S0、S1、S2 manifests 与 selection records 全部锁定后，一次性运行 capstone policy charter v1 的相同 frozen suite，比较 Qwen3.5-4B Base、selected coding SFT 和 selected direct coding RL 的能力、失败模式与成本。Teacher/OPD candidates 不存在，也不是本日阻塞项。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 41 — Matched Eval 与 Cost](../SCALING-BOOK-READING-GUIDE.md#day-41)。

## 比较规则

先确认 S0/S1/S2 manifests、processor/loader/render contracts 和 selection records 均不可变，再揭盲 frozen：

```text
S0 Qwen3.5-4B Base
  -> S1 selected coding SFT
  -> S2 selected direct coding RL
  -> same eval_suite_hash
  -> checkpoint-specific render/execution comparison keys
  -> paired sample-ID analysis
```

`eval_suite_hash` 表示相同 raw task IDs、references、sandbox/scorer 和 aggregation；每个 checkpoint 仍保留自己的 model/processor/render/execution hashes。不得用共同 model family 掩盖实际 protocol diff。

Day 41 前必须把 `teacher_branch_decision` 冻结为 `deferred_for_policy_charter_v1` 或在新的 charter v2 中显式激活。若 policy charter v1 在本日消费 frozen，未来 teacher/OPD extension 必须创建新的 frozen suite，不能使用已揭盲数据选择 T2/S3。

## 报告

- E2E primary、所有 slices、paired deltas 和 uncertainty；
- parse/compile、public/hidden tests、runtime error、timeout、unsafe action 和 reward-hacking taxonomy；
- S0→S1、S1→S2 与 S0→S2 的 paired stage deltas；
- SFT train、RL train、rollout 和 eval GPU-hours；
- generated/trained/rollout tokens、peak memory、throughput、checkpoint size 与推理成本；
- capability-per-cost 仅在口径完整时计算。

Teacher preparation/scoring、T2 retention 和 S2↔S3 只属于未来 charter v2 报告，不得在当前账本中填入猜测值。

第一次读取 frozen score 时写 consumption record。无论结果如何，不重新选 S1/S2。

## Evidence-first 产物

- `../artifacts/eval/capstone/frozen-{S0,S1,S2}-predictions.<run_id>.jsonl`
- `../artifacts/eval/capstone/frozen-run-manifest.json`
- `../artifacts/eval/capstone/frozen-consumption-record.json`
- `../artifacts/reports/capstone/day41-matched-eval-and-cost.md`

## 验收

- [ ] 全部预注册 stage checkpoints 使用相同 eval suite 和 scorer/environment；模型特定 keys 未被混淆。
- [ ] Frozen 只揭盲一次，selection 未改变。
- [ ] 逐样本 paired evidence、bad cases 和 uncertainty 均保存。
- [ ] SFT 与 direct coding RL 分别报告 trained/rollout budgets 和总 GPU-hours。
- [ ] Teacher/OPD 保持 deferred；没有生成 T*/S3* predictions 或伪造 teacher cost。
- [ ] 结论区分 measured、inferred、unknown；小差异允许 `inconclusive`。
