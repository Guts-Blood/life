# Day 21 — Qwen3.5 v2 SFT Candidate Selection Audit

计划日期：`2026-08-16`

提前完成日期：`2026-08-13`

状态：`done_with_handoff_pending`

强度：1 小时，阅读、复盘与已有 evidence 审计

## 完成结论

Day 21 的学习与 fixed-full112 checkpoint selection audit 已完成。旧状态
`blocked_no_day16_candidate` 已被后续批准的 RSI v0002 charter 取代：Primary
seed `20260809` 的 early、mid、final 三个 main checkpoints 均通过冻结门槛，
确定性 selector 按预注册规则选择并 promote
`main-s20260809-lr1e-4-final`。

Primary checkpoint 自身通过全部冻结门槛；同配方、同 checkpoint label、fresh
Base 起点、独立训练 seed `20260810` 产生的另一个 checkpoint 也通过同一
full112。因此可以声明：该 recipe 获得 two-seed same-suite qualification，Primary
仍是 policy-selected operational winner。不能据此声明 Primary 权重被第二次复验、
统计显著优于 early、独立 held-out 泛化已经确认，或一般金融 Agent 能力已经提升。

Day 21 学习与选模判断已经完成；`downstream-ready S1` 交接仍等待 winner 权重
归档、winner-only merged export、adapter/merged parity 与正式 downstream
manifest。这是交付 blocker，不是重新训练 blocker。

## Candidate Boundary

- Base：`Qwen/Qwen3.5-4B-Base@1001bb4d…`，只作 comparator/fallback，不参加 LoRA 排名。
- Probe：`probe-s20260809-lr1e-4-t12000` 只授权 Main 使用 `LR=1e-4`；Main 从 fresh Base 训练，不从 Probe warm-start。
- Primary candidates：`main-s20260809-lr1e-4-{early,mid,final}`。
- Confirmation：`main-s20260810-lr1e-4-final`，只验证锁定配方，不参加第二轮选模。
- Excluded：Day 01–12 的 Qwen3-0.6B checkpoints、Day 19 Full-SFT diagnostics、旧 Day 20 v1 probes、失败/不完整 attempts 与临时 debug exports。
- Downstream：只有完成归档、export parity 和 manifest 的 winner 才能作为 Day 23/25 的正式 S1 parent。

## 冻结 Selection Policy

所有 Primary 对象使用同一 full112、每项技能 28 条、同一 sample-order SHA、
同一 normalized/E2B/complete comparison identity，以及 hash-locked scorer 和
sandbox contract。

Hard gates：

| 指标 | 门槛 |
|---|---:|
| Total | `>=65/112` |
| General | `>=5/28` |
| Math | `>=17/28` |
| Finance | `>=10/28` |
| Code / E2B | `>=14/28` |
| Format | `>=90/112` |
| Code execution eligible | `>=26/28` |
| Infrastructure failures | `=0` |

只有通过全部 hard gates 的 candidate 才参加排名。排名顺序在结果前冻结为：

```text
Total desc -> General desc -> Code desc -> checkpoint target tokens asc
```

## Selection Result

| 对象 | Total | General | Math | Finance | Code | Eligible | Format | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base | 59 | 1 | 24 | 16 | 18 | 27 | 81 | fallback only |
| early | 81 | 21 | 21 | 22 | 17 | 28 | 112 | eligible |
| mid | 75 | 21 | 20 | 17 | 17 | 28 | 112 | eligible |
| final | 81 | 21 | 21 | 19 | 20 | 28 | 112 | Primary winner |
| Confirmation final | 73 | 21 | 21 | 14 | 17 | 28 | 112 | pass |

early 与 final 的 Total、General 相同；final 依据冻结规则的第三项 Code
`20 > 17` 胜出。它是 `operational winner`，不是已被证明显著优于 early 的
`statistically best`。

## Evidence

- [`RSI goal`](../rsi-control/goal.json)：冻结目标、hard gates、confirmation 与 claim boundary。
- [`Primary selection`](../rsi-control/versions/rsi-v0002/runs/run-002-main-primary/attempts/attempt-001/primary-selection.json)：共同 comparison identity、候选分数、排序与 winner。
- [`Final promotion`](../rsi-control/versions/rsi-v0002/runs/run-003-main-confirmation/attempts/attempt-001/final-promotion.json)：independent-seed confirmation 与 promote decision。
- [`Version metrics`](../rsi-control/versions/rsi-v0002/metrics.json) 与 [`result summary`](../rsi-control/versions/rsi-v0002/RSI-V0002-RESULT.md)：Primary/Confirmation gates、成本和结论边界。
- [`Archive audit`](../rsi-control/ARCHIVE-AUDIT-20260813.md)：checkpoint identity、compact archive 边界与恢复规则。

原计划的 `day21-qwen35-sft-selection-policy.md`、
`day21-qwen35-blinded-selection.json` 与 downstream-ready
`day21-qwen35-sft-promotion-manifest.json` 没有被事后补造。RSI ledger 是 superseding、
machine-verifiable 的 fixed-suite selection evidence；正式 S1 handoff manifest 仍需在
权重恢复和 export parity 后生成。

## 验收

### Day 21 学习与 selection audit

- [x] candidate set 只含 RSI v0002 Primary early/mid/final；Base、Probe 与 Confirmation 的角色没有混入排名。
- [x] full112 cohort、sample order、comparison keys、scorer、sandbox 和 hard gates 在比较中保持冻结。
- [x] 先应用全部 hard gates，再按冻结 tie-breaker 选择唯一 operational winner。
- [x] winner checkpoint integrity、snapshot、adapter hashes 与 `resumable=true` 已记录。
- [x] fresh-Base、独立训练 seed 的同套件 Confirmation 已一次性运行并通过。
- [x] 已完成 Day 21 对话式 quiz：能区分 train loss/eval/guardrail、operational/statistical winner、multiple comparisons、winner's curse、CI、training-seed 与 task-sampling uncertainty，以及 export parity。

### 剩余 handoff 与证据边界

- [ ] frozen selector 未预注册 minimum meaningful difference 或 CI-based superiority gate；因此 statistical-best 结论为 `inconclusive`，不影响 operational selector decision。
- [ ] paired per-sample delta/bootstrap CI：compact archive 没有本地逐样本原始字节，当前不可重算。
- [ ] 独立 held-out confirmation：当前 Confirmation 复用了 full112，只提供 one-additional-seed same-suite reproducibility check；它不能估计 seed 分布，也不能冒充新题泛化证据。
- [ ] 完整 raw predictions、E2B results 与 44-item inventory 的本地不可变归档。
- [ ] winner checkpoint bytes 的本地/长期归档与 hash 复核。
- [ ] winner-only merged inference export、fresh reload 与 adapter/merged parity。
- [ ] downstream-ready Day 21 S1 promotion manifest/downstream key；完成前 Day 23/25 仍不得消费该权重。

## Daily Log

### Candidate set / excluded set

Primary candidate set 为 early/mid/final；Base 只作 comparator，Probe 只选 LR，
Confirmation 只验证锁定配方。所有历史 v1、diagnostic 和失败 artifacts 均排除。

### Blinded decision

严格说这不是事后补做的盲化 rehearsal；它是由 hash-locked contract 与确定性
selector 在完整 cohort 到齐后自动应用冻结规则。结果为 final 胜出。

### Promoted SFT anchor or blocker

`main-s20260809-lr1e-4-final` 已通过 Primary fixed-full112 gates，并由 RSI selector
作出 operational promotion decision；同 recipe 的另一个 seed checkpoint 也通过同套件。
正式 downstream-ready S1 仍受权重归档、merge/export parity 和 manifest 阻塞，不需要重新训练。

### Remaining selection risks

Primary `81` 与第二个 seed 的 `73` 构成 observed two-seed spread；它与 training-seed
sensitivity 和 selection optimism 相容，但两个点不能估计或识别任何一项效应。
同一 full112 的重复使用不能证明 held-out 泛化；缺少逐样本原始字节也使 paired
uncertainty audit 暂不可重算。
