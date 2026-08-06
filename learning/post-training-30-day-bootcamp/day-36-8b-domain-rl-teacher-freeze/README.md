# Day 36 — 8B Domain RL、T2 Selection 与 Teacher Candidate Freeze

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时人工工作；GPU wall time 按 Day 35 runbook

## 主要目标

从 `T1` 运行受控 domain RL/RLVR，选择并 hash-lock `T2` teacher candidate。今天只使用 capstone dev，不揭盲 frozen；相对 S1 的最终 teacher advantage promotion 留到 Day 37，因为 S1 此时尚未产生。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 36 — 8B Domain RL](../SCALING-BOOK-READING-GUIDE.md#day-36)。

## Gate 顺序

1. rollout-only 保存 raw trajectories 与 policy version；
2. reward replay 与在线 reward 完全对齐；
3. one-update 保存 old/current/ref log-prob、advantage、mask 和 grad evidence；
4. weight sync 后证明下一批 rollout 使用新版本；
5. 按 frozen update/rollout-token budget 运行 candidate trajectory；
6. 在 dev 上选择 T2，执行 reward-hacking、length、error 和 general guardrail audit；
7. 冻结 inference/train checkpoints、teacher serving config 与 immutable teacher-candidate manifest。

## Evidence-first 产物

- `../artifacts/logs/capstone/day36-8b-domain-rl-trajectories/`
- `../artifacts/reports/capstone/day36-8b-domain-rl.md`
- `../artifacts/checkpoints/capstone/T2-teacher-manifest.json`
- teacher serving config、candidate predictions、promotion record

## 验收

- [ ] T2 相对 T1 达到预注册 domain improvement，且不是 reward/length/format hack。
- [ ] T2 相对 T1 的领域提升成立；相对 S1 的 teacher advantage 项标记为 `pending_day37`，未提前伪造结论。
- [ ] T2 checkpoint、tokenizer/template、generation 和 serving hashes 已冻结。
- [ ] Teacher 能对给定 prefix/token IDs 返回可重放 log-prob evidence。
- [ ] `capstone_frozen` 未揭盲。

## Handoff

Day 37–40 不得继续修改 T2。Day 37 只做相对 S1 的 promotion probe；任何 teacher 权重修订产生 `T2-v2`，并使依赖旧 teacher 的 OPD comparison 成为不同实验。
