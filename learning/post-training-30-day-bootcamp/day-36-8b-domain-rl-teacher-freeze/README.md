# Day 36 — Deferred Teacher Domain RL、T2 Selection 与 Freeze

状态：`deferred_unselected`
日期：`unscheduled_after_day30`
强度：当前 0 GPU；仅独立 teacher-extension charter v2 激活后按 Day 35 runbook

## 当前执行状态

当前 `teacher_model_id/revision: null`，`T0/T1/T2` 均不存在。本页只是未来模板，不得执行、不得生成空壳 T2 manifest，也不得把任意 8B/9B 模型补入配置。

## 主要目标

若未来 charter v2 激活，从 exact `T1` 运行受控 domain RL/RLVR，选择并 hash-lock `T2` teacher candidate。只使用 extension dev；相对 S1 的 teacher advantage 仍需独立 promotion gate。

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

- `../artifacts/logs/capstone/day36-teacher-domain-rl-trajectories/`
- `../artifacts/reports/capstone/day36-teacher-domain-rl.md`
- `../artifacts/checkpoints/capstone/T2-teacher-manifest.json`
- teacher serving config、candidate predictions、promotion record

## 验收

- [ ] 独立 teacher-extension charter v2 与用户 model decision 已存在；否则本页保持 deferred 且 T2 不存在。
- [ ] 激活后 T2 相对 T1 达到预注册 domain improvement，且不是 reward/length/format hack。
- [ ] T2 相对 T1 的领域提升成立；相对 S1 的 teacher advantage 项标记为 `pending_extension_probe`，未提前伪造结论。
- [ ] T2 checkpoint、tokenizer/template、generation 和 serving hashes 已冻结。
- [ ] Teacher 能对给定 prefix/token IDs 返回可重放 log-prob evidence。
- [ ] `capstone_frozen` 未揭盲。

## Handoff

只有激活并生成 T2 后，后续 extension 才适用 teacher immutability；任何 teacher 权重修订产生新 extension version。当前活动 Day 37 不读取、比较或等待 T2。
