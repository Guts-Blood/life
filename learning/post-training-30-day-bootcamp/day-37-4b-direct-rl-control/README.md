# Day 37 — Qwen3.5-4B Common Anchor 与 Direct Coding RL

状态：`deferred_after_architecture_study`
日期：`unscheduled_after_day30`
强度：4–5 小时人工工作；GPU wall time 由 smoke 决定

## 当前状态

Day 27–30 已转向 Megatron/slime architecture study。本 direct-RL 训练任务暂停，不因前置阅读完成而自动获得 GPU 或训练授权；只有用户明确重启执行型 Capstone 后才恢复。

## 主要目标

从 Day 31 冻结的 exact `S0 = Qwen/Qwen3.5-4B-Base@revision` 选择 coding SFT 起点 `S1`，随后按 Day 24–29 + Day 31 的 sandbox/reward contract 训练并选择 direct coding RL/RLVR checkpoint `S2`。S0→S1→S2 是当前唯一活动 trainable policy lineage，完全独立于未选择的 teacher。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 37 — Common S1 与 Direct RL](../SCALING-BOOK-READING-GUIDE.md#day-37)。

## Common anchor

- 从 frozen `S0` 与 Day 31 的 `sft_train` manifest/label-token budget 训练并选择 `S1`；使用 full、LoRA 或 QLoRA 必须引用 Day 32 capacity/smoke evidence并单独冻结，本文不预选；
- 若已有历史 Qwen3.5-4B SFT checkpoint，只有 exact S0 revision、processor/template、data、eval、module coverage 和可继续训练状态全部通过时才能采用；
- `S1` 的 exact inference/train hashes写入 direct-RL manifest；未来 charter v2 只能引用该 immutable S1，不得回写它；
- 在分叉前复制的是 manifest/reference，不手工复制后修改权重目录。

S1 冻结后直接进入活动 direct-RL gate，不等待 T1/T2。若未来用户通过 charter v2 激活 teacher extension，才另建新的 `teacher_advantage_probe`，比较 T2 与 exact S1；该 optional probe 不回写或阻塞当前 capstone policy charter v1。

## Direct RL control

1. 在 `S1` 上运行 Day 24–29 + Day 31 冻结的 coding rollout/sandbox/reward dry gate。
2. 预注册 policy optimizer-update、trained-token 与 rollout-token budgets。
3. 保存 early/mid/final candidates，使用 capstone dev 选择 `S2`。
4. 记录 reward、KL/entropy、length、zero-variance、compile/test/runtime-error slices 和 GPU-hours。
5. 不因未来可能选择 teacher/OPD 而回头扩大 S2 budget。
6. 保存 compile/test/timeout/runtime-error、reward components、response/code length、policy version 与 weight-sync evidence。

## Evidence-first 产物

- `../artifacts/checkpoints/capstone/S1-common-anchor-manifest.json`
- `../artifacts/checkpoints/capstone/S2-direct-rl-manifest.json`
- `../artifacts/logs/capstone/day37-4b-direct-rl/`
- `../artifacts/reports/capstone/day37-direct-rl-control.md`
- optional charter v2 only：`../artifacts/reports/capstone/T2-vs-S1-teacher-advantage.md`

## 验收

- [ ] S1 的起点 hash、SFT lineage 和 train/eval evidence 完整。
- [ ] S0/S1/S2 model、processor、loader、modality、module-coverage 和 recipe hashes 完整。
- [ ] S2 使用 frozen policy prompt pool、sandbox/scorer 和 max-response contract。
- [ ] S2 budget 在训练前冻结；teacher/OPD 不属于当前 budget。
- [ ] S2 dev selection 依据预注册规则，未运行 frozen。
- [ ] Coding reward、compile/test components 与 E2E task success 分开报告。
- [ ] `teacher_model_id/revision` 仍为 `null`，未因本日执行自动激活 Day 38–40。
