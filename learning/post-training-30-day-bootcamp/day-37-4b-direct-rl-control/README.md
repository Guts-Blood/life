# Day 37 — 4B Common Anchor 与 Direct-RL Control

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时人工工作；GPU wall time 由 smoke 决定

## 主要目标

选择并冻结 <=4B student 的共同 SFT 起点 `S1`，随后按与 teacher domain RL 相同的 environment/reward contract 训练 direct-RL control `S2`。这是 OPD 必须击败或解释的基线。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 37 — Common S1 与 Direct RL](../SCALING-BOOK-READING-GUIDE.md#day-37)。

## Common anchor

- 从 frozen `S0` 与 T1 使用的同一 `sft_train` manifest/label-token budget 进行 full-parameter SFT，训练或选择 `S1`；若要改变规模相关超参，必须预注册并把跨规模比较标成 recipe comparison；
- 若已有历史 4B SFT checkpoint，只有 lineage、template、data、eval 和可继续训练状态全部通过时才能采用；
- `S1` 的 exact inference/train hashes 同时写入 direct-RL 与 OPD run manifests；
- 在分叉前复制的是 manifest/reference，不手工复制后修改权重目录。

S1 冻结后先在 Day 31 的独立 `teacher_advantage_probe` manifest 上比较 T2 与 S1，确认 T2 提供逐样本可定位的新能力且 tokenizer/token IDs 完全一致。通过后把 T2 从 `teacher_candidate` 标记为 `promoted_for_opd`；不通过则停止 Day 38–40。Dev 仍只用于 checkpoint selection，不用它反复筛选“适合蒸馏”的 prompts。

## Direct RL control

1. 在 `S1` 上运行与 Day 35 相同的 rollout/reward dry gate。
2. 预注册 student optimizer-update、trained-token 与 rollout-token budgets。
3. 保存 early/mid/final candidates，使用 capstone dev 选择 `S2`。
4. 记录 reward、KL/entropy、length、zero-variance、tool/error slices 和 GPU-hours。
5. 不因将来 OPD 结果更好或更差而回头扩大 S2 budget。

## Evidence-first 产物

- `../artifacts/checkpoints/capstone/S1-common-anchor-manifest.json`
- `../artifacts/checkpoints/capstone/S2-direct-rl-manifest.json`
- `../artifacts/reports/capstone/T2-vs-S1-teacher-advantage.md`
- `../artifacts/logs/capstone/day37-4b-direct-rl/`
- `../artifacts/reports/capstone/day37-direct-rl-control.md`

## 验收

- [ ] S1 的起点 hash、SFT lineage 和 train/eval evidence 完整。
- [ ] S2 与后续 OPD 使用同一 S1、prompt pool、environment/scorer 和 max response contract。
- [ ] S2 budget 在运行 OPD 前冻结。
- [ ] S2/S3 的 unique policy prompt IDs/exposures、student trained tokens、rollout-token cap、max response 与 checkpoint cadence 匹配。
- [ ] T2 相对 S1 的 teacher advantage probe 已完成并记录 promotion/stop decision。
- [ ] S2 dev selection 依据预注册规则，未运行 frozen。
- [ ] 训练 reward 与 E2E task success 分开报告。
