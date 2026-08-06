# Day 30 — Post-training Design、Clean Reproduction 与综合口述

日期：`2026-08-25`
状态：`not_started`
强度：4–5 小时

## 主要目标

把一个月收敛为一份小模型 post-training 设计和一条可从干净环境复现的证据链。毕业作业是训练知识与实验判断，不建设 auto-train/auto-harness 控制面。

## 理论（45 分钟）

精读清单：[Day 30 — Post-training design review](../SCALING-BOOK-READING-GUIDE.md#day-30)。

- SFT、DPO、online GRPO 分别解决什么数据/目标问题，何时不该使用后者。
- Data contract、optimizer/schedule、resume、eval selection、reward/verifier、on-policy 和 weight-version 如何连接。
- 对每个结论标记 `measured / inferred / unknown / next experiment`。

## Coding（45–60 分钟）

- 在新 shell/干净镜像中只根据保存的 README、lockfile 和 config 建环境，不读取旧 shell history。
- 运行 Day 15/17/22/24 的关键 validators/tests，并验证 dataset/checkpoint checksums。
- 检查启动脚本能解析 frozen manifest、输出目录、`max_steps`、resume 和 held-out config。
- 任何需要手工补入的隐式参数都先修正文档/config，再启动 smoke。

## 训练 / 实验：Clean Reproduction（45–60 分钟）

从新 shell/干净镜像选择本月一条最小但完整的训练链复现：

```text
environment + pinned versions
  -> frozen data manifest/validator
  -> 5–10 step DPO 或 GRPO smoke
  -> checkpoint save/load
  -> frozen held-out eval
  -> evidence report
```

- 对比原 run 与 clean run 的 sample IDs、resolved config、首步 metrics 和输出；记录 reproducibility 等级。
- slime 受多卡资源限制时不重复大 run，但 Day 29 dump、train-only replay、weight-version evidence 必须可定位和复查。

## Post-training Design Doc（75–90 分钟）

为一个实际小模型任务写设计，包含：

- 目标能力、数据 provenance/split、SFT/preference/online trajectories contract。
- 选择 SFT、DPO 或 GRPO 的理由与不采用其他方法的条件。
- optimizer/LR/warmup/batch/token budget、checkpoint/resume 与 reproducibility。
- reward/verifier 版本、failure semantics、hacking guardrails。
- checkpoint selection、frozen held-out、slice/CI 和 promotion/inconclusive policy。
- 资源估算、最小 pilot、failure-injection/rollback runbook。

不得加入 scheduler、自动控制面或 auto-harness 产品设计。30B dense/MoE capacity 只能作为 Stretch appendix，标明估算和必须 benchmark 的未知量。

## 资源与租卡

- Clean smoke：优先 1×H100 80GB，预计 2–3 小时；若 DPO smoke 可在更小卡完成则使用更小资源。
- Design/口述：CPU only。完成 smoke 后立即关机。
- 不为 30B appendix 或 8×H100 slime Stretch 追加租卡。

## 综合口述（45 分钟）

录制一次 15 分钟说明：

1. 数据从 raw sample 到 loss token。
2. Packing、batch、LR/warmup 和梯度稳定性证据。
3. Exact resume 需要哪些 state。
4. Eval 如何选择 checkpoint 而不泄漏。
5. Preference provenance、DPO 与 online GRPO 的差异。
6. slime 中 Sample/DataSource/rollout/train/weight-version 的闭环。
7. 一次真实失败如何归因，以及仍未知什么。

再用 15 分钟接受反问，剩余时间修正设计中没有 evidence 的断言。

## Evidence-first 产物

- `../artifacts/reports/post-training-design.md`
- `../artifacts/reports/clean-reproduction.md`
- 15 分钟口述录音/提纲与 evidence index
- 可选 `30b-capacity-stretch-appendix.md`
- `../artifacts/reports/phase1-capstone-handoff.md`（只列接口/readiness，不要求执行 Capstone）

## 最终验收

- [ ] Clean smoke 无隐式手工步骤，能从 manifest 跑到 held-out result。
- [ ] 能解释 Day 15–29 每个关键结果的 config/log/sample 证据。
- [ ] Post-training design 覆盖数据、训练、恢复、Eval、RL 和故障处理。
- [ ] slime Core 与 reward/replay evidence 可复查；8×H100 不作为毕业条件。
- [ ] 30B capacity 仅为可选 appendix。
- [ ] 完成 [`../PROGRESS.md`](../PROGRESS.md)，并列出五个仍需用实验回答的问题。

## Optional Capstone Handoff

Day 30 仍是 30-Day Core final，不因 Day 31–42 未执行而降级。额外生成 `phase1-capstone-handoff.md`，只汇总：

- 可复用的 eval/scorer/consumption schema 与需新建的 capstone eval v2；
- SFT recipe lock、selected-checkpoint promotion manifest 与 inference/resumable 两类路径；
- length/packing、optimizer、resume state inventory 的 measured defaults 和禁止外推边界；
- TP parity launcher、checkpoint conversion、trajectory/reward/replay/weight-version contracts；
- pinned environment/container、known-good commands、evidence index；
- 尚未满足的 4B/8B model/data/license/cache/topology/framework compatibility 和预算项。

该 handoff 是 Day 31 readiness 输入，不预先宣称 OPD framework、teacher checkpoint 或 capstone dataset 已存在。

## Final Log

### 实际复现链

### 最强证据

### 最大缺口

### 五个下一步问题

1.
2.
3.
4.
5.
