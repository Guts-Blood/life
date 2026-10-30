# Day 20 — 周末 Reading：Training Failure Signatures

日期：`2026-08-15`
状态：`not_started`
强度：Core 1 小时，仅阅读/复盘；Optional Resume Lab 另计 3–5 小时

说明：这里是**顺序课程 Day 20**，不是已提前执行的 `day-20-qwen35-balanced-lora-sft` standalone run。Day 17 当前的 `blocked` 状态和审计结论保持不变；原 Day 17 exact-resume 实验从主线硬任务降级为本日可选择的 Optional R。

## 主要目标

把 Day 16–19 的 Qwen3.5 v2 优化、resume、兼容性与故障证据压缩成看到指标后可以迅速展开的 signature map。Core 不再跑 GPU；只有 Optional R 的 entry gate 全部满足、且 exact-resume 诊断对当前决策确有价值时，才选择性执行连续/恢复轨迹对照。

## 理论与复盘（60 分钟）

精读清单：[Day 20 — Training failure signatures](../SCALING-BOOK-READING-GUIDE.md#day-20)。

- 15 分钟：重看 Day 19 A–F step metrics，只描述观察，不先看 run 名。
- 15 分钟：重看 decoded samples、processor/template hash、label-token ratio、modality/freeze inventory 和数据分布。
- 15 分钟：重看 baseline/throughput profiler 与 Day 18 compatibility evidence，区分 data wait、compute/GDN、communication 和 OOM。
- 15 分钟：完成下表，并为每类写一个“最便宜的下一步 probe”。

| Signature | 可能原因 | 区分性指标 | 最小 probe | 停止条件 |
|---|---|---|---|---|
| loss/grad spike | | | | |
| loss 平稳但能力退化 | | | | |
| loader/GDN/processor 不一致 | | | | |
| 吞吐骤降 | | | | |
| resume 后分叉 | | | 先审 optimizer/scheduler/RNG/cursor；必要时选 Optional R | state inventory 不完整则停止，不开 GPU |
| 只在某个 data slice 退化 | | | | |

## Optional R — Exact Resume Failure Lab

默认：`skip`。跳过 Optional R 不影响 Day 20 Core 完成，也不阻塞 S1 promotion；S1 仍必须有通过 integrity audit 的 checkpoint 和可加载的 inference export，但不强制证明 bitwise/逐 step exact resume。

只有以下 entry gate 全部成立时才允许选择 `run`：

- 新 SFT charter 已批准，且存在合法 selected Qwen3.5 LoRA candidate；
- checkpoint 明确为 `resumable=true`，包含 adapter、optimizer、scheduler、RNG、global step 和可验证的数据游标；
- initialization、数据顺序、packing、batch、seed、processor/template、runtime 和 GPU topology 已冻结；
- 当前确实需要判断 resume divergence，而不是仅为了补课程勾选；
- 3–5 小时 GPU window 与 checkpoint/evidence 磁盘预算已单独批准。

选择 `run` 后，复用 [`Day 17 Readiness Contract`](../artifacts/configs/day17-qwen35-resume-readiness.json)：

```text
Run A：step 0 → 40，不中断

Run B1：step 0 → 20
        → 保存完整 checkpoint
        → 退出进程

Run B2：新进程恢复 step 20 → 40

比较 step 21–40：
sample/render/label tokens、LR、loss、grad norm、adapter、
optimizer、scheduler、RNG、sampler cursor、累计 supervised tokens
```

若还需要验证诊断能力，再复制 checkpoint，删除 scheduler 或 sampler cursor，只跑 3–5 steps；故障副本不得覆盖正确 checkpoint。无法 bitwise deterministic 时，记录首个分叉 step、`max_abs_diff` 和已定位的 nondeterministic source。

Optional R 有三种合法结果：

- `skipped_not_needed`：默认结果；不影响 Core 和 S1。
- `blocked_entry_gate`：用户想运行，但 candidate/checkpoint/合同不完整；不开 GPU。
- `run_complete`：完整 Run A/B 与可选 failure injection 已产生证据；它是补充诊断，不把 resumed path 计成新 candidate。

## Coding

Core 无。Optional R 只有在 entry gate 通过后，才允许针对真实 checkpoint schema 实现/验证 auditor；不预写一个无法绑定 candidate 的假 runner。

## 训练 / 实验

Core 无；只能使用 Day 15–19 已保存的 v2 证据，不重跑 GPU，也不引用 v1 指标填空。Optional R 是唯一可选 runtime extension，必须单独记录选择理由和 entry-gate 结果。

## 资源与租卡

Core：CPU only，严格 60 分钟，不租 GPU。

Optional R：默认 `0 GPU`。若选择运行，使用 candidate 的同一单卡 topology；candidate 若来自单卡 H800 80GB，就继续使用 `1×H800 80GB`，不为了 4B 模型额外引入 TP/多卡。计划 3–5 小时，独立于 Core 计时。

## Evidence-first 产物

Core：完成本页 signature map 和六条 takeaway。

Optional R 若跳过，只记录 `skipped_not_needed` 和理由；若运行，生成：

- `../artifacts/reports/day20-optional-resume-equivalence.md`
- `../artifacts/logs/day20-optional-resume/` 下的 Run A/B config、timeline 与 state manifests
- adapter/optimizer/scheduler tensor diff、首个分叉和可选 failure-injection evidence
- selected candidate 的 adapter reload / merged export parity

## 验收

- [ ] 六类 signature 都有可证伪的替代解释。
- [ ] 每类都有低成本 probe 和明确停止条件。
- [ ] 写出一个曾经会误诊、现在能区分的例子。
- [ ] Optional R 明确记录为 `skipped_not_needed`、`blocked_entry_gate` 或 `run_complete`；没有把“可选”写成 Core 未完成。
- [ ] 若 Optional R 运行，确实退出并启动新进程，且 resumed path 没有被登记为新 candidate。

### 六条 takeaway

1.
2.
3.
4.
5.
6.
