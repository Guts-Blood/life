# 30-Day LLM Post-Training Bootcamp

状态：`in_progress`（Day 15/16 历史 closeout 保持不变；Day 21 fixed-full112 selection 与 downstream-ready S1 handoff 已完成；Day 22 已按 experimental AI-assisted 路径关闭，formal-human 路径仍 pending；Day 23 experimental DPO 路径已解锁但尚待 preflight，GRPO 仍有独立 data/reward/runtime gates）

执行日期：`2026-07-27`（周一）至 `2026-08-25`（周二）  
建议投入：工作日 4–5 小时；周末严格控制为 1 小时 reading/review，不租 GPU  
后置扩展：Day 31–42 无固定日期，有空且 readiness/budget 通过后执行

主线：`0.6B 历史证据 -> Qwen3.5-4B 迁移验收 -> Coding SFT anchor -> DPO / Coding GRPO -> 可复现训练设计 -> Qwen3.5-4B Policy Capstone`。Teacher/OPD 扩展保持 deferred，未选择 teacher。

## 本月真正要补什么

本月目标是补全 training 判断力，不是构建 auto-train 或 auto-harness 系统。最终要能跟踪一条样本如何变成训练信号、一次训练如何改变模型状态、一次异常如何被证据定位，以及 SFT/DPO/在线 RL 各自需要什么数据和运行时组件。

优先级固定为：

1. 数据契约、模板、tokenization、loss mask、mixture 与 lineage。
2. 训练 step、优化器、batch、checkpoint、resume、复现与失败诊断。
3. Preference data、DPO 与在线 RL 的 rollout/reward/logprob/advantage 数据流。
4. Eval 的冻结、逐样本证据和训练阶段之间的可比性。
5. Sharding、collective 和 Megatron：学到能读配置、判断 OOM/吞吐、理解状态归属和排障；本月不追求手写并行框架。

《How To Scale Your Model》仍贯穿 30 天，但 Day 04 以后按训练问题精选。Scaling 题必须落到 `OOM、throughput、global batch、并行配置或 checkpoint`，不再重复纯公式推导。

## 30-Day Core 毕业标准

- [ ] 能从 raw sample 追到 rendered text、tokens、labels、loss mask、source/mixture metadata，并识别数据泄漏与模板错误。
- [ ] 能为 Qwen3.5 冻结 revision、processor/template、loader、modality/freeze policy 与 golden token/mask，并解释为何它不是旧 CausalLM runner 的 model-ID drop-in。
- [ ] 能解释 `forward -> loss -> backward -> gradient accumulation -> optimizer/scheduler -> checkpoint/eval`，并用 tiny overfit 验证训练链路。
- [ ] 能设计并运行受控 SFT，区分 checkpoint integrity、普通 continuation 与 exact resume；严格连续性对照按需要作为 Optional Lab 执行。
- [ ] 能根据 loss、grad norm、learning rate、tokens/s、显存、样本输出和 eval 区分数据、优化、系统与评测问题。
- [ ] 能解释 DP/FSDP/ZeRO/TP/PP/CP/EP 切什么、通信什么，以及 global batch 为什么不乘 TP。
- [ ] 能构造与审计 coding preference pair，解释 DPO objective，并从 promoted Qwen3.5 SFT anchor 跑通一次 DPO smoke。
- [ ] 能画出 `prompt -> rollout -> reward -> advantage/logprob -> update -> weight sync -> next rollout`，识别 stale rollout、mask/logprob 错位和 reward hacking。
- [ ] 能用 ms-swift 跑 Qwen3.5-4B SFT、DPO 和 coding GRPO；只有在固定 release 通过 Qwen3.5 load→rollout→train→weight-sync gate 时才用 slime 跑闭环，并能说明 Tulu/Open-Instruct、TRL、verl 提供的参照。
- [ ] 能从干净环境复现一条最小训练链，并提交一份包含数据、状态、指标、失败处理和扩展边界的 training design。

## Qwen3.5-4B Policy Capstone 毕业标准

- [ ] 能对 Qwen3.5-4B `S0` 做 single-GPU/TP2 数值、状态与 checkpoint parity，并用实测峰值选择 full/LoRA/QLoRA。
- [ ] 能从 `S0` 晋级一个 coding SFT anchor `S1`，再从同一 `S1` 运行并选择 direct coding RL 分支 `S2`。
- [ ] 能在新 frozen suite 上比较 `S0/S1/S2` 的能力、guardrails、reward hacking、吞吐和总 GPU 成本，并允许结论为 `inconclusive`。
- [ ] 若以后单独批准 Teacher/OPD charter v2，才创建 `T0/T1/T2/S3`；当前 capstone 不因 teacher 未选而阻塞活动主线。

## 框架各自承担什么

| 角色 | 本月定位 | 是否实跑 |
|---|---|---|
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | Qwen3.5-4B SFT、DPO、GRPO 主入口；processor、GDN backend 与 rollout 配置必须 pin | 30-Day Core 与 Policy Capstone 主栈 |
| [THUDM/slime](https://github.com/THUDM/slime) | 理解 Megatron train、SGLang rollout、reward、buffer、weight sync；先验证固定 release 对 Qwen3.5 的完整闭环兼容 | Core 阅读；兼容 gate 通过才实跑 |
| [AllenAI Open Instruct/Tülu](https://allenai.github.io/open-instruct/) | 参照公开的 post-training stage、数据 mixture 与 recipe | 阅读/对照 |
| [Hugging Face TRL](https://huggingface.co/docs/trl/) | 用紧凑 trainer API 对照 SFT/DPO/GRPO 的输入输出 | 阅读/小型对照 |
| [verl](https://verl.readthedocs.io/) | 对照 actor/rollout/ref/reward resource-pool；teacher/OPD 用法只保留为 deferred extension | 30-Day 阅读；可选 Systems Stretch |
| [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 建立最小多卡 codepath、rank/state/checkpoint 心智模型 | 最小 smoke，不做全仓通读 |

模型策略分为两条不可混写的 lineage：Day 01–12 的 `Qwen/Qwen3-0.6B-Base@ddc928429ed09d9ad603fd762053d0434c15e865` 是 immutable `v1` 历史；Day 13+ 的唯一活动模型是 `Qwen/Qwen3.5-4B-Base`，属于 `v2`。Standalone Day 18–20 已把 exact revision `1001bb4d826a52d1f399e183466143f4da7b741b` 与文件 hash 固定为共同运行根，并以更强实际运行证据关闭 Day 15 onboarding；旧 Day 20 的三条 LoRA probe 又以合法的 no-candidate 结果收束 Day 16。`2026-08-12` 的独立 RSI v0002 charter 随后修复 Code target boundary，训练 early/mid/final 并选中 `main-s20260809-lr1e-4-final`；同 recipe 的 fresh-Base independent-training-seed checkpoint 也通过同一 full112。RSI ledger 截止时的 `merge_performed=false` 保持为历史事实；`2026-08-13` 的 append-only Day 21 evidence 进一步完成 checkpoint archive、winner-only merged export、fresh-process exact parity 与 downstream manifest/key，没有追溯改写 Day 16 或 RSI 旧记录。

Qwen3.5-4B-Base 是含 vision encoder 的原生多模态 checkpoint。coding 主线保留完整官方 processor/conditional-generation loader，但输入固定为 text-only，并冻结 vision tower 与 aligner、断言 LoRA module coverage；不把它当旧 `AutoModelForCausalLM` 脚本的直接替换。DPO 和 GRPO 必须从 Day 21 已完成 handoff 的 downstream-ready coding SFT `S1` promotion manifest/downstream key 起步，不能从 Base、Day 12 的 0.6B export 或未合并 adapter URI 起步。详见 [Qwen3.5-4B 迁移计划](QWEN35-4B-MIGRATION-PLAN.md)。

slime 阅读基线仍固定为 `v0.3.0`，但运行时必须验证该 release 或另一个明确 pin 的 release 是否支持 Qwen3.5 完整 load→rollout→train→weight-sync；不为跑通框架而静默换模型。Teacher/OPD 没有活动模型，只有用户另行决定后才建立 charter v2。

## 提前执行的 Standalone Experiments

下列目录使用实验发生时的 Day 18–20 run identity，不替代顺序课程中同编号的学习任务。前三项用于历史 Day 15/16 closeout；RSI v0002 是后续独立 charter，只完成 selection qualification，downstream S1 由再后续的 append-only Day 21 handoff 单独交付：

| 实验 | 状态 | 结论边界 |
|---|---|---|
| [Day 18 Megatron compatibility](day-18-megatron-minimum-codepath/README.md) | closed_pass | C0–C5 与 [`closeout audit`](artifacts/reports/day18-close.md) 通过；0 追加 GPU。是 Day 15 close 的 runtime/learnability 证据，但不等于 LoRA exact resume 或 S1。 |
| [Standalone Day 19 Full-SFT comparison](day-19-qwen35-sft-comparison/README.md) | done / diagnostic only | A/B/E comparison 与 Qwen3.5 v2 rescoring 用于定位评测合同和 Full-SFT 退化；没有产生 promoted S1。 |
| [Standalone Day 20 balanced LoRA probes](day-20-qwen35-balanced-lora-sft/README.md) | closed / no passing probe | 三档 LR probe 已完成但每条都有必要门禁失败；作为 Day 16 no-candidate close evidence，main、winner merge 与 S1 均未发生。 |
| [RSI v0002 / Day 20 v3 target-boundary SFT](rsi-control/versions/rsi-v0002/RSI-V0002-RESULT.md) | confirmed-qualified / Day 21 handoff complete | Primary final `81/112`、独立训练 seed Confirmation `73/112`，全部 frozen gates 通过；RSI ledger 的 `merge_performed=false` 保持不变，新的 Day 21 evidence 已完成 merged downstream-ready S1。 |

原 [`day-19-training-diagnostics-failure-injection`](day-19-training-diagnostics-failure-injection/README.md) 和 [`day-20-weekend-training-failures`](day-20-weekend-training-failures/README.md) 仍是顺序课程 Day 19/20，状态保持 `not_started`。

## 每天固定输出

每个工作日保留至少一种可复查证据：dataset manifest、样本审计表、配置、结构化日志、checkpoint 对比、恢复记录、failure report、逐样本预测或 runbook。只跑出一个 loss 数字不算完成。

每天 quiz 固定三题：

1. **对象/数据题**：今天有哪些数据对象或字段，它们怎样变换和对齐？
2. **状态/训练题**：哪些模型、优化器、调度器、RNG、dataloader 或 rollout 状态被读取和修改？
3. **诊断/判断题**：给一个异常或两套方案，依据哪些观测作判断，下一项最小验证是什么？

工作日建议节奏：

| 时间 | 内容 |
|---|---|
| 60–75 分钟 | 定向阅读与三题 quiz |
| 120–150 分钟 | 数据审计、训练、代码阅读或受控实验 |
| 60–75 分钟 | 曲线、逐样本结果、恢复或 failure analysis |
| 20–30 分钟 | 固化证据、结论、阻塞和下一步 |

周末只安排 60 分钟：45–50 分钟定向阅读，10–15 分钟写复盘。未完成的工作日先删 Stretch，不占用周末补长训练。

## 30 天导航

### Week 1：Scaling 基础与 Training 全景（07-27 至 08-02）

- [x] [Day 01 · 07-27 — 环境与可复现基线](day-01-environment-baseline/README.md)
- [x] [Day 02 · 07-28 — Transformer 参数、FLOPs、显存](day-02-transformer-accounting/README.md)
- [x] [Day 03 · 07-29 — Roofline 与 H100](day-03-roofline-h100/README.md)
- [x] [Day 04 · 07-30 — 分布式并行地图](day-04-parallelism-map/README.md)
- [x] [Day 05 · 07-31 — Training lifecycle 与框架职责图](day-05-training-lifecycle-framework-map/README.md)
- [x] [Day 06 · 08-01（周末 1h）— Post-training 中的 Scaling 问题](day-06-weekend-posttraining-scaling/README.md)
- [x] [Day 07 · 08-02（周末 1h）— Week 1 复盘](day-07-weekend-week1-review/README.md)

### Week 2：数据契约与受控 SFT（08-03 至 08-09）

- [x] [Day 08 · 08-03 — SFT 数据契约与 loss token](day-08-sft-data-contract/README.md)
- [x] [Day 09 · 08-04 — 数据质量、mixture 与 lineage](day-09-data-quality-mixture-lineage/README.md)
- [x] [Day 10 · 08-05 — Frozen eval 与 Base baseline](day-10-frozen-eval-baseline/README.md)（自动评测完成；30 条 human review 待补）
- [x] [Day 11 · 08-06 — 一个 SFT step 与 tiny overfit](day-11-sft-step-tiny-overfit/README.md)（Step 50；teacher-forced accuracy 98.89%；exact resume pass）
- [x] [Day 12 · 08-07 — 受控 SFT 与 checkpoint 选择](day-12-controlled-sft-checkpoints/README.md)（10/10 recovery 完成；无 eligible checkpoint；frozen test 未消费）
- [ ] [Day 13 · 08-08（周末 1h）— Qwen3 历史证据 × Qwen3.5 迁移阅读](day-13-weekend-sft-reading/README.md)（reading backlog）
- [ ] [Day 14 · 08-09（周末 1h）— Week 2 历史复盘与 Qwen3.5 readiness](day-14-weekend-week2-review/README.md)

### Week 3：稳定训练、恢复与诊断（08-10 至 08-16）

- [x] [Day 15 · 08-10 — Qwen3.5-4B onboarding 与迁移验收](day-15-packing-sequence-ablation/README.md)（08-09 `closed_superseded_by_day18_20`；不重跑、不补造原 artifacts、无 S1）
- [x] [Day 16 · 08-11 — 受控 coding LoRA SFT、packing parity 与 S1 candidates](day-16-optimization-stability-ablation/README.md)（08-09 `closed_no_eligible_candidate_by_day20_evidence`；packing=false，无 S1）
- [ ] [Day 17 · 08-12 — Exact checkpoint resume 与可复现](day-17-checkpoint-resume-repro/README.md)（[`gap audit`](artifacts/reports/day17-gap-audit.md) 已完成；RSI v0002 已有 selected/resumable candidate，exact-resume 为 optional_not_run，不阻塞 S1）
- [x] [Day 18 · 08-13 — Megatron 最小源码链、双卡 TP/DP 与 distributed checkpoint](day-18-megatron-minimum-codepath/README.md)（08-10 `closed_pass_c0_c5`；[Close 报告](artifacts/reports/day18-close.md)，0 追加 GPU，不推进 S1）
- [ ] [Day 19 · 08-14 — Optimizer/LR 稳定性与 failure injection](day-19-training-diagnostics-failure-injection/README.md)（not_started；RSI v0002 recipe 可作为 baseline）
- [ ] [Day 20 · 08-15（周末 1h）— Training failure signatures](day-20-weekend-training-failures/README.md)（Core CPU；原 Day 17 exact-resume 实验移为 Optional R，默认可跳过）
- [x] [Day 21 · 08-16（周末 1h）— Qwen3.5 checkpoint selection 与 fixed-suite qualification](day-21-weekend-eval-reading/README.md)（08-13 提前完成；selection/confirmation 与 downstream-ready S1 handoff 均完成）

### Week 4：Preference、DPO 与 Online RL（08-17 至 08-23）

- [x] [Day 22 · 08-17 — Preference provenance、length bias 与 held-out](day-22-preference-data/README.md)（08-13 执行、08-14 `closed_experimental`；experimental path ready，formal-human path pending）
- [ ] [Day 23 · 08-18 — Qwen3.5 coding DPO smoke（parent=S1）](day-23-dpo-theory-smoke/README.md)（not_started；S1 parent 与 Day 22 experimental manifest gates 已通过，先做 loss/mask/reference-policy memory preflight）
- [ ] [Day 24 · 08-19 — Coding online-RL dataflow 与 sandbox reward contract](day-24-online-rl-dataflow-reward/README.md)
- [ ] [Day 25 · 08-20 — Qwen3.5 coding GRPO lab（parent=S1）](day-25-grpo-small-model-lab/README.md)（S1 parent gate 已通过；仍待 online dataflow、reward 与 runtime gates）
- [ ] [Day 26 · 08-21 — slime 固定 release 的 Qwen3.5 兼容 gate](day-26-slime-codepath-prep/README.md)
- [ ] [Day 27 · 08-22（周末 1h）— GRPO 与 on-policy 边界](day-27-weekend-slime-rl-reading/README.md)
- [ ] [Day 28 · 08-23（周末 1h）— slime debug/replay/repro/observability](day-28-weekend-slime-architecture/README.md)

### Final：真实闭环与训练设计（08-24 至 08-25）

- [ ] [Day 29 · 08-24 — 已验证 runtime 的最小闭环、reward 修改与 replay](day-29-slime-rl-run-debugging/README.md)
- [ ] [Day 30 · 08-25 — Qwen3.5 Base→SFT→DPO/GRPO clean reproduction](day-30-training-design-reproduction/README.md)

## Qwen3.5-4B Policy Capstone + Deferred Teacher–Student Extension（Day 31–42，无固定日期）

完整章程：[Qwen3.5-4B Policy Capstone + Deferred Teacher–Student Extension](SCALED-TEACHER-STUDENT-CAPSTONE.md)

### Week 5：活动 Policy 主线与 Deferred Teacher 模板

- [ ] [Day 31 — 冻结 Qwen3.5 S0、domain eval、teacher deferred 状态](day-31-capstone-charter-eval/README.md)
- [ ] [Day 32 — Qwen3.5 single/TP2 parity 与 full/LoRA/QLoRA capacity](day-32-tp-parity-scale-accounting/README.md)
- [ ] [Day 33 — Teacher TP SFT gate 模板](day-33-8b-tp-sft-gate/README.md)（deferred；需 charter v2）
- [ ] [Day 34 — Teacher SFT/T1 selection 模板](day-34-8b-sft-selection/README.md)（deferred；需 charter v2）
- [ ] [Day 35 — Teacher domain-RL readiness 模板](day-35-domain-rl-teacher-readiness/README.md)（deferred；需 charter v2）
- [ ] [Day 36 — Teacher T2 freeze 模板](day-36-8b-domain-rl-teacher-freeze/README.md)（deferred；需 charter v2）

### Week 6：Direct Coding RL、Deferred OPD 与 Final Comparison

- [ ] [Day 37 — 从 S1 运行 direct coding RL 并选择 S2](day-37-4b-direct-rl-control/README.md)
- [ ] [Day 38 — Teacher-trace cold-start 模板](day-38-teacher-trace-cold-start/README.md)（deferred；需 charter v2）
- [ ] [Day 39 — OPD one-update/replay 模板](day-39-opd-one-update-replay/README.md)（deferred；需 charter v2）
- [ ] [Day 40 — Controlled OPD/S3 模板](day-40-opd-controlled-run/README.md)（deferred；需 charter v2）
- [ ] [Day 41 — S0/S1/S2 matched eval、confirmation 与成本](day-41-matched-eval-cost/README.md)
- [ ] [Day 42 — S1/S2 clean reproduction、failure review 与报告](day-42-capstone-clean-reproduction/README.md)

## Tracking 规则

1. 开始训练前写清输入数据版本、初始 checkpoint、唯一自变量、成功条件和停止条件。
2. 每个 run 记录 lineage、parent checkpoint、code commit、容器/环境、model revision、architecture/loader、processor/tokenizer/template hashes、modality/freeze policy、seed、完整配置与硬件。
3. SFT 记录 raw/rendered/token/label/mask 审计；DPO 记录 prompt/chosen/rejected；RL 记录 prompt/response/reward/logprob/mask/policy version 对齐。
4. 必须保存逐样本 eval 和 bad cases；aggregate score 不能单独驱动下一轮。
5. Checkpoint 至少区分“仅可推理权重”和“可连续训练状态”并通过完整性审计；只有研究 continuation/exactness 时，才运行 Day 20 Optional R 验证 step、LR、optimizer、数据位置与 RNG 的逐步连续性。
6. 当天失败要保存最小证据和下一项验证，不为打勾隐藏失败。
7. Day 07、14、21、28 使用 [`templates/weekly-review.md`](templates/weekly-review.md)。
8. v2 的 DPO/GRPO 必须从 downstream-ready `S1` manifest 分叉；仅有 selected/confirmed-qualified checkpoint 在 merged export parity 与 immutable manifest 完成前仍不能充当 parent。Base 和 0.6B checkpoint 也不能充当 parent。Teacher/OPD 若以后启用，额外记录 student rollout/policy version、teacher checkpoint/log-prob、token alignment、distillation mask/objective 和各角色 GPU-hours。

## 配套文件

- [整体进度表](PROGRESS.md)
- [Qwen3.5-4B Day 13+ 迁移计划](QWEN35-4B-MIGRATION-PLAN.md)
- [Scaling Book 与每日三题路线](SCALING-BOOK-READING-GUIDE.md)
- [一手学习资源索引](RESOURCES.md)
- [AutoDL 与 GPU 资源计划](AUTODL.md)
- [每日记录模板](templates/daily-log.md)
- [每周复盘模板](templates/weekly-review.md)
- [实验记录模板](templates/experiment-record.md)
- [Artifacts 说明](artifacts/README.md)
- [Qwen3.5-4B Policy Capstone + Deferred Teacher Extension](SCALED-TEACHER-STUDENT-CAPSTONE.md)
- [Standalone Day 19 Full-SFT comparison](day-19-qwen35-sft-comparison/README.md)
- [Standalone Day 20 balanced LoRA probes](day-20-qwen35-balanced-lora-sft/README.md)
