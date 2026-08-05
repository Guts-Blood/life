# 30-Day LLM Post-Training Bootcamp

状态：`in_progress`（Week 1 Day 01–07 与 Week 2 Day 08–10 核心产物已完成；下一执行项为 Day 11）

执行日期：`2026-07-27`（周一）至 `2026-08-25`（周二）  
建议投入：工作日 4–5 小时；周末严格控制为 1 小时 reading/review，不租 GPU  
主线：`数据 -> SFT -> 训练诊断与恢复 -> Preference/DPO -> Online RL -> 可复现训练设计`

## 本月真正要补什么

本月目标是补全 training 判断力，不是构建 auto-train 或 auto-harness 系统。最终要能跟踪一条样本如何变成训练信号、一次训练如何改变模型状态、一次异常如何被证据定位，以及 SFT/DPO/在线 RL 各自需要什么数据和运行时组件。

优先级固定为：

1. 数据契约、模板、tokenization、loss mask、mixture 与 lineage。
2. 训练 step、优化器、batch、checkpoint、resume、复现与失败诊断。
3. Preference data、DPO 与在线 RL 的 rollout/reward/logprob/advantage 数据流。
4. Eval 的冻结、逐样本证据和训练阶段之间的可比性。
5. Sharding、collective 和 Megatron：学到能读配置、判断 OOM/吞吐、理解状态归属和排障；本月不追求手写并行框架。

《How To Scale Your Model》仍贯穿 30 天，但 Day 04 以后按训练问题精选。Scaling 题必须落到 `OOM、throughput、global batch、并行配置或 checkpoint`，不再重复纯公式推导。

## 一个月后的毕业标准

- [ ] 能从 raw sample 追到 rendered text、tokens、labels、loss mask、source/mixture metadata，并识别数据泄漏与模板错误。
- [ ] 能解释 `forward -> loss -> backward -> gradient accumulation -> optimizer/scheduler -> checkpoint/eval`，并用 tiny overfit 验证训练链路。
- [ ] 能设计并运行受控 SFT，对比 checkpoint，完成中断恢复，并判断 resume 是否真的连续。
- [ ] 能根据 loss、grad norm、learning rate、tokens/s、显存、样本输出和 eval 区分数据、优化、系统与评测问题。
- [ ] 能解释 DP/FSDP/ZeRO/TP/PP/CP/EP 切什么、通信什么，以及 global batch 为什么不乘 TP。
- [ ] 能构造与审计 preference pair，解释 DPO objective，并跑通一次小模型 DPO smoke。
- [ ] 能画出 `prompt -> rollout -> reward -> advantage/logprob -> update -> weight sync -> next rollout`，识别 stale rollout、mask/logprob 错位和 reward hacking。
- [ ] 能用 ms-swift 跑 SFT、DPO 和小模型 GRPO，用 slime 跑一次受控在线 RL 闭环；能说明 Tulu/Open-Instruct、TRL、verl 提供的参照。
- [ ] 能从干净环境复现一条最小训练链，并提交一份包含数据、状态、指标、失败处理和扩展边界的 training design。

## 框架各自承担什么

| 角色 | 本月定位 | 是否实跑 |
|---|---|---|
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | 统一的小模型 SFT、DPO、GRPO 实验入口 | Core |
| [THUDM/slime](https://github.com/THUDM/slime) | 理解并运行 Megatron train、SGLang rollout、reward、buffer、weight sync 的在线 RL 闭环 | Core；8×H100 仅 Stretch |
| [AllenAI Open Instruct/Tülu](https://allenai.github.io/open-instruct/) | 参照公开的 post-training stage、数据 mixture 与 recipe | 阅读/对照 |
| [Hugging Face TRL](https://huggingface.co/docs/trl/) | 用紧凑 trainer API 对照 SFT/DPO/GRPO 的输入输出 | 阅读/小型对照 |
| [verl](https://verl.readthedocs.io/) | 对照 actor/rollout/ref/reward/resource-pool 的职责边界 | 阅读/架构对照 |
| [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 建立最小多卡 codepath、rank/state/checkpoint 心智模型 | 最小 smoke，不做全仓通读 |

实操模型优先使用 `Qwen/Qwen3-0.6B-Base`；受控 SFT 资源允许时使用 `Qwen/Qwen3-1.7B-Base`。DPO 必须从已验证的 SFT checkpoint 起步；GRPO 先用最小可运行模型和可验证 reward。所有仓库、模型和数据都固定 revision/commit，不依赖滚动的 `main/latest`。slime 学习基线固定为 `v0.3.0`，运行时必须再核对 tag SHA、官方 examples 和实际 CLI。

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
- [ ] [Day 11 · 08-06 — 一个 SFT step 与 tiny overfit](day-11-sft-step-tiny-overfit/README.md) ← next
- [ ] [Day 12 · 08-07 — 受控 SFT 与 checkpoint 选择](day-12-controlled-sft-checkpoints/README.md)
- [ ] [Day 13 · 08-08（周末 1h）— Qwen/Tülu post-training 阅读](day-13-weekend-sft-reading/README.md)
- [ ] [Day 14 · 08-09（周末 1h）— Week 2 复盘](day-14-weekend-week2-review/README.md)

### Week 3：稳定训练、恢复与诊断（08-10 至 08-16）

- [ ] [Day 15 · 08-10 — Packing、length 与有效 label token ablation](day-15-packing-sequence-ablation/README.md)
- [ ] [Day 16 · 08-11 — Optimizer/LR/warmup/batch 稳定性 ablation](day-16-optimization-stability-ablation/README.md)
- [ ] [Day 17 · 08-12 — Exact checkpoint resume 与可复现](day-17-checkpoint-resume-repro/README.md)
- [ ] [Day 18 · 08-13 — Megatron 最小源码链、双卡 TP/DP 与 distributed checkpoint](day-18-megatron-minimum-codepath/README.md)
- [ ] [Day 19 · 08-14 — 训练诊断与 failure injection](day-19-training-diagnostics-failure-injection/README.md)
- [ ] [Day 20 · 08-15（周末 1h）— Training failure signatures](day-20-weekend-training-failures/README.md)
- [ ] [Day 21 · 08-16（周末 1h）— Eval 与 checkpoint selection 可靠性](day-21-weekend-eval-reading/README.md)

### Week 4：Preference、DPO 与 Online RL（08-17 至 08-23）

- [ ] [Day 22 · 08-17 — Preference provenance、length bias 与 held-out](day-22-preference-data/README.md)
- [ ] [Day 23 · 08-18 — DPO 推导与小模型真实 smoke](day-23-dpo-theory-smoke/README.md)
- [ ] [Day 24 · 08-19 — Online RL dataflow 与 reward/verifier contract](day-24-online-rl-dataflow-reward/README.md)
- [ ] [Day 25 · 08-20 — ms-swift 小模型 GRPO lab](day-25-grpo-small-model-lab/README.md)
- [ ] [Day 26 · 08-21 — slime v0.3.0 主链路与最小运行准备](day-26-slime-codepath-prep/README.md)
- [ ] [Day 27 · 08-22（周末 1h）— GRPO 与 on-policy 边界](day-27-weekend-slime-rl-reading/README.md)
- [ ] [Day 28 · 08-23（周末 1h）— slime debug/replay/repro/observability](day-28-weekend-slime-architecture/README.md)

### Final：真实闭环与训练设计（08-24 至 08-25）

- [ ] [Day 29 · 08-24 — slime 最小闭环、reward 修改与 train-only replay](day-29-slime-rl-run-debugging/README.md)
- [ ] [Day 30 · 08-25 — Post-training design、clean reproduction 与综合口述](day-30-training-design-reproduction/README.md)

## Tracking 规则

1. 开始训练前写清输入数据版本、初始 checkpoint、唯一自变量、成功条件和停止条件。
2. 每个 run 记录 code commit、容器/环境、model/tokenizer revision、dataset manifest、template、seed、完整配置与硬件。
3. SFT 记录 raw/rendered/token/label/mask 审计；DPO 记录 prompt/chosen/rejected；RL 记录 prompt/response/reward/logprob/mask/policy version 对齐。
4. 必须保存逐样本 eval 和 bad cases；aggregate score 不能单独驱动下一轮。
5. Checkpoint 至少区分“仅可推理权重”和“可连续训练状态”；resume 后验证 step、LR、optimizer、数据位置与 RNG。
6. 当天失败要保存最小证据和下一项验证，不为打勾隐藏失败。
7. Day 07、14、21、28 使用 [`templates/weekly-review.md`](templates/weekly-review.md)。

## 配套文件

- [整体进度表](PROGRESS.md)
- [Scaling Book 与每日三题路线](SCALING-BOOK-READING-GUIDE.md)
- [一手学习资源索引](RESOURCES.md)
- [AutoDL 与 GPU 资源计划](AUTODL.md)
- [每日记录模板](templates/daily-log.md)
- [每周复盘模板](templates/weekly-review.md)
- [实验记录模板](templates/experiment-record.md)
- [Artifacts 说明](artifacts/README.md)
