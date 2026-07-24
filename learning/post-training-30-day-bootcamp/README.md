# 30-Day LLM Post-Training Bootcamp

状态：`not_started`  
执行日期：`2026-07-27`（周一）至 `2026-08-25`（周二）  
建议投入：工作日 4–5 小时；周末严格控制为 1 小时 reading/review，不租 GPU  
主线：`Qwen3/ms-swift 闭环 -> Megatron SFT 源码与多卡 -> Eval -> slime RL 源码与运行 -> 30B+ design`

## 为什么这样组织

这不是一份只用于阅读的 syllabus。每一天都必须留下至少一种可复查的证据：配置、日志、计算表、预测结果、Profiler trace、Eval 报告或 runbook。

这是一份连续 30 个自然日的计划。Day 编号已绑定日期：22 个工作日承担理论、coding 和训练，8 个周末日只做 1 小时阅读或复盘。若某个工作日没有完成，先记录阻塞，再压缩后续 Stretch；不要占用周末补长训练。

## 一个月后的毕业标准

- [ ] 能解释并手算 Transformer 参数量、FLOPs 和主要显存项。
- [ ] 能解释 DP、FSDP/ZeRO、TP、PP、CP、EP 的切分与通信。
- [ ] 能独立跑通 Qwen3 全参 SFT、LoRA SFT、checkpoint resume 和推理。
- [ ] 能构建 frozen eval、deterministic scorer、pairwise judge 和置信区间。
- [ ] 能通读 Megatron SFT 的端到端主链路，在双卡上运行、插桩、恢复 checkpoint，并解释每个 rank 的职责。
- [ ] 能通读 slime 的 rollout/train/weight-sync 主链路，在真实多卡上运行并修改 reward 后重跑。
- [ ] 能推导 DPO objective，跑通一次 slime GRPO/RL 最小闭环，并判断 reward 是否被 hack。
- [ ] 能为 Qwen3-32B dense 或 Qwen3.5-35B-A3B-Base MoE 写出可评审的训练设计，并说明与 Qwen3.6-35B-A3B 的差异。
- [ ] 能从干净环境复现最小闭环，并用 15 分钟讲清楚整个项目。

## 每天 README 的固定结构

每一天都必须明确区分：

1. **主要目标**：今天结束时新增的能力或产物。
2. **理论**：公式、论文、系统概念和必须回答的问题。
3. **Coding**：自己实现、读代码、写测试或整理可复现配置。
4. **训练/实验**：要实际运行的 model、dataset、对照和成功标准。
5. **资源与租卡**：CPU/GPU、卡数、显存、预计占用时间、关机条件。

## 固定工作日节奏

| 时间 | 内容 |
|---|---|
| 75–90 分钟 | 理论：阅读、公式推导、回答当天问题 |
| 120–150 分钟 | 工程：训练、代码阅读、实验或工具实现 |
| 45–60 分钟 | 分析：Eval、Profiler、bad case 或容量计算 |
| 20–30 分钟 | Tracking：整理证据、结论、阻塞和下一步 |

长训练放在当天最后启动。第二天先分析结果，再决定是否继续跑，不要在没有 baseline 和验收条件时扩大训练。

周末只安排 60 分钟：45–50 分钟定向阅读，10–15 分钟写 5 条 takeaway 或一页复盘。周末没有 coding、训练和租卡要求。

## 工程路线

- 通用闭环：[modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- SFT infra：[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)；Qwen 适配/转换可借助 Megatron-SWIFT/Mcore-Bridge
- RL infra：[THUDM/slime](https://github.com/THUDM/slime)，训练端 Megatron、rollout 端 SGLang
- 快速 debug：`Qwen/Qwen3-0.6B-Base`
- 全参 SFT：`Qwen/Qwen3-1.7B-Base`
- LoRA smoke：`Qwen/Qwen3-4B-Instruct-2507`
- GRPO：`Qwen/Qwen3-0.6B`，资源允许时升级到 1.7B
- Scaling 推演：`Qwen3-32B` dense 与 `Qwen3.5-35B-A3B-Base` MoE；以 `Qwen3.6-35B-A3B` 作为当前架构参照

版本策略：实际训练使用 Qwen3 的 0.6B/1.7B/4B 成熟阶梯，避免把一个月耗在新模型兼容性上；30B+ 设计使用有 Base checkpoint 的 Qwen3.5-35B-A3B-Base，并对照当前 Qwen3.6-35B-A3B。每次运行须同时固定 model revision、`ms-swift`、Megatron、slime、SGLang commit 和容器 digest，不依赖不断变化的 `main/latest`。

“通读 repo”在本计划中不是逐文件浏览，而是完成三件事：画出带 `file:function` 的端到端调用链；在真实运行日志中证明调用链确实发生；修改一个关键扩展点或观测点后重跑并解释结果。只看 README、只复制命令或只跑出 loss 都不算完成。

所有第三方仓库放在 `vendor/` 或工作区其他位置，不直接修改本目录中的计划文件。实验配置和证据集中放在 [`artifacts/`](artifacts/README.md)。

## 30 天导航

### Week 1：Scaling 基础（07-27 至 08-02）

- [Day 01 · 07-27 — 环境与可复现基线](day-01-environment-baseline/README.md)
- [Day 02 · 07-28 — Transformer 参数、FLOPs、显存](day-02-transformer-accounting/README.md)
- [Day 03 · 07-29 — Roofline 与 H100](day-03-roofline-h100/README.md)
- [Day 04 · 07-30 — 分布式并行地图](day-04-parallelism-map/README.md)
- [Day 05 · 07-31 — Ultra-Scale 与三套框架分层](day-05-ultrascale-code-reading/README.md)
- [Day 06 · 08-01（周末 1h）— Scaling 定向阅读](day-06-weekend-scaling-reading/README.md)
- [Day 07 · 08-02（周末 1h）— Week 1 复盘](day-07-weekend-week1-review/README.md)

### Week 2：跑通 Qwen SFT（08-03 至 08-09）

- [Day 08 · 08-03 — Qwen SFT smoke test](day-08-qwen-sft-smoke/README.md)
- [Day 09 · 08-04 — 数据、chat template、loss mask](day-09-data-and-loss-mask/README.md)
- [Day 10 · 08-05 — Frozen eval 与 Base baseline](day-10-frozen-eval-baseline/README.md)
- [Day 11 · 08-06 — Qwen3-1.7B 全参 SFT smoke](day-11-qwen17b-full-sft-smoke/README.md)
- [Day 12 · 08-07 — Qwen3-1.7B 主 SFT](day-12-qwen17b-main-sft/README.md)
- [Day 13 · 08-08（周末 1h）— Qwen3 技术报告阅读](day-13-weekend-sft-reading/README.md)
- [Day 14 · 08-09（周末 1h）— Week 2 复盘](day-14-weekend-week2-review/README.md)

### Week 3：训练系统与多卡（08-10 至 08-16）

- [Day 15 · 08-10 — Packing 与 sequence length ablation](day-15-packing-sequence-ablation/README.md)
- [Day 16 · 08-11 — Batch 与显存优化 ablation](day-16-batch-memory-ablation/README.md)
- [Day 17 · 08-12 — Megatron SFT 主链路通读与运行准备](day-17-megatron-codepath-prep/README.md)
- [Day 18 · 08-13 — Megatron 双卡 SFT 运行与 runtime trace](day-18-megatron-multigpu-sft/README.md)
- [Day 19 · 08-14 — Megatron checkpoint/Profiler 与 32B 规划](day-19-profiler-dense-capacity/README.md)
- [Day 20 · 08-15（周末 1h）— MoE 阅读](day-20-weekend-moe-reading/README.md)
- [Day 21 · 08-16（周末 1h）— Eval 阅读](day-21-weekend-eval-reading/README.md)

### Week 4：30B+、Eval 与 Preference（08-17 至 08-23）

- [Day 22 · 08-17 — Qwen3.5/3.6 35B-A3B MoE 容量规划](day-22-moe-capacity/README.md)
- [Day 23 · 08-18 — 公共 benchmark 与 deterministic eval](day-23-public-deterministic-eval/README.md)
- [Day 24 · 08-19 — Pairwise、统计与 SFT Eval Report](day-24-pairwise-sft-report/README.md)
- [Day 25 · 08-20 — DPO 理论与 preference data](day-25-dpo-theory-data/README.md)
- [Day 26 · 08-21 — slime 主链路通读与运行准备](day-26-slime-codepath-prep/README.md)
- [Day 27 · 08-22（周末 1h）— GRPO/Online RL 阅读](day-27-weekend-slime-rl-reading/README.md)
- [Day 28 · 08-23（周末 1h）— slime 架构与源码复核](day-28-weekend-slime-architecture/README.md)

### Final：GRPO 与综合设计（08-24 至 08-25）

- [Day 29 · 08-24 — slime 多卡 RL 运行、插桩与 reward 修改](day-29-slime-rl-run-debugging/README.md)
- [Day 30 · 08-25 — 30B+ 设计、复现与模拟汇报](day-30-final-design-reproduction/README.md)

## Tracking 规则

1. 开始一天前，在当天 README 填日期、状态、可用 GPU 和预计时长。
2. 开始训练前，先写假设和成功条件。
3. 所有运行都记录 model、dataset、code commit、seed 和完整 config。
4. 不用“loss 看起来正常”作为结论；至少保存一张曲线或结构化日志。
5. Eval 必须保存逐样本预测，不能只保存 aggregate score。
6. 当天未完成时，写清楚阻塞，不要为了打勾隐藏失败。
7. Day 07、14、21、28 使用 [`templates/weekly-review.md`](templates/weekly-review.md)；Day 21/28 的模板可以只填写阅读相关部分。

## 配套文件

- [整体进度表](PROGRESS.md)
- [Scaling Book 逐日精读路线](SCALING-BOOK-READING-GUIDE.md)
- [学习资源索引](RESOURCES.md)
- [AutoDL 与 GPU 资源计划](AUTODL.md)
- [每日记录模板](templates/daily-log.md)
- [每周复盘模板](templates/weekly-review.md)
- [实验记录模板](templates/experiment-record.md)
- [Artifacts 说明](artifacts/README.md)
