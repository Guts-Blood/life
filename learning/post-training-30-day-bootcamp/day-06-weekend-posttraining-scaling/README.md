# Day 06 — 周末 Reading：Tülu 3 × Applied Training

日期：`2026-08-01`

状态：`done`（完成一手资料对读与纸面决策；未启动 GPU）

强度：1 小时，仅阅读

视觉路线图：[`day-06-posttraining-scaling-decision-roadmap.svg`](../artifacts/reports/day-roadmaps/day-06-posttraining-scaling-decision-roadmap.svg)。

## 主要目标

把“post-training 阶段为什么存在”与“训练是否放得下、跑得完、值得跑”放进同一张图。周末只做对读与判断，不新增代码或 GPU 任务。

## 理论（60 分钟）

精读清单：[Day 06 — Tülu 3 与 Applied Training 对读](../SCALING-BOOK-READING-GUIDE.md#day-06)。

- 25 分钟：读 [Tülu 3](https://arxiv.org/abs/2411.15124) 的 pipeline overview、SFT/DPO/RLVR 阶段和 evaluation/decontamination 总览。
- 25 分钟：读 [Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/applied-training/) 中参数/FLOPs、minimum memory、training time 的代表性例题；先写判断再展开答案。
- 10 分钟：完成一张 crosswalk：

| 决策 | Tülu 3 提供的依据 | Applied Training 提供的约束 |
|---|---|---|
| 是否需要该训练阶段 | 先定义待改善能力与可用信号：SFT demonstration、DPO preference pair、RLVR verifiable reward；每阶段从前序已验证 checkpoint 开始 | 只回答候选阶段能否执行，不替代行为目标与数据判断；先估 peak live memory、总 FLOPs、有效吞吐和 wall-clock |
| 是否扩大规模 | dev 与 unseen/frozen slices 显示收益可泛化，bad cases 没有暴露不可接受的回退 | token/response budget、MFU、通信 roofline 和硬件数量表明扩大规模能缩短时间且不会因小 local batch 变成通信瓶颈 |
| 是否停止 | 目标 slice 无增益、非目标能力退化、污染、length bias、reward hacking 或 KL/response length 异常 | 达到预注册 token/step/time/cost cap，或新增并行只增加通信、无法改善有效吞吐 |

必须回答：

1. SFT、preference tuning、RLVR 各自需要什么数据和上游 checkpoint？
2. memory-feasible、compute-feasible、time/cost-feasible 为什么是三个判断？
3. Scale Your Model 的成本结论如何约束 post-training recipe，而不替代数据与评测决策？

## Training stage × data × state × scaling pressure

| Stage | 数据对象与学习信号 | 上游 checkpoint 与模型角色 | Objective、状态与输出 | 主要 scaling pressure | 进入 / 停止证据 |
|---|---|---|---|---|---|
| SFT | `prompt/messages + target completion`；被 loss mask 选中的 assistant target tokens 提供绝对监督 | 从 pinned Base checkpoint 开始；一个可训练 policy | masked causal-LM loss；更新 policy parameters 与 optimizer/scheduler，保存 RNG、sampler/data progress；输出 SFT checkpoint | 所有 input tokens 都参加 forward；sequence length、micro-batch、activation、label-token ratio 和 total supervised-token budget 决定显存与计算 | 进入前通过 template/token/label audit 和 frozen Base eval；零有效 label、NaN/零梯度、resume 不连续或 dev 回退时停止 |
| Preference / DPO | 同一 prompt 的 `chosen/rejected` responses；相对偏好提供监督 | 从已验证 SFT policy 开始；训练 policy，同时需要固定 reference policy 或预先缓存 reference logprobs | DPO 类 pairwise objective；输出 DPO checkpoint，并保存 pair provenance、policy/reference revision、beta 与逐样本 logprob delta | 每个 pair 含两条 response；policy/reference 的 chosen/rejected forward 增加计算与显存压力。Tülu 3 通过缓存 reference logprobs 避免常驻 reference model，但不能消除 policy 训练成本 | pair 对齐、judge/provenance、length 分布和 held-out eval 通过后进入；length bias、mask/logprob 错位、target gain 不泛化或 general slice 回退时停止 |
| Online RL / RLVR | prompt 经过当前 policy 生成 on-policy rollouts；verifier 对可验证结果给 reward | Tülu 3 最终 recipe 从较强 DPO checkpoint 开始；包含 policy/rollout、fixed reference、value 与 verifier 等角色 | PPO-style reward/KL update；输出 RLVR/final checkpoint，同时保存 rollout、reward、response mask/logprobs、policy version 和训练状态 | autoregressive rollout、response length、每 prompt rollout 数、多个模型角色、policy freshness 与 weight sync 往往比单纯 backward 更昂贵 | 仅在 verifier contract 可单测且目标能力需要在线探索时进入；reward hacking、zero variance、stale rollout、KL/长度失控、平均能力回退或预算 cap 时停止 |

DP/FSDP/TP/PP 只改变参数、梯度、optimizer state、activation 与计算的 ownership/communication；它们不会把 demonstration 变成 preference，也不会把 preference 变成 verifier reward，因此不是新的 training stage。

## 三个必答题

### 1. 三阶段的数据与上游 checkpoint

- SFT 消费 prompt/completion 或 messages/assistant response，从 Base checkpoint 学习被监督 target tokens，产出 SFT checkpoint。
- DPO 消费同一 prompt 下的 chosen/rejected pair，从已验证 SFT checkpoint 训练 policy，并使用固定 reference policy 或缓存的 reference logprobs，产出 DPO checkpoint。Tülu 3 同时使用 off-policy 与由 SFT model 生成的 on-policy preference data。
- RLVR 消费 prompts、当前 policy rollouts 与可执行 verifier/reward；Tülu 3 的最终路线从 DPO checkpoint 开始。报告也比较了从 SFT 开始的情况：训练 reward 可接近，但相同 beta 下 KL 更大，较强起点通常有更好的 test performance。

### 2. 三种 feasibility 为什么不能合并

| 判断 | 问题 | 最小证据 | 可能出现的反例 |
|---|---|---|---|
| Memory-feasible | 单个时刻的 peak live tensors 是否放进总 HBM？ | weights、gradients、optimizer、activations、temporary working set，及其 sharding/checkpointing 假设 | 模型能放下，但每 step 很慢 |
| Compute-feasible | 给定 workload 和 topology，是否有足够有效 FLOPs/带宽执行，且没有被通信 roofline 卡死？ | 总 FLOPs、local tokens、peak FLOPs、HBM/network bandwidth、MFU scenario | 理论能算完，但所需 wall-clock 不能接受 |
| Time/cost-feasible | 在租卡窗口、预算、deadline 内是否值得完成？ | `T ≈ total FLOPs / (devices × peak FLOPs × MFU)`，再乘硬件单价并加入 eval/rollout/checkpoint 时间 | 最低显存拓扑便宜/能跑，但要运行数月或数年 |

Scaling Book 的 LLaMA 3-70B 例子在其特定假设下估算约 225 张 TPU v5p 即可满足最低 memory capacity，但训练约需 1752 天；使用 8960 张 TPU、按 40% MFU 估算则约 44 天。这正说明 memory-feasible 不等于 time/cost-feasible。

### 3. Scaling 结论如何约束 recipe

先由目标行为、数据对象和 frozen eval 决定“需不需要某阶段”；再把该阶段的模型 shape、input/response length、label-token ratio、batch、rollout count 和 token budget 代入 memory/FLOPs/roofline 估算，选择最小可验证 pilot、硬件与停止上限。若估算超预算，优先减少无效 token/response、micro-batch、rollout 数或总 budget，再判断是否需要 FSDP/TP/PP；不能仅因为某种并行能放下就宣布阶段有效，也不能用更低训练成本掩盖数据污染或 eval 回退。

## Coding

无。周末不打开 IDE，不补工作日任务。

## 训练 / 实验

无。

## 资源与租卡

CPU/手机阅读即可；不要启动 AutoDL GPU。

## 验收

- [x] 写出 3 个可复述结论，每条都同时连接“训练阶段”与“规模约束”。
- [x] 标出一个可迁移到 Qwen/H100 的方法，以及一个不能直接迁移的 TPU 数字。
- [x] 留下一个仍不确定、可在 Week 2 用证据回答的问题。

### 三个结论

1. **阶段由信号决定，规模只约束执行。** SFT、DPO、RLVR 分别依赖 demonstration tokens、preference pairs、on-policy verifiable rewards；显存不足时可以改变 batch/sharding，但不能因此替换训练信号或跳过前序 checkpoint gate。
2. **放得下、算得动、按期值得跑是三个独立判断。** Peak HBM 给 memory lower bound，总 FLOPs/roofline 给有效吞吐边界，token/response budget、MFU、卡时与单价才给 wall-clock/cost；任一不通过都先缩小 pilot，而不是直接增加并行维度。
3. **规模扩大必须由冻结评测收益批准。** Tülu 3 用 development 与 unseen evaluation、decontamination 和逐阶段 checkpoint 判断收益；即使硬件允许，目标 slice 无增益、非目标回退、DPO length bias 或 RLVR overoptimization 仍是停止理由。

### 可迁移 / 不可直接迁移

- **可迁移到 Qwen/H100 的方法**：从实际 Qwen config 计算参数/训练 FLOPs 与状态量；为 SFT/DPO/RLVR 分别加入 sequence/response length、label-token ratio、pair/rollout multiplier；使用 H100 的实际 peak/HBM/network 与保守 MFU 情景估算区间，再用短 pilot 测 step time、label tokens/s、peak memory 和通信暴露时间。
- **不可直接迁移的数字**：`225 × TPU v5p` 的最低 memory topology、`8960 × TPU v5p @ 40% MFU ≈ 44 days`，以及 TPU v5p 的 `96GB HBM / 4.59e14 BF16 FLOPs/s` 都绑定 LLaMA 3-70B、4M-token batch、TPU 拓扑和文中 checkpointing 假设，不能用于 Qwen/H100 租卡决策。

### 未决问题

在 `Qwen/Qwen3-0.6B-Base` 上，按 example 数、raw tokens 或 supervised assistant tokens 配置 general/math/code mixture，会产生多大的实际 label-token 比例差异；`mix-B-targeted` 能否在等 supervised-token budget 下改善 target dev slice，而不超过 Day 10 预注册的 general-slice 退化阈值？

Week 2 证据路线：Day 08 做 token/label audit，Day 09 冻结 mixture/lineage，Day 10 冻结 Base eval，Day 11 tiny overfit 验证链路，Day 12 用等 token-budget A/B 回答。

## 一手资料

- [Tülu 3 report（v5，2025-04-14）](https://allenai.org/papers/tulu-3-report.pdf)：阶段 checkpoint、data/object、DPO reference-logprob cache、RLVR 起点/过优化与 development/unseen evaluation。
- [Ai2 Tülu 3 technical overview](https://allenai.org/blog/tulu-3-technical)：五段 recipe、SFT/DPO/RLVR data pipeline、decontamination 与标准化 evaluation 概览。
- [How To Scale Your Model — Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/applied-training/)：参数/FLOPs、最低 memory、MFU wall-clock 与 sharding/communication 题型。

访问日期：`2026-08-03`。
