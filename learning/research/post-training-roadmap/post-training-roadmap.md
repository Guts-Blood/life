# GLM-5.2、Kimi K3、DeepSeek V4 后训练路线图

## Training Report 对照研究、工程 Roadmap 与教学 Walk-through

**研究时点：2026-08-03｜版本：1.0｜语言：中文**

> 本报告只把官方技术报告、官方发布材料和明确写明“已部署”的方法论文作为事实证据。所有数字分为三类：**[D] 官方直接披露**、**[C] 基于官方数字的算术推导**、**[R] 本报告的落地建议**。三者不可混用。

---

## Executive Summary

### 先给结论

1. **三家没有一条共同的“标准后训练 recipe”。** GLM-5 的公开主线是串行塑形；Kimi K3 是 3 个领域 × 3 档 effort 的 9 专家分叉；DeepSeek V4 是按领域和 reasoning effort 训练 specialist，再用超过 10 个 teacher 合并。

2. **共同的战略结构是“先分化，再合并”。** SFT 提供格式与冷启动，RL 把单项能力推高，On-Policy Distillation（OPD）负责把专家能力重新压回一个统一 student，旨在降低顺序 RL 的遗忘和多能力混训的梯度干扰。

3. **长程 Agentic RL 的真正瓶颈不只是算法，而是状态和算账。** GLM-5.2 用 single-rollout critic PPO、learned compaction、token-level accounting；Kimi K3 用 partial rollout、持久 sandbox 和外部 KV pool；DeepSeek V4 用 token-level WAL、可抢占恢复和大规模 DSec。

4. **“reasoning effort”是被训练出来的 policy family，不等于推理时改 `max_tokens`。** Kimi 明确从 max 向 high/low 退火预算；DeepSeek 按不同 context 与 length penalty 训练 Non-think/High/Max；GLM-5.2提供 effort control，但没有公开完整配方。

5. **最想知道的“后训练到底用了多少数据”，三家都没有完整公开。** 没有一家给出可复现的 SFT/RL/OPD 总样本、总 rollout、总生成 token、总步数、总 GPU-hours 和完整 domain mix。报告中最显眼的环境数、网页数、sandbox 数、并发数都不能替代 RL 数据量。

6. **如果要落地，优先复制它们的结构原则，而不是猜一个神秘比例。** 推荐先完成数据账本、可验证环境、trajectory schema 和回归矩阵；中等资源团队先做 3–4 个专家，小团队采用 GLM 式串行 checkpoint + cross-stage OPD，frontier 规模再扩到 9 或 10+ experts。

### 一页比较

| 维度 | GLM-5 / GLM-5.2 | Kimi K3 | DeepSeek V4 |
|---|---|---|---|
| 完整报告状态 | GLM-5 有完整报告；5.2 只有官方增量博客 + 方法论文 | 47 页 K3 technical report | 58 页 V4 Preview technical report |
| 主拓扑 | GLM-5：SFT → Reasoning RL → Agentic RL → General RL → cross-stage OPD；5.2 增加并行 OPD | SFT → 3 域 × 3 effort = 9 RL experts → 专家轨迹 SFT consolidation + MOPD | Domain SFT → specialist GRPO → >10 teachers → full-vocab OPD |
| 长程 RL | 5.2：single-rollout critic PPO/SAO + CompactionRL | partial rollout + pause/resume + staleness regularization | WAL + 可抢占恢复 + DSec；specialist 内仍用 GRPO |
| GRM / reward integrity | GLM-5 General RL 用 GRM/ORM；5.2 anti-hack judge 是 rollout guard，不是主观质量奖励 | tournament-style Agentic GRM + rubric/scorepad | actor-as-GRM + rubric-guided RL；不使用传统 scalar RM |
| OPD 信号 | GLM-5：sampled-token teacher/student log-ratio；5.2 只披露 >10 experts，loss 形式未知 | clipped sampled-token dense reward；9 teachers | student on-policy 的加权 reverse KL；完整词表 logits；>10 teachers |
| 公开后训练量 | 环境池、并发、group/batch、teacher 数；无总 RL 量 | teacher 与 infra 数；无总 RL 量 | teacher 数；无总 RL 量 |
| 最突出亮点 | compaction 进入 policy；token-level credit；在线 anti-hack | effort curriculum；partial rollout；harness augmentation | full-vocab OPD；teacher 调度；WAL/provenance |

---

# 第一部分｜证据边界与阅读方法

## 1.1 三个“report”并不对称

- **GLM-5.2：** 截至研究时点，没有题为《GLM-5.2 Technical/Training Report》的独立完整报告。官方模型卡并列引用 GLM-5.2 blog 与 GLM-5 technical report。因此本报告把 **GLM-5 的完整 pipeline** 与 **5.2 的增量披露**分开，不假定旧 recipe 全部原样继承。
- **Kimi K3：** 有正式 47 页 technical report 和公开权重，但没有开放训练代码、训练数据集或足以完整复现的规模超参。
- **DeepSeek V4：** 有正式 58 页 V4 Preview technical report；截至 2026-08-03 仍为 Preview，报告未披露完整后训练数据量。

## 1.2 什么才叫“RL 做了多少数据”

对 agentic RL，单一“样本数”会严重误导。至少要同时报告：

```text
RL 规模向量 = {
  unique prompt/task seeds,
  environments / repository states,
  attempts per seed,
  completed / accepted trajectories,
  generated policy tokens,
  observation tokens,
  tool calls / interaction steps,
  sandbox-hours,
  optimizer steps / effective batches,
  GPU-hours / FLOPs
}
```

因此：

- 环境数 ≠ rollout 数；
- 来源网页数 ≠ RL prompt 数；
- sandbox 创建数 ≠ accepted episode 数；
- concurrent sandboxes ≠ 总训练数据量；
- 预训练 tokens ≠ 后训练 tokens；
- OPD 用时天数 ≠ OPD 样本数或计算量。

## 1.3 证据等级

| 标记 | 含义 | 示例 |
|---|---|---|
| [D] | 官方材料直接披露 | Kimi 为 9 experts；GLM-5.2 OPD 合并 >10 experts |
| [C] | 只做透明算术，不添加隐含假设 | GLM 32K/128K/200K mid-training 合计 1.55T，占 28.5T 的约 5.44% |
| [R] | 本报告依据三家共性给出的工程建议 | 试点期按生成 token 而非 prompt 数做 domain budget |

---

# 第二部分｜三条后训练 Pipeline

## 2.1 GLM：从串行塑形到长程 single-rollout RL

### 2.1.1 GLM-5 报告中的完整基础流水线 [D]

```text
Serial path:
Overall SFT ──► Reasoning RL ──► Agentic RL ──► General RL

Explicit teacher taps in Figure 5:
Overall SFT   ┐
Reasoning RL  ├──► Cross-Stage OPD ──► Unified final model
General RL    ┘
Agentic RL as a direct teacher: not disclosed
```

来源：GLM-5 Figure 5（p.4）与 §3（pp.10–14）。

| Node | 目标与做法 | 训练信号 | 已披露数量 | 未披露 |
|---|---|---|---|---|
| Overall SFT | General Chat、Reasoning、Coding & Agent；支持 interleaved/preserved/turn-level thinking；错误 execution segment 保留在上下文但 mask loss | token CE；环境验证、rejection sampling、loss masking | 最大 SFT 长度 202,752 tokens | 三类比例、样本、tokens、epochs；5.2 是否同配方 |
| Reasoning RL | 数学、科学、代码、TIR 混训；GRPO + IcePop，处理训练/推理不匹配；无 KL | source-specific judge / evaluation system 的 binary outcome | 四域 roughly balanced；group 32、batch 32；clip low 0.2/high 0.28 | prompts、rollouts、steps、真实比例 |
| Agentic RL | 编码与搜索；trainer/rollout 完全异步；TITO、Direct Double-sided IS、DP-aware routing | 可执行环境、测试和任务成功 | >10K SWE env；数千 repos；9 语言；>1K concurrent rollout | 总 episode、rollout、token、GPU-hours |
| General RL | foundational correctness、情商、task-specific quality | rules + ORM + GRM；human-authored quality/style anchors | 无数据量 | reward mix、样本量、人类偏好量 |
| Cross-stage OPD | Figure 5 明确以 Overall SFT、Reasoning RL、General RL checkpoints 作 teachers；未明确 Agentic RL 是否直接作 teacher | sampled-token teacher/student logit gap，作为 dense token signal | group 1、batch 1024；teacher prompt 来自前序 RL sets，比例仅称 appropriate | teacher mix、prompts、tokens、steps |

### 2.1.2 GLM-5.2 的增量 [D]

```text
GLM-5 family foundation
        ├─ substantially expanded 1M coding-agent training
        ├─ long trajectories → trainable compaction sub-traces
        ├─ group-wise RL → single-rollout critic PPO / SAO
        ├─ online anti-hack: rule recall → LLM intent judge → block tool call
        └─ parallel OPD: merge >10 experts in ~2 days
```

**增量一：1M coding-agent training。** 官方只说 substantially expanded，覆盖大型实现、自动研究、性能优化和复杂 debugging；没有给样本、token、step 或 SFT/RL 占比。

**增量二：Single-Rollout Asynchronous Optimization（SAO）。** 5.2 官方博客明确：长轨迹的 group sampling 会产生 straggler 和 policy lag，因此转向 critic-based PPO，每 prompt 可只采一条 rollout，由 critic 给 token-level advantage。SAO 方法论文还提出 rollout-policy importance ratio、strict double-sided token clipping、冻结 critic attention 与 skip-observation GAE；论文实验中的 critic:actor 2:1 更新比不能当作 5.2 production 超参。

**增量三：CompactionRL。** summary 不再是 inference heuristic，而是同一 trainable policy 的动作。summary tokens 与 execution tokens 共享最终 task reward；token-level loss normalization 避免“压缩次数更多的 trajectory 权重更大”；cross-trajectory GAE 负责跨 compaction segment 分配 credit。方法论文实验默认保留最近 k=2 个 interaction steps，5.2 production 的实际 k 未披露。

**增量四：Online anti-hack。** 规则过滤器先追求高 recall，LLM judge 再识别意图；被判定为作弊的 tool call 被阻断并返回 dummy result，但整条 rollout 继续。这保留了未作弊片段的训练价值。

**增量五：Parallel OPD。** slime 支持 white-box/black-box rollout、compact trajectory 与 sub-agent workflow；5.2 在约两天内合并超过 10 个 expert，但没有公开 teacher 清单、配比、GPU 或 OPD token 量。

### 2.1.3 GLM 的核心设计逻辑

- 短、可验证推理可以依靠 group-relative advantage；长、异长、会 compaction 的 agent 轨迹需要 critic 解决 group size 1 和跨 segment credit。
- TITO、token-level loss、skip-observation GAE 和 cross-trajectory GAE 都指向同一个问题：**长程 RL 首先是一套 token accounting system。**
- GLM-5 的 cross-stage OPD 同时承担能力合并与 anti-forgetting；5.2 的 >10 experts 只能证明最终阶段存在并行专家汇合，expert 来源及其是否由前序串行 checkpoints 演化而来未披露。

## 2.2 Kimi K3：3 域 × 3 effort 的 9 专家系统

### 2.2.1 总流水线 [D]

```text
Native multimodal pretraining / cooldown / 1M context curriculum
                              │
                              ▼
SFT cold start: prior Kimi expert trajectories
multi-stage verification + HITL annotation + XTML schema
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
       General Tasks RL  General Agents RL  Coding Agents RL
              │               │                │
       max → high → low  max → high → low  max → high → low
              └───────────────┼────────────────┘
                              ▼
                         9 experts
                              ▼
       collect expert trajectories → supervised consolidation + MOPD
                              ▼
             Unified Kimi K3: low / high / max
```

QAT 从 SFT 开始贯穿 RL；routed MoE experts 在 trainer 与 rollout 中使用对齐的量化方案。主模型之外，MTP 层被微调为 EAGLE-3-style draft model。

### 2.2.2 节点拆解 [D]

| Node | 方法 | 已披露数量 | 未披露 |
|---|---|---|---|
| SFT | 前代 Kimi domain models 合成长智能体轨迹；multi-stage verification + HITL；XTML 分隔 think/response/tool | 只称 large-scale | 样本、tokens、域比例 |
| 领域分叉 | general tasks、general agents、coding agents | 3 domains | 每域数据量与采样率 |
| Effort curriculum | 从 cold-start 估计每题 b0(x)；超过 τ·b0(x) 的 trajectory reward=-1；先 max，再 anneal τ 得 high/low | 3 efforts；合计 9 experts | τ、hard cap、各档 token 分布 |
| Partial rollout | N prompts × K completions；全局完成 λNK 后暂停剩余生成；只有某 prompt 的 K 条 response 全部完成，该组才进入 optimization；其余跨 iteration 恢复 | N、K、λ 仅为符号 | 实际 batch、iteration、rollout 数 |
| Agentic GRM | tournament-style binary comparison；judge 读产物、生成 rubric、逐项打分并写 scorepad；verbosity 过高会输 | 无 | judge 型号、样本、reward mix |
| Supervised consolidation | 收集 9 专家生成的轨迹，用于 unified student 的 supervised fine-tuning，再进入 MOPD | 来源为 9 experts | 轨迹、token 与领域比例 |
| MOPD | 按 domain+effort 选 teacher；student 自采样；clipped teacher/student log-prob ratio 形成 dense token reward | 9 teachers | Rmax、prompt mix、steps、tokens |
| QAT | routed MoE expert weights 用 MXFP4，其输入 activations 用 MXFP8；attention/latent-MoE projections、shared experts、routers 保持更高精度 | 从 SFT 开始 | 额外成本和数据量 |
| Draft model | 冻结 target，训练 MTP/draft layer 与 fusion；LK acceptance-rate loss | 7-step unroll | 数据量 |

### 2.2.3 为什么 partial rollout 很关键

传统同步 group rollout 必须等最慢的 trajectory。K3 在全局达到 λNK 条 completion 后暂停剩余 generation；只把已经凑齐 K 条 response 的 prompt groups 送入 policy optimization，其余 rollout 保存后跨 iteration 恢复。代价是恢复 trajectory 来自较旧 policy，因而需要 per-token off-policy regularization 限制 staleness。

报告引用 Kimi K2.5 的 policy optimization，而不是重新给出 K3 完整超参。因此 K2.5 公式可以帮助理解机制，但不能把 K2.5 的数值配置写成 K3 的训练规模。

### 2.2.4 Kimi 的训练对象已经是“环境包”

```text
training item = initial state
              + tool / system prompt / memory / skill / subagent harness
              + execution budget
              + evolving environment state
              + public / hidden verifier
              + terminal outcome
```

Unified White-Box RL Environment 可组合工具、system prompt、context management、skills、memories 与 subagents，并实例化 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 风格 harness。Harness 变化本身就是 data augmentation，防止模型只适应固定框架。

## 2.3 DeepSeek V4：Specialist GRPO 与完整词表 OPD

### 2.3.1 总流水线 [D]

```text
DeepSeek-V4 Base
        │
        ├─ Domain SFT: math / code / agent / instruction following / ...
        │        │
        │        ▼
        └─ Specialist GRPO RL
                 ├─ easy-to-verify: rule / test-case verifier
                 └─ hard-to-verify: rubric-guided data + actor-as-GRM
                          │
                          ▼
                    >10 teachers
                          │
                          ▼
 Student on-policy trajectories + weighted full-vocabulary reverse KL
                          │
                          ▼
 Unified V4: Non-think / Think High / Think Max
```

一个必须澄清的误读：**V4 没有取消 RL。** 它仍在每个 specialist 内做 SFT + GRPO；被 OPD 替换的是 V3.2 最终用于能力合并的 mixed RL。

### 2.3.2 节点拆解 [D]

| Node | 方法 | 已披露数量 | 未披露 |
|---|---|---|---|
| Domain SFT | 数学、代码、agent、instruction following 等领域独立 SFT | 无 | 样本、tokens、domain mix |
| Specialist RL | GRPO；训练设置只称与 R1/V3.2 接近 | 无 | group、batch、steps、rollout、tokens |
| Effort specialists | RL 中使用不同 context window 与 length penalty | Non-think / High / Max | 训练 context、penalty、token budget |
| Verifiable reward | simple rule verifier / test cases | 无 | 题量与 rollout |
| Hard-to-verify reward | rubric-guided RL data + Generative Reward Model；不用传统 scalar RM；actor 本身兼任 GRM，GRM 也被 RL 优化 | 人工标注只称 minimal | 标注数、rubric 粒度、GRM mix、独立校准 |
| Multi-teacher OPD | student 自采样 trajectory；加权 reverse KL；完整词表 logits | >10 teachers | teacher 清单、w_i、trajectory、token、step、算力 |
| QAT | MoE expert weights 与 CSA indexer QK path 用 MXFP4；rollout 原生 FP4 | top-k selector 2×，KV recall 99.7% | QAT 数据与后训练成本 |
| Agent infra | DSec、全序 trajectory log、token-level WAL、可抢占恢复 | 单集群几十万 concurrent sandboxes | 总 episode 与 RL 数据量 |

### 2.3.3 Full-vocabulary OPD 为什么不同

DeepSeek 的概念目标为：

```text
L_OPD(θ) = Σ_i w_i · KL( π_student(·|s) || π_teacher_i(·|s) )
```

trajectory 来自 student 本身；在每个 token 位置比较整个词表分布，而不是只比较实际采样 token。优点是更低方差、更完整地保留 teacher 的“次优选项结构”；代价是词表超过 100K 时 teacher logits 成本巨大。

DeepSeek 的工程解法是把 teacher 权重放在集中存储，缓存最后一层 hidden state，需要时只运行对应 teacher LM head 重建 logits，并配合专用 kernel。报告没有给 full-vocab OPD 对 sampled-token OPD 或 mixed RL 的直接完整消融，因而“更稳定”仍需在自己的栈里验证。

### 2.3.4 WAL 解决的不只是可靠性

长 rollout 被抢占后若从头生成，短回复更容易完成并进入训练，造成隐性 length bias。token-level Write-Ahead Log 记录生成 token 并配合 KV cache 从精确位置恢复；独立的 DSec trajectory log 记录环境命令、结果、provenance 并支持 deterministic replay。两套日志共同服务统计正确性与可恢复性，而不只是运维优化。

---

# 第三部分｜横向比较：相同目标，不同技术选择

## 3.1 Pipeline 拓扑

| 问题 | GLM-5.2 | Kimi K3 | DeepSeek V4 |
|---|---|---|---|
| 能力如何分开训练 | GLM-5 先串行 Reasoning/Agentic/General；5.2 另有 >10 experts，但来源未完整重述 | 明确 3 域 × 3 effort | 按 domain + effort 建 specialist，>10 teachers |
| 如何统一 | cross-stage / parallel OPD | 9-teacher MOPD | weighted full-vocab reverse-KL OPD |
| 对遗忘的处理 | 前序 checkpoints 做 teacher | 专家先独立，再统一蒸馏 | specialist learning 与 capability merging 解耦 |
| 适合的组织形态 | 小中团队可串行；5.2 已向并行扩大 | 需要至少 9 个训练分支及环境工厂 | 需要 teacher scheduling 与 logits 基础设施 |

## 3.2 RL 算法选择

| 场景 | 最匹配的公开设计 | 原因 |
|---|---|---|
| 短、同质、可验证问题 | GLM-5 GRPO / DeepSeek specialist GRPO | group-relative baseline 简单，无 critic；同 prompt 多样本可比较 |
| 超长、长度差异大、每 prompt 采样昂贵 | GLM-5.2 single-rollout critic PPO / SAO | group size 1；critic 给 token-level advantage；减少等待和 policy lag |
| 大量长尾 straggler，但希望保留 group-based 更新 | Kimi partial rollout | 完成阈值后先更新，未完成任务跨 iteration 恢复 |
| 强抢占、多租户、超长 agent | DeepSeek WAL + persistent sandbox | 精确恢复，避免重复 rollout 与 length bias |

这些选项并非互斥：可在同一平台上对短推理用 GRPO，对长 agent 用 single-rollout PPO；partial rollout 和 WAL 属于 rollout scheduler/state layer，可以同时存在。

## 3.3 Reward 设计

三家都形成了分层 reward stack：

```text
Level 1  deterministic verifier / tests / final state
Level 2  hidden verifier + anti-hack checks
Level 3  rubric-guided GRM / LLM judge for subjective outcomes
Level 4  cost, verbosity, length or effort-budget constraints
Level 5  OPD dense token signal for consolidation
```

差异在于：

- GLM-5.2 把 anti-hack 放进 rollout 主循环，阻断作弊动作但继续 trajectory。
- Kimi 的 verifier 与具体环境紧密绑定：kernel 正确性/性能、AET final state、web build/pixel、personal-assistant events。
- DeepSeek 让 actor 兼任 GRM，减少独立 RM，但更容易出现 self-evaluation 的相关性错误；落地时应增加独立 judge 与人工校准。

## 3.4 OPD 的三种实现

| 维度 | GLM | Kimi | DeepSeek |
|---|---|---|---|
| teacher 来源 | GLM-5：Overall SFT/Reasoning RL/General RL checkpoints；5.2：>10 experts，来源未披露 | 明确 3 domain × 3 effort | >10 teacher models covering various domains；具体 domain/effort 构成未披露 |
| student 数据 | on-policy | on-policy | on-policy |
| 信号粒度 | GLM-5 为 sampled-token log gap；5.2 parallel OPD 的 loss 未披露 | clipped sampled-token log-ratio dense reward | full-vocabulary weighted reverse KL |
| 主要优点 | 实现较轻；适合 cross-stage retention | teacher routing 清晰；与 RL reward 同框架 | 低方差、保留完整分布结构 |
| 主要成本/风险 | 只看采样 token，信息较少 | staleness、teacher routing、clip 设计 | teacher hidden/logit 调度昂贵；w_i 未公开 |

## 3.5 长上下文的三种“状态观”

- **GLM：把 memory/compaction 变成 policy action。** 固定阈值触发 compaction；模型学习摘要内容以及压缩后如何继续执行，最终任务奖励反传到 summary。
- **Kimi：把未完成 trajectory 视为可暂停的作业。** AgentENV sandbox 支持 pause/resume/fork/snapshot；KV/KDA/MLA state 另行 retention、offload 与 prefetch。
- **DeepSeek：把每个 token 与环境事件视为可重放日志。** WAL 和全序 trajectory log 保证 provenance 与确定性恢复。

它们可以组合成更完整的架构：**policy-level compaction + scheduler-level partial rollout + storage-level WAL。**

---

# 第四部分｜数据比例与 RL 规模披露审计

## 4.1 直接公开的数字

| 模型 | 阶段 | [D] 官方数字 | 正确解释 |
|---|---|---|---|
| GLM-5 | 全基础训练 | 28.5T tokens | pre/mid-training 总预算背景，不是后训练量 |
| GLM-5 | long-context mid-training | 32K: 1T；128K: 500B；200K: 50B | 合计 1.55T；相对 28.5T 的 [C] 约 5.44%，不是官方 data mix 百分比 |
| GLM-5 | issue–PR | 约 10M pairs；160B unique tokens | 数据源/unique token 规模；有 upsampling，不能直接除以总 token 得采样比例 |
| GLM-5 | Reasoning RL | math/science/code/TIR roughly balanced；group 32、batch 32 | “roughly” 只能说近似 1:1:1:1，不能写精确 25% |
| GLM-5 | SWE environments | >10K env，数千 repos，9 languages | 环境池，不是 rollout 总量 |
| GLM-5 | terminal/search | 数千 terminal env；>2M high-information pages | env/source count，不是 RL samples |
| GLM-5 | rollout infra | >1K concurrent rollouts | 并发能力，不是总 rollouts |
| GLM-5 | cross-stage OPD | group 1、batch 1024 | optimizer configuration；无总 prompts/tokens |
| GLM-5.2 | parallel OPD | >10 experts；约 2 days | teacher 数与墙钟时间；无 GPU、tokens、steps |
| Kimi K3 | RL experts | 3 domains × 3 efforts = 9 | teacher count，不能推出每域比例 |
| Kimi K3 | single experiment | few hundred GPUs for co-located 1M-context RL | 单实验资源量级，不是总 GPU-hours |
| Kimi K3 | AgentENV | 51,219,741 sandboxes / 1,505,678 images | training + evaluation 总创建规模；images 是 container filesystem images |
| Kimi K3 | checkpoint/resume | 最低 133ms / 49ms；最高 6.5× memory overcommit | infra latency/density，不是训练量 |
| Kimi K3 | long trajectory | 单条可达数千 tool calls、数百万 cumulative context tokens | 代表性上限，不是均值或总量 |
| DeepSeek V4 | pretraining | Flash 32T；Pro 33T | 预训练 tokens，不是后训练 |
| DeepSeek V4 | token batch | Flash 75.5M；Pro 94.4M | pretraining batch setting |
| DeepSeek V4 | OPD | >10 teachers | 无 teacher mix、trajectory 或 token 量 |
| DeepSeek V4 | DSec | 单集群几十万 concurrent sandboxes | 容量，不是 episode 总量 |

## 4.2 训练数据比例：能答到什么程度

| 问题 | GLM-5.2 | Kimi K3 | DeepSeek V4 |
|---|---|---|---|
| SFT 样本/token 总量 | 未披露 | 未披露 | 未披露 |
| SFT domain 比例 | 三大类，无比例 | 无比例 | 无比例 |
| RL prompt/episode/rollout 总量 | 未披露 | 未披露 | 未披露 |
| RL generated tokens | 未披露 | 未披露 | 未披露 |
| RL 各域比例 | GLM-5 Reasoning 四域 roughly balanced；5.2 未披露 | 只有 3 域结构，无比例 | 只有 domain specialist 结构，无比例 |
| RLVR vs GRM/RLAIF-like 比例 | 未披露 | 未披露 | 未披露 |
| OPD 数据量与 teacher 权重 | 未披露 | 未披露 | 未披露 |
| 后训练 GPU-hours/FLOPs | 未披露 | RL scaling 图无绝对刻度 | 未披露 |

### 核心审计结论

**不能从官方报告还原三家的真实 post-training data mixture，也不能严格比较谁“RL 做得更多”。** 最多可以比较 pipeline 的结构复杂度、环境/系统能力、teacher 数和少数局部训练配置。

## 4.3 容易被误写进报告的数字

- SAO 的 batch 128、group 1、max 128K、约 1,000 steps 与 critic:actor 2:1 更新比来自 Qwen3-30B-A3B 方法实验，不是 GLM-5.2 744B/750B 的 production config。
- CompactionRL 的 50-step critic pretrain、64K/80K context、最多 3 次 compaction 与最近 k=2 步保留设置来自 GLM-4.7-Flash / GLM-4.5-Air 方法实验，不是 GLM-5.2 实际规模。
- K3 第三方 Day-0 LoRA/DAPO 示例的 64 samples/rollout、60 steps、12 小时不是 Moonshot 原始 K3 training recipe。
- DeepSeek-V3.2 的 >1,800 environments、85K prompts 和数千 RL steps不能归给 V4。
- Kimi 的 51M sandbox、DeepSeek 的几十万并发 sandbox、GLM 的 >1K 并发 rollout都属于 infrastructure 指标。

---

# 第五部分｜可执行 Roadmap（12–16 周参考架构）

> 以下均为 **[R] 本报告建议**，不是三家公开的实际训练配方。假设已经有可用 base/cooldown model、一个中等规模训练团队，并希望做 reasoning + coding + agent 的统一模型。

## 5.1 总路线

```text
Phase 0        Phase 1          Phase 2             Phase 3
测量与账本 ─► SFT cold start ─► Specialist RL ───► Long-horizon system
  W0–2          W2–4            W4–9                W5–10
                                                       │
                                                       ▼
Phase 6        Phase 5          Phase 4
持续迭代  ◄── 发布门禁/QAT ◄── Multi-teacher OPD
 W16+           W13–16           W10–13
```

## 5.2 Phase 0：先建测量系统，不先训模型（W0–2）

**交付物**

- 统一 trajectory schema：model messages、thinking visibility policy、tool call/result、environment state、reward components、termination reason。
- 数据账本：origin、license、domain、difficulty、harness version、teacher、verifier、model version、tokens、tool steps、sandbox time、reward、hack flag、compute。
- capability vector 与 retention matrix：reasoning、code、agent、general、safety、latency/cost。
- public verifier 与 hidden verifier 分离；构造 reward-hacking red-team set。

**进入下一阶段的门槛**

- 95%+ 轨迹可 deterministic replay；
- verifier 在人工审计集上的误判率可接受并有 confidence interval；
- 每个样本都能追溯数据、环境、模型与 reward 版本；
- 评估集与训练源严格去重。

## 5.3 Phase 1：SFT 只负责“会说、会调用、会恢复”（W2–4）

**建议起始 domain mix（按有效训练 token 计）**

| Domain | [R] 初始比例 | 目的 |
|---|---:|---|
| reasoning / science / TIR | 25% | 稳定思考、计算与工具推理格式 |
| coding / SWE / terminal | 30% | executable task 与纠错轨迹 |
| agent / search / knowledge work | 30% | 多步工具、状态管理、长任务 |
| general / instruction / style / safety | 15% | 基础可用性、风格与边界 |

这是启动 prior，不是固定真理。每周按 holdout marginal gain、生成 token 成本和 regression 调整，而不是追求某家未公开的比例。

**数据策略**

- 优先 high-quality execution trajectories，而不是堆静态问答。
- 错误动作可保留为上下文，但 mask 对应 loss，让模型学习 recovery 而不 imitation error。
- Tool schema、harness、system prompt、memory 配置做受控随机化。
- QAT/serving precision 如已确定，应从 SFT 开始进入闭环。

**门槛**

- 格式、tool schema、argument typing、parallel tool call 成功率稳定；
- recovery task 明显优于 base；
- SFT 不再带来显著收益时停止，避免把 specialist 能力都寄托于 imitation。

## 5.4 Phase 2：按领域训练 specialist RL（W4–9）

**分支建议**

- 小团队：reasoning → coding/agent → general 的串行 checkpoint，每个阶段保留 teacher snapshot。
- 中等团队：reasoning、coding、agent/search 三个并行 specialist；先做 high/max effort。
- 大团队：再按 low/high/max 分叉，扩到 9 experts；只有在 teacher 增量互补性成立时再扩到 >10。

**算法选择**

- 短、可验证、可做同 prompt 多样本：GRPO，group 4–16 作为试点范围。
- 长、昂贵、异长：single-rollout critic PPO；critic 单独校准并监测 value drift。
- Subjective task：rubric-guided GRM，但 judge 与 actor 解耦，至少加入一个独立模型和 5–10% 人工抽检。

**预算口径**

不要只按 prompt 数配比；至少以 `generated policy tokens × environment cost factor` 做等价负载，并同时报告 tool steps、sandbox-hours 和成功率。

**门槛**

- 每个 specialist 在独立 holdout 上有显著提升；
- 计算边际收益（Δquality / GPU-hour）未进入平台期；
- hidden verifier 与 public verifier gap 不扩大；
- hack rate、verbosity、tool failure 和 cost 均有 guardrail。

## 5.5 Phase 3：长程 rollout 系统（W5–10，与 RL 并行）

**最小组合**

1. persistent sandbox + exact environment version；
2. token/tool-event WAL 与 deterministic replay；
3. pause/resume/fork + KV retention/offload；
4. partial rollout scheduler，记录 policy lag；
5. compaction policy 与最近交互保留策略；
6. token-level loss normalization 与 observation mask；
7. online anti-hack guard，阻断动作但尽量保留 trajectory。

**选择建议**

- 轨迹主要受 context window 限制：优先 learned compaction。
- 轨迹主要受 straggler/环境等待限制：优先 partial rollout。
- 集群抢占或多租户明显：优先 WAL。
- 三者同时存在：组合使用，分别作用于 policy、scheduler、storage 三层。

## 5.6 Phase 4：Multi-teacher OPD（W10–13）

**推荐实验顺序**

1. 先做 3–4 teachers，保留一个 mixed-RL 或 trajectory-SFT baseline；
2. 对比 sampled-token OPD 与 full-vocab OPD 的 capability retention / cost；
3. 建 teacher routing 与权重表，按 domain × effort 显式记录；
4. student 必须自己 rollout，避免只吃 teacher 离线静态数据；
5. 只有当新增 teacher 在 retention matrix 上提供互补能力时扩到 9 或 >10。

**门槛**

- 各 specialist 的主要能力保留率达到预设阈值；
- general/safety 不因合并明显回退；
- OPD 收益相对 GPU-hour 和 serving cost 合理；
- teacher routing 失败案例经过抽样分析。

## 5.7 Phase 5：发布门禁、QAT 与反作弊（W13–16）

- rollout/trainer/serving precision、kernel、sampling mask、tool template 对齐；
- reward-hack suite、hidden verifier、cross-harness evaluation；
- low/high/max 分别评估质量、延迟、token、tool steps、环境失败；
- 线上灰度记录 preserved reasoning/history 缺失、compaction failure、过度主动、无效 debugging；
- 回滚点保留 SFT、每个 specialist、OPD 前后 checkpoints。

## 5.8 一个可落地的后训练算力 prior

| 工作包 | [R] 起始占比 | 调整依据 |
|---|---:|---|
| SFT cold start | 12% | 格式/工具/恢复是否已饱和 |
| specialist RL | 50% | 每域 Δquality / GPU-hour |
| multi-teacher OPD | 23% | retention 增益与 full-vocab 成本 |
| hardening、QAT、ablation、red-team | 15% | hack/回归/部署差异风险 |

这个 12/50/23/15 只是项目管理初值。实际应设置周度再分配，而不是把一次比例写死。

## 5.9 每周 Dashboard

| 类别 | 必看指标 |
|---|---|
| Data | unique seeds、domain/difficulty mix、accepted/rejected、dedup、source coverage |
| Rollout | generated/observation tokens、tool steps、completion rate、pause/resume、policy lag、sandbox-hours |
| Learning | reward、value calibration、clip fraction、KL/OPD loss、gradient/entropy、Δquality/GPU-hour |
| Reward integrity | public-hidden gap、judge disagreement、hack rate、false positive、reward/real-success correlation |
| Capability | domain score vector、retention matrix、cross-harness generalization、low/high/max Pareto front |
| Serving | latency、cost、KV、compaction frequency、tool error、QAT parity |

## 5.10 架构决策表

| 如果你的约束是… | 推荐起点 | 不要急着做… |
|---|---|---|
| 资源较小、团队 <20 人 | GLM 式串行 specialist snapshots + cross-stage token OPD | 一开始训练 9–10+ experts |
| 中等资源、3 个核心能力域 | 3–4 并行 specialists + Kimi 式 teacher routing | 没有 retention matrix 就做大规模 MOPD |
| 长 agent rollout 极昂贵 | group 1 critic PPO + partial rollout + WAL | 强制每 prompt 等 K 条完整轨迹 |
| 有 full-logit/teacher infra | DeepSeek 式 full-vocab OPD 小规模 A/B | 默认认为它一定优于 sampled-token |
| Reward hacking 高 | hidden verifier + online action blocking + independent judge | 只在训练后离线删 trajectory |
| serving 要 FP4 | 从 SFT 开始做 train/rollout/serve parity | 最后一步才量化 |

---

# 第六部分｜教学计划：4 周、8 节、每节 90 分钟

## 6.1 学习目标

完成后，学习者应能：

1. 从一份 technical report 重建后训练 DAG，而不把 benchmark 或 infrastructure 数字当训练量；
2. 解释 SFT、GRPO、PPO/critic、RLVR、GRM、OPD 的角色边界；
3. 为短推理与长 agent 分别选择 rollout/optimization 方案；
4. 比较 sampled-token 与 full-vocabulary OPD；
5. 设计 domain × effort 专家与合并实验；
6. 写出含数据账本、验证器、算力与发布门槛的 roadmap。

## 6.2 课前阅读顺序

1. GLM-5：Figure 5 p.4；§3 pp.10–14；§4 pp.15–20。
2. GLM-5.2 官方博客：1M context、slime、long-horizon RL、anti-hack。
3. SAO §3 与 CompactionRL §4；只理解机制，不把方法实验超参当 5.2 production 配置。
4. Kimi K3：§4.1–4.2 pp.12–17；§5.3 pp.21–22。
5. DeepSeek V4：§4 pp.24–26；§5.1–5.2 pp.28–36。

## 6.3 Session 1｜建立后训练地图

**讲授（35 分钟）**

- Base/mid-training/SFT/RL/OPD 各解决什么问题；
- 数据、环境、rollout、optimizer step、算力五种“规模”口径；
- 为什么 agentic task 的样本是 environment package。

**练习（35 分钟）**

把三份报告的所有数字贴到五列：data source、environment、rollout、optimization、infrastructure。凡不能确定的写“未知”。

**验收（20 分钟）**

能解释为什么 51M Kimi sandboxes、>1K GLM concurrent rollouts、DeepSeek 几十万 concurrent sandboxes都不能回答“RL 做了多少条数据”。

## 6.4 Session 2｜GLM-5 串行 pipeline 与 cross-stage OPD

**讲授**

- Overall SFT → Reasoning RL → Agentic RL → General RL；
- group-relative advantage 的直觉；
- 顺序 RL 为什么造成能力回退，checkpoint teacher 如何修复。

**白板练习**

同一 prompt 的 4 个 reward 为 `[0, 0, 1, 1]`。计算组均值并判断哪些 token 更新方向为正/负；讨论 group 全为 0 时发生什么。

**验收问题**

为什么 group size 1 的 GRPO 不成立，而 OPD 可以 group size 1？

## 6.5 Session 3｜GLM-5.2：SAO 与 CompactionRL

**讲授**

- async rollout 的 straggler 与 policy lag；
- critic、GAE、importance ratio、double-sided clipping；
- observation token 为什么不应像 action token 一样传播 advantage；
- summary 如何成为有 reward 的 policy action。

**Walk-through**

教学例子（沿用方法论文实验的 k=2，并非 5.2 production 参数）：假设任务需要 120K token，而训练窗口 64K；在第 45K token 生成 summary，保留最近 2 个交互，继续执行；最终测试通过。沿 execution → summary → 未来 action 画出 credit path，并解释为何要 token-level normalization。

**验收问题**

CompactionRL 与“运行时让另一个模型做摘要”有何本质区别？

## 6.6 Session 4｜Kimi K3：9 experts 与 effort curriculum

**讲授**

- 3 domains × 3 efforts；
- b0(x)、τ·b0(x)、超预算 reward=-1；
- general task 与 agentic task 的 token 计量口径；
- low/high/max 为什么不是 max_tokens 三档。

**练习**

给 reasoning、coding agent、personal assistant 三类题分别设计 b0 与 cost metric，说明哪些 token/tool call 应计入预算。

**验收问题**

为什么先训 max，再 anneal 得到 high/low，通常比直接截断 max policy 更合理？

## 6.7 Session 5｜Kimi K3：Partial rollout 与环境工厂

**Walk-through**

教学例子（非 K3 真实超参）：`N=4, K=2, λ=0.5`，共 8 条 active rollout。假设最先完成的 4 条恰好组成 2 个完整的 K-response prompt groups，则这 2 组先更新；其余 rollout 保存 sandbox 与 KV-related state，在下一 iteration 恢复。若 4 条分散在 4 个 prompt，则没有完整 group 可直接送入 optimization。画出恢复 rollout 的 behavior policy 与 current policy 差异，解释 staleness regularization。

**环境练习**

为一个 web-development task 写出 initial state、tools、public verifier、hidden verifier、anti-hack、terminal state 和 harness variants。

**验收问题**

为什么 harness diversification 是 data augmentation？为什么 fork、重复生命周期以及 training+evaluation 混合，会使 sandbox count 与 episode count 不存在公开的一一映射？

## 6.8 Session 6｜DeepSeek V4：Specialist、GRM 与 OPD

**讲授**

- “最终 mixed RL 被 OPD 替换”不等于“没有 RL”；
- easy-to-verify 与 rubric-guided hard-to-verify；
- actor-as-GRM 的效率与相关性错误；
- reverse KL 的 mode-seeking 直觉。

**验收问题**

为什么让 actor 自评可能放大系统性盲点？怎样用独立 judge、人工抽检和 hidden verifier 缓解？

## 6.9 Session 7｜OPD 算法诊所

**Toy example**

词表只有 A/B/C。Math teacher 分布为 `[0.8, 0.1, 0.1]`，Code teacher 为 `[0.1, 0.8, 0.1]`。若 student 只采样到 A，sampled-token reward 只看到 A；full-vocab KL 同时看到 B/C 的概率结构。讨论：

- teacher routing 错误会怎样；
- 两个 teacher 权重都为 0.5 是否一定合理；
- full-vocab 信息更多为什么也更贵；
- 如何用 retention matrix 评估能力是否合并成功。

**作业**

设计 3-teacher A/B：trajectory SFT、sampled-token OPD、full-vocab OPD 三组；写清主指标、成本、停止规则。

## 6.10 Session 8｜Capstone：设计自己的 16 周 pipeline

**输入**

一个已有 base model、有限 GPU、目标为 code + research agent + general assistant 的团队。

**必须交付**

- 一张 pipeline DAG；
- 4 个 node 的输入、输出、数据、reward、算法与 gate；
- 一张 data ledger schema；
- 一个 domain/effort budget；
- 一张 retention matrix；
- 一个 reward-hack test plan；
- sampled-token vs full-vocab OPD 决策；
- 12–16 周里程碑和回滚点。

**评分标准**

| 维度 | 权重 |
|---|---:|
| 证据与假设分离 | 20% |
| 节点输入输出清晰 | 20% |
| reward/verifier 可审计 | 20% |
| rollout 与 compute 口径完整 | 15% |
| OPD/retention 实验有效 | 15% |
| 发布门禁与风险控制 | 10% |

---

# 第七部分｜Walk Me Through：把三份报告串成一个系统

## 7.1 从一条 coding-agent trajectory 开始

**Step 1｜SFT 冷启动。** 模型先学会结构化输出、工具参数、观察环境、失败恢复。GLM 的 loss masking 告诉我们：错误动作可以留作上下文，但不要把它当正确 target；Kimi 的 XTML 说明 trajectory schema 本身是 alignment infrastructure。

**Step 2｜选择 RL 节点。** 短测试题可对同一 prompt 采多条并做 GRPO；长 SWE/agent task 若每条耗时不同，group 等待会浪费资源，转向 Kimi partial rollout 或 GLM single-rollout critic PPO。

**Step 3｜奖励分层。** 编译、单测、最终环境状态优先；再加 hidden verifier 抓 reward hacking；审美或开放式产物才交给 rubric GRM。不要让单一 LLM judge 同时决定所有 reward。

**Step 4｜管理上下文。** 若超窗，GLM 路线让 policy 学 compaction；若主要问题是等待和抢占，Kimi pause/resume 与 DeepSeek WAL 保留精确状态。

**Step 5｜训练 effort。** Max policy 先学会解决问题；逐步收紧 token/tool budget，形成 High/Low。评估应比较 quality-cost Pareto，而不是只看准确率。

**Step 6｜能力合并。** Reasoning、code、agent experts 都很强后，不直接混在一次 RL 里互相干扰；让 unified student 自己 rollout，再接受对应 teacher 的 dense OPD 信号。

**Step 7｜发布前闭环。** QAT 与最终 serving kernel进入 rollout；WAL/provenance 使失败可重放；retention matrix 确认合并没有丢能力；anti-hack suite确认 reward 没被投机。

## 7.2 你应该记住的五句话

1. **SFT 教会模型“怎样参与训练”，RL 教会它“怎样赢”。**
2. **长程 RL 的单位不是一条回答，而是一段可恢复、可验证、可计费的环境交互。**
3. **Group-relative RL 适合廉价多样本；昂贵长轨迹需要 critic 或异步恢复。**
4. **OPD 的价值是把能力学习与能力合并解耦。**
5. **看不见的数字就是未知，不要用 infra 数字替它填空。**

---

# 第八部分｜术语表

| 术语 | 一句话解释 |
|---|---|
| SFT | 用监督 target 做 token-level imitation，主要负责格式、基本行为和冷启动 |
| RLVR | 用规则、测试或可执行环境提供可验证 reward 的强化学习范式 |
| GRPO | 同一 prompt 多条 completion 内做 group-relative advantage，通常不需要 critic |
| PPO / critic | critic 估计 value/advantage；适合 group size 1 和长程 credit，但系统更复杂 |
| GAE | 在 bias/variance 间折中地估计序列 advantage |
| Importance ratio | 当前 policy 与 behavior/rollout policy 的概率比，用于修正 off-policy drift |
| GRM | 生成式奖励模型，用 rubric、解释或比较生成评价，而非只输出 scalar |
| OPD | Student 在自己的 on-policy states 上向 teacher 分布蒸馏 |
| Sampled-token OPD | 只在实际采样 token 上计算 teacher/student signal，成本较低 |
| Full-vocabulary OPD | 每个位置比较全词表分布，信息更完整、成本更高 |
| Reverse KL | KL(student ∥ teacher)，更强调 student 高概率区域与 teacher 对齐 |
| Partial rollout | 未等所有 trajectory 完成就更新，未完成项保存状态后跨 iteration 恢复 |
| WAL | Write-Ahead Log；记录生成 token 并配合 KV cache 精确恢复；环境 provenance/replay 由独立 trajectory log 负责 |
| CompactionRL | 把上下文摘要/压缩作为可训练 policy action，用最终任务 reward 监督 |
| QAT | Quantization-Aware Training；在训练中模拟最终低精度部署行为 |
| Retention matrix | 行为 teacher/能力域，列为统一 student checkpoints，用来量化能力合并与遗忘 |

---

# 第九部分｜风险、未知与研究问题

## 9.1 三家共同未知

- SFT、RL、OPD 的完整样本、token、rollout、step、GPU-hour；
- 各 domain 与 reward 类型真实比例；
- safety alignment 是否作为独立节点、使用多少数据；
- OPD 相对 mixed RL、offline distillation、weight merging 的充分消融；
- actor-as-judge 与 teacher self-bias 的跨模型校准；
- long-context agent 在真实用户分布上的失败率和成本。

## 9.2 可直接立项的研究问题

1. 对同一长轨迹任务，SAO、partial rollout GRPO 和 standard async PPO 的 quality/GPU-hour Pareto 如何？
2. learned compaction、external memory 与 larger context 的最优切换点在哪里？
3. sampled-token OPD 何时已足够，full-vocab OPD 的增益来自低方差还是 richer dark knowledge？
4. teacher 数从 3→9→10+ 的增益是否饱和，routing error 怎样增长？
5. actor-as-GRM 如何避免 shared blind spot；是否需要跨模型 judge ensemble？
6. online anti-hack 的 false positive 如何影响探索和能力上限？

---

# 第十部分｜官方来源与页码导航

## GLM

1. [**GLM-5: from Vibe Coding to Agentic Engineering**](https://arxiv.org/abs/2602.15763)，arXiv:2602.15763。重点：Figure 5 p.4；§3 pp.10–14；§4 pp.15–20。
2. [**GLM-5.2: Built for Long-Horizon Tasks**](https://z.ai/blog/glm-5.2)，Z.ai 官方博客，2026-06-16。重点：1M Context、slime、Long-Horizon RL、Anti-Hack。
3. [**GLM-5.2 官方模型卡**](https://huggingface.co/zai-org/GLM-5.2)。
4. [**Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning**](https://arxiv.org/abs/2607.07508)，重点 §3。
5. [**CompactionRL**](https://arxiv.org/abs/2607.05378)，重点 §4。

## Kimi

6. [**Kimi K3: Open Frontier Intelligence**](https://arxiv.org/abs/2607.24653)，重点 §4.1–4.2 pp.12–17；§5.3 pp.21–22。
7. [**Kimi K3 官方仓库**](https://github.com/MoonshotAI/Kimi-K3)。
8. [**Kimi K3 官方模型卡**](https://huggingface.co/moonshotai/Kimi-K3)。

## DeepSeek

9. [**DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**](https://arxiv.org/abs/2606.19348)，重点 §4 pp.24–26；§5.1–5.2 pp.28–36。
10. [**DeepSeek-V4 Preview 官方发布页**](https://api-docs.deepseek.com/news/news260424/)。
11. [**DeepSeek-V4 官方模型集合**](https://huggingface.co/collections/deepseek-ai/deepseek-v4)。

---

## 最终结论

这三份报告最值得学的，不是某个无法还原的“神奇数据比例”，而是后训练架构的三次重构：

- 从单一混合 RL，转向 **domain/effort specialists**；
- 从一次性完整 rollout，转向 **可压缩、可暂停、可恢复、可重放的长程轨迹**；
- 从最后一轮 mixed RL，转向 **student-on-policy 的 multi-teacher capability merging**。

真正可复制的 roadmap 是：**把数据账本与 verifier 建在前面，把能力学习和能力合并拆开，把部署精度与 rollout 对齐，并把“未知”保留为未知。**
