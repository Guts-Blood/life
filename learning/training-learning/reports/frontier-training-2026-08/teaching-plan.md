# Walk-through 教学计划：真正讲懂三份 Training Report

配套报告：[GLM-5/5.2、Kimi K3、DeepSeek-V4 Training Pipeline Roadmap](roadmap-report.md)
建议形式：8 讲 × 2 小时 + 1 次 3 小时 capstone
目标：不是背模型参数，而是能从一条 sample/trajectory 追到最终 checkpoint，并能解释每个 node 为什么存在、改变了什么状态、有哪些可验证的失败模式。

## 结课标准

完成后应能不看材料做到：

1. 10 分钟内从空白画出三条 pipeline，并区分 pre-training、mid/cooldown、SFT、specialist RL、OPD、deployment-aware training。
2. 准确复述公开数字，同时把“未披露”留空，不把 sandbox/GPU/并发数冒充 RL 数据量。
3. 解释 DSA/IndexShare、KDA/AttnRes、CSA/HCA/mHC 分别解决 token、depth、KV 或 residual 的哪个瓶颈。
4. 从 `prompt → rollout → environment → reward → logprob/advantage → update → weight sync` 跟踪一个 RL sample。
5. 解释 student-generated on-policy batch 为什么不同于 teacher trace SFT，以及 sampled-token OPD 与 full-vocabulary OPD 的取舍。
6. 设计一个有 policy-version、resume、anti-hack、replay 和 matched control 的最小 agent RL/OPD 实验。

## 课前诊断（45 分钟）

不查资料回答六题，答案作为个人 baseline：

1. 预训练的 `raw token`、SFT 的 `supervised token`、RL 的 `environment token` 有什么区别？
2. 为什么同一个 prompt 的 32 条 rollout 不能简单当成 32 条独立训练样本？
3. 1M context 的主要显存对象有哪些？
4. GRPO/group-relative advantage 与 critic-based PPO 的 baseline 从哪里来？
5. OPD 中谁生成 trajectory：student 还是 teacher？
6. rollout server crash 后从头重采样为什么可能产生 length bias？

保留答案；最后一课逐题重答并解释证据。

## 第 1 讲：共同语言与 12-node lifecycle

### 目标

建立三份报告共用的坐标系，先知道“在比较哪个 node”，再进入模型细节。

### Walk-through

```text
source acquisition
  → filtering / dedup / scoring / synthesis
  → tokenizer / packing / mixture
  → base pre-training
  → mid-training / cooldown / context extension
  → SFT cold start
  → domain specialist RL
  → rollout / environment / verifier
  → policy update / weight sync
  → OPD/MOPD integration
  → QAT / speculative draft
  → frozen eval / serving
```

逐 node 回答四个固定问题：

- 输入/输出对象是什么？
- 谁拥有和修改状态？
- 报告给了哪些数量与比例？
- 哪个观测能证明该 node 正常？

### 动手

- 把 `raw documents / packed tokens / SFT trajectories / RL rollouts / OPD batch` 写成五张 object card。
- 对每张卡列出最小 IDs、version、mask、count 与 lineage 字段。

### 验收

能解释为什么“训练数据 30T”不能回答“SFT 和 RL 各做了多少”。

### 对应已有课程

- [Day 05 · Training lifecycle](../../../post-training-30-day-bootcamp/day-05-training-lifecycle-framework-map/README.md)
- [Day 09 · Data mixture 与 lineage](../../../post-training-30-day-bootcamp/day-09-data-quality-mixture-lineage/README.md)

## 第 2 讲：GLM-5 → 5.2，串行 RL 如何演化成专家 OPD

### 必读

- [GLM-5 official report](https://arxiv.org/pdf/2602.15763)：Figure 5；§2.2–2.3；§3.1–3.6；§4.1。
- [GLM-5.2 官方博客](https://z.ai/blog/glm-5.2)：Architecture、slime、long-horizon RL、anti-hack 四节。

### Walk-through 顺序

1. 用 4K 阶段的 18T General + 9T Code & Reasoning，再接 32K/128K/200K 三阶段，还原 base-model timeline。
2. 手算 mid-training 内部的 1T/500B/50B 比例，并解释 rounding caveat。
3. 从 dense MLA 到 DSA：为什么先 warm up indexer，再 joint train；为什么 RL 要 deterministic top-k。
4. SFT 中 interleaved/preserved thinking 与 erroneous-segment masking。
5. reasoning→agentic→general RL 的能力冲突。
6. cross-stage OPD 如何追回 earlier-stage skills。
7. 5.2 为什么因 compaction 改用 critic PPO，以及在线 anti-hack 如何不中断整条轨迹。
8. 从“cross-stage recovery”到“>10 experts merge”的 pipeline 拓扑变化。
9. GLM-5.1 没有独立完整 report：只能确认 multi-turn SFT、RL 与 process-quality evaluation，不能补造 data volume、算法或完整 pipeline。
10. 5.2 的 MTP IndexShare/KVShare、rejection sampling 与 end-to-end TV loss 如何改进 draft；博客里的 acceptance-length ablation 使用 GLM-5.1 backbone/data，不能直接当成 5.2 最终生产超参。

### 动手

- 画 GLM-5 policy/checkpoint timeline：`base → SFT → R-RL → A-RL → G-RL → OPD`。
- 给三个 teachers 和一条 student rollout，逐 token 计算 teacher/student log-ratio 的符号。
- 设计一个可识别 `curl raw target source` 的 anti-hack verifier contract。

### 验收题

“为什么 GLM-5 的 OPD group size 可以设为 1、batch 提到 1024，而 reasoning RL 不行？”

## 第 3 讲：Kimi K3，native multimodal 与 1M agent state

### 必读

- [Kimi K3 official report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)：§2；§3.1–3.4；§4.1–4.2；§5.2–5.3。

### Walk-through 顺序

1. 沿三条轴讲架构：token mixing（KDA/MLA）、depth mixing（AttnRes）、width mixing（Stable LatentMoE）。
2. 为什么 native multimodal from scratch 改变 data loader、load balance 和 pipeline bubbles。
3. 8K→64K→256K→1M curriculum 与 long-dependency synthesis。
4. 3 domains × 3 effort = 9 experts；effort budget `τ × b0(x)`。
5. partial rollout：到 λNK 就更新，unfinished trajectories 跨 iteration resume。
6. stale/off-policy data 为什么需要 per-token policy-drift control。
7. MOPD 如何按 domain+effort 选择 teacher；为什么 top-k objective 可能不一定更好。
8. external KV pool、AgentENV pause/resume/fork 和 QAT parity。
9. 预训练 MTP head 如何转成 EAGLE-3 speculative draft，以及它与主模型 QAT/rollout parity 的关系。

### 动手

- 给 4 条同 prompt rollout，手算 group mean reward；再讨论其中一条跨两个 policy versions 时要额外记录什么。
- 给 teacher/student 对某 action token 的概率，计算 clipped MOPD signal。
- 设计一个跨 Kimi Code/Claude Code/Codex 三种 harness 的同义 tool-use task。

### 特别辨析

- K3 report 没有独立重写完整 policy optimizer 公式，而是继承 K2.5；准确表述是 group-relative outcome signal + per-token likelihood-ratio control，不直接贴“标准 PPO/GRPO”标签。
- 51,219,741 是 training+evaluation 的 sandbox creation count；1,505,678 指 sandbox images，不是视觉图片，两者都不是 RL trajectories。

### 验收题

“如果 sandbox 可以 resume，但 KV prefix 丢了，partial rollout 的系统与统计后果分别是什么？”

## 第 4 讲：DeepSeek-V4，base recipe 与 full-vocabulary OPD

### 必读

- [DeepSeek-V4 official report](https://arxiv.org/pdf/2606.19348)：§2.2–2.4；§4.1–4.2；§5.1–5.2。

### Walk-through 顺序

1. Pro 与 Flash 的 parameter/active-parameter 对照；为什么同时做两个规模点。
2. CSA/HCA：sequence compression、sparse selection、dense compressed attention 如何分工。
3. mHC 与 residual mixing；Muon+AdamW 的参数分工。
4. 32T/33T、4K→16K→64K→1M；Flash dense first 1T 后再 sparse。
5. Anticipatory Routing 与 SwiGLU clamping 如何处理 MoE loss spikes。
6. specialists：domain SFT→GRPO→不同 reasoning effort。
7. mixed RL integration 被 OPD 完全替换意味着什么。
8. full-vocabulary reverse KL、last-hidden-state cache、teacher head scheduling。
9. token WAL 为什么同时解决 preemption efficiency 与 length-selection bias。
10. FP4 expert/CSA-QK QAT 与 native FP4 rollout 为什么属于 post-training correctness，而不只是部署压缩。

### 动手

- 对二分类词表手算 `KL(student || teacher)`；再比较只在 sampled token 上估计 log-ratio 的方差来源。
- 画出 >10 teachers 不同时驻 GPU 时的一批样本调度顺序。
- 写一个 failure injection：rollout 在第 1,000 token 被 preempt；比较 WAL resume 与从头 resample 的样本选择偏差。

### 验收题

“为什么 full-vocabulary OPD 不是简单保存所有 teacher logits？报告用了哪两个内存/调度技巧？”

## 第 5 讲：数据处理与比例——能说什么，不能说什么

### 目标

把 report reading 变成可审计的 quantity ledger。

### Walk-through

| 口径 | 示例 | 可否直接比较 |
|---|---|---|
| Raw documents | repo、网页、PDF、image | 否，长度不同 |
| Unique tokens | GLM issue–PR 160B | 只在 tokenizer/去重口径相同才可 |
| Processed pretrain tokens | GLM 28.5T；Kimi 未披露；DeepSeek Flash 32T / Pro 33T | 可比数量级，不代表质量 |
| SFT supervised tokens | 三家未披露 | 不可比较 |
| RL generated tokens | 三家未披露 | 不可比较 |
| Loss tokens | 必须逐系统记录 trainable mask；tool-call/action tokens 通常可训练，environment/tool-result observations 通常不进 loss | 报告未给 |
| Sandbox/concurrency | 系统 capacity proxy | 不能当数据量 |

### 动手

- 为三家填写 `known / computed / unknown / proxy-only` 四列 ledger。
- 用 GLM mid-stage 数字计算比例；给结果加 rounding 与 denominator 注释。
- 把 Kimi sandbox 数、DeepSeek concurrent capacity、GLM >1K rollouts 分别归到 system capacity，而不是 dataset。
- 复原三家 data-node：GLM 的 DCLM/World Knowledge classifier、math/science synthetic/template filter、code +28% unique-token inventory；Kimi 的 exact/fuzzy dedup、video perceptual hash、quality/structure filter 与 1M long-dependency synthesis；DeepSeek 的 auto-generated/template filtering、multilingual/long-academic curation 与 mid-training agentic data。
- 为每家画 `raw source → filtering/scoring → dedup/synthesis → packing/mask → consumed tokens`，并在报告没有数字的位置明确写 `unknown`。
- 起草一封向模型团队索要 RL disclosure 的技术问题清单。

### 验收题

给十个数字，逐项判定 `dataset size / processed tokens / batch / environment capacity / trajectory upper bound / wall-clock`。

## 第 6 讲：Agentic RL 的真正难点——长尾、off-policy 与环境状态

### Walk-through

1. 同步 group rollout 的 straggler 问题。
2. GLM async、Kimi partial、DeepSeek preemptible 三种解法。
3. policy lag、behavior logprob、current logprob、importance ratio。
4. tokenization mismatch、compaction 和 trainable sub-traces。
5. sandbox crash、timeout、environment failure 与 model failure 的标签分离。
6. KV affinity、external pool、WAL、trajectory replay。

### Lab

构造 8 条 synthetic trajectories：

- 2 条正常完成；
- 1 条 zero reward；
- 1 条 sandbox crash；
- 1 条 timeout；
- 1 条跨 policy version；
- 1 条 compaction 成三段；
- 1 条发现 verifier artifact 并 hacking。

为每条决定：继续训练、mask 部分、resume、drop、重采样或进入 audit；写出原因和统计偏差。

### 对应已有课程

- [Day 24 · Online RL dataflow](../../../post-training-30-day-bootcamp/day-24-online-rl-dataflow-reward/README.md)
- [Day 25 · GRPO lab](../../../post-training-30-day-bootcamp/day-25-grpo-small-model-lab/README.md)
- [Day 28 · slime replay/observability](../../../post-training-30-day-bootcamp/day-28-weekend-slime-architecture/README.md)

### 验收题

“提高 rollout throughput 后 frozen eval 下降，最小排查顺序是什么？”

## 第 7 讲：Reward、verifier 与 anti-hacking

### Walk-through

1. rule/verifier、ORM、GRM、rubric-guided GRM 的信号差异。
2. outcome reward 与过程/稠密 token signal 的角色边界。
3. verbosity/effort budget 如何成为 reward component。
4. public verifier + hidden verifier、isolated judge、intent judge。
5. 为什么 reward 上升可能只是 exploit rate 上升。
6. 在线阻断单个 action 与整条 trajectory rejection 的稳定性取舍。

### Lab

设计一个 GPU kernel optimization task：

- correctness gate；
- performance score；
- hardware roofline normalization；
- CUDA graph replay/input caching/precision reduction 检测；
- hidden cases；
- raw evidence schema；
- reward version 与离线重算测试。

### 验收题

“什么时候应该给 reward=0，什么时候 drop sample，什么时候保留轨迹但 mask 某一段？”

## 第 8 讲：OPD/MOPD 深入与三家统一比较

### 核心推导

从同一 student-generated trajectory 出发，对比：

1. Teacher-trace SFT：teacher 生成 action，student 做 CE；
2. Offline KD：固定数据上匹配 teacher distribution；
3. Sampled-token OPD：student 生成 action，teacher 在 visited prefix 上给 action-token log-ratio；
4. Full-vocabulary OPD：student 生成 prefix，在该位置计算完整 `KL(student || teacher)`。

### Walk-through

- on-policy 到底指哪个 distribution；
- reverse KL 的 mode-seeking 倾向；
- stop-gradient、mask、temperature、clip/top-k；
- 多 teachers 的 routing 与 mixture；
- student/teacher tokenizer、template、vocab compatibility；
- same-model near-zero sanity；
- teacher quality 与 objective quality 的因果混淆；
- matched direct-RL control 与 cost ledger。

### 动手

- 手算一个三 token 词表的 full KL 与 sampled-token estimator。
- 标注 tool/environment tokens 是否参与 distillation loss。
- 分别写出 GLM、Kimi、DeepSeek 的 OPD signal 与系统代价。

### 对应已有课程

- [Day 39 · OPD one-update/replay](../../../post-training-30-day-bootcamp/day-39-opd-one-update-replay/README.md)
- [Day 40 · Controlled OPD](../../../post-training-30-day-bootcamp/day-40-opd-controlled-run/README.md)

### 验收题

“为什么 student 必须生成 trajectory，teacher 只在 student prefixes 上评分，才能称这里讨论的 on-policy distillation？”

## Capstone（3 小时人工设计 + 可选小模型实验）

### 任务

设计一个 `general reasoning + terminal coding` 的双专家/单学生最小 pipeline，不追求 frontier 规模，复刻正确的接口和 gate。

### 必须提交

1. Pipeline DAG；
2. dataset/trajectory/teacher scoring schema；
3. common SFT anchor；
4. reasoning teacher 与 coding teacher readiness gate；
5. matched direct-RL control；
6. sampled-token OPD one-update；
7. tokenizer/template/token alignment test；
8. policy lag、resume、anti-hack failure injection；
9. frozen eval + guardrails + GPU-hour/cost ledger；
10. 允许 `inconclusive` 的 checkpoint selection rule。

### 口述答辩

用 15 分钟回答：

- 哪些 node 来自三家共同收敛？
- 哪些 node 只借鉴某一家，为什么？
- 哪些数字是你的预算假设，而不是 report 事实？
- 最可能产生虚假 gain 的三处是什么？
- 什么时候停止扩规模？

## 建议节奏

| 周 | 内容 | 输出 |
|---|---|---|
| Week 1 | 第 1–2 讲 | common lifecycle + GLM timeline |
| Week 2 | 第 3–4 讲 | Kimi/DeepSeek pipeline cards |
| Week 3 | 第 5–6 讲 | quantity ledger + failure taxonomy |
| Week 4 | 第 7–8 讲 | verifier contract + OPD derivation |
| Week 5 | Capstone | pipeline design review |

如果并入现有 30-Day Bootcamp，不额外挤占训练日：

- 第 1/5 讲叠加 Day 05/09；
- 第 2–4 讲作为 Day 13/20/21 的定向 reading；
- 第 6/7 讲叠加 Day 24–29；
- 第 8 讲与 capstone 叠加 Day 38–41。

## 每讲固定互动模板

1. `5 分钟 retrieval`：不看报告画今天的 node；
2. `25 分钟 guided reading`：只读指定图表/section；
3. `35 分钟 concept walkthrough`：公式与对象对齐；
4. `35 分钟 exercise`：手算、schema 或 failure case；
5. `15 分钟 teach-back`：由学习者讲回；
6. `5 分钟 evidence log`：记录事实、推断、未知与下一项验证。

## 最终评分 Rubric

| 维度 | 通过标准 |
|---|---|
| Pipeline | 三条路线顺序、分叉、merge node 正确 |
| Quantity | 不混用 token/example/trajectory/sandbox/GPU；未知项明确 |
| Algorithm | 能区分 group-relative、critic PPO、sampled-token OPD、full-vocab OPD |
| Systems | 能解释 tail、policy lag、KV、resume、determinism |
| Data | 能定义 mixture denominator、lineage 与 long-context synthesis |
| Safety/correctness | 能设计 anti-hack、failure classification 与 replay |
| Causal judgment | 有 matched control、frozen eval、cost accounting；允许 inconclusive |

## 开始方式

从第 1 讲开始时，只需提出：`开始第 1 讲，先做课前诊断`。每讲按“问题 → 你的回答 → 纠偏 → 报告原文 → 小练习 → teach-back”推进，不一次灌完答案。
