# Scaling Book 与 Training 精读路线

主教材：[How To Scale Your Model](https://jax-ml.github.io/scaling-book/)  
适用范围：30-Day Core（Day 01–30）+ Deferred Capstone Archive（原 Day 31–42）

状态：Day 01–03 已由用户确认完成；从 Day 04 起按本路线继续。

## 阅读原则

《How To Scale Your Model》继续作为系统直觉主线，但不再支配整个 bootcamp。Day 04 以后只在它能回答实际训练问题时回读：

- `OOM`：参数、梯度、optimizer、activation、temporary buffer 到底谁超了？
- `throughput`：compute、HBM、通信、数据加载或 padding 谁在限制？
- `batch`：micro batch、accumulation、DP、有效 label tokens 如何组成 global batch？
- `并行配置`：某种 sharding 解决了什么容量问题，又引入了什么 collective？
- `checkpoint/resume`：保存的是完整训练状态还是仅可推理权重？

本月不要求复现 collective 的精确环形推导、记 TPU 峰值数字或学习 JAX API。公式必须带单位，但 Day 04 后 quiz 不重复纯推导。

## 固定阅读和 Quiz 协议

工作日 reading block 为 60–75 分钟，周末严格 60 分钟：

1. 先看当天三题，写下初始判断。
2. 只读指定的一手材料和与问题直接相关的小节。
3. 回答三题并引用具体 config、sample、log 或 codepath。
4. 留下一项最小验证；不把“继续多读”当验证。

每天 quiz 固定为且仅为三类：

1. **对象/数据题**：对象、字段、shape、版本和变换关系是什么？
2. **状态/训练题**：哪些状态被读取、修改、聚合、保存或恢复？
3. **诊断/判断题**：面对异常或方案选择，依据什么证据判断，下一项最小验证是什么？

Daily Log 至少留下：三题答案、一个工程映射、一个证据路径、一个未解决问题。

## Week 1：Scaling 基础与 Training 全景

<a id="day-01"></a>

### Day 01 — 术语、上下界与可复现环境

阅读：[Introduction](https://jax-ml.github.io/scaling-book/) 的全书地图；[Rooflines](https://jax-ml.github.io/scaling-book/roofline/) 的 `Where Does the Time Go?` 与 roofline 图。

- **对象/数据题**：一次 run 的 code、model、tokenizer、dataset、config、environment、seed、hardware 分别用什么不可变标识？
- **状态/训练题**：哪些状态只影响复现，哪些会直接改变下一次 optimizer update？
- **诊断/判断题**：模型能放入显存但 step 很慢时，先收集哪三类证据区分 compute、HBM 和数据问题？

当日落地：环境与 run identity 清单。历史完成状态不意味着补写不存在的 artifact。

<a id="day-02"></a>

### Day 02 — Transformer 参数、FLOPs 与训练状态

阅读：[Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) 的 `Counting Dots`、forward/reverse FLOPs、Transformer accounting 与 global params/FLOPs。

- **对象/数据题**：从 Qwen config 如何得到 embedding、attention、MLP、norm 与 lm_head 的 shape 和参数量？
- **状态/训练题**：BF16 全参训练中 weight、gradient、FP32 master weight、Adam moments 和 activation 各何时存在？
- **诊断/判断题**：估算参数量正确但实测显存差很多时，按什么顺序检查 activation、temporary buffer、allocator 和 checkpointing 假设？

当日落地：基于 config 的模型与训练状态 accounting。

<a id="day-03"></a>

### Day 03 — Roofline 落到 H100

阅读：[Rooflines](https://jax-ml.github.io/scaling-book/roofline/) 的 matmul/network roofline；[GPUs](https://jax-ml.github.io/scaling-book/gpus/) 的 GPU、memory 与 hardware specs。

- **对象/数据题**：一次 SFT step 中哪些 tensor bytes 经过 HBM，哪些 bytes 经过 GPU 间链路？
- **状态/训练题**：micro batch、sequence length 与 accumulation 如何改变 local tokens、activation 和单次 update？
- **诊断/判断题**：`GPU utilization=99%` 但 tokens/s 低，哪些观测可以判断 MFU 低、padding 高或通信等待？

当日落地：H100 worksheet 与后续实测所需字段。

<a id="day-04"></a>

### Day 04 — Sharding 与 collective 的因果链

阅读：[Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/) 的统一记号、四类 matmul、AllGather/ReduceScatter/AllReduce/AllToAll。跳过 JAX API 和精确 ring 成本推导。

- **对象/数据题**：给定 `A[I,J]B[J,K]`，每个 rank 实际持有哪些 local shards，输出是完整块还是 partial sum？
- **状态/训练题**：AllGather、AllReduce、ReduceScatter、AllToAll 分别改变复制、切分、未规约状态中的哪一项？
- **诊断/判断题**：遇到 OOM 或多卡吞吐下降时，如何从 global/local shape、collective tensor bytes 和期望输出布局判断是分片不足还是通信过重？

当日落地：把四种 collective 各映射到 DDP、FSDP/ZeRO、TP 或 MoE 的一个真实场景。

<a id="day-05"></a>

### Day 05 — Training lifecycle 与框架职责

阅读：[Training](https://jax-ml.github.io/scaling-book/training/) 中 DP/FSDP/TP/PP 的概念段落；[ms-swift](https://github.com/modelscope/ms-swift)、[Open Instruct](https://allenai.github.io/open-instruct/)、[slime](https://thudm.github.io/slime/) 和 [verl](https://verl.readthedocs.io/) 的首页/架构入口。

- **对象/数据题**：SFT、DPO、在线 RL 各自消费什么样本对象，产出什么 checkpoint 和评测对象？
- **状态/训练题**：training repo、训练 backend、rollout engine、通信库和 CUDA 各自决定什么，不决定什么？
- **诊断/判断题**：一个 run 失败时，如何先按 data/framework/backend/runtime/hardware 分层，而不是直接归因 CUDA？

当日落地：`data -> collator -> model -> loss -> backward -> optimizer -> checkpoint -> eval` 图，以及框架 ownership 表。

<a id="day-06"></a>

### Day 06 — 周末：Post-training Stage 与 Scaling 对读

对读：[Tülu 3](https://arxiv.org/abs/2411.15124) 的 SFT→DPO→RLVR pipeline 总览，以及 [Applied Training](https://jax-ml.github.io/scaling-book/applied-training/) 的 memory/compute/time feasibility 题型。目的不是复制 LLaMA 数字，而是判断不同训练 stage 为什么有不同的数据、状态与资源预算。

- **对象/数据题**：SFT、DPO、RLVR 分别消费什么数据对象；估算每阶段资源还需要 model shape、sequence/response length、label-token ratio 和 batch 中哪些字段？
- **状态/训练题**：三个阶段的起始 checkpoint、训练中模型角色和需要保存的状态有什么不同；DP/FSDP/TP 只改变哪些 ownership？
- **诊断/判断题**：面对一个阶段的 OOM/高成本，应先缩短数据/response、调 batch，还是引入新的并行维度？依据是什么？

当日落地：`training stage × data × state × scaling pressure` 对照表。

<a id="day-07"></a>

### Day 07 — 周末：Week 1 复盘

不读新章节；回看 Day 01–06 的 config、计算与问题。必要时只回查 [Scaling Book](https://jax-ml.github.io/scaling-book/) 对应段落。

- **对象/数据题**：选一个未来 SFT run，列全输入对象、版本和最小证据。
- **状态/训练题**：口述一个 update 中状态变化，并指出 DP/TP size 在 global batch 公式中的不同角色。
- **诊断/判断题**：给出 OOM、低 throughput、loss 不降各自第一项最小验证，说明为什么。

当日落地：一张 `Base -> SFT -> preference/DPO -> online RL/RLVR -> eval` training-stage decision map，标明每条边何时进入、凭什么退出；再保留三项需要在真实训练中证伪的假设。

## Week 2：数据契约与受控 SFT

<a id="day-08"></a>

### Day 08 — SFT 数据契约与 loss token

阅读：[ms-swift Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) 的 messages/schema、template 与 loss；[Open Instruct dataset transformations](https://allenai.github.io/open-instruct/algorithms/dataset_transformation/) 的 tokenize/filter 示例。

- **对象/数据题**：一条 raw messages 如何变成 rendered text、input IDs、labels、attention mask 与 source metadata？
- **状态/训练题**：template、truncation、EOS、padding 和 `-100` mask 分别在哪一步改变可学习 token？
- **诊断/判断题**：loss 正常下降但模型复读 user/system 内容时，怎样用一条 sample 的 token-level 审计定位问题？

当日落地：至少一条样本的五列对照，不启动长训练。

<a id="day-09"></a>

### Day 09 — 数据质量、mixture 与 lineage

阅读：[Open Instruct dataset transformations](https://allenai.github.io/open-instruct/algorithms/dataset_transformation/) 的 mixer/filter/cache；[Tülu 3](https://arxiv.org/abs/2411.15124) 的 SFT data mixture 与 decontamination 相关部分。

- **对象/数据题**：dataset manifest 如何记录 source、revision、license、split、过滤规则、样本数、token 数和 hash？
- **状态/训练题**：mixture weight、shuffle、dedup、filter 与 truncation 如何改变模型在一个 epoch 实际看到的分布？
- **诊断/判断题**：某能力上涨但 frozen benchmark 异常暴涨时，如何检查 contamination、重复和 source 过采样？

当日落地：数据审计表、长度/source 分布和一版可重建 mixture manifest。

<a id="day-10"></a>

### Day 10 — Frozen eval 与 Base baseline

阅读：[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 的 task、few-shot、generation 与 sample logging；[Inference](https://jax-ml.github.io/scaling-book/inference/) 的 prefill/generation 基础。

- **对象/数据题**：frozen eval 要冻结哪些 sample IDs、prompt/template、decoder、stop、scorer 和 model revision？
- **状态/训练题**：eval 时哪些模型状态必须只读，哪些缓存或随机状态会让两次结果不可比？
- **诊断/判断题**：aggregate score 变化时，如何用逐样本 prediction 区分能力变化、抽取器错误和 decoding 漂移？

当日落地：训练前 Base checkpoint 的逐样本 baseline。

<a id="day-11"></a>

### Day 11 — 一个 SFT step 与 tiny overfit

阅读：[ms-swift command parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md) 中 batch、precision、optimizer、loss、logging；[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 作为输入/输出对照。

- **对象/数据题**：一个 batch 中总 tokens、non-padding tokens、label tokens 与 samples 分别是多少？
- **状态/训练题**：一次 accumulation window 内 gradient、optimizer step、scheduler step、global step 各更新几次？
- **诊断/判断题**：tiny dataset 无法快速 overfit 时，按 label mask、LR、冻结参数、gradient、data repeat 的什么顺序排查？

当日落地：同一小批样本的前向、反向、更新和生成闭环。

<a id="day-12"></a>

### Day 12 — 受控 SFT 与 checkpoint 选择

阅读：[ms-swift command parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md) 的 `max_steps/epochs`、warmup、save/eval、best model 与 resume 参数；回读 [Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) 的训练 FLOPs 近似用于预算。

- **对象/数据题**：一次受控 run 的 train/validation/frozen eval、token budget 和 early/middle/final checkpoint 如何绑定？
- **状态/训练题**：每个 checkpoint 应包含哪些 model、optimizer、scheduler、scaler、RNG、progress 与 data-position 状态？
- **诊断/判断题**：final train loss 最低但 frozen eval 退化时，如何选择 checkpoint，并排除 eval noise？

当日落地：预注册一个主 SFT，只改变一个明确变量并保存 checkpoint 对比。

<a id="day-13"></a>

### Day 13 — 周末：Qwen3 历史证据 × Qwen3.5 迁移阅读

先把 [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 当作 Day 01–12 的历史背景，再读 [Qwen3.5-4B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B-Base)、[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5) 与 [ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)。重点是迁移边界，不抄 benchmark 大表。

- **对象/数据题**：Day 08–12 的 raw sample/provenance/scorer 中哪些可重新审核，哪些 rendered text、token IDs、label spans、schedule 和 comparison key 必须作废重建？
- **状态/训练题**：`Qwen3_5ForConditionalGeneration`、processor、vision tower、aligner、GDN/full-attention backend 与旧 Qwen3 CausalLM runner 的状态边界有什么不同？
- **诊断/判断题**：为什么“同属 Qwen”不能证明 tokenizer/template、loader、LoRA target、packing 或 resume 兼容？每项最小证据是什么？

当日落地：一页 `v1 immutable history -> v2 migration gates -> S0/S1/S2` lineage/差异 memo；周末不租 GPU。

<a id="day-14"></a>

### Day 14 — 周末：Week 2 历史复盘与 Qwen3.5 Readiness

不读新章节；回看 Day 08–12 的样本审计、Base predictions、Day 11 pass、Day 12 recovery C–L 与 [`QWEN35-4B-MIGRATION-PLAN.md`](QWEN35-4B-MIGRATION-PLAN.md)。

- **对象/数据题**：随机抽一条 v1 样本，能否追到 source/loss token；迁到 v2 时哪些 immutable raw IDs 与哪些 model-specific artifacts 要分别登记？
- **状态/训练题**：为什么 Day 12 的结果是“合法选择 none”而不是未完成；为什么任何 0.6B checkpoint 都不能成为 4B `S1`？
- **诊断/判断题**：M0–M6 各自需要谁产出什么证据、何时 fail closed；哪一项没过会阻塞 Day 16/23/25？

当日落地：完成 Week 2 v1 复盘和 Qwen3.5 readiness table；不选择 teacher，不启动 GPU。

## Week 3：稳定训练、恢复与诊断

<a id="day-15"></a>

### Day 15 — Qwen3.5-4B Onboarding 与迁移验收

阅读：[Qwen3.5-4B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B-Base)、[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、ms-swift best practice，以及 [Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) 的 activation/context 预算。今天不做完整 packing ablation。

- **对象/数据题**：exact revision、完整 shards/index、processor/tokenizer/template/special tokens、重新分词后的 train/dev/confirmation manifests 如何组成 v2 identity？
- **状态/训练题**：完整 multimodal checkpoint 中哪些模块加载、哪些冻结、哪些挂 LoRA；GDN/full-attention kernel、MTP、dtype 和 checkpoint/export 如何记录？
- **诊断/判断题**：如何用 pure-text forward/generate、golden token/mask、Base baseline、one-step、tiny-overfit、fresh resume 与峰值显存逐级排除“能加载但不能训练”？

当日落地：完成 M1–M5。exact revision、环境 lock、processor/template、retokenization、新 Base baseline、tiny-overfit/resume/export 和 10–15% 显存余量缺一不可；否则 Day 16 blocked。

<a id="day-16"></a>

### Day 16 — Controlled Coding LoRA SFT、Packing Parity 与 S1 选择

阅读：[ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md) 的训练参数和 packing 约束；只回查 [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 与 AMP 作为更新语义参照。

- **对象/数据题**：unpacked baseline 与受约束 packing candidate 如何保持 raw IDs、processor、label-token budget、order、eval inputs 和 parent S0 一致？
- **状态/训练题**：LoRA 覆盖哪些 text modules、如何证明 vision/aligner 冻结；checkpoint 如何同时保存 processor、adapter/full state、optimizer/RNG 与 BF16 export？
- **诊断/判断题**：code 改善但 retention/termination/sandbox error 退化时，怎样拒绝候选；packing tokens/s 上升时如何验证 attention boundary、position/mask 与输出 parity？

当日落地：从 S0 跑受控 coding LoRA/QLoRA SFT，只做一个小型 packed/unpacked semantic/parity 对照，并冻结 provisional candidate set 或记录 `no-candidate`。Day 17 补 resume evidence，Day 21 才最终晋级 `S1`；学习率与 token budget 不继承 0.6B。

<a id="day-17"></a>

### Day 17 — Exact checkpoint resume 与复现

阅读：[PyTorch Saving and Loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) 与 [ms-swift FAQ](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md) 的 resume。

- **对象/数据题**：selected Qwen3.5 config 的 checkpoint manifest 如何关联 shards、config、processor/tokenizer、freeze/module coverage、dataset/data position 与代码版本？
- **状态/训练题**：model、optimizer、scheduler、scaler、RNG、global step、sampler/dataloader state 中漏哪项会怎样？
- **诊断/判断题**：resume 后 loss/LR/样本顺序跳变时，如何用“连续 run vs 中断恢复 run”最小对照定位？

当日落地：使用 Day 16 选定配置比较连续 40-step 与 20-step 中断后新进程恢复到 40-step，逐步核对 sample ID、LR、loss、model/adapter/optimizer/scheduler/RNG/dataloader state 与 processor/export parity。

<a id="day-18"></a>

### Day 18 — Megatron Minimum Codepath 与 Qwen3.5 兼容 Gate

阅读：[Megatron Core first training run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)、[parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) 与 [distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)；回读 Scaling Book Training 对应概念。

- **对象/数据题**：最小 run 中 processor output、GDN/full-attention layer、model/adapter shard、gradient buffer 和 checkpoint shard 由哪些 rank 持有？
- **状态/训练题**：TP/DP process groups、Qwen3.5 conditional-generation/MCore mapping、forward/backward、collective 与 optimizer step 怎样连接？
- **诊断/判断题**：2 卡 smoke OOM、hang、loss 分叉或 export 失败时，如何区分 unsupported architecture、batch/state replication、collective/group 与 backend kernel？

当日落地：一条能在日志中验证的最小 Qwen3.5 codepath；支持时做 `TP=1/DP=2`、`TP=2/DP=1` smoke 和 distributed checkpoint reload/export parity。不支持时保存明确 blocker，不换模型伪造通过。

<a id="day-19"></a>

### Day 19 — Optimizer/LR 稳定性与 Failure Injection

阅读：[Scaling Book Profiling](https://jax-ml.github.io/scaling-book/profiling/) 的 trace/memory profile；[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) 与 [autograd anomaly detection](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)。

- **对象/数据题**：failure report 必须绑定哪个 batch/sample、run/config、rank、step、checkpoint 与 trace window？
- **状态/训练题**：在同一 Qwen3.5 SFT baseline 上改变 LR/warmup/effective batch，或注入坏 mask、错误 resume/processor 与慢 data source 时，预期哪些状态和指标首先变化？
- **诊断/判断题**：怎样用最小复现把 loss 问题分到 data/objective/optimization，把慢分到 input/compute/communication/checkpoint I/O？

当日落地：先完成 AdamW baseline、2×LR、无 warmup、2×effective batch 的一变量短对照，再做高 LR、错误 mask/processor、错误 resume、吞吐退化的最小 failure injection 与恢复 runbook。

<a id="day-20"></a>

### Day 20 — 周末：训练失败案例复盘

阅读：[PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)、[NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) 与 ms-swift FAQ 中和 Qwen3.5 的 OOM/resume/packing/GDN backend 直接相关的条目。

- **对象/数据题**：每个 failure case 缺失或损坏的对象是什么，如何在 run 前验证？
- **状态/训练题**：OOM、NaN、hang、resume drift 分别涉及哪些状态边界？
- **诊断/判断题**：为四类失败各写“症状 -> 证据 -> 最小验证 -> 停止条件”，避免只列可能原因。

当日落地：一页 failure signature 表。

<a id="day-21"></a>

### Day 21 — 周末：Eval 与 checkpoint selection 可靠性

阅读：[LM Evaluation Harness docs](https://lm-evaluation-harness.readthedocs.io/) 的 reproducibility、sample logging 与 task guide；回读 v2 Qwen3.5 Base/SFT predictions。Day 10/12 的 0.6B candidates 只作历史诊断，不进入 v2 candidate set。

- **对象/数据题**：capability、style、safety、regression 各需要什么样本和 metric，哪些不能混成一个总分？
- **状态/训练题**：checkpoint、template、decoder、judge/scorer version 哪些必须冻结才能比较训练阶段？
- **诊断/判断题**：如何在看结果前冻结 candidate set、primary metric、guardrails、最小差异、CI、tie-breaker 与 `inconclusive` 条件？

当日落地：一版 checkpoint-selection policy 和一次盲化 rehearsal；独立 held-out confirmation 不参与反复调参。

## Week 4：Preference、DPO 与 Online RL

<a id="day-22"></a>

### Day 22 — Coding Preference Provenance、Processor 与 Held-out

阅读：[Open Instruct synthetic preference dataset](https://allenai.github.io/open-instruct/algorithms/synthetic_preference_dataset/)、[TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats) 与 Tülu 3 的 preference data 部分。

- **对象/数据题**：coding prompt/chosen/rejected、tests/sandbox、generator、judge、score/margin、source/license/creation method、processor hash 与 policy version 如何组成可追踪 pair？
- **状态/训练题**：pair 过滤、顺序交换、长度控制和 reference policy 选择如何改变 DPO 训练信号？
- **诊断/判断题**：如何发现 chosen/rejected 反转、模板不一致、近重复、judge length/style bias，以及 prompt/source-family 跨 split 泄漏？

当日落地：在 Qwen3.5 processor/template 下重建 preference schema，完成 50-pair 盲审、execution/length/source slices 和 group-held-out；不复用 v1 token IDs。

<a id="day-23"></a>

### Day 23 — Qwen3.5 Coding DPO 理论与 Smoke

阅读：[DPO paper](https://arxiv.org/abs/2305.18290) 的 objective 与 assumptions；[TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) 或 ms-swift 官方参数作为实现对照。

- **对象/数据题**：chosen/rejected 的 policy/reference token logprob 怎样按同一 prompt、template 和 mask 对齐？
- **状态/训练题**：为什么 policy/reference 必须共同从 promoted `S1` 派生；DPO update 修改什么，reference 是否更新，beta 如何改变 preference margin？
- **诊断/判断题**：DPO loss/accuracy 变好但生成退化时，如何检查 pair quality、长度偏置、KL 漂移和 frozen eval？

当日落地：只有 `S1` 存在才运行同一小型 coding preference set 的 loss 单元检查与 5–10 step smoke；Base 或 Day 12 export 不能代替 parent。

<a id="day-24"></a>

### Day 24 — Coding Online RL Dataflow 与 Sandbox Reward Contract

阅读：[slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) 的 rollout/train batch 关系与 reward 配置；[verl PPO architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html) 作为角色边界对照。

- **对象/数据题**：coding prompt、grouped responses、tokens、tests、sandbox result/error/timeout、response mask、reward、old/ref/current logprob、advantage、processor 与 policy version 如何关联？
- **状态/训练题**：rollout policy、train policy、reference、reward/verifier、buffer 和 weight sync 在一轮中怎样变化？
- **诊断/判断题**：reward 上升时，哪些独立证据才能排除 length hacking、格式投机、stale rollout 和 mask/logprob 错位？

当日落地：带字段、数量、producer/consumer 的在线 RL 数据流图；实现 CPU 隔离、无网络、资源限额、可重算、可版本化且带 timeout/error taxonomy 的 coding verifier contract。LLM judge 不默认启用。

<a id="day-25"></a>

### Day 25 — Qwen3.5-4B Coding GRPO Lab

阅读：[ms-swift GRPO](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html) 与 [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) 的数据/reward/config；Scaling Book 只回读 inference 的 KV cache 与 generation throughput。

- **对象/数据题**：每个 prompt 产生多少 completions，reward 如何绑定 response，group 内 advantage 如何生成？
- **状态/训练题**：每轮采样和 update 之间哪些 policy/logprob/optimizer 状态必须一致，哪些可以重算？
- **诊断/判断题**：zero reward variance、OOM、生成过长、sandbox failure 或 KL 快速增长时，首先检查 G、max completion/model length、colocate/topology、reward 与 mask 中哪一项？

当日落地：仅从 `S1` 跑 5–10 step GRPO smoke；显式将 rollout model length 先 cap 到 8K（通过新 gate 后最多 12K），记录每卡 allocated/reserved、G、KL/reward/length、policy version 与逐样本 sandbox 表。官方无 NVIDIA coding-GRPO 峰值可照抄，必须实测。

<a id="day-26"></a>

### Day 26 — slime 固定 Release 的 Qwen3.5 兼容 Gate

阅读：[slime Releases](https://github.com/THUDM/slime/releases)、[Architecture](https://thudm.github.io/slime/blogs/introducing_slime.html)、[Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) 和 [Customization](https://thudm.github.io/slime/get_started/customization.html)。先把 `v0.3.0` 固定为源码阅读基线，再核对是否有明确 release 支持 Qwen3.5 的 load→rollout→train→weight-sync；不从滚动 main 猜测兼容。

- **对象/数据题**：slime `Sample`/buffer、Megatron batch 与 SGLang request/response 之间如何转换？
- **状态/训练题**：Ray placement、rollout engine、trainer、reward 和 weight sync 各拥有何种 GPU/模型状态？
- **诊断/判断题**：在租多卡前，哪些 processor/model mapping、Megatron/SGLang loader、checkpoint conversion、CPU schema/reward/replay dry checks 能证明 Qwen3.5 支持而不是“CLI 能启动”？

当日落地：固定 tag/SHA/container，完成 Qwen3.5 support matrix、Sample/DataSource/rollout/train/weight-version 图、reward tests 与历史 live-run contract。若完整兼容未证实，标记 slime runtime blocked，禁止换模型；这些 static/runtime 边界继续作为 Day 27–30 architecture study 的案例输入。

<a id="day-27"></a>

### Day 27 — 周末：Training System 总图与框架分层

回看 [Day 05 framework map](day-05-training-lifecycle-framework-map/README.md)、[slime Architecture](https://thudm.github.io/slime/blogs/introducing_slime.html) 与 [Megatron Core quickstart](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)。阅读目标不是记 API，而是区分 recipe、orchestrator、rollout engine、learner、distributed runtime、device compute 和 persistence。

- **对象/数据题**：offline training 与 online RL 分别有哪些对象；新增的 trajectory、reward、policy-version 和 weight-sync 边在哪里？
- **状态/训练题**：ms-swift、slime、Ray、SGLang、Megatron、PyTorch/NCCL/CUDA 各拥有或委托什么状态？
- **诊断/判断题**：一个 import/placement failure、一个 loss failure 和一个 weight-version failure分别应先归到哪层？

当日落地：两张不以框架名起笔的 lifecycle 图，以及含 `producer / object / transport / consumer / owner / evidence` 的 node ledger。

<a id="day-28"></a>

### Day 28 — 周末：Megatron × slime 对象、状态与接口

只读两份已有 evidence：[Day 18 Megatron runtime report](artifacts/reports/day18-qwen35-megatron-compatibility.md) 与 [Day 26 slime runtime config](artifacts/configs/day26-slime-qwen35-runtime.json)。把 Sample/trajectory、learner batch、training state、serving weights、policy version、checkpoint/export 分开定义。

- **对象/数据题**：slime `Sample` 怎样才能成为 Megatron 可消费的 tensor batch，中间必须补齐哪些 mask/log-prob/version 字段？
- **状态/训练题**：Ray actor、GPU process、distributed rank、rollout model 与 learner shard 为什么不是同一个对象？
- **诊断/判断题**：static source evidence 能回答哪些架构问题，哪些结论必须保留为 `RUNTIME UNKNOWN`？

当日落地：共同术语表与 Day 29/30 source-reading question list；周末不启动 GPU。

<a id="day-29"></a>

### Day 29 — Megatron Architecture：进程组、状态所有权与 Training Step

阅读 [Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html) 与 Day 18 的 8-node runtime codepath。重点是从 config/source 推导 state 和 communication，不再开一个训练 run。

- **对象/数据题**：DP/TP/PP/CP/EP 分别切 batch、parameter、activation、sequence 或 expert 的哪一维；global batch 为什么只随数据副本和 accumulation 改变？
- **状态/训练题**：parameter、gradient、optimizer、activation、RNG、data cursor 和 checkpoint metadata 在哪些 rank 上 replicated/sharded/transient/persisted？
- **诊断/判断题**：TP logits 分叉、collective hang、OOM、optimizer 不更新、checkpoint 只能推理不能 resume 时，第一检查节点是什么？

当日落地：process-group 图、train-step `file:function` 链、state-ownership ledger 与 failure tree。Day 26 的旧 run no-go 是历史记录，不阻塞 CPU 学习。

<a id="day-30"></a>

### Day 30 — slime Architecture 与 Training System 集成

阅读 pinned slime source anchors、[Architecture](https://thudm.github.io/slime/blogs/introducing_slime.html)、Debug/Trace/Reproducibility/Fault-tolerance 文档，并回看 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 中 old/current/reference policy 的定义。算法只服务于解释系统中的 policy-version 和 staleness。

- **对象/数据题**：`DataSource -> SGLang rollout -> environment/reward -> Sample/buffer -> Megatron learner batch` 每条边的 schema、grain、transport 和 failure semantics 是什么？
- **状态/训练题**：Ray control plane、SGLang serving state、Megatron training state、weight sync 和 resumable checkpoint 怎样连接又怎样彼此独立？
- **诊断/判断题**：Day 18 learner pass、Day 25 ms-swift GRPO pass 和 Day 26 slime static pass/runtime S0 fail各覆盖 integrated graph 的哪些节点，为什么不能拼成不存在的 E2E success？

当日落地：control/data/weight/evidence 四张 flow、ms-swift↔slime 职责 crosswalk、20 分钟架构口述与 `KNOWN / INFERRED / RUNTIME UNKNOWN` 清单；不启动 GPU。

## Deferred Archive：Qwen3.5-4B Policy Capstone + Teacher 扩展

Day 31–42 的执行型计划已暂停，不是 Day 30 后的活动学习主线。下列内容仅保留为未来可能重启时的实验设计参考；任何条目都需要用户明确重启和新 charter。

<a id="day-31"></a>

### Day 31 — 冻结 Qwen3.5 S0、Domain Eval 与 Teacher-null Charter

状态：`deferred_after_architecture_study`。当前不执行；以下仅为历史设计。

阅读：[`QWEN35-4B-MIGRATION-PLAN.md`](QWEN35-4B-MIGRATION-PLAN.md)、Day 10/21 held-out hygiene 和 pinned Qwen3.5 runtime 文档。Teacher/OPD 只登记为 deferred，不选择候选。

- **对象/数据题**：`sft_train/policy_train/dev/confirmation` 为什么必须分开；processor/template/render/execution keys 如何冻结？
- **状态/训练题**：活动 DAG `S0 Base -> S1 selected coding SFT -> S2 direct coding RL` 的 parent/objective/immutable identity 如何组成？
- **诊断/判断题**：revision、processor、freeze policy、runtime 或预算缺一项时，为什么不能打开 capstone？

当日落地：S0 exact revision、charter、compatibility manifest、新 eval suite、预算/selection policy，以及 `teacher_model_id=null` 的显式状态。

<a id="day-32"></a>

### Day 32 — Qwen3.5 Single/TP2 Parity 与 Full/LoRA/QLoRA Capacity

状态：`deferred_after_architecture_study`。当前不执行 GPU parity/capacity run。

阅读：[Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、distributed optimizer 与 Day 18 runtime evidence。

- **对象/数据题**：single 与 TP2 中 parameter/gradient/optimizer/activation shards、processor output、sample IDs 和 label tokens 如何对应？
- **状态/训练题**：TP degree、sequence parallel、RNG、loss reduction 与 checkpoint metadata 如何改变可复现状态？
- **诊断/判断题**：TP2 loss 分叉或更慢时，怎样区分 Qwen3.5 mapping/backend、batch 口径、collective 和 4B 通信开销；full/LoRA/QLoRA 如何用实测而不是名义参数判断？

当日落地：Qwen3.5-4B parity report、checkpoint conversion/export evidence、每卡 peak 与 full/LoRA/QLoRA topology decision。

<a id="day-33"></a>

### Day 33 — Deferred Template：Teacher TP SFT Gate

状态：`deferred`。只有用户另行选择 teacher、冻结 exact revision 并批准 Teacher/OPD charter v2 后才阅读或执行。

- **对象/数据题**：未来 teacher batch 如何从 raw sample 追到各 TP rank 的 processor output、label 和 loss？
- **状态/训练题**：full/PEFT update、distributed optimizer、RNG/data cursor 和 export 分别保存在哪里？
- **诊断/判断题**：one-step 能跑但 resume/export 失败时，为什么不能启动 teacher SFT？

当日落地：当前只保留模板；不得创建 T0 或租卡。

<a id="day-34"></a>

### Day 34 — Deferred Template：Teacher SFT 与 T1 Selection

状态：`deferred`；依赖已批准的 charter v2 和 Day 33 gate。

- **对象/数据题**：未来 T0、SFT manifest、candidate 与 dev predictions 如何绑定？
- **状态/训练题**：哪些 student 参数不能迁成 teacher 默认值，哪些必须重新 smoke？
- **诊断/判断题**：final loss 更低但 guardrail 不通过时，如何返回 `inconclusive`？

当日落地：当前只保留 promotion 模板；`T1` 不存在。

<a id="day-35"></a>

### Day 35 — Deferred Template：Teacher Domain-RL Readiness

状态：`deferred`；依赖存在的 T1 和单独预算。

- **对象/数据题**：未来 teacher trajectory 中 assistant/tool/environment tokens、reward components 与 train mask 如何关联？
- **状态/训练题**：teacher learner/rollout/reference/reward workers 与 policy version 如何流动？
- **诊断/判断题**：reward 提升时怎样排除格式、长度、伪造 observation 与 stale weight hacking？

当日落地：当前只保留 runbook/placement/hacking-test 模板。

<a id="day-36"></a>

### Day 36 — Deferred Template：Teacher RL 与 T2 Freeze

状态：`deferred`；依赖 Day 35 readiness。

- **对象/数据题**：未来 candidate 的 prompts、rollouts、rewards、logprobs、checkpoint 与 dev evidence 如何追踪？
- **状态/训练题**：T1→T2 哪些 policy/optimizer/RNG/buffer 状态改变，serving 状态如何冻结？
- **诊断/判断题**：T2 提升为什么仍不自动证明对 S1 有可蒸馏 advantage？

当日落地：当前只保留 T2 manifest 模板；`T2` 不存在。

## Deferred Archive Week 6：Direct Coding RL、OPD 与 Final Comparison

<a id="day-37"></a>

### Day 37 — S1→Direct Coding RL→S2

状态：`deferred_after_architecture_study`。当前不启动 direct RL。

阅读：Day 16 SFT promotion、Day 25 coding GRPO 和 Day 31 charter。活动任务与 teacher 无依赖。

- **对象/数据题**：S1、policy prompt universe、sandbox tests、rollouts、reward 与 S2 candidates 如何绑定？
- **状态/训练题**：从同一 S1 分叉时哪些 checkpoint/processor states 必须相同，S2 direct RL 更新哪些状态？
- **诊断/判断题**：reward 上升而 dev/guardrail/termination 下降时，如何拒绝 S2 并保留 S1？

当日落地：运行受控 direct coding RL，保存 S1/S2 manifests、逐样本 rollout 与 frozen budget；选择 S2 或返回 `no-promotion`。

<a id="day-38"></a>

### Day 38 — Deferred Template：Teacher-trace Cold-start

状态：`deferred`；只有 charter v2、T2 与兼容性 gate 同时存在才启用。当前 `S1d` 不存在，活动 S1 不被覆盖。

当日落地：只保留 trace manifest/audit 模板，不生成 teacher trace。

<a id="day-39"></a>

### Day 39 — Deferred Template：OPD One-update 与 Replay

状态：`deferred`；无 teacher 时不得构造伪 payload 或用另一个模型顶替。

当日落地：只保留 processor/token alignment、student rollout、teacher scoring、distill mask、replay 和 version timeline 的 schema。

<a id="day-40"></a>

### Day 40 — Deferred Template：Controlled OPD 与 S3 Selection

状态：`deferred`；依赖 Day 39 one-update gate。当前 `S3` 不存在。

当日落地：只保留 S3 selection、token-divergence slices 与完整成本账本模板。

<a id="day-41"></a>

### Day 41 — S0/S1/S2 Matched Eval 与 Cost Accounting

状态：`deferred_after_architecture_study`。依赖已暂停的 Capstone lineage。

阅读：Day 10/21 held-out 与 paired comparison；不再阅读训练 recipe。

- **对象/数据题**：`eval_suite_hash` 与 model-specific render/execution keys 为什么必须分开？
- **状态/训练题**：所有 candidates 锁定、frozen 首次揭盲和 consumption record 的顺序是什么？
- **诊断/判断题**：S2 能力更强但 rollout/sandbox compute 更高时，怎样分开能力、效率和摊销结论；无显著差异时如何写 `inconclusive`？

当日落地：一次性揭盲 `S0/S1/S2` paired confirmation，产出 fully-loaded/amortized cost report。只有 charter v2 已完成时才附加 teacher/OPD candidates。

<a id="day-42"></a>

### Day 42 — S1/S2 Clean Reproduction 与 Capstone Report

状态：`deferred_after_architecture_study`。不属于当前毕业 gate。

阅读：只使用 pinned manifests、runbooks、framework docs 和已有 evidence；不新增方法。

- **对象/数据题**：另一环境重建 S0/S1/S2 DAG、coding rollout/sandbox batch 和 E2E trajectory 需要哪些不可变对象？
- **状态/训练题**：distributed restore、processor/policy versions、replay 与 scorer state 如何证明连续？
- **诊断/判断题**：哪些 failure 是 Qwen3.5/TP/RL 才暴露，哪些从 0.6B v1 历史已能提前阻止？

当日落地：clean restore/replay/score S1/S2、final report 与 evidence index；deferred teacher 文件不计失败或完成。

## 30-Day Core 明确不做

- 不构建 auto-train scheduler、自动搜索器或 auto-harness 产品。
- 不把 30B+ full training、8×H100 slime、Policy Capstone 或未选择的 teacher 当 30-Day Core 验收；Day 27–30 只做架构学习和已有 evidence 复盘。
- 不通读 TPU/JAX API，不实现 NCCL collective 或手写 TP kernel。
- 不以“章节读完”“loss 下降”或 aggregate score 单独作为完成标准。
- 不让 Scaling Book 的纯推导挤占数据审计、训练恢复和 RL 数据流验证。
