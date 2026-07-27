# Scaling Book 与 Training 精读路线

主教材：[How To Scale Your Model](https://jax-ml.github.io/scaling-book/)  
适用范围：Day 01–30

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

### Day 13 — 周末：Tülu 3 与 Qwen3 Post-training 对读

对读：[Tülu 3 paper](https://arxiv.org/abs/2411.15124) 的 pipeline、SFT、preference 与 RLVR，以及 [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 的 post-training pipeline；用 [Open Instruct](https://allenai.github.io/open-instruct/) 对应 recipe 入口核对可复现细节，不抄 benchmark 大表。

- **对象/数据题**：Tülu 3 与 Qwen3 的各阶段数据对象、来源和过滤有什么共同点与差异？
- **状态/训练题**：两条 pipeline 每阶段从什么 checkpoint/模型角色开始，哪些状态跨阶段继承，哪些重新初始化？
- **诊断/判断题**：哪些公开 recipe 结论可迁移到当前 Qwen 小模型，哪些因模型、数据或 reward 不同必须重新做 pilot？

当日落地：把公开 pipeline 映射到本计划 Day 08–29，标注只参考不实跑的部分。

<a id="day-14"></a>

### Day 14 — 周末：Week 2 复盘

不读新材料；回看 Day 08–13 的样本审计、Base predictions、tiny overfit 和 SFT checkpoints。

- **对象/数据题**：随机抽一条训练样本，能否从 manifest 一直追到 loss token 和 source？
- **状态/训练题**：从 Base 到选中 SFT checkpoint，哪些状态和配置构成完整 lineage？
- **诊断/判断题**：当前 SFT 提升最可能来自训练、数据选择还是评测波动？缺哪项证据？

当日落地：Week 2 gate；从候选问题中只选 **一个** 最高优先级单变量实验进入 Day 15–16，其余放入 backlog。

## Week 3：稳定训练、恢复与诊断

<a id="day-15"></a>

### Day 15 — Packing、sequence length 与有效 label token

阅读：[Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) 的 attention cost、context length 与 gradient checkpointing；ms-swift 参数文档中的 packing/max length。

- **对象/数据题**：原始长度、截断长度、packed sequence、padding tokens 与 label tokens 如何统计？
- **状态/训练题**：packing 改变 attention boundary、position/segment metadata 和 batch composition 中的哪些项？
- **诊断/判断题**：tokens/s 上涨但效果下降时，如何检查 cross-sample attention、EOS、mask 和有效 label-token ratio？

当日落地：保持有效训练 token budget 可比的 packed/unpacked 对照。

<a id="day-16"></a>

### Day 16 — Optimizer、LR、warmup、batch 与梯度稳定性

阅读：[PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)、[gradient clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) 与 [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)；Scaling Book 只回查 activation/FLOPs 对预算的影响。

- **对象/数据题**：对照实验必须固定哪些 dataset order、tokens、batch、checkpoint 与 eval inputs？
- **状态/训练题**：LR、warmup、weight decay、clip、precision 分别在哪个时刻影响 gradient 或 parameter update？
- **诊断/判断题**：loss spike/NaN 出现时，怎样用 grad norm、scale、LR、具体 batch 和参数统计区分数据异常与数值不稳定？

当日落地：使用同一数据顺序和 label-token budget，对 AdamW baseline、2×LR、无 warmup 与 2×effective batch 做一次一变量对照，保留 grad/update/clip 证据。

<a id="day-17"></a>

### Day 17 — Exact checkpoint resume 与复现

阅读：[PyTorch Saving and Loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) 与 [ms-swift FAQ](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md) 的 resume。

- **对象/数据题**：checkpoint manifest 如何关联 shards、config、tokenizer、dataset/data position 与代码版本？
- **状态/训练题**：model、optimizer、scheduler、scaler、RNG、global step、sampler/dataloader state 中漏哪项会怎样？
- **诊断/判断题**：resume 后 loss/LR/样本顺序跳变时，如何用“连续 run vs 中断恢复 run”最小对照定位？

当日落地：比较连续 40-step 与 20-step 中断后新进程恢复到 40-step，逐步核对 sample ID、LR、loss、model/optimizer/scheduler/RNG/dataloader state。

<a id="day-18"></a>

### Day 18 — Megatron minimum codepath、双卡 TP/DP 与 distributed checkpoint

阅读：[Megatron Core first training run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)、[parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) 与 [distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)；回读 Scaling Book Training 对应概念。

- **对象/数据题**：最小 run 中 dataset iterator、microbatch、model shard、gradient buffer 和 checkpoint shard 由哪些 rank 持有？
- **状态/训练题**：TP/DP process groups、forward/backward schedule、reduce-scatter/all-gather 与 optimizer step 怎样连接？
- **诊断/判断题**：2 卡 smoke OOM、hang 或不提速时，如何先判断 batch/状态复制、collective/group 配置还是环境问题？

当日落地：一条能在日志中验证的最小 codepath，双卡 `TP=1/DP=2`、`TP=2/DP=1` 两个 smoke，以及 distributed checkpoint 新进程 reload；不做全仓逐文件通读。

<a id="day-19"></a>

### Day 19 — 训练诊断与 failure injection

阅读：[Scaling Book Profiling](https://jax-ml.github.io/scaling-book/profiling/) 的 trace/memory profile；[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) 与 [autograd anomaly detection](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)。

- **对象/数据题**：failure report 必须绑定哪个 batch/sample、run/config、rank、step、checkpoint 与 trace window？
- **状态/训练题**：故意注入坏 mask、过高 LR、错误 resume 或慢 data source 时，预期哪些状态和指标首先变化？
- **诊断/判断题**：怎样用最小复现把 loss 问题分到 data/objective/optimization，把慢分到 input/compute/communication/checkpoint I/O？

当日落地：高 LR、错误 mask、数据分布偏移、吞吐退化四类单一 failure injection 及对应恢复 runbook。

<a id="day-20"></a>

### Day 20 — 周末：训练失败案例复盘

阅读：[PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)、[NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) 与 ms-swift FAQ 中与 OOM/resume/packing 直接相关的条目。

- **对象/数据题**：每个 failure case 缺失或损坏的对象是什么，如何在 run 前验证？
- **状态/训练题**：OOM、NaN、hang、resume drift 分别涉及哪些状态边界？
- **诊断/判断题**：为四类失败各写“症状 -> 证据 -> 最小验证 -> 停止条件”，避免只列可能原因。

当日落地：一页 failure signature 表。

<a id="day-21"></a>

### Day 21 — 周末：Eval 与 checkpoint selection 可靠性

阅读：[LM Evaluation Harness docs](https://lm-evaluation-harness.readthedocs.io/) 的 reproducibility、sample logging 与 task guide；回读自己的 Base/SFT predictions。

- **对象/数据题**：capability、style、safety、regression 各需要什么样本和 metric，哪些不能混成一个总分？
- **状态/训练题**：checkpoint、template、decoder、judge/scorer version 哪些必须冻结才能比较训练阶段？
- **诊断/判断题**：如何在看结果前冻结 candidate set、primary metric、guardrails、最小差异、CI、tie-breaker 与 `inconclusive` 条件？

当日落地：一版 checkpoint-selection policy 和一次盲化 rehearsal；独立 held-out confirmation 不参与反复调参。

## Week 4：Preference、DPO 与 Online RL

<a id="day-22"></a>

### Day 22 — Preference provenance、length bias 与 held-out

阅读：[Open Instruct synthetic preference dataset](https://allenai.github.io/open-instruct/algorithms/synthetic_preference_dataset/)、[TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats) 与 Tülu 3 的 preference data 部分。

- **对象/数据题**：prompt/chosen/rejected、generator、judge、score/margin、source/license/creation method 与 policy version 如何组成可追踪 pair？
- **状态/训练题**：pair 过滤、顺序交换、长度控制和 reference policy 选择如何改变 DPO 训练信号？
- **诊断/判断题**：如何发现 chosen/rejected 反转、模板不一致、近重复、judge length/style bias，以及 prompt/source-family 跨 split 泄漏？

当日落地：preference schema、50-pair 盲审、length/source slices，以及训练前冻结的 group held-out。

<a id="day-23"></a>

### Day 23 — DPO 理论与小模型 smoke

阅读：[DPO paper](https://arxiv.org/abs/2305.18290) 的 objective 与 assumptions；[TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) 或 ms-swift 官方参数作为实现对照。

- **对象/数据题**：chosen/rejected 的 policy/reference token logprob 怎样按同一 prompt、template 和 mask 对齐？
- **状态/训练题**：DPO update 修改什么，reference model 是否更新，beta 如何改变 preference margin？
- **诊断/判断题**：DPO loss/accuracy 变好但生成退化时，如何检查 pair quality、长度偏置、KL 漂移和 frozen eval？

当日落地：同一小型 preference set 的 loss 单元检查与短 smoke。

<a id="day-24"></a>

### Day 24 — Online RL dataflow 与 reward/verifier contract

阅读：[slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) 的 rollout/train batch 关系与 reward 配置；[verl PPO architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html) 作为角色边界对照。

- **对象/数据题**：prompt、grouped responses、tokens、response mask、reward、old/ref/current logprob、advantage 与 policy version 如何关联？
- **状态/训练题**：rollout policy、train policy、reference、reward/verifier、buffer 和 weight sync 在一轮中怎样变化？
- **诊断/判断题**：reward 上升时，哪些独立证据才能排除 length hacking、格式投机、stale rollout 和 mask/logprob 错位？

当日落地：带字段、数量、producer/consumer 的在线 RL 数据流图；实现可重算、可版本化、带 timeout/error semantics 的 verifier contract。

<a id="day-25"></a>

### Day 25 — ms-swift 小模型 GRPO lab

阅读：[ms-swift GRPO](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html) 与 [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) 的数据/reward/config；Scaling Book 只回读 inference 的 KV cache 与 generation throughput。

- **对象/数据题**：每个 prompt 产生多少 completions，reward 如何绑定 response，group 内 advantage 如何生成？
- **状态/训练题**：每轮采样和 update 之间哪些 policy/logprob/optimizer 状态必须一致，哪些可以重算？
- **诊断/判断题**：zero reward variance、OOM、生成过长或 KL 快速增长时，各自第一项配置/数据验证是什么？

当日落地：可验证 reward 的小模型 GRPO run 与逐样本 rollout 表。

<a id="day-26"></a>

### Day 26 — slime v0.3.0 主链路与最小运行准备

阅读：[slime Releases](https://github.com/THUDM/slime/releases)、[Architecture](https://thudm.github.io/slime/blogs/introducing_slime.html)、[Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) 和 [Customization](https://thudm.github.io/slime/get_started/customization.html)。先固定 `v0.3.0` 对应 SHA，再通过 symbol search 定位该 tag 的实际入口；不从滚动 main 硬编码文件路径。

- **对象/数据题**：slime `Sample`/buffer、Megatron batch 与 SGLang request/response 之间如何转换？
- **状态/训练题**：Ray placement、rollout engine、trainer、reward 和 weight sync 各拥有何种 GPU/模型状态？
- **诊断/判断题**：在租多卡前，哪些 CPU/config/schema/reward/debug-replay dry checks 能排除最昂贵的失败？

当日落地：固定 tag/SHA/container，完成 Sample/DataSource/rollout/train/weight-version 图、reward tests、最小支持 recipe 与 Day 29 runbook。

<a id="day-27"></a>

### Day 27 — 周末：GRPO 与 on-policy 边界

阅读：[DeepSeekMath](https://arxiv.org/abs/2402.03300) 的 GRPO method/objective；对照 slime Quick Start 的 batch invariant 和 reward fields。

- **对象/数据题**：同一 prompt 的 group、response、reward、advantage 和 token mask 的 grain 分别是什么？
- **状态/训练题**：old/reference/current policy 在 objective 中承担什么角色，哪些量来自 rollout 时刻？
- **诊断/判断题**：generation、buffer、training 和 weight sync 的哪些延迟会让 rollout stale；importance correction 的有效边界是什么？

当日落地：用自己的 Day 25 样本标出 rollout/old/current/reference policy，并画同步与 stale 两条 timeline。

<a id="day-28"></a>

### Day 28 — 周末：slime debug、replay、repro 与 observability

阅读 pinned slime repo 的 Debug、Trace/Profiling、Reproducibility 与 Fault-tolerance 文档；所有参数和路径以 Day 26 checkout 解析结果为准。

- **对象/数据题**：一轮中每条边传什么对象、多少条、由哪个 `file:function` 生产和消费？
- **状态/训练题**：哪些组件持久、哪些每 rollout 重建、何时 policy version 发生变化？
- **诊断/判断题**：rollout-only、reward replay、train-only replay、weight sync 与 next-version rollout 各需要什么证据，失败时在哪一层停止？

当日落地：Day 29 五级 gate 表；周末不启动 GPU。

<a id="day-29"></a>

### Day 29 — slime 最小闭环、reward 修改与 train-only replay

开机前重读 slime Quick Start 的 batch invariant、reward customization 与 checkpoint/debug 部分；需要 profiling 时只回查 [Inference](https://jax-ml.github.io/scaling-book/inference/) 的 generation bottleneck。

- **对象/数据题**：每轮 prompt/response/reward/sample 数是否满足 rollout 与 train 消费守恒，坏样本能否反查原 prompt？
- **状态/训练题**：运行日志能否证明 rollout、reward、train、checkpoint 和 weight sync 的实际顺序与 policy version？
- **诊断/判断题**：修改 reward 后指标变化，怎样判断代码确实生效且不是 sample mix、长度或旧权重造成？

当日落地：按 rollout-only → train-only replay → full loop → reward change 顺序跑最小闭环；8×H100 官方规模仅 Stretch。

<a id="day-30"></a>

### Day 30 — Post-training design、clean reproduction 与综合口述

阅读：[Scaling Book Conclusion](https://jax-ml.github.io/scaling-book/conclusion/)；回看 [Tülu 3](https://arxiv.org/abs/2411.15124) 的阶段设计和本月所有 manifest/run records，不引入新框架。

- **对象/数据题**：clean reproduction 所需的 code/model/tokenizer/data/template/config/eval/checkpoint lineage 是否能被另一环境完整解析？
- **状态/训练题**：设计如何定义每阶段初始状态、保存状态、恢复边界和从 SFT 到 DPO/RL 的交接？
- **诊断/判断题**：哪些结论是已实测、哪些是估算、哪些扩到多卡或更大模型前必须重新 benchmark？

当日落地：从干净环境跑一条最小训练链，完成可评审的 post-training design 和 15 分钟综合口述；30B capacity 只允许作为 Stretch appendix。

## 本月明确不做

- 不构建 auto-train scheduler、自动搜索器或 auto-harness 产品。
- 不把 30B+ full training 或 8×H100 slime 当 Core 验收。
- 不通读 TPU/JAX API，不实现 NCCL collective 或手写 TP kernel。
- 不以“章节读完”“loss 下降”或 aggregate score 单独作为完成标准。
- 不让 Scaling Book 的纯推导挤占数据审计、训练恢复和 RL 数据流验证。
