# Resource Index

资料按训练问题组织，优先列项目官方文档、官方仓库和论文原文。每天只读当天 README/reading guide 指定的小节，不要求通读所有链接。

## 课程主线与边界

- [每日精读与三题路线](SCALING-BOOK-READING-GUIDE.md)
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)：系统直觉与容量/吞吐判断；不是本月唯一主线
- [Tülu 3 paper](https://arxiv.org/abs/2411.15124)：公开的 SFT → DPO → RLVR pipeline 参照
- [Open Instruct 官方文档](https://allenai.github.io/open-instruct/) / [官方仓库](https://github.com/allenai/open-instruct)：数据和训练 recipe 参照

本月 Core 是数据、SFT、恢复/诊断、DPO、在线 RL。auto-train/auto-harness 构建、30B+ full training 和 8×H100 slime 不属于 30-Day Core。Day 31–42 另有可选的 8B teacher→<=4B student OPD capstone。

## 数据、模板与 Lineage

- [ms-swift Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)
- [Open Instruct Dataset Transformations](https://allenai.github.io/open-instruct/algorithms/dataset_transformation/)
- [Open Instruct Synthetic Preference Dataset](https://allenai.github.io/open-instruct/algorithms/synthetic_preference_dataset/)
- [Transformers Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Datasets main classes/version fingerprints](https://huggingface.co/docs/datasets/package_reference/main_classes)
- [Datasets data loading](https://huggingface.co/docs/datasets/loading)
- [TRL Dataset Formats](https://huggingface.co/docs/trl/dataset_formats)

阅读数据文档时固定追踪：

`raw source/revision -> filter/dedup/mix -> messages -> rendered text -> tokens -> labels/mask -> batch -> checkpoint/eval lineage`

## SFT、优化与恢复

### 主实操：ms-swift

- [modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- [Command-line Parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [Frequently Asked Questions](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md)
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)
- [Qwen3-0.6B-Base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

### 训练语义与诊断

- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch gradient clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)：紧凑 API 对照，不替换 ms-swift 主实操

## Scaling、并行与 Megatron

- [Roofline](https://jax-ml.github.io/scaling-book/roofline/)
- [Transformer Math](https://jax-ml.github.io/scaling-book/transformers/)
- [Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/)
- [Parallelize a Transformer for Training](https://jax-ml.github.io/scaling-book/training/)
- [Applied Training](https://jax-ml.github.io/scaling-book/applied-training/)
- [Inference](https://jax-ml.github.io/scaling-book/inference/)
- [Profiling](https://jax-ml.github.io/scaling-book/profiling/)
- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core first training run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)
- [Megatron Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)

目标是会解释 state ownership、global batch、OOM 和 throughput，能跑/读最小 codepath；不要求手写 collective、kernel 或通读 Megatron 全仓。

## Eval

- [LM Evaluation Harness 文档](https://lm-evaluation-harness.readthedocs.io/)
- [LM Evaluation Harness 仓库](https://github.com/EleutherAI/lm-evaluation-harness)
- [LightEval 官方文档](https://huggingface.co/docs/lighteval/)

Eval 的最低证据是 frozen config 加逐样本 prediction。aggregate score、训练 loss 或 judge 单次输出都不能单独驱动下一轮训练。

## Preference Optimization

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Open Instruct synthetic preference data](https://allenai.github.io/open-instruct/algorithms/synthetic_preference_dataset/)
- [Tülu 3 paper](https://arxiv.org/abs/2411.15124)

主实验从已经验证的 SFT checkpoint 出发；先审计 prompt/chosen/rejected/template/mask，再解释 DPO 指标。

## Online RL 与 GRPO

### 主实操

- [ms-swift GRPO](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)：先完成单机小模型数据/reward/训练闭环
- [THUDM/slime 官方文档](https://thudm.github.io/slime/)
- [slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)
- [slime Architecture](https://thudm.github.io/slime/blogs/introducing_slime.html)
- [slime Customization](https://thudm.github.io/slime/get_started/customization.html)
- [slime 官方仓库](https://github.com/THUDM/slime)
- [slime Releases](https://github.com/THUDM/slime/releases)：本计划以 `v0.3.0` 为学习基线，运行前记录 tag SHA
- [slime Debug](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/developer_guide/debug.md)
- [slime Trace](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/developer_guide/trace.md)
- [slime Reproducibility](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/advanced/reproducibility.md)
- [slime Fault Tolerance](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/advanced/fault-tolerance.md)

### 算法与架构对照

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [verl 官方文档](https://verl.readthedocs.io/)
- [verl PPO architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)

ms-swift 和 slime 要实际运行；Tülu/Open-Instruct、TRL、verl 用于比较 stage、schema、角色和设计选择，不要求本月把四套框架都跑一遍。

## On-Policy Distillation（Optional Capstone）

- [GKD / On-Policy Distillation of Language Models（ICLR 2024）](https://arxiv.org/abs/2306.13649)：student-generated states、teacher feedback 与 divergence 选择的基础。
- [Rethinking On-Policy Distillation（2026）](https://arxiv.org/abs/2604.13016)：teacher/student compatibility、teacher novelty、cold-start 与 prompt selection 风险。
- [ms-swift stable distillation docs](https://swift.readthedocs.io/en/v4.4/Instruction/Distillation.html)：GKD/OPD-RL 入口；运行时必须 pin release、SHA 和 resolved config。
- [ms-swift v4.4.2 release](https://github.com/modelscope/ms-swift/releases/tag/v4.4.2)：当前审计过的起始版本候选，不代表 Day 31 必须使用旧版本。
- [ms-swift v4.4.2 Megatron OPD-RL example](https://github.com/modelscope/ms-swift/blob/v4.4.2/examples/megatron/grpo/opd_rl.sh)：只作 pinned baseline；示例中的 checkpoint/RNG 保存选项不得直接复制到正式可恢复 run。
- [verl OPD docs](https://verl.readthedocs.io/en/latest/algo/opd.html)：teacher resource pool、same-tokenizer约束、loss modes 与 systems migration 参照。
- [verl v0.8.0 release](https://github.com/verl-project/verl/releases/tag/v0.8.0)：当前审计过的 OPD systems-stretch 起点；完整训练不与 ms-swift Core 重复。
- [TRL DistillationTrainer](https://huggingface.co/docs/trl/main/en/distillation_trainer)：小型 API/correctness 对照；experimental API 不作为多卡 Core 的稳定承诺。
- [NeMo-RL On-policy Distillation](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html)：独立实现与当前 backend 边界参照。

Capstone 第一次运行前冻结 compatibility manifest：framework tag/SHA、container digest、Torch/CUDA/NCCL、Megatron、vLLM/SGLang、model/tokenizer revisions、template、verifier 和 dataset hashes。文档的 `latest/main` 只用于发现入口。

## AutoDL

- [计费规则](https://www.autodl.com/docs/price/)
- [实例数据保留](https://www.autodl.com/docs/instance_data/)
- [保存与加载镜像](https://www.autodl.com/docs/image/)
- [文件存储](https://www.autodl.com/docs/fs/)
- [省钱与无卡模式](https://www.autodl.com/docs/save_money/)

## 使用规则

1. 论文、文档和 repo 都记录访问日期；repo 第一次使用后固定 commit，模型/tokenizer/dataset 固定 revision。
2. 链接到 `main/latest` 只用于发现入口，实验记录必须写实际 commit/version。
3. 资料与运行行为冲突时，以 pinned codepath、完整 config 和 runtime evidence 为准，并记录差异。
4. 每篇材料只需产出与当天对象、状态、诊断三题有关的结论；不抄长摘要。
