# Resource Index

资料按训练问题组织，优先列项目官方文档、官方仓库和论文原文。每天只读当天 README/reading guide 指定的小节，不要求通读所有链接。

## 课程主线与边界

- [每日精读与三题路线](SCALING-BOOK-READING-GUIDE.md)
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)：系统直觉与容量/吞吐判断；不是本月唯一主线
- [Tülu 3 paper](https://arxiv.org/abs/2411.15124)：公开的 SFT → DPO → RLVR pipeline 参照
- [Open Instruct 官方文档](https://allenai.github.io/open-instruct/) / [官方仓库](https://github.com/allenai/open-instruct)：数据和训练 recipe 参照

本月 Core 是数据、SFT、恢复/诊断、DPO、在线 RL，以及收官阶段的 training-system / Megatron / slime 架构学习。auto-train/auto-harness 构建、30B+ full training、8×H100 slime 和原 Day 31–42 Policy Capstone 都不属于当前 Core；后者已整体 deferred。

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

### Day 1–12 历史实操：ms-swift/Qwen3

- [modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- [Command-line Parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [Frequently Asked Questions](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md)
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)
- [Qwen3-0.6B-Base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

这些链接和 Transformers 4.57.3 环境继续服务于 Day 1–12 checkpoint 的历史复现，不因 v2 迁移而改写。

### Qwen3.5-4B v2 迁移

- [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base)：模型 identity、原生 262,144 context、统一 vision-language 架构和 Base 使用边界。
- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)：完整 checkpoint 的 processor/conditional-generation 接口与纯文本 backbone 类的区别。
- [Transformers Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto)：`AutoProcessor`、`AutoModelForMultimodalLM` 与 exact runtime class 的入口。
- [ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)：当前安装下限建议、Dense training、LoRA、RL 与 backend 示例。
- [ms-swift Supported Models](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md)：Qwen3.5 model type、template、multimodal dependencies 与支持矩阵。
- [ms-swift FAQ](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md)：VLM freeze、显存、packing 与运行问题边界。
- [Qwen3.5 官方仓库](https://github.com/QwenLM/Qwen3.5)：模型家族与官方变更入口。

v2 的第一阶段只接受 text-only coding 数据，但仍把下载的 full model 视为 VLM：标准 loader 是 `AutoProcessor` + `AutoModelForMultimodalLM`/`Qwen3_5ForConditionalGeneration`，vision tower 与 aligner 必须以 trainable parameters、adapter targets、gradient 和 optimizer state 四类断言证明冻结。`Qwen3_5ForCausalLM` 只适用于明确的 text-backbone 合同；从 full checkpoint 提取 backbone 属于新的转换产物，必须另做 hash、lineage 与 parity，不能作为旧 runner 的无差别替换。

Transformers 5.2.0+ 是已确认包含 Qwen3.5 支持的最低候选线；ms-swift 当前 Best Practice 建议安装 `transformers>=5.9`。课程不保留浮动依赖：Day 15 在目标 GPU 上从该建议线选择一个 exact Transformers 版本，并同时冻结 Python 3.12、ms-swift commit/version、`qwen_vl_utils`、`decord`、FlashAttention/GDN 依赖、PyTorch/CUDA/NCCL 与 rollout backend。`main/latest` 只用于发现，不能写入正式 run identity。

资源文档中的 24/48/80GB 与多卡层级只是 planning envelope。官方示例、Ascend/NPU PR 或单步验证不等于本项目在 NVIDIA 上的 peak memory；每种 dtype、sequence、learner/rollout placement 都必须跑完整 optimizer-step peak smoke，保留 10%–15% headroom。虽然模型原生支持 262K context，首轮 RL 的总长度 cap 只允许 8K–12K；reward/verifier 优先放在 CPU sandbox。

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

目标是能从 config/source 推导 process groups、state ownership、global batch、训练 step、checkpoint、OOM 和 throughput；复用 Day 18 已有 runtime evidence 深挖 codepath，不以再跑一个 smoke 代替理解，也不要求手写 collective/kernel 或通读 Megatron 全仓。

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

### 已有实操与架构主读

- [ms-swift GRPO](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)：Qwen3.5 coding GRPO 主线；从 promoted S1 开始，并冻结 rollout/model-length/topology
- [THUDM/slime 官方文档](https://thudm.github.io/slime/)
- [slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)
- [slime Architecture](https://thudm.github.io/slime/blogs/introducing_slime.html)
- [slime Customization](https://thudm.github.io/slime/get_started/customization.html)
- [slime 官方仓库](https://github.com/THUDM/slime)
- [slime Releases](https://github.com/THUDM/slime/releases)：通用文档以 `v0.3.0` 为阅读基线；Qwen3.5 源码追踪以 Day 26 的 `v0.3.1@a6272da...` 为锚
- [slime Debug](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/developer_guide/debug.md)
- [slime Trace](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/developer_guide/trace.md)
- [slime Reproducibility](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/advanced/reproducibility.md)
- [slime Fault Tolerance](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/advanced/fault-tolerance.md)

### 算法与架构对照

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [verl 官方文档](https://verl.readthedocs.io/)
- [verl PPO architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)

ms-swift 已完成 SFT/DPO/GRPO 实跑，后续不再为积累框架经验重复训练。Day 27–30 用 slime、Megatron 和已有 ms-swift evidence 做职责 crosswalk：追踪 control/data/weight/evidence flow 与 node ownership。Day 26 未通过的 live 边保持 `RUNTIME UNKNOWN`；不为跑通 slime 而换模型、换 runtime 或追加 GPU。Tülu/Open-Instruct、TRL、verl 只用于检验这套节点心智模型能否迁移。

## On-Policy Distillation（Deferred Teacher Extension）

以下资料只服务未来单独批准的 charter v2。当前 `teacher_model_id/revision=null`，不创建 T0/T1/T2/S3，不分配 teacher GPU；资料存在不等于模型已选或 OPD 已获授权。

- [GKD / On-Policy Distillation of Language Models（ICLR 2024）](https://arxiv.org/abs/2306.13649)：student-generated states、teacher feedback 与 divergence 选择的基础。
- [Rethinking On-Policy Distillation（2026）](https://arxiv.org/abs/2604.13016)：teacher/student compatibility、teacher novelty、cold-start 与 prompt selection 风险。
- [ms-swift stable distillation docs](https://swift.readthedocs.io/en/v4.4/Instruction/Distillation.html)：GKD/OPD-RL 入口；运行时必须 pin release、SHA 和 resolved config。
- [ms-swift v4.4.2 release](https://github.com/modelscope/ms-swift/releases/tag/v4.4.2)：当前审计过的起始版本候选，不代表 Day 31 必须使用旧版本。
- [ms-swift v4.4.2 Megatron OPD-RL example](https://github.com/modelscope/ms-swift/blob/v4.4.2/examples/megatron/grpo/opd_rl.sh)：只作 pinned baseline；示例中的 checkpoint/RNG 保存选项不得直接复制到正式可恢复 run。
- [verl OPD docs](https://verl.readthedocs.io/en/latest/algo/opd.html)：teacher resource pool、same-tokenizer约束、loss modes 与 systems migration 参照。
- [verl v0.8.0 release](https://github.com/verl-project/verl/releases/tag/v0.8.0)：当前审计过的 OPD systems-stretch 起点；完整训练不与 ms-swift Core 重复。
- [TRL DistillationTrainer](https://huggingface.co/docs/trl/main/en/distillation_trainer)：小型 API/correctness 对照；experimental API 不作为多卡 Core 的稳定承诺。
- [NeMo-RL On-policy Distillation](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html)：独立实现与当前 backend 边界参照。

只有 Teacher/OPD charter v2 激活时才冻结对应 compatibility manifest：framework tag/SHA、container digest、Torch/CUDA/NCCL、Megatron、vLLM/SGLang、teacher/student model/processor/tokenizer revisions、template、verifier 和 dataset hashes。文档的 `latest/main` 只用于发现入口。

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
