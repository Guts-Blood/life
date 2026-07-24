# Resource Index

资料按问题组织。每天只读对应 README 指定的部分，不要求一次读完。

## Scaling 与硬件

- [本计划的 Scaling Book 逐日精读路线](SCALING-BOOK-READING-GUIDE.md)
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Roofline](https://jax-ml.github.io/scaling-book/roofline/)
- [Transformer Math](https://jax-ml.github.io/scaling-book/transformers/)
- [Parallelize a Transformer for Training](https://jax-ml.github.io/scaling-book/training/)
- [Applied Training: LLaMA 3](https://jax-ml.github.io/scaling-book/applied-training/)
- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Picotron](https://github.com/huggingface/picotron)
- [Nanotron](https://github.com/huggingface/nanotron)

## 分布式训练

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core User Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
- [Megatron Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)
- [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-SWIFT Quick Start](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [DeepSpeed ZeRO](https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/zero.md)
- [Megatron Core Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)

## Qwen 与 SFT

### 版本选择

- 实操主线固定为 Qwen3 0.6B/1.7B/4B：尺寸阶梯完整、示例成熟，适合在一个月内反复验证同一训练链路。
- 30B+ 训练设计使用 [Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)：它是 Base checkpoint，35B total/3B active，适合讨论 SFT/RL 的初始化和模型状态。
- 当前架构对照使用 [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 与 [Qwen3.6 官方仓库](https://github.com/QwenLM/Qwen3.6)。不要直接把 post-trained checkpoint 当作 Base SFT 起点。

- [ms-swift](https://github.com/modelscope/ms-swift)
- [ms-swift Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)
- [ms-swift Command-line Parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [ms-swift Supported Models and Datasets](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md)
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)
- [Qwen3 ms-swift SFT/Megatron Recipe](https://github.com/QwenLM/Qwen3/discussions/1301)
- [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- [Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)
- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

## Eval

- [LM Evaluation Harness](https://lm-evaluation-harness.readthedocs.io/)
- [LM Evaluation Harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)
- [LightEval](https://github.com/huggingface/lighteval)

## Preference Optimization 与 RL

- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [THUDM/slime](https://github.com/THUDM/slime)
- [slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)
- [slime Architecture Blog](https://thudm.github.io/slime/blogs/introducing_slime.html)
- [slime Customization Guide](https://thudm.github.io/slime/get_started/customization.html)
- [ms-swift GRPO](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)
- [veRL PPO Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
- [vLLM Online Serving](https://docs.vllm.ai/en/latest/serving/online_serving/)

## 阅读规则

1. 先回答当天 README 中的问题，再扩展阅读。
2. 每篇材料至少产出一个公式、一个工程映射和一个未解决问题。
3. 资料与实际框架冲突时，记录具体 commit/version；不要用记忆猜测。
4. Repo 的 `main` 会变化。第一次安装后固定 commit，并在每次报告中引用它。

## AutoDL

- [计费规则](https://www.autodl.com/docs/price/)
- [实例数据保留](https://www.autodl.com/docs/instance_data/)
- [保存与加载镜像](https://www.autodl.com/docs/image/)
- [文件存储](https://www.autodl.com/docs/fs/)
- [省钱与无卡模式](https://www.autodl.com/docs/save_money/)
- [容器实例 Pro API](https://www.autodl.com/docs/instance_pro_api/)
