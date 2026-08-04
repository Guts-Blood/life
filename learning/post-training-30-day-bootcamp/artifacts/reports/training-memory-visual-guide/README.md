# Transformer / MoE 训练显存图解

这组图回答两个问题：

1. Dense Transformer 在 full-parameter SFT 时，GPU/CPU 内存里到底要保存什么？
2. Qwen3-30B-A3B 一类 MoE 每个 token 只激活约 3B 参数，为什么训练状态仍然按约 30B total parameters 计算？

## 图

- [`dense-transformer-training-memory.svg`](./dense-transformer-training-memory.svg)：Dense Transformer 数据流、forward/backward 保存项，以及 10B full SFT 显存账本。
- [`qwen3-30b-a3b-moe-training-memory.svg`](./qwen3-30b-a3b-moe-training-memory.svg)：Qwen3-30B-A3B 风格 MoE block、router/expert/AllToAll，以及 total/active parameter 的显存与计算区别。
- [`distributed-training-memory-checkpointing.svg`](./distributed-training-memory-checkpointing.svg)：从全局 weights/gradients/master/Adam 账本到 ZeRO/FSDP 每卡 peak，并解释 activation checkpointing 为什么省显存、disk training checkpoint 为什么不省。

## 首先固定口径

所有 GB 数值均使用十进制：

```text
1 GB = 10^9 bytes
```

图中的 `12 / 16 / 18 bytes per parameter` 是三种常见的 full-parameter Adam 训练账本，不是所有框架都必须采用同一口径：

| 口径 | Low-precision weights | Gradients | Adam m/v | FP32 master weights | 合计 |
|---|---:|---:|---:|---:|---:|
| BF16-native、无独立 master | 2 B/P | 2 B/P | 8 B/P | 0 | 12 B/P |
| BF16 + FP32 master | 2 B/P | 2 B/P | 8 B/P | 4 B/P | 16 B/P |
| Classic mixed precision | FP16 + FP32 = 6 B/P | 4 B/P | 8 B/P | 已含在 weights | 18 B/P |

实际 dtype 由 optimizer、mixed-precision 实现、distributed optimizer 和框架版本决定。做容量规划时必须打印真实参数、gradient 和 optimizer-state dtype，不能只背一个常数。

## 10B Dense：不含 persistent KV cache 的 full SFT

参数量 `P = 10B`：

| 项目 | 12 B/P 口径 | 16 B/P 口径 | 18 B/P 口径 |
|---|---:|---:|---:|
| Parameter-proportional model states | 120 GB | 160 GB | 180 GB |
| Persistent inference KV cache | 0 | 0 | 0 |
| Activations | input-dependent | input-dependent | input-dependent |
| Temporary/workspace/allocator | runtime-dependent | runtime-dependent | runtime-dependent |

结论：

```text
10B full-parameter Adam 的 model states 下界已经约 120–180GB；
activations 和临时峰值还要继续加。
```

因此单张 80GB GPU 不能用朴素方式完成 10B full-parameter SFT。需要 FSDP/ZeRO、distributed optimizer、offload、低精度 optimizer states，或改用 LoRA/其他 PEFT。

### 为什么不给 activation 一个“每参数固定字节数”

Activation memory 不能只从 `P` 推出。它依赖：

```text
local micro-batch tokens
× number of layers
× hidden size
× 每层为 backward 保存的 tensor 数量和 dtype
```

并继续受这些因素影响：

- sequence length、packing 与 padding；
- attention implementation：naive attention 是否 materialize `[B,H,T,T]`，还是使用 FlashAttention；
- activation checkpointing / recomputation；
- MoE token routing、expert capacity 与 dispatch buffers；
- tensor/pipeline/expert/sequence parallel 的 local shape；
- fused kernels、temporary workspace 和 allocator fragmentation。

Activation checkpointing 的本质是少存 forward intermediates，并在 backward 时重新计算，所以它交换的是 memory 与 compute。

这里必须区分两个同名概念：

```text
activation / gradient checkpointing
→ 训练运行时少存 activation，backward 重算局部 forward
→ activation memory 减少，FLOPs 与 step time 增加
→ weights / gradients / optimizer states 不变

training checkpoint saved to disk
→ 保存 model / optimizer / scheduler / RNG / global step
→ 用于容错和 resume
→ 不降低正常训练时的 live GPU memory
```

### Training 中为什么没有 persistent KV cache

标准 teacher-forced SFT 通常关闭 generation 用的 persistent KV cache，因为一次 forward 已经拿到整段训练序列，不需要像逐 token decode 一样跨 decode step 复用历史 K/V。

但是以下 tensor 仍然存在：

- 当前 forward 的 Q/K/V activations；
- attention output；
- backward 所需的 saved tensors；
- naive attention 的 score/probability intermediates，或 FlashAttention 的分块状态。

所以：

```text
no persistent KV cache ≠ no Q/K/V activation memory
```

## Qwen3-30B-A3B：MoE 的 total 与 active

官方 Qwen3-30B-A3B 模型卡与 config 给出的关键规格：

| 字段 | 值 |
|---|---:|
| Total parameters | 30.5B |
| Activated parameters | 3.3B |
| Non-embedding parameters | 29.9B |
| Layers | 48 |
| Hidden size | 2048 |
| Attention | GQA: 32 Q heads / 4 KV heads |
| Experts | 128 |
| Experts selected per token | 8 |
| MoE expert intermediate size | 768 |

每个 token 的 router 从 128 个 experts 中选择 8 个，因此每-token matmul compute 接近 active 3.3B 口径；但是 full-parameter training 必须维护所有 experts 的 weights、optimizer states，以及通常为全参数分配的 gradient/state storage。

### 从 config 独立算出 30.5B total

定义：

```text
D = 2048
Q output = 32 × 128 = 4096
KV output = 4 × 128 = 512
F_expert = 768
E_total = 128
E_active = 8
L = 48
V = 151,936
```

Embedding 和 untied LM head：

```text
V × D = 151,936 × 2,048 = 311,164,928 each
```

Attention projections per layer：

```text
Q = D × 4096
K = D × 512
V = D × 512
O = 4096 × D

Q + K + V + O = 18,874,368
```

One gated expert：

```text
gate + up + down
= 3 × D × F_expert
= 3 × 2,048 × 768
= 4,718,592
```

All experts 和 router per layer：

```text
128 × 4,718,592 = 603,979,776
router = D × 128 = 262,144
```

加入 RMSNorm/QK-Norm 小项后：

```text
parameters per layer = 623,120,640

P_total
= embedding + LM head + 48 × layer + final norm
= 30,532,122,624
≈ 30.532B
```

### 独立算出约 3.3B active

每个 token 只通过 8 个 experts：

```text
active parameters per layer
= attention + router + 8 × one_expert + norms
= 56,889,600

P_active
= embedding + LM head + 48 × active_layer + final norm
= 3,353,032,704
≈ 3.353B
```

这与官方模型卡的 `30.5B total / 3.3B activated` 一致。

以 30.5B total parameters 计算：

| 项目 | 大小 |
|---|---:|
| BF16 weights，2 B/P | 61 GB |
| BF16 gradients，2 B/P | 61 GB |
| FP32 Adam m/v，8 B/P | 244 GB |
| Optional FP32 master，4 B/P | 122 GB |
| 12 B/P model states | 366 GB |
| 16 B/P model states | 488 GB |
| 18 B/P classic mixed precision | 549 GB |

核心结论：

```text
MoE 的 compute 主要看 active parameters；
full-training model-state memory 主要看 total parameters。
```

如果把 16 B/P 的 488 GB 完美均分到 8 张 GPU，理想平均值已经约 `61 GB/GPU`，还没有加入 activations、temporary buffers、通信 buffer、未均匀 expert load、full-layer materialization 和 allocator overhead。真实显存不能只做 `total / GPU count`。

## Qwen3-30B-A3B：一次 forward 的 activation 例子

固定一个与本地 ms-swift full-SFT 示例相近的逻辑输入：

```text
B = 1 sequence
T = 8192 tokens
dtype = BF16 = 2 bytes/element
```

单层逻辑 tensor：

| Tensor | Shape / elements | BF16 大小 |
|---|---:|---:|
| Residual / layer input | `[1,8192,2048]` | 33.6 MB |
| Q | `[1,8192,4096]` | 67.1 MB |
| K | `[1,8192,512]` | 8.4 MB |
| V | `[1,8192,512]` | 8.4 MB |
| Router logits | `[1,8192,128]` | 2.1 MB；若 FP32 则 4.2 MB |
| Expert assignments | `8192 × 8 = 65,536` | indices/weights 另算 |
| One gate or up intermediate | `[65,536,768]` | 100.7 MB |
| Token dispatch buffer | `[65,536,2048]` | 268.4 MB |
| Naive attention score | `[1,32,8192,8192]` | 4.295 GB |
| 48 layer-input checkpoints only | `48×[1,8192,2048]` | 1.611 GB |

这些是 logical shapes，不是最终 peak-memory 求和表：

- FlashAttention 不 materialize 完整 4.295GB attention score；
- grouped GEMM、permutation fusion 会改变 dispatch/intermediate 的同时存活集合；
- full recomputation 通常只长期保存 layer boundary checkpoint，然后在 backward 重算内部 tensor；
- router logits、softmax/reduction 可能使用 FP32；
- expert load imbalance 会改变每个 rank 的 local assignments；
- pipeline/expert/tensor/sequence parallel 会改变每卡 local shape。

因此正确流程是：

```text
先用 shape 算 logical bytes
→ 再根据 kernel fusion 和 tensor lifetime 判断哪些会 materialize
→ 再根据 checkpointing 判断哪些会保存到 backward
→ 最后用 memory snapshot / profiler 校准 peak
```

## 与本地 ms-swift 训练路线的连接

Pinned ms-swift upstream snapshot 已包含 Qwen3-30B-A3B-Base 的 Megatron full-SFT 示例：

- [`examples/megatron/moe/qwen3_moe.sh`](https://github.com/modelscope/ms-swift/blob/565a1ad586a21d24b23931c52d2c62b49c39bee8/examples/megatron/moe/qwen3_moe.sh)

该参考脚本使用：

```text
pipeline parallel size = 2
expert parallel size = 8
sequence parallel = true
micro batch size = 1
max length = 8192
full recomputation
FlashAttention
```

脚本注释及本地文档给出的参考 benchmark 是双机 16×A800；Megatron 路线约 `16 × 60GiB`、`9.6s/it`。这是特定版本、硬件、数据和配置下的观测值，不是 H100 租卡数量的直接预测。

## 公式速查

### Parameter-proportional states

```text
M_states = P_total × bytes_per_parameter
```

### Ideal sharded average（只作为下界）

```text
M_states_per_gpu_ideal = M_states / world_size
```

更准确的 ZeRO/FSDP 分类账：

```text
M_states_per_rank ≈
  M_weights / s_weights
  + M_gradients / s_gradients
  + (M_master + M_Adam) / s_optimizer
```

- ZeRO-1：主要分片 optimizer states；
- ZeRO-2：继续分片 gradients；
- ZeRO-3 / FSDP full-shard：继续分片 parameters；
- TP/PP/EP 会进一步改变 layer、matrix、expert 的 ownership，不能把所有状态无条件除以 `world_size`。

### 峰值显存

```text
M_peak ≈
  local model states
  + saved activations
  + temporary/workspace
  + communication buffers
  + 当前 materialized full parameters
  + CUDA/framework/allocator overhead
```

### MoE 需要同时记两本账

```text
capacity / optimizer-state book → P_total
per-token compute book           → P_active
```

## 资料来源

- [Qwen3-30B-A3B official model card](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [Qwen3-30B-A3B official config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
- [Hugging Face Transformers — Qwen3 MoE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [How to Scale Your Model — Transformer Math](https://jax-ml.github.io/scaling-book/transformers/)
- [Hugging Face — GPU memory usage](https://huggingface.co/docs/transformers/model_memory_anatomy)
- [NVIDIA Transformer Engine — low-precision training and master weights](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/introduction/introduction.html)
- [PyTorch — activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [DeepSpeed — memory estimators for ZeRO](https://deepspeed.readthedocs.io/en/stable/memory.html)
