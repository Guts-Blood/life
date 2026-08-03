# 2026-06-13 学习记录：How To Scale Your Model Part 0 / Part 1

今天读的是 JAX Scaling Book 的 Part 0: Introduction 和 Part 1: All About Rooflines。

- Part 0: [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)
- Part 1: [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)

## 今天学了什么

### 1. Scaling 不是玄学，而是物理约束问题

Part 0 最重要的是把“大模型 scaling”从经验问题拉回到系统问题：模型能不能高效变大，不只取决于算法指标，也取决于它能不能在真实硬件上高效运行。

这里的核心目标是 strong scaling：增加更多芯片之后，吞吐也应该尽量按比例增加。但问题是，多加芯片通常会减少单个芯片上的计算量，同时引入更多芯片间通信。如果通信不能被计算覆盖，继续堆卡就不会线性加速，甚至会浪费硬件。

所以，模型结构、并行策略、batch size、sequence length、sharding 方式，都不能只从机器学习视角看，还要从硬件利用率和通信瓶颈视角看。

### 2. Roofline 是判断瓶颈的最小模型

Part 1 的主线是 Roofline：一个计算任务的性能上限，基本由两个东西决定：

- 算力上限：硬件每秒最多能做多少 FLOPs。
- 带宽上限：硬件每秒最多能搬多少 Bytes。

如果任务的 arithmetic intensity 足够高，也就是“每搬 1 byte 数据能做很多计算”，它就更可能是 compute-bound，能跑满计算单元。

如果 arithmetic intensity 很低，也就是“搬了很多数据但没做多少计算”，它就更可能是 bandwidth-bound，计算单元会等数据，硬件算力闲置。

关键公式是：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

硬件也有自己的临界点：

$$
\text{Critical Intensity} = \frac{\text{Peak FLOPs/s}}{\text{Bandwidth Bytes/s}}
$$

算法的 intensity 高于这个临界点，通常就是 compute-bound；低于这个临界点，通常就是 bandwidth-bound。

### 3. Matmul 的关键简化：intensity 近似等于本地 token 数 B

大模型里最核心的算子是矩阵乘法：

$$
X[B, D] \times Y[D, F] \rightarrow Z[B, F]
$$

在 bf16 下，近似计算量是：

$$
2BDF
$$

需要搬运的数据量是：

$$
2BD + 2DF + 2BF
$$

所以 arithmetic intensity 是：

$$
\frac{2BDF}{2BD + 2DF + 2BF}
$$

当 $D$ 和 $F$ 很大，而 $B$ 相对小的时候，分母主要由权重矩阵 $DF$ 主导，于是这个式子可以近似成：

$$
\text{Intensity} \approx B
$$

这是今天最关键的数学连接：大模型矩阵乘法能不能跑满硬件，很大程度上可以被简化成“每个模型副本上到底有多少 token”。

如果以 TPU v5e 为例，临界 arithmetic intensity 大约是 240 FLOPs/byte。于是工程上可以得到一个很直观的判断：

$$
B_{\text{local}} \gtrsim 240
$$

也就是说，如果每个独立模型副本上的本地 token 数足够大，核心 matmul 更可能进入 compute-bound 区域；如果为了并行把 batch 切得太碎，本地 token 数太小，就可能掉进 bandwidth-bound 区域。

这里的 $B_{\text{local}}$ 可以先用这个方式理解：

$$
B_{\text{local}} =
\frac{\text{global sequences} \times \text{average tokens per sequence}}
{\text{number of model replicas}}
$$

这个式子让我把 batch size、sequence length、卡数、副本数和硬件利用率真正连起来了。

### 4. H100 spec 里的 sparsity 坑

今天还记住了一个很容易踩的坑：NVIDIA spec sheet 里的 Tensor Core FLOPs 经常会标一个 with sparsity。这个数字不是 dense bf16 matmul 的真实可用算力，而是利用 structured sparsity 之后的理论峰值。

如果我们要算普通 dense bfloat16 矩阵乘法的 roofline，应该把这个 with sparsity 的 FLOPs/s 除以 2。

以 H100 SXM 为例，spec sheet 里 bfloat16 Tensor Core FLOPs 如果写成大约：

$$
1.979 \times 10^{15} \text{ FLOPs/s}
$$

但这个值是 with sparsity 的。dense bf16 真实应该按一半算：

$$
9.89 \times 10^{14} \text{ FLOPs/s}
$$

再除以 HBM bandwidth：

$$
3.35 \times 10^{12} \text{ Bytes/s}
$$

得到：

$$
B_{\text{crit}} = \frac{9.89 \times 10^{14}}{3.35 \times 10^{12}} \approx 295
$$

所以这里真正要记住的是：做 roofline 估算时，要先确认峰值算力是不是 dense 真实值。带 structured sparsity 的 advertised FLOPs 基本上要打半折，才能拿来估普通 dense matmul。

### 5. HBM 和芯片间网络是两层不同的瓶颈

今天还厘清了 HBM、ICI、DCN、PCIe 之间的关系。

HBM 是每个加速器自己的本地高带宽内存。计算核心不能凭空计算，它要从 HBM 读权重、activation、KV cache 等数据，算完再写回去。所以 HBM bandwidth 决定的是单芯片内部“数据喂给计算核心”的速度。

ICI / DCN / PCIe 是芯片之间的数据通道。当模型被切到多张卡上，或者不同副本之间要同步梯度时，数据要先从一颗芯片的 HBM 出来，通过芯片间网络传到另一颗芯片，再进入对方的 HBM。

因此分布式训练不是只有“单芯片 compute vs HBM bandwidth”的 Roofline，还会出现“多芯片 compute vs network bandwidth”的 Roofline。很多并行策略的本质，就是让芯片间通信足够少，或者让通信能被计算时间 overlap 掉。

## 真正的 Key Takeaway

今天真正带走的不是 240 这个数字本身，而是一个判断框架：

> 做大模型系统设计时，先问这个操作的瓶颈是什么：是算力、HBM 带宽、芯片间网络，还是内存容量。

更具体地说：

1. 模型 scaling 的目标不是“用了更多卡”，而是“用了更多卡之后吞吐接近线性增长”。
2. Roofline 给了一个一阶判断：算法的 FLOPs/Bytes 和硬件的 FLOPs/s / Bytes/s 谁更强。
3. 对 LLM 核心 matmul 来说，arithmetic intensity 可以近似理解为本地 token 数 $B_{\text{local}}$。
4. 所以 batch size、sequence length、replica 数量不是单纯的训练超参，它们直接决定硬件是否吃得饱。
5. 一个模型结构即使 benchmark 更好，如果让 roofline efficiency 明显下降，也可能不是一个真正可 scale 的好结构。

## 可以持续写的 Track

我觉得这里可以开一个长期 track：

### Track: Roofline-first LLM Systems Reasoning

这个 track 的目标不是背硬件参数，而是建立一套从模型结构到硬件瓶颈的估算能力。

可以按这个顺序继续记录：

1. Single-op roofline
   - matmul 的 FLOPs 和 Bytes 怎么算。
   - elementwise、norm、softmax 为什么更容易 bandwidth-bound。
   - 理论 arithmetic intensity 和 profiler 里实际表现为什么会不同。

2. Single-chip roofline
   - HBM bandwidth 和 peak FLOPs/s 如何共同决定临界点。
   - bf16、int8、mixed precision 为什么会改变或不改变临界 batch。
   - 什么情况下优化 kernel fusion 比换更强芯片更有效。

3. Multi-chip roofline
   - ICI / NVLink / DCN / PCIe 分别限制什么。
   - tensor parallel、data parallel、FSDP、pipeline parallel 分别引入什么通信。
   - 哪些通信可以 overlap，哪些会直接暴露成 step time。

4. Transformer-level roofline
   - attention、MLP、embedding、KV cache、optimizer state 分别受什么约束。
   - training 和 inference 的瓶颈为什么不同。
   - prefill 和 decode 为什么要分开看。

5. Real-model estimation
   - 给定一个模型规模、batch、sequence length 和卡数，手算一次 step time 下界。
   - 给定一个 serving 目标，估算吞吐、延迟、KV cache 和所需拓扑。
   - 用 profiler 验证理论估算和真实运行的差距。

这个 track 可以成为后续阅读 Part 2 到 Part 9 的主线：每读一章，都问同一个问题：

> 这一章新引入了什么瓶颈？它改变的是 FLOPs、Bytes、Bandwidth、Memory Capacity，还是 Communication Pattern？

## 下一步问题

后面继续读的时候，我要重点追这几个问题：

1. 当 $B_{\text{local}}$ 不够大时，除了增大 batch，还有没有其他方式提高 effective intensity？
2. Tensor parallel 把矩阵切开之后，瓶颈从 $B$ 转向 $D$ 或 $F$ 的具体条件是什么？
3. 对 inference 来说，prefill 和 decode 的 roofline 为什么完全不一样？
4. Profiler 里看到的实际瓶颈，怎么和手算的 Roofline 对上？
5. 一个新模型结构如果论文指标更好，应该如何快速判断它是否会损害 scaling efficiency？
