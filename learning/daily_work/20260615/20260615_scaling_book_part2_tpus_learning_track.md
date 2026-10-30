# 2026-06-15 学习记录：How To Scale Your Model Part 2

今天读的是 JAX Scaling Book 的 Part 2: How to Think About TPUs，也就是 All About TPUs 这一章。

- Part 2: [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/)
- 前置主线：Part 1 的 [Rooflines](https://jax-ml.github.io/scaling-book/roofline/)

## 今天学了什么

### 1. TPU 可以先简化成一台矩阵乘法机器

Part 2 最重要的视角是：不要一上来把 TPU 想成复杂黑盒，先把它想成一个非常强的矩阵乘法机器，再逐层加上内存和网络约束。

一颗 TPU 的核心结构可以先简化成：

```text
HBM -> VMEM -> VREG -> MXU / VPU -> VREG -> VMEM -> HBM
```

其中：

- HBM 是容量大的本地高带宽内存，用来存权重、activation、KV cache、optimizer state 等大块张量。
- VMEM 是 TensorCore 内部的片上 scratchpad，比 HBM 小很多，但离计算单元近，带宽高很多。
- VREG 是更靠近计算单元的寄存器层，MXU 和 VPU 都围绕它做数据交接。
- MXU 负责矩阵乘法，是 TPU 的主力计算单元。
- VPU 负责向量、逐元素、activation、reduction 等更通用但 arithmetic intensity 更低的操作。

所以 TPU 的计算不是“从 HBM 直接算”，而是数据必须被搬到片上存储层级里，才能喂给 MXU/VPU。

### 2. MXU 和 VPU 的大小差异来自算术强度

今天对 MXU/VPU 的理解更清楚了：MXU 之所以远大于 VPU，不是因为向量操作不重要，而是因为向量操作很难靠堆更多 ALU 提速。

矩阵乘法有 $O(n^3)$ 计算量和 $O(n^2)$ 数据量。一个数据块搬进来后，可以被重复使用很多次，所以 arithmetic intensity 很高，适合用巨大的 systolic array 把算力堆满。

逐元素操作、activation、简单 reduction 往往是 $O(n)$ 计算量和 $O(n)$ 数据量。每搬一个元素只做很少计算，所以它们更容易被内存带宽卡住。此时就算把 VPU 做得像 MXU 一样大，大量计算单元也会因为等数据而闲置。

这解释了 TPU 的不对称设计：把主要面积和功耗预算给 MXU，因为 LLM 的主计算负载是 matmul；VPU 保持足够处理向量操作和融合后的后处理即可。

### 3. VMEM 是 Part 1 Roofline 的一个关键补丁

Part 1 里我们主要用 HBM bandwidth 算 roofline，所以会得到类似：

$$
B_{\text{crit}} \approx \frac{\text{Peak FLOPs/s}}{\text{HBM bandwidth}}
$$

Part 2 加了一层重要现实：如果数据已经在 VMEM 里，而不是每次都从 HBM 读，那么通信带宽变成 VMEM 到计算单元的带宽。

官方文档里提到，VMEM bandwidth 大约是 HBM bandwidth 的 22 倍。这个数量级会极大改变临界 batch。

例如同一个 matmul：

- 从 HBM 读权重时，可能要 $B > 267$ 才能 FLOPs-bound。
- 如果权重已经在 VMEM，临界值可以降到 $B > 11$，实际考虑竞争后大概接近 20。

这让我意识到：Roofline 不是只有一条线，而是取决于你从哪一级存储系统取数据。

同一个算子，如果数据从 HBM 来，可能 bandwidth-bound；如果数据已经被 prefetch 到 VMEM，就可能变成 compute-bound。

### 4. TPU 网络不是全互联，拓扑会影响通信时间

Part 2 还把 TPU 的多芯片通信讲清楚了。TPU 芯片之间主要通过 ICI 连接，但它不是任意两颗芯片之间都有直连，而是 nearest-neighbor topology。

常见结构是：

- v5e / v6e：更像 2D torus。
- v4 / v5p：更像 3D torus。
- 小 topology 如果没有 wraparound，远端通信就要多跳。

这点非常关键：在 GPU 集群里，我们常常先想交换机和近似全互联；但 TPU 的思维方式更像“数据要沿着网格走”。所以通信时间不只看 bytes / bandwidth，还要看 hop count 和 topology。

可以粗略记成：

$$
T_{\text{comm}} \approx T_{\text{latency per hop}} \times \text{hops} + \frac{\text{bytes}}{\text{available bandwidth}}
$$

这就是为什么 sharding strategy 不能只看切得是否数学正确，还要看切出来的数据流是不是贴合 TPU 拓扑。

### 5. HBM、ICI、PCIe、DCN 是一组速度阶梯

今天把 TPU 系统里的通信层级重新排了一遍：

1. HBM bandwidth：TensorCore 和本地 HBM 之间，通常是 TB/s 级别。
2. ICI bandwidth：TPU 芯片和近邻 TPU 芯片之间，通常是 GB/s 到百 GB/s 级别。
3. PCIe bandwidth：CPU host 和 TPU tray/chip 之间，比 HBM 慢很多。
4. DCN bandwidth：host 到 host，跨 slice 或跨 pod 时用，通常更慢，也更容易成为瓶颈。

这说明一个操作到底慢不慢，不只取决于计算量，而取决于数据有没有掉到更慢的层级。

特别是 DCN：如果一个 TPU 到另一个 TPU 的数据必须先经过 PCIe 到 host，再过 DCN 到另一个 host，再过 PCIe 回 TPU，那它已经不是普通的“芯片间通信”了，而是走到了非常慢的系统路径。

### 6. MXU 有形状约束，小矩阵会被 padding

另一个容易忽视的点是 MXU 的 tile size。很多 TPU generation 的 MXU 是 128x128，v6e 是 256x256。

这意味着矩阵维度如果太小或不能对齐，就会被 padding 到 MXU 需要的块大小。算术上你以为只算了很小的矩阵，硬件上可能实际跑了一个被补齐后的块。

所以 TPU 性能不是只由 FLOPs 公式决定，还和矩阵形状是否适配 MXU tile 有关。小矩阵、奇怪维度、没有对齐的 hidden size，可能会造成实际利用率下降。

## 真正的 Key Takeaway

今天真正带走的是：Part 2 把 Part 1 的 Roofline 从抽象公式变成了具体机器模型。

> TPU 性能分析的第一步，是把每个操作映射到它使用的计算单元、存储层级和通信路径。

更具体地说：

1. 不要只问一个算子有多少 FLOPs，要问它的数据从 HBM、VMEM、ICI、PCIe 还是 DCN 来。
2. MXU 强大是因为 matmul 的 arithmetic intensity 足够高；VPU 小是因为向量操作通常被带宽限制。
3. VMEM prefetch 可以把某些操作从 HBM roofline 切到 VMEM roofline，临界 batch 会显著下降。
4. TPU 的 ICI 是拓扑网络，不是任意芯片直连；sharding 要考虑 hop count 和 wraparound。
5. MXU tile size 会让小矩阵或不规则矩阵被 padding，真实计算量可能比数学公式更大。

一句话总结：

> TPU 不是“更快的 GPU”，而是一套围绕 matmul、片上 scratchpad、邻居互联和拓扑通信设计出来的系统。

## 可以持续写的 Track

我觉得这里可以把 2026-06-13 的 `Roofline-first LLM Systems Reasoning` 继续扩展成一个子 track：

### Track: TPU-aware Roofline Reasoning

这个 track 的目标是：看到一个模型结构或并行策略时，能快速判断它在 TPU 上到底会被哪一层硬件限制。

可以按这个顺序继续记：

1. TPU single-chip dataflow
   - HBM、VMEM、VREG、MXU、VPU 的职责边界。
   - 一个 matmul 从读 HBM 到写回 HBM 的完整流水线。
   - 哪些中间结果能留在 VREG/VMEM，哪些必须写回 HBM。

2. VMEM-first optimization
   - 哪些权重或 activation 有机会被 prefetch 到 VMEM。
   - attention 和 MLP 之间能不能互相 hide load latency。
   - VMEM 容量不足时，哪些算子会重新掉回 HBM roofline。

3. MXU shape efficiency
   - 128/256 tile 对 hidden size、head dim、FFN dim 的约束。
   - padding 后的真实 FLOPs 如何估算。
   - 小 batch、小矩阵、MoE expert routing 为什么可能降低利用率。

4. TPU topology-aware sharding
   - data parallel、tensor parallel、pipeline parallel 分别对应什么通信图。
   - ICI 上的 all-reduce、all-gather、reduce-scatter 会经过多少 hop。
   - 什么 topology 有 wraparound，什么 topology 没有。

5. Slow path avoidance
   - 什么情况下数据会掉到 PCIe 或 DCN。
   - host offload 为什么可能毁掉 latency。
   - multi-slice training 中哪些通信必须尽量少。

这个 track 可以作为后面读 Part 3 Sharded Matmuls 的入口。下一章应该重点问：

> 每一种 sharded matmul 到底是在省 FLOPs、分摊 memory，还是把瓶颈转移到了 ICI？

## 下一步问题

后面继续读的时候，我要重点追这几个问题：

1. Tensor parallel 里不同矩阵切法分别会触发 all-gather、reduce-scatter 还是 all-reduce？
2. 一个 sharding strategy 的通信量如何按 HBM/ICI/DCN 分层估算？
3. Transformer 的 MLP 和 attention 哪些部分最适合用 VMEM prefetch hide 掉？
4. MXU padding 对 LLM hidden size、head dim、MoE expert size 的影响到底有多大？
5. 如果 profiler 显示 VPU 或 HBM bottleneck，应该优先改模型结构、kernel fusion，还是并行策略？
