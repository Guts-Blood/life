# Day 04 Reading Quiz — Sharding、Collective 与分布式诊断

开始日期：`2026-07-27`

状态：`completed`

方式：Socratic quiz；一次只讨论一道题，先回答、再纠错、再进入下一题。

## Quiz 目标

| 模块 | 希望真正掌握的能力 | 状态 |
|---|---|---|
| A. Sharded matmul | 从 global/local shape 判断输出是完整块还是 partial sum | `completed` |
| B. Collective 与布局 | 从目标输出布局推出 AllGather、AllReduce、ReduceScatter 或 AllToAll | `completed` |
| C. 训练状态 | 说明 DP、FSDP/ZeRO、TP、PP 下参数、梯度、优化器状态和 activation 的布局 | `completed` |
| D. 配置与并行组 | 计算 world size、DP/TP/PP group 和 global batch | `completed` |
| E. 故障诊断 | 对 OOM、hang、checkpoint mismatch 给出有证据顺序的首轮排查 | `completed` |

题目会根据回答动态增加或跳过追问；不重复 Day 01–03 已经稳定掌握的纯 Roofline 推导。

## 当前进度

- D4-Q1：`passed`
- D4-Q2：`passed`
- D4-Q3：`passed`
- D4-Q4：`passed`
- D4-Q5：`passed`
- D4-Q6：`passed`
- D4-Q7：`passed_after_teaching`
- D4-Q8：`passed`
- D4-Q9：`completed_guided`
- D4-Q10：`passed_core; evidence_order_taught`
- D4-Q11：`passed_after_teaching`
- Day 04 Reading Quiz：`completed`
- 后续实践重点：在实际训练 repo 中识别 checkpoint format、rank divergence 与 memory timeline。

## 当前题目

### D4-Q1 — 从 local shard 推出 collective

有矩阵乘法：

```text
A[I,J] @ B[J,K] -> C[I,K]

A.shape = [8,12]
B.shape = [12,16]
world_size = 2
```

现在沿收缩维 `J` 把 `A` 和 `B` 对齐切到两个 rank：

```text
rank 0:
  A0 = A[:, 0:6]    # [8,6]
  B0 = B[0:6, :]    # [6,16]
  P0 = A0 @ B0      # [8,16]

rank 1:
  A1 = A[:, 6:12]   # [8,6]
  B1 = B[6:12, :]   # [6,16]
  P1 = A1 @ B1      # [8,16]
```

请先独立回答四点：

1. `P0` 或 `P1` 是否已经是完整的 `C`？为什么？请从 dot product 覆盖的 `J` 范围解释。
2. 用一个等式写出完整 `C` 与 `P0`、`P1` 的关系。
3. 如果希望两个 rank 最终都持有完整的 `C[8,16]`，应使用什么 collective？它做了哪两个逻辑动作？
4. 如果下游只希望每个 rank 持有 `C` 的一部分，而不是都复制完整 `C`，可以改用什么 collective？它与第 3 问的最终布局有何不同？

暂时不要讨论 ring 实现或通信成本；这里只考察 `global shape -> local computation -> output state -> collective` 的因果链。

## 回答与点评

### D4-Q1 第一轮回答

学习者回答：

```text
1. 肯定不是，这是被切分了的。
2. C = P0 + P1。
3. AllReduce。
4. ReduceScatter。
```

点评：

1. 结论正确，但“被切分了”还不够精确。这里切的是收缩维 `J`：`P0[i,k]` 只累加 `j=0..5`，`P1[i,k]` 只累加 `j=6..11`。两者的 shape 都已经是 `[8,16]`，缺少的是另一半 `J` 对应的数值贡献，而不是缺少一块输出坐标。因此它们是同形状的 partial sums。
2. `C=P0+P1` 正确；这里是同一 `[i,k]` 位置上的 elementwise sum。
3. `AllReduce` 正确。以 `SUM` 为 reduce op 时，它在逻辑上先规约各 rank 的 partial sums，再把完整结果复制到所有 rank；最终两个 rank 都持有相同的 `C[8,16]`。
4. `ReduceScatter` 正确。它先完成同样的 elementwise reduction，再把规约后的 `C` 沿约定维度切成互不重叠的 shards；每个 rank 只持有其中一块，而不是完整复制。

判定：`core collective mapping passed; explanation/layout needs_followup`。

### D4-Q1 定向追问

假设 `ReduceScatter` 最终约定沿 `K` 维切分：

```text
C.shape = [8,16]
world_size = 2
```

只回答两点：

1. rank 0 和 rank 1 最终分别持有 `C` 的哪个 slice？各自的 local shape 是什么？
2. 为什么不能把 `P0`、`P1` 沿 `K` 维直接 concat 成完整 `C`？

学习者回答：

```text
肯定不行；这是两个不同的事情，但暂时不知道怎样严格表达。
```

严格表达：

```text
P0[i,k] = Σ_{j∈J0} A[i,j]B[j,k]
P1[i,k] = Σ_{j∈J1} A[i,j]B[j,k]
```

`P0`、`P1` 的输出坐标域相同，都是所有 `(i,k)`；区别只在每个坐标累加了哪个 `J` 子集。因此它们是同一输出坐标的不同数值贡献，正确组合是 elementwise reduction：

```text
C[i,k] = P0[i,k] + P1[i,k]
```

沿 `K` concat 会产生 `[8,32]`：前 16 列仍只是 `J0` 的 partial sums，后 16 列仍只是 `J1` 的 partial sums。它既改变了目标 shape，也没有完成任何一个 `C[i,k]` 所需的完整求和。

只有当两个 rank 持有互不重叠的输出坐标时才能沿 `K` concat。例如：

```text
C0 = A @ B[:, 0:8]    # C[:, 0:8], shape [8,8]
C1 = A @ B[:, 8:16]   # C[:, 8:16], shape [8,8]
C  = concat_K(C0, C1)
```

判定：`passed`。学习者已区分两种情况；补齐形式化表达。

### D4-Q2 — AllGather 还是 AllReduce

现在不是切收缩维 `J`，而是沿输出维 `K` 切参数：

```text
W.shape = [12,16]

rank 0: W0 = W[:, 0:8]    # [12,8]
rank 1: W1 = W[:, 8:16]   # [12,8]
```

某个后续计算要求每个 rank 都临时持有完整的 `W[12,16]`。请回答：

1. 应使用 `AllGather` 还是 `AllReduce`？
2. 为什么？请用“输出坐标”或“数值贡献”解释。
3. collective 完成后，每个 rank 持有的 local shape 是什么？与操作前相比，参数 working set 变成多少倍？

学习者回答：

```text
1. AllGather。
2. 因为这是矩阵不同坐标的切分到聚集，而不是求和。
3. 都拥有完整的 slice，2 倍。
```

点评：

1. `AllGather` 正确。
2. 解释正确。`W0`、`W1` 对应互不重叠的 `K` 坐标，不是同一坐标的 partial contributions，因此应拼合坐标块，不做数值 reduction。
3. 倍数正确。更精确地说，完成后每个 rank 持有完整 tensor `W[12,16]`，不再只是一个 slice；每个 rank 的参数 working set 从 `12×8=96` 个元素变为 `12×16=192` 个元素，即 `2×`。

判定：`passed`。

### D4-Q3 — AllToAll 改变谁持有什么

考虑一个简化的 MoE token routing。两个 rank 分别承载 expert 0 和 expert 1。路由前：

```text
rank 0 持有:
  token a -> expert 0
  token b -> expert 1

rank 1 持有:
  token c -> expert 0
  token d -> expert 1
```

路由后希望：

```text
rank 0 持有: token a, token c    # 都交给 expert 0
rank 1 持有: token b, token d    # 都交给 expert 1
```

请回答：

1. 应使用哪个 collective？
2. 这个过程是否需要把 token 表示做 elementwise sum？
3. 它与 AllGather 的关键区别是什么：操作后每个 rank 是拥有全部 token，还是只拥有发给自己的 token？

学习者回答：

```text
1. AllToAll。
2. 不需要。
3. 只是重新分布。
```

点评：

- 三项均正确。AllToAll 根据目的地重新分布 token ownership，不做 elementwise reduction。
- 与 AllGather 的区别是：AllGather 后每个 rank 都拥有全部输入 shards；AllToAll 后每个 rank 只拥有路由到自己的 token。

判定：`passed`。Collective 与布局模块完成。

### D4-Q4 — DDP 与 FSDP：persistent state 和临时 working set

假设用两个 rank 训练同一个模型，优化器为 Adam，不考虑 CPU offload、activation checkpointing 和混合精度细节。

配置 A 是普通 DDP；配置 B 是 FSDP `FULL_SHARD`。请填写下表中的“复制”或“分片”，并回答 collective：

| 状态/时刻 | DDP | FSDP `FULL_SHARD` |
|---|---|---|
| step 之间持久保存的 parameters | ? | ? |
| optimizer states | ? | ? |
| gradient 同步完成后的 gradients | ? | ? |
| 某一层 matmul 执行前所需 parameters | ? | ? |

然后回答：

1. DDP 在 backward 中主要用哪个 collective 同步 gradients？
2. FSDP 在一层计算前为什么需要 AllGather parameters？
3. FSDP 在 backward 后为什么更适合 ReduceScatter gradients，而不是 AllReduce 后让每个 rank 都保留完整 gradients？

注意区分：

```text
persistent state：大部分时间由 rank 长期持有的状态
temporary working set：某层实际计算时临时 materialize 的状态
```

学习者第一轮回答：

```text
DDP 是复制，FSDP 是分片；DDP 只有数据是切分的。
1. AllToAll。
2. 因为没有完整数据分片。
3. 不需要，每层只需要那一层的 gradient。
```

点评：

- “DDP 复制模型状态、不同 rank 处理不同数据”总方向正确，但表格尚未逐项填写。
- 普通 DDP 的 gradient synchronization 是 `AllReduce`，不是 `AllToAll`。
- FSDP 在计算前 AllGather 的对象是该层的 parameter shards；目的不是获得完整训练数据，而是临时 materialize 普通 layer kernel 所需的完整 parameters。
- “每层只需要该层 gradient”解释了 layer-wise 生命周期的一部分，但没有回答为什么使用 ReduceScatter。各 data rank 基于不同 local micro-batch 得到不同的 gradient contributions，仍需跨 rank 求和/平均；ReduceScatter 在完成 reduction 的同时，只让每个 rank 保留自己负责的 parameter/optimizer shard 对应的 gradient shard。

正确状态表：

| 状态/时刻 | DDP | FSDP `FULL_SHARD` |
|---|---|---|
| step 之间持久保存的 parameters | 复制 | 分片 |
| optimizer states | 复制 | 分片 |
| gradient 同步完成后的 gradients | 复制 | 分片 |
| 某一层 matmul 执行前所需 parameters | 复制/完整 | 临时 AllGather 为复制/完整 |

判定：`DDP/FSDP direction recognized; collective objects need_retry`。

### D4-Q4 定向复测

两个 data-parallel rank 分别用不同 local micro-batch 算出同一层完整参数 `W` 的本地 gradient contribution：

```text
rank 0: G0 = ∂L_local0/∂W
rank 1: G1 = ∂L_local1/∂W
```

只回答三点：

1. 普通 DDP 做完 gradient AllReduce 后，rank 0、rank 1 各持有什么？用 `G0`、`G1` 写出结果。
2. FSDP 做 ReduceScatter 后，跨 rank 是否仍然完成了 `G0+G1` 的 reduction？每个 rank 最终保留完整 gradient 还是其中一个 shard？
3. FSDP 在某层 matmul 前 AllGather 的究竟是 training samples、activations、parameters，还是 gradients？

学习者第二轮回答：

```text
1. 完整的 gradient。
2. 是的，但是分别保留 G0、G1 的切片。
3. weights、gradient，以及矩阵乘法需要的参数和状态切片。
```

点评：

1. 正确。更完整地说，两个 rank 都持有相同的 aggregated full gradient，通常是 `(G0+G1)/2`；若只描述 SUM reduction，则写作 `G0+G1`。
2. “完成 reduction”正确，但最终对象需要纠正：每个 rank 保留的是聚合结果 `G0+G1` 的不同 shard，不是分别保留 `G0`、`G1` 的 shard。
3. FSDP forward matmul 前 AllGather 的是该层 parameters，例如 weight/bias。gradient 尚未产生；Adam optimizer states 不参与 forward matmul，仍保持分片。

最小符号例子：

```text
G0 = [a0, a1]
G1 = [b0, b1]

ReduceScatter-SUM:
rank 0 -> [a0 + b0]
rank 1 -> [a1 + b1]
```

最终确认回答：

```text
1. AllGather 计算所需的 parameter shards。
2. 每个 rank 保留 G0+G1 结果矩阵的一个 shard。
```

判定：`passed`。

### D4-Q5 — TP 中 activation 布局为什么会变化

仍然计算：

```text
X[8,12] @ W[12,16] -> Y[8,16]
```

两个 TP rank 有两种切法。

Case A：沿输出维 `K` 切 `W`：

```text
rank 0: W0 = W[:, 0:8]    # [12,8]
rank 1: W1 = W[:, 8:16]   # [12,8]
```

Case B：沿收缩维 `J` 切 `W`，并对齐切 `X`：

```text
rank 0: X0 = X[:, 0:6],  W0 = W[0:6, :]    # [8,6] @ [6,16]
rank 1: X1 = X[:, 6:12], W1 = W[6:12, :]   # [8,6] @ [6,16]
```

请回答：

1. Case A 中，每个 rank 计算出的 local output shape 是什么？两个输出是不同坐标的 shards，还是同一坐标的 partial sums？
2. Case A 若需要完整 `Y`，应该 concat/AllGather，还是 AllReduce？
3. Case B 中，两个 local output 的 shape 是什么？完整 `Y` 与它们是什么关系，需要哪种 reduction collective？
4. 根据这两个 case，“TP 的 activation 永远是分片的”这句话是否准确？为什么？

学习者回答：

```text
1. [8,8]，不同坐标。
2. AllGather/concat。
3. [8,16]，需要 AllReduce 相加。
4. 不准确；TP 可以分片但不一定永远分片，可能取决于矩阵大小。
   大多数情况下应该是分片的，本质是把矩阵计算分成 partial 矩阵计算。
```

点评：

1. 正确。Case A 的两个 `[8,8]` local outputs 对应互不重叠的 `K` 坐标。
2. 正确。需要完整 `Y` 时拼合不同坐标，使用 concat/AllGather，而不是数值 reduction。
3. 正确。Case B 的两个 `[8,16]` local outputs 是同一输出坐标的 partial sums，完整 `Y=P0+P1`，需要 AllReduce；若目标仍是分片输出，也可使用 ReduceScatter。
4. “不永远分片”正确，但布局不是由矩阵大小直接决定。矩阵大小会影响是否值得采用 TP、采用多大 TP degree；具体 activation 是复制、分片还是未规约 partial，取决于 parameter sharding 维度、当前计算产生的布局、下游期望布局和所选 collective。

需要修正的概括：

- column-parallel 产生的是不同输出坐标上的完整值，不是 partial sums；
- row-parallel 沿收缩维计算，才产生同一输出坐标的 partial sums；
- 经典 column-parallel/row-parallel 组合会让 activation 在复制与分片布局之间切换，以避免每层都 AllGather 完整 activation。

判定：`shape and collective passed; activation-layout cause needs_final_check`。

### D4-Q5 最终确认

按上述两个 case 填四个空：

```text
Case A（column-parallel）：
input X  = __________
output Y = __________（不做 AllGather 时）

Case B（row-parallel + AllReduce）：
input X  = __________
output Y = __________（AllReduce 后）
```

学习者回答：

```text
Case A:
input X  = [8,12]，完整复制
output Y = [8,8]，分片的 Y

Case B:
input X  = 分片的 X；W 也沿收缩维分片
output Y = AllReduce 后的完整 Y[8,16]
```

点评：四项均正确。

判定：`passed`。

### D4-Q6 — Pipeline Parallelism 切了什么

把一个 24-layer Transformer 切成两个 pipeline stages：

```text
rank 0 / stage 0: layers 0..11
rank 1 / stage 1: layers 12..23 + loss
```

暂时不叠加 DP、FSDP 或 TP，只看纯 PP。对一个 micro-batch：

```text
tokens -> stage 0 -> boundary hidden state h -> stage 1 -> loss
```

请回答：

1. 两个 rank 是否都持有完整模型 parameters 和完整 Adam optimizer states？各自实际持有什么？
2. forward 时，stage 0 需要把什么对象发送给 stage 1？这是 parameter AllGather、gradient AllReduce，还是 stage 间的 activation send/recv？
3. backward 时，stage 1 需要把什么对象传回 stage 0？
4. 为什么只使用一个 micro-batch 时会有明显 pipeline bubble？增加多个 micro-batches 主要改善的是模型显存、单次 matmul shape，还是 stage 间的时间重叠？

学习者第一轮回答：

```text
1. 取决于 FSDP 或 DDP；FSDP 不需要 Adam states，只需要本层 parameters，
   forward 还需保留 activation；DDP 需要。
2. 发送 activation X；具体可能取决于架构，FSDP 可能使用 ReduceScatter；
   主要目的是 activation send/recv。
3. backward 需要 gradient 和 Adam states，传完后 activation 可以清除。
4. 不理解 pipeline bubble；猜测多个 micro-batches 主要改善显存和 arithmetic intensity，
   但提高 OOM 风险。
```

点评：

1. 题目限定纯 PP。每个 stage 只持有自己负责的 layers 的 parameters、parameter gradients 和对应 Adam states。Adam states 是优化器更新所必需的；FSDP 也需要，只是按 data-parallel ranks 分片或选择 offload。
2. 核心正确：stage 0 向 stage 1 发送 boundary activation `h`。这是 stage 间 point-to-point send/recv，不是 FSDP ReduceScatter；FSDP/DP/TP 可以与 PP 正交叠加，但不改变这条 PP 边界语义。
3. stage 1 传回的是 `∂L/∂h`，即 boundary activation 的 gradient。parameter gradients 和 Adam states 都留在拥有相应 layers 的本地 stage，不跨 PP 边界传递。某个 activation 在其 backward 完成后才可释放；是否保存或重算还受 activation checkpointing 影响。
4. pipeline bubble 是某个 stage 因依赖尚未到达或流水线正在填充/排空而空闲的时间槽。增加 micro-batches 的主要收益是让不同 stages 同时处理不同 micro-batches，提高 stage 间时间重叠和设备利用率。若每个 micro-batch shape 不变，单次 matmul shape/arithmetic intensity 并不会因此自动改变；多个 in-flight micro-batches 通常会增加 activation memory，而不是改善显存。

只看 forward 的最小时间线：

```text
一个 micro-batch：
time:    t0      t1
stage 0: F0      idle
stage 1: idle    F0

四个 micro-batches：
time:    t0      t1      t2      t3      t4
stage 0: F0      F1      F2      F3      idle
stage 1: idle    F0      F1      F2      F3
```

首尾仍有 fill/drain bubble，但中间两个 stages 可以同时工作。训练还需加入 backward 和具体调度（如 1F1B），核心含义不变。

判定：`forward boundary object passed; PP state/backward/bubble need_retry`。

### D4-Q6 定向复测

只回答三点：

1. 纯 PP 中，stage 0 是否保存 stage 1 的 Adam states？它保存哪些 Adam states？
2. backward 跨 `stage 1 -> stage 0` 边界发送的是 parameter gradient，还是 `∂L/∂h`？
3. 保持单个 pipeline micro-batch 的 shape 不变，增加一次 schedule 中的 micro-batch 数量，主要减少什么空闲现象？通常会让 in-flight activation memory 增加还是减少？

学习者第二轮回答：

```text
1. 保存，跨 step 的 Adam states 给 backward 使用。
2. 传 gradient、optimizer states。
3. 减少 stage 0/1 之间的时间重叠；in-flight activation memory 通常增加。
```

点评：

1. 不正确。stage 0 只保存自己 layers 对应 parameters 的 Adam states，不保存 stage 1 的 Adam states。Adam states 的确跨 step 持久存在，但由 `optimizer.step()` 使用，不由 backward 使用。
2. 不正确。stage 1 向 stage 0 只发送 boundary activation gradient `∂L/∂h`。stage 1 的 parameter gradients 和 optimizer states 都留在 stage 1。
3. activation memory 增加正确；时间方向说反。增加 pipeline micro-batches 会增加 stages 之间的时间重叠，从而减少 fill/drain bubble 和 idle fraction。

严格区分三个阶段：

```text
forward:
stage 0 -- h --> stage 1

backward:
stage 0 <-- ∂L/∂h -- stage 1
各 stage 使用本地 parameters、保存/重算的 activations 和上游 gradient，
计算自己的 parameter gradients。

optimizer step:
stage 0: Adam(theta0, dtheta0, m0, v0)
stage 1: Adam(theta1, dtheta1, m1, v1)
```

`m0/v0` 与 `m1/v1` 是各 stage 本地持久状态，不跨 PP boundary 传输。

判定：`activation-memory direction passed; state ownership/backward object/overlap direction need_retry`。

### D4-Q6 最终确认

只用下面六个符号填空：

```text
theta0：stage 0 parameters
theta1：stage 1 parameters
dtheta0/dtheta1：对应 parameter gradients
m0,v0 / m1,v1：对应 Adam states
h：forward boundary activation
dh：∂L/∂h
```

```text
forward，stage 0 -> stage 1：________
backward，stage 1 -> stage 0：________

stage 0 本地 optimizer.step 使用：________
stage 1 本地 optimizer.step 使用：________

增加 pipeline micro-batches：
stage 时间重叠 ________，bubble ________，in-flight activation memory 通常 ________
```

学习者回答：

```text
1. h
2. dtheta0/dtheta1，dh
3. theta0，m0，v0
4. theta1，m1，v1
5. 增加，减少，增加
```

点评：

1. `h` 正确：forward 跨 PP 边界发送 boundary activation。
2. 应只发送 `dh`。`dtheta0`、`dtheta1` 是各 stage 对本地 parameters 求出的 gradients，留在对应 stage，不跨 PP 边界传输。
3. stage 0 的 Adam update 还需要本地 parameter gradient，因此完整输入是 `theta0, dtheta0, m0, v0`。
4. 同理，stage 1 使用 `theta1, dtheta1, m1, v1`。
5. 三个方向均正确：micro-batches 增加后 stage 时间重叠增加、bubble 减少、in-flight activation memory 通常增加。

判定：`boundary activation and pipeline-bubble directions passed; parameter-gradient ownership needs_final_retry`。

### D4-Q6 最小复测

只填三个空：

```text
backward 跨 stage 1 -> stage 0 边界发送：________
stage 0 本地 optimizer.step 使用：________
stage 1 本地 optimizer.step 使用：________
```

学习者回答：

```text
1. dh
2. theta0, m0, v0, dh
3. theta1, m1, v1, dh
```

点评：

1. `dh` 正确。
2. `dh` 是 stage 0 backward 的上游输入，不是 Adam 的直接输入。stage 0 使用 `dh`、本地 forward 状态和 `theta0` 计算出 `dtheta0`；随后本地 Adam update 使用 `theta0, dtheta0, m0, v0`。
3. stage 1 同理：backward 计算出 `dtheta1`，本地 Adam update 使用 `theta1, dtheta1, m1, v1`。stage 1 还计算 `dh` 并发给 stage 0，但不把 `dh` 交给 Adam。

对象流：

```text
stage 1 backward -> dtheta1（留在 stage 1）+ dh（发给 stage 0）
stage 0 backward(dh) -> dtheta0（留在 stage 0）

Adam0(theta0, dtheta0, m0, v0)
Adam1(theta1, dtheta1, m1, v1)
```

判定：`PP boundary object passed; optimizer input still needs_retry`。

### D4-Q6 最终对象检查

只回答两个符号，不重复 parameters 和 Adam states：

```text
Adam0 更新 theta0 时使用的 gradient：________
Adam1 更新 theta1 时使用的 gradient：________
```

学习者回答：

```text
1. theta0, m0, v0, dtheta0/dtheta1
2. theta1, m1, v1, dtheta0/dtheta1
```

点评：

- 已确认 Adam 使用 parameter gradient，而不是 boundary gradient `dh`。
- 但两个 stage 不能都使用 `dtheta0/dtheta1`。纯 PP 中状态按 layer/stage ownership 放置：`theta0` 及其 gradient `dtheta0` 属于 stage 0，`theta1` 及其 gradient `dtheta1` 属于 stage 1。stage 0 不读取 `dtheta1`，stage 1 不读取 `dtheta0`。

判定：`optimizer gradient type recognized; stage-local gradient identity needs_final_confirmation`。

### D4-Q6 Stage-local ownership 最终确认

每行只能二选一：

```text
Adam0 更新 theta0：dtheta0 / dtheta1
Adam1 更新 theta1：dtheta0 / dtheta1
```

学习者确认继续 quiz。最终映射明确为：

```text
Adam0 更新 theta0：dtheta0
Adam1 更新 theta1：dtheta1
```

判定：`passed`。训练状态模块完成。

### D4-Q7 — World size、并行组与 global batch

有如下训练配置：

```text
TP = 2
PP = 2
DP = 2
B_micro = 4 sequences / DP rank / micro-step
gradient_accumulation = 8
sequence_length = 2048 tokens
```

为避免 rank 编号约定不清，使用：

```text
global_rank = ((dp_rank × PP) + pp_rank) × TP + tp_rank
```

请回答：

1. `world_size` 是多少？
2. 一个 optimizer step 的 `B_global` 是多少 sequences？一共消费多少 tokens？说明为什么 TP/PP degree 不再乘入 batch。
3. 写出全部 TP groups、PP groups 和 DP groups。
4. `global rank 5` 的 `(dp_rank, pp_rank, tp_rank)` 是什么？它分别属于哪个 TP、PP、DP group？

学习者回答：

```text
1. 什么是 world size？
2. 2048 × 4 × 8，一个 optimizer step。
3. 完全不懂。
4. 不懂，需要讲解。
```

点评与讲解：

#### 1. World size

`world_size` 是参与这个 distributed job 的全部 processes/ranks 数。常见的一卡一进程配置中，也就是训练使用的 GPU 总数。

三个相互正交的并行维度共同构成全部 ranks：

```text
world_size = DP × PP × TP
           = 2 × 2 × 2
           = 8
```

直观上：

- `TP=2`：两个 rank 合作计算同一个 pipeline stage；
- `PP=2`：两个 stages 合作执行同一个模型；
- `DP=2`：上述整套 `TP×PP=4` rank 的模型副本有两份，各自处理不同数据。

#### 2. Global batch 与 tokens

学习者的：

```text
2048 × 4 × 8 = 65,536 tokens
```

是一个 DP replica 在一个 optimizer step 处理的 tokens。跨两个 DP replicas 后：

```text
B_global
= B_micro × gradient_accumulation × DP
= 4 × 8 × 2
= 64 sequences

tokens / optimizer step
= B_global × sequence_length
= 64 × 2048
= 131,072 tokens
```

TP/PP 不乘入 global batch，因为它们是在合作处理同一批 samples：

- TP ranks 分担同一层的 tensor computation；
- PP ranks 分担同一模型的不同 layers；
- 只有不同 DP replicas 读取不同 samples，因此只有 DP 扩大 global batch。

#### 3. Rank 坐标

由：

```text
global_rank = ((dp_rank × PP) + pp_rank) × TP + tp_rank
```

得到：

| global rank | dp_rank | pp_rank | tp_rank |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 |

一个 parallel group 的构造规则是：固定另外两个坐标，只让本 group 对应的坐标变化。

```text
TP groups（固定 dp、pp，改变 tp）：
[0,1], [2,3], [4,5], [6,7]

PP groups（固定 dp、tp，改变 pp）：
[0,2], [1,3], [4,6], [5,7]

DP groups（固定 pp、tp，改变 dp）：
[0,4], [1,5], [2,6], [3,7]
```

#### 4. Rank 5

```text
rank 5 = (dp_rank=1, pp_rank=0, tp_rank=1)
```

因此：

```text
TP group：固定 dp=1, pp=0，改变 tp -> [4,5]
PP group：固定 dp=1, tp=1，改变 pp -> [5,7]
DP group：固定 pp=0, tp=1，改变 dp -> [1,5]
```

判定：`per-DP-replica token count recognized; world size, group construction, and DP factor taught; needs_transfer_check`。

### D4-Q7.1 — Rank group 迁移检查

沿用上面的 rank 表，只回答 rank 2：

```text
rank 2 的 (dp_rank, pp_rank, tp_rank) = ?
rank 2 的 TP group = ?
rank 2 的 PP group = ?
rank 2 的 DP group = ?
```

### D4-Q7 概念澄清 — DP、TP、PP 分别是什么

学习者在迁移检查前询问 DP、TP、PP 的含义。

| 并行方式 | 切分对象 | 各 rank 是否处理同一批 samples | 主要跨 rank 对象 |
|---|---|---|---|
| DP：Data Parallelism | 数据 batch | 否；不同 DP replicas 处理不同 samples | parameter gradients |
| TP：Tensor Parallelism | 同一层内部的 parameter/tensor 与 matmul | 是；多个 TP ranks 合作算同一层、同一批 samples | activation shards 或 partial sums |
| PP：Pipeline Parallelism | 模型的 layers/stages | 是；同一样本依次经过不同 stages | boundary activation `h` 与 gradient `dh` |

#### DP：切数据

每个 DP replica 都具有执行完整模型的能力，但读取不同的 local micro-batch：

```text
DP replica 0：samples A -> model -> local gradients G0
DP replica 1：samples B -> model -> local gradients G1
                           |
                   gradient reduction
```

因此 DP 扩大 global batch：

```text
B_global = B_micro × gradient_accumulation × DP
```

若 replica 内还使用 TP/PP，“完整模型副本”可以由多个 ranks 联合组成，不要求单个 rank 保存完整模型。

#### TP：切同一层内部的 tensor computation

例如把 linear weight `W[D,F]` 沿 `F` 切给两个 ranks：

```text
TP rank 0：W[:, 0:F/2] -> 计算 Y 的前半列
TP rank 1：W[:, F/2:F] -> 计算 Y 的后半列
```

两个 ranks 在合作处理同一层和同一批 samples，因此 TP 不扩大 global batch。它通常需要 AllGather、AllReduce 或 ReduceScatter 等 collective 来转换 tensor layout。

#### PP：切模型的 layers

例如 24-layer Transformer：

```text
PP stage 0：layers 0..11
      | forward 发送 h
      v
PP stage 1：layers 12..23 + loss
      | backward 发送 dh
      v
PP stage 0
```

两个 stages 依次处理同一样本，因此 PP 也不扩大 global batch。增加 pipeline micro-batches 可以增加 stages 间重叠并减少 bubble。

#### 本题 `DP=2, PP=2, TP=2` 的整体结构

```text
DP replica 0：
  PP stage 0 = TP ranks [0,1]  ->  PP stage 1 = TP ranks [2,3]

DP replica 1：
  PP stage 0 = TP ranks [4,5]  ->  PP stage 1 = TP ranks [6,7]
```

每个 DP replica 使用 `PP×TP=4` 个 ranks 联合执行完整模型；共有两个 replicas，所以：

```text
world_size = DP × PP × TP = 2 × 2 × 2 = 8
```

概念检查：

```text
1. 两组模型分别读取不同训练数据 -> DP
2. 两张卡合作计算同一个 linear layer -> TP
3. 一张卡负责前 12 层，另一张负责后 12 层 -> PP
```

判定：`passed`。

### D4-Q7 概念澄清 — Rank 坐标

`global rank` 是 distributed job 中某个 process 的唯一整数编号，范围是：

```text
0 .. world_size-1
```

在本题中是 `0..7`。`(dp_rank, pp_rank, tp_rank)` 是同一个 process 在三个并行维度上的逻辑坐标：

```text
dp_rank：它属于第几个 DP replica，范围 0..DP-1
pp_rank：它负责第几个 pipeline stage，范围 0..PP-1
tp_rank：它是该 stage 内第几个 TP worker，范围 0..TP-1
```

可把它类比为：

```text
(第几个模型副本, 第几个流水线 stage, stage 内第几个 tensor worker)
```

本题约定把三维坐标压成一个整数：

```text
global_rank = ((dp_rank × PP) + pp_rank) × TP + tp_rank
```

由于 `tp_rank` 是最内层坐标，它变化最快；然后是 `pp_rank`，最后是 `dp_rank`：

```text
(0,0,0) -> rank 0
(0,0,1) -> rank 1
(0,1,0) -> rank 2
(0,1,1) -> rank 3
(1,0,0) -> rank 4
(1,0,1) -> rank 5
(1,1,0) -> rank 6
(1,1,1) -> rank 7
```

从整数 rank 反解坐标：

```text
tp_rank = global_rank % TP
q       = global_rank // TP
pp_rank = q % PP
dp_rank = q // PP
```

以 rank 5 为例：

```text
tp_rank = 5 % 2 = 1
q       = 5 // 2 = 2
pp_rank = 2 % 2 = 0
dp_rank = 2 // 2 = 1

rank 5 -> (dp=1, pp=0, tp=1)
```

坐标的价值在于构造通信组：

- TP group：固定 `dp,pp`，只改变 `tp`；
- PP group：固定 `dp,tp`，只改变 `pp`；
- DP group：固定 `pp,tp`，只改变 `dp`。

注意：这种整数编号顺序是本题明确选择的 mapping convention，不是唯一可能的顺序；真实框架可能采用其他 rank mapping，必须查看其配置和 group 初始化逻辑。

### D4-Q7.2 — Rank 坐标检查

沿用 `DP=PP=TP=2`，只回答：

```text
rank 2 -> (dp_rank=?, pp_rank=?, tp_rank=?)
```

学习者确认已理解 rank 坐标并要求继续 quiz。D4-Q7 判定：`passed_after_teaching`。

### D4-Q8 — 固定 GPU 数时改变 TP

一台机器共有 16 张 GPU，全部用于一个训练 job：

```text
world_size = 16
TP = 2
PP = 2
B_micro = 2 sequences
gradient_accumulation = 8
```

请回答：

1. 当前 `DP` degree 是多少？
2. 当前 `B_global` 是多少 sequences？
3. 保持 `world_size=16`、`PP=2` 不变，把 `TP` 从 2 提高到 4，新 `DP` 是多少？
4. 若 `B_micro=2`、`gradient_accumulation=8` 不变，新 `B_global` 是多少？相对原来如何变化？
5. 若保持 `B_micro=2`，只调整 gradient accumulation，要把 `B_global` 恢复到原值，新的 accumulation 应是多少？

学习者回答：

```text
1. DP = 4
2. B_global = DP × B_micro × gradient_accumulation = 64
3. 新 DP = 2
4. 新 B_global = 32，变成原来的一半
5. gradient_accumulation = 16
```

点评：

- `DP = world_size / (TP×PP) = 16/(2×2) = 4`，正确。
- 原 `B_global = 2×8×4 = 64`，正确。
- TP 提高到 4 后，`DP = 16/(4×2) = 2`，正确。
- 新 `B_global = 2×8×2 = 32`，减半，正确。
- 要恢复到 64，`G = 64/(2×2) = 16`，正确。

判定：`passed`。配置与并行组模块完成。

### D4-Q9 — FSDP 后仍然 OOM

一个模型使用 FSDP `FULL_SHARD`，parameters、gradients 和 Adam states 的 persistent layout 已确认按 DP ranks 正确分片，但训练仍在某层 forward 时 OOM：

```text
B_micro = 2
sequence_length = 8192
OOM 位置：attention forward
```

请先回答三点，不要求给唯一根因：

1. 为什么 persistent model states 已经分片，仍可能 OOM？至少列出三类没有随 FSDP persistent sharding 同比例下降的显存对象或 temporary peak。
2. 第一轮你会按什么顺序核对三类证据？请覆盖：
   - tensor shape；
   - per-rank memory timeline/peak；
   - FSDP wrapping、AllGather 或 prefetch 行为。
3. 下面三个实验分别主要验证什么假设？

```text
A. B_micro: 2 -> 1
B. sequence_length: 8192 -> 4096
C. 限制同时 materialize/prefetch 的 FSDP units
```

学习者选择将此题作为实操 guided walkthrough，不要求独立作答。

#### 核心心智模型

FSDP `FULL_SHARD` 主要降低 persistent model-state memory：

```text
parameters + gradients + optimizer states
```

但一次 forward/backward 的 per-rank peak 近似是：

```text
sharded persistent state
+ 当前 FSDP unit 临时 materialize 的 full parameters
+ 可能已预取的下一 unit full parameters
+ saved activations
+ attention/linear/kernel temporary buffers
+ collective buffers 与非 PyTorch CUDA allocations
+ replicated buffers / ignored or incorrectly wrapped modules
+ allocator inactive/fragmented reserved blocks
```

所以“模型状态已经分片”和“当前峰值一定放得下”是两件事。

#### 第一轮排查顺序

##### 1. 先审 tensor shapes

确认实际运行值，而不是只看配置：

```text
actual B_micro
padding/packing 后的 actual sequence length
Q/K/V shape
attention mask shape
attention score 是否显式 materialize 为 [B, H, S, S]
hidden/logits shape 与 dtype
```

`B_micro=2 -> 1` 通常让按 batch 线性增长的 activation 约减半。`S=8192 -> 4096` 会让普通 `[B,H,S,D]` activation 约减半，但若实现显式保存 `[B,H,S,S]` attention matrix，对应对象约降为四分之一。因此 sequence length 实验比单纯减 batch 更能识别 attention 的二次项。

还要确认实际使用的 attention backend。PyTorch SDPA 可在满足输入约束时选择 FlashAttention、Memory-Efficient Attention 或 math backend；不能只因为调用了 `scaled_dot_product_attention` 就假定一定走 fused kernel。

##### 2. 再看 per-rank memory timeline

不要只看 OOM 最后一行。给每个 FSDP unit 或关键 phase 加标记，记录：

```text
memory_allocated
memory_reserved
max_memory_allocated
max_memory_reserved
```

回答：

- 哪个 rank 最先达到峰值？
- 峰值发生在 parameter AllGather 后、attention kernel 内，还是 saved activations 累积后？
- `allocated` 与 `reserved` 的差距多大？
- ranks 是否对称，还是只有某个 rank 异常？

需要进一步定位 allocation stack 时，使用：

```python
torch.cuda.memory._record_memory_history()
# run the reproducer
torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")
```

再用 PyTorch Memory Visualizer 查看 active allocations、inactive blocks 和 OOM 前的 allocation history。注意 PyTorch snapshot 默认看不到 NCCL 等直接通过 CUDA API 分配的显存；需要把 PyTorch allocator 统计与 device-level used memory 对比。

##### 3. 最后审 FSDP wrapping 与 AllGather concurrency

打印每个 FSDP unit：

```text
module name
unsharded parameter bytes
local shard bytes
是否被 ignored
进入/退出 forward 与 backward 的顺序
AllGather 开始/结束
```

重点检查：

- wrap unit 是否过大，导致一次临时 materialize 太多 parameters；
- 是否有 module 没有被预期地 sharding；
- current unit 与 prefetched next unit 是否同时驻留；
- static execution order 是否与 prefetch 假设一致；
- 是否启用限制 AllGather 并发的机制。

PyTorch FSDP 的 `limit_all_gathers=True` 会限制内存中连续 FSDP instances 的 AllGather working set；`forward_prefetch=True` 会提前发出下一个 forward AllGather，官方说明它主要适合 CPU-bound、static-graph workload。不能为了吞吐盲目增加 prefetch，而不测 peak memory。

#### 三个单变量实验如何解释

| 实验 | 主要验证 | 若修复 OOM，优先怀疑 |
|---|---|---|
| `B_micro: 2 -> 1` | batch-linear activation/temporary memory | saved activations、QKV/MLP intermediates、batch-dependent workspace |
| `S: 8192 -> 4096` | sequence-linear或 attention-quadratic memory | attention backend、mask/score materialization、long-context activations |
| 限制同时 materialize/prefetch units | FSDP parameter working-set overlap | wrap 粒度过大、AllGather/prefetch concurrency |

额外的正交实验：

- activation checkpointing：减少 forward 保存的 activations，在 backward 重算；若有效，说明主要矛盾是 saved activations；
- 强制/验证 fused SDPA backend：若显存显著下降，说明此前可能落到 math/非 memory-efficient path；
- 固定 shape 重跑：若固定后稳定、动态 shape 才 OOM，再检查 allocator fragmentation 和 compilation/workspace variants。

#### 不应优先尝试的“假修复”

- 只降低 gradient accumulation、但保持 `B_micro` 不变：不会降低单个 micro-step 的 forward peak；
- 只调用 `empty_cache()`：不会释放仍被 live tensors 占用的显存；
- 只增加 DP/FSDP ranks：persistent state 会更小，但 activation 和某些临时对象不会自动同比缩小；
- 一上来改 allocator 参数：应先证明是 reserved/fragmentation 问题，而不是 tensor shape 或 live working set。

#### 实操结论

本题位于 attention forward，第一优先级是：

```text
actual shapes / backend
-> per-rank allocation timeline
-> FSDP unit materialization overlap
```

而不是因为已经启用 FSDP 就排除 activation 或 attention temporary memory。

官方参考：

- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/main/torch_cuda_memory)
- [Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)

判定：`completed_guided`。

### D4-Q10 — 部分 rank hang

四个 DP ranks 在 backward 中应按相同顺序执行 gradient ReduceScatter。日志显示：

```text
rank 0: enter ReduceScatter bucket 17，之后无输出
rank 1: enter ReduceScatter bucket 17，之后无输出
rank 2: enter ReduceScatter bucket 17，之后无输出
rank 3: last log = "no valid label tokens; skip backward"
```

请回答：

1. 最可能的直接原因是什么？
2. 为什么 rank 0–2 的 collective timeout 不代表它们是根因？
3. 第一轮应对齐比较各 rank 的哪三类信息，才能找到“第一个发生分歧的位置”？

学习者回答：

```text
1. 反向传播架构或切分有问题；rank 0–2 在等待 rank 3。
2. 它们可能只是在等待 rank 3。
3. 比较 tensor shape 和 mask 信息，检查是否有 bug。
```

点评：

1. “rank 0–2 在等待 rank 3”是核心正确判断。更精确的直接原因不是已经证明 sharding 架构错误，而是 collective participation/order 发生分歧：rank 3 因 local control flow 跳过 backward，没有调用其他 group members 正在等待的 bucket 17 ReduceScatter。
2. 正确。Collective timeout 通常在等待者上暴露；报错 rank 可能只是 victim，不一定是第一个偏离预期执行序列的 culprit。
3. Tensor shape 和 mask 属于必要证据，但不足以重建分歧点。第一轮应按 rank 对齐三类信息：

```text
A. collective 序列与 metadata
   sequence number、process group、op、bucket、
   enter/exit timestamp、tensor shape/dtype/numel

B. control flow 与 data state
   batch/sample ids、valid-label-token count、mask summary、
   loss 是否 finite、是否 skip forward/backward/optimizer step

C. rank health 与最后成功位置
   最后完成的 layer/bucket、exception/OOM、
   dataloader progress、stack trace、GPU activity
```

诊断原则：

```text
不要从“哪个 rank 最先 timeout”倒推根因；
要寻找“哪个 rank 最先没有执行与其他 ranks 相同的下一步”。
```

本例的修复策略必须保证所有 group members 保持一致的 collective 序列：

- 若整个 global step 应跳过，先让所有 ranks 对 skip 条件达成一致，然后全部跳过；
- 若只有 rank 3 没有有效 labels，但 global step 仍应训练，则让 rank 3 产生 graph-connected zero loss 并执行 backward，使它按相同顺序参与 gradient collectives，而不是单独 return。

判定：`passed_core; evidence_order_taught`。

#### Q10 在“使用别人的 repo、主要改配置”时何时出现

即使不手写 distributed code，配置和数据也可能触发 rank-local 分歧：

- assistant-only/loss-mask 配置加上 truncation，导致某个 rank 的 batch 没有任何 valid label token；
- packing、dynamic batching、streaming dataset 或 sampler 配置使不同 ranks 得到不等长输入或不同 step 数；
- `drop_last`、resume offset 或数据坏样本处理不一致，使某个 rank 提前耗尽 dataloader；
- 某个 rank OOM、NaN 或数据异常，但 repo 捕获异常后只在本地 `continue/return`；
- gradient accumulation 或 local skip 条件使部分 ranks 执行 backward，其他 ranks 跳过；
- evaluation、save 或 logging 的 rank guard 写错，把包含 collective 的代码只放到部分 ranks；
- data-dependent routing/conditional module 改变了某些 ranks 的 backward graph 或 collective 序列。

对只改配置的使用者，最实用的判断不是先假设 NCCL 或网络损坏，而是：

```text
找每个 rank 最后一条成功日志
-> 对齐 batch/valid-token/skip 决策
-> 对齐最后进入和退出的 collective
-> 查找最先偏离共同执行序列的 rank
```

### D4-Q11 — 改变并行度后 checkpoint 无法加载

旧任务保存 checkpoint 时：

```text
world_size = 8
DP = 2
TP = 2
PP = 2
format = rank-local sharded checkpoint
```

新任务仍使用 8 张 GPU，但配置改为：

```text
world_size = 8
DP = 1
TP = 4
PP = 2
```

加载时报 parameter shard shape 和 optimizer-state mapping mismatch。

请回答：

1. 为什么 `world_size` 同样是 8，checkpoint 仍不能直接逐 rank 加载？
2. 第一轮需要核对哪些 checkpoint metadata？至少列出四项。
3. “loader 支持 reshard”至少意味着它需要知道哪两个布局，才能把旧 shards 转为新 shards？
4. 如果当前格式只保存了与旧 rank topology 绑定的 local shards，loader 又不支持这种 topology 变化，应该直接忽略 mismatch，还是先恢复/转换为 topology-independent 的 full/global state 再重新分片？

学习者回答：

```text
1. 分片变了；TP 增大后坐标体系和 rank ownership 都变化，加载前需要处理。
2. 不知道。
3. 需要知道 reshard 前后的布局。
4. 先转成 topology-independent 的 full/global state，再重新分片。
```

点评：

1. 正确。`world_size` 只是 rank 总数；`DP/TP/PP` factorization、rank mapping、parameter shard axis 和 stage ownership 都可能不同。旧 rank 0 的 bytes 不再必然对应新 rank 0 所需的 tensor slice。
2. Metadata 需要按以下层次核对：

```text
A. 格式与兼容能力
   checkpoint format/schema version
   full、sharded 还是 rank-local
   model 与 optimizer state 是否都支持目标 topology 的 reshard
   framework/checkpoint-library version

B. 模型逻辑身份
   canonical parameter name/FQN
   global shape、dtype
   tied/shared parameter identity
   layer count、hidden size、vocab size、attention/expert config

C. 旧并行布局与 shard spec
   saved DP/TP/PP/world size
   rank mapping / device mesh
   每个 tensor 的 shard axis、global offset、local shape、placement
   PP stage ownership
   FSDP wrap/flat-parameter mapping（若格式依赖它）

D. Optimizer mapping
   optimizer type 与 state schema
   parameter identity 到 exp_avg/exp_avg_sq/master-weight 的映射
   param-group ordering/hyperparameters
   optimizer state 的 shard axis、offset、dtype

E. 完整恢复状态
   global step、scheduler、grad scaler
   RNG/data-sampler state
   checkpoint manifest/completeness
```

其中 A–D 直接决定能否正确重组 model/optimizer tensors；E 决定 resume 是否语义完整和可复现。
3. 正确。Reshard planner 至少需要 source global tensor/shard layout 与 target model/device-mesh layout，并通过稳定的 logical parameter identity 把两边对应起来。
4. 正确。不能忽略 mismatch；否则可能加载错 slice 或丢失 optimizer state。安全路径是使用格式原生支持的 load-time reshard，或先转成 canonical full/global representation，再按目标 topology 分片。

实践边界：

- PyTorch Distributed Checkpoint 使用 canonical FQN，并提供 load-time resharding；repo 必须实际通过对应 state-dict/DCP API 保存和加载，普通 rank-local `torch.save` 文件不会自动获得该能力。
- Megatron Core 的 `torch_dist` distributed checkpoint 支持跨并行配置 reshard，但 optimizer 的具体格式仍有能力差异：文档中的 `dp_reshardable` 只允许 DP 维变化，`fully_reshardable` 才支持任意 model-parallelism 变化。

官方参考：

- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)
- [PyTorch DCP Recipe](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [Megatron Core Distributed Checkpointing](https://docs.nvidia.com/megatron-core/developer-guide/0.16.1/api-guide/core/dist_checkpointing.html)

判定：`passed_after_teaching`。

## Day 04 Quiz 最终结果

| 模块 | 结果 |
|---|---|
| A. Sharded matmul | `completed` |
| B. Collective 与布局 | `completed` |
| C. 训练状态 | `completed` |
| D. 配置与并行组 | `completed` |
| E. 故障诊断 | `completed` |

稳定掌握：

- 从 shard 维度区分 output-coordinate shards 与 same-coordinate partial sums；
- 从目标布局选择 AllGather、AllReduce、ReduceScatter、AllToAll；
- 区分 DP、FSDP、TP、PP 的 state ownership 与 activation layout；
- 使用 `world_size=DP×TP×PP` 与 `B_global=B_micro×G×DP`；
- 识别 OOM 的 persistent state、activation 与 temporary working-set 来源；
- 对 hang 寻找第一个 rank execution/collective divergence；
- 对 checkpoint mismatch 核对 source/target layout 与 reshard capability。

后续实践中需要继续强化：

- 从真实 repo 配置和日志还原 rank groups、FSDP units 与 checkpoint format；
- 使用 memory snapshot/collective logs 将概念映射到真实故障证据；
- 在改变 TP/PP/DP 前确认 model state 和 optimizer state 是否都支持目标 reshard。

## 附录：Q10 empty-label 早期引导记录

以下记录来自较早的 Q10 理解阶段，已被上方 `D4-Q10` 的最终回答与 `passed_core; evidence_order_taught` 判定覆盖；保留作为错误画像和复习材料，不阻塞 quiz 的 `completed` 状态。

学习者回答：

```text
1. 不理解 no valid label tokens；若是 OOM，可能是 activation 累计。
2. timeout 和后面的日志关系不大。
3. 比较 tensor shape，确认是否有 bug。
```

点评：

#### `no valid label tokens` 的含义

SFT 中通常只有目标回答 token 参与 loss；prompt、padding 等位置会被设为 ignore label（常见值为 `-100`）。如果某个 local batch 在 masking 后所有 positions 都被忽略：

```text
valid_label_count = 0
```

代码可能进入：

```text
if valid_label_count == 0:
    skip backward
```

这不是 OOM，也不是 activation 累计。它是 rank-local 数据或控制流条件导致某个 rank 没有进入 backward。

#### 为什么其他 ranks 会 timeout

一个 collective 的基本契约是：同一 process group 的所有 ranks 必须按兼容的顺序进入对应 collective。

本题实际执行流：

```text
rank 0 -> enter ReduceScatter bucket 17 -> 等 rank 3
rank 1 -> enter ReduceScatter bucket 17 -> 等 rank 3
rank 2 -> enter ReduceScatter bucket 17 -> 等 rank 3
rank 3 -> skip backward -> 永远不调用 bucket 17
```

所以 timeout 与 rank 3 的日志有直接因果关系。rank 0–2 只是最先报告“等不到其他参与者”的受害者；最早发生分歧的位置在 rank 3 的 `skip backward` 分支。

#### 第一轮需要对齐的三类信息

1. 数据与控制流：

```text
global step / micro-step id
batch/sample ids
valid label count
skip flags
是否调用 loss.backward()
异常/OOM/提前 return
```

2. Collective 序列：

```text
process group
collective sequence number
op type
bucket id
enter/exit timestamp
```

3. Tensor metadata：

```text
shape
numel
dtype
device
bucket/parameter identity
```

排查方法不是从最终 timeout rank 倒猜，而是把所有 rank 的事件按 step 和 sequence 对齐，找到第一个：

```text
某 rank 缺失 op
或 op 顺序不同
或 tensor metadata 不兼容
```

#### 正确处理 empty-label local batch 的原则

不能让某个 DP rank 单独跳过 backward。即使本地有效 token 数为 0，它也必须通过 graph-connected zero loss 等机制参与相同的 backward/collective 序列；同时全局 loss normalization 应基于跨 DP ranks 聚合的有效 token 数，避免改变训练目标。

历史判定：`tensor-shape evidence recognized; collective contract and control-flow divergence taught`。最终状态以上方 D4-Q10 判定为准。

### 可选练习 — Hang 迁移检查（不阻塞完成状态）

另一个 step 中：

```text
rank 0/1/3：调用 backward，进入 gradient AllReduce
rank 2：empty local batch，直接 return，不调用 backward
```

只回答两点：

1. 哪个 rank 是第一个发生分歧的位置？
2. rank 2 应该直接 return，还是构造 graph-connected zero loss 并继续 backward？为什么？
