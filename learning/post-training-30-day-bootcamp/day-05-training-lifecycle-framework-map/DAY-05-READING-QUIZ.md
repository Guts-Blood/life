# Day 05 Reading Quiz — Training Lifecycle 与框架职责边界

开始日期：`2026-07-29`

状态：`in_progress`

方式：Socratic quiz；一次只讨论一道题，先回答、再纠错、再进入下一题。

## Quiz 目标

| 模块 | 希望真正掌握的能力 | 状态 |
|---|---|---|
| A. 单步 lifecycle | 从 batch 到参数更新，按依赖关系讲清 forward、loss、backward、同步、clip 与 optimizer step | `completed` |
| B. 数据与 loss 边界 | 说明 template、tokenize、collate、labels/mask 分别产出什么对象 | `completed` |
| C. 框架职责 | 区分训练 repo、Trainer/runtime、PyTorch、NCCL 与 CUDA 各自决定和执行什么 | `in_progress` |
| D. 源码追踪 | 从一条 `swift sft` 命令定位 dataset、model forward、loss、optimizer 与 checkpoint 边界 | `not_started` |
| E. Checkpoint/resume | 列出连续恢复所需状态，并预测遗漏 optimizer、scheduler、RNG 或 dataloader progress 的后果 | `not_started` |
| F. Eval 边界 | 区分 train/eval mode、无梯度评估、指标聚合与 checkpoint 选择 | `not_started` |

题目会根据回答动态增加或跳过追问。Day 04 已掌握的 collective 名称和并行组计算不单独重复，只有在 lifecycle 边界中需要时才复用。

## 当前进度

- D5-Q1：`passed_after_teaching`
- D5-Q2：`passed_after_teaching`
- D5-Q3：`awaiting_answer`
- Day 05 Reading Quiz：`in_progress`

## 当前题目

### D5-Q1 — 先搭出一个 optimizer update 的骨架

假设：

```text
gradient_accumulation_steps = 2
使用普通 data parallel
启用 gradient clipping
每完成一次 optimizer update，scheduler 前进一步
```

下面这些动作被打乱了：

```text
A. optimizer.step()
B. forward
C. gradient synchronization
D. scheduler.step()
E. zero_grad()
F. 从 dataloader 取一个 micro-batch
G. backward
H. 对有效 label tokens 计算 masked loss
I. gradient clipping
```

请回答三点：

1. 从一个 update 开始到下一个 update 开始，写出正确顺序。`F -> B -> H -> G` 需要执行几次？其余动作各在什么时机执行？
2. 在 `backward`、`gradient synchronization`、`gradient clipping`、`optimizer.step()` 之后，分别是哪类训练状态发生了变化？
3. 为什么不能在第一个 micro-batch 的 `backward` 后就立刻 `zero_grad()`？

暂时忽略 AMP、overflow、ZeRO/FSDP 和 checkpoint；先只建立最小且正确的单步状态机。

## 回答与点评

### D5-Q1 第一轮回答

学习者回答：

```text
1. 除了 optimizer.step，其他都要执行两次，因为 accumulation=2。
2. backward 在计算 gradient；gradient sync 同步切片状态（TP）；
   clipping 对 gradient 去噪和平滑；zero_grad 主要处理 0 loss。
3. 不知道。
```

点评：

第一问的关键纠正是：`gradient_accumulation_steps=2` 只要求
micro-batch 计算链执行两次，并不要求所有非 optimizer 动作执行两次。
一个常见的逻辑顺序是：

```text
E. zero_grad()

micro-batch 1:
F. 取 batch -> B. forward -> H. masked loss -> G. backward
                                        （累积本地 gradients，不做 DP sync）

micro-batch 2:
F. 取 batch -> B. forward -> H. masked loss -> G. backward
                                        （累积 gradients，并完成 DP sync）

I. gradient clipping
A. optimizer.step()
D. scheduler.step()

进入下一 update，再执行 E. zero_grad()
```

因此：

- `F -> B -> H -> G` 执行两次；
- 普通 DDP 通常用 `no_sync()` 抑制非最后一个 micro-step 的同步，最后一个
  `backward` 中由 autograd hooks 触发 gradient synchronization；
- gradient clipping、`optimizer.step()`、`scheduler.step()` 和
  `zero_grad()` 每个 optimizer update 各执行一次。

第二问需要按“哪块状态被修改”来区分：

| 动作 | 发生变化的状态 |
|---|---|
| `backward` | 根据当前 loss 计算梯度，并加到参数的 `.grad` buffer；参数值尚未改变 |
| DP gradient synchronization | 合并各 data-parallel replicas 对同一参数算出的梯度；不是同步 TP 参数切片 |
| gradient clipping | 当整体 gradient norm 超过阈值时，按比例缩小 gradients；它不是去噪或平滑 |
| `optimizer.step()` | 使用 gradients 更新 model parameters，同时更新 momentum、Adam moments 等 optimizer state |
| `scheduler.step()` | 更新当前或下一 update 使用的 learning rate 等 scheduler state |
| `zero_grad()` | 清空或设为 `None` 的是 gradient buffers；它不处理 zero loss |

第三问：第一次 `backward` 后，`.grad` 中保存的是第一个 micro-batch
对本次 update 的梯度贡献。若立即 `zero_grad()`，该贡献会被删除；第二个
micro-batch 的梯度将独自决定更新，gradient accumulation 失效，有效 batch
也退化为最后一个 micro-batch。

简化地写，目标是保留：

```text
g_update = contribution(g_micro_1) + contribution(g_micro_2)
```

至于是除以 `2`，还是按跨 micro-batch 的有效 label-token 总数归一化，取决于
loss normalization 契约；但无论哪种契约，都不能在两个贡献合并前清空
gradient buffers。

判定：`backward computes gradients recognized; accumulation boundary,
DP sync, clipping and zero_grad semantics taught`。

### D5-Q1 定向追问

假设单卡、一个标量参数 `w`，两个 micro-batch 分别产生：

```text
micro-batch 1: dL1/dw = 6
micro-batch 2: dL2/dw = 2
```

训练代码在每个 micro-batch 上都先使用：

```text
loss = loss / gradient_accumulation_steps
```

并且 `gradient_accumulation_steps=2`。

只回答两点：

1. 两次 `backward()` 后、`optimizer.step()` 前，`w.grad` 应该是多少？
2. 如果在第一次 `backward()` 后错误地执行 `zero_grad()`，最终
   `w.grad` 又是多少？

#### D5-Q1 定向追问回答与点评

学习者回答：

```text
1. w.grad = 4；每次 backward 前 loss 都除以 2。
2. w.grad = 1；前面的被清空了。
```

两项均正确：

```text
正常累积：
w.grad = 6/2 + 2/2 = 4

第一次 backward 后清空：
w.grad = 0 + 2/2 = 1
```

关于“gradient 是 data 而不是 parameter”：

- 从存储形式看，parameter、gradient 和 Adam state 通常都是 tensor data；
- 从训练语义看，它们是三类不同状态，不能因为底层都是 tensor 就混为一类。

```text
parameter:
  模型当前的可训练数值，例如 w
  forward 直接读取它
  optimizer.step() 修改它

gradient:
  当前 update 对 parameter 的导数，例如 w.grad
  backward 计算/累积它
  optimizer.step() 消费它
  zero_grad() 清除它

optimizer state:
  optimizer 为如何更新 parameter 保存的历史，例如 Adam 的
  step、exp_avg（一阶矩）、exp_avg_sq（二阶矩）
  optimizer.step() 读取并修改它
```

所以，更严格的说法是：gradient 不是 parameter value，但它是与 parameter
关联的、形状通常相同的临时训练状态；Adam moments 也是 data/tensors，
但属于 optimizer state，而不属于 model parameters。

判定：`passed_after_teaching`。模块 A 完成。

### D5-Q2 — 从 raw sample 到 model inputs

有两条长度不同的对话样本：

```text
sample 1:
  user: "1+1 等于几？"
  assistant: "2"

sample 2:
  user: "用一句话解释梯度累积。"
  assistant: "多个 micro-batch 的梯度合并后再更新参数。"
```

训练目标规定：

```text
只让 assistant answer tokens 参与 loss；
user prompt 和 padding tokens 不参与 loss。
```

请沿着下面的边界回答：

```text
raw sample
-> template
-> tokenize
-> collator
-> model forward
-> masked causal-LM loss
```

1. `template`、`tokenize`、`collator` 各自主要完成什么工作？在哪一步把两条
   不同长度的样本变成形状统一的 batched tensors？
2. 进入 model 前，`input_ids`、`attention_mask`、`labels` 各自表达什么？
   user prompt 和 padding 对应的 `labels` 通常应设为什么值？
3. 假设 batch shape 是 `[B,S]`、词表大小是 `V`，model 输出的
   `logits` shape 是什么？计算 loss 时，哪些位置会被忽略？

#### D5-Q2 第一轮回答

学习者回答：

```text
1. template 把自然语言 words 切分，决定训练位置、attention_mask 和 labels；
   tokenize 把 input ids 映射到 d_model vector；
   collator 组装 batch 和 padding。
2. input_ids 是 vocab 中的位置；attention_mask mask next token；
   labels 表示是否计算梯度，不训练的位置设为 -100。
3. logits shape = [B,S,V]；label=-100 的位置不参与 loss。
```

点评：

正确部分：

- collator 组装 batch、padding，使不同长度样本成为统一的 `[B,S]` tensors；
- `input_ids` 是 tokenizer vocabulary 中的整数 token IDs；
- model logits shape 是 `[B,S,V]`；
- `labels=-100` 的位置被 loss 函数忽略。

需要纠正三个边界。

第一，template 主要把结构化样本序列化为模型约定的文本/token 结构：

```text
messages
-> system/user/assistant role markers
-> BOS/EOS、分隔符和固定提示格式
-> 哪一段属于 prompt，哪一段属于 assistant answer
```

它可能同时产出 assistant loss span/mask，但“把 words/subwords 切开并映射为
整数 ID”是 tokenizer 的职责，不是 template 的核心职责。

第二，tokenizer 的输出仍是离散整数：

```text
text/token pieces -> input_ids [S]
```

把 IDs 映射到连续 `d_model` 向量的是 model 内的 embedding layer：

```text
input_ids [B,S] -> hidden states [B,S,D]
```

第三，需要区分两类 mask：

```text
attention_mask:
  常见的 [B,S] 输入，1 表示真实 token，0 表示 padding；
  控制哪些 token 可作为 attention 的有效输入。

causal mask:
  防止位置 t 看见未来位置 >t；
  causal-LM 通常在 model 内部构造，并与 attention_mask 合并。

labels / loss mask:
  labels 保存目标 token ID，-100 表示该位置不计入 cross-entropy；
  它控制 loss 位置，不是一个“是否计算参数梯度”的布尔 tensor。
```

关键例子：prompt token 通常是

```text
attention_mask = 1
labels         = -100
```

这意味着 assistant token 在 forward 中仍然可以关注 prompt、利用问题上下文，
但 prompt 位置本身不贡献 supervised loss。若把 prompt 的
`attention_mask` 也设为 0，模型连问题内容都看不到。

不同库可能把 labels/loss mask 的具体构造放在 template encoder、dataset
preprocessor 或 collator 中；源码追踪时要查真实 owner。但概念职责不变：
attention visibility 和 loss inclusion 是两个独立维度。

判定：`collation, logits shape and ignore index passed; template/tokenizer/
embedding and attention/loss masks taught`。

### D5-Q2 定向追问

某条已 padding 的样本为：

```text
位置:              0       1       2       3       4       5
内容:            <user>   问题   <assistant>  回答1    回答2    <pad>
input_ids:         10      21       11       31       32       0
attention_mask:     1       1        1        1        1       0
labels:           -100    -100     -100       31       32     -100
```

只回答三点：

1. 哪个组件把 `input_ids [B,S]` 映射成 `[B,S,D]`？
2. 为什么 prompt 位置可以是 `attention_mask=1`，同时又是
   `labels=-100`？assistant answer 是否仍能利用 prompt？
3. 位置 5 的 padding 为什么同时需要 `attention_mask=0` 和
   `labels=-100`？两者分别阻止什么？

#### D5-Q2 定向追问回答与点评

学习者回答：

```text
1. embedding layer。
2. attention_mask 主要控制是不是 padding，labels 判断是否参与传播计算。
3. 同 2。
```

第一点正确。第二、三点方向正确，但“传播计算”应更精确地说成“该位置是否
参与 loss 计算”：

```text
prompt:
  attention_mask=1 -> 作为有效上下文参与 forward/attention
  labels=-100      -> prompt 位置不直接贡献 supervised loss

padding:
  attention_mask=0 -> 不让 padding 作为有效上下文影响其他 token 的 attention
  labels=-100      -> 不对 padding 位置计算 cross-entropy loss
```

因此 assistant answer 仍然能利用 prompt。`labels=-100` 不会从 forward
中删除 prompt；它只删除该位置的直接 loss term。反过来，padding 即使已经
被 attention mask 屏蔽，仍需设置 `labels=-100`，因为 attention visibility
和 loss inclusion 是独立契约。

判定：`passed_after_teaching`。模块 B 完成。

### D5-Q3 — “谁决定，谁执行”

一条 `swift sft` 训练使用 DeepSpeed ZeRO-3。某一步观察到：

```text
1. CLI 参数被解析，Qwen 模型与对话数据集被加载并适配。
2. Trainer 开始一个 train step，并调用 model(**batch)。
3. 某层计算前，分片参数需要临时 AllGather。
4. GPU 执行该层的 BF16 GEMM。
5. backward 后，梯度被 ReduceScatter；随后 optimizer state 的本地 shard 被更新。
```

请在以下层次中分配主要职责；一个事件可以涉及多层，但要区分“决定/发起”和
“底层执行”：

```text
ms-swift
Transformers/TRL Trainer
DeepSpeed
PyTorch autograd/optimizer
NCCL
CUDA/cuBLAS/Triton
```

回答四点：

1. 事件 1 主要是谁负责？
2. 谁组织训练循环并调用 model？谁追踪 backward graph、计算 gradients？
3. 谁决定参数采用 ZeRO-3 分片，并在正确时机发起 AllGather/ReduceScatter？
   谁实际执行跨 GPU collective？
4. 谁最终执行单卡 GEMM kernel？“CUDA 自动把参数做 ZeRO-3 分片”这句话为什么错误？
