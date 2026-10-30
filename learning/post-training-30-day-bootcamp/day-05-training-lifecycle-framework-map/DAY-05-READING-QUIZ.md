# Day 05 Reading Quiz — Training Lifecycle 与框架职责边界

开始日期：`2026-07-29`

状态：`completed`

方式：Socratic quiz；一次只讨论一道题，先回答、再纠错、再进入下一题。

## Quiz 目标

| 模块 | 希望真正掌握的能力 | 状态 |
|---|---|---|
| A. 单步 lifecycle | 从 batch 到参数更新，按依赖关系讲清 forward、loss、backward、同步、clip 与 optimizer step | `completed` |
| B. 数据与 loss 边界 | 说明 template、tokenize、collate、labels/mask 分别产出什么对象 | `completed` |
| C. 框架职责 | 区分训练 repo、Trainer/runtime、PyTorch、NCCL 与 CUDA 各自决定和执行什么 | `completed` |
| D. 源码追踪 | 从一条 `swift sft` 命令定位 dataset、model forward、loss、optimizer 与 checkpoint 边界 | `completed` |
| E. Checkpoint/resume | 列出连续恢复所需状态，并预测遗漏 optimizer、scheduler、RNG 或 dataloader progress 的后果 | `completed` |
| F. Eval 边界 | 区分 train/eval mode、无梯度评估、指标聚合与 checkpoint 选择 | `completed` |

题目会根据回答动态增加或跳过追问。Day 04 已掌握的 collective 名称和并行组计算不单独重复，只有在 lifecycle 边界中需要时才复用。

## 当前进度

- D5-Q1：`passed_after_teaching`
- D5-Q2：`passed_after_teaching`
- D5-Q3：`passed_after_code_guided_teaching`
- D5-Q4：`passed_after_evidence_order_teaching`
- D5-Q5：`passed_after_checkpoint_term_correction`
- D5-Q6：`passed_after_teaching`
- D5-Q7：`completed_guided`
- Day 05 Reading Quiz：`completed`

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

#### D5-Q3 学习者反馈

```text
这些我都不知道，希望结合代码讲解。
```

#### D5-Q3 代码引导

先看 `swift sft` 的入口：

```python
# swift/cli/sft.py
from swift.pipelines import sft_main
sft_main()

# swift/pipelines/train/sft.py
def sft_main(args=None):
    return SwiftSft(args).main()
```

`SwiftPipeline.__init__` 使用 `SftArguments` 解析 CLI；`SwiftSft.__init__`
加载 model/processor 和 template；`SwiftSft.run` 加载/编码 dataset，并根据
任务选择 trainer：

```python
trainer_cls = TrainerFactory.get_trainer_cls(args)
trainer = trainer_cls(
    model=self.model,
    args=self.args.training_args,
    template=self.template,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
return self.train(trainer)
```

对 causal LM，factory 映射到：

```python
'causal_lm': 'swift.trainers.Seq2SeqTrainer'
```

而该类的定义是：

```python
class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    ...
```

这说明 ms-swift 主要提供 CLI/recipe、模型和数据适配、template/collator
以及对 upstream trainer 的扩展；通用训练循环来自 Transformers
`HfSeq2SeqTrainer`。

真实 forward 边界在：

```python
# swift/template/base.py
outputs = model(**inputs)
```

这里的 `model` 是 PyTorch `nn.Module`。例如 tiny 教学模型中：

```python
token_embeddings = self.token_embedding(input_ids)  # [B,T] -> [B,T,D]
attention_scores = q @ k.transpose(-2, -1)           # [B,H,T,T]
logits = self.lm_head(final_hidden_states)            # [B,T,D] -> [B,T,V]
```

PyTorch 负责 tensor operators、autograd graph 和 `.grad`。CPU 运行时这些
operators 使用 CPU kernels；CUDA tensor 运行时，PyTorch 会把 matmul/linear
dispatch 到 GPU kernel，常见实现来自 cuBLAS、CUDA fused kernels 或 Triton。

Transformers Trainer 的主要 loop 负责 accumulation boundary、clip、step
和 scheduler。在当前本地安装版本中，其逻辑等价于：

```python
tr_loss_step = self.training_step(model, inputs, ...)
self.accelerator.backward(loss)

if do_sync_step:
    self.accelerator.clip_grad_norm_(...)
    self.optimizer.step()
    self.lr_scheduler.step()
    model.zero_grad()
```

如果没有 DeepSpeed，Accelerate 最终调用普通 PyTorch backward。若启用
DeepSpeed，它转发给 DeepSpeed engine：

```python
if distributed_type == DEEPSPEED:
    self.deepspeed_engine_wrapped.backward(loss, ...)
```

ZeRO-3 的选择来自 ms-swift preset：

```json
{
  "zero_optimization": {
    "stage": 3
  }
}
```

ms-swift 把 `--deepspeed zero3` 解析成这份 config；Transformers/Accelerate
负责集成和初始化：

```python
engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    config_params=deepspeed_config,
    ...
)
```

随后职责边界是：

```text
DeepSpeed:
  解释 stage=3，建立 shard ownership；
  在参数被某层使用前发起 materialization/AllGather；
  在 backward 中安排 gradient reduction/partition；
  更新本 rank 持有的 optimizer-state shard。

PyTorch distributed:
  提供 collective API 和 process group。

NCCL:
  作为 CUDA 多 GPU process-group backend，执行实际的数据传输和 reduction；
  它不知道训练 recipe，也不决定哪一个 parameter 应被分片。

CUDA/cuBLAS/Triton:
  执行单 GPU 的 matmul、attention、normalization 等 kernels；
  它们不拥有 ZeRO-3 的全局 shard policy。
```

完整调用链：

```text
swift sft
-> ms-swift CLI / SftArguments / model-data-template adapters
-> ms-swift Seq2SeqTrainer
-> Transformers Trainer loop
-> Accelerate backend dispatch
-> DeepSpeed ZeRO engine（若启用）
-> PyTorch autograd / torch.distributed
-> NCCL collectives + CUDA/cuBLAS/Triton local kernels
```

验证过的 CPU 教学脚本：

```text
day-05-training-lifecycle-framework-map/annotated_tiny_causal_lm.py
```

它显式实现了同一个单卡状态机：

```python
outputs = model(...)
loss_for_backward.backward()
clip_grad_norm_(...)
optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

运行结果显示两个 optimizer updates 均完成；裁剪后 gradient norm 约等于
`1.0`，checkpoint 重新加载后 model parameters 完全匹配。当前环境没有安装
DeepSpeed，因此 ZeRO-3 部分本次只做静态源码和 config trace，没有实际执行。

### D5-Q3 定向追问

不要求背文件名，只根据上面的调用链回答：

1. `--deepspeed zero3` 最终由谁解释成参数/梯度/optimizer state 的分片行为？
2. DeepSpeed 发起一次 AllGather 后，谁负责真正执行跨 GPU 通信？
3. `model(**inputs)` 中的 `nn.Linear` 在 GPU 上最终由哪一层执行 GEMM？
4. 用一句话说明 ms-swift 在这条链路中的主要价值。

#### D5-Q3 定向追问回答与点评

学习者回答：

```text
DeepSpeed 决定分片和通信策略，NCCL 实际执行通信策略，CUDA 纯做计算。
```

主干正确：

- DeepSpeed 根据 ZeRO stage 决定 state ownership、parameter materialization
  和 collective schedule；
- NCCL 执行 DeepSpeed/PyTorch distributed 请求的具体 collective，不理解
  model、parameter 或分片原因；
- CUDA 侧执行本地 GPU kernels。

两点精度修正：

1. NCCL 执行的是 collective operation，而不是理解并执行高层“训练策略”。
2. CUDA 不只包含数学计算，还提供 GPU runtime、device memory、streams/events
   等基础设施；NCCL 的 GPU 通信也建立在 CUDA 环境上。单卡 GEMM 常由
   cuBLAS、Triton 或其他 CUDA kernels 完成。

判定：`passed_after_code_guided_teaching`。模块 C 完成。

### D5-Q4 — 遇到现象，应该在哪个边界打断点

沿用已经追过的真实路径：

```text
dataset row
-> template.encode
-> template.data_collator
-> Seq2SeqTrainer.compute_loss
-> template.compute_sft_loss
-> model(**inputs)
-> Trainer/Accelerate backward
-> DeepSpeed engine / optimizer
```

现在分别出现四个问题。请为每个问题选“第一处最有信息量的检查点”，并说明
在那里应该打印或检查什么对象：

1. 训练 loss 几乎为 0，怀疑所有 assistant answer 都被 mask 掉了。
2. GPU 报 shape mismatch，怀疑 padding 后的 batch 不一致。
3. forward 有 loss，但执行两次 micro-batch 后参数一直没有变化。
4. 配置写了 `--deepspeed zero3`，但怀疑运行时根本没有启用 ZeRO-3。

不要求给出精确文件行号；回答类似下面的格式即可：

```text
问题 1：
第一检查点 = ...
检查对象 = ...
因为 = ...
```

#### D5-Q4 第一轮回答与点评

学习者回答：

```text
1. template.encode。
2. collator；组装 batch 时 padding 有问题。
3. 可能是 loss、Trainer/Accelerate backward，也可能 template.encode 的
   labels/mask 有问题。
4. DeepSpeed 的问题。
```

第 1、2 题定位正确：

```text
问题 1:
  第一检查点 = template.encode 的输出
  检查对象 = input_ids、labels、(labels != -100).sum()
  原因 = 在进入 collator/model 前确认每条样本是否存在有效监督 token

问题 2:
  第一检查点 = template.data_collator 的输出
  检查对象 = 每个 key 的 shape、dtype，以及 input_ids/labels/attention_mask
             的 batch 和 sequence 维是否相同
  原因 = 这是 variable-length lists 变成 dense [B,S] tensors 的边界
```

第 3 题列出的原因都可能成立，但题面已经给出“forward 有 loss”。第一轮排查
应从离症状最近的 update boundary 倒查，而不是重新从 template 开始：

```text
第二个 micro-batch backward 后:
  是否到达 accumulation boundary
  parameter.grad 是否为 None
  global grad norm 是否为 0/NaN/finite
  sync_gradients 是否为 True

optimizer.step 前后:
  optimizer step 是否实际调用/是否因 AMP overflow 被 skip
  learning rate 是否为 0
  optimizer state["step"] 是否增加
  一项固定 parameter 的 max_abs(after-before) 是否 > 0
```

如果 gradients 已经 finite 且非零，而 parameter 不变，就继续查
optimizer/LR/step skip；只有 gradients 为 None/0 时才向 backward/loss graph
追；只有 loss/valid-label count 也异常时才继续追到 template。

第 4 题不能直接写成“DeepSpeed 的问题”。先验证配置传递和 runtime identity：

```text
配置侧:
  resolved training_args.deepspeed["zero_optimization"]["stage"] == 3

runtime 侧:
  trainer.is_deepspeed_enabled == True
  accelerator.distributed_type == DEEPSPEED
  model/optimizer 是否已被 DeepSpeed engine/wrapper 替换
  engine 的 zero stage 是否为 3
```

如果 resolved config 是 stage 3，但 runtime 仍是普通 PyTorch/DDP，应先查
launcher、Transformers/Accelerate 集成或环境依赖，而不是把它归因为 NCCL
或 ZeRO collective 故障。

判定：`template/collator boundaries passed; evidence order for update and
DeepSpeed activation taught`。

### D5-Q4 定向追问

只判断下面两组证据：

```text
Case A:
  loss = 2.3
  grad_norm_before_step = 0.8
  optimizer_step_was_skipped = False
  learning_rate = 0.0
  max_abs(parameter_after - parameter_before) = 0.0

Case B:
  resolved deepspeed config: zero_optimization.stage = 3
  trainer.is_deepspeed_enabled = False
  accelerator.distributed_type = NO
  model type = 原始 PyTorch model，不是 DeepSpeed engine
```

回答：

1. Case A 中，最直接的根因是什么？还需不需要先回头检查 labels？
2. Case B 说明问题发生在“ZeRO-3 collective 执行时”，还是在“DeepSpeed
   runtime 根本没有启用”这一更早边界？

#### D5-Q4 定向追问回答与点评

学习者回答：

```text
1. 不需要检查 labels；learning rate 明显有问题。
2. DeepSpeed 没有启用。DeepSpeed 主要是分布式训练优化框架。
```

两项正确：

- Case A 已证明 loss、gradient 和 optimizer step control flow 正常，
  `learning_rate=0` 足以直接解释 parameter delta 为 0；
- Case B 中 DeepSpeed runtime 没有初始化，ZeRO collective 尚未发生，
  因而不能归因于 NCCL 或 collective failure。

DeepSpeed 的定位可以写成：

```text
DeepSpeed =
  建立在 PyTorch 之上的 distributed training optimization/runtime framework

主要能力包括：
  training engine
  ZeRO parameter/gradient/optimizer-state sharding
  CPU/NVMe offload
  mixed precision 与 loss scaling 集成
  communication scheduling/overlap
  distributed checkpoint integration
  以及部分 pipeline/tensor-parallel/inference 能力
```

它决定高层 state layout 和执行计划，再通过 PyTorch distributed/NCCL
完成通信，并通过 CUDA kernels 完成本地计算。Accelerate 更像统一后端接入层；
NCCL 是 collective 通信执行库；三者不在同一抽象层。

判定：`passed_after_evidence_order_teaching`。模块 D 完成。

### D5-Q5 — “能加载”不等于“连续恢复”

假设训练使用：

```text
AdamW
cosine learning-rate scheduler
dropout
shuffle=True 的 dataloader
gradient_accumulation_steps=2
```

checkpoint 在完成 `optimizer_step=1000` 并 `zero_grad()` 后保存。现在从该目录
恢复，分别考虑：

```text
A. 只加载 model weights
B. 加载 model + optimizer state，但不加载 scheduler/global_step
C. 加载 model + optimizer + scheduler/global_step，但不加载 RNG 和
   sampler/dataloader progress
D. 加载全部相关状态
```

请回答：

1. A 与真正的 continuous resume 相比，AdamW 的什么状态丢失？下一次更新
   为什么即使使用相同 batch 也可能不同？
2. B 会对 learning rate trajectory 造成什么影响？
3. C 能否从相同的“下一批样本”和相同 dropout mask 继续？训练是否仍可运行，
   是否还能称为 exact/continuous resume？
4. 请列出 D 至少应包含的状态类别。`.grad` 是否必须保存？注意 checkpoint
   是在完成 update 并 `zero_grad()` 后保存。

#### D5-Q5 第一轮回答与问题

学习者回答：

```text
1. 会丢失 m、v；优化是在整个 distribution 上优化。
2. 不知道，可能会多训练几步。
3. 不知道 RNG 和 dropout mask 的逻辑。
4. 请求讲解。

追加问题：Accelerate 主要做什么？是否只写 training 逻辑而不管分布式？
```

第一问的状态识别正确，但需要补充：

```text
AdamW optimizer state（典型）:
  step
  exp_avg     = 梯度的一阶指数滑动平均 m
  exp_avg_sq  = 梯度平方的二阶指数滑动平均 v
  可能还有 mixed-precision master weights 等实现相关状态
```

Adam 并不保存完整的 data/gradient distribution。它按 parameter element
维护历史梯度的一阶、二阶统计量。普通 data parallel 中，各 rank 先把当前
global batch 的梯度同步，再使用相同的 m/v 更新 replicated parameters；
ZeRO 中这些 optimizer states 可以被分片，但数学角色不变。

即使恢复相同 model weights，并让下一 batch 完全相同：

```text
current gradient g 相同
旧 run 使用已有的 m_1000、v_1000、step=1000
新 optimizer 使用 m=0、v=0、step=0
```

Adam bias correction 和 update direction/scale 因此不同。

第二问：scheduler state 记录当前位于 LR trajectory 的哪一步，例如
`last_epoch/step_count/current lr`。若它从 0 重新初始化，LR 可能重新经历
warmup，或从初始/峰值 LR 重新开始，而不是继续 step 1001 的 cosine LR。
这不是简单地“多训练几步”，而是每个后续 update 使用了错误的 learning rate。

`global_step`/trainer progress 还会影响：

- 训练循环是否跳过已经消费的 batches；
- 何时结束 `max_steps`；
- 何时 logging、eval 和 save；
- checkpoint 命名与 best-checkpoint tracking。

第三问：RNG 是 pseudo-random number generator 的内部状态。给定相同 RNG
state，后续生成的“随机”序列可重复。Dropout 的逻辑可简化为：

```python
mask = bernoulli(keep_probability, rng_state)
output = input * mask / keep_probability
```

训练中每次 forward 都会消耗 RNG。若 resume 时不恢复 PyTorch CPU/CUDA
RNG state，即使 model 和 batch 相同，也会产生不同 dropout mask，继而得到
不同 activation、loss、gradient 和后续 parameters。

Sampler/dataloader progress 则决定下一条 sample/index 和 shuffle 顺序。
因此缺少 RNG 或 sampler progress 时，训练通常仍能正常运行，但只是从相同
weights 开始的一条新随机轨迹，不是 exact/continuous resume。

完成 update 并 `zero_grad()` 后保存时，continuous resume 的主要状态包括：

```text
model:
  parameters + persistent buffers

optimizer:
  m/v/step、param groups、实现相关 master weights

scheduler:
  当前 scheduler position/state

training progress:
  global_step、epoch、已完成 batch/micro-step、callback/trainer state

randomness:
  Python、NumPy、PyTorch CPU、每个 CUDA device/rank 的 RNG state
  dataloader worker/sampler RNG（按实现）

data progress:
  sampler/dataloader position、shuffle permutation/epoch

numeric runtime:
  AMP GradScaler（若使用）

distributed metadata:
  checkpoint manifest、world topology、global tensor/shard mapping
  或格式支持 load-time reshard 所需的 metadata
```

`.grad` 在本题保存点不必保存，因为 update 已完成且 gradients 已被清空；
下一次 update 本来就应从空 gradient buffers 开始。若 checkpoint 位于
gradient accumulation 中途，则 exact resume 必须额外恢复已经累积的
gradients 和当前 micro-step，或者干脆只允许在 update boundary 保存。

#### Accelerate 的职责

Accelerate 不是“不管分布式的 training logic”。恰恰相反，它是
Transformers Trainer 与具体 distributed backend 之间的 execution adapter：

```text
Trainer 决定“做什么、什么时候做”:
  取 batch -> forward/loss -> backward
  到 accumulation boundary 后 clip/step/scheduler
  logging/eval/save cadence

Accelerate 决定“在当前设备/后端上怎样做”:
  process/device 初始化
  model/optimizer/dataloader/scheduler 的 prepare/wrap
  DDP/FSDP/DeepSpeed plugin 选择与接入
  accumulation/no_sync 和 gradient sync 边界
  mixed precision / GradScaler
  backward、clip、gather/reduce、distributed state save/load 的统一接口

DeepSpeed plugin 启用时:
  Accelerate 调用 deepspeed.initialize(...)
  accelerator.backward(...) 转发到 DeepSpeed engine.backward(...)
```

因此最短记忆法是：

```text
Trainer   = training state machine
Accelerate = distributed/device backend adapter
DeepSpeed = specialized distributed optimization runtime
NCCL      = collective executor
```

### D5-Q5 定向追问

只回答三点：

1. Adam 的 `m/v` 是完整 gradient distribution，还是每个 parameter element
   的哪两类历史统计？
2. 相同 model weights、相同 batch，但 RNG state 不同，dropout mask、gradient
   和下一步 parameter 是否还能保证相同？
3. 为什么 update boundary 的 checkpoint 不需要 `.grad`，而 accumulation
   中途的 checkpoint 若要 exact resume 就需要？

#### D5-Q5 定向追问回答与点评

学习者回答：

```text
1. m/v 是一阶、二阶 momentum；v 好像不是二阶，但 idea 类似。
2. RNG 不同时不能保证 gradient 相同。
3. 因为 checkpoint 保存了。

追加问题：一般什么时候保存 checkpoint？为什么听说 checkpoint 能省显存？
```

第一点需要术语校准：

```text
m_t = beta1 * m_(t-1) + (1-beta1) * g_t
      梯度的一阶 raw moment 的 EMA

v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2
      梯度平方的二阶 raw moment 的 EMA
```

`v` 确实是 second-moment estimate，但不是 Hessian、二阶导数或真正的
second-order optimizer curvature。把 `m` 口语称为 momentum 很常见；
更严谨的说法是 first-moment EMA，`v` 是 second-raw-moment EMA。

第二点正确。第三点的因果关系应改成：

```text
update boundary:
  已累积 gradients -> optimizer.step 已消费 -> zero_grad 已清空
  下一 update 本来就从空 grad 开始
  所以无需保存 .grad

accumulation 中途:
  .grad 中仍包含前几个 micro-batches 尚未消费的贡献
  若不保存，resume 后这些贡献丢失
  所以 exact resume 需要保存 .grad + 当前 micro-step
```

不是“因为 checkpoint 已经保存了”所以不用 grad，而是保存点的训练语义决定
此时 grad 是否仍是未消费状态。

#### 两种同名 checkpoint

```text
1. Training checkpoint / resume checkpoint
   把 model、optimizer、scheduler、RNG、progress 等写到持久存储
   目的：故障恢复、继续训练、模型选择
   通常不节省训练显存

2. Activation checkpointing / gradient checkpointing
   forward 时不保留某些中间 activations
   backward 需要这些 activations 时重新执行对应 forward
   目的：以额外计算换显存
   不能代替断点续训 checkpoint
```

Activation checkpointing 节省的是：

```text
普通训练:
  forward 保存每层 backward 所需 activations
  -> activation memory 随 layers、batch、sequence 增长

activation checkpointing:
  只保存选定 block 的输入/边界 activations
  丢弃 block 内部 Q/K/V、MLP intermediates 等
  backward 时重新 forward 该 block
  -> activation memory 降低，计算量和 step time 增加
```

它通常不减少 parameters、gradients 或 Adam states 的持久显存。带 dropout
的重计算还需要一致的 RNG；PyTorch activation checkpointing 默认会处理
相关 RNG state，以便 recompute 使用兼容的随机 mask。

Training checkpoint 的常见保存时机：

- 首选完整 optimizer update boundary：
  `backward/sync -> clip -> optimizer.step -> scheduler.step -> zero_grad` 之后；
- 每隔固定 optimizer steps 或固定墙钟时间，频率由可接受的故障重算时间决定；
- epoch/数据阶段结束；
- eval 产生新的 best metric 时；
- 预计抢占、维护或租卡到期之前；
- 最终完成时。

生产环境还需避免写出半成品 checkpoint：分布式 ranks 协同写入，成功后再提交
manifest/完成标记。ZeRO/FSDP 的 sharded checkpoint 可以避免在保存时把完整
状态集中到单卡；这是避免保存过程显存峰值，不等于 training checkpoint
本身能降低日常训练显存。

### D5-Q5 Checkpoint 术语追问

只回答三点：

1. 哪一种 checkpoint 用于机器故障后的 continuous resume？
2. 哪一种 checkpoint 能降低 activation memory？它用什么代价换显存？
3. 为什么 training checkpoint 最适合在完整 optimizer update boundary 保存？

#### D5-Q5 Checkpoint 术语追问回答与点评

学习者回答：

```text
1. Training checkpointing。
2. Activation checkpointing；代价是更多计算、更长时间。理解成保存一部分
   参数、丢掉完整 activation。
3. 认为 update boundary 的理由已经直接明显。
```

第 1、2 题主干正确。关键纠正：activation checkpointing 不保存“一部分参数”。
Parameters 仍按原本的 replicated/sharded/offloaded 方案存在；该机制改变的是
autograd 为 backward 保留哪些 activations。

```text
x0 -> checkpointed Transformer block -> x1

普通模式:
  保存 block 内部的 Q/K/V、attention probabilities、MLP intermediate 等

activation checkpointing:
  保存少量边界 activation（例如 x0）和必要 RNG context
  丢弃 block 内部 activation
  backward 时重新执行 block(x0)，重建内部 activation 后求 gradient
```

所以“checkpoint”指 computation graph 中保留的重算边界，不是 parameter
checkpoint，也不是把 gradient 写到磁盘。

第 3 题的最短答案是：update boundary 没有未消费的 partial gradients，
model/optimizer/scheduler/global_step 已共同落在同一个稳定 step；中途保存则
还要额外捕获 `.grad`、micro-step 和部分执行状态。无需继续追问。

判定：`passed_after_checkpoint_term_correction`。模块 E 完成。

### D5-Q6 — Eval 不是“少做一个 backward”

一个固定 eval set 包含两个 batch：

```text
batch 1:
  valid label tokens = 10
  mean token loss = 1.0

batch 2:
  valid label tokens = 90
  mean token loss = 3.0
```

训练代码准备这样做：

```python
model.eval()
with torch.no_grad():
    ...
```

请回答：

1. `model.eval()` 与 `torch.no_grad()` 分别改变什么？为什么二者不是同一件事？
2. 整个 eval set 的正确 mean token loss 是多少？为什么不能直接算
   `(1.0 + 3.0) / 2`？
3. Eval 期间是否应该调用 `backward()`、`optimizer.step()` 或
   `scheduler.step()`？结束后为什么通常还要恢复 `model.train()`？

#### D5-Q6 第一轮回答与点评

学习者回答：

```text
1. 不知道，希望先讲 model.eval() 和 torch.no_grad()。
2. 2.8，需要按有效 token 数加权。
3. 等第 1 题讲解后再回答。
```

第二题正确：

```text
eval mean token loss
= (10 * 1.0 + 90 * 3.0) / (10 + 90)
= 2.8
```

直接平均两个 batch mean 得到 `2.0`，错误地让 10-token batch 和
90-token batch 拥有相同权重。分布式 eval 也应跨 ranks 聚合 loss numerator
与 valid-token denominator，最后只除一次。

第一题：

```python
model.eval()
```

递归地把 module 的 `training` flag 设为 `False`，改变依赖 train/eval mode
的 layer 行为。典型例子：

- Dropout 在 train mode 随机置零，在 eval mode 关闭；
- BatchNorm 在 train mode 使用/更新当前 batch statistics，在 eval mode
  使用已有 running statistics；
- Transformer 常用的 LayerNorm 通常在 train/eval mode 下行为相同。

`model.eval()` 不会冻结 parameters，也不会关闭 autograd。若不使用
`no_grad()`，PyTorch 仍会构建计算图，仍可以对 eval forward 的 loss 调用
`backward()`。

```python
with torch.no_grad():
```

临时关闭 gradient recording。其内部大多数 operations 不构建 backward
graph，从而减少 activation memory 和 autograd 开销；离开 context 后恢复
原来的 grad mode。它不修改 `model.training` flag，也不会自动关闭 dropout。

因此存在两种有意但不同的组合：

```python
model.train()
with torch.no_grad():
    output = model(batch)
# dropout 仍启用，但不记录 graph

model.eval()
output = model(batch)
# dropout 已关闭，但若没有 no_grad，仍会记录 graph
```

标准 deterministic eval 通常同时使用：

```python
model.eval()
with torch.no_grad():
    ...
```

### D5-Q6 第三问重答

基于上述区别，只回答：

1. 标准 eval 是否应调用 `backward()`、`optimizer.step()`、
   `scheduler.step()`？
2. Eval 完成后若继续训练，为什么要调用 `model.train()`？

#### D5-Q6 第三问回答与点评

学习者回答：

```text
1. 不调用；eval 更多是纯 forward。
2. 把 module.training 重新设为 True，恢复训练模式，重新启用 dropout 等行为。
```

两项均正确。标准 eval 不更新 gradients、parameters、optimizer state 或
scheduler state；结束后恢复 train mode，保证后续 forward 使用训练期行为。

判定：`passed_after_teaching`。模块 F 完成。

### D5-Q7 — 综合验收：一条 sample 的完整旅程

不要求展开数学推导，用自己的话把下面这条链补完整。每一项用一句话即可：

```text
1. raw conversation
-> 2. template/tokenizer
-> 3. collator
-> 4. model forward + masked loss
-> 5. backward + gradient accumulation/sync
-> 6. clipping + optimizer/scheduler + zero_grad
-> 7. training checkpoint
-> 8. eval
```

回答时至少覆盖：

- 第 2、3 步分别产出什么；
- 第 5、6 步分别修改哪类状态；
- 第 7 步为了 continuous resume 保存哪些核心状态；
- 第 8 步为何同时使用 `model.eval()` 和 `no_grad()`；
- 在链末补一句框架职责：
  `ms-swift / Trainer / Accelerate / DeepSpeed / NCCL / CUDA` 分别位于哪一层。

#### D5-Q7 回答与引导完成

学习者回答摘要：

```text
tokenize 后得到 X[T,D]；
collator 组装 batch、padding 和 mask；
resume 要保存 optimizer 状态；
no_grad 节约 eval 显存；
ms-swift 负责高层数据/训练组装，Trainer 负责训练逻辑和状态管理，
Accelerate 定义分布式逻辑，DeepSpeed 执行分布式训练，NCCL 通信，CUDA 计算。
```

已经正确掌握：

- collator 是组装/padding batch 的边界；
- `no_grad()` 通过不构建 backward graph 降低 eval memory；
- ms-swift、Trainer、NCCL、CUDA 的大体层级；
- optimizer state 是 continuous resume 的必要组成。

需要纠正：

1. Tokenizer 输出离散 integer IDs `[T]`，不是 `[T,D]`；model embedding
   才把 collated `input_ids [B,T]` 映射为 hidden activations `[B,T,D]`。
2. Resume 除 optimizer state 外，还需要 model、scheduler、progress、RNG、
   sampler/dataloader progress 等。
3. Accelerate 是 distributed/device backend adapter；DeepSpeed 不只是被动
   执行，它解释 ZeRO 配置并决定 sharding/state layout/communication schedule。

完整链路：

```text
1. raw conversation
   messages/metadata 等语义样本

2. template/tokenizer
   template 加 role markers、BOS/EOS 并标出监督 span
   tokenizer 把文本映射为 variable-length input_ids/labels [T]

3. collator
   padding 并组装 input_ids/labels/attention_mask [B,T]

4. model forward + masked loss
   embedding: [B,T] -> [B,T,D]
   Transformer/lm_head: -> logits [B,T,V]
   causal shift + labels != -100 -> scalar token-normalized loss

5. backward + accumulation/sync
   backward 计算并累积 parameter.grad
   DP sync 合并不同 data replicas 对同一参数的 gradient contributions
   model parameter 此时尚未被更新

6. clipping + optimizer/scheduler + zero_grad
   clipping 必要时缩放 gradients
   optimizer.step 修改 model parameters 和 Adam m/v/step
   scheduler.step 推进 LR state
   zero_grad 清空已消费 gradients

7. training checkpoint
   保存 model、optimizer、scheduler、global/trainer progress、RNG、
   sampler/dataloader progress、GradScaler（若有）和 distributed metadata

8. eval
   model.eval() 关闭 dropout 等 train-mode 行为
   no_grad() 关闭 autograd graph recording
   只做 forward 和正确加权的 metric aggregation，不做 backward/step
   继续训练前恢复 model.train()
```

框架职责：

```text
ms-swift:
  CLI/recipe、模型/数据/template/tuner 适配

Transformers Trainer:
  training state machine、step/eval/save cadence

Accelerate:
  device/process/distributed backend 统一接入和 dispatch

DeepSpeed:
  ZeRO sharding、state ownership、communication schedule 与 specialized runtime

NCCL:
  执行跨 GPU collectives

CUDA/cuBLAS/Triton:
  GPU runtime、memory/streams 与本地 compute kernels
```

判定：`completed_guided`。

## Day 05 Quiz 最终结果

| 模块 | 结果 |
|---|---|
| A. 单步 lifecycle | `completed_after_teaching` |
| B. 数据与 loss 边界 | `completed_after_teaching` |
| C. 框架职责 | `completed_after_code_guided_teaching` |
| D. 源码追踪 | `completed_after_evidence_order_teaching` |
| E. Checkpoint/resume | `completed_after_teaching` |
| F. Eval 边界 | `completed_after_teaching` |

已形成的核心能力：

- 区分 parameter、gradient、optimizer state 与 activation；
- 解释 accumulation boundary 和一个 optimizer update 的状态变化；
- 区分 template、tokenizer、embedding、collator、attention mask 与 loss mask；
- 从 `swift sft` 入口追到 Trainer、model forward、DeepSpeed 和底层执行边界；
- 说明 training checkpoint 与 activation checkpointing 的不同；
- 列出 continuous resume 的主要状态；
- 区分 eval mode、no-grad 与 token-weighted metric aggregation。

需要在后续实践继续强化：

- tokenizer 输出 `[T]` IDs，embedding 才产出 `[B,T,D]`；
- 遇到训练故障时从最近的 state boundary 按证据倒查；
- Accelerate 是 backend adapter，DeepSpeed 是会决定 ZeRO state layout 的 runtime。
