# Day 05 — ms-swift Codebase Walkthrough

上游源码：`modelscope/ms-swift@565a1ad586a21d24b23931c52d2c62b49c39bee8`

Life repo 学习快照：`94f904c`；源码内联注释保存在 `learning/training-learning/patches/ms-swift-565a1ad-day05-comments.patch`。

状态：`living_document`

用途：把 ms-swift 全仓结构映射到 Day 05 的 training lifecycle、state ownership、framework boundaries 和故障检查点。外部源码 clone 默认保持干净；需要查看内联学习注释时应用 patch。后续行号可能移动，因此优先以函数名和对象边界定位。

## 0. 一句话模型

ms-swift 的核心不是重新实现 PyTorch，而是把大量模型、数据格式、训练目标和执行后端接到一套统一入口上：

```text
Registry 选择具体实现
-> Arguments 把 CLI 解析为一份已归一化配置
-> Pipeline 组装 model/template/dataset/trainer(or engine)
-> Trainer/Engine 执行训练、推理、评测或导出
-> PyTorch/Accelerate/DeepSpeed/Megatron/NCCL/CUDA 完成底层 runtime 工作
```

三个最重要的架构概念：

1. `Registry`：模型、template、dataset、trainer、loss、metric、optimizer、callback 都有映射或注册机制。
2. `Pipeline`：负责编排一次完整用户任务，但尽量不重新实现底层训练数学。
3. `upstream inheritance / backend delegation`：普通 SFT 复用 Transformers Trainer；RLHF trainer 多数复用 TRL；分布式执行再交给 Accelerate、DeepSpeed/FSDP 或独立 Megatron 栈。

## 1. 全仓分层地图

### 1.1 用户表面层

| 位置 | 作用 | 不负责什么 |
|---|---|---|
| `README*.md`、`docs/` | 概念、参数和使用文档 | runtime 实现 |
| `examples/` | 可运行 recipe；训练示例最多 | 通用业务逻辑 |
| `swift/ui/` | Web UI 表单和运行页面 | 核心训练计算 |
| `tests/` | 行为契约、回归和模型/训练覆盖 | 生产入口 |

### 1.2 入口与控制平面

| 目录 | 作用 |
|---|---|
| `swift/cli/` | `swift sft`、`swift infer` 等命令路由；必要时启动 `torchrun` |
| `swift/arguments/` | 组合式 dataclass 参数、默认值归一化、兼容检查、模型/数据/template 工厂入口 |
| `swift/pipelines/` | 按任务组装对象并把控制权交给 Trainer 或 Engine |

### 1.3 适配与注册平面

| 目录 | 作用 | Day 05 对应 |
|---|---|---|
| `swift/model/` | 模型识别、hub 下载、loader、processor、模型 patch 和 architecture metadata | model construction / model state owner |
| `swift/dataset/` | dataset registry、加载、列规范化、语义 row preprocessing、packing | raw data boundary |
| `swift/template/` | messages 格式化、tokenization、labels/loss mask、collator、多模态适配 | template/tokenize/collate/loss contract |
| `swift/agent_template/` | tools/tool-call 的提示格式与解析 | agent message contract |
| `swift/loss_scale/` | assistant/tool/thinking 等 token span 的 loss weighting | masked loss contract |
| `swift/tuner_plugin/` | 可插拔 tuner 接口和轻量映射 | trainable state selection |
| `swift/tuners/` | LoRA、LongLoRA、Adapter、ReFT 等具体参数高效微调实现 | 哪些 parameters 可训练 |

### 1.4 执行平面

| 目录 | 作用 | 主要上游/下游 |
|---|---|---|
| `swift/trainers/` | 普通 SFT/分类/embedding/reranker Trainer 扩展 | 继承 Transformers Trainer，调用 Accelerate |
| `swift/rlhf_trainers/` | DPO、GRPO、KTO、PPO、Reward、GKD 等 | 多数继承 TRL trainer，并接 rollout/reward |
| `swift/infer_engine/` | Transformers/vLLM/SGLang/LMDeploy 的统一推理协议 | 模型/template 到 backend engine |
| `swift/sequence_parallel/` | sequence parallel sampler、layout 和通信实现 | PyTorch distributed collectives |
| `swift/megatron/` | 独立 Megatron-Core 参数、pipeline、trainer、checkpoint 与 HF↔MCore 转换栈 | 不只是 HF Trainer 的一个小开关 |
| `swift/ray/`、`swift/ray_utils/` | Ray worker/resource orchestration，尤其是 Megatron + rollout/RL | 多角色、多资源池执行 |
| `swift/rollout/`、`swift/rl_core/` | multi-turn rollout、environment、advantage/GRPO 数据与算法辅助 | online RL dataflow |

### 1.5 横切插件

| 目录 | 作用 |
|---|---|
| `swift/loss/` | causal-LM、embedding、reranker 等 loss 插件 |
| `swift/metrics/` | accuracy、NLG、embedding/reranker 和运行统计 |
| `swift/optimizers/` | GaLore、Muon、LoRA+、多模态 optimizer 等扩展 |
| `swift/callbacks/` | early stop、性能日志、activation offload、DeepSpeed elastic 等 |
| `swift/rewards/` | ORM/PRM/reward function 插件 |
| `swift/dataloader/` | batch sampler/shard/dispatcher |
| `swift/hub/` | ModelScope/Hugging Face hub 抽象 |
| `swift/utils/` | logging、I/O、distributed/device、HF config、processor 等通用设施 |
| `swift/config/` | ZeRO/FSDP preset JSON |

## 2. 命令是如何进入代码的

安装入口位于 `setup.py`：

```python
entry_points={
    'console_scripts': [
        'swift=swift.cli.main:cli_main',
        'megatron=swift.cli._megatron.main:cli_main',
    ]
}
```

因此：

```text
shell: swift sft --model ... --dataset ...
-> swift.cli.main.cli_main()
-> ROUTE_MAPPING['sft'] == 'swift.cli.sft'
-> 若设置 NPROC_PER_NODE/NNODES，则包装为 torch.distributed.run
-> swift/cli/sft.py
-> swift.pipelines.sft_main()
```

`swift/cli/main.py` 还负责：

- 将 YAML/JSON 第一参数展开为 CLI flags；
- 读取多机/多卡环境变量；
- 为训练路线选择普通 Python 或 `torchrun` 子进程；
- 把真正任务交给具体 CLI 文件。

Day 05 映射：这是 `CLI/config -> runtime process topology` 的第一条边界。此处还没有 dataset tensor、loss 或 optimizer。

## 3. Arguments：从字符串变成可执行配置

普通 SFT 使用：

```python
class SftArguments(
    SwanlabArguments,
    TunerArguments,
    BaseArguments,
    Seq2SeqTrainingArguments,
):
    ...
```

`BaseArguments` 又组合：

```text
GenerationArguments
QuantizeArguments
DataArguments
TemplateArguments
ModelArguments
RayArguments
```

这层不是单纯存字段。`__post_init__` 会做真实决策：

- 识别 rank/world size 和 device；
- 解析 model metadata、dtype、device map、quantization；
- 选择 template 和 max length；
- 注册/解析 dataset；
- 设置 lazy tokenize、packing、streaming；
- 把 `zero3` 名称解析为 `swift/config/zero3.json`；
- 构造传给 HF Trainer 的 `training_args`。

关键工厂方法：

```text
args.get_model_processor() -> swift.model.get_model_processor(...)
args.get_template(processor) -> swift.template.get_template(...)
args.load_dataset() -> swift.dataset.load_dataset(...)
args.save_args() -> output_dir/args.json
```

Day 05 隐藏默认值通常首先在这里寻找，而不是在 CUDA kernel 中寻找。

## 4. Registry：ms-swift 扩展性的骨架

### 4.1 Model registry

```text
model id/path + config
-> get_model_info_meta(...)
-> MODEL_MAPPING[model_type] -> ModelMeta
-> model_meta.loader(...)
-> loader.load()
-> (PyTorch model, processor/tokenizer)
```

`ModelMeta` 连接：

- model family/type；
- 默认 template；
- loader；
- model architecture keys；
- multimodal/reward/MoE 等能力 metadata。

`swift/model/models/*.py` 是大量模型族注册文件，不是每个模型都重写训练循环。

### 4.2 Template registry

```text
model_meta.template or explicit --template
-> TEMPLATE_MAPPING[template_type]
-> TemplateMeta
-> Template(processor, template options)
```

`swift/template/templates/*.py` 描述不同模型的 role markers、prefix/suffix、特殊 token、多模态规则等。

### 4.3 Dataset registry

```text
dataset name/path
-> DatasetMeta / SubsetDataset
-> hub/local loader
-> RowPreprocessor
-> 统一语义 row（messages/media/label/chosen/rejected 等）
```

### 4.4 Trainer 和插件 mappings

```text
task_type/rlhf_type -> TrainerFactory -> Trainer class
tuner_type -> tuner implementation
loss name -> loss implementation
metric name -> metric implementation
optimizer name -> optimizer callback
callback name -> callback implementation
```

Registry 回答“选谁”；Pipeline 回答“怎样把这些对象连起来”。

## 5. `swift sft` 的完整主链

### 5.1 Pipeline 构造

```text
sft_main(args)
-> SwiftSft(args).main()
-> SwiftPipeline.__init__
   -> parse SftArguments
   -> set seed
-> SwiftSft.__init__
   -> _prepare_model_tokenizer()
   -> _prepare_template()
   -> optional flash checkpoint setup
-> SwiftPipeline.main()
   -> SwiftSft.run()
```

### 5.2 Model/processor

`SwiftSft._prepare_model_tokenizer()` 调用：

```python
self.model, self.processor = args.get_model_processor()
```

输出对象：

```text
model: PyTorch/Transformers model，持有 model parameters/buffers
processor: tokenizer 或 multimodal processor，持有 vocab/special-token/media contract
```

此时 optimizer 尚未创建，`.grad` 也尚未存在。

### 5.3 Template

```python
template = args.get_template(self.processor)
template.set_mode('train')
```

Template 同时服务多个边界：

- semantic messages -> role-aware context；
- context string/pieces -> integer token IDs；
- 根据 loss scale 生成 `labels` 与 `-100`；
- 对 variable-length rows 做 padding/collation；
- 多模态 processor hooks；
- 调用 model 并适配 SFT loss。

所以在 ms-swift 中，Template 比“一个 prompt 字符串模板”更重。

### 5.4 Dataset load 与 encode

```text
args.load_dataset()
-> swift.dataset.load_dataset()
-> DatasetLoader
-> hub/local dataset
-> RowPreprocessor
-> semantic rows
```

随后 `Template.encode(row)` 输出单样本对象：

```text
input_ids: [T] integer IDs
labels: [T] integer target IDs or -100
loss_scale: [T] optional token weights
length(s), multimodal fields, task-specific fields
```

实际 encode 可以按配置在 preprocessing 或 dataset item fetch 时发生；重要的语义边界不变：此处仍是 variable-length 单样本 lists，不是 `[B,T,D]` embeddings。

### 5.5 Collator

Trainer 初始化时，`SwiftMixin` 将：

```python
data_collator = partial(template.data_collator, padding_to=...)
```

DataLoader 取到多条 encoded rows 后：

```text
List[{input_ids:[T_i], labels:[T_i], ...}]
-> template.data_collator(...)
-> input_ids/labels/attention_mask [B,T_max]
```

packing、padding-free、多模态、RLHF 和 sequence-parallel 路径会使用不同物理布局，不能把所有路径都假设成普通 `[B,T]` padding。

### 5.6 Tuner / trainable parameters

`SwiftSft.run()` 在创建 Trainer 前调用：

```python
self.model = self.prepare_model(...)
```

`TunerMixin.prepare_model()` 决定：

- full fine-tuning：哪些参数 `requires_grad=True`；
- LoRA/PEFT：向哪些 target modules 插入 adapter；
- resume adapter：从哪里恢复 adapter weights；
- trainable dtype、freeze/activate rules；
- GaLore 等 optimizer 前置选择。

Day 05 映射：这是“哪些 model state 会在 optimizer.step 中被修改”的所有权边界。

### 5.7 Trainer selection

```python
trainer_cls = TrainerFactory.get_trainer_cls(args)
```

主要映射：

```text
causal_lm -> swift.trainers.Seq2SeqTrainer
seq_cls -> swift.trainers.Trainer
embedding -> EmbeddingTrainer
reranker -> RerankerTrainer
DPO/GRPO/KTO/... -> swift.rlhf_trainers.*
```

普通 causal-LM Trainer：

```python
class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    ...
```

这说明通用 loop 属于 Transformers；ms-swift mixin 负责注入 collator、loss、metric、checkpoint、model/template 特殊处理和兼容逻辑。

### 5.8 Forward 与 loss

调用链：

```text
Transformers Trainer.training_step(...)
-> ms-swift Seq2SeqTrainer.compute_loss(...)
-> template.compute_sft_loss(model, inputs, ...)
-> model(**inputs)
```

普通 padded causal-LM shape：

```text
input_ids/labels/attention_mask: [B,T]
embedding/internal hidden: [B,T,D]
logits: [B,T,V]
loss: scalar
```

默认路径可让 Transformers model 自己完成 shifted causal-LM loss；自定义 loss-scale/DFT/channel/sequence-parallel 路径会取出 labels，对 per-token loss 做额外处理，再按有效 token 数归一化。

Day 05 关键点：ms-swift 不在 Trainer 中手写 Q/K/V；Q/K/V 属于具体 model forward。

### 5.9 Backward、accumulation、sync、clip、step

主循环来自 Transformers Trainer，经过 Accelerate 统一接口：

```text
training_step
-> accelerator.backward(loss)
-> accumulation boundary?
   no: 保留/累积 gradients
   yes:
     gradient sync/partition complete
     clip_grad_norm
     optimizer.step
     scheduler.step
     zero_grad
     global_step += 1
```

后端分工：

```text
Trainer:
  决定何时发生这些动作

Accelerate:
  统一 device/process/backend，wrap model/optimizer/dataloader，dispatch backward/clip

DeepSpeed/FSDP/DDP:
  决定具体 sharding/sync/runtime 行为

PyTorch autograd/optimizer:
  构建 graph、计算 gradients、执行本地 update 数学

NCCL:
  执行 CUDA ranks 之间的 collective

CUDA/cuBLAS/Triton:
  执行本地 GPU runtime 与 kernels
```

### 5.10 Checkpoint/resume

Pipeline 最终调用：

```python
trainer.train(resume_checkpoint)
```

`SwiftMixin._save_checkpoint()` 通常委托 parent Trainer 保存完整训练快照，并处理 ZeRO-3/flash-checkpoint 等后端差异。

典型状态：

```text
model parameters/buffers
optimizer m/v/step/param-groups
scheduler state
TrainerState/global_step/callback state
Python/NumPy/PyTorch CPU/CUDA RNG
GradScaler（若有）
distributed checkpoint metadata/shards
data skip/sampler progress（由 trainer/dataloader resume 语义共同决定）
```

`args.json` 是 recipe/config provenance，不等同于完整训练状态。

### 5.11 Eval

要区分两条 eval 路径：

```text
训练内 eval:
  Trainer.evaluate()/prediction_step
  用于 validation loss/metrics、best checkpoint、训练期间决策

独立 swift eval:
  SwiftEval + EvalScope + deployment/API service
  用于 benchmark suite，不是 Trainer 内部 validation loop
```

训练内标准 eval 仍遵守：

```text
model.eval()
no_grad/inference context
forward-only
跨 batch/rank 正确聚合 metrics
继续训练前 model.train()
```

## 6. Day 05 lifecycle 映射表

| Lifecycle stage | ms-swift 主要位置 | 输入 -> 输出 | 修改的状态 | 第一检查点 |
|---|---|---|---|---|
| CLI/config | `cli/`、`arguments/` | strings/YAML -> `SftArguments` | config/process env | resolved args、rank/world size |
| model load | `model/`、`SwiftSft._prepare_model_tokenizer` | model id -> model/processor | model parameters loaded | model type/dtype/device map |
| raw dataset | `dataset/loader.py`、preprocessor | external rows -> semantic rows | dataset object | columns/messages/sample IDs |
| template/tokenize | `template/base.py:encode` | semantic row -> `[T]` lists | no training state | input_ids、labels、valid-label count |
| collate | `template.data_collator` | rows -> batch tensors | no model state | shapes/dtypes/padding/masks |
| forward | concrete model via `compute_sft_loss` | `[B,T]` -> logits/loss | activations/RNG consumed | inputs、logits shape、finite loss |
| backward | HF Trainer -> Accelerate -> backend | scalar loss -> `.grad` | gradient buffers | grad None/zero/NaN/norm |
| sync/shard | DDP/FSDP/DeepSpeed/SP/Megatron | local contributions -> target layout | distributed gradient/parameter working set | process group、collective sequence/layout |
| clip/update | Trainer/optimizer | gradients -> new parameters | gradients、parameters、optimizer state | LR、step skip、parameter delta |
| scheduler/zero | Trainer | update complete -> next boundary | scheduler、grad buffers | scheduler step、grad cleared |
| checkpoint | `trainers/mixin.py` + upstream/backend | runtime states -> files/shards | persistent snapshot | manifest/files/global step |
| resume | Pipeline + Trainer/backend load | checkpoint -> runtime states | all restored state | optimizer/scheduler/RNG/data progress |
| train eval | Trainer/metrics | val batches -> aggregated metric | eval metrics only | eval mode/no_grad/token denominator |

## 7. 其他执行路线怎样复用主链

### 7.1 Pre-training

`SwiftPretrain(SwiftSft)` 几乎完全复用 SFT pipeline，主要通过 `PretrainArguments` 改变 template/loss/data contract。它证明 pipeline skeleton 与目标数据契约可以分离。

### 7.2 RLHF

`SwiftRLHF(SwiftSft)` 复用 model/data/template/trainer orchestration，但增加：

- policy 以外的 ref/reward/value/teacher models；
- chosen/rejected/KTO/rollout 数据结构；
- template mode；
- TRL-derived DPO/GRPO/KTO/PPO/GKD trainer；
- rollout engine、reward functions 和权重同步。

这条路线将在 Day 22–29 深挖；Day 05 只需要知道它不属于普通 SFT 单 model 主链。

### 7.3 Inference/deploy

`SwiftInfer` 复用 Arguments、model registry 和 template，但把 Trainer 换成 InferEngine：

```text
transformers
vLLM
SGLang
LMDeploy
```

统一协议位于 `infer_engine/protocol.py`；后端选择位于 `SwiftInfer.get_infer_engine()`。

### 7.4 Sampling/eval/export/app

- `sampling/`：批量生成/蒸馏数据，并有自己的进度文件；
- `eval/`：启动/连接服务后交给 EvalScope benchmark；
- `export/`：merge LoRA、quantize、Ollama、cached dataset、HF↔MCore 转换、push hub；
- `app/` 与 `ui/`：用户界面层，调用已有 pipeline，不应承载核心训练数学。

### 7.5 Megatron 与 Ray

`swift/megatron/` 是一套平行训练栈：

```text
MegatronArguments
-> Megatron pipeline
-> MCore model conversion/preparation
-> Megatron trainer/callback/checkpoint
-> TP/PP/DP/CP runtime
```

`swift/ray/` 再在其上提供多 worker/resource pool、rollout server、weight transfer 和 checkpoint engine orchestration。读普通 SFT 时不要跳进这里；只有问题涉及 Megatron、online RL 或多角色资源拓扑时才进入。

## 8. 源码阅读顺序

### 第一圈：普通 SFT 主链

1. `setup.py`
2. `swift/cli/main.py`
3. `swift/cli/sft.py`
4. `swift/pipelines/base.py`
5. `swift/arguments/sft_args.py` 与 `arguments/base_args/`
6. `swift/pipelines/train/sft.py`
7. `swift/model/register.py`
8. `swift/template/base.py`
9. `swift/dataset/loader.py` 与 `dataset/utils.py`
10. `swift/pipelines/train/tuner.py`
11. `swift/trainers/trainer_factory.py`
12. `swift/trainers/seq2seq_trainer.py`
13. `swift/trainers/mixin.py`
14. upstream `transformers.Trainer` 与 `accelerate.Accelerator`

### 第二圈：选择性扩展

- LoRA/adapter：`tuner.py`、`tuner_plugin/`、`tuners/`；
- 特定模型：对应 `model/models/<family>.py` 和 `template/templates/<family>.py`；
- DeepSpeed/FSDP：`arguments/sft_args.py`、`config/*.json`、Accelerate plugin；
- 自定义 loss/metric/optimizer/callback：对应 mapping 文件；
- inference：`pipelines/infer/`、`infer_engine/`；
- RLHF：`pipelines/train/rlhf.py`、`rlhf_trainers/`、`rollout/`；
- Megatron：最后单独阅读 `megatron/`，不要与 HF Trainer 路线混读。

## 9. 按问题找代码

| 想问的问题 | 第一站 |
|---|---|
| 某 CLI 参数为何变成这个默认值？ | `arguments/` 对应 dataclass 和 `__post_init__` |
| 模型为何选了这个 template/loader？ | `model/model_meta.py`、`model/register.py`、`template/register.py` |
| 数据列为何变成 messages？ | `dataset/preprocessor/`、dataset registry |
| 哪些 tokens 参与 SFT loss？ | `template/base.py:_encode_context_list`、`loss_scale/` |
| Padding/shape 为何异常？ | `template.data_collator`、`dataloader/` |
| LoRA 插到了哪些模块？ | `pipelines/train/tuner.py`、model architecture metadata |
| Loss 在哪里计算？ | `trainers/seq2seq_trainer.py:compute_loss`、`template.compute_sft_loss`、model forward |
| Gradient 为什么没更新参数？ | HF Trainer update boundary、Accelerate backward、optimizer/LR/AMP skip |
| ZeRO-3 是否真的启用？ | resolved deepspeed config、Accelerate distributed type、DeepSpeed engine identity |
| Checkpoint 是否可连续恢复？ | `trainers/mixin.py`、upstream Trainer/backend checkpoint files |
| 训练内 eval 与 `swift eval` 有何不同？ | Trainer evaluate vs `pipelines/eval/eval.py` |
| GRPO rollout/reward 在哪里？ | `rlhf_trainers/grpo_trainer.py`、`rollout_mixin.py`、`rewards/`、`infer_engine/` |

## 10. 当前学习边界

Day 05 要达到的是：

- 可以从一个命令追到每个主要对象边界；
- 可以说明每层决定什么、执行什么；
- 可以按状态变化定位故障；
- 可以识别普通 SFT、RLHF、inference、Megatron 是不同路线。

Day 05 不要求：

- 记住所有模型/template registry 项；
- 通读全部 526 个 `swift/` 文件；
- 现在就理解 GRPO rollout、Megatron TP/PP 或 Ray weight transfer 的全部实现；
- 把框架阅读扩展成重新设计 auto-training platform。

后续讨论以本文件为索引。任何时刻可以用“路径 / lifecycle stage / object / symptom”四种方式提问。
