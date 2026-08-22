# RSI failure analysis 与迭代 taxonomy

这份文档定义 RSI 在发现失败、判断根因和选择下一轮改动时使用的语义边界。
它是面向后续版本的诊断规范；机器可读的 failure / symptom ID 以
[`failure-taxonomy.json`](failure-taxonomy.json) 为准，intervention lever ID 以
[`taxonomy.json`](taxonomy.json) 为准。

## 1. 三条轴必须分开

一次 RSI 分析至少包含三条彼此独立的轴：

```text
symptom（观察到了什么）
    ↓ 证据定位
first broken invariant / failure（流水线中最先在哪里违反了约束）
    ↓ 因果假设与最小改动
intervention（下一版本实际改变什么）
```

它们不能共用一个标签：

- **Symptom** 是结果或轨迹，例如 Code eligibility 从 7/8 降到 0/8、loss
  发散、Finance 回退或证据缺失。它可以帮助搜索，但不能单独证明根因。
- **Failure** 是实际执行链上第一个有证据的 broken invariant，例如源 target
  正确，但经过训练模板后首个监督 token 丢失。failure 描述“在哪里坏了”。
- **Intervention** 是下一版本获准改变的变量，例如
  `model.target.representation.boundary_tokens`、`model.data.replay.replay_ratio`
  或 `model.objective.loss_weighting.token_type`。intervention 描述“准备改什么”。

failure 所在组件、代码归属团队和最终修改位置也可能不同。比如模板造成的
label construction failure 可以通过 target representation 修复；不能因为修复
发生在 model 配置中，就把根因重新命名为 model architecture failure。

## 2. First broken invariant 的判定规则

沿真实执行路径逐层检查，而不是从最终分数倒推：

```text
source → ingest/parse → select/mix/order → target construction
       → tokenize/render/mask → forward/objective → optimize/checkpoint
       → inference/harness → verifier/evaluation → promotion evidence
```

主 failure 应放在最早满足以下条件的边界：

1. 上游输入或约束已经被证据确认正确；
2. 该边界的实际产物违反了预先声明的 invariant；
3. 下游 symptom 与这个破坏存在可检验的机制链；
4. 关键相邻解释已被排除，或被明确记录为尚未排除的 confounder。

“最早”指**实际运行产物中最先被证明破坏的边界**，不是目录层级最靠前、
最容易修改或最符合直觉的组件。若证据只能支持相关性，使用 `suspected` 或
`supported`，不要写成 `causally_confirmed`。

建议使用四级因果状态：

- `suspected`：只有 symptom 与候选机制；
- `supported`：有正向证据，但仍存在重要替代解释；
- `causally_confirmed`：边界审计、干预或反事实证据闭合了机制链；
- `inconclusive`：证据不足或相互冲突，应先做诊断 probe。

一个 diagnosis 可以有一个 primary failure 和多个 contributing failures，但
不能把一串未经区分的可能性都当作根因。无法定位时，下一步应是便宜、可证伪
的 probe，而不是直接启动完整训练。

## 3. Symptom taxonomy

Symptom 允许多选，主要用于检索、聚类和触发诊断：

- **Outcome**：未过 gate、总分下降、cost-to-qualification 变差；
- **Domain / capability**：General、Math、Finance、Code 或特定能力回退；
- **Format / eligibility**：结构不合规、parser 拒绝、无法进入 sandbox；
- **Trajectory**：loss 发散、长期平台、checkpoint peak 后下跌、seed 方差过大；
- **Transfer / retention**：目标能力提升但基础能力遗忘、负迁移、跨域干扰；
- **Evidence / infrastructure**：产物缺失、hash 或比较身份不一致、sandbox 或
  runtime 失败。

Symptom 标签不得直接决定 intervention。例如“Code 降分”既可能来自 target
rendering，也可能来自 data mixture、objective、architecture、decoding 或 scorer。

## 4. Failure taxonomy 层级摘要

下面的层级用于 failure analysis。路径表示语义层级，不自动授予修改权限。

完整 leaf 列表见 [`failure-taxonomy.json`](failure-taxonomy.json)；下面只解释用于
定位 first broken invariant 的 family/topic 边界。

### 4.1 数据层 `failure.data.*`

- `failure.data.source.*`：来源身份、版本或 lineage 错误；
- `failure.data.ingestion.*`：读取、解析、schema、字段映射、编码或 join 错误；
- `failure.data.content.*`：噪声、冲突、标签错误、缺失或歧义；
- `failure.data.canonicalization.*`：whitespace、Unicode、单位和序列化错误；
- `failure.data.supervision.*`：
  - chat-template rendering；
  - tokenizer 与特殊 token 边界；
  - truncation、padding、packing；
  - label 与 loss-mask 对齐；
- `failure.data.population.*`：覆盖、类别平衡、mixture、选择偏差和质量分布；
- `failure.data.integrity.*`：重复、泄漏、benchmark contamination 和 identity collision；
- `failure.data.shift.*`：domain、temporal、train/serve 或 template distribution shift。

数据文件内容正确并不代表数据层没有 failure。若正确 target 在送入 forward
之前被模板、tokenizer、truncation 或 masking 改坏，仍属于
`failure.data.supervision.*`，更具体地说是 data-model interface failure。

### 4.2 模型层 `failure.model.*`

- `failure.model.representation.*`：tokenizer、embedding、position、modality 或 output representation 不匹配；
- `failure.model.architecture.*`：inductive bias、attention、routing 或结构不适配；
- `failure.model.capacity.*`：参数、上下文、共享瓶颈或 specialist 容量不足；
- `failure.model.initialization.*`：base checkpoint、初始化或权重身份不兼容；
- `failure.model.trainable_scope.*`：冻结范围、LoRA rank/target/layer 或 adapter 放置不当；
- `failure.model.objective.*`：loss formulation、mask policy、reward 或多目标组合错误；
- `failure.model.regularization.*`：KL、weight decay、dropout 或约束强度不当；
- `failure.model.numerics.*`：静态量化或精度表示导致模型语义受损；
- `failure.model.calibration.*`：confidence、abstention 或 threshold 错配。

只有在数据进入模型前的真实 labels、mask、顺序和身份均通过审计后，才应把
failure 提升到 objective、capacity 或 architecture。不能用“模型没学会”替代
具体边界分析。

### 4.3 训练方案层 `failure.training.*`

- `failure.training.optimizer.*`：optimizer 类型、超参数或状态错误；
- `failure.training.schedule.*`：learning rate、warmup、decay 或阶段切换不当；
- `failure.training.batch.*`：per-device/global batch、accumulation 或 batch composition 错误；
- `failure.training.budget.*`：token/step 预算不足、过训练或资源分配错误；
- `failure.training.order.*`：阶段顺序、curriculum、interleaving 或 warm start 策略不当；
- `failure.training.transfer.*`：catastrophic forgetting、negative transfer、domain interference；
- `failure.training.dynamics.*`：gradient explosion/vanishing、NaN、停滞或训练震荡；
- `failure.training.checkpoint.*`：early stopping、peak 选择或 selection bias；
- `failure.training.stochasticity.*`：seed 敏感、方差过大或不可复现；
- `failure.training.distributed.*`：sampler padding/duplicate、global order、reduction 或
  world-size 语义漂移；
- `failure.training.resume.*`：optimizer/scheduler/RNG 状态或断点恢复不完整；
- `failure.training.precision.*`：AMP、underflow/overflow 或 activation-checkpointing 语义错误。

“训练后旧能力下降”只是 transfer symptom。只有排除数据覆盖、评测、seed 与
checkpoint selection 等解释，并证明新任务更新持续覆盖或破坏旧能力后，才标为
`failure.training.transfer.catastrophic_forgetting`。

### 4.4 Harness / inference 层 `failure.harness.*`

- prompt、system context、memory、retrieval 与 compression；
- inference template、response adapter、parser、decoding、stop condition；
- tool schema、参数校验、error semantics；
- planner、router、retry、workflow 与 skill 组合。

训练 labels 正确但线上使用不同 template，或正确模型输出被 adapter 改坏，应
归到这一层，而不是训练 failure。

### 4.5 Evaluation / verifier 层 `failure.eval.*`

- case coverage、数据泄漏和分布代表性；
- scorer semantics、normalization、parser 与 threshold；
- sandbox / external verifier 集成；
- calibration、false positive / false negative；
- candidate selection、multiple comparison 与统计不确定性；
- evidence completeness、comparison identity 与 hash integrity。

evaluation failure 可以被诊断，但当前 goal 把 eval cases、scorer semantics、
promotion thresholds 和 evidence integrity 等列在 action space 之外。发现问题时
应 fail closed、建立新 goal 或做获准的 operational remediation，不能为了 promotion
静默改评测。

### 4.6 Runtime 与 governance

- `failure.runtime.*`：环境、依赖、硬件、拓扑、orchestration、resume、observability、
  artifact identity；
- `failure.governance.*`：权限、审批、预算、审计、数据策略和历史完整性。

Runtime failure 只有在 run contract 与训练语义未变时才能作为 retry 修复；任何
影响数据、seed、recipe、执行顺序或结果可比性的改变都必须进入新 run 或新 version。

若证据还不能唯一定位上述任何一层，使用
`failure.unknown.localization.unlocalized`，并把下一步定义为 diagnostic probe。
这比为了满足 schema 而猜一个 data/model/training 根因更诚实，也更利于后续统计。

## 5. Intervention taxonomy 与 failure 的关系

当前机器可读的 intervention families 位于 [`taxonomy.json`](taxonomy.json)：

- `model.data.*`：preprocessing、selection、quality、coverage、mixture、replay、curriculum、augmentation；
- `model.target.*`：representation、masking、canonicalization；
- `model.objective.*`：loss weighting、KL、SFT、preference、RL；
- `model.optimization.*`：learning rate、schedule、optimizer、batch、token budget、gradient 与 numerics；
- `model.trainable_scope.*`：method、LoRA rank/targets、layers、specialists；
- `model.architecture.*`：backbone、capacity、attention、context、routing 与 output head；
- `harness.context.*`、`harness.tools.*`、`harness.workflow.*`、`harness.generation.*`；
- `verifier.*`、`runtime.*`、`governance.*`。

这些是“允许改变的旋钮”，不是 failure 分类。一个 failure 可以对应多个候选
intervention；同一个 intervention 也可以修复不同 failure。选择时应优先满足：

1. 能直接切断已证明的机制链；
2. 改动最小，且冻结更多替代解释；
3. 预测与 falsifier 可在预算内观测；
4. 不越过当前 goal 的 allowed prefixes 或 outside-action-space 边界。

`detailed_levers` 已给出 canonical leaf；从 v0003 起必须选择 exact leaf。taxonomy
只声明“有哪些旋钮”，不等于当前 goal 已授权这些旋钮。例如 `model.architecture.*`
虽已进入全局 taxonomy，但当前 `goal-0001` 没有把它列入 allowed prefixes；要执行
architecture intervention，仍需由新 goal 明确开放。

## 6. 典型 crosswalk

| 情况 | 先验证的 first broken invariant | Failure | 候选 intervention |
|---|---|---|---|
| v0002 Code 边界丢失 | source target 正确；live rendered/nonmasked label 首 token 错误 | `failure.data.supervision.template_rendering`，contributor 为 `failure.data.supervision.mask_materialization` | primary `model.target.representation.boundary_tokens`；necessary dependent `model.target.masking.token_eligibility` |
| Catastrophic forgetting | labels/eval 正确；新任务更新后旧能力持续丢失，跨 checkpoint/seed 可复现 | `failure.training.transfer.catastrophic_forgetting` | `model.data.replay.replay_ratio`、`model.data.mixture.domain_weights`、`model.data.curriculum.domain_schedule`、`model.objective.kl.coefficient` 或更小 trainable scope |
| Loss function 不匹配 | actual labels/mask 正确，但 objective 没有编码目标优先级，且梯度贡献与失败一致 | `failure.model.objective.loss_formulation` 或 `failure.model.objective.loss_weighting` | `model.objective.sft.loss_family`、`model.objective.loss_weighting.token_type`、`model.objective.kl.coefficient` 等 |
| Architecture / capacity 不足 | 数据、target、objective 与 optimization 均通过；多预算/seed/合理 recipe 仍出现同一可复现瓶颈 | `failure.model.architecture.*` 或 `failure.model.capacity.*` | `model.architecture.capacity.parameter_scale` 等；也可先用 `model.trainable_scope.*` probe 排除 adaptation scope 瓶颈 |

Loss mask 错误尤其要注意边界：若 mask 在 forward 前把正确 target token 变成
`-100`，first broken invariant 是 target construction / mask alignment；若 mask
正确而优化目标本身不合适，才是 objective failure。

## 7. v0002 case：failure 与 intervention 不同层

v0001 的诊断证据确认：

- 105/105 stored Code targets 以四个空格开头；
- pinned training template 对 assistant content 执行 `strip()`；
- `ignore-empty-think` mask 使 0/105 live nonmasked labels 保留该边界；
- Code eligibility 随训练从 Base 7/8 降到 3/8，随后为 0/8；
- t18000 与 t24000 的 Code 输出已经平台化，继续加 token 或扫 LR 缺乏依据。

因此：

```text
symptom:
  code eligibility collapse / leading-indentation failures

first broken invariant:
  correct source target
  → training-template rendering + loss-mask boundary
  → incorrect actual supervised labels

failure:
  primary     = failure.data.supervision.template_rendering
  contributor = failure.data.supervision.mask_materialization

intervention:
  primary   = model.target.representation.boundary_tokens
  dependent = model.target.masking.token_eligibility
```

dependent lever 是必要的，因为只保留 rendered indentation、却仍把它 mask 为
`-100`，不会改变监督信号。这个 case 不能标为 source-data quality，也没有证据
支持 architecture failure。

## 8. Diagnosis record 的最小字段

从 v0003 起，每个 causal intervention 版本应在执行前把以下结构嵌入
`version.json`；其中 `evidence_refs` 可以指向不可变 diagnosis artifact：

```json
{
  "diagnosis_id": "diagnosis-NNNN",
  "observed_symptoms": [
    {"id": "symptom.capability.regression", "scope": "candidate/checkpoint/domain", "evidence_refs": ["path/to/evidence.json"]}
  ],
  "primary_failure": {
    "id": "failure.training.transfer.catastrophic_forgetting",
    "first_broken_invariant": "expected invariant and observed violation",
    "pipeline_boundary": "input artifact -> output artifact",
    "causal_status": "suspected|supported|causally_confirmed|inconclusive",
    "evidence_refs": ["path/to/evidence.json"]
  },
  "contributing_failures": [],
  "excluded_alternatives": [
    {"id": "failure.eval.scorer.metric_semantics", "reason": "why excluded", "evidence_refs": ["path/to/scorer-audit.json"]}
  ],
  "intervention": {
    "primary_lever": "model.data.replay.replay_ratio",
    "dependent_lever": null,
    "dependent_lever_reason": null,
    "expected_mechanism": "how the change repairs the invariant"
  },
  "prediction": "observable precommitted outcome",
  "falsifier": "observable condition that rejects the hypothesis",
  "frozen_invariants": ["exact variables that remain unchanged"],
  "guardrails": ["metrics that may not regress"]
}
```

最小要求不是字段数量，而是能回答五个问题：看到了什么、最先哪里坏了、证据
是什么、下一轮只改什么、什么结果会证明判断错了。执行后的 observed outcome 和
decision 应追加记录，不能反向改写 precommitted prediction/falsifier。

## 9. v0003+ prospective policy

后续新版本遵循以下规则：

1. 在 GPU 执行前完成 symptom、primary failure、evidence、prediction 和 falsifier；
2. 一个版本只改变一个 primary lever，最多增加一个机制上不可分割的 dependent
   lever，并写明依赖关系；
3. 诊断置信度不足时先运行最小 probe；完整训练不是诊断工具的默认选项；
4. failure taxonomy 不扩大权限，最终仍受 goal、`taxonomy.json`、预算和 frozen
   evaluation 约束；
5. operational retry 必须保持 run contract 与训练语义不变；语义变化创建新版本；
6. promotion 只看冻结 gates 与完整证据，不能因 diagnosis 看起来合理而放宽标准；
7. controller 对 v0003+ 强制 failure/lever exact leaf 与结构化 diagnosis；历史 alias
   只用于读取和查询，不能继续生成新的自由文本 diagnosis code。
8. canonical failure、symptom、lever ID 与 `effective_from_version` 都是 append-only；
   已被完成版本引用后只能增加 deprecation metadata，不能改名、删除、移动或后推
   activation boundary。否则会破坏历史版本的可重验性。

`rsi-v0001` 是 baseline-reset exception，`rsi-v0002` 是已完成的 causal
intervention；这份规则从新建版本开始生效，不追溯要求旧记录满足新字段。

## 10. Append-only 历史边界

以下目录是完成版本的历史账本，不得为应用新 taxonomy 而回写：

- `versions/rsi-v0001/**`
- `versions/rsi-v0002/**`

尤其不能修改旧 run contracts、attempts、operation/retry ledgers、self-hashed
evidence、artifacts 或 metrics。旧记录中的 `training_template_label_construction`、
`evaluation`、`sandbox_integration` 等自由文本可以在新的全局 taxonomy 中声明为
legacy alias，但原字节必须保留。

如果以后需要为历史记录建立统一查询，应新增 append-only crosswalk 或新 evidence
记录，引用旧文件及其 hash；不要在原文件中替换 failure code。完整限制见
[`ARCHIVE-AUDIT-20260813.md`](ARCHIVE-AUDIT-20260813.md)。
