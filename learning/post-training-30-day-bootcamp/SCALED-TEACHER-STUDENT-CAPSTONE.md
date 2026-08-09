# Qwen3.5-4B Policy Capstone + Deferred Teacher–Student Extension

状态：`optional_after_day30`

执行方式：Day 31–42 是无固定日期的后置扩展；前 30 天完成后，按 readiness gate 和预算分阶段执行。未执行本扩展不影响 30-Day Bootcamp 的完成状态。

当前模型决策只冻结到以下范围：

- 活动 policy model ID：`Qwen/Qwen3.5-4B-Base`；
- immutable revision：`pending`，由 Core Day 15 完整下载/hash 验收后冻结；Day 31 只复核并继承该 revision，不得改用滚动 `main` 或猜测 SHA；
- `S0 -> S1 -> S2` 是唯一获准接收 optimizer update 的 policy lineage；
- `teacher_model_id: null`、`teacher_revision: null`、`teacher_branch_status: deferred_unselected`；
- 本文中的 teacher、teacher SFT/RL、teacher trace 和 OPD 都是未激活扩展，不代表已经选择 8B、9B 或任何其他 teacher。

## 当前活动目标

在同一模型、同一 coding 任务和同一冻结评测下，完成一条可审计的 policy post-training 链：

```text
S0 = Qwen/Qwen3.5-4B-Base@<pending immutable revision>
  -> S1 = selected coding SFT checkpoint
  -> S2 = selected direct coding RL/RLVR checkpoint

frozen comparison:
  S0 Base vs S1 coding SFT vs S2 direct coding RL

deferred extension（未激活）:
  T0 -> T1 -> T2 teacher lineage
  S1 -> S3 OPD(T2)
  S1 -> S1d -> S3d recovery
```

当前活动链回答四类问题：

1. **能力**：coding SFT 与 direct coding RL 是否提升可执行任务成功，且不破坏 general guardrails？
2. **算法**：SFT 与 direct RL 各自改变什么训练信号、失败模式和稳定性？
3. **系统**：single/TP2、rollout、weight sync 和 distributed checkpoint 是否数值与状态一致？
4. **成本**：S1/S2 的能力增量是否值得新增训练与 rollout 成本？

若未来通过独立 charter v2 激活 teacher extension，才额外回答 teacher advantage、direct RL vs OPD 和 teacher total-cost 问题。

## Checkpoint graph 与状态

活动链必须保留 S0/S1/S2 三个独立 checkpoint，不允许只留下最终权重；下表同时记录 deferred 节点的状态，但 deferred 节点不对应当前 checkpoint 文件：

| ID | Checkpoint | 作用 |
|---|---|---|
| `S0` | `Qwen/Qwen3.5-4B-Base@<pending immutable revision>` | 活动；训练前 Base baseline |
| `S1` | `S0 + coding SFT` selected | 活动；direct coding RL 起点 |
| `S2` | `S1 + direct coding RL/RLVR` selected | 活动；当前最终 policy candidate |
| `T0/T1/T2` | teacher Base/SFT/RL lineage | `deferred_unselected`；charter v2 前不存在 |
| `S3` | `S1 + OPD(T2)` | `deferred_unselected`；charter v2 前不存在 |
| `S1d/S3d` | teacher-trace cold start / OPD recovery | `deferred_unselected`；不能替代活动链 |
| `S4` | `S3 + short RL` | deferred Stretch |

当前 Core 只要求 `S0/S1/S2`。若未来激活 teacher extension，`S2` 与 `S3` 必须从同一个 exact `S1` hash 开始；cold-start 输出必须继续使用 `S1d/S3d` 命名，并另配 matched direct-RL control 才能做 objective 归因。

## 默认任务：可执行 Coding

默认领域是有 deterministic sandbox/verifier 的 coding。也可以使用 tool calling、数学或其他可验证领域，但 Day 31 必须冻结任务对象、语言/runtime、sandbox 和 scorer，不能在看到训练结果后更换主指标。

Coding E2E 证据链：

```text
prompt + task specification
  -> code/text response
  -> extraction + parse
  -> sandbox compile/run
  -> deterministic tests
  -> final task state / score
```

最低指标：

- extraction/parse 与 compile success；
- public/hidden deterministic test pass rate；
- executable end-to-end task success；
- timeout、runtime error、unsafe sandbox action 和 invalid-output rate；
- pass@1 或预注册 sampling aggregation；
- response/code length 与 reward-hacking slices；
- general/math/tool-use 或既有能力 guardrails。

Primary metric 优先使用 `executable end-to-end task success`，不以“看起来像 JSON”或训练 reward 代替真实任务成功。

## 数据与评测边界

复用 Day 08–10 的 schema、hash、scorer 和逐样本 evidence 方法，不复用旧模型绑定的 rendering/token artifacts。Day 12 的 frozen test **未消费**，但旧 dev 已支持十轮连续选择；未消费 raw IDs 只有在不查看内容、通过脚本化 crosswalk 并创建全新 Qwen3.5 processor/render/comparison manifest 后才可候选复用。

活动链使用四类职责互斥的 prompt manifests；teacher extension 另有两类 deferred manifests：

1. `sft_train`：S1 的 coding supervised demonstrations；
2. `policy_train`：S2 direct coding RL 的 prompt universe；若未来激活 OPD，charter v2 必须显式声明是否复用；
3. `capstone_dev`：可用于 checkpoint selection、error analysis 和一次版本内的诊断；
4. `capstone_frozen`：只在 S0/S1/S2 全部选定后揭盲一次，随后标记 `consumed`；
5. `teacher_advantage_probe`：deferred，仅用于激活后的 T2 vs S1 gate；
6. `trace_cold_start`：deferred，仅用于 S1d offline-KD ablation。

必须按 scenario/template/tool family 做 group-aware split 和 train↔eval exact/near-overlap 检查。仅更换参数值但保留相同任务模板，不算真正的 held-out 泛化。

活动链使用共同 `eval_suite_hash` 对齐 raw task IDs、references、sandbox/scorer 和 aggregation，并让 S0/S1/S2 各自保留 model/processor/render/execution `protocol_hash/comparison_key`。若 capstone policy charter v1 已消费 frozen，未来 teacher extension 必须创建 charter v2 与新的 frozen suite，不能用已经揭盲的 policy-v1 frozen 选择 teacher 或 OPD recipe。

## Deferred Teacher Extension Activation Gate

当前 `teacher_model_id/revision` 均为 `null`。只有先获得用户对 exact teacher model 的独立决定，并在 charter v2 中同时满足以下条件，才可创建 `T0/T1/T2/S3`：

- exact teacher model ID、immutable revision、license、processor/loader 和 training scope 已冻结；
- 相对 `T1` 在预注册 coding dev primary metric 上达到最小有意义提升；
- 相对 `S1` 有明确、逐样本可定位的新能力，不只是更长或更像 rubric 的输出；
- general guardrails、format/error rate 与 reward-hacking audit 通过；
- teacher checkpoint、tokenizer、template、generation config、RL policy version 和 eval evidence 已冻结；
- teacher 与 student 在 Core token-level OPD 中具有完全相同的 tokenizer JSON/hash、vocabulary/token-ID map、added/special tokens 和 tool/chat template；20 条 golden prompts 的 rendered token IDs 必须逐项相同。跨 tokenizer distillation 只能另开 Stretch protocol；
- teacher 能对 student rollout 的确切 prefix/token IDs 返回可复查 log-prob evidence。

不得用“8B”“9B”“同系列”代替 exact teacher 选择。若 teacher advantage 不成立，停止 OPD，不用规模或训练 reward 掩盖无效 teacher。

## 活动 Direct RL 预算与 Optional OPD 公平比较

活动 S2 必须预注册 optimizer updates、trained tokens、unique prompt exposures、rollout-token cap、max response、checkpoint cadence 和总 GPU-hours。只有 charter v2 激活 OPD 后，才要求至少同时报告下表预算：

| 预算口径 | `S2` Direct RL | `S3` OPD |
|---|---|---|
| Student optimizer updates / trained tokens | 匹配 | 匹配 |
| Unique prompt IDs/exposures、max response | 匹配 | 匹配 |
| Student-generated rollout-token cap | 匹配 | 匹配 |
| 总 GPU-hours | actor/rollout/ref | student/rollout/teacher scoring |

激活后的 teacher 训练成本单独报告：

- `one-off`：只服务这一个 student 时计入总成本；
- `amortized`：若 teacher 服务多个 student/domain，写清摊销假设；
- 不得从 OPD 成本表中静默删除 teacher preparation。

活动 S0/S1/S2 使用同一份 capstone policy charter v1 的 `capstone_dev/frozen` manifest、sandbox/scorer 和预注册 selection rule。若 teacher extension 后续激活，则使用 charter v2 冻结的新 dev/frozen suite；不同 objective 本身是 intended treatment，其余数据版本、起点、prompt universe/exposure、最大 response、student update budget、rollout-token cap、checkpoint cadence 和评测必须固定。Cold-start recovery 的额外 trained tokens 与 teacher trace compute 另计，不能混入 matched claim。

## TP 与运行时设计

### Correctness reference

先在 exact `Qwen/Qwen3.5-4B-Base@<immutable revision>` 上比较：

```text
single GPU
vs
TP=2
```

检查初始 logits/loss、前 5–20 steps、gradient norm、sample IDs、global batch、checkpoint merge/load 和 deterministic eval。小模型 TP 变慢是允许结果，但数值或状态分叉必须解释。

### Qwen3.5-4B capacity run

Day 32 同时核算 full-parameter、LoRA 和 QLoRA 的 weights、gradients、optimizer、activation、rollout copy、KV/cache 与 temporary peak，但不在文档中替用户选择 recipe。是否使用 TP2、FSDP/ZeRO、offload 或 PEFT 必须由 measured smoke、可恢复性和预算共同决定；不能把“启用了 TP”写成“必须使用 TP”。

活动 direct RL 至少区分：

- train policy/optimizer workers；
- policy rollout workers；
- environment/reward workers；
- checkpoint/weight-sync path。

`frozen teacher scoring workers` 只在 charter v2 激活 OPD 后加入 placement。

每个资源池记录 `DP/TP/PP`、replica 数、GPU bundle、模型版本和 producer/consumer。开卡前先通过 placement dry check，禁止依赖未解析的默认 placement。

本文不预选 GPU topology。Day 32/37 必须分别以 single/TP2 smoke、目标 sequence/completion length、full/LoRA/QLoRA 状态量和 rollout placement 冻结实际资源。未选择 teacher 时不得预留、租用或写死 teacher GPU bundle。

## 框架策略

- SFT/TP：为 Qwen3.5-4B 复用 Day 18 已验证的 Megatron/ms-swift 路线或当时 pinned 的正式支持路径；Day 31 必须重新核对 release/SHA/container 与 multimodal loader/module coverage；
- Direct coding RL：复用 Day 24–29 的 verifier、trajectory、replay 和 weight-version contract；
- OPD：仅在 charter v2 激活后，从官方文档选择并 pin 支持 teacher resource pool、student on-policy rollout 和逐 token distillation evidence 的实现；
- TRL experimental trainer 只用于小型 correctness/API 对照；不作为 TP Capstone Core 的稳定承诺；
- veRL `v0.8.0` 可作为 10–30 step OPD systems migration Stretch，用于验证独立 teacher resource pool/actor/rollout 边界，不重复完整 Core 训练；
- 不在同一个对照中同时更换 model family、tokenizer、framework、objective 和数据。

## Day 31–42 导航

执行时先复制并填写以下模板，不直接覆盖模板本体：

- [Capstone charter template](templates/capstone-charter.yaml)
- [Checkpoint promotion manifest template](templates/checkpoint-promotion-manifest.yaml)
- [OPD trajectory schema v2](templates/opd-trajectory-schema-v2.json)

### Week 5 — Policy Freeze/Capacity + Deferred Teacher Templates（Day 31–36）

- [Day 31 — Qwen3.5-4B Charter、Coding Eval 与版本冻结](day-31-capstone-charter-eval/README.md)（active）
- [Day 32 — Qwen3.5-4B Single/TP2 Parity 与 Capacity Plan](day-32-tp-parity-scale-accounting/README.md)（active）
- [Day 33 — Deferred Teacher TP SFT One-step/Resume Gate](day-33-8b-tp-sft-gate/README.md)（deferred）
- [Day 34 — Deferred Teacher Controlled SFT 与 T1 Selection](day-34-8b-sft-selection/README.md)（deferred）
- [Day 35 — Deferred Teacher Domain RL Readiness](day-35-domain-rl-teacher-readiness/README.md)（deferred）
- [Day 36 — Deferred Teacher Domain RL 与 T2 Freeze](day-36-8b-domain-rl-teacher-freeze/README.md)（deferred）

### Week 6 — Direct Coding RL + Deferred OPD + Final Comparison（Day 37–42）

- [Day 37 — Qwen3.5-4B Common Anchor 与 Direct Coding RL](day-37-4b-direct-rl-control/README.md)（active）
- [Day 38 — Deferred Teacher-trace Cold Start](day-38-teacher-trace-cold-start/README.md)（deferred）
- [Day 39 — Deferred OPD One-update 与 Replay Gate](day-39-opd-one-update-replay/README.md)（deferred）
- [Day 40 — Deferred Controlled OPD 与 S3 Selection](day-40-opd-controlled-run/README.md)（deferred）
- [Day 41 — S0/S1/S2 Matched Eval、Frozen Confirmation 与 Cost](day-41-matched-eval-cost/README.md)（active）
- [Day 42 — S1/S2 Clean Reproduction 与 Capstone Report](day-42-capstone-clean-reproduction/README.md)（active）

## 当前活动 Core 验收

- [ ] `S0/S1/S2` lineage、immutable model/processor/template/data/code hashes 完整。
- [ ] Qwen3.5-4B single/TP2 parity 与 chosen full/LoRA/QLoRA runtime 有 per-rank evidence；recipe 决策可追溯。
- [ ] `S1` 通过 coding SFT selection gate，`S2` 通过 direct coding RL reward-hacking、general guardrail 和 checkpoint gate。
- [ ] `S0/S1/S2` 只在候选选定后运行同一 capstone frozen confirmation，并写 consumption record。
- [ ] 报告同时包含能力、失败 slice、GPU-hours、tokens、吞吐、显存和 rollout cost；允许结论为 `inconclusive`。
- [ ] 从干净环境至少重放一条 direct-RL batch、恢复一个训练 checkpoint，并重算一条 coding E2E score。
- [ ] `teacher_model_id/revision` 仍为 `null`，所有 T*/S3* 节点保持 `deferred_unselected`，除非存在独立批准的 charter v2。

## Deferred Extension 验收（仅 charter v2 激活后）

- [ ] `T0/T1/T2/S3` lineage、teacher advantage、token alignment 与 teacher immutability 通过。
- [ ] `S2/S3` 从同一 S1 分叉且 student budgets 可比；S1d/S3d 单独命名。
- [ ] OPD 保存 policy rollout、teacher/student log-prob、loss mask、versions 和 replay evidence。
- [ ] teacher preparation/scoring 成本没有从 total-cost 账本中删除。

## Stop Conditions

任一条件触发即停在当前 gate，不因已租 GPU 而扩大：

- S0 immutable revision、processor/loader/modality 或 training-scope contract 未冻结；
- train/eval overlap 未处理；
- TP parity 无法解释；
- checkpoint 无法在新进程恢复或转换后 hash/score 不一致；
- reward 可被格式/长度投机；
- direct coding RL reward 无法从 sandbox raw evidence 重算；
- GPU-hour 上限已达但关键证据仍缺失。

Deferred teacher extension 另加：teacher 未被用户明确选择、teacher 没有可测 advantage、teacher/student token alignment 不可审计，或 OPD loss 有信号但 frozen capability 不升时立即停止。
