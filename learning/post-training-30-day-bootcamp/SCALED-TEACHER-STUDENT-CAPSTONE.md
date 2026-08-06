# Scaled Teacher–Student Post-Training Capstone

状态：`optional_after_day30`

执行方式：Day 31–42 是无固定日期的后置扩展；前 30 天完成后，按 readiness gate 和预算分阶段执行。未执行本扩展不影响 30-Day Bootcamp 的完成状态。

## 目标

在同一模型家族、同一领域任务和同一冻结评测下，完成一条可审计的规模迁移链：

```text
8B Base
  -> TP full-parameter SFT
  -> domain RL/RLVR
  -> frozen teacher

<=4B Base
  -> common student SFT anchor
  -> A: direct domain RL
  -> B: on-policy distillation
  -> Optional recovery: teacher-trace cold start -> on-policy distillation

frozen comparison:
  8B teacher vs <=4B direct-RL vs <=4B OPD
```

本扩展同时回答四类问题：

1. **能力**：8B teacher 的领域能力能否被 <=4B student 迁移，且不破坏 general guardrails？
2. **算法**：在共同 student 起点下，direct RL 与 OPD 的收益、失败模式和稳定性有何差异？
3. **系统**：单卡 reference、TP training、rollout、teacher scoring、weight sync 和 distributed checkpoint 是否数值与状态一致？
4. **成本**：OPD 的 student 收益是否值得 teacher inference、额外 rollout 和基础 teacher 训练成本？

## 不是简单把阶段串起来

Capstone 必须保留以下独立 checkpoint，不允许只留下最终权重：

| ID | Checkpoint | 作用 |
|---|---|---|
| `T0` | 8B Base | teacher 训练前基线 |
| `T1` | 8B SFT selected | domain RL 起点 |
| `T2` | 8B SFT + domain RL selected | 冻结 teacher |
| `S0` | <=4B Base | student 训练前基线 |
| `S1` | <=4B SFT selected | direct RL 与 OPD 的共同起点 |
| `S2` | `S1 + direct RL` | 小模型直接强化 control |
| `S3` | `S1 + OPD(T2)` | 与 S2 共享起点的 teacher–student distillation candidate |
| `S1d/S3d` | `S1 + teacher-trace cold start (+ OPD)` | 分布不兼容时的恢复/ablation 路线，不替代 Core |
| `S4` | `S3 + short RL` | 可选 Stretch；不能替代 `S2 vs S3` Core 对照 |

`S2` 与 Core `S3` 必须从同一个 exact `S1` checkpoint hash 开始。若 cold start 修改了 `S1`，其输出另记为 `S1d`，后续 OPD 结果记为 `S3d`；`S2 vs S3d` 只能解释“direct-RL pipeline vs cold-start+OPD pipeline”，不能声称是纯 OPD objective 的因果差异。若要做算法归因，必须另跑 `S1d + direct RL` matched control。

## 默认任务：可执行 Tool Calling

默认领域是有 deterministic environment/verifier 的 tool calling。也可以替换为数学、代码或其他可验证领域，但 Day 31 必须冻结任务对象和 scorer，不能在看到训练结果后更换主指标。

Tool-call E2E 证据链：

```text
prompt + available tool schema
  -> call / no-call decision
  -> tool name
  -> arguments parse + schema validation
  -> environment execution
  -> observation handling
  -> final task state / answer
```

最低指标：

- call/no-call decision accuracy；
- tool-name accuracy；
- parse success 与 schema-valid rate；
- argument semantic accuracy；
- execution success 与 end-to-end task success；
- hallucinated/unnecessary call rate；
- multi-turn recovery rate、response length、timeout/error rate；
- general/math/code 或既有能力 guardrails。

Primary metric 优先使用 `executable end-to-end task success`，不以“看起来像 JSON”或训练 reward 代替真实任务成功。

## 数据与评测边界

复用 Day 08–10 的 schema、hash、rendering、scorer 和逐样本 evidence 方法，但不自动复用已被 Day 12 消费的 frozen samples。

Capstone 使用六类职责互斥的 prompt manifests：

1. `sft_train`：T1/S1 的 supervised demonstrations；
2. `policy_train`：S2 direct RL 与 S3 OPD 共享的 prompt universe；
3. `teacher_advantage_probe`：只用于判断 T2 是否比 S1 提供新能力，不进入 dev/frozen；
4. `trace_cold_start`：只用于 S1d offline-KD ablation，不能和 Core policy prompts 静默混用；
5. `capstone_dev`：可用于 checkpoint selection、error analysis 和一次版本内的诊断；
6. `capstone_frozen`：只在所有候选选定后揭盲一次，随后标记 `consumed`。

必须按 scenario/template/tool family 做 group-aware split 和 train↔eval exact/near-overlap 检查。仅更换参数值但保留相同任务模板，不算真正的 held-out 泛化。

跨模型规模比较使用共同 `eval_suite_hash` 对齐 raw task IDs、references、environment/scorer 和 aggregation；T0/T1/T2 与 S0/S1/S2/S3 仍各自保留 model/render/execution `protocol_hash/comparison_key`。因此 8B↔4B 是同一 E2E 任务尺子上的 paired outcome comparison，不伪装成“只改变 checkpoint weights”的严格同协议比较。

## Teacher Readiness Gate

只有同时满足以下条件，`T2` 才能成为 OPD teacher：

- 相对 `T1` 在预注册 domain dev primary metric 上达到最小有意义提升；
- 相对 `S1` 有明确、逐样本可定位的新能力，不只是更长或更像 rubric 的输出；
- general guardrails、format/error rate 与 reward-hacking audit 通过；
- teacher checkpoint、tokenizer、template、generation config、RL policy version 和 eval evidence 已冻结；
- teacher 与 student 在 Core token-level OPD 中具有完全相同的 tokenizer JSON/hash、vocabulary/token-ID map、added/special tokens 和 tool/chat template；20 条 golden prompts 的 rendered token IDs 必须逐项相同。跨 tokenizer distillation 只能另开 Stretch protocol；
- teacher 能对 student rollout 的确切 prefix/token IDs 返回可复查 log-prob evidence。

若 teacher advantage 不成立，停止 OPD，不用规模或训练 reward 掩盖无效 teacher。

## Direct RL 与 OPD 的公平比较

至少同时报告三套预算，不能只选择对 OPD 有利的一种：

| 预算口径 | `S2` Direct RL | `S3` OPD |
|---|---|---|
| Student optimizer updates / trained tokens | 匹配 | 匹配 |
| Unique prompt IDs/exposures、max response | 匹配 | 匹配 |
| Student-generated rollout-token cap | 匹配 | 匹配 |
| 总 GPU-hours | actor/rollout/ref | student/rollout/teacher scoring |

8B teacher 的训练成本单独报告：

- `one-off`：只服务这一个 student 时计入总成本；
- `amortized`：若 teacher 服务多个 student/domain，写清摊销假设；
- 不得从 OPD 成本表中静默删除 teacher preparation。

能力比较使用同一 `capstone_dev/frozen` manifest、相同 environment/scorer 和预注册 selection rule。不同 objective 本身是 intended treatment；其余数据版本、起点、prompt universe/exposure、最大 response、student update budget、rollout-token cap、checkpoint cadence 和评测必须固定。Cold-start recovery 的额外 trained tokens 与 teacher trace compute 另计，不能混入 Core matched claim。

## TP 与运行时设计

### Correctness reference

先在一个能单卡运行的 <=4B checkpoint 上比较：

```text
single GPU
vs
TP=2
```

检查初始 logits/loss、前 5–20 steps、gradient norm、sample IDs、global batch、checkpoint merge/load 和 deterministic eval。小模型 TP 变慢是允许结果，但数值或状态分叉必须解释。

### Genuine scale run

8B teacher SFT 使用 full-parameter TP2/TP4 作为默认目标，使 TP 承担真实 model-state/activation pressure。若最终采用 LoRA/QLoRA、FSDP 或 distributed optimizer，必须分别说明实际解决容量问题的机制；不能把“启用了 TP”写成“必须使用 TP”。

RL/OPD 至少区分：

- train policy/optimizer workers；
- student rollout workers；
- frozen teacher scoring workers；
- environment/reward workers；
- checkpoint/weight-sync path。

每个资源池记录 `DP/TP/PP`、replica 数、GPU bundle、模型版本和 producer/consumer。开卡前先通过 placement dry check，禁止依赖未解析的默认 placement。

个人单机的初始 planning topology（最终必须由 smoke 覆盖）是：8B SFT 使用同机 2×80GB TP2；8B RL 使用同机 4×80GB（learner TP2 + rollout TP2）；4B direct RL 使用 2×80GB（learner + rollout）；4B OPD 使用 4×80GB（student learner + student rollout + frozen teacher TP2）。2-GPU 串行 OPD 只在每个 batch 都由当前 student 生成、update 前完成 teacher scoring 且 `policy_lag <= 1` 时成立。

## 框架策略

- SFT/TP：复用 Day 18 已验证的 Megatron/ms-swift 路线或当时 pinned 的正式支持路径；当前审计起点是 ms-swift `v4.4.2` 的正式 distillation/Megatron examples，但 Day 31 必须重新核对 release/SHA/container；
- Domain RL：复用 Day 24–29 的 verifier、trajectory、replay 和 weight-version contract；
- OPD：Day 31 运行时从官方文档选择并 pin 支持 teacher resource pool、student on-policy rollout 和逐 token distillation evidence 的实现；
- TRL experimental trainer 只用于小型 correctness/API 对照；不作为 TP Capstone Core 的稳定承诺；
- veRL `v0.8.0` 可作为 10–30 step OPD systems migration Stretch，用于验证独立 teacher resource pool/actor/rollout 边界，不重复完整 Core 训练；
- 不在同一个对照中同时更换 model family、tokenizer、framework、objective 和数据。

## Day 31–42 导航

执行时先复制并填写以下模板，不直接覆盖模板本体：

- [Capstone charter template](templates/capstone-charter.yaml)
- [Checkpoint promotion manifest template](templates/checkpoint-promotion-manifest.yaml)
- [OPD trajectory schema v2](templates/opd-trajectory-schema-v2.json)

### Week 5 — Scale-up Teacher（Day 31–36）

- [Day 31 — Capstone Charter、Domain Eval 与版本冻结](day-31-capstone-charter-eval/README.md)
- [Day 32 — 4B Single/TP2 Parity 与 8B Capacity Plan](day-32-tp-parity-scale-accounting/README.md)
- [Day 33 — 8B TP SFT One-step/Resume Gate](day-33-8b-tp-sft-gate/README.md)
- [Day 34 — 8B Controlled SFT 与 T1 Selection](day-34-8b-sft-selection/README.md)
- [Day 35 — Domain RL Contract 与 Teacher Readiness](day-35-domain-rl-teacher-readiness/README.md)
- [Day 36 — 8B Domain RL、T2 Selection 与 Teacher Candidate Freeze](day-36-8b-domain-rl-teacher-freeze/README.md)

### Week 6 — Student Controls、OPD 与 Final Comparison（Day 37–42）

- [Day 37 — 4B Common Anchor 与 Direct-RL Control](day-37-4b-direct-rl-control/README.md)
- [Day 38 — Teacher-trace Cold Start 与 Distillation Data Audit](day-38-teacher-trace-cold-start/README.md)
- [Day 39 — OPD One-update、Teacher Scoring 与 Replay Gate](day-39-opd-one-update-replay/README.md)
- [Day 40 — Controlled OPD Run 与 S3 Selection](day-40-opd-controlled-run/README.md)
- [Day 41 — Matched Eval、Frozen Confirmation 与 Cost Accounting](day-41-matched-eval-cost/README.md)
- [Day 42 — Clean Reproduction、Failure Review 与 Capstone Report](day-42-capstone-clean-reproduction/README.md)

## 最终 Core 验收

- [ ] `T0/T1/T2/S0/S1/S2/S3` lineage 与不可变 hashes 完整；`S1d/S3d` 等可选节点明确标记。
- [ ] 4B single/TP2 parity 与 8B genuine TP runtime 都有 per-rank evidence。
- [ ] `T2` 通过 teacher advantage、reward hacking、general guardrail 和 frozen checkpoint gate。
- [ ] `S2` 与 Core `S3` 从共同 `S1` 起点出发，student budget 和 eval protocol 可比；cold-start 路线单独标识。
- [ ] OPD 保存 student rollout、teacher/student log-prob、loss mask、policy/teacher version 和 replay evidence。
- [ ] `T0/T1/T2/S0/S1/S2/S3` 只在全部候选选定后运行同一 capstone frozen confirmation，并写 consumption record；recovery checkpoints按预注册选择加入。
- [ ] 报告同时包含能力、失败 slice、GPU-hours、tokens、吞吐、显存和 teacher cost；允许结论为 `inconclusive`。
- [ ] 从干净环境至少重放一条 OPD batch、恢复一个 distributed checkpoint，并重算一条 E2E score。

## Stop Conditions

任一条件触发即停在当前 gate，不因已租 GPU 而扩大：

- teacher 没有可测 advantage；
- train/eval overlap 未处理；
- TP parity 无法解释；
- checkpoint 无法在新进程恢复或转换后 hash/score 不一致；
- reward 可被格式/长度投机；
- teacher/student token alignment 不可审计；
- OPD loss 有信号但 frozen capability 不升或 guardrail 失败；
- GPU-hour 上限已达但关键证据仍缺失。
