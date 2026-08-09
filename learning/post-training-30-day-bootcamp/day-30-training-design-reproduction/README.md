# Day 30 — Qwen3.5 v2 Clean Reproduction、Design 与综合口述

日期：`2026-08-25`
状态：`not_started`
强度：4–5 小时

## 主要目标

把一个月收敛为一份可审计的 post-training 设计，并从干净环境复现一条 Qwen3.5 v2 证据链：

```text
Day 15 frozen Qwen3.5 Base revision
  -> controlled LoRA coding SFT
  -> selected SFT anchor
  -> DPO 或 coding GRPO smoke
  -> checkpoint save/reload
  -> v2 held-out evidence
```

Day 01–12 是已完成的 v1/0.6B 学习历史，必须在 evidence index 中保留原 config、checkpoint、结果与日期，但其 token cache、checkpoint、baseline 和指标不能充当 v2 链的一部分。毕业作业检验训练知识与实验判断，不建设 auto-train/auto-harness 控制面。

## v1 / v2 Evidence Boundary

| Lineage | 用途 | 允许进入 Day 30 active reproduction 吗？ |
|---|---|---|
| v1：Day 01–12 0.6B artifacts | 历史学习证据、方法对照、踩坑记录 | 否；只读索引 |
| v2 Base：Day 15 冻结的 exact Qwen3.5 revision/hash | 新 baseline 与所有 v2 descendants 的唯一根 | 是 |
| v2 SFT：Day 16 candidates / Day 21 promotion criteria | DPO/GRPO parent 与 clean reproduction anchor | 是，必须可追溯到 v2 Base |
| v2 DPO：Day 23 | Preference branch | 是，仅从 selected SFT anchor 派生 |
| v2 GRPO：Day 25；可选 Day 29 verified slime rerun | Coding online RL branch | 是，仅从 selected SFT anchor 派生 |

禁止把 v1 checkpoint 改名、合并进 v2，或用跨 lineage 指标差异宣称训练收益。

## 理论（35–45 分钟）

精读清单：[Day 30 — Post-training design review](../SCALING-BOOK-READING-GUIDE.md#day-30)。

- SFT、DPO、online GRPO 分别解决什么数据/目标问题，何时不该使用后者。
- Processor/template、data contract、optimizer/schedule、resume、selection、reward/verifier、on-policy 和 weight-version 如何连接。
- 对每个结论标记 `measured / inferred / unknown / next experiment`。
- 先选择本日 downstream branch：优先复现本月证据最完整且资源已验证的 DPO 或 GRPO；选择理由和未选分支的 blocker 必须写入报告。

## Clean-room Preflight（45–60 分钟）

- 在新 shell/干净镜像中只根据保存的 README、lockfile、manifest 和 config 建环境，不读取旧 shell history。
- 解析 Day 15 冻结的 exact Base revision/hash、processor/template revisions、text-only coding modality/freeze policy 与 retokenized v2 manifest。
- 运行 Day 15/17/22/24 的关键 validators/tests，验证 dataset、split、sandbox、reward 和 checkpoint checksums。
- 检查启动脚本能解析输出目录、`max_steps`、save/reload、resume 和 v2 eval config；任何隐式参数先补回 resolved report/config 再启动。
- 使用 Day 15–29 已实测的显存/topology/length 上限；不得默认假设 1×H100 可容纳所选分支。

任一上游 gate 未通过时，只复现到最深的有效 v2 节点并记录 blocker；不使用 v1、Base-only downstream policy 或其他模型补齐图示。

## 训练 / 实验：v2 Clean Reproduction（90–120 分钟）

### Stage A — Base → controlled LoRA coding SFT

1. 从 Day 15 exact Base revision 和 retokenized v2 coding manifest 启动 5–10 step controlled LoRA SFT smoke。
2. 固定 processor/template、trainable module set、ViT/aligner freeze、batch/token budget、optimizer/LR/warmup、seed 和 max lengths。
3. 保存 adapter/resumable state、resolved config、sample IDs、首步/末步 metrics 和 memory snapshot；在新进程 reload。
4. 用 Day 21 的 v2 selection contract 在 smoke candidates 中选择一个 reproduction SFT anchor；只读 v2 dev evidence，v2 confirmation 按预注册规则处理，旧 v1 frozen test 不进入 active chain。

### Stage B — selected SFT anchor → DPO 或 GRPO

- **DPO branch**：严格以 Stage A selected SFT anchor 同时初始化 policy/reference 的逻辑身份，加载 Day 22 retokenized coding preference pairs，完成 5–10 step smoke，保存逐 pair margin/loss 与 adapter/checkpoint。
- **GRPO branch**：严格以 Stage A selected SFT anchor 初始化 policy，加载 Day 24 coding trajectory/reward/sandbox contract，沿用 Day 25 已验证 runtime/topology/length caps，完成 rollout → reward → 1 update → weight sync → next-version rollout。
- 只有 Day 26 为 `go_day29` 且 Day 29 gates 已通过时，slime 才可作为 GRPO clean reproduction runtime；否则使用 Day 25 已验证的 ms-swift path 或明确记录 GRPO branch blocked。

完成所选 branch 后，从新进程 reload 最终 checkpoint/adapter，运行预注册的 v2 held-out/dev smoke。对比原 run 与 clean run 的 sample IDs、hashes、resolved config、首步 metrics 和输出，记录 reproducibility 等级；不要求浮点逐 bit 相同，但必须解释差异来源和 acceptance threshold。

## Post-training Design Doc（60–75 分钟）

为实际 Qwen3.5-4B coding 任务写设计，包含：

- v1/v2 lineage 边界、exact Base revision、processor/template、modality/freeze 与数据 provenance/split。
- SFT/preference/online trajectories contract，以及选择 SFT、DPO 或 GRPO 的理由和停止条件。
- optimizer/LR/warmup/batch/token budget、checkpoint/resume 与 reproducibility。
- reward/verifier/sandbox 版本、failure semantics、hacking guardrails。
- checkpoint selection、v2 frozen held-out、slice/CI 和 promotion/inconclusive policy。
- 实测资源、最小 pilot、failure-injection/rollback runbook 与仍未验证的兼容性边界。

不得加入 scheduler、自动控制面或 auto-harness 产品设计。30B dense/MoE capacity 只能作为 Stretch appendix，标明估算和必须 benchmark 的未知量。

## 资源与租卡

- 只使用 Day 15–29 memory reports 已验证的最小资源/topology；DPO 与 GRPO 分别引用各自实测配置。
- 若所选链在可用预算内没有已验证 topology，正确结果是 resource-blocked，不临时缩容或切换模型。
- 开卡前冻结 GPU-hour 上限，完成 smoke/eval/evidence sync 后立即关机。
- Design、evidence index 与口述为 CPU only；不为 30B appendix 或未验证 8×H100 Stretch 追加租卡。

## 综合口述（30–40 分钟）

录制一次 15 分钟说明：

1. v1 历史为何保留，但不能进入 v2 lineage。
2. Qwen3.5 数据从 raw sample 经 processor/template 到 loss token。
3. Base → LoRA SFT → selected anchor → DPO/GRPO 的身份与 hash 如何连接。
4. Packing、batch、LR/warmup、exact resume 和梯度稳定性证据。
5. Eval 如何选择 checkpoint 而不泄漏，reward/sandbox 如何重放。
6. Online RL 中 rollout/train/weight-version 的闭环，以及一次真实失败如何归因。

再接受反问并修正设计中没有 evidence 的断言。

## Evidence-first 产物

- `../artifacts/reports/post-training-design.md`
- `../artifacts/reports/clean-reproduction-qwen35-v2.md`
- v2 Base/SFT/DPO-or-GRPO lineage manifest 与 hashes
- v1 historical evidence index（只读引用，不复制为 v2）
- 15 分钟口述录音/提纲与 evidence index
- 可选 `30b-capacity-stretch-appendix.md`
- `../artifacts/reports/phase1-capstone-handoff.md`（只列接口/readiness，不要求执行 Capstone）

## 最终验收

- [ ] Clean run 从 Day 15 exact Qwen3.5 Base 到 selected SFT anchor，再到一个 DPO/GRPO branch、save/reload 和 v2 eval，无隐式手工步骤。
- [ ] Active artifacts 的 revision/hash/processor/template/data parentage 连续，Day 01–12 v1 仅以历史 evidence index 保留。
- [ ] 若链条 blocked，报告停在最深有效 v2 节点并给出真实 blocker，没有 fallback 或虚构成功。
- [ ] 能解释 Day 15–29 每个关键结果的 config/log/sample 证据。
- [ ] Post-training design 覆盖数据、训练、恢复、Eval、RL、资源和故障处理。
- [ ] 完成 [`../PROGRESS.md`](../PROGRESS.md)，并列出五个仍需用实验回答的问题。

## Optional Capstone Handoff

Day 30 仍是 30-Day Core final，不因 Day 31–42 未执行而降级。额外生成 `phase1-capstone-handoff.md`，只汇总：

- v1/v2 lineage boundary、可复用 eval/scorer/consumption schema 与需新建的 capstone eval v2；
- Qwen3.5 Base/processor/template lock、selected SFT promotion manifest 与 inference/resumable 两类路径；
- length/packing、optimizer、resume state inventory 的 measured defaults 和禁止外推边界；
- TP parity launcher、checkpoint conversion、trajectory/reward/replay/weight-version contracts；
- pinned environment/container、known-good commands、evidence index；
- 尚未满足的 model/data/license/cache/topology/framework compatibility 和预算项。

该 handoff 是后续阶段的 readiness 输入，不预先宣称 OPD framework、teacher checkpoint 或 capstone dataset 已存在。

## Final Log

### v1 historical index / v2 active lineage

### 实际复现链与 branch 选择

### 最强证据

### 最大缺口 / blockers

### 五个下一步问题

1.
2.
3.
4.
5.
