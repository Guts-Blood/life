# Daily Learning Roadmaps

这里存放可直接提交到 Git 的每日学习路线图。SVG 是可编辑源文件，同目录 PNG 用于 Codex 手机端和其他不支持 SVG 的预览环境。

修改 SVG 后，在 bootcamp 根目录执行下面的命令。脚本会重新生成并覆盖全部同名 PNG：

```bash
node artifacts/render_svg_previews.cjs
```

Day 13–22 的固定数据版路线图可先统一重建，再渲染 PNG：

```bash
node artifacts/scripts/generate_day13_22_roadmaps.cjs
node artifacts/render_svg_previews.cjs
```

命名约定：

```text
day-XX-<topic>-roadmap.svg
```

当前文件：

- Day 01：[PNG 预览](./day-01-roofline-foundations-roadmap.png) · [SVG 源文件](./day-01-roofline-foundations-roadmap.svg)——从环境基线、矩阵乘法 accounting 到 Roofline、tokens/s 与训练显存的基础地图。
- Day 02：[PNG 预览](../../../day-02-transformer-accounting/DAY-02-LEARNING-ROADMAP.png) · [SVG 源文件](../../../day-02-transformer-accounting/DAY-02-LEARNING-ROADMAP.svg)——从 Transformer 参数量、激活量、FLOPs 到训练容量估算。
- Day 03：[主图 PNG](./day-03-roofline-h100-roadmap.png) · [主图 SVG](./day-03-roofline-h100-roadmap.svg) · [数值例题 PNG](./day-03-matmul-roofline-worked-example.png) · [数值例题 SVG](./day-03-matmul-roofline-worked-example.svg)——从 matmul accounting、H100 Roofline、local tokens 到 MFU scenario 与 profiler 校准；例题用同一 GEMM 的 `N=128/256` 算出 knee 两侧的 FLOPs、最低 HBM bytes 与理想时间下界。
- Day 04：[主图 PNG](./day-04-parallelism-roadmap.png) · [主图 SVG](./day-04-parallelism-roadmap.svg) · [故障证据 PNG](./day-04-distributed-failure-evidence-worked-examples.png) · [故障证据 SVG](./day-04-distributed-failure-evidence-worked-examples.svg)——从 sharded tensor、collective、DP/FSDP/TP/PP 与 rank groups，到 OOM、hang、checkpoint mismatch 的证据化诊断；配套图补出 perturbation、首个 collective divergence 与拓扑变化 reshard 三个 worked examples。
- Day 05：[PNG 预览](../day05-training-lifecycle-roadmap.png) · [SVG 源文件](../day05-training-lifecycle-roadmap.svg)——从数据边界、forward/backward、optimizer/checkpoint 到 ms-swift、DeepSpeed、NCCL 与 CUDA 的职责分层。
- Day 06：[主图 PNG](./day-06-posttraining-scaling-decision-roadmap.png) · [主图 SVG](./day-06-posttraining-scaling-decision-roadmap.svg) · [阶段可行性例题 PNG](./day-06-stage-feasibility-worked-example.png) · [阶段可行性例题 SVG](./day-06-stage-feasibility-worked-example.svg)——把 SFT、DPO、RLVR 的阶段价值与 memory、compute、time/cost feasibility、扩展和停止证据放进同一张决策图；例题从 7B policy 展开角色总账、逻辑 FLOPs、rollout KV 与 2×H100 placement。
- Day 07：周复盘与 Gate Review；复用 Day 01–06 的图和证据，不单独制作 SVG。
- Day 08：[PNG 预览](./day-08-sft-data-contract-roadmap.png) · [SVG 源文件](./day-08-sft-data-contract-roadmap.svg)——从 raw schema、冻结 Chat Template、role-aware labels 与 causal shift，到三种 mask 分工、逐 token 审计和两阶段 validator。
- Day 09：[流水线 PNG](../day09-pipeline.png) · [流水线 SVG](../day09-pipeline.svg) · [Mixture findings PNG](../day09-mixture-findings.png) · [Mixture findings SVG](../day09-mixture-findings.svg)——流水线图说明数据治理状态机；findings 图把三种 mixture 分母、synthetic exposure、等监督预算下的 workload 差异和证据边界展开为可核验结论。
- Day 10：[PNG 预览](../day10-frozen-eval-baseline.png) · [SVG 源文件](../day10-frozen-eval-baseline.svg)——从冻结 suite、deterministic Base generation 与 E2B code scoring，到四-slice failure taxonomy、comparison identity 和 Day 12 gate。
- Day 11：[PNG 预览](../day11-tiny-overfit-retrospective.png) · [SVG 源文件](../day11-tiny-overfit-retrospective.svg)——总结 18 条样本 tiny overfit、assistant-only mask、checkpoint resume 精确一致性，以及 teacher-forced evaluation 与 autoregressive generation 的边界。
- Day 12：[PNG 预览](../day12-checkpoint-score-summary.png) · [SVG 源文件](../day12-checkpoint-score-summary.svg)——汇总 Base、原始 A/B 三阶段与 recovery C–L 的 checkpoint 分数、总分排名和 math/code 验收门槛，明确“最高总分”不等于“通过全部 gate”。
- Day 13：[PNG 预览](./day-13-qwen35-lineage-migration-roadmap.png) · [SVG 源文件](./day-13-qwen35-lineage-migration-roadmap.svg)——重建 Qwen3 v1 到 Qwen3.5 v2 的 lineage 迁移边界：方法与原始样本可复用，token、预测、checkpoint 和能力结论必须在新 lineage 中重建；原 Day 13 checklist 未完成，因此图中明确标为 reconstructed roadmap。
- Day 14：[PNG 预览](./day-14-week2-evidence-readiness-roadmap.png) · [SVG 源文件](./day-14-week2-evidence-readiness-roadmap.svg)——把 Day 08–12 的数据契约、mixture、frozen eval、tiny overfit 和 recovery 证据串成 Week 2 readiness review；原指定 review 文件不存在，不把后续证据倒写成当日完成。
- Day 15：[PNG 预览](./day-15-qwen35-onboarding-close-roadmap.png) · [SVG 源文件](./day-15-qwen35-onboarding-close-roadmap.svg)——用后续 Day 18/20 的运行证据关闭 Qwen3.5 onboarding 风险：源码兼容、TP/DP、checkpoint、tiny overfit 与单卡 LoRA 均有实测；状态是 superseded closeout，不是原协议通过。
- Day 16：[PNG 预览](./day-16-controlled-lora-no-candidate-roadmap.png) · [SVG 源文件](./day-16-controlled-lora-no-candidate-roadmap.svg)——三档 LR 各跑 16k supervised tokens / 103 steps，但 0/3 同时满足 retention 与 Code eligibility；保留 fail-closed 结论，256k main、merge、confirmation 与 S1 均未启动。
- Day 17：[PNG 预览](./day-17-exact-resume-evidence-roadmap.png) · [SVG 源文件](./day-17-exact-resume-evidence-roadmap.svg)——从 state inventory、gap audit 到 uninterrupted/resume 对照与 failure injection 设计，说明“可恢复 checkpoint”不等于“已证明 exact resume”；正式 A/B comparator 仍未运行。
- Day 18：[PNG 预览](./day-18-qwen35-megatron-compatibility-roadmap.png) · [SVG 源文件](./day-18-qwen35-megatron-compatibility-roadmap.svg)——以 C0–C5 串起 HF↔Megatron conversion、数值 parity、TP/DP 更新、checkpoint continuation/export 和 150-step tiny overfit；10/10 required gates 通过，同时保留“不等于正式 SFT 或 exact resume”的边界。
- Day 19：[PNG 预览](./day-19-full-sft-regression-diagnostic-roadmap.png) · [SVG 源文件](./day-19-full-sft-regression-diagnostic-roadmap.svg)——复盘 standalone Qwen3.5 Full-SFT A/B/E 回退：84/84 wrapper+def 只证明格式存在，63/84 syntax salvage 也不等于准确率；把失败定位到 completion format、dataset contract 与 capability retention。顺序课程 optimizer/failure-injection 轨道未执行。
- Day 20：[PNG 预览](./day-20-target-boundary-qualified-lora-roadmap.png) · [SVG 源文件](./day-20-target-boundary-qualified-lora-roadmap.svg)——用 v0001→v0002 因果回路修复 Code target boundary，再在单卡 LoRA 中选出 81/112 的 qualified checkpoint；核心经验是先修 supervision contract，再比较 checkpoint。
- Day 21：[PNG 预览](./day-21-s1-selection-handoff-roadmap.png) · [SVG 源文件](./day-21-s1-selection-handoff-roadmap.svg)——从完整 comparison key、逐域 guardrail 与最早合格 tie-break，完成 S1 选择、resumable archive、merged export、4/4 adapter↔merged parity 和 downstream identity handoff。
- Day 22：[PNG 预览](./day-22-preference-data-audit-roadmap.png) · [SVG 源文件](./day-22-preference-data-audit-roadmap.svg)——从 promoted-S1 on-policy rollout、双跑 E2B、adaptive coverage、pair selection、processor audit 与 family split，到 200 对 machine-ready preference pairs。图中保留 08-13 formal audit 快照：正式人工盲审仍为 0/50 complete pairs，因此 formal DPO 继续 blocked；08-14 另行完成的 [experimental AI-assisted close](../day22-qwen35-experimental-ai-assisted-close.md) 已解锁实验路径，但不改写该 formal 边界。
