# Daily Learning Roadmaps

这里存放可直接提交到 Git 的每日学习路线图。SVG 是可编辑源文件，同目录 PNG 用于 Codex 手机端和其他不支持 SVG 的预览环境。

修改 SVG 后，在 bootcamp 根目录执行下面的命令。脚本会重新生成并覆盖全部同名 PNG：

```bash
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
