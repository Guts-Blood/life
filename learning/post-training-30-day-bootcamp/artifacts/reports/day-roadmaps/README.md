# Daily Learning Roadmaps

这里存放可直接提交到 Git 的每日学习路线图。统一使用 SVG，便于在 GitHub、Markdown 和浏览器中直接查看。

命名约定：

```text
day-XX-<topic>-roadmap.svg
```

当前文件：

- [`day-01-roofline-foundations-roadmap.svg`](./day-01-roofline-foundations-roadmap.svg)：从环境基线、矩阵乘法 accounting 到 Roofline、tokens/s 与训练显存的基础地图。
- [`DAY-02-LEARNING-ROADMAP.svg`](../../../day-02-transformer-accounting/DAY-02-LEARNING-ROADMAP.svg)：从 Transformer 参数量、激活量、FLOPs 到训练容量估算。
- [`day-03-roofline-h100-roadmap.svg`](./day-03-roofline-h100-roadmap.svg)：从 matmul accounting、H100 Roofline、local tokens 到 MFU scenario 与 profiler 校准。
- [`day-04-parallelism-roadmap.svg`](./day-04-parallelism-roadmap.svg)：从 sharded tensor、collective、DP/FSDP/TP/PP 与 rank groups，到 OOM、hang、checkpoint mismatch 的证据化诊断。
- [`day05-training-lifecycle-roadmap.svg`](../day05-training-lifecycle-roadmap.svg)：从数据边界、forward/backward、optimizer/checkpoint 到 ms-swift、DeepSpeed、NCCL 与 CUDA 的职责分层。
