# Day 18 — Megatron 双卡 SFT 运行与 Runtime Trace

日期：`2026-08-13`  
状态：`not_started`  
强度：4–5 小时人工工作；多卡实例 6–8 小时

## 主要目标

在真实双卡上运行 upstream Megatron smoke 和 Qwen SFT，用 rank 日志把 Day 17 的静态调用链逐边验证，并完成一次 checkpoint save/load。

## 理论（45 分钟）

精读清单：[Day 18 — Megatron 双卡 runtime](../SCALING-BOOK-READING-GUIDE.md#day-18)。Part 12 的 node-level networking/collective 必须用实际 topology 与 NCCL 数字校准。

- 复习 Day 17 的 TP/DP/process-group 与 collective 表。
- 定义 global throughput、per-GPU throughput、strong scaling efficiency。
- 明确 world size 改变时如何保持 effective batch 或解释差异。

## Coding / 插桩（60 分钟）

- 开机先运行 topology/NCCL sanity test。
- 启用 Day 17 的轻量插桩：rank、TP/DP/PP group、input/label/loss-mask shape、forward/backward boundary、memory。
- 保存 Megatron clean commit 与本地 diff；不要把大量 print 永久散落在源码。

## 训练 / 实验（180 分钟启动与分析）

按 gate 顺序执行：

1. Run A：官方 `NVIDIA/Megatron-LM` synthetic GPT，2 ranks、10–20 steps，验证 upstream 环境与 process groups。
2. Run B：通过 Megatron-SWIFT/Mcore-Bridge 跑 Qwen3-1.7B 或 4B SFT，10-step gate 后到 30 steps。
3. 保持 global batch/data/seed，比较 `TP=1, DP=2` 与 `TP=2, DP=1`；小模型 TP 变慢也属于有效结果。
4. 保存 `torch_dist`/sharded checkpoint，结束进程后重新启动并加载到更高 step。
5. 用固定 5 prompts 导出/加载 checkpoint 后 inference，确认不是只会保存不能使用。

## 资源与租卡

- 推荐：同一主机 2×H100 80GB，6–8 wall-clock 小时。
- 可替代：同机 2×A100 80GB；第一次不使用跨节点。
- 不做 4 卡 Stretch；把时间用于 runtime-to-source 对照和 checkpoint reload。
- 全部 run、per-rank logs、diff、checkpoint metadata 保存后关闭实例。

## 验收

- [ ] Run A 与 Run B 都真实完成 optimizer step，不是仅 initialize。
- [ ] 至少五条 Day 17 调用链边获得 runtime log/stack 证据。
- [ ] TP/DP 两组配置的 rank groups、loss、memory、step time 可比较。
- [ ] 至少一次多卡 checkpoint save、进程退出、reload、继续训练成功。
- [ ] 能解释哪个行为来自 Megatron Core，哪个来自 Megatron-SWIFT/Mcore-Bridge。

## Daily Log

### GPU topology / hours

### Static call chain vs runtime evidence

### TP=1/2 对比

### Checkpoint reload

### Day 19 第一动作
