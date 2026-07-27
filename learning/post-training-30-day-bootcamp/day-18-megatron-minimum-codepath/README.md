# Day 18 — Megatron 最小源码链、双卡 TP/DP 与 Distributed Checkpoint

日期：`2026-08-13`
状态：`not_started`
强度：4–5 小时人工工作；双卡实例 3–5 小时

## 主要目标

把 Megatron 压缩为一天的最小必要理解：从入口走到 data/model/loss/optimizer/checkpoint，并在双卡上分别做 DP 与 TP smoke。今天不追求“通读 Megatron”。

## 理论（45 分钟）

精读清单：[Day 18 — Megatron minimum codepath](../SCALING-BOOK-READING-GUIDE.md#day-18)。

- TP=1/DP=2 与 TP=2/DP=1 的 parameter、gradient、optimizer ownership。
- 两种配置分别需要的 collective，以及小模型 TP 可能变慢的原因。
- Distributed checkpoint 为什么必须同时记录 sharding metadata 与非 tensor state。

## 最小源码链（75 分钟）

固定 Megatron commit/镜像，用该 commit 的搜索结果确认实际文件路径，只追以下符号和调用关系：

```text
entry/config
  -> dataset provider / batch
  -> model provider / forward step
  -> loss
  -> backward / optimizer step
  -> distributed save/load
```

每个节点只记录 `file:function`、输入/输出 shape、运行在哪些 ranks、拥有哪类 state。Parallel schedules、tensor mapping 和 kernels 只在真实 stack 经过时展开一层；其余标为 `OUT_OF_SCOPE`。

## Coding / 插桩（45 分钟）

- 记录 pinned commit、image digest、GPU topology 和完整启动 config。
- 用可开关 hook/logger 记录 rank、TP/DP group、batch/label shape、loss、optimizer step、checkpoint save/load。
- 准备两组只改变 TP/DP 的 config，并断言 global batch 与 label-token budget 一致。

## 训练 / 实验（120–150 分钟）

1. 先用 pinned commit 的官方最小 GPT/synthetic recipe 完成 5-step 单配置 gate。
2. Run A：2 ranks，`TP=1, DP=2`，10–20 optimizer steps。
3. Run B：2 ranks，`TP=2, DP=1`，10–20 optimizer steps。
4. 比较 rank groups、loss、per-GPU memory、step time 和 exposed communication；不把两卡 scaling 当主目标。
5. 保存 distributed checkpoint，退出所有进程，新进程恢复并再跑 3 steps；审计 model/optimizer/scheduler/RNG/iteration metadata。

若 Qwen adapter/转换在预定 30 分钟内已有可用 recipe，可增加 5-step Qwen smoke；它是 Stretch，不得挤掉 TP/DP 和 checkpoint Core。

## 资源与租卡

- 同机 2×H100 80GB，预计 3–5 小时；可替代同机 2×A100 80GB。
- 不跨节点、不做四卡、不做长训练。
- import/config/topology gate 未通过时不开正式 run；证据同步后立即关机。

## Evidence-first 产物

- `../artifacts/reports/day18-megatron-minimum-codepath.md`
- 两组 per-rank logs/configs
- distributed-checkpoint manifest 与 reload 证据

## 验收

- [ ] 最小调用链每条边有 pinned `file:function` 和 runtime 证据。
- [ ] TP/DP 两组都真正完成 optimizer step。
- [ ] 能解释两组 rank ownership 与 collective 差异。
- [ ] distributed checkpoint 在新进程成功恢复并继续三步。
- [ ] 清楚列出未深入的 Megatron 区域，不把一天阅读称为全仓通读。

## Daily Log

### Pinned environment / topology

### Minimum codepath

### TP vs DP

### Distributed resume

### Day 19 第一动作
