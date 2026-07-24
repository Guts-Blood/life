# Day 28 — 周末 Reading：slime 架构与源码复核

日期：`2026-08-23`  
状态：`not_started`  
强度：1 小时，仅阅读

## 主要目标

在不开 GPU 的前提下复核 Day 26 的 slime 主链路，重点消除 weight sync 与 Ray placement 两个 `STATIC_ONLY` 区域。

## 理论（60 分钟）

精读清单：[Day 28 — slime runtime 架构](../SCALING-BOOK-READING-GUIDE.md#day-28)。

- 20 分钟：重读 `train.py`/`train_async.py` 与 `ray/placement_group.py`，画 actor/GPU placement。
- 20 分钟：重读 `rollout/sglang_rollout.py` 与 `backends/megatron_utils/actor.py`，补齐 train/rollout handoff。
- 20 分钟：重读 `backends/megatron_utils/update_weight/`，比较 disk/tensor/distributed sync 的触发和一致性风险。

最终图必须是：`prompt -> N rollouts -> reward -> buffer/Sample -> advantage/loss -> Megatron update -> weight sync -> next rollout`，每条边标数量、GPU role、函数和可观测日志。

## Coding

无。

## 训练 / 实验

无。

## 资源与租卡

CPU only；确认 Day 29 的 Docker digest、Qwen3-4B、torch_dist checkpoint、dataset、两个 reward 和 8 卡 config 均在可靠存储，不开卡。

### 最终静态架构图 / 剩余 STATIC_ONLY
