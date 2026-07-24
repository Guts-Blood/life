# Day 26 — slime 主链路通读与运行准备

日期：`2026-08-21`  
状态：`not_started`  
强度：4–5 小时人工工作；训练可后台继续

## 主要目标

完成 slime 的第一遍端到端主链路通读，固定官方 Docker/commit，准备好 Qwen3-4B、数据、HF↔Megatron checkpoint 和 8 卡 runbook，使 Day 29 开机后直接验证系统而不是安装环境。

## 理论（45 分钟）

精读清单：[Day 26 — slime 的 train/rollout/weight-sync 系统边界](../SCALING-BOOK-READING-GUIDE.md#day-26)。先回答为什么 RL infra 不能等同于一个 GRPO loss 函数。

## Repo 通读（180 分钟）

固定 slime commit 后按数据生命周期阅读：

1. **入口/参数（30 分钟）**：`train.py`、`train_async.py`、`slime/utils/arguments.py`；找到同步/异步分叉。
2. **Ray 资源角色（40 分钟）**：`slime/ray/placement_group.py`、`actor_group.py`、`rollout.py`、`train_actor.py`；画 GPU/rank/actor ownership。
3. **rollout/data/reward（50 分钟）**：`slime/rollout/base_types.py`、`sglang_rollout.py`、data source 与 `rm_hub`；记录 `Sample` 在每一步新增的字段。
4. **training/weight sync（60 分钟）**：`backends/megatron_utils/actor.py`、`sglang.py`、`update_weight/`；区分 disk/tensor/distributed update，找到触发新权重生效的位置。

每条边写 `producer -> object/schema -> transport -> consumer` 和 `file:function`。未验证的地方标 `STATIC_ONLY`，Day 29 用日志/trace 消除。

## Coding / 运行准备（60–75 分钟）

- 使用官方 Docker，记录 image digest；固定 slime、Megatron、SGLang commits/patches。
- 下载/确认 Qwen3-4B、数学数据集和 tokenizer，完成 checksum/磁盘预算。
- 按官方工具完成或 dry-run HF → Megatron `torch_dist` 转换，并写回转 HF 命令。
- 固化 8×H100 official-style config：placement、train/rollout GPUs、TP、batch、`n_samples_per_prompt`、max steps=5–10。
- 准备两个 reward：原始 accuracy+format，以及受控修改版；为 reward 写 5 个 unit cases。

## 训练 / 实验

- 今天不启动多卡 RL；只运行 CPU unit tests、Docker import/config check，以及必要的 checkpoint conversion。
- 成功标准是 Day 29 开机后直接进入 1-rollout/1-update gate，不在 8 卡实例上下载、编译或首次理解参数。

## 资源与租卡

- CPU/no-card 为主；checkpoint conversion 若 CPU 太慢，可用 1×H100 不超过 2 小时。
- Day 29 推荐同机 8×H100 走官方 Qwen3-4B recipe；今天绝不提前租 8 卡。
- Docker、模型、数据、转换后 checkpoint 必须放可靠存储，并确认 AutoDL 实例能直接挂载/复制。

## 验收

- [ ] 调用链覆盖 entry、Ray roles、rollout、buffer/sample、reward、Megatron train、weight sync。
- [ ] 每条边有 object/schema 和 `file:function`；未知项明确标 `STATIC_ONLY`。
- [ ] Docker/commits/patches、模型、数据、转换 checkpoint、8 卡 config 均已固定。
- [ ] 原始/修改版 reward tests 通过，Day 29 第一条命令、5-step gate 和停机条件明确。

## Daily Log

### 读完的文件 / commits / image digest

### STATIC_ONLY edges

### Day 29 runbook / resource map
