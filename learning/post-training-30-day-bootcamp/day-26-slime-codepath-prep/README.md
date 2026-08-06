# Day 26 — slime v0.3.0 主链路与最小运行准备

日期：`2026-08-21`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

以 slime `v0.3.0` 为学习基线，理解 `Sample → DataSource → rollout/reward → train → weight update/version` 主链路，并准备 Day 29 的最小可行运行。框架迭代快，必须先验证 tag/commit，再记录实际路径，不能依赖计划里硬编码的源码布局。

## 版本 Gate（20 分钟）

1. 查看 [官方 Releases](https://github.com/THUDM/slime/releases)，确认 `v0.3.0` tag 存在。
2. checkout `v0.3.0`，记录 tag 对应 SHA、submodule/dependency versions、Docker digest。
3. 若工作目标指定另一 commit，另外记录 diff；本日概念图仍以 pinned baseline 为准。
4. 用该 checkout 的 README、examples、`--help` 和源码搜索验证所有符号/参数后再写命令。

## 理论（45 分钟）

精读清单：[Day 26 — slime pinned architecture](../SCALING-BOOK-READING-GUIDE.md#day-26)。

- Megatron learner、SGLang rollout、Ray orchestration、DataSource/Data Buffer 的 ownership。
- 同步 loop 与 pipelined/async loop 的差别。
- 训练数据为什么必须携带 rollout/group ID、loss mask、reward 和 policy/weight version。
- Weight sync 成功与“下一次 rollout 确实使用新权重”不是同一条证据。

## Repo 主链路（150 分钟）

不要按目录浏览。用 `rg`/symbol search 在 pinned checkout 中解析并记录下列符号的实际 `file:function`：

1. 同步/异步入口和一次训练 loop。
2. `Sample` schema、status、rollout/group ID、loss mask、reward、weight version。
3. `DataSource` 取样、buffer/requeue 和 dataset cursor。
4. rollout generation、reward/verifier、filter/group。
5. rollout data 转成 train batch、advantage/loss、optimizer step。
6. learner 权重同步到 rollout engine，以及下一轮 version 证据。

每条边写 `producer -> object/schema -> transport -> consumer -> observable evidence`。路径只写进当天 artifact；如果 tag 中名字变化，以搜索到的代码为准。

## Coding / Day 29 准备（75 分钟）

- 运行 pinned tag 自带的 CPU/contract tests 或最小 import/config check。
- 从该 tag 的 examples 中选择**最小受支持**的 model、dataset、GPU topology；记录选择依据和预算上限。
- 实现 Day 24 reward 的 slime adapter，并跑相同 unit cases。
- 验证该 tag 的 rollout-only、train-only/replay、debug dump 和 full-loop 参数；保存 `--help`/docs 证据。
- 准备 baseline reward 与单一受控修改版、1-rollout/1-update gate 和停止条件。

## 训练 / 实验

- 今天不启动正式 RL；允许 CPU tests、Docker import、两条 fake samples 的 schema/reward dry-run。
- 不做大模型权重转换，除非 pinned 最小 recipe 明确要求且能在预算内完成。

## 资源与租卡

- CPU/no-card 为主；可选单卡不超过 1 小时做 import/权重加载 gate。
- Day 29 Core 使用 Day 26 实测确认的最小受支持 topology。
- 8×H100 官方规模 recipe 仅列为 Stretch，不是 Day 29 或毕业必做项。

## Evidence-first 产物

- `../artifacts/reports/day26-slime-pinned-codepath.md`
- `../artifacts/configs/day26-slime-environment.json`
- reward adapter tests、resolved example/config、Day 29 runbook

## 验收

- [ ] `v0.3.0` tag、SHA、dependencies 与 image digest 已固定。
- [ ] Sample/DataSource/rollout/train/weight-version 每条边有实际 `file:function` 和 schema。
- [ ] 所有 CLI/config 来自 pinned tag 验证，不依赖当前 main 或记忆。
- [ ] 最小 topology、reward tests、debug/replay/full-loop gates 和预算已确定。
- [ ] 8×H100 明确标为 Stretch。

## Optional Capstone Handoff

输出 runtime lock、resource-placement map、rollout/train object schema、debug/replay entrypoints 和 weight-sync evidence contract。当前 slime 主链只作为 RL runtime reference；除非 Day 31 对 pinned release 的官方代码确认 teacher scoring 与 OPD objective，不能假定 `v0.3.0` 能直接承担 Capstone OPD。Framework-selection spike 必须基于官方 release/docs/runtime dry run。

## Daily Log

### Tag / SHA / image / resolved paths

### Object lifecycle

### Minimum supported recipe

### Day 29 gates
