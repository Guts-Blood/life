# Day 13 — 周末 Reading：Tülu 3 × Qwen3.5 迁移边界

日期：`2026-08-08`

状态：`not_started`

强度：1 小时，仅阅读

## 主要目标

用一个完整开放 recipe（Tülu 3）和 Qwen3.5 的官方模型/训练资料，定义从历史 `Qwen3-0.6B` 实验迁移到 `Qwen/Qwen3.5-4B-Base` 的边界。今天只建立 lineage 与 readiness 判断，不把 production recipe 简化成框架命令列表，也不改写 Day 01–12 的任何证据。

## 理论（60 分钟）

精读清单仍从 [Day 13 — Tülu 3 与 Qwen3 post-training 对读](../SCALING-BOOK-READING-GUIDE.md#day-13) 进入；其中 Qwen3 材料只作为历史架构背景，迁移判断以下列 Qwen3.5 一手资料为准。

- 30 分钟：[Tülu 3 paper](https://arxiv.org/abs/2411.15124) 的 pipeline、SFT mixture、preference data、RLVR、evaluation/decontamination 与明确报告的 negative results。
- 15 分钟：[Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) 的模型类型、上下文、processor/template 与 Base 边界。
- 10 分钟：[ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md) 的 Dense model、依赖、packing/padding-free、LoRA 与 RL 入口；只记录当前文档事实，不提前冻结滚动版本。
- 5 分钟：完成对照表：

| 维度 | Tülu 3 | Day 01–12：Qwen3-0.6B v1 | Day 13+：Qwen3.5-4B v2 |
|---|---|---|---|
| 上游 checkpoint | | `Qwen3-0.6B-Base` 与历史 run | `Qwen3.5-4B-Base` exact revision 待 Day 15 冻结 |
| data signal | | 历史 tokenizer/template/token budget | 必须重新 render/tokenize/hash |
| objective/stage | | Base baseline、tiny overfit、受控 SFT | Base → 新 SFT anchor → DPO/GRPO |
| eval/gate | | Day 10/12 v1 证据；旧 dev 已支持十轮选择 | scorer/task schema 可复用；新 v2 selection dev/confirmation 与 Base baseline |
| 可复现信息与缺口 | | 不再改写 | processor、loader、vision freeze、runtime、显存待 gate |

必须回答：

1. 哪些能力问题适合先改 SFT data，哪些需要 preference 或可验证 reward signal？
2. Tülu 3 的 openness 让哪些因果判断更可信？Qwen3.5 官方资料中哪些细节仍必须由 pinned runtime 与 pilot 补齐？
3. Day 12 的 0.6B 结果最多支持什么方法论结论，哪些 tokenizer、token budget、checkpoint 和能力结论不能迁移到 4B？

## Coding

无。

## 训练 / 实验

无；只确认 Day 12 artifacts 已同步且保持只读，不开启新 run。

## 资源与租卡

阅读不开 GPU。若 Day 12 任务仍在运行，只做健康检查和到点停止，不临时改参；Qwen3.5 权重下载、依赖安装和显存 smoke 留到 Day 15。

## Lineage Memo

今天必须写一页迁移 memo，并使用以下不可变边界：

```text
v1 / historical / read-only:
  Day 01–12, Qwen3-0.6B tokenizer/data/eval/checkpoints/results

v2 / active / new lineage:
  Qwen/Qwen3.5-4B-Base -> new tokenizer/processor contract
  -> new Base dev baseline -> new tiny-overfit/resume proof
  -> new selected SFT anchor -> DPO/online RL
```

允许复用的是 raw sample IDs、清洗方法、scorer 和证据模板；禁止复用或覆盖 rendered text、token IDs、label spans、token-budget manifest、model-specific predictions、checkpoint 或 hash。

## 验收

- [ ] 完成对照表并写 3 个带适用边界的 takeaway。
- [ ] 至少记录一个公开 recipe 的 negative result 或无效方向。
- [ ] 写出 v1/v2 lineage memo，并明确两条 lineage 没有权重连续性。
- [ ] 能把新的 4B 路线放回 `Base -> SFT anchor -> preference/DPO 或 online RL -> eval` 链路。

### 三个 takeaway

-
-
-
