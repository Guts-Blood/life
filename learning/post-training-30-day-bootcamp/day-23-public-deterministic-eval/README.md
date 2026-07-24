# Day 23 — 公共 Benchmark 与 Deterministic Eval

日期：`2026-08-18`  
状态：`not_started`  
强度：4–5 小时；Eval 可后台继续

## 主要目标

同时建立公共 benchmark baseline 和自己的规则评分链路；两者都保存逐样本证据和版本信息。

## 理论（60 分钟）

精读清单：[Day 23 — Public benchmark 与 offline inference](../SCALING-BOOK-READING-GUIDE.md#day-23)。区分会改变能力结论的 eval config 与只改变推理吞吐/成本的 serving config。

- Multiple-choice loglikelihood vs free generation。
- Few-shot、chat template、stop sequence、normalization。
- Exact/numeric/schema/format scorers 的边界。
- Public benchmark 与业务 frozen eval 各回答什么问题。

## Coding（135 分钟）

- 固定 [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) commit 与 task config。
- 实现 `scorers.py`：exact、normalized、numeric、JSON schema、format。
- 为每个 scorer 写正例、反例和边界 unit tests。
- 输出逐样本 score、error type、aggregate 和 slices。

## 训练 / 实验（75–90 分钟启动/检查）

- 不训练。
- Base/SFT 对 2–4 个任务跑小 `limit` 检查 rendered prompts，再跑完整或合理子集。
- 使用 Day 10 frozen predictions 运行 deterministic scorers；缺失结果再批量生成。
- 人工审查至少 30 条 scorer 判定。

## 资源与租卡

- 1×H100 80GB，预计 4–7 小时用于 Base/SFT benchmark 与缺失生成。
- Coding/tests 先在 CPU 完成；GPU 只用于批量 inference。
- 结果落盘后关机，报告在本地写。

## 验收

- [ ] Model/harness/task/template/backend 版本固定。
- [ ] Scorer tests 通过，人工审查误判有记录。
- [ ] Base/SFT 使用一致参数。
- [ ] 保存逐样本结果、样本量和不确定性，不只 aggregate。

## 产物

- `../artifacts/eval/scorers.py`
- `../artifacts/eval/test_scorers.py`
- `../artifacts/eval/day23-public-results/`
- `../artifacts/eval/day23-deterministic-results.jsonl`

## Daily Log

### Public tasks / versions

### Scorer audit

### Day 24 第一动作
