# Day 10 — Frozen Eval 与 Qwen3-0.6B Base Baseline

日期：`2026-08-05`

状态：`not_started`

强度：工作日 4–5 小时

## 主要目标

在第一次 SFT 前冻结评测协议，并保存 `Qwen/Qwen3-0.6B-Base` 的逐样本 baseline。后续训练只能比较同一 prompt/template/decoding/scorer 下的变化。

## 理论（60 分钟）

精读清单：[Day 10 — Frozen eval 与 base baseline](../SCALING-BOOK-READING-GUIDE.md#day-10)。

- train/dev/frozen test 的职责；dev 用于迭代，frozen test 不参与 mixture 与 checkpoint 选择。
- deterministic generation、答案抽取、slice metric、置信区间与人工 error taxonomy。
- benchmark contamination、selection bias 与“只看 aggregate score”的风险。
- 少量阅读《How To Scale Your Model》Part 7：prefill/generation 和 throughput，只用于估算评测预算。

## Coding（120 分钟）

- 冻结 90–150 条 eval manifest，明确划分 `dev` 与 `frozen_test`，至少覆盖 general instruction、math、code 三个与 Day 09 mixture 对齐的 slice。
- 每条记录 `sample_id/prompt_hash/slice/reference/scorer_version`；保存整个 manifest hash。
- 固定 model/tokenizer revision、chat template hash、generation config、answer extractor 与 scorer。
- generation 输出逐样本 JSONL：input ID、raw output、parsed answer、score、latency、error。

## 训练 / 实验（90–120 分钟）

- 不训练。
- 用 `Qwen/Qwen3-0.6B-Base` 跑 deterministic baseline；同配置重复 10 条验证可重复性。
- 人工审查至少 30 条，建立初版 error taxonomy；后续 checkpoint 选择只看 dev，frozen test 留到 Day 12 选定 checkpoint 后使用。
- 在看 SFT 结果前预注册 Day 12 的主指标、slice 指标、允许退化范围和 checkpoint 选择规则。

## 资源与租卡

- [Inference](https://jax-ml.github.io/scaling-book/inference/)
- [Tülu 3 Evaluation](https://github.com/allenai/olmes)
- 推荐 1×H100 80GB 或同类 GPU，预计 1–2 小时；0.6B baseline 结束并同步 predictions 后立即关机。
- 开卡前完成 manifest、5-sample CPU/tokenizer dry run 和磁盘检查。

## 产物

- `../artifacts/eval/day10-frozen-eval-manifest.json`
- `../artifacts/eval/day10-qwen3-0.6b-base-predictions.jsonl`
- `../artifacts/reports/day10-base-baseline.md`
- `../artifacts/reports/day10-eval-protocol.md`

## 验收

- [ ] Eval manifest、scorer、template 和 generation config 均有版本/hash。
- [ ] 10 条重复生成与解析结果一致；不一致项有原因和处理规则。
- [ ] 保存逐样本预测、slice 指标和 30 条人工审查，不只保存平均分。
- [ ] train/eval decontamination 报告无未处理的高相似 overlap。
- [ ] Day 12 的成功阈值和 checkpoint 选择规则已在训练前冻结。

## Daily Log

### Base 的三个主要 failure slice

### 评测协议的最大盲区

### Day 11 第一动作
