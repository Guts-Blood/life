# Day 10 — Frozen Eval 与 Qwen3-0.6B Base Baseline

日期：`2026-08-05`

状态：`base_dev_automated_eval_complete / human_review_pending`（Day 09 以 Gate C waiver 完成；`frozen_test` 未消费）

强度：工作日 4–5 小时

## 主要目标

在第一次 SFT 前冻结评测协议，并保存 `Qwen/Qwen3-0.6B-Base` 的逐样本 **dev-only** baseline。后续训练只能比较同一 prompt/template/decoding/scorer 下的变化；`frozen_test` 留到 Day 12 选定 checkpoint 后才揭盲。

## 理论（60 分钟）

精读清单：[Day 10 — Frozen eval 与 base baseline](../SCALING-BOOK-READING-GUIDE.md#day-10)。

- train/dev/frozen test 的职责；dev 用于迭代，frozen test 不参与 mixture 与 checkpoint 选择。
- deterministic generation、答案抽取、slice metric、置信区间与人工 error taxonomy。
- benchmark contamination、selection bias 与“只看 aggregate score”的风险。
- 少量阅读《How To Scale Your Model》Part 7：prefill/generation 和 throughput，只用于估算评测预算。

## Coding（120 分钟）

- 在 Day 09 Gate C waiver 的声明边界内，冻结全部 160 条 eval candidates：general knowledge、math、code、finance 各 40 条；每 slice 固定为 28 `dev` + 12 `frozen_test`。
- 每条记录 `sample_id/raw_prompt_hash/rendered_prompt_hash/input_ids_hash/slice/reference/scorer_version`；保存整个 manifest hash。
- 固定 model/tokenizer revision、chat template hash、generation config、answer extractor 与 scorer。
- generation 输出逐样本 JSONL：input ID、raw output、parsed answer、score、latency、error。

## 训练 / 实验（90–120 分钟）

- 不训练。
- 只在 112 条 `dev` 上用 `Qwen/Qwen3-0.6B-Base` 跑 deterministic baseline；同配置重复 10 条验证可重复性。
- 人工审查至少 30 条，建立初版 error taxonomy；后续 checkpoint 选择只看 dev，frozen test 留到 Day 12 选定 checkpoint 后使用。
- 在看 SFT 结果前预注册 Day 12 的主指标、slice 指标、允许退化范围和 checkpoint 选择规则。

## 资源与租卡

- [Inference](https://jax-ml.github.io/scaling-book/inference/)
- [OLMES](https://github.com/allenai/olmes)
- 本轮 0.6B baseline 优先使用本地 `post_training_lab` CPU 环境，不需要租卡；若改用 GPU，必须冻结新的 execution protocol，并在后续比较时用相同后端重跑 Base。
- 开卡前完成 manifest、5-sample CPU/tokenizer dry run 和磁盘检查。

## 产物

- `../artifacts/eval/day10-frozen-eval-manifest.json`
- `../artifacts/eval/day10-qwen3-0.6b-base-predictions.jsonl`
- `../artifacts/eval/day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl`
- `../artifacts/eval/day10-base-human-review-packet-30.jsonl`（30 条、已合并 E2B code outcome，全部待人工填写）
- `../artifacts/reports/day10-base-baseline-summary.json`
- `../artifacts/reports/day10-base-baseline.md`
- `../artifacts/reports/day10-eval-protocol.md`
- [Day 10 Frozen Eval 总结图（SVG，可编辑）](../artifacts/reports/day10-frozen-eval-baseline.svg)
- [Day 10 Frozen Eval 总结图（PNG，手机预览）](../artifacts/reports/day10-frozen-eval-baseline.png)
- `../artifacts/configs/day12-checkpoint-selection-policy.json`
- `day10_e2b_sandbox_config.json`
- `score_day10_code_e2b.py`

## 验收

- [x] Eval manifest、scorer、template 和 generation config 均有版本/hash。
- [x] 10 条重复生成与解析结果一致；token IDs/raw output/scorer 均为 0 mismatch。
- [x] 保存 112 条逐样本预测、slice 指标/CI 和 30 条 dev-only 待审 packet；code review 行包含已验证的 E2B syntax/runtime outcome 与完整 provenance。
- [ ] 用户完成 30 条人工审查并确认/修订 error taxonomy。
- [x] 在逐样本新建、禁公网的 E2B sandbox 中完成 28 条 HumanEval；0/28 pass，四-slice macro 13.39%。
- [x] 对冻结的 Day 09 train manifests 与 matcher/threshold，train/eval 无未处理的高相似 overlap。
- [x] Day 12 的成功阈值和 checkpoint 选择规则已在训练前冻结，并要求匹配 generation `comparison_key`、code execution protocol 与 `complete_comparison_key`。

## Optional Capstone Handoff（不增加 Day 10 Core）

- 复用 eval manifest/scorer registry、逐样本 prediction、selection policy 与 frozen-consumption ledger 的 schema。
- Day 12 一旦揭盲 `frozen_test`，它只能作为已消费 regression suite；Day 31 必须建立新的 capstone domain/tool `dev/frozen` confirmation set。
- 跨 8B/4B 比较使用同一个 `eval_suite_hash` 对齐 raw task IDs、references、environment/scorer 和 aggregation；只有 inputs/rendering/execution 也相同时才共享 `comparison_key`，model/config/weight identity 由各自 `run_hash` 区分。
- Capstone baseline 要在训练前保存 T0/S0 dev evidence，并在所有 stage checkpoints 选定后一次性运行新 frozen suite。

## Daily Log

### Base 的四个主要 failure slice

- general：17/28 parse failure，2/28 correct。
- math：答案都能解析，但 16/28 wrong answer，12/28 correct。
- code：0/28 correct；25 条 syntax error、3 条 runtime error，E2B infrastructure failure 为 0。
- finance：16/28 parse failure，1/28 correct。

共同现象：112/112 都 hit `max_new_tokens` ceiling，符合 Base 模型把 chat prompt 当续写的行为。

### 评测协议的最大盲区

30 条 audit packet 尚无人类 judgment；此外 28-per-slice 只是一组学习型 pilot，不能外推成人口总体 benchmark。code 的 0/28 只代表当前冻结 prompt/generation/E2B scorer 协议，不能替代对更大模型或不同 harness 的横向结论。

### Day 11 第一动作

在 `post_training_lab` 环境中复用已冻结 tokenizer/template，运行 tiny-overfit training lifecycle gate；不读取 `frozen_test`，不因 Base bad cases 原地修改本轮 prompt/scorer/A-B manifests。
