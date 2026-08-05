# Bootcamp Progress

开始日期：`2026-07-27`  
目标完成日期：`2026-08-25`  
当前 Phase：`Week 2 Day 10 核心产物已完成；Day 11 tiny overfit 为下一执行项`

当前最大阻塞：`none；Day 10 自动评测已完成，30 条人工复核保留为补充 gate，不阻塞 Day 11`

状态使用：`not_started`、`in_progress`、`blocked`、`done`。Day 01–03 的 `done` 来自用户确认；未据此补写不存在的 artifact 或用时。

| Day | 主题 | 状态 | 日期 | 用时 | 核心产物 | 一句话结论 |
|---:|---|---|---|---:|---|---|
| 01 | 环境与可复现基线 | done | 2026-07-27 | | | |
| 02 | Transformer accounting | done | 2026-07-28 | | | |
| 03 | Roofline 与 H100 | done | 2026-07-29 | | | |
| 04 | 分布式并行地图 | done | 2026-07-30 | | `day-04-parallelism-map/DAY-04-READING-QUIZ.md` | 已完成 sharding、collective、状态布局、并行组与首轮故障诊断。 |
| 05 | Training lifecycle 与框架职责图 | done | 2026-07-31 | | `day-05-training-lifecycle-framework-map/DAY-05-READING-QUIZ.md`、codepath walkthrough、tiny causal LM | 已完成 guided quiz、`swift sft` 边界追踪与 training-state walkthrough。 |
| 06 | 周末：Post-training scaling | done | 2026-08-01 | | `day-06-weekend-posttraining-scaling/README.md` | 完成 Tülu 3 × Applied Training crosswalk，区分 stage signal 与 memory/compute/time-cost feasibility。 |
| 07 | 周末：Week 1 复盘 | done | 2026-08-02 | | `artifacts/reports/week1-training-stage-decision-map.mmd`、`artifacts/reports/week1-gate-review.md` | Day 7 产物完成；Week 1 Gate 4/5，仅 stage-signal 独立口述复核待完成。 |
| 08 | SFT 数据契约与 loss token | done | 2026-08-03 | | `inspect_sft_sample.py`、data contract、20 条 edge cases、token-label audit | 已冻结 Qwen tokenizer/template 契约，验证 role-aware assistant loss、causal shift、截断与零监督拒绝逻辑。 |
| 09 | 数据质量、mixture 与 lineage | done | 2026-08-04 | | `artifacts/data/day09-dataset-manifest.json`、`artifacts/reports/day09-pipeline.svg`、`artifacts/reports/day09-mixture-findings.svg` | 完成 7,860 条 clean parent pool、等 supervised-token A/B mixture 与 frozen lineage/rebuild evidence。 |
| 10 | Frozen eval baseline | done | 2026-08-05 | | `artifacts/eval/day10-frozen-eval-manifest.json`、Base predictions、E2B code sidecar、`artifacts/reports/day10-base-baseline.md` | 冻结 160 条 eval；完成 112 条 dev Base 自动评测与重复性/重建验证，四-slice 15/112（code 0/28）；30 条人工复核仍为补充 gate，48 条 frozen_test 未消费。 |
| 11 | SFT step 与 tiny overfit | not_started | 2026-08-06 | | | |
| 12 | 受控 SFT 与 checkpoint 选择 | not_started | 2026-08-07 | | | |
| 13 | 周末：Qwen/Tülu post-training | not_started | 2026-08-08 | | | |
| 14 | 周末：Week 2 复盘 | not_started | 2026-08-09 | | | |
| 15 | Packing/length/effective-label-token ablation | not_started | 2026-08-10 | | | |
| 16 | Optimizer/LR/warmup/batch stability | not_started | 2026-08-11 | | | |
| 17 | Exact checkpoint resume/repro | not_started | 2026-08-12 | | | |
| 18 | Megatron min codepath/2-card TP-DP/dist checkpoint | not_started | 2026-08-13 | | | |
| 19 | 训练诊断/failure injection | not_started | 2026-08-14 | | | |
| 20 | 周末：Training failure signatures | not_started | 2026-08-15 | | | |
| 21 | 周末：Eval/checkpoint-selection 可靠性 | not_started | 2026-08-16 | | | |
| 22 | Preference provenance/length bias/held-out | not_started | 2026-08-17 | | | |
| 23 | DPO 推导与真实 smoke | not_started | 2026-08-18 | | | |
| 24 | Online RL dataflow/reward-verifier contract | not_started | 2026-08-19 | | | |
| 25 | ms-swift 小模型 GRPO lab | not_started | 2026-08-20 | | | |
| 26 | slime v0.3.0 主链路/准备 | not_started | 2026-08-21 | | | |
| 27 | 周末：GRPO/on-policy | not_started | 2026-08-22 | | | |
| 28 | 周末：slime debug/replay/repro | not_started | 2026-08-23 | | | |
| 29 | slime min loop/reward/replay | not_started | 2026-08-24 | | | |
| 30 | Post-training design/clean reproduction | not_started | 2026-08-25 | | | |

## 每周 Gate

- [ ] Week 1：能从一个 training step 说明数据、模型、优化器、checkpoint 与 eval 的职责边界；能把 sharding 映射到 OOM、batch 和吞吐问题。
- [ ] Week 2：能审计 SFT 数据契约，从 Base baseline 经 tiny overfit 到受控 SFT，并用逐样本 eval 选择 checkpoint。
- [ ] Week 3：能完成可验证 resume，做只改变一个变量的优化实验，并通过 failure injection 区分数据、数值、状态和系统故障。
- [ ] Week 4：能审计 preference data、跑通 DPO 与小模型 GRPO，并画出 slime 的在线 RL 数据和状态流。
- [ ] Final：能运行 slime 受控闭环，从干净环境复现最小 post-training pipeline，并用证据讲清数据、状态、Eval、RL 和失败归因。
