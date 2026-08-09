# Bootcamp Progress

开始日期：`2026-07-27`  
目标完成日期：`2026-08-25`  
当前 Phase：`Day 12 已收束；Day 18 standalone pull-forward 已完成；Day 13 仍为下一顺序执行项，Day 13+ 为 Qwen3.5-4B v2 lineage`

当前最大阻塞：`Day 13–14 阅读/合同无阻塞；在 Day 16 启动正式 SFT 前，必须完成 Qwen3.5 exact revision、processor/template、环境/loader、重分词、新 Base baseline、tiny-overfit/resume 与显存 gate；Day 18 standalone evidence 不替代 Day 15 onboarding/migration acceptance。`

状态使用：`not_started`、`in_progress`、`blocked`、`deferred`、`done`。`deferred` 表示需要新决策/章程，不是当前主线阻塞。Day 01–03 的 `done` 来自用户确认；未据此补写不存在的 artifact 或用时。

## 活动模型迁移

| 项目 | 状态 | 说明 |
|---|---|---|
| `qwen3-0.6b-day01-12-v1` | closed / immutable | Day 11 tiny-overfit pass；Day 12 recovery C–L 10/10 完成、0 eligible checkpoint、frozen test 未消费。 |
| `qwen35-4b-day13-plus-v2` | standalone revision frozen / sequential acceptance pending | 唯一活动模型为 `Qwen/Qwen3.5-4B-Base@1001bb4d…`；该 revision 已供 Day 18–20 standalone runs 使用，Day 15 的完整 onboarding/data/eval acceptance 仍待顺序执行。 |
| v2 training scope | selected | text-only coding；完整 processor/conditional-generation loader；vision tower 与 aligner 冻结并做 trainable coverage 断言。 |
| v2 `S1` | nonexistent | 必须先通过新 baseline、one-step、tiny-overfit/resume 与受控 coding SFT promotion gate。 |
| DPO / GRPO | blocked until `S1` | 不能从 Base、Day 12 0.6B export 或不合格候选起步。 |
| Teacher / OPD | deferred / unselected | `teacher_model_id=null`；不自动选择 8B/9B teacher。 |

迁移的完整 gate 与机器可读状态见 [`QWEN35-4B-MIGRATION-PLAN.md`](QWEN35-4B-MIGRATION-PLAN.md) 和 [`artifacts/configs/qwen35-4b-migration-contract.json`](artifacts/configs/qwen35-4b-migration-contract.json)。

顺序边界：Day 18–20 的 standalone evidence 是独立 pull-forward/diagnostic 结果；`qwen35-4b-day13-plus-v2` 的 Day 15 acceptance、顺序课程数据/评测迁移与 `S1` promotion 状态均不因此改变。

## 提前执行的 Standalone Experiments

| 实验 | 状态 | 日期 | 核心产物 | 结论 |
|---|---|---|---|---|
| Day 18 Megatron compatibility | done | 2026-08-08 | `artifacts/reports/day18-qwen35-megatron-compatibility.md` | C0–C5 通过；只覆盖冻结的 2×H800 text-only compatibility/learnability envelope。 |
| [Day 19 Qwen3.5 Full-SFT comparison](day-19-qwen35-sft-comparison/README.md) | done / diagnostic only | 2026-08-08–09 | A/B/E comparison、Qwen3.5 response/HumanEval adapters、root-cause report | 确认评测合同错位与真实 Full-SFT 退化同时存在；没有 eligible/promoted S1。 |
| [Day 20 balanced LoRA probes](day-20-qwen35-balanced-lora-sft/README.md) | blocked / no passing probe | 2026-08-09 | Base + `1e-5/3e-5/1e-4` probes、静态/v2 diagnostic report | 三档 probe 均未通过预注册 gate；未生成 canonical selection、main checkpoint、results 或 PASS，Base 保持 active。 |

这里的 Day 19/20 是 standalone run identity；下方顺序课程表中的 Day 19 optimizer/failure injection 与 Day 20 weekend review 仍为 `not_started`。

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
| 11 | SFT step 与 tiny overfit（v1） | done | 2026-08-06 | | `day-11-sft-step-tiny-overfit/DAY11-LOOKBACK.md`、`artifacts/reports/day11-tiny-overfit-retrospective.svg` | Step 50 teacher-forced accuracy 98.89%；6-step continuous 与 3+resume+3 exact probe 通过。 |
| 12 | 受控 SFT 与 checkpoint 选择（v1） | done | 2026-08-07 | | `artifacts/reports/day12-recovery-final-retrospective.md`、`artifacts/reports/day12-recovery-final-summary.json` | Recovery C–L 10/10 完成，0 个 checkpoint 通过 math/code/total 联合 gate；frozen test 未消费。 |
| 13 | 周末：Qwen3 历史证据 × Qwen3.5 迁移 | not_started | 2026-08-08 | | | next；完成 lineage/差异 memo，不租 GPU。 |
| 14 | 周末：Week 2 历史复盘 + Qwen3.5 readiness | not_started | 2026-08-09 | | | M0–M6 的 owner/evidence/stop 条件必须完整。 |
| 15 | Qwen3.5 onboarding/migration acceptance | not_started | 2026-08-10 | | | Freeze revision/processor/runtime；retokenize；new Base baseline；one-step/tiny/resume/memory gates。 |
| 16 | Controlled coding LoRA SFT/packing parity/candidate set | not_started | 2026-08-11 | | | 未通过 Day 15 则 blocked；不继承 0.6B LR/token budget。 |
| 17 | Exact checkpoint resume/repro | not_started | 2026-08-12 | | | |
| 18 | Megatron min codepath + Qwen3.5 GDN/loader gate | done | 2026-08-08（pull-forward） | | `artifacts/reports/day18-qwen35-megatron-compatibility.md` | 独立提前完成 C0–C5：同一 Megatron SFT 入口覆盖 parity、TP1/DP2、TP2/DP1、fresh-process full-state resume/export 与 150-step two-row overfit；仅证明该冻结 text-only 路径未检出 bug，不回填 Day 15 或晋级 S1。 |
| 19 | Optimizer/LR stability + failure injection | not_started | 2026-08-14 | | | |
| 20 | 周末：Training failure signatures | not_started | 2026-08-15 | | | |
| 21 | Qwen3.5 candidate audit / final S1 promotion | not_started | 2026-08-16 | | | 旧 Day 10/12 candidates 只作历史诊断；输出 S1 或 no-anchor。 |
| 22 | Coding preference provenance/processor/held-out | not_started | 2026-08-17 | | | |
| 23 | Qwen3.5 coding DPO smoke（parent=S1） | not_started | 2026-08-18 | | | 执行 gate=`S1` 已晋级；Base 不能代替。 |
| 24 | Coding online-RL dataflow/sandbox reward contract | not_started | 2026-08-19 | | | |
| 25 | Qwen3.5 coding GRPO lab（parent=S1） | not_started | 2026-08-20 | | | 执行 gate=`S1` 已晋级；首轮显式 cap 8K。 |
| 26 | slime fixed-release Qwen3.5 compatibility gate | not_started | 2026-08-21 | | | 不支持则 runtime blocked，模型不回退。 |
| 27 | 周末：GRPO/on-policy | not_started | 2026-08-22 | | | |
| 28 | 周末：slime debug/replay/repro | not_started | 2026-08-23 | | | |
| 29 | Verified-runtime min loop/reward/replay | not_started | 2026-08-24 | | | 只用 Day 26 通过的 runtime；否则保留 ms-swift evidence。 |
| 30 | Qwen3.5 Base→SFT→DPO/GRPO clean reproduction | not_started | 2026-08-25 | | | 同时展示 v1/v2 lineage boundary。 |

## Qwen3.5-4B Policy Capstone + Deferred Teacher Extension

Day 31–42 无固定日期，在 30-Day Core 后按 readiness 与预算执行；不改变 `2026-08-25` Core 目标日期。

| Day | 主题 | 状态 | 日期 | 用时 | 核心产物 | 一句话结论 |
|---:|---|---|---|---:|---|---|
| 31 | Freeze Qwen3.5 S0/domain eval/teacher-null charter | not_started | unscheduled | | | |
| 32 | Qwen3.5 single/TP2 parity + full/LoRA/QLoRA capacity | not_started | unscheduled | | | |
| 33 | Teacher TP SFT gate template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 34 | Teacher SFT/T1 selection template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 35 | Teacher domain-RL readiness template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 36 | Teacher RL/T2 freeze template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 37 | S1→direct coding RL→S2 | not_started | unscheduled | | | 独立于 teacher 分支。 |
| 38 | Teacher-trace cold-start template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 39 | OPD one-update/scoring/replay template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 40 | Controlled OPD/S3 template | deferred | unscheduled | | | Requires separate teacher charter v2. |
| 41 | S0/S1/S2 matched eval/cost accounting | not_started | unscheduled | | | |
| 42 | S1/S2 clean reproduction/final report | not_started | unscheduled | | | |

## 每周 Gate

- [x] Week 1：能从一个 training step 说明数据、模型、优化器、checkpoint 与 eval 的职责边界；能把 sharding 映射到 OOM、batch 和吞吐问题。（guided quiz/review evidence；runtime evidence 后补。）
- [x] Week 2（v1）：完成数据合同、Base baseline、tiny-overfit/exact resume 与受控 SFT；逐样本 eval 的合法选择结果为“无 eligible checkpoint”，未消费 frozen test。
- [ ] Week 3（v2）：完成 Qwen3.5 processor/runtime/data/eval 迁移、可验证 resume、S1 promotion、单变量优化与 failure injection。
- [ ] Week 4（v2）：审计 coding preference data，从 S1 跑通 DPO 与 coding GRPO，并画出在线 RL 数据和状态流。
- [ ] Final：用已验证 runtime 从干净环境复现 Qwen3.5 Base→SFT→DPO/GRPO，并用证据讲清 lineage、状态、Eval、RL 和失败归因。
- [ ] Policy Capstone Week 5：冻结 Qwen3.5 S0，single/TP2 parity 通过，依据实测选择 full/LoRA/QLoRA；teacher 仍可保持 deferred。
- [ ] Policy Capstone Week 6：从 S1 运行 direct coding RL，完成 S0/S1/S2 一次性 confirmation、能力/成本对照与 clean reproduction；Teacher/OPD 仅在 charter v2 后另行验收。
