# Bootcamp Progress

开始日期：`2026-07-27`  
目标完成日期：`2026-08-25`  
当前 Phase：`Day 22 已按 experimental AI-assisted 路径关闭；下一步 Day 23 Qwen3.5 coding DPO smoke`

当前最大阻塞：`Day 23 尚未执行 DPO loss/mask/reference-policy memory preflight；Day 22 experimental manifest 已冻结 200 个 on-policy、non-synthetic、E2B/processor 可重放 pairs并通过独立 validator。Formal human-review 路径仍 pending，但不再阻塞用户授权的 experimental 主线。`

状态使用：`not_started`、`in_progress`、`blocked`、`deferred`、`done`、`done_with_handoff_pending`、`closed_pass`、`closed_experimental`、`closed_superseded`、`closed_no_candidate`。`done_with_handoff_pending` 表示当日学习与决策目标已完成，但其下游可消费包仍有明确交付 gate；它不解锁依赖项。`closed_pass` 表示原协议及 closeout audit 均通过，不再重跑；`closed_experimental` 表示用户明确接受记录完整的协议偏离，实验下游可消费但不得宣称 formal pass；`closed_superseded` 表示原任务不再重跑、由后续更强证据关闭，但不等于原协议逐项 `pass`；`closed_no_candidate` 表示预注册 fail-closed 路径已产生合法的零候选结果。`deferred` 表示需要新决策/章程，不是当前主线阻塞。Day 01–03 的 `done` 来自用户确认；未据此补写不存在的 artifact 或用时。

## 活动模型迁移

| 项目 | 状态 | 说明 |
|---|---|---|
| `qwen3-0.6b-day01-12-v1` | closed / immutable | Day 11 tiny-overfit pass；Day 12 recovery C–L 10/10 完成、0 eligible checkpoint、frozen test 未消费。 |
| `qwen35-4b-day13-plus-v2` | RSI v0002 confirmed-qualified / S1 downstream-ready | 历史 Day 16 no-candidate 不改写；后续独立 charter 从同一 Base 训练 main，Primary `81/112`、独立训练 seed Confirmation `73/112`，两次均过全部 fixed-full112 gates；winner handoff 已完成。 |
| v2 training scope | selected | text-only coding；完整 processor/conditional-generation loader；vision tower 与 aligner 冻结并做 trainable coverage 断言。 |
| v2 `S1` | downstream-ready | `main-s20260809-lr1e-4-final` 已完成 immutable AutoDL checkpoint archive、winner-only merged export、两个 fresh process 的 `4/4` exact token-ID parity，以及 self-hashed promotion manifest/downstream key。 |
| DPO / GRPO | S1 parent gate passed / downstream gates remain | 只能消费正式 S1 merged export 与 downstream key；Base、Day 12 0.6B export、裸 adapter URI 或不合格候选仍不得替代。 |
| Teacher / OPD | deferred / unselected | `teacher_model_id=null`；不自动选择 8B/9B teacher。 |

迁移的完整 gate 与机器可读状态见 [`QWEN35-4B-MIGRATION-PLAN.md`](QWEN35-4B-MIGRATION-PLAN.md) 和 [`artifacts/configs/qwen35-4b-migration-contract.json`](artifacts/configs/qwen35-4b-migration-contract.json)。

关闭边界：Day 18–20 v1 合并证据关闭 Day 15 onboarding；旧 Day 20 LoRA probe evidence 另被接受为 Day 16 的 no-candidate 退出证据，这两个历史结果不追溯改写。`2026-08-12` 的 RSI v0002 是后续批准的新 charter：它解除 no-candidate readiness block并完成 fixed-full112 selection/confirmation；`2026-08-13` 的 append-only Day 21 handoff 进一步完成 checkpoint archive、merged export/parity 与可下游消费的 S1 manifest/key，没有改写旧账本。Day 17 exact-resume 仍未运行，现为按诊断需要执行的 Optional R，不阻塞 S1。详见 [`Day 15 Close`](artifacts/reports/day15-close.md)、[`Day 16 Gap Audit`](artifacts/reports/day16-gap-audit.md)、[`Day 17 Gap Audit`](artifacts/reports/day17-gap-audit.md)、[`RSI v0002 result`](rsi-control/versions/rsi-v0002/RSI-V0002-RESULT.md) 与 [`S1 handoff`](artifacts/reports/day21-qwen35-s1-handoff.md)。

## 提前执行的 Standalone Experiments

| 实验 | 状态 | 日期 | 核心产物 | 结论 |
|---|---|---|---|---|
| Day 18 Megatron compatibility | closed_pass | 2026-08-08 run；2026-08-10 close | compatibility report、`artifacts/reports/day18-close.md`、closeout JSON | C0–C5 与本地 closeout audit 通过；只覆盖冻结的 2×H800 text-only compatibility/learnability envelope，0 追加 GPU。 |
| [Day 19 Qwen3.5 Full-SFT comparison](day-19-qwen35-sft-comparison/README.md) | done / diagnostic only | 2026-08-08–09 | A/B/E comparison、Qwen3.5 response/HumanEval adapters、root-cause report | 确认评测合同错位与真实 Full-SFT 退化同时存在；没有 eligible/promoted S1。 |
| [Day 20 balanced LoRA probes](day-20-qwen35-balanced-lora-sft/README.md) | closed / no passing probe | 2026-08-09 | Base + `1e-5/3e-5/1e-4` probes、静态/v2 diagnostic report | 三档 probe 均有与 E2B 无关的必要门禁失败；作为 Day 16 no-candidate close evidence，main/PASS/S1 均未发生。 |
| [RSI v0002 / Day 20 v3 target-boundary SFT](rsi-control/versions/rsi-v0002/RSI-V0002-RESULT.md) | done selection / Day 21 handoff complete | 2026-08-12–13 | RSI state/metrics、Primary selection、Confirmation、Final promotion、archive audit、S1 handoff report | Primary `main-s20260809-lr1e-4-final` 通过 frozen full112；同 recipe 的第二个训练 seed checkpoint 也过门。旧 RSI `merge_performed=false` 记录保持不变；新的 append-only Day 21 evidence 已完成 merged downstream-ready S1。 |

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
| 15 | Qwen3.5 onboarding/migration acceptance | closed_superseded | 2026-08-09（提前关闭） | | `artifacts/reports/day15-close.md` | Day 18–20 已用更强实际运行证据消解 onboarding 风险；原 7,860 条 manifest、新 split 与 LoRA exact resume 未执行且不补造；无 S1。 |
| 16 | Controlled coding LoRA SFT/packing parity/candidate set | closed_no_candidate | 2026-08-09（提前收束） | | `artifacts/reports/day16-gap-audit.md`、trajectory、packing、no-anchor JSON | Day 20 三档 16k-token/103-step probe 均失败必要门禁；main 未启动，packing=false，无 provisional anchor。 |
| 17 | Exact checkpoint resume/repro | deferred | 2026-08-12 audit | | `artifacts/reports/day17-gap-audit.md`、resume status、readiness JSON | selected/resumable checkpoint 已由 RSI v0002 产生；exact-resume comparator 仍未运行，仅在有诊断价值时作为 Optional R 执行，不阻塞 S1。 |
| 18 | Megatron min codepath + Qwen3.5 GDN/loader gate | closed_pass | 2026-08-08 run；2026-08-10 close | | compatibility report、`artifacts/reports/day18-close.md`、closeout JSON | C0–C5 与本地 closeout audit 通过；0 追加 GPU。仅关闭冻结 envelope，不等于 LoRA exact resume，也不晋级 S1。 |
| 19 | Optimizer/LR stability + failure injection | not_started | 2026-08-14 | | | RSI v0002 的冻结 recipe/checkpoints 可作为新 baseline；本日 failure-injection 课程尚未执行。 |
| 20 | 周末：Training failure signatures + optional exact-resume lab | not_started | 2026-08-15 | | `day-20-weekend-training-failures/README.md` | Core 为 60 分钟 CPU signature map；Optional R 默认跳过，仅在 selected/resumable candidate 存在且诊断有价值时使用同一单卡跑 3–5h，不是 S1 硬 gate。 |
| 21 | Qwen3.5 candidate selection / fixed-suite qualification | done | 2026-08-13（提前；原计划 08-16） | | `day-21-weekend-eval-reading/README.md`、`artifacts/reports/day21-qwen35-s1-handoff.md`、promotion/key | fixed-full112 selection、independent-training-seed same-suite confirmation 与 downstream-ready S1 handoff 完成；final 是 policy-selected winner，不声明统计唯一最优。 |
| 22 | Coding preference provenance/processor/held-out | closed_experimental | 2026-08-13 执行、08-14 关闭（原计划 08-17） | | formal machine bundle + 10 human/90 sub-agent audit + 11-case Codex adjudication + experimental close manifest | 200 个 on-policy/non-synthetic pairs、split 154/17/29 与全部 machine gates PASS；11/11 争议方向支持 verifier-chosen。Experimental DPO ready；formal-human 未声明通过。 |
| 23 | Qwen3.5 coding DPO smoke（parent=S1） | not_started | 2026-08-18 | | | 直接消费 Day 22 `completed_experimental_ai_assisted` manifest；先做 loss/mask/reference-policy memory preflight，再启动受限 smoke。 |
| 24 | Coding online-RL dataflow/sandbox reward contract | not_started | 2026-08-19 | | | |
| 25 | Qwen3.5 coding GRPO lab（parent=S1） | not_started | 2026-08-20 | | | `downstream-ready S1 manifest` parent gate 已通过；仍须满足在线数据流/reward/runtime gate，首轮显式 cap 8K。 |
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
- [ ] Week 3（v2）：完成 Qwen3.5 processor/runtime/data/eval 迁移、checkpoint integrity、S1 promotion、单变量优化与 failure diagnosis；exact-resume parity 为 Day 20 Optional R。
- [ ] Week 4（v2）：审计 coding preference data，从 S1 跑通 DPO 与 coding GRPO，并画出在线 RL 数据和状态流。
- [ ] Final：用已验证 runtime 从干净环境复现 Qwen3.5 Base→SFT→DPO/GRPO，并用证据讲清 lineage、状态、Eval、RL 和失败归因。
- [ ] Policy Capstone Week 5：冻结 Qwen3.5 S0，single/TP2 parity 通过，依据实测选择 full/LoRA/QLoRA；teacher 仍可保持 deferred。
- [ ] Policy Capstone Week 6：从 S1 运行 direct coding RL，完成 S0/S1/S2 一次性 confirmation、能力/成本对照与 clean reproduction；Teacher/OPD 仅在 charter v2 后另行验收。
