# Day 42 — S1/S2 Clean Reproduction、Failure Review 与 Report

状态：`deferred_after_architecture_study`
日期：`unscheduled_after_day30`
强度：4–5 小时；最小 GPU replay + CPU 报告

## 当前状态

Day 27–30 已转向 Megatron/slime architecture study。本 clean reproduction 依赖已暂停的执行型 Capstone，不是当前毕业要求。

## 主要目标

证明 capstone policy charter v1 不是一次性手工成功：从干净环境解析 S0/S1/S2 manifests，恢复训练 checkpoint，重放一个 direct coding RL batch，重算一条 coding E2E trajectory，并形成可评审的最终报告。Teacher/OPD extension 保持 deferred，不是当前复现要求。

## 理论 / 定向阅读（30–45 分钟）

精读清单：[Day 42 — Clean Reproduction](../SCALING-BOOK-READING-GUIDE.md#day-42)。

## Clean reproduction

在新 shell/镜像中，不读取旧 shell history：

1. 校验 code/container/model/tokenizer/data/environment/scorer hashes；
2. 解析 S0/S1/S2 manifests、processor/loader/modality/module-coverage contract 与 GPU placement；
3. 恢复一个 S1 或 S2 training checkpoint 并导出 inference weights；
4. 从 Day 37 persisted batch 重算 policy old/current log-prob、reward components、mask 和 direct-RL loss；
5. 对一条 frozen coding trajectory 重放 extraction、sandbox/tests 和 scorer；
6. 对比原 run 与 clean run 的 token IDs、versions、loss/score 和 artifact hashes。

## Failure review

至少复盘：

- 一个 TP/collective/checkpoint 问题；
- 一个 rollout/reward/weight-version 问题；
- 一个 processor/render/module-coverage 或 policy loss-mask 问题；
- 一个 eval/selection/contamination 风险；
- 哪些问题在历史 0.6B smoke 中不会出现、为什么 Qwen3.5-4B hybrid/multimodal checkpoint 才暴露。

## 最终报告必须回答

1. S1 coding SFT 是否相对 S0 提升可执行任务能力？
2. S2 direct coding RL 是否相对 S1 提升 E2E success，且 guardrails 未回退？
3. 差异来自训练信号、数据/rollout budget 还是 scorer/sandbox 偏差？
4. Full/LoRA/QLoRA 与 single/TP2 的实际容量、吞吐和可恢复性证据是什么？
5. Rollout、sandbox/reward、weight sync 和 checkpoint 的主要系统瓶颈是什么？
6. 下一轮应继续 direct RL、调整 coding SFT/RL protocol，还是停止？

Teacher advantage、S3 retention、S2↔S3 和 OPD+RL 只在未来 charter v2 激活后作为独立 appendix 问题，不得在当前报告中用假设作答。

## Evidence-first 产物

- `../artifacts/reports/capstone/capstone-final-report.md`
- `../artifacts/reports/capstone/capstone-clean-reproduction.md`
- `../artifacts/reports/capstone/capstone-evidence-index.md`
- clean replay configs/logs 与 final checkpoint graph

## 最终验收

- [ ] [`../SCALED-TEACHER-STUDENT-CAPSTONE.md`](../SCALED-TEACHER-STUDENT-CAPSTONE.md) 的所有 Core 验收项有直接 artifact evidence。
- [ ] Clean environment 能恢复 S1/S2 checkpoint、replay direct-RL batch、重算 coding E2E score。
- [ ] 报告没有把 training reward、teacher imitation 或 aggregate score单独当成能力证明。
- [ ] 能解释 coding SFT 与 direct RL 各自改变了什么信号。
- [ ] 能说明 TP、DP、rollout replicas 与 global batch/资源成本的关系。
- [ ] Teacher/OPD 节点仍为 deferred_unselected；未把未执行的 extension 冒充失败或完成。
- [ ] 未完成或失败项明确标记，不以最终模型文件存在冒充实验完成。
