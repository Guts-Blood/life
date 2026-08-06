# Day 42 — Clean Reproduction、Failure Review 与 Capstone Report

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；最小 GPU replay + CPU 报告

## 主要目标

证明 capstone 不是一次性手工成功：从干净环境解析 manifests，恢复 distributed checkpoint，重放一个 OPD batch，重算一条 E2E trajectory，并形成可评审的最终报告。

## 理论 / 定向阅读（30–45 分钟）

精读清单：[Day 42 — Clean Reproduction](../SCALING-BOOK-READING-GUIDE.md#day-42)。

## Clean reproduction

在新 shell/镜像中，不读取旧 shell history：

1. 校验 code/container/model/tokenizer/data/environment/scorer hashes；
2. 解析 T2/S1/S2/S3 manifests 与 GPU placement；若使用 recovery，再解析 S1d/S3d；
3. 恢复一个 TP/distributed checkpoint 并导出 inference weights；
4. 从 Day 39 persisted batch 重算 teacher/student log-prob、mask 和 OPD loss；
5. 对一条 frozen trajectory 重放 tool environment 和 scorer；
6. 对比原 run 与 clean run 的 token IDs、versions、loss/score 和 artifact hashes。

## Failure review

至少复盘：

- 一个 TP/collective/checkpoint 问题；
- 一个 rollout/reward/weight-version 问题；
- 一个 teacher/student alignment 或 distillation signal 问题；
- 一个 eval/selection/contamination 风险；
- 哪些问题在 0.6B smoke 中不会出现、为什么 scale-up 才暴露。

## 最终报告必须回答

1. 8B teacher 是否真的优于 student anchor？
2. S3 保留了多少 teacher domain capability？
3. S3 是否优于 matched student direct-RL S2？
4. 差异来自能力、训练预算还是额外 teacher compute？
5. TP/rollout/teacher scoring 的主要系统瓶颈是什么？
6. 下一轮应选 direct RL、OPD、OPD+RL 还是停止？

## Evidence-first 产物

- `../artifacts/reports/capstone/capstone-final-report.md`
- `../artifacts/reports/capstone/capstone-clean-reproduction.md`
- `../artifacts/reports/capstone/capstone-evidence-index.md`
- clean replay configs/logs 与 final checkpoint graph

## 最终验收

- [ ] [`../SCALED-TEACHER-STUDENT-CAPSTONE.md`](../SCALED-TEACHER-STUDENT-CAPSTONE.md) 的所有 Core 验收项有直接 artifact evidence。
- [ ] Clean environment 能恢复 checkpoint、replay OPD batch、重算 E2E score。
- [ ] 报告没有把 training reward、teacher imitation 或 aggregate score单独当成能力证明。
- [ ] 能解释 direct RL、offline cold start、OPD 各自改变了什么信号。
- [ ] 能说明 TP、DP、rollout replicas、teacher replicas 与 global batch/资源成本的关系。
- [ ] 未完成或失败项明确标记，不以最终模型文件存在冒充实验完成。
