---
id: optimization-automation-runs
title: Optimization 自动化迭代链路
created_at: 2026-05-11
updated_at: 2026-06-16
---

# Optimization Automation Runs

## Pipeline Map

| Stage | Purpose | Current State | Owner | Notes |
| --- | --- | --- | --- | --- |
| Case selection | 选择需要优化的任务/失败样本 | active | 我 | 本轮重点：常规 case 路径、表格/文档场景 |
| Diagnosis | 判断问题来自 prompt、tool、model、skill、UI 还是 infra | active | 我 | 今日新增 UID 幻觉/输出格式、skill 工程卡点和 doc/sheet 回归线索 |
| Trajectory extraction | 从真实 trajectory 抽取可复用模式 | active | 我 | 2026-06-08 新增：作为 skill 生成输入 |
| Scenario mining | 从 URL 高频站点和任务类别抽 auto-skill 场景 | scoping / delegated | 老板 / 我 | benchmark pipeline 已 assign 给老板；输出回流 auto-skill taxonomy |
| Candidate generation | 生成可测试的 prompt/tool/strategy/skill 改动 | active | 我 | trajectory -> skill 第一版先无验证看效果提升 |
| Eval execution | 跑对比实验或回归 case | deferred | 我 | 第一版无验证；后续 auto-skill pipeline 需要补验证分支 |
| Decision | keep / revert / iterate | pending | 我 | 依据回归结果判断上线/迭代 |
| Rollout / service | 合入、灰度、上线或同步到 repo/service | P0 / active | 我 | 2026-06-16 当前最高优先级：auto-skill pipeline 最终做成线上 service |

## Run Ledger

| Date | Run ID | Input Cases | Candidate Change | Target Workstream | Eval Result | Decision | Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-16 | priority-focus-auto-skill-2026-06-16 | auto-skill pipeline + Qwen support constraint | 聚焦 auto-skill pipeline，模型 harness 因公司暂不支持 Qwen 后置 | AV2-W08 / AV2-W16 | priority updated | focus AV2-W08 | 拆 pipeline stage、service API、状态机、版本/回滚、benchmark taxonomy 接入 |
| 2026-06-08 | url-site-task-taxonomy-2026-06-08 | URL 高频站点 / 任务类别挖掘 | 用 URL 高频站点 -> 任务类别 taxonomy 作为 auto-skill 场景来源 | AV2-W08 / AV2-W13 | taxonomy 待产出 | delegated / scoping | 跟老板对齐字段、样本、skill candidate、coverage gap 和 eval case |
| 2026-06-08 | auto-skill-service-design-2026-06-08 | trajectory -> skill 实验 + 场景 skill 需求 | 设计完整 auto-skill pipeline 和线上 service | AV2-W08 | design pending | first-priority | 拆 pipeline stage、service API、状态机、版本/回滚、验证策略 |
| 2026-06-08 | trajectory-skill-2026-06-08 | 真实 trajectory 样本 | 根据 trajectory 生成 skill，第一版无验证 | AV2-W08 | 待观察效果提升 | experimenting | 记录输入 trajectory、输出 skill、效果信号；规划 auto-skill pipeline |
| 2026-05-14 | engineering-scope-2026-05-14 | 实习生 skill run 工程卡点 + doc/sheet 回归 + UID 格式 bug | 可能新增后端 URL -> create skill 流程 | AV2-W05-W08 | pending | pending | 估算开发时间、依赖、风险和工程影响面 |
| 2026-05-12 | regression-2026-05-12 | 常规 case 路径 + 表格/文档场景 | agent_v2 已推改动：prompt/tool/skill/router/model 相关 | AV2-W01-W08 | pending | pending | 记录 score、rounds、time、path drift |
| 2026-05-11 | TBD | TBD | TBD | AV2-W08 | TBD | TBD | 补第一轮真实 run |

## Bottlenecks

| Bottleneck | Why It Matters | Evidence | Next |
| --- | --- | --- | --- |
| 回归记录需要统一字段 | 否则分数、轮次、时间和路径偏离难以比较 | 2026-05-12 回归阶段启动 | 统一记录 score / rounds / time / path drift / issue type |
| URL -> create skill 流程待评估 | 如果后端入口能直接把 URL 变成 skill，可能减少人工准备和实习生 run 摩擦 | 2026-05-14 实习生工作定位为跑 skill 找工程卡点 | 估算开发量、依赖、失败态和对现有 skill pipeline 的影响 |
| 无验证 trajectory -> skill 只能看趋势 | 可以快速探索效果，但无法确认稳定性和泛化 | 2026-06-08 新任务 | 记录样本、初步效果和失败样式，后续补 auto-skill 验证分支 |
| auto-skill 线上 service 需要状态和版本 | 没有状态机、版本、回滚和审计，service 化后会难运营 | 2026-06-08 主要 priority | 设计 API、状态字段、发布/回滚和日志边界 |
| URL/task taxonomy 需要可执行映射 | 如果只产出分类表，没有 skill candidate / coverage gap / eval case，难以驱动 auto-skill | 2026-06-08 benchmark pipeline 新方向 | 和 AV2-W13 对齐 taxonomy 字段并回流 AV2-W08 |
| Qwen 支持不足导致模型 harness 卡住 | 公司侧暂不支持 Qwen，继续推进 harness 会卡在平台前置条件 | 2026-06-16 priority update | AV2-W16 后置，把 P0 集中到 auto-skill pipeline |

## Reusable Outputs

| Output | Where Used | Link | Notes |
| --- | --- | --- | --- |
| 回归结果表 | dashboard + workstreams | TBD | 用于判断上线、回滚或下一轮优化 |
| trajectory -> skill 样本 | AV2-W08 + future auto-skill pipeline | TBD | 用于沉淀场景 skill 生成和 service 化 |
| URL 高频站点 -> 任务类别 taxonomy | AV2-W08 + AV2-W13 | TBD | 用于沉淀真实任务族、skill candidate 和 benchmark coverage gap |
