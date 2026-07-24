---
id: AV2-W08
title: Optimization / Auto-skill 自动化迭代链路
status: P0 / active-service-design
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-11
updated_at: 2026-06-16
---

# AV2-W08 Optimization / Auto-skill 自动化迭代链路

## Scope

追踪自动化 optimization 迭代链路：case/trajectory 收集、问题诊断、候选改动生成、trajectory -> skill 生成、eval 执行、决策、回灌到 prompt/tool/model/skill。2026-06-08 新任务是先根据 trajectory 生成 skill，用无验证方式看是否有初步效果提升；当前最高优先级是设计并完成整个 auto-skill pipeline，最终做成线上 service，对各个场景做 skill 细化和 service 化。benchmark pipeline 的新输入方向是 URL 高频站点 -> 任务类别 taxonomy，后续可作为 auto-skill 场景来源。2026-06-16 优先级更新：auto-skill pipeline 升为唯一 P0，模型 harness 因公司侧暂不支持 Qwen 先后置。

## Why It Matters

这条线是把一次性效果调优变成可重复系统的关键。它应该帮助你更快定位问题、更稳定比较改动、更少依赖手工记忆。

## Pipeline Link

主表见：[Optimization automation runs](../runs/optimization-automation-runs.md)

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Run reproducibility | scripts / logs | 同一输入能复现结果 |
| Diagnosis usefulness | reviewer notes | 能把问题归到 prompt/tool/model/skill/UI/infra |
| Candidate quality | eval | 候选改动有可验证提升 |
| Cycle time | run logs | 从 case 到 decision 的时间下降 |
| Reuse | merged changes / eval set | 输出能回流到 agent_v2 workstreams |
| Auto-skill uplift | trajectory experiment | 无验证生成 skill 后是否有可见效果提升 |
| Serviceability | scenario coverage / pipeline design | 生成的 skill 能按场景细化，并具备后续 service 化可能 |
| Online service readiness | service design / rollout | pipeline 有清晰接口、状态、版本、回滚和线上运行边界 |
| Scenario mining quality | URL/site taxonomy + benchmark output | 高频站点能稳定映射到任务类别，并能进一步映射到 skill / tool / benchmark |
| Priority alignment | dashboard / weekly review | auto-skill pipeline 是当前最高优先级，其他工作围绕它提供场景、case 或能力组件 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-16 | recent-master, tab-web | 优先级更新：auto-skill pipeline 成为当前最高优先级；高速模型 harness 因公司暂不支持 Qwen 后置 | 把注意力集中到可推进的 pipeline/service 化工作，减少卡在模型平台支持上的等待成本 | dashboard priority updated | P0 | focus here | 优先拆 pipeline stage、service API、状态机、版本/回滚、benchmark taxonomy 接入 |
| 2026-06-08 | recent-master, tab-web | benchmark pipeline 方向接入 auto-skill：URL 高频站点 -> 任务类别 -> skill 解决 | 高频站点和任务类别 taxonomy 可以作为 auto-skill service 的场景来源，让 skill 生成更贴近真实用户任务族 | benchmark pipeline 已 assign 给老板；taxonomy 待产出 | scoping | use as scenario input | 和 AV2-W13 对齐字段：site frequency、task category、skill candidate、coverage gap、eval case |
| 2026-06-08 | recent-master, tab-web | 主要 priority 更新：设计并完成整个 auto-skill pipeline，最终做线上 service | 把 trajectory -> skill 从单次实验推进到线上 service，可以让不同场景的 skill 生成、细化、维护和回滚进入稳定流程 | 待定义 pipeline stage、service API、状态/版本、验证策略 | first-priority | design and build service pipeline | 画 pipeline、拆服务接口、定义场景 taxonomy、版本/状态/回滚和验证/无验证分支 |
| 2026-06-08 | recent-master, tab-web | 新任务：trajectory -> 生成 skill，先无验证看效果提升 | 从真实 trajectory 抽取技能模式并生成 skill，可能在没有完整验证 pipeline 前先带来效果提升信号 | 待观察：生成 skill 可用性、效果 uplift、失败样式 | experimenting | run no-validation trial | 跑第一版 trajectory -> skill，记录样本、生成结果、效果信号；规划 auto-skill pipeline |
| 2026-05-14 | recent-master, tab-web | 识别后续可能评估后端直接给 URL -> create skill 的流程 | 如果 skill 创建入口能从 URL 自动化，可能减少手工准备和实习生 run 的工程摩擦 | 待基于今日 skill run 卡点和回归结果评估 | pending | scope later | 估算开发时间、依赖、风险和工程影响面 |
| 2026-05-12 | recent-master, tab-web | agent_v2 进入回归评估阶段 | 回归结果可以沉淀为自动化优化链路的核心输入 | 待收集：分数、轮次、时间、常规路径偏离、doc/sheet 效果 | pending | keep iterating | 将回归 case 结果写入 run ledger |
| 2026-05-11 | TBD | Tracking created | 把自动化优化拆成 pipeline 阶段，便于定位瓶颈 | TBD | TBD | TBD | 补当前 pipeline 和最近一轮 run |

## Open Risks

- 自动化生成的改动如果缺少 eval guardrail，容易优化到局部指标。
- 诊断标签不稳定时，后续统计会失真。
- 如果 run 输出不能直接链接到 workstream，改动落地会断。
- trajectory -> skill 的第一版无验证实验只能看初步效果信号，不能替代稳定 eval。
- auto-skill pipeline 如果没有场景 taxonomy 和 service 边界，后续会难以维护、复用和回滚。
- URL 高频站点 -> 任务类别 taxonomy 如果不能映射到具体 skill/tool/eval case，就只能成为分析表，不能驱动 auto-skill service。
- auto-skill 做成线上 service 后，如果没有版本、状态机、回滚和审计字段，生成出的 skill 很难运营。
- 当前 auto-skill 是唯一 P0，如果继续被生图、小红书、模型 harness 等支线打断，pipeline/service 化会被拖散。

## Next Actions

- [ ] 画出当前 auto-skill pipeline 的真实阶段：input、scenario mining、skill generation、verification/no-verification、publish、service。
- [ ] 记录本轮回归输入、输出、结论。
- [ ] 定义 issue taxonomy：prompt / tool / model / skill / UI / infra / data。
- [ ] 记录最终评估分数、轮次、时间。
- [ ] 确认哪些输出能自动回流到 eval 或 PR。
- [ ] 评估 URL -> create skill 后端流程的开发量和工程影响。
- [ ] 跑第一版 trajectory -> skill 无验证实验。
- [ ] 记录生成 skill 的输入 trajectory、输出 skill、初步效果信号和失败样式。
- [ ] 设计 auto-skill pipeline：场景 taxonomy、生成策略、验证/无验证分支、service 化边界。
- [ ] 接入 AV2-W13 benchmark taxonomy：URL 高频站点、任务类别、skill candidate、coverage gap、eval case。
- [ ] 拆 auto-skill online service：API、状态机、版本、发布/回滚、日志和权限边界。
- [ ] 明确各场景的 skill 细化策略和 owner/reviewer。
