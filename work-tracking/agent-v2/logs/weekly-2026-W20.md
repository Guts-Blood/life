---
period: 2026-W20
date_range: 2026-05-11 to 2026-05-17
created_at: 2026-05-11
---

# Weekly Review: Agent V2 / 2026-W20

## What Moved

| Workstream | Progress | Evidence |
| --- | --- | --- |
| AV2-W01 SP prompt | 主要改动已推，进入回归验证 | 待补 PR / eval link |
| AV2-W02 Tool description | 主要改动已推，待回归 tool path | 待补 PR / eval link |
| AV2-W03 New tools | 新增复制产物->粘贴一体化工具 | 待补 PR / case result |
| AV2-W04 Old tool click feedback | 主要改动已推，待验证 feedback 信号和老 case path | 待补 trace / event evidence |
| AV2-W05 doc/sheet skill orchestration | skill 更新新 SOP，增加更多场景 router；表格 skill 初步效果不错，测试回归未跑完 | 待补 master merge PR / commit 和 doc/sheet 回归结果 |
| AV2-W06 GLM model + prompt | 修复 UID 模型幻觉和输出格式 bug，加工程兜底 | 待补 eval / 回归结果 |
| AV2-W07 Intern skill runs | 实习生工作已基本 align：跑 skill 找工程卡点；今日 24 个 skill run 目标 | 待收 batch 输出和工程卡点分类 |
| AV2-W08 Optimization automation | 回归阶段启动；后续可能评估后端 URL -> create skill 流程 | 待补 regression run 和工程影响评估 |

## Decisions

| Decision | Why | Owner | Follow-up |
| --- | --- | --- | --- |
| 2026-05-14 实习生工作定位为跑 skill 找工程卡点 | 对齐职责后，case 输出更容易回流成可执行工程问题 | 我 + 实习生 | 收集 24 个 case 输出，评估 URL -> create skill 后端流程 |
| 2026-05-14 今日先处理 master merge、AI paper catch-up、实习生 tracking | 三件事分别对应上线合入、个人信息补课、团队执行跟进 | 我 | 建 daily todo，分别更新 AV2-W05、research、AV2-W07 |
| 本阶段从开发推进切到回归验证 | 主要改动已推完，下一步风险在路径偏离和效果回退 | 我 | 跑常规 case、doc/sheet case，记录分数/轮次/时间 |
| 实习生开始跑 skill 任务 | 增加 skill 场景覆盖和问题发现速度 | 我 + 实习生 | 收集 batch 输出并做问题分类 |

## Evidence Collected

| Evidence | Linked Workstream | Takeaway |
| --- | --- | --- |
| 2026-05-14 进展更新 | AV2-W05-W08 | 实习生工作已基本 align；UID 幻觉/输出格式 bug 已修并加兜底；表格 skill 初步效果不错，回归未完成 |
| 2026-05-14 todo 记录 | AV2-W05, AV2-W07, research | 今日新增 master merge、AI 动向/paper catch-up、实习生 tracking |
| 2026-05-12 进展记录 | AV2-W01-W08 | 主要改动已推；新增复制产物->粘贴一体化工具；skill 更新新 SOP 和更多场景 router |

## Risks / Blockers

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| 常规 case 路径偏离 | 影响上线可信度，即使总分看起来不错也可能有体验回退 | 逐 case 记录 path drift、tool call、rounds、time | 我 |
| doc/sheet 场景效果不稳定 | 高价值场景会影响 agent_v2 上线质量 | 单独跑表格/文档场景，记录产物质量和触发准确率 | 我 |
| 多项改动同时变化，难归因 | prompt/tool/model/skill/router 混在一起会增加 debug 成本 | 回归记录 issue type，必要时拆分回滚或 A/B | 我 |

## Rest Of Week Focus

- [ ] Merge tabhobor 上传的 skill 新改动到 master branch。
- [ ] 补最近 AI 动向和 paper catch-up。
- [x] Track 实习生今日进度、输出、阻塞、下一步。
- [ ] 跑常规 case，确认路径是否偏离。
- [ ] 跑表格/文档场景，确认 skill 新 SOP 和 router 效果。
- [ ] 记录最终评估分数、轮次、时间。
- [ ] 收实习生 skill run 输出并归类问题。
- [ ] 评估后端 URL -> create skill 流程开发量和工程影响。
- [ ] 根据回归结果决定 keep / iterate / rollback。
