---
id: AV2-W13
title: Benchmark 深度 Case 建设
status: delegated / taxonomy-design
owner: 老板（pipeline）/ 我（tracking & skill mapping）
repos:
  - recent-master
  - tab-web
created_at: 2026-05-21
updated_at: 2026-06-16
---

# AV2-W13 Benchmark 深度 Case 建设

## Scope

建设更多、更深度、可复现、可归因的 agent v2 benchmark case，用于比较 GLM、Gemini 3.5、Sonnet baseline、deepresearch skill、doc/sheet skill 和 tool/prompt 迭代。2026-06-08 最新状态：benchmark pipeline 任务已 assign 给老板，当前思路是从 URL 高频站点 -> 任务类别挖掘分类，再用 skill 去解决这部分任务族。

## Why It Matters

模型和 skill 的迭代速度会越来越快，如果 benchmark 只覆盖浅层 happy path，就无法判断真实上线风险。新的 benchmark 要覆盖长链路、多约束、跨工具、artifact 质量、失败恢复、平台工具映射和真实 query 复杂化。

## Benchmark Dimensions

| Dimension | Purpose | Example Signals |
| --- | --- | --- |
| Depth | 覆盖多步推理和长任务坚持度 | rounds、time、goal retention |
| Tool use | 覆盖正确工具选择和参数稳定性 | tool path drift、invalid args、recovery |
| Artifact | 覆盖文档/表格/研究产物质量 | formatting、range、source quality |
| Platform | 覆盖前端 / E2B / 本地文件工具边界 | platform mismatch、wrong surface |
| Research | 覆盖 deepresearch 相关任务 | citation quality、synthesis quality |
| URL / Site frequency | 从高频 URL / 站点挖真实任务族 | site frequency、domain cluster、task source |
| Task taxonomy | 把高频站点映射到任务类别和 skill 解决策略 | task category、skill candidate、coverage gap |
| Site control | 覆盖强封控站点的可执行边界和降级策略 | login/captcha/rate-limit、blocked content、fallback quality |
| Regression | 覆盖主链路不可回退能力 | pass/fail、quality score、latency |

## Pipeline Link

主表见：[Benchmark deep cases](../runs/benchmark-deep-cases.md)

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-15 | recent-master, tab-web | 小红书 skill 迭代后发现站点封控非常强 | 强封控站点需要进入 benchmark taxonomy，否则浏览类 skill 容易只在开放站点上表现好 | manual iteration; high site-control risk | add site-control case | track in AV2-W17 | 把小红书加入 site-control stress case，记录可做/不可做任务和降级策略 |
| 2026-06-08 | recent-master, tab-web | benchmark pipeline 任务 assign 给老板；方向是 URL 高频站点 -> 任务类别 -> skill 解决 | 从真实高频站点和任务类别出发，比单纯手写 query 更容易得到可规模化、可回流 skill 的 benchmark taxonomy | handoff done; taxonomy 待产出 | delegated / scoping | track taxonomy output | 跟老板对齐样本来源、分类字段、任务类别、skill 映射和 benchmark case schema |
| 2026-05-27 | recent-master, tab-web | 完成 query 挖掘 + 用户画像分析 -> benchmark query pipeline | 用用户画像和 query 挖掘生成 benchmark query，可以提高 case 真实性和覆盖深度 | pipeline 已完成，已开始支持 Gemini 3.5 Flash 跑分 | completed / active input | keep expanding | 扩充更多更深度 case，覆盖 image generation tool、script skill、deepresearch skill |
| 2026-05-21 | recent-master, tab-web | 新增 benchmark 深度 case 建设 workstream | 更多更深度 case 能支撑 GLM release、Gemini 3.5、deepresearch skill 的同口径对比 | 待建立 case taxonomy 和第一批 case | pending | scope benchmark | 定义字段、评分维度、case 来源和第一批深度 case |

## Open Risks

- 只增加 case 数量，不增加任务深度和可归因字段。
- case 过于人工构造，脱离真实用户 query。
- benchmark 没有统一记录 score、rounds、time、tool path、artifact quality，导致结果不可比较。
- URL 高频站点只能代表高频需求，如果不加价值/风险权重，可能漏掉低频但重要的 agent 场景。
- 任务类别如果太粗，会难以映射到具体 skill；如果太细，会导致 skill 和 benchmark 都不可维护。
- pipeline 已 assign 给老板后，需要明确 handoff 字段和同步节奏，否则产出很难回流到 AV2-W08 auto-skill。
- 强封控站点如果不纳入 benchmark，浏览 skill 可能在真实高频站点上失败但评估里看不出来。

## Next Actions

- [x] 定义 benchmark case schema：source、task、depth、tools、expected artifact、score、rounds、time、issue type。
- [x] 从 AV2-W09 query pipeline / 用户画像分析抽第一批 benchmark query。
- [x] benchmark pipeline 任务 assign 给老板。
- [ ] 和老板对齐 URL 高频站点抽样来源、频率口径、过滤规则和 owner/reviewer。
- [ ] 设计 URL 高频站点 -> 任务类别 taxonomy：类别名、触发条件、典型 query、成功标准、风险标签。
- [ ] 把任务类别映射到 skill 解决策略：已有 skill、待生成 skill、需要新 tool、仅 benchmark。
- [ ] 加入小红书 site-control stress case：封控样式、任务边界、失败态、降级策略。
- [ ] 从实习生 skill run、deepresearch skill、doc/sheet 场景继续抽 case。
- [ ] 标注核心红线 case和探索型深度 case。
- [x] 建 GLM / Gemini 3.5 Flash / Sonnet 对比表初版，Gemini 当前 SOTA。
- [ ] 写 benchmark 运行和复盘流程。
