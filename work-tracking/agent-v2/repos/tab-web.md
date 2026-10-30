---
id: repo-tab-web
title: tab-web tracking
created_at: 2026-05-11
updated_at: 2026-06-16
---

# tab-web

## Role In Agent V2

主要追踪前端交互、tool 点击 feedback、skill orchestration 展示、跨 repo schema 对齐和体验验证。

## Active Touchpoints

| Workstream | Area | Current Link | State | Notes |
| --- | --- | --- | --- | --- |
| AV2-W01 | SP prompt behavior visible in UX | TBD | validating | 回归常规 case 的用户可见路径是否偏离 |
| AV2-W02 | Tool description behavior | TBD | validating | 回归 tool 选择是否符合 UI 预期 |
| AV2-W03 | New tool UX | TBD | scoping / validating | 新增 UID -> APC detail lookup 工具方向；spreadsheet artifact paste payload 待回归用户可见粘贴结果 |
| AV2-W04 | Old tool click feedback | TBD | validating | 待确认 feedback 信号可用且不改变老 case 路径 |
| AV2-W05 | doc/sheet skill orchestration UX | TBD | validating | doc/sheet skill contracts 已收紧，待回归进度、结果、失败展示 |
| AV2-W07 | Intern skill run review surface | TBD | doing | 21 号前持续 track skill 工作，输出回流 GLM eval / query pipeline |
| AV2-W08 | Auto-skill UX / service surface | TBD | P0 / active-service-design | 当前最高优先级：auto-skill pipeline 要做成线上 service，可能需要场景、版本、状态、发布/回滚展示；URL 高频站点 -> 任务类别 taxonomy 可作为场景来源 |
| AV2-W09 | Query mining -> benchmark query pipeline | TBD | completed / feeding benchmark | query 挖掘 + 用户画像分析 -> benchmark query pipeline 已完成 |
| AV2-W10 | Platform -> tool mapping UX behavior | TBD | scoping | prompt 需要区分用户前端、E2B、用户本地文件，避免 UI 状态和文件状态混用 |
| AV2-W11 | DeepResearch skill UX / artifact | TBD | doing | 已 assign 给实习生调研，待对齐进度、引用、artifact、失败态展示 |
| AV2-W13 | Benchmark deep case review | TBD | delegated / taxonomy-design | benchmark pipeline 已 assign 给老板；前端侧关注 URL/site、任务类别、skill candidate 和 review 字段 |
| AV2-W14 | Image generation tool UX | TBD | daily / input-review-shipped | 生图输入审核已进 daily；后续关注更多 input parameters 展示和 file id 替代 `image_url` |
| AV2-W17 | Xiaohongshu skill UX / site control | TBD | iterating / blocked-by-site-control | 小红书封控非常强，前端/agent 侧需要识别封控态并给出清晰降级 |

## Event / Feedback Tracking

| Event / Signal | Source UI | Payload Fields | Downstream Consumer | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Tool click feedback | TBD | TBD | recent-master | scoping | 确认点击、取消、失败、重试等状态是否都需要 |
| UID -> APC detail lookup | take_snapshot / browser state | uid list, APC detail, budget metadata, stale/missing uid errors | recent-master | scoping | 按少量 UID 渐进式展开网页内容，避免 snapshot 全量 APC 爆上下文 |
| Skill orchestration feedback | TBD | TBD | recent-master | scoping | 确认 doc/sheet skill 的触发和结果反馈 |
| Platform context signal | 用户前端 / E2B / 本地文件 | TBD | recent-master | scoping | 确认模型能否从上下文识别当前平台和对应工具面 |
| Image generation result display | agent artifact surface | image artifact id, preview url, status, metadata, save/edit actions | recent-master | mostly closed / final smoke | 生图结果需要能预览、确认、保存/复用、继续修改，并保留失败态；chat 第一版不展示生图入口 |
| Image generation review failure | agent request / moderation surface | request id, prompt, review result, poresche status, retry state, user-facing fallback | recent-master | historical risk / monitor | 审核链路历史上出现过 `poresche` 偶发失败，需要保留可观测和可恢复入口 |
| Image generation input review | agent request / moderation surface | request id, input prompt, input parameters, file refs, review result, user-facing fallback | recent-master | daily | 生图输入审核已进 daily，需要观察误杀、拦截和用户态文案 |
| Image generation badge | agent artifact surface | badge state, label, tooltip, moderation status, generation status, retry/failure reason | recent-master | scoping | 生图角标需要区分生成中、审核中、完成、审核拦截、生成失败、重试中等状态 |
| Runtime tool loading UX | agent skill/tool surface | tool availability, schema status, loading state, error state, skill version | recent-master | mostly closed / config follow-up | runtime tool 加载小幅重构需要前端同步展示/触发状态 |
| Image config surface | settings / agent artifact surface | config key, default value, UI state, backend schema, error state | recent-master | open | 剩余可配置项需要前后端展示、默认值、schema 和失败态对齐 |
| Image input parameters | settings / agent request surface | parameter schema, default value, validation state, visible label, error state | recent-master | todo | 支持更多 input parameters 时，前端展示和后端 schema/default 需要一致 |
| Image file reference | agent artifact surface | file id, previous image_url, permission state, artifact id, preview url | recent-master | todo | 用 file id 替代 `image_url`，需要保证预览、复用、权限和旧链路兼容 |
| E2B image artifact display | agent artifact surface | e2b artifact id, file path, preview url, permissions, status | recent-master | mostly closed / final smoke | 生图放 E2B 链路后，需要验证产物能稳定预览、保存和复用 |
| Benchmark taxonomy review | benchmark review surface | url/site, frequency, task category, skill candidate, coverage gap, eval case id | recent-master | scoping | URL 高频站点 -> 任务类别 pipeline 需要可 review 字段，方便回流 skill 和 eval |
| Xiaohongshu site-control state | browser / agent trace | block type, login state, captcha state, rate limit, inaccessible content, fallback text | recent-master | scoping | 小红书强封控需要可识别状态和清晰降级提示 |

## Open Checks

- [ ] 点击 feedback 是否有统一事件命名？
- [ ] 老 tool 的 feedback 是否和新 tool 兼容？
- [ ] 用户取消、tool failure、timeout 是否单独记录？
- [ ] 前端展示是否会影响 agent 行为调优判断？
- [ ] recent-master 消费字段是否已经对齐？

## Repo Log

| Date | Change / PR | Workstream | Risk | Validation | Next |
| --- | --- | --- | --- | --- | --- |
| 2026-06-16 | 优先级更新：auto-skill pipeline 成为当前最高优先级 | AV2-W08 | 如果早期没有服务侧/前端状态字段，后续 service 化管理会困难 | priority updated | 优先梳理 skill 场景、版本、状态、发布/回滚和 review 字段 |
| 2026-06-15 | 生图链路新增输入审核并推到 daily | AV2-W14 | 输入审核用户态不清会让用户以为生成失败；误杀需要可观测 | daily pushed; 样本待记录 | 补通过/拦截/误杀样本和用户态文案 |
| 2026-06-15 | 生图后续 todo：更多 input parameters + file id 替代 `image_url` | AV2-W14 | 参数展示、默认值和后端 schema 不一致会导致 UI 误导；file id 迁移不稳会让图片引用断链 | 待设计 schema 和兼容策略 | 对齐参数展示、错误态、file id 预览/复用/权限 |
| 2026-06-15 | 迭代小红书 skill，发现站点封控非常强 | AV2-W17 | UI/agent 如果识别不了封控态，会出现长时间无效重试或不清楚的失败态 | manual iteration; high site-control risk | 记录 block type、login/captcha/rate-limit 状态和降级文案 |
| 2026-06-08 | 生图主链路基本 close，剩可配置项前后端收尾 | AV2-W14 | 配置项如果前端展示、默认值和后端真实调用不一致，用户会看到错误状态或参数 | basic close; config smoke 待补 | 对齐 config surface、默认值、错误态和最终 artifact/badge smoke |
| 2026-06-08 | benchmark pipeline 已 assign 给老板，方向是 URL 高频站点 -> 任务类别 -> skill 解决 | AV2-W13 / AV2-W08 | benchmark review 字段如果缺 URL/site、任务类别和 skill candidate，后续难回流 auto-skill | taxonomy 待产出 | 预留 review 字段：site frequency、task category、skill candidate、coverage gap、eval case |
| 2026-06-08 | 主要 priority：auto-skill pipeline 最终做成线上 service | AV2-W08 | 如果前端/服务侧没有 skill 状态、版本、场景、来源 trajectory 和回滚字段，service 化管理会困难 | 待拆 service UX / 状态字段 | 记录需要的 skill 管理字段和发布/回滚展示 |
| 2026-06-08 | trajectory -> skill 新任务，后续 auto-skill pipeline 可能做场景细化和 service 化 | AV2-W08 | 如果 auto-skill 需要前端管理/展示入口，早期缺少状态字段会影响后续 service 化 | 待观察无验证实验结果 | 记录可能需要的 skill 版本、场景、来源 trajectory 和状态字段 |
| 2026-06-08 | 2W 行生图 PR 已合入 | AV2-W14 | 大 PR 合入后如果前端 artifact、badge、runtime loading 状态未 smoke，用户可见问题会难定位 | merged; smoke 待记录 | 补 E2B artifact display、badge/status、tool availability smoke |
| 2026-06-05 | 生图最新 todo：E2B 链路、runtime tool loading 前端配合、线上配置、`image_gen` skill 上传 | AV2-W14 | 前端如果没有同步 runtime tool loading 和 E2B artifact 状态，会出现后端可用但 UI 不可见/不可操作 | 待验证 tool availability、E2B artifact display、badge/status | 对齐 runtime loading UI 状态，补 E2B artifact 展示和 skill version 展示 |
| 2026-06-02 | 重构 agent 生图链路；确认审核链路和生图角标 | AV2-W14 | 审核/生成/展示状态不同步会导致角标错误或失败态不清楚 | 待验证：角标状态映射、审核失败态、daily 主链路 | 对齐角标文案/样式/状态映射，补审核失败和重试展示 |
| 2026-05-29 | Agent 生图代码进 daily；审核/`poresche` 偶发失败 | AV2-W14 | 审核失败会让用户看不到最终产物或收到不清楚的失败态 | daily 已进；失败样本待收集 | 对齐审核失败态展示、重试/兜底和日志字段 |
| 2026-05-28 | 生图 mode scope 收敛：chat 不做生图；agent 通过 skill runtime 加载生图 tool；第一版不做模型选择 | AV2-W14 | 如果 UI 仍暴露 chat 生图入口，会造成预期错位；agent artifact 展示仍需验证 | scope decision 已记录 | 对齐 agent artifact display、chat exclusion 和 runtime failure display |
| 2026-05-27 | 新增 agent/chat 生成图片 tool + skill first priority；补充产物展示要求 | AV2-W14 | 图片 artifact 展示、预览、保存/复用、继续修改、失败态和权限如果不统一，会影响 chat/agent 体验 | 待验证：tool contract、skill contract、产物展示 UX | 对齐生成图片结果展示和后续编辑/引用路径 |
| 2026-05-27 | query/user profile -> benchmark query pipeline 完成；Gemini 3.5 Flash 当前 SOTA | AV2-W09 / AV2-W12 / AV2-W13 | benchmark UI/review 字段如果不足，会影响模型效果复盘 | pipeline 完成；Gemini SOTA；content thinking UX gap 已记录 | 补 benchmark review 字段和 Gemini UX 风险记录 |
| 2026-05-27 | 面试新实习生并 assign deepresearch skill task | AV2-W07 / AV2-W11 | deepresearch artifact / 引用 / 进度展示需要提前对齐 | 待收实习生调研输出 | review 后确定 deepresearch UI / artifact 需求 |
| 2026-05-21 | 新增 deepresearch skill 和 benchmark 深度 case 后续重点 | AV2-W11 / AV2-W13 | DeepResearch 长任务需要 UI / artifact / 引用 / 失败态支撑；benchmark 需要能 review trace 和产物 | 待验证：deepresearch 展示需求、benchmark review 字段 | 和 recent-master 对齐 skill contract 和 case schema |
| 2026-05-19 | 记录 UID -> APC 渐进式披露工具方向 | AV2-W03 | UI / snapshot 状态需要支持按 UID 取 APC 细节，否则模型只能用低置信度 UID 操作 | 待验证：少量 UID 查询、过期 UID、上下文预算、操作成功率 | 和 recent-master 对齐 tool schema / payload / failure states |
| 2026-05-19 | 新增平台 -> 工具映射 prompt 工作 | AV2-W10 / AV2-W01 / AV2-W02 | 模型可能把用户前端可见状态、E2B 文件、本地文件混为一谈 | 待验证：前端、E2B、本地文件正负例和混合场景 | 补 platform context signal 和回归 case |
| 2026-05-18 | spreadsheet artifact paste payload hardening + doc/sheet contract clarification | AV2-W03 / AV2-W05 | 粘贴 payload 或 contract 变化可能影响用户可见 artifact、表格范围、失败恢复 | 已补 clipboard paste payload 单测；待 UI / 场景回归 | 回归 spreadsheet paste、doc/sheet live-state、exact range、retry budget |
| 2026-05-18 | 新增 21 号前重点：query 挖掘 -> 复杂化 pipeline；继续 track 实习生 skill 工作 | AV2-W07 / AV2-W09 | query 来源和复杂化标准不清会导致 case 不可复用 | 待验证：看板字段、stage、质量门槛、第一批样本入口 | 补第一批 query / skill run 输出，明确 reviewer 和回流目标 |
| 2026-05-14 | 实习生工作基本 align：跑 skill 找工程卡点；表格 skill 初步效果不错，测试回归未跑完 | AV2-W05 / AV2-W07 | 仍需把 case 输出转成可 review 的工程卡点，避免只停留在沟通结论 | 待验证：doc/sheet 触发、进度、结果、失败态；24 个 skill run 输出 | 收集 case 输出并补回归结果 |
| 2026-05-14 | 今日 todo：跟进 tabhobor 上传的 skill 新改动进入 master 后的 UI/场景回归 | AV2-W05 | merge 后可能出现 UI 状态或场景 router 行为差异 | 待验证：doc/sheet 触发、进度、结果、失败态 | merge 后补回归结果 |
| 2026-05-12 | 主要 tab-web 相关改动已推，包含新工具 UX/反馈/skill 场景相关验证 | AV2-W01-W07 | 前端路径或 feedback 可能影响回归判断 | 待回归：常规路径偏离、doc/sheet 场景、复制->粘贴一体化工具 | 补 PR 链接和 case 结果 |
| 2026-05-11 | Tracking page created | all | TBD | TBD | 补真实 PR / branch / case 链接 |
