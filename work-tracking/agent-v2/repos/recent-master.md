---
id: repo-recent-master
title: recent-master tracking
created_at: 2026-05-11
updated_at: 2026-06-16
---

# recent-master

## Role In Agent V2

主要追踪 agent 行为、prompt、tool/schema、skill orchestration、模型配置和自动化优化链路相关改动。

## Active Touchpoints

| Workstream | Area | Current Link | State | Notes |
| --- | --- | --- | --- | --- |
| AV2-W01 | SP prompt | TBD | validating | 主要改动已推，待回归路径/分数/轮次/时间 |
| AV2-W02 | Tool descriptions | TBD | validating | `search_in_file` 支持 stringified JSON query arrays，单测已扩展 |
| AV2-W03 | New tools | TBD | scoping / validating | 新增 UID -> APC detail lookup 工具方向；spreadsheet artifact paste payload 已 harden |
| AV2-W05 | doc/sheet skill orchestration | TBD | validating | doc/sheet skill contracts 已收紧，待场景回归 |
| AV2-W06 | GLM model + prompt | TBD | shipping | 今日 P0：GLM 替代合入 master 并发版 |
| AV2-W08 | Optimization / auto-skill automation | TBD | P0 / active-service-design | 当前最高优先级：设计并完成 auto-skill pipeline，最终做线上 service；trajectory -> skill 和 URL 高频站点 -> 任务类别 taxonomy 作为输入 |
| AV2-W09 | Query mining -> complexification pipeline | TBD | completed / feeding benchmark | query 挖掘 + 用户画像分析 -> benchmark query pipeline 已完成 |
| AV2-W10 | Platform -> tool mapping prompt | TBD | scoping | 新增用户前端 / E2B / 用户本地文件对应工具的 prompt 调优 |
| AV2-W11 | DeepResearch skill | TBD | doing | 已 assign 给实习生调研，后续顺手融合 |
| AV2-W12 | Gemini 3.5 Flash model integration | TBD | validating / SOTA | Gemini 3.5 Flash 跑分当前 SOTA；缺 content thinking UX |
| AV2-W13 | Benchmark deep cases | TBD | delegated / taxonomy-design | benchmark pipeline 已 assign 给老板；方向是 URL 高频站点 -> 任务类别 -> skill 解决 |
| AV2-W14 | Agent image generation skill/runtime | TBD | daily / input-review-shipped | 生图链路新增输入审核并推到 daily；后续支持更多 input parameters，并用 file id 替代 `image_url` |
| AV2-W15 | Script skill | TBD | online | 脚本 skill 已上线，待补 smoke、典型 case 和安全/执行边界 |
| AV2-W16 | High-speed model harness | TBD | postponed / blocked-by-qwen-support | 公司侧暂时不支持 Qwen，auto harness 往后放，保留设计草稿 |
| AV2-W17 | Xiaohongshu skill / site control | TBD | iterating / blocked-by-site-control | 小红书 skill 迭代后发现站点封控非常强，需要单独记录边界和降级策略 |

## Cross-Repo Contracts

| Contract | Producer | Consumer | Status | Evidence |
| --- | --- | --- | --- | --- |
| Tool click feedback event/schema | tab-web | recent-master | TBD | TBD |
| Tool description/schema exposed to UI or eval | recent-master | tab-web | TBD | TBD |
| Skill orchestration state shown in UI | recent-master | tab-web | TBD | TBD |
| Platform -> tool mapping | recent-master | tab-web / eval / GLM prompt | scoping | 2026-05-19 新增，需要明确用户前端、E2B、本地文件工具边界 |
| Benchmark case schema | recent-master | model eval / skill eval / tab-web review | scoping | 2026-05-21 新增，需要统一 score / rounds / time / artifact quality / issue type |

## Open Checks

- [ ] 当前 agent_v2 分支 / PR 是哪一个？
- [ ] baseline eval 数据在哪里？
- [ ] prompt 和 model 改动是否能拆开验证？
- [ ] tab-web 依赖的 schema 是否已经稳定？
- [ ] 自动化优化链路的输出是否可回灌到 prompt/tool 迭代？

## Repo Log

| Date | Change / PR | Workstream | Risk | Validation | Next |
| --- | --- | --- | --- | --- | --- |
| 2026-06-16 | 优先级更新：auto-skill pipeline 成为当前最高优先级 | AV2-W08 | 如果 P0 被生图、小红书、模型 harness 等支线打散，pipeline/service 化会延后 | dashboard priority updated | 优先拆 pipeline stage、service API、状态机、版本/回滚、benchmark taxonomy 接入 |
| 2026-06-16 | 高速模型 harness / Qwen 因公司侧暂不支持 Qwen 后置 | AV2-W16 | 继续作为 P0 会卡在公司模型支持前置条件上 | postponed | 保留设计草稿，等 Qwen 支持或替代高速模型候选明确后恢复 |
| 2026-06-15 | 生图链路新增输入审核并推到 daily | AV2-W14 | 输入审核如果误杀或用户态不清，会被误认为生图链路失败 | daily pushed; 样本待记录 | 记录通过/拦截/误杀样本；继续支持更多 input parameters 和 file id |
| 2026-06-15 | 生图后续 todo：支持更多 input parameters，并用 file id 替代 `image_url` | AV2-W14 | 参数 schema、默认值、校验和 skill prompt 不一致会导致调用不可归因；URL 引用存在生命周期/权限问题 | 待设计 schema 和迁移策略 | 定义 parameters schema；设计 file id 入参、artifact 引用和旧链路兼容 |
| 2026-06-15 | 迭代小红书 skill，发现站点封控非常强 | AV2-W17 | 强封控会导致 skill 在登录、验证码、频控和内容不可见里消耗大量轮次 | manual iteration; high site-control risk | 收集封控样式，定义任务边界、降级策略和 benchmark case |
| 2026-06-08 | 生图主链路基本 close，剩一个可配置项的前后端收尾 | AV2-W14 | 配置项如果前后端 schema/default/settings 不一致，线上展示和真实调用参数会偏离 | basic close; config smoke 待补 | 收尾可配置项前后端，补最终 smoke、线上默认值和失败态 |
| 2026-06-08 | 脚本 skill 已上线 | AV2-W15 | 上线后如果缺少 smoke、典型 case 和边界记录，后续问题难归因 | online; smoke 待补 | 记录线上 smoke、典型 case、执行/安全边界和 benchmark 回流 |
| 2026-06-08 | benchmark pipeline 已 assign 给老板，方向是 URL 高频站点 -> 任务类别 -> skill 解决 | AV2-W13 / AV2-W08 | 如果 handoff 字段不清，taxonomy 很难回流到 auto-skill 和 eval | handoff done; taxonomy 待产出 | 对齐高频站点样本、任务类别、skill candidate、coverage gap 和 eval case |
| 2026-06-08 | 主要 priority：换模型调高速模型 harness，候选 Qwen 3.7 Plus 或其他高速模型 | AV2-W16 | 如果没有统一 harness，质量/延迟/成本/tool calling 和 rollout 风险会难比较 | 待建 harness 和候选模型矩阵 | 定义 eval cases、latency/cost/quality 指标、切换策略和 rollback |
| 2026-06-08 | 主要 priority：设计并完成整个 auto-skill pipeline，最终做成线上 service | AV2-W08 | service 化如果没有状态、版本、回滚和审计，后续难运营 | 待拆 pipeline 和 service API | 设计 trajectory -> skill -> refine -> verify/no-verify -> publish/service 的完整链路 |
| 2026-06-08 | 新任务：trajectory -> 生成 skill，先无验证看效果提升；后续 auto-skill pipeline 场景细化/service 化 | AV2-W08 | 无验证实验不能证明稳定性；pipeline 如果没有场景 taxonomy 会难维护 | 待记录生成 skill 样本和效果信号 | 跑第一版 trajectory -> skill，沉淀 auto-skill pipeline 草案 |
| 2026-06-08 | 上次 2W 行生图 PR 已合入 | AV2-W14 | 大 PR 合入后如果没有 smoke 和线上配置记录，问题会难定位 | merged; smoke 待记录 | 补 E2B artifact、runtime loading、SQL/settings、`image_gen` skill 上传状态 |
| 2026-06-05 | 生图上线收口 todo：E2B 链路、runtime tool 加载小重构、SQL/settings 线上配置、`image_gen` skill 上传线上 | AV2-W14 | 执行环境、runtime loading、线上配置和 skill registry 任一环节不一致，都会导致链路半通或线上不可复现 | 待验证 E2B route、runtime loading、settings/SQL、skill registry smoke | 先对齐 E2B 路由和 runtime loading，再做线上配置和 skill 上传 smoke |
| 2026-06-02 | 重构 agent 生图链路代码，重点确认审核链路和生图角标 | AV2-W14 | 重构可能影响 runtime/tool 调用、审核顺序、artifact 写入和失败态；角标状态若不准会误导用户 | 待验证 daily smoke、审核/`poresche` 日志、状态映射 | 明确链路边界、审核调用点、日志字段、角标状态和回归 case |
| 2026-05-29 | Agent 生图代码已合进 daily；审核/`poresche` 偶发失败 | AV2-W14 | 审核链路不稳定会阻断真实请求；没有日志/样本会难以复现和归因 | daily 已进；审核问题待定位 | 收集失败样本、请求 id、审核返回、重试状态和最终用户态，明确修复/兜底 |
| 2026-05-28 | 生图 skill/mode scope：chat 不做生图；agent 需要 skill runtime 加载生图 tool；第一版不做模型选择 | AV2-W14 | skill runtime 契约不清会导致 tool 加载、schema、错误码和日志不可控；chat 入口如果没排除会误触发 | scope decision 已记录 | 定义 agent skill runtime 加载契约、image tool schema、默认模型/参数和回归 case |
| 2026-05-27 | 完成 query 挖掘、用户画像分析 -> benchmark query pipeline | AV2-W09 / AV2-W13 | pipeline 如果不持续补深度字段，会只产出浅层 query | 已完成 pipeline，开始支撑 Gemini 3.5 Flash 跑分和 benchmark case | 持续扩充更多更深度 case |
| 2026-05-27 | Gemini 3.5 Flash 调试和跑分完成，目前 SOTA | AV2-W12 | 不输出 content thinking 会影响用户体验，合入前需要补救方案或明确 tradeoff | 跑分 SOTA；UX gap 已记录 | 设计 content thinking/progress UX 方案并准备接入 agent v2 |
| 2026-05-27 | 面试新实习生，并 assign deepresearch skill task | AV2-W07 / AV2-W11 | 实习生输出如果缺少统一格式，会难以融合 skill | 待收 deepresearch skill 调研输出 | review 输出并回流 AV2-W11 / AV2-W13 |
| 2026-05-27 | 新增 first priority：agent/chat 生成图片 tool+skill、脚本 skill；补充生图产物展示要求 | AV2-W14 / AV2-W15 | 新 tool/skill 需要清晰触发、artifact、产物展示、执行/安全边界 | 待设计 contract、产物展示 UX 和回归 case | 先做 image generation tool+skill 和 script skill |
| 2026-05-21 | 今日 P0：GLM 替代合入 master 并发版 | AV2-W06 | 发版后如果没有 smoke、rollback 和观察口径，会影响后续 Gemini / skill / benchmark 归因 | 待记录 master merge、release、smoke、rollback note | 合入 master，发版并记录 release evidence |
| 2026-05-21 | 新增后续重点：deepresearch skill、Gemini 3.5、benchmark 深度 case | AV2-W11 / AV2-W12 / AV2-W13 | 三条线如果没有统一 benchmark 和合入策略，容易变成散点调优 | 待验证：skill contract、Gemini eval matrix、benchmark schema | 补调研结论、接入计划和第一批深度 case |
| 2026-05-19 | 迭代方向：按少量 UID 查看具体 APC 内容的渐进式披露工具 | AV2-W03 | 全量 APC 会爆上下文；只看 UID/简略内容会降低操作置信度；UID 过期/不存在需要清晰失败态 | 待验证：snapshot -> UID 选择 -> APC detail lookup -> 操作 的链路 | 设计 tool contract、预算上限、批量上限、失败态和回归 case |
| 2026-05-19 | 新增平台 -> 工具映射 prompt 工作：用户前端、E2B、用户本地文件 | AV2-W01 / AV2-W02 / AV2-W06 / AV2-W10 | 平台工具边界不清会导致模型在前端、E2B、本地文件之间串台 | 待验证：platform x tool matrix、正负例、混合场景、GLM before/after | 补 prompt diff、工具矩阵和回归 case |
| 2026-05-18 | feat(agent-v2): tighten GLM workflows and artifact paste handling | AV2-W02 / AV2-W03 / AV2-W05 / AV2-W06 | 多条链路同时变化，需要拆开回归 GLM、paste、doc/sheet、file search | 已补 clipboard paste payload 和 file search query normalization 单测；待集成回归 | 跑 GLM before/after、doc/sheet contracts、spreadsheet paste、search_in_file 输入归一化 |
| 2026-05-18 | Sonnet agent v2 已合入主链路；GLM 替换和 query 复杂化 pipeline 成为 21 号前重点 | AV2-W06 / AV2-W09 | GLM 替换可能引入主链路回退；query 复杂化可能脱离真实需求 | 待验证：GLM before/after matrix、query pipeline 第一批样本 | 建 baseline / candidate / eval matrix，补 query board 第一批样本 |
| 2026-05-14 | 修复 UID 模型幻觉和输出格式 bug，加工程兜底；表格 skill 初步效果不错，测试回归未跑完 | AV2-W05 / AV2-W06 | 回归未完成前还不能判断是否有路径偏离或格式残留问题 | 待验证：UID 稳定性、输出格式、doc/sheet 场景、score/rounds/time | 等测试回归结果，补 PR / commit 和 case 结论 |
| 2026-05-14 | 今日 todo：merge tabhobor 上传的 skill 新改动到 master branch | AV2-W05 | merge 后仍需确认 master 上行为和回归结果 | 待验证：doc/sheet 场景、常规 case path、score/rounds/time | merge 到 master 并记录 PR / commit |
| 2026-05-12 | 主要 agent_v2 改动已推；skill 更新新 SOP 和更多场景 router；新增复制产物->粘贴一体化工具 | AV2-W01-W08 | 多项改动同时进入回归，需避免归因混乱 | 待回归：常规 case 路径、doc/sheet 效果、分数、轮次、时间 | 补 PR / branch 链接，记录回归结果 |
| 2026-05-11 | Tracking page created | all | TBD | TBD | 补真实 PR / branch / case 链接 |
