---
id: agent-v2-dashboard
title: agent_v2 mainline tracking
status: mainline-iterating
created_at: 2026-05-11
updated_at: 2026-06-16
primary_role: 效果调优
repos:
  - recent-master
  - tab-web
---

# Agent V2 Mainline Dashboard

## North Star

Agent v2 主链路进入模型、skill、benchmark 和自动化优化并行迭代阶段。当前最高优先级是 AV2-W08 auto-skill pipeline：把 trajectory / URL 高频站点任务类别沉淀成 skill 生成、验证、发布和 service 化链路。高速模型 harness / Qwen 方向因为公司侧暂时不支持 Qwen，先从 P0 往后放。最新状态：生图链路新增输入审核并推到 daily，后续 todo 是支持更多输入 parameters，并用 file id 替代 `image_url`；脚本 skill 已上线；benchmark pipeline 已 assign 给老板推进，方向是从 URL 高频站点 -> 任务类别挖掘分类，再用 skill 去解决这些任务族。小红书 skill 迭代后发现站点封控非常强，需要单独记录可行边界和降级策略。

## Workstream Board

| ID | Workstream | Repo(s) | State | Last Update | Next Action | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| AV2-W01 | SP prompt | recent-master, tab-web | validating | 2026-05-12 主要改动已推 | 回归平常 case 路径是否偏离，记录分数/轮次/时间 | [file](workstreams/01-sp-prompt.md) |
| AV2-W02 | Tool description 修改 | recent-master, tab-web | validating | 2026-05-18 `search_in_file` 支持 stringified JSON query arrays | 回归 file search query normalization 和 tool path drift | [file](workstreams/02-tool-description.md) |
| AV2-W03 | 新加 tool 修改 | recent-master, tab-web | scoping / validating | 2026-05-19 新增 UID -> APC detail lookup 工具方向；spreadsheet paste payload 继续 validating | 设计按少量 UID 查看 APC 细节的渐进式披露工具，并继续回归 spreadsheet paste | [file](workstreams/03-new-tools.md) |
| AV2-W04 | 老 tool 点击 feedback 增加 | tab-web, recent-master | validating | 2026-05-12 主要改动已推 | 回归 feedback 是否影响老 case 路径和可用信号 | [file](workstreams/04-old-tool-click-feedback.md) |
| AV2-W05 | doc/sheet skill 合入和 orchestration | recent-master, tab-web | validating | 2026-05-18 收紧 doc/sheet skill contracts：live-state、exact ranges、heading semantics、retry budgets | 回归 doc/sheet contract 是否减少误判、重试浪费和 artifact 质量问题 | [file](workstreams/05-doc-sheet-skill-orchestration.md) |
| AV2-W06 | GLM 换模型和 prompt | recent-master | shipping | 2026-05-21 今日 P0：GLM 替代合入 master 并发版 | 完成 master merge、发版、冒烟/回归结论、rollback note | [file](workstreams/06-glm-model-prompt.md) |
| AV2-W07 | 实习生跑 skill tracking | recent-master, tab-web | doing | 2026-05-27 面试新实习生，并给实习生 assign deep research skill task | 跟进 deepresearch skill 输出、阻塞、review 结论和回流动作 | [file](workstreams/07-intern-skill-runs.md) |
| AV2-W08 | Optimization / Auto-skill 自动化迭代链路 | recent-master, tab-web | P0 / active-service-design | 2026-06-16 当前最高优先级：设计并完成整个 auto-skill pipeline，最终做线上 service | 从 trajectory -> skill 实验推进到完整 pipeline，并把 URL 高频站点 -> 任务类别 taxonomy 作为场景来源 | [file](workstreams/08-optimization-automation-loop.md) |
| AV2-W09 | 用户 query 挖掘 -> 复杂化 pipeline | recent-master, tab-web | completed / feeding benchmark | 2026-05-27 完成 query 挖掘、用户画像分析 -> benchmark query pipeline | 继续把高价值 query 回流到 AV2-W13 benchmark 和模型/skill eval | [file](workstreams/09-query-complexification-pipeline.md) |
| AV2-W10 | 平台 -> 工具映射 prompt | recent-master, tab-web | scoping | 2026-05-19 新增：告知模型不同平台对应工具 | 梳理用户前端、E2B、用户本地文件三类平台的工具边界和回归 case | [file](workstreams/10-platform-tool-mapping-prompt.md) |
| AV2-W11 | DeepResearch skill 调研+合并 | recent-master, tab-web | doing | 2026-05-27 已给实习生 assign deep research skill task | 收实习生调研输出，顺手融合 deepresearch skill 并补 benchmark case | [file](workstreams/11-deepresearch-skill.md) |
| AV2-W12 | Gemini 3.5 Flash 调效果+接入 agent v2 | recent-master | validating | 2026-05-27 Gemini 3.5 Flash 调试和跑分完成，目前 SOTA；缺 content thinking 输出影响用户体验 | 评估 content thinking / UX 补救方案，并准备合入 agent v2 链路 | [file](workstreams/12-gemini-35-agent-v2.md) |
| AV2-W13 | Benchmark 深度 case 建设 | recent-master, tab-web | delegated / taxonomy-design | 2026-06-08 benchmark pipeline 已 assign 给老板，方向是 URL 高频站点 -> 任务类别 -> skill 解决 | 跟老板对齐高频站点抽样、任务类别 taxonomy、skill 映射和 benchmark 字段 | [file](workstreams/13-benchmark-deep-cases.md) |
| AV2-W14 | Agent 生图 tool + skill runtime | recent-master, tab-web | daily / input-review-shipped | 2026-06-15 新增输入审核并推到 daily | 支持更多输入 parameters，并用 file id 替代 `image_url` | [file](workstreams/14-image-generation-tool-skill.md) |
| AV2-W15 | 脚本 skill | recent-master | online | 2026-06-08 脚本 skill 已上线 | 补线上 smoke、典型 case、执行/安全边界和 benchmark 回流记录 | [file](workstreams/15-script-skill.md) |
| AV2-W16 | 高速模型 harness / 换模型 | recent-master | postponed / blocked-by-qwen-support | 2026-06-16 公司侧暂时不支持 Qwen，auto harness 往后放 | 保留 harness 设计草稿，等公司侧模型支持或新候选明确后再恢复 | [file](workstreams/16-high-speed-model-harness.md) |
| AV2-W17 | 小红书 skill / 站点封控 | recent-master, tab-web | iterating / blocked-by-site-control | 2026-06-15 迭代小红书 skill，发现网站封控非常强 | 记录封控触发样式、可执行任务边界、失败态和降级策略 | [file](workstreams/17-xiaohongshu-skill.md) |

## Repo Pages

- [recent-master](repos/recent-master.md)
- [tab-web](repos/tab-web.md)

## Run Trackers

- [Intern skill runs](runs/intern-skill-runs.md)
- [Optimization automation runs](runs/optimization-automation-runs.md)
- [Query complexification pipeline](runs/query-complexification-pipeline.md)
- [Benchmark deep cases](runs/benchmark-deep-cases.md)

## Today Todo: 2026-06-16

| Priority | Todo | Area | Done Signal |
| --- | --- | --- | --- |
| P0 | 设计并完成 auto-skill pipeline | AV2-W08 | trajectory / URL 任务类别 -> skill 生成 -> 验证/无验证 -> 发布/service 的 pipeline 设计清楚，并拆出实现步骤 |
| P0 | 明确 auto-skill service 边界 | AV2-W08 | API、状态机、版本、发布/回滚、日志和权限边界有初稿 |
| P0 | 接入 benchmark taxonomy 作为场景来源 | AV2-W08 / AV2-W13 | URL 高频站点、任务类别、skill candidate、coverage gap 和 eval case 能回流 pipeline |
| P1 | 生图输入审核 daily 后验证 | AV2-W14 | 输入审核链路在 daily 可观测，合规输入不误杀，违规/高风险输入有清晰失败态 |
| P1 | 生图输入参数扩展 | AV2-W14 | 更多输入 parameters 的 schema、默认值、校验和前后端展示一致 |
| P1 | 生图引用从 `image_url` 切到 file id | AV2-W14 | tool / skill / artifact 链路使用 file id 引用图片，并保留兼容或迁移策略 |
| P1 | 小红书 skill 封控边界 | AV2-W17 | 封控触发样式、可执行任务、不可执行任务、降级提示和 benchmark case 清楚 |

## Next Focus

| Priority | Focus | Success Signal |
| --- | --- | --- |
| P0 | Auto-skill pipeline -> online service | pipeline 能从 trajectory 和 URL/任务类别 taxonomy 生成、细化、验证/无验证分支到线上 service，按场景稳定维护 skill |
| P1 | Benchmark pipeline: URL 高频站点 -> 任务类别 -> skill | 高频站点和任务类别可挖掘、可分类、可回流到 benchmark，并能映射到 skill 解决策略 |
| P1 | Agent 生图输入审核 + 参数/file-id 收口 | 输入审核已进 daily；更多输入 parameters 和 file id 替代 `image_url` 的 contract 清楚 |
| P1 | 小红书 skill / 站点封控评估 | skill 能识别强封控场景，给出可执行边界、失败态和降级策略 |
| P2 | 脚本 skill 上线后验证 | 脚本 skill 已上线，线上 smoke、典型 case、执行/安全边界和 benchmark 回流记录清楚 |
| P2 | 高速模型 harness / 换模型 | 公司侧暂时不支持 Qwen，先保留 harness 设计；等模型支持或替代候选明确后再恢复 |
| P1 | DeepResearch skill 融合 | skill contract、trigger、artifact、benchmark case 和 UI 展示路径清楚 |
| P1 | Gemini 3.5 Flash 合入 agent v2 | SOTA 跑分可复现，content thinking UX 缺口有补救或明确 tradeoff |

## Key Risks

- GLM 替换时要守住 Sonnet agent v2 已合入主链路后的 baseline，不然主链路回退会很难归因。
- GLM 合入 master 并发版后，如果没有 smoke、rollback 和观察口径，后续 Gemini / skill / benchmark 迭代会难以归因。
- 生成图片 tool 如果没有清晰触发边界、artifact 交付契约和产物展示路径，容易误触发或让 agent 产物体验断掉。
- 生图产物如果只是生成成功但展示不稳定，用户无法确认、保存、复用或继续修改，能力感知会明显打折。
- 生图主链路基本 close 后，剩余可配置项如果前后端不同步，容易出现线上默认值、UI 展示和实际调用参数不一致。
- 生图输入审核进 daily 后，如果误杀率或失败态不可观测，会让用户以为生成链路坏了而不是输入被审核拦截。
- 生图继续扩展输入 parameters 时，如果 schema、默认值、校验和前端展示不一致，后续模型/skill 问题会很难归因。
- 生图从 `image_url` 切到 file id 时，如果兼容策略和 artifact 生命周期没处理好，会出现旧链路引用失效或图片无法复用。
- chat 已明确第一版不做生图，如果 prompt / routing / UI 入口没有排除干净，容易产生预期错位或误触发。
- agent 生图依赖 skill runtime 加载 tool，如果 runtime 契约不稳定，会影响 tool 可见性、参数校验和失败恢复。
- 生图审核链路历史上出现过 `poresche` 偶发失败；即使主链路基本 close，也要保留失败样本和日志入口，避免复现和归因困难。
- 生图链路重构如果没有保住审核链路和角标展示，可能出现“生成成功但 UI 状态不可信”或“审核失败但用户看不懂”的体验断点。
- 生图角标如果状态语义不清，用户会分不清生成中、审核中、生成完成、失败、被审核拦截等状态。
- 生图切到 E2B 链路后，如果平台边界、文件路径和 artifact 回传没有对齐，agent 可能生成成功但用户侧拿不到产物。
- runtime tool 加载小重构同时影响前后端，如果 schema、错误码、展示状态不同步，线上会出现 tool 可见性或触发不一致。
- SQL 模型配置和 settings 配到线上如果缺少灰度、回滚和验证口径，生图默认模型/参数问题会很难归因。
- `image_gen` skill 上传线上后，如果版本、触发条件和 runtime tool schema 不匹配，会出现 skill 能加载但不能稳定调用 tool 的半上线状态。
- trajectory -> skill 如果完全无验证，只能作为效果探索，不能直接作为可靠上线依据。
- auto-skill pipeline 如果没有场景 taxonomy、输入输出格式和后续 service 化边界，容易生成一堆不可维护的 skill。
- benchmark pipeline 已 assign 给老板，需要明确 handoff 字段和同步节奏，不然 URL/task taxonomy 的产出很难回流到 skill 和 eval。
- 从 URL 高频站点挖任务类别如果只看频率不看任务价值，可能漏掉低频但高风险或高价值的 agent case。
- 小红书这类强封控站点如果不单独建边界和降级策略，skill 会在登录、风控、验证码、访问限制里消耗大量轮次。
- 高速模型 harness 当前依赖公司侧模型支持；Qwen 暂不支持时继续投入会卡在平台前置条件上，因此先降优先级。
- 未来恢复高速模型 harness 时，仍需要统一质量、延迟、成本、tool calling 和回滚口径，避免模型问题与 prompt/skill/runtime 问题混在一起。
- 脚本 skill 如果没有执行/安全边界，可能引入本地副作用、权限和不可复现问题。
- 脚本 skill 已上线后，如果不补线上 smoke、典型 case 和失败样式记录，后续回归问题会很难归因。
- DeepResearch skill 如果只看 demo 成功，不看长任务过程、产物质量和失败恢复，合入后风险会被低估。
- Gemini 3.5 Flash 虽然当前跑分 SOTA，但不输出 content thinking 会影响用户体验，需要单独处理。
- Benchmark 如果只增加数量、不增加深度和可归因字段，不能支撑模型和 skill 决策。
- query 复杂化如果没有质量门槛，容易变成“写难题”而不是可复现、可评估的真实需求扩展。
- 平台 -> 工具映射如果写得模糊，模型可能把用户前端、E2B、用户本地文件混用，造成路径偏离或数据/文件上下文错误。
- UID -> APC 细节工具如果没有预算和批量限制，可能重新引入上下文爆炸；如果缺少过期/不存在 UID 处理，会误导后续操作。
- 实习生跑 case 如果没有统一 artifact 和 issue taxonomy，工程卡点会散在聊天里，后面很难回流。

## Update Protocol

1. 每天先更新这张 dashboard 的 `State`、`Last Update`、`Next Action`。
2. 有 PR、prompt diff、eval result、case 结论时，进对应 workstream 加 ledger 行。
3. 有跨 repo 依赖时，同步更新 `repos/recent-master.md` 和 `repos/tab-web.md`。
4. 每周把 dashboard 里的真实进展收束到 `logs/weekly-YYYY-Www.md`。

## Changelog

| Date | Change |
| --- | --- |
| 2026-06-16 | 优先级更新：auto-skill pipeline 成为当前最高优先级；高速模型 harness / Qwen 方向因公司侧暂不支持 Qwen，先往后放 |
| 2026-06-15 | 生图链路新增输入审核并推到 daily；后续 todo 是支持更多输入 parameters，并用 file id 替代 `image_url` |
| 2026-06-15 | 迭代小红书 skill，结论是该站点封控非常强，需要记录封控触发样式、可执行边界、失败态和降级策略 |
| 2026-06-08 | 生图主链路基本 close，剩一个可配置项的前后端收尾；后续聚焦配置一致性、线上默认值、展示和最终 smoke |
| 2026-06-08 | 脚本 skill 已上线，后续补线上 smoke、典型 case、执行/安全边界和 benchmark 回流 |
| 2026-06-08 | benchmark pipeline 任务已 assign 给老板；当前思路是从 URL 高频站点 -> 任务类别挖掘分类，再用 skill 去解决对应任务族 |
| 2026-06-08 | 更新主要 priority：换模型调高速模型 harness，候选 Qwen 3.7 Plus 或其他高速模型；设计并完成 auto-skill pipeline，最终做线上 service |
| 2026-06-08 | 新任务：根据 trajectory -> 生成 skill，先用无验证方式看效果提升；后续演进 auto-skill pipeline，对各场景做细化和 service |
| 2026-06-08 | 上次 2W 行 PR 已合入，进入 merge 后 smoke / 线上配置 / artifact 回归记录阶段 |
| 2026-06-05 | 新增最新 todo：生图放到 E2B 链路；runtime tool 加载小幅重构（含前端）；SQL 模型配置/settings 配线上；`image_gen` skill 上传线上 |
| 2026-06-02 | 今日重构 agent 生图链路代码；重点确认审核链路和生图角标展示 |
| 2026-05-29 | Agent 生图代码已合进 daily；审核链路有问题，`poresche` 偶发失败，需定位样本、日志、影响范围和兜底 |
| 2026-05-28 | 生图 skill/mode 结论：chat 不接入生图；agent 需要 skill runtime 加载生图 tool；第一版先不做模型选择 |
| 2026-05-27 | 完成 query 挖掘、用户画像分析 -> benchmark query pipeline；Gemini 3.5 Flash 调试跑分达到当前 SOTA；面试新实习生并 assign deep research skill |
| 2026-05-27 | 新增 first priority：agent/chat 生成图片 tool + skill、脚本 skill；deepresearch skill 顺手融合 |
| 2026-05-27 | 补充生图产物展示要求：生成图片不只验证 tool 调用成功，还要验证结果预览、保存、复用和继续编辑路径 |
| 2026-05-21 | 今日 P0：GLM 替代合入 master 并发版；新增后续重点：deepresearch skill、Gemini 3.5、benchmark 深度 case |
| 2026-05-19 | 记录 UID -> APC 渐进式披露工具方向：按少量 UID 查看具体 APC 内容，提高网页操作置信度且避免全量 snapshot 爆上下文 |
| 2026-05-19 | 新增平台 -> 工具映射 prompt 工作：明确用户前端、E2B、用户本地文件对应工具和选择边界 |
| 2026-05-18 | Commit `feat(agent-v2): tighten GLM workflows and artifact paste handling`：推进 GLM defaults/prompts、doc/sheet contracts、artifact paste payload、file search normalization 和单测覆盖 |
| 2026-05-18 | Sonnet agent v2 已合入主链路；21 号前重点切到 GLM 替换、query 挖掘复杂化 pipeline、实习生 skill tracking |
| 2026-05-14 | 更新今日进展：实习生工作基本 align；修复 UID 模型幻觉和输出格式 bug，并加工程兜底；表格 skill 初步效果不错，测试回归未跑完 |
| 2026-05-14 | 新增今日 todo：merge tabhobor 上传的 skill 新改动到 master；补 AI 动向/paper；track 实习生工作 |
| 2026-05-12 | 主要改动已推完，进入回归验证；新增复制产物->粘贴一体化工具；skill 更新新 SOP 和更多场景 router；实习生任务已交代 |
| 2026-05-11 | 初始化 agent_v2 tracking 文件系统 |
