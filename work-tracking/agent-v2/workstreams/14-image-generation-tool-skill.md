---
id: AV2-W14
title: Agent 生图 tool + skill runtime
status: daily / input-review-shipped
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-27
updated_at: 2026-06-16
---

# AV2-W14 Agent 生图 Tool + Skill Runtime

## Scope

给 agent 模式新增生成图片的 tool 和对应 skill runtime。2026-05-28 scope decision：chat 第一版不接入生图；agent 通过 skill runtime 加载生图 tool；第一版先不做选模型能力。2026-06-02 进入生图链路代码重构，重点确认审核链路和生图角标。2026-06-05 最新 todo：生图放到 E2B 链路，runtime tool 加载小幅重构（前端也需要），SQL 模型配置/settings 配到线上，`image_gen` skill 上传到线上。2026-06-08 状态：生图主链路基本 close，剩一个可配置项的前后端收尾。2026-06-15 最新状态：生图链路新增输入审核并推到 daily；后续 todo 是支持更多输入 parameters，并用 file id 替代 `image_url`。

## Why It Matters

生成图片是 agent 的高感知能力。关键不只是能调用模型生成图，而是 agent 知道什么时候该触发、如何通过 skill runtime 拿到可用 tool、如何收集需求、如何交付并展示图片 artifact、如何处理失败和修改请求。产物展示是核心体验：用户需要稳定看到结果、确认质量、保存复用，并能基于同一张图继续编辑。chat 第一版不做生图，可以降低误触发和入口复杂度。

## Integration Board

| Area | Current State | Needed For Ship | Notes |
| --- | --- | --- | --- |
| Chat scope | decided | chat 第一版不暴露生图能力，不触发 image generation skill/tool | 2026-05-28 decision |
| Agent skill runtime | mostly closed | agent 能通过 skill runtime 加载生图 tool，并获得稳定 tool schema / error surface | 只剩配置项相关收尾 |
| Tool contract | mostly closed | 输入 prompt / style / size / reference / output artifact；失败态清楚 | 第一版不做模型选择 |
| Skill contract | mostly closed | 何时使用、如何澄清需求、如何迭代修改 | 仅 agent 模式 |
| Routing / mode gating | mostly closed | agent 可触发；chat 明确不触发 | 需要最终 smoke 记录 |
| Artifact delivery | mostly closed | 图片结果可查看、可保存、可继续编辑/引用 | 需要最终 smoke 记录 |
| Result presentation | mostly closed | 生成结果在对话中有清晰预览、状态、元信息和后续操作入口 | 重点验证用户是否能确认、保存、复用、继续修改 |
| Review / moderation | mostly closed / monitor | 审核链路稳定通过；`poresche` 偶发失败需要保留可观测样本、日志和兜底 | 2026-06-02 重点确认 |
| Input review / moderation | daily | 生图输入在进入生成链路前完成审核，合规输入不误杀，违规/高风险输入有清晰失败态 | 2026-06-15 已推 daily |
| Image badge | mostly closed / final smoke | 生图角标的展示时机、文案/样式、状态映射和失败/审核场景明确 | 2026-06-02 重点确认 |
| Refactor boundary | mostly closed | 重构后保持 daily 主链路、runtime/tool schema、审核、产物展示和失败态可回归 | 2026-06-08 基本 close |
| E2B execution route | mostly closed | 生图请求放到 E2B 链路，明确平台、文件、artifact 回传和失败态边界 | 需要最终 smoke 记录 |
| Runtime tool loading | mostly closed | 后端 runtime tool discovery/schema/error 小幅重构，前端同步展示/触发契约 | 需要最终 smoke 记录 |
| Online model/settings config | mostly closed / config follow-up | SQL 模型配置和 settings 配到线上，并保留灰度/回滚/验证口径 | 剩可配置项前后端收尾 |
| Configurable item FE/BE | open | 可配置项在前端入口、后端 schema/default、线上 settings 和失败态上对齐 | 2026-06-08 latest remaining item |
| Input parameters | todo | 支持更多生成输入 parameters，并明确 schema、默认值、校验、skill prompt 和前端展示 | 2026-06-15 follow-up |
| Image reference | todo | 用 file id 替代 `image_url`，避免 URL 生命周期、权限和复用问题 | 2026-06-15 follow-up |
| Online skill upload | mostly closed | `image_gen` skill 上传线上，版本、触发条件、tool schema 和 artifact 展示对齐 | 需要最终 smoke 记录 |
| Model selection | deferred | 第一版不提供选模型能力 | 后续按效果/成本再评估 |
| Safety / permissions | scoping | 有内容边界、引用图/本地文件处理策略 | TBD |
| Eval / benchmark | scoping | 正例、负例、修改请求、失败恢复 case | 回流 AV2-W13 |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Mode gating | traces / manual review | chat 不触发生图；agent 在需要生图时能触发 |
| Runtime tool loading | traces / logs | agent skill runtime 能稳定加载生图 tool，并暴露正确 schema / errors |
| Prompt quality | artifact review | 生成需求被完整转成图像 prompt |
| Artifact delivery | UI / logs | 用户能看到、复用、继续修改图片 |
| Result presentation | UI review | 图片预览、生成状态、保存/复用/编辑入口清楚且不断链 |
| Review stability | daily logs / failure samples | 审核链路稳定；`poresche` 偶发失败能定位、恢复或降级 |
| Input review correctness | daily traces / manual cases | 合规输入通过，违规/高风险输入被拦截，用户态和日志可解释 |
| Badge correctness | UI review / traces | 生图角标在生成中、审核中、完成、失败、审核拦截等状态下展示正确 |
| Refactor regression | daily smoke / unit / traces | 重构后原 daily 生图主链路不回退 |
| E2B route correctness | traces / artifact review | 生图请求走 E2B，产物能回传到 agent artifact surface |
| Runtime loading compatibility | backend + frontend smoke | runtime tool 能加载、前端能识别/展示，schema 和错误态一致 |
| Online config health | settings / SQL / smoke | 线上配置生效且可灰度、可回滚、可验证 |
| Config parity | frontend + backend smoke | 可配置项前端展示、后端 schema/default、线上 settings 和真实调用参数一致 |
| Parameter extensibility | schema review / smoke | 新增 input parameters 能被 skill/tool/UI 一致消费 |
| File id reference health | artifact review / traces | file id 能替代 `image_url`，图片引用可验证、可复用、可权限控制 |
| Skill upload health | online skill registry / smoke | `image_gen` skill 在线上可加载、可触发、可调用 tool |
| Iteration quality | follow-up turns | 用户要求修改时能保留上下文并迭代 |
| Failure recovery | failure cases | 生成失败、参数缺失、权限问题能解释和恢复 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-15 | recent-master, tab-web | 生图链路新增输入审核，并已推到 daily | 在生成前审核输入可以提前拦截高风险请求，减少后续生成/展示链路的不可控失败 | daily pushed; 待记录通过/拦截样本 | shipped to daily | monitor daily | 观察误杀率和失败态；后续支持更多输入 parameters，并用 file id 替代 `image_url` |
| 2026-06-08 | recent-master, tab-web | 生图主链路基本 close，剩一个可配置项的前后端收尾 | 主链路风险已从大功能接入转为配置一致性和最终 smoke | basic close; config follow-up | mostly closed | shift to follow-up | 收尾可配置项前后端，补线上默认值、展示、失败态和最终 smoke 记录 |
| 2026-06-08 | recent-master, tab-web | 上次 2W 行 PR 已合入 | 大体量生图链路 PR 合入后，风险从代码接入转向 merge 后 smoke、线上配置和 artifact 回归 | merged; smoke 待记录 | merged | rollout follow-up | 补 daily/线上 smoke、E2B artifact、runtime loading、settings/SQL、`image_gen` skill 上传状态 |
| 2026-06-05 | recent-master, tab-web | 新增最新 todo：生图放 E2B 链路；runtime tool 加载小重构（前端也需要）；SQL 模型配置/settings 配线上；`image_gen` skill 上传线上 | 生图进入线上前，需要把执行环境、runtime tool 加载、线上配置和 skill 发布四个环节收口，否则会出现链路半通或线上不可复现 | 待验证 E2B route、runtime loading、online config、skill registry smoke | todo | rollout prep | 先对齐 E2B 路由和 runtime loading，再配线上 SQL/settings 和上传 `image_gen` skill |
| 2026-06-02 | recent-master, tab-web | 今日重构生图链路代码；重点确认审核链路和生图角标 | 把审核、runtime/tool 调用和 UI 角标状态收敛到清晰链路，可以减少 daily 后的偶发失败和状态误导 | 待验证 daily smoke、审核/`poresche` 日志、角标 UI 状态 | doing | refactor today | 明确审核链路、角标状态映射、失败态和回归 case |
| 2026-05-29 | recent-master, tab-web | 生图代码已合进 daily；审核有问题，`poresche` 偶发失败 | 代码进 daily 后，真实请求链路的主要风险从 tool 接入转向审核稳定性；需要先把偶发失败变成可复现/可观测问题 | daily 环境；审核/`poresche` 失败样本待收集 | review-risk | 暂不视为完全验收 | 收集失败样本、日志、影响范围，明确兜底/修复方案，再回归产物展示 |
| 2026-05-28 | recent-master, tab-web | 生图 skill/mode scope 收敛：chat 不做生图；agent 需要 skill runtime 加载生图 tool；第一版不管选模型 | 先把能力收敛到 agent + runtime loaded tool，可以降低入口复杂度和误触发风险，更快验证产物展示链路 | 待验证 mode gating、runtime loading、artifact display | scoped | chat excluded; agent runtime required; model selection deferred | 设计 skill runtime 加载契约、tool schema、agent 触发正负例和产物展示 |
| 2026-05-27 | recent-master, tab-web | 新增 first priority：agent/chat 生成图片 tool + skill；补充产物展示要求 | 把图片生成做成 tool + skill，并明确结果展示路径，可以让 agent/chat 在合适时机稳定交付可确认、可保存、可复用、可继续编辑的图片 artifact | 待设计 contract、展示 UX 和回归 case | pending | first priority | 定义 tool/skill 契约、产物展示 UX、正负例和安全边界 |

## Open Risks

- 触发边界不清会导致普通对话误触发图片生成。
- 图片 artifact 交付如果没有统一格式，后续编辑/下载/引用会断。
- 产物展示如果只停留在链接或日志里，用户无法直接确认生成质量，也无法自然进入保存、复用或继续修改。
- chat 第一版不做生图，如果入口、prompt 或 routing 没排除干净，会造成用户预期和模型行为不一致。
- agent 依赖 skill runtime 加载 tool，如果 runtime 没有清楚的加载失败态、schema 校验和日志，问题会难以定位。
- 第一版不选模型可以降低 scope，但需要记录默认模型/默认参数，避免后续效果回归不可归因。
- 审核链路历史上出现过 `poresche` 偶发失败；即使主链路基本 close，也要保留请求 id、输入、审核返回、重试状态和最终用户态，避免复现和归因困难。
- 输入审核进 daily 后，如果误杀率不可观测，用户会把审核拦截误认为生图失败；需要记录通过/拦截样本和用户态文案。
- 重构生图链路时，如果审核链路、tool 调用、artifact 写入和 UI 角标的状态顺序没有对齐，容易产生重复状态、漏状态或用户看到错误角标。
- 生图角标如果只按“是否生成成功”展示，会漏掉审核中、审核失败、重试中、生成失败等用户真正需要理解的状态。
- 生图放到 E2B 链路后，如果 artifact 路径、权限和回传 surface 不一致，会出现 E2B 内生成成功但 agent/chat 看不到结果。
- runtime tool 加载小重构如果前端没有同步，会造成 tool 后端可用但 UI/agent 状态不可见，或者错误态展示不一致。
- 剩余可配置项如果前后端 schema/default/展示不一致，会导致用户看到的配置和真实调用参数不一致。
- 更多 input parameters 如果只在 tool 层支持、skill prompt 或前端没有同步，会出现模型不知道怎么填或用户看不到真实参数的问题。
- 用 file id 替代 `image_url` 时需要处理旧链路兼容、artifact 生命周期和权限，否则可能出现图片引用断链。
- SQL 模型配置/settings 上线如果没有记录版本和生效范围，默认模型/参数导致的效果问题会难以回滚。
- `image_gen` skill 上传线上如果和 runtime tool schema 不匹配，会出现 skill 触发成功但调用失败。
- 生成失败或需要澄清时，如果没有 skill 流程，体验会变成一次性调用。

## Next Actions

- [x] 明确第一版 mode scope：chat 不做生图；agent 做生图。
- [x] 明确第一版不做模型选择。
- [x] 记录代码已合进 daily。
- [x] 记录 2W 行 PR 已合入。
- [x] 生图主链路基本 close。
- [x] 输入审核已加入生图链路并推到 daily。
- [ ] 记录输入审核 daily 样本：通过、拦截、误杀、失败态和用户态文案。
- [ ] 支持更多输入 parameters：schema、默认值、校验、skill prompt 和前端展示一致。
- [ ] 用 file id 替代 `image_url`：tool 入参、artifact 引用、权限/生命周期和旧链路兼容明确。
- [ ] 收尾可配置项前后端：schema、默认值、settings、展示状态和失败态一致。
- [ ] 补最终 smoke：agent mode gating、E2B artifact、runtime loading、审核链路、角标、产物展示。
- [ ] 记录线上配置和 `image_gen` skill 版本/触发条件/tool schema 的最终状态。
- [ ] 保留审核/`poresche` 失败样本记录入口：请求 id、输入、审核返回、重试状态、最终用户态。
- [ ] 补正例、负例、修改请求、失败恢复 benchmark case。
- [ ] 明确安全/权限/引用图和本地文件处理策略。
