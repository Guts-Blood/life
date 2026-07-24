---
period: 2026-W21
date_range: 2026-05-18 to 2026-05-24
created_at: 2026-05-18
---

# Weekly Review: Agent V2 / 2026-W21

## Focus / Transition

| Priority | Focus | Owner | Success Signal |
| --- | --- | --- | --- |
| P0 | GLM 替代合入 master 并发版 | 我 | master merge、release、smoke / 回归结论、rollback note 已记录 |
| P0 | DeepResearch skill 调研+合并 | 我 | skill contract、merge path、risk、benchmark case 已明确 |
| P0 | Gemini 3.5 调效果+合入 agent v2 | 我 | 和 GLM / Sonnet 同口径 eval 后有接入策略和 rollback |
| P0 | Benchmark 深度 case 建设 | 我 | 更多更深度 case 有 schema、分类、评分和第一批样本 |
| P0 | 搭 query 挖掘 -> 复杂化 pipeline 看板 | 我 | 能从真实 query 进入 intake，经过复杂化和 review，回流 eval / skill / product |
| P0 | 调平台 -> 工具映射 prompt | 我 | 模型能区分用户前端、E2B、用户本地文件，并选择对应工具面 |
| P0 | Track 实习生 skill 工作 | 我 + 实习生 | 每批 case 有 artifact、issue type、owner / reviewer 和工程 follow-up |

## Workstream Snapshot

| Workstream | State | Next |
| --- | --- | --- |
| AV2-W02 Tool description | validating | 回归 `search_in_file` string / array / stringified JSON array 输入 |
| AV2-W03 New tools / artifact paste / UID APC detail | scoping / validating | 设计 UID -> APC detail lookup；回归 spreadsheet paste payload、merged cell、CSV/TSV cap warning |
| AV2-W05 doc/sheet skill orchestration | validating | 回归 live-state、exact ranges、heading semantics、retry budgets |
| AV2-W06 GLM model + prompt | shipping | 今日完成 master merge、发版、smoke 和 rollback note |
| AV2-W07 Intern skill runs | doing | 收今日输出，按工程卡点和可回流 case 归类 |
| AV2-W09 Query complexification pipeline | scoping | 补第一批 query 样本和复杂化规则 |
| AV2-W10 Platform -> tool mapping prompt | scoping | 建 platform x tool matrix，补前端 / E2B / 本地文件正负例 |
| AV2-W11 DeepResearch skill | scoping | 调研 skill contract、合并路径和回归 case |
| AV2-W12 Gemini 3.5 model integration | scoping | 建 Gemini 3.5 vs GLM / Sonnet eval matrix |
| AV2-W13 Benchmark deep cases | scoping | 定义 benchmark schema，补第一批更深度 case |

## Decisions

| Decision | Why | Follow-up |
| --- | --- | --- |
| 2026-05-21 今日 P0 是 GLM 替代合入 master 并发版 | GLM release 将成为后续 Gemini、deepresearch 和 benchmark 的近期 baseline | 记录 master merge、release、smoke、rollback note |
| 2026-05-21 后续重点切到 deepresearch skill、Gemini 3.5、benchmark 深度 case | GLM 发版后需要继续扩展 skill、模型候选和评估深度 | 建 AV2-W11 / AV2-W12 / AV2-W13 并回流 case |
| 2026-05-19 记录 UID -> APC 渐进式披露工具方向 | `take_snapshot` 只给 UID/简略内容会降低操作置信度，全量 APC 又会爆上下文；按需展开少量 UID 更适合渐进式网页理解 | 设计 tool contract、预算上限、失败态和 snapshot -> detail -> action 回归链路 |
| 2026-05-18 commit 先推进 GLM defaults/prompts，同时 harden artifact paste 和 doc/sheet contracts | GLM 替换需要执行层 prompt/config 先稳定；doc/sheet 和 paste 是高价值回归面 | 拆成 GLM、paste、doc/sheet、file search 四组回归 |
| 2026-05-19 新增平台 -> 工具映射 prompt | 需要让模型理解不同平台对应不同工具面，避免前端、E2B、本地文件串台 | 建工具矩阵和回归 case，回流 AV2-W01 / AV2-W02 / AV2-W06 |
| Sonnet agent v2 合入主链路后，GLM 替换成为 21 号前主线 | 主链路 baseline 已稳定，下一步要验证 GLM 能否替代或灰度 | 建 GLM eval matrix 和 guardrail |
| 新增 query 挖掘 -> 复杂化 pipeline 看板 | 需要系统化把真实用户需求变成可复用 case | 建 stage、字段、质量门槛和第一批样本入口 |
| 继续 track 实习生 skill 工作 | 实习生 run 能提供工程卡点和真实 case 输入 | 输出回流 AV2-W06 / AV2-W09 |

## Evidence

| Evidence | Linked Workstream | Takeaway |
| --- | --- | --- |
| 2026-05-21 工作重点更新 | AV2-W06 / AV2-W11 / AV2-W12 / AV2-W13 | 今日发版 GLM，后续推进 deepresearch skill、Gemini 3.5、benchmark 深度 case |
| UID -> APC detail lookup idea | AV2-W03 | 用按 UID 查询 APC 细节替代全量 snapshot 扩容，作为网页内容渐进式披露机制 |
| 2026-05-19 新增平台工具映射 prompt 工作 | AV2-W01 / AV2-W02 / AV2-W06 / AV2-W10 | 新增用户前端、E2B、用户本地文件三类平台的工具选择边界 |
| feat(agent-v2): tighten GLM workflows and artifact paste handling | AV2-W02 / AV2-W03 / AV2-W05 / AV2-W06 | GLM executor/fallback defaults、browser-use prompt、doc/sheet contracts、artifact paste payload、file search normalization 已进入一版工程改动 |

## Open Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| GLM 替换导致主链路回退 | 影响 Sonnet agent v2 已合入后的稳定性 | 以 Sonnet mainline 做 baseline，设置红线 case 和回滚策略 |
| GLM 发版证据不足 | 后续 Gemini / skill / benchmark 问题难以归因 | 记录 master merge、release、smoke、rollback 和观察项 |
| DeepResearch skill 合入过早 | 长任务质量、引用、失败恢复和 UI 展示可能不稳定 | 先调研 contract 和 benchmark，再合并 |
| Gemini 3.5 调效果无同口径 baseline | 容易用主观样例判断模型好坏 | 用 AV2-W13 benchmark 对比 GLM / Sonnet |
| Benchmark case 只堆数量 | 无法支撑模型和 skill 决策 | 增加深度、评分字段、issue type 和 artifact quality |
| 平台工具映射不清 | 模型可能把用户前端、E2B、本地文件混用，导致路径偏离和上下文错误 | 建 platform x tool matrix，加入正负例和混合场景回归 |
| UID -> APC detail lookup 无预算控制 | 可能重新引入上下文爆炸，或者让模型依赖过期 UID 操作 | 限制 UID 数量/返回大小，明确 stale/missing UID failure state |
| artifact paste / doc-skill contract 回归不充分 | 用户可见产物可能出现表头注入、范围误判或失败恢复不清 | 单测已补，继续跑 spreadsheet paste 和 doc/sheet 场景回归 |
| query 复杂化脱离真实需求 | eval 结果不能指导产品或工程 | 每条 case 保留 source、context 和 acceptance criteria |
| 实习生输出不可 review | 无法沉淀工程卡点 | 固定 artifact、issue type、owner / reviewer 字段 |

## Checklist

- [ ] GLM master merge、发版、smoke、rollback note 完成。
- [ ] DeepResearch skill 调研+合并计划完成。
- [ ] Gemini 3.5 eval matrix 和接入策略完成。
- [ ] Benchmark deep cases schema 和第一批 case 完成。
- [ ] GLM eval matrix 建好并开始填结果。
- [ ] Platform -> tool mapping prompt 矩阵和回归 case 建好。
- [ ] UID -> APC detail lookup tool contract 和回归链路建好。
- [x] Query complexification pipeline 第一版可用。
- [ ] 实习生 skill 工作今日进展写入 run ledger。
- [ ] 21 号前形成 keep / iterate / rollback / scope decision。
