---
period: 2026-W24
date_range: 2026-06-08 to 2026-06-14
created_at: 2026-06-08
---

# Weekly Review: Agent V2 / 2026-W24

## Focus

| Workstream | Focus | Evidence / Notes |
| --- | --- | --- |
| AV2-W16 高速模型 harness | 换模型调 harness | 候选 Qwen 3.7 Plus 或其他高速模型 |
| AV2-W08 Auto-skill | trajectory -> skill 无验证实验 | 先看是否有初步效果提升 |
| AV2-W08 Auto-skill | auto-skill pipeline -> online service | 设计并完成整个 pipeline，最终做成线上 service |
| AV2-W13 Benchmark | URL 高频站点 -> 任务类别 -> skill | pipeline 已 assign 给老板，产出回流 benchmark 和 auto-skill taxonomy |
| AV2-W14 Agent 生图 | 生图主链路基本 close | 剩一个可配置项的前后端收尾 |
| AV2-W15 脚本 skill | 脚本 skill 已上线 | 后续补线上 smoke、典型 case 和安全/执行边界 |

## Priority Board

| Priority | Focus | Owner | Success Signal |
| --- | --- | --- | --- |
| P0 | 高速模型 harness | 我 | Qwen 3.7 Plus / 其他高速模型能用统一 harness 比较质量、延迟、成本、tool calling 和 rollout 风险 |
| P0 | trajectory -> skill | 我 | 输入 trajectory、输出 skill、初步效果信号和失败样式被记录 |
| P0 | auto-skill pipeline / service | 我 | 场景 taxonomy、生成策略、验证/无验证分支、线上 service API、状态/版本/回滚边界明确 |
| P1 | benchmark pipeline handoff / taxonomy | 老板 / 我 | URL 高频站点、任务类别、skill candidate、coverage gap 和第一批 eval case 字段明确 |
| P1 | 生图可配置项前后端收尾 | 我 | 可配置项前后端一致，线上默认值、展示、失败态和最终 smoke 清楚 |
| P2 | 脚本 skill 上线后验证 | 我 | 线上 smoke、典型 case、执行/安全边界和 benchmark 回流记录清楚 |

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| 高速模型只看单一分数 | 可能选出延迟、成本或 tool calling 不适合线上链路的模型 | harness 同时记录质量、延迟、成本、tool calling、失败样式和 rollout 风险 |
| 无验证生成 skill | 只能看到趋势，不能判断稳定性或泛化 | 明确标注为 experiment，记录失败样式，后续补验证分支 |
| auto-skill 场景边界不清 | 生成的 skill 难维护、难复用、难 service 化 | 先建场景 taxonomy、输入输出契约、状态机和版本/回滚策略 |
| URL 高频站点 taxonomy 只看频率 | 可能漏掉低频但高价值或高风险任务 | taxonomy 增加价值/风险/coverage gap 字段 |
| 生图剩余配置项前后端不一致 | 线上默认值、UI 展示和真实调用参数可能不一致 | 收尾 schema/default/settings/展示/失败态，并补最终 smoke |
| 脚本 skill 上线后缺 smoke | 后续效果或安全问题难归因 | 补线上 smoke、典型 case、失败恢复和边界记录 |

## Checklist

- [ ] 高速模型 harness 初稿。
- [ ] trajectory -> skill 第一版实验记录。
- [ ] auto-skill pipeline / online service 初稿。
- [ ] benchmark pipeline handoff 字段和同步节奏明确。
- [ ] 生图可配置项前后端收尾记录。
- [ ] 脚本 skill 上线后 smoke / benchmark 回流记录。
