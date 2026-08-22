---
period: 2026-W23
date_range: 2026-06-01 to 2026-06-07
created_at: 2026-06-02
---

# Weekly Review: Agent V2 / 2026-W23

## Focus

| Workstream | Focus | Evidence / Notes |
| --- | --- | --- |
| AV2-W14 Agent 生图 | 重构生图链路代码 | 重点确认审核链路和生图角标 |
| AV2-W14 Agent 生图 | E2B + 线上配置收口 | 生图放 E2B 链路；runtime tool 加载小重构；SQL/settings 上线；`image_gen` skill 上传线上 |

## Priority Board

| Priority | Focus | Owner | Success Signal |
| --- | --- | --- | --- |
| P0 | 生图链路重构 | 我 | runtime/tool 调用、审核、artifact 写入、产物展示和失败态可回归 |
| P0 | 审核链路 | 我 | `poresche` 调用点、日志、重试/兜底和用户失败态明确 |
| P0 | 生图角标 | 我 | 角标状态映射准确，能覆盖生成中、审核中、完成、失败、审核拦截、重试中 |
| P0 | 生图 E2B 链路 | 我 | 生图请求走 E2B，产物能稳定回传到 agent artifact surface |
| P0 | runtime tool loading 重构 | 我 | 后端 runtime tool loading 和前端展示/触发契约一致 |
| P0 | 线上配置和 skill 发布 | 我 | SQL/settings 生效；`image_gen` skill 在线上可加载、可触发、可 smoke |

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| 审核链路不稳定 | 生图请求在 daily 或线上被不稳定阻断 | 收集请求 id、输入、审核返回、重试状态和最终用户态 |
| 生图角标状态不准 | 用户无法判断图片是生成中、审核中、失败还是已完成 | 明确状态机和 UI 映射，补失败/审核 case |
| 重构引入回退 | 已进 daily 的生图主链路被改坏 | 做 daily smoke，记录 runtime/tool、审核、artifact、展示链路 |
| E2B 链路 artifact 回传不一致 | E2B 内生成成功但用户侧看不到或不能复用产物 | 明确文件路径、权限、artifact id 和回传 surface |
| 线上配置漂移 | SQL/settings 生效范围不清，默认模型/参数问题难归因 | 记录配置版本、灰度范围、回滚路径和 smoke 结果 |

## Checklist

- [ ] W14 生图链路重构完成或拆出剩余项。
- [ ] 审核链路和 `poresche` 问题有明确结论。
- [ ] 生图角标状态映射有明确结论。
- [ ] daily 主链路 smoke / 回归结论记录。
- [ ] 生图 E2B 链路 smoke 记录。
- [ ] runtime tool loading 前后端契约对齐。
- [ ] SQL/settings 配线上并记录验证/回滚。
- [ ] `image_gen` skill 上传线上并 smoke。
