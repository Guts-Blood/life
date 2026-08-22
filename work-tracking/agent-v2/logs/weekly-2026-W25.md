---
period: 2026-W25
date_range: 2026-06-15 to 2026-06-21
created_at: 2026-06-16
---

# Weekly Review: Agent V2 / 2026-W25

## Focus

| Workstream | Focus | Evidence / Notes |
| --- | --- | --- |
| AV2-W08 Auto-skill | 当前最高优先级：pipeline -> online service | 2026-06-16 priority update；高速模型 harness 因公司暂不支持 Qwen 后置 |
| AV2-W14 Agent 生图 | 输入审核已进 daily | 2026-06-15 新增输入审核，后续看误杀、失败态和日志 |
| AV2-W14 Agent 生图 | 更多 input parameters + file id | 后续 todo：支持更多输入 parameters；用 file id 替代 `image_url` |
| AV2-W17 小红书 skill | 站点封控边界 | 迭代后发现小红书封控非常强，需要记录边界和降级策略 |
| AV2-W16 高速模型 harness | 后置 | 公司侧暂时不支持 Qwen，先保留设计草稿 |

## Priority Board

| Priority | Focus | Owner | Success Signal |
| --- | --- | --- | --- |
| P0 | auto-skill pipeline / online service | 我 | trajectory / URL 任务类别 -> skill 生成 -> 验证/无验证 -> 发布/service 的链路清楚，并拆出实现步骤 |
| P1 | 生图输入审核 daily 观察 | 我 | 合规输入不误杀，高风险输入能拦截，用户态和日志清楚 |
| P1 | 生图 input parameters 扩展 | 我 | schema、默认值、校验、skill prompt 和前端展示一致 |
| P1 | 生图 file id 替代 `image_url` | 我 | tool / skill / artifact 都使用 file id 引用图片，旧链路兼容策略明确 |
| P1 | 小红书 skill 封控边界 | 我 | 封控样式、可做/不可做任务、降级策略和 benchmark case 明确 |
| P2 | 高速模型 harness / Qwen | 我 | 等公司侧 Qwen 支持或替代高速模型候选明确后恢复 |

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| auto-skill pipeline 被支线打散 | 当前唯一 P0 延误，service 化链路难收束 | 生图/小红书作为 P1 支撑 case 和边界，P0 集中拆 pipeline/service |
| 输入审核误杀或用户态不清 | 用户会把审核拦截误认为生图失败 | 记录 daily 通过/拦截/误杀样本，补清晰失败态 |
| 新 input parameters 前后端不同步 | 模型、tool 和 UI 对参数理解不一致 | 先定义 schema/default/validation，再同步 skill prompt 和前端展示 |
| file id 替代 `image_url` 兼容不足 | 旧 artifact 引用失效，图片无法复用 | 明确迁移期兼容、权限和生命周期策略 |
| 小红书封控强 | skill 会在登录、验证码、访问限制里浪费轮次 | 建封控态检测、任务边界和快速降级策略 |
| Qwen 暂不支持 | 高速模型 harness 会卡在平台前置条件 | 将 AV2-W16 后置，只保留设计草稿 |

## Checklist

- [ ] auto-skill pipeline / online service 初稿。
- [ ] auto-skill service API、状态机、版本、发布/回滚边界明确。
- [ ] 生图输入审核 daily 样本记录。
- [ ] input parameters 扩展方案明确。
- [ ] file id 替代 `image_url` 方案明确。
- [ ] 小红书 skill 封控边界和 benchmark case 记录。
- [x] 高速模型 harness 因 Qwen 支持不足后置。
