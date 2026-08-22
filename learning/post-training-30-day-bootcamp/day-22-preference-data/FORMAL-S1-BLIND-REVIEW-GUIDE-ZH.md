# Day 22 Formal S1 中文盲审指南

中文 reviewer 页面：[打开离线盲审页](../artifacts/data/day22-qwen35-formal-s1-blind-review-zh.html)

页面只把操作说明、字段标签和选项中文化。每个 case 的题目、回答 A、回答 B 均保持冻结原文；页面不包含 concealed key，也不会显示哪一边是 verifier chosen。

## 怎么审

- 共 50 个独立 pair，每个换序展示两次，所以要完成 100 次判断。
- 按页面顺序独立判断，不寻找对应的换序 case，不查看 concealed key，不运行测试。
- 优先判断代码功能是否正确，再判断是否遵守 prompt 的格式要求。
- 不要因为答案更长、写法更漂亮或位于 A/B 的某一侧而偏好它。

判定选项：

- `A 更优`：A 在正确性和指令遵循上明显更好。
- `B 更优`：B 在正确性和指令遵循上明显更好。
- `基本相同`：两边看起来同样好或同样差。
- `无法判断`：仅凭题目和可见回答无法可靠判断。
- `样本有问题`：题目或展示存在污染、损坏等问题，不适合纳入审计。

置信度选 `低 / 中 / 高`。选择“基本相同 / 无法判断 / 样本有问题”时必须填写备注。

## 保存与提交

页面会把进度保存在当前浏览器的 localStorage。可以随时点击“导出草稿 JSONL”备份；“导出完成版 JSONL”只有在 100 条全部有效填写后才会启用。

导出的 JSONL 保留原始 schema，仅把：

- `verdict` 写为 `A / B / tie / ambiguous / reject`；
- `confidence` 写为 `low / medium / high`；
- `notes` 写为你的备注。

因此 completed JSONL 可以直接交给 `finalize_day22_formal_s1_review.py`，无需再次转换。

注意：冻结源 worksheet、concealed key 和 pending manifest 都不要编辑。中文页面是独立 reviewer view，不属于正式证据本体。
