# Day 25 — DPO 理论与 Preference Data

日期：`2026-08-20`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

理解 DPO objective 与 reference policy，并从 SFT bad cases 构造可审计的 chosen/rejected 数据。

## 理论（90 分钟）

精读清单：[Day 25 — DPO objective](../SCALING-BOOK-READING-GUIDE.md#day-25)。按 preliminaries、objective 推导、experiments、limitations 阅读，不只看公式截图。

- 阅读 [DPO Paper](https://arxiv.org/abs/2305.18290) 的 abstract、RLHF setup、DPO objective、实验/限制。
- Policy/reference、Bradley–Terry、implicit reward、beta、preference margin。
- DPO 不需要在线 rollout，但仍可能过拟合 preference artifacts。

必须能用自己的话解释 DPO loss 中 `log πθ - log πref` 对 chosen/rejected 的作用。

## Coding（90 分钟）

- 实现/完善 preference-data validator：chosen/rejected 非空、重复、长度差、模板、泄漏。
- 建立 DPO train/eval split 和 dataset manifest。
- 用手写的小 tensor/log-prob 示例实现 DPO loss 单元测试，验证 chosen/rejected、reference 和 beta 的方向，不调用高层 Trainer。

## 训练 / 实验（90 分钟数据实验）

- 从 Day 24 bad cases 构造或筛选 200–500 pairs。
- 统计 chosen/rejected length、类别、难度、来源。
- 人工审查至少 50 pairs；剔除偏好不明确或只因长度占优的样本。

格式参考：[ms-swift DPO 数据格式](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#dpo-orpo-cpo-simpo-rm)

## 资源与租卡

- CPU only；不租 GPU。本月不把 DPO 长训练放在 Core，重点保留 objective/data/eval 判断力。
- 确认 Day 26 slime clone、Docker、Qwen3-4B、数据和存储预算清单，但今天不下载/转换。

## 验收

- [ ] 200–500 pairs，有独立 held-out preference eval。
- [ ] 长度偏差和类别分布已报告。
- [ ] 50 对人工审查通过预设质量阈值。
- [ ] 10-pair encode 与手写 DPO loss test 通过，能解释每个 log-ratio 的方向。

## Daily Log

### Preference 数据统计

### 最大数据风险

### Day 26 slime 第一动作
