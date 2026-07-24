# Day 09 — 数据、Chat Template 与 Loss Mask

日期：`2026-08-04`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

能够逐 token 解释一条 SFT 数据经过 template 后的 input、labels 和 loss mask，并实现训练前数据校验器。

## 理论（75 分钟）

精读清单：[Day 09 — Dataset、template 与 loss token](../SCALING-BOOK-READING-GUIDE.md#day-09)。按 schema、SFT loss、多轮 mask、真实 sample 源码对照四步完成。

- Causal LM shift、ignore index、response-only loss、EOS/padding/truncation。
- 多轮对话训练所有 assistant turns 还是最后一轮。
- Packing 时的 position、attention 与 label boundary。
- 阅读 [ms-swift Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) 的 SFT/loss 字段。

## Coding（150 分钟）

- 实现 `validate_sft_data.py`：role 顺序、空回答、重复 ID、超长、非法字符、train/eval 泄漏。
- 实现 `inspect_template.py`：打印 raw messages、rendered prompt、token、labels、loss token count。
- 为单轮、多轮、system、空 response、截断、`loss=false` 添加测试。

## 训练 / 实验（45–60 分钟）

- 不跑长训练。
- 用 20 条 edge cases 编码；人工审查其中 5 条完整 token/label。
- 可选：0.6B 跑 5–10 steps，确认 trainer 的有效 label tokens 与脚本一致。

## 资源与租卡

- CPU only；如果模型只在 AutoDL cache，使用无卡模式。
- 可选 GPU smoke 不超过 30 分钟，不为 tokenizer/data validation 开 H100。

## 验收

- [ ] 随机给一个 messages sample，能预测哪些 token label 为 `-100`。
- [ ] 校验器主动拒绝至少五类错误。
- [ ] 形成 `artifacts/data/sft-schema.md` 和 20 条 edge cases。

## Daily Log

### Loss mask 结论

### 校验器抓到的问题

### Day 10 第一动作
