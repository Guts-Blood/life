# Day 08 — SFT 数据契约、Chat Template 与 Loss Mask

日期：`2026-08-03`

状态：`not_started`

强度：工作日 4–5 小时

## 主要目标

在任何 SFT 运行之前，固定一条样本从 raw messages 到监督 token 的完整契约。今天的成功标准是“知道模型到底在学哪些 token”，不是先把训练命令跑起来。

## 理论（75 分钟）

精读清单：[Day 08 — SFT data contract 与 loss token](../SCALING-BOOK-READING-GUIDE.md#day-08)。

- SFT schema：稳定 sample ID、`system/user/assistant` role 顺序、多轮边界、metadata。
- Qwen chat template：special tokens、generation prompt、thinking/non-thinking 控制与 EOS。
- Causal LM shift：`logits[t]` 预测哪个 label；`-100`/ignore index 如何屏蔽 prompt、padding 和被截断区域。
- assistant-only loss 与 all-token loss 的差异；多轮中训练全部 assistant turns 还是指定 turns。
- packing、padding、truncation 的顺序，以及它们如何改变有效监督 token 数。

## Coding（135 分钟）

- 实现最小 `inspect_sft_sample.py`，对单条 sample 输出：
  `raw messages -> rendered text -> input_ids/tokens -> labels -> supervised mask -> shifted targets`。
- 给以下 5 类样本建立 golden cases：单轮、多轮、含 system、超长截断、空/缺失 assistant。
- 实现窄范围 schema validator：role 顺序、空响应、重复 ID、最大长度、EOS 与有效 label token 数。
- 固定 tokenizer revision、chat template 内容及其 hash；不依赖“当前默认模板”。

## 训练 / 实验（45–60 分钟）

- 不训练模型。
- 用 Qwen3 tokenizer 编码 20 条 edge cases，人工逐 token 审查其中 5 条。
- 比较至少两种 mask 策略，只报告监督 token 数和边界差异，不根据 loss 猜测正确性。

## 资源与租卡

- [ms-swift Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)
- [Transformers Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 中与 thinking mode/template 相关段落
- CPU only；若 tokenizer 已在远端缓存，使用无卡实例，不开 H100。

## 产物

- `../artifacts/data/day08-sft-data-contract.md`
- `../artifacts/data/day08-template-golden-cases.jsonl`
- `../artifacts/reports/day08-token-label-audit.md`

## 验收

- [ ] 随机给一条 messages sample，能在编码前预测哪些文本会参与 loss。
- [ ] 5 个 golden cases 都有 raw/rendered/token/label/mask 对照。
- [ ] validator 能拒绝空监督、非法 role 顺序和截断后零有效 label 三类关键错误。
- [ ] tokenizer revision、template hash、max length、EOS/padding/truncation policy 已冻结。
- [ ] 没有在数据契约未通过时启动训练。

## Daily Log

### 一个此前误判的 loss 边界

### 有效监督 token 统计

### Day 09 第一动作
