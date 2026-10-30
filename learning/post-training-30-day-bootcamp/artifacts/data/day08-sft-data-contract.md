# Day 08 — SFT Data Contract

状态：`frozen-for-audit`

范围：纯文本 `system/user/assistant`，监督所有 assistant 回合，不含 tools、multimodal、packing 或训练。

## 1. Raw sample schema

```json
{
  "id": "stable-non-empty-string",
  "messages": [
    {"role": "system|user|assistant", "content": "non-empty-string"}
  ],
  "metadata": {"source": "optional", "license": "optional"}
}
```

- `system` 可省略；若存在，只能有一个并位于开头。
- 后续严格按 `user -> assistant` 交替，最后一条必须是非空 assistant。
- 数据集范围内 `id` 唯一；`metadata` 若存在必须是 object。
- raw content 禁止包含 `<|im_start|>`、`<|im_end|>`，避免模板控制标记注入。

## 2. Frozen tokenizer/template

| 项目 | 固定值 |
|---|---|
| Model/tokenizer | `Qwen/Qwen3-0.6B-Base` |
| Revision | `ddc928429ed09d9ad603fd762053d0434c15e865` |
| Transformers | `4.57.3` |
| Tokenizer class | `Qwen2TokenizerFast` |
| Template SHA-256 | `44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5` |
| tokenizer config SHA-256 | `7d0fc4691628c9f6b0c77f8138a80517896a307e074dd5ac87e279ea85515774` |
| tokenizer.json SHA-256 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| vocab.json SHA-256 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

冻结的完整 `tokenizer_config.json`（含 chat template）保存为
[`day08-qwen3-tokenizer-config.json`](day08-qwen3-tokenizer-config.json)。

特殊 token：

| 语义 | Token | ID |
|---|---|---:|
| BOS | `None` | `None` |
| assistant/message end 与 tokenizer EOS | `<|im_end|>` | `151645` |
| padding | `<|endoftext|>` | `151643` |

该 revision 的模板具有两个需显式记录的行为：

1. raw messages 没有 system 时，自动插入 `You are a helpful assistant.`。
2. 模板不引用 `enable_thinking`，因此传入 `enable_thinking=False` 不改变输出；本契约的 rendered text 中没有 thinking block。

训练已有 assistant 回答时固定 `add_generation_prompt=False`。若先渲染文本再 tokenize，固定 `add_special_tokens=False`，避免模板 special tokens 被二次添加。

## 3. Labels and loss mask

- `attention_mask=1`：所有未 padding 的 rendered tokens，包括 system、user、assistant header、assistant body。
- `labels=input_ids`：每个 assistant body 以及紧随其后的 `<|im_end|>`。
- `labels=-100`：system、user、所有 role header、role 之间的换行和 padding。
- 多轮样本监督所有 assistant turns；不是仅监督最后一轮。

因果 shift 固定为：`logits[t]` 预测 `labels[t+1]`。有效 loss 项数等于 shift 后非 `-100` 的目标数，不等于 input token 数或样本数。

## 4. Length, truncation, padding

| 项目 | Policy |
|---|---|
| Default audit max length | `256` tokens |
| Truncation side | right |
| Incomplete final assistant | reject/drop；不得把答案前缀当完整 target |
| Zero effective labels | reject |
| Missing supervised `<|im_end|>` | reject |
| Padding side | right |
| Padding attention mask | `0` |
| Padding label | `-100`，即使 padding input ID 是 special token |
| Packing | disabled |

必须先按相同索引截断 `input_ids/attention_mask/labels`，再运行 post-truncation validator。`attention_mask=0` 不会自动让 cross entropy 忽略该位置。

## 5. Validator boundary

Schema 阶段拒绝：重复 ID、非法 role 顺序、空 content、缺失 assistant、非法 metadata、模板控制标记注入。

Encoded 阶段拒绝：token 数策略违规、截断后零有效 label、最终 assistant `<|im_end|>` 被截掉或没有被监督。

实现：[`inspect_sft_sample.py`](../../day-08-sft-data-contract/inspect_sft_sample.py)。20 条 edge cases：[`day08-template-golden-cases.jsonl`](day08-template-golden-cases.jsonl)。

## 6. Known limits

- 只支持 string content 和三个 role；tools、reasoning content、图片及音视频必须建立新契约。
- 当前 role-aware mask 依赖冻结的 ChatML 标记与模板 hash；template hash 变化必须重新跑全部 golden cases。
- 本文件不决定真实训练数据的 source/license/revision；Day 09 的 dataset manifest 仍需补齐。
