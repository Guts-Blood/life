# Day 08 — Token / Label Audit

日期：`2026-08-03`

Tokenizer 与 policy 见 [`day08-sft-data-contract.md`](../data/day08-sft-data-contract.md)。本报告只审计数据变换，不运行模型或训练。

## Golden result

20 条 edge cases 的结果：

| 结果 | 数量 | 覆盖 |
|---|---:|---|
| accept | 10 | 单轮、多轮、system、Unicode、代码、多行、长 prompt/answer、空白包围的非空回答 |
| schema reject | 8 | 空/缺失 assistant、非法顺序、重复 system、空 user、非法 role、控制标记注入 |
| encoded reject | 2 | 最终 assistant 被截断、截断后零监督 token |

独立 unit tests 覆盖合法 schema、空 assistant、非法 role 顺序与重复 ID。

## Five manually inspected samples

`all-token terms` 按 causal shift 后最多 `T-1` 个目标计算；比例用于说明同一 raw sample 在两种 mask 策略下的 objective 权重差异。

| Sample | Input tokens | Assistant labels | Supervised spans | Assistant / all-token terms | Supervised end tokens |
|---|---:|---:|---|---:|---:|
| `single-turn` | 32 | 2 | `29–30` | `2/31 = 6.5%` | 1 |
| `multi-turn` | 53 | 10 | `26–28`, `45–51` | `10/52 = 19.2%` | 2 |
| `with-system` | 49 | 23 | `25–47` | `23/48 = 47.9%` | 1 |
| `multiline-code` | 44 | 17 | `26–42` | `17/43 = 39.5%` | 1 |
| `long-answer-fits` | 71 | 40 | `30–69` | `40/70 = 57.1%` | 1 |

五条样本均确认：assistant header 被 mask，assistant body 与其 `<|im_end|>` 被监督，结束标记后的换行重新被 mask。

## Single-turn boundary trace

Raw：

```json
{"id":"single-turn","messages":[{"role":"user","content":"2 + 2 等于多少？"},{"role":"assistant","content":"4"}]}
```

Rendered：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
2 + 2 等于多少？<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

边界：

| Position | Token | Label | Direct loss? | Shifted source |
|---:|---|---:|:---:|---:|
| 24 | user `<|im_end|>` | `-100` | no | — |
| 26 | `<|im_start|>` | `-100` | no | — |
| 27 | `assistant` | `-100` | no | — |
| 28 | newline | `-100` | no | — |
| 29 | `4` | `19` | yes | `logits[28]` |
| 30 | assistant `<|im_end|>` | `151645` | yes | `logits[29]` |
| 31 | newline | `-100` | no | — |

此前容易误判的是：位置 28 的 label 虽然是 `-100`，但 `logits[28]` 仍参与 loss，因为它预测位置 29 的有效 label。

## Mask strategy comparison

assistant-only 与 all-token loss 不能直接比较数值高低。后者把 system、user、role/header 等额外目标加入分子与分母；固定模板 token 往往较容易预测，可能降低平均 loss，却不能证明回答能力更强。

若 loss 正常下降但模型复述 system/user，第一项最小验证是抽取同一条样本的 `role/token/label/predicted_by_logit`，检查 prompt labels 是否意外没有写成 `-100`。随后再检查 generation prompt 与 stop token，而不是先修改学习率。

## Validation evidence

```text
unit tests: 4 passed
golden cases: 20/20 matched expected stage and error code
offline reload: passed from pinned local tokenizer snapshot
training/GPU: not started
```
