# Day 22 混合盲审审计（10 人工 + 90 AI-assisted）

结论：这是一份校准与风险审计，不是 100 条人工盲审。Formal human-review gate 保持 `BLOCKED`，不得把本报告或 annotations 文件作为 ready manifest 的替代品。

## 结果

- 展示数：100（人工 10，sub-agent 90）
- unique pairs：50
- 全部 directional 展示与执行 verifier 方向一致：84/89（94.4%）
- 人工 10 条一致：6/10（60.0%）
- sub-agent 90 条一致：78/79（98.7%）
- 两次展示都给出方向的 pair：43
- 方向 position-consistent：39（90.7%）
- human/sub-agent 混合判断的 pair：10；这些不能解释为单一 reviewer 的 position-bias 测试。
- 40 个纯 sub-agent pair：35 个双向明确，其中 35 个换位后一致；5 个含非定向判断。
- 10 个 human/sub-agent 混合 pair：8 个双向明确，其中 4 个语义一致；由于 reviewer 不同，不能把差异直接归因于 position bias。

## 与执行 verifier 方向不一致的展示

| 行 | reviewer | pair | 判断 | 置信度 | 备注 |
|---:|---|---|---|---|---|
| 3 | human | `mbpp:task:867:s1pair:d8a396966399e806` | B | high |  |
| 6 | human | `mbpp:task:869:s1pair:a7969ac9f64de058` | B | medium |  |
| 7 | human | `mbpp:task:868:s1pair:4c5243bdedae0c83` | B | high |  |
| 8 | human | `mbpp:task:758:s1pair:c29a1d61f894b721` | A | high |  |
| 81 | subagent | `mbpp:task:777:s1pair:7520fe1d8a56a799` | A | high | A只累加出现恰好一次的元素；B会把每个不同值累加一次，包括重复出现的值。 |

## 非定向判断

| 行 | reviewer | pair | 判断 | 置信度 | 备注 |
|---:|---|---|---|---|---|
| 17 | subagent | `mbpp:task:822:s1pair:307dad4b07891d89` | ambiguous | high | 题面没有定义密码有效规则；A 假定长度和字符集规则，B 假定大小写及数字规则，无法据此判定。 |
| 18 | subagent | `mbpp:task:899:s1pair:faa9e3c477db73eb` | tie | high | 两者同样只有局部判断：A 仅看边缘相邻项，B 遇一组局部条件即提前返回；均未验证完整取数过程。 |
| 24 | subagent | `mbpp:task:799:s1pair:5fce947002254843` | tie | high | 两者同样未实现正确的 32 位回卷：A 只用单比特掩码，B 使用 31 位掩码且未把高位正确右移到低位。 |
| 26 | subagent | `mbpp:task:750:s1pair:05972b444fd5576d` | ambiguous | high | “将 tuple 加到 list”未说明是把 tuple 作为单个元素追加还是把其元素展开；A、B 分别实现这两种合理语义。 |
| 27 | subagent | `mbpp:task:822:s1pair:307dad4b07891d89` | ambiguous | high | 题面缺少有效密码的必要条件；A 与 B 采用不同且互不包含的规则，不能客观择优。 |
| 28 | subagent | `mbpp:task:713:s1pair:647b18a8964e1794` | ambiguous | high | 题面未定义“valid values”：A 将其解释为正数，B 实际仅排除 NaN，二者都依赖未说明的标准。 |
| 48 | subagent | `mbpp:task:719:s1pair:bac0ac6ff201aedd` | ambiguous | medium | 两者使用相同正则，差别仅是返回布尔值还是提示字符串；题面未规定返回类型，无法据此判定优劣。 |
| 53 | subagent | `mbpp:task:899:s1pair:faa9e3c477db73eb` | ambiguous | medium | 题面未定义“只取角元素”的具体操作及排序过程；A与B检查的是不同局部条件，均无法仅由可见描述推出。 |
| 56 | subagent | `mbpp:task:719:s1pair:bac0ac6ff201aedd` | ambiguous | medium | 两者使用相同正则，差别仅是提示字符串与布尔值；题面未规定返回类型，无法据此判定优劣。 |
| 61 | subagent | `mbpp:task:867:s1pair:d8a396966399e806` | ambiguous | medium | 奇数和时两者都补1；偶数和时A允许补0、B要求补最小正偶数2，而题面未说明所加数字是否必须为正数。 |
| 79 | subagent | `mbpp:task:713:s1pair:647b18a8964e1794` | ambiguous | high | 题目未定义“valid values”的判定标准；A隐含排除NaN，B隐含要求正数，无法仅凭提示确定。 |

## 使用边界

- verifier agreement 不是 ground-truth accuracy；MBPP prompt/test 可能存在口径歧义，公开测试也较弱。
- preference 的首要判断标准是功能正确性，其次是格式遵循；代码风格只在功能等价时作为次要因素。
- 高度 verifier agreement 也不能证明 sub-agent 可以替代人类：reviewer 与生成模型可能共享代码先验，而且当前 pair 本来就是按执行 verifier 选出的。
- 本产物可以发现歧义、弱测试和明显偏好捷径，但不完成原协议要求的人工盲审。
