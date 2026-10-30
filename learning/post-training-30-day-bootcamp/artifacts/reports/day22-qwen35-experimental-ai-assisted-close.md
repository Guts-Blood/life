# Day 22 experimental AI-assisted close

状态：`completed_experimental_ai_assisted`。这是用户明确授权的实验关闭路径，不是 formal human-reviewed readiness。

## 关闭结果

- 原 machine-verified pairs：200
- Codex 复核争议 pairs：11
- 保留为实验性 DPO 输入：200
- 裁决后剔除：0
- split：train/dev/heldout = 154/17/29
- Formal human-review readiness：`false`（没有伪装成人工审阅）。

## Codex 争议裁决

| case | pair | A/B 分数 | decision | 映射 | action | issue | rationale |
|---|---|---:|---|---|---|---|---|
| `adjudicate:1ae243b4ad3bbdb8604d9391` | `mbpp:task:867:s1pair:d8a396966399e806` | 9/5 | prefer_A | chosen | keep | ambiguous_spec | 公开测试将“最小数”限定为最小正整数：奇数和加1、偶数和加2，A三测全符；B在偶数和时返回0。两者都依赖n与数组长度一致。 |
| `adjudicate:62e775ec2aa061126bc964dc` | `mbpp:task:869:s1pair:a7969ac9f64de058` | 7/2 | prefer_A | chosen | keep | both_incorrect | A符合全部公开测试，但只检查子列表首项，未确保其余元素也在范围内，空子列表还会越界；B遍历的x必然属于list1，过滤条件恒假，结果恒为空。 |
| `adjudicate:304f22506accbec8be0daf0b` | `mbpp:task:868:s1pair:4c5243bdedae0c83` | 7/2 | prefer_A | chosen | keep | boundary_or_type_risk | A三项公开测试均正确，但尾随空格会把计数清零，且只把普通空格视作分隔符；B对含空格字符串多算分隔符长度，对单词字符串返回0，仅空串碰巧正确。 |
| `adjudicate:be637de836af7b7c38ce6e80` | `mbpp:task:777:s1pair:7520fe1d8a56a799` | 9/5 | prefer_A | chosen | keep | ambiguous_spec | 测试明确要求每个不同值只计一次，A三测全符；其缺点是原地排序会修改输入。B只累加出现恰好一次的值，是对“non-repeated”的另一种理解，但与全部期望值不符。 |
| `adjudicate:4193eb5bcd7c8dfb6287896f` | `mbpp:task:899:s1pair:faa9e3c477db73eb` | 2/6 | prefer_B | chosen | keep | ambiguous_spec | 题目未定义“只取角元素”的具体操作与排序判定，且公开样例全为True。B静态可判三测全符，A前两测即返回False；不过B的局部相邻条件也不足以一般性刻画可排序性。 |
| `adjudicate:37c8dcc0cc44ae65a952bdc6` | `mbpp:task:799:s1pair:5fce947002254843` | 5/2 | prefer_A | chosen | keep | boundary_or_type_risk | A在所有公开小数样例上等同左移并通过，但没有截断到32位，也未正确回卷溢出位，d越界还可能产生负移位。B掩码仅31位且回卷表达式错误，除首例外多项公开测试失败。 |
| `adjudicate:e69a4a278fa873a5339c497b` | `mbpp:task:750:s1pair:05972b444fd5576d` | 5/10 | prefer_B | chosen | keep | none | 测试要求把元组元素展开追加到列表。B用list(test_tup)拼接，结果、顺序及非修改性均正确；A把整个元组作为单个嵌套元素，三项测试都不符。 |
| `adjudicate:1ddf27e3b2d2da3d895ab6ae` | `mbpp:task:822:s1pair:307dad4b07891d89` | 7/3 | prefer_A | chosen | keep | ambiguous_spec | 题目未说明有效密码规则；按公开测试可推断至少需大小写字母和数字，A三测全符。B禁止特殊字符，因此会拒绝唯一正例Password@10，且其规则也未要求小写字母。 |
| `adjudicate:5aea24051a6d8b2ea9ed13a6` | `mbpp:task:758:s1pair:c29a1d61f894b721` | 10/2 | prefer_A | chosen | keep | none | A把每个子列表转成元组键并正确累计，覆盖重复项、单项和空输入。B转成元组后却在原始列表的列表元素中计数，列表与元组不相等，因而所有计数都会是0。 |
| `adjudicate:a4aa1991f760b3dd6d76c489` | `mbpp:task:719:s1pair:bac0ac6ff201aedd` | 7/10 | prefer_B | chosen | keep | ambiguous_spec | 两者使用相同且可匹配a后零个或多个b的搜索模式；题面未写明返回格式，所以A的布尔值有语义合理性，但公开测试明确要求两种提示字符串，只有B三测全符。 |
| `adjudicate:f960be9558218890d1828fcf` | `mbpp:task:713:s1pair:647b18a8964e1794` | 4/7 | prefer_B | chosen | keep | ambiguous_spec | “valid values”未被定义；对公开布尔元组，B以大于0判定，三测全符。A检查自反相等，False也等于自身，故错误接受含False样例；B推广到字符串、None或负数时仍有类型或语义风险。 |

## 边界

- 原 formal manifest、pair 文件与 human-review blocker 保持不变。
- 实验输入只做子集过滤；每个保留 pair 的文本、执行证据、processor audit 与 pair hash 均与 machine-verified source 完全相同。
- Codex 若偏好 verifier-rejected 一侧，只会导致该 pair 被剔除，不会翻转 preference label。
