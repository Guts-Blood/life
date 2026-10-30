# Chart map

| Section | Question | Family / type | Dataset and fields | Supported takeaway | Palette policy | Delivery |
|---|---|---|---|---|---|---|
| 非 Code 结果 | Base 与三个 LR 在 General、Math、Finance 各答对多少题？ | Comparison / grouped bar | `slice_scores`: candidate × slice → correct, denominator | 高 LR 改善 General，但 1e-5/3e-5 牺牲 Math；1e-4 非 Code 总分最高 | relaxed multi-category；三项 slice 用受控分类色，候选由横轴和直接标签识别 | canonical artifact chart in `report.html` |

只有一个图，因为 Code 尚无 E2B 正确率；静态资格、格式率和工程事件保留为精确表格与叙述，避免把不同分母或不同语义强行画成同一“性能”曲线。
