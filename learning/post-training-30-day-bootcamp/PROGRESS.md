# Bootcamp Progress

开始日期：`2026-07-27`  
目标完成日期：`2026-08-25`  
当前 Phase：`not_started`  
当前最大阻塞：`none`

状态使用：`not_started`、`in_progress`、`blocked`、`done`。

| Day | 主题 | 状态 | 日期 | 用时 | 核心产物 | 一句话结论 |
|---:|---|---|---|---:|---|---|
| 01 | 环境与可复现基线 | not_started | 2026-07-27 | | | |
| 02 | Transformer accounting | not_started | 2026-07-28 | | | |
| 03 | Roofline 与 H100 | not_started | 2026-07-29 | | | |
| 04 | 分布式并行地图 | not_started | 2026-07-30 | | | |
| 05 | Ultra-Scale 与框架分层 | not_started | 2026-07-31 | | | |
| 06 | 周末：Scaling 阅读 | not_started | 2026-08-01 | | | |
| 07 | 周末：Week 1 复盘 | not_started | 2026-08-02 | | | |
| 08 | Qwen SFT smoke | not_started | 2026-08-03 | | | |
| 09 | 数据、template 与 loss mask | not_started | 2026-08-04 | | | |
| 10 | Frozen eval baseline | not_started | 2026-08-05 | | | |
| 11 | 1.7B full SFT smoke | not_started | 2026-08-06 | | | |
| 12 | 1.7B 主 SFT | not_started | 2026-08-07 | | | |
| 13 | 周末：Qwen3/SFT 阅读 | not_started | 2026-08-08 | | | |
| 14 | 周末：Week 2 复盘 | not_started | 2026-08-09 | | | |
| 15 | Packing/sequence ablation | not_started | 2026-08-10 | | | |
| 16 | Batch/显存优化 ablation | not_started | 2026-08-11 | | | |
| 17 | Megatron 主链路通读/准备 | not_started | 2026-08-12 | | | |
| 18 | Megatron 双卡 SFT/runtime trace | not_started | 2026-08-13 | | | |
| 19 | Megatron checkpoint/Profiler/32B | not_started | 2026-08-14 | | | |
| 20 | 周末：MoE 阅读 | not_started | 2026-08-15 | | | |
| 21 | 周末：Eval 阅读 | not_started | 2026-08-16 | | | |
| 22 | 35B-A3B MoE capacity | not_started | 2026-08-17 | | | |
| 23 | 公共与 deterministic eval | not_started | 2026-08-18 | | | |
| 24 | Pairwise、统计与 SFT 报告 | not_started | 2026-08-19 | | | |
| 25 | DPO 理论与数据 | not_started | 2026-08-20 | | | |
| 26 | slime 主链路通读/运行准备 | not_started | 2026-08-21 | | | |
| 27 | 周末：GRPO/Online RL 阅读 | not_started | 2026-08-22 | | | |
| 28 | 周末：slime 架构/源码复核 | not_started | 2026-08-23 | | | |
| 29 | slime 多卡 RL/修改重跑 | not_started | 2026-08-24 | | | |
| 30 | 30B+ 设计与 clean reproduction | not_started | 2026-08-25 | | | |

## 每周 Gate

- [ ] Week 1：能手算参数/FLOPs/显存，解释 Roofline 与主要并行策略，并指到实现代码。
- [ ] Week 2：能从零跑 Qwen LoRA/full SFT，验证 loss mask，并用 frozen set 比较 Base 与 SFT。
- [ ] Week 3：能从 `pretrain_gpt.py` 讲到 optimizer/checkpoint，运行双卡 Megatron SFT、插桩并解读 trace。
- [ ] Week 4：能设计 35B-A3B 容量方案、构建可信 Eval、推导 DPO，并画出 slime RL 数据流。
- [ ] Final：能运行 slime 多卡闭环、修改 reward 后重跑，提交 30B+ design 并完成 15 分钟模拟评审。
