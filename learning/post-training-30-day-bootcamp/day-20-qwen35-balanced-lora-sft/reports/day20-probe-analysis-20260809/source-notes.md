# Day20 报告证据说明

本报告只使用已同步并逐文件核验的 canonical run `day20-qwen35-lora-20260809T042005Z`、AutoDL 工程快照和两份来源合同。所有候选输出只做 JSON、哈希、parser、AST 与 `compile(..., 'exec')` 静态核验；没有在本机执行模型生成的候选代码。

## 同步完整性

- canonical run：119 个文件、253,175,927 bytes；远端/本地相对路径 + SHA-256 清单逐字一致；清单 SHA-256 为 `3b4575b11d4e964dd2d0ea839054a4b77eb26cc2a43596a3aa8457284538fb32`。
- 工程快照 + 来源合同：18 个文件、600,236 bytes；远端/快照一致，远端 16 个工程文件与本地工作副本一致；清单 SHA-256 为 `5469b872f6dbdaf3bfe139c9323ecec707c62c532fa6abb62bc5a903b03c2a9c`。
- canonical `DAY20-MANIFEST.json` 文件 SHA-256 为 `32c5dafc4a273b7067091244379233e45dad6a02a64d168165b6da04d049720f`，内容自哈希为 `017ac87be748146b278601eba063bd77504f71d0ce95de46d5990f94fc99d30e`。
- 来源适配器清单内容自哈希为 `3971777b376495ece3212ffec8e77bd6768184943c018eb77b4f67bc76ac368b`。

## 评分证据

四个候选各有 32 条固定 diagnostic 记录，每个 General/Math/Finance/Code slice 各 8 条。所有 v2 row self-hash、raw/normalized output hash、summary self-hash 与 predictions 文件 SHA-256 均独立复算通过。详细 hash 位于 `evidence/probe_metrics.json`。

Code 的 `sandbox_execution_eligible` 只是 frozen parser + 静态组合检查，不是 HumanEval 正确率。E2B 结果、`PROBE-SELECTION.json`、main checkpoint、`DAY20-RESULTS.json` 与 `DAY20-PASS.json` 均不存在。

## 数据和训练身份

- probe：821 records，精确 16,000 supervised tokens。
- main（仅准备，未训练）：13,197 records，精确 256,000 supervised tokens。
- 来源池真实模板复核：16,123 records，794,383 supervised tokens。
- Base snapshot SHA-256：`a176a3c982da5480a0ca98280848474133d105d2e4ab45dc37ce2c68a7c2195b`。
- 运行环境：ms-swift commit `565a1ad586a21d24b23931c52d2c62b49c39bee8`；torch `2.10.0+cu126`；transformers `5.12.1`；peft `0.19.1`。

## 解释边界

报告中的“首行补 4 空格后 10/10 静态通过”只是一项反事实诊断：它没有执行候选，也没有跑 HumanEval tests，且不属于当前 frozen v2 评分协议。如采用，必须注册为新的 adapter/parser 协议并保留 raw/repaired 双哈希。
