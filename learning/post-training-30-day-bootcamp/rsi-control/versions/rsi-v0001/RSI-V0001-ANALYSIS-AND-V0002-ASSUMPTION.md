# RSI v0001 失败分析与 v0002 训练假设

## 结论

rsi-v0001 没有产生有效 candidate。问题不在 E2B、冻结评测或源数据 target，主因位于训练模板的 target representation / loss masking 层：数据中的 105/105 个 Code target 都以严格的四个 ASCII 空格开始，但实际进入 loss 的 labels 中 0/105 保留了这四个空格。

因此 LoRA 学到了“从第 0 列开始输出函数体”。随着 Code 监督剂量增加，Code execution eligibility 从 Base 的 7/8 下降到 t6000 的 3/8，并在 t12000–t24000 降为 0/8。t18000 与 t24000 的 8 个 Code raw outputs 全部逐字节相同，说明继续使用同一模板增加 token 已经平台化，只会浪费 GPU。

完整可重算证据见 [stage-a-root-cause-diagnosis.json](runs/run-001-probe-primary/attempts/attempt-001/evidence/stage-a-root-cause-diagnosis.json)。其 file SHA256 为 `604b38a32e49305d94aea06a86e714e58213d8309c9c9f3767c1b1ee46921742`，content SHA256 为 `c816ff51eca41f119493288c96e7b20c1b8513988ed9e1df4db7e4ec555da468`。

## 已验证 trajectory

| Candidate | Total | General | Math | Finance | Code E2B | Code eligible | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Base | 15 | 1 | 7 | 4 | 3 | 7/8 | 比较基线 |
| t6000 | 15 | 5 | 5 | 4 | 1 | 3/8 | gross fail |
| t12000 | 18 | 5 | 7 | 6 | 0 | 0/8 | gross fail |
| t18000 | 18 | 5 | 7 | 6 | 0 | 0/8 | gross fail |
| t24000 | 18 | 5 | 7 | 6 | 0 | 0/8 | gross fail |

冻结的 Probe strict gate 要求：总分至少 Base+2；Math、Finance、Code 各自最多回退 1；Code eligible 至少 7/8；基础设施错误为 0。四个 checkpoint 均未满足，selector 已发布 `stop_no_passing_probe`，selection SHA256 为 `5fdd36eb38271b458d78612b637ee26ca3e7168b84777df0178c46b4f24da519`。

## 失败层级

因果链已经逐层排除：

1. 数据层：105/105 Code assistant targets 的 byte 0–3 是四个空格，strict target contract 通过。
2. 模板表示层：pinned ms-swift `Qwen3_5Template._swift_prepare_inputs` 对 assistant content 调用 `strip()`，删除开头空格。
3. Loss masking 层：即使恢复文本，`default+ignore_empty_think` 的空 think 正则仍会把紧邻的 indentation token 掩为 `-100`。
4. 模型输出层：所有失败 Code 行都具有同一错误签名——未在 byte 0 输出恰好四个空格；给现有输出静态补四空格后，函数体的 continuation AST contract 可通过。
5. 基础设施层：E2B live preflight 在训练前完成，全部 Probe sandbox 的 infrastructure failures 为 0。

所以这是 `training_template_label_construction` 的确定性故障，不应通过扫更多 LR、增加 token 或降低评测门槛处理。

## rsi-v0002 假设与唯一改动

Primary lever：`model.target.representation`。

必要 dependent lever：`model.target.masking`。它不是第二个性能假设，而是让 representation 改动真正产生梯度的必要条件。

实现只做一次 balanced token exchange：

- 在 Code 训练 target 中恢复并监督开头的四空格 token（token id 257）；
- 保留 `<|im_end|>` 的监督，只把 ChatML 自动追加、且不属于 canonical assistant target 的最后 newline token（token id 198）改为 `-100`；
- 因此每条 Code row 的 supervised-token count 不变；Non-Code 的 input IDs 与 labels 必须逐字节不变；
- 推理仍使用原生 `qwen3_5`，只在训练侧使用独立 template alias。

以下变量全部冻结：样本 ID、messages、样本顺序、四技能 token 预算、temporal mix、LR `1e-4`、seed `20260809`、LoRA rank/alpha/dropout/target modules、checkpoint token milestones、eval cases、scorer、E2B、selector 和 promotion gates。

## GPU 前硬预检

任何一项失败都禁止启动 Swift：

- Probe 1232/1232、Main 15226/15226 的 sample ID、messages、source identity、顺序与 v0001 相同；
- 每行 supervised-token count 相同，Probe/Main 总量仍为 24k/320k；
- Probe 105/105、Main 1540/1540 Code labels 解码后逐字节以 canonical 四空格 target 开始；
- 所有 Code row 的最后 `<|im_end|>` 保持 supervised，只有结构性尾 newline 被 mask；
- Probe 1127/1127、Main 13686/13686 Non-Code input IDs 与 labels 和原生模板完全一致；
- core manifest、outer manifest、template contract、全量 token audit 和 E2B preflight 的 hash 链全部通过。

## 预测、falsifier 与预算

预测：至少一个 Stage-A checkpoint 达到 `eligibility >= 7/8`、`Code >= 2`、`total >= 17`、`Math >= 6`、`Finance >= 3`、`infra = 0`；最可能是 t12000，预期约为 G5/M7/F6/C2–4、eligible 7–8、total 20–22。Non-Code 预期保持至少 17/24。

Falsifier：预检无法满足上述 token-label 不变量；或四个完整 checkpoint 全部 eligibility < 7/8；或 eligibility 恢复但所有 checkpoint 仍 Code < 2 / strict gate 不通过。预检失败属于实现失败且使用 0 GPU；模型 falsifier 成立则停止 v0002，不运行 LR bracket，下一版才单独评估 data quality / coverage。

预算：materialize/audit 为 0 GPUh；Probe Stage-A 预提交上限 0.20 GPUh、最多 5 次 probe32 eval（含 Base）。只有 Probe strict pass 才允许进入 Main；整版 primary + confirmation 的硬上限为 2.5 GPUh。
