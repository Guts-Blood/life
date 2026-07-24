# AutoDL 与 GPU 资源计划

更新日期：2026-07-22。卡价和具体库存以 AutoDL 页面实时显示为准；本计划不写死单价，只规划卡型、数量和使用时长。

## 原则

1. 理论阅读、数据清洗、写 scorer、报告和 capacity planning 不开 GPU。
2. 在 CPU/无卡环境完成命令检查、数据抽样和 5-sample dry run 后才开 H100。
3. 先用 0.6B/1.7B 做 10–50 step smoke，通过后再扩大 steps/data。
4. 所有长任务先确认日志落盘、checkpoint 可恢复，再离开终端。
5. 训练结束立即保存关键产物并关机，不用 GPU 实例看文档。

AutoDL 按量实例以实例开关机时间计费，并非以 GPU kernel 是否运行计费；关机结束实例费，但 GPU 不再预留。官方同时提供无卡模式、升降配置和训练结束后自动关机建议。[AutoDL 计费](https://www.autodl.com/docs/price/)；[省钱说明](https://www.autodl.com/docs/save_money/)

## 推荐卡型

优先级是为了减少环境与显存变量，不代表性价比排名。

| 场景 | 首选 | 可替代 | 原因 |
|---|---|---|---|
| 0.6B/1.7B debug、eval | 1×H100 80GB | A100 80GB、L40S 48GB | 环境统一，避免频繁换 CUDA/显存条件 |
| 1.7B full SFT | 1×H100 80GB | A100 80GB | 给 optimizer、activation 和临时 buffer 留余量 |
| 4B/8B LoRA | 1×H100 80GB | A100 40/80GB、L40S 48GB | 单卡足够学习流程；不需要为了 LoRA 租多卡 |
| Megatron SFT/Profiler | 2×H100 同机 | 2×A100 80GB 同机 | 运行真实 TP/DP/process groups、distributed checkpoint 和 NCCL |
| slime RL 主闭环 | 官方拓扑 8×H100 同机 | 4×H100 缩容后运行 | 8 卡增加费用但减少改官方 recipe 的调试时间，能看到 train/rollout 角色编排 |
| 30B+ full post-training | 本月不实际租 | 只做容量设计 | 一次昂贵大 run 不能替代小规模闭环与多卡诊断能力 |

选择多卡实例时确认 GPU 在同一主机、互联拓扑、CPU 内存、数据盘和驱动版本。卡名相同不代表通信拓扑相同。

## 分阶段租卡预算（GPU hours）

| Block | Days | 配置 | 预计 wall time | 目的 | 停机条件 |
|---|---|---|---:|---|---|
| A | 01（可选） | 1×H100 | 0.5–1h | CUDA、模型加载与 inference 验证 | 环境报告和 5-sample smoke 完成 |
| B | 08–12 | 1×H100 | 20–35h | LoRA/full SFT、Base baseline、主训练 | 逐样本 baseline、checkpoint、训练日志完整 |
| C1 | 15–16 | 1×H100 | 9–13h | packing、batch/显存 ablation | 对照表完成并选定 Megatron 输入配置 |
| C2 | 17 | CPU/no-card | 0h | Megatron 主链路通读、容器/数据/命令准备 | 调用链与 2 卡 runbook 完成 |
| C3 | 18–19 | 2×H100 同机 | 8–12h | upstream smoke、Qwen SFT、checkpoint、Profiler | 实际 runtime trace 对应源码，修改观测点后重跑 |
| D | 23 | 1×H100 | 4–7h | 公共 benchmark 与自定义 Eval 生成 | 所有逐样本 predictions 落盘 |
| E1 | 26 | CPU/no-card；可选 1×H100 conversion | 0–2h GPU | slime 通读、Docker/数据/HF↔Megatron 权重准备 | Day 29 不再下载、编译或临时猜参数 |
| E2 | 29 | 8×H100 同机；预算受限改 4× | 4–8h | slime rollout→reward→train→weight sync，修改 reward 后重跑 | 两轮闭环、runtime evidence、bad cases 完整 |
| F | 30 | 1×H100 | 2–4h | clean-room 最小闭环 | 从新环境跑到预测与报告成功 |

粗略总量：约 37–64 个单卡 H100 wall-hours，另加 2 卡 Megatron 实例 8–12 小时（16–24 GPU-hours）和 8 卡 slime 实例 4–8 小时（32–64 GPU-hours）。slime 若缩到 4 卡可降低 GPU-hours，但需要预留更多配置与显存调试时间。Day 06、07、13、14、20、21、27、28 是纯阅读周末，明确不租 GPU。先按 Block 租，不要一开始包满整月。

## 每天开机前 Checklist

- [ ] 今天必须使用 GPU 的步骤已经明确。
- [ ] 命令在 CPU/小数据上完成语法和路径检查。
- [ ] model/dataset 已下载或确认缓存命中。
- [ ] 输出路径、日志路径、save_steps 正确。
- [ ] 有 `max_steps` 或明确停止条件，不会无限跑。
- [ ] `nvidia-smi topo -m`、磁盘空间、CPU 内存已记录。
- [ ] SSH 断开后任务仍会运行，日志不只留在终端标准输出。
- [ ] 训练结束后的备份和关机方式已准备。

## 数据与镜像

AutoDL 官方说明，本地系统盘/数据盘通常性能较好但没有冗余保证；重要代码、配置、日志和最终 adapter/checkpoint 应同步到可靠存储。实例连续关机达到平台释放周期后，本地数据可能被清空。[实例数据规则](https://www.autodl.com/docs/instance_data/)

建议：

- 环境稳定后保存一次名为 `posttrain-qwen-base` 的系统镜像；AutoDL 支持保存系统盘镜像并在新实例恢复。[镜像文档](https://www.autodl.com/docs/image/)
- 训练数据在本地数据盘运行，避免网络文件存储 IO 成为瓶颈。
- 代码、配置、精简日志、最终报告同步 Git。
- 关键 adapter/checkpoint、环境 lock、dataset manifest 同步文件存储或个人可靠存储。
- 文件存储适合共享与备份，但官方提示其 IO 一般，训练数据应复制到本地盘再跑。[文件存储文档](https://www.autodl.com/docs/fs/)

## 关机策略

按量模式下：

1. 训练完成。
2. 确认日志 flush、checkpoint 完整、Eval 结果已保存。
3. 同步小型关键产物。
4. 记录结束时间和 run status。
5. 关机。

不要直接把自动关机接在未经验证的训练命令后。先验证脚本退出码、日志与备份；之后才可按 AutoDL 官方方式在成功后调用 `/usr/bin/shutdown`。关机后 GPU 不预留，下一次可能需要等待或换实例，因此固定镜像和外部备份比依赖同一台机器更可靠。

## 何时升级到多卡

只有同时满足以下条件才租 Megatron 2×H100：

- [ ] 单卡训练已稳定运行至少 50 steps。
- [ ] checkpoint save/load 已验证。
- [ ] 单卡 baseline 的 tokens/s、step time、peak memory 已保存。
- [ ] 多卡实验只改变 parallel strategy/world size，其他配置可对齐。
- [ ] 明确今天要观察的 collective、sharding 或 scaling efficiency。

slime 的 8 卡是一次短而明确的官方-recipe 验证，不是扩大战果。开机前必须完成 Docker、模型、数据、Megatron checkpoint 转换、启动参数 dry review 和停止条件；否则不开 8 卡实例。
