# AutoDL 与 GPU 资源计划

更新日期：2026-07-27。卡价、库存和实例拓扑以开机时 AutoDL 页面为准；本计划给出资源上限和 readiness gate，不写死单价。

## 原则

1. 数据 schema、清洗、mixture、manifest、静态 codepath、quiz 和报告均不开 GPU。
2. 开机前必须有 pinned code/model/data、可执行命令、`max_steps`、成功条件、停止条件和输出路径。
3. 先用 tiny overfit/5–20 step smoke 验证数据与 loss，再进行受控 SFT、DPO 或 GRPO。
4. 每个实验只改变一个预注册变量；不为“卡还开着”临时扩大训练。
5. checkpoint 必须先通过 save/load；长任务再验证中断恢复。
6. 8×H100 slime 是 Stretch，不是毕业门槛。Core 先完成单卡 ms-swift 与不超过 4×H100 的 slime 最小闭环。

AutoDL 按实例开关机时间计费，不按 GPU kernel 活跃时间计费。关机后 GPU 不再预留，实例数据也有保留边界；使用前复核官方[计费规则](https://www.autodl.com/docs/price/)、[省钱说明](https://www.autodl.com/docs/save_money/)与[实例数据规则](https://www.autodl.com/docs/instance_data/)。

## 推荐资源

| 场景 | 默认资源 | 说明 |
|---|---|---|
| 数据审计、manifest、quiz、配置 review | CPU/无卡模式 | 不能用 GPU 掩盖数据准备不足 |
| 0.6B tiny overfit、DPO smoke、eval | 1×H100 80GB | A100 80GB 可替代，但同一对照组不要换硬件 |
| 1.7B 受控全参 SFT | 1×H100 80GB | 先用 accounting 与短 smoke 确认 optimizer/activation 余量 |
| Packing、优化、resume 对照 | 1×H100 80GB | 保持卡型与 baseline 相同 |
| Megatron minimum codepath | 2×H100 同机 | 只验证 process group、collective、state ownership 与 checkpoint |
| ms-swift 小模型 GRPO | 1×H100 起步；需要 rollout 并发时 2× | 先验证 reward、mask、group 与 policy version |
| slime 在线 RL Core | Day 26 验证的最小支持拓扑，预算上限 4×H100 同机 | 实跑最小 rollout→reward→train→weight-sync；不追求吞吐纪录 |
| slime 官方 8×H100 recipe | 8×H100 同机 | Stretch；Core 全通过且单独批准预算后才开 |
| 30B+ full post-training | 本月不租 | 只保留配置判断题，不作为最终项目 |

选择多卡实例时记录 `nvidia-smi topo -m`、GPU 型号/显存、CPU 内存、数据盘、driver/CUDA/NCCL、同机与否。卡名相同不代表互联相同。

## 分阶段租卡上限

下表是预算窗口，不是必须耗尽的配额。wall time 到上限仍未得到关键证据时应关机复盘。

| Block | Days | 配置 | 建议 wall time | 目标 | 关机 Gate |
|---|---|---|---:|---|---|
| A | 08–10 | CPU；eval 时 1×H100 | 2–4h GPU | 数据契约、manifest、Base 逐样本 baseline | sample audit 与 frozen eval 已落盘 |
| B | 11–12 | 1×H100 | 8–14h | tiny overfit、受控 SFT、checkpoint 对比 | loss/mask/生成闭环与 early/mid/final 证据完整 |
| C | 15–17 | 1×H100 | 8–12h | packing/优化单变量对照、中断恢复 | baseline 可比，resume continuity 通过 |
| D | 18 | 2×H100 同机 | 2–4h | Megatron 最小多卡 codepath | rank/group/collective/checkpoint 有 runtime evidence |
| E | 19 | 1×H100；必要时复用 2 卡 | 3–5h | failure injection 与 profiler | 高 LR、mask、distribution、throughput 四类 failure 可识别、可恢复 |
| F | 22–23 | CPU 准备；1×H100 | 3–6h GPU | preference audit 与 DPO smoke | pair/logprob/mask 检查和 frozen eval 完整 |
| G | 24–25 | 1×H100，最多 2× | 4–8h | ms-swift 小模型 GRPO | reward/group/rollout/update 守恒，逐样本结果落盘 |
| H | 26–28 | CPU/无卡模式 | 0h | slime pinned environment、conversion、runbook、代码路径 | Day 29 无下载、编译、猜参数或未审数据 |
| I | 29 | Day 26 确认的最小同机拓扑，≤4×H100 | 4–8h | slime rollout-only、train-only replay、full loop 与 reward 修改 | replay、policy version、weight sync、reward evidence 完整 |
| J | 30 | 1×H100 | 2–3h | clean reproduction | 新环境完成最小数据→训练→held-out→报告 |

8×H100 Stretch 另设 2–4 小时硬上限，只允许在 Block I 已成功、官方 recipe 与当前 pinned commit 完全对齐、数据/模型缓存命中且用户再次确认预算后执行。

## 开机前 Checklist

- [ ] 今日问题只能通过 GPU 回答，CPU 准备已结束。
- [ ] model/tokenizer/dataset/code/container 都有固定 revision 或 digest。
- [ ] 抽样检查了 raw、rendered、tokens、labels/mask；RL 还检查 reward 与 metadata。
- [ ] 命令已做 help/config/path dry check，输出目录不覆盖既有 run。
- [ ] 写明唯一自变量、baseline、`max_steps`、timeout、成功和停止条件。
- [ ] 模型、数据与镜像已缓存或估算下载时间；多卡前不临时编译大依赖。
- [ ] `nvidia-smi topo -m`、磁盘、CPU 内存、driver/CUDA/NCCL 已记录。
- [ ] 日志、逐样本输出、checkpoint 和 profiler 路径均可持久化。
- [ ] SSH 断开不影响任务；训练异常不会无限重启。
- [ ] 备份和关机步骤已准备。

## 各阶段 Readiness Gate

### 受控 SFT

- [ ] 一条样本能追到 label token，tiny dataset 能 overfit。
- [ ] Base frozen eval 已完成。
- [ ] accounting 显示目标配置有显存余量。
- [ ] save/load smoke 已通过。

### DPO

- [ ] SFT 起点 checkpoint 已冻结。
- [ ] prompt/chosen/rejected 使用同一 template，顺序与 mask 已抽查。
- [ ] preference margin、长度/source 分布和重复率已审计。
- [ ] 短 smoke 的 chosen/rejected logprob 对齐已验证。

### ms-swift GRPO

- [ ] reward 在手工构造的正/负样本上结果正确。
- [ ] prompt/group/completion/reward/advantage 的数量与 grain 明确。
- [ ] rollout response mask、EOS/truncation 和 policy version 可追踪。
- [ ] zero-variance 与过长生成有停止条件。

### slime 多卡

- [ ] ms-swift GRPO 已跑通，算法与数据 bug 不再借多卡定位。
- [ ] slime commit、官方容器、SGLang/Megatron 版本与启动脚本固定。
- [ ] model conversion/load、prompt data 和 custom reward 在无卡/单卡层面已验证。
- [ ] rollout 与 train 的 batch invariant 已手算并由配置检查通过。
- [ ] 2–4 卡拓扑、角色放置、显存预算和 weight-sync 路径已写入 runbook。
- [ ] 8 卡参数未混入 Core 配置。

## 数据、镜像与关机

- 环境稳定后保存一次基础镜像；参考[镜像文档](https://www.autodl.com/docs/image/)。
- 训练数据先复制到本地数据盘。共享[文件存储](https://www.autodl.com/docs/fs/)用于备份，不默认作为训练热路径。
- Git 保存代码、配置、manifest、精简日志与报告；大 checkpoint 保存到可靠对象/文件存储并记录 checksum。
- 关机前依次确认：进程退出、日志 flush、checkpoint 可读、逐样本结果存在、关键产物已同步、run status 已更新。
- 未验证退出码与备份前，不把自动关机直接拼到长训练命令后。
