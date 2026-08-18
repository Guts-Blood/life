# Day 26 — slime v0.3.1 × Qwen3.5-4B GPU compatibility final report

执行时间：`2026-08-17`  
运行 ID：`day26-slime-qwen35-20260817T141117Z`  
终态：`slime_qwen35_compatibility_blocked`  
失败层：`S0_container_import_config`  
Day 29：`no-go`

## 结论

Day 26 已按 fail-closed 合同完成，唯一合法终态为 `slime_qwen35_compatibility_blocked`。

固定的 Day 21 merged S1 本身完整，workspace 同步也通过逐文件 SHA-256 复核；但开出的 runtime 不满足冻结 S0 边界：物理可见两张 GPU，容器内无法证明目标 OCI digest，CUDA/NCCL 与冻结镜像漂移，且 Ray、SGLang、Megatron、`/root/slime` 和 `/root/Megatron-LM` 均缺失。S0 任一项失败即必须停止，因此没有进入 S1–S5，也没有通过安装依赖、clone 源码、换环境或换 topology 绕过失败。

## Gate 结果

| Gate | 结果 | 运行证据 |
|---|---|---|
| Pre-GPU freeze | pass | 冻结文件、prompt、promotion/schema/report SHA 全部匹配；本地 Day 26 reward `4/4`、Day 25 adapter `7/7`、Day 24 contract `22 run, 1 live-E2B skipped` |
| S0 container/import/config | **fail** | image digest 无法证明；物理 GPU `2`；Ray/SGLang/Megatron 缺失；exact checkouts 缺失；CUDA/NCCL 漂移 |
| S1 conversion/load | not run | `not_run_due_to_s0_fail`；仅 S1 input identity 预条件通过，不等于 conversion/load |
| S2 rollout-only | not run | `not_run_due_to_s0_fail` |
| S3 live reward/replay | not run | `not_run_due_to_s0_fail`；未读取 E2B credential |
| S4 train-only step | not run | `not_run_due_to_s0_fail`；无 optimizer update/checkpoint |
| S5 full sync/next version | not run | `not_run_due_to_s0_fail`；无 weight-version 边 |

## 冻结边界

| 项 | 期望 |
|---|---|
| slime | `v0.3.1@a6272da0d4f3d0a08520c99a2f3b4f6c887960dc` |
| image | `slimerl/slime:nightly-dev-20260804a` |
| OCI manifest digest | `sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7` |
| CUDA / NCCL | `12.9.1` / `2.27.3` |
| SGLang / FLA | `0.5.15.post1` / `0.4.2` |
| Megatron | `1dcf0dafa884ad52ffb243625717a3471643e087` |
| topology | 物理单卡、`97887 MiB`、colocate、TP=PP=CP=1 |
| parent | Day 21 promoted merged S1，禁止加载原 LoRA adapter |

## Runtime evidence：事实

### 输入与 parent 通过的预条件

- append-only remote root：`/root/autodl-tmp/runs/day26-slime-qwen35-20260817T141117Z`。
- local mirror：`tmp/day26-slime-qwen35-20260817T141117Z`。
- 同步 `62` 个 Day 20/22/24/25/26 依赖与 artifacts；source/remote manifest SHA 均为 `3ddba920ce3198490c959eb446566c26a7af471fe9749027b2222958885f0030`。
- Day 21 merged S1：`11` 个文件、`9,098,708,091` bytes；逐文件复算后的 inventory seal 为 `16bc212df51ecac9a5b3b34062e0ccc880c719d63248f04949be8636550e53be`，manifest content seal 为 `660eed4af7f76796631561275f0190c402952520a8ccce358289e6269fb8f8d3`，全部匹配。
- `CUDA_VISIBLE_DEVICES=0` 后，torch 的逻辑视图为一张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`、`101975851008` bytes。

这些只证明输入和逻辑遮罩，不证明 S0 或 S1 通过。

### S0 决定性失败

1. **物理 topology drift**：`nvidia-smi` 实际列出 `2 × RTX PRO 6000 Blackwell Server Edition`，每卡 `97887 MiB`。逻辑遮罩不能把物理双卡事实改写成冻结的物理单卡；Ray placement 又因 Ray 缺失而无法证明。
2. **image digest 不可证明**：`/.dockerenv` 和 overlay mount 只能证明处于 Docker；容器内没有 Docker CLI/socket、image marker 或目标 digest 命中，actual OCI digest 必须保持 `null/unknown`。合同明文规定 digest 无法证明即 S0 fail。
3. **runtime drift**：active runtime 为 CUDA toolkit `12.4`、torch `2.5.1+cu124`、NCCL `2.21.5`，而非冻结的 CUDA `12.9.1` / NCCL `2.27.3`。torch 还报告当前 build 不支持该 Blackwell GPU 的 `sm_120`。
4. **核心依赖缺失**：active Python 中无 Transformers、Ray、SGLang、FLA、FlashQLA、Megatron 和 E2B。唯一的 Day 25 venv 虽有 torch `2.10.0+cu128`、Transformers `5.12.1`、FLA `0.4.2` 与 E2B `2.37.0`，仍无 Ray、SGLang、Megatron 和 FlashQLA，且 NCCL 为 `2.27.5`。
5. **exact checkout 缺失**：`/root/slime` 与 `/root/Megatron-LM` 均不存在，因此无法验证 frozen SHA、运行 runtime static audit、source `qwen3.5-4B.sh` 或解析 `MODEL_ARGS`。

`s0/stage-status.tsv` 中某些采集项 exit `0` 只表示采集命令成功完成，不代表其语义 gate pass；真正的 gate 结论以 `decision/s0-decision.json` 为准。

## 推断

当前卡不是可用于验证固定 Day 26 合同的 runtime。现场安装依赖、clone upstream、切换到 Day 25 venv 或继续用未知 image，会把任务改成构造一个新 runtime，而不是验证冻结的 `slimerl/slime` digest；这超出授权边界。

即使不把物理双卡视为独立 blocker，image digest 不可证明、runtime/package drift、exact checkout 缺失和 Ray placement 失败也分别足以触发同一个 S0 终态。

## Unknown / 未声称

- 当前容器的真实 OCI digest 未知，未伪造为冻结 digest。
- 固定 slime image 在合规物理单卡上能否通过 S1–S5 仍未知；本次没有运行到这些层。
- 没有 conversion tensor inventory、SGLang/Megatron model load、rollout/train dump、live reward ledger、checkpoint 或 next-version evidence。
- 静态复核另发现 Day 24 advantage epsilon `1e-4` 与 slime 原生 `1e-6` 的潜在非零方差差异；由于 S0 已停止，本次没有产生 runtime group 来判断其后续影响，也没有修改 reward/post-process。

## Evidence index

下列 `tmp/` 路径标识已完成哈希回验的本地 runtime mirror，不纳入 Git；远端仓库公开保留本报告、runtime config、static audit 与 Day 29 no-go JSON。原始镜像缺席不改变报告中的 claim boundary。

- S0 machine decision：`tmp/day26-slime-qwen35-20260817T141117Z/decision/s0-decision.json`，SHA-256 `cb20804fa756d580e10fd6753981801659a2d25bd443b803768af450d429d8b3`
- Day 29 no-go：`artifacts/eval/day26-slime-qwen35-day29-go-no-go.json`
- 完整 local mirror：`tmp/day26-slime-qwen35-20260817T141117Z`
- 90-file remote manifest：`tmp/day26-slime-qwen35-20260817T141117Z/manifests/final-file-sha256.txt`，SHA-256 `d0f1787c88493b31ec17189f327faf81b31bbc17f03037de38c6ce88e58cfad9`
- 本地 mirror verification：`tmp/day26-slime-qwen35-20260817T141117Z/manifests/local-mirror-verify.log`，`90/90 OK`
- 核心日志：`s0/nvidia-smi-q.log`、`s0/gpu-inventory.log`、`s0/container-provenance.log`、`s0/cuda-nccl.log`、`s0/package-versions.log`、`s0/alternate-env-inventory.log`、`s0/source-checkouts.log`、`s0/ray-placement.log`、`s0/s1-export-verification.log`

## Day 29 go/no-go

`go_day29=false`。Day 29 的 slime run 未获授权，禁止换 release、rolling main、模型、image 或 topology 做 fallback。

若要重新验证，必须先开出能证明 exact OCI digest、物理单卡 `97887 MiB`，并已具备 frozen `/root/slime` 与 `/root/Megatron-LM` checkout 的 runtime；随后建立新的 append-only Day 26 run，从 S0 重新开始。
