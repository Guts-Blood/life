# Training Learning

用于把大模型训练框架源码、学习计划和实验笔记放在同一个 `life` repo/context 中，并配合 [30-Day LLM Post-Training Bootcamp](../post-training-30-day-bootcamp/README.md) 做代码阅读和实验。

## ms-swift

- 本地路径：`../../vendor/ms-swift/`
- Upstream：<https://github.com/modelscope/ms-swift.git>
- Pinned upstream commit：`565a1ad586a21d24b23931c52d2c62b49c39bee8`
- 集成方式：被父级 `life` repo 忽略的独立 clone；不是 submodule。
- 学习注释：`patches/ms-swift-565a1ad-day05-comments.patch`

最初的源码快照于 `2026-07-23` 导入 life repo（`5ccb102`），在 `94f904c` 时仍与 upstream 一致；`1dc0e4e` 只新增了 5 个文件、54 行教学注释。当前版本不再跟踪整份 upstream 源码，而是保留学习文档、revision 和可重放 patch。实验 run manifest 仍应记录 upstream revision；升级前先保存对应实验的快照。

## 建立与检查本地源码

```bash
git init vendor/ms-swift
git -C vendor/ms-swift remote add origin https://github.com/modelscope/ms-swift.git
git -C vendor/ms-swift fetch --depth 1 origin 565a1ad586a21d24b23931c52d2c62b49c39bee8
git -C vendor/ms-swift checkout --detach FETCH_HEAD
git -C vendor/ms-swift apply --check ../../learning/training-learning/patches/ms-swift-565a1ad-day05-comments.patch
```

默认保持外部 clone 干净。需要复盘源码内联注释时，再显式执行不带 `--check` 的 `git apply`；完成后可在外部 clone 中丢弃这些本地注释。
