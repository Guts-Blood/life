# Training Learning

用于把大模型训练框架源码、学习计划和实验笔记放在同一个 `life` repo/context 中，并配合 [30-Day LLM Post-Training Bootcamp](../post-training-30-day-bootcamp/README.md) 做代码阅读和实验。

## ms-swift

- 本地路径：`ms-swift/`
- Upstream：<https://github.com/modelscope/ms-swift.git>
- Clone 日期：`2026-07-23`
- 初始分支：`main`
- 初始 commit：`565a1ad586a21d24b23931c52d2c62b49c39bee8`
- 集成方式：上游源码快照，作为父级 `life` repo 的普通目录管理。
- 当前只完成源码导入，尚未安装依赖或修改源码。

clone 完成后已移除嵌套 `.git`，因此 `ms-swift/` 不是 submodule 或独立 Git 仓库；源码、计划和后续笔记都由父级 `life` repo 统一提供 context。初始上游 commit 保留在本文件中，实验 run manifest 仍应记录这个 revision；升级上游前先保存已有实验对应的快照。

## 常用检查

```bash
git status --short -- learning/training-learning
rg --files learning/training-learning/ms-swift | wc -l
```
