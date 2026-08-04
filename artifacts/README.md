# Artifacts

这里存放需要长期版本化的最终交付物，以及缺少可重复生成链路的二进制文件。

## 目录

- `presentations/`：最终演示文稿。
- `reports/`：研究源文、生成脚本与最终报告。
- `resumes/`：内容独立的 DOCX 与最终 PDF 简历版本。

## 版本策略

可以提交：

- 最终 PPTX、PDF、DOCX；
- 生成最终产物所需的 Markdown、脚本和小型配置；
- 无法从仓库内容可靠重建的二进制文件。

不要提交：

- 逐页 PNG、contact sheet 与视觉 QA 截图；
- `__pycache__`、文本提取结果和下载缓存；
- 可重新下载的外部论文或参考资料。

这些中间文件统一写入已忽略的 `tmp/`，完成验收后删除。

## 重建后训练研究报告

```bash
python3 artifacts/reports/post-training-roadmap/render_pdf.py
```

脚本从自身目录读取 `post-training-roadmap.md`，并在同一目录生成 `post-training-roadmap.pdf`。
