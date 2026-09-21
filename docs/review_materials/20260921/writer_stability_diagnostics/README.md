# Writer stability diagnostics：远程结果包

这是2026-09-21诊断的已完成部分，仅包含E0/E1。先读[报告](report.md)，与专家讨论时可直接使用
[问题清单](expert_handoff.md)。

主要文件：

- `e0_integrity.json`：完整恢复及主/教学梯度路径；
- `frozen_probe_rows.csv`、`probe_manifest.jsonl`：1728条E1-A原始探针；
- `frozen_rollout_rows.csv`：280条E1-B闭环原始行；
- `e1_closed_loop_paired.csv`：按固定condition对齐的模型比较；
- `trajectory_manifest.csv`：280个本地轨迹的脱敏索引，不含轨迹张量；
- `e1_probe_summary.json`、`e1_rollout_summary.csv`：汇总统计；
- `e1_overview.*`、`e1_closed_loop_by_task.*`：PNG/SVG图；
- `plot_e1_results.py`：绘图源码；
- `export_completed_results.py`：从canonical study重新生成脱敏结果包。

`completion_status.json`是范围裁决：E2/E3未完成，任何部分E2文件都不在本包内。路径中的`<repo>`、
`<coverage-study>`、`<old-video-study>`和`<writer-stability-study>`是脱敏后的canonical根标签。
