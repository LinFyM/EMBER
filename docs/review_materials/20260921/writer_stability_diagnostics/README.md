# Writer stability diagnostics：远程结果包

这是2026-09-21诊断的远程材料。E0/E1原始诊断保持不变；专家修订后取消原E2，并完成四臂36步终点
诊断。先读原[E0/E1报告](report.md)和[问题清单](expert_handoff.md)，再查看
[修订终点原始包](corrected/README.md)。

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

`completion_status.json`只记录第一次上传时E0/E1的范围；它不代表专家修订后的最终状态。原E2部分文件仍不在
远程包内，也不作为结果。`corrected/completion.json`是修订四臂的完成裁决。路径中的`<repo>`、
`<coverage-study>`、`<old-video-study>`、`<writer-stability-study>`和`<STUDY>`是脱敏后的canonical根标签。
