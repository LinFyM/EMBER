# Writer output-space diagnostics

本目录是2026-09-21冻结诊断的轻量、可复算交付包。主结论见[报告](report.md)。

- `paired_summary.csv`、`per_task_success.csv`、`paired_rows.csv`：O1200与三臂的配对结果；
- `rollout_rows.csv`：64条参照/干预闭环的脱敏行及阶段谓词；
- `a1_geometry_summary.csv`、`a2_geometry_summary.csv`：code与投影几何汇总；
- `raw/`：A0/A1/A2轻量原始CSV、completion及A1工程事故记录；
- `*.png`、`*.svg`：报告图；`analyze.py`可从正式study重新生成全部表和图；
- `completion.json`：执行范围和版本。

正式study保留两份约3.8 MiB的A1 code safetensors、48条compact轨迹、launch合同和完整日志；这些本机资产
没有复制到Git。运行`analyze.py`需显式传入该study根：

```bash
MPLCONFIGDIR=/tmp/ember-mpl python analyze.py --study-root /path/to/study
```

初版A1把词典序hook调用误当数值layer顺序。旧结果已在study内封存；`raw/a1_layer_order_incident.json`
记录影响边界。当前A1表来自修复提交`36e7f791`，重建误差已回到BF16量级。
