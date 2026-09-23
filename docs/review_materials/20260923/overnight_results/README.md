# 2026-09-23 夜间实验的汇报材料

本目录是夜间12任务条件fresh及前置小试的轻量指标/图表导出，用于当时组会PPT。
它不是当前实验状态，也不包含稍后完成的Source/MT-BC coverage Test；最新快照看[progress](../../../../progress.md)，
推理与比较边界看[findings](../../../../findings.md)§130–133。

- `metrics/`：导出的曲线、配对比较及逐任务统计；保留各自原始比较口径。
- `figures/`：基于上述指标生成的汇报图。
- `manifest.json`：导出文件清单与来源，不是新的科学结果或执行许可。

唯一selected600的完整Validation correct/other/wrong为142/132/105（各400），未超过MT-BC300的155；
训练侧小试正证据不代表fresh泛化成功。shuffle/reverse未完成，当前EMBER Test、FT、RL未做。
正式原件与完整checkpoint在`/data0/user/ymdai/ember_runs/coverage_task_mixing_20260923`，
小试在`/data0/user/ymdai/ember_runs/overnight_root_cause_20260922`及其中记录的pilot根。
图表用于展示，任何统计复算应回到对应run的原始配对行和冻结合同。
