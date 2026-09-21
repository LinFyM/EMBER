# 旧/新 Writer 差异审计数据包

本目录把已完成的正式实验原件整理为远端可直接读取的标量表和成功标记。它支持分析旧 Writer 在原24任务上为何表现较稳、新 Writer 在36任务上为何波动；**不提供唯一根因的既成结论，也不授权续训或调整配方**。新 Test 尚未执行，不能把旧 Test 与新 Validation 合并。

## 数据来源和范围

- 旧单相机同视频 Writer：`video_teaching_20260919` 的主组，correct Validation 节点900、1200、1500、1800、2100，各400行。训练指标与任务曝光取其 `continuation/main/training/{metrics,exposures}.jsonl`，包含完整1–2100步；900–1500步原始训练目录另有同一段记录。旧1200的other和旧1500的other/wrong/shuffled/reversed也纳入精简成功表。
- 新覆盖重训 Writer：`coverage_retraining_20260920` 的200–1800共9个correct Validation400节点，以及冻结1000节点的四个视频对照。训练指标与曝光取 `training/writer/{metrics,exposures}.jsonl`。
- 新 MT-BC：50–500共10个correct Validation400节点与 `training/mtbc/metrics.jsonl`；新Source correct400一并纳入成功表。MT-BC每步覆盖全部36任务，其表没有伪造Writer式的单任务曝光记录。
- 所有成功表从正式 `results.json` 导出，逐面板核对400行、8任务各50行、逐任务成功数与overall成功数。评测Git提交、物化Git提交和合同版本列于 `validation_panel_provenance.csv`。未复制原始run contract中的主机资源快照、视频帧、动作数组或checkpoint。

原件在两份study根目录中保存；Git包使用相对study目录的产物约定。复算时运行：

```bash
python diagnostics/export_diagnostics.py \
  --old-study /path/to/video_teaching_20260919 \
  --new-study /path/to/coverage_retraining_20260920 \
  --output diagnostics
```

上述命令从本报告父目录运行。远端读者只分析现有CSV时，无须访问study目录。

## 文件索引

| 文件 | 内容 |
| --- | --- |
| `validation_per_task_nodes.csv` | 旧Writer5、新Writer9、新MT-BC10及新Source1节点的逐任务成功数，共200行；可画完整任务曲线。 |
| `validation_success_rows.csv` | 34个正式Validation面板共13,600行精简结果；含task、初态、成功标记、环境/策略种子、视频ordinal与teacher demo，可复算成功集合和配对差值。 |
| `validation_task_transitions.csv` | 相邻correct节点逐任务保留/获得/丢失及视频映射一致性。 |
| `shared_held_old1200_new1800.csv` | 旧1200与新1800的五个共同held任务；完整列出同初态、RNG、视频映射检查及配对变化。 |
| `writer_training_steps.csv` | 两轮Writer的每步实际应用LR（第1步留空）、下一步LR、全局loss、Writer/Meta梯度范数和耗时。 |
| `writer_training_task_exposures.csv` | 每次更新4个实际任务的标量记录，共15,600行；含`occurrence`、FM/教学loss及经权重后的LoRA梯度范数。 |
| `writer_training_group_step_windows.csv` / `writer_training_group_occurrence_windows.csv` | 按全局step的200步窗及每任务25次访问窗，对共同20任务、各自独有4任务和新辅助12任务分别汇总均值、中位数、95分位。 |
| `writer_training_task_groups.csv` / `writer_training_config_summary.csv` | 任务分组、两轮真实训练配置及任务权重解释入口。 |
| `writer_validation_node_context.csv` | 各节点实际LR、轮转位置、前6或9步任务顺序与组成、训练段起点、frame chunk、world size及训练/评测Git提交。`recent_window`指最近若干更新，并不保证它恰好是一个完整打乱周期。 |
| `mtbc_training_steps.csv` | 新MT-BC每步LR、loss、梯度、查询数与耗时。 |
| `validation_panel_provenance.csv` | 34个面板的行数、总成功数、评测Git提交、物化Git提交及合同标识。 |

## 已可直接核查的两个现象

1. 新Writer1000→1200的正确视频总分117→80。Long/1从27→6，Object/1从43→35，Goal/6从39→33，Spatial/3从8→5，Spatial/6从0→1。随后1200→1400→1600总分80→97→111，但不同任务交替恢复和下降；不能把总分波动直接归因于某一个全局工程错误。
2. 旧1200与新1800按每任务访问次数均约200次，`lr_next`均约`1.627965e-4`。在五个共同held任务且相同state/RNG/video映射上，旧144/250、新92/250；配对保留84、获得8、丢失60。Spatial/3单项36→6，贡献净差-30/50。两轮完整8任务Validation的174与92不是同一任务集合，不能直接做400行配对比较。

`teaching_lora_gradient_norm`在训练原件中已经包含教学损失权重，汇总时**没有再次乘1/3**。梯度范数只测强度，不测方向，不能由此证明任务梯度相互抵消。旧/新训练同时改变任务集合、任务权重、轮转周期、LR时间尺度和物理frame chunk；共同任务的结果差异不单独识别其中哪一项是根因。`lr_next`是该更新之后的scheduler值；`lr_applied`取上一更新的`lr_next`，第1步因没有前项而留空。

进一步结论应先从这些表中对齐具体任务的loss、梯度和成功集合，并结合原始训练/评测合同核查。旧Test正式负结果仍见[旧报告](../../../20260920/test_capacity/report.md)；新训练与冻结选择见[阶段报告](../report.md)。
