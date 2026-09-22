# Writer 辅助 episode 配对 fresh 对照：专家复核包

这是对正式 study `coverage_retraining_cross_episode_aux_20260922` 的轻量、可逐行复核摘录。它只复制结构化指标与元数据，不复制 checkpoint、视频、动作标签、原始 rollout 日志或大资产；所有数值都来自已经 sealed 的正式产物。需要直接阅读结果时先看[轻量正式复核报告](report.md)。

本包不启动新训练、物化或评测，也不把辅助 episode 配对的对照外推为端点前缀辅助目标本身的价值。

## 给复核者的入口

| 需要回答的问题 | 文件 |
| --- | --- |
| 六个 correct400 节点的逐条件结果 | `correct_nodes_rows.csv` |
| C600 correct 与 same-task-other 的逐条件配对 | `c600_correct_vs_other_rows.csv` |
| C600 correct 与 cross-suite-wrong 的逐条件配对 | `c600_correct_vs_wrong_rows.csv` |
| C600 与 historical same-video、MT-BC300、Source 的逐条件配对 | `c600_vs_same_video_rows.csv`、`c600_vs_mtbc300_rows.csv`、`c600_vs_source_rows.csv` |
| 所有配对的总计、suite、task R/G/L 与完整性 | `c600_pairing_summary.json` |
| 每个面板的 400 行、manifest、worker completion 与 return-code 核验 | `panel_integrity.json` |
| 1200 次更新的主/辅助损失、LR、梯度与耗时 | `training_metrics.csv` |
| 36 个任务的 exposure、主/辅助查询、loss、梯度和耗时汇总 | `task_exposure_summary.csv` |
| 实际用于正式运行的 effective config（从 sealed run contract 摘录） | `effective_training_config.json` |
| 正式合同、停止/选点记录、C600 完整 resume 文件及原始资产路径 | `cost_and_assets.json` |

CSV 中的 `language_global_task_id` 是执行环境的目标任务；`used_video_*` 是实际送入 Writer 的视频来源。尤其 `cross_suite_wrong` 中二者故意不同。`teacher_demo_indices` 仅是视频 demo 编号，不含 action 标签。

## 复核结论的范围

- C600 correct 为 154/400，MT-BC300 为 155/400。`c600_vs_mtbc300_rows.csv` 给出同一 task/state 条件的成功配对，便于判断两者是否共享能力，而不只比较总分。
- C600 correct/wrong 为 154/153；完整性字段明确区分共同 task/state、环境 seed、policy seed root，以及由于不同终止步数而可能不同的 policy-noise 后缀。所有比较的共同 noise 前缀均通过核验。
- six correct 节点与 C600 controls 都保留每行 task、state、视频 demo、成功、步数、seed 与面板标识；面板级文件记录 400 行完整性、materialization manifest 和所有 worker 零退出。
- 物化总条件数已记录，但正式 manifest 没有记录 materialization elapsed；本包明确标为 `not_recorded`，不作估计。
- 该包不添加置信区间或新诊断。若需统计推断，应以本包逐行表并按既有合同的 task cluster 为单位计算。

## 可再生成性与边界

这些文件由 sealed JSON/JSONL 机械导出；`effective_training_config.json` 是运行时实际配置的摘录，源文件的精确本地路径、输出合同和 C600 checkpoint 文件清单均在 `cost_and_assets.json`。原始正式报告仍在 study 根的 `analysis/report/`，本包是可推送的复核副本而不是新的 canonical result root。

“C600 完整 resume state 存在”只表示 `checkpoint_manifest.json` 声明的 Writer、optimizer/trainer 和四个 rank state 文件均仍存在；任何新分支是否可恢复、以何种拓扑恢复，仍须先登记新的正式合同。
