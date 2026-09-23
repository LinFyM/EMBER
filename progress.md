# EMBER progress

## 当前授权：仓库收尾交接，研究暂停（2026-09-23）

Owner最新要求只整理仓库，不再开展或派发实验。Luna由Owner停止；**没有创建Sol任务**。
没有active experiment design。旧计划、设计中的启动许可、GPU特例与时间窗口均不自动生效。
本任务只整合长期理解、清理文档/代码/可删除资产并封存交接；最终由Owner另外安排研究接手。

## 接手先读

1. [Owner要求](docs/current_owner_requirements.md)：固定LoRA可达下界并非公共底座课程；有益视频利用、理论解释深度与协作边界。
2. [Concept](docs/concept.md)：完整pipeline，以及生成Jacobian、端点/前缀监督、任务关系覆盖三类待检验解释。
3. [Findings](findings.md)§127–133与[研究历史](docs/research_history.md)：近期原件、正负证据和不能外推的结论。
4. [暂停的条件编译设计](docs/designs/conditional_compilation_diagnostics_design.md)：四臂用途、混杂与未覆盖的数据原因。

## 当前实证快照

| 方法／固定checkpoint | coverage Validation correct/other/wrong | coverage Test |
| --- | --- | --- |
| Source1000 | 51／不适用／不适用 | 75 |
| MT-BC300，rank128 | 155／不适用／不适用 | 121 |
| 原跨episode辅助Writer C600 | 154/160/153 | 未跑 |
| 夜间12任务条件fresh，selected600 | 142/132/105 | 未跑 |

所有表中分数均为每臂400条；不同task集合不能跨Val/Test逐行配对。baseline Test是Owner授权的历史暴露任务补测，
不能描述为新盲测，不反哺选点/划分/方法。Source Test更高、MT-BC更低不证明划分无效，也不支持“Test统一更难”。
夜间1200步fresh完整曲线103/117/142/141/138/107；训练侧小试收益没有转为超过155的正式泛化结果。
已完成correct/other/wrong；shuffle/reverse在完成前被停止，没有完整分数；当前EMBER Test、FT、RL均未启动。
目前没有验证出同时解决绝对性能与有益视频特异性的方案，亦没有证据证明架构理论上已到上限。

原件根：
- `/data0/user/ymdai/ember_runs/overnight_root_cause_20260922`：近期有界诊断和小试。
- `/data0/user/ymdai/ember_runs/coverage_task_mixing_20260923`：夜间fresh、完整checkpoint、配对原件与selection。
- `/data0/user/ymdai/ember_runs/coverage_baseline_test_20260923`：Source/MT-BC Test原件，交付commit`7b18030c`。
- [夜间小型结果包](docs/review_materials/20260923/overnight_results/README.md)：当时PPT使用的指标与图，不包含后续baseline Test。

## 未完成代码的唯一交接位置

已push分支 **`codex/conditional-compilation-diagnostics`，commit `73267f53`** 保存Luna全部未完成实现。
没有合入main；主线只保留[设计](docs/designs/conditional_compilation_diagnostics_design.md)、
`configs/conditional_compilation_diagnostics_v1/experiment_spec.json`和`partition_audit.json`。
登记根为`/data0/user/ymdai/ember_runs/conditional_compilation_diagnostics_20260923/registration.json`，状态是暂停。

候选fit28＝官方Train中的seen target16＋已审计aux12；diagnostic-held8是其余官方Train任务，官方24/8/8未改。
A直接共享rank16、B语言Writer、C完整视频Writer、D相同视频结构但第二监督组改tau1/前5步；均计划fresh。
这只能定位整体环节，不能唯一分离参数化/活动容量，也没有数据构成干预。旧MT-BC看过这组held8，不能充当A的未见任务结果。
完整分组、步数、节点与每个比较的边界以冻结spec和设计为准；本批**未做GPU smoke、未启动四臂训练、没有新成绩**。

已知未修复错误：WIP `supervised.py`中`validate`错误缩进在`configured_endpoint`的return之后，未成为
`SupervisedEngine.validate`；只做syntax compile不能发现该问题。后继若获新授权，应先审查整个WIP diff及真实调用接口，
不能把已完成的CPU配置/事件审计误称训练实现已通过。不要为清理而删除这条未合并分支。

恢复WIP需从该已推送分支新建隔离worktree，并把本次main清理逐项协调后验证；不在main直接覆盖未验证代码。
本段只提供恢复位置，不构成运行指令；由Owner与新session决定继续、修正或放弃候选。

## 本次收尾记录

- 长期要求和concept已整合；docs按designs/analyses/review_materials分工，历史材料有明确状态。
- task_plan和progress只保留当前状态，重复历史移至既有研究历史/发现入口；原详细过程可从Git恢复。
- 已结束的stability/output-space/causal专用诊断和low-LR执行面退役；历史实现以`7b18030c`及run记录的commit恢复。
- 代码验证、资产回收数字及最终Git/worktree状态将在本次收尾完成时登记于本节。

没有后继实验派发；不要根据文档清理恢复任何运行。
