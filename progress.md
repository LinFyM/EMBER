# EMBER progress

## 当前状态：两轮收尾完成，研究保持暂停（2026-09-23）

Owner认为仅清理缓存不充分，明确要求进一步裁剪历史checkpoint。本轮按关键权重/历史参照/当前依赖/恢复用途择点，
额外回收360.759 GiB；连同首轮53.007 GiB，两轮累计413.766 GiB。data1用户配额实测从846.3降至485.6/1024 GiB，
data0仍为122.1/1024 GiB。原始评测、配置、关键模型及当前诊断依赖继续保留。

Owner最新要求只整理仓库，不再开展或派发实验。Luna由Owner停止；**没有创建Sol任务**。
没有active experiment design。旧计划、设计中的启动许可、GPU特例与时间窗口均不自动生效。
长期理解、文档/代码/缓存与worktree整理、追加资产退休均已完成；后续研究由Owner另外安排。

## 追加checkpoint裁剪

- 9月8–14日33条结束路线：回收222.737 GiB，保留70个关键权重路径；8月3–7日12条结束/作废路线：
  回收115.220 GiB，保留24个代表权重及64份完整评测原件。合计174个历史checkpoint：94个`weights_only`、80个`metadata_only`。
- Source1000预先固定使用raw policy；退休无使用用途的optimizer及未选EMA副本，回收22.802 GiB。
  raw policy的canonical inspector在删除前后返回相同结果，正式推理仍可用；Source完整训练恢复和EMA不再可用。
- 合计删除256个大载荷路径、255个独立inode：trainer约287.989 GiB、Source optimizer 14.090 GiB、
  非关键Writer权重49.968 GiB、未用EMA 8.712 GiB。保留原合同/manifest、日志/指标、原始结果和小型rank RNG记录。
- 删除trainer的历史Writer不再exact resume；其原冻结CLI会因完整文件校验而拒绝加载，今后重放需要明确支持
  weights-only的读取路径，不能伪造trainer或重写历史manifest。已导出9月模型的标量/结构元数据；8月模型所需元数据在原合同/consumed中。
  9个受影响的物化缓存退役记录已同步上游可用性。当前checkpoint retirement记录优先于旧日志的“完整保留/可直接重建”描述。
- 94个保留权重的safetensors头均可读；O1200/C600/C1200/MT-BC300的全部manifest文件和大小核对通过，
  库存无计划外消失文件。数据、teacher/native observer/stable carrier/fit19依赖、旧V5.2参照和近期完整训练轨迹未动。
  本轮没有运行模型、GPU实验或重算科研结果。data1已盘点checkpoint载荷从592.366降至231.607 GiB；
  剩余小型历史节点不一概宣称必不可少，但本轮未对缺少逐项退休依据的资产继续扩大删除。
- 精确清单及验证：`runs/analysis/workspace_cleanup_20260923/checkpoint_retirement/closeout.json`；
  每个受影响checkpoint原目录有`checkpoint_retirement.json`，历史manifest保持原内容。

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

## 首轮收尾记录（追加checkpoint裁剪前）

- 长期要求和concept已整合；docs按designs/analyses/review_materials分工，历史材料有明确状态。
- task_plan和progress只保留当前状态，重复历史移至既有研究历史/发现入口；原详细过程可从Git恢复。
- 已结束的stability/output-space/causal专用诊断和low-LR执行面退役；历史实现以`7b18030c`及run记录的commit恢复。
- 代码retirement净删5101行、删除15个专用文件；保留常规resume、topology migration和1500→2100 continuation。
- 108项受影响CPU检查：107项首次通过；1项引用旧文档路径的断言修正后单独通过。三个canonical CLI的`--help`均exit0。
  40份配置JSON解析通过，Markdown本地链接无失效，源码/脚本/测试无对退役诊断模块的残留引用；未运行GPU或新科研验证。
- 三个临时worktree已移除，仅剩main；已合并的cleanup-code分支删除，未合并WIP的本地/远程分支保留且远端核对为`73267f53`。
- 大资产已完成回收：52个可再生LoRA bank共9739个路径（9339个唯一inode），44.924 GiB；
  明确一次性的profile载荷0.276 GiB；worktree副本0.731 GiB、已消费handoff/下载缓存/派生预览0.246 GiB。
  追加核实574个旧物化manifest、1717个payload，回收6.794 GiB。连同生成cache和本次临时文件，总计回收53.007 GiB：
  data0为38.682 GiB、data1为14.326 GiB。Hardlink按实际inode计数；原manifest和全部重建依赖保留，载荷原目录有退役标记。
  strg01收尾配额实测data0 122.1/1024 GiB、data1 846.3/1024 GiB；这是实时用户用量，不是共享磁盘空闲量。
- 清理账目：`runs/analysis/workspace_cleanup_20260923/storage_cleanup.json`；Source/teacher/正式checkpoint、raw rows与
  不可确认可再生的authority/task-local/轨迹证据保留。data1的大头是约530 GiB formal checkpoint载荷，另有约59 GiB探索checkpoint，
  不能根据负结果或目录名当作缓存删除。277组authority/task-local/特殊转换或provenance不全的产物有逐项保留原因，
  不凭“cache”命名推断可删。旧scratch中的唯一诊断记录和可用uv安装器也保留。
- 最终main包含文档/代码清理与验证修正，按仓库交付规则推送远端；仅保留canonical worktree。
  收尾没有启动新GPU实验、创建Sol或给任何任务派发后继实验。

没有后继实验派发；不要根据文档清理恢复任何运行。
