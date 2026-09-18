# EMBER task plan

## 已完成：v5.2冻结机制与SFT差距诊断（2026-09-18）

1. 已核对共享SFT与A900的实际函数、数据/优化合同及既有五组400行；wrong净优势主要集中task3/31。
2. 已完成train24×10臂冻结功能对照。相同真实RGB下改变Writer语言、固定同一LoRA及关闭Procedure均按预注册执行。
3. 已完成全部960条配对闭环：正确A58、关闭Procedure34、固定视频保留目标语言38/38、固定LoRA13/10、Source12、SFT47，各/96。
   旧Procedure有明确当前功能贡献；正确视频增量主要经该路径。没有新训练、held actions、Test或checkpoint选择。
4. 已按Owner约130–140及视频特异性的要求收窄后续设计依据；少量正常churn不单独否决。
   当前尚未选定新架构，也没有把冻结消融当fresh删除收益；统一图多变量退化的唯一内部原因仍未定位。

原件：[预注册](runs/analysis/v52_mechanism_audit_20260918/registration.json)、
[完整报告](runs/analysis/v52_mechanism_audit_20260918/report.md)。本轮诊断已结束，无待自动恢复的运行或新训练goal。

## 本轮目标与完成状态（2026-09-18）

Owner于2026-09-17授权的统一Writer整套实验工作已完成：仓库/data1整理、新架构实现与高效训练、
比较已有v5.2、审视有依据的修正，以及报告交付。本轮在预注册1500节点作有界non-pass裁决。
该实验没有active design、运行或待自动执行的计划；实际证据与结束状态见[progress](progress.md)。

## 已完成事项

1. **仓库与空间整理。** 核实data0/data1独立quota，审计源码、测试、脚本、配置、文档及运行资产。
   退役确认过时/重复的代码与配置；清理23,224个可重建LoRA payload、13个干净已集成历史工作树，
   以及本轮完成的profile和临时实现/运行工作树。保留正式证据、唯一checkpoint、数据/source和未合入工作。
2. **统一模型与六卡执行。** 完成中层/末端同构Z/H处理、一次native双写回、连续参数状态和完整38-target A/B。
   三Meta与Writer fresh联合纯FM。六卡真实帧/query分片保持4task/global84及一次更新，
   通过梯度、原生接口、最长视频与完整恢复检查；profile选20帧chunk，六卡相对同事件四卡吞吐1.45倍。
3. **正式学习与既有比较。** clean pushed frozen commit运行至1500，6000条件/126000queries，15个完整恢复点。
   600/900/1200/1500各完成correct400与train96，共1984条完整闭环；同source配对A只复用已完成原件。
   新模型验证90/109/67/84，train38/54/54/52；原v5.2 132/复核125作为不同source/video映射的描述性参照。
4. **修正可行性裁决。** 没有发现可复现工程错误或能限定具体失败接口的证据。
   已复核73/75-task及meta73/target18历史，不把95-task扩展或rank/scale/seed/LR/dtype扫描当作默认修复。
   后段缺乏持续整体获取，1500反弹集中在单个Long任务；按停止条款结束，没有启动第二轮重训。
5. **分析与交付。** 保留per-task/suite、breadth、retained/gained/lost、churn、Jaccard、配对区间与资源成本。
   已生成[完整报告](runs/analysis/unified_writer_20260917/experiment_report.md)、
   [逐任务表](runs/analysis/unified_writer_20260917/unified/analysis/primary_tables.md)及可导出图表；
   当前状态、findings和research_history同步，源码、科学证据及正式checkpoint保留。

## 已关闭而非待执行的分支

- 初始窗口上限2400，按预注册1200后的相邻证据条款在1500停止；1800/2100/2400未执行。
  不宣称已训练2400、已完全收敛或穷尽所有统一架构的潜力。
- 无节点达到correct严格>145/400及相邻资格，因此没有selected checkpoint。
  后续same-task-other、learned language/static参照及冻结后最终wrong/no-video/shuffled/reversed controls未启动。
  本轮未证明动态视频必要增量或时序特异性改善，未使用这些controls返工。
- Owner取消的300点从未启动；误启动的重复v5.2在73步停止，不纳入科学比较，此后没有重启。
- 未恢复旧A3000评测、C或其它无关实验；无Test使用、无held梯度、无RL。

## 保留的依据

- [封存统一设计](docs/v52_evidence_based_writer_design.md)及[设计前证据审计](docs/v52_evidence_audit_20260917.md)。
- findings§117、[研究历史](docs/research_history.md)及study的`experiment_completion.json`。
- 正式运行代码commit `184947cbd9bb1f05c4a4684f390633f10374e0b2`及study内sealed config/命令/原件。
  临时frozen worktree已在确认进程退出、干净且集成后删除；这不删除历史代码或checkpoint。
