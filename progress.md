# EMBER progress

## 当前状态：冻结视频先验候选实现中（2026-09-12）

Owner授权自主高效推进有益视频特异性及validation迁移，暂不要求145/400。
**当前active design为[Pretrained Video Grounded Writer](docs/pretrained_video_grounded_writer_design.md)，阶段：隔离实现与资产准备。**
官方先验权重正在下载，尚未启动新GPU作业；Local Action Grounded全部16面板已完成关闭，整体goal未达。
原始信息墙、完整H、source冻结、主LoRA跨episode及零交互部署不变；K1/train24，未恢复95-task、RL或Test。

### 当前决定与工作

新增冻结V-JEPA2.1过去四帧视觉特征，task-conditioned dense Value读取进入现有完整native H及LoRA主FM。
沿用teacher单agentview，避免同时改变视角；仅主FM，退役已关闭的局部动作辅助支路。
匹配frame_set在同video encoder上逐帧输入四张相同真实图，不携带历史；只有两节点资格成立后才追加
原生image全帧静态参照，防止静态分支过弱。完整理论、可辨别预测、停止条件及最终controls均在active design。

接下来按隔离实现、信息路径／梯度验证、quota／资产与最长视频profile、clean pushed frozen学习推进。
约一小时窗口的实际checkpoint节点须在profile后、学习分数前登记；目前没有新formal命令或性能证据。

实施分支`codex/pretrained-video`位于`.codex/worktrees/pretrained-video`，基于已push设计登记`3c1d8155`。
已新增冻结先验owner模块，完成窗口索引与官方导入／确定性384预处理检查；尚未接入主图或验证真实权重forward。
官方代码已固定到登记commit；资产会话`ember-vjepa-assets`下载单份5,151,198,524-byte权重，
脚本、日志与状态位于`runs/analysis/pretrained_video_grounded_20260912/`；无GPU占用。
`asset_storage_budget.json`记录strg01当前/data1用量910.9GiB／1TiB、个人目录du911G、共享83TiB可用，
新增峰值预算61GiB，预计971.9GiB。formal前须刷新预算；不建立dense磁盘缓存。
官方代码所需`timm==1.0.29`已仅添加该包到现有环境，未改变其依赖；固定配置／lock正在隔离分支更新。
rollback命令与临时uv工具位置记录在同一资产预算文件。下一步完成唯一主图／重放接线、局部路径退役及针对性验证。


### 上一候选：Local Action Grounded已关闭

### 完整结果与当前判断

| 节点 | train96 ordered correct/other | train96 frame_set | validation400 ordered | validation400 frame_set |
| --- | --- | --- | --- | --- |
| 100 | 40/35 | 39/38 | 45/48 | 58/53 |
| 200 | 46/49 | 42/45 | 30/27 | 30/26 |

- validation100两视频有序差额−13/−5，task-cluster95%CI[−.06,−.01]/[−.025,−.0025]，均为负；
  200差额0/+1，CI[−.015,.0125]/[−.005,.01]，均无可信正增量。四项train匹配差额区间也均跨零。
- ordered的两视频100→200均退化，Long8/8→0/0；frame_set也退化，Long14/9→0/0。
  ordered相邻correct/other的J=.38889/.31579，frame_set=.33333/.36207；未获相邻或跨视频有益过程保持。
- 局部留出100/200有序改善.00006528/.00117193，区间下界为正；200为24/24任务正向。
  主动作FM差额−.00018689/+.00003742，区间均跨零。局部头小幅优势没有兑现主LoRA收益。
- 局部头有独立时间Key路由，不更新Compiler时间投影；不能认定共享E已经形成完整过程、只剩Compiler失败。
  同图无局部目标归因臂未获触发资格，不能把与历史配方的差异单因归给局部监督。

完整归纳见[findings§69](findings.md)，launch、学习、评测与成功集合摘要见
[research_history](docs/research_history.md#local-action-grounded)。本轮是有效科学non-pass，不是已发现的工程故障。

### 执行与证据封存

- 实现与formal来自clean pushed detached `5f4f440c992e470aaffcc873c847373ce6df43e5`，冻结运行面
  `.codex/worktrees/local-action-frozen`仍干净。两臂world4、同拓扑、fresh Writer/Meta/局部头及optimizer/scheduler；source始终冻结。
- 每臂200更新、800条件曝光、51,200主FM queries、800局部片段及6,400noise/time draws；实际任务、teacher、query、
  局部片段及noise流逐字段匹配。训练分别3337.34/3307.92秒；完整100/200 checkpoint及0/100/200两类无梯度诊断保留。
- 16个LoRA库及16个登记闭环面板完整，共3,968rows。所有launcher completion均为完整队列、worker exit0；
  最后controller及worker PID已核实退出，本轮无GPU进程继续运行。没有selected checkpoint、pureFM第三臂、Test或sealed controls。
- canonical输出：`runs/outputs/local_action_grounded_20260912/`；分析：`runs/analysis/local_action_grounded_20260912/`。
  `bounded_200_decision.json`登记完整裁决，`paired_summary.json`保存所有task/suite、source、breadth、R/G/L、churn、相邻及换视频比较；
  `step100/learning_comparison.json`与`step200/learning_comparison.json`保存两类学习及曝光对齐。
  exact命令、两节点preflight、quota预算、launch records与raw rows/aggregate/completion均在原根，未删除checkpoint或大资产。

### 当前停止范围与下一步

[Local Action Grounded设计](docs/local_action_grounded_writer_design.md)已按预登记条件关闭；不续训、不扫局部头容量、
辅助系数、LR/rank/seed，不启动条件触发的pureFM第三臂或最终视频controls。旧Video Functional、C/无变化参照、
95-task及其它历史候选仍不因旧文件的“下一步”而恢复。

下一候选已在上方登记；旧实验停止范围不因新候选激活而解除。
已核查任务覆盖旧实验（findings§68）和外部视频先验边界（§64），新设计是可检验的知识来源假设，
不是把本次负结果当成预训练缺失的唯一根因。具体工作与授权只看本文当前段和active design。

以下为旧暂停时点的历史记录，不能覆盖上面的当前状态、目标或授权。

## 暂停时点的已完成实验与未完成范围

无变化参照候选从冻结`64eba75bcf316a1f6792ab8ddf5eb1aaefded378`完成fresh200更新/51200queries，
用时3327.11秒，100/200完整checkpoint保留。唯一主要变化为C局部更新采用同gap/角色/窗口的无变化参照，
保留train24、每task两个K1条件各32queries、普通正样本FM、source冻结；没有RL、错误视频惩罚或任务覆盖变化。

训练task独立留出动作FM由.151451降至.109323，24/24任务改善。固定teacher46、states32–35的train100→200
correct为36→45，各96；相邻保留22/新增23/丢失14，churn37/J=.37288。200正确/换视频/同suite错/跨suite错/乱序/静态
为45/42/41/37/41/40，各视频差额的task-bootstrap95%区间均含0。200 validation完整为34/400，source47、旧C20050；
Spatial/Object/Goal/Long为4/24/5/1，breadth5。初步训练条件分化未成为可信视频机制收益或迁移改善。

100 validation在Owner暂停时终止，没有可报告的strict400成绩；300/400未启动。首段不能证明候选收敛或整个机制被否定。
`runs/analysis/video_change_reference_20260911/owner_pause.json`记录定向停止及checkpoint保留；本节不是实时GPU进程检查。

此前完整C冻结诊断共1472条新闭环及14336次配对动作预测：C200 correct42、两错46/46、乱序45、静态49、S51、source15，
各96；固定语义S后correct/static过程同为15/32。静态首帧仍产生正常视频.692–.935倍中心化P4，支持一个具体表示性质，
尚非全局根因。无变化参照修正已在真实native输入上削弱该性质，但行为收益未被证明。

| 暂停前路线 | validation /400 | 训练任务观察与边界 |
| --- | --- | --- |
| R，off图+两个独立K1条件 | 100/200/300/400=37/83/63/85 | train200/400=46/56，各96；同query预算旧off400=126/56，未获迁移收益 |
| C，语义条件化过程消费 | 100/200=43/50 | 轮换held视频train96=49；固定teacher46机制面板42，不能混用 |
| S，无序帧集合参照 | 100/200=59/45 | 轮换held视频train96=52；固定teacher46机制面板51；不是单帧static模型 |
| off7续训 | 400/500/600=126/73/54 | 轮换held视频train96=56→64；训练获取与未见task能力分离 |

R/C/S及两项续训共4480条闭环，原件根`runs/analysis/video_consumption_20260911/`；C诊断根
`runs/analysis/video_mechanism_20260911/`；无变化参照根`runs/analysis/video_change_reference_20260911/`。
远程副本统一由新专家材料的index定位，`runs/`路径本身不表示已经发布。

95-task提案未启动GPU实验，其两个未提交worktree原样保留，不集成或清理。精确历史和旧合同由Git与
[research_history](docs/research_history.md)保留。后续执行只能依据Owner新授权及新登记，不能沿本文历史段恢复。
