# EMBER progress

## 当前状态：冻结视频先验正式有序臂学习中（2026-09-12）

Owner授权自主高效推进有益视频特异性及validation迁移，暂不要求145/400。
**当前active design为[Pretrained Video Grounded Writer](docs/pretrained_video_grounded_writer_design.md)，阶段：fresh正式学习与配对评测准备。**
Local Action Grounded全部16面板已完成关闭，整体goal未达。新候选尚无正式闭环成绩。
原始信息墙、完整H、source冻结、主LoRA跨episode及零交互部署不变；K1/train24，未恢复95-task、RL或Test。

### 当前决定与工作

冻结V-JEPA2.1过去四帧dense特征，经task-conditioned Value读取进入现有完整native H及LoRA主FM。
沿用teacher单agentview，只有主FM；旧局部动作辅助支路已退役。主比较为同encoder的ordered／逐帧四张重复图frame_set；
只有两节点资格成立后才追加原生image全帧静态参照。完整预测、停止条件和最终controls见active design。

实现与节点配置已集成并push，正式训练从clean detached `611770d13ab71bcee8284539914372935d07387e`
（`.codex/worktrees/pretrained-video-frozen`）运行；370项共享源码测试通过。真实官方EMA权重strict加载通过，
最长93帧teacher的两次完整单条件反向已完成；第二次19.5573秒、allocated39.0794GiB／reserved42.0840GiB，
Writer、两组Meta和prior投影获得梯度，source／prior始终无梯度。profile没有保存或复用正式初始化。
集成完成且profile进程退出后，已移除干净的task-owned实现worktree／分支；正式冻结运行树保留。

在学习前固定100/200节点，每臂200updates、800个K1条件及51,200主FM queries；四suite每update各一task、64queries／task。
保留frame chunk8、policy microbatch8、prior window batch4。profile速度仅是约65分钟纯更新成本参考，
不含启动、诊断、checkpoint和数据读取成本；实际段长以formal日志为准。

ordered已在gpu01的0/1/3/4四卡fresh启动，tmux `ember-prior-ordered`；实查torchrun及四rank存活，
正式run contract确认clean pushed commit；现已完成100更新、0/100各24-task留出诊断及完整100 checkpoint，继续向200学习。
100 checkpoint通过现有正式身份／完整性入口检查；correct的train96／validation400物化全部完成、两个库sealed，
`ember-prior-mat100`以exit0退出，验证了validation每task的50视频无放回及固定state映射。
`ember-prior-train100`已在gpu01第6卡启动train96动态队列评测，正式合同确认24tasks／96states，worker已ready；
`ember-prior-val100`已在释放的第5卡启动validation strict400，两个worker均ready，正式合同为8tasks／400states／28动态分片。
单worker实测约10.8GiB显存，据此采用双worker并记录吞吐；训练与两类评测合计6卡，尚无完整闭环面板结果。
frame_set启动脚本已准备但尚未运行；前臂结束后须按live资源重新选择并记录，不能把历史空闲卡当成预留。
训练与全部评测共同遵循当前6卡额度；待200节点与训练96面板完成后，按live资源安排200物化及匹配frame_set学习。

本轮证据根为`runs/analysis/pretrained_video_grounded_20260912/`，输出为`runs/outputs/pretrained_video_grounded_20260912/`。
exact command、GPU UUID、quota和fresh合同在`ordered/launch_contract.json`；profile证据在`profile/results.json`。
正式启动前strg01/data1用量916.2GiB／1024GiB、共享83TiB可用；剩余研究峰值50GiB及冻结worktree0.3GiB，预计966.5GiB。
单份官方权重4.8GiB已落canonical模型根，临时下载文件已消失；不建立dense磁盘缓存。

两个主臂100/200的train96与validation400 correct物化请求已准备，沿用固定seed20260911、held46–49／states32–35
及validation各50视频映射。先完成这8个面板；有正向候选才补other和条件性的image静态检查。
统计报告包括source、task/suite、breadth、R/G/L、churn、相邻及换视频成功集合；最终内容／顺序controls仍待方法选定后。

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
