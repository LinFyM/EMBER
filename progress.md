# EMBER progress

## 当前状态：冻结正例复核未通过，回到机制理论裁决（2026-09-13）

Owner授权的自主goal保持：取得有益视频增量、跨视频／初始化／相邻保持及固定validation迁移，暂不要求145/400。
**整体goal未完成。当前无active实验design；时间对齐与冻结正例复核均已按完整证据关闭。**
当前工作转为综合机制与可识别性判断，不恢复历史训练、不自动扩大冻结复核或运行最终controls。
下一项实验须先有相对已失败近邻的实质机制区别、可区分竞争解释的预测和停止条件。

### 理论与只读审计补充

[视频信息与可识别性复核](docs/video_information_identifiability.md)已整理当前竞争解释及停止分支：
frame_set保留整条视频的全部采样画面，是无显式顺序的多状态参照；它与ordered打平不能证明没有使用视频。
train24的19个单原子／5个合取目标均检查当前状态，物理前置条件仍可能使时序有用。
原贡献+12/96的9个task在新交叉面板上合计−1/288（4正／1零／4负），旧优势也未在该task群中保持；
这项事后描述不筛选任务、不增加资格检验。原件在既有复核analysis根的`conditional_information_audit.py/json`。

采样合同的风险分解说明：已知task上的跨episode FM不强制依赖teacher的个体差异；这不等于未见task的执行知识
可以从语言免费获得，也不是视频无用定理。下一机制必须明确新增的知识与可失败预测；当前没有新模型、GPU任务或active design。

### 冻结正例复核：完整1,728次rollout

固定旧611770d1 ordered200与frame_set200；train24×teacher46–49×states0–7全交叉，
每模型768rows，source独立192rows。复用原192个LoRA，零新训练／Writer调用，不覆盖原manifest。

| 教师video | ordered /192 | frame_set /192 | 净增 |
| --- | --- | --- | --- |
| 46 | 90 | 90 | 0 |
| 47 | 93 | 89 | +4 |
| 48 | 89 | 87 | +2 |
| 49 | 87 | 95 | −8 |
| 总计 | 359/768 | 361/768 | −2 |

有序−静态为−0.26个百分点；固定task交叉、多层交叉、task均值三种20,000次配对bootstrap95%CI分别
[−4.04,+3.52]、[−4.69,+4.30]、[−2.73,+2.21]个百分点，均跨零。8task净正、6个零、10个净负；
S/O/G/L有序114/84/104/57，静态118/87/102/54。相对静态保留316／新增43／丢失45，churn88、J=.78218。
四条video并非均正，预注册的固定模型跨条件优势要求未通过；不扩大面板或追加训练。

source32/192（16.67%），有序359/768（46.74%）；有序−source+30.08pp，三种CI下界均>0，
最宽多层交叉区间[+18.88,+41.67]pp。说明训练任务的适应能力仍在，未兑现额外有序收益。
source广播只用于等权差额计算，实际样本始终192。两模型breadth22/24、21/24；192个task/state中
四条视频均成功74、69，不用这一单项保持指标替代总体／逐video的增量判据。

原52/96对40/96仍是原条件的有效历史事实；在本次登记假设下，新面板不支持其12.5pp量级的平均优势，
仍未排除小幅正负效果。新旧初始化不重合，旧96不是交叉面板且阳性受开发选择影响；
不能唯一归因到video、state、训练随机性或某个模块，也不证明视频／普通FM普遍不可能。

全部1,728行、model/video/task/state/env-policy RNG及共有noise序列已核验；9面板全部worker exit0，
累计评测墙钟4089.74秒，controller56404已退出。source入口首次prepare在GPU模型／rollout前退出，
原件保留于failed_attempts/attempt1；一行registered-subset screen8支持经116项测试后推送，
source用clean pushed c5a28747冻结运行面，两Writer保留原611770d1，source评测路径只有该准入行差异。
每项现场准入记录保留；data1最后阶段约966.2/1024GiB，2GiB新增峰值预算覆盖输出与218MiB source冻结树。
没有新模型／数据大副本或删除formal证据。当前没有selected checkpoint，未用最终controls、Test或RL。

原件：`runs/analysis/frozen_positive_replication_20260913/REPLICATION_READOUT.md`、
`paired_replication_summary.json`、`replication_decision.json`、`study_contract.json`、`mapping_provenance.json`，
以及同名outputs根的9项raw rows／completion。全部task/suite、逐video、source与换视频成功集合均保留。
完整判断见findings§73；原预注册合同为docs/frozen_positive_replication_audit.md§6，现已关闭。

下一步先把视频所需信息、初始化变化和共享参数竞争与已保存的证据对齐，提出能够产生不同观察的机制预测。
现有结果不支持“只差Compiler”、默认增加任务／模块或把有意的state-free教师输入当作新bug。
在预测与合法判别路径明确前不再投入完整Writer训练；整体goal继续，不以评测完成或局部指标完成目标。

### 时间对齐实验：完整结果与裁决

| 节点 | train96 ordered / frame_set | validation400 ordered / frame_set |
| --- | --- | --- |
| 100 | 44 / 36 | 61 / 65 |
| 200 | 51 / 53 | 72 / 81 |

- train100有序净增8，task-cluster95%CI[.03125,.13542]；200净额−2，CI[−.08333,.04167]，早期优势未相邻保持。
- validation100净额−4，CI[−.0325,.01]；200净额−9，CI[−.0425,0]，两节点均未达到正向资格。
  200有序相对静态保留62／新增10／丢失19，churn29、J=.68132；S/O/G/L为0/62/8/2对1/67/12/1，
  breadth4对6，仅Long净增1。区间端点0不写成严格负区间，也不是证实零效应。
- 有序validation61→72：保留50／新增22／丢失11，churn33、J=.60241，breadth5→4，Goal13→8。
  静态65→81：保留55／新增26／丢失10，churn36、J=.60440，breadth5→6。
  正确条件绝对能力改善未兑现有序增量；source47的任务异质性很大，不能仅凭总数超source完成目标。
- 2ecf1770 clean pushed冻结树，两臂world3各fresh200、800条件／51,200queries；100/200完整checkpoint、
  配对采样、offset1与finite验证通过。两节点动作留出FM有序差额均跨零；全部8个bank／1,984rows及worker exit0已核验。
  唯一Writer保留obs[i]→actions[i+1:]正确时序；该修正不被负结果撤销，也不能将历史差异单因归给offset。
- 不续训、不扫描参数，未触发other／原生image资格；没有selected checkpoint，不运行最终controls、Test或RL。
  这是有效科学non-pass，关闭的是此有界配方，不证明视频到LoRA或普通FM普遍不可能。

原件：`runs/outputs/execution_aligned_video_20260912/`；
`runs/analysis/execution_aligned_video_20260912/paired_summary.json`、`bounded_200_decision.json`、
`training_step200_pairing.json`、`materialization_mapping_audit.json`及各completion保留全部task/suite/source与相邻成功集合。
后台tmux ember-aligned-overnight已complete，controller3865623已退出。首面板结束后的共享文件可见性时差恢复记录及
attempt1日志保留；其余交接均通过，没有重复训练、物化或rollout。

### 上一候选：冻结视频先验比较已关闭

Pretrained Video Grounded的正式学习与8个correct面板全部完成，停止其原合同追加投入。
整体goal未完成；旧checkpoint和sealed原件保留。

### 本轮完整结果与决定

| 节点 | train96 ordered / frame_set | validation400 ordered / frame_set |
| --- | --- | --- |
| 100 | 41 / 41 | 53 / 48 |
| 200 | 52 / 40 | 33 / 48 |

- train200差额+12，task-cluster95%CI[.05208,.20833]为正，suite净额S+4/O+5/G0/L+3；100差额0、CI跨零。
  保留这一单节点闭环正证据，不能笼统宣布毫无过程获取，也不能据此认定共享表示已充分或只差Compiler。
- validation100差额+5、CI[0,.025]未满足严格正下界；200差额−15、CI[−.0825,−.005]为负。
  200有序相对静态保留24／新增9／丢失24，churn33，J=.42105；suite净额+1/−4/−11/−1。
- ordered100→200从53降至33，保留17／新增16／丢失36，churn52，J=.24638，Long5→0、breadth7→5。
  frame_set48→48却保留22／新增26／丢失26，churn52，J=.29730；总分持平不等于行为稳定。
- source validation47；两节点有序53/33、静态48/48。尚无可信稳定迁移与跨视频资格；未触发other和原生image参照，
  没有selected checkpoint，不做最终视频controls、Test、RL或训练续跑／参数扫描。

完整逐任务／suite、source对照、breadth、R/G/L、churn与相邻成功集合见
`runs/analysis/pretrained_video_grounded_20260912/paired_summary.json`；裁决为同根`bounded_200_decision.json`。
综合边界见[findings§70](findings.md)和[历史记录](docs/research_history.md)。

### 已完成执行及证据

两臂各fresh200／800条件／51,200主FM queries，实际18个采样字段及world4曝光匹配；0/100/200留出诊断完整。
100主FM有序小幅改善但CI跨零，200差额近零；平均损失不替代行为裁决。
全部8面板／1,984rows完整，所有worker return0；最后静态200 validation48/400、28分片，用时3636.50秒。
两臂100/200全部LoRA库sealed，跨臂／节点state–video映射一致，validation每task50条视频各一次；生成与学习均exit0。
全部本轮训练、生成、评测进程已退出，无GPU占用维持任务。所有checkpoint、raw rows、aggregate、completion与launch合同保留。

正式运行来自clean pushed detached `611770d13ab71bcee8284539914372935d07387e`，冻结运行树
`.codex/worktrees/pretrained-video-frozen`保留；本轮没有held梯度、Test或训练后task-local交互。
最后一次新评测启动前data1用量941.0GiB／1024GiB，study约25GiB，整项剩余峰值27GiB，预计968.0GiB；
这是当时launch预算而非实时quota，后续大增长前重新检查。source、资产和单份视频权重保持canonical，无dense磁盘缓存。

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
