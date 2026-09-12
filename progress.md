# EMBER progress

## 当前状态：冻结视频先验两臂学习完成，配对评测中（2026-09-12）

Owner授权自主高效推进有益视频特异性及validation迁移，暂不要求145/400。
**当前active design为[Pretrained Video Grounded Writer](docs/pretrained_video_grounded_writer_design.md)，阶段：两臂fresh正式学习完成，配对评测中。**
Local Action Grounded全部16面板已完成关闭，整体goal未达。新候选已有7/8个correct面板：训练200有序正差额，validation100差额下界为0、有序200后段退化；尚未获迁移与稳定资格，静态200 validation仍在运行。
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

ordered已完成200updates／800条件／51,200主FM queries，用时3884.59秒；0/100/200各24-task留出诊断完整，
100/200 checkpoint均通过现有正式身份与完整性入口检查，训练以exit0结束且原四rank已退出。

100 correct的train96／validation400 LoRA库全部sealed，物化以exit0结束；validation每task50条视频无放回、跨节点固定state映射。
首个train96面板完整为**41/96**，source为15/96；Spatial/Object/Goal/Long为9/8/16/8，breadth18（source7）。
相对source保留11／新增30／丢失4，churn34，J=.24444；task-cluster95%成功率增量区间[.13542,.38542]。
其24个任务分片、96rows及worker exit0已核验；该相对source结果不证明视频内容／有序过程收益或validation迁移。
200 correct train96也已完整结束，为**52/96**，Spatial/Object/Goal/Long17/13/13/9，breadth18。
相对source保留14／新增38／丢失1；100→200保留26／新增26／丢失15，churn41，J=.38806，
相邻成功率增量95%CI[−.04167,.28125]跨零；Goal16→13，整体增长不代表低churn或已获得稳定视频过程增益。
200的30个分片／96rows、双worker exit0及进程退出已核验，用时939.64秒；与100的耗时差同时受checkpoint行为与并发数影响，
不单因归给双worker。`paired_summary.json`保留逐任务、suite、相邻及source成功集合；其它参照和视频尚未形成完整比较。

有序100 validation strict400已完整结束，为**53/400**（source47）；Spatial/Object/Goal/Long1/33/14/5，breadth7。
相对source保留17／新增36／丢失30，churn66，J=.20482，task-cluster95%增量区间[−.1925,.2075]跨零；
Goal41→14、Object5→33的变化相互抵消，尚无可信source相对迁移增量。28分片／400rows、双worker exit0及退出均已核验，
用时3793.60秒。
有序200 validation完整为**33/400**，Spatial/Object/Goal/Long1/17/15/0，breadth5；
相对source保留15／新增18／丢失32，churn50，J=.23077，增量95%CI[−.1925,.0675]。
100→200保留17／新增16／丢失36，churn52，J=.24638，增量CI[−.1525,.0175]；Object33→17、Long5→0，
训练200的有序正差额没有兑现该节点的source相对迁移收益或相邻保持。32分片／400rows、四worker exit0及退出已核验，2065.68秒。
静态100 validation完整为**48/400**，Spatial/Object/Goal/Long0/32/12/4，breadth5；
source相对保留16／新增32／丢失31，churn63，J=.20253，增量CI[−.2075,.195]。
有序100对静态100为53对48，保留41／新增12／丢失7，churn19，J=.68333，配对增量CI[0,.025]，
下界不严格大于0，未满足登记资格；suite有序净额S+1/O+1/G+2/L+1。静态32分片／400rows、四worker exit0及退出已核验，2129.39秒。
静态100 train96完整结束，为**41/96**，Spatial/Object/Goal/Long10/10/16/5，breadth19；
相对source保留11／新增30／丢失4，churn34，J=.24444，task-cluster95%增量区间[.13542,.38542]。
有序／静态100总分同为41；以静态为参照，有序保留35／新增6／丢失6，churn12，J=.74468，
配对增量95%CI[−.0625,.0625]跨零。Long+3、Spatial−1、Object−2、Goal0，该100节点尚无可信有序优势。
30分片／96rows、双worker exit0及退出已核验，用时969.27秒；全部per-task和成功集合在`paired_summary.json`。
静态200 train96也已完整结束，为**40/96**，Spatial/Object/Goal/Long13/8/13/6，breadth18；
相对source保留12／新增28／丢失3，churn31，J=.27907，95%增量区间[.15625,.375]。
200有序52对静态40，保留37／新增15／丢失3，churn18，J=.67273；配对增量95%CI[.05208,.20833]为正，
Spatial+4、Object+5、Goal0、Long+3。该正差额仅支持本节点train96，100为0差额，尚不证明跨节点或validation有益过程资格。
静态100→200保留25／新增15／丢失16，churn31，J=.44643，成功率增量CI[−.125,.10417]跨零，Goal16→13。
200的30分片／96rows、双worker exit0和退出已核验，用时977.87秒；两臂训练面板与全部成功集合均已汇总。

静态臂也已完成fresh200／800条件／51,200主FM queries，用时3510.27秒；0/100/200各24-task留出诊断齐全，
200 checkpoint通过正式身份入口检查，训练exit0且原四rank已退出。两臂配置仅`model.process_mode`和`video_prior.mode`不同。
`step200/learning_comparison.json`核验全部800条件／51,200queries、18个采样字段、world4拓扑与信息墙一致。
100留出主FM有序改善.0002066，95%CI[−.0000675,.0005043]；200改善−.000000342，
95%CI[−.0004171,.0004319]，10/24任务方向为正。初始诊断一致；尚无可信主FM有序优势，损失不替代闭环裁决。

静态100 correct的train96／validation400 LoRA库全部sealed、物化exit0且worker退出；
全部496条state–video映射与有序100一致，validation每task50条视频各一次。
有序200的两库此前已sealed，与100全部496条映射一致；train200已结束，结果见上。
全部两臂／两节点train96和validation400 LoRA库已sealed；每轮496条state–video映射跨臂、跨节点完全一致，
validation每task50条视频各一次。物化全部exit0，原生成进程已退出。
当前仅`ember-prior-static-val200`在第0卡以双worker继续最后一个validation strict400 correct面板，28个动态分片。
其它训练与评测已正常退出并释放设备；初始8个correct面板已有7个完成、1个运行。各exact command、进程及GPU身份记录在step launch artifacts。
完成该面板后作整体裁决；不因train200正差额启动条件性的other／image，不延长有序训练追峰值。

本轮证据根为`runs/analysis/pretrained_video_grounded_20260912/`，输出为`runs/outputs/pretrained_video_grounded_20260912/`。
exact command、GPU UUID、quota和fresh合同在`ordered/launch_contract.json`；profile证据在`profile/results.json`。
静态200 validation启动前已再次同时检查gpu01／gpu02，第0卡已释放、无进程；其它用户任务保持原状。
strg01/data1用量941.0GiB／1024GiB，当前study约25GiB，整项研究剩余峰值按27GiB估计，
预计968.0GiB，低于额度；共享83TiB可用。预算为整项研究的剩余增长，不对每个job重复增加。
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
