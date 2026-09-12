# EMBER progress

## 当前状态：100步train96为有序44／静态36，validation与200节点待裁决（2026-09-13）

Owner授权自主推进有益视频特异性、跨视频／初始化／相邻保持及validation迁移，暂不要求145/400；整体goal未完成。
**当前active design：[Execution-Aligned Video Writer](docs/execution_aligned_writer_design.md)。**
综合机制判断及旧endpoint／expert近邻核对见findings§71；前轮冻结视频先验比较保持关闭。

Owner新增要求整理最后一次专家意见后的情况与咨询prompt，由Owner自行转发。
已形成[整体复审prompt](docs/review_materials/20260912_followup/EXPERT_PROMPT.md)：包含已完成的正负证据、
在途结果边界及假设裁决／视频知识可识别性／下一项判别实验的问题。当前运行仍按下述既定合同执行。

Owner随后提供专家整体复审，并在睡前要求自主有效推进，随后明确要求重新设置goal。
新goal已激活：在证据支持下自主完成分析、理论修正、设计、实现与验证；目标仍是可重复视频收益及保持／迁移，
不以评测收尾或局部涨分完成目标。失败后先综合判断；优先审计并登记冻结正例复核，只有实质机制证据支持时才开启方法修正。
当前夜间队列为已登记的100/200、ordered/frame_set、train96/validation400共8个correct面板与配对汇总；
在形成新的实质机制证据前暂停完整Writer训练。
专家建议先收尾本轮，若未获资格，优先复核旧冻结模型52/96对40/96的跨视频／初始化证据。
该复核尚未启动；先核对顺序controls用途、完成具体预登记与成本判断，后续执行依据最新Owner授权和登记设计。
后台队列只执行当前8面板，按首阶段判据输出初步候选／关闭建议；候选不等于选点或goal完成。
等待期完成[冻结正例复核审计](docs/frozen_positive_replication_audit.md)：旧96配对核验为新增15／丢失3；
原面板混淆video与state，768交叉面板的判别力依变异结构而异。正确臂可复用已有192个LoRA，
两模型加source共1,728次新rollout；8个fixed-video×states0–7映射已通过原运行面CPU合同核验，
无需改模型或冻结源码。顺序干预用途／合同尚待登记，未启动新复核。

唯一Writer已改为post-action obs[i]→actions[i+1:]，排除无未来动作标签的最后query；source与normalization冻结。
624条train动作episode的CPU审计、四suite真实dataset首／尾8例、378项全套测试通过。
首次两臂在GPU／output初始化前exit1，原因是入口仍要求上一候选登记名称；已修正并通过43项相关测试。
外层shell曾在preflight拒绝后继续调用launch，重试已分离资格检查与启动调用。失败日志和原命令保留在analysis/failed_attempts，
首次无训练更新或checkpoint。新formal来自clean pushed detached `2ecf17704c68644b948ad6655b2d73d468cf6200`，
冻结树`.codex/worktrees/execution-aligned-frozen`；旧checkpoint没有用于初始化。

### 本轮实际执行

- ordered：gpu01 `[0,5,6]`，tmux `ember-aligned-ordered`，controller3864138、rank3864157/58/59。
- frame_set：gpu02 `[0,1,2]`，tmux `ember-aligned-static`，controller3163713、rank3164131/34/36。
- 两臂world3，共6张真实工作卡；每更新仍四suite等权4tasks、256queries，microbatch8、fresh纯主FM至100/200。
  gpu02既有低利用率148–186MiB进程已核实所有权，启动余量覆盖约42GiB实测reserved峰值；未修改其他用户进程。
- 两臂100/200检查点均已完成formal inspector检查：模型／trainer／三rank状态完整，新身份与offset1一致；训练已退出。
  首段各400条件／25,600queries，18项任务／video／动作起点／noise等字段配对一致，各suite100条件，全部训练指标finite。
  初始24task诊断FM均.1532853388；100步有序.1146888600／静态.1146933126，静态减有序+.0000044525，
  paired-task bootstrap95%CI[−.0001895658,+.0002019832]跨零，9/24task为正；没有可信的诊断有序优势。
  这是训练侧动作留出结果，不是视频闭环收益或checkpoint选择依据，不触发配方调整或提前结束。
  两臂训练更新分别累计2568.99／2667.07秒（不含加载／诊断／保存），reserved峰值42.0645／42.0625GiB。
- 启动前strg01独立data1用量941.2/1024GiB，已含冻结树；初始阶段增长预算28GiB，整个study含条件后续70GiB，
  预计峰值1011.2GiB，共享余量83TiB。后续物化/新增长前刷新剩余quota与GPU。

输出：`runs/outputs/execution_aligned_video_20260912/`；分析／exact launch／设备UUID／startup验证：
`runs/analysis/execution_aligned_video_20260912/`。formal run_contract记录source和prior trainable均0及新offset1。
首段证据为analysis下`training_step100_pairing.json`、`diagnostic_step100_comparison.json`及各step100/checkpoint_inspection.json。
两臂200步均segment_complete且exit0，各800条件／51,200queries；全部18项配对字段、offset1和finite检查通过。
200动作留出有序.1068289227／静态.1068355200，静态减有序+.0000065973，
task-bootstrap95%CI[−.0002747199,+.0002935941]跨零；10/24task正向。此项仍不是闭环资格。
学习总墙钟有序5465.92秒／静态5708.51秒；完整证据见training_step200_pairing.json与diagnostic_step200_comparison.json。

夜间后台tmux `ember-aligned-overnight`继续运行，当前controller PID3865623（attempt2）；
队列复用原materialize.sh/evaluate.sh，已从原2ecf1770 frozen运行面完成4组、合计8个sealed LoRA库，
全部物化exit0；1,984个条件的跨arm／checkpoint映射一致，validation每task全部50视频各一次，
证据为analysis/materialization_mapping_audit.json。已进入顺序8面板评测，dynamic queue／persistent workers／long-first保持。
首个ordered100 train96在gpu01[0,1,3,4,5,6]、每卡2worker完成：44/96，全部12个worker exit0，
墙钟250.98秒。匹配frame_set100已完成36/96，墙钟255.45秒、所有worker exit0。
完整配对有序新增10／丢失2、保留34，churn12、J=.73913；task-cluster95%差额CI[.03125,.1354167]为正。
Spatial/Object/Goal/Long有序9/13/15/7，静态7/12/12/5，四suite净收益均为正，有序breadth18。
证据为analysis/step100_train_comparison.json（含全部task/suite/source比较）；只支持该训练任务面板的局部增量，
还不能证明视频顺序因果性、相邻保持或validation迁移。队列正在ordered100 validation400。
首个面板结束后，远端已写completion约1秒，controller首次读取却报FileNotFound，随后该文件可读且完整。
按共享文件可见性时差在ops脚本加入最长60秒的有界读取等待，具体NFS缓存层未独立定位；不改变评测实现。
旧controller已退出，attempt1日志保留；恢复前核验已完成96rows及checkpoint身份并跳过全部8库／首面板，
只接续剩余7面板，无重训、重复物化或重复rollout。frame_set100完成后已自动进入validation，实际交接通过。
每次launch现场检查两节点，选同节点至多6张有效设备；计入其它ymdai GPU后仍不超过保守总上限6。
物化前quota实测956.4/1024GiB，当前study16GiB，初始8面板剩余峰值12GiB，预计968.4GiB；
首个评测准入时data1实测966.0GiB，评测剩余预算2GiB，预计968.0GiB；
队列各阶段再次检查quota和共享容量，不启动任何条件性额外增长。
状态为analysis/overnight_status.json，确切命令与现场准入证据为overnight_*日志／admission文件。
完成后自动生成paired_summary.json、bounded_200_initial_decision.json与OVERNIGHT_READOUT.md；
失败或资源不合格时停止后续启动并保留日志，不自行重训、参数扫描或扩大实验。100步训练侧局部正例不构成方法资格，原注册分支与条件性冻结复核触发条件保持。
接下来按实际checkpoint完成物化与strict配对，并报告全task/suite、breadth、R/G/L、churn与相邻稳定性。
other／原生image参照只在资格触发后补；没有selected checkpoint，不运行sealed controls、Test或RL。

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
