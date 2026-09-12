# EMBER progress

## 当前状态：完成在途比较，停止自动串联局部修正（2026-09-12）

Owner已授权自主高效推进有益视频特异性及validation迁移，暂不要求145/400。
**唯一active design：[Video Functional Writer](docs/video_functional_writer_design.md)**，第7–9节已完成；第10节teacher侧VL Meta已完成fresh200及全部8个闭环面板；本项不追加。第11节单独移除辅助表示FM已完成fresh200，100/200既定闭环证据进行中。
旧C/无变化参照、95-task等历史路线继续停用；以下旧暂停记录不是当前执行授权。

### Owner最新纠正与执行调整

Owner指出近几小时没有根本进展，要求调整负结果后的分析与修正方式。此前去蒸馏、额外读出与VL适配虽各有
限定结论，但没有充分收窄原主假设；“尚不能否定整体”不能继续充当追加投入的依据。综合初判见findings§62。
正在运行的第11节保持200步上限，完成原登记100/200配对证据。其后先综合裁决主假设与竞争解释、明确停止投入
的路径，再决定有辨别力的修正；新候选和自动frame_set暂不启动。无需另设人工审批，不恢复旧任务或改变资格标准。

### 当前执行与完整结果

- 第11节fresh200训练完成，正式用时3339.18秒；800条件/51,200queries与teacher_vl参照逐字段匹配，100/200完整checkpoints齐备，训练进程退出。source保持冻结，峰值37.83GiB。
- held student FM100/200=.114742605/.107176777；相对参照改善.000179162/.000175662，task-cluster95%CI分别[.000050021,.000325814]/[-.000226025,.000602552]，18/24和13/24任务改善。差额很小，200区间跨零，不能替代行为证据。原件`direct_fm_vl/step{100,200}/learning_comparison.json`。
- 100训练两臂完整37/40（各96），teacher_vl参照34/37；correct R/G/L28/9/6、breadth19，other33/7/4、breadth18；差额CI[-.052083,.114583]/[-.052083,.125]均跨零。自身换视频重合33、churn11、J=.75；两臂进程均退出。原件`direct_fm_vl/step100/train_paired.json`。
- 100/200全部8个LoRA banks已sealed，物化进程退出；100 validation correct在gpu01:0/4、other在gpu02:0/1，各2GPU×3 workers运行。200 train correct/other在gpu01:6/5，各1GPU×3 workers，合计6≤6；200 validation待额度释放。
- 最新launch前两节点live检查通过，/data1 quota886.0GiB/1TiB、当前run13GiB，全部banks落盘后剩余为小型rows/logs，仍低于原893.6GiB投影峰值，共享83TiB；精确命令与资源原件`direct_fm_vl/step100/{validation_correct,validation_same_task_other}_launch.json`及`step200/{materialization,train_correct,train_same_task_other}_launch.json`。本轮未新增候选或额外面板。
- 第11节无辅助路径相关83测试通过，配置与8个评测请求的科学字段匹配。初始化按共同CPU构造顺序保持VL随机流；reader在进入GPU/state optimizer前丢弃，无新的执行模块或第二trainer。
- 真实最长task38/demo0 93frames：micro16 OOM已结束；micro8两次更新通过，第二次16.86秒/64queries（3.795queries/s）、峰值37.81GiB。Writer/Action/VL均有实际梯度，source冻结、source辅助forward=0。固定frame8/policy8；profile无checkpoint，原件`runs/analysis/video_functional_20260911/direct_fm_vl_profile/`。
- 第11节已从clean pushed detached `9b21b0ed34d5501f931b8c44f8922c2e6d254849`启动，执行checkout `.codex/worktrees/direct-fm-vl`；gpu01:0/4/5/6，world4，tmux `ember-direct-fm-vl`，fresh200/800条件/51,200queries，保存100/200。实际run contract确认reader0/source0、Writer337,568,000/Action626,688/VL921,600、NCCL/NUMA及物理8；初始held FM=.154849362与参照一致，已进入有效更新。
- 100/200全部评测请求已准备。启动前本人用卡0→4≤6，/data1 quota873.6GiB/1TiB，新增峰值预算20GiB、投影893.6GiB（含checkpoints/banks/临时与执行checkout），共享83TiB。精确命令与现场记录`runs/analysis/video_functional_20260911/direct_fm_vl/launch_contract.json`，输出`runs/outputs/video_functional_20260911/direct_fm_vl/`。
- 首次启动脚本误指旧output，fresh-run合同保护在任何训练更新前拒绝；已修正参数并刷新GPU后从fresh重启。旧formal结果未覆盖，失败日志与精确命令保留在新analysis目录`startup_output_path_rejected.{log,json}`，实际新run contract已核对。

- 第10节完整Z/R双路VJP已接入唯一observer；跨update只缓存pre-Gemma embeddings，新VL Meta在共同模块后初始化，新增921,600参数。旧KV缓存入口退役，model schema v2拒绝旧checkpoint；无第二trainer或执行adapter。
- CPU相关168项通过，配置小改后3项复核通过。真实native identity Z差0，两条VL梯度均非零，联合重放与直接梯度一致，source trainable=0；真实学习第2次后Writer/Action/VL/reader均有梯度。
- 最长合法视频task38/demo0，93frames/64queries：frame_chunk4/8分别24.97/21.55秒，后者2.970queries/s、峰值37.84GiB。选择frame8、policy micro8；profile只运行gpu01:4且已退出，无保留训练checkpoint。原件`runs/analysis/video_functional_20260911/teacher_vl_profile/`。
- 结构变更净增15行active source/test、无新文件；配置检查复杂度从28降为27，其余guard提示为既存大文件/函数与目录规模，未增长相关复杂职责。实现已集成并推送；formal冻结`a89648740c28d1a2f4aff07b6e42bc7b35e44223`；已结束的clean detached执行checkout在核对main ancestry后清理，全部formal资产保留。
- 第10节fresh训练已完成，原gpu01:0/4/5/6 world4进程全部退出；200updates/800条件/51,200queries与参照曝光匹配，训练4010.96秒，100/200完整checkpoints已保存。两组读取Meta共同学习、source trainable=0，NCCL/NUMA与逻辑batch已由实际run contract确认。
- 启动前两节点本人占用0，启动后4≤6；/data1独立quota861.2GiB/1TiB，新增峰值预算20GiB、投影881.2GiB，共享83TiB。精确命令与资源记录`runs/analysis/video_functional_20260911/teacher_vl/launch_contract.json`，日志`train.log`；输出`runs/outputs/video_functional_20260911/teacher_vl/`。100/200全部物化与评测请求已准备，训练及评测共享Owner全局额度。
- 100节点实际400条件/25,600queries与第8节逐字段匹配；held student=.114921766，参照.114956222，改善仅.000034455（task-clusterCI[.000004203,.000064595]）；reader=.153159816、差额CI跨0。差额很小，不能据此推定闭环收益；原件`teacher_vl/step100/learning_comparison.json`。
- 100四个banks已完整物化：train96与validation400各正确/同task另一视频，other复用相同条件LoRA而保留不同state-video映射。训练两臂已在gpu02:0/3各3 persistent workers完成并退出；原件`teacher_vl/step100/{materialization_launch,train_correct_launch,train_same_task_other_launch}.json`及对应preflight/logs。
- 100训练两臂完整34/37（各96），参照42/38；correct保留31/新增3/丢失11、差额CI[-.166667,0]，other33/4/5、CI[-.083333,.0625]。自身换视频重合30、J=.73171；100节点未见训练收益。原件`teacher_vl/step100/train_paired.json`。
- 200留出student=.107352440、相对参照改善−.000013986（CI[-.000078555,.000057563]）；reader=.149220013、微小改善.000016497，不能作为目标证据。全部800条件/51,200queries匹配，原件`teacher_vl/step200/learning_comparison.json`。
- 100/200全部8个banks、8面板/1,984rows完成；study累计56面板/13,888rows。两节点本轮进程已退出，峰值合计6张卡。最后200 validation两臂各2GPU×3 workers，均36jobs完整400rows；完整命令、资源与结果保留在`teacher_vl/step100`及`step200`。最近quota873.6GiB/1TiB、run13GiB，低于原881.2GiB投影峰值。
- 200训练两臂56/56（各96），参照55/55；净差均+1、CI均跨0，breadth21/20。100→200训练获取提高，但同节点换视频重合46、J=.69697，参照50、J=.83333。
- validation100两臂61/61，参照57/61；差额CI[-.02,.0425]/[-.0325,.025]。S/O/G/L=2/42/2/15、3/45/3/10，breadth8/7；correct的Long+7未在other复现（−3）。
- validation200两臂62/64，参照72/67；差额−10/−3，CI[-.0575,.0025]/[-.0375,.03]。S/O/G/L=2/42/6/12、2/45/4/13，breadth5/6；source47仅保留8/6，新增54/58、丢失39/41。
- 自身100→200 validation correct R/G/L32/30/29、churn59/J=.35165；other35/29/26、churn55/J=.38889。参照J=.32990/.34737，但correct保留仍32、获取更少，breadth8→5；不能把小幅J增加当作保持改善。200换视频重合44、churn38/J=.53659，差额CI跨0。
- **第10节裁决：**未形成值得追加的跨task收益与保持，不延长或扫描VL Meta rank/层位/LR，不自动补匹配frame_set。该因素并非input信息或全部VL学习的否证。原件`teacher_vl/bounded_200_decision.json`，全部逐task/suite、source及相邻配对在统一`paired_summary.json`。
- 当前goal未完成，无selected checkpoint、Test或最终sealed controls。第11节只移除L_R、保留当前完整图，fresh200已完成，配对闭环进行中。
- 此前Writer学习、动作头诊断、LoRA生成和闭环评测均已完成，无待恢复任务。第9节中层读出诊断也已完成并退出；它们不是当前训练的初始化或恢复来源。
- 去蒸馏配方固定rho=0、mu=1，fresh200更新/800条件/51,200queries，训练3967.10秒，峰值37.82GiB。完整100/200 checkpoints及全部800条件曝光匹配证据保留；source trainable=0。
- formal代码为clean pushed detached `ba4d1f4da16759a5ea1c5d7dec0dce1b0bd1b5e4`，原执行checkout已清理；原main/frame_set代码为`a81a38edd055034a4a080215bdc4362310500678`，其已结束checkout亦清理。配置解析及相关49测试通过，未改变原科学实现。
- 新配方8面板/1,984闭环rows全部完成；连同原40面板，study共48面板/11,904rows。精确命令、preflight、raw rows、aggregate和completion保留；统一逐task/suite及配对统计见`runs/analysis/video_functional_20260911/paired_summary.json`，裁决见`auxiliary_fm/bounded_200_decision.json`。

| 配方/更新 | train correct /96 | train other /96 | validation correct /400 | validation other /400 |
|---|---:|---:|---:|---:|
| 原main/100 | 35 | 38 | 72 | 71 |
| 去蒸馏/100 | 42 | 38 | 57 | 61 |
| 原main/200 | 45 | 47 | 74 | 74 |
| 去蒸馏/200 | 55 | 55 | 72 | 67 |

- 去蒸馏200训练两臂S/O/G/L=17/16/13/9、17/17/13/8，breadth21/20；对main的correct净增10，task-cluster95%CI[.020833,.1875]，other净增8、CI[-.020833,.197917]。同checkpoint两视频重合50，J=.83333。训练获取改善保留为正证据。
- 200 validation correct/other S/O/G/L=3/47/5/17、2/47/6/12，breadth7/6。对main净差−2/−7、CI[-.0875,.06]/[-.0925,.0425]；R/G/L37/35/37、34/33/40。Spatial/Long有局部收益，但Goal较main少15/12，不能以总获取替代迁移保持。
- 自身100→200 validation correct57→72：R/G/L32/40/25、churn65/J=.32990、CI[-.0675,.14]；other61→67：33/34/28、churn62/J=.34737、CI[-.075,.1125]。原main相邻J=.32727/.39423；没有可信保持改善。
- 200同task换视频other→correct67→72：R/G/L48/24/19、churn43/J=.52747、差额CI[-.0025,.03]；原main200两视频J=.66292。source47→新72/67只保留9/8，Goal task6原41→5/5，广泛保留未形成。
- 100对pureFM的validation只+1/+1，CI跨0；留出动作0/100/200 student FM=.154849/.114956/.107338，reader=.154849/.153161/.149237。动作拟合继续改善，不能替代闭环迁移或过程资格。
- 最后两项200 strict400分别耗时1202.78/1204.78秒，各42个queue jobs完整400rows。原新增存储预算20GiB，最近quota861.4GiB/1TiB、投影869GiB；未新增数据或模型副本。
- **本项裁决：**不延长去蒸馏训练，不自动启动同rho0 frame_set。两个节点、两臂validation都低于main，局部Spatial/Long正证据尚不足以证明保持改善；训练收益不能成为追加资格训练的唯一理由。
- **已完成诊断：**第9节仅改变固定表示后的功能读出：在Action Expert中层读取E，由后续冻结原生层转成动作；原Writer完全冻结，32epochs已全部完成。新头未形成更强教师，不追加拟合、层位扫描或自动重训Writer。没有新的Writer训练、selected checkpoint或最终sealed controls；Test继续封存，goal保持未完成。
- 新诊断从clean pushed detached `b1d3fa25`执行，已结束checkout随结果归档清理；临时单模块335行已退役，复用既有reader与原生cache。语法/CLI和真实smoke通过，source identity保持、输出/query/Value梯度存在、source无梯度；缓存/完整forward的最大FM差.000381，峰值9.93GiB。
- 新launch与完整资源记录在`runs/analysis/video_functional_20260911/native_reader_launch.{json,sh}`，运行日志`native_reader.log`、输出`native_reader_diagnostic/`。32epochs/192updates固定，1536支持queries复用、768留出queries无梯度；没有新的Writer训练或部署checkpoint。
- 启动前两节点本人作业0，启动后1≤6。/data1 quota861.6GiB/1TiB，新增预算<1GiB、投影862.6GiB，共享83TiB；gpu01可用RAM444GiB，缓存预计数十GiB且不落盘。两个旧任务运行树清理已核对clean与main ancestry，全部正式资产保留。
- 结构复核曾允许一个有明确退出条件的诊断入口；cache helper的循环/字段拼装复杂度17及writer目录27个文件为临时例外。问题关闭后已删除入口，未创建第二Writer运行面。
- 最终32epochs/192updates学习用时325.06秒；真实profile选择micro32约143queries/s，profile最高显存17.95GiB（包含micro64试测）。48条件实际video/action/query/RNG与旧诊断匹配，缓存/完整source FM最大差.000389，接受正常BF16路径差异；所有参照在同缓存路径重算。
- 新中层读出held0/4/16/32=.154897/.154597/.144469/.139792；旧拟合头=.140281，原LoRA学生=.110005。最终对旧头平均改善.000488、95%CI[-.003509,.003975]，15/24更好；对学生劣化.029787、CI[.021072,.039954]，24/24仍更差。
- 相对source改善.015083、CI[.010621,.020128]，22/24更好；16→32仍改善.004677、CI[.002959,.006544]。这些学习事实保留，但固定预算下没有更强教师证据，不能据末段获取无限追加。
- held原E相对跨task E优势.001718、CI[.001015,.002516]，19/24；E含语言/静态，仍非视频过程资格。完整原件`native_reader_diagnostic/results.json`，配对汇总`native_reader_analysis.json`。临时诊断入口随问题关闭退役，原代码由b1d3fa25及artifacts保留。
- 综合复核后，第10节只开放teacher侧PaliGemma VL Meta，保留第8节其余图和学习配方。历史审计曾提出该因素；现已按上方合同完成匹配检验，未获可信收益。两类弱头不证明输入无信息或唯一编译失败，goal保持未完成。
- Owner双节点GPU总上限持续有效：最多8张，空闲总数不超过10张时最多6张；全部训练、物化与评测共享。

### 已完成学习与验证

- 原Writer正式实验代码保存在Git `a81a38edd055034a4a080215bdc4362310500678`，已结束执行checkout已清理；原CPU套件378 passed，真实native梯度和最长93frames profile通过，source trainable=0。新诊断入口另以语法、结构检查及真实smoke验证，未改原Writer训练路径。
- main与frame_set均按登记完成300更新、1,200条件、76,800queries，checkpoint50/100/200/300完整保存；全部task/video/action/query/RNG/weight实际匹配。证据`frame_set/step300/exposure_alignment.json`。pureFM仅完成原登记100更新，无追加续训。
- main200/300续训段8657.20秒，frame_set对应段10470.90秒；frame_set峰值39.05GiB。所有LoRA banks及正式rows、aggregate、completion与恢复证据保留在`runs/outputs/video_functional_20260911/`。
- 40个完整面板共9,920条保留闭环rows，source参考train15/96、validation47/400；task/state/env/policy RNG配对通过。逐task/suite、breadth、R/G/L、churn、Jaccard和task-cluster bootstrap95%CI均见`paired_summary.json`。

| 配方/更新 | train correct /96 | train other /96 | validation correct /400 | validation other /400 |
|---|---:|---:|---:|---:|
| main/50 | 26 | 29 | 69 | 71 |
| main/100 | 35 | 38 | 72 | 71 |
| main/200 | 45 | 47 | 74 | 74 |
| main/300 | 43 | 46 | 32 | 32 |
| pure_fm/50 | 30 | 22 | 69 | 64 |
| pure_fm/100 | 41 | 40 | 56 | 60 |
| frame_set/50 | 29 | 24 | 66 | 67 |
| frame_set/100 | 35 | 38 | 73 | 74 |
| frame_set/200 | 46 | 47 | 69 | 61 |
| frame_set/300 | 48 | 44 | 27 | 31 |

### 当前裁决与后续

- 本轮未获得视频资格。50/100/200/300的main相对frame_set correct差额+3/-1/+5/+5，other差额+4/-3/+13/+1；primary正确视频95%CI均跨0。预注册必要门槛核对见`bounded_300_decision.json`，不以差额方向或单点峰值代替可信收益。
- main200→300 validation两臂均74→32，correct/other差额95%CI分别[-.18,-.0325]/[-.1925,-.0225]。frame_set相邻correct69→27、other61→31，CI[-.195,-.025]/[-.1425,-.015]。两方案后段均迁移退化，训练任务获取近似保持不能代替迁移保持。
- 已有train24留出动作100步main source/reader/student FM=.154849/.153163/.116660；251–300同批为.157011/.148191/.102797。辅助读出明显落后学生，尚无“好教师已学会、仅编译失败”的证据；这是诊断线索，不是退化的单一根因。
- 固定表示probe的support reader FM .150983→.136589（24/24改善），held .149662→.140251（23/24改善，平均降幅.009411、task-cluster95%CI[.006285,.012963]）；原学生held .110025，24/24任务仍优于reader。当前接口能学习可迁移到同训练task留出动作的修正，但尚无更好功能教师。
- held换入跨task E后FM .148876，相对原条件高.008625、CI[.005880,.011906]；E含语言与静态内容，此接口依赖不等于视频动态因果资格。逐task/节点见`reader_fit_analysis.json`及probe原始results。
- 同批损失可直接恢复输出误差内积：`<S-y,S-T>=(L_C+L_D-L_R)/2`。main/frame_set 151–200、251–300各50/50更新为负，汇总cos≈−.15。它限定在预测空间，不能推出参数梯度或Adam更新相反，更不能把迁移退化单因归给蒸馏。现有pureFM同时移除了两种辅助作用，故已单独完成去蒸馏比较；其训练收益未转为可信validation收益，不增加reader优化轮数或无依据延长。
- 当前goal仍未完成；无selected checkpoint，最终sealed视频controls未执行，Test继续封存。完整阶段结论与历史细节已转入`docs/research_history.md`和`findings.md`。

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
