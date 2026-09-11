# EMBER progress

## 当前状态：辅助FM有界比较全部完成；视频目标未达，进入机制复核（2026-09-12）

Owner已授权自主高效推进有益视频特异性及validation迁移，暂不要求145/400。
**唯一active design：[Video Functional Writer](docs/video_functional_writer_design.md)**，第7/8节的诊断及单变量比较均已完成；尚无新的GPU实验登记。
旧C/无变化参照、95-task等历史路线继续停用；以下旧暂停记录不是当前执行授权。

### 当前执行与完整结果

- 所有Writer学习、读出诊断、LoRA生成和闭环评测均已完成。两节点已确认没有本轮训练/物化/评测进程或auxiliary tmux；无待恢复任务。
- 去蒸馏配方固定rho=0、mu=1，fresh200更新/800条件/51,200queries，训练3967.10秒，峰值37.82GiB。完整100/200 checkpoints及全部800条件曝光匹配证据保留；source trainable=0。
- formal代码为clean pushed detached `ba4d1f4da16759a5ea1c5d7dec0dce1b0bd1b5e4`，`.codex/worktrees/auxiliary-fm-frozen`；原main/frame_set代码仍为`a81a38edd055034a4a080215bdc4362310500678`。配置解析及相关49测试通过，未改变原科学实现。
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
- **下一步：**结合本轮与既有正负证据复核最早失效接口，形成可区分竞争解释的新机制判断后再登记具体干预。当前没有新训练命令、selected checkpoint或最终sealed controls；Test继续封存，goal保持未完成。
- Owner双节点GPU总上限持续有效：最多8张，空闲总数不超过10张时最多6张；全部训练、物化与评测共享。

### 已完成学习与验证

- 原Writer实现及正式实验代码冻结在`a81a38edd055034a4a080215bdc4362310500678`、`.codex/worktrees/video-functional-frozen`；原CPU套件378 passed，真实native梯度和最长93frames profile通过，source trainable=0。新诊断入口另以语法、结构检查及真实smoke验证，未改原Writer训练路径。
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
