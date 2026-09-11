# EMBER progress

## 当前状态：Video Functional Writer设计与实现（2026-09-11）

Owner最新明确“开始推进”，给予足够自由度并要求高效率。已创建活动goal：先实现有益视频特异性及validation迁移，
暂不要求145/400；达到后再保持视频收益提升性能。此前暂停只继续适用于旧C/无变化参照、95-task等历史路线，
不阻止本次新设计的实现、profile、正式学习、匹配对照和证据驱动修正。

**唯一active design：[Video Functional Writer](docs/video_functional_writer_design.md)。**
选择时间×任务token有序表示＋执行query条件化辅助真实FM；完整LoRA从第一步真实FM，
仅Compiler/native D接受归一化蒸馏，encoder的真实LoRA FM系数固定1。部署仍只有唯一完整38-target A/B。

### 当前工作

- 新模型、分组信用、runtime/config/schema及评测适配已合并并推送main a81a38ed；完整CPU测试378 passed。
- 旧图运行路径已退役，原结果/冻结worktrees保留；本次实现工作树已进入集成后清理。
- 最长合法训练视频93frames真实profile通过：原生/捕获FM均.106989；rho=.25时batch4/8为3.10/3.20queries/s、峰值25.32/37.84GiB；batch16 OOM。未保存profile权重。
- 首段在看到正式学习结果前登记50/100更新；两配方均已完成100更新，首批闭环评测进行中。
- 训练来自clean pushed detached `.codex/worktrees/video-functional-frozen`，commit `a81a38edd055034a4a080215bdc4362310500678`。
- 主方案：gpu01:4,5,6，microbatch8×3，tmux `ember-video-functional-main`；pureFM：gpu02:3,6，microbatch8,4，tmux `ember-video-functional-fm`。后者与低负荷服务共驻，留出显存；不干预其它进程。
- 运行根 `runs/outputs/video_functional_20260911/{main,pure_fm}`，精确命令/设备/数据/预算见父目录 `launch_contract.json` 及各run contract。
- 实际run contract已核对：同冻结commit、完整schema，source trainable=0；主方案reader864000参数，pureFM无reader参数。
- /data1独立quota使用787.1GiB/1TiB，runs实测636GiB；两配方首段checkpoint及全配对物化预算40GiB，预计峰值827GiB，共享空闲83TiB；不复制基础资产。
- 下一步核查首批真实更新，再对50/100单checkpoint做训练获取与validation correct/other strict400；不能用学习loss宣称视频资格。

### 首段实时证据

- pureFM已完成fresh100/25,600queries，用时3185.59s；50/100完整checkpoint通过身份检查。固定train24留出动作FM从.154865降至.114877（32queries/task、相同视频/query面板）。主方案也已完成100更新/25,600queries，用时4123.94s，留出动作FM从.154849降至.116660；辅助读出仅降至.153163，尚不能称为合格功能教师。
- 两配方完整100更新的400条件/25,600queries实际曝光逐条匹配：任务、视频、动作query、RNG种子及权重一致；记录`runs/analysis/video_functional_20260911/first100_exposure_alignment.json`。
- pureFM50固定训练面板correct30/96，严格配对source15/96；Spatial/Object/Goal/Long为9/2/12/7，对source净增3/2/5/5，breadth12/24。保留11、增19、失4，churn23，Jaccard.3235；task-cluster bootstrap净成功率增益95%CI[.0521,.2708]。
- 以上是训练任务行为获取，尚无视频特异性或validation迁移结论；不是checkpoint选择。完整统计见`runs/analysis/video_functional_20260911/paired_summary.json`。
- pureFM50/100、主方案50的训练96+validation400 LoRA均已物化；每个checkpoint的other共496条件全部hardlink复用，无重复编译。主方案100也已完成全部物化。
- 四轮validation strict400并行：pureFM50 correct在gpu02:4，pureFM100 correct在gpu02:6，主方案50 other在gpu02:3，以上各3workers；主方案50 correct在gpu01:5,6共6workers。采用long-first动态队列与持久worker。
- 主方案100训练correct已在物化释放的gpu01:4启动3workers。gpu02:0曾通过空闲检查，但另一作业在启动窗口进入；主方案100 validation correct加载OOM、0rows，失败原件保留于`runs/analysis/video_functional_20260911/failed_attempts/main100_validation_correct_gpu02p0`。未干预他人作业。
- 主方案50 validation correct完整69/400，source47；S/O/G/L=4/48/16/1，breadth7/8。保留20/新增49/丢失27，churn76/J=.2083；task-cluster95%净增益CI[-.1700,.2975]。主要task11为5→42、task26为41→15，未形成可信广泛迁移结论。
- 主方案100 train correct完整35/96，source15；S/O/G/L=7/8/15/5，breadth13/24，保留11/新增24/丢失4，churn28/J=.2821，task-cluster95%CI[.0729,.3333]。
- 上述两面板完成后，live gpu01:4/5/6均空闲；100 validation correct已重调度到5,6共6workers，50 train correct在4上3workers。原gpu02三轮验证继续运行。
- 后续补齐50/100两配方的correct/other及训练面板，再决定辅助信用是否保留，并采用对应配方的匹配无序视觉参照。4states训练诊断使用screen，validation400使用formal；不以部分队列结果选点。

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
