# EMBER progress

## 当前状态：Video Functional Writer设计与实现（2026-09-11）

Owner最新明确“开始推进”，给予足够自由度并要求高效率。已创建活动goal：先实现有益视频特异性及validation迁移，
暂不要求145/400；达到后再保持视频收益提升性能。此前暂停只继续适用于旧C/无变化参照、95-task等历史路线，
不阻止本次新设计的实现、profile、正式学习、匹配对照和证据驱动修正。

**唯一active design：[Video Functional Writer](docs/video_functional_writer_design.md)。**
选择时间×任务token有序表示＋执行query条件化辅助真实FM；完整LoRA从第一步真实FM，
仅Compiler/native D接受归一化蒸馏，encoder的真实LoRA FM系数固定1。部署仍只有唯一完整38-target A/B。

### 实现、训练与证据

- 科研源码已合并并推送main a81a38ed；CPU测试378 passed，真实native梯度及最长93frames profile通过。正式执行来自clean pushed detached `.codex/worktrees/video-functional-frozen`、commit `a81a38edd055034a4a080215bdc4362310500678`；原生source trainable=0。
- 主方案与pureFM均完成fresh100更新，各400条件/25,600queries；50/100完整checkpoint保留。用时4123.94/3185.59秒。全部任务、视频、动作query、RNG及权重实际匹配，见`runs/analysis/video_functional_20260911/first100_exposure_alignment.json`。
- 固定train24留出动作诊断32queries/task：主方案FM .154849→.116660，辅助reader仅至.153163；pureFM .154865→.114877。辅助教师尚弱；该动作证据不证明视频资格。
- 四个checkpoint的全部训练96/validation400 correct及other LoRA已物化；每个other面板完整hardlink复用，无重复编译。精确输入、运行命令及设备保留在`runs/outputs/video_functional_20260911/`各run/evaluation contract。
- 首段独立/data1预算787→预计827GiB/1TiB，之后物化前live quota为809.1GiB，共享空闲83TiB；新增训练前重新检查。基础模型、数据及normalization均复用canonical资产。

### 已完成闭环

| 配方/更新 | train correct /96 | train other /96 | validation correct /400 | validation other /400 |
|---|---:|---:|---:|---:|
| main/50 | 26 | 运行中 | 69 | 71 |
| main/100 | 35 | 运行中 | 72 | 运行中 |
| pure_fm/50 | 30 | 运行中 | 69 | 运行中 |
| pure_fm/100 | 41 | 运行中 | 56 | 运行中 |

- 配对source为train15/96、validation47/400；实际task/state/env/policy RNG检查通过。所有逐task/suite、breadth、R/G/L、churn、Jaccard和task-cluster bootstrap95%CI见`runs/analysis/video_functional_20260911/paired_summary.json`。
- 主方案validation50→100为69→72，S/O/G/L=4/48/16/1→0/55/11/6，breadth7→6；相邻保留50/新增22/丢19、churn41、J=.54945。pureFM69→56，3/53/11/2→0/43/3/10、breadth7→5；相邻34/22/35、churn57、J=.37363。主方案100比pureFM多16，但task-cluster95%差额CI[-.0075,.0925]仍含0，不能宣称辅助信用有效。
- 主方案50 correct/other=69/71，重合58、互换24、J=.70732；净差95%CI[-.03,.0125]。仅支持该节点同task换视频较稳健，不能替代内容/顺序必要性。
- train correct主方案26→35、pureFM30→41；pureFM训练获取更强而validation下降，不能按训练FM或训练成功数选择迁移方案。

### 正在运行与下一步

- gpu01:4/5/6各3workers分别运行pureFM50/main50/main100的train other；gpu02:1运行pureFM100 train other。gpu02:4/3/6分别运行pureFM50/main100/pureFM100 validation other，各3workers。均为long-first动态队列和持久workers；每次launch live检查两节点。
- 一次gpu02:0在空闲检查后遭遇其它作业进入，主方案100 validation加载OOM、0rows；完整失败证据保留于`runs/analysis/video_functional_20260911/failed_attempts/main100_validation_correct_gpu02p0`。已换gpu01:5,6完成正式400；未干预其它作业。
- 按原设计准备与主方案匹配的frame_set参照；目前主方案correct相邻轨迹较好，因此先以其配方检验有序处理的增量，不等于辅助优势已确立。配置唯一变化为`model.process_mode: ordered→frame_set`，保留完整帧、同参数、同FM/辅助/蒸馏和曝光。50/100请求及理由见`runs/analysis/video_functional_20260911/frame_set/registration.json`；尚未启动训练。
- 收齐换视频结果后结合必要无序参照决定接续。任何明显仍在获取的参照都需匹配充分曝光，不能靠弱训参照制造视频资格；不从两个早期节点宣称平台或整套失败。
- 当前goal未完成，未选定checkpoint；无序参照与最终sealed视频controls尚未执行，Test继续封存。

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
