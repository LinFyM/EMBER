# EMBER progress

## 当前实施：局部物理效果的因果可用性（2026-09-13）

当前active诊断为[操作语义§6](docs/operation_semantics_feasibility.md#6-从语义读出转向实际物理后果当前有界诊断)：
固定train24×demo16–19×三个时间位置，288条件，各17种预登记动作干预；CPU、无渲染／模型／梯度。
检验局部连续动作Jacobian对未拟合组合扰动的物理变化预测，不以语义辅助头、reward或success评分。
新增512MiB空间上限；不通过则停止该局部效果度量，不扫扰动或默认训世界模型；通过也需另登记有界Writer比较。
下文已关闭状态属于先前标签诊断／Writer；当前没有新Writer训练、selected checkpoint或最终controls。整体goal未完成。

## 当前补充实施已完成：操作语义标签与设计修正（2026-09-13）

Owner最新要求继续仔细推导、实施并按结果调整。[操作语义可行性](docs/operation_semantics_feasibility.md)已完成关闭：
train24×8episode、6,668个CPU状态位置，零梯度／GPU／环境步；源7bd8a6f5、84.34秒exit0，原始数组重算通过。
保留对象／部件及左右指接触，186/192episode有双指接触转换；抽屉8条中6条在采样位置无双指同时接触，
region所属body也不等于区域谓词。因此撤去“通用接触阶段→语义rank”的直接依据，不以数据覆盖默认启动Writer。
新增实际A监督的零B与共享A反例：准确接触响应仍可没有行为／视频贡献，须先明确可执行效果传递，不能重做无关辅助头。
一次性入口已退役，约1.36MiB原件留在runs/analysis/operation_semantics_20260913；无active design、无在途运行。
整体goal未完成；下一项需收敛真实RGB操作证据与唯一LoRA实际效果之间的可失败合同，保持旧失败边界及现行资格。

## 当前状态：Native双相机有界实验关闭，回到机制分析（2026-09-13）

Owner授权的自主goal保持：取得有益视频增量、跨视频／初始化／相邻保持及固定validation迁移，暂不要求145/400。
**整体goal未完成，当前无active design。[跨初态相对几何检索](docs/cross_init_relation_retrieval_audit.md)已完成关闭。**
[Native Dual-View Writer](docs/native_dual_video_writer_design.md)已关闭。时间对齐、冻结正例复核及
[冻结局部动作生成诊断](docs/frozen_local_action_decode_audit.md)均已按完整证据关闭。
新增[冻结source状态输入诊断](docs/source_state_input_audit.md)也已完成关闭；不恢复历史训练、不扩大旧闭环复核或运行最终controls。
下一项实验须先有相对已失败近邻的实质机制区别、可区分竞争解释的预测和停止条件。

已完成[Meta动作校准提案复核](docs/video_information_identifiability.md#8-为什么当前不单独追加meta动作校准诊断)：
原生头误差变化不能单独区分完整H知识丢失与下游读取／传递不足，旧v4也出现校准改善而闭环未修复。
不追加仅测Meta动作MSE后冻结Meta／增加校准头的诊断链；没有新增forward或训练。
当前转向明确可由合法数据辨别的跨初态操作关系及行为预测，避免重做旧功能读出、蒸馏或native-span方案。

新增[跨初态关系复核§9](docs/video_information_identifiability.md#9-跨初态操作关系新增监督必须区别于旧状态条件化与任务记忆)：
旧A/B已具有执行状态依赖，旧局部头也已反演视频中发生的动作；这两项改名不能支持新训练。
四个train任务demo16各恢复首／中／末三状态，共12个CPU forward完成，证明位姿标签来源可用，
并暴露实例绑定及关节部件不能由统一根部位移替代。没有渲染、模型forward、标签库或新学习。
同task oracle关系图可退化为任务索引，不能据其拟合决定换RGB读取器；后续判别须处理任务层面泛化。
已向Owner提出独立frame_set是否为持续硬门槛的明确选择，尚未据此变更资格。整体goal未完成。

### 跨初态相对几何检索：13,064位置完整，替换前提未通过

先不训练关系到参数映射：固定train24、teacher参照action16–19、query动作诊断42–45，比较同一条演示的
绝对几何／相对几何1-NN及不检索的video动作均值。每条query保持stride5，预测真实后续5×7动作chunk；
全部384个episode对、13,064位置完整。对象／部件固定来自官方obj_of_interest，含body与region site，未按动作选物体。
绝对／相对／均值MSE=.12443553/.12634790/.25459689；绝对−相对CI[−.00421347,+.00027549]跨零，
四suite均无相对改善，按登记关闭。两种匹配均在24/24task胜过均值，但旋转分项未优于均值，不是完整教师的证明。
这是privileged CPU数据参照，不是部署Writer、闭环或最终controls，也不恢复95-task或任一旧训练。
9f90a14d冻结执行exit0、99.37秒，6,513个几何状态恢复及raw rows加权重算通过；原始动作／预测／索引及全部结果保留。
评分前发现并修正2ms运动学缓存对应，原失败证据保留，生产Writer与高层动作offset未改变。一次性入口退役。
停止直接以相对中心化替换作为修复依据，不扫描检索变体或默认追加完整Writer；完整边界见findings§81。

最新[运动对应机制审查](docs/video_information_identifiability.md#7-显式运动对应能补什么以及为什么尚不足以启动新writer)
不采用直接追加／替换dense-flow编码器的提案：文献的物体运动收益依赖物体绑定与执行状态对应，单独跟踪质量
不能检验跨初态的LoRA行为传递。保留目标物体运动作为候选，但尚无能区分获取与编译不足的具体判别合同；
运动对应审查本身没有新权重下载、模型forward或实验启动。完整依据见findings§76。

### Native双相机修正：8面板完成，按预登记资格关闭

两臂各fresh200，100/200四checkpoint、8个sealed bank及1,984条闭环记录完整；运行代码defcf734。
新旧四模型800条件的18个采样字段匹配，8面板及相机参照的task/state/language、teacher真实帧索引、
环境／policy RNG配对核验通过。validation各task50条teacher整轮各一次；train96为46–49/states32–35有限池。
全部worker正常退出，累计评测墙钟4417.77秒，controller正常exit0，无在途训练／评测。

| 节点 | train ordered / frame_set | validation ordered / frame_set | validation差额95%CI |
| --- | --- | --- | --- |
| 100 | 36/96 / 32/96 | 64/400 / 56/400 | [0,+4.5]pp |
| 200 | 54/96 / 50/96 | 53/400 / 39/400 | [−.5,+9.25]pp |

100节点S/O/G/L为1/52/8/3对0/48/6/2、breadth6/4；200为0/52/0/1对1/37/0/1、breadth3/4。
200仅Object净正，有序53次成功中52次来自Object；相邻Spatial／Goal归零。
有序相对无序100保留51／新增13／丢失5、churn18、J=.73913；200为35／18／4、churn22、J=.61404。
相邻有序64→53，保留41／新增12／丢失23、churn35、J=.53947；无序56→39为32／7／24、churn31、J=.50794。
两个节点的有序净额均正，但CI下界均未严格>0，200也未达至少两个suite净正；不满足登记资格和稳定保持。

预登记相机参照中，ordered validation单→双为61→64、72→53；frame_set为65→56、81→39。
200同模式差额CI分别[−9.5,−.75]pp、[−20.25,−2]pp；相机×顺序交互虽为正，伴随明显绝对性能损失。
不能把相对无序少退化当作足够的方法收益。冻结source仍为47/400；有序总数高于source但配对task区间跨零。

**当前无active design，无selected checkpoint。** 关闭本有界组合，不追加训练、节点、seed、LR、rank、flow或Meta冻结扫描；
不触发same-task-other／强静态／最终controls、Test或RL。原生双相机动作读出的正证据仍成立，
它未转化为本共享Meta／表示／Compiler的稳定闭环收益，不能唯一归责某个接口或否定全部视频方法。
下一步先综合局部功能诊断、既有共享／蒸馏负结果及当前训练—迁移分离，明确能区分获取与传递不足的预测；
在实质机制与合法判别证据成立前不启动新Writer。完整边界见findings§79。

原件：`runs/analysis/native_dual_video_20260913/`的`paired_summary.json`、`camera_comparison.json`、
`bounded_200_decision.json`、`evidence_audit.json`、`training_pairing.json`和`OVERNIGHT_READOUT.md`。
全部训练／物化／评测contract、四完整checkpoint、8个bank、raw rows和completion保留；整体goal未完成。

### 原生端点读出：已完成，保留视觉范围限制

[端点诊断](docs/source_endpoint_readout_audit.md)新增三单元各384位置完整exit0，复用既有dual/full10。
agentview full10/t1均值/t1固定probe为.36548/.34789/.34732，均差于task mean .25060；
dual对应.13673/.13217/.13177，均胜过task mean。三个同读出相机差额均24/24task为正、区间严格为正。
触发视觉范围分支；同视角增加flow步数没有改善，停止以denoise深度为当前首要修复依据。

双相机的单步H有原生可读动作价值，不能转写成当前agentview或学习后Meta／E已充分；
也未区分腕部信息与source输入分布匹配，不能由动作预测直接声称视频过程／LoRA收益。
已核对双视角实现与V-JEPA单视角合同；1,669份范围内库存中38份可判定observer均为agentview。
上述证据已用于登记native双相机与原单视角视频先验的最小输入修正及可失败预测；
不将双视频K结果混为双相机实验，不恢复已关闭checkpoint训练。原始endpoint_*证据完整保留；代码e1a06b46。

### 待Owner明确的后续方法选择口径

当前goal要求可重复有益视频增量及最终内容／顺序因果确认；已完成实验另预注册了独立训练ordered相对
全帧frame_set的优势。后者不是固定模型的顺序干预，也不是单帧或语言参照。后续是否把“胜过独立训练的
全帧无序方法”设为持续硬门槛，需要Owner明确；这会改变候选与比较设计，不据已有分数自行放宽或加码。
在澄清前不重判任何已关闭实验，不选checkpoint或提前运行最终controls。目标及其它科学合同保持。
本次同时核对findings§68的73/75任务原件索引与现有sampler；没有新增支持恢复95-task的证据，不重复该审计。
该疑问不阻塞与它独立的机制诊断；已完成的source输入诊断沿用全部现行科学边界，不据未回答的问题放宽方法资格。

### 冻结source输入诊断：原生state-free采样有能力，状态补全前提未通过

固定train24×16位置，三臂各384位置、8个配对噪声、10步完整采样，均exit0；没有Writer／Meta／LoRA／梯度或rollout。
生成均值MSE：state-free .13672735、mean-state .14173378、true-state .12469236、action-mean .25059744。
free−true的task-cluster95%CI[−.00247592,+.02751481]跨零；mean-state−true区间[+.00965492,+.02384471]为正，
action-mean−true亦为正。三项要求未同时通过，停止直接新增状态补全模块的依据，不追加采样／拟合。

描述性追踪中action-mean−state-free为+.11387009、区间[+.08481614,+.13969462]，22/24task为正。
保留这一原生动作生成正证据，但它不证明单步H／Meta／E已充分，也不证明视频动态的因果收益或新初始化闭环。
后续端点×视角比较已区分这个混杂，完整结果见上节；不默认扫描flow阶段或恢复旧forecast。
代码冻结6fa6177f，每臂约311.18秒、9.88GiB峰值，全部原件位于`runs/analysis/source_state_input_20260913/`。

### 冻结局部动作生成：完整768片段，未通过

固定旧local两臂step200、train24×原留出16片段，各从8个独立噪声经10步生成。两臂exit0，每臂约98.3秒，
没有梯度、LoRA生成或环境rollout。动作均值MSE有序.31625053、无序.31676502、task mean .25059744。
有序相对无序小幅更好（95%差额CI[+.00032845,+.00070563]），但24/24task均差于task mean；
后者−有序差额CI[−.08381484,−.05098097]。预登记两项参照未同时通过，停止现成局部动作中间量提案。

解析task Gaussian不看视频、使用带噪动作，其FM=.26389586，24/24任务优于旧头.81011446/.81128639。
这解释了为什么局部小幅有序差额不足以证明功能教师已充分；不否定全部过程知识，也不把解析task身份用于部署。
原件为`runs/analysis/local_action_grounded_20260912/inverse_action_READOUT.md`及同前缀raw/summary/launch。
诊断入口由clean pushed 017430b3冻结，模型仍原5f4f440c；active脚本已退役，冻结源码与原始证据保留。

### 理论与只读审计补充

[视频信息与可识别性复核](docs/video_information_identifiability.md)已整理当前竞争解释及停止分支：
frame_set保留整条视频的全部采样画面，是无显式顺序的多状态参照；它与ordered打平不能证明没有使用视频。
train24的19个单原子／5个合取目标均检查当前状态，物理前置条件仍可能使时序有用。
原贡献+12/96的9个task在新交叉面板上合计−1/288（4正／1零／4负），旧优势也未在该task群中保持；
这项事后描述不筛选任务、不增加资格检验。原件在既有复核analysis根的`conditional_information_audit.py/json`。

采样合同的风险分解说明：已知task上的跨episode FM不强制依赖teacher的个体差异；这不等于未见task的执行知识
可以从语言免费获得，也不是视频无用定理。新完成诊断进一步否决了把现成局部读出视为可直接使用动作教师的前提；
下一机制仍须明确视频补充的知识和可失败预测，不默认已有好过程只差编译或保持。

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
