# EMBER progress

## 当前：固定A误差分解完成，转入共享纠正获取分析（2026-09-13）

Owner完整有益视频特异性goal保持active且未完成，暂不强制145/400。
**[原生纠正获取诊断](docs/native_correction_acquisition_audit.md#6-完整结果与关闭裁决)已完整完成并关闭；
当前无active design、在途GPU／CPU运行或selected checkpoint。** Native Correction Writer及全部旧组合保持关闭。

8c714e91 clean pushed detached固定ordered/frame_set的100/200四checkpoint与train24/demo16–19，
完成384套合法单视频LoRA，四worker均exit0、501–505秒。CPU分解26.286秒exit0，全部同视频label、
真实帧／末帧、76因子、单次Writer／checkpoint身份、四sealed bank及误差正交分解通过。
无新动作读取、参数更新、query、环境步、validation/Test或checkpoint选择。

| 模型 | 实际相对误差E | 固定A下界F | 空间内差额D | F−D 95%CI |
| --- | --- | --- | --- | --- |
| ordered100 | .951379 | .313762 | .637617 | [−.371229,−.277388] |
| frame_set100 | .950511 | .314918 | .635593 | [−.366038,−.275823] |
| ordered200 | .900546 | .270922 | .629623 | [−.404789,−.314671] |
| frame_set200 | .900279 | .265043 | .635235 | [−.411855,−.330517] |

四个task-cluster区间均严格负，各模型24/24task均值D>F；各suite与四个teacher序号汇总亦如此。
66.9%–70.6%的实际参数误差在当前A空间内，按登记第二分支停止把扩A空间当必要的首项修正。
固定A下界仍约.27–.31，且允许任意条件专属B及任意幅度；不能据其认定A充分、有限B易学、RGB充分或B是唯一根因。

100/200实际已见教学条件为49/67条；200已见条件的空间内差额仍为有序.628765、无序.636050。
不是仅未见视频的问题；该分组不是随机因果比较。相邻有序总误差下降.05083中F下降.04284，D仅下降.00799；
无序对应.05023/.04987/.00036。保留A覆盖变化与空间内获取有限的事实，不把它变成续训、尺度或rank扫描的依据。
本96标签87.6726%能量在action-out，真实A有弱奇异方向；这是原参数度量的定位，不替代视频必要性或闭环价值。

下一步先综合当前可表达方向、有限共享纠正预测、原生坐标可用性和学习信用的竞争解释，
只有可失败论证和有信息量的最小干预成立后再登记；不从“B侧差额大”直接启动另一个Writer或辅助动作头。
原件在`runs/analysis/native_correction_writer_20260913/acquisition_audit/`：完整readout、summary／decision、
四decomposition与bank、registration／launch／requests／logs／exit全部保留。跨轮结论见findings§92；旧闭环裁决如下。

## 最近：原生纠正Writer有界闭环完成并关闭（2026-09-13）

Owner的完整有益视频特异性goal保持active且未完成，当前暂不强制145/400。
**[Native Correction Writer](docs/native_correction_writer_design.md#10-完整有界结果与关闭裁决)已按原资格完成关闭；
该轮结束时无active design、selected checkpoint或在途训练／评测。** 旧Writer、冻结正例复核及原生纠正oracle诊断继续保持关闭。

09c1a0d6 clean pushed frozen完成两臂各fresh200、800教学条件与51,200主query；四完整checkpoint、
8个sealed bank和1,984条配对闭环记录完整。训练18个曝光字段及0/100/200独立诊断配对通过，teacher逐条件排除于主query。
全部阶段与96个评测worker均exit0，真实stride5帧／末帧、50视频无放回、RNG、单次Writer与76因子／checkpoint身份、
raw和aggregate审计通过；累计评测launcher墙钟4,051.92秒，controller已正常退出，无本研究在途Python进程。

| 节点 | train ordered / frame_set | validation ordered / frame_set | validation O−F 95%CI |
| --- | --- | --- | --- |
| 100 | 15/96 / 15/96 | 50/400 / 51/400 | [−.75,0]pp |
| 200 | 19/96 / 20/96 | 50/400 / 49/400 | [−.75,+1.5]pp |

有序净增−1→+1，两个区间下界均未严格正，净正suite数0/1，未满足相邻同向和至少两个suite净正。
按原设计§6关闭本共享获取／原生因子组合，不追加训练或rank／λ／seed／LR／scale扫描，不补未获资格的
other／强静态／跨初始化或最终内容／shuffled/reversed controls、Test、RL；整体goal不因此完成。

validation有序两个节点S/O/G/L均为0/6/41/3，frame_set为0/6/41/4及0/6/42/1，breadth均3/8。
source47/400为0/5/41/1；有序两个节点相对source均R/G/L=45/5/2、churn7、J=.86538，净率区间[0,+1.75]pp。
保留名义+3及Goal总数保持，不能据此宣称统计明确的source收益或有益视频特异性。
有序对frame_set的100／200分别R/G/L=45/5/6与46/4/3、churn11/7、J=.80357/.86792。
相邻有序50→50为46/4/4、churn8、J=.85185；frame_set51→49为45/4/6、churn10、J=.81818。
较低churn成立，成功集合仍有变化，Spatial仍零成功，共五个validation task零成功。

训练侧frame_set15→20保留15、新增5、丢失0，净率区间[+1.042,+9.375]pp；相对source15亦有正区间。
有序15→19的相邻区间跨零。两臂train breadth均7/24，不能把本轮写成完全没有学习。
固定独立动作FM初始.15328534，100有序／无序.15307564/.15304530，200为.15281900/.15280163；
末25条件L_update约.90642/.90657，空间信用亦有拟合。它们不能代替真实行为或唯一定位失败接口。
完整624标签的共同成分与样本内常量参照继续作为限定统计，见findings§90；真实纠正oracle的功能前提保留，见§88。

本轮有序200的Goal41优于上一空间监督组合的9，Object6则低于59、总数50低于71；
教学池16–41与旧0–15不同，出口和纠正监督共同改变，不宣称匹配曝光的单变量消融。
下一步先区分有限R/A空间、B的纠正预测／学习信用、读取与视频信息限制，形成有停止条件的最小冻结诊断；
不由本轮non-pass自动启动另一完整Writer，也不从参数存在性或共同标签统计推断RGB获取已解决。

原件在`runs/analysis/native_correction_writer_20260913/`（物理根`/data0/user/ymdai/ember_runs/native_correction_writer_20260913`）：
完整`OVERNIGHT_READOUT.md`、`paired_summary.json`、`bounded_200_decision.json`、自动初步decision、
`evidence_audit.json`、`training_pairing.json`、`training_readout.json`与`launch_contract.json`均保留。
训练、四checkpoint、8个bank与8个原始评测目录在同物理根`training/`；相关标签、profile及只读统计原件保留。
跨轮结论见findings§91，实施与数值准入历史见设计§1–9。以下历史段落不恢复执行。

## 当前：原生纠正跨episode传递前提通过，转入合法生成推导（2026-09-13）

Owner授权继续仔细推导、实施并按结果调整，整体有益视频特异性goal保持active且未完成。
**[原生条件与真实纠正的跨episode传递](docs/native_corrective_transfer_audit.md)已完成关闭，两个oracle均通过其登记前提。**
没有active design、Writer训练、在途GPU运行或selected checkpoint；Visible-Object及更早组合全部保持关闭。

f39d594f clean pushed frozen完成train24/demo16–19、192套完整38-target rank16 LoRA，
在42–45的384个独立query上构成3,072个配对组合，无query梯度或环境步；三个worker与formal均exit0。
worker墙钟110.63／113.15／113.28秒、峰值11.014GiB。真实动作只用于训练侧oracle构造与独立预测后评分。

| Teacher合同 | 读出 | source→oracle MSE | 改善95%CI | 正向task |
| --- | --- | --- | --- | --- |
| state-free | t1 | .11977705→.11351420 | [.00251749,.01101049] | 17/24 |
| state-free | full10 | .16493043→.15574849 | [.00322478,.01688248] | 21/24 |
| true-state oracle | t1 | .11977705→.11022396 | [.00485088,.01539874] | 21/24 |
| true-state oracle | full10 | .16493043→.15252233 | [.00534316,.02148158] | 22/24 |

两个oracle的两读出均四suite净正，四个teacher序号汇总均正；state-free full10为83/96个task-video条件改善，
task2/34/37汇总非正，不筛除。该正事实支持固定原生条件与真纠正的一次参数作用具有跨episode功能前提；
尚不证明有序RGB获取、共同偏置之外的视频必要性、闭环或validation迁移，也不恢复task-local优化。

历史近邻复核发现旧95-task authority的.716/.801、.2695是梯度因子cosine，而非安装更新后的FM；
J2真实FM正控、EBSRI及PNBTT则是不同的优化／共享生成合同，不能拼成当前固定算子的行为验证。
原始MSE、teacher幅度／线性读出、配对与76张量shape审计通过；从原HDF补核768个teacher标签位置、
384个query标签、冻结quantile及构造位置均通过。完整逐task／suite、四teacher及负条件保存在
`runs/analysis/native_corrective_transfer_20260913/TRANSFER_READOUT.md`、`paired_summary.json`、
`bounded_decision.json`、两个audit与formal原件。正式准入data1为1016.9/1024GiB、新增总预算2GiB。

按正分支继续推导合法RGB纠正获取与纯前向参数生成；须保留同帧条件和纠正的联合关系，不能把普通自由B head改名
当作新机制，也不能用推理时loss/VJP/参数更新绕过部署边界。本次只关闭有界诊断；整体有益视频特异性goal未完成。

## 最近完整闭环：物体与运动落点监督关闭（2026-09-13）

Owner授权继续仔细推导、实施并按结果调整，完整有益视频特异性goal仍未完成。
本组合无active design、无在途训练／物化／评测、无selected checkpoint。下方各历史阶段的“当前／下一步”
均不恢复执行；最新执行状态为顶部。[Visible-Object设计§9](docs/visible_object_grounded_writer_design.md#9-完整有界结果与关闭裁决)
与findings§85保存完整正负事实，旧时间对齐、冻结正例复核、局部动作、native双相机及其它已关闭诊断不重开。

b304cde6 clean pushed frozen完成ordered/frame_set各fresh200、800条件、51,200主FM queries，
两臂墙钟6514.54／6515.52秒，四完整100/200 checkpoint和八sealed LoRA bank保留。
8个correct面板、1,984条raw rows及对应编译记录全审计通过，全部worker／阶段exit0，队列exit0；
累计评测墙钟4775.89秒。训练、物化和评测均已按有界登记完成，未追加节点、训练或controls。

| 节点 | train ordered / frame_set | validation ordered / frame_set | validation净率95%CI |
| --- | --- | --- | --- |
| 100 | 32/96 / 33/96 | 20/400 / 24/400 | [−3,+1]pp |
| 200 | 50/96 / 50/96 | 71/400 / 70/400 | [−2,+2.25]pp |

validation100 S/O/G/L为0/16/3/1对1/19/4/0，breadth5/4；200为0/59/9/3对0/61/7/2，breadth6/5。
有序对无序R/G/L在100为12/8/12、churn20、J=.375，200为62/9/8、churn17、J=.78481。
相邻有序20→71保留19／新增52／丢失1，churn53、J=.26389；无序24→70为18/52/6、churn58、J=.23684。
相邻有序95%旧成功保持，低J主要来自新增；不能把它写成后段明显退化。但两节点CI下界不严格>0、
有序净差−4→+1未相邻同向，100也未达至少两个suite净正；按登记基础资格关闭，未放宽frame_set口径。

**保留新增正证据：**200节点旧同模式纯FM→空间监督为有序53→71、无序39→70，
CI分别[+1.5,+8.5]pp、[+1,+16.75]pp；100节点则64→20、56→24。空间信用有后段绝对收益，不能概括为无效。
监督×顺序交互200为−3.25pp、CI[−11,+2]pp，未支持有序结构更好消费该信用；一般正则／共享学习与视频必要性未分开。
frame_set拥有全部真实frames，独立训练差额不是固定模型的顺序干预，也不是单图对照。

source47/400的S/O/G/L为0/5/41/1；有序200为0/59/9/3，source→有序R/G/L=12/59/35、churn94、J=.11321。
Goal41→9丢失34次旧成功，Object5→59保留5并新增54。更高总数不等于广泛保持source；
source净率区间[−20.25,+31.75]pp属于解释限制，不是补加资格门槛。当前仍不能唯一定位Meta、Compiler或视频Value根因。

证据边界：新旧四模型全部800条件的18个曝光字段、0/100/200留出动作诊断字段匹配；raw/aggregate、
真实teacher stride5帧和末帧、环境／policy RNG、checkpoint/commit、38-target rank16 A/B及single invocation核验通过。
validation同task50teacher整轮各一次，train96只复用46–49/states32–35有限池。主FM200=.107831/.107719，
实际Q/K空间读取拟合成立，但都只是定位证据。384条CPU标签、128项原有检查与最长视频profile在设计§7–8及原件留存。

正式原件：`runs/analysis/visible_object_grounding_20260913/`中的`bounded_200_decision.json`、`OVERNIGHT_READOUT.md`、
`paired_summary.json`、`supervision_comparison.json`、`evidence_audit.py/json`、`training_readout.py/json`、
`training_pairing.json`、launch/run/completion与queue状态；checkpoint和bank保留在对应runs/outputs根。
最后评测准入strg01/data1为1016.9/1024GiB、剩余预算2GiB，完整阶段29GiB预算已登记；后续若有新增长须重新准入。

按停止分支不追加本组合训练／节点／seed／LR／λ／rank／层位／标签定义，不触发other、frame_set_image、
额外初始化资格或最终内容／shuffled/reversed controls；没有Test、held梯度、RL或模型融合。
**综合理论判断已补充：**[可识别性§10–12](docs/video_information_identifiability.md)与findings§86明确，
当前空间loss对query角色分配的置换不敏感；边际拟合不能证明操作关系充分。裸source自蒸馏又有identity零损失解，
单纯复制其预测没有新增纠正信息。v4的预测差Value与Local的独立反演头已按实际源码分开，不能换名称重做。
固定source prior＋真实转移纠正在这些近邻中有区别，但还缺另一初态下的策略作用合同；δa=0也不等于视频没有价值。
因此不先训练新动作头、不新建cache。下一步先推导状态／部件条件与效果的联合关系怎样进入实际策略函数，
审查task索引、native span及旧辅助头反例；满足可失败、能改变决策的条件后才登记新实验。整体goal保持active。
本轮task-owned frozen worktree及重复profile／queue脚本已退役，精确源在Git及formal artifacts保留；见closure_lifecycle.json。

## 当前物理效果诊断已关闭：部分可预测性成立，替代度量未获资格（2026-09-13）

[操作语义§6–7](docs/operation_semantics_feasibility.md)完成固定train24×demo16–19×3位置的288条件，
4,896条短段动作干预／24,480高层步，CPU、无模型／LoRA／梯度。fcd9a613冻结434.79秒exit0、raw重算通过。
零变化／局部预测MSE=.00089956/.00026749，差额CI[.00048754,.00076067]、23/24task及四suite正；
但预测误差比例.29736不满足预登记<.25（RMS=.54531），因此关闭该J加权FM提案，不实施条件性§6.5。
Object比例.01887，Goal／Long为.41550/.51493；保留部分物理可预测性的正事实，不挑suite或扫描扰动挽救。
夹爪离散分支存在真实效果差，但幅度与连续扰动不同，不能推断夹爪是主因或默认重加权。
原件留在runs/analysis/operation_semantics_20260913/effect_replay，CLI退役。无active design／运行／selected checkpoint，整体goal未完成。

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
