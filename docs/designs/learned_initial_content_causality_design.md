# 学习后的首帧内容参照：逐帧证据是否提供额外能力

状态：2026-09-26登记实施与有界执行；是否active以progress为准。
机器范围：`configs/learned_initial_content_causality_v1/experiment_spec.json`。
主讨论`01a0cd94-65da-7b22-8ca9-7ba35f454632`；执行者为现有Sol
`01a0cd90-ebb7-77a1-a20b-a858825d2f66`。这是一项学习参照，不是新部署方法或已验证修复。

## 1. 为什么这一项能改变判断

Owner重申的最终目标是找到视频知识的有效利用方法，并使EMBER绝对闭环能力超过强MT-BC。
本批只提供一个方法设计所需的学习反事实；C0胜S0不构成目标实现，S0追平也不意味着视频没有潜力。
结果应约束特征提取/学习/参数编译的具体机制，不能将继续追逐完整视频对删减输入的差额变成研究目的。
这项澄清不改变已登记训练、数据、矩阵、资源或执行安排。

固定C0的完整400为correct120、other132、语言121，不能继续把旧视频巨大劣势当稳定前提。
随后E/H冻结四格为20/24/20/27，各80；完整相对首帧有9得2失，主要比较描述区间[2.5,16.25]pp。
这证明了给定模型/面板中首帧以外输入的有益作用，尚不证明模型必须靠这些信息才能学到该能力。
源于后续内容的有用知识、对训练输入分布的依赖、以及两者共存，仍能解释冻结损伤。

本批只补一个缺失的学习反事实：同一模型从fresh状态学习首帧内容，能否在相同监督预算内消除上述缺口。
若可以，就不能再以冻结重复首帧的低分选择“加强动态路径”的架构；若完整视频仍占优，逐帧内容的学习价值
才得到比损伤消融更强的支持。两种结果都改变后续设计依据，因此不为实验不断运行而额外铺矩阵。

历史近邻已完整核对：
- `learned_frameset_reference_design.md`保留全部逐帧E/H，只改时间算子；900/1200的set为142/118，ordered140/135。
- `pretrained_video_grounded_writer_design.md`的static对每个真实I_t重复四张，仍读取所有t；它不是只看初始场景。
- findings§128首帧广播及末P屏蔽是冻结诊断和18步父模型续训，L18/J18 correct均15/32；不能重新叫做fresh学习参照。
- 旧A正确P在固定错误Core下38→56/96说明可消费视频差异存在；没有同时隔离初始内容、顺序与学习补偿。
不重跑这些历史试验；不把本批改称它们从未研究过时序。

## 2. 特征层变量、导数与理论边界

令原生读取为q(L)、E_t(L,I_t)和H_t(L,I_t,公共噪声)，后者保留50×1024完整action-horizon。
E是图文task token与任务查询读取的patch内容；H是原生动作计算响应，不是真实teacher动作。
原完整模型为`G_phi(q,E_0:T,H_0:T,tau_0:T)`，G包括现有Core、Procedure、Compiler和全部FactorHeads。
新S0只改**原生读取送入Core/Procedure的内容**：

```text
E^S_t = E_0
H^S_t = H_0
I^S_t = W_I mean_horizon(H_0)
theta_S = G_phi(q, E^S, H^S, original_positions_and_masks)
```

每条video独立广播，不平均帧、特征、视频或LoRA。q、原始frame count、stride5+真实末帧位置、mask和边界全部保留。
S0仍能利用视频长度/原时钟，因此准确名称是“首帧内容＋原时钟的学习参照”，不是纯图像模型或no-video。
Core/Procedure的时间计算与末帧边界继续存在；不修旧边界、不删Procedure、不新增a*q或别的参数。

对每个标量特征通道，时间投影为`P0=1 e_0^T`、秩1；所有非首帧内容方向位于其零空间。
反传必须满足`dL/dE_0=sum_t dL/dE^S_t`（H同理，并包括I路径），而不是detach、仅取一个时间项，
或人为除以帧数。所有原Writer/三Meta仍学习；Source保持冻结。这个算子让整个模型有机会重新分配信用，
其意义与冻结phi后直接删输入不同。

可以只对真实首帧运行原生encoder，再用原有时间metadata扩展，以省去确定不用的后续原生读取。
必须核对这与“所有帧原生读取后取首帧广播”的函数/梯度语义一致，接受正常BF16/分块差异；
不得保留真实后续E/H/I作为隐含旁路，不得删除full-H位置，不缓存跨更新的Meta激活。
原生首帧使用真实图文prefix和原公共probe，禁止fake image/action query。

当前FrameRead可写为`W_O[mean(V)+sum_t(w_t-1/T)V_t]`。
已保存BF16权重的CPU核算还须将第二项拆成`(sum w-1)mean(V)+sum w(V-mean(V))`，
避免把有限精度的权重和余项当成内容选择。C0完整输入下均值RMS1.33049、中心内容选择RMS.00146468，
只是同一读出空间的数值构成，不证明小项无功能作用，更不能据此增大gate或改dtype。
E同时进入Procedure；因此本批不预设全部视频作用来自Core均值，也不把E/H混合格推成单模块归因。

`video_information_identifiability.md`已推导理想独立跨episode分布下允许同task恒定LoRA的最优解，
但当前有限池排除teacher不严格独立；且可表示、能学到、能迁移均需另证。唯一语言不等于模型已获得执行知识。
本批只检验匹配有限学习窗口中的额外内容价值，不能证明总体信息不存在、充分收敛或唯一根因。

## 3. 唯一新学习臂与冻结参照

- S0：从fresh初始化训练630宏步；基础配置为`configs/language_content_path_causality_v1/train_C0.json`。
  唯一主要变化为上述原生内容投影，模型参数集合、rank16/38targets、初始化、全部loss/权重和学习率不变。
- C0：只读复用F=dca1b550的原C0@630；不重新训练、不选择别的节点。
  使用原language-content study的correct80/other80/seen64三份bank，不从本次27/80重新物化bank择优。

fit28/teacher和action demos0..45/保留46..49、teacher-query跨episode、各随机流、事件计划容量1260及其前630，
均与原C0完全相同：2520 task events、70560真实queries，每task90次事件、2520queries。
world2，同逻辑任务/查询/噪声流；每task权重、main21+extra7跨episode/full-H50 FM不变。
AdamW及scheduler全部继承原配置数值但状态fresh；不继承C0训练权重/Adam，不是公共底座课程。
正式checkpoint每105保存完整状态，仅评630；固定630是匹配预算，不称训练已经充分收敛。

配置必须明确标注diagnostic initial-content路径，canonical默认完整视频。复用owner模块，不另造Writer。
隔离开发、CPU及有界smoke通过后集成push，正式新训练/新bank/新闭环统一clean detached实现E。
工程验收包括：投影及sum梯度、padding、Source冻结/有效Meta梯度、world2四宏步与2→4恢复、最长原时钟反传，
至多2次无环境10-flow接口（fit28条件）；所有工程GPU消耗计入预算。不得另跑未计量的smoke闭环。
正式四条pilot承担环境/模型/采集接口验证，并计入总评测，技术通过不看成功分数放行。

## 4. 六个面板，448条闭环

对C0与S0各运行：held8×states0..9的correct80、same-task-other80，以及seen16×states0..3的correct64。
held8=0,1,14,15,20,21,36,38；seen16及teacher46/47/48/49映射继承原language-content合同。
held使用原50视频池seed20260911前10映射，other offset17，不能另建10视频池或挑视频。
六面板全部新闭环、同一E；C0银行F只读引用224条件，S0新物化224条件，不新增C0 Writer forward。
Source/B只引用已验收fixed400历史参照，不能放进本批新的配对差或称方法超过语言。
S0物化的同一次forward保留q、首帧E/H/I、真实位置/mask、Core/P及compiler slots/调制的只读trace，
不为记录额外forward；E/H只保存首帧及广播定义，无需存重复T份。C0旧特征只能按原来源作结构参照，
不得把不同物化的特征悄悄绑定到复用的F银行，或用跨模型特征范数差直接归因。

固定pilot为两模型×task0/state0、task36/state0共4，计入held correct。
全部448条保存实际执行T条action及T+1 body/EEF/夹爪/BDDL谓词；
full固定两模型×两种held视频×task0/14/21/38/state0=16，另seen task2/17/22/32/state0×两模型=8，共24双相机。
首轮真实10-flow的full50×7和前5从原rollout提取，无新增独立function query。
报告normalized action、环境命令和controller clip/scale后的6维OSC（夹爪另列），不能把环境命令误标为OSC。
在原pilot已创建的环境读取实际controller输入/输出范围并记录，不新增环境probe或episode。
跨分支只在相同首态比较函数；后续轨迹不称同观测函数差。

主比较是held correct的C0−S0；other同方向和能力保持作必要佐证，seen反映学习窗口是否出现明显拟合差异。
共同task-cluster bootstrap20000/seed20260926，8个held task和16个seen task分别抽样；correct/other共享held draws。
报告全部任务/suite/breadth、R/G/L/churn/Jaccard和每任务两种正确视频差额，不只报合并净分。
相邻checkpoint未评，无邻点稳定性声明。不得按本screen选择部署模型或从两个video条件中挑高者。

## 5. 预测与停止分支

1. S0在seen和held均接近/超过C0且两种正确视频一致：削弱“冻结损伤足以说明逐帧内容具有额外学习价值”。
   重点回到教学内容的监督辨识/泛化；不增大动态分支、gate、rank以修一个未经证明的信息瓶颈。
2. C0在held两种正确视频稳定占优，而seen没有S0整体学习失败：支持这个预算内首帧以外内容提供可迁移能力。
   仍不能区分场景覆盖与操作关系，也不证明超过语言或MT-BC；后续须解释哪些视频知识能够改善强MT-BC的执行不足，
   据此设计提取、学习及LoRA写入机制，而不是把超过语言/首帧作为新的终点，不转去通用VLA修补。
3. S0在seen也明显较差，或两种正确视频方向相反/区间宽：解释为有限学习差异或未识别，不能宣布信息必要。
   不自动延长训练、扫LR/seed或补400；主讨论先核对特征图、实际学习和原始行为再裁决是否值得追加。

本批完成即停止新增实验并主动Queue主讨论；主讨论核验后继续科学工作。既有长期授权足够，不等Owner再次批准。
本实验不改canonical部署路线，不声称发现统一根因，也不把learned静态参照当EMBER最终方法。
任何后继利用方案都须给出特征含义及其证据、跨初态适用条件、实际参数/动作作用与可否证预测；
最终以相对强MT-BC的绝对能力、正确视频有益贡献和能力保持共同验收，不能只扩大本批两臂间隔。

## 6. 资源、来源与有界异常处理

根`/data0/user/ymdai/ember_runs/learned_initial_content_causality_20260926`。
预算9完整GPU-hours，含所有工程、失败启动、训练、物化及448评测；预计比原C0单臂5.66 GPUh训练更省原生读取，
但不得假设实际提速。若profile显示完整固定矩阵预计越界，停止新增阶段、报告实测与剩余量，不自行缩科学参数。
新data0峰值≤10GiB，开发+formal≤768MiB；保留六个完整checkpoint、224bank、trace/full/CPU分析及必要临时载荷。
执行者在建root/启动前查strg01独立quota与共享容量、估峰值，复用Source/数据/原C0银行。
训练world2单节点；其它独立阶段可成本均衡并行，全项目本批最多6物理卡，每launch核对双节点live所有权/余量。

修正上批三面板外层误记节点的执行入口：实际host、选中node/index/UUID、worker看到的UUID必须在启动前绑定一致，
不能SSH核对gpu01却在gpu02本地tmux启动。使用现有launcher，不额外搭调度系统。
旧批有效原件不改、不重跑；其单位纠正/特征权重和拆分以旧study的coordination/main_recheck.json与
action_units_amendment.json为准。下批机械分析采用正确命名即可，不将旧CPU读数纠正扩大为GPU试验。

仅CPU验收/统计/调度的窄修允许另记commit及原因，保留原bank/有效行；不得因此要求重训或重跑来追求单提交形式。
若改变GPU模型、优化或rollout科学计算，则先停止受影响阶段回报主讨论；不热改冻结树、不混入修复前无效行。
正常运行只等退出事件，不轮询训练日志/cache。非finite/OOM、信息墙、来源/配对、采集或预算违约时停受影响步骤。
