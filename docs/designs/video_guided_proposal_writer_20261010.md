# 视频指导的完整参数提案 Writer：双meta、MT合并基底与多起点rank8修订

状态：**Owner 已授权自主实施；由 progress 登记为当前唯一 active design。**
本稿纳入 Owner 最新选择：不用 T；Gemma 和 action expert 各自 fresh 初始化可训练读取 meta LoRA；
执行从强MT开始：把MT合并为冻结基底，fresh rank8残差用非零A0/B0初始化。完整当前批次见§13。
本稿替代
[连续推导](../analyses/architecture_rethink_discussion_20261010.md)§9–11 中仍开放的实例选择，
保留那些段落作为推导历史，不恢复已关闭的实验。
[专家原文](../review_materials/20261010_writer_design_review/EXPERT_REVIEW.md)审阅的快照为 `be0d469a`。
实现、launch 与实证状态看 progress；设计定义不冒充已经实现或已经学成。数值是本实例选择，不是 Owner 永久要求。

## 当前定案与证据身份（2026-10-11）

main已消费原八条rank128教师及全部配对原行：[完整报告与科学判断](../analyses/video_guided_proposal_teacher_batch_20261011.md)。
audit MT16/128→T16074→T48083支持原参数族中有益教师供给；相邻丢13、task29下降和selection排序错位保留。
本稿现登记后继完整实例：MT合并冻结基底、fresh 38-target/rank8、预先准备16个多parent/H教师事件，
G从首个更新联合学习，随后固定ψ训练π并完成规定读出。累计预算96GPUh，包含已记账18.640412104GPUh；新增峰值64GiB。
原八教师不重启；原G320后一次四学生刷新未实施，其合同由Git `e0d09485`、原run contract及历史记录保存。
当前rank8实现与计算尚待实验session按本稿承接，设计授权不等于已经开跑或验证。原rank128端点不截断为新标签，不恢复旧optimizer。
rank8降低生成坐标数而不是证明容量、泛化或不遗忘；π冻结G时不能修复未学会的parent编辑。§13给出具体范围与科学停止线。

## 1. 结论、目标和真正改变的假设

保留一个提案生成器 G 和一个编译决策器 π。G 学习从有序教学、实际经历和父策略，生成一套协调的完整 LoRA；
π 学习依据同一教学和实践证据，决定从哪里继续、试用哪套参数、何时结束及最终留下哪套。
视频的职责是提供操作对象、相对运动及变化顺序的线索，减少控制提案与实际决策的错误；
注意力、可解码性、参数随视频变化，均不等于已经产生这种收益。

两项核心信用分别是：**真实完整编辑的参数监督教会 G 提案；实际候选后果及完整编译终局回报教会 π 使用提案。**
G 不通过旧 `P J^T q` 出口生成参数；π 不沿估计价值的动作导数修改参数。
这撤换具体写入与信用组织，不宣称旧方法没有完整 A/B、经历、功能梯度或真实 RL。

当前实例把同source的MT300合并为冻结执行基底，K=1、38 targets/rank8残差；不加载T Writer、T公共参数或T特征缓存。
对38层以FP32计算`W_base=W_source+B_MT A_MT`，按原生执行dtype使用；原MT alpha/rank=1，其它source权重不变。
最终是一套登记清楚的冻结基底与一套完整任务残差，Gemma执行prefix保持原source。原source/MT资产只读。
这不是压缩MT：其完整更新已进入基底；新增任务改变量才受rank8约束。原教师的`B′A′−B_MT A_MT`不能直接等价搬入此族。
新初始化的两组读取 meta LoRA 属于共享特征学习，不是初始任务策略或最终的第二套 adapter。
弱执行起点在数学上允许；首批不展开弱起点学习臂，其父参数、失败历史及训练监督须在真正采用时实际覆盖。
若采用恒等LoRA，沿现有合同取确定性非零A、B=0，避免教师功能优化的A/B双零一阶死点；
这不保证实际梯度有益。恒等只是固定初值，教师可优化其副本，G可生成从它出发的完整修改；不能将强起点训练直接外推。
去掉强执行起点、fresh读取meta、fresh G/π和不使用合法共享教师是不同选择；冻结预训练source仍保留。
新 condition 中不运行训练教师，不恢复或准备动作轨迹再做 BC/FM；G/π 共享权重冻结，变化的是真实历史和生成/选择的完整参数。

本稿补齐教师算法/目标分布、参数块耦合、空历史、决策类型、版本冻结及诊断归因。
仍需实证的命题是教师是否有益、G 是否传递控制、视频是否提供操作收益、有限预算下能否保留并迁移能力。
**定义清楚不等于这些命题已成立。**

## 2. 原始信息、可学习读取与训练标签的边界

| 对象 | 内容与身份 | 可以进入新 condition 的哪里 |
| --- | --- | --- |
| 教学 V/L | exact language、同步双 RGB、stride5 原始顺序；不含 teacher action/state/reward/ID/路径 | 共享meta读取及G、π的教学输入 |
| 自身历史 H | 实际前后 RGB、8D proprio、实际执行动作及 mask、reward/success、自然终止/预算截断、episode/顺序、实际完整参数引用 | G/π 的经历输入 |
| 候选响应 | 在真实已见观测上，用某套候选及保存的 native noise 计算的动作预测 | π 的预测输入，必须与实际执行事实分开 |
| 共享训练标签 | 36 个授权任务的跨 episode actions；经验证的训练专家响应；教师编辑；独立 query 结果 | 只进入相应共享训练 loss，不进入条件特征 |
| 教师模拟分支 | 从合法自身前缀复制的环境、专家继续执行及独立参数优化 | 仅构造训练标签，不追加到学生当次 H |
| 最终 query | 最终参数锁定后，在预留新初态的真实回报 | non-held 共享学习的终局目标；正式 held 反馈不回流 |

H 可以由多套参数产生。每段保留原产生者；选回旧 parent 不得把后来其它参数的经历改写成该 parent 的经历。
参数引用只用于取回真实张量、连接经历及标记是否实践过；不把随机编号、文件名或 task ID 做成可学习特征。

首个实例直接使用当前环境接口已经返回的 RGB/proprio/动作/反馈；不假装已有接触标签或对象 3D 标注。
保存自身环境快照只服务训练期恢复教师的真实前缀验证，不把 task-specific 成功程序送给学生。
现有 Chain 的 RGB/proprio 不足以重建完整模拟状态；共享训练采集须在现场另存必要的 sim/wrapper/RNG 快照，
作为标签侧资产，不进入 G/π 输入。缺该快照的旧事件不能补造恢复标签，也不能以图像推测状态冒充真实恢复。
[连续推导](../analyses/architecture_rethink_discussion_20261010.md)§9曾讨论的额外物理事实辅助头不纳入这个首版 loss；
未来增加时需单独给出可取得标签、消费者和效用理由。
这不是禁止合法自身特权反馈，而是避免用尚未取得的事实字段填补机制。

**教学读取使用两组 fresh meta LoRA，全部 source 基础权重冻结。**
Gemma语言/视觉prefix支路与action expert支路分别在各自18层q_proj/v_proj上安装rank32、alpha32、dropout0的读取LoRA；
两组均采用确定性非零A、B=0，独立seed为202610101和202610102。它们是共享参数，归ψ，不按教学生成。
meta rank是本轮读取容量选择，与最终38-target/rank8残差不同，也不承诺为最优值。

逐帧使用真实双RGB和exact language构造原生prefix，固定probe seed1729的完整`50×32`噪声、flow time=1；
省略teacher state，不补零state。一次原生双stream前向取Gemma最后层图像位置的contextual `Z_t:512×2048`，
以及action expert的`H_t:50×1024`；语言另有冻结token embedding输入。Z不是embed_image刚输出的raw patch。
raw image embedding仍可缓存；新Gemma meta通过prefix实际参与Z及H，action meta通过suffix参与H。
沿原生mask，prefix不看suffix，suffix可以读取prefix；因此Z依赖Gemma meta，H依赖两组meta，
不能将这两条实际依赖关系写成互相交换完整物理状态的假设。两者均取原生最后norm后的返回值。
action_in/out基础投影保持冻结；不创建只计算却没有特征消费者的action_out读取meta参数。
图像、旋转、tokenizer、source normalization沿官方合同；两组meta的梯度通过真实读取回到其A/B。

G监督CFM阶段共同训练这两组meta及G的Reader/融合/数值网络；π用途学习及完整RL阶段冻结全部ψ，
包括这两组meta。π自己的下游Reader仍可训练，G不接收π的trainable embedding。
共享特征也由π梯度更新会改变候选生成核K；本实例不采用这种未计入完整score的更新。
meta更新后所有依赖它的特征缓存失效；不得继续调用旧`@torch.no_grad()`读取或使用T/MT特征缓存来冒充训练。

读取与执行有明确参数上下文：教学读取使用原source加两组meta，不叠MT/任务残差；候选响应、新教师功能优化、
实际实践与最终执行使用MT合并基底加当前rank8残差，关闭meta，Gemma恢复原source。
旧对齐专家作为标签消费者时，使用原source加其完整rank128，不能在MT基底上再叠专家128。
复用同一policy实现及显式权重上下文，正确承接functional_call、并发、重计算、冻结与缓存身份；不复制整套policy或保留平行fallback。
38层合并权重BF16共81.125MiB，必要原始/合并target缓存纳入实际存储；允许正常数值差异，不要求逐bit等价。
rank32读取因子、rank8任务因子、旧专家128和两种冻结基底的所有权/导出/schema分开；以实际消费者核验上下文。
meta只在编译期读取教学，最终只留下完整rank8任务LoRA及登记基底。

复用原生图文/attention算子、合法teacher store与官方执行接口；工程落点及原件定位由执行者登记，
旧特征函数不是可直接复用的梯度合同。实际模块shape、双stream返回与两条meta梯度由消费者核验。

## 3. 教学和经历如何参与每一次写入

G 和 π 各自拥有可训练 Reader/融合参数；共享§2读取器，在G学习时可训练，在π学习时冻结。
每帧用 8 个 width256 的学习 query，读取投影后的contextual图像位置、suffix hidden 和语言 tokens；
保留相机、空间位置、原 frame index，随后用 4 层时序 Transformer 编码所有帧的 slots。
这是对学习后特征的注意力读取，不平均原始 frames/raw features，不重排实际帧序。

自身每个 replan 事件分别编码前后图像、proprio、真实执行动作/mask、反馈、边界类型和实际参数表示；
按 episode 与事件顺序进入 4 层 width256 编码器。自身图像使用冻结source的raw图像embedding，前后变化保留；
不因教学采用meta读取而把旧自身hidden改写为meta产生的事实，也不假装两种特征具有已知语义对齐。
事件 query 读取完整有序教学 memory，可以对照不同阶段、重试和回退；不强制归一化时间或帧号对齐。
语言 token 与一个始终存在的决策 token 也直接读取教学，H 为空时只有 EMPTY 标记，仍有合法视频通路。

G 的父参数表示与自身历史不是一个训练 expert ID。每条事件引用其实际参数的 38 个 target 摘要，
使模型能够区分“同一动作后果来自哪个实际控制器”。为避免循环依赖，先用独立的静态因子编码器：
实际 `(Λ−Λ0)/S_G` 与地址进入64数值块，经2层§4同类 block/rank/target 通信得到38个摘要，**不读取 H、视频或流变量 x_t**；
再由摘要与真实事件组成 H 编码，最后供 G 的6层速度网络读取。静态编码器参数归 ψ，随 G 一起训练/冻结。
π 使用相同类型但独立权重的 Reader。其候选 query 还要读取候选的预测动作和参数，见§5。

这些算子提供视频/自身/候选对应的可训练通路；未预设 slot 已代表抓取、接触或成功必要步骤。
G 的视频梯度来自完整编辑目标；π 的视频梯度来自实际用途差异与最终回报。
两者仍可能学习语言或场景捷径，必须按§10检验操作内容的净效用。

## 4. G：在完整因子坐标中生成协调编辑

### 4.1 坐标、输入和真实跨块通信

每个 target 的 `A[r,:]` 与 `B[:,r]` 沿实际坐标分成 64 数值块；短块 padding 并 mask。
沿原38-target顺序使用rank8/alpha8/dropout0：643,584个有效数值、10,064块、304个target/rank组、38个target，
padding共512个值。所有任务共享由seed7按canonical target/side顺序生成的Kaiming非零A0、B0=0；Λ0与基底身份保存。
A/B均可优化或生成，A0不是永久冻结的容量限制；不得逐condition改随机基、截断旧128因子或重分解教师坐标。
同一完整样本只有一组完整噪声和一个 flow time，不逐块独立采样再拼接。

每个块输入自己的 `x_t[64]`、实际 `(parent−Λ0)/S_G` 数值、有效位mask、target/side/rank/块坐标及t；
数值投影到 width256，6 层按以下顺序通信：

1. 每块保留本地数值残差和共享 FFN；同 target/rank 的 A/B 块汇入同一个 rank query。
2. 每个target内8个rank tokens交互；target query读取这些rank tokens。
3. 38 个 target tokens 相互注意，并读取 G 的有序教学/经历 memory。
4. 更新后的 target 回灌到各 rank；rank 与 target 上下文回灌到**每个本地块**，更新其数值状态。

每层 FFN 内宽1024、4个注意力 heads，采用 LayerNorm、无 dropout/运行均值更新；
最终每个块直接输出64维速度，padding 不参与输出/loss。
摘要用于跨块通信，不成为唯一的低维输出码；每个局部块一直保留自己的随机和数值通路。
这个层次结构允许全局控制模式在 A/B、rank、target 间协调，但不保证有限宽度足以学成。
不做10,064块的全局self-attention；同一BF16 block hidden约4.914MiB，仍未计反传、FFN、视频和物理batch。
输出坐标数缩小16倍不代表共享网络参数或原生读取/教师训练成本同比缩小；具体吞吐必须实测。

### 4.2 参数尺度与条件 Flow Matching

区分教师的固定参数邻近单位`S_prox`与G的编辑尺度`S_G`，二者不可在实现中混成一个layout字段。
先由冻结MT及共同初值对每target定义

    S_prox,A,l = RMS(A0,l),
    S_prox,B,l = RMS(B_MT,l A_MT,l) / (sqrt(8) * RMS(A0,l)).

按独立零均值单位方差B与固定A0估算，后一式使单位化残差的期望稠密更新能量与该层MT更新同阶。
这只定义坐标单位，不等于函数邻近或能力保持；原MT稠密更新与Kaiming期望RMS给出的B单位估计约0.01422–0.15957，
formal数值使用实际保存的A0之RMS计算，不能把此估计区间当固定常量。
B0=0不能用其RMS的1e-6地板作新残差尺度。原source normalization不变。

全部16事件的合法教师bank及selection完成后，在任何G正式更新前按§13固定task/event/q_T权重计算

    S_G,l = max(sqrt(E_registered mean((Λ_T,l−Λ_p,l)^2)), 1e−3 S_prox,l).

missing事件按原机会贡献0且保留分母，合法节点按q_T；不按单条标签、当前batch、audit或held重估。
保存权重、各因子统计和地板生效情况；全无有效标签不得靠地板启动G。源标签与统计只来自授权non-held共享训练。
静态参数输入以Λ0为中心，零坐标对应MT有效策略；CFM编辑以真实parent为中心。后续数据刷新保持S_G，
改变尺度属于共享模型输入/optimizer语义迁移，不能称普通exact resume。
教师从真实parent原始A/B副本出发，禁止SVD重分解、rank置换或gauge restart。

设完整教师编辑 `d*=S_G^{-1}(Λ_T−Λ_p)`，同一次训练抽样使用

    ξ ~ N(0,I_D),  t ~ Uniform[0,1),
    x_t = (1−t)ξ + t d*,
    L_CFM = (1/D) Σ_valid [v_ψ(x_t,t,V,L,H,Λ_p) − (d*−ξ)]².

loss 的坐标可加性不意味着输出分布独立。六层通信联合决定所有块的速度。
推断从完整 ξ 出发，固定16步 Euler、步长1/16，评估 t=0,…,15/16：

    x_{k+1}=x_k+v_ψ(x_k,k/16,X)/16,
    Λ_c=Λ_p+S_G x_16.

这是参数分布生成，不是生成动作轨迹后微调。有限网络/16步积分不等于精确复现理想条件分布。
**零速度头留下的是 ξ，不是零编辑。** 随机初始化 G 只接受共享监督，不能先当作可用实践策略；
原样 parent 是显式候选，不靠 G 恰好生成零编辑。

相同原始坐标与 proximal 只固定具体标签算法，不消除 `BA=(BR)(R^{-1}A)` 的非唯一性。
若学生误差为 E_A/E_B，实际权重误差是 `B E_A+E_B A+E_B E_A`；小因子 MSE 不保证小函数误差或控制保持。
故纯 CFM 是本实例的可失败选择；其后必须检查积分端点、实际函数和完整闭环，不能交给 π 自动修复。

### 4.3 生成参数怎样改变自身执行

物化后的候选已经是一套完整rank8 A/B，使用同一MT合并基底，不另部署T、MT动态adapter或第二个residual。
每个目标线性层对**当前自身执行**的真实 hidden h_l 计算

    y_l = Linear_MTbase,l(h_l) + B_l A_l h_l           (alpha/rank=1).

Linear_MTbase包含原source冻结bias及已合并MT更新。q/v投影的变化改变action expert内的注意力及后续hidden，
action_in/out 改变流变量的输入/速度输出。
这些作用在当前自身 RGB、proprio 和 exact L 的原生 prefix 下重复经过10次流更新，形成真实执行的前5个动作；
后续观测再进入同一套固定参数。这里的 h_l 不同于教学读取时由双meta产生的 state-free hidden。
由于 h_l 也依赖其它层和此前流更新，分块参数 loss 不能逐层等同闭环收益。
共享教师的功能目标通过这条实际路径更新其参数副本；G接受端点坐标监督；π接受完整实际后果信用。
这三种梯度消费者分别定义，不能将“存在原生可微路径”写成新 condition 正在进行动作拟合。

## 5. π：空历史、三个决策阶段和候选作用

π 的全部 trainable Reader、经历/参数编码、候选 trunk 和 heads 归 θ，不与 G 共享可训练张量。
各候选的完整因子通过§3独立的2层静态因子编码器形成38个 target tokens；其参数归 θ，π 不生成参数。
每个候选始终有一个 candidate query，读取 V/L、预算、实际参数、实践身份与历史摘要。
另有 decision query。即使 H 为空，这两种 query 也存在，不能只靠事件查询间接看到视频。

有 H 时，从全部真实 replan 观测按 chronological 等距索引取 `min(16,n)` 个，固定取样不依赖 θ、回报或候选打分。
每个候选在这些实际 RGB/8D proprio/L 上，复用原事件保存的 native noise，计算完整 `50×32`、10步 ODE 的前 `5×7` 响应。
输入 π 的是 source-normalized **绝对动作预测**；practice 阶段另附相对本次 parent 的差，parent/final 阶段不构造虚假参照。
相同真实动作、官方后处理和实际环境后果另在 H 中记录；预测不被标记为已执行或接触证据。
H 为空时响应集合为空，不伪造图像、state 或原生 prefix。

响应事件与 candidate query 对有序教学做 cross-attention，再汇入候选 trunk；
候选集合用无任意序号 embedding 的集合等变注意力，每个 head 产生一个 masked categorical 分布：

| head | 合法选择 | 所需价值 |
| --- | --- | --- |
| parent/STOP | 初始 MT、已实际试过的有效完整参数，或 STOP | 以该 parent 继续生成/实践后的剩余编译价值 |
| practice | 原样 parent 与本次有效 G 提案 | 这次真实经历对最后一套参数的价值 |
| final | 初始 MT 与实际试过并保留的有效参数 | 这一套固定参数本身的新初态能力 |

三个 heads 共享 π trunk，带明确 decision-type；不可把固定策略的成功率标签当成 parent/practice 的完整探索价值。
首版均为 temperature=1 的 softmax，无额外探索混合、dropout、top-k 硬筛或 argmax；训练与新 condition 同规则。
数值无效、资格、资源决定的 mask 只依赖原始已知事实和冻结规则，不依赖 θ 的预测分数。
未来若加温度、混合或学习筛选，必须将**实际行为概率/新动作**纳入目标，不能只记录旧 softmax。

## 6. 编译状态机、预算与最终部署

条件级输入预算明确区分环境步与生成计算。这个实例取环境步上限1024（含 settling）、
每次至多4个新提案、整条 condition 至多32次提案尝试、每个提案16步参数积分、每候选至多16个真实响应观测。
这些是可修改但须事前登记的资源选择，不是固定观看/实践轮数。

```text
预先分开 practice 与独立 query 初态/RNG；最终 query 不向 Writer 暴露
H ← 空；P ← {MT基底上的完整rank8初值Λ0}；U ← P
while 还可完成10步settling及至少1个真实控制步:
    a ← π_parent/STOP(V,L,H,P,budget)
    若 a=STOP: break
    parent ← P[a]
    m ← min(4, 剩余提案尝试额度)
    由固定 G 生成 m 个完整提案；每次尝试计费，包括无效提案
    U 加入有效提案；C ← {原样 parent} ∪ 本次有效提案
    c ← π_practice(V,L,H,parent,C,budget)
    在一个新实践 episode 中固定 c；成功/自然horizon/预算耗尽分别结束
    将真实经历及其实际参数身份追加 H；P 保留实际试过的有效 c
Λ_final ← π_final(V,L,H,P,budget) 的一次实际抽样
锁定 Λ_final；独立新初态只运行 MT合并冻结基底+Λ_final
```

生成额度耗尽仍可实践原样 parent，环境预算保证过程有限；新候选全无效也不无限重采。
剩余环境步不足11时强制进入 final；不以只做 settling 的空 episode 制造信息或循环。
每次实际 episode 内参数固定是本实例的选择，不是 Owner 的永久限制。
差候选可以成为后续 parent，不设置单调上涨门。成功只是 H 的事实，不强制 STOP，也不认证新初态成功。
P 不包含未实践的新提案；U 只供记录/诊断，不能因分析知道其成绩而改变最终选择资格。

形状/non-finite 生成失败按固定规则计为无效尝试，不能借重试按结果筛选。
若有限参数到实际原生执行时才产生 non-finite 动作，该次尝试按模型数值失败结束、保留已消耗资源和实际观测，
用缺失动作 mask/失败类型记录，不把 NaN 当环境事实送入 Reader；该候选退出合法 P，不偷偷换策略补成成功 episode。
OOM、资产或进程故障属于工程失败并保留费用，不能悄悄作为零回报训练样本。
正式 query 过程中不换参数、不继续读取教学、没有 π/G/专家/搜索或外部阶段选择。
保存初始参数只保护其可选性；实际误选仍可能损失能力。

首版目标是**硬预算内最大化最终能力**，不声称最优早停或最省教学读取。
理想情况下继续实践后可忽略新信息并选回旧参数，因此无成本目标不能推出主动节约。
任何效率主张需要真实费用比较，或另行定义明确的成本目标。

## 7. 训练专用教师：实际可执行的标签算法

### 7.1 父策略、历史与函数标签

只使用显式train24+support12共36任务，保持当前覆盖协议、source和normalization：
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
前24是target train，后12是已审计non-held LIBERO-90。沿用allowlist不是加载旧T模型。
每个事件记录 `(V,L,H,Λ_p)` 的真实来源；包含空H、MT实践以及后续真实学生父策略/混合历史。
不把同task额外视频、参数节点或标签数量称为更多独立 meta tasks。

从 Λ_p 的原始 A/B 副本和 fresh optimizer 开始，目标是

    L_T = L_demo + 1_{rec有标签} L_rec + 1_{keep有标签} L_keep
          + 1e−3 ||S_prox^{-1}(Λ−Λ_p)||² / D.

各项定义如下；不存在的标签项为缺失，不补零动作，也不重新归一化其它项的权重。

- `L_demo`：同task、跨当前教学 episode 的原生 action-FM。复用实际 query sampler、offset1、随机 τ/noise、
  完整50×7目标与现有补齐/mask语义；不从 teacher 视频这一 episode 取其动作送给条件分支。
- `L_rec`：从 H 的非成功/截断 episode 的实际 replan 前缀，按固定等距规则取至多4个状态。
  只对事前选定、同task且与当前执行接口一致的合法专家，复制这些**自身**环境快照并真实继续执行。
  专家须在原 episode 剩余 horizon 内完成完整目标，才记录该次验证首个 noise 下的原生前5动作功能目标；
  提前成功只监督实际执行的坐标。候选 LoRA 在相同 RGB/state/L/noise 上作完整10步预测，与该目标作有效坐标 MSE。
- `L_keep`：只取 H 中**实际由当前 parent 产生且完整成功**的 episode，等距至多16个真实状态；
  用相同输入/noise 的 frozen parent 原生前5函数作软保持。其它 parent 的成功不能冒称当前 parent 的成功。
- proximal使用§4固定S_prox与原始parent，不使用尚未形成的bank尺度，不重分解因子或重算source normalization。

恢复的“验证成功”只证明这次前缀/noise 的适用，不证明专家在附近状态普遍可靠。
当前四任务 aligned 专家并不覆盖全部36任务；其它任务缺恢复标签就如实缺失，L_demo仍可用。
不可恢复末态不等于任务无可改进之处；较早前缀和新初态避免失败是不同问题。
任何旧 source 专家都不能直接把 A/B 当作当前 source 的标签；本实例不增加未经接口核对的专家池。

这个有限优化真实依赖 parent、其成功/失败经历及固定坐标约束，不把每个 H 都写成同一个预存 task adapter。
但它也不强迫不同 H 或同task正确视频有不同解。视频标签合法不意味着学生已能推断，
教师不会被冒称视频理解 oracle；最终是否需要视频仍由真实迁移和参照裁决。

### 7.2 优化、预定节点与两类数据

实例采用已用过的 AdamW 数值口径：A/B 与 optimizer FP32，source BF16/正常TF32，
lr1e−4、betas(.9,.95)、eps1e−8、weight_decay1e−4、clip1、无 scheduler。
每事件只有一条至多480更新的轨迹，预定节点160/480；不用结果追加 LR/rank/seed 搜索或事后改节点。
每更新112个 demo queries；rec/keep各在自己的有效池中均匀抽至多4个，分别求均值，不按池大小隐式改项权重。
其余固定 τ/noise/query 流及恢复前缀验证都登记来源和费用；这不是声称480步已足够。

事件开始前独立预留两个与 H 不重叠的初态池：4个 `label-selection`、16个 `report-audit`。
parent 与两个预定节点均在各池按相同环境/原生政策 RNG 配对；任何 query 都不回填 H 或教师拟合目标。
selection 池的实测净增 `g_hat` 只定义训练偏好；audit 池只报告实际效果和不确定性，
**不再按 audit 挑选标签后，用同一 audit 宣称教师收益无偏。** 两个节点和负例都报告。
同一有限估计不能给单事件真实成功率提供保证，也不能把 task 内多初态当成独立任务。

令 C_X 为这条有限轨迹的合法、数值可执行预定节点，A_X 为 selection 上 `g_hat>0` 的节点，定义

    q_coverage = Uniform(C_X),
    q_preferred = Uniform(A_X),
    q_T = 3/4 q_preferred + 1/4 q_coverage       若 A_X 非空；
          q_coverage                          若 A_X 为空。

这里特意称 `preferred` 而不称“已证实有益”；小 selection 池可误判，独立 audit 揭示这种误差。
C_X 为空则记缺标签，无 G loss，不补 identity 或重采容易事件；A_X 空则记无正偏好，而非任务不可能。
外层任务/已登记事件按等机会采样；缺标签仍占其机会和逻辑分母，不 resample 成标签丰富任务。
报告每task实际标签率和有效梯度覆盖，不能把采样等权写成已获同等学习内容。

coverage 把部分零/负/不确定编辑真正放进 G 的目标分布，不只是留日志；仍不证明它含专门的信息探测策略。
**首版只依赖任务候选附带产生的信息，不宣称解决任意主动系统辨识。**
专家的探测 E 反例保留为范围限制：若完成任务必须先执行 G 支持外的辨识行为，当前实例可能失败；
完整 score 和多模态生成都不会创造缺失的信息或候选。

## 8. G 和 π 的共享学习、完整信用及刷新

### 8.1 先形成提案，再训练用途

G 按§7事件和 q_T 做完整 CFM；这里下标T表示训练教师分布，与历史T Writer无关。
原始正确教学是主输入分布，不要求先生成另一套视频或拥有每条视频对应的 expert。
两组meta及G的所有Reader、融合、块网络归ψ。参数监督不反传给frozen source、MT或训练教师optimizer。
用积分后样本与 parent/教师的实际闭环比较判断控制传递，不能用低 CFM loss 解锁昂贵决策 RL。
出现教师正例不自动代表完整方法合格；教师未找到优势也不能包装成 G 的学习失败。

π 的初始密集学习只训练 final 用途：在固定 G 后，以有预算的均匀 parent/practice 行为采真实训练历史，
到环境预算结束，保存真实 prefix/终局的 H 与合法保留集 P。不得把未实践新提案冒充 final 的合格集合。
对同一合法 `(V,L,H,P)` 的候选在预留新初态上配对执行，得到 `R_hat_k`，优化

    J_final_local(θ)=Σ_{k∈P} π_final(k|V,L,H,P) R_hat_k.

query结果只作标签，不进输入；包括失败候选。空H的合法P只有MT，此时final强制选择MT、score为零；
不能宣称这个单选样本教会了候选排序，实际候选差异来自后续真实实践。
两候选 logit 的导数是 `π_1(1−π_1)(R_hat_1−R_hat_2)`：真实能力差给视频/经历读取信用，
相同能力的风格不需要区分，全失败不产生虚构的有效方向。
此阶段是监督用途学习，不冒称当前 π 的 on-policy RL；parent/practice heads 不直接拟合固定策略成功率。

### 8.2 一条完整路径的真实目标

冻结G的**全部**meta读取/Reader/融合/数值网络/S_G/Λ0/normalization/运行统计/积分和无效规则，以及MT基底/source。
π 无可训练状态送入 G；G 只从 raw 合法输入与自己的冻结编码生成参数。
给定原始信息 I_j、decision type d_j 和固定 mask M_j，路径为

    p_θ(τ)=p_0(I_0) ∏_j π_θ^{d_j}(a_j|I_j,M_j) K_j(I_{j+1}|I_j,a_j) K_q(R|Λ_final),
    J(θ)=E[R_query(Λ_final)].

K 包含提案噪声、原生 ODE 的正常噪声、环境、预算与原始事实演化；给定历史/所选动作后不含 θ。
query 初态/RNG 在采集前独立保留，不在看到最终策略或实践结果后重新挑选；实践调度排除它们。
因此在通常可交换求导条件下

    ∇J = E[Σ_j (R_query−b_j(I_j)) ∇ log π_θ^{d_j}(a_j|I_j,M_j)].

包括 parent、STOP、practice 和 final；早期选择改变后续 H、G 提案及实践访态的信用也包含在内。
原生执行始终使用正式10步 ODE，不为了动作 log-prob 改成 SDE 或添加另一种动作探索核。
这只确认梯度目标身份，不确认方差小、候选好或有限数据可学。

### 8.3 不能由实现者自行改变的更新语义

1. θ 在整个完整条件 batch 的采集与 score 重算期间固定；一批只做一次 actor 更新，随后重新采集。
   不把同一旧 batch 多轮 SGD 称为上述 on-policy 更新，不未经校正混入旧 RL 路径。
2. batch 先按任务等权组织；每条 condition **累加**其全部 decision scores，再按 condition/batch 平均。
   不除以可变决策数，不让短路径因长度归一化获得不同目标权重。
3. raw RGB/动作/参数样本/环境事实停止梯度；π 自己的历史与候选编码必须重算并正常求导。
   冻结特征可缓存，θ 的旧缓存不能 detach 后冒充完整 Reader 信用。
4. baseline 采用独立参数 ω、当前 batch 开始前冻结的版本；不共享 actor 梯度。当前回报可在 actor 更新后训练下一版 baseline。
   合法性的本质是条件于 I_j 后与当前抽样无关，不是必须在某个时刻计算；独立预留 query 上的固定 MT 回报也可作控制变量。
   不用本条路径自己的未来回报拟合当前 baseline，再声称满足独立性。
5. forced budget stop 或只剩一个合法动作的概率为1、score为0；其它实际抽样概率全部记录。
   不在 mask 前加未计分的学习 top-k，不把部署改成 argmax，也不把当批 advantage 标准化当作无影响细节。
6. RL 阶段不并行用监督 loss 更新 ψ，也不经共享 trainable encoder 偷偷改变 K。
   修改 G/积分/尺度/特征版本后，旧 RL 路径全部退出 on-policy 池；仅校正 π 比率不能修复 G 核变化。

旧教师编辑和后果样本可以作为注明来源的监督条件复用；这与旧路径的策略梯度身份不同。
有限 categorical 的 logit score 范数界不控制网络 Jacobian、长路径方差或成功稀疏性。

### 8.4 学生分布刷新

本批从首个G更新混合§13四类真实parent/H，不沿用原全MT320后一次刷新的课程。
这些父参数来自训练教师，尚不是真实G样本；多producer长H与连续学生parent的覆盖仍有限。
本批用完整G480→π→新初态及指定U读出检验这种近似是否足以形成能力，不预称任意parent可迁移。
若实际学生分布暴露缺口，main从已有路径选择有意义的后继投入，另登记有限教师事件、权重、版本、更新数与预算；
本批不预留自动刷新的额外教师，也不以固定一次刷新作为普遍充分条件。
未来刷新须冻结当次G/采集行为，保存全部实际产生者及成功/失败；更新G后重新固定ψ，旧路径退出on-policy池。
π可保留兼容初始化，不能只校正π概率便把新生成核下的旧路径当on-policy；不无限刷新保护弱假说。

## 9. 能力来源与损失：修正原来的诊断分解

对同一实际编译路径，U是全部有效生成候选加初始MT，P是实际试过并保留的候选加初始MT。
令`p(Λ)=E_s R(Λ,s)`是**一套固定参数**在同一新初态分布的能力，`b=p_MT`，则

    p_selected − b
      = [max_U p − b]
        − [max_U p − max_P p]
        − [max_P p − p_selected].

三项分别对应实际路径的生成新增、实践/保留漏选、最终选择损失；旧稿只看 P，不能单独归因 G。
例如 G 已生成0.9和0.2，π只实践0.2，而MT=0.6；U有能力增量，P没有，不能称 G 没生成好候选。
U 仍不包含未走 parent 分支或过早 STOP 后可能生成的提案；需要时只在小型 non-held 面板强制分支并继续完整剩余编译，
其标签是终局收益，不用当次固定策略成功率冒充实践价值。

这是分析恒等式，不是可供选择器读取的 oracle。估计 maxima 有有限样本择优偏差，须独立核对并报告不确定性；
不得使用 `E_s max_Λ R` 的逐初态 success union 替代 `max_Λ E_s R`，也不得用正式 query 换最终参数。
oracle 差距可以包含合法信息尚不能辨识的损失，不都属于优化错误。

## 10. 视频价值、完整比较和失败裁决

早期证据必须贯穿“训练教师 → G 样本 → 实际实践/决策 → 新初态”，并在适量 non-held 条件检查教学的作用；
不把内部 loss、局部衰减修复、一个 teacher 正例拼成整体资格。

| 完整结果 | 更新哪项判断 |
| --- | --- |
| teacher 独立 audit 无可靠净优势 | 本实例教师供给未建立；不把不存在的监督留给大 G/RL 自行解决 |
| teacher 有益，G 样本丢控制 | 当前联合生成实例未传递能力；π RL不能修好没有的候选 |
| U 有益、P 或最终选择丢失 | 区分实践选择、最终选择与信息不足；必要时审计早期分支 |
| 最终总是退回MT，与初态配对MT相同 | 没有取得新控制；不能把实践次数或选择器训练写成能力新增 |
| 训练链有效、held 无效 | 加强表示/条件/任务覆盖或迁移解释，仍不由单例确诊 |
| 全方法提高，但公平视频参照无增量 | 参数适应可以有效；EMBER 视频指导主张尚未建立 |

保留静态MT与实际生成/实践/最终选择的能力分解。初始集合只有MT，因而只选初始策略不能产生新控制，
不再运行原稿T/MT选择器对照或加载T参照。相同query初态/RNG下只部署MT的等预算选择器与静态MT功能相同。
认证视频增量还需充分训练的语言/无视频Compiler：去掉共享meta读取中的视频及G/π的全部视频通路，保留强共同能力、
相同合法共享监督及可比实践/计算预算，不能只对完整模型输入零图。
区分操作内容与场景识别时，在小型 non-held 面板使用训练过的静态场景参照：只看原视频首个 stride 帧及语言，
给予容量/训练机会，不手选最有信息的时刻。它的局限和“静态布局也可能有用”一并报告。
这些参照需要证据时再按预算落实，不把全部消融设为开工前置门，也不降格视频必须有用的目标。

固定 parent/H 后只屏蔽 π 的视频是重读消融，不能称全系统无视频。
same-task 另一条正确视频应检验覆盖/保持，不要求它变差；wrong-video 变差本身不排除分布外异常。
shuffled/reversed 仅在最终共享 checkpoint 冻结后检验，不进入训练、选点或架构修正。

正式结论继续用 single-checkpoint strict paired400、50教学/task整轮无放回、相邻 checkpoint 稳定性、
per-task/suite、retained/gained/lost、breadth/churn 及视频因果证据。初态/RNG/视频映射不由 worker 或结果决定。
历史153/161不是新面板的自动分数；仅在配对身份真正一致时复用历史参照。
首要目标仍是明显超过强 MT；不把旧145阈值或专家举例的5个百分点升级为 Owner 永久门槛。

## 11. 有界验证组织、成本与实现边界

本稿没有启动或承诺专家建议的整个大额度。按专家216事件、每事件480×112 query 直接展开，
仅 demo 教师训练就达11,612,160 queries，尚未计10步功能反传、恢复验证、G或π；不能称作已有依据的“小试”。
首先以一个完整且有上限的教师—G—用途—适应单元取得有判断价值的控制/吞吐证据，
随后由主讨论在同一科学边界内登记实际均衡36任务规模、采集/更新数、相邻400节点、净改善目标及 wall-clock/GPUh 上限。
这不是等待 Owner 例行批准，也不是自动逐模块晋级；没有实际吞吐时不虚报 ETA。

首批科学预算、固定学习节点与报告范围由§13登记；物理batch/并行、精确命令和实时准入由执行者在运行前补齐。
36任务扩展及正式400尚须由首批实际机制与吞吐决定；不是等Owner再次许可，也不从旧实验恢复数值合同。
G 的6层/16步、教师目标权重、候选/环境额度都是当前可检验实例，不承诺为最佳配置或永久结构。

成本至少分别登记教师拟合、专家前缀验证、selection/audit、G训练及积分、候选原生响应、实践、
meta queries、正式读出、视频编码/重读/缓存和 I/O；无效提案、失败与取消都计费。
当前完整rank8 factor FP32约2.455MiB；待更新batch的全部候选需要支持π重算，不能只存stale trainable embeddings。
已消费 batch 可按生命周期退役可重建候选，预登记诊断面板保留 U 的必要参数；不默认永久保存每条路径全部32个大样本。
训练标签、正式最终参数、raw rows、来源与必要复核证据按仓库合同保留。总峰值还须计 RGB、optimizer、缓存、临时副本，
不能用上述因子尺寸冒充总空间；运行前按 data1 独立 quota 和实时 GPU 规则准入。

工程复用现有`ember.proposal_writer`的source、teacher store、官方执行/队列、G、π与完整condition score，
由唯一实验session在独占分支承接基底上下文、rank/尺度、四类事件与恢复schema，再集成main并冻结。
旧experience算法已退役；本次不再创建rank8平行runner，不以兼容名保留旧rank128 G作为active fallback。
旧rank128教师/模型读取仅服务明确的历史原件或训练标签专家上下文，生命周期和类型不可混同当前任务残差。
必要公共执行/数据职责复用，旧算法与计算身份由Git、原formal冻结版本和run contract保存。

## 12. 与专家意见的取舍及结论层级

| 专家指出的问题 | 当前处理 |
| --- | --- |
| 教师分布、探索供给未定义 | §7固定有限教师算法与 preferred/coverage；首版只承诺候选附带信息，保留 E 反例 |
| gauge 与跨块耦合不明 | §4原始parent坐标/固定尺度/真实分层通信；不宣称几何或控制保证 |
| 空 H、parent/final 相对参照缺口 | §3/5常驻query、绝对响应和三个不同用途head |
| 完整冻结、缓存、概率/更新条件 | §8冻结全部生成核、raw与θ编码分开、一batch一次更新、完整condition SUM |
| 只看保留集会错归因 G | §9使用U/P/selected，并限制到实际路径 |
| 冷启动及零速度的错误直觉 | §4/7明确噪声端点，先监督G、显式原样parent，弱起点另需真实覆盖 |
| 视频参照、STOP/强策略保持 | §6/10保留MT强参照，定义合理视频比较，不将存在STOP或MT选项写成保证 |
| 专家数值建议是否直接成为合同 | 接受可计算网络实例和明确目标；未接受无实测依据的整批资源/轮数或收益保证 |

另外补正两项相互作用：用于偏好标签的 selection 与用于报告的 audit 不混用；
密集 final 训练只使用真实合法 P，不能让 untried 候选资格与最终部署不一致。
未发现推翻 G＋π 分工的硬数学矛盾；以上已知定义缺项有了具体处理。
但有益教师覆盖、参数到控制的传递、可由合法视频/经历辨识的效用差、有限预算泛化与误选控制仍是可失败假说。
不能把“目前没有未处理的已知接口反例”扩写为“这套方法必然学成、交互后的历史问题全部解决”。

## 13. 当前正式实例：rank8多起点完整链（2026-10-11）

### 13.1 科学取舍与宏观路线

本节由main消费原八教师完整结果后定案，授权现有实验session实现、核验、启动并完成下述整批。
Owner已授权主讨论自主调整方法和投入；常规实现、参数族迁移和本节预算不再请求Owner批准。
原批只证明有标签训练教师在原rank128族能改善当前四任务；本批检验较小残差族的能力能否经G与π传递。
同时改变任务参数族与parent训练覆盖，是明确登记的完整后继，不把结果单独归因于rank或某个新模块。
保留两个教师节点及原selection/q_T规则；task29的反例降低“晚节点更强”和小池偏好可靠性的支持，
不据旧audit追选480、重加权个别task或扩大selection池来保护本次假说。

| 层级 | 完整问题 | 依据与后继 |
| --- | --- | --- |
| A：本批4个non-held任务 | rank8教师是否可用，G是否传递控制，π是否通过实践留下净收益 | 完整bank、实际G/π路径、指定U与新初态结果；连同成本及少量视频干预 |
| B：36任务共享学习 | 参数编译能否迁移，视频是否超过语言/共同能力 | 据A行为与吞吐确定等权规模、真实学生数据及公平语言/静态参照，不自动复制大矩阵 |
| C：正式held结论 | 是否明显且稳定超过强MT | 相邻strict single-checkpoint paired400、task/suite覆盖、R/G/L/churn及冻结后controls；Test保持最终方法冻结边界 |

A是有限训练诊断，可能凭语言记住四任务，不凭A正例认定视频操作知识或held泛化。A阴性约束实际受检链路，
由main消费既有原件、更新假说并自主接续；停止无效投入不等于停止整个研究。

### 13.2 身份、数据与初态预留

新run ID为`video_guided_proposal_writer_20261011/rank8_multistart`，ROOT在`/data1/user/ymdai/ember_runs/`下。
原`video_guided_proposal_writer_20261010/pilot`只读保留全部原件/失败/16个完整checkpoint，不作为新rank8恢复源。
新实例只用train tasks `[12,29,32,38]`；沿现有coverage24/8/8、source1000、MT300、source normalization和官方执行。
任务来自预先已有的同source恢复专家/接口覆盖，不按本批分数换task；Spatial仍未覆盖。

学习前复用原已冻结`panel.json`中的50视频/50初态排列与角色，保存新manifest及引用；
原数组来自`numpy.default_rng(SeedSequence([20261010,domain,task]))`，domain0/1为video/init。
每类事件、阶段和condition的环境/政策/query/CFM RNG实际根与调度写入新manifest，在计算前固定。
同一比较按task/state/env/policy/first-noise配对；GPU、分片、worker顺序和恢复不改变这些逻辑身份。
task ID、文件名、event ID仅调度审计，不作为可学习条件。

- 教学前两条记V0/V1，用于四类教师事件；第三/四条用于π训练，第五/六条用于最终报告，第七条用于same-task-other。
- 初态前4个是教师selection，随后16个是教师audit；都不进入H、教师拟合或Reader。
- 第21–36个是practice有限池，部署路径按固定循环使用；首批代理parent数据按下节明确采集。
- 第37–40个用于π-local候选用途标签，第41–44个用于π-RL独立query，第45–50个用于最终报告。
  同一condition的实践排除对应query池；query在最终参数锁定后执行，不回填当次H。
- 所有teacher输入action-hidden；动作监督仍可使用合法train池中除当前教学episode之外的49条。
  教学condition留出不等于所有训练监督的episode留出，不宣称本批测到了新episode标签泛化。

原source/MT、dataset、tokenizer和四个已审计专家只读复用；准确路径/版本由现有asset入口登记。
canonical目标顺序与原38-target相同，fresh任务rank8/alpha8/dropout0；新Λ0、合并规则、S_prox及最终S_G完整保存。
原专家执行仍为原source+原128，不截断、不叠在MT基底上。新教师优化自身rank8副本，最终condition只用G生成参数。

### 13.3 先完成16个真实多起点事件，再共同训练G

每task预注册四个事件，序号0/1/2/3固定对应seed/mid/late/return；缺失仍占原机会和分母，不重采容易事件。
先从Λ0做seed教师，教学V0、空H，fresh optimizer训练至480，保留160/480。
然后分别从空H开始，在practice池第一个初态执行seed160完整episode、第二个初态执行seed480完整episode，
得到H160与H480。两条为独立的真实实践路径，按自然成功/horizon结束；每条最多horizon+10≤530，
保留足够1024步预算供下一次生成/实践。收集行为不读取教学、不使用G，身份记为训练教师产生的代理parent。
不因节点成功与否或新旧次序挑选/替换节点；160/480不是预称“弱/强”。

| 事件 | V | parent | 真实H | 教师端点 |
| --- | --- | --- | --- | --- |
| seed | V0 | Λ0，即MT有效策略 | 空 | fresh160/480 |
| mid | V1 | seed160 | H160，实际由seed160产生 | 从该parent fresh160/480 |
| late | V1 | seed480 | H480，实际由seed480产生 | 从该parent fresh160/480 |
| return | V0 | Λ0 | H160，产生者仍是seed160 | 从MT fresh160/480 |

return覆盖“保留新经历后选回MT”，当前parent可与经历产生者不同；其keep不能借用seed160的成功。
mid/late的keep只来自各自实际成功episode；所有rec来自真实可恢复前缀并经专家在剩余horizon内验证。
H、参数引用、首末观测/hidden/实际动作、反馈、完整快照先保存再产生标签；缺项如实保留，不补造或只保留成功。
两个种子检查点复用为parent，不冒充已经针对新H优化的标签；mid/late/return各自有fresh optimizer与§7目标。
每个事件仍112跨episode demo queries/更新、480步上限、160/480固定节点；§7优化器、rec/keep/prox权重不暗改。
最多32个预定教师节点中的合法节点与各自parent均作selection4/audit16配对；q_T只读selection，audit不重选节点、不进入尺度或梯度。

两条完整长任务episode可能各需530步，串接后已用尽1024预算；本实例因此分别采集H160/H480，
不把预算终点的H_mix包装成仍可继续修订的部署状态。当前四类事件均对应可实际继续的状态；
它们并未覆盖完整多producer长历史或真实G误差，后两者由本批学生路径检验，不声称已被穷举。
更多parent/节点是四任务内的状态覆盖，不是新增独立meta-task；数据由教师生成，不冒称当前G的on-policy采样。

全部16事件的训练/selection/audit完成后一次形成bank和§4 S_G，再开始G正式学习；
只有全bank无合法标签，或所有合法事件的两个节点均未在audit对各自parent取得净增时，先交完整科学结果而不投入G/π。
这条停止线不用于挑选好task、删除负标签或改q_T；普通单task回退不改变登记矩阵。

G与两组meta fresh，seed7为网络/训练随机身份，meta A/B按§2独立固定seed；不加载profile更新。
AdamW lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1、无scheduler，480次更新，有效batch4且每task一个事件。
从第1次起，task内四事件各1/4，节点按q_T；完整CFM噪声/time独立抽样，missing事件保留0贡献与逻辑分母。
所有ψ由CFM共同学习，不暗加功能loss、teacher动作输入、参数clipping或零门控。240/480保存完整恢复状态，
240只作预定学习记录，480是固定的后续π版本；不按有限小面板选择共享checkpoint。
本批没有自动学生教师刷新；若完整证据要求，main另定具体后继，不因已有代码实现了旧刷新便执行它。

### 13.4 固定G后的π学习与完整报告

固定G480的全部ψ、双meta、S_G/Λ0/基底、积分规则及缓存身份；π下游Reader/trunk/三个head独立fresh。
π与独立baseline各自使用上述AdamW口径，不能让π梯度改变生成核。

**局部用途。** 4task×2训练教学各采2条完整路径，共16条，以均匀parent/practice行为、不主动STOP采到预算结束。
每条最大1024环境步（含settling）、32次提案尝试、每次至多4个、16步参数积分，沿§6开放循环；不是固定实践轮数。
各真实episode边界保存H与合法P，对P候选用4个local query初态配对取得固定参数用途标签，失败候选也保留。
作128次final目标更新，有效batch4/task等权，task内condition等权、condition内真实prefix均匀。
单选P保留零排序信用；不因没有正例而重采容易条件。该行为数据不冒称π的on-policy路径，parent/practice头不拟合局部成功率。

**完整RL。** 固定ψ，16个actor更新，每次4条当前π完整condition、每task一条，共64条新路径。
训练视频按预注册循环交替；参数锁定后从4个RL query初态按固定循环取一个取得R，同task/state/RNG的MT基底初值作控制变量。
每个batch固定θ采集并重算完整score，condition内parent/STOP/practice/final的score求和，condition间平均；
只更新一次再采新batch。baseline当前batch独立/冻结，actor后拟合下一版。原生10步ODE不改成SDE。
无旧G版本路径或旧θ路径多轮SGD；强制单选/预算停止score0。全部原始行为概率、失败、预算截断和实际产生者保留。
这阶段真实看到不同学生parent/H，更新的是选择与实践，不能宣称因此教会了被冻结G的新parent修订。

**报告。** 预定π-local128及π-RL16各在4task×2报告教学完成一次全编译；每condition固定唯一LoRA在6个新初态执行，
各48 rows，与MT合并基底/Λ0逐行配对。读出不反哺梯度、q_T、G或π选择；不由两个报告挑最好版本，后者仍为预定终点。
报告绝对成功、task/suite、R/G/L、churn、相邻success-set、实际parent深度/来源、是否选择生成参数、停止与重读/实践成本。
同一教学在6行复用，只有8个教学condition；不是50视频无放回、不是held或正式paired400。

在task12/32各第一条报告教学的π-RL16实际路径上，对全部有效U（含MT）作同6个report初态完整读出，
支持§9的U/P/selected分解。未走的分支仍未知；max的择优偏差明确报告，不用逐初态success union替代固定参数能力。
其它条件保留原件但不凭不完整U归因G；完整诊断不改变当次已锁定的最终参数。

冻结π-RL16后，同两个条件补same-task-other（第7条教学）和cross-task-wrong两种完整编译；wrong互换task12/32教学，
语言仍是目标任务。practice/query/RNG/预算配对，same-task正确视频不要求下降，wrong下降本身不证明操作理解。
不跑shuffled/reversed。充分训练的语言/无视频与静态参照仍在B的早期按实际缺口登记，不无限推迟视频必要性。

### 13.5 资源、工程、失败与交付

累计硬上限由52修订为**96 GPUh**，包含原批18.640412104GPUh（其中0.0375是未计时短诊断的保守占用）及此后全部开销。
新批可用上限77.359587896GPUh，不把96当作再加96；profile、工程失败/取消、采集/恢复、教师/readout、G、π和全部报告统一计费。
原8×480教师实耗15.65815GPUh，按相似消费16条约31.32GPUh，按各设备10.01–20.44秒更新的外侧范围约21.35–43.61GPUh；
新成功parent可能引入keep，rank8教师真实速度仍需profile，不能按输出减少16倍外推。
暂规划教师更新32–40、教师/恢复/配对读出5–7、G8–14、π及完整报告8–12、profile与新采集1–2GPUh，
整批新增约54–75GPUh。分项可在总上限内据实调度；首个真实G/π消费者修正预测，预留规定报告费用。
工程预计4–8h，主要计算预计14–24h，依据上述费用、实际4–6卡有用并行及teacher→G→π依赖；不是保证或已测rank8时长。
明显超期、预测超96或科学范围改变时向main交具体原件，由main自主裁决，不能静默缩科学矩阵或向Owner例行求许可。

新增峰值仍64GiB，按整个proposal工作自原批开始的保留输出及新批峰值合计，不是每子目录各64。
复用source/data/环境，不复制整套policy；计入合并target、完整checkpoint、rawH/快照、可重算候选、optimizer/cache/临时副本。
launch前执行者核strg01的data1独立quota、相关目录用量和预计峰值。主讨论当前只作小型文档/CPU分析，不以旧quota代替新准入。
已消费、可重建的候选仅在消费者与完整checkpoint边界按已有生命周期退休；指定U、最终参数、raw事实、必要恢复与失败原件保留。

先核真实rank8/MT基底及原source/meta/专家上下文、非零A/B0的实际教师梯度、两种尺度消费者和四事件/均衡采样。
仅做必要的真实消费者/吞吐profile，不重复已核且未变的整个机制；profile参数不装入formal初值。
frozen base只读、meta/G/π/教师参数所有权、实际10ODE、finite/shape、保存恢复和新schema须由真实消费者承接。
physical query/frame chunk/worker/rank数按显存与真实吞吐选择，保持逻辑batch/数据流/任务权重；有明显余量须验证扩大而非沿保守默认。
遵守两节点live准入、合计8/闲卡≤10时6、单节点6；不跨节点拼同一训练，不因共驻高util自动排除，不操作他人进程。

实现/配置在独占`codex/`分支，验后集成main并push，从clean detached冻结版本formal运行；冻结源码不原地修改。
原teacher/schema不可当rank8 resume；同族完整恢复保存G/meta/π、optimizer、sampler/RNG、坐标、拓扑和版本，
设备/物理batch迁移保持更新语义并登记，正常BF16/TF32差异不做逐bit验收。原失败与原成本不覆盖。
接口违约由实验session在原授权范围修复并用新clean冻结代码接续；科学阴性不作为bug或参数小扫理由。

整批范围是16教师→G480→πlocal128/RL16→规定48行/U/视频干预；除明确停止线、资源或原则边界外直接完成，不逐模块等待main工程验收。
main据完整结果判断教师供给、参数到控制的传递、实际学生分布、用途选择与视频信息；π不能补救缺失候选，更多teacher也不能自动解决条件不可预测。
不无限续训或追加rank/seed/LR臂；需要后继时重新解释可失败预测并有界接续。
正常长任务持续等待退出事件，不周期查训练log/checkpoint、不心跳或阶段自Queue；只对整批完成/异常可靠回报一次。
tracked/Git由main完成本合同提交后正式交实验session独占；整批回报实际处理后再串行交回main科学消费并自主推进。
