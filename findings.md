# EMBER Findings

本文只记录跨实验仍成立的持久认知。精确分数与逐架构否决边界见`docs/research_history.md`，当前进度见
`progress.md`。

## 1. EMBER真正要学什么

teacher video应提供跨初始化成立的高层任务知识，而不是原demo低层轨迹：

- 任务涉及哪些对象、属性和关系；
- 最终目标状态是什么；
- 完成任务需要哪些动作阶段；
- 阶段之间的有向因果顺序；
- 哪些速度、路径、视角、抓取角度和扰动只是demo nuisance。

语言和视频的合理分工是Query与Value：语言规定关注点与目标，视频展示正确动态过程。language-only和video-only都
不是完整问题。

训练时video与action episode同task跨episode错开，可以阻断轨迹复制；但同task恒定target也会让模型仅学task
identity。架构必须正面提供same-task跨video过程可识别性。

## 2. 四个接口必须分开诊断

```text
language/video evidence
    -> high-level task Program
    -> native policy-effective LoRA
    -> policy action
    -> stable multi-task closed-loop success
```

历史反复说明，前一个箭头接通不代表后一个箭头成立：

1. 视频可改变hidden/LoRA/action，但可能使用错误phase shortcut；
2. same-task视频可形成共同表示，但compiler可能把它缩成identity或错误子空间；
3. native BA/action可以material，reward方向仍可能伤害held occupancy；
4. 一个checkpoint可以偶然高分，多一步更新又发生task换手。

分析必须定位最早失效接口，不能用“视频没学到”“LoRA不健康”或“task drift”一词覆盖全部原因。

## 3. 视频被使用不等于被正确理解

v4、K4 Trace、Grounded-Video、LPCP和GOMQ都证明视频能进入Program、LoRA或action。与此同时，历史多次出现
shuffled/reversed/wrong更好、correct无margin或同task更新稳定但held性能下降。

因此以下证据都不充分：

- correct与negative的latent距离大；
- reversal显著改变LoRA；
- video route attention不为零；
- same-task不同video correction cosine很高；
- negative LoRA被人为推坏。

最终证据必须是严格配对closed-loop中，correct沿有用policy direction稳定优于wrong、shuffled、reversed和
no-video，同时same-task-other不明显下降。

## 4. Few-shot的价值与边界

多视频的第一性原理价值是过滤单demo nuisance，并提供共同高层程序的可识别性。K4、Dynamic-K Direct-Family-B、
V6 Slot-Set和GOMQ已经多次显著降低same-task表示或BA方差。

但聚合只会稳定输入给它的东西。历史上K4也会把错误task mean稳定下来，甚至让closed-loop下降。因此：

- 每条video必须先独立保序；
- set聚合必须发生在有语义的Program层，而不是frames/raw features/final LoRA平均；
- 同时检查within-task alignment与between-task separability；
- dynamic K若被声称支持，训练必须覆盖每个cardinality；
- 不以增加K、挑video或平均LoRA救性能。

few-shot仍是开放方向，历史K4失败没有否定它。

## 5. 正确顺序是结构约束，不只是negative loss

correct展示从初态到目标态的物理可行过程；reversed颠倒因果，shuffled破坏阶段连续性。合理表示应在每条video
内部保留有向过程，在video集合维保持置换不变。

v5/v6证明Semantic Core与Procedure分离有价值：Core描述任务语义与目标，Procedure描述阶段演化。失败通常发生在
Procedure经过fusion/compiler后衰减，或模型只读取absolute phase。后续设计应保留“语义query + 有序动态Value”
这一分工，但不必机械复刻某个旧前端。

## 6. V6不是神秘高山，但其有效机制要继承

v6-fast达到143来自architecture×recipe的联合效果，不是单一模块：task-grounded Core、visual-transition
Procedure、task-complete训练和高增益native compiler共同成立。后续很多架构改善内部指标却丢失其中某个接口。

LPCP又表明：在不破坏V6主图的情况下，用同一次真实context forward中的layerwise Action probes读取有序视频，
可以保持143并扩到breadth7。它验证了carrier，但其更新仍太接近AS139且高churn。

所以未来既不应“回到v6什么都不改”，也不应因追求新颖性抛弃V6已经证明有效的native topology、factor ownership
和强baseline。每次修改要明确继承了什么、替换了哪个最早失效接口。

### 6.1 时序响应不等于使用了完整Procedure

LMMPC-v1给出了新的实现级反例：correct/reverse的parameter memory几乎严格反号，表面看起来具有极强顺序响应；但
reader只返回最后一个Procedure query的输出，并额外加入不经过Procedure的首尾memory差。macro25/50中attention只占
reader输出约`2.75%/2.46%`，endpoint是它的`41.0x/45.7x`；把整条Procedure替换成重复`P_last`，输出逐元素不变。

因此今后的有向过程门必须同时检查：内部阶段替换是否改变readout、是否存在endpoint/absolute-phase旁路、reverse
差异是否只是硬编码反号，以及这种差异是否最终沿有用closed-loop方向。正确结构应让固定policy地址读取完整阶段轴，
并让dynamic Value在Procedure无阶段信息时不能独立通过；不能只报告correct/reverse latent距离。

### 6.2 完整Procedure也可能被后置compiler重新抹平

LMMPC-v2删除endpoint旁路并让每个layer/rank地址真正读取完整Procedure，机制门通过；但macro25/50 strict只有
`71→73`。macro50 Core-fused grid仍有correct/reverse relative-L2`.2573`、between-task cosine`.3381`，同时
same-task不同K4 set cosine`.9922`。两层unbounded axial M2P却把前两项改成`.0938/.6560`，最终BA order差异仅
`.0862`；output RMSNorm单独几乎不改变这些指标。第一层输出相对输入改写`4.500x`，第二层再改写`1.753x`。

这说明“结构化M2P”也不能默认获得覆盖parameter-aligned Program的权限。memory grid已经包含layer/rank地址时，跨层/
rank通信应是受约束的refinement而不是新的主Value。只读bounded counterfactual表明，把每cell correction限制为anchor
RMS的`.25/.5x`可分别保留order`.2479/.2308`、between-task`.3608/.4056`，且same-task K4仍为`.9928/.9939`。
该证据只支持下一次局部实验，不替代fresh训练和closed-loop。

## 7. Memory token有真实价值，但不是目标

SHINE/Doc2LoRA式memory的核心价值是内容处理和目标参数层之间的结构对应，而不是token数量本身。对EMBER，
layer-matched memory可能把language/video Program放入policy topology，再由共享compiler生成LoRA。

历史Dynamic-K memory路线证明它可训练、可部署、支持动态K；CMBG/CFMG证明literal one-way layer memory可形成强
held跨video坐标；GOMQ更直接证明learned memory query曾带来显著closed-loop增益。

但GOMQ随后连续回落，说明“memory读得好”不等于“shared LoRA tail写得稳”。被否决的是当前memory + independent
rank32 direct-B tail + reward update组合，不是memory token一般。

还需区分两种coherence：GOMQ相邻更新的完整BA在four-K4 conditions下约`.993`一致，但隔离memory-only contribution
在held same-task videos上只有约`.127`。前者说明shared update没有被video-set相消，不能反推learned memory已经形成
完整高层Program；151只证明memory query有真实增量价值。

memory也不得为了形式而强塞入Action Expert。若调用原生backbone，必须保留有意义的图像、语言和原生prefix语义；
不能运行zero-image或无context action query后把输出称为policy grounding。

## 8. LoRA输出维度不是已证明的首因

Writer生成上百万LoRA参数确实是困难的结构问题，但现有证据不支持“参数太多所以失败”的单因解释：

- task-local rank16 experts有效；
- v6/LPCP的native rank16 compiler达到143；
- rank8 fixed-A失败主要是可达右子空间窄，不是rank8理论容量不足；
- dynamic full A/B和rank32 zero bank也没有自动提高闭环；
- uniform rank14 compression本身会损伤support。

合理方向是结构化生成：Program先进入与policy layer/target/rank坐标对齐的中间表示，再由共享compiler生成native
LoRA。rank8、rank16或其它rank应由性能和可达性证据选择，不是先验目标。owner当前接受继续使用rank16。

## 9. LoRA健康度只作定位

健康度可用于发现：

- 写出过小、近identity；
- 所有targets完全共线；
- native BF16量化后continuous residual消失；
- A/B参数化开放错误子空间；
- regeneration或compression破坏已有support。

但SFT experts本身常低stable-rank、q-dominant、跨列coherent。强制均匀能量、正交、高rank、更多atoms/lanes/
experts多次降低closed-loop。优先分析effective BA、fixed-action response和真实rollout，而非raw A/B gauge。

## 10. Task experts的作用边界

24个task experts到step2000取得`658/1200` direct success，23/24 tasks非零，证明task-local LoRA是有效task-level
target，也提供正常policy geometry和成功occupancy。

但同一task expert对所有videos恒定，因此不包含：

- same-task视频差异；
- correct与shuffle/reverse的时序证据；
- held task泛化；
- shared Writer多task共存。

Expert-Manifold reconstruction、routing和hard expert复现均未转化为held performance。experts可以继续作为train24
privileged teacher或geometry诊断，不能作为held dictionary、task-ID router或部署第二套LoRA。

## 11. Functional和reward surrogate长期错位

variance reduction、reconstruction、RLS row retention、success-key protection、negative guards、all-view margin、
endpoint preference和expert occupancy都曾让局部objective更健康，但closed-loop gained/lost仍不可分。

Reward并非无效：它可以形成Program gradient、native BA/action response，GOMQ甚至产生过151。但当前训练credit在
held on-policy occupancy上缺少可靠方向，并在shared update中受到cross-task conflict、norm dominance和policy
Jacobian差异影响。

所以：

- 可以继续考虑RL训练Writer，但每个surrogate必须尽快接受paired400裁决；
- 不用loss选择checkpoint；
- 同task跨video gradient一致只是必要条件；
- task-local descent不等于multi-task shared retention；
- 生成LoRA后的task-local RL仍是后续独立阶段。

## 12. Task drift是症状集合

task drift可能来自：

- 视频表示偏向nuisance或task identity；
- 同task不同video correction正交；
- 不同task functional/reward gradients冲突或幅度失衡；
- shared condition map把异质cotangent压成common update；
- Program到LoRA compiler丢失方向或量化为零；
- factorization、compression或online regeneration破坏已有support；
- offline query distribution与真实rollout occupancy错位。

实验已分别观察到上述现象，因此不能把task drift归为单一“容量不够”。更大模型可能改善表达，但如果credit和
坐标不对，只会更有容量地换手。

## 13. 截至当前真正解决和未解决的问题

已解决到足以复用：

- fixed split/source policy/information wall与strict evaluator；
- task-local effective LoRA参照；
- action-hidden有序video carrier；
- Dynamic-K和per-video/set数据流的工程可行性；
- V6 native rank16高增益compiler基线；
- literal/learned memory进入policy topology的可行性；
- reward→gradient→native BA/action局部链路。

尚未解决：

- 可验证的高层task Program，而非identity或nuisance；
- correct顺序沿有用方向形成稳定优势；
- same-task鲁棒与cross-task separability同时成立；
- 一个shared checkpoint持续积累多个tasks；
- 约145+相邻稳定且六臂合格的方法；
- Program到LoRA的可扩展、material又support-preserving的统一compiler。

LMMPC-v2还补充了一个明确负边界：完整stage reader、Dynamic-K与native rank16均接通，仍会被无上界后置M2P
重新变成task-common、order-insensitive方向；因此当前最早修复接口是Program-preserving compiler commitment，而非
继续增强Procedure loss、恢复negative训练或改变rank。

LMMPC-v3进一步表明，约束后置M2P不会自动解决上游集合表示：M2P改写已被限制到约`.25x`，macro25一度恢复到
`102`，继续到macro50却降至`60`。更早的unbounded K-set correction相对per-video mean仍改写`5.8--10.2x`，把
between-task cosine推到`.90+`且没有改善same-task coherence。Few-shot set模块必须以已经保序、已对齐地址的
per-video Program为anchor；learned cross-video correction可以存在，但不能默认拥有覆盖anchor的权限。

LMMPC-v4证明bounded K-set确实关闭了上述表示覆盖，却没有带来闭环突破：macro25/50 strict=`104→102`，breadth均6，
相邻checkpoint churn=`52`。25→50 effective-BA norm从`27.28`增到`49.08`且relative-L2=`1.44`，说明Writer并非没写，
但新增能力主要在task间换手。逐stage最早异常位于Procedure reader：macro50 raw Procedure的correct/reverse差异为
`.4350`，读出H_set只剩`.1844`；后续Core fusion/M2P/FactorHeads没有更早地制造该衰减。

与V6同口径比较排除了“Procedure趋同本身就是必须先修的根因”。V6 raw Procedure的between-task cosine反而高达
`.9973`、correct/reverse relative-L2仅`.1093`，其policy-routed、Core-conditioned slot reader却把有向差异放大到
约`1.30`；v4则只保留约`.43x`。因此下一局部变量不是增加reverse/matching/contrastive loss，而是让layer/rank地址
先从Core获得task-conditioned Query，再用完整有位置的Procedure作Keys读取centered native memory Values。该判断
已经在LMMPC-v5的clean macro2机制门得到支持：validation8 raw Procedure reverse差异几乎不变
（`.72035→.71525`），H_set却由`.51687`升到`1.30135`，reader/raw由`.7175x`升到`1.8194x`；H_set
within/between-task cosine仍为`.97045/.24495`。这只证明最早结构接口已修复，不证明方向对closed-loop有用；若
fresh macro25 strict仍无提升，最早断点才后移到shared functional credit。

LMMPC-v5进一步给出了闭环裁决：Core-conditioned layer/rank reader把validation8的reader/raw从v4的`.7175x`提高到
`1.7627x--1.8194x`，macro25 strict也从matched v4的`104`提高到`123`；400行严格配对为`38 gained / 19 lost`。
因此读取器修复是有效机制，而不是又一个只让内部指标变漂亮的改动。但v5相对LPCP143仍净丢20，成功的
`91.1%`集中在三个task；all400 BA已相对v4显著换向，却无法用norm、cosine或relative-L2区分gained与lost。
这把最早未解接口后移到`functional cotangent -> native FactorHeads -> held on-policy direction`。Procedure趋同仍可作
诊断，但在该接口解决前不应成为新的contrastive/matching训练目标。

v5 macro50把上述判断升级为稳定性终局。同一run继续训练使末5轮functional loss降到`.10962`、BA norm从
`27.23→44.00`、effective targets从`16.88→19.50`，strict却从`123→84`；严格配对为`13 gained / 52 lost`，
Object3和Goal6能力大幅流失而Long1增加。LoRA继续变强、变广并不等于共同积累。更关键的是，25→50 update在同一task
四组K4 conditions间有`.980--.996`的task-mean/sample energy，却在task mean之间只有`.360` cosine：当前问题不是
video-local correction正交或一个common update压过所有task，而是task-specific、cross-video coherent的functional
方向没有保留held on-policy support。

raw Procedure在macro50确实进一步趋同（between-task cosine`.9717`、correct/reverse relative-L2`.4683`），但
Core-conditioned reader把它放大到H_set=`1.0652`，compiled/effective-BA仍为`1.1619/.9515`；同task condition
cosine保持约`.99`。因此Procedure趋同不是本轮123→84的最早断点。当前证据把首要责任压到
`offline functional credit -> native factor coordinates -> held occupancy retention`；其中更像credit/retention问题，
但尚不能完全排除FactorHeads可达坐标对该credit的系统性扭曲。v5精确recipe终局，reader正机制应继承，不能把该负
结果扩写成memory token、Dynamic-K、rank16或LMMPC整体失败。

整理后的stage-wise裁决进一步定位：Dynamic-K的between-task结构曾首先在nonlinear family/B readout变同向；LPCP
冻结tail又把新Procedure压成AS139邻域小修；direct native/rank32路线打开写出后，shared reward仍不能保留held
support。LMMPC-v1进一步说明，即使建立layer/rank memory和共同native compiler，Procedure reader也可能被endpoint
旁路；v2/v3依次暴露M2P和K-set覆盖，v4用bounded commitments关闭这两处，v5再用Core-conditioned Query关闭reader
衰减。当前表示链已经能把task-specific、有向、cross-video coherent的Program material地写成LoRA，最早未解处因此
正式后移到functional credit与held support retention。下一变量应先区分loss提供的cotangent方向是否错误，还是
FactorHeads坐标系统性扭曲该方向；不能再回头增强已经通过的K-set、Procedure margin或memory容量。

## 14. 方法选择与实验原则

1. closed-loop absolute首先选方法，稳定性和视频因果性决定方法资格。
2. 每轮只改变一个主要因果变量，先写可证伪门。
3. 机制门回答“图是否接通”，paired400回答“方向是否有用”。
4. 好结果多训练到相邻稳定性有信息；坏结果不靠无限训练和小扫。
5. 报告per-task/per-suite/breadth/retained/gained/lost/churn，不只看aggregate。
6. 不用union、融合、挑video或task checkpoint冒充shared method。
7. 不为正常BF16/TF32、batch和kernel低位差异牺牲吞吐。
8. 负结果只淘汰实际组合；局部建议不能触发无证据的整套摇摆。
