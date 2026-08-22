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

- fixed split、source policy、information wall与strict evaluator；
- task-local effective rank16 LoRA参照；
- action-hidden有序video carrier；
- Dynamic-K和per-video/set数据流的工程可行性；
- layer/rank memory进入真实native policy context的可行性；
- bounded K-set/M2P避免后置模块无界覆盖上游Program；
- Core-conditioned reader从弱Procedure差异中提取parameter-addressed Value的闭环正收益；
- reward/functional gradient到native BA/action的局部链路。

尚未解决：

- 可验证的高层task Program，而非task identity、static cue或demo nuisance；
- correct顺序沿有用policy方向形成稳定优势；
- same-task video鲁棒与cross-task能力共存；
- 一个shared checkpoint持续积累多个tasks；
- 约145+相邻稳定且六臂合格的方法；
- Program到完整LoRA的material、policy-effective且support-preserving编译。

### 13.1 当前架构已经解决的局部断点

LMMPC历史revision依次暴露并关闭了三个明确结构问题：v1的Procedure endpoint旁路、v2的unbounded M2P覆盖、v3的
unbounded K-set覆盖。v4关闭集合覆盖后仍只有`104→102`；Core-addressed reader随后把matched macro25从104提高到
123，严格配对为`38 gained / 19 lost`，并把validation8 reader/raw correct-reverse relative-L2从约`.718x`提高到
`1.819x`。因此当前reader是正机制，而不是只改善内部指标。

### 13.2 Absolute缺口和checkpoint漂移是两个问题

当前完整轨迹为strict=`123→84→89→87`、breadth=`8→5→6→4`。后期回升没有恢复macro25：400个固定rows中只有
49行始终成功，macro25到50丢失的52行到macro100仅恢复15行。能力轮换已经由逐行证据确认。

但最佳点123本身也弱于同schedule LPCP143和GOMQ151。相对LPCP143，当前为
`100 retained / 23 gained / 43 lost`；GOMQ151相对同一LPCP为`126/25/17`。两者新增成功数接近，当前差距主要来自
更多旧success rows被替换。当前123相对GOMQ151为`100 retained / 23 gained / 51 lost`，28分缺口主要集中在Long
（-23）和Object（-12），并非所有task均匀减弱。

因此“task drift”不能同时代替两个概念：一是单checkpoint是否形成足够强且广的policy support，二是相邻训练更新
是否保留该support。历史151仍有第二个问题；当前架构两个问题同时存在。

### 13.3 当前归因边界

固定B20 loss在四点为`.112124/.099353/.098427/.101337`，25到50显著改善时strict净丢39。相邻compiled Program
relative-L2为`.770/.730/.710`；早期FactorHeads主导norm扩张，后期heads-only与Program-only BA变化已接近。
same-task四K4 updates保持约`.98--.996`能量一致，LoRA也持续material增长。

这些证据排除了“Writer没写出”“同task视频相消”“只需更多训练”“K-set/M2P仍无界覆盖”“单纯LoRA太小”作为
完整解释，也不支持把Procedure趋同或FactorHeads单独漂移当成唯一根因。

仍无法从现有实验唯一拆开的因素包括：静态B20 occupancy与真实rollout occupancy的错配、Program是否主要编码task
identity、Program到native A/B坐标的可达性，以及shared optimizer对跨task support的保存。当前没有六臂数据，视频
因果资格也保持未知。

耐久结论是：**同task跨video update coherent、Program有序且material、LoRA健康，都不足以推出shared held support
会积累；functional query distribution、compiler coordinates和optimizer必须作为相互作用的训练合同分析。**

## 14. 方法选择与实验原则

1. closed-loop absolute首先选方法，稳定性和视频因果性决定方法资格。
2. 每轮只改变一个主要因果变量，先写可证伪门。
3. 机制门回答“图是否接通”，paired400回答“方向是否有用”。
4. 好结果多训练到相邻稳定性有信息；坏结果不靠无限训练和小扫。
5. 报告per-task/per-suite/breadth/retained/gained/lost/churn，不只看aggregate。
6. 不用union、融合、挑video或task checkpoint冒充shared method。
7. 不为正常BF16/TF32、batch和kernel低位差异牺牲吞吐。
8. 负结果只淘汰实际组合；局部建议不能触发无证据的整套摇摆。

## 15. 固定提交外部复核补充的代码级断点

外部专家复核`947c0e3`后指出，当前`LayerMatchedBackboneMemoryEncoder`先正确detach frozen backbone的
`prefix_hidden/action_hidden`，经fresh Writer-local `language_projection`、`patch_grounding`和
`interaction_projection`得到逐帧evidence后，又在返回`LayerMatchedVideoEncoding`时把
`frame_evidence/grounded_evidence/interactions`全部detach。仓库侧逐行核验确认：

- source policy hidden的第一次detach足以阻止主干梯度；
- 后一次输出detach额外切断了`patch_grounding`和`interaction_projection`的全部functional credit；
- `language_projection`仍可从独立text query与后端language gate获得梯度，但其逐帧视觉使用分支被切断；
- Core、Procedure和后端仍能在这些固定随机投影特征上学习，所以这不等于视频路径完全无效；
- 现有梯度测试检查FactorHeads、memory token、Reader、K-set与M2P，没有检查上述两个fresh前端模块；
- 历史V6 semantic forward返回同类evidence时没有这层输出detach。

这将“最早未解接口”从宽泛的functional occupancy进一步向前扩展为一个可直接检验的上游credit断点。它与后续
occupancy/retention问题可以同时存在：该断点更可能限制macro25的absolute support形成，而25到50在offline loss继续
改善时崩落仍需要occupancy、shared credit和decoder co-drift解释。没有matched干预前，不把detach、FactorHeads或
occupancy任何一项写成唯一根因。

同一复核还澄清了若干结构边界：Reader的“减首帧再减时间均值”代数上等价于单纯时间中心化；memory rank只实现
index correspondence而非已知policy-functional correspondence；Action probe把50 tokens直接均值会丢失horizon
结构；M2P输出RMSNorm会消除cell magnitude；action-in/out坐标由首末expert layer派生；FactorHeads每个宽输出row
位于共享256维末层子空间且fresh identity产生B-first信用打开。这些都是合理诊断假设，不是单靠代码即可裁决的
性能根因。

owner同时明确后续不再使用Text Meta-LoRA。当前sealed formal确实使用rank4 Text Meta-LoRA、VL Meta-LoRA为0；
后续语言仍必须通过冻结原生text/VLM表示进入Writer，Action Meta-LoRA按独立作用继续判断。外部复核的完整根因排序、
反证、建议门槛和远程证据缺口记录在`docs/external_review_20260818.md`。

## 16. 外部复核逐项实验后的耐久结论

### 16.1 代码级credit断点真实，但必须拆开“因果资格”和“absolute性能”

旧A与no-Text B在heads已经打开的macro1/macro25仍对`patch_grounding`与`interaction_projection`完全无functional
gradient；C只移除Writer-local projected outputs的第二次detach后，两组在macro1首次获得nonzero finite gradient，
source policy仍为0。这个工程断点及其测试缺口已经关闭。

但matched B→C只把correct strict从104提高到110，breadth@1/@5/@10仍为`6/3/3`，C继续到macro50为101，发生
`24 gained / 33 lost`、churn57。它不是absolute低上限或shared retention的主要单因。controls却显示另一层真实作用：
B的correct-reverse只有`104-96=8`且不显著，C变为`110-69=41`且显著；correct-wrong也由39扩大到56。
因此恢复credit明显改善了视频内容与整体正向过程的资格，但没有把这些方向转化成足够强、广、稳定的held support。

### 16.2 Text Meta-LoRA提供真实但混合的support，不是literal language-only也不是科学上干净的正机制

A→B只移除Text Meta-LoRA时，correct从123降到104，证明Text路径有真实闭环贡献；但shuffled从122降到83、
shuffled-keep-first从131降到90，降幅远大于correct，使correct-shuffle margin从1变为21。与此同时reversed从90升到
96，使correct-reverse margin从33缩到8。Text路径既帮助correct相对reverse，也更强地支撑order-corrupted视频，不能被
简化为“全是shortcut”或“全是有用语言”。未来canonical不再使用Text/VL Meta-LoRA，但必须另行恢复它曾提供的
absolute support，而不是把清理本身当成方法提升；exact language仍通过冻结原生表示进入Writer，Action Meta-LoRA保留。

### 16.3 aggregate视频因果性存在，但仍集中且没有达到跨demo稳定高层Program

C macro25为correct/same/wrong/shuffle/keep-first/reverse/no-video=`110/111/54/91/93/69/47`；B的对应面板为
`104/101/65/83/90/96/47`，correct-no-video净`+57`且显著；F5 PCGrad为
`107/111/51/92/105/53/47`。C对wrong、shuffle、keep-first、reverse、no-video分别净`+56/+19/+17/+41/+63`，
均达到显著；F5除keep-first仅`+2`外，其余为`+56/+15/+54/+60`并显著。说明当前no-Text Writer不是language-only，
正确视频内容、全局箭头方向和部分中间顺序都能影响有用policy方向。

这还不等于通用高层过程理解。C的no-video→correct净收益几乎全部来自Object（+59），correct-reverse在Long反而
`-12`；C的keep-first顺序优势也主要来自Object。F5把correct-reverse优势扩到Object/Goal并提高macro25 suite floor，
却把correct-keep-first margin压到2，说明“首帧/端点/正反方向”与“中间阶段有向连续关系”必须分别控制。四个arm的
same-task-other总分都在correct±10，但correct-success row retention仅A/B/C/F5=`85.37/83.65/87.27/85.05%`，
均未达到专家建议90%；aggregate相近会掩盖具体初始化换手。

### 16.4 head漂移是放大器，fixed-head reachability也是真实瓶颈

冻结A macro25的八个FactorHeads续训到macro50得到117而正常续训为84，证明head co-drift显著放大早期崩落；但固定
heads仍相对123丢33个success rows。首次F4评测曾错误地没有把投影LoRA安装到policy，所得`659/1200`无效；修复
wiring并从同一clean authority重跑1200行后，完全固定heads的free-Program投影只有`307/1200`，direct experts为
`658/1200`。严格配对为253 retained、54 gained、405 lost，Jaccard `.35534`，projected/direct仅`46.66%`，明确未过
预注册90%门。effective-BA relative L2均值`.93571`也不再与闭环证据矛盾。

因此F3只证明co-drift是放大器，正确F4同时证明A-macro25 FactorHead manifold的reachability不足；不能再以旧结果排除
decoder/head坐标。后继路线仍优先检验固定、功能锚定decoder，因为它正面处理moving coordinate问题；nonheld-meta
held-task闭环将决定当前code width/decoder是否足够，若仍失败则需要架构性扩大或重参数化，而不是小扫。free Program
是privileged接口诊断，不是held route。

### 16.5 简单occupancy故事和arithmetic mean单因均未通过

A macro25→50的136个lost/gained/retained rows在两checkpoint真实occupancy上均从首次replan就分歧。lost rows在
macro50自身occupancy的checkpoint disagreement没有按假设变大，gained反而显著变大；没有validation expert且读取held
teacher action违反信息墙，所以哪个动作更正确只能记为`underdetermined-after-audit`，不能用disagreement冒充error。

PCGrad相对C把25→50 lost从33降到25、churn从57降到39，但lost改善不显著，同时gained从24降到14且该抑制显著，
score为`107→96`而C为`110→101`，macro50两者都收缩到breadth@1=4和top-3约98%。PCGrad在macro25有更好的
breadth@5与suite floor，并强化wrong/reverse/no-video区别，却丢失keep-first中间顺序优势。cross-task gradient conflict
会改变能力分布和换手方式，但standard PCGrad没有解决shared accumulation，也不是absolute上限的主要解释。两臂都保留
AdamW，故Adam moment独立效应仍不可由本实验裁决。

### 16.6 当前最早未解接口

经过上述反事实后，不能再把当前失败主要归因于“前端没梯度”“LoRA没写出”“rank16太大/太小”、
“简单self-occupancy divergence”或“arithmetic mean必然错误”；但修正后的F4说明FactorHead/decoder reachability不能
排除，而且是已实证的接口限制。absolute问题至少同时落在两处：固定输出坐标能否覆盖policy-effective directions，
以及`language/video/Action/memory -> learned code`能否从未见task证据预测这些方向；稳定性问题则落在shared objective与
有限长更新能否保留它们。三者相互作用但必须分别报告。当前没有任何新arm达到约145或相邻稳定资格，不能把更好的
controls当成性能pass，也不能因absolute低而抹掉已验证的因果改进。

## 17. 第二轮独立审查后的后继因果模型

第二轮审查没有否定EMBER的科学问题，而是把失败定位为训练问题的参数化：当前系统要求一个仅见24个task映射、
correct-only且同task target恒定的video/language encoder，与一个持续移动的complete-LoRA decoder共同发现潜在坐标，
再用不直接评价closed-loop success的expert-state functional objective驱动它。这个组合同时产生三类欠识别：

1. 同一成功策略存在大量parameter gauge，raw A/B或rank cell不是唯一功能坐标；
2. object、goal、affordance、端点与粗方向足以解释多数训练监督，完整多阶段process没有被要求成为必要变量；
3. 即使code局部有用，moving decoder和shared optimizer也会改变它对应的policy方向，造成absolute support弱与相邻换手。

F3/F4应据此重新解释：F3证明冻结heads能缓解co-drift但不能保持support；修正后的F4证明在A-macro25固定heads中，
即使每task自由优化Program也只能保留direct expert的46.66%，所以输出坐标本身不足。该oracle仍不回答未见task的
视频/语言能否预测code，故后继路线必须把“固定decoder可实现性”和“evidence-to-code推断”设为两个独立gate。F2只
否定一个关于self-occupancy disagreement的窄故事，不能外推为closed-loop distribution不重要。F5说明raw parameter
conflict会改变能力分布，但standard PCGrad会同时消除有益与有害task-specific directions，因此不能代替功能坐标或
闭环外目标。

后继路线的最小因果分解是：

```text
successful adapters / task experts
    -> policy-functional fingerprints
    -> fixed compact code-to-complete-LoRA decoder

exact language -> learned prior z_L
action-hidden ordered videos -> process evidence -> posterior delta(L,V)
z_L + delta(L,V) -> fixed decoder -> one complete LoRA -> closed-loop success
                                      ^
                          train/meta rollout outer credit
```

其中language-only必须成为learned baseline，不能由架构强制identity；视频的科学价值是相对该prior提供可复现的净增量。
旧A/B/C没有覆盖`Text Meta-LoRA + repaired front-end`第四格，因此不能从那组三臂推出语言先验无用。owner随后禁止新
canonical继续使用额外Text/VL Meta-LoRA，故不为补齐旧析因而重启已退役LMMPC；后继在同一固定decoder上用learned
language-only、video-only和language+video重新回答语言独立贡献与视频条件增量，这保留了专家批评中的实质信息。
Program首先应表达object/relation、initial/goal、contact event、ordered subgoal与completion condition，再映射到固定policy
code；50个Action tokens、LoRA rank index、首末policy layer和时间中心化memory都不能未经功能验证就被当成过程坐标。

24 target-train tasks不足以同时发现functional manifold和task inference规律。固定24/8/8 target ID继续保留，但需要从
不与validation/test语义重叠的LIBERO-90 non-held tasks建立显式meta pool和task-level folds，分离source skill
pretraining、video-adaptation meta-training、architecture validation与final Test。train/meta actions与reward可服务inverse
dynamics、phase alignment和closed-loop outer objective；validation/test deployment仍严格action-hidden和zero-interaction。

这一路线不是为了速度舍弃其它意见。shared base adapter + video residual可在rollout前merge为唯一LoRA；sealed held
action/reward诊断、runtime video policy、生成后task-local RL、更丰富传感器以及video-to-reward/skill/plan等方向都有
明确位置。它们按所改变的科学合同和触发证据分层进入，而不是在核心fixed-decoder因果链尚未受检验时并行堆叠。

当前LMMPC 320-cell joint-moving grid的增量路线到此停止；可继承的是严格评测、每视频保序/集合聚合、Core/Procedure
职责、Core-addressed读取和bounded refinement等已有正证据，不继承其具体坐标作为架构信条。

## 18. Fixed functional decoder的首个机制证据

train24 fold0非正式profile第一次把“参数作用”和“真实policy响应”放在同一decoder上配对比较。19-task decoder在
gauge-invariant随机输入`BA·probe`上由identity-relative `1.000`降到`0.447`，冻结decoder后5个未见task仅优化code可到
`0.805`；但把同一warm-start放入完整PI0.5 Action Expert 50-token flow response，独立episode初始仍为fit `0.999`、
held `1.008`。因此低层有效更新相似度最多是便宜预热/定位面，不能作为functional equivalence或模型选择代理。

直接用expert相对source的完整flow增量训练后，仅2次/task decoder更新已把独立fit loss降到`0.833`，仅5次/task held
code更新把未见task loss降到`0.933`；18/19 fit和4/5 held优于identity。该结果证明完整policy-functional梯度、固定
decoder和新task code三者已同时接通，并给出继续训练的合理信号；它仍是dirty-worktree、单fold、单train/eval panel、
无closed-loop的机制profile，不能声称decoder range通过，更不能据此选择最终架构。

## 19. Train24 fixed decoder的正式闭环裁决

同一fold0 decoder在clean pushed commit上完成24 tasks × 50 states formal rollout。单一fixed-decoder投影为
`388/1200`，direct experts为`658/1200`；严格配对得到332 retained、56 gained、326 lost，success-set Jaccard
`.46499`。19个decoder-fit tasks是`326/950`对direct `550/950`，5个冻结decoder后只拟合code的held tasks是
`62/250`对direct `108/250`。两侧保留率相近，说明主要限制不是单独的held-code optimizer，而是当前32维
code + shared fixed decoder尚不能稳定表达train24 task-expert的policy-effective功能。

因此首个formal realizability gate明确失败。这个负结果淘汰的是“当前train24 fold0训练合同已经足够”，不是淘汰
fixed decoder思想本身：专家指出24 tasks不足以同时识别功能流形和推断规律，owner也已授权使用去重后的LIBERO-90。
下一步按预注册顺序训练56 meta-train / 15 meta-validation-oracle的non-held expert family，再以真正task-level held
closed loop复验；不通过rank、scale、seed、dtype或flow-loss小扫掩盖本次失败。

## 20. Non-held meta source prior的正式覆盖与新欠识别边界

default fold0的15个meta-validation tasks已在完全相同的50个固定states上完成frozen source formal：`646/750`
（86.13%），15/15 tasks非零，14/15 tasks达到40--50/50。唯一明显缺口是task73 `4/50`；它贡献46/104个总失败，
其余14 tasks合计`642/700`。因此当前证据不支持在fixed-decoder gate之前先强化source，且说明source具备广泛LIBERO-90
闭环能力。

但这15个task identity都参加过71-task source训练，因此高source分数不能证明meta adaptation或decoder泛化。后续必须在
相同750行严格比较source、uniform-step direct experts与fixed-decoder oracle code：既看source成功的retention，也看direct
相对source的gained/lost及projected对这些增量的复现。若direct experts不能产生跨task的policy-effective增量，则这个
重叠source/meta panel对decoder range是欠识别的，应转向role-disjoint meta-task构造，不能让identity LoRA凭高base分过门。
remote-safe结果见`docs/evidence/functional_adaptation_20260819/nonheld_meta_source_baseline.json`。

## 21. 完整71-task source覆盖：总体充足，Study缺口局部化

补齐default fold其余56个meta-train tasks后，frozen source为`2272/2800`（81.14%）；与fold0的`646/750`合并，
71-task总计`2918/3550`（82.20%），71/71 tasks非零，52/71达到40--50/50。按场景分解为Kitchen
`1596/1850`（86.27%）、Living Room `808/900`（89.78%）、Study `514/800`（64.25%）；按任务族分解时
single-place为88.67%、actuation为87.33%、stack为83.00%、compound为76.00%、pick-place为74.67%。9个低于
25/50的task中7个是Study pick-place，最低为task73 `4/50`、task80/81各`7/50`。

因此“当前source整体缺乏LIBERO控制能力”不成立，不应在fixed-decoder gate前做无差别source强化；但Study/book-placement
是明确的局部能力缺口，direct experts必须证明能在这些弱task上增加功能，同时不破坏source已成功的rows。若direct本身也
失败，最早失效接口是source/task-expert ceiling；若direct有广泛净增量，才进入fixed-decoder复现。由于全部71个task identity
都参加过source训练，这个panel仍不是source/meta role-disjoint证据，paired增量门不能省略。remote-safe结果见
`docs/evidence/functional_adaptation_20260819/nonheld_meta_source_coverage_71.json`。

## 22. Non-held direct ceiling成立，fixed decoder仅获有保留条件的推断资格

71-task uniform-step expert bank完成后，default fold0在完全相同的固定states上形成了source/direct/projected三联面板。
56个meta-train tasks由`2272→2519→2451/2800`：fixed decoder相对source净增179，保留direct successes的92.62%，
34 tasks正、11负、11持平；Study与pick-place分别净增99与102。15个meta-validation tasks由
`646→684→659/750`：projected保留direct successes的90.79%，15/15 tasks都有breadth@10，弱task73从`4→28→15/50`。
这同时排除了“direct experts在高source上没有增量”和“decoder只靠identity aggregate过门”两个担忧。

但held source→projected只有612 retained、47 gained、34 lost、净`+13`、exact McNemar `p=.18208`；6 tasks正、
6负、3持平，而且只复现direct gain rows的54.67%。因此本fold当时登记为
`qualified_pass_to_writer_inference`并只授权首轮language/video实验，而没有把fixed decoder升级为方法贡献。macro10之后
的§25进一步证明train/held code尺度不一致，故该资格现已降级为“decoder range值得用统一坐标重验”；单fold、privileged
free held code不能再充当Writer code泛化证据，多fold复现、task58退化定位与最终video inference仍不能省略。
完整裁决见`docs/evidence/functional_adaptation_20260819/nonheld_meta_fixed_decoder_fold0_20260820.json`。

## 23. Successor Writer训练面已满足正式启动条件，但还没有推断证据

successor Writer的首个有效profile在Text/VL Meta-LoRA均为0、冻结π0.5 VLM和fixed decoder均无trainable parameters的
合同下完成。它以8,218,216个Writer参数在4-task macro1得到total loss `1.828327`、finite gradient norm `2.490839`；
A40峰值32.52 GB、实际训练段36.82秒。此前profile暴露并修正了autocast下probability BCE和过大的frame microbatch，
也及时阻止了旧Text/VL Meta-LoRA配置重新进入后继路线。

随后fresh 56-task、6-GPU formal在macro10自然停止：780.43秒内total objective由`2.082406→1.853265`，
combined code由`.995820→.838643`、language由`.994485→.861964`、video由`.984880→.974670`，全程finite，峰值
32.60 GB。macro10的真实video→完整LoRA profile在同一32-entry最长优先面板上比较B8/B16/B32，三者均稳定且零OOM；
吞吐为`.898928/.897234/.896828 LoRA/s`，因此按预注册规则封存B8，而不是凭最低显存或形式上的大batch选择。

这仍只构成训练、authority和部署吞吐证据，不构成language prior、video posterior或closed-loop pass。下一科学节点是
15-task matched language/video/process screen；不能用下降的训练loss、free code或profile throughput代替。

## 24. Macro10推断门失败：高分来自共享均值，不来自正确视频过程

15个meta-validation tasks的固定前10 states上，macro10得到correct/language-only/video-only/first+final=
`131/130/134/130`；补齐reversed/shuffled/static-first-repeated后为`134/133/132`。correct相对language只有5 gained、
4 lost，relative source则8 gained、8 lost；相对video-only和reversed都只有1 gained、4 lost。task73为`3/10`，但没有
一个过程条件优于它；task63反而由correct `8/10`降于video/reverse/shuffle/static的`10/10`。因此full video既没有
language净增量，也没有order/dynamics优势，Gate 3明确失败。

这不是“aggregate差三分但方向正确”。在同一fixed decoder输出上做gauge-invariant effective-BA诊断时，15个
projected task adapters彼此本就很接近；video-only的task-mean adapter距它们的共享均值只有`.000998` relative error，
15个任务中仅1个最近邻到自己的projected adapter。correct、language和first+final也分别只有0/1/0个任务最近邻正确。
因此video-only的`134`主要复用了一个跨task carrier以及高source competence，不能证明视频识别了目标task，更不能用它
启动outer RL。完整配对与per-task证据见
`docs/evidence/functional_adaptation_20260819/writer_macro10_inference_gate_20260820.json`。

## 25. 旧fixed-code合同把train whitening与held near-zero自由码混为同一坐标

macro10负结果定位出一个此前被Gate 2 aggregate掩盖的合同错误。decoder-fit的56个codebook vectors经过gauge loss后
coordinate std约`1.000`、平均task norm`5.589`；冻结decoder后，15个held codes却从全零开始，仅40次/task Adam更新，
最终coordinate std`.089`、平均norm`.505`，到最近train code的平均距离`4.523`。Writer只接受前一种targets监督，
leave-task-out却由后一种near-origin自由码定义“正确答案”。这两组数不是同一分布上的固定功能坐标。

near-origin code经decoder产生的非零共同输出又恰好形成shared carrier，所以held free-code projected与macro10 video-only
都能在高source面板获得较高absolute；这解释了“closed-loop看似qualified、task inference却1/15”的组合。原fold0
source/direct/projected rows仍是有效的range和retention证据，但不能再授权Writer泛化，也不能通过延长旧Writer训练修复。

修复必须回到专家原始阶段1：让所有experts在同一meta-train anchor queries上产生`expert-source`完整flow response
fingerprint；PCA与whitening只拟合56个meta-train fingerprints，15个held fingerprints只通过同一个固定变换。这样train与
held code天然同坐标、同尺度，且每个维度来自统一policy function变化，而不是独立task-ID自由优化。随后固定这些codes
重训decoder并重新做source/direct/projected closed loop；只有新Gate 2通过才fresh训练后继Writer。

## 26. Unified fingerprint修复了坐标，但flow-only Decoder仍未恢复闭环功能

统一fingerprint formal已在完全相同的8个meta-train anchors上收集71个expert-source flow responses，feature width为
`25600`。PCA/whitening只拟合56个meta-train tasks：32维解释train variance的`.887516`；train/held coordinate std分别
为`1.0000/.7248`，平均norm为`5.5701/4.1436`。因此旧版held near-zero free code的坐标错误已被实质修复，而不是继续
由shared carrier掩盖。

固定这些codes后，flow-only Decoder在56-task fit panel从`1.040443→.445721`，15-task held panel从
`1.035567→.664218`，14/15 held tasks改善且held没有梯度；但严格相同750 rows的closed loop只有`644/750`，低于
frozen source `646`、direct experts `684`和旧free-code projected `659`。相对source为606 retained、38 gained、40 lost，
净`-2`、churn78、exact McNemar `p=.909946`；相对direct净`-40`且`p=3.67e-5`。弱task73仍为`4/50`，没有复现direct的
`28/50`。所以统一坐标是必要修复，但并不足以让task-local flow response surrogate产生可部署LoRA；不进入56-task
closed-loop复评，也不启动Writer。

最早失效接口进一步由gauge-invariant effective `BA`定位：相对direct experts，新fingerprint Decoder的平均
relative-L2为`2.8576`、cosine仅`.0254`，生成update norm是direct的`2.7004`倍；旧free-code Decoder也有
`2.4616/.0240/2.2718`。也就是说，低flow loss对应的是一个巨大、近乎正交且离开expert adapter support的off-manifold
解，而不是有效update恢复。专家C3关于offline functional surrogate与closed-loop错位的判断因此获得直接证据。
下一次只改变Decoder训练锚点：固定同一fingerprint codes，改用gauge-invariant effective-update probe拟合direct expert
`BA`；flow panel保留为无梯度诊断，最终仍由同一750-row closed loop裁决。shared-zero carrier同时作为matched架构对照，
用于区分task-conditioned code贡献与Decoder共享输出。

shared-zero formal为`640/750`，相对source净`-6`；从shared-zero换成task fingerprint code只净增`+4`，30 gained、
26 lost、churn56、`p=.68888`。因此flow-only Decoder的`644`几乎全由共享输出解释，task-specific code没有形成可靠功能；
shared carrier只保留为诊断，不作为架构fallback。

首个8个固定effective-update probes的Decoder也已裁决：fit/held probe loss为`1→.610951/.911115`，但全空间effective
`BA`在meta-train/held的mean relative-L2仍为`1.1387/1.1292`、cosine仅`.0642/.0449`。连训练tasks都没有恢复expert
support，说明下降主要来自固定probe方向过拟合；未浪费750-row闭环。下一实现使用低秩Gram恒等式计算exact full-BA
Frobenius loss，不物化dense `BA`，不改变codes、Decoder拓扑、task schedule或下游裁决。

## 27. Exact-BA改善参数几何但闭环更差，当前Decoder objective family关闭

低秩Gram exact full-BA Decoder在同一56/15 fingerprint、拓扑与1120-step schedule上，把train/held exact loss从约1降到
`.71068/.92128`；materialized BA的mean cosine也从fixed-probe的`.064/.045`提高到`.536/.303`。但15-task×50严格配对
closed loop只有`638/750`：相对source `646`为608 retained、30 gained、38 lost、净`-8`；相对direct `684`净`-46`、
`p=1.96e-5`；相对shared-zero `640`也净`-2`。task73仍只有`6/50`。step280/560/840/1120的held exact loss为
`.92575/.92002/.91853/.92128`，说明held已早期平台化，不是训练不足。

作为不训练任何Decoder的上界定位，以32维fingerprint做full effective-BA Hilbert空间无正则仿射最小二乘，train可达
relative-L2`.5244`、cosine`.8439`，held却只有`.9797/.3648`；design condition number仅`1.009`，不是数值病态。
因此当前失败不是再换一个factor gauge、probe数或训练时长即可解决：单个task expert的BA不是唯一功能标签，32维code也
没有在source/meta重叠任务上识别held expert update。flow/probe/exact/affine这组组合整体关闭，不继续objective/LR/seed小扫。

回看专家原始意见后，下一最早接口是数据与功能等价类：当前71个meta identity全部参加过source训练，source已达646，
direct只到684，而各Decoder几乎由shared output解释；同时每task仅一个expert标签，无法估计成功adapter等价类。后继先用
预注册step2000的sealed validation8 task-local oracle裁决target rank16 ceiling，再以source未训练过的train24和现有多
checkpoint successful adapters重建role-disjoint manifold。该oracle不更新共享模型、不选择checkpoint、不读取Test。

## 28. Role-disjoint与denoised action response把问题进一步定位到成功expert ceiling

target train24 fold0第一次真正分开了source-skill与adaptation-meta身份：冻结source从未在这24个task上训练，只用19个
fit tasks建立统一anchors、PCA/whitening和仿射映射，5个held tasks只做固定变换。flow与10-step denoised action两种
fingerprint的held coordinate std分别为`.7816/.7884`，平均norm为`3.333/3.292`；因此改善不是由held code塌到零点产生。

step2000单expert标签在held上的flow/action cosine为`.4310/.4341`、relative-L2为`.9509/.9451`。这比旧source/meta
重叠面板的无训练仿射cosine `.3648`更好，支持专家关于数据角色混叠的诊断；但两次诊断的任务集合也不同，所以这只是
方向性证据，不能把`.07`差值全部归因于角色隔离。19个fit tasks仍分别达到`.9688/.9510` cosine，train到held的落差
依然很大，不能据此重启fixed decoder或Writer。

把step250/500/1000/1500/2000中达到`25/50`成功阈值的checkpoint视为同一task的候选成功集后，aggregate held
cosine仅由flow/action的`.4310/.4341`提高到`.4355/.4394`。这组checkpoint来自同一条优化轨迹而非独立seed，故结果
只说明“沿一条轨迹平均若干成功点”不是当前主要解，尚未真正估计成功policy等价类。

更有信息量的是按ceiling分层。held tasks 0/9/18存在至少一个达到阈值的local expert；在这三项上action-response
prototype达到cosine `.5942`、relative-L2 `.8394`，相对flow的`.5753/.8712`进一步接近预注册`.60/.85`几何screen。
tasks 25/36没有任何checkpoint达到阈值，只能使用best-ceiling fallback，action-response仅为`.2071/1.0766`，直接拖低
全5-task aggregate。这说明action response与role separation确实恢复了部分可识别结构，但表示质量和expert/source
闭环能力仍纠缠；不能用失败expert的参数更新来否定成功策略流形，也不能把fallback称为成功等价类。

因此下一步不是继续Decoder objective变体，也不是立即大规模fresh source重训。先完成已经预注册的validation8
step2000 task-local rank16 oracle并与frozen source `48/400`严格配对：若local expert ceiling广泛且明显更高，最早接口
转为成功/on-policy occupancy、JVP与stage behavior上的功能标签；若ceiling本身仍低或集中，则先修source primitive或
local adaptation能力。完整remote-safe数值与边界见
`docs/evidence/functional_adaptation_20260819/role_disjoint_manifold_20260821.json`。

## 29. Policy Jacobian标签可诚实计算，但单样本信号不能升级为目标

按专家原始功能表示清单补做了一个不参与选择的PI0.5机制smoke：在固定source/task0-step2000 expert、同一action query、
Gaussian flow point、Beta(1.5,1) time和unit-RMS方向上，对完整noisy action sequence计算`torch.func.jvp`，不是有限差分。
source/expert输出均为完整`1×50×32`、finite，峰值显存`17.44 GiB`；JVP RMS为`1.10755/1.10828`，两者差的RMS
为`.028757`，cosine为`.999663`。这证明A40上可用一个冻结policy计算精确、配对的local response，不需要读取rollout
reward或把LoRA参数变成梯度目标。

该单样本同时说明不能把JVP本身当作答案：expert-source变化material但远小于共同response，而且尚无held geometry或
closed-loop预测证据。因此当前只保留统一owner；若validation8 ceiling广泛成立，再在少量successful/on-policy states上
把action、JVP与stage behavior配对比较，不运行71-task全量JVP fingerprint，也不据此重启Decoder。机制证据见
`docs/evidence/functional_adaptation_20260819/policy_jvp_feasibility_20260821.json`。

## 30. BDDL谓词可在同一rollout提供sealed阶段代理

为落实专家关于“failure从哪个阶段开始”的诊断要求，evaluator新增可选的formal-validation-only BDDL谓词change-point
capture。它在dummy settling后记录初值，此后只在真实执行动作使goal predicate布尔向量变化时追加step；每行保留
predicate labels、ever/final、peak count与完整change points。该路径不改变policy输入、动作、success、RNG或checkpoint，
不读取teacher action/reward，也不产生梯度；它明确使用privileged simulator state，因此只能作为owner已授权的sealed
held diagnosis。

真实LIBERO smoke在`libero_10/task1/state0`解析出cream-cheese与butter各自进入basket的两个谓词，settling后均为false；
synthetic rollout验证`step0 [F,F] → step1 [T,F] → step3 [T,T]`并保持原occupancy与普通rollout测试通过。边界是：BDDL
goal state只是无序最终合取，能定位部分完成、丢失与最终失败，却不能冒充完整有序procedure或recovery标签。下一次
validation8 step2000 strict400会在不额外rollout的情况下同时收集该trace。机制证据见
`docs/evidence/functional_adaptation_20260819/stage_predicate_capture_smoke_20260821.json`。

## 31. Validation8 local oracle广泛通过，首因转为successful/on-policy功能标签

八套彼此隔离的validation8 rank16 task-local LoRA均按统一合同训练到step2000；step1000没有评测或选模。clean detached
`5fd224a`上的strict paired400为`250/400`，相同rows与RNG合同的frozen source为`48/400`。source到oracle共有43 retained、
207 gained、5 lost、145 retained failures，净`+202`、churn212；八个task全是正净增量，四suite全部非零，breadth@1/@5/
@10均为8。oracle的Spatial/Object/Goal/Long分别为`73/78/58/41`，所以预注册的`>150`、至少5个正增量task、四suite
非零强门明确通过。该结果否定“target40的source/local rank16 ceiling整体太弱，以至于当前应先重训source”的分支；它不
证明一次视频编译可达，也不把held oracle变成deployment route或共享监督。

stage trace进一步把Long失败分开：task1中第一对象ever为31/50、第二对象ever为13/50，而full peak和最终成功均12/50；
task2第一阶段stove-on为50/50、第二阶段moka-on-stove与成功均29/50。因而当前主要缺口至少包含第二阶段完成与阶段保持，
不是所有Long primitive都不存在。Goal task3的BDDL只暴露最终合取，无法观察打开drawer等中间动作，提醒我们不能把final
predicate代理扩大成完整procedure annotation。

回到专家原始意见，单expert在离线anchors上的flow/BA几何已经不足以定义“成功策略功能等价类”，而成功/on-policy
occupancy、action response、Jacobian与stage behavior才是尚未完成的关键标签。因此下一步固定为8条non-held成功轨迹的小
面板：四个有source→direct正增量的task，各取一条gained和一条retained-success最短成功轨迹；每轨迹按8个进度strata选
最大expert-source action差异点，联合比较source-subtracted denoised action、exact JVP与stage。所有8条必须原样复现成功，
不按结果替换；至少3/4 task的same-task两轨迹必须比cross-task更近且不能只由final predicates解释，才把相应response
family升级为新manifold标签。JVP可以补充action，但不能独自覆盖失败的action geometry。

完整400-row pairing、per-task/suite、transition与stage证据见
`docs/evidence/functional_adaptation_20260819/validation8_task_local_oracle_step2000_20260821.json`。

## 32. Successful occupancy是必要条件，但直接串联action/JVP仍不是task-invariant code

预注册non-held小面板的8/8 direct-expert rows均在clean detached `febdff0`上复现成功，未替换任何row；所以后续负结果
不是由失败expert或best-ceiling fallback混入造成。每条trajectory被分为8个有序progress strata，并在每段选
expert-source executed-prefix action disagreement最大的replan。clean detached `1e45c66`随后对64个selected states配对
重算完整`50x7` denoised action delta和exact `50x32` action JVP delta。

直接把八段按顺序串联后，action只有task23和task86的两条trajectory互为全局cosine最近邻，即`2/4`；task26的gained
trajectory最近邻是task86-retained，task80的retained最近邻是task86-gained。对应same-task action cosine为task23
`.3529`、task26`.0615`、task80`.2013`、task86`.5104`。exact JVP只有task80通过，即`1/4`，跨trajectory cosine整体更
接近零。full与early-half给出相同action最近邻结构，且全部64个selected states均未完成BDDL full goal conjunction，
所以这不是末端success predicate或只看最后一帧造成的表面失败。

该实验只淘汰`单一成功adapter + 两条独立occupancy + 各stratum独立max选择 + direct concatenation`作为task code。
它不能外推为action response没有信息：task23/86在最相似cross-task干扰下仍成立，且task26/80各有一个方向正确；真正暴露
的是同一task的初始化、trajectory duration与event phase nuisance仍可大于task差异。JVP在该证据下不应升级为primary
目标，但可保留为局部辅助标签。

回到专家原始因果链，后继必须同时补两个缺口：多个独立successful adapters，而不只是同一优化轨迹或单adapter；以及
显式monotone phase correspondence，而不是假设相同相对stratum就是相同控制阶段。下一面因此使用source从未训练的target
train24：从既有step250--2000 formal rows为每task预注册最早/最晚成功checkpoint自己的最短成功trajectory，23 tasks为
K2、唯一只有一次成功的task为K1；只在fit19学习phase alignment/PCA，held5固定变换比较aligned与unaligned invariance。
该标签门通过前不再训练Decoder，避免再次让参数objective掩盖功能表示失败。

完整pairwise矩阵、selected replans、stage状态与裁决见
`docs/evidence/functional_adaptation_20260819/successful_onpolicy_response_panel_20260821.json`。

## 33. 多成功checkpoint的完整轨迹响应首次通过role-disjoint held几何门

复用既有train24 step250/500/1000/2000 experts与五次1200-row formal结果，为每task固定最早/最晚有成功的checkpoint及
其最短成功row；23 tasks形成K2，task39仅K1。四个capture在clean detached `545b43c`上复现47/47成功，没有换row或
重新训练。随后在clean detached `7258487`上对每条trajectory的全部replan重算expert-source完整`50x7` action chunk，
而不是再从每个相对时间bin独立挑最大点。

只在fit19以task/member/state等权拟合350→32 PCA/whitening，解释方差`.923430`；held5只经同一固定变换。8点等时间
resample本身已让held5全部同task pair互为全局cosine最近邻，说明主要突破来自成功checkpoint、完整trajectory响应和统一
坐标，不应把收益全归因于弧长。功能弧长resample同样为`5/5`，并在tasks 0/9/25/36提高same-task cosine、task18下降，
即`4/5`优于等时间；fit19则从等时间`15/18`降为弧长`14/18`。它仍超过预注册的`>=4/5 mutual-nearest + >=2/5改善`门，
返回`advance_to_phase_aligned_fixed_decoder`，但不支持“arc length普遍更优”的更强主张。

这项正证据第一次使role-disjoint、无held拟合的successful/on-policy功能标签具备重建Decoder资格。下一步fresh Decoder以
fit19 compact task code和多个成功成员的phase-state functional监督训练；held5 earliest/latest codes分别物化和闭环
裁决，不平均LoRA、不优化held code。完整证据见
`docs/evidence/functional_adaptation_20260819/train24_successful_equivalence_phase_20260821.json`。

## 34. Phase-aware Decoder产生真实净增量，但没有保留successful expert support

fresh Decoder只在fit19优化，held5 earliest/latest codes均为冻结变换且零步拟合。5-rank、950 task visits后，fit/held
identity-relative flow loss为`.323930/.616152`；earliest/latest held family为`.636755/.595550`，内部functional门通过。
这说明固定16维phase code确实能驱动Decoder在未见task的成功轨迹状态上产生方向正确的policy response，但它仍不是
closed-loop裁决。

同一held5×50 rows上，source/direct-earliest/projected-earliest/direct-latest/projected-latest依次为
`21/74/44/108/44`。两套projected相对source都是净`+23`：earliest为14 retained、30 gained、7 lost，latest为12/32/9；
5/5 tasks均不退化，3/5严格提高，exact McNemar分别为`.000191/.000431`。earliest/latest之间28 successes重合、各16独有，
Jaccard`.466667`，也超过预注册`.44`。因此phase表示与fixed Decoder不是纯粹identity或完全无效的负结果。

决定性失败在direct support：earliest只保留`20/74=.27027` direct successes与`11/62=.17742` direct gains；latest只保留
`28/108=.25926`与`21/96=.21875`。相对direct分别净`-30/-64`，远低于`.75/.60`门。Goal25与Long36在source/projected
都为0，却分别有direct `3/8`与`3/3`，进一步证明低held flow loss没有把成功策略的关键闭环support带过来。

最早失效接口现在比“offline surrogate错位”更具体：首轮虽使用真实successful expert trajectories，却只在expert
occupancy上拟合；decoded adapter一旦漂移到自己的learner occupancy，训练面板不再约束恢复动作。这正是专家原始方向A中
“统一probe panel和closed-loop state bank”尚未真正覆盖的部分。下一轮不改phase code、expert bank或held split，复用
fit19 successful初始状态收集projected-policy trajectories，并在这些状态上查询对应privileged experts做一次staged
on-policy aggregation。约26%的support retention也已触发专家挑战十二的shared prior + residual分支，但该架构变量留到
state coverage单独裁决后实施，避免把两个原因混在一轮。完整证据见
`docs/evidence/functional_adaptation_20260819/train24_phase_decoder_held5_20260821.json`。

## 35. Learner-state聚合有真实增量，但不足以恢复direct support

回查专家原始方向A后，复用旧Decoder、phase codes、experts与成功初态，只在fit19增加decoded-policy learner occupancy。
30条唯一projected trajectories覆盖37个earliest/latest member targets；每个target沿projected action-chunk功能弧长选择8个
真实learner states，再在相同状态查询对应privileged expert。successful/learner panels严格1:1，held5没有梯度、code重拟合
或checkpoint选择。6-rank、912 task visits后，learner-state mean functional loss从`.629034`降到`.155116`，held mean也由
旧`.616152`降到`.560983`，所以closed-loop state bank不是无效建议。

同一held5×50固定rows上，source/direct-earliest/projected-earliest/direct-latest/projected-latest为
`21/74/54/108/47`。相对旧投影，earliest有22 gained、12 lost、净`+10`；latest为18/15、净`+3`。earliest相对source
净`+33`且5/5不退化、3/5严格提高；latest净`+26`，但Spatial0净`-3`且仅Object一项严格提高。更关键的是direct success
retention仍只有`23/74=.31081`与`30/108=.27778`，direct gain retention仅`.19355/.26042`；earliest/latest成功集Jaccard
从`.46667`降到`.40278`。因此Gate 2继续明确失败。

科学边界是：当前一次staged functional state aggregation不足以解决support retention，不是“occupancy不重要”。它改善了
absolute与监督面误差，却没有形成稳定task-conditioned闭环程序；继续增加相同state-bank visits、LR/seed/rank小扫没有新
因果信息。按专家挑战十二，下一独立变量是显式稳定的shared behavior prior与task-code residual；两者在rollout前合并为
唯一complete LoRA，并必须同时报告shared-only，防止把carrier能力误写成task inference。证据：
`docs/evidence/functional_adaptation_20260819/train24_phase_decoder_state_aggregation_held5_20260821.json`。

## 36. 稳定shared prior有真实价值，但当前task residual在闭环中反向

专家挑战十二的公式不能通过直接相加两套full-rank A/B实现，否则effective `BA`会出现交叉项。本轮把public rank16精确
分成shared rank12与task rank4：前12 ranks由固定zero code在fit19 successful/learner 1:1 panels上学习，后4 ranks冻结
shared后由zero-code-centered phase code写入。两块按rank拼接为唯一complete LoRA；`D(0)`逐tensor等于shared-only，部署
没有第二adapter、task-ID route、LoRA平均或checkpoint fusion。

两阶段各完成912 task visits/152 updates。task residual把fit mean从shared的`.575078`降到`.403687`、held mean从
`.680319`降到`.659049`，说明它确实拟合了额外functional target；但同一held5 fixed250闭环为source/shared/
composite-earliest/composite-latest=`21/43/37/33`。source→shared为17 retained、26 gained、4 lost、净`+22`，exact
McNemar `p=5.95e-5`，所以稳定task-independent behavior prior不是无效carrier。其能力却高度集中于Spatial0的`38/50`，
只能作为架构证据，不能称为task或video inference。

决定性反事实是shared-only。shared→earliest为29 retained、8 gained、14 lost、净`-6`；shared→latest为29/4/14、净
`-10`，latest退化的exact p为`.03088`。两套composite只保留direct successes的`.22973/.15741`和direct gains的
`.09677/.07292`，虽然成员Jaccard升到`.62791`，也只是更稳定地保留了窄support。因此当前`shared12 + functional
phase-code residual4`明确淘汰，不能把shared的43分归功于条件残差；同时不能把该负结果扩大为“shared carrier不合理”或
“task residual一般无效”。

这轮再次复现专家的核心判断：held functional loss改善并不保证closed-loop success，最早失效接口已经从“有没有稳定
底座”前移到“什么外层credit能让task-conditioned residual增加而非覆盖底座support”。所以不再小扫rank、LR、seed或相同
state-bank visits，下一独立主要变量是挑战十四/方向E的train/meta closed-loop outer objective：固定已验证shared support，
复用现有functional资产作warm start，以授权train/meta simulator success/progress训练条件推断，held仍zero-interaction且
reward零梯度。证据：
`docs/evidence/functional_adaptation_20260819/train24_shared_prior_residual_held5_20260821.json`。

## 37. 当前单方向outer credit没有越过shared support

按专家挑战十四/方向E，首个outer实现固定rank12 shared prior与rank4 complete-LoRA decoder，只训练
language+ordered-video到functional code的推断。fit19按task等权；两轮functional warm-start后，每个task使用一个Gaussian
antithetic方向、每个正负方向各两个common-random-number初态，以success、执行效率和BDDL goal-predicate progress形成
closed-loop advantage。held5不读reward/action、不拟合code，部署仍是视频与语言一次生成唯一完整LoRA。

macro2 warm-start在同一held5 fixed250上为`41/250`，per-task为Long0、Goal0、Object4、Spatial0为36、Spatial9为1，
breadth `3/5`；它几乎复现shared-only的`43/250`，41个成功row还是shared成功集的严格子集。一次outer macro耗时
`300.74s`，19 tasks中10项产生非零advantage，plus/minus各11次成功，mean advantage仅`.00027165`，说明训练图和reward
credit确实接通，但该数值不能替代held裁决。

macro3降到`39/250`，per-task只把Spatial0从36降到34，其余完全不变。macro2→3为37 retained、2 gained、4 lost，
Jaccard`.86047`；那2个gained只是恢复macro2相对shared丢失的rows，shared→macro3仍是39 retained、0 gained、4 lost，
Jaccard`.90698`。因此当前outer更新没有创造任何shared support之外的task-conditioned success，Goal与Long也仍为0。
按预注册Gate 4停止macro4，避免把负结果靠更多相同更新或rank/LR/epsilon/seed小扫拖长。

该结论只淘汰当前`one direction × two rollouts/sign`的finite-difference参数化和这一次outer update，不否定专家关于
closed-loop credit的核心判断，也不否定learned progress、更多方向/结构化credit、skill composition、video初始化的
task-local RL或runtime conditioning。下一步先对最佳macro2补matched learned language-only与video-only，确认41分里的视频
净增量，再回到专家账本按最早失效接口选择结构上不同的方向。完整证据见
`docs/evidence/functional_adaptation_20260819/train24_functional_outer_credit_held5_20260821.json`。

## 38. matched面板把最早失效接口前移到outer之前的过程推断

macro2没有重复训练Writer，而是复用同一checkpoint、held5 fixed250 rows、task/state/env/policy RNG，补齐专家要求的
learned language-only、video-only、first+final与same-task-other反事实。correct/language/video/first+final/same分别为
`41/39/40/39/40`，五臂breadth均为`3/5`，而Goal与Long在每一臂都是0。language→correct为35 retained、6 gained、
4 lost、净`+2`、exact `p=.75391`；video→correct为37/4/3、净`+1`、`p=1`；first+final→correct为36/5/3、
净`+2`、`p=.72656`。所以完整视频没有形成可与语言或端点区分的闭环增量，41分不能再解释为过程posterior已经成立。

同任务换另一条未挑选视频后为`40/250`，但correct→same是36 retained、4 gained、5 lost，Jaccard`.8`；36/41的
correct-success retention为`.87805`，低于专家给出的90%--95%资格门。aggregate只降1分会掩盖成功row换手，正是专家要求
同时报告retention/churn的原因。所有差异仍集中在Spatial0，Object8的correct/first+final/same均为4，Goal/Long没有任何
视频增量；当前模型继续表现为窄shared/task-template support，而不是跨suite procedure compiler。

回查初始化合同排除了“完全遗漏旧process预训练”：56-task macro10 Writer共有212 tensors、8,121,416 values迁移到本轮，
旧decoder明确未加载；仅language-code、video-code与posterior-delta的三个最终层因为旧32维到新16维坐标而重新初始化。
关键缺口是这三个新heads只接受两轮低学习率correct-only target-code拟合，本轮的warm-start loss又把
control-confidence、control-update与action-alignment权重设为0。也就是说，旧frontend知道的过程/Action结构没有在新固定
坐标上被直接约束，而不是专家方向I/G已经在当前16维接口被充分检验。

因此下一主要变量不是恢复macro4、扩大finite-difference方向数或扫rank/LR/seed；那会在未过Inference Gate的warm-start上
增加昂贵噪声。应复用现有`PrivilegedMetaActionStore`、`controlled_process_input`与dynamic-K schedule，在fit19上让
新16维heads接受跨episode action-phase alignment及reversed/shuffled/first+final/endpoints-middle-shuffled反事实，并至少
覆盖K1--4。先用同一correct/language/first+final/same panel裁决；只有过程增量与跨视频稳定性过门，才为该checkpoint接入
结构不同的outer estimator。该结论淘汰的是当前两轮correct-only 16维warm-start，不否定fixed coordinate、旧process
frontend、train/meta action correspondence或outer credit一般。证据仍归档在
`docs/evidence/functional_adaptation_20260819/train24_functional_outer_credit_held5_20260821.json`。

## 39. EMBER-ECP首版native observer优化成立但task-conditioned event识别失败

EMBER-ECP Stage 0A首版在71个audited non-held meta tasks与train24 fold0 fit19上完成10个formal macros、900次task-equal
visits。total、cross-episode action alignment、uncertainty和same-task consistency都稳定下降，全部gradient与checkpoint
finite，因此这不是工程失败或训练图未接通。但cross-task contrast反而由`1.721745`升到`1.759694`，接近6-way chance；
posterior entropy由`1.464783`降到`.241540`。macro10的mean presence在90 tasks上均值`.348687`、标准差仅`.001770`，
source71与fit19的均值差只有约`.000162`。这与“简单任务激活更少slot、复杂任务激活更多slot”的动态E合同不符。

固定train24 observer panel提供了独立反事实。same-task other-video summary cosine达到`.999985`看似稳定，但mean cross-task
cosine也为`.996493`，nearest cross-task更为`.999125`；fit19/held5的nearest margin只有`.000927/.000604`。换成
antithetic fixed Gaussian probe后summary/event cosine仍为`.998766/.998409`。因此高same-task一致性不是event abstraction
的正证据，而是几乎所有条件都映射到同一表示的坍缩。首版native observer明确未过Gate 1，不授权compiler、`q_V`或正式
Action Meta训练。

代码级最早失效接口与这些数值一致。learned `minimum_duration`没有约束每个slot的真实最短持续时间，只作为
`1-exp(-occupancy/duration)`的全局可学习分母；正presence sparsity会奖励所有task共同放大分母。与此同时，首版action
alignment先用posterior把所有frame targets平均为event target，再回归event均值；把不同阶段帧压到同一slot仍可降低平均
误差，缺少逐帧重构责任。两者共同形成“全局presence尺度 + 少数event均值”的捷径。

所以下一变量不是更多相同训练、LR/rank/seed小扫，也不是规定某段视频必须对应某个slot。修正保持slot语义完全learned，
只把presence改为相对有效帧数的固定occupancy fraction，并用每帧posterior加权event action prediction重构该帧的
cross-episode action target。它同时消除速度/视频长度对presence尺度的影响，并让event collapse直接承担时变动作误差；
其余native graph、38 owners、50 horizons、correct-only数据墙和固定panel均不变。完整证据见
`docs/evidence/ecp_20260822/stage0_native_macro10_gate1.json`。

## 40. Stage 0 v2消除presence尺度捷径后，event posterior仍先于action grounding坍缩

Stage 0 v2从clean pushed `395912a`使用同一90-task、world-size6合同完成10 macros/900 task-equal visits。训练段253.23秒，
全部loss、gradient和checkpoint finite；因此结果不是运行失败。固定occupancy-fraction presence确实改变了首版动力学：
macro1有`6.85`个active events、presence sum`5.592081`。但active events随后按
`5.26→3.02→1.39→1.03→1.00`坍缩，macro6--10均严格为1；macro10逐帧posterior action reconstruction仍为`.251337`，
cross-task contrast为`1.733799`。固定48-row panel的correct、保序2x speed、same-task other和antithetic四个条件也全部只有
1个active event；same-task-other summary cosine`.999981`的同时nearest cross-task cosine仍有`.998369`，nearest margin
仅`.001611`。所以v2 Gate 1继续失败，未启动Action Meta、compiler或closed loop。

独立检查真实`PrivilegedMetaActionStore.phase_targets`排除了“跨episode action target没有时序信息”。在8个固定task
authorities、每task 2 views上，逐视频最优常数action的MSE均值为`.178693`，按真实视频顺序分成8个等时bin后的oracle均值
为`.034727`，降低`80.57%`；实际训练loss甚至高于常数oracle。最早接口因此不是target identifiability，而是监督到达顺序：
teacher action只有先穿过随机event pooling和随机event action head才接触frame evidence；在frame/process尚未学会phase时，
same-task/uncertainty/presence/entropy项已经给posterior提供更容易的单event解。

后继v3不规定事件数量、slot身份或action-derived硬边界。它在segmentation之前从每帧`B[s,m,j]`形成owner evidence，使用与
event分支完全共享的owner pooling和action decoder直接预测该帧cross-episode phase action；event分支仍通过soft posterior
重构逐帧action。frame grounding建立前将premature consistency、uncertainty、presence sparsity和posterior entropy权重置零，
使event assignment首先面对真实有序phase差异。该修正只把expert建议中已有的train-only action-horizon/phase calibration
前移到可识别接口，deployment输入仍只有language与action-hidden video。完整证据见
`docs/evidence/ecp_20260822/stage0_native_v2_macro10_gate1.json`。

## 41. Pre-segmentation action grounding首次建立跨episode、跨task的动态event几何

Stage 0 v3只在event pooling前增加training-only frame action grounding，并与event branch共享owner pooling/action decoder；
direct grounding建立期间关闭same-task consistency、uncertainty、presence consistency/sparsity和posterior entropy，保留
cross-task contrast。clean pushed `2d19ea8`在同一90 tasks完成10 macros/900 visits，frame/event action loss由
`.312545/.312241`降到`.243966/.246427`，cross-task contrast由`1.721222`降到`1.376669`。更重要的是active events在
10 macros始终约6--7，macro10为`6.97`，没有重演v2在macro6前严格坍缩成1。故teacher-action phase监督的到达顺序确实是
v2最早失败接口；固定presence本身既不是充分修正，也不是动态E失败的根因。

固定48-row panel把“只是多开slot”与task-conditioned geometry区分开。correct active events均值`6.48`且按row覆盖4--8；
same-task-other summary/event cosine为`.999601/.999270`，mean/nearest cross-task为`.909019/.980528`，margin为
`.090582/.019073`。48/48 rows的nearest margin都为正，held5同样10/10为正且最小`.001715`；速度视图summary/event cosine
`.999975/.999871`。这与v2的same-task `.999981`、nearest-cross `.998369`、单event完全不同：v3表示既跨视频稳定，又保留
task差异，并在未拟合held5上成立。

剩余边界是fixed-probe robustness，而不是再次否定native observer。antithetic summary/event cosine为`.978224/.976424`；
只有16/48 rows中，antithetic same-task summary仍比canonical nearest-cross更近。极端probe翻转的变化因此常大于最近task gap，
native还不能直接冻结为最终authority。根据owner已确认的必做合同，下一步只训练shared、无task-ID的Action Meta-LoRA，冻结
native observer其余参数，并用完全相同panel做matched裁决；只有Meta不造成task/event geometry退化且改善或保持probe稳定时
才采用。此时不启动compiler或用内部几何替代闭环Gate 2。证据见
`docs/evidence/ecp_20260822/stage0_native_v3_macro10_gate1.json`。

## 42. Action Meta-LoRA是中性校准，按owner规则采用并冻结Stage 0 authority

冻结native v3 macro10后，只训练18层Action Expert q/k/v/o的shared rank4 Meta-LoRA，共626,688 values；native observer、
source、PaliGemma与Action Expert原参数全部冻结。clean pushed `a42601a`完成10 macros/900 visits，耗时825.15秒，active events
全程`6.83--6.91`，所有adapter gradients与checkpoint finite。训练只证明adapter图接通，不能选择authority。

完全相同48 rows的matched panel显示Meta几乎不改变native表示。nearest margin均值从`.0190726705`到`.0190832342`，逐row
33项提高、15项降低，最大绝对变化低于`7.5e-5`；mean margin提高`2.95e-5`。same-task-other summary cosine只变化
`-1.51e-7`，antithetic summary cosine变化`-3.55e-6`；48/48 positive margins、held5 10/10和antithetic-closer 16/48全部
不变。panel显存完全相同，耗时只从188.16到188.85秒。这不支持“Action Meta解决了probe instability”的更强说法，也没有
可复现退化。

owner已明确要求Action Meta必须尝试且只要没有负面效果就启用，因此最终Stage 0 authority固定为native checkpoint与Meta
checkpoint的组合；后继只在observer forward安装Meta并永久冻结，不能与`q_pi`、compiler或`q_V`共同旋转。它不是deployment
第二adapter：rollout前仍只物化compiler生成的一套完整rank16 LoRA。Gate 1由此完成，下一阶段进入visible-event-anchored
privileged `q_pi + compiler`，内部几何不替代held5 closed-loop Gate 2。证据见
`docs/evidence/ecp_20260822/stage0_action_meta_v3_gate1.json`。

## 43. Stage 1首版不是held过拟合，而是Program修正不足后被compiler进一步压成共享更新

EMBER-ECP首个privileged Stage 1从clean pushed `6d71cb8`完成fit19共1,140 task visits、190次world-size6更新；held5只做
同一冻结变换，shared gradient严格为0。三个预注册checkpoint的held5 strict250为`23→27→27`，而同rows的
source/shared/direct-earliest/direct-latest是`21/43/74/108`。1140逐task只有`24/1/2/0/0`；相对570是
`17 retained / 10 gained / 10 lost`，不是稳定积累。相对source虽净`+6`，McNemar仍为`p=.30746`；direct earliest/latest
success retention只有`.16216/.13889`，Goal与Long持续为0。因此Gate 2失败，不能训练`q_V`。

内部loss继续下降不代表只是训练不够。1140的fit19与held5 member exact-BA loss为`.92146/.89844`，held反而略低，排除了
“主要是held Program泛化断裂”的窄解释。held generated相对selected direct的平均effective norm ratio/cosine从228的
`.2518/.1117`、570的`.3200/.2721`升到1140的`.3694/.3290`，但闭环在后半程零增长并有10/10 churn，说明沿当前方向继续
靠近并没有保留policy support。

跨task几何揭示了更早的共同更新坍缩。24个生成LoRA的effective-update cosine均值`.996807`、最小`.986824`，而24个
direct step2000 adapters均值只有`.131914`；generated的mean own-direct cosine`.250756`低于nearest-other`.365999`，
只有`1/24` tasks自身方向最近。visible anchor与`q_pi` teacher process本身跨task cosine约`.9466/.9461`；privileged
correction单独更有差异，cosine`.8690`，也把与direct pair geometry的相关从`.4271`提高到`.4993`。但其norm只约anchor的
8%，代码中的唯一`residual_scale`训练后仍是`.1006`。compiler family output scales也仍约`.096--.099`，并要求full Program
在A、B两因子上共同从幅度更大的stable-prior template做残差；最终把尚存差异重新压到`.996807`。

所以首版最早失效接口不是单独的observer、held split、rank16容量、矩阵shape或浮点误差，而是两个连续参数化：

1. `q_pi`用单一小scalar限制所有event/owner privileged correction，不能按policy证据需要改变幅度；
2. full compiler必须先以近task-invariant A/B residual取消stable prior，再写入task update，双线性优化收敛到低能共享方向。

该结果只关闭当前scalar-bounded teacher、template-residual full output及其相同训练延长，不关闭privileged Program、stable-prior
反事实或fixed compiler顺序。后继应让`q_pi`使用presence-bound content gate，并让同一compiler对prior-only与full Program
分别输出绝对完整rank16坐标；compact-SVD canonical factors只作优化warm-start，最终仍由multi-state functional support、
fit reward/progress与held closed loop裁决。完整证据见
`docs/evidence/ecp_20260822/stage1_privileged_compiler_fold0_gate2.json`。

## 44. Absolute factors移除了template取消项，但参数warm-start仍沿address捷径收缩到近零共享update

content-gated `q_pi`与prior/full absolute compiler v2从clean pushed `7ca808d`完成fresh 228 visits/38 updates，运行与
checkpoint均正常；full不再叠加stable-prior A/B，prior-only exact-BA loss按结构为0。尽管如此，24-task物化candidate的
跨task effective cosine仍为`.994192`，direct仅`.131915`；mean own-direct `.183969`低于nearest-other `.282906`，只有
`2/24`自身检索正确，candidate/direct norm ratio只有`.099771`。fit19/held5 member loss分别`.94840/.95781`，不是单纯held
泛化差。预注册几何门失败，因此没有运行held250，也没有启动`q_V`。

这轮只训练了BA/factor坐标与locality：配置中的successful-policy functional从228之后才启用，38次更新实际为0个functional
updates。它因此直接验证了专家警告的窄情况——raw A/B重建即使换成canonical absolute坐标也不会自动得到policy-effective
Program。结果只关闭“absolute坐标加参数warm-start足以摆脱坍缩”，不能用来否定尚未训练的policy-functional oracle链。

代码复核还定位到独立的内容旁路。Program owner/type/event embeddings与task内容共同进入attention values；constant
target/rank query又在cross-attention后以`hidden + query`直达factor heads；q_pi factor tokens同样加入owner/rank embeddings。
因此numeric address不仅定位读取位置，还能在Program内容近似时直接生成一套地址特定但task近共享的LoRA。后继不删除
event/layer/family轴，而是把**地址与内容分责**：地址只用于key/query/locality，value与最终factor hidden只来自Program内容；
同时successful-policy functional从第一个update启用，BA/canonical只保留低权重优化坐标。证据见
`docs/evidence/ecp_20260822/stage1_absolute_compiler_fold0_geometry.json`。

## 45. Content path恢复了task差异，但functional credit在坐标建立前把差异转向了错误policy方向

Stage 1 v3从clean pushed `cba8caf`完成228 visits/38 updates。地址只影响key/query/locality、Program content全零不能写出
full LoRA的结构修正确实生效：24-task candidate跨task effective cosine由v2的`.994192`降到`.939205`，candidate/direct
norm ratio由`.099771`升到`.487844`；Program anchor/teacher/correction的跨task cosine也分别降到
`.914320/.900172/.854394`。因此不能再把失败归因于numeric address仍独立生成一套近共享LoRA。

但恢复的差异没有对准同task successful policy。candidate own-direct cosine只有`.012822`，低于nearest-other `.026238`，
只有`2/24`任务检索正确。fit19/held5 member exact-BA为`1.48396/1.29375`，两者都比stable prior约`1.104`更差，说明这不是
held泛化断裂。更直接的训练轨迹是：前5到后5 updates的functional response由`.995844`改善到`.871159`，member exact-BA
却由`1.14167`恶化到`1.37474`，canonical factor由`1.24101`恶化到`1.56286`。有限successful-occupancy panel先提供了可降的
响应捷径，shared compiler尚未建立task-to-LoRA坐标就被该gradient旋转。

因此v3只关闭“content/address separation加从首步joint functional即可建立oracle compiler”，不否定同一Program、`q_pi`、
compiler或policy-functional目标。后继v4不新增架构：前228 visits用exact-BA/canonical/prior/locality做显式coordinate
bootstrap，几何过门后才从同一checkpoint启用functional，并补齐专家要求的source/shared support与fit reward/progress。
bootstrap不能选择最终方法，仍必须通过held5 closed-loop Gate 2。证据见
`docs/evidence/ecp_20260822/stage1_policy_functional_compiler_fold0_geometry.json`。

## 46. Coordinate bootstrap下降了监督loss，但query只选位置不足以建立task-to-LoRA方向

Stage 1 v4从clean pushed `fc0b84e`完成fresh 228 visits/38 updates；全段只做coordinate bootstrap，functional panels
加载数与functional updates均为0。前5到后5 updates的member exact-BA由`1.05391`降到`1.00938`，canonical factor由
`1.14880`降到`1.08904`，说明训练顺序修正确实让坐标监督按预期优化。

但24-task物化没有恢复正确policy方向：candidate跨task cosine为`.858906`，已经不是近全局同一输出；own-direct却只有
`.018399`、低于nearest-other `.029450`，自身检索`1/24`，candidate/direct norm ratio也只有`.091336`。fit19与held5的
member loss分别为`1.01542/1.00234`，均与stable-prior约`1.00658/1.00469`接近，所以失败既不是functional共同旋转，也
不是held泛化单独断裂。几何门失败，held5 closed-loop rows为0，`q_V`没有启动。

只读rank定位进一步缩小了compiler接口：candidate raw A/B内部rank向量平均cosine为`.8501/.7779`；gauge-invariant
participation rank为`1.0733`、top1 energy fraction为`.9664`，direct为`1.2616/.9127`。direct本身也高度低秩，因此这些
数值不是追求均匀rank的目标；它们只说明v3/v4删除所有query-to-hidden作用后，numeric rank/target identity主要停留在
attention选址，未充分调制读出的Program内容。

后继v5保持地址不能脱离内容写LoRA的反事实：不恢复`hidden + query`，也不把地址常量放回values；只用
`1+tanh(Wq)`逐维乘性调制cross-attended Program content。这样零Program仍严格产生零full output，但target/rank query可
参与内容读出。v4只关闭“content-only attention readout加coordinate bootstrap足够”，不关闭ECP Program、`q_pi`、absolute
single-LoRA surface或尚未执行的support/reward目标。完整证据见
`docs/evidence/ecp_20260822/stage1_coordinate_bootstrap_fold0_geometry.json`。

## 47. Query-content路径接通但仍收敛到近零update，Stage 1缺的是policy support而非下一次局部compiler修补

Stage 1 v5从clean pushed `ae15e47`在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates；全段仍是
coordinate bootstrap，没有加载functional panels，也没有held shared gradient。前5到后5 updates的member exact-BA由
`1.05208`降到`1.00924`，canonical factor由`1.14952`降到`1.13216`。

乘性query-content确实产生了局部正作用：own-direct cosine由v4 `.01840`提高到`.08214`，own retrieval由`1/24`提高到
`3/24`。但candidate仍更接近nearest-other `.10771`，norm ratio由`.09134`降到`.08643`，candidate pair cosine为
`.87652`，fit19/held5 member loss只有`.99507/.99141`，均接近“输出零update”的relative exact-BA约1.0基线。有效rank的
participation/top1为`1.0972/.9563`，也没有恢复direct的`1.2616/.9127`。因此几何门失败，held5 rows为0。

这轮排除了“只需让numeric query乘性调制Program内容，再做同一coordinate bootstrap”这一解释。继续修改query、rank、
初始化或loss权重，已经是在同一欠识别证据上反复优化compiler。回到专家Stage 1原始合同，当前最早缺口是privileged teacher
与policy support：`q_pi`只拥有successful-member factor和压缩successful occupancy response，没有learner-policy occupancy、
source/shared support、完整多状态policy response或task-equal success/progress。后继保持v5 Program/compiler反事实，改为从
首步用successful、经过agreement/outcome加权的learner、source与shared多策略response约束同一个Program/compiler，随后接入
fit simulator reward/progress。rank继续只是无技能语义的numeric factor coordinate，不增加到Program语义轴。证据见
`docs/evidence/ecp_20260822/stage1_query_content_bootstrap_fold0_geometry.json`。

## 48. 多策略support恢复了update幅度，但moving-panel响应拟合仍没有识别task policy方向

Stage 1 v6从clean pushed `85477ea`构建24-task policy-support bank：188个successful panels、120个learner panels，五个
source-subtracted response通道RMS均非零。fresh 228 visits/38 updates在115/113条successful/learner records上训练；前5到
后5 updates的functional response由`.64456`降到`.50289`。24-task物化的candidate/direct norm ratio由v5 `.08643`提高到
`.64465`，candidate effective participation rank/top1也达到`1.3871/.8494`，因此v6明确排除了“policy-support仍只能写近零、
近rank1 update”这一窄解释。

恢复的幅度仍未对准本任务完整policy：member exact-BA由`1.15677`恶化到`1.90182`，mean own-direct cosine只有`.01618`，低于
nearest-other `.02816`，自身检索仅`2/24`。candidate跨task cosine`.85242`说明它不是简单全局坍缩；更准确的边界是同一shared
compiler可以在轮换的局部successful/learner states下降response loss，却尚未证明冻结single LoRA在完整panel集合上保留同一
task的功能等价类。预注册几何门因此失败，held5 rows为0，不续训或扫rank/LR/seed/loss权重。

下一最早诊断是对冻结checkpoint遍历完整support bank，分别比较fit19/held5 candidate与source/shared的normalized response。
若冻结support成立，则direct参数cosine只是多策略等价类下的定位指标，下一major variable按专家合同加入fit-task task-equal
success/progress；若冻结support不成立，则v6淘汰的是当前policy-support teacher/轮换panel识别方式，先修该接口。证据见
`docs/evidence/ecp_20260822/stage1_policy_support_fold0_tv228_geometry.json`。

## 49. v6广泛改善source却一致破坏stable shared，最早断点是independent absolute full surface

clean pushed `a4928ce`对v6冻结single checkpoint遍历全部308个policy-support panels。task-equal fit19 candidate/source/shared
response为`.56706/.70633/.40514`：candidate相对source为`.80282x`且`19/19`任务更好，但相对shared为`1.39966x`且
`0/19`更好。held5为`.79756/.88454/.62434`：相对source `.90167x`且`5/5`更好，相对shared `1.27745x`且`0/5`更好。

这把v6负结果分成两层。multi-policy evidence和训练图有真实、跨fit/held breadth的正作用，因为它在所有24 tasks上都改善了
source；但full Program触发的absolute factor heads独立重写整套LoRA，没有继承prior-only surface中已经更接近successful experts
的stable shared，因此所有24 tasks又一致丢失shared support。此时先加simulator reward会把credit交给一个连现有底座都不能保留
的坐标系，不能解决最早断点。

下一major variable是prior-preserving low-rank union，而不是恢复已失败的raw A/B template addition或固定shared12/task4分槽：
heads只生成residual rank16 factors；每target将shared与residual拼成rank32 low-rank product，经thin QR和`32x32` core SVD取最佳
rank16，再输出唯一adapter。它在residual为零时精确保留shared，允许内容证据逐步替换弱shared modes，同时没有第二adapter、
raw factor交叉项或rank技能语义。v7仍须通过同一冻结support/几何门，不能由结构自证。证据继续见
`docs/evidence/ecp_20260822/stage1_policy_support_fold0_tv228_geometry.json`。

## 50. v7 prior union消除了大部分shared破坏，但raw参数坐标项反而成为主目标

v7从clean pushed `6987933`完成228 visits/38 updates，并由clean pushed `55b9065`遍历冻结的全部308个support panels。
relative v6，prior union把fit19 candidate/shared从`1.39966x`改善到`1.02429x`，held5从`1.27745x`改善到
`1.09995x`；胜过shared的task breadth也由`0/19、0/5`变为`9/19、1/5`。candidate仍在fit/held全部24 tasks上胜过source，
ratio为`.58751/.77638`。所以union是强正修正，但八项预注册门仍只通过四项source条件，不能进入held闭环、reward或`q_V`。

本轮暴露的最早实现偏差不是SVD数值问题。v7从shared精确起点出发时，direct exact-BA约为`9.43`而非此前absolute head的约1；
四个所谓“低权重”BA/canonical项因此在前5步贡献`1.06492/1.65750`总目标，末5步仍贡献`.45761/.86428`，始终超过一半。
与此同时shared-support由`.03149`升到`.12364`。这说明优化主要在追逐多套任意direct参数坐标，而不是专家要求的多状态policy
response等价类；它不是值得小扫的权重问题，而是错误目标仍留在active梯度里。

下一fresh v8保持Program、`q_pi`、prior union、数据、seed与228-visits节点不动，只把四个direct BA/canonical项从优化目标
中完全移除并保留为诊断。successful/learner policy response、局部source/shared support和locality成为唯一有效梯度。只有同一
冻结support gate通过后才加入task-equal simulator success/progress；失败后再依据fit/held分解决定是扩展71-task meta mappings
还是替换fixed SVD compiler。证据见
`docs/evidence/ecp_20260822/stage1_prior_union_fold0_tv228_support.json`。

## 51. v8删除参数坐标梯度和修正task balance后仍被Frobenius union捕获

v8从clean pushed `ae4805e`完成228 visits/38 updates，参数坐标四项权重严格为0；clean pushed `1659bb6`随后遍历全部
308个冻结support panels。fit19 candidate/source/shared为`.46988/.70633/.40514`，candidate相对source/shared为
`.66524x/1.15980x`，19/19 tasks胜过source、0/19胜过shared。held5为`.71738/.88454/.62434`，ratio为
`.81102x/1.14903x`，5/5胜过source、0/5胜过shared。candidate/direct norm ratio由v7的`1.37562`膨胀到`6.54391`，
candidate pair cosine由`.95247`升到`.97804`，own-direct由`.05039`降到`.01411`。所以删除raw坐标目标没有建立
policy-functional等价类，反而允许局部functional residual接管SVD union；该checkpoint不进入held闭环、reward或`q_V`。

formal metrics曾揭示一个必须先修的科学合同错误。`build_stage1_schedule`只保证完整456 visits中每个fit task恰好24次；它把
全部rows按cost全局排序成world6 groups后再整体打乱，导致228决策前缀每task实际只有`5--18`次访问，只有3/19 tasks恰为应有
的12次。冻结audit是task-equal的，所以负结果本身有效；但训练checkpoint没有满足其声明的task-equal比较条件，不能直接把
全部失败归因于SVD参数化。

schedule修复后，clean pushed `0b63da1`的fresh复验在228节点使19个fit task均恰好12 visits，successful/learner各114 records。
结果并未恢复shared support：前5到末5 updates的total/functional/shared-support从`.40702/.29527/.02232`恶化到
`.72780/.55854/.21882`；candidate/direct norm ratio升到`8.75029`，cross-task cosine升到`.99433`，own-direct只有
`.01106`且自身检索`1/24`。clean pushed `c1f485d`的冻结审计中，fit candidate/shared ratio为`1.17823`且仅`2/19` tasks
更好，held为`1.11729`且仅`2/5` tasks更好；88/308 individual panels胜出仍不足以满足任何shared aggregate/breadth门。

这排除了“短节点仅因task exposure不平衡失败”的解释。最早失败接口是Frobenius top-SVD按参数能量保留rank16：无界、近共享的
residual通过放大范数接管subspace，而不是由policy response选择有用mode。v8最终关闭；下一fresh实现只把该压缩换成bounded、
exact-prior、content-derived per-rank functional selector，保持其它变量不动。证据见
`docs/evidence/ecp_20260822/stage1_functional_union_fold0_tv228_support.json`。

## 52. bounded rank selector保住shared但学成近全局修正，最早缺口转到process因果Value路径

v9从clean pushed `dc5dff6`完成strict task-equal 228 visits/38 updates。它把v8无界SVD union的candidate/direct norm ratio
从`8.75029`压到`1.94717`，末5步shared-support也由`.21882`降到`.02773`；clean pushed `bb3bc59`的308-panel冻结审计中，
fit candidate/shared达到`.98369x`且19/19 tasks优于source，held也只比shared差`.69%`并在5/5 tasks优于source。这证明
exact-prior、bounded replacement是实质正修正，而不是参数数值装饰。

但完整门仍失败：fit/held只有`10/19、2/5` tasks优于shared，低于`13/19、3/5`；held aggregate为`1.00692x`。24-task
candidate跨task cosine为`.99779`，own-direct `.04015`低于nearest-other `.06247`且自身检索`1/24`。更直接地，去掉shared
后的correction pair cosine仍为`.97482`，其norm只有shared的`.24768x`；各task selector fraction也只分布在
`.08031--.09164`。因此v9学到的是一个安全、接近全局的shared修正，不是task Program编译。

当前compiler在process存在时仍把language、scene和process共同作为Value；exact prior则由特殊反事实直接返回shared。这允许
静态Value或presence开关在不同task上写出同一修正。下一fresh实现保持v9的bounded selector和全部训练变量，只把replacement
reader改成language/scene-conditioned query读取present process-only Values，并要求zero process即使presence为真也严格不能
写LoRA。证据见`docs/evidence/ecp_20260822/stage1_functional_rank_selector_fold0_tv228_support.json`。

## 53. process-only Value是正修正，但task-relative Program仍在compiler/objective中被压成近全局correction

v10从clean pushed `13dfc25`完成strict task-equal 228 visits/38 updates，clean pushed `85dc2fc`完成全308-panel
frozen support audit。它将replacement Value限制为present process/uncertainty，language/scene只条件化query，zero process
无论静态输入如何都精确返回shared。相对v9，candidate-minus-shared pair cosine从`.97482`降至`.95088`，
selector task范围从`.08031--.09164`扩大到`.06844--.12597`；fit candidate/shared从`.98369x`改善到
`.96892x`，breadth从`10/19`改善到`12/19`，individual panel wins从`132/308`提高到`158/308`。因此
去掉静态Value/presence旁路是真实正修正，不是结构自证。

它仍没有过门。held candidate/shared仅从`1.00692x`改善到`1.00285x`，breadth仍为`2/5`；full candidate
pair cosine仍为`.99641`，own-direct `.03983`低于nearest-other `.06200`，own retrieval仍是`1/24`。关键的接口
对照是：privileged Program correction的pair cosine已是`.82475`，但编译后correction回到`.95088`。这说明可见/
privileged Program中存在的task-relative变化仍被compiler与当前绝对functional objective压缩为共同安全方向。v10因此
关闭，不进入held闭环、reward或`q_V`；后继必须修复最早的task-relative Program-to-policy接口，而不是续训
或再做一个局部attention/routing微调。证据见
`docs/evidence/ecp_20260822/stage1_process_value_selector_fold0_tv228_support.json`。

## 54. OCPB program credit有微小正向证据，首个compiler credit因shared-rank尺度违反必须重做

OCPB v11从clean pushed `86ed95b`复用v10冻结checkpoint，不重复228 visits。macro1对fit19各运行一个
`event x owner` program-binding paired panel；训练臂success为`9/8`，但训练reward不参与checkpoint裁决。物化与同一
308-panel冻结audit显示，fit/held candidate-to-shared由v10的`.96892/1.00285`小幅改善为`.96786/1.00171`，candidate
pair cosine由`.99641`降到`.99616`；breadth仍为`12/19、2/5`，own-direct约`.03983`且自身检索仍`1/24`。所以program
credit并非零作用，但证据不足以进入held闭环或`q_V`。

exact-resume的macro2表面上得到`10/8` successes，却使fit/held support退到`.97049/1.00366`，shared panel wins由
macro1的`139/16`降到`131/14`，裁剪前梯度从`1.0904`升到`11.8792`。代码反查确认这不是可接受的科学负结果：每个owner的
perturbation把相同`delta`加到全部16个numeric rank angles，但differentiable surrogate使用angles的`sum`，因此surrogate
coordinate实际变化`16 delta`，而antithetic estimator仍以注册的`2 sigma`归一化。compiler reward lift被放大16倍后才进入
loss和gradient clipping，违反“一个跨rank共享owner coordinate”的预注册合同。

clean pushed `d06842c`把该coordinate改为rank mean，并以OCPB v12从v11 macro1恢复model、optimizer、scheduler、六份rank
RNG和world topology；`credit_macro_offset=1`复用原macro2的video、support、perturbation、environment与policy-noise seeds，
只重做一个corrected compiler-binding macro。v11 macro2只作尺度错误诊断，macro3/4取消；在v12物化和冻结audit前，不能据
macro2否定compiler-binding，也不能转入LIBERO-90扩容、held5闭环或`q_V`。证据见
`docs/evidence/ecp_20260822/stage1_ocpb_v11_rank_credit_diagnosis.json`。

corrected rank-mean真实单task profile在clean pushed `7d77eb8`上复用原macro2的task、video、support panel、perturbation与
paired rollout seeds。两臂仍为`2/2` successes且replacement fraction与原perturbation一致；surrogate为`-.000381`、裁剪前
梯度`2.08455`且finite，排除了16倍lift。首个profile暴露的support cache offset遗漏在optimizer update前退出，修复后临时目录
均已删除；profile只解除formal运行门，不改变上述科学裁决。

## 55. corrected compiler credit有效但未改善映射；support preservation的实现仍把有益偏移拉回shared

OCPB v12从同一个v11 macro1 checkpoint完整恢复model、optimizer、scheduler、rank RNG和world6 topology，并通过
`credit_macro_offset=1`复用原macro2的video、support panel、Rademacher perturbation、environment seed与policy-noise seed。
正式两臂仍为`10/8` successes、mean progress `.31140/.27193`；除一个成功rollout因允许的数值差异在109而非110步终止外，
task outcome pattern与replacement fractions保持一致。rank-mean修正使mean outcome surrogate从错误实现的`-.14340`回到
`-.00896`，裁剪前梯度从`11.8792`回到`1.63196`，所以v12是有效、可裁决的compiler-binding实验。

它没有改善最早科学接口。物化后candidate pair cosine仍为`.99586`，own-direct `.03972`低于nearest-other `.06186`，自身
检索仍`1/24`；Program correction的pair cosine却为`.82561`。冻结308-panel audit中，fit candidate/shared为
`.39272/.40514`、ratio `.96934`、breadth `12/19`；held为`.62623/.62434`、ratio `1.00303`、breadth `2/5`。相对保留的
v11 macro1，fit/held ratio从`.96786/1.00171`退化，shared panel wins从155降到149。按逐级门不运行held5 closed loop、
不轮fold、不扩meta、不启动`q_V`；继续相同macro也没有依据。

代码级反查进一步收紧了原因。`policy_support_loss_from_response`一方面用candidate到successful experts的response error训练任务
功能，另一方面把`candidate-source`和`candidate-shared`的响应距离直接作为“support preservation”损失。后两项不会区分
有益task-specific移动与有害漂移；在shared已是强安全底座时，它们为近共同修正提供持续吸引力。与此同时OCPB
`compiler_binding`的reward coordinate只扰动并监督38个selector angles，不能直接识别replacement factor方向；factor heads仍
只由上述functional objective决定。这与“.82561的Program差异在compiler后变成.99586”一致。

下一单变量不是调loss权重，而是改正support preservation的定义：保留同一Program、compiler、rank selector、v11 macro1
初始化、paired seeds和outcome coordinate，把source/shared项改成baseline-relative functional barrier——只有candidate对own
expert response比source或shared baseline更差时才惩罚，优于baseline的偏移不再被拉回。先用真实profile验证该barrier、factor
gradient和信息墙，再重做一个matched corrected compiler macro并立即物化/审计。证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v12_corrected_compiler_gate.json`。

## 56. baseline-relative barrier恢复了support breadth，但没有识别task-relative compiler方向

OCPB v13只把source/shared response proximity改为相对各自own-expert baseline不退化的hinge barrier；Program、compiler、
rank selector、v11 macro1初始化、paired seeds与outcome coordinate均未改变。正式macro保持与v12相同的`10/8` successes、
`.31140/.27193` progress和`-.00896` outcome surrogate，mean functional total从`.25295`降到`.15441`；source/shared barrier
只在`1/19、6/19` tasks激活，说明candidate已经优于baseline的方向不再被无条件拉回。

这一语义修正带来了截至目前最强的冻结support：fit/held candidate-to-shared为`.96741/1.00168`，breadth首次达到
`13/19、3/5`，8项预注册support条件通过7项；唯一失败是held aggregate比shared高`.1675%`。但task-relative几何没有
改善：Program correction pair cosine为`.82546`，编译后candidate pair cosine仍`.99595`，own-direct `.03980`低于
nearest-other `.06201`，自身检索仍`1/24`。所以v13证明support barrier定义值得保留，却也把剩余问题从“是否允许离开shared”
收紧为“compiler如何在policy-native功能空间识别应该沿哪个task-relative factor方向移动”。

按预登记联合门，breadth改善不能抵消geometry失败和held aggregate失败；不运行held5闭环、不轮fold、不扩meta、不进入
`q_V`，也不继续selector-angle宏或权重扫描。下一主要变量应让successful-policy与task-equal outcome evidence在
owner-resolved policy response空间直接约束compiler factor方向，同时避免退回专家已否定的raw A/B重建。证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v13_support_barrier_gate.json`。

## 57. owner-resolved response图已接通，但一个optimizer update不能裁决该Stage 1目标

OCPB v14从v13 checkpoint增加了同状态、同noise/time下的source与successful-member全层owner response。v2 bank覆盖
24 tasks，每个policy response为`[2,38,4,128]`；candidate在同一次可微PI0.5 forward中同时产生最终flow和owner response，
不读取或重建raw A/B。真实profile的owner loss为`.82564`、active owners为`92.11%`、梯度范数`15.1782`，正式19-task
gradient sum的mean owner loss为`.77170`且每task active fraction为`.7895--1.0`，因此“新objective没有进入真实图”已排除。

一个outcome macro没有改善科学接口。candidate pair cosine仅`.99595→.99575`，own-direct `.03980→.03971`、retrieval
仍`1/24`；fit candidate/shared由`.96741`变为`.96795`，held由`1.00168`变为`1.00313`，held breadth从`3/5`降到
`2/5`。所以v14 checkpoint明确不通过geometry/support门，held5和`q_V`继续禁止。

但该结果不能淘汰owner-resolved policy-response distillation本身。OCPB outcome macro把19个fit task的loss相加后只执行一次
optimizer step；此前可裁决的Stage 1短程节点228 visits则是38 updates、每task 12 visits。新objective的优化暴露量相差38倍，
而每个outcome macro还为paired simulator rollouts支付约255秒。将“one-update没有改变LoRA”解释为专家Stage 1目标失败，属于
以执行速度为由遗漏核心训练责任。

因此下一节点不是权重、LR、rank或seed扫描，也不重复昂贵outcome macro。保留v13 barrier、同一Program/compiler/rank与v2
bank，从v13 model weights和fresh optimizer建立task-balanced response-only阶段；114 visits/19 updates后先检查真实24-task
geometry，只有方向开始形成才继续至最多228 visits和full support audit，再允许一个matched outcome macro。正式证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v14_owner_response_gate.json`。

## 58. task-balanced owner response打破部分跨task坍缩，却没有识别owner-local成功策略方向

v15从v13 model weights和fresh optimizer完成228 visits/38 updates，每个fit task恰好12 visits；FactorHead裁剪前梯度在
`.7801--4.2447`内持续非零，排除了v14只有一个optimizer update的暴露不足。114节点candidate pair cosine已由v13
`.99595`降到`.99229`，own-minus-nearest margin、norm ratio与有效参与秩也同向移动，因此按预登记合同exact-resume至228。
最终pair cosine继续降到`.98727`、norm ratio降到`1.71187`、参与秩升到`1.12559`，说明目标确实改变了compiler输出，不能再
把失败解释为“owner loss没有进入factor heads”。

改变的方向仍不对应own successful policy。228节点own/nearest-direct为`.03980/.06075`，own retrieval仍`1/24`；冻结
308-panel audit中fit candidate/shared为`.97253x`、breadth`13/19`，held为`1.02171x`、breadth仅`1/5`。相对v13的
`.96741/1.00168x`、`13/19、3/5`和155个shared panel wins，v15退到146 wins。分解后learner ratio从v13`.96013x`改善到
`.95158x`，successful却从`.97171x`退到`.98404x`，进一步说明它学到的是可降低部分局部响应误差的相关变化，而不是完整
successful-policy等价类。联合geometry/support门失败，不能用outcome credit替这个坐标发明方向，也不运行held5或`q_V`。

最早接口是“owner-resolved”命名没有带来真正的owner-local梯度。当前target来自某层累计hidden与residual的投影；owner `j`
处的hidden可同时被所有上游LoRA targets改变，因此对应loss可以由多head协同变化满足。下一主要变量不是继续训练、调LR/rank/
权重或恢复raw A/B，而是在相同successful/learner occupancy和detached source reference input上直接监督每个target的
gauge-invariant局部activation effect `B_j(A_j x_j^{ref})`。它保留owner与horizon，只约束expert adapter在被访问子空间上的功能，
不会要求复制任意factor gauge；其余v13 barrier、Program/compiler、schedule和联合门保持不变。正式证据：
`docs/evidence/ecp_20260822/stage1_owner_response_bootstrap_v15_gate.json`。

## 59. owner-local activation effect能形成弱task检索，但孤立局部等价不保证完整policy组合

v16在同一detached source target input上直接监督38个LoRA targets的`B(Ax_ref)`，从v13 model weights与fresh optimizer完成
228 visits/38 updates。target-local目标不是dead graph：loss由前114的`1.69620`降到后114的`1.31481`，首个successful panel
上的own MSE由v13`.48016`降到114节点`.35668`和228节点`.25057`；cosine retrieval从`1/24`提高到`5/24、11/24`。
candidate跨task effective pair cosine也由v13`.99595`降到`.94932`。因此真正target-local的功能标签比v15累计hidden标签更能
传递task差异，不能把这项结果写成“局部effect完全无信息”。

但它没有组成成功策略。shared-subtracted effect显示direct expert correction的跨task pair cosine为`.82316`，v16 candidate仍为
`.97657`；candidate对own expert correction cosine虽有`.87419`，却低于nearest-other`.87516`，自身检索仅`2/24`。其幅度也只到
expert correction的`.49176x`，即主要形成一个跨task共同的stable-prior到expert半程插值。raw effective-BA own/nearest仍为
`.03873/.05976`、retrieval`1/24`。

冻结308-panel结果给出更强反证：fit candidate/shared从v13`.96741x`退到`1.08581x`，breadth仅`2/19`；held从
`1.00168x`退到`1.08815x`，`0/5` task优于shared；shared panel wins由155降到85。训练后半段successful、learner与shared
barrier也同时恶化。故继续相同local-effect曲线不能由loss下降授权，matched outcome、held5和`q_V`均不启动。

最早接口不是局部方向完全错误，而是**孤立target effect与38-target组合policy之间仍有缺口**：`B_j(A_jx_j^{ref})`在冻结source
输入上给每个head分配局部方向，却不约束上游改变后下游输入如何变化，也不直接优化最终action target。下一主要变量因此复用v16
已经获得的task discrimination，不从头重复训练；只在successful或verified-success跨episode train action panels上，以冻结PI0.5
的exact flow-matching loss计算完整generated LoRA的leaf gradient并反传`q_pi`/compiler。failed learner actions不作oracle，
local effect降为结构锚，v13 barrier继续保护source/shared。该路径直接检验policy composition，仍不重建raw A/B，也不向deployment
加入action。正式证据：`docs/evidence/ecp_20260822/stage1_owner_local_activation_v16_gate.json`。

## 60. exact action loss能打破task坍缩，但不能替代closed-loop factor-direction credit

OCPB v17从v16 model weights、fresh optimizer完成114 visits/19 updates。action图不是dead path：57个successful与18个
verified-success learner records产生非零leaf gradient，FactorHeads持续收到梯度；39个failed learner records按合同为零。
candidate pair cosine降到`.90236`、norm ratio达到`1.14481`，说明完整LoRA已不再只是小幅task-common correction。

方向仍然错误。own-direct`.03855`低于nearest-other`.05779`、retrieval`1/24`；308-panel fit/held relative-shared为
`1.13962/1.12203x`且breadth`2/19、0/5`，比v16的`1.08581/1.08815x`继续恶化。故“cross-episode exact action imitation自然
恢复successful composed policy”被否定，不能靠续228或调anchor权重挽救。

尚未被否定的是专家明确要求的task-equal closed-loop success/progress，因为历史outcome coordinate只对evidence gate或selector
angle给credit，未触及replacement factor方向。下一结构性变量应让exact action gradient提供38-owner/family局部proposal，paired
train simulator outcome只选择这些proposal的闭环符号和幅度；这既比全局随机LoRA扰动有结构，也不让reward负责发明Program。

## 61. factor-space closed-loop credit有效但仍不是固定Program坐标上的可识别监督

OCPB v18用exact action gradient构造38个owner-local A/B方向，再用每task两条paired lanes的success、progress与效率给方向
分配credit。4个macros每轮都有`8--10/19` tasks产生非零advantage，plus/minus successes也持续不同，因此不能再说“缺少
closed-loop reward”或“outer图没有接通”。

它仍没有恢复策略。candidate pair cosine从v17`.90236`降到macro4`.89140`，但own-direct始终约`.038`、retrieval始终
`1/24`；fit/held support最终为shared的`1.14317/1.13244x`，breadth仍`2/19、0/5`。短暂macro2 fit改善还在macro4反转。
所以被否定的是：**在compiler输出的完整factor空间同时扰动38个owner，再把一个task scalar reward投回同时可动的Program与
compiler，足以识别task-native policy方向。**

专家要求的“固定、功能锚定policy-adaptation坐标”仍未被这个实验真正检验。下一步必须把compiler固定，把perturbation放在其
真实可达的event/layer/family Program切空间，并让一次paired reward只裁决一个预注册结构块；否则scalar reward仍同时承担
Program坐标识别和decoder旋转，重复旧问题。该结论不否定closed-loop credit、action gradient或structured Program本身。
