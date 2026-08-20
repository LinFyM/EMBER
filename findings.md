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
