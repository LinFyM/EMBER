# EMBER Findings

更新时间：2026-08-15。本文只保留跨架构仍成立、会约束下一轮判断的结论。逐方法严格结果和禁止重复项见
`docs/research_history.md`；当前run只取`docs/active_session_handoff.md`。

## 1. 当前经验边界

长期继续追求同一shared Writer、同一single checkpoint的strict paired correct`>150/400`；owner也接受约145
的稳定有效方法，但必须同时通过相邻checkpoint低换手、same-task-video鲁棒和correct视频因果性。目前尚未达到。

最新CV-CSD给出完整负结果。它保持PCSD完全相同的48 pairs、9次唯一成功分歧与`5/4` candidate/reference gains，
把同一成功trajectory分别放到4个disjoint same-task correct K4 conditions下计算完整functional gradient。36个
view gradients全都finite/nonzero，full24 wall只为PCSD的`1.0307x`；所以“跨video成功credit无法工程化或没有
信号”已被排除。

cycle1 strict只有`134/400`、breadth7；相对LPCP143=`122 retained / 12 gained / 21 lost`、churn33、四suite
全降；相对PCSD135也有39条episode换手。全400 CV-CSD/LPCP effective-BA relative-L2 mean=`.000683702`，
gained/lost均约`.000679`。FP64 first4进一步显示同task四个correct K4增量pairwise cosine=`.000205`、
mean/sample energy=`.250155`；相对PCSD也为`-.001908/.248578`。

这把最早缺口推进到：**正确的cross-video成功objective已经存在，但全局shared query commitment经每个视频条件的
Jacobian仍写成近正交局部BA方向，无法在实际policy layer/rank/target topology上形成共同且可保留的承诺。**
CV-CSD只否定query-only四view exact mean这一组合，不否定multi-video、memory、reward、rank16或完整LoRA生成。
下一轮可以使用layer-aligned memory，但它必须直接解决commitment，而不是替换已通过的视频carrier或单纯加容量。

| 方法 | correct | same | wrong | shuffled | reversed | 主要结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v5.2 old | 132 | 138 | 74 | 82 | 83 | 视频内容特异性强，但absolute不足 |
| v5.2 task-complete | 120 | 109 | 107 | 111 | 124 | recipe削弱Procedure传递 |
| v6 old | 121 | 122 | 111 | 84 | 47 | 顺序影响能进闭环，但absolute与稳定性不足 |
| v6-fast task-complete | 143 | 135 | 125 | 128 | 129 | 历史最佳single checkpoint，仍未过150且视频margin弱 |
| V6-LPCP PCSD cycle1 | 135 | — | — | — | — | reward/action链路通，但跨video credit近正交且相对LPCP净丢8 |
| V6-LPCP CV-CSD cycle1 | 134 | — | — | — | — | 四correct K4 exact credit仍落成近正交BA，证明失效在query-only commitment |
| Dynamic-K backbone-memory rank8 | 100 | — | — | — | — | 动态K与真实backbone memory可训练部署，但task方向高度集中 |
| Dynamic-K semantic-address rank8 | 101 | — | — | — | — | absolute Core只作Query不足以修正policy方向 |
| Dynamic-K Direct-Family-B rank8 | 102 | — | — | — | — | BA共线略降但breadth5，mapper简化未解决共同积累 |
| Direct-Family-B K4 deployment | 98 | — | — | — | — | set把same-task方差降约6.3x，却稳定同一错误task mean |
| Task-Grounded Visual-Value rank8 | 88/86/86/96 | — | — | — | — | 视觉Value分化BA，但四点都远低于门且持续换手 |

Direct-Family-B只检验一个窄接口：保留semantic-address全部上游，删除family hidden/GELU，让shared projector
直接生成四类B。macro50 K1 strict=`102/400`、breadth5、per-task=`0/1/40/11/0/43/7/0`。相对semantic101为
`82 retained/20 gained/19 lost`，aggregate几乎不变却继续换手。task-mean effective-BA cosine
`.77947→.74895`证明几何略改善，但真实性能没有改善；因此“common-direction主要由该hidden造成且删除即可提高
policy effectiveness”被否决，不能继续靠mapper小修或内部几何选择方法。

Task-Grounded Visual-Value在其预注册macro50节点得到`88/400`、breadth5、per-task=
`4/0/34/2/0/41/7/0`。相对Direct-Family-B 102为`74 retained/14 gained/28 lost`，不是共同积累。它没有让LoRA
变成identity或增加same-task噪声：task/video BA SNR `16.34→19.05`，task-mean offdiag cosine`.749→.707`；但
BA相对前代平均cosine`.831`、relative-L2`.584`，说明视觉路径把LoRA显著旋向了对held rollout无用的方向。
functional loss轨迹几乎不变进一步复现offline surrogate/on-policy outcome错位。macro50仍只是完整0→200曲线的
首点，不能提前外推终局。

按target kind拆开后，q/v norm仅为Direct-B的`.947/.952x`，但action-in/out effective norm从`.446`增至
`1.545`、即`3.49x`。这不是简单能量爆炸：old134的action norm为`1.573`，新架构实际上恢复到`.98x old`；但
相对old134的action effective-BA cosine只有`.031`，方向仍近正交。Object3的净损失最大而总体BA变化幅度反而
最小之一，也排除“旋转越大越差”的标量解释。因此后续不得用全局scale、action scale或SFT能量匹配救该方法；
需要解决视觉credit如何选择policy/occupancy有效方向。

macro100进一步得到`86/400`、breadth6、per-task=`1/3/34/0/0/35/12/1`。相对macro50为
`62 retained/24 gained/26 lost`、churn50，两点union=`112`而single best仅88。BA 50→100平均cosine`.809`、
relative-L2`.696`；action norm ratio近1但方向cosine仅`.739`。尤其Object1总数同为34却7 gain/7 lost，证明
“总分稳定”不是能力稳定；当前functional credit继续在task和episode decision boundary之间换手。

完整终点macro150/200=`86/96`，macro200 per-task=`1/0/37/2/0/42/13/1`，top3占`92/96`。150→200虽为
`71 retained/25 gained/15 lost`、净增10，仍有churn40且没有解锁持续失败的Spatial3/Goal3；相对old134为
`68/28/66`，相对compiler138为`73/23/65`。因此四点`88/86/86/96`终局否决当前fixed-A组合；不能用晚期净增、
union或继续训练为它辩护。

## 2. 真正的学习问题

EMBER不是从视频复刻动作轨迹。Writer要结合exact language和action-hidden正确示范，提取在新初始化下仍成立的：

- 对象、关系和目标状态；
- 必要动作阶段及其有向顺序；
- 哪些信息是任务本质，哪些只是某条demo的速度、路径、视角、抓取角度或扰动。

Writer在rollout前运行一次，生成一套完整task adaptation；frozen policy随后不再观看视频。语言说明“要做什么”，
视频说明“正确过程是什么”，两者缺一不可。language可作query/context/address，但不能形成独立LoRA旁路；video
必须提供写出的dynamic value。

监督训练让video与action query同task、跨episode，能阻断逐帧动作复制并要求LoRA跨初始化有效。但这也造成
不可识别性：同task不同videos面对近似相同的task-level target，普通functional loss仍可能只学习task identity。
因此正确视频是否真正必要，必须由结构约束与闭环反事实共同证明。

## 3. 当前完整架构判断来自三条证据

当前Dynamic-K memory架构不是凭空替换历史，也不是机械复制外部Hypernetwork：

1. owner的现实启发要求语言与正确视频共同形成高层任务程序，允许一条或多条视频；
2. SHINE/Doc2LoRA类工作提供“少量memory进入原生context、保留layer-aligned状态、结构化生成LoRA”的成熟原则；
3. EMBER历史要求保留v5/v6的Semantic Core和有向Procedure、K4的逐视频保序/跨视频集合边界，以及完整
   policy-effective LoRA，同时避开language-only bypass、简单平均和无依据wide heads。

当前数据流是：

```text
exact language + K=1..4 ordered action-hidden videos
-> 每帧真实image/language/Action-probe context + 8 memory tokens
-> per-video adjacent transition D + terminal goal residual G
-> absolute memory mean只作temporal Query semantic address
-> causal temporal encoder（video内保序）
-> permutation-invariant set attention（video间聚合）
-> 20 policy groups x 8 rank coordinates M2P
-> shared projector + policy-shape-family readout
-> one complete 38-target rank-8 LoRA
```

8个backbone memory tokens与20×8 Program不是同一概念。前者在真实图文prefix下经过Action Expert层并读取输入；
后者是后处理的policy-group/rank-aligned状态。外部论文只影响设计原则，没有带来额外target-task数据。

## 4. 视频被使用不等于被正确理解

历史v4已经证明shuffled/reversed会改变hidden、LoRA和action，但`shuffled=148`反而高于`correct=109`。这说明
模型可以读取视频，却利用absolute frame phase、平移轨迹等错误捷径。

需要区分四个命题：

1. 视频改变representation；
2. 视频改变effective LoRA；
3. 视频改变policy action；
4. correct的内容与顺序沿有用policy direction提高闭环成功率。

只有第4项支持教学视频学习。shuffled/reversed必须对真实输入帧重排后完整forward，并与correct严格配对
task、state、policy RNG和video ordinal。不能只把negative人为推坏制造漂亮margin；absolute性能仍是第一目标。

## 5. Dynamic-K与few-shot的准确边界

多条同task视频有合理价值：跨demo共同部分更可能是高层程序，单条demo特有部分更可能是nuisance。有效架构应
逐video保序、跨video置换不变聚合，不平均frames/features/final LoRAs，也不挑最好video。

历史K4改善了部分permutation、same-video和leave-one-out内部稳定性，但best strict只有108，且未解决full24
credit retention、正确顺序或checkpoint drift。这只否定旧K4组合，不否定few-shot本身。

当前Writer训练时每macro让K1/K2/K3/K4各覆盖6个tasks，避免只见两个端点却宣称动态cardinality。同一checkpoint
K1/K4 strict为`102/98`；nested K1→K4=`80/18/22` retained/gained/lost，breadth均5。K4把same-task
effective-BA centered variance/sample从`.021674`降为`.003438`，task mean却保持cosine`.99604`且没有解锁
新task。这是比“few-shot没涨分”更具体的结论：set确实提取了跨video共性，但当前共性主要是错误task mean。
所以不能继续调K、挑video或平均LoRA；也不能由此否定few-shot。下一断点必须前移到per-video高层evidence及
task-level functional credit如何识别正确过程。

## 6. Task drift是核心症状，不是单一病因

SFB八个checkpoint success union=`193`，single best只有127。能力在训练轨迹中出现过，却没有稳定共存于一个
Writer checkpoint。已观察到的来源包括：

- 不同task、query和flow的functional gradients冲突或近正交；
- 同task不同video correction在effective LoRA空间近正交；
- shared condition map或compiler把异质更新压成common direction；
- offline action rows不覆盖held on-policy occupancy；
- compression、factorization和online regeneration跨越闭环decision boundary；
- optimizer轮流获得能力，而非形成共享可累积表示。

因此不能用“训练更久”、checkpoint union、挑task checkpoint或多checkpoint融合解释成功。每轮都要报告
per-task/per-suite、breadth、retained/gained/lost、相邻checkpoint churn与最早失效接口。

## 7. LoRA几何只用于定位

task-local SFT experts证明完整LoRA可以policy-effective，也提供正常能量、rank participation和target结构参考。
但历史同时否定了两种机械规则：近identity、完全共线通常是异常；强制均匀谱、高rank、正交、更多atom/lane/
expert或SFT量级norm也反复降低closed-loop。

优先分析effective `BA`、fixed-action response和closed-loop，不把raw A/B gauge符号当结论。当前fresh rank8不是从
旧rank16压缩，因此历史uniform rank14的support损伤不能直接否定它；反过来，低rank更易生成也不能自动证明更好。

## 8. Functional objective与真实reward长期错位

source-action SmoothL1、reconstruction、expert cosine、gradient consistency、MC variance、tube radius和offline
row retention都曾改善而closed-loop下降。离线B20只覆盖局部功能切片，rollout会改变状态分布并跨越离散成功
阈值；所以证据层级固定为：

1. shape/freeze/resume/finite：工程合同；
2. representation→Program→LoRA→action：机制证据；
3. strict paired400：方法裁决；
4. same/wrong/shuffled/reversed/no-video：因果解释。

若监督Writer持续错位，可以在独立设计中用train24 reward微调Writer；但当前目标先让初次生成LoRA超过150。
“在生成LoRA上做task-local RL”是更后的独立实验，不能混入zero-interaction分数。

## 9. Task experts的作用边界

24个task experts到step2000的direct-expert train成绩为`658/1200`，23/24 tasks非零，证明task-local target LoRA
是有效policy方向。它们不携带same-task视频差异、correct/shuffled/reversed时序、held task泛化或shared Writer
共同积累。soft/hard bank在held仅`15/80`与`3/80`，说明准确重建或正确routing也不能替代held support。

experts可作train24 privileged teacher或几何参照，不能在held部署成为task ID route、nearest bank或第二套LoRA。

## 10. Fixed random A is a reachable-subspace bottleneck, not a rank-8 capacity result

已完成的Direct-Family-B及Visual-Value都只生成B，A固定为step0随机template。对`W=B@A`做低秩QR能量投影后，
old134 validation的32套强LoRA中，逐样本最优rank8平均可保留`.99999946` effective-BA能量，当前固定随机A
行空间只保留`.0195042`。24个step2000 task experts中对应为`.998094/.184501`。所以“rank8不够”与“任意
task都只能写入同一随机输入行空间”是两个不同假设；证据支持后者是严重限制，不能用增加rank或追stable rank救。

同样重要的是，一套在train24 experts上逐target最优的共享rank8 A虽然保留`.940630`训练能量，应用到old134 held
LoRA只保留`.068108`。因此静态SFT/expert A basis并不是合理解法；若当前完整曲线失败，窄候选应由task/video
conditioned Program动态生成A，同时保留现有rank8、direct readout及全部视频结构。该分析只证明可达空间限制，
不证明动态A闭环有效；历史v6-fast完整动态A/B只提供可行性先例，不能替代新的单变量fresh实验。

old134同一task的四条独立视频还提供了关键区分：用其中三条联合拟合rank8 A行空间、对第四条leave-one-video-out，
overall effective-BA能量保留`.9997255`，q/v/action=`.9997540/.9996504/.9992049`，逐task最低仍`.9991674`。
所以强方法所需的A是跨same-task视频高度稳定的task-level functional input subspace；它既不是当前随机固定A，也不应
成为不受约束的video-specific A。现有Dynamic-K的shared Program与singleton Program一致性正好可把这个归纳偏置传给
同一Program上的dynamic-A readout，不需要另造expert bank、task ID route或额外一致性损失。

当前Full-Factor successor严格只开放这一接口：同一`20×8`Program经同一shared projector同时进入四个direct
A residual heads与四个direct B heads；全部zero-init，故step0仍为`A=A_template,B=0`。它不是增加rank、expert
basis或第二套LoRA，也不改变视频表示与训练objective。若它不能把absolute明显拉回至少125附近，就应停止当前
前端的mapper修补，回到v6-fast骨架做受控桥接。

该裁决已经发生：Full-Factor macro50 K1 strict=`91/400`，相对matched fixed-A 88仅净增3。raw A虽与fixed-A
cosine`.735`且norm`1.376x`，B却只有`.062x`，effective BA只有`.245x`且cosine`.0585`。因此“打开dynamic A”
没有把已读到的视频证据写入强policy方向，而是让同一个B20 surrogate找到更弱的factor gauge。当前Program/mapper
小修到此终止；这不否定rank8理论容量或memory/few-shot原则，只否定该完整组合。

当前最小桥接以v6-fast为baseline，只在每条video已经形成原生320个policy slots之后增加跨video集合层。K=1
严格恒等，K>1用mean backbone加task-conditioned centered residual，最后只运行一次原生factor heads。这样既
保留v6的143性能几何，又把昨晚“逐video保序、video间置换不变、不平均最终LoRA”的要求隔离成唯一变量。

该bridge的真实GPU机制门已经接通：K1的76个LoRA tensors逐元素保持native v6，只有197120个集合参数可训练，
base无梯度；K2/K4换位只产生BF16 batched-forward低位差异，而真实video倒序的Program mean abs变化为`.21703`。
因此接下来的full24/closed-loop失败若发生，不能归因于K1底座被代码改坏或Procedure完全忽略顺序。

该bridge随后在macro25 K4 strict得到`130/400`、breadth6，相对old134为`117/13/17` retained/gained/lost。更关键的
nested-dose证据是：same-task centered BA variance约降`9.26x`，但task mean K1→K4 cosine仍`.999832`，全400
relative BA改变量约`.047`。所以few-shot set不是无效；它成功过滤了一部分demo-specific nuisance，却因为位于
完整compiler之后，只能稳定已有错误或不足的task mean，不能为held occupancy增加有用方向。下一轮不能放大这个
residual或调K，而应让多video在compiler承诺最终policy方向之前形成shared semantic Core并比较有序Procedure。

Shared-Core Procedure-Set只前移这一边界：每条video仍独立形成v6 Core与有向Procedure；原生Core reader对
无序Core union联合读出一个shared Core；每条Procedure在该shared Core下独立解释；同一197120参数set只聚合这些
Procedure readouts，之后原生AdaLN/post-fusion/factor heads一次完成LoRA。它避免历史K4 phase alignment，也不把
不同demo的Procedure拼成虚假物理序列。其CPU实现先证明K1严格保留v6和集合/顺序合同，随后完成下述GPU与
closed-loop裁决。

Shared-Core Procedure-Set随后得到K4 strict=`139/400`、breadth6，per-task=`1/4/46/34/0/36/18/0`。它相对
post-compiler K4 130净增9、相对K1 old134净增5，证明“在native compiler最终承诺前共享Core”不是无效讨论；
但四个suite净变化只有`0/-2/+1/+6`，Long1贡献净7，Goal3/Long2仍为0，因此没有解决共同breadth。

决定下一接口的关键不是aggregate，而是matched归零：在同一macro25 checkpoint和first4×8 K4输入上把
Procedure-Set output置零，effective-BA只变化`.000918`，task mean只变化`.000574`；相对K1的`.039674/.016982`
几乎完全由无参数Core union和Procedure mean产生。训练25 macros没有把B20 credit通过后端set变成有用修正。
这淘汰继续训练/放大Procedure-Set，也说明下一步应把可学习集合比较前移到语言token对齐的Semantic Core：先让
多视频共同语义本身可学习，再交给native Core reader；后端有向Procedure只作对称mean，避免同时改变两个接口。

## 11. 连续历史认知

1. v4暴露错误absolute-time/action-phase shortcut；
2. v5/v5.2分离Semantic Core与Procedure，但正确时序可在fusion/compiler后衰减；
3. v6-fast达到143，证明强absolute可达，也暴露架构×recipe耦合和晚期换手；
4. v7/v8/v10、Loom/Core/Prior说明漂亮内部时序或fusion不保证policy-effective方向；
5. Target-Spectral、Lane、Atom、Owned-Factor说明高rank、均匀能量和更多capacity不是目标；
6. SFB union193直接证明single-checkpoint共同积累失败；
7. variance reduction证明functional estimator不是首因；
8. K4和video routing证明视频可被读取，但不自动解决正确时序与shared credit；
9. task experts证明task-level LoRA有效，但expert reconstruction/routing不提供held support；
10. Balanced residual、RLS、Reward-Credit依次定位跨video正交、offline/on-policy错位和native-factor精度边界；
11. uniform rank14证明compression和regeneration可分别破坏old support；
12. Dynamic-K 100与semantic-address 101把断点推进到Program→mapper→policy direction；
13. Visual-Value四点`88/86/86/96`证明video Value能分化effective BA，但当前B20 credit不能辨认held on-policy
    有用方向，且aggregate近稳时仍可发生显著checkpoint churn；该组合终局non-pass。
14. fixed-A能量投影与same-task video LOO把最早可隔离接口推进到task/video-conditioned A row space；它只授权
    Full-Factor单变量实验，不证明动态A一定有效。
15. Full-Factor `91`表明dynamic A在当前rank8 Program+B20下反而诱导tiny-B/weak-BA重参数化；下一轮不能再修
    mapper，而应在v6强底座上隔离检验跨video Program aggregation。
16. V6 Dynamic Slot-Set K4=`130`且same-task方差降`9.26x`，说明post-compiler few-shot nuisance reduction有效，
    但task mean几乎锁死；下一最早接口是compiler之前的shared Core/Procedure解释边界。
17. Shared-Core Procedure-Set K4=`139`证明边界前移有9分matched收益，但trained set只贡献`.000918` BA变化；
    下一最早接口是native Core reader之前的语言对齐Semantic Core共识，而非继续调后端set。
18. Semantic-Core Set的真实world6 profile表明该边界移动没有引入吞吐或显存阻塞：steady`24.277s/macro`、
    reserved`40.758GB`、最长K4 323帧无截断；zero-init output打开后q/k在macro1→2均发生非零更新。它只证明
    functional credit可达该层，是否形成有用task mean仍必须由macro25 strict400裁决。
19. Semantic-Core Set macro25 K4 strict=`135/400`、breadth7，per-task=`1/2/46/30/0/35/20/1`；相对matched
    Shared-Core139为`120/15/19`净`-4`、churn34。归零trained output只改变`.001763` BA，而原始Core correction
    只有`1.8275e-5`、attention entropy/log4=`.999885`。所以不是compiler衰减，而是centered Value在近均匀
    attention下先相消；下一单变量应让共有Semantic Core成为trainable Value，同时保留K1恒等与有向Procedure。
20. Common-Value world6真实profile在同预算同位置下把gradient norm从centered路径约`3.25e-6`打开到
    `.00270--.00280`，macro1→2 q/k delta=`6.55e-6/6.48e-6`且最长K4 323帧吞吐不退化。这验证“centered
    Value构造性相消”是可修复的最早机制断点；fresh macro25的gradient仍为`.00315`、output norm到`.26152`，
    说明它没有在训练中重新关闭。strict却只有`133/400`、breadth6；相对135净`-2`、相对139净`-6`，Long
    相对135净丢7。first4 Core correction/effective-BA改写=`.065856/.053648`，远大于上一轮，而attention
    entropy仍`.999885`。因此它不是“没写进去”，而是offline B20把强common-mean修正写向held on-policy无效且
    task间换手的方向；下一接口应是credit/occupancy对齐，不是继续放大Value、加容量或移动compiler。
21. 同一Common-Value macro25在train-seen 8-task×10 states上的严格output-zero反事实为trained/zero=
    `63/59`，zero→trained=`57 retained / 6 gained / 2 lost`、net`+4`，而held相对Semantic135为net`-2`。
    这把“offline credit完全错误”修正为“task-local credit可形成，但静态Semantic common mean没有形成held可组合
    程序”。下一窄假设应让trainable Value只来自有向Procedure，同时保留冻结shared Core锚点；若仍呈现seen增益/
    held失败，才把最早接口进一步推进到on-policy/generalization credit。
22. Shared-Core Ordered-Procedure Common-Value把有向Procedure correction/effective-BA打开到`.09601/.01397`，但
    K4 strict仍为`139/400`、breadth6，相对matched139=`120 retained / 19 gained / 19 lost`。更关键的是同一
    output-zero在train-seen也为trained/zero=`64/64`、`4 gained / 4 lost`。因此当前B20 credit不是“task-local
    有效、只没泛化”，而是在train与held真实occupancy上都只有方向换手；下一接口必须改变credit来源，不能继续
    调Value幅度、rank、attention、compiler或训练长度。
23. Ordered-Procedure On-Policy Preference的真实task4 smoke首次把binary reward经executed-prefix CFM、native
    compiler与Adam传到同一19.7万FP32 Writer参数：LoRA gradient/Writer gradient=`1.3138e-5/.0008012`，更新后
    effective-BA/fixed-action response=`.00018146/.00557193`。这明确越过历史Reward-Credit的sub-ULP接口，但只
    证明credit可部署，不证明full24共存或held closed-loop改善；cycle1 strict400仍是唯一方法裁决。
24. 首个world5 reward formal的时序证据是等待rank先触发600秒watchdog，PG失败约一分钟后最慢rank才报告OOM；
    所以首因是合法长task尾部超过默认collective timeout，不能反向归因。与此同时，可训练compiler graph跨完整
    Nmc4 replay存活是历史已知且无需保留的额外风险。改为detached LoRA先求cotangent、再从缓存readout单次重解
    compiler后，同一task4的objective、梯度、参数delta、BA/action response全部逐位不变且B8 exit0；正确修复是
    reward专用30分钟timeout加graph生命周期收窄，不是降batch或改变estimator。
25. graph-release reward cycle1已完整证明on-policy credit可改变部署与闭环，但没有形成共同积累：formal为24
    tasks/96 rollouts/64 successes、14 mixed、Writer gradient=`9.0937e-5`，strict=`138/400`、breadth7；相对
    同schedule AS139严格=`120 retained / 18 gained / 19 lost`、churn37。Spatial净`+4`而Long1净`-7`，说明
    proposal含有真实acquisition方向，但task汇合后的shared update覆盖已有support。AS→reward effective-BA只移动
    `.003323` relative-L2、cosine`.999995`且norm ratio`1.000762`，故不能用降LR/scale/rank解释或修复；下一
    可证伪接口是在同一actual Writer parameter delta上加入成功occupancy的一阶support约束。
26. ADSP task4 live smoke证明success-support可以与raw preference共享同一次policy forward：相对raw仍是80次
    forward，只新增16次support backward，preference gradient与BA/action response逐位一致；cycle墙钟仅
    `1.077x`、peak reserved降至`36.774GB`。该单task raw delta本来就满足自己的support，solver按合同严格identity
    fallback；因此smoke只封存实现/效率，不提供“projection能改善多task”的科学证据，后者必须由full24裁决。
27. 首次ADSP full24在任何metric/checkpoint前暴露了旧raw replay builder的真实兼容边界：它对all-success与
    all-failure都只返回summary，因为旧reward会跳过所有homogeneous；新方法却需要all-success产生support-only
    tangent。最窄修复是只让all-failure保持summary-only，all-success完整collate replay。该错误解释task4 mixed
    smoke为何通过，也证明失败不是world6/GPU/NCCL或scientific non-pass；新增data-boundary回归后full CPU=`401 passed`。
28. 修复后的ADSP full24证明actual-delta support projection在真实规模并非identity：22条成功task rows秩22，raw
    AdamW delta违反6条；small dual激活6项后final violation=0，同时保留`.963787` preference descent与`.980958`
    delta energy，raw/projected cosine=`.990433`，BA/action响应非零。它以`1033.501s`、raw的`1.5333x`完成，说明
    该机制在数值、吞吐和部署链上都成立，后续闭环失败不能归因于“约束未生效”或“更新被机械归零”。
29. ADSP strict仍为`138/400`、breadth7、per-task=`3/2/45/30/0/36/21/1`；相对AS139严格=
    `116 retained / 22 gained / 23 lost`、churn45，三项终局门失败。相对raw138=`117/21/21`：Long1从15恢复
    到21、Long净`+6`，但Spatial/Object净`-2/-4`。因此support projection只把能力换手重新分配到另一suite，
    没有形成同一checkpoint共同积累；不能再靠同类constraint、margin、阈值或cycle2小修。
30. first4 AS→ADSP effective-BA relative-L2均值`.002976`，比AS→raw的`.003323`更小；raw→ADSP仅`.001272`、
    cosine`.99999919`，仍翻转42条episode。更接近AS的LoRA不但没有减少held churn，反而从raw的37升到45。
    这直接否决“LoRA距离更小或train24成功prefix一阶loss不增即可代表held support”；下一轮应考虑改变
    representation/compiler的架构级共存接口。memory token可服务layer-aligned LoRA生成，但它不是自动解法，
    历史91--102分memory路线仍要求继承V6的absolute Core、有向Procedure与policy-effective compiler优势。
31. 对V6真实执行图与SHINE/Doc-to-LoRA的逐层机制对齐后，下一未检验接口不是“有没有memory token”，而是
    **真实context中的分层证据是否直接条件化对应policy layer/rank的Procedure readout**。V6现有320 slots只有
    learned layer/rank routing identity；每帧Action Expert内容仍只取最终层50 probes的mean。首轮因此冻结AS139
    全部强路径，从同一次forward旁读18层native probes，经shared rank-query与video内causal delta形成zero-init
    Procedure-query conditioner。它在step0保留V6 LoRA geometry，也把literal memory变成可干净替换carrier的
    后继反事实：只有native probes在correct/reverse/static之前就缺少material差异，才授权加memory；若差异在
    compiler/action后才失效，memory不会自动修复shared credit。
32. V6-LPCP canonical实现证明该假设无需第二次backbone forward或显式memory token即可接线：18层tap只旁读
    现有joint forward，zero-init时完整退化AS139，base与trained K-set冻结，首步打开query projection、第二步
    credit进入probe reader与causal controller。最初“每个layer/rank slot独立跑temporal Transformer”的实现草稿
    会在K4最长条件形成1152条长序列，已在GPU前因效率不合格被否决；当前改为每video一次shared causal
    controller，再轻量汇聚288个slot deltas。该工程/机制通过不提供closed-loop收益证据，仍须真实profile和
    strict400裁决。
33. V6-LPCP live profile进一步证明实现合同在真实full24 B20上成立：gpu02 world3两步macro wall=
    `66.134/61.544s`，K1--K4各6，最长323帧完整，peak reserved=`41.385GB`，0 OOM/nonfinite；第二步reader与
    controller已有非零参数变化。79帧真实载体需要的joint backbone forward=`4`，与native V6预期相同；倒序使
    query-delta/Program relative-L2=`2.0572/.40414`，常量视频query-delta max-abs=`3.38e-8`。这否决了“native
    probes没有任何有向载体证据”这一早期失败分支，因此formal前不换memory token；但它不提供closed-loop收益，
    不能把继承的AS139底座或内部大relative-L2误报成新架构变强。
34. clean detached `515f91e`的world6 fresh macro0->25完整exit0，macro mean=`26.462s`，functional
    `.10115173->.09563028`，K1--K4每macro各6、最长359帧完整、0 OOM/nonfinite；macro25 query projection norm=
    `.142632`，reader/controller相对macro1 delta=`.071824/.052637`。随后K4 generation B8/B16/B32=
    `.221225/.221402/.221500 LoRA/s`，均stable并锁B32，峰值reserved仅`22.628GB`。因此新增接口既持续获得credit，
    又没有明显牺牲部署吞吐；但两者仍都是机制证据，是否超过AS139/143及是否减少churn只认接下来的strict400。
35. V6-LPCP macro25 K4 strict最终为`143/400`、breadth7、per-task=`1/4/48/35/0/38/16/1`、per-suite=
    `5/83/38/17`。相对同schedule AS139严格=`120 retained / 23 gained / 19 lost / 238 both-fail`、churn42、
    net`+4`、p=`.643969`；suite净变化=`+2/+5/+2/-5`，Long1以`7 gains/13 losses`净丢6，Goal3仍0，
    Long2仅解锁1。它count-only追平不同schedule的历史v6-fast143且breadth`6->7`，但同时触发`<144`与lost>10门，
    所以不是突破，不resume50、不补controls或小扫。
36. 全400套AS139/LPCP cache的effective-BA relative-L2 mean/median仅`.002653/.001916`，cosine mean=
    `.99999479`、norm ratio=`.99997391`；first4 LoRA norm、stable rank和q/v/action能量比例不变。first4同task
    correction coherence mean/median=`.61786/.56804`，明显好于“同task corrections近正交”的历史失败形态；
    但Goal3以第二大改写`.004224`和最高coherence`.88373`仍为0，Long1只改`.001324`却净丢6，gained/lost
    改写幅度高度重叠。故LoRA健康与视频集合一致性都不是当前最早缺口。
37. native probe carrier已由reverse/static/one-forward证据通过，conditioner也获得持续梯度；strict与BA联合证据把
    失效接口推进到**conditioned Procedure经冻结fusion/compiler承诺为policy-effective方向，以及blind B20 credit
    对held occupancy的选择**。Program顺序响应可达`.404` relative-L2，最终BA只移动约千分之几；train24
    functional first5/last5也仅`.098880/.097109`，14 tasks改善而10 tasks变差。literal memory若只替换已经通过的
    carrier、仍进入同一Query与同一credit，不会针对该缺口；后继应改变Procedure-to-LoRA commitment或
    policy-aligned shared credit，同时保留V6 absolute与LPCP有序分层读取，不得把143中的继承能力误报成新学习。
38. CV-CSD full24与PCSD使用完全相同的paired outcomes，却把每个active task的同一selected-success replay扩展到
    4个disjoint correct K4 conditions；36/36 LoRA/query gradients非零，cycle wall仅`1.0307x`、三rank负载
    max/min=`1.0828x`。因此134不能归因于额外view破坏rollout、rank分工失衡、某个view零梯度或吞吐妥协。
39. CV-CSD strict=`134/400`、breadth7；相对LPCP=`122 retained / 12 gained / 21 lost`、churn33，四suite
    `-2/-4/-2/-1`全部下降；相对PCSD=`115/19/20`、churn39。aggregate只差1仍有39条success-set翻转，进一步
    证明单一总分接近不代表能力稳定，必须保留paired episode集合与逐suite分析。
40. CV-CSD/LPCP全400 BA改写mean=`.000683702`且gained/lost不可分；FP64四correct-view增量cosine=`.000205`、
    energy ratio=`.250155`，CV-CSD/PCSD也为`-.001908/.248578`。cross-video监督没有让部署修正形成共向程序，
    所以最早接口已从credit coverage推进到query-only commitment。下一架构应把task/video Program直接提交到
    layer/rank/target-owned写出槽；memory若使用，只是实现这一接口的手段，不能成为static bypass或旧低分路线复刻。
41. 新SFMC authority把本轮memory精确定义为同一cached condition的`FrozenSet(P_LPCP)-FrozenSet(P_AS139)`：
    它是K-set之后、layer/rank对齐、query-disabled/constant近零的Procedure innovation，不是literal backbone token、
    raw feature平均或第二套LoRA。exact language只选择四个连续semantic bases；八factor families把innovation写到
    冻结V6 `W2` output basis之前。这样step0 exact LPCP143，并同时继承SFB已证明的软语义分工、LPCP已证明的
    有序carrier和CV-CSD已证明可高效运行的cross-video success credit；这是设计推导，性能仍待真实closed-loop。
42. SFMC已原位实现到唯一Writer/reward/evaluator graph，而不是新增平行Writer：旧V6 FactorHead只拆出同一
    `W1→GELU` hidden与`W2` output边界，八family的4-way memory maps在hidden处提交。真实枚举trainable恰为
    `2,164,224`；所有maps zero-init使step0逐tensor exact LPCP，language只作soft Q/K address，`M=0`不能产生
    residual。CPU已验证K-set permutation、constant-memory zero、factor family/layer/rank ownership、cycle1先开maps/
    后续router才得梯度，以及历史LPCP checkpoint只完整缺失新增12 tensors。architecture guard无hard violation；
    这些只证明实现合同，不提供closed-loop结论。

负结果只淘汰实际受检验的组合。新设计必须保留未被否定且已接通的机制，只改变有证据指向的最早接口。

## 12. 选择方法与工程原则

Task-Grounded Visual-Value实现与profile新增的边界：同一次joint forward中的task-conditioned raw patch Value可在
不更新frozen policy、不增加第二次backbone forward的前提下接入visual D/G与完整LoRA梯度链。直接实现的
matched吞吐为`1.2603x`；截断frozen prefix无用反向图并合并projection/reader后为`1.061727x`，functional
loss逐位一致，科学主变量不变。这只证明机制与效率合同健康，不能证明视频知识更有用或task drift改善；后者只认
fresh single-checkpoint strict closed-loop。

同一生成图在正式macro25 checkpoint上的K1 B8/B16/B32 deployment LoRA/s为
`.984266/.976097/.971736`，均stable、最长视频覆盖且0 OOM，按规则锁B8。它与Direct-Family-B B8 `.977325`
接近，只支持新增视觉Value没有造成deployment吞吐退化；不能推断closed-loop更强。

- 不用loss、small panel、union、norm、rank、cosine或内部margin选择最终checkpoint；
- 不靠rank/scale/seed/dtype/temperature小扫救失败checkpoint；
- 不恢复language-only LoRA bypass、平均LoRA、checkpoint融合或held expert route；
- 不为了逐元素低位一致固定batch1、重复forward、扩dtype、逐tensor扫描或内容hash；
- GPU上限6张但不要求凑满；每次双节点live检查，在单节点用真正有益且不会明显干扰他人的设备；
- profile/smoke只证明工程合同，formal结果必须保留checkpoint、raw rows、aggregate和completion；
- 一次尽量只改一个主要因果变量，并尽快回到真实paired400。
