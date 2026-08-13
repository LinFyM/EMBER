# EMBER Findings

更新时间：2026-08-14。本文只保留跨架构仍成立、会约束下一轮判断的结论。逐方法严格结果和禁止重复项见
`docs/research_history.md`；当前run只取`docs/active_session_handoff.md`。

## 1. 当前经验边界

长期目标是同一shared Writer、同一single checkpoint的strict paired correct严格`>150/400`，并同时保持
视频因果性、task breadth和多任务共同积累。目前尚未达到。

| 方法 | correct | same | wrong | shuffled | reversed | 主要结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v5.2 old | 132 | 138 | 74 | 82 | 83 | 视频内容特异性强，但absolute不足 |
| v5.2 task-complete | 120 | 109 | 107 | 111 | 124 | recipe削弱Procedure传递 |
| v6 old | 121 | 122 | 111 | 84 | 47 | 顺序影响能进闭环，但absolute与稳定性不足 |
| v6-fast task-complete | 143 | 135 | 125 | 128 | 129 | 历史最佳single checkpoint，仍未过150且视频margin弱 |
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
