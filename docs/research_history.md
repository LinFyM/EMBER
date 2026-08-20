# EMBER Research History

本文是EMBER唯一的架构与实验ledger。它回答“改了什么、真实结果是什么、实际证明了什么、哪条假设被否决”，
不记录当前执行计划，也不授权恢复旧实验。

整理前完整设计语料封存在Git commit `8553b613de7791df50e0f3ef85678fcaca1cac0c`；更早v4--rank14语料可追溯
至`3a6f801d08facb3e855ab24f84e0b53cb8802e88`及其祖先。正式run contract、raw rows、checkpoint、completion和
analysis保留在`runs/outputs/`，其事实优先于叙述性文档。

## 1. 固定问题与证据口径

EMBER输入exact task language与一条或多条action-hidden正确teacher videos，shared Writer一次生成一套完整task
LoRA，挂载到冻结π0.5-LIBERO source policy后，从未见初始化闭环完成任务。

正式选择只认single-checkpoint strict paired400。结果需要同时解释：

- correct、per-task、per-suite和breadth；
- 相对最接近baseline的retained/gained/lost、churn和Jaccard；
- 相邻checkpoint是否共同积累而非轮换；
- same-task不同video鲁棒性；
- correct相对wrong、shuffled、reversed、no-video的因果优势；
- Program→LoRA→effective BA→action→closed-loop的最早断点。

训练loss、functional loss、LoRA norm/rank/cosine、reconstruction、hidden margin和small panel只作诊断。五臂统一
顺序为`correct/same-task-other/cross-suite-wrong/shuffled/reversed`。

## 2. 固定基线

| 基线 | 严格结果 | 解释边界 |
| --- | ---: | --- |
| generic π0.5 | `0/400` | 原始policy缺少LIBERO embodiment能力，不是Writer结论 |
| filtered frozen source base | `48/400` | 71个不重叠source tasks建立的共享起点 |
| shared mixed-task Source-SFT | `109/400` | 读目标actions的privileged参考，不是同信息墙方法 |
| v5.2 old | `132/138/74/82/83` | 视频内容特异性强，但absolute不足 |
| v5.2 task-complete | `120/109/107/111/124` | recipe改变Procedure传递，absolute与margin都退化 |
| v6 old | `121/122/111/84/47` | 时序差异能进闭环，但absolute低且checkpoint漂移 |
| v6-fast task-complete | `143/135/125/128/129` | 有完整五臂的历史最好；margin弱且后续训练回落 |
| V6-LPCP | K4 correct `143/400`, breadth7 | 最强可复用carrier；无六臂资格，改写仍在AS139邻域 |

v5.2/v6的交叉recipe结果证明架构与recipe强耦合，不能把“old”或“task-complete”外推为普遍好坏。

## 3. 因果演进账本

### 3.1 从时序敏感到v6强基线

| 方法 | 最强证据 | 接通的接口 | 最早失败与保留结论 |
| --- | ---: | --- | --- |
| Action-Forecast v4 | best`109`; 五臂`109/104/99/148/126` | 视频顺序可显著改变LoRA与action | 学到absolute-time/action-phase shortcut，shuffle反而最好；时序敏感不等于正确理解 |
| Semantic Core + Procedure v5 | best`115`; `115/108/74/113/114` | task语义与有序Procedure可分开 | Procedure差异在fusion/compiler衰减，correct与shuffle/reverse未拉开 |
| v5.1/v5.2 | best`132` | language-axis Core、task-token evidence和Procedure可提高absolute或视频margin | 架构与训练配方耦合，单独追margin或续训均不可靠 |
| v6-fast | best`143` | task-grounded semantic set、visual-transition Procedure和高增益decoder形成强图 | 后续450/500/550/600=`131/130/132/126`，能力没有稳定累积 |
| v7/v8/v10 | 均低于v6 | 更强时序/fusion结构能进入LoRA | 更漂亮内部因果会放大demo nuisance，不能自动提高closed-loop |
| Loom/Recenter/Core/Prior | 均未过143 | patch correspondence、去DC、静/动态分解可实现 | 未解决policy-effective conditional credit；不再靠同类fusion补丁重复 |

### 3.2 LoRA几何、容量、few-shot与共享表示

| 方法 | 最强证据 | 接通的接口 | 最早失败与保留结论 |
| --- | ---: | --- | --- |
| Target-Spectral | best`34` | 谱/秩正则能强烈改变LoRA | 高stable-rank、均匀能量破坏q-dominant policy manifold |
| CV-ADR RAW/GROUP4 | `117/110` | 可构造更大、更coherent更新 | video gradient主效应约`.1%`，query/flow variation和credit错位主导 |
| Target-Bound | best`120` | remove-A/D、reversal和动态路径8/8达门 | shared factor coexistence与checkpoint漂移仍失败；视频并非未使用 |
| Semantic Factor-Basis | single`127`, union`193` | 多checkpoint合起来含更多task能力 | union-single gap66直接证明能力换手，union不能冒充模型 |
| Variance-reduced estimator | best`126` | exact-Beta/antithetic改善functional estimator | held functional更好而closed-loop更差，flow MC方差不是主因 |
| Semantic Direction Store | best`129` | 独立store改善早期acquisition | 同分checkpoint breadth不同，Program→factor压缩和漂移仍在 |
| Policy-Target-Owned Factor | best`99` | 解除38-target共享，增加target ownership | action与absolute显著下降；ownership不是充分条件 |
| Policy-Lane Hyperdecoder | best`70` | 约10条lanes和SFT量级专门化 | video BA能量仅约`.02%`，容量健康不能替代动态credit |
| Policy-Wide Atom Dictionary | best`80` | 64 atoms被广泛调用 | effective LoRA仍近rank1；更多atoms/rank/正交loss不是答案 |
| Factorized Condition-Kernel | best`49` | full-rank stable kernel与跨video差异可形成 | LoRA比direct SFT小约200×、近identity；低增益decoder是局部瓶颈 |
| Few-Shot Invariant K4 | best`108` | K4 permutation、same/LOO/wrong/order路径工作 | full24 retention约`.043`，few-shot不自动解决共享credit |
| K4 Policy-Layer Trace | best`99` | all-layer trace可形成correct>wrong | 单位化放大低能DCT高频约140×，reversal仍高 |
| Energy-Preserving Trace | best`85` | 修复频率真实能量 | correct/wrong由`99/57`缩到`85/80`，能量保真不是语义保真 |
| Evidence-Factorized Trace | best`84` | trace→BA→action闭合 | shared Reader retention约`.05`，参数隔离仍不够 |
| Sparse Semantic Expert | best`78` | expert-local retention提高 | language route固定owner，wrong/order更成功 |
| Grounded-Video Expert | best`88` | video route、BA、action和rank均material | correct无margin且task轮换；视频敏感+隔离仍不充分 |
| K4 Phase-Aligned v6 | best`108`, reversed`121` | 视频没有被忽略 | Program retention约`.04`且reversed更好，phase alignment不足 |

### 3.3 Task experts、manifold与早期reward credit

| 方法 | 最强证据 | 接通的接口 | 最早失败与保留结论 |
| --- | ---: | --- | --- |
| AS125 + semantic-progress RL | `97→104→102` | failure trajectory提供action-free credit | breadth下降、继续训练换手 |
| Program-Credit RL | `106` | CRN与Program gradient可到达 | task cotangent近正交，被shared condition map压成common update |
| SFT-Anchored Tangent Basis | `143→142` | 强warm-start上reward update可运行 | gained/lost=`20/21`，保持分数不是净改进 |
| task experts step2000 | train`658/1200`, 23/24非零 | task-local SFT LoRA是policy-effective target | 同task target对所有video恒定，不提供时序、video specificity或held共享能力 |
| addressless Expert-Manifold | `48/400` | raw expert reconstruction可达SFT量级norm | decoder后topology identity坍缩，nearest cosine约`.008` |
| topology-address binding | `75/400` | 静态chunk/rank地址可调制video Value | 输出仍task-common、absolute低 |
| Causal Barycentric | `63/400` | temporal coefficient和raw-factor组合可部署 | raw A/B组合产生cross terms，不保持effective BA |
| soft/hard expert bank | held`15/80` / `3/80` | hard compiler可近精确复现所选expert | 24-expert dictionary对held无support，不外推所有manifold监督 |
| v6-prior whole-LoRA objective | `134→127→105→123` | 冻结上游、只训写出端可高吞吐 | norm/方向吸引主要径向收缩，macro0仍最好 |
| Expert-Component Projection | `134→133→120` | expert component按构造提高 | orthogonal drift增大，macro25净`-14` |
| Condition-Local Tangent Tube | `134→131` | relative tube半径约束有效 | direction ratio`108.93/126.88`、completion`0/24`，只压小未旋正 |
| Expert-Flow Teacher Audit | no rollout | expert residual gradient非冗余 | flow loss仅`2/24` tasks、`0/4` suites优于baseline |
| Frozen-v6 residual v1 | no rollout | correct retention`.807966`、A/B/action closure成立 | DC key condition`1315.33`、null15/24，未训练 |
| Balanced DC-Causal v2 | `134/140/139`, union`153` | 13/13机制、24/24 null、部署闭合 | 10→25=`12/13`换手；50-video correction近随机正交 |
| Exact Anchored Reconciliation | `134→140` | offline RLS row保留可实现 | full400 lost15；offline row不保护held on-policy occupancy |
| Reward-Credit Program Cotangent | cycle1`134`, `14/14`换手 | on-policy reward产生连续Program tangent | q/v约`1e-8 RMS`，低于native BF16 factor约`1e-4` ULP |
| uniform pivot-rank14 | online`128`; compiler-only`138` | compression与online regeneration可去混杂 | old→compiler lost15，compiler→online lost23；两者独立破坏support |

task-expert formal root：
`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`。compiler-only rank14 root：
`runs/outputs/pi05_v6_qv_rank_reserved_compiler_only_old134_to_rank14_correct400_20260811`。

### 3.4 Policy key、candidate guard与blind credit

| 方法 | 最强证据 | 接通的接口 | 最早失败与保留结论 |
| --- | ---: | --- | --- |
| Policy-Innovation Consensus Key | no rollout | same/order、full48、Program→LoRA→action与吞吐闭合 | full48 condition=`483.62>200`，static common mode key collision |
| Policy-Innovation Goal-Causal Key | `138/400`, breadth6 | condition降到`152.61`，native切向写出闭合 | old→macro10=`118/20/16`、churn36；blind B20不覆盖held support |
| On-Policy Success-Guarded Program Credit | no rollout | success-prefix guard执行图和schema可实现 | world6长尾触发NCCL watchdog，wall至少`1.912x>1.25x` |
| Success-Key Nullspace Consolidation | `137/400`, breadth7 | 4/4 success key硬保护Program/LoRA/action | old→SKNC=`121/16/13`；train single-video key不外推held occupancy |
| Negative-Preserving Candidate Guard | `135/400`, breadth5 | candidate与negative point guard同一affine修正闭合 | old→NPCG=`117/18/17`；point guard不保护held support |
| Cross-Video Equivariant Candidate Guard | directional`131/400`, breadth6 | companion E、negative和guard数值闭合 | K2 outcome低位翻转被hard equality放大；NPCG→CVEG lost21 |
| Paired Candidate-Update Guard | no mechanism result | canonical paired candidate graph实现 | world4 Phase-A前wall下界`2.256x>1.5x`；只淘汰静态执行合同 |
| Work-Queue Candidate Guard | mechanism stop | 动态queue完成96 paired rollouts，guard/projection/action接通 | 19项仅negative-null失败；final composition丢失negative抑制 |
| Shared Reward-Tangent Projection | no mechanism result | fixed-landmark reward projection可构造 | 两个clean world3均在mixed CFM forward OOM；只淘汰当前graph lifetime |
| Paired-Video Joint Functional Credit | no formal | 24/24双view下降、negative ratio`.0727`、action和wall通过 | full96 regularized condition=`597.861>200` |
| Causal-Goal Interaction Key | no formal | correct/reverse等能正交key将条件改善到`270.188` | full96仍高于200；不是video读取或compiler断裂 |
| Magnitude-Gated Causal Interaction | `134/400`, breadth6 | condition=`174.813`并接通paired video、negative、BA/action | old→MGCI=`114/20/20`；key谱修复仍未改变blind credit换手 |

这一阶段连续证明：保护train key、负样本point、candidate局部可行集或更好condition number，都不能替代held
on-policy有用方向。工程non-pass只淘汰相应执行图，不应被写成科学假设的普遍否决。

### 3.5 Dynamic-K、rank8 SHINE-like路线与V6 bridge

| 方法 | 最强证据 | 接通的接口 | 最早失败与保留结论 |
| --- | ---: | --- | --- |
| Dynamic-K Backbone-Memory rank8 | `100/400`, breadth4 | 图文+Action probes中8 memory tokens、动态K和完整rank8 LoRA可训可部署 | 去掉absolute Semantic Core后任务集中，task-mean BA offdiag`.702` |
| Dynamic-K Semantic-Address rank8 | `101/400`, breadth6 | absolute Core只作temporal Query，static bypass为零 | task-mean BA更同向到`.776`，Query address不足 |
| semantic mapper stage probe | no rollout | M2P/final/shared-project task offdiag`.492/.529/.530` | family hidden/B升到`.634/.779`，失效在nonlinear family readout |
| Direct-Family-B rank8 | K1`102`, K4`98`, breadth5 | direct readout和nested K可部署；K4降低same-task方差`6.3x` | set稳定错误task mean，无新task；mapper/K小修终止 |
| Task-Grounded Visual-Value rank8 | `88/86/86/96` | language查询raw patch Value进入有向D/G，action norm恢复 | top3占95.83%，相邻churn40，持续任务集中 |
| fixed-A reachable-subspace audit | old134 fixed/optimal rank8=`.0195/.999999` | rank8本身足以表达强BA | 随机fixed-A开放极窄右子空间；不支持静态expert basis |
| Task-Grounded Full-Factor rank8 | `91/400`, breadth5 | dynamic A/B相对matched fixed-A净增3 | tiny-B、BA norm`.245x`且近正交；offline loss接受弱重参数化 |
| V6 Dynamic Slot-Set Bridge | K4`130`, breadth6 | same-task BA方差降低`9.26x`且保留old support | post-compiler set令K1/K4 task mean cosine`.999832`，只稳定nuisance |
| V6 Shared-Core Procedure-Set | K4`139`, breadth6 | 更早Core union比130净增9，方差降低`9.69x` | Procedure-Set BA只改`.000918`，增益集中Long1 |
| V6 Semantic-Core Set | K4`135`, breadth7 | trainable set进入native policy topology | attention近均匀、correction`1.83e-5`，相对139净`-4` |
| V6 Semantic-Core Common-Value | K4`133`, breadth6 | BA改写打开到`.053648` | 少量task-local credit未形成held可组合Program，Long净丢7 |
| V6 Shared-Core Ordered-Procedure | K4`139`, breadth6 | 有向Procedure BA改写`.01397`，保持absolute | train-seen trained/zero均64/64且4/4换手，B20无净收益 |
| V6 Ordered-Procedure On-Policy Preference | K4`138`, breadth7 | 18条paired新success，q/k/output→BA/action全链路非零 | 相对AS139=`120/18/19`，Spatial增益由Long损失支付 |
| Actual-Delta Success-Support Projection | K4`138`, breadth7 | 6条raw violation投影后为0，保留`.964/.981` descent/energy | 相对AS=`116/22/23`、churn45；一阶local support不代表held共存 |
| V6-LPCP | K4`143`, breadth7 | 同一native context forward的18层Action-probe carrier读取顺序/动态，追平历史absolute | 相对AS139=`120/23/19`、churn42；BA只改`.002653`，仍是baseline邻域换手 |

这条路线给出两个稳定结论：多视频集合聚合可以显著降低same-task nuisance，但不会自动修正task mean；V6的Core、
Procedure和native rank16 compiler仍是最强可复用结构，而不是不可逾越的神秘checkpoint。

### 3.6 LPCP之后：跨video Program、native commitment与reward方向

| 方法 | 最强证据 | 接通的接口 | 最早失败与保留结论 |
| --- | ---: | --- | --- |
| Paired Causal Success Distillation | `135/400`, breadth6 | 9条unique success使query/BA/action非零 | 相对LPCP lost22；四K4增量cosine约0、energy`.2486` |
| Cross-Video Causal Success Distillation | `134/400`, breadth7 | 同一success在4个disjoint K4下完整反传，wall仅`1.031x` | 四view增量cosine`.000205`、energy`.250155`，query-only仍近正交 |
| Semantic Factor-Memory Commitment | `144/400`, breadth7 | layer/rank innovation memory和8 family maps获得reward credit | lost15/churn31；BA改写`2.899e-7`，主要是video-local ULP crossing |
| Gradient-Open Semantic Commitment | `141/400`, breadth7 | W1 anchors打开router，BA改写放大33.3×且几乎全样本native非零 | first4 cosine`.000144`、energy`.250124`；写出打开仍跨video近正交 |
| Causal Coefficient Transport | `142/400`, breadth6 | train task4两系数共同方向`.576/.682` | held residual到BA缩小249.9×，native compiler未承诺held Program |
| Native Probe-Value Commitment | `136/400`, breadth6 | validation8共同写出`.449/.572`，关闭CCT held compiler断裂 | full24后task4坍塌到`.057/.295`，persistent failures改写最大 |
| Pre-Addressed Factor-Selective Value | mechanism stop | 八family可独立选择native Value | train24 address effective rank`2.16`，validation8仅3/8过门 |
| Shared Joint Native-Value Gate | mechanism stop | hidden跨videocosine约`.94` | frozen W2后factor`.021/.266`、action`.0027`，validation8仅2/8 |
| Direct Joint Native-Factor Residual | `136/400`, breadth6 | direct heads绕过W2，post-full24 validation8 BA 8/8健康 | 相对LPCP=`120/16/23`；生成端接通后credit仍选择错误held方向 |
| Paired Common-State Preference | mechanism stop | 同一初态共同观测的winner/loser credit可形成 | 三anchors只有task15通过；task9 held/train`.105x`，task18跨video弱 |
| Successful-Occupancy Counterfactual Preference | mechanism stop | winner success occupancy形成强train/held方向 | 两臂batch形不同造成contrast混杂，wall过高，task9 held/train`.118x` |
| Matched-Batch Stratified Occupancy Preference | mechanism stop | 同B8、8进度strata修复混杂并降wall | task15/18 Adam后margin增加，task9 held/train`.1096x` |
| Adam-Radius Euclidean Commitment | mechanism stop | final方向严格回到raw shared gradient | Adam radius为local gradient L2的4,294--7,988×，三task仅1/4 views下降 |
| All-View Monotone Backtracking | mechanism stop | exact evaluator上搜索raw-mean ray | task18通过；task9需1/1024且held弱；task15 no-op |
| Maximum-Margin Common Descent | mechanism stop | continuous worst margin提高1.22--1.36× | native finite step与held transport仅1/3通过 |
| Preconditioned All-View Backtracking | mechanism stop | 在实际Adam candidate ray搜索 | task9 held弱，task15/18 no-op；0/3通过 |
| Anchored Linear-B Native Value | mechanism stop | fixed A线性化消除A/B gauge cross-term | 仅task18全门；task9 no-op、task15 raw-B coherence失败 |
| Native-Zero Residual Bank | mechanism stop | rank32 second-B zero bank令2/3 anchors native/held coherence健康 | task9 no-op且wall超门；不否定rank8/16或memory |
| Native Endpoint Action Preference | mechanism stop | 10-step endpoint把task9跨video从`.286/.448`提到`.846/.865` | direct rows令held/train幅度降到`.234x` |
| Task-Complete Endpoint Coexistence | mechanism stop | 首次把三task endpoint credit放进shared update | norm差41.45×、task cosine均值`-.145`，global exact no-op |
| Capacity-Matched Action-Probe Grid | mechanism stop | same-task cosine显著提高，native coverage由8/12到10/12 | task15幅度主导、task18反向，0/11 scales全门 |

这一长链逐步关闭了“video没读到、gradient没开、BF16没写出、A/B gauge错误、batch混杂、endpoint不对”等局部
解释。关闭局部断点后，最早问题不断后移到两点：task/video Program是否处于共同policy坐标；shared reward update
是否沿held on-policy有用方向并保留已有support。

### 3.7 Literal memory、训练剂量与最高单点

| 方法 | strict轨迹 | 已解决 | 终局结论 |
| --- | ---: | --- | --- |
| Capacity-Matched Backbone-Memory Grid | no strict | 37个layer-matched one-way memory使3/3 raw、12/12 native与validation8 held transport通过 | full24被task38以54.45×支配，best17/24 margins，exact no-op |
| Content-First Memory Grid | no strict | zero gate移到完整content后，validation8仍8/8，raw/final与native全通 | full24 cross-task cosine`.0092`，task38 58.73×，best14/24，exact no-op |
| Unit-Secant Endpoint Preference | mechanism stop | task38 dominance降到6.15×，3/3 raw、12/12 deployed下降 | task34 raw four-view仍2/4；冲突在policy/action Jacobian后 |
| Unit-Secant Finite Commitment | no strict | full24 actual Adam使17/20 margins下降 | hard 20/20门拒绝并恢复LPCP；只否定该绝对保存门 |
| Unit-Secant Direct Commitment | `138/400`, breadth6 | 不再用20/20作硬门；memory-conditioned rank32写出material | 相对LPCP=`120/18/23`，task38仍以6.27×支配，Goal3仍0 |
| Median-Capped Task Tangent | `142→142→136`; breadth`7→6→7` | median upper cap去除幅度outlier，24/25 groups训练且跨video更新`.99`级一致 | 多训后仍下降；surrogate对persistent failures改写最大 |
| Successful-Expert Occupancy Distillation | `129→135→143→136`; breadth`6→6→5→5` | dense expert occupancy、同task跨video更新`.994`一致，29条成功trajectory | cycle3峰值不稳定；cross-task gradient mean`.053`，不能保留held support |
| Gradient-Open Memory Query | `151→135→131`; breadth`6→6→6` | learned memory query相对matched fixed-query135产生显著+16，same-task更新`.993`一致 | 相邻两步持续回落；memory有价值，但当前independent rank32 direct-B tail不稳定 |

GOMQ cycle2是第一步真实memory update：`151/400`、per-task=`0/3/47/34/0/40/26/1`，相对LPCP为
`126 retained / 25 gained / 17 lost`，相对matched fixed-query SEOD135为`122/29/13`、net`+16`、
McNemar `p=.0195205`。这是真实closed-loop正证据，不是内部指标。

cycle3=`135/400`，cycle2→3=`122/13/29`、churn42；cycle4=`131/400`，cycle3→4=`116/15/19`、churn34，
cycle2→4净`-20`。三次相邻更新的same-task four-K4 BA cosine/energy持续约`.993/.993`，训练内多数views下降，
因此回落不能归因于video-set相消、漏梯度或训练量不足。GOMQ没有运行六臂，不能宣称通过或失败视频因果资格。

整理后的held memory gate进一步校正了一个容易过度解读的口径：learned memory本身的held/train L2 ratio为
`1.1116`，但隔离出的memory-only same-task cosine/energy只有`.1266/.3432`；完整residual才是`.9829/.9854`。
因此`.993`证明的是相邻shared BA update对four-K4 conditions高度一致，不证明isolated learned memory已经形成高
coherence高层Program。memory的`+16`闭环贡献仍成立，但Program质量、旧residual support和reward credit必须分开。

这一结果同时否定两个极端叙述：不能说memory token没用，也不能说151证明完整方法成功。被否决的是“learned
one-way memory + 当前K4/reward + independent rank32 direct-B shared tail连续更新即可稳定积累”的组合。

### 3.8 Layer-Matched Memory Program Compiler v1

LMMPC-v1把V6 Core/Procedure、16个one-way layer/rank memory、地址保持dynamic-K、同一20x16 axial M2P与共同训练的
native rank16 A/B接成fresh统一Writer。它证明这条完整工程图可在每macro等权train24、K1--K4平衡下稳定训练和部署，
但absolute明显低于继承旧support的V6/LPCP：

| checkpoint | strict | breadth | per-task | 相对LPCP143 retained/gained/lost |
| --- | ---: | ---: | --- | --- |
| macro25 | `81/400` | 5 | `2/0/32/3/0/39/5/0` | `69/12/74` |
| macro50 | `101/400` | 5 | `3/1/48/0/3/46/0/0` | `83/18/60` |

macro25→50为`68 retained / 33 gained / 13 lost`、churn46、net`+20`；训练量尚能增分，但Object3和Long1等能力被换手。
functional/matching从`.115512/.021139`降到`.105596/.001378`，loss不能选择闭环。

逐stage held cross-task cosine在macro25→50分别为：Core`.92185→.84859`、Procedure`.78993→.95031`、per-video
memory`.71089→.73133`、Core-fused`.49402→.44842`、compiled`.68476→.75632`、final BA`.74886→.64307`。同task
four-K4 final BA约`.974--.998`一致，排除了K-set首先相消；Procedure却明显趋同。

实现counterfactual最终定位到reader：它为每个`t`建立query却只取`attended[:, -1]`，所以只有`P_last`直接参与；随后
独立相加的首尾memory endpoint又绕过Procedure。macro25/50 attention只占reader输出`.02753/.02461`，endpoint是它的
`41.04x/45.73x`。重复`P_last`替换整条Procedure逐元素不改H；macro50换成另一task Procedure时direct H只改`.125%`。
reverse endpoint严格为负，故旧`0.5*(correct-reverse)`的强时序响应主要是结构反号，不是高层阶段理解。

因此LMMPC-v1不续macro75/100。该结果只否定“last-query + independent endpoint + pure-odd channel + language/order
matching”这一Procedure路径，不否定四流、memory token、dynamic-K、axial M2P、rank16或生成LoRA。后续v2在相同
主链内改为固定layer/rank地址读取完整Procedure keys、centered dynamic memory Values，并回到correct-order dense
functional-only训练。

### 3.9 Layer-Matched Memory Program Compiler v2

v2只替换v1最早失败的Procedure reader：每个固定layer/rank地址用task-level final Procedure和地址作Query，读取完整
Procedure stage Keys与同地址centered native memory Values；删除独立endpoint、内部correct-minus-reverse和matching
loss。机制确认重复`P_last`不再复现H，correct/reverse不是硬反号，constant/K置换/八family/source-zero-grad通过。

同一fresh world5 run的正式结果为：

| checkpoint | strict | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `71/400` | 6 | `2/0/31/2/0/34/1/1` | `2/33/34/2` |
| macro50 | `73/400` | 6 | `1/0/35/13/5/15/4/0` | `1/48/20/4` |

25→50=`42 retained / 31 gained / 29 lost`、churn60、Jaccard`.411765`、net`+2`；Object suite净增15而Goal净丢14，
并非共同积累。macro50相对LPCP143=`61/12/82`、churn94；相对v1 macro50 101=`49/24/52`、churn76。训练loss在
macro41--50约`.112`平台，故不续macro75或controls。

validation8逐stage定位显示reader本身已经有内容：macro50 Procedure correct/reverse relative-L2=`.820147`，H_set=
`.398023`，Core-fused=`.257271`；Core-fused同task不同K4 cosine=`.992207`、between-task=`.338088`。但unbounded M2P
block0相对输入改写`4.50034x`，把order压到`.150168`、between-task升到`.494320`；block1再改写`1.75311x`，最终为
`.093797/.655989`。effective BA order差异仅`.086224`、different-K4 cosine=`.999775`。output RMSNorm单独几乎不
改变task/order指标，因此最早失败接口是axial M2P覆盖已经形成的parameter-aligned Program。

只读counterfactual把最终proposal的逐cell correction限制为anchor RMS的`.25/.5x`，分别得到order=
`.247873/.230772`、between-task=`.360797/.405605`、same-task K4=`.992778/.993867`。它支持后续v3把M2P改成
identity-anchored bounded refinement；不构成fresh closed-loop成绩。v2负结果不否定stage reader、memory token、
Dynamic-K、rank16或axial通信本身，只否定unbounded proposal直接取代Core-fused grid。

### 3.10 Layer-Matched Memory Program Compiler v3

v3只把v2的unbounded axial M2P改成逐cell bounded commitment；Core/Procedure、stage reader、K-set、Core fusion、
native rank16 A/B和B20 functional recipe不变。clean `af76558075315b6ea954e60feff44dfaac0637e3`的同一world3 run
完成macro25与exact-resume macro50：

| checkpoint | strict | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `102/400` | 5 | `2/0/47/8/0/37/8/0` | `2/55/37/8` |
| macro50 | `60/400` | 6 | `2/0/24/2/1/26/5/0` | `2/26/27/5` |

25→50=`46 retained / 14 gained / 56 lost / 284 both-fail`、churn70、net`-42`、Jaccard`.396552`；macro50相对
LPCP143=`46/14/97`、churn111、net`-83`。这不是尚未充分训练的共同上升：继续25个macro后Object/Goal/Long均降，
v3不续macro75或六臂。

机制上v3确实关闭了v2断点。macro25/50 raw axial proposal相对Core-fused anchor仍为`15.29x/12.83x`，但实际
commitment只有`.24979/.24953x`；Core-fused→compiled的order仅`.3338→.3208`与`.2904→.2714`。新的最早异常位于
K-set：raw set output相对per-video mean的relative-L2为`10.188/5.831`，between-task cosine由`.6543/.7672`升到
`.9025/.9218`，within-task condition cosine反而由`.9954/.9967`降到`.9649/.9804`。Core fusion随后把between-task
恢复到`.6462/.6502`，因此不是更早断点。

只读mean-only与bounded counterfactual均改善task分离和same-task coherence，但降低部分order幅度，说明raw branch
含有有用有向成分而不应删除。后续v4因此只把K-set correction变为per-video mean-anchored逐cell bounded
commitment，fresh gate初始`.25`、最大`.5`；不改变rank、loss、时序监督或训练recipe。v3负结果只否定
`bounded M2P + unbounded nonlinear K-set + 当前functional credit`组合，不否定memory、Dynamic-K、rank16或
cross-video learned aggregation。

### 3.11 Layer-Matched Memory Program Compiler v4

v4只把v3的K-set raw nonlinear correction改成per-video mean-anchored逐cell bounded commitment；Core/Procedure、
layer/rank memory reader、Core fusion、bounded M2P、native rank16 FactorHeads和B20 functional-only recipe不变。
同一fresh world4 run完成macro25及exact-resume macro50：

| checkpoint | strict | breadth | per-task | per-suite |
| --- | ---: | ---: | --- | --- |
| macro25 | `104/400` | 6 | `3/0/41/14/2/39/5/0` | `3/55/41/5` |
| macro50 | `102/400` | 6 | `1/0/39/4/0/46/10/2` | `1/43/46/12` |

25→50=`77 retained / 25 gained / 27 lost / 271 both-fail`、churn52、net`-2`、Jaccard`.596899`。macro50相对
LPCP143=`79/23/64`、churn87、net`-41`。Object suite净丢12，Goal/Long分别净增5/7，仍是task换手；不续macro75或
六臂。BA并未冻结：first4 norm由`27.2849→49.0819`，25→50 cosine`.613461`、relative-L2`1.440609`，所以绝对
低分不能归因于Writer没有产生material update。

validation8逐stage显示raw Procedure correct/reverse relative-L2在macro2/25/50为`.720346/.497569/.434961`，而
H_set为`.516868/.231996/.184355`，macro50 reader只保留约`.424x`。Core-fused、compiled和effective-BA进一步为
`.156779/.156534/.143189`。同task不同K4的compiled/BA cosine仍约`.9941/.9962`，说明K-set稳定性并非当前首因。

为判断Procedure趋同是否应先修，使用历史V6-LPCP做同口径只读诊断。V6 raw Procedure更趋同：between-task cosine
`.997294`、correct/reverse relative-L2`.109306`；但Procedure slot reader把有向差异放大到`1.292--1.301`，compiled/
BA仍约`.262/.263`。因此“让raw Procedure彼此更远”不是充分必要修复，v4最早失效是Core-unconditioned endpoint
Query与Procedure Keys共漂后抵消memory Value差异。

后续v5只替换该reader：固定layer/rank address先查询Core得到task-conditioned Query；完整Procedure以真实采样帧
position作Keys；同地址首帧中心化memory dynamic作唯一Values。K-set、Core fusion、M2P、rank16、loss和recipe不变，
且必须fresh。该实验检验V6有效的“task-conditioned query读取微弱有向证据”原则能否在LMMPC四流中成立，不恢复
V6 checkpoint、frozen compiler或旧slot实现，也不增加negative/matching loss。

### 3.12 EMBER-LMMPC Core-Addressed Reader（历史revision v5）

v5唯一把v4的Core-unconditioned endpoint Query替换为Core-conditioned layer/rank address Query，其余bounded K-set、
Core fusion、bounded axial M2P、native rank16 FactorHeads和B20 functional-only recipe完全不变。机制门显示raw
Procedure correct/reverse差异几乎不变时，reader/raw由v4的`.718x`提高到`1.819x`；fresh macro25 strict也从matched
v4的`104`提高到`123`，严格配对为`85 retained / 38 gained / 19 lost`。因此该reader是有真实closed-loop收益的
正机制。

同一world6/topology exact-resume到macro50后，functional loss继续降、checkpoint完整exit0，但K4 strict降至
`84/400`、breadth5，per-task=`0/1/45/1/0/29/8/0`、per-suite=`1/46/29/8`。25→50严格配对为
`71 retained / 13 gained / 52 lost / 264 both-fail`、churn65、net`-39`、Jaccard`.52206`；相对LPCP143为
`74/10/69/247`、net`-59`。Object3从25降到1、Goal6从43降到29，Long1从3升到8，证明训练在tasks间换手。

这不是Writer停止学习：all400 BA norm从`27.225→43.999`，25→50 cosine`.60033`、relative-L2=`1.30602`；
effective targets增加且q dominance下降。也不是同task不同video update分裂：四K4条件的update task-mean/sample
energy为`.9796--.9965`，而不同task mean update cosine仅`.35996`。macro50 raw Procedure虽更趋同
（correct/reverse `.46834`），reader仍放大到H_set=`1.06524`，compiled/BA=`1.16193/.95151`，same-task约`.99`。
所以最早未解接口后移为`offline B20 functional cotangent -> task-specific native factor commitment -> held on-policy
support retention`，不是carrier、Procedure reader、K-set或LoRA未写出。

macro50是显著负点，但不再作为整条recipe终局：历史v5.2/v6均出现先降后升，且该formal config本来预注册
`25/50/75/100`。因此同一world6/topology将exact-resume到macro100，并对macro75/100做同口径strict paired400；
四checkpoint联合判断共同积累、same-row恢复或循环task换手，并拆分Program、FactorHeads、B20 credit与shared
retention责任。当前不改架构、不扫参数、不补六臂且不实现successor。已有负结果仍证明25→50未稳定共同积累；
reader本身保留为正机制，memory token、Dynamic-K、Core/Procedure、rank16和EMBER-LMMPC主思想均未被否定。

### 3.13 Core-Addressed Reader完整macro100轨迹与漂移归因

同一formal run从macro50 exact-resume到macro100，保持world6、gpu01物理`1/2/4/5/6/7`、B20、optimizer、
scheduler、sampler/RNG和架构不变。macro75/100 checkpoint及六rank state完整，0 OOM/nonfinite。四点strict
paired400与breadth为`123/8 → 84/5 → 89/6 → 87/4`；per-task依次为
`3/3/44/25/1/43/3/1`、`0/1/45/1/0/29/8/0`、`3/0/36/1/2/44/3/0`、
`0/4/38/0/0/42/3/0`。macro50→75为`59 retained / 30 gained / 25 lost`、churn55、net`+5`；
macro75→100为`70/17/19`、churn36、net`-2`。四点400 rows中49行始终成功、150行曾成功；25→50丢失52行
只有22行在任一后续点恢复、15行到macro100恢复，新增13行只有6行到macro100仍成功。因此续训观察到了局部回升，
但没有恢复macro25，更没有共同积累。

同一schedule的额外严格配对把absolute缺口与相邻漂移分开。当前best123相对LPCP143为
`100 retained / 23 gained / 43 lost`；相对GOMQ151为`100 / 23 / 51`、churn74、net`-28`。后者suite差为
Spatial`+3`、Object`-12`、Goal`+4`、Long`-23`，最大单task缺口是Long1的`26→3`。作为对照，GOMQ151相对
LPCP143为`126/25/17`。当前与GOMQ获得的新success rows数量接近，主要差异来自当前fresh完整Writer替换了更多旧
support；该配对事实本身不能唯一归因到Program、FactorHeads、functional occupancy或optimizer。

固定K4+B20 train24 panel的mean loss为`.112124/.099353/.098427/.101337`；相邻改善/变差tasks为`19/5`、
`14/10`、`11/13`。25→50 offline loss大幅改善同时held strict净丢39，75→100连固定support也开始忘记。
Program×FactorHeads交叉解码中，相邻compiled Program relative-L2为`.770/.730/.710`；25→50 heads-only旧Program
BA relative-L2=`1.320`、Program-only旧heads=`.582`，后两段分别为`.676/.583`和`.585/.575`。FactorHeads主导
早期norm扩张，但Program始终material旋转，后期两者责任相当；只冻结heads或只约束Program都不能解释整条漂移。

该recipe正式non-pass：best为macro25 `123/400`，无点达到145，最终breadth4，不做六臂controls、不再续训或小扫。
负结果只否定`Core-Addressed Reader + bounded K-set/M2P + native rank16 FactorHeads + static cross-episode offline
B20 credit`能够稳定积累held support；reader、memory、Dynamic-K、rank16和完整EMBER-LMMPC信息流仍可继承。
现有证据把未决接口限定在functional query occupancy、shared support-preserving credit、Program与FactorHeads有限长
更新到held occupancy的组合，但尚不能唯一拆分这些因素。当前没有active design或预选successor。

正式证据：

- train：`runs/outputs/pi05_lmmpc_v5_formal_fresh_r6_b20_aecbce5_gpu01p124567_20260818`；
- macro75 strict：`runs/outputs/pi05_lmmpc_core_addressed_macro0075_k4_correct400_noreplacement_seed7_trainr6_evalr3_f42edfc_gpu02p237_20260818`；
- macro100 strict：`runs/outputs/pi05_lmmpc_core_addressed_macro0100_k4_correct400_noreplacement_seed7_trainr6_evalr5_f42edfc_gpu01p12456_20260818`；
- four-checkpoint rows：`runs/analysis/lmmpc_four_checkpoint_strict_trajectory_20260818.json`；
- B20/Program/FactorHeads联合归因：`runs/analysis/lmmpc_macro25_50_75_100_drift_diagnosis_20260818.json`。

### 3.14 固定提交外部独立复核

外部专家对`codex/bci-continuation@947c0e308c0b16bea97f0a3d157a3fe7b570a074`完成只读复核。实验数值来自tracked
文档重述；专家另外按paired gains/losses复算两侧精确McNemar：当前123相对LPCP143为`p=.018657`、相对
GOMQ151为`p=.001516`，macro25到50的`13 gained / 52 lost`为`p=1.17e-6`，确认这些差异不是普通paired采样波动。

复核发现一个先前未登记的代码级问题：当前layer-matched memory路径在已经detach frozen backbone hidden后，又把
fresh `patch_grounding`、`interaction_projection`和逐帧evidence的输出detach。仓库侧核验确认这会使前两者收不到
functional gradient，且现有动态路径测试没有覆盖它们；旧V6 semantic路径无相同输出detach。该事实尚未经过matched
closed-loop干预，不能单独归因全部123低上限或123到84漂移。

专家把问题暂分为三类：前端过程credit/shortcut、offline expert-state occupancy到rollout support retention错配、
Program到共享FactorHeads坐标的可达性和co-drift；建议先做全模块梯度审计与macro25视频因果面板，再按证据依次
检验仅恢复本地前端gradient、固定状态occupancy反事实、FactorHead冻结、train24 decoder reachability oracle与最后的
gradient aggregation。所有顺序和数值门槛均是外部advisory proposal，不是active design。完整记录见
`docs/external_review_20260818.md`。

### 3.15 外部复核建议的完整执行结果

本轮没有设计新架构，始终基于Core-Addressed Reader主链、rank16、38 targets、Dynamic-K、Action Meta-LoRA、
memory Reader、bounded K-set/M2P和FactorHeads。为了区分owner要求的no-Text边界与专家指出的output detach，使用：

- A：历史Text rank4、保留detach，correct123；
- B：fresh no-Text/VL、保留detach，correct104；
- C：fresh no-Text/VL、只移除Writer-local projected outputs的第二次detach，correct110→101；
- F5：相对C只把24-task arithmetic mean换成fixed-order deterministic PCGrad，correct107→96。

真实functional backward确认A/B的`patch_grounding`与`interaction_projection`在macro1/25均无gradient，C在macro1
首次非零，所有arm的source policy gradient为0。A→B为23 gained / 42 lost；B→C为25/19、净+6且不显著，C25→50
为24/33、churn57。F5 25→50为14/25、churn39；相对C减少lost但显著抑制gained，breadth@1同为6→4。

四个macro25 arm的完整strict video controls为：

| arm | correct | same | wrong | shuffled | keep-first | reversed | no-video |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Text+detach | 123 | 125 | 81 | 122 | 131 | 90 | 48 |
| B noText+detach | 104 | 101 | 65 | 83 | 90 | 96 | 47 |
| C noText+credit | 110 | 111 | 54 | 91 | 93 | 69 | 47 |
| F5 C+PCGrad | 107 | 111 | 51 | 92 | 105 | 53 | 47 |

B的correct-shuffle/keep-first/reverse margin为`21/14/8`；C为`19/17/41`，表明credit恢复主要增强视频内容与全局
正向过程资格，而不是absolute或中间顺序margin。F5为`15/2/54`，显示PCGrad强化全局正反方向但丢失keep-first后的
中间顺序优势。same-task correct-success retention依次为`85.37/83.65/87.27/85.05%`，均未到90%。C的主要视频
收益集中Object且Long reverse反向12；F5的suite分布较均衡但仍未稳定积累。B no-video最终为47，与C/F5的
identity对照一致；B correct-no-video为64 correct-only / 7 control-only，净`+57`、exact McNemar
`p=1.26e-12`。

F2 fixed-occupancy审计不支持“lost rows只因macro50自身occupancy分歧更大”的简单故事；合法action correctness
reference因validation expert不存在且held teacher action受信息墙禁止而不可得。F3冻结heads得到117，对照正常84，
但仍丢33。F4首次评测没有实际安装投影LoRA，旧659/1200无效；修复wiring后fixed-head free-Program为307/1200，
对照direct experts658/1200，只保留46.66%，支持decoder/head reachability瓶颈。故不启动occupancy replacement；
head/rank/decoder应由后继task-held功能gate决定，而非沿用旧“不扩大”结论。F5只证明gradient conflict影响换手和能力
分布，不证明mean或AdamW是主要根因。完整逐项裁决、remote-safe rows与provenance见
`docs/external_review_claim_ledger_20260818.md`、`docs/external_review_followup_20260819.md`和
`docs/evidence/external_review_20260818/`。

### 3.16 第二轮后继路线的train24 fixed-decoder机制profile

2026-08-19在结果无关的train24 ordinal-mod-5 fold0上，以19 tasks拟合共享decoder、冻结decoder后对5个未见tasks只
优化free code。该划分只用于回答decoder是否超越task字典；完整计划轮换五折，最终训练不会永久舍弃这5个tasks。

CPU `BA·probe`预热380/250步得到fit `1.000→0.447`、held `0.805`。同一checkpoint进入冻结π0.5的完整50-token
Action Expert flow面时，独立episode初始为fit `0.999`、held `1.008`，说明低层有效更新相似不能代理policy功能。
随后38个task-equal decoder steps（2/task）和25个held-code steps（5/task）把demo40--49独立评测降到fit `0.833`、
held `0.933`；18/19与4/5 tasks分别优于identity，各有一个task退化。A40峰值18.81 GB，实际优化段约22秒。

这些结果来自dirty worktree的非正式profile，只有一个训练panel与一个独立评测panel，没有closed-loop rollout，故只证明
链路与早期学习信号，不构成realizability pass或架构选择。remote-safe摘要见
`docs/evidence/functional_adaptation_20260819/train24_profile_summary.json`。

### 3.17 Non-held meta-validation frozen source formal

2026-08-19从clean detached `f7ff654`在default fold0的15个LIBERO-90 meta-validation tasks运行source-only formal，
每task固定50 states、seed7，共`646/750`，15/15 tasks非零。14个tasks为40--50/50，task73为4/50并贡献46个失败。
这证明当前source prior在该panel总体很强，同时因15个tasks均参与过71-task source训练，形成了identity/source可能掩盖
decoder效果的欠识别边界。后续预注册为同750行source/direct/projected paired transition；direct若没有跨task增量，
则该pool不能验证decoder泛化。remote-safe摘要为
`docs/evidence/functional_adaptation_20260819/nonheld_meta_source_baseline.json`。

### 3.18 完整71-task non-held frozen source formal

同日从clean detached `3502d13`补齐56个meta-train tasks，每task固定50 states、seed7，得到`2272/2800`
（81.14%）。与3.17的15-task结果合并后为`2918/3550`（82.20%），71/71 tasks非零，52/71达到至少40/50。
Kitchen、Living Room、Study分别为86.27%、89.78%、64.25%；9个低于25/50的tasks中7个来自Study，说明source缺口
集中在book/caddy/shelf pick-place，而非全局控制能力不足。

第一次launcher因NFS-backed SQLite queue出现ownership记录丢失而停止；原地resume复用25个complete shards并收养1个已发布
orphan shard，最终336/336 shards、2800/2800 rows、六worker return code均为0，未改变科研合同或重跑已完成rows。
该恢复事实只属运行证据，不改变科学结论。下一步仍是uniform-step direct experts在相同rows上的paired retained/gained/lost，
并特别检查弱Study tasks是否存在可适配增量。remote-safe摘要见
`docs/evidence/functional_adaptation_20260819/nonheld_meta_source_coverage_71.json`。

### 3.19 Non-held expert ceiling与fixed functional decoder fold0 formal

2026-08-20复用同一71-task step1000 expert bank，在固定50 states/task、seed7上完成source/direct/projected严格配对。
56-task meta-train direct为`2519/2800`，相对source `2272/2800`是2155 retained、364 gained、117 lost、净`+247`；
15-task meta-validation direct为`684/750`，相对source `646/750`是609 retained、75 gained、37 lost、净`+38`。
两侧均有跨task增量，故不触发“当前pool完全欠识别、立即另建role-disjoint tasks”的分支。

共享decoder从56-task fit loss `1.040443→.481957`，冻结decoder后15-task free-code loss
`1.035567→.830093`，14/15 held tasks改善。导出的projected adapters在meta-train为`2451/2800`：相对source
净`+179`，相对direct少68，保留direct successes的92.62%；在meta-validation为`659/750`：相对source612 retained、
47 gained、34 lost、净`+13`、churn81、exact McNemar `p=.18208`，相对direct保留90.79% successes。

因此Gate 2裁决为`qualified_pass_to_writer_inference`，不是fixed decoder方法pass。正证据是train端广泛增益、held
breadth@10覆盖15/15及task73 `4→15/50`；限制是held增量不显著、6正/6负、只复现54.67% direct gain rows、task58
functional loss退化，且只有一个privileged free-code fold。完整remote-safe证据为
`docs/evidence/functional_adaptation_20260819/nonheld_meta_fixed_decoder_fold0_20260820.json`。

projected evaluator实际运行commit `247e6a8`的SQLite `DELETE` journal；当时继承run contract仍写旧`WAL`描述。
这是配置标签滞后而非执行或科学合同变化，rows、pairing、adapter和队列claim均不受影响；active config随后改为
`sqlite_delete_full_sync_atomic_claim`。

### 3.20 Successor Writer formal前机制与资源profile

同日从`main@ef8eb3a`完成4-task、1-macro profile。当前authority明确将Text/VL Meta-LoRA rank设为0，冻结VLM与fixed
decoder无trainable parameters，只训练8,218,216参数的language-prior/video-posterior Writer。profile total loss
`1.828327`、gradient norm `2.490839`，A40峰值32.52 GB、训练段36.82秒。早先两次失败分别定位并修正autocast下
probability BCE与32-frame encoder microbatch OOM；失败产物已清理，没有重复expert/decoder大训练。

该节点只证明正式训练图、信息墙和显存合同成立。其后必须fresh训练56-task Writer，并在formal checkpoint上完成一次
online generation profile后，才可运行15-task language/video/process matched screen。

## 4. 截至整理边界的已解决与未解决接口

### 已有充分正证据

1. source policy具备可适配的LIBERO基础能力；task-local rank16 SFT LoRA确实policy-effective。
2. action-hidden视频内容和顺序能够改变Program、LoRA、action与闭环结果。
3. 每video保序、跨video set聚合、Dynamic-K和one-forward layerwise carrier均可高效实现。
4. 多video可以显著降低same-task nuisance；learned memory query可以产生真实closed-loop增益。
5. V6 native rank16 topology/FactorHeads能达到143，不能把LoRA输出维度本身简单归罪为首因。
6. reward credit、native BF16写出、q/v/action response和endpoint preference都可以在局部接通。

### 尚无方法解决

1. 没有证明模型稳定提取了跨初始化成立的高层task Program，而不是task identity或demo nuisance。
2. 没有一个约145+方法同时具备相邻checkpoint稳定、低churn、same-task-other鲁棒和完整视频因果controls。
3. same-task video-coherent update与cross-task可共存不是一回事；后者仍反复失败。
4. training surrogate/reward下降仍不能选择held on-policy有用方向。
5. Program如何进入native LoRA而既material又不破坏已有support，仍缺统一可扩展解。

## 5. 负结果边界与禁止重复

- rank14只否定uniform pivot14 compression/regeneration合同，不否定所有rank reservation。
- rank8路线低分不证明rank8容量不足；fixed-A audit已证明表示能力与当前可达子空间不同。
- task expert bank失败不否定task-level manifold supervision；它只否定held expert dictionary/routing。
- K4失败不否定few-shot；它证明聚合前的per-video表示和policy credit必须正确。
- SFMC/Gradient-Open/CCT等只否定各自一轮commitment，不否定memory、reward或生成LoRA。
- GOMQ不稳定不否定learned memory；它否定当前tail和shared update的稳定性。
- 不续terminal checkpoints，不用rank/LR/scale/seed/dtype小扫救单点，不用union或controls缺失的峰值选方法。
- 不把高rank、正交、均匀能量、更多atom/lane/expert当性能目标。
- 不恢复language-only LoRA bypass、video/LoRA平均、held expert route、80-row选模或validation-task topology调参。

## 6. 保留的深结构锚点

tracked tree只保留下列少数精确历史设计文档：

- `action_forecast_writer_v4_root_cause.md`：错误absolute-time/action-phase shortcut；
- `action_forecast_writer_v6_design.md`：历史v6-fast143结构；
- `action_forecast_writer_video_expert_manifold_design.md`：task experts/manifold路线；
- `action_forecast_writer_qv_rank_reserved_native_reward_design.md`：rank14 compression与regeneration裁决；
- `action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md`：LPCP143 carrier；
- `action_forecast_writer_v6_lpcp_cfmg_gradient_open_memory_query_design.md`：GOMQ memory正证据与稳定性失败。

其余逐轮design已由本ledger覆盖，不在active tree重复保留。精确恢复方式：

```bash
git show 8553b61:docs/<historical-design>.md
git show 8553b61:task_plan.md
git show 8553b61:findings.md
```

若追查具体run identity、per-episode row、manifest或命令，应读取相应`runs/outputs/*/{run_contract,results,analysis,
completion,evidence}.json`。Git或artifact中旧“下一步”仅是历史，不构成执行授权。
