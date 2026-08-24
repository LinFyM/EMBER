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

随后从clean pushed detached `a2999d5`在gpu02六张A40 fresh训练56-task Writer到预注册macro10，耗时780.43秒。
total objective由`2.082406→1.853265`，combined/language/video code分别由
`.995820/.994485/.984880→.838643/.861964/.974670`；10行metrics、梯度与checkpoint均finite，单卡峰值
32.60 GB。checkpoint保留Writer、optimizer/scheduler、sampler/RNG和world-size6 trainer状态，没有续训或挑选中间点。

macro10随后在一张空闲A40完成32-entry、最长视频优先的B8/B16/B32真实generation profile。三者吞吐分别为
`.898928/.897234/.896828 LoRA/s`，均零OOM、稳定且显存余量至少35.50 GB；按既定“最高实测吞吐且有余量”规则选择B8。
profile读取teacher action/state/reward/terminal均为0，结果已封入active config。该节点仍不构成closed-loop方法证据，
只解除15-task language/video/process matched screen的运行门。

### 3.21 Macro10七臂推断失败与functional-coordinate回退

同一macro10在15个meta-validation tasks×10固定states完成correct、language-only、video-only、first+final、reversed、
shuffled、static-first-repeated七臂matched screen，依次为`131/130/134/130/134/133/132`。correct相对source净0，
相对video-only/reversed均净`-3`；order/static negatives全部不差于correct，故没有视频过程正证据，也没有进入outer RL。

事后定位发现原56个train codebook vectors为whitened、平均norm`5.589`，15个held free codes则从零独立拟合到平均norm
`.505`。macro10 video-only在effective-BA空间几乎等于15-task shared mean，且仅1/15 tasks最近邻到自己的projected
adapter；其`134`是shared carrier与高source能力，不是task inference。原fold0 decoder closed-loop证据保留，但
`qualified_pass_to_writer_inference`不再足以授权继续训练旧Writer。

后继改为专家原始的policy-functional manifold：统一meta-train anchor queries收集71个expert-source flow
fingerprints，PCA/whitening仅拟合56 train tasks，held15只做同变换；固定这些同坐标codes后重新拟合decoder并重跑Gate 2。
完整证据为`docs/evidence/functional_adaptation_20260819/writer_macro10_inference_gate_20260820.json`。

### 3.22 Unified fingerprint坐标通过、flow-only Decoder闭环失败

从clean pushed `b86800e`在8个固定meta-train anchors上统一收集71个expert-source flow fingerprints，feature width
`25600`；PCA/whitening只拟合56个meta-train tasks，32维解释方差`.887516`。train/held coordinate std为
`1.0000/.7248`、平均norm为`5.5701/4.1436`，因此旧held free-code near-origin问题已修复。

固定这些codes训练Decoder后，fit/held flow loss分别由`1.040443→.445721`和`1.035567→.664218`，held没有梯度；但
15-task×50严格paired closed loop仅`644/750`，对照source/direct/旧projected为`646/684/659`。相对source净`-2`，
相对direct净`-40`且exact McNemar `p=3.67e-5`，task73仍为`4/50`。因此未运行56-task full复评，也未训练Writer。

gauge-invariant effective-BA诊断显示新Decoder相对direct的平均relative-L2 `2.8576`、cosine `.0254`、norm ratio
`2.7004`；旧projected也为`2.4616/.0240/2.2718`。低flow loss由巨大近正交off-manifold update实现，定位出训练
surrogate而非坐标为当前最早失效接口。后继只改变Decoder objective为effective-update probes，并保留flow panel和
closed loop作独立裁决；同一commit另行评测shared-zero carrier，避免把共享输出误认成task-conditioned功能。

shared-zero carrier随后在相同750 rows得到`640`；相对source净`-6`，而shared-zero→task fingerprint只净`+4`、
churn56、`p=.68888`。因此`644`几乎全由Decoder共享输出解释，task code没有可靠功能增量。

从clean pushed `c3e5bc1`运行固定8-probe effective objective，1120 steps耗时466秒。fit/held probe loss由
`1→.610951/.911115`，但完整BA检查在train/held仍为relative-L2 `1.1387/1.1292`、cosine `.0642/.0449`；这是固定
probe过拟合，连train expert support都未恢复，故没有启动closed loop。后继改用exact低秩Gram Frobenius loss，其他合同
不变。remote-safe汇总见`docs/evidence/functional_adaptation_20260819/fingerprint_decoder_followup_20260820.json`。

### 3.23 Exact-BA Decoder与仿射上界共同关闭单expert objective变体

从clean pushed `423a9b2`完成1120-step exact低秩Gram训练：train/held exact loss为约`1→.71068/.92128`，完整BA的
relative-L2/cosine/norm ratio为train `.8423/.5365/.5067`、held `.9591/.3032/.4248`。held15×50闭环为`638/750`，
低于source646、direct684、旧projected659、flow-only644和shared-zero640；相对direct净`-46`且`p=1.96e-5`。
held exact loss在step280已为`.92575`，之后到1120仍约`.92128`，故不续训或扫超参。

无训练仿射诊断直接在full effective-BA Hilbert空间用32维fingerprint预测expert update：train为relative-L2/cosine
`.5244/.8439`，held仅`.9797/.3648`，且设计矩阵condition number`1.009`。这排除了“只是当前12M Decoder非线性或
factor gauge不合适”的窄解释。当前flow/probe/exact objective family按formal证据关闭，下一阶段转向专家原始提出的
多成功adapter功能等价类、source/meta角色分离和sealed target ceiling诊断。

### 3.24 Role-disjoint定位、validation8 local oracle与on-policy分支

source未训练过的target train24上，19/5 role-disjoint flow/action fingerprint把held cosine从旧重叠面板的方向性基准
`.3648`提高到`.4310/.4341`；同一优化轨迹的多成功checkpoint只到`.4355/.4394`。有成功expert的三项action-response为
cosine `.5942`、relative-L2 `.8394`，两项无成功expert的fallback仅`.2071/1.0766`。该结果关闭继续把失败expert或
同轨迹checkpoint集合当作成功功能等价类的做法，并触发预注册target ceiling诊断。

validation8八套独立rank16 LoRA统一训练至step2000，step1000不评测；只在clean detached `5fd224a`上执行一次strict400。
结果为`250/400`，frozen source同一400 rows为`48/400`：43 retained、207 gained、5 lost、净`+202`，八task全正、四suite
非零，Spatial/Object/Goal/Long为`73/78/58/41`。因此预注册强门返回
`advance_to_successful_on_policy_manifold_panel`；不触发全局source重训，也不将validation local LoRA用于共享训练或部署。

同一rollout的BDDL trace显示Long task1第一/第二对象ever为31/13，full success12；task2第一阶段50、第二阶段及success29。
这把残余失败定位到多阶段完成与保持，但BDDL只表示无序final-goal合取，不能替代完整procedure标注。后继从既有non-held
formal direct rows固定四task、每task两条成功trajectory，以denoised action、exact JVP和stage在successful/on-policy
occupancy上比较同task与cross-task几何；不重训已完成expert bank。remote-safe证据为
`docs/evidence/functional_adaptation_20260819/validation8_task_local_oracle_step2000_20260821.json`。

### 3.25 Direct successful-occupancy action/JVP task-vector未通过

non-held task23/26/80/86各固定一条source→expert gained与一条retained-success row；clean detached `febdff0`复跑后
8/8 expert trajectories成功，未替换row。随后从每条trajectory的8个progress strata各取最大expert-source executed-prefix
action disagreement点，在clean detached `1e45c66`上重算paired `50x7` denoised action与`50x32` exact JVP。

把八段直接串联时，action的same-task mutual-nearest只在task23/86成立，即`2/4`；task26 gained最近task86，task80
retained也最近task86。JVP只在task80成立，即`1/4`。early-half与full action最近邻完全相同，且全部selected states都在
BDDL full conjunction之前，因此不是final-goal捷径。该结果按预注册规则返回`do_not_advance_response_family`，不训练
Decoder；它只关闭direct concatenation，不关闭phase-aligned action response。

专家原始意见中的两个未完成条件因此成为下一主变量：同task多个独立successful adapters，以及显式phase
correspondence。后继复用target train24既有step250--2000 bank/formal rows，以每task最早/最晚成功checkpoint自己的成功
occupancy形成23个K2与一个K1；fit19学习对齐与坐标，held5只变换和裁决。完整证据为
`docs/evidence/functional_adaptation_20260819/successful_onpolicy_response_panel_20260821.json`。

### 3.26 Multi-checkpoint完整轨迹与显式phase表示通过held5门

train24既有step250--2000 bank形成47个预注册successful members；clean detached `545b43c`复现47/47成功，无换样本。
clean detached `7258487`对每条完整occupancy的全部replan计算paired expert-source `50x7` denoised action。fit19-only、
task/member/state等权的350→32 PCA/whitening解释方差`.923430`，held5仅固定变换。

等时间与功能弧长8点表示在held5均达到`5/5`同task mutual-nearest；弧长在`4/5`任务提高same-task cosine，但task18下降，
fit19 mutual-nearest也由`15/18`降至`14/18`。因此预注册门通过并授权fresh fixed Decoder，科学结论限定为
`multiple successful checkpoints + complete trajectory response + fixed phase-aware coordinates`组合可识别，不声称弧长
单独或普遍优于等时间。证据：
`docs/evidence/functional_adaptation_20260819/train24_successful_equivalence_phase_20260821.json`。

### 3.27 Fresh phase Decoder的held5闭环增益与support-retention失败

从clean pushed `73c2a32`训练的fresh Decoder完成5-rank、950 task visits/190 updates，fit/held identity-relative flow
loss为`.323930/.616152`；held earliest/latest family为`.636755/.595550`，先通过内部functional安全门。held5 codes没有
梯度或自由优化，两套完整LoRA分别来自earliest/latest member固定坐标，最终LoRA不平均。

三组并行strict250给出source=`21`、projected-earliest=`44`、projected-latest=`44`；既有direct earliest/latest为
`74/108`。两套投影相对source均净`+23`，5 tasks不退化、3 tasks严格提高；earliest/latest成功集Jaccard`.466667`。
但direct success retention只有`20/74=.27027`与`28/108=.25926`，direct gain retention只有`.17742/.21875`，均远低于
预注册`.75/.60`。因此联合Gate 2返回`do_not_promote_decoder_to_video_writer`，没有训练新Writer或进入outer RL。

该结果只关闭identity-centered Decoder在successful expert occupancy上做flow distillation后直接推广到held闭环；它没有
关闭多成功phase表示，因为两套投影都产生显著source净增量与跨member稳定。按专家原始遗漏项，下一步在fit19收集同一
Decoder的learner occupancy，以privileged task experts在这些漂移状态上提供聚合监督；held5继续零梯度。shared prior +
task residual也因support loss满足触发条件，但作为下一独立架构变量，不与首轮state aggregation混合。证据：
`docs/evidence/functional_adaptation_20260819/train24_phase_decoder_held5_20260821.json`。

### 3.28 Fit19 learner-state aggregation的有限正增量与Gate 2失败

从clean pushed `966353e`复用首轮Decoder、phase codes、experts和30个既定成功初态，收集30条fit19 projected-policy
trajectories并绑定37个earliest/latest expert members。每个member取8个真实learner states，successful与learner panels
严格1:1；6-rank完成912 task visits/152 updates。learner-state flow loss`.629034→.155116`，held mean
`.616152→.560983`，held仍零梯度。

同一held5固定250 rows上，earliest/latest由旧`44/44`变为`54/47`，相对旧投影净`+10/+3`；相对source21净
`+33/+26`。但direct success retention仅`.31081/.27778`、direct gain retention`.19355/.26042`，latest只有1/5 task
严格提高，成员成功集Jaccard`.40278<.44`。Gate 2继续返回`do_not_promote_decoder`，没有训练Writer或进入outer RL。

该轮只关闭当前staged learner-state functional aggregation，不关闭occupancy或train-task reward。专家挑战十二的触发条件
已经满足；下一独立架构裁决为stable shared prior + task residual，rollout前merge为唯一complete LoRA，并配套shared-only
baseline。证据：
`docs/evidence/functional_adaptation_20260819/train24_phase_decoder_state_aggregation_held5_20260821.json`。

### 3.29 Stable shared prior的正证据与task residual失败

clean pushed `e948fca`把专家挑战十二精确实现为互斥rank块：shared rank12与zero-code-centered task rank4在rank维
拼成一套rank16 LoRA，避免full-rank A/B相加的`BA`交叉项；zero code严格等于shared-only，部署无第二adapter。stage1与
stage2各用6-rank完成912 task visits/152 updates，held functional mean由`.680319`降到`.659049`。

同一held5 fixed250闭环却给出source/shared/composite-earliest/composite-latest=`21/43/37/33`。source→shared为
17 retained、26 gained、4 lost、净`+22`，exact McNemar `p=5.95e-5`；shared→earliest为29/8/14、净`-6`，
shared→latest为29/4/14、净`-10`，后者`p=.03088`。composite只保留direct successes的`.22973/.15741`与direct gains
的`.09677/.07292`。earliest/latest success Jaccard`.62791`通过稳定性下限，但稳定负增量不能构成task residual资格。

因此方向M得到分解裁决：稳定shared behavior base具有真实科学价值，但当前`shared12 + functional phase-code residual4`
implemented-fail，不进入Writer、不续训或小扫。该结果不关闭shared carrier、closed-loop训练的residual、language prior、
video posterior或outer reward。按专家挑战十四，下一主要变量是授权train/meta simulator closed-loop outer credit；held仍
zero-interaction且reward零梯度。证据：
`docs/evidence/functional_adaptation_20260819/train24_shared_prior_residual_held5_20260821.json`。

### 3.30 EMBER-ECP native Stage 0A首版formal与Gate 1负裁决

新active design EMBER-ECP的首个native observer从clean pushed `f6389af`在gpu01 physical `1,2,3,4,5,7`完成10个
world-size6 macros、90 tasks/macro；Prohibited physical0未使用。正式段900 task visits耗时417.55秒，所有metrics、
gradient和macro10 checkpoint finite。total/action-alignment由`.587576/.144328`降到`.447487/.065853`，但cross-task
contrast由`1.721745`恶化到`1.759694`，posterior entropy由`1.464783`降到`.241540`。macro10的90-task presence均值
`.348687`、标准差`.001770`，source71与fit19均值为`.348722/.348560`，已出现与task复杂度无关的全局尺度迹象。

固定macro10后在train24全部fit19+held5、每task两个demo pairs运行48-row observer panel，只比较canonical、保序2x速度、
same-task other video与antithetic fixed probe；没有使用shuffled/reversed/wrong或action/reward。same-task other summary
cosine为`.999985`，但mean/nearest cross-task也为`.996493/.999125`，fit19与held5 nearest margin只有
`.000927/.000604`；antithetic summary/event cosine仍为`.998766/.998409`。所以首版高一致性来自近全局表示坍缩，
而不是可泛化task event结构，Gate 1返回fail。

根因定位为learned global duration denominator与event-mean action target共同允许稀疏/均值捷径。本轮不续训、不扫
LR/rank/seed，不启动compiler或在该observer上做正式Action Meta校准。后继fresh版本只把presence改为固定的
speed-normalized occupancy fraction，并把action grounding改为soft event posterior下的逐帧action reconstruction；slot
对应的视频段落仍完全由模型学习。remote-safe证据为
`docs/evidence/ecp_20260822/stage0_native_macro10_gate1.json`。

### 3.31 Stage 0A v2的单event坍缩与pre-segmentation grounding裁决

clean pushed `395912a`的v2 formal在gpu01 physical `1,2,3,4,5,7`完成10 macros/900 task visits，耗时253.23秒，
峰值11,652,725,248 bytes。固定occupancy-fraction presence移除了v1的learned duration denominator，但active events从
macro1的`6.85`在macro6前降为`1.0`并保持到macro10；逐帧posterior action loss只到`.251337`。clean pushed `3b6df9a`
修复panel可选路径解析后重跑同一48 rows，四个条件全部只有1个active event，same-task-other/nearest-cross-task summary
cosine为`.999981/.998369`，nearest margin`.001611`。Gate 1继续fail，Action Meta与compiler均未启动。

对8个固定tasks×2 views的真实cross-episode action targets做无训练诊断：逐视频最优常数预测MSE为`.178693`，有序8-bin
oracle为`.034727`，降低`80.57%`。所以phase信号存在，v2失败是它只在随机event pooling之后监督frame evidence，posterior
先被其它目标压成单event。下一fresh版本在segmentation前增加training-only frame-action grounding，与event分支共享owner
pooling/action decoder；不规定slot identity或event count，且在direct grounding建立前关闭premature consistency、uncertainty、
sparsity和entropy项。remote-safe证据为
`docs/evidence/ecp_20260822/stage0_native_v2_macro10_gate1.json`。

### 3.32 Stage 0A v3首次通过native非退化门并进入Action Meta matched arm

clean pushed `2d19ea8`的v3 formal在gpu01 physical `1,2,3,4,5,7`完成10 macros/900 visits，耗时252.66秒，峰值
11,664,016,896 bytes；Prohibited physical0未使用。frame/event action loss由`.312545/.312241`降到`.243966/.246427`，
cross-task contrast由`1.721222`降到`1.376669`；active events从`6.85`结束于`6.97`，全程未坍缩。

同commit固定48-row panel中，correct active events均值`6.48`且range 4--8；same-task-other summary/event cosine
`.999601/.999270`，mean/nearest cross-task cosine`.909019/.980528`，48/48 nearest margins为正，held5也为10/10；速度
summary/event cosine`.999975/.999871`。这使native observer首次通过“非全局、跨episode、held可泛化”的结构门，授权执行
独立Action Meta-LoRA arm。antithetic summary/event cosine仍只有`.978224/.976424`，且只有16/48 rows优于nearest-cross，
所以最终observer authority仍待matched Meta panel，不进入compiler。证据：
`docs/evidence/ecp_20260822/stage0_native_v3_macro10_gate1.json`。

### 3.33 Action Meta-LoRA中性matched裁决与Stage 0 authority冻结

clean pushed `a42601a`从冻结native v3 macro10训练shared rank4 Action Meta-LoRA，world-size6完成10 macros/900 visits，耗时
825.15秒、峰值15,432,632,320 bytes。adapter gradient与checkpoint均finite，active events始终`6.83--6.91`。

完全相同48-row panel中，native→Meta的nearest margin为`.01907267→.01908323`，mean margin
`.09058216→.09061162`，antithetic summary cosine`.97822402→.97822047`；48/48 positive margins、held5 10/10与
antithetic-closer 16/48不变，所有逐row差异约在`1e-4`以内。结果是无显著收益也无可复现负面效果的中性校准。按owner
预先确认的采用规则，`native v3 macro10 + Action Meta v3 macro10`被永久冻结为Stage 0 observer authority；Meta只在observer
内部安装，不是rollout第二adapter。Gate 1完成，下一步进入privileged `q_pi + compiler`。证据：
`docs/evidence/ecp_20260822/stage0_action_meta_v3_gate1.json`。

### 3.34 EMBER-ECP Stage 1首版fold0 Gate 2负裁决与输出坍缩定位

clean pushed `6d71cb8`的Stage 1首版使用冻结native v3 + Action Meta authority、47个train24 successful-policy members、
完整rank16 direct states和8-phase successful occupancy，在fit19完成1,140 visits/190 updates；held5 shared gradient为0。
228/570/1140三个checkpoint的held5 strict250依次为`23/27/27`，对照source/shared/direct-earliest/direct-latest为
`21/43/74/108`。1140逐task为Spatial0 `24`、Spatial9 `1`、Object8 `2`、Goal5 `0`、Long6 `0`；570→1140为
`17 retained / 10 gained / 10 lost`。因此继续训练没有稳定积累，Gate 2 fail，`q_V`未启动。

一次冻结checkpoint的Program-to-LoRA诊断定位了首个结构性原因。visible anchor、`q_pi` teacher、`q_pi` correction的跨task
cosine均值分别为`.94661/.94608/.86898`，correction把与direct pair geometry的相关提高到`.49930`，说明privileged输入并非
完全无效；但唯一global residual scale仍为`.10060`。compiler输出随后达到`.99681`跨task cosine，而direct step2000只有
`.13191`；generated own-direct只在`1/24` tasks胜过nearest-other。结合held effective norm ratio/cosine仅`.3694/.3290`，
裁决为`q_pi`幅度不足后，stable-prior A/B template-residual compiler又把差异压成低能共享update。

本轮停止同曲线延长、fold rotation和video encoder。active Stage 1改为presence-bound content-gated `q_pi`与同一compiler内
prior/full absolute output surfaces；canonical factor只作坐标warm-start，闭环仍由functional support、fit reward/progress和
held oracle gate决定。正式训练、materialization、250-row evaluation、paired analysis与Program geometry均完整保留；临时
分析脚本已删除。remote-safe证据：
`docs/evidence/ecp_20260822/stage1_privileged_compiler_fold0_gate2.json`。

### 3.35 Stage 1 absolute compiler v2几何门失败与policy-functional合同纠正

clean pushed `7ca808d`删除了`q_pi`全局scalar cap与stable-prior factor residual；successful members和prior进入compact-SVD
canonical gauge，full Program直接输出absolute完整rank16 LoRA。真实K2+functional profile把family-specific head初始化校准到
expert canonical factor尺度的`.4x`，initial effective loss从`19.5`降到`1.1172`。随后gpu01 physical `1,2,3,4,5,7`
完成fresh 228 visits/38 updates，exit0、六卡峰值约10.25 GB，Prohibited physical0未使用。

该checkpoint的24-task materialization未过昂贵rollout前几何门：candidate跨task cosine `.994192`，own/nearest-other
`.183969/.282906`，own retrieval `2/24`，norm ratio`.099771`；held closed-loop rows为0。该段
`functional_start_task_visits=228`，所以没有successful-policy functional gradient，只检验了absolute BA/canonical warm-start。
代码复核同时发现address embeddings进入values且constant query直达factor heads的内容旁路。active v3因此保留同一
event/layer/family Program与absolute compiler，但令地址只控制key/query/locality、LoRA hidden必须读取Program values，并从
第一个update启用successful-policy functional response。证据：
`docs/evidence/ecp_20260822/stage1_absolute_compiler_fold0_geometry.json`。

### 3.36 Stage 1 content-address v3恢复task差异但未恢复own-policy方向

clean pushed `cba8caf`切断Program/compiler的address-to-output捷径，并从第一个update启用successful-occupancy functional
response。gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates，exit0、checkpoint完整，Prohibited physical0未用。

24-task materialization的candidate跨task cosine为`.939205`、norm ratio`.487844`，相对v2的`.994192/.099771`证明
Program content已经实质控制输出。但own-direct/nearest-other cosine为`.012822/.026238`，own retrieval仍只有`2/24`；
预注册几何门失败，held closed-loop rows为0。前5到后5 updates中functional response由`.995844`降到`.871159`，member
exact-BA却由`1.14167`升到`1.37474`，canonical factor由`1.24101`升到`1.56286`。因此active v4保持同一结构，只把前228
visits设为coordinate bootstrap；通过几何门后才恢复functional并补source/shared support与fit reward/progress。证据：
`docs/evidence/ecp_20260822/stage1_policy_functional_compiler_fold0_geometry.json`。

### 3.37 Stage 1 coordinate-bootstrap v4下降监督loss但仍未建立rank-conditioned policy方向

clean pushed `fc0b84e`在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates；全段为coordinate bootstrap，
functional panels与updates均为0，Prohibited physical0未用。前5到后5 updates的member exact-BA由`1.05391`降到
`1.00938`，canonical factor由`1.14880`降到`1.08904`。

24-task materialization的candidate跨task cosine为`.858906`，但own/nearest-other仅`.018399/.029450`、自身检索
`1/24`、norm ratio `.091336`。几何门失败，held closed-loop rows为0。post-hoc gauge-invariant participation rank/
top1 energy为candidate `1.0733/.9664`、direct `1.2616/.9127`；raw candidate A/B rank-vector cosine为
`.8501/.7779`。这些只用于定位query-conditioned readout不足，不作为模型选择或鼓励rank均匀的指标。

active v5保持Program、`q_pi`、absolute single-LoRA surface和content/address separation，只让target/rank query乘性调制
cross-attended Program content；零Program仍不能独立写LoRA。它先重跑同一228-visits几何门，通过后才恢复完整
successful/source/shared support与fit reward/progress。证据：
`docs/evidence/ecp_20260822/stage1_coordinate_bootstrap_fold0_geometry.json`。

### 3.38 Stage 1 query-content v5有限改善但仍未建立policy support

clean pushed `ae15e47`在gpu01 physical `1,2,3,4,5,7`完成fresh 228 visits/38 updates；全段为coordinate bootstrap，
functional panels/updates均为0，Prohibited physical0未用。member exact-BA前5到后5 update均值由`1.05208`降到
`1.00924`，canonical factor由`1.14952`降到`1.13216`。

24-task materialization中candidate pair cosine为`.876522`；own-direct由v4 `.018399`提高到`.082145`，own retrieval由
`1/24`提高到`3/24`，说明乘性query-content路径生效。但nearest-other仍更高为`.107706`，candidate/direct norm ratio降到
`.086427`，fit19/held5 member loss为`.995066/.991406`，所有几何门失败，held5 rows为0。

因此v5关闭“继续局部修补compiler query即可建立oracle compiler”。active Stage 1保持v5 Program/compiler和零内容反事实，
转向专家要求而尚未覆盖的policy-support teacher：successful occupancy、加权learner occupancy、source/shared support、
multiple-member consistency与fit reward/progress共同约束Program到单套LoRA的行为。rank仍没有技能语义。证据：
`docs/evidence/ecp_20260822/stage1_query_content_bootstrap_fold0_geometry.json`。

### 3.39 Stage 1 policy-support v6恢复输出幅度但未通过task方向门

clean pushed `85477ea`先在gpu01 physical `1--6`并行构建78 MiB固定bank，覆盖24 tasks、188个successful panels与
120个learner panels；五个successful/learner/source/shared response通道均非零。双visit真实profile走通两类panel，随后同一
authority完成fresh 228 visits/38 updates，耗时115.68秒、峰值16,626,005,504 bytes，Prohibited physical0未使用。

moving-panel functional response前5到后5 update均值由`.64456`降到`.50289`，24-task candidate/direct norm ratio由v5
`.08643`提高到`.64465`，表明multi-policy support不再只生成近零update。但member exact-BA由`1.15677`恶化到`1.90182`；
materialization的own/nearest-other cosine仅`.01618/.02816`，自身检索`2/24`。candidate pair cosine为`.85242`，所以失败不是
全局同一输出，而是局部response拟合没有建立本任务完整policy方向。预注册几何门失败，held5 closed-loop rows为0，`q_V`
未启动。

本轮不延长、不做小超参扫。下一步先在冻结single checkpoint上完成全panel support audit；它若证明fit与held都相对source
保留功能，下一major variable加入fit-task task-equal success/progress，否则先修policy-support teacher。证据：
`docs/evidence/ecp_20260822/stage1_policy_support_fold0_tv228_geometry.json`。

### 3.40 v6冻结全panel audit定位到stable-shared support丢失

clean pushed `a4928ce`用gpu01 physical `1--6`并行遍历v6冻结checkpoint的全部308个support panels。fit19 candidate相对
source为`.80282x`且`19/19` tasks更好，held5为`.90167x`且`5/5`更好；这证明v6的multi-policy evidence存在跨held的真实
功能增量。但candidate相对stable shared在fit/held为`1.39966x/1.27745x`，胜出任务数均为0，八项预注册support条件只通过
四项source条件。

因此不加入simulator reward、不运行held闭环。active v7保留同一Program、q_pi、support bank与single rank16输出，只把
independent absolute full surface改为prior-preserving low-rank union：generated residual与shared effective update经thin QR和
`32x32` core SVD重压回rank16，避免raw A/B addition交叉项及固定rank分槽。证据仍为
`docs/evidence/ecp_20260822/stage1_policy_support_fold0_tv228_geometry.json`。

### 3.41 v7 prior union接近shared但未过门，参数坐标辅助项被证实主导训练

clean pushed `6987933`完成v7 228 visits/38 updates，clean pushed `55b9065`完成24-task物化与全部308个冻结support panels
审计。fit19 candidate/source/shared为`.41498/.70633/.40514`，candidate相对source/shared为`.58751x/1.02429x`，
19/19 tasks胜过source、9/19胜过shared；held5为`.68674/.88454/.62434`，ratio为`.77638x/1.09995x`，5/5胜过source、
1/5胜过shared。相较v6的shared ratio `1.39966/1.27745`和`0/19、0/5` breadth，prior union有实质正作用，但仍未通过
预注册support gate，held闭环rows保持0。

目标分解发现四项direct BA/canonical辅助项在前5步贡献`1.06492/1.65750`总loss，末5步仍为`.45761/.86428`；它们因
shared-to-direct坐标距离放大而成为多数梯度，与专家“目标不是raw A/B重建”的Stage 1合同冲突。v7不续训、不加reward。
下一fresh v8只移除这些参数坐标梯度，继续报告相同诊断并复跑同一冻结support gate。证据：
`docs/evidence/ecp_20260822/stage1_prior_union_fold0_tv228_support.json`。

### 3.42 v8 functional-only union在task-balanced复验后最终失败

clean pushed `ae4805e`完成v8 228 visits/38 updates，clean pushed `1659bb6`完成24-task物化与308-panel冻结审计。四项
direct BA/canonical梯度均为0，但fit/held candidate-to-shared为`1.15980/1.14903`，胜过shared的task数均为0；相对source
仍为`.66524/.81102`且24/24 tasks更好。materialization的candidate/direct norm ratio达到`6.54391`，candidate pair cosine
`.97804`，own-direct `.01411`，所以该checkpoint未获held闭环、reward或`q_V`授权。

随后从formal metrics核对schedule prefix发现，每个fit task在228节点实际只访问`5--18`次，而task-equal预期为12次；完整
456-visits虽最终平衡，当前全局cost-sort与group shuffle没有保证决策prefix平衡。该工程/科学合同偏差先于下一架构裁决：保留
v8全部方法变量，只把schedule改为每6轮一个task-balanced/cost-balanced block并fresh复验一次。

clean pushed `0b63da1`的balanced复验使每个fit task在228节点恰好12 visits，但candidate/direct norm ratio进一步升到
`8.75029`，candidate pair cosine`.99433`，own-direct `.01106`且自身检索`1/24`。clean pushed `c1f485d`的308-panel冻结审计
给出fit/held candidate-to-shared `1.17823/1.11729`，breadth仅`2/19、2/5`；相对source虽为`.67581/.78862`，但shared四项门
全部失败。故schedule偏差已修复且排除，v8最终关闭；后继只替换Frobenius top-SVD为bounded exact-prior的policy-functional
rank selector，不延长曲线或扫描loss权重。证据：
`docs/evidence/ecp_20260822/stage1_functional_union_fold0_tv228_support.json`。

### 3.43 v9 bounded rank selector近乎保住shared但仍是global correction

clean pushed `dc5dff6`完成v9 fresh balanced 228 visits/38 updates与24-task物化，clean pushed `bb3bc59`完成308-panel冻结审计。
v9把candidate/direct norm ratio从v8的`8.75029`压到`1.94717`，fit candidate-to-shared达到`.98369`，held达到`1.00692`；
相对source为`.56423/.71072`且24/24 tasks更好。这是Stage 1迄今最强的shared-preserving修正。

它仍未通过方法门。fit/held breadth只有`10/19、2/5`，candidate pair cosine`.99779`、own retrieval `1/24`；去掉shared后
correction pair cosine仍`.97482`，selector fraction仅`.08031--.09164`。因此v9关闭且held闭环rows保持0。后继保留bounded
exact-prior selector，但replacement只允许language/scene-conditioned query读取present process Values，消除静态Value或presence
开关写近全局修正的旁路。证据：
`docs/evidence/ecp_20260822/stage1_functional_rank_selector_fold0_tv228_support.json`。

### 3.44 v10 process-only Value连续改善support但仍未建立task mapping

clean pushed `13dfc25`完成v10 fresh balanced 228 visits/38 updates和24-task物化，clean pushed `85dc2fc`完成
308-panel冻结审计。相对v9，process-only Value使candidate-minus-shared cosine从`.97482`降至`.95088`，fit
candidate/shared从`.98369x`改善到`.96892x`，breadth从`10/19`改善到`12/19`。这确认language/scene
静态Value或presence开关是一条真实的近全局旁路。

但held candidate/shared仍为`1.00285x`且breadth只有`2/5`；candidate pair cosine`.99641`、own retrieval `1/24`。
Program correction已有`.82475` pair cosine，编译后correction却回到`.95088`，故最早未解接口收紧为
task-relative Program变化在compiler/绝对functional objective中的坍缩。v10最终关闭，held闭环、reward与`q_V`仍为0。
证据：`docs/evidence/ecp_20260822/stage1_process_value_selector_fold0_tv228_support.json`。

### 3.45 OCPB v11首轮outcome calibration与shared-rank credit修正

clean pushed `86ed95b`从v10冻结checkpoint启动OCPB。program-binding macro1的配对训练臂为`9/8` successes；随后24-task
物化和308-panel audit使fit/held candidate-to-shared从`.96892/1.00285`小幅改善到`.96786/1.00171`，但breadth仍为
`12/19、2/5`，own retrieval仍`1/24`。因此没有运行held5 closed loop或`q_V`。

exact-resume compiler-binding macro2为`10/8` successes，但support退到`.97049/1.00366`，裁剪前梯度由macro1的
`1.0904`升到`11.8792`。反查发现同一个owner perturbation加到全部16个rank angles，而surrogate取rank sum，导致coordinate
step与antithetic归一化相差16倍；macro2不能作为科学负证据。clean pushed `d06842c`注册OCPB v12：coordinate改为rank mean，
从macro1恢复完整optimizer/RNG/topology并复用原macro2 paired seeds，只重做一次compiler-binding，旧macro3/4取消。证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v11_rank_credit_diagnosis.json`。
clean pushed `7d77eb8`的真实single-task corrected profile随后得到rank-mean surrogate`-.000381`、finite梯度`2.08455`和
`16.43GB`峰值，确认修正后的运行面可进入一次formal复验；profile产物在记录后删除。

### 3.46 OCPB v12 corrected compiler-binding有效但未过geometry/support门

clean pushed `16f9e55`从v11 macro1完整恢复model、optimizer、scheduler、六份rank RNG和world6 topology，并以原macro2的
全部paired authority完成唯一corrected compiler-binding macro。两臂仍为`10/8` successes、progress
`.31140/.27193`；rank-mean surrogate为`-.00896`、裁剪前梯度`1.63196`，相对错误macro2的`-.14340/11.8792`确认16倍
shared-rank lift已经消除。一个成功rollout的终止步由110变为109属于允许的低位执行差异，success pattern与replacement
fractions未变。

24-task物化给出candidate pair cosine`.99586`、own/nearest-other `.03972/.06186`、own retrieval `1/24`，而Program
correction pair cosine为`.82561`。clean authority上的308-panel audit给出fit/held candidate-to-shared
`.96934/1.00303`、breadth `12/19、2/5`；相对v11 macro1的`.96786/1.00171`与155个shared panel wins均未改善，v12只有149个
wins。因此v12有效但Gate 2早门失败，held5 closed loop、fold rotation、meta expansion与`q_V`均未启动。

最早接口仍是task-relative Program到policy direction。现行functional anchor把candidate到source/shared的响应距离直接当作
support loss，会把有益task-specific移动与有害漂移一起拉回；compiler outcome coordinate又只监督selector angles，不直接
识别replacement factor方向。后继保持同一结构、checkpoint与paired实验，只把support preservation改成baseline-relative
functional barrier，再做一个bounded相邻节点。正式证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v12_corrected_compiler_gate.json`。

### 3.47 OCPB v13 support barrier达到breadth门但未改变compiler坍缩

clean pushed `b371463`将source/shared preservation从无条件response proximity改为相对各自own-expert baseline不退化的
hinge barrier，其余Program、compiler、rank、初始化、paired seeds与outcome coordinate不变。真实profile确认candidate优于
baseline时两项barrier为0且梯度finite；正式macro保持`10/8` paired successes与`-.00896` outcome surrogate，mean functional
total由v12的`.25295`降到`.15441`。

24-task物化仍给出candidate pair cosine `.99595`、own/nearest-other `.03980/.06201`和own retrieval `1/24`，Program
correction则为`.82546`。308-panel冻结audit的fit/held candidate-to-shared改善到`.96741/1.00168`，breadth首次达到
`13/19、3/5`，8项support条件通过7项；唯一失败是held aggregate比shared高`.1675%`。因此barrier是保留的正修正，v13整体
仍未过联合geometry/support门，同曲线关闭，不运行held5、fold rotation、meta expansion或`q_V`。最早缺口变为
policy-native task-relative compiler factor direction identification。证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v13_support_barrier_gate.json`。

### 3.48 OCPB v14接通owner-resolved response，但one-update checkpoint不通过门

clean pushed `6e927fc`在24-task frozen support bank中新增同状态source与successful-member的full-layer owner response，
每个policy张量为`[2,38,4,128]`，task payload共152,948,592 bytes。candidate LoRA在同一次可微PI0.5 forward中返回最终
flow与owner response；loss按member reliability、signal和disagreement监督38个owner对应的policy effect，不读取raw A/B。
真实profile的owner loss为`.82564`、active owners `92.11%`、gradient norm `15.1782`、峰值16.43 GB，确认运行图有效。

v14 formal在gpu01 physical `1--6`完成一个task-equal outcome macro和完整checkpoint，prohibited physical0未运行CUDA
process。19个fit task的mean owner loss为`.77170`，owner active fraction为`.7895--1.0`；两臂success为`9/11`，mean
progress `.28947/.35088`。24-task物化的candidate pair cosine为`.99575`、own/nearest-other `.03971/.06190`、retrieval
`1/24`，几何几乎等同v13。308-panel audit的fit/held candidate-to-shared为`.96795/1.00313`、breadth `13/19、2/5`，
shared panel wins为152；相对v13的`.96741/1.00168`、`13/19、3/5`和155 wins没有改善。v14 checkpoint因此不promote，
held5、fold rotation与`q_V`均未启动。

复核训练拓扑发现，该macro先对19 tasks逐项backward、再只执行一次optimizer step；历史Stage 1的228-visits信息节点则包含
38 updates。v14因此只裁决了“一个owner-response update是否足够”，没有合理优化专家要求的multi-state policy-response
distillation。后继从v13 model weights和fresh optimizer建立无simulator的task-balanced response stage，首个节点为
114 visits/19 updates；先以24-task geometry决定是否继续至228 visits、full support与一个matched outcome macro。证据：
`docs/evidence/ecp_20260822/stage1_ocpb_v14_owner_response_gate.json`。

### 3.49 v15完整优化owner response后仍未建立同task policy方向

clean pushed `b2faaeb`把owner-response distillation从昂贵outcome macro拆回唯一Stage 1 trainer。从v13只加载model weights、
fresh AdamW在gpu01 physical `1--6`完成228 visits/38 updates，每个fit task恰好12 visits；prohibited physical0无CUDA
process。真实profile一次更新2.24秒、峰值16.40 GB，Writer/FactorHead梯度分别`5.1564/2.2342`。114节点pair cosine由
v13`.99595`降到`.99229`，按预登记方向门exact-resume；228节点继续降至`.98727`，norm ratio从`1.97`收至`1.71`。

完整优化没有恢复task定位。228节点own/nearest-direct为`.03980/.06075`、retrieval仍`1/24`；308-panel fit/held
candidate-to-shared为`.97253/1.02171x`、breadth`13/19、1/5`，shared panel wins为146。相对v13的
`.96741/1.00168x`、`13/19、3/5`和155 wins均未改善。learner panels虽达到`.95158x`，successful panels却退到
`.98404x`，说明当前projected full-layer hidden target能够推动输出分散，却不能识别对应successful LoRA target的局部功能。
联合geometry/support门失败，故不接matched outcome macro、不运行held5、fold rotation或`q_V`，也不续同一训练曲线。

代码级边界是：标记为owner `j`的累计hidden/residual仍可被所有上游LoRA targets共同改变，owner loss并非head-local。
后继只替换监督对象为同一detached source target input上的gauge-invariant activation effect `B_j(A_j x_j^{ref})`，保留owner/horizon、
multi-state occupancy、member disagreement和v13 barrier，不恢复raw A/B target。证据：
`docs/evidence/ecp_20260822/stage1_owner_response_bootstrap_v15_gate.json`。

### 3.50 OCPB v16建立局部task信号但破坏完整policy support

clean pushed `cd96b42`先在gpu01 physical `1--6`并行生成v3 support bank，覆盖24 tasks、188 successful与120 learner
panels，并为每个panel保存38个detached source target inputs和successful-member `B(Ax_ref)` effects；task payload共
688,985,254 bytes，validation/Test action或reward读取均为0。真实task1 profile的local loss为`2.83512`、active owner
fraction`.97368`、FactorHead梯度`4.13349`、2.32秒、峰值16.40 GB。

同一commit从v13 model weights、fresh AdamW完成228 visits/38 updates，每个fit task恰好12 visits。local loss由前后两个
114-visit block的`1.69620→1.31481`；目标对首个successful panel的own MSE由v13`.48016`降到`.25057`，cosine retrieval
由`1/24`升至`11/24`。24-task candidate pair cosine也降至`.94932`，说明局部功能标签真实打破了部分跨task坍缩。

然而candidate shared-subtracted correction pair cosine仍为`.97657`，而expert为`.82316`；candidate只走到expert correction
约`.49176x`幅度，own correction cosine`.87419`仍低于nearest-other`.87516`。raw effective-BA own/nearest为
`.03873/.05976`、retrieval`1/24`。完整308-panel audit中fit/held candidate-to-shared为`1.08581/1.08815x`、breadth
`2/19、0/5`，shared panel wins仅85，均显著差于v13与v15。v16因此不promote、不续相同local curve，也不运行outcome、held5、
fold rotation或`q_V`。

该结果否定的是“匹配冻结source输入上的孤立target effect足以定义完整compiler策略”，不是target-local功能标签本身。它能提供
head-local task方向，却不能约束38个target改变后相互影响的下游输入与最终action。后继保留v16 model weights而fresh optimizer，
以successful/verified-success跨episode train actions上的exact PI0.5 flow-matching loss作为完整组合policy梯度；failed learner
action不作oracle，local effect与v13 barrier只作锚。证据：
`docs/evidence/ecp_20260822/stage1_owner_local_activation_v16_gate.json`。

### 3.51 OCPB v17 exact action grounding形成task差异但继续破坏successful support

clean pushed `72b4a1c`复用v16 task-visit228 model weights、fresh AdamW，在gpu01 physical `1--6`完成114 visits/19 updates；
每个fit task恰好6 visits。57个successful panel全部产生exact PI0.5 action leaf gradient，18个verified-success learner也启用，
39个failed learner严格为零；最大显存16,659,303,424 bytes，更新段75.64秒，prohibited physical0未使用。

24-task materialization显示candidate pair cosine降到`.90236`，相对v16`.94932`更task-specific；candidate/direct effective norm
ratio也从不足恢复到`1.14481`。但mean own-direct`.03855`仍低于nearest-other`.05779`，自身检索只有`1/24`。同一308-panel
audit更直接否定policy恢复：fit/held candidate-to-shared为`1.13962/1.12203x`、breadth`2/19、0/5`，虽然相对source达到
`19/19、5/5`，仍显著弱于稳定shared底座。v17因此不续至228、不运行held5、fold rotation或`q_V`。

这说明absolute action imitation足以制造material的task-conditioned参数变化，却没有给shared Writer提供“哪些完整factor方向在
闭环中应保留”的识别信号。专家Stage 1列出的task-equal closed-loop success/progress仍未在当前factor方向上被真正实现：旧
v11--v13 outcome只扰动q_pi gate或38个selector angles，不能裁决replacement A/B方向。后继以exact action leaf gradient作为
每owner/family局部proposal，并用fit19 paired success/progress给这些proposal structured outcome credit；不是继续v17曲线或
扫描小权重。证据：`docs/evidence/ecp_20260822/stage1_action_grounded_v17_gate.json`。

### 3.52 OCPB v18真实closed-loop factor credit仍未识别own successful Program

clean pushed `026e756`以v17 macro114 model weights和fresh optimizer运行OCPB v18。single-task paired profile先确认
38/38 owner action-gradient proposals、strict common-random simulator、非零success-efficiency advantage与LoRA-leaf到
FactorHead反传。旧outcome的`.01` multiplier会把标准`2 sigma`归一后的leaf再次任意缩小；matched profile改为自然`1.0`
尺度后，outcome leaf由`.002038`变为`.20380`，FactorHead/total gradient仍为finite的`1.367/4.785`，显存不变。

正式4个macros每轮均覆盖fit19各一次并只做一个等权optimizer update；非零credit breadth为`9/8/10/9` tasks，证明训练
outcome不是dead signal。macro2的candidate pair cosine由v17`.90236`降到`.89302`，fit frozen-support ratio也由
`1.13962`暂时改善到`1.13120`，所以按预注册合同exact-resume到macro4。最终pair cosine继续小降到`.89140`，但
own/nearest-direct为`.03836/.05799`、retrieval仍`1/24`。308-panel macro4 audit的fit/held candidate-to-shared为
`1.14317/1.13244x`、breadth`2/19、0/5`，既反转macro2的fit改善，也比v17更差。

Program correction pair cosine从macro2`.84710`到macro4`.84690`几乎不变，而compiled adapter继续分散；这与实现中的
坐标错位一致：paired plus/minus先在compiler输出之外的完整A/B factor states上构造，scalar reward再经leaf投回同时可动的
Program和compiler。它能重定义decoder输出，却没有证明reward识别了compiler可达的own Program方向。v18因此在macro4关闭，
不运行held5、fold rotation或`q_V`。后继回到v13 support最强坐标并冻结compiler，只在event × layer-group × family Program
切空间做逐family action-guided paired credit；先验证fixed compiler reachability。证据：
`docs/evidence/ecp_20260822/stage1_action_guided_v18_gate.json`。

### 3.53 OCPB v19证明fixed compiler局部可达，但shared Program credit没有形成own-policy mapping

clean pushed `75ee051`从v13 macro1只加载model weights并创建fresh optimizer；visible Program与compiler永久冻结，optimizer
只拥有privileged `policy_teacher`。single-task q-family profile中，`.05` relative Program perturbation经同一个冻结compiler
产生`.110166` compiled-LoRA relative delta，paired两臂各`1/2` success并形成非零credit；teacher gradient为`.56058`，
compiler/visible梯度严格为0，所以fixed compiler的Program切向不是dead path。

正式macro1/2分别只扰动q/v family block并各覆盖fit19一次。非零credit breadth为`8/9` tasks，plus/minus successes为
`10/8、11/10`，mean compiled perturbation为`.10901/.05426`；训练信号真实。两步后policy teacher参数相对v13移动
`.0005276`，其余两模块精确不变。但v19自身24-task materialization仍坍缩：pair cosine`.99594891`，own/nearest-direct
`.03980/.06198`，retrieval`1/24`。历史v13 reference为`.99595249、.03980/.06201、1/24`，但其video visit12099与v19的
18199使24/24 demo pairs不同；它只说明v19仍在同一坍缩区间，不构成matched training delta。

同一308-panel audit也没有给出续训依据。fit candidate/shared为`.96830x`、breadth`12/19`，held为`1.00335x`、breadth
`3/5`；历史v13 reference为`.96741x、13/19`与`1.00168x、3/5`，同样不是matched video对照。v19以自身绝对门失败为依据
停在macro2，不运行action-in/out、held5、
fold或`q_V`。该结果否定的是“在未识别的v13 decoder image中，只靠少数shared q_pi reward updates即可找到own Programs”，
不否定structured Program perturbation或fixed compiler原则。

下一互补锁保持v13 privileged Program完全固定，只训练compiler，并用dense exact action与multi-state support识别稳定映射；
若仍失败，再以task-local free Program/fixed-compiler oracle裁决当前bounded compiler image是否可达。证据：
`docs/evidence/ecp_20260822/stage1_fixed_compiler_program_v19_gate.json`。

### 3.54 OCPB v20识别了task-diverse compiler输出，但没有识别own successful policy

clean pushed detached `d42a026`从v13 macro1只加载model weights并创建fresh optimizer，永久冻结visible Program与
privileged `q_pi`，只训练compiler。single-task profile中exact action LoRA-leaf、FactorHead与compiler梯度分别为
`.12943/.43119/7.62951`，其余两模块梯度为0。formal节点覆盖19 fit tasks每个6 visits，57个successful、18个
verified-success learner panels产生exact action gradient，39个failed learner panels不产生action gradient。compiler相对v13移动
`.00146889`，Program与`q_pi`参数精确不变。

matched 24-task materialization使用与v13完全相同的visit12099和24/24 demo pairs。candidate pair cosine由
`.99595249`降至`.97120404`，证明compiler-only训练不是dead update，且确实扩大了task间差异。但own-direct
cosine为`.03945`，仍低于nearest-other `.06104`，retrieval仍`1/24`，所以差异没有指向own successful policy。

matched 308-panel audit是更强反证：fit candidate/shared由v13`.96741x`、breadth`13/19`退到`1.00874x`、
`8/19`；held由`1.00168x`、`3/5`退到`1.02932x`、`2/5`。因此v20在114 visits关闭，不续训、不接
outcome、held rollout或`q_V`。它否定的是“固定task-diverse Program后，用当前dense action/response/support目标只训练
bounded compiler即可识别own mapping”；尚未区分compiler image不可达和shared `q_pi` Program坐标错位。下一步因此
冻结v20 compiler，仅优化fit19 task-local privileged free Programs做reachability oracle。证据：
`docs/evidence/ecp_20260822/stage1_program_locked_compiler_v20_gate.json`。

### 3.55 OCPB v21证明当前bounded compiler image在可写Program区域内不可达

clean pushed detached `ecab6ec`从v20 macro114只加载model weights并创建fresh optimizer，冻结compiler、visible Program、
privileged `q_pi`、source与observer。fit19每task各有一行privileged free Program：固定visit12099的两条correct videos经
冻结`q_pi`只初始化一次，之后固定language/scene/presence，只训练bounded process correction与positive uncertainty scale；
held5没有free parameters。228 visits/38 updates使每个fit task精确出现12次，114个successful和42个verified-success
learner panels产生exact action gradient，72个failed learner panels保持action gradient为0。全部157个冻结base state keys
精确不变，mean update `1.185s`，峰值显存`16,772,373,504` bytes。

free Programs本身已经形成显著task差异：process相对anchor correction为`.176--.531`、mean`.40179`，correction pair
cosine`.36624`。但固定compiler后的candidate pair cosine仍`.96595`，own/nearest-direct为`.03936/.06070`，retrieval
只有`1/24`。同一308-panel support audit中，fit candidate/shared为`1.02708x`、breadth`7/19`，比v20的
`1.00874x、8/19`更差；held按合同不变，为`1.02932x、2/5`。因此v21不续训、不接held rollout或`q_V`。

该结果排除了“shared `q_pi`坐标是唯一障碍”：当前QR-normalized、固定factor能量、角度有界的rank-mode retraction在本次
可写Program区域内没有恢复own successful policy。它没有检验完专家建议的direct absolute family-specific A/B compiler。
历史v6 macro228正是这一完整absolute surface，故下一互补锁复用该昂贵checkpoint做同一free-Program reachability，而不是
重训compiler或做小超参扫描。证据：
`docs/evidence/ecp_20260822/stage1_fixed_compiler_free_program_v21_gate.json`。

### 3.56 OCPB v22打开direct absolute几何，但暴露prior/full非同一surface

clean pushed detached `7a1f233`复用v6 macro228的direct-absolute compiler与`q_pi`，只优化fit19 task-local
process/uncertainty；没有重复训练昂贵compiler。228 visits/38 updates中每个fit task精确出现12次，114个successful和42个
verified-success learner panels产生action gradient，72个failed learner panels保持action gradient为0；153个冻结base keys
精确不变。mean update为`1.098s`，峰值显存`16,759,229,952` bytes，gpu01 prohibited physical0未使用。

direct heads让输出真正摆脱bounded collapse：candidate pair cosine为`.70280`、own retrieval`10/24`、mean effective norm
ratio`.46548`，Program correction pair cosine`.44268`。但own/nearest-direct仍只有`.01663/.01846`，且308-panel fit/held
candidate-to-shared恶化到`1.45056/1.27628x`、breadth`0/19、0/5`；因此不运行held5 closed loop或`q_V`。

冻结checkpoint只读消融显示process attention mass均值`.53913`，process-only Values有`9/19` own retrieval，证明最早问题不再是
Program可写性或process被忽略。代码合同复核发现prior-only通过hard process gate直接返回exact shared template，而full
Program才经过direct A/B heads；这不是专家定义的同一个`D(P_lang,P_scene,0/full)` surface。下一active v23移除该旁路，冻结
visible Program与`q_pi`，只训练一个同时承担prior shared support和full own-policy mapping的single-surface absolute compiler。
证据：`docs/evidence/ecp_20260822/stage1_direct_absolute_free_program_v22_gate.json`。

### 3.57 OCPB v23证明同一direct surface仍会丢失layer与static/process policy坐标

clean pushed detached `ef7100b`冻结visible Program与privileged `q_pi`，fit19每task在visit12099捕获一次Program后只训练
compiler。114 visits/19 updates包含57个successful与18个verified-success learner action panels；39个failed learner panels
不产生action gradient。compiler/FactorHead梯度mean为`4.48596/4.46896`，两个冻结模块梯度始终为0，mean update
`1.252s`，峰值`22,990,815,744` bytes。

同一checkpoint一次materialization同时保存24套full/prior LoRA。full candidate pair cosine为`.83977`，own/nearest-direct
`.03196/.05191`，retrieval`2/24`。full 308-panel fit/held相对shared为`1.13175/1.11069x`、breadth`4/19、0/5`；prior为
`1.27744/1.21881x`、`0/19、0/5`。故两臂都失败，不进入held5 closed loop或`q_V`。

冻结hidden的只读SVD/ridge定位显示stable prior在width256中几乎完全可表示；family-shared q/v heads的prior/full residual约
`.53--.56/.96`，target-local readout降至约`.0003/.21--.25`。但target-local同一个线性map联合拟合prior/full仍有
`.16--.20/.74--.76` residual；同时process delta norm约等于完整full adapter。下一active v24因此保留一个complete-rank16
compiler，但把static/process分开local read、在head前连续非线性融合，并使用38个target-local A/B heads；不扩大width、不扫
小超参，也不把static与process变成两套adapter。证据：
`docs/evidence/ecp_20260823/stage1_single_surface_absolute_compiler_v23_gate.json`。

### 3.58 OCPB v24恢复layer-resolved surface仍未识别process-conditioned own policy

clean pushed detached `631aab7`从v23迁移145个shape-compatible tensors，并把8个family-head tensors展开为76个target-local
A/B heads；新static/process read与zero-start continuous fusion fresh初始化。fit19-only ridge calibration使用19个冻结prior
Programs，把stable-prior factor residual从`1.21412`降到`.02583`，不读取held/validation/Test action或reward，也不保存task
table。

gpu02 physical `0,1,2,3,4,7`的world-size6 formal完成114 visits/19 updates；compiler/FactorHead/process-fusion gradient
mean为`9.33603/4.75648/2.06507`，Action LoRA leaf gradient mean `.07975`，visible Program与`q_pi` gradient始终为0。
mean update `1.304s`，峰值`23,285,629,952` bytes，gpu01 prohibited physical0未使用。

一次materialization同时保存24套full/prior LoRA。full candidate pair cosine为`.97092`，own/nearest-direct
`.04779/.06674`，retrieval `1/24`，norm ratio`1.32925`；因此不是输出近零，而是再次收缩为非own的近共享方向。一次六worker
dual audit在同一policy/panel加载中评测两臂：full fit/held相对shared为`1.19865/1.05384x`、breadth`1/19、2/5`；prior为
`1.18717/1.04658x`、breadth相同。response越低越好，full相对prior在fit/held高`.00465/.00453`，22/24 tasks更差。

v24由此关闭，不进入held5 closed loop或`q_V`。它排除了“target-local heads + simple continuous fusion即可把v23固定hidden
capacity转化为可训练compiler”的解释；最早接口前移到19个独立mapping和当前action/response/support objectives无法识别
process-conditioned own-policy direction。v24之后不自动建立v25，而先完成Stage 1 v1--v24复盘；正式证据为
`docs/evidence/ecp_20260823/stage1_layer_resolved_single_surface_compiler_v24_gate.json`，复盘见
`docs/ecp_stage1_iteration_retrospective_20260823.md`。

### 3.59 MDCO authority与structured calibration图完成，等待首次source-unseen闭环裁决

v24复盘后的唯一MDCO合同复用了71个audited non-held experts，没有重训同类资产。142条fixed occupancy中141条保持成功；
task32/state22未复现并被逐条剔除。source与target response最终组装为95-task、118-member、178-fit-sequence authority，包含76个
LoRA tensors和`118 x 8 x 32` phase responses，explained variance为`.88402`；独立95-task policy support bank同步完成。fit集合
固定为71 non-held加target fit19，held5仍不产生梯度。

canonical Stage 1新增一次且仅一次的fit90 structured calibration：在540 dense visits后，每task由exact successful-action LoRA
leaf gradient构造38个owner-local A/B方向，一组Rademacher owner signs形成完整plus/minus LoRA，两条common-random lanes按
success、BDDL peak progress和成功效率给出antithetic credit；该leaf semi-gradient与原action/support/prior/activation锚共同回传
event-owner `q_pi`及layer/target-local compiler，90 tasks各除以90并跨rank求和。held5、validation8与Test8 reward读取均为0，
materializer要求formal calibration artifact后才接受540 checkpoint。

gpu02 physical0上的单任务真实profile完成38/38 active owners，plus/minus各`2/2` success，效率差产生mean advantage
`.001375`与outcome leaf norm`.06683`；`q_pi/compiler`裁剪前梯度为`.14077/320.67142`，visible Program梯度为0。校准段
`26.97s`，峰值`23,325,685,760` bytes。该profile只证明真实simulator、配对、LoRA proposal与联合反传图接通，不作为held性能
证据；下一科学节点仍是clean pushed detached commit的540 formal及held5 strict paired250。

### 3.60 MDCO 540首次source-unseen闭环失败并关闭当前Stage 1 compiler family

clean pushed detached `419fa84`在gpu02 physical`0,1,2,3,7`完成90 tasks各6 visits、540 dense visits/108 updates。全部task
精确等权，compiler与privileged `q_pi`梯度持续非零，visible Program梯度始终为0，峰值显存`24,341,012,992` bytes。随后
唯一一次fit90 structured calibration由5 ranks各处理18 tasks，plus/minus各123次成功、75 tasks产生非零advantage；
`q_pi/compiler`裁剪前梯度为`.11708/62.25886`，校准峰值`24,450,292,224` bytes。held、validation与Test reward读取均为0。

冻结checkpoint为held5各生成唯一一套完整rank16 LoRA；物化candidate pair cosine `.95842`、direct pair `.14064`，
own/nearest-direct `.02321/.02968`、retrieval `1/5`，norm ratio `.94215`。随后gpu02 physical`0,1,2,3,4,7`的18个persistent
workers用`727.89s`完成strict paired250，candidate/source/shared/direct-earliest/direct-latest=`20/21/43/74/108`，candidate
per-task global0/9/18/25/36=`18/1/1/0/0`。episode key、env seed、policy seed root与policy-noise common prefix均零mismatch。

candidate只保留direct-latest `10/108` successes、`3/96` source-failure gains与shared `18/43` successes；source→candidate
retained/gained/lost=`11/9/10`，shared→candidate=`18/2/25`，direct-latest→candidate=`10/10/98`。Goal与Long为0，breadth
`3/5`，故完整门与near-pass都失败，不续1080、不轮fold、不进入`q_V`。

同checkpoint的40-panel full/prior support相对shared response为`1.24958/1.24794x`，per-task符号与closed-loop都只对齐`2/5`，
低于`4/5`资格门。于是MDCO同时关闭了“增加现有71个source-seen mappings足以挽救当前接口”的假设和这两种open-loop proxy
替代早期闭环的资格；它不否定EMBER总目标、Stage 0 observer、task-local rank16上界或全部compiler形式。正式证据为
`docs/evidence/ecp_20260823/stage1_mapping_diverse_compiler_oracle_tv540_gate.json`。

收尾时又把MDCO与首版Stage 1的228/570/1140同seed strict250逐row配对。MDCO的20个成功有19个已在source、shared或任一
首版节点出现，18个与shared重合、19个与首版三节点union重合；相对首版1140为retained/gained/lost=`15/5/12`。MDCO与
direct-latest重合的10个成功全部来自global task0，其余held4为0。与此同时，MDCO与首版1140同task effective-update cosine
均值只有`.09128`、own retrieval`2/5`，MDCO norm平均为首版`2.8521x`。故重复失败是outcome层easy-state/shared-support
吸引子，而非同一个LoRA参数方向；该结论不外推为逐action policy等价。

### 3.61 PECS exact-effect oracle成为唯一运行面

MDCO后的二次复盘没有继续建立v25，而是选择Policy-Effect Constrained Solver作为新的单一可证伪接口。首个实现不训练video
predictor或共享decoder：K2 action-hidden videos只经冻结Stage 0形成ordered event posterior；每个event从每条视频取一个
最大posterior frame，source/shared/successful experts在不安装Action Meta、固定canonical/antithetic `u=1` probes下形成
38-owner DCT4与完整flow effect distribution。固定solver从stable shared开始，对同一套LoRA leaves做exact VJP、逐owner
normalized step与thin-QR/core-SVD rank16 regauge；没有task ID、per-task early stop、持久optimizer或第二adapter。

新canonical owner为`effect_solver.py + effect_oracle.py + run_ecp_effect_oracle.py`。旧learned `q_pi + Program-to-LoRA`
trainer、calibration、materialization、support builder、配置与对应测试从active tree删除，精确历史继续由Git和formal artifacts
保存；这使仓库不再同时暴露可误启动的旧Writer。

fit ordinal71随后在gpu02 physical0完成唯一数值/资源profile。constant-step候选把effect `3.660019→.748501`但出现2次回升，
未过预注册单调门；唯一一次任务无关数值修正加入inverse-sqrt step decay。clean pushed `b7c87e7`的12步结果为
`3.660019→.774046`、final/initial `.211487`、0次回升、峰值18.72 GB、耗时153.67秒，故固定solver合同通过并冻结。
这只证明exact-effect VJP求解链可达，不是closed-loop正证据。

clean pushed detached `c400feb`随后为held5物化五套LoRA并完成strict250。candidate/source/shared/direct-earliest/
direct-latest=`58/21/43/74/108`，per global0/9/18/25/36=`31/11/16/0/0`。candidate相对source为13 retained、45 gained、
8 lost、净`+37`；相对shared为30 retained、28 gained、13 lost、净`+15`，exact McNemar `p=.02753`。因此local effects比
MDCO的20分产生了真实新能力，不是inner-loss-only假象。

但candidate只保留direct-latest `34/108` successes、`25/96` source-failure gains与shared `30/43` successes，breadth`3/5`，
Goal/Long均0；所有baseline配对字段零mismatch。该oracle按Gate 2失败，说明selected video frames上的owner DCT4与单点`u=1`
canonical/antithetic velocity不足以定义跨初始化successful-policy basin。按预登记合同只允许一次更强target：在相同support frames
加入successful expert从fixed noise完成official去噪的完整action/flow trajectory，再复跑同一oracle；若仍失败则停止PECS family。

### 3.62 PECS完整去噪trajectory完成最终复验并停止family

clean pushed `1142e5b`把唯一预登记增强接入同一PECS运行面：support frames、K2、fixed noise、rank16、stable-shared起点、
12-step solver与信息墙全部不动；每个probe沿PI0.5官方10步Euler网格保留全部`50x32` velocity与积分后`50x7` action，首步
继续保留38-owner DCT4。单event与独立官方循环的最终action最大误差为0，完整trajectory对LoRA leaves有`1,135,487`个非零finite
gradients。fit ordinal71 effect `5.82335→1.09769`、ratio`.18850`，held5五项ratio为
`.3012/.2797/.1341/.3258/.2338`，说明目标与固定solver都真实可达。

五套LoRA随后在gpu02 physical`0,1,2,3,7`并行物化；18个persistent workers在physical`0,1,2,3,4,7`用`707.43s`完成
54 shards与strict paired250。trajectory/local/source/shared/direct-earliest/direct-latest/MDCO=
`59/58/21/43/74/108/20`，trajectory per global0/9/18/25/36=`31/11/17/0/0`。它相对source为17 retained、42 gained、4 lost；
相对shared为30/29/13。但direct-latest success/gain retention仅`37/108、26/96`，shared retention`30/43`，breadth`3/5`且
Goal/Long均0，因此最终Gate 2失败。

trajectory相对local的effective-update cosine按task为`.753/.765/.777/.830/.848`，250行中47 retained、12 gained、11 lost。
所以完整target确实改变了LoRA和23个outcomes，却没有扩展suite/task支持；最早断点定为稀疏teacher-frame function constraints对
跨初始化closed-loop state distribution的欠识别，而非trajectory未接通。按card停止PECS family，不训练video effect predictor、
不再改solver/target、不建立下一版本。tracked PECS runtime与evaluator compatibility入口从active tree退休，Git `1142e5b`、
formal artifacts及`docs/evidence/ecp_20260823/pecs_complete_trajectory_held5_gate_20260823.json`保存精确历史。

### 3.63 GOMQ cycle 2完成迟到的validation8因果资格补审

跨版本复盘发现，历史GOMQ cycle 2是shared Writer最高single checkpoint：K4 correct validation8为`151/400`，但当时
same-task-other命令在CLI解析失败后未重试，也没有wrong/shuffled/reversed formal panel。2026-08-23从历史`8553b61`恢复
该冻结方法，只增加无梯度evaluation conditions；adapter checkpoint、teacher-video schedule、初态与policy RNG不变，没有训练、
checkpoint选择或validation action/reward梯度。

补齐的same-task-other/cross-suite-wrong/shuffled/reversed分别为`139/131/127/115`。correct相对四者的paired
correct-only/control-only为`28/16、43/23、40/16、50/14`，margin为`+12/+20/+24/+36`，exact p为
`.09614/.01866/.001842/7.07e-6`。五个panels的episode key、env seed、policy seed root、noise common prefix和teacher
reference videos均零mismatch。旧frozen source strict400作为identity/no-adapter proxy为48；与correct为43 both、108
correct-only、5 source-only，净`+103`、`p=2.83e-26`，但它不是learned language-only Writer baseline。

因此151具有真实视频内容与时序因果性；然而same-task correct-success retention仅`123/151=81.46%`，相邻cycles为
`151→135→131`，breadth`6/8`、两task为0且top3占`80.13%`，不满足鲁棒、稳定与breadth资格。GOMQ输出磁盘rank32在实数代数上
等价于rank不超过16的update，但BF16 rank16序列化形式未做formal paired400。裁决保留GOMQ为最强经验锚点，不恢复训练或自动
建立successor。

reversed首轮还暴露一个独立执行教训：历史工作树遗漏main已有的NFS-safe SQLite `DELETE` journal修复，在NFS WAL上出现重复
claim。已完成33/36 shards精确复用，只补48 rows后得到上述完整结果；随后把`247e6a8`修复叠加到审计authority `e0d6532`，
focused queue tests通过。科学历史代码恢复应叠加最新运行基础，不能连已修复的基础设施缺陷一起恢复。正式证据为
`docs/evidence/gomq_20260823/gomq_cycle2_causal_controls_strict400.json`、
`docs/evidence/gomq_20260823/gomq_cycle2_causal_adjudication.json`与
`docs/evidence/gomq_20260823/process_identifiability_and_rank_audit.json`。

### 3.64 Occupancy-complete privileged oracle通过teacher gate但未通过realization gate

独立专家复核把ECP Stage 1重新分成successful-policy equivalence、privileged realization与后续shared video inference三道门。
clean pushed `4e00982`为fold0 held5各训练一个不同seed37、固定step2000的独立rank16 expert；五项fixed250合计`113`，逐
global0/9/18/25/36为`26/32/37/13/5`，全部取得strict-success轨迹。随后每task建立48-state bank：8 initial、三个members各
8 successful、8 PECS candidate与8 source recovery states，并在同一官方双相机observation、fixed antithetic noise上缓存三
members以及source/carrier的owner DCT4、10-step flow与integrated action。Goal/Long也具备成功独立member与完整occupancy，
所以Stage 1A通过，不能再把后续失败归因于teacher缺失或稀疏教学帧support。

clean pushed `c2aaac1`先在非held fit ordinal71固定microbatch4，再用stable carrier A、Delta-B-only、rank16、12 steps并行求解
held5。五项objective ratio为`.5040/.5667/.5278/.6055/.4373`，trust penalty均0。final LoRAs直接进入原strict paired250，
得到`78`，逐task`36/12/30/0/0`；相对carrier43为retained/gained/lost=`35/43/8`。earliest/latest/independent success union为
`146`，扣除carrier已会的rows后有115个recoverable successes，final只恢复35个，overall `.304`且仅3/5 tasks为正。尽管
absolute、net gain与carrier retention通过，breadth、4/5严格胜carrier、Goal/Long和oracle recovery均失败，因此正式裁决为
Realization non-pass。

这一结果把最早断点从state coverage推进到realization坐标：当前48-state三particle owner/flow/action effect objective在Goal/
Long上也显著下降并具有高member confidence，却没有产生任何闭环成功，inner effect距离不能充当困难任务success-basin代理。
fixed-A capacity和effect objective/calibration仍未被本卡彼此分离；被关闭的只是当前bank、当前effect distance、stable fixed-A
carrier与当前12-step solver的组合，不是ECP核心目标或所有policy-effect compiler。按预注册合同暂停，不补step10/11、不做
solver小扫、不训练video predictor、不建立successor。正式证据为
`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`。

### 3.65 Fixed-A解析容量诊断判为binding并停止该参数化

重新核对专家最终意见后确认，fixed-A只是“strong carrier + small effective correction”的一种实现候选，不是ECP硬约束。为把
fixed-A capacity与既有effect objective/calibration分开，clean pushed `cc70aa6`加入零训练解析投影：对latest、independent、
earliest三个已知成功members逐target求
`B_star = B_expert A_expert A_carrier^T (A_carrier A_carrier^T)^+`，直接物化为一套完整rank16 LoRA。离线审计显示所需
correction energy coverage为`83.3%--96.7%`，expert absolute effective-update coverage为`41.5%--62.7%`，且Long高于
Goal，因此没有用内部几何代替闭环裁决。

从clean detached `cc70aa6`在gpu01 physical`1,2,3,4,5,7`并行完成三个strict250 arms；physical0为Prohibited且未使用。
latest、independent、earliest投影得分为`49/41/35`，逐global0/9/18/25/36分别为
`26/4/19/0/0、22/4/15/0/0、23/2/10/0/0`。相对matched direct的retained/gained/lost为
`31/18/77、22/19/91、14/21/60`，合计只保留`67/295=22.71%`。Goal的24个direct successes与Long的11个全部丢失；三个
projected arms合计125分，比三次carrier panel合计129还少4。

全部750行的episode key、environment seed、policy seed root、language与policy-noise common prefix零mismatch，18个workers
返回码全0。于是overall、Goal和Long三条capacity-binding预注册判据同时触发。当前fixed-A row space停止作为ECP Stage 1B
主线，不扫solver、rank、步数或插值，不训练video predictor，也不进入Stage 1C。该负结果只否定当前fixed-A表达合同；下一允许
问题是一个zero-correction精确返回carrier、effective-additive、允许row/column-space移动且确定性retract为single rank16的
realization operator。正式证据为
`docs/evidence/ecp_20260823/ecp_fixed_a_capacity_gate_20260823.json`。

### 3.66 Mobile-rank4 residual容量门恢复direct级表现但裁决为mixed

fixed-A关闭后没有立即实现free-A/B solver。资产审计确认stable carrier是精确rank12、后4个B columns为0；历史
`shared12 + phase-code residual4`的`37/33`只关闭当时learned mapping，未分离mobile residual容量。clean pushed `083ed98`
因此预注册零训练解析门：逐target对`W_expert-W_carrier`取best-rank4 SVD，再与不变carrier12按rank拼成一套rank16。15个
member-task的correction/expert energy coverage为`99.49%--99.69%/95.34%--98.90%`。

从clean detached `083ed98`在gpu01 physical`1,2,3,4,5,7`并行完成三个strict250 arms，physical0未使用。latest、
independent、earliest为`110/120/76`，逐global0/9/18/25/36分别为
`28/31/40/4/7、27/36/38/11/8、18/18/33/2/5`，全部5/5非零并略高于matched direct `108/113/74`。matched
retained/gained/lost合计`245/61/50`，overall retention`83.05%`；Goal为`15/24=62.5%`。

唯一未过capacity-supported条款是Long matched-member retention：direct/projected absolute合计`11/20`，但只保留同row
`4/11=36.36%`。capacity-binding条款也未触发，故按卡裁决为mixed。post-hoc、无新rollout的multiple-member union分析为
direct/projected/overlap=`11/16/6`，union retention`54.55%`；它定位到member-specific policy identity churn，但不修改正式门。
750行episode key、env/policy seed、language与noise common prefix零mismatch，18 workers全部返回0。当前不实现solver、不扫
rank，先明确exact-row与policy-equivalence union的科学口径。正式证据为
`docs/evidence/ecp_20260823/ecp_mobile_rank4_residual_capacity_gate_20260823.json`。

### 3.67 Mobile-rank4 raw-factor solver闭环non-pass

在不改48-state bank、three-member objective、stable carrier、12 steps或原Gate的条件下，clean pushed `f75bafc`把
fixed-A runtime替换为`carrier12 + jointly mobile residual4`。ordinal71 profile与五个held solvers全部finite，objective严格
下降，A/B梯度接通，输出一套complete rank16 LoRA；因此随后负结果不是工程invalidation。

gpu01 physical`1,6`的6个persistent workers完成strict paired250，physical0未选择，全部返回码0。final为`49`，逐global
`0/9/18/25/36`为`40/3/6/0/0`；相对carrier43的retained/gained/lost为`41/8/2`，而multiple-member union recoverable gap
只恢复`3/115=.0261`。Goal/Long、breadth、absolute、net和union recovery门均失败，故按卡停止当前operator，不补step10/11，
不扫rank/step/LR/init/weight/member，不进入Stage 1C。

零新rollout定位显示，真实successful-member responses在同一objective上的最低值仅`.060--.163`，而solver final为
`1.915--3.262`；known-success mobile-rank4 projections的trust为`1.341--2.281`，final仅
`.000915--.001171`，且effective correction cosine只有`.041--.077`、norm ratio约1%。因此下一科学接口不是再调当前
raw-factor solver，而是独立检验gauge-invariant effective-update direction或很小的target-local preconditioner。正式证据：
`docs/evidence/ecp_20260823/ecp_mobile_rank4_solver_gate_20260823.json`。

### 3.68 Effective-update reachability在非held profile初始回溯处停止

为分离raw-factor rank-zero奇点，clean pushed `fc678f3`按事前card把唯一Stage 1B runtime替换为matrix-free effective-update
solver。它在exact carrier以固定8列input sketch和4次VJP构造rank4方向；只有首步通过固定
`1,1/2,1/4,1/8,1/16`完整objective回溯后，才会执行最多8次Gram-preconditioned tangent VJP。effect bank、three-member
objective、stable carrier、rank4 residual、trust1.5、总VJP12与held closed-loop Gate全部冻结。

gpu01 physical5上的非held ordinal71 profile完整生成formal result与adapter，physical0未使用。初态精确等于carrier，
initial/best-member objective为
`2.214329/.127698`；initial sketch finite且方向导数`-81.8873`。然而五个固定尺度均未被接受，故final仍为carrier，accepted
steps、gap recovery与trust全为0，stop reason为`initial_backtracking_failed`，只消耗4次VJP，峰值allocated约18.95GB。

按预注册Profile non-pass条款，没有物化held5、运行closed loop、Action Meta control或shared Writer，也没有扫更小alpha、seed、
trust、damping、rank或VJP预算。该结果只关闭当前固定solver合同；它未执行Gram tangent，不能外推否定任意更小局部步、
mobile-rank4容量、effect target或ECP核心。正式证据：
`docs/evidence/ecp_20260823/ecp_effective_update_profile_gate_20260823.json`。

### 3.69 ECP全过程专家对齐审计修正了阶段命名，不改写实验结果

2026-08-24没有启动训练、评测或新架构。仓库逐项对照专家原始方案、最后修正案、retained source和全部ECP formal evidence，
确认occupancy-complete工作后的阶段叙述过强：48-state、three-member bank是successful-policy evidence/state-support资产，
不是输出同构Program posterior的distributional `q_pi(P)`；当前`stage1_oracle.py`直接加载effect bank进入solver，没有
Program-to-effect forward。因此此前“Stage 1A通过”“Stage 1B失败并阻止Stage 1C”的表述被校正为“evidence prerequisite
完成、direct-effect realization子门已裁决、完整privileged Program链尚未实现”。

所有既有数值与窄结论保持不变：independent members `113/250`；fixed-A learned oracle `78/250`；fixed-A analytic
`49/41/35`；mobile-rank4 analytic `110/120/76`；mobile raw-factor solver `49/250`；effective-update profile 0 accepted
steps。它们仍分别证明independent policy/state coverage、fixed-A binding、mobile-rank4 capacity及两个具体operator non-pass，
但不裁决distributional `q_pi(P)`、Program-to-effect compiler或`q_V(P|L,V)`。

第3.68之后提出的process-identifying data feasibility仍是有效数据发现，但“它已成为唯一下一主线”被降级为竞争性资格缺口。
在专家重新确定direct-effect oracle、`q_pi(P)`、Program-to-effect、fixed realization与`q_V`的阶段关系前，仓库不创建
successor或启动GPU。完整落实矩阵、成果边界、疑问与候选顺序见
`docs/ecp_expert_alignment_audit_20260824.md`。

### 3.70 专家最终复核固定deployment-time occupancy completion与后继顺序

独立专家对`main@6a97185126ab640c3f9a6a719084dc0268ddd8e9`完成全过程复核。专家确认第3.69节的核心纠偏：当前effect bank
不是distributional Program posterior，direct-effect solver绕过Program，完整ECP Stage 1未执行也未被整体证伪。进一步的最终
修正是，最深未解问题不是单独缺少`q_pi`或更好solver，而是deployment-time occupancy completion：Writer只见language/videos，
却必须生成一套在编译时不可见的future policy occupancy上有效的静态LoRA。

最终阶段名固定为Stage 0-V、1A-E、1A-P、1B-R0、1B-C、1B-O、2、3、4。Stage 1B-R0仍是ECP必要的privileged lower-bound
diagnostic，但读取future occupancy；正式Stage 1B-C只能从Program生成effective update。学习顺序固定为Program schema、
effective coordinate与deployment realizer先建立并冻结，再训练distributional `q_pi`，privileged full-chain过门后才训练同构`q_V`。

专家还校正了当前evidence边界：48-state资产改称four-category structured occupancy panel；antithetic probe已提前平均；
off-policy member response没有continuation/recovery validity；stage-wise soft-min可能拼接不存在的policy。后继必须保留probe
particles、global member identity与member-state validity，区分`P_visible`和rollout-only `Z_robust`。

`carrier rank12 + mobile rank4 residual`解析投影`110/120/76`被认定为最近最强的结构正证据，并成为默认输出拓扑。下一realizer
不再读取future occupancy，而是在balanced-SVD rank4或fixed two-sided sketch等固定canonical coordinate上训练小型target-local
amortized map；known-success correction先用于objective/path calibration。Action Meta默认关闭，只作matched control。

专家最终执行计划为：先把冻结GOMQ cycle2确定性canonicalize成真实rank16并补一次strict400 archival baseline；随后并行完成
process-identifying最小pair feasibility与mobile-rank4 realizer calibration；再fresh建立owner-specific
`P_lang/P_scene/P_process`、distributional `q_pi`、多fold frozen privileged full-chain、deployment `q_V`、ordinary joint Writer、
structured outer credit与最终validation/Test资格。最小process pair必须先于新`q_pi/q_V`训练，完整family-disjoint suite必须先于
最终Stage 0/`q_pi`/`q_V`共同训练。该复核只改变当前阶段解释和未来顺序，不改写任何历史分数；完整记录见
`docs/ecp_expert_alignment_audit_20260824.md`第9节。

### 3.71 GOMQ真实rank16 Phase 0以136/400关闭

在clean pushed `ac233fa0e94b40c525d75746ef2d8fdfb4dc0046`上，将GOMQ cycle2已封存的400套rank32 episode LoRA按
`A32=[A0;A0]、B32=[B0,deltaB] -> A16=A0、B16=B0+deltaB`确定性canonicalize。38/38 targets的A半块逐元素相等；输出
为完整76-tensor、1,287,168参数的真实rank16 LoRA。全程没有训练、Writer/video forward、checkpoint选择或adapter融合。

唯一strict400 correct评测得到`136/400`，逐task为`16/0/0/35/46/34/0/5`、breadth`5/8`。相对历史rank32的151，
retained/gained/lost=`123/13/28`、churn41、Jaccard`.75`；suite变化为Long`-11`、Goal`-5`、Object`-1`、Spatial`+2`。
400行episode、environment seed与policy-noise common prefix严格匹配，72/72 shards和18 workers全部正常完成。

由于结果低于事前固定的145门，真实rank16结果不成为absolute Phase 0基线，历史151也只保留机制与历史证据。实数代数上的
effective-rank事实仍成立，但native BF16 regrouping后的实际闭环行为不能视为等价。依预注册合同不做dtype、rank、scale、seed或
checkpoint救援，GOMQ正式归档关闭。完整证据：
`docs/evidence/gomq_20260824/gomq_cycle2_effective_rank16_strict400.json`。

### 3.72 首个process minimal pair在teacher Gate A以19/100关闭

按预注册`docs/ecp_process_minimal_pair_gate_20260824.md`，clean pushed
`d1975c3d1526e091e8675ea7df7178e6410b4a7d`在gpu01 physical2--7并行采集soup→butter与butter→soup各50条同面板轨迹。
六个workers全部返回0，100个privileged ledgers、50对state IDs与policy-noise common prefix完整；19条公开成功video通过信息墙。

结果分别为`0/50`与`19/50`，总计`19/100`，低于每方向20和总计50的固定门。两方向第一predicate都完成`50/50`，第二predicate
分别只完成`0/50`和`19/50`，invalid均为0；所以失败定位在phase switch后的sequential teacher support，不在wrapper或首个
primitive。19个成功均在step275--385；旧collector让66个失败chunk尾部到401--404，但严格horizon success不变，边界在
`90090bf38874eb6d4202d3f2a7b262a4eced736a`修正且没有科学重跑。

Gate B、process suite扩展和`q_pi/q_V`均未启动。后继只允许更强privileged sequential teacher或有明确依据的替代family；与此
并行进入专家既定的Phase 2 mobile-rank4 realizer calibration。证据：
`docs/evidence/ecp_20260824/ecp_process_minimal_pair_teacher_gate_20260824.json`。

### 3.73 known-success effect-path calibration全门通过

clean pushed `4cddcabead992476483ea337046728374ba87b9b`在gpu01 physical1--5查询五个tasks、三个independent
successful members的15条balanced-SVD rank4 effect paths，没有新closed-loop rows或optimizer step。matching verified、
global-particle与legacy stage-wise loss均在15/15条路径上严格单调下降；15/15在`alpha=1/8`已改善，15/15
的最低点在`alpha>=3/4`，5/5 tasks的global objective均优于carrier，包括Goal与Long。

这将deterministic-sign balanced-SVD rank4确定为Phase 2B canonical coordinate，并将旧solver失败定位为没有到达
已存在的successful basin。当前bank的probe轴已被平均，candidate/recovery也缺continuation validity，所以本轮
不证明realizer、distributional `q_pi`、Program-to-effect或video inference。后续须先建立successor evidence与只读
structured code的target-local amortized realizer，不再做新task-local solver。证据：
`docs/evidence/ecp_20260824/ecp_effect_path_calibration_gate_20260824.json`。

### 3.74 fixed effect realizer fold0以33/37低于carrier关闭

Phase 2B在clean pushed `565c055`上捕获118个successful members、95 tasks、188条on-policy trajectories与376个保留
probe-sign的effect particles；fold0的fit-only owner-local `512 -> 128` coordinate由90 fit tasks/108 members拟合，5 held
tasks/10 members只做transform。随后小型target-local realizer在fit mappings上训练1000步，step800/1000 total loss分别为
`.31668/.26605`，held target residual在训练与materialization中的读取次数均为0。

clean pushed `0247a19`物化两个相邻checkpoint的single carrier12+residual4 rank16 LoRA；step1000 invalidity screen为
`8/50`，matching carrier为`9/50`，没有用于选择checkpoint。clean pushed detached `e806693`随后同时完成两个strict250：
step800 `33=32/0/1/0/0`，step1000 `37=36/0/1/0/0`；carrier为`43=38/1/4/0/0`，direct-latest为
`108=27/30/40/8/3`。step1000相对carrier retained/gained/lost=`33/4/10`，两节点breadth均为`2/5`且Goal/Long为0。
12 workers全部返回0，episode/env/policy/language/noise pairing零mismatch，因此fold0强门明确non-pass。

两节点完成后的post-hoc定位显示，held latest在冻结PCA中仍保留`79.1%--89.2%`中心化response energy；但step1000
对Goal/Long只生成known residual `7.1%/22.9%`的effective energy。失败由此落在cross-task effect-code-to-residual mapping，
不是rank4容量、balanced-SVD表达或单独PCA压缩。按预注册边界不触发two-sided fallback，不启动fold1，也不以更多训练或
超参变体救援；fresh Program、`q_pi/q_V`继续等待新的realizer裁决与process Gate。完整证据：
`docs/evidence/ecp_20260824/ecp_fixed_effect_realizer_fold0_gate_20260824.json`。

### 3.75 centered two-sided coordinate以80/250关闭

依据fixed realizer结束后的mean/innovation分解，专家预留的第二种principled coordinate在任何实现和rollout前重新预注册。
clean pushed detached `8aab214`只用fold0的90 fit tasks/108 members拟合fixed width8 input/output probes、task-equal sketch mean与
centered whitened sample-space basis；38个targets的active rank均为107。五个held tasks/10 members只在coordinate冻结后做
transform，held coordinate updates与optimizer steps均为0。top4 deterministic reconstruction输出五套single
carrier12+residual4 rank16 LoRA。

五个held latest corrections的reconstructed effective cosine逐global为`.957/.950/.960/.877/.953`，但唯一strict250只得到
`80=24/10/46/0/0`、breadth3。结果低于83门，Goal/Long均为0；相对carrier只保留`23/43` successes，相对known-latest
`110`只保留`59`，均低于预注册retention门。gpu01 physical`1,2,3,4,5,7`上的12 workers全部返回0，36/36 shards与250 rows
完整；两套reference的episode/env/policy/language/noise pairing均零mismatch。

本轮说明高aggregate update cosine不足以保证task-specific closed-loop support：Object从known-latest 40升到46，但Goal/Long
全部丢失且Spatial高churn。依合同停止当前coordinate，不训练centered-innovation realizer、不启动fold1、不扫probe/width/rank/
threshold。该结果不否定Program、distributional `q_pi`、`q_V`、video inference或完整ECP；下一shared-realizer机制须先重新
咨询专家。证据：`docs/evidence/ecp_20260824/ecp_centered_two_sided_coordinate_gate_20260824.json`。

Gate后只读lineage分解显示，exact earliest→latest correction仅占latest update能量`1.3%--10.8%`，却对应mobile projection
从76到110的34个success增量；fit90 span对该innovation的cosine只有`.318--.647`。q_proj占总update能量`91.3%--92.6%`，
所以aggregate cosine掩盖了低能量行为方向。fit basis已达到108 centered members的最大rank107；在同一evidence上增加width
不能补方向，后继若存在必须引入span外独立mapping，而非继续coordinate/decoder sweep。

### 3.76 phase task-local experts的process Gate A2以44/100关闭

原Gate A的两个primitive第一事件都能完成，但shared source在phase switch后只得到`0/50、19/50`。
Gate A2唯一替换为LIBERO-90 task55/56已有step1000 rank16 task-local experts；两个expert在各自primitive
formal panel均为`50/50`，本轮没有新训练或checkpoint选择。

clean pushed detached `24c5bdc3ec83c78ff36ec514717d15da8560e81f`在gpu01 physical`1,2,3,4,5,7`完成
100行strict400，physical0未使用，6 workers全部返回0。100份privileged ledgers、50对state IDs/noise common
prefix、100/100 phase-expert alignment完整；44条公开成功video仅含camera1/camera2/language/stride/schema/source_steps。

结果为soup→butter `0/50`、butter→soup `44/50`，总计`44/100`。两向第一event均为`50/50`；
soup-first的50行全部只完成soup，butter-first有44行完成两事件。相对旧teacher的`0/19`，task-local experts产生
25条净增，但没有让pair双向可行；剩余失败是soup occupancy之后task56 butter primitive的恢复支持，而非
adapter加载、phase route、首事件或wrapper。

结果同时未达每方向20和总计50的门，故Gate A2 non-pass。当前family对phase-composed source/task-local primitive
teacher关闭，Gate B、suite扩展与`q_pi/q_V`不启动，也不以step2000、延长horizon、挑state或改predicate救援。
下一决策是true composite privileged expert/data acquisition或物理机制不同的source-unseen family。证据：
`docs/evidence/ecp_20260824/ecp_process_phase_expert_teacher_gate_20260824.json`。

### 3.77 separate-plates Gate A3以37/100关闭phase-composed primitive teacher

为去掉scene3 shared-tray occupancy confound，后继选用LIBERO-90 task65/68：scene、fixtures、objects与init specification
相同，red mug→left plate和yellow-white mug→right plate的目标objects/receptacles互不共享；source只见过两个primitive，
没见过统一conjunctive goal或任一required order。两个step1000 task experts原primitive formal为`43/50、47/50`。

新custom BDDL、统一goal-only language、同task65的50个init states、相反required orders与render256/stride5合同在
`docs/ecp_process_separate_plates_teacher_gate_20260824.md`冻结。双向state0 smoke都成功后，clean pushed detached
`4bf50394f75307568339143d17a39c0bfe2c2829`在gpu01 physical`1,2,3,4,5,7`完成100行formal；六个worker返回0，
physical0 Prohibited且未使用。

结果为red→yellow-white `28/50`、yellow-white→red `9/50`、总计`37/100`。两向第一event完成`43/50、46/50`，
第二event只完成`30/50、9/50`；invalid与phase/expert错配均为0。100份ledger、50对noise和37条公开成功video完整，
public字段与统一language无泄漏。反向37个second-phase失败在第一event后仍剩余median297步，故不是horizon-only问题。

Gate A3未达每方向20和总计50，判为non-pass。结合soup/butter A2，phase-composed primitive expert在共享receptacle与物理分离
目标上都缺少可靠的post-sibling-goal恢复支持；该teacher mechanism关闭，不以task66/67同机制full formal、step2000、
horizon或state/seed修改救援。Gate B、process suite与`q_pi/q_V`仍不启动。下一数据前置改为真正order-specific composite
privileged policy/data；37条完整成功轨迹只作bootstrap evidence。正式证据：
`docs/evidence/ecp_20260824/ecp_process_separate_plates_teacher_gate_20260824.json`。

### 3.78 Gate A3成功轨迹已转换为两个composite policy-SFT数据集

Gate A3的37条成功episode具备完整teacher actions与每步前后render256双相机公开RGB，但缺少现有task-expert trainer所需的
逐action 8维PI0.5 state。新增builder从clean pushed detached `b8fb0bfc5b8991b3af646a5f7f700c89a5ca2ad0`
确定性重放保存actions，不重新执行PI0.5 inference；每个action前采集eef position、axis-angle orientation和gripper qpos，
并与原RGB/actions写成标准LIBERO风格HDF5。

red→yellow-white的28/28与yellow-white→red的9/9 replay全部success、0 divergence，completion steps逐条匹配原ledger。
两份HDF5分别有`9,479/3,248` action rows，bytes为`3,728,011,012/1,277,410,960`；现有
`FunctionalQueryDataset`读取双相机`[3,256,256]`、8维state和`[50,7]`action chunk的正式门通过。target40 action
reads与新增PI0.5 forwards均为0。

该结果解决的是privileged composite policy的数据authority，不是process teacher性能。后继固定为两个独立rank16、
step1000 composite experts，再在原50×2面板报告bootstrap retention与原failure acquisition。通过原teacher门之前，
Gate B、process suite和deployment compiler仍不启动。证据：
`docs/evidence/ecp_20260824/ecp_composite_teacher_bootstrap_data_20260824.json`。

### 3.79 fixed-step1000 composite SFT未通过state0，转入唯一一轮on-policy phase-expert distillation

两个order-specific rank16 LoRA已从clean pushed detached `38dbffd34951cd4d6c76584f137d0e870bdbe073`各完成固定1000步，
worker均exit0、各1000行metrics，mean loss分别`.135630→.113355`与`.124071→.103308`。Gate A4 state0资格检查却双向
non-pass：red-first在step114 wrong-first invalid；yellow-first在step114完成第一事件后，到400仍未完成第二事件。正式100行
面板因此没有启动。

paired离线诊断确认bootstrap动作authority、adapter差异与primitive teacher复现都正确，并发现composite checkpoint可改善原
成功trajectory transition上的局部误差或后45 token，但不能稳定控制实际执行前5 token及自身访问状态。最早失效接口定位为
successful-occupancy SFT的闭环distribution shift。后继合同只允许一轮：固定step1000 policy收集两variant各50条on-policy
episodes，在每个replan state由对应phase expert提供相同observation/noise下的privileged 50-step标签；从step1000权重fresh
optimizer固定训练两遍数据后直接重跑原Gate A4，不做第二轮或超参救援。state0采集smoke的23个queries已通过标签逐项对齐。

合同：`docs/ecp_composite_teacher_distillation_gate_20260824.md`；证据：
`docs/evidence/ecp_20260824/ecp_composite_teacher_step1000_preformal_20260824.json`。

### 3.80 唯一一轮on-policy phase-expert distillation数据已完成

clean pushed detached `7527568`在gpu01 physical1--6并发采集两variant各50个固定states，未使用prohibited physical0。
red-first得到`2773` queries，step1000 behavior为`5/50`成功、24 invalid；yellow-first得到`3998` queries，behavior为
`5/50`成功、0 invalid。所有episodes无论结果均保留，phase expert只在相同observation/noise上提供privileged full50 action
chunk，target40 action/reward读取为0。

六个HDF5总计`1,594,115,631` bytes，100个state groups无缺失重复，manifest、metadata query count和training reader的
camera/state/action shape及finite检查通过。固定两遍、batch16机械派生formal final steps为`347/500`；两个config均从对应
step1000 LoRA权重warm-start、fresh optimizer、peak LR `1e-5`，不按behavior outcome或loss选checkpoint。

证据：`docs/evidence/ecp_20260824/ecp_composite_teacher_distillation_data_20260824.json`。

### 3.81 专家取消现有distillation并固定composite-context recovery teacher Gate

独立专家对远程`main@d8eca7987a4ad2a59c5d27738b29a8d4d9bfd161`复核后，明确判定已封存但未启动的
`347/500`步on-policy phase-expert distillation不应执行。采集虽然覆盖两个step1000 composite policies自己的固定100-row
occupancy，却直接把已经在A2/A3 composite occupancy上失败过的primitive experts当成动作oracle；代码没有从query state执行
teacher continuation，也没有验证当前goal完成、已完成goal保持或最终composite success。

监督目标另有独立phase语义错误。真实privileged controller在event完成时丢弃旧action chunk、切换expert并重新规划；当前
HDF5和reader却把查询时primitive expert的完整`[50,7]`全部当作有效标签，不读取phase boundary或mask切换后的tail。较长失败
episode又按query自然获得更大权重，且没有混入原37条成功轨迹保护success support。因此训练无论提高或降低Gate A，都不能
清楚区分weak oracle、phase-tail错位、occupancy coverage、SFT优化和static LoRA能力。现有数据改作student occupancy与
weak-teacher response历史资产，不再是formal oracle训练集。

专家确认路线在GOMQ、Gate A/A2/A3、37条成功轨迹bootstrap、两个composite SFT、effect-path calibration和两种realizer裁决
之前总体对齐；偏移从composite SFT闭环失败后重新升级primitive experts为student-state oracle开始。当前唯一下一步骤改为：
从A3的28/9条真实成功轨迹截取second-phase动作，分别与对应primitive成功数据按50/50混合；从原primitive LoRA初始化两个
direction-specific rank16 recovery experts。正式controller在first event使用冻结primitive，phase transition时丢弃旧chunk，
second event使用recovery expert。原100-row Gate A及双向各`>=20/50`、total `>=50/100`、invalid/route/pairing/泄漏全为0的门
保持不变。

若该recovery teacher仍失败，则task65/68上的primitive composition、composite SFT、phase-expert distillation与recovery SFT
共同关闭，不做第二轮DAgger、训练延长、超参扫描或task66/67同机制替换；下一process controller必须来自独立成立并先过Gate A
的scripted planner、human/teleoperation、privileged MPC或task-local simulator RL。当前fit90 shared-realizer family继续关闭；
recovery Gate即使通过也只授权Gate B，不等于fresh Stage 0、`q_pi/q_V`或Writer已经启动。完整裁决：
`docs/ecp_recovery_teacher_expert_ruling_20260824.md`。

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
