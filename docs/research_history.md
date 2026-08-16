# EMBER Research History

本文是历史实验的唯一精炼索引。它记录“测了什么、严格结果是什么、实际证明了什么、不能重复什么”；
不是活动设计，也不提供可直接启动的命令。删除的旧设计与逐日运行日志完整保存在 Git commit
`3a6f801d08facb3e855ab24f84e0b53cb8802e88`及其祖先，正式结果保存在`runs/outputs/`。

当前真相只取`AGENTS.md`、`docs/active_session_handoff.md`和`docs/execution_brief.md`。截至
2026-08-16，最高correct单点SFMC144因lost15/churn31不具稳定资格；最新NPVC为136、breadth6且相对LPCP143
为`120/16/23`、churn39。它虽形成held跨视频coherent、时序敏感、native-scale写入，但写入幅度/一致性与
closed-loop增益错位，full24还使train task4共同方向坍塌。v6-fast仍是有完整五臂的历史最好，长期严格
`>150/400`追求及稳定约145资格均未完成。Dynamic-K
backbone-memory、semantic-address与Direct-Family-B macro50 K1 strict分别为`100/101/102`。Direct-Family-B
相对semantic101为`82 retained/20 gained/19 lost`，breadth从6降到5；相对old134为`80/22/54`。task-mean
effective-BA cosine虽从`.77947`降到`.74895`，closed-loop只增1，证明删除family hidden/GELU只轻微改善几何，
不是解决Program→policy effectiveness和共同积累的充分条件。同一checkpoint K4=`98/400`，相对K1为
`80 retained/18 gained/22 lost`；set将same-task相对方差约降`6.3x`却保持task mean cosine`.99604`，证明
few-shot nuisance reduction已工作但没有修正高层任务方向。该arm终局non-pass。Task-Grounded Visual-Value完整
曲线=`88/86/86/96`，macro200 breadth6、top3=`92/96`；150→200仍churn40，相对old134丢66只得28。Full-Factor
随后macro50=`91`，只比matched fixed-A净增3，并形成tiny-B/weak-near-orthogonal BA，终局non-pass。V6 Dynamic
Slot-Set随后K4=`130`、breadth6，same-task方差降`9.26x`而task mean几乎不变，终局non-pass。Shared-Core
Ordered-Procedure AS达到139，raw reward与ADSP均为138；ADSP虽修正6条train24 support violations，相对AS仍为
`116/22/23`且churn45。其后V6-LPCP、PCSD、CV-CSD、SFMC、Gradient-Open与CCT继续把接口推进到factor
commitment；CCT证明held residual在native BF16 factor/compiler处消失，NPVC随后关闭了该断裂，但strict降至136。
PAFS fixed address与SJNV shared joint gate均在formal前held机制门终局；SJNV进一步证明continuous hidden可有约`.94`
跨视频cosine，却在冻结W2后变成raw factor`.02135/.26592`。DJNFR以八个direct heads绕过W2，post-full24
validation8 BA仍为`.790242/.785834`且8/8过门，但strict仅136，相对LPCP为`120/16/23`；persistent failures
获得最大改写。最早断点由生成端后移到selected-success-only credit的held reward方向。当前DF-PCSP只把credit
改成同一初态共同观测处的winner-vs-loser首段preference，DJNFR carrier/direct LoRA图完全不变。首次clean smoke
确认同seed顺序reset并不保证两臂首观测逐元素相同；`07b764b`又确认flattened state后再次hard reset仍令两相机
分别21,423/27,429个像素值不同，而language/state tokens完全相同。两次均在credit前工程终止。canonical现每lane
只hard reset一次，每臂deterministic soft reset controller/observables后恢复同一qpos/qvel；不增加rollout或
forward。最终exact task4/task7均为tie；task9/15/18均产生真实discordant credit与margin descent，但task9
held/train BA仅`.105x`、task18 train跨video仅`.290/.428`，三个anchors只有task15全门通过。DF-PCSP因此在
full24前终局。
最新终局DF-SOCP保留同一生成图，把exact preference扩展到winner成功occupancy全部replans。固定task9/15/18均形成
强train/held跨video共同方向，但stored winner actions来自动态B2/B1、loser counterfactual来自B8；task9/15的批形
数值差异为名义策略contrast的`1.086x/1.693x`。三项wall又为DF-PCSP的`3.083x/5.335x/3.887x`，task9
held/train仅`.118x`。按门未full24/strict；后继MB-SOP用同B8重查两臂，并把每条成功轨迹压成8个等进度
max-disagreement states。clean `ad65347`三anchor全部复现固定outcomes/counts，wall降到DF-PCSP的
`1.655/2.119/1.542x`，train/held跨video BA均健康；但task15/18同一panel在AdamW后margin增加，task9
held/train仅`.1096x`，故仍未full24/strict。额外四view flat-gradient测得三anchor raw mean对4/4 views均为下降，
view-to-mean cosine minimum仍有`.695/.629/.601`。因此最早接口后移到raw functional gradient如何形成实际finite
parameter delta，而不是matched panel、carrier、video aggregation或LoRA topology。
AR-EC保留MB-SOP全部科学图，把final delta严格放回负raw shared gradient方向，同时保留AdamW候选global L2
radius与moments。clean `b578d56`三anchor raw coverage均为`4/4`、final cosine均为1，但每个任务都只有`1/4`
post margins下降；Adam radius为raw gradient L2的`6333/7988/4294x`。train/held BA coherence、q/v/action、时序和
core wall均健康，证明方向旋转不是最早问题，full radius已经越过共同局部下降区间。AR-EC未full24/strict。
AV-MBC随后只把该upper radius改成固定顺序的all-view monotone backtracking，接受第一个四view全下降
candidate；不改credit、carrier、rank、video aggregation或direction。其clean `aa819f2`首次三anchor完整exit0，
但搜索把gradient-enabled before margin与inference-only candidate margin混比，恢复step0后的action probe也混合
rollout batch与batch1路径，故该轮仅是工程诊断而非科学non-pass。canonical已改为同inference evaluator的exact
step0 baseline与candidate比较，并统一fixed-action batch1路径。修正版clean `202a64d`终局：task18在1/32半径
全门通过；task9仅在1/1024接受且held4/8、held/train`.18446x`；task15到1/1024仍无all-view step并恢复exact
no-op。AV-MBC因此没有full24/strict。它只否定固定raw-mean ray上的scalar radius，不否定direction-changing
commitment。MMCD clean `fc3bdd7`保留全部science graph，唯一从已有四view gradients确定性求maximum-margin
direction。task9/15/18 continuous worst margin提高`1.216/1.334/1.356x`，但native分别为j0大步且held/train
`.160558x`、j0--10仍no-op、j6且全门通过；只有1/3 anchors通过，未full24/strict。它证明更好的continuous
first-order direction不稳定等价于native BF16 finite-step或held policy-effective commitment。
PAV-BC clean `581140c`保留同一science graph，唯一把final ray换成实际AdamW candidate delta并做同路径all-view
backtracking。task9在j5接受但held/train仅`.109466x`，几乎复现MB-SOP的`.109639x`；task15与task18到j10均无
共同candidate并exact no-op，0/3 anchors通过。至此raw equal-mean、raw maximum-margin和Adam-preconditioned三类
parameter rays均不能形成跨task稳定native commitment。ALB-NV clean `0899166`只改变输出参数化：固定LPCP A、
删除A residual heads，只用四个B heads令新增effective-BA严格为`delta-B A0`。固定task9/15/18结果为1/3：task9
连续gradient共同但native ray全拒绝；task15从PAV no-op变成j5且held BA aggregate/held-train通过，但仅5/8和raw-B
coherence失败；task18从no-op变成j0并以held BA `.774/.785`、8/8、held/train`1.030x`全门通过。故fixed-A
线性化确实消除了部分gauge/cross-term问题，却未使向非零B0追加的小residual跨task稳定native-safe；未启动
full24/strict。

NZRB-C clean `d4fc92e`随后只把ALB residual移入同一rank32 adapter的native-zero second-B bank，first rank16
LPCP carrier不压缩。三anchor stable rank-bank合同五项误差均精确0；初版结构gate约`1e-3`失败是跨autocast
重算carrier的analysis counterfactual错误。task15/18 held BA分别达到`.952/.940`与`.934/.922`，raw-B=
`.953/.941,.933/.920`且均8/8，说明zero origin修复accepted update的native/held coherence。task9仍无任何
all-view candidate并exact no-op，且rank32正常BF16 compute shape令固定outcome从`2/1,26`漂为`1/0,25`；三anchor
wall/ALB=`1.16565x>1.15x`。纠正后2/3 anchors与吞吐门失败，故未启动full24/strict。该轮否定继续改变factor
origin，不否定memory、rank8、few-shot或生成LoRA；最早缺口转为reward-useful Value/acceptance的finite跨video/task
policy step。
精确状态取`docs/active_session_handoff.md`。

## 1. Stable problem definition

EMBER研究：给定exact task language与一条或多条action-hidden teacher videos，shared Writer一次生成一套完整
task adaptation；该adaptation挂到同一个frozen π0.5-LIBERO source policy后，应在未见初始化上闭环完成任务。
视频是唯一dynamic value，不允许language-only LoRA bypass、teacher action/reward/proprio读取、task ID、
文件名、object pose、多个独立LoRA/checkpoint平均或held oracle。rank8、rank14、rank16均是具体实验合同，不是
长期问题定义；最近Shared-Core Procedure-Set使用历史v6的完整rank16 topology完成了受控机制开发。

最终方法必须由同一single checkpoint的strict paired closed-loop裁决，并同时满足高absolute、task breadth、
低checkpoint换手、same-task鲁棒和correct优于wrong/shuffled/reversed/no-video。训练loss、functional loss、
LoRA能量/秩/cosine、重建误差和漂亮内部margin都只能作机制证据。

## 2. Fixed baselines

| 基线 | 严格证据 | 解释边界 |
| --- | ---: | --- |
| generic π0.5 | `0/400` | 原始policy缺少LIBERO embodiment能力，不是EMBER方法失败 |
| frozen source base | `48/400` | 过滤LIBERO-90训练后的共享起点；不读目标视频 |
| mixed-task Source-SFT rank128 | `109/400` | privileged target-action shared LoRA参照，不是同信息墙baseline |
| v5.2 old | `132/138/74/82/83` | 最强correct-vs-negative视频特异性形态之一，absolute不足 |
| v5.2 task-complete | `120/109/107/111/124` | recipe改变Procedure传递，absolute与margin都退化 |
| v6 old | `121/122/111/84/47` | 强时序差异可传到闭环，但absolute低且task旋转 |
| v6-fast task-complete | `143/135/125/128/129` | 历史最佳eligible single checkpoint；仍未达到151且视频margin弱 |

五臂顺序统一为`correct/same-task-other/cross-suite-wrong/shuffled/reversed`。v5.2与v6的recipe交叉结果
证明架构和训练配方强耦合：不能把“task-complete”或“old recipe”当成普遍好坏结论。

## 3. Cumulative intervention ledger

| 方法/干预 | 最强strict证据 | 实际证明 | 失败接口与保留结论 |
| --- | ---: | --- | --- |
| Action-Forecast v4 | best`109`; five-arm`109/104/99/148/126` | 视频顺序可显著改变LoRA/action | 学到absolute-time/action-phase shortcut，shuffle反而更好；时序敏感不等于正确理解 |
| Semantic Core + Procedure v5 | best`115`; `115/108/74/113/114` | task语义与有序Procedure可分离，wrong-video margin可形成 | Procedure差异在fusion/compiler后衰减，correct与shuffle/reverse等价 |
| v5.1/v5.2 | best`132` | language-axis Core、task-token evidence与Procedure能提高absolute或视频margin | 配方/架构耦合，不能只续训或按一个margin选方法 |
| v6-fast | best`143` | 高增益decoder和task-complete能达到历史最高absolute | 后续450/500/550/600=`131/130/132/126`，训练不是稳定累积 |
| v7/v8/v10 | 低于v6 | 更强内部Procedure/时序结构可以落到LoRA | 更漂亮内部因果不保证closed-loop，可能放大same-task demo nuisance |
| Loom/Recenter/Core/Prior | 均未过143 | patch correspondence、去DC、静/动态分解等接口可独立实现 | 均未解决policy-effective conditional credit；不再以confidence或单一fusion补丁重跑 |
| Target-Spectral | best`34` | 强制谱/秩形态会明显改变LoRA | “更均匀、更高秩”不是健康的同义词，可破坏q-dominant policy manifold |
| CV-ADR RAW/GROUP4 | best`117/110` | 更大、更coherent更新可构造 | video梯度主效应约`.1%`，query/flow variation和credit错位主导；大更新不等于好闭环 |
| Target-Bound | best`120` | remove-A/D与memory reversal 8/8达门，动态路径确实工作 | shared factor coexistence与checkpoint漂移仍失败；不能再把首因写成“视频未使用” |
| Semantic Factor-Basis | best`127`, union`193`, gap`66` | 一度形成更多共同能力 | 单checkpoint远低于union，核心现象是能力换手而非积累 |
| Variance-reduced estimator | best`126` | exact-Beta/antithetic可略稳gradient | held functional loss变好但closed-loop更差；flow MC方差不是主因 |
| Semantic Direction Store | best`129` | 独立store改善早期acquisition | 同分checkpoint breadth不同，Program→factor压缩与漂移仍在 |
| Policy-Target-Owned Factor | best`99` | 解除38-target共享改善跨层异质性 | action效果和absolute差；target ownership/健康几何不是充分条件 |
| Policy-Lane Hyperdecoder | best`70` | 形成约10条有效lanes和SFT量级专门化 | video BA能量约`.02%`，容量健康不能替代动态credit |
| Policy-Wide Atom Dictionary | best`80` | 64 atoms被广泛使用 | mixing/effective LoRA仍近rank1；不靠增atom/rank/正交loss救活 |
| Factorized Condition-Kernel | best`49` | full-rank stable kernel与跨video差异可形成 | LoRA约比direct SFT小200×、近identity；低增益decoder是局部瓶颈 |
| Few-Shot Invariant K4 | best`108` | K4置换、same/LOO/wrong/order路径可工作，能削弱单video偶然性 | full24 gradient retention约`.043`；few-shot不自动解决共享credit或正确时序 |
| K4 Policy-Layer Trace | best`99` | all-layer trace产生correct>wrong | 逐频单位化放大低能DCT高频约`140×`，reversal仍高 |
| Energy-Preserving Trace | best`85` | 修复真实频率能量比例 | correct/wrong从`99/57`缩到`85/80`；能量保真不等于语义保真 |
| Evidence-Factorized Trace | best`84` | trace→BA→action闭合且correct>wrong | shared Reader retention约`.05`，参数隔离仍非答案 |
| Sparse Semantic Expert | best`78` | expert-local retention提高 | language route固定owner，wrong/order更成功；language-only ownership不足 |
| Grounded-Video Expert | best`88` | video route、Reader、BA、action与rank均material | correct无margin且task轮换；视频敏感+隔离仍不充分 |
| K4 Phase-Aligned v6 | best`108`, reversed`121` | 视频未被忽略 | 近rank1、高能量、program retention约`.04`；phase alignment不足 |
| AS125 + semantic-progress RL | `97→104→102` | failure trajectory可提供非零action-free credit | breadth下降、继续训练换手；reward信号存在但共享更新不稳 |
| Program-Credit RL | `106` | CRN与Program gradient可到达 | task cotangent近正交却被shared condition map压成common update |
| SFT-Anchored Tangent Basis | `143→142` | 强warm-start上小幅reward update可运行 | gained/lost=`20/21`，没有净提升；保持分数不能冒充Writer改进 |
| task experts step2000 | train`658/1200`, 23/24非零 | task-local SFT LoRA是policy-effective task-level target | task9仍0，且不提供held泛化、视频特异性或时序证据 |
| addressless Expert-Manifold | `48/400` | raw-expert reconstruction可训到SFT量级norm | decoder后topology identity坍缩，nearest expert cosine约`.008` |
| topology-address binding | `75/400` | 静态chunk/rank地址可调制video dynamic value并进闭环 | 输出仍task-common、absolute低；只调address不够 |
| Causal Barycentric | `63/400` | temporal coefficients和raw-factor组合可运行 | raw A/B组合有`k≠j` cross terms，不保持effective update |
| policy-effective soft/hard bank | `15/80` / `3/80` | hard compiler可近精确复现所选expert | 当前24-expert deployment dictionary无held support；不外推所有流形方法 |
| v6-prior whole-LoRA objective | `134→127→105→123` | 冻结上游、只训写出端可高吞吐运行 | norm/方向吸引主要径向收缩，macro0仍最佳；objective退役 |
| Expert-Component Projection | `134→133→120` | `a_correct`与expert component按构造提高 | 正交漂移增大，macro25 net`-14`, p=`.038477`；不续/不扫权重 |
| Condition-Local Tangent Tube | `134→131` | relative tube中位`.01390/.01408`，半径约束工作 | direction ratio`108.93/126.88`、completion`0/24`，只压小未旋正 |
| Expert-Flow Teacher Audit | no rollout | gradient residual`.6864/.8387`，expert方向非冗余 | flow loss仅`2/24` tasks、`0/4` suites优于baseline；CEFD否决 |
| Frozen-v6 residual v1 | no rollout | correct retention`.807966`与A/B/action closure成立 | DC key condition`1315.33`、null15/24；不训练、不调lambda/seed/P |
| Balanced DC-Causal v2 | `134/140/139`, union`153` | 13/13机制门、24/24 null、部署/吞吐闭合 | 10→25=`12/13`换手；50-video correction近随机正交，blind-add退役 |
| Exact Anchored Reconciliation | `134→140` | RLS/历史row保留机制可运行 | full400 lost15，correct80误导；offline row保留不保护held occupancy |
| Reward-Credit Program Cotangent | cycle1`134`, `14/14`换手 | on-policy reward可形成有内容Program与continuous tangent | q/v约`1e-8 RMS`运动低于非零BF16 factor约`1e-4` ULP；不续cycle2 |
| Q/V uniform pivot-rank14 | online`128`; compiler-only`138` | 去混杂后可分离compression与regeneration影响 | old→compiler`119/19/15`，compiler→online`115/13/23`；两者独立换手，统一rank14退役 |
| Policy-Innovation Consensus Key | no rollout | raw same/order、full48 correct/null、Program→LoRA→action与吞吐全部闭合 | exact full48 condition=`483.61515>200`；static common mode导致key collision，未获formal训练资格 |
| Policy-Innovation Goal-Causal Key | `138/400`, breadth6 | full48 condition修到`152.61`，FP32 Program与effective BA切向写出闭合 | macro0→macro10=`118/20/16`、churn36；blind offline source-action credit不覆盖held on-policy support，组合退役 |
| On-Policy Success-Guarded Program Credit | no rollout | success-prefix continuous guard的canonical执行图与fresh schema可实现 | world6 rank-local长尾触发600s NCCL watchdog，wall至少matched `1.912x>1.25x`；只淘汰current replay/VJP graph |
| Success-Key Nullspace Consolidation | `137/400`, breadth7 | 4/4 success key可在healthy rank/energy下硬保护完整conditioned Program/LoRA/action | old134→SKNC=`121/16/13`、churn29；train24 single-video key不外推held support，blind B20继续suite换手 |
| Negative-Preserving Candidate Guard | `135/400`, breadth5 | actual candidate与negative response可在同一affine correction中闭合 | old134→NPCG=`117/18/17`；point guard未保护held occupancy，约束健康不等于共同积累 |
| Cross-Video Equivariant Candidate Guard | directional `131/400`, breadth6 | single companion E与negative/guard可数值闭合 | K2 outcome经hard equality放大低位差异；hard E又使NPCG→CVEG=`114/17/21`，稳定性与absolute均non-pass |
| Magnitude-Gated Causal Interaction Joint Credit | `134/400`, breadth6 | full96 condition修到`174.813`，paired video、negative与Program→BA→action全链路闭合 | old134→MGCI=`114/20/20`；key谱修复未改变blind offline credit的held task换手 |
| Dynamic-K Backbone-Memory rank8 | `100/400`, breadth4 | 真实图文+Action probes中8 memory tokens、动态K与完整rank8 LoRA图可训练部署 | 删除absolute Semantic Core后任务集中；task mean BA offdiag`.702`，不resume |
| Dynamic-K Semantic-Address rank8 | `101/400`, breadth6 | absolute Core只作temporal Query address，constant/static bypass仍为零 | old134→semantic=`82/19/52`；query address不足，task mean BA更同向到`.776` |
| Semantic mapper stage probe | no rollout; 32 correct + 128 controls | M2P/final/shared-project task offdiag`.492/.529/.530`，视频与顺序结构健康 | family hidden/B升到`.634/.779`；下一可证伪接口是nonlinear family readout，不是继续堆前端 |
| Dynamic-K Direct-Family-B rank8 | K1/K4=`102/98`, breadth5 | direct readout与nested动态K可稳定部署；K4把same-task方差约降`6.3x` | K1→K4=`80/18/22`且无新task，set稳定错误task mean；mapper/K/set小修全部退役 |
| Task-Grounded Visual-Value rank8 | `88/86/86/96`, breadth5/6/6/6 | exact language查询raw patch Value能进入有向D/G；action norm恢复到old134约`.98x` | macro200 top3占95.83%，150→200仍churn40；完整曲线终局non-pass |
| Fixed-A reachable-subspace diagnostic | no rollout；old134 fixed/optimal rank8=`.01950/.999999` | rank8容量足够表示已知强BA，但当前随机固定A只开放极窄右子空间 | train24最优共享A在experts保留`.94063`、到old134 held仅`.06811`；只支持后续task/video-conditioned A候选，不支持静态expert basis或性能claim |
| Task-Grounded Full-Factor rank8 | `91/400`, breadth5 | 完整dynamic A/B可训练且相对matched fixed-A净增3 | B norm仅`.062x`、BA norm`.245x`且近正交；offline loss接受弱重参数化，当前前端/mapper组合终局non-pass |
| V6 Dynamic Slot-Set Bridge | K4 `130/400`, breadth6 | same-task BA方差降约`9.26x`且基本保留old134支持 | post-compiler set使task mean K1→K4 cosine`.999832`，只稳定nuisance；不续训/调K |
| V6 Shared-Core Procedure-Set Bridge | K4 `139/400`, breadth6 | 相对post-compiler130净`+9`；更早Core union改善absolute且same-task方差降约`9.69x` | 相对old134只净`+5`且增益集中Long1；trained Procedure-Set仅`.000918` BA改写，Goal3/Long2仍0，按门终止 |
| V6 Semantic-Core Set Bridge | K4 `135/400`, breadth7 | Long2从0到1；trainable set从语言对齐Core进入native policy拓扑 | 相对matched139=`120/15/19`净`-4`；attention entropy`.999885`、Core correction仅`1.83e-5`，centered Value近均匀时相消，按门终止 |
| V6 Semantic-Core Common-Value Set Bridge | K4 `133/400`, breadth6 | Core correction/effective-BA改写打开到`.065856/.053648`；train-seen trained/zero=`63/59` | 相对135=`118/15/17`、相对139=`119/14/20`；少量task-local credit未形成held可组合程序，Long净丢7 |
| V6 Shared-Core Ordered-Procedure Common-Value | K4 `139/400`, breadth6 | 有向Procedure correction/effective-BA打开到`.09601/.01397`，保持matched139 absolute | 相对matched=`120/19/19`；train-seen trained/zero=`64/64`、`4/4`换手，B20 credit在train/held均无净on-policy收益 |
| V6 Ordered-Procedure On-Policy Preference | K4 `138/400`, breadth7 | train24 reward形成18条paired新success，q/k/output、BA与action全链路非零 | 相对同schedule AS139=`120/18/19`、churn37；Spatial净+4但Long1净-7，最终shared update未保住support |
| V6 Actual-Delta Success-Support Projection | K4 `138/400`, breadth7 | 22条train24成功task约束中raw违反6条；投影后0 violation并保留`.964/.981` descent/energy | 相对AS139=`116/22/23`、churn45；相对raw138=`117/21/21`，Long净+6由Spatial/Object净-6支付，一阶local support不代表held共存 |
| V6 Layerwise Action-Probe Conditioned Procedure Reader | K4 `143/400`, breadth7 | 同一次真实context forward的18层probe carrier具有reverse/static差异；保留AS139强图并追平历史absolute | 相对AS139=`120/23/19`、churn42；BA只改`.002653`且Goal3高coherence仍0，冻结Procedure commitment加blind B20 credit只形成AS139邻域换手 |
| V6-LPCP Paired Causal Success Distillation | K4 `135/400`, breadth6 | 48 paired states产生9条唯一成功轨迹，positive CFM使query、BA与action全链路非零 | 相对LPCP143=`121/14/22`、churn36；PCSD/LPCP BA只改`.0006834`，FP64同task跨K4 video-set增量cosine约0、mean energy仅`.2486`，稀疏reward credit未合并成共享程序 |
| V6-LPCP Cross-Video Causal Success Distillation | K4 `134/400`, breadth7 | 同一成功trajectory在4个disjoint correct K4下完整反传，36/36 view gradients非零且仅`1.0307x` PCSD wall | 相对LPCP=`122/12/21`、四suite全降；FP64四view部署增量cosine`.000205`、energy`.250155`，exact cross-video credit仍未越过query-only commitment |
| V6-LPCP Semantic Factor-Memory Commitment | K4 `144/400`, breadth7 | K-set后layer/rank innovation memory可在8个factor families的冻结V6 output basis前获得四view reward credit；8/8 maps更新且cycle仅`1.0662x` CV-CSD | 相对LPCP=`128/16/15`、churn31；稳定FP64 BA改写仅`2.899e-7`且q/v/action非零样本=`249/16/1`，router未学成、native ULP crossing未形成跨video共同方向 |
| V6-LPCP Gradient-Open Semantic Commitment | K4 `141/400`, breadth7 | W1 anchors使semantic router首步打开，BA改写较SFMC放大`33.3x`且q/v/action非零覆盖=`400/399/368` | 相对LPCP=`128/13/15`、churn28，suite=`-1/-6/-2/+7`；first4增量cosine`.000144`、energy`.250124`，写出打开后仍是跨video近正交与task换手 |
| V6-LPCP Causal Coefficient Transport | K4 `142/400`, breadth6 | train-seen task4纯CCT增量被限制到共同方向，corrected cosine/energy=`.575776/.681821`；formal工程完整 | 相对LPCP=`125/17/18`、churn35；held first4约`0/.25`，hidden只缩小1.7x而BA缩小249.92x，held native compiler commitment失败 |
| V6-LPCP Native Probe-Value Commitment | K4 `136/400`, breadth6 | validation8 preformal cosine/energy=`.44940/.57150`且held/train BA L2=`.7525x`，关闭CCT held compiler消失 | 相对LPCP=`120/16/23`、churn39；full24后task4从`.5929/.6792`坍塌到`.0569/.2951`，gained/lost改写不可分 |
| V6-LPCP Pre-Addressed Factor-Selective Native Value | no strict; mechanism stop | task4八family/q-v-action/reverse/wall健康，factor可独立选择native Value | train24 address effective rank=`2.1575`，validation8仅`.1681/.3729`、3/8过门，未full24 |
| V6-LPCP Shared Joint Native-Value Gate | no strict; mechanism stop | task4 `.47227/.59781`，reverse/static与reward链健康；gate/hidden held cosine约`.94` | frozen W2后raw factor仅`.02135/.26592`、action`.00267`，validation8 BA `.20190/.39645`、2/8过门，未full24 |

ADSP、V6-LPCP、PCSD、CV-CSD、SFMC、Gradient-Open与CCT均已按各自预注册门终局。ADSP authority=
`docs/action_forecast_writer_v6_ordered_procedure_final_shared_support_projection_design.md`，只否决实际检验的
train24 success-prefix一阶约束。V6-LPCP authority=
`docs/action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md`：它证明layerwise有序carrier可以在
不损伤V6 absolute的前提下接通，但只经冻结Procedure Query和blind functional credit时仍是23 gains/19 losses，
没有越过150。PCSD authority=`docs/action_forecast_writer_v6_lpcp_paired_causal_success_distillation_design.md`：
它只否定query-only一轮稀疏positive CFM足以合并same-task跨video credit。CV-CSD authority=
`docs/action_forecast_writer_v6_lpcp_cross_video_causal_success_distillation_design.md`：它进一步否定四个correct K4
conditions的exact selected-success gradient mean足以越过同一query-only map。SFMC authority=
`docs/action_forecast_writer_v6_lpcp_semantic_factor_memory_commitment_design.md`：它进一步否定当前zero-init
family-hidden memory residual在单cycle内足以跨过native factor量化边界并形成learned semantic route。这些路线均不
否定V6、literal memory token、few-shot、rank8或其它架构级policy-aligned生成方式；下一轮不得继续加view、
constraint/LR/scale/rank小修，也不得原样恢复历史低分memory路线。

Gradient-Open随后只改变上述最早接口：用balanced冻结V6-W1 anchors让zero-init family maps与
semantic query首步同时获得credit。实现与分族response统计由`5b14c89`封存，full CPU=`396 passed`；task4 B8
真实smoke中semantic query delta=`1.1979e-4`（SFMC=`1.7564e-9`），8/8 maps更新，q/v/action native
effective-BA response=`6.6169e-7/9.1517e-7/4.8908e-8`，总BA为SFMC的`19.7x`，fixed-action仍为
`.0027033`。cycle=`132.458s`=`.9501x` SFMC。它关闭两个机制快速否决条件并获fresh full24 cycle1授权，
机制门只授权formal，不提供absolute、稳定性或视频因果结论。

fresh full24 cycle1随后由clean detached `eb543d3`在gpu01 world5完成：24 tasks/48 pairs/96 rollouts，10个
active tasks覆盖四suite；semantic query delta=`6.9499e-5`，5/5 probes的q/v、3/5的action native BA非零。
cycle=`581.924s`，完整world5 checkpoint/completion、禁读与数值门均通过。rank task count虽为
`3/5/2/5/9`，recorded wall max/min仅`1.2121x`，再次说明动态队列按cost而非数量平衡。训练内candidate/
reference=`33/31`不能跨world严格比较或预告held结果，后续结论只取同checkpoint strict400。

同checkpoint strict最终=`141/400`、breadth7、per-task=`1/3/48/29/0/36/23/1`；相对LPCP严格=
`128 retained / 13 gained / 15 lost`、churn28、net`-2`、Jaccard`.82051`，suite净变化=`-1/-6/-2/+7`；
相对SFMC=`124/17/20`、churn37。FP64 all400 BA relative-L2 mean=`9.6632e-6`，q/v/action非零样本=
`400/399/368`，证明gradient-open和native writeout真实成立；但same-task first4 cosine=`.0001442`、energy=
`.250124`且gained/lost幅度不可分。故本轮终局否定“共享language address + W1 anchors足以把不同正确视频的
success credit编译为共同task direction”，不否定memory token、few-shot、rank8或生成LoRA。下一变量必须在
compiler前直接形成跨video可复现的causal task Program，而不是继续放大当前residual或扫训练超参。

CCT随后只改变Program的自由度：每个policy/rank slot的视频memory不再决定任意256维Value，而只产生两个
coefficients，exact language与冻结V6-W1/GELU提供共享policy axes。clean `18bd363` world5 formal cycle1完整：
24 tasks/48 pairs/96 rollouts、candidate/reference=`33/32`、9 active tasks、cycle=`577.729s`。strict=
`142/400`、breadth6、per-task=`1/2/48/31/0/37/23/0`；相对LPCP143严格=`125 retained / 17 gained /
18 lost`、churn35、Jaccard`.78125`。breadth与retention门失败，故不续cycle2或六臂。

机制审计修正了旧v1 counterfactual标签：train task4纯CCT相对exact LPCP的four-view cosine/energy实际为
`.575776/.681821`，而不是旧文件中LPCP+CCT相对AS139的`.563803/.672852`；修正后局部门仍成立。但held
first4纯CCT增量约为`0/.25`。exact evaluator worker确认65,536个semantic-query元素逐元素完整加载。train→held
transported coefficient和pre-W2 hidden只缩小`1.63x/1.70x`，pure-CCT effective-BA L2却缩小`249.92x`。
所以CCT的最早失败接口是**held nonzero Program residual经native BF16 factor/compiler没有成为稳定BA**，不是
carrier、loader、reward或梯度。该结果只否定当前两系数CCT与一轮four-view selected-success，不否定V6/LPCP、
literal memory token、rank8、few-shot、reward credit或生成LoRA。

## 4. Final rank14 adjudication

immutable old full-rank macro0为`134/400`，per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
`0/5/48/34/0/35/11/1`。

online-regenerated rank14 root得到`128/400`、breadth7、per-task=
`1/1/47/29/0/36/13/1`，old→online retained/gained/lost=`113/15/21`。由于old/new使用18/12个
generator且旧调度在worker内局部拼B8，它是真实端到端non-pass，但不是干净compression反事实。

一次性compiler-only root：

`runs/outputs/pi05_v6_qv_rank_reserved_compiler_only_old134_to_rank14_correct400_20260811`

它从old exact cache做50×B8 q/v transform，action 1600 tensors与400 video identities exact，0 Writer/
teacher read/policy forward/update。strict=`138/400`、breadth7、per-task=`1/1/46/32/0/35/22/1`；
old→compiler retained/gained/lost=`119/19/15`、net`+4`、churn34。预注册lost上限是10，因此hard gate失败。
Long1净`+11`掩盖Spatial/Object净`-3/-4`；aggregate提高不是稳定共同积累。compiler→online又是
`115/13/23`、net`-10`，证明regeneration是第二个独立换手源。

正式状态：`original_gate_b_passed=false`、`counterfactual_gate_passed=false`、
`retroactively_changes_original_gate_b=false`、`authorizes_cycle1=false`。不能恢复Gate C、cycle1、
controls或rank14训练，也不能把该结果外推成“视频/Reward/continuous tangent整体无效”。

## 5. Task experts and few-shot

正式task-expert root：

`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`

24 tasks统一续到step2000；step250/500/1000/1500/2000 direct-expert closed-loop=
`432/557/624/638/658` of 1200，step2000为23/24 tasks非零。它们定义policy-effective task-level parameter
manifold，并提供SFT LoRA能量、rank坐标和跨target参考；但同一task的expert target对所有video恒定，所以不
包含same-task video差异或时间顺序。soft/hard bank held=`15/80`/`3/80`进一步否定直接部署字典。

K4实验说明few-shot能过滤单条示范的部分偶然低层细节、改善内部same/LOO稳定性，但未解决full24 shared
credit retention、正确顺序的policy-effective方向和single-checkpoint漂移。最新Dynamic-K nested K4又给出
同样但更强的定位：same-task effective-BA方差约降低`6.3x`，closed-loop却`102→98`且无新task。后续应按真实
最强设定报告one-shot、few-shot或scaling claim，不因K4失败否定多视频，也不靠增加K、挑video或平均LoRA制造
增益；优先修正被聚合的per-video高层证据与policy credit。

## 6. Stable cross-experiment cognition

1. **视频被使用不等于被正确使用。** hidden/LoRA/action对视频敏感仍可让wrong/shuffle/reverse更好。
2. **正常顺序有因果含义。** shuffled/reversed真实破坏展示顺序，模型不能靠原时间戳恢复；correct必须沿有效
   policy update胜过negative，而不只是把negative推坏。
3. **LoRA健康度是约束，不是目标。** 低能量、过度rank1可解释局部失败，但SFT量级能量、多lane、高秩和
   正确expert cosine都没有自动带来高closed-loop。
4. **functional surrogate长期错位。** loss、gradient consistency、reconstruction和MC方差改善可与closed-loop
   退化并存；checkpoint选择必须及时跑真实paired400。
5. **task drift有多个来源。** query/flow variation、full24正交抵消、shared parameter coexistence、
   Program→factor压缩、condition-map common update、compression和regeneration都只解释部分换手。
6. **架构与recipe耦合。** 新结果应与最接近历史架构的per-task成功集合比较，不能按aggregate 180度转向。
7. **small panel与union会误导。** correct80曾与full400给出相反保留结论；checkpoint union不能代表single model。
8. **新topology不能用held outcome设计。** target/rank/routing必须由train24机制和policy geometry推导，不能
   因某个validation task得失而手调。
9. **吞吐优先于低位复现。** 原生BF16/TF32、batch shape和reduction order的正常微差不是科学精度；不得为了
   `.001953125`级roundoff固定batch1、重复forward、扩dtype或做逐tensor/内容hash门禁。
10. **负结果只淘汰实际假设。** rank14失败淘汰uniform pivot14 support合同，不淘汰所有rank reservation；
    expert bank失败淘汰当前reader+24-expert字典，不淘汰所有task-level manifold监督。
11. **train success不是held support坐标。** 对单条train video完整LoRA做hard zero-motion仍可能在held video与
    initialization上lost13；support表示必须跨video/occupancy成立，blind nullspace写入本身不提供改善方向。

## 7. Do-not-repeat registry

- 不续任何已封存non-pass checkpoint，不扫仅为挽救单点的scale、seed、rank、dtype、dither或ULP参数。
- 不恢复language-only LoRA bypass、multi-video/LoRA/checkpoint平均、validation-task expert routing或held oracle。
- 不用80-row screen、training loss、held functional loss、LoRA norm/rank/cosine或union选择正式checkpoint。
- 不把强制正交、均匀能量、高stable rank、更多atom/lane/expert当成性能目标。
- 不把task expert reconstruction写成视频时序学习，也不把same-task恒定target写成video specificity。
- 不恢复旧SmolVLA、70/10/10 split、`pi05_libero`、flat task-local RL、progress proxy、SPSA或success-only replay。
- 不为普通BF16低位差异降低batch、显存利用或吞吐；不新增SHA-256/MD5或大量防御性校验。
- 不按held task得失选择mixed topology。pivot15+1或mixed `{16+0,14+2}`只可作为未来未授权问题；其中
  rank1 tangent capture历史仅约`.9185`，不能因多保留一列就直接授权Reward cycle1。

## 8. Historical detail retrieval

精确旧设计可通过下式读取，而不需要在active tree保留几十个互相矛盾的文件：

```bash
git show 3a6f801:docs/<historical-design>.md
git show 3a6f801:findings.md
git show 3a6f801:progress.md
```

保留的深证据文档是：

- `docs/action_forecast_writer_v4_root_cause.md`：错误时序shortcut的第一性原理定位；
- `docs/action_forecast_writer_v6_design.md`：历史143强基线的结构；
- `docs/action_forecast_writer_video_expert_manifold_design.md`：压缩后的task-expert/Expert-Manifold/Reward链；
- `docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`：最终rank14设计与闭环否决。

正式artifact不得因代码或文档退役而删除。若要追查精确root、per-episode row、manifest或旧命令，优先读
对应`runs/outputs/*/{run_contract,results,analysis,completion,evidence}.json`和上述Git快照。
