# EMBER Task Plan

状态：2026-08-21 **active；首个fixed-decoder outer-credit实现及其matched因果面板均已formal裁决。macro2的
correct/language-only/video-only/first+final/same-task-other分别为`41/39/40/39/40`，Goal与Long全为0；correct相对
language与first+final都只净增2且不显著，换同任务视频只保留36/41个correct successes。最早失效接口因此前移到outer之前的
过程推断。原计划在现有16维fixed code上补action-phase supervision；owner与执行方随后基于专家最新思想启动更根本的
policy-native双时间架构复核。候选方案与开放问题已写入
`docs/policy_native_dual_time_program_compiler_review_20260821.md`。复核完成前不启动原16维formal warm-start、新outer
estimator或held评测；现有实现、checkpoint与smoke只作为可复用资产和反事实。held始终零梯度、zero-interaction。**

owner于本节点要求：process-supervised路径完成机制验证并集成后暂时收尾，先进行owner/执行方/专家中期讨论；讨论前不启动
formal训练或下一轮held评测。goal保持active，暂停不等于路线完成或裁决。

## Goal

全面推进EMBER：从frozen source PI0.5出发，让shared Writer只根据exact language与K条action-hidden有序正确视频，在
rollout前一次生成一套完整task-conditioned LoRA，并从未见初始化闭环完成task。全面吸收第二轮专家意见、最新policy-native
思考与全部历史证据，不把compact code、fixed decoder、memory、rank或某个旧Writer写成目标；具体架构若被证据否决就按最早
失败接口重构。最终选择仍由held zero-interaction closed-loop absolute、breadth、相邻稳定性、same-task跨video鲁棒与视频
必要因果增量共同决定。

## Done when

- `docs/expert_second_round_implementation_ledger_20260819.md`中A--N及五个替代方向均不再是`queued`：核心路线已有实现和
  formal证据，条件方向已有触发条件、实际裁决或实施结果；
- non-held meta-task allowlist、语义去重、source/meta/architecture-validation/final-test边界与provenance可由仓库直接复核；
- 目标task的task-local/rank16 ceiling与source primitive coverage已被测量；唯一一次validation8 task-local oracle按owner
  授权使用validation actions更新彼此隔离的诊断LoRA，不更新共享模型；若source prior不足，只在授权train/meta数据上
  强化并重建可部署基线；
- 固定decoder在task-level leave-out上能把compact code解码为policy-effective完整38-target LoRA，而不是只在raw A/B
  L2或train-task free-Program上可达；
- learned language-only、video-only、language+video、first/final/endpoints/middle/order controls可以在同一推断与评测
  合同下比较，视频净增量不会由架构性identity开关自证；
- language prior + video posterior保留每视频内部有序过程与完整Action probe结构，并在声明Dynamic-K时真实覆盖各K；
- functional warm-start后的train/meta closed-loop outer objective已有严格对照，held部署仍只在rollout前生成一次完整LoRA；
- 每个正式候选均报告single-checkpoint paired400、per-task/suite、breadth@1/@5/@10、suite floor、retained/gained/lost、
  churn、相邻success-set重合、same-task不同视频以及process-identifying controls；
- 方法若通过则在多个task-level split复现并冻结后进入sealed held诊断和最终Test；若按预注册停止条件失败，则完成广义
  video-to-LoRA路线的证据化裁决并实际转入最有根据的替代问题，而不是继续小扫；
- 验证后的代码、配置、文档与remote-safe证据均已及时合并到`main`并推送，formal artifacts保留完整provenance。

## Fixed scientific boundaries

- deployment输入仍是exact task language与一条或多条same-task、action-hidden、内部有序teacher videos；rollout前只运行
  Writer一次，输出并部署唯一一套完整38-target LoRA；source policy冻结；
- validation/test actions或reward不训练共享模型。唯一梯度例外是已经预注册的一次validation8 task-local rank16 oracle：
  validation actions只更新八套彼此隔离的诊断LoRA到统一step2000，不选checkpoint、不进入Writer/decoder/source或部署；
  train24及显式allowlist中的non-held LIBERO-90 meta tasks可使用teacher action alignment、task-local experts和simulator
  reward训练共享表示、decoder与Writer；Test没有该例外；
- learned language-only adapter是必要baseline；它不取消“视频必须提供必要Value增量”的最终科学要求；
- shared language/base prior + video-conditioned residual是允许的，只要rollout前merge成唯一完整LoRA；不得部署第二adapter、
  held expert route、task-ID字典、挑video、checkpoint fusion或多LoRA平均；
- 除上述唯一validation8 oracle外，sealed held action/reward只可用于预注册、共享模型冻结、无梯度、无checkpoint选择的
  post-hoc诊断；Test默认留到方法冻结后；
- 当前LMMPC增量路线停止。可复用其严格评测、per-video/set数据流、Core/Procedure概念、Core-addressed Reader、bounded
  K-set/M2P等经验证机制，但不把旧320-cell grid或moving FactorHeads当成后继架构前提；
- closed-loop absolute首先选择方法；functional fingerprint、code reconstruction、norm/rank/cosine和内部margin只作定位；
- 一次尽量只改变一个主要因果变量；不以rank/LR/scale/seed/dtype小扫挽救明确坏结果，不添加hash、重复forward、
  大量防御性检查或为正常BF16/TF32低位差异牺牲吞吐；
- formal训练/评测来自clean pushed commit的detached frozen worktree。日常结构性开发在`codex/<topic>`隔离分支完成，
  验证后及时合并`main`并推送；
- launch前live检查gpu01/gpu02与独立存储quota。资源紧张时最多跨两个节点合计6张合适GPU；只有空闲资源明显充裕时
  才可合计最多8张。允许与低显存、低util进程安全共驻，不等待凑满、不占卡、不抢占或干预他人进程。

## Evidence ladder and decision gates

1. **Contract gate**：信息墙、task split、allowlist、pairing、single-LoRA和冻结source policy正确；失败只修合同。
2. **Functional realizability gate**：fixed decoder在未参与decoder拟合的meta tasks上复现expert的closed-loop功能；raw
   LoRA reconstruction不构成通过。
3. **Inference gate**：language/video可预测leave-task-out code，full video相对learned language-only与first+final有净增量；
   失败先定位source coverage、process representation或code inference，不解冻decoder追逐latent gauge。
4. **Outer-credit gate**：functional warm-start固定后，train/meta closed-loop objective相对matched functional-only提高
   held absolute/breadth/retention且不破坏process controls；否则淘汰该outer objective实现。
5. **Method qualification gate**：single-checkpoint paired400达到有意义absolute并同时满足breadth、suite floor、Long、
   same-video robustness、相邻稳定与视频因果资格；约145可构成有价值结果，150+仍需上述联合证据。
6. **Stop/redirect gate**：更强clean source prior、更多meta tasks、fixed functional decoder、process representation、outer
   closed-loop objective和多split评测都完成后仍`<=120`、top3贡献`>80%`、full不优于first+final、Long无收益、
   same-task换手且checkpoint漂移，则停止广义zero-interaction video-to-LoRA并转入ledger中证据最强的替代方向。

## Phase 0 — Successor contract、覆盖账本与现状基线（active）

- [x] owner确认train/meta experts、learned language baseline、outer RL、non-held LIBERO-90、sealed diagnostics与
  mergeable base+residual均被允许；
- [x] 从最新`main`建立`codex/functional-adaptation-successor`独立worktree；
- [x] 建立A--N及五个替代方向逐项ledger，登记依赖、触发条件、证据与状态；
- [x] 扫描现有`expert_manifold`、Writer、reward、evaluation和数据合同，标记可复用owner、退役路径与新模块边界；
- [x] 审计LIBERO-90与target40语义映射，产生显式non-held meta-task allowlist与去重证据，不复制大数据；
- [x] 定义task-level交叉验证split、process-control panel、learned language-only基线和source/task-expert ceiling协议；
- [x] 完成结构基线检查，冻结后继design与最小首轮实现范围，再进入Phase 1。

**Gate 0：** 数据边界、功能decoder训练/leave-out边界和每条专家意见的去向必须可审计；不能以“后续再考虑”代替状态。

## Phase 1 — 数据可识别性、ceiling与process controls

- [x] 在现有成功expert/adapter上建立统一policy-functional probe panel；flow、显式paired-noise的10-step denoised action
  response、exact action-sequence JVP与BDDL stage trace均已接通。8条固定non-held successful/on-policy trajectories全部
  复现成功，但直接串联八个progress strata时action/JVP只过`2/4`与`1/4` task；两类直接task-vector标签均未获升级，
  下一表示必须显式解决跨初始化phase/occupancy对齐；
- [x] 用train/meta leave-task-out测量task-local LoRA ceiling与source primitive coverage，分reach/grasp/place/open/toggle/
  multi-object sequence/recovery，区分“Writer不会推断”与“source不存在能力”；
- [x] 实现learned language-only、video-only、language+video共享评测面；no-video不得再由结构强制identity；
- [ ] 补first-only、final-only、first+final、endpoints-fixed-middle-shuffle、monotone-sparse、flow-only、static-only、
  robot/object mask、同endpoint异procedure、跨view、paraphrase、同对象异目标与stage success中当前数据可诚实支持的项；
- [x] 对暂时缺少配对数据的controls登记具体数据缺口并在non-held meta pool构造，而不是伪造对照；
- [x] 若source primitive/target ceiling不足，先强化clean source policy并重新冻结baseline；否则不无条件扩大source训练。

**Gate 1：** full video是否含超出语言与端点的可识别过程、source policy是否具有目标控制能力必须分开回答。

## Phase 2 — Policy-functional manifold与fixed decoder

- [x] 将完整PI0.5 flow decoder拟合收敛为单一formal/exact-resume入口，冻结56/15 task-equal schedule及下游authority门；
- [x] 在train24与新增non-held meta tasks训练/整理task-local successful adapters；每个任务允许多成功adapter用于估计
  功能等价类，不把raw A/B gauge当标签；
- [x] 已用统一probe panel形成71个expert的functional fingerprint，并在source未见的target train24上完成19/5
  role-disjoint flow/action fingerprints；同轨迹多checkpoint只带来极小aggregate改善。validation8 local ceiling已证明
  八项都存在明显policy-effective local update，而8-row面板证明同一adapter在不同successful occupancy上的直接响应仍
  可能比相似cross-task更远；train24多个成功checkpoint各自的完整成功trajectory与phase-aligned固定坐标已使held5达到
  `5/5` mutual-nearest并授权fresh Decoder；
- [x] 学习whitened/gauge-fixed compact code与`code -> complete LoRA` decoder，固定decoder后做task-level leave-out；
- [ ] 比较fully fixed与有明确two-timescale/EMA合同的decoder；默认主线为fully fixed，只有固定版明确欠拟合才启用慢更新；
- [x] 分析shared language/base prior + task residual：rank12 shared-only由source21提高到43，但rank4 earliest/latest
  composite降到37/33；唯一LoRA合同成立，当前residual implemented-fail，shared prior作为后续固定底座保留；
- [x] decoder在leave-task-out closed loop仍不能接近对应expert功能，因此没有把functional residual升级为视频Writer；下一轮
  不再继续相同loss/rank/seed小扫，而是独立改变outer credit。

**Gate 2：** 通过的是未见meta task的policy功能，不是train-task reconstruction、free Program或参数相似度。
default fold0还必须先证明uniform-step direct experts相对`646/750` source baseline产生有信息量的跨task功能增量；decoder
以paired source/direct/projected的retained、gained、lost、churn和per-task delta裁决，不能由identity/source高分过门。
若direct增量不足，当前source/meta任务重叠使Gate 2未识别，转入role-disjoint meta-task构造而不是继续训练Writer。

**Default fold0原裁决：** direct experts相对source在56/15任务面板分别净增`+247/+38`，因此当前pool具有信息量；
fixed-decoder projected为`2451/2800`与`659/750`，分别保留direct successes的`92.62%/90.79%`。train侧广泛净增，
held侧仅净`+13`且`p=.18208`、只复现54.67% direct gains，故登记为`qualified_pass_to_writer_inference`而非方法通过。
macro10推断后又确认train codebook为whitened、平均范数`5.589`，held free codes从零拟合后平均范数仅`.505`；两者不是
同一个可推断分布。原closed-loop结果仍是fixed decoder range证据，但不再作为Writer code泛化authority。下一Gate 2必须
用统一probe response、只在meta-train拟合的PCA/whitening同时变换train/held，不能再以独立held free-code优化过门。

**Unified fingerprint flow-only裁决：** 新codes的train/held std为`1.000/.7248`、平均norm为`5.570/4.144`，坐标修复
成立；固定code Decoder的held flow loss降至`.664218`，但closed loop为`644/750`，低于source `646`、direct `684`和旧
projected `659`。生成effective `BA`相对direct的relative-L2为`2.8576`、cosine`.0254`、norm ratio`2.7004`，证明
flow surrogate允许巨大off-manifold update。该路径登记为implemented-fail，不进入Writer；下一次只改为effective-update
probe objective，并用同一750 rows复验。shared-zero carrier并行完成matched对照，不作为fallback部署。

**Shared/probe/exact最终裁决：** shared-zero为`640/750`，task fingerprint仅在其上净增`+4`且`p=.68888`。固定8-probe
完整BA在train/held的cosine仅`.0642/.0449`。exact低秩Gram把该值提高到`.5365/.3032`，但closed loop只有
`638/750`，相对source净`-8`、相对direct净`-46`，并比shared-zero低2；held exact loss从step280的`.9258`到
step1120的`.9213`已平台化。无正则仿射full-BA上界诊断在held也只有relative-L2`.9797`、cosine`.3648`。因此本轮
Decoder objective family整体implemented-fail，不续训、不扫objective/LR/seed，也不进入Writer；回到专家原始的
功能等价类、多成功adapter与source/meta角色分离问题。

**Multi-success phase Decoder首轮裁决：** 5-rank Decoder完成950 task visits后，fit/held flow loss为
`.323930/.616152`；held5上source/direct-earliest/projected-earliest/direct-latest/projected-latest为
`21/74/44/108/44`。两套projected均相对source净`+23`，5 tasks不退化、3 tasks严格提高，成员Jaccard`.4667`；但只保留
direct successes的`.2703/.2593`与direct gains的`.1774/.2188`，未过`.75/.60`门。故这一fresh identity-centered、
successful-expert-state-only Decoder不能进入Writer。正向的source增量和成员稳定不允许把phase表示整体丢弃；下一主要
变量是专家原始方向A尚未覆盖的decoded-policy closed-loop state aggregation，之后才独立裁决shared prior + residual。

**Learner-state aggregation裁决：** fit19上30条projected trajectories绑定37个earliest/latest member targets，按每成员8个
真实learner states与successful panels严格1:1训练。learner-state mean loss从`.62903`降到`.15512`，held mean也从
`.61615`降到`.56098`；但同一held5 rows上的earliest/latest仅为`54/47`。earliest相对旧投影净`+10`，latest净`+3`，
说明occupancy覆盖不是无效变量；决定性门仍失败：direct success retention `.3108/.2778`、direct gain retention
`.1935/.2604`，latest只有1/5 task严格优于source，成员Jaccard`.4028<.44`。因此只关闭这一轮staged state aggregation，
不把它扩大为“occupancy无关”；下一主变量按专家原文改为显式稳定shared prior与受限task residual。

**Shared-prior residual裁决：** 为精确实现专家的`Delta_shared + D(z)`，rank16被分为互斥的shared12/task4，避免factor
相加产生`BA`交叉项；zero code逐tensor等于shared-only，rollout只部署一套complete LoRA。两阶段各完成912 task visits，
held functional mean由shared的`.68032`降到composite的`.65905`，但闭环结论相反：source/shared/composite-earliest/
composite-latest为`21/43/37/33`。source→shared为17 retained、26 gained、4 lost；shared→composite则为earliest
29 retained、8 gained、14 lost，latest 29/4/14。两套composite只保留direct successes的`.22973/.15741`与direct gains
的`.09677/.07292`。因此当前task residual失败，不能把shared carrier的43分写成task-conditioned能力；同时shared的净
`+22`证明专家挑战十二关于稳定行为底座的架构判断具有真实价值。下一轮固定这一区分，转查挑战十四的closed-loop外目标。

## Phase 3 — Language prior + video process posterior

- [x] 构建`z_L=f(language)`与`z_LV=z_L+delta(language, videos)`，同时保留video-only反事实；
- [x] 每条video独立编码initial relations、goal relations、contact/events、ordered subgoals、completion conditions与
  uncertainty；跨videos只在event/relation/code-evidence层置换不变聚合；
- [x] 不再平均50个Action tokens；保留flow-time/horizon-phase/noise-probe维，使用phase-specific policy response地址；
- [x] 接入frame-count-matched static-first-repeated与同episode eye-in-hand cross-view；确认现有HDF5无可用mask authority，
  不从teacher state派生mask绕过信息墙；
- [ ] 在train/meta actions可用处加入inverse dynamics、visual transition-to-latent-action与phase correspondence辅助目标，
  held输入继续action-hidden；56-task process frontend已接受跨episode action-phase监督，但新16维code heads尚未在fit19固定坐标
  上接受该监督；当前训练路径与1-task真实smoke已接通，formal结果等待中期讨论后再启动；
- [ ] Dynamic-K若进入论文候选，训练与formal评测真实覆盖各cardinality；不平均frames/raw features/final LoRAs；
- [x] 固定decoder、仅训练language/video inference并完成held5 closed-loop matched panel；当前两轮correct-only版本未过Gate 3，
  下一轮保持decoder不变，只补process/action supervision；

**Gate 3：** full video必须稳定优于learned language-only与first+final，收益不能只集中Object，Long不得系统性反向。

**Macro10裁决：** correct=`131/150`，language-only=`130`、video-only=`134`、first+final=`130`、reversed=`134`、
shuffled=`133`、static-first-repeated=`132`；correct相对source净`0`且所有order/static negatives都不差。video-only在
effective-update诊断中只对1/15 tasks最近邻到正确projected adapter，反而几乎等于共享均值carrier。因此本次Gate 3明确
失败；不续训旧Writer、不进入outer RL，先修复functional coordinate与task-conditioned inference可识别性。

**Macro2 fixed-coordinate裁决：** historical 56-task process Writer共有212 tensors/8,121,416 values成功迁移，只有
language/video/posterior三个`32→16`末层因shape变化重新初始化，旧decoder没有迁移。两轮fit19 correct-only后，held5
fixed250的correct/language-only/video-only/first+final/same-task-other=`41/39/40/39/40`，breadth均为`3/5`且Goal/Long
全0。language→correct为35 retained/6 gained/4 lost、`p=.75391`；first+final→correct为36/5/3、`p=.72656`；
correct→same-task-other为36 retained/4 gained/5 lost，correct-success retention仅`.87805`。因此本轮Gate 3仍失败，且
失败发生在outer之前。下一主要变量不是再换fusion，而是把专家方向I/G要求的跨episode action-phase与时序反事实直接作用到
新16维heads，并真实覆盖K1--4；该过程warm-start不过门则不启动更昂贵outer credit。

## Phase 4 — Train/meta closed-loop outer objective

- [x] 以已验证有闭环support的shared prior和fixed functional decoder为底座完成functional warm-start；当前Writer只推断
  task code，decoder/source/shared均冻结，没有把旧phase-code residual冒充通过的decoder；
- [x] 在fit19用simulator success、执行效率与BDDL goal-predicate progress完成一次closed-loop outer更新；held5只做
  zero-interaction rollout，没有reward/action梯度；
- [x] macro2 matched functional-only、learned language-only、video-only、first+final与same-task-other已在同一held5 rows完成；
  full相对language净`+2`、相对video净`+1`、相对first+final净`+2`且均不显著，未证明视频条件增量或过程价值；
- [x] reward采样按task等权，checkpoint保存sampler/cursor、RNG与world topology并已完成world6 exact-resume；held5不产生
  梯度；
- [ ] 若需要stability regularization，约束policy-functional response或已验证成功support，不再用raw parameter gradient
  PCGrad作为默认解；当前functional anchor没有阻止4个shared success rows丢失，不能记作通过；
- [x] task-local post-generation RL继续单独保留为扩展评测，本轮没有混入zero-interaction分数。

**Gate 4：** outer RL必须带来held closed-loop净收益与breadth/retention改善；train reward或critic loss不能选方法。

**首个Outer-credit裁决：** macro2 functional warm-start为`41/250`、breadth `3/5`；macro3一次outer更新后为`39/250`、
breadth仍`3/5`。macro2→3为37 retained/2 gained/4 lost、Jaccard`.86047`；shared→macro3为39 retained/0 gained/4 lost。
训练侧19 tasks中10项有非零advantage，但plus/minus都只有11次成功，held没有新增shared support，Goal与Long仍为0。因此按
Gate 4停止macro4，淘汰当前`one antithetic direction × two CRN rollouts/sign`实现；不外推否定更丰富outer estimator、
learned progress、composition或task-local RL。证据见
`docs/evidence/functional_adaptation_20260819/train24_functional_outer_credit_held5_20260821.json`。

**回溯后的接口裁决：** matched面板证明macro2在outer前已不满足Inference Gate，所以不能把macro3下降全部归因于credit
estimator，也不能直接投入更昂贵多方向outer。下一轮先使用已存在的process-control/action-alignment owner训练新16维输出坐标；
只有full稳定胜过language与first+final、same-video retention过门后，才为该warm-start接入结构不同的outer estimator。

## Phase 5 — Formal迭代、相邻稳定与多split复现

- [ ] 每个有信息量candidate在clean pushed commit的detached frozen worktree训练；机制smoke后尽快strict paired400；
- [ ] 报告base/language/video/L+V/process controls的严格matched矩阵，含per-task/suite、breadth、floor、集中度与Long；
- [ ] 好结果继续相邻checkpoint，检查same-task different-video retention达到至少90%--95%、Jaccard与suite floor；
- [ ] 首次达到约145且breadth合理时立即补全部因果controls和多split复现，不用继续训练掩盖峰值；
- [ ] 明确坏结果按最早失效接口转回对应phase，不做无关小扫；
- [ ] 只有architecture-validation完成并冻结方法后才进入final Test；sealed held诊断不回流模型选择。

## Phase 6 — 条件方向与最终收口

- [ ] 若静态一次编译在完整主线后触发stop gate，按证据选择并实际启动video->reward/progress、skill composition、
  object-centric plan、runtime video-conditioned policy、video-initialized task-local RL或observation-only offline imitation；
- [ ] richer sensing只有在明确改变纯RGB问题合同后才启动，结果与核心EMBER分开报告；
- [ ] 每条A--N和替代方向更新为`implemented-pass / implemented-fail / not-triggered-with-evidence / superseded`；
- [ ] 将最终设计、formal结果、停止边界和可复用机制写入`findings.md`、`progress.md`与`research_history.md`；
- [ ] 验证、合并`main`、推送远程、清理已集成worktree与临时输出；目标真实完成后再标记goal complete。

## Current next actions

当前先完成中期架构复核，而不是直接恢复GPU推进：

1. 由owner把`docs/policy_native_dual_time_program_compiler_review_20260821.md`及本仓库commit交给专家，要求其从核心
   可行性、PI0.5机制、Program监督/编译、数据可识别性与跨embodiment边界独立审查；执行方未经owner明确授权不向专家发送；
2. 专家意见与owner裁决后，只保留一个active design：若采用新方案，则明确替代本轮16维code主线并先做native lattice与
   compiler oracle gate；若否定，则记录根本原因并据此重构，不机械回到旧LMMPC/V6或继续小扫；
3. 复核期间不启动formal process warm-start、outer credit或held评测，不重训source/expert bank；已有smoke、checkpoint、
   successful trajectories、teacher-action store、controls和evaluator全部保留复用。

以下1--13项是本轮复核前已经完成的证据与原暂停点，不再自动授权第11--13项所述的16维formal路径：

1. validation8八套rank16 local LoRA均按预注册合同训练到step2000；strict paired400为`250/400`，source为`48/400`，
   43 retained、207 gained、5 lost、净`+202`，八项全为正增量、四suite均非零，故通过`>150 / >=5 tasks / 4 suites`
   强门并关闭“target local ceiling整体不足”的分支；
2. stage trace把Long失败进一步定位到多阶段完成：task1首对象ever `31/50`、第二对象ever `13/50`、最终全成`12/50`；
   task2第一阶段`50/50`、第二阶段与最终成功均`29/50`。该证据只作BDDL final-goal阶段代理，不冒充完整procedure；
3. `configs/pi05_successful_expert_occupancy_panel_v1.json`的8条轨迹全部原样复现成功；但直接串联八个progress strata后，
   denoised action只有task23/86通过same-task mutual-nearest门，即`2/4`，exact JVP只有task80通过，即`1/4`；全部早期
   selected states都尚未完成BDDL goal conjunction，失败不能归因于final predicate捷径；
4. 不把该负结果扩大成“action response无用”。它实际淘汰的是`单adapter + independently selected strata + direct
   concatenation`标签：task26的gained trajectory更接近task86，task80的retained trajectory也更接近task86，说明
   occupancy与phase nuisance仍压过task invariance；JVP因`1/4`不再作为primary label；
5. target train24多成功checkpoint/on-policy bank已完成：47/47条预注册轨迹复现成功；fit19-only、task/member/state
   等权的32维功能坐标解释`.92343`方差。held5中等时间与功能弧长均为`5/5`同task mutual-nearest，弧长相对等时间在
   `4/5`任务提高same-task cosine，超过预注册`>=4/5 + >=2/5`门；fit19为等时间`15/18`、弧长`14/18`，故不把正结果
   误写成“弧长全面优于基线”，真正通过的是`多成功checkpoint + 完整trajectory response + 固定坐标/显式phase`组合；
6. fresh fixed decoder已按预注册task-visits950完成：内部flow门通过，held5 earliest/latest均为`44/250`，相对source
   `21/250`各净`+23`，5/5不退化、3/5严格提高，Jaccard`.4667`；但direct success retention只有`.2703/.2593`，
   direct gain retention`.1774/.2188`，因此联合Gate 2明确失败，不进入Writer；
7. fit19 learner-state aggregation已完成30条唯一trajectories、37个member targets、每成员8个phase states、
   successful/learner严格1:1及6-rank 912 visits。learner-state loss显著下降，held earliest/latest从`44/44`变为`54/47`，
   但direct success retention仅`.3108/.2778`、direct gain retention`.1935/.2604`，Jaccard`.4028`；不续训、不扫state-bank；
8. expert挑战十二已完成：shared-only=`43/250`，相对source净`+22`且exact McNemar `p=5.95e-5`；earliest/latest
   composite=`37/33`，相对shared为8 gained/14 lost与4 gained/14 lost，即净`-6/-10`。成员Jaccard`.62791`虽稳定，仍不能
   挽救负增量；当前12+4 functional residual不进入Writer，shared prior只作为明确标注的固定底座保留；
9. fixed-decoder outer-credit合同已实现并exact-resume：macro2 warm-start=`41/250`；macro3一次outer更新=`39/250`，相对
   shared43没有新成功row。训练侧虽然10/19 tasks有非零advantage，held absolute、breadth与retention均未提高，故明确停止
   macro4；当前单方向antithetic estimator implemented-fail，不扩大为outer credit一般失败；
10. macro2 matched panel已完成：correct/language/video/first+final/same-task-other=`41/39/40/39/40`；full无显著增量、
    Goal/Long全0、same-task retention `.87805`，故最早失效接口在outer前的16维task/process inference；
11. 回查专家原文确认当前没有遗漏56-task process预训练：212 tensors已迁移，仅三个`32→16`code末层重新初始化，且这三个
    heads只接受两轮低学习率correct-only拟合。下一轮复用旧checkpoint、fixed decoder、fit19 videos/actions与现有control owner，
    用跨episode action-phase alignment、reversed/shuffled/first+final/endpoints-middle-shuffled和K1--4训练新heads；
12. 先以同一held5 correct/language/first+final/same-task panel裁决process warm-start。通过才实施结构不同的outer credit；失败则
    按专家stop gate核对尚未覆盖的meta-task数量/多split，并准备转向video→progress或skill composition，不围绕rank/LR/seed小扫。
13. process-supervised路径已复用单一loss owner接入跨episode actions与真实frame controls；gpu02单任务K1 reversed smoke得到
    action-alignment `.22331`、control-update `.002316`、finite total loss与grad norm `8.394`，峰值约32.49 GB。该证据只证明
    机制接通；按owner要求在formal launch前暂时收尾，待中期讨论。
