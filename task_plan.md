# EMBER Task Plan

状态：2026-08-21 **active；fit19 learner-state aggregation把两套held5投影从`44/250、44/250`提高到
`54/250、47/250`，证明closed-loop state bank有真实增量，但direct success retention仍只有`31.08%/27.78%`，成员
Jaccard也降到`.4028`，Gate 2继续失败。当前不进入Writer、不续做state-bank小扫，按专家挑战十二正式转入稳定
shared prior + task residual，并在rollout前合并为唯一complete LoRA。**

## Goal

全面落实第二轮独立专家意见与owner随后确认的授权：停止在当前LMMPC 320-cell joint-moving compiler上做增量修补，先建立
跨任务、policy-functional且坐标固定的adaptation manifold，再学习exact language与action-hidden有序视频到compact
skill code的推断，并以train/meta-task closed-loop credit改善一次性完整LoRA。所有方向必须进入实现、实验裁决或明确的
条件转向，不得静默遗漏；最终选择仍由held zero-interaction closed-loop absolute、breadth、相邻稳定性与视频净因果增量共同决定。

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
- [ ] 分析shared language/base prior + video residual是否提高code效率与功能保真；若采用，训练/部署前merge并验证唯一LoRA；
- [ ] decoder若在leave-task-out closed loop不能接近对应expert功能，先修manifold/data，不进入视频推断。

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

## Phase 3 — Language prior + video process posterior

- [x] 构建`z_L=f(language)`与`z_LV=z_L+delta(language, videos)`，同时保留video-only反事实；
- [x] 每条video独立编码initial relations、goal relations、contact/events、ordered subgoals、completion conditions与
  uncertainty；跨videos只在event/relation/code-evidence层置换不变聚合；
- [x] 不再平均50个Action tokens；保留flow-time/horizon-phase/noise-probe维，使用phase-specific policy response地址；
- [x] 接入frame-count-matched static-first-repeated与同episode eye-in-hand cross-view；确认现有HDF5无可用mask authority，
  不从teacher state派生mask绕过信息墙；
- [ ] 在train/meta actions可用处加入inverse dynamics、visual transition-to-latent-action与phase correspondence辅助目标，
  held输入继续action-hidden；
- [ ] Dynamic-K若进入论文候选，训练与formal评测真实覆盖各cardinality；不平均frames/raw features/final LoRAs；
- [ ] 固定decoder，仅训练language/video inference；按task-level split报告code与closed-loop泛化。

**Gate 3：** full video必须稳定优于learned language-only与first+final，收益不能只集中Object，Long不得系统性反向。

**Macro10裁决：** correct=`131/150`，language-only=`130`、video-only=`134`、first+final=`130`、reversed=`134`、
shuffled=`133`、static-first-repeated=`132`；correct相对source净`0`且所有order/static negatives都不差。video-only在
effective-update诊断中只对1/15 tasks最近邻到正确projected adapter，反而几乎等于共享均值carrier。因此本次Gate 3明确
失败；不续训旧Writer、不进入outer RL，先修复functional coordinate与task-conditioned inference可识别性。

## Phase 4 — Train/meta closed-loop outer objective

- [ ] 以Phase 3 functional/code distillation为warm start，保持decoder固定；
- [ ] 在train24/non-held meta tasks用simulator rollout reward或经验证progress critic优化video encoder/code posterior；
- [ ] 与matched functional-only、language-only、video-only比较，明确outer RL改变的是shared zero-interaction Writer；
- [ ] reward采样按task等权，保存可exact-resume的sampler/topology状态；held validation/test reward不产生梯度；
- [ ] 若需要stability regularization，约束policy-functional response或已验证成功support，不再用raw parameter gradient
  PCGrad作为默认解；
- [ ] task-local post-generation RL单独保留为扩展评测，绝不混入zero-interaction分数。

**Gate 4：** outer RL必须带来held closed-loop净收益与breadth/retention改善；train reward或critic loss不能选方法。

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
8. expert挑战十二的shared prior + residual现已正式进入实现：复用phase codes、successful/learner panels与held5固定rows，
   建立不会被task code改写的稳定shared behavior底座。为忠实实现专家的`Delta_shared + D(z)`而不在A/B相加时引入
   `BA`交叉项，rank16预注册为shared rank12与task residual rank4两个互斥rank块；两块在rank维拼成唯一完整LoRA。
   `configs/pi05_train24_stable_shared_prior_v1.json`与
   `configs/pi05_train24_shared_prior_residual_decoder_v1.json`分别冻结两阶段合同，并同时评测shared-only以防把carrier收益
   误记为task-conditioned能力；
9. 新参数化重新过Gate 2前不训练新Writer、不进入outer RL。旧macro10、七臂screen、fingerprints、expert banks、
   oracle和本轮训练产物全部复用，不重复昂贵训练；遇到阻塞继续先回查专家原始因果链。
