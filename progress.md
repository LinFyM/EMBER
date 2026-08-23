# EMBER Progress

更新时间：2026-08-24。稳定目标见`docs/current_owner_requirements.md`，active计划见`task_plan.md`，历史见
`docs/research_history.md`。

## Current authority

- current design contract：`docs/event_conditioned_policy_compiler_design.md`（专家最终修正已纳入，恢复执行）；
- ECP全过程对齐审计与专家最终裁决：`docs/ecp_expert_alignment_audit_20260824.md`第9节；
- completed falsification card：`docs/ecp_occupancy_complete_oracle_card_20260823.md`；
- completed capacity card：`docs/ecp_fixed_a_capacity_card_20260823.md`；
- completed mixed capacity card：`docs/ecp_rank4_residual_capacity_card_20260823.md`；
- completed realization card：`docs/ecp_mobile_rank4_solver_card_20260823.md`；
- completed reachability card：`docs/ecp_effective_update_solver_card_20260823.md`；
- formal adjudication：`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`；
- fixed-A adjudication：`docs/evidence/ecp_20260823/ecp_fixed_a_capacity_gate_20260823.json`；
- mobile-rank4 adjudication：`docs/evidence/ecp_20260823/ecp_mobile_rank4_residual_capacity_gate_20260823.json`；
- mobile-rank4 solver adjudication：`docs/evidence/ecp_20260823/ecp_mobile_rank4_solver_gate_20260823.json`；
- effective-update profile adjudication：`docs/evidence/ecp_20260823/ecp_effective_update_profile_gate_20260823.json`；
- completed GOMQ rank16 Phase 0 card：`docs/gomq_rank16_archival_card_20260824.md`；
- GOMQ rank16 Phase 0 adjudication：`docs/evidence/gomq_20260824/gomq_cycle2_effective_rank16_strict400.json`；
- completed process minimal-pair Gate card：`docs/ecp_process_minimal_pair_gate_20260824.md`；
- process minimal-pair Gate A adjudication：
  `docs/evidence/ecp_20260824/ecp_process_minimal_pair_teacher_gate_20260824.json`；
- completed process phase-expert Gate A2 card：`docs/ecp_process_phase_expert_teacher_gate_20260824.md`；
- process phase-expert Gate A2 adjudication：
  `docs/evidence/ecp_20260824/ecp_process_phase_expert_teacher_gate_20260824.json`；
- completed Phase 2A card：`docs/ecp_effect_path_calibration_card_20260824.md`；
- Phase 2A effect-path adjudication：
  `docs/evidence/ecp_20260824/ecp_effect_path_calibration_gate_20260824.json`；
- completed Phase 2B/2C card：`docs/ecp_fixed_effect_realizer_card_20260824.md`；
- Phase 2B/2C fold0 adjudication：
  `docs/evidence/ecp_20260824/ecp_fixed_effect_realizer_fold0_gate_20260824.json`；
- completed centered two-sided coordinate fallback card：
  `docs/ecp_centered_two_sided_coordinate_card_20260824.md`；
- centered two-sided coordinate adjudication：
  `docs/evidence/ecp_20260824/ecp_centered_two_sided_coordinate_gate_20260824.json`；
- Phase 2B formal particle authority：
  `runs/analysis/ecp_fixed_effect_particles_565c055_gpu01p123457_20260824/manifest.json`；
- Phase 2B fold0 fixed-code authority：
  `runs/analysis/ecp_fixed_effect_code_fold0_e05ffca_gpu01p1_20260824/manifest.json`；
- Phase 2C fold0 formal training authority：
  `runs/outputs/pi05_ecp_fixed_effect_realizer_fold0_e05ffca_gpu01p1_20260824/`；
- active goal：完整实现并验证EMBER-ECP；goal仍在进行中；
- canonical workspace：本仓库`main`；最新formal evaluation authority为clean pushed `24c5bdc`。当前没有active GPU job。
  process Gate A/A2、balanced-SVD learned realizer fold0与centered two-sided coordinate Gate均为non-pass；fold1、Gate B、
  process suite、fresh Program、`q_pi/q_V`及joint Writer均未启动。process acquisition与shared-realizer两侧均已到专家裁决点。

## Current scientific state

- ECP核心假设未被证伪；native Stage 0 v3只作为candidate observer。它通过non-degeneracy与task separation，但未完全通过
  process semantics和probe invariance。现有数据只支持scene/goal/order claim。
- 历史Action Meta matched结果中性，现保留为control而非默认authority。
- 旧deterministic/mean `q_pi -> Program -> A/B hyperdecoder`家族已经关闭。v24、MDCO以及width/rank/head/fusion/LR/seed
  后继均不再恢复；这个裁决不等于专家最终要求的distributional `q_pi(P)`已经实现或被否定。
- four-category structured occupancy oracle已经补齐PECS缺少的initial/successful/candidate/recovery四类state support以及独立成功策略。它从
  stable carrier `43/250`提高到`78/250`，说明realization仍有真实增量；但breadth只有3/5，Goal/Long仍为0，完整门失败。
- Stage 1A-E的fold0 evidence prerequisite已部分完成：fold0五项各自的新seed37独立member在fixed250合计`113`，逐task为
  `26/32/37/13/5`；
  Goal/Long均有strict success，五个48-state、三particle effect banks完整。当前失败不能再归因于没有独立successful
  members或没有闭环occupancy，但retained source没有输出同构Program posterior的`q_pi(P)`，所以完整Stage 1A未通过。
- direct-effect realization子门为non-pass：final逐task`36/12/30/0/0`，只在2/5 tasks严格胜carrier；carrier retention为
  `35/43`，oracle-normalized recovery为`35/115=.304`且仅3/5 tasks为正。该solver直接读取effect bank，不读取Program，
  因而不能写成完整Stage 1B或Program compiler的负裁决。
- 当前没有`Program -> event/layer/family policy-effect distribution`模块，也没有`q_V(P|L,V)` checkpoint；普通Writer
  联合训练和最终结构化outer credit尚未开始。
- GOMQ历史151只作为“强carrier + 小有效更新可保留support”的结构证据，不恢复其Writer或checkpoint作为答案。
- GOMQ真实rank16 Phase 0已完成：确定性native-dtype canonicalization在strict400得到`136/400`、逐task
  `16/0/0/35/46/34/0/5`、breadth`5/8`。它相对历史rank32的`151`保留123、获得13、丢失28；因低于预注册145门，
  不成为absolute基线，历史151也只保留机制/历史证据。该差异不触发dtype、scale、seed、rank或checkpoint救援。
- 首个process-identifying pair的teacher Gate A已完成并non-pass：soup→butter为`0/50`、butter→soup为`19/50`，总计
  `19/100`，两方向首事件都为`50/50`，第二事件分别为`0/50、19/50`，invalid均为0。故失败接口是phase switch后的顺序组合
  支持，而非custom wrapper、wrong-first判定或第一primitive。Gate B未运行，不能从本结果裁决video order observer。
- phase-expert Gate A2已完成：现有task55/56 step1000 rank16 experts在各自primitive panel均为`50/50`，但组合后
  `soup -> butter=0/50`、`butter -> soup=44/50`、总计`44/100`。两向首事件均为`50/50`，100/100行phase/expert
  对齐，44条公开video全部通过信息墙。相对旧teacher的`0/19`净增25条证明expert切换真正生效；反向仍为0将
  失败定位到soup occupancy之后task56 butter primitive的恢复支持，而非adapter加载、phase route、第一事件或wrapper。
- fixed effect realizer的held-only materialization与fold0 Gate已完成：step800/1000 strict250为`33/37`，逐global分别为
  `32/0/1/0/0、36/0/1/0/0`；两者均低于carrier `43=38/1/4/0/0`，breadth均为`2/5`且Goal/Long均为0。
  step1000相对carrier只保留/新增/丢失`33/4/10`，相对direct-latest `108`只保留19个成功，因此不是near-pass。
- 两节点评测完成后的post-hoc定位显示：held latest在fit-only `512 -> 128`坐标中仍保留`79.1%--89.2%`中心化response
  energy，cosine为`.889--.944`；但step1000生成的Goal/Long residual effective energy仅为known target的`7.1%/22.9%`。
  最早失败由此定位为cross-task effect-code-to-residual mapping，而非rank4容量、known correction不可达或单独PCA压缩。
  按当时仅检查input-PCA的判据没有触发fallback；本模型仍不续训、不扫width/LR/seed/head，也不进入fold1。
- 进一步分解output target后发现，fit residual的task-equal expected energy为`94.1161`，其中shared mean为`89.1989`
  （`94.775%`）；当前prediction与held target去掉该mean后的innovation cosine仅`.0012--.0573`。这比input PCA解释更早：
  absolute A/B/effective loss被共享更新主导，没有学习低能量但闭环关键的task innovation。
- fixed width8 two-sided coordinate已完成唯一strict250：`80=24/10/46/0/0`、breadth3，低于83门且Goal/Long为0；只保留
  carrier `23/43`和known-latest `59/110` successes。250行相对两套reference的episode/env/policy/language/noise pairing均
  零mismatch。尽管重建update cosine仍为`.877--.960`且Object达到46，fit-span投影没有保住task-specific closed-loop support。
  当前coordinate停止，不训练centered-innovation realizer、不启动fold1、不扫probe/width/rank/threshold。
- Gate后零新forward/rollout定位显示，exact earliest→latest只增加`1.3%--10.8%` update energy，却把mobile projection从76
  提高到110；fit90 span对该innovation的cosine仅`.318--.647`。q_proj占总能量`91.3%--92.6%`，aggregate cosine遮蔽了
  低能量行为方向。combined basis已达到108 members中心化rank上限107；同数据加width无效，下一实质选择是新增span外
  source-unseen/process-diverse mappings，或停止当前shared-realizer family。
- 重新阅读专家最终复核后确认：fixed-A只是一种carrier-preserving realization候选，不是ECP核心硬约束；必须先把它与
  effect objective/calibration分离，不能继续把二者打包成新版本盲目迭代。
- fixed-A容量现已被直接闭环分离：三个成功members的解析最优投影只得到`49/41/35`，合计matched retention
  `67/295=22.71%`；Goal与Long三个members全部0。当前fixed-A row space停止作为主线。
- stable carrier的38个targets都是精确rank12且后4个B columns为0；最佳任意row/column-space rank4 correction在15个matched
  member-task上覆盖`99.49%--99.69%`所需修正能量。它成为下一次解析闭环容量问题，不直接视为正结果。
- mobile-rank4 strict250为`110/120/76`，全部5/5 tasks非零并逐arm略高于direct；pooled matched retention为83.05%，但
  Long同member retention只有36.36%，故预注册裁决为mixed而非supported。Long union retention为54.55%只作失败定位。
- 依据专家“multiple successful members的union进入Gate、exact row只作辅助”的原始意见，旧mixed不改名，但另行授权一次
  单变量mobile-rank4 realization oracle；bank、objective、12-step数值与原完整closed-loop Gate全部冻结。
- 该oracle正式只有`49/250`，逐task`40/3/6/0/0`；相对carrier保留41、获得8、丢失2，但member-union recoverable gap只
  恢复`3/115=.0261`且仅2/5 tasks为正。它是明确non-pass，不补step10/11、不扫参数、不授权Stage 1C。
- 同一objective在真实successful-member responses上存在`.060--.163`的低值，而当前solver final仍为`1.915--3.262`；
  known-success projections的trust为`1.341--2.281`，当前只有`.000915--.001171`，且effective correction方向cosine仅
  `.041--.077`。因此当前最早失败进一步收窄为zero-residual raw-factor动力学未到达successful effect basin；effect target
  是否充分仍未被最终回答。
- effective-update successor已在ordinal71判为Profile non-pass：initial精确等于carrier，matrix-free sketch的方向导数为
  `-81.8873`且全部finite，但五个固定回溯尺度均未被完整objective接受；final objective仍为`2.214329`、gap recovery与trust
  都为0。它只运行4次initial VJP，未进入后续Gram tangent，也未启动held5。
- Phase 2A effect-path calibration已通过：15/15 known-success balanced-SVD paths的matching verified loss、
  global-particle loss与legacy stage-wise loss都从carrier到endpoint严格单调下降；15/15在`alpha=1/8`已改善，
  15/15最低点在`alpha>=3/4`，5/5 tasks均改善且Goal/Long非例外。这授权fixed deterministic-sign
  balanced-SVD rank4 coordinate，也证明旧solver失败是未到达已存在的successful basin，但不证明realizer、
  distributional `q_pi`或video inference已成立。

## Verified reusable assets

- source policy、normalization、tokenizer与fixed evaluation assets保持canonical，不重训；
- fit19 stable shared carrier：
  `runs/outputs/pi05_train24_stable_shared_prior_formal_r6_v48_e948fca_gpu02p123467_20260821/shared_prior.safetensors`，
  held5 fixed250为43；
- fold0 held5 source/direct-earliest/direct-latest为`21/74/108`；
- 新独立members为`113/250`；earliest/latest/independent逐row success union为`146/250`，逐task
  `38/40/41/16/11`；
- 24-task task-expert bank、三套member成功occupancy、五个48-state effect banks及五套final LoRA完整保留；
- Stage 1 authority含95 tasks、118 successful members与`[118,8,32]` phase response；
- PECS local/trajectory adapters及paired rows完整保留，可作为candidate occupancy生成policy；
- validation8 sealed task-local rank16 oracle为250/400，只作ceiling evidence。

## Corrections now applied

- 不再把48-state effect bank称为完整Stage 1A pass：它是`q_pi(P)`所需的evidence输入，当前没有Program posterior；
- 不再把direct privileged effect solver称为完整Stage 1B：它是绕过Program的lower-level realization子门；
- 不再由该子门non-pass直接推导“全部shared compiler/video inference被证伪”；完整privileged Program链尚未实现；
- process-identifying data保留为最终claim与数据资格缺口，但不再冒充已经证明的唯一根因或替代缺失Program桥；
- 当前不再把更多task数量等同于更多可识别信息：71个LIBERO-90 tasks全部被source见过，现成source-unseen mapping只有
  target train24；现有BDDL没有证明same-endpoint/different-required-process pair。
- held5只用于与`43/58/59/74/108`的预注册机制比较，没有用于训练shared模型或选择solver量纲。
- earliest/latest不再被冒充为独立lineages；五个不同seed、固定step2000的独立task experts已经补齐并全部产生strict success。
- oracle直接在PI0.5官方双相机rollout observations上查询source/carrier/三个expert particles，已经覆盖initial、successful、
  prior-candidate与recovery states。
- fixed-A历史realization从stable carrier出发只求Delta-B；后续mobile operator改为冻结carrier12并共同更新residual4 A/B，
  两者都在effective-update层严格相加且zero correction精确返回carrier。
- 两次完整oracle都直接closed loop，没有geometry预筛；mobile operator重大失败后已停止，没有启动新的GPU版本。

## Completed execution

1. 五个独立experts固定step2000完成，fixed250及成功occupancy完成；
2. 五个48-state effect banks完成，每项保留initial8、successful24、candidate8、recovery8及三member轴；
3. fit ordinal71 profile完成，只把实现microbatch固定为4；12-step objective ratio为`.59779`，峰值18.94 GB；
4. 五个held solvers从clean pushed `c2aaac1`完成，objective ratio为
   `.5040/.5667/.5278/.6055/.4373`，均无trust penalty；
5. original fixed250 strict paired panel完成，250行相对source/carrier/direct/independent均无episode、seed、language或
   policy-noise common-prefix mismatch；
6. Gate 1B判为Realization non-pass并暂停，未补step10/11、未扫solver、未训练video predictor；
7. 从clean pushed `cc70aa6`解析生成15套fixed-A投影；gpu01 physical`1,2,3,4,5,7`并行完成三个strict250 arms，
   physical0 Prohibited未使用；formal results及paired gate完整；
8. 从clean pushed `083ed98`解析生成15套mobile-rank4投影并完成三个strict250 arms `110/120/76`，裁决mixed；
9. 从clean pushed `f75bafc`完成ordinal71 profile与held5五套mobile-rank4 solver LoRA；五项objective均严格下降；
10. gpu01 physical`1,6`完成最终strict250，physical0未使用；6 workers返回码全0，最终`49/250`，五套reference严格配对零错配；
11. 完成零新rollout的member-objective、trust、effective-correction方向审计，停止当前raw-factor operator。
12. 从clean pushed `fc678f3`在gpu01 physical5完成唯一ordinal71 effective-update profile；physical0未使用。initial sketch
    finite且为负方向，但0步被接受，按卡停止并取消held5 launch。
13. 从clean pushed `ac233fa`物化400套真实rank16 archival caches，并在gpu01 physical`1,2,3,4,5,7`以18 workers完成唯一
    strict400 correct评测；physical0未使用，72/72 shards与400/400 rows完整、workers全0返回。结果`136/400`，按预注册门关闭
    GOMQ；formal产物与paired evidence保留。
14. 从clean pushed `d1975c3`在gpu01 physical`2,3,4,5,6,7`以6 workers完成process pair的100行Gate A；两方向结果为
    `0/50、19/50`，未达到各20与总50的门。19条成功全在step275--385；旧collector中66条失败尾部达到401--404，不影响
    strict success，内层horizon边界已在`90090bf`修正且不重跑同一科学panel。
15. 从clean pushed `4cddcab`在gpu01 physical`1,2,3,4,5`完成15条known-success effect paths的formal
    calibration；5 workers全部返回0，15/15三种objective严格单调下降，5/5 tasks通过global gate，
    balanced-SVD rank4被冻结为Phase 2B coordinate。
16. 从clean pushed `565c055/e05ffca/0247a19/e806693`依次完成probe-preserving evidence、fold0 fit-only code、1000-step
    fixed realizer与两个strict250；step800/1000=`33/37`、breadth2且Goal/Long为0，按门停止learned realizer与fold1。
17. 从clean pushed detached `8aab214`建立fit90 centered two-sided coordinate并物化五套held latest single rank16 LoRA；gpu01
    physical`1,2,3,4,5,7`以12 workers完成唯一strict250，physical0未使用，36/36 shards与250 rows完整。结果`80/250`、
    breadth3、Goal/Long为0；carrier/known-latest retention仅`23/43、59/110`，全部Gate条款失败，successor未启动。
18. 从clean pushed detached `24c5bdc`在gpu01 physical`1,2,3,4,5,7`完成phase-expert teacher的100行Gate A2；
    6 workers全部返回0，100 ledgers、50对state/noise与44条public videos完整。结果`0/50、44/50`、总计
    `44/100`，未达双向与总量门；Gate B与后续process suite未启动。

## Completed fixed-A capacity diagnostic

- 三个successful members的零训练几何审计已完成：相对`expert-carrier`所需correction的energy coverage为
  `83.3%--96.7%`，但相对expert绝对effective update只有`41.5%--62.7%`；
- Goal的latest/independent coverage最低（约`41.5%--42.0%`），但Long反而最高（约`59.2%--59.6%`），所以不能用
  row-space数值直接解释Goal/Long共同0分；
- latest/independent/earliest投影后分别为`49/41/35`，逐global0/9/18/25/36为
  `26/4/19/0/0、22/4/15/0/0、23/2/10/0/0`；
- 相对matched direct的retained/gained/lost分别为`31/18/77、22/19/91、14/21/60`；总体只保留67个、丢失228个，
  同时产生58个不同rows上的success，不能解释成纯粹分数缩放；
- 三个projected arms absolute合计125，比三次stable carrier panel的129还低4；Goal direct24与Long direct11全部丢失；
- 750行episode key、env seed、policy seed root、language和policy-noise common prefix均零mismatch，18个workers返回码全0；
- capacity-supported全部失败，overall、Goal、Long三条capacity-binding判据全部触发。

## Current implementation milestone

- retained Stage 1运行面由`Stage1EffectBank -> stage1_oracle -> fixed solver -> single rank16 LoRA`组成；bank字段为
  prefix/noise/category/stage/progress及source/carrier/member owner/flow/action。仓库没有当前`q_pi(P)`或Program-to-effect
  runtime，这一事实已在全过程审计中显式登记；
- 已接通真实PI0.5 official prefix cache与10-step denoising路径：在同一policy observation/noise上与official action输出的
  max-abs差约`0.00668`、RMS约`0.00208`，属于允许的BF16/batch reduction差异；
- native owner为`[batch,38,4,128]`，同时保留`[batch,10,50,32]` flow与`[batch,10,50,7]` integrated action；
- fixed-A路径只为38个target建立`Delta B`叶子；真实PI0.5 smoke中38/38梯度均finite且非零，峰值allocated约18.72GB；
- 已实现48-state effect bank、stage-consistent particle soft-min、carrier barrier、preservation/trust及统一12-step solver；
- profile没有借held5做量纲选择：另用ordinal71/global2独立member、四类occupancy和同构solver冻结实现合同；
- projection helper现可直接解析并行solver的per-task子目录，不再需要临时symlink surface；
- fixed-A analytic projection用低秩闭式解直接求`argmin_B ||B A_c - B_e A_e||_F`并输出single rank16 LoRA；focused
  realization/manifold tests为23/23通过；该已完成diagnostic入口现由Git保存；
- mobile-rank4 helper用thin-QR/core-SVD直接求`expert-carrier`的best-rank4 correction，再与不变carrier12按rank拼接；真实
  latest-task0资产得到76/76 finite tensors与`.99610/.98062` correction/expert coverage，focused tests为24/24通过；
- mobile solver canonical runtime只保留`carrier12 + residual4`一条路径；profile与五个held outputs均为76 tensors、finite、
  single rank16，focused tests为24/24通过。科学non-pass不是工程invalid。
- effective-update successor只保留一个solver入口：4次matrix-free sketch后最多8次Gram-preconditioned tangent VJP，所有候选
  通过固定objective-only trust回溯；解析参数化另归一个owner模块，raw-factor runtime/config已移除。focused ECP tests为
  `15/15`通过，profile gate显式核对exact carrier、负方向、finite、至少一步、gap recovery、trust与VJP预算；formal profile
  在initial backtracking处停止，未读取held结果。
- Phase 2B successor capture已实现原生probe-preserving response：旧路径仍显式求两个probe均值，新路径落盘
  `trajectory x sign x event8 x owner38 x horizon-mode4 x hidden128`粒子与canonical residual4 target。一个两轨迹
  member的真实GPU profile为18.72 GB peak，模型初始化后capture `5.25s`，输出1.91 MB，全部finite；
  正负probe response RMS差为`.05092`，证明新轴非重复。formal authority现已完成：118 members、95 tasks、
  188 successful trajectories和376 particles全部落盘，6 shards均0返回，capture耗时`83.79--87.40s/shard`，
  tensors合计195,744,144 bytes；未读validation/Test，未训练held参数。下一接口是fit-only owner-local
  `512 -> 128` code coordinate。
- Phase 2B/2C retained实现已接通fit-only owner-local PCA whitening、held transform-only code materialization、按task等权的
  in-memory batch，以及保持event/particle/owner结构的38-head fixed realizer与factor/effective/null losses；focused tests
  已增至`17/17`通过。
- fold0 code authority已完成：90 fit tasks/108 members拟合，5 held tasks/10 members transform-only；owner-local
  `512 -> 128`解释方差最低`.90695`、平均`.94106`，全部code finite。formal realizer训练只加载fit members，1000步
  `136.99s`，step800/1000 total loss为`.31668/.26605`，峰值1.90 GB；两checkpoint均已保存。loss下降不能替代held
  closed-loop。
- held-only materialization在`0247a19`生成step800/1000各五套single carrier12+residual4 rank16 LoRA；held target residual
  reads为0。step1000 invalidity screen为`8/50`，matching carrier为`9/50`，没有用于选checkpoint。
- clean pushed detached `e806693`同时完成两个strict250：step800使用gpu01 physical`1,2,3`，step1000使用`4,5,7`，
  各6 workers且全部返回0，physical0未使用；250行均完整并与carrier在episode/env/policy/language/noise common-prefix上
  零mismatch。结果分别为`33/250、37/250`，Phase 2C fold0正式non-pass。
- clean pushed detached `8aab214`的two-sided transform由90 fit tasks/108 members拟合，38/38 targets active rank均为107；
  held 10 members只做transform。五个latest update cosine为`.877--.960`；五套single rank16 LoRA在唯一strict250得到80，
  未达到任何性能/retention门。focused ECP tests为`29/29`通过；工程有效但科学non-pass。

## Current unresolved interface

- 最深缺口是deployment-time occupancy completion：Writer只见language/videos，不能在编译时读取future initial/candidate/
  recovery occupancy，却必须生成在该未见分布上有效的静态LoRA。task-local occupancy solver只作Stage 1B-R0 lower bound；
  deployment Stage 1B-C必须只读取Program；
- 时间上最早未证明的接口仍是video到process-semantic Program。native v3只证明非退化task/time表示；最终必须fresh建立
  owner-specific `P_lang/P_scene`并用process-identifying data排除task identity、relative phase与endpoint捷径；
- distributional `q_pi(P_visible,Z_robust)`与Program-to-effective-update bridge尚不存在。现有evidence panel可作输入资产，
  但antithetic probes已被平均、off-policy member response未验证、stage-wise soft-min可拼接不存在的policy；
- fixed-A capacity已被证明binding，mobile-rank4 topology则能解析恢复direct级闭环；当前失败不再是这两个容量问题；
- 当前zero-residual raw A/B solver从rank-zero correction奇点出发，12步后effective correction norm约为known-success
  projection的1%，方向cosine也只有`.041--.077`。它没有进入successful-member effect basin，不能靠放大LR或增加步数盲救；
- successful-member effect responses在当前objective上确有显著低值，因此effect target尚未被本结果否定；但“低值存在”也不等于
  任何由视频预测的effect distribution都会充分，后续必须把gauge-invariant realization与objective sufficiency分开检验；
- recovery occupancy是rollout-only privileged information，任何后续deployment Program仍不得读取；
- 现有数据仍不足以最终检验general process understanding；process-identifying source-unseen meta data仍是未来方法资格前置；
- effective-update profile已依合同停止：负的一阶草图没有在事前固定的最小`1/16`尺度产生可接受完整objective下降。这关闭的是
  当前sketch归一化、固定回溯网格与objective组合，不是mobile-rank4容量、任意更小局部步或ECP核心；后续不复活raw-factor
  dynamics，而用known-success corrections校准固定canonical residual coordinate与small deployment realizer；
- process-identifying data与realizer calibration现为并行前置。只读接口核查仍成立：现有LIBERO只把最终谓词合取作为
  `done`；若要形成同language/同终点/相反必需顺序的任务对，需要不向policy暴露状态的temporal environment wrapper。
  feasibility保留在`docs/ecp_process_identifying_meta_task_feasibility_20260823.md`；最小pair必须在新`q_pi/q_V`前完成，完整
  family-disjoint suite必须在最终Stage 0、`q_pi`与`q_V`共同训练前完成；
- 零GPU资产核查仍可复用：target task37三条保留成功轨迹都只有`soup -> cream cheese`一种顺序；non-held
  LIBERO-90 task55/56的正式source panels则均为`50/50`成功，成功步数median分别为`123.5`与`107`。首个pair因此可优先用privileged phase
  switch串联现成primitive，而不先重训teacher。custom language不能通过当前benchmark-locked environment pool，正式实现需独立
  meta manifest/collector并共享唯一temporal wrapper，不放松target40 asset gate。
- 当前没有active GPU job。Phase 0已归档；首个process family Gate A/A2、balanced-SVD learned realizer fold0与
  two-sided coordinate Gate均为non-pass，Gate B、process suite、fold1和Phase 3以后均未启动。process侧现在只剩
  true composite privileged expert/data或物理机制不同的source-unseen family；realizer侧没有剩余预注册successor。
  两侧都应先由专家按全部formal证据重新裁决，不得自行继续版本化。
