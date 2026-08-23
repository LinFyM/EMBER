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
- active goal：完整实现并验证EMBER-ECP；goal仍在进行中；
- canonical workspace：本仓库`main`；GOMQ rank16 Phase 0的formal code authority为clean pushed `ac233fa`，评测已结束；
  当前没有active GPU job，active双前置已转为process-identifying最小pair与mobile-rank4 realizer calibration。

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
- 当前没有active GPU job。Phase 0 GOMQ archival已结束；process最小pair与realizer coordinate可并行；后续顺序固定为：fresh
  Stage 0 Program；distributional `q_pi`；冻结privileged full-chain；`q_V`；ordinary joint Writer；structured outer credit；
  final qualification。当前active双前置就是前两项，不建立新的ECP版本号。
