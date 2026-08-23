# EMBER Task Plan

状态：2026-08-24 **Phase 0以`136/400`归档关闭；Phase 1首个process teacher Gate A以`0/50、19/50`判为non-pass，
复用两个primitive task-local experts的Gate A2仍为`0/50、44/50`、总计`44/100`，当前scene3 soup/butter
family对phase-composed teacher关闭；
Phase 2A的15/15 known-success paths通过，但Phase 2B/2C fixed realizer在fold0 strict250仅为step800/1000=`33/37`，
均低于carrier `43`，breadth `2/5`且Goal/Long均为0。后置mean/innovation分解触发专家预留的two-sided fallback；该fit90
coordinate唯一strict250又得到`80=24/10/46/0/0`、breadth3、Goal/Long为0，只保留carrier `23/43`与known-latest
`59/110` successes，全部预注册门失败。当前coordinate与conditional successor均已停止；fold1、fresh Program、`q_pi/q_V`
未启动。下一shared-realizer机制须先回到专家讨论，不自行版本化。process侧Gate B、suite扩展与
`q_pi/q_V`不启动；下一实质选择是获得true composite privileged expert/data，或改用物理机制不同的
source-unseen process family，需要专家重新裁决。**

当前设计合同：`docs/event_conditioned_policy_compiler_design.md`

本次复核索引与专家最终裁决：`docs/ecp_expert_alignment_audit_20260824.md`第9节

已完成falsification card：`docs/ecp_occupancy_complete_oracle_card_20260823.md`

已完成process Gate card：`docs/ecp_process_minimal_pair_gate_20260824.md`；裁决：
`docs/evidence/ecp_20260824/ecp_process_minimal_pair_teacher_gate_20260824.json`

历史diagnostic card：`docs/ecp_fixed_a_capacity_card_20260823.md`

历史：`docs/research_history.md`与`docs/ecp_stage1_iteration_retrospective_20260823.md`

## Goal

让shared Writer仅根据exact task language与K条action-hidden、内部有序的正确教学视频，在rollout前一次生成唯一一套
完整38-target rank16 LoRA，使冻结source PI0.5在未见初始化上获得强、稳定、广泛且具有视频时序特异性的
zero-interaction闭环能力。正式目标仍为strict paired correct严格`>145/400`，同时满足高breadth、低churn、跨video
鲁棒、Goal/Long贡献和最终correct相对wrong/shuffled/reversed/no-video优势。

## 当前科学锁

旧deterministic/mean `q_pi -> Program -> A/B hyperdecoder`家族已经关闭；不恢复v24、MDCO或小结构/超参变体。这个裁决
不等于专家最终要求的distributional `q_pi(P)`已经实现或被否定。PECS只证明selected-context exact effects能从shared43
提高到58/59；后续four-category structured occupancy panel补齐了initial/successful/candidate/recovery四类support，但当前
retained solver仍直接读取effect bank，绕过Program且依赖future occupancy。

本轮已经回答：结构化闭环状态支持上的多成功策略evidence可把stable carrier从`43/250`提高到fixed-A oracle的`78/250`，但只
覆盖3/5 tasks；known-success mobile-rank4解析投影则可恢复到`110/120/76`，排除明显rank4 topology binding。随后保持同一
bank/objective/12-step数值，只让residual A/B共同移动的正式solver却只有`49/250`、逐task`40/3/6/0/0`。它保留
`41/43` carrier successes，却只恢复`3/115` member-union缺口。当前raw-factor operator停止；仍不训练video predictor、
shared compiler、joint Writer或outer credit，也不扫rank、step、LR、初始化、权重或member。

## Fixed boundaries

- deployment输入只有exact language与K条same-task action-hidden ordered videos；Writer只运行一次；
- source PI0.5、PaliGemma/VLM与native Action Expert冻结；
- native v3是candidate observer；历史Action Meta matched中性，只作control，不默认加载；
- shuffled/reversed不进入训练、loss或checkpoint选择，只在最终候选checkpoint冻结后评测；
- 每个condition只输出一套complete rank16 LoRA，不平均video LoRA、不部署第二adapter、不用task-ID/expert route；
- validation/test action或reward不训练共享模型；fold0 held5只作当前同面板privileged机制比较；
- 当前LIBERO只支持scene/goal/order claim；general process claim等待process-identifying source-unseen数据；
- formal train/eval来自clean pushed commit的detached frozen worktree；不重复已有source、stable carrier或兼容轨迹资产。

## Phase A — 阶段重构与合同

- [x] 冻结learned compiler家族并保留Git/formal历史；
- [x] 将Stage 0 v3降级为candidate，Action Meta降级为control；
- [x] 将Stage 1拆为equivalence identification、privileged realization、shared video inference三个门；
- [x] 将compiler合同改为`Program -> policy-effect distribution -> fixed solver -> LoRA`；
- [x] 将claim收窄到现有数据实际能识别的scene/goal/order；
- [x] 预注册首个occupancy-complete oracle的数据、参数化和直接闭环门。
- [x] 复核后校正阶段命名：effect/evidence bank不是完整`q_pi(P)`，direct-effect solver只是realization子门；
- [x] 形成`docs/ecp_expert_alignment_audit_20260824.md`供专家对照原始方案、修正案、代码与raw evidence重新裁决。

## Phase B — Stage 1A evidence prerequisite：policy-equivalence bank

- [x] 为fold0五项各训练一个不同seed的独立rank16 task expert；只读train actions，固定step2000，不按task选step；
- [x] 对新member跑fixed50 closed loop，五项均捕获一条strict-success完整occupancy；
- [x] 复用现有earliest/latest成功occupancy与stable carrier，不重复大规模资产；
- [x] 收集每task四类等权state bank：8 initial、24 successful、8 prior-candidate、8 recovery；
- [x] 实现官方双相机policy prefix、fixed antithetic noise及source/carrier/三个member的owner/flow/action particle缓存；
- [x] 报告member independence、success union、state/stage覆盖、disagreement和video-observable/recovery信息分界。

**Fold0 evidence prerequisite：** 每task至少两个独立optimization lineages有strict success；48个anchor四类齐全；member轴未被
均值压平但antithetic probe轴尚未保留；Goal/Long也有成功member与完整occupancy。**已通过：** 新独立members fixed250为`113`，逐task
`26/32/37/13/5`，5/5成功occupancy及五个48-state effect banks均完整。

这一步没有输出专家所定义的同构Program posterior。当前bank保存privileged prefix/noise/category/stage/progress与
source/carrier/member owner/flow/action particles；distributional `q_pi(P)`仍未实现，因此不能再写成完整Stage 1A通过。

## Phase C — Direct-effect realization子门：fixed-A privileged oracle

- [x] 从verified stable carrier出发，固定全部`A_c`，只优化`Delta B`，确保zero correction精确返回carrier且无A/B交叉项；
- [x] 实现stage-consistent particle soft-min、carrier no-worse barrier、source/shared preservation、trust与category balance；
- [x] 在一个fit task做真实数值/吞吐profile，只将microbatch定为4，未用held结果调科学量纲；
- [x] 从clean pushed commit并行求解held5五套task-local rank16 oracle LoRA；
- [x] 第一次完整设计后直接跑原fixed250 strict closed loop，不用geometry预筛；
- [x] 按absolute、per-task oracle-normalized recovery、carrier retention、member success union、breadth、Goal/Long裁决；
- [x] 重大失败后暂停复盘，未自动建立下一版本。

**Direct-effect Gate：** final12至少74/250、相对carrier净增至少20、5/5非零、4/5严格胜carrier、Goal/Long各非零、carrier
retention至少33/43、overall oracle-normalized recovery至少0.35。**Realization non-pass：** final为`78/250`，absolute、净增
`+35`及carrier retention `35/43`通过；breadth `3/5`、严格胜carrier `2/5`、Goal/Long `0/0`、recovery
`35/115=.304`且仅3/5为正，因此整体失败。正式证据：
`docs/evidence/ecp_20260823/ecp_occupancy_complete_oracle_gate_20260823.json`。

该门直接读取privileged effect particles，不读取Program。后续C2--C5均只分离这一lower-level realization问题；它们不能单独
裁决`q_pi(P)`、Program-to-effect bridge或`q_V`。

## Phase C2 — Fixed-A capacity separation

- [x] 重新对齐专家最终复核与owner合同，确认fixed-A是候选carrier-preserving参数化而非不可更改的ECP核心；
- [x] 复用三个successful members完成零训练行空间审计：correction energy coverage为`83.3%--96.7%`，但expert absolute
  update coverage只有`41.5%--62.7%`，且Long高于Goal，内部几何不能直接解释共同的0分；
- [x] 预注册解析最优fixed-A投影、三个固定member arms、paired750与事前裁决边界；
- [x] 从clean pushed frozen authority解析生成15套single-LoRA projected member adapters；
- [x] 分别跑latest/independent/earliest原fixed250 matched panels并报告retained/gained/lost；
- [x] 裁决fixed-A为capacity-binding：三个arms=`49/41/35`，matched retention=`31/108、22/113、14/74`，
  Goal/Long全部归零。

本诊断没有optimizer、checkpoint/member选择、interpolation或video输入，不构成Stage 1C授权。card：
`docs/ecp_fixed_a_capacity_card_20260823.md`；evidence：
`docs/evidence/ecp_20260823/ecp_fixed_a_capacity_gate_20260823.json`。

## Phase C3 — Mobile-rank4 residual容量分离

代码与资产审计确认stable carrier本身是精确的`shared rank12 + 4个zero-B reserved ranks`。对15个matched successful
adapters的离线最佳rank4 correction审计达到`99.49%--99.69%` correction energy coverage和
`95.34%--98.90%` expert effective-update energy coverage。因此在写solver前，先直接检验这个最小、严格effective-additive的
参数化是否能保留已知成功行为：

- [x] 复核历史rank12+rank4 decoder的`37/33`只关闭当时的code/objective/decoder，不回答参数容量；
- [x] 确认38/38 carrier targets的后4个B columns精确为0、A仍full-row-rank；
- [x] 完成15个member-task的best-rank4 correction零训练几何审计；
- [x] 在任何formal materialization/GPU rollout前冻结三个matched arms与裁决门；
- [x] 解析物化三个members的15套`carrier12 + mobile residual4` single-rank16 adapters；
- [x] 完成三个strict250 arms：`110/120/76`均略高于matched direct `108/113/74`，breadth均5/5；
- [x] 按预注册门裁决为mixed：pooled retention `245/295=83.05%`、Goal `15/24=62.5%`通过，但Long matched
  retention `4/11=36.36%`未达到50%；capacity-binding条款均未触发；
- [x] 按mixed条款只做无新rollout定位：Long direct/projected union=`11/16`、overlap6、union retention`54.55%`，说明
  失败集中于member-specific row identity，而非Long能力缺失，但不事后改门。

card：`docs/ecp_rank4_residual_capacity_card_20260823.md`。该卡自身的mixed不自动授权solver；后续是否继续必须显式处理
“matched member row”与专家要求的policy-equivalence union口径。Phase C4记录了这个独立决策，不复跑当前arms、不扫rank，
也没有授权Stage 1C或共享Writer训练。

## Phase C4 — Rank-reserved mobile-rank4 realization oracle

原capacity gate不重写；依据专家明确要求的absolute、breadth、Goal/Long、shared retention与multiple-member union综合口径，
解析结果已足以排除明显topology binding。现在只把fixed-A参数面替换为`carrier12 + jointly mobile residual4`，其余bank、
objective、12-step数值、held rows和Gate全部不变。

- [x] 在retained implementation与GPU前冻结operator、profile、formal rows与完整closed-loop Gate；
- [x] 用一个canonical solver替换退役fixed-A runtime，zero residual精确返回carrier并输出single rank16；
- [x] ordinal71完成一次无科学调参权的数值/资源profile；
- [x] held5各物化final12并完成一次strict250闭环裁决；
- [x] 以`49/250`判为Objective/solver non-pass，停止该operator；未补相邻step、未启动Stage 1C；
- [x] 用已有bank/member/projection做零新rollout失败定位：successful endpoints目标显著更低，但当前修正只有其约1%的
  effective norm且方向cosine仅`.041--.077`；
- [x] 在新falsification card前分离“raw-factor zero-start方向/尺度失败”与“effect objective本身不足”，不把LR放大或更多步
  冒充新架构。

card：`docs/ecp_mobile_rank4_solver_card_20260823.md`；evidence：
`docs/evidence/ecp_20260823/ecp_mobile_rank4_solver_gate_20260823.json`。禁止rank/step/LR/初始化/权重/member sweep，也不启动Writer。

## Phase C5 — Gauge-invariant effective-update reachability

raw-factor solver从exact-zero residual奇点出发，12步后correction norm约为known-success projection的1%，方向cosine只有
`.041--.077`；因此它尚未检验一个在effective-update坐标中可达的fixed solver。本轮冻结所有科学输入与Gate，只替换求解几何：

- [x] 在实现前冻结matrix-free rank4 gradient sketch、trust backtracking、VJP预算、fit profile门和held closed-loop门；
- [x] 以一个canonical runtime替换已关闭raw-factor solver，不保留并行fallback；
- [x] ordinal71只运行一次formal profile；实际gap recovery为`0`、trust为`0`，未达到`.50`与`.10--1.50`门；
- [x] profile未过，因此没有物化held5或运行strict paired250；
- [x] 按Profile non-pass停止本operator与Stage 1C授权，不扫sketch/trust/damping/backtrack。

card：`docs/ecp_effective_update_solver_card_20260823.md`；evidence：
`docs/evidence/ecp_20260823/ecp_effective_update_profile_gate_20260823.json`。Action Meta仍保持control；由于本realization
坐标未成立，本轮不启动该matched control。

## Phase D — 专家最终复核与执行合同

- [x] 审计retained source：当前没有distributional `q_pi(P)`或Program-to-effect模块，Stage 1 oracle直接读取effect bank；
- [x] 对照原始方案与最后修正案，形成全过程落实/部分落实/未落实矩阵；
- [x] 由专家固定最终阶段：Stage 0-V、1A-E、1A-P、1B-R0、1B-C、1B-O、2、3、4；
- [x] 将最深缺口固定为deployment-time occupancy completion，而不是“只缺一个solver”或“只缺`q_pi`”；
- [x] 将48-state资产改称four-category structured occupancy panel，并登记probe averaging、member-state validity和stage-wise
  policy splicing问题；
- [x] 冻结默认输出拓扑为stable carrier rank12 + mobile residual rank4，并固定“schema/coordinate/realizer先于`q_pi`”的顺序；
- [x] 恢复执行；不再等待专家二次定稿。

## Phase E — Phase 0：GOMQ真实rank16 archival baseline

- [x] 定位cycle2冻结checkpoint、rank32构造、K4 schedule与历史strict400 exact rows；
- [x] 在实现前冻结确定性canonicalization：逐target保留实数effective update，输出真实76-tensor rank16 BF16 adapter，不训练、
  不选checkpoint、不融合；
- [x] 从clean pushed detached authority物化rank16并做focused effective-update/serialization验证；
- [x] 在同一strict400 correct面板完成一次archival评测，与历史151逐row配对；
- [x] 结果`136/400`低于预注册`145`门；历史151只保留机制/历史证据，归档后不恢复GOMQ训练。

正式rank16逐task为`16/0/0/35/46/34/0/5`，breadth为`5/8`。相对历史rank32的
retained/gained/lost为`123/13/28`、churn41、Jaccard`.75`，400行episode、environment seed与policy-noise common prefix
均严格配对。证据：`docs/evidence/gomq_20260824/gomq_cycle2_effective_rank16_strict400.json`。

## Phase F — Phase 1：process-identifying最小pair与suite

- [x] 按冻结Gate完成soup/butter order pair的50×2 teacher acquisition；结果为`0/50、19/50`、总计`19/100`，低于两个方向
  各20及总计50的固定门，判为Gate A non-pass且不做救援重跑；
- [x] 实现repo-owned temporal wrapper：同scene/language/init/final predicates，wrong-first永久invalid，variant/predicate/phase不进入
  policy或Writer；不放松正式target40 evaluator；
- [x] 选择不重训的更强teacher：phase切换时加载task55/56各自formal `50/50`的step1000 rank16 expert；两个state0工程
  smoke均正确切换LoRA，`butter -> soup`成功、反向完成首事件后失败；
- [x] 按`docs/ecp_process_phase_expert_teacher_gate_20260824.md`完成一次100行strict400 Gate A2；结果为
  `0/50、44/50`、总计`44/100`，尽管相对source-phase teacher净增25条，仍同时未达双向floor与总量门，判为non-pass；
- [x] 完成A2后数据路线只读审计：当前pair缺soup-first successful actions，现有SFT/outer-credit不能直接产生该teacher；
  task21/45共100条source demonstrations则全部为stove-on→pan-on-stove，推荐作为下一个双顺序最小family；
- [ ] 由专家在true composite soup/butter data acquisition与stove/pan替代family之间裁决，并同时裁决新process mappings
  是否足以重开shared realizer；决策前不实现新family或realizer。
- [ ] 取得两个方向均可靠的更强privileged sequential teacher，或基于明确primitive/scene证据预注册一个替代process family；
- [ ] 验证两个privileged task-local experts均能闭环完成，correct video胜sibling wrong，而language-only/no-video/first+final不能
  区分，且无variant leakage；
- [ ] 最小pair通过后扩成跨scene、object与物理约束的family-disjoint process-meta train/held suite；
- [ ] 完整suite在最终Stage 0、`q_pi`、`q_V`共同训练前完成。该准备可与Phase G并行。

## Phase G — Phase 2：fixed mobile-rank4 coordinate与deployment realizer

- completed Phase 2A卡：`docs/ecp_effect_path_calibration_card_20260824.md`；配置：
  `configs/pi05_ecp_effect_path_calibration.json`。
- completed Phase 2B/2C卡：`docs/ecp_fixed_effect_realizer_card_20260824.md`；配置：
  `configs/pi05_ecp_fixed_effect_realizer.json`。
- [x] 用existing known-success mobile projections完成path calibration：global member identity与conservative
  verified-state validity下15/15路径严格单调下降，5/5 tasks的global-particle objective改善；
- [x] 选定并冻结deterministic-sign balanced-SVD canonical rank4为首个principled coordinate；fixed two-sided
  sketches只作有原则fallback；
- [x] 在successor effect evidence中保留probe particles并只采集successful member自身的on-policy anchors，不把旧
  antithetic-average bank冒充distributional authority；
- [ ] 在后继`q_pi`前另补continuation/progress/recovery validity；当前realizer不把这些尚未验证的states冒充target；
- [x] 训练fold0小型target-local amortized `D_eff(structured code)->Delta W_rank4`；training只读fit mappings，held
  inference loader只读冻结code、不打开held target residual或future occupancy；
- [x] 完成fold0 privileged-code transform-only closed-loop：step800/1000=`33/37`，逐global分别为
  `32/0/1/0/0、36/0/1/0/0`；两者均低于carrier `43`、breadth `2/5`、Goal/Long为0，故强门失败且fold1不启动；
- [x] 在两个固定checkpoint均评测后完成一次post-hoc定位：held latest PCA response energy保留
  `79.1%--89.2%`，但step1000 Goal/Long residual effective energy只恢复target的`7.1%/22.9%`；判定为cross-task mapping
  失败而非coordinate-only collapse，不触发two-sided fallback，不做训练/solver小扫。

Phase G当前关闭的是本轮balanced-SVD effect-code learned realizer，不是否定mobile-rank4容量或完整ECP。任何新realizer
successor必须先针对跨任务映射给出与本次证据不同的因果机制，并重新预注册；不能把fresh Program或joint Writer提前用于掩盖
下游必要条件失败。正式裁决：
`docs/evidence/ecp_20260824/ecp_fixed_effect_realizer_fold0_gate_20260824.json`。

新增的fit-task mean/innovation分解满足这一重开条件，并直接选择专家已登记、而非任意发明的第二种坐标：

- [x] 冻结`docs/ecp_centered_two_sided_coordinate_card_20260824.md`与
  `configs/pi05_ecp_centered_two_sided_coordinate.json`；
- [x] 只用fit90构建fixed width8 two-sided probes、task-equal sketch mean与最多128维centered whitened basis；
- [x] held latest只做transform，使用固定top4 core pseudoinverse重建single rank16 adapters；
- [x] 运行唯一strict250 coordinate oracle：`80=24/10/46/0/0`、breadth3，低于83且Goal/Long为0；只保留carrier
  `23/43`与known-latest `59/110` successes；
- [x] 依据失败合同不训练reliability-free centered-innovation realizer、不启动fold1、不扫coordinate超参；shared realizer是否
  放弃或需要新的source-unseen mappings回到专家讨论。

本轮证明two-sided实现没有数值坍塌：held update cosine仍为`.877--.960`，Object甚至达到`46/50`；但fit-span投影不能保留
task-specific closed-loop support，尤其Goal/Long全部丢失。该结果关闭本卡固定coordinate，不外推否定Program、`q_pi/q_V`或
完整ECP。正式裁决：
`docs/evidence/ecp_20260824/ecp_centered_two_sided_coordinate_gate_20260824.json`。

## Phase H — Phase 3：fresh完整Stage 0-V Program

- [ ] 建立owner-specific `P_lang[38,128]`和`P_scene[38,128]`，与
  `P_process[8,38,128]/rho/sigma`形成固定同构schema；
- [ ] 复用native v3初始化、event/horizon binding与Dynamic-K，但默认关闭Action Meta；probe粒子不得提前平均；
- [ ] 用content-grounded frame/action/transition、event/contact/predicate phase、process-pair、cross-video consistency、speed、
  same-procedure/different-object与group-disjoint目标训练，不以pure relative-time schedule为主监督；
- [ ] 在process-held families通过full-vs-first+final、sibling discrimination、same-task稳定、probe稳定、event boundary/non-collapse与
  language/scene/process职责门。

## Phase I — Phase 4--5：distributional `q_pi`与完整privileged ECP

- [ ] 在fit/meta、process、Goal/Long与不同families补足至少两个独立successful lineages，统计独立映射远多于19；
- [ ] 训练`q_pi(P_visible,Z_robust)` particles/mixture；不使用raw task ID、filename或raw A/B rank embedding；
- [ ] held folds只forward privileged evidence，不拟合free code；先通过多fold Stage 1A-P门；
- [ ] 冻结schema、`q_pi`、carrier、coordinate、realizer、checkpoint/member/video规则，运行Stage 1B-O：每task一个
  posterior-marginalized LoRA，无task-local optimizer/early-stop/particle选择；
- [ ] 必须相对carrier显著净增、接近全breadth、Goal/Long非零、高retention、best-member gap恢复约40%、多fold和相邻点同向；
  失败则不进入`q_V`。

## Phase J — Phase 6：deployment `q_V(P|L,V)`

- [ ] 冻结source/backbone、Stage 0 schema、`q_pi`、Program coordinate、realizer与carrier，只训练`q_V`和Dynamic-K aggregation；
- [ ] 仅用exact language与K条action-hidden ordered videos预测与`q_pi`同构的`P_visible` posterior；
- [ ] 直接做deployment-only closed-loop，比较language、language+scene/endpoints、full、same-task-other、K1/K2/K4与process sibling；
- [ ] full须显著胜endpoints、增量分布多数tasks、same-task retention>=90%、Goal/Long非零、process sibling正确且相邻稳定；
- [ ] shuffled/reversed/cross-suite wrong/static/no-video只在最终冻结checkpoint做，不进入训练或选模。

## Phase K — Phase 7--8：ordinary joint Writer与structured outer credit

- [ ] `q_V`通过后先冻结realizer，只解冻semantic/transition/event/Dynamic-K/`q_V`；稳定后才小学习率解冻Program-to-code；
- [ ] 每个主要节点直接closed-loop；连续两个预注册节点absolute/breadth/same-task共同无恢复则退回frozen-realizer checkpoint；
- [ ] 只有full video已有闭环增量后，outer credit才作用于event posterior、Program与mobile residual coefficients；
- [ ] 使用task-equal CRN、success+temporal progress+efficiency、prior/full和retained-success barrier；两节点无净改善即停止estimator。

## Phase L — Phase 9：最终方法资格

- [ ] 全部授权train/meta fresh训练，validation8 deployment-only single-checkpoint paired400；
- [ ] 目标correct严格`>145/400`，并满足high breadth、Goal/Long贡献、same-task鲁棒、相邻稳定及视频controls显著优势；
- [ ] preferably超过GOMQ真实rank16 archival baseline；
- [ ] 方法、checkpoint与controls冻结后才打开Test8；报告K变化及全部paired conditions。

## Done when

- 最终shared Writer满足输入墙与single-LoRA部署合同并达到正式性能/因果/稳定性资格；或
- 完成专家定义的强oracle（包括process-identifying source-unseen data、固定Program schema、独立lineages、verified support、
  deployment-compatible realizer、distributional `q_pi`与多fold privileged full-chain）后仍触发广义ECP停止条件，并以closed-loop证据定位
  最早根本失败接口；
- 验证后的代码、配置、文档和remote-safe evidence及时合并`main`并推送，task-owned worktree/branch/temp产物清理完毕。
