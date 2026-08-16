# EMBER Active Session State

更新时间：2026-08-16。本文是唯一实时实验状态入口；旧文档、Git快照与formal artifacts中的“当前/下一步”只
表示当时时点。稳定目标与owner要求见`docs/current_owner_requirements.md`，历史结果见`docs/research_history.md`。

## 1. Current truth

- 长期Goal处于active：性能继续追求`>150/400`；owner最新接受约145的稳定方法，但必须由相邻single
  checkpoints低换手、same-task-video鲁棒和correct相对negative/no-video的明确因果性共同认证；
- 最新DJNFR cycle1 strict=`136/400`、breadth7、per-task=`1/2/44/35/0/35/18/1`、per-suite=
  `3/79/35/19`；相对LPCP143为`120 retained / 16 gained / 23 lost`、churn39、Jaccard`.754717`。correct与
  retention两门失败，已终局且不resume cycle2、补六臂或做参数小扫；
- SFMC144仍是最高correct单点但lost15/churn31，同样未获稳定或视频因果资格；v6-fast仍是有完整五臂的
  历史最好=`143/135/125/128/129`；
- DJNFR formal训练完整：24 tasks/48 pairs/96 rollouts，candidate/reference=`34/33`、9 active tasks，cycle=
  `717.940s`，world4 checkpoint/completion完整，八head全部更新且0禁读/OOM/nonfinite；
- DJNFR post-train task4/validation8 four-view BA cosine/energy=`.802835/.800444`与`.790242/.785834`，held8
  8/8过门；raw factor/action cosine=`.677321/.686125`、held/train=`1.08903x`、reverse=`1.316782`。因此SJNV的
  hidden->W2/native-factor断裂已被真正关闭，video evidence、顺序、direct LoRA写出和跨视频共同方向不是当前
  最早断点；
- DJNFR all400相对LPCP effective-BA relative-L2 mean/median=`7.773e-5/5.080e-5`，但gained/lost/
  retained-failure分别=`4.522e-5/7.248e-5/8.987e-5`；persistent failure改写最大。最早失败接口是
  selected-success-only credit经一个shared direct-factor update不能形成held reward-useful方向；
- 最新完成design是**V6-LPCP Native Probe-Value Commitment**（NPVC），authority=
  `docs/action_forecast_writer_v6_lpcp_native_probe_value_commitment_design.md`。它保留LPCP143、rank16、CCT
  transport与matched reward recipe，只把factor Value换成已有320 slots的ordered native Action-probe delta，
  并复用同一Procedure-set attention聚合K videos；canonical实现、fresh-incompatible config/checkpoint/eval
  schema和聚合合同已完成，定向CPU=`43 passed`、完整CPU在canonical LIBERO assets环境=`398 passed`；formal前
  validation8×4 K4只读视频held gate已通过：train task4 cosine/energy=`.592915/.679176`，held8平均=
  `.449398/.571497`、6/8 tasks过门，held/train BA L2=`.752521x`、held relative-L2=`9.040e-4`；reverse的
  probe/BA relative-L2=`1.84084/1.37485`，constant norm ratio=`9.167e-4/1.267e-5`，cycle wall=
  `136.063s=1.04074x CCT`。这些preformal门后来没有转化为strict增益，证明内部coherence不能选方法；
  当前没有active GPU run或可resume checkpoint；
- 最新终局的是**V6-LPCP Pre-Addressed Factor-Selective Native Value**（PAFS-NV），authority=
  `docs/action_forecast_writer_v6_lpcp_preaddressed_factor_selective_native_value_design.md`。它保留LPCP143、NPVC
  native Value、rank16、FactorHeads与matched selected-success reward；唯一把common zero-init semantic router
  换成reward前已存在的fixed four-way language address，并给八factor families各自四basis×两轴的zero-init
  diagonal selectors，总trainable=`16,384`。step0/no-video exact LPCP；同task跨video地址相同，video仍是唯一
  dynamic Value。真实task4 smoke八family/q-v-action/reverse/wall均健康，task4 cosine/energy=
  `.435164/.570296`；但train24 address effective rank=`2.15753<4`，validation8=`.168111/.372863`且仅3/8
  过门，相对NPVC held cosine/energy/L2只保留`.3741x/.6524x/.1396x`。机制门失败，未启动full24/strict并
  终局；
- SJNV-Gate已终局：clean `913d3d3` task4 cycle=`135.757s`，train four-view cosine/energy=
  `.472272/.597814`且reward/q-v-action/reverse/static链健康；validation8仅`.201903/.396448`、2/8 tasks过门，
  action cosine=`.042986`、held/train BA L2=`.452509x`，未启动full24/strict。stage localization显示gate/continuous
  hidden cosine约`.94`，但冻结W2后的raw factor cosine/energy=`.021353/.265925`、action factor cosine=`.002672`；
  最早断点是coherent hidden到native public A/B；
- 最新终局successor是**V6-LPCP Direct-Factor Paired Common-State Preference**（DF-PCSP），authority=
  `docs/action_forecast_writer_v6_lpcp_direct_factor_paired_common_state_preference_design.md`。它从sealed LPCP fresh，
  完整保留DJNFR输入、时序、K-set、rank16与八个direct factor heads；唯一把selected-success整轨迹正蒸馏改为
  paired AS139/LPCP两臂分叉前同一初始观测处的winner-vs-loser首段flow preference。四个disjoint correct K4
  views仍只在shared Writer gradient处等权；不加memory、rank、scale、rollout或第二adapter。首次clean `de6c812`
  task4 smoke在credit前发现相同seed的顺序reset不能保证两臂首观测逐元素相同，按工程合同exit1且没有科学结果；
  `07b764b`虽恢复相同flattened state，hard reset仍令两相机分别21,423/27,429个像素值不同，而language/state tokens
  完全相同。canonical现每lane只做一次hard reset+settling；每臂在同一model上deterministic soft reset以清空
  controller/observables，再恢复相同qpos/qvel，不增加rollout或forward。exact task4/task7均为tie；task9/15/18产生
  1/2/1个discordant pairs且preference margin均下降，八head及q/v/action链均非零。但task9 held/train BA幅度仅
  `.105x`，task18 train跨video cosine/energy仅`.290/.428`；三个有效anchors只有task15全门通过。不能挑单一通过
  task冒充shared方法，故终局不启动full24/strict/cycle2；
- 最新终局successor是**V6-LPCP Direct-Factor Successful-Occupancy Counterfactual Preference**（DF-SOCP），
  authority=`docs/action_forecast_writer_v6_lpcp_direct_factor_successful_occupancy_counterfactual_preference_design.md`。
  它只把first-prefix preference扩展为winner成功occupancy的全部replans，并在每个相同observation和policy noise上
  批量查询loser counterfactual action；不增加env rollout，不改carrier、direct heads、rank16、K4或optimizer。
  canonical实现已原位完成，旧DF-PCSP executable/schema已移除；全量CPU=`401 passed`。clean `16cedb9`固定
  task9/15/18 outcomes=`2/1,2/0,1/2`、replay=`26/65/44`均复现，三者train cosine/energy=
  `.825/.838,.721/.757,.770/.757`，held8=`.773/.780,.761/.771,.787/.794`且均8/8过门。但stored
  B2/B1 loser第一action到B8 requery RMS=`.00415/.00595/.00440`，相对名义winner/B8-loser contrast=
  `1.086x/1.693x/.219x`；task9/15 negative受batch-shape混杂。wall相对DF-PCSP=`3.083x/5.335x/3.887x`，
  task9 held/train又仅`.118x`。按门终局，不full24/strict/cycle2；下一接口随后由MB-SOP的同B8双臂重查与
  时间分层informative occupancy实际检验；
- 最新终局successor是**MB-SOP**，authority=
  `docs/action_forecast_writer_v6_lpcp_direct_factor_matched_batch_stratified_occupancy_preference_design.md`。它不改
  LPCP/DJNFR生成图，只把credit panel改成同B8双臂requery，并在每条winner trajectory的8个等进度strata各选
  matched action分歧最大的一项。clean `ad65347`固定task9/15/18复现outcomes=`2/1,2/0,1/2`、complete=
  `26/65/44`与selected=`8/16/8`；wall=`1.655/2.119/1.542x` DF-PCSP，所有matched/info-wall/native门健康。
  post train/held四video BA也健康，但task15/18同一view0 panel margin分别增加`+.0003545/+.0004093`，task9
  held/train仅`.1096x`，故按门终局，不full24/strict/cycle2。额外无forward flat-gradient复跑显示三任务
  view-to-mean cosine minimum=`.695/.629/.601`且raw mean下降覆盖均为`4/4`；最早缺口是raw shared gradient经
  coordinate-preconditioned finite AdamW delta后的方向承诺；
- 最新终局successor是**AR-EC**，authority=
  `docs/action_forecast_writer_v6_lpcp_direct_factor_adam_radius_euclidean_commitment_design.md`，clean `b578d56`。
  task9/15/18 fixed outcomes/counts均复现，raw four-view coverage均`4/4`、final到`-g` cosine均1；但post-margin每项
  只有`1/4` views下降。Adam radius相对raw gradient L2=`6332.5/7988.2/4293.9x`。train BA cosine/energy=
  `.867/.868,.765/.738,.863/.844`，held8=`.782/.779,.829/.811,.817/.811`；q/v/action、reverse/constant、
  info-wall与core wall全部健康。故最早失败接口是全Adam半径越出共同局部下降区间，不是direction/carrier/LoRA。
  按门不full24/strict/cycle2或固定scale sweep；
- 最新终局successor是**AV-MBC**，authority=
  `docs/action_forecast_writer_v6_lpcp_direct_factor_all_view_monotone_backtracking_commitment_design.md`。保留AR-EC方向、
  optimizer state与全部科学图，只从Adam upper radius沿`-g`依次检验`1,1/2,...,1/1024`，接受同一panel/noise下
  四个views全部严格下降的第一个candidate。clean `aa819f2` task9/15/18均完整exit0并复现固定outcomes/counts，
  但三者11个scale全拒绝不是有效科学结果：candidate margins相对gradient-enabled baseline计算，而非同一个
  inference evaluator的step0 baseline；恢复零参数后fixed-action仍分别有`.002624/.002081/.002964` RMS，又证明
  stored rollout B2/B1 action与batch1重算action混比。该轮只作为工程诊断保留。canonical现已改为先用同一
  inference evaluator测step0再rebase全部candidate，并让fixed-action前后都走相同batch1路径；CPU仍为
  `404 passed`、architecture guard无hard violation。修正版clean `202a64d`随后完整复跑：task18在`j=5`、1/32
  radius通过全部门；task9只在`j=10`、1/1024接受，held仅4/8且held/train BA=`.184464x`；task15到`j=10`
  仍无共同下降candidate并精确恢复BA/action零响应。三项cycle=`356.624/659.982/264.169s`。因此scalar radius
  对不同task呈有效/near-identity/空集三种状态，full24共同交集必为空或近零；不启动full24/strict/resume；
- 最新终局successor是**MMCD**，authority=
  `docs/action_forecast_writer_v6_lpcp_direct_factor_maximum_margin_common_descent_commitment_design.md`。它保留AV-MBC
  全部carrier、credit、Adam upper radius、native backtracking和rank16，只用已有四view gradients的`4x4` Gram
  确定性求maximum-margin common-descent direction，并保持每task原mean norm与跨task等权。clean `fc3bdd7`
  task9/15/18的continuous worst margin均提高，但native结果分别为j0大步且held/train`.160558x`、j0--10无共同
  candidate并exact no-op、j6且全门通过；只有1/3 anchors过门。MMCD终局，不full24/strict/resume或小扫；
- 当前没有active GPU run或可resume checkpoint。下一变量必须针对continuous direction到native BF16 finite-step及
  held amplitude接口；不能退回替换已通过的LPCP carrier，也不能把memory/rank8等仍开放候选误判为已否决；
- 最新终局successor是**PAV-BC**，authority=
  `docs/action_forecast_writer_v6_lpcp_direct_factor_preconditioned_all_view_backtracking_commitment_design.md`。它不改
  carrier/credit/optimizer/rank/FactorHeads/acceptance，只把final ray换成实际AdamW candidate delta并沿
  `1...1/1024`回退。clean `581140c`中task9 j5但held/train仅`.109466x`，task15/18全部11项拒绝并exact no-op；
  0/3 anchors通过，终局不full24/strict/resume；
- 当前没有active GPU run或可resume checkpoint。raw/MMCD/Adam三类parameter ray均已否决；下一设计必须转LoRA
  输出/effective-BA参数化的native-safe线性Value路径，不能继续做ray mixture、trust scale或替换已通过的carrier；
- 最新终局successor是**ALB-NV**，authority=
  `docs/action_forecast_writer_v6_lpcp_anchored_linear_b_native_value_commitment_design.md`。它只把八个A/B residual heads
  改为四个fixed-A B-only heads，使新增`delta-BA=delta-B A0`严格线性；rank16、LPCP、MB-SOP与PAV acceptance不变。
  correct400固定几何给出B-side相对A-side灵敏度q/v/action-in/action-out=
  `1.049/1.411/2.594/8.258x`，故不是按anchor或held选择side。clean `0899166`三anchor训练/分析均exit0，但只
  1/3过门：task9 `j0--10`无共同native下降并exact no-op；task15首次从PAV no-op变为j5，held BA
  `.375/.513`、held/train`.333x`，却只有5/8 held tasks且raw-B `.101/.323`；task18从PAV no-op变为j0，held BA
  `.774/.785`、8/8、held/train`1.030x`并全门通过。三项A tensor变化均精确0，合计wall/PAV=`.75247x`；
- ALB-NV按固定3/3门终局，不full24/strict/resume、A-side completion、family-side mix或小扫。最早缺口是小的
  continuous B-only共同reward方向仍不能在不同task上稳定变为native finite-step：可能exact no-op，也可能在小j
  下先丢raw-B/held task coverage。当前没有active GPU run或可resume checkpoint；下一输出设计必须让残差从
  native-zero坐标可见，同时完整保留LPCP rank16 carrier，不能压缩baseline或回到parameter ray；
- 最新终局successor是**NZRB-C**，authority=
  `docs/action_forecast_writer_v6_lpcp_native_zero_residual_bank_commitment_design.md`。唯一把ALB public factor表示改为
  rank32单adapter：`A=[A0;A0]`、`B=[B0,delta-B]`，令数学上相同的`delta-B A0`从zero-B bank进入native BA；
  `alpha=rank=32`保持scale1，原rank16 carrier逐元素保留，四head 860,160 trainable与上游全部不变。clean
  `d4fc92e` fixed task9/15/18完整exit0；稳定结构五项误差均精确0。task15/18纠正后全门通过，held BA=
  `.95235/.93984,.93418/.92186`、raw-B=`.95322/.94073,.93272/.92047`且均8/8，证明zero bank修复accepted
  update的native可见性与跨video factor coherence；task9 outcome由预定`2/1,26`漂为`1/0,25`，j0--10仍无
  candidate并exact no-op。三anchor wall/ALB=`1.16565x>1.15x`，故2/3与吞吐两门失败，不full24/strict/resume或
  bank/rank/scale小扫。初版约`1e-3`结构报警是跨autocast重算carrier的analysis错误，三条
  `nzrb_stable_rank_bank_contract.json`已纠正；
- 最新终局successor是**NEAP-C**，authority=
  `docs/action_forecast_writer_v6_lpcp_native_endpoint_action_preference_design.md`。它保留LPCP、NZRB rank32 zero-bank、
  MB-SOP同B8 panel、四correct K4 views与PAV acceptance，唯一把随机flow-time CFM preference改为完整10步
  PI05部署action endpoint preference。generated action只运行一次并同时比较winner/loser executed-prefix距离；
  gradient与native backtracking用同一个endpoint metric。canonical已原位实现，定向CPU=`50 passed`、完整CPU=
  `405 passed`、compileall与architecture guard 0 hard，active source净`-16`行且没有平行runner。clean pushed
  `33f69fd`、gpu02物理1完成task9，root=
  `runs/outputs/pi05_v6_lpcp_native_endpoint_action_preference_task9_mechanism_b8_33f69fd_gpu02p1_20260816`。
  outcome/count=`1/0,25/8`，physical B8，cycle=`97.107s`、reserved=`19.367GB`；gradient cosine/energy=
  `.846/.865`且j0一次接受。held8为8/8，BA/raw-B/action cosine=`.953/.955/.485`，reverse/constant与rank-bank
  全通；但held/train BA L2=`.234<.30`，26/27门仍按authority终局，不跑task15/18、full24/strict或小扫。
  probe/joint幅度为`.671/.665x`，direct rows骤降`.223x`，最早缺口是one-task condition到shared direct-head的
  跨task幅度。NEAP没有可resume checkpoint；
- 最新终局successor是**TCEC**，authority=
  `docs/action_forecast_writer_v6_lpcp_task_complete_endpoint_coexistence_design.md`。它完整保留NEAP/NZRB/LPCP，唯一
  把commitment单位从一个task-local probe改成所有active tasks：task9/15/18先在world3 fresh共享一个等权mean
  Adam candidate，只有三个task共12个correct K4 endpoint margins全部下降才由所有ranks接受同一个scale。
  clean frozen `9ed6a08`在gpu02物理`1/2/3`完整exit0：task9/15/18预定outcome/count全部复现，总计6 paired states、
  12 rollouts、134 chunks、32 selected pairs与48条correct videos，cycle=`182.142s`。每task内部four-view
  cosine/energy=`.846/.865,.596/.645,.448/.557`且均4/4下降；但task间cosine mean=`-.14513`，task15 norm为
  task9/task18的`41.45x/10.43x`，global raw mean仅1/3 task下降。11个native scales最多8/12 margins下降，
  全部拒绝，最终860,160个B-head参数逐元素全零。按门不full24/strict/held controls或小扫；终局artifact位于
  fresh root的`tcec_shared3_terminal_adjudication.json`。当前无active successor、GPU run或可resume checkpoint；
- owner已明确与`ycliu`沟通过共享设备；若没有足够空卡，可在实时显存余量充足、利用率低且不造成明显干扰时与其
  进程安全共驻。仍不reset/kill/pause他人进程，也不为凑卡等待或跨节点拼接；
- 唯一主工作树：`/data1/user/ymdai/projects/EMBER`；唯一主写分支：`codex/bci-continuation`；
- 当前最强zero-interaction carrier baseline：**V6 Layerwise Action-Probe Conditioned Procedure Reader**（V6-LPCP）macro25 K4
  strict=`143/400`、breadth7、per-task=`1/4/48/35/0/38/16/1`、per-suite=`5/83/38/17`、top3=
  `121/143=.84615`；按`<144`和lost>10门终局，不resume50、不补controls或扫memory/rank/LR/scale/seed；
- 同AS139逐项验证400个task/state、K4 teacher sets/order、env seed和policy RNG prefix后，严格配对=
  `120 retained / 23 gained / 19 lost / 238 both-fail`、churn42、net`+4`、p=`.643969`；suite net=
  `+2/+5/+2/-5`。Long1=`7 gained / 13 lost`净丢6，Goal3仍0，Long2只从0解锁1；
- count-only相对不同teacher schedule的v6-fast143逐task差=`+1/+1/+2/-2/0/+2/-4/0`，aggregate追平且breadth
  `6->7`，但没有超过150，也没有正确视频controls，因此不能称为有效视频教学突破；
- 全400 AS139/LPCP cache的effective-BA relative-L2 mean/median=`.002653/.001916`，cosine mean=
  `.99999479`、norm ratio=`.99997391`；first4 LoRA norm/rank/q-v-action结构不变，同task correction coherence
  mean/median=`.61786/.56804`。Goal3改写`.004224`且coherence`.88373`仍`0/50`，Long1只改`.001324`却净丢6；
- carrier并未失败：79帧真实视频仍是同一次joint forward，reverse query-delta/Program relative-L2=
  `2.0572/.40414`，constant query-delta近零，reader/controller持续更新。失效接口在conditioned Procedure经冻结
  fusion/compiler只形成AS139邻域小方向，以及blind B20 functional credit不能选择held occupancy有用方向；
  因此本结果不直接触发只替换carrier的literal-memory重跑；
- clean detached `515f91e` world6 formal完整exit0，macro mean=`26.462s`，K1--K4每macro各6、最长359帧完整；
  generation B32=`.221500 LoRA/s`。正确资产合同strict root完成72/72 jobs、400/400 rows、exit0，rollout-only=
  `671.501s`；root=
  `runs/outputs/pi05_v6_layerwise_probe_conditioned_procedure_macro0025_k4_correct400_noreplacement_seed7_trainr6_evalr6_assetfix_07ec4d8_gpu01_20260814`；
  同root封存`as139_to_v6_lpcp_macro0025_paired_transition.json`和`v6_lpcp_macro0025_effective_ba_analysis.json`；
- 首次strict root因临时launcher误写旧`/data0` LIBERO assets，在400份LoRA生成后、首个环境创建前工程失败，
  没有任何rollout分数；旧contract封存错误路径，未resume为正式结果。新root先按同一contract真实创建8/8 validation
  环境再启动，失败只作工程证据；
- 已完成predecessor **V6-LPCP Paired Causal Success Distillation**（PCSD），authority=
  `docs/action_forecast_writer_v6_lpcp_paired_causal_success_distillation_design.md`。它不把memory token或rank变化
  预设为答案：冻结LPCP carrier与AS139/rank16 tail，只训练`query_delta.weight`；train24每task以K2严格同初态
  AS139-reference/LPCP-candidate arms产生唯一成功轨迹credit。AS139/LPCP validation union=`162`只作设计依据，
  不能冒充可部署分数。PCSD已在canonical reward path原位实现：共享一次detached conditioning state，顺序执行
  exact K2 reference/candidate arms，只collate唯一成功轨迹，active-task等权更新单一`query_delta.weight`；旧LOO、
  support projection、K4 replay和outcome-only active runtime均已移除。全量CPU=`387 passed`，architecture guard
  无hard violation；clean pushed/frozen commit=`efc17bead8528f3ca731bd99ab0c44b9fe1c4a7b`在gpu01物理
  `5/6/7`、world3完成full24 cycle1，root=
  `runs/outputs/pi05_v6_lpcp_paired_causal_success_distillation_formal_cycle0to1_r3_k4_nmc4_b8_efc17be_gpu01_20260815`；
  24 tasks/48 paired states/96 rollouts完整，reference/candidate success=`33/34`、candidate/reference gains=
  `5/4`、both-success/failure=`29/10`，9 discordant/active tasks覆盖Spatial/Object/Goal三suite，319 replay chunks、
  1,582 executed steps。selected-success objective=`.101456`，Writer grad RMS=`3.4288e-8`，query-delta delta RMS=
  `.000189653`，3项deployment probe的effective-BA/fixed-action response均非零；0 teacher/target/validation/test
  action/reward read、OOM、nonfinite或watchdog，wall=`837.694s`。机制门通过但训练内净优势仅1，不能写成性能
  提升；formal checkpoint=`checkpoints/cycle_00000001`完整；
- PCSD cycle1 K4 correct strict paired400已完成400/400 LoRAs、42/42 shards、400 rows、9 workers exit0：
  `135/400`、breadth6、per-task=`0/4/48/32/0/35/15/1`、per-suite=`4/80/35/16`、top3=
  `115/135=.85185`。相对LPCP143严格=`121 retained / 14 gained / 22 lost / 243 both-fail`、churn36、
  net`-8`、Jaccard`.77070`；相对AS139严格=`115/20/24/241`、churn44、net`-4`、Jaccard`.72327`。
  count-only相对v6-fast143/old134/compiler138/online128分别=`-8/+1/-3/+7`；四项预注册失败条件全部触发，
  因而不续cycle2、不补controls、不做query/LR/rank/scale小扫；
- 全400 PCSD相对LPCP effective-BA relative-L2 mean/median=`.0006834/.0006767`、cosine=
  `.999999765`、norm ratio=`.999998471`，比LPCP相对AS139的`.002653`又小约3.9倍；gained/lost的改写
  mean=`.0006873/.0006830`，幅度不能区分好坏。训练时三项fixed-action response RMS=`.00215--.00407`，
  说明LoRA→action链路没有断；
- 对validation8每task前4个不同K4 correct video sets做FP64 trace，PCSD增量的同task pairwise cosine跨task
  平均=`-.00187`，task-mean/sample energy ratio=`.24860`，几乎正好是四个相互正交修正平均后的`1/4`。
  最早失效接口因此是**9条稀疏paired reward trajectories经一个shared query commitment不能把同task不同video
  的局部成功方向合并为可保留的共同程序**，而不是reward无信号、carrier没读视频或LoRA/action没变化；
- 终局artifact均在PCSD strict root：`pcsd_cycle1_benchmark_comparison.json`、
  `as139_to_pcsd_cycle1_paired_transition.json`、`pcsd_cycle1_effective_ba_analysis.json`、
  `pcsd_cycle1_first4_correction_fp64.json`、`pcsd_cycle1_terminal_adjudication.json`；
- 最新完成successor是**V6-LPCP Cross-Video Causal Success Distillation**（CV-CSD），authority=
  `docs/action_forecast_writer_v6_lpcp_cross_video_causal_success_distillation_design.md`，clean pushed/frozen commit=
  `c1d8952819a0051b6b608f2caadb8568a415f502`。它只改变success credit覆盖：同一selected-success trajectory在四个
  disjoint same-task correct K4 conditions下分别做完整Writer→LoRA→policy CFM，再平均`query_delta.weight`梯度；
  deployment、rank16、AS139 tail、optimizer、rollout数和信息墙不变；
- CV-CSD full24 cycle1在gpu01物理`5/6/7`、world3完整exit0：24 tasks/48 paired states/96 rollouts，reference/
  candidate success=`33/34`、candidate/reference-only=`5/4`、9 active tasks/3 suites，与PCSD paired outcome逐项相同；
  36个credit conditions全部LoRA/query gradient finite/nonzero。cycle=`863.432s`，仅为PCSD的`1.0307x`；rank0/1/2
  各8 tasks/3 active tasks，recorded load max/min=`1.0828x`，所以没有rank分配失衡或吞吐违约；formal root=
  `runs/outputs/pi05_v6_lpcp_cross_video_causal_success_distillation_formal_cycle0to1_r3_k4_views4_nmc4_b8_c1d8952_gpu01_20260815`；
- cycle1 K4 strict paired400完成400/400 LoRAs、42/42 shards、400 rows、9/9 workers exit0：`134/400`、breadth7、
  per-task=`1/2/47/32/0/36/15/1`、per-suite=`3/79/36/16`、top3=`115/134=.85821`。相对LPCP143严格=
  `122 retained / 12 gained / 21 lost / 245 both-fail`、churn33、net`-9`、Jaccard`.78710`且四suite全降；
  相对AS139=`121/13/18/248`、churn31、net`-5`；相对PCSD135=`115/19/20/246`、churn39、net`-1`；
  count-only相对v6-fast143/old134/compiler138/online128=`-9/0/-4/+6`。五项门只有breadth通过，终局不续cycle2、
  controls或小扫；strict root=
  `runs/outputs/pi05_v6_lpcp_cross_video_causal_success_distillation_cycle1_k4_correct400_noreplacement_seed7_trainr3_evalr3_c1d8952_gpu01_20260815`；
- 全400 CV-CSD相对LPCP effective-BA relative-L2 mean/median=`.000683702/.000677739`，gained/lost=
  `.000679265/.000679434`。FP64 first4同task四correct K4增量pairwise cosine=`.00020542`、mean/sample
  energy=`.25015457`；相对PCSD也为`-.00190786/.24857794`。跨video exact成功信用仍经video-specific Jacobian
  落成近正交局部BA方向；最早失效接口是shared query-only commitment，而非video/reward/LoRA链路；
- 最新完成**V6-LPCP Semantic Factor-Memory Commitment**，authority=
  `docs/action_forecast_writer_v6_lpcp_semantic_factor_memory_commitment_design.md`，clean frozen commit=`8994180`。
  full24 cycle1=`24 tasks / 48 pairs / 96 rollouts`，candidate/reference success=`34/34`，8 active tasks、32
  credit conditions、128 unique videos；8/8 family maps更新，cycle=`920.555s`=`1.0662x` CV-CSD，三rank任务=
  `8/9/7`、负载max/min=`1.0653x`，禁读/OOM/nonfinite为0；
- SFMC strict correct400=`144/400`、breadth7、per-task=`1/3/47/36/0/38/18/1`、per-suite=`4/83/38/19`、
  top3=`121/144=.84028`。相对LPCP143严格=`128 retained / 16 gained / 15 lost / 241 both-fail`、churn31、
  net`+1`、Jaccard`.805031`、p=`1`；suite net=`-1/0/0/+2`。lost≤10门失败，终局不续cycle2或六臂；
- 稳定FP64差分显示SFMC/LPCP all400 effective-BA relative-L2 mean/median=`2.899e-7/1.066e-9`，非零样本
  q/v/action=`249/16/1`；first4 pairwise cosine=`-8.10e-6`、mean/sample energy=`.249995`。semantic query/
  basis-key delta约`1.7e-9`，连续hidden residual主要被native factor量化为稀疏q-family ULP crossing，未形成
  learned semantic route或跨video共同方向；
- 最新终局successor是**V6-LPCP Gradient-Open Semantic Commitment**，authority=
  `docs/action_forecast_writer_v6_lpcp_gradient_open_semantic_commitment_design.md`。它从sealed LPCP macro25 fresh
  初始化commitment与optimizer，只把SFMC的zero-init staged函数改成step0严格identity、但family delta maps与
  semantic query首步同时有梯度的V6-W1 anchored参数化；其余carrier、K4 four-view credit、rank16与训练合同
  全部冻结。实现与分族response统计已push到`5b14c89`，full CPU=`396 passed`；gpu02物理1 task4真实smoke
  完整exit0，8/8 maps更新，semantic query delta=`1.1979e-4`（SFMC为`1.7564e-9`），q/v/action native
  effective-BA=`6.6169e-7/9.1517e-7/4.8908e-8`，总BA=`6.9391e-7`、fixed-action=`.0027033`，
  cycle=`132.458s`=`.9501x` SFMC，机制/效率门已过。clean detached `eb543d3`随后在gpu01物理
  `2/4/5/6/7` world5 fresh完成full24 cycle1：24 tasks/48 pairs/96 rollouts，candidate/reference=
  `33/31`、gains=`6/4`、10 active tasks覆盖四suite、40 credit views/160 unique videos；semantic query
  delta=`6.9499e-5`（SFMC的约3.96万倍），5/5 probes的q/v与3/5的action native BA非零。cycle=
  `581.924s`，5 rank任务=`3/5/2/5/9`但recorded wall max/min仅`1.2121x`，world5 checkpoint/completion
  完整exit0。随后同checkpoint K4 strict完成400/400 LoRAs、60/60 jobs、400 rows、15/15 workers exit0：
  `141/400`、breadth7、per-suite=`4/77/36/24`、top3 share`.80142`；相对LPCP=`128/13/15`、churn28、
  net`-2`，suite净变化=`-1/-6/-2/+7`，相对SFMC=`124/17/20`、churn37。BA relative-L2 mean=
  `9.6632e-6`、为SFMC约`33.3x`，q/v/action非零样本=`400/399/368`；但first4 cross-video cosine=
  `.0001442`、energy ratio=`.250124`。训练paired outcome跨world存在正常低位轨迹差异，不能以`33/31`
  预告held性能；strict已证实能力换手，四项门失败，不续cycle2或六臂。当前没有可resume的active checkpoint；
- 首次ADSP formal commit=`b38a644`、world6物理`1/2/4/5/6/7`在任何metric/checkpoint前工程失败：旧raw replay
  builder对all-success homogeneous panel只返回summary，而ADSP首次需要其完整support batch。根因已在最早data
  boundary修复为“仅all-failure summary-only，all-success完整collate”；mixed与task4 smoke语义不变，新增集成
  回归后full CPU=`401 passed`。失败root只作工程证据、不resume；随后`ad2e1be` fresh formal与strict结果见上；
- reward graph-release实现commit=`fa53ce43c92915229ca4c49fe47d2aa6f16bef0c`已push；独立config=
  `configs/pi05_writer_v6_ordered_procedure_on_policy_preference_v1.json`，reward checkpoint/evaluator明确使用cycle schema，
  不冒充AS exact-resume；正确LIBERO assets下full CPU=`395 passed`；
- 首个world5 full24 attempt root=
  `runs/outputs/pi05_v6_ordered_procedure_on_policy_preference_formal_cycle0to1_r5_k4_nmc4_b8_039cbbf_gpu01_20260814`；
  四个已完成rank先在barrier等待600秒触发watchdog；约一分钟后、PG已经失败时，仍在CFM的最慢rank才在一次
  242MiB申请中报告OOM。exit1、无metrics/checkpoint/completion；可证首因是合法长task尾部超过默认timeout，
  不是科学non-pass。可训练compiler graph跨Nmc4 replay存活是历史已知且无需保留的额外显存风险，一并移除；
- `fa53ce4`在gpu01物理2对同一task4 B8真实mixed smoke完整exit0：`1/4` success、157 chunks、80次functional forwards，reward LoRA
  gradient RMS=`1.3138e-5`、Writer grad norm=`8.0119e-4`；q/k/output均更新，effective-BA/fixed-action response=
  `.00018146/.00557193`，与修复前逐位相同；peak reserved=`40.712GB`、wall=`146.383s`。task0 homogeneous面板严格
  zero forward/gradient并按门硬停；reward专用collective timeout为30分钟；
- graph-release正式cycle1使用clean frozen commit=`9c2638658e71095525444efbcb5e9dd86c62926c`、gpu01物理
  `2/4/5/6/7`、world5完整exit0，总wall=`674.031s`、peak reserved=`40.758GB`、0 forbidden read/OOM/
  nonfinite/watchdog；root=
  `runs/outputs/pi05_v6_ordered_procedure_on_policy_preference_formal_cycle0to1_r5_k4_nmc4_b8_graphrelease_9c26386_gpu01_20260814`；
- cycle1 strict评测同commit、同5卡、15 persistent workers完成400/400 LoRAs、60/60 jobs与400 rows，wall=
  `1340.128s`、rollout-only=`783.248s`；root=
  `runs/outputs/pi05_v6_ordered_procedure_on_policy_preference_cycle1_k4_correct400_noreplacement_seed7_trainr5_evalr5_9c26386_gpu01_20260814`；同root封存
  `reward_preference_strict_adjudication.json`与`reward_preference_first4_geometry.json`；
- gpu01物理7真实smoke通过K1、倒序、freeze与梯度门；clean detached `50a3c36`在gpu01物理`0/1/2/4/5/6`
  完成full24 B20 macro1/2 profile：`26.112/22.543s`，gradient=`.0003266/.0003663`，q/k delta=
  `.0001158/.0001183`，K各6、最长323帧完整、peak reserved=`40.758GB`、0 OOM/nonfinite；root=
  `runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_profile_r6_b20_50a3c36_gpu01_20260814`；
- clean detached `d316623`已在gpu01物理`2/4/5/6/7`以fresh world5完成macro0->25，绝未加载profile state：
  25/25 metrics、完整world5 checkpoint、completion与exit0，总elapsed=`745.622s`，macro mean=`29.790s`，
  loss `.10118184->.09564162`，gradient范围`.00025272--.00046269`，K各6、最长359帧完整、peak reserved=
  `40.758GB`；root=
  `runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_formal_fresh0to25_r5_b20_d316623_gpu01_20260814`；
- macro2->25 q/k delta=`.08636/.08605`，output norm `.009275->.277774`；first4 output-zero反事实中Procedure
  correction=`.09601`，entropy/log4=`.99443`，但effective-BA mean/task-mean只改写`.01397/.01392`，action=
  `.00989`。raw ordered Procedure已打开完整credit，但native compiler后的policy改写仍较小；
- macro25 live K4 deployment profile在gpu01物理2完成：B8/B16/B32=
  `.2250164/.2247286/.2247036 LoRA/s`，三者stable、0 OOM/nonfinite，peak reserved约
  `12.952/12.973/13.011GB`，最长226帧，按最高吞吐锁B8；root=
  `runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_k4_writer_generation_profile_val8x4_correct_gpu01p2_d316623_macro0025_retry1_20260814`；
- strict evaluation commit=`c7e6666e807a3dfab97ae31684640ccfc5e09c79`，400/400 LoRAs、60/60 jobs、
  400 rows、15 workers全部exit0，wall=`1302.949s`、rollout-only=`765.171s`；root=
  `runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_k4_correct400_noreplacement_seed7_macro0025_trainr5_evalr5_c7e6666_gpu01_20260814`；
- train-seen trained root=
  `runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_train_seen8x10_trained_macro0025_k4_screen_c7e6666_gpu01_20260814`，output-zero root=
  `runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_train_seen8x10_outputzero_macro0025_k4_screen_c7e6666_gpu02_20260814`；两者exit0且adjudication已写入trained root；
- 本轮terminal design是V6 Semantic-Core Common-Value Set Bridge；authority=
  `docs/action_forecast_writer_v6_semantic_core_common_value_set_bridge_design.md`；canonical实现已原位替换centered
  Value与旧schema。显式K1旁路的零导数训练边界已修正，正式环境full CPU=`374 passed`；
- 当前Common-Value profile commit=`2eb9da9efae0cead6e0d936172eed7165ea6b8bf`，gpu01 world6
  macro1/2=`25.930/22.530s`，peak allocated/reserved=`36.495/40.758GB`，K各6、最长323帧无截断、
  0 OOM/nonfinite；gradient norm=`.002698/.002795`，较centered路径约`.00000325`打开约三阶，macro1→2
  q/k delta=`6.552e-6/6.480e-6`；formal config已seal；
  root=`runs/outputs/pi05_v6_semantic_core_common_value_set_bridge_profile_r6_b20_2eb9da9_gpu01_20260814`；
- 当前Common-Value formal commit=`12311bd88a81847cf108598379b043f971fd6c85`，gpu01 world6 fresh
  macro0→25完整：25/25 metrics、checkpoint、6 rank states、completion、exit0齐全，总耗时`614.636s`；macro
  min/mean/max=`22.512/24.559/26.239s`，loss `.10118184→.09558529`，gradient范围
  `.002501--.003254`，peak reserved=`40.758GB`，0 OOM/nonfinite；macro25 output norm=`.261523`；root=
  `runs/outputs/pi05_v6_semantic_core_common_value_set_bridge_formal_fresh0to25_r6_b20_12311bd_gpu01_20260814`；
- 同一macro25在gpu01物理4完成一次实际K4 B32 longest-panel确认：32 LoRAs / `141.995s`=
  `.225360 LoRA/s`，peak allocated/reserved=`12.144/13.181GB`，最长视频226帧，0 OOM/nonfinite；B32选择继承
  同形predecessor完整B8/B16/B32 profile，不重复21分钟比较；root=
  `runs/outputs/pi05_v6_semantic_core_common_value_set_bridge_k4_b32_confirmation_val8x4_correct_gpu01p4_12311bd_macro0025_20260814`；
- strict evaluation commit=`0ead61e2c45f4a5cfe129ffbee3dfa51b2ddfb60`，400/400 LoRAs、72/72 jobs、
  400 rows与18/18 workers全部exit0，wall=`1225.323s`、rollout-only=`677.538s`；root=
  `runs/outputs/pi05_v6_semantic_core_common_value_set_bridge_k4_correct400_noreplacement_seed7_macro0025_trainr6_evalr6_0ead61e_gpu01_20260814`；
- 同root保留`common_value_strict_adjudication.json`与`common_value_mechanism_first4.json`；机制runtime root=
  `runs/outputs/pi05_v6_semantic_core_common_value_set_bridge_mechanism_first4_runtime_gpu01p5_0ead61e_macro0025_20260814`；
- 上一轮Semantic-Core Set profile commit=`7883fa6b71c361a28722ef9ce5047043b2966ebc`，macro1/2=
  `27.214/24.277s`，peak
  allocated/reserved=`36.495/40.758GB`，K各6、最长condition 323 stride-5 frames且无截断、0 OOM/nonfinite；
  macro1→2 q/k delta=`7.859e-7/7.736e-7`；root=
  `runs/outputs/pi05_v6_semantic_core_set_bridge_profile_r6_b20_7883fa6_gpu01_20260814`；
- 上一轮Semantic-Core Set formal commit=`884e55e18fad84c4266e3d857754a9538c59d20a`，gpu01 world6 macro0→25共
  `619.319s`，25/25 metrics、checkpoint、completion与exit0完整；loss first/last=`.10118184/.09564428`，
  q/k相对macro2继续移动`.003205/.003040`，0 OOM/nonfinite；root=
  `runs/outputs/pi05_v6_semantic_core_set_bridge_formal_fresh0to25_r6_b20_884e55e_gpu01_20260814`；
- 上一轮Semantic-Core Set K4 profile在gpu01物理4测得B8/B16/B32=`.2231465/.2231839/.2232875 LoRA/s`，三者stable、
  0 OOM/nonfinite，按最高吞吐锁B32；root=
  `runs/outputs/pi05_v6_semantic_core_set_bridge_k4_writer_generation_profile_val8x4_correct_gpu01p4_884e55e_macro0025_20260814`；
- 上一轮Semantic-Core Set strict evaluation commit=`850bd38fb40ed5ec5c9c813aaa65380f1ec6de53`，400 LoRAs、72 jobs、18 workers
  全部完整exit0，wall=`1222.766s`、rollout-only=`682.032s`；root=
  `runs/outputs/pi05_v6_semantic_core_set_bridge_k4_correct400_noreplacement_seed7_macro0025_trainr6_evalr6_850bd38_gpu01_20260814`；
- 当前暂不使用subagents；实现、训练、评测和分析由当前主任务持续完成；

## 2. Latest completed architecture

完整设计authority：
`docs/action_forecast_writer_dynamic_k_semantic_address_direct_family_b_design.md`。

数据流：

```text
exact language + K=1..4 same-task action-hidden ordered videos
-> 每帧真实joint image/language/50 Action-probe context + 8 memory tokens
-> per-video signed adjacent transitions D + terminal goal residual G
-> absolute memory mean只作temporal Query semantic address
-> causal temporal encoder
-> permutation-invariant cross-video set attention + symmetric reduction
-> 20 policy groups x 8 rank coordinates M2P
-> shared bias-free 256->1024 projector
-> four bias-free zero-init direct family-B readouts
-> one complete 38-target rank-8 task LoRA
```

当前方法保留Dynamic-K semantic-address的全部输入、memory、temporal、set、M2P、fixed A和B20 recipe，只删除
旧mapper的四个family `1024->1024` hidden/GELU与未启用dynamic-A heads。最新mapper为5个trainable matrices、
`3,702,784`参数；整个Writer共`9,987,840`个trainable参数，输出76 tensors / `643,584`个LoRA scalars。

单变量依据：semantic-address macro50 strict=`101/400`，但逐接口probe中correct task-mean off-diagonal cosine从
M2P/final/shared-project的`.492/.529/.530`到family hidden的`.634`和dynamic-B/effective-BA的`.779/.779`。
因此首个新增common-direction接口是旧nonlinear family mapper，不是继续重写已经能保留task/order差异的视频
前端。上一代结果和probe只支持这个窄假设；下文的closed-loop结果已经否决其充分性。

## 3. Completed formal training

第一次world6 formal attempt：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_formal_fresh0to50_r6_b20_c5353f3_20260813`

owner要求停止时停在完整macro16，无macro25 checkpoint、无completion、无strict评测。它只记录一次用户中止的
非完整run，不得resume、不得作为正式成绩，也不得覆盖。

完整fresh run：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_formal_fresh0to50_r5_b20_c5353f3_retry1_20260813`

- frozen worktree：`/data1/user/ymdai/worktrees/EMBER-direct-family-b-formal-c5353f3`；
- clean commit：`c5353f3442a88565eded3b968dda104df5acc5cb`，与origin一致；
- host/devices：`gpu01`物理GPU`0,4,5,6,7`，world5；启动时五卡空闲健康，gpu01 1/2/3属于他人；
- launch：fresh macro0，formal total400，当前段stop-after50，B20，checkpoint every25，num_workers0；
- dynamic K：每macro的24 tasks中K1/K2/K3/K4各6，task等权、跨episode action queries；
- environment：BF16/TF32、`NCCL_P2P_DISABLE=1`、GPU-local NUMA、deferred NCCL；
- 原训练tmux `ember_dfb_r5_retry1`已正常退出；
- log：`runs/logs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_formal_fresh0to50_r5_b20_c5353f3_retry1_20260813.log`；
- output是fresh root，不从world6中止run、旧semantic checkpoint或profile checkpoint迁移任何state；
- storage preflight：`/data1` user quota约`493 GiB / 1 TiB`，两个约185MiB checkpoint加run metadata远低于余量。

训练已完整结束：`metrics.jsonl`有50条，`completion.json.completed_macro=50`，macro25/50两个checkpoint均有
完整manifest，tmux/torchrun正常退出；总耗时`2138.7067s`。macro50 functional/consistency loss=
`.115038/.005875`、gradient norm=`.050324`且K1--K4各6，只证明训练合同健康，不是性能结论。

## 4. Sealed profile evidence

- canonical implementation：`3866f50`；runtime profile seal：`c5353f3`；完整CPU回归=`372 passed`；
- world5 full24 B20 profile：`39.4234s/macro`，相对matched semantic-address world5=`1.00476x`，K1--K4各6，
  loss/gradient finite，峰值allocated/reserved=`39.093/45.445 GB`；
- fixed validation8x4 deployment B8/B16/B32 LoRA/s=`.97732/.96489/.96513`，全部覆盖最长视频且0 OOM，正式
  evaluator锁B8；
- source policy、normalization、24/8/8 split、official LIBERO preprocessing和38-target topology不变。

## 5. Completed K1 strict400 and terminal analysis

正式root：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_correct400_noreplacement_seed7_macro0050_trainr5_evalr6_c5353f3_gpu01_retry1_20260813`

- frozen eval worktree：`/data1/user/ymdai/worktrees/EMBER-direct-family-b-eval-c5353f3`，clean detached `c5353f3`；
- host/devices：gpu01物理GPU`0,2,4,5,6,7`，六卡；每卡3个persistent rollout replicas、1个Writer generator；
- arm：validation8×50、correct K1、without-replacement seed7、macro50 single checkpoint、generation B8；
- 原tmux `ember_dfb_correct400_m50_retry1`已正常退出；
- log：`runs/logs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_correct400_noreplacement_seed7_macro0050_trainr5_evalr6_c5353f3_gpu01_retry1_20260813.log`；
- 400-entry LoRA cache估算peak新增`535,986,176` bytes，仍在已检查quota内；
- 第一份无`retry1`的eval root只完成prepare；启动瞬间GPU1被他人新占约34GB，原子preflight拒绝启动，因而没有
  Worker、LoRA cache或rollout结果。它不得冒充失败实验或活动root。

完成边界：

- `completion.json`且`completed_macro=50`；
- `metrics.jsonl`覆盖macro1--50；
- macro25与macro50 checkpoint完整；
- launcher/torchrun正常退出，无failure artifacts或nonfinite；
- macro50 checkpoint schema、world5 rank states和run contract一致。

- 72/72 shards complete，18/18 worker return code为0；
- 400 rows及`(suite, task, init_state)` keys均唯一，无failure artifact；
- strict=`102/400`、breadth5，per-task=`0/1/40/11/0/43/7/0`，per-suite=`1/51/43/7`；
- top3=`94/102`，说明能力仍集中；相对semantic101为`82 retained/20 gained/19 lost`、churn39；
- 相对Dynamic-K100为`79/23/21`、churn44；相对old134为`80/22/54`、churn76；相对compiler138为
  `79/23/59`、churn82；相对online128为`79/23/49`、churn72；
- 相对v6-fast143的per-task差=`0/-2/-6/-26/0/+7/-13/-1`。

exact effective-BA对照中，task-mean offdiag cosine从semantic的`.77947`降到Direct-B的`.74895`，但K1
closed-loop只`101→102`且breadth`6→5`。因此删除family hidden/GELU只轻微缓解几何压缩，不能解决
Program到policy direction、held on-policy usefulness或shared capability coexistence。该方法按门终局non-pass，
不resume到100、不做小超参sweep、不补K1五臂controls。

分析artifact：同一root下`benchmark_comparison.json`与`effective_ba_task_geometry_comparison.json`。分析工具同时
修复了一个窄工程问题：registered historical Dynamic-K families应使用各自episode schema，而runtime通用
writer-input事实不应被方法名措辞误拒；该修复不改变任何raw row、score或科学合同。

## 6. Completed K4 nested video-dose adjudication

formal K4 root：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_k4_correct400_noreplacement_seed7_macro0050_trainr5_evalr5_73b9514_gpu01_retry1_20260813`

- frozen worktree clean detached `73b9514`；gpu01物理`2,4,5,6,7`，5个Writer generators、15个persistent
  rollout workers；启动时同节点只有这5张卡适合使用，未等待或跨节点拼卡；
- validation8×50、correct、without-replacement seed7、显式K4、每condition总frame budget64；K1视频是K4集合
  的严格nested第一个元素；generation B8来自sealed K4 profile；
- 60/60 shards、400/400 rows、15/15 workers exit0、0 failed；wall=`1098.3835s`，overall=`.36417`
  episodes/s；Writer 400 entries由50个B8 batches一次生成，无重复forward；
- strict=`98/400`、breadth5，per-task=`1/0/42/8/0/41/6/0`，per-suite=`1/50/41/6`，top3=
  `91/98=92.86%`；相对v6-fast143逐task差=`+1/-3/-4/-29/0/+5/-14/-1`；
- nested K1→K4为`80 retained/18 gained/22 lost/280 retained failure`，net`-4`、churn40、Jaccard
  `.6667`、exact McNemar `p=.635828`；没有新task被解锁；
- 全400对effective-BA K1→K4 cosine mean/median=`.98787/.99225`、relative-L2 mean/median=
  `.15325/.13764`、norm ratio mean=`.99876`，排除K4能量坍缩；gained与lost没有可分离的有用方向；
- validation每task前4 states中，same-task centered variance/sample从K1 `.021674`降到K4 `.003438`
  （约`6.3x`），task-mean K1→K4 cosine=`.99604`；跨task mean offdiag只从`.74895`轻降到`.73816`。

因此set aggregator和动态K训练合同本身已工作：更多视频成功滤掉单demo偶然性，却主要稳定同一个窄而错误的
task mean。当前最早失效接口是set之前的task-grounded高层evidence/Procedure及其task-level functional credit，
不是视频数量、集合不变性、mapper、rank或LoRA能量。正式分析为同一root下
`k1_k4_nested_dose_analysis.json`。

第一次formal root（commit `ca28da8`）在任何GPU工作前因resume/start adapter inspection漏传`evaluation_k=4`
而fail closed；queue保持60 pending、无rollout/cache。`73b9514`只修复该窄工程合同并有回归测试，不改变Writer、
checkpoint、schedule或科学panel。

## 7. Completed Task-Grounded Visual-Value implementation and profile

- canonical实现`9d43e82`；数学等价吞吐优化`690dea5`；均已push，完整CPU=`378 passed`；
- 唯一主变量是同一次joint forward中的task-token query/raw-patch Value，按真实输入顺序重算visual D/G，再以
  semantic address + layer/rank route写入原18x8 Program；无额外backbone forward、prediction/negative loss、
  expert target、reward或language-only Value；
- 全新config/launch/checkpoint/evaluator schema，不能误resume Direct-Family-B或profile state；
- 首次world4 profile=`58.2544s/macro`，同gpu02物理0--3、同world4/B20/K调度的Direct-Family-B matched为
  `46.2242s`，即`1.2603x`，超过`1.15x`门，因而没有启动formal；
- 只截断frozen prefix无用backward、合并bias-free evidence projection GEMM和共享visual reader调用；优化后
  `49.0775s/macro`，matched=`1.061727x`，通过门。K1--K4各6，functional/consistency=`.156108/.009484`，
  gradient norm=`.068287`，峰值allocated/reserved=`39.303/45.561GB`，无OOM且完整checkpoint/completion；
- profile只证明机制与吞吐，不是closed-loop成绩。正式训练必须fresh macro0，不加载任何profile checkpoint。

## 8. Retained canonical assets

- source policy：
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`；
- tokenizer：`models/tokenizers/openpi/paligemma_tokenizer.model`；
- target data：`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`；
- split：`configs/libero_24_8_8_v1/`；
- LIBERO assets：`.env.local`中的`EMBER_LIBERO_ASSETS_ROOT`；
- task experts：`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`中的统一step2000；
- historical config由commit与formal root保存；当前active config与状态见§13；
- historical exact roots与逐方法negative boundaries：`docs/research_history.md`和retained formal artifacts。

## 9. Completed Visual-Value formal curve

K4裁决已经完成，不能再通过增加K、调set或mapper救这个checkpoint。下一fresh design保留动态K1--K4、真实
image/language/Action-probe context中的8个memory tokens、逐video保序、跨video置换不变set、20×8 M2P、
direct family-B与完整rank8 LoRA；只改变set之前的evidence owner：让冻结VLM中由exact language定位的真实视觉
内容成为LoRA Value，并同时提供稳定Semantic Core与有向视觉transition，而不是只让absolute Action memory作Q。
该设计必须继承v5.2/v6已经验证的task-grounded patch/value原则，又避免恢复其多套旧前端或language-value旁路。
正式公式、机制门、吞吐门和macro0→200裁决见
`docs/action_forecast_writer_dynamic_k_task_grounded_visual_value_design.md`。

正式fresh root：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_formal_fresh0to50_r3_b20_caa2e30_gpu01_20260813`

- frozen worktree：`/data1/user/ymdai/worktrees/EMBER-tgvv-formal-caa2e30`，clean detached
  `caa2e3045fee7d4a5a2bbcfcc126fad9ec61832f`且commit已push；
- host/devices：gpu01物理GPU`4,5,6`，world3；launch前双节点live检查后，这三张卡是同节点当时真正适合且
  稳定的设备，不等待凑6卡；
- fresh macro0、B20、stop-after50、checkpoint25/50、num_workers0；不加载profile或Direct-Family-B state；
- `NCCL_P2P_DISABLE=1`，三rank均映射到physical GPUs 4--6对应的NUMA1；
- 首个macro为`65.1458s`，K1--K4各6，functional/consistency=`.156108/.009484`、gradient norm=
  `.068287`，三卡约100%利用率且峰值allocated/reserved=`39.298/45.546GB`；只证明正式合同已健康运行；
- tmux：gpu01 `ember_tgvv_formal_caa2e30`；精确命令与环境以该root的`run_contract.json`为准。

fresh 0→50现已完整结束：`metrics.jsonl`恰好50条、macro1--50唯一连续，K1--K4全程各6，全部loss/gradient/
seconds finite；macro25/50两个checkpoint均为world3且5个声明文件尺寸完整；`completion.completed_macro=50`，
总耗时`3398.5443s`。macro50 functional/consistency/gradient=`.114084/.005499/.049901`，只作训练健康证据。

同一root已从macro50以相同gpu01物理`4,5,6`和world3 exact-resume到100，tmux
`ember_tgvv_resume50to100_caa2e30`；首个续训macro51已完成，证明optimizer/scheduler/RNG/cursor恢复链工作。

macro25 checkpoint已完整写出且训练继续。用该checkpoint在gpu02物理GPU1完成K1部署定标：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_k1_writer_generation_profile_val8x4_correct_gpu02p1_caa2e30_macro0025_retry1_20260813`

- fixed validation8×4、longest-first B8/B16/B32 LoRA/s=`.984266/.976097/.971736`；三者均stable、包含最长64帧、
  0 OOM，peak reserved=`12.973/13.451/13.455GB`；按规则锁B8；
- profile使用clean frozen `caa2e30`和同一正式macro25 checkpoint，只决定相同生成图的deployment batch；不读取
  closed-loop结果、不选择checkpoint；
- 第一份无`retry1` root在任何GPU forward前因误传LIBERO assets旧路径fail-closed；没有profile结果，不得冒充
  方法失败或正式定标。

macro50完成后立即做K1 strict paired correct400。历史v6-fast macro50也只有106，因此不以单个50点低于120提前
杀死整个0→200裁决；按design继续100/150/200并分析相邻checkpoint共同积累。

macro50 K1 strict400已从clean frozen `99c2323`完整结束：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_correct400_noreplacement_seed7_macro0050_trainr3_evalr5_99c2323_gpu02_20260813`

- gpu02物理`1,2,3,4,6`，5个Writer generators、15个persistent rollout workers；validation8×50、correct K1、
  without-replacement seed7、generation B8；
- GPU0约36GB、GPU5/7高util而未选；所选1--4约349MiB/0% util，GPU6约4.9GB/0% util，符合owner允许安全
  低util共驻的边界，不等待第6卡；
- evaluator准入窄修复`99c2323`只允许util≤10%、used≤8GiB、free≥32GiB，超门仍fail-closed；全量CPU
  `383 passed`；不改变scientific panel、Writer、随机性或rollout；
- 60/60 shards、400/400 rows、15/15 workers exit0，无failure artifact；wall=`1100.373s`；
- strict=`88/400`、breadth5、per-task=`4/0/34/2/0/41/7/0`、per-suite=`4/36/41/7`，top3=
  `82/88=93.18%`；
- 相对Direct-Family-B 102为`74 retained/14 gained/28 lost`、churn42、McNemar `p=.0436`；相对semantic101、
  Dynamic-K100、old134、compiler138、online128的净差分别为`-13/-12/-46/-50/-40`；
- 相对v6-fast143的per-task差=`+4/-3/-12/-35/0/+5/-13/-1`；能力仍高度集中，Object3由Direct-B的11降到2；
- exact effective-BA前4 state/video对照：task/video SNR `16.34→19.05`、task-mean offdiag cosine
  `.74895→.70731`、norm mean `136.64→126.47`，但paired BA cosine mean`.83086`、relative-L2 mean`.58417`；
  functional loss轨迹与Direct-B几乎相同。新增Value确实进入LoRA且形成更task-specific方向，但这些方向没有
  提高held closed-loop；完整分析artifact为同root下`benchmark_comparison.json`和
  `effective_ba_task_geometry_comparison.json`。

50→100已正常完成：metrics共100条、macro1--100连续finite，macro100 checkpoint含world3完整trainer/rank/
Writer state，`completion.completed_macro=100`。当前100→150 tmux为`ember_tgvv_resume100to150_caa2e30`；macro101
准确从`task_visit=100`恢复，K1--K4各6、`64.218s`、finite。

macro100 K1 strict400 root已完整结束：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_correct400_noreplacement_seed7_macro0100_trainr3_evalr5_99c2323_gpu02_20260813`

- clean frozen evaluator仍是`99c2323`；validation8×50、correct K1、without-replacement seed7、B8不变；
- gpu02物理`1,2,3,4,6`，5个Writer generators、15个persistent workers；启动preflight仍为1--4约350MiB/0%
  util、GPU6约4.9GiB/0% util，预计新增`535,986,176` bytes；
- 60/60 shards、400/400 rows、15/15 workers exit0，无failure artifact；wall=`1116.200s`；
- strict=`86/400`、breadth6、per-task=`1/3/34/0/0/35/12/1`、per-suite=`4/34/35/13`，top3=
  `81/86=94.19%`；
- 相对macro50的88为`62 retained/24 gained/26 lost`、churn50、net`-2`；两点episode-level union=`112`，
  single-best gap=`24`。Object1内部7 gain/7 lost但总数仍34；Goal6净`-6`、Long1净`+5`，是能力换手而非共同积累；
- 相对Direct-Family-B102为`63 retained/23 gained/39 lost`、churn62、net`-16`；相对old134/compiler138/
  online128净`-48/-52/-42`；相对v6-fast143逐task差=`+1/0/-12/-37/0/-1/-8/0`；
- macro50→100前4个严格配对state/video的implicit effective-BA：cosine`.80856`、relative-L2`.69630`、norm
  ratio`1.17376`；action部分cosine`.73896`、relative-L2`.73396`、norm ratio`.98674`。各task变化与净得失不
  单调，正式artifact为同root下`benchmark_comparison.json`与`checkpoint_transition_geometry_macro0050_to0100.json`。

macro100→150随后以相同world3 topology正常完成：metrics共150条连续finite，macro150 checkpoint含完整
`writer.safetensors`、`trainer_state.pt`和3个rank state，`completion.completed_macro=150`。随后从该checkpoint
exact-resume 150→200，macro151准确从`task_visit=150`恢复，K1--K4各6、`64.127s`、finite。

macro150 K1 strict400已在clean frozen evaluator `99c2323`完整结束：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_correct400_noreplacement_seed7_macro0150_trainr3_evalr5_99c2323_gpu02_20260813`

启动前双节点live检查后，gpu02仍只有物理`1,2,3,4,6`五张合适卡：1--4约350MiB/0% util，6约4.9GiB/0%；
GPU0已占36.6GiB，GPU5占30.7GiB且100% util，GPU7占19.7GiB且91% util，均不适合约13.5GiB峰值的Writer
生成；不跨节点拼gpu01单卡。400/400 LoRA由5个generators写出，随后15个persistent rollout workers完成
60/60 shards、400/400 rows且全部exit0；wall=`1034.173s`。

- strict=`86/400`、breadth6、per-task=`1/0/36/1/1/40/7/0`、per-suite=`1/37/41/7`，top3=
  `83/86=96.51%`；
- macro100→150严格同episode为`62 retained/24 gained/24 lost`、churn48、net0、Jaccard`.56364`；Goal6净`+5`
  与Long1净`-5`互换，Object1虽净`+2`仍有8 gain/6 loss；
- macro50/100/150 success union=`125`，single-best仅88、gap37，三点共同成功仅53；Long1为`7→12→7`、
  union22但三点intersection为0，不能解释为稳定平台；
- 相对Direct-Family-B102为`68 retained/18 gained/34 lost`、churn52、net`-16`；相对old134/compiler138/
  online128净`-48/-52/-42`；相对v6-fast143逐task差=`+1/-3/-10/-36/+1/+4/-13/-1`；
- macro100→150 first4×8 implicit effective-BA cosine`.87843`、relative-L2`.52839`、norm ratio`1.08149`；
  action为`.85946/.62125/1.14658`。正式artifact为同root下`benchmark_comparison.json`、
  `checkpoint_transition_geometry_macro0100_to0150.json`与`checkpoint_curve_success_sets_macro0050_0100_0150.json`。

同topology训练与macro200评测已经完成；结果见§11。best未达125，按预注册门不续到400。

## 10. Fixed-A reachable-subspace diagnostic

macro100 strict root内新增只读CPU artifact：

`fixed_a_reachable_subspace_analysis.json`

它用低秩QR精确计算`W=B@A`的可达右子空间能量，不构造dense BA；样本为old134的8个validation tasks×first4
states共32套native rank16 LoRA，以及24个统一step2000 task experts：

- old134：逐样本最优rank8保留`.99999946`总能量，当前固定随机A只保留`.0195042`；q/v/action分别为
  `.021978/.012768/.262524`；
- train24 experts：逐expert最优rank8保留`.998094`，当前固定随机A只保留`.184501`；
- 每target在全部24 experts上拟合一套最优共享rank8 A，可在train experts保留`.940630`，但应用到old134 held
  LoRA只保留`.068108`，q/v/action为`.079511/.037639/.422943`。

因此rank8本身不是这里的首要容量瓶颈；当前direct-family-B把所有task/video限制在同一随机A行空间，而换一套
train24静态A也没有足够held外推。历史v6-fast143使用完整task-conditioned A/B，提供结构先例但没有隔离本变量。
该offline几何不能选择checkpoint或证明closed-loop收益。若且仅若当前完整0→200曲线non-pass，下一候选才是保留
全部evidence、temporal/set、20×8 M2P、rank8和B20，只给同一projected Program增加bias-free direct family-A
readout；不恢复旧nonlinear family hidden、expert bank、rank sweep或旧前端。

macro150 strict root内的`fixed_a_same_task_video_loo_analysis.json`进一步对old134的每个validation task用3条
视频拟合rank8 A行空间并leave-one-video-out测试第4条：overall保留`.9997255`，q/v/action分别为
`.9997540/.9996504/.9992049`，8个task逐task均高于`.99916`。这说明强Writer中的有效A行空间不是任意
video-specific噪声，而是跨same-task视频极稳定、同时不能由一套train24静态A向held task外推的task-level结构。
因此候选A必须由完整language+video Program生成并继承现有shared-vs-singleton Program一致性，不能变成task ID、
language-only route或逐视频自由漂移；B及其余Program仍承载正确视频证据。该结论仍只界定结构，不替代closed-loop。

完整曲线后的推进顺序保持：真实结果 -> 相邻checkpoint churn与接口分析 -> 一个主要因果变量 -> authority ->
canonical实现/机制/吞吐 -> fresh训练 -> single-checkpoint strict评测。memory token、rank8和Dynamic-K都是方法
变量，不是Goal。

## 11. Macro200 terminal result and Full-Factor successor

Visual-Value macro200 K1 strict root：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_correct400_noreplacement_seed7_macro0200_trainr3_evalr4_99c2323_gpu02_20260814`

- 48/48 shards、400/400 rows、12/12 workers exit0、0 failed，wall=`1244.5794s`；
- strict=`96/400`、breadth6、per-task=`1/0/37/2/0/42/13/1`、per-suite=`1/39/42/14`；
- top3=`92/96=95.83%`；150→200=`71 retained/25 gained/15 lost`、churn40；
- 相对Direct-B102=`74/22/28`，old134=`68/28/66`，compiler138=`73/23/65`，online128=`74/22/54`；
- 完整曲线`88/86/86/96`，best远低于125，按门终局non-pass；不resume到400、不补五臂。

active Full-Factor authority：
`docs/action_forecast_writer_dynamic_k_task_grounded_full_factor_design.md`。

当前canonical graph：

```text
exact language + dynamic K ordered videos
-> joint image/language/Action probes + 8 memory tokens
-> task-grounded visual D/G -> causal temporal -> set -> 20x8 M2P
-> shared 256->1024 project
-> 4 direct dynamic-A residual heads + 4 direct dynamic-B heads
-> one complete 38-target rank-8 LoRA
```

所有A/B heads zero-init，step0仍exact identity；首次effective-BA credit只打开B，B非零后A与上游获得真实
functional gradient。未改变数据、K schedule、frame budget、objective、optimizer、rank或source policy。canonical
config=`configs/pi05_as_writer_dynamic_k_task_grounded_full_factor_rank8_v1.json`。live profile root：

`runs/outputs/pi05_dynamic_k_task_grounded_full_factor_rank8_profile_r4_b20_d58e3f8_gpu01_20260814`

- gpu01物理`1,4,5,6`，world4，完整full24 B20 macro1=`47.440897s`；
- K1--K4各6，functional/consistency=`.156107822/.009484296`，gradient norm=`.068287469`；
- peak allocated/reserved=`39,332,965,888/45,562,724,352` bytes，无OOM/nonfinite，completion macro1；
- 与fixed-A matched world4的`49.0775s`相比没有吞吐回退；首macro三项数值完全一致，符合zero-init staged
  full-factor合同；
- formal config已seal，下一步从新的clean pushed commit做fresh macro0→50，不加载profile checkpoint。

macro25 checkpoint完整写出后，用clean frozen `0d6cda7`在gpu02物理GPU1完成K1部署定标：

`runs/outputs/pi05_dynamic_k_task_grounded_full_factor_rank8_k1_writer_generation_profile_val8x4_correct_gpu02p1_0d6cda7_macro0025_20260814`

- B8/B16/B32 LoRA/s=`.979553/.975323/.972106`，三者均stable、最长64帧、0 OOM/nonfinite；
- 按highest-throughput规则锁B8；peak reserved分别=`12.973/13.453/13.474GB`；
- profile只决定同一Writer graph的evaluation batch，不选择checkpoint或读取closed-loop outcome；formal训练继续到50。

## 12. Full-Factor terminal result and V6 Slot-Set successor

formal training root：

`runs/outputs/pi05_dynamic_k_task_grounded_full_factor_rank8_formal_fresh0to50_r4_b20_0d6cda7_gpu01_20260814`

- gpu01物理`1,4,5,6`、world4，macro1--50完整finite，K1--K4每macro各6；
- macro50 functional/consistency/grad=`.114267/.002160/.307680`，总耗时`2372.278s`；
- macro25/50 checkpoint和completion完整；全量CPU在正确LIBERO assets路径下=`383 passed`。

K1 strict root：

`runs/outputs/pi05_dynamic_k_task_grounded_full_factor_rank8_correct400_noreplacement_seed7_macro0050_trainr4_evalr5_c0501bc_gpu02_20260814`

- gpu02物理`1,2,3,4,6`，5 generators + 15 persistent rollout workers；60/60 shards、400 rows、全部exit0；
- strict=`91/400`、breadth5、per-task=`4/1/38/0/0/37/11/0`、per-suite=`5/38/37/11`，top3占
  `86/91=94.51%`；
- vs fixed-A macro50 88=`70/21/18`，vs fixed-A best96=`69/22/27`，vs Direct-B102=`72/19/30`；
- vs old134/compiler138/online128分别净`-43/-47/-37`；不满足125门，不resume到100。

matched first4 states/task的factor定位：Full-Factor vs fixed-A raw A cosine/norm ratio=`.735154/1.376207`，raw B=
`.248553/.062232`，effective BA=`.058529/.244792`。offline loss几乎相同而完整factor学成larger-A/tiny-B的弱更新，
所以最早失败接口是functional surrogate下的factor credit/gauge allocation，不是训练时长或rank8理论容量。

active successor是`docs/action_forecast_writer_v6_dynamic_slot_set_bridge_design.md`。它以v6-fast为baseline：每条
video独立形成原生Core/Procedure和320 slots，只沿K轴新增mean-backbone + selected-centered-residual集合层；K1
结构上严格等于原v6，K>1才训练，最后原生factor heads只运行一次。首轮warm start只作机制开发，成功后仍需
fresh recipe。

canonical实现已完成：复用native v6 owner，只新增约197k参数的`PolicySlotSetFusion`，并删除退役rank8
backbone-memory/memory-program/LoRA-mapper active路径。全量CPU=`370 passed`；真实GPU smoke中K1的76个LoRA
tensors逐元素等于native v6，base无梯度，K2/K4 video换位最终LoRA/Program只出现正常BF16低位差异，真实video
倒序Program mean abs变化=`.21703`，峰值reserved=`19.27GB`。

full24 profile root：

`runs/outputs/pi05_v6_dynamic_slot_set_bridge_profile_r5_b20_07e9477_gpu01_20260814`

- clean detached `07e9477`，gpu01物理`0,1,4,5,6` world5，24/24 tasks，K1--K4各6；
- 所有selected videos完整保留，最长condition=`323` frames；`30.7422s/macro`，functional=`.101173`，gradient
  norm=`1.7725e-6`，peak allocated/reserved=`36.48/40.75GB`，0 OOM/nonfinite；
- checkpoint、completion和exit0完整；profile state不进入formal训练；
- 首次`8278f74`尝试在任何step前因worktree-relative v6 asset路径fail-closed；`07e9477`改为runtime显式asset root，
  全量CPU仍`370 passed`，科学合同未变。

formal training root：

`runs/outputs/pi05_v6_dynamic_slot_set_bridge_formal_fresh0to25_r5_b20_26ebc43_gpu01_20260814`

- clean detached `26ebc43`，gpu01物理`0,1,4,5,6` world5，fresh macro0→25；
- 25/25 metrics、completion和macro25 checkpoint完整，总耗时=`750.446s`；
- 末步functional/gradient=`.095644/5.152e-6`，K1--K4每macro各6，peak reserved=`40.75GB`，0 OOM/nonfinite。

K4 generation profile root：

`runs/outputs/pi05_v6_dynamic_slot_set_bridge_k4_writer_generation_profile_val8x4_correct_gpu01p2_26ebc43_macro0025_20260814`

- fixed validation8x4 correct panel，B8/B16/B32=`.224364/.224185/.224350 LoRA/s`，全部stable、0 OOM；
- peak reserved约`12.95/12.97/13.01GB`，最长condition=`226` frames，按最高实测吞吐锁B8；
- 该历史run随后strict130并终局，详见本节后续；这里的“下一步”只属于当时时点。

## 13. V6 Dynamic Slot-Set terminal result and Shared-Core Procedure-Set successor

K4 strict root：

`runs/outputs/pi05_v6_dynamic_slot_set_bridge_k4_correct400_noreplacement_seed7_macro0025_trainr5_evalr6_34c0431_gpu01_20260814`

- clean detached `34c0431`，gpu01物理`0,1,2,4,5,6`，6 Writer generators + 18 persistent rollout workers；
  400/400 LoRAs、72/72 queue jobs、400 rows、18/18 workers均完整exit0；
- wall=`1182.222s`、rollout-only=`677.514s`；strict=`130/400`、breadth6、per-task=
  `1/2/48/32/0/34/13/0`、per-suite=`3/80/34/13`、top3=`114/130=87.69%`；
- vs old134=`117 retained / 13 gained / 17 lost`、net`-4`、churn30、Jaccard`.795918`、paired
  McNemar p=`.584665`；每task净变化=`+1/-3/0/-2/0/-1/+2/-1`；
- count-only相对compiler138/online128/Full-Factor91/v6-fast143分别为`-8/+2/+39/-13`；
- 未超过old134且breadth低于7，按门终局non-pass；不resume50、不补五臂、不扫K/LR/seed/temperature。

matched nested-dose分析保存在同root的
`k1_old134_to_k4_slot_set_nested_dose_analysis.json`：

- 400套LoRA的K4 vs K1 effective-BA cosine mean/median=`.998690/.999275`，relative-L2 mean/median=
  `.046910/.040562`，norm ratio mean=`.998592`；q/v/action结论一致；
- first4 states/task的task mean K1→K4 cosine=`.999832`；same-task centered variance/sample从`.002281`降到
  `.000246`，约`9.26x`；跨task mean offdiag cosine只从`.49935`到`.49749`；
- retained success的BA变化反而小于retained failure（`.03745` vs `.05165`），gained/lost=`.04881/.03998`，没有
  可用的scalar correction-size分界。

因此上一轮不是读取失败：多视频确实稳定了同task nuisance，也基本保住old134支持；但在每条video已经分别经过
完整compiler之后，set只能形成约4.7%的邻域修正，无法改变高层task mean。最早可检验接口前移到“Core/Procedure
何时跨video共享”，而不是继续改post-compiler set或放大修正。

active successor为`docs/action_forecast_writer_v6_shared_core_procedure_set_bridge_design.md`。唯一主变量是把相同
197120参数集合层前移：native Core reader先从无序per-video Core union产生一个shared Core；每条有序Procedure以
该shared Core独立读出；Procedure-set再置换不变聚合；原生AdaLN/post-fusion/factor heads只运行一次。没有frame
拼接、phase alignment、LoRA平均、rank变化、negative、expert或RL。

canonical实现已原位替换旧schema/config/runtime，旧路径由commit`34c0431`与formal artifacts保存。64项定向CPU
测试通过：compiler阶段化与旧图逐tensor相等，K1在任意set参数下严格等于native v6，K>1 video set换位不变且
video内倒序敏感，base无梯度、Procedure-Set梯度非零；全量CPU=`371 passed`。

真实机制smoke在gpu01物理4完成：K1在zero/nonzero set output下均与native v6的76 tensors逐元素相等；倒序
Program mean abs=`.217034`；trainable=`197120`，rank16 LoRA=`1,287,168`参数，base无梯度，peak reserved=
`19.367GB`。

full24 B20 profile root：

`runs/outputs/pi05_v6_shared_core_procedure_set_bridge_profile_r6_b20_97c0de2_gpu01_20260814`

- clean detached `97c0de2`，gpu01物理`0,1,2,4,5,6` world6；macro1/2均K1--K4各6；
- wall=`26.01095/24.24948s`，functional=`.10118184/.09570904`，gradient=`4.3242e-6/6.2963e-6`；
- 最长condition=`323`帧且全部未截断；peak allocated/reserved=`36.495/40.758GB`；0 OOM/nonfinite；
- macro1→2 query/key delta norm=`4.1365e-5/3.9905e-5`，证明首步后完整set credit展开；
- completion macro2、两组checkpoints、6 rank states、exit0完整。profile checkpoint不进入formal。

profile evidence现已seal入config。clean detached `502618b`随后完成正式fresh macro0→25：

`runs/outputs/pi05_v6_shared_core_procedure_set_bridge_formal_fresh0to25_r6_b20_502618b_gpu01_20260814`

- gpu01物理`0,1,2,4,5,6` world6；25/25 metrics、macro25完整checkpoint、completion、exit0；
- 总耗时=`662.7296s`，每macro K1--K4各6，loss first/last=`.10118184/.09565479`；
- gradient范围=`3.5721e-6..1.3584e-5`，peak reserved=`40.758GB`，0 OOM/nonfinite。

macro25 K4 deployment profile root：

`runs/outputs/pi05_v6_shared_core_procedure_set_bridge_k4_writer_generation_profile_val8x4_correct_gpu01p4_502618b_macro0025_20260814`

- gpu01物理4；固定val8×4 correct最长优先面板；B8/B16/B32分别=`.2233579/.2233132/.2233235 LoRA/s`；
- 三者stable、0 OOM，峰值reserved约`12.95/12.97/13.01GB`；按最高吞吐锁B8；
- profile只裁决deployment batch，不提供closed-loop方法证据。

strict K4 root：

`runs/outputs/pi05_v6_shared_core_procedure_set_bridge_k4_correct400_noreplacement_seed7_macro0025_trainr6_evalr6_64c91a4_gpu01_20260814`

- clean detached `64c91a4`；gpu01物理`0,1,2,4,5,6`，6 generators + 18 persistent workers；400/400 LoRAs、
  72/72 shards、400 rows、18/18 workers exit0；wall=`1165.9373s`、rollout-only=`674.9167s`；
- strict=`139/400`、breadth6、per-task=`1/4/46/34/0/36/18/0`、per-suite=`5/80/36/18`、top3=
  `116/139=83.45%`；
- vs old134 strict paired=`118 retained / 21 gained / 16 lost / 245 both-fail`，net`+5`、churn37、
  Jaccard`.76129`、McNemar p=`.511376`；per-suite net=`0/-2/+1/+6`；
- vs matched post-compiler K4 130 strict paired=`118/21/12/249`，net`+9`、churn33、p=`.162756`；
- count-only vs v6-fast143/compiler138/online128分别=`-4/+1/+11`；breadth低于7，终局停止。

同root封存三份分析：

- `k1_old134_to_k4_shared_core_procedure_set_nested_dose_analysis.json`：全400 K4/K1 BA cosine mean=`.9985641`、
  relative-L2=`.0477494`、norm ratio=`1.002122`；first4/task variance `.00228114→.00023539`约`9.69x`，
  task-mean cosine=`.9998256`；
- `postcompiler130_to_sharedcore139_paired_analysis.json`：严格episode/video/RNG一致；当前相对上一K4只需
  `.010965` BA变化便净增9，证明边界位置有效；
- `trained_vs_zero_procedure_set_first4_analysis.json`：只归零trained `Procedure-Set.output`，当前→zero BA
  relative-L2 mean=`.0009181`、task-mean=`.0005745`；K1→zero为`.0396742/.0169820`。所以训练层几乎没有贡献，
  无参数shared-Core union才是本轮增益来源。

以上是Semantic-Core Set实现完成时的历史状态；其后续profile、formal、strict135与Common-Value successor状态只取
本文§1，不得从这段历史“下一步”恢复执行。

## 14. SFMC completed formal result and terminal analysis

SFMC formal训练来自clean pushed/frozen `899418087aee9f7dd5c51045aa190ac7481dcf3a`，gpu01物理
`5/6/7`、world3、fresh cycle0→1、K4、四个互斥correct views、Nmc4、B8。训练root：

`runs/outputs/pi05_v6_lpcp_semantic_factor_memory_commitment_formal_cycle0to1_r3_k4_views4_nmc4_b8_8994180_gpu01_20260815`

- 24 tasks、48 paired states、96 rollouts；reference/candidate=`34/34` successes，candidate/reference-only=
  `4/4`，both-success/failure=`30/10`；8 active tasks覆盖Spatial/Object/Goal；
- 32个credit conditions全部LoRA/factor gradient finite/nonzero，128个unique videos；8/8 family maps更新；
- family-map delta从action-in-A的`4.12e-7`到q-B的`4.98e-5`；semantic query/basis keys仅约`1.7e-9`，
  两个norm仅`5.96e-8`，符合zero-init staging但说明cycle1 router尚未获得material学习；
- 三项deployment probe effective-BA=`0/1.056e-8/0`，fixed-action=`.003839/.002210/.002808`；连续训练图有
  action响应，但native LoRA写出已处于量化边界；
- cycle=`920.555s`，为CV-CSD matched cycle的`1.06616x`；三rank任务=`8/9/7`、active=`2/3/3`、记录时长
  max/min=`1.06531x`；peak allocated/reserved=`36.505/40.767GB`，无禁读、OOM、nonfinite或watchdog。

同一cycle1 single checkpoint的strict root：

`runs/outputs/pi05_v6_lpcp_semantic_factor_memory_commitment_cycle1_k4_correct400_noreplacement_seed7_trainr3_evalr3_8994180_gpu01_20260815`

- 400/400 LoRAs、42/42 shards、400 rows、9/9 persistent workers exit0；panel wall=`2011.670s`，有效吞吐=
  `.19884 episodes/s`，worker完成shards最大差2；
- strict=`144/400`、breadth7、per-task=`1/3/47/36/0/38/18/1`、per-suite=`4/83/38/19`、top3=
  `121/144=.84028`；
- 相对LPCP143严格=`128 retained / 16 gained / 15 lost / 241 both-fail`、churn31、net`+1`、Jaccard
  `.805031`、p=`1`；Long1贡献10 gains/8 losses，Spatial/Object/Goal suite净=`-1/0/0`，没有共同稳定增益；
- 相对AS139/PCSD135/CV-CSD134分别=`121/23/18`、`119/25/16`、`124/20/10`；count-only相对
  v6-fast143/old134/compiler138/online128=`+1/+10/+6/+16`；
- cycle1的correct、breadth、gained>lost与3-suite non-down通过，但LPCP lost≤10失败；按预注册合同终局，
  不续cycle2、不做same/wrong/shuffled/reversed/no-video或参数小扫。

普通trace identity在该极小改写上发生大数消去，因此终局使用FP64稳定低秩展开
`Δ(BA)=B_candidate·ΔA+ΔB·A_reference`：

- all400 effective-BA relative-L2 mean/median/max=`2.899e-7/1.066e-9/4.428e-6`；255/400样本非零；
- q/v/action非零样本=`249/16/1`，action relative-L2 mean仅`1.65e-13`；raw factors平均只改
  `1.22e-6`元素比例；
- gained/lost的relative-L2 mean=`2.172e-7/1.594e-7`，均为相同数量级的微小阈值扰动，不能区分有用方向；
- first4同task correction pairwise cosine=`-8.10e-6`，mean/sample energy=`.249995`；八个tasks虽至少有一个
  非零q修正，但多数view落在不相交的量化坐标，未形成跨video共同方向；
- SFMC相对CV-CSD仍为BA relative-L2 mean/median=`.000675/.000669`，first4 cosine/energy=
  `.000205/.250154`，几乎就是CV-CSD→LPCP的历史距离，证明SFMC在部署上基本退回LPCP143邻域。

终局最早失败接口是**continuous SFMC hidden residual -> frozen factor W2 -> native public LoRA**：family maps
确实获得reward credit，但cycle1语义router尚未形成，输出又大多低于原生factor局部ULP，只产生稀疏q-family
crossing。144因此是LPCP边界附近的高churn阈值重排，不是稳定145，也不能证明same-task-video鲁棒或视频特异性。
完整终局artifact为`sfmc_cycle1_terminal_analysis.json`；这里的SFMC终局不得恢复或续训。

## Gradient-Open terminal evidence

训练root：

`runs/outputs/pi05_v6_lpcp_gradient_open_semantic_commitment_formal_cycle0to1_r5_k4_views4_nmc4_b8_eb543d3_gpu01_20260815`

strict与终局分析root：

`runs/outputs/pi05_v6_lpcp_gradient_open_semantic_commitment_cycle1_k4_correct400_noreplacement_seed7_trainr5_evalr5_eb543d3_gpu01_20260815`

- train24 cycle1为24 tasks/48 pairs/96 rollouts，candidate/reference=`33/31`、10 active tasks覆盖四suite，
  semantic query delta=`6.9499e-5`，5/5 q/v与3/5 action probes非零，cycle=`581.924s`；
- strict完整400 cache entries、60 jobs、400 rows、15 workers exit0，wall=`1405.667s`、`.28456 rollout/s`；
- correct=`141/400`、breadth7、per-task=`1/3/48/29/0/36/23/1`、per-suite=`4/77/36/24`；
- 相对LPCP143严格=`128 retained / 13 gained / 15 lost / 244 both-fail`、churn28、net`-2`、Jaccard
  `.82051`；相对SFMC144=`124/17/20`、churn37；
- suite相对LPCP为`-1/-6/-2/+7`，Long1 gain由Object3/Goal6/Spatial3 loss换得，breadth与持续失败task均未改善；
- FP64 BA relative-L2 mean=`9.6632e-6`、为SFMC约`33.3x`，q/v/action非零=`400/399/368`；first4
  cross-video cosine=`.0001442`、energy ratio=`.250124`，证明写出打开但共同方向没有形成；
- terminal artifacts为`gosc_cycle1_strict_adjudication.json`、`gosc_cycle1_stable_effective_ba_analysis.json`、
  `gosc_vs_sfmc_stable_effective_ba_analysis.json`和`gosc_cycle1_terminal_analysis.json`。

本架构按correct、lost、net与suite四项失败终局，不resume cycle2、不做六臂或小扫。当前没有active GPU run或
可resume checkpoint。后继CCT authority已经建立：保留LPCP carrier与single-LoRA信息墙，把video memory从
高维Value direction改为language/policy-aligned causal coefficients；其实现与结果必须fresh，不能写回本段历史。

## Causal Coefficient Transport live mechanism evidence

task4 post-update smoke root：

`runs/outputs/pi05_v6_lpcp_causal_coefficient_transport_task4_mechanism_state_b8_3b55feb_gpu02p1_20260815`

- clean pushed `3b55feb`、gpu02物理1、B8，四个互斥K4 conditions共16条视频和64次CFM forward/backward；
- candidate/reference successes=`2/1`，semantic query delta=`1.460305e-4`，q/v/action effective-BA=
  `4.45616e-7/8.76854e-7/2.01053e-8`，fixed-action=`.00267335`；
- cycle=`130.7366s`=`.9870x` GOSC，peak reserved=`40,751,857,664` bytes，无禁读、OOM、nonfinite；
- 旧分析把LPCP+CCT减AS139误标成CCT-only；该v1标签已失效。按exact same-state LPCP重算后的纯CCT
  effective-BA aggregate cosine/energy=`.575776/.681821`；q=`.593590/.695181`、v=`.528289/.646104`、
  action=`.081102/.310853`。修正后仍越过formal aggregate门，故原formal授权结论不变；
- natural→reversed CCT修正cosine=`.014842`、relative-L2=`1.15358`；逐video常量首帧使factor memory与
  transported coefficient norm分别降到natural的`2.42e-5/2.74e-5`。当前结构读取了有向过程，静态输入不能
  伪造同一新增响应；
- `mechanism_analysis.json`只保留错误标签的provenance；正式机制数值取`mechanism_analysis_corrected.json`。
  `natural_constant_analysis.json`仍有效。以上只关闭了train-seen机制否决条件；其后formal与held证据见下节。

## 15. CCT formal result, corrected audit and terminal boundary

formal训练来自clean pushed/frozen `18bd3632cb49174e1fe589d0e8caf9cfc322c954`，gpu01物理
`2/4/5/6/7`、world5：

`runs/outputs/pi05_v6_lpcp_causal_coefficient_transport_formal_cycle0to1_r5_k4_views4_nmc4_b8_18bd363_gpu01_20260815`

- 24 tasks/48 paired states/96 rollouts，candidate/reference=`33/32`、candidate/reference-only=`5/4`，
  9 active tasks覆盖四suite；36 credit conditions、144 unique videos；
- semantic query delta RMS=`6.08551e-5`，q/v probes `4/4`非零、action `2/4`非零、fixed-action `4/4`非零；
- cycle=`577.7288s`，max allocated/reserved=`36.46/40.756GB`，checkpoint/completion完整且0禁读、OOM、
  nonfinite或watchdog；CCT之外624个writer state keys与LPCP macro25逐元素完全相同。

strict root：

`runs/outputs/pi05_v6_lpcp_causal_coefficient_transport_cycle1_k4_correct400_noreplacement_seed7_trainr5_evalr5_18bd363_gpu01_20260815`

- K4 correct400=`142/400`、breadth6、per-task=`1/2/48/31/0/37/23/0`、per-suite=`3/79/37/23`、
  top3=`116/142=.81690`；
- 相对LPCP143严格=`125 retained / 17 gained / 18 lost / 240 both-fail`、churn35、net`-1`、
  Jaccard`.78125`；相对GOSC141=`121/21/20`，相对SFMC144=`127/15/17`；
- 相对v6-fast143逐task count差=`+1/-1/+2/-6/0/+1/+3/-1`；相对old134为`+8`、compiler138为`+4`、
  online128为`+14`；
- breadth6<7、LPCP lost18>15及held four-view cosine/energy门失败。按预注册合同终局：不resume cycle2，
  不做same/wrong/shuffled/reversed/no-video，也不扫axis count、scale、rank、LR或seed。

稳定FP64与loader postmortem：

- all400 CCT/LPCP effective-BA relative-L2 mean/median=`4.665401e-6/4.221081e-6`；gained/lost=
  `3.174026e-6/5.319738e-6`，改写幅度不选择有用方向；
- held first4 pure-CCT aggregate cosine/energy=`7.75e-8/.249999`，q、v、action也都约`0/.25`；
- exact evaluator worker在validation Spatial1 state0逐元素加载全部65,536个非零semantic-query元素，L2=
  `.015578908`，排除checkpoint loader或旧schema遗漏；
- train task4与held state0 transported coefficient RMS=`5.24818e-6/3.21672e-6`，pre-W2 hidden RMS=
  `2.56037e-6/1.50327e-6`，只差`1.63x/1.70x`；pure-CCT BA L2却为`.164125/.000656710`，相差
  `249.92x`。因此CCT强烈拟合train task-language/compiler response，held residual在native BF16 factor边界
  退回LPCP邻域。

终局artifact为`cct_cycle1_terminal_analysis.json`、`cct_cycle1_strict_adjudication.json`、
`cct_cycle1_stable_effective_ba_analysis.json`、`cct_cycle1_trainseen_task4_hidden_postmortem.json`与
`cct_cycle1_validation_state0_live_load_probe_v2.json`。旧`mechanism_analysis.json`只保留provenance，正式机制
结论取`mechanism_analysis_corrected.json`。本轮只否定当前两系数CCT与一轮稀疏selected-success组合；不否定
V6/LPCP、literal memory token、rank8、few-shot、reward credit或生成LoRA。
