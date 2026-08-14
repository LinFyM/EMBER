# EMBER Active Session State

更新时间：2026-08-15。本文是唯一实时实验状态入口；旧文档、Git快照与formal artifacts中的“当前/下一步”只
表示当时时点。稳定目标与owner要求见`docs/current_owner_requirements.md`，历史结果见`docs/research_history.md`。

## 1. Current truth

- 长期Goal处于active：性能继续追求`>150/400`；owner最新接受约145的稳定方法，但必须由相邻single
  checkpoints低换手、same-task-video鲁棒和correct相对negative/no-video的明确因果性共同认证；
- v6-fast与最新V6-LPCP的correct single checkpoint同为143；v6-fast仍是有完整五臂的历史最好=
  `143/135/125/128/129`，LPCP只完成correct且breadth7；
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
- 当前没有active successor或EMBER GPU进程。下一轮先写新的单变量authority：保留V6/LPCP absolute carrier、
  cross-video selected-success credit与完整rank16 LoRA，只把commitment移到policy layer/rank/target ownership附近。
  layer-aligned memory是由本轮证据触发的候选，不得误写成项目goal，也不得用更多views或query/LR/rank/scale小扫；
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
