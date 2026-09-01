# EMBER progress

- 2026-09-01 首版absolute-route quotient S0已从clean pushed detached `6c33760`在gpu01物理5/6完成双task formal，aggregate为
  `runs/outputs/pi05_ecp_event_bank_set_quotient_s0_gate_s110_0d8d901_gpu01p56_retry1_20260901/aggregate.json`，结论`non_pass`。
  task1 correct fit0/fit1/held为`.851/.846/.877`、wrong为`-.372/-.369`，仅fit1差`.0044`；task93为
  `.709/.727/.725`与`-.182/-.189`，correct/held明确未过。两run均自然exit 0、110步与五臂Panel-B完整、checkpoint70/110
  finite，Action Meta及held/wrong-fit1/Panel-B backward为0，validation-test和shuffled/reversed未读，且只生成一套完整rank16；故这是
  结构性科学non-pass，不是运行错误，也不进入S1或靠续训/LR修补。

- 对冻结R5在10个fixed-route task上的`rank_event[task,38,4,8,128]`做CPU平衡分解后发现，跨task平均的结构部分占raw energy约
  `37.49%`；在该task-independent结构的centered variance中，首版quotient保留的rank+event轴只解释`4.10%`，固定38-target owner轴
  单独解释`90.86%`，owner+rank+event解释`94.96%`，而把rank-event改为自由pair只再增加约`.008%`。因此首版在切除absolute task
  state时误删了LoRA target ownership这一合法固定坐标，S0结果不能单独否定“task-code旁路损害shared LOTO”。当前唯一机制修正是在
  B0/B1加入无task轴的trainable owner38 slot，并与既有rank4/event8 slots组合；仍完全忽略`program_event_state`数值，不恢复task code，
  也不改loss、步数、width、rank、data或Gate。全仓`254 passed`；gpu01物理0的一步真实task1 profile自然完成，step为`2.758s`、
  peak allocated `41.258GB`，五臂materialization及finite gradient成立，inventory含owner/rank/event slots，Action Meta为0且信息墙完整。
  profile根为`runs/analysis/pi05_ecp_event_bank_set_owner_quotient_s0_task1_profile_s1_6c33760_gpu01p0_20260901`；其一步Gate non-pass不作
  科学裁决。下一步从clean pushed detached authority fresh重跑同一S0。

- 2026-09-01 absolute-route quotient EBSRI已在唯一canonical interaction中接通：可训练B0/B1不再读取absolute `rank_event`，改用无task轴的
  trainable rank4/event8 structural slots；Program仍经冻结R5 base、event weights、native queries、`kappa`及video-local context作用。
  全仓`253 passed`。clean pushed `main@0d8d901`在gpu01物理0完成task1一步真实S0 profile：38-target forward/backward、wrong teacher、
  五臂materialization与Panel-B forward均自然exit 0，step为`2.768s`、peak allocated `41.258GB`；writer inventory显式包含两个slot
  parameters，Action Meta module/parameter为0，source/Native Stage0/set encoder冻结，held/wrong-fit1/Panel-B backward与validation-test、
  shuffled/reversed reads均为0，每condition仍只有一套完整rank16。一步Gate non-pass只反映未训练profile，不参与科学裁决。profile根为
  `runs/analysis/pi05_ecp_event_bank_set_quotient_s0_task1_profile_s1_0d8d901_gpu01p0_20260901`。

### Absolute-route quotient S0 formal launch contract

- scientific implementation authority为clean pushed `0d8d901`；formal从包含本记录、但不改变科学代码/config的clean pushed detached
  descendant执行。两个独立single-process task分别是task1/program-bank wrong8与task93/wrong94，各自从冻结R5 scorer fresh初始化
  interaction、free summaries、optimizer/scheduler/RNG；不加载旧S0/S1/S2 interaction或optimizer。每task运行10 warmup+100 effective
  optimizer steps并保存actual step70/110，随后以既有五臂Panel-B和双taskaggregate裁决。S0只检验给定free whole-bank condition时，
  quotient B1 factorization能否同时保correct、压wrong；通过不代表真实B0 summary或shared compiler成立。
- exact entry为两个并行的`python scripts/train_ecp_bank_set_tasklocal.py run --mode formal --stop-after-step 110`，配置
  `configs/pi05_ecp_event_bank_set_s0_free_summary_v1.json`，共享只读source step1000、sealed dataset/tokenizer、R5 scorer、gpu01
  `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` condition cache和既有90-pair Program-bank cache。task1固定gpu01物理0，
  task93固定物理2；任务本身是独立优化问题，不用DDP，也不以额外空卡dummy并行。`NCCL_P2P_DISABLE=1`保留在环境中但不建立NCCL组。
- 输出固定为`runs/outputs/pi05_ecp_event_bank_set_quotient_s0_task{1,93}_s110_0d8d901_gpu01p{0,2}_20260901`，launch前必须均不存在；
  aggregate另写`runs/outputs/pi05_ecp_event_bank_set_quotient_s0_gate_s110_0d8d901_gpu01p02_20260901`。旧同类两task各约`21MB`，本轮连同
  aggregate预计低于`100MB`。2026-09-01 07:47 CST live `/data1` quota为`777230572/1073741824KiB`、shared余`84TiB`；gpu01物理
  0/2均为`15MiB/0%`且无compute process，1/3/4为他人满载，5/6空闲。gpu02只有物理5完全空闲但所需node-local cache在gpu01，故用
  gpu01两张完整空闲卡并行，不跨节点复制cache。
- 任一输出异常不覆盖或冒充formal；只有同一clean commit、同一task/world1/device与完整macro70 checkpoint才允许exact-resume到110。
  S0 Gate仍严格要求每task correct fit0/fit1`>=.85`、held`>=.80`、wrong fit0/fit1`<=.25`、margin`>=.50`、all-pairs、family与
  saturation通过。只有双taskaggregate pass才fresh进入S1。
- 07:56 CST首次在物理0/2启动后，另一用户于约两分钟内新占用这两卡各约18.5GB；与本任务`41.26GB`实测峰值不相容。为避免OOM和
  互相干扰，只向本任务两个tmux发送SIGINT，二者均在准备descriptors期间、step1之前退出；原p0/p2 roots各只含run contract，无metrics、
  checkpoint或科学结果，保留为明确aborted provenance且不resume、不聚合。更新后的fresh roots为
  `runs/outputs/pi05_ecp_event_bank_set_quotient_s0_task1_s110_0d8d901_gpu01p5_retry1_20260901`与
  `runs/outputs/pi05_ecp_event_bank_set_quotient_s0_task93_s110_0d8d901_gpu01p6_retry1_20260901`；改用仍完全空闲的物理5/6并在新
  clean pushed detached authority上fresh重启，其余科学合同不变。

- 2026-09-01 S2 functional-polish已从step70 exact-resume到110并完成step70/110共100-job正式Panel-B Gate，结果根为
  `runs/outputs/pi05_ecp_event_bank_set_s2_functional_polish_gate_s70s110_bb98b81_gpu01p0256_w4_20260901`。step110
  meta/target gradient correct为`.827/.772`、same-task held为`.811/.757`、wrong为`-1.060/-.082`、margin为
  `1.744/.715`；held task1 correct/held/wrong/margin为`.514/.498/.097/.381`，task93为
  `.613/.614/.566/.030`，held/train为`.622/.794`。10/10 task仍保持全部correct view严格优于wrong，且step70→110相邻稳定，
  但两个checkpoint和两个role均primary non-pass；因此“旧bank-discriminative初始化+paired unit-gradient direct polish”机制已经
  裁决完毕，不再续训或调质量，也不能进入S3。Action Meta、held/Panel-B/validation-test backward、forbidden reads和
  shuffled/reversed均为0，每condition仍只生成一套完整rank16。

- 对step110做的四组无训练context-path消融显示，旧checkpoint同时依赖B1 absolute event context和真实B0 summary：保留summary但
  去掉B1 raw context时，meta/target gradient correct降至`-.151/.129`；只保留B1 raw context时为`.373/.490`；只去掉B0 absolute
  context时仍有`.636/.720`；两条absolute context均去掉时为`-.032/.138`。这是OOD inference intervention，只能证明旁路被使用，
  不能证明fresh quotient训练会成功。代码审查确认fixed Hadamard task token经冻结R5成为`rank_event`后同时直达B0 inducing和B1
  generated head，而basis-invariant内容路径另由native query形成`kappa`。当前唯一可证伪结构候选是从可训练B0/B1删除该absolute
  code直达，以task-independent rank/event slots保留结构容量；R5 base、event weights、native query→`kappa`、video-local context、
  真实X/Y、bounded correction、exact signed pooling和唯一rank16全部保持。因B0/B1 factorization均改变，必须从R5 fresh依次重跑
  S0→S1→S2；若S0/S1通过而S2仍失败，就否定absolute-code旁路为主因并转查`kappa`/summary decodability或task diversity。

- 2026-09-01 direct-functional S2的110步formal与100-job相邻Gate已经完整结束。step70→110的meta gradient correct从`.682`升至
  `.880`、target从`.923`升至`.931`，task1/93 correct分别保持`.939→.949`与`.915→.899`；但step110 meta/target wrong仍为
  `.444/.905`，task1/93 wrong为`.931/.900`，margin也仅`.266/.017`与`-.003/-.015`。相邻稳定性通过而两个checkpoint primary均
  non-pass。Action Meta、held/Panel-B/validation backward、forbidden reads与shuffled/reversed均为0，每condition仍是一套38-target
  rank16。结果根为`runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_gate_25477c9_eval3e15632_gpu01p013456_w6_20260901`。

- 同checkpoint的16-condition梯度审计显示8个task内correct与oriented-wrong cosine为`-.935--.659`，direct checkpoint上的raw mean与
  unit-normalized mean分别有多臂负投影；MGDA虽存在严格共同方向，但不作为训练算法。旧effective-surrogate step110的16个
  unit-normalized direct gradients简单均值则全部正投影，最小约`.0368`（排除已inactive的task52 wrong后仍成立），说明旧训练已形成
  bank-discriminative representation，而direct-from-zero主要败在representation bootstrap与raw scale composition。该结论不证明held
  task93会迁移；它只是支持一次限定的旧interaction初始化+normalized direct polish，不授权MGDA、LR/seed/rank/width小扫。

### S2 functional-polish launch contract

- 唯一科学改动组合为：只加载旧effective-surrogate step110的完整`EventConditionedBankSetInteraction`；optimizer、scheduler、RNG与
  8个task cursor从零开始。R5 fixed route、真实X/Y、EBSRI B0/B1、signed pooling、rank4、carrier12、split、Panel-B Gate和
  Action Meta缺席均不变。该run明确不是fresh，也不加载旧optimizer或任何held outcome。
- 每个optimizer step覆盖全部8个gradient tasks；correct按step奇偶轮换fit0/fit1，并与wrong-fit0成对。16个条件分别求真实Panel-A
  VJP，active gradient各自unit-L2后以预注册`1/16`质量合成；inactive wrong hinge贡献显式零，保持task/role名义等权。不采用MGDA，
  held task1/93、correct-held、wrong-fit1、Panel B、validation/test与shuffled/reversed继续零梯度。
- 先从clean pushed detached authority做world-size可调的1--2步真实profile，核验旧interaction逐tensor加载、fresh cursor/optimizer、
  16 VJP、finite combined gradient、唯一rank16、Action Meta 0、显存与吞吐。随后运行到既有第一个预注册checkpoint70并做Panel-B
  信息性screen；只有证据仍支持才续到110和完整相邻Gate。该screen不降低或替代正式S2 Gate，也不自行新增性能通过线。
- clean pushed `25e4101`的gpu01物理`0,2,6` world3真实profile已自然exit 0。step1/2分别覆盖全部8 task的
  `correct_fit0/fit1 + wrong_fit0`，每步16次Panel-A VJP，15/16与16/16个scheduled condition有active finite unit gradient；
  combined norm为`.1882/.1413`，游标全部严格推进至`2`。步时`98.66/92.28s`，peak allocated约`30.68GiB`。run contract确认
  旧interaction只作为fresh初始化加载，旧optimizer/scheduler/cursor均未加载；Action Meta module/parameter为0，source/Stage0冻结，
  held/Panel-B/validation-test/shuffled-reversed读写均为0且每condition仍只生成一套完整rank16。profile根为
  `runs/analysis/pi05_ecp_event_bank_set_s2_functional_polish_profile_s2_25e4101_gpu01p026_r3_20260901`。

### S2 functional-polish formal launch record

- scientific implementation/config authority仍为clean pushed `25e4101`，profile证据已由`3bb8972`登记；formal从包含本记录且不再改变
  科学代码/配置的clean pushed detached descendant执行。唯一输入复用source step1000、sealed dataset/tokenizer、gpu01既有23GB
  condition cache、既有90-pair Program-bank cache、R5 scorer与旧effective-surrogate interaction step110；不复制或重建资产。
- exact training entry为world3 `torchrun --standalone --nproc-per-node=3 scripts/train_ecp_bank_set_shared.py`，配置
  `configs/pi05_ecp_event_bank_set_s2_functional_polish_v1.json`、`--mode formal --stop-after-step 70`、
  `CUDA_VISIBLE_DEVICES=0,2,6 NCCL_P2P_DISABLE=1`。每步固定8 task、16个Panel-A paired VJP；只保存既有预注册checkpoint70，先做
  Panel-B信息性screen，证据支持才按完全相同world topology exact-resume到110。任何不兼容或不完整输出都不覆盖或冒充formal。
- fresh输出固定为
  `runs/outputs/pi05_ecp_event_bank_set_s2_functional_polish_s70_3bb8972_gpu01p026_r3_20260901`。2026-09-01 03:40 CST live检查
  gpu01物理`0,2,6`分别为UUID `GPU-658b6043`、`GPU-47449b15`、`GPU-21b5514a`，均仅`15MiB/0%`、Default且无compute
  process；其余同节点卡在他人高负载任务中，故使用这三张完整空闲卡而不跨节点。`/data1` quota为
  `777184320/1073741824` blocks，shared余`84TiB`；同类110步run约`25MB`，本段预计远低于`100MB`。launch前再核对live设备与
  output仍为空；profile实测约`92--99s/step`、peak约`30.68GiB`，容量与预计两小时内完成step70均成立。
- detached `bb98b81`已在gpu01物理`0,2,6` world3自然完成step70并保存完整checkpoint；70步共1120次Panel-A VJP，全部数值
  finite，步时中位`88.54s`、peak allocated约`30.68GiB`，8个cursor均为70。completion确认Action Meta 0、held/Panel-B/
  validation-test/shuffled-reversed backward或reads均为0，仍只生成一套完整rank16。
- checkpoint70的单点Panel-B screen已用gpu01物理`0,2,5,6`四个独立persistent workers完成50/50 jobs并通过raw-row、pairing、
  information-wall与bank-lifecycle校验。meta/target gradient-task correct-fit中位为`.767/.749`、wrong为`-1.030/-.086`、margin为
  `1.656/.685`；held task1 correct/wrong/margin为`.509/.099/.370`，task93为`.616/.549/.048`。虽然单点primary diagnostics未过，
  但10/10 task均满足每个correct view严格优于每个wrong view，首次同时取得广泛bank specificity与仍为正的correct capacity；相对旧
  direct step70的correct后移恢复证据足以支持按预注册合同exact-resume到110，而不支持宣布Gate通过或进入S3。screen根为
  `runs/analysis/pi05_ecp_event_bank_set_s2_functional_polish_screen_s70_bb98b81_gpu01p0256_w4_20260901`。

- 2026-09-01 S2首轮effective-rank4 shared LOTO在step70/110稳定non-pass；真实policy梯度审计随后证明surrogate从fresh起即错配：
  16臂factor-vs-functional cosine中位`.0219`、6臂为负。相同shared EBSRI图的2:1 direct-functional锚点对8个gradient tasks却有
  raw共同下降方向，minimum-norm最小投影`.1129`。因此唯一修正保持架构/split/rank/Gate不变，只改为Panel-A direct VJP：correct
  质量1、wrong有界neutralization质量.5、LR `1e-4`，不做MGDA或普通超参小扫。

- direct-functional实现`25477c9`的gpu01 world6两步真实profile已经自然exit 0，输出根为
  `runs/analysis/pi05_ecp_event_bank_set_s2_direct_functional_profile_s2_25477c9_gpu01p013456_r6_20260901`。step0六个wrong与step1四个
  correct/两个wrong均有finite nonzero aggregate gradient，步时`27.70/17.72s`，peak allocated `32.93GB`；12次VJP、Action Meta 0、
  held/Panel-B/validation backward 0、target-cache build 0且唯一rank16均由run contract/completion确认。

### S2 direct-functional fresh formal launch contract

- scientific authority为包含`25477c9`及本记录的clean pushed `main`，从其detached frozen worktree fresh执行；不resume旧surrogate或profile，
  不覆盖已有输出。仍只训练8个gradient tasks的同一shared `EventConditionedBankSetInteraction`，task1/93为interaction holdout；fixed route、
  wrong rings、四拍arms、真实X/Y、signed pooling、rank4、carrier12、两相邻checkpoint70/110和完整Panel-B Gate均不变。
- formal训练只读Panel A跨episode actions并用直接policy VJP；correct raw flow、wrong有界neutralization hinge按schedule总质量2:1。
  Panel B、correct-held、wrong-fit1、task1/93、validation/test、shuffled/reversed均零梯度；source、Native Stage0、R5 scorer、carrier、scale与
  Action Meta冻结，最终每condition仍只物化一套38-target rank16。
- 2026-09-01 00:31 CST live检查gpu01物理`0,1,3,4,5,6`为空闲合适卡，物理2有他人约2.7GB且持续利用，故不用；gpu02多数卡已有
  约30GB任务。formal在launch前再核对同一live状态，有几张合适卡就用几张、单节点最多6张；world size一经formal启动即固定。
  `/data1` quota为`776989728/1073741824` blocks，复用23GB node cache及既有model/data，新增run预计远低于1GB。
- 训练输出预定为`runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_s110_25477c9_gpu01p013456_r6_20260901`。从fresh运行110步，
  保存single checkpoints70/110；完成后以独立persistent workers执行原100个task-arm-checkpoint jobs和相邻稳定性。只有完整S2 Gate通过才
  做10-task fresh refit并进入S3；non-pass按最早失效接口分析，不以内部loss、续训或seed/LR/width小扫替代Gate。

- 2026-08-31 EBSRI S1 real-summary task-local双task aggregate正式通过，formal根为
  `runs/outputs/pi05_ecp_event_bank_set_s1_gate_s110_a1f14e4_gpu01p01_20260831`。task1 correct fit0/fit1/held
  recovery为`.942/.953/.962`，wrong fit0/fit1为`-.529/-.517`；task93分别为`.928/.905/.881`与
  `-.188/-.180`。两task全部Gate checks通过，Action Meta为0、Panel B backward为0，并且只物化一套38-target
  carrier12+residual4 rank16。该结果证明真实B0 set summary在task-local条件下足以驱动direct condition-generated head，
  不证明跨task shared mapping；当前下一阶段是S2 fixed-route shared task-LOTO。随后`main@cdcae8b`将B0 summary与B1 replay
  chunk解耦；同口径profile中wrong约由`12s`降至`6.3s`，task1 correct约由`35.5s`降至`13.3s`，峰值约`41.1GB`。

- 2026-08-31 EBSRI S0 direct-summary formal双task aggregate已经通过。旧FiLM+共享zero-head在task1/93上correct与wrong均近似R5、
  summary-token swap几乎无效；Panel-B teacher与wrong-only scale-matched overfit随后证明真实bank、signed pooling、rank4与factor网络可产生
  强抑制，最早失效是summary条件没有取得candidate-head控制。当前实现让Program/event context与summary直接生成每family candidate
  linear head，仍保持step0严格interaction-off。clean pushed `3b7124e`的task1 correct fit0/fit1/held recovery为
  `.948785/.922930/.929913`，wrong fit0/fit1为`-.535224/-.491055`；task93为
  `.905449/.909439/.894417`与`-.161546/-.169201`。两task全部absolute、margin、all-pairs、saturation与family checks通过，
  Action Meta 0、held/wrong-fit1/Panel-B backward 0、validation/test reads 0、无shuffled/reversed且唯一rank16。aggregate位于
  `runs/outputs/pi05_ecp_event_bank_set_direct_s0_gate_s110_3b7124e_gpu01p01_20260831/`。当前立即进入S1：从fresh R5初始化，唯一科学
  变量是删除training-only free summaries并训练真实B0 set encoder；不加载S0 checkpoint，不改变loss、LR、teacher、宽度、rank或Gate。

- 2026-08-31 EBSRI S0实现面已经接通：B0按真实input与joint abs/adj/init/goal output候选流式形成event bank-set summary，
  B1从冻结run-local query-relative descriptors批量计算shared correction，并仍由每个native owner/group独立对真实X/Y执行FP32 signed
  pooling；output时间边界跨chunk保持、不同video不串接。S0只训练candidate trunk/direct condition-generated heads与training-only free correct/wrong
  summaries，真实set encoder、R5 Program/native heads、source、Native Stage0、scale/carrier全部冻结，Action Meta module/parameter为0，最终只
  物化一套38-target carrier12+residual4 rank16。task1/93真实profile分别为`2.841s/step`、`10.726s/step`，peak allocated约
  `41.09GB/29.08GB`；zero equivalence、wrong teacher suppression、forward/gradient/materialization均成立。一次跨group batched signed
  accumulator虽在小张量测试等价，却在真实R5零修正产生`.269/.180`偏差，已在formal前撤回；不以吞吐放宽数值/功能合同。当前
  `113 passed`，随后已在上述clean authority完成并行task1/93 formal S0。

- 2026-08-31 owner已恢复持续性goal并授权按第六次专家复核继续高效推进。1320行原始回复已逐字保存为
  `docs/expert_review_20260831_event_conditioned_bank_set_relative_interaction.md`。专家锁定并复核`main@92617d0`与未合并
  `codex/g3-vector-interaction@2295f48`，确认此前正确执行第五次方案并已到其失败边界；被淘汰的是当前scalar/base-score及32维vector
  set-independent pointwise interaction，不是whole-bank-conditioned continuous interaction、真实X/Y、signed pooling、rank4、full base、
  Natural Program、Stage0或Native-Factor整体。

- 当前唯一活动设计为Event-Conditioned Bank-Set Relative Interaction（EBSRI）：B0用Program的4 rank x 8 event native queries构造
  candidate的32维相对坐标，并在每条video内累计event moments与线性induced summaries；B1由Program/local event/whole-bank summary
  直接生成每family candidate head及bounded correction，再以真实X/Y形成唯一signed measure、rank4 residual与完整rank16。B0不得成为task matcher、binary/
  continuous gate或LoRA generator；Pass A/B仍读取同一video set，K视频逐条独立后uniform聚合。

- 当前执行S2 fixed-route shared task-LOTO：共享同一set encoder/interaction，在8个gradient tasks训练并hold out一个meta与一个
  target interaction task；通过后才用全部10 tasks fresh refit形成component initialization，随后才进入Natural Program joint S3。
  S1的task-local通过不能替代这个shared mapping Gate；吞吐秒数只作工程目标，不作科学Gate。

- clean pushed detached v4的step70/110相邻Gate已经完整结束。correct fit为`.922509/.929947`、same-task held为
  `.926447/.953521`，但unseen wrong-on为`.930806/.933331`、correct-minus-wrong为`-.001784/-.006375`，正确bank更好只有
  `5/10`和`4/10`；信息墙与四family通过，bank因果分离失败。B1 base score没有修复shared correction acquisition，v4 strict
  non-pass。formal aggregates位于
  `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v4_base_score_gate_step{70,110}_b7d2638_gpu01p012_gpu02p47_w5_20260831/`。

- 六task首步梯度分解显示v2/v3/v4的correct-vs-wrong functional gradient cosine中位为
  `-.961291/-.961291/-.966288`，wrong/correct norm ratio中位为`1.0597/.5298/.5038`。这解释了前三轮的表面反复：权重变化只在
  “共同破坏correct以压wrong”和“保住correct也保住wrong”之间移动，当前local candidate chart没有产生第三个选择性方向。

- 最小32维native query/key vector interaction先用pointwise free-delta gauge回归得到wrong压低但correct同样被破坏，说明该内部目标
  与最终rank4功能不对齐；随后改用完整effective-rank4矩阵距离作task-local上界，不改变真实X/Y、signed pooling、rank4、
  carrier12+residual4、唯一rank16或信息墙。80 updates后task1的correct fit0/fit1/held panel-B recovery为
  `.720904/.717564/.711262`，wrong fit0/fit1为`-.527627/-.519287`；task93为`.591613/.601969/.569709`与
  `-.379331/-.418162`。wrong fit1与correct held均零梯度却分别泛化到同类结果，证明模型真的读取bank内容；但correct远低于R5强
  正控，两个代表task一致暴露capacity--specificity冲突。本次诊断结果在
  `runs/analysis/pi05_ecp_g3_effective_rank4_tasklocal_2295f48_gpu01p01_20260831/`，不作shared或deployment成功声明。

- 当前结论不是“bank没有信息”或“rank4不够”，而是第五次专家建议的首版逐candidate local交互仍不足以同时表达capacity与
  specificity。继续扫loss、LR、seed、width或bound没有新机制信息；后继需要决定是否加入bank-set/global-event级统计或更换交互
  因子化，抑或终止这一candidate-interaction函数类。在owner决定是否再次咨询专家前停止实施。

- clean pushed detached `main@248d768`已完成2:1 positive-anchor candidate interaction的110步训练、step70→110 exact resume及
  五worker相邻Gate。step70/110 correct fit为`.922565/.929101`、same-task held为`.931639/.953285`，但unseen wrong-on仍为
  `.932045/.934305`、correct-minus-wrong为`-.002346/-.005576`，只有`5/10`和`4/10` task正确bank更好；信息墙、四family、
  absolute correct及same-task检查通过，bank因果分离检查失败。该结果证明positive anchor已解决共同破坏，却没有让shared correction
  学会选择性作用于wrong bank，因此strict non-pass，不能接回Natural Program。

- step110 full10逐层审计显示input feature、LayerNorm/MLP与最终pooled update对wrong bank都有明显可分信号，wrong相对natural的
  separation median分别约`3.105/3.299/3.142/4.306`；但实际correction gauge RMS只有约`1.5e-5`量级，base-score RMS约`.0202`，
  远低于`.0125`的event-envelope，pooling KL约`1e-8`。同一十task的task-local free-delta反事实随后在保持真实X/Y、signed pooling、
  rank4与唯一rank16时，以absolute delta p95 median仅`.0020`把wrong panel-B recovery压到`-.5277`中位，10/10 task均低于`.25`，
  且无bound saturation。最早失效接口由此锁定为shared scorer没有取得可用selection correction，而不是operator、bound或bank容量。

- 当前唯一机制修正为v4 base-score-conditioned interaction：把B1实际使用的
  `stop_gradient(q0·(value-global_B0_mean))/.02`加入逐candidate feature；不改Program/event、candidate measure、signed pooling、
  loss、seed、LR、步数、rank、scale或Gate。input仍无output-type轴，output仍保留abs/adj/init/goal；base feature只沿event展开，
  compact与streaming均复用同一query/全局mean。实现分支已通过`238 passed`、chunk/compact、translation、detached-gradient及
  architecture review。gpu01物理0/1/2真实world3一步为`51.326s`，三卡peak reserved最高约`33.52GiB`；四family final weights均有
  finite nonzero gradient，Action Meta/source/Stage0/scale trainable均0、native teacher reads 0，仍只物化唯一完整rank16。真实smoke
  已通过；下一步clean集成并从fresh detached authority启动同一formal。

### Base-score-conditioned candidate interaction formal launch contract

- scientific implementation authority为clean pushed `main@90cd380936b8c2bd933499b419ae5bf9486e73aa`，全仓CPU合同`238 passed`，
  architecture guard无hard violation。formal从只新增本launch记录、不改变科学实现的clean pushed detached descendant执行。相对v3的
  唯一机制变化是scorer额外读取detached B1 base score feature；interaction层因13维输入而全部fresh初始化，只复用R5 frozen primal
  scorer及既有raw-X/Y、B0和Program-bank caches。loss、seed、LR、warmup/effective steps、data、rank、scale与Gate逐项保持v3不变。
- 输入继续复用source step1000、sealed dataset/tokenizer、gpu01
  `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` base cache与
  `runs/caches/pi05_ecp_program_bank_candidate_interaction_v1_90pair_200a778_20260831/` cross-language cache；不复制或重建。
  输出固定为`runs/outputs/pi05_ecp_program_bank_candidate_interaction_v4_base_score_s110_90cd380_gpu01p012_r3_20260831/`，launch前确认不存在。
  上一formal仅`51MiB`；2026-08-31 06:43 CST `strg01` live `/data1` quota usage为`776282020/1073741824KiB`、limit
  `1084227584KiB`，共享空间余`84TiB`，预计新增远低于`1GiB`。
- 同时live检查两节点：gpu01物理0/1/2均为`15MiB/0%`且无compute process，物理3--6为他人约34.6GiB/100%任务；gpu02物理7空闲，
  其余只有物理4低显存但有他人进程，训练必须single-node collective且gpu01已有三张完整空闲卡与本地cache，因此使用gpu01 `0,1,2`
  world3，不跨节点拼卡或干扰他人。固定`NCCL_P2P_DISABLE=1`、NUMA0与deferred NCCL；step70→110 exact resume锁定world3。
- exact entry为detached formal worktree中的`torchrun --standalone --nproc-per-node=3 scripts/train_ecp_joint_program_primal.py`，配置
  `configs/pi05_ecp_program_bank_candidate_interaction_v4.json`与base `configs/pi05_ecp_shared_compiler_g3_v5.json`。先fresh执行
  `--stop-after-step 70`，首个global step核对v4 schema、interaction-only、四family gradient、Action Meta/source/Stage0/scale 0、
  native teacher reads 0及single rank16；成立后连续到70，再从该checkpoint exact resume到110。完整Gate固定使用
  `configs/pi05_ecp_program_bank_candidate_interaction_gate_v4.json`；efficiency只报告，shuffled/reversed不运行。

### Positive-anchor candidate interaction formal launch contract

- scientific implementation authority为clean pushed `main@fd202516c71baf183a04d181f72fe9a2ae08f2df`，全仓CPU合同`236 passed`；
  formal从只新增本launch记录、未改变科学实现的clean pushed detached descendant执行。相对raw-unit v2的唯一科学变化是active
  wrong backward由`-1/6`变为`-1/12`，两条correct仍各为`+1/12`，因此correct:wrong总质量为`2:1`；architecture、R5 zero-init、
  十task/两fit视频、wrong cycle、panel A、seed、LR、microbatch4、10 warmup + 100 effective、step70/110 checkpoints和完整Gate
  全部保持。必须fresh启动，不读取或resume `c7874f3`或`cbe3124` interaction checkpoint。
- 输入继续复用source step1000、sealed dataset、tokenizer、gpu01
  `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` 23GB base cache与
  `runs/caches/pi05_ecp_program_bank_candidate_interaction_v1_90pair_200a778_20260831/` 57GB immutable cross-language cache；
  不复制或重建。输出固定为
  `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v3_anchor_s110_fd20251_gpu01p012_r3_20260831/`，launch前已确认不存在；
  参照上一formal仅51MB，预计新增远低于1GiB。2026-08-31 04:11 CST `strg01` live `/data1` quota为`740.2G/1T`，共享空间余
  `84TiB`，预算充分。
- 2026-08-31 04:11 CST live GPU：gpu01物理0/1/2分别为UUID `GPU-658b6043`、`GPU-845a7b73`、`GPU-47449b15`，均仅
  `15MiB/0%`且无compute process；物理3--6为他人约34.6GiB/100%任务。gpu02物理0--3满载、5/6有显著他人负载，物理4有他人
  低UTL进程且只有7完全空闲；训练是single-node collective，故使用gpu01 `0,1,2` world3，固定`NCCL_P2P_DISABLE=1`、NUMA0与
  deferred NCCL，不跨节点拼卡或干扰他人。exact resume锁定world3。
- exact entry为detached formal worktree中的
  `torchrun --standalone --nproc-per-node=3 scripts/train_ecp_joint_program_primal.py`，使用
  `configs/pi05_ecp_program_bank_candidate_interaction_v3.json`与base `configs/pi05_ecp_shared_compiler_g3_v5.json`。先fresh执行
  `--stop-after-step 70`，首个global step立即检查v3 schema、只训练interaction scorer、两条correct各`+1/12`、active wrong
  `-1/12`、分臂gradient finite且wrong/correct相对v2按权重下降、Action Meta module/parameter 0、source/Stage0/scale trainable 0、
  native teacher reads 0及76 tensors唯一rank16；不成立则在形成科学结论前停止。成立后连续到70，再从step70 exact resume到110；
  formal Gate固定使用`configs/pi05_ecp_program_bank_candidate_interaction_gate_v3.json`，shuffled/reversed不运行。

- clean pushed detached `main@c7874f3`已经完成candidate-level interaction首轮formal训练、step70→110 exact resume及两个checkpoint的
  五worker完整Gate。step70/110 correct fit为`-.388363/-.386363`、same-task held为`-.392916/-.393941`、unseen wrong-on为
  `-.405269/-.398702`，correct-minus-wrong为`-.018599/-.020456`且只有`5/10` task正确bank更好；interaction-off始终为
  `.940432`。两个checkpoint均strict non-pass。Action Meta为0、deployment native-teacher reads为0、held/panel-B backward为0，且每个
  condition仍只有一套完整38-target rank16；因此这是科学non-pass，不是加载、信息墙、adapter或R5 base capacity故障。

- 当前最早失效接口已由三路独立审计与formal轨迹共同锁定为loss/gradient单位。每条correct view权重`1/12`，同task两条合计`1/6`；
  active wrong原按`-1/[6(B_free+eps)]`反传，在实际小`B_free`下先验放大约`15.7--359.6x`。global clip只统一缩放合成梯度，不能恢复
  correct/wrong相对方向。训练前段wrong hinge迅速归零、correct loss反而恶化，最终correct与wrong recovery Pearson达到`.95/.96`，
  精确对应“共同破坏所有bank”的便宜解；这淘汰当前normalized-gradient objective，尚未淘汰candidate-interaction函数类。

- 当前唯一修正位于`codex/g3-interaction-balanced-credit`：架构、真实X/Y、candidate集合、signed pooling、rank4、唯一rank16、十task数据、
  seed/LR/步数和原Gate全部保持，只把wrong backward改成raw functional-loss单位的固定`-1/6`，与两条correct总质量1:1；
  `B_free`归一hinge继续报告但不进入梯度。首个真实step额外记录一次correct/wrong分臂gradient norm/cosine，避免再让隐藏尺度问题进入
  formal。evaluation同时修正相邻checkpoint复用diagnostic teacher cache时物理read delta可为0的记账语义；deployment read仍必须严格为0。
  该修正完成clean验证后从fresh detached authority重跑，不resume首轮checkpoint。

### Balanced-credit candidate interaction formal launch contract

- scientific implementation authority为clean pushed `main@6694e9966ead24be393451d2d1ec9b1fd9ef3755`，全仓CPU合同`236 passed`；
  formal从只增加本launch记录的clean pushed detached descendant执行。唯一科学变化是wrong backward从
  `-1/[6(B_free+eps)]`改为raw-unit `-1/6`；architecture、R5 zero-init、十task/两fit视频、wrong cycle、panel A、seed、LR、
  microbatch4、10 warmup + 100 effective、step70/110 checkpoints及完整Gate均不变。fresh启动，不读取或resume首轮
  `c7874f3` interaction checkpoint；首个global step只额外报告一次六task correct/wrong分臂gradient norm/cosine，不作为Gate。
- 输入authority继续复用source step1000、sealed dataset、tokenizer、gpu01 `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`
  23GB base cache与`runs/caches/pi05_ecp_program_bank_candidate_interaction_v1_90pair_200a778_20260831/` 57GB immutable
  cross-language cache；不重建或复制cache。输出固定为
  `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v2_balanced_s110_6694e99_gpu01p012_r3_20260831/`，预计新增远低于1GiB，
  launch前确认root不存在。2026-08-31 02:25 CST `strg01` `/data1` quota为`740.2G/1T`，预算充分。
- 2026-08-31 02:24 CST live GPU：gpu01物理0/1/2分别为UUID `GPU-658b6043`、`GPU-845a7b73`、`GPU-47449b15`，均仅
  `15MiB/0%`且无compute process；物理3--6为他人约34.6GiB/100%任务。gpu02仅物理7完全空闲，物理4虽低UTL但有他人进程；本训练
  必须single-node collective，故使用gpu01 `0,1,2` world3，而不是跨节点拼卡或干扰他人。固定`NCCL_P2P_DISABLE=1`、NUMA0与deferred
  NCCL；exact resume锁定world3。
- exact entry为detached worktree内`torchrun --standalone --nproc-per-node=3 scripts/train_ecp_joint_program_primal.py`，配置
  `configs/pi05_ecp_program_bank_candidate_interaction_v2.json`与base `configs/pi05_ecp_shared_compiler_g3_v5.json`。先fresh执行
  `--stop-after-step 70`，首个step落盘后立即检查分臂梯度、finite、参数所有权、Action Meta 0、native teacher reads 0与唯一rank16；
  若不成立则在产生科学checkpoint前停止，若成立则让同一进程连续到70，再从step70 exact resume到110。formal Gate仍使用
  `configs/pi05_ecp_program_bank_candidate_interaction_gate_v2.json`；shuffled/reversed不运行。

- 2026-08-30第五次专家复核已经完整消费，1132行原文逐字保存为
  `docs/expert_review_20260830_program_bank_interaction.md`并与owner最新授权共同更新active design。专家确认R5/P0/P1是capacity
  primitive，R12/R13则充分终止binary full/half门卫；早期“soft mixture失败意味着后继必须近二值”的表述已被明确纠正：失败只
  淘汰两套谱端点的hard/soft选择，下一接口必须让Program与bank在candidate层共同形成唯一连续signed measure。

- 已从clean pushed `main@b59d7bdd5fd7c2990c2f6e0eb28f170419ac7a84`建立唯一实现分支
  `codex/g3-program-bank-interaction`与worktree
  `/data1/user/ymdai/projects/EMBER-worktrees/ecp-g3-program-bank-interaction`。authority/active-contract已经先行提交并推送；当前科学实现、
  sealed config、train/eval Gate与cache-seal入口已接通，正在完成formal前的clean commit和真实smoke。

- 当前下一阶段固定为co-conditioned bank-interaction positive control：R5 fixed token、feature chart、native heads、source、Native
  Stage0、B0 full solve、carrier、scale与Action Meta冻结，只训练event-specific candidate interaction scorer。correct/wrong bank使用
  同一base-full-plus-correction deployment forward，loss只含correct functional flow和bounded wrong-bank neutralization；step0必须
  严格复现R5 full path。实现先扩展fixed-microblock branch bias，再接通event-native queries、local `ProgramBankContext`、唯一signed
  pooling和精简Gate；通过定向合同及真实forward/gradient/materialization/throughput smoke后才进入formal。

- canonical streaming signed pool现支持fixed microblock下的branch-specific bias并保持跨frame chunk pending state；candidate scorer严格区分
  input `(frame,probe,horizon)`与output `(frame,probe,horizon,type)`，读取未聚合event query、当前bank local context及真实native content，
  输出`+delta/-delta`后只执行一套exact signed pooling。delta的全局measure均值是两个branch各自softmax内的常数gauge，canonical实现
  保留与显式center完全等价的未定gauge表示，避免为无功能差异增加第三次全视频读取。

- active R12/R13 support probe、threshold、condition-dependent selected power、full/half hard/soft route及对应训练评测入口已从唯一执行面删除；
  历史config会fail loudly，原始结果继续由Git与formal artifacts保存。底层固定inverse-power primitive仅保留给历史分析/正控，不再由
  condition动态选择。generic历史J2/J3 pairing明确保持`interaction_off`与旧cache语义，避免半迁移入口。

- interaction qualification只checkpoint/优化`ProgramBankInteractionScorer`；R5 Program/native heads、source、Native Stage0、scale、carrier与
  Action Meta全部冻结。correct与wrong使用同一exact-language forward；训练完成/70→110 exact resume、single-node world topology、
  checkpoint/commit、六worker coverage、相邻Gate及aggregation commit均已绑定。缓存脚本精确预封80个training wrong pairs与10个unseen
  Gate pairs；新cache直接读取封存native means，旧v4 cache才惰性回算fallback。

- 当前定向`tests/ecp`为`111 passed`、全仓CPU合同为`234 passed`，并额外修复了审计发现的optimizer-step控制流错位、multi-rank fresh-root竞态、aggregate authority
  漏洞与non-finite branch bias污染风险。clean pushed detached `02b3588`在gpu01物理0/1/2完成最终真实smoke：zero-init interaction-on/off
  的76个rank16 state tensors与a/b residual逐值完全相等，maximum difference为0，K1 weight为1；Action Meta/source/Stage0/scale trainable、
  native teacher reads均为0。world3真实step在policy microbatch2下为`43.889s`、peak reserved `20.07--26.70GiB`；唯一吞吐裁决把microbatch
  提到4后为`39.847s`、peak `27.29--33.52GiB`，仍保留至少约12.5GiB物理余量，故formal固定microbatch4，不再做小扫。

- cross-language 90-pair cache预计约`60GB`，不能放入gpu01当前仅约`39GB`空闲的`/dev/shm`。formal前将重新核对`/data1`独立quota并把该
  operational cache放入有足够预算的`/data1`根，复用既有23GB base cache且不删除任何历史资产；seal workers完成后先验证worker union
  精确90、无重复/遗漏，再启动训练。

### Candidate-level Program--bank interaction qualification formal launch contract

- scientific implementation authority为clean pushed `main@200a7784fad1c0312f5b2cdea893f64eae95a60b`；cache seal、训练和Gate从只新增
  本launch记录、未再修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant执行。初始化固定为R5 step110 shared
  functional chart；source、Native Stage0、R5 scorer、full inverse operator、carrier12、scale与Action Meta冻结，只训练
  `ProgramBankInteractionScorer`。fresh运行10 warmup + 100 effective updates，在actual step70与110保存相邻single checkpoints；
  step70到110只允许同一output root、world topology与sampler state的exact resume。
- 数据固定为10个gradient tasks各两条fit K1 video、panel A functional flow与same-role其它gradient-task wrong banks；wrong bank始终使用
  correct task exact language。same-task held video、unseen task2/74 wrong banks、panel B、validation/test与shuffled/reversed全部零梯度。
  每个condition仍只物化一套carrier12 + continuous rank4 residual的完整38-target rank16；fixed routing token仅是本positive control的
  training-only Program authority，任何结果都不冒充Natural Program shared mapping或最终deployment Writer。
- 2026-08-31 00:05 CST live storage authority：`strg01` `/data1` quota为`716461192/1073741824KiB`，约余341GiB；共享
  filesystem尚余84TiB。90-pair cache精确预估`60,690,522,734B`（56.52GiB），故固定写入共享
  `runs/caches/pi05_ecp_program_bank_candidate_interaction_v1_90pair_200a778_20260831/`，不写gpu01 `/dev/shm`；base frozen-condition
  cache继续复用gpu01的`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`。独立seal workers结束后必须核对同一commit/root、
  完整worker index、90个唯一`(program task, bank task, video demo)`、无遗漏/重复且native teacher reads为0。
- 同时刻gpu01物理0/1/2均为A40、`15MiB/0%`且无compute process，3--6为他人约34.6GiB/100%任务；gpu02物理7完全空闲，
  其余设备有他人显存或高UTL进程。训练是single-node collective，计划在紧邻launch复查后使用gpu01 `0,1,2` world3、
  `NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；不跨节点拼卡。cache seal和后续独立Gate workers可在不干扰他人的前提下使用
  两节点实时合适设备。
- formal训练输出固定为
  `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v1_s110_200a778_gpu01p012_r3_20260831/`；入口为detached worktree中的
  `torchrun --standalone --nproc-per-node=3 scripts/train_ecp_joint_program_primal.py`，training config固定使用canonical asset root中的
  `configs/pi05_ecp_program_bank_candidate_interaction_v1.json`，base config为`configs/pi05_ecp_shared_compiler_g3_v5.json`，并绑定现有
  source run、source step1000、tokenizer、sealed dataset、base condition cache及上述cross-language cache。先fresh `--stop-after-step 70`，
  再从`checkpoints/macro_00000070` exact resume到`--stop-after-step 110`。
- Gate同时评价actual step70/110的correct two-fit + same-task-held、unseen wrong interaction-on及同一wrong interaction-off。资格要求仍为
  fit/held/held-to-fit至少`.85/.80/.85`、unseen wrong至多`.25`、correct-wrong至少`.50`、10/10 task correct更好、wrong off-on至少
  `.40`、四family不系统反向、step110相对step70 fit下降不超过`.05`。内部loss、cache完成或单个checkpoint均不能替代该Gate；
  non-pass按最早失效接口解释，不做seed/LR/width/rank小扫。owner于首个formal吞吐诊断后明确：wall、显存、UTL及
  evaluation/training ratio继续报告和优化，但不再作为科学qualification硬门，不能否决满足机制条件的checkpoint。

- clean detached `8043148`首个formal attempt在6步时主动停止且没有checkpoint，不能作为科学结果。实测step2--5为`54.67--61.79s`，
  根因是旧frame-only greedy assignment在world3把三个均含固定256-row policy functional成本的tasks放到同一rank、另一个rank只放一个；
  并非GPU闲置、cache miss或模型规模本身。当前窄修正保持每步原六task、3 meta/3 target、每task权重、loss与optimizer cadence不变，
  先把rank task-count限制在`ceil(6/world_size)`，再按frame cost平衡；同时按owner authority删除自行设置的`45s`与evaluation/training
  qualification checks，仅保留效率诊断。修正完成后从新clean pushed detached authority fresh重跑，不resume该invalid attempt。

- 2026-08-30 01:05:31 CST是owner本轮睡眠推进锚点；后续汇报按该机器确认绝对时刻核对，不以对话压缩位置替代。

- full-inverse raw bank审计已经把新接口从“是否有兼容信号”推进到“shared Program是否学会该信号”。R5成功primal在三条same-task
  videos与五个same-role wrong banks上的input projection p10达到AUC `1.0`、逐task严格分离`10/10`；正确全局minimum
  `.907248`仍高于错误global maximum `.905998`。单纯把该support乘到最终rank4 residual则明确失败：correct/wrong recovery
  中位`.954661/.930365`、margin仅`.031766`，证明兼容性不能只是最终音量旋钮。

- 同一固定support阈值在operator坐标上做task-local hard positive control取得correct/wrong/margin
  `.950915/.005173/.908899`，correct-better与margin达`.10`均为`10/10`。因此强full方向与bank-specific half方向本身足以组合，缺口是
  **Program--bank compatibility决定operator坐标**，不是继续调谱幂。固定sigmoid后在线性混合full/half query却把correct中位降到
  `.238736`，说明两端之间的signed-attention几何不可线性插值；后继必须保留近二值/离散选择，不能用温度小扫修饰。

- 该hard switch随后原样作用到R10 Natural Program step70/110完整12-task Gate。step110 train/held/task-held从原
  `.559896/.544189/.151475`降到`-.482993/-.631937/-.533894`；matched/mismatched Program-bank support AUC仅`.558160`、逐task
  严格分离`0/12`，中位`.867304/.864599`，多数正确视频也被错误送入half。由此排除“固定门直接救活R10”，并把最早接口精确锁定为
  R10 shared Program/scorer没有学习R5已有的内容兼容几何。

- R12是R10-initialized shared compatibility qualification：保持真实X/Y、signed pooling、rank4、carrier12+residual4唯一
  rank16、Natural Program、functional panel与Action Meta 0；用同task两条fit video交叉形成positive Program-bank pairs、同role
  cyclic task形成negative pairs，以input projection低分位band的共享calibration loss显式训练Program/native heads，同时保留correct
  cross-episode functional loss。deployment只按当前Program与当前bank内容做固定近二值full/half选择，不读task ID或成对反事实；
  held video、task2/74、panel B保持零梯度。先做定向CPU合同、真实多卡gradient/materialization与loss-gradient比例profile，再决定
  formal，不再扫threshold、temperature、谱幂、LR或seed。

- R12 canonical实现已接通：`SpectralNativeCovariance`暴露可微retained projection；每个K1 bank独立计算固定p10 route support与排序
  12--20位training support；deployment在当前Program/current bank上near-binary选择full/half operator，历史config保持原路径。正确
  functional训练分支teacher-force已验证的full endpoint，cross-video positive与same-role cyclic negative只监督共享support；evaluation
  额外记录matched/mismatched route并把`.80/.20/.001`机制门纳入原完整G3 Gate。没有新增task/video lookup、第二adapter或Action Meta。

- 230项全量CPU合同通过。gpu01物理`0,1,2`三卡真实六task profile完整运行：R10 step110 tensors严格加载，11,178,369个trainable仅为
  Natural Program与native heads，Action Meta/source/Stage0/scale trainable均为0，native teacher reads为0；单步`20.92s`，三卡peak
  reserved约`20.4--21.8GB`。weight `1.0`使总gradient norm达`19.58`并压倒原functional约百倍；按同图线性比例唯一修正为`.01`后
  norm为`.2451`、不触发clip，input/output head gradient为`.001164/.000523`，Program process为`.12195`，与R10 functional基线
  `.08331`同量级。初始positive/negative full-route fraction为`.0833/0`、training-support margin `.004154`，因此该run有真实学习空间且
  negative没有先验误过门。

- R12已从clean pushed detached `fdab4ae`完成step70及exact-resume step110，两个完整12-task paired Gate均为strict non-pass。
  matched full-route fraction由`.444444`升至`.527778`，仍低于`.80`；mismatched full-route维持`.083333`并通过`.20`上限，paired
  support margin中位由`.018712`升至`.021072`。step110 correct-vs-wrong bank margin与interaction已达`.145007/.578436`，但
  train/held/task-held仅`.298505/-.504329/-.129071`，q/v/action-out仍未过门。六个独立workers与两个single checkpoints均完整，
  Action Meta、held backward、validation/test与shuffled/reversed使用均为0；formal artifacts位于
  `runs/outputs/pi05_ecp_bank_compatibility_r12_s70_fdab4ae_gpu01p012_r3_20260830/`及对应step70/110 Gate roots。

- R12最早失效接口是**正确bank召回不足，不是hard-route utility或native direction capacity**。10个gradient tasks的30条正确视频中，
  17条走full时functional recovery中位`.583340`，13条走half时中位`-1.092634`；task1/73/93/94三条视频全走full并保持正收益，
  task8/52/75全走half，task9/32/72的held video也被误路由。route与fit/held恢复强对应，而wrong-bank rejection、Program margin和
  interaction已经成立。继续延长同一schedule、移动固定阈值或扫compatibility weight都不能解决task52/72等correct/wrong support
  排序冲突。

- 下一轮只作有界诊断，不把二值route登记为最终G3候选。functional primals继续唯一负责生成真实signed-pooling rank4 residual；新增共享、
  Program-conditioned compatibility probes只读取当前native bank的retained support并选择full/half坐标，不成为LoRA factor、不读取
  task/video lookup。先冻结R12 functional basin，单独验证probe的matched/mismatched分离及held-task泛化；`.80/.20`只是implementation
  预注册的诊断标准，不是owner或专家authority。即使达到也不宣布G3通过；完成该证据后暂停，与owner讨论如何向专家询问二值route
  与真正Program--bank联合方向生成的边界。这样直接检验credit ownership冲突，不改变rank、scale、operator端点、数据或阈值。

- R13诊断实现已经接通：新增的38个compatibility input heads从R12 functional input heads逐tensor初始化，functional Program、input/output
  primals与scale全部冻结；route support只读新probe，真实signed-pooling query仍只读原functional primals。31项定向合同及全仓`231/231`
  CPU测试通过。gpu01物理0/1/2的world3真实六task step1自然exit 0，global update `2.809s`，loss `.581739`，positive/negative
  support `.909952/.892334`，probe总gradient norm `.029251`且finite，native teacher reads为0；实际trainable只有38个probe weights
  `4,853,760`参数，Action Meta/source/Stage0/Natural Program/functional scorer/scale均0 trainable，峰值reserved约`19.46GB/rank`。
  当步24/38 heads非零是固定低分位support的预期稀疏credit：其余heads仍有zero gradient tensor并保持R12初始化，不是target缺失。
  该smoke只证明诊断图、参数所有权和吞吐成立，不是support泛化或G3结果。

### R13 decoupled compatibility diagnostic formal launch contract

- implementation authority为clean pushed `main@82607b64e3c515611b7b0e82d63019ceb5b2d8e9`；formal从只增加本launch记录的clean pushed
  detached descendant fresh启动，不resume、不覆盖任何现有输出。固定10 warmup + 100 effective updates，在actual step70/110各保留一个
  checkpoint；只训练38个compatibility input heads，R12 Natural Program与functional primals、source/Stage0、scale、carrier12及Action
  Meta全部冻结。该run是credit-ownership诊断；`.80/.20`是implementation可分性标准，任何结果都不把full/half二值route宣布为G3通过。
- 2026-08-30 19:58 CST紧邻launch检查：gpu01物理0/1/2均约`15MiB`、UTL 0且无compute process，3--6由他人约`34.6GB`、
  UTL 100任务占用；gpu02只有物理7完整空闲，4/5/6及0--3均有他人显存或任务，故选择gpu01 `0,1,2` world3，不跨节点拼卡。
  当前物理0 UUID为已复核可用的`GPU-658b6043-6454-1228-bffc-0e2fe22e5013`，旧prohibited设备未枚举。
- 复用gpu01 `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`的23GB frozen-condition cache；`strg01` `/data1` quota为
  `712361984/1073741824` blocks。R12同规模正式训练根为272MB，R13预计峰值新增低于0.5GB。输出固定为
  `runs/outputs/pi05_ecp_decoupled_compatibility_r13_s110_82607b6_gpu01p012_r3_20260830`且launch前不存在。
- exact entry使用`CUDA_VISIBLE_DEVICES=0,1,2`、`NCCL_P2P_DISABLE=1`、GPU-local NUMA/deferred NCCL及world3
  `torchrun scripts/train_ecp_joint_program_primal.py --config configs/pi05_ecp_decoupled_compatibility_r13_v1.json --base-config
  configs/pi05_ecp_shared_compiler_g3_v5.json --mode formal --phase joint ... --stop-after-step 110 --log-every 1`。完成step70/110后对同一
  12-task panel运行support与必要functional对照；validation/test及shuffled/reversed均不读取。证据足够区分compatibility acquisition、
  same-task video与task-held泛化后立即暂停，交由owner决定专家询问方式。

### R13 decoupled compatibility diagnostic formal结果与暂停点

- clean pushed detached `0489da362508a199236583ad9f910c73a1dd5c5c`已在gpu01物理0/1/2完成10 warmup + 100 effective
  updates；actual step70/110两个world3 checkpoints完整，训练墙钟`415.093s`。实际trainable仍只有38个compatibility input heads
  `4,853,760`参数；R12 Natural Program、functional input/output primals、scale、source、Native Stage0、carrier12和Action Meta均冻结，
  native teacher reads为0。六个独立workers随后在gpu01物理0/1/2与gpu02物理4/5/7完成两个checkpoint的完整12-task paired panel，
  全部自然exit 0；validation/test及shuffled/reversed均未读取。
- step70/110都strict non-pass。train recovery为`.298505/.483082`，same-task held均为`.048744`，true task-held均为
  `.032951`；step110 q/v/action-in/action-out为`.262289/.333634/.570474/.318540`。correct-wrong-bank与interaction中位已达
  `.688182/.843922`，但不能补偿正确视频误路由造成的闭环损失。evaluation throughput check也失败：慢worker每checkpoint约
  `765s`，相对`415s`训练墙钟ratio约`1.84`；这是次要系统non-pass，不改变科学结论。
- matched/mismatched full-route fraction由step70的`.638889/.166667`变为step110的`.666667/.166667`；support AUC由
  `.826389`变为`.831019`，逐task严格分离均为`9/12`。step110按数据角色拆分，gradient fit正确pair只放行`16/20`，第三条
  same-task held只放行`5/10`，true task-held两task共只放行`3/6`。task52/72/74仍存在正确minimum低于wrong-bank support的
  排序冲突：`.882963<.901375`、`.892215<.903325`、`.876816<.909853`。阈值无关审计显示，在wrong full-route不超过`.20`
  时，任何全局阈值最多放行`.722222`的正确pairs，因此不能靠移动固定阈值通过`.80/.20`诊断标准。
- 功能分解再次确认端点有用但门卫不可靠。step110的36条正确conditions中，24条走full时recovery中位`.572070`且minimum
  `.181790`；12条走half时中位`-.893770`且maximum仍为`-.300972`。step70到110只新增task8 video6一条full route：support从
  `.906201`升到`.906683`，仅高于固定阈值`.906623`约`.000060`，却令task8 fit recovery从`-.207688`跳到`.963754`；其held
  video仍走half并保持`-1.597779`，所以整体held与task-held逐值不变。这是离散route阈值脆弱性，不是相邻功能稳定跃升。
- R13证明把compatibility credit从functional primal中拆开会改善识别，但独立共享线性probe既未完整拟合gradient tasks，也没有
  same-task held或task-held泛化；“credit ownership冲突”是R12的问题之一，却不是充分根因。该诊断不建立G3、full/half或其它
  二值route的最终合理性。按owner最新边界，当前不续训、不扫阈值/temperature/LR/seed/weight/谱幂、不实现下一版架构；证据已经足够
  向专家询问如何让Program与当前bank共同生成唯一functional direction，现于该咨询节点暂停。
- formal artifacts：
  `runs/outputs/pi05_ecp_decoupled_compatibility_r13_s110_82607b6_gpu01p012_r3_20260830/`、
  `runs/outputs/pi05_ecp_decoupled_compatibility_r13_gate_step70_82607b6_gpu01p012_gpu02p457_w6_20260830/`、
  `runs/outputs/pi05_ecp_decoupled_compatibility_r13_gate_step110_82607b6_gpu01p012_gpu02p457_w6_20260830/`。

- half-operator task-local formal与held/wrong Gate已从clean pushed detached `55fded4`完成。10/10 tasks各自只用两条fit K1
  video、panel A与100步真实functional flow优化一个共同code；10个checkpoint均为`321,792` task-local trainable parameters，Writer/
  source/Stage0为0，Action Meta为0，held/wrong/panel-B backward均为0，仍只生成一套完整rank16。五个独立workers随后评价同一sealed
  code在same-task held bank与same-role cyclic wrong bank上的panel B结果；correct/wrong recovery中位为`.725204/.188873`，margin
  中位`.541238`，正确bank在`10/10`更好且`10/10` margin达到`.10`。bank interaction两项强过门，但correct低于预注册`.75`，所以
  总Gate严格non-pass，不能因只差`.024796`而放宽。

- 分层结果把最早失效接口锁定为half operator的fit-to-held谱坐标转移，而不是bank因果、断图或普通优化不足。训练后fit-video
  recovery中位`.950541`，held下降`.225337`到`.725204`；meta的fit/held为`.997452/.898189`，target为
  `.796767/.614878`，而旧full-inverse target correct上界仍为`.945032`。初始fit-transport held与最终held的跨task相关为约`.91`，
  普通transport cosine和teacher-factor recovery均不能解释弱task；functional优化改善9/10 task但无法消除第三video坐标偏移。
  这说明`C_B^{-1/2}`保留了足够bank特异性，却留下过强的`C_B^{1/2}`效果畸变。

- 唯一tempered `.75` zero-training bridge已从clean detached `db88418`完成。held correct/wrong recovery中位为
  `.925312/.885043`，margin仅`.054500`；correct bank在`8/10`更好，但只有`2/10` margin达到`.10`，wrong bank
  `10/10`仍有正收益，收益保留中位`.941988`。它恢复了capacity却也恢复了wrong-bank utility，与`.5`的高margin/
  低capacity及`1.0`的高capacity/零margin共同确认单一谱幂是明确Pareto，因此停止幂次调参。下一步不直接叠加新模型：先
  审计full-inverse强方向在被统一缩放到`0.02`之前的raw dual energy，检验correct/wrong bank是否已有可用的内容
  兼容信号；只有该证据成立才为full-inverse direction增加独立bounded compatibility gate。

- bank-interaction positive control的retained实现已接通，并已完成下述formal launch预检。唯一canonical operator新增固定
  `inverse_covariance_power=.5`；每个gradient task只用两条fit video的teacher初始化形成共同half-whitened task-local code，随后只以
  panel A真实functional flow优化100步。same-task held video、same-role wrong bank和panel B全部零梯度；最终分别把同一code作用于held
  正确bank与错误bank，仍只物化carrier12 + residual4的一套完整38-target rank16。trained-code诊断不读取或使用Program，Action Meta
  loader/module/parameter均为0。全量`tests/ecp`为`105 passed`，architecture guard无hard violation。结构owner仍是已有
  `PrimalDualVideoOperator`、positive-control runner与cross-bank analyzer，没有新增module、entrypoint或fallback；task-local
  half-transport/trained-code分支只活到本Gate裁决。若non-pass则在报告后退役，若pass则只把half operator保留给shared G3并在shared
  qualification接通后移除task-local执行分支，formal config/artifact与Git历史保存证据。

- gpu01物理0上的task32单步真实smoke自然exit 0：实际可训练参数只有task-local code `321,792`，Writer、source policy与Stage0均为0；
  held/wrong-bank/panel-B backward calls全为0，Action Meta module/parameter为0，唯一rank16 checkpoint完整。fit-symmetric初始化的两video
  transport alignment median为`.916812`，单步held factor diagnostic为`.683043`，三条video的panel B均高于carrier；实际train/eval/总
  计算为`9.59/10.48/23.24s`，峰值allocated/reserved为`22.38/23.66GB`。该smoke只证明实现、信息墙与吞吐，不构成Gate结果。

### G3 bank-interaction positive-control formal launch contract

- scientific authority为clean pushed `main@89b130af2fbc9c486d3e2e74349aa517be504cdc`及只增加本launch记录和吞吐配置的
  clean pushed descendant；formal从该descendant的detached frozen worktree fresh执行，不resume、不覆盖已有输出。10个gradient
  tasks分别只优化一个task-local code 100步，使用两条fit K1 video、panel A action/flow与fixed half operator；Natural Program、
  shared scorer、source、Native Stage0、carrier12、scale及Action Meta全部冻结。same-task held video、same-role wrong bank与panel B
  全程零梯度，最终只以同一sealed code在held correct / wrong bank上的真实functional recovery裁决`.75/.10/8-of-10` Gate；该Gate
  即使通过也只证明operator/task-local capacity，不冒充shared Program mapping或G3通过，且不使用shuffled/reversed。
- task32真实一步把functional microbatch从2提高到4后，train/eval/总计算由`9.59/10.48/23.24s`降至
  `8.90/9.46/20.88s`，峰值reserved由`23.66GB`升至`31.01GB`，信息墙、初始化与唯一rank16 inventory不变。由于总计算提高约
  `10.2%`且仍有安全A40余量，formal固定microbatch 4；不继续以更高batch换取很小的额外收益和更窄的共享卡余量。
- 2026-08-30 13:41 CST紧邻launch检查：gpu01物理`0,1,2`均15MiB、UTL 0且无compute process；`3--6`为他人
  `34.6GB/100%`任务。gpu02物理7完全空闲；物理4只有同一他人的两个`982MiB`进程、合计1.98GB且UTL 0，按本图实测
  `31.01GB`峰值仍留约13GB，故安全共驻；其余设备忙或峰值余量不足。所有四张完全空闲卡均优先使用，再增加这一张低占用卡，
  不抢占、暂停或干扰他人。gpu01当前物理0为已复核可用UUID，旧prohibited设备未枚举。
- 五个独立single-GPU workers按旧同图cost配对：gpu01 `p0:[93,8]`、`p1:[94,52]`、`p2:[1,72]`，gpu02
  `p7:[32,9]`、`p4:[73,75]`，每卡同时最多一个task；gpu01使用NUMA0，gpu02使用NUMA1，不跨节点NCCL。gpu01复用
  23GB `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` cache且shm尚余39GB；gpu02使用新
  `/dev/shm/ember_ecp_bank_interaction_89b130a_gpu02_20260830`，shm尚余250GB。`strg01` `/data1` quota为
  `711995060/1073741824` blocks，formal checkpoint/result预计远低于1GB；输出根已确认不存在。
- exact entry为每task设置对应`CUDA_VISIBLE_DEVICES`、GPU-local `numactl`和`PYTHONPATH=src`后运行
  `scripts/train_ecp_joint_program_primal.py --config configs/pi05_ecp_bank_interaction_positive_control_v1.json --base-config
  /data1/user/ymdai/projects/EMBER/configs/pi05_ecp_shared_compiler_g3_v5.json --mode formal --phase positive-control --task <task>
  --asset-root /data1/user/ymdai/projects/EMBER --source-run .../pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  .../checkpoints/step_00001000 --tokenizer-path .../models/tokenizers/openpi/paligemma_tokenizer.model --data-root
  .../data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir
  .../runs/outputs/pi05_ecp_g3_bank_interaction_positive_control_10task_89b130a_gpu01p012_gpu02p47_20260830/task_<id>
  --condition-cache-root <node-cache> --log-every 10`。每个worker完成第一项后立即执行配对第二项；全部sealed code完成后才启动
  `trained_code` held-correct / same-role-wrong分析与aggregate Gate。

- 两条clean detached bridge diagnostics已限定当前正控。直接把R5旧坐标交给half operator会令correct recovery中位降至`.076821`；先用
  两条fit bank分别做inverse-square-root transport再平均，则zero-training的held correct/wrong recovery中位为
  `.647543/.134170`，margin中位`.480161`，正确bank在`10/10` task更好且`10/10`达到`.10` margin。它证明half operator能恢复
  bank interaction，但correct中位仍低于预注册`.75`，所以不能冒充通过；当前100步fit-only task-local functional优化只检验能否补足
  这段capacity，而不训练shared Program mapping。

- R5成功primal的10-task cross-bank正控已在clean detached `2090799`完成。正确bank functional recovery中位
  `.930860`，same-role cyclic错误task bank反而为`.945799`；correct-minus-wrong中位`-.003819`，只有`2/10`
  task正确bank更好且`0/10`达到`.10` margin，错误bank在`10/10` task保持正收益，收益保留中位`1.003960`。
  因而R10/R11的near-zero wrong-bank不是shared Program偶然没学到，而是当前全局`C^+d` operator把高覆盖bank退化为
  可互换的实现基底：它保留task-local capacity，却没有为wrong-bank Gate提供可识别交互。下一修正先建立既保持same-task
  跨video能力、又让correct bank有必要增量的operator正控；在该正控通过前不再训练Program/scorer版本。

- R11 matched raw-Stage0 sufficiency已经在clean detached `25f38ce`完成110/110步训练及step70/110完整12-task Gate。
  step70/110 train recovery为`.218691/.292321`，held-video为`.232166/.288053`，true task-held为
  `-.139011/-.092369`；step110 task2/74分别为`.116054/-.300793`。raw task-held不但没有相对R10提高`>=.15`
  并达到`.40`，反而比R10 `.151475`下降`.243844`，因此明确排除“Natural Program schema/压缩是当前首因”。

- R11 step110 q/v/action-in/action-out分别为`.550257/.101550/.494693/.474379`；meta/target gradient-task train
  medians分别为`.470816/.110012`。在与R10逐step完全相同的task、action demo/frame和panel schedule上，R11对五个target
  tasks的训练loss全部更差，task72/73在110步中从未优于R10；但q与两类action仍可读，只有v统一坍缩。R11全部有效梯度
  finite/nonzero、无clip/teacher read/Action Meta，same-task video稳定且110相对70继续改善，故不是断图、数值错误、普通
  video variance或偶然checkpoint。该证据既不支持Stage0全量解冻，也未满足专家“所有family均缺少可读信息”的frozen-Stage0
  停止条件；它把最早接口留在raw/shared decoder函数类、target-role functional归纳偏置及Program--bank交互之间。

- R11六卡训练墙钟`1435.70s`，最大reserved `32.94GB`；两个checkpoint由六个独立workers完整评价，step70/110主循环墙钟
  `571.45/502.77s`，raw rows、aggregate、completion及两个world6 checkpoints均完整，无OOM/NCCL/NaN或掉worker。正式
  artifacts为`runs/outputs/pi05_ecp_raw_stage0_sufficiency_r11_s110_0590f63_gpu01p012346_r6_20260830/`及对应
  `...r11_gate_step70/110_25f38ce_gpu01p012346_w6_20260830/`。

- matched raw-Stage0 sufficiency retained执行面已经接通。R11从与R10相同的R9 step110完整Writer初始化，保持相同12-task split、
  panel-A真实functional flow、110步预算、scorer容量、native heads、bank/operator/carrier/scale和Gate；唯一输入改动是绕过Natural
  Program process fusion与canonical aligner，让同一scorer直接读取exact-language embedding、first/final patch relation、two-probe
  owner/event process、presence、uncertainty及normalized event time。该NaturalProgram-shaped对象只是scorer的固定shape carrier，R11
  明确是diagnostic而非deployment Writer。

- 20项定向CPU合同通过；gpu01物理0上的真实六task单rank step1也通过。初始mean functional loss`.112984`，language/scene及全部233
  native heads梯度finite/nonzero，native teacher reads为0；Action Meta、source/Stage0、process fusion、aligner、feature chart、scale均
  0 trainable，实际trainable `11,061,760`，peak reserved `21.88GB`。单rank因串行承担六个task用`53.94s`，只作图与materialization
  smoke；formal仍按world6每rank一task获取真实吞吐。

### R11 matched raw-Stage0 sufficiency formal launch contract

- implementation authority为clean pushed `main@0590f63f5f0f98f687818cedc596155f171410f1`；formal从包含本launch记录的clean
  pushed detached authority启动，不resume、不覆盖现有输出。训练为10 warmup+100 effective、actual step70/110，之后用同一12-task
  panel-B Gate只读评价；R11只作根因诊断，即使内部Gate通过也不是deployment checkpoint或整个G3 pass。
- 2026-08-30 10:07 CST紧邻launch检查：gpu01物理`0--6`均仅`14--97MiB`、UTL 0且无compute process；gpu02物理0--3满载、
  6有高UTL任务，只有4/7适合而不能形成更高吞吐同节点组。选择gpu01物理`0,1,2,3,4,6` world6；当前逻辑0是已复核可用
  UUID `GPU-658b6043-6454-1228-bffc-0e2fe22e5013`，旧prohibited设备未枚举。owner预告gpu01约11--12时可能重启；按R10同图
  `24.0min`正式墙钟，10:10前launch预计约10:35结束，保留约25分钟以上安全余量，不在窗口附近续跑或启动新长任务。
- 复用gpu01 `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` 23GB frozen-condition cache，shm尚余39GB；
  `strg01` `/data1` quota为`711675436/1073741824` blocks，预计新增低于1GB。exact entry使用`CUDA_VISIBLE_DEVICES=0,1,2,3,4,6`、
  `NCCL_P2P_DISABLE=1`、GPU-local NUMA/deferred NCCL和world6 `torchrun scripts/train_ecp_joint_program_primal.py --config
  configs/pi05_ecp_raw_stage0_sufficiency_r11_v1.json --phase joint --mode formal ... --output-dir
  runs/outputs/pi05_ecp_raw_stage0_sufficiency_r11_s110_0590f63_gpu01p012346_r6_20260830 --condition-cache-root
  /dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829 --stop-after-step 110 --log-every 1`；输出根已确认不存在。

- R10 clean detached formal与原12-task functional Gate已经完整结束。训练110/110步连续、墙钟`1442.70s`，step70/110两个
  world6 checkpoints完整；全部110步Program/native-head gradients finite/nonzero，native teacher reads、Action Meta、source/Stage0/
  operator/scale trainable均为0。六个独立worker在gpu01物理`0,1,2,3,4,6`完成两个checkpoint的12/12 paired panel-B与全部controls，
  raw rows、aggregate和completion均齐全，无错误签名。

- R10 step70/110 train recovery为`.532227/.559896`，held-video为`.500728/.544189`，held/train为`.940817/.971946`；相对R9
  的负值是决定性功能跃升。step110 q/v/action-in/action-out为`.645745/.614858/.717575/.548006`，same-task retention`.990228`、
  wrong-Program margin`.279494`，均通过；但task-held mean仅`.151475`（meta2 `.375386`、target74 `-.072436`），wrong-bank margin
  `.007864`、interaction`-.002683`、full-over-endpoints`.061382`，train也仍低于`.60`，所以两个checkpoint正式non-pass。

- per-role分解显示五个meta gradient tasks在step110为`.620--.947`量级（task9为`.679`，亦为正），五个target gradient tasks仅
  `.186--.499`，且未参与梯度的target74为负；同一target tasks在R5 fixed-route正控均约`.84--1.01`，故不是bank/operator/rank/scale容量。
  correct Program换wrong bank几乎不变，而wrong Program明显变差，说明R10已建立Program主导的功能route，但尚未建立对未见target task
  可迁移的Program--bank配对交互。不能靠续训、seed/LR/width/rank小扫或outer loss修复。

- 该结果符合最新专家预注册的“fit与same-task held明显可学、true task-held低”分支，当前只进入matched raw-Stage0 sufficiency probe：
  保持同一12-task split、primal scorer容量、functional loss、训练预算、冻结bank/operator/carrier/scale和完整Gate，只把Natural Program
  压缩输入替换为部署可见raw frozen Stage0 evidence。它仅诊断Program压缩、Stage0信息与shared decoder/generalization三者，不是新的
  deployment Writer；raw task-held只有相对R10提高至少`.15`且达到`.40`才允许把首因归给Program schema。

- R10 retained实现与真实world6 step1已经接通。唯一`joint_program_primal`执行面严格加载R9 step110完整Writer tensor inventory，
  但不加载旧optimizer、fixed route或task lookup；feature chart与source/Stage0/operator/carrier/scale冻结，仅Natural Program和233个
  native heads共`11,178,369`参数可训练，outer-code label/loss完全退出。18项定向CPU合同通过；gpu01物理`0,1,2,3,4,6`上的
  六任务真实panel-A PI0.5 flow step1自然exit 0，loss`.1093948570`，Program language/scene/process和native input/output
  gradient probes全部finite/nonzero，frozen parameters无gradient，native teacher reads、Action Meta module/parameter、task/video/frame
  free parameters均为0。38-target rank12+4唯一rank16对两fit视频实际完成materialization；global update`11.489s`，各卡peak reserved
  `19.62--21.80GB`，所以实现、信息墙、显存与吞吐均有formal资格。该smoke只证明真实功能梯度接通，不构成G3 Gate结果。

### R10 R9-initialized functional refinement formal launch contract

- implementation authority为clean pushed `main@731a769becdb80b1e3d470ddb00911402d5e1fb4`；formal从只增加本launch记录、
  不再改变`src/ scripts/ configs/ tests/`的clean pushed detached descendant fresh执行。初始化只加载R9 step110的Program/scorer
  model tensors，不加载旧optimizer；feature chart、source、Native Stage0、bank operator、carrier12、shared scale和Action Meta冻结，
  Natural Program与233个native heads使用fresh optimizer/scheduler。
- 科学合同固定原10 gradient tasks、每task两条fit K1 videos、panel-A跨episode PI0.5 flow；10 warmup+100 effective，actual
  step70/110保存两个single checkpoints。outer-code labels/loss、native teachers、panel B、held video、task2/74、validation/test、
  shuffled/reversed、fixed route、task lookup均不产生训练梯度。正式裁决只使用原12-task panel-B functional Gate，不以内loss替代。
- 2026-08-30 08:53 CST紧邻launch检查：gpu01物理`0--6`均无compute process、显存`14--97MiB`且UTL 0，选择
  `0,1,2,3,4,6` world6；当前逻辑0 UUID仍为已确认可用的`GPU-658b6043-6454-1228-bffc-0e2fe22e5013`，旧prohibited设备
  未枚举。gpu02只有零散空闲卡，故不跨节点拼卡。owner预告gpu01约11--12时可能重启；按R6同机制约25分钟训练预算，08时段
  launch并在重启窗口前留有充分余量。
- 复用gpu01 `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`的23GB frozen-condition cache，`/dev/shm`
  尚余39GB；`strg01` `/data1` quota为`711365652/1073741824` blocks，R9同规模输出292MB，R10峰值新增预计低于1GB。
  exact entry使用`CUDA_VISIBLE_DEVICES=0,1,2,3,4,6`、`NCCL_P2P_DISABLE=1`、deferred NCCL、GPU-local NUMA与world6
  `torchrun scripts/train_ecp_joint_program_primal.py --config configs/pi05_ecp_r9_initialized_functional_refinement_r10_v1.json
  --phase joint --mode formal ... --output-dir runs/outputs/pi05_ecp_r9_initialized_functional_refinement_r10_s110_731a769_gpu01p012346_r6_20260830
  --condition-cache-root /dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829 --stop-after-step 110 --log-every 1`；输出根已确认不存在。

- R9 clean detached formal与原12-task functional Gate已经完整结束。训练110/110步连续，step70/110 loss为
  `.354164/.334220`，两个world6 checkpoints完整；Action Meta/source/Stage0/scale trainable与native teacher reads均为0，
  Program和完整scorer梯度持续finite/nonzero。六个独立worker在gpu01物理`0,1,2,3,4,6`顺序评价两个checkpoint，12/12 tasks、
  panel-B、language/endpoints、wrong Program/bank、interaction、same-task和信息墙证据齐全，evaluation自然exit 0。

- R9 step70/110真实functional train recovery为`-.181514/-.131825`，held-video为`-.175532/-.129718`，true task-held mean为
  `-.009468/-.011724`；step110 task2/74分别为`.561846/-.585295`，五个target-role gradient tasks全部为负。与此同时step110
  q/v/action-in/action-out outer family medians为`.728694/.744085/.745741/.642526`，全部超过内部阈值；full-over-language通过，
  但full-over-endpoints `-.082736`、wrong-bank margin `-.003829`、interaction `.001946`均失败。R9因此是正式scientific non-pass；
  evaluation/training wall ratio`3.68`也未过，但不是科学结论的原因。

- 该分解证明R9确实解决了fresh coupled chart难优化的问题，却同时证明约`.60--.75`的outer方向近似不足以保证真实policy utility，
  尤其不能靠继续优化内部cosine修复target-role负收益。当前最早接口是code-to-utility，不触发raw Stage0分支，也不重开bank、operator、
  rank4、scale或Action Meta。下一R10只把预注册R9 step110 model tensors作为稳定内容坐标初始化；根据R4/R5正式证据冻结feature chart，
  让Natural Program与233个native heads接受真实panel-A cross-episode flow，outer-code loss退出。先做真实forward/backward/update和
  materialization smoke，再以同一step70/110 12-task functional Gate裁决；内部loss仍不能通过G3。

- R7之后的fit-only诊断已经把下一修正收敛到稳定功能坐标初始化。fresh Program+fresh scorer在同一10-task、110-step、
  四family等权outer-update目标下最终loss为`.550870`；绕过learned process fusion、直接使用raw Stage0 process为`.543792`，
  再加入contextual language与scene transition的完整raw Stage0版本为`.548213`。三者几乎相同，且contextual language/scene
  本身在12 task三video上可作`12/12`最近中心识别，所以当前没有证据把首因归给遗漏Program字段、视频噪声或Stage0压缩。

- R9只改变一个有因果依据的变量：从已通过R5 Gate的共享functional chart初始化scorer，然后让Natural Program与完整scorer在
  同一绝对outer-update标签下联合训练；fixed routing token和task lookup均不加载。110步最终loss`.334220`，训练任务直接
  outer recovery中位约`.7121`、四family约`.66--.75`，明显越过fresh约`.45`与R7平台，证明随机moving bilinear坐标确是一个真实
  优化瓶颈。更关键的是，R9 checkpoint直接作用于两个全程零梯度task-held时，task2/task74三video中位分别为`.7307/.5501`，
  总中位`.6404`且same-task cross-video约`.9998`；这排除了“只记住10个监督任务”的简单解释。task74仍更接近task2 label，
  总correct-vs-wrong margin为`-.0139`，所以该内部结果只赋予完整12-task functional Gate资格，不是G3 pass。

- R9 retained实现已接到唯一`joint_program_primal`执行面：加载R5 scorer但不加载fixed route，随后Program与scorer全部
  `12,648,965`参数可训练；source、Native Stage0、scale与Action Meta均冻结/缺席，task/video/frame free parameter为0。
  retained world6真实step1与disposable R9逐值一致：loss`.9462876469`、全部Program/scorer梯度probe finite/nonzero、
  native teacher reads 0、global update`1.132s`、peak reserved约`19.46GB/GPU`。17项定向CPU合同通过。下一步从clean pushed
  detached authority fresh运行step70/110，并只由真实Program→bank→signed replay→唯一rank16的原12-task panel-B Gate裁决。

### R9 stable-chart joint acquisition formal launch contract

- implementation authority为clean pushed `main@43be484cd22d73468741600bdb660fbef5975181`；formal将从只增加本launch记录、
  不再修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant fresh执行。初始化加载G2
  `c1493a1/macro20` Program和R5 passed shared scorer tensors，但不加载fixed routing token、task lookup或旧optimizer；
  Program与完整scorer随后共同训练，optimizer/scheduler fresh。
- 科学合同固定10个gradient tasks、每task两条fit K1 videos共享同一validated task-level outer-update label、10 warmup+100
  effective、actual step70/110。训练只读fit-only functional-code labels，不读panel actions、held video、task2/74、validation/test、
  shuffled/reversed或task-local scale；source、Native Stage0、current-bank operator、carrier12、shared scale和Action Meta冻结。
  R9 acquisition loss只作取得functional chart的训练信号，正式结论仍由两个checkpoint的原12-task cross-episode panel-B Gate给出。
- 2026-08-30 08:08 CST紧邻launch检查：gpu01物理`0--6`均无compute process、显存`14--97MiB`且UTL 0；按单节点上限选择
  `0,1,2,3,4,6` world6，不占第七张。gpu02物理`0--3,5,6`为他人高UTL，4有约2GB服务、7空闲，故不跨节点拼卡。
  owner预告gpu01约11--12时可能重启；本run按实测启动加训练远短于该窗口，在08时段启动，不在临近窗口发起长任务。
- gpu01复用23GB frozen-condition cache，`/dev/shm`尚余39GB；`strg01` `/data1` quota为
  `711064088/1073741824` blocks，R5同类完整run约280MB，R9预估低于1GB，峰值远低于剩余额度。exact entry使用
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,6`、`NCCL_P2P_DISABLE=1`、deferred NCCL和world6 torchrun；输出固定为
  `runs/outputs/pi05_ecp_functional_code_stable_chart_joint_r9_s110_43be484_gpu01p012346_r6_20260830/`。训练自然结束后以最多六个
  独立single-GPU workers评价step70/110；不以acquisition loss、outer cosine或checkpoint union替代functional Gate。

- 2026-08-30 R7 functional-code chart acquisition formal与完整12-task Gate已结束。clean pushed detached
  `024fc55`在gpu01物理`0,1,2,3,4,6`完成110步，110条metrics连续、step70/110 world6 checkpoints完整；训练墙钟
  `114.12s`，最大peak reserved `19,367,198,720` bytes，Action Meta/source/Stage0/scale trainable和native teacher reads
  均为0。六个独立worker随后在同六卡依次评价两个checkpoint。step70/110 train recovery为`-.183186/-.133386`，
  held-video为`-.177017/-.129792`，task2/74分别为`.580899/-.594765`与`.554032/-.576694`；step110
  q/v/action-in/action-out内部outer direction为`.725096/.737788/.743253/.642195`，但wrong Program/bank margin只有
  `.091741/-.003368`、interaction`.001064`。两个checkpoint均primary non-pass；evaluation/training wall ratio也因本轮
  acquisition训练仅114秒而为`>3.8`，但这不是科学non-pass的原因。

- R7把最早失效接口进一步收窄为**冻结R5 chart与Natural Program的联合函数类不相容**，不能靠续训或小调参修复。R5 fixed
  route在同一heads下functional-code cosine约`.9985`且真实闭环通过；R7的dense、四family等权outer-code监督虽把内部方向提高到
  `.64--.74`，110步已平台在loss约`.337`，却仍使全部target-role gradient tasks闭环为负，而同task fit/held video同步，排除
  video overfit。既有shared-scale transfer又已证明原成功方向换用相同shared scale仍有`.9398`中位recovery，故当前不重开bank、
  rank4、scale、Action Meta或functional panel。下一单变量修正是保留同一validated functional-code labels与完整Gate，但让Natural
  Program和整个primal scorer共同取得该绝对输出坐标；direct code supervision持续锚定输出，因此不同于R4/R6的functional-only
  moving-coordinate训练。先以真实fit/gradient smoke证明它能越过R7训练上限，再决定formal；内部code结果仍不能替代真实Gate。

- 2026-08-30 R6 Natural Program chart reconnect正式训练与step70/110完整12-task Gate已结束。clean pushed detached
  `1a6a59b`在gpu01物理`0,1,2,3,4,6`自然完成110步，训练墙钟`1508.97s`、最大peak reserved
  `32,937,869,312` bytes，Action Meta/source/Stage0/scale trainable与native teacher reads均为0。step70/110 train recovery
  为`.145063/.165181`，held-video为`.138406/.143114`，task-held mean为`.012597/-.034333`；step110 q/v/action-in/
  action-out为`.038887/-.025071/-.007718/.319160`，wrong Program/bank margin`.077886/.001131`、interaction`.000018`。
  两点均明显低于原Gate，R6正式non-pass；同一checkpoint不续训，也不做LR/seed/width/rank小扫。

- R6后只读坐标审计把最早接口进一步锁定。R5 fixed token通过R5 heads的functional-code cosine为`.998514`，G2 Natural
  Program通过同一heads仅`.010736`，R6最终Program分别通过R5/R6 heads也只有`.020074/.020914`；R6 heads几乎没有改变这一结果。
  同task跨video输出却约`.9994`稳定，说明模型稳定地产生了错误的公共坐标，而不是普通video抖动。对G2 Program hidden做两fit-video
  minimum-norm拟合可精确插值训练两view，但第三held view只有`.353777`，task2/74 task-held约零；这证明R5通过的fixed-token chart
  是无deployment内容几何的训练task codebook，不能靠重新拟合heads自然接回Program。

- 已完成的R7 fit-only functional-code chart acquisition保留R5已验证的38 input+195 output native heads并全部冻结，
  只训练Natural Program readers/fusion/aligner及其共享feature chart，以10个gradient tasks各自已验证的task-level positive-control
  primal作为training-only label；两条fit videos共享同一label，目标只用按四family等权、对rank gauge不变的完整outer-update direction
  cosine。R7不读取policy functional actions、counterfactual、teacher member或task-local scale，labels不进入deployment forward，
  task2/74、held video、panel B和validation/test零梯度；正式资格仍必须由真实Program->bank->signed replay->唯一rank16的原12-task
  functional Gate裁决，内部code loss不能通过G3。

- R7 dirty真实gpu01物理0单步已通过。6 tasks x 2 fit videos的初始mean acquisition loss为`.946288`，global step`4.076s`，
  peak allocated/reserved约`10.41/19.35GB`；Program language/scene/process以及program/rank context、event score、input/output trunk、
  owner/rank embedding梯度全部有限非零，冻结native heads无gradient，Action Meta/source/Stage0/scale trainable和native teacher reads均为0。
  27项定向回归、旧J3/R6/R7 config兼容加载与compile检查通过。该smoke只证明图、所有权和吞吐，不替代R7 formal/Gate。

### R7 functional-code chart acquisition formal launch contract

- implementation authority为clean pushed `main@89131fef43f7e5468adc8e4d9c4eab4ef9af1009`；formal从只增加本launch记录、未再
  修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant fresh执行，不resume R4--R6 optimizer或dirty profile。
  Program加载G2 `c1493a1/macro20`，scorer加载R5 step110；R5 native heads冻结，optimizer/scheduler fresh。
- 科学合同固定10个gradient tasks及两个true task-held：每个gradient task两条fit K1 videos共享一个validated task-level outer-update
  direction label，10 warmup+100 effective、actual step70/110。只训练Natural Program readers/fusion/aligner和共享feature chart；
  native heads、source、Stage0、current-bank operator、carrier12、scale、policy和Action Meta冻结。训练不读functional panel action、
  counterfactual、member tensor、task-local scale、held video、task2/74、validation/test或shuffled/reversed。
- 2026-08-30 05:26 CST live检查gpu01物理`0--6`均无compute进程、显存`14--97MiB`且UTL 0；按单节点最多6卡合同选择
  `0,1,2,3,4,6` world6，不占用第七张。gpu02物理0--3、5、6高UTL，4有常驻服务、7空闲，故不跨节点拼卡。R7单卡真实峰值
  reserved约`19.35GB`，六卡同一task权重有充分余量；launch紧邻时若状态变化则只按实时可用1--6卡弹性分片。
- gpu01复用`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` 23GB cache，shm尚余39GB；`strg01` `/data1` quota为
  `709,740,920/1,073,741,824` blocks，R6完整run约300MB，R7预估同量级，峰值远低于剩余额度。exact entry使用
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,6`、`NCCL_P2P_DISABLE=1`、GPU-local NUMA/deferred NCCL及world6 torchrun；输出固定为
  `runs/outputs/pi05_ecp_functional_code_chart_acquisition_r7_s110_89131fe_gpu01p012346_r6_20260830/`。自然完成后用最多六个独立
  single-GPU workers依次评价step70/110原12-task Gate；不以acquisition loss、fit code cosine或checkpoint union替代真实functional结果。

- 2026-08-30 R5 fixed feature-chart正式训练与step70/110 paired Gate已完整结束。训练从clean pushed detached
  `9e6b6a7`在gpu01物理`0,1,2,3,4,6`自然完成110步，110 rows连续、两个world6 checkpoint完整，墙钟
  `1503.63s`、最大peak reserved `32,916,897,792` bytes；Action Meta、source/Stage0/scale trainable和native teacher reads
  均为0。六个独立worker随后在物理`0--5`完成同一paired Gate。step70/110 train recovery为`.933583/.940336`，held-video
  为`.957202/.963277`；step110 q/v/action-in/action-out为`.815834/.839439/.820583/.837113`，wrong-token margin
  `.895772`、same-task retention`1.006591`、held/train`1.024396`。两点全部primary checks通过，step110相邻稳定性也通过，
  所以R5 Gate正式pass。该结果直接确认R4唯一缺口是初始化后feature chart漂移；冻结chart后233 heads既能接受真实functional
  gradient，又能完整保留强功能坐标。R5仍含fixed task route和privileged一次性初始化，不是deployment Writer或G3 pass。

### R6 Natural Program chart reconnect formal launch contract

- implementation authority为clean pushed `main@43dca6b8de937ea74fcbe38bd0a9d464f8619644`；formal从只增加本launch记录、未再
  修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant fresh执行，不resume J2/J3/R1--R5 optimizer或任何dirty
  profile。初始化只加载R5 step110通过Gate的共享scorer tensors及G2 `c1493a1/macro20` Program tensors；fixed token与task lookup均不
  是checkpoint参数且不进入forward；optimizer/scheduler fresh。
- 科学合同固定为12 tasks：gradient meta`[1,8,9,32,52]`、gradient target`[72,73,75,93,94]`、true task-held`2/74`；每个
  gradient task两fit K1 videos与panel A，10 warmup+100 effective、actual step70/110。只训练Natural Program readers/fusion/aligner
  与233 heads；feature chart、source、Native Stage0、current-bank dual/exact replay、carrier12、scale和Action Meta冻结。loss只有正确
  generated rank16的cross-episode PI0.5 flow；panel B、same-task held、task2/74、validation/test零梯度，不使用shuffled/reversed。
- 2026-08-30 03:41 CST live检查gpu01物理0/1/2完全空闲；3/4为他人约`5.33GB/0%`、5约`1.67GB/0%`，profile自身peak
  reserved`21.88GB`，有充分余量；物理6约`5.34GB/99%`故排除。gpu02物理0--3约30GB，5/6重占用，4仅约2GB、7空闲但不跨节点
  拼卡。计划紧邻launch复查后用gpu01物理`0--5` world6；若3/4/5任一卡升为持续高UTL，则按实际合适卡弹性降为world5/4，不改变
  task权重或科学batch，不等待凑6、不干扰他人。
- gpu01复用`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` 23GB cache，shm尚余39GB；`strg01` `/data1`
  quota为`709,430,440/1,073,741,824` blocks，R5完整run 280MB，R6预计同量级，峰值远低于剩余额度。exact entry使用
  `NCCL_P2P_DISABLE=1`、GPU-local NUMA/deferred NCCL；预注册输出为
  `runs/outputs/pi05_ecp_natural_program_chart_reconnect_r6_s110_43dca6b_gpu01p012345_r6_20260830/`。自然完成后用最多六个独立
  single-GPU workers依次评价step70/110完整12-task Gate；不以训练loss或内部geometry替代Gate。
- 03:43 CST紧邻launch复查时物理5的他人进程升至`100%` UTL、物理6降至`0%`，故实际排除5并使用gpu01物理
  `0,1,2,3,4,6` world6。formal authority为clean pushed detached
  `1a6a59bdc2f623dc76c89f4cac3b5e9f279c351b`，实际输出为
  `runs/outputs/pi05_ecp_natural_program_chart_reconnect_r6_s110_1a6a59b_gpu01p012346_r6_20260830/`；只发生实时调度、
  provenance命名与docs-only authority变化，科学config、初始化、loss、budget和Gate均未改变。

- 2026-08-30 R3 grouped-output formal与step70/110 Gate已完整结束。step110 train/held-video recovery为
  `.305293/.287486`，q/v/action-in/action-out为`.285260/.277323/.656235/.668922`，wrong-token margin
  `.118578`、same-task retention`.986958`。R3确实修复action-in并保持action-out、路由因果和跨video稳定，但train/held及q/v仍未过
  `.60/.50/.35/.35`，所以明确non-pass，不能接回Natural Program或冒充G3通过。

- R3后续只读根因分解排除了继续调critic权重、family权重或scale。六task上fit-only critic相对真实functional gradient的全局cosine
  median为`-.148903`，q为`-.269630`；成功task-local code的直接监督梯度相对functional也只有median`.032449`，故二者都不是适合
  继续叠加的joint loss。相反，把成功code原方向换成R3冻结shared scale后，六task真实functional recovery中位`.939783`、范围
  `.753420--1.023163`，6/6均`>=.60`。因此最早剩余接口不是bank/operator/rank4/scale，而是fresh shared scorer从随机点发现强双因子
  方向的credit/initialization。

- 当前R4 fixed-route边界只在训练开始前用十个已formal通过的task-local functional code，按现有owner×group hidden做一次FP64
  minimum-norm shared-head插值；task-local scale不加载，之后只使用真实panel-A functional loss，critic完全删除。clean detached
  `0b51c57`已在gpu01物理0--5自然完成全部110步formal，墙钟`1517.15s`，actual step70/110两个single checkpoints及六rank
  optimizer/scheduler/sampler/RNG状态完整。step70/110前十个training visits的recovery中位分别`.7132/.7887`，minimum view分别
  `.4735/.5648`；早期warmup把step0约`.962`中位一度扰动到约`.30`，随后稳定恢复，说明强初始化没有随functional-only训练系统崩落。
  全程Action Meta module/parameter为0、native teacher reads 0、source/Stage0/scale trainable均0，最大peak reserved
  `32,944,160,768` bytes，无OOM/non-finite。原step70/110 paired Gate也已完整结束：step110 train/held-video为
  `.819437/.839139`、q/v/action-in/action-out为`.439578/.388131/.249310/.400750`，wrong-token margin`.913637`、same-task
  retention`1.002751`；11项主检查只剩action-in低于`.30`，因此R4 formal仍为non-pass，但已证明fixed route、真实bank/operator、
  shared scale、functional-only训练和跨video稳定可共同保留强功能解。

- R4 action-in只读反事实已把最早失效接口锁定为**初始化后feature chart漂移**，不是head、scale或训练不足。真实step0 action-in
  outer recovery约`.999988`，step70/110降为`.298330/.301140`；在各checkpoint当前hidden上只重拟合33个action-in heads即可恢复到
  `1.0`，hidden仍为full rank`40/40`。initial→step110的all-head relative drift仅`.000810`，feature-chart drift为`.008930`；把initial
  heads接到checkpoint chart仍只有`.301099`，把checkpoint heads接回initial chart则保留`.998320`。program/rank context及input/output
  trunk任一单独移动都可破坏action-in，说明这是整条特征链的distributed coordinate co-adaptation，而非单模块断图或超参问题。

- 当前R5只改变这一项参数所有权：同样一次性functional-code初始化后冻结完整feature chart，仅让38 input和195 output native heads
  接受原functional loss。13项定向CPU合同与gpu01物理0单卡真实step已通过；实际trainable为`10,297,344`、Action Meta/source/
  Stage0/scale均0，全部233 heads有finite nonzero gradient，frozen chart无gradient，native teacher reads 0，唯一rank16被真实policy消费；
  单卡step`53.05s`、peak allocated/reserved`21.19/21.87GB`。该profile只证明执行面，不冒充R5 Gate或G3通过。

### R5 fixed feature-chart formal launch contract

- implementation authority为clean pushed `main@9c5ff0d6772c7ac974273b30f61545a1662cf067`；formal从只增加本launch记录、未再修改
  `src/ scripts/ configs/ tests/`的clean pushed detached descendant fresh执行，不resume R4或任何旧G3 checkpoint；
- 科学合同与R4完全相同：10 gradient tasks、每task两fit K1 views、panel A、10 warmup+100 effective、actual step70/110、fixed
  routing token、一次fit-only functional-code head初始化、真实X/Y、current-bank dual/exact signed replay、frozen shared scale和唯一rank16。
  唯一变量是初始化后feature chart冻结，optimizer只含233 native heads；critic、task-local scale和teacher tensor不读，panel B、same-task
  held、task2/74、validation/test零梯度，Action Meta 0，shuffled/reversed不用；
- 2026-08-30 02:37 CST live检查gpu01物理0/1/2完全空闲，3--6为他人约`5.3--5.45GB`且`0%` UTL轻进程；单卡profile自身
  peak reserved`21.87GB`，即便共驻也有约18GB余量。计划用物理`0--5`做world6，并在实际launch紧邻时重查，若任一卡变为高UTL则
  换用当时低UTL的同节点卡或按可用1--6卡弹性分片，不等待、不抢占。gpu02只有物理7全空，物理4约2GB，其余重占用，故不跨节点；
- gpu01复用现有23GB condition cache，`/dev/shm`尚余39GB；`strg01` `/data1` quota为`676.3G/1T`，R4完整run仅302MB，
  R5预计同量级且远低于剩余额度。exact entry使用`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL的既有torchrun；原计划输出为
  `runs/outputs/pi05_ecp_routing_functional_code_chart_frozen_r5_s110_9c5ff0d_gpu01p012345_r6_20260830/`。自然完成后以最多六个独立
  single-GPU workers顺序评价step70/110同一paired Gate；不以训练loss、step0或内部geometry替代Gate。
- 02:39 CST紧邻launch重查时，物理5的他人进程升至`100%` UTL，故未与其共驻；物理3已降至约`1.64GB/0%`且物理6仍约
  `5.34GB/0%`，实际选择`0,1,2,3,4,6`做world6。formal authority为clean pushed detached
  `9e6b6a71cb724bcfe37cb915902c2da7df18de28`，实际输出为
  `runs/outputs/pi05_ecp_routing_functional_code_chart_frozen_r5_s110_9e6b6a7_gpu01p012346_r6_20260830/`；只发生调度与
  provenance命名变化，科学配置、初始化、loss和Gate均未改变。

- owner在`2026-08-30T01:05:31+08:00`留下本轮真实推进锚点；此后进度按该绝对时间记录，不用对话压缩中的相对时点替代。

### R4 functional-code initialization formal launch contract

- implementation authority为clean pushed `main@69a6b2440a33d614dbad6295fde0685524365be8`；formal从仅增加本launch记录、
  不再改训练图/config的clean pushed detached descendant fresh执行，不resume R1--R3/J2/J3或dirty profile；
- 数据、10 tasks、两fit K1 views、panel A、10 warmup+100 effective、actual step70/110、fixed routing token、真实X/Y、
  current-bank dual/exact replay、frozen shared scale和唯一rank16与R3相同。唯一变化是optimizer构造前用10个fit-only functional
  positive-control code一次性插值现有shared heads，随后只保留correct functional loss；task-local scale、critic、teacher tensor均不进入
  训练forward，held/panel B/task2/74/validation/test零梯度，Action Meta 0；
- 2026-08-30 01:17 CST live检查gpu01物理0--6均仅有他人约`2.4--3.2GB`、`0--3%` UTL进程，R4单卡实测自身
  peak reserved`21.882GB`，叠加后仍有约20GB以上余量且不会干扰。按单节点上限选物理`0--5`做world6，不等待完全空卡；gpu02
  0--3约30GB、5--6较重、4/7虽轻但不跨节点拼卡。显存只以不OOM、安全余量和真实吞吐判断，不执行35GB阈值；
- gpu01复用`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`现有23GB cache，`/dev/shm`尚余39GB；
  `strg01` `/data1` quota为`708770936/1073741824KiB`、约余348GB，formal输出预计远低于1GB且root启动前不存在；
- exact command使用`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  TOKENIZERS_PARALLELISM=false PYTHONPATH=src .../.venv/bin/torchrun --standalone --nproc-per-node=6
  scripts/train_ecp_joint_program_primal.py --config configs/pi05_ecp_routing_token_functional_code_init_r4_v1.json --mode formal
  --phase joint`及既有source/tokenizer/data/base/cache authorities；输出固定为
  `runs/outputs/pi05_ecp_routing_functional_code_init_r4_s110_69a6b24_gpu01p012345_r6_20260830/`。训练自然到step110后用同一clean
  authority六个独立workers评价step70/110；不以step0、loss或内部code fit替代Gate，不运行shuffled/reversed。

### R4 step70/110 formal Gate launch contract

- evaluator scientific authority固定为clean pushed detached `0b51c57c1c266931bd56b23f63aece7dbf65a50e`；训练authority固定为上述
  naturally completed R4 run及其actual step70/110，不融合checkpoint、不重新初始化head、不读task-local scale；
- 2026-08-30 01:50 CST live检查gpu01物理0/1/2完全空闲，3--6为他人约`.49--3.14GB`轻进程；紧邻launch的二次检查发现
  物理3已升到`99%` UTL，故不按旧快照共驻，最终选择物理`0,1,2,4,5,6`做六个独立single-GPU workers；其中4/5/6约
  `3.01/3.14/3.08GB`且当时`0%` UTL，仍有充分峰值余量。gpu02物理0--3约30GB、5/6高UTL，只有4/7较空，不如gpu01
  同节点组合；不跨节点、不NCCL、不占卡等待；
- 复用gpu01的23GB condition cache与1.1GB endpoint cache，`/dev/shm`尚余39GB；`strg01` `/data1` quota为
  `709087004/1073741824KiB`、约余348GB。每worker只加载一次runtime，按预注册wall-time cost queue承担任务并顺序评价两个
  checkpoints；输出固定为
  `runs/outputs/pi05_ecp_routing_functional_code_init_r4_gate_step{70,110}_0b51c57_gpu01p012456_w6_20260830/`；
- Gate仍只认train/held-video、held/train、q/v/action-in/action-out、wrong-token、same-task retention及step70--110相邻稳定性。
  panel B与held video零梯度，fixed token/privileged初始化只作training-only边界解释，validation/test及shuffled/reversed均不使用。

- 2026-08-29 clean pushed `67a49f8`的R3三卡真实profile完成：同一global update由每rank两task执行，墙钟`25.334s`，
  三rank最大reserved为`21.815/20.684/20.462GB`（十进制bytes），所有owner×group output heads均有finite nonzero gradient，
  aggregate output-head gradient norm `.005195`。run contract实测trainable参数`11,767,940`，Action Meta module/parameter均0，source、
  Native Stage0、scale均0 trainable，task/video/frame-free parameter为0，仍只生成一套完整38-target rank12+4 rank16。该三卡结果只证明
  图和资源合同，不冒充六卡吞吐。当前formal改用可用卡数决定的world size；六task梯度始终按固定`1/12`逐view加权并SUM all-reduce，
  因此只改变任务在rank间的分配与reduction低位顺序，不改变task权重。Gate现按checkpoint记录的1--6 world topology读取对应rank states，
  不再错误硬编码world6。

- 2026-08-29 R2 formal及完整Gate已结束，step110 train/held-video为`.205796/.193603`，低于R1；q/v/action-in/action-out则由
  R1近零提高到`.220453/.407617/.166808/.663453`，held/train`.940749`、same-task retention`.978070`、wrong-token margin
  `.090559`。这证明critic进入并恢复了v/action-out，但q/action-in仍受group-shared decoder限制，不完整组合也损害真实functional。
  critic recovery最后20步平台约`.322`，不续训或扫weight。固定R2 hidden的解析反事实给出shared-group q/action-in上限
  `.691/.392`，owner×group heads则四family median/min均为`1.0/1.0`、每组hidden rank `40/40`。当前实现面已进入R3：只替换
  output group heads，保持R2其余机制与Gate；20项定向CPU合同已通过，尚未启动真实profile或formal。

- 2026-08-29 owner纠正此前由执行面自行加入的`<35GiB/GPU`限制：它不是当前authority，也不再作为后续profile或launch硬门。
  最长真实样本、allocator波动和共驻进程仍有安全余量且不OOM时，可以使用更高显存；运行选择以真实吞吐和持续UTL为准。
  已从active requirements/design/plan移除该人为上限。当前从detached `a4b91bb`启动的R2配置仍把旧值记录为该run的历史provenance，
  但runtime代码不执行此阈值、当前run也不会因越过35GiB自动停止；后续配置不再继承该字段。

## 2026-08-29 R1 partial/non-pass，R2 fixed-route set-valued critic接通

R1已在clean detached `8c213c5`完成110-step formal与step70/110六worker Gate。step110 train/held-video recovery为
`.267809/.279828`，held/train`1.044879`，wrong-token margin`.238352`，same-task retention`.990982`；相对J2 fit中位`.1708`
提升约57%，10 tasks中9个改善，证明固定清晰route被scorer实质使用且跨video稳定。但q/v/action-in/action-out仅
`.003698/.007820/.001111/.033335`，train/held与四family主Gate仍明确non-pass；结果不是G3通过。

只读checkpoint几何显示四family hidden cross-task cosine约`.18--.27`，q/v正确task-local code检索8--9/10，排除route在scorer内部再次
坍缩；但coupled primal alignment只有约`.0015/.0051/.0085/.0675`。step70--110各family参数更新同量级；固定hidden最优last-head
least-squares可精确拟合全部input及v/action-out output，但q/action-in output受现有group-shared 128D head限制在约`.658/.363`。
同时，J2 task-local正控由teacher consensus初始化，step1已经拥有最终功能收益的中位`.431`，因此它没有证明functional loss能从随机
方向发现强解。R1的精确结论是：Natural Program近公共表示确是一个瓶颈，但即使route清晰，functional-only credit仍只找到
action-out shortcut，且grouped-output函数类另有容量限制。

R2保持R1全部部署图与Gate，只为gradient-task fit views加入已有fit-only、set-valued paired-update critic；teacher/member不进入
deployment，held/panel B/task2/74/validation/test零梯度。weight1六卡真实一步在gpu01物理`0,1,3,4,5,6`为`15.094s`、最大reserved
`20.29GiB`、联合gradient norm`.1201`，相同初始化R1 functional-only norm`.02575`；据此formal固定weight`.2`作一次量纲校准，避免
critic压倒functional，不做weight/LR/seed sweep。下一步是完成clean集成、formal launch contract，然后fresh step70/110与原Gate。

关键artifacts：

- `runs/outputs/pi05_ecp_routing_token_control_r1_s110_ec86fdb_gpu01p123456_r6_20260829/`；
- `runs/outputs/pi05_ecp_routing_token_control_r1_gate_step110_8c213c5_gpu01p123456_w6_20260829/`；
- `runs/analysis/pi05_ecp_routing_critic_r2_weight1_profile_dirty_gpu01p013456_r6_20260829/`。

### R2 fixed-route set-valued critic formal launch contract

- implementation authority为clean pushed `main@6c41926417ca7985583994e59b8c9fc52658d782`；formal从只新增本launch contract、
  不再修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant执行。scorer、optimizer、scheduler全部fresh，不resume
  R1/J2/J3或dirty profile；
- 数据固定为10个gradient tasks、每task两条fit K1 videos、panel A 16 visits和fit-only consensus member set。每view同时使用原correct
  functional primary与weight`.2`的set-valued paired effective-update critic；第三same-task video、panel B、task2/74、validation/test
  全部零梯度。固定token只由训练期authority ID选择；source、Native Stage0、Natural Program、current-bank operator、scale、carrier、
  Action Meta与policy weights冻结，唯一trainable仍为`ProgramNativePrimalScorer`；
- world6固定每step 3 meta+3 target、每task两video，10 warmup+100 effective，在actual step70/110保存single checkpoints。exact training
  entry为`CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_ecp_joint_program_primal.py --config
  configs/pi05_ecp_routing_token_critic_r2_v1.json --base-config /data1/user/ymdai/projects/EMBER/configs/pi05_ecp_shared_compiler_g3_v5.json
  --mode formal --phase joint`，其余source/tokenizer/data参数沿R1 authority；condition cache固定复用
  `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`，不重建Stage0/X/Y；
- formal输出根固定为`runs/outputs/pi05_ecp_routing_token_critic_r2_s110_6c41926_gpu01p013456_r6_20260829/`，log使用同stem；launch前
  root不存在。R1同结构train+Gate仅约`206MiB`，本轮预计低于`1GiB`；2026-08-29 21:12 CST `/data1` quota为
  `708237836/1073741824KiB`，剩余约`348.6GiB`，`/dev/shm` cache为`23GiB`且尚余`39GiB`；
- 2026-08-29 20:59 CST live GPU检查：gpu01物理3/4/6完全空闲，5仅`.10GiB/0%`，0/1为他人`.47/.55GiB`且`0/4%`，峰值余量
  充分；物理2为他人`3.1GiB/49%`，故避开。gpu02物理0--3各约`30GiB`，5/6忙，4/7虽低但不足同节点六卡；不跨节点拼卡。
  选择gpu01物理`0,1,3,4,5,6`，runtime按真实PCI自动绑定各GPU本地NUMA，deferred NCCL且`NCCL_P2P_DISABLE=1`；
- formal训练完成后，从同一clean authority用六个独立single-GPU workers依次评价actual step70/110，只跑R1已预注册的三条correct views、
  wrong-token/correct-bank与八target family必要臂。Gate仍为train/held `.60/.50`、q/v `.35`、action-in/out `.30`、wrong-token
  margin`.10`、retention与相邻稳定；critic loss或内部cosine不能替代functional Gate，shuffled/reversed不使用。

## 2026-08-29 J3 Gate non-pass，R1 routing-token边界对照接通

J3 step70/110六worker paired Gate已全部自然完成。step70 train/held-video recovery为`.136913/.131572`，step110为
`.148649/.147689`，都低于J2对应`.159588/.148662`与`.170800/.164623`；step110 q/v/action-in/action-out仅
`.000466/-.004513/.008217/-.001500`。wrong Program、wrong bank和interaction为`.010192/.012540/.005426`，仍远低于`.10/.10/.05`
Gate。逐task correct fit只有task52/72/75三项优于J2，而wrong Program、wrong bank、endpoints分别有8/10、7/10、9/10改善：J3确实收到
counterfactual credit，但主要学会让negative变坏，没有学会正确task-specific positive route。该结果满足active design停止继续
contrastive/normalization/optimizer技巧的条件；raw Stage0 probe不适用，因为train与held-video从未变强。

当前单一R1边界对照给10个gradient tasks固定、非参数化、均值零且两两正交的128D routing token，只训练现有
`ProgramNativePrimalScorer`。同一task两fit video共享token，held video与panel B零梯度；每条video仍独立提供真实X/Y/current-bank
operator，最终仍是唯一完整rank12+4 rank16。该对照显式标记`deployment_candidate=false`，不能证明shared Natural Program mapping或G3
通过；Gate解释后、下一canonical architecture实现前删除执行代码，由Git/formal artifacts保留。

定向CPU合同`8 passed`，10-token Gram精确为`128I`且每token均值0。gpu01物理1--6真实一步profile全部复用既有23GiB `/dev/shm`
condition cache，没有重建Stage0/X/Y；只有scorer的`7,512,196`参数trainable，source/Stage0/Natural Program/operator/scale/Action Meta
trainable均0，native teacher tensor reads 0，所有关键scorer gradient probes有限非零。global step为`12.383s`，per-rank peak reserved
最大`21,778,923,520` bytes（`20.28GiB`），符合六卡`<=15/25s`与`<35GiB`合同。下一步是完成diff审查、clean main集成，随后从detached
authority fresh运行step70/110 formal与只含correct/wrong-token/family必要臂的六worker Gate。

### R1 routing-token formal launch contract

- implementation authority为clean pushed `main@ec86fdb74b4271b0a70a65db30b5c6af82168d02`；formal从包含本段launch contract、但不再修改
  `src/ scripts/ configs/ tests/`的clean pushed detached descendant执行。scorer、optimizer和scheduler fresh，不resume任何J2/J3/profile；
- 数据固定为10个gradient tasks、每task两fit K1 videos、panel A的16 visits；第三same-task video、panel B、task2/74、validation/test全部
  零梯度。训练只有correct functional loss，不含J3 counterfactual；Natural Program模型、source、Stage0、bank operator、scale、carrier和
  Action Meta冻结，唯一trainable为现有`ProgramNativePrimalScorer`；token由training-only authority ID确定，不能作为deployment输入；
- world6固定global每step 3 meta+3 target、每task两video与原optimizer cadence；10 warmup+100 effective，在actual step70/110保存唯一
  checkpoints。exact command使用gpu01物理1--6、GPU-local NUMA、deferred NCCL、`NCCL_P2P_DISABLE=1`，入口仍为
  `scripts/train_ecp_joint_program_primal.py --config configs/pi05_ecp_routing_token_control_r1_v1.json --base-config
  /data1/user/ymdai/projects/EMBER/configs/pi05_ecp_shared_compiler_g3_v5.json --mode formal --phase joint`，其余source/tokenizer/data参数沿J3
  authority，condition cache固定`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`；
- formal输出根固定为
  `runs/outputs/pi05_ecp_routing_token_control_r1_s110_ec86fdb_gpu01p123456_r6_20260829/`，log固定为同stem的`runs/logs/*.log`；launch前
  已确认output root不存在。预计checkpoint/log远低于`1GiB`，不复制dataset/model/cache；
- 2026-08-29 19:52 CST live检查：gpu01物理3/4/6完全空闲、5仅`.10GiB/0%`，1/2各有他人约`2.98GiB/5%`短进程；以profile
  `20.28GiB`峰值仍保留约`22.8GiB`余量且不会显著干扰，故使用1--6。gpu02物理0--3与5满载、6占`17.6GiB`、7占`3.0GiB/8%`，
  没有更优同节点六卡组；不跨节点拼卡。`/data1` quota为`708013312/1073741824KiB`，剩余约`348.8GiB`；
- formal完成后只用同一clean authority的六个独立single-GPU workers评价actual step70/110，每worker顺序复用一次runtime；只计算三条correct
  views、wrong-token/correct-bank和八target family必要臂，不运行language/endpoints/wrong-bank/true-task-held冗余arms。Gate不以训练loss、
  hidden cosine或checkpoint union代替functional recovery，shuffled/reversed仍不使用。

## 2026-08-29 J3 formal训练完成，step70/110 Gate启动

clean detached `f8bfb7a`在gpu01物理`0--5`自然完成全部110 optimizer steps；actual step70/110两个single checkpoints、
optimizer/scheduler/sampler/RNG与六rank state均已落盘，输出根为
`runs/outputs/pi05_ecp_j3_counterfactual_program_primal_12task_s110_9af7c19_gpu01p012345_r6_20260829/`。训练主体约
44分钟，单步约`17.7--27.1s`，六卡peak reserved最大约`30.63GiB`，没有OOM、non-finite、gradient clip、native teacher
tensor read或Action Meta漂移。counterfactual normalized gap随训练有所增加但远未稳定达到`.10` margin；这是定位信号，不能替代
预注册Gate。

### J3 step70/110 formal Gate launch contract

- evaluator scientific authority固定为clean pushed detached `3f6f94ee4a1bf3a930142d585d352b271e048737`；training authority为上述
  clean detached `f8bfb7a`及其同一run中的actual step70/110，不融合checkpoint，不新增训练或选模信号；
- 六个独立single-GPU workers各负责两个balanced tasks，每个worker只加载一次source/Stage0/Writer runtime并按step70、step110顺序
  评价，复用`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829` condition cache与
  `/dev/shm/ember_ecp_j2_gate_endpoints_2cd4091_gpu01_20260829` endpoint cache；worker之间不用NCCL；
- 2026-08-29 launch前live检查gpu01物理3/4/5/6完全空闲，1/2仅有他人约`.5GiB`、`0--1%` UTL轻进程且峰值余量充分，
  0约`.9GiB/7%`故避开；gpu02没有更合适的同节点六卡组。选择gpu01物理`1--6`，每worker由runtime绑定GPU-local NUMA；
  `/data1` quota为`708009412/1073741824KiB`，formal training根仅`226MiB`，两个Gate根预计远低于剩余约`349GiB`；
- 每个worker执行`CUDA_VISIBLE_DEVICES=<1..6> PYTHONPATH=src ... evaluate_ecp_joint_program_primal.py worker`，共同使用canonical
  J3 train/gate/base configs、source checkpoint、tokenizer、dataset、compiler run、step70/110 checkpoints及上述两cache roots，
  `--worker-index <0..5> --worker-count 6`；输出根固定为
  `runs/outputs/pi05_ecp_j3_counterfactual_gate_step{70,110}_3f6f94e_gpu01p123456_w6_20260829/`。六worker成功后分别aggregate，
  step110只以step70 aggregate作为相邻稳定性previous report；失败或不完整输出不得冒充formal evidence；
- Gate仍只认gradient-task train recovery、same-task held-video retention、两task true task-held、四family、correct-vs-wrong
  Program/bank/interaction、信息墙与相邻稳定性。内部loss和训练时hinge不构成通过，shuffled/reversed不在本阶段使用。

## 2026-08-29 J2 formal non-pass，J3 counterfactual functional routing启动

clean detached `5fd80b6`在gpu01物理`0--5`完成12-task J2全部110 optimizer steps（10 warmup+100 effective），保存actual
step70/110两个single checkpoints；训练自然结束，所有Program/scorer gradient finite且非零，Action Meta、source、Stage0、operator、
carrier和scale均冻结。step70/110的gradient-task train recovery为`.159588/.170800`，same-task held-video
`.148662/.164623`；step110 task-held task2/task74为`.122798/-.109179`，四family medians均约零，correct-vs-wrong
Program/bank仅`.008033/.007142`、interaction`.002387`。same-task retention`.9771`、held/train`.9638`、event/K1和信息墙通过；
结果仍远低于Gate，不能以稳定或video retention冒充primary通过。正式报告位于
`runs/outputs/pi05_ecp_j2_joint_program_primal_12task_s110_1d775a4_gpu01p012345_r6_20260829/`及两个
`pi05_ecp_j2_joint_gate_step{70,110}_2cd4091_gpu01p012345_w6_20260829/`根。

non-pass后的零optimizer-step checkpoint110审计排除gradient clip、断图和单纯video overfit。十task Program/primal pairwise cosine
median约`.93--.95`，generated effective update median`.678`且action-in`.997`；成功task-local free-primal input/output code
median却只有`.203/.149`。同一functional panel的task-gradient pairwise cosine median`-.0229`、`62.22%`为负，预注册六task组
cancellation ratio`.4208--.5360`。纯correct-pair functional loss因此学到common residual；它没有建立Program/bank对task-specific
policy effect的ownership。审计artifact为
`runs/analysis/pi05_ecp_j2_checkpoint110_gradient_geometry_2cd4091_gpu01p5_20260829/report.json`。

当前只做一个有机制依据的J3修正：在原joint执行面保留correct functional primary，增加same-role cyclic、与correct共享task panel和
policy RNG的单条counterfactual view；wrong Program与wrong bank逐step交替，用bounded margin在满足后停止反向推动。该修正不增加
deployment参数、输入或forward，不改native bank、rank、Stage0、scale或Action Meta。每task从2次变3次policy functional forward，六卡
目标仍`<=30s/update`。

dirty六卡真实step0 qualification已经通过。六个same-role cyclic wrong-Program pairs均与correct共享task panel、policy RNG和bank view，
6/6初始normalized gap接近0且hinge激活，符合J2的common-residual诊断；完整gradient到达Program/scorer，native teacher tensor reads为0，
Action Meta module/parameter为0，38-target唯一rank16及冻结ownership不变。global step`21.696s`，较J2两forward的`11.73s`符合三forward
规模且低于30秒目标；六卡peak reserved最大`20.395GiB`，全部冻结cache命中。确定性合同锁定step1切换wrong-bank和第二fit view，不为
该同构分支重复一次完整加载。profile位于
`runs/analysis/pi05_ecp_j3_counterfactual_world6_profile_dirty_gpu01p012345_20260829/`。下一步是clean commit/push和detached fresh
12-task/100-effective formal，不从J2 checkpoint resume。

### J3 12-task counterfactual formal launch contract

- scientific implementation authority为clean pushed `main@9af7c194df2f121369e2f4ad7098563b6ddcc3fd`；formal从包含本合同且不再
  修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant执行。Natural Program仍从`c1493a1/macro20` model tensors
  初始化，primal scorer、optimizer和scheduler fresh；禁止resume J2/J3 profile或任何旧G3 checkpoint；
- 数据、10 gradient+2 task-held、两fit/一held K1 video、panel A/B、logical16/physical2、100 effective updates及step70/110
  checkpoints全部沿用J2。唯一改动是每task每step在两条correct views外增加一条counterfactual view：same-role active三task循环配对，
  step0起wrong Program与wrong bank交替，fit view 0/1同步交替。hinge margin为该task formal positive-control两fit-video panel-A mean
  benefit的`.10`倍，权重按六task等质量；margin满足即不反传negative；
- source、Native Stage0、current-bank operator、carrier12、scale和Action Meta冻结；训练只读gradient-task panel A actions/flow及已封存
  positive-control fit-panel标量，不读teacher factors、panel B、held video、task-held、validation/test、reward或outcome。每个condition仍只
  生成一个完整38-target rank12+4 rank16 adapter，shuffled/reversed不使用；
- formal输出根为
  `runs/outputs/pi05_ecp_j3_counterfactual_program_primal_12task_s110_9af7c19_gpu01p012345_r6_20260829/`，预计新增`<1GiB`；复用
  `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`的23GiB frozen cache，其content authority未变且profile全部命中，不重建
  language/Stage0/X-Y/covariance；
- 2026-08-29 launch前live状态：gpu01逻辑0--5对应已复核UUID/serial，显存占用约`0.01--0.92GiB`、UTL`0--2%`，0--2的他人
  小进程峰值余量充足；旧prohibited设备未枚举，当前逻辑0不继承旧限制。gpu02 0--3满载、5--6忙、7空闲，因此不跨节点拼卡；
  `/data1` user quota为`707772604/1073741824KiB`，`/dev/shm`尚余39GiB。launch使用gpu01 0--5、GPU-local NUMA、deferred NCCL、
  `NCCL_P2P_DISABLE=1`；
- profile实测`21.696s/global step`、peak reserved`20.395GiB`，故formal继续锁定world6并预期约40分钟训练主体；目标/硬上限仍为
  `<=30/45s`及`<35GiB/GPU`。完成后对step70/110执行同一J3 Gate；correct recovery不提升时，negative变坏本身不构成通过。

## 2026-08-29 第四次专家复核已采纳，J2 joint Program--primal functional qualification启动

第四位专家锁定远程`main@910fb204e8e3a5374ec988aa5e1da5bc042754aa`及`9b52e59..910fb20`的完整历史，复核了P0/P1、P2执行面、
95-task authority和G2-B pointwise/v3--v5全部formal与诊断。1075行原始回复已逐字保存为
`docs/expert_review_20260829_joint_program_primal.md`。owner接受其主裁决并许可继续推进。

最新边界是：V5仍是合法的protocol non-pass，但15次updates中前9次仍在warmup，且固定block-equal、单位化、等权Program Gram并非
最终functional坐标；它只淘汰该目标，不能证明Program schema或Native Stage0结构性失败。P0/P1已经证明shared primal经每条current
bank的dual/replay在六tasks、四family和held videos上保留`.94--.995`量级容量。因此当前最早未解决接口是Natural Program与shared
`ProgramNativePrimalScorer`之间的联合可识别性和functional credit。

J2实现已集成canonical `main`。当前复用P2 compact condition cache、current-bank operator和functional
gradient，联合训练Program与primal scorer；source、Native Stage0、operator、carrier、scale与Action Meta冻结。先完成同loss的task-local
free-primal正控，再执行固定10 gradient tasks+2 true task-held的12-task J2 Gate。冻结evidence/X-Y/covariance/action panels只缓存一次，
不缓存Program或LoRA；六卡global update目标`<=30s`、硬上限`45s`、每卡peak reserved`<35GiB`。只做一个最小真实
forward/backward/materialization和必要定向合同检查，随后尽快进入有信息量的正控/formal。

J2实现面现已接通：Natural Program拆为冻结evidence capture与可微Program compile，condition cache升到v4且不缓存Program；
`joint_program_primal/`唯一子包分别由`runtime.py`持有authority/model/cache/data、`train_step.py`持有role-balanced functional update、
`gate.py`持有task-local正控与Gate、`training.py`持有CLI循环，唯一脚本只做phase dispatch。旧P2 runner继续作为历史artifact解释器，
不再是active optimizer路径；J2作出阶段结论后该子包将被吸收进后续canonical Writer或退役，不保留第二部署实现。

gpu01物理0上的task1真实positive-control profile完成。38-target rank12+4唯一rank16、两fit视频共享free primal、第三held视频零梯度、
panel A/B和carrier replay均接通；Action Meta argument/module/parameter均为0，source/Stage0/scale/temporal decoder trainable均为0。
physical microbatch8时单步`9.459s`但peak reserved`42.154GiB`，超过合同；只把keyed logical16的物理microbatch改为4后，单步
`9.497s`、peak allocated/reserved`28.370/28.951GiB`，无吞吐损失且低于`35GiB`。三条bank cold build合计`27.877s`，hot load
`.655s`。一步后两个fit与held视频在独立panel B均优于carrier，held benefit`.00601`、factor recovery`.92696`；这只是机制smoke，
不是10-task正控Gate。封存carrier loss与当前replay绝对差`8.52e-5`，在rows/seed/checkpoint/carrier/microbatch identity均锁定后按项目
BF16/TF32数值政策使用`1e-4` replay容差并把实际误差留在报告。下一步是定向回归、clean pushed detached authority及六卡并行
10-task formal positive control；尚不启动12-task joint formal。

## 2026-08-29 J2 10-task functional positive control通过，physical microbatch2锁定

clean pushed detached `main@f3677a5`在gpu01以最多六个独立workers完成全部10个gradient tasks、每task 100 updates及完整16-visit
panel A/B与第三held video评价。两条fit K1 videos共享一个task-local free primal，第三video、panel B和held teacher均零梯度；所有task
Action Meta 0、source/Stage0/Program/shared scorer/scale冻结，实际物化唯一38-target rank12+4 rank16，task-local checkpoint明确不是
deployment candidate。

科学正控通过：held/fit functional benefit retention median/min为`1.0144/.8896`，held native-factor recovery median为`.8078`；
q/v/action-in/action-out family medians为`.7973/.7722/.8436/.8481`。10/10 task的held panel-B均显著优于carrier，单侧paired
t-test最大p值`.00275`、符号检验最大p值`.03841`；held平均benefit median/min为`.00870/.00302`，所有30个video条件的panel-B
mean benefit均为正。该结果证明当前action/flow panel、frozen scale与native free-primal之间存在稳定、跨video可达的功能下降方向，
不证明shared Program mapping已经成立。

formal训练每task`686.97--1274.58s`，完整评价`130.44--134.83s`，最大eval/train ratio`.1957`，全部低于`.5`。physical
microbatch4时9/10 task低于35GiB，但最长task93为`37.07GiB`，故原系统配置单点non-pass。只把keyed logical16的physical microbatch
改为2后，同task93、同banks/seed一步profile的loss绝对/相对差为`3.99e-5/.060%`、gradient norm差`2.08%`，step`13.32s`、
peak allocated/reserved`26.89/32.41GiB`，低位规约与吞吐可接受且内存过线；active config因此唯一锁定physical2，不重复科学formal。
aggregate与raw evidence位于`runs/outputs/pi05_ecp_j2_functional_positive_control_10task_c4704cb_gpu01p012345_20260829/`，系统profile
位于`runs/analysis/pi05_ecp_j2_pc_task93_mb2_memory_profile_f3677a5_gpu01p3_20260829/`。下一步只做六卡joint一步profile，验证Program与
primal scorer梯度、role/task权重、global step墙钟及峰值；通过后才启动12-task joint formal。

### J2 10-task functional positive-control formal launch contract

- implementation authority为clean pushed `main@c4704cb04d521154faf384abd1bf5b9af7ead9c2`；formal从包含本合同、但不再修改
  `src/ scripts/ configs/ tests/`的clean pushed detached descendant执行。每task fresh task-local code/AdamW，不resume profile或其它task，
  profile checkpoint不进入formal；失败输出保留并标invalid，修复后使用新run identity；
- 固定gradient meta`[1,8,9,32,52]`与gradient target`[72,73,75,93,94]`。每task 100 updates，每update用同一panel-A visit的
  logical16 actions及两条fit K1 videos等权，合计32,000个task-view-action rows；第三held video、16个panel-B visits及held teacher只在
  训练结束后零梯度读取。source、Stage0、Natural Program/shared scorer、operator、carrier、scale与Action Meta全冻结，只优化该task
  跨两fit videos共享的free primal；输出不是deployment candidate；
- model/data authority沿用source checkpoint`runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`、
  tokenizer`models/tokenizers/openpi/paligemma_tokenizer.model`和target data
  `data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`。输出根为
  `runs/outputs/pi05_ecp_j2_functional_positive_control_10task_c4704cb_gpu01p012345_20260829/`，每task独立子目录；node-local cache为
  `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`。预计formal输出`<0.1GiB`，10task cache约`24GiB`，不复制模型或dataset；
- 运行方式为gpu01上独立single-process workers，不使用NCCL：首波GPU`0--5`并行task`[1,8,9,72,73,75]`，第二波用可用GPU并行
  `[32,52,93,94]`。物理0--2绑定NUMA0、3--6绑定NUMA1；每worker logical16/physical microbatch4，实测peak reserved
  `28.95GiB`。实际launch前重新检查gpu01/gpu02、进程/显存/UTL、NUMA、`/data1` quota、gpu01 `/dev/shm`及output不存在；有几张
  合适卡就并行几张，不等待凑6且不改变task权重；
- 单worker命令模板：`env CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  numactl --cpunodebind=<NUMA> --membind=<NUMA> /data1/user/ymdai/projects/EMBER/.venv/bin/python -u
  scripts/train_ecp_joint_program_primal.py --config configs/pi05_ecp_joint_program_primal_j2_v1.json --base-config
  /data1/user/ymdai/projects/EMBER/configs/pi05_ecp_shared_compiler_g3_v5.json --phase positive-control --mode formal --task <TASK> --asset-root
  /data1/user/ymdai/projects/EMBER --source-run <SOURCE_RUN> --checkpoint <SOURCE_CHECKPOINT> --tokenizer-path <TOKENIZER>
  --data-root <TARGET_DATA> --output-dir <OUTPUT_ROOT>/task_<TASK> --condition-cache-root <CACHE_ROOT> --log-every 1`；所有相对
  script/config路径从同一detached formal worktree解析，asset authority仍指canonical main根；
- 每task必须保留run contract、100-step curve、唯一task-local code、完整panel A/B与held-video report及completion。Gate要求10task
  held factor recovery median`>=.80`、q/v/action-in/action-out各task-family汇总`>=.70`、每task held panel-B显著优于carrier，且
  Action Meta 0、held/panel-B backward 0、唯一rank16和数值/内存合同全部成立。该正控通过只证明functional objective能驱动
  native free-primal，不证明shared Program mapping；形成aggregate Gate结论后才允许做六卡joint one-step profile和12-task formal。

## 2026-08-29 J2六卡joint速度资格通过，12-task functional qualification启动

clean pushed `main@3fd5c6f`在gpu01物理`0--5`完成world6真实joint一步profile。固定task group为
`[1,8,9,72,73,75]`（3 meta + 3 target），12/12 condition全部命中冻结cache；global step为`11.7321s`，六rank分别为
`11.7235--11.7321s`，负载均衡。per-rank peak reserved为`18.25--20.29GiB`，远低于`35GiB`；Program language/scene/process/context、
primal input/output/event score的七组gradient probe全部finite且非零。native teacher tensor reads为0，source、Stage0、scale、decoder、
Action Meta以及task/frame free parameter均为0，trainable参数只有共享Program与primal scorer的`8,393,221`个；全部条件最终仍物化唯一
38-target rank12+4 rank16 adapter。profile evidence位于
`runs/analysis/pi05_ecp_j2_joint_world6_mb2_profile_3fd5c6f_gpu01p012345_20260829/`。随后`main@1d775a4`只把formal stdout收紧为
step/rank摘要，完整condition与solve diagnostics继续保留在JSONL，不改变科学图或optimizer；定向16项回归与compile通过。

### J2 12-task joint formal launch contract

- scientific implementation authority为clean pushed `main@1d775a445486d84a9743e363360ca7a2863cac46`；formal从只新增本合同文档、
  不再修改`src/ scripts/ configs/ tests/`的clean pushed detached descendant执行。Natural Program只加载`c1493a1/macro20`的model
  tensors，primal scorer、optimizer与scheduler fresh；不加载positive-control、P2或G3 v3--v5 checkpoint；
- gradient tasks固定为meta`[1,8,9,32,52]`和target`[72,73,75,93,94]`，true task-held固定为`[2,74]`。每个global step取
  3 meta + 3 target，按确定性cycle使每个gradient task在每5步出现3次；每个task使用两条fit K1 video与同一panel-A visit的
  keyed logical16 actions，physical microbatch2。第三same-task video、panel B、task-held、validation/test全部零梯度；
- 总计110 actual optimizer steps，其中前10步warmup、后100步为effective joint training；保存actual step70与110两个single
  checkpoints。source policy、Native Stage0、current-bank operator、rank12 carrier、scale、temporal decoder与Action Meta全冻结；
  只训练共享Natural Program与Program-to-primal scorer，不使用teacher LoRA target、shuffled或reversed；
- model/data authority沿用source run
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722`、checkpoint`checkpoints/step_00001000`、tokenizer
  `models/tokenizers/openpi/paligemma_tokenizer.model`与target data`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`。
  复用node-local冻结cache`/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`，其中不含Program或LoRA；formal输出根为
  `runs/outputs/pi05_ecp_j2_joint_program_primal_12task_s110_1d775a4_gpu01p012345_r6_20260829/`，预计新增小于`1GiB`；
- launch固定gpu01 world6物理`0--5`、`NCCL_P2P_DISABLE=1`、deferred NCCL与GPU-local NUMA；实际launch前重新检查gpu01/gpu02、
  process/显存/UTL、`/data1` quota、`/dev/shm`、output不存在和detached worktree等于origin/main。已测global step`11.73s`、
  per-rank peak reserved最大`20.29GiB`，满足`<=30s`目标、`45s`硬上限和`<35GiB`内存门；exact resume锁定world6/topology；
- step70/110使用同一固定零梯度evaluator报告functional recovery、family、same-task held video、true task-held、language/endpoints、
  wrong Program、wrong bank、interaction、same-task retention、event/K1与信息墙。primary为相对同一carrier与task-local free-primal正控的
  generated-LoRA functional recovery；训练loss、factor/update cosine及checkpoint union不能代替Gate。若non-pass，按最早失效接口定位，
  不用seed/LR/width/slot/rank小扫修饰结果。

## 2026-08-29 G2 global-calibrated behavior-kernel v5 formal non-pass，暂停新增版本

clean pushed `main@7f4df1b`的detached frozen worktree已在gpu01物理`0--4`完成五卡macro5/15 updates、唯一checkpoint和
全部预注册internal Gate。run自然结束且无工程错误；source policy与Native Stage0 trainable均为0，Action Meta argument为null、
module/parameter均为0。旧动态职责继续通过并略有提高：full相对endpoints改善`20.8602%`、active events中位数3、one-event 0，
same-task/probe/K1/K4/tau全部通过。但新增behavior Gate明确失败：train60 topology A/B仅`.2160/.2208`，internal meta
`.2022/.2169`，internal target`.7508/.7670`仍是此前已存在的偶然高值；fit60-only exact panel-B/consensus role-equal仅
`.1054/.1289`，wrong-Program margin为`-.0466`。official held20保持未读，不能用旧动态通过、target小组高值或internal loss冒充
policy-behavior sufficiency。

冻结checkpoint的block geometry确认v5确实消除了“完全不展开”的表象，但展开方向仍错。相对v4，full Program跨task cosine的
均值/标准差从约`.965/.020`移到`.926/.046`，process块从`.898/.086`移到`.750/.220`；它们朝固定lift目标的尺度移动，且两套
video的cross-view仍分别约`.970/.994`。然而full/process对独立teacher consensus的相关反而从约`.150/.135`变为
`.142/.131`，train/meta及exact读出也没有改善。训练末期Program spread仍远低于固定teacher尺度，behavior alignment在macro4--5
约`12.6190/12.6196`平台化；梯度有限且非零，说明不是loss未接通或数值NaN，而是当前目标让Program学会了“把任务拉开”，没有学会
“按真实policy效果正确拉开”。

v3、v4、v5在同一macro5口径的train A/B依次为`.2315/.2358`、`.2360/.2362`、`.2160/.2208`；exact panel-B依次
`.1207/.1129/.1054`，wrong margin均为负。根据预注册停止条件与owner本轮要求，不续训、不读official held20、不新增v6，也不恢复
G3 P2。当前结论只淘汰pointwise decoder及v3--v5这组三种直接Program behavior-credit实现，不证明Stage0、Program schema或整个
Native-Factor路线根本不可能；最早未解决接口仍是G2部署Program的跨task policy-behavior可识别性。下一步暂停在专家复核前。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_global_behavior_kernel_fold0_m5_2d859f0_gpu01p01234_r5_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_block_geometry_v5_2d859f0_gpu01p01234_r5_20260829.json`；
- 对照：`runs/analysis/pi05_ecp_g2_behavior_block_geometry_init_v3_v4_20260829.json`。

## 2026-08-29 G2 behavior-kernel v4 formal non-pass，global-calibrated v5已接通

clean detached `main@4eb8b8c`的五卡v4完成macro5/15 updates及全部internal Gate。旧动态职责仍通过：full相对
endpoints改善`14.6553%`、active events中位数4、one-event 0、same-task/probe/K1/K4全通。但全量train60
topology A/B仅`.2360/.2362`，internal meta为`.2064/.2257`，internal target的`.7512/.7634`仍是旧Program已有的
偶然高值；fit60-only reader的panel-B/consensus exact role-equal仅`.1129/.1177`，wrong margin为负。因此加入joint edges
虽然把监督图连通，却没有产生全局behavior geometry；official held20仍未读取。

固定checkpoint的block geometry进一步定位了比Stage0更早的目标退化：v4 full Program的跨task off-diagonal cosine均值约
`.965`、标准差约`.020`，teacher behavior kernel则为`.145/.316`。v3/v4使用的逐batch centered、Frobenius-normalized
kernel loss会消除每批的平移与尺度，因而允许不同batch各自使用不同affine gauge，也不会惩罚接近公共向量的低方差Program；连通
pair graph本身不能修复这个自由度。这解释了local correlation可达`.7`而全量train60与exact readout几乎不升。

当前唯一v5修正不增加模型参数、reader或deployment路径。它把teacher cosine确定性lift为
`K_target=(1+K_behavior)/2`，直接对齐raw Program off-diagonal cosine；每个owner的残差及跨view差异只除以由完整
train60、meta45或target15预先计算的固定teacher dispersion，不再按当前mini-batch重心或范数重新定标。这样公共轴仍允许
lifted kernel精确成为Gram，但task差异的绝对均值与幅度不能再由batch-local gauge抹掉。数据、5+5 task权重、两组video、
Program schema、旧动态Gate、internal15/official20边界、纯Native Stage0和Action Meta关闭均不变；v4 config已由v5替换，
保持一个canonical执行面。

三卡真实一步profile已验证该目标不是弱梯度或数值死路：behavior alignment loss `12.2518`，behavior-kernel与Program梯度
分别`1.7323/2.7450`，总梯度`3.4997`后沿用既有clip；step `18.35s`、peak allocated `9.98GB`。初始Program/teacher
off-diagonal std为`.0141/.1478`，符合诊断预期；source与Stage0 trainable 0，Action Meta module/parameter 0。定向回归
`19 passed`、全仓`204 passed`、diff check与config load通过；该profile不是Gate结果，下一步只能从clean pushed detached
authority运行同一macro5 formal，若仍无数量级改善则停止继续叠加G2版本并整理专家复核证据。

### G2 global-calibrated behavior-kernel v5 macro5 formal launch contract

- scientific code authority为clean pushed `main@2d859f0`；formal从只新增本合同文档、不改`src/ scripts/ configs/ tests/`的
  clean pushed descendant建立detached frozen worktree。fresh optimizer，从`c1493a1/macro20`只加载Program model tensors；不resume
  v3/v4，不复用dirty profile checkpoint；
- 唯一因果变量是v5固定global calibration：teacher使用`(1+K_behavior)/2`，raw Program off-diagonal Gram及cross-view差异
  按完整joint/meta/target fit scope的固定per-owner teacher dispersion缩放。fit60用于梯度与fixed evaluator，internal15只作Gate，
  official held20保持0 reads；数据、每step 5 meta+5 target、两组disjoint same-K views、`.5/.25/.25`scope权重、Program schema、
  dynamic loss/Gate和optimizer全部与v4一致；
- formal规模为5 macros × 3 optimizer steps，world5每rank一对role task。只训练Natural Program；source与Native Stage0冻结，
  Action Meta argument为null、module/parameter必须为0；不使用shuffled/reversed；
- command：`env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=5 scripts/train_ecp_natural_program.py --config
  configs/pi05_ecp_natural_program_g2_behavior_kernel_v5.json --mode formal --asset-root /data1/user/ymdai/projects/EMBER --source-run
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path
  /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --data-root
  /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --label-root
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_labels_g2_v2_cpu_20260825 --output-dir
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_g2_global_behavior_kernel_fold0_m5_2d859f0_gpu01p01234_r5_20260829
  --stop-after-macro 5 --log-every 1`；
- `2026-08-29 11:47 CST` live preflight中gpu01物理`0--6`均无compute process、util 0且free memory约`45.9--46.1GiB`；
  选择`0--4`满足world5且不占第六卡。gpu02仅物理7空闲，其余均有他人任务，故不跨节点拼卡。`/data1` user quota为
  `674.7GiB/1TiB`，shared filesystem可用`84TiB`；同规模v4 formal为`16MiB`，本run预计峰值新增小于`0.1GiB`，output事前不存在；
- macro5后一次性执行旧dynamic与完整internal behavior Gate。通过仍要求train topology`>=.50`、internal role-equal及meta/target
  各`>=.25`，再满足exact/family/wrong/language/view阈值；训练mini-batch loss/std不代替Gate。若v5相对v3/v4仍没有数量级改善，
  不延长、不读official held20、不新增v6，固化最早接口与完整证据后暂停并准备专家复核。

## 2026-08-29 G2 behavior-kernel v3 formal non-pass，joint-role v4已接通

clean detached `main@60fb18b`的五卡v3从`c1493a1/macro20` model-only初始化、fresh optimizer运行到macro5/15
updates，数值、五卡利用、checkpoint、纯Native Stage0与Action Meta 0全部正常。最后一个local 5+5 batch的panel-A/B
correlation达`.7036/.7037`，旧动态Gate也仍全通：full-vs-endpoints改善`13.945%`、active events中位数4、one-event 0、
same-task/probe/K1/K4全通。但全量train60 topology A/B只有`.2315/.2358`，低于`.50` Gate；internal-held15
role-equal为`.4997/.5131`看似高，其中四个target held的`.7842/.7930`是旧Program已有的偶然高值，11个meta held仅
`.2152/.2332`，未过两role各`.25`。fit60-only reader的panel-B/consensus exact role-equal只`.1207/.1253`，wrong margin为负；
因此该checkpoint明确non-pass，official held20没有被读取。

最早失效接口不是Stage0 grounding，而是v3的局部关系目标没有定义全局坐标。当前15个optimizer batches在meta45上只覆盖
126/990对（`12.7%`），且明确分成5个互不连通的components；所以每个5-task小组可以单独获得高correlation，却没有
任何梯度规定不同组之间的相对几何。authority中meta-target跨role的panel-A与consensus关系是稳定的，平均相关`.8629`；
在现有每批5 meta+5 target上加入跨role共同topology后，15个batches的监督图会变为覆盖483/1770对（`27.3%`）、
minimum degree 9且唯一一个60-task component。

当前单一v4修正因此保留原5+5、15 updates、两组video、六个Program blocks与全部Gate，只把behavior loss改为
`0.5 * joint + 0.25 * meta + 0.25 * target`的centered off-diagonal topology。它不增加reader、参数、forward、task route或
deployment路径。实现已经过`204 passed`、diff与architecture检查后clean pushed到`main@37885a6`，无hard violation。clean
detached三卡真实一步的joint/meta/target A关联为`.1615/.8314/.4287`，直接kernel/process/owner梯度为
`1.391/.842/.0434`，step `18.31s`、peak allocated `9.98GB`，source/Stage0 trainable 0、Action Meta 0；与dirty profile逐项一致。
该结果只证明新credit graph接通，不是Gate。

### G2 joint-role behavior-kernel v4 macro5 formal launch contract

- scientific code authority为clean pushed `main@37885a6`；formal从只新增本合同文档、不改动`src/ scripts/ configs/ tests/`的
  clean pushed descendant建立detached frozen worktree。fresh optimizer，从同一`c1493a1/macro20` Program model-only初始化，不resume v3、
  不复用profile checkpoint；
- data、authority和Gate与v3完全相同：fit60 behavior tensors用于梯度与fixed evaluator，internal-held15只作Gate，
  official held20不读、不训练、不选模、不修正。唯一因果变量是每批在原role kernels外加入等质量joint kernel，权重
  固定`.5/.25/.25`；
- formal规模仍为5 macros × 3 optimizer steps，每step 5 meta+5 target、每rank一对role task、两组disjoint same-K views。仅训练
  Natural Program；source、Native Stage0冻结，Action Meta必须为0。macro5存一个checkpoint后执行旧dynamic Gate和完整internal
  behavior Gate；
- command：`env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=5 scripts/train_ecp_natural_program.py --config
  configs/pi05_ecp_natural_program_g2_behavior_kernel_v4.json --mode formal --asset-root /data1/user/ymdai/projects/EMBER --source-run
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path
  /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --data-root
  /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --label-root
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_labels_g2_v2_cpu_20260825 --output-dir
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_g2_joint_behavior_kernel_fold0_m5_37885a6_gpu01p01234_r5_20260829
  --stop-after-macro 5 --log-every 1`；
- 预定使gpu01物理`0--4`、world5和`NCCL_P2P_DISABLE=1`，不使用dummy第六卡。launch前再次live检查gpu01/gpu02的显存、
  utilization和process，并复核`/data1` quota；output预计小于`0.1GiB`且目录必须事前不存在；
- Gate与失败解释不变：train topology `>=.50`、internal held role-equal `>=.25`、internal meta/target各`>=.25`，再评
  exact/family/wrong/language/view和旧dynamic合同。joint mini-batch metric不代替全train60 Gate；train仍不升时先分析Program表示可达性，
  train明显升而internal失败时才重开窄Stage0 grounding tail；不做seed/LR/width/rank小扫。

## 2026-08-29 G2-B pointwise formal non-pass，decoder-free behavior kernel已接通

clean detached `main@5cbe76e`的首轮G2-B从旧`c1493a1/macro20`只加载共同model tensors，以fresh decoder/optimizer运行到
macro60。训练behavior loss由`1.2723`降至`.7080`，旧动态Gate持续成立且full-vs-endpoints由macro10的`31.85%`升至macro60的
`39.40%`；但独立panel-B exact rank4在macro10/20/40/60仅`.1837/.2622/.2938/.2828`，最终consensus`.3027`、meta/target held
`.3803/.1853`。因此该轮不是训练没接通，而是明确scientific non-pass。

冻结macro60后的新reader、linear/kernel CV及Program geometry给出一致归因：reader可把fit code读到近1，但task-holdout仍约`.262`；
full Program的fit behavior pairwise correlation只从旧`.1610`到`.1694`，official held仍约0。训练reader吸收了pointwise coordinate
loss，部署Program没有形成可迁移的policy-behavior邻接关系。该结果排除继续增加decoder容量、续训、LR或普通参数微调；不排除固定Program
schema、Stage0、native bank或G3 primal-to-dual operator。

为避免反复消费official held20，现已固定原fit75内的role-stratified train60/internal-held15。train60 rank16 basis在internal15对
panel-B/consensus为`.6184/.7158`，q/v/action-in/action-out为`.6556/.7373/.4550/.6676`；official held20本轮不进入训练、
checkpoint选择或架构修正。当前`codex/g2-behavior-kernel`已删除pointwise decoder和旧v2 config，新增一个无参数的deployment-Program
behavior kernel：完整`P_lang/P_scene/P_process/rho/tau/sigma`按固定block-equal feature保留owner/event顺序，两组disjoint same-K
video views在每个role内直接对齐panel-A与consensus factor-cosine topology，并保持cross-view kernel一致。梯度直接归Program，不经过
training reader；fixed kernel-ridge只在internal Gate上由train60拟合，不是模型或deployment路径。

五卡formal合同为每step 5 meta+5 target、每rank一对role任务；Stage0 v3/source冻结、Action Meta 0、uniform K、原动态loss/Gate和
Program schema不变。三卡真实一步已验证global autograd gather、全部Program梯度路径、纯Native loader与100% UTL，约`19.73s`、peak
allocated`9.98GB`；最终factor-cosine authority schema随后升级为v3，故该dirty profile只作执行证据。实现已以clean
pushed `main@c8fee96`集成，diff/architecture检查无hard violation，全仓回归`204 passed`。从该detached commit重封的v3
authority只含train60/internal-held15的behavior tensors，official held20仅保留ID provenance且tensor overlap为0；其
factor-cosine对角最大误差为`1.19e-7`。clean三卡profile的单步为`19.66s`，alignment/cross-view loss为
`.7062/.00958`，Program panel-A/panel-B correlation为`.6300/.6311`，直接behavior-kernel/process/temporal梯度分别为
`2.239/1.008/.0434`，Stage0/source trainable 0、Action Meta module/parameter 0。这仍只是执行profile，不是Gate结果。

### G2 behavior-kernel macro5 formal launch contract

- scientific code authority为clean pushed `main@c8fee96`；formal从只新增本合同文档、不改动`src/ scripts/ configs/ tests/`的
  clean pushed descendant建立detached frozen worktree，运行时再核对commit与code diff。fresh optimizer，不resume、不复用profile checkpoint；
- behavior authority固定为
  `/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_g2_behavior_authority_fold0_v3_20260829`；其中fit60用于梯度与fixed
  evaluator fit，internal-held15只用于Gate，official held20不读behavior tensors、不训练、不选模也不修正架构；
- formal规模为5 macros × 3 optimizer steps；每step为5 meta+5 target、每rank一对role task、两组disjoint same-K views。仅训练
  Natural Program；source、Native Stage0冻结，Action Meta必须为0。macro5存一个checkpoint后执行旧dynamic Gate和internal behavior Gate；
- command：`env CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=5 scripts/train_ecp_natural_program.py --config
  configs/pi05_ecp_natural_program_g2_behavior_kernel_v3.json --mode formal --asset-root /data1/user/ymdai/projects/EMBER --source-run
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path
  /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --data-root
  /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --label-root
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_labels_g2_v2_cpu_20260825 --output-dir
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_g2_behavior_kernel_fold0_m5_c8fee96_gpu01p01234_r5_20260829
  --stop-after-macro 5 --log-every 1`；
- 预定使gpu01物理`0--4`、world5和`NCCL_P2P_DISABLE=1`，不使用dummy第六卡。launch前必须重新live检查gpu01/gpu02的显存、
  utilization与process，并重查`/data1` quota；output预计小于`0.1GiB`且目录必须事前不存在；
- Gate先要求train topology `>=.50`、internal held role-equal `>=.25`、internal meta/target各`>=.25`，再用fit60-only
  kernel-ridge产生rank16 coordinates评估exact behavior/family/wrong/language/view和旧dynamic合同。train不升时修Program credit；train升而
  internal失败时才重开窄Stage0 grounding tail；不做seed/LR/width/rank小扫。只有internal全Gate通过后才允许一次读official held20。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_behavior_fold0_m10_5cbe76e_gpu01p012345_r6_20260829/`；
- `runs/analysis/pi05_ecp_g2b_frozen_program_probe75_20_5cbe76e_gpu02p7_20260829/`；
- `runs/analysis/pi05_ecp_g2b_program_geometry_old_vs_macro60_5cbe76e_20260829.json`；
- `runs/analysis/pi05_ecp_g2b_kernel_readout_old_vs_macro60_5cbe76e_20260829.json`；
- `runs/analysis/pi05_ecp_g2b_internal_task_holdout_basis60_15_5cbe76e_gpu02p7_20260829.json`。

## 2026-08-29 G2 behavior sufficiency失败，最小行为对齐修正已通过真实smoke

P2后续分析没有继续改covariance、polar或SVD。当前先补齐了55个任务的两组disjoint 256-row cross-episode flow-gradient panels，与已有
40-task authority合并为95 tasks；同时从冻结`c1493a1/macro20`为95 tasks提取两套disjoint K4和一套K1 Program，共285份。所有authority
均只覆盖non-held LIBERO-90与train24，validation/test、Action Meta及held optimizer gradient为0。

75 fit task按meta56/target19各占50%建立的behavior manifold在20个未见task上有明确上限：两panel repeatability`.6666`，rank16
oracle对panel-B/consensus为`.7160/.8006`，四family为`.7571/.8009/.5482/.7779`，universal仅`.1908`；rank32相对rank16只增加
约`.0526`。相反，固定1000-step Program probe能把full/process/dynamic在fit75读到约`.97--.98`，held exact rank4却仅
`.2695/.2470/.2368`，与language-only`.2687`同量级；full的meta-held/target-held为`.3684/.1707`。因此旧G2动态Gate继续成立，
但Program没有把未见task组织到可迁移的policy-behavior方向，G3 shared mapping non-pass的最早接口已经上移到G2 sufficiency。

当前`codex/g3-behavior-primal`只在同一个Natural Program执行面增加训练期behavior decoder与Gate：decoder只读
`P_process/rho/tau/sigma`，不读`P_lang/P_scene`、task ID或任何deployment外输入；训练target是75 fit task的role-balanced rank16
behavior code，20 held只作fixed exact-rank4 qualification。旧`c1493a1/macro20`只初始化共同model tensors，新decoder fresh，optimizer/
scheduler/data cursor全部fresh；原action/progress/event/robustness loss与旧held动态Gate保持。formal首站固定macro10，使用3 meta+3 target
role-balanced optimizer step、六卡并行；新Gate要求exact overall`.50`、consensus`.60`、q/v`.50/.50`、action-in/out`.35/.50`、
wrong margin`.10`、两role各`.45`，并保留K1/K4与cross-video qualification。

真实task74 K4 profile已经完成：behavior loss`1.3104`，decoder/process-fusion/旧temporal-owner梯度分别`.6286/4.4444/.0805`，
source与Native Stage0 trainable 0、Action Meta module/parameter 0；一步`13.30s`，peak allocated约`9.28GiB`。新exact Gate用旧frozen
Program和同构decoder做反向smoke得到`.2121`、oracle`.7160`并正确non-pass，12项checks中仅wrong-margin和cross-view一致性通过。
架构检查没有新增hard violation，全仓回归`201 passed`；下一步是clean commit/push、从detached authority正式封存93KiB behavior-code asset并
启动六卡macro10，不再消费旧P2 cache或续训旧shared compiler。

关键artifacts：

- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_t0_missing55_5781694_gpu01p012345_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_program90_5781694_gpu01p012345_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_probe75_20_5781694_gpu01p6_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_manifold95_rolebalanced_5781694_gpu01p6_20260829.json`。

## 2026-08-29 P2执行面接通，等待clean authority六卡profile

P2保持唯一active v5数学不变，只训练共享full-Program-to-primal scorer。为避免329个fit conditions在每次optimizer update都重新运行
冻结的Pass A和38-target policy capture，当前实现增加run-local、node-local frozen-condition cache：每个condition只保存冻结Program、
raw native X/Y、final Y和B0截断谱operator；abs/adj/init/goal仍在每次B1 replay中按单视频边界在线构造。cache不进入Writer
checkpoint、不是deployment输入，也不保存展开四倍的Y bank。正式训练只持久缓存329个mapping-fit conditions，40个held-video和82个
task-holdout conditions在451评估中即时构造后释放，避免`/dev/shm`容量被held evidence占用。

原先统一`.1*s_ref`的冻结scale在“方向完全正确”的解析自相似上，fit task median仅`.767177`、p10`.751008`，离P2 held
median `.75` Gate过近，会把方向学习与scale ceiling混为一谈。首版P2因此使用一个由40个mapping-fit tasks、排除预注册held video的
complete member consensus按task等权中位数导出的共享`[38,4]` rank template，再乘每target `s_ref`；它对所有task/video相同，scale
head仍完全冻结且不含lookup。该模板的fit解析ceiling为median `.997017`、p10`.974083`、minimum `.964334`；held仅作post-hoc
诊断，video/task-holdout medians为`.996952/.997577`。这只移除P2的scale混杂，不证明shared Program mapping；scale学习仍推迟到F4。

gpu01物理0上的真实task1/video16 smoke已证明：cache miss `11.320s`，命中后的完整38-target forward/backward `2.625s`，文件
`783,773,240` bytes，peak allocated/reserved约`14.19/19.73GiB`；217个primal-scorer参数梯度全部有限非零，Action Meta module为0、
scale trainable parameter为0。相同raw X/Y与B0 operator在chunked streaming和compact replay间最大绝对误差
`2.384185791015625e-07`，cache write/read factor误差为0；每次算子调用前后TF32全局状态均恢复。独立重新capture的深层BF16 native
activation可有一个量化步差异，随机近零primal会放大为最低`.867` update cosine；同bank交叉臂为约1.0，故这不是cache或chunk错误，
且P1已在不同真实videos的有意义primal上证明`.9545` held recovery。

全仓`198 passed`。clean pushed detached `0a37170`的world6 cold/hot profile均已完成：固定同一3 meta+3 target update的12条K1
conditions在cold时全部并行构建，单condition build为`4.13--9.95s`，step为`24.84s`、cache合计`6.522GB`；hot时12/12命中，
单文件load最多`.219s`，完整forward/backward/all-reduce/optimizer step为`6.177s`。两次mean recovery、gradient norm和rank assignment
完全相同，分别为`.0012211`、`.206293`及每rank一个task；peak allocated/reserved约`15.32/19.73GiB`。六份冻结policy的一次性进程
启动约139s，但不随step重复。该吞吐足以在六卡上约数分钟构建每macro的新conditions并以约31s执行五个hot updates，不再有
hours-per-condition问题。

### P2 macro5 formal launch contract

- scientific code authority为clean pushed `main@0a37170`；formal必须从包含本合同、可由`origin/main`到达的clean detached commit执行。
  本合同后的提交只允许文档更新，不改变P2 config/code/test；fresh optimizer，不resume、不复用profile cache或checkpoint；
- data/optimization固定为329 mapping-fit conditions、40 held-video和82 task-holdout；训练只用fit，5 macros × 5 optimizer steps，
  每step固定3 meta+3 target、每task两条K1 videos，world6只做cost-balanced分片。只训练shared full-Program-to-primal scorer；G2、source、
  scale head、carrier、teacher与Action Meta冻结，scale使用唯一fit-only shared `[38,4]` template；
- command：`env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_ecp_shared_compiler.py --config
  configs/pi05_ecp_shared_compiler_g3_v5.json --phase f3 --mode formal --asset-root /data1/user/ymdai/projects/EMBER --source-run
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path
  /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --data-root
  /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir
  /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_shared_compiler_g3_p2_fold0_m5_gpu01p012345_r6_20260829
  --condition-cache-root /dev/shm/ember_ecp_g3_p2_formal_20260829 --stop-after-macro 5 --log-every 1`；
- runtime：launch前重新live检查gpu01/gpu02；首选gpu01物理0--5、自动GPU-local NUMA、deferred NCCL且`NCCL_P2P_DISABLE=1`。
  profile实测六卡有效并行，非dummy占卡。`/data1` quota为`703874736/1073741824` KiB，formal输出预计`<1GiB`；329 fit cache的
  raw X/Y为`161.10GiB`，即使所有谱basis满秩的严格上界也约`221.81GiB`，低于gpu01 `/dev/shm 252GiB`，held conditions评估时
  ephemeral不累计；host available RAM在profile清理前仍约`474GiB`；
- macro5完成后立即以6个独立GPU workers覆盖全部451 conditions和40对correct-vs-wrong Program；若macro5通过absolute/causal Gate，
  继续同一run到下一预注册checkpoint形成adjacent稳定性。若non-pass，先按fit、video-held、task-held、role、family和Program-causality
  定位最早失效接口；不续训掩盖明显fit失容，不做seed/LR/width小扫。shuffled/reversed不使用。

## 2026-08-28 v5 P0通过，P1六任务资格已预注册

clean pushed detached `main@e2f9d33`上的真实38-target P0已在gpu01物理1完成并通过。K1/K4均使用纯Native Stage 0，实际
Action Meta module/parameter为0，source与G2 Program trainable为0；38 targets最终物化76 tensors的唯一rank12+4 rank16 adapter，
并由真实policy消费。外部frame chunk为4与one-chunk的raw primal、solve、conditioning和relative error均为0，minimum update
cosine为`0.9999999999999998`；K4 video集合置换最大误差`2.384185791015625e-07`且`beta=[.25,.25,.25,.25]`。全部
Program/primal/event/scale梯度有限非零。K1/K4分别耗时`38.07/148.15s`，policy consumption为`.164s`，peak allocated/reserved
约`15.83/18.57GiB`；artifact为
`runs/analysis/pi05_ecp_shared_compiler_g3_v5_f0_e2f9d33_task93_gpu01p1_20260828.json`。

P0前两次真实运行暴露并修复了一个工程而非科学失败：FP32 covariance按外部chunk分组会在近`1e6`条件数下改变截断谱；随后即使
solve一致，signed replay仍因chunk归并顺序有约`.0176`最大相对差。当前B0与B1都按固定candidate microblock归并（input 400、
output 1600 candidates），外部frame chunk只影响流式读取与至多一个microblock remainder，不再改变数值定义。旧functional-polar
没有恢复为active路径。

P1固定使用`configs/pi05_ecp_primal_capacity_p1_v1.json`：3个meta-fit tasks`[1,8,9]`、3个target-fit tasks`[72,73,75]`，
覆盖q/v浅中深及action-in/out八个targets；每task用最低ordinal的两条mapping-fit videos等权优化，同一预注册held video全程零梯度、
不选checkpoint。每task只优化跨两条fit video共享的input/output native primals；scale固定为held-excluded fit-consensus初始化并受
`s_ref`上界约束，G2/source/shared compiler/scale/Action Meta全部冻结。固定500 steps且只保留final task code。Gate要求fit/held
median recovery分别`>=.80/.75`、held/fit`>=.85`、held相对optimistic native projection median`>=.80`、四family held median
均`>=.65`且六个task held均`>=.65`。六个独立单GPU worker只并行task、不改变task权重或科学batch；P1只裁决current-bank
primal→dual/replay的多task方向容量，不证明shared Program mapping。

P1 retained-code ownership保持阶段专用且不产生第二个Writer：`primal_capacity.py`只拥有task-local primal/fixed-scale数学，
`primal_dual_runtime.py`新增的`prepare/apply`接口只让同一detached bank operator在固定步优化中复用，`primal_capacity_run.py`只拥有
单task worker与固定训练合同，`primal_capacity_aggregate.py`只拥有六任务Gate聚合，脚本只是薄入口。新增代码量来自三条真实视频的
重资产准备、固定步训练证据与并发安全的task-level artifact三项职责；没有通用框架、版本fallback或并行deployment路径。P1完成后
runner/config作为可复现capacity evidence保留但不被P2/Writer forward调用；P2只复用已验证的primal-dual operator与损失原语。

clean pushed detached code authority `5706cc8`增加了不改变P1数学的固定bank训练快路径：三条视频各自完成一次canonical B0和边界正确的
X/Y materialization，随后每步以全candidate exact softmax批量重放；canonical deployment Writer及P0仍使用chunked streaming。
真实task1 smoke中，materialized对canonical streaming的minimum update cosine为`.9999081`，maximum relative error为`.01356`，
而相同初始化recovery只相差约`2e-6`；一步forward/backward从`1.703s`降到`.2266s`，约`7.5x`，固定500步预计`113.3s`。
三视频准备为`131.3s`，peak allocated/reserved约`11.87/18.03GiB`；Action Meta/source/G2/shared compiler/scale trainable均为0。

### P1 formal launch contract

- canonical authority：包含本合同、可由`origin/main`到达的clean pushed detached commit；scientific code authority为`5706cc8`，
  后续只允许文档性launch-contract提交，不改变config/code/test；每个task report必须记录同一exact commit；
- scale/data：固定六tasks`[1,8,9,72,73,75]`、每task两条最低ordinal mapping-fit videos、一个预注册zero-gradient held video、
  八个targets、500 steps、final-step-only；六个worker分别占一张GPU且无NCCL，task聚合等权；
- command：在同一frozen worktree分别执行
  `env CUDA_VISIBLE_DEVICES=<0..5> PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false numactl --cpunodebind=<0|1> --membind=<0|1> /data1/user/ymdai/projects/EMBER/.venv/bin/python -u scripts/probe_ecp_g3_primal_capacity_p1.py worker --config configs/pi05_ecp_primal_capacity_p1_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --task <TASK> --output-dir <FORMAL_ROOT>`；
  全部worker成功后仅执行一次同脚本`aggregate`子命令；不resume、不覆盖，任一失败保留partial并以新root修正后重跑；
- live resources：launch前gpu01物理0--6均为Default、空闲且无compute process，首版用物理0--5；0--2绑定NUMA0、3--5绑定
  NUMA1。gpu02物理0--6有他人任务、7空闲，不跨节点拼卡；`/data1` quota为`703699480/1073741824` KiB，formal新增预计
  `<0.1GiB`，输出到唯一`runs/analysis/pi05_ecp_primal_capacity_p1_v1_<COMMIT7>_gpu01p012345_20260829/`；
- evidence/Gate：保留每task final code、curve、fit/held/optimistic rows、冻结状态、显存/耗时、aggregate与completion。Gate沿本节前述
  `.80/.75/.85/.80/.65`合同；P1通过只解封P2 shared mapping，不证明Writer或closed-loop通过。shuffled/reversed不使用。

P1 formal已由clean pushed detached `main@c9e8198`在gpu01物理0--5完成并通过，artifact为
`runs/analysis/pi05_ecp_primal_capacity_p1_v1_c9e8198_gpu01p012345_20260829/`。fit/held median recovery为
`.971731/.954539`，held/fit `.982308`，held相对optimistic native projection median `.992193`；held q/v/action-in/action-out
family medians分别为`.939825/.941630/.995402/.945222`，minimum task held为`.935001`。全部七项Gate checks为true，六task、
meta/target两role及四family均非擦线通过。逐task held为task1 `.960931`、8 `.954449`、9 `.935001`、72 `.955273`、
73 `.953605`、75 `.954629`；最弱细项是task9 action-out `.874307`，仍远高于`.65`门槛。

六个worker都保存固定step500的17-tensor task code和完整`0/1/50/100/200/350/500`曲线，无partial/failure；final gradients有限，
held backward为0，Action Meta/source/G2/shared compiler/scale trainable全部为0。每task total约`224.8--226.8s`，train约
`78.5--99.6s`，peak allocated/reserved约`9.83--11.57/18.01--18.11GiB`；六卡总墙钟约3分47秒。P1因此证明v5 current-bank
global dual/exact signed replay在多task、多family、浅中深及未参与梯度的视频上保留接近optimistic的task-local primal方向；它不证明
shared Program mapping。当前最早未验证接口唯一收敛到P2 frozen full-Program-to-primal scorer，scale仍冻结到F4。

更新时间：2026-08-28。G2 formal authority仍是clean pushed `main@c1493a1/macro_00000020`，F1 bank-operator仍由clean pushed
`main@435cb4a`通过。最新完整F3仍是clean pushed detached `main@78b7e58`的fresh IEEE macro5：fit/held/p10/task-holdout为
`.086508/.083131/.072629/.096191`，四family held为`.021698/.065269/.085933/.173804`，明确non-pass且未续训。

full functional-polar、S1 native-Q sketch与S2 set-summary/query-conditioned/cross-image scorer均已按各自Gate non-pass，不再是deployment
候选。behavior-aligned identifiability也已完成：真实cross-episode flow-gradient验证rank4 policy descent且三条bank均有`.90--.91`
可达方向；旧selector与bank-independent dual的held仅`.0229/.0745`，而Program/task primal经每条当前bank的global covariance对偶化后
fit/held立即恢复到`.904--.911/.901`。v5的38-target P0现已通过，当前下一步是上述六任务P1；只有其通过才训练P2 shared
Program-to-primal mapping。

第三位专家已锁定远程`main@9b52e59`及其可达历史完成审计；1033行原文逐字保存为
`docs/expert_review_20260828_g3_functional_sketch.md`。最新裁决是：full functional-polar保留为fit-only teacher/reference和容量诊断，
不再作为deployment候选；其S1失败分支要求full polar/native-Q sketch都只作diagnostic，后继轻量student先通过task-local正控。
S1、S2及behavior反事实现已把根因进一步定位到bank-specific dual坐标；当前v5是根据后续真实证据作出的active修正，不把专家未写死的
跨video归一化或Final训练形式冒充专家原话。

owner接受该方向，并明确：fit row的universal baseline必须leave-one-task-out，task-holdout只用fit tasks；12-task阶段每role各保留
一个true task-holdout；具体causal阈值在shared结果前只用universal negative、task-local free-query positive和`78b7e58`失败checkpoint
一次校准并sealed。S1失败后不再冻结/训练其native-Q sketch，首版改为冻结现有candidate encoder并训练set-summary/scalar-energy。

当前clean pushed `main@27bde62`已接通sealed fixed nested projection、流式
native/key cross-image、`r_s={16,32,64}`共享前缀、`C_rQ` full-native free-query正控和exact signed replay。28项相关合同通过，实际
source/G2/compiler trainable均为0且Action Meta实例为0。clean detached task93/q20 formal已完成：两条既定K1、两个members的rank64
effective-update为`.156687--.157438`，input/output full-native free-query最低约`.41397/.25373`，streaming/materialized最低
`.9999769`；同task/video/member/target的sealed F1 analytic positive仍为`.995560--.997907`。因此含per-row minimum`.95`的S1
容量Gate已被正式反例否决，不运行其余96 conditions，也不训练该native-Q sketch。exploratory完整`H`最佳top64仍远低于门，补充排除
fixed random projection抽坏；该早停不估计50-task分布。当前按专家预定fallback设计不经过native-Q rank64瓶颈的pure low-dimensional
set-summary candidate-logit执行面；task93/q20首轮机制witness已formal non-pass，尚未进入12-task或shared训练。

当前实现面已把S1的frozen runtime与真实candidate-bank构造抽成共享owner，并新增measure-normalized mean/variance DeepSets summary、
candidate-local nonlinear basis×bank/code-conditioned coefficients的bounded positive/negative scalar energy、event-normalized exact X/Y
pooling和固定task-local rank/event code；不含native-Q lift、per-video/per-candidate参数或teacher forward。31项相关合同通过。gpu01物理2
两步真实smoke已完成：task93的fit videos 18/48与zero-gradient held video 0全部成功capture，Action Meta/source/G2/compiler实际可训练
参数均为0；去除event-invariant key的8倍重复编码后，训练为`4.27 step/s`、peak allocated/reserved约`5.42/5.84GiB`。clean pushed detached
`main@4d84dee`随后完成固定1000-step formal：fit median仅`.328188`，held`.175318`，held/fit`.5342`，held input/output仅
`.100649/.042760`，全部Gate checks失败；total`296.52s`，训练`7.72 step/s`，不是吞吐或运行故障。

fit-only nested oracle把失败进一步定位：同一video18真实bank的global free logits和eventwise free logits分别达到
`.9999996/.9999861`，排除native X/Y、teacher、signed pooling与eventwise reduction的硬容量上限；固定formal candidate basis后即使移除
summary/code映射、每video直接优化低维系数并加入强factor credit，fit仍只有`.359/.353`，canonical global reduction更只有`.048`。
从fresh训练candidate basis并用短期subspace credit时，eventwise bound8/bound14及global bound14也只有`.233/.241/.203`，且均未发生
logit饱和；因此bound与normalization不是`.3`量级主因。最早实现接口是S2所谓“frozen existing candidate encoder”实际由reference config
按seed新建、没有加载任何G3 checkpoint，故冻结的是随机128D native/key projection。下一单一因果修正只加载clean detached
`78b7e58/macro5`中fit-trained candidate encoder/trunk/metadata/key projection并继续冻结它；summary、score、loss、videos、steps和Gate不变。
该精确加载路径的一步真实smoke已通过：609 tensors、8,006,400 parameters来自登记authority，旧query/Program路径未加载；
source/G2/compiler trainable与Action Meta均为0，三视频capture、forward、gradient和checkpoint写入正常。

clean pushed detached `main@6b97100`随后在gpu01物理0完成同一1000-step v2 formal witness。fit videos 18/48的best update为
`.355018/.349191`，fit median`.349191`；zero-gradient held video0仅`.131624`，held/fit`.37694`，held input/output为
`.112037/.038104`，五项Gate全部失败。相对v1，fit只提高约`.021`而held更差；训练仍为`7.55 step/s`、total`279.57s`，不是
吞吐、梯度、authority或Action Meta故障。这确认`78b7e58/macro5`的fit-trained frozen 128D candidate chart也不足以让当前
set-summary/scalar-energy学生获取fit方向，但不等于所有set函数失容。按预注册分支没有进入12-task/shared训练。

后续只读重放首先修正了一个更早的event measure错误：per-event unit-mass bank必须按`rho * pre-normalization event volume`混合；
raw `rho`或uniform mixing把同一解析dual从`.9956--.9979`降到约`.043--.055`，正确混合恢复`.9757--.9876`。在正确测度下，冻结
chart的matched 1000-step fit/held为`.31884/.04363`；bound0.1、antithetic branches与native-diagonal三个机制arm仍分别只有约
`.209/.046`、`.223/.052`和`.050/.034`。这些诊断没有改变formal v1/v2 artifact，只修正后继执行合同。

预注册candidate-chart acquisition也已完成：只解冻q20所选chart的`363,520`参数，并与scorer/free code合计训练`2,648,100`参数；
source/G2/旧compiler/Action Meta保持冻结，videos18/48训练、video0零梯度。1000步final fit为`.30286`，held仅`.03527`，因此没有形成
“解冻chart”的formal修正；首版128D mean/variance separable scalar-energy函数类已经按Gate淘汰。

同bank nested operator curve又显示q20 input稳定秩为`483--487/1024`，output groups为`243--256/256`；64/128维低秩修正的完整update
仅约`.28/.51`，256维约`.83--.85`，input约到384维、output约到192--224维才接近exact。普通或diagonal-preconditioned PCG到256次
仍不能稳定恢复input，完整update约`.48--.64`。query-conditioned三次set read的真实step500在update-only credit下fit/held仅
`.15209/.09229`；同一cross-image直接扩大到rank224/384仍只有`.1593--.1630`。所以不再把对角、少量rank、朴素迭代solve、加宽
cross-image或多次read包装成新deployment路径。下一诊断只使用授权fit-task cross-episode action/flow或冻结functional gradient，
分别检查direct free logits和轻量selector能否形成实际policy descent且跨视频稳定的rank4 native residual；在结果前不发射12-task/shared。

本轮retained-code ownership与lifecycle明确如下：`native_bank_runtime.py`唯一拥有frozen source/G2/compiler加载、K1 capture及真实candidate
bank边界，并已从S1 analyzer删除213行重复实现，供S1诊断和S2共同复用；`set_summary.py`唯一拥有当前deployment-candidate的集合摘要、
scalar energy与exact signed pooling数学；`probe_ecp_g3_set_summary_witness.py`只拥有固定task93/q20资格流程和formal artifact写入，不成为
第二个Writer。S1 analyzer/config保留为明确`deployment_candidate=false`的历史诊断入口。witness runner的移除触发点是12-task正控把同一
student执行面吸收到canonical S2 trainer后；届时只由Git与formal artifact保留task93专用流程，不继续并行增长。新增代码量来自真实
bank共享owner、独立数学owner和可复现资格入口三个不可混合职责，不构造通用框架或版本fallback。共享runtime现同时负责显式加载并记录
fit-trained candidate encoder authority，避免把fresh seeded projection误报为existing encoder；其余compiler仍冻结且不进入student。

## S2 task93/q20 set-summary witness v1 formal launch contract与结论

- canonical code：包含本合同的clean pushed `origin/main`，从exact commit建立clean detached frozen worktree；实现authority为
  `ce47ff8`，其后的launch-contract提交不得改变代码、配置或科学口径；formal runner必须核对detached、clean且commit可由
  `origin/main`到达；
- fixed evidence：task93、q target20；fit videos为18与48，zero-gradient held video为0；teacher固定为target step1000/2000两个
  verified members。只训练set-summary scalar-energy selector与跨三条video共享的task-local rank/event free code；source、G2
  Program、旧compiler、candidate encoder、carrier与scale全部冻结，Action Meta module/parameter必须实测为0；
- objective：fit-only set-valued paired effective-update loss加跨fit-video dispersion；held video不参与梯度、loss、checkpoint选择或
  fixed-step schedule。固定AdamW、1000 steps、LR `1e-3`，无early stop、无resume；最终只报告single final checkpoint；
- Gate：两个fit videos的effective-update median `>=.90`，held effective update、input pushforward和output pushforward均`>=.80`，
  held/fit `>=.8`；任一失败即淘汰首版mean/variance set-summary + separable scalar-energy函数类，不进入12-task正控。通过也只证明
  当前函数类具有task-local跨video容量，不代表shared Program mapping或G3通过；
- execution：单进程单A40。launch前live核对gpu01/gpu02后选择gpu01物理0（UUID
  `GPU-658b6043-6454-1228-bffc-0e2fe22e5013`、serial `1322123016829`）作为当时空闲设备；它当前没有prohibited authority。
  两步真实smoke训练为约`4.27 step/s`，1000步连同一次runtime/capture预计总wall约6--7分钟，单task图不能从数据并行获益；
- storage/output：`/data1` live quota为`703635280/1073741824` KiB，预计新增低于0.1GiB；formal root固定为
  `runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v1_gpu01p0_20260828/`且launch前必须不存在。有效non-pass亦保留完整
  config、checkpoint、metrics、report和completion；shuffled/reversed不使用。

精确命令（`<FORMAL_WORKTREE>`在launch时替换为exact detached路径）：

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data1/user/ymdai/projects/EMBER/.venv/bin/python -u scripts/probe_ecp_g3_set_summary_witness.py --config configs/pi05_ecp_set_summary_s2_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v1_gpu01p0_20260828 --formal
```

该历史命令已由detached `4d84dee`在gpu01物理0完整执行；artifact有效保留。v1 non-pass只淘汰fresh随机projected key与首版
set-summary/scalar-energy/paired-credit的组合，不能冒充fit-trained candidate encoder或整个pure set-summary路线已失败。

## S2 task93/q20 set-summary witness v2 formal launch contract与结论

- canonical code：包含本合同与v2 loader/config的clean pushed `origin/main`，从exact commit建立clean detached frozen worktree；formal
  runner核对detached、clean且commit可由`origin/main`到达；
- 唯一因果变量：candidate chart从v1的fresh seeded compiler改为clean detached `78b7e58` formal F3 macro5中fit-trained
  candidate encoders、family trunks、合法metadata和key projections；精确加载609 tensors/8,006,400 parameters。旧query projection、
  Program query与其余compiler不加载或训练；source、G2、compiler均须实测0 trainable，Action Meta module/parameter为0；
- 保持不变：task93/q20、fit videos18/48、zero-gradient held video0、两个verified members、mean/variance summary、separable scalar
  energy、eventwise exact signed pooling、bound8、task-local code、paired update+dispersion、AdamW `1e-3`、固定1000 steps、无early stop/
  resume、final-step single checkpoint及全部`.90/.80/.80/.80/.8` Gate；
- 决策：通过才进入12-task free-query正控；fit仍失败则确认fit-trained 128D chart/score image不足，下一接口是解冻或替换candidate
  encoder，不改summary或启动shared训练。held先失败才讨论跨video泛化。global/eventwise与bound不在本次变更中，因为nested free-logit
  oracle已证明二者均约1.0，且bound8/14、global对fresh basis均未改变当前数量级；
- execution/output：单进程单A40，预计runtime/capture约145s、训练约130s、总wall约5分钟，peak reserved约18GiB capture与6GiB
  training；formal root固定为`runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v2_gpu01pX_20260828/`并在launch时以实时设备替换
  `X`，root必须不存在。launch前重新live检查gpu01/gpu02与`/data1`独立quota；有效non-pass亦保留，shuffled/reversed不使用。

精确命令（`<FORMAL_WORKTREE>`、`<GPU>`与output中的`X`在launch时替换）：

```bash
env CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data1/user/ymdai/projects/EMBER/.venv/bin/python -u scripts/probe_ecp_g3_set_summary_witness.py --config configs/pi05_ecp_set_summary_s2_v2.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v2_gpu01pX_20260828 --formal
```

该合同已由detached `6b97100`在gpu01物理0完整执行，exact output为
`runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v2_gpu01p0_20260828/`。fit median、held、held/fit、held input/output分别为
`.349191/.131624/.37694/.112037/.038104`，全部低于`.90/.80/.8/.80/.80`必要门。artifact含single final checkpoint、逐步metrics、
report与completion；没有held梯度、outcome读取、shuffled/reversed或teacher tensors。该结果淘汰“冻结`78b7e58` candidate chart + 当前
mean/variance summary + separable scalar energy + paired-only credit”的组合，尚不淘汰允许candidate chart从native value获得直接
functional credit的学生。

## S1 task93/q20 formal early-disqualifier launch contract

- canonical code：包含本合同的clean pushed commit，并从该commit建立detached frozen worktree；worker强制核对clean detached Git；
- scale：既定F1 panel中的task93、q target20、全部两条sealed K1 conditions、两个verified members与同一nested rank16/32/64曲线；
  task93/q20在此前actual-operator根因审计中已是固定witness。S1 Gate含每row至少`.95`，任一正式反例即可否决合取条件，但不能估计
  50-task family/depth分布；
- authority：source step1000、G2 `c1493a1/macro20`、native-teacher v2、F1 50-task/98-condition analytic operator与固定LIBERO
  dataset commit；每条sketch row必须匹配同task/video/member/target的sealed F1 positive row；
- execution：单进程单A40，不训练、不建optimizer、不读action/outcome/held gradient；source、Program、compiler全部冻结，Action Meta 0；
  runtime初始化只支付一次并与两条condition分开计时。smoke峰值reserved约18.02GiB，输出预计远低于1MiB；
- Gate：报告rank64 sketch-to-teacher、streaming/materialized和exact F1 analytic-to-teacher；若sketch任一row低于`.95`则status为
  `complete_capacity_disqualifier`，停止其余S1条件并转入专家规定的pure low-dimensional set-summary free-query函数类；
- output：`runs/analysis/pi05_ecp_functional_sketch_s1_early_q20_<authority>_gpu01pX_20260828/`，启动时替换exact commit/device；fresh-only，
  root必须不存在，失败attempt不得覆盖。launch前live复核gpu01/gpu02及`/data1`独立quota；shuffled/reversed不使用。

该合同已由detached `27bde62`在gpu01物理2完整执行，exact output为
`runs/analysis/pi05_ecp_functional_sketch_s1_early_q20_27bde62_gpu01p2_20260828/`，status
`complete_capacity_disqualifier`。runtime初始化`128.37s`只支付一次；两条condition分别`12.63/12.84s`，其中Pass A+native capture
`5.34/5.43s`、bank key/measure`5.00/5.28s`、三rank curve+exact replay`2.29/2.13s`；peak allocated/reserved约`10.34/18.02GiB`，
formal raw rows、worker completion与report共约28KiB。结果不是训练、closed-loop或shared mapping结论，只关闭该S1 sketch函数类。

以下launch contracts与逐阶段记录是历史证据，不是当前待执行命令；不得从中恢复v4 formal路线。

## IEEE fresh F3 formal launch contract

- canonical code：clean pushed、detached且可由`origin/main`到达的`78b7e58`；formal worktree为
  `/data1/user/ymdai/projects/EMBER-worktrees/ecp-g3-ieee-f3-formal-78b7e58`，不读取任何旧compiler checkpoint；
- upstream authority保持不变：G2 `c1493a1/macro_00000020`、source
  `pi05_source_base_v1_seed7_1k_e2cc238/step_00001000`、同一50-task/451-condition native-teacher manifest、mapping split、
  tokenizer与LIBERO data root；Action Meta、source、Program与scale仍冻结；
- 唯一因果变量是native dual FP32 matmul从TF32改为IEEE，并由F0证明实际生效；model、loss、rank12+4、data、seed、LR、schedule、
  每步固定6 logical tasks（3 meta+3 target）、每task两条K1 fit videos及F3 `.75/.50/.8` Gate均不变；
- fresh训练到macro5，共25 optimizer steps；gpu01物理`0,1,2,3,4,5`、world6、`NCCL_P2P_DISABLE=1`。若产生有效step，任何
  topology或科学合同变化都必须新root fresh；只有optimizer step前的工程失败才可在保留错误证据后清理incomplete root；
- output为`runs/outputs/pi05_ecp_shared_compiler_g3_f3_ieee_fold0_m5_gpu01p012345_r6_20260828/`，log同stem放在
  `runs/logs/`；checkpoint、trainer state、metrics、run contract与completion保留。预计新增低于0.5GB；launch前`/data1`
  quota使用约`670.9GiB/1TiB`，余量充分；
- macro5后必须完整只读评估451 conditions并报告fit/held/p10/task-holdout、四family、held/fit、breadth与相邻趋势；不以train loss、
  direct-anchor panel或12-condition诊断代替。shuffled/reversed、validation/test、closed-loop held5均不进入该Gate。

精确训练命令：

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_ecp_shared_compiler.py --config configs/pi05_ecp_shared_compiler_g3_v3.json --phase f3 --mode formal --asset-root /data1/user/ymdai/projects/EMBER --source-run /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_shared_compiler_g3_f3_ieee_fold0_m5_gpu01p012345_r6_20260828 --stop-after-macro 5 --log-every 1
```

## F3 functional-anchor formal contract与结论

- canonical workspace：`/data1/user/ymdai/projects/EMBER-worktrees/ecp-g3-functional-anchor-formal`，从包含本记录的clean pushed
  `origin/main` detached创建；训练前再次核对branch为空、status clean且HEAD等于origin/main；
- upstream authority：G2 `c1493a1/macro_00000020`、source `pi05_source_base_v1_seed7_1k_e2cc238/step_00001000`、现有
  50-task/451-condition native-teacher manifest、固定mapping split与LIBERO data root；Action Meta关闭；
- scale与科学合同：F3 global statistics on，从fresh到macro5，共25 optimizer steps；每步固定6 logical tasks、3 meta+3 target、
  每task两条K1 fit videos，fit-video consensus仅作训练label，held/validation/test gradient为0。macro5后完整只读451-condition Gate，
  不用train loss或consensus ceiling代替；
- runtime：gpu01物理`0,1,2,3,4,5`，world6 torchrun/DDP-style手工gradient sum，`NCCL_P2P_DISABLE=1`、`OMP_NUM_THREADS=8`；
  启动前live snapshot显示六卡均为空闲且Default mode，gpu02已有他人任务，故不跨节点；
- output：`runs/outputs/pi05_ecp_shared_compiler_g3_f3_functional_anchor_fold0_m5_gpu01p012345_r6_20260827/`，日志在
  `runs/logs/`同stem文件；预计单checkpoint与trainer state低于0.5GB，`/data1` launch前quota为`670.8G/1T`；
- resume：fresh run不读取旧checkpoint；若科学合同、world topology或实现改变则使用新root fresh。只有optimizer step前的明确工程
  失败可在保存错误证据后清理incomplete root；一旦产生有效step/checkpoint便不得覆盖或伪装续跑。

该合同已完整执行并正常结束。macro1到macro5 train mean recovery为`.001057/.011125/.031016/.053658/.077663`；随后固定六worker
完整评估451 conditions并形成上述non-pass。checkpoint、trainer state、raw rows、aggregate及completion均保留；没有resume macro10。

精确训练命令：

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_ecp_shared_compiler.py --config configs/pi05_ecp_shared_compiler_g3_v3.json --phase f3 --mode formal --asset-root /data1/user/ymdai/projects/EMBER --source-run /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_shared_compiler_g3_f3_functional_anchor_fold0_m5_gpu01p012345_r6_20260827 --stop-after-macro 5 --log-every 1
```

## 2026-08-26至v4的历史状态（已被页首最新状态覆盖）

两轮专家回复均已收到、完整阅读并固化。active design现为`docs/event_conditioned_policy_compiler_design.md`中的
**ECP Native-Factor Compiler**，核心架构、数据角色、阶段Gate、最终controls和停止条件均已明确。

专家1416行原始回复已完整保存为`docs/expert_review_20260824_native_factor.md`，逐行内容与附件一致，仅换行从CRLF标准化为LF；
第二位专家1538行原始回复已逐字保存为`docs/expert_review_20260826_bank_conditioned_native_factor.md`并与owner提供附件byte-identical；
active design明确是解释/执行层，不能替代任一原文。

全仓库orientation和两轮owner复核已完成，owner已正式许可推进。G1 task-local free-code capacity oracle与G2 Natural Program
均已通过；当前处于G3 shared compiler，v1 macro10、v2 macro5、低维dual-basis与跨视频score acquisition已形成连续non-pass及
根因证据。固定target-specific raw-dual basis不具备首版所需的compact functional capacity；进一步证据否定了把逐video解析
dual/score直接作为candidate-local shared mapping标签。新专家确认主因是stable functional target被表达成随current-bank covariance
旋转的脆弱inverse coordinates；当前活动修正为两阶段流式bank-conditioned Pass B：B0累计每视频单位质量的statistics/native anchors，
regularized solve后由B1重放同一bank做exact signed pooling。没有恢复旧Writer/realizer/GOMQ/PECS/人工process路线。

owner接受该G3修正，同时明确覆盖专家的Final初始化偏好：整套Writer完全随机初始化、从头端到端fresh联合训练必须保留为Final
正式可选项；G1--G3是组件因果验证而非Final强制课程。Final仍不预设目标LoRA，具体初始化与最小监督由matched closed-loop证据裁决。

F1隔离operator已从clean pushed detached `435cb4a`完成formal Gate：新增的`ember.ecp.bank_conditioning`只拥有B0 sufficient
statistics、截断谱query solve与B1 exact signed pooling，随后由同一模块直接供canonical compiler复用；analytic anchor、teacher
lookup和四family对照只存在于单一formal analyzer，不进入deployment模块或checkpoint。固定50 tasks/98 conditions/536
member-family rows上，q/v/action-in/action-out的task-mean median分别为`0.999871/0.999824/0.999960/0.999884`，minimum为
`0.999757/0.999544/0.999951/0.999743`；streaming-to-materialized row minimum为`0.99999988`，全部远高于预注册门。
Action Meta实际未加载、held reads为0；全仓`177 passed`。ridge在约`1e6`条件数bank上会收缩有效方向，故首版采用与G1稳定span
一致的FP64谱截断，并保留q/action-in真实output-group相对gain，未做width/LR/seed扫。该pass只证明operator capacity与数值合同，
不证明shared Program-to-anchor mapping。analytic-only analyzer只保留为formal evidence入口，旧G3 v2 compiler将在
bank-conditioned canonical实现通过对应Gate后删除而不是长期并行。

bank-conditioned canonical实现面现已接通但尚未冒充formal Gate：`SharedNativeFactorCompiler`执行B0流式单位measure统计、
Program/native anchor compatibility、可开关的FP64谱solve和B1 exact signed replay；输入候选不含output type，四类Y bank保持
各video的adj/init/goal边界。F2的off模式精确定义为`C=I`：保留centered first-moment native anchor与B1，不是固定query或第二套
deployment Writer。50-task/451-condition split预注册为329 fit、40 held-video、82 held-task；每macro用固定5次`3 target + 3 meta`
更新覆盖全部15个target-fit并轮换15/25个meta-fit，world size只做cost-balanced吞吐分片。mapping训练只解冻anchor scorer，
Program/source/scale/Action Meta均冻结；相邻checkpoint稳定性和held/fit口径已进入Gate。全仓当前为`184 passed`。

实现ownership保持单一：`shared_compiler.py`拥有唯一deployment Writer的B0 functional-statistics与B1 replay orchestration；
`bank_conditioning/anchor.py`、`functional_polar.py`、`anchor_solve.py`和`operator.py`分别拥有Program/candidate content、actual-operator
gauge、B0 anchor/native solve与exact pooling。`mapping*.py`只拥有F3 acquisition/evaluation/Gate，`f0.py`只拥有一次formal prelaunch
qualification，scripts均为薄入口。旧v1--v3实现只由Git与formal artifacts保存，active train入口唯一指向v4 functional-polar，
没有`C=I`、Euclidean compatibility或旧Writer fallback；F3结束后mapping-only训练面随F4 joint owner演化或退役。

clean pushed detached `main@19b5b3f`的formal F0已通过。task93真实K1 gradient与rank12+4 materialization有限，唯一完整LoRA为
76 tensors/38 targets并被policy实际消费；source/Program trainable与Action Meta module/parameter均为0，checkpoint无lookup/teacher
state，K4 teacher reads为0、四video权重均为`.25`、反序置换最大误差`1.43e-6`。同一cached X/Y bank按chunk4与one-chunk重放时，
38个最终低秩更新的cosine最低`.99999976`、相对Frobenius误差最高`.00066443`，solve metric误差`3.71e-14`；内部rank槽位最大
差`.00311`来自等价small-core SVD gauge，只作诊断。峰值allocated/reserved为`33.85/34.43GB`，独占gpu01当前逻辑0且计算段
UTL约`61--100%`；F0只解封F2，不代表shared mapping已通过。

clean pushed detached `main@2199a76`随后完成一次且仅一次预注册F2 `C=I` formal。六卡world6从fresh训练到macro5/25
optimizer updates，mean recovery由macro1的`.000639`升到macro5的`.019690`；随后六个独立只读worker完整覆盖451个唯一条件
（329 fit、40 held-video、82 task-holdout），Action Meta module/parameter、held gradient及shuffled/reversed use均为0。正式
task-equal aggregate的fit/held-video/task-holdout recovery median分别为`.022243/.022858/.018919`；held-video四family median
action-in/action-out/q/v分别为`.039958/.022185/.004722/.023158`，远低于`.65`，所有F2 primary checks均失败。因为fit本身已经
接近零，最早失效接口是candidate-local compatibility加first-moment anchor没有学会，而不是held泛化、训练/评测信息墙或F1
operator。该结果只淘汰`C=I`假设，不反证F1已证明有容量的current-bank solve；不续训F2、不做LR/seed/width小扫，下一步从fresh
进入F3。

clean pushed detached `main@c1e26ce`随后从fresh运行F3并以同一world6 topology exact-resume到macro10/50 updates。训练
mean recovery由macro1 `.002204`持续升到macro5 `.042865`和macro10 `.087444`，但后段增量已经放缓。macro5与macro10均用六个
独立只读worker完整覆盖同一451条件；macro10 fit/held-video/task-holdout task median为`.089915/.089704/.096849`，held/fit
为`.997650`，说明跨video与跨task泛化并未崩溃。held-video action-in/action-out/q/v median为
`.125947/.177230/.013288/.052761`；overall median/p10仍远低于F3 Gate的`.75/.50`。macro5到10的held median提高`.041271`，
同任务median absolute delta `.041962`且无held下降；相邻稳定性数值成立，但两个checkpoint都未通过primary。当前不续到macro20，
先定位q/v远弱于action families的shared anchor scorer表达/梯度接口；F1 operator capacity、信息墙和泛化已被分别排除为最早问题。

针对同一`c1e26ce` macro5/macro10、同一真实task93/video31 bank的只读factor几何与单family backward进一步把接口收窄到
**两侧rank4子空间获取的credit starvation**，而不是pairing、solve或长训不足。macro10的q实际update recovery仅`.012892`，其
input/output one-sided span ceiling为`.192547/.094122`；v为`.046297`与`.260870/.282059`；action-in为`.125741`与
`.775570/.145645`；action-out为`.094190`与`.237240/.657518`。q从macro5到10的两侧ceiling只由`.184690/.089680`
增至`.192547/.094122`，held task2 q同样只有`.013565` recovery与`.204308/.088427` ceiling，排除了fit特例。旧F3虽按family
等权完整update cosine，但q input/output key gradient norm只有`.0643/.0313`，action-out则为`2.6973/.6383`；双线性update-only
credit没有给高维q/v足够的一侧方向信号。

当前单一机制修正不改变bank、Program、rank、query/key宽度、group gain、data、LR或Gate：仍由完整四family update产生一个
set-valued global member posterior，再将其detach后同时监督同一member的gauge-invariant input subspace、output subspace和paired
update direction，三项固定等权；scale仍冻结且无selection梯度。六卡真实5-macro qualification中三项loss分别从
`.939056/.922342/.999256`降至`.923254/.902963/.997695`，graph、梯度和唯一checkpoint均正常；run contract再次确认
Action Meta module/parameter为0、source/G2 Program/scale全冻结。该结果只解封从fresh重跑formal F3，不是451-condition Gate。

该修正随后由clean pushed detached `main@84903aa`从fresh训练到macro5并以同一world6 topology exact-resume到macro10。训练
mean recovery由`.001262`升至macro5 `.021016`和macro10 `.070337`；三项训练loss在macro10降至
`.771307/.825056/.930808`，说明新credit确实被优化，但六worker完整451-condition结果仍明确non-pass。macro5的
fit/held-video/task-holdout task median为`.024363/.025418/.034375`；macro10为`.073151/.073029/.087636`，held p10
`.057174`、held/fit `.998320`。macro5到10同任务median absolute delta `.049104`且held没有下降，但两个checkpoint都未过
`.75/.50` primary，因此相邻Gate也不能通过。macro10 held action-in/action-out/q/v median仅
`.098990/.146806/.008482/.040693`，甚至低于旧`c1e26ce`的`.125947/.177230/.013288/.052761`；等权subspace
credit作为充分修正已被formal证伪，不续到macro20。

进一步只读分解把剩余最早接口定位到shared scorer的parameter ownership，而不是再调loss。macro5同一task93/video31上，四family
output-key gradient norm为q/v/action-in/action-out `.022128/.056223/.014089/.251221`，output-query为
`.018642/.055363/.016834/.305811`，跨family方向大多近正交而action-out幅度支配。q的18个固定层目标内部同样近正交：input/output
key pairwise median cosine为`.001448/-.000865`，aggregate gradient只有各target norm和的`.306/.320`；output target norm
最大最小约`20.5x`。独立task94/video11复现q output pairwise median约0、aggregate ratio仅`.282/.276`。macro10 q两侧span
ceiling均值已到`.235654/.093766`，实际update cosine仍只有`.008407`，排除了只是再训练同一共享trunk即可取到现有span。

基于该证据，当时的canonical修正按专家明确建议实现为四family各自共享Program/rank/query/event/gain/candidate trunks，并由38-target
固定LoRA拓扑的zero-init bounded FiLM调制candidate hidden direction；没有task/video/member/authority/frame lookup，也不增加
第二Writer。完整真实profile在gpu01单卡完成一组`3 target + 3 meta` forward/backward/update/checkpoint：222/222 trainable
parameter tensors进入optimizer state，3,187,080个scorer参数、Action Meta 0、source/Program/scale trainable 0，耗时`179.81s`，
peak allocated/reserved `25.16/25.45GB`。全仓`181 passed`且architecture guard无hard violation。该checkpoint-incompatible修正
该profile只为随后从clean pushed detached commit fresh运行同一F3 Gate解封；没有被解释为shared mapping成功。

上述family/fixed-owner修正已由clean pushed detached `main@c3fc8e3`从fresh训练到macro5并以同一world6 topology exact-resume到
macro10；两个checkpoint均完整评估451 conditions。macro10 fit/held-video/task-holdout task median为
`.074715/.074620/.081644`，held p10 `.058381`、held/fit `.998724`，held四family q/v/action-in/action-out为
`.027938/.066509/.044464/.164942`。它没有超过旧`c1e26ce`，故family/fixed-owner ownership是必要工程隔离但不是充分机制，
不续macro20。

同一checkpoint的fixed-key image在稳定`1e-3`相对谱floor下给出q/v/action-in/action-out teacher update ceiling约
`.226/.315/.975/.629`；降到`1e-6`时大多恢复到`.97--.99`，说明信息被压入极弱奇异尾。raw-native与FiLM tangent并未解决：
对应ceiling约`.250/.336/.960/.600`与`.280/.381/.973/.645`。尤其action-in已有`.975`表示容量而训练held只有`.044`，证明
问题不只在key压缩，也在stable task code到current-bank selection的识别。clean pushed `main@4117117`曾接通direct-native F0，
但进一步代数复核发现native query经同一bank covariance逆解会退化为已失败的raw-query transfer；因此没有启动无信息的formal F3，
该实现已从活动分支回退，只作为工程/反证历史保留。

新的same-task三video bank-global oracle用task93的videos 31/32拟合共同feature code、video46作holdout。full feature inverse与
per-event symmetric inverse-square-root均显示：只用两条video的minimum-norm code时q/v/action-out held update近零，而把第三条
video只加入共同code估计的transductive正控制立刻达到约`.90--.93`；action-in的per-event两video inductive held已达`.986`。
这不是held梯度或deployment方案，而是证明共享feature chart中存在强same-task code，失败来自两video解落入巨大train-only
nullspace。冻结G2 `P_lang`由exact language单独产生且same-task不同video确定性相同，故当前有机制依据的修正是用`P_lang`产生
task-stable anchor query，动态`P_scene/P_process/rho/tau/sigma`只控制event/frame measure；每video、每event先对candidate keys做
detached symmetric inverse-square-root，再形成native anchors与原有native-bank solve。

该修正已在唯一canonical `SharedNativeFactorCompiler`中接通三次流式读取：B0a feature statistics、B0b native anchors/solve、B1
exact signed replay；没有task/video/member/frame lookup、第二adapter或Action Meta。定向CPU合同现为14 tests通过。gpu01逻辑
`0,1,2,3,4,6`的world6真实profile完成固定`3 meta + 3 target`一组forward/backward/update/checkpoint，step为`77.806s`，峰值
allocated/reserved为`25.59/25.99GB`；feature retained rank最低约`94--96/128`、retained trace最低约`.999999`，native solve
residual约`1e-12`，held/validation/test gradients、Program/source/scale trainable与Action Meta均为0。该profile只证明工程图与六卡
吞吐。全仓`183 passed`，architecture guard无hard violation；随后clean pushed detached `main@20acc33`的F0也通过：K1有效更新
cosine最低`.99999826`、相对误差最高`.001863`，K4置换误差`1.91e-6`，Action Meta 0、38 targets/76 tensors且policy实际消费
唯一rank16。

同一commit从fresh训练F3至macro5并按world6 exact-resume到macro10/50 updates。macro10完整451-condition fit/held-video/
task-holdout median为`.141080/.142120/.145828`，held p10 `.116653`、held/fit `1.00737`；macro5到10的40/40 held-video tasks均改善，
held median净增`.092745`。stable anchor因此实质修复了跨video/task迁移并显著提高绝对获取，但仍远低于`.75/.50` primary。
macro10 held q/v/action-in/action-out仅`.030186/.110266/.180031/.253562`，q明显成为最早接口。

六个task-local、单fit-video 20-step定位probe覆盖meta `1/32/52`与target `92/93/94`：overall train recovery由
`.113--.163`升到`.203--.251`，另一fit video和held video均紧随至`.200--.247`，但q update只到`.0197--.0277`；action-in/out
通常升至约`.31--.49`。这排除了当前最早问题是视频泛化或纯多任务语义竞争。macro10 task93的q target-gradient分解进一步显示，
18个q owners在family-shared input/output query heads上的aggregate gradient仅为各target norm和的`.272/.268`，153对中分别
`76/74`对为负；candidate key共享trunk为`.364/.602`且已有fixed-owner FiLM。语言诊断同时确认`P_lang`去owner baseline后的
task variation非零、same-task跨video严格相同，故不以未证明的G2重训替代最早q接口。

当前唯一机制修正对现有family-shared query trunks增加zero-init bounded fixed-owner input FiLM及fixed-owner/output-group FiLM，
让38个真实LoRA targets获得互不相消的query梯度路径；task dependence仍只来自`P_lang`，没有task/video/member/frame lookup。
该修正与旧checkpoint不兼容，必须从fresh验证。首个clean pushed `7e232b0` F0在任何GPU计算前暴露出
`FixedOwnerQueryFiLM._apply`与`torch.nn.Module._apply`的生命周期命名冲突；这属于可复现工程失败，不是科学non-pass。唯一修复
将内部helper改名为`_modulate`并新增`.to(device)`回归，形成clean pushed `main@d64f7ad`；全仓184项CPU回归与architecture guard
hard checks通过。

clean pushed detached `main@d64f7ad`的formal F0随后通过全部资格项：新input/output owner-query gradient norm分别为
`.015828/.000958`，source/Program trainable与Action Meta module/parameter均为0；K4四video权重均为`.25`、置换最大误差
`1.91e-6`且teacher reads为0。chunk4相对one-chunk的38-target有效更新cosine最低`.99999826`，相对误差最高`.001863`、median
`6.04e-5`，feature metric误差`5.96e-8`；唯一完整rank16仍为38 targets/76 tensors并被policy实际消费。F0总时长`592.16s`，
峰值allocated/reserved约`34.00/42.81GB`。该结果只证明新query路径与部署合同接通；下一步必须从fresh运行同一F3 primary及
相邻checkpoint Gate，不用macro20盲续旧结构。

该fresh F3已由clean pushed detached `main@3e4e9a0`完成macro5/macro10及451-condition完整评估。macro10 fit、held-video、
task-holdout task median为`.162011/.163128/.164562`，held p10 `.133783`、held/fit `1.00689`；相对stable-anchor macro10，40/40
held tasks全部改善，held median增加`.02007`。但held q/v/action-in/action-out只有`.032001/.111951/.256629/.256391`，仍远低于
F3 `.75/.50` primary，两个相邻checkpoint均不能通过Gate。增量几乎全部来自action-in；因此不续macro20，也不把内部loss下降冒充
shared mapping成功。

同一checkpoint的四臂只读ablation显示：移除input owner-query路径使overall仅下降`.000193`，移除output路径下降`.014820`，其中
action-in单独下降`.056749`而q/v变化约`1e-3`。随后六个fit-only task-local probes覆盖meta `1/36/43`与target `85/93/94`：仅优化
fixed input FiLM并让query移动约`.53--.56`个base RMS时，q/v input recovery median只由`.09844/.09725`升至`.13306/.13525`，完整
update反而由`.02983/.10730`降至`.02686/.10292`；仅优化fixed output FiLM并移动约`.48--.50`个base RMS时，q/v output由
`.02150/.14045`升至`.05619/.21256`，完整update仅为`.03356/.08511`。这排除了“formal FiLM只是幅度太小”的解释。

更强的正控制直接把同一六task的q/v input与output owner/rank/event queries全部变成task-local free tensors，同时冻结Program、
candidate encoders、policy和bank operator；20步后q/v update median只由`.02983/.10730`升至`.06519/.14487`，q input/output为
`.18546/.09746`，v为`.19614/.29710`。六task方向一致，Action Meta、held/validation/test gradients及shuffled/reversed use均为0。
该20-step probe不是严格数学上界，但它已经否定“只把fixed FiLM换成更大的target-specific query head就足够”的直接推断；结合F1同bank
analytic operator约`.9998`，下一步必须先对当前candidate feature/compatibility image做cached-to-convergence或解析容量裁决，再决定
是修candidate canonicalizer还是让Program mapping利用更合适的bank-global sufficient statistics。

真实吞吐profile没有用dummy占卡：同一固定六任务、同一梯度结果的F3 optimizer step，gpu01单卡为`181.21s`、峰值allocated/reserved
约`25.14/25.42GB`，计算段SM/UTL为`70--100%`；物理1/2/4/5/6五卡弹性分片为`44.96s`，约`4.03x`加速，各卡约
`19.5--25.1GB`且计算段大多`100%`。当时物理3因他人约`78--92%` UTL而未使用，旧物理0按当时prohibited标记未使用。gpu01在
2026-08-26重启后只枚举7张卡；UUID复核证明当前逻辑0是原物理1，旧prohibited物理0已不在列表，故不能按裸index继续排除。
正式launch仍重新live检查UUID、index、进程与状态；
每卡以真实step time、LoRA/s和持续UTL为准，显存安全余量不是必须填满的配额。

吞吐profile同样按真实稳态而非卡数解释：单worker峰值allocated约`10.1GB`但启动主要是CPU runtime装载；同一A40共驻两个长寿命
worker时实测约`37.5/46.1GB`、稳定GPU UTL `94--100%`、memory UTL约`50--73%`，两条件总墙钟相对串行提升约`66%`。
第三worker无安全显存余量；formal F1实际使用gpu01 p1--p6每卡两个长寿命worker，六卡稳态均约`37.5--37.8GB`且UTL `100%`，
12个worker全部完成，最长总时长`228.44s`。这只优化分片与吞吐，不改变50-task/98-condition authority或Gate。

G2实现面现已接通并通过最小真实检查。Program严格输出`P_lang/P_scene/P_process/rho/tau/sigma`固定schema；每条video分别运行
两条fixed antithetic native probes，再做monotonic canonical alignment与`beta_k=1/K`集合聚合。首轮真实held检查发现把多视频frames
先扁平后按全局chunk分批会使K4集合置换最大误差达`0.132`；按阶段合同改为每条video完全独立native forward后，同一检查降为
`2.38e-7`，K1为bitwise exact identity。该修正针对真实失效接口，不引入learned video reliability或task/frame route。

95-task training-only dynamic label authority已升级并在
`runs/outputs/pi05_ecp_natural_program_labels_g2_v2_cpu_20260825`封存完成：meta56/held15、target-fit19/held5共735,519 frames，
BDDL goal predicates为80个单predicate、14个双predicate、1个三predicate task；`obs[i]`按真实`state[i+1]`恢复，缺失terminal
post-action contact显式mask，`rising[0]`明确比较`states[0] -> states[1]`。全量复核发现4750个demo均无首action完成goal，故v2相对v1
rising数值不变，schema升级防止旧语义被静默复用。四个旧LIBERO-90 scene4任务的HDF5 XML使用pre-rename `salad_dressing_1`，当前BDD model使用
`new_salad_dressing_1`；只在内存中对模型identifier做已验证alias后补齐，未修改原始数据。

formal前代码复核发现并修正两项会污染解释的问题：旧rank-local negative queue与`local_index % 4` robustness使辅助loss随rank分配
不等权；现改为每task一次robustness及固定8个、target/meta fit各4个且与rank/world-size无关的content negatives。旧action target
先按video长度取整再映射action episode，而其它动态标签直接按action episode取整；现统一使用唯一action-episode query grid。修正后
architecture guard无hard violation，全仓库149 tests通过。修正后的真实K4 profile在v2 label authority上完成：macro time 23.53s、
peak allocated 18,853,217,280 bytes、84/84 trainable tensors均进入optimizer state、loss/gradient finite；run contract实际记录固定
role-balanced negatives、每task robustness、Action Meta module/parameter 0及source trainable 0。

clean pushed `main@141a110`的首轮G2 macro10 formal及meta-held15+target-held5 Gate已经完成。same-task nearer为`1.0`、probe
margin为`0.9`、one-event为`0`、median active events为`6`、K1 exact identity与K4 permutation均通过；唯一non-pass是full相对
endpoints的action/progress改善仅`0.0226%`，低于`10%` Gate，因此没有进入G3。

read-only held20机制诊断显示，full相对endpoints的`P_process/rho/tau`差异分别是same-task不同视频差异的约`2.20x/13.77x/60.00x`，
说明最早失效接口不是native动态捕获。相反，decoder action时序标准差仅`0.00060`而target为`0.33789`，清零`P_process`的静态路径
combined loss反而由`0.39574`降到`0.39088`。当前唯一修正因此移除`P_lang/P_scene`对`P_process` fusion及action/progress等时序heads的
直接加性旁路；静态scene-only head保持独立。修正后的task92 K4真实forward/backward已完成，84/84 trainable tensors均进入optimizer
state，gradient有限、peak allocated 18,851,367,936 bytes，Action Meta module/parameter 0且source trainable 0；下一步是集成后fresh
训练并复评同一held20 Gate。

该静态旁路修正已由clean pushed `main@30b98ef`完成，fresh macro10 formal与同一held20 Gate也已完成，但仍为non-pass：
same-task nearer `1.0`、K1 identity、K4 permutation与median active events `3`通过；full相对endpoints改善为`-0.0570%`，
one-event fraction `0.30`，probe margin `0.65`。无梯度temporal诊断显示event weights虽随时间变化（std `0.04398`），最终pooled
state std仅`0.00111`，action预测std仅`0.00093`而target为`0.32725`；hard-nearest、uniform event measure和mean-repeated
process消融几乎不改变loss，最早接口已推进到event/owner内容而不是query measure。

进一步在target-held5对比初始frozen Stage 0与macro10 encoder状态：Stage 0的event/owner relative RMS为`0.06069/0.36992`，
训练后的raw encoder降为`0.02601/0.22824`，fusion后owner进一步降至`0.14837`。这证明G2梯度在Program读出成立前先侵蚀了已有
Stage 0 event grounding。当前唯一机制修正因此冻结已验证的Stage 0 v3 observer，仅训练新的language/scene/process readers、alignment
与training-only heads；不同时修改owner readout、slot、width、seed、LR或Gate。run contract将实际强制Action Meta 0、source trainable 0
及native observer trainable 0。

该修正的task92真实K4 forward/backward smoke已经完成：46/46个新增Program parameter tensors进入optimizer state，loss与gradient
有限，peak allocated 10,016,652,800 bytes；实际run contract记录native observer trainable 0且training mode为eval、source trainable 0、
Action Meta module/parameter 0。checkpoint中39个encoder tensors与Stage 0 v3 authority逐tensor相等，证明optimizer/forward没有改写
frozen observer。profile输出核对后将删除，不冒充formal evidence。

clean pushed `main@db84a50`的frozen-observer formal已从fresh训练到macro10，并按原world5 topology exact-resume到macro20。macro10
held20 Gate中same-task nearer `1.0`、K1/K4、median active events `5`与one-event `0`通过，但full相对endpoints仅`+0.0051%`、
probe margin `0/40`；macro20仍为non-pass，full相对endpoints为`-0.0207%`、probe margin `0/40`、one-event `0.025`，其它上述
资格项保持通过。fit total从macro10的`1.17260`降至macro20的`0.97637`，因此内部loss下降没有转化为视频动态因果增量。

macro20无梯度诊断确认冻结修正确实保存了native结构：raw full event/owner relative RMS为`0.06252/0.36771`，fused为
`0.05590/0.26447`，full与endpoints的fused Program RMS差异仍有`0.00618`。最早失效接口现为temporal owner readout：当前共享
`Linear(128,1)`对38-owner轴严格置换不变，owner entropy为`0.99898`，action prediction temporal std仅`0.00173`，而target为
`0.32725`；从macro10继续到20没有改善该比例。两个raw antithetic branches仍不稳定，但把零均值residual只在辅助Gate路径缩小、
而不改变部署`P_process/rho/tau/sigma`，会成为Gate-only旋钮，已明确不采用。

当前隔离实现只把training-only temporal readout改为38个固定LoRA owner各自的linear query；38条query从旧共享Linear完全相同的
向量初始化，保持其余head的旧RNG序列，之后只能由owner-specific梯度分化。queries跨task共享且只读取
`P_process` content，不是task-ID route。scene head、probe、Stage 0、Program schema、数据、loss、seed/LR/slot/width与Gate均不变。
task92真实K4 profile已通过：owner-query gradient norm `0.01827`，一步后query row centered RMS为`3.23e-5`，证明共享初始化已由
owner-specific梯度分化；46个Program parameter tensors/915,554 parameters trainable，native observer/source policy/Action Meta
trainable均为0，39个observer tensors逐tensor不变，peak allocated 10,016,671,744 bytes；profile只作机制smoke，核对后删除。

clean pushed `main@407340b`的owner-specific scalar-query formal已从fresh macro10按同一world2 topology exact-resume到macro20。
macro10/macro20 held20 Gate的full相对endpoints改善分别为`+0.0158%/-0.0340%`，probe margin均为`0/40`；same-task、K1/K4、
active-event范围继续通过，所以仍未进入G3。owner queries的row-centered RMS从自身RMS的`1.58%`增长到`2.94%`，但macro20
actual与强制shared-query的held combined loss只差约`4.9e-5`，action prediction temporal std仍只有`0.00171`，target为
`0.33589`；hard-owner readout同样不改善。该结果淘汰owner-specific scalar selection作为充分修正，也排除继续到macro40只等待
query分化的解释。

进一步无梯度反事实把frozen Stage 0 raw process与其已训练action head重新配对：held action absolute loss从当前fused/current的
`0.25511`降至`0.20767`，说明Stage 0坐标与head包含可复用信息；但full相对endpoints仍只改善`0.2467%`，prediction temporal std
仅`0.00298`，所以只复用旧head不足以满足G2动态门；该反事实也没有提供直接增加owner value map的充分证据。当前最早接口是
absolute cross-episode MSE被trajectory mean解主导，未单独约束query-time action/progress residual。当前机制修正保留absolute losses并新增等权query-centered action/progress
residual MSE，不改模型、Program schema、数据、K、seed/LR或Gate。task92真实K4 profile得到`action_temporal=0.14324`、
`progress_temporal=0.08779`、owner-query gradient norm `0.01839`，39个observer tensors不变，Action Meta/observer/source trainable均为0，
peak allocated 10,016,671,744 bytes；profile已删除。

clean pushed `main@68f8705`的temporal-residual objective已从fresh训练到macro10。held20 Gate中same-task nearer `1.0`、K1/K4、
median active events `5`与one-event `0`继续通过，但full相对endpoints只改善`0.0381%`，probe margin为`0/40`，因此明确non-pass，
没有进入G3，也没有用fit loss下降或继续同一低更新数run冒充进展。

该轮结果冻结后的read-only根因分析排除了新的表示架构猜测：固定Program的full-owner temporal readout相对endpoints可改善
`15.17%`；tied-query与independent-query初始化曲线近乎相同；cross-episode监督可识别。旧trainer每macro访问38个task却只执行
一次Adam更新，所以macro10仅10次更新。同一frozen readout temporal loss从`0.311873`开始，10/60步仅为
`0.311827/0.311164`，到200/500步才降至`0.294034/0.257824`。最早接口由此定位为optimizer cadence，而不是再次增加Program
slot、width或readout结构。

当前隔离实现保持模型、数据、loss、K、seed/LR峰值和Gate不变，把每macro拆为10个role-balanced optimizer steps：常规
2 target-fit+2 meta-fit，尾部1+1并随macro轮换；scheduler与exact-resume cursor按真实optimizer step计数。单卡及gpu02 world4
真实profile均完成：world4实际聚合4个互异task、role为2+2、finite owner-query/全局gradient，46/46 Program tensors进入Adam，
四个rank checkpoint齐全；run contract记录source/observer trainable 0、Action Meta module/parameter 0。profile只作执行证据，
核对后删除，不冒充formal。

该cadence修正已由clean pushed `main@49e7769`完成，并从fresh训练至macro10/100 optimizer steps。held20 Gate仍为non-pass：
full相对endpoints改善`0.3080%`、probe margin `13/40=0.325`，低于`10%/0.75`门；same-task、K1/K4、event范围与tau资格项均通过。
相对旧10-update checkpoint，动态增量由`0.0381%`提高约`8.1x`，20个held task中17个方向改善，meta/target-held分别为
`0.2781%/0.3891%`，所以这是宽泛但幅度不足的真实信号，不是偶然峰值，也不能进入G3。

冻结macro10后仅用12个fit task（target/meta各半，K=1/2/4等量）做gradient diagnostic；held gradient为0。full与endpoints的
`P_process`差异仍有`0.07296 RMS`，但full action/progress prediction temporal std仅`0.00379/0.00160`，而target为
`0.35248/0.32500`。temporal梯度相对non-temporal在Program process参数上为`0.01031/0.10345`，在temporal decoder上为
`0.00885/0.18567`；cosine仅`-0.065/-0.071`，说明问题不是方向性强抵消，而是近常数读出使temporal梯度小约`10--21x`。
结合frozen readout在100--500步才开始明显展开的既有曲线，下一步按同一commit/world4 topology exact-resume到预注册macro20；
这是对“有效但尚未跨过学习时标”的可证伪检验。若held增量和prediction temporal std没有实质继续增长，该解释即被否定，下一修正
必须直接针对Program-to-temporal-readout的梯度饥饿/近常数结构，而不能靠继续训练或超参小扫。

macro20首次resume在训练前被exact-contract拒绝：旧v2 contract把当时`origin/main` tip记录为`authority_commit`，后续纯文档提交使该
浮动字段变化，尽管detached formal code仍是同一clean pushed `49e7769`。失败attempt没有追加invocation、metrics或checkpoint；本轮
用可逆local ref pin通过旧contract后立即恢复`origin/main=e952823`。主线窄修复现让formal contract固定记录自身detached commit，
profile仍记录当前authority tip，并以定向回归保护；它不改变模型、数据、优化或Gate。

同一run随后已成功exact-resume到macro20/200 updates并完成held20 Gate。full相对endpoints改善从macro10的`0.3080%`跃升到
`8.6878%`，probe margin由`13/40`升到`36/40`，same-task与K1/K4 invariance继续通过；fit-only prediction temporal std从
`0.00379/0.00160`升到`0.03393/0.04789`。这验证了readout学习时标，但Gate仍non-pass：median active events `1`、one-event
fraction `1.0`，且动态增量尚未严格超过`10%`。

按K分解已把最早接口定位到canonical alignment：K1在macro20仍平均`6.42` active events，而全部K2/K4训练条件均为one-event；
每条video的local presence仍约7--8个有效槽，DP却把约`6/8` path mass集中到同一canonical slot。fit-only、held-gradient 0的
counterfactual中，identity会产生5--8 events而过强；仅给现有DP首尾加canonical 0/7边界锚点就恢复为稳定3 events，同一frozen
decoder的full增量从`15.82%`略升到`16.47%`。当前隔离实现只做这个boundary修正，保留中间stay/skip、content/time score、
readout、loss、数据、K、seed/LR和Gate。全量合同测试为`155 passed`；真实macro0 K4 profile读取4条视频、102个采样帧并完成
forward/backward/optimizer step，gradient norm与owner-query gradient均finite/nonzero，active events为2、one-event为0，峰值显存约
`9.97 GB`。run contract实测Action Meta module/parameter均为0，source policy与native observer trainable parameters均为0。
clean pushed `main@c1493a1`随后从fresh训练到macro10/100 updates并按同一world4 topology exact-resume到macro20/200 updates；两段均
exit 0，metrics/invocations严格为20/2。macro10已把event Gate修复为median 2、one-event 0，但动态增量仅`0.8268%`；macro20 held20
Gate全部通过：full/endpoints action+progress loss为`0.28167/0.36207`，相对改善`22.2047%`，median active events 4、one-event 0、
probe `38/40`、same-task 1.0、K1 identity 1.0、K4 permutation 1.0（max abs `4.77e-7`）、tau violation `0.00357`。因此G2的
最早失效接口确为无边界K>1 alignment，而不是readout容量；当前冻结`macro_00000020` Program并进入G3。

G3首个共享编译器实现面已接通但尚未形成formal科学结果：frozen G2 Program现暴露每条video的canonical event assignment；Pass B
按真实native content与Program query计算正负两支softmax，输入候选严格为`(video,frame,probe,horizon)`，输出候选严格为额外含
`type={abs,adj,init,goal}`的集合。每条video先以event assignment和时间quadrature构成单位质量measure并独立chunked pooling，再由
uniform初始化、最大修正0.5的置换不变bounded beta合并K=1/2/4；K=1严格identity。实现不含task/video/frame selection参数，最终只
输出一套rank4 residual并复用唯一rank12+4 rank16 materialization。

同时已接通95-task/118-member authority、G2 checkpoint冻结加载、member相对carrier的small-core最佳rank4投影、set-valued
functional effect losses，以及target-fit successful-member occupancy的窄 evaluator capture合同。全仓`158 passed`；覆盖
chunk/non-chunk边界等价、gradient、K1 identity、K4 permutation、bounded beta及无free logits。target-fit occupancy和75-task/93-member
fit effect authority现已完成，真实GPU forward/backward/materialization也已通过；这些仍只是formal launch资格，不是G3闭环Gate结果。

该实现面随后补齐了canonical训练runtime：每个optimizer step严格一项target-fit与一项meta-fit，member identity只拥有training-only
critic/sampler；deployment forward只接收Program与native candidates。loss由single-global-member log-sum-exp、四family等权functional、
cross-episode flow、sensitivity-normalized mobile update、carrier preservation与定期same-task不同video functional consistency组成；
source、Native observer、G2 Program、carrier及experts全部冻结，只更新shared query/key/signed pooling/scales/bounded beta。由于每step
只有两个独立task，允许world size收紧为1--2，避免多GPU空转。复核发现旧实现只拟合member flow response而未实际使用预留action demos；
现已改为fit75独立action episodes上的真实PI0.5 flow loss，held actions读取为0，member flow只保留在global set-valued effect中。

target-fit verified occupancy首次clean formal capture中，step1000为`19/19`，step2000为`17/18`。唯一失败是global38 moka-pots
state4；新旧结果的adapter、init state与全部policy-noise seeds一致，但旧evaluator在step434成功，新clean evaluator到520未成功，说明
该晚完成轨迹对允许的BF16/kernel低位差异缺少裕量，而不是authority错配。修正不试seed：只用旧sealed fixed50结果预先选择每member
完成步数最短的成功state（再以state ID破同分）；global38因此改为state36、旧证据step401成功。clean重采后step2000为`18/18`，
与step1000的`19/19`共同进入critic；失败run作为formal capture evidence保留，不混入effect authority。

一项meta-fit真实GPU effect-bank smoke已完成：复用的旧meta occupancy schema为`ember_writer_occupancy_trajectory_v1`，新target schema为
`ember_pi05_occupancy_trajectory_v1`；sealer现在同时接受并逐项核对suite/task/state/success/adapter，不做宽松fallback。task1输出
4个trajectory states、1个global member、38-owner response、flow/action response及76个rank4 projection tensors，全部finite；实际对象图
Action Meta为空。约33.7MB smoke artifact核对后已删除，不冒充formal evidence。

完整effect root为75个fit tasks、93个successful members、约3.0GB，manifest状态`complete`；meta56与target-fit19角色及全部task manifest
均核对。三步真实训练profile覆盖普通K1/K4、触发same-task-other的K2/K4，以及target93共332个采样帧的长K4：三条compiler gradient
probe均finite/nonzero，真实cross-episode flow分别进入loss，唯一rank12+4 rank16被policy实际消费；Action Meta module/parameter为0，
source/Program trainable为0。最初双条件同时驻留在约44.39GiB OOM，已按最早接口改为primary主loss先backward、other对detached primary
response做轮换consistency，并对每video signed pooling做activation checkpoint；最终三步峰值分别约16.68/17.39/29.28GB，均exit0。

G3 held5 Gate执行面已补齐：同一冻结compiler checkpoint可一次性分别物化`correct_full`、`first_final`和disjoint
`same_task_other`三套评测条件，每个条件仍只有一套完整rank12+4 rank16；另有fit75 frozen-`P_lang` linear-kernel ridge到verified
rank4 effect的learned language-only control，held video/action/reward读取均为0。paired Gate report强制核对250行source、task、normalization、
tokenizer与RNG身份、三条video arm的唯一compiler checkpoint、carrier retention、breadth、Goal/Long、full相对language/endpoints及
same-task retention；shuffled/reversed未进入该Gate。

G3 v1随后从同一clean detached `5140362` fresh训练到macro10/190 updates，并完成同一五臂strict250。正式Gate仍为non-pass：
carrier/language/full/endpoints/same-task=`43/42/38/39/40`；full逐task为Spatial0 `32`、Spatial9 `2`、Object8 `4`、Goal5 `0`、
Long6 `0`，breadth`3/5`、carrier retention `32/43`、相对language/endpoints `-4/-1`。仅same-task retention
`32/38=84.2%`与adapter/checkpoint/配对authority通过；报告为
`runs/analysis/pi05_ecp_shared_compiler_g3_gate_m10fresh_5140362_4770c5e_20260826/report.json`。shuffled/reversed未使用。

macro10把“只是warmup后更新不足”证伪：total/global-member/effective-update虽从约`2.381/1.015/0.929`降至
`2.135/0.926/0.894`，190个optimizer steps却全部触发同一全局gradient clip；macro10 pre-clip median约`10.87`，scale path
gradient均值约`13.88`，而input/output query约`0.754/1.057`。macro5到10的input/output query-key相对参数变化只有约
`2.14%/1.70%`，scale组约`14.4%`，且held闭环没有跨suite或方向性跃升。因此不能再靠续到macro20或调LR/seed解释。

独立fit-only K1几何同时排除了“真实native bank不能承载member效果”：6个fit tasks、9个verified members的full-to-native
update cosine median为`0.7029`，native named/global functional retention median为`0.7855/0.7981`，9个member均有正action benefit；
full-to-mobile update cosine为`0.9978`。最早失效接口由此确定为shared Program-to-content selection的监督与优化，而不是rank4压缩、
K1 native span或多视频鲁棒性。当前活动G3修正为fit-only、K1-only的离线native-feasible teacher：对formal40 schedule实际覆盖的
50个fit tasks/451个task-video条件，把verified member投影到对应真实K1 bank，训练时以detached set-valued responsibilities监督
共享query/key产生的input/output subspace、paired update direction与独立small-core spectrum。K2/K4严格不读teacher；teacher键和
factors不进入compiler forward、checkpoint model state或deployment。selection与scale/video使用分离clip预算，首版保持原Program、
sampler、K、LR、rank、bounded beta和无confidence gate。

该G3 v2修正已在clean pushed `main@93dffc7`实现并通过全仓`168 passed`。formal teacher authority已由同一detached commit在
gpu02 p4/p5/p6封存：50个K1-covered fit tasks（meta31/target19）、451个唯一task-video、covered tasks内68个verified members、
662个teacher states、828MiB；三个worker、aggregate和master均exit0。root明确登记full fit authority 75 tasks、missing25、held reads 0、
Action Meta 0、deployment use false，且只含38-target rank4 pre-scale directions、scales与provenance。

随后单GPU真实三步profile覆盖K1/K2/K4、same-task consistency和target93长K4：两个K1条件分别精确读取1个task tensor shard并找到
`2/1`个members，所有K2/K4条件teacher reads与lookups均为0；input/output query和scale gradients全部finite/nonzero，selection与
scale/video pre-clip norms被分别记录，scale/video heads不反传shared context。所有条件均物化76 tensors的唯一rank16，K>1 beta
从uniform的最大偏差低于`1e-6`；Action Meta module/parameter、source与Program trainable均0，峰值allocated
`29,320,510,976` bytes。该profile只证明v2训练面和信息墙接通，不是G3 Gate；下一步从fresh到macro5并复评同一五臂strict250。

G3 v2随后由clean detached `2a7f760`从fresh训练至macro5/95 optimizer updates，并以同一checkpoint分别物化full、first+final和
same-task K4 banks。五臂strict250的carrier/language/full/endpoints/same-task为`43/42/41/38/37`；full逐task为Spatial0 `34`、
Spatial9 `5`、Object8 `2`、Goal5 `0`、Long6 `0`，breadth`3/5`、carrier retention`33/43`、相对language/endpoints为`-1/+3`、
same-task retention`30/41=73.2%`。只有carrier retention和全部authority检查通过，因此明确non-pass；shuffled/reversed未使用。

同一fit K1 `meta9/video40`真实bank上的固定条件审计排除了loader或梯度墙故障。step0到macro5的input/output subspace从
`0.9298/0.9292`轻微降至`0.9070/0.9083`，但paired update cosine由`0.00409`降至`0.00299`，spectrum loss由`3.7536`升至
`4.2118`。macro5梯度分解显示teacher-selection与其它selection梯度范数为`0.3235/21.8015`（约`67x`），teacher spectrum与其它
scale梯度cosine为`-0.989657`；两条显式gradient wall泄漏均为0。teacher-only反事实能同时改善selection/update/spectrum，证明分支
可优化，但v2把几乎正交且量级悬殊的selection credit以及近乎反向的scale credit放在同一步，旧functional职责实际覆盖了direct
mapping监督。当前不续训v2，也不通过seed/LR/loss系数小扫修饰结果。

隔离credit之后又完成了更早selection接口的固定bank反事实。target20的free full-native query到500步仍只有`0.4313` factor、`0.1624`
update cosine；相同真实X/Y、teacher和冻结G2 `rho` measure下的FP64 inverse-covariance dual达到input/output
`0.99628/0.99997`、update`0.99750`，retained scatter condition约`1e6`；将dual缩成最大logit`0.1`并继续使用现有
`softmax(+s)-softmax(-s)`仍为`0.99749`。最早失效接口由此定位为shared query隐式获取高条件数dual，而不是native banks或pooling表达力。

clean pushed detached `main@e7d86b0`已完成上述50-task、98-condition、四family formal probe。full-dual reference的task-mean
update cosine median/p10/min为`0.996949/0.995468/0.993884`，worst-video为`0.996487/0.994944/0.991649`，证明capture、dual、真实
X/Y回放与signed pooling合同有效。相同LOTO basis压到最大128维后，overall task-mean median/p10仅`0.288444/0.249615`，50 tasks中
没有一个达到`0.95`；family median分别为action-in `0.999983`、action-out `0.146885`、q `0.000490`、v `-0.000586`，故Gate明确
non-pass，不能扩到38 targets或把compact raw-dual code接入compiler。

同一formal artifact上的后续fit-only机制反事实把失败再分解了一层：对最差q/v/action-out条件，不再欧氏投影dual，而是在当前bank的
functional geometry中求固定basis内最小二乘，128维update cosine分别由`-0.0119/-0.0288/-0.1178`升至约
`0.684/0.507/0.805`，证明旧realizer几何错误；但仍远低于门。把同一LOTO构造延伸到实际稳定秩后，q在384维为`0.981`、
action-out在384维约`0.978`，v到其稳定span仍只有约`0.933`，说明128维raw query span本身也不够。selected-task effect-basis
反事实同样需要约384--512维，因而不恢复fixed effect-code/fit-span realizer。

直接对现有content-key做小型functional-span acquisition进一步发现：当前按width共享的64维近线性key在两条同task视频训练后，第三条
视频的三family update cosine均值只由`0.380`升到峰值约`0.521`；改为owner-specific keys改善已见拟合但没有消除未见视频缺口。
按解析曲线指定的512维owner-specific key在相同train/third-video条件上，其解析functional span从step0已达
`0.9907/0.9904`，50步为`0.9997/0.9932`，但cross-covariance最小/最大奇异值比仍约`1e-8--1e-6`。因此下一接口不是继续宽度扫，
exact bounded-score反事实确认该问题不可忽略：随机512-key必须动用约`1e7`条件数才使q/v/action-out达到约
`0.993/0.984/0.994`，在`1e6`内只有约`0.956/0.936/0.966`，且同task跨video query cosine很低甚至为负。相反，直接以真实native
X/Y作为content keys时，`1e6`截断配合不读取完整bank的固定`0.01` score scale，三family跨三video update cosine均值为
`0.99886/0.99551/0.99788`、minimum为`0.99810/0.99447/0.99703`。q的八个output groups若各自单位化会降至约
`0.967--0.985`，保留归一到`[0,1]`的相对group gain后恢复约`0.999`。因此下一canonical候选是owner-native direct content score、
非线性高容量Program query与显式bounded group gain；先做隔离的K1 mapping acquisition，不能把这些selected-condition内部值冒充
G3 Gate或shared mapping成功。

该候选随后经过同task跨video反事实复核，结论已被收紧。q/v/action-out的verified teacher effective update在三条video间仍较稳定，
mean cosine约`0.873/0.866/0.884`；冻结G2 Program更稳定，same-task flattened cosine均值约`0.9971`。但逐video minimum-norm
native dual明显旋转：直接把一个video的raw query用于另一条video时q/v几乎为零，action-out仅约`0.086`；同一条raw query对三条
video联合求解的update upper bound也只有q/v/action-out约`0.736/0.381/0.823`。保留8个event query虽把两条训练video拟合到约
`0.965/0.525/0.986`，第三条held video仍为`-0.004/0.012/0.049`；稀疏event anchor同样不迁移。candidate-local nonlinear
512D key以factor loss训练时，train/held update仅为q `0.177/0.105`、v `0.244/0.175`、action-out `0.593/0.487`。

最后以逐video FP64 analytic dual产生直接score标签，并把同一candidate-local nonlinear scorer固定训练到2000步。训练score已持续升至
q input/output `0.887/0.699`、v `0.897/0.722`、action-out `0.912/0.979`，排除了500步欠拟合解释；但held-video q为
`0.133/0.111`、v为`-0.246/-0.232`，action-out虽为`0.491/0.961`，完整held paired update仍分别只有
`-0.001/-0.003/0.114`。结合50-task/98-condition frozen-Program decoder的task-holdout与held-video dual decodability低值，当前
最早失效接口是：解析dual/score依赖整条bank的高条件数协方差，既不是稳定Program的确定标签，也不是单candidate内容可唯一决定的量。
因此不再把direct score supervision或owner-native raw key写成已确定的canonical修正；现存代码仍是已记录non-pass的G3 v2实现，尚无
新架构被保留。后续必须在两项有区别的假设间做机制裁决：一是保持one-pass合同、用跨task/video factor监督学习真正的functional
canonicalizer；二是让Pass B利用bank-global sufficient statistics/preconditioning，后者可能需要修订“query预先确定且只流式一遍”
的当前合同。该分叉先交由全新专家基于完整远程历史复核，不能用width/LR/seed或更多逐video score拟合替代。

owner明确formal训练实现不得固定world2：保持固定全局task group、role权重、loss归一化和optimizer cadence，launch时按1--6张有效GPU
弹性分片；exact-resume锁定该run首次launch topology。该要求同样适用于后续G3/G4/Final训练，不能让卡数改变科学batch定义。

owner再次明确G1--G3的分段冻结是组件因果验证，不是Final默认训练模板。组件Gate通过后，G4/Final优先直接联合优化完整Writer并使用
最小充分loss集合；只有后续机制证据要求时才采用有退出条件的warmup或分段。该建议与当前joint Writer目标一致，具体loss删留仍由
闭环和最早失效接口决定。

G1 canonical实现面已接通。首轮formal held5 free-code优化与strict250已完成：唯一rank16 candidate为`88/250`，relative recovery
`45/67=0.6716`、breadth`3/5`、高于carrier`2/5`、carrier retention`30/43`，逐task为`33/18/37/0/0`；因此Gate为
`non_pass`。全部250 paired rows、47 shards与15 workers完整，Action Meta关闭，失败是科学结果而非运行故障。

按Gate合同完成的read-only span/response分析定位到当前scalar output pooling的结构性上限：无bias q/v组合位于base weight列空间；
q只能覆盖`1024/2048`输出维。action-in带bias且可跨output types相减，精确上限为`span(column_space(W),bias)`、至多`33/1024`。
15个known-success mobile-rank4
reference整体只保留约55--56% update energy。将independent member正交投影到该上限后的paired strict250为`109/250`，逐task
`34/30/45/0/0`；原independent mobile authority为`120/250`且Goal/Long为`11/8`，投影单独抹掉了两个process-sensitive suite。

q-head修正的formal optimization与strict250已经完成：唯一rank16 candidate为`84/250`，逐task`28/21/35/0/0`、relative recovery
`0.6119`、breadth3/5、高于carrier2/5、carrier retention`24/43`，Gate non-pass。step500 generated update与known-success references
整体cosine仅约`0.06`，所以增加的q自由度没有被随机近均匀dense logits实际利用。

随后稳定native-bank投影诊断把latest member materialize为同一唯一rank16，在strict250达到`94/250`、逐task`24/24/44/1/1`；
relative recovery、breadth、Goal/Long和四task高于carrier均成立，但retention仅`22/43`，故不是Gate pass。该结果证明稳定bank内存在
process-sensitive闭环方向，并把最早失效接口推进到free-logit可达优化与retention。

latest-only解析free logits的精确step0 strict250已完成：`100/250`、逐task`24/28/45/3/0`，relative recovery`0.851`；breadth4/5、
Long 0、仅3/5高于carrier且retention`22/43`，Gate仍non-pass。step0与解析投影residual cosine为`0.952--0.964`，第一次Adam更新后即
降至`0.039--0.070`；五task 500-step formal也未在预注册内部effect/update证据上恢复step0，因此没有用held reward选择被扰动checkpoint。

set-valued formal与strict250已完成：每task按fixed50 count选择carrier/independent/latest/independent/independent，结果`111/250`、
逐task`35/29/45/2/0`，relative recovery`1.015`且retention`34/43`；breadth4/5、Long 0、仅3/5高于carrier，Gate non-pass。

最早接口现为signed-measure闭式初始化的数值稳定性：`1e-3` span threshold使scatter inverse condition number可达约`1e6`，task94
FP32实际direction cosine最低为input `0.978`、output `0.883`。只把小型初始化solve的sufficient statistics改为FP64后，真实task94
forward/materialization两侧minimum cosine均恢复到`>=0.99999988`；candidate、rank、pooling、loss、38 hooks、唯一rank16和Action Meta 0
均不变。clean pushed formal与同一strict250已经完成：`116/250`、逐task`35/34/44/3/0`、relative recovery`1.090`、
retention`35/43`；但breadth4/5、Long0且仅3/5高于carrier，Gate仍non-pass。

FP64已排除数值失真后，最早剩余结构接口是action-in whole-vector output pooling：`32 -> 1024`真实Y共享一个scalar signed measure
时必然受限于`span(column_space(W),bias)`、至多`33/1024`。paired response只把task94的action-in target恢复为known-success
independent mobile，其它37 targets保持当前native candidate，Long从`0/50`变为`1/50`。当前canonical修正因此按native input width
把action-in真实Y切成32个32D blocks，各block独立signed pooling；完整response counterfactual为`118/250`、逐task
`35/35/44/3/1`、breadth5/5、4/5高于carrier、retention`35/43`，数值上满足全部G1门，但因task94 action-in来自privileged
reference而不是native pooling，不能冒充G1 pass。候选索引、四类bank、rank、scale、唯一rank16和G1/G3边界不变。

action-in native-block修正由clean pushed `main@31f0053`完成，142项CPU回归及task94真实forward/gradient/materialization smoke通过。
从detached frozen worktree生成的五task step0 bank完成同一four-arm strict250：`114/250`、逐task`35/31/45/2/1`，relative
recovery为`71/67=1.060`，breadth5/5、四suite非零、Goal2、Long1、4/5 task高于carrier、carrier retention`35/43`，全部Gate
checks通过。54/54 shards、250/250 rows与18/18 workers完整，Action Meta module/parameter为0，adapter为唯一完整rank12+4 rank16，
没有使用shuffled/reversed。该pass只回答native X/Y banks与signed pooling形式的capacity问题；shared Program query-key attention仍由G3验证。

专家复核锁定的是远程`main@7ab5a04`。其后`6fdaeb8`只删除退役代码/人工资产并整合文档，没有新增实验结果；专家指出的当前
Stage 0实现缺口已在瘦身后的代码中复核：q/v owner仍来自layer input/residual，尚无真实38-target input/output hooks。因此该科学
裁决可直接应用于当前活动树。

## 专家裁决已固化

- ECP继续推进，名称细化为ECP Native-Factor Compiler；
- 取消neural `q_pi -> fixed effect-code realizer -> LoRA`前置链；
- privileged experts/effects只作nonparametric set-valued training critic；
- Video Program固定为owner-specific language/scene/ordered events及`rho/tau/sigma`；
- 第二pass读取38个target的真实native inputs/outputs与动态differences；
- Program通过signed pooling产生mobile rank4，与frozen rank12 carrier拼成唯一rank16；
- 当时唯一下一步是fold0 held5 task-local free-code strict250；该Gate现已通过；
- 通过后依次进行Natural Program、frozen-Program shared compiler、joint Writer、conditional outer credit和final fresh；
- validation8与完整video controls的资格门、Test8 sealed规则及ECP根本失败条件均已固定。

owner已接受专家的Action Meta门槛：只有base Writer先产生明确闭环增量，matched Action Meta又有明确净收益且无breadth/retention
损害时才加入，否则保持关闭。rank12 carrier + mobile rank4是首版有证据配置，不是永久锁死；active design保留了rank-ceiling
诊断通过后重开分配的正式分支。

owner最新取消所有人为阶段工期、固定修正次数、结构版本和训练轮数上限。Gate与失败定位仍保留；有新机制证据可继续修正，
无信息超参小扫不算推进。执行应积极复用、并行和提升吞吐，顺利时力争数天内完成整体架构实现并推进关键Gate。

owner最新进一步明确：唯一正式性能目标线是validation8 strict paired correct严格`>145/400`，且必须同时满足
相邻稳定性、breadth、四suite非零、Goal/Long贡献、same-task鲁棒性和视频因果controls，不能用偶然峰值通过。
shuffled/reversed只在最终selected checkpoint选定并冻结后测试时序特异性，不进入训练、loss、checkpoint选择、
G1--G5 Gate或架构修正依据。

## 本轮仓库整理结果

- 退役Writer、functional decoder、ECP v1--v24后继、MDCO/PECS、fixed/two-sided realizers与人工process模块已删除；
- evaluator保留source/task-expert adapter、dynamic queue、occupancy diagnostics和strict aggregation；
- canonical基础模块为source/corpus/SFT、LoRA、task experts、Stage 0、policy effects、functional loss、reward/occupancy与evaluation；
- 旧41份Markdown、87份分散证据JSON、退役配置/测试及约11.6GB可重建人工datasets/runs已清除；recovery Gate A残留
  作为历史formal evidence保留，不删除也不恢复为当前路线；
- 瘦身提交`6fdaeb8`的126项活动CPU测试、compile、脚本入口与引用审计均通过；
- orientation清理节点当时只有`main`一个worktree、无task-owned branch或GPU job；后续G1按合同使用隔离实现面与detached formal
  worktree，动态状态以本节“当前下一步”和live检查为准。

## 当前可复用资产

- 固定24/8/8 split、71-task source corpus、五fold meta manifests与target fold0 manifests；target其余folds在G4多fold验证前补齐，
  不阻塞G1；
- frozen source PI0.5 authority、rank16 LoRA topology/materialization；
- task-expert bank、independent successful members、mobile-rank4解析容量与effect calibration；
- Stage 0 v3 full-layer/horizon observer、transition matcher、event binding/segmenter；
- cross-episode video/action schedule、functional flow loss与detached LoRA gradient bridge；
- natural reward rollout、occupancy capture、BDDL progress与cost-balanced strict evaluator；
- ignored `runs/`中的唯一formal checkpoints、raw rows和aggregate。

## G1真实smoke证据

- 使用纯`load_frozen_native_observer`路径，`action_meta_lora=None`、`install_action_meta_lora=False`；实际对象图中无
  `MetaLoRAStack/MetaLoRAProjection`，policy与Stage 0 trainable列表均为空；
- 38个target均从identity LoRA wrapper的真实`base_layer`捕获X/Y；输入候选不含output type，输出候选含四类bank；
- task 90的一步真实优化中`rank_queries/event_logits/input_logits/output_logits/scale_logits`均有有限非零梯度；
- 峰值allocated约27.24GB A40显存；输出checkpoint为single complete rank16、76 tensors、carrier slots`[0,12]`与task slots`[12,16]`；
- profile输出已核对后删除，不作为formal evidence。
- q-head修正后task93一步真实profile中`output_logits`的16,793,600个元素全部获得非零梯度，其余四类free variables也全部非零；
  peak allocated为28,332,442,624 bytes，single complete rank16与纯Native/Action-Meta-off合同保持不变。
- reference-projected初始化后task93的pre-update latest loss从旧随机路径约`1.32`降至`0.817`，global-member effect为`0.107`；
  全部五类free variables仍有非零梯度，峰值28,676,537,344 bytes，真实chunk cache为521,625,600 bytes。
- action-in 32×32D修正的task94真实profile中，32个output blocks均为stable rank32，input/output minimum direction cosine仍为
  `>=0.99999988`；一步真实loss backward使全部26,208,000个`output_logits`及其余四类free variables获得有限非零梯度，
  peak allocated为29,771,734,528 bytes。纯Native loader、Action Meta module/parameter 0、38 hooks和76-tensor唯一rank16均保持。

behavior-identifiability已完成：task93/q20真实cross-episode flow-gradient rank4使policy loss从`.09911`降至`.08802`，反向升至
`.11481`；三条真实bank的optimistic signed recovery均约`.90--.91`。旧query-conditioned selector与bank-independent dual的held仅
`.0229/.0745`；同一primal经每条bank全局covariance对偶化并以同一全局measure replay后，fit/held立即为
`.9112/.9043/.9005`，三个q20 inverse仅`.734s`。因此当前单一修正是Program-primal/current-bank-global-dual，不再迭代旧scorer。

实现面正在`codex/g3-sketched-bank`替换为v5 canonical路径：新增共享Program-to-native-primal、流式全局covariance/截断谱solve与exact
signed replay；旧functional-polar不再由活动compiler编排。全仓CPU合同当前194项通过，尚未形成clean detached真实P0结果。

## 2026-08-28历史下一步与延期漂移（已被页首J2状态覆盖）

1. 完成v5 active design/config/code/diff审查，集成clean pushed authority；从detached worktree执行真实38-target K1/K4 P0，验证IEEE、
   两次native read、Y边界、chunk equivalence、全部primal梯度、Action Meta 0、uniform K、唯一rank16、policy consumption、显存和吞吐；
2. P0通过后做预注册多task、多family P1 task-local primal capacity，要求相对各自optimistic native projection无数量级损失且held-video
   保持；task93/q20单点不能直接外推；
3. P1通过后才从fresh训练P2 full-Program-to-primal shared mapping并完整评估451 conditions。两个相邻single checkpoints须满足held
   median`>=.75`、p10`>=.50`、held/fit`>=.8`，correct-vs-wrong Program同role median margin须`>=.10`；
4. G2没有learned video reliability；v5首版固定`beta=1/K`。只有F5证据需要才从uniform初始化有界correction，并防止单条video覆盖；
5. universal rank4/new-carrier只在shared selection已明显成立且残余确指向decomposition时重开；Action Meta仍按原条件式合同；
6. target当前只有fold0 manifests；在G4需要至少两个train24 folds前补齐，不阻塞G3；
7. 32-task fresh refit与71 meta+train24 development recipe冲突延迟到Final前解决，不阻塞G3--G5。
