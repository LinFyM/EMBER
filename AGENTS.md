# EMBER Repository Instructions

## Authority and current override

本文件和`docs/execution_brief.md`是长期实验authority。2026-08-02 19:18 UTC owner在
迁移已经由另一session启动后，重新授权约十小时A100 post-seal研究窗口：允许在既有
信息墙、split与GPU4--7边界内恢复环境、设计/实现架构、profile、训练、评测和内部
分析。必须以`f9a144c`为迁移封存基线，把全部新增Git提交和外部artifact登记为可供
迁移智能体二次同步的delta；关键代码优先push。约`2026-08-03 05:18 UTC`后不得再
启动或继续A100 GPU工作，操作上最迟`03:45 UTC`冻结新实验并完成封存。

2026-08-03，owner已明确授权在当前BCI服务器继续推进EMBER：可使用计算节点
`gpu01`和`gpu02`（owner口语中的GPU01/GPU02），每次先实时检查两节点GPU ownership、进程、利用率和显存，
只使用空闲卡，跨两节点合计最多同时使用6张。当前单卡约46GB显存，必须先profile并通过
microbatch、梯度累积、activation checkpoint、精度或必要的模型分片适配，不能直接照搬
A100 80GB配置；任何适配不得悄悄改变sealed scientific contract。不得reset、kill、pause
或干扰他人进程。owner当前还明确要求推进过程中不使用subagent，直至owner另行解除。
2026-08-03完成本轮VR正式训练、四点rollout与全部预注册分析后，owner曾要求暂停并
汇报现状；随后owner已恢复持续推进授权：保持one-shot，取消Writer参数量上限，优先
重构条件生成方向的存储与组合，并允许配套修改训练方式。仍须服从实时BCI设备边界、
最多6张空闲卡、信息墙、single-checkpoint裁决和不使用subagent的要求。效率优先，
只保留会改变实验可信度的聚焦检查，不用重复全量hash或无关旧artifact复核拖慢推进。
Semantic Direction Store正式训练、四点rollout和winner全部内部分析完成后owner曾
要求暂停；2026-08-04 owner已明确恢复长期推进，并要求后续科学问题自行深入分析，
无需为中间判断请求确认。该授权允许继续设计、实现、profile、正式训练、严格配对评测
和内部分析，但不改变上述设备、信息墙、single-checkpoint、不使用subagent与安全边界。
2026-08-05 SFT-Anchored Tangent-Basis已完成formal cycle1与strict correct400；owner确认
理解其为消融后恢复自主推进。该轴`143→142`且gained/lost=`20/21`，未过预注册续训门，
不得resume cycle2。`142`是v6 Writer warm-start继承结果，不是fresh架构成绩。下一阶段
必须回到functional identity fresh Writer的condition-to-policy根因设计；RL可在同一
健康架构上作后续校准，但不得继续用v6 warm-start替代LoRA generator重构。
2026-08-06（BCI local）owner要求当前Factorized Condition-Kernel实验与全部分析完成后
停止推进并讨论。该实验现已完整负裁决、GPU释放；在owner明确恢复前只允许只读解释、
结果核对和文档封存，不得启动reward、下一架构、profile、训练、rollout或GPU分析。

## Efficiency and validation boundary

- 2026-08-06 owner明确覆盖此前规则：后续EMBER研究、实现、profile、训练、评测、分析、
  checkpoint封存和交接均不得生成、重算、比较或门禁SHA-256、MD5或其他文件内容hash；
  即使authority改变、正式checkpoint封存或出现身份疑问也不做hash校验。历史文档中已经
  记录的hash只保留为历史原文，不复核、不扩展。Git正常commit/object ID不视为内容hash
  校验，但不得为验证身份额外计算Git blob/tree hash。
- 本项目默认效率优先。不得把大量防御性校验、全仓/全个人目录扫描、历史artifact复核或
  与当前假设无关的测试当作推进前置仪式。同一不可变输入或同一实验阶段已经核验过的事实
  不重复核验；优先读取现成manifest、run contract和定向指标，并以路径、schema、shape、
  配置字段、checkpoint可加载性、行数和真实运行结果建立必要身份与完整性证据。
- 代码验证只覆盖本次改动的shape、identity/freeze、信息墙、梯度可达性、随机样本等价、
  OOM/finite和resume等真实合同；不为弱指标、科学负结果或纯理论风险新增大而泛的
  fallback、test harness或旁路实现。
- GPU启动前的必要现场检查仍保留：设备ownership/进程、显存与健康、适用storage
  quota和预计峰值；正式run仍保留config/checkpoint身份、clean代码、finite/OOM、
  exact-resume及结果完整性。这些检查每个状态边界做一次，不在轮询中反复重跑。
- 当聚焦测试、真实vertical path和正式实验能直接给出证据时，不增加额外中间层或
  “以防万一”的验证流程。发现问题按其实际层定位，修复后只重跑受影响的最短证据链。
- 正式Writer config在磁盘上始终保留formal teacher-video seed；`--mode profile`必须
  由canonical runtime从`profile_evidence.profile_teacher_video_seed`解析有效seed并
  写入run contract，禁止为profile手工来回改正式seed。profile/formal切换后不得复用
  另一模式的checkpoint或输出root。

## Multi-GPU reliability boundary

- 多卡并行故障必须先按rank、device、process-group生命周期、collective序列、CUDA
  初始化和I/O/NUMA层定位根因；不得用盲目重试、增加timeout、关闭NCCL watchdog、
  heartbeat环境变量或减少科学batch来掩盖可复现故障。
- BCI `gpu01`/`gpu02`的A40与当前NCCL 2.28组合已复现direct P2P/CUMEM collective
  hang，稳定合同是显式`NCCL_P2P_DISABLE=1`并走SHM transport。所有BCI多卡launcher
  必须显式传入且代码fail-fast核验，不能依赖`.env.local`被偶然source；这不是通用到
  其他主机的默认设置，迁移或升级NCCL/driver后须用最小collective重新裁决。
- 耗时的rank-local CUDA模型/optimizer构造必须与NCCL生命周期分离：各rank先绑定唯一
  设备并完成本地构造，通过不依赖NCCL的all-rank ready rendezvous后才建立NCCL
  process group；不得让快rank提前创建NCCL等待仍在构造的慢rank。process group建立
  后，各rank的collective类型、shape、顺序和次数必须严格对称。
- `LOCAL_RANK`只是`CUDA_VISIBLE_DEVICES`内的进程局部序号。任何要求host物理卡号的
  外部runtime（尤其`MUJOCO_EGL_DEVICE_ID`）必须解析
  `CUDA_VISIBLE_DEVICES[LOCAL_RANK]`并把映射写入run contract；不得直接把local rank当
  物理卡号，否则非连续选卡会误触未授权或他人占用GPU。
- outcome决定的rank-local长计算（如只有mixed-reward task才做的credit反向）不得让快
  rank提前enqueue NCCL collective。每个collective阶段必须先用非NCCL all-rank ready
  rendezvous确认本地计算全部结束，再让所有rank按同序进入collective；不得用增大或关闭
  watchdog heartbeat掩盖负载不均造成的提前入场。
- credit反向的“本地计算完成”必须包含本rank显式CUDA synchronize，不能只表示Python已
  enqueue kernel。不得再用一个会被不同rank独立构造/清理的临时`FileStore`文件作这种
  outcome-skewed barrier；canonical合同是先CUDA完成，再以本次torchrun唯一session、
  cycle、epoch和rank写原子marker，观察实际world-size的全部marker后才进入NCCL。marker
  在本次run内保留，新launch必须用新session隔离失败重试，不能靠删除共享ready文件消除
  stale state。
- 多卡任务分配和结果封存必须显式读取本次实际`world_size`并为每个rank生成ownership
  记录，不得保留4卡/8卡等历史默认值后再由launcher要求另一拓扑。整任务LPT或动态
  队列必须覆盖全部rank与全部task，最终聚合必须按实际ownership推导每rank行数；卡数
  变化至少用聚焦分配/Cartesian sealing测试和真实目标规模验证，不能靠减少卡数绕过
  缺失rank。
- Direction Store内部分析已在`f82c7cd`/`a115b06`根修并实证上述合同：6 ranks必须
  生成6份ownership，8 tasks按实际LPT分配，final seal接收实际6个payload并验证
  `8 tasks × 5 conditions × refs` Cartesian覆盖。以后新增多卡分析不得复制rank数或
  每rank task数常量，必须从运行context和ownership唯一推导。
- timeout/heartbeat覆盖只允许作为一次性诊断，用来区分慢初始化与真实collective
  deadlock，不得进入canonical launcher、formal config或长期文档命令。根因修复必须
  用原始失败规模重放，并至少通过真实finite step和exact-resume边界。
- 只终止本项目本次启动且PID/设备已明确的进程；不得用模糊进程名、全用户kill或设备
  reset处理多卡错误，也不得影响共享节点上的他人进程。

当前跨session入口：

1. `docs/a100_to_bci_migration_handoff.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`

修改代码、数据、split、模型或实验状态前，主进程必须完整读到EOF：

1. `README.md`
2. `docs/a100_to_bci_migration_handoff.md`
3. `docs/active_session_handoff.md`
4. `docs/execution_brief.md`
5. `docs/action_forecast_writer_expert_consultation.md`
6. `docs/action_forecast_writer_design.md`
7. `docs/action_forecast_writer_v4_root_cause.md`
8. `docs/action_forecast_writer_v5_design.md`
9. `docs/action_forecast_writer_v5_1_proposal.md`
10. `docs/action_forecast_writer_v5_2_design.md`
11. `docs/action_forecast_writer_v5_3_design.md`
12. `docs/action_forecast_writer_v6_design.md`
13. `docs/action_forecast_writer_v7_design.md`
14. `docs/action_forecast_writer_v8_design.md`
15. `docs/action_forecast_writer_v10_design.md`
16. `docs/action_forecast_writer_loom_derivation.md`
17. `docs/action_forecast_writer_loom_design.md`
18. `docs/action_forecast_writer_recenter_design.md`
19. `docs/action_forecast_writer_core_program_design.md`
20. `docs/action_forecast_writer_prior_innovation_design.md`
21. `docs/action_forecast_writer_target_spectral_design.md`
22. historical coherent-procedure authority from commit`35fb28f`：
    `git show 35fb28f:docs/action_forecast_writer_coherent_procedure_design.md`
23. `docs/action_forecast_writer_semantic_program_grid_design.md`
24. `docs/action_forecast_writer_unified_causal_program_design.md`
25. `docs/action_forecast_writer_amplitude_preserving_dual_read_design.md`
26. `docs/action_forecast_writer_contextual_value_dual_read_design.md`
27. `docs/action_forecast_writer_semantic_factor_basis_design.md`
28. `docs/action_forecast_writer_variance_reduced_functional_estimator_design.md`
29. `docs/action_forecast_writer_semantic_direction_store_design.md`
30. `docs/action_forecast_writer_target_owned_factor_design.md`
31. `docs/action_forecast_writer_relative_flow_credit_design.md`
32. `docs/action_forecast_writer_task_grounded_progress_credit_design.md`
33. `docs/action_forecast_writer_sft_anchored_tangent_basis_design.md`
34. `docs/action_forecast_writer_policy_wide_atom_dictionary_design.md`
35. `docs/action_forecast_writer_policy_lane_hyperdecoder_design.md`
36. `docs/action_forecast_writer_antithetic_program_credit_design.md`
37. `docs/action_forecast_writer_factorized_condition_kernel_memory_design.md`
38. `docs/action_forecast_writer_fewshot_invariant_m2p_design.md`
39. `docs/action_forecast_writer_k4_layer_trace_m2p_design.md`
40. `task_plan.md`
41. `findings.md`
42. `progress.md`
43. `docs/concept.md`
44. `docs/decisions_and_open_questions.md`
45. `docs/novelty_and_landscape.md`

本post-seal分支已完成Target-Bound裁决并打开Semantic Factor-Basis；修改或运行前
仍必须完整阅读Target-Bound与Semantic Factor-Basis两份design。历史CV/Target-Bound
实现由Git和frozen artifacts保存，不保留可执行并行版本。

旧SmolVLA、70/10/10、Phase A–F、flat task-local RL和flat Writer-RL路径已退役，
不得恢复为活动实现，也不得混入MemLLM。

## Current focused task

- 2026-08-06当前唯一活动方法为
  `docs/action_forecast_writer_k4_layer_trace_m2p_design.md`：保持exact task language与K=4条
  action-hidden same-task videos联合生成一套LoRA，但不再把PI05 final hidden随机压到128维
  后让fresh通用M2P猜policy拓扑。新方法读取冻结PI05 action expert的20组all-layer trace，
  对每组形成K4×16 temporal tokens，再以layer/parameter-slot双轴M2P直接reshape完整rank16
  public LoRA。禁止language-only value bypass、逐视频LoRA平均/挑选、历史Writer warm-start、
  未经证据直接复制多expert或恢复旧store/kernel/atom/head路径；长期single-checkpoint
  strict correct400 `>150`目标不变。
- clean`a2c6d94`已原位实现上述layer-trace Writer；唯一活动config为
  `configs/pi05_as_writer_k4_layer_trace_m2p_bci_v1.json`，旧K4 config/checkpoint family只由
  Git与frozen artifacts保存，不得resume或warm-start。全仓BCI assets下`190 passed`；下一步
  只做live A40 longest105、K4/B20/B2、fresh0→1与exact-resume1→3 profile，profile通过前
  formal保持blocked。
- 首个`89f5384`一次性0→3 diagnostic不是正式profile，并因axis FFN pre-LayerNorm放大极小
  bootstrap memory而在step2/3 loss爆炸；该root禁止resume。clean`ed4f46e`已改为raw-value
  FFN并通过zero邻域幅度合同与全仓`191 passed`；必须从新root严格fresh0→1再resume1→3。
- Few-Shot Invariant-Program M2P已完成fresh0→200、四点strict rollout与macro200全部内部
  分析：correct=`70/94/99/108`、breadth=`6/6/6/7`。K4置换、zero-video identity、
  same-task/LOO/wrong/order到Program→BA→action路径和高增益LoRA均成立，证明它没有忽略视频；
  但最后50步full24 gradient retention仅`.04326`，共享Writer credit近1/24正交抵消。
  旧K4不续训、不warm-start、不用loss挑点；其正机制作为新layer-trace方法必须保留的合同。
- 先前“讨论期间暂停新实验”的边界已由上述owner授权解除。Condition-Kernel已经完成
  全部formal/rollout/internal并作负裁决，不再是可续训活动root；其结果只作为few-shot
  设计的低增益decoder与condition-credit证据。
- Factorized Condition-Kernel从generic source与functional identity fresh完成AS0→200：
  50/100/150/200 strict correct=`46/46/45/49`、breadth始终3，union/intersection=`55/40`，
  macro200未过预注册`correct≥120 && breadth≥6`；因此direct reward阶段禁止。不得挑best、
  延长FactorHeads bootstrap、resume400、调RFF seed/维度或用global scale救活。
- explicit kernel机制本身成立：200步Gram全rank24、condition=`5.139--7.750`、cap scale=1，
  macro200 predicted/observed update relative RMS=`.001304`且macro50后freeze violation=0。
  same-task feature/Program/BA差异约`.786/.783/.767`，reversed/shuffled BA约`1.39/1.36`；
  但LoRA norm仅`.176→.178`（比direct SFT小约200倍），即使stable rank约3.7、q/v B-column
  已去同向，fixed action效应仍只有`.19--.24%`。四点低换手是low-gain identity-like稳定，
  不能写成解决task drift。最早失败接口是fresh zero-B decoder在固定macro50前没有形成
  足够增益、policy-effective的Program→LoRA写出基底。精确结果取design第11节、handoff顶部
  与internal root的`experiment_analysis.json`。
- Policy-Lane已完成fresh0→200、四点strict correct400与全部内部分析并正式负裁决：correct=
  `70/63/37/61`、breadth=`6/4/6/6`、union/intersection=`117/14`。它真实形成约10个有效
  output lanes、stable rank`1.34--1.54`和direct-SFT量级跨层专门化，但same-task video
  hidden/BA能量只有约`.05%/.02%`，证明几何/容量不是充分条件。禁止resume400、warm-start、
  增加lane/store/rank、调scale或强制SFT几何。
- Policy-Wide Atom Dictionary已完成fresh0→200、四点strict correct400与all-four内部分析并
  负裁决：correct=`77/71/80/80`、breadth=`5/6/5/5`、union/intersection=`115/44`；
  64 atoms广泛使用但A/B mixing row stable rank约`1.000002`，effective LoRA约
  `1.0000002`且q/v B-column cosine约`.999998`。禁止resume到400、增加K、调scale或用
  rank/正交loss救活。
- Program-Credit canonical原位实现已完成：恢复v6唯一Writer并显式拆出
  `encode_program/decode_program`；删除Policy-Lane model/config/checkpoint family、旧
  Flow-Credit数学owner和progress diagnostic。runtime只保留两对antithetic program扰动、
  binary-first pair credit与每cycle一次direct program backward；ledger/exact-resume绑定
  actual world size、direction seed、四个artifact cursor、pair randomness和credit identity。
  首次clean`318b6f6`六卡profile在68/96 rollout、0 update处发现LIBERO同一hard-reset env
  无法仅凭同seed恢复同一随机初态。canonical根修是每task两条lockstep persistent lanes：
  plus固定lane0、minus固定lane1，两lane共享reset index/seed/policy noise且从不读取固定init
  state；真实三轮XML/state/双相机逐字节复现已通过，v2 ledger绑定lane identity。失败root
  禁止resume。clean`f3f6b15`全新六卡v2 profile已完成cycle0→1及exact-resume1→2：两轮
  各96 rollout/24 credits/48 valid CRN pairs/54 successes，一次finite update，四上游block
  可达且冻结梯度0，wall约431秒、峰值约19.33GB、0错误。profile权重永久弃用。fresh formal
  cycle1已完成96 rollout、54 successes、6 binary-discordant pairs和一次finite update；strict
  correct400=`106`、breadth5，相对AS125=`97/5` gained/lost=`18/9`、union/intersection=
  `115/88`。三suite改善但Spatial task1→task3换手，净增9未过预注册净增10门；禁止resume
  cycle2/4/8。内部分析随后已完成48/48 rows：exact task cotangent pair cosine mean/median=
  `.000107/0`、full24 retention=`.041874`，共享Writer更新后的program delta却变成
  `.5801/.6128`、retention=`.55537`且0负pair；same-task program/BA更新task-mean energy
  fraction=`.82990/.91623`。binary cotangent energy约为semantic tie-break的`72.7×`，held
  gained/lost LoRA变化不可区分。Program-Credit由此正式负裁决，旧root禁止resume，当前
  最早接口是共享condition-map把不同closed-loop credit压成task-common/video-insensitive
  更新；不得用rank、scale、head/store扩容或续旧RL回避。
- SFT-Anchored Policy Tangent-Basis authority
  `docs/action_forecast_writer_sft_anchored_tangent_basis_design.md`已完成formal cycle1并负
  裁决：历史v6-fast macro400=`143` warm-start经固定8个factor-output basis的reward更新后
  为`142`，breadth`6→7`但gained/lost=`20/21`、union/intersection=`163/122`。因此禁止
  resume cycle2，也不补跑五臂把消融包装成主方法。
- AS125 Task-Grounded Semantic Progress Credit已完成同一formal root cycle1/2及两轮strict
  correct400。AS125/cycle1/cycle2=`97/104/102`，breadth=`5/4/4`；cycle1→2
  gained/lost=`15/17`，Object-1`31→26`、Object-3`19→22`且无新task coverage，
  因而同recipe续训轴正式负裁决，禁止resume cycle4/8。完整结果取
  `docs/action_forecast_writer_task_grounded_progress_credit_design.md`第16节与cycle2
  eval root的`paired_to_cycle1_and_as125_analysis.json`。随后固定参数hybrid已完成：
  effective BA由upstream贡献更多，但fixed action由factor-output贡献更多，且随suite反转；
  完整证据取同design第18节。该结果已打开上条新方法，旧root仍禁止resume cycle4/8。
- Task-Relative Flow-Credit Writer的binary-only阶段已按
  `docs/action_forecast_writer_relative_flow_credit_design.md`完成并负裁决：它恢复历史
  single-checkpoint最强且时序路径已验证的v6 Writer做独立AS cold start，随后永久关闭
  teacher action入口，但24 train tasks的official random-reset binary reward与同task
  K4 leave-one-out仍无法覆盖全任务。随后method authority切换为
  `docs/action_forecast_writer_task_grounded_progress_credit_design.md`：先用冻结AS125
  semantic encoder从task language、action-hidden teacher首尾内容变化与rollout自身
  首尾agentview构造start-relative semantic potential，完成预注册只读机制门后才决定
  是否训练；该方法现已按第16节完成并负裁决。不得恢复旧success-filtered Writer-RL、
  flat task-local RL、Target-Owned或Direction Store活动路径。
- canonical实现已原位替换旧RL路径：success与failure executed prefixes均保留；正
  advantage用PPO clip，负advantage用SPO pullback；Nmc4、full24等权、最多6 ranks、
  deferred NCCL和完整cycle exact-resume。source policy与normalization冻结，Writer输入
  仍只有task language + exactly one action-hidden teacher video。
- v6 AS A40六卡profile已通过：logical B20、policy B2、16-frame encoder chunk、最长
  105帧；三步约`33.46/30.89/30.98s`，峰值allocated/reserved
  `34,948,858,880/44,816,138,240` bytes，0 OOM/clip。独立fresh0→1再exact-resume1→3
  通过，累计1,440 queries/72 videos，五个主block到step3均可达，source trainable=0。
- sealed AS config是`configs/pi05_as_writer_v6_relative_flow_coldstart_bci_v1.json`；同一
  fresh root已exact-resume到macro125，共60,000 queries/3,000 videos、125个finite
  full24 macros，0 OOM/clip和0 validation/test action reads。step100→125首次启动因
  漏传sealed CLI `--num-workers 0`在step101前被contract拒绝且没有写入训练状态；补全
  exact命令后原root正常完成。这不是代码、namespace或checkpoint身份故障。
- step25/50/75/100/125的K4 success=`25/38/47/52/50`，coverage=`12/14/18/17/19`。
  step125为14 mixed、5 all-success、5 all-failure；相对step100严格配对
  gained/lost=`10/12`，新增覆盖task5/29且没有旧coverage完全掉出。长期全失败task
  36/38/39五点均为0/4，task4与20又在早期成功后晚期归零。24-task binary-credit门仍
  未过，binary-only正式RL不得启动，也不得从任一reward-profile checkpoint继续。
- 上述四点内部审计已在clean`2b775f0`完成96/96 Cartesian rows：effective LoRA norm
  中位数=`53.40/80.37/93.17/99.18`，stable rank=`1.000028/1.000055/1.000153/
  1.000176`，same-task五video centered/sample energy=`.0813%/.1309%/.1333%/.1300%`。
  rank塌缩只是历史v6结论的当前复核；direct SFT约1.52和Target-Spectral负结果仍禁止
  强制升rank。step100全失败tasks的same-video BA差异中位数反而高于有success tasks
  (`.0634>.0405`)，排除“只需放大视频差异”；reversed/shuffled到fixed action中位
  relative-L2=`.0498/.0474`又证明时序路径未断。
- AS checkpoint BA/action相邻中位churn从50的`1.116/.187`降至100的`.608/.142`，仍有
  大量能力轮换。新增step100/125审计显示norm中位`99.18→109.11`，但same-task video
  energy`.1300%→.1154%`、demo1 BA差异`.0475→.0448`、fixed-action demo差异
  `.0101→.0086`均未增强；BA/action churn仍为`.536/.138`。24-task成功数变化与video
  energy变化Spearman=`-.521,p=.0090`，与BA churn=`.416,p=.0430`。持续全失败组的
  video变化和demo间BA差异反而更大，正式否定“继续放大条件差异即可获得正确LoRA”。
- step125 binary profile完整收集96 rollout/24,600 actions并完成两轮finite credit：
  ratio范围=`[.9871,1.0124]`与`[.7646,1.1015]`、grad norm=`.0361/.0531`、0 clip，
  peak reserved=`40,338,718,720` bytes，完整cycle1 checkpoint与0 watchdog。该健康
  profile只证明实现，不改变coverage负裁决。下一活动设计必须让全失败轨迹也获得
  teacher-video内容约束的相对credit，同时避免恢复normalized-video-time时钟、target
  action、privileged state或LIBERO特化reward；binary success保留最高优先级。
- Task-Grounded Semantic Progress Credit只读机制裁决已在clean`c483497`、AS125与严格
  配对K4上通过全部预注册门：50/96 successes、14 mixed、5 all-success、5 all-failure，
  mixed agreement=`13/14`、同task success/failure AUC=`.8913`；五个all-failure utility
  range=`.1228/.5712/.3338/.2554/.2371`。successful rollout的correct分别优于wrong/
  shuffled/reversed比例=`1/.88/1`，对应margin中位=`.4889/.3557/1.6208`；all-failure
  utility与pixel-change Spearman=`.5564`。96/96旧profile身份与outcome严格一致，0
  optimizer/backward/checkpoint，不能把该证据冒充Writer性能改善。
- clean`84d856c`的fresh Writer-update profile已通过：96 rollouts/24,593 actions、50
  successes、14 mixed、5 all-success、5 all-failure；5/5 all-failure task均有nonzero
  generated-LoRA gradient，五个Writer下游block两epoch均可达，observer gradient=0。
  ratio=`[.99077,1.02504]`/`[.74545,1.09294]`、grad=`.03715/.05521`、clip0、peak
  reserved`19,455,279,104` bytes，0 watchdog/OOM。profile checkpoint永久禁止续训。
- 历史Task-Grounded formal曾封存为从AS125 fresh进入的6-rank、K4/Nmc4、two-epoch、最多8 cycles，checkpoint
  `1/2/4/8`；首段只跑0→1，然后在同一strict panel比较AS125 baseline与cycle1 paired
  correct400再决定续段。首次fresh0→1完整产生96 rollout和24 task credit，但旧
  `FileStore` barrier在第一轮gradient sum发生rank序列分裂：rank0/1/2/5进入seq18，
  rank3/4停在seq17，600秒watchdog终止；没有update、metrics或checkpoint，失败root禁止
  resume/评测。根因不是NCCL transport、OOM或科学结果，而是ready只覆盖CPU enqueue且
  临时FileStore生命周期不能可靠证明所有rank的CUDA工作已结束。
- canonical修复改为每rank先CUDA synchronize，再按本次torchrun唯一session/cycle/epoch
  写原子rank marker，6/6可见后才进入NCCL；marker不在run内竞态删除。相同输出目录连续
  两个新torchrun session的六卡真实探针都完成6/6 marker和all-reduce sum21，旧session
  marker没有污染重启。clean/pushed`30977b5`已在全新retry1 root重放原96-rollout/
  two-epoch规模：两轮分别形成6/6 marker后才进入NCCL，完成2次finite update、完整
  cycle1 checkpoint、0 watchdog/OOM，peak reserved`19,455,279,104` bytes。5/5
  all-failure task均有LoRA梯度，五个下游block可达且observer grad0；六rank各16 rollout/
  4 progress-credit双ledger通过checkpoint validator。profile或失败root权重仍禁止进入。
- AS125 baseline与formal cycle1已在同一strict correct400 panel完成：`97→104`，
  gained/lost=`22/15`，breadth=`5→4`；净增7几乎全部来自Object-1的`24→31`，Spatial-1
  丢失唯一成功。400对effective BA的变化中位仅`.01677`、余弦`.999860`；AS/cycle1
  stable rank中位均约`1.000017`，B-column cosine均约`.99884`，因此cycle1主要是在
  既有近rank1方向上做小幅task-dependent调节，没有解决LoRA几何或task drift。由于仅
  完成2次full24 optimizer update、held aggregate未下降且19/24 train tasks有credit，
  当时只授权同一formal root exact-resume `1→2`后重跑同一correct400；该动作现已完成并
  由本节首条的cycle2负裁决覆盖，不得把这条历史授权解释成继续4/8。
- AS50→75首次resume因所选物理卡对应rank形成`4+2` NUMA分布，而root已封存`3+3`
  topology，被resume contract在模型训练前正确拒绝；无metrics或checkpoint写入。随后在
  同一节点改用满足原`3+3` rank topology的六张空闲卡，原命令完成step75。正式
  exact-resume必须保存sealed rank/NUMA topology；live选卡只能在该边界内更换空闲物理卡。
- 长期目标仍是同一single checkpoint strict correct`>150/400`且越高越好；未完成前
  不因loss、训练reward、内部几何或单一screen停止。当前GPU无EMBER进程；每次launch
  仍实时比较`gpu01/gpu02`并只用最多6张空闲卡。

- Policy-Target-Owned Factor已完成正式负裁决。clean`34be4a0`从fresh identity完成
  macro0→200；50/100/150/200严格配对correct400=`99/76/86/68`，breadth=
  `6/6/7/5`，union/intersection=`136/37`。winner macro50=99低于Direction Store129、
  v6-fast143和严格门151；不得续400或从任一checkpoint warm-start下一方法。
- macro50五条件内部分析证明76个完整heads真实解除跨policy-target硬共享：q/v跨层
  effective-BA cosine从Direction Store`.932/.967`降到约0。但LoRA norm均值仅
  `19.03`，q/v top-4能量占`.733/.853`，比direct SFT更过度集中；Program差异扩大到
  BA后没有形成等比例action变化。因此不得把漂亮的异质LoRA几何写成方法有效。
- 200步内部梯度显示factor占单task能量`69.25%`，24-task median cosine`.0040`、
  负pair`.4457`、full24能量保留`.0484`；相同task+demo的方向不稳定重现。正式拒绝
  policy-target parameter sharing作为task drift主要根因。当前最早接口是视频条件如何
  获得policy-aware、闭环有效、跨随机query可累积的credit，而不是继续增加head、调
  layer gate/scale、强制SFT profile或加入监督专用辅助loss。
- 本轮训练、四点rollout和全部分析已结束，GPU已释放。该次暂停随后已由owner解除；
  Target-Owned只保留为负结果，不再是活动实现。长期single-checkpoint correct严格
  超过150仍未完成。精确
  roots、曲线、几何和禁调项取`docs/active_session_handoff.md`与
  `docs/action_forecast_writer_target_owned_factor_design.md`。

- Semantic Direction Store已从clean`91feeef`完成fresh0→200与四点严格配对
  correct400：`129/107/120/129`，breadth=`7/7/7/5`。macro50与200同分但前者breadth
  更高，single winner选macro50=`129`；仍低于v6-fast143和严格目标`>150`，不续400。
- 四点success union/intersection=`174/65`、single envelope gap45，相邻gained/lost
  `17/39,43/30,27/18`，checkpoint能力轮换未解。它相对SFB macro50提高60，说明完整
  独立stores改善早期acquisition，但不能写成解决task drift。
- macro50 refs1确认固定语义route、A/E/Core/Program到BA/action路径都工作；然而
  same-task Program relative-L2 `.93377`到factor/BA只剩`.01935/.03242`。全部16个
  rank坐标虽active，effective LoRA stable rank仅`1.000043`、首奇异值能量`.999957`、
  B-column cosine`.999971`。当前最早结构瓶颈是Program→public A/B的多维方向形成，
  不是继续增加store、调K/route或放大scalar。
- 六卡内部分析的assignment与final sealing已由`f82c7cd`/`a115b06`绑定实际
  `world_size`并在真实6-rank、8-task、5-condition规模通过；多卡长期规则见上文。
- Direction Store阶段的暂停已由2026-08-04新授权解除；上述结果只作负裁决和新设计
  证据，不能恢复其active router/store或checkpoint。

- Semantic Factor-Basis完整correct400曲线为
  `69/91/118/127/117/81/126/120`；single winner仍是macro200=`127`，八点
  success union=`193`、single envelope gap=`66`，没有解决漂移或超过v6。
- variance-reduced functional estimator已从fresh identity在BCI完成macro0→200；
  correct400曲线为`76/88/126/107`，breadth=`7/4/7/5`，single winner是
  macro150=`126`。四点success union/intersection=`158/49`、single envelope
  gap=`32`；150→200 gained/lost=`21/40`。它既没有超过SFB winner`127`、v6-fast
  winner`143`或严格门`>150`，也没有解决checkpoint能力轮换，正式裁决为负。
- 有效训练commit为`d9130c9`：200 finite macros、96,000 logical queries、4,800
  one-video conditions、8个checkpoint、0 clip、validation/test action reads=0；
  64/64 checkpoint payload hash通过。VR相对普通SFB的全200步same-task CountSketch
  cosine只提高`.00263`（factor`.00510`），raw mean/sample energy retention只提高
  `.00191`，且分阶段反复，不是material稳定化。
- macro200 held functional loss由SFB`.13178`改善为VR`.12915`，paired correct却
  `127→107`；这再次证明functional loss不能选择closed-loop checkpoint，但不能单独
  解释task漂移。当时由此打开shared factor parameter-coexistence假设；该假设随后已由
  Direction Store正式实验作出上文所述部分支持但总体负裁决。
- owner已取消Writer参数量软上限并维持one-shot。最近裁决的canonical候选为
  `docs/action_forecast_writer_semantic_direction_store_design.md`：冻结task-language
  语义地址固定选择top2完整容量direction stores，完整Core/A/E/D只作为value；目标是
  让不同task方向拥有独立参数存储并可由language组合。实现、profile、formal训练、
  四点rollout和winner内部分析均已完成并按上述结果负裁决。
- 旧A100“只使用物理GPU4--7”的边界已退役；当前只按上文BCI设备授权使用
  `gpu01`/`gpu02`的实时空闲卡，跨节点合计最多6张。
- 迁移封存基线是`f9a144c94e71bb44373d7247ed0fded2ed835305`；当前BCI写分支为
  `codex/bci-continuation`。A100 post-seal ledger只作迁移provenance；BCI新增runtime
  roots统一留在本项目`runs/`，迁移/验收证据留在`evidence/`，不得写回旧
  `/data/ymdai`。
- frozen source step1000仍是下游inference/source asset，不支持source-SFT exact
  resume。A100 Codex、venv、cache与worktree仍不迁移。
- A100窗口的GPU工作已停止；BCI迁移、46GB适配、VR与Direction Store的训练、rollout
  和分析均已完成。当前执行边界重新收敛为owner要求的结果后暂停，长期Goal仍未完成。
- 当前最低科研目标是同一single checkpoint的paired correct aggregate严格超过
  `150/400`，并在达到后继续追求更高absolute、breadth和视频因果性；不得用多
  checkpoint、挑video或违反信息墙的方法过门。

## Long-term objective

从generic`lerobot/pi05_base`出发，在过滤后的LIBERO-90 source tasks上训练并冻结
共享π0.5-LIBERO source base；随后在固定24 train / 8 validation / 8 test开发split
上完成AS-Writer、RL-Writer、Source-SFT、视频因果对照与方法选择，最终合并32 source
并做zero-interaction test、test-task local RL和privileged direct-action oracle。

focused AS-Writer追求的是同一single checkpoint同时具备：

- 高绝对closed-loop性能；
- 多task breadth与低checkpoint能力轮换；
- teacher-video语义与时序因果性；
- same-task跨video鲁棒性；
- Core/Procedure到effective LoRA再到policy action的有效传递；
- coherent、高增益、闭环有效而非形式漂亮的LoRA几何。

达到某个aggregate不自动停止；训练loss、smoke、单一checkpoint或漂亮五臂margin也
不能单独宣告成功。

## Current scientific boundary

正式四格五臂：

| architecture × recipe | correct | same | wrong | shuffled | reversed |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5.2 old | 132 | 138 | 74 | 82 | 83 |
| v5.2 task-complete | 120 | 109 | 107 | 111 | 124 |
| v6 old | 121 | 122 | 111 | 84 | 47 |
| v6 fast task-complete | 143 | 135 | 125 | 128 | 129 |

task-complete在v5.2与v6上都压弱Procedure→BA/action和顺序margin，但absolute分别
下降和上升；旧recipe增强动态写出也增强task旋转。post-v5各架构与recipe混杂，不能
因低分整体判死，也不能退回旧recipe。

最新CV-ADR RAW correct400完整曲线为
`76/111/99/117/77/69/80/82`；GROUP4为`82/77/73/110`。两者都未解决漂移，
GROUP4使写入更大、更coherent但更static。matched gradient分析中video主效应约
`.1%`，query/flow噪声主导，functional loss与closed-loop发生明确错位。详细数值、
artifact roots和hash只取`docs/active_session_handoff.md`。

Target-Bound correct曲线`75/120/90/110`仍漂移，但内部remove-A/remove-D/
memory-reversal均`8/8` tasks达门，说明动态路径已工作而shared factor共存仍失败。
Semantic Factor-Basis完整曲线`69/91/118/127/117/81/126/120`较Target-Bound形成
更可信共同累积，但相邻checkpoint仍大量换手且晚期CountSketch梯度稳定性没有改善。
variance-reduced estimator只改变exact-Beta Latin time与随机antithetic Gaussian
noise，保持架构、objective期望、B20、full24 raw mean和optimizer不变；其正式曲线
`76/88/126/107`与微小、非持续的机制改善共同否定“可约flow Monte Carlo方差是主要
漂移根因”。Direction Store曲线`129/107/120/129`改善early acquisition但未解漂移。
Target-Owned Factor又以`99/76/86/68`证明：解除policy-target硬共享可修复跨层同向
几何，却不能让条件差异落入闭环有效方向。下一候选必须重做condition-to-policy
credit与closed-loop有效流形的对应设计；当前只记录边界，不在owner暂停期间实现或
launch。

## Data and split

- 目标benchmark为`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，
  共40 tasks。
- development split封存在`configs/libero_24_8_8_v1/`：每suite 6 train / 2
  validation / 2 test，总计24/8/8；不得按outcome改task IDs。
- validation选定方法后才合并为32 source / 8 test并从规定初态重训。
- LIBERO-90与目标40的3,600-pair specification-only audit已封存：排除19个exact
  semantic/composition重合source tasks，保留71 active tasks；不得按结果重开。
- source base使用每个active source task全部50条成功teacher episodes。不得使用
  已读过目标40 actions的`pi05_libero`。
- normalization只从过滤后的LIBERO-90 source actions/states计算并冻结，所有下游
  方法共享；validation/test不得重算。

## Information wall and Writer contract

- Writer输入固定为task language + exactly one action-hidden teacher video。
- Writer不得读取teacher action、proprio、reward、terminal、task ID、filename、
  object pose或隐藏normalization；source actions只进入AS functional loss。
- 每task每次读取一条teacher video并生成一套完整rank-16 public LoRA；不做多video
  平均、多LoRA平均、checkpoint融合或第二套LoRA。
- frame stride固定5；保持Q/M/G、Action probe、Core/Program及真实38-target public
  topology的信息墙。任何改变必须有新的设计authority。
- template A/zero B保证step0 functional identity；frozen source policy不得有
  trainable parameters。

## AS training contract

- development在24 train tasks上做task-complete宏步；每task一条video、一套LoRA、
  B20条同task跨episode独立action queries，先task内mean再24-task等权。
- 每macro/cycle的optimizer语义由该架构sealed config决定；一次-Adam full24、
  SERIAL/GROUP4等不能互相冒充。无冲突时的projected更新必须严格退化为raw mean。
- 不读取validation/test actions产生梯度；video与action query不得用same-episode
  pairing制造低层捷径。
- 一小时门：fresh约0→200、每25保存，paired correct400评测50/100/150/200；只有
  absolute/breadth/趋势/内部路径有充分理由才exact-resume第二小时。
- checkpoint选择只认single-checkpoint paired closed-loop结果；functional loss和
  内部几何只作机制证据。

失败后先定位evidence extraction、Core、Procedure、fusion/compiler、optimizer或
functional surrogate的最早失效接口。禁止用scalar gate、全局scale、B-only residual、
static bypass、confidence、强制正交/rank diversity、multi-video或checkpoint融合救
一个失败checkpoint。

## Baselines and later stages

- frozen source base、AS-Writer、Source-SFT及后续RL路线共享同一source policy、
  normalization和policy接口。
- corrected mixed-task Source-SFT使用shared rank-128 LoRA，observed-best`109/400`；
  其参数预算约束Writer但不要求机械相同steps/examples。
- RL-Writer若恢复，必须从架构规定的短、task-balanced AS cold start开始，之后关闭
  action入口做纯reward；不得从完整AS best继续，也不得用`.pruned_init`训练。
- task-local RL、final32、test、joint oracle和ViVLA均不因focused Writer结果自动
  获得launch authority。
- direct target-action oracle是privileged shared multi-task reference，不是同信息墙
  baseline，且必须在其他方法结果封存后进行。

## Evaluation and video causality

- official π0.5/LIBERO preprocessing保持：render256、model224、两相机180° rotate、
  state/action 7维、10 flow steps、执行前5 actions后replan、dummy settling10、成功即
  终止、suite horizons 220/280/300/520。
- zero-interaction evaluation每rollout从正确task的50条teacher videos无放回取一条；
  不挑最好video。
- correct/same-task-other/cross-suite-wrong/shuffled/reversed必须严格配对state、policy
  RNG、video ordinal等字段；shuffled/reversed需对真实输入帧重排后完整forward。
- evaluator使用cost-balanced dynamic queue、long-first和persistent model/env；不用
  静态task/GPU分配或dummy显存占用。
- 报告aggregate、per-task、per-suite、gained/lost、breadth、能力轮换和内部
  Core→Program→effective BA→fixed-action传递。raw A/B gauge符号不是跨模型正式结论。

## Host, paths and GPU

- A100时期的“只用物理GPU4–7”不适用于BCI。当前owner设备边界为
  `gpu01`与`gpu02`的实时空闲卡、跨节点合计最多6张；每次运行前仍必须
  live检查GPU ownership、telemetry、进程、CUDA、storage和峰值预算。
- 不得reset、kill、pause或干扰他人进程；共享设备只在owner明确授权时使用。
- BCI不得依赖A100绝对路径或500GB旧cap。设置`EMBER_STORAGE_ROOT`与owner给出的
  `EMBER_STORAGE_CAP_BYTES`供preflight容量检查；source、checkpoint、tokenizer、
  data、output继续通过CLI显式传入。
- `hf-libero` simulation assets是当前runtime依赖；迁移精确
  `lerobot/libero-assets@0b3ea86...`或在BCI按同revision重下，不能保留指向A100的
  site-packages绝对symlink；用`EMBER_LIBERO_ASSETS_ROOT`指向BCI snapshot。
- 历史config、run contract和analysis中的`/data/ymdai`是provenance，不批量改写。
- 具体BCI数据盘映射、rsync staging和MemLLM symlink看迁移handoff。

## Checkpoint, artifacts and evidence

- checkpoint必须含Writer/model、optimizer、scheduler/scaler、sampler/cursor、每rank/
  worker RNG和完整schema；fresh incompatible架构不得误载旧checkpoint。
- smoke只证明load、shape、freeze、gradient、OOM、resume和环境，不解释性能。
- formal结果必须保留run contract、checkpoint manifest、metrics、raw rows、aggregate、
  completion和必要analysis；screen/profile/smoke不得冒充formal。
- 不比较未严格配对的不同state/video/RNG panel，不把不同估计器百分比写成严格倍数。
- 原迁移封存保留60个正式checkpoint roots；post-seal又新增Target-Bound与SFB两个
  正式训练root和12个正式correct400 roots。它们共同构成训练漂移证据，迁移时不得
  只留winner。
- cleanup删除清单位于A100
  `/data/ymdai/migration_manifests/a100_cleanup_20260802`；历史文档中的已删profile路径
  不表示正式artifact损坏。

## Engineering and Git

- 一个canonical Writer path；替换行为时旧实现由Git/frozen artifact保存，不保留
  executable parallel version。
- 先检查现有owner和contract，再新增模块、抽象、runner或fallback。非平凡结构变化
  使用code-architecture-gate。
- 保持main clean、任务diff聚焦；不得提交checkpoint、cache、dataset或大binary。
- 正式run需frozen worktree；并发写实现需独立worktree；不得让两个writer重叠写。
- meaningful状态更新`docs/active_session_handoff.md`、`docs/execution_brief.md`、对应
  design、`task_plan.md`、`findings.md`、`progress.md`并commit/push。
- A100 pre-cleanup全refs bundle只作灾难恢复；BCI默认从GitHub clone，不批量恢复
  Codex refs或旧local branches。

## Migration verification

迁移智能体必须按`docs/a100_to_bci_migration_handoff.md`完成：Git/bundle hashes、
source policy/tokenizer hashes、formal manifests、MemLLM hashes、路径/symlink、环境
重建和CPU tests。不得复制venv或Codex auth。迁移成功后先向owner报告，再等待新的
实验authority。
