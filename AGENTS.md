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
Semantic Direction Store正式训练、四点rollout和winner全部内部分析完成后，owner最新
要求先暂停了解推进现状；没有owner新指示前不得启动下一架构、训练、评测或GPU分析。

## Efficiency and validation boundary

- 本项目默认效率优先。不得把大量防御性校验、重复hash、全仓/全个人目录扫描、历史
  artifact复核或与当前假设无关的测试当作推进前置仪式。
- 同一不可变输入或同一实验阶段已经核验过的事实不重复核验；优先读取现成manifest、
  run contract和定向指标。只有authority首次建立/改变、正式checkpoint封存或实际
  发现身份冲突时才补做对应hash。
- 代码验证只覆盖本次改动的shape、identity/freeze、信息墙、梯度可达性、随机样本等价、
  OOM/finite和resume等真实合同；不为弱指标、科学负结果或纯理论风险新增大而泛的
  fallback、test harness或旁路实现。
- GPU启动前的必要现场检查仍保留：设备ownership/进程、显存与健康、适用storage
  quota和预计峰值；正式run仍保留config/checkpoint身份、clean代码、finite/OOM、
  exact-resume及结果完整性。这些检查每个状态边界做一次，不在轮询中反复重跑。
- 当聚焦测试、真实vertical path和正式实验能直接给出证据时，不增加额外中间层或
  “以防万一”的验证流程。发现问题按其实际层定位，修复后只重跑受影响的最短证据链。

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
30. `task_plan.md`
31. `findings.md`
32. `progress.md`
33. `docs/concept.md`
34. `docs/decisions_and_open_questions.md`
35. `docs/novelty_and_landscape.md`

本post-seal分支已完成Target-Bound裁决并打开Semantic Factor-Basis；修改或运行前
仍必须完整阅读Target-Bound与Semantic Factor-Basis两份design。历史CV/Target-Bound
实现由Git和frozen artifacts保存，不保留可执行并行版本。

旧SmolVLA、70/10/10、Phase A–F、flat task-local RL和flat Writer-RL路径已退役，
不得恢复为活动实现，也不得混入MemLLM。

## Current focused task

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
- owner要求完成rollout和全部分析后先暂停了解现状。当前不启动下一架构、训练目标、
  训练、评测或GPU分析，等待owner明确继续指示；长期Goal仍未完成。

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
漂移根因”。下一候选必须重做functional/training target与closed-loop有效流形的对应
设计；当前只记录边界，不在owner暂停期间实现或launch。

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
