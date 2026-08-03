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
29. `task_plan.md`
30. `findings.md`
31. `progress.md`
32. `docs/concept.md`
33. `docs/decisions_and_open_questions.md`
34. `docs/novelty_and_landscape.md`

本post-seal分支已完成Target-Bound裁决并打开Semantic Factor-Basis；修改或运行前
仍必须完整阅读Target-Bound与Semantic Factor-Basis两份design。历史CV/Target-Bound
实现由Git和frozen artifacts保存，不保留可执行并行版本。

旧SmolVLA、70/10/10、Phase A–F、flat task-local RL和flat Writer-RL路径已退役，
不得恢复为活动实现，也不得混入MemLLM。

## Current focused task

- Semantic Factor-Basis完整correct400曲线为
  `69/91/118/127/117/81/126/120`；single winner仍是macro200=`127`，八点
  success union=`193`、single envelope gap=`66`，没有解决漂移或超过v6。
- variance-reduced functional estimator的实现与mode接线修复已push到`50662a8`；
  longest105 B20与formal-seed fresh0→1/exact-resume1→3通过。matched前三步只给出
  小幅正机制证据，尚无0→200训练或closed-loop结果，不得写成有效方法。BCI上已用
  6 ranks×4 tasks、逻辑B20、policy microbatch2完成未冻结工程profile与
  fresh0→1/exact-resume1→3；必须在代码提交后重放，不能把dirty-worktree结果当作
  formal seal。
- 旧A100“只使用物理GPU4--7”的边界已退役；当前只按上文BCI设备授权使用
  `gpu01`/`gpu02`的实时空闲卡，跨节点合计最多6张。
- 迁移封存基线是`f9a144c94e71bb44373d7247ed0fded2ed835305`；当前BCI写分支为
  `codex/bci-continuation`。A100 post-seal ledger只作迁移provenance；BCI新增runtime
  roots统一留在本项目`runs/`，迁移/验收证据留在`evidence/`，不得写回旧
  `/data/ymdai`。
- frozen source step1000仍是下游inference/source asset，不支持source-SFT exact
  resume。A100 Codex、venv、cache与worktree仍不迁移。
- A100窗口的GPU工作已停止；BCI上的环境恢复、profile、训练和评测已获owner授权。
  迁移与46GB适配验证通过后，可继续VR fresh0→200及后续有证据支持的实验。
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
下一候选只改变functional Monte Carlo estimator：
exact-Beta Latin time加随机antithetic Gaussian noise；架构、objective期望、B20、
full24 raw mean和optimizer保持不变。

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
