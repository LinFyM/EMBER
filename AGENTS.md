# EMBER Repository Instructions

## Authority

本文件和 `docs/execution_brief.md` 是当前活动 authority。2026-07-21 generic π0.5 feasibility 已结束；其后 owner 明确批准继续完整 EMBER 主线，并以本文件记录的共享 π0.5-LIBERO source base、one-video Writer 和 test-task training 口径替换此前“结果后停止”的临时边界。

修改代码、数据、split、模型或实验状态前，完整阅读：

1. `README.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_expert_consultation.md`
5. `docs/action_forecast_writer_design.md`
6. `docs/action_forecast_writer_v4_root_cause.md`
7. `docs/action_forecast_writer_v5_design.md`
8. `docs/action_forecast_writer_v5_1_proposal.md`
9. `docs/action_forecast_writer_v5_2_design.md`
10. `docs/action_forecast_writer_v5_3_design.md`
11. `docs/action_forecast_writer_v6_design.md`
12. `docs/action_forecast_writer_v7_design.md`
13. `docs/action_forecast_writer_v8_design.md`
14. `docs/action_forecast_writer_v10_design.md`
15. `docs/action_forecast_writer_loom_derivation.md`
16. `docs/action_forecast_writer_loom_design.md`
17. `docs/action_forecast_writer_recenter_design.md`
18. `task_plan.md`
19. `findings.md`
20. `progress.md`
21. `docs/concept.md`
22. `docs/decisions_and_open_questions.md`
23. `docs/novelty_and_landscape.md`

`docs/active_session_handoff.md`是当前跨session恢复入口，集中摘要研究证据链、
v5失败证据、v5.1设计理由、运行状态和下一动作，但不覆盖架构或长期科学
authority；focused AS/RL完成或其不再承担跨session恢复作用时应更新或删除。

旧 `docs/expert_plan.md`、SmolVLA/70-10-10 runner/config/checkpoint 和
Phase A–F 可执行路径已从工作树退役，只由 Git 历史保存 provenance；不得恢复为
活动路径，不得依赖或混入 MemLLM。

## Active objective

以 generic `lerobot/pi05_base` 为起点，先在与目标 LIBERO-40 specification 无 exact semantic/composition 重合的 LIBERO-90 source tasks 上做联合 action-SFT，得到并冻结一个共享、多任务、语言条件的 π0.5-LIBERO source base；随后在固定 24 train / 8 validation / 8 test 目标 split 上完成 AS-Writer、RL-Writer、Source-SFT、seen/wrong-video 机制对照、合并 32 source 后的单 seed 重训和 zero-interaction test；再直接在 8 个 test tasks 上把三种 task-local LoRA RL initialization 训练到各自最佳，最后用 8 个 test tasks × 50 action episodes 联合训练一个 privileged shared-LoRA oracle。ViVLA-style matched baseline 和 source-only outer learning 只在核心闭环之后有时间再做。

任何单一 source base、训练 loss、smoke、局部 seen 结果或一个 Writer 阶段都不能单独触发长期 Goal complete。

## Current focused execution task

owner于2026-07-30在Loom首段correct400仅为`79/106/105/112`且内部
correspondence/confidence/gap缺少可靠锚点后，授权从第一性原理继续重设计和
迭代。当前唯一canonical authority改为
[`docs/action_forecast_writer_recenter_design.md`](docs/action_forecast_writer_recenter_design.md)；
Loom推导与设计只作失败证据和provenance。

Recenter保留稳定text query、task-grounded patch evidence和v6 Semantic Core；
恢复原生50-token suffix mean作为policy-native Action主干，只允许
task-grounded视觉transition以Action RMS四分之一为径向上限作残差修正。单路
causal Procedure进入新的Core-keyed compiler：Core先提供slot地址，value只读
raw time-centered Procedure；Core仅作`[0.75,1.25]`乘性调制，slot mixer在
混合方向后恢复输入RMS。constant/zero Procedure无论Core为何均产生identity，
且不存在terminal normalization、Teacher/Policy gap、confidence、raw patch
correspondence或Action/Effect局部配对。canonical参数量为`10,709,248`。

Recenter继续使用v6封存的task-complete fast-decay训练合同：GPU4–7、4 ranks、
每rank long-first处理6 tasks、每task一条video和B20独立action queries、
每macro覆盖24 tasks并只做一次AdamW update。Loom的B20/profile/resume证据
不能继承；当前Recenter配置保持pending，必须先用真实105-frame视频完成B20
三macro profile和fresh1→resume3，再seal正式首段macro0→200。B20失败时只
回退B16。训练完成后固定测试macro50/100/150/200，不做checkpoint融合。

v4、v5、v5.1、v5.2和v6均已完成所需根因或上限证据；旧架构与可执行配置只作
provenance。v5.2 step900五臂
`correct/same/wrong/shuffled/reversed=132/138/74/82/83`，证明Semantic Core
与Causal Procedure可以同时通过视频语义和顺序行为门，但absolute不够。

v6 task-complete fast-decay的single-checkpoint best为macro400=`143/400`，
五臂`143/135/125/128/129`；续到macro600后显著下降。相同v6拓扑改用旧
rank-rotating recipe时best为`121/400`，五臂`121/122/111/84/47`：顺序门显著
增强，但absolute、breadth和wrong-video语义门下降。该对照证明训练粒度能调节
Procedure→compiler增益，但简单恢复旧recipe不是答案。corrected mixed-task
rank-128 Source-SFT的development best已封存为`109/400`；full-24与global-8
都出现task能力漂移，后者没有提高上限。

owner于2026-07-29按第一性原理批准的v7已经完成fresh macro0→400和机制
检查。correct400为`82/106/114/120/101/114/115/106`
（macro50→400），macro200五臂为`120/112/91/100/69`。它的顺序特异性强于
v6，但absolute明显下降。内部证据显示`8×L` joint attention的熵达到理论均匀
熵约`99.96%`、有效Action probes约`7.998/8`，没有形成选择性Action–Effect
binding；同时fixed-Procedure只改变Core时effective-LoRA差异仅约
`0.1–0.2%`，模型实际退化成Procedure-only。macro400仍未修复且性能下降，
因此v7停止，不再续训。

v8 Hierarchical Action–Effect + Core-Gated Procedure也已完成fresh
macro0→400和机制检查。correct400八点最高为macro300=`125/400`，五臂
`125/121/110/110/117`，没有达到v6 absolute，也只保留较弱视频margin。内部
检查显示固定Effect改变Action时event L2仅约`8–10%`，固定Action改变Effect时
约`147–300%`，EventRead熵约为理论均匀熵的`99.67%`。因此严格局部
Action–Effect binding缺少信息墙内可识别依据：Action Expert probe只是冻结
policy对当前画面的action hypothesis，并非造成相邻视觉变化的teacher action。
v8停止，不再把8个probes压成effect-dominant单event。

历史v10 authority为
[`docs/action_forecast_writer_v10_design.md`](docs/action_forecast_writer_v10_design.md)
定义的Evidence-Preserving Dual-Stream Writer。它恢复text-only task axis与
v6 Semantic Set Core；8个稀疏Action probes形成保留raw mean的Action stream，
task-grounded patch forward difference形成Visual-Effect stream，二者以
`A0,V0,A1,V1,...`进入两层causal Procedure，不做strict multiplication、
joint `8×L` softmax或EventRead。compiler先读Core，再让Core-conditioned
query读取按Action/Effect分别中心化的Procedure；Procedure通过
`256→512→(gamma,beta)`直接提供content并门控full-rank Core。所有线性层
bias-free，结构保证`Procedure=0→public LoRA identity`，Core不能独自生成
adapter。

v10真实参数为`11,627,520`，相对corrected rank-128 Source-SFT多约12.9%；
canonical源码/config原位切换到不兼容v10 schema，不保留v8/v9并行可执行
路径。GPU4–7最长105-frame真实视频B20 profile与exact-resume通过后，正式
identity-fresh task-complete fast-decay run已完成macro0→400。12点paired
correct400为`95/103/84/89/82/90/96/96/89/96/97/91`，observed-best
macro50=`103/400`。其五臂为`103/94/75/67/43`：same同档且wrong/shuffled/
reversed行为门均通过，但absolute低于corrected Source-SFT `109`且距150为47。

内部证据显示v10并非没有读取视频：Core保持frame-set顺序不变，Procedure差异
完整传到effective LoRA/action，Procedure=0严格identity。但fixed Effect只变
Action时的顺序差异远强于fixed Action只变Effect，Effect attention仍近均匀；
Procedure-slot RMSNorm调制同时高增益放大同task不同正确视频的方差。训练loss
继续下降而online/closed-loop best均停在macro50，因此同recipe不再续训。
该“v10后暂停”边界先被owner对Loom的授权替换，Loom负结果后又由Recenter
授权替换；不得再据此阻塞当前Recenter实验。

focused AS硬门统一为single-checkpoint
`correct400 >= max(150, corrected Source-SFT best+30)=150`。达到absolute后
还必须same≈correct、correct显著优于wrong/shuffled/reversed、多个tasks共同
贡献、独立RNG/video permutation复测成立，且内部Core→Procedure→effective
LoRA→policy action符合职责。达到150不是自动停止；未达到时也不能因少数略低
checkpoint随意放弃，必须继续到真实瓶颈或当前recipe的充分负证据成立。

当前Writer通过后才做严格配对one-shot baseline与独立
short-AS-cold-start→pure-reward RL-Writer；不得把完整AS best冒充RL cold
start。focused闭环不自动继续final-32、test task-local RL、joint oracle或
ViVLA。

当前及后续GPU工作固定frame stride=5，只使用物理GPU 4、5、6、7；0–3不进入
visible set。4–7即使已有他人进程也按owner授权共卡，但不得杀、暂停、重置或
干扰。以推进效率为最高工程优先级：只保留直接防止无效实验、信息墙泄漏、OOM、
错误冻结/LoRA schema或不可恢复checkpoint的最小校验。最短垂直路径通过
shape/gradient/identity/freeze和一次resume smoke后立即进入真实GPU
profile/训练；不得用广泛全仓测试、重复流程门槛或文档整理延迟可运行实验。

## Data and split

- 目标 benchmark 为 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，共 40 tasks。
- 活动 development split 已封存在 `configs/libero_24_8_8_v1/`：每 suite 6 train / 2 validation / 2 test，总计 24/8/8；不得按 outcome 改 task IDs。
- validation 完成方法选择后，将 8 validation tasks 合入 source，形成最终 32 source / 8 test，并从规定初态重训已选方法。
- shared source-base corpus 来自 LIBERO-90。完整3600-pair specification-only
  audit已在看新policy outcome前封存：排除19个与目标40 exact
  semantic/composition重合的source tasks，保留71个active tasks。task44
  （`turn on the stove`）和task77
  （`pick up the book and place it in the back compartment of the caddy`）只是
  其中两项；不得把audit误写成尚待完成，也不得按outcome重开source IDs。
- source base 使用过滤后每个 active LIBERO-90 task 的全部 50 条成功 teacher episodes。不得使用 `pi05_libero`，因为它已读过目标 40 tasks actions。
- source-base action/state normalization 只从过滤后的 LIBERO-90 source actions/states 计算并冻结；所有下游方法共用，validation/test 不单独重算。

## Common frozen source base

活动文档中的 frozen π0.5-LIBERO source base 统一指：

```text
generic lerobot/pi05_base
→ 在过滤后的 LIBERO-90 source tasks × 每 task 50 条成功 episodes 上联合 action-SFT
→ 得到共享、多任务、语言条件的 π0.5-LIBERO policy
→ 若训练 recipe 使用 source LoRA，先 merge 成 base
→ 冻结，作为所有后续方法的共同起点
```

- 先调研官方/成熟 π0.5 fine-tuning 与 LoRA 实现，不自行猜 targets 或 runner 参数。
- source base 不追求高 ceiling；用全部目标 40 tasks 的小型快速 screen 确认它已开始在该 benchmark 上产生跨多个 task 的部分真实成功，不能只靠一个易 task 的 aggregate。这里不要求每个 task 已有高成功率。generic π0.5 的 `0/400` 只作原始校准，新 source base 必须另测。
- owner 于 2026-07-22 将 source-base 正式训练锁定为从 generic base fresh 运行 1,000 optimizer steps；不续接已停止且无 checkpoint 的旧 30k attempt。历史非focused阶段的约120分钟guardrail保留为其原实验合同；当前v5.1 AS/RL按上述focused authority分段探索，不受该旧guardrail限制。
- source base 冻结后，AS-Writer、RL-Writer、Source-SFT、三臂 task-local RL、联合 target-action oracle 和 ViVLA-style baseline（若做）均从它开始。
- 下游只保留一个活动 LoRA 空间；不得叠加未 merge 的 shared source adapter。

## Writer and source baselines

- 核心固定为 `task language + exactly one action-hidden teaching video -> shared Writer -> complete task-specific LoRA`。
- Writer 不得接收 action、proprio、reward、terminal、task ID、filename 或隐藏 normalization；source actions 只能进入 AS functional loss。
- `Action-Supervised Writer (AS-Writer)`：development在24 train tasks上做上述task-complete宏步；每个task只读1条teacher video并生成1套one-shot LoRA，`B_a`条独立同task action queries在该LoRA下各计算一次functional loss、先task内求均值，再让24 tasks等权。下一次macro访问该task时换一条video；video与action episode/chunk不要求同episode配对。frozen source base只通过functional LoRA forward参与，更新Writer。
- 历史task-local RL的总预算合同不影响当前focused v6 AS/RL，但每个新增训练段仍须通过当前证据门。
- `Reward-Trained Writer (RL-Writer)` 是独立路线：按当前 focused task 从新架构规定初态做短、task-balanced AS cold start，直到24个development-train tasks各在官方random-reset rollout中至少成功一次，再关闭action数据入口并跨source tasks做纯reward训练；它不从完整AS-Writer best继续，cold-start消耗必须完整报告。
- RL-Writer rollout 使用 LIBERO 官方随机 reset/BDDL 初态；不使用 `.pruned_init`。只用官方 env reward/success，不从 object pose 等内部状态手工构造 privileged shaping。
- `Source-SFT` 是在同一 frozen source base 上、跨 24 development train tasks fresh训练的一套 shared rank-128 LoRA，test 不看 held video/action。physical batch必须混合tasks，以`task→episode→chunk`分层均匀采样并做task-balanced loss，不得让rank固定为单一task。v6确认后默认重训并根据validation找最佳；它和AS-Writer不要求机械匹配optimizer steps或consumed examples，但必须报告训练数据、steps、GPU-hours、参数量和搜索上限。
- 所有方法共享同一frozen source base、normalization和policy接口，但不再机械要求相同trainable参数化或LoRA rank。Writer继续生成sealed rank-16 public task LoRA；capacity-matched Source-SFT可使用rank128，其10,297,344个trainable参数用于约束Writer本体参数预算。各方法的targets/rank/alpha/dropout与identity初始化都必须显式报告。

## Seen and video-causality evidence

- 必须增加 source/seen-task performance comparison；seen panel 在看 outcome 前按 specification 预声明并覆盖四 suites，不用它替代 validation/test。
- 必须做 wrong-video control：evaluation task、正确 language、init state、policy RNG 均不变，只把 Writer 输入换成另一 suite 的 teacher video。
- 对 AS-Writer 和可用的 RL-Writer均报告 source base、correct-video LoRA、cross-suite wrong-video LoRA；核心视频特异性量是 correct-video 与 wrong-video 的差异，而不是只看两者是否各自高于 base。
- zero-interaction held evaluation 每个 rollout 从正确 task 的 50 条 teacher videos 随机抽一条；不得挑最好视频。

## Final retraining and zero-interaction test

- development 只先跑一个 training seed。AS-Writer、RL-Writer（若成立）和 Source-SFT 在 24 train / 8 validation 上选定配置后，合并成 32 source tasks，从规定初态各自重训一次。
- 在打开最终 test 前先完成 final seen-task comparison。
- zero-interaction test 统一比较新的 frozen source base、Source-SFT、AS-Writer、RL-Writer（若成立）及 correct/wrong-video controls。旧 generic base `0/400` 不可冒充新 source base 结果。
- 旧 test 已做 generic/source-base feasibility audit，owner 明确不把这视为阻塞；不得再以“untouched test”异议停止推进。

## Test-only task-local RL

- task-local RL 不在 validation 上预训练、预冻结或选择算法；在最终 test 阶段打开后，直接把每个 test task 当作 adaptation training domain，在该 task 上调优并训练到 reward/性能曲线接近最佳。
- 三臂为：source base + functionally identity LoRA、AS-Writer LoRA、RL-Writer LoRA。RL-Writer路线失败时如实缺席，不伪造。
- 每个 `(task, adaptation seed)` 开始时随机选一条该 task teacher video；AS/RL Writer 两臂使用同一条并固定生成的初始化 LoRA，随后只原位更新该 LoRA。
- 三臂使用相同 task、env/policy seed schedules、官方随机 BDDL 初态序列、相同 RL 实现和可比的调优/资源上限；保存完整 optimizer、worker RNG、seed schedule、interaction cursor 与 exact-resume state。
- adaptation、调参和 checkpoint 选择可使用该 test task 的官方随机-reset reward rollouts；固定 50 `.pruned_init` states 只作训练分离的 fresh evaluation，仍执行 dummy settling、suite horizon 和成功即终止。

## Privileged direct-action oracle

- direct target-action baseline 不是 task-local per-task LoRA。
- 在三臂 RL 和无 action 方法结果封存后，从同一 frozen source base 出发，使用 8 个 test tasks、每 task 全部 50 条 action episodes，联合训练一套 shared multi-task LoRA；第一轮只做完整 50/task，不做 action-budget 曲线。
- 它是 privileged oracle/reference，不属于与 EMBER 同信息墙的主 baseline，也不得反向修改前面方法。

## Evaluation and efficiency

- official π0.5/LIBERO preprocessing 保持：render 256、model 224、两相机 180° rotate、state/action 7维、10 flow steps、执行前 5 actions后重规划、dummy settling 10、成功即终止、suite horizons 220/280/300/520。
- generic feasibility 已证明固定“一 task/一 GPU”会被两个 horizon-520 tasks拖尾；新 evaluator 必须先调研其他成熟项目，并按预计 `episodes × horizon` 做 cost-balanced state shards、动态任务队列和持久 model/env，而不是静态 task/GPU。
- Writer每 rollout LoRA 不同时，真实 profile batched functional LoRA 与每卡统一 1/2/3 个 policy replicas；选择有效 rollouts/s 最优且稳定的方案。所有卡使用相同 CUDA process count，GPU0 不得额外堆 controller/server/model。
- batch 8→16 只带来约 0.9% per-episode 提升，不能把继续堆同 adapter batch 当作唯一优化。
- 训练最多使用 8 张 A100 80GB，一卡一 DDP rank 为默认；用真实数据尽量利用显存并平均预留约 10GB。评估只优化有效 rollout/s，不用 dummy tensors填显存。
- 任何 GPU launch 前实时检查 GPU owner/telemetry、进程拓扑、CUDA/runtime、storage 和 `/data/ymdai` 500GB cap；不得干扰无关进程。

## Engineering, evidence, and delivery

- 只保留一条 canonical π0.5 path；不恢复旧 runner，不新增平行版本、bank、geometry、shared update subspace、residual escape 或额外 shared trainable adapter。
- smoke 只检查 load、shape、gradient、冻结对象、OOM、resume 和环境；不解释小分母性能。
- checkpoint 保存 model/Writer/LoRA、optimizer、scheduler/scaler、sampler/data cursor、每 rank/worker RNG、env seed schedule、interaction cursor、step、episode和consumed-data state。
- 等待下载、训练或 rollout 时推进不污染运行的后续代码、文档、hash和离线验证；精度细节不改变科学结论时效率优先。
- meaningful state 后更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push。核心闭环完成前不要停在只写脚手架或只报告单一 smoke。
- optional ViVLA-style matched reproduction 和 source-only outer learning 只在核心结果之后有时间再做，不阻塞长期 Goal complete。
