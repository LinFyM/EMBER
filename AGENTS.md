# EMBER Repository Instructions

## Authority and current truth

本文件、`docs/active_session_handoff.md`和`docs/execution_brief.md`共同定义当前实验
authority。三者顶部的当前状态覆盖历史设计、日志或artifact中残留的“当前”“下一步”表述；
历史文档只提供已发生事实和机制证据，不能自行恢复旧路线。

Owner已授权持续自主推进，直到同一shared method、同一single checkpoint的strict paired
correct严格超过`150/400`并尽可能继续提高。只有出现无法通过安全、只读检查或当前授权解决的
实质阻塞时才请求owner决策。当前明确边界：

- Owner已允许使用subagent加速独立、边界清晰的工作流。主进程负责统一科研判断、集成和最终验证；
  不允许多个write-capable agents重叠写同一worktree。
- 当前唯一主写分支是`codex/bci-continuation`；正式训练或评测使用该分支clean pushed commit的
  frozen worktree。历史分支和Git只作provenance，不恢复并行活动实现。
- 当前没有已授权继续训练的Writer候选。第33节whole-LoRA、第34节Expert-Component Projection（ECP）
  和第35节Condition-Local Dynamic Expert Tangent Tube均已由正式closed-loop证据裁决并退役，不能续训、
  扫权重或恢复并行活动实现。当前唯一focused科学动作是第36节的matched Expert-Flow Teacher Viability
  Audit：它是不更新参数的train24诊断，不是新部署方法；只有该门证明task expert在相同B20/noise/time上
  是更好的policy-flow teacher且提供非冗余梯度，才授权Cross-Episode Expert Flow Distillation（CEFD）。
- 第36节audit已在唯一canonical CLI的`teacher-audit` mode完成实现和CPU封存，但尚未运行GPU，所以当前
  没有teacher-quality、gradient-nonredundancy或CEFD科学结论。固定合同为world6、4 tasks/rank、logical
  B20/physical B10+10、每task 6次PI05 policy forward、三臂共享noise/time、真实7维FP32 loss、full24
  等权gradient、Gram pinv `rtol=1e-5`、0 optimizer/scheduler/update/rollout。下一步只允许从clean pushed
  frozen commit经live GPU/quota preflight运行这一次audit。
- 历史最好single checkpoint仍是v6-fast macro400的`143/400`。同一current schedule的v6-prior
  macro0/10/25/50 strict correct400=`134/127/105/123`，所以macro0是该轮winner且新训练没有改进。
- `30b2ccf`的batch8诊断显示普通BF16 batch-shape roundoff：相对single forward最大差
  `.001953125`、mean约`4.70e-5`，direct repeat为零。此前据此固定batch1、重复direct forward和
  逐tensor门禁的决定已被owner撤回：这些微差不是科学精度，不得以牺牲吞吐保留。
- 当前没有运行中的EMBER GPU任务。clean frozen`b308941`的Tangent formal fresh0→10已自然exit0：
  10 macros总step wall=`207.444s`、input wait=`.265s`、peak allocated/reserved=
  `43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite，只有macro3一次clip。macro10的
  correct/negative相对anchor正交半径中位=`.01390/.01408`，说明局部半径约束工作；但方向比中位=
  `108.93/126.88`、两臂均`0/24` tasks通过`≤1`，task median `|a_correct-1|=.25229`且`0/24`
  tasks通过completion门。随后同一one-shot correct400严格评测为`131/400`、breadth5、per-task=
  `0/3/46/31/0/40/11/0`；相对同schedule macro0=`134`为gained/lost=`16/19`、churn35、net`-3`。
  因而不续25、不补六臂、不扫tube weight/LR/WD；该结果淘汰当时的tangent recipe/window，但由于completion
  从未成立，不能写成对expert-component假设的干净证伪。代码/config已切为formal non-pass后fail-closed，
  Tangent证据已由`9f8f638`封存；当前待封存audit实现并执行一次matched诊断，决定是否值得实现CEFD。

## Mandatory reading

修改代码、配置、数据、split、模型或实验状态，或启动任何GPU工作前，主进程必须完整读到EOF：

1. `README.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_video_expert_manifold_design.md`
5. `task_plan.md`
6. `findings.md`
7. `progress.md`
8. `docs/concept.md`
9. `docs/decisions_and_open_questions.md`
10. `docs/novelty_and_landscape.md`

涉及迁移、路径或环境恢复时再完整读`docs/a100_to_bci_migration_handoff.md`。改变某个历史架构拥有的
接口前，必须按handoff实验谱系定位并完整阅读对应design；不得凭架构名或aggregate猜测旧方法。

## Objective and scientific decision rule

EMBER从generic`lerobot/pi05_base`出发，在过滤后的LIBERO-90 source tasks上训练并冻结共享
π0.5-LIBERO source policy，研究能否把目标任务的exact language和无action教学视频一次性编译为
完整task-specific LoRA，使同一frozen policy在未见初始化上闭环完成任务。

当前focused目标要求同一single checkpoint同时具备：

- strict paired correct严格`>150/400`且继续提高absolute；
- 多task breadth、相邻checkpoint低能力轮换和可重复累积；
- correct实质优于wrong、shuffled、reversed和no-video；
- same-task-other视频鲁棒；
- 视频语义/时序经Core/Procedure、effective LoRA传到policy action；
- 生成LoRA在能量、秩和跨target结构上足以驱动policy，但几何健康度本身不是目标。

每次结果必须与历史最接近架构、143起点、逐task成功集合、checkpoint gained/lost、五臂和内部
transfer共同比较。先找最早失效接口，只改一个有因果指向的结构或objective变量。负结果只淘汰其
实际检验的假设；没有被充分检验的结构不随意放弃。除非证据否定整条因果链，不作180度换路线。
训练loss、held functional loss、smoke、LoRA几何、视频hidden差异或单一margin都不能选择或宣告方法；
最终只认真实、严格配对的closed-loop single-checkpoint结果。

## Data and split

- 目标benchmark是LIBERO Spatial/Object/Goal/Long共40 tasks。
- development split封存在`configs/libero_24_8_8_v1/`：每suite 6 train / 2 validation / 2 test，
  共24/8/8；不得按outcome改变task IDs。
- LIBERO-90 specification-only audit排除19个与目标40 exact semantic/composition重合的source
  tasks，保留71 tasks；source base使用每task全部50条成功teacher episodes。
- 不得使用已经读过目标40 actions的`pi05_libero`。
- normalization只从过滤后的source actions/states计算并冻结；validation/test不得重算。
- validation选择方法后才合并32 source / 8 test并按authority从规定初态重训。

## Information wall and deployment contract

- Writer部署输入只能是exact task language加恰好一条action-hidden teacher video。
- video是唯一dynamic value。language可以作query/context/address，但不能单独生成LoRA或形成
  language-only LoRA bypass。
- Writer不得读取teacher action、proprio、reward、terminal、task ID、filename、object pose、
  hidden normalization或其他元数据。
- 每episode一条video生成一套完整38-target public rank-16 LoRA；不做多video、LoRA、checkpoint
  平均或融合，不挑最好video。
- frame stride固定5；frozen source policy不得有trainable parameters。template A/zero B保证
  no-video和step0 functional identity。
- task experts及phase feature cache只可进入train24监督或历史分析，不能成为部署输入、held oracle、
  nearest-expert route或第二套LoRA。

## Current scientific boundary and reusable training contract

当前没有可继续训练的active Writer。historical v6-fast macro400仍是下一候选的唯一允许load-only
初始化；encoder、Semantic Core、visual transition和Causal Procedure共483 tensors、`7,060,992`
parameters冻结，若audit授权CEFD，首版仍只训练compiler与factor heads共41 tensors、`3,714,304`
parameters。旧Tangent/ECP optimizer、scheduler、sampler、RNG和checkpoint不得加载或冒充exact resume。

下一候选若获audit授权，继续保持：

- train24 task-complete宏步，6 ranks×4 tasks；每task一条correct video、一套LoRA、B20同task
  跨episode独立action queries；先task内mean，再24-task等权并一次flat all-reduce。
- correct positive functional loss与bounded video ranking；不得恢复whole-LoRA attraction、ECP completion
  或Tangent Tube。CEFD若通过audit，只能把同一B20/noise/time上的step2000 task-expert PI05 flow velocity
  作为stop-gradient train24 teacher，不匹配A/B、dense BA、norm、cosine、rank或部署时expert输出。
- correct video与action queries错开episode；same-task不同video是共同positive分布，不作negative。
- negative只重排真实输入frames或使用预封存cross-suite wrong mapping；不得最大化negative action
  error、无限放大LoRA或读取negative任务隐藏信息。
- task experts统一使用step2000，不按task混合checkpoint；它们定义train-task policy-effective目标，
  不保证held泛化、视频顺序或same-task specificity。
- validation/test actions不产生梯度。formal checkpoint必须含Writer、optimizer、scheduler/scaler、
  sampler/cursor、每rank RNG、counterfactual schedule和完整schema；fresh/exact-resume不可混用。Tangent
  dynamic anchor及其auxiliary只由Git与retained artifacts保存，不进入下一活动runtime。

禁止用scalar/global scale、confidence gate、B-only residual、static/language bypass、强制正交或rank
diversity、multi-video、checkpoint fusion去救一个失败点。few-shot只有在one-shot跨video方差被严格定位为
最早瓶颈后才重新设计；不能用平均无效expert字典替代视频理解。

## Throughput and numerical policy

用户明确要求吞吐、有效GPU利用率和尽快获得真实科学证据优先：

- 不为了底层微小浮点差异降低batch、关闭并行、重复forward或逐元素比对。接受正常BF16 kernel、
  batch shape和reduction order roundoff，只门禁OOM、nonfinite、shape、信息墙、明显串样和合同变化。
- Writer生成从单卡profile中稳定且仍有实用显存余量的候选选择实测samples/s最高点；`8/16/32/...`
  必须处理同一request panel和同一总sampled frames，只改变forward分批，再比较wall、samples/s、peak
  allocated/reserved和端到端rollout。不预设batch1，也不机械选择最大batch。
- 最大化有用显存：优先增加真实batch、microbatch、prefetch、并发generator/worker；不能用dummy tensor
  占显存。LoRA cache保持template原生BF16/F32 dtype，不强制扩宽FP32。
- 减少热路径CPU同步、重复finite扫描和重复analysis；单物理batch走直接gradient path，真实
  multi-microbatch才保留FP32 gradient accumulation以避免有效小梯度丢失。
- 数值稳定所必需的FP32 normalization、softmax、RMSNorm、objective reduction等只有profile证明为
  明显瓶颈且低精度不伤闭环时才改变；吞吐优先不等于无证据破坏模型语义。
- 不生成、重算、比较或门禁SHA-256、MD5等内容hash；Git commit ID不属于额外内容校验。
- 聚焦测试和真实vertical path优先，不做大量防御性扫描、全量artifact复核或与当前假设无关的
  test harness。科学负结果不是工程故障。

## Evaluation and video causality

- official π0.5/LIBERO preprocessing固定：render256、model224、两相机180° rotate、state/action 7维、
  10 flow steps、执行前5 actions后replan、dummy settling10、成功即终止、suite horizons
  220/280/300/520。
- zero-interaction rollout从正确task的50条teacher videos无放回取一条；不得按结果选video。
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对state、env seed、policy
  RNG、video ordinal和其他身份字段。shuffled/reversed在真实frames重排后完整forward。
- evaluator使用cost-balanced dynamic queue、long-first和persistent model/env；不用静态task/GPU分配
  或dummy显存。
- 小panel只作执行或早期screen，不能冒充真实水平。关键checkpoint必须跑paired correct400；候选winner
  再跑完整五臂/六臂。报告aggregate、per-task、per-suite、breadth、gained/lost、union/intersection、
  checkpoint churn及Core→Procedure→effective BA→fixed-action传递。

## GPU, storage, and multi-GPU

- 每次GPU launch前实时检查`gpu01`和`gpu02`的device index/UUID、显存、利用率、进程和ownership，
  只用完全空闲且健康的A40，跨两节点合计最多6张；不得reset、kill、pause或干扰他人。
- 同时检查`/data1`个人quota、相关目录用量和本次峰值增长。大数据、模型和run留在
  `/data1/user/ymdai/projects/EMBER`既有canonical roots，避免重复资产。
- BCI多卡必须显式`NCCL_P2P_DISABLE=1`并遵守仓库launcher的SHM、NUMA、physical/local rank映射、
  Ring/Simple和deferred-NCCL合同。resume保持原world size、rank/NUMA topology和collective序列。
- 故障按transport、rank/device、process-group生命周期、CUDA初始化、I/O/NUMA和workload分层定位；
  不通过盲重试、加timeout、关闭watchdog或缩减科学batch掩盖工程故障。
- 单卡smoke/profile用最小直接机制；昂贵canonical训练前必须检查clean frozen worktree、commit、命令、
  env、资产路径、output root、overwrite语义、设备拓扑和storage预算，并记录launch contract。

## Artifacts, engineering, and Git

- 一个canonical Writer implementation、config和evaluator path。替换行为后旧实现由Git、design和frozen
  artifact保存，不保留可执行并行版本或兼容fallback。
- 新模块、抽象或入口必须有当前第二用途或现有owner无法承载的证据。非平凡结构变化使用
  `code-architecture-gate`；清理使用evidence-driven workspace cleanup。
- 保留formal run contract、checkpoint manifest、metrics/raw rows、aggregate、completion和必要analysis；
  profile/smoke权重不得进入formal，工程smoke不得冒充性能。
- 不比较未严格配对panel，不因held outcome改split、video schedule、checkpoint集合或objective。
- meaningful状态更新`docs/active_session_handoff.md`、`docs/execution_brief.md`、当前design、
  `task_plan.md`、`findings.md`和`progress.md`，然后聚焦commit/push。不得提交dataset、cache、checkpoint
  或大binary，不提交无关用户改动。
- 清理只删除已验证obsolete/temporary路径；formal outputs、unique diagnostics、datasets、checkpoints和
  含科学证据的历史产物默认保留。Git历史足以保存已退役代码，不在仓库内另建archive。

## Retired boundaries and later stages

旧SmolVLA、70/10/10、Phase A–F、flat task-local RL、flat Writer-RL、K4/few-shot executable、
hard/soft/sparse online expert dictionary、topology-address binding和Causal Barycentric部署均已退役，
不得从历史命令恢复活动路径。它们的已验证机制与负结论继续作为当前设计约束。

RL只有在同一健康video-to-LoRA图上、从架构规定的短task-balanced AS cold start重新设计；不得从完整AS
best直接续、不得打开teacher action入口或使用`.pruned_init`。final32、test、task-local RL、privileged
direct-action oracle和其他later stage不会因focused Writer局部结果自动获得launch authority。
