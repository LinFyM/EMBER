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
- 第33节whole-LoRA、第34节Expert-Component Projection（ECP）和第35节Condition-Local Dynamic Expert
  Tangent Tube均已由正式closed-loop证据裁决并退役，不能续训、扫权重或恢复并行活动实现。第36节matched
  Expert-Flow Teacher Viability Audit也已从clean frozen`e8e4728`正式完成：expert/macro0/tangent10的
  matched真实7维flow loss=`.098631/.091802/.091843`；expert仅`2/24` tasks且`0/4` suite means同时优于
  两baseline，远未过`18/24+3/4` teacher门。gradient residual在compiler/factor为`.6864/.8387`虽非冗余，
  但方向来自整体更差teacher，因此`authorize_cefd=false`，不得实现CEFD或做weight profile。
- audit完整覆盖480/480 queries、144次policy forwards、0 update/rollout/OOM/nonfinite，wall=`39.698s`；
  一次性mode/config/code已formal non-pass后退役。第37节Frozen-v6 Counterfactual-Null Program Residual v1
  随后从clean frozen`6903ee6`完成唯一macro49 profile：13项门中10项通过，correct retention=`.807966`
  且24/24，但旧DC-dominated key使condition=`1315.33`、negative/correct=`.264351`、null仅15/24；
  production ratio=`1.115458`。因此v1不训练、不降lambda、不扫seed/P/阈值。
- 第38节Balanced DC--Causal v2的机制、部署、zero-memory identity、formal0→25和macro10/25 strict400均已
  完成。clean frozen`abd8e08`的25个task-complete macros总step wall=`535.465s`，0 OOM/nonfinite/negative
  policy forward；macro0/10/25 correct=`134/140/139`、breadth均6，per-task分别=
  `0/5/48/34/0/35/11/1`、`1/2/48/31/0/38/20/0`、`2/4/48/30/0/38/17/0`。
- 严格配对转移为0→10 gained/lost=`19/13`、0→25=`18/13`、10→25=`12/13`；macro0与macro10的
  success union=`153`但任何single checkpoint都未超过历史`143`，且10→25已净退化。v2因此不续50、
  不补五臂/六臂、不扫lambda/seed/P；其机制成功但没有形成稳定共同累积。
- v2 macro10相对macro0的effective-BA delta/base中位仅`1.69498e-4`，effective stable rank中位仍约
  `1.000022`、top-1 energy约`.999978`，LoRA形态几乎未改变。更关键的是同一task的50条视频所诱导raw
  correction consistency=`.141539--.142175`，等于随机正交基准`1/sqrt(50)=.141421`；固定10-video的
  effective-BA pair cosine跨8 tasks落在`[-.001371,.003280]`。因此blind-add把每个新video写成近正交小扰动，
  能偶然换手但不能保留此前能力；这比单看aggregate更直接定位到跨macro reconciliation失效。
- 第39节**Exact Anchored Reconciliation**已从clean frozen`25bbd52`完成formal fresh0→10和预注册
  macro10 strict correct400。训练natural exit0、10 rows、0 OOM/nonfinite，六卡结束后释放；strict=
  `140/400`、breadth6、per-task=`2/3/47/35/0/34/19/0`。相对exact balanced macro0 retained/gained/lost=
  `119/21/15`、churn36，未过`lost<=6`门；相对blind-v2 macro10同为140但仍`17/17`换手。RLS没有减少
  held closed-loop旧成功丢失，因此不续25、不补六臂、不扫damping/step/window/forgetting。
- correct80在同一比较中为`31 vs 26`、gained/lost=`5/0`，但full400为`21/15`；不得用80-row screen选
  checkpoint。RLS内部current/blind降到`.230340`、final precision condition约`8325`，而formal的
  `reference_correct_rows=0`，old-row improved=`1`是空集合值。短历史feature-row保留成立但不足以保护
  validation闭环能力，最早接口转为functional cotangent与on-policy occupancy/reward credit错位。
- RLS config/runtime已由formal completion、macro10 checkpoint、strict results和immutable transition共同
  封为`retired_after_macro10_strict_closed_loop_nonpass`，fresh/restart/resume均fail closed。第39.5
  Reward-Credit Program Cotangent已完成canonical实现和首次live discarded profile：保持Balanced-v2
  one-shot部署图、P256、frozen v6 decoder、single Program、full48 RLS和完整rank16 LoRA，只把offline
  source-action cotangent换成train24 K4 binary-LOO、Nmc4 executed-prefix on-policy Program cotangent。
  禁止继承RLS10 precision、恢复old/current双forward、第二epoch、shared Adam、progress reward、SPSA、
  few-shot或multi-video。
- clean frozen`c4507e9`的首次full24×K4×Nmc4 B2 profile在`gpu02:0--5`完成24 tasks/96 rollouts，11 mixed/
  13 homogeneous、full48 rank48、closure0、LoRA/action response和runtime health均成立，但按旧预注册门正式
  non-pass：固定probes`0/7/14/21`中只有0是mixed，另外三个homogeneous按合同zero credit，故不应被要求
  action-nonzero。旧artifact不可追认或续用。
- clean frozen`e6024cf97200721b13834c6ad81de85ce6588ffb`的新profile v2随后在`gpu02:0--5`自然完成并
  正式`passed=true`：24 tasks/96 rollouts、11 mixed/13 homogeneous、60/36 successes/failures、full48 rank48、
  negative/correct=`.01704835`、closure=`0`；11/11 mixed tasks的Program→LoRA A/B→真实BF16 action均
  finite/nonzero且覆盖四suite，homogeneous direct credit全部exact zero。raw forward计数唯一反推出physical
  B8（928 invocations；同面板B2为3648），0 OOM/nonfinite/watchdog且不保留checkpoint。wall=`507.305s`，
  peak allocated/reserved=`36,575,930,368/40,928,018,432B`；六卡结束后均已释放。
- Reward-Credit随后已从clean frozen`e3857f7`完成formal cycle0→1和预注册correct400。训练natural exit0、
  24 tasks/96 rollouts、B8、0 OOM/nonfinite，checkpoint完整；strict仍为`134/400`、breadth6、per-task=
  `1/4/46/31/0/38/14/0`，相对zero-Program macro0严格`14 gained / 14 lost`。未达到cycle2或control门，
  不续cycle2、不扫reward scale/K/Nmc/RLS参数；这是真实closed-loop non-pass。
- 分层诊断没有把该non-pass误写成“视频或Reward方向无效”：correct Program、analytic FactorHead tangent和
  continuous effective-BA都保留强task-common与same-video结构；首个失效接口是36个q/v target把约`1e-8`
  RMS的factor delta加到非零BF16 A/B，远小于约`1e-4`局部ULP。own-target native cosine约`.037`，而action
  四个FP32 factors不存在同一问题。FP16、dither、local-CD、gauge/global-scale和absolute full-rank refactor
  都已直接失败，不得重扫。
- 唯一active设计改为`docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`，其canonical
  load-only实现、family/config和ordered gate已落在当前worktree：q/v保留14个
  pivot-selected原生B columns并重解A，另用两个physical zero-B slots承载condition-local rank2 Reward
  residual；action保持原full-rank16 FP32。80-row generation-only门已通过：q/v base error约`.000752`、
  tangent capture`.999709`、dynamic cosine`.997525`、video-centered cosine`.95056`，action exact；但0 policy
  forward/0 rollout，不能冒充性能。
- 活动config是`configs/pi05_v6_qv_rank_reserved_native_reward_v1.json`；cycle1只经
  `configs/pi05_v6_qv_rank_reserved_cycle1_program_load_only_v1.json`读取原Program tensor，旧Reward训练入口
  在distributed/runtime初始化前fail closed。历史最好single checkpoint仍是v6-fast macro400的`143/400`，
  长期`>150/400`目标未达到。当前config仍为`awaiting_live_a40_rank_reserved_deployment_profile`且
  `online_smoke_evidence=null`，所以尚未formal ready。当前没有运行中的EMBER GPU任务；带真实LIBERO assets
  的全仓CPU seal已完成（`386 passed`）。下一顺序是clean commit/push/frozen后独立完成真实profile/vertical，
  再跑新rank14 macro0 strict400；只有base保留后才评估cycle1 load-only。
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
  Tangent证据已由`9f8f638`封存；audit实现/修复和正式结果分别由`7be51b1`/`e8e4728`及retained artifact保存。

## Mandatory reading

修改代码、配置、数据、split、模型或实验状态，或启动任何GPU工作前，主进程必须完整读到EOF：

1. `README.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_video_expert_manifold_design.md`
5. `docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`
6. `task_plan.md`
7. `findings.md`
8. `progress.md`
9. `docs/concept.md`
10. `docs/decisions_and_open_questions.md`
11. `docs/novelty_and_landscape.md`

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

当前没有运行中的GPU Writer。RLS及Reward-Credit cycle1均已有full400 closed-loop non-pass；历史artifact继续
作机制证据，但RLS不得恢复，Reward-Credit不得续cycle2或扫训练超参。现有cycle1 checkpoint只允许作为
rank-reserved compiler的load-only learned Program，不是训练起点。

当前只允许推进Q/V Rank-Reserved Native Reward Compiler，并保持以下边界：

- exact language + exactly one action-hidden teacher video仍是唯一部署输入；P256 Balanced key、frozen-v6、
  single Program、38 targets、public rank16和one-shot不变。没有language bypass、expert bank、second LoRA、
  multi-video或deployment reward/action读取。
- 36个q/v target使用pivot-preserving rank14 base + condition-local rank2 physical zero-B residual；两个action
  target保持原full-rank16 FP32。q/v tangent为`B0 dA+dB A0`且不含二阶cross term；action实际候选保留
  `(B0+dB)(A0+dA)`。base/residual slots不得gauge mixing，carrier不得固定global化或来自task experts。
- 先在同一32-request panel实测B8/16/32并取最高稳定吞吐；单个更大候选OOM时只记录为ineligible，不抹掉
  已完成的稳定候选。随后做native cache/adapter load、Writer release与五臂fixed-action vertical closure；
  full Reward和q/v-only两臂必须使用cache重新加载的q/v state，第五臂用rank14-base action排除健康action
  residual冒充q/v传递。8-request vertical的实际cache forward自然为`min(selected,8)`，不得冒充selected batch；
  selected batch本身由32-request profile证明。随后用
  official paired validation
  8×50分别做zero-Program新macro0与现有cycle1 Program load-only correct400。小panel、几何或reconstruction
  不能选择方法。
- correct-video zero-Program是rank14 v6 base，不是identity；no-video必须显式跳过compiler并返回
  template-A/zero-B source identity。
- 新macro0若correct<130、相对同schedule旧macro0 134 lost>10或breadth<6则reject；per-suite退化必须报告，
  但在没有预注册数值定义时只作诊断，不能事后成为硬门。
  只有其保留base后才运行
  cycle1 load-only；低于新macro0即reject，只有`>=144`、breadth`>=6`、gained>lost且lost`<=6`才算通过并补
  同checkpoint controls。140--143只作诊断性non-pass，不授权新训练。
- 这两个行为门完成前不实现新训练。若load-only通过，fresh可导版本必须另写authority，用native forward与
  明确continuous surrogate/STE；不得对zero tangent做SVD backward或通过扩dtype/缩batch换取精度。

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

- 每次GPU launch前实时检查`gpu01`和`gpu02`的device index/UUID、显存、利用率、进程和ownership。
  比较后选择一个节点，使用该节点当时所有真正空闲、健康且能提高有效吞吐的A40；没有固定6卡上限，不等待
  凑卡、不dummy占位、不为跨节点碎片改launcher，也不为多用卡启动没有科学价值的重复工作。不得reset、
  kill、pause、触碰有他人compute process的卡或干扰他人。
- 同时检查`/data1`个人quota、相关目录用量和本次峰值增长。大数据、模型和run留在
  `/data1/user/ymdai/projects/EMBER`既有canonical roots，避免重复资产。
- BCI多卡必须显式`NCCL_P2P_DISABLE=1`并遵守仓库launcher的SHM、NUMA、physical/local rank映射、
  Ring/Simple和deferred-NCCL合同。需要exact-resume的训练保持该run原world size、rank/NUMA topology和
  collective序列；独立评测按所选单节点的live空卡数动态扩展cost-balanced queue。
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
