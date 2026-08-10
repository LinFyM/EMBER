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
- 当前唯一active implementation改为第39节**Exact Anchored Reconciliation**。它完全保留v2的one-shot
  deployment graph、P256 balanced video key、frozen v6 decoder、完整rank16 LoRA、B20/B10+10和full48
  functional rows，只把training-only blind-add改成FP64 feature-space recursive anchored ridge/RLS：每次
  目标锚在更新前的既有condition输出，再叠加当前correct descent、negative零增量；同时保存`[256,256]`
  precision和`assimilated_rows`。首步数学上精确退化为v2 blind solve，之后显式抑制旧correct-row漂移。
- RLS为fresh-incompatible family，禁止从v2 macro10/25伪resume。clean pushed/frozen`f0c3f51`已在
  `gpu02:0--5`完成首次fresh0→3 discarded profile，exit0且0 checkpoint/OOM/nonfinite/negative forward。
  old-row drift/blind=`.248611/.213872`、旧rows改善=`100%/100%`、current/blind=
  `.999980/.784334/.640650`，其余24-task/LoRA/action/null/closure/state门均通过；原18门只有ppm级首步
  比值和单个fresh宏步对warm基线的wall门失败，旧artifact必须保持16/18 non-pass。
- 第39.4.1已据此只修正测量合同而不改架构、RLS数学、dtype、batch、worker或forward：GPU首步ratio继续
  记录但不以`1e-5`低位差作hard gate，精确等价由CPU FP64 oracle负责；吞吐改为三个diagnostic macros的
  production算术均值对原sealed baseline仍`≤1.10`。当前mean ratio=`1.029799`，同schedule v2总wall只差
  `.277%`。必须从新clean pushed/frozen commit再做一次fresh0→3，不能事后seal或复用`f0c3f51` artifact；
  新run的17项门全部通过后才formal fresh0→10并立即strict400。
  fresh formal必须在结果出现前把固定macro0=`6b5f7a6` root与唯一尚不存在的macro10 strict root登记进原
  run contract；formal evaluator只接受predeclared macro10/25。10→25会从两份immutable 400-row roots
  重聚合并核对commit/checkpoint/state/RNG/language/actual video，只有correct≥140、lost≤6、breadth≥6才
  允许进入同commit exact-resume，否则fail closed；macro25不是预授权动作。
- 历史最好single checkpoint仍是v6-fast macro400的`143/400`，长期`>150/400`目标未达到。当前没有运行中的
  EMBER GPU任务；不得把RLS的CPU矩阵等价、profile门或reconstruction/几何当作性能结果。
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

当前唯一active Writer是第39节Exact Anchored Reconciliation。第38节v2已完成并因blind-add缺少跨macro
能力保留而退役；RLS只改变训练更新和checkpoint sufficient state，部署计算图与v2同构且代码路径不变。historical
v6-fast macro400仍是唯一load-only初始化；expert flow、whole-LoRA/ECP/Tangent/CEFD、旧ranking/completion
cotangent和task-expert输出不得进入活动update。当前方法严格冻结historical v6整套600 tensors，在fused
Program后只加入同一个video-keyed FP32 residual；没有optimizer/scheduler/scaler、language bypass、expert
bank或第二套LoRA。

当前RLS family保持：

- train24 task-complete宏步，6 ranks×4 tasks；每task一条correct video、一套LoRA、B20同task
  跨episode独立action queries；先task内mean，再24-task等权。每rank只all-gather本地8个condition features
  和4个correct cotangents，六rank按固定full48排序独立形成同一Program write；不all-reduce约80MiB memory。
- correct只使用真实source-action functional cotangent；不得恢复whole-LoRA attraction、ECP completion、
  Tangent Tube或CEFD。反事实条件只可约束structured residual不被当前correct更新带动，不得最大化negative
  action error、加载wrong-task expert或形成第二套policy/LoRA。
- correct video与action queries错开episode；same-task不同video是共同positive分布，不作negative。
- negative只重排真实输入frames或使用预封存cross-suite wrong mapping；不得最大化negative action
  error、无限放大LoRA或读取negative任务隐藏信息。
- 统一step2000 task experts只定义历史train-task policy-effective参照；当前RLS训练和部署均不读取其输出，
  它们也不保证held泛化、视频顺序或same-task specificity。
- full48条件`F=[F_correct;F_negative]`，当前functional cotangent只形成correct增量`E=[-G;0]`。目标是
  `T=F M_before+E`；累计目标的feature precision从`I_256`开始，每macro同化48 rows并用Woodbury/RLS精确更新
  Program memory。首步必须与旧blind solve等价，后续必须同时保留旧condition输出和当前有效motion。
- validation/test actions不产生梯度。formal checkpoint分别保存deployment-owned单个FP32 Program memory与
  training-only FP64`[256,256]` precision、`assimilated_rows=48*macro`、cursor和六rank RNG；historical v6
  600 tensors与fixed projection必须strict重建且不能被checkpoint覆盖。部署只加载Program tensor，不加载
  precision。fresh/exact-resume不可混用；v2 checkpoints缺少precision，绝对禁止伪resume。
- A40只先做fresh0→3 disposable mechanism/throughput profile；通过预注册保留门后formal只允许0→10，随后
  立刻跑严格paired correct400。macro10支持门为correct≥140、相对macro0 lost≤6、breadth≥6；correct>140
  才是强absolute证据。未通过则先裁决能力保留是否为根因，再考虑已设计的on-policy reward-credit后备，
  不能边跑边改阈值或直接续25。
- fresh0→10必须预注册固定macro0 strict root和唯一macro10 output root；root路径写入run contract，不能在
  看到outcome后另选。10→25前由canonical paired analysis从queue/shards重聚合，不信任手填aggregate或
  passed flag；formal deployment也拒绝未声明的中间macro。

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
