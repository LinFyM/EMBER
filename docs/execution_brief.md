# EMBER Current Execution Brief

## 0. Outcome and current operation

目标是把同一shared method、同一single checkpoint的strict paired correct从历史最好`143/400`推进到
严格`>150/400`并继续提高，同时保留真实视频时序因果、same-task鲁棒、breadth和稳定积累。当前没有
运行中的EMBER GPU任务。v6-prior whole-LoRA objective已完成formal 0→50，同一schedule
macro0/10/25/50 strict correct400=`134/127/105/123`；macro0仍最佳，因此该objective已退役。

第34节v6-Initialized Expert-Component Projection（ECP）已完成formal0→25并负裁决。
macro10/25 strict correct400=`133/120`；同schedule macro0=`134`。macro25对macro0的精确
paired gained/lost=`13/27`、net=`-14`、McNemar `p=.038477`，per-task=
`0/1/43/27/0/33/15/1`。内部`a_correct`已从`.736`增到`.884`且23/24 tasks向1移动，
expert component也在24/24 tasks上升；但macro10→25的expert-orthogonal norm增量`8.471`远大于
component增量`.228`。所以ECP的实现和机制生效，但held closed-loop明显退化；按预注册
不续50/100、不扫权重、不补六臂。

第35节**v6 Condition-Local Dynamic Expert Tangent Tube**也已正式完成并退役。clean frozen
`b308941`的fresh0→10自然exit0；10 macros总step wall=`207.444s`、input wait=`.265s`、peak
allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite。macro10的correct/
negative relative-anchor tube中位=`.01390/.01408`，但directional ratio中位=`108.93/126.88`、
两臂均`0/24` tasks通过`≤1`；task median `|a_correct-1|=.25229`且completion为`0/24`。
共享Writer因此学到的是“小而几乎全正交的更新”，不是所设计的expert方向修正。

紧接着同一frozen checkpoint的one-shot correct400自然完成：`131/400`、correct80=`27/80`、breadth5、
per-task=`0/3/46/31/0/40/11/0`、per-suite=`3/77/40/11`。相对同schedule macro0=`134`严格
paired gained/lost=`16/19`、churn35、net`-3`、McNemar `p=.735879`。所以不续25、不补六臂、
不扫tube weight/LR/WD；current config/runtime已封为formal non-pass并fail-closed。该结果只淘汰当前
tangent recipe/window，不能把未达到completion的实验扩大解释为expert-component假设整体无效。

第36节matched Expert-Flow Teacher Viability Audit已从clean frozen`e8e4728`正式完成并自然exit0。
480/480 queries、24 tasks、suite 6×4、8/8/8 negatives、144 PI05 forwards、0 update/rollout/OOM/nonfinite；
wall/input wait=`39.698/.684s`，peak allocated/reserved=`43,418,974,720/47,133,491,200` bytes。
expert/macro0/tangent10 matched真实7维flow loss=`.098631330/.091801740/.091843160`。expert只在`2/24`
tasks和`0/4` suite means同时优于两baseline，明确未过`18/24+3/4`teacher门。compiler/factor gradient
residual=`.6864/.8387`虽通过非冗余门，但来自整体更差teacher，因此`authorize_cefd=false`。

第37节**Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual**现已完成canonical实现和CPU
seal，但尚无A40或closed-loop结果。实现严格冻结historical v6 macro400的600 tensors，以固定
zero-preserving时序video feature读取一个`[256,320,256]` FP32 Program memory；24个correct B20
functional cotangents与24个wrong/shuffled/reversed zero-motion rows通过48×48 FP64 Gram直接形成FP32
manual write，没有optimizer/scheduler/scaler、expert target或negative policy forward。step0精确保留v6
LoRA；deployment仍只输出一套38-target rank16 LoRA。

一次性teacher-audit/effective-objective/flow-teacher执行路径已删除；checkpoint只保存Program memory、
cursor和六rank RNG，historical v6 base与fixed projection不归checkpoint所有。训练、checkpoint、deployment
v8 adapter、one-shot六臂证据和cross-family analysis已经联锁；错误family不能继承本候选的profile seal。
profile artifact会从raw macro重算全部门并匹配完整scientific run，formal result必须绑定completion、50-row
metrics与macro10/25/50 manifests，trained deployment checkpoint必须位于active authority lineage；clean
detached frozen authority ancestor可直接运行v8 evaluator。CPU最终证据为全仓`280 passed in 21.02s`、
compileall、JSON/diff-check通过，architecture guard无hard violation。
这些只证明实现合同，不构成机制或性能结果。

当前操作顺序：

1. v6-prior whole-LoRA、ECP与Tangent三条连续单变量实验的formal/strict/机制诊断均已封存并释放GPU；
2. Tangent只解决了局部半径，没有解决共享decoder Jacobian把更新旋进expert方向的问题；strict同时降到
   `131`，所以禁止靠续训、扫权重或硬加大auxiliary掩盖首个失效接口；
3. audit teacher-quality已方向性失败，禁止CEFD、weight profile、换expert step或把gradient novelty当价值；
4. 一次性audit路径已退役；第37节实现和CPU门已完成，不能再从旧Tangent/audit owner恢复执行；
5. 下一次GPU只做clean pushed/frozen commit上的macro49 mechanism profile：保持B10+10和0 negative policy
   forward；production wall对sealed v6 `21.095109596s` ratio`≤1.10`，至少18/24 correct retention、
   18/24 negative null、4/4 fixed-action breadth；8次fixed-action verification forward排除在production wall外；
6. mechanism seal后单卡profile新residual deployment graph的batch8/16/32并做correct smoke；两类seal齐全才
   评测zero-memory macro0、formal fresh0→10和strict correct400；
7. 任一后续候选仍须clean push/frozen、live preflight、短正式训练和及时strict correct400，
   不从Tangent checkpoint resume。

不得从下文自行跳到later stage，也不得从历史文档恢复已退役命令。

## 1. Fixed scientific contract

- 方法：one-shot Video-Conditioned Writer总路线；whole-LoRA/ECP/Tangent/CEFD均已退役或否决。当前唯一
  active implementation是第37节frozen-v6 counterfactual-null Program residual；尚未获得live profile seal，
  因而formal训练仍被配置硬阻塞。
- 输入：exact task language + exactly one action-hidden raw teacher video。
- 视频是唯一dynamic value；无language-only LoRA bypass、expert-bank部署、multi-video/LoRA/checkpoint
  平均或融合。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing与paired evaluator固定。
- historical v6-fast macro400是当前active candidate唯一允许的load-only Writer初始化；不得加载任何退役
  optimizer/scheduler/sampler/RNG，也不得用task expert输出作为cotangent。
- train24 task-complete、每task logical B20跨episodequeries、每visit一条correct video、24-task等权；六rank
  各gather本地4个correct cotangents和4+4 condition features后独立形成同一manual write，不all-reduce
  80MiB memory。
- 不恢复global norm attraction、whole-LoRA cosine、ECP completion、Tangent dynamic anchor、CEFD或
  parameter-distance代理。structured candidate只读correct真实functional cotangent；当前counterfactual
  可作为zero-motion condition constraint，但不得读取wrong-task expert或最大化negative action error。
- step2000 task experts仅作train supervision，不进入deployment或held选择。

## 2. Throughput-first runtime contract

“精度”指科学合同和真实closed-loop，不指底层逐元素复现。执行中遵守：

- 接受正常BF16 kernel、batch shape和reduction order的低位差异；不因`.001953125`级LoRA roundoff固定
  batch1、重复single forward或逐tensor扫描。
- 只门禁shape、finite、信息墙、cache完整性、明显跨样本污染、OOM、错误asset和resume语义。
- Writer生成从batch8起profile，逐步增加到16/32/更高，直到吞吐进入平台、allocator明显抖动或接近
  OOM；选择samples/s最高且能稳定完成真实最长video batch的点。目标是让A40 reserved memory保持高
  利用而非留出大块空闲，不用dummy tensor填充。
- LoRA cache保持template原生72 BF16 + 4 F32 tensors，单entry tensor data=`2,641,920` bytes；不强制
  扩宽FP32。每batch集中nonblocking D2H，只同步一次。
- action query DataLoader默认2 spawn workers、persistent workers、prefetch2；profile若显示GPU仍等待
  data，再实测4 workers/prefetch而不是猜测。
- functional scientific batch固定logical B20。live A40已证明physical B20和B16+4容量不足，当前以完整
  有序logical-B20 panel identity为key生成同一20个独立flow time/noise，并用balanced physical B10+10
  与FP32 leaf-gradient加权累积；不减少queries、不改变task mean或objective分布。
- policy activation checkpointing当前保持关闭；现有Writer checkpointing不覆盖OOM所在的frozen PI05
  Gemma MLP。只有B16/B10都无法形成高吞吐有效配置时，才把policy重算作为独立正式候选。
- 不做SHA/MD5，不重复全仓hash或历史artifact扫描。CPU全仓回归只在代码合同变化后运行一次。
- mechanism profile的production wall只计correct functional work与full48 gather/solve/write；task-local
  evidence、application closure、LoRA A/B和4-task fixed-action probe是一次性verification，单独计时且不进入
  formal热路径。fixed-action使用同observation/同noise的before/after推理，不读取target action；不能把它
  误算成negative functional forward，也不能让它掩盖生产吞吐。

保留FP32 RMSNorm/softmax/ROPE/image normalization和policy-effective reduction，除非profile证明它们是
显著瓶颈且降低精度不伤真实闭环；吞吐优先不是盲目改变模型数学。

## 3. CPU seal before GPU

GPU前一次性要求：

- `git diff --check`和聚焦/全仓CPU tests通过；
- historical 600-tensor Writer strict-load并全部冻结：`10,775,296` base parameters、0 trainable
  parameters；新增`20,971,520`个FP32 memory values由manual update拥有，但仍不注册为optimizer parameter；
- validation8 real-asset inspect和CLI prepare产生正确one-shot requests，部署expert-bank reads=`0`；
- no-video不读frames并返回template-A/zero-B identity；wrong/shuffle/reverse只改变允许的视频输入；
- native LoRA storage descriptor从checkpoint metadata贯穿run contract与cache write/load；resident policy的
  destination dtype由已验证的同一template决定，正常路径不发生额外转换，不在每次replan加dtype扫描；
- 2-worker prefetched sampler与serial、prefix+resume逐row一致；
- retired Tangent config只继承未改变推理图的retained evaluation throughput seal；ECP gradient、aux
  weights和resume profile没有被复用；v3 gradient与resume evidence曾从各自clean frozen lineage重新
  实证并原样封存，formal0→10完成后已切为non-pass并关闭runtime。任何status-only、旧family evidence或
  stale config仍fail-closed。
- retired teacher-audit CPU oracle曾覆盖physical B20的3次forward与B10+10的6次forward；正式artifact已
  证明480/480 queries、144 forwards、real7、8/8/8 negatives和0 update/rollout。其一次性feature tests随
  runtime退役；canonical contract owner继续保留。该证据不能解锁CEFD或新候选训练。
- residual CPU oracle覆盖zero memory identity、真实frame order、wrong-video exact-language、full48
  solve/application、negative-null、A/B response、memory-only exact-resume、六rank RNG、v8 deployment
  asset和strict paired row evidence；全仓`276 passed`。profile仍必须live验证真实A40 feature rank、
  task-local motion、fixed-action传递、wall和显存，CPU结果不能代替。

CPU门不要求batched Writer与single Writer逐元素相同，也不解释性能。

## 4. Historical single-A40 deployment seal（只作吞吐参照）

旧v3只改变训练objective而未改变部署图，所以当时可以继承legacy v6-prior seal；第37节在frozen fused
Program后新增video-keyed residual memory，**部署图已经改变**，不能再继承或重标旧seal。下列数据只用于
预估batch与吞吐，不解锁当前v8 residual evaluator：

- retained 32-request/1093-frame panel的batch8/16/32吞吐为
  `.911427/.905107/.906432 LoRA/s`，batch8峰值reserved约`12.8GB`且稳定；
- retained validation8×state0 vertical smoke为8 requests/8 LoRAs/8 rows，0 retry/failure/OOM/
  nonfinite/forbidden reads，Writer释放、source policy复用且GPU自然释放；
- 当时用于从两个retained roots重建seal的旧assembler与其runtime状态已退役，由Git和artifact保留；
  它不是当前可执行入口；
- 当前必须在mechanism profile通过后以新v8 residual adapter重新实测batch8/16/32和correct smoke；artifact
  必须绑定同目录run contract、actual clean commit、A40和正确family，不能因旧batch8曾最快而跳过实测。


## 5. Historical six-A40 profile provenance（已完成，不恢复）

旧whole-LoRA、ECP与Tangent依次完成了physical B20、B16+4容量裁决和balanced B10+10 gradient/resume/
throughput seal。B20与B16+4都在真实A40容量上失败；B10+10保持logical B20、480/480 unique queries和
task mean不变，并达到约21秒/full24 macro、43.3/47.1GB allocated/reserved。对应gradient assembler、
resume assembler、600/41-tensor ownership、6-rank RNG及NUMA/NCCL证据均已封存。

这些profile只解释为何当前audit继承B10+10、workers2、六rank与default allocator；它们不授权重跑v3
gradient profile、恢复profile checkpoint、复用旧auxiliary weights或启动训练。详细ECP和Tangent证据保留在
下列5.1/5.2；任何活动GPU命令只取Section 7。

### 5.1 Historical ECP resume evidence

clean pushed/frozen`5fbcb27`在live空闲`gpu02:0--5`完成retry1：resumed链fresh0→1再exact-resume1→3，
contiguous链独立fresh0→3。两run contracts逐字相同，所有invocation exit0；各3 metrics、macro1/3
checkpoints和completion，0 OOM/nonfinite/clip。contiguous/resumed合计step wall=`61.368/64.450s`，
input wait=`.203/1.153s`；steady-state macro3约`20.018/19.698s`且input wait约`.0006s`，故不测试workers4。
峰值allocated/reserved=`43,265,769,984/47,118,811,136` bytes。

cursor、checkpoint contract、6-rank RNG、scheduler、AMP、559 frozen Writer tensors精确相等；41个
trainable Writer与82个Adam moment tensors通过逐tensor`atol=2e-4, rtol=2e-3`，scientific metrics最大
tolerance ratio=`.233773`。macro3 Writer maxabs/relative-L2=`4.6033e-5/1.06393e-5`，差异L2仅为两步
真实update的`1.023%`；Adam maxabs=`2.6865e-6`，其`.007719` relative L2由近零moment分母放大。

原aggregate gate把Writer的同一阈值误用于Adam，造成工程false negative。只读v2比较器现要求Writer
global relative L2`≤.002`，并对Adam各moment要求symmetric norm ratio`≥.99`、cosine`≥.999`；raw
maxabs/relative-L2只作诊断，同时保留上述逐tensor科学门和所有exact语义门。训练kernel、dtype、
reduction、B10、objective及artifacts均未改变，也没有重跑GPU追逐逐元素一致。原roots重新assemble
通过并原样写入当时ECP config，profile/formal均为`sealed_from_live_a40_resume_profile_evidence`。
这些证据只说明B10/deferred-NCCL/checkpoint比较器曾按合同工作；不能解锁v3 formal，也不构成
closed-loop性能结论。

### 5.2 Historical tangent-tube gradient and resume seal

clean pushed/frozen`2616773`已完成一次macro49 gradient/whole-macro profile：root=
`runs/outputs/pi05_v6_tangent_tube_gradient_profile_macro49_r6_lb20_mb10_2616773_20260809`，24 tasks、
480/480 queries、8/8/8 negatives、最长105帧，wall/input wait=`21.53076/.60603s`，peak
allocated/reserved=`43,353,948,672/47,112,519,680` bytes，0 OOM/nonfinite。macro0双臂tube与delta
24/24 exact zero；唯一projection/ranking weights=`.00686480847114155/.010514453175708578`已由
assembler原样写回v3 config。相对ECP whole-macro raw wall仅增约`5.4%`、显存约`36/18MiB`，不启用cache。

clean pushed strict后继`c1bdcae`已在`gpu01:0,1,2|4,5,7`完成fresh0→1、same-root
exact-resume1→3和independent contiguous0→3。原自动chain在fresh结束后的inter-phase selected-GPU
preflight发现设备不再满足expected-idle合同并由live gate fail-close；重新live检查通过后分别启动剩余
两段。三段科学invocation均exit0，不能把原chain的exit1误写成训练失败，也不能写成整条chain自然exit0。

resumed/contiguous总step wall=`62.34061/61.95860s`，input wait=`.09366/.13220s`，peak
allocated/reserved=`43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite。assembler核对dynamic
anchor为41 tensors/`3,714,304` parameters且optimizer/checkpoint/deployment ownership全false；macro1/3
cursor、6-rank RNG、scheduler/AMP、559 frozen tensors和checkpoint contract语义相等。macro3 trainable
Writer maxabs/relative-L2=`8.5067e-6/1.14428e-6`，82个Adam moments的最低direction/norm门通过；scientific
metrics最大tolerance ratio=`.67790`。evidence已原样写回v3 config，当时的Section 6由此解锁，profile runtime
按状态机关闭且checkpoint永久弃用。

三步只作工程与早期机制证据：macro2有21/24 tasks把`a_correct`推向1，macro3却变为0/24；macro3
`a_correct=.71744`，task median `|a-1|≈.2799`，correct/negative orthogonal-relative-anchor median=
`.03158/.03173`且仅`10/24`、`6/24`低于`.03`，orthogonal-to-direction median约`60.98/61.2`，
`gradient_norm_before_clip≈1.45294>1`。quadratic tube在anchor处一阶梯度为零，首步正交漂移后才产生
回锚力，这是当时formal0→10必须直接证伪的结构风险。该formal和strict已完成，最新裁决见第0节与
第6.4节；不能从本段恢复训练。

## 6. Historical Tangent formal and truthful evaluation（已完成）

### 6.1 Completed baseline and training

Tangent已从clean pushed frozen`b308941`按预注册完成fresh0→10并自然停止；historical macro0=`134`来自同一
video schedule，未用历史143替代paired baseline。训练root、完整机制metrics、cursor/RNG和资源证据见6.4。
本段只作provenance，禁止从macro10续训。

### 6.2 Completed checkpoint decision

macro10已完成完整paired correct400，correct80仅由相同rows派生；native-family验证和cross-family historical
transition均通过。结果`131`虽处旧条件区间，但breadth5、directional/completion门失败，因此预注册决策是
不续25、不补六臂。旧checkpoint cadence全部关闭，不能恢复成活动命令。

### 6.3 Reusable rule for what counts as real improvement

方法改善至少需要absolute和任务分布证据，而不是一个suite换手。完整报告：

- aggregate/per-suite/per-task successes；breadth；
- 相对macro0和历史best的retained/gained/lost/both-fail；
- 相邻checkpoint成功集合Jaccard、union/intersection和single-envelope gap；
- paired correct-vs-same/wrong/shuffled/reversed/no-video；
- same-task不同teacher video方差；
- Core→Procedure→effective BA→fixed-action的条件传递；
- LoRA norm、stable rank、top energy、B-column cosine和跨target energy作为机制参照。

LoRA健康但分数低仍是失败；LoRA近rank1但分数提高不能仅因“不像SFT”淘汰。

### 6.4 Tangent formal结果与关闭状态

第6.1--6.3的预注册流程已完整执行。formal root、strict root和historical transition分别为：

```text
runs/outputs/pi05_v6_tangent_tube_formal_r6_lb20_mb10_b308941_20260810
runs/outputs/pi05_v6_tangent_tube_correct400_noreplacement_seed7_method_macro0010_b308941_20260810
runs/outputs/pi05_v6_tangent_tube_macro0010_historical_baseline_transition_b308941_20260810
```

macro10 correct=`131`落在条件续训区间，但breadth仅5、directional两臂`0/24`且completion`0/24`，
所以明确不满足续25门。完整六臂只在`≥144`触发，本点不运行。current config只保留formal result并
fail-closed；所有Section 6 launch语句现为已执行provenance，不是活动命令。

## 7. Post-result diagnosis and next single variable

当前最早接口已定位到`condition cotangent -> shared decoder/Adam -> next-condition output motion`。matched
audit已证明expert flow不是更好的监督，因此下一变量不改target，而改变更新输运：冻结historical v6的
600 tensors和高增益factor decoder，在其320×256 fused Program后加入zero-init、action-hidden-video-keyed
linear residual memory。full48显式Gram把24个correct真实functional cotangent与24个轮换counterfactual的
zero-motion约束写入同一memory；不增加PI05 forward、不使用expert/ranking cotangent。该design必须先证明
fixed feature对真实order敏感、Gram健康、correct motion保留、negative motion近零和macro0 exact identity。
CPU已证明algebra、topology、identity和fail-closed合同；尚待live mechanism profile证明真实A40 feature/
motion/action/吞吐，再由strict closed-loop裁决价值。

按以下顺序定位最早接口：

1. **evidence extraction**：correct/shuffle/reverse/wrong在Core前是否含任务和时序信息；
2. **Core/Procedure**：信息是否形成正确阶段/顺序，而非仅对低层异常敏感；
3. **compiler**：Procedure差异是否保留到policy-effective BA，是否被task-common方向压缩；
4. **factor heads/topology**：能量是否进入正确targets/rank coordinates而非同向高能量；
5. **functional/expert/ranking credit**：梯度是否对task可累积、是否由auxiliary绑到train expert流形；
6. **optimizer/task aggregation**：full24更新是否抵消并导致checkpoint轮换；
7. **closed-loop mismatch**：内部改善是否落到真实action和成功集合。

相应决策：

- absolute升、视频margin弱：下一变量只改counterfactual credit或Procedure temporal objective；
- margin升、absolute降：ranking伤害policy，不解释成“还没训够”；
- expert cosine升、held迁移降：减弱/重构train-expert流形约束，不转回online expert bank；
- compiler梯度健康但BA传递弱：修改compiler ownership/topology，不换video encoder；
- BA与action传递健康但任务继续换手：处理task aggregation/credit coexistence，不靠增rank/scale；
- same-task跨video方差成为最早限制：才设计固定K few-shot invariant aggregation；
- 所有下游目标与传递成立而上游无正确顺序语义：证据才允许解冻/重构Procedure或更早表示。

每个新设计必须明确继承旧方法已证明有效的组件、只改变哪个假设、预期影响哪些内部和closed-loop指标、
何种结果淘汰该假设。禁止东一榔头西一棒。

## 8. Evidence, Git, and retention

- canonical branch：`codex/bci-continuation`；正式root绑定包含该run contract的clean pushed commit。
- retired config：`configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`；Tangent与teacher-audit均已
  formal non-pass/fail-closed，不能作为活动入口。
- active config：`configs/pi05_v6_counterfactual_null_condition_kernel_program_residual_v1.json`；当前状态只允许
  `mechanism-profile`，formal和v8 deployment evaluation均等待各自live artifact seal。
- current design authority：`docs/action_forecast_writer_video_expert_manifold_design.md`第37节；实现和CPU seal
  已完成，但没有新A40或strict结果。
- training/evaluation entries为`scripts/train_v6_prior_writer.py`与`scripts/evaluate_pi05.py`；retired
  `--mode teacher-audit`及其owners/tests已经删除。
- fresh/profile要求HEAD等于当前remote authority；同root exact-resume固定原frozen commit且只要求它仍为
  当前authority ancestor。不得把remote后续文档提交写入旧run contract或冒充原训练commit。
- formal保留config、command/env、GPU topology、checkpoint schema、raw rows、aggregate、completion和必要
  mechanism analysis；不提交checkpoints/cache/data/binaries。
- failed `30b2ccf` smoke及其`batch_equivalence.json`保留为BF16诊断；不resume、不作为性能证据。
- 历史design/findings/progress/Git/formal artifacts保存所有已验证经验；当前authority不重复堆叠旧
  “下一步”。清理只移除已确认obsolete的活动path/worktree/temporary output。

## 9. Stop or ask conditions

只有以下情形暂停并请求owner：设备边界/ownership不明确、需要干扰他人进程、quota无法容纳、必须改变
split/信息墙/test边界、需要删除含唯一证据或不明ownership的资产、或同一实质阻塞在安全替代方案后仍
无法推进。普通科学负结果、一次OOM、工程bug或候选不达标不构成沟通门，应按上述证据链自主继续。
