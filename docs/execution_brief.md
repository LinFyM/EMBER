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

当前没有运行中的EMBER GPU任务，也没有已授权继续训练的Writer。第36节matched Expert-Flow Teacher
Viability Audit是唯一下一动作：不更新参数，在完全相同的train24 B20/noise/time上比较step2000 task
expert、historical macro0和tangent macro10的PI05 flow velocity相对真实`u_t`的误差，并比较候选
distillation gradient与positive/completion/ranking的norm/cosine。只有task expert在大多数tasks上是更好
teacher且gradient非冗余，才允许实现CEFD；否则转向直接约束shared update geometry的结构候选。

当前操作顺序：

1. v6-prior whole-LoRA、ECP与Tangent三条连续单变量实验的formal/strict/机制诊断均已封存并释放GPU；
2. Tangent只解决了局部半径，没有解决共享decoder Jacobian把更新旋进expert方向的问题；strict同时降到
   `131`，所以禁止靠续训、扫权重或硬加大auxiliary掩盖首个失效接口；
3. 先完成matched flow-teacher audit；该诊断不得更新参数、不得读取validation/test action、不得改变
   deployment graph，也不得因task9 expert closed-loop为0而删除或降权；
4. audit若通过，首版CEFD只增加correct arm的一次stop-gradient expert PI05 flow forward，复用student
   positive forward和同一B20/noise/time，只比较真实7维action；negative仍保持现有bounded ranking；
5. audit若失败，不实现CEFD，转向能在输出层直接保证condition-specific structured update的单变量设计；
6. 任一新候选仍须clean push/frozen、live preflight、吞吐profile、短正式训练和及时strict correct400，
   不从Tangent checkpoint resume。

不得从下文自行跳到later stage，也不得从历史文档恢复已退役命令。

## 1. Fixed scientific contract

- 方法：one-shot Video-Conditioned Expert-Manifold总路线；whole-LoRA/ECP/Tangent均已退役，当前只执行
  matched Expert-Flow Teacher Viability Audit，尚无active正式训练方法。
- 输入：exact task language + exactly one action-hidden raw teacher video。
- 视频是唯一dynamic value；无language-only LoRA bypass、expert-bank部署、multi-video/LoRA/checkpoint
  平均或融合。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing与paired evaluator固定。
- historical v6-fast macro400仍是下一候选唯一允许的load-only Writer初始化；若audit授权CEFD，冻结
  encoder/Core/transition/Procedure，只训练compiler+factor heads，并创建全新optimizer/scheduler/sampler/RNG。
- train24 task-complete、每task logical B20跨episodequeries、每visit一条correct video、24-task等权、一次flat
  all-reduce。
- 不恢复global norm attraction、whole-LoRA cosine、ECP completion、Tangent dynamic anchor或parameter-
  distance代理。CEFD若获授权，只能蒸馏同B20/noise/time上的task-expert PI05 flow velocity，同时保留
  positive functional和bounded video ranking；auxiliary weight只由一次train24 gradient profile选择，
  不做held sweep。
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

保留FP32 RMSNorm/softmax/ROPE/image normalization和policy-effective reduction，除非profile证明它们是
显著瓶颈且降低精度不伤真实闭环；吞吐优先不是盲目改变模型数学。

## 3. CPU seal before GPU

GPU前一次性要求：

- `git diff --check`和聚焦/全仓CPU tests通过；
- historical 600-tensor Writer strict-load，frozen/trainable ownership为
  `7,060,992/3,714,304` parameters；
- validation8 real-asset inspect和CLI prepare产生正确one-shot requests，部署expert-bank reads=`0`；
- no-video不读frames并返回template-A/zero-B identity；wrong/shuffle/reverse只改变允许的视频输入；
- native LoRA storage descriptor从checkpoint metadata贯穿run contract与cache write/load；resident policy的
  destination dtype由已验证的同一template决定，正常路径不发生额外转换，不在每次replan加dtype扫描；
- 2-worker prefetched sampler与serial、prefix+resume逐row一致；
- retired Tangent config只继承未改变推理图的retained evaluation throughput seal；ECP gradient、aux
  weights和resume profile没有被复用；v3 gradient与resume evidence曾从各自clean frozen lineage重新
  实证并原样封存，formal0→10完成后已切为non-pass并关闭runtime。任何status-only、旧family evidence或
  stale config仍fail-closed。

CPU门不要求batched Writer与single Writer逐元素相同，也不解释性能。

## 4. Historical single-A40 deployment seal（仅继承，不重跑）

v3只改变训练objective、trainable-state恢复范围和训练期dynamic anchor；部署仍是同一historical v6
Writer图、一次读取exact language+一条action-hidden video并生成一套完整LoRA。因而当前config只读继承
clean frozen `ded0c80`的legacy v6-prior v1部署seal，不把它重标为ECP，也不再启动single-A40 sweep或
vertical smoke。

- retained 32-request/1093-frame panel的batch8/16/32吞吐为
  `.911427/.905107/.906432 LoRA/s`，batch8峰值reserved约`12.8GB`且稳定；
- retained validation8×state0 vertical smoke为8 requests/8 LoRAs/8 rows，0 retry/failure/OOM/
  nonfinite/forbidden reads，Writer释放、source policy复用且GPU自然释放；
- 当时用于从两个retained roots重建seal的旧assembler与其runtime状态已退役，由Git和artifact保留；
  它不是当前可执行入口；
- 若以后实际改变部署图或LoRA生成批处理，再以新design/schema重新授权single-A40 profile；否则
  Section 5的六卡v3 gradient/whole-macro profile是唯一当前GPU入口。


## 5. Six-A40 gradient, resume, and throughput profile

重新live比较两节点，最多选择6张空闲A40；按实际设备建立3+3或合同允许的NUMA physical/local rank
拓扑，显式`NCCL_P2P_DISABLE=1`并使用repository deferred-NCCL launcher。

profile固定train24 macro49，覆盖24×B20=480 unique跨episodequeries并包含最长105 sampled-frame video：

已完成的容量诊断：clean frozen`a17805c`在当时空闲`gpu01:0,1,2,4,5,7`运行physical B20。默认allocator
OOM时allocated=`42.29GiB`、reserved-unallocated=`1.29GiB`；`expandable_segments:True`把后者降到约
`157MiB`，但active allocated升至`43.43GiB`且仍无法再分配`606MiB`。所以不得再做allocator盲重试；两个
root不resume、不合并。新方法仍以B10+10/default allocator为首个高吞吐候选；失败retry只作诊断，
不固化为科学或runtime门。

随后clean frozen`eddba96`的B16+4在同一`gpu01:0,1,2,4,5,7` 3+3 NUMA拓扑完整进入start，六rank第一条
functional eager-attention均在申请`254MiB`时OOM：allocated=`42.49GiB`、reserved-unallocated=
`1.25GiB`、free=`235.31MiB`。因此B16没有whole-step吞吐点；当前直接运行balanced B10+10，不再做
allocator retry、A-B-A或宽batch sweep。

旧whole-LoRA clean frozen`9c814ff`的balanced B10+10已在同一拓扑完成：wall=`21.0951s`、input wait=`.0763s`
（`.36%`）、peak allocated/reserved=`43,305,942,016/47,093,645,312` bytes、0 OOM/nonfinite；assembler
复验24 tasks、480/480 queries、最长105帧和完整provenance。当时推荐expert/ranking weights=
`.008355172068998324/.28570466890490887`；它们只属于已退役whole-LoRA objective。ECP随后在同一B10图上
封存`.006883349605446485/.010514451404229894`，也只作新方法macro0 gradient identity的预期参照，
不得直接写入v3 config。dynamic anchor每臂只增加一个小decoder forward，不重跑video encoder或B20 policy；
仍须一次真实v3 gradient/whole-macro profile确认权重、wall和VRAM，不扫workers4或更小microbatch。

1. 分别测positive、projection、ranking在compiler和factor heads的未加权gradient norm；
2. 一次性选择`lambda_projection/lambda_rank`，使每个auxiliary在两个trainable blocks都不超过positive的
   `.25`；若初始化已满足而梯度近零，不人为放大；
3. retained artifact记录完整宏步wall、DataLoader input wait和peak allocated/reserved；不为细分阶段
   在热路径插入额外CUDA同步。只有这些证据定位不出真实瓶颈时，才用一次性disposable profiler细分；
4. 运行同logical B20、同panel-keyed randomness和同六卡拓扑的balanced B10+10；记录完整macro wall、
   input wait、peak allocated/reserved和异常计数。B16已容量失败，因此不为新方法重复宽batch sweep；
   若B10+10因新增decoder不可行，先定位同时存活tensor而不是降低scientific batch；只有不可避免OOM或
   含构建成本的end-to-end实测更快才启用full-condition anchor cache；
5. 用丢弃型profile权重完成fresh0→1、same-root exact-resume1→3、独立contiguous0→3。要求scientific
   metrics、cursor、Writer/RNG和optimizer/scheduler语义一致；允许正常并行低位roundoff，不要求不同
   reduction schedule逐bit相同；
6. profile checkpoint永久禁止warm-start formal。

两个verifier现已实现并通过全仓CPU回归。gradient assembler从`gradient_profile.json`、completion、
invocation和run contract重算推荐权重，精确重建24-task deterministic video/negative panel、canonical
config、clean pushed Git及六卡拓扑；resume assembler还调用只读checkpoint inspector，核对600 Writer
tensors、41 trainable tensors、6-rank RNG、Adam moments、scheduler/AMP、cursor、Git phase ancestry和
scientific tolerance。task/frame provenance还会回查frozen target manifest、HDF5 path/bytes和对应demo的
真实frame metadata。二者不做hash，也不能被人工status、stale tracked config或复制到外部的config替代。

工程故障按rank/device/process-group/CUDA/I/O/NUMA层定位。不得用加timeout、关watchdog、减少logical B20
或盲重试掩盖问题；physical microbatch是保持科学batch的容量实现，不属于减少B20。

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
metrics最大tolerance ratio=`.67790`。evidence已原样写回v3 config，Section 6现已解锁，profile runtime
按状态机关闭且checkpoint永久弃用。

三步只作工程与早期机制证据：macro2有21/24 tasks把`a_correct`推向1，macro3却变为0/24；macro3
`a_correct=.71744`，task median `|a-1|≈.2799`，correct/negative orthogonal-relative-anchor median=
`.03158/.03173`且仅`10/24`、`6/24`低于`.03`，orthogonal-to-direction median约`60.98/61.2`，
`gradient_norm_before_clip≈1.45294>1`。quadratic tube在anchor处一阶梯度为零，首步正交漂移后才产生
回锚力，这是当时formal0→10必须直接证伪的结构风险。该formal和strict已完成，最新裁决见第0节与
第6.4节；不能从本段恢复训练。

## 6. Formal training and truthful evaluation

### 6.1 Baseline and training

另建clean pushed frozen worktree、fresh output root和formal launch contract。method macro0是同一当前
video schedule下的historical macro400 load-only状态；不能仅引用旧143替代当前paired baseline。

第一段只fresh训练0→10并自然停止，保存10；不在未看到真实macro10行为前自动跑到25/50。训练期间持续记录：

- positive/projection+tube/ranking loss和unweighted component gradient；
- per-task student/anchor coefficient与norm、signed parallel delta、orthogonal delta、relative tube和
  correct-negative margins；
- full24 gradient norm、clip/nonfinite、input wait、step wall和peak VRAM；
- video schedule、counterfactual counts、sampler cursor和每rank RNG。

loss下降不能延迟或替代rollout。只有OOM/nonfinite、信息墙、拓扑、数据或run/checkpoint合同失效才在
macro10前立即abort；普通loss波动、三步机制恶化或全task同相振荡不能替代已注册的macro10 strict400，
除非产物已失去科学有效性。

### 6.2 Checkpoint cadence

- macro10直接跑完整paired correct400；correct80只从相同400 rows的`state<10`子集派生，不另启动rollout，
  也不能选winner或代表真实水平。
- 历史current-schedule macro0=`134`与tangent candidate分别按native family验证，再由显式标注cross-family
  的historical-baseline transition逐row核对共同state/RNG/language/video。不得复制或重标旧rows，也不得
  把两点伪装成同-family checkpoint curve。
- 每点与macro0、历史143、v5.2-old和v6交叉recipe逐task比较：per-suite、breadth、gained/lost、
  union/intersection、capability churn和视频ordinal依赖。
- macro10 `≤129`且多task净损失即停；`130--134`只有tube门成立、breadth`≥6`、churn`≤35`且最近3个
  macro projection方向斜率`≥0`才从同一root exact-resume到25。macro25必须`≥135`且至少3 tasks、
  2 suites净正增才继续50；任何100扩展需先更新sealed config。
- single checkpoint首次`≥144/400`即运行完整paired correct/same/wrong/shuffled/reversed/no-video；若之后
  不同winner首次`≥151`，再对实际goal winner重跑六臂。达标后仍继续验证更高性能、breadth和稳定性。

### 6.3 What counts as real improvement

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

当前最早接口已定位到`LoRA cotangent -> shared decoder/Adam -> next-condition output motion`；但在改结构
前必须先判定expert policy flow是否提供现有positive functional之外的新监督。第36节matched audit是当前
唯一动作：零update、零rollout，train24同B20/noise/time比较expert/macro0/tangent10 flow error，并计算
CEFD gradient相对positive/completion/ranking span的残差。teacher质量至少18/24 tasks、3 suites更优，且
compiler/factor残差比例都`≥.25`才授权CEFD。

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
- retired result config：`configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`，formal runtime已关闭。
- current design：`docs/action_forecast_writer_video_expert_manifold_design.md`第36节。
- current training/eval entry：`scripts/train_v6_prior_writer.py`、`scripts/evaluate_pi05.py`。
- formal保留config、command/env、GPU topology、checkpoint schema、raw rows、aggregate、completion和必要
  mechanism analysis；不提交checkpoints/cache/data/binaries。
- failed `30b2ccf` smoke及其`batch_equivalence.json`保留为BF16诊断；不resume、不作为性能证据。
- 历史design/findings/progress/Git/formal artifacts保存所有已验证经验；当前authority不重复堆叠旧
  “下一步”。清理只移除已确认obsolete的活动path/worktree/temporary output。

## 9. Stop or ask conditions

只有以下情形暂停并请求owner：设备边界/ownership不明确、需要干扰他人进程、quota无法容纳、必须改变
split/信息墙/test边界、需要删除含唯一证据或不明ownership的资产、或同一实质阻塞在安全替代方案后仍
无法推进。普通科学负结果、一次OOM、工程bug或候选不达标不构成沟通门，应按上述证据链自主继续。
