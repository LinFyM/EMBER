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

当前活动阶段是设计**same-video dynamic-baseline tangent completion**：同一exact language +
correct video同时产生frozen v6 anchor和当前Writer输出，只把增量中缺失的expert分量与
expert-orthogonal drift隔离。这一轮不改encoder/Core/Procedure/compiler topology、functional query、
negative schedule或deployment input，也不允许static/language bypass、B-only residual、第二套部署LoRA、
expert-bank deployment或global scale。新design封印前不启动GPU。

当前操作顺序：

1. v6-prior 0/10/25/50与ECP 0→10→25的formal、strict、机制诊断和GPU释放已封存；
2. 用ECP的“直接方向正确、闭环反向、正交漂移主导”证据完成dynamic baseline tangent数学设计；
3. 与历史SFT-Anchored Tangent-Basis、短LR/weight decay、decoder freeze和behavior distillation去重；
4. 在唯一canonical ECP vertical path上原位替换objective/schema，先用CPU dense oracle验证
   dynamic anchor、gauge-invariant residual、gradient、information wall和exact-resume；
5. clean push/frozen后只做一次A40 gradient/throughput profile；优先复用同memories的小decoder forward，不为低位
   数值一致降batch/并行度；
6. formal仍从historical v6 macro400 fresh开始，及时跑同schedule strict correct400；若限制正交漂移后
   仍不超macro0，干净证伪expert-component completion并转policy-output behavior distillation。

不得从下文自行跳到later stage，也不得从历史文档恢复已退役命令。

## 1. Fixed scientific contract

- 方法：one-shot v6-Initialized same-video dynamic-baseline tangent completion Writer（设计中，
  ECP已退役）。
- 输入：exact task language + exactly one action-hidden raw teacher video。
- 视频是唯一dynamic value；无language-only LoRA bypass、expert-bank部署、multi-video/LoRA/checkpoint
  平均或融合。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing与paired evaluator固定。
- historical v6-fast macro400只作load-only Writer初始化；冻结encoder/Core/transition/Procedure，只训练
  compiler+factor heads；全新optimizer/scheduler/sampler/RNG。
- train24 task-complete、每task logical B20跨episodequeries、每visit一条correct video、24-task等权、一次flat
  all-reduce。
- 新objective只能在同video frozen-v6 effective BA baseline上限制增量，同时保留positive
  functional和bounded temporal/wrong ranking；不含global norm attraction、whole-LoRA cosine、static anchor
  或parameter-distance代理。auxiliary weight只由预注册train24 gradient profile选择一次，不做held sweep。
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
- config只继承未改变推理图的retained evaluation throughput seal；ECP gradient、aux weights和resume
  profile已全部重置，必须从v2 clean frozen lineage重新实证后才能解锁formal。任何status-only、旧v1
  evidence或stale config仍fail-closed。

CPU门不要求batched Writer与single Writer逐元素相同，也不解释性能。

## 4. Single-A40 throughput and vertical smoke

### 4.1 Live preflight

同时查询`gpu01/gpu02`，记录每张卡index/UUID、memory、utilization、process和owner；只选一张完全空闲
A40。检查`/data1`个人quota、当前项目用量和本次cache/log峰值。不得查询后长时间等待再沿用旧快照；
真正launch前再做一次短确认。

### 4.2 Writer batch sweep

使用historical macro400、真实validation视频和包含最长/异长样本的固定请求集。canonical入口是现有
`scripts/evaluate_pi05.py profile-writer-generation`子命令，不另建平行evaluator。batch候选至少覆盖
`8,16,32`，若32仍有明显显存和吞吐空间再在新profile加入更高点。所有候选必须处理同一个由最大候选
确定的longest-first request panel、相同entry IDs和相同总sampled frames；较小候选把该panel切成多个
forward，不能各自只测不同长度的prefix。每点最低必须记录：

- 一次warmup、至少两次真实完整video→Writer→LoRA→native D2H的repeat wall与LoRAs/s；
- actual forward batch分组、完整固定entry IDs、sampled-frame counts/total和panel最长视频覆盖；
- peak allocated/reserved、device total、显存余量与稳定性；
- 76 tensor shape/dtype/finite、0 forbidden reads和模型释放证据。

同一模型一次load后完成候选点，profile输出保留为evaluation-seal evidence root，任何OOM候选不重复盲试。选择吞吐最高且
最长真实batch连续运行稳定、保留必要headroom的batch；不是选择数值最接近single的batch。如果候选吞吐
接近、重复波动大或GPU未吃满而原因不清，再补encoder/compiler/D2H分段计时、allocator retry、data wait和
连续GPU utilization trace；第一轮不为尚未出现的瓶颈增加同步和instrumentation。

### 4.3 End-to-end smoke

在选定batch上从fresh root运行validation8×state0、correct、seed7、without-replacement：8个唯一video
→8套完整LoRA→cache→释放Writer/encoder/tokenizer→原位复用同一source policy→8个rollouts。要求：

- 8 requests/generated/cache/rows，0 retry/failure/OOM/nonfinite；
- teacher action/proprio/reward/task-ID等forbidden reads均0；
- Writer modules释放，source policy不reload；
- cache native dtype/bytes正确；进程退出后GPU自然释放。

smoke的success count只作执行信息，不是性能证据。通过后把device、commit、root、batch、wall、peak、
release/reuse和错误计数写回config，状态改为gradient-profile ready。不得人工手填evidence：必须用
`assemble_v6_prior_evaluation_smoke_evidence(profile_root=..., vertical_root=...)`从两个retained roots重建
并复验seal。

### 4.4 Completed evidence

clean frozen `ded0c80`在`gpu02:0`完成同一32-request/1093-frame panel：
batch8/16/32=`.911427/.905107/.906432 LoRA/s`，三者峰值reserved约`12.8GB`且稳定，故按真实吞吐
选择batch8。fresh vertical smoke生成8套native LoRA并完成8 rows，single attempt、0 retry/failure/
OOM/nonfinite/forbidden reads；Writer释放、source policy复用/no-reload，GPU自然回到0MiB。总wall
`325.540s`，其中rollout window `196.816s`；`4/8` success仅作execution信息。两个retained roots已
经assembler写入config，Section 5是当前唯一GPU入口。

## 5. Six-A40 gradient, resume, and throughput profile

重新live比较两节点，最多选择6张空闲A40；按实际设备建立3+3或合同允许的NUMA physical/local rank
拓扑，显式`NCCL_P2P_DISABLE=1`并使用repository deferred-NCCL launcher。

profile固定train24 macro49，覆盖24×B20=480 unique跨episodequeries并包含最长105 sampled-frame video：

已完成的容量诊断：clean frozen`a17805c`在当时空闲`gpu01:0,1,2,4,5,7`运行physical B20。默认allocator
OOM时allocated=`42.29GiB`、reserved-unallocated=`1.29GiB`；`expandable_segments:True`把后者降到约
`157MiB`，但active allocated升至`43.43GiB`且仍无法再分配`606MiB`。所以不得再做allocator盲重试；两个
root不resume、不合并。当前B10继续使用default allocator；失败retry只作诊断，不固化为科学或runtime门。

随后clean frozen`eddba96`的B16+4在同一`gpu01:0,1,2,4,5,7` 3+3 NUMA拓扑完整进入start，六rank第一条
functional eager-attention均在申请`254MiB`时OOM：allocated=`42.49GiB`、reserved-unallocated=
`1.25GiB`、free=`235.31MiB`。因此B16没有whole-step吞吐点；当前直接运行balanced B10+10，不再做
allocator retry、A-B-A或宽batch sweep。

旧whole-LoRA clean frozen`9c814ff`的balanced B10+10已在同一拓扑完成：wall=`21.0951s`、input wait=`.0763s`
（`.36%`）、peak allocated/reserved=`43,305,942,016/47,093,645,312` bytes、0 OOM/nonfinite；assembler
复验24 tasks、480/480 queries、最长105帧和完整provenance。当时推荐expert/ranking weights=
`.008355172068998324/.28570466890490887`；它们只属于已退役objective，ECP不得继承。B10吞吐图可以复用，
但projection/ranking gradient和weights必须从v2重新profile；不扫workers4或更小microbatch。

1. 分别测positive、projection、ranking在compiler和factor heads的未加权gradient norm；
2. 一次性选择`lambda_projection/lambda_rank`，使每个auxiliary在两个trainable blocks都不超过positive的
   `.25`；若初始化已满足而梯度近零，不人为放大；
3. retained artifact记录完整宏步wall、DataLoader input wait和peak allocated/reserved；不为细分阶段
   在热路径插入额外CUDA同步。只有这些证据定位不出真实瓶颈时，才用一次性disposable profiler细分；
4. 运行同logical B20、同panel-keyed randomness和同六卡拓扑的balanced B10+10；记录完整macro wall、
   input wait、peak allocated/reserved和异常计数。B16已容量失败，因此B10成功即成为当前A40可行点；
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

### 5.1 Completed resume evidence

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
通过并原样写入config，profile/formal均为`sealed_from_live_a40_resume_profile_evidence`。这些证据只
授权Section 6，不构成closed-loop性能结论。

## 6. Formal training and truthful evaluation

### 6.1 Baseline and training

另建clean pushed frozen worktree、fresh output root和formal launch contract。method macro0是同一当前
video schedule下的historical macro400 load-only状态；不能仅引用旧143替代当前paired baseline。

第一段只fresh训练0→10并自然停止，保存10；不在未看到真实macro10行为前自动跑到25/50。训练期间持续记录：

- positive/expert/ranking loss和unweighted component gradient；
- per-task loss、effective cosine/norm、correct-negative margins；
- compiler/factor task-gradient cosine、full24 retention、clip/nonfinite和step wall；
- video schedule、counterfactual counts、sampler cursor和每rank RNG。

loss下降不能延迟或替代rollout。若出现非finite、明显全task退化或合同破坏，立即停；普通loss波动不作
科学淘汰。

### 6.2 Checkpoint cadence

- macro10直接跑完整paired correct400；correct80只从相同400 rows的`state<10`子集派生，不另启动rollout，
  也不能选winner或代表真实水平。
- 历史current-schedule macro0=`134`与ECP macro10分别按native family验证，再由显式标注cross-family的
  historical-baseline transition逐row核对共同state/RNG/language/video。不得复制或重标旧rows，也不得
  把两点伪装成ECP checkpoint curve；只有后续确需同-family完整曲线时才补跑ECP-v2 macro0。
- 每点与macro0、历史143、v5.2-old和v6交叉recipe逐task比较：per-suite、breadth、gained/lost、
  union/intersection、capability churn和视频ordinal依赖。
- macro10过门后才从同一root exact-resume到25；只有多个task共同上升并超过134才继续50/100（必要时
  200），每个决策点及时correct400；不因单点低分180度转向，也不因loss好看无限续训。
- 任何single checkpoint correct严格`>150/400`，或成为有可信共同提升的当前winner时，运行完整paired
  correct/same/wrong/shuffled/reversed/no-video。达标后仍继续验证更高性能、breadth和稳定性。

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

## 7. Post-result diagnosis and next single variable

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
- current config：`configs/pi05_v6_ecp_policy_effective_writer_v2.json`。
- current design：`docs/action_forecast_writer_video_expert_manifold_design.md`第34节。
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
