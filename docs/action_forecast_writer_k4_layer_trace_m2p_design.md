# K4 Policy-Layer Trace M2P Writer

状态：2026-08-06已完成fresh0→200、四点strict rollout、winner五臂与全部内部分析，正式负裁决。
活动authority已切换为`action_forecast_writer_energy_preserving_layer_trace_design.md`；本方法
不得续训、warm-start或恢复逐`group × frequency`单位化的trace路径。

状态：2026-08-06 fresh0→200与四点strict correct400完成，等待single winner五臂及内部裁决。本文覆盖
Few-Shot Invariant-Program M2P的活动地位；旧方法及其checkpoint只保留为历史证据。

## 1. 决策

下一轮canonical Writer改为：

> exact task language + four action-hidden same-task videos
> -> frozen PI05 policy-layer video traces
> -> layer/token aligned M2P
> -> one complete rank-16 PI05 LoRA.

本方法不忽略视频，也不把task language作为LoRA value旁路。四条视频在一次forward中
联合生成一套LoRA；训练与部署均为K4。长期裁决仍只认同一single checkpoint strict paired
correct400，最低目标严格大于150。

## 2. K4上一轮已经证明什么、尚未证明什么

K4 Invariant-Program M2P的macro200机制分析已经排除“视频完全没被使用”和“LoRA写出
增益太小”：K4置换等价、zero-video identity成立，same-task set、leave-one-out、wrong、
shuffle与reverse差异均从Program穿到BA和fixed action；LoRA norm中位27.59，identity到
correct action变化中位.2543。失败发生在共享Writer参数的24-task credit coexistence：
最后50步full24 gradient retention中位仅.04326，四个block都接近1/24正交抵消。

但不能据此直接复制八个完整Writer experts。上一轮还存在更早的结构错配：它把冻结PI05
最终层3072维信号先压入固定随机128维，再要求一个仅在24 tasks上从fresh训练的256维M2P
凭608个抽象target/rank queries重新发现PI05的20组layer/module拓扑。给这种未对齐表示加
experts，只会把同一错配分桶，并把fresh参数扩大到约2.5亿；它不是根因修复。

## 3. 参考结构与本项目取舍

SHINE的关键不是“更大hypernetwork”，而是从冻结backbone的全部layer memory构造
semantic memory，再沿layer轴和parameter-token轴交替通信，最后直接reshape为目标参数。
Doc-to-LoRA同样显式保留layer/module/rank结构：从冻结模型activation，经Perceiver聚合后，
用per-layer/module输出组织LoRA，而不是把最终语义向量交给一个无policy拓扑的通用头。

EMBER采用这两个可迁移原则：

1. 视频信息先进入冻结source policy的真实action-expert层级，而非只取最终层随机投影；
2. M2P按policy layer与parameter slot双轴通信，并直接reshape为public LoRA。

不复制它们的文本重建目标、document setting或监督专用loss。EMBER的信息墙、action-hidden
视频、functional PI05 action supervision、K4部署接口和未来reward credit均保持不变。

## 4. Frozen policy-layer trace

每帧使用真实图像、exact task language与固定`t=1` Gaussian action suffix运行冻结PI05。
同一task/language和同一suffix另运行一次no-image baseline。捕获20组action-expert trace：

- action-in：suffix embedding在token轴的均值；
- 18个action-expert layers：每层q/v projection之前、input norm之后的50个suffix token均值；
- action-out：最终normalization之后的suffix hidden均值。

每组均减去对应no-image baseline，得到video-conditioned innovation。source policy与baseline
全部冻结且无梯度；task text只能影响video grounding，不能独立产生非零动态LoRA。

每条视频在时间轴投影到16个正交DCT-II basis。K4因此对每个policy group产生64个无序
trace tokens，整体shape为`[20,64,1024]`。四条video不带shot ordinal embedding；每条视频
内部保留相同16个temporal routes，因此交换四条video必须数值等价。当前数据stride5下
train video长度为16--105帧，held evaluation最短18帧，16项不会超过最短训练序列。

## 5. Layer-matched set reader

20个policy groups各有68个输出memory slots。对18个q/v layer group，68×1024恰好覆盖：

`qA + qB + vA + vB = 16*(1024+2048+1024+256) = 69,632`。

action-in/out各只需要16,896个元素，读取前17个slots；其余slots仍参加跨层通信，不产生
额外public参数。

每组的68个learned slot queries cross-attend该组K4×16 trace tokens。group、slot与
temporal identity只进入Q/K；V严格只来自baseline-subtracted video trace。q/k/v projection
跨20组共享，但20个1024×1024 group output projections独立，并以物理零初始化。由此step0
reader memory严格为零，且不同policy group从第一次非零更新起拥有独立写入子空间。

训练开放顺序固定：step1只允许20个group output projections更新；step2起开放reader的
共享q/k/v与后续M2P。该顺序只保证fresh identity下第一步存在可达输出，不冻结decoder、
不改变objective，也不引入macro阶段性trick。

## 6. Layer/token M2P与直接参数化

reader输出`[20,68,1024]`，随后经过四个zero-preserving blocks：

1. 沿20个policy groups做column attention/FFN；
2. 沿68个parameter slots做row attention/FFN；
3. 再做一次column block；
4. 再做一次row block。

每个block为8 heads、bias-free attention、2×FFN expansion，并以zero-preserving residual
初始化，不能在zero video时产生常量输出。column block让同一parameter slot跨layer/module
交换信息；row block让同一policy group内部组织A/B、q/v与rank coordinates。

输出不再经过target-specific小MLP。每个group直接flatten其68×1024 memory并按固定public
topology slice/reshape为qA/qB/vA/vB；action-in/out取前16,896个元素。动态结果加到唯一的
identity template A/zero B上，形成一套完整rank16 LoRA。预计trainable约58M：reader共享
q/k/v约3.15M、20个group outputs约20.97M、四个axis blocks约33.55M。

## 7. 训练与未来RL合同

- 从generic frozen source与functional identity fresh训练，不加载任何历史Writer；
- 24 train tasks，每task一组K4 action-hidden videos、一套LoRA、B20跨episode action queries；
- 先task内mean，再full24 raw equal mean；logical batch、query数与A100时期scientific
  contract不因A40或K4改变；
- 不加SFT reconstruction、rank、orthogonality、contrastive或video auxiliary loss；
- validation/test actions不产生梯度，也不用held functional loss选checkpoint；
- fresh0→200、每25保存，预注册strict correct400为50/100/150/200。

未来若打开RL，完全复用相同K4 trace reader、layer/token M2P和public LoRA接口，只把
functional cotangent替换为rollout reward credit。因此本方法解决的是condition-to-policy
参数生成本身，不是只适用于监督学习的训练技巧。

## 8. Canonical与retirement边界

- 原位替换`video_program.py`中的旧final-layer descriptor、`fewshot_m2p.py`中的32×256
  invariant program/608-token decoder以及对应model/config/checkpoint/task-gradient owner；
- 不保留可执行的旧K4 Writer、expert并行版本或兼容loader；历史由Git与frozen artifacts保存；
- 新checkpoint family必须拒载旧K4 family；
- 若layer-aligned方法仍出现分层task-gradient近1/24抵消，才允许根据group-wise/trace-wise
  证据设计稀疏共享或experts。不能在没有该证据前直接增加expert数量。

## 9. 聚焦验证与A40 profile

CPU/小规模合同只覆盖：20×64 trace shape、no-image subtraction、K4 permutation equality、
zero-video identity、68-slot exact slicing、38 public targets、step1/step2 gradient ownership、
source freeze、B20/K4 episode排斥、actual-world-size full24和旧family拒载。

正式训练前必须在live空闲A40上profile longest105、K4、B20/B2、16-frame source encoder
chunk、fresh0→1及exact-resume1→3；显式`NCCL_P2P_DISABLE=1`并保持六卡3+3 NUMA合同。
profile只验证finite/OOM/grad/resume，不作为性能证据。

## 10. 预注册裁决

行为报告strict correct400、breadth、per-task/suite、相邻gained/lost、union/intersection和
能力换手。single winner再做correct/same/wrong/shuffled/reversed、另K4 set、leave-one-out。

机制按最早接口检查：

1. 各PI05 layer trace是否有非零、非同向的video innovation；
2. reader后同task set与order反事实是否仍有界且穿过20 groups；
3. column/row M2P是否形成跨层共同但不坍缩的parameter memory；
4. effective BA与fixed action是否有足够leverage；
5. task-gradient cancellation是否相对旧K4得到实质改善；
6. closed-loop能力是否形成单checkpoint累积而非换手。

漂亮的layer geometry、LoRA rank、functional loss或多checkpoint envelope都不能替代
single-checkpoint strict correct400 `>150`。若失败，只根据上述最早失败接口设计下一轮。

## 11. 当前实现证据

- clean`a2c6d94`原位替换旧descriptor、invariant program、608-token decoder、config、
  checkpoint family与task-gradient owner；旧K4 executable config已退休，历史只在Git与
  frozen artifacts中保留。
- frozen descriptor逐帧捕获action-in、18层pre-q/v input-normalized suffix和final-normalized
  suffix，再减同language/suffix的zero-image baseline；16项DCT-II保持video内部顺序。
  action-in与最早层innovation允许为零，column M2P通过只进入Q/K的group/slot routes从后续
  video-aware layers向这些public targets传递value，route本身不能产生动态LoRA。
- `PolicyLayerTraceM2P`的reader Q/K使用group/slot/temporal address，V只用video trace；20个
  group output matrices物理零初始化。四个axis blocks同样仅在Q/K加入coordinate route，
  V/residual/FFN严格video-owned；direct slice覆盖全部38 targets的76个A/B tensors。
- 聚焦合同与全仓在BCI assets下`190 passed`，compileall、config/schema/family real load及
  diff check通过；step0、zero-video、K4 permutation、DCT order、direct tensor shape、自然
  step1→step2梯度可达、完整参数ownership和旧family拒载均闭合。ruff未安装，不作为项目门。
- 首个clean`89f5384`三步diagnostic错误地一次跑0→3，故本就不属于预注册profile；它进一步
  暴露真实parameterization bug：step1 group-output bootstrap后，axis FFN对极小memory做
  pre-LayerNorm，把任意非零幅度归一到O(1)，使loss从`.1504`跳到`58.93/96.82`，三步均clip，
  grad norm=`7.38e3/2.29e9/4.88e8`。显存和时间本身安全：peak reserved约20.48GB、每步
  34.38--34.76秒；该root只作诊断，永不resume或进入正式证据。
- clean`ed4f46e`根修不是调scale：axis attention仍可在Q/K使用normalized content与coordinate
  route，但FFN改为直接作用raw dynamic value，保证zero附近连续、幅度随video memory线性
  缩放。新增小幅度2×输入约2×输出合同，全仓更新为`191 passed`。下一次profile必须另起fresh
  root，严格fresh0→1后再exact-resume1→3。
- clean`44e248b`在live空闲`gpu01:0,1,2|4,5,7`完成严格fresh0→1与同root
  exact-resume1→3。三步loss=`.150377/.152815/.148507`，grad norm=
  `.001012/.000994/.001012`，0 clip/OOM/nonfinite；step2起reader与axis M2P都出现finite实际
  update。步时`34.708/34.655/34.609s`，peak allocated/reserved=
  `18,112,406,528/20,375,928,832` bytes。累计1,440 queries/288 videos，source trainable=0，
  六rank/3+3 NUMA/checkpoint/RNG/sampler/scheduler exact-resume闭合。profile权重永久弃用。
- launch commit`1b868ed`的独立fresh formal root已自然完成0→200：200 finite macros、96,000
  queries、19,200 K4 videos、8个every25 checkpoints、0 clip/OOM/nonfinite，wall=`7350.114s`，
  peak allocated/reserved=`18,096,154,112/20,478,689,280` bytes，validation/test action reads=0。
  最后50步full24 gradient retention/cosine/negative-pair中位为`.04573/.00400/.47464`，相对
  旧K4的`.04326/.00038/.49275`只有小幅改善；该证据只说明晚期共享credit cancellation仍可能
  是瓶颈。
- 50/100/150/200 strict correct400=`69/99/88/94`、breadth=`5/6/6/6`；相邻
  gained/lost=`42/12,28/39,28/22`，union/intersection=`145/37`。macro100为single winner，
  但只比旧K4同点94提高5，低于旧K4最终108与v6-fast143；150/200回落且继续换手。下一步固定
  macro100完成五臂与本节预注册内部接口分析，不能续训、warm-start或按loss改选checkpoint。
