# K4 Sparse Semantic-Expert Policy-Layer Trace Writer

状态：2026-08-07设计与实现authority。canonical实现、fresh schema/checkpoint family、独立
route artifact、聚焦CPU合同及A40 fresh/exact-resume profile已完成；formal launch尚未开始。
任何旧Writer checkpoint均不兼容且不得加载。

## 1. 决策

下一轮从functional identity fresh训练唯一一条canonical路径：

```text
exact task language + K4 action-hidden same-task videos
  -> frozen PI05 20-group DCT16 evidence-factorized traces
  -> frozen task-language semantic top2 route
  -> two complete independently-owned Trace Reader + axis-M2P experts
  -> equal-weight one policy-layer memory
  -> one complete public rank-16 LoRA.
```

这不是忽略视频或用language直接生成LoRA。language route只选择哪两个parameter owners读取
视频；全部动态value仍来自K4 baseline-subtracted action-expert traces。任一专家面对zero video
都必须输出zero memory，最终严格回到template-A/zero-B identity。

## 2. 当前最早失败接口

Evidence-Factorized macro200五臂为`84/85/66/83/78`，correct相对wrong的paired
gained/lost=`36/18,p=.01983`。视频task identity已真实进入closed loop；same/order没有形成
有效margin，但不能据此说视频被旁路。

8-task内部路径进一步闭合了表示与读出：correct相对wrong的
`physical/direction/attention/Reader/BA/action` relative-L2中位为
`.310/1.319/.647/.432/.620/.155`，shuffle/reverse也从direction穿到BA/action。direction与
physical Reader分支能量中位`14.65/27.52`，两支cosine`.510`且合并能量放大，不存在单分支
吞并；去掉evidence key仍产生非零BA/action反事实。完整LoRA norm`60.31`、stable rank`1.291`、
top singular energy`.847`，identity action effect`.373`，所以低增益、完全rank collapse和
Reader丢video都不是最早原因。

最后50步却在两个共享trainable owners同时回到近独立task平均：

| block | full24 retention median | pair cosine median | negative pair fraction |
| --- | ---: | ---: | ---: |
| shared policy-layer Reader | .05527 | .00954 | .47464 |
| shared four-axis M2P | .04650 | .00266 | .48007 |

这满足上一design预注册的expert开启条件：冻结表示之后的完整condition→policy Jacobian仍由
24 tasks硬共享。下一变量必须是parameter ownership，而不是再次改频谱、branch scalar、rank、
loss或训练时长。

## 3. 为什么不是恢复Direction Store

历史Semantic Direction Store只让最终factor heads拥有八个language-routed stores；其上游
Core/Program/compiler仍共享，Program差异到factor/BA被压到`.019/.032`，effective LoRA stable
rank约1，并最终只得129。它证明“只隔离最后一个输出头”不足，不是否定固定semantic route。

当前layer trace已经直接对齐20个PI05 policy groups，video差异能到Reader、BA和action。新方法
从**冻结descriptor之后的第一个trainable参数开始**隔离完整Reader与四个axis blocks，不复用旧
Core、Program、factor head、checkpoint或LoRA。这样每个task credit只进入其固定top2 experts，
不会先经过一个24-task共享composer再在末端分流。

## 4. Frozen semantic route

沿用已审计过的foundation-language address原则，但重新由当前source policy和train24 authority
生成独立route artifact：

1. 只保留authoritative task-span tokens，前置一个固定prompt-start token；
2. 用冻结PI05 PaliGemma text path取最后层task-token mean并L2 normalize，得到2048维anchor；
3. 只对24 train languages计算anchor mean并中心化；
4. seed7 deterministic farthest initialization + spherical k-means形成8个centers；
5. runtime按cosine固定选择两个不同centers，权重恒为`.5/.5`。

route artifact只含train24 input-derived mean、centers、task route与usage audit，不读teacher
action、video outcome、validation/test input或rollout。validation/test只应用已冻结映射。相同
exact language下的correct、same、wrong、shuffle与reverse必须得到完全相同route，因此行为
差异只能来自video value，不能由router换专家冒充video causality。

固定8×top2让24 train tasks产生48个assignments，平均每expert约6个task。正式实现前必须确认
每expert至少一个primary owner、top2 usage不塌缩；不按rollout选择center、K、top-k或route
weight。route是persistent buffer，无梯度、无temperature、无load-balance loss，也不随
checkpoint漂移。

## 5. 完整独立Trace experts

每个expert拥有一套完整的当前`PolicyLayerTraceM2P` trainable parameters：

- group/slot/temporal routes；
- query/key与evidence key；
- direction/physical value projections及vector fusion；
- 20个独立zero-initialized group outputs；
- 四个alternating policy-group/parameter-slot axis blocks。

冻结descriptor一次产生每condition的K4 physical traces。对expert`e`，只gather选择了`e`的
conditions，运行自己的完整Reader+axis M2P，再按固定route weight scatter-add：

```text
M(c) = 0.5 M_expert_i(traces(c)) + 0.5 M_expert_j(traces(c))
```

随后只有一个parameter-free direct slice/reshape把`20×68×1024` memory变为38 targets的76个
A/B tensors。没有expert LoRA ensemble、逐video LoRA平均或第二套policy；两个expert memory在
生成阶段先组成一套LoRA，rollout始终只执行这一套。

当前单expert trainable为`60,926,976`，8 experts预计`487,415,808`，实现后以真实enumeration
封存。Writer参数量已无owner上限。top2 conditional execution只产生两路activation；未选expert
对该condition梯度严格为零。每expert group outputs仍exact-zero，step0和zero-video identity
不变；step1先打开所选group outputs，step2起其Reader/axis自然可达。

实际实现enumeration封存为`487,415,808` trainable，其中完整Reader owners合计
`218,980,352`、四轴M2P owners合计`268,435,456`，共272个parameter tensors。独立生成的
`configs/pi05_sparse_semantic_expert_route_v1.json`与同一冻结source/train24原则的历史审计数值
逐元素一致；primary usage=`5/7/6/1/2/1/1/1`，top2 usage=`7/11/6/4/4/5/3/8`，没有expert塌缩。

## 6. AS、未来RL与信息墙

- K4、DCT16、direction/physical/evidence分解、38 targets、rank16、B20、full24 task equal、
  optimizer/scheduler和200-step四点合同均不变；
- Writer仍只读exact task language与K4 action-hidden videos；不读teacher action、state、reward、
  terminal、task ID、suite、filename、object pose或hidden normalization；
- AS functional cotangent和未来rollout reward cotangent都反向通过同一个top2 expert图，没有
  SFT-only auxiliary、route label、outcome gate或LIBERO heuristic；
- route只定义哪些parameter owners可共享credit，不改变每task loss/advantage含义；
- source policy、normalization与route全部冻结，任何旧Writer checkpoint都fresh incompatible。

因此它解决的是一般conditional hypernetwork中的稀疏参数共存：已验证的video representation
如何避免在一个全共享生成器里把相互独立的task credit平均掉，而不是监督微调专用trick。

## 7. 实现与聚焦验证

保持一个model、trainer、checkpoint family和evaluator，原位替换：

1. `video_program.py`增加冻结task anchor返回；video trace路径和数值不改；
2. `fewshot_m2p.py`增加fixed semantic router与top2 complete-expert gather/scatter owner，旧single
   expert class不作为活动runtime保留；
3. `model.py`传递condition anchor并只decode一次combined memory；
4. architecture/config/checkpoint/task-gradient owner改为fresh sparse-expert family；
5. 生成一个只依赖train24 language/source policy的route authority，config只引用该sealed artifact。

聚焦合同只覆盖：route冻结与usage、同language五条件route一致、未选expert梯度零、selected
expert step1/2可达、top2 gather/scatter与dense reference等价、K4 permutation、zero identity、
76 tensor shape、source freeze、actual-world-size gradient block ownership、fresh checkpoint与
exact resume。不增加大而泛的fallback或防御性hash。

## 8. A40 profile与formal裁决

实现后live比较`gpu01/gpu02`，只用最多6张空闲A40。先profile longest105、logical B20、policy
B2、16-frame chunk、fresh0→1及same-root exact-resume1→3；保持3+3 NUMA和显式
`NCCL_P2P_DISABLE=1`。若OOM，先减descriptor encoder chunk或做optimizer state分片，不改B20、
K4或task数。profile权重弃用。

formal从identity fresh0→200、每25保存，严格评50/100/150/200 correct400。只由single
checkpoint absolute、breadth、gained/lost与churn选winner，再做五臂和内部分析。内部必须同时
报告aggregate与expert-local task-gradient retention、route overlap、视频path、LoRA geometry和
fixed action；functional loss不选点。

若expert-local cancellation明显下降且single-checkpoint上升，说明parameter ownership方向成立；
若所有接口闭合但行为仍低，下一步才根据expert-local credit与closed-loop结果决定route粒度或
reward credit，不能用旧best warm-start、增加训练步数、挑video、调route或融合checkpoint救点。
最低目标仍是同一single checkpoint strict correct`>150/400`，达到后继续提高。

六卡A40 profile已按上述合同完成。`gpu01:0,1,2|4,5,7`、3+3 NUMA下fresh0→1再
same-root exact-resume1→3；三步wall=`48.051/48.732/48.535s`，loss=
`.150377/.152819/.148503`，grad norm=`.0002896/.0003028/.0002790`，0 clip/OOM/nonfinite。
step1八个Reader owners均可达，step2起八个experts的Reader/axis共16 blocks全部非零；source
trainable=0，累计1,440 queries/288 videos。峰值allocated/reserved=
`36,709,104,128/45,589,987,328` bytes，B20/B2/K4/16-frame chunk可运行但显存余量有限，formal
不得扩大batch/K或额外保留activation。profile checkpoints永久弃用。

## 9. 禁调项

本轮不改K、DCT、evidence公式、rank、LR、B20、AS objective、checkpoint schedule或source policy；
不加learned router、task-ID experts、one-task-one-expert、load-balance/contrast/rank/order loss、
scalar gate、global scale、reward、multi-LoRA、checkpoint融合或outcome-based route search。
