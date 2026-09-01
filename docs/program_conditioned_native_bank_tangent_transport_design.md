# Program-Conditioned Native-Bank Tangent Transport

状态：**active design**
owner裁决日期：2026-09-02
方法authority：`docs/expert_review_20260902_global_route_reassessment.md`第5--8节。

本文只把专家原文收敛为当前实现合同。若本文与专家原文发生方法冲突，以owner最新表达和专家原文为准；具体
dimension、数值solver、cache shard和microbatch属于实现选择，但不得改变数据、信息墙、梯度或Gate口径。

## 1. 目标与停止边界

当前唯一active路线是保留Stage0、G2 Natural Program、真实native X/Y、四类output bank、frame quadrature、exact signed
pooling、small-core canonicalization和首版carrier12+residual4，删除deployment主路径中的：

- absolute Program primal；
- fixed base + bounded correction；
- family scalar gate；
- shared/free native anchor；
- global/free summary token；
- binary operator route与effective-teacher训练目标。

已停止的只是`summary -> family-scalar gate -> event-additive anchor`函数类。PNBTT必须让Program只提供低维query，让当前
bank的真实candidate同时决定key-space几何和最终native value方向。

## 2. 唯一数据流

### Pass A：Natural Program

每条视频独立保序编码，经过boundary-anchored monotonic event alignment后，以固定`1/K`质量进入联合candidate measure。
保留唯一Program schema：

```text
P_lang    [38, 128]
P_scene   [38, 128]
P_process [E, 38, 128], E <= 8
rho       [E]
tau       [E, 2]
sigma     [E, 38, 128]
```

对target `j`、rank `r`、event `e`、side/type `s`，Program与固定target/rank/event/type结构身份形成query context `z[j,r,e,s]`，
共享query trunk只输出归一化低维query `q[j,r,e,s] in R^m`。Program不得有任何到`d_in/d_out` native vector的直接路径。

### Native candidates：key与真实value分离

Pass B继续捕获真实`X`以及`Y_abs/Y_adj/Y_init/Y_goal`。对每个candidate `v[j,s,n]`：

1. 在当前bank内做固定center/RMS归一化，仅供key projection使用；
2. target/side-aware、跨环境task共享的projection加允许的frame/horizon/probe/type/family metadata，形成低维key
   `k[j,s,n] in R^m`；
3. 最终pooled value始终是未被神经decoder替换的真实native `v[j,s,n]`。

metadata不得包含task ID、filename、pose、reward、outcome或teacher action。

### B0：current-bank key geometry

按event联合measure流式累计key mean与regularized covariance：

```text
k_bar = sum_n mu[e,n] * k[n]
C     = sum_n mu[e,n] * (k[n]-k_bar)(k[n]-k_bar)^T + lambda I
u     = C^(-1/2) q
```

`C^(-1/2)`必须在训练时保持对query/key projection可微；只在数值必要时对小矩阵使用FP64。B0不产生base primal或native
direction，语义上的持久输出只有`O(m^2)` sufficient statistics。当前E1/K1实现为尽快取得transport资格证据，会在单target内瞬时
物化candidate/key/event-mass张量并在targetwise回传后释放；进入E2的K2/K4前必须改成真正chunk-streaming moments/replay，不能把当前
瞬时张量实现误报成`O(m^2)`峰值内存。

### B1：唯一signed bank transport

重放同一bank，用whitened query对centered key打分，按weighted RMS规范化logit，并形成measure-aware antithetic
softmax：

```text
logit[n] = <u, k[n]-k_bar> / (weighted_rms + eps)
d        = sum_n (softmax_mu(+logit/theta) - softmax_mu(-logit/theta)) * v[n]
```

跨视频只先形成固定等质量的联合candidate measure，再执行一次pool；不得先为每条视频生成direction或LoRA后平均，也不得学习
video reliability。

### 唯一rank4 residual与rank16

event只通过归一化`rho`聚合。output types先各自用同一固定epsilon做无参数safe-RMS normalization，再按预注册的family/type固定权重
混合；该归一化规则可随当前bank数值确定尺度，但没有condition输出的可学习scalar，也不能重新引入family gate。每个target形成四对
`a_r/b_r`并物化rank4 residual，再复用small-core balanced SVD，与frozen carrier12严格
拼接为唯一38-target rank16 LoRA。

代码事实修正：历史`A_free`实际位于`gate * (candidate_anchor + free_anchor)`内部，且按task-local module、target、group、
rank/event与native width参数化；这一旧实现整体退役，不进入PNBTT。

## 3. 梯度与冻结合同

功能梯度必须沿以下完整路径传播：

```text
functional loss -> frozen policy with generated LoRA -> rank4 residual
-> real-value signed pool -> logits -> q/key/whitening -> Natural Program
```

- source/backbone与carrier始终冻结；native X/Y可作为只读cache常数；
- E1冻结Stage0和Natural Program，只训练同task跨correct/wrong共享的free query以及task-shared key projections；
- E2冻结G2 Natural Program，训练shared Program-to-query trunk与key projections；不存在task token或task ID；
- E3先冻结Natural Program，训练shared PNBTT；通过后同一fresh run进入E4并解冻全部Writer；
- E4同时保留component-init和同拓扑fully-random两个matched fresh候选；
- scale只在PNBTT方向成立且证据表明幅度受限时解冻；Action Meta保持关闭。

## 4. 不可协商的E0合同

只实现专家规定的十项hard tests，不增加流程性Gate：

1. zero native values时residual精确为零；
2. Program无native-value直达路径；
3. bank candidate permutation invariance；
4. video permutation invariance；
5. 固定Program换bank会改变geometry、weights与direction；
6. E1 correct/wrong/fit/held共享同一query；
7. 每condition只物化一套38-target rank16；
8. K视频固定等质量、无learned reliability；
9. deployment state无task/task-pair lookup参数；
10. Action Meta参数为零。

最小真实smoke还需证明chunked replay、finite differentiable solve、forward、functional gradient、rank4 materialization和policy真实加载。
完成这些直接合同后立即进入E1，不用通用重构、全量测试或额外文档阻塞实验。

## 5. E1--E4科学顺序

### E1：free-query transport capacity

- task1/wrong8与task93/wrong94；correct fit0/fit1/held，wrong fit0/fit1，K=1；
- 梯度只来自correct fit0/fit1与wrong fit0；held、wrong fit1、Panel-B、validation/test均零梯度；
- correct fit各自`>=.85`、held`>=.80`、wrong各自`<=.25`、min correct - max wrong`>=.50`、
  all-correct > all-wrong、near-bound不劣于既有合同且相邻checkpoint结论一致。

E1失败只说明transport函数类不足，应按bank tangent机制证据修正；不得责怪Natural Program或做无信息超参小扫。通过后fresh进入E2，
不加载task-local query或optimizer。

首个E1 single-key-chart formal已在macro70/110相邻一致`non_pass`。step110 task1 correct fit0/fit1/held为
`.641984/.660311/.622909`、wrong为`.122637/.186146`；task93 correct为`.713247/.737497/.685649`、wrong为
`.006121/.269427`。all-pairs与near-bound成立，主要缺口是correct/held与`.50` margin；70到110改善仅`.013--.037`。
因此当前不进入E2，也不追加训练或做普通超参扫。train-task `T=Cov(v,k)L^{-T}`功能梯度投影谱已经完成：`m=128`对应的1024
operator列没有截断有效谱，而q/v各side的功能梯度保留率与correct/wrong operator重合表明单一线性key坐标不足。按专家§5.10，
v2保持`m=128`、rank4及E1其余合同不变，使用四个family各自共享的input/output nonlinear trunk与family-side head，并对每个
target/side加rank16线性低秩residual projection。该target residual只调整key，不读取task、arm、filename或policy outcome；所有视频
仍先独立保序、集合阶段固定等质量。该v2 fresh formal也在macro70/110相邻一致`non_pass`：step110 task1 correct/held为
`.616630/.620958/.601512`、wrong为`.027332/.051458`；task93 correct/held为`.707775/.725727/.655429`、wrong为
`.047247/.223365`。它明显提高wrong specificity，但没有恢复absolute correct capacity，70到110改善仅`.0053--.0210`。
因此当前仍不进入E2；先在v2 step110上复跑同一train-only tangent spectrum。只有同构PNBTT task-local full-rank16 oracle明显优于
rank4 residual时才按专家§5.10重开carrier/task rank分配，不做width、LR、seed或更多key-chart小扫。该结论不涉及冻结的Natural Program。

### E2：真实frozen Natural Program到bank transport

- 数据与E1相同，增加matched K=1/2/4；
- 真实G2 Program作为query source；旧S1只把G2 canonical assignment用于event measure、主query来自fixed token，不能复用该语义；
- 除E1 Gate外，要求same-task held与K1/K2/K4保持，且full Program相对language-only与first+final/endpoints在task1、task93
  均有相邻checkpoint同号的paired功能增量。

E1通过而E2失败时启动专家次选B：移除固定Natural Program瓶颈，让language与ordered native-bank tokens直接产生signed measure；
不继续修补ECP G3局部接口。E2通过后不再增加task-local数学变体。

### E3：shared PNBTT早期资格

- gradient meta tasks：`1,8,9,32,52`；gradient target tasks：`72,73,75,93,94`；
- true task-held：meta task2、target task74；每task两fit视频、一held视频，K按固定分布采样1/2/4；
- held median recovery`>=.75`、held p10`>=.50`、held/fit`>=.80`、correct-wrong median margin`>=.10`，
  四family正贡献、task2/74为正、K与same-task保持、相邻checkpoint一致；direct functional为Gate。

E3是whole-Writer run的早期检查点，不是独立长期课程。E2通过而E3失败时保持同一PNBTT，先扩大到全部授权non-held tasks并直接joint；
扩大后仍只记忆gradient tasks才裁决是否进入B。

### E4：matched whole-Writer joint adjudication

- component-init与fully-random使用同一图、参数量、数据、loss、sampling、update数和checkpoint cadence；
- train24与审计后的non-held LIBERO-90产生梯度；validation8只选择，test8保持sealed；
- 唯一正式通过线为validation8 strict paired correct稳定`>145/400`，并满足相邻checkpoint、低churn、高breadth、四suite
  非零、Goal/Long、same-task retention以及full相对language/no-video/endpoints/wrong的必要增量；selected checkpoint冻结后才做
  shuffled/reversed。

只有matched两候选都未产生稳定闭环增量，才接近ECP或zero-interaction根本停止讨论。

## 6. 最小loss与扩容

只使用专家规定的三项：跨episode correct functional loss、wrong-video necessity loss、carrier/source preservation loss。E1的
preservation在task8/task94 Panel-A states上，以同一keyed flow time/noise直接计算generated与carrier真实policy action-velocity MSE，
并保留wrong-video adapter不能显著劣于carrier的单侧约束；不能用第二份teacher-action flow loss冒充`D_policy`。权重只从train tasks的
梯度量级预注册校准；不使用validation调权。不加入behavior-Gram、factor reconstruction、cosine、hidden separation、chart
alignment、effective teacher或polish loss。

on-policy仅在离线functional loss已产生稳定闭环增量后加入；Action Meta仅在base Writer有明确闭环增量后做matched control。

扩容只由机制证据触发：先看`Cov(v,k)`功能梯度投影谱；key dimension截断有效谱时才增大`m`；单chart跨family不足时才引入
family trunk加target-specific低秩projection；同构full-rank16 oracle明显优于rank4时才重开rank分配。不能靠width、seed、LR或
scale小扫替代接口判断。

## 7. 执行与证据

- 运行面所有权：`shared_compiler.py`只编排canonical PNBTT；`tangent_parameterization.py`拥有Program/free query与native key；
  `key_value_replay.py`拥有可微moments/whitening与antithetic replay；`tangent_transport.py`拥有joint-K native scopes和38-target
  transport；`pnbtt_runtime.py`、`pnbtt_tasklocal.py`、`pnbtt_training.py`、`pnbtt_evaluation.py`分别拥有authority/data、E1 arms与唯一
  rank16、Panel-A VJP和零梯度Panel-B。旧`bank_operator`只为既有frozen cache schema提供序列化兼容，live forward不得调用其spectral solve；
- 实现面保持一个canonical PNBTT runner/config schema；旧sealed configs只由Git和formal artifacts保存，不原地改status字节；
- E0 smoke后立即profile真实condition，再按live gpu01/gpu02状态选择1--6张A40；不固定两卡或35GB；
- formal run来自clean pushed commit的detached frozen worktree，保留exact command、环境、data/cache authority、checkpoint、raw rows、
  aggregate与completion；
- G2逐conditionProgram tensor不在现有cache中，E2必须从frozen G2 checkpoint重算；不得把fixed-token S1 cache误当真实Program cache；
- 文档、通用测试、清理与非必要合同不得阻塞可裁决实验，能并行的工作移到GPU等待期间。
