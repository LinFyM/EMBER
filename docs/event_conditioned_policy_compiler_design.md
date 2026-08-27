# EMBER ECP Native-Factor Compiler

状态：2026-08-27第二次专家复核、owner最新裁决与F3连续formal/机制证据后更新的active architecture contract。第一次专家审查锁定
`main@7ab5a04`并建立Native-Factor主线；第二次专家审查锁定`main@ed2883b`及其可达历史，针对G3跨视频dual/score旋转给出
bank-conditioned两阶段Pass B修正。本文是当前唯一架构依据。

两次专家原文分别逐字保存于`docs/expert_review_20260824_native_factor.md`和
`docs/expert_review_20260826_bank_conditioned_native_factor.md`。本文是将专家原文与owner后续裁决转成可执行合同的解释层，
不替代原文；任何疑似曲解或冲突先核对原文，再按owner最新明确表达修正。第二次专家建议Final默认从通过Gate的组件初始化；owner
最新明确补充，整套Writer完全随机初始化并直接端到端fresh训练必须保留为Final正式可选项，G1--G3不构成强制训练课程。

owner后续明确取消专家原文中的阶段工期估计、固定修正次数、结构版本上限和训练轮数上限。本文保留专家的Gate与失败定位逻辑，
但不把时间或尝试次数当停止条件；修正必须由新证据驱动，整体推进应在保质前提下尽可能快，顺利时力争数天内完成完整架构实现
并推进关键Gate。

人工process数据、神经`q_pi`、fixed effect-code realizer、fit-task fixed span、PECS/GOMQ/v24和并行旧Writer均不属于本路线。

## 1. 核心裁决

ECP目标不变：exact language和`K`条action-hidden ordered videos在rollout前一次性生成唯一一套38-target rank16 LoRA。

新的最深假设是：自然视频在冻结PI0.5各LoRA目标中诱发的原生input/output向量，可以作为task-specific低秩参数基底；Video
Program不再凭空生成高维因子，而只决定从这些原生向量中选择什么、怎样有符号地组合和缩放。

```text
exact language + K ordered action-hidden videos
        |
        +-- Pass A: frozen PI0.5 native observation
        |       -> owner-specific language / scene / ordered-event Program
        |
        +-- Pass B0a: same frames, per-event candidate-feature statistics
                -> detached current-bank feature gauge
        +-- Pass B0b: stable task anchor + whitened candidate content
                -> native anchors and regularized native-bank query solve
        +-- Pass B1: replay same native bank
                -> Program-conditioned exact signed pooling
                -> current first implementation: task-specific rank4 residual
                -> concatenate frozen shared rank12 carrier
                -> one complete 38-target rank16 LoRA
```

Pass A与Pass B读取同一个冻结backbone，并共享owner、event、video assignment。Pass A回答视频表达什么目标、场景和过程；Pass B0a
先在当前video bank中建立每个canonical event的candidate-feature gauge，Pass B0b再以same-task稳定的language anchor读取白化后的
candidate content、形成native anchors并求解native-bank-conditioned query；Pass B1重放同一bank并精确pool真实X/Y。它们不是独立
video/hypernetwork分支。可复用image/language prefix cache；固定的内部三阶段读取仍只属于rollout前一次Writer调用，不是交互式适配。

## 2. 固定目标与Program schema

38个owner严格对应18层`q_proj`、18层`v_proj`、`action_in_proj`和`action_out_proj`。最大event容量`E=8`，active数量动态。

唯一Program schema为：

```text
P_lang    [38, 128]
P_scene   [38, 128]
P_process [8, 38, 128]
rho       [8]             # soft event presence
tau       [8, 2]          # normalized center and duration
sigma     [8, 38, 128]    # frame/candidate/probe uncertainty
```

event轴和owner轴意义固定；128维channel允许内部旋转，不给单个channel或LoRA rank赋人工技能语义。

### 2.1 Owner-specific language

冻结PaliGemma/Gemma产生原生language tokens：

```text
H_L [N_L, 2048]
Q_L [38, 128]                         # learned fixed owner queries
P_lang[j] = CrossAttn(Q_L[j], Pi_L(H_L))
P_lang [38, 128]
```

不使用Text Meta-LoRA，也不再把language跨帧平均为一个全局summary。

### 2.2 Per-frame compact observation

每帧在两个固定、task-independent antithetic Action probes下运行冻结PI0.5，probe轴在Program形成前不得平均：

```text
Z_patch [K, T_k, 256, 128]
Z_owner [K, T_k, 2, 38, 50, 128]
```

继续复用当前full-layer observer、四类transition candidates、event-horizon双向binding和有序segmenter。`Z_owner`是紧凑语义面，
不是后续LoRA因子的高维原生bank。

### 2.3 Ordered events and scene

每条视频独立形成adjacent、short-window、initial-to-current和current-to-final candidates；semi-Markov posterior沿时间单调，只能stay
或向前skip，不强迫使用全部8 slots，不做离散剪枝。每条视频产生：

```text
P_process,k [8, 38, 128]
rho_k       [8]
tau_k       [8, 2]
sigma_k     [8, 38, 128]
```

`P_scene,k [38,128]`由每个owner分别读取task-grounded first/final patches、first/final relation和final-initial relation；不得退回
全局first/final/difference向量。

### 2.4 Dynamic-K aggregation

训练真实覆盖`K in {1,2,4}`。每条视频先独立编码、分段和提取scene/event，再由8个fixed canonical event queries形成保序soft
alignment `M_k[8,8]`：

- `K=1`严格identity；
- video顺序置换不改变输出；
- slot映射保序，inactive slot可skip；
- 不允许单条视频完全覆盖其它视频。

按canonical event/owner聚合加权均值与variance。可使用zero-init DeepSets correction，但其幅度不得超过anchor RMS的25%。不得
平均frames、raw features或最终LoRA。

## 3. Target-native factor banks

当前observer只用层input/residual区分q/v owner，尚未捕获真实目标线性层的input/output空间。必须新增hook，对每个target`j`、
frame、probe和horizon捕获：

```text
target       X_j native input       Y_j native output
q_proj       [50, 1024]             [50, 2048]
v_proj       [50, 1024]             [50, 256]
action_in    [50, 32]               [50, 1024]
action_out   [50, 1024]             [50, 32]
```

输入侧候选索引严格为`n_A=(video k, frame t, probe p, horizon h)`，value为`X_j[k,t,p,h]`。输出侧候选索引为
`n_B=(video k, frame t, probe p, horizon h, type u)`，其中`u in {abs, adj, init, goal}`，value为
`Y_j^u[k,t,p,h]`。`X`没有output-bank type轴；不得为了统一实现而把同一个`X`复制四次并改变输入attention measure。

输出侧同时构造同维bank：

```text
Y_abs(t)  = Y(t)
Y_adj(t)  = Y(t) - Y(t-1)
Y_init(t) = Y(t) - Y(1)
Y_goal(t) = Y(T) - Y(t)
```

Pass A只保留128维Program。Pass B按每条video独立执行三个流式native子阶段，不得物化完整`K*T*38*50*2048` tensor：

1. **Pass B0a feature statistics**：按每视频、每canonical event单位质量累计共享candidate encoder输出的均值与128维covariance，
   用固定相对谱floor形成detached symmetric inverse-square-root；不保留完整keys，也不对当前video做梯度适配。
2. **Pass B0b native anchors/solve**：重读同一video，以task-stable anchor query和白化后的candidate keys形成有符号scalar
   compatibility；再按native group累计真实X/Y的均值、covariance与centered anchors，并形成regularized bank-conditioned queries。
3. **Pass B1 exact replay**：再次重读同一video的native chunks，精确重建四类output bank，再以已求得query执行positive/negative
   online softmax和真实X/Y weighted sum。

三个子阶段都必须按video维护首个采样帧、末个采样帧和跨chunk的previous activation：`Y_adj`使用同一video的上一采样帧，
`Y_init`始终使用该video首帧，`Y_goal`始终使用该video末帧。chunk边界不得重置这些状态，video边界必须隔离并重置；可以预缓存
端点或采用等价的分阶段读取。B1每个softmax分支分别维护running maximum、normalizer与weighted sum。相同输入下chunked B0/B1
必须在正常数值误差内等价于non-chunked reference。内部重读同一授权bank仍是一次rollout前Writer调用，不读取action/state/reward/
outcome，也不进行task-local优化。

## 4. Native-factor compiler

以下factor形式由G1 capacity oracle与最终compiler共同遵守，但selection logits的来源必须分开：G1允许每个held task直接优化free
logits/weights；从G3开始的shared deployment compiler才由共享Program query与native candidate keys按内容计算logits，禁止task/frame
查表、固定系数或普通平均。

G3当前修正不直接回归每条video的解析dual/score，也不把最终deployment query写成task/frame查表。family-shared candidate encoder
读取真实native direction/log-magnitude及normalized time、probe、horizon、output type；固定owner只作bounded调制。对每个video、target、
native group和canonical event，先在candidate-feature空间形成detached、对称的当前bank gauge：

```text
f_n = CandidateKey(native_n, metadata_n)
m_e, K_e = event-normalized mean/covariance of f_n
f_tilde_e,n = K_e^(-1/2) (f_n - m_e)
```

anchor query只由`P_lang`、固定owner/event/rank topology和family-shared参数产生，因此同一task不同video使用相同task anchor code；
`P_scene/P_process/rho/tau/sigma`及Pass A canonical assignment仍决定每个video的event/frame measure。query与`f_tilde`的content dot
product经有界非线性产生正/负compatibility。它不是固定系数、普通平均或自由查表，candidate bank仍是LoRA方向的必要Value路径。
以每视频单位质量的基础measure记native value为`v_n`，随后构造：

```text
mu = sum_n pi_n v_n
C  = sum_n pi_n (v_n - mu)(v_n - mu)^T
a_plus/minus = sum_e alpha_e sum_n pi_e,n (v_n - mu) * g_plus/minus(P_lang, f_tilde_e,n, event)
q_plus/minus = (C + lambda I)^-1 a_plus/minus
logit_plus/minus,n = q_plus/minus^T (v_n - mu) + bounded_local_bias
```

`lambda`使用fit/formal conditioning evidence确定的固定相对谱floor，不按held结果扫描；branch logits在measure下中心化以去掉常数
gauge。输入侧对真实`X` group构造统计，输出侧分别按q的8个256D groups、v的256D group、action-in的32个32D blocks和
action-out的32D group构造统计；不得把X复制到四个output type。Program/anchor网络不输出高维factor，所有native anchors和最终factor
仍是当前视频真实X/Y的加权和。owner/family/rank/event/group embedding只表示固定拓扑，不得包含task、authority、video、member或
frame absolute ID。

`84903aa`之后的family/fixed-owner scorer在clean detached `c3fc8e3`上从fresh运行到macro5/macro10，fit/held-video/task-holdout
median仅`.074715/.074620/.081644`，held p10 `.058381`；因此parameter ownership本身不是充分修正。固定key image进一步显示稳定
`1e-3`谱下q/v/action-out可达ceiling约`.226/.315/.629`，而action-in虽有`.975` ceiling、训练held仍仅约`.045`；raw-native与
FiLM tangent也只有约`.250/.336/.600`与`.280/.381/.645`。直接把query写入native坐标再经同一bank逆解在代数上退化为脆弱raw-query
transfer，所以clean pushed `4117117`只完成F0工程验证，没有浪费formal F3。

同一task三video的bank-global反事实则给出更精确证据：两video minimum-norm feature code对第三video的q/v/action-out近零，但把第三
video只加入共同code估计后，三者立即达到约`.90--.93`；每event symmetric inverse-square-root保持该正控制，同时action-in的两video
inductive held已达`.986`。这把当时最早接口定位为**task-stable anchor code的可识别性**，而不是再调loss、谱floor或candidate width。
`main@20acc33`随后以same-task稳定`P_lang`产生anchor query、按video/event白化candidate features，macro10完整451-condition的
fit/held-video/task-holdout median提高到`.141080/.142120/.145828`，held/fit为`1.00737`且40/40 held tasks从macro5改善；迁移接口
因此获得正证据，但`.75/.50` primary仍未通过，q held仅`.030186`。

后继六task单任务probe与q的18-target gradient decomposition把最早接口进一步收窄到query parameter ownership：另一fit/held video
能跟随train，但q 20-step update仍仅`.0197--.0277`；q input/output shared query梯度合成后只剩各target norm和的`.272/.268`。
当前唯一canonical修正因此保留四family共享Program/query/candidate trunks、stable `P_lang`和per-event gauge，同时在query输出加入
zero-init bounded fixed-owner input FiLM及fixed-owner/output-group FiLM。它们只表示38个真实LoRA targets及其native groups，不是task/
video/member/frame表；task dependence仍由共享query trunk基于`P_lang`计算。该修正不改变Program schema、真实banks、rank、teacher、
data、optimizer或F3 Gate，与旧checkpoint不兼容，必须fresh验证。若仍失败，应按F3 owner/group与task-content分解继续定位，不能用
task lookup或LR/seed/width小扫替代。

同一canonical实现保留一次预注册的`global_statistics_off`消融：令`C=I`，关闭current-bank covariance/preconditioning，但仍用B0
按单位measure形成centered native anchor并由B1 exact replay。它隔离的是“candidate-local compatibility加first-moment anchor是否已足够”，
不是另一套deployment路径，也不能被误写成完全不读bank的固定query或字面上的单pass实现。若off失败而bank-conditioned通过，严格
candidate-local假设即被正式淘汰并删除消融执行面。只有materialized FP64 bank-conditioned reference通过而显式covariance/Cholesky
replay因数值或内存不能恢复时，才启用
matrix-free covariance-vector product配合block-CG/Lanczos；不能因shared mapping泛化失败触发该fallback。

对target`j`和task residual rank slot`r=1..4`，先分配event权重。输入分支只在`n_A=(k,t,p,h)`候选上pool真实`X_jn_A`；
输出分支在`n_B=(k,t,p,h,u)`候选上pool真实`Y_jn_B^u`。两个分支各自由positive/negative两个softmax之差产生signed weights，
最后才分别对native values加权求和：

```text
wA_nA = softmax(lA_nA_plus) - softmax(lA_nA_minus)
wB_nB = softmax(lB_nB_plus) - softmax(lB_nB_minus)

a_jr = Normalize(sum_nA wA_jrnA * X_jnA)                 [d_in_j]
b_unit_jr = Normalize(sum_nB wB_jrnB * Y_jnB^u)          [d_out_j]
s_jr = s_ref_j * tanh(s_hat_jr)
b_jr = s_jr * b_unit_jr
DeltaW_task_j = sum_r b_jr a_jr^T
```

上式的softmax归一化域服从阶段合同：G1先在每条video内部归一化再以固定`beta_k=1/K`合并；G3才可根据G2证据使用从均匀值
初始化的bounded跨video correction。

`s_ref_j`来自fit-task expert correction的target-wise median RMS，冻结后不得按held结果调节。每个target的rank4 update经过small-core
balanced SVD确定性canonicalization，保持effective update不变且rank不超过4。

无条件把近零、未识别的pooled difference做满幅`rms_normalize`会放大随机方向。F1--F3先把operator与shared mapping本身分开验证；
进入F4后，若证据显示低可辨识selection破坏carrier，则confidence只能由当前bank的pooled raw RMS、固定fit reference、solve residual
与retained covariance energy等deployment可见量构成，并使低置信residual连续退回carrier。不得读取held outcome，也不得把confidence
作为掩盖mapping non-pass的捷径。

该compiler没有跨任务固定parameter span、全局16/32维effect code、21M hyperdecoder或从128维直接输出2048维的FactorHeads。
held task的新方向来自其自身视频在PI0.5目标层中的native activations。

## 5. 唯一rank16输出

首版canonical采用冻结rank12 shared carrier与mobile rank4 task residual。这是基于held5 mobile-rank4解析投影`110/120/76`且三个
arms均5/5非零的容量证据，不是沿用fixed-A或旧solver坐标；它是当前最合理的实验起点，不是专家已经证明不可改变的最终分解。

```text
A_final_j = concat_rows(A_carrier_j[12], A_task_j[4])
B_final_j = concat_cols(B_carrier_j[12], B_task_j[4])
B_final_j A_final_j = B_carrier_j A_carrier_j + B_task_j A_task_j
```

最终仍是一套76 tensors、38 targets、rank16 LoRA，没有cross terms、second adapter或并行expert。

只有同时证明native bank能表示task update、rank4 free-code已经收敛、剩余误差确由rank ceiling造成，并且同构full-rank16 oracle
显著通过，才重开完整task rank16诊断并据结果重新分配carrier/task rank。无论如何仍输出唯一一套完整rank16 LoRA，不增加第二
adapter。这一证据分支是正式设计的一部分，不能把12+4误记成永久硬约束。

## 6. Privileged evidence不是`q_pi`

canonical首版不存在神经`q_pi`，也不存在privileged Program teacher。训练期保留successful members、verified states、policy
effects、actions和reliability组成的非参数set-valued critic；它不输出Program，也不进入deployment forward。

generated LoRA的response使用global-member log-sum-exp equivalence loss：

```text
L_equiv = -eta * log sum_m w_m * exp(-D(R_generated, R_m) / eta)
```

一个logical trajectory只能由一个global member解释，不能按event拼接不同members。只有经short continuation验证的member-state pair
可作target；不同successful policies在loss层形成等价类，不要求generated LoRA逼近任一raw A/B。

policy-effect space继续用于消除LoRA gauge、参数不等价和q-family能量支配，但只作训练critic，不再形成
`Program -> effect code -> fixed inverse -> LoRA`部署链。

## 7. 冻结关系与Action Meta

永久冻结：source PI0.5、PaliGemma/VLM、native Action Expert、原始38个target weights、最终rank12 carrier。默认没有Action Meta。

Writer可训练：local patch/language projections、38个language queries、owner scene reader、transition matcher、event binding、ordered
segmenter、Dynamic-K alignment/aggregation、rank queries、native bank key/query projections、signed pooling logits和target scales。

Program与compiler分别通过后必须联合解冻全部Writer；backbone与carrier始终冻结。outer credit只更新event posterior、Program、rank
attention和scale，不直接扰动百万级A/B tensor。

Action Meta只在base Writer出现明确闭环增量后做matched controls，Stage 0/compiler冻结。只有明确净收益且无breadth/retention
损害才启用并永久冻结；否则保持关闭，不再把中性结果解释为加入理由。

## 8. 数据authority与fold

- 71个审计后的non-held LIBERO-90 tasks：source-seen，用于observer预训练、scene/language覆盖、carrier和preservation；
- target train24：source-unseen adaptation mappings；
- validation8：只用于后期deployment development；
- Test8：最终方法冻结前sealed。

开发使用五折：non-held每折56 fit/15 held，train24每折19 fit/5 held。每个macro访问全部19个target-fit，并从meta56轮换采样19个；
两种role各占50% task weight。fit可读language、videos、actions、experts、reward/BDDL和successful trajectories；held只做预注册机制
评测，validation/test不产生共享梯度。

## 9. 实际执行序列与Gate

专家原回复按概念依赖把Natural Program列为Stage 1、capacity oracle列为Stage 2，但最终明确capacity oracle是当前唯一下一步，
通过后才训练fresh Program。为避免编号混淆，仓库使用以下实际序列。

### G0. Authority冻结

固定fold manifests、baseline rows、carrier/task-expert/effect authority和role-balanced sampler，不做模型选择，不创建新数据。

### G1. Native-factor task-local capacity oracle（当前首个机制Gate）

先实现真实q/v/action-in/action-out input/output hooks与signed rank4 factor capacity path。使用冻结的现有Stage 0 v3、fold0 held5自然
teacher videos、known successful experts/mobile projections和carrier43；不碰validation/test。G1不得因最终deployment禁止task-local查表而
收紧这个capacity upper bound。

每个held task可以单独优化4个rank queries、event weights、输入/输出signed-pooling weights或logits和per-target scales。它们是task-local
free variables，不负责证明共享Program到attention的映射。Stage 0、shared compiler、source、carrier和task expert均冻结。

若G1使用`K>1`，每条video固定等质量`beta_k=1/K`，且每条video内部按event/frame assignment归一化，不能因更长或候选更多获得更大
总质量；`K=1`时该聚合严格退化为identity。G1不学习video reliability或learned `beta_k`。

G1必须选择纯Native Stage 0 observer加载路径；run contract与最小真实forward需枚举实际module及trainable parameter状态，确认旧
Action Meta mandatory loader没有装载任何Action Meta module或parameter。本阶段不修改Action Meta架构。

这是free-code upper bound，只回答真实native banks与signed pooling形式中是否存在强闭环rank4 residual；通过不代表deployment Writer或
shared Program-to-attention映射已经成立。输出仍必须是frozen rank12 carrier加该rank4 residual组成的唯一完整rank16 adapter。

loss：global-member effect、sensitivity-normalized effective update、independent action-query functional、carrier preservation。

同一strict250比较carrier、direct latest、known mobile-rank4 projection和native-factor free-code。必须同时满足：

- `(S_free-S_carrier)/(S_mobile-S_carrier) >= 0.70`，按当前43/110参考约`S_free >= 90/250`；
- breadth 5/5；Goal和Long均非零；至少4/5 tasks高于carrier；
- carrier successes保留至少33/43；
- single complete rank16，strict pairing，无second adapter。

失败时先做read-only span与response分析，定位native bank、hook、pooling或优化的最早问题，再进行有机制依据的修正和复评。修正
次数不预设上限，但每次必须带来新的可检验证据，不能退化为slot/width/seed版本链。只有充分修正后证据持续表明native basis本身
不可达，才停止Native-Factor主线；不得仅因一次或预定次数的non-pass把失败归因于数据量。

首轮scalar-output实现的formal结果为`88/250`、breadth3/5、Goal/Long 0。随后解析证明：对linear target，整条输出vector共享一个
signed scalar measure时，无bias的q/v outputs受限于`column_space(W)`；action-in因带bias且abs/difference可跨类型相减，结构上限为
`span(column_space(W),bias)`。因此q只能覆盖`1024/2048`维，action-in至多覆盖`33/1024`维，known-success rank4能量总体仅保留
约55--56%。paired response projection又把
independent mobile从`120/250`、Goal/Long=`11/8`变为`109/250`、Goal/Long=`0/0`，因此最早失效接口是q-output scalar pooling，
不是训练轮数。

当时的q-head G1修正保持输出候选索引`n_B=(k,t,p,h,u)`、四种type和真实native Y完全不变；只把q value按PI0.5真实的八个query heads
恢复为`[8,256]`，同一event measure下每个head独立进行positive/negative softmax归一化，再拼接成2048维`b`并做既定factor
normalization。它不复制候选、不增加fake type或非native value，也不引入task/frame route。v、action-in、action-out首轮保持整vector
一个signed measure，以便一次只修正有闭环证据的最早接口；若复评仍失败，再依据新span/response证据决定是否处理action-in或优化面。

q-head formal复评为`84/250`、逐task`28/21/35/0/0`，且生成update与known-success references的整体cosine仅约`0.06`。对同一真实
video bank做relative singular threshold `1e-3`的稳定中心子空间投影并materialize latest member后，strict250为`94/250`、逐task
`24/24/44/1/1`：breadth、Goal/Long与四task高于carrier已恢复，但carrier retention只有`22/43`，故仍不是Gate pass。这个成对闭环
结果把最早接口从bank span推进到随机稠密free logits的可达优化。

把latest member稳定子空间系数解析分解为positive/negative simplex并写入实际free logits后，精确step0 strict250达到`100/250`、逐task
`24/28/45/3/0`，relative recovery `0.851`；但breadth4/5、Long 0、仅3/5高于carrier且retention仍为`22/43`，所以Gate non-pass。
实际step0 residual与解析投影cosine为`0.952--0.964`，第一次Adam更新即降至`0.039--0.070`；500-step formal的最终effective-update
loss也在全部task差于未扰动step0。因此optimizer前的initialization checkpoint（step0）是预注册解析点，不能用step1冒充，也不因
内部loss选择被扰动路径。

set-valued free code按同一paired fixed50 success count在`carrier/latest/independent/earliest`中逐task选择，fold0固定为task90
carrier、91 independent、92 latest、93 independent、94 independent。其strict250达到`111/250`、逐task`35/29/45/2/0`，relative
recovery`1.015`且retention`34/43`，但breadth4/5、Long 0、仅3/5高于carrier，故仍是Gate non-pass。

该次失败把最早接口进一步定位到解析signed-measure系数的数值求解：relative singular threshold `1e-3`对应scatter逆的条件数可达
约`1e6`，FP32在task94把input/output direction cosine降至最低`0.978/0.883`。仅将这个小型、初始化时一次性的eigenspace与
inverse-scatter sufficient statistics改为FP64后，真实task94 forward/materialization的两侧minimum cosine均恢复到
`0.99999988`以上；candidate、softmax signed pooling、rank、scale、loss和checkpoint合同均不变。该修正仍只是G1 privileged
capacity solve，不进入G3，也不声称shared Program query-key attention成立。

FP64 clean formal的同一strict250达到`116/250`、逐task`35/34/44/3/0`，relative recovery `1.090`、carrier retention`35/43`；
总分、Goal与retention已通过，但breadth4/5、Long0且仅3/5 task高于carrier，故Gate仍non-pass。task94 signed solve此时两侧minimum
direction cosine已为`>=0.99999988`，因此最早剩余接口不再是数值求解。

解析结构中只有action-in仍存在whole-vector必然上限：其`32 -> 1024`真实Y共享一个scalar measure时，只能位于
`span(column_space(W),bias)`、至多`33/1024`。paired response保持task94其它37 targets为当前native candidate，仅恢复known-success
independent mobile的action-in target，Long从`0/50`变为`1/50`；完整counterfactual为`118/250`、逐task`35/35/44/3/1`、
breadth5/5、4/5高于carrier、retention`35/43`，数值上满足全部G1门。该action-in来自privileged reference，故这个response
不能冒充G1 pass，只证明该排除方向有独立闭环作用。下一修正保持
`n_B=(k,t,p,h,u)`、四类bank和真实Y不变，将action-in的1024D Y按native input width分为32个真实32D blocks，各block独立
signed pooling后拼回完整b；这是由Linear shape推出的最小full-width partition，不是group-count sweep。G1仍可直接持有这些free logits；
G3必须由共享Program query与content keys计算每个group的weights，不能转化为task/frame查表。

该修正由clean pushed `main@31f0053`完成。五task step0唯一rank16 bank的同一strict250为`114/250`、逐task`35/31/45/2/1`，
relative recovery`71/67=1.060`、breadth5/5、Goal2、Long1、4/5 task高于carrier、carrier retention`35/43`，全部G1 checks通过；
54/54 shards、250/250 rows、18/18 workers完整，Action Meta为0且没有使用shuffled/reversed。因此G1 capacity问题正式通过，下一阶段
进入G2；此结果仍不声称shared Program query-key attention成立。

### G2. Natural Program训练

仅在G1通过后，使用meta56+target-fit19、exact language、自然action-hidden videos，`K`均匀采样1/2/4，stride5并保留端点；video
demos与functional/action query episodes错开。允许从现有demonstrations派生action chunks、gripper/contact、BDDL progress、object
relations、speed perturbation和temporal crop，不创建新task、trajectory或成功语义。

训练owner-specific `P_lang/P_scene/P_process/rho/tau/sigma`与K aggregation，backbone冻结。loss覆盖local next-10-phase 7D action
summary、predicate progress/rising/contact、scene relation、cross-video event consistency、speed/crop robustness与probe stability。shuffled/reversed
遵守全局post-selection规则：不进入训练、loss、checkpoint选择、G1--G5 Gate或架构修正依据。

G2负责学习并验证每条video独立的event assignment与canonical alignment，以及跨video event variance、uncertainty、`K=1` identity和
video集合置换不变性；它不把learned video reliability混入G1容量结论。

每个macro仍对19个target-fit和19个轮换meta-fit各访问一次，但optimizer不能把38个task全部累积成一次更新。当前执行合同按
role-balanced microsteps训练：常规step包含2个target-fit和2个meta-fit，尾step包含1+1且随macro轮换，因而每macro为10次更新；
LR warmup/decay与resume cursor按真实optimizer step计数。该修正只改变已被诊断为不足的优化时间尺度，不改变Program结构、数据、
loss、K、Gate或task role质量。

clean pushed `main@49e7769`的fresh macro10/100-step formal把held full-vs-endpoints增量从旧`0.0381%`提高到`0.3080%`，并使
17/20 held task方向改善，但仍远低于`10%` Gate。fit-only gradient diagnostic确认动态Program存在，temporal梯度也未被其它目标
强方向性抵消；最早接口是近常数temporal readout使该梯度在Program/decoder上比non-temporal小约`10--21x`。当前先按同一
commit/topology exact-resume到预注册macro20，检验既有frozen-readout在100--500步展开的时标预测；若held增量和prediction
temporal std不实质增长，则该预测被否定，允许并要求依据此证据修正Program-to-temporal-readout结构后fresh复评，而不是继续训练或
做无机制超参小扫。

同一authority exact-resume到macro20/200-step后，held增量跃升到`8.6878%`、probe达到`36/40`，fit-only prediction temporal std
增长约`9x/30x`，所以readout时标预测成立；但所有K=2/K=4条件同时坍为one-event，K=1仍保留平均`6.42` events。local native
presence未坍缩，learned DP却把多数alignment mass集中到单一canonical slot，最早接口因此转到跨视频alignment。fit-only
counterfactual表明，仅锚定monotonic path首尾canonical边界、保留中间stay/skip与原content/time score，即可把K>1恢复为3 events并
保持`16.47%`视频增量；当前据此fresh复评，不改readout、loss、K、数据、LR/seed或Gate。

clean pushed `main@c1493a1`的boundary-anchored fresh run在macro10已把event指标修复为median 2、one-event 0；同一world4
exact-resume到macro20/200 updates后，held full相对endpoints改善达到`22.2047%`，probe为`38/40`，median active events为4、
one-event为0，same-task、K1 identity与K4 permutation全部通过，tau violation为`0.00357`。G2因此正式通过，冻结
`runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/checkpoints/macro_00000020`
作为G3唯一Program authority。

在meta-held15+target-held5同时要求：

- 至少90% same-task pairs在owner/event distance上近于nearest cross-task；
- 至少75% rows中probe变化小于nearest cross-task margin一半；
- one-event占比不超过25%，median active slots为2--6；
- full相对first+final在held action/progress loss上改善至少10%；
- `K=1` identity与K permutation invariance通过。

失败说明Program仍未满足动态资格，暂不进入LoRA训练。每次non-pass先冻结证据，并依次检查native capture、event grounding、
Program readout、监督可识别性、optimizer cadence等最早接口；用可证伪的counterfactual/decodability probe区分“信号不存在”和
“信号存在但没有被当前训练读出”。不设次数上限，但每次修正都必须只针对有新机制证据的接口并重新通过同一资格门，不能靠
slot数、width、seed或连续架构版本形成无信息迭代。

### G3. Frozen-Program shared compiler

使用meta56+target-fit19、自然videos和现有95-task/118-member evidence，跨episode采样并保持两种task role各50%。冻结通过Gate的
G2 Program、source、carrier和task experts，只训练共享的Program/candidate anchor compatibility、bank-conditioned solve、signed
pooling、target scales及后续bounded K correction。selection必须来自Program、candidate content与当前bank statistics，禁止task/frame
查表；training-only analytic dual/score只作operator oracle和诊断，不再作为deployment标签。

G3按以下最小因果顺序推进，前一接口通过后才恢复后一职责：

1. **F0 信息墙与算术smoke**：一个真实K1和一个K4条件；验证Program/source/carrier冻结、Action Meta 0、无ID输入、K2/K4
   teacher reads 0、finite/nonzero gradients、76 tensors且policy实际消费唯一rank16。
2. **F1 bank-operator capacity**：使用50 tasks/98 conditions的q/v/action-in/out authority，以free或analytic native anchor隔离shared
   mapping；materialized reference与streaming B0/solve/B1 replay每family median update cosine至少`0.995`、minimum至少`0.99`，且
   chunked/full在正常数值误差内一致。analytic强而streaming弱只说明实现、正则化或数值接口错误。
3. **F2 strict candidate-local消融**：在50 K1-covered tasks/451 task-video的预注册video/task holdout上令`C=I`并关闭covariance
   preconditioning，但保留单位measure的B0 centered native anchor与B1 exact replay；仅训练shared anchor scorer。held-video
   oracle-normalized recovery median至少`0.75`、每family至少`0.65`、task-holdout至少`0.60`才保留candidate-local假设。若不通过而
   F3通过，正式淘汰并删除off执行面，不继续pointwise key变体。
4. **F3 bank-conditioned mapping**：开启regularized current-bank solve，冻结Program/source/carrier；held-video recovery median至少
   `0.75`、p10至少`0.50`、train/held ratio至少`0.8`，且两个相邻checkpoint稳定。train高held低时先检查functional anchor seed与
   content泛化，不靠加宽或回归逐video dual。
5. **F4 scale/functional qualification**：恢复全部75 fit tasks；K1 teacher只覆盖既有50 tasks并以mapping loss保护selection，
   scale/video职责独立更新。teacher paired update不得退化，functional、cross-episode flow、carrier preservation要改善；若低置信
   residual仍破坏carrier，再按第4节deployment-visible evidence加入confidence退回机制。
6. **F5 K恢复**：按K1到K2再到K4恢复多视频职责；K2/K4不读teacher，要求K1 identity、K2/K4集合置换不变、bounded beta且
   same-task mapping retention至少`80%`。跨视频correction从uniform初始化并防止单条video覆盖其余videos。
7. **F6 held5 strict250**：同一冻结checkpoint比较carrier、learned language-only、full、first+final、same-task-other。Gate为full
   至少`60/250`、breadth至少4/5、carrier retention至少`33/43`、Goal或Long至少一项非零、相对language与first+final各净增至少5、
   same-task retention至少80%。

F2--F4的K1 teacher仍只来自fit-task当前真实native bank中对verified member rank4 residual的离线投影；loader可用provenance键定位
training label，但这些键、teacher factors、analytic dual、per-condition covariance及member identity都不得进入compiler forward或
checkpoint model state。多个verified members由detached whole-trajectory responsibilities选择，不平均不兼容members。selection、scale/
video与functional职责保持明确parameter ownership；旧functional不得再次以更大梯度覆盖mapping acquisition。通过F3后teacher-factor
loss只作有退出条件的warmup，不成为G4/Final永久监督。

F2/F3预注册fit split为25个meta-fit与15个target-fit tasks。每个macro完整覆盖15个target-fit，并从25个meta-fit中按固定seed轮换15个，
形成5次固定`3 target + 3 meta`全局optimizer updates；跨macro覆盖全部meta-fit，同时每一步保持两种role各50%。launch时可按1--6张
有效GPU cost-balanced分片。world size只改变每rank承担哪些task，不改变每个optimizer step的全局task集合、两种role质量、loss
归一化或scheduler cadence；exact-resume锁定首次launch topology。

若F1 operator本身不能恢复analytic reference，停在数值/measure接口；若F3 shared mapping在operator通过后仍低于门槛，结论只针对
当前Program-to-functional-content mapping，不得用joint training掩盖，也不得恢复旧realizer/FactorHead/task lookup。若mapping通过而
F6 closed-loop失败，再定位Program-to-function、scale/confidence、critic或teacher-to-utility的最早失效接口。G1/G2通过结论不受影响。

### G4. Joint Writer

只有G3有真实闭环信号才解冻全部Writer，继续冻结backbone、carrier和task experts。

G3的native-feasible LoRA teacher只作shared mapping组件验证，不是G4/Final必须存在的监督资产。正式联合训练不得预设目标LoRA；
可直接使用授权fit/meta tasks的teacher actions、functional及on-policy闭环信号，并由实际closed-loop证据选择最小充分loss集合。
若机制证据要求warmup，4A可用仍有效的G3 functional losses、Program checkpoint作anchor和较小compiler学习率；4B在fit natural tasks上以generated
LoRA rollout收集student visited states，再查询多个task experts；member-state pair只有在fixed short continuation中最终成功、严格提升
BDDL progress且不撤销predicate，或明显优于carrier/source，才进入set loss。无valid member的state只用reward/progress，不制造伪动作
label。

至少两个train24 folds分别要求：oracle-normalized recovery至少0.40、breadth 5/5、Goal/Long均非零、carrier retention至少75%、
same-task retention至少85%、相邻checkpoint差不超过10且success-set Jaccard至少0.75、full相对first+final和language-only均有正增量。

若G3有信号而joint崩落，优先尝试冻结compiler、Program proximal/EMA、更低Program LR或其它由崩落证据直接支持的稳定化方式。
不设joint轮数或修正次数上限，但不得无依据地扫architecture/slot/rank/seed。

### G5. Structured outer credit（条件阶段）

只有G4已证明full高于carrier、language和endpoints且breadth成立才进行。仅在fit/meta natural tasks上用task-equal CRN rollouts优化event
posterior、Program、rank attention和scales；reward包括success、BDDL progress、efficiency、carrier/full paired advantage和retained-
success barrier。

预注册outer评测节点必须相对G4净增至少10 held successes，且breadth、Goal/Long、same-task retention不下降。无净改善时先分析
credit estimator的失效机制；允许有依据地修正或更换estimator，不设family或尝试次数上限，但不做无信息的seed/sigma/step扫。

## 10. Final fresh训练与部署

进入Final的只有固定Program schema、native-factor operator、rank12 carrier recipe、经至少两个train folds验证的joint recipe，以及仅在
G5通过时的outer recipe。G1--G3的冻结/分段是机制验证，不是Final必须重演的训练课程。Final保留两类fresh初始化候选：

1. 从通过Gate的Program/compiler参数初始化，但使用fresh run、optimizer、scheduler，并联合解冻完整Writer；
2. 除冻结source/backbone与既定carrier外，整套Writer参数完全随机初始化，从头直接端到端联合训练，允许整体梯度下降自行形成
   Program、anchor、selection和scale的内部功能分化。

两类候选都必须遵守同一数据、信息墙、唯一rank16、最小充分loss和closed-loop评测合同；是否保留及最终采用哪一类由G4/Final前
预注册的matched evidence决定，不能因为探索阶段使用G1--G3就默认强制分段，也不能用内部loss偏爱某种初始化。Final正式训练不预设
存在目标LoRA，优先使用cross-episode teacher action/flow、set-valued functional、自然on-policy evidence、video necessity与
carrier/source preservation组成的最小充分监督；G3 factor teacher在mapping资格成立后退出。

Final前待owner裁决：本节描述71 meta+train24 fresh development recipe，而`docs/current_owner_requirements.md`同时记录了
方法选定后的32-task fresh refit。两者的精确顺序与validation8数据角色延迟到Final前明确，不阻塞G1--G5，
当前不默认任一种合并方式。

部署时Pass A生成唯一Program；Pass B0流式统计当前native banks并求bank-conditioned queries，Pass B1重放同一banks做exact signed
pooling、生成rank4 residual并与carrier拼接；安装唯一rank16 LoRA后闭环运行，不再观看teacher video。

## 11. Validation8、controls与Test8

validation8不再开放式选模。final只保留三个预注册相邻checkpoints。先跑correct、same-task-other、learned language-only和
first+final；checkpoint资格为correct至少135、breadth@1 8/8、breadth@5至少6/8、四suite非零、same-task retention至少90%、
相邻分数不低于它10以上、Jaccard至少0.80、top3 task share不超过70%。多个通过取correct最高，平分取更早；无通过者不跑完整
controls且不打开Test。这里的`135`只是是否展开完整controls的预筛，不是额外正式性能目标线。

selected checkpoint冻结后依次评测correct、same-task-other、cross-suite wrong、video-only、language-only、no-video/carrier、
static-first-repeated、first-only、final-only、first+final，最后才跑shuffled/reversed。后两者只确认视频时序特异性，
绝不进入训练、loss、checkpoint选择、G1--G5 Gate或架构修正依据。

最终方法资格：

- 唯一正式性能目标线是validation8 strict paired correct严格`>145/400`；
- 两个相邻checkpoint在selected的10分内、Jaccard至少0.80、breadth@5下降不超过1且无suite归零；
- same-task-other总分在correct正负10内，correct成功rows保留至少90%；
- correct相对language/no-video/static/first+final/wrong每个arm paired净增至少10、exact McNemar `p<0.05`、至少3/4 suites不负；
- correct相对shuffled和reversed分别paired净增至少15、`p<0.05`，Goal/Long不系统反向，且correct绝对分数本身保持。

全部通过才冻结method/checkpoint/K/controls并打开Test8一次；Test不反哺设计。

## 12. 停止ECP的充分条件

只有Natural Program、native-factor free-code、至少一fold shared compiler、至少两fold joint Writer、verified natural on-policy evidence、
一次structured outer credit、全部自然授权数据fresh validation和完整video controls都已完成，仍同时出现validation correct低于
130--135、breadth@5不超过4/8、Goal或Long为0、full不优于language/endpoints、same-task retention低于80%且相邻checkpoint持续大幅
换手，才可判断现有LIBERO与zero-interaction static-LoRA合同不足以支持稳定跨任务amortized policy compilation。

在此之前的局部门失败只淘汰对应接口。达到上述充分条件后应停止ECP，而不是继续修改slot、rank、width或decoder。

## 13. 当前实现边界

继续复用：Stage 0 v3、full-layer/horizon capture、transition matcher、event binding/segmenter、strict evaluator/controls、task expert bank、
successful members、effect calibration、probe-particle capture、carrier/mobile-rank4容量证据，以及natural occupancy/action/reward基础设施。

当前尚未实现且已完成全仓库orientation；得到owner明确许可后首先负责：真实38-target native input/output hooks、chunked bank accumulator、
signed rank4 compiler、free-code oracle optimizer及其strict250 evaluation wiring。它们必须形成一个canonical实现面，不恢复任何
退役Writer或realizer。
