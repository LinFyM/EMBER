# EMBER ECP Native-Factor Compiler

状态：2026-08-28第三次专家复核、owner最新裁决与F3连续formal/机制证据后更新的active architecture contract。第一次专家审查锁定
`main@7ab5a04`并建立Native-Factor主线；第二次专家审查锁定`main@ed2883b`及其可达历史，针对G3跨视频dual/score旋转给出
bank-conditioned两阶段Pass B修正；第三次专家审查锁定`main@9b52e59`及其可达历史，把高成本full functional-polar降为fit-only
teacher/reference，并提出低维bank-adaptive functional sketch与轻量shared student。clean pushed `main@27bde62`的S1 formal
counterexample随后证明`r_s<=64` native-query sketch失容；当前严格采用专家预先规定的fallback：full polar与native-Q sketch都只作
teacher/diagnostic，部署候选改为不经过该native-Q瓶颈的pure low-dimensional set-summary student，且必须先通过task-local free-query
正控。本文是当前唯一架构依据。

三次专家原文分别逐字保存于`docs/expert_review_20260824_native_factor.md`、
`docs/expert_review_20260826_bank_conditioned_native_factor.md`和`docs/expert_review_20260828_g3_functional_sketch.md`。本文是将专家
原文与owner后续裁决转成可执行合同的解释层，不替代原文；任何疑似曲解或冲突先核对原文，再按owner最新明确表达修正。第二次专家建议Final默认从通过Gate的组件初始化；owner
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
        +-- Pass B0: target-native values + compact candidate features
                -> per-video/event measure-normalized set summary
                -> Program/free-code + current-bank summary + candidate content
                -> bounded positive/negative scalar logits for each candidate
        +-- Pass B1: replay or reuse the same native bank
                -> exact signed pooling of real native X/Y
                -> current first implementation: task-specific rank4 residual
                -> concatenate frozen shared rank12 carrier
                -> one complete 38-target rank16 LoRA
```

Pass A与Pass B读取同一个冻结backbone，并共享owner、event、video assignment。Pass A回答视频表达什么目标、场景和过程；Pass B0
在同一当前video bank上形成不含task/video ID的低维集合摘要，并让完整Program与该摘要、当前candidate content共同产生scalar
positive/negative logits；Pass B1最后重放或复用同一bank并精确pool真实X/Y。低维只约束Program、candidate feature与集合摘要，不再
约束最终native factor所在子空间。B0与B1是同一个Writer的规划和精确Value阶段，不是两套模型；二者共同发生在rollout前
唯一一次Writer运行中，不进行交互式适配。full functional-polar只在fit条件离线产生teacher/reference，不属于deployment forward。

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

Pass A只保留128维Program。Pass B不得物化完整`K*T*38*50*2048` tensor，当前deployment候选分两步流式执行：

1. **Pass B0 low-dimensional set summary and scalar energy**：对每个target/native group编码低维candidate feature；在每条video、每个
   canonical event内部按G2 assignment形成unit-mass measure，对共享非线性feature的均值和二阶离差作集合摘要。q family读取全部8个有序
   event摘要，其它family先读取当前event摘要。Program或task-local free code、current-bank摘要、candidate feature与合法metadata通过
   family-shared非线性compatibility直接产生每个candidate的有界正负scalar logits；不存在`Q_g q_tilde` native lift，也不输出高维factor。
2. **Pass B1 exact replay**：重读或复用同一video的native chunks，精确重建四类output bank，再以已求得的candidate logits执行
   positive/negative online softmax和真实X/Y weighted sum。低维summary只规划selection，不近似或蒸馏最终Value。

两步读取都必须按video维护首个采样帧、末个采样帧和跨chunk的previous activation：`Y_adj`使用同一video的上一采样帧，
`Y_init`始终使用该video首帧，`Y_goal`始终使用该video末帧。chunk边界不得重置这些状态，video边界必须隔离并重置；可以预缓存
端点或采用等价的分阶段读取。B1每个softmax分支分别维护running maximum、normalizer与weighted sum。相同输入下chunked sketch/B1
必须在正常数值误差内等价于non-chunked reference。内部重读同一授权bank仍是一次rollout前Writer调用，不读取action/state/reward/
outcome，也不进行task-local优化。full functional-polar允许在fit-only离线teacher/reference中做额外读取，但不得进入shared deployment
forward、held teacher构造或最终checkpoint state。

## 4. Native-factor compiler

以下factor形式由G1 capacity oracle与最终compiler共同遵守，但selection logits的来源必须分开：G1允许每个held task直接优化free
logits/weights；从G3开始的shared deployment compiler才由共享Program query与native candidate keys按内容计算logits，禁止task/frame
查表、固定系数或普通平均。

G3不回归每条video的解析dual/score，也不把deployment selection写成task/frame查表。每个固定LoRA target由共享candidate encoder
读取真实direction与metadata；normalized time、probe、horizon以及仅输出侧存在的type可以进入低维feature。固定target/group参数只表示
38个LoRA owner及native groups，跨所有tasks/videos共享，不是task route。

以某个target/native group的真实value `v_n`、低维feature `h_n`和event measure `pi_e,n`记，首个可证伪函数类固定为：

```text
h_n        = phi_g(v_n, time, probe, horizon[, type])          [m]
s_e        = concat(E_pie[psi(h_n)], Var_pie[psi(h_n)])        [2m]
c_e        = ordered_q_context(s_1:8) or local_context(s_e)     [m]
z_jre      = full Program readout or task-local free code       [m]
B_n^+-     = candidate_basis_g(h_n)                             [2,m]
C_jre^+-   = condition_g(z_jre, c_e, z_jre * c_e)               [2,m]
l_n^+-     = b * tanh(<B_n^+-, C_jre^+-> / sqrt(m) + bias_jre^+-)
```

这里每个event/video的`pi`先归一化为unit mass，因此长视频或候选数更多不会自动获得更大质量；K1严格identity。q family的score读取全部
8个有序event summaries，v/action先读取当前event summary。上式是candidate-local非线性basis与Program/bank-conditioned系数形成的
轻量separable scalar-energy，并非native value上的`q^T X/Y`或固定key查表；它只限制标量打分器，不限制最后真实native factor的维度。
正负分支独立计算且有界，最后才分别softmax并相减。首个正控冻结现有
candidate encoder/source/G2/carrier/scale，只训练family-shared summary/scorer与task-local low-dimensional free codes，以隔离
“该函数类是否有容量”；若连fit rows都学不到，再以该证据决定是否解冻或替换candidate encoder，而不把所有职责一次联合训练。

S1使用的fixed nested projection、native/key cross-image、`Q_g`、small functional operator与full-native `C_rQ` solve完整保留在
`functional_sketch.py`及formal evidence中，但其rank64正式反例已经关闭deployment资格。不得继续搜索projection seed/rank、恢复
native-Q scorer或向deployment叠加polar/SVD；它们只作teacher/diagnostic。

full functional-polar仍可在fit videos上离线计算解析上限和canonical effective-update teacher，但其full covariance、native eig/SVD、
per-condition inverse state和teacher factor都不得进入deployment forward或checkpoint。它接近`.996/.999/1.000/.998`的task-local
见证只证明当前bank/key/signed-pooling函数类存在强方向；`58.332s`的真实K1 profile和约49分钟的六卡macro5下限已使其deployment
执行形态吞吐non-pass，也从未形成新的F3 shared-mapping结果。

G3 training target是fit-only稳定functional object，而不是不稳定的逐video factor gauge。每个fit task/member只用预注册mapping-fit
videos（排除该task held video）产生cross-video consensus effective update；set-valued whole-member posterior、四family等权和
paired effective-update credit保持。由fit tasks构造的task-independent universal update `U`只用于common shortcut的负控与centered
credit：fit row的`U`必须leave-one-task-out，task-holdout的`U`只能由全部fit tasks构造，禁止把primary task泄入自己的baseline；同时
报告`||T-U||`并对近零specific residual设定预注册数值口径，不能靠不稳定余弦制造因果提升。必要时只按预注册meta/target role分层，
不得形成task lookup。

最终compatibility仍由共享Program query、current-bank summary和candidate content计算；Program/anchor网络不输出高维factor。
owner/family/rank/event/group embedding只表示固定拓扑，不得包含task、authority、video、member或frame absolute ID。此前`C=I`、
P_lang-only、fixed-owner、small-residual joint、full-Program Euclidean bilinear、fresh IEEE和full functional-polar deployment均已冻结为
历史机制证据或fit-only reference，不是并行active fallback。

首版candidate logit采用“current-bank set summary调制candidate-local energy”这一单一内容路径：

```text
l_n^+- = bounded_energy_g^+-(Program/free_code, summary_current_bank,
                             candidate_feature_n, metadata_n)
```

task-local free code只属于S2容量正控，跨该task全部fit/held videos共享，不能含per-video/per-candidate状态；shared阶段必须由完整Program
替代。不得恢复task-local deployment lookup、直接Program到高维factor或绕过current-bank的language-only selection旁路。

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

使用meta56+target-fit19、自然videos和现有95-task/118-member evidence，跨episode采样并保持两种task role等质量。G2 Program、source、
carrier和task experts冻结；首个shared student只训练完整Program与低维current-bank summary到shared query/compatibility的映射，scale、
sketch basis和candidate statistics冻结。selection必须来自Program、candidate content与当前bank statistics，禁止task/frame查表。

G3按以下最小因果顺序推进，前一接口通过后才恢复后一职责：

1. **S1 无训练sketch容量与吞吐Gate（formal non-pass）**：复用现有F1的50 tasks/98 K1 conditions和同一次真实native capture；对预注册浅层、layer9、
   layer17 q/v及action-in/action-out groups构造一个最大rank64 nested sketch，并只读报告`r_s={16,32,64}`前缀曲线。全部模型冻结，
   full IEEE functional-polar只作fit-only reference。每family task median至少`.98`、minimum至少`.95`，深层q/v median至少`.95`、
   minimum至少`.90`，streaming/materialized cosine至少`.9999`；单A40 post-capture不慢于native capture、目标不超过约6秒，K1总
   condition约不超过20秒、peak reserved低于35GB。这里的速度还必须结合完整formal条件数给出真实wall-time预算；仅单condition达标
   但formal规模仍不相称也不通过。固定最小通过rank与唯一sketch公式后才进入S2；rank64仍失败时full polar永久只作teacher/reference，
   不再往deployment叠加polar/SVD技巧。该失败条件现已由task93/q20两条sealed K1的clean detached formal反例触发：同条件F1
   analytic-to-teacher为`.99556--.99791`，rank64 sketch仅`.15669--.15744`，而chunk一致性最低`.9999769`。完整`H`最佳top64的
   exploratory对照仍远低于门，故不是fixed random projection抽坏，而是`r_s<=64` native-query basis失容；无需运行其余row即可否决
   含per-row minimum的合取Gate，但该早停不能估计全panel分布。
2. **S2 12-task pure low-dimensional set-summary student容量—因果Gate**：固定选择6 meta+6 target tasks，覆盖困难q/v深层与action controls。每个role各留1个task作
   零梯度true task-holdout，其余10 tasks各使用两条fit video和一条zero-gradient video-holdout。先在同一student执行面把Program query
   换成task-local free low-dimensional query作正控；该执行面不得以`Q_g q_tilde`限制native query，而应从Program/free code、
current-bank低维集合摘要、candidate key与metadata直接产生正负candidate logits，再对真实X/Y exact pool。正控不通过就先核对
candidate encoder authority与score image；只有确实加载既定fit-trained encoder后仍失败，才淘汰当前summary/score函数类，禁止用
shared训练掩盖。正控通过后才
   训练shared Program-to-summary query，loss只保留set-valued paired effective update、universal-centered task-specific excess和
   cross-video dispersion，scale继续冻结。fit、video-holdout、task-holdout以及q/v/action families必须分开报告。
   在12-task Gate前只允许一个最小机制witness：沿既定task93/q20，task-local free code在两条fit videos共同优化，第三条video全程
   zero-gradient；不得有per-video/per-candidate参数。fit effective-update median至少`.90`，held effective-update及input/output
   pushforward各至少`.80`，held/fit至少`.8`。未达到就立即淘汰首版mean/variance DeepSets scalar-energy函数类，不进入12-task或shared
   Program训练；达到只证明函数类值得进入正式正控，不冒充S2 Gate。
3. **S2因果Gate预注册**：在看到shared student结果前，仅用三个固定anchor一次性校准数值门槛：fit-only universal negative、task-local
   free-query positive和当前fresh IEEE失败checkpoint；随后写入sealed Gate manifest，不再按student结果移动。专家建议的free-query
   held overall/q/v/action `.90/.80/.90`、shared fit/held `.60/.50`、causal margin `.10`、interaction `.05`与70% task-positive只作为
   预注册起点，不在校准完成前冒充owner硬裁决。正式判定至少要求correct Program+bank同时显著胜wrong Program、wrong bank与
   leave-one-task-out universal，meta/target role分别成立，并要求q/v不能被action family掩盖。
4. **S3 完整451-condition shared Gate**：只有S2 absolute、video/task泛化和因果资格同时通过才恢复现有329 fit、40 held-video、82
   task-holdout与两个相邻single checkpoints。只训练通过S2的轻量student；G2/source/carrier/scale继续冻结。absolute容量必要条件保留
   held median至少`.75`、p10至少`.50`、held/fit至少`.8`；同时满足S2封存的universal-centered、correct/wrong Program、correct/wrong
   bank、crossed interaction、q/v因果、own-vs-wrong teacher及速度Gate。absolute高而causal低不得通过。
5. **F4 scale/functional qualification**：S3通过后才解冻独立scale/spectrum职责并恢复全部75 fit tasks；selection与scale彼此
   stop-gradient。paired update不能退化，functional、cross-episode flow和carrier preservation要改善；低置信退回机制只有在
   deployment-visible confidence证据明确时才加入。
6. **F5 K恢复**：按K1到K2再到K4恢复多视频职责；K2/K4不读teacher，要求K1 identity、K2/K4集合置换不变、bounded beta且
   same-task mapping retention至少`80%`。跨视频correction从uniform初始化并防止单条video覆盖其余videos。
7. **F6 held5 strict250**：同一冻结checkpoint比较carrier、learned language-only、full、first+final、same-task-other。Gate为full
   至少`60/250`、breadth至少4/5、carrier retention至少`33/43`、Goal或Long至少一项非零、相对language与first+final各净增至少5、
   same-task retention至少80%。

S1--S3的K1 teacher只来自fit-task当前真实native bank中对verified member rank4 residual的离线投影或full-polar fit-only reference；
provenance键只用于loader pairing。teacher factors、analytic dual、per-condition covariance、member identity与authority ID不得进入student
forward或checkpoint model state，held video与true task-holdout不得参与teacher构造或梯度。多个verified members由detached whole-
trajectory responsibilities选择，不平均不兼容members。通过S3后teacher-factor loss只作有退出条件的warmup，不成为G4/Final永久监督。

S1现已失败，结论是`r_s<=64` native-Q sketch未保留operator，因此不训练该sketch shared student；当前先实现pure set-summary
free-query函数类。若free-query失败，修正candidate/summary函数类；free-query通过而
shared fit低，重开Program sufficiency或credit可识别性，不再修改operator；fit高而video/task held低，处理shared generalization；absolute
高而causal低，修正teacher/common decomposition。不得用joint training掩盖G3 non-pass，也不得恢复旧realizer/FactorHead/task lookup。
G1/G2通过结论不受影响。

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

当前活动树已经实现真实38-target native input/output hooks、abs/adj/init/goal banks、跨chunk/视频边界状态、G1 free-code、G2 Natural
Program、feature/functional sufficient statistics、functional-polar Program queries、native spectral solve、B1 exact signed replay、
small-core rank4 canonicalization、rank12+4唯一rank16、S1 functional sketch/formal early disqualifier和451-condition evaluator wiring；
Action Meta默认关闭。fit-only consensus只属于mapping loader，full-polar/sketch statistics不进入checkpoint model state。task93/q20
pure low-dimensional set-summary scalar-energy执行面已实现，但首轮formal发现其runtime把fresh seeded random chart误作existing candidate
encoder；当前只修正为显式加载并冻结`78b7e58/macro5` fit-trained candidate chart。尚未实现的是12-task free-query/shared资格、
leave-one-task-out universal-centered credit和
Program×bank crossed Gate。

旧`C=I`、P_lang-only、joint residual和full-Program Euclidean normalized-bilinear实现均已有formal non-pass并从active执行面退役；历史
由Git/config/artifacts保留。v4 full functional-polar的K1吞吐资格non-pass，native-Q sketch的S1容量资格也formal non-pass；两者只保留为
fit-only teacher/reference或diagnostic，不得再以deployment候选启动formal。当前唯一待资格化实现是pure set-summary scalar-energy
student：先以真实fit-trained candidate encoder重做task93/q20最小witness，再做S2 12-task free-query正控与shared Program映射，并按S3完整451、F4 scale/functional、
F5 Dynamic-K、F6 held5 closed-loop、G4/G5和Final推进。
后续必须保持一个canonical运行面；不得恢复退役Writer/realizer、GOMQ/PECS、人工process、task lookup或第二adapter。
