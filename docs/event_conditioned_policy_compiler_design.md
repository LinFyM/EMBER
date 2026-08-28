# EMBER ECP Native-Factor Compiler

状态：2026-08-28第三次专家复核、owner最新裁决与其后behavior-identifiability证据更新的active architecture contract。三次专家审查
依次建立Native-Factor主线、要求current-bank conditioning，并把不可接受吞吐的full functional-polar降为fit-only reference。随后
formal S1与定向S2依次淘汰rank64 native-Q sketch、mean/variance scalar energy、query-conditioned多步set scorer及rank224/384
cross-image。授权fit-task的cross-episode真实flow-gradient诊断进一步证明：behavior rank4方向在三条真实bank中均有约`.90--.91`
signed-pooling可达性，但旧共享selector在held video只有`.023`，bank-independent native dual只有`.075`；同一task primal经每条当前
bank的全局单位质量covariance对偶化后，fit/held立即恢复到`.904--.911/.901`。因此当前活动修正不是继续叠加scorer，而是让共享
Program预测target-native primal，由当前video bank的确定性全局covariance solve把它变为dual query，再对真实X/Y做exact signed replay。
P1随后证明该operator在六tasks、四family和held videos上达到`.9545`。但95-task behavior-sufficiency诊断又发现：fit75行为流形对
held20具有`.7160` rank16可达性，旧frozen G2 Program的shared读出却只有`.247--.270`且不优于language-only，因此最早失效接口
进一步上移为G2跨task behavior identifiability。当前先保持Program schema与旧动态Gate，最小增加fit-only behavior alignment并重新
资格；通过后才恢复G3 Program-to-primal训练。该诊断仍只定位接口，不等于G2/G3 Gate；本文是当前唯一架构依据。

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
        +-- Pass B0: target-native values
                -> per-video global unit-mass native covariance
                -> Program-conditioned target-native primal direction
                -> current-bank primal-to-dual spectral solve
        +-- Pass B1: replay or reuse the same native bank
                -> q_dual dot real X/Y content logits
                -> exact antithetic signed pooling of real native X/Y
                -> current first implementation: task-specific rank4 residual
                -> concatenate frozen shared rank12 carrier
                -> one complete 38-target rank16 LoRA
```

Pass A与Pass B读取同一个冻结backbone，并共享owner、event、video assignment。Pass A回答视频表达什么目标、场景和过程，并把8个
event的内容与权重聚合成每个owner/rank的primal意图；Pass B0只从当前video的真实native candidates累计全局单位质量covariance，
以其截断谱逆把primal转为该bank自己的dual query。Pass B1重读或复用同一bank，由`q_dual^T v_n`的正负softmax之差精确pool真实X/Y。
Program不直接成为LoRA factor，covariance/dual也不进checkpoint；最终Value仍必须来自当前视频真实activations。B0与B1是同一个
Writer的两次流式读取，不是两套模型，且共同发生在rollout前唯一一次Writer运行中。full functional-polar只作历史fit-only reference。

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

Pass A只保留128维Program。Pass B不得物化完整`K*T*38*50*2048` tensor，当前deployment候选固定为两次流式读取：

1. **B0 global native statistics and dualization**：每条video独立、按采样时间quadrature形成全局unit-mass measure；对每个input target与
   每个output native group在线累计mean/covariance。共享Program网络把`P_lang/P_scene/P_process/rho/tau/sigma`映射为每个
   target/rank/event的native primal，并先在Program空间按event weights合并。以相对特征值门`1e-6`截断当前bank covariance，计算
   `q=C^+d`；covariance/eigensystem全部detached，不进入checkpoint。
2. **B1 exact global signed replay**：重读或复用同一video chunks，以同一个全局measure计算`l_n=q^T v_n`，把每个rank的score RMS统一到
   `.02`，再执行显式`softmax(+l)-softmax(-l)`并只对真实X/Y values加权求和。input侧没有type轴；output侧四类bank共同构成候选集合。

solve与replay必须使用完全相同的全局measure。此前“全局solve后按event分别replay”实际施加`C_e C_global^+`，会重新旋转方向，已由
task93/q20反例淘汰；event assignment与canonical alignment仍通过G2 Program的`rank_event/event_weights`决定primal内容，不再改变
native replay measure。covariance条件数可接近`1e6`，所有stats、dual score与signed replay必须使用禁用TF32的IEEE FP32；策略冻结forward
仍可使用BF16。

两次读取都必须按video维护首帧、末帧和跨chunk previous activation：`Y_adj`使用同一video上一采样帧，`Y_init`固定该video首帧，
`Y_goal`固定该video末帧。chunk边界不得重置，video之间绝不串接。B1两个softmax分支分别维护running maximum、normalizer与weighted
sum；chunked结果必须与同输入non-chunked reference在正常数值误差内一致。

## 4. Native-factor compiler

G1与G3的边界不变：G1可直接优化task-local selection logits/weights，回答是否存在强rank4组合；G3必须由跨tasks共享的完整Program
产生primal，不能包含task/video/frame/member lookup。固定target-native heads和owner/family/rank/event/group embeddings只表示38个
真实LoRA owners及其拓扑，跨所有tasks/videos共享，不是task route。

对某个target/native group，令当前video真实candidate为`v_n`、全局单位质量为`mu_n`、Program primal为`d_jr`：

```text
m       = sum_n mu_n v_n
C       = sum_n mu_n (v_n-m)(v_n-m)^T
q_jr    = C^+ d_jr                         # retained current-bank dual
l_jrn   = alpha_jr q_jr^T v_n              # alpha fixes score RMS to .02
w_jrn   = softmax_n(+l_jrn) - softmax_n(-l_jrn)
f_jr    = sum_n w_jrn v_n                  # real native Value only
```

input侧每target/rank有一个scale；output groups各自求dual，但同一target/rank使用跨groups公共scale，避免独立归一化抹掉block相对幅度。
小score下`f≈2 alpha C C^+ d`，所以Program学习的是跨video稳定的primal，而当前bank负责把它转换成自己的dual坐标。该机制仍是
Program query与native candidate content的signed cross-attention，不是固定系数、普通平均或free table；Program高维输出是查询意图，
不是直接部署factor，最终factor必须经过当前bank真实X/Y Value路径。

每条video独立完成B0/B1后，首版固定`beta_k=1/K`合并其raw factors再统一normalize；这保证K1 identity、视频集合置换不变，且不让长视频
仅因candidate更多支配。只有F5已有证据时才从uniform初始化有界learned beta，并必须防止单条video覆盖其余videos。

输入与输出signed factors分别形成`a_jr`和`b_unit_jr`；`s_jr=s_ref_j*tanh(s_hat_jr)`，`b_jr=s_jr*b_unit_jr`。当前rank4 residual与冻结
rank12 carrier只在最终materialization合并为唯一一套完整rank16 adapter。`s_ref`来自fit authority，不能按held结果调节。若低可辨识
raw factor在F4确实破坏carrier，confidence只能由当前bank的projection、retained energy、raw RMS等deployment可见量构成；不能掩盖G3
mapping non-pass。

已淘汰的full functional-polar、native-Q sketch、set-summary scalar energy、query-conditioned scorer与cross-image宽化只保留在Git、
formal artifacts和fit-only诊断代码中，不是并行active fallback。当前compiler没有跨任务固定parameter span、effect code realizer、
task lookup或第二套adapter；held task的最终方向仍只能由共享Program和它自己的真实video activations共同形成。

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

#### G2-B. policy-behavior sufficiency修正

旧G2 Gate通过后，P1已把native bank、rank4和current-bank primal-to-dual operator排除为首因；P2及其后95-task probe则证明旧
Program仍缺一项此前没有测量的资格。75 fit tasks各用两组disjoint、与video跨episode的flow-gradient panels，按meta56/target19
各50%建立normalized behavior kernel PCA；20个meta/target held tasks只作一次固定final诊断。rank16 span在held panel-B/consensus
达到`.7160/.8006`，universal只有`.1908`，而旧frozen Program即使由能把fit75读到约`.97`的shared reader解释，held exact rank4
仍只有full`.2695`、process`.2470`、language-only`.2687`。因此旧`22.2047%`动态增量保持有效，但不能再被解释为Program已经足够
指定未见task的policy adaptation。

当前唯一G2修正保持：

- `P_lang/P_scene/P_process/rho/tau/sigma`固定schema、E=8、boundary-anchored alignment和`beta_k=1/K`；
- frozen source PI0.5与Native Stage0 v3，Action Meta关闭；
- 原action/progress/predicate/event/probe/robustness目标与held动态Gate；
- canonical `scripts/train_ecp_natural_program.py`入口，不增加第二套Program或deployment路径。

新增的训练期behavior decoder只读取`P_process/rho/tau/sigma`，不能读取task ID、`P_lang/P_scene`、behavior label或teacher action作为
forward输入。它在75 fit tasks上预测上述8个浅/中/深q/v与action-in/out targets的rank16标准化坐标，并把一项role-balanced behavior
alignment loss反传到Program；decoder与坐标authority都不是deployment输入，G3/Writer抽取Program时不调用它。初始化只加载
`c1493a1/macro20`共同model tensors，新decoder、optimizer、scheduler与data cursor fresh；这不是old checkpoint续训或target lookup。

20 held tasks同时重新通过旧动态Gate，并在固定K2-A/K2-B/K1/K4 videos上满足：exact rank4对独立panel-B role-equal median至少`.50`、
对consensus至少`.60`，q/v各`.50`、action-in/out`.35/.50`，correct-vs-same-role-wrong Program margin至少`.10`，meta-held与
target-held各至少`.45`，相对预封存language-only`.2687`至少增加`.10`，每种video view coordinate至少`.65`且cross-view specific
consistency至少`.70`。held behavior labels不产生梯度；checkpoint只在预注册macro评估，不用shuffled/reversed。若fit behavior loss
不降，先检查decoder/Program优化接口；fit高而两role held低，说明Stage0/Program输入仍缺跨task sufficient statistic；behavior通过而旧动态
Gate回退，则处理表示—事件职责冲突。只有这些机制分支能触发下一修正，不能用seed、LR、width或rank小扫。

### G3. Frozen-Program shared compiler

以下G3 operator与Gate合同不变，但当前暂停在G2-B资格之前；旧`c1493a1/macro20`仍是动态正证据和初始化authority，不再作为足够的
final frozen Program。只有G2-B同时通过旧动态Gate与新增behavior Gate，才更新唯一G3 Program checkpoint并恢复P2。

使用meta56+target-fit19、自然videos和现有95-task/118-member evidence，跨episode采样并保持两种task role等质量。G2 Program、source、
carrier和task experts冻结；首版只训练共享Program-to-primal scorer，当前bank covariance、谱solve、真实X/Y和scale冻结。selection logits
最终仍为`q_dual^T v_n`，禁止task/frame查表。

G3按以下因果顺序推进：

1. **P0真实38-target operator F0**：在clean pushed detached authority上验证K1/K4 forward、全部primal/Program梯度、Action Meta 0、
   source/G2冻结、四类Y边界、chunked/non-chunked等价、uniform K aggregation、唯一rank12+4 materialization和一次真实policy consumption；
   同时报告每bank retained rank/trace、IEEE状态、显存与wall time。P0只证明新operator执行正确，不证明shared mapping。
2. **P1 multi-family task-local primal capacity**：在预注册meta/target fit tasks、浅/中/深q/v与两个action targets上，直接用fit-only
   teacher/behavior方向作为primal，经过每条当前bank的同一global dual/replay；video holdout零梯度。要求相对各自optimistic native
   projection无数量级损失并保持跨video，先证明该operator不只对task93/q20成立。失败按family、input/output、group、retained spectrum
   定位，不修改Program scorer。首版固定为meta-fit`[1,8,9]`、target-fit`[72,73,75]`及八个浅/中/深/action targets；每task两条
   fit video等权、固定500 steps，只优化跨videos共享的input/output primals。scale使用held-excluded fit-consensus常量并受`s_ref`
   上界约束，不进入optimizer；其独立学习仍属于F4。Gate为fit/held median`>=.80/.75`、held/fit`>=.85`、held相对optimistic
   median`>=.80`、四family held median及每task held均`>=.65`。
3. **P2 Frozen-Program shared mapping**：P1通过后，从fresh训练共享full-Program-to-primal scorer；使用原329 fit、40 held-video、82
   task-holdout与set-valued whole-member paired effective-update credit，scale继续冻结。两个相邻single checkpoints必须满足held median
   `>=.75`、p10`>=.50`、held/fit`>=.8`，并分别报告q/v/action与meta/target。correct-vs-wrong Program同role panel的role median margin
   必须`>=.10`，防止学习universal primal。P2的冻结scale使用排除held video、由40个mapping-fit tasks按task等权导出的唯一共享
   `[38,4]` rank template乘`s_ref`，不含task/video lookup且不进入optimizer；它只隔离方向Gate，scale学习仍属于F4。冻结Program、
   raw X/Y与B0 operator可放入run-local node-local cache以消除重复policy capture，但cache不是checkpoint或deployment输入，四类Y bank
   仍须按单视频边界在线构造。内部loss、projection或task93见证不能代替。
4. **F4 scale/functional qualification**：P2通过后才解冻独立scale职责并恢复全部75 fit tasks；paired mapping不能退化，functional、
   cross-episode flow和carrier preservation要改善。只有证据指向低置信随机residual破坏carrier时才加入deployment-visible confidence。
5. **F5 K恢复**：按K1到K2再到K4恢复多视频职责；K2/K4不读teacher，要求K1 identity、集合置换不变、same-task retention`>=80%`。
   首版维持`beta=1/K`；只有实证需要才从uniform初始化有界correction，且防止单条video覆盖其它videos。
6. **F6 held5 strict250**：同一冻结checkpoint比较carrier、learned language-only、full、first+final、same-task-other。Gate为full至少
   `60/250`、breadth至少4/5、carrier retention至少`33/43`、Goal或Long非零、相对language与first+final各净增至少5、same-task
   retention至少80%。

P1/P2的teacher只来自fit-task mapping-fit videos；provenance键仅用于loader pairing。teacher factor、behavior gradient、covariance、
dual、member identity与authority ID不得进入checkpoint或deployment输入，held video/task不得产生梯度。通过P2后factor teacher只作有退出
条件的组件监督，不成为G4/Final永久依赖。若P1高而P2 fit低，最早接口是Program-to-primal可识别性；fit高而video/task held低，处理
shared generalization；absolute高而Program causal低，处理universal shortcut。不得用joint training掩盖G3 non-pass，也不得恢复旧
realizer、functional-polar deployment或task lookup。G1/G2通过结论不受影响。

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
Program、Program-to-target-native primal scorer、per-video global covariance、截断谱primal-to-dual solve、IEEE exact signed replay、uniform
K aggregation、small-core rank4 canonicalization、rank12+4唯一rank16和451-condition mapping/evaluator wiring；Action Meta默认关闭。
fit-only consensus只属于mapping loader，covariance/dual/teacher不进入checkpoint。P2另有非checkpoint、非deployment的run-local compact
frozen-condition cache，仅保存Program、raw X/Y、final Y和B0 operator；fit retained、held ephemeral，四类Y bank仍在线构造。clean pushed
detached `e2f9d33`的完整38-target K1/K4
P0已通过：chunk4/one-chunk等价、全部梯度、Action Meta 0、uniform K、唯一rank16 materialization与真实policy consumption均成立。
clean pushed detached `c9e8198`的P1六任务formal也已通过：fit/held median`.971731/.954539`、held/fit`.982308`、held相对
optimistic median`.992193`，四family medians均`>=.9398`且minimum task held`.935001`。这把当前最早未验证接口收敛到P2 shared
full-Program-to-primal mapping；scale继续冻结到F4，P1内部结果不能冒充shared或closed-loop Gate。

旧`C=I`、P_lang-only、joint residual和full-Program Euclidean normalized-bilinear实现均已有formal non-pass并从active执行面退役；历史
由Git/config/artifacts保留。v4 full functional-polar、native-Q sketch、set-summary与query-conditioned scorer均已non-pass，只保留为
fit-only reference或diagnostic，不得再以deployment候选启动formal。当前唯一canonical deployment函数类是Program primal + current-bank
global dual + exact signed replay，按P0、P1、P2、F4、F5、F6、G4/G5和Final推进。
后续必须保持一个canonical运行面；不得恢复退役Writer/realizer、GOMQ/PECS、人工process、task lookup或第二adapter。
