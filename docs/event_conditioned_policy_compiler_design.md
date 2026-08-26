# EMBER ECP Native-Factor Compiler

状态：2026-08-24专家复核后冻结的active architecture contract。专家审查的是远程`main@7ab5a04`；当前仓库随后只进行了
代码/文档瘦身，没有产生新的科学结果。本文已结合当前`main@6fdaeb8`的保留实现复核，可作为下一session的唯一架构依据。

专家1416行回复已逐字保存于`docs/expert_review_20260824_native_factor.md`（仅将CRLF标准化为LF，逐行内容无差异）。本文是将
专家原文与owner后续裁决转成可执行合同的解释层，不替代原文；任何疑似曲解或冲突先核对原文，再按owner最新明确表达修正。

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
        +-- Pass B: same frames, target-native input/output readout
                -> Program-conditioned signed pooling
                -> current first implementation: task-specific rank4 residual
                -> concatenate frozen shared rank12 carrier
                -> one complete 38-target rank16 LoRA
```

两次pass读取同一个冻结backbone，并共享owner、event、video assignment。Pass A回答视频表达什么目标、场景和过程；Pass B用同一
Program在目标层原生空间中读取LoRA因子。它们不是两套独立video/hypernetwork分支。可复用image/language prefix cache，但Writer
仍只在rollout前运行一次。

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

Pass A只保留128维Program。Pass B按frame chunk重跑冻结backbone，并按已确定query在线累计sufficient statistics；不得物化完整
`K*T*38*50*2048` tensor。每个softmax分支分别维护running maximum、normalizer与weighted sum，同时按video维护首个采样帧、
末个采样帧和跨chunk的previous activation：`Y_adj`使用同一video的上一采样帧，`Y_init`始终使用该video首帧，`Y_goal`始终使用
该video末帧。chunk边界不得重置这些状态，video边界必须隔离并重置；可以预缓存端点或采用等价的分阶段读取。相同输入下chunked
结果必须在正常数值误差内等价于non-chunked reference。

## 4. Native-factor compiler

以下factor形式由G1 capacity oracle与最终compiler共同遵守，但selection logits的来源必须分开：G1允许每个held task直接优化free
logits/weights；从G3开始的shared deployment compiler才由共享Program query与native candidate keys按内容计算logits，禁止task/frame
查表、固定系数或普通平均。

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

使用meta56+target-fit19、K=1/2/4自然videos、现有95-task/118-member evidence，跨episode采样并保持两种task role各50%。冻结Program，
只训练共享的Program-to-rank query、native candidate key、signed pooling、target scales和bounded K correction。这里必须验证selection
logits确由共享Program query与candidate content计算而非task/frame查表。根据G2证据，跨video权重可从均匀权重初始化为有界learned
`beta_k`或其他bounded K correction；参数化必须防止单条video覆盖其余videos，具体形式由G2结果决定，不回写G1。

loss：whole-trajectory single-member equivalence；q/v/action-in/action-out四family等权的functional loss；cross-episode action flow；
K1 native-feasible mapping supervision；carrier preservation；same-task video functional consistency。K1 teacher只来自fit-task当前真实
native bank中对verified member rank4 residual的离线投影，training loader可用`(authority_id, video_demo, member)`定位标签，但这些键、
teacher factors及member identity都不进入compiler forward、checkpoint model state或deployment。teacher只保存pre-scale A/B directions、
scales与provenance，不保存native banks、free logits或weights；cache miss必须hard error，不能在线投影或回退mobile target。

K1 selection loss以input/output subspace和paired low-rank update direction等权构成，student scale在selection分支stop-gradient；独立
small-core singular-spectrum loss只更新scale，student directions在该分支stop-gradient。多个verified members仍由set-valued functional
critic的detached whole-trajectory responsibilities选择，不平均不兼容members。K2/K4不读取teacher，也不施加未投影mobile参数目标，
继续承担functional、cross-episode flow、carrier preservation、same-task与bounded multi-video组合职责。shared selection/query/key/context
和scale/video reliability使用独立gradient-clip预算，防止scale梯度吞噬selection更新；scale/video heads的输入对shared selection
context stop-gradient，使两个clip组具有真实互斥的parameter owner。首版保持原sampler、LR、K、rank、bounded beta、
`rms_normalize`和scale初始化，不加入confidence gate；只有teacher方向已在fit K1明显学会、而闭环仍被低置信随机residual破坏时，才以
独立机制证据重开confidence。

formal训练的全局task group固定为3个target-fit加3个meta-fit，19+19 task的尾step自然为1+1；launch时可按1--6张有效GPU
cost-balanced分片。world size只改变每rank承担哪些task，不改变每个optimizer step的全局task集合、两种role质量、loss归一化或
scheduler cadence；exact-resume锁定首次launch topology。

held5门比较carrier、learned language-only、full、first+final、same-task-other。继续条件：full至少60/250、breadth至少4/5、保留
carrier至少33/43、Goal或Long至少一个非零、full相对language-only与first+final各净增至少5、same-task retention至少80%。

若G1很强而shared compiler低于carrier或breadth不超过2，结论是当前source-unseen mappings不足以学习该共享映射；不得用joint
training掩盖失败。可以依据mapping、factor selection或critic的具体证据修正并复评，不设结构版本上限；无机制差异的小变体不算
有效推进。

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

进入final的只有固定Program schema、Stage 0初始化、native-factor compiler、rank12 carrier recipe、经两个train folds验证的joint
recipe，以及仅在G5通过时的outer recipe。fresh使用全部71 meta tasks与train24，仍保持train24/meta各50% adaptation weight：

1. fresh训练rank12 carrier；
2. fresh训练Natural Program；
3. 冻结Program训练compiler；
4. 联合解冻全部Writer；
5. 训练到train24 cross-validation预先确定的horizon；
6. 仅当outer已通过，追加固定数量outer updates。

Final前待owner裁决：本节描述71 meta+train24 fresh development recipe，而`docs/current_owner_requirements.md`同时记录了
方法选定后的32-task fresh refit。两者的精确顺序与validation8数据角色延迟到Final前明确，不阻塞G1--G5，
当前不默认任一种合并方式。

部署时Pass A生成唯一Program；Pass B读native banks、生成rank4 residual并与carrier拼接；安装唯一rank16 LoRA后闭环运行，不再观看
teacher video。

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
