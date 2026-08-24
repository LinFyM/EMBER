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

输出侧同时构造同维bank：

```text
Y_abs(t)  = Y(t)
Y_adj(t)  = Y(t) - Y(t-1)
Y_init(t) = Y(t) - Y(1)
Y_goal(t) = Y(T) - Y(t)
```

Pass A只保留128维Program。Pass B按frame chunk重跑冻结backbone，并按已确定query在线累计sufficient statistics；不得物化完整
`K*T*38*50*2048` tensor。

## 4. Native-factor compiler

对target`j`和task residual rank slot`r=1..4`，对应owner Program与rank embedding产生query`q_jr`，先分配event权重，再在所有
video/frame/probe/horizon/native feature type上做两个softmax分支之差的signed pooling：

```text
w_n = softmax(l_n_plus) - softmax(l_n_minus)

a_jr = Normalize(sum_n wA_jrn * X_jn)                    [d_in_j]
b_unit_jr = Normalize(sum_n wB_jrn * Y_type_jn)          [d_out_j]
s_jr = s_ref_j * tanh(s_hat_jr)
b_jr = s_jr * b_unit_jr
DeltaW_task_j = sum_r b_jr a_jr^T
```

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

先实现真实q/v/action-in/action-out input/output hooks与Program-conditioned signed rank4 compiler。使用冻结的现有Stage 0 v3、fold0
held5自然teacher videos、known successful experts/mobile projections和carrier43；不碰validation/test。

每个held task只优化4个rank queries、event/signed-pooling weights和per-target scales。Stage 0、shared compiler、source、carrier和task
expert均冻结。这是free-code upper bound，不是部署方法。

loss：global-member effect、sensitivity-normalized effective update、independent action-query functional、carrier preservation。

同一strict250比较carrier、direct latest、known mobile-rank4 projection和native-factor free-code。必须同时满足：

- `(S_free-S_carrier)/(S_mobile-S_carrier) >= 0.70`，按当前43/110参考约`S_free >= 90/250`；
- breadth 5/5；Goal和Long均非零；至少4/5 tasks高于carrier；
- carrier successes保留至少33/43；
- single complete rank16，strict pairing，无second adapter。

失败时先做read-only span与response分析，定位native bank、hook、pooling或优化的最早问题，再进行有机制依据的修正和复评。修正
次数不预设上限，但每次必须带来新的可检验证据，不能退化为slot/width/seed版本链。只有充分修正后证据持续表明native basis本身
不可达，才停止Native-Factor主线；不得仅因一次或预定次数的non-pass把失败归因于数据量。

### G2. Natural Program训练

仅在G1通过后，使用meta56+target-fit19、exact language、自然action-hidden videos，`K`均匀采样1/2/4，stride5并保留端点；video
demos与functional/action query episodes错开。允许从现有demonstrations派生action chunks、gripper/contact、BDDL progress、object
relations、speed perturbation和temporal crop，不创建新task、trajectory或成功语义。

训练owner-specific `P_lang/P_scene/P_process/rho/tau/sigma`与K aggregation，backbone冻结。loss覆盖local next-10-phase 7D action
summary、predicate progress/rising/contact、scene relation、cross-video event consistency、speed/crop robustness与probe stability；
shuffled/reversed不进入训练。

在meta-held15+target-held5同时要求：

- 至少90% same-task pairs在owner/event distance上近于nearest cross-task；
- 至少75% rows中probe变化小于nearest cross-task margin一半；
- one-event占比不超过25%，median active slots为2--6；
- full相对first+final在held action/progress loss上改善至少10%；
- `K=1` identity与K permutation invariance通过。

失败说明Program仍是task/endpoint code，暂不进入LoRA训练。修正聚焦native capture、event grounding或owner-specific
language/scene；不设次数上限，但每次都要针对已定位机制并重新通过同一资格门，不能靠slot数、width、seed形成无信息版本链。

### G3. Frozen-Program shared compiler

使用meta56+target-fit19、K=1/2/4自然videos、现有95-task/118-member evidence，跨episode采样并保持两种task role各50%。冻结Program，
只训练Program-to-rank query、signed pooling、target scales和bounded K correction。

loss：whole-trajectory single-member equivalence；q/v/action-in/action-out四family等权的functional loss；cross-episode action flow；
sensitivity-normalized mobile update辅助；carrier preservation；same-task video functional consistency。

held5门比较carrier、learned language-only、full、first+final、same-task-other。继续条件：full至少60/250、breadth至少4/5、保留
carrier至少33/43、Goal或Long至少一个非零、full相对language-only与first+final各净增至少5、same-task retention至少80%。

若G1很强而shared compiler低于carrier或breadth不超过2，结论是当前source-unseen mappings不足以学习该共享映射；不得用joint
training掩盖失败。可以依据mapping、factor selection或critic的具体证据修正并复评，不设结构版本上限；无机制差异的小变体不算
有效推进。

### G4. Joint Writer

只有G3有真实闭环信号才解冻全部Writer，继续冻结backbone、carrier和task experts。

4A先用G3全部loss做functional joint warmup，Program checkpoint作anchor，compiler较小学习率。4B在fit natural tasks上以generated
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

部署时Pass A生成唯一Program；Pass B读native banks、生成rank4 residual并与carrier拼接；安装唯一rank16 LoRA后闭环运行，不再观看
teacher video。

## 11. Validation8、controls与Test8

validation8不再开放式选模。final只保留三个预注册相邻checkpoints。先跑correct、same-task-other、learned language-only和
first+final；checkpoint资格为correct至少135、breadth@1 8/8、breadth@5至少6/8、四suite非零、same-task retention至少90%、
相邻分数不低于它10以上、Jaccard至少0.80、top3 task share不超过70%。多个通过取correct最高，平分取更早；无通过者不跑完整
controls且不打开Test。

selected checkpoint冻结后依次评测correct、same-task-other、cross-suite wrong、video-only、language-only、no-video/carrier、
static-first-repeated、first-only、final-only、first+final，最后才跑shuffled/reversed。后两者绝不进入训练、loss或选模。

最终方法资格：

- correct必须`>145/400`；项目继续追求`>150/400`；
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

当前尚未实现且下一session完成全仓库orientation后首先负责：真实38-target native input/output hooks、chunked bank accumulator、
signed rank4 compiler、free-code oracle optimizer及其strict250 evaluation wiring。它们必须形成一个canonical实现面，不恢复任何
退役Writer或realizer。
