# EMBER ECP Native-Factor Compiler

状态：**历史架构合同，已于2026-09-02被
`docs/program_conditioned_native_bank_tangent_transport_design.md`取代。** 下文“当前/active/下一步”只表示当时历史时点，不得恢复执行。

2026-09-01第七次专家复核与当时owner裁决曾将其登记为active architecture contract。前四次专家审查依次建立Native-Factor主线、
要求current-bank conditioning，并把不可接受吞吐的full functional-polar降为fit-only reference。随后
formal S1与定向S2依次淘汰rank64 native-Q sketch、mean/variance scalar energy、query-conditioned多步set scorer及rank224/384
cross-image。授权fit-task的cross-episode真实flow-gradient诊断进一步证明：behavior rank4方向在三条真实bank中均有约`.90--.91`
signed-pooling可达性，但旧共享selector在held video只有`.023`，bank-independent native dual只有`.075`；同一task primal经每条当前
bank的全局单位质量covariance对偶化后，fit/held立即恢复到`.904--.911/.901`。因此当前活动修正不是继续叠加scorer，而是让共享
Program预测target-native primal，由当前video bank的确定性全局covariance solve把它变为dual query，再对真实X/Y做exact signed replay。
P1随后证明该operator在六tasks、四family和held videos上达到`.9545`。但95-task behavior-sufficiency诊断又发现：fit75行为流形对
held20具有`.7160` rank16可达性，旧frozen G2 Program的shared读出却只有`.247--.270`且不优于language-only，因此最早失效接口
进一步上移为G2跨task behavior identifiability。随后pointwise decoder与decoder-free v3--v5依次失败；v5虽扩大Program跨task
spread并保持旧动态Gate，却没有对齐真实policy behavior。第四次专家复核明确：这些实验只淘汰对应的独立reader与behavior-Gram目标，
尚不能把联合失败归因于Program schema或Stage0。当前取消“Program先独立通过behavior-Gram”的硬Gate，唯一活动资格改为联合训练
Natural Program与shared target-native primal scorer，并直接以generated rank16 LoRA的跨episode functional loss给credit；P1的
current-bank operator在J2--R11中保持冻结。后续cross-bank正控现已重开该operator的**因果交互**而非容量，详见下文当前裁决。
本文当时仍是唯一架构依据。第五次专家复核进一步确认P0/P1与R5保留的是capacity primitive，而R12/R13已经充分终止binary
full/half门卫。随后scalar/base-score及32维vector pointwise candidate interaction均在fixed-route资格中暴露capacity--specificity
冲突：它们要么同时保留correct/wrong，要么以损失correct换取wrong suppression。第六次专家复核据此终止当前set-independent
pointwise函数类；当前唯一新增接口是在capacity-preserving full query之后、exact signed pooling之前加入Program-relative、
event-conditioned whole-bank summary，再由该summary条件化同一套逐candidate continuous correction。最终仍只形成一套signed measure、
rank4 residual和完整rank16。

R5随后用training-only fixed route建立并正式通过稳定functional chart，但R6移除fixed route、接回Natural Program后的step110
train/held只有`.165/.143`，Program-to-code约`.02`；同task跨video却约`.9994`稳定。共同heads与minimum-norm held-view审计证明R5 chart
是没有Natural Program内容几何的fixed-token codebook，R6 functional-only credit没有完成内容坐标获取。R7冻结R5 native heads并以
gradient-task validated outer-update directions训练Natural Program及feature chart，内部方向升至`.64--.74`，但step110真实train/held
仍为`-.133/-.130`且target role全部为负；因此“Program侧适配冻结任意chart”也已淘汰。当前下一资格保持同一绝对code labels、数据、
bank、rank与scale，只让Program和完整primal scorer共同取得输出坐标；训练标签仅训练期可见，任何内部fit都不能通过G3，最终仍由真实
bank、signed replay、唯一rank16和原12-task functional Gate裁决。

R8--R11随后完整检验了fresh联合、稳定content初始化、真实functional refinement和matched raw Stage0。R10已把train/held提高到
`.560/.544`并通过四family与wrong-Program，但task-held仅`.151`且wrong-bank/interaction近零；R11 raw Stage0反而把task-held降到
`-.092`，排除Program schema首因。最新R5成功primal cross-bank正控又证明错误task bank可保留`100.4%`中位功能收益，故当前最早接口
已经从Program/scorer移到global-`C^+d` operator的bank交互可识别性。下一资格不是新的Program版本，而是先建立同时保留same-task
跨video能力与correct-over-wrong bank必要增量的task-local operator正控。

七次专家原文分别逐字保存于`docs/expert_review_20260824_native_factor.md`、
`docs/expert_review_20260826_bank_conditioned_native_factor.md`、`docs/expert_review_20260828_g3_functional_sketch.md`和
`docs/expert_review_20260829_joint_program_primal.md`、`docs/expert_review_20260830_program_bank_interaction.md`和
`docs/expert_review_20260831_event_conditioned_bank_set_relative_interaction.md`和
`docs/expert_review_20260901_program_through_bank_bottleneck.md`。本文是将专家
原文与owner后续裁决转成可执行合同的解释层，不替代原文；任何疑似曲解或冲突先核对原文，再按owner最新明确表达修正。第二次专家建议Final默认从通过Gate的组件初始化；owner
最新明确补充，整套Writer完全随机初始化并直接端到端fresh训练必须保留为Final正式可选项，G1--G3不构成强制训练课程。

owner后续明确取消专家原文中的阶段工期估计、固定修正次数、结构版本上限和训练轮数上限。本文保留专家的Gate与失败定位逻辑，
但不把时间或尝试次数当停止条件；修正必须由新证据驱动，整体推进应在保质前提下尽可能快，顺利时力争数天内完成完整架构实现
并推进关键Gate。

人工process数据、神经`q_pi`、fixed effect-code realizer、fit-task fixed span、PECS/GOMQ/v24和并行旧Writer均不属于本路线。

### 2026-09-01 active amendment：Program-through-Bank Bottleneck

本节覆盖下文与其冲突的旧EBSRI B0/B1细节。Program仍可生成native event queries、base primal和event weights，也可作为query读取当前
真实candidate set；但B1不得直接读取raw、centered或relational Program state及其它高维task code。B0 set read必须形成与真实执行作用域
匹配的input `[target,rank,event,S]`、output all/by-type `[target,group,rank,event,(type),S]` response；B1只由这些bank responses与
固定owner/rank/event结构形成candidate correction，不得存在summary-independent的task correction。summary values只能来自candidate-
derived信息，Program query本身或hidden residual不得旁路传给B1。

最终候选轴、单位质量、video边界、chunked online accumulator、exact positive-minus-negative pooling、rank4 residual、carrier12与唯一
完整rank16合同均不变。当前资格链只包含：topology-matched structured free-summary S0 → real Program-through-bank S1 → fresh
direct-functional shared S2。不得恢复quotient、effective surrogate、unit-gradient polish或把内部loss用于选模。若S0在实现与Panel-B
核验后失败，停止当前fixed-base+summary-only correction函数类并把bank response前移到primal；若S0/S1通过而S2在gradient tasks通过、
task1/93 held interaction稳定失败，则停止当前shared coordinate并重新裁决canonical coordinate或task diversity。

执行结果：scope-matched S0正式通过；real Program-through-bank S1在两task均因correct/held不足正式non-pass，故按上段预注册条件没有启动
shared S2，而是执行专家§7.1的bank-conditioned-primal失败分支。该pivot能恢复task1/93 correct与same-task held，但原query、充分校准
Q_free、nested full-native A_free及anchor与candidate同量级后的最终裁决都无法在task93同时满足wrong与margin。最终充分行使版本为
correct `.853/.859/.818`、wrong `.612/.669`；同checkpoint零锚和逐层几何把最早缺口定位为高相似summary经family-scalar gate调制
共享event-additive anchor时的功能分离不足。该具体parameterization现已停止，本文下方旧“当前S2”等时点措辞不再恢复为执行路线；
在owner作出新的结构裁决前不启动shared/Natural Program或新的训练版本。

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
                -> current-bank full primal-to-dual base query
                -> Program-relative rank x event candidate coordinates
                -> event-conditioned whole-bank moments / induced summaries
        +-- Pass B1: replay or reuse the same native bank
                -> unaggregated Program event queries
                -> direct condition-generated candidate head from Program / local event / whole-bank summary
                -> bounded centered per-candidate branch-logit correction
                -> one exact antithetic signed measure over real native X/Y
                -> one task-specific rank4 residual
                -> concatenate frozen shared rank12 carrier
                -> one complete 38-target rank16 LoRA
```

Pass A与Pass B读取同一个冻结backbone，并共享owner、event、video assignment。Pass A回答视频表达什么目标、场景和过程，并把8个
event内容保留为owner/rank/event query；Pass B0从同一video的真实native candidates累计全局单位质量covariance、full base query，
并以全部4 rank x 8 event native queries测量整个candidate set，形成每event的相对分布摘要。Pass B1重读或复用同一bank；每个candidate
的correction同时读取自身Program-relative坐标与B0 whole-bank summary，再以一个positive/negative softmax之差精确pool真实X/Y。
Program不直接成为LoRA factor，B0 summary不得输出match类别、开关或LoRA方向，covariance/dual/summary与interaction output也不进
checkpoint；最终Value仍必须来自当前视频真实activations。B0与B1是同一个Writer的两次流式读取，不是两套模型，且共同发生在rollout
前唯一一次Writer运行中。full/half二值route、set-independent pointwise scorer、full functional-polar与其它谱端点只作历史诊断，
不是active forward。

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
2. **B1 co-conditioned exact global signed replay**：重读或复用同一video chunks，以同一个全局measure计算base score
   `l_n=q^T v_n`，把每个rank的base score RMS统一到`.02`；随后由未聚合的Program event query、当前video的local event context和
candidate content产生bounded、measure-centered、逐candidate的positive/negative branch correction，最后执行一套显式signed
softmax并只对真实X/Y values加权求和。input侧没有type轴；output侧四类bank共同构成候选集合。interaction最后层零初始化时必须
严格退化为既有full base replay。

实现允许不显式物化`delta`的全局measure均值：对每个rank减去该均值只会给positive与negative branch分别增加一个候选无关常数，
两个softmax各自严格不变。因此canonical streaming保留这一softmax gauge的未定center表示，功能上与显式measure-centered公式等价，
并避免为不可观测常数增加第三次全视频读取；不得把frame chunk各自中心化，因为那会改变chunk间相对measure。

solve与replay必须使用完全相同的全局measure。此前“全局solve后按event分别replay”实际施加`C_e C_global^+`，会重新旋转方向，已由
task93/q20反例淘汰；event assignment与canonical alignment仍通过G2 Program的`rank_event/event_weights`决定primal内容，不再改变
native replay measure。covariance条件数可接近`1e6`，所有stats、dual score与signed replay必须使用禁用TF32的IEEE FP32；策略冻结forward
仍可使用BF16。

两次读取都必须按video维护首帧、末帧和跨chunk previous activation：`Y_adj`使用同一video上一采样帧，`Y_init`固定该video首帧，
`Y_goal`固定该video末帧。chunk边界不得重置，video之间绝不串接。B1两个softmax分支分别维护running maximum、normalizer与weighted
sum；chunked结果必须与同输入non-chunked reference在正常数值误差内一致。

## 4. Native-factor compiler

G1与G3的边界不变：G1可直接优化task-local selection logits/weights，回答是否存在强rank4组合；G3必须由跨tasks共享的完整Program
与当前bank candidate共同产生weights，不能包含task/video/frame/member lookup。固定target-native heads和
owner/family/rank/event/group embeddings只表示38个真实LoRA owners及其拓扑，跨所有tasks/videos共享，不是task route。

对某个target/native group，令当前video真实candidate为`v_n`、全局单位质量为`mu_n`、Program primal为`d_jr`。保留full base query：

```text
m       = sum_n mu_n v_n
C       = sum_n mu_n (v_n-m)(v_n-m)^T
q0_jr   = C^+ d_jr                         # capacity-preserving full base query
l0_jrn  = alpha_jr q0_jr^T v_n             # alpha fixes base-score RMS to .02
```

继续保留未聚合的`rank_event[j,r,e]`，复用owner×group native heads得到event-specific full-native query `u_jre`。对candidate
`n=(t,p,h,u)`，interaction scorer同时读取：`u_jre`与centered native value的normalized full-native alignment、Program event query与
当前video local process/sigma/presence/tau及frame-to-canonical-event assignment的semantic alignment、二者乘积、candidate RMS以及
probe/horizon/type metadata。它输出最后层zero-init、bounded且在当前measure下centered的逐candidate correction`delta_jrn`：

```text
lplus_jrn  = log(mu_n) + l0_jrn + delta_jrn
lminus_jrn = log(mu_n) - l0_jrn - delta_jrn
w_jrn      = softmax_n(lplus_jrn) - softmax_n(lminus_jrn)
f_jr       = sum_n w_jrn v_n                 # real native Value only
```

input侧每target/rank有一个scale；output groups各自求base dual，但同一target/rank使用跨groups公共scale，避免独立归一化抹掉block相对
幅度。interaction不是condition级开关或幅值：换bank会改变candidate方向、local semantics、assignment、softmax normalizer与最终真实
Value。`interaction_off`只作同一checkpoint的因果评测臂；canonical forward始终只有一套continuous signed measure，不再选择full/half、
`.75`或其它谱坐标。Program高维输出仍只是查询意图，最终factor必须经过当前bank真实X/Y Value路径。

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

J2可训练：Natural Program已有的local patch/language projections、38个language queries、owner scene reader、transition matcher、event
binding、ordered segmenter、Dynamic-K alignment/aggregation，以及共享`ProgramNativePrimalScorer`。source、Native Stage0、current-bank
covariance/dual/replay operator、carrier、target scale和Action Meta冻结；真实X/Y与operator仍保持可微地把functional credit传回Program和
primal scorer。J2通过后的F4才恢复scale，G4才解冻其余经资格验证的Writer职责；backbone与carrier始终冻结。

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

2026-08-24专家原回复按概念依赖把Natural Program列为Stage 1、capacity oracle列为Stage 2，但当时最终明确capacity oracle是首个
下一步；该阶段现已完成。为避免历史编号混淆，仓库沿用以下阶段名，当前执行点以J2小节为准。

### G0. Authority冻结

固定fold manifests、baseline rows、carrier/task-expert/effect authority和role-balanced sampler，不做模型选择，不创建新数据。

### G1. Native-factor task-local capacity oracle（已通过）

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

首轮process-only pointwise decoder已由clean detached `5cbe76e`从fresh optimizer运行到macro60。训练behavior loss由`1.2723`
降到`.7080`，旧动态增量仍从macro10的`31.85%`升到macro60的`39.40%`，但独立panel-B exact rank4只沿
`.1837/.2622/.2938/.2828`变化，最终consensus仅`.3027`；meta/target held分别`.3803/.1853`。冻结macro60 Program后的新reader仍只有
约`.262` held，full Program的fit pairwise behavior topology只从旧`.1610`变为`.1694`，official held仍约0。该实现因此明确
non-pass：decoder学会了fit task code，却没有迫使部署Program形成跨task可迁移的policy-behavior几何。继续增加decoder容量、训练时长或
普通超参不会修复这个credit ownership问题，pointwise decoder已从active路径删除。

下一资格先把原fit75固定拆成role-stratified internal train60/internal held15；原meta-held15+target-held5作为official held20保持
不参与本轮训练、选模或架构修正。train60建立的rank16 basis在internal15对独立panel-B/consensus为`.6184/.7158`，q/v/action-in/
action-out为`.6556/.7373/.4550/.6676`，证明这个内部task holdout具有足够但非平凡的可判别上限。

首个decoder-free修正不再增加training-only reader参数，而直接让部署Program拥有behavior credit。每个condition从完整固定schema构造block-equal
feature：`P_lang`、`P_scene`、`sqrt(rho)*P_process`、`sqrt(rho)*sigma`、`rho`、`tau`；每个block独立单位化后保留owner/event顺序。
每个role内把两组disjoint same-K video views的centered off-diagonal Program cosine kernel，对齐到train60的panel-A与consensus
factor-cosine kernels，并约束两view kernel一致。该loss不含task decoder、task/frame lookup或held label；其梯度直接进入现有
language/scene/process/alignment Program参数。Stage0 v3、source、Action Meta、Program schema、原动态loss、uniform K及唯一执行入口均保持。

每个formal optimizer step固定5 meta+5 target、五卡各处理一对role任务；每macro访问全部15 target-gradient tasks和轮换的15/45
meta-gradient tasks。Gate先只读internal15：train topology相关至少`.50`、internal held role-equal至少`.25`且meta/target各至少`.25`；
随后用仅在evaluator中由train60拟合的fixed kernel-ridge readout，执行原exact rank4、四family、wrong Program、language增量、K/view与
旧动态资格。该readout不是模型参数或deployment路径。official held20继续冻结；只有internal Gate先通过后才获得一次最终确认资格。
若train topology本身不升，最早接口仍是Program credit；若train升而internal meta/target不升，才依据该证据重开Stage0 grounding的窄尾部，
不解冻整个observer。behavior通过而动态Gate回退时再处理表示—事件职责冲突；shuffled/reversed仍不使用。

v3已从clean detached `60fb18b`完成macro5/15 updates和全部internal Gate。局部5+5 batch的correlation最高达
`.7036/.7037`，但全量train60只`.2315/.2358`；internal target的`.7842/.7930`基本复现旧Program偶然高值，
internal meta仅`.2152/.2332`，exact panel-B/consensus仅`.1207/.1253`。旧动态Gate仍通过，official held20未读。

对确定性sampler的图分析进一步表明，v3在meta45上只给126/990个task pairs提供相对约束，并分成5个互不连通的
components。这个objective允许每个小组在彼此无约束的坐标中各自对齐；因此该non-pass的最早接口是全局credit graph，
不是Stage0、Program schema、梯度路径或训练时长。

v4在每个原5 meta+5 target batch中以`.5/.25/.25`对齐joint/meta/target kernels，把监督图连成唯一60-task component；但
clean detached macro5 formal的train topology仍只有`.2360/.2362`，internal meta`.2064/.2257`、exact panel-B/consensus
`.1129/.1177`，旧动态Gate则保持通过。固定checkpoint显示full Program跨task cosine约为均值`.965`、标准差`.020`，teacher为
`.145/.316`。原逐batch centered、normalized objective允许每批独立的affine gauge和near-collapse，因此joint graph没有规定
全局绝对几何；v4被淘汰，official held20未读。

v5保持全部Program schema、数据、5+5 role质量、K、动态loss/Gate、Stage0冻结与information wall，只把behavior target改为固定PSD
lift `K_target=(1+K_behavior)/2`；raw Program off-diagonal Gram及cross-view差异按完整scope的固定per-owner teacher dispersion
缩放。clean detached `7f4df1b`的macro5 formal中旧动态增量`20.8602%`通过，但train topology仅`.2160/.2208`、internal meta
`.2022/.2169`、exact panel-B/consensus`.1054/.1289`且wrong margin`-.0466`，明确non-pass。冻结block审计显示full/process
std从v4的`.020/.086`扩大到`.046/.220`，其teacher consensus相关却从`.150/.135`降到`.142/.131`：固定绝对gauge能让
Program拉开tasks，却没有把拉开方向对齐policy behavior。official held20未读。

因此没有v6或其它独立G2 behavior-geometry修正；不续训v5、不解冻整个Stage0，也不做seed/LR/width/rank小扫。第四次专家复核指出，
固定block-equal、各block单位化并等权的Program Gram既过强又与最终功能坐标不匹配；`rho/tau`还被复制到selected owners，使约三分之一
几何在不同target native kernels之间被强制共享。v5仍是合法的protocol non-pass，但不能证明Program schema或Stage0结构性不可能。
当前因此删除这个独立硬Gate，让Program和相邻primal scorer共同接受generated-LoRA functional credit。

### J2 / G3. Joint Program--primal functional qualification

当前最早未解决接口是Natural Program与shared `ProgramNativePrimalScorer`之间的联合可识别性和functional credit，不再要求二者先后
独立通过人为规定的参数或behavior几何。使用cross-episode teacher action/flow panel A计算generated-LoRA真实功能损失；完整链为
`raw frozen evidence -> Natural Program -> shared primal -> frozen current-bank dual/replay -> unique rank16 LoRA -> functional loss`。
只有Program与primal scorer训练；source、Native Stage0、current-bank operator、carrier、scale、Action Meta与policy weights均冻结。
teacher action/flow只在授权训练task上形成loss，绝不进入deployment输入；task-held、held-video panel B和checkpoint选择外条件均零梯度。

首个12-task qualification固定gradient meta`[1,8,9,32,52]`、gradient target`[72,73,75,93,94]`、true task-held meta`2`和
target`74`。每个gradient task以两条mapping-fit K1 videos和panel A训练，第三条预注册video与disjoint panel B只读；selected浅/中/深
q/v及action-in/out八targets只负责分family报告，Writer始终实际生成38-target完整LoRA。Natural Program从
`c1493a1/macro20`共同model tensors初始化，primal scorer、optimizer和scheduler fresh；不得加载v3/v4/v5 checkpoint。

在完全相同functional loss上先优化每task跨两条fit videos共享的task-local free primal，第三video零梯度。正控要求held-video recovery
median`>=.80`、四family各`>=.70`且每task显著优于carrier；失败首先归因于functional panel、scale或teacher-to-utility authority，
不能归因于shared Program。正控通过后进行joint训练。为消解第四次专家原文“100 total updates”与停止条件“约100 post-warmup
updates”的口径冲突，formal按最多10步warmup后100个有效joint updates计数，在post-warmup 60和100保存相邻checkpoints；总
optimizer steps最多110，不把低LR warmup冒充充分优化。

该10-task正控现已formal通过：held/fit functional benefit retention median`1.0144`、held factor recovery median`.8078`，四family
medians均`>=.7722`，且10/10 held panel-B分别经paired t与sign test显著优于carrier。physical microbatch4只在最长task93出现
`37.07GiB`峰值；不改变logical16/noise/bank/loss的physical2 profile将其降到`32.41GiB`，step1 loss相对差`.060%`，因此active
runtime锁定physical2。该pass只清除了functional authority/scale/free-primal与系统吞吐前置问题，不构成shared Writer通过。

Primary normalized functional recovery为
`(L_carrier-L_shared)/(L_carrier-L_free_primal)`。12-task Gate同时要求：gradient train median`>=.60`、same-task held-video
median`>=.50`、两个true task-held平均`>=.40`且各`>=.30`、held/train`>=.80`；q/v各`>=.35`、action-in/out各`>=.30`；
full相对language-only与endpoints各`>=.10`；correct-vs-wrong Program和bank margins各`>=.10`，interaction`>=.05`；same-task-other
retention`>=.80`；post-warmup 60到100 task-median回落不超过`.05`；event不坍缩、K1 identity和信息墙继续通过。factor/update
cosine只作secondary geometry diagnostic，不能替代generated-LoRA functional result。

J2现已按上述完整预算formal执行并明确non-pass。step110 train/held-video recovery仅`.1708/.1646`，四family约零，wrong Program/bank
与interaction也近零；checkpoint审计同时证明gradient到达全部有效模块、clip未触发，但生成Program/primal仍高度task-common，而
task-local成功primals和真实functional gradients均强烈task-specific。因此不能进入“train高、task-held低”才允许的raw Stage0分支，
也不能续训或做普通optimizer/width/rank小扫。

J3 counterfactual functional routing已按上述边界formal non-pass。step110 correct train/held-video仅`.1486/.1477`，低于J2；错误
Program/bank虽在多数task上变坏，correct只有3/10改善，四family仍约零。因此不再叠加counterfactual、normalization或optimizer技巧。

后续R1用固定、非参数化的10个正交task token替代Natural Program内容，只训练同一scorer。step110 train/held-video提高到
`.2678/.2798`，wrong-token margin`.2384`且9/10 tasks优于J2，证明清晰route有实质作用；但q/v/action-in仍约零，完整Gate non-pass。
checkpoint几何确认route在scorer内部没有重新塌缩，而task-local正控在functional优化前已由teacher consensus初始化获得最终收益约43%。
R2在同一fixed-route边界加入fit-only set-valued paired-update critic后，v/action-out恢复到`.408/.663`，q/action-in只到`.220/.167`；
R3的owner×group decoder把action-in/out恢复到`.656/.669`，但primary仍non-pass。R4再用functionally validated code一次性初始化
shared heads，step110 train/held-video达到`.819/.839`，q/v/action-out及route、video稳定均过门，仅action-in`.249<.30`。只读graft
证明heads自身几乎未动，失败来自初始化所依赖的feature chart在整条context/trunk链中漂移；checkpoint heads接回initial chart仍有
`.998` action-in outer recovery。R5只冻结初始化后的feature chart、训练233 native heads后，step70/110 train recovery达到
`.934/.940`、held-video`.957/.963`，四family全部`.816--.839`且相邻稳定，正式通过全部Gate。因此moving-coordinate根因已经
  闭环成立。R4/R5仍含privileged初始化和fixed task route，不能冒充G3或deployment。R6加载passed R5共享scorer、移除fixed route并
  恢复Natural Program后，step110 train/held仅`.165/.143`，Program-to-functional-code约`.02`，完整Gate non-pass；共同heads与held-view
  拟合证明R5 chart本身没有deployment content geometry。当前R7冻结R5 native heads，只训练Program和feature chart，以fit-only task-level
  outer-update direction获取内容坐标；不读取functional action loss、task-local scale或held信息。R7内部fit不能通过G3，仍由原12-task
  generated-rank16 functional Gate负责证明shared mapping、task-held泛化和视频因果。

速度是必须持续优化和报告的系统诊断，但不参与科学qualification：冻结language embeddings、raw Stage0 evidence、X/Y、covariance
eigensystem和fixed action batches只捕获一次；不得缓存Program输出或generated LoRA。报告global update wall、evaluation/training wall、
显存峰值与持续UTL，并以真实profile选择并行度和microbatch；不能用自行设定的wall阈值否决满足机制Gate的checkpoint。显存没有人为
`35GiB`上限；以最长真实样本不OOM、allocator与共驻进程仍有安全余量为边界，在提高真实LoRA/s、functional updates/s和持续UTL时
允许使用更高显存。world size按live availability和上述实测量弹性选择，不能改变task权重或科学batch。

通过12-task Gate后恢复40 fit/10 task holdout、329 fit/40 held-video/82 task-held的完整shared functional qualification，仍联合
Program+primal，primary Gate为held-video median`>=.60`、p10`>=.35`、task-holdout median`>=.40`、held/train`>=.80`、
meta/target各`>=.35`、q/v各`>=.35`、Program/bank margins各`>=.10`、interaction`>=.05`、多数tasks为正且相邻checkpoint稳定。

若12-task train与held-video高而true task-held低，才做同split、同capacity、同loss/budget的raw frozen Stage0 sufficiency probe；输入只换为
deployment-visible owner/event process、presence、uncertainty、patch/scene transition、exact-language embedding和normalized time。
raw task-held比Program高`>=.15`且达到`.40`，才把最早接口判为Program压缩/schema；raw也低于`.25`且free primal强，才判为frozen
Stage0上游瓶颈并允许窄解冻process/presence/uncertainty tail。不得直接解冻VLM、source policy或整个Stage0。

R11已完成该matched probe并明确non-pass：step110 raw train/held/task-held为`.292321/.288053/-.092369`，相对R10 task-held
`.151475`下降`.243844`，因此Program schema/压缩不是当前首因。其q/action仍约`.47--.55`而v仅`.101550`，target-role train
median仅`.110012`，所以也未满足“所有family均缺少可读信息”的frozen-Stage0停止条件，不能据此解冻VLM、source或整个Stage0。
在新增Writer结构前，先以R5/task-local成功primal做cross-task wrong-bank functional upper-bound：若正确成功primal换错bank仍保持
效用，说明当前primal-to-dual operator与wrong-bank Gate的组合缺少可识别交互，先修正bank interaction；若该正控有强bank margin，才
继续把R10/R11失败定位到shared Program/scorer的target-task归纳偏置。该诊断只读，不恢复旧candidate scorer或退役路线。

该cross-bank正控现已在clean detached `2090799`完成：正确/错误bank recovery中位`.930860/.945799`，correct-minus-wrong
中位`-.003819`，正确bank仅`2/10`更好且`0/10`达到`.10`，错误bank`10/10`保持正收益。由此保留P0/P1的capacity结论，但撤销
“capacity通过即可把operator从bank因果首因中排除”的过强解释。当前下一Gate先要求task-local operator同时保持same-task跨video
能力与correct-over-wrong bank必要增量；通过前不得再用当前global-`C^+d`路径训练Program/scorer版本。具体interaction修正必须由
正控证据选择，不恢复退役candidate scorer、full functional-polar deployment或task/video lookup。

half operator的task-local formal在held correct/wrong上为`.725204/.188873`，margin`.541238`；唯一tempered `.75`
bridge又得到`.925312/.885043/.054500`。结合full inverse的`.930860/.945799/-.003819`，已确认单一谱幂的
capacity--specificity Pareto，停止谱幂小扫。当前先做一个只读内容信号审计：保持full-inverse成功方向，检查其
query被统一缩放到fixed replay score RMS之前所需的raw dual energy，是否能以deployment可读的绝对内容信号区分correct/
wrong bank。若成立，才在强direction外加独立、有界、不允许单bank覆盖的compatibility gate；若不成立，则重查
Program--bank semantic key，不得把成对反事实比值、task/video lookup或退役scorer作为deployment输入。

该审计与三项正控现已完成。普通raw dual energy不具正确方向，但gauge-free input retained projection p10在R5成功primal的30个
same-task video与50个same-role wrong banks上AUC `1.0`、逐task严格分离`10/10`，全局正确minimum `.907248`高于错误maximum
`.905998`。把support只乘到最终rank4 residual不能建立utility margin；固定阈值改为选择full/half operator坐标则得到
correct/wrong/margin `.950915/.005173/.908899`及`10/10`强margin。相反，固定sigmoid线性混合两套query令correct降至`.238736`，
故首版route必须是近二值内容选择，而不是连续query插值、谱幂或幅值调节。

同一hard route对R10 Natural Program step70/110完整Gate的诊断进一步显示：现有shared primal的matched/mismatched support AUC仅
`.551/.558`，逐task严格分离均为`0/12`；多数正确video被选为half，step110 train/held/task-held降至
`-.482993/-.631937/-.533894`。这不否定operator正控，而是明确证明R10没有学到该compatibility。当前shared qualification从R10
step110 model tensors、fresh optimizer开始；每个Program view以另一条same-task fit bank为positive、同role cyclic task bank为negative，
用固定input projection低分位calibration训练Natural Program与native heads，并继续用原correct pair的cross-episode functional loss维持
task direction。held video、task2/74、panel B及validation/test零梯度；route只读当前Program与当前bank，不读task ID、文件名或成对
反事实。首个mechanism Gate要求positive跨阈值、negative低于阈值、correct functional不退化；未满足前不运行完整formal G3。

首版训练实现把route supervision与task-direction supervision严格分工。每条fit Program以另一条same-task fit bank为positive，并以当前
同role cyclic task的对应交换view为negative；training support使用input projection排序第12--20位均值以分散梯度，而部署判定保持固定
p10。compatibility用温度`.02`的二元calibration；正确functional分支在训练时teacher-force已通过正控的full endpoint，避免尚未学会的
hard route先把R10功能basin送入half并制造错误credit，evaluation/deployment则没有override。三卡同图梯度profile显示weight `1.0`
压倒functional约百倍，唯一机制比例修正为`.01`后总norm`.2451`且不clip，input/output/Program梯度均finite；该值固定进入首轮formal，
不作weight sweep。

R12 formal已经裁决上述“同一functional primal兼任route probe”的实现。step110 matched/mismatched full-route为`.528/.083`，
bank margin与interaction通过，但正确视频一旦误入half，functional recovery中位从full组的`.583`降到`-1.093`；train/held/task-held
因此只有`.299/-.504/-.129`。task52/72及zero-gradient task74还出现correct support低于wrong-bank support，说明这不是移动固定阈值
或延长同一schedule能修复的校准误差。

有界credit-ownership诊断R13已完成：保持hard full/half operator、functional primals、真实X/Y、signed pooling、rank4、carrier12
与唯一rank16不变，只把route职责交给38个独立共享compatibility input heads。step70/110 matched full-route为`.639/.667`，
mismatched均为`.167`，support AUC为`.826/.831`、逐task严格排序均为`9/12`；这相对R12有增量，但step110 train/held/task-held
仅`.483/.049/.033`。gradient fit、same-task held与true task-held正确放行分别只有`16/20`、`5/10`和`3/6`，task52/72/74仍
存在正确support低于wrong-bank support的排序冲突。任何保持wrong full-route不超过`.20`的全局阈值最多放行`.722`的正确pairs，
所以失败不是固定阈值校准。

R13也显示二值route的功能脆弱性：step70到110只有task8一个fit video以约`.000060`越过固定阈值，就令该task fit recovery从
`-.208`跳到`.964`，而same-task held与task-held aggregate完全不变。独立probe因此只证明credit ownership可以部分迁移，未证明
可靠的跨video/task compatibility，更不确立full/half或其它二值开关为最终G3架构。第五次专家复核据此正式退役binary门卫；不授权
threshold、temperature、weight、LR、seed、谱幂或同类probe小扫。

#### 当前G3：Event-Conditioned Bank-Set Relative Interaction

scalar/base-score pointwise v3/v4在十task上correct/held约`.93/.95`而wrong仍约`.93`；六task correct--wrong首步functional gradient
cosine约`-.96`。随后32维vector pointwise task-local诊断虽能把task1/task93 wrong压至负recovery，却把correct降到约`.72/.60`。
这些结果终止当前set-independent pointwise函数类及其loss/width/bound/LR/seed小扫，但不淘汰whole-bank-conditioned continuous
interaction、真实X/Y、signed pooling、rank4、full base、Natural Program或Stage0。

当前唯一后继是Event-Conditioned Bank-Set Relative Interaction（EBSRI）。对每个input/output native owner，Program的4 rank x 8 event
queries直接定义每个candidate的32维相对坐标`kappa_n[r,e]`；Pass B0在每条video内按unit-mass measure与frame-to-event assignment流式
累计event mean、dispersion及一对antithetic induced summaries。input X保持`(t,p,h)`候选且没有type轴；output Y保持
`(t,p,h,u)`联合候选，并允许B0同时形成all-type与candidate自身type summary。Pass B1对`kappa`作event-relative标准化；S0实验证明
间接FiLM+共享zero-head没有取得summary控制后，当前canonical由`Program/local event/summary`直接生成每family candidate linear head，
再对每个candidate产生唯一bounded branch correction。最终仍只对真实X/Y执行一个exact
positive-minus-negative pooling，形成一个rank4 residual与一套carrier12+residual4完整rank16。

EBSRI不得成为task matcher或连续开关：B0 summary不输出类别、residual或LoRA factor，Program仍直接定义query，真实X/Y仍是唯一Value
路径。每条video独立完成B0/B1并以`beta_k=1/K`聚合；K1 identity、video集合置换不变、帧内保序、unit-mass与previous/first/final状态
不跨video。B0/B1必须支持chunked online moments/log-sum-exp，并与相同输入的non-chunked reference在正常数值误差内一致。

资格严格按一个架构的四个接口推进：

1. **S0 free-summary factorization**：只用task1/93。R5 base、native heads、真实X/Y、signed pooling、scale/carrier全冻结；同task correct
   fit views共享一个training-only free summary，wrong fit views共享另一个。只训练candidate trunk/direct condition-generated head，correct held、wrong fit1、panel B
   零梯度。free summary只是容量上界，不是部署输入、binary route或checkpoint候选。训练可用exact-effective-rank4高速目标，但裁决必须
   看panel-B functional recovery；若teacher代表元与功能结论冲突，补一次直接constrained functional feasibility，不得误杀架构。
2. **S1 real-summary task-local（已通过）**：S0通过后，以同一代码把free summary替换为真实B0 set encoder；每task独立训练，检查correct held与
   zero-gradient wrong view泛化。S0失败不实现复杂set encoder；S0通过而S1失败，首因落在set encoder/query-relative descriptor或真实
   bank content。
3. **S2 fixed-route shared task-LOTO（当前）**：S1通过后共享同一set encoder/interaction，在8个gradient tasks训练并hold out一个meta与一个
   target interaction task；通过才在全部10 tasks fresh refit形成component initialization。
4. **S3 Natural Program joint G3**：移除fixed route，联合训练Natural Program、set interaction与native heads；source、Stage0、scale、
   carrier与Action Meta继续冻结，恢复原12-task true task-held functional Gate，不加入teacher reconstruction、Program Gram、support、
   binary或谱loss。

S0/S1每task机制资格为correct fit0/fit1各`>=.85`、correct held`>=.80`、wrong fit0/fit1各`<=.25`、correct-minus-wrong`>=.50`、
全部pair correct>wrong、zero-init逐tensor复现R5、无大面积bound saturation且input/output family不系统反向。这些是专家建议的接口资格，
不是正式validation目标；唯一正式性能线仍是validation8 strict paired correct严格`>145/400`。吞吐秒数只作工程目标，不作科学Gate；
fixed-route阶段必须缓存冻结`kappa/base score/metadata/event assignment`并按family/shape批处理，避免逐target/rank/event Python小kernel。

S1 formal aggregate `runs/outputs/pi05_ecp_event_bank_set_s1_gate_s110_a1f14e4_gpu01p01_20260831`已正式`pass`：task1 correct
fit0/fit1/held为`.942/.953/.962`、wrong fit0/fit1为`-.529/-.517`；task93分别为`.928/.905/.881`与`-.188/-.180`。
Action Meta为0、Panel B backward为0，并且只物化一套完整rank16。该结果只通过task-local real-summary接口，不证明shared mapping；
当前执行点因此是S2。`main@cdcae8b`解耦B0 summary与B1 replay chunk后，wrong约`12s -> 6.3s`、task1 correct约
`35.5s -> 13.3s`，峰值约`41.1GB`；吞吐profile不参与S2资格。

#### 已完成的Frozen-Program G3历史证据与可复用面

下列P0/P1与旧F0--F3结果继续约束实现；编号不再构成当前执行顺序。P0/P1有效，旧P2 frozen-Program formal被J2取代，不能因旧配置
中的`deployment_candidate`恢复。

此前G3按以下因果顺序推进：

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

上述旧P1/P2 teacher边界仍适用于历史artifact：provenance只用于loader pairing，teacher factor、behavior gradient、covariance、dual、
member identity与authority ID不得进入checkpoint或deployment输入，held video/task不得产生梯度。J2不以teacher factor为primary，而以
generated-LoRA cross-episode functional result为primary；它只联合两个相邻未识别接口，不得解冻已通过的operator或掩盖正控失败。
任何阶段都不得恢复旧realizer、functional-polar deployment或task lookup。G1、原G2动态Gate与P0/P1结论不受影响。

### G4. Joint Writer

只有J2及完整shared functional qualification通过并取得真实closed-loop信号，才解冻全部Writer；backbone、carrier和task experts继续冻结。

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

部署时Pass A生成唯一Program；Pass B0流式统计当前native banks并求full base queries，Pass B1重放同一banks，以Program event、
local event context和candidate content形成逐candidate correction后执行唯一exact signed pooling，生成rank4 residual并与carrier拼接；
安装唯一rank16 LoRA后闭环运行，不再观看teacher video。

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

当前基线代码已经实现真实38-target native input/output hooks、abs/adj/init/goal banks、跨chunk/视频边界状态、G1 free-code、G2 Natural
Program、Program-to-target-native primal scorer、per-video global covariance、截断谱primal-to-dual solve、IEEE exact signed replay、uniform
K aggregation、small-core rank4 canonicalization、rank12+4唯一rank16和451-condition mapping/evaluator wiring；Action Meta默认关闭。
fit-only consensus只属于mapping loader，covariance/dual/teacher不进入checkpoint。P2另有非checkpoint、非deployment的run-local compact
frozen-condition cache，仅保存Program、raw X/Y、final Y和B0 operator；fit retained、held ephemeral，四类Y bank仍在线构造。clean pushed
detached `e2f9d33`的完整38-target K1/K4
P0已通过：chunk4/one-chunk等价、全部梯度、Action Meta 0、uniform K、唯一rank16 materialization与真实policy consumption均成立。
clean pushed detached `c9e8198`的P1六任务formal也已通过：fit/held median`.971731/.954539`、held/fit`.982308`、held相对
optimistic median`.992193`，四family medians均`>=.9398`且minimum task held`.935001`。这在当时排除了operator的capacity与
same-task跨video稳定性；最新cross-bank正控说明它并未验证跨task bank因果可识别性。随后
95-task证据及G2-B v3--v5只证明独立Program behavior-geometry目标失败。第四次专家裁决要求复用已有P2 cache/operator/scorer，新增
Program与primal联合functional反传、task-local functional正控和12-task Gate。当前代码已在唯一`joint_program_primal`执行面接通
可微Program compile、functional LoRA chain rule、free-primal正控、role-balanced joint update、Gate wiring及不缓存Program的v4 frozen
condition cache。10-task正控已通过；J2/J3与R1--R3均形成formal non-pass/partial结论。R4 functional-code初始化已把train/held提高到
`.819/.839`，但action-in因feature chart drift未过门；R5冻结初始化chart、仅训练全部native heads后step110 train/held为
`.940/.963`且四family全部通过。R6--R9排除了简单Program接回、冻结chart acquisition与outer-code替代utility；R10真实functional
refinement取得`.560/.544`，R11 matched raw Stage0则明确更差。clean detached `2090799`的cross-bank正控最终显示正确/错误bank
recovery中位`.931/.946`，因此global dual只作为capacity-preserving base query，不再是独立的deployment-qualified causal operator。

旧`C=I`、P_lang-only、joint residual和full-Program Euclidean normalized-bilinear实现均已有formal non-pass并从active执行面退役；历史
由Git/config/artifacts保留。旧full functional-polar、native-Q sketch、旧query-conditioned set scorer及scalar/vector set-independent
pointwise correction只保留为fit-only reference或diagnostic，不得再以deployment候选启动formal。R12/R13 full/half route、support
classifier、threshold与selected inverse power同样只由Git/config/formal artifacts保留，不属于active forward。当前唯一实现面在Program
primal + current-bank full base query + exact signed replay之间加入Program-relative event bank-set summary与summary-conditioned continuous
correction；它继续复用真实X/Y、rank4与唯一rank16，并严格按S0 free summary、S1 real summary、S2 fixed-route task-LOTO、S3 Natural
Program joint四个最早接口推进。S0/S1已经通过，当前唯一下一接口为S2 shared task-LOTO；S1不构成shared mapping证据。旧vector分支不得合并，新EBSRI必须在同一canonical scorer/streaming owner内替换旧pointwise职责。
后续必须保持一个canonical运行面；不得恢复退役Writer/realizer、GOMQ/PECS、人工process、task lookup或第二adapter。
