# EMBER-ECP: Event-Conditioned Policy Compiler

状态：2026-08-22 **active design / implementation authority**。本文吸收第二轮专家最终复核与owner随后裁决，取代
`docs/functional_adaptation_successor_design.md`和
`docs/policy_native_dual_time_program_compiler_review_20260821.md`的执行权。前者继续描述已经封存的16维
functional-adaptation基线，后者保留最初送审方案及其问题；两者均不得再启动formal训练。

## 1. 目标、边界与核心判断

部署合同不变：

```text
exact task language + K条same-task、action-hidden、内部有序的正确教学视频
    -> shared Writer在rollout前运行一次
    -> 一套完整38-target rank16 task-conditioned LoRA
    -> frozen source PI0.5
    -> 从未见初始化闭环完成同一task
```

新主线的核心判断是：语言与视觉变化先形成有序event，event再与PI0.5完整的layer × joint-horizon控制结构条件绑定，
形成target-family-aware Program；固定compiler先证明该Program能产生policy-effective LoRA，之后才训练action-hidden
video到Program。视频理解与LoRA生成共用同一个Program，但不把Action horizon误写成第二条因果时间轴。

首版不做以下事情：

- 不把视频转成伪action再模仿；
- 不让部署policy在rollout中反复观看教学视频；
- 不部署第二adapter、expert、task-ID字典或checkpoint融合；
- 不给LoRA rank赋予event、phase、layer或skill语义；
- 不把raw A/B、单一expert response或人工规则构造的确定性`P*`当成Program真值；
- 不恢复全局16维code、hard `video-time ↔ action-horizon` transport或全局无界M2P。

## 2. PI0.5中两个容易混淆的坐标

PI0.5的flow训练定义是：

```text
x_u = u * epsilon + (1-u) * a,       u in [0, 1]
v*(x_u, u) = epsilon - a
```

这里用`u`表示flow/noise time，避免与教学视频frame index混淆。推理从`u=1`的纯Gaussian noise向`u=0`的clean
action chunk积分。canonical observer在每帧输入：

```text
epsilon_fixed : [50, 32]
u              : scalar = 1
```

因此，`u=1`表示50个suffix token的**数值内容处在纯噪声端**。它不取消这些token的horizon位置：第`h`个token仍是
future action slot `h`的占位，具有独立position id。PI0.5的suffix mask为`[1, 0, ..., 0]`，50个token处在同一个
attention block中并可相互读取；它们是有位置组织的joint future chunk，不是从`h=1`自回归推进到`h=50`。

canonical probe一次forward得到的是：

- 每个layer、每个horizon位置上，source policy在固定纯噪声flow point的context-conditioned内部response；
- 50个位置联合预测的初始velocity field，而不是已经去噪完成的50步动作；
- 一个可比较的policy坐标。替代noise/time与完整denoised action只用于稳定性门和privileged teacher，不扩成部署输入。

教学视频的有向时间使用`s=0...T_k-1`。模型只对`s`的event order施加单调/有序约束；event可自由读取全部50个`h`。

## 3. 固定尺寸与最终张量合同

首版默认值如下；它们是实现选择，可在正式launch前因profile修改，但不得改变因果轴：

| 名称 | 默认值 | 含义 |
| --- | ---: | --- |
| `K` | 训练覆盖1--4 | 同task教学视频数，部署动态 |
| `T_k` | variable | stride=5后的第k条视频frame数 |
| `P_img` | 256 | 每帧PaliGemma image tokens |
| `L_A` | 18 | Action Expert layers |
| `H` | 50 | joint future-horizon positions |
| `D_A` | 1024 | Action Expert hidden width |
| `D_P` | 128 | compact observer/Program width |
| `M` | 4 | 每个局部时间位置的event candidates上限 |
| `E_max` | 8 | 每视频Program event容量；有效数量动态 |
| `J` | 38 | LoRA target-owner数 |
| `R` | 16 | 每target LoRA rank |
| `D_C` | 256 | compiler query/trunk width |

38个owner严格对应部署LoRA targets：

```text
j = 0..35 : 18层 × {q_proj, v_proj}
j = 36    : action_in_proj
j = 37    : action_out_proj
```

它是专家所述ragged `layer × target-family`结构的紧凑表示；每个`j`都保留`layer(j)`、`family(j)`、输入宽度与输出宽度，
不会把不存在的`layer × action-in/out`单元填成伪语义。

最终Program为：

```text
P_lang    : [38, 128]
P_scene   : [38, 128]
P_process : [8, 38, 128]
rho       : [8]             event presence probability
sigma     : [8, 38, 128]    process uncertainty / disagreement
```

同一结构由privileged policy teacher `q_pi(P)`与部署video encoder `q_V(P|L,V)`共同识别。Program不是手工标签；
teacher输出分布，video encoder学习其中由视频可识别的部分。

## 4. 部署前向：逐网络、逐矩阵

### 4.1 Frozen semantic prefix

exact language与每帧RGB进入冻结PI0.5 PaliGemma原生prefix：

```text
language native states         Z_L_native : [N_L, 2048]
frame image native states      Z_I_native : [K, T_k, 256, 2048]
Writer semantic projections                  2048 -> 128
task-grounded language queries Q_L          : [N_L, 128]
projected patch states         S             : [K, T_k, 256, 128]
```

PaliGemma、vision tower与原生language model始终冻结；不训练Text/VL Meta-LoRA。language形成task query、目标和prior，
不能独立替代video process Value。

### 4.2 Frozen native Action observer

对每个教学帧，使用该帧真实image prefix、exact language、固定`epsilon_fixed[50,32]`和`u=1`运行一次native PI0.5
joint forward。首版捕获全部18个layer地址，但在线立即压缩：

```text
X_in[s,l]    : [50,1024]   每层input LayerNorm之前的Action hidden
DeltaX[s,l]  : [50,1024]   该层residual increment
X_final[s]   : [50,1024]
V_flow[s]    : [50,32]     只作owner端点摘要/诊断
```

不长期保存所有raw tensors。共享post-capture projections加入小型layer/family conditioning，得到：

```text
H_owner[k,s,j,h,:] : [K,T_k,38,50,128]
```

- q/v owner分别读取对应层的`X_in`与`DeltaX`，使用独立family conditioning；
- action-in owner读取fixed suffix、`action_in_proj`前后状态与早层摘要；
- action-out owner读取final hidden、velocity response与末层摘要；
- 同一个source-policy prior与真实visual transition都保留，不把Action hidden仅当Query后丢弃。

canonical probe另有一个antithetic或替代probe，只在Stage 0稳定性面板运行。Program若随noise realization大幅换手，observer
gate失败。

### 4.3 Task-grounded visual transition与event candidates

相邻帧、短窗口、initial/current/final relation经共享TransitionMatcher处理：

```text
MatchedPair[k,s,delta] = CrossAttention(Q_L, S[k,s], S[k,s+delta])
delta in {1, short-window, initial-current, current-final}

C_event[k,s,m,:] : [K,T_k,4,128]
conf[k,s,m]      : [K,T_k,4]
```

candidate表达task-relevant对象/关系的before、after、change、持续/完成证据和置信度。它不是简单hidden相减，也不由
absolute frame index定义。

### 4.4 Event-Conditioned Horizon Binding

每个event candidate自由读取其owner对应的全部50个horizon positions：

```text
A_h[k,s,m,j,h] = softmax_h(
    q_event(C_event[k,s,m]) dot k_policy(H_owner[k,s,j,h])
)

R_event_to_policy[k,s,m,j,:] = sum_h A_h * v_policy(H_owner)
```

同时做反向co-attention，让owner lattice读取本地event candidates：

```text
A_e[k,s,j,h,m]
R_policy_to_event[k,s,j,h,:]
```

两方向经bounded residual fusion形成：

```text
B[k,s,m,j,:] : [K,T_k,4,38,128]
```

这里没有`video frame s ↔ horizon h`的对角、带状或单调transport。`h`是joint future coordinate；单调性只在下一步的
视频event顺序中出现。teacher action overlap仅作为train/meta辅助alignment loss。

### 4.5 Learned ordered event segmentation

固定容量8不表示每个task固定8个事件。content-driven differentiable semi-Markov segmenter从`B`学习有序soft assignment：

```text
alpha[k,e,s,m] : [K,8,T_k,4]
rho_k[k,e]     : [K,8]
```

约束只包括：segment按视频方向有序、允许duration、skip、null/missing和相邻event合并；不规定“slot 0必须抓取”之类语义。
具体视频段落对应哪个slot完全由内容与训练信号学习。简单task可以只激活少数`rho_e`，困难task可以激活更多，输出tensor
容量仍固定。

event presence使用相对有效视频帧数的固定occupancy尺度，而不是所有task共享的learned duration分母：

```text
f[k,e]   = sum_(s,m) alpha[k,e,s,m] / valid_frames[k]
rho[k,e] = 1 - exp(-f[k,e] / 0.08)
```

因此同一有序过程的1x/2x采样不会仅因帧数改变presence，且模型不能通过全局放大duration参数让所有task一起减少激活；
`rho`仍由soft segmentation学习，`.08`只定义可比较的占用尺度，不规定slot语义或event数量。

每条视频独立得到：

```text
P_process_k[k,e,j,:] : [K,8,38,128]
rho_k[k,e]           : [K,8]
sigma_k[k,e,j,:]     : [K,8,38,128]
P_scene_k[k,j,:]     : [K,38,128]
```

### 4.6 Dynamic-K event alignment与聚合

先在event层按内容与顺序对齐不同视频，再聚合完整Program：

```text
Align(P_process_1 ... P_process_K, rho_1 ... rho_K)
    -> event-aligned mean, variance, presence, bounded set residual
```

输出为`P_process[8,38,128]`、`rho[8]`、`sigma[8,38,128]`。videos集合维置换不变，视频内部顺序不变。合法的不同procedure
不会先被raw mean抹除：聚合保留variance与bounded multimodal residual。禁止平均frames、raw lattice、最终LoRA或选择best
video。

`P_lang[38,128]`只来自exact language；`P_scene[38,128]`来自language-grounded initial/final relation；
`P_process`必须读取中间动态event。language-only baseline使用`D(P_lang,0,0)`，language+scene prior使用
`D(P_lang,P_scene,0)`，完整方法使用全部三路；三者都生成一套完整rank16 LoRA。

### 4.7 Target-family-aware compiler

compiler输入最多380个Program tokens：

```text
38 language + 38 scene + 8*38 process
```

每个LoRA target/rank有一个numeric query：

```text
Q_comp[j,r,:] : [38,16,256]
```

query优先读取同owner Program，并以强layer-distance bias读取邻层；允许小型learned nonlocal contribution，但不允许global
M2P无界覆盖。rank query只提供数值生成地址，不承载event语义。family-specific heads输出：

```text
q_proj layer l : A[16,1024], B[2048,16]
v_proj layer l : A[16,1024], B[ 256,16]
action_in       : A[16,  32], B[1024,16]
action_out      : A[16,1024], B[  32,16]
```

最终严格为76个tensors、1,287,168个LoRA values。首版不再hard split shared12/task4：prior与full都由同一个compiler各自
输出完整rank16 LoRA，不能把两套full-rank A/B相加或部署两套adapter。

## 5. Privileged policy teacher：分布式共同识别而非确定性`P*`

训练期建立：

```text
q_pi(P | L, visible video events, successful policy evidence)
q_V (P | L, action-hidden videos)
```

两者共享Program schema和visible event anchors。privileged teacher不能创造与视频event无对应的自由latent；只把policy证据
附着到语言/视频可见的event、layer与target-family地址。输入可包括：

- 同task多个真正successful adapter members；
- 每个member的完整successful trajectory与multi-phase occupancy；
- verified learner recovery trajectories，member consensus与progress-supported states；
- paired source/expert全层response、denoised action及少量JVP诊断；
- source/shared support反事实、member success与uncertainty。

某个task expert在任意learner state的输出不自动是真实recovery action。未验证off-policy states降低权重；强监督只来自真实
successful recovery、多member共识或环境progress支持。

teacher输出Program均值、对角不确定性与event presence。`q_V`不需要复制teacher不可见的policy细节；训练只对齐视频可识别
部分，同时以compiler后的policy response与closed-loop功能裁决。

## 6. 数据角色：held5为什么存在，validation8何时使用

### 6.1 train24内部19/5

held5只是一折train-authorized leave-task-out机制面，不是最终held benchmark。它存在的必要原因是Stage 1要使用successful
adapters、trajectories、policy response和reward构造`q_pi`输入与oracle Program；validation8除已完成的一次local oracle外
不能把actions/reward用于共享模型或反复设计。

每折中：

- fit19训练observer校准、`q_pi`与compiler；
- held5的privileged evidence只作为冻结teacher/compiler的输入或无梯度诊断，不拟合held code、不更新共享参数；
- 至少轮换多个固定19/5 folds，避免把一个5-task集合调成隐性validation。

### 6.2 validation8

held5不能替代validation8。Stage 0--2的结构与loss在train24 folds收敛后：

1. 冻结架构、loss、controls与checkpoint规则；
2. 使用全部24个train tasks和已授权non-held meta数据重新训练最终shared components；
3. validation8只输入language与action-hidden videos，做正式single-checkpoint development evaluation；
4. validation actions/reward不产生梯度，不拟合Program，不选择task-local code；
5. 方法冻结后才进入Test8。

71个source-seen non-held tasks可训练generic observer和提供diversity pretraining，但不能单独证明source-unseen adaptation。

## 7. 分阶段训练与checkpoint

### Stage 0A — Native observer、transition与event calibration

训练数据：71个audited non-held tasks加train24当前fit fold；只使用正确教学视频、跨episodeteacher actions、真实相邻/
短窗口帧与不改变事件顺序的速度扰动。source policy、PaliGemma与Action Expert冻结。`shuffled`、`reversed`不进入
Stage 0训练或机制选择，只保留到最终候选checkpoint的时序特异性评测。

训练参数：semantic projections、TransitionMatcher、q/v post-capture projections、event-horizon binding、semi-Markov segmenter、
presence/uncertainty heads。

跨episode action grounding必须在每个有效视频帧上先建立可识别的phase evidence，再约束event识别。对binding输出先沿
candidate置信度聚合但保留frame与38 owners：

```text
F_owner[k,s,j,:] = sum_m softmax_m(conf[k,s,m]) * B[k,s,m,j,:]
a_frame[k,s,p]   = SharedActionDecoder(F_owner[k,s])[p]
```

同一个owner pooling与ActionDecoder再读取每个event的`P_process`，由该帧soft posterior重构：

```text
a_event[k,e,p] = SharedActionDecoder(P_process[k,e])[p]
a_hat[k,s,p]   = sum_e posterior[k,s,e] * a_event[k,e,p]
L_frame        = mean_k mean_valid_s ||a_frame[k,s] - a_cross_episode[k,s]||^2
L_event        = mean_k mean_valid_s ||a_hat[k,s]   - a_cross_episode[k,s]||^2
```

`L_frame`是training-only pre-segmentation calibration，不向deployment增加action输入；它让共享decoder和frame evidence在随机
event posterior之外先看见真实phase差异。首个v3 grounding阶段不启用same-task consistency、uncertainty calibration、presence
sparsity或posterior entropy，避免这些弱目标先把posterior定义成单event；`L_event`仍直接要求segmenter解释有序phase。不得先
把所有frame action targets平均成event target后再回归，也不得把action phase转成规定slot identity、硬boundary或固定event数。

输出：`observer_native_stage0.ckpt`。门：同task跨episodeevent一致性、速度鲁棒与替代probe稳定性；固定checkpoint上的
process controls只作诊断，hidden margin不能代替后续闭环。

### Stage 0B — Action Meta-LoRA必做独立对照

Action Meta-LoRA不再是可遗忘的fallback。首轮native observer得到可比较基线后，必须单独训练一个所有train/meta tasks共享、
无task-ID的Action Meta-LoRA observer calibration：

- 只在Stage 0目标上训练；
- 不与`q_pi`、compiler或`q_V`共同更新；
- 用matched native/Meta-LoRA observer面板比较probe稳定、event一致性、process specificity、same-task鲁棒与成本；
- 若没有可复现的负面效果，则采用Meta-LoRA版observer；若有明确退化，保留native版并归档负证据；
- 选择后永久冻结Meta-LoRA，使其成为observer/backbone calibration，不允许后续与compiler共同旋转。

该裁决发生在昂贵Stage 1之前，避免以后重训整套compiler。输出：`observer_action_meta_stage0.ckpt`或native authority记录。

### Stage 1 — Privileged `q_pi`与compiler realizability

在当前fit folds训练`q_pi`与compiler。loss至少包括：

1. successful/verified-learner multi-state policy-response distillation；
2. successful behavior与source/shared support preservation；
3. task-equal closed-loop success/progress；
4. multiple-member set consistency与uncertainty calibration；
5. event order、presence、owner/layer/family locality约束；
6. prior-only与full Program的一套完整LoRA反事实。

输出：`policy_teacher_compiler_stage1.ckpt`。held5 oracle gate不拟合held Program，只让冻结`q_pi`从held privileged evidence
产生分布、compiler生成一套LoRA并直接closed-loop。默认强门：显著高于source；direct success retention至少75%；direct gain
retention至少60%；增量跨tasks且Goal/Long不能系统性为零。未过门时video encoder不启动。

#### Fold0首版裁决与唯一后继参数化（2026-08-22）

首版`scalar-bounded q_pi + stable-prior factor-template residual compiler`已经完成全部1,140 task visits和三个预注册
checkpoint。held5在228/570/1140分别只有`23/27/27`，而source/shared/direct-earliest/direct-latest为
`21/43/74/108`；570到1140虽有`10 gained / 10 lost`，绝对分数完全不动，Goal与Long始终为0。24个生成LoRA的
跨task effective-update cosine均值为`.996807`，direct step2000只有`.131914`；自身direct direction只在`1/24` tasks上
比最近其它task更近。因此继续同一训练曲线、调小超参或直接进入`q_V`均不再授权。

定位把失败分成连续两个接口。visible anchor与`q_pi` teacher process的跨task cosine均约`.946`，privileged correction虽把
direct pair-geometry correlation从`.4271`提高到`.4993`，但唯一全局`residual_scale`训练后仍只有`.1006`；compiler随后又把
剩余差异压到`.996807`。所以后继仍属于同一个Stage 1和同一Program schema，但替换这两个错误参数化：

1. `q_pi`使用event/owner/content-dependent evidence gate，让successful-policy correction可按visible event、layer和family
   获得实质幅度；删除单一约`.1`的全局残差上限。correction仍只能写入visible presence已激活的位置，不能成为task-ID或
   与视频无关的free latent。
2. compiler保留一个共享target-/rank-query主干和family heads，但区分同一函数的两个**绝对输出面**：process缺失时输出一套
   完整stable-prior LoRA；process存在时heads直接输出一套完整task LoRA。full输出不再令A、B分别等于
   `stable_prior_factor + residual_factor`，避免每个task先以几乎相同的修正取消大幅stable prior并产生交叉项。它仍只部署
   一套rank16 adapter、全部16 ranks可写，不是shared carrier加第二adapter。
3. successful adapters先用rank16小矩阵分解得到确定的gauge-canonical factors，作为absolute heads的坐标warm-start；
   canonical factor loss只解决双线性优化与moving gauge，不负责方法选择。exact effective-BA、multi-state policy response、
   source/shared support、train-task reward/progress和held closed loop继续决定是否有效。
4. 昂贵closed loop前先做Program到LoRA的task-discrimination门：fit与held都必须摆脱近全局输出、mean own-direct similarity
   必须高于nearest-other且大多数task自身检索正确，同时effective norm/cosine要出现实质恢复。该门只避免重复评测明显坍缩
   checkpoint，不能替代Gate 2。fold0首轮固定数值门为candidate跨taskpair cosine均值不高于`.95`、mean own-direct
   cosine至少`.30`且高于mean nearest-other、own retrieval至少`13/24`、candidate/direct effective norm ratio均值至少`.30`。
5. 坐标门通过后，在fit tasks补齐successful-member、source/shared support functional panels和task-equal reward/progress；
   若19 mappings仍限制泛化，再使用已授权且排除validation/Test的LIBERO-90 meta-task expert family扩大独立映射数。不得用
   更多同task episodes冒充更多meta tasks。

这一选择相对“只把当前训练延长更多steps”的决定性优势是直接移除了已观测到的两个幅度/坐标瓶颈；风险是process presence
可能退化成full/prior开关，因此后续`q_V`仍必须用full相对language+scene、wrong与最终shuffled/reversed的正向闭环增量证明
内容和顺序，而不能凭结构开关自证视频必要性。首版实现由Git和formal artifacts保留，active tree只保留修正后的Stage 1路径。

v2的可执行坐标合同是fresh-incompatible的：successful members与stable prior在进入Stage 1前逐target做compact rank16 SVD并
固定rank sign；prior-only forward在数值上精确返回该完整prior，full forward通过straight-through process-presence开关只读取
absolute family heads。开关只决定反事实输出面，不是task内容路由；task差异仍必须来自Program tokens和`q_pi`的
event/owner/content gate。materializer在同一次24-task forward内计算上述几何门，避免为每个checkpoint再做一轮重复observer
推理；v1 config/schema/evaluator不在active tree保留兼容分支。

v2在228 visits的几何门证明上述坐标修正仍不充分：candidate跨task cosine为`.994192`，own-direct低于nearest-other且只有
`2/24`自身检索正确，故没有进入held closed loop。该段没有successful-policy functional gradient；同时实现中address
embeddings既进入attention values，constant target/rank query又以`hidden + query`直达factor heads，允许numeric address绕过
task Program内容。这与“target query读取Program”的设计因果方向不一致。

v3据此进一步固定**content/address separation**，不改变Program轴或deployment合同：

1. visible Program的`event/layer/family`由tensor位置承担地址，不再向language/scene/process values重复加入可独立写参数的
   owner/layer/family常量；Stage 0已经产生owner-specific process content；
2. `q_pi`的canonical factor tokens只编码successful-policy factor内容，不加入rank/owner embedding；rank依旧没有技能语义；
3. compiler key由Program content加owner/type/event address形成，value只含Program content；target/rank/family/layer query只
   决定读取位置和locality，不再加到cross-attended hidden；无Program content时地址路径不能单独写出full LoRA；
4. successful-policy multi-phase functional response从第一个optimizer update启用并成为主目标；exact-BA与canonical factors仅
   以低权重提供坐标warm-start。source/shared success-state support与fit reward/progress仍在几何门后补齐，不能被参数loss替代。

这不是另建decoder：prior/full absolute surface、38-owner/rank-query、family heads、single rank16 LoRA与held信息墙全部保持。
v2由Git和formal artifacts复现，不保留并行运行面。

v3在228 visits的正式裁决又分离出一个更早的训练顺序问题。content/address separation确实把candidate跨task cosine从
`.994192`降到`.939205`、norm ratio从`.099771`提高到`.487844`，说明Program内容已经实质控制输出；但own-direct cosine
只有`.012822`、own retrieval仍为`2/24`。同时前5到后5 updates的functional response由`.995844`降到`.871159`，member
exact-BA却由`1.14167`升到`1.37474`，canonical factor由`1.24101`升到`1.56286`。因此同一shared checkpoint先学会了有限
successful-occupancy panel上的响应捷径，并在task-to-LoRA坐标建立前旋转了compiler。held5没有运行，`q_V`仍未启动。

v4保持v3的全部结构，只改变Stage 1内部顺序：前228 visits为显式**coordinate bootstrap**，只用member/consensus
exact-BA、canonical gauge、prior反事实与locality建立固定compiler坐标，不加载functional panels；几何门通过后从同一
checkpoint exact-resume，才启用successful-policy functional response，并补齐source/shared support与fit reward/progress。
bootstrap不是方法选择或raw重建终点；它只要求target query先能从Program内容读出正确task方向，最终仍必须通过held5
closed-loop Gate 2。

v4从clean pushed `fc0b84e`完成228 visits/38 updates后，member exact-BA与canonical坐标loss都稳定下降，candidate跨task
cosine也降到`.858906`，但own-direct仍只有`.018399`、低于nearest-other `.029450`，自身检索`1/24`且effective norm ratio
`.091336`。这排除了“只需先做coordinate bootstrap”的解释。进一步只读定位显示candidate的gauge-invariant participation
rank为`1.0733`、top1 energy为`.9664`，比direct的`1.2616/.9127`更集中；原始A/B的rank向量平均cosine仍为
`.8501/.7779`。这些rank统计只定位接口，不作为性能目标。

active v5因此保持v3/v4的content/address separation和全部Program轴，只把numeric target/rank query的职责具体化：query仍
决定attention读取位置，同时通过`1+tanh(Wq)`逐维**乘性调制**已经读取到的Program content，再进入共享trunk与family heads。
它不把query相加到hidden，不把address放回values；所以Program content为零时输出仍严格为零，query无法独立写出LoRA，
但同一Program内容可按target/rank形成不同factor readout。这是对专家“numeric target/rank query读取Program”的窄修正，
不是新decoder。v5继续先跑同一228-visits coordinate gate；过门后才恢复完整successful/source/shared support与fit reward/
progress。真实K2单卡profile已确认零functional cache、finite modulation gradient、2.02秒更新与10.13 GB峰值；active tree只
保留v5 schema/config/evaluator，v4由Git、formal artifacts与remote-safe evidence保存。

v5正式228-visits裁决确认这条query-content路径生效但不足：own-direct由`.018399`提高到`.082145`、retrieval由`1/24`
提高到`3/24`，可是仍低于nearest-other `.107706`，effective norm ratio只有`.086427`，所以没有运行held5。这个结果把
最早未覆盖接口从compiler读出推回了privileged evidence与policy objective；不再授权继续做query、rank或loss权重的局部版本。

active Stage 1后继保持v5的Program、absolute compiler、content/address separation和零内容反事实，只替换teacher evidence与
训练信号：

1. successful member的完整多phase response继续作为确定support；多个独立member作为集合输入和一致性监督，不平均成一个
   raw factor真值；
2. 复用fit19已有projected learner-policy trajectories，在相同状态上计算多个successful expert的response。successful learner
   trajectory作为verified support；failed trajectory不把任一expert action直接当真值，而按member response agreement与已知
   outcome降权，随后由simulator progress/reward补足恢复方向；
3. 在同一successful/learner状态面计算source与stable shared-prior response。只在已有成功证据或与member consensus一致的
   局部支持上施加preservation，避免“保留baseline”退化为处处把candidate推回source；
4. 首个optimizer update即联合使用multi-policy functional response、multiple-member set consistency和低权重坐标锚；不再先
   训练一个允许近零update达到约1.0 loss的raw BA-only阶段；
5. functional warm-start确认输出不再近零且能保留多状态support后，在fit task simulator加入task-equal success/progress。
   该outer signal仍只训练shared Stage 1 teacher/compiler，held5、validation8和Test8都不产生梯度；
6. Program仍是event × layer/owner × family。rank仅用于compiler的numeric factor query，在member证据中按无序集合处理，
   不被解释为phase、skill或可跨adapter对齐的语义轴。

当前v6把上述evidence具体化为同一条矩阵流水线。每个successful/learner occupancy先按动作弧长选择8个保序状态；对每个状态、
source/shared/learner/successful-member adapter，在相同flow noise与time下运行一次冻结PI0.5，捕获Action Expert 19个layer boundary、
50个action tokens与flow output。永久冻结的Stage 0 owner projector把它变成`[state,38,50,128]`，再对完整50-token horizon作
4项正交DCT低频展开，得到`[event,38,4,128]`；这一步只压缩joint future coordinate，不给horizon或rank赋予技能语义。随后形成
五个有明确反事实含义的通道：successful expert-source、successful shared-source、learner-state expert-source、occupancy-
generating learner-source、learner-state shared-source。`q_pi`对channel×DCT basis作content attention；channel/basis embedding
只进入key，Value始终来自policy response，因此缺失或零response不能凭地址写出Program correction。

同一缓存还保存完整flow response而不是只保存上述DCT特征。Stage 1每个task visit在successful与learner panel间交替，candidate
只forward一次，同时对多个successful members做response distillation；verified-success learner轨迹全权重，failed learner轨迹
用member agreement乘`.25`基础权重；source/shared preservation只在其response靠近successful-member consensus的局部状态上
连续加权。raw exact-BA/canonical只保留低权重坐标锚。首个update即有policy support，不再设置BA-only bootstrap。

这不是恢复旧global phase decoder：所有policy evidence共同监督同一个event-conditioned Program和同一个single-LoRA compiler，
不存在deployment carrier、第二adapter或task-ID route。已有30条learner occupancy与successful panel会直接复用，不重跑此前的
长时轨迹采集；先跑一个短而有信息量的fresh checkpoint，再依据多状态functional support和预注册几何门决定是否进入held5。

v6从clean pushed `85477ea`完成该短节点。五通道bank覆盖24 tasks、188个successful panels和120个learner panels；228 visits中
moving-panel functional response由前5步`.64456`降到末5步`.50289`，candidate/direct norm ratio由v5 `.08643`恢复到
`.64465`。但own-direct仍只有`.01618`、低于nearest-other `.02816`，自身检索`2/24`，所以预注册几何门失败且held5不运行。
这不是继续延长或调loss权重的理由：下一步在冻结single checkpoint上遍历完整panel bank，判断它是真正的多策略functional
equivalence还是轮换panel捷径；前者进入已规划的fit-task task-equal success/progress，后者修正support teacher最早接口。

冻结全bank audit进一步给出单一接口裁决：candidate在fit19/held5相对source分别为`.80282x/.90167x`且24/24 tasks更好，
但相对stable shared为`1.39966x/1.27745x`且24/24 tasks更差。multi-policy evidence因此不是零作用；错误发生在full Program
触发后，independent absolute heads重写整套LoRA并丢掉prior-only surface已有的shared support。未解决这一层前不加入reward。

v7只替换full output parameterization。heads生成一套rank16 residual factors；对每个target把stable-shared与residual写成
`[B_shared,B_residual] @ [A_shared;A_residual]`的rank32 low-rank union，再对两侧作thin QR，仅在`32x32` core上SVD并保留
top16，得到部署的唯一rank16 A/B。residual为零时结果精确等价于shared；非零时可替换其弱singular modes。该方案不直接相加
raw A/B，所以没有旧template cross terms；也不部署第二adapter、不固定shared12/task4或赋予rank技能语义。其资格仍由同一
冻结support、几何与held closed-loop逐级裁决，结构本身不构成成功证据。

v7的冻结裁决确认union方向有效但尚未过门。fit19/held5 candidate-to-shared从v6的`1.39966/1.27745`改善到
`1.02429/1.09995`，task breadth从`0/19、0/5`提高到`9/19、1/5`；相对source仍在24/24 tasks上更好。但训练目标分解暴露
一个与专家原始Stage 1定义直接冲突的实现问题：direct exact-BA与canonical factor只被称为“低权重坐标锚”，在union的shared
起点上数值却放大到约`9.43/3.04`，使四项参数坐标loss在前5步贡献`1.06492/1.65750`总目标、末5步仍贡献
`.45761/.86428`。因此v7多数梯度实际仍在复现任意direct A/B坐标，而不是共同识别多策略policy-functional等价类。

active v8不修改Program、`q_pi`、factor heads、low-rank union、数据或schedule。member/consensus exact-BA与canonical factor
全部保留为无梯度诊断，优化权重严格归零；梯度只来自successful/learner multi-state policy response、局部source/shared
support与locality，prior-only仍精确返回stable shared。它不是小权重扫描，而是删除与“目标不是raw A/B重建”相冲突的训练
目标。v8仍从fresh训练并复跑同一冻结support门；未过门不得加simulator reward、运行held closed loop或启动`q_V`。

v8首个228-visits节点确认“删除raw坐标梯度”本身不够：fit19/held5 candidate-to-shared为`1.15980/1.14903`，两边均0个task
胜过shared；candidate/direct norm ratio达到`6.54391`，candidate pair cosine为`.97804`，own-direct只有`.01411`。该checkpoint
明确未过门。与此同时，运行记录暴露了独立的schedule合同问题：完整456-visits schedule最终task-equal，但先对所有visit全局
按cost分组再随机排序，使228决策前缀的每task访问数为`5--18`而不是12。冻结audit自身仍是task-equal有效负证据，但不能把这个
不平衡训练前缀作为v8参数化的最终裁决。

严格checkpoint-balanced复验已从clean pushed `0b63da1`完成：每6个visit rounds形成一个114-visits block，228节点中19个
fit task均恰好12 visits。它没有挽救v8。物化的candidate/direct norm ratio进一步达到`8.75029`，candidate跨task cosine
`.99433`，own-direct `.01106`且自身检索`1/24`；冻结审计的fit/held candidate-to-shared为`1.17823/1.11729`，只有
`2/19、2/5` tasks胜过shared。因此v8最终关闭，不延长、不调loss/rank/seed，也不加入reward或`q_V`。

active v9只替换最早失败接口：stable-shared的16个canonical rank-one modes保持固定尺度，factor heads的raw A/B分别经FP32
thin QR形成正交方向，再按对应target shared factors的跨rank RMS能量定标；每个target/rank由从已读取Program content产生的
零初始化selector angle，在shared mode与replacement mode之间作有界rank-one retraction。selector为零时76 tensors精确等价于
完整stable shared；扩大raw head幅度不能改变方向或抢占rank；全部16 ranks仍可写，输出仍是唯一一套完整rank16 LoRA。训练信号仍只来自successful/learner
multi-state policy response、局部source/shared support、exact-prior counterfactual和locality；raw A/B距离继续只作诊断。
该实现以replacement fraction和冻结全bank support裁决是否真正选择了policy-functional modes，而不是参数能量最大的modes。

retained v9实现已通过40项聚焦合同和真实successful/learner双visit profile。首步selector保持exact shared且replacement
fraction为0；一次optimizer update后第二步fraction为`3.9241e-6`，两步梯度finite、峰值约16.44 GB。这个结果只证明有界
retraction图和预期的两阶段梯度到达顺序成立；方法资格仍只由fresh balanced 228节点的24-task geometry与冻结full-bank
support决定。

v9的正式裁决表明bounded retraction解决了“破坏shared”，却没有解决“按task编译”。fit aggregate相对shared达到`.98369x`，
held为`1.00692x`，远好于v8的`1.17823/1.11729`；但breadth仅`10/19、2/5`，candidate跨task cosine`.99779`，
candidate-minus-shared correction本身也有`.97482` pair cosine，24个task的replacement fraction全部挤在
`.08031--.09164`。因此不续训v9，也不因接近shared就运行held闭环。

v10只修正这条已观测因果旁路。numeric target/rank query继续由exact language和scene条件化，但replacement attention
只在present process tokens上归一化，factor与selector的Value只来自process/uncertainty content；language/scene不再作为可直接
写replacement的Value，process presence也不能只充当full/prior开关。process values全零时，即使language/scene与presence存在，
replacement hidden仍严格为零并返回完整shared；真实process则保留event/owner顺序、locality和全部16 ranks。bounded QR/RMS
replacement、零初始化selector、support bank、task-balanced schedule、seed和functional-only objective全部不变。该单变量直接
检验“近全局修正来自静态Value/开关旁路”，资格仍由同一24-task geometry与308-panel frozen support gate决定。

真实successful/learner双visit profile已在gpu01 physical 1通过：两次更新约`2.305/2.082s`，峰值
`16,439,940,608` bytes，裁剪前梯度`.29288/1.90028`且finite；selector fraction从exact shared的`0`
打开至`4.4849e-6`。因而v10图和两阶段梯度顺序通过实底座检查，可进入fresh balanced 228-visits；
该profile不预判科学门结果。

clean pushed `13dfc25`的fresh 228-visits与24-task物化已完成。process-only Value使candidate-minus-shared
correction pair cosine从v9的`.97482`降到`.95088`，selector的task间范围也扩大到`.06844--.12597`；
但candidate pair cosine仍为`.99641`，own-direct `.03983`低于nearest-other `.06200`，own retrieval仍只有`1/24`。
同一frozen full-bank audit中，fit相对shared进一步改善到`.96892x`、breadth `12/19`，held为
`1.00285x`、breadth `2/5`。因而该修正弱化了全局旁路且连续改善support，但尚未建立
task-conditioned policy direction，v10按原门关闭。后继必须回到Stage 1的task-relative Program-to-policy合同，不再续训、
扫小超参或把同一functional-only曲线继续解释成可能过门。

#### Stage 1 OCPB v11/v12 — Outcome-Calibrated Program--Policy Binding

重新逐条核对专家Stage 1原文后，此前“只有冻结support先过门才加入fit simulator reward”的顺序属于仓库自行增加的限制，
不是专家合同。专家把task-equal closed-loop success/progress与multi-state response、successful/learner occupancy、source/shared
preservation、member consistency和Program结构约束并列为`q_pi + compiler`的共同识别目标。v6--v10只实现了其余项；若在
functional-only门失败后永远禁止reward，等于遗漏了原方案中唯一直接判断静态LoRA是否闭环完成task的Stage 1信号。

唯一后继称为 **OCPB v11（Outcome-Calibrated Program--Policy Binding）**。它不再改attention、query、rank、factor容量或
Program轴，也不从scratch重复v10的228 visits；以v10冻结single checkpoint作为prior-preserving functional warm start，fresh
optimizer只引入一个major variable：fit19 task-equal success与BDDL peak progress对Program--policy binding的闭环校准。

credit按同一接口的两个block-coordinate交替分配：

1. `program_binding`：对`q_pi`的`event x owner` evidence logits施加共享于successful members的paired antithetic offset，
   直接回答哪些可见event、layer与target family的privileged correction对当前task闭环有用；
2. `compiler_binding`：对38个`owner = layer x target-family`的selector angle施加跨16个numeric ranks共享的paired offset，
   直接回答同一Program应在哪些policy target families替换shared modes。rank仍无phase、skill或跨adapter语义；
3. 每个fit task使用两个common-random lanes，plus/minus严格共享初始化、environment seed与policy-noise stream；score只由terminal
   success、BDDL peak progress和successful efficiency组成；
4. 每个macro覆盖全部19个fit tasks且等权。outcome surrogate与同一task的完整successful/learner multi-state response、
   source/shared preservation、prior counterfactual和locality共同反传；factor方向仍由functional support锚定，避免四条rollout
   让百万参数decoder任意漂移；
5. held5、validation8与Test8 reward/action均为零读取、零梯度；部署输入输出、single-LoRA、冻结source/observer与最终
   shuffled/reversed只测试的边界完全不变。

这与Stage 4的video-posterior outer credit不是同一件事：OCPB使用privileged `q_pi`并在video inference进入前共同识别可实现的
Program/compiler坐标；Stage 4则在full-video必要性已经成立后冻结compiler，只给deployment event inference、policy binding与
Program posterior分配结构化credit。v11的首个裁决节点只允许短相邻macros；每个节点都物化train24并复跑冻结support，只有
task-relative geometry、fit/held support与held5 direct-success retention共同改善才promote，不能以train reward上升选模。

真实单task profile已经让两种block-coordinate各完成一次plus/minus × 2 common-random lanes并与完整functional anchor共同
backward。program/compiler两步分别为`74.08/38.58s`、峰值`16.43/16.47 GB`、裁剪前梯度`3.51/2.00`且finite；两者都产生
非零paired efficiency advantage。该结果只通过运行合同，不能证明task-equal reward、跨task Program mapping或held泛化。

v11 macro1的program-binding只带来很小但方向一致的冻结改善：fit/held candidate-to-shared由v10的
`.96892/1.00285`变为`.96786/1.00171`，breadth仍为`12/19、2/5`，geometry own retrieval仍为`1/24`。原v11 macro2不能
用于裁决compiler-binding：扰动合同给一个owner的16个numeric rank angles共同增加同一`delta`，surrogate却读取16个angles
的`sum`，使实际coordinate step成为注册`delta`的16倍，而antithetic credit仍按`2 sigma`归一化。它对应的裁剪前梯度升至
`11.8792`，fit/held support退到`.97049/1.00366`；这是实现尺度违反，不是架构反证。

唯一active修正为OCPB v12。shared-rank differentiable coordinate改为16个angles的`mean`，使共同offset精确对应一个注册步长；
从v11 macro1恢复model、optimizer、scheduler、六份rank RNG与world topology，并以`credit_macro_offset=1`复用原macro2的
videos、support panel、perturbation、environment和policy-noise seeds，只重做一次compiler-binding。v11 macro3/4不再运行，
也不重复program macro1。v12仍须经同一train24物化、geometry、308-panel冻结support和held5 oracle逐级裁决。

v12已完成上述matched复验。rank-mean把裁剪前梯度从错误macro2的`11.8792`恢复到`1.63196`，且paired success/progress保持
`10/8`与`.31140/.27193`，所以尺度修正成立；但物化仍只有`1/24` own retrieval、candidate pair cosine `.99586`，冻结
fit/held support为`.96934/1.00303x` shared、breadth `12/19、2/5`，比v11 macro1略差。逐级门因此在held5 rollout前停止，
v12关闭，不能进入`q_V`。

#### Stage 1 OCPB v13 — baseline-relative functional support barrier

v12后的最早接口不是Program缺少task差异：privileged Program correction pair cosine已经是`.82561`，task差异是在compiler和
functional objective后被压回`.99586`的近共同LoRA。当前`source_support/shared_support`把candidate与source/shared响应的
距离直接加入loss；该实现会同时惩罚有益task-specific移动与有害support丢失。专家要求的是已有行为support preservation，
不是所有响应都必须接近shared。

v13只改变这一项定义。对同一panel记candidate、source、shared相对own successful-policy evidence的归一response error为
`R_c, R_s, R_h`，则：

\[
L_{source\_preserve}=w_s [R_c-R_s]_+,\qquad
L_{shared\_preserve}=w_h [R_c-R_h]_+.
\]

candidate优于baseline时barrier为零，不再被拉回；差于baseline时仍保留原panel的outcome/reliability权重。successful/learner
expert-response、stable-prior counterfactual、locality、Program/compiler结构、bounded rank selector、rank-mean outcome
coordinate、v11 macro1初始化和原macro2全部paired seeds均不变。这是support preservation语义的单变量修复，不是loss weight、
rank或seed sweep。先做真实single-task profile；随后只运行一个matched corrected compiler macro并立即物化/复跑同一geometry与
308-panel门。若仍不能同时改善task-relative geometry和fit/held support，不延长该曲线，也不先用LIBERO-90掩盖fit映射未形成。

retained实现将旧response-proximity保留为sealed v10历史路径的默认语义；v13 outcome入口显式选择baseline barrier，run contract
记录该选择。v12 active config/schema/materialization resolver已由v13单路径替换，20项Stage 1聚焦CPU合同覆盖barrier正反例、
非零退化梯度、rank-mean coordinate、信息墙与v13 materialization cursor。

v13现已完成并关闭。它把fit/held candidate-to-shared改善到`.96741/1.00168`，breadth首次达到`13/19、3/5`，8项冻结support
条件通过7项，证明baseline-relative barrier是应保留的目标语义；但candidate pair cosine仍`.99595`、own retrieval仍`1/24`，
没有把Program中`.82546`的task差异编译为task-relative LoRA。联合门失败，因此不运行held5、不轮fold、不扩meta、不进入
`q_V`，也不延长selector-angle OCPB curve。

#### Stage 1 OCPB v14 — owner-resolved policy-response distillation

v14只修正剩余最早接口。现有support bank虽然用`event × owner × channel × basis × width`的full-layer response形成`q_pi`，
compiler objective却只读取最后的`[batch,50,32]` flow；38个factor方向没有收到与自身layer/family对应的功能监督。v14把同一次
policy forward中的owner response也接入compiler gradient，不改变Program、compiler容量、rank selector或信息墙。

对每个同状态、同noise/time seed的successful或learner panel，v2 frozen bank保存source与多个successful members的：

\[
S,E_m\in\mathbb R^{2\times38\times4\times128},
\]

其中2是同一phase panel的两个occupancy states，4是50个horizon positions的固定低频DCT basis。candidate完整LoRA经冻结
PI0.5和同一冻结owner projector得到可微的`C`；以member reliability形成expert effect共识
`T=\sum_m w_m(E_m-S)`，按owner计算member disagreement与signal confidence，再最小化`C-S`到`T`的全局signal归一误差。
因此梯度直接到对应owner的family-specific A/B heads，却从不读取或重建raw A/B target。failed learner occupancy继续沿用既有
低outcome weight与member disagreement降权，不能把不确定expert recovery当确定oracle。

同一candidate forward同时复用最终flow response；v13 baseline-relative source/shared barrier、prior counterfactual、locality、
member set consistency和task-equal paired success/progress均保留。v14从冻结v13 checkpoint初始化，只运行一个bounded
compiler-binding macro并立即物化与复跑geometry/308-panel门。若owner loss有梯度却仍不能降低compiled pair cosine、提高own
retrieval且保持v13 support，就关闭这一机制，不靠权重扫或更多meta tasks掩盖失败。

v14的一个outcome macro现已完成。owner loss在真实图和19个fit task上均有有限非零值，owner active fraction覆盖
`.7895--1.0`；但24-task compiled pair cosine只从v13的`.99595`变为`.99575`，own retrieval仍`1/24`，fit/held
candidate-to-shared为`.96795/1.00313x`且held breadth降到`2/5`。该checkpoint不promote，也不进入held5或`q_V`。

这一结果同时暴露了训练合同而非新架构的遗漏：OCPB macro把19个task的梯度合并为**一个optimizer update**，而此前可裁决的
task-balanced Stage 1短节点是228 visits/38 updates。专家要求的multi-state policy-response distillation已经具备数据与可微图，
却只得到一次参数更新；因此v14是有效的one-update负结果，不能作为该目标已经被合理优化后的反证。

唯一active修正称为 **task-balanced owner-response bootstrap**，不改变Program、compiler、rank或loss定义：

1. 从v13 model weights初始化fresh optimizer，不继承outcome-estimator moment；
2. 暂时不运行simulator两臂，只在world6 cost-balanced schedule上每update访问6个fit tasks；
3. 每visit使用v3 bank中的successful/learner multi-state final-flow、source/shared baseline barrier与owner-local activation effect；
4. 首个信息节点为114 visits/19 updates，即每个fit task恰好6 visits；先物化24-task geometry，不用内部loss选点；
5. 只有candidate pair cosine、own-direct与retrieval出现一致实质移动，才exact-resume至最多228 visits/38 updates并运行
   308-panel冻结support；
6. geometry与support共同成立后，才从该checkpoint运行一个matched compiler-binding outcome macro并进入held5 oracle。

这不是延长v14的昂贵rollout曲线或扫学习率，而是把专家Stage 1中的policy-response identification与closed-loop credit按成本和
因果职责分开。baseline-relative barrier、raw A/B零目标、held/validation/Test信息墙及single-LoRA合同全部不变。

retained实现已将这一路径收敛回唯一`train_ecp_stage1.py`入口和一个v15完整config：从v13只载入model weights，fresh
optimizer按114-visit task-balance block更新；每visit的同一candidate policy forward同时产生final-flow与owner response，
factor heads、selector、`q_pi`和共享compiler接收同一task loss。checkpoint cursor重新使用task visits而非outcome macro，
materializer只接受114/228节点。旧v10/v14 active configs与outcome专用orchestration已从active tree删除；未来若geometry/support
通过，closed-loop outcome作为同一Stage 1 trainer的后续阶段接回，不恢复第二套可执行Writer路径。

v15已经完成上述完整裁决。task-balanced梯度确实让candidate pair cosine从v13`.99595`依次降到114节点`.99229`和228节点
`.98727`，norm ratio也由`1.97`收至`1.71`；因此“owner objective只缺optimizer updates”的优化混淆已排除。但own-direct
始终约`.0398`、nearest-other仍约`.0608`、retrieval保持`1/24`。更关键的是冻结308-panel support相对v13退化：fit/held
candidate-to-shared由`.96741/1.00168x`变为`.97253/1.02171x`，held breadth由`3/5`降到`1/5`。所以geometry与support没有
共同成立，matched outcome macro、held5和`q_V`都不允许启动。

代码与专家原始“更贴近LoRA target的policy-native量”要求共同指出下一个更早接口：当前所谓owner response是某层累计hidden/
residual的冻结投影；即使标记为owner `j`，所有上游LoRA targets都能改变它，loss梯度也能经多head相关变化满足。这解释了
“跨task输出开始分散，但同task successful support没有提高”。后继称为 **OCPB v16 owner-local activation-effect
distillation**，只替换功能监督对象：

\[
\Delta y_{j,s,h}=B_j(A_j x^{ref}_{j,s,h}),
\]

其中`x_ref`是successful/learner occupancy上冻结source policy在相同noise/time下实际产生、随后detach的target input；q/v、
action-in与action-out各自使用对应原生输入，所有expert/candidate在同一个reference input上比较。它保留owner与50-step horizon并
只在缓存阶段做低维basis压缩。这个对象对`A/B` gauge不变、梯度严格落到对应target
effect，却不是raw A/B重建，也不要求复制某个expert的任意参数化。source、多个successful members与candidate都在相同state、
noise/time和输入合同上计算局部effect；member disagreement与outcome继续决定confidence。v16保持v13 barrier、Program、
q_pi、compiler容量、rank16、114/228 task-balanced schedule与联合geometry/support门不变；只有该门通过后才接回一个
matched closed-loop outcome macro。

v16现已完成。它确实建立了比v15更强的target-local task信号：own MSE由v13`.48016`降到`.25057`，cosine retrieval由
`1/24`升到`11/24`，candidate pair cosine降到`.94932`。但shared-subtracted candidate correction仍有`.97657`跨task
pair cosine，而expert correction为`.82316`；candidate幅度只有expert correction的`.49176x`。更重要的是308-panel
fit/held candidate-to-shared退到`1.08581/1.08815x`、breadth`2/19、0/5`。所以`B(Ax_ref)`不是无信息目标，但**孤立
target effect不足以定义38-target组合policy**；v16不续训、不接outcome、held5或`q_V`。

随后裁决的 **OCPB v17 action-grounded composed-policy recovery**复用v16已经形成的task-discriminative model
weights，但因训练合同改变而创建fresh optimizer；每个visit仍由相同language、correct action-hidden videos和train-only
`q_pi`生成一套完整LoRA，再在同task不同episode的policy batch上计算冻结PI0.5原生flow-matching loss：

\[
\mathcal L_{act}
=
\mathbb E_{a,\epsilon,t}
\left\|
v_{\Delta\theta(P)}(x_t,L,I,q)- (\epsilon-a)
\right\|^2.
\]

policy与source参数继续完全冻结；先把generated LoRA作为detached leaves求exact gradient，再按链式法则反传`q_pi`、compiler和
FactorHeads，不保留完整policy参数梯度。action target只来自fit19 successful expert occupancy或verified-success learner
occupancy，并与教学视频跨episode；failed learner trajectory的action不能冒充恢复oracle。held5、validation8和Test8 actions仍
为零gradient/零读取。

v16 target-local effect不再承担主要policy目标，只作为owner/family结构锚；v13 baseline-relative barrier继续保护source/shared，
successful-member final-flow与set uncertainty继续保留。该阶段先做一个真实单task profile，确认exact action loss、LoRA-leaf到
FactorHead梯度、显存和吞吐，再运行一个bounded task-balanced节点。只有action loss、task discrimination和冻结support共同
改善，才允许held5 oracle；如果action loss下降而support继续恶化，就关闭当前Program/compiler mapping，而不是靠继续训练或
小权重扫描解释。

v17现已完成并触发上述停止条件。114 visits/19 updates后candidate pair cosine由v16`.94932`降到`.90236`、effective norm
ratio升到`1.14481`，证明exact action gradient确实让完整LoRA组合离开task-common半程插值；但own-direct仍只有`.03855`、低于
nearest-other`.05779`，retrieval退回`1/24`。更关键的是308-panel fit/held candidate-to-shared为`1.13962/1.12203x`，
breadth`2/19、0/5`。所以action imitation提供了material但未被closed-loop识别的task-dependent方向；同一曲线不续228。

随后裁决的 **OCPB v18 action-guided structured outcome binding**补齐专家Stage 1原始目标中尚未被当时
factor方向真正执行的“task-equal closed-loop success/progress”：对每个fit task先在cross-episode successful panel上求完整LoRA
的exact action leaf gradient，再按38个owner及其family/layer分组归一化为局部proposal direction；paired plus/minus LoRA只沿
这些action-informed directions扰动，使用相同init state、environment RNG和policy-noise RNG，以success、BDDL peak progress与
成功效率计算antithetic credit。reward只决定每个owner方向的符号和幅度，不能凭空发明task identity，也不向deployment加入
action/reward。该坐标直接触及A/B factor方向，不能恢复v11--v13只扰动evidence gate或selector angle的旧outcome实现。held5、
validation8和Test8仍为零gradient/零读取；先做单task paired profile，再给fit19至少多个task-equal optimizer updates，不能再次
用一个update裁决昂贵outcome目标。

v18具体坐标现已收敛为单一路径。对owner \(j\)的完整A/B pair，先由successful cross-episode panel得到
\(g_j=\nabla_{A_j,B_j}\mathcal L_{action}\)，再令
\(d_j=-g_j\lVert(A_j,B_j)\rVert_2/\lVert g_j\rVert_2\)。paired两臂使用
\((A_j,B_j)\pm0.05\epsilon_jd_j\)，所以不同family、layer和tensor大小都具有相同的owner-relative factor步长；每个owner共用
一个Rademacher sign，但38个owner彼此独立。由两条严格配对lanes得到的reward差估计每个relative coefficient的梯度，再除以
\(\lVert d_j\rVert_2^2\)投影回A/B leaf，使其与\(d_j\)的内积精确等于负的reward-coordinate gradient。该leaf随完整candidate
LoRA图反传，不向compiler增加第二套head或adapter。exact action gradient只定义proposal，不再像v17一样作为独立下降目标；
successful/learner policy response、local activation、source/shared barrier、prior counterfactual和Program locality继续提供结构锚。
formal每个macro覆盖fit19各一次并只做一个等权optimizer update，首个bounded节点至少含两个macros。

首个真实single-task paired profile已确认上述图完整接通：38/38 owner方向均active，两个严格配对init分别给出相反的
`plus-only`与`minus-only` success，因成功效率差异产生非零antithetic coordinate gradient，且LoRA leaf能够反传到
FactorHeads。profile也暴露了一个formal前必须修正的尺度问题：旧selector-coordinate outcome的`.01` surrogate multiplier
不适用于这里已经除以`2\sigma`的标准有限差分reward estimator；它把outcome leaf norm压到`.002038`，而定义proposal的
action leaf norm为`.12304`，使新增closed-loop目标几乎只是名义接线。v18 canonical因此不继承这个历史超参，也不做标量扫；
使用`1.0`保留estimator本身的自然尺度，再以现有global gradient clipping处理optimizer步长。该变更必须经过一次matched
single-task profile确认finite gradient和显存后，才授权fit19 formal。matched profile现已通过：保持task、videos、paired init与
全部RNG不变时，outcome leaf norm由`.002038`严格变为`.20380`，surrogate由`-.000166`变为`-.01662`；FactorHead与总裁剪前
gradient仅为`1.36716/4.78533`且finite，峰值显存仍为`16,443,782,144` bytes。因此`1.0`成为v18唯一canonical尺度，fit19
bounded节点已获运行授权。

v18随后完成4个fit19 task-equal macros。每轮有`8--10/19` tasks产生非零paired credit，证明闭环目标不是稀疏dead path；
但macro2/macro4的own-direct cosine只有`.03862/.03836`，retrieval均为`1/24`。308-panel fit/held relative-shared在macro2为
`1.13120/1.12216x`，到macro4又退到`1.14317/1.13244x`，breadth始终`2/19、0/5`。Program correction pair cosine几乎不动
（`.84710→.84690`），而compiled adapter继续分散（`.89302→.89140`）；说明直接在完整A/B factor空间构造plus/minus，随后把
scalar reward leaf投回同时可动的Program与compiler，主要仍在重新定义编译坐标，没有识别own successful Program。v18因此在
macro4关闭，不运行held5或`q_V`。

下一Stage 1变量称为 **OCPB v19 fixed-compiler structured Program binding**。它回到v13这个已证明最能保留shared support的完整
Program/compiler坐标并冻结compiler；exact action gradient只在该compiler真实可达的Program切空间定义proposal，按
event × layer-group × target-family分块。首个bounded实现每次只扰动一个target-family block、保留全部active ordered events和
该family的native layers，使一个paired reward差只回答一个明确问题，不再用一次scalar advantage同时估计38个A/B owner坐标。
reward surrogate只能更新privileged Program inference，FactorHeads、rank selector与其余compiler永久冻结。先用单task profile
证明Program perturbation能产生足够的compiled LoRA差异、paired outcome差异和finite upstream gradient；若固定compiler局部
不可达，就在训练前关闭，而不是靠放大sigma或重新解冻decoder。

v19 retained实现已经替换唯一Stage 1运行面。模型从v13 macro1只加载weights并创建fresh optimizer；`visible_program`与
`compiler`在整个阶段永久冻结，optimizer只拥有`policy_teacher`参数。第(m)个macro按预注册
`q → v → action_in → action_out`选择一个family：先把successful cross-episode action loss对冻结compiler输入
`P_process`的exact gradient反传出来，只保留全部active events与该family原生owners的block，再把负梯度归一到当前block的
L2 norm。两臂使用`P_process ± .05 d`并分别经过同一冻结compiler；paired success/progress estimator只形成对
`q_pi(P).process`的semi-gradient。结构/support objective仍约束同一个`q_pi`输出，但无法旋转compiler坐标。active config为
`configs/pi05_ecp_stage1_fixed_compiler_program_binding_v19.json`；旧v18 config和factor-space perturbation已从active tree删除。
当前CPU合同已通过，下一门只是真实单task profile中的compiled relative delta、paired credit、上游梯度与冻结所有权。

该profile现已通过。q-family的18个native owners与全部active events形成`18,432`个有效Program elements；`.05`相对切向
扰动经同一个冻结compiler产生`.110166` compiled-LoRA relative delta。两臂在两个paired init上各成功一次但成功互换，
并由成功步数差形成非零credit；action Program gradient为`.0005131`，`policy_teacher`裁剪前梯度为`.560583`，compiler与
visible Program梯度均严格为0。它证明固定compiler局部可达且reward semi-gradient接通，授权fit19 macro2 bounded节点；
它仍只是运行图证据，不替代24-task geometry/support裁决。

v19 macro2的完整门随后否定了“只固定v13 compiler并用少数shared reward steps更新`q_pi`已经足够”。q/v两个macros分别在
`8/9`个fit tasks上得到非零credit，Program perturbation经compiler的相对LoRA差异为`.10901/.05426`；但最终24-task
candidate pair cosine仍为`.99594891`，own retrieval仍`1/24`；308-panel fit/held support为`.96830/1.00335x`，绝对门
失败。历史v13 reference的`.99595249/1-of-24`与`.96741/1.00168x`使用另一组24/24 demo pairs，只能证明v19仍处于相同
坍缩区间，不能把小差值解释为matched退化。所以失败不是Program切向不可达，而是已有bounded compiler没有
把shared `q_pi`更新识别成own successful policy；v19不续action-in/out macros，也不进入held5或`q_V`。

下一互补因果锁为 **OCPB v20 Program-Locked Compiler Identification（PLCI）**：回到v13 weights，固定visible Program与
privileged `q_pi`的全部参数，只训练compiler。这样task-diverse Program坐标保持不变，compiler必须用successful
cross-episode exact action、owner-local/multi-state response和baseline-relative support barrier学习一致映射，不能再与
Program共同旋转。首个bounded节点为114 visits/19 task-equal updates；只有24-task own matching与308-panel support同向改善，
才加入fixed-Program下的structured outcome calibration。若compiler-only仍失败，下一诊断是对当前fixed compiler直接优化
task-local free Programs，以区分compiler image不可达和shared `q_pi` inference不足；这些free Programs只作fit19 privileged
reachability oracle，不成为部署task-ID route或唯一`P*`监督。

v20当时按该合同实现为唯一Stage 1运行面。它从v13 macro1只加载model weights并新建optimizer，
`model.requires_grad_(False)`后只解冻compiler；每个fit task在一个114-visit block中精确出现6次，6-GPU下汇总为
19个等权updates。exact action leaf仅通过完整38-target LoRA回传compiler，policy-response/support作为baseline-relative
structural anchor；每步显式记录compiler/FactorHead梯度并要求`q_pi`/visible Program梯度精确为0。物化固定使用
video visit12099，与v13 historical authority形成真正matched geometry/support对照。v19的active config与三个outcome
runtime模块已删除；其formal artifact与Git仍保留历史证据。

clean pushed `a1689ee`的真实单task profile确认这一运行图已接通：exact action LoRA-leaf、FactorHead和
compiler裁剪前梯度分别为`.129426/.431188/7.62951`，`q_pi`/visible Program梯度精确为0；单次完整
update 2.80秒，峰值显存16,385,350,144 bytes。它是formal 114-visit节点的运行门，不替代geometry/support
或closed-loop裁决。

formal 114-visit节点随后否定了该方法。compiler相对v13参数移动`.146889%`，Program/`q_pi`参数精确
不变；在24/24完全相同的video pairs上，candidate pair cosine由`.995952`降至`.971204`，却仍没有产生
own mapping：own/nearest-direct为`.03945/.06104`，retrieval `1/24`。matched 308-panel fit/held support由v13
`.96741/1.00168x`、breadth `13/19、3/5`退到`1.00874/1.02932x`、`8/19、2/5`。因此不能把“更分散”
当成compiler identification，v20关闭且不接structured outcome。

下一有界诊断称为 **OCPB v21 Fixed-Compiler Free-Program Reachability（FPR）**。它把v20 compiler、source、observer、
visible Program和shared `q_pi`全部冻结，只在fit19为每task维护一个可优化的privileged Program tensor；优化仍使用
cross-episode successful/verified-success exact action leaf、multi-state response和baseline-relative support，failed learner action梯度为0。
这是reachability oracle，所以task-local Programs不是deployment model、不读held5，不成为task-ID route或唯一`P*`。若它恢复
own geometry/support，证明compiler image可达而shared `q_pi`坐标错；若它仍失败，则当前bounded rank-one compiler
image本身不能表示所需policy，后续应替换compiler parameterization，不扫LR/rank/seed。

v21的具体自由度保持最小且与shared `q_pi`可写面一致。每个fit task先用固定visit12099的两条correct action-hidden videos经
冻结observer、visible Program和v20 `q_pi`产生一次anchor；随后固定`language/scene/presence`，只训练
`process=base+2*tanh(delta)`与`uncertainty=base*exp(2*tanh(zeta))`。19行参数彼此独立，6-GPU每步all-reduce后只让该步
全局实际访问的task行进入AdamW，避免inactive行积累optimizer动量；held5不创建参数。formal节点为每fit task 12 visits、
总228 visits/38个等权updates，结束后只做一次matched geometry与308-panel support audit。

真实单task profile确认实现合同成立：action-LoRA leaf与free-Program梯度norm为`.129141/.052106`，一步后process相对
修正`.092908`，uncertainty scale范围`.90491--1.10508`；compiler、shared `q_pi`和visible Program梯度均为0。训练update
为`1.26s`，峰值显存`16,396,331,008` bytes。该profile只证明fixed image对task-local Program有可微、非死写入路径，
不能预判228-visit后的own-policy reachability。

v21 formal已经给出否定裁决。fit19 free Programs在228 visits后形成明显task-specific corrections：process相对anchor
`.176--.531`，correction pair cosine`.36624`；这排除了shared `q_pi`输出完全同质和Program没有移动。但固定bounded
compiler仍把它们压到candidate pair cosine`.96595`，own/nearest-direct`.03936/.06070`、retrieval`1/24`。308-panel
fit/held support为shared的`1.02708/1.02932x`、breadth`7/19、2/5`，没有进入held rollout或`q_V`的依据。

这次失败也暴露了实现与专家原始compiler定义之间尚未裁决的主要差异。当前bounded surface先把A/B heads输出QR归一化，
再用固定于shared template平均factor能量、最大`pi/2`角度的rank-mode replacement生成LoRA；raw head幅度和独立
task-dependent rank gain不会进入最终factor。专家方案则是每个target/rank query读取layer/family-local Program后，使用
family-specific A/B heads直接写出完整absolute rank16 LoRA。后者不是“给bounded selector再加一点自由度”，而是不同的
output image。

因此v21之后的唯一Stage 1因果锁称为 **OCPB v22 Direct-Absolute Compiler Free-Program Reachability（DA-FPR）**。
仓库已有v6 direct-absolute compiler formal macro228 checkpoint，先复用它及同checkpoint的privileged `q_pi`，保持v21
fit19 free-Program、cross-episode action、multi-state response、task-equal schedule和geometry/support裁决不变。这样不用再做
一次长compiler训练，就能直接回答专家所述absolute surface是否含有own-policy preimage。v22通过后才在该固定坐标上训练
shared `q_pi`和Dynamic-K `q_V`；若失败，则继续拆解Program可写区域、attention/value binding与family A/B gain，而不恢复
bounded selector、task-ID route或小超参扫描。正式v21证据见
`docs/evidence/ecp_20260822/stage1_fixed_compiler_free_program_v21_gate.json`。

### Stage 2 — Frozen compiler下训练Dynamic-K `q_V`

固定source、observer authority、`q_pi`和compiler。每个training sample使用K=1--4条action-hidden视频，并用同task不同episode
action query。训练：

- `q_V`与`q_pi`可识别Program分布对齐；
- compiler后flow/action与cross-episode behavior loss；
- same-task different-video与nested-K consistency；
- uncertainty随K收缩但不抹掉合法procedure差异；
- 只用正确视频学习跨episode与跨K的Program；训练期不使用wrong、static、shuffled、reversed或其它破坏视频内容的
  conditions。language/scene/first+final只作为信息路径baseline；shuffled/reversed留到最终候选checkpoint评测；
- same-language cross-scene、same-endpoint different legal procedure等数据存在时的process-identifying contrasts。

输出：`video_writer_stage2.ckpt`。先在fold-held tasks裁决，再按6.2用全部train-authorized数据重训并进入validation8。

### Stage 3 — 除backbone外的Writer联合收敛

fixed compiler是识别坐标的阶段性手段，不是最终永久冻结要求。Stage 1和Stage 2分别过门后，增加端到端联合阶段：

- 冻结PaliGemma、native Action Expert、source policy和train-only `q_pi` teacher；
- 若Stage 0B采用Action Meta-LoRA，它已作为observer/backbone calibration永久冻结；
- 解冻全部普通部署Writer参数：semantic projections、TransitionMatcher、post-capture projections、event binding、segmenter、
  Dynamic-K aggregator、`q_V`、compiler与全部LoRA heads；
- `q_V`/frontend用主学习率，compiler使用较慢但非零学习率，所有普通Writer参数都接收真实联合梯度；
- 冻结`q_pi`分布锚定Program坐标，Stage 1 compiler快照的multi-state policy response与support preservation防止再次共同旋转；
- 持续使用correct-video cross-episode actions与task-equal functional panels；shuffled/reversed不参与联合训练、阶段选择或
  checkpoint选择，只在最终候选checkpoint冻结后评测。

这一步不是从scratch直接联合训练；它从已经分别可实现、可推断的checkpoint消除模块失配。必须与Stage 2 frozen-compiler
checkpoint严格配对；若联合训练降低absolute、breadth、same-video retention或correct-vs-shuffle/reverse specificity，则不
promote joint checkpoint。输出：`video_writer_joint_stage3.ckpt`。

### Stage 4 — Structured closed-loop outer credit

只有Stage 3通过video necessity gate后，才在授权train/meta simulator接outer credit。credit按event、owner family、layer与
phase分配，不再对全局16维code做单方向Gaussian估计。可更新Stage 3的普通Writer参数，source、backbones、frozen observer
calibration与validation/Test边界不变。输出可选`video_writer_outer_stage4.ckpt`；只在held/validation paired absolute、
breadth、retention和process specificity同时提升时保留。

## 8. 视频特异性与正式资格

correct视频必须提供正向能力，而不是仅把negative做坏。正式paired条件至少包括：

- learned language-only；
- language+scene/first+final；
- video-only；
- no-video/static-first-repeated；
- cross-suite wrong；
- shuffled；
- reversed；
- endpoints-middle-shuffled；
- same-task-other与Dynamic-K。

这些conditions只在最终候选checkpoint已经由correct表现冻结后运行，不产生梯度，也不参与checkpoint选择。最终必须追求真实
的order specificity：在有顺序依赖的tasks上，correct稳定优于shuffled与reversed，并出现correct-only的新success rows；
差异应跨tasks/suites，而非只集中Object或由某个identity开关产生。简单或本来order-insensitive的task不要求人为崩坏
negative，但process-identifying subset必须显示明显正确顺序优势。

方法资格要求strict paired correct严格`>145/400`，并且必须同时满足：高breadth、Goal/Long真实贡献、相邻checkpoint低churn、
same-task-other success retention至少90%并争取95%、full优于language+scene、correct优于shuffled/reversed/wrong/no-video，
且不依赖task-ID、teacher action、第二adapter或checkpoint union。

## 9. 根本失败与转向条件

以下条件在完成合理结构修正、多个19/5 folds与必要数据补充后仍成立，才构成根本性失败：

1. `q_pi + compiler`无法在video encoder进入前保留successful policy support；
2. oracle compiler通过，但`q_V`在source-unseen、process-identifying tasks上始终无法预测可用Program；
3. full video在合法same-endpoint/different-procedure数据上始终不优于language+scene或endpoints；
4. 任何成功都依赖task-ID、held actions/runtime video、第二adapter、task-local interaction或大规模checkpoint挑选；
5. native与Action Meta-LoRA observer都不稳定，event结构主要由noise、frame index或endpoint shortcut决定；
6. 完成structured outer与joint training后仍`<=120/400`、top3贡献`>80%`、Long为0、same-video高churn且无full-video增量。

失败时按最早接口转向video-to-progress/reward、skill composition、runtime conditioning或video-initialized task-local RL；不围绕
rank、LR、seed、dtype或小容量继续扫。

## 10. 实现所有权、复用与生命周期

复用：frozen source、38-target LoRA合同、task expert banks、47条successful trajectories、privileged action store、
Dynamic-K sampler、video cache、paired evaluator、persistent workers、flow/action/JVP工具与formal evidence schema。

新运行面需要清晰owner，至少拆分为：

- native layer capture与compact owner lattice；
- visual transition/event segmentation；
- event-conditioned horizon binding与Program aggregation；
- privileged `q_pi`；
- target-family compiler；
- staged/joint training与checkpoint owner。

Stage 1实现据此固定为一条调用链，而不是14条平行架构：`stage0.py`只扩展冻结observer的language/scene可见输出；
`program.py`独占visible Program与K-video集合聚合；`policy_teacher.py`独占train-only privileged `q_pi`；`compiler.py`独占
38-owner/rank-query到完整LoRA；`stage1.py`只组合上述科学图；`stage1_data.py`拥有task/member/video authority，
`policy_response.py`拥有冻结PI0.5 full-layer capture与target-local activation effect，`stage1_support.py`与`stage1_support_building.py`拥有policy-support
bank及其运行时panel；`stage1_objective.py`拥有结构/support loss，`stage1_train_step.py`拥有一次task-equal compiler
update及exact LoRA-leaf回传，`stage1_training.py`拥有唯一formal/profile orchestration、authority加载与checkpoint生命周期；
`stage1_materialization.py`只把冻结checkpoint变成held oracle所需的每task单LoRA，
现有`expert_manifold` evaluator继续拥有闭环执行。已完成的OCPB outcome实现由Git和formal artifacts保存，不在active tree保留
第二套orchestration；通用paired trajectory score仍由`reward/credit.py`拥有。当前唯一Stage 1训练入口为
`train_ecp_stage1.py`；v20必须复用`writer.functional`已有的exact frozen-policy LoRA leaf-gradient owner，再经固定Program
链到compiler，不新增第二个policy-loss实现或平行Writer。只有Program/compiler identification与geometry/support联合门通过后，才在同一
canonical Stage 1生命周期进入held oracle，不恢复平行Writer或退役入口。

生命周期也固定：Gate 2未过时只修正上述最早接口；Gate 2通过后`q_pi`保留为训练锚但永不进入deployment，Stage 1
materializer只保留为oracle/evidence工具，正式部署由后续`q_V`经同一个Program/compiler生成LoRA。旧16维decoder、
shared12/task4 residual、LMMPC Writer和single-direction outer不会从这些模块导入或恢复；历史实现只由Git、sealed configs和
formal artifacts复现。这样新增代码是active ECP职责的替换面，不是对旧运行面继续堆叠兼容分支。

不得继续把这些结构塞进已经很大的`functional_adaptation/inference.py`或旧LMMPC classes。代码完成oracle compiler与端到端
smoke后，只保留一个canonical Writer registry/entrypoint；旧16维与LMMPC运行面由Git、sealed configs和formal artifacts复现，
不保留永久并行fallback。

formal GPU训练必须从clean pushed commit的detached frozen worktree启动。现阶段先实现Stage 0和Stage 1运行面；当前16维
process warm-start与旧single-direction outer不再启动。
