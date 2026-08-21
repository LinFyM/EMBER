# EMBER-ECP: Event-Conditioned Policy Compiler

状态：2026-08-21 **active design / implementation authority**。本文吸收第二轮专家最终复核与owner随后裁决，取代
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
   checkpoint，不能替代Gate 2。
5. 坐标门通过后，在fit tasks补齐successful-member、source/shared support functional panels和task-equal reward/progress；
   若19 mappings仍限制泛化，再使用已授权且排除validation/Test的LIBERO-90 meta-task expert family扩大独立映射数。不得用
   更多同task episodes冒充更多meta tasks。

这一选择相对“只把当前训练延长更多steps”的决定性优势是直接移除了已观测到的两个幅度/坐标瓶颈；风险是process presence
可能退化成full/prior开关，因此后续`q_V`仍必须用full相对language+scene、wrong与最终shuffled/reversed的正向闭环增量证明
内容和顺序，而不能凭结构开关自证视频必要性。首版实现由Git和formal artifacts保留，active tree只保留修正后的Stage 1路径。

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
38-owner/rank-query到完整LoRA；`stage1.py`只组合上述科学图；`stage1_data.py`与`stage1_panels.py`分别拥有task/member/video
authority和multi-phase functional evidence；`stage1_objective.py`、`stage1_train_step.py`、`stage1_training.py`分别拥有loss、
一次task-equal更新和formal编排/checkpoint；`stage1_materialization.py`只把冻结checkpoint变成held oracle所需的每task单LoRA，
现有`expert_manifold` evaluator继续拥有闭环执行。拆分的理由是信息墙、共享梯度、policy functional调用和静态物化具有不同
生命周期与测试责任；它们全部只服务同一个`train_ecp_stage1.py`训练入口和一个materialization入口，没有版本化fallback。

生命周期也固定：Gate 2未过时只修正上述最早接口；Gate 2通过后`q_pi`保留为训练锚但永不进入deployment，Stage 1
materializer只保留为oracle/evidence工具，正式部署由后续`q_V`经同一个Program/compiler生成LoRA。旧16维decoder、
shared12/task4 residual、LMMPC Writer和single-direction outer不会从这些模块导入或恢复；历史实现只由Git、sealed configs和
formal artifacts复现。这样新增代码是active ECP职责的替换面，不是对旧运行面继续堆叠兼容分支。

不得继续把这些结构塞进已经很大的`functional_adaptation/inference.py`或旧LMMPC classes。代码完成oracle compiler与端到端
smoke后，只保留一个canonical Writer registry/entrypoint；旧16维与LMMPC运行面由Git、sealed configs和formal artifacts复现，
不保留永久并行fallback。

formal GPU训练必须从clean pushed commit的detached frozen worktree启动。现阶段先实现Stage 0和Stage 1运行面；当前16维
process warm-start与旧single-direction outer不再启动。
