# 1. 对三项澄清的直接理解

## 1.1 四种“时间”必须彻底分开

我接受并强化这一区分：

| 轴                                       | 它表示什么                                                               | 它不表示什么                                  | 新架构中的处理                                                                  |
| --------------------------------------- | ------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| teacher-video frame time \(t\)          | 教学视频中观察状态的先后、局部变化、阶段推进                                              | 机器人执行时的绝对控制时钟                           | 每条视频独立保序；使用相对帧位置和单调event，不跨视频建立绝对时间                                      |
| Action Expert horizon \(h=0,\ldots,49\) | 在一个静态观测和固定probe下，π0.5内部50个未来action positions上的policy response field | teacher视频的后续50帧，也不是已经预测好的teacher action | 在每帧内完整保留到第一次task/event-conditioned读取；严禁用 \(t+h\) 跨帧对齐                    |
| flow/denoising time \(s\)               | flow matching从噪声到动作的内部积分坐标                                          | video frame time或action horizon         | 首版固定在已有正证据最强的 \(s=1\)；不引入多flow-step序列                                    |
| Action Expert layer depth \(\ell\)      | 网络不同深度逐步形成的policy representation                                    | 物理时间或任务阶段                               | 通过19个layer boundaries、residual increments和38个target owners保留；作为有序拓扑而非时间轴 |

此外还有第五个必须保留的轴：固定antithetic probe \(p\in\{+,-\}\)。它既不是flow time，也不是两个视频样本。当前Stage0确实在每个无state的image-language prefix上运行固定`[50,32]` suffix probe，捕获19个Action Expert边界、50个horizon positions、flow velocity，再映射为38-owner lattice；当前代码的真实结构见：

* `c61518d:src/ember/ecp/stage0.py:L42-L276`
* `c61518d:src/ember/ecp/observer.py:L12-L259`。

因此，**Action Expert内部future-action response应当是新Writer理解视频过程的核心证据，而不是附加在普通视频Transformer旁边的一组feature**。

## 1.2 模块化没有问题；连续建立新坐标才是问题

我不再坚持“一个统一大网络从输入直通LoRA”。更合理的边界是：

1. 冻结π0.5的逐帧policy-response捕获；
2. 一个可扩展的policy-response视频过程编码器；
3. 一个可扩展的current-video native-factor composer；
4. 一次确定的LoRA规范化和materialization。

前两个learned模块内部都由少数可重复block组成。两者之间只传递有明确轴含义的event-owner tokens和soft temporal assignment，不再依次经过固定Program summary、canonical code、covariance solve、transport、anchor、family scalar gate等专用坐标。

## 1.3 我撤回上一版的negative-control训练主张

上一份归档回复明确提出了`paired causal specificity loss`，把wrong、shuffled、reversed和no-video放入训练目标；还把task experts和历史LoRA描述成可构造静态policy-native字典。这两点与owner当前澄清不一致，必须撤回。原表述见：

* `c61518d:docs/expert_review_20260902_full_history_policy_native_meta_writer.md:L465-L545`。

修订后的规则是：

* 训练梯度只来自正确视频、授权train/meta任务、正的cross-episode functional evidence、必要的source preservation及轻量边界约束；
* wrong、no-video、language-only只在模型冻结后评价；
* shuffled、reversed不参与训练、loss、checkpoint选择、内部Gate或局部架构修补；
* 若冻结模型最终被shuffle/reverse改善，该candidate的时序因果主张失败，但不能把该control重新做成训练标签。

这与当前owner长期合同一致。

---

# 2. 重新复核后，我如何修正上一版建议

固定提交`c61518d`本身只归档了上一份审查，没有增加新实现或新实验。

我的架构判断发生了七项实质修订。

| 上一版建议                                       | 本次修订                                                                  |
| ------------------------------------------- | --------------------------------------------------------------------- |
| 把`native taps`笼统视为视频token来源                 | 明确拆成**紧凑policy-response lattice**与**真实native X/Y factor bank**两条不同证据流 |
| 倾向一个统一大Meta-Writer                          | 改为两个主要learned模块：视频过程编码器、native factor composer                        |
| 38×16 target-rank queries                   | 首版改为**38×4 mobile-rank queries**；rank12 carrier固定，最终仍是一套rank16        |
| task experts/历史LoRA形成静态字典                   | 删除。task-expert factors不进入deployment，也不形成可检索字典                         |
| native bank加learned free residual作为主路径      | 首版完全关闭free residual；只有严格support反证后才开放                                 |
| task-expert functional distillation作为主要训练解释 | 承认其目标对同task视频近似task-constant；改为主功能loss之一，但不负责识别视频                     |
| paired causal specificity loss              | 完全删除；所有负control只在冻结模型后评价                                              |

修订后的主案可简称为：

> **Policy-Response Event-to-Factor Writer**
> 冻结π0.5逐帧产生owner/horizon policy-response field和真实X/Y bank；learned视频模块沿teacher-frame time形成ordered policy-response events；learned factor模块用这些events在当前视频的真实bank中执行signed selection；最后产生rank4 residual，与固定rank12 carrier一次拼成完整rank16。

名称不构成设计约束。核心变化是：**保留G1和G2的正证据，取消冻结Natural Program到专用compiler的强制接口，同时不退回任意A/B hypernetwork。**

---

# 3. PI0.5内部时序证据的精确定义

## 3.1 每个teacher frame实际读取什么

对于第 \(k\) 条视频的第 \(t\) 帧，使用：

$$
\text{image}_{k,t}+\text{exact language}
$$

构造不含state/proprio的原生π0.5 prefix。当前实现已经严格如此；image和language被拼成prefix，固定probe在flow time 1进入Action Expert。`c61518d:src/ember/ecp/stage0.py:L107-L211`。

冻结捕获层应输出以下内容。

### A. VLM上下文

$$
P_{k,t}\in\mathbb{R}^{256\times 2048}
$$

为image patch hidden；同时保留：

* exact language的冻结token embedding；
* 当前image-language contextualized language hidden。

语言负责“看什么、目标是什么”，patch负责当前场景内容。它们不直接生成mobile LoRA，只用于grounding和query调制。

### B. Action Expert layer states

对两个antithetic probes：

$$
H_{k,t,p,\ell,h}\in\mathbb{R}^{1024},
\quad
p\in\{+,-\},\quad
\ell=0,\ldots,18,\quad
h=0,\ldots,49.
$$

其中19个位置是18层的input layernorm前状态加final norm前状态。

### C. Layer residual increments

$$
\Delta H_{k,t,p,\ell,h}
=
H_{k,t,p,\ell+1,h}-H_{k,t,p,\ell,h},
\quad \ell=0,\ldots,17.
$$

它表示同一frame、同一relative horizon position在一层计算中新增了什么policy evidence。它不是视频运动，也不是动作差分。

### D. flow velocity与probe noise

$$
V_{k,t,p,h}\in\mathbb{R}^{32},
\qquad
\epsilon_{p,h}\in\mathbb{R}^{32}.
$$

这里的flow velocity是\(s=1\)噪声端点处的Action Expert输出，不是完整denoised动作。当前`TargetOwnerProjector`把layer input、residual、flow velocity和probe noise以family-dependent方式映射到38 owners。`c61518d:src/ember/ecp/observer.py:L63-L259`。

### E. 38-owner紧凑policy-response lattice

现有设计将上述量压为：

$$
Z^{\text{owner}}
\in
\mathbb{R}^{K\times T_k\times2\times38\times50\times128}.
$$

38 owners对应18个q targets、18个v targets、action-in和action-out。该结构由2026-08-24审查明确提出，并由当前实现落地。`c61518d:docs/expert_review_20260824_native_factor.md:L120-L219`。

本次修订不直接沿用当前`state + gate×residual`的早期相加。建议在第一次frame-level attention之前把以下channel保留为不同token type：

* layer input；
* residual increment；
* action-in的probe-noise channel；
* action-out的flow-velocity channel。

这样增加的是一个小的channel-type轴，不是另一个latent坐标系。标准attention自行判断某个任务更依赖当前policy state、层内innovation还是flow response。

### F. 当前视频的真实target-native factor bank

对每个target \(j\)，另行捕获：

$$
X_j[k,t,p,h]\in\mathbb{R}^{d_j^{in}},
\qquad
Y_j^{abs}[k,t,p,h]\in\mathbb{R}^{d_j^{out}}.
$$

并沿teacher-video frame time构造：

$$
Y_j^{adj}(t,h)=Y_j(t,h)-Y_j(t-1,h),
$$

$$
Y_j^{init}(t,h)=Y_j(t,h)-Y_j(0,h),
$$

$$
Y_j^{goal}(t,h)=Y_j(T_k-1,h)-Y_j(t,h).
$$

这里比较相同的relative horizon index \(h\)，含义是“当前观测下，policy对lead-\(h\)位置的响应怎样变化”，**不声称不同frame的这些位置属于同一个绝对机器人控制时刻**。

实际target形状为：

| family      | \(d^{in}\) | \(d^{out}\) |
| ----------- | ---------: | ----------: |
| q，18个target |       1024 |        2048 |
| v，18个target |       1024 |         256 |
| action-in   |         32 |        1024 |
| action-out  |       1024 |          32 |

这些真实X/Y及dynamic differences是当前视频产生的部署期bank，不是task-expert字典。`c61518d:docs/expert_review_20260824_native_factor.md:L120-L219`；实现见`c61518d:src/ember/ecp/native_factors.py:L1-L520`。

## 3.2 四个轴分别怎样处理

### teacher-video frame time

完整保留到ordered event module。采用：

* 原始frame index；
* 视频内归一化相对位置；
* 相邻、short-window、initial-relative、goal-relative四类关系；
* 单调event assignment。

不使用跨视频共享的absolute action clock。

### action horizon 50

50个positions不做均值池化。它们保留到**relation-conditioned horizon attention**，再被learned compression为每个frame、每个owner、每类transition的一组token。

这是有证据的压缩位置：当前G2不是先平均50个horizon，而是让transition candidate与owner×horizon lattice双向cross-attend后再形成event evidence。`c61518d:src/ember/ecp/events.py:L22-L178`。

历史v6只对最终层50个suffix hidden取均值，随后V6-LPCP才补回18层native probes；v6/LPCP最多达到143但仍有高churn，说明“final-layer horizon mean”可作matched ablation，不应作为新主路径。`8553b613:docs/action_forecast_writer_v6_design.md:L1-L330`。

### flow time

首版只使用\(s=1\)：

* G1、G2、PNBTT的现有机制证据均建立在固定noise endpoint；
* Action-Forecast的完整10-step flow没有证明多flow-time本身无效，但它引入了机器人forecast语义和巨额开销；
* 当前没有证据证明多flow-time比更多任务、更多正确视频或更好的shared mapping更重要。

因此多flow-step不是工程开关，而是关闭的后备科学能力。

### layer depth

在进入learned event compression前，通过：

* owner对应层；
* layer input；
* residual increment；
* family/owner/depth embedding；

完整保留。压缩后不再保留显式19层轴，而由38-owner轴承载target-layer identity，并允许owner间attention比较浅层和深层证据。

## 3.3 probe轴

不提前平均两个probe。建议做一次可逆变换：

$$
Z^{even}=\frac{Z^++Z^-}{2},
\qquad
Z^{odd}=\frac{Z^+-Z^-}{2}.
$$

它不丢信息。`even`更接近probe共有响应，`odd`表达对noise方向的局部敏感性。两个channel继续作为独立type进入frame-response block。

当前2026-08-24合同也明确要求probe轴在Program形成前保留，不能像旧effect bank一样先平均。`c61518d:docs/expert_review_20260824_native_factor.md:L120-L219`。

## 3.4 Action-Forecast历史真正证明了什么

Action-Forecast v4确实让每个静态frame经完整π0.5产生future-action forecast，再沿视频时间形成Plan/Revision/Belief；它不是普通视觉encoder。`e8920cc7:docs/action_forecast_writer_design.md:L24-L190`。

失败接口被后续四臂因果移植定位得很清楚：

$$
u=t_i+h
$$

把不同teacher frames的局部forecast放入一个共享robot absolute-time轴，但teacher video、监督action episode、速度和embodiment并没有识别该对应关系。shuffle主要通过重排这些错配关系制造了一条偶然改善Object translation的Revision direction。`e8920cc7:docs/action_forecast_writer_v5_decision.md:L1-L360`。

因此应继承：

* 每帧policy response；
* frame-local relative horizon；
* adjacent policy-response change；
* ordered transition；
* task-grounded视觉变化。

应避免：

* \(t+h\)绝对时间对齐；
* 把不同frame forecasts解释为同一未来动作的重复测量；
* Plan/Revision residual的机器人时钟语义；
* 仅靠正的task-level functional loss期待普通RoPE自动学会顺序。

---

# 4. 独立决定的模块化Writer数据流

我建议的完整主路径为：

```text
exact language + K ordered action-hidden videos
        │
        ▼
[Module 0, frozen]
Per-frame π0.5 evidence capture
  patch/language hidden
  layer inputs + residual increments
  2 probes × 38 owners × 50 horizons
  flow velocity
  true per-target X/Y banks
        │
        ├──────── compact policy-response lattice ───────┐
        │                                                 │
        ▼                                                 │
[Module 1, learned]                                       │
Policy-Response Video Process Encoder                     │
  task-grounded frame relations                           │
  relation-conditioned horizon read                       │
  boundary-anchored ordered events                        │
  static context C + zero-mean process innovation D       │
        │                                                 │
        ▼                                                 │
[Module 2, learned]                                       │
Current-Video Native Factor Composer ◄── streamed X/Y bank┘
  38 targets × 4 mobile rank queries
  whole-bank standard cross-attention
  event-conditioned independent +/- logits
  exact signed pooling of raw X/Y
        │
        ▼
38-target rank4 residual A/B
        │
        ▼
[Module 3, fixed]
one small-core balanced canonicalization
+ frozen rank12 carrier concatenation
        │
        ▼
one complete 38-target rank16 LoRA
        │
        ▼
frozen π0.5 closed-loop rollout
```

只有Module 1和Module 2是主要learned模块。

---

# 5. 每个模块的输入、输出、shape、训练关系和扩展方式

## 5.1 Module 0：Frozen PI0.5 Evidence Capture

### 输入

* exact language；
* 单个teacher frame；
* fixed antithetic `[2,50,32]` probes；
* flow time固定为1。

### 输出

紧凑过程证据：

$$
Z_{k,t,p,j,h,c}\in\mathbb{R}^{d},
$$

其中：

* \(p=2\)；
* \(j=38\)；
* \(h=50\)；
* \(c\)表示state、residual、noise或velocity channel；
* 首版\(d=128\)，沿用Stage0已有投影宽度。

真实factor bank：

$$
X_j[k,t,p,h],\qquad
Y_j^u[k,t,p,h],
\quad
u\in\{abs,adj,init,goal\}.
$$

### frozen/trainable

* π0.5、Gemma、Action Expert及原权重全部冻结；
* component-init arm可以加载Stage0的projection；
* fully-random arm使用相同projection拓扑随机初始化并允许训练；
* 原始policy本身从不更新。

### 扩展方式

它不是靠增加新运算扩展，而是：

* 更宽的compact projection；
* 更多source/meta任务；
* 更高吞吐的frame batching；
* 必要时才增加flow-time samples。

### 是否接受functional gradient

冻结π0.5不接受。learned compact projection可接受；首轮component-init为控制成本也可以先冻结projection，只让后续video blocks接受functional gradient。

---

## 5.2 Module 1A：Frame Policy-Response Block

### 输入

每个frame：

* exact language tokens；
* 256个patch hidden；
* \(2\times38\times50\times c\) policy-response tokens；
* owner、layer、horizon、probe、relation-type embeddings。

### 操作

首先让language query读取当前frame patches，形成task-grounded视觉token。随后构造四类关系：

1. adjacent；
2. short-window；
3. initial-to-current；
4. current-to-final。

对每个owner \(j\) 和relation \(m\)，形成query：

$$
q_{k,t,m,j}.
$$

该query通过标准multi-head cross-attention读取同frame的：

$$
Z_{k,t,:,j,:,c}.
$$

输出：

$$
F_{k,t,m,j}\in\mathbb{R}^{d},
\qquad
m=1,\ldots,4.
$$

这里才第一次learned-compress 50-horizon轴。

### 标准重复block

每个Frame Policy-Response Block包含：

1. horizon/probe/channel cross-attention；
2. owner-axis self-attention；
3. gated MLP；
4. pre-norm residual。

深度扩展只复制相同block。

### 与现有资产的关系

component-init可加载：

* `ECPNativeObserver`中的patch/language/state/delta projections；
* `TaskGroundedTransitionMatcher`的grounding与relation参数；
* `EventConditionedHorizonBinding`的owner/horizon attention参数。

现有四类transition及owner-horizon binding已经在代码中完整实现。`c61518d:src/ember/ecp/events.py:L22-L178`。

### 是否接受functional gradient

是。最终correct-video functional loss从LoRA反传到此模块。

---

## 5.3 Module 1B：Ordered Event Block

### 输入

$$
F_k\in\mathbb{R}^{T_k\times4\times38\times d}.
$$

### 输出

每条视频独立得到：

$$
E_k\in\mathbb{R}^{8\times38\times d},
$$

以及：

$$
\alpha_k[e,t,m],\qquad \rho_k[e].
$$

### 顺序机制

保留：

* \(E_{\max}=8\)；
* soft occupancy；
* stay/forward；
* 首尾boundary anchor；
* relative frame positions；
* 不固定实际event数量。

G2曾发现K>1时alignment将多个local events压到单一canonical slot；只加入首尾边界锚定后，held full-vs-endpoints改善达到22.2047%，probe 38/40，median active events 4，K1、K4和same-task均通过。`c61518d:docs/research_history.md:L260-L390`。

因此：

* **单调顺序和boundary anchor保留为结构约束；**
* 当前semi-Markov emission、`tau`、`sigma`和固定Natural Program tuple不继续作为下游硬接口。

### context与innovation

对每条视频、每个owner：

$$
C_{k,j}
=
\frac{\sum_e\rho_{k,e}E_{k,e,j}}
{\sum_e\rho_{k,e}},
$$

$$
D_{k,e,j}=E_{k,e,j}-C_{k,j}.
$$

这只做一次event-axis centering。

* \(C\)保留task、scene和过程共有信息；
* \(D\)保留event-relative innovation；
* language与\(C\)可以调制“看哪类动态”；
* 只有\(D\)可以产生positive/negative signed branch之间的任务特异差异。

这一设计继承了两组直接证据：

1. Belief-v3中time-constant分量淹没temporal innovation；仅减去masked时间均值就把最终effective-LoRA顺序差异从约\(10^{-4}\)恢复到\(4\times10^{-2}\)量级。`e8920cc7:findings.md:L800-L1100`。
2. G2首轮把`P_lang/P_scene`直接注入动态decoder时形成静态旁路，清零`P_process`反而改善loss；后续才把动态head限制到process字段。`c61518d:docs/research_history.md:L130-L390`。

### K视频

每条视频先独立完成Module 1。随后保留：

$$
\{E_k,C_k,D_k,\alpha_k,\rho_k\}_{k=1}^{K}.
$$

不立刻求单个mean Program。Module 2把它们视为没有video-order embedding的集合：

* event index有序；
* video index无序；
* attention对video permutation等变；
  -最终单一LoRA对video permutation不变。

候选bank的base mass按每视频固定\(1/K\)归一，避免长视频吞噬短视频；learned scorer仍可在candidate层使用不同证据。

### 是否接受functional gradient

是。功能梯度到达：

* frame relation；
* horizon readout；
* event slots；
* assignment；
* process innovation。

---

## 5.4 Module 2：Current-Video Native Factor Composer

### 输入

* 每条视频的\(C_{k,j},D_{k,e,j}\)；
* exact language，仅作query/FiLM条件；
* 当前视频的真实\(X_j,Y_j^u\)；
* frame、probe、horizon、event assignment、relation type、owner/layer metadata。

### 查询数量

首版不是608个query，而是：

$$
38\ \text{targets}\times4\ \text{mobile ranks}=152.
$$

固定rank12 carrier不需要query。

### 标准重复block

每个target和mobile rank有一个共享参数化query。一个Bank Context Block包含：

1. query读取同target的ordered event innovations；
2. query读取当前target bank的低维projected candidate context；
3. rank-axis self-attention；
4. gated MLP。

重复2至若干层即可扩展能力。没有covariance、Cholesky、SVD transport或手工anchor solve。

### whole-bank context

第一遍bank read只把raw X/Y投影到低维context，使query知道当前candidate集合的分布。

第二遍才产生最终candidate logits并对raw X/Y做exact pooling。

这提供EBSRI一类whole-bank相对上下文，但用标准重复cross-attention实现，不再输出summary token、再解码scalar gate、再修正base primal。

### 是否接受functional gradient

是。功能loss直接到：

* signed logits；
* bank-context blocks；
* event innovations；
* Frame Policy-Response blocks。

---

## 5.5 Module 3：唯一LoRA Materializer

输入为每个target的rank4 factors：

$$
A_j^r\in\mathbb{R}^{4\times d_j^{in}},
\qquad
B_j^r\in\mathbb{R}^{d_j^{out}\times4}.
$$

每个target只做一次现有small-core balanced SVD：

* QR分解A/B；
* 只对\(4\times4\) small core做SVD；
* 平衡A/B scaling；
* 按A最大绝对pivot固定符号；
* 保持\(B_jA_j\)不变。

随后与rank12 carrier直接concat为完整rank16。现有实现正是这一语义，见：

* `c61518d:src/ember/ecp/native_materialization.py:L1-L146`。

这一次canonicalization处理rank permutation、A/B scale和rotation gauge。除此之外不再增加factor SVD、projection或transport。

---

# 6. 动态bank、静态共享先验、task-expert credit和learned residual的关系

## 6.1 部署期动态bank

部署时随当前teacher video变化的只有：

$$
\mathcal B_V
=
\left\{
X_j[k,t,p,h],Y_j^u[k,t,p,h],
\alpha_k[e,t,m],E_{k,e,j}
\right\}.
$$

它保留：

* video；
* frame；
* probe；
* action horizon；
* event；
* target；
* target source layer；
* output relation type。

它是mobile residual的唯一原始vector value来源。

## 6.2 静态共享先验

首版只存在三类静态共享先验：

1. frozen source π0.5；
2. 一套固定、task-independent rank12 carrier；
3. Writer共享网络参数和target-wise全局scale prior。

没有：

* task-indexed expert bank；
* task-name到factor的dictionary；
* nearest expert retrieval；
* held-task字典；
* language-conditioned静态adapter。

rank12+rank4的选择有明确历史依据：G1最终真实native pooling在held5达到114/250、breadth5/5、Goal2、Long1；full-rank16 PNBTT并未在task1和task93上形成一致改善。`c61518d:docs/research_history.md:L130-L260`。

## 6.3 task experts

task experts只在训练期提供：

* cross-episode action/flow target；
* successful policy response；
* task-local functional upper bound；
* source-preservation参考。

不使用其A/B factor作为deployment candidate value，也不把member identity输入模型。

训练loader知道某个视频和某组query属于同一个task，只用于配对batch；模型forward看不到task ID。多个successful lineages可以作为set-valued行为teacher，但不能形成可检索参数表。

## 6.4 learned residual

**首版没有free learned residual。**

上一版提出的小型free residual会重新打开：

$$
\text{language or static scene}\rightarrow\text{arbitrary factor}
$$

的旁路。在positive-only目标下，这一旁路尤其容易学成task-constant LoRA。

只有以下三项同时成立，才有资格开放一个小型event-conditioned residual basis：

1. 同一current-video bank上的task-local free signed logits已经充分优化；
2. 其正确视频closed-loop仍明显低于task-local rank16 oracle，且差距跨至少两个suite、包含Goal或Long；
3. 用相同video/event表示的task-local direct-factor正控能稳定补回这一差距。

即使开放，residual系数也只能由\(D_{k,e,j}\)产生；language和static context只能调制，不能独立产生系数。其norm初始必须严格小于native component，并单独报告占比。

目前G1已经证明真实bank存在rank4闭环容量，所以这项后备能力没有当前启动资格。

---

# 7. 最终LoRA输出几何

## 7.1 candidate logits

对target \(j\)、rank \(r\)、candidate \(n=(k,t,p,h,\ldots)\)，Module 2输出两组独立signed logits：

$$
\ell^{A,+}_{jrn},\quad \ell^{A,-}_{jrn},
$$

$$
\ell^{B,+}_{jrgn},\quad \ell^{B,-}_{jrgn},
$$

其中\(g\)是output native group。

令\(\mu_n\)为固定base measure，包含：

* \(1/K\)视频质量；
* frame quadrature；
* soft event assignment；
* probe/horizon正规化。

则：

$$
w^{A,\pm}_{jrn}
=
\operatorname{softmax}_n
\left(
\log\mu_n+\ell^{A,\pm}_{jrn}
\right),
$$

$$
\widetilde a_{jr}
=
\sum_n
\left(
w^{A,+}_{jrn}-w^{A,-}_{jrn}
\right)X_{jn}.
$$

输出侧：

$$
w^{B,\pm}_{jrgn}
=
\operatorname{softmax}_n
\left(
\log\mu_n+\ell^{B,\pm}_{jrgn}
\right),
$$

$$
\widetilde b_{jrg}
=
\sum_n
\left(
w^{B,+}_{jrgn}-w^{B,-}_{jrgn}
\right)Y_{jng}.
$$

各output groups拼接为完整\(\widetilde b_{jr}\)。

## 7.2 动态必要性约束

logit由common base与dynamic offset构成：

$$
\ell^\pm=b_{jrn}+\delta^\pm_{jrn}.
$$

其中：

* \(b\)可读取task-independent rank identity、current-bank context和静态语义；
* \(\delta^\pm\)必须经过bias-free的\(D_{k,e,j}\)路径；
* 当所有event innovations为零时：

$$
\delta^+=\delta^-=0
\quad\Rightarrow\quad
w^+=w^-
\quad\Rightarrow\quad
A_j^r=B_j^r=0.
$$

因此：

* no-video或静态重复帧只能得到carrier；
* language不能独立产生mobile residual；
* 正确、错误或乱序视频之间的差异只能由它们真实产生的process evidence和native bank带来。

这并不保证正确视频一定更好，但排除了最明显的language-only移动adapter旁路。

## 7.3 q、v、action-in、action-out怎样统一处理

固定native output grouping：

| family     | output grouping          |
| ---------- | ------------------------ |
| q          | 8个256D query-head groups |
| v          | 1个256D group             |
| action-in  | 32个32D blocks            |
| action-out | 1个32D group              |

G1的结构分析已证明：

* q若用一个scalar measure覆盖2048D output，会受base q projection列空间限制；
* action-in若用一个scalar measure覆盖1024D output，只能进入至多约33维的`span(W,bias)`；
* action-in按32个native-width blocks独立pool后，G1恢复Long并正式通过。

这不是四条不同compiler链。实现上是同一个ragged native-group signed-attention operator，只根据真实linear shape设置group mask。

## 7.4 参数共享

共享：

* Frame Policy-Response blocks；
* Event blocks；
* Bank Context blocks；
* 32D和256D native-block encoders；
* rank embedding；
* relation、probe、horizon、family metadata embeddings。

不共享最终factor：

* 每个target有固定target embedding；
* candidate bank按target隔离；
* 输出vector来自该target自己的真实X/Y；
* q/v不同层不能被一个全局vector直接广播。

首版不做跨target query self-attention。跨target协调由共同functional loss和共享video representation产生，避免历史common direction再次通过一个全局query覆盖38 targets。

## 7.5 scale与gauge

* 每个target使用fit-only、task-equal得到的一个全局scale reference；
* network只预测bounded relative rank/group gains；
* 不使用task-specific scale表；
* 最后一次small-core SVD同时解决rank-slot permutation、A/B scaling和rotation；
* 不再在中间做SVD、whitening、projection或transport。

---

# 8. positive-only训练为什么可能自然产生视频因果优势

## 8.1 先明确一个不可回避的事实

若给定exact language后，所有正确视频的最优策略函数完全相同：

$$
p(f^\star\mid L,V)=p(f^\star\mid L),
$$

并且模型存在language到adapter的直达路径，那么任何positive-only functional loss都无法迫使模型使用视频。

因此，**单纯task-expert distillation确实会产生task-constant target风险**。它能训练“正确策略”，不能单独识别“为什么由这个视频得到该策略”。

本次设计通过架构、自然视频监督和数据结构共同缓解，不宣称存在数学保证。

## 8.2 第一层：移除language-only移动adapter路径

* rank12 carrier与任务无关；
* mobile rank4必须读取当前视频的event innovation和真实X/Y；
* language只在Q/K或FiLM中告诉模型关注哪种变化；
* language不能作为factor value或free residual source。

因此，模型至少必须从当前视频获得某种task-specific value。

## 8.3 第二层：使控制LoRA的变量接受正确视频内部自然监督

新增一个positive-only的**causal policy-response prediction objective**。

对每条正确视频，在真实顺序中随机选择\(t\)和未来间隔\(\delta>0\)。用截至\(t\)的frame-response/event state预测：

$$
\Delta Z_{t,\delta,p,j,h}
=
Z_{t+\delta,p,j,h}-Z_{t,p,j,h}.
$$

预测目标：

* 来自冻结π0.5；
* 保留全部38 owners和50 horizons；
* 不使用teacher action；
* 不制造错误、乱序或倒序样本；
* target frame在该辅助forward中被mask，避免复制。

这迫使同一组event variables保留：

* 当前frame之后policy response怎样变化；
* 不同relative action horizons怎样变化；
* 哪些layer/owners先发生变化；
* 哪些变化在后续frame持续或消失。

它直接继承了Action-Forecast“逐帧policy response”思想，同时避免绝对机器人时钟。

历史v5因果结论本来也把下一原则性候选指向positive-only causal future frozen-feature prediction，而没有建议加入order contrast。`e8920cc7:docs/action_forecast_writer_v5_decision.md:L280-L360`。

## 8.4 第三层：cross-episode functional supervision

对视频episode A生成LoRA，在同task但独立episode B的真实observations/actions上计算π0.5 flow loss：

$$
\mathcal L_{\text{func}}
=
\mathcal L_{\text{flow}}
\left(
\pi_{\theta+\Delta\theta(V_A,L)}(o_B),
a_B
\right).
$$

这阻断逐帧动作复制。现有J2合同已经实现了：

* 两条fit video；
* 第三条same-task held video；
* panel A训练；
* panel B零梯度评估；
* video/action跨episode。`c61518d:configs/pi05_ecp_joint_program_primal_j2_v1.json:L1-L260`。

它仍允许task-constant解，所以必须与前两层一起使用。

## 8.5 正确视频之间提供什么

同task正确视频不需要生成参数上相同的LoRA。它们提供：

* **共同过程结构**：接近、接触、抓取、移动、建立目标关系；
* **不同scene realization**：对象位置、视角、速度和轨迹不同；
* **不同native support**：每条视频产生不同X/Y候选；
* **共同功能目标**：在独立query states上完成同一任务。

训练只要求功能正确，不做A/B或LoRA cosine一致性。若不同视频通过不同native directions实现相同函数，这是允许的。

## 8.6 数据结构比“更多episode”更重要

positive-only video causal identifiability要求训练任务形成交叉组合，而不是一条language对应一个孤立task：

* 同一verb出现在多个object和goal relation中；
* 同一object出现在多个process中；
* 相近scene支持不同操作；
* 相同高层goal可由不同阶段结构实现；
* held task是已有components的新组合。

71个source tasks和train24的使用需要先做这一factorial coverage audit。更多同task videos增加scene与trajectory覆盖，却不增加新的language–process–function mapping。

若审计发现给定language后，现有LIBERO数据几乎不存在会改变最优adapter的video variation，那么当前benchmark上的positive-only视频因果训练在统计上欠识别。这是数据事实，不应靠wrong-video loss伪造识别信号。

## 8.7 怎样裁决视频是否自然被使用

训练和checkpoint选择只看正确视频的：

* cross-episode functional；
* task-disjoint closed-loop；
* same-task新视频；
  -相邻checkpoint稳定性。

selected checkpoint冻结后才运行：

* language-only；
* no-video；
* first+final；
* wrong-video；
* shuffled；
* reversed。

只有冻结后correct自然优于这些条件，才能声称视频因果优势。attention weights、event distance、Program差异都不够。

---

# 9. 最小充分loss

主训练目标只保留三项，另加硬边界参数化。

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{func}}
+
\mathcal L_{\text{process}}
+
\lambda_{\text{pres}}\mathcal L_{\text{pres}}
}
$$

各loss先按其冻结baseline RMS或初始variance归一为无量纲量，不做LR或loss-weight小扫。

## 9.1 \(\mathcal L_{\text{func}}\)：正确视频真实策略功能

主目标。

* 输入只有正确视频与language；
* video与action query跨episode；
* generated唯一rank16实际安装到frozen π0.5；
* 使用ground-truth action flow loss或授权task-expert policy response；
* task等权、meta/target role等权；
* 不匹配teacher A/B；
* 不优化parameter cosine。

task experts只提供行为teacher。若有多个成功lineages，可对其policy response做set-valued minimum或soft-min，不选择task factor。

## 9.2 \(\mathcal L_{\text{process}}\)：正确视频内部policy-response变化预测

主辅助目标。

$$
\mathcal L_{\text{process}}
=
\mathbb E_{t,\delta,p,j,h}
\left[
\operatorname{Huber}
\left(
\widehat{\Delta Z}_{t,\delta,p,j,h},
\operatorname{stopgrad}\Delta Z_{t,\delta,p,j,h}
\right)
\right].
$$

要求：

* 预测全部50 horizons，不做horizon mean；
* owner、horizon和probe task-equal正规化；
* 使用真实frame gap，而非\(t+h\)；
* target interval在辅助forward中mask；
* 不读取teacher action；
* 不产生人工negative。

它的职责是让控制LoRA的event variables保留有序policy-response dynamics。它不是方法成功指标。

## 9.3 \(\mathcal L_{\text{pres}}\)：已有能力保护

在carrier/source已有正功能的训练states上使用单侧hinge：

$$
\mathcal L_{\text{pres}}
=
\max
\left(
0,
\mathcal L_{\text{gen}}-
\mathcal L_{\text{carrier}}-\epsilon
\right).
$$

它只防止mobile residual无故破坏已有能力，不要求LoRA接近carrier，也不使用wrong video。

## 9.4 固定边界，不作为主要loss

* 每target effective-update RMS cap；
* finite检查；
* event mask；
* boundary-anchored monotone assignment；
* K-set permutation invariance；
* small-core canonicalization；
* 唯一rank16 shape。

不需要强制dead-rank entropy。某个rank自然为零不一定是错误。

## 9.5 初始化期辅助

component-init可以短期复用G2已有的positive：

* action-temporal；
* progress-temporal；
* predicate rising；
* contact；
* scene relation。

一旦真实functional loss开始稳定下降，这些training-only heads线性衰减到零，不进入最终checkpoint forward。

不复用当前objective中的：

* `cross_task_contrast`；
* negative summaries；
  -内部margin；
* Program reconstruction；
* factor reconstruction。

当前`stage0_objective.py`与`natural_program_objective.py`都仍含`cross_task_contrast`，新主路径不应机械继承。`c61518d:src/ember/ecp/stage0_objective.py:L1-L130`；`c61518d:src/ember/ecp/natural_program_objective.py:L1-L200`。

明确排除：

* wrong-video ranking/margin；
* shuffled/reversed loss；
* no-video/language-only contrast；
* task-ID classification；
* A/B或LoRA cosine主loss；
* attention occupancy作为成功标准；
* Program reconstruction替代policy function。

---

# 10. 第一项决定性实现与GPU实验

## 10.1 第一轮只实现什么

只接通四项：

1. 现有冻结per-frame no-state PI0.5 capture；
2. Frame Policy-Response Blocks；
3. Ordered Event Blocks；
4. current-video signed native-factor composer加现有materializer。

不实现：

* free residual；
  -多flow-time；
* task-specific full rank16；
* learned video reliability；
* Action Meta；
* on-policy RL；
  -新functional evaluator；
  -复杂通用缓存框架。

## 10.2 直接复用的资产

复用：

* source π0.5和官方preprocessing；
* `ECPNativeObserver`的19-layer/50-horizon hooks；
* fixed antithetic probes；
* 38-owner target contract；
* task-grounded patch projections；
* 四类transition定义；
* boundary-anchored event logic；
* native X/Y hooks与streaming output bank；
* q八组、action-in三十二组native pooling；
* rank12 carrier；
  -small-core materializer；
* J2的12-task panel A/B、video split和cross-episode functional runner；
* task experts和已有action queries。

不接入新主路径：

* fixed Natural Program tuple；
* `OwnerLanguageReader`和`OwnerSceneReader`的下游absolute code；
* PNBTT whitening/Cholesky/tangent transport；
* bank summary、anchor、family scalar gate；
* R1–R13 fixed token/chart；
* neural \(q_\pi\)；
* effect-code realizer；
* Action-Forecast Plan/Revision/Belief；
* GOMQ open memory；
* LMMPC phase/controller；
* Action Meta。

## 10.3 最小smoke

一个train task、一个K1和一个K4 batch，只检查：

* `[2,38,50]` response轴未被误平均；
* 19 layer hooks和38 targets对应正确；
* no state/proprio/action进入Writer；
* `L_func`梯度非零到达FrameResponse、Event、FactorComposer；
* `L_process`梯度非零到达FrameResponse和Event；
* K4 video permutation输出一致；
* chunked与one-chunk的effective \(BA\)一致；
* 38 targets、76 tensors、唯一rank16被真实policy消费；
* source、carrier、Action Meta均冻结。

通过后立即上GPU科学实验，不增加大量合同测试。

## 10.4 第一项科学实验：Positive-Only 12-task Process-to-Function Qualification

复用J2的既有split：

* gradient meta：1、8、9、32、52；
* gradient target：72、73、75、93、94；
* true task-held meta：2；
* true task-held target：74；
* 每个gradient task两条fit correct videos；
* 第三条same-task held video零梯度；
* panel A训练、panel B零梯度；
* video/action跨episode。

这组任务同时包含历史PNBTT中具有代表性的task1和task93。

首轮K=1，避免把shared mapping与K聚合同时改变。K1/K4结构只在smoke中验证。该轮出现明确shared信号后，下一轮才将K∈{1,2,4}混合训练。

### 两个matched arms

只比较一个主要因果变量。

#### Arm A：完整policy-response

使用：

* 19 layer boundaries；
* residual increments；
* 38 owners；
* 50 horizons；
* antithetic probes；
* flow velocity；
  -当前视频X/Y bank。

#### Arm B：coarse-response ablation

保持：

* 完全相同的event和factor composer；
* 完全相同的dynamic X/Y bank；
* 相同参数量和训练预算；

但video process query只读取：

* task-grounded patches；
* final-layer 50-horizon mean。

它近似历史v6的coarse Action probe。A相对B的差异直接回答：

> 完整Action Expert layer/horizon response是否为shared correct-video功能提供超出普通视觉／final-mean表示的价值？

两个arms都只用correct positive训练。

### 训练参数

训练：

* Frame Policy-Response Blocks；
* Ordered Event Blocks；
* Factor Composer。

冻结：

* source π0.5；
* raw taps；
* native X/Y values；
* carrier12；
* global scale reference；
* Action Meta。

使用现有J2的10 warmup + 100 effective updates，保存effective step60/100，即global step70/110。这样可与既有J2、R系列证据直接比较，不新增训练时标搜索。

## 10.5 primary评估只看positive evidence

step70和step110都报告：

1. 10个gradient tasks的correct-video panel-B functional recovery；
2. 第三条same-task held video；
3. task2与task74的true task-held correct video；
4. q、v、action-in、action-out；
5. held process-prediction loss；
6. 每task方向；
7. 相邻checkpoint变化。

进入correct-only闭环held5的条件采用J2已有、直接对应功能的问题，而删除其negative arms：

* train median ≥0.60；
* same-task held median ≥0.50；
* held/train ≥0.80；
* true task-held mean ≥0.40；
* 两个true-held各≥0.30；
* q/v各≥0.35；
* action-in/out各≥0.30；
* step70→110 task median下降不超过0.05。

此外，Arm A必须满足：

* 至少8/10 gradient tasks不低于Arm B；
* task2和task74都不低于Arm B；
* task1与task93的correct capacity不能再次一高一低地稳定反转。

这些都是正确视频功能，不含negative control。

## 10.6 尽快进入closed-loop

positive panel通过后，立即对step70和step110做held5 correct-only strict250：

* single checkpoint；
* correct videos only；
* paired carrier baseline；
* 报告breadth、Goal/Long、retention和success-set overlap。

进入更大shared训练的最低信号：

* 两个相邻checkpoint均至少60/250；
* breadth至少4/5；
* carrier retention至少33/43；
* Goal或Long至少有一个非零；
* Arm A稳定优于Arm B。

60/250不是最终成功线，只表示新Writer已经产生了足够宽泛的真实shared增量，值得扩大数据。

## 10.7 冻结后才看因果controls

基于correct-only结果和相邻稳定性预先选定一个checkpoint并冻结。然后一次性运行：

* same-task other；
* wrong video；
* no-video；
* language-only；
* first+final；
* shuffled；
* reversed。

这些结果：

* 不改变checkpoint选择；
* 不回传梯度；
* 不决定loss权重；
* 不触发“针对shuffle再加一项loss”。

若correct没有自然优于controls，candidate不具备视频因果主张。新的研究判断必须回到positive-side证据：数据是否欠识别、process objective是否未到达composer、或dynamic bank是否可互换，而不能把control类别作为训练标签。

## 10.8 失败模式如何定位

| 结果                                          | 最早解释                                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| Arm A的process prediction也很差                 | frame policy-response或event表示失败                         |
| process prediction好，correct functional fit低 | factor composer或positive functional credit失败            |
| task-local free signed logits高、shared fit低  | shared mapping/credit/identifiability失败，bank support未失败 |
| train高、same-task held低                      | video-specific overfit                                  |
| train和same-task held高、task2/74低             | task-disjoint generalization或任务组合覆盖不足                   |
| Arm A≈Arm B                                 | 完整layer/horizon证据尚无增量，复杂前端不具资格                          |
| task1高、task93低或相反                           | capacity仍有task-dependent几何冲突                            |
| correct高，冻结wrong也高                          | positive distribution没有学出bank/task specificity          |
| correct高，shuffle/reverse更好                  | 时序被使用但语义错误，重演Action-Forecast v4                         |
| score/LoRA内部指标好而闭环低                         | 重演G3 proxy-to-function断裂                                |
| 单点高、邻点换手                                    | 重演v6/GOMQ support churn                                 |

## 10.9 component-init与fully-random

第一项实验先用component-init，原因是其目标是裁决新的模块边界和证据路径，而非同时测试从零发现所有结构。

一旦Arm A通过12-task positive Gate和held5 correct-only Gate，立即运行同拓扑fully-random fresh arm：

* 相同raw taps；
* 相同模块；
  -相同数据；
  -相同optimizer和更新数；
  -不加载Stage0/G2 projection、event或factor参数；
  -仍冻结source和carrier。

两者都必须在进入validation8之前完成。

解释：

* init只加快收敛，终点相同：G1/G2主要是优化资产；
* init终点更高：已有native/event归纳偏置有独立价值；
* random更高：旧Program初始化带有坐标包袱；
* 两者都失败：不扩大模型或做普通超参扫。

## 10.10 何时进入validation8 strict paired400

必须先满足：

1. 12-task positive functional Gate；
2. held5 correct-only稳定增量；
3. Arm A优于coarse-response ablation；
4. component-init与fully-random matched比较完成；
5. K∈{1,2,4}混合训练后same-task视频保持；
6. 相邻checkpoint低churn。

随后在扩大后的train24+审计meta tasks上fresh训练，validation8只在预注册相邻checkpoint运行correct arm。selected checkpoint冻结后，再补完整controls。

最终仍只认：

* strict correct >145/400；
  -相邻checkpoint稳定；
  -低churn；
  -四suite非零；
* Goal/Long实质贡献；
* same-task视频鲁棒；
  -冻结后完整因果controls成立。

---

# 11. 主要风险、停止条件和仍缺失的artifact

## 11.1 主要风险

### 风险A：positive-only统计欠识别

这是最大风险。若language已经唯一决定最优task adapter，且不同正确视频不改变功能目标，\(\mathcal L_{\text{func}}\)没有理由偏爱过程。

停止信号：

* full-response与coarse-response在true task-held上相同；
* \(\mathcal L_{\text{process}}\)明显改善，但功能不使用它；
  -冻结controls显示full、endpoints、shuffle几乎等价。

此时优先结论是现有task结构缺少video-dependent function variation，而非再加更大Writer。

### 风险B：zero-mean innovation限制过强

若某些任务主要依赖绝对goal state，只有\(D\)能打开signed residual可能损伤容量。

判别：

* event/process预测正常；
* task-local current-bank free logits强；
  -新的innovation-constrained composer在correct fit上仍显著低；
  -允许同一current-video absolute event token进入branch difference后correct恢复。

只有出现这一正控，才允许放宽动态必要性；不能直接恢复language-only路径。

### 风险C：current bank仍只是可互换数值基底

历史full-inverse operator使wrong bank也能实现同一primal。新方案删除inverse solve，但whole-bank attention仍可能从不同bank取出近似公共方向。

只能由冻结wrong-video评测发现。不能用wrong loss修复。

### 风险D：process auxiliary学习通用运动而非policy功能

\(\mathcal L_{\text{process}}\)可能很好，但composer不使用或使用错误。最终依据仍是cross-episode policy loss和closed loop。

### 风险E：task-specific rank4真实不足

目前没有充分证据支持这一结论。G1通过、full-rank16 PNBTT未形成一致改善，都反对立即扩rank。

### 风险F：端到端gradient破坏G2结构

通过三层控制：

* per-video独立、K permutation和boundary monotonicity是硬结构；
* process prediction持续保留有序response；
* G2 same-task/event指标只监控，不阻塞真实闭环。

若G2指标下降而closed-loop和冻结controls改善，应以行为为准。

## 11.2 明确停止条件

停止当前函数类，不做LR/seed/width小扫的情形：

1. 两个matched arms在12-task train fit本身都近零；
2. task-local composer也不能利用同一bank，而现有G1 free logits仍强；
3. full-response不优于coarse-response；
4. train高但两个true task-held持续无增量，扩大自然meta-task映射后仍不改善；
5. task1与task93继续稳定相反；
6. 相邻checkpoint持续高churn；
7. 冻结controls显示correct无自然优势；
8. throughput主要耗在每condition重复全bank的大矩阵分解——本设计不应引入这类算子。

## 11.3 当前未能直接核验的原始artifact

我此次通过远程GitHub读取了tracked代码、config、历史文档和formal summary，但无法直接打开本地ignored `runs/`中的raw rows、checkpoint tensor和日志。因此下列事实属于“仓库中已登记的formal结果”，没有在本次会话中从raw rows重新聚合：

* G1：
  `runs/outputs/pi05_ecp_native_factor_g1_action_in_groups_step0_strict250_31f0053_gpu01p234567_r3_20260825/`
* G2：
  `runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/`
* Action-Forecast v4 step825本地400-row controls；
* PNBTT gate-aligned E1：
  `runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`
* PNBTT full-rank16：
  `runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`

仓库tracked summaries仍清楚记录：G1通过、G2通过，而PNBTT最终将wrong和margin压好后correct/held仍不足，full-rank16在task1与task93呈现相反行为。

此外，当前还缺三项决定性科学artifact：

1. **full 19-layer×50-horizon response与coarse final-mean response的positive-only matched shared对照；**
2. 给定language后，现有train/meta tasks是否存在video-dependent optimal function的factorial data audit；
3. 新event-conditioned signed composer的task-local正控，确认无PNBTT solve时仍保留G1 bank容量。

第一项实验正是为了生成这三类证据中的前两类关键部分。

---

# 12. 科学合同、证据决定、工程选择和后备能力

## 12.1 owner与信息墙规定的科学合同

这些不能由实现者调整：

* exact language + K条same-task、action-hidden、内部有序正确视频；
* source π0.5冻结；
* teacher frame path不得读取state/proprio/action/reward/task ID/filename；
* teacher-video time、action horizon、flow time和layer depth不可混淆；
* 每条视频独立保序；
* 跨视频只做置换不变处理；
* 最终task-specific mobile update必须依赖视频；
* functional gradient到达video/event learned模块；
* 每condition只生成一套38-target rank16 LoRA；
* validation/test不产生梯度；
* wrong/no-video/language-only不作专门训练contrast；
* shuffled/reversed只作selected checkpoint冻结后的最终control；
* Action Meta首版关闭。

## 12.2 由历史正负证据支持的架构决定

这些是本次主张：

* fixed flow time \(s=1\)；
* 固定antithetic probes；
* 19 layer boundaries和50 horizon positions在第一次learned读取前保留；
* state与residual increment不提前相加；
* 四类frame relation；
* max 8 ordered event slots；
* boundary-anchored monotonic order；
* current-video真实X/Y作为factor value；
* no task-expert dictionary；
* no free residual首版；
* rank12 carrier + mobile rank4；
* 38×4 queries；
* q八组、action-in三十二组；
  -一次small-core canonicalization；
* static context只能调制，event innovation才能产生signed branch差异；
* full Natural Program schema不再是强制中间瓶颈；
* PNBTT covariance/whitening/transport不进入新主路径。

## 12.3 可根据profile和吞吐选择的工程参数

这些不改变主要科学假设：

* hidden width是128还是256；
* FrameResponse/Event/BankContext block各有几层；
* attention head数量；
* FlashAttention、SDPA或等价kernel；
* frame microbatch；
* target chunk；
* X/Y cache布局；
* CPU offload；
* activation checkpoint；
  -每卡任务数；
* 1–6张A40的data-parallel布局；
* optimizer精确实现；
* BF16/IEEE FP32边界；
* checkpoint cadence的工程间隔。

但以下不属于普通工程选择：

* 是否保留50-horizon；
* 是否读取layer residual；
* 是否允许language-only residual；
* 是否使用current-video bank；
* 是否top-k删bank candidates；
* event slot上限；
* mobile rank；
  -是否加入多flow-time；
  -是否开放free residual。

它们都会改变科学假设。

## 12.4 只有特定信号后才有资格打开的后备能力

| 后备能力                            | 唯一启动条件                                                                  |
| ------------------------------- | ----------------------------------------------------------------------- |
| event-conditioned free residual | bank task-local support显著低于direct-factor正控                              |
| 更多mobile ranks                  | rank4 free-code已收敛且同构full-rank16跨任务一致显著提高                               |
| 多flow-time                      | fixed-\(s=1\) representation失败，而完整denoising read-only正控在held正确视频上提供稳定增量 |
| 增加event slots                   | event occupancy长期顶满8且长视频正确能力显著低于短视频                                     |
| cross-target query interaction  | 各target单独方向正确，联合物化后出现可重复跨target干扰                                       |
| Action Meta                     | base Writer已有稳定、跨checkpoint闭环增量，剩余错误明确集中在action control detail          |

---

## 最终主张

我对上一版建议作了实质收缩和重构：

> **下一代EMBER不应从普通视频embedding直接生成LoRA，也不应继续把冻结Natural Program送入更复杂的bank transport。它应首先把每个静态teacher frame在π0.5 Action Expert中的layer×horizon×probe policy-response field视为视频过程的核心观测，再沿teacher-frame time形成ordered、owner-aligned event innovations；这些innovations与当前视频的真实X/Y bank在一个标准、可重复的set-attention composer中共同产生rank4 signed native factors，最后一次性与固定carrier拼成rank16。**

这套设计只保留三个不可复制的固定边界：

1. π0.5原生证据捕获；
2. boundary-anchored视频顺序；
3. LoRA factor gauge与materialization。

其余能力都通过可重复attention/MLP blocks、更多自然meta-task mappings和更大训练规模扩展。

同时必须保持一个清醒的科学边界：

> **positive-only functional监督本身不能保证视频必要性。**
> 新设计通过取消language-only移动adapter、让event innovation成为signed residual的必要路径、加入正确视频内部的causal policy-response prediction，并使用task-disjoint组合数据来建立可辨识性。最终是否真正正确使用视频，仍只能由冻结后的closed-loop controls裁决，不能由训练loss或内部表示提前宣布。
