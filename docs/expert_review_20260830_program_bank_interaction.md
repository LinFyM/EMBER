# 核心结论

我严格锁定并核对了 `main@b59d7bdd5fd7c2990c2f6e0eb28f170419ac7a84`。`910fb204..b59d7bd` 是无分叉的线性历史，`b59d7bd` 比上次审查提交超前 62 个提交；当前 authority 确实停在 R13 non-pass 后的专家复核点，没有旧配置所指向的活动路线可直接恢复。

**EMBER 现在卡住的地方已经比上次更清楚，但还没有找到最终根因。** 已经找到并可以判定的是：现行

$$
P\longrightarrow d_P,\qquad
q_B=C_B^+d_P
$$

这一分解把当前 bank 当成了“表达同一个方向的坐标系”，而不是“参与决定应该写什么方向的条件证据”。在小 logit 区域，

$$
f_B\approx C_Bq_B=C_BC_B^+d_P=P_Bd_P.
$$

只要正确和错误 bank 的 retained subspace 都覆盖 \(d_P\)，两者就都能重放几乎相同的方向。P1 证明的是**可表达性和同任务跨视频容量**；R5 wrong-bank 反事实证明的是**这种可表达性并不专属于正确 bank**。二者完全一致，并不矛盾。

因此，依据 wrong-bank 结果重新打开 bank/operator 因果接口是合理的，也是对我上次结论边界的正确修正。不过需要纠正一个表述：R5、R10、R13 的这些结果来自 cross-episode PI0.5 flow functional panel，不是环境 rollout 意义上的闭环成功。它足以重新打开机制接口，但不能替代后续 held5 和 validation8 的真实闭环裁决。当前 evaluator 确实是在冻结 policy 上计算 panel-B functional loss，而不是运行环境。

**binary full/half 主线应当停止。** R13 并没有数学上否定所有可能的离散内部机制，但已经充分否定了当前抽象：

* 一个只看 input retained projection 的全局标量；
* 一个阈值；
* 同时切换全部 38 targets、所有 rank 和所有 output groups 的 \(C^{-1}\) / \(C^{-1/2}\)；
* 错误分支本身对正确视频通常是强破坏性的。

它不是一个自然的“匹配/不匹配”政策变量，而是一个会整体改变谱方向的坐标开关。`.000060` 的 support 越界造成约 `1.17` 的 recovery 跳变，不是单纯阈值没调好，而是表示方式把很小的估计误差放大成了整个 LoRA 的离散翻转。

我推荐的唯一下一主线是：

> **保留已验证的 full-inverse 路径作为零初始化时的容量保持项，但在 exact signed pooling 之前加入 Program–bank 候选级联合交互，让 Program event query、当前视频的局部事件语义和每个真实 native X/Y candidate 共同产生连续、逐候选的 signed-logit 修正。最终只形成一套 signed measure、一组 rank4 factors 和一套完整 rank16 LoRA，不再先分类再选择 full/half。**

这可以理解为一个**零初始化的、bank-conditioned native-anchor correction**，但第一版应直接实现为 exact softmax 的候选级残差 logits，从而保持现有两次 native read，不增加第三次全 bank pass。

我主观判断：

* 这条新接口在 R5 fixed-route 机制 Gate 上通过的概率约 **40%–60%**；
* 它最终转化为跨任务 Natural Program 能力并达到 validation8 `>145/400` 的概率约 **20%–30%**。

最大风险不是优化器，而是 LIBERO 中不同任务的 native banks 和冻结 Stage0 动态证据本身高度共享：候选级模块可能仍然只学到语言/场景身份，或者在十个任务上记住 bank 指纹，却无法形成真正可迁移的过程—政策对应关系。

---

# 1. 对远程仓库和实验事实的复核

## 1.1 对上次建议的执行基本正确

上次建议中最关键的工程改造已经被准确实现：

1. `NaturalProgramModel` 现在把冻结 source/Stage0 工作拆成 `encode_frozen_evidence()`，将 language embedding、patch states、两probe process、presence、uncertainty、posterior和视频边界缓存为 `FrozenProgramEvidence`。
2. `compile_program()` 在这些冻结证据上重新运行可微的 language reader、scene reader、process fusion、aligner 和 Dynamic-K aggregation。
3. `prepare_joint_program_primal_condition()` 不缓存可训练 Program，而是每步从冻结 evidence 重新编译，恢复 Program 梯度。
4. native X/Y、current-bank eigensystem和action panels可缓存，避免重复运行冻结PI0.5。
5. J2仍只输出唯一完整38-target rank16 LoRA。

这正是我上次要求的“冻结重计算资产，但不把Program输出缓存成不可微常量”。没有发现实现偏移。

## 1.2 J2 positive control确实排除了基础功能链故障

J2 task-local control使用与joint路线相同的：

* current native banks；
* current-bank solve；
* exact signed replay；
* carrier12+rank4；
* cross-episode functional loss；
* 两条fit视频共享一个task-local primal；
* 第三条视频和panel B零梯度。

其held/fit benefit retention中位数为`1.0144`，held factor recovery为`.8078`，四family约`.77–.85`，10/10 held panel-B优于carrier。因此以下问题可以继续视为已排除：

* functional panel方向完全无效；
* rank4无法在该数据上产生改善；
  -固定scale必然把正确方向压死；
  -跨两条视频训练后无法迁移第三条同任务视频；
  -链式 functional LoRA gradient 没有接到primal。

需要保留一个限定：task-local code由fit-only teacher consensus初始化，因此它证明的是“可优化和可保持”，不证明从完全随机native方向能够自然发现该解。

## 1.3 J2/J3的结论成立

J2 joint在step110只有：

* train `.170800`；
* held-video `.164623`；
* true task-held `.122798/-.109179`；
  -四family及wrong Program/bank interaction接近零。

梯度审计又显示各模块有有限梯度，但不同任务的functional gradients大量相消，生成方向高度task-common。因此“正确pair functional-only允许共享残差shortcut”是合理归因。

J3增加wrong Program/wrong bank counterfactual后，正确方向没有提高，主要变化来自错误组合变坏。这确实淘汰了“继续叠加cyclic negative hinge即可恢复routing”的具体假设，而没有反证functional credit本身。J3配置也表明唯一变化是counterfactual training credit，Stage0、operator、carrier和scale均保持冻结。

## 1.4 R1—R5的分解总体严谨

R1—R5形成了一个有信息量的函数类排除链：

| 阶段                        | 实际证明                                                            |
| ------------------------- | --------------------------------------------------------------- |
| R1 fixed orthogonal token | 清晰task route能够被共享scorer利用，但functional-only从随机方向发现完整四family解很弱   |
| R2 privileged critic      | 稠密teacher geometry能恢复部分v/action-out，但与真实functional gradient并不一致 |
| R3 owner×group heads      | 原共享output head确实限制action-in；修改后action-in/out显著恢复                |
| R4 functional-code init   | 在清晰route下，成功task-local方向可以被共享heads插值；训练时chart微小漂移会破坏已初始化坐标      |
| R5 frozen chart           | 固定功能坐标后，共享heads和真实functional训练可达到`.94/.963`且四family约`.82–.84`   |

R5因此是很强的**条件可行性证明**，但只针对十个见过的fixed route tokens，不证明部署Program或未见任务。

一个次要仓库问题是：R2、R3、R4、R5配置都继续声明R1的同一个`schema_version/status`，尽管critic、decoder、initialization和trainable partition已经不同。运行时通过附加字段进行了严格验证，所以不使结果失效；但未来每个具有独立科学含义的formal机制应有唯一schema，避免仅靠文件名和Git diff才能恢复实验语义。

## 1.5 R6—R10最重要的正证据是R10，而不是outer-code

R6证明R5 fixed-token chart无法直接接受Natural Program；同任务多视频输出约`.9994`稳定，说明问题不是普通video noise，而是稳定地落在错误坐标。

R7和R9进一步表明：即便outer direction cosine达到约`.60–.75`，真实policy utility仍可能为负。因此outer code、factor reconstruction或内部cosine均没有选模权。

R10则不同：它删除outer-code loss，只用真实generated-rank16 flow loss，在R9初始化上把train/held提高到约`.560/.544`，wrong-Program margin也达到`.279`。这是真正支持上次主张的结果：**只有最终policy functional credit才开始把Natural Program推入有用方向。** R10的task-held仍只有约`.151`，wrong-bank和interaction接近零，说明它解决了部分Program→utility，却没有解决跨任务与bank因果性。R10配置确认其训练目标确实只有generated rank16 cross-episode flow。

## 1.6 R11按预注册条件启动是正确的，但结论边界要收窄

R10满足“train/held-video明显强、task-held弱”，因此启动raw frozen-Stage0 probe完全符合上次建议。

你们的保守结论——“目前没有证据把首因归到Natural Program压缩，也没有依据解冻Stage0”——是正确的。

但不能进一步声称R11已经证明Program压缩无问题。R11并不是完全matched的raw-Stage0 universal decoder：

* 它把raw Stage0 fields强制装入同一个`NaturalProgram`形状；
  -继续使用R9稳定chart；
  -冻结process fusion和canonical aligner；
  -只训练language/scene readers及native heads。

因此raw Stage0输入与R9 chart存在明显坐标分布错配。R11证明的是“raw view不能直接救活R9固定chart”，而不是“raw evidence不比Program更充分”。

未来若再次做raw-Stage0 sufficiency probe，必须让raw arm和Program arm使用相同、可训练的新Program–bank interaction模块，而不能继续复用只为Program坐标训练出的固定chart。

## 1.7 wrong-bank结果足以重新打开接口，但不是闭环证据

R5成功primal在correct/wrong bank上的functional recovery约为`.931/.946`，错误bank保留约全部收益。该结果直接否定：

> 只要Program产生正确task primal，current-bank full inverse就会自然保证视频bank特异性。

它没有否定：

* native values有容量；
* covariance计算正确；
* exact signed pooling正确；
* rank4有容量；
  -同任务多视频可保持。

当前评测代码将目标任务的Program与另一个任务的prepared bank组合，仍在目标任务panel B上计算functional loss；配对方式和归一化是合理的。

---

# 2. 哪些结论正确，哪些过强

## 2.1 正确结论

### 重新打开operator/bank interaction是正确的

我上次的P0/P1判断只排除了“operator不能表达正确方向”和“同任务视频无法稳定重放”，从未证明wrong-bank specificity。新反事实回答了一个P1没有检验的问题，因此是正常的证据更新，不是无依据改线。

### R12存在credit ownership冲突

让同一个primal既要最大化policy utility，又要被retained-projection loss推成bank classifier，确实可能产生冲突。R12中正确视频进入half分支后功能严重下降，说明compatibility loss可以把functional方向推出原有basin。

### R13解耦有真实但有限的帮助

独立probe把AUC从约`.762`提高到`.831`、严格task排序从8/12提高到9/12，说明把classifier职责从functional primal分开不是完全无效。但它仍无法覆盖同任务held和task-held，所以不是充分解。

### 停止阈值、温度、谱幂和probe-width小扫是正确的

R13已经出现排序冲突：一些正确pair的support低于wrong pair。任何单一阈值都无法同时修复这种交错；继续移动阈值只是在不同false positive/false negative之间选择。

## 2.2 过强或需要纠正的结论

### “soft mixture失败，所以后继必须近二值”是错误外推

当前文档曾据`.2387` soft mixture得出“后继必须保留近二值/离散选择”。这个结论应从active design中删除。

full和half不是两个功能等价的专家。在线性区：

$$
f_1\approx d,\qquad
f_{1/2}\approx C^{1/2}d.
$$

如果混合query：

$$
q_\lambda=\lambda C^{-1}d+(1-\lambda)C^{-1/2}d,
$$

则：

$$
f_\lambda\approx
\lambda d+(1-\lambda)C^{1/2}d.
$$

\(d\) 与 \(C^{1/2}d\) 可能旋转、抵消或改变family间平衡；随后还有独立score-RMS normalization和非线性softmax。因此线性插值比两个端点都差完全可能，并不说明真实Program–bank映射本质上离散。

### retained projection不是functional compatibility

代码中的support是：

$$
s_B(d)=\frac{\|U_BU_B^\top d\|}{\|d\|},
$$

它只测量primal有多少落在bank保留的特征子空间。它：

* 不看特征值；
  -不看方向在子空间内的具体位置；
  -不看output primals；
  -不看政策Jacobian；
  -不看该方向在目标任务状态上的行为作用。

所以它是**support/capacity指标**，不是政策兼容性定义。

更具体地说，当前route：

1. 只拼接38×4个input support；
2. 取全局p10附近一个标量；
3. 用这个标量同时决定所有input和所有output groups的谱幂。

这使一个或少数input rank方向可以切换整个LoRA，即使output侧支持完全不同。

### R12不只是credit ownership冲突，还存在明确的train–deployment mismatch

R12配置明确规定：

* 正确functional training使用`full_inverse_teacher_forced`；
  -部署forward使用`hard_full_if_supported_else_half`。

因此functional loss从未通过实际hard route惩罚“正确视频被误送进half”的损失；compatibility loss单独训练support，部署再执行不可微Python分支。R12发生false-negative灾难并不意外。

R13虽然冻结functional方向并单独训练probe，但仍只优化support labels，不读取functional panel。它能够回答“这个support定义能否从固定hidden预测”，不能回答“真实utility所需要的bank interaction能否从这些hidden预测”。

### R13不能唯一归因于线性函数类或Program表示

R13失败至少混合了四项因素：

1. 固定Program hidden可能没有被训练来表达retained-subspace fingerprint；
2. 每owner一个线性head可能不足；
3. p10 input-only support可能不是正确标签；
4. 全局binary route会把轻微误差放大。

因此不能从“部分fit pair也排不对”直接断言Natural Program不包含任何bank信息。更合理的判断是：

> 当前固定hidden → 线性vector probe → input-subspace p10 → global binary transport，这整个组合不可用。

---

# 3. 当前根因分层

## 3.1 已经证明的根因

### A. 当前bank进入得太晚

现行顺序是：

```text
Program
  → 全局event聚合
  → native primal dP
  → 当前bank covariance C
  → C⁺dP
  → signed replay
```

`ProgramNativePrimalScorer`先把`rank_event`按event weights聚合成一个全局`rank`，再生成高维input/output primals。当前bank内容在这些方向生成完毕后才出现。

虽然`SharedCompilerVideo`携带：

* canonical assignment；
* local scene；
* local process；
* local presence；
* local tau；
* local sigma；

但当前`SharedNativeFactorCompiler`只将Program生成的primals交给bank operator；这些逐视频局部字段没有参与query形成。

### B. full inverse主动消除了大部分bank spectrum

代码也明确把full inverse描述为保持原方向：

$$
q=C^{-1}d,
\quad
Cq\approx d.
$$

half和`.75`则分别保留\(C^{1/2}\)或\(C^{1/4}\)作用。

所以wrong bank重放强方向不是异常；它是当前数学在banks共享大子空间时的自然结果。

### C. binary route是放大器，不是根因解决方案

R13的support通过`float(...detach())`转成Python标量，然后一次性选择`.5`或`1.0`。它对support没有functional梯度，也没有逐target、逐rank或逐group的平滑责任分配。

## 3.2 最可能但尚未证明的根因

> **Program与当前bank缺少候选级、事件条件化的联合方向形成接口。**

R5说明清晰route足以产生强方向；R10说明Natural Program在真实functional credit下可以产生中等强度task direction；wrong-bank说明当前bank没有参与决定该direction。这三项共同指向上述接口，而不是继续改scale、rank或优化器。

## 3.3 仍未知的问题

1. 冻结Stage0的逐事件证据是否足以区分正确和错误bank的功能过程。
2. Natural Program schema是否保留了这种信息，还是在aggregation中丢失。
3. 候选级Program–bank交互能否跨未见任务泛化，而不是记住十个task-bank fingerprints。
4. 当前functional panel上产生的bank specificity能否转化为环境闭环成功。
5. carrier12是否在更强residual出现后仍是合适分配。

---

# 4. 推荐架构：Program–Bank Co-conditioned Signed Attention

## 4.1 总体数据流

```text
exact language + K ordered action-hidden videos
        │
        ▼
frozen source PI0.5 + frozen Native Stage0
        │
        ├── global Natural Program
        │     P_lang / P_scene / P_process / rho / tau / sigma
        │
        ├── per-video local event evidence
        │     local_process / sigma / presence / tau / frame→event assignment
        │
        └── per-video real native X/Y candidates
              input:  frame × probe × horizon
              output: frame × probe × horizon × {abs,adj,init,goal}
        │
        ▼
existing ProgramNativePrimalScorer
        ├── base full-inverse primal dP
        └── event/rank/owner query states z[j,r,e]
        │
        ▼
new Program–bank interaction scorer
        Program event query × local video event key × native candidate content
        → per-candidate continuous signed-logit correction
        │
        ▼
one exact signed measure over each real native bank
        │
        ▼
real X/Y weighted sums
        │
        ▼
one rank4 residual
        │
        ▼
carrier12 + residual4
        │
        ▼
one complete 38-target rank16 LoRA
```

## 4.2 Program query

继续使用现有、尚未聚合的：

$$
z_{jre}=\texttt{rank\_event}[j,r,e]\in\mathbb{R}^{128}.
$$

不要先把event轴消掉再让bank出现。

为避免新增大量高维参数，可以复用现有native heads和family trunks，把event-specific hidden映射到native query：

$$
u^{X}_{jre}=H^{X}_j(z_{jre}),\qquad
u^{Y}_{jgre}=H^{Y}_{jg}(z_{jre}).
$$

当前owner×group output heads已经存在，R3/R5证明这种参数所有权是必要的。

## 4.3 当前bank的候选key

对candidate \(n=(t,p,h,u)\)，不要求一个低维key承担完整factor reconstruction。使用：

1. **直接full-native alignment**

$$
\xi_{jren}
=
\frac{
\langle
\operatorname{RMSNorm}(u_{jre}),
\operatorname{RMSNorm}(v_{jn}-\mu_j)
\rangle
}{
\sqrt{d_j}
}.
$$

2. **local semantic alignment**

由当前视频在frame \(t\) 的canonical event assignment \(M_{te}\)，读取：

$$
c_{jte}
=
[
\text{local-process}_{je},
\text{local-sigma}_{je},
\rho_e,\tau_e,
t/T,p,h,u
].
$$

形成：

$$
\eta_{jren}
=
\frac{\langle Q_fz_{jre},K_fc_{jte}\rangle}{\sqrt{d_s}}.
$$

3. **乘性交互**

候选修正至少显式读取：

$$
[\xi,\eta,\xi\eta,\log\|v-\mu\|,
\text{probe},\text{horizon},\text{bank type}].
$$

这样若模型只使用线性native alignment，wrong-bank functional loss仍会直接暴露；语义与native内容的乘积为真正的条件交互提供最小函数类。

## 4.4 唯一连续signed measure

保留当前capacity-preserving full query：

$$
q^0_{B,jr}=C_{B,j}^{+}d_{P,jr}.
$$

新模块输出零初始化、bounded、在当前measure下中心化的candidate correction：

$$
\delta_{B,jrn}
=
\sum_e
\omega_{jre}M_{te}\,
h_f(\xi_{jren},\eta_{jren},\xi\eta,\ldots).
$$

最终两个branch logits为：

$$
\ell^+_{jrn}
=
\log\mu_n+\alpha q^{0\top}_{jr}v_n+\delta_{jrn},
$$

$$
\ell^-_{jrn}
=
\log\mu_n-\alpha q^{0\top}_{jr}v_n-\delta_{jrn}.
$$

然后：

$$
f_{B,jr}
=
\sum_n
\left[
\operatorname{softmax}(\ell^+)_{n}
-
\operatorname{softmax}(\ell^-)_{n}
\right]v_n.
$$

`h_f`最后一层零初始化，因此step0严格退化为R5/R10的full路径。随着训练，\(\delta\)逐candidate改变signed measure，而不是选择另一个全局谱坐标。

从一阶角度看，它产生的factor修正近似为：

$$
\Delta f_B
\approx
2\operatorname{Cov}_{\mu_B}(v,\delta_B),
$$

即一个由当前bank真实values与Program-conditioned scores共同形成的full-native anchor。

## 4.5 为什么bank不能退化成开关或幅度

在这个接口中，bank参与：

* 每个candidate的native方向；
* candidate的局部事件语义；
* frame→canonical-event assignment；
  -候选分布及softmax normalizer；
  -最终被加权的真实X/Y value。

它不是一个condition级标量。即使总scale固定，改变bank也会改变每个candidate的relative signed weight和最终factor方向。

必须保留一个`interaction_off`评测臂；只有interaction-on在保持correct utility的同时显著降低wrong-bank utility，这个模块才具有资格。

## 4.6 多视频与信息墙

对每条视频独立执行：

1. 单位质量时间quadrature；
2. native covariance和base full query；
3. 当前视频local event context；
4. candidate-level correction；
5. exact signed pooling。

随后按当前首版：

$$
f_{jr}=\frac1K\sum_{k=1}^{K}f_{kjr},
$$

再统一normalize和materialize。

因此：

* K=1严格identity；
  -视频集合置换不变；
  -每条视频内部保序；
  -不同视频不拼frame或previous/final状态；
  -长视频仍为unit-mass measure；
  -最终只有一个rank4 residual；
  -没有video选择、LoRA平均、第二adapter或task lookup。

## 4.7 为什么这不是恢复S1/S2

S1/S2要求低维chart或set-summary从头承担整个factor选择，fit本身就只有约`.15–.35`。

新接口有三项本质差异：

1. step0保留R5/R10已经验证的强full path；
2. 低维语义只决定一个candidate-level correction，真实factor仍由full-native X/Y承载；
3. 直接使用event-specific full-native query与真实candidate inner product，没有fixed random native-Q projection作为唯一瓶颈。

---

# 5. 最小决定性实验

建议把它登记为一个新的、唯一的 **co-conditioned bank-interaction qualification**，而不是R14式binary变体。

## 5.1 数据

使用现有十个gradient tasks：

* meta：`1, 8, 9, 32, 52`
* target：`72, 73, 75, 93, 94`

每个task：

* 两条既有mapping-fit K1 videos：训练correct；
  -第三条预注册same-task held video：零梯度；
  -训练negative：同role其他gradient task的bank，按确定性cycle轮换；
  -评测negative：额外使用task2或task74的零梯度bank，形成未参与negative训练的same-role wrong-bank；
  -panel A训练；
  -panel B评价；
  -不读取shuffled/reversed。

## 5.2 参数所有权

**初始化并冻结：**

* R5 step110 fixed route token；
* R5通过的feature chart和native heads；
* source PI0.5；
* Native Stage0；
* covariance/full query；
* exact signed replay；
* carrier12；
* scale；
* Action Meta。

**只训练：**

* 新的event-specific Program–bank interaction scorer；
  -候选语义投影；
  -branch-logit correction的最后层及其上游共享trunk。

这使实验只回答：

> 在已有强task direction的条件下，候选级Program–bank交互能否保持正确bank容量并连续消除wrong-bank utility？

fixed token仍只是training-only positive control，不构成部署路径。

## 5.3 主要loss

设functional benefit为：

$$
B(s)=L_{\rm carrier}-L(s).
$$

对两条correct视频：

$$
L_{\rm correct}
=
\frac12\left[
L_{\rm flow}(P,B_{c1})
+
L_{\rm flow}(P,B_{c2})
\right].
$$

对wrong bank只要求它不优于carrier，而不是无限恶化：

$$
L_{\rm wrong}
=
\operatorname{ReLU}
\left(
\frac{B(P,B_w)}
{B_{\rm free\text{-}primal}+\epsilon}
\right).
$$

总loss：

$$
L=L_{\rm correct}+L_{\rm wrong}.
$$

不加入：

* support BCE；
  -projection p10；
  -route labels；
  -factor cosine；
  -outer-code reconstruction；
  -subspace或spectral-power loss；
  -teacher LoRA target。

wrong-bank credit直接通过最终signed measure和唯一rank16 functional loss进入新模块。

## 5.4 Gate

在两个相邻预注册checkpoint上同时要求：

| 指标                                          |     建议阈值 |
| ------------------------------------------- | -------: |
| correct fit functional recovery median      | `>= .85` |
| same-task held functional recovery median   | `>= .80` |
| held / fit                                  | `>= .85` |
| unseen wrong-bank recovery median           | `<= .25` |
| correct − wrong median                      | `>= .50` |
| task级correct优于wrong                         |  `10/10` |
| correction-off相对correction-on的wrong margin差 | `>= .40` |
| 相邻checkpoint correct median下降               | `<= .05` |
| K1、唯一rank16、信息墙                             |      全通过 |

四family factor recovery继续报告，并要求没有family系统性反向；但它们不能在functional Gate失败时挽救结果。

## 5.5 吞吐

R5的110次训练在6卡上约`1503.6s`，即约`13.7s/update`；新增一个顺序执行的wrong functional arm，合理目标约为`20–25s/update`，不应超过现有45秒global-update硬上限。新scorer只作用于已缓存X/Y，不应重新运行冻结PI0.5。

R13完整评测每checkpoint约`765s`，且包含当前实验不再需要的language、endpoints、多个route统计和binary diagnostics。新Gate只需correct fit、correct held、interaction-off和wrong banks；应通过共享cache和六个独立workers显著缩短，不应再出现evaluation/training ratio约`1.84`。

## 5.6 结果解释

### correct fit无法保持

说明新的candidate scorer、streaming bias实现或其函数类破坏了已知R5容量。停止该实现，不应接回Natural Program。

### fit高、same-task held低

说明candidate interaction记住了fit视频，未学到同任务bank invariance。检查event/frame conditioning和candidate normalization，而不是增加classifier容量。

### correct和held都高、wrong仍高

说明模块仍在忽略bank内容，或当前local/native candidates缺少可用特异信息。此时不应接回Natural Program。

### correct/held高、wrong低

说明“唯一连续joint direction”接口成立。随后应立即把fixed token替换为Natural Program，联合训练Program、interaction scorer和native heads；不再插入新的chart、support或binary阶段。

---

# 6. G3、G4、Action Meta和Final计划修订

## 6.1 G3修订

原顺序：

```text
J2
→ compatibility classifier
→ binary full/half
→ G4
```

应改为：

```text
P0/P1容量证据（保留）
→ co-conditioned candidate interaction positive control
→ Natural Program + interaction + native heads joint functional Gate
→ scale / preservation
→ K2 / K4
→ held5 strict250
```

P0/P1仍然是重要正证据，但其结论应重新表述为：

> covariance、full solve、signed replay和rank4是可复用的容量primitive；Program-only primal + full inverse不是合格的bank-causal compiler。

## 6.2 是否现在直接做完整Writer端到端训练

**还不应立刻跳到完全随机、全部Writer联合训练。**

原因不是坚持分段课程，而是当前已经知道canonical operator缺少bank-conditioned direction formation。若直接全联训：

* 成功时不知道模型是否绕过native bank；
  -失败时无法区分Program、interaction、Stage0和credit；
  -强common residual仍可能吸收gradient。

但只应再做上述**一个局部接口Gate**。若通过，就应比原计划更早进入Program+compiler整体functional训练，不再重复R4/R5/R7/R9式坐标课程。

## 6.3 G4

G4首版应联合训练：

* Natural Program；
* Program–bank candidate interaction；
* native heads；
  -后续才加入scale/confidence。

继续冻结source PI0.5。Stage0首版仍冻结。

G4 primary loss保持：

1. correct cross-episode PI0.5 flow/action；
2. bounded wrong-bank benefit hinge；
3. 一个必要的carrier/source preservation guardrail。

只有出现广泛、稳定的真实closed-loop成功后，才加入on-policy success/progress。

## 6.4 Final两种初始化

Final必须在同一架构、数据、loss、训练节点和evaluation合同下比较：

### Component initialization

使用已通过Gate的：

* Stage0；
* Natural Program；
  -candidate interaction；
  -native heads；
  -scale初始化。

所有optimizer/scheduler fresh。

### Fully random Writer

随机初始化所有Writer-owned trainable模块：

* Stage0 Writer projections/observer heads；
* Program readers/fusion/alignment；
  -Program–bank interaction；
  -native heads；
  -scale/confidence。

source PI0.5仍冻结；rank与carrier合同必须与component arm完全相同，不能把架构差异混入初始化比较。

这符合仓库长期authority：G1–G3只是机制验证，不是Final强制课程；完全随机Writer必须保留为正式候选。

## 6.5 Final最小loss集合

建议最终只保留：

$$
L_{\rm Final}
=
L_{\rm correct\ flow/action}
+
\lambda_w L_{\rm wrong\text{-}bank\ neutralization}
+
\lambda_p L_{\rm preservation}
+
\lambda_o L_{\rm on\text{-}policy},
$$

其中\(L_{\rm on-policy}\)只有G4已有稳定closed-loop信号后才打开。

必须删除：

* Program Gram；
  -behavior coordinate；
  -functional-code cosine；
  -factor reconstruction；
  -native teacher LoRA loss；
  -p10 support classifier；
  -full/half route；
  -spectral-power选择；
  -outer-code chart；
  -task routing token；
  -所有只为探索期服务的subspace/polar losses。

## 6.6 Action Meta仍应关闭

当前没有证据表明action侧是剩余首因：

* R5 action-in/out已达到约`.821/.837`；
* R13反而是q/v低于action-in；
* bank interaction失败同时影响多个family。

只有同时满足以下条件才开放一次matched control：

1. 新base Writer已有稳定、广泛的closed-loop增量；
2. q/v和bank interaction Gate已通过；
3. 两个相邻checkpoint、两个task role中，action-in/out功能贡献持续比q/v低至少约`.15`；
4. rollout失败集中为动作幅值、时序或短程控制，而不是错误对象、目标或过程；
5. matched Action Meta on/off除该变量外完全一致。

保留条件仍是实际paired closed-loop净收益，而且breadth、Goal/Long、retention和video causal margin均不下降。

## 6.7 随未来数据扩大的Writer规模

新模块参数应按：

* family共享trunk；
  -固定owner embedding；
  -owner×native-group heads；
  -event/rank embedding；

组织。不要按task、video或frame增加参数。

高维event-native query可复用现有native heads，新增参数主要是小型semantic interaction MLP。因此任务数增加只扩大训练样本，不扩大模型参数或checkpoint表。

---

# 7. 具体代码级修改建议

## `src/ember/ecp/bank_conditioning/program_primal.py`

**保留：**

* `PrimalProgramState.rank_event`
* `rank`
* owner/family/rank/event embeddings
* owner×group output heads
* base input/output primals

**新增：**

```python
def input_event_queries(
    self, state: PrimalProgramState
) -> tuple[Tensor, ...]:
    # [rank, event, native_width]

def output_event_queries(
    self, state: PrimalProgramState
) -> tuple[Tensor, ...]:
    # [group, rank, event, native_group_width]
```

这些方法应复用现有heads，不再建立另一套全高维hyperdecoder。

**退役active路径：**

* `compatibility_input_heads`
* `compatibility_input_primals`
* `initialize_compatibility_probes_from_functional`

R13历史由Git/config/artifact保留。

## 新增 `src/ember/ecp/bank_conditioning/program_bank_interaction.py`

建议拥有：

```python
@dataclass(frozen=True)
class ProgramBankContext:
    canonical_assignment: Tensor
    local_scene: Tensor
    local_process: Tensor
    local_presence: Tensor
    local_tau: Tensor
    local_sigma: Tensor
    frame_positions: Tensor

class ProgramBankInteractionScorer(nn.Module):
    def input_logit_corrections(...)
    def output_logit_corrections(...)
```

输出必须是逐target、rank、candidate的连续correction，不输出route类别或LoRA factor。

## `src/ember/ecp/shared_compiler_data.py`

当前joint接口已经返回`NaturalProgramOutput`，应继续把其中：

* local fields；
* canonical assignment；
  -frame positions；

整理成`ProgramBankContext`传给compiler。

compact cache继续只保存冻结evidence和raw X/Y；Program及interaction输出不得缓存。

## `src/ember/ecp/bank_conditioning/operator.py`

`StreamingSignedPool`已经支持`logit_bias`和branch-specific bias，但当前在`canonical_block_candidates`非空时明确拒绝bias。新实现必须扩展fixed-microblock状态，同时buffer：

* candidate values；
* mass；
  -正负branch correction。

然后以相同canonical block归并，不能退回依赖外部frame chunk的浮点定义。

需要新增回归：

* biased materialized vs streaming；
  -chunk4 vs one-chunk；
  -branch bias gradient；
  -外部frame chunk不改变结果；
  -长K4内存。

## `src/ember/ecp/bank_conditioning/primal_dual_runtime.py`

**保留：**

* B0 covariance；
  -full \(C^+\) base query；
  -score RMS；
  -output bank boundary；
  -B1 exact replay；
  -fixed candidate microblocks。

**删除active控制：**

* `compatibility_support_threshold`
* `_plan()`中的Python hard branch
  -一个标量同时控制全部target/group的selected power
  -`.5/.75`作为canonical deployment选项

允许历史task-local scripts继续由旧commit复现，不应让这些分支继续污染新canonical forward。

## `src/ember/ecp/shared_compiler.py`

修改forward签名，使compiler同时接收：

```python
program: NaturalProgram
program_output_or_bank_context: ProgramBankContext
videos: Sequence[...]
```

执行顺序：

1. Program scorer产生base primals和event queries；
2. B0产生base full queries；
3. interaction scorer逐candidate产生correction；
4. exact signed pooling；
5. uniform K aggregation；
   6.唯一rank4 residual。

删除active output中的：

* `compatibility_supports`
* `selected_inverse_covariance_powers`

可将其移入历史diagnostic report，而不是deployment output。

## `src/ember/ecp/joint_program_primal/train_step.py`

新增一个新的明确schema，只训练新interaction模块。

functional loss通过实际deployment forward，不允许：

* correct arm override到full；
  -deployment arm走另一条hard path；
  -support surrogate替代functional gradient。

wrong bank loss必须使用同一unique forward。

## `src/ember/ecp/joint_program_primal/evaluation.py`

保留：

* correct；
  -same-task held；
  -wrong Program；
  -wrong bank；
  -interaction；
  -functional normalization。

删除binary Gate依赖：

* route fraction；
  -p10 threshold；
  -AUC；
  -selected power。

AUC可留作read-only分析，但无选模权。

## 文档

`docs/event_conditioned_policy_compiler_design.md`应删除“soft mixture失败意味着后继必须近二值”的结论，改成：

> 只淘汰full/half spectral endpoint及其hard/soft选择；下一接口让Program与bank在候选级共同形成唯一signed measure。

`task_plan.md`应把当前唯一下一项登记为上述co-conditioned interaction qualification，而不是R14 compatibility版本。

---

# 8. 若下一实验失败，怎样定位最早接口

| 观察结果                                             | 最早失效接口                                                |
| ------------------------------------------------ | ----------------------------------------------------- |
| correction零初始化不能复现R5                             | 实现、bias streaming或checkpoint装载错误                      |
| R5 correct fit从`.94`明显下降                         | 新interaction破坏容量，函数类/数值定义不合格                          |
| correct fit高、same-task held低                     | candidate scorer记视频；local event normalization或跨视频归纳失败 |
| correct/held高、wrong也高                            | 当前interaction忽略bank，或native/local evidence缺少特异信息      |
| fixed-route interaction通过，Natural Program train低 | Program→event-query/function acquisition失败            |
| Natural Program train/held高、task-held低           | 跨task generalization或Program schema问题                 |
| matched raw Stage0、同一可训练interaction显著高于Program   | Program压缩/aggregation是首因                              |
| raw Stage0与Program均低，但fixed route高               | Stage0 evidence或跨task可识别性不足                           |
| functional Gate通过、held5 rollout不升                | flow authority与真实闭环utility不一致                         |
| held5有增量、validation8不泛化                          | task diversity、meta-distribution或容量问题                 |

这里再次强调：若需要判定Program schema，raw Stage0对照必须使用同一个可训练interaction函数类；R11当前结果不足以承担这一停止裁决。

## 最小raw artifact需求

本次架构判断不依赖未跟踪raw banks。但若要精确判断R5 support为何AUC 1.0而R13不可迁移，建议从R5 cross-bank分析与R13 step70/110 `worker_*/result.json`导出每condition的：

* task、role、video demo、wrong task/video；
  -全部152个input retained-projection值，而不是只有p10；
  -每target/group retained rank、trace fraction、condition number；
  -对应output-side projection；
  -full和half在完全相同panel seed下的raw functional loss与normalized recovery；
  -每family factor recovery；
  -Program input-head hidden与probe output norms；
  -selected power及support；
  -panel name、visit、policy RNG seed。

不需要导出完整X/Y或checkpoint。这些字段足以检验：

* tiny p10 gap是否由少数target主导；
  -output侧是否与input route冲突；
  -support与full/half utility是否真正单调；
  -R5 AUC是否只是十个已知task的高维fingerprint。

---

# 9. 真正充分的停止条件

## 9.1 现在就可以停止的

以下已经有充分证据，应从canonical路线停止：

* Program-only primal + full inverse天然具有bank specificity的假设；
  -full/half binary route；
  -retained-projection p10作为部署compatibility定义；
  -独立linear compatibility probe；
  -soft或hard spectral-power mixture；
  -support阈值、温度、LR、seed、probe width的小扫。

## 9.2 停止当前Program schema所需证据

必须同时满足：

1. fixed-route candidate-interaction Gate通过；
2. Natural Program arm能够fit gradient tasks；
3. same-task held保持；
4. true task-held仍明显失败；
5. matched、同函数类、可训练的raw Stage0 arm比Program task-held至少提高约`.15`并达到有意义水平。

缺少第5项，不能把问题唯一归到Program压缩。

## 9.3 停止当前native candidate interaction所需证据

在R5 fixed route和强base方向下：

* correct fit仍无法保持；
  -或correct与wrong始终同时高；
  -新模块确实读取candidate native values和local events；
  -没有streaming、bias、scale或loss分支错误；
  -结果跨input/output及多个task一致。

这才足以淘汰该候选级函数类。

## 9.4 停止冻结Stage0所需证据

需要：

* fixed route + candidate interaction强；
  -Program arm弱；
  -matched raw Stage0 arm也弱；
  -train functional gradient接通且可拟合；
  -失败跨多个role和family；
  -不是固定chart错配。

满足后，才允许解冻Stage0最末端的process/presence/uncertainty投影；仍不应直接解冻VLM或source policy。

## 9.5 停止整个Native-Factor路线所需证据

至少需要完成：

1. 候选级Program–bank联合方向正控；
2. component-init整体joint Writer；
3. 完全随机fresh整体joint Writer；
4. 两个fold或等价独立任务切分；
5. 足够相邻checkpoint稳定性；
6. held5真实闭环；
7. validation8 strict paired400；
8. full、language、endpoints、wrong和same-task完整比较；
9. selected checkpoint冻结后再运行shuffled/reversed。

若两种Final初始化均无法形成：

* broad且稳定的closed-loop提升；
  -full相对language/endpoints/wrong的必要增量；
  -Goal/Long非零；
  -same-task视频保持；
  -或最终始终远低于`>145/400`；

此时才有充分依据判断，在现有LIBERO任务多样性、action-hidden video和static rank16 LoRA合同下，Native-Factor路线无法提供所需的跨任务amortized policy compilation。

R13远未达到这一停止标准。它已经足以终止binary门卫，但没有终止Program–bank共同生成唯一功能方向的研究主线。
