# 1. 核心结论

我对 `main@910fb204e8e3a5374ec988aa5e1da5bc042754aa` 及 `9b52e59..910fb20` 的 25 个提交进行了重新审计。我的核心判断是：

> **当前最早尚未解决的接口，不是 native bank、rank4、current-bank dual、exact signed replay，也还不能归因于 Stage0 或 Program schema 本身；它是“部署 Program 与 target-native primal scorer 之间的联合可识别性和功能 credit”接口。**

P0/P1 已经给出相当强的下游排除证据：同一个跨视频稳定 primal，经每条当前视频 bank 的 covariance 变换为 dual，再对真实 X/Y exact replay，可以在浅、中、深 q/v 和两个 action family 上取得约 `.94–.995` 的 held recovery。由此，继续修改 covariance、polar、低维 sketch、signed pooling 或 rank4 没有依据。

V3、V4、V5 则可靠地淘汰了三种**让 Program 自身直接形成全局 behavior cosine geometry** 的具体做法。但是，V5 只有 15 次 optimizer updates，其中前 9 次仍在 warmup；而且它监督的是固定、block-equal、单位化的 Program Gram，并不是最终由 Program→primal→真实 bank→LoRA 产生的功能。因此：

* V5 是一个合法的、预注册节点上的 **protocol non-pass**；
* 它不是“Program schema 结构性不可能”的证明；
* “macro4—5 平台 + 所有梯度被 clip”也不足以证明已经公平收敛；
* 但因为 V5 要求的几何本身过强且坐标不匹配，**不值得把主要资源用于续训 V5 或调整 clip**。

当前最有依据的主路线是：

> **取消“Program 必须先独立通过 behavior-Gram 硬 Gate”的要求，直接联合训练 Natural Program 与 `ProgramNativePrimalScorer`，用生成后的唯一 rank16 LoRA 在跨 episode teacher-action/flow panel 上的真实功能损失提供 credit；source、Native Stage0、current-bank primal-to-dual operator、carrier 和 scale 先全部冻结。**

这不是提前进入一个无法解释的全 Writer 黑箱训练。它只联合当前证据无法分开的两个相邻接口——Program 与 primal decoder；P1 已通过的 bank operator仍固定，信息墙和 Program×bank 因果 controls仍然保留。

下一步最多需要两个实验：

1. **12-task joint Program–primal functional qualification**：先回答当前 schema 与共享 nonlinear mapping 是否能在真实 functional loss 下同时取得 fit、held-video 和 true task-held；
2. 通过后恢复完整 task/video holdout；未通过且表现为“fit高、task-held低”时，只做一个同数据的 raw-Stage0 sufficiency probe，决定瓶颈究竟在 Program 压缩还是 Stage0。

---

# 2. 从远程仓库核实到的关键事实，以及对摘要的纠正

## 2.1 Git 与当前 authority

`9b52e59..910fb20` 确实是线性向前的 25 个提交，依次经历 S1、S2、Program-primal/current-bank dual、P0/P1/P2、95-task behavior authority、pointwise decoder、V3、V4、V5，没有隐藏的并行 active Writer 被重新引入。

仓库 authority 规定：`progress.md` 明确登记的当前状态高于配置或旧设计中的 `active/current` 字样。因此：

* `configs/pi05_ecp_natural_program_g2_behavior_kernel_v5.json` 虽仍写有 `active_globally_calibrated...`；
* `configs/pi05_ecp_shared_compiler_g3_v5.json` 也仍写有 `deployment_candidate: true`；

但 `910fb20` 的当前执行状态是 **V5停止、G3 P2暂停、等待新裁决**，不能因配置状态字段恢复运行。

## 2.2 S1、S2 的结论边界正确

S1 rank64 sketch-to-teacher 约 `.1567`，full-native free-query 两侧最低只有约 `.414/.254`，而同一 exact key 的 analytic reference约 `.996`。这明确淘汰了该 nested native-Q sketch，不支持继续加 rank。S2 的随机 chart 和旧 F3 chart均不能把两条fit video学到的选择迁移到第三条held video，而同bank free logits接近1。负结果只淘汰这两种低维 chart/summary实现，并不反证真实 bank或signed pooling。

## 2.3 P0/P1 已经把下游边界推进得足够远

当前 `SharedNativeFactorCompiler` 的逻辑是：

$$
\text{full Program}
\rightarrow \text{target-native primal}
\rightarrow C_B^+\text{ primal}
\rightarrow \text{exact signed pool}(X/Y)
\rightarrow \text{rank4 residual}
\rightarrow \text{carrier12+rank4}.
$$

`ProgramNativePrimalScorer` 按 family 共享上下文网络、按固定 LoRA owner 使用 native-width heads；任务依赖只来自 Program，没有 task/video/member lookup。

`PrimalDualVideoOperator` 则只用当前视频的真实 native covariance，把 primal 确定性转换为 dual，并以统一 score RMS重放真实候选。P0已经验证 IEEE FP32、chunk不变性、K4置换、唯一76 tensor rank16和真实policy consumption；P1进一步验证跨视频共享 primal 在六任务、四family、浅中深target上的held能力。

因此以下说法已经没有证据支持：

* “q/v native bank本身没有方向”；
* “rank4在当前接口上必然太小”；
* “current-bank global dual无法稳定”；
* “B1 exact signed replay是主要误差源”；
* “还需要恢复full polar”。

## 2.4 Behavior authority是有价值的，但不是 set-valued policy authority

当前 behavior authority 对每个 task、每个 selected target保存两组 disjoint 256-row cross-episode flow-gradient rank4 factors，随后将 panel A/B 的更新平均并重新做rank4截断，得到一个 consensus。

这能回答：

* task-specific局部下降方向是否存在；
  -方向是否跨panel重复；
  -跨task是否有低秩几何。

它不能自动回答：

* 所有successful policy是否应映射到同一个方向；
  -这个局部一阶方向是否等同于最终闭环最优LoRA；
  -两个不同successful members是否应在参数或梯度空间被平均。

所以 behavior authority适合作为正控、诊断和辅助 Gate，不适合强迫部署 Program 本身成为它的等距坐标系。

## 2.5 V5 的 Program feature约束比摘要中表现得更强

`program_behavior_features()` 对以下六个block分别做单位化，然后固定等权拼接：

$$
P_{\rm lang},\ P_{\rm scene},\
\sqrt{\rho}P_{\rm process},\
\sqrt{\rho}\sigma,\
\rho,\ \tau.
$$

最后再对整体单位化。也就是说，幅度信息被全部删除，六个异质block被假定具有相同几何权重。更重要的是，`rho` 和 `tau` 没有owner轴，代码把它们复制到八个selected targets中；因此每个target的Program Gram至少有三分之一来自完全相同的全局event geometry，而八个teacher target kernel并不相同。

这是一项关于 **V5 feature map** 的具体结构问题，不等于原始 Program schema有问题。

## 2.6 “macro4—5平台”不是严格同数据收敛证据

Natural Program schedule中：

* 每macro固定使用15个target-fit；
* 从meta-fit中轮换15个；
* K、video demos、action demos和第二组disjoint view都随macro变化。

因此macro4和macro5的`12.61899/12.61955`是两个不同task/view样本集合上的macro平均，不是同一固定objective batch的严格平台。

它仍然是“不再出现明显宏观改善”的迹象，但不能单独承担“已优化到结构上限”的结论。

## 2.7 当前代码已经使联合Program训练在执行上可行

current-bank covariance采用纯frame quadrature，不依赖Program或canonical assignment；X/Y capture也只依赖冻结policy、frames、language prefix和fixed probes。Program的event字段只在 primal scorer中形成primal。因此，联合更新 Program时：

* 原始 X/Y；
* output四类bank所需边界；
* covariance eigensystem；
* action query batch；

都可以缓存而无需每步重新运行冻结PI0.5。

当前唯一阻碍是 `shared_compiler_data.py` 为 frozen-G2 P2服务：它在 `torch.inference_mode()` 下运行Program，并通过 `_ordinary(...detach().clone())`切断梯度。联合训练需要重构这个数据接口，而不是重写operator。

---

# 3. 当前根因：哪些已证明，哪些只是推测

## 3.1 已经由证据证明

| 问题                                         | 状态   | 依据                                                                   |
| ------------------------------------------ | ---- | -------------------------------------------------------------------- |
| 真实native bank是否有方向容量                       | 已通过  | G1、P1、analytic reference                                             |
| current-bank covariance dual是否必要且可用        | 已通过  | bank-independent dual弱，current-bank primal→dual约`.90+`，P1 held约`.95` |
| rank4是否足以做当前机制测试                           | 已通过  | P1四family、浅中深均高                                                      |
| exact signed replay是否正确                    | 已通过  | P0 chunk/K4/IEEE和P1                                                  |
| task-specific behavior是否真实存在               | 已通过  | universal约`.19`，rank16 held oracle约`.716/.801`                       |
| 冻结旧G2 Program是否已有简单可迁移behavior坐标           | 明确没有 | fit reader约`.97–.98`、task-held只有`.20–.27`                            |
| V3 role-local Gram是否足够                     | 已淘汰  | 局部高相关，全局低，监督图断开                                                      |
| V4 connected Gram是否足够                      | 已淘汰  | 图已连通但全局geometry不升                                                    |
| V5 raw globally calibrated Gram是否在macro5通过 | 明确没有 | train、exact、wrong margin全部失败                                         |
| 当前问题是否只是held泛化                             | 否    | 多轮fit、held-video和task-held长期一起低                                      |

相关完整历史记录位于43—51节。

## 3.2 最合理但尚未最终证明的根因

目前最窄、最可靠的根因表述是：

> **冻结Stage0 evidence经过Natural Program压缩，再由共享Program→primal mapping解释时，没有得到对未见task可迁移的功能credit。现有实验尚不能把这个联合失败唯一拆成“Program信息不足”或“decoder/loss不可达”。**

这比“G2 Program失败”更精确，也比“P2训练不够”更有解释力：

* P2冻结Program，所以无法让上游表示配合primal任务；
* V3—V5只更新Program，却要求它直接匹配一个人为固定的cosine kernel，没有让实际primal/LoRA反馈告诉Program保留什么；
* 两端分别优化都低，不代表两端联合优化仍低。

## 3.3 仍未实验回答的问题

### Stage0是否没有保留足够behavior信息

**未证明。**

旧G2动态Gate、event、same-task、K1/K4均通过，说明Stage0至少保留了相当多的视频动态；但仓库没有一个严格matched的：

$$
\text{raw Stage0 evidence}
\rightarrow \text{same-capacity shared primal scorer}
\rightarrow \text{task-held functional Gate}
$$

正控。因此不能排除Program压缩丢失信息，也不能排除Stage0本身缺少所需behavior变量。

### Program schema是否不合适

**未证明。**

当前schema很宽，包含owner-specific language/scene、8×38 process和sigma，以及rho/tau。失败的是：

* pointwise decoder；
* block-equal Gram；
  -冻结Program→primal短训练。

尚未测试该schema在真实functional loss下与primal scorer联合适配。

### Behavior authority是否与部署目标不一致

**部分成立。**

它是有用的一阶局部行为信号，但不是完整闭环成功集合，也不是唯一policy等价类。它应降级为辅助诊断，而不应继续作为Program必须机械复现的全局坐标。

### Meta/source-seen 与 target/source-unseen混合是否导致不可识别

**没有主要证据。**

角色权重是显式平衡的，V4也接入了cross-role关系；behavior manifold对held有明确上限。internal target高值只有少量task，不能据此证明target role已解决，但也没有证据表明role mixing是首因。

### 数据量和任务多样性是否不足

**当前不是最早接口。**

60或75 fit task建立的behavior basis对held有 `.62–.80` 上限，说明现有数据足以支持一次显著强于language/universal的共享机制测试。未来达到validation闭环可能仍需更多多样性，但现在先增加任务不会修复错误credit。

## 3.4 根因优先级

1. **Program与primal scorer被人为分开训练，且缺少generated-LoRA functional credit。**
2. **V3—V5监督对象坐标不匹配：Program Gram被要求等距复现局部behavior kernel。**
3. **V5存在优化时标和梯度分配混杂，但这不是继续V5的充分理由。**
4. **Program schema可能压缩错误，尚需joint-vs-raw Stage0对照。**
5. **Stage0可能缺少behavior信息，当前没有直接证据。**
6. **role mixture或任务数量不足，目前证据最弱。**

---

# 4. 对V5训练规模、gradient clip和停止判断的独立审计

## 4.1 15 updates不足以证明一般结构性失败

V5 formal每macro有3次optimizer update，`warmup_macros=3`，因此：

* 总共15 updates；
* 前9 updates位于线性warmup；
* 只有最后6 updates处于峰值附近或之后。

这与旧G2 temporal readout约200 updates才展开、pointwise behavior decoder训练到macro60的历史时标并不对称。

所以，V5不能支持以下强结论：

> “当前Program模型无论怎样合理优化，都无法形成behavior相关表示。”

它只支持：

> “从`c1493a1/macro20`初始化、采用现有混合loss、clip=1和15-update预注册schedule的V5，没有在决定节点形成所需geometry。”

## 4.2 但不应据此续训V5

继续V5仍然低信息量，因为它要求一个没有必要性的中间性质：

$$
\cos(z_\tau,z_{\tau'})
\approx
\frac{1+\cos(g_\tau,g_{\tau'})}{2}.
$$

实际下游只需要存在某个共享函数：

$$
d_\tau=f_\theta(P_\tau),
$$

使 \(d_\tau\) 经当前bank dualization和真实X/Y replay后产生有用LoRA。只要 \(f_\theta\) 是非线性的，Program无需与behavior kernel等距，也不必线性可读。

因此，V5即使再训练后上升，也最多说明“Program可被迫扭成这个kernel”，并不能证明这是最自然或最可迁移的Writer接口。

## 4.3 “所有梯度被clip到1”不能直接解释失败

当前是对所有Program参数做一次全局 `clip_grad_norm_`。这会给总梯度乘以一个公共标量，保留总梯度方向。对AdamW而言，梯度整体缩放还会在一、二阶矩中部分抵消，所以不能从`7.4–13.7 → 1`直接推出“有效更新太小”。

真正需要知道的是：

$$
g_{\rm total}=g_{\rm dynamic}+g_{\rm behavior}
$$

中：

* 两者的范数；
* 参数组内cosine；
* behavior梯度被dynamic梯度抵消的比例；
* clip后的实际Adam parameter delta；
  -该delta对全train60目标的有限差分作用。

当前代码没有记录这些量。它只对language key、scene key和process-fusion第一层三个tensor单独调用一次 `autograd.grad(kernel_objective)`；随后名为 `behavior_program_gradient_norm_before_clip` 的量实际上是在总loss backward后读取的process-fusion总梯度，不是纯behavior梯度。

所以：

* 没有依据直接增大clip；
* 没有依据直接增大behavior loss权重；
* 也没有依据声称clip就是根因。

## 4.4 对停止判断的裁决

我的独立裁决是：

* **停止V5作为active路线：正确。**
* **把V5称为该具体objective/schedule的scientific non-pass：正确。**
* **把它上升为Program schema或Stage0的结构性non-pass：不成立。**
* **把macro4—5称为严格收敛平台：证据偏强。**

为了把历史归因做得更严谨，可以在V5初始化与macro5 checkpoint上做一次无optimizer-step的forensic gradient decomposition：

1. 全train60固定条件累计纯behavior梯度；
2. 单独累计旧dynamic梯度；
3. 记录每个Program模块的norm与cosine；
4. 用真实Adam state构造一次virtual update；
5. 比较virtual behavior-only、dynamic-only和combined update对全train60 raw-Gram误差、correlation及exact readout的影响。

若behavior-only virtual step本身只扩大spread、仍不提高teacher correlation/exact readout，则V5目标可被更强地正式关闭；若behavior-only明显改善而combined step为负，才证明存在明确gradient allocation冲突。这个诊断不应再触发V6，也不影响下一主路线。

---

# 5. 推荐的唯一主路线：Joint Program–Primal Functional Qualification

## 5.1 完整数据流

```text
exact language + ordered action-hidden videos
        │
        ▼
frozen source PI0.5 + frozen Native Stage0
        │
        ├── detached language embeddings / patch evidence
        ├── local process / presence / uncertainty / temporal evidence
        └── frozen native X/Y banks + current-bank covariance eigensystems
        │
        ▼
trainable Natural Program
P_lang / P_scene / P_process / rho / tau / sigma
        │
        ▼
trainable ProgramNativePrimalScorer
target × rank × event → native input/output primals
        │
        ▼
frozen current-bank primal-to-dual operator
q_B = C_B⁺ d
        │
        ▼
exact positive-minus-negative softmax over current real X/Y
        │
        ▼
rank4 residual + frozen carrier12
        │
        ▼
one complete 38-target rank16 LoRA
        │
        ▼
frozen PI0.5 on disjoint cross-episode action/flow panels
        │
        ▼
functional gradient → Program + primal scorer
```

## 5.2 训练与冻结关系

初版只训练：

* `OwnerLanguageReader`；
* `OwnerSceneReader`；
* `process_fusion`；
* canonical aligner与Program aggregation相关参数；
* `ProgramNativePrimalScorer`全部参数。

继续冻结：

* source PI0.5；
* Native Stage0 encoder；
* current-bank covariance/eigensystem；
* exact replay operator；
* carrier12；
* scale prior和scale head；
* Action Meta；
  -旧training-only temporal/behavior decoder。

Program schema保持不变。current-bank primal-to-dual保持不变。

## 5.3 最小训练目标

对同一task，使用两组disjoint video views \(V_a,V_b\) 和与它们均不重合的action-query panels。每个video view生成自己的唯一LoRA：

$$
L_a=W_\theta(V_a,\text{lang}),\qquad
L_b=W_\theta(V_b,\text{lang}).
$$

主loss直接作用在generated policy：

$$
L_{\rm func}
=
\frac12
\sum_{v\in\{a,b\}}
\left[
-\eta\log
\sum_m w_m
\exp\left(
-\frac{
\ell_{\rm flow}(\pi_{L_v},Q_m)
}{\eta}
\right)
\right].
$$

其中 \(m\) 只在确有多个完整successful policy members时表示member；每个member的loss必须覆盖其完整state/action panel，不能逐state拼接不同members。只有普通demonstration episodes时，对disjoint episodes取task-equal平均即可。

不加入：

* Program Gram；
* behavior-coordinate regression；
* factor reconstruction；
  -input/output subspace loss；
* polar或dual reconstruction；
  -旧十五项探索loss的永久叠加。

当前 `functional_lora_loss_gradient()` 已能在冻结policy下求LoRA leaf梯度，并明确支持把这些leaf梯度按链式法则反传到Writer。联合训练只需把返回的detached梯度与当前generated LoRA做内积surrogate，再对Program与primal scorer backward。

## 5.4 为什么这比恢复P2更有信息

P2冻结Program，只问“旧坐标能否被primal scorer读出”；V5固定primal scorer不存在，只问“Program能否自行变成某个kernel”。

Joint qualification直接问真正需要的问题：

$$
\exists\ \theta_P,\theta_D:
\quad
\pi_{W(P_{\theta_P}(V),D_{\theta_D})}
\text{ 是否改善跨episode行为？}
$$

同时，由于bank operator、scale、carrier和Stage0仍冻结，若失败，仍然可以清晰区分：

* train fit不可达；
* video泛化失败；
* task泛化失败；
* Program压缩失败；
* Stage0 evidence不足。

## 5.5 防止shortcut

不靠增加训练loss，而靠严格controls：

* deployment forward无task ID、filename、teacher factors或actions；
* video/action episode完全错开；
* language-only；
* first+final/endpoints；
* correct Program + correct bank；
* wrong Program + correct bank；
* correct Program + wrong bank；
* wrong Program + wrong bank；
* leave-one-task-out universal；
* same-task-other video；
* meta/target分开报告。

建议继续使用交叉交互量：

$$
I=
R(P_c,B_c)-R(P_w,B_c)-R(P_c,B_w)+R(P_w,B_w).
$$

这样可以区分language/task lookup、bank-common residual和真实Program×video组合。

---

# 6. 两个决定性实验

## 实验一：12-task Joint Program–Primal Functional Gate

### 数据

使用现有P2 authority，不创建新task。建议固定：

* gradient meta：`[1, 8, 9, 32, 52]`
* gradient target：`[72, 73, 75, 93, 94]`
* true task-held meta：`2`
* true task-held target：`74`

这些均来自当前40/10 mapping split；task-held两项不参与任何Program、primal或checkpoint梯度。现有split与完整条件数已在v5配置中冻结。

每个gradient task使用：

* 两条mapping-fit K1 videos作为两个独立views；
* 第三条预注册held video；
* behavior/action panel A用于训练；
* disjoint panel B用于评价；
  -selected targets继续使用浅中深q/v与action-in/out八target；
  -完整38-target LoRA仍实际生成。

### 正控

在完全相同的functional loss上，先为每个task优化一个跨两条fit video共享的task-local native primal，第三条video零梯度。

这不是部署候选，只用于确认当前action/flow objective确实在current operator中有可达下降方向。要求：

* held-video functional recovery median `>=0.80`；
* q/v/action-in/action-out各family `>=0.70`；
* 每个task均显著优于carrier。

若这个正控失败，问题是functional panel、scale或teacher-to-utility，而不是Program。

### 正式训练

初始化：

* Natural Program从`c1493a1/macro20` model tensors；
* primal scorer完全fresh；
* optimizer/scheduler fresh；
  -不使用V3/V4/V5 checkpoint。

100个optimizer updates，warmup最多10步；预注册checkpoint 60与100。每步保持meta/target等权，并轮换video/action panels。

### 速度资格

利用现有冻结evidence和current-bank operator cache：

* 6卡global update目标不超过30秒，硬上限45秒；
* 每卡peak reserved低于35GiB；
* 100步训练墙钟应与约100次实际功能更新相称，而不是由重复冻结policy/native capture主导；
* 12-task完整评价不应长于训练主体的一半。

若超过，应先缓存：

* language embeddings；
* patch/local Stage0 evidence；
* raw X/Y；
* covariance eigensystem；
  -固定action batches。

不得缓存Program输出或generated LoRA。

### Gate

定义task-local正控归一化功能恢复：

$$
R_{\rm func}
=
\frac{
L_{\rm carrier}-L_{\rm shared}
}{
L_{\rm carrier}-L_{\rm free\ primal}
}.
$$

要求：

* gradient-task train median `>=0.60`；
* same-task held-video median `>=0.50`；
* 两个true task-held平均 `>=0.40`，且任何一个不低于`.30`；
* held/train ratio `>=0.80`；
* q、v family各 `>=0.35`；
* action-in、action-out各 `>=0.30`；
* full相对language-only与endpoints各 `>=0.10`；
* correct-vs-wrong Program margin `>=0.10`；
* correct-vs-wrong bank margin `>=0.10`；
  -交叉interaction \(I\ge0.05\)；
* same-task-other retention `>=0.80`；
  -checkpoint 60到100无超过`.05`的task-median回落；
* event不坍缩、K1 identity和信息墙继续通过。

### 结果解释

| 结果                                | 结论                                                  |
| --------------------------------- | --------------------------------------------------- |
| task-local正控低                     | behavior/action authority、scale或functional panel有问题 |
| 正控高，joint train也低                 | 当前Program+primal函数类或功能credit不可优化                    |
| train高、held-video低                | same-task video invariance失败                        |
| train和video-held高、true task-held低 | 跨task表示失败；进入实验二的raw-Stage0分支                        |
| absolute高、wrong Program/bank也高    | language/universal/common-bank shortcut             |
| 12-task全部通过                       | 允许进入实验二的完整shared分支                                  |

---

## 实验二：条件式唯一分支

### 分支A：实验一通过——完整shared functional qualification

恢复当前预注册的：

* 40 task fit；
* 10 task holdout；
* 329 fit conditions；
* 40 video-held；
* 82 task-held；
* 40-task Program/bank因果panel。

仍联合训练Program+primal scorer，使用panel A functional loss；panel B、video-held、task-held只读。

这里应修改旧F3 Gate：

* generated-LoRA的cross-episode functional recovery成为primary；
  -旧native-teacher update cosine降为secondary geometry diagnostic；
  -不能要求一个可能功能等价但参数不同的LoRA机械复现teacher factor。

建议primary要求：

* held-video \(R_{\rm func}\) median `>=0.60`、p10 `>=0.35`；
* task-holdout median `>=0.40`；
* held/train `>=0.80`；
* meta/target两个role分别不低于`.35`；
* q/v各不低于`.35`；
* Program、bank margin各`>=.10`；
* interaction `>=.05`；
  -多数task为正；
  -两个相邻checkpoint稳定。

通过后才进入scale、K2/K4和closed-loop。

### 分支B：实验一train高、task-held低——raw Stage0 sufficiency probe

保持同一12-task split、相同primal scorer容量、相同functional loss与训练预算，只把输入从固定Program schema换成部署可见的raw frozen Stage0 evidence：

* owner/event process；
* presence；
* uncertainty；
* patch/scene transition；
* exact-language embedding；
* normalized time。

该probe仅作诊断，不登记为deployment Writer。

判定：

* raw Stage0 task-held比Program高至少`.15`，且达到`.40`：Program schema、aggregation或压缩是最早接口；
* raw Stage0也低于`.25`，而task-local primal仍高：frozen Stage0是最早上游瓶颈；
* raw和Program都能fit但都不能held：task diversity、authority或共享decoder归纳偏置有问题；
* raw train本身也低：不是表示压缩，而是joint functional optimization或decoder函数类问题。

只有raw Stage0显著通过，才允许重新设计Program schema；只有raw Stage0也明确失败，才允许解冻Stage0最末端的process/presence/uncertainty投影。不得直接解冻VLM、source policy或整个Stage0。

---

# 7. 对当前G2→G3→G4→Final计划的具体修订

## 7.1 G2 behavior qualification不应继续作为恢复G3前的独立硬Gate

建议将当前：

```text
G2 Program behavior-Gram pass
→ frozen-Program P2
→ F4
```

修改为：

```text
G2 temporal/dynamic mechanism evidence（已保留）
→ P0/P1 current-bank operator capacity（已通过）
→ J2 joint Program-primal functional qualification
→ full shared functional/causal qualification
→ scale/preservation
→ K2/K4
→ held5 closed-loop
→ G4/Final
```

理由是：现在需要验证的恰恰是Program与primal scorer是否必须共同形成内部功能分化。如果继续把Program独立Gate设为前置条件，就把尚未证明必要的内部坐标性质当成项目合同。

## 7.2 P2的地位

现有frozen-Program P2应保留为：

-历史诊断；

* joint模型的冻结Program ablation；
  -证明旧Program坐标不易读出的对照。

不再作为恢复后续阶段的必经训练课程，也不应从现有checkpoint续训。

## 7.3 F4调整

Joint qualification已经使用真实functional loss，因此旧F4不再需要重新“恢复functional职责”。F4应收窄为：

* scale/spectrum；
* confidence；
* carrier preservation；
* unrelated/source-state guardrail。

selection/primal职责必须由joint Gate保护，scale与selection仍需gradient ownership分离。

## 7.4 F5、F6顺序保留

方向取得后再恢复：

1. K2；
2. K4；
3. uniform初始化的bounded video correction；
4. same-task video robustness；
5. held5 strict250。

这是合理顺序。多视频职责不应在K1 task-specific selection尚未成立时混入。

## 7.5 G4更早联合，但不是直接跳Final

J2已经是一个受控的局部联合训练：

* source和Stage0冻结；
* operator冻结；
* scale冻结；
  -只联合Program与primal scorer；
* task/video/action split严格。

它不会掩盖P1或Stage0，因为下游已固定且有task-local正控。通过后，G4才扩展为：

-全部75个授权fit tasks；

* scale/preservation；
  -更多cross-episode states；
  -必要时student-visited states和short continuation；
* later on-policy success/progress。

## 7.6 应删除而不是继续叠加的内容

从active训练面删除：

* pointwise behavior decoder；
* V3 centered Gram loss；
* V4 connected Gram loss；
* V5 raw lifted Gram loss；
* Program必须线性/kernel可读的硬Gate；
  -旧subspace、polar、factor reconstruction warmups；
  -针对已淘汰chart的candidate losses。

继续保留为只读诊断：

* behavior panel A/B；
* rank16 behavior oracle；
* universal baseline；
* block geometry；
* Program/kernel ridge readout；
* P1 task-local primal codes；
* correct/wrong Program/bank controls。

---

# 8. 对Final随机初始化联合训练、简化loss和Action Meta的意见

## 8.1 整套Writer随机fresh必须保留

owner要求合理，而且当前历史进一步支持这一点：多次分段训练都可能人为固定一个下游真正不需要的中间坐标。

Final至少应有两个matched候选：

### 组件初始化

* Stage0、Program和primal scorer从通过Gate的组件初始化；
* fresh optimizer/scheduler；
  -联合端到端训练。

### 完全随机Writer

随机初始化全部Writer组件，包括：

* Native observer的trainable Writer部分；
* language/scene/process Program；
* alignment；
* primal scorer；
* scale/confidence。

仍冻结：

* source PI0.5；
  -原始38 target weights；
  -固定carrier authority。

两者架构、参数量、数据、loss、训练节点和closed-loop选择必须一致。仓库总合同也明确G1—G3只作机制验证，不是Final必须照搬的课程。

## 8.2 Final loss应保持简洁

建议Final核心只保留：

$$
L_{\rm Final}
=
L_{\rm cross\text{-}episode\ action/flow}
+
\lambda_{\rm preserve}L_{\rm preservation}
+
\lambda_{\rm outer}L_{\rm success/progress}.
$$

其中：

* 第一项负责形成初始LoRA；
* preservation只防止破坏已有能力；
* outer success/progress只有base Writer已有闭环增量后才启用。

Program Gram、behavior coordinates、factor cosine、subspace、polar、reconstruction都不应成为永久项。

## 8.3 Action Meta

继续默认关闭。

只有base joint Writer已经满足：

* full明显高于carrier；
* full高于language/endpoints；
* breadth成立；
* Goal/Long至少有真实贡献；
* same-task视频稳定；

才做一次matched Action Meta control。对照必须：

-相同Writer checkpoint；
-相同数据和optimizer节点；
-相同唯一rank16；
-仅Action Meta on/off变化。

只有连续checkpoint均显示净收益，而且breadth、retention、Goal/Long和video causality均不下降，才保留。

## 8.4 最终裁决仍只能来自closed-loop

内部functional Gate只能决定是否值得进入closed-loop。正式方法仍必须达到：

* validation8 strict correct `>145/400`；
  -相邻single checkpoint稳定；
  -低churn；
  -四suite非零；
* Goal/Long贡献；
* same-task不同视频鲁棒；
  -选定checkpoint后才运行shuffled/reversed；
* full显著优于language、endpoints和wrong video。

任何Program/kernel、behavior cosine或P1 recovery都不能替代这些要求。

---

# 9. 需要修改的具体文件、模块和接口

## 9.1 `src/ember/ecp/natural_program.py`

将当前单体forward拆为两个接口：

```python
encode_frozen_evidence(...)
    -> FrozenProgramEvidence

compile_program(
    evidence: FrozenProgramEvidence,
    video_set_offsets,
    query_times,
) -> NaturalProgramOutput
```

前者在inference mode下运行冻结source/Stage0，输出：

* language token embeddings；
* patch states；
* local process/presence/uncertainty；
* local temporal moments；
* frame posterior和必要mask。

后者保持可微，拥有：

* language reader；
* scene reader；
* process fusion；
* alignment；
  -aggregation；
* Program schema。

这样可以缓存冻结证据而不切断Program梯度。

## 9.2 `src/ember/ecp/shared_compiler_data.py`

当前 `_run_frozen_pass_a()` 和 `_ordinary()` 专门服务冻结P2。应保留为eval路径，同时增加：

```python
prepare_joint_program_primal_condition(...)
```

它读取缓存的：

* frozen Program evidence；
* raw X/Y；
* final outputs；
* covariance operators；
* action-query panels；

但每步重新计算可微Program和primal。

## 9.3 `src/ember/ecp/bank_conditioning/program_primal.py`

核心架构可以保留。建议增加：

* condition batch API；
  -可选的Program block attribution metrics；
  -每family、owner、rank的primal norm和gradient监控。

不要加入task embedding、behavior coordinate input或teacher-factor input。

## 9.4 `src/ember/ecp/bank_conditioning/primal_dual.py` 与 `primal_dual_runtime.py`

数学不改。

只需保证：

* cached spectral operator完全detached；
* primal输入保留梯度；
* `dual_and_score_rms()`对primal可微；
* exact replay对query可微；
* IEEE scope仍在每次调用后恢复；
* compact cache不保存Program、task-specific learned state或teacher。

## 9.5 `src/ember/ecp/shared_compiler.py`

增加联合训练入口，例如：

```python
forward_compact_from_program(
    program: NaturalProgram,
    videos: Sequence[CompactPrimalDualVideo],
    s_ref: Tensor,
) -> SharedCompilerOutput
```

现有 `forward_compact()` 已接近该接口，主要问题不在compiler，而在上游Program被detach。

## 9.6 `src/ember/writer/functional.py`

复用现有 `functional_lora_loss_gradient()`，新增一个显式工具：

```python
def writer_chain_rule_surrogate(
    generated_state,
    detached_leaf_gradients,
) -> Tensor:
    return sum(
        (generated_state[name] * detached_leaf_gradients[name]).sum()
        for name in generated_state
    )
```

并测试其梯度与直接小模型functional call一致。

## 9.7 新增唯一joint train owner

建议新增一个明确职责文件，而不是继续扩展G2或P2旧train step：

```text
src/ember/ecp/joint_program_primal_training.py
src/ember/ecp/joint_program_primal_train_step.py
src/ember/ecp/joint_program_primal_gate.py
configs/pi05_ecp_joint_program_primal_j2_v1.json
```

它只拥有：

* task/video/action sampling；
* functional leaf-gradient；
* Program+primal optimizer；
* causal/full-vs-controls Gate；
* cache provenance。

## 9.8 `src/ember/ecp/behavior/kernel.py`

从active optimizer路径移除。可保留：

* `program_behavior_features`用于只读block分析；
* behavior Gram和kernel ridge用于诊断；
* universal、panel A/B、consensus作为辅助报告。

不再由 `natural_program_train_step.py` 调用 `distributed_behavior_kernel_loss()`。

## 9.9 `src/ember/ecp/natural_program_train_step.py`

V5训练路径退役后，应恢复到历史G2 dynamic owner，或停止作为active入口。不要在同一个train step中继续混合：

* 旧15项dynamic loss；
* behavior Gram；
* future functional loss。

Joint functional training应由新的owner文件负责。

## 9.10 文档

需要同步改写：

* `progress.md`：登记新joint路线；
* `task_plan.md`：删除behavior-Gram硬前置；
* `docs/event_conditioned_policy_compiler_design.md`：将P0/P1之后的active序列改成J2；
* `docs/concept.md`：更新旧B0-anchor表述与当前primal-dual数据流；
* `README.md`：说明V5 behavior Gram已退役、current active是joint Program-primal待实现。

---

# 10. 哪些结果会构成真正的路线停止证据

## 10.1 已经可以停止的路线

以下路线已有足够停止证据：

* full per-bank functional-polar deployment；
* fixed native-Q sketch；
  -旧candidate-chart set-summary；
* direct-native raw-query transfer；
* pointwise behavior decoder；
* V3/V4/V5直接Program-Gram训练；
* frozen旧Program P2作为必经课程。

## 10.2 停止当前Program schema所需的证据

只有同时出现以下结果，才足以判定当前schema是首因：

1. task-local free-primal functional正控高；
2. joint Program+primal能在gradient tasks上明显fit；
3. same-task held-video也高；
4. true task-held仍明显低；
   5.同容量raw Stage0 probe在task-held上比Program高至少`.15`。

缺少第5项时，只能说“当前joint representation/generalization失败”，不能指认schema。

## 10.3 停止frozen Stage0所需的证据

需要：

1. task-local primal和behavior basis仍强；
2. joint Program方案失败于task-held；
3. raw Stage0同容量probe也不超过约`.25`；
4. fit本身可优化，排除训练图错误；
5. family分解确认q/v/action都缺少可读信息。

满足后，才允许测试Stage0最末端窄解冻。整个VLM或source policy仍不应解冻。

## 10.4 停止Program→primal函数类所需的证据

需要在功能正控强的同一数据上：

* 至少约100个有效post-warmup joint updates；
  -纯functional gradient确实到达Program和primal scorer；
* train normalized recovery仍低于`.40`；
  -不同video views同向低；
  -没有global clip/gradient conflict的未决混杂；
  -增加非线性容量并非唯一缺失接口。

当前P2的25步、V5的15步均达不到这个停止标准。

## 10.5 停止behavior authority作为训练信号所需的证据

若模型显著提高：

* panel A/B factor cosine；
  -behavior kernel readout；
* consensus reconstruction；

却不改善独立cross-episode flow loss，或者改善flow loss却不改善held5 closed-loop，则该authority只能保留为诊断，不能继续作为primary teacher。

V5已经提供了一部分这种警示，但因为它没有经过generated-LoRA functional链，尚不是最终裁决。

## 10.6 停止current primal-to-dual operator所需的证据

当前没有停止依据。只有在扩展到独立任务和真实functional目标后，P1式task-local shared-across-video primal正控在多个family普遍跌破约`.70`，才应重开operator。现有`.94–.995`结果与这种判断相反。

## 10.7 真正的全路线停止证据

需要同时具备：

* task-local primal functional上限强；
* raw Stage0可读性强；
* component-init joint Writer和全随机fresh joint Writer都经过足够优化；
  -两者都不能形成train/held functional增量；
  -或内部functional增量始终不能转化为held5及validation8 closed-loop；
* correct、same-task、wrong、language、endpoints controls均无法证明视频必要Value；
  -结果在相邻checkpoint和多个task上稳定，不是单点波动。

在到达这些证据之前，当前负结果只支持停止错误的中间监督和分段课程，不支持停止EMBER的核心研究命题。

## 需要未跟踪artifact才能进一步裁决的部分

本次主架构判断不依赖远程不存在的raw artifacts。唯一仍需原始运行文件的，是对“V5究竟优化不足还是gradient冲突”的精确归因。需要：

* V5 run的 `metrics.jsonl`；
* macro5 optimizer/scheduler state；
* `pi05_ecp_g2_behavior_block_geometry_v5...json`；
* behavior authority的panel A/B factor tensors及member/episode provenance；
  -重新计算并保存全模型、分objective的per-module gradient norm/cosine；
  -真实Adam parameter delta；
  -virtual step前后全train60 raw Gram、correlation和exact readout。

这些字段当前tracked文档与train-step日志并未完整提供，因此不能从“preclip norm都大于1”推断clip就是根因。
