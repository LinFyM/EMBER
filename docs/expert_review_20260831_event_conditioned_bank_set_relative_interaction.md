# 1. 一页以内的核心结论

我已核验两个对象均可从远程访问：

* canonical：`main@92617d070e3a573f640875b9f5bd355c162177d7`，提交说明为记录本轮 G3 candidate-interaction 咨询边界；
* 诊断分支：`codex/g3-vector-interaction@2295f481dcd284e4bae92afeaf2cf5c4b2d3e5c2`，提交只增加 vector-valued interaction 及对应测试，没有合并到 canonical。

## EMBER 现在卡在哪里

当前已经找到一个**足够具体、足以停止现有实现家族的直接瓶颈**：

> 现有 interaction 虽然逐 candidate 读取了 Program event、当前视频 local context、native value、base score 等信息，但 correction 仍是一个对每个 candidate 独立应用的 pointwise 函数；除全局 mean、covariance-derived base query 外，它没有读取“当前整个 candidate set 在 Program-relative 坐标中的分布”。
> 因此，同一套参数很难同时学会：在 correct bank 上保持 R5 的强方向，在 wrong bank 上仅施加小而选择性的修正。

这不是“bank 没有信息”。vector task-local 诊断已经表明，模型能读取并跨视频泛化 bank 差异。也不是 signed pooling、correction bound 或 rank4 没有容量；free-delta 只需约 `.002` 的 p95 correction 就能压低 wrong bank。真正失败的是：**当前共享、set-independent 的 local chart 没有形成保留 correct、压低 wrong 的第三个选择性参数方向。** V2–V4 中 correct/wrong 梯度 cosine 约 `-.96`，正是该局部函数空间冲突的直接表现。

## 是否已经找到“最终根因”

**找到了当前函数类的根因，但还没有找到 EMBER 的最终根因。**

已经可以停止的是：

* main 上的 scalar/base-score pointwise MLP；
* vector 分支上的 32D、family×side 共享投影、逐元素乘积、仍然 pointwise 的扩展；
* 在这两个函数类内部继续扫描 loss weight、width、bound、temperature、LR、seed。

尚未被淘汰的是：

* 使用整个 native candidate set 的 event-conditioned、set-conditioned continuous interaction；
* exact positive-minus-negative pooling；
* 真实 X/Y 作为必要 Value；
* rank4 residual、唯一 rank16；
* full-inverse 作为 zero-init capacity anchor；
* Natural Program、Stage0 和整个 Native-Factor/ECP 命题。

## 是否应停止 binary 或 pointwise 路线

binary full/half 已经应该永久退出 canonical。当前新的 scalar 和 vector local pointwise 路线，也已到达我上次 §9.3 为“停止当前 native candidate interaction”规定的证据边界：correct/wrong 要么共同保持、要么以牺牲 correct 换取 wrong suppression；模块确实读到 bank，基础执行合同也没有明显故障。

但两任务结果仍不是“所有可能的 pointwise 网络均数学不可能”的定理。它只足以作工程和科研路线裁决：**不再沿这个家族创建 V5。**

## 唯一推荐主线

我建议实现：

> **Event-Conditioned Bank-Set Relative Interaction**
> 先把当前 native candidate set 映射到由 Program native event queries 定义的、basis-invariant 的相对坐标；在 B0 中按 event 流式累计 set mean、dispersion 和少量 induced-attention summaries；在 B1 中让这些 whole-bank summaries 通过 FiLM 调制每个 candidate 的 continuous correction，随后仍由一套 exact signed measure 对真实 X/Y pooling。

它不做 task matching，不输出类别，不选择两套 operator，也不直接生成因子。bank 不再只是 base query 的坐标系或一个 gate，而是**改变整个 candidate correction 函数的条件状态**。

我的主观概率判断是：

* 该方案通过 fixed-route、same-task/wrong-bank 机制 Gate：约 **35%–50%**；
* 最终经 Natural Program、task holdout、held5 到 validation8 `>145/400`：约 **10%–20%**。

最大风险是：correct 与 wrong LIBERO banks 在 Program-relative set distribution 上仍过于相似；届时 set summary 也可能只学到 task/scene fingerprint，而不能学到可迁移的过程—政策对应关系。

---

# 2. 对远程代码、文档与结果的复核

## 2.1 你们正确执行了上一次方案

main 当前实现与第五次专家建议基本一致：

1. 保留 R5 full-inverse base path；
2. `ProgramNativePrimalScorer`同时产生聚合后的 base primal 和未聚合的 `rank_event`、event weights、event-specific native queries；
3. `ProgramBankInteractionScorer`读取当前视频的 canonical assignment、frame position、local scene/process/presence/tau/sigma 和真实 candidate；
4. correction 在 exact signed pooling 前进入两个 branch；
5. 每条视频独立执行，最后 uniform 聚合；
6. 只产生一套 rank4 residual 和一套完整 rank16。

main scorer 当前实际读取的逐 candidate 特征是：

$$
\begin{aligned}
&\text{native scalar alignment},\\
&\text{Program/local semantic scalar alignment},\\
&\text{两者乘积},\\
&\text{detached base signed score},\\
&\log \|v_n-\mu_B\|,\\
&\text{frame/probe/horizon/type metadata}.
\end{aligned}
$$

然后 family-shared MLP 输出 bounded correction。它确实读取 global mean 和由 covariance 形成的 base query，因此不能简单称为“完全没有 global 信息”；更准确的说法是：

> 它缺少 **Program-conditioned、event-conditioned 的 native candidate-set distribution summary**。
> correction 函数看不到其它 candidates 在相同 query 坐标下如何分布。

## 2.2 branch 的 vector extension仍属于相同的 set-independent 家族

`2295f48` 增加了：

* input/output × family 共享的 native query projection；
  -对应 native candidate projection；
  -32D normalized query/key 向量；
  -与 semantic query/key 的逐元素乘积；
  -该32D向量并入原 per-candidate MLP。

但每个 candidate 的输出仍只依赖它自己的 projected native value、当前 query、local event和metadata；没有 set attention、candidate distribution、event-level native moments或其它 cross-candidate状态。

此外，这个 vector 扩展不是一个“任意高容量 pointwise 网络”：

* native projection在每个 family×side内跨18层owner共享；
* q各层、v各层的native坐标未必天然对齐；
  -向量交互采用 separable elementwise product；
  -最终仍由family-shared小MLP压成一个标量。

因此，它的失败不能外推到所有 owner-specific、group-specific 或任意通用 pointwise map。

## 2.3 V1—V4的实验归因基本成立

### V1

V1 的 wrong 梯度除以很小的 \(B_{\rm free}\)，导致相对 correct 梯度被放大约15.7–359.6倍。correct 与 wrong 一起降至约 `-.39`，只淘汰这个量纲不一致的 objective，不淘汰架构。

### V2

V2 改为 raw-unit 1:1 后：

* correct fit `.672942`；
* held `.663154`；
* wrong `.345229`；
* margin `.185745`；
* 9/10 correct优于wrong。

它证明 scorer 不是完全 bank-blind。但两条correct总权重与一条wrong的负权重相抵，使三臂共同恶化为一条一阶零代价方向。把 V2 归为 objective-identifiability non-pass 是正确的。

### V3

2:1 positive anchor 后：

* correct恢复到`.929101`；
* held `.953285`；
* wrong仍`.934305`；
* margin约0。

这一结果已经排除了“只需保护correct质量或调整正负loss比例”作为充分解释。

### V4

加入真实 B1 base score 后，correct/held继续约`.93/.95`，wrong仍约`.933`。结合六任务首步梯度 cosine约`-.966`，可以正式排除：

> 当前 scalar alignment + semantic alignment + base-score pointwise chart，只是缺少一个简单标量坐标。

上述数值及停止边界在当前 authority 中记录一致。

## 2.4 training/evaluation执行面没有发现明显偏离

正确与错误bank走的是同一个deployment-shaped生成路径：

* fixed route只替代Program task content，属于training-only mechanism control；
  -wrong bank仍用目标任务exact language重新提取local context；
  -真实X/Y、current bank、interaction scorer、signed pooling和rank16 materialization均实际执行；
  -correct与wrong的区别不是wrapper选择不同operator；
  -只训练interaction scorer，R5 base、Stage0、carrier、scale和Action Meta冻结。

因此 V3/V4 的 wrong-bank保持不能合理归因于：

* 错误bank没有真正替换；
  -correct/wrong走了不同forward；
  -interaction correction没有接入pooling；
  -第二adapter或task lookup；
  -Action Meta污染。

## 2.5 vector exact-effective-rank4诊断的证据强度

这项诊断有两条真实正证据：

1. wrong fit1从未反传，却与wrong fit0同样被压低；
2. correct held从未反传，却与correct fit views接近。

所以模型确实依据bank内容形成了可跨同任务视频复用的变化，而不是单纯记住单个video row。

与此同时，两任务都出现相同 tradeoff：

* task1：correct约`.71–.72`，wrong约`-.52`；
* task93：correct约`.57–.60`，wrong约`-.38至-.42`。

这足以否定“只有shared跨task学习失败，task-local函数类一定可以轻易通过”的解释。

但它仍有三个限制：

### 第一，effective rank4 target仍是一个代表元

它比pointwise logits可靠得多，也已经消除了LoRA factor gauge；但功能等价的rank4 updates仍可能不唯一。wrong-bank free-delta teacher只是“能压低wrong”的一个解，未必是最容易与correct R5 update共同表示的那个解。

因此当前结果证明的是：

> 当前优化和target选择下，vector chart没有同时逼近这组correct/wrong effective-update representatives。

还不是：

> 不存在任何pointwise vector function能够维持correct utility并压低wrong utility。

### 第二，只有两个task

task1和task93覆盖了两个不同role并取得一致结果，足够作路线停止决策；不足以形成跨全部family/task的理论不可能性结论。

### 第三，task-local driver和raw rows不在Git

远程branch commit本身只修改scorer与测试，没有包含这次 exact-effective-rank4 优化的正式driver/config。main固化了指标、数据边界与结论，但我无法仅凭远程Git重新核对：

-每step具体target aggregation；
-family归一化；
-optimizer状态；
-correction saturation；
-最终每target误差分布。

这不否定tracked结果，但其证据级别应标为：

> **文档化formal/diagnostic evidence，尚非完全代码可复现证据。**

---

# 3. 是否已经到达上次意见的停止边界

## 结论：已经到达，但范围应严格限定

我上次 §9.3 的停止条件是，在R5强base与fixed route下出现以下组合：

* correct无法保持，或correct与wrong始终同时高；
  -模块确实读取native candidates和local events；
  -没有streaming、bias、scale或loss分支错误；
  -结果跨多个task和input/output family成立。

当前证据满足：

| 条件               | 当前证据                                 |
| ---------------- | ------------------------------------ |
| correct与wrong同时高 | V3、V4在10 tasks上成立                    |
| 降wrong时损失correct | V2及两taskvector上界成立                   |
| 模块读取bank         | vector wrong fit1和correct held的零梯度泛化 |
| pooling/bound有容量 | free-delta约`.002`即可10/10压低wrong      |
| 非简单loss问题        | V1→V2→V3解析修正及V4                      |
| 多family执行正常      | V3/V4四family与correct capacity通过      |
| 基础实现正常           | zero-init、chunk、唯一rank16、信息墙通过       |

所以：

> **不应再在当前 scalar/vector local chart 上启动 task-LOTO、Natural Program joint G3 或 V5。**

## 但不能宣称“所有逐candidate continuous interaction均已淘汰”

上次停止条件中的“该候选级函数类”应解释为当时提出并实际实现的候选类，而不是数学上所有可能的permutation-equivariant functions。

当前充分淘汰的是：

$$
\delta_{B,n}
=
h_\theta\!\left(
z_{re},
c_{B,e},
v_{B,n},
\mu_B,
q_B^0,
q_B^{0\top}(v_n-\mu_B),
m_n
\right),
$$

其中 \(h_\theta\) 独立作用于每个candidate，且不读取除\(\mu_B,q_B^0\)之外的whole-bank相对分布。

下一架构必须改变**依赖关系**：

$$
\delta_{B,n}
=
h_\theta\!\left(
\phi_{B,n},
S_B(\{\phi_{B,m}\}_{m=1}^{N})
\right),
$$

而不只是把 \(\phi_{B,n}\) 再扩宽32或64维。

---

# 4. 当前根因排序

## 第一：缺少Program-conditioned native bank-set相对上下文

这是当前最可能、且最早尚未测试的接口。

main scorer知道：

* 当前candidate；
  -全局native mean；
  -full-inverse base query；
  -该candidate的base score；
  -Program/local event语义。

但它不知道：

-这个candidate在全bank中是常见还是异常；
-它相对其它event/rank queries处于什么位置；
-correct和wrong bank的alignment分布是否不同；
-当前type相对其它abs/adj/init/goal candidates的结构；
-应怎样根据whole-bank状态改变自己的参数作用。

这就是pointwise local mapping与set-conditioned mapping的本质差异。

## 第二：full-inverse base path主动消除了主要bank spectrum

在小score区域：

$$
F_B^0
\approx C_Bq_B^0
=
C_BC_B^+d_P
\approx P_Bd_P.
$$

当correct和wrong banks的retained subspaces都覆盖\(d_P\)时，二者自然实现相同方向。full inverse不是bug；它正是R5 correct capacity高、wrong也高的原因之一。

它当前应保留，因为：

* R5 correct capacity依赖它；
  -free-delta证明在其上叠加很小correction即可产生wrong specificity；
  -尚没有证据说明必须牺牲base capacity。

但它把新模块的任务变得很明确：从一个强、近乎bank-invariant的base方向周围，找到小而选择性的bank-dependent residual。

## 第三：当前pointwise参数空间缺少选择性切向方向

设最终effective update为：

$$
F_B(\theta)
=
\sum_n w_{B,n}(\theta)v_{B,n}.
$$

pointwise scorer的Jacobian为：

$$
J_B
=
\sum_n
\frac{\partial F_B}{\partial \delta_{B,n}}
\frac{\partial h_\theta(\phi_{B,n})}{\partial\theta}.
$$

当correct/wrong的局部feature distributions相似时，两个 \(J_B\) 的主要参数方向也相似。于是：

-保correct的梯度；
-压wrong的梯度；

在参数空间中近乎相反，而不是存在一个只作用于wrong的正交方向。V2–V4约`-.96`的cosine正是这一现象。

## 第四：Program/native坐标不兼容仍可能存在，但不是当前最早接口

vector task-local诊断使用fixed route并允许每task独立优化，已经绕过Natural Program acquisition。它仍出现tradeoff。因此当前不能先把失败归给：

* Natural Program schema；
* G2训练；
* Stage0；
  -跨任务泛化。

这些问题会在set-conditioned fixed-route Gate通过后重新出现，但不是当前第一修复对象。

## 第五：signed correction形式本身不是首因

free-delta在同一：

* `[+\delta,-\delta]`；
  -真实X/Y；
  -exact softmax；
  -rank4；
  -scale；
  -完整rank16；

接口上成功压低wrong，所以不存在“antithetic correction从数学上无法表达选择性”的证据。

## 第六：effective-rank4 targets之间可能有代表元冲突

这是仍然存在的不确定项。

correct target被固定为R5 interaction-off update，wrong target被固定为某个free-delta update。两者都具有功能意义，但未证明是联合表示最容易的representatives。

这也是我不把两task结果上升为所有pointwise函数不可能性的主要原因。

## 第七：optimizer或credit不再是首选解释

V1确有objective bug；V2也有flat direction。但V3/V4已经移除这些已知问题，vector又做了task-local exact-target优化。不能绝对排除非凸优化失败，但继续普通optimizer或loss比例扫描的信息增益很低。

---

# 5. 被证据淘汰和未被淘汰的精确范围

| 类别                                                           | 裁决           |
| ------------------------------------------------------------ | ------------ |
| V1 normalized wrong-benefit objective                        | 淘汰           |
| V2 raw 1:1且允许共同破坏的objective                                  | 淘汰           |
| V3 2:1 scalar/local scorer作为充分解                              | 淘汰           |
| V4加入base-score scalar后的pointwise MLP                         | 淘汰           |
| 32D family×side共享投影、separable elementwise vector interaction | 不再作为active候选 |
| pointwise free-delta logit imitation                         | 淘汰为资格目标      |
| 对上述实现继续扫width/bound/LR/seed/temperature                      | 无依据          |
| 所有数学上可能的pointwise universal function                         | 未被严格证明不可能    |
| whole-bank/set-conditioned candidate correction              | 未测试，推荐下一步    |
| exact signed pooling                                         | 保留           |
| `[+\delta,-\delta]` antithetic correction                    | 保留           |
| 真实X/Y Value                                                  | 保留           |
| full-inverse zero-init base                                  | 暂时保留         |
| rank4 residual + carrier12                                   | 保留           |
| Natural Program schema                                       | 未被当前证据淘汰     |
| frozen Stage0                                                | 未被当前证据淘汰     |
| Native-Factor/ECP整体路线                                        | 远未达到停止条件     |

---

# 6. 唯一推荐架构：Event-Conditioned Bank-Set Relative Interaction

以下简称 **EBSRI**。

它不是full self-attention，也不是N×N Set Transformer。采用少量Program-conditioned inducing summaries，复杂度保持线性：

$$
O(NRED)
$$

而非：

$$
O(N^2).
$$

固定形状：

* targets \(J=38\)；
* residual ranks \(R=4\)；
* events \(E=8\)；
  -set coordinate width \(D=R\times E=32\)。

## 6.1 保留的base路径

现有Program scorer继续产生：

$$
z_{jre}
=
\text{rank\_event}[j,r,e]
\in\mathbb R^{128},
$$

以及event-native queries：

$$
u^X_{jre}\in\mathbb R^{d^X_j},
$$

$$
u^Y_{jgre}\in\mathbb R^{d^Y_{jg}}.
$$

聚合primal和full base query仍为：

$$
d_{jr}
=
\sum_e\omega_{jre}u_{jre},
$$

$$
q^0_{B,jr}
=
C_{B,j}^{+}d_{jr}.
$$

base score仍按现有定义归一到RMS `.02`。

## 6.2 用Program-native queries定义canonical candidate坐标

对一个input或output group，记candidate value为 \(v_n\in\mathbb R^d\)。

不再给各层native空间强加一个跨owner共享的任意32D线性投影。改为计算：

$$
\kappa_n[r,e]
=
\frac{
\left\langle
\widehat u_{re},
\widehat{v_n-\mu_B}
\right\rangle
}{
\sqrt d
},
$$

并展平：

$$
\kappa_n\in\mathbb R^{32}.
$$

这是一个**Program-relative、basis-invariant**坐标：

* native basis同时旋转query和candidate时不变；
  -每个owner在自己的native空间内解释；
  -32个channel有明确来源：4 ranks × 8 events；
  -不假设不同层的raw coordinate轴对应。

当前main已计算每个\((r,e,n)\)的native scalar alignment，但只把当前单个alignment交给该rank/event的MLP；EBSRI让模型看到整个candidate相对全部rank/events的结构。

## 6.3 B0同时累计event-conditioned set summaries

对每个video独立，base candidate measure为\(\mu_n\)，frame-to-event assignment为 \(M_{te}\)。

先累计基础矩：

$$
\bar\kappa_e
=
\frac{
\sum_n\mu_nM_{t(n),e}\kappa_n
}{
\sum_n\mu_nM_{t(n),e}
},
$$

$$
\nu_e
=
\frac{
\sum_n\mu_nM_{t(n),e}
(\kappa_n-\bar\kappa_e)^2
}{
\sum_n\mu_nM_{t(n),e}
}.
$$

再由Program/local event生成一对antithetic inducing queries：

$$
a_e
=
W_A[\bar z_{je},c_{B,je}],
\qquad
-a_e,
$$

其中：

$$
c_{B,je}
=
[
P^{local}_{process},
P^{local}_{scene},
\sigma,\rho,\tau
].
$$

定义：

$$
\alpha^\pm_{e,n}
=
\operatorname{softmax}_n
\left(
\log\mu_n
+
\log(M_{t(n),e}+\epsilon)
\pm
\frac{a_e^\top
\operatorname{LN}(\kappa_n)}
{\sqrt{32}}
\right),
$$

$$
s^\pm_e
=
\sum_n\alpha^\pm_{e,n}\,
\psi(\kappa_n,m_n).
$$

最终每个event的bank summary为：

$$
S_e=
[
\bar\kappa_e,\log(\nu_e+\epsilon),
s^+_e,s^-_e,\log Z^+_e,\log Z^-_e
].
$$

这些量通过online log-sum-exp和running moments流式累计，不物化candidate×candidate矩阵。

## 6.4 input与output bank分别处理

### Input X

候选：

$$
n_X=(t,p,h).
$$

只有一个summary scope：

$$
S^X_{j,e}.
$$

绝不能添加假的output type轴。

### Output Y

候选：

$$
n_Y=(t,p,h,u),
\qquad
u\in\{\text{abs,adj,init,goal}\}.
$$

最终signed pooling仍对四type构成的**一个联合候选集合**执行。

summary则保留：

$$
S^{Y,\mathrm{all}}_{jg,e}
$$

以及：

$$
S^{Y,u}_{jg,e}.
$$

每个output candidate只读取：

* all-type summary；
  -自己的type summary。

这不会变成四套LoRA，也不会把最终measure拆成四个独立factor；它只是让模型知道一个abs/adj/init/goal candidate在当前联合bank中的相对角色。

## 6.5 B1 summary-conditioned FiLM

candidate的set-relative坐标为：

$$
\widetilde\kappa_{e,n}
=
\frac{
\kappa_n-\bar\kappa_e
}{
\sqrt{\nu_e+\epsilon}
}.
$$

对每个rank/event：

$$
(\gamma_{re},\beta_{re})
=
G_f[z_{re},c_e,S_e].
$$

candidate hidden：

$$
h_{re,n}
=
\gamma_{re}\odot
\Phi[
\widetilde\kappa_{e,n},
\kappa_n,
s^0_{r,n},
m_n
]
+
\beta_{re}.
$$

event correction：

$$
\delta_{re,n}
=
b\tanh(H_f(h_{re,n})),
\qquad b=0.1.
$$

汇总事件：

$$
\delta_{r,n}
=
\sum_e
\omega_{re}M_{t(n),e}
\delta_{re,n}.
$$

最后严格使用：

$$
\ell^+_{r,n}
=
\log\mu_n+s^0_{r,n}+\delta_{r,n},
$$

$$
\ell^-_{r,n}
=
\log\mu_n-s^0_{r,n}-\delta_{r,n},
$$

$$
f_{B,r}
=
\sum_n
\left[
\operatorname{softmax}(\ell^+)_{r,n}
-
\operatorname{softmax}(\ell^-)_{r,n}
\right]v_n.
$$

真实X/Y仍是唯一Value路径。

## 6.6 为什么它可以产生“第三个方向”

当前pointwise模型是：

$$
\delta_n=h_\theta(\phi_n).
$$

即使两个banks的candidate排列不同，只要局部feature分布高度重叠，相同candidate pattern就倾向收到相同correction。

EBSRI是：

$$
\delta_n=h_\theta(\phi_n,S_B).
$$

即使某个candidate自身的\(\phi_n\)相同，只要其余candidate集合不同：

$$
S_{B_c}\ne S_{B_w},
$$

它就可以收到不同的FiLM参数：

$$
(\gamma_c,\beta_c)\ne(\gamma_w,\beta_w).
$$

因此correct和wrong banks对应的参数Jacobian也不同：

$$
J_{B_c}\ne J_{B_w}.
$$

这在函数空间中新增的不是几个hidden units，而是一个新的条件变量和新的bank-specific tangent space，正是现有约`-.96`梯度冲突中缺少的第三方向。

## 6.7 多视频合同

每条视频独立计算：

* unit-mass quadrature；
  -B0 covariance；
  -set summaries；
  -B1 exact pooling。

随后继续：

$$
f_{jr}
=
\frac1K\sum_{k=1}^{K}f_{kjr}.
$$

因此：

* K=1 identity；
  -video集合置换不变；
  -视频内部保序；
  -frame previous/final状态不跨视频；
  -长视频仍unit-mass；
  -没有video选择；
  -没有多LoRA平均；
  -最终只有一个rank4 residual和一个rank16。

## 6.8 初始化

从R5冻结继承：

* fixed route/chart；
  -233个native heads；
  -full-inverse base；
  -scale；
  -carrier12；
  -source/Stage0。

fresh初始化：

* set-summary inducing-query网络；
  -summary value网络；
  -summary-conditioned FiLM；
  -final correction trunk。

最后correction层严格zero-init，所以step0必须逐tensor复现R5 interaction-off完整rank16。

不加载V1–V4 interaction checkpoint；也不直接合并`2295f48`。vector分支可借用测试和向量化代码，但其scientific参数状态不应成为初始化资产。

---

# 7. 最小实现与实验顺序

这里不是并列提出多个架构，而是一个架构的逐接口资格。

## 阶段S0：free-summary factorization正控

### 目的

先回答：

> 如果已经给出一个whole-bank condition token，新的summary-conditioned head能否同时保持correct和压低wrong？

### 数据

只用现有task1和task93：

* correct fit0、fit1；
  -correct held，零梯度；
  -wrong fit0；
  -wrong fit1，零梯度；
  -panel B只评价。

### 训练

冻结：

* R5 base；
  -native heads；
  -candidate descriptors；
  -real X/Y；
  -signed pooling；
  -scale/carrier。

只训练：

* 新FiLM/correction；
  -每task一个training-only `s_correct`；
  -每task一个training-only `s_wrong`。

同task的correct fit/held共享`s_correct`；wrong fit0/fit1共享`s_wrong`。这些token只做上界，不进入deployment或checkpoint候选。

### loss

可以继续用当前exact-effective-rank4 targets作高速机制loss，但最终Gate必须看panel-B functional recovery。

这一arm同时回答target representative是否联合可表示：

-若free summary能通过，说明correct/wrong targets本身并非不可兼容；
-若effective-MSE不低但functional Gate通过，应判定target过强，不应淘汰架构；
-若两者都失败，summary-conditioned factorization本身不足。

## 阶段S1：真实bank-set summary task-local Gate

保持同一代码，只把free summary替换为上述B0 set encoder。

每个task独立训练一套set encoder和interaction head，避免把跨task泛化与函数类容量混在一起。

训练条件：

* correct fit0/fit1；
  -wrong fit0；
  -held correct、wrong fit1、panel B零梯度。

这一步是当前唯一应立即实施的决定性实验。

## 阶段S2：fixed-route shared task-LOTO

S1通过后：

-冻结R5 base；
-共享一套set encoder和interaction；
-在8个gradient tasks上训练；
-固定hold out一个meta task和一个target task的interaction梯度；
-held tasks仍使用其R5 fixed token/base方向，只检验bank-interaction规则能否迁移。

这个LOTO不证明Natural Program，但能排除“set encoder只记住十个task-bank fingerprints”。

若LOTO通过，可在全部10 tasks上fresh refit同一模块以形成component initialization；这个refit不是新科学版本。

## 阶段S3：Natural Program joint G3

随后才：

-移除fixed route；
-接回Natural Program；
-联合训练Natural Program、set interaction和native heads；
-继续冻结source、Stage0、scale、carrier；
-使用原J2 12-task split、true task-held和same-task held。

不加入：

* functional-code cosine；
  -outer-update reconstruction；
  -pointwise delta teacher；
  -Program Gram；
  -support classifier；
  -binary bank labels；
  -full/half route。

---

# 8. 每阶段Gate与停止条件

## S0 / S1：两个task均需逐项通过

沿用现有fixed-route Gate的绝对口径，不降低到“相对vector分支改善”：

| 指标                        |              要求 |
| ------------------------- | --------------: |
| correct fit0/fit1         |       每项`>=.85` |
| correct held              |       每项`>=.80` |
| wrong fit0                |         `<=.25` |
| wrong fit1零梯度             |         `<=.25` |
| correct−wrong             |    每task`>=.50` |
| correct优于wrong            |          全部pair |
| zero-init interaction-off |            复现R5 |
| correction saturation     | 不允许靠大面积触bound通过 |
| input/output family       |        不得有系统性反向 |
| panel B                   |   与内部target结论一致 |

### S0失败

说明仅增加一个global condition vector也不能在当前base+signed-correction factorization中产生所需选择性。此时不应实现复杂set encoder。

如果S0的effective-target loss失败但functional Gate接近通过，则问题是teacher representative，不是factorization；应改为task-local constrained functional feasibility，而不是换网络。

### S0通过、S1失败

说明“给定bank class状态可以实现”，但native candidate set无法生成足够的summary。最早接口是：

* set encoder；
  -query-relative descriptor；
  -或真实bank content不足。

此时才需要检查set summary decodability，不应返回pointwise width scan。

### S1通过

才有资格进入共享task-LOTO。

## S2：fixed-route LOTO Gate

训练task与held-interaction task分别报告：

* correct fit median `>=.85`；
  -same-task held median `>=.80`；
  -wrong median `<=.25`；
  -margin `>=.50`；
  -held interaction tasks每个correct优于wrong；
  -held/train不低于`.85`；
  -相邻checkpoint正确capacity下降不超过`.05`；
  -两个role均通过。

如果task-local通过、LOTO失败，说明bank interaction规则不可跨task共享；这时才应质疑现有task diversity或set coordinate，而不是Natural Program。

## S3：Natural Program joint G3 Gate

恢复原12-task主Gate，同时保留更严格的bank因果项：

* gradient train median `>=.60`；
  -same-task held `>=.50`；
  -true task-held mean `>=.40`，每项`>=.30`；
  -q/v各`>=.35`；
  -action-in/out各`>=.30`；
  -correct−wrong Program `>=.10`；
  -correct−wrong bank `>=.10`；
  -Program×bank interaction `>=.05`；
  -full高于language和endpoints；
  -same-task retention `>=.80`；
  -相邻checkpoint稳定。

若train/held强、task-held弱，才做matched raw Stage0 probe，而且raw arm必须使用**同一套可训练set interaction**；历史R11使用R9固定chart，不能作为停止Program schema的充分证据。

## 停止新架构的充分条件

EBSRI应被停止，若：

1. free summary S0无法在两task上同时保持correct和压低wrong；
2. 或S0通过但真实set summary S1在两个task上均无法泛化到zero-gradient views；
3. 且candidate summary实际读取、streaming、gradient、target和functional evaluation均经核验；
4. 没有teacher representative冲突这一未决变量。

此时才应重开full-inverse base分解，例如让bank-set summary直接形成bank-conditioned primal，而不只是修正base logits。

---

# 9. 吞吐优化方案

当前16–26分钟/task的task-local实现不具扩展性。新方案在正式实验前必须重构执行，而不是先照原Python循环跑完。

## 9.1 复用现有frozen bank cache

已有cache包含：

* raw X/Y；
  -final Y；
  -frame measure；
  -B0 mean/covariance/eigensystem；
  -local Program context；
  -canonical assignment。

这些都不必重跑冻结PI0.5。

## 9.2 fixed-route阶段预计算query-relative descriptors

在S0–S2中，R5 fixed route、feature chart、native heads和base query均冻结，因此可一次性缓存：

$$
\kappa_n\in\mathbb R^{32},
$$

以及：

* base score；
  -metadata；
  -event assignment；
  -log norm；
  -type index。

训练时只运行：

* set summary；
  -FiLM correction；
  -exact signed pooling；
  -effective-rank4 materialization。

这些缓存是node-local、run-local，不进checkpoint，也不是deployment输入。

## 9.3 按shape批处理全部targets

不能继续在Python中逐：

* target；
  -rank；
  -event；
  -group；

发射小kernel。

应按以下bucket展平：

* input q；
  -input v；
  -input action-in；
  -input action-out；
  -output q groups；
  -output v；
  -output action-in groups；
  -output action-out。

将：

$$
[\text{group},R,E,N,D]
$$

作为batched GEMM和softmax输入。

## 9.4 summary采用线性induced attention

没有N×N self-attention。

每个event只有一对antithetic summary heads，因此：

$$
O(NED)
$$

或含rank descriptor构造时：

$$
O(NRED).
$$

online log-sum-exp可以沿candidate chunk累计，和现有exact signed pooling使用相同的数值模式。

## 9.5 task-local多GPU方式

task1和task93彼此独立：

-各占一张GPU；
-不做DDP或梯度all-reduce；
-同时运行；
-功能panel仅在step0、最终和必要相邻节点评价。

S2 shared模型再恢复DDP/task-balanced分片。

## 9.6 吞吐资格

在启动full10或task-LOTO前要求：

* cached task-local update相对当前实现至少有约4倍加速；
  -80-step单task不应再超过约5–7分钟；
  -fixed-route shared global update回到既有`<=30s`目标、`45s`硬上限；
  -evaluation wall不超过训练主体约一半；
  -最长真实video无OOM且GPU计算持续饱和。

不应通过减少Gate rows或删掉zero-gradient views来换速度。

---

# 10. 对Natural Program、Stage0、整体joint Writer、随机fresh和Final的修订

## 10.1 是否正确执行了第五次专家方案

**是。**

你们已经：

-退役binary route；
-接入continuous candidate correction；
-保持真实X/Y和唯一rank16；
-让correct/wrong使用同一forward；
-先做fixed-route Gate；
-在non-pass后按objective、feature、gradient和task-local上界逐层定位。

当前不是实现偏离，而是已经走到了该方案预设的失败边界。

## 10.2 是否继续坚持“Program与bank共同形成唯一continuous direction”

**应继续。**

失败的是：

> Program/bank/candidate各自提供local features，再由set-independent MLP独立打分。

没有失败的是：

> Program和当前bank共同决定一套continuous signed measure。

EBSRI正是对上位原则的最小结构修正。

## 10.3 是否先修复fixed-route正控

**必须。**

在fixed route下都不能同时保correct和压wrong时，接回Natural Program只会重新混入：

* Program acquisition；
  -task泛化；
  -Stage0；
  -optimizer interference。

所以顺序仍应是：

$$
\text{set-conditioned fixed-route}
\rightarrow
\text{interaction LOTO}
\rightarrow
\text{Natural Program joint}.
$$

## 10.4 是否重新考虑full-inverse base

现在不应先删除。

full inverse确实产生bank interchangeability，但free-delta证明：

$$
\text{full base}+\text{small bank-conditioned correction}
$$

在每个condition内具备所需容量。

只有EBSRI的free-summary正控也失败，才说明“强base anchor + correction”本身可能造成不可联合的capacity–specificity约束，届时才应让bank summary更早进入primal形成过程。

## 10.5 是否回到G2或解冻Stage0

当前没有依据。

fixed-route vector task-local已经在Natural Program之前失败，所以：

-修改Program schema不能修复当前局部接口；
-解冻Stage0也不能解释fixed route task-local tradeoff。

只有fixed-route set interaction通过、Natural Program失败，并且matched raw Stage0 +同interaction明显通过时，才重新打开Program压缩；raw Stage0也失败后，才允许窄解冻process/presence/uncertainty tail。

## 10.6 是否现在直接整体端到端Writer训练

现在仍不应。

当前局部函数类尚未通过机制容量Gate。直接全联训会允许模型：

-绕过bank interaction；
-退回language/task identity；
-学习common residual；
-把失败分散到Stage0、Program、scale等多个模块。

但一旦S2 fixed-route LOTO通过，就不应再机械重复R4–R10式坐标课程；应较早进入Natural Program+interaction+native heads联合functional训练。

## 10.7 G4与Final

Final仍必须正式比较：

### Component-init joint

从已通过Gate的：

* Stage0；
  -Natural Program；
  -set interaction；
  -native heads；
  -scale；

初始化，fresh optimizer/scheduler，整体联合训练。

### Fully random Writer joint

除冻结source PI0.5和既定carrier外，Writer全部随机初始化：

* Stage0 Writer projections；
  -Program；
  -set interaction；
  -native heads；
  -scale/confidence。

两者架构、参数量、数据、loss、checkpoint节点和evaluation必须完全一致。owner authority明确要求保留这一matched fresh候选。

## 10.8 Final最小loss

最终默认只保留：

$$
L_{\text{Final}}
=
L_{\text{correct cross-episode action/flow}}
+
\lambda_wL_{\text{wrong-bank/video necessity}}
+
\lambda_pL_{\text{preservation}}
+
\lambda_oL_{\text{on-policy}}.
$$

其中outer/on-policy项只有base Writer已显示真实闭环增量后才启用。

必须退出正式loss：

* Program Gram；
  -functional-code cosine；
  -effective-rank4 teacher reconstruction；
  -pointwise free-delta imitation；
  -support AUC；
  -binary labels；
  -full/half；
  -polar/subspace losses；
  -fixed routing tokens。

## 10.9 Action Meta

继续关闭。

R5中action-in/out已经强；当前瓶颈是跨bank选择性，不是action侧容量。只有base joint Writer已经：

-在held5产生稳定净增；
-q/v与bank interaction通过；
-剩余失败明确集中在action timing/amplitude；
-两个相邻checkpoint均显示action family落后；

才进行一次matched Action Meta on/off。没有净closed-loop收益或损害breadth/retention，即永久关闭。

## 10.10 整体路线停止条件

当前结果只足以停止local pointwise interaction，不足以停止Native-Factor/ECP。

整个路线停止至少需要：

1. set-conditioned fixed-route正控；
2. interaction task-LOTO；
3. Natural Program joint；
4. matched raw Stage0分解；
5. component-init和fully-random joint Writer；
6. 至少两个fold；
7. held5真实rollout；
8. validation8 strict paired400；
9. full相对language/endpoints/wrong/same-task完整controls；
10. selected checkpoint冻结后才做shuffled/reversed。

这些路径均完成后仍无法形成稳定、广泛、Goal/Long非零且full-video必要的闭环增量，才足以停止整个研究命题。

---

## 远程证据仍缺少的最小内容

为了把“pointwise函数类存在capacity–specificity冲突”从强路线证据升级为完全可复核结论，无需上传checkpoint或完整X/Y，只需补充：

1. exact-effective-rank4 task-local driver或完整伪代码；
2. task1/task93的run contract与optimizer配置；
3. 每step correct/wrong、四family effective-MSE；
4. correction RMS、p95、bound saturation；
5. fit0/fit1/held与wrong fit0/fit1的task/video IDs及zero-gradient flags；
6. final panel-B raw losses、carrier/free reference和normalization；
7. 每target最终effective-rank4误差；
8. 当前scalar/vector per-candidate feature、mass、event assignment和gauge-centered teacher delta的一个压缩导出。

最后一项可以直接测量：

$$
\operatorname{Var}(\delta^*\mid \phi_{\rm local})
$$

是否很高，以及加入bank-set summary后该条件方差是否显著下降。它会为“缺少set context”提供比继续训练另一个pointwise版本更直接的identifiability证据。

**最终唯一推荐动作是：在main的新隔离分支上实现Program-relative、event-conditioned bank-set summary，并先完成task1/task93的free-summary与真实summary成对资格；不合并现有vector分支，不启动V5、task-LOTO或Natural Program joint，直到该fixed-route Gate同时保住correct并压低wrong。**
