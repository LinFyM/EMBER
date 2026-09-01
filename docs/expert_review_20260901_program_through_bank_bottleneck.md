# 1. 核心结论

我已确认 `main@d6f5715bf49277f1d8618e34fa9da84981eb827c` 可以从远程访问；它是上次锁定的 `92617d0` 的线性后继，区间共 32 个提交、无分叉。当前最高优先级状态是：relational quotient S0 non-pass 后，EBSRI 新架构、训练和 GPU 工作均已暂停，等待本轮审查。

## 当前究竟卡在哪里

EMBER 当前卡在：

> **如何把 Program 中的任务信息通过当前真实 native bank 转换成一个可跨任务共享的选择规则，而不是把 Program 本身当作任意任务代码，或让 whole-bank summary 只是一个附加特征。**

一天内完成的实验并非“白做了”，而是连续排除了三类表面解释：

1. **不是 task-local 容量不足。** full-z S0 与 real-summary S1 在 task1、task93 上都能同时保住 correct 并压低 wrong。
2. **不是只要换成真实 functional VJP 就自然解决。** fresh direct-functional S2 能恢复 correct，却几乎不产生 wrong-bank specificity。
3. **不是有一个旧表示再配合梯度归一化就能稳定迁移。** functional-polish 能在八个训练任务上制造 bank specificity，但 correct absolute capacity 和完全未见任务迁移仍不通过。

真正尚未完成的是一个**无旁路、拓扑匹配的共享接口验证**：

* Program 必须能够查询当前 bank；
* 但 raw/absolute Program state 不能直接生成 B1 candidate head；
* B1 必须只从 Program 查询当前 bank 后得到的、逐 target/rank/event/group/type 对齐的 set response 中得到任务条件；
* 最终功能方向仍由当前真实 X/Y 的唯一 signed measure 产生。

## absolute Program code 是不是主因

它是目前最可信的主因之一，但**三次 quotient S0 并没有干净地证明它**。

关键代码事实是：在 S0 中，`ProgramBankSetSummaries.with_condition()` 会把同一个 training-only `[E,S]` condition 覆盖到全部 input targets、全部 output groups、all-type 和四个 by-type scopes。该 override 替换了由 B0 inducing 产生的 induced summaries 和 log-partitions；B0 的 mean/log-variance仍保留，但它们本来就不依赖 inducing query。换言之：

> 虽然 full-z 与 quotient 的代码拓扑在 B0 上确实不同，但在已经执行的 **S0 free-summary 实验中，删除 Program-conditioned B0 inducing 基本不会影响传给 B1 的 condition**。

所以，当前文档把三次 quotient S0 的失败部分归因于“同时删除了有用的 B0 Program read”，对于 S0 而言说得过强。真正严重的混杂是：

> full-z 拥有逐 target/rank/event 的 raw `z` 直达 B1，而 quotient 只剩一个广播到所有 native scopes 的全局 free token和少量结构 slot。full-z 可以用 `z` 补偿粗粒度 token；quotient 不可以。

相关执行路径可由 `with_condition()`、S0 `_interaction_output()` 和 B1 batched condition 代码直接确认。

因此，rank/event、owner、relational quotient 只证明：

* 删除 raw absolute B1 code；
* 同时仍用一个过窄的 global free condition；
* 再以当前结构 slots 或 centered relation补偿；

无法在 task93 上恢复 correct capacity。

它们没有证明：

* summary-only B1 不可行；
* Program-conditioned real B0 不可行；
* fixed full-base + bounded correction 不可行。

## 唯一推荐路线

我建议继续 EBSRI，但进行一次明确的因果重构：

# **Program-through-Bank Bottleneck EBSRI**

其核心是：

1. Program 仍生成 native event queries、base primal、event weights；
2. Program 只能作为 query 去读取当前真实 candidate set；
3. B0 输出逐 target × rank × event、逐 output group/type 的 bank response；
4. B1 不再读取 raw、centered 或 relational Program state，也不直接读取高维 local Program code；
5. B1 的 candidate head只能由 bank response和合法固定 owner/rank/event结构生成；
6. 最终继续对真实 X/Y 做唯一 positive-minus-negative exact pooling。

先做一个**真实拓扑匹配的 free-summary S0**，再做真实 Program-conditioned B0 的 S1，之后只运行 fresh direct-functional shared S2。不要再做 quotient 组合、effective-surrogate shared训练或unit-gradient polish。

我对根因的主观置信度是：

| 解释                                         |      置信度 |
| ------------------------------------------ | -------: |
| raw Program旁路 + S0 summary拓扑不匹配，使共享表示欠识别   |     约60% |
| 八个gradient tasks不足以识别可迁移bank rule          |     约20% |
| functional梯度合成/优化仍是主要问题                    | 约10%–15% |
| fixed full-base + bounded correction本身不可兼容 |     约10% |
| Stage0、真实bank、signed pooling或rank4容量为首因    |  当前低于10% |

---

# 2. 对现有证据链与实现忠实度的审计

## 2.1 EBSRI 主体实现总体忠实

上次意见要求的主要结构已经实现：

* candidate坐标是4 rank × 8 event的32维Program-relative native alignment；
* B0按单位质量、event assignment累计mean、variance和antithetic induced summaries；
* input X没有伪造output-type轴；
* output Y保留all-type及四个by-type summary，但最终仍使用一个联合candidate measure；
* B1产生bounded、branch-specific correction；
* correction进入现有exact streaming signed softmax；
* 每条视频独立处理；
* 最终只产生一个rank4 residual，并与carrier12物化成唯一完整38-target rank16。

当前缓存和执行面也较合理：

* frozen X/Y、B0 operator、candidate coordinates、base score、metadata可以复用；
* B0 summary与B1 replay已经分离；
* target/family同shape路径有batched实现；
* evaluation使用100-job动态队列、两个相邻checkpoint和五臂Panel-B；
* formal信息墙要求held、Panel-B、validation/test零梯度，Action Meta为0。

## 2.2 S0/S1结果是可靠正证据，但证明范围有限

full-z S0：

* task1 correct fit0/fit1/held约`.949/.923/.930`，wrong约`-.535/-.491`；
* task93约`.905/.909/.894`，wrong约`-.162/-.169`。

real-summary S1：

* task1 correct约`.942/.953/.962`，wrong约`-.529/-.517`；
* task93约`.928/.905/.881`，wrong约`-.188/-.180`。

这些结果可靠地证明：

$$
\forall t\in\{1,93\},\quad
\exists\,\theta_t
$$

使完整full-z EBSRI在该任务内部同时具备capacity和specificity，并跨同任务视频泛化。

它们没有证明：

$$
\exists\,\theta_{\rm shared}
$$

能对新任务使用同一映射。

原因是S0/S1对task1、task93分别训练独立interaction模型；fixed Hadamard token在单任务中是常量，不存在跨task code泛化问题。当前active design也明确把S1解释为task-local接口通过，而非shared mapping通过。

还需纠正一项历史归因：`3b7124e` 被文档描述为“把FiLM改成direct condition-generated head的单变量修正”，但同一提交还把peak/decay LR从`1e-4/1e-6`提高为`7e-4/7e-6`。所以S0 pass证明的是：

> direct-head factorization加校准后的acquisition schedule具有容量。

它不能把全部跃升唯一归因于head factorization。

## 2.3 effective-rank4 surrogate S2

该轮最可靠的结论不是“shared EBSRI只能达到`.60`”，而是：

> **该effective-rank4 teacher不是可靠的shared优化目标。**

证据包括：

* effective recovery与Panel-B functional recovery只有Pearson `.417`、Spearman `.433`；
* fresh点的surrogate梯度与direct functional梯度严重错配；
* 后续direct-functional恢复了correct能力，而surrogate模型没有。

因此effective-rank4仍可用于task-local容量优化、family定位和诊断，但不能再用于shared checkpoint选择或正式训练。

它留下的一个有效线索是：surrogate训练过的模型已经形成某种bank-discriminative Jacobian；这解释了为什么后续polish能够压低训练任务wrong，而不是说明surrogate方向本身正确。

## 2.4 fresh direct-functional S2

fresh direct-functional的结果是：

* gradient meta/target correct约`.880/.931`；
  -held task1/93 correct约`.949/.899`；
* gradient wrong约`.444/.905`；
  -held wrong约`.931/.900`。

它可靠证明：

1. direct policy functional VJP能保护或恢复R5 correct basin；
2. wrong hinge长期活跃并不自动产生bank specificity；
3. zero-init set interaction倾向于保持strong base，未自然获取选择性表示。

但该轮与effective-surrogate S2并非严格的“只换loss”实验：`25477c9`还修改了LR、任务/Panel调度、target cache用途和VJP执行方式。因此不能仅由两轮终点差异量化“direct loss比surrogate好多少”。不过低surrogate—functional相关、fresh梯度审计和Panel-B结果共同支持停止surrogate。

## 2.5 functional-polish S2

functional-polish同时改变了：

* interaction初始化：加载旧surrogate checkpoint；
  -optimizer/scheduler/cursor：fresh；
  -每步任务覆盖：全部8 tasks；
  -每task条件：一个correct、一个wrong；
  -梯度合成：每个condition独立unit-L2后按`1/16`等质量平均；
  -wrong mass和Panel调度。

所以它是一个有意设计的**组合机制测试**，不是表示初始化或梯度归一化的单独消融；Git和文档对此已有明确说明。

它最可靠地证明：

* 在一个已经形成bank-discriminative Jacobian的表示上，存在能够让八个训练任务全部correct>w﻿rong的共享下降方向；
* 但该表示没有同时保留足够correct absolute capacity；
* 也没有稳定迁移到held interaction tasks，尤其task93 wrong仍`.566`。

unit-gradient combiner主动丢弃真实functional梯度尺度；它适合判断“是否存在共享方向”，不应成为Final默认优化器。当前代码确实对每个active condition先归一，再等质量合成。

## 2.6 三类S2合起来真正证明了什么

三轮合起来证明的是一个**双稳态困难**：

### 状态A：base-preserving、bank-insensitive

fresh direct-functional：

$$
\text{correct高},\qquad \text{wrong也高}.
$$

### 状态B：training-bank-discriminative、capacity/transfer受损

surrogate bootstrap + unit-gradient polish：

$$
\text{训练wrong降低},
$$

但：

$$
\text{correct下降},\qquad
\text{held-task迁移弱}.
$$

因此当前最早未解决接口是：

> 共享interaction表示是否存在一个既能在train tasks上区分bank、又以相同规则外推到未见task，并保持R5 base capacity的规范坐标。

这比“optimizer失败”“B0没用”或“Program schema失败”都更准确。

## 2.7 三种quotient的实际证明范围

full-z旧路径明确是：

$$
\text{B0 inducing}\leftarrow \operatorname{mean}_{r}z_{jre},
$$

$$
\text{B1 head}\leftarrow z_{jre}.
$$

当前relational路径则是：

* B1读取target-centered \(z_{\rm rel}\) 加固定rank/event slots；
* B0 inducing完全不读\(z\)，只读task-independent event slots和local context。

rank/event、owner、relational三个提交分别改变了这些路径。

不过，S0中的global condition override意味着B0 inducing产生的induced summaries不会进入B1。因此对已执行的S0而言：

* “B0也被删除”是代码拓扑事实；
* “S0结果因B0删除而混杂”却不是主要实际影响。

最主要混杂仍是free condition拓扑：

```text
一个 [E,S] token
  → 广播给38个input targets
  → 广播给全部output groups
  → 广播给all-type与4个by-type scopes
```

代码对此有明确注释。

所以文档中“full-z也使用同一个global token并通过，故global token不是瓶颈”的推理不成立。full-z还有逐target/rank/event的raw z直达B1；两者有效容量并不相同。

---

# 3. 最可能根因与主要替代解释

## 3.1 第一位：共享坐标被absolute Program旁路欠识别

置信度约60%。

fixed Hadamard token本身没有跨任务语义邻域。经过冻结R5 scorer后得到的raw `rank_event`可以在八个训练任务中作为高维代码：

$$
z_t\longrightarrow \theta_t^{\rm candidate-head}.
$$

对于未见token \(z_{t'}\)，没有理由让这个映射自然外推。

S0/S1每task独立训练无法暴露这一问题；S2才第一次要求跨code共享。functional-polish能在训练任务上形成specificity、held interaction却失败，符合“按训练code专门化”的预期。

但是它尚未被干净证明，因为quotient S0同时让B1失去target/rank/event条件容量，且free condition过度广播。

## 3.2 第二位：B0 set representation与B1 condition的作用域不匹配

置信度约45%–55%，与第一项重叠。

真实B0具有：

* input：每target一个summary；
* output：每target、每native group一个summary；
  -每group还有all-type与四个by-type summary。

但S0只给一个global token。删除raw z后，B1没有足够的scope-specific条件去保持不同层、不同rank和不同output type的correct方向。

此外，当前real B0 summary只有event轴，而B1 correction是rank×event。full-z可以用raw \(z_{jre}\)提供rank-specific内容；summary-only结构若不把B0扩展为rank-specific，就可能再次丢失task93所需的q-family rank/event关系。

## 3.3 第三位：task diversity不足

置信度约20%。

S2只有八个gradient tasks、两个interaction-held tasks，而interaction参数量和condition-generated heads仍较大。即便消除raw code旁路，LIBERO任务之间的bank-set分布可能不足以识别一个真正共享的规则。

不过现在不应先增加任务数量，因为现有实验还没有用无旁路、作用域匹配的架构测试这八个任务。先增加数据会掩盖结构问题。

## 3.4 第四位：functional gradient composition

置信度约10%–15%。

证据表明梯度几何依赖表示：

* fresh direct状态中，多任务correct/wrong梯度存在明显冲突；
  -旧surrogate表示上，unit-normalized mean却能对所有训练条件形成正投影。

这说明optimization不是完全无关。但polish仍没有held迁移，且unit normalization会丢失真实尺度。因此不能把下一步变成MGDA、PCGrad或gradient-weight算法研究。

## 3.5 第五位：fixed base + bounded correction

置信度约10%。

full base会主动消除大量bank spectrum，使correct和wrong都能重放相似方向。理论上这增加了correction的难度。

但现有证据尚不足以停止该分解：

* free-delta只需约`.002` p95 correction就能压低wrong；
  -full-z S0/S1已经在真实signed pooling和rank4下通过；
  -当前尚未测试scope-matched summary-only B1。

只有该干净正控失败，才应把bank summary移到更早的primal形成阶段。

## 3.6 当前不应优先归因的接口

目前没有依据优先修改：

* Native Stage0；
  -Natural Program schema；
  -真实X/Y bank；
  -signed pooling；
  -rank4；
  -scale；
  -Action Meta。

固定route的S0/S1/S2问题发生在Natural Program正式接回之前；G1、P0/P1和R5的容量证据仍成立。

---

# 4. 唯一推荐架构：Program-through-Bank Bottleneck EBSRI

## 4.1 总体数据流

```text
exact language + ordered action-hidden videos
        │
        ▼
frozen PI0.5 / frozen Native Stage0
        │
        ▼
Natural Program
P_lang / P_scene / P_process / rho / tau / sigma
        │
        ├── Program → native event queries u[j,r,e]
        ├── Program → base primals d[j,r]
        └── Program → event weights ω[j,r,e]
        │
        ▼
B0a: current-video covariance
        → q0[B,j,r] = C[B,j]⁺ d[j,r]
        → base candidate score s0
        │
        ▼
B0b: Program-through-bank set read
        Program query × real candidate set
        → scope-matched S[B,j,r,e,(g,u)]
        │
        ▼
B1: summary-only candidate head
        S + fixed owner/rank/event identity
        + candidate-relative κ/base score/metadata
        → one continuous ±δ per candidate
        │
        ▼
one exact positive-minus-negative measure over real X/Y
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

## 4.2 Program允许走的路径

Program必须保留三条路径：

### 路径一：native event query

$$
u^X_{jre}\in\mathbb R^{d^X_j},
\qquad
u^Y_{jgre}\in\mathbb R^{d^Y_{jg}}.
$$

这由现有owner/group native heads从未聚合的`rank_event`产生。

### 路径二：Program-relative candidate coordinate

对candidate \(v_n\)：

$$
\kappa_{j,n,r,e}
=
\frac{
\langle
\widehat u_{jre},
\widehat{v_{j,n}-\mu_{B,j}}
\rangle
}{
\sqrt{d_j}
}.
$$

展平后：

$$
\kappa_{j,n}\in\mathbb R^{32}.
$$

这是合法的Program→candidate query-key路径，因为它无法脱离当前bank生成值。

### 路径三：B0 set query

Program可以生成B0 read query：

$$
a_{jre}
=
A_f\!\left(
\operatorname{LN}(z_{jre}),
E_j,E_r,E_e
\right)
\in\mathbb R^{32}.
$$

但这个query本身不得传给B1。它只能通过对真实candidate set做attention产生bank response。

## 4.3 必须删除的旁路

B1不得直接读取：

* raw `program_event_state`；
  -target-centered \(z_{\rm rel}\)；
  -\(P_{\rm lang}\)、\(P_{\rm scene}\)、\(P_{\rm process}\)的高维向量；
  -高维local scene/process/sigma作为condition head代码；
  -task token、task ID或authority ID。

固定owner/rank/event embeddings可以保留，因为它们描述LoRA结构，不描述任务。

event weights和canonical assignment可以继续用于event collapse；它们不能独立生成candidate head。

## 4.4 B0 summary必须匹配真实作用域

设summary value width仍为 \(V=16\)，coordinate width为 \(D=32\)。

### Input

对每个target \(j\)、rank \(r\)、event \(e\)：

$$
S^X_{jre}.
$$

整体shape：

```text
S_X [38, 4, 8, S]
```

### Output

对target \(j\)、native group \(g\)、rank \(r\)、event \(e\)：

```text
S_Y_all  [target, group, 4, 8, S]
S_Y_type [target, group, 4, 8, 4, S]
```

四个type summary只提供上下文，最终candidate axis仍为联合：

```text
(frame, probe, horizon, type)
```

不能形成四套独立output factors。

## 4.5 B0 set read公式

event moments可继续共享rank：

$$
\bar\kappa_{je}
=
\frac{\sum_n\mu_nM_{ne}\kappa_{jn}}
{\sum_n\mu_nM_{ne}},
$$

$$
\nu_{je}
=
\frac{\sum_n\mu_nM_{ne}
(\kappa_{jn}-\bar\kappa_{je})^2}
{\sum_n\mu_nM_{ne}}.
$$

rank-specific induced summaries：

$$
\alpha^\pm_{jre,n}
=
\operatorname{softmax}_n
\left[
\log\mu_n
+
\log(M_{ne}+\epsilon)
\pm
\frac{
a_{jre}^{\top}\operatorname{LN}(\kappa_{jn})
}{
\sqrt{32}
}
\right],
$$

$$
s^\pm_{jre}
=
\sum_n\alpha^\pm_{jre,n}
\psi_f(\kappa_{jn},m_n).
$$

最终：

$$
S_{jre}
=
[
\bar\kappa_{je},
\log(\nu_{je}+\epsilon),
s^+_{jre},
s^-_{jre},
\log Z^+_{jre},
\log Z^-_{jre}
].
$$

必须满足：

* \(\psi\)只读candidate-derived信息；
  -不得把\(z_{jre}\)拼入summary value；
  -不得把B0 query或hidden residual直接传给B1；
  -所有summary都依赖当前真实candidate set。

## 4.6 B1 summary-only head

candidate feature继续使用：

$$
\phi_{jn}
=
\Phi_f[
\widetilde\kappa_{je,n},
\kappa_{jn},
s^0_{jrn},
\log\|v_n-\mu_j\|,
m_n
].
$$

其中：

$$
\widetilde\kappa_{je,n}
=
\frac{\kappa_{jn}-\bar\kappa_{je}}
{\sqrt{\nu_{je}+\epsilon}}.
$$

condition-generated head改为：

$$
w_{jre}
=
g(E_j,E_r,E_e)
\odot
W_f\operatorname{LN}(S_{jre}),
$$

$$
b_{jre}
=
b_f^\top\operatorname{LN}(S_{jre}).
$$

这里不应有一个只靠owner/rank/event embedding就能输出非零task correction的独立bias路径。summary为零或被neutralized时，condition-specific部分必须为零。

candidate correction：

$$
\delta_{jre,n}
=
b_{\max}
\tanh
\left(
\frac{w_{jre}^{\top}\phi_{jn}}{\sqrt H}
+b_{jre}
\right),
$$

$$
\delta_{jr,n}
=
\sum_e
\omega_{jre}M_{ne}\delta_{jre,n}.
$$

最终：

$$
\ell^+_{jr,n}
=
\log\mu_n+s^0_{jr,n}+\delta_{jr,n},
$$

$$
\ell^-_{jr,n}
=
\log\mu_n-s^0_{jr,n}-\delta_{jr,n}.
$$

继续执行：

$$
f_{jr}
=
\sum_n
\left[
\operatorname{softmax}(\ell^+)_{jr,n}
-
\operatorname{softmax}(\ell^-)_{jr,n}
\right]v_n.
$$

## 4.7 为什么这比现有quotient更合理

它同时解决三个混杂：

1. **切断raw code旁路**：B1不能按Hadamard code直接生成head。
2. **恢复合法Program内容**：rank/event Program通过query当前bank形成scope-specific response，而不是被全部删除。
3. **修复S0正控拓扑**：free condition与真实input/output/group/type/rank/event作用域一致，不再由一个global token承担全部38-target LoRA。

它也回答了B0是否“独立必要”：

> 科学上必要的不是名为B0的模块，而是candidate logits形成前必须有一次whole-set read。该read可以在代码中称B0.5或B1-set阶段；只要不物化完整bank，系统上就需要先累计summary再重放candidate。

现有两次流式读取和缓存接口已经适合实现，不需要第三次PI0.5 native forward。

---

# 5. 下一组最小但决定性的实验

只沿上述一条架构推进，不再创建并行quotient版本。

## 5.1 实验A：Topology-matched free-summary S0

### 数据

继续使用：

* task1，wrong task8；
* task93，wrong task94；
* correct fit0/fit1；
* correct held零梯度；
* wrong fit0训练；
* wrong fit1零梯度；
* Panel A只用于训练teacher；
* Panel B为最终裁决。

### 冻结

冻结：

* source PI0.5；
  -Native Stage0；
  -Natural Program；
  -R5 primal scorer/native heads；
  -full-inverse base；
  -real X/Y；
  -exact signed pooling；
  -carrier12；
  -scale；
  -Action Meta。

### 训练

只训练：

* summary-only candidate trunk/head；
  -固定合法owner/rank/event结构；
  -training-only free summaries。

free summaries必须匹配真实scope：

```text
free_X_correct/wrong
  [38,4,8,S]

free_Y_all_correct/wrong
  [target,group,4,8,S]

free_Y_type_correct/wrong
  [target,group,4,8,4,S]
```

同任务的correct fit0/fit1/held共享correct tree；wrong fit0/fit1共享wrong tree。

### Loss

现有family-equal effective-rank4 target可以作为高吞吐优化辅助，但没有选模权。最终只看Panel-B functional recovery。

若effective target与functional结论冲突，只允许同一架构做一次直接functional feasibility裁决，不得据内部MSE淘汰架构。

### Gate

沿用现有S0/S1 Gate：

-每个correct fit `>=.85`；
-correct held `>=.80`；
-每个wrong `<=.25`；
-min(correct)-max(wrong) `>=.50`；
-所有correct views严格优于所有wrong views；
-correction不大面积触bound；
-两task均通过；
-zero-init逐tensor复现R5；
-信息墙与唯一rank16通过。

### 解释

* **通过**：summary-only B1与fixed base+correction分解有容量，进入B。
* **失败**：当前fixed-base + summary-only bounded correction函数类被淘汰。此时不再做owner+relation、normalization或width组合，直接把bank-set response移到bank-conditioned primal形成阶段。

## 5.2 实验B：Real Program-through-bank S1

### 变化

只把free summary tree换为真实B0 set read：

* Program raw \(z\)只用于产生B0 rank-event query；
* B1仍严格summary-only；
* set encoder与candidate head可训练；
  -其余完全不变。

### Gate

与实验A完全相同。

### 解释

* A通过、B失败：真实B0 representation或Program→set query不可用；问题不在B1容量。
* B通过：fixed-route下真实bank response有充分capacity与same-task泛化，进入C。

不再需要rank-only、owner-only或relational S0。

## 5.3 实验C：Fresh direct-functional shared S2

### 数据

保持现有LOTO：

* gradient meta：8、9、32、52；
  -gradient target：72、73、75、94；
  -interaction-held：task1、task93；
  -每task两条fit video、一条same-task held；
  -wrong task映射沿用当前sealed contract；
  -Panel A训练，Panel B评价。

### 初始化

* R5 base/native heads冻结；
  -新B0 set encoder和summary-only B1从fresh初始化；
  -不加载effective-surrogate interaction；
  -不加载functional-polish interaction；
  -optimizer/scheduler/cursors fresh。

### Loss

只使用direct cross-episode policy functional VJP：

$$
L
=
\frac1{8}
\sum_t
\left[
L_{\rm correct,t}
+
\frac12
\max
\left(
L_{\rm carrier,t}-L_{\rm wrong,t},
0
\right)
\right].
$$

要求：

* correct与wrong使用相同Panel-A visit和policy RNG；
  -每步任务等质量、meta/target等质量；
  -correct fit0/fit1轮换；
  -wrong fit0/fit1轮换；
  -不用effective teacher、unit-gradient normalization、MGDA、Program Gram或code reconstruction。

可以记录per-condition梯度几何作为诊断，但不得据此通过。

### Gate

沿用当前S2门槛：

* gradient correct fit median `>=.85`；
  -gradient same-task held `>=.80`；
  -gradient wrong median `<=.25`；
  -gradient margin `>=.50`；
  -held task correct fit `>=.85`；
  -held same-task `>=.80`；
  -held wrong `<=.25`；
  -held margin `>=.50`；
  -held/train `>=.85`；
  -meta与target两个role都通过；
  -task1、task93所有correct views均优于wrong；
  -两个相邻checkpoint稳定。

Panel-B aggregate代码已经实现role、held、family和adjacent-checkpoint检查。

---

# 6. 从当前状态到G3、G4和Final的具体计划

## 6.1 G3

推荐顺序改为：

```text
A. topology-matched free-summary S0
        ↓
B. Program-through-bank real-summary S1
        ↓
C. shared fixed-route direct-functional S2
        ↓
D. Natural Program joint S3
        ↓
F4 scale/preservation
        ↓
K2 → K4
        ↓
held5 strict250 closed-loop
```

这里的S0/S1/S2仍然是functional mechanism Gate，不是环境closed-loop。

## 6.2 Natural Program joint S3

只有C通过后：

-移除fixed Hadamard route；
-接回Natural Program；
-联合训练Natural Program、B0 set encoder、summary-only B1、native heads；
-source、Stage0、carrier、scale首轮继续冻结；
-恢复原12-task true task-held Gate；
-不加入effective-rank4 surrogate、Program Gram、outer-code、support或binary loss。

若：

* train/held-video强而task-held弱，才执行matched raw Stage0 probe；
* raw arm必须使用同一个可训练B0/B1架构，不能复用历史R11的R9固定chart。

## 6.3 held5真实closed-loop

S3 functional Gate通过后，恢复当前F6：

* full至少`60/250`；
  -breadth至少4/5；
  -carrier retention至少`33/43`；
  -Goal或Long非零；
  -full相对language与first+final各净增至少5；
  -same-task retention至少80%。

只有这一阶段才第一次证明functional改进能转化为真实环境闭环。

## 6.4 G4 joint Writer

held5有真实闭环信号后，再联合解冻完整Writer-owned模块：

* Stage0 Writer部分；
  -Natural Program；
  -B0 set encoder；
  -B1 interaction；
  -native heads；
  -scale/confidence。

source PI0.5和carrier继续冻结。

G4不应预设每个task都有目标LoRA；主监督应来自：

* cross-episode teacher action/flow；
  -set-valued successful-policy functional evidence；
  -carrier/source preservation；
  -出现稳定base Writer成功后，才加入自然on-policy success/progress。

当前设计也规定G4只有在shared functional与真实closed-loop信号成立后启动。

---

# 7. 明确的停止与转向条件

## 7.1 停止当前summary-only correction函数类

满足任一项即可停止：

1. topology-matched free-summary S0在task1和task93仍不能同时：

   * correct `>=.85/.80`；
     -wrong `<=.25`；
     -margin `>=.50`；
2. free summary S0通过，但real B0 S1在两task均无法保持zero-gradient correct held或wrong fit1；
3. 实现、summary作用域、梯度、signed pooling和Panel-B均核验无误，仍出现稳定capacity–specificity tradeoff。

停止后唯一合理转向是：

> 让whole-bank response更早形成bank-conditioned primal，而不再把它只作为base query之后的logit correction。

例如：

$$
d_{B,jr}
=
d^0_{P,jr}
+
\sum_e
G_f(S_{B,jre})\,A_{B,jre},
$$

其中 \(A_{B,jre}\) 仍是由真实candidate pooling得到的native anchor；随后再做：

$$
q_{B,jr}=C_B^+d_{B,jr}
$$

和唯一exact replay。

这属于实验A失败后的转向，不与当前路线并行实施。

## 7.2 停止当前shared EBSRI coordinate

若A、B通过，而C出现：

-八个gradient tasks的correct和wrong均通过；
-task1/93两个held tasks在两个相邻checkpoint上都明显失败；
-held/train持续低于`.85`；
-且不再有raw code旁路；

则应停止当前set coordinate/shared head，而不是继续添加quotient或gradient tricks。

这时剩余解释主要是：

* currentκ/summary不是跨任务canonical coordinate；
  -或八个任务不足以识别该规则。

## 7.3 停止Program schema

必须先满足：

1. fixed-route C通过；
2. Natural Program joint能fit gradient tasks和same-task held；
3. true task-held仍失败；
4. matched raw Stage0 + 同一B0/B1架构比Program arm task-held提高至少约`.15`并达到有意义绝对水平。

当前三次quotient和历史R11均不足以停止Program schema。

## 7.4 停止frozen Stage0

只有：

* fixed-route interaction通过；
  -Program arm失败；
  -matched raw Stage0 arm也失败；
  -train functional目标可优化；
  -失败跨q/v/action和两个role；

才允许窄解冻Stage0最后的process/presence/uncertainty投影。没有依据直接解冻VLM或source policy。

## 7.5 停止整个Native-Factor/ECP

当前结果远未达到。

必须至少完成：

-无旁路shared G3；
-Natural Program joint；
-至少两fold G4；
-component-init和fully-random Writer；
-held5真实closed-loop；
-validation8 strict paired400；
-full、language、endpoints、wrong、same-task controls；
-selected checkpoint冻结后的shuffled/reversed。

仍持续表现为低absolute、低breadth、Goal/Long为0、full不优于language/endpoints、same-task不稳，才足以停止整体路线。

---

# 8. 对Final整体端到端训练、简化loss与随机fresh的意见

## 8.1 Final必须保留完全随机Writer

owner要求正确。

Final应做两个matched候选：

### Component-init

从通过Gate的Stage0、Program、B0/B1 compiler、native heads和scale初始化；optimizer、scheduler和数据cursor全部fresh。

### Fully-random Writer

除source PI0.5和既定carrier外，Writer-owned模块全部随机：

* Stage0 Writer projections；
  -Natural Program；
  -B0 set encoder；
  -B1 interaction；
  -native heads；
  -scale/confidence。

两者必须：

-架构和参数量相同；
-训练数据相同；
-loss相同；
-checkpoint节点相同；
-validation合同相同。

不得因为component arm内部loss更低就淘汰random arm。该要求已写入owner authority和Final设计。

## 8.2 Final最小loss

建议正式loss只保留：

$$
L_{\rm Final}
=
L_{\rm cross\text{-}episode\ action/flow}
+
\lambda_w L_{\rm wrong\ video/bank\ necessity}
+
\lambda_p L_{\rm preservation}
+
\lambda_o L_{\rm on\text{-}policy}.
$$

其中 \(L_{\rm on-policy}\) 只有base Writer已经产生真实闭环增量后才启用。

必须从正式loss删除：

* effective-rank4 reconstruction；
  -Program Gram；
  -functional-code cosine；
  -pointwise/free-summary teacher imitation；
  -quotient regularization；
  -unit-gradient polish；
  -support分类；
  -full/half route；
  -polar/subspace loss；
  -fixed route token supervision。

这些可以保留为诊断或初始化研究，不应永久叠加。

## 8.3 Action Meta继续关闭

当前没有证据表明action侧是剩余首因：

* R5四family均可恢复；
  -EBSRI失败同时涉及q、correct capacity和跨任务bank interaction；
  -最新问题发生在Action Meta之前。

只有base Writer已在held5形成稳定闭环增量，并且：

* q/v和bank interaction已通过；
  -两个相邻checkpoint均显示剩余误差集中在action-in/out；
  -rollout失败表现为动作幅值、时序或短程控制，而非错误对象/目标/过程；

才做一次matched Action Meta on/off。

只有实际closed-loop净增且不损害breadth、retention、Goal/Long和video causal margin，才保留。

---

# 9. 吞吐与实现建议

## 9.1 不增加新的native forward

复用现有：

* frozen X/Y cache；
  -covariance/eigensystem；
  -candidateκ；
  -base scores；
  -metadata；
  -event assignments；
  -action batches。

新的rank-specific B0 summary只在这些冻结descriptor上运行。

## 9.2 B0不做N×N attention

复杂度保持：

$$
O(NRED)
$$

而不是：

$$
O(N^2).
$$

每个rank/event只有少量inducing read；mean/variance在rank间共享，只有induced summaries增加rank轴。

## 9.3 按family和native shape批处理

继续使用当前batched path，把同shape scopes打包为：

```text
[group, rank, event, candidate, feature]
```

避免逐target/rank/event Python kernel。

## 9.4 S0两个task独立并行

task1和task93各一张GPU，不用DDP。当前quotient S0 profile约`2.755s/step`、约41.3GB allocated，说明拓扑匹配S0本身应维持分钟级，而不是16–26分钟/task的早期原始实现。

## 9.5 Shared S2

沿用现有：

* no-grad生成rank16 leaves；
  -policy VJP；
  -leaf gradients CPU offload；
  -fresh Writer replay；
  -动态evaluation queue。

新的shared global step应继续控制在当前约数十秒量级，不能因rank-specific summaries变成大矩阵solve。若普通机制训练再次达到小时级且瓶颈来自每condition重复summary计算，应先作吞吐non-pass。

---

# 10. 具体代码修改点

## `src/ember/ecp/bank_conditioning/set_summary.py`

修改：

* moments保留`[E,D]`；
  -induced positive/negative和logZ扩展为`[R,E,...]`；
  -`EventBankSetSummary.condition`产生`[R,E,S]`；
  -online accumulator支持batched rank-specific inducing queries。

现有online moments/log-sum-exp可复用。

## `program_bank_interaction.py`

拆分当前 `_event_context()`：

```python
def b0_program_query_context(...):
    # 可读 Program，用于查询真实 bank；
    # 输出 query，不直接进入 B1。

def b1_structural_context(...):
    # 只含固定 owner/rank/event identity；
    # 不含 raw/relational Program 或高维 local code。
```

删除当前B1的：

```python
relational = program_event_state - mean(...)
```

也不恢复旧full-z：

```python
rank = cat(program_event_state, local)
```

旧full-z和当前relational代码分别可在历史提交与canonical中恢复。

## `ProgramBankSetSummaries.with_condition`

不能再广播单个tensor。改为显式structured free tree：

```python
FreeProgramBankSetConditions(
    inputs=...,
    outputs_all=...,
    outputs_by_type=...,
)
```

并逐scope核对shape。

## `batched_interaction.py`

当前condition为：

```text
rank_context + summary_condition
```

应改为：

```text
fixed owner/rank/event structural gate × summary_condition
```

禁止summary-independent非零head。

candidate features中的：

* standardized κ；
  -raw κ；
  -base score；
  -log norm；
  -metadata；

可以保留。

## `compact_replay.py`

保留：

* B0 summary和B1 replay分离；
  -input/output边界；
  -by-type summary；
  -最终联合signed pool；
  -frozen descriptors；
  -chunked replay。

只扩展rank-specific summary batch。

## task-local与shared contracts

新增唯一schema，不覆盖旧配置：

```text
pi05_ecp_event_bank_set_bottleneck_s0_v1.json
pi05_ecp_event_bank_set_bottleneck_s1_v1.json
pi05_ecp_event_bank_set_bottleneck_s2_direct_v1.json
```

S2不得加载surrogate或polish interaction。

## 必需测试

至少增加：

1. 改变raw Program、但固定B0 summary与κ时，B1输出不变；
2. 改变Program query或bank candidates时，B0 summary变化；
3. summary置零时condition-specific correction为零；
4. 不同target/group/type的free summary只影响自己的scope；
5. candidate permutation不改变summary和pooling；
6. chunked/non-chunked B0及B1一致；
7. K1 identity、K4集合置换不变；
8. input不存在type轴；
9. output仍为一个联合candidate measure；
10. zero-init逐tensor复现R5完整rank16。

当前测试已覆盖quotient invariance、relational路径live、free summary梯度以及output all/by-type联合轴，但尚未覆盖“raw Program不能绕过B0 summary”和“free summary不广播”的新合同。

---

# 11. 文档、代码和归因冲突

## 冲突一：当前执行状态

`progress.md`和`task_plan.md`明确表示暂停、无active successor；active design标题仍称EBSRI为active architecture contract。按AGENTS authority，应以pause为准。

## 冲突二：`concept.md`滞后

`concept.md`仍把最早接口描述成旧candidate-level Program–bank interaction，没有反映whole-bank EBSRI、三类S2和quotient结果，需在owner裁决新路线后更新。

## 冲突三：`findings.md`头部

文件头仍称“四次专家原文”，只列到20260829，实际仓库已有20260830和20260831两份原文。

## 冲突四：S0“单变量”表述

`3b7124e`同时修改head factorization和LR；不能称为严格单变量。

## 冲突五：quotient S0的B0混杂

代码层面full-z和quotient的B0确实不同；但S0 condition override使inducing-dependent summary字段被覆盖。因此B0删除不是S0结果的主要实际混杂。真正混杂是global free token与raw z的有效容量差异。

## 冲突六：owner quotient对global token的归因

“full-z使用同一个global token也通过，因此token容量不是问题”不成立；full-z同时拥有逐target/rank/event raw z，quotient没有。

---

# 12. 仍需的最小原始artifact

当前主裁决不要求完整checkpoint或X/Y bank。若要把shared-coordinate根因进一步量化，只需从现有S2和quotient runs导出：

* task、role、arm、video demo、Panel visit、policy RNG；
  -carrier/generated/free-primal raw loss；
  -functional recovery；
  -每个input target与output group/type的summary mean、variance、induced norm和logZ；
  -B1 condition-head weight/bias norm；
  -correction RMS、p95、near-bound fraction；
  -per-condition direct functional gradient norm；
  -correct/wrong gradient cosine；
  -raw z、centered z、native query、κ和summary之间的linear/kernel readout；
  -summary condition置换或清零后的functional结果；
  -每family recovery。

无需提供完整candidate values。上述字段足以判断：

$$
I(\text{task};z\mid S_B)
$$

是否仍通过B1旁路存在，以及scope-matched summary能否解释task93所需的correct更新。

---

## 最终裁决

S0/S1的通过与三类S2的失败并不矛盾：

* 前者证明每任务存在解；
* 后者暴露共享、可迁移解尚未被识别。

absolute Program旁路是可信根因，但三次quotient因free-summary作用域不足，没有完成干净裁决。当前不应终止EBSRI，也不应恢复full-z、继续quotient拼接、修改Program/Stage0或直接跳整体端到端。

**下一步唯一合理动作是实现“Program只能通过真实bank set read进入B1”的scope-matched EBSRI：先做拓扑匹配S0，再做真实B0 S1，随后只运行fresh direct-functional shared S2。**

这条链通过后再接回Natural Program；失败则依据最早失败接口，明确转向bank-conditioned primal，而不再在当前correction函数类上做小版本迭代。
