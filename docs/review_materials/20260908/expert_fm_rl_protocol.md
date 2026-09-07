# EMBER 架构与 FM＋Writer RL 训练协议定稿建议

**我建议采纳你提出的四点，组成同一条主线。**其中第四点采用最小而明确的实现：时间信息进入帧对消息，并显式进入邻帧聚合的attention bias；不另加一个邻帧Transformer。

最终选择是：

> **末层完整horizon → 逐帧对软对应 → 逐horizon读取两端视觉 → 形成经视觉修正的帧对消息 → 带时间信息的窗口聚合 → 同步更新完整horizon。这个单元重复四轮，之后才压缩H，再做单视频长程组织、集合compiler和完整A/B生成。**

训练采用**严格on-policy的单次联合更新**：同一版本Writer生成LoRA、采集rollout、计算FM和RL梯度；全部梯度合并后，才更新一次参数。旧轨迹不跨参数更新重复用于普通score梯度。

下面把这些选择的依据、精确定义和实现默认补齐。文中的公式性质与推荐默认分开说明；默认值用于形成一套可实施协议，不代表已经证明其性能最优。

---

# 一、对四点整体意见的数学评判

## 1. 只取最后完整horizon：合理，且不必保持输入层轴与输出target一一对应

### 1.1 推荐读取位置

**只读取原生最终归一化之后、`action_out_proj`之前，真正送入动作输出投影的最后50个hidden tokens：**

$$
R_t\in\mathbb R^{50\times1024}.
$$

实现定义以**`action_out_proj`实际接收的输入张量**为准，而不是“第18个block的输出hook”。这样不会误把最终归一化之前的残差状态当成最终动作表示，也不需要先收集18层再融合。

source仍运行完整Action Expert；观察侧Meta仍安装在全部既定Action Expert目标上。去掉的是**Writer保存和处理的层轴**，不是source的计算层，也不是早层Meta的梯度。

当前EMBER读取器捕获的是18层post-layer状态，与你这次提出的接口不同；原生动作路径则会取最终suffix输出、截取完整chunk，再送入动作输出投影。本次建议明确切换到后一个接口。

### 1.2 保留了什么

仍然保留：

* 每帧的完整50个相对动作位置；
* 最终表示对当前图文prefix的依赖；
* 完整Action Expert已完成的动作计算；
* 跨帧错位、重叠、重新规划变化的建模通道；
* functional和RL梯度经末层响应回到全部观察侧Meta的通路。

后续仍可生成全部38个目标的LoRA：

$$
\{R_t,Z_t\}_{t=1}^{T}
\longrightarrow P
\longrightarrow
\{A_m,B_m\}_{m=1}^{38}.
$$

**输入表示的计算层身份，与输出参数的target身份，没有数学上的一一对应要求。**超网络完全可以从共同任务表示生成不同层的更新。

输出target queries的作用正是回答：

“这段过程知识对第 \(m\) 个执行算子应产生什么改变？”

它们不要求输入必须包含“第 \(m\) 层的对应教师特征”。

### 1.3 确实失去了什么

失去的是**对中间计算状态的直接访问**：

* 某些在早中层明显、到末层被压缩或重编码的信息；
* 不同计算深度之间的变化；
* 可用于观察source内部处理过程的多层轨迹。

不能宣称这些信息一定能从末层反推出。compiler也不能恢复已经不可区分的信息。

但保留18层同样不能保证这些信息可被有效利用。项目已有证据没有证明“保留18条过程流”是共享闭环能力的必要条件。早期强Writer也并不依赖当前这种18层过程轴；反过来，它们的能力不能证明本次末层接口必然充分。

**我的判断：这里值得用较清楚的任务表示和显著降低的过程计算复杂度，交换多层直接访问。信息充分性由行为裁决，不把18层当作默认必要结构。**

---

## 2. “1次关系＋1次视觉核实”优于必须先做两次关系

一次帧对关系已经能使query同时依赖两帧：

$$
q_{t\leftarrow u,h}
=
f\big(
U_{t,h},U_{u,0:H-1},
\Pi_{t\leftarrow u},
\tau_u-\tau_t
\big).
$$

因此，在形成第一次对应后立即读取视觉，已经实现“用动作关系解释画面”。不存在必须先经过两个关系block，视觉query才获得跨帧条件的数学要求。

连续两次关系更新的潜在优势，是第一次视觉读取前就获得更宽的响应上下文；代价是未经视觉核实的对应先被传播两次。

在当前单probe响应并非可靠未来轨迹的条件下，我更倾向于：

**让实际视觉证据更早进入，而不是先传播更多未经核实的动作假设。**

首版采用四轮，因此计算次数明确为：

| 方案                 | 关系计算 |   视觉核实 | 窗口聚合与状态更新 |
| ------------------ | ---: | -----: | --------: |
| 上一版“两个关系后读视觉”，重复两次 |   4次 |     2次 |    关系更新4次 |
| 本轮推荐的帧对单元，重复四次     |   4次 | 4次逐对核实 |        4次 |

这保留了原先四次关系更新的数量级，但视觉核实更紧密。不能将它简称为“都是四层，所以成本相同”。

---

## 3. 先逐对核实、再窗口聚合：我推荐这一顺序

你的第三点解决了上一版一个真实的信息组织问题。

### 3.1 两种顺序并不等价

上一版大致是：

$$
p_t=\operatorname{Aggregate}_u r_{tu},
\qquad
v_t=\operatorname{ReadVision}(p_t,\{Z_u\}).
$$

聚合先把多个邻帧的关系压进 \(p_t\)。若这个压缩丢失“哪条对应由哪一帧支持”，后续视觉读取不能保证重新恢复。

本轮则是：

$$
\widetilde r_{tu}
=
\operatorname{Verify}
(r_{tu},Z_t,Z_u),
$$

$$
U_t^+
=
\operatorname{Aggregate}_u\widetilde r_{tu}.
$$

这保留了一个更直接的对应关系：

> 某条动作对应 → 它读取到的两端视觉证据 → 该条经修正的消息。

聚合权重可以依据已经核实的消息决定贡献，而不是先混合再尝试解释。

### 3.2 首轮只有两帧，会不会不够

会有这个限制。例如，两帧都有遮挡，需要第三帧才能判定书是否被抓住，第一轮某条帧对就可能无法单独判断。

但本方案中，第 \(b\) 轮帧对读取的是更新后的 \(U^{(b)}\)，不是永远读取原始单帧 \(R\)。从第二轮开始，两端状态已经含有各自邻域的信息：

$$
U^{(b)}_t
=
f_b\big(\text{以 }t\text{ 为中心的更大邻域}\big).
$$

因此，“逐对核实”并不意味着每轮都只能看两张孤立图片。它意味着：

**每条视觉读取保持两端证据的归属，而发起读取的过程query可以带有多帧上下文。**

有限轮更新仍不保证解决任意长遮挡或所有多帧歧义；最终长程组织负责更大范围的信息交换。

### 3.3 是否还保留一次“先汇总、再读视觉”

**首版不保留。**

原方案的优势可以由后续轮次的上下文和长程组织部分承担。当前没有足够证据支持额外保留另一条视觉读取链，付出重复计算和更复杂归因。

这是对读取顺序的明确选择，不是认为先聚合再读取永远无用。

---

## 4. 时序聚合需要保留时间，但不需要另造一套时序模块

帧对消息已经包含signed gap和对应位移，因此加权求和不必丢失时序。

例如，

$$
\sum_u\alpha_{tu}M(U_t,U_u,\tau_u-\tau_t)
$$

虽然形式上是求和，但每一项已经随相对时间改变。它不是对无时间标记的原始帧做平均。

不过，只让消息MLP间接传递时间，会要求聚合器从压缩后的消息中重新辨认时间。首版增加一个很小的明确通道：

$$
\text{聚合logit}
=
\text{内容score}
+
b_{\rm agg}(\tau_u-\tau_t).
$$

同时，Value仍是包含时间与视觉核实结果的帧对消息。**不额外给Value添加一条独立的“纯时间向量”，也不增加邻帧间Transformer。**

这种操作具备以下结构性质：

* 邻帧与其时间标记、消息一起改变存储排列：输出不变；
* 固定时间位置而替换对应内容：存在改变score和Value的通道；
* 不要求每一次内容—时间变换都必然使输出变化。

第二项是时序信息可被利用，第三项避免把“对任何扰动都敏感”误当作正确过程理解。

---

# 二、最终推荐架构

## 1. 完整数据流与主要维度

```text
exact language + K条独立有序正确视频
    ↓
冻结图文prefix
    ├─ 逐帧原生图文状态 Z[t, token, 2048]
    └─ prefix KV
          ↓
单固定probe + Action Expert + 观察侧Meta
          ↓
最终归一化后、action_out_proj前的 R[t, 50, 1024]
          ↓
投影为 U⁰[t, 50, 256]
          ↓
四轮同步帧对核实单元：
    帧对horizon软对应
    → 逐horizon帧对query
    → 分别读取两端Z
    → 经视觉核实的帧对消息
    → 含signed-time bias的窗口聚合
    → 更新完整50-horizon
          ↓
最终learned H-read：E[t, 256]
          ↓
两层单视频长程temporal Transformer：P[t, 256]
          ↓
多视频集合Compiler，38×16个paired queries
          ↓
c[38, 16, 256]
          ↓
target/rank/side独立native A/B读出
          ↓
唯一完整38-target rank16 LoRA
```

主干宽度256、8个attention heads、FFN扩展比4。首版不使用dropout，避免采集、重放和物化间增加无必要随机性。

---

## 2. 冻结图文证据与末层响应

对每帧，保留同一次真实prefix计算中的：

$$
Z_t\in\mathbb R^{N_t\times2048},
$$

其中包含有效视觉tokens和exact task span对应的图文tokens；保留原有mask与token顺序，不读teacher state。

语言条件 \(\bar\ell\in\mathbb R^{256}\) 由exact language tokens的带位置、带mask读取形成，不增加第二次Gemma视觉前向。

观察侧：

$$
R_t=
\operatorname{PreActionOut}
\big(
\operatorname{AE}_{\theta_0,\mu}
(\operatorname{KV}_t,\xi_0,s=1)
\big).
$$

选择：

* \(\xi_0\in\mathbb R^{50\times32}\)，固定公共probe；
* \(s=1\)；
* Meta沿用Action Expert q/k/v/o、rank4的当前范围；
* vision、Gemma和source基础参数冻结；
* rollout执行不安装读取侧Meta。

初始过程状态：

$$
U^{(0)}_{t,h}
=
W_RR_{t,h}+e_h.
$$

\(e_h\) 是horizon位置身份，不是任务阶段标签。这里没有 \(j\) 轴，也没有隐含的多层融合。

**单probe末层响应仍只是一种动作生成计算的条件响应，不是已校准的教师未来动作。**后续视觉核实继续承担纠正这一语义不足的职责。

---

## 3. 首版时间匹配：选择joint-gap bias，不加入显式速度模型

### 3.1 时间单位

当前LIBERO合同下，使用**原始控制采样格**为共同单位：

$$
\tau_t=n_t,\qquad \delta_h=h,\qquad h=0,\ldots,49.
$$

若采样帧原始索引是 \(0,5,10,\ldots\)，就直接使用这些索引。之后的 `/5` 仅用于数值归一化，**不是再乘一次stride**。

对不同帧率或变速的外部视频，必须先从合法元数据定义共同时间单位；本首版不声称解决跨具身时标标定。

### 3.2 统一矩阵方向

对一个无序帧对，固定按时间排序为 \(a<b\)：

* 行 \(h\)：较早帧 \(a\)；
* 列 \(g\)：较晚帧 \(b\)；
* \(\Delta=\tau_b-\tau_a>0\)。

候选对应关系为：

$$
\tau_a+h\approx\tau_b+g
\quad\Longleftrightarrow\quad
h-g-\Delta\approx0.
$$

例如 \(\Delta=10\)，早帧 \(h=20\) 对应晚帧 \(g=10\)。

定义每个attention head的score：

$$
C^{(b,\alpha)}_{ab}(h,g)
=
\frac{
F^{(b,\alpha)}(U^{(b)}_{a,h})^\top
F^{(b,\alpha)}(U^{(b)}_{b,g})
}{\sqrt{32}}
+
B^{(b,\alpha)}
\left(
\frac{\Delta}{5},
\frac{h-g-\Delta}{50}
\right).
$$

这里 \(B\) 是小型joint-gap MLP。两个输入等价地保留起点差和horizon错位，只是显式使用了时间残差坐标。

**首版不再添加速度混合、可学习全局warp或固定高斯对齐带。**原因是当前hidden并非校准轨迹；joint-gap与内容已经能表达宽松对应，显式速度估计会新增尚未获得监督支持的解释变量。

时间MLP的最后投影零初始化，使初始模型不被一条人为设定的窄对齐带支配。时间结构提供可学习接口，不预先宣称哪条对应正确。

### 3.3 空匹配与双向读取

对两个方向分别归一化：

$$
[\Pi_{a\leftarrow b}(h,:),\,\pi^{a}_{\varnothing}(h)]
=
\operatorname{softmax}
[C_{ab}(h,:),\,c^a_{\varnothing}(h)],
$$

$$
[\Pi_{b\leftarrow a}(g,:),\,\pi^{b}_{\varnothing}(g)]
=
\operatorname{softmax}
[C_{ab}^{\top}(g,:),\,c^b_{\varnothing}(g)].
$$

空匹配logit由当前端状态和signed gap产生。它表示“这组horizon候选不能充分解释当前位置”，不表示整条视频无用。

当前关系实现已经有双向分别归一化、joint-gap bias以及对应位移读取；新增的是空匹配和后续逐对视觉核实，不是从零发明一套时间处理。

不设置“远端必然低可靠性”的固定衰减。\(h,g\) 的近远身份通过位置编码保留，可靠性由学习和视觉证据决定；\(|h-g|\) 不作为近远的替代。

---

## 4. 帧对对应怎样成为视觉query

以下对任意有向读取 \(t\leftarrow u\) 定义，记

$$
d_{tu}=\tau_u-\tau_t.
$$

对每个 \(h\)，先形成匹配内容：

$$
m_{tu,h}
=
\operatorname{ConcatHeads}
\left[
\sum_g\Pi^\alpha_{t\leftarrow u}(h,g)
V^\alpha(U^{(b)}_{u,g})
\right].
$$

保留非空质量：

$$
w^\alpha_{tu,h}=\sum_g\Pi^\alpha_{t\leftarrow u}(h,g).
$$

保留对应位置分布的learned读取：

$$
r_{tu,h}
=
\sum_{\alpha,g}
\Pi^\alpha_{t\leftarrow u}(h,g)
e^\alpha_{g-h}.
$$

不把它归一化成一个必须存在的匹配，也不因空匹配概率接近1而除以很小的数。

帧对query为：

$$
q_{tu,h}
=
f_Q^{(b)}
\left(
U^{(b)}_{t,h},
m_{tu,h},
m_{tu,h}-U^{(b)}_{t,h},
r_{tu,h},
w_{tu,h},
\frac{d_{tu}}5,
\bar\ell
\right).
$$

因此query同时依赖：

1. 当前horizon位置；
2. 邻帧完整horizon的软匹配内容；
3. 对应位置如何错位；
4. 两帧真实时间间隔；
5. 当前任务语言。

它没有退回“同帧R查询同帧Z”。

### 为什么逐horizon读取

**首版采用每个有向帧对50个query，不提前合并H。**

同一帧对中，近端可能表达“接近”，远端可能表达“搬运或放置”。若先用少量公共query压缩，再读取视觉，会在最需要核实的地方提前合并这些假设。

代价是视觉cross-attention增加。下面通过帧级K/V复用、edge batching和checkpoint控制，而不是先删掉这项区分能力。

---

## 5. 两端视觉读取与消息修正

对两端分别读取，保持证据归属：

$$
z^{\rm self}_{tu,h}
=
\operatorname{Attn}_Z^{(b)}
(q_{tu,h}+e_{\rm self}, Z_t,Z_t),
$$

$$
z^{\rm other}_{tu,h}
=
\operatorname{Attn}_Z^{(b)}
(q_{tu,h}+e_{\rm other}, Z_u,Z_u).
$$

两端共享K/V投影和读取参数，只用固定角色身份区分“当前端”和“邻帧端”。每帧 \(Z_t\) 的K/V在一轮中只投影一次，供所有相关帧对复用。

视觉核实后的帧对消息：

$$
M^{(b)}_{tu,h}
=
f_M^{(b)}
\left(
U^{(b)}_{t,h},
m_{tu,h},
r_{tu,h},
w_{tu,h},
\frac{d_{tu}}5,
z^{\rm self}_{tu,h},
z^{\rm other}_{tu,h},
z^{\rm other}_{tu,h}-z^{\rm self}_{tu,h}
\right).
$$

这里的视觉差并不被声明为真实位移或接触真值。它是两端读取后的学习特征，需通过功能与行为监督获得含义。

**不把整条消息硬乘以非空匹配质量 \(w\)。**否则，当动作假设不可靠时，模型连用视觉发现重新规划或失败的机会都会被关掉。空匹配应削弱对旧horizon延续的信任，而不是禁止读取真实视觉变化。

在拿书例子中：

* 软对应可能提出“早帧远端的抬起，对应晚帧近端的抬起”；
* 视觉读取发现夹爪上移而书未动；
* 消息便有条件表达“原先的抓取—抬起假设不成立”；
* 下一轮从更新后的完整horizon重新建立对应。

这正是“核实后再汇总”的具体作用。

---

## 6. 带时间信息的窗口聚合

窗口按采样帧位置定义：

$$
\mathcal N(t)=\{u:0<|u-t|\le4\}.
$$

时间特征始终使用真实索引差 \(d_{tu}\)，因此末帧不足5步的间隔不会被误当成整步长。

对每个horizon位置和head：

$$
s^\alpha_{tu,h}
=
\frac{
Q^\alpha(\operatorname{LN}U^{(b)}_{t,h})^\top
K^\alpha(\operatorname{LN}M^{(b)}_{tu,h})
}{\sqrt{32}}
+
b^\alpha_{\rm agg}\left(\frac{d_{tu}}5\right),
$$

$$
\alpha^\alpha_{tu,h}
=
\operatorname{softmax}_{u\in\mathcal N(t)}
s^\alpha_{tu,h},
$$

$$
\widehat U_{t,h}
=
U^{(b)}_{t,h}
+
W_O
\operatorname{ConcatHeads}
\left[
\sum_u
\alpha^\alpha_{tu,h}
V^\alpha(\operatorname{LN}M^{(b)}_{tu,h})
\right],
$$

$$
U^{(b+1)}_{t,h}
=
\widehat U_{t,h}
+
\operatorname{FFN}^{(b)}
(\operatorname{LN}\widehat U_{t,h}).
$$

时间进入两个地方：

* **权重：**明确的 \(b_{\rm agg}(d_{tu}/5)\)；
* **消息Value：**\(M_{tu,h}\) 已经非线性消费signed gap和对应位移。

不额外加纯时间Value旁路，也不增加邻帧间交互模块。

### 同步更新合同

一轮内所有帧对、视觉query和邻帧聚合都读取同一个 \(U^{(b)}\)。计算完成后统一得到 \(U^{(b+1)}\)。

不得因为edge遍历顺序不同，让后处理帧对读到已经更新的状态。

四轮参数各自独立，但每轮参数在所有帧、视频和task之间共享。四轮后的局部依赖半径最多为16个采样位置；这仍不是全视频上下文，后续长程模块继续必要。

---

## 7. H-read、长程组织和多视频compiler

### 7.1 H只在这里压缩

$$
E_t
=
\operatorname{HRead}
\left(
q_H(\bar\ell),U^{(4)}_{t,0:49}
\right)
\in\mathbb R^{256}.
$$

此前任何视觉query都没有替代或删除完整H状态。

### 7.2 单视频长程组织

$$
P_{1:T}
=
\operatorname{TemporalTransformer}_{2\text{层}}
(E_{1:T};\tau).
$$

使用真实时间索引的RoPE，完整视频双向读取，保留每个时间位置的输出：

$$
P\in\mathbb R^{T\times256}.
$$

这里不再有layer轴，也不再需要层间attention。

长程模块负责把局部核实后的“接近—抓取—搬运—放置关系”组织成更长过程。它不把教师时钟直接复制为执行时钟。

### 7.3 集合compiler

每条视频独立得到 \(P^{(k)}\)，之后共同进入compiler：

$$
c_{mr}
=
\operatorname{Compiler}_{mr}
\left(\bar\ell,\{P^{(k)}_t\}_{k,t}\right),
$$

$$
c\in\mathbb R^{38\times16\times256}.
$$

保留两层整策略cross-attention＋query self-attention。集合读取给每视频相同基础质量：

$$
\operatorname{logit}_{mr,k,t}
=
\operatorname{contentScore}_{mr,k,t}-\log T_k.
$$

不设置video ordinal embedding，不跨视频建立教师时间边，不生成每视频LoRA后平均。

---

## 8. Native完整A/B读出继续采用已讨论方案

$$
h^A_{mr}=\operatorname{GELU}(U_Ac_{mr}),\qquad
h^B_{mr}=\operatorname{GELU}(U_Bc_{mr}),
$$

$$
a_{mr}=a^0_{mr}+D^A_{mr}h^A_{mr},
\qquad
b_{mr}=D^B_{mr}h^B_{mr}.
$$

选择：

* code与读出宽度均256；
* \(D^A,D^B\) 按target/rank/side独立；
* 跨task共享；
* \(D^A,D^B\) 零初始化；
* \(A^0\) 沿用现有规范identity模板；
* rank16、alpha16，生成完整76个A/B张量；
* 无独立carrier、无第二adapter。

不再重开多套decoder候选。

它提供更直接的native因子学习通道，但不消除共享干扰。对共享D，单步梯度仍具有

$$
\delta b_j
\propto
-\sum_iw_i\langle h_j,h_i\rangle g_i
$$

这样的跨condition耦合。前面的过程表示必须使共享相似性服务于控制规律，而不是把所有任务压成公共code。

P/Q clone与shared的差距、当前读出干预的有限收益，都要求保留这个边界。

---

## 9. 主要计算与激活规模

设单视频 \(T=100\)、\(H=50\)、视觉与task tokens合计 \(N=280\)、8heads、窗口半径4，则无序帧对数为

$$
E=\sum_{d=1}^{4}(T-d)=390.
$$

下表是张量规模算术，不是新图实测峰值：

| 项目                    |       新方案 |    原18层过程轴对应规模 |
| --------------------- | --------: | -------------: |
| FP32原生R缓存             | 约19.5 MiB |     约351.6 MiB |
| 单份BF16过程U             | 约2.44 MiB |     约43.95 MiB |
| 每轮唯一帧对score元素         |      780万 |        约1.404亿 |
| 新方案逐对双向、两端视觉读取score元素 | 约1.747亿/轮 | 不能直接按18倍比较旧读取链 |

最后一项若全部物化，仅BF16 attention scores就约333 MiB/轮，尚未计梯度与其它激活。因此逐对视觉读取是真正新增的工作，不能被“删掉18层”掩盖。

实现默认：

* 每次处理8个无序帧对，批量产生两个方向、两个端点的视觉读取；
* 每帧每轮的视觉K/V只投影一次；
* 使用高效attention；
* 对轮次和edge chunk做activation checkpoint；
* 不保存全部视觉attention矩阵作为常驻训练产物。

**没有因此消失的成本：**

* Gemma与完整Action Expert前向；
* Meta反向仍需经过完整Action Expert；
* 执行侧FM与十步flow重放；
* 大型native D的参数、梯度、Adam状态和跨卡通信。

256维native读出末层约3.295亿参数，FP32参数、梯度及两个Adam moment合计约4.91 GiB。省掉过程层轴会明显减轻一部分负担，但不等于整个系统缩小18倍。

---

# 三、与最终架构一致的 FM＋Writer RL 协议

## 1. 主选择：严格on-policy，一批轨迹只贡献一次联合梯度

记全部共享可训练参数为

$$
\psi=(\phi,\mu).
$$

第 \(n\) 个训练迭代开始时，冻结版本

$$
\psi_n.
$$

本迭代顺序固定为：

```text
冻结ψn
→ 采样教学条件与FM queries
→ 用ψn生成LoRA
→ 用这些LoRA采集完整rollouts
→ 在ψn上计算FM梯度和RL梯度
→ 合并所有共享梯度
→ 形成一个受约束的optimizer proposal
→ 接受一次更新，得到ψn+1
```

**在rollout采集期间，不先做FM optimizer step。**允许并行计算FM梯度，但只能累积，不能改变参数。

旧轨迹在这次联合更新后：

* 保留为正式证据；
* 可以用于统计和分析；
* **不再用于下一版本的普通score梯度。**

本首版不做多epoch PPO式复用，也不引入重要性采样比率。这样直接消除“FM已经改了参数，却继续把旧rollout当当前策略”的不一致。

后面的KL检查只约束候选步幅，**不承担任何采样分布修正作用**。

---

## 2. 首版采样与优化默认

### 2.1 数据与批量

| 项目                     | 首版选择                       |
| ---------------------- | -------------------------- |
| 训练task                 | 固定train24，不先加入额外meta tasks |
| 每迭代task                | 每suite均匀抽1个，共4个；跨迭代重新随机抽样  |
| 每task教学condition       | 1个                         |
| K                      | 从1/2/4等概率抽样，集合内视频互异        |
| teacher视频池             | 沿用0–15                     |
| FM action episodes     | 沿用16–41，与teacher角色互斥       |
| 每condition FM queries  | 64个，episode均匀后frame均匀      |
| 每condition RL episodes | 4条                         |
| 每episode RL反传decision  | 均匀抽取 \(\min(16,Q)\) 个      |
| optimizer更新            | 整个迭代只更新一次                  |

每suite六个训练task，因此上述分层采样配合四个task等权，对固定train24均匀目标给出无偏的task采样估计。它不保证每六次更新恰好覆盖全部task；覆盖按实际访问次数记录，不把GPU数量或update数冒充task曝光。

RL执行初态建议在新run中固定为训练任务states0–31，训练侧行为诊断使用32–49。这里仅声明**新run内部互斥**，不把它们说成项目历史上从未使用过的状态；validation8/test划分不变。

### 2.2 联合目标

首版主目标：

$$
L(\psi)
=
L_{\rm FM}(\psi)-0.1J_\Sigma(\psi).
$$

FM系数1，RL系数0.1，是明确的工程默认，不是理论最优权重。RL从第一次联合更新就启用，不等待FM失败或接近145。

优化默认：

* AdamW；
* peak LR \(3\times10^{-5}\)；
* 前8个接受的联合更新线性warmup，之后先保持常数；
* betas \(=(0.9,0.95)\)，eps \(=10^{-8}\)；
* weight decay \(=10^{-4}\)；
* 全部FM＋RL共享梯度合并后，global norm clip为1；
* 不分别把FM/RL或各task梯度归一到同一norm；
* source参数始终冻结。

不在总训练长度未登记前使用一个名义“完整cosine schedule”。曝光与继续投入由真实行为节点裁决。

---

## 3. FM目标与梯度

对归一化动作chunk \(a\)、噪声 \(\epsilon^{\rm FM}\) 和flow time \(s\)：

$$
x_s=(1-s)a+s\epsilon^{\rm FM},
\qquad
u_s=\epsilon^{\rm FM}-a.
$$

$$
L_{\rm FM}
=
\mathbb E_{i,\mathcal C,o,a,s,\epsilon^{\rm FM}}
\frac1{50\cdot7}
\left\|
v_{\theta_0+G_\psi(\mathcal C)}(o,\ell,x_s,s)-u_s
\right\|^2.
$$

沿用既有normalization、flow-time分布及动作补齐合同，不在本次主架构变更中顺便修改padding目标。

当前processor以冻结分位数变换处理state/action，执行query可读自己的state；teacher路径不读该state。两条信息流继续分开。

---

## 4. RL实际概率：flow noise是外生变量，额外动作探索有显式密度

### 4.1 一次重新规划

对decision \(q\)，记录观测 \(o_q\)，抽取原生flow noise：

$$
\epsilon_q\sim\mathcal N(0,I_{50\times32}).
$$

通过实际十步Euler flow：

$$
x_0=\epsilon_q,
$$

$$
x_{k+1}
=
x_k-0.1\,
v_{\theta_0+\Lambda_i}(o_q,\ell,x_k,1-0.1k),
\qquad k=0,\ldots,9,
$$

其中

$$
\Lambda_i=G_{\psi_n}(\mathcal C_i)
$$

是本condition固定的完整LoRA。

取得归一化前五步动作均值：

$$
m_{\psi_n,q}
=
\operatorname{vec}(x_{10}[0:5,0:7])
\in\mathbb R^{35}.
$$

然后：

$$
z_q\sim\mathcal N(m_{\psi_n,q},\Sigma).
$$

注意：必须完整积分50×32，再截取前5×7。不能把flow内部提前缩成5×7，否则执行函数已经改变。

原生LeRobot实现确有独立噪声初始化、逐步Euler更新，以及整体带`no_grad`的采样包装；训练重放必须复用其数学路径，而不能直接依赖该无梯度包装。

### 4.2 可用的对数概率

$$
\log q_\psi(z_q\mid o_q,\mathcal C_i,\epsilon_q)
=
-\frac12(z_q-m_{\psi,q})^\top
\Sigma^{-1}(z_q-m_{\psi,q})
-\frac12\log|\Sigma|-\frac{35}{2}\log(2\pi).
$$

原生flow noise的密度 \(p(\epsilon_q)\) 与 \(\psi\) 无关。

因此，这里使用的是**扩展动作变量 \((\epsilon_q,z_q)\) 的联合策略**。不需要求原生flow输出的边缘密度，不把FM loss当log probability。

score恒等式给出：

$$
\nabla_\psi J_\Sigma
=
\mathbb E\left[
\sum_q
A_q
\nabla_\psi
\log q_\psi(z_q\mid o_q,\mathcal C,\epsilon_q)
\right].
$$

这是对本探索策略的policy gradient，不是对部署 \(J_0\) 的无条件等价替代。([NeurIPS 会议录][1])

---

## 5. Reward、baseline与三层权重

### 5.1 Reward

首版只使用官方episode成功：

$$
Y_e\in\{0,1\}.
$$

* 无人工阶段reward；
* 无视频预测reward；
* 无未经验证的视觉语言进度打分；
* 无默认privileged critic；
* 无默认expert纠错。

成功终止或suite horizon终止，均按现有环境规则处理；有限时域超时计失败，不做bootstrap。

对未折扣的最终成功目标，每个已执行decision的Monte Carlo return为同一个 \(Y_e\)。

### 5.2 Baseline：同condition的leave-one-out成功均值

每condition采集4条相互独立episode：

$$
b_{i,e}
=
\frac1{3}\sum_{e'\ne e}Y_{i,e'},
$$

$$
A_{i,e}=Y_{i,e}-b_{i,e}.
$$

这些量全部stop-gradient。

要求四条episode的初态抽样、环境随机流、原生flow noise和额外探索噪声彼此独立；不能使用依赖自身动作或自身结果的“baseline”。

在给定condition和冻结策略的条件下，其它episode的回报不依赖本episode的当前探索动作，因此baseline不改变score梯度的期望。

这个选择避免首版再增加一套尚需训练和校准的critic。

### 5.3 稀疏成功时，实际有什么信号

若同condition四条episode全部失败：

$$
Y_e=0,\quad b_e=0,\quad A_e=0.
$$

**这组RL梯度就是零。**FM仍提供稠密动作学习，不能把“代码里有RL分支”写成“RL已经有效学习”。

若全部成功，leave-one-out advantage同样为零，FM继续提供动作约束。

如果某些任务长期没有混合成功／失败组，需要判断的是探索支持或基础能力不足，而不是人为制造非零reward。可靠expert纠错或critic可以在出现具体必要性后另行加入，但不预装进首版。

GOMQ已经使用过成功expert occupancy和前五动作蒸馏，不能把这些重新命名为本次RL；本次新增的是当前learner状态上的显式探索概率信用，并更新完整共享映射。

### 5.4 Task、episode、decision权重

完整RL估计为：

$$
\widehat g_{\rm RL}
=
\frac14
\sum_{i=1}^{4}
\frac14
\sum_{e=1}^{4}
A_{i,e}
\sum_{q=1}^{Q_{i,e}}
\nabla_\psi\log q_{\psi_n}(z_q\mid o_q,\mathcal C_i,\epsilon_q).
$$

即：

* task平均；
* task内episode平均；
* episode内decision求和。

不能再除以 \(Q_{i,e}\)，否则会改变不同轨迹长度的目标权重。

---

## 6. 降低十步flow反传成本：无偏decision子采样

每episode从 \(Q\) 个decision中均匀无放回抽取

$$
M=\min(16,Q)
$$

个，集合记为 \(\mathcal S\)。

使用：

$$
\sum_{q=1}^{Q}g_q
\quad\longrightarrow\quad
\frac{Q}{M}\sum_{q\in\mathcal S}g_q.
$$

条件于已采集轨迹，这仍是完整decision总和的无偏估计。

可以在采集时使用与reward无关的reservoir sampling，只保留最多16组完整观测、\(\epsilon,z,m_{\rm old}\)，其余保留必要标量证据。不能按“最大动作分歧”“成功前最后几步”等规则选点后仍使用上述无偏公式。

这降低反传工作量，但增加方差；它不增加新信息，也不保证关键接触点每条轨迹都被选到。

---

## 7. 实际可微重放与LoRA cotangent累积

对保存的decision：

* \(o_q\)：固定；
* \(\epsilon_q\)：固定；
* \(z_q\)：固定；
* \(m_{\rm old,q}\)：采集时保存；
* \(Y,b,A\)：全部固定。

在仍未更新的 \(\psi_n\) 上，重放实际十步flow得到 \(m_{\psi_n,q}\)。

对RL最小化loss，动作均值cotangent为：

$$
g_{m_q}
=
-0.1\,
\frac1{4\cdot4}
\frac{Q}{M}
A_{i,e}\,
\Sigma^{-1}(z_q-m_{\rm old,q}).
$$

于是：

$$
g_{\Lambda_i}^{\rm RL}
=
\sum_{e,q\in\mathcal S}
J_{m_q,\Lambda_i}^{\top}g_{m_q}.
$$

FM同样产生：

$$
g_{\Lambda_i}^{\rm FM}
=
\frac14\nabla_{\Lambda_i}L_{{\rm FM},i}.
$$

先合并：

$$
g_{\Lambda_i}
=
g_{\Lambda_i}^{\rm FM}+g_{\Lambda_i}^{\rm RL}.
$$

再对本condition只做一次Writer重放：

$$
g_{\Lambda_i}
\rightarrow
g_{\phi,i},g_{R,i}
\rightarrow
g_{\mu,i}.
$$

### 必须保留的链式依赖

十步flow可以按步checkpoint，但不能在步间detach \(x_k\)。否则遗漏

$$
x_k\rightarrow v(x_k)\rightarrow x_{k+1}
$$

的连续依赖，得到的就不是当前动作均值的完整梯度。

执行prefix由冻结图文基础网络产生，且LoRA目标不在该prefix内，可以对固定观测复用；Action Expert后缀和flow状态则必须按当前生成A/B计算。

不对环境状态反向传播。policy-gradient已通过动作score处理状态分布的影响；保存观测在重放中视为常量是正确做法。

---

## 8. 反归一化、裁剪、夹爪与提前终止

探索发生在**归一化的前五步7维动作**上。

执行映射为：

$$
a^{\rm raw}
=
\frac{z+1}{2}
(q_{99}-q_{01}+\varepsilon_{\rm norm})+q_{01},
$$

然后严格调用现有canonical evaluator的动作范围与夹爪处理，不新增sign、二值阈值或额外tanh。冻结分位数变换的实现已核对。

计算log probability使用裁剪前的 \(z\)，不是裁剪后的动作。

因为整个 \(z\in\mathbb R^{35}\) 在decision时已经抽取，即使执行到第2个动作就成功终止，仍可使用该35维联合score。未执行尾部是扩展随机动作的一部分，在期望中不会产生错误信用；它可能增加方差，但不能依据终止结果随意删去部分密度项，尤其当五步噪声相关时。

dummy settling动作不来自该探索策略，不计入RL score。

---

## 9. 探索协方差：首版固定，不自动退火

选择：

$$
\Sigma=C_\rho\otimes
\operatorname{diag}(\sigma_1^2,\ldots,\sigma_7^2),
$$

$$
(C_\rho)_{ab}=\rho^{|a-b|},\qquad a,b=0,\ldots,4,
$$

$$
\rho=0.8,
$$

$$
\sigma_{1:6}=0.05,\qquad \sigma_7=0.10.
$$

这里的标准差都是冻结action normalization后的单位：

* 前六维0.05，对应各维分位数跨度约2.5%的物理扰动标准差；
* 夹爪维0.10，对应其跨度约5%；
* 五步内相关，使扰动较连续；
* 不同重新规划之间独立，避免额外引入跨decision噪声状态及其条件密度。

**首个正式训练协议固定 \(\Sigma\)，不自动退火，也不学习其参数。**这样优化目标 \(J_\Sigma\) 在该段训练内明确稳定。

固定噪声不是部署保证。FM始终作用于无额外探索扰动的原生flow函数，为 \(J_0\) 提供动作层面的学习；还必须直接检查去掉额外噪声后的闭环能力。

建议每24个接受的联合更新，在训练侧保留初态上做固定的 \(\Sigma\)／0配对诊断：相同task、教学条件、初态和原生flow-noise schedule，仅改变额外探索噪声。两种执行都保留原生flow noise。

若只提高 \(J_\Sigma\)，而 \(J_0\) 不改善，训练—部署衔接就是未通过。不能在看到validation后临时改变噪声再称同一合同通过。

---

## 10. Trust约束：限制一次候选步，不假装修正off-policy

只在 \(\psi_n\) 处计算一次FM＋RL梯度：

$$
g_n=\nabla L_{\rm FM}(\psi_n)-0.1\widehat g_{\rm RL}.
$$

经联合clip后，AdamW形成完整候选方向 \(d_n\)。

对候选

$$
\psi'=\psi_n+\alpha d_n
$$

检查保存观测和flow noise上的条件高斯KL：

$$
K_q(\psi')
=
\frac12
(m_{\psi',q}-m_{{\rm old},q})^\top
\Sigma^{-1}
(m_{\psi',q}-m_{{\rm old},q}).
$$

每episode从已保存decision中再均匀取最多4个用于该检查，按episode、task平均。首版限制：

$$
\max_{\text{本批task }i}\widehat K_i\le0.02.
$$

这是**每个完整35维动作块的平均KL**，不是每维0.02，也不是整条轨迹KL。

使用有限候选：

$$
\alpha\in\{1,\tfrac12,\tfrac14,\tfrac18\}.
$$

按顺序接受第一个满足条件的候选；全部失败则本次不更新参数，Adam状态也回滚。接受缩放步时，提交本次梯度形成的moments，并把实际参数步统一缩放；不分别缩放某个模块。

这里有三个重要边界：

1. 不能简单在旧参数处加 \(D_{\rm KL}(\pi_{\rm old}\|\pi_\psi)\) 然后认为单步已经受到约束——该KL在旧点的一阶梯度为零。
2. 候选KL检查只做前向，不在候选点重新用旧轨迹计算普通score梯度。
3. 这是有限样本旧状态上的步幅约束，不保证真实成功单调提升，也不能代替相邻行为保持。

原始score梯度在指定采样分布下无偏；经过clip、Adam和接受规则后的实际更新，不再被描述为“无偏梯度更新”。两者需要区分。

---

## 11. 一次完整训练迭代的伪代码

```python
# 固定默认：
# 4 tasks：每suite独立均匀抽1个
# 每condition 64 FM queries、4 RL episodes
# 每episode最多16个均匀抽样decision参与RL反传
# lambda_FM=1.0, lambda_RL=0.1
# 全迭代只允许一次参数更新

def joint_iteration(writer, meta, frozen_source, optimizer, sampler):
    old_state = snapshot_parameters_and_optimizer(writer, meta, optimizer)
    version = old_state.version
    freeze_parameter_version(version)       # 禁止采集中穿插FM更新

    tasks = sampler.one_uniform_train_task_per_suite()
    task_weight = 1.0 / 4.0
    records = []

    # 1. 用同一版本生成条件LoRA并采集数据
    for task in tasks:
        K = sampler.uniform_choice([1, 2, 4])
        condition = sampler.correct_videos(
            task, K=K, pool=range(0, 16), distinct_within_set=True
        )
        fm_queries = sampler.cross_episode_queries(
            task, episodes=range(16, 42), count=64
        )

        # prefix/Z可跨step缓存；R只属于当前Meta版本
        prefix_Z = frozen_prefix_and_visual_states(condition)
        R = final_post_norm_horizon(
            frozen_source, meta, prefix_Z, fixed_probe, flow_time=1
        )
        lora_old = writer_no_grad(R, prefix_Z.Z, condition.language)

        episodes = []
        for e in range(4):
            env = reset_authorized_train_env(
                task,
                initial_state=sampler.uniform_choice(range(0, 32)),
                independent_rng=True,
            )
            decision_reservoir = UniformReservoir(capacity=16)
            Q = 0

            while not env.terminated:
                obs = env.observation()
                epsilon = independent_standard_normal(shape=(50, 32))
                m_old = native_flow10_no_grad(
                    frozen_source, lora_old, obs, epsilon
                )[:5, :7].reshape(35)

                z = conditional_gaussian_sample(m_old, Sigma)
                logq_old = gaussian_log_prob(z, m_old, Sigma)

                decision_reservoir.observe(
                    observation=obs,
                    epsilon=epsilon,
                    z=z,
                    m_old=m_old,
                    logq_old=logq_old,
                    policy_version=version,
                )
                Q += 1
                env.execute_canonical_first_five(z)  # 成功可提前终止

            episodes.append({
                "Y": official_success(env),
                "Q": Q,
                "decisions": decision_reservoir.items(),
            })

        records.append((condition, prefix_Z, R, lora_old, fm_queries, episodes))

    assert_parameter_version_unchanged(version)
    zero_all_shared_gradients()

    # 2. FM与RL都在ψ_old上求LoRA cotangent
    for condition, prefix_Z, R, lora_old, fm_queries, episodes in records:
        L_leaf = detached_complete_lora_leaves(lora_old)

        g_L = task_weight * functional_FM_lora_gradient(
            frozen_source, L_leaf, fm_queries
        )

        rewards = [ep["Y"] for ep in episodes]
        for e, ep in enumerate(episodes):
            baseline = mean(rewards[e2] for e2 in range(4) if e2 != e)
            advantage = stop_gradient(rewards[e] - baseline)
            M = len(ep["decisions"])

            for item in ep["decisions"]:
                m = differentiable_native_flow10(
                    frozen_source,
                    L_leaf,
                    stop_gradient(item.observation),
                    stop_gradient(item.epsilon),
                )[:5, :7].reshape(35)

                coefficient = (
                    -0.1 * task_weight / 4.0
                    * ep["Q"] / M
                    * advantage
                )
                cotangent_m = coefficient * solve(
                    Sigma, stop_gradient(item.z - item.m_old)
                )
                g_L += VJP(m, L_leaf, cotangent_m)

        # 3. 每condition只做一次Writer与Meta反传
        g_R = writer_replay_backward(
            writer, R, prefix_Z.Z, condition.language, g_L
        )
        observer_replay_backward(
            frozen_source, meta, prefix_Z, fixed_probe, g_R
        )

    # 所有权重已包含于cotangent；跨卡只SUM，不再除world_size
    distributed_sum_shared_gradients()
    clip_joint_gradient_norm(max_norm=1.0)

    # 4. 只形成一个AdamW方向，不在候选点重新算旧轨迹score
    proposal, proposed_moments = adamw_proposal_without_commit(optimizer)

    accepted = False
    for alpha in [1.0, 0.5, 0.25, 0.125]:
        candidate = old_state.parameters + alpha * proposal
        K_by_task = estimate_conditional_action_KL(
            candidate, records, Sigma, decisions_per_episode=4
        )
        if finite(candidate) and max(K_by_task.values()) <= 0.02:
            commit(candidate, proposed_moments)
            accepted = True
            break

    if not accepted:
        restore_parameters_and_optimizer(old_state)

    seal_iteration_evidence(records, accepted)
    discard_from_future_actor_training(records)   # 不跨版本复用
```

伪代码中的decision replay可以按microbatch批量执行；它不要求逐decision进行Python小调用。各condition的LoRA cotangent也可以分块累积，避免保存多条十步flow反向图。

---

# 四、训练成本、验证边界与少数实验问题

## 1. 首版训练成本应按真实调用量理解

上述一次联合迭代包含：

* 4个教学condition；
* 256个FM queries；
* 16条on-policy episodes；
* 最多256个RL decision重放；
* 每个RL decision是完整十步flow。

因此，仅RL重放的向量场计算量上限约为2,560个query-step，尚未计反向重算、采集rollout和trust候选前向。它明显不同于“又加一个FM batch”。

主要节省来自：

* 同condition的LoRA服务4条episode；
* 轨迹中只均匀保留有限decision做反传；
* 先累积A/B cotangent，再统一反传Writer/Meta；
* 冻结执行prefix在十步flow中复用；
* 不跨多个RL epochs反复使用同一轨迹；
* 末层过程轴替代18层过程轴。

trust检查最多使用64个decision的前向，通常先测试完整步；它仍有实际代价，必须计入吞吐，而不是隐藏在“优化器开销”里。

---

## 2. 实现验证只围绕真正的合同

首批工程核验应集中于：

**接口正确性：**读取点确为最终归一化后、动作投影前；没有先收集18层；最终H-read前未压缩H。

**依赖正确性：**视觉query确实依赖帧对软对应；视觉核实发生在窗口聚合前；四轮同步读取旧状态；邻帧连同时间标签重排保持集合语义。

**RL正确性：**采集与FM/RL求梯度时参数版本不变；Gaussian log probability对应裁剪前完整 \(z\)；flow步间未detach；decision子采样有 \(Q/M\) 权重；task/episode权重只乘一次。

**缓存与重放：**候选Meta变化后不复用旧R；冻结Z/KV可复用；十步flow的普通执行与可微路径在允许数值误差内表达同一函数。

这些通过后就进入行为学习，不再以“还没有证明末层充分”为由回到无止境架构讨论。

---

## 3. 仍需实验回答的三个主要问题

### 问题一：末层完整H是否提供了足够的可学习动作证据

这是本次明确接受的信息取舍。若新图能获得广泛训练任务能力，就没有理由立即恢复18层。

只有出现具体反证，例如某些任务在可靠监督和充足学习信号下持续缺少基本能力，并且诊断指向末层丢失的可用信息，才重新讨论层访问；不是先并行维护两套主图。

### 问题二：逐对视觉核实是否真正改善了控制

要看真实行为中错误阶段转换、抓取失败后继续搬运、过早释放等现象是否减少，以及跨视频和相邻checkpoint是否保持。

对应矩阵、attention或空匹配看起来更合理，只是解释材料，不能替代行为。

### 问题三：Writer RL是否产生了可部署、可保持的增益

需要同时看到：

* RL确有非零、非纯噪声的成功信用；
* \(J_\Sigma\) 的增益能保留到无额外噪声的 \(J_0\)；
* 训练task获取、同task新视频、未见task迁移能被区分；
* 容易任务收益没有持续换走Goal/Long；
* 相邻成功集合不是高churn换手。

如果长期全失败，首版会如实显示RL信号不足，不能靠一个额外critic名称掩盖。届时需要针对探索或可靠纠错支持作决定，而非自动重推整个架构。

---

## 4. 正式裁决边界不变

当前方案不改变：

* teacher部署信息墙；
* source冻结；
* Writer与读取侧Meta fresh联合训练；
* 单次生成一套完整38-target LoRA；
* 固定24/8/8划分；
* rollout期间不重看视频、不做task-local优化；
* validation/test零梯度；
* shuffled/reversed只用于selected checkpoint冻结后的最终controls。

正式选择仍要求validation8 strict paired single-checkpoint correct严格大于145/400，并满足既有相邻稳定、低churn、breadth、四suite、Goal/Long、跨视频和最终因果合同。80-row screen、内部loss和checkpoint union不选择最终模型。

---

## 定稿结论

**架构上，采纳你的四点：末层完整H、一次关系紧接一次视觉核实、逐对核实后聚合、聚合显式保留signed time。首版重复四轮，不保留18层融合或另一条先汇总再读视觉的并行路径。**

**训练上，采用fresh FM＋Writer RL，从同一冻结参数版本采集和求梯度，一批轨迹只用于一次联合更新；使用显式条件高斯动作探索、leave-one-out baseline、无偏decision子采样、完整十步flow重放和有限候选KL步幅约束。**

这已经是一套可以落实的单一主方案。尚待证明的是它能否形成足够强、稳定且有视频增量的共享控制能力；这些问题应由接下来的有效实验回答，不再作为继续悬置计算定义的理由。

[1]: https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html "https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html"
