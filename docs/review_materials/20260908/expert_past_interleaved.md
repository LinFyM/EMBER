# EMBER 聚焦定稿：过去定向局部关系、完整H查询与局部—长程交替

**本轮推荐采用一套明确的主方案：**

**过去四帧的单向对应 → 沿完整H联合形成视觉查询 → 逐帧对读取两端画面 → 按历史起点从早到晚，用短GRU组织经核实的关系消息 → 更新完整H → 临时H-read与双向长程组织 → 按当前horizon状态回写 → 重复四组 → 集合compiler与完整A/B。**

这里有两个必须明确的性质：

**局部关系边只指向过去；长程层采用双向读取。**因此第一组局部计算只使用当前及过去画面，从第二组开始，局部查询可以通过长程反馈携带全视频上下文。它是一种“过去定向的局部关系＋离线全视频修正”，**不是整个网络前缀因果**。

数学分析能够说明这些选择新增了哪些依赖、哪些操作等价、哪些限制仍然存在；历史证据约束其预期。它们不能在没有行为比较的情况下证明全局性能最优，但已经足以选定首版，不需要继续保留并行架构候选。

---

## 一、从任务需要推导三个关键选择

继续用“拿起书，放入收纳架后侧隔间”解释。

当前帧 \(t\) 中，书可能已经随夹爪运动，也可能仍在桌上。来自较早帧的消息和较近帧的消息分别提供：

* 从较早起点看，是否发生了“接近到抬起”的整体变化；
* 从较近起点看，是否已经建立稳定抓取，还是仅机械臂动了；
* 这些判断是否相互支持，或需要修正。

这些消息覆盖的区间是

$$
[u_1,t],\ [u_2,t],\ldots,[u_L,t],
\qquad u_1<u_2<\cdots<u_L<t.
$$

**它们不是首尾相接的动作片段，而是以同一当前帧为终点、不同历史起点为参照的重叠证据。**

所以本轮真正需要的三个计算能力是：

1. 从完整horizon的对应模式，而不只从一个位置，决定视觉查询；
2. 在接受某条关系消息时，能够考虑已经读取的其它历史消息；
3. 让全局过程解释影响下一轮“怎样建立对应、到原始画面里看什么”，而不只影响最后的参数读出。

---

## 二、窗口内有序组织：为什么选择短GRU，而不是再加一个名义上的causal attention

### 2.1 上一版时间条件聚合已经不是“无时序平均”

上一版可概括为

$$
y=\sum_{u}\alpha_uV(M_u),
\qquad
\alpha_u=
\frac{\exp a(U_t,M_u,\Delta_u)}
{\sum_v\exp a(U_t,M_v,\Delta_v)},
$$

其中

$$
\Delta_u=\tau_t-\tau_u.
$$

只要 \(M_u\) 和logit包含真实时间，它就能区别：

“很久之前到现在发生的变化”与“刚才到现在发生的变化”。

因此，不能说加权求和天然丢失时序。多头读取和后续MLP也能间接表达不少关系。

但在**这一次聚合的固定输入处**，有

$$
\frac{\alpha_u}{\alpha_v}
=
\exp\left[a(U_t,M_u,\Delta_u)-a(U_t,M_v,\Delta_v)\right].
$$

该相对权重不直接取决于第三条消息 \(M_w\)。Value变换也逐条独立计算。

这意味着：它没有在本次聚合内部显式实现

> “先看较早消息，再根据已形成的判断，决定较近消息是补充、重复还是反证。”

不是说整个旧网络绝对不能表达这件事，而是这项交互被留给其它层、多头分工或前面已经压缩的 \(U_t\) 间接承担。

当前代码已经有时间与相对对应分布进入消息、再进行邻居attention的结构，不能把它误写成完全无时间处理。

### 2.2 一层causal self-attention只取最后token，mask究竟增加了什么

设四条消息为 \(x_1,\ldots,x_L\)，一层self-attention的最后位置是

$$
y_L=
\sum_{i=1}^{L}
\operatorname{softmax}_i
\left(
\frac{q(x_L)^\top k(x_i)}{\sqrt d}+b_{Li}
\right)v(x_i).
$$

对于最后位置，causal mask允许它读取全部 \(1,\ldots,L\)。在相同输入、参数和位置处理下，**这一行与不加causal mask的self-attention完全相同**。第一层其它位置的输出尚未成为这一行的Key/Value；只取最后输出时，mask在那些行上的区别不会被消费。这由标准attention的并行计算定义直接得到。([数字对象识别][1])

两层则不同：

$$
y_i^{(1)}=f(x_{1:i}),
\qquad
y_L^{(2)}=g(y_1^{(1)},\ldots,y_L^{(1)}).
$$

第二层能够读取多个已经形成的前缀表示。此时causal mask提供了真实的前缀分工，而不只是名称。

但当前最多四条消息。为了获得这种前缀条件化，没有必要默认引入两层decoder式Transformer。

### 2.3 首版选择：长度最多4的有序GRU

采用

$$
c_i=\operatorname{GRU}(x_i,c_{i-1}),
\qquad i=1,\ldots,L.
$$

状态 \(c_i\) 表示：

> 已经综合了前 \(i\) 个历史起点后，对当前帧过程状态形成的内部判断。

它不是机器人状态积分，也不是把区间动作累加。

它提供明确的非交换运算。即使简化成固定门值 \(1/2\)：

$$
c_2=\tfrac14c_0+\tfrac14n_1+\tfrac12n_2,
$$

交换消息顺序后：

$$
c'_2=\tfrac14c_0+\tfrac14n_2+\tfrac12n_1.
$$

一般有 \(c_2\ne c'_2\)。实际门值还依赖此前状态，所以模型可以让同一条较近消息，在不同历史证据下产生不同修正。

GRU的门控递推是现成、定义明确的算子；本方案不为它额外赋予“已经理解因果”的含义。([PyTorch Docs][2])

**选择它的具体理由是：**

* 最多四步，顺序链很短；
* 每一步显式依赖已读消息；
* 可以选择保留或替换已有判断，而非只有归一化加权；
* 所有 \((t,h)\) 可同时批量计算，只有四个邻居位置之间顺序执行；
* 不需要另加窗口attention、窗口Transformer和窗口readout三套操作。

重叠消息的重复计数不会被GRU数学上自动消除；模型仍须学会冗余处理。不过它至少不被定义成“把这些区间变化直接相加”。

---

## 三、完整H怎样真正参与视觉动态理解

### 3.1 原有逐 \(h\) 查询确实已经包含序列信息

原生 \(R_{t,h}\) 来自有跨horizon交互的Action Expert；匹配内容又是

$$
m_{tu,h}=\sum_g\Pi_{tu}(h,g)V(U_{u,g}).
$$

因此不能说它完全不理解其它动作位置。

但若视觉query仅为逐位置函数

$$
q_{tu,h}=f(U_{t,h},m_{tu,h},r_{tu,h},\ldots),
$$

那么，在固定当前输入状态 \(U\) 的一次计算里，改变另一行对应分布 \(\Pi_{tu}(h',:)\)，通常没有到 \(q_{tu,h}\) 的直接路径：

$$
\frac{\partial q_{tu,h}}
{\partial \Pi_{tu}(h',g)}
=0,\qquad h'\ne h.
$$

这正是现有接口未明确处理的部分：

**新计算出来的整条对应模式，尚未被联合用来形成视觉query。**

例如，两组帧对可以在位置 \(h=10\) 给出相似匹配内容，却在后面的horizon上分别表现为：

“持续延续到搬运”与“多处不匹配、需要重新规划”。

逐行query不必能直接利用这种差别。

### 3.2 首版增加一个职责明确的H-query block

对每个帧对，先形成50个基础关系token，再做**一层完整H self-attention＋FFN**：

$$
Q_{tu,0:H-1}
=
\operatorname{HQueryBlock}
(X_{tu,0:H-1}).
$$

使用horizon位置的RoPE，输出仍为50个query，不做pooling。

于是

$$
q_{tu,h}
=
f_h\big(X_{tu,0},\ldots,X_{tu,49}\big),
$$

且一般可以有

$$
\frac{\partial q_{tu,h}}
{\partial\Pi_{tu}(h',g)}\ne0.
$$

它新增的不是另一个没有职责的时序模块，而是：

> **在查询画面之前，把刚计算出的匹配内容、位移分布和不匹配模式沿完整动作时间轴联合解释一次。**

H-query采用双向H attention。H是同一次静态观测下的动作位置轴，不是教学视频的未来帧；这里没有必要施加teacher-time causal mask。

这一层不能保证形成真实阶段或速度解释。它只是补齐一条当前缺失的直接计算依赖。

---

## 四、局部与长程：采用交替，并明确长程双向

### 4.1 为什么不继续等到最后才做长程组织

若旧链条是

$$
E_t=f(\text{局部响应与局部视觉}),
\qquad
P=g(E_{1:T}),
$$

那么长程模块只能操作已经压缩后的 \(E_t\)。如果局部视觉读取遗漏了某个对象或接触证据，长程模块不能从丢失的内容中重新读取它。

交替结构增加了：

$$
P^{(b)}
\rightarrow U^{(b)}
\rightarrow Q^{(b+1)}_{tu}
\rightarrow Z_t,Z_u.
$$

因此，后段过程提出的解释能够改变下一轮对原始画面的读取，而不只改变最后的compiler权重。

以拿书为例：后段画面显示书始终没有随夹爪移动，全局表示便有条件提示下一轮更仔细地检查早期接触和夹爪附近的书，而不是继续沿用第一次局部读取的“已抓住”判断。

这是新增的**条件化信息访问能力**，不是新增外部信息。

### 4.2 为什么不直接把长程向量加到所有H位置

简单广播

$$
U_{t,h}^+=U_{t,h}+WP_t
$$

给所有horizon位置增加相同向量，没有显式的当前位置—全局条件交互。

另一方面，若对每个 \(h\) 使用只有一个Key的cross-attention：

$$
\operatorname{Attn}(q_h,k_t,v_t),
$$

softmax恒为1，输出仍不依赖 \(q_h\)。所以“单个全局token cross-attention回写”不能自动解决这个问题。

本方案采用**依赖当前H状态的非线性回写**，具体公式见后文。

### 4.3 长程双向意味着什么

首版长程层读取同一视频全部 \(t=0,\ldots,T-1\)。

于是：

* 第一组局部消息只由两端及过去窗口构成；
* 第一组长程输出可以依赖全视频；
* 从第二组起，局部查询可通过回写携带未来教学画面的信息；
* 直接读取的视觉端点仍只有 \(Z_u,Z_t\)，其中 \(u<t\)。

这符合完整教学视频在编译前可用的条件，适合做离线理解和反复核实。

**不能再把第二组以后的 \(M_{tu}\) 称为仅由前缀得到的因果估计。**它描述的是方向为 \(u\to t\) 的关系，但使用了离线全视频上下文。

P/Q历史上已经做过过程与策略表示交互，且共享模型仍明显弱于同图clone。这说明交互本身不是性能保证。本次值得采用的具体变化，是全局信息返回完整H，并重新影响**逐帧对、逐H的原始视觉读取**，而不是仅在压缩代码之间继续交换。P/Q的3/20对14/20只证明特定共享学习存在代价，不能证明或否定这条新读取依赖。

---

# 五、唯一推荐计算图与完整定义

## 1. 数据流

```text
exact language + K条独立有序正确视频
    ↓
冻结图文prefix
    ├─ 逐帧图文证据 Z[k,t,token,2048]
    └─ prefix KV
          ↓
单固定probe + Action Expert + 观察侧Meta
          ↓
最终归一化后、action_out_proj前：
R[k,t,50,1024]
          ↓
U⁰[k,t,50,256]

对每条视频独立执行四组：

    当前t只与过去四个实际存在的u建立对应
        ↓
    当前H位置h → 前序H位置g的软对应
        ↓
    匹配内容、位移分布、非空质量、真实gap
        ↓
    一层沿完整H的H-query block
        ↓
    50个帧对query分别读取Z_u、Z_t
        ↓
    经视觉核实的M[t,u,h]
        ↓
    对固定(t,h)，按u从早到晚用短GRU读取
        ↓
    更新完整H：Ũᵇ[t,h]
        ↓
    临时H-read：Eᵇ[t]
        ↓
    一层双向长程temporal Transformer：Pᵇ[t]
        ↓
    前三组：按Ũᵇ[t,h]与Pᵇ[t]非线性回写，进入下一组
    第四组：不再回写，P⁴进入compiler

    ↓
多视频集合Compiler，38×16个paired queries
    ↓
完整native A/B读出
    ↓
唯一38-target rank16 LoRA
    ↓
冻结source依据自己的观测闭环执行
```

不恢复18层表示，不同时保留旧窗口attention，不额外增加一套终局长程层。

---

## 2. 记号与输入

每条视频含 \(T\) 个采样帧。首版固定：

$$
H=50,\quad d_R=1024,\quad d=256,\quad A_{\rm heads}=8.
$$

$$
R_t\in\mathbb R^{H\times1024},
\qquad
Z_t\in\mathbb R^{N_t\times2048}.
$$

\(R_t\) 是实际送入 `action_out_proj` 的输入，处于最终原生归一化之后。source全部计算层仍运行；观察Meta的反向也仍经过完整Action Expert，只是不收集18层作为Writer输入。既有读取器收集18层，与这里的首版接口不同。

初始化：

$$
U^{(0)}_{t,h}=W_RR_{t,h}+e_h.
$$

\(e_h\) 是公开horizon位置身份，不是人工操作阶段。

下文所有局部运算都只在

$$
\mathcal N^-(t)=\{\max(0,t-4),\ldots,t-1\}
$$

上进行。

---

## 3. 时间、score方向与归一化

### 3.1 时间约定

沿用当前LIBERO实验的名义控制记录步：

$$
\tau_t=n_t,\qquad \delta_h=h,\qquad h=0,\ldots,49.
$$

\(n_t\) 为真实原始帧索引。通常是 \(0,5,10,\ldots\)，末帧可能不符合整5间隔。

对 \(u<t\)：

$$
\Delta_{tu}=\tau_t-\tau_u>0.
$$

候选对应为

$$
\tau_u+g\approx\tau_t+h
\iff
g-h-\Delta_{tu}\approx0.
$$

例如 \(\tau_t=20,\tau_u=10,h=5\)，名义对应的前序位置为 \(g=15\)。

这里的时间关系是hidden响应的**结构先验**，不是这些hidden已经预测出真实未来轨迹的证明。

### 3.2 当前为行、前序为列

对第 \(b\) 组：

$$
C^\alpha_{tu}(h,g)
=
\frac{
F_b^\alpha(\operatorname{LN}U^{(b-1)}_{t,h})^\top
F_b^\alpha(\operatorname{LN}U^{(b-1)}_{u,g})
}{\sqrt{32}}
+
B_b^\alpha
\left(
\frac{\Delta_{tu}}5,
\frac{g-h-\Delta_{tu}}{50}
\right).
$$

矩阵维度为

$$
C_{tu}\in\mathbb R^{8\times50\times50}.
$$

与上一版“前序为行、当前为列”的原始score相比，交换轴后就是对应的转置；**必须转置未归一化score，再按当前行重新归一化**。不能把已经row-softmax的概率矩阵转置后当成新条件分布。

本轮只计算

$$
t\leftarrow u.
$$

不计算反向消息，不更新 \(u\)，也没有为读取未来帧保留第二条路径。

### 3.3 首版仍采用joint-gap bias

选择小型MLP \(B_b\)，不增加显式速度混合或时间warp。

事实上，输入

$$
\left(\frac{\Delta}{5},\frac{g-h-\Delta}{50}\right)
$$

与

$$
\left(\frac{\Delta}{5},\frac{g-h}{50}\right)
$$

之间是可逆线性变换。因此在含首层仿射映射的MLP中，它们具有相同的可表达函数类。采用残差形式是为了定义清楚物理方向和尺度，**不是声称仅重写坐标就增加了新能力**。

`/5`和`/50`仅用于数值缩放，不重复计入stride。

### 3.4 空匹配

$$
[\Pi^\alpha_{tu}(h,:),\,\pi^\alpha_{\varnothing,tu}(h)]
=
\operatorname{softmax}
[C^\alpha_{tu}(h,:),\,c^\alpha_{\varnothing,tu}(h)].
$$

空匹配logit由当前端状态与真实gap产生。首版保留一个空匹配项，不使用人工“远端低置信”衰减。

它允许“前序horizon中没有合适对应”，但不是已校准置信度。后续也不使用它硬关闭整条视觉消息。

---

## 4. 从软对应到完整H视觉query

形成匹配内容：

$$
m_{tu,h}
=
W_m\operatorname{Concat}_\alpha
\left[
\sum_g\Pi^\alpha_{tu}(h,g)
V_b^\alpha(U^{(b-1)}_{u,g})
\right].
$$

非空质量：

$$
w^\alpha_{tu,h}=\sum_g\Pi^\alpha_{tu}(h,g).
$$

相对位移分布：

$$
\rho^\alpha_{tu,h}(q)
=
\sum_{g:g-h=q}\Pi^\alpha_{tu}(h,g),
\quad q\in[-49,49].
$$

用一个learned线性读取得到 \(r_{tu,h}\in\mathbb R^d\)，但不把 \(\rho\) 仅压成单个平均位移。当前源码的relative correspondence本来就具备读取位移分布的接口；本轮保留这一资产。

基础帧对token：

$$
X_{tu,h}
=
f_{Q,b}\left[
U^{(b-1)}_{t,h},
m_{tu,h},
U^{(b-1)}_{t,h}-m_{tu,h},
r_{tu,h},
w_{tu,h},
e_h,\bar\ell,
\frac{\Delta_{tu}}5
\right].
$$

方向统一为“当前减前序匹配”。但这个差仍是hidden差，不解释成物理速度。

沿完整H联合处理：

$$
Y_{tu}
=
X_{tu}
+
\operatorname{MHSA}_{H,b}
(\operatorname{LN}X_{tu};\operatorname{RoPE}(h)),
$$

$$
Q_{tu}
=
Y_{tu}+
\operatorname{FFN}_{H,b}(\operatorname{LN}Y_{tu}).
$$

其中

$$
Q_{tu}\in\mathbb R^{50\times256}.
$$

这是一次完整H读取，不增加层轴，也不压缩H。

**时间先验可能推动一条匹配带，却不能单独说明任务在推进。**本接口让匹配内容、位移分布、空匹配和整条H模式共同决定视觉query，再由真实画面和控制损失校正；不为匹配带的形状设置“必须像斜线”的辅助目标。

---

## 5. 两端视觉核实

$$
z^{\rm past}_{tu,h}
=
\operatorname{Attn}_{Z,b}
(Q_{tu,h}+e_{\rm past},Z_u,Z_u),
$$

$$
z^{\rm now}_{tu,h}
=
\operatorname{Attn}_{Z,b}
(Q_{tu,h}+e_{\rm now},Z_t,Z_t).
$$

形成消息：

$$
M^{(b)}_{tu,h}
=
f_{M,b}\left[
Q_{tu,h},
U^{(b-1)}_{t,h},
m_{tu,h},
r_{tu,h},
w_{tu,h},
\frac{\Delta_{tu}}5,
z^{\rm past}_{tu,h},
z^{\rm now}_{tu,h},
z^{\rm now}_{tu,h}-z^{\rm past}_{tu,h}
\right].
$$

注意：

**\(Z_u\) 是前序起点的实际画面，不是前序horizon位置 \(g\) 的未来画面。**软对应帮助形成“前后动作条件怎样延续或修正”的查询；两端视觉读取核实的是起点之间真实发生的变化。

因此，这里没有把horizon索引误当成能直接取到的教师未来图像。

每轮的视觉K/V对每帧只投影一次，供所有相关帧对复用。不同轮使用各自参数，冻结的原生 \(Z\) 可以跨轮、跨optimizer step缓存，learned K/V则不能跨参数更新复用。

---

## 6. 按历史起点排序的GRU聚合

将实际存在的前序帧排序为

$$
u_1<\cdots<u_L,\qquad L\le4.
$$

对每个 \((t,h)\) 独立运行同一组共享GRU参数。

时间输入：

$$
\gamma_i=
\left[
\frac{\tau_t-\tau_{u_i}}5,\,
\mathbf1_{i>1}\frac{\tau_{u_i}-\tau_{u_{i-1}}}5,\,
\mathbf1_{i>1}
\right].
$$

第一个位置不制造一个不存在的更早帧；第二项为0并用第三项标识“不适用”。

$$
x_i=\operatorname{LN}(M^{(b)}_{t,u_i,h})+\eta_b(\gamma_i),
$$

$$
c_0=\tanh\left(W_{0,b}\operatorname{LN}U^{(b-1)}_{t,h}\right).
$$

采用标准GRUCell定义：

$$
r_i=\sigma(W_rx_i+V_rc_{i-1}+b_r),
$$

$$
z_i=\sigma(W_zx_i+V_zc_{i-1}+b_z),
$$

$$
n_i=
\tanh\left(W_nx_i+b_n+
r_i\odot(V_nc_{i-1}+\widetilde b_n)\right),
$$

$$
c_i=(1-z_i)\odot n_i+z_i\odot c_{i-1}.
$$

这里的公式与常用GRUCell实现一致。([PyTorch Docs][2])

回到完整H：

$$
U'_{t,h}
=
U^{(b-1)}_{t,h}
+
W_{O,b}(c_L-c_0),
$$

$$
\widetilde U^{(b)}_{t,h}
=
U'_{t,h}
+
\operatorname{FFN}_{U,b}(\operatorname{LN}U'_{t,h}).
$$

\(W_{O,b}\) 不带bias。\(L=0\)时 \(c_L=c_0\)，邻帧关系更新严格为0，不对空序列做softmax，也不填入伪造邻帧。逐位置FFN仍可运行。

GRU状态在每个 \((b,t,h)\) 内重新初始化，**不作为跨帧滚动缓存，更不是部署期记忆**。

### 这里的数学保证和非保证

若 \(c_0\) 有界，则由于门值在 \([0,1]\)、候选经过tanh，\(c_i\)逐坐标保持有界。这有助于避免把重叠区间简单重复累加。

但它不保证：

* 重复证据严格不改变输出；
* 较近证据一定更可靠；
* 信息不会被遗忘；
* 学到的递推就是贝叶斯滤波。

这些都要由实际学习决定。

---

## 7. 临时H-read、长程组织与回写

### 7.1 每组形成一个临时长程输入

$$
E^{(b)}_t
=
\operatorname{HRead}_b
\left(q_b(\bar\ell),\widetilde U^{(b)}_{t,0:H-1}\right)
\in\mathbb R^{256}.
$$

这是读出副本；\(\widetilde U^{(b)}\)仍保留。

单视频长程层：

$$
P^{(b)}
=
\operatorname{TemporalBlock}_b
(E^{(b)};\operatorname{RoPE}(\tau/5)).
$$

每组一层标准pre-norm self-attention＋FFN，**双向读取整条视频**，输出

$$
P^{(b)}\in\mathbb R^{T\times256}.
$$

### 7.2 前三组按当前H状态回写

$$
U^{(b)}_{t,h}
=
\widetilde U^{(b)}_{t,h}
+
W_{o,b}
\operatorname{GELU}
\left(
W_{u,b}\operatorname{LN}\widetilde U^{(b)}_{t,h}
+
W_{p,b}\operatorname{LN}P^{(b)}_t
\right),
\quad b=1,2,3.
$$

中间宽度512，输出宽度256。

它对全局表示的局部Jacobian为

$$
\frac{\partial\,\delta U_{t,h}}{\partial P_t}
=
W_o
\operatorname{Diag}
\left[
\operatorname{GELU}'(W_u\widetilde U_{t,h}+W_pP_t)
\right]W_p
$$

——省略LayerNorm项只为显示结构。

由于不同 \(h\) 的当前状态不同，回写对同一 \(P_t\) 的作用可以不同。这里确实有逐H的内容条件化，而不是对全部H广播同一个增量。

下一组使用这个 \(U^{(b)}\) 重新计算score、位移分布、完整H query和视觉读取。

### 7.3 第四组不做无用途回写

第四组得到 \(P^{(4)}\) 后直接进入集合compiler。没有第五轮局部读取，就不增加最后一次回写。

总计：

| 计算               |             次数 |
| ---------------- | -------------: |
| 过去帧对对应           |              4 |
| 完整H-query block  |              4 |
| 逐帧对两端视觉核实        |              4 |
| 短GRU窗口组织         |              4 |
| H-read           | 4：前三次临时、最后一次终局 |
| 长程temporal block |              4 |
| 长程→完整H回写         |              3 |

四组是首版容量与可执行性的选择：保留原先四次局部更新，提供三次真正可被后续视觉查询使用的长程反馈；不是已证明四组最优。

当前代码是在局部block结束后才H-read，随后进入compiler，并没有这条回写路径。

---

## 8. 集合compiler与native读出保持既定方案

每条视频独立执行上述计算，得到 \(P^{(k,4)}\)。集合compiler使用38×16个paired queries：

$$
c_{mr}
=
\operatorname{Compiler}_{mr}
\left(
\bar\ell,\{P^{(k,4)}_t\}_{k,t}
\right),
$$

$$
c\in\mathbb R^{38\times16\times256}.
$$

两层compiler block，保留跨query协调。Key可使用每条视频内部的时间路由；Value为过程表示。每视频基础质量用 \(-\log T_k\) 修正，不赋予video ordinal身份。

完整读出：

$$
a_{mr}=a^0_{mr}+D^A_{mr}\operatorname{GELU}(U_Ac_{mr}),
$$

$$
b_{mr}=D^B_{mr}\operatorname{GELU}(U_Bc_{mr}).
$$

继续采用：

* rank16、完整38 targets；
* code和native读出宽256；
* \(D^{A/B}\)按target/rank/side独立、跨task共享；
* \(D^{A/B}\)零初始化，\(A^0\)为既定identity模板；
* 不增加第二adapter或固定执行carrier。

最终是

$$
\Delta W_m=\sum_{r=1}^{16}b_{mr}a_{mr}^{\top}.
$$

输入只有末层响应，不妨碍compiler输出全部target；但它也不能恢复末层已经丢失的不可区分信息。这一信息取舍不在本轮重新打开。

---

## 9. 参数共享与同步顺序

同一组内的参数在所有task、视频、帧对和horizon位置之间共享；四组参数各自独立。GRU在最多四个消息位置间共享权重，不按“第几个邻帧”建立独立模型。

一组内：

1. 所有帧对读取同一份 \(U^{(b-1)}\)；
2. 并行形成全部经视觉核实的 \(M^{(b)}\)；
3. 对每个 \((t,h)\) 按 \(u\) 排序递推；
4. 同步形成全部 \(\widetilde U^{(b)}\)；
5. 完成全视频H-read与长程组织；
6. 同步回写，进入下一组。

**只有窗口内的消息读取按顺序递推，不允许按视频帧遍历顺序原地更新U。**否则处理较晚帧时可能读到本轮刚更新的前序状态，计算语义和有效深度都会变化。

首版其余默认：width256、8heads、FFN扩展比4、pre-norm、dropout0；edge chunk初始8；轮次和edge计算使用activation checkpoint。除了既定D和时间bias末层的初始化，不额外添加多层零门链。

---

# 六、主要计算与激活成本

以下是张量与算子规模估计，不是新图profile结果。

设

$$
T=100,\quad H=50,\quad d=256,\quad N=280.
$$

过去四帧的有向边数：

$$
E=\sum_{t=0}^{T-1}\min(t,4)=390.
$$

## 1. 窗口改成过去方向，具体省掉什么

上一版对390个无序帧对计算一次score，但生成两个方向的消息。本轮仍有390个当前—过去帧对：

* 原始score数量不必减半；
* 反方向的归一化、消息和视觉读取消失；
* 两端视觉读取依然保留，因为每条单向关系仍需核实过去与当前画面。

| 项目                                          |         本轮每组规模 |
| ------------------------------------------- | -------------: |
| 帧对score \(E\times8\times H^2\)              |         780万元素 |
| 新增H-query attention score                   |        另780万元素 |
| 两端视觉attention \(2E\times8\times H\times N\) |       8,736万元素 |
| 帧对query或消息的一份 \(E\times H\times d\)         |  BF16约9.52 MiB |
| 完整U的一份                                      |  BF16约2.44 MiB |
| 末层R的一份                                      | FP32约19.53 MiB |

上一版双向帧对的两端视觉score约1.747亿，本轮约减半。若本轮视觉score全部物化，BF16约166.6 MiB；高效attention不需要常驻整张矩阵，但反向激活仍须实测。

## 2. 完整H-query增加的成本

一层标准H-query block的主要复杂度是

$$
O(EH^2d+EHd^2).
$$

在上述例子中，按标准attention投影与4倍FFN估算，约15.8G MAC/组。新增的是对**新帧对关系序列**的联合解释，不能算进原来的对应矩阵后声称没有额外成本。

## 3. 有序GRU的成本

最多执行

$$
EH=19{,}500
$$

个256维状态更新。标准GRU主要矩阵乘约

$$
6EHd^2,
$$

在该例中约7.7G MAC/组。

这里只有四个顺序位置，所有 \((t,h)\) 批量处理。它比原来的单次邻居attention更重，但避免为了前缀交互堆叠多层窗口Transformer。不能仅凭MAC断言实际吞吐，短递推的kernel与调度仍需要profile。

## 4. 长程交替的成本

每组长程attention score仅有

$$
8T^2=80{,}000
$$

个元素。时间attention本身不大；其投影、FFN，以及逐H回写仍有成本。

一次512维中间层回写约

$$
6THd^2,
$$

在该例中约2.0G MAC，三次回写约5.9G MAC。

相比“最后才做两层长程”，本轮明确增加到四次长程block和三次回写，换取全局信息参与下一轮视觉查询。

## 5. 不会随过程层轴消失的成本

18层过程轴已经去掉，但以下仍在：

* 完整Gemma与Action Expert前向；
* 观察Meta经过完整网络的反向；
* 逐轮原生视觉K/V投影；
* 执行侧FM与十步flow重放；
* 约3.295亿参数的native D及其优化器状态。

因此不能用“过程张量缩小18倍”外推端到端速度。实际实现仍应复用每帧视觉K/V、批量处理edge和H，不逐token循环；也不能为省算力跨参数版本复用旧 \(R\)。

---

# 七、训练如何接到最终图

## 1. 两个选择要分开

**Writer与Meta端到端联合训练**，说的是可训练范围与梯度连通性。

**FM和RL在同一步合并**，说的是数据采集、策略版本与optimizer cadence。

前者不必然要求后者。交替做FM与RL也可以，但每次参数改变之后，后续普通on-policy RL都必须重新采样，或者明确采用带分布修正的surrogate。

**首版继续选择同版本采样、FM＋RL合并一次更新。**理由是：

* 避免FM更新后又把旧rollout当当前策略；
* 两种信号可以先合并到同一套LoRA cotangent；
* 每condition只需组织一次Writer／Meta重放；
* 稀疏reward组没有有效RL信号时，FM仍提供真实动作梯度。

这不是架构所强制的数学定理，而是当前最清楚、最少分布歧义的执行协议。

## 2. 梯度链

最终图中的全部操作都处于同一条可微链：

$$
L_{\rm FM},L_{\rm RL}
\rightarrow A/B
\rightarrow c
\rightarrow P^{(4)}
\rightarrow
\text{各组长程、回写、GRU、视觉读取、H-query、对应}
\rightarrow R
\rightarrow\mu.
$$

不在临时H-read、全局回写或组间detach。

帧索引、排序和窗口mask是固定结构，不需要梯度；learned时间函数、软对应、GRU和全部读取参数正常反向。

冻结 \(Z\) 和prefix KV可缓存；训练中的R、视觉投影、过程表示和全局状态不能跨版本当作冻结证据复用。

---

## 3. 保留上一版RL定义及其适用范围

对当前版本 \(\psi_n\)，实际十步flow与原生噪声产生前五步均值：

$$
m_{\psi_n}(o,\mathcal C,\epsilon)\in\mathbb R^{35},
$$

$$
z\sim\mathcal N(m_{\psi_n},\Sigma).
$$

仍使用显式条件高斯score，不使用未知flow边缘概率。

同condition四条独立episode采用leave-one-out baseline：

$$
A_e
=
Y_e-\frac13\sum_{e'\ne e}Y_{e'}.
$$

每episode均匀抽取 \(M=\min(16,Q)\) 个decision，以 \(Q/M\) 修正。给定轨迹，这仍是decision总和的无偏子采样；baseline与被减去的本episode探索独立时，不改变score梯度期望。

联合梯度目标仍为：

$$
L_{\rm joint}=L_{\rm FM}-0.1J_\Sigma.
$$

所有score梯度在采集版本 \(\psi_n\) 计算。轨迹不跨参数更新复用；KL检查只限制候选步，不修正off-policy分布。

本轮沿用的首版默认：

| 项目         | 默认                                       |
| ---------- | ---------------------------------------- |
| 每次task采样   | 每suite均匀抽1个，共4个                          |
| 每condition | 64个跨episode FM queries、4条RL episodes     |
| 额外探索       | 五步相关高斯，\(\rho=0.8\)                      |
| 归一化标准差     | 前六维0.05，夹爪维0.10                          |
| FM／RL系数    | \(1/0.1\)                                |
| AdamW      | LR \(3\times10^{-5}\)，8个接受更新warmup，随后先保持 |
| 梯度处理       | 合并后global clip1                          |
| trust检查    | 每动作块平均条件KL的task最大值不超过0.02                |
| 辅助项        | 不默认增加视频预测、shaping、critic或expert纠错        |

这些是一个可运行起点，不是数学保证。长期全失败组的RL advantage可能全零；必须如实记录，而不是以“存在梯度通路”冒充已有有效RL学习。

去掉额外探索噪声后的 \(J_0\) 仍保留原生flow noise。正式性能只按规定部署接口计算，不能以 \(J_\Sigma\) 的提升替代。

---

## 4. 一次训练迭代流程

```text
1. 固定ψn：Writer、Meta、optimizer状态保持不变。

2. 分层抽取4个训练task。
   每task采样一个正确K1/2/4视频集合和64个跨episode FM queries。

3. 计算冻结Z/KV、当前版本的末层R。
   运行四组局部—长程交替，生成该condition唯一完整LoRA。

4. 用该LoRA采集4条独立训练rollout。
   记录原生flow noise、额外探索动作、旧均值和必要观测。
   均匀保留最多16个decision；获得官方成功Y。

5. 在仍未更新的ψn上：
   计算FM对A/B的cotangent；
   用baseline和Q/M权重重放十步flow，计算RL对A/B的cotangent。

6. 合并同condition的FM＋RL cotangent。
   重放一次完整Writer图，再重放观察侧Meta；
   组间、H-read和全局回写均不detach。

7. 跨task和GPU按既定权重合并梯度，clip，形成一个AdamW候选方向。

8. 只做候选KL前向检查，接受一次有限缩放更新或回滚。
   不在候选参数上继续用旧轨迹计算普通score梯度。

9. 得到ψn+1。
   本批轨迹封存为证据，不进入下一轮actor训练复用。
```

架构变更没有迫使RL改变概率模型。真正改变的是：A/B的信用现在能沿完整H-query、关系消息递推和全局回写，回到影响下一次视觉核实的参数。

---

# 八、行为裁决与剩余问题

这次不再把每个新增接口都拆成必须先证明的独立课程。先让完整首版进行有信息量的共享学习，随后按出现的行为差距选择诊断。

## 1. 是否真的利用了完整H关系模式

本轮的结构已经允许另一行对应模式改变当前视觉query，但是否值得采用，要看它是否改善实际控制与任务覆盖。

若只得到更清晰的匹配带、更低的内部loss，却没有抓取、阶段切换和闭环成功收益，不能继续以“模型现在更懂时序”解释结果。

## 2. 全局反馈是否帮助核实，而非放大错误公共解释

全局信息可能帮助解释遮挡，也可能把全部局部关系拉向同一个错误任务叙述。

残差保留完整U、逐H条件化回写和每轮重新读取原生Z，给了模型纠正这一问题的通道；它们不保证模型一定使用该通道。

如果全局反馈提高某个suite却持续损失另一类能力，历史中的任务漂移解释仍然适用，不能只看总分。

## 3. 共享学习与部署行为是否共同改善

继续分开观察：

* 训练task获取；
* 同task不同正确视频；
* 未见task迁移；
* 相邻成功集合保持；
* \(J_\Sigma\) 到 \(J_0\) 的收益保留。

当前读出干预的局部正结果、P/Q的clone/shared差距，以及历史G2和强Writer的不同能力，只能支持继续探索这些接口，不能拼成新方案已经成功的证据。

正式选择仍按既定strict400、单checkpoint、相邻稳定、breadth、Goal/Long、跨视频及最终controls合同。存储排列不变性、梯度依赖等可以用合成张量做工程验证；**不通过真实视频shuffled/reversed结果反向选择本轮架构。**

---

## 最终定稿

首版确定为：

**过去四帧单向对应；一层完整H-query；逐帧对两端视觉核实；按历史起点排序的短GRU；四组局部—双向长程交替；前三组逐H非线性回写；最后集合compiler和既定native完整A/B。**

这一选择的数学依据分别是：

* 单向条件归一化统一了 \(u\to t\) 的物理解释；
* 完整H-query增加了新对应模式之间的直接依赖；
* 短GRU让当前消息的使用显式依赖此前已读的重叠证据；
* 全局回写让离线过程解释重新控制原始视觉读取，而非只处理已经压缩的结果；
* 逐H非线性回写避免了单Key attention或相同向量广播的退化。

这些定义已经足以形成一套可实现算法。剩下要裁决的是其共享闭环能力，而不是继续悬置窗口、读出或回写的计算含义。

[1]: https://doi.org/10.48550/ARXIV.1706.03762?utm_source=chatgpt.com "[1706.03762] Attention Is All You Need"
[2]: https://docs.pytorch.org/docs/main/generated/torch.nn.modules.rnn.GRUCell.html?utm_source=chatgpt.com "GRUCell — PyTorch main documentation"
