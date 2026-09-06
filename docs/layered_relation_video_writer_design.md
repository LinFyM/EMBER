# 分层局部关系视频 Writer：科学动机、推导与实现设计

设计更新日期：2026-09-07。本文记录 owner 已对齐并要求交接的新候选，以及为实现它所需的首版参数化。
本次更新将过程模块收口为双向局部帧对关系、对应内容与位移模式联合解释、逐帧聚合和同型 block 堆叠。
它是新 session 的完整设计依据，不是训练启动记录，也不宣称本图已实现、已验证或已经通过任何科学资格。
当前工作授权和是否存在 active design 只由 owner 最新指令及 [progress.md](../progress.md) 决定。

长期目标与信息墙见 [current_owner_requirements.md](current_owner_requirements.md)，概念入口见
[concept.md](concept.md)，历史裁决入口见 [research_history.md](research_history.md)。本文中的固定数值分为两类：
数据/部署合同必须遵守；width、block 数等首版默认须经真实成本与科学证据裁决，不能因写入本文就升级为永久原则。

## 1. 问题起点与科学精神

人能够从不同身体、视角与初始场景的正确教学视频中学习“在什么条件下，经过什么过程，完成什么目标”，再让自己的身体执行。
视频本身常没有可直接用于机器人行为克隆的 action labels。EMBER 研究能否把这种迁移写成一次视频条件参数生成：

\[
\Delta\theta=\mathcal W_{\phi,\mu}(\ell,\{V_k\}_{k=1}^{K}),\qquad
a\sim\pi_{\theta_0+\Delta\theta}(o),
\]

其中 \(\ell\) 是 exact task language，\(V_k\) 是同 task 的 action-hidden 正确视频，\(\theta_0\) 是从 generic
`lerobot/pi05_base` 建立的冻结 LIBERO source policy。Writer 在 rollout 前运行一次，生成完整 task LoRA；随后 policy
依据自己的当前观测闭环执行。视频到执行的桥梁是功能性参数改变，不能依赖部署时的 teacher action、task ID 或再次观看视频。

原始动机允许人类、其它机器人等不同具身的视频；当前验证使用既定 LIBERO 数据。当前同域成功不会自动证明跨具身能力，
也不授权为论证愿景另造人工数据集。Action Expert 的原生动作知识是可利用的归纳偏置，其对单张教学图像的响应并不是正确
teacher action、真实未来轨迹或已经识别的子任务程序。

方法需要依次回答三个问题：

1. 单帧的图文语义与动作生成响应，如何形成有方向的视频过程证据？
2. 不同长度、不同 K 的正确视频，如何共同约束同一套策略，而不混淆各自的时间？
3. 过程证据如何写到 policy 的真实输入/输出坐标，使生成 LoRA 改变实际闭环行为？

推导服务这三个接口。可解释的公式、漂亮的 latent separation、梯度接通和非零更新，都不能代替闭环结果。完整目标仍是
validation8 的 single-checkpoint strict paired correct 严格 `>145/400`，同时具备相邻稳定性、低 churn、广 breadth、
四 suite 均非零、Goal/Long 贡献、same-task 不同视频鲁棒性与最终冻结后的必要视频因果证据。

## 2. 原则、已对齐结构与待检验命题

| 层次 | 内容 | 对后续工作的含义 |
|---|---|---|
| Owner 科学原则 | 动态视频必须是必要 Value 证据；完整单套 LoRA；零部署交互；真实闭环优先 | 不能由更低 loss 或更清楚的数学图替换 |
| Owner 已对齐结构 | frozen vision/Gemma；读取侧 Action Expert shared Meta-LoRA；单 fixed probe；保留 T/J/H 到跨帧消费；窗口内独立50×50帧对关系、两端读取、关系MLP后逐帧聚合与可堆叠更新；共同 Q 读取视频集合；坐标条件化 A/B | 首版沿此图实现，不恢复被覆盖的旧分支 |
| 推导支持的性质 | softmax 的熵优化形式；关系行与相对位移分布的可恢复性；有限双侧上下文；集合置换不变；VJP 链式梯度；普通末投影的固定 span | 证明范围仅限各自数学条件，不证明科学效果 |
| 首版实现默认 | rank4 Meta、d256/8heads、4 个 w4 temporal blocks、2 个 compiler blocks、p64 坐标 MLP、K1/2/4 | 可执行起点；batch、chunk、kernel 等需真实 profile |
| 未验证命题 | Meta 能否改善教学输入域；分层局部对应及过程是否可学；输出方向自由度是否改善共享学习；多 K 是否增益；能否稳定超过145 | 必须由后续真实实验回答 |

18 个 Action Expert post-layer states 表示计算深度，50 个 horizon positions 表示相对动作时间。二者都不是人工定义的
“接近、抓取、搬运、放下”标签。这样的语义可作为理解目标的例子，不能直接给 layer/horizon/rank slot 指派固定阶段。

### 2.1 从旧讨论中明确移除的继承项

首版采用一个固定 Gaussian probe。旧 `+ξ/-ξ` 与 even/odd 通道没有在本图中获得独立必要性论证，不因旧 capture 已有它们而保留。
不单列 noise、初始 layer boundary、layer difference 或 velocity 额外分支；不在 native Action Expert 已读 prefix 后再默认
增加一次初始 `R → Gemma Z` grounding；不先做全局 horizon mean；不恢复双向 P/Q 反馈；不产生每视频 LoRA 再聚合。

首版不显式读取 raw X/Y bank，也不强制最终 A/B 位于它们的 span。不增加 whitening、covariance solve、phase anchor、
手工 transport 或固定事件链。标准 attention/MLP 的可复制结构必须承担主要学习职责。今后若证据支持新的职责，应先说明
哪个接口出现了什么缺口，而不是把历史分支按名称接回去。

## 3. 完整数据流与符号

```text
exact task language ℓ ── frozen token embeddings ── language-only routing query
          │
K ordered videos, independently processed (frame stride 5)
          │ real images + exact language; no teacher proprio/actions
          ▼
frozen vision/Gemma native prefix and prefix KV
          ▼
Action Expert θ0 with shared reading-only Meta-LoRA μ
one public fixed ξ0[50,32], flow_time=1
          ▼
Rk[t,j,h,1024], j=18 image-conditioned post-layer boundaries, h=50
          ▼
shared-across-layer stackable local relation blocks
one 50x50 score per unordered frame pair within radius w=4
separate row-normalized reads for both endpoints
matched content + relative correspondence pattern + signed time gap
pair-message MLP -> per-frame neighbor attention -> residual + FFN
          ▼
task-conditioned learned H read after temporal consumption
Ek[t,j,256]; each token has finite two-sided context, layer identity retained
          ▼
{Ek} as a set; no cross-video time concatenation
shared Q[38,16,256]: cross-attention + whole-policy self-attention + MLP ×2
          ▼
c[m,r,256] → coordinate-conditioned shared scalar MLPs
          ▼
all A[m,r,i] and B[m,o,r], one complete 38-target rank16 LoRA
          ▼
frozen execution policy θ0 + generated LoRA; reading Meta μ is absent here
official current-observation rollout; Writer is not called again
```

| 符号 | 含义/shape | 不能混同的对象 |
|---|---|---|
| \(k,K\) | 视频索引与 cardinality | task ID、执行 expert 索引 |
| \(t,T_k,\tau_{k,t}\) | 第 k 条视频的采样帧序号、长度与已知时间戳/原帧序号 | action horizon、跨视频公共物理时钟 |
| \(j,J=18\) | 已读图的 Action Expert post-layer boundary | 子任务/语义 phase |
| \(h,g,H=50\) | native relative action-time positions | teacher 的未来第 t+h 帧 |
| \(s=1\) | native flow time，固定在噪声端点 | teacher-video time |
| \(\xi_0\) | 一个公共固定 `[50,32]` Gaussian probe | action labels、task-specific 随机码 |
| \(\mu\) | 共享读取侧 Action Expert Meta-LoRA | 每个新任务优化出的参数或第二执行 adapter |
| \(\phi\) | temporal/readout/compiler/decoder 等 Writer 参数 | 冻结 source 参数 |
| \(R_k\) | `[T_k,18,50,1024]` 原生层响应 | 已正确对齐的世界状态或 action 真值 |
| \(U_k\) | `[T_k,18,50,d]` learned temporal states | 原生 R 的无损复制 |
| \(E_k\) | `[T_k,18,d]` 分层局部过程记忆 | 每个 token 对整段视频的完整理解 |
| \(c_{mr}\) | `[d]`，同一 `(target,rank)` 的 paired A/B code | 单独 A-query 或 B-query、语义 phase |

下文省略视频 k 时，所有 teacher-time 运算都在单个视频内部进行。本文将 local block 数写作 \(B_T\)，以免和 LoRA 因子 B 混淆。

## 4. 原生观察：为什么需要 Meta、单 probe 和分层响应

### 4.1 读取侧与执行侧的不同职责

给定一帧图像与 exact language，冻结 vision/Gemma 形成原生 prefix 及其 KV：

\[
Z_t,\mathrm{KV}_t=\operatorname{Prefix}_{\theta_0}(I_t,\ell),\qquad
R_{t,j,h}=\operatorname{AE}^{(j,h)}_{\theta_0,\mu}
  (\mathrm{KV}_t,\xi_0,s=1).
\]

教学图像没有 teacher proprio，而且可能有视角或具身差异。共享 \(\mu\) 让 Action Expert 的读取响应可以在跨任务训练中
适应这种输入职责。它只安装在 observer 的 Action Expert q/k/v/o，rank4 为首版默认；vision、Gemma 和原始 source weights
均冻结。\(\mu\) 在一次新视频编译中固定，推理时不优化；执行 policy 只安装 Writer 生成的完整 LoRA，绝不顺带安装 \(\mu\)。

这里是“基础权重冻结、读取增量可学”。把整个读取路径包进长期 `no_grad`、并一直复用旧 R，会使这个定义失效。训练如何在
节省显存的同时保留 \(\mu\) 梯度，见第10节。

### 4.2 一个 fixed public probe 已形成完整定义

首版固定生成一次

\[
\xi_0\sim\mathcal N(0,I),\quad \xi_0\in\mathbb R^{50\times32},
\]

在所有 tasks、videos、frames 上复用同一 tensor，记录公开 seed/生成规范；不能根据 task 或视频内容挑 probe。固定噪声使
跨帧响应的外部变化来源更明确，同时保留原生50位置与 flow 条件。这里不是声称单 probe 无损或一定优于双 probe，而是本图
不需要第二次 opposite-noise observation 才成立。双 probe 的额外信息与行为收益尚无本图证据。

32 是模型 padded action width；LIBERO 的真实 action 是7维，执行 state 是8维。读取时通过真实图像与 exact language
建立原生 prefix，并使用合法 native suffix probe。不能构造零图像、伪 action query、伪 state 或缺 prefix 的 forward，来让
某个 backbone 名义上“参与”计算。

### 4.3 保留哪些 native layer states

现有 capture 在18个 layer 的 input-layernorm 输入及最终 norm 输入上取得19个 boundaries。boundary0 是第一次读图前的
suffix 起点；boundary1…18 分别对应18个 block 的 post-layer states，最后一个是 final norm **之前**的最后 block 输出。
本图使用后18个 states，保持 `[18,50,1024]`。初始 boundary 已作为 native forward 的输入发挥作用，不再给它独立 learned
证据通道，也不把19个 boundary 全都声称为已读图的状态。

固定 \(\xi_0,s\) 和冻结 action-in 使初始 boundary 不随 teacher 图像改变。跨层差
\(R_{j+1}-R_j\) 是网络计算增量，不能当 teacher 时间差；native velocity 是 flow 输出，不能当真实视频速度。
删去这些独立分支不代表它们数学上无信息，而是它们在本图中没有已建立的额外职责。

### 4.4 Gemma Z 在哪里

Z 通过原生 prefix/KV 被 Action Expert 使用，因此 R 已经是图文条件化的动作响应。本图没有独立的初始 `R→Z` 再 grounding
路径。不能由这一选择推出“R 保留了 Z 的所有信息”或“语义不重要”；R、U 与 E 都可能压缩或丢失证据。
额外直接读取 Z 若将来出现必要性，须有独立功能缺口与 matched 证据，不能因“Gemma 输出还没单独接线”而添加。

task-conditioned read 中 \(q(\ell,j)\) 的首版具体参数化为：直接取 exact language 的冻结 token embeddings，做一次
masked learned read 得到 \(\lambda_\ell\)，再以共享线性映射和 layer identity 形成查询。它只细化原有的 language-only
依赖，不是新增科学模块；token embeddings 不是独立运行 Gemma 得到的 contextual language hidden states。
不因此再做缺 prefix 的 Gemma forward，也不声称它优于其它合法语言表示。\(\lambda_\ell\) 可用于 H-read/Q 的查询与路由，
视频记忆的 Value 仍来自 R/过程。这个 language-only 查询不读取图像或视频总长度；过程 Value 则可使用窗口内前后两侧的教学帧。

## 5. 从帧对软对应推导可堆叠的局部过程

### 5.1 内容对应与过程推进是两种相关证据

owner的核心直觉是：执行进度推进时，同一段尚待完成的内容可能在新观察下移到更靠前的relative horizon位置。例如新帧位置1
对应旧帧位置4，位置2对应旧帧位置5，关系矩阵会出现偏离主对角线的带。允许这种对应随内容和帧间隔变化，不能人为规定
整体平移、单位矩阵、严格单调性或物理等式`t+h`。停顿、重新规划、多峰和局部没有可靠对应都可能存在。

这里输入的是单probe下的Action Expert hidden responses，不是已经正确生成的未来动作。共享probe本身也可能形成共同位置
结构；漂亮的对角线或偏移不能证明模型理解了运动。horizon仍是相对动作位置，j仍是计算层，不能给它们预设子任务标签。

过程模块需同时消费两类信息：匹配到的内容，以及匹配位置的变化。只在对齐后计算内容差，会在完美匹配时消去后一类证据。
因此每个帧对先形成独立关系消息，再由两端各自聚合；不要先把所有邻居的frames/horizons混成一个读出。

### 5.2 同层、共享投影、带帧间隔条件的50×50关系

跨 j 共享的输入映射形成：

\[
U^{(0)}_{t,j,h}=W_{\rm in}R_{t,j,h}+e_j\in\mathbb R^d.
\]

下文省略同一个原生层 j 和单个head下标 a。每层归一化沿feature维；不同head可以有各自参数。同型block之间参数独立，
每个block内跨t、j和视频共享主运算。H从输入至所有关系blocks结束保持50。

\[
\mathcal N_w(t)=\{u:1\le u\le T,\ 0<|u-t|\le w\},\qquad
x^{(b)}_{t,h}=\operatorname{LN}(U^{(b)}_{t,j,h}).
\]

对每个无序帧对，只按“较晚帧t为行、较早帧u为列”建立一次score；此时t>u、\(\Delta=\tau_t-\tau_u>0\)：

\[
C^{(b)}_{t,u}[h,g]
=\frac{F_b(x^{(b)}_{t,h})^\top F_b(x^{(b)}_{u,g})}{\sqrt{d_h}}
+b_b(\Delta,g-h),\qquad C^{(b)}_{u,t}=(C^{(b)}_{t,u})^\top.
\]

F在两端共用，首版采用共享线性投影；每head的\(d_h=d/n_{\rm heads}\)。内容关联与二维相对偏置共同决定软对应。
b必须能表达帧间隔与horizon位移的交互：若仅写成\(b_T(\Delta)+b_H(g-h)\)，固定帧对中的\(b_T\)是整行常数，
会被逐行softmax抵消。联合表或联合MLP是可执行参数化，具体形式在运行配置冻结；不能凭间隔指定平移量。

\(\tau\)使用该视频的真实原帧序号/已知时间戳，不是total-length归一化进度。stride5下相邻间隔通常规则，若canonical sampler
保留末帧，则使用实际尾间隔。反向score直接转置整张表，包括偏置；若实现为有符号统一函数，须满足
\(b_b(-\Delta,-\delta)=b_b(\Delta,\delta)\)，与只计算chronological pair的定义相同。
self-pair不进入邻居集合，当前状态经残差保留，不需要生成自对齐矩阵或单位矩阵标签。

### 5.3 一张关系表、两个条件读取

对一个接收位置，在H个候选位置上采用熵正则simplex读取：

\[
\max_{\alpha\ge0,\ \sum_g\alpha_g=1}
\sum_g\alpha_g C[h,g]-\eta\sum_g\alpha_g\log\alpha_g.
\]

内点驻点满足 \(C[h,g]-\eta(1+\log\alpha_g)+\lambda=0\)，因此

\[
A_{t\leftarrow u}=\operatorname{softmax}_{\rm row}(C_{t,u}/\eta),\qquad
A_{u\leftarrow t}=\operatorname{softmax}_{\rm row}(C_{t,u}^{\top}/\eta).
\]

首版\(\eta=1\)。二者可看作同一关联表的两个条件读取，通常不互为转置或逆矩阵。先转置score再归一化，不能把已归一化的A
直接转置作为另一端。每个帧对内部独立归一化，不能恢复在全部`(u,g)`上的联合softmax。

A是模型的读取权重，不是经过校准的真实对应概率；softmax不会证明某个真实对应存在。多对一、多峰与相对位置不变都是允许的，
不用Sinkhorn、硬argmax、伪alignment标签或人为斜对角loss代替功能学习。

### 5.4 对应内容与相对位置分布

对接收端\((t,h)\)，先读取邻帧内容：

\[
m_{t\leftarrow u,h}=\sum_g A_{t\leftarrow u}[h,g]\,V_b(x_{u,g}).
\]

把同一行权重改写到相对位置\(\delta=g-h\)：

\[
\rho_{t\leftarrow u,h}(\delta)=
\begin{cases}
A_{t\leftarrow u}[h,h+\delta],&1\le h+\delta\le H,\\
0,&\text{其它},
\end{cases}
\quad \delta=-(H-1),\ldots,H-1.
\]

H50时每head的rho有99个位置，只有对应合法g的50项可能非零。这只是索引重排：给定h和rho可以恢复A的该行，尚未丢失
对应模式。它不是额外的监督、teacher action、几何测量或第二次视频读取。

若局部存在理想匹配\(V_b(x_{t,h})=V_b(x_{u,h+s})\)、\(A[h,g]=\mathbf1[g=h+s]\)，对合法h有

\[
V_b(x_{t,h})-m_{t\leftarrow u,h}=0,\qquad
\rho_{t\leftarrow u,h}(\delta)=\mathbf1[\delta=s].
\]

对齐后的内容差为零，位置推进证据仍存在。因此不把内容差单独定义为完整动态。rho也不直接等于机器人真实速度或世界状态差。

### 5.5 一个共享MLP解释每个帧对

概念上直接输入当前状态、匹配内容、对应模式和带符号的时间间隔：

\[
\psi_{t\leftarrow u,h}
=\phi_b\!\left[x_{t,h},\ m_{t\leftarrow u,h},
\rho_{t\leftarrow u,h},\ \operatorname{enc}(\tau_t-\tau_u)\right]\in\mathbb R^d.
\]

多head版本的m拼接为d维，rho按head拼接为`heads*(2H-1)`维；当前状态只输入一次。phi为一个共享两层GELU MLP，
第一层hidden首版取d，输出d；时间编码可采用带固定单位的signed gap或共享编码，其参数化在首个配置记录。
phi可以比较当前与匹配内容、识别对应模式并取舍消息，不要求先手工构造唯一的差分D或额外gate链。

这一输入有直接的代数实现。对MLP第一层，令\(e_{a,\delta}=W_{\rho,a}[:,\delta]\)，则

\[
W_{\rho,a}\rho_a=\sum_g A^a[h,g]\,e_{a,g-h}.
\]

整个第一层可写为

\[
z=W_xx+W_mm+\sum_{a,g}A^a[h,g]e_{a,g-h}
+W_\tau\operatorname{enc}(\tau_t-\tau_u)+b_1,\qquad
\psi=W_2\operatorname{GELU}(z)+b_2.
\]

因此可以直接读取可学习的相对位置向量，不必先物化99维rho，更不需要另设“位移摘要网络”。内容和对应模式由各自权重进入
同一MLP后融合；不预先将原始m与一个位置向量直接相加，并假定两者无损可分。phi输出仍是learned compression，不能声称
整张对应矩阵被无损保存。

相对关系进入attention输出与已有relation-aware attention一致；本图按帧对形成消息、再逐帧聚合，是当前任务的参数化选择。
[Shaw等，Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074.pdf)不证明本Writer的性能。

### 5.6 每帧聚合自己的最多8条关系消息

对固定\((t,j,h)\)，先得到所有\(\psi_{t\leftarrow u,h}\)，再使用小范围多head attention：

\[
\beta_{t,u,h}=\operatorname{softmax}_{u\in\mathcal N_w(t)}
\left(\frac{q_b(x_{t,h})^\top k_b(\psi_{t\leftarrow u,h})}{\sqrt{d_h}}\right),
\]

\[
\widetilde U_{t,h}=U^{(b)}_{t,h}
+W_O\operatorname{Concat}_{a}
\left[\sum_u\beta^a_{t,u,h}\,v^a_b(\psi_{t\leftarrow u,h})\right],
\quad
U^{(b+1)}_{t,h}=\widetilde U_{t,h}
+\operatorname{FFN}_b(\operatorname{LN}\widetilde U_{t,h}).
\]

公式省略j；FFN首版hidden为4d。帧对MLP与邻居聚合承担不同职责：前者解释一项关系，后者决定这些关系如何共同更新当前位置。
一般\(\sum_u\beta_u\phi(z_u)\ne\phi(\sum_u\beta_u z_u)\)，所以不能先平均邻居特征再调用phi，也不能把两次归一化当作旧联合
attention的同义改写。若没有中间非线性且恰当选择邻居权重，两级softmax可退化为flat softmax；两级归一化本身不是表达优势证明。

所有消息仍保留h，直到所有blocks完成。phi与更新路径能够表达零/抑制性更新，残差保留当前状态，但是否学会忽略无效配对
须由真实证据判断，不能用attention峰度当置信度。T=1时邻居更新规定为0；首尾仅使用实际邻居，空邻居不调用全masked softmax。
当前状态仍可经过FFN，无须用假邻帧制造事件。

### 5.7 堆叠、方向和可证明的上下文边界

每个block内所有帧只读取上一block的U，再同时更新；不能读取Python遍历中刚更新的邻居。同型block之间参数独立，每层重新
建立C/A并解释关系。高层U仍锚定(t,j,h)，但已含局部上下文，高层矩阵不能自动称为原始未来动作的物理对齐。

归纳可得，在固定采样位置与局部mask下：

\[
U^{(B_T)}_{t,j,h}
=f_{\phi,\mu}(I_{\max(1,t-B_Tw):\min(T,t+B_Tw)},\ell).
\]

首版\(B_T=4,w=4\)，上下文最多前后各16个采样间隔，规则stride5时各约80个原始帧间隔。每个E是有限双侧上下文的表示，
整段远距离证据仍由最终Q共同读取；不能把局部E称为完整长任务程序。

Writer在rollout前获得完整视频，早期U/E可以依赖附近的后续教学帧。带符号的时间间隔、关系两端和相对位置模式保留时间方向；
不再要求teacher-time past-only mask，也不保留“追加任何未来帧都不得改变早期E”的旧测试。局部性仍要求窗口之外的图像
不能影响该层输出，且不得通过全视频汇总、总长度归一化或Q反馈偷偷扩大局部上下文。原生H轴attention保持source原生语义。

此图具有局部窗口和参数共享，属于局部关系attention/message passing。输入相关的对应与非线性消息提供明确归纳偏置，
没有一般定理证明它优于同预算的标准Transformer或任意直接attention。数学一致性、可堆叠性与真实行为优势分别判断。

### 5.8 跨帧消费以后再压缩H

\[
q^{\rm read}_{\ell,j}=W_\ell\lambda_\ell+e^{\rm read}_j,\qquad
\gamma_{t,j,h}=\operatorname{softmax}_{h}
\left(\frac{(q^{\rm read}_{\ell,j})^\top W_K^{\rm read}
\operatorname{LN}(U^{(B_T)}_{t,j,h})}{\sqrt{d_h}}\right),
\]

\[
E_{t,j}=W_O^{\rm read}\operatorname{Concat}_{a}
\left[\sum_h\gamma^a_{t,j,h}W_{V,a}^{\rm read}
\operatorname{LN}(U^{(B_T)}_{t,j,h})\right]\in\mathbb R^d.
\]

公式省略标准head索引细节；每条视频保留`[T,18,d]`，不再压成一个全视频平均向量。language决定读取哪些过程内容；
跨帧消费之前不做H mean。相对模式经内容条件化的A进入消息Value，不能用位置编码替代真实视频证据。

H-read是learned compression，不保证E无损保留R、rho或原始X/Y。Q的语言残差也可能成为静态捷径；图中存在动态Value路径
不等于视频已成为必要条件。是否利用对应、能否共享迁移和最终闭环的视频必要性都需后续实验，不能由内部attention图替代。

## 6. 多视频为什么有用，数学结论在哪里停止

### 6.1 共同潜在规则与每条视频的干扰因素

用 \(\zeta\) 表示完成任务所需的潜在条件—过程规律，\(\eta_k\) 表示第 k 条视频的初态、视角、速度等因素。
\(\zeta\) 是解释学习问题的变量，不是部署中可读取的 task ID，也不是假定已有的语义标签。
若给定 \(\zeta,\ell\) 后视频近似条件独立，则

\[
p(\zeta\mid\ell,V_{1:K})\propto p(\zeta\mid\ell)
\prod_{k=1}^{K}p(V_k\mid\zeta,\ell).
\]

语言给出目标与关注对象；每条视频排除与真实过程不相容的解释。不同视角、不同初态或不同速度可以减少偶然场景特征的影响。
真实演示共享 policy、背景与采集方式，条件独立只是近似，不能把 K 条相关视频计为 K 个独立 meta-task mappings。

不依赖条件独立，信息论的链式等式给出

\[
I(\zeta;V_{1:K+1}\mid\ell)-I(\zeta;V_{1:K}\mid\ell)
=I(\zeta;V_{K+1}\mid\ell,V_{1:K})\ge0.
\]

这是理想变量分布下的平均条件信息。若决策类允许忽略新增输入，最优 Bayes 决策的期望风险不会因可用信息增加而变坏。
实际有限数据、有限容量、梯度训练得到的 Writer 可能利用不好甚至被干扰；不能把上述不等式改写成每个 task、每个样本或
实际 closed-loop 的 \(\mathrm{success}(K+1)\ge\mathrm{success}(K)\)。

### 6.2 相关噪声决定收益上限

为了直观分析重复观察，假设已经对齐的一个标量估计为 \(\hat z_k=z+b+\epsilon_k\)，其中偏差 b 共享、噪声方差
\(\sigma^2\)、零均值、两两相关系数 \(0\le\rho\le1\)。对均值的误差有

\[
\mathbb E[(\bar z-z)^2]=b^2+\sigma^2\left(\rho+\frac{1-\rho}{K}\right).
\]

独立扰动随 K 减少，相关扰动与共同偏差留下误差 floor。这个简化公式说明多视频的统计可能性，不是让实现平均 frames、raw
features 或 LoRAs。实际过程尚需对齐，不满足“已有同坐标标量估计”的前提。

多视频还可能带来互补覆盖。若一项关键证据在一条视频中以概率 p 可见，独立可见性近似下，至少一条看到它的概率为

\[
P(\text{covered by }K)=1-(1-p)^K.
\]

这解释了不同遮挡/初态可能提供互补信息；共同遮挡会削弱收益。增加同 task 的视频能够改善条件证据，却不等于增加独立的
任务到参数映射。若训练 task 支持仍窄，K 本身不会解决跨任务学习不足。

### 6.3 多种正确策略不能通过参数平均混合

两条正确视频可能展示不同但有效的抓取/搬运策略。合理功能决策是从共同证据形成一套可执行策略；概念上可写为
\(\arg\min_{\Delta\theta}\mathbb E[\mathcal R(\Delta\theta,\zeta)\mid\ell,V_{1:K}]\)，
但实现用已训练 Writer 一次前向近似这个映射，部署时不做这个优化。

若先生成 \((A_k,B_k)\) 再平均，

\[
\bar B\bar A=\frac{1}{K^2}\sum_{i,j}B_iA_j,
\]

会产生无独立依据的交叉项。即使直接平均 \(B_kA_k\)，也可能提高 rank，且非凸策略行为不保证保留两种成功策略。
LoRA 还有 \((A,B)\mapsto(SA,BS^{-1})\) 的 gauge 自由度，因子平均依赖无行为意义的表示选择。
因此先融合过程证据，最后只产生一次完整 A/B。

## 7. 从多视频过程集合到整套策略代码

### 7.1 一个共同 Q 读取所有视频

执行 LoRA 覆盖38个 target，每个 target rank16，建立

\[
Q^{(0)}=\{q^{(0)}_{m,r}\}_{m=1,r=1}^{38,16},\qquad |Q|=608.
\]

同一 `(m,r)` 同时负责一对 A/B，不再以 A/B side 翻倍为1216个 queries。初始 query 可由 target/layer/family/rank
结构 embedding 与 \(\lambda_\ell\) 的映射相加；这些身份指向实际 policy 结构，不是 task identity。rank r 是因子槽位，
没有固定语义阶段。初始 query 是查询/路由条件，所有被读取的过程 Value 来自真实 E。

每个视频先独立得到 \(E^{(k)}\)。把所有 `(k,t,j)` 放入共同 memory 时，“flatten/concatenate”仅是物理存储布局；不在
`V1` 的末帧和 `V2` 的首帧之间建立 temporal edge。保留每条视频自己的 \(\tau_{k,t}\) 和 layer identity，禁止 learned
video-ordinal embedding。Q 从第一层起就读取所有视频的过程，不先为每视频单独编译一套 Q/LoRA。

### 7.2 长度校正来自基础采样质量

令 \(N_k=T_kJ\) 为第 k 条视频的有效 memory token 数。若对拼接 token 直接等基础质量 softmax，内容得分相同时长视频
会因 token 更多获得更大总权重。首版明确选“每视频相同基础质量、视频内每 token 相同基础质量”，于是对 query a 有

\[
s_{a,k,t,j}=\frac{q_a^\top K(E^{(k)}_{t,j},\tau_{k,t},j)}{\sqrt{d_h}},\quad
\alpha_{a,k,t,j}=\frac{\exp(s_{a,k,t,j})/N_k}
{\sum_{k',u,j'}\exp(s_{a,k',u,j'})/N_{k'}},
\]

\[
C_a=\sum_{k,t,j}\alpha_{a,k,t,j}V(E^{(k)}_{t,j}).
\]

实现即 score 加 \(-\log(T_kJ)\)。padding 不计入 \(N_k\)；不同视频的内容分数仍可使其实际权重不同。
时间和结构位置主要用于 K/Q 路由，不能用它们替代真实过程 Value。这个校正防止纯长度造成的基础权重偏置，不保证不同
视频等质量，也不证明学到了正确策略融合。

等价分解可以看清其作用：

\[
C_{a,k}=\sum_{t,j}\operatorname{softmax}_{t,j}(s_{a,k,t,j})V(E^{(k)}_{t,j}),\quad
\gamma_{a,k}=\operatorname{softmax}_{k}
\big(\operatorname{logsumexp}_{t,j}s_{a,k,t,j}-\log N_k\big),
\quad C_a=\sum_k\gamma_{a,k}C_{a,k}.
\]

这是同一次 attention 的代数分解，不要求另加一层专用视频融合，也不是平均最终 LoRA。视频置换只重新排列求和项，因此
在相同有效 mask/内容下输出置换不变，正常浮点 reduction 低位差异按既定合同接受。K=1 时长度项在 softmax 中抵消。

该 learned attention 不是前述 Bayes 似然乘积的严格实现。特别是把完全重复的同一证据集合整体复制，归一化读出可以不变；
它不自动表达独立重复观察的 posterior confidence。因此多 K 的收益仍须验证，不在首版加未讨论的 K confidence 分支。

### 7.3 两个标准 compiler blocks

每个 block 用 pre-norm/residual 的 cross-attention、whole-policy self-attention 与 MLP：

\[
\widetilde Q=Q+\operatorname{CrossAttn}(\operatorname{LN}Q,E; -\log N_k),
\]
\[
\widehat Q=\widetilde Q+\operatorname{SelfAttn}(\operatorname{LN}\widetilde Q),\qquad
Q'=\widehat Q+\operatorname{MLP}(\operatorname{LN}\widehat Q).
\]

首版重复2个同型 block，输出 \(c_{mr}=Q^{(2)}_{mr}\)。self-attention 允许 target/rank 槽位在读取共同过程后协调整套更新，
第二层可以在协调后重新读 E。这里没有 Q→早期 E/U 的反馈；E 保持只读。self-attention 的存在也不证明“多任务冲突已解决”。

## 8. 从过程代码到真实 native 坐标上的 LoRA

### 8.1 因子必须改变执行函数

对执行 policy 的一个 target，\(x\in\mathbb R^{d_{in}}\)、\(y\in\mathbb R^{d_{out}}\)：

\[
y=Wx+b+BAx
=Wx+b+\sum_{r=1}^{16}b_r(a_r^\top x).
\]

\(a_r\) 决定什么输入方向激活这次改动，\(b_r\) 决定输出怎样改动。一个固定 task LoRA 并非固定动作轨迹：系数
\(a_r^\top x\) 随执行中的真实观测/隐藏状态变化，所以同一套参数可以表示条件化动作规律。不能把 teacher 时间 t 当部署
控制时钟，也不能给每个 teacher 阶段换一套 LoRA。

### 8.2 普通固定末投影的确切边界

若 family head 为

\[
a=P_A f_A(c),\qquad b=P_B f_B(c),
\]

则所有任务生成的 a、b 分别在 \(\operatorname{col}(P_A)\)、\(\operatorname{col}(P_B)\) 中。对固定 target/family，

\[
\Delta W(V,\ell)=P_B M(V,\ell)P_A^\top.
\]

带 bias 时把 bias 方向加入生成空间；带公开 A0 时把对应固定 A0 行方向也计入。真正造成固定线性 span 的是最后一层
共享线性投影，不是“输入 code 的维数小”这一事实本身。

这项几何观察不能被命名为过去低分的已证实根因。早期 ordinary factor heads 路径具有真实强闭环证据；适当的固定子空间
也可能足够有效。解除一个表示限制会改变容量、参数量、优化与成本，并没有通用的性能改善定理。

### 8.3 将 code 与 native coordinate 联合非线性化

当前受控版本对每个真实输入坐标 i、输出坐标 o 使用坐标条件 MLP，末读出按 target/rank 独立：

\[
\delta a_{mri}=(u^A_{mr})^\top\operatorname{GELU}(U_Ac_{mr}+e^A_{mi}),
\qquad
b_{mro}=(u^B_{mr})^\top\operatorname{GELU}(U_Bc_{mr}+e^B_{mo}),
\]

\[
A_{mri}=A^0_{mri}+\delta a_{mri},\qquad B_{mor}=b_{mro}.
\]

\(U_A,U_B\in\mathbb R^{p\times d}\)，\(u^A_{mr},u^B_{mr}\in\mathbb R^p\)，\(e^A_{mi},e^B_{mo}\in\mathbb R^p\)，
\(p=64\)。A/B各自的code投影跨targets/ranks共享，native坐标跨rank共享；末读出在每个target/rank独立，
全部参数仍跨tasks学习。m/r只标识输出坐标，不能成为task路由。相同 \(c_{mr}\) 生成paired A/B。

联合非线性在坐标相关平移之后发生，因此可随 c 改变输出方向，不再强制形如固定 p 维末投影的 col-space。
一个直接使用本图激活的例子是 \(a(c)=[\operatorname{GELU}(c),\operatorname{GELU}(c-1)]\)：c=0 时第一坐标为0、
第二坐标非零；c=1 时第二坐标为0、第一坐标非零，因此两向量线性独立。一维输入的输出线性 span 可以达到二维。
这并不证明任意函数、任意因子或任意策略都可表达，rank16 限制及有限共享参数仍然存在。

\(A^0\) 是公开、与视频/task 无关的固定非零初始化模板；初始 \(\delta A=0,B=0\)，使执行增量为0。通过零初始化输出
scalar readout 等正常方式实现 identity 起点，随后完整 A/B 都可学习。它不是独立 carrier、第二 adapter 或12+4分配。
由于 B 初始为0，A 侧及更上游第一步可出现预期零梯度；Meta-LoRA 的零 B 初始化也有类似阶段。验证时须区分这种已知
初始化代数与永久 detach，不能要求所有参数在第一个 identity step 同时非零。

### 8.4 坐标初始化的受控修正（2026-09-07）

首轮实现将 native 坐标向量按 token embedding 的 std0.02 初始化，但这里的坐标会直接与 code 相加后进入 GELU，
前面没有坐标 LayerNorm。原图96步的两个训练任务、三个代表target中，code RMS约1.10–1.16、坐标RMS约0.02；
B的native-channel常量分量占99.995%以上，而真实policy功能梯度在该方向仅占约0.003%–2.23%。
这支持输出端初始化条件不佳的竞争解释，不证明它是弱闭环的唯一原因，也不否定关系图、动态视频信息或坐标MLP的表达能力。

已完成的受控fresh对照只将 A/B native 坐标向量改为独立标准正态，使不同坐标进入GELU的不同斜率区域。
不改变原生A0、零scalar readouts、rank/width/图、Meta、seed、监督、optimizer/schedule或曝光；初始执行仍严格identity。
这是一个固定的初始化干预，不扫scale、不增加gain/gate/旁路，也不从旧checkpoint继续。原std0.02结果和代码由frozen commit/run保留。
判据仍为同一短面板的绝对闭环、breadth、相邻与跨视频success集合；几何改善不足以接受该修改。当前参数量和形状均不变。

### 8.4.1 末读出共享方式的受控修正（2026-09-07）

标准正态坐标对照在96步K1 correct/other均6/40、K4为8/40；原初始化对应4/4/6。局部Long和冻结功能loss有改善，
但Object/Goal仍0，不能视为广泛共享能力。两个训练任务的三个代表target中，B的native常量能量降至70.7%–85.5%，
rank常量能量仍>99.995%；真实功能梯度的rank常量分量为2.56%–30.06%。代表target的局部Jacobian投影解除rank共享后
5/6方向改善，Long action_out反而下降；它没有覆盖38-target总梯度或Adam更新，不能证明唯一根因。

首版全局共享的u_A/u_B各为[p]。本轮只改为[target,rank,p]，保留联合GELU、std1 native坐标和全部上游图。
这解除不同输出槽必须通过同一末读出更新的约束，增加77,696参数；Writer总14,190,240、Meta仍626,688。
末读出仍零初始化，public A0与初始identity不变；零张量扩展不消耗RNG，其它初始化随机抽样保持。
不新增按task的参数、rank/width/gain扫描、旁路或第二adapter。张量形状改变，必须fresh全图联合训练。

训练/采样/权重/96步schedule、world3、frame_chunk4及16/48/96双视频K1和96K4配对40诊断均沿用初始化对照。
实现与结果前登记见target_rank_readout_control/registration.json，比较绝对行为、breadth及相邻/跨视频success集合；
不能凭几何或loss接纳该修改。全局共享读出的代码、配置和结果由旧frozen commits/runs保存，active source只保留本条路径。

### 8.5 为什么首版不需要 raw X/Y bank

设真实执行 loss 对 target 输出的梯度为 \(g=\partial\mathcal L/\partial y\)，忽略固定 scale 或将其吸收，则

\[
\frac{\partial\mathcal L}{\partial\Delta W}=gx^\top,\qquad
\frac{\partial\mathcal L}{\partial b_r}=(a_r^\top x)g,\qquad
\frac{\partial\mathcal L}{\partial a_r}=(b_r^\top g)x.
\]

这些真实 policy gradients 已经在 execution 的 native 坐标上训练 D、compiler、process 和读取 Meta。
raw X/Y conditioning 是另一种可选证据来源，限制输出在 X/Y span 又是另一项约束；两者不能混为“要不要尊重 native 坐标”。
尤其 \(Y=WX+b\) 是原层输出，通常不等于希望施加的修正方向 g。读取侧 Meta 的 X/Y 也不等于执行 source 在另一组真实
observations 上的 X/Y。首版因此不加 X/Y bank、不限制因子 span，但也不声称 E 是 X/Y 的无损等价物。

## 9. 首版结构、数据与监督合同

### 9.1 具体 shape 与默认配置

| 部分 | 首版定义 | 数值性质 |
|---|---|---|
| teacher sampler | 每视频内部保序，frame stride5；K 条不同视频无放回 | stride 是现行合同；末帧处理沿 canonical sampler |
| probe | 一个公共 \(\xi_0[50,32]\)，所有输入复用，flow time1 | 已对齐，不能继承 antithetic P2 |
| frozen 部分 | vision、Gemma、source policy 原始参数 | 不产生参数更新 |
| observer Meta | Action Expert18层 q/k/v/o，rank4 | 读取侧共享可训练；推理编译内固定 |
| native R | `[T_k,18,50,1024]` | 18个 post-layer states，完整50H |
| process U | d256，8heads，head-dim32 | 首版容量默认 |
| relation blocks | 4 blocks，双侧半径w4，每帧最多8个邻居；跨j共享，blocks间独立 | 前后各最多16采样间隔 |
| pair correspondence | 每无序帧对/同j/head一个50×50 C，共享F、联合时间/位移偏置；两端分别softmax，eta1 | 不在全部邻居H上联合归一化 |
| pair message | 当前d、匹配内容d、各head的99维相对对应模式、signed gap → shared GELU MLP，hidden d/output d | rho的第一层投影可等价为相对位置Value读取 |
| frame update | 每h对最多8条消息做multihead attention，residual+FFN；FFN hidden4d | 一个block内读旧U、同步生成新U |
| H read | `(t,j)` learned task-conditioned multihead read | H 跨帧消费后才压缩 |
| memory E | `[T_k,18,256]`，Q阶段只读 | 分层且局部，不是无损全局语义 |
| policy Q | `[38,16,256]`，608 paired A/B slots | 2个 cross/self/MLP compiler blocks |
| set prior | 每视频基础质量相同，score `−log(T_k*18)` | valid token计数，不含 padding |
| coordinate decoder | shared scalar A/B MLP，GELU，p64 | code/native coordinate联合非线性 |
| generated LoRA | 38 targets，rank16，alpha16/scale1，无 dropout 的现有物化合同 | 76个 A/B tensors，一套部署 |
| 初始化 | 公共非零 A0，deltaA/B零；其余按标准随机初始化；Meta采用合法LoRA初始化 | 当前主线直接fresh端到端联合训练 |
| dynamic K | 首版真实训练 K∈{1,2,4}，不同正确视频无放回 | 不重复一条视频凑K，不由可变shape冒充训练覆盖 |

当前38-target topology：

| target family | 数量 | \(d_{in}\) | \(d_{out}\) | A / B 每 target |
|---|---:|---:|---:|---|
| Action Expert q_proj | 18 | 1024 | 2048 | `[16,1024]` / `[2048,16]` |
| Action Expert v_proj | 18 | 1024 | 256 | `[16,1024]` / `[256,16]` |
| action_in_proj | 1 | 32 | 1024 | `[16,32]` / `[1024,16]` |
| action_out_proj | 1 | 1024 | 32 | `[16,1024]` / `[32,16]` |

每 condition 生成 \(16\sum_m(d_{in,m}+d_{out,m})=1,287,168\) 个标量，FP32约4.91MiB。
native coordinate embeddings 在 p64 时有 \(80,448\times64=5,148,672\) 个参数；这不是整个 Writer 的参数量。
最终完整参数总数、optimizer state 和显存必须由实际实现统计，不能沿用旧4.75M或旧width256图的数字。

### 9.2 任务、视频与动作 query 的采样

固定 development split 为24 train /8 validation /8 test，不按结果改 task IDs。source corpus 的71-task排除审计与
冻结 normalization 继续生效；不能使用读过目标40 actions 的 `pi05_libero`。train24 是目标开发梯度来源，经明确登记的
non-held LIBERO-90 allowlist 可提供额外 meta gradients，名单、语义重复排除及 provenance 必须可查。

一个训练 condition 是一个 task 的 exact language 加 K 条正确视频。action queries 来自同 task 的其它 episodes，
与这 K 条视频严格跨 episode；它们只进入训练监督路径，不被部署 Writer 读取。K1/2/4 可等频作为首版安排，但 cardinality
与 task 权重分别定义：K4 不能仅因帧更多就给这个 task 四倍梯度权重。视频 pool 必须实际支持不同视频采样，不能继续用
固定两视频池同时声称真实 K4。具体 episode role 分配沿已有数据资产审计，本文不凭空指定新 held/fit IDs。

增加每 task 的不同视频与增加独立 task mappings 是两个变量；扩大 K、pool、task 数和每 task 更新数也应分开记录。
小型训练面板可以定位接口，但不能让最终路线永久局限于18 targets、两个视频或缺少 Object 的老三任务面板。

### 9.3 用真实 policy 功能训练共同映射

对 task i、condition v、跨 episode action query batch \(\mathcal B_i\)，首版复用已有 positive-only functional
flow objective：

\[
\mathcal L_i(\phi,\mu)=
\mathcal L_{\rm flow}\big(\theta_0,
\mathcal W_{\phi,\mu}(\ell_i,V_{i,1:K});\mathcal B_i\big),\qquad
\mathcal L=\sum_i w_i\mathcal L_i.
\]

query/noise/time sampling、normalizer、每 task 权重与 optimizer cadence 要在实现后的实验合同中明确，不默默换 loss。
观察 probe 的 P=1 与训练 action-query loss 现有的 noise sampling 是不同职责；不能把“单 probe”错误扩展为修改执行训练
或评测的随机性。source 原始权重冻结，梯度经过它的实际函数传向生成 A/B，再回到 \(\phi,\mu\)。

不预设每 task 存在目标 LoRA，也不要求先拟合一个 privileged LoRA 字典。已有 experts 可以做获准 train-side 几何或容量
诊断，不能做 held route 或部署专家集合。模型冻结、无选择、预登记的 sealed held post-hoc 诊断按仓库合同处理；
validation/test actions/reward 始终不产生梯度。

当前主线将 Writer 与读取侧 Meta 从头初始化，使用 fresh optimizer/scheduler，直接端到端联合训练，source 基础权重始终冻结。
不实施 G1--G3 分阶段冻结课程，也不为旧措辞额外建立 component 初始化候选。第8节的公共非零 A0、零 deltaA/B 与 Meta
合法 LoRA 初始化保持；fresh 不要求每个张量随机非零。短学习、扩大覆盖与闭环是实验推进节点，不是分段冻结。
不兼容架构必须 fresh；同一 run exact-resume 仍锁原 world topology 和完整状态。

## 10. GPU 算法：保留完整梯度而不保留三套大图

### 10.1 可重排的是链式法则，不是学习目标

记 \(R=R_\mu(V)\)，\(L=F_\phi(R,\ell)\) 为全部76个 LoRA tensors，执行 loss 为 \(\mathcal J(L)\)。
训练需要

\[
G_L=\frac{\partial\mathcal J}{\partial L},\quad
\nabla_\phi\mathcal J=J_{F,\phi}^{\top}G_L,\quad
G_R=J_{F,R}^{\top}G_L,\quad
\nabla_\mu\mathcal J=J_{R,\mu}^{\top}G_R.
\]

这些 VJP 可以分阶段重放得到，无须同时保留 observer、完整视频 Writer 和 execution-policy 的所有 autograd activations。
首版算法如下，`stopgrad` 只切断临时图，后续 replay 显式恢复同一链式梯度：

```text
固定本 step 的全部 task weights、视频、action queries、probe、policy noise/time、RNG 与参数
optimizer.zero_grad()
for cost-assigned complete condition on this GPU:
    KV = frozen_prefix(real frames, exact language)       # chunked, bounded cache
    R0 = stopgrad(observer_mu(KV, fixed_xi0, flow_time=1)) # temporary at current μ
    L0 = stopgrad(writer_phi(R0, language))
    G_L = policy_loss_VJP(L0, cross_episode_queries)       # free policy graph afterwards

    R_leaf = R0.requires_grad_(True)
    L_replay = writer_phi(R_leaf, language)               # identical logical forward
    S_writer = sum_tensor_inner_product(L_replay, stopgrad(G_L)) * task_weight
    backward(S_writer)                                   # accumulate grad_phi and G_R
    G_R = R_leaf.grad                                    # retains cross-frame dependencies
    free writer graph and L tensors

    for observer frame chunk:
        R_chunk = observer_mu(KV_chunk, fixed_xi0, flow_time=1, grad_enabled=True)
        backward(inner_product(R_chunk, stopgrad(G_R_chunk)))
        free this observer graph
    free temporary R0, G_R, expired KV
all_reduce_weighted_gradients(phi_and_mu)
one optimizer/scheduler step
```

若 task weight 已在 policy VJP 中计入，Writer surrogate 不得重复乘；整个链只应用一次预登记权重。
R 的 frame chunks 独立于其它 teacher frames 的 native forward，所以 observer replay 可逐 chunk 累加
\(J_{R_c,\mu}^{\top}G_{R_c}\)。跨帧 attention 的梯度已经体现在 \(G_R\)，不能把 temporal block 按帧独立 backward
或截断跨帧依赖来冒充同一算法。

精确性要求 initial forward、policy VJP、Writer replay、observer replay 使用同一参数版本、样本、mask、RNG和数值合同。
期间不能更新 \(\mu/\phi\)、改变视频、随机换 probe、重抽 dropout 或更新 running statistics。本文的“exact”指链式梯度
语义；正常 BF16/TF32、批量/reduction顺序的低位差异按项目合同接受，不要求跨 batch 逐元素相同。

### 10.2 缓存的有效边界

冻结 vision/Gemma 在相同真实图片、exact language、预处理、mask 与冻结模型版本下的 prefix/KV 可复用。只建立有界的
实际工作集缓存；不能把“可缓存”自动变成复制整语料的巨大新 cache。执行 query prefix 与 teacher prefix 是不同输入，
不得凭同 task 混用。

读取 Meta 更新后，\(R_\mu\) 改变；R/U/E/LoRA 都不能跨 optimizer step 作为冻结 evidence cache 继续复用。
同一 step 的临时 R0 可在 VJP 分阶段中保存，随后用相同 \(\mu\) replay；checkpoint 恢复后不得接续旧参数版本的 R cache。
prefix cache 也不能缓存依赖 \(\mu\) 的 Action Expert 输出。Q 的多次 E read 与 inference 内的 fixed read-only replay
仍属于一次 Writer 调用，不引入环境交互。

### 10.3 帧对关系与消息的批量布局

按真实视频长度分桶，把condition/video/j/head放入适当batch/group维；保持H50。每视频只枚举一次无序局部边
\(\mathcal E_w=\{(t,u):0<t-u\le w\}\)，不沿frame、horizon或channel写逐项Python主循环。
令 \(E_w(T)=\sum_{\delta=1}^{w}\max(T-\delta,0)\)；T>w时为\(wT-w(w+1)/2\)。

对一批实际帧对，逻辑C为`[edge_batch,J,heads,H,H]`。shared F可对全部U先投影一次，随后batch gather得到两端，
一次GEMM形成C；时间/位移偏置按实际gap查表或批量计算。两个endpoint分别归一化C和转置，分别读取邻端内容与相对对应模式，
然后产生`[directed_edge,J,H,d]`消息，按receiver组装最多8个邻居的attention，更新完整U。padding边不产生消息。

rho不必成为长期张量：其`99→relation_hidden`首层权重可重排为共享的`e[head,delta,hidden]`，
按\(g-h\)索引并与A收缩。只共享位置表，不复制到每个edge/j；也不物化
`[edge,J,heads,H,H,hidden]`的逐位置消息。block/edge chunk与activation checkpoint用于约束峰值，
但chunk内外必须维持完整跨帧反向依赖。

唯一score的数量为\(JH^2E_w(T)\) / head / block；两端各有一套条件归一化和读出。
令\(d_\phi\)为关系MLP首层hidden（默认d），主项包括

\[
O(JE_wH^2d)
\quad\text{（score与内容读取）},
\qquad
O(JE_wn_{\rm heads}H^2d_\phi)
\quad\text{（直接相对模式投影的一种实现）},
\]

以及pair-message MLP、neighbor attention与frame FFN的projection成本。j仅作为batch维，保持线性J。
这些项只描述候选张量算法，不等于实际GPU时间；位置模式投影和每邻居的MLP可能改变主要瓶颈，不能沿用旧单向图的step成本。

首版可用batched GEMM、row softmax和收缩直接实现完整公式，再按真实profile优化融合与重算。标准SDPA只返回加权Value读出，
不会自动提供本图需要的两端对应模式；不能把过程模块整体换成一次`T×(wH)`的SDPA并宣称同义。
下游标准Q/H-read/neighbor attention仍可按其实际mask/layout选择高效attention实现。自定义融合是同一数学图的工程优化，
不另建一套科学fallback；kernel名称本身不证明吞吐收益。

staged VJP需要对R leaf和可学参数求导；activation checkpoint应与这种autograd使用方式相容，
例如明确使用非reentrant路径。重放仍须保持同一函数、参数与随机状态，checkpoint wrapper不能修复detach或改变任务权重。

### 10.4 坐标 MLP 与 LoRA 应用

按四种实际矩阵 shape 分组批处理 decoder，先计算全部 `U_A c` / `U_B c`，再在 coordinate tiles 中广播加 e、GELU、
与 u 收缩。避免逐 target/rank/channel 的 Python scalar 调用，也避免长期物化整块 broadcast hidden tensor。
完整 coordinate intermediate 若一次以 BF16 物化，大小为 `1,287,168*64*2` bytes，约157.13MiB/condition；
这只是一项 activation，不是完整训练峰值。

多个 condition 的 policy query 可以用已有 batched LoRA hooks/bmm：\((xA^\top)B^\top\)，不必构造巨大 \(BA\) dense
delta，也不必反复把每套 LoRA copy 进物理 PEFT adapter。物理 identity adapter、generated states 与 observer Meta
的安装作用域必须清楚，防止执行时同时叠加两份增量。

### 10.5 可核算的尺度与不能预告的速度

以 T=87、J=18、H=50、d256 为一条视频的张量算术：

| 对象 | BF16 大小/有效对数 | 含义 |
|---|---:|---|
| R `[87,18,50,1024]` | 152.93MiB | 单 probe、18个已读图 states，未计反向图 |
| 一份 U `[87,18,50,256]` | 38.23MiB | 一层 states，未计 q/k/v/MLP/grad |
| E `[87,18,256]` | 0.765MiB | H-read 后分层 memory |
| unique local w4 score entries | 15.21M / head / block | 338个无序帧对；每个C只建立一次 |
| directed conditional correspondence entries | 30.42M / head / block | C与其转置各自归一化/读出；不要求同时常驻 |
| full same-j T×H dense pairs | 340.605M / head / block | 仅是数学比较，不是本图执行方案 |

唯一局部score减少不等于整步获得同倍率提速；两个方向的内容/对应模式读出、pair MLP和邻居聚合均须计入。单 probe 减少 observer probe 工作与 raw-state 存储，也不等于整体
训练成本减半；frozen prefix、compiler、policy VJP 与 replay 成本仍在。K 增大使视频读取/过程成本约随总帧数增加，
最终只写一次 LoRA；实际瓶颈须按阶段测量。

恢复实现后用真实 K1/K4、最长视频、真实 action queries 测：LoRA/s、queries/s、step wall time、prefix/observer/
temporal/compiler/decoder/policy-VJP/replay 时间、GPU利用率与峰值显存。测量包含必要的数据供给，不以最低显存为目标，
不自设35GiB等固定上限，不靠 dummy 占卡。选出正常安全余量下吞吐好的 batch/chunk；不能现在空称新图“足够快”。

多卡将完整 condition 分配给单个 GPU，按 K、T 与实测 cost 平衡，但 task 权重、occurrence 和 optimizer cadence 保持。
只归约 Writer/Meta 的可学梯度；复用 deferred NCCL、`NCCL_P2P_DISABLE=1` 和 GPU-local NUMA。launch 时单节点1–6张
合适 GPU，exact-resume 锁该 run 的 topology；不跨节点拼碎片。GPU实时准入、独立 storage quota、clean pushed commit
与 frozen worktree 规则在真正 launch 前执行；本文不登记卡号或现成启动命令。

## 11. 已有代码基础与迁移边界

下表区分“可复用能力”与“旧运行链”。旧代码的精确基线是 `fcdb6e43706c5fcedf10eaa5d2d459602b263016`，
可用 `git show <commit>:<path>` 查看。清理后不为了维持链接保留退役可执行图；不把“存在 helper”声称为新架构已实现。

| 职责 | 可复用代码/基线 | 新图需要完成的适配 |
|---|---|---|
| 原生18层 boundary capture | `src/ember/ecp/observer.py:ActionLayerStateCapture` | 保留 `detach=False` 能力；选 boundaries1…18；不要带回旧 TargetOwnerProjector |
| 真实图像/语言 prefix | `src/ember/ecp/policy_effects.py:ExecutionPolicyPrefix, prepare_execution_policy_prefix` | teacher 无 proprio；保留官方 preprocessing 和 masks |
| frozen prefix KV | 同文件 `prepare_prefix_kv_cache`，原基线名 `_prepare_prefix_cache` | 单份 prefix；不复用 antithetic wrapper；实现时审视旧强制 eager 是否必要 |
| 读取 Meta | `src/ember/writer/meta_lora.py:MetaLoRAStack` | 只装 Action Expert q/k/v/o；vision/Gemma冻结；train/inference作用域明确 |
| policy VJP / Writer surrogate | `src/ember/writer/functional.py:functional_lora_loss_gradient, writer_chain_rule_surrogate` | 补 R-leaf VJP 和 observer replay；不改变 query/noise/weight 语义 |
| 通用同step重放编排 | 清理后的 `src/ember/writer/replay.py:functional_objective, functional_writer_backward, sum_writer_gradients` | 复用leaf A/B→query microbatch VJP→Writer replay及加权SUM梯度；它尚未实现新Meta的AE chunk replay或跨condition批量VJP |
| 38-target合同与 rank/scale | `src/ember/pi05_lora.py` | rank16完整76tensors；复用真实 module shape 校验 |
| batched policy execution | `src/ember/batched_lora.py:BatchedLoRAInference` | 批量应用生成LoRA；身份adapter不可叠加旧运行增量 |
| task sampling与cost placement | 清理后的 `src/ember/writer/task_schedule.py`、`task_execution.py` | 复用纯调度职责；真实K1/2/4、任务权重与跨episode隔离需新实验配置 |
| 已审计数据来源 | 清理后的 `configs/pi05_writer_data_v1.json` | 复用既有来源/角色说明；历史episode分配不自动成为新run采样合同或启动授权 |
| 既有 functional panels | `src/ember/writer/functional_data.py:resolve_functional_panel_records, load_functional_panels` | 复用role/episode互斥读取；新run仍须独立明确task allowlist、query与video采样 |
| checkpoint与topology | `src/ember/ecp/checkpoint.py`、`src/ember/writer/topology.py`、`src/ember/pi05_source_setup.py` | 复用完整恢复状态、NUMA/deferredNCCL；新图需新schema，不接旧optimizer |
| 旧训练编排的经验 | 基线 `src/ember/ecp/policy_response_writer/shared_training.py` | 学习其 policy VJP→Writer replay，不继承 frozen-R cache、旧 runtime 或旧schema |
| 旧 capture 的反例 | 基线 `src/ember/ecp/policy_response_writer/capture.py` | `@no_grad`、detach、双probe和多额外通道属于旧图，不能原样充当Meta-on observer |

新实现的责任面应保持少数清楚模块：原生读取适配与训练重放、分层局部关系过程与集合 compiler、坐标 decoder、训练/评测入口。
具体文件归属在实现时基于清理后的现有 owner 决定，不要求为了这个列表创建一整套新目录。保留一个 canonical active
Writer，退役 P/Q、固定 Natural Program、raw-bank compiler 和 old stage launchers 由 Git/正式证据恢复。

新的 checkpoint schema 应保存 \(\phi,\mu\)、optimizer/scheduler/scaler、sampler/cursor、各 rank RNG、world topology、
probe/shape/config contract。架构不相容时 fresh，不接旧 optimizer；可复用资产不等于可复用旧运行状态。
仓库既有必要 provenance 合同保持，不新增 MD5/SHA sidecars 或全树完整性扫描来证明编辑完成。

## 12. 历史如何约束本图，而不替代它的实验

这里保留影响本次推导的结论及适用条件。完整原始论证、专家意见、正式 rows 与配置从
[research_history.md 的历史原件索引](research_history.md#archive-index) 继续读取。当前精简入口按
[早期Writer](research_history.md#early-writers)、[原生容量](research_history.md#native-capacity)、
[过程与编译器](research_history.md#program-compilers)、[完整响应](research_history.md#full-response)、
[近期共同学习](research_history.md#recent-learning) 和 [吞吐](research_history.md#throughput) 分层。
以下节号锁定基线 `fcdb6e43:docs/research_history.md`，
避免精简历史后同名“当前”段落重新获得授权。

| 历史事实 | 对本图的帮助 | 不能推出的结论 |
|---|---|---|
| §2：v5.2 old 五臂132/138/74/82/83；v6-fast143/135/125/128/129，后续131/130/132/126 | 视频→LoRA曾有实质闭环能力，保持强基线和Goal/Long参照 | 不能把各版本优势拼成一个模型；旧143不能证明现在新图或特定组件 |
| §3：GOMQ151→135→131，相邻churn42/34；rank32写法实数有效rank≤16 | 单点过线与稳定能力不同；参数近似或重物化仍需真实行为裁决 | 不能把重物化136自动归因有效rank损失，也不能用151宣布完成 |
| 早期K4/fusion记录：跨视频BA变化更小，性能仍可下降 | 去噪/稳定参数与新增任务能力是不同目标；共同读过程要早于单视频LoRA决定 | 不保证加K有益；不宣称方差下降就是视频理解 |
| G1 native-factor几何：q输出Y-span、action-in低维span会限制已知成功更新；当时解除相关限制并采用privileged初始化的组合过门 | 原生坐标与完整输出方向确有可检验意义 | 不把耦合过门唯一归因span解除；不证明raw X/Y signed pooling是唯一decoder，也不证明本坐标MLP必优 |
| G2 ordered-response有功能动态；旧dynamic-K slots曾K1多事件、K2/K4坍为一事件，后经当时修正过门 | native响应含可学习动态；多K必须真实检查训练与内容消费 | 不强迫新图恢复DP/event-boundary链；G2通过不等于完整闭环通过 |
| §169：K仍1时fit视频池2→4，固定64更新使每视频曝光减半，64 fit/held44/45→41/39 | 池大小、K、预算是不同变量，采样记录必须清楚 | 这不是K1→K4实验，不能据此否定few-shot |
| §170/173：旧P/Q64 fit/held41/39，完整输出64/62（各150），Goal主要改善、Spatial有损失 | 完整A/B职责重构有训练侧正证据，行为代价须保留 | 同时改carrier分配、span约束与head，不能唯一归因rank或XY移除 |
| §174–181：mixed/target18/random均未获得强共享行为；同预算两task clones14/20，共享3或4/20 | 未见task迁移之外还有训练共同学习缺口，容量/条件表示/优化仍需区分 | 不命名“固定head span/梯度冲突/初始化”为已证根因，不部署clone字典 |
| §177/178：失败阶段异质，Long93专家弱，Long35专家强；示范起点也可能失败 | 小面板要覆盖suite与有意义的Long能力参照，区分接触/规划/恢复 | 不把所有失败统一归为occupancy或时序丢失 |
| §114/123：批量、SDPA、cache、placement有实测吞吐收益；共享mmap减少重复读取 | 复用执行经验，设计阶段考虑真实数据布局 | 历史8.48倍不是新图速度承诺；Meta-on不能复用跨step冻结R-cache |

旧 v6 的强结果还伴随完整 train24、更广视频池、更多每 task queries、width256 与 trainable observer，和近期18task/两视频/
Meta-off图有多个耦合差异。不能将历史差距简单归因训练步数、单个 Meta 开关或普通 FactorHeads；本图没有检验这些单因果解释。
旧专家所建议的 Meta-off、双向 P/Q 或12+4路径属于当时上下文，最新 owner 已覆盖对应结构；其科学边界与证据仍保留。

## 13. 验证顺序、证据用途与失败分支

本节定义执行时应回答的问题；owner已于9月7日明确授权理解后连续科研执行，当前记录见progress。
具体 task/step/arms/预算在实现与真实 profile 后登记；不从旧运行清单恢复无关实验。

### 13.1 首先检查真实工程合同

用最小真实样本检查 input wall、18×50 capture、single probe、38-target materialization、finite、forward与梯度。
检查源参数与 vision/Gemma 不更新，读取 Meta 确实经实际 loss 获得梯度，执行不装读取 Meta。对 identity-init 的预期初步
零梯度按第8节解释；用启动后的正常小步确认上游通路，不加入人为 fake loss 冒充功能学习。

针对具体风险做有限验证：

- 验证每个帧对两端的row normalization、转置方向与signed gap；检查rho与A行的索引恢复，以及显式rho首层投影和相对位置读取
  的代数等价。多head/边界/padding不应混合不同j或video，也不把已softmax的A直接转置当另一端。
- 在固定采样位置的小真实condition上核对局部依赖：第b层输出不受半径bw外的图像内容影响，窗口内前后帧均允许参与；同一block
  读取旧U并同步更新。不要沿用“早期E不能读取任何未来帧”的旧测试，也不把每个输入扰动都应产生非零变化当模型合同。
  这些是工程语义核对，不以shuffled/reversed视频参与科学选择。
- K1/2/4真实不同视频，交换视频排列应保持生成输出的集合语义；不要求逐元素低位一致，不做LoRA均值基线来替代合同。
- 用小真实 condition 对照完整 autograd 与 staged VJP 的关键梯度/一个实际更新，验证链式语义与权重；这是有明确重放合同的
  限定验证，不是对全模型/全数据集逐 tensor 扫描。
- 验证当前 step 临时 R 的 replay 有效，更新 \(\mu\) 后不命中旧 R cache；冻结 prefix cache 的输入条件与有效期正确。
- longest K1/K4 profile 与真实 query batch通过；内存峰值、数据供给与阶段成本用于调整执行实现，不用降低H或剥离Meta梯度换速度。

### 13.2 尽快用训练侧学习与闭环决定投入

工程 smoke只证明运行。使用已有授权 tasks/真实视频，包含 Spatial/Object/Goal 和有实质上界的 Long，做有信息量的短学习
与行为检查；在固定训练预算、episode/query/weight口径下报告与强可比参照的 retained/gained/lost，避免只看均值loss。
如果要判断 shared 的代价，可在相同图与每 task监督量下用少量 clone诊断；不能把clone集合用于部署或泛化得分。

明确区分“可学但目前共享不足”“同task新视频失败”“未见task迁移失败”“functional与闭环分离”。不能因同一接口non-pass
就同时扫rank/seed/scale/LR/dtype、扩大所有模块或恢复旧冻结课程。需先用能区分竞争解释的最小证据定位接口。

新图同时改变多个已对齐职责，首次整体结果只能检验这个组合。若要论证坐标decoder、Meta或局部关系过程的单项效果，需要后续
matched 干预；ordinary FactorHeads是合理的输出参数化对照，但其参数/成本差异须报告。不能为了做“所有消融”延迟关键
closed-loop节点，也不能把此文列出的每个候选都自动排进一次全面扫描。

### 13.3 正式性能选择与最终视频 controls

有接近强基线/目标的广泛行为趋势后，及时做 fixed validation8 的 single-checkpoint strict paired400。
80-row screen、训练面板、checkpoint union、融合、内部 surrogate 都不选择最终模型。正式资格必须同时报告：

- 每 task、每 suite 与 breadth；Goal/Long 的实际成功贡献。
- 相邻 checkpoint 的 retained/gained/lost、churn、success-set overlap，排除偶然峰值。
- 同 task 无放回抽取另一组正确视频后的行为保留；K如果声称支持必须按登记cardinality验证。
- 正式同口径强参照。train24 SFT历史109/400与旧Writer143/400是能力参照；需明确哪部分已配对历史证据、哪部分新运行。

qualification arms 与相邻稳定口径先登记。selected checkpoint 选定并冻结后再做 full video 相对 language/no-video、
scene/first+final、cross-suite wrong 的必要条件controls，以及最终 shuffled/reversed 时序特异性测试。
shuffled/reversed 必须重排真实frames再完整forward，只作冻结后的最终证据，绝不进入训练、loss、checkpoint选择、Gate或
架构修正依据。禁止通过反复看它们来选择更合意的时序结构。

官方 rollout 固定 render256/model224、双相机180度rotate、执行state8/action7、10 flow steps、每次前5 actions后replan、
dummy settling10、成功即终止、suite horizon220/280/300/520。各臂严格配对 task/state/env RNG/policy RNG/video ordinal；
evaluator用cost-balanced dynamic queue、long-first、persistent workers。方法冻结后才按32 source/8 test合同fresh训练并
进行最终test；不能提前消费Test修正设计。

### 13.4 科学 non-pass 与工程故障的不同下一步

| 观测 | 可支持的定位 | 下一步边界 |
|---|---|---|
| permanent Meta零梯度、错误mask、越过声明局部范围的依赖、cache参数版本错、task权重错 | 可重复的工程合同违反 | 在原图职责内修复并重做受影响检查，不把故障结果作科学non-pass |
| 单task也缺少实际功能/行为 | 当前整个条件→LoRA组合未建立足够能力 | 检查最早可证伪接口和正确执行参照；不先宣称迁移问题 |
| clone强、shared弱 | 共享训练/表示/容量/优化有代价 | 用matched数据与有限干预区分，不能凭gradient cosine命名唯一根因 |
| train行为可学，同task新video下降 | 视频nuisance或过程证据消费有缺口 | 核对实际不同视频、条件覆盖与记忆消费；不复制更多同一视频凑K |
| K4更稳定但无性能/覆盖增益 | 可能只减少nuisance或丢失互补信息 | 报告真实non-pass，检查集合读出与不同策略冲突；不强迫few-shot优于K1 |
| functional好、closed-loop弱 | 局部监督与rollout能力没有等价关系 | 检查真实失败阶段和已有几何/行为证据；不统一归因occupancy |
| 单点高分、相邻或Goal/Long保持差 | 尚未满足稳定科学资格 | 完成必要相邻裁决，不用union或参数接近替代 |
| 最终视频因果controls不支持必要性 | 高分仍不足以完成EMBER因果目标 | 如实封存结论；不能用shuffled/reversed反馈修改本次已冻结方法 |

## 14. 实现前仍需明确的具体项

以下是执行配置与验证细节的未定范围，不需要重新讨论已认可的数据流：

1. 在清理后的代码 owner 下确定新图的单一入口与checkpoint schema；确认 prefix helper 和迁移后 schedule 的实际公开接口。
2. task/video/action episode roles、额外meta allowlist、K频率、每task查询量、optimizer与normalizer口径；pool要真实支持K4。
3. 公共probe seed、联合time/horizon bias及signed-gap编码的具体参数化、canonical末帧采样记录，以及新鲜随机初始化的完整参数规范。
4. 真实最长K1/K4下的batch/chunk/checkpoint/kernel选择及其测得峰值/吞吐；不能沿用旧Meta-off成本外推正式run。
5. 第一批能区分学习接口的训练侧任务、预登记行为节点、何时进入strict400和相邻qualification。不能预写新图分数或根因。

本文已经给出结构上完整的首版：一个读取侧Meta、一个单视频分层局部关系过程主干、一个多视频共同compiler、一个完整A/B
坐标decoder。接续 session 应理解这条链和历史边界，再按progress已登记的持续科研执行授权推进；无需再次批准实施计划。
本设计不授权外部专家联系或无依据的旧运行续接。
