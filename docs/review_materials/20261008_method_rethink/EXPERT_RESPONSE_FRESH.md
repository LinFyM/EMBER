# 新专家完整回复：真实实践条件下的共享参数修订（2026-10-09）

> 来源：Owner 于2026-10-09提供的完整粘贴文本。以下原文按原顺序保留；专家核查锚点为984b28ed，早于主讨论独立分析3a8969e0。正文的首批合同与启动建议属于待审阅提案，不是active design或实验授权。主讨论交叉审阅见[FRESH_RESPONSE_REVIEW](FRESH_RESPONSE_REVIEW.md)。

---

# 核心判断与推荐方案

**我推荐保留冻结 source、冻结强 MT、原生执行与功能信用实现，替换旧 Writer，建立一个“从完整教学直接生成参数、由真实实践修订参数、在独立新初态上学习修订效果”的共享编译器。**

具体选择是：

- 以强 MT 的完整 rank128 LoRA 为固定控制底座。
- 从教学视频和 exact language 直接生成覆盖全部 38 targets 的 rank64 残差 LoRA。
- 在两个不同的适应初态上，依次用当前完整 LoRA 做两次真实、完整任务尝试。
- 用实际执行前后的视觉、proprio、实际执行动作和结果，重新读取教学，更新持续存在的参数 token 状态，再直接输出下一套 LoRA。
- 最终将 MT 与残差精确拼成**一套 rank192、38-target、固定的完整 LoRA**。
- 新 condition 中没有动作目标数据集、拟合目标、任务内优化器或参数梯度更新。参数变化全部来自已经共享训练好的读取、修订和解码算子。

本方案最重要、也最可能失败的主假说是：

> **共享训练可以教会一个参数修订算子：把“当前控制器实际做成了什么、没有做成什么”与教学中的过程证据结合，改写同一套完整参数，使它在没有参与适应的新初态上更有效。**

这不是“再给视频接上功能梯度”。旧 T/C 已经有真实自身状态、跨 episode FM、完整参数作用以及视频模块的功能信用。这里真正改变的是：**共享学习的基本事件，变成了真实执行之后的参数修订，并把修订后的独立新初态能力作为训练和判断对象。**

我选择先完成 **K1**。本阶段不声称 few-shot 或动态 K，也不把未经相应 cardinality 训练的集合输入列为已有能力。

## 证据范围与结论强度

此次核查以仓库快照 `984b28ed8181f4cc59f910c9d2b6ce482088afb2` 为锚点，覆盖了最新入口、Owner 要求、概念和合同、历史图谱与机制，以及与本方案最近的完整 P/Q、self-read、Writer RL、跨情境配对、前四轮方法论证和 task23 的后续诊断。**本轮用户明确给出的排除边界优先于旧材料中的接受意见。**[当前入口](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/docs/review_materials/20261008_method_rethink/README.md) · [讨论衔接](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/docs/review_materials/20261008_method_rethink/DISCUSSION_BRIEF.md)

仓库中的源码、逐行 CSV 和归档论证可以核查；集群上的权重二进制、完整原始视频与所有运行日志没有在本环境逐一打开。因此，下面的历史分数是对既有记录的核对，**不是本次重跑结果**；新架构的速度、显存和成功率尚未测量。本次也没有启动新实验。

---

# 一、完整架构：三个阶段与参数归属

## 1. 三个阶段如何衔接

| 阶段 | 实际输入 | 核心计算 | 更新对象 | 产物与消费者 |
|---|---|---|---|---|
| **共享训练** | 合法 train24＋support12 的教学 RGB、exact language；同 task 其他 episode 的执行 query 与动作标签；当前模型真实实践和独立 query rollout | 教学读取 → 初始完整 LoRA → 两轮真实实践与共享修订 → 最终策略的 FM、真实终局信用 | 共享 Writer 参数 \(\chi\)；单独的训练期 value baseline 参数 \(\psi\) | 一个共享编译算法；不形成 held task 参数表 |
| **新 condition 编译** | 一条指定视频、exact language；两次由当前完整 LoRA 产生的真实尝试 | 三次教学读取、三次参数形成、两次反馈编码与修订 | 仅任务内激活状态 \(\mathcal Q_0,\mathcal Q_1,\mathcal Q_2\)；共享权重全部冻结 | 唯一最终 \(\Lambda_2\)，覆盖全部 38 targets |
| **最终执行** | 新初态的双 RGB、8 维 proprio、exact language、固定 \(\Lambda_2\) | 原生 10 步 ODE flow；生成 50 步 horizon，执行前 5 步再规划 | 无参数更新 | 完整任务闭环；Writer、视频、反馈编码器和 value baseline 均退出 |

两轮实践期间，每个 episode 内的 LoRA 也是固定的。Writer 只在 episode 结束后修订，不插入运行时阶段切换。

## 2. 冻结什么，训练什么

**冻结部分：**

- 已审计建立的 source 基础权重及其原生图文 prefix。
- source normalization、相机处理、动作和状态格式。
- 强 MT 的 38 对 \(A^{MT},B^{MT}\)。
- 用于教学底层特征的固定 source/MT 版本。

**需要新训练的部分：**

1. 教学的空间读取、时间编码及反馈条件重读模块。
2. 初始参数 token 生成器。
3. 真实经验编码与教学匹配模块。
4. 两轮共用的参数状态更新器。
5. 全坐标 A/B 解码头。
6. 仅共享训练使用的 value baseline。

**任务内状态不是可优化参数。** \(\mathcal Q_j\)、教学表示 \(P_j\)、经验表示 \(E_j\) 都是前向计算得到的激活；没有为它们建立 Adam 状态，也没有对它们做任务内搜索。

新 condition 中，实际动作的消费者是**经验编码器**。它们不会成为回归标签，更不会被整理成成功轨迹后训练该 condition 的 LoRA。

---

# 二、教学中读取什么：保留过程证据，同时避免伪造物理标签

## 1. 冻结特征的真实来源

对 stride5、包含末帧的教学序列 \(D=(I_{1:N},L)\)，每帧提取：

\[
\Phi_t\in\mathbb R^{512\times2048},
\qquad
H_t^{MT}\in\mathbb R^{50\times1024}.
\]

其中：

- \(\Phi_t\) 是现有 `embed_image` 路径的双相机视觉 token，来自视觉编码和投影，**不是经过完整图文融合后的 hidden**。
- \(H_t^{MT}\) 是固定 MT 在合法教学 prefix 下，对固定高斯探针的原生 action-expert 响应。
- 探针固定为 seed1729 的 \(50\times32\) 高斯噪声，flow time 固定为 \(\tau=1\)。
- 教学 prefix 使用现有 **state-omitted** 路径：没有示范 proprio，也不填一份假的零 proprio。
- exact language 的 token、embedding 和 mask 由现有原生处理路径产生。

这里的 \(H_t^{MT}\) 表示：

> 固定控制器在这个画面和语言下，如何处理同一个动作噪声探针。

它不是示范动作、逆动力学恢复结果，也不是示范者的接触或控制意图真值。[原生读取实现](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/src/ember/operator_writer/native.py)

教学和自身执行的 prefix 不强行做成相同长度。当前典型教学 prefix 为 \(512+200=712\) tokens；自身执行路径还包含缺失相机的 masked 槽，典型长度为 968。必须沿用真实 mask 和消费者。执行继续使用 render256/model224、双图 180° rotate、8 维 proprio、7 维物理动作和 32 维模型动作空间。[原生处理实现](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/src/ember/pi05_processing.py)

固定 \((\Phi,H^{MT})\) 可以缓存。缓存身份必须包含：

\[
(\text{RGB内容},L,\text{source hash},\text{MT hash},
\text{processor hash},\text{probe seed},\text{mask contract}).
\]

缓存文件名、task ID 和 episode ID 不进入网络。

## 2. 每帧读取，再做保序时间编码

Writer 宽度固定为 \(d=256\)，每帧使用 8 个读取 query，8 个 attention heads，FFN 宽度 1024。

对第 \(j\) 次读取，令上下文为：

\[
c_0=c_{\varnothing},
\]

\[
c_j=
\operatorname{LN}\!\left(
W_Q\,\operatorname{Pool}(\mathcal Q_{j-1})
+
W_E\,\operatorname{Pool}(E_{0:j-1})
\right),\quad j=1,2.
\]

`Pool` 是学习到的 attention pooling，不是原始帧平均。读取 query 为：

\[
r_m^{(j)}
=
\operatorname{LN}\left[
\bigl(1+\tanh(W_\gamma c_j)\bigr)\odot e_m
+W_\beta c_j
+W_L\bar L
\right],\quad m=1,\ldots,8.
\]

每个 query 分别读取：

- 全部 512 个视觉 tokens，带相机和空间位置标识；
- 全部 50 个原生 \(H\) tokens，带 action-horizon 位置标识；
- exact language tokens，带原生有效 mask。

例如，一帧的读取可以写成：

\[
f_{tm}^{(j)}
=
\operatorname{LN}\left[
r_m^{(j)}
+\operatorname{Attn}_{img}(r_m^{(j)},\Phi_t)
+\operatorname{Attn}_{H}(r_m^{(j)},H_t^{MT})
+\operatorname{Attn}_{L}(r_m^{(j)},L)
\right].
\]

各模态先单独 LayerNorm，再投影到 256 维；attention 采用标准带 mask 的多头计算。**不先平均 50 个 H，也不先平均相机、帧或最终 LoRA。**

随后对 \(N\times8\) 个 tokens 使用 4 层双向时间 Transformer。教学已经完整给出，编译期允许同时看前后文。时间编码保留真实呈现顺序、stride 和末帧间隔；不会把第几帧变成运行时阶段时钟。

过程 token 显式保留两个端点及其变化：

\[
p_{tm}^{(j)}
=
\operatorname{LN}\,
F_{\rm proc}
\left[
\tilde f_{tm}^{(j)},
\tilde f_{t+1,m}^{(j)},
\tilde f_{t+1,m}^{(j)}-\tilde f_{tm}^{(j)},
\bar L,
e_{\rm gap}
\right].
\]

末帧单独保留 endpoint token。这里没有除以未经核实的物理时间，也没有将差分命名为速度。

**同编号 slot 不等于同一个物体。** 时间 Transformer 可以跨 slot 交换信息；不同读取轮的 slot 也没有天然一致的对象身份。因此，不对 \(P_0,P_1,P_2\) 的同编号 slot 做所谓物理状态差分。

## 3. 视频相对语言和首末图像增加的内容

exact language 通常给出对象角色和任务目标。首末图像能够补充对象外观、初始关系和目标外观。中间视频还有机会提供：

- 哪个可动部件被操作，以及操作过程中相对几何怎样变化；
- 手、物体、容器之间的共同运动与分离；
- 接近、移动、退出和后续操作之间的实际衔接；
- 完成后一操作之前，前一操作留下了什么可见状态；
- 同一成功过程中出现的回访、重新接近和非单调进展。

这些是**候选可迁移控制知识**。本方案不把它们自动升级为接触、抓稳、阶段完成或因果必要性标签。

| 信息 | 本方案中的地位 |
|---|---|
| 视频帧、顺序、相机身份、exact language | 直接观测 |
| 冻结视觉和原生 H | 模型计算出的特征 |
| 视觉共同运动、关系变化、可能的操作段落 | 由共享网络推断 |
| 示范中的真实接触、力、物体位姿 | 教学输入中没有 |
| “这个动作是成功的必要条件” | 单条成功路径不能提供；本方案不构造此标签 |

相对于旧的差分写入，这里保留静态端点、变化和完整上下文；同时，实践可以在**压缩为 8 slots 之前**改变读取 query，重新选择原始 patch 和 H 中的证据。底层 VLA 特征仍只计算一次。

---

# 三、教学怎样形成每层 LoRA

## 1. 持续参数状态

参数状态固定为：

\[
\mathcal Q_j\in\mathbb R^{38\times64\times256}.
\]

它的两个离散轴分别对应目标层和残差 rank 分量，不对应 task。

初始状态：

\[
\mathcal Q_0=G_\chi(P_0).
\]

初始 query 由 layer、target type 和 rank-index embedding 相加形成。\(G\) 使用 4 个块，每块包含：

1. 同一 target 内沿 64 个 rank tokens 的 attention；
2. 同一 rank index 沿 38 个 targets 的 attention；
3. 对完整教学表示的 cross-attention；
4. FFN 和残差连接。

这样既允许不同层协同，也避免对 2432 个参数 tokens 做昂贵的全连接 self-attention。

修订使用两轮共享的 \(U_\chi\)：

\[
\mathcal Q_{j+1}
=
\mathcal Q_j+
g_j\odot
F_U\!\left(
\operatorname{LN}\mathcal Q_j,\,
\operatorname{Attn}(\mathcal Q_j,P_{j+1}),\,
\operatorname{Attn}(\mathcal Q_j,E_{0:j})
\right),
\]

\[
g_j=\sigma(W_g\operatorname{LN}\mathcal Q_j+b_g).
\]

\(U\) 使用 2 个同类 axial blocks，并增加经验 cross-attention。门 bias 初始化为 \(-2\)，更新分支最后一层初始化为零。因此初始更新接近 identity，但门不为零，不会同时封死两个梯度入口。

完整视频每轮都可读取，累计经验 \(E_0,E_1\) 都保留。不会用“当前阶段”裁掉后续教学。

## 2. 全坐标、分块 A/B 解码

原生 38 targets 的形状为：

| Target | 数量 | \(d_{\rm in}\) | \(d_{\rm out}\) |
|---|---:|---:|---:|
| action-expert Q projection | 18 | 1024 | 2048 |
| action-expert V projection | 18 | 1024 | 256 |
| action_in | 1 | 32 | 1024 |
| action_out | 1 | 1024 | 32 |

这些维度取自现有消费者，不能把 action_in/out 的真实 32 维接口改成 7 维。[LoRA 实现](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/src/ember/pi05_lora.py)

对每个 \(\mathcal Q_{j,\ell r}\)，使用两个共享坐标头，分别形成 A 行和 B 列。坐标块宽固定为 64：

\[
a^{raw}_{j,\ell r,c}
=
f_A\!\left(
\operatorname{LN}\mathcal Q_{j,\ell r}
+e_\ell+e_{\rm type}+e_c
\right)\in\mathbb R^{64},
\]

\[
b^{raw}_{j,\ell r,c}
=
f_B\!\left(
\operatorname{LN}\mathcal Q_{j,\ell r}
+e_\ell+e_{\rm type}+e_c
\right)\in\mathbb R^{64}.
\]

每个头为 \(256\rightarrow512\rightarrow64\) 的非线性 MLP。**坐标块标识在非线性之前参与计算。** 拼接后按真实维度裁切，尤其处理 32 维边。

残差因子定义为：

\[
\tilde a_{j,\ell r}
=
a_{\ell r}^{seed}+
\operatorname{Concat}_c a^{raw}_{j,\ell r,c},
\]

\[
A^R_{j,\ell}[r,:]
=
\frac{\tilde a_{j,\ell r}}
{\sqrt{\|\tilde a_{j,\ell r}\|_2^2+10^{-6}}},
\]

\[
B^R_{j,\ell}[:,r]
=
\frac{1}{\sqrt{64}}
\operatorname{Concat}_c b^{raw}_{j,\ell r,c}.
\]

其中 \(a^{seed}\) 是固定、非零的随机行模板；两个输出头末层可从零开始。这样初始 \(A^R\neq0\)、\(B^R=0\)。

**不能将残差 A/B 同时初始化为零。** 对 \(BAx\)，若二者都为零，两侧的一阶梯度都会为零。

分块非线性头解除的是旧 family-wide 线性输出 span 的一种限制。它仍然存在 256 维 token 和共享解码器带来的条件流形约束，**不等于每个 condition 都能自由指定任意五百万维参数**。

## 3. 最终唯一完整 LoRA

所有目标层都使用：

\[
A^*_{j,\ell}
=
\begin{bmatrix}
A^{MT}_\ell\\ A^R_{j,\ell}
\end{bmatrix},
\qquad
B^*_{j,\ell}
=
\begin{bmatrix}
B^{MT}_\ell & B^R_{j,\ell}
\end{bmatrix}.
\]

设最终 rank 为 192、\(\alpha=192\)，保持 \(\alpha/r=1\)，于是：

\[
B^*_{j,\ell}A^*_{j,\ell}
=
B^{MT}_\ell A^{MT}_\ell
+
B^R_{j,\ell}A^R_{j,\ell}.
\]

这是精确的算子相加，不会把 MT 缩成原来的 \(2/3\)。

**实现时 MT 只能计入一次。** 最终完整因子应安装在冻结 source、identity physical PEFT 的执行模型上；不能在已经物理安装非零 MT 的模型上再加一份含 MT 的完整 concat。现有 batched LoRA 执行路径对这一点有明确要求。[批量 LoRA 消费者](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/src/ember/batched_lora.py)

所有 targets 合计：

\[
\sum_\ell(d_{{in},\ell}+d_{{out},\ell})=80{,}448.
\]

因此：

- rank64 残差：\(5{,}148{,}672\) 个数；
- 完整 rank192：\(15{,}446{,}016\) 个数；
- FP32 完整单 condition：约 **58.92 MiB**。

选择 rank64 残差是当前的表达与成本取舍，不是由 held 分数扫描出来的最优值。

## 4. 固定参数如何在不同自身状态和 flow 阶段发挥不同作用

对真实执行 hidden \(x_\ell(o,z,\tau)\)：

\[
\Delta y_\ell
=
B^R_\ell A^R_\ell x_\ell
=
\sum_{r=1}^{64}
b_{\ell r}\,
\langle a_{\ell r},x_\ell(o,z,\tau)\rangle .
\]

这提供了一个明确的调用机制：

- A 行选择当前 hidden 中的方向；
- B 列把被选择的分量写回实际投影空间；
- 同一参数在不同观察、proprio、噪声 latent 和 flow time 下，产生不同作用。

对于 Q projection，修改会改变 action query 对原生 prefix 和 suffix 的读取：

\[
\delta{\rm score}
=
\frac{(\Delta q)^\top k}{\sqrt{d_h}}.
\]

对于 V projection，直接改变 action suffix 的 value 消息；后续层再传播这一影响。action_in 改变动作 latent 的初始表示，action_out 改变原生 velocity 输出。**不是每个 target 都直接读取操作阶段；例如 action_in 的真实输入首先是动作 latent。**

因此，完整固定 LoRA 可以表达依赖自身状态的控制变化，而不必复制教学时间表。但它能否学到正确的选择方向，取决于后面的功能监督；是否能迁移，则还取决于 source hidden 是否包含足够可用的信息。

---

# 四、实践反馈怎样改变理解和参数

## 1. 一条合法实践记录

每次原生重规划记录：

\[
e_n^{raw}
=
\left(
o_n,p_n,\,
a^{exec}_{n,1:m},\,
o_{n+1},p_{n+1},\,
r_n,d_n,\,
H_n^{1},H_n^{0.1},\,
{\rm masks},j
\right),
\]

其中：

- \(m\le5\)，只包含真正执行的动作；
- \(o_n,o_{n+1}\) 是真实执行前后双 RGB；
- \(p_n,p_{n+1}\) 是真实 proprio；
- \(r_n,d_n\) 区分实际成功、terminal 与时限截断；
- \(H_n^1,H_n^{0.1}\) 挂在该次真实 ODE 的第一次和最后一次 denoise forward 上；
- `masks` 标记提前终止和有效动作；
- \(j\) 标记该记录由哪一轮参数产生。

10 步原生调用没有额外的 \(\tau=0\) forward，因此不能把后一份 H 写成“\(\tau=0\) 的最终状态”。

H 仍然是模型内部信号；后 45 步未执行计划不能被标为真实后果。主方法不需要把物体真值或接触输入 Writer；这些允许采集的自身特权量可留给诊断，避免再引入一套尚未验证的物理标签体系。

## 2. 经验不是只剩一份“与教学匹配后的摘要”

先用不依赖教学匹配筛选的自身视觉读取器形成 \(u_n^-,u_n^+\)。真实 proprio、执行动作、结果和两份 H 也直接进入事实 token：

\[
e_n^{fact}
=
F_{fact}\left[
u_n^-,u_n^+,u_n^+-u_n^-,
p_n,p_{n+1},a^{exec}_n,
{\rm Read}(H_n^1),{\rm Read}(H_n^{0.1}),
r_n,d_n,{\rm masks}
\right].
\]

这条通路保证：**即使当前教学解释错了，未匹配上的实际后果也不会先被删掉。**

再做内容匹配。将自身起点特征作为 query，对当前教学的起点表示作 soft attention：

\[
w_{ntm}
=
\operatorname{softmax}_{t,m}
\frac{
q(e_n^{fact})^\top k(\tilde f_{tm}^{(j)})
}{\sqrt{256}}.
\]

分别汇聚教学端点和变化：

\[
c_n^-=\sum_{tm}w_{ntm}v^-(\tilde f_{tm}^{(j)}),
\quad
c_n^+=\sum_{tm}w_{ntm}v^+(\tilde f_{t+1,m}^{(j)}).
\]

关系 token 使用：

\[
e_n^{rel}
=
F_{rel}\left[
e_n^{fact},
c_n^-,c_n^+,c_n^+-c_n^-,
T(u_n^+-u_n^-)-(c_n^+-c_n^-),
{\cal H}(w_n)
\right].
\]

这里的“差”是**可学习特征空间中的不一致量**。它不是以米为单位的轨迹误差，也不是一个已经知道正确动作的控制残差。

将每个重规划的事实、关系两个 tokens 按真实顺序交给 2 层时间 Transformer，再用 16 个 attention queries 汇聚成：

\[
E_j\in\mathbb R^{16\times256}.
\]

两份经验都带轮次和时间标记保留。

## 3. 明确、无环的两轮修订

完整计算顺序固定为：

\[
P_0={\rm Read}_\chi(D,\varnothing),
\qquad
\mathcal Q_0=G_\chi(P_0),
\qquad
\Lambda_0={\rm Decode}_\chi(\mathcal Q_0).
\]

第 \(j=0,1\) 轮：

\[
e_j^{raw}
=
{\rm RealRollout}_{ODE}(\pi_{\theta_{src},\Lambda_j}),
\]

\[
E_j={\rm Experience}_\chi(e_j^{raw},P_j),
\]

\[
P_{j+1}
=
{\rm Read}_\chi
(D,\operatorname{Pool}\mathcal Q_j,\operatorname{Pool}E_{0:j}),
\]

\[
\mathcal Q_{j+1}
=
U_\chi(\mathcal Q_j,P_{j+1},E_{0:j}),
\]

\[
\Lambda_{j+1}={\rm Decode}_\chi(\mathcal Q_{j+1}).
\]

第二次尝试由 \(\Lambda_1\) 执行，不是再用 \(\Lambda_0\) 收一条数据。

每个 condition 固定两次尝试；即使第一次成功，也在另一适应初态执行第二次。每个 episode 仍遵守官方成功终止和原 horizon。之后直接交付 \(\Lambda_2\)，不在 \(\Lambda_0,\Lambda_1,\Lambda_2\) 中用评测结果择优。

---

# 五、共享训练怎样教会这些联系

## 1. 数据关系

只使用已经授权的 36 个映射任务，不把 source 见过的 71 个任务自动视为 Writer 训练 allowlist。

每个训练 condition：

1. 取一个 task 和一条教学，教学输入仍严格 action-hidden。
2. 从同 task 的其他 49 个合法 episode 中取 7 条 query episodes。
3. 每条 query episode 的四个时间区间各取一个有效 query，共 28 个。
4. 采集两轮当前策略的 ODE 实践。
5. 对最终 \(\Lambda_2\)，再在两个独立新初态采集 SDE query rollouts。

两个 SDE query 初态与两个实践初态互异。它们只存在于**合法 non-held 共享训练**；新 condition 编译没有这两个带训练信用的额外 query。

任务采用均匀轮转，teacher 使用每 task 的随机无放回袋。task ID 只用于合法数据采样和审计，不进入 Writer。

## 2. FM：所有阶段的完整策略功能监督

沿用原生动作 offset、padding 与 loss 消费者。当前路径是 action offset1；不足 50 步时重复最后动作；虽然 dataset 存在 `action_is_pad`，现用功能信用路径并未据此屏蔽尾部。因此本方案明确保持当前实际语义，不能由实现者自行换 mask。[数据实现](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/src/ember/writer/data.py) · [原生 FM 实现](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/src/ember/writer/functional.py)

令：

\[
a_\tau=\tau\epsilon+(1-\tau)a,\qquad
v^*=\epsilon-a,
\]

\[
\epsilon\sim{\cal N}(0,I_{50\times32}),
\qquad
\tau=0.001+0.999u^{2/3},\quad u\sim U(0,1).
\]

FM loss 为：

\[
\ell_q(\Lambda_j)
=
\frac{1}{50\cdot7}
\left\|
v_{\theta_{src},\Lambda_j}(o_q,a_\tau,\tau)_{:,:7}
-
(\epsilon-a)_{:,:7}
\right\|_F^2 .
\]

完整 32 维 latent 参与原生 forward，FM 直接监督前 7 维。

三个阶段的权重固定为：

\[
{\cal L}_{FM}
=
\frac14\bar\ell(\Lambda_0)
+\frac14\bar\ell(\Lambda_1)
+\frac12\bar\ell(\Lambda_2).
\]

增加一个低权重的功能退化惩罚：

\[
{\cal L}_{keep}
=
\frac{1}{2|Q|}
\sum_{j=1}^{2}\sum_{q\in Q}
\left[
\ell_q(\Lambda_j)
-
\operatorname{sg}\ell_q(\Lambda_{j-1})
\right]_+ .
\]

各阶段使用相同 query、noise 和 flow time。对前一阶段 loss 停止梯度，避免通过恶化前一阶段来减小惩罚。

这个目标保留的是**合法 query 上的动作功能约束**，不是不遗忘证明，也不是新初态成功保证。

## 3. 真实终局信用：直接作用到完整参数输出

我选择复用已经核实过的原生 SDE 概率核，只在共享训练的两个新初态 query rollout 中使用。实践和最终评测保持 ODE。

令 \(h=0.1,\ \tau_k=1-k/10\)，对全部 \(50\times32=1600\) 个坐标：

\[
c_k=1+\frac{1-\tau_k}{2},
\]

\[
m_k
=
z_k-h\left[c_kv_\Lambda(o,z_k,\tau_k)+\frac12z_k\right],
\]

\[
z_{k+1}
=
m_k+\sqrt{h\tau_k}\,\xi_k,
\qquad \xi_k\sim{\cal N}(0,I).
\]

这是实际采样的十个高斯转移，不是给 ODE 轨迹补一个假的 Gaussian likelihood。[已核实的实际 SDE 代码](https://github.com/LinFyM/EMBER/blob/1c90e7d5681bf18c99f9dc4084225d03e76ee929/src/ember/operator_writer/denoising_policy.py)

单个转移：

\[
\log p(z_{k+1}\mid z_k,o,\Lambda)
=
-\frac{\|z_{k+1}-m_k\|_F^2}{2h\tau_k}
-\frac{1600}{2}\log(2\pi h\tau_k).
\]

因此：

\[
\frac{\partial\log p}{\partial v}
=
-\frac{c_k}{\tau_k}(z_{k+1}-m_k).
\]

对 actor loss \(-\hat A\log p\)，传入真实 velocity forward 的余切为：

\[
g_v
=
\hat A\frac{c_k}{\tau_k}(z_{k+1}-m_k).
\]

\(\hat A=R-b_\psi\)，其中 \(R\) 是该 query rollout 的官方完整任务 success。baseline 只读取 query 开始前可知的、停止梯度的 condition 和初态特征；预测值在 rollout 前固定，不能先拟合本条回报再用于该条 actor advantage。

为了限制成本：

- 每条 query rollout 均匀 reservoir 保存最多 16 个真实重规划；
- 每个保存的重规划均匀选 2 个去噪转移；
- 保留逆采样概率权重：
  \[
  \frac{N_{\rm replans}}{\min(16,N_{\rm replans})}\cdot\frac{10}{2}.
  \]
- 不按 reward 选择记录，不以长度归一化替换这个权重。

通过真实 policy 求 \(\partial v/\partial\Lambda\)，得到全部 76 个 A/B tensors 的余切，再经同版本编译图回传给整个 Writer。现有代码已经证明这种“完整因子余切 → 同版本编译 replay”的实现路径存在；本方案增加的是实践修订图和混合功能目标。[实际信用代码](https://github.com/LinFyM/EMBER/blob/1c90e7d5681bf18c99f9dc4084225d03e76ee929/src/ember/operator_writer/denoising_credit.py)

## 4. 必须明确的停止梯度边界

采集两轮 ODE 实践时使用当前参数快照 \(\bar\chi\)。采集后：

- 实际 RGB、proprio、动作和结果停止梯度；
- 实际捕获的 \(H^1,H^{0.1}\) 也停止梯度；
- 经验编码、反馈条件重读、参数状态更新和解码器本身继续可训练；
- source、MT 和原生 prefix 始终冻结。

因此真正优化的是：

\[
J_{\rm cond}(\chi;\bar\chi)
=
\mathbb E_{
E\sim p^{ODE}_{\bar\chi},
\ \zeta\sim p^{SDE}_{\chi}(\cdot\mid D,E)
}
[R(\zeta)],
\]

在求 \(\partial/\partial\chi\) 时固定实践历史的生成分布。

**这是固定已收集实践历史条件下的 actor 梯度，是一种明确的 semi-gradient 共享学习；不是完整期望 meta-return 的无偏梯度。**

它没有直接优化：

- 第一轮如何专门探索，以获得最有信息的反馈；
- 改变 \(\Lambda_j\) 后，环境历史分布怎样随之变化；
- 捕获的自身 H 对其产生参数的导数。

每个 batch 都重新采集当前版本的实践，解决经验长期陈旧的问题，但不消除上述梯度截断。

另一个真实限制是 **SDE→ODE 代理差异**。十步离散随机核和最终 ODE 不是同一个目标。若 SDE 回报改善而 ODE 新初态能力不改善，应停止对这条训练代理的投入，不能更换正式采样器来兑现分数。

## 5. 固定训练选择

主 actor 目标：

\[
{\cal L}_{actor}
=
{\cal L}_{FM}
+0.2{\cal L}_{keep}
+\beta{\cal L}_{PG}.
\]

具体选择：

- AdamW，学习率 \(3\times10^{-5}\)，betas \((0.9,0.95)\)，weight decay 0；
- actor 全局 gradient clip 1；
- \(\beta\) 在第一份有效训练 meta batch 上，用共享参数梯度范数做一次校准，使初始 PG 梯度范数约为 FM 的 \(1/4\)，并上限截到 1；随后冻结；
- 校准只用 train 数据，不用 validation；
- value baseline 使用独立优化器和真实 query 回报监督，特征停止梯度，不参与 actor 的全局 clip；
- 训练时以 \(1/8\) 概率将两轮反馈都替换为 null，仍做两次 U 调用。该 mask 在采集前确定，并在 replay 中保持一致；
- 其余网络 dropout 设为 0，便于同版本确定性重放。

null 分支用于检验“真实反馈的作用”和“多做两次读取/编译计算的作用”。它不是 shuffled/reversed 视频训练。

## 6. 共享训练伪代码

```python
freeze(source, source_prefix, normalization, MT)
initialize_shared_writer()  # nonzero residual A template; zero residual B
initialize_value_baseline()

# Warm start: only legal cross-episode FM of lambda_0
for step in range(128):
    conditions = sample_4_legal_conditions_from_36_tasks()
    cotangents = []
    for D in conditions:
        frozen_features = read_or_cache_rgb_language_only(D)
        lambda_0 = compile_initial(frozen_features, chi)
        Q = sample_28_same_task_queries(exclude_teacher_episode=True)
        cotangents += native_FM_factor_credit(lambda_0, Q)
    replay_compiler_and_update_shared_chi_once(cotangents)

# Meta-training: never update chi between collection and score replay
for meta_step in range(meta_budget):
    version = snapshot_identity(chi, psi)
    batch = sample_4_legal_conditions()
    collected = []

    with no_grad():
        for D in batch:
            F = read_or_cache_rgb_language_only(D)
            P = reader(F, null_context)
            Qstate = initialize_parameter_tokens(P)
            lambdas, raw_history = [], []

            for j in (0, 1):
                lam = decode_complete_38_target_lora(Qstate, MT)
                lambdas.append(lam)
                raw = real_ODE_trial(
                    source, lam, distinct_adaptation_init(D, j)
                )
                raw_history.append(raw)
                E_j = encode_experience(raw, P)
                P = reader(F, pool(Qstate), cumulative_E())
                Qstate = update_tokens(Qstate, P, cumulative_E())

            lambda_2 = decode_complete_38_target_lora(Qstate, MT)
            query_rollouts = two_real_SDE_queries_on_fresh_inits(
                source, lambda_2, baseline_before_rollout=psi
            )
            legal_queries = sample_28_cross_episode_action_queries(D)
            collected.append(all_records_and_version_hashes())

    assert_shared_parameters_unchanged(version)

    # Native policy produces factor cotangents; factor leaves are not optimized.
    g0, g1, g2 = FM_and_keep_credit(collected)
    g2 += beta * true_SDE_score_credit(collected)

    # Replay only the finite compiler graph with raw experience treated as data.
    compiler_VJP_to_all_shared_modules(collected, g0, g1, g2)
    actor_optimizer.step()       # one shared update
    fit_separate_value_baseline_on_real_query_returns()
```

## 7. 新 condition 编译伪代码

```python
def adapt_one_condition(video, exact_language, adaptation_init_ids):
    assert_all_shared_weights_frozen()
    F = frozen_features_from_allowed_video_and_language_only(
        video, exact_language
    )

    P = reader(F, null_context)
    Qstate = initialize_parameter_tokens(P)
    E_history = []

    with no_grad():
        for j in (0, 1):
            lam = decode_complete_38_target_lora(Qstate, MT)
            raw = real_ODE_trial(source, lam, adaptation_init_ids[j])
            E_history.append(encode_experience(raw, P))
            P = reader(F, pool(Qstate), pool(E_history))
            Qstate = update_tokens(Qstate, P, E_history)

        final_lora = decode_complete_38_target_lora(Qstate, MT)

    return final_lora  # no loss, no optimizer, no candidate selection
```

---

# 六、用 task23 贯穿：开上层抽屉并把碗放进去

这个例子需要先保留真实历史的复杂性。

仓库已有只读 RGB 核查记录：teacher30 和 teacher16 都显示先操作抽屉，再取碗并放入；teacher16 还在取碗前后回到柜体附近。后续实际自身记录又表明，失败既包括没有充分操作上层，也包括曾打开却没有搬碗、开度明显缩回，以及搬碗后仍未放入。不能统一归因为一个“切换阶段失败”。[task23 完整核查与修订](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/findings.md)

官方完整目标是：

\[
{\rm In}(\text{bowl},\text{top drawer region}).
\]

“Open 曾经为真”是描述量，不是额外成功条件。既有诊断续行里，C16 完成 In 时开度约 10.07 cm，原生 Open 为 false；该续行也不能回填为正式400的新成功。[官方 BDDL](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl)

下面是**新方法尚待验证的计算推演**，不是声称我已经运行了新 \(\Lambda_0\)。

| 环节 | 证据与计算 | 预期参数作用及可失败预测 |
|---|---|---|
| 初次读教学 | 读取柜体可动部件、手的接近与退出、碗的后续运动、回访柜体等完整视觉变化 | \(P_0\) 同时包含早段和后段信息；没有手工输入“Open→Pick→Place”程序 |
| 形成 \(\Lambda_0\) | 参数 tokens 读取 \(P_0\)，输出全部 A/B 残差 | 在 MT 基础上形成对这条完整教学的控制调制；尚不能假设已会正确衔接 |
| 第一次实践 | 假如出现历史中那类“长时间在柜体附近，碗没有移动”的行为，真实动作、proprio、前后 RGB 和结果构成 \(e^{fact}\) | 反馈能区分“命令已经发出”与“场景发生了相应变化”；不能仅由接触或夹爪命令断言原因 |
| 反馈条件重读 | 实际停滞改变读取 query，重新查看抽屉操作后的退出、回访和搬碗证据 | \(P_1\) 可能增加此前未充分读取的退出和衔接内容；这种改变本身不算成功 |
| 修订为 \(\Lambda_1\) | \(U\) 将实际后果、重新读取的教学和 \(\mathcal Q_0\) 合并，改变 A 行选择方向与 B 列输出方向 | 应在“仍需操作柜体”的自身 hidden 与“可以转向碗”的 hidden 上产生不同作用，而不是整体压低所有柜体行为 |
| 第二次实践 | 在另一适应初态检查修订是否只对第一条经历有效，并获取新的实际后果 | 若修订只是过早转向碗，第二条经历会暴露它；共享 U 必须学会综合两轮证据 |
| 最终 \(\Lambda_2\) | 输出同一套完整参数，随后 Writer 退出 | 在第三个及其他新初态，自身观察应决定何时调用柜体相关作用、何时调用后续搬运作用 |

这个例子里，我不预先指定某一 rank 就是“退出抽屉”，也不手工给出一个动作方向。这样的语义必须由共享训练形成，并用行为检验。

**关键反例预测**是：如果修订只使机器人更早离开柜体，却没有学会在必要时保留前半段能力，那么它可能改善局部搬碗次数，同时降低完整 In 成功率。出现这种结果，应判为完整修订失败，不能用“阶段覆盖变多”保住主假说。

---

# 七、为什么它可能有效，以及哪些部分还没有依据

## 1. 三个层次必须分开

| 问题 | 目前能够说明什么 | 仍需实验证明什么 |
|---|---|---|
| **能否表示** | 38-target 残差可改变真实 Q/V、action_in/out；固定参数能通过当前 hidden 产生状态与 flow 相关作用 | 选定 rank、token 宽度和 source 表征是否足以表达所需完整控制 |
| **能否学到** | 合法 FM 和真实回报对完整参数输出、重读和修订模块都有明确消费者与信用路径 | 模型是否实际利用过程和反馈，还是靠语言、外观或训练 task 记忆降低目标 |
| **能否迁移** | 参数共享、跨 episode query、独立新初态 outcome 使迁移成为训练任务的一部分 | 36 个映射是否足以支持 held 任务上的关系重组与控制泛化 |

梯度存在、attention 改变、A/B 不同，都只回答其中很小一部分。

## 2. 旧能力的保留机制和限制

本设计用四项机制争取保留：

- MT 的算子项始终存在；
- 三个编译阶段都有完整任务 FM；
- 修订增加低权重的 query 功能退化惩罚；
- 最终训练信用来自新初态的完整任务结果。

但残差能够改变后续 hidden，从而干扰 MT 原有作用。精确相加、零初始化和小步训练都不能证明行为保留。必须报告 retained、gained、lost 和相邻 checkpoint 的成功集合。

长任务尤其如此：同一份最终参数必须同时保存不同自身状态需要的作用。保留完整视频和完整参数状态有利于做到这一点，但没有阶段 LoRA 拼接或外部 selector 作为补救。

另一个结构限制是可观测性：如果两个真实状态在原生 RGB/proprio 输入下不可区分，却需要不同动作，当前固定、无新增运行时记忆的策略未必能解决。编译期看过历史不会自动赋予最终策略运行时记忆。

## 3. 与最近似历史工作的关系

| 历史机制与证据 | 已经存在的功能 | 本方案实际改变的部分 |
|---|---|---|
| T/C | 真实自身状态、跨 episode FM、视频功能梯度、完整参数作用 | 改变共享训练事件：真实实践之后修订持续参数状态，并评价独立新初态 |
| C 的条件读写 | 条件读取和更强 seen 学习；held 未可靠突破 | 反馈在每帧压缩前改变读取 query；条件来自真实后果，而不只是教学内部特征 |
| 完整 P/Q | 未汇聚图像/语言、完整 H、过程 slots、完整 A/B 输出 | 保留其完整信息出口思路；用坐标块进入非线性的解码头，避开旧 family-wide 固定输出 span |
| 旧 self-read | 当前 LoRA 重读教学，完整梯度 | 新方法真的进入环境；保留持续参数状态及两轮累计经验 |
| LocalField / pullback | 已有直接参数作用、真实余切或功能信用 | 不再以增加另一种局部信用为主假说；真实信用用来训练“经历→修订→新初态能力” |
| 旧 Writer RL | 实际 on-policy rollout、非零 advantage、完整 Writer 梯度 | 在带真实修订过程的 condition 上混合 FM 与最终 outcome；承认旧 RL 阴性仍降低成功先验 |
| Within / Product | 已实际改变跨情境配对 | 不把小幅配对差额当成新编码器、自由输出和实践修订组合的实验结果 |
| 第四轮方法论证 | 已提出 MT、完整参数、真实实践、共享修订的大方向 | 这是概念前件；本方案补齐真实算子、标签消费者、停止梯度、概率核、预算和失败合同 |

历史上 T 的 161/400 对 MT 的 153/400 是保留117、新增44、丢失36；邻近点147/154/160/154/159没有形成稳定明显优势。correct／同 task 另一视频／public-only／wrong 的161/150/103/65支持教学条件有作用，但 public-only 不是独立训练的强语言 baseline。[逐行主结果](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/docs/review_materials/20260930_t_architecture/validation_rows.csv) · [条件对照逐行结果](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/docs/review_materials/20260930_t_architecture/selected_condition_rows.csv)

C 的 held 结果、旧 self-read、完整 P/Q、在线读视频和共享 RL 的正反证据都不能被清零。尤其旧 RL 的后点154/158没有超过 T161；本方案不能将其简单归咎于没有真实回报或没有更新完整 Writer。[机制汇总](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/docs/review_materials/20261007_research_reassessment/MECHANISM.md) · [完整机制复核](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/docs/analyses/feature_to_operator_mechanism_20260926.md)

外部方法中，**Watch, Try, Learn** 是较近的学习组织前件。它的视觉实现实际读取有序 demo/trial 图像和 trial reward，并用独立示范动作训练再尝试策略；不能因为形式定义里写了 \((s,a)\)，就说视觉 context 必然读取示范动作。不过它训练的是运行时条件策略，采用分开的两阶段策略和固定 trial 数据，既不是冻结 source 的参数编译，也没有 EMBER 的完整固定 LoRA 约束。[arxiv.org](https://arxiv.org/pdf/1906.03352?utm_source=chatgpt.com)

RL² 说明可以通过跨 episode 的观察、动作、奖励和终止信息共享学习快速适应，但它保留运行时 recurrent state，并优化整个 trial 的行为目标。它不能替本方案证明“经验可以被压进固定 LoRA”，也不能替这里的条件 semi-gradient 补上探索信用。[arxiv.org](https://arxiv.org/pdf/1611.02779?utm_source=chatgpt.com)

---

# 八、推进规划：先做小而完整的循环

下面的数值分成两类：

- **已测依据**：仓库历史运行中的吞吐、存储形状和 episode 数。
- **规划估计**：新 Reader、rank192、两轮串行实践与新训练图的成本，必须由首批 profile 替换。

## 1. 首批唯一独立静态参照

我选择**独立训练的“首末两帧＋exact language”参照**，与完整视频模型使用：

- 相同架构和初始化；
- 相同 task、teacher 映射、FM query 和更新数；
- 相同两次真实实践预算；
- 相同三次条件重读和两次 U；
- 各自策略产生的实际经验。

首末是两个有序 endpoint，不是相隔五步的真实局部 transition。该臂使用 unknown-gap 标记，不传原始视频长度和末帧绝对时间。

这匹配的是学习事件、动作监督和交互预算；两帧编码更便宜，GPU 实耗分别报告。

完整视频胜过它，支持**首末目标外观之外的中间教学内容具有整体功能价值**。仍不能单独归因为时间顺序理解或反馈模块。

## 2. 批次与依赖

| 批次 | 目标与工作 | 数据、参数更新范围 | 产物与判断 | 预计成本与扩大条件 |
|---|---|---|---|---|
| **B0：完整链实现与 profile** | 打通合法 teacher loader、冻结特征缓存、rank192 消费、两轮实践、同版本 FM/SDE VJP；测完整 meta batch 与 cold/warm 编译 | 仅合法 train/support；最多24个 profile episodes；不形成 held 选点 | 一条能运行的完整循环、来源审计、实际吞吐与显存表 | 工程约1–3天；1–4 GPUh估计。先修合同错误；不能凭局部模块指标批准学习结论 |
| **B1：小而完整的学习批** | 两臂各128个 FM warm steps＋32个 meta steps；meta16做训练域节点，meta32做训练域和小 validation 节点 | 只更新各臂共享 Writer 和独立 baseline；每 meta step 4 conditions | 检验视频中间内容、实际反馈、最终新初态能力是否同时出现方向性信号 | 连同 profile 预留30–55 GPUh，约10–20小时运行时间；新增存储硬上限128GiB。均为待测估计 |
| **B2：有上限的扩训与相邻正式节点** | 最多再96个 meta steps，到128；预登记96、112、128三个主模型 strict paired400 节点 | 数据 allowlist、架构、loss、LR、rank、R2预算冻结；两臂保持同更新预算 | 正式单 checkpoint 能力、稳定性、成功集合和覆盖 | 扩训约35–70 GPUh；三次主方法400另18–30 GPUh。只在B1真实功能证据满足资格后启动 |
| **B3：选点冻结后的机制资格** | 对冻结选点做独训首末帧参照、同 task 另一视频、\(\Lambda_0\) 与 null-feedback 诊断；需要时序主张时再做 shuffled/reversed | 不更新主模型、不重新选点；所有实践按各自 condition 独立执行 | 有益中间视频、真实反馈增量、时序敏感性的边界 | 约26–45 GPUh估计，分项启动。累计归档与缓存的全研究新增峰值按0.6–0.8TiB预留 |
| **B4：封存 Test 正式评估** | 固定模型、编译算法、RNG和预算，一次执行 Test K1 strict paired400 | 仅预注册编译中的真实实践可进入反馈；最终评测结果不得回流 | 最终泛化结果与完整成本 | 主方法6–10 GPUh估计；通过能力和视频增量判断后才解封 |

B2 的扩训是**一次有上限的机会**。如果只是辅助 loss 下降，或者反馈表示发生变化，没有训练域新初态的真实改善，就没有资格进入扩训。

## 3. B1 的明确规模

每臂：

\[
128\times4\times28
+
32\times4\times28\times3
=
25{,}088
\]

个 FM query 计算。

两臂共 **50,176 个 FM queries**。

真实训练交互：

\[
2\text{臂}\times32\text{步}\times4\text{conditions}
\times(2\text{实践}+2\text{query})
=
1024\text{ episodes}.
\]

每臂只有128条真实条件修订链，平均每 task 约3.6条。**这足以暴露强方向性信号和工程失效，不足以把 held 没有增益直接宣布为整个机制不可能。**

小节点固定为：

- train：按 manifest 预定8个 tasks，每 task 1个 condition，每 condition 3个新初态，共24行；
- validation：8个 tasks，每 task 2个 conditions，每 condition 3个新初态，共48行；
- 这些是小批迁移读数，不是 paired400。

meta32 比较：

\[
\text{完整视频 }\Lambda_2,\quad
\text{完整视频 }\Lambda_0,\quad
\text{首末帧 }\Lambda_2,\quad MT,\quad T.
\]

再对24个 train 行和24个 held 行，计算：

\[
\Lambda_{2,\varnothing}
=
\text{同一模型，null反馈，两次U调用}.
\]

meta16 只做固定 train 面板的 \(\Lambda_0/\Lambda_2\)。加上相应适应，开发节点约568个 episodes；再加最多24个 profile episodes，首批总交互上限取 **1650**，实际计划数约1616。

小面板的多初态属于同一 condition，分析必须按 condition 聚类，不能把48行当作48个独立教学条件。

## 4. B1 如何决定下一步

以下是我建议的新决策阈值，不是把它们冒充现有 Owner 合同：

**直接获得扩训资格的方向性信号：**

- validation48 上，完整视频 \(\Lambda_2\) 比 MT、T、独训首末帧三者中的最好者至少多4个完整成功；
- 新增不能全部来自一个 task；
- \(\Lambda_2\) 相对 \(\Lambda_0\) 有正增量；
- held 的 null-feedback 小诊断没有显示全部增益都来自额外两次编译计算。

**只获得一次受限扩训机会的情形：**

- train 两个节点都显示真实修订方向一致；
- meta32 的 train24 上，\(\Lambda_2-\Lambda_0\) 至少为 +3；
- held 没有相对 MT 或 \(\Lambda_0\) 净丢失4行以上；
- 信用来源、执行接口和数据边界均有效。

若这两类都不满足，不直接增加训练时长。先进入后面的失败分支判断，最多做与具体竞争解释相符的一项最小干预。

---

# 九、一个 condition 的成本，以及400次独立适应

## 1. 正确的计费单位

一次新 condition 的适应成本是：

\[
C_{\rm adapt}
=
C_{\rm frozen\ teacher}
+
3C_{\rm reader}
+
C_G
+
2C_U
+
3C_{\rm decode}
+
2C_{\rm real\ ODE}
+
2C_{\rm experience}.
\]

最终执行另计：

\[
C_{\rm final}=C_{\rm native\ ODE\ episode}.
\]

不是每个 task 适应一次后服务该 task 的50条视频。正式400对应：

- 400次初始编译；
- 400次 \(\Lambda_0\) 实践；
- 400次第一次修订；
- 400次 \(\Lambda_1\) 实践；
- 400次最终修订；
- 400次最终新初态执行。

因此，一次主方法 strict paired400 是：

\[
\boxed{800\text{实践}+400\text{最终执行}=1200\text{完整环境episodes}}
\]

并形成1200次参数输出，其中只有400套最终参数交付评测。

按四 suite 的当前 horizon，400个最终 episodes 的控制步上限为132,000；R2加最终执行为396,000步，另有1200次 reset 对应的 settling。成功提前终止可以降低实耗，但不能预先假定。[执行合同与资源要求](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/AGENTS.md)

## 2. 历史测量能提供什么参照

旧正式400的记录中：

- bank 形成约0.3445–0.3751 GPUh；
- 400条原生执行约1.0166–1.2508 GPUh；
- 即旧流程约每 condition 3.1–3.4 GPU秒的批量摊销 bank 成本；
- 每 episode 约9.1–11.3 GPU秒的批量摊销执行成本。

这些是批量摊销 GPU 时间，**不是单 condition 的墙钟延迟**。[历史运行记录](https://github.com/LinFyM/EMBER/blob/984b28ed8181f4cc59f910c9d2b6ce482088afb2/progress.md)

机械地将旧 bank 和执行各乘3，得到约4.1–4.9 GPUh的参照。新方法有缓存收益，也有更宽参数、两轮依赖、反馈存储和新 Reader 开销，不能把这个外推当作测量或严格下界。

我的规划占位值是：

| 项目 | 当前规划值 | 必须实测的内容 |
|---|---:|---|
| 单 condition 冷启动适应墙钟 | 约2–10分钟 | 单条件实际 episode 延迟、重读/解码、缓存未命中 |
| 正式400整轮，含800实践与400最终执行 | 6–10 GPUh | 完整调度、长任务尾部、I/O、rank192影响 |
| 6张获准 GPU 的400整轮墙钟 | 约1–3小时 | 实际并发、每轮依赖和资源占用 |
| 最终单 episode 执行 | 单独 profile | Writer 已退出后，rank192 对原生执行的增量 |

**在 B0 替换这些占位值之前，不据此批准大规模400适应。**

## 3. 缓存与存储

从现有 manifest 元数据计算：

- train36 的1800条教学约59,114个 stride5 frames；
- validation400约14,528 frames。
- 每帧 \(\Phi\) 与 H 的 BF16 缓存：
  \[
  512\cdot2048\cdot2+50\cdot1024\cdot2
  =2{,}199{,}552\text{ bytes}.
  \]

因此：

| 内容 | 体量 |
|---|---:|
| train36 完整固定特征缓存 | 约121.09 GiB |
| validation400 固定特征缓存 | 约29.76 GiB |
| 两者合计 | 约150.86 GiB |
| 400套最终 rank192 FP32 LoRA | 约23.02 GiB |
| 800次实践的双H与重规划前后原始双RGB上界估算 | 约30 GiB，另加少量动作、状态和元数据 |

首批采用**总额有界的按需缓存**，不提前生成整个150GiB缓存池。学习到的8-slot特征不能跨参数更新当作冻结缓存使用。

全研究需要保留多个 checkpoint、多个正式条件对照的真实实践原件和最终参数，累计存储明显超过“一个缓存＋一个bank”。因此我建议：

- 首批新增峰值硬上限128GiB；
- 完整研究累计新增峰值按0.6–0.8TiB预留；
- 可再生缓存与不可替代的实际交互原件分别计账；
- 不默默在两个节点复制两份完整缓存而只计一份。

## 4. 哪些可以复用

**可以合法复用：**

- 冻结 source、MT、normalization；
- 同 RGB、同 exact language、同冻结版本的 \(\Phi,H^{MT}\)；
- 同 condition 的 \(\Lambda_0\) 用于该次修订前后诊断；
- 冻结的正式 teacher、init、RNG 映射。

**必须独立产生：**

- 不同 condition 的实践；
- 不同 checkpoint 的实践；
- 完整视频与首末帧两臂各自策略产生的实践；
- correct、同 task 另一视频、shuffled/reversed 各自完整编译条件下的实践。

正式最终执行的结果永远不能回流到该 condition 的修订。

## 5. 资源约束

计划默认在**合计不超过6张 A40**的范围内完成，启动前再按现场空闲量准入；不会因为物理上有8张就默认可占满。单节点上限、所有 cache/train/eval 进程的合计占用按现有合同执行。

本架构已经为迭代成本做了实质取舍：

- 固定 MT 的原生教学特征可缓存；
- 实践 H 挂在已有 denoise 上，不增加 action-expert forward；
- 参数 attention 使用 axial 结构；
- 没有额外大型视觉模型、动力学模型或动作搜索器；
- 固定两轮，不进行无限重试；
- 只训练两个早期完整系统。

如果实测仍无法在首批预算内完成，应先解决重复原生计算、缓存和批处理问题。若必须删去一个核心学习环节才能运行，就应修改完整方法合同，不能在成本表中隐藏删减。

---

# 十、正式评估、视频证据与稳定性

B2 预登记主模型的 meta96、112、128三个 checkpoint，teacher映射跨 checkpoint 固定。最终只选择一个 checkpoint，最高 strict paired400 成功数胜出，同分取较早者。

建议把 **180/400** 作为这轮值得继续投入的能力目标：相对历史 MT153多27例、相对 T161多19例。它是研究决策阈值，不是统计显著性的替代。

正式报告同时包含：

- per-task、per-suite；
- 非零覆盖和新增能力集中程度；
- 相对 MT、T 的 retained/gained/lost；
- 相邻 checkpoint 的成功交集、双向流失和 churn；
- 按 task/condition 结构处理的不确定性，不把400行视为400个独立任务。

稳定性建议单列：至少两个预定邻点达到180，三个都不低于170，并检查相邻成功保留率是否达到约80%。如果只有一个高点，结论应是单点能力进展，不能称稳定突破。

K1 正式每 task 的50条合法 teacher 各用一次；同 task 另一视频按既有合同做固定无重复置换，不能重复挑好视频。小批中一套 LoRA 跑三个新初态的结果不承担这项资格。

选点冻结后：

1. 独训首末帧参照用与主选点相同的训练更新预算，不替它另做有利选点。
2. 比较 \(\Lambda_2\)、\(\Lambda_0\) 与两次 U 的 null-feedback 输出。
3. 做同 task 另一视频。
4. 若要提出时序特异性主张，再按既有合同做 shuffled/reversed。

shuffled/reversed 的下降只说明模型对顺序扰动敏感，还可能包含分布外失效。它不能单独证明对操作因果结构的理解。其每个视频条件同样需要独立R2适应，不能沿用 correct 条件的实践后只换一下输入。

只有主能力与有益视频增量获得支持后，才进入封存 Test。Test 的最终评测反馈不用于重新选点、改方法或增加适应。

---

# 十一、失败分支：什么结果会改变哪项决定

| 观测 | 竞争解释 | 优先核对的真实证据与最小干预 | 决定 |
|---|---|---|---|
| **视频内容可读，但没有参数效用** | 模型靠语言/端点完成监督；解码器忽略过程；过程输出改变了参数却没改善控制 | 独训首末帧对照；相同新初态的完整成功与 gained/lost；必要时只在合法 train 做一次限定的输出表达诊断 | 若 train 都没有中间视频增量，停止“继续堆过程编码会自然奏效”的假说；不凭 probe 准确率扩训 |
| **反馈表示改变，但修订无效** | 额外两次计算本身有效；反馈匹配错误；自然实践无信息；U 没学会有效参数变化 | \(\Lambda_0,\Lambda_2,\Lambda_{2,\varnothing}\) 的配对新初态结果；检查实际两次行为是否重复同一种失败 | 若 real feedback 不优于 null，R2主张判负；停止为两次实践付费的依据。任何改成零实践编译的后继都须重新登记资格 |
| **train 学会，held 无增益** | 36映射记忆；对象/关系组合覆盖不足；source特征在held不可用；经历分布移位 | 固定 train/held 新初态曲线、条件级成功集合、实际经验分布；用已存在原件定位一种具体缺口 | 仅按预定上限给一次扩训机会。到上限仍无held增益，降低共享映射迁移假说；新增meta数据须单独审计allowlist |
| **局部阶段改善，完整成功或保持变差** | 过早切换、前置状态没有维持、残差干扰、FM路径约束与真实成功冲突 | 官方完整goal、全过程实际记录、前后LoRA的R/G/L；禁止用阶段计数替代 | 不承认控制突破。只有证据直接指向某个训练联系时，做一次对应修改；不能无限加辅助头 |
| **适应初态有效，新初态失败** | 记住一次路径、场景偶然性、依据不充分的反馈过度修订 | 同一套LoRA在至少3个未适应初态的完整执行；检查动作是否依赖教师时间表或适应轨迹细节 | 直接降低“编译得到任务能力”的支持度；不增加同一初态重试充当进步 |
| **SDE回报上升，ODE不升** | 随机核代理获益、有限步离散差异、unused latent坐标上的取巧 | 同版本、固定新初态的ODE与SDE实际行为；先确认真实采样核和score一致 | 停止当前SDE训练代理的主张；不改正式ODE协议来兑现成绩 |

还要区分三类负结果：

- **工程合同违约**：MT双计、alpha错误、teacher偷读state、mask改变、旧版本score、错用未执行动作等。该结果不回答科学问题，应修复并保留失败记录。
- **学习尚未充分**：训练域完整功能仍在明确改善，且没有明显保持恶化，可以获得一次预登记、有限扩训。
- **有效科学阴性**：合同有效、训练域已有足够获取或已进入平台，而 held、反馈增量或新初态迁移持续不成立。它必须降低对应主假说的支持度。

一次低分不否定 EMBER 整个目标；但多次有效阴性也不能只被重新命名为“还缺一点训练”。

---

# 十二、首批合同草案

| 项目 | 固定内容 |
|---|---|
| **首批目的** | 检验完整教学→完整LoRA→两轮真实实践→共享修订→新初态闭环是否出现可复核的功能增量 |
| **唯一主设计** | 冻结source＋冻结MT128；生成残差64；完整rank192、alpha192、38 targets；K1；两轮ODE实践 |
| **共享模型** | d256；每帧8 slots；4层时间编码；参数状态38×64×256；axial初始生成与两轮共享U；64坐标块A/B头 |
| **参数初始化** | 残差A非零、B为零；U非零sigmoid门、零输出末层；MT只计一次 |
| **数据边界** | train24＋已审计support12；teacher仅RGB与exact language；跨episode动作只用于共享训练功能监督；无held动作资料 |
| **独立参照** | 首末两帧＋语言，独立训练；同更新、FM和两次实践预算；未知端点间隔，不传原视频长度 |
| **学习规模** | 每臂128 warm steps＋32 meta steps；每step4 conditions；每condition28个合法跨episode queries；meta含2实践＋2独立SDE queries |
| **信用与冻结** | 实践原始记录及实际H停止梯度；真实query score＋FM回到完整共享Writer；source/MT/prefix冻结；采集与replay之间不更新参数 |
| **学习参数** | Actor AdamW \(3\times10^{-5}\)，clip1；FM权重1/4、1/4、1/2；keep系数0.2；PG系数train-only一次校准；baseline独立优化 |
| **新condition规则** | 仅前向计算；两次实际尝试分别使用\(\Lambda_0,\Lambda_1\)；累计经验重读教学；固定交付\(\Lambda_2\)，无拟合和择优 |
| **节点与读数** | meta16固定train面板；meta32 train24行＋validation48行；保留MT/T、独训首末帧、\(\Lambda_0/\Lambda_2\)，并在train和held做少量null-feedback两次U诊断 |
| **预算** | 最多1650个实际episodes；规划30–55GPUh，首批硬上限60GPUh；新增存储≤128GiB；现场准入后合计≤6张A40 |
| **扩训条件** | 按正文B1真实功能门槛；仅loss、probe、attention或参数变化不构成资格；后继总meta上限128，不进行rank/LR/seed小扫 |
| **禁止事项** | 无新condition动作数据拟合LoRA；无任务ID路由；无阶段teacher/外部selector；无shuffled/reversed训练或选点；Test封存 |
| **交付** | 冻结代码与配置、完整来源/mask/因子身份、逐condition实践与最终执行记录、R/G/L与成本、所有不利结果及明确下一决定 |
| **职责** | 实验session负责实现、合同验证和运行；主讨论读取原件，判断继续、一次限定修订或停止相应假说，不等待Owner逐项指定实验 |

这套设计值得做首批完整验证，因为它把过去尚未落实的“真实经历怎样改变最终固定参数”变成了可执行、可追责的学习问题。它的成败应由**中间教学信息和真实实践是否共同带来新初态上的完整能力增量**决定。若只能读懂更多、改变更多，却仍没有这种参数效用，就应当停止保护这条具体机制。