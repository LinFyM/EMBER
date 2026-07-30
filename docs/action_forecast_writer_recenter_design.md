# EMBER Recenter：Action-Anchored Core-Keyed Centered Procedure Writer

状态（2026-07-30）：Recenter正式首段负结果后本文已退役，只作科学
provenance；当前authority为
[`action_forecast_writer_core_program_design.md`](action_forecast_writer_core_program_design.md)。
下文保留当时从v4、v5.2、v6、v7、v8、v10与Loom证据形成的设计合同，不得恢复
其可执行路径或把当时的“当前”表述当作活动状态。

## 1. 研究目标

EMBER Writer 要完成的映射始终是：

```text
exact task language
+ exactly one ordered raw action-hidden teacher video
→ one complete task-specific rank-16 LoRA
→ frozen π0.5-LIBERO source policy improves on new initial states
```

Writer 不应尽可能完整地重建视频，也不应根据静态任务描述重新学习一套通用
policy。它只需要回答两类问题：

```text
Core:
  这是什么任务，涉及哪些语义角色、目标、关系和约束？

Procedure:
  source policy在视频各状态上理解到怎样的高层操作？
  task-grounded环境证据沿真实顺序发生了怎样的变化？
```

LoRA 必须主要由有效 Procedure 决定；Core 只负责给 Procedure 定址和提供
受限条件。没有有效的时变 Procedure 时，Writer 应回到 public LoRA identity，
不能仅凭语言和静态场景生成一套有较大增益的通用 task policy。

## 2. 信息墙和不变输出

Writer 输入严格只有：

```text
task language
+ one raw teacher video
```

禁止输入：

- teacher action；
- proprio、state、object pose；
- reward、success、terminal；
- task/suite ID、filename、episode identity；
- hidden normalization；
- source policy rollout outcome。

teacher action 只进入 frozen source policy 下的 AS functional loss。video 与
action query 只要求 task 相同，不要求来自同一 episode，也不做逐帧配对。

输出保持：

```text
38 public LoRA targets
76 A/B tensors
320 routing slots
rank 16
alpha 16
dropout 0
```

source base、normalization、24/8/8 split、frame stride=5 和 official
evaluation contract 均不改变。

## 3. 为什么不沿用 Loom

Loom 的动机是显式构造 Teacher Procedure、Policy Procedure，再以两者差值
决定 LoRA。这个目标在信息墙内不可辨识：

1. Action Expert suffix 是 source policy 对当前画面的 action hypothesis，
   不是 teacher 在相邻两帧之间实际执行的 action；
2. raw-patch correspondence 能表示画面对应关系，但不能可靠区分任务相关
   效果、相机/遮挡变化与 teacher 高层意图；
3. teacher confidence 与 teacher-policy gap 因此没有可靠监督锚点。

Loom 首段 macro50/100/150/200 的 correct400 为
`79/106/105/112`。内部量显示 correspondence、confidence 和 gap 路径弱、
失活或方向反转。继续调整 confidence、gap scale 或 correspondence 权重，
只会给不可辨识变量打补丁。

更早版本提供了相反且可复用的证据：

- v5.2 的 `132/138/74/82/83` 证明稳定 Semantic Core、policy-native Action
  probe 与 causal Procedure 可以同时产生 absolute 和视频语义/顺序信号；
- v6 task-complete 的 single-checkpoint best 为 `143/400`，说明
  task-grounded visual transition 有用，但其下游增益偏弱；
- v6 old recipe 的 `121/122/111/84/47` 证明同一拓扑可以强烈传递顺序信号，
  也证明训练粒度会改变 Procedure 增益；
- v7 的 `8×L` joint matching 接近完全均匀，说明局部 action-effect 配对
  没有信息墙内的可识别依据；
- v8 的 event 几乎被 Effect 单边支配，说明把 Action 与 Effect 强行压成
  单事件会破坏 policy-native 主干；
- v10 的 Procedure-slot 归一化把很小的 Procedure 差异放大到固定尺度，
  absolute 降至 `103/400`，说明“Procedure 必须非零”还不够，其物理幅度也
  必须保留。

Recenter 因此删除 Loom 的 Teacher/Policy 双流、raw correspondence、
confidence 和 gap，不试图估计信息墙内不可辨识的 teacher action。它恢复已有
证据最强的 Action 主干，只让视觉变化作有界修正，并重建一个保留 Procedure
幅度的 compiler。

## 4. 总体拓扑

```text
task language
  └─ text-only Gemma + Text Meta-LoRA
       → Q_text [L,256]

each teacher frame I_f + task language
  └─ multimodal Gemma + VL Meta-LoRA
       ├─ multimodal task-token states M_f [L,256]
       ├─ image-position states P_f [256,256]
       └─ same-prefix Action Expert + Action Meta-LoRA
            → native suffix hidden [50,1024]
            → mean over all 50 positions
            → A_f [256]

Q_text reads P_f
  → G_f [L,256]

X_f = M_f + G_f
  → permutation-invariant Semantic Core
  → C [L,256]

G_f - G_(f-1)
  → A_f queries task-grounded transition
  → bounded residual R_f
  → Z_f = A_f + R_f
  → one causal Procedure
  → P_f^proc [256]

320 routing identities
  → read Core C for slot addresses
  → Core-keyed read of time-centered raw Procedure values
  → bounded multiplicative Core gate
  → amplitude-preserving slot mixer
  → eight factor heads
  → complete public rank-16 LoRA
```

Core 与 Procedure 在 compiler 前保持独立。没有 raw image、Core 或 Action
到 factor head 的额外旁路。

## 5. Foundation evidence 与 Meta-LoRA

### 5.1 三套 rank-4 Meta-LoRA

| 模块 | 作用 | 参数 |
|---|---|---:|
| Text Meta-LoRA | 生成 video-independent task query axis | 921,600 |
| VL Meta-LoRA | 适配每帧 image+language evidence | 921,600 |
| Action Meta-LoRA | 适配 Action Expert probe | 626,688 |

三者均覆盖各自 18 层 q/k/v/o。A 正常初始化、B 物理零初始化，frozen source
policy 本体不更新。

### 5.2 稳定 task query 与显式 patch evidence

task language 单独经过 text-only Gemma：

\[
Q_{\text{text}}\in\mathbb{R}^{L\times256}.
\]

每帧 multimodal forward 产生 task-token states 和 image-position states，
并共用一个 bias-free `2048→256` semantic projection：

\[
M_f\in\mathbb{R}^{L\times256},\qquad
P_f\in\mathbb{R}^{256\times256}.
\]

稳定 task query 显式读取真实 patch values：

\[
G_f =
W_o^G\operatorname{Attn}
\left(
W_q^G\operatorname{RMSNorm}(Q_{\text{text}}),
W_k^G\operatorname{RMSNorm}(P_f),
V=P_f
\right).
\]

`G_f` 不是新传感器，也不增加信息墙外信息。它只把 Gemma 已有的 patch
content 放到跨帧稳定的 task-token axis 上。v5.2/v6 的结果已证明这一路径是
可访问语义证据，而不是冗余装饰。

逐帧 Core evidence：

\[
X_f=M_f+G_f.
\]

## 6. Semantic Core

Core 沿用 v6 已验证的 frame-set 结构：

1. 对所有有效帧计算稳定 mean backbone；
2. 用 `Q_text` 对逐帧 normalized evidence 做 task-selected attention；
3. attention 的 value 是去 frame mean 后的 centered evidence；
4. mean backbone 与 selected residual 相加；
5. 两层沿 task-token ordinal 使用 RoPE 的 bidirectional content blocks。

对任意 frame permutation \(\pi\)：

\[
\operatorname{Core}(X_1,\ldots,X_F)
=
\operatorname{Core}(X_{\pi(1)},\ldots,X_{\pi(F)}).
\]

Core 表达任务语义角色、对象/目标关系、约束和语言中声明的目标结构；它不读取
frame position，也不负责从视频顺序推导实际教学过程。

## 7. Action-anchored visual transition

### 7.1 恢复 policy-native Action 主干

每帧使用原生 50-token suffix，不再稀疏抽取 8 个位置，也不构造局部
Action–Effect pairing：

\[
A_f =
W_A\left(
\frac1{50}\sum_{k=1}^{50}H^{AE}_{f,k}
\right)
\in\mathbb{R}^{256}.
\]

这里的 50-position mean 是 v5.2/v6 已验证的 source-policy 高层 action
hypothesis。它保留完整 action horizon 的统计信息，同时避免把 50 个低层
位置直接当作 50 个高层事件。

### 7.2 transition 只作有界修正

在该 arm 的实际输入顺序内重新计算：

\[
D_0=0,\qquad D_f=G_f-G_{f-1}.
\]

shuffled/reversed 必须先重排输入帧，再计算 \(D_f\)，不能重排正确顺序下预先
计算的 transition。

Action probe 查询 task-grounded transition：

\[
\begin{aligned}
Q_f &= W_q\operatorname{RMSNorm}(A_f),\\
K_f &= W_k\operatorname{RMSNorm}_0(D_f),\\
V_f &= D_f,\\
R_f^0 &= W_o\operatorname{Attn}(Q_f,K_f,V_f).
\end{aligned}
\]

其中 `RMSNorm_0` 保持零输入为零；V 使用 raw transition，不做 value
projection，也不先把幅度归一化。

视觉变化不能取代 Action 主干。令：

\[
\begin{aligned}
c_f &= 0.25\,\operatorname{stopgrad}(\operatorname{RMS}(A_f)),\\
s_f &=
\frac{c_f}{
\sqrt{c_f^2+\operatorname{RMS}(R_f^0)^2+\epsilon}
},\\
R_f &= s_fR_f^0,\\
Z_f &= A_f+R_f.
\end{aligned}
\]

因此：

\[
\operatorname{RMS}(R_f)\le 0.25\operatorname{RMS}(A_f).
\]

这不是经验 scale sweep，而是职责边界：Action 是 Procedure 的可用主干，
视觉 transition 只能说明“环境证据刚刚怎样改变”，不能凭自身构造一个
effect-only Procedure。若 \(A_f=0\)，则 \(R_f=0\)，不存在视觉旁路。

## 8. 单路 causal Procedure

\[
P^{proc}_{1:F}
=
\operatorname{CausalProcedure}
(Z_{1:F},\text{frame ordinals},\text{valid frames}).
\]

配置：

```text
width       256
heads       8
blocks      2
position    sampled-frame ordinal RoPE in Q/K only
value       raw content
causal      true
```

每个采样帧只保留一个高层 Procedure token。不存在 Teacher/Policy 双流、局部
8×L softmax、EventRead 或额外 confidence。causal contract 保证未来帧不能
改变更早的 Procedure state；完整 Writer 在编译时仍可读取整条视频。

## 9. Core-keyed centered Procedure compiler

### 9.1 320 个 routing identities

```text
18 expert layers × rank16 = 288
action_in × rank16        =  16
action_out × rank16       =  16
total                     = 320
```

module/layer/rank/query identity 只参与寻址，不直接进入 value content。

### 9.2 先读 Core，只形成地址

\[
C_s =
\operatorname{CoreRead}
(R_s,\operatorname{Norm}(C),V=C).
\]

Core slot 不直接送入 factor head。它只在下一步帮助每个 LoRA slot 判断应读取
Procedure 的哪部分。

### 9.3 读取 time-centered raw Procedure values

对有效时间：

\[
\bar P=\frac1F\sum_f P_f^{proc},\qquad
\widetilde P_f=P_f^{proc}-\bar P.
\]

实现用数学等价但数值更稳定的 reference-centered 形式：先在float32中取每条
序列第一个valid token \(P_r\)，计算
\(\Delta_f=P_f-P_r\)，再令
\(\widetilde P_f=\Delta_f-\operatorname{mean}(\Delta)\)。这样fp32/bf16和
非2幂长度下的constant Procedure都会逐元素精确为零，避免低精度直接求均值
留下伪Procedure残差。

\[
H_s =
\operatorname{ProcedureRead}
\left(
R_s+\operatorname{Norm}(C_s),
\operatorname{Norm}(P^{proc})\text{ with RoPE},
V=\widetilde P
\right).
\]

Q/K 使用完整 normalized Procedure 与 frame ordinal RoPE 选择时间位置；V
保留 raw centered content。若 Procedure 在整条视频上为常量，则
\(\widetilde P=0\)，所有 slot read 精确为零，而不是由 softmax 被迫读取某个
无效 token。

centering 删除的是 video-wide constant policy bias；保留的是随过程变化的
内容。它不把 Procedure 归一化到固定幅度。

### 9.4 Core 只能有界乘性调制

\[
g_s =
1+0.25\tanh(W_g\operatorname{Norm}(C_s)),
\qquad
\widehat H_s=H_s\odot g_s.
\]

\[
g_s\in[0.75,1.25].
\]

`W_g` 零初始化，所以训练开始时 \(g_s=1\)。不存在 additive Core、beta 或
Core-only residual。Core 可以根据任务语义增强或抑制 Procedure 的部分通道，
但不能在 \(H_s=0\) 时凭空产生 LoRA content。

### 9.5 保持幅度的 slot mixer

v10 的关键失败之一是 terminal normalization 把微小 Procedure 信号提升到
近固定尺度。Recenter 把幅度设为硬合同。对每个 slot：

\[
a_s=\operatorname{RMS}(\widehat H_s),\qquad
U_s=\frac{\widehat H_s}{\max(a_s,10^{-6})}.
\]

只在单位方向上做 slot self-attention 与 FFN：

\[
\begin{aligned}
B &= U+\operatorname{SlotAttn}(R+\operatorname{Norm}(U),V=U),\\
B &\leftarrow B+\operatorname{FFN}(\operatorname{Norm}(B)),\\
Y_s &= a_s\frac{B_s}{\max(\operatorname{RMS}(B_s),\epsilon)}.
\end{aligned}
\]

物理RMS用`torch.linalg.vector_norm / sqrt(width)`计算；PyTorch在精确零点
定义零subgradient。分母下限只用于稳定方向，不把小幅content判成inactive，
输出仍乘原始物理RMS。因此zero-input输出/梯度均精确零，near-zero信号仍可
传播，正常幅度保持齐次。

因此：

```text
H = 0        → Y = 0
H scaled by k→ Y scaled by k
RMS(Y_s)     ≈ RMS(H_s)
```

没有 terminal RMSNorm、额外 scale head 或 factor 后乘回的隐式增益。

## 10. Factor decoding 与初始化

`Y` 按 288/16/16 slots 切为 expert、action-in、action-out，分别进入八个
bias-free factor heads：

```text
256 → 256 → output width
GELU
```

最后一层物理零初始化。因此 step0 生成值精确等于 sealed template：

```text
all public LoRA-B = 0
public task LoRA is functionally identity
```

Meta-LoRA、Core、Procedure 和 compiler 使用正常可学习初始化；zero-init
factor output 形成明确的两阶段梯度：

1. 第一步先打开 factor output；
2. 随后梯度进入 Core、transition、Procedure 与 compiler。

## 11. 精确参数量

| 模块 | 参数 |
|---|---:|
| Text Meta-LoRA | 921,600 |
| VL Meta-LoRA | 921,600 |
| Action Meta-LoRA | 626,688 |
| shared semantic projection | 524,288 |
| task-query patch grounding | 197,120 |
| Action projection | 262,144 |
| Semantic Core | 1,836,544 |
| bounded visual transition | 197,120 |
| causal Procedure | 1,573,888 |
| Core-keyed centered compiler | 1,469,184 |
| factor heads | 2,179,072 |
| **total Writer** | **10,709,248** |

参数量略高于 corrected rank-128 Source-SFT 的 `10,297,344`，约 `+4.0%`。
该预算是软参照而非需要机械相等的目标。每个参数模块都有当前必要职责，不为
凑数缩窄硬件友好的 256 维，也不保留 Loom 的不可辨识模块。

## 12. 训练合同

首个实验继续使用已封存的 v6 task-complete fast-decay 合同，以隔离架构变量：

```text
GPU4–7
4 DDP ranks
每 rank 每 macro 依次处理6个不同tasks
每个task先处理long视频
每task 1 teacher video + B20 independent action queries
task内求均值，24 tasks等权
每macro只做一次AdamW update
cosine decay steps=400
fresh identity initialization
```

Recenter schema 与 Loom/v10/v6 checkpoint 不兼容，不能 resume 或 warm-start。
正式 launch 前必须在真实105-frame视频上依次测试 B20；只有 OOM 或连续多步
不稳定才直接退到 B16。profile、exact-resume、gradient reachability 和正式
run 必须属于 Recenter 本身，不能继承 Loom 的 seal。

首段计划 macro0→200，每25 macro checkpoint；固定评测
macro50/100/150/200 的 paired correct400，不做 checkpoint 融合。是否继续
第二小时由 absolute 曲线和 breadth 决定。

## 13. 实现与内部验收

实现前的确定性验收：

1. Core 对 frame-set permutation 数值不变；
2. transition residual RMS 不超过 Action RMS 的25%；
3. Action 为零时 transition 不能形成独立 Procedure；
4. causal Procedure 的未来帧不改变过去输出；
5. 任意 Core + constant Procedure 产生零 compiler slots；
6. 任意 Core + zero Procedure 产生 public LoRA identity；
7. Core gate 始终位于 `[0.75,1.25]`；
8. Procedure 乘常数 \(k\) 时 compiler content 同比例变化；
9. step0 public LoRA 精确 identity；
10. factor heads 打开后，Core、visual transition、Procedure 和 compiler
    均可收到非零梯度；
11. frozen source policy 的 trainable parameter count 为零；
12. Writer 精确参数量为 `10,709,248`。

真实训练后的机制分析至少报告：

```text
Action RMS
raw / bounded transition RMS
radial-cap activation fraction
Procedure temporal variance
centered Procedure RMS
Core gate range/distribution
pre/post mixer slot RMS ratio
correct/same/wrong/shuffled/reversed 的
  Core → Procedure → effective LoRA → policy action 差异
```

## 14. 成功门与非目标

focused absolute 硬门保持：

\[
\operatorname{correct400}
\ge
\max(150,\operatorname{SourceSFT}_{best}+30)
=150.
\]

absolute 达门后，再要求：

- same-task-other 与 correct 同档；
- correct 显著优于 cross-suite wrong、shuffled、reversed；
- 改善来自多个 tasks，而不是单个 task；
- 独立 RNG/video permutation 复测成立；
- 内部差异沿预定职责传到 effective LoRA 和 policy action。

当前版本明确不做：

- teacher action reconstruction；
- teacher/policy latent subtraction；
- raw patch correspondence；
- learned teacher confidence；
- 8-token local Action–Effect binding；
- optical flow、3D geometry 或额外视觉编码器；
- order/contrast auxiliary loss；
- shared trainable adapter 或 residual escape；
- multi-checkpoint fusion；
- 通过扩大 LoRA scale 补偿结构问题。

若 Recenter 未达到同期 v5.2/v6 水平，应先用内部数值判断失败位于 Action 主干、
transition 使用、Procedure 动态范围还是 compiler 传递；只有形成充分根因证据
后才重新推导下一架构，不能围绕单个低分 checkpoint 追加门、scale 或旁路。
