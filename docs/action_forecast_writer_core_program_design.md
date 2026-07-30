# EMBER Core-Program Writer：Core Semantic License × Raw Video Program

状态（2026-07-30）：本文是当前唯一 canonical Writer 架构 authority。它以
Recenter 的正式负结果为新证据，从 EMBER 所需的信息与必要计算重新推导；不是
在 Recenter 上增加 scale、gate 或旁路补丁。Recenter、Loom 与更早架构只保留
为科学 provenance，活动源码和配置只有本文定义的一条 fresh、不兼容路径。

## 1. 任务与信息墙

EMBER Writer 要学习：

```text
exact task language
+ exactly one ordered raw action-hidden teacher video
→ one complete task-specific rank-16 LoRA
→ improve the frozen π0.5-LIBERO source policy on new initial states
```

Writer 只能读取 task language 和一条 raw teacher video。它不能读取 teacher
action、proprio/state、object pose、reward/success/terminal、task/suite ID、
filename、episode identity、hidden normalization 或 policy outcome。source
action 只进入 frozen policy 下的 AS functional loss；video 和 action query
只要求 task 相同，不要求 episode 或逐帧配对。

public adapter 合同不变：

```text
38 LoRA targets
76 A/B tensors
320 rank slots
rank 16 / alpha 16 / dropout 0
```

source base、source-only normalization、24/8/8 split、frame stride=5 与
official evaluation contract 均不改变。

## 2. Recenter 负结果改变了什么

Recenter 首段 macro50/100/150/200 的 paired correct400 为：

```text
55 / 84 / 79 / 85
```

它不只是略低于 v6 best `143/400`：所有 validation tasks 都低于 v6 best，
Object-3 发生明显坍塌。内部分析同时排除了“模型没更新”和“factor head 太弱”：
各模块参数持续移动，factor 输出幅度反而更大，能力却没有回升。

Recenter 的结构性错误是 semantic-basis starvation：

1. Procedure 在进入 compiler 前被 time-centering，删除了整条视频共享的
   DC component；
2. Core 只提供 slot address 和 `[0.75,1.25]` 标量调制，不再提供 LoRA
   content basis；
3. slot mixer 再把剩余的小 AC Procedure 恢复到固定幅度。

于是模型被迫从很小的时变残差中同时重建“任务语义方向”和“视频程序系数”。
历史量级中 centered Procedure RMS 只有约 v5.2 `.024`、v6 `.062`；放大它并
不能恢复已被结构删除的语义基底。constant nonzero Procedure 被强制视为
identity 也不合理：跨时间稳定的 policy-native action hypothesis 可以是
完成任务所必需的公共程序，而不是无效视频。

因此新架构不再问“如何给 centered Procedure 更大增益”，而重新满足两个必要
条件：

```text
Core      提供这个 task/slot 可以使用的高层语义基底
Procedure 提供完整视频程序，包括公共 DC 和视频特异 AC
```

任一缺失都不能产生 adapter；二者同时存在时才生成 LoRA content。

## 3. 第一性原理职责

### 3.1 Core

Core 表达跨帧保持身份的高层信息：任务是什么、对象和目标分别扮演什么语义
角色、有哪些关系/约束，以及 task language 声明了怎样的目标结构。它必须：

- 以 task-token 为可变长度容量；
- 对 frame-set permutation 严格不变；
- 能适应视角、遮挡和像素位置变化；
- 不凭自身直接生成有效 LoRA。

### 3.2 Procedure

Procedure 表达 source policy 对视频中高层操作的理解，以及 task-grounded
环境证据沿真实顺序怎样变化。它必须：

- 保留 policy-native Action 主干；
- 用视觉 transition 补充而非替代 Action；
- 对 shuffled/reversed 的实际输入顺序重新计算；
- causal；
- 保留完整 raw time signal，包含 DC 与 AC；
- 不凭自身直接生成有效 LoRA。

### 3.3 Compiler

compiler 必须让 Core 与 Procedure 第一次发生任务化融合：

```text
Core 决定语义基底与读取地址
Procedure 决定该基底上的视频程序
二者严格相乘后才产生 LoRA content
```

这不是 Core gate，也不是 Procedure residual。Core 与 Procedure 在生成
content 时具有对称的必要性，但职责不同。

## 4. 总体拓扑

```text
task language
  └─ text-only Gemma + rank-4 Text Meta-LoRA
       → Q_text [L,256]

each frame I_f + task language
  └─ multimodal Gemma + rank-4 VL Meta-LoRA
       ├─ task-token states M_f [L,256]
       ├─ image-position states P_f [256,256]
       └─ same-prefix Action Expert + rank-4 Action Meta-LoRA
            → native suffix hidden [50,1024]
            → mean over all 50 positions
            → A_f [256]

Q_text reads raw P_f values
  → G_f [L,256]

X_f = M_f + G_f
  → permutation-invariant Semantic Core
  → C [L,256]

D_0=0; D_f=G_f-G_(f-1)
A_f queries D_f with raw D_f values
  → uncapped R_f
  → Z_f=A_f+R_f
  → two-layer causal Procedure
  → P [F,256]

320 routing identities
  → read raw Core values into C_s
  → routing+Norm(C_s) reads full raw Procedure values into H_s
  → width-512 strict bilinear content F_s
  → zero-preserving slot coordination
  → eight factor heads
  → complete public rank-16 LoRA
```

不存在 raw image、Core、Action、Procedure 或 routing 到 factor content 的
额外旁路。

## 5. 已验证上游证据

### 5.1 Meta-LoRA 与稳定 task axis

保留三套已验证的 rank-4 Meta-LoRA：

| owner | 作用 | 参数 |
|---|---|---:|
| Text Meta-LoRA | video-independent task-token query | 921,600 |
| VL Meta-LoRA | frame+language evidence | 921,600 |
| Action Meta-LoRA | policy-native Action probe | 626,688 |

同一 bias-free `2048→256` 投影形成：

\[
Q_{\text{text}}\in\mathbb{R}^{L\times256},\qquad
M_f\in\mathbb{R}^{L\times256},\qquad
P_f\in\mathbb{R}^{256\times256}.
\]

稳定 task query 读取真实 patch values：

\[
G_f =
W_o^G\operatorname{Attn}
\left(
W_q^G\operatorname{RMSNorm}(Q_{\text{text}}),
W_k^G\operatorname{RMSNorm}(P_f),
V=P_f
\right).
\]

Q/K/O 可学习、bias-free；V 不经过 \(W_v\)。`G_f` 把每帧视觉证据对齐到
稳定 task-token 语义轴。逐帧 Core evidence 为：

\[
X_f=M_f+G_f.
\]

保留 \(M_f\) 是因为它包含 frozen Gemma 已完成的图文交互；保留 \(G_f\) 是
因为它显式地从真实 patch value 读取可跨帧比较的证据。二者来自同一 prefix，
不是两套视觉编码器。

### 5.2 Semantic Core

Semantic Core 原样沿用 v6 已验证结构：

1. 对有效帧计算不可丢失的 mean backbone；
2. `Q_text` 选择相对 frame mean 的 centered residual；
3. 两者相加；
4. 经过两层沿 task-token ordinal 使用 RoPE 的双向 content blocks。

它对任意 frame permutation \(\pi\) 满足：

\[
\operatorname{Core}(X_1,\ldots,X_F)
=
\operatorname{Core}(X_{\pi(1)},\ldots,X_{\pi(F)}).
\]

Core 输出 \(C\in\mathbb{R}^{L\times256}\)；task-token 数 \(L\) 自然随任务描述
与目标关系复杂度变化。

## 6. Raw Video Program

### 6.1 policy-native Action

每帧仍使用 source Action Expert 原生 50-token suffix：

\[
A_f =
W_A\left(
\frac1{50}\sum_{k=1}^{50}H^{AE}_{f,k}
\right)
\in\mathbb{R}^{256}.
\]

它是 frozen source policy 对当前 frame+task prefix 的 action hypothesis，
不是 teacher action，也不用于局部 action-effect 配对。

### 6.2 uncapped task-grounded transition

在该 arm 的实际输入顺序内计算：

\[
D_0=0,\qquad D_f=G_f-G_{f-1}.
\]

Action 读取 task-grounded 视觉变化：

\[
\begin{aligned}
Q_f &= W_q\operatorname{RMSNorm}(A_f),\\
K_f &= W_k\operatorname{RMSNorm}_0(D_f),\\
V_f &= D_f,\\
R_f &= W_o\operatorname{Attn}(Q_f,K_f,V_f),\\
Z_f &= A_f+R_f.
\end{aligned}
\]

这里恢复 v6 的 uncapped residual。Recenter 的四分之一 cap 并没有解决
semantic-basis starvation，反而人为限制了已被 v6 证明能提高 absolute 的
视觉路径。`RMSNorm_0` 保持零输入为零，V 使用 raw transition 且无 \(W_v\)；
不加入 gain、confidence、null token 或 Action-zero 特判。

### 6.3 causal Procedure

\[
P_{1:F}=
\operatorname{CausalProcedure}
(Z_{1:F},\text{sampled frame ordinals},\text{valid frames}).
\]

配置为 width256、8 heads、2 blocks，RoPE 只进入 Q/K，V 保留 raw content。
每个采样帧只有一个高层 token。compiler 读取完整 raw \(P\)，不做跨时间
centering、terminal normalization 或 amplitude restoration。

## 7. Core-licensed raw-Program compiler

### 7.1 routing identities

320 个 routing identities 仍为：

```text
18 Action-Expert layers × rank16 = 288
action_in × rank16               =  16
action_out × rank16              =  16
```

module/layer/rank identities 只构造 routing address，不作为 value/content。

### 7.2 Core semantic basis

令 normalized routing 为 \(R_s\)。先读取 raw Core values：

\[
C_s =
W_o^C\operatorname{Attn}
\left(
W_q^C R_s,
W_k^C\operatorname{RMSNorm}(C),
V=C
\right).
\]

没有 \(W_v^C\)。\(C_s\) 是该 LoRA slot 的 task-conditioned semantic basis，
不是一个标量 gate。

### 7.3 Core-keyed full Procedure read

\[
H_s =
W_o^P\operatorname{Attn}
\left(
W_q^P(R_s+\operatorname{RMSNorm}(C_s)),
\operatorname{RoPE}(W_k^P\operatorname{RMSNorm}(P)),
V=P
\right).
\]

Procedure value 是完整 raw \(P\)，不减 time mean、无 \(W_v^P\)。因此：

- constant nonzero \(P\) 可以形成有用的公共视频程序；
- AC 顺序差异也完整保留；
- Core 决定每个 LoRA slot 应从 Procedure 读取什么；
- routing/Core 只进入 Q/K，不直接注入 Procedure value。

### 7.4 strict bilinear content

\[
F_s=
W_o^F\left[
\operatorname{SiLU}
\left(W_c\operatorname{RMSNorm}(C_s)\right)
\odot
W_pH_s
\right],
\]

其中：

```text
Wc: 256 → 512, bias-free
Wp: 256 → 512, bias-free
Wo: 512 → 256, bias-free
```

width512 不是为了机械吃满参数，而是让 Core 可提供多个语义 basis、Procedure
可独立提供对应 coefficients，再进行逐通道组合。该结构严格保证：

\[
C_s=0\Rightarrow F_s=0,\qquad H_s=0\Rightarrow F_s=0.
\]

因此 Core-only 与 Procedure-only 都不能生成 adapter；无需人为 gate、scale
或 penalty 来维护职责。

### 7.5 zero-preserving slot coordination

320 个 \(F_s\) 仍需在生成公开 A/B rows 前交换 layer/rank/module 上下文。单层
slot block 使用：

```text
Q/K address = RMSNorm(F) + routing
V/content   = bias-free Wv(F)
residual attention
residual bias-free 4× FFN
```

routing 只影响 Q/K；所有 value/content 都来自 \(F\)。全部线性层 bias-free，
没有 terminal norm 或输入幅度恢复。因此：

\[
F=0\Rightarrow \operatorname{SlotBlock}(F)=0.
\]

最后八个 bias-free `256→256→factor width` heads 生成所有公开 LoRA rows；
末层物理零初始化，step0 public LoRA 保持 functionally identity。

## 8. 可验证不变量

实现必须逐项满足：

```text
frame permutation      → Core unchanged
future frame change    → earlier Procedure states unchanged
zero transition        → no fabricated transition content
zero Procedure         → exact identity LoRA
zero Core              → exact identity LoRA
constant nonzero P     → preserved, not forced to identity
routing-only input     → cannot create factor content
all forbidden inputs   → absent from Writer forward
```

shuffled/reversed arm 必须先重排 raw frames，再重新计算 \(G_f,D_f,P_f\)；不得
复用 correct-order transition。

## 9. 精确参数预算

以真实 module enumeration 为准：

| 模块 | trainable 参数 |
|---|---:|
| Q_text / M/G / Action semantic encoder | 3,453,440 |
| v6 Semantic Core | 1,836,544 |
| uncapped Visual Transition | 197,120 |
| 2-layer causal Procedure | 1,573,888 |
| Core-Program compiler | 1,665,792 |
| eight factor heads, hidden256 | 2,179,072 |
| **总计** | **10,905,856** |

compiler 内部为：

| 子模块 | 参数 |
|---|---:|
| routing tables + routing norm | 91,648 |
| raw-value Core reader | 196,864 |
| Core-keyed raw Procedure reader | 197,120 |
| width512 bilinear fusion | 393,216 |
| zero-preserving slot block | 786,944 |
| **compiler 总计** | **1,665,792** |

总量比 corrected rank-128 Source-SFT 的 `10,297,344` 多 `608,512`
（约 `5.91%`）。参数上限在当前阶段是软约束；新增容量全部服务于“Core
semantic basis × Procedure program”的必要计算，没有扩大 public LoRA、
增加旁路或保留旧 executable。

## 10. 训练与判定合同

首个实验继续使用已封存的 task-complete fast-decay 合同，以隔离模型架构变量：

```text
GPU4–7 only
4 DDP ranks
6 long-first tasks per rank
one video / one generated LoRA / B20 action queries per task
mean within task, equal mean over 24 tasks
one AdamW update per macro
fresh identity initialization
macro0→200, every25 checkpoint
```

新模型必须独立完成最长105-frame B20真实profile、fresh0→1→exact-resume1→3
和全参数gradient reachability；不得继承 Recenter 的 profile/resume evidence。

一小时段只对 macro50/100/150/200 做 paired correct400，选择 single
checkpoint，不做融合。未恢复同期 v5.2/v6 水平时，不浪费 full specificity
rollout，先做 Core/Procedure/compiler/effective-LoRA/action 内部反事实。
恢复同期水平则按右端趋势决定第二小时；达到 absolute `150/400` 或稳定接近
该门后，才做 same/wrong/shuffled/reversed full400，并要求多个 tasks 共同
贡献。

视频特异性是 EMBER 确实从输入视频学习的证据，不是牺牲 absolute 的独立优化
目标。最终目标仍是 single-checkpoint absolute performance 显著高于封存的
Source-SFT `109/400`。
