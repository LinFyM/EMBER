# EMBER Writer v8：Hierarchical Action–Effect Event + Core-Gated Procedure

状态：2026-07-29 已完成并因event被Effect主导停止；本文只作provenance。
当前canonical authority为
[`action_forecast_writer_recenter_design.md`](action_forecast_writer_recenter_design.md)。
不得恢复v8/Recenter并行可执行路径。

## 1. 为什么必须从 v7 继续修改

v7 的设计目标是让 8 个 Action Expert suffix anchors 与任务语义变化共同产生
每个帧间隔一个高层事件。它完成 task-complete macro0→400 后：

```text
correct400:
macro50/100/150/200/250/300/350/400
82 / 106 / 114 / 120 / 101 / 114 / 115 / 106

macro200 five-arm:
correct / same / wrong / shuffled / reversed
120     / 112  / 91    / 100      / 69
```

相对 v6，v7 的 temporal direction specificity 更强，但 absolute 更低。内部
检查把失败精确定位到两个接口，而不是笼统归因于“训练不够”：

1. `8×L` joint softmax 的 pair-logit std 只有约 `0.058`，归一化熵约为理论
   均匀熵的 `99.96%`，8 个 Action anchors 的有效数量约 `7.998/8`。它几乎
   平均读取所有 action–effect pairs，没有学会 Action 对 effect 的选择性绑定。
2. 固定 Procedure、只改变 Core 时，effective-LoRA relative L2 的中位数只有
   约 `0.001–0.002`；固定 Core、只改变 Procedure 时则几乎完整复现最终 LoRA
   差异。也就是说 v7 实际退化成了 Procedure-only Writer，Core 只进入 query
   并没有获得足够的有效控制权。
3. 这两个数值现象从 macro200 到 macro400 基本不变，而 correct 从 120 降到
   106。因此继续同架构延长训练没有依据。

v8 只补这两个已经实证缺失的最小接口：

```text
joint 8×L competition
→ each Action independently reads L effects
→ 8 bound action–effect tokens are pooled into one event

Core only perturbs Procedure attention query
→ Core also applies a bounded multiplicative gate to Procedure content
```

Semantic Core、Causal Procedure、task-complete训练、公开 LoRA 空间和信息墙均
保持不变。

## 2. 输入与信息墙

Writer 输入仍严格为：

```text
task language + exactly one raw action-hidden teacher video
```

Writer 不得接收 teacher action、state/proprio、reward、terminal、task ID、
filename、hidden normalization 或 policy outcome。source actions 只进入 frozen
policy 的 functional loss。frame stride 固定为 5。

每帧只做一次带 Meta-LoRA 的 π0.5 multimodal/Action-Expert forward，并得到：

```text
G_f ∈ R^(L×256)      task-aligned multimodal semantic evidence
A_f ∈ R^(8×256)      Action Expert suffix anchors at native positions
                     [0,7,14,21,28,35,42,49]
```

8 个 anchors 来自一次 50-position suffix forward 的稀疏读取，不是 8 次
forward，也不改变 execution policy 的 50-action chunk。

## 3. Semantic Core

Core 沿用已经满足需求的最小结构：

\[
C^{(0)}_l =
\frac{1}{F}\sum_{f=1}^{F}G_{f,l}.
\]

随后用两层、宽度 256、8 heads 的 bidirectional task-token Transformer 组合
对象、语义角色、目标关系与语言指定的多子目标信息。RoPE 只使用 task-token
ordinal，不使用 frame position，因此同一 frame set 的任意排列产生相同 Core。

保留 `L` 个 Core tokens，而不是压成固定单向量，使有效容量自然随任务描述中的
对象、关系和子目标数量变化。Core 不单独提供 LoRA value path。

## 4. Hierarchical Action–Effect Event

### 4.1 环境语义变化

先对每帧每个 task token 的语义证据做共享 RMSNorm，再按当前 arm 的实际输入
顺序重新计算：

\[
D_{f,l}
=
\operatorname{RMSNorm}(G_{f+1,l})
-
\operatorname{RMSNorm}(G_{f,l}).
\]

`D_f` 是 raw transition value。用于 attention key 时再经过独立的
`transition_key_norm`：

\[
\widehat D_{f,l}=\operatorname{RMSNorm}_{key}(D_{f,l}).
\]

这一步是必要修正：v7 直接把两个已归一化 frame embeddings 的小差值投影为
key，导致 attention logits 几乎全相等；对差值本身再归一化只改善匹配尺度，
不改变 value 的真实变化幅度。

### 4.2 每个 Action anchor 独立读取 effect

对每个 interval、head 和 Action anchor \(k\)，只在有效 task tokens \(l\)
上归一化：

\[
q_{f,k}=W_q\operatorname{RMSNorm}(A_{f,k}),
\qquad
k_{f,l}=W_k\widehat D_{f,l},
\]

\[
\alpha_{f,h,k,l}
=
\operatorname{softmax}_{l}
\left(
\frac{q_{f,h,k}^{\top}k_{f,h,l}}{\sqrt{32}}
\right).
\]

value 使用未做 post-difference key norm 的 transition，并保留完整 learned
projection：

\[
v_{f,h,l}=W_v^hD_{f,l},
\]

\[
g_{f,h,k}
=
1+\tanh\left(W_g^h\operatorname{RMSNorm}(A_{f,k})\right),
\]

\[
U_{f,h,k}
=
\sum_l \alpha_{f,h,k,l}\,
v_{f,h,l}\odot g_{f,h,k}.
\]

8 heads 拼接并经 bias-free output projection，得到：

\[
U_f\in\mathbb{R}^{8\times256}.
\]

关键区别是 8 个 Action anchors 不再竞争同一个 softmax 概率总量。每个 anchor
都必须先完整回答“与我匹配的环境变化是什么”，避免 v7 的全局平均退化。

### 4.3 Procedure-only EventRead

8 个 bound tokens 仍需压成一个低频高层事件；否则把 `8F` 个 suffix anchors
直接送入时序 Transformer 会重新混入低层 horizon 结构。EventRead 完全位于
Procedure 分支：

\[
q_f^{event}
=
\operatorname{RMSNorm}
\left(
\frac{1}{8}\sum_{k=1}^{8}U_{f,k}
\right),
\]

\[
E_f
=
W_o^{event}
\operatorname{Attention}
\left(
W_q^{event}q_f^{event},
W_k^{event}\operatorname{RMSNorm}(U_f),
W_v^{event}U_f
\right)
\in\mathbb{R}^{256}.
\]

attention 只在 8 个 bound tokens 上归一化。没有 learned pooling token、null
token、Core query 或 Action-only residual。所有 projection 均无 bias，因此：

```text
D_f = 0 → U_f = 0 → E_f = 0
```

Action 不能脱离视觉语义变化单独形成 Procedure 旁路。

## 5. Causal Procedure

每个有效 frame interval 只向 Procedure 输入一个 `E_f`。三层、宽度 256、
8 heads 的 causal pre-norm Transformer 用 interval 起始 sampled-frame ordinal
做一维 RoPE，只作用于 Q/K：

\[
P_1,\ldots,P_{F-1}
=
\operatorname{CausalProcedure}(E_1,\ldots,E_{F-1}).
\]

它只负责把高层 action–effect events 组织成有向过程，不重新读取 Core、绝对
patch、teacher action 或低层 7D forecast。

## 6. Core 与 Procedure 的唯一融合

320 个公开 LoRA routing slots 先读取 Core：

\[
C^{slot}
=
\operatorname{CoreRead}(R,C).
\]

随后以 Core-conditioned query 读取有序 Procedure：

\[
P^{slot}
=
\operatorname{ProcedureRead}
\left(
\operatorname{RMSNorm}(R+C^{slot}),
P
\right).
\]

v8 在这里加入已被 v7 证据要求的 bounded multiplicative interaction：

\[
\gamma
=
\tanh\left(
W_{\gamma}\operatorname{RMSNorm}(C^{slot})
\right),
\]

\[
\widetilde P^{slot}
=
P^{slot}\odot(1+\gamma).
\]

`\widetilde P^{slot}` 再进入既有 content-only post-fusion block 和 factor
heads。这里没有 additive Core residual 或 Core-to-factor path，因此：

```text
Procedure = 0 → P_slot = 0 → modulated P_slot = 0 → public LoRA delta = 0
```

Core 可以逐 slot、逐 channel 决定“这个任务应该怎样解释当前 Procedure”，但
不能在没有教学过程时单独生成一套通用 task policy。这同时满足：

```text
先通过 Core 了解任务
→ 再依据 Procedure 了解怎样完成
→ Core-only 最多保留 frozen source policy，而不成为捷径
```

## 7. 初始化与参数预算

所有主宽度为 256，attention 为 8 heads × 32。Action feature gate 和 Core
modulation 使用 `Normal(0, 0.02)` 非零初始化；它们从第一批有效梯度开始就能
表达条件化。所有 factor-head 最终 projection 继续严格 zero-init，因此 step0
生成的公开 LoRA 与模板逐 tensor 完全相同。

真实模块枚举：

| 模块 | 参数 |
|---|---:|
| VL Meta-LoRA | 921,600 |
| Action Meta-LoRA | 626,688 |
| semantic projection | 524,288 |
| patch grounding | 197,120 |
| Semantic Core | 1,573,888 |
| Action probe projection | 262,144 |
| hierarchical Action–Effect + EventRead | 590,848 |
| Causal Procedure | 2,360,832 |
| Core-gated compiler | 1,469,696 |
| factor heads | 2,179,072 |
| **v8 Writer total** | **10,706,176** |

相对 corrected rank-128 Source-SFT 的 `10,297,344`，v8 多 `408,832`
（约 `3.97%`）。该上限是软预算；新增参数全部对应两个已证实缺失的接口，没有
用于扩大下游 decoder 或增加旁路。

## 8. 训练与判定合同

首轮保持 v7 使用的单变量训练合同，避免把模型修正与 recipe 同时改变：

```text
4 ranks on physical GPU4–7
× each rank sequentially processes 6 different tasks, long-first
× one video/LoRA per task
× B_a independent action queries per task
→ one task-complete 24-task macro update
```

先对最长 105-frame 真实视频测试 B20 连续 3 个完整 macro；仅在 OOM 或重复
不稳定时直接降到 B16，不扫描 B17–B19 或 B21+。profile 后做 macro-boundary
exact-resume smoke，再从 identity fresh 运行首个 200-macro fast-decay400
segment。是否续到 400 由 correct400 曲线和内部信号决定。

v8 的最低成功门仍为：

```text
single-checkpoint correct400 >= 150
same-task-other 与 correct 同档
correct 显著优于 wrong / shuffled / reversed
增益由多个 tasks 共同贡献
独立 RNG / video permutation 复测成立
```

内部必须同时检查：

- per-probe effect attention 的 logit spread、entropy 与跨 anchor 差异；
- EventRead 是否避免 8-anchor uniform collapse；
- Core gate 的幅度和 fixed-Procedure/vary-Core effective-LoRA 差异；
- Procedure 差异是否传到 effective LoRA 和 policy action；
- `D=0→E=0` 与 `Procedure=0→LoRA delta=0` 是否持续成立。

若 v8 仍未达门，只依据最早被证实的瓶颈做下一项单变量实验；不得用
checkpoint融合、ensemble、order/contrast loss、信息墙捷径或 Core-only
residual 把失败掩盖成成功。
