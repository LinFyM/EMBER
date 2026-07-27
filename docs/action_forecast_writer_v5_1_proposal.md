我按 brainstorming 的取舍原则把方案收敛成了一条单一路径：把预算从已经证明不是瓶颈的下游 factor decoder，重新分配到上游语义表征；不保留备选分支，也不引入辅助顺序损失。以下内容可以直接复制给主进程。

---

# EMBER Writer v5.1 架构提案：Language-Axial Semantic Core + Causal Action Procedure + Slot-Normalized Fusion

状态（2026-07-27）：owner已批准为下一唯一focused架构，尚未实现。它应原位
替换当前canonical Writer，使用新schema并从identity fresh训练；旧v5
checkpoint不兼容，不能续训。

> 2026-07-27 owner 执行更新：v5 step1400 observed-best 的正式五条件已经
> 完成，`correct/same/wrong/shuffled/reversed=115/108/74/113/114`。
> wrong-video语义性通过，但shuffle/reverse与correct等价，顺序行为硬门失败，
> 因而“不再等待讨论、直接建立v5.1 Goal”的条件已经触发。本提案由候选提升为
> 唯一下一实现方向。v5.1完成实现、真实显存/吞吐上限profile和必要smoke后，
> 首个formal训练只跑约一小时量级的fresh segment，其optimizer step数由新
> 架构的实测训练效率决定，不继承v5的900/1800 step坐标；早期先用内部五条件
> 与轻量paired rollout确认final effective LoRA / policy action的
> `same < shuffled/reversed`关系确有改善。只有特异性改善且absolute/训练曲线
> 值得继续时，才逐段决定是否exact-resume额外训练；第二段乃至第三段都不得
> 自动启动。

## 1. 设计目标

Writer 合同不变：

```text
task language
+ exactly one action-hidden teacher video
→ shared Writer
→ complete task-specific rank-16 LoRA
→ frozen π0.5-LIBERO source policy
```

新架构要明确分离两类信息：

- Semantic Core：任务在做什么、涉及什么对象和关系、目标状态是什么。它应对视频帧顺序严格不变，对同任务不同正确 demo 尽量稳定。
- Causal Procedure：上述任务如何有向推进。它应对 normal/shuffled/reversed 明显敏感，但不能退化成具体 demo 的低层位移、速度、抓取角度或绝对 phase controller。
- 最终融合：Core 决定内容主体，Procedure 调制完成方式；在最终 LoRA 与 policy function 层面，应实现：
  `same-task difference < shuffled/reversed difference`。

这一区分很重要：

- 在 Core 层，shuffle/reverse 使用同一帧集合，所以差异应接近零；same-task-other 换了真实视频，Core 差异大于 shuffle 是正常的。
- 当前真正的问题发生在融合以后：same-task 的 Core 变化压过了 Procedure 的顺序变化，导致 final content、effective LoRA 和 policy action 都出现 `same > shuffled`。
- 新设计不要求 Core 的 same 差异小于其 shuffle 差异，而要求最终功能层恢复 `same < shuffled/reversed`。

## 2. 非目标与硬边界

新架构不做：

- 不预测 teacher action；
- 不恢复 7D forecast、Plan/Revision/Belief 或 absolute robot clock；
- 不把 Core 命名成固定的“抓取/移动/放置”事件槽；
- 不把可变长任务压成固定数量 Core tokens；
- 不把 text-only hidden 直接送入 Core value、LoRA slots 或 factor heads；
- 不做 `h(language,image)-h(language,null-image)` subtraction；
- 不增加 contrast/order/classification/margin loss；
- 不引入第二套 adapter、shared trainable execution adapter 或并行 Writer；
- 不允许 raw image-position hidden 绕过新的 language bottleneck。

训练仍只用 normal-order positive AS functional loss。

## 3. 完整数据流

```text
纯任务语言 token ids
    │
    ├─ Text-only Gemma + Text Meta-LoRA
    │      → video-independent contextual queries Q_text [L,2048]
    │
teacher frames + multimodal prompt
    │
    ├─ PaliGemma + VL Meta-LoRA
    │      → 每帧 task-language hidden H_vl [T,L,2048]
    │
    │   Q_text 与 H_vl
    │      → token-aligned frame-set attention
    │      → bidirectional language Core Transformer
    │      → Core C [L,256]
    │
    └─ 同一次 multimodal prefix KV
           + fixed native suffix
           + Action Expert + Action Meta-LoRA
           → suffix hidden A [T,50,1024]
           → suffix mean + projection
           → causal Procedure Transformer
           → Procedure P [T,256]

Core C
    → 320 LoRA slots读取Core
    → S_C [320,256]

Core-conditioned slots + Procedure P
    → 读取中心化的Procedure内容
    → S_P [320,256]
    → Procedure生成AdaLN参数调制Core
    → post-fusion slot self-attention
    → Z [320,256]

Z
    → 8个factor heads
    → 38 targets / 76 A-B tensors
    → complete public rank-16 LoRA
```

## 4. 输入语言与精确 token 对齐

### 4.1 Multimodal 分支

每帧仍使用：

```text
image tokens
+ "Task: {cleaned_task};\nAction: "
```

但不再读取 image-position hidden 作为 Core。只截取最后一层、final norm 后，真正属于 `{cleaned_task}` 的 task-description token hidden。

### 4.2 Text-only 分支

Text-only Gemma 的内容只是纯任务描述：

```text
{cleaned_task}
```

不输入 `Task:`、`;`、`Action:` 等模板文本。

但不能单独重新 tokenize 后假设 token 对齐。正确合同是：

1. 对完整 multimodal prompt 做一次权威 tokenization；
2. 精确记录 task-description span 对应的 token IDs；
3. Text-only Gemma 输入为 `BOS + 同一组 task-span token IDs`；
4. 两路均只保留这 L 个 task token 的 hidden states。

这样能够避免 leading-space、SentencePiece/BPE 边界导致的 token 不一致。

### 4.3 为什么 Core 长度是 L

Core 输出长度直接等于任务描述 token 数：

\[
C\in\mathbb{R}^{L\times256}.
\]

因此较长、组合更复杂的任务自然拥有更多 Core tokens；不再把所有任务压入固定数量 summary slots。L 不是完美的“任务复杂度”，但比固定 8/16 个 Core tokens 更符合 LIBERO 的组合语言结构。

Padding 只用于 batch，所有后续模块携带 task-token valid mask。

## 5. 三个直接 Meta-LoRA

三路都使用同一种普通、直接安装的 LoRA 机制：

\[
y=Wx+B(Ax).
\]

不存在 hypernetwork，也不是由任务动态生成 Meta-LoRA。

统一配置：

- 18 层；
- targets：`q_proj/k_proj/v_proj/o_proj`；
- rank 4；
- A 正常初始化；
- B 物理零初始化；
- frozen Gemma/Action Expert 权重不更新。

三套参数彼此独立：

```text
Text Meta-LoRA   → text-only Gemma
VL Meta-LoRA     → per-frame multimodal PaliGemma
Action Meta-LoRA → native Action Expert
```

设计意图：

- Text Meta-LoRA：让纯语言 Gemma 学会产生适合“检索视频证据”的 contextual queries。
- VL Meta-LoRA：适配无 state、第三人称/observer teacher，并把图像证据写入 task-language hidden。
- Action Meta-LoRA：把 observer 视频转换成机器人交互语义，而不是继续拟合具体 action trajectory。
- Action rank 8 降至 4，是因为该分支现在只是 semantic probe，不承担完整 action generation；省下的预算用于新增 Text 分支和真正的 Core 表征。

## 6. Semantic Core 分支

### 6.1 Text-only contextual queries

纯语言序列执行一次独立 Gemma 前向：

\[
Q^{text}
=
Gemma_{\text{Text-LoRA}}
(BOS,x_{1:L})_{task}
\in\mathbb{R}^{L\times2048}.
\]

合同：

- 使用最后一层、final norm 后 hidden；
- task tokens 之间使用完整双向 attention；
- Q 完全不读取视频；
- Q 只用来生成 frame-set attention 的 query；
- Q 不允许作为 value、residual 或 LoRA 内容。

为什么不是原始 token embedding：

- 最后一层 hidden 已包含完整句子上下文；
- “cream cheese”之类多 token 概念不会被拆成互不相关的查询；
- 每个 token query 已理解动词、对象、目标容器和句法角色。

为什么它不等价于“multimodal prefix 再多跑几层”：

- Multimodal hidden 的 query 本身会被具体视频控制；
- 这里的查询问题只由任务语言决定，视频只能提供被查询的证据；
- 这从结构上避免了某条 demo 的几何和轨迹同时决定“问什么”与“回答什么”。

### 6.2 Per-frame multimodal language K/V

对每个 frame，取 PaliGemma 最后一层 task-description span：

\[
H^{vl}_{t}
\in\mathbb{R}^{L\times2048}.
\]

这些 hidden 同时具有：

- 完整任务语言上下文；
- 当前帧图像条件；
- 与 Text-only 分支严格一致的 token identity。

不再把 `[256,2048]` image-position hidden、`8×8`空间网格或 raw visual tokens送入 Core compiler。

原因：

- 当前 Core 使用 64T 个空间 tokens，仍能保留较多 demo 外观、几何和低层轨迹信息；
- language-position bottleneck 强迫视觉证据先被解释为“与任务词语有关的内容”；
- 它不会直接保存整幅图的高频空间细节。

### 6.3 共享语义投影

两路最后一层 hidden 使用同一个 bias-free 投影：

\[
X^{text}_l=W_LQ^{text}_l,
\qquad
X^{vl}_{t,l}=W_LH^{vl}_{t,l},
\]

\[
W_L\in\mathbb{R}^{256\times2048}.
\]

共享投影的意图：

- 两路来自同一 frozen Gemma 表征空间；
- 强制它们进入同一个 256 维语义坐标系；
- 避免为 query 和 evidence 各自学习一套任意投影；
- 节省预算。

### 6.4 Token-aligned frame-set attention

第一阶段只沿 frame 轴聚合，不跨 language-token 位置混合。

对每个 token \(l\)、每个 head \(h\)：

\[
q_{l,h}
=
W^Q_h\,Norm(X^{text}_l),
\]

\[
k_{t,l,h}
=
W^K_h\,Norm(X^{vl}_{t,l}),
\]

\[
v_{t,l,h}
=
W^V_hX^{vl}_{t,l}.
\]

分数：

\[
s_{t,l,h}
=
\frac{q_{l,h}^{\top}k_{t,l,h}}{\sqrt{32}}.
\]

只在有效 frame 维度做 softmax：

\[
p_{t,l,h}=softmax_t(s_{t,l,h}).
\]

为了从稳定 mean 起步，但允许选择少数关键帧，使用 mean-anchored 权重：

\[
\lambda_h=\sigma(\rho_h),
\]

\[
a_{t,l,h}
=
(1-\lambda_h)\frac1T
+
\lambda_h p_{t,l,h}.
\]

配置：

- width 256；
- 8 heads，head width 32；
- 每个 head 一个全局可学习 \(\rho_h\)；
- 初始化 \(\lambda_h=0.05\)，即 95% uniform mean、5% learned selection；
- 不使用 frame ordinal、temporal position、first/last 标记；
- 不使用 token/video-dependent dynamic gate；
- K/V 只来自同一个 language-token 位置跨 frames 的集合。

输出：

\[
U_l
=
W^O
Concat_h
\left(
\sum_t a_{t,l,h}v_{t,l,h}
\right),
\]

\[
U\in\mathbb{R}^{L\times256}.
\]

设计意图：

- mean 部分降低单帧外观、视角和瞬时运动的影响；
- learned 部分保留只在少数帧清楚出现的关键对象、接触关系或终态证据；
- 查询由完整任务语言决定，例如与 “cream cheese” 对应的 contextual query 可以从整条视频选择相关证据；
- frame permutation 只会重排 softmax 输入行，聚合结果结构上不变；
- 不直接 flatten 全部 `(t,l)`，避免不同语言概念在第一阶段互相抢夺证据。

应记录每个 head 的：

- \(\lambda_h\)；
- attention entropy；
- 最大 frame 权重；
- effective frame count：

\[
N_{\mathrm{eff}}
=
\frac1{\sum_t a_t^2}.
\]

这用于确认 Core 没有退化成复制某一个具体 frame。

### 6.5 Bidirectional language Core Transformer

第二阶段沿 language-token 轴组合概念：

```text
U [L,256]
→ 2 × bidirectional pre-norm Transformer blocks
→ C [L,256]
```

每个 block：

- width 256；
- 8 heads；
- full bidirectional self-attention；
- task-token ordinal 通过 1D RoPE 进入 Q/K；
- V 与 residual 只传递 content；
- FFN `256→1024→256`，GELU；
- valid task-token mask；
- 无 frame position。

两个轴的职责因此明确：

```text
[T,L,2048]
   │
   ├─ 沿T做无序、task-conditioned证据聚合
   ↓
[L,256]
   │
   ├─ 沿L做有序概念组合
   ↓
Core [L,256]
```

第一阶段回答“每个任务概念在整条视频里有什么证据”；第二阶段回答“这些对象、动作、关系和目标如何组成一个任务”。

使用两层而不是继续扩大 factor decoder，是因为已有证据表明 decoder 能传递差异，当前最早失效点在上游语义和融合。

## 7. Causal Procedure 分支

Procedure 主体保留当前已经证明有明显顺序敏感性的 Action Expert 路径，只做必要收缩。

### 7.1 Native Action Expert probe

每帧使用同一次 multimodal prefix KV，加：

- 固定 persistent Gaussian suffix `[50,32]`；
- native flow time `t=1`；
- frozen native `action_in_proj` 与 time embedding；
- Action Meta-LoRA rank 4。

只执行一次 Action Expert forward：

- 不做 flow integration；
- 不调用 `action_out_proj`；
- 不输出 7D action；
- 不把 50 suffix positions解释成 future action clock。

取 final norm 后、action output 前的 hidden：

\[
A_t\in\mathbb{R}^{50\times1024}.
\]

### 7.2 Suffix semantic bottleneck

\[
\bar A_t
=
\frac1{50}\sum_{j=1}^{50}A_{t,j},
\]

\[
s_t=W_A\bar A_t,
\qquad
W_A\in\mathbb{R}^{256\times1024}.
\]

输出：

\[
S=[s_1,\ldots,s_T]\in\mathbb{R}^{T\times256}.
\]

继续使用 mean 的原因：

- 当前 Procedure sequence 已经对 shuffle/reverse 有强差异，说明 Action Expert hidden 含有足够的过程信息；
- mean 主动删除 suffix-slot/lead 对应，防止再次形成 50-step future clock；
- 此处没有证据支持换成更复杂的 suffix attention。

### 7.3 两层 causal Procedure Transformer

```text
S [T,256]
→ 2 × global causal pre-norm Transformer
→ P [T,256]
```

每层：

- width 256；
- 8 heads；
- causal attention；
- sampled-frame ordinal `0…T-1`只通过 RoPE 进入 Q/K；
- token t 只能读取 `1…t`；
- V/residual 只传 content；
- FFN `256→1024→256`；
- valid-frame mask；
- 不使用 normalized video progress、绝对机器人时间或 learned position value。

设计意图：

- Procedure 长度随真实视频长度 T 变化；
- 它描述交互状态如何有向推进；
- causal attention提供“先发生的内容怎样解释后发生内容”的归纳偏置；
- 输入不是 action forecast，因此不再天然携带前三维 translation controller。

## 8. 320 个 LoRA routing identities

公开 LoRA routing slots 保持：

```text
18 Action Expert layers × rank16 = 288
action_in rows                    = 16
action_out rows                   = 16
total                             = 320
```

每个 slot 的 routing identity 由：

- per-query identity；
- module identity；
- layer identity；
- rank identity；

组合成：

\[
R\in\mathbb{R}^{320\times256}.
\]

硬约束：

- R 只进入 attention Q/K；
- R 不进入 V；
- R 不作为 content residual；
- factor heads 永远不能直接读取 R。

它只能决定“哪个 LoRA row 应读取什么”，不能独立生成 task adapter。

## 9. Core-conditioned Procedure fusion

这是相对当前 v5 的第二个核心修改。

当前：

\[
Z=Z_C+D_P
\]

属于原始幅度上的 additive residual。step900 已证明 Procedure 内部有顺序差异，但 Core 的跨 demo 数值变化在相加后占主导，最终形成 `same > shuffled`。

新融合改为三个阶段。

### 9.1 LoRA slots读取Core

从零动态 content 开始：

\[
S_C
=
CrossAttn
\left(
Q=R,\;
K=Norm(C),\;
V=C
\right),
\]

\[
S_C\in\mathbb{R}^{320\times256}.
\]

这里不再额外做一个完整 Core slot self-attention block，因为 Core 的跨 token 组合已经由两层 language Core Transformer 完成。slot 间协调统一留到融合后处理，减少重复 decoder 容量。

### 9.2 Core-conditioned Procedure readout

先对 Procedure value 做无参数时间中心化：

\[
\tilde P_t
=
P_t
-
\frac1T\sum_{u=1}^{T}P_u.
\]

然后每个 slot 使用自己的 Core 内容读取 Procedure：

\[
S_P
=
CrossAttn
\left(
Q=R+Norm(S_C),\;
K=RoPE(Norm(P)),\;
V=\tilde P
\right).
\]

含义：

- Query 不是一组与任务无关的固定 slots，而是“这个 LoRA row 当前从 Core 理解到了什么”；
- 同一个 Procedure 阶段对不同对象、不同任务和不同 LoRA row 可以有不同意义；
- K 保留时间位置，用于有序检索；
- V 使用中心化内容，使 Procedure 的静态 task/demo offset不能充当第二条 Core；
- Procedure 必须通过视频内相对变化和有向组织影响 LoRA。

Procedure cross-attention 使用正常初始化，不能与下一层同时 zero-init，否则两层相乘会形成永久零梯度路径。

### 9.3 Procedure 通过 AdaLN 调制Core

\[
[\gamma,\beta]
=
W_{\mathrm{mod}}Norm(S_P),
\]

其中：

\[
W_{\mathrm{mod}}
:
256\rightarrow512
\]

为 bias-free、zero-init projection。

融合：

\[
F
=
(1+\gamma)\odot Norm(S_C)
+
\beta.
\]

设计意图：

- 先归一化 Core，消除当前 raw Core amplitude 对 Procedure 的压制；
- Procedure 不再与 Core 直接比拼向量范数，而是改变 Core 的通道选择与偏置；
- fresh 时 `W_mod=0`，严格有 `F=Norm(S_C)`；
- 训练后 Procedure 可以增强、抑制或重定向 Core 内容；
- 不人为设置 transition gain、margin 或顺序 loss。

只将 `W_mod` zero-init；Procedure cross-attention必须非零初始化，以保证 `W_mod` 打开后上游可以获得梯度。

### 9.4 一个统一的 post-fusion slot Transformer

用户所问的“Core 和 Procedure 是否还需要统一注意力”，建议保留一个，但位置必须在上述有向融合之后：

\[
Y
=
F+
SelfAttn
\left(
Q=R+Norm(F),
K=R+Norm(F),
V=F
\right),
\]

\[
Z
=
OutputNorm
\left(
Y+FFN(Norm(Y))
\right).
\]

配置：

- 1 block；
- width 256；
- 8 heads；
- FFN `256→1024→256`；
- routing只进Q/K。

它负责：

- 协调不同 layer/rank/module slots；
- 让已经完成 Core–Procedure 绑定的内容在整套 LoRA 内保持一致；
- 协调 q/v、action_in/action_out 等更新。

它不再读取 `[Core;Procedure]` 拼接 memory，因此不会重新模糊两路职责，也不能让 Procedure 绕过 Core。

## 10. Factor heads 与 public LoRA

\[
Z\in\mathbb{R}^{320\times256}
\]

不是 LoRA 本身。

它按现有 routing layout拆分为：

```text
expert:    [18,16,256]
action_in: [16,256]
action_out:[16,256]
```

随后进入现有八类共享 factor heads：

```text
256 → 240 → target_width
GELU
final Linear zero-init
all Linear bias-free
```

输出宽度保持：

| head | width |
|---|---:|
| q_A | 1024 |
| q_B | 2048 |
| v_A | 1024 |
| v_B | 256 |
| action_in_A | 32 |
| action_in_B | 1024 |
| action_out_A | 1024 |
| action_out_B | 32 |

最终仍生成：

- 38 targets；
- 76 A/B tensors；
- public rank 16；
- alpha 16；
- dropout 0；
- `1,287,168` public LoRA scalars。

真实 identity template 继续提供确定性非零 A basis 与物理零 B。factor-head final weights为零，因此 fresh Writer 输出 functionally identity LoRA。

Factor hidden width从420降到240的原因：

- v4/v5 已证明下游 decoder 能把上游差异传到 LoRA；
- 当前问题不是 factor head 宽度不足，而是 Core 语义和融合比例错误；
- 将参数从无证据的 decoder 宽度转移到 text query、frame-set pooling和Core Transformer更符合根因。

## 11. 参数预算

设计中所有 Linear/MHA 均 bias-free，以下计数包含 RMSNorm 和 frame-attention gate。

| component | params |
|---|---:|
| Text Meta-LoRA rank4 | 921,600 |
| VL Meta-LoRA rank4 | 921,600 |
| Action Meta-LoRA rank4 | 626,688 |
| shared language projection `2048→256` | 524,288 |
| token-aligned frame-set attention | 262,664 |
| 2 bidirectional Core blocks | 1,573,888 |
| Action interaction projection `1024→256` | 262,144 |
| 2 causal Procedure blocks | 1,573,888 |
| routing + Core read + Procedure read + AdaLN + post-fusion block | 1,535,232 |
| 8 factor heads，hidden 240 | 2,042,880 |
| **total** | **10,244,872** |

比较：

```text
rank-128 Source-SFT = 10,297,344
new Writer          = 10,244,872
difference          = -52,472
relative difference = -0.510%
```

相对当前 v5 的 `10,301,440`，新设计还少 `56,568` 参数。它不是通过扩大 Writer 获得优势，也不需要用无意义参数凑到精确相等。

实现后仍必须从真实模型逐模块打印计数；若与上述不一致，应先检查 bias、重复 norm 或重复 adapter，而不是增加科学模块补数。

## 12. 初始化与梯度打开顺序

初始化合同：

- 三个 Meta-LoRA：A正常初始化，B=0；
- frame-set attention：正常初始化，`\lambda=0.05`；
- 两层 Core Transformer：正常初始化；
- 两层 Procedure Transformer：正常初始化；
- Procedure cross-attention：正常初始化；
- AdaLN `W_mod=0`；
- post-fusion slot block：正常初始化；
- factor-head final Linear=0；
- public LoRA identity template：A非零、B=0。

预期梯度阶段：

1. 第一次 backward 主要打开 factor-head final layers；
2. factor输出路径打开后，Core、post-fusion和AdaLN projection开始获得梯度；
3. `W_mod` 非零后，Procedure reader、Procedure Transformer与Action Meta-LoRA开始获得完整梯度；
4. 不能错误要求第一个 optimizer step 所有上游模块均有非零梯度。

## 13. 结构性保证与禁止旁路

必须在实现中明确保证：

- Text hidden只进入 frame-set query；
- Core value只来自 image-conditioned multimodal task-language hidden；
- raw image positions不能进入Core、slots或factor heads；
- Action Expert suffix hidden只进入Procedure；
- Procedure不能直接进入factor heads；
- Core和Procedure只能经规定的slot fusion汇合；
- routing identity不能成为value/content；
- 同一帧集合的 permutation 不改变Core；
- task-token顺序可以改变Core；
- frame顺序可以改变Procedure；
- 最终只输出一套 public LoRA。

本设计不使用 text-only hidden 与 multimodal hidden 的直接 subtraction。两路拥有不同上下文、position IDs和独立 Meta-LoRA，直接相减并不天然等于“visual grounding”。

## 14. 对当前两个核心疑问的回答

### 14.1 Core和Procedure是否真是高层语义

架构本身不能“证明”语义，但新设计提供了比当前 v5 强得多的归纳约束：

- Core 被限制在 task-language token 轴；
- query 由完整纯语言句子产生，且与视频无关；
- value必须来自视频条件下的相同 task token；
- frame维度先做稳定、无序的多帧证据汇聚；
- token维度再组合对象、动作和目标关系；
- raw spatial tokens、frame position和Action Expert hidden都不能进入Core；
- Procedure输入经过50 suffix mean，不是7D forecast；
- Procedure的静态时间均值不能直接进入融合value；
- Core–Procedure融合发生在归一化slot空间，不由原始向量幅度决定。

这仍需要训练后证据确认，不能仅凭模块命名把它称为高层语义。

### 14.2 为什么当前same比shuffled差异大，新设计如何修复

当前现象并不违反数学直觉：

- shuffled使用同一批frames，所以Core结构上几乎完全相同；
- same-task-other换了另一条真实视频，外观、几何、物体位置和动作风格都会变化，所以Core变化大于shuffle正常；
- 问题是当前 `Z=Z_C+D` 让这部分Core变化继续主导最终LoRA，Procedure的顺序变化被压小。

新设计同时在两端修复：

1. 上游通过 language-token bottleneck、跨帧mean-anchor和video-independent query，主动降低same-task的高频demo差异；
2. 下游先归一化Core，再让Procedure用AdaLN调制它，避免Core raw amplitude压过顺序信号；
3. Procedure value去除时间均值，减少它复制第二份静态task/demo content；
4. post-fusion attention只协调已经融合的slots，不允许任何一路重新旁路。

期望层级关系是：

```text
Core:
normal ≈ shuffled ≈ reversed
same-task difference < wrong-video difference

Final Z / effective LoRA / policy function:
same-task difference
    < shuffled difference
    <≈ reversed difference

behavior:
correct ≈ same-task-other
correct > wrong
correct > shuffled
correct > reversed
```

这里不要求 Core 层 `same < shuffled`，因为 Core 的 shuffle 差异按设计应接近零。

## 15. 必须完成的内部验证

### 15.1 训练前结构检查

- Text/VL task token IDs逐位置完全一致；
- Text输入不含固定模板token；
- Text query shape `[B,L,2048]`；
- multimodal K/V shape `[T,L,2048]`；
- Core shape `[B,L,256]`，L随任务语言变化；
- Procedure shape `[B,T,256]`，T随视频变化；
- Core对normal/shuffle/reverse在浮点reduction容差内相同；
- Procedure满足prefix causality；
- Procedure normal/shuffle/reverse非同值；
- Text query无法直接进入Core value；
- fresh LoRA functionally identity；
- fixed suffix在resume后逐bit一致；
- frozen source policy trainable count为0；
- 真实参数量符合预算；
- checkpoint完整保存第三套Text Meta-LoRA和新模块状态。

### 15.2 训练后轻量数值检查

固定 correct/same/wrong/shuffled/reversed references，逐层记录：

```text
text-only contextual query
multimodal task-language hidden
frame-set attention weights
Core before/after language Transformer
Action Expert suffix hidden
interaction tokens
Procedure sequence
Core slot content S_C
Procedure slot content S_P
AdaLN gamma/beta
post-fusion Z
raw LoRA factors
effective B@A
fixed-query policy action
```

关键判据：

- Core shuffle/reverse保持数值不变；
- Core same-task变化显著小于wrong；
- Core换wrong/null-like视觉时不能完全不变，排除纯语言捷径；
- Procedure normal/shuffle/reverse差异跨tasks/videos稳定；
- final Z、effective LoRA和policy action应满足：
  `same < shuffled/reversed`；
- fixed-Core只替换Procedure时，顺序差异必须完整穿过AdaLN和policy；
- 禁用Procedure时性能不应反而系统提高；
- Core不应重新出现强translation/normalized-progress probe；
- frame-set attention不能系统坍缩成单帧复制。

可继续使用train teacher action/proprio做post-inference低层probe，但不得进入Writer或optimizer。

### 15.3 行为门

训练合同继续使用单视频完整action-batch：

```text
每rank每step：
1 task
1 teacher video
1 generated LoRA
B_a independent same-task action queries
```

不增加辅助loss。新架构必须fresh训练，并重新profile显存；原F32/B20只能作为首个候选，不能未经实测直接继承。

完整验证仍要求：

- absolute correct-video达到既有预门后才跑full specificity；
- same-task-other约等于correct；
- correct显著优于wrong；
- correct显著优于shuffled/reversed；
- 多个tasks共同贡献；
- observed-best后有明确、可复测的持续下降，才能判断训练充分；
- 全部AS门通过后才允许进入独立cold-start RL。

## 16. 实现边界建议

建议保持单一 canonical owner：

- `video_program`：新增Text-only query forward；multimodal输出改为task-language span；保留Action Expert probe；删除空间Core输出。
- `temporal`：新增token-aligned frame-set attention、bidirectional Core encoder和新的slot fusion；保留causal Procedure基础模块。
- `model`：组合双路、变长L/T mask、factor heads hidden 240。
- config/schema/checkpoint：升级为不兼容的新版本。
- 删除旧raw image-set Core compiler和additive Procedure refiner，不保留runtime兼容分支。

旧v5代码与结果通过Git和文档保留，不在活动源码中维护双架构。

## 17. 最终建议

推荐采用这条设计作为下一版唯一候选，核心理由是：

1. 它直接针对目前最早暴露的问题——Core仍保留过多demo-specific内容；
2. 它使用纯语言Gemma产生video-independent、句子级query，解决原始embedding缺乏多token语义的问题；
3. 它让Core长度随任务语言长度变化；
4. 它把frame-set evidence collection与language composition沿两个轴明确分解；
5. 它保留当前已经证明有效的Action Expert Procedure主干；
6. 它通过归一化AdaLN和统一slot attention修复当前same变化压过order变化的融合问题；
7. 三个Meta-LoRA统一rank4，完整Writer仍严格受Source-SFT参数预算约束；
8. 它不依赖人为顺序损失，最终特异性仍必须由正常AS functional objective自然学出。

这是一套更强的“语义归纳偏置”，不是语义成功的先验声明。是否真正学到高层Core和高层Procedure，仍应由逐层反事实、低层probe、effective LoRA、policy function和最终paired rollout共同判定。

---

side chat原文结束时说明“本次仅完成只读设计，没有修改代码、配置、Git或运行
状态”；该句只描述提案产生时刻。当前执行状态以上方“2026-07-27 owner执行
更新”为准。提案正文仍完整保留，便于核对来源与设计取舍。
