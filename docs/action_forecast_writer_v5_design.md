# EMBER Writer v5：Semantic Core + Causal Procedure

状态：2026-07-26，owner 已批准；这是当前唯一活动 Writer 架构 authority。

本文记录可以直接实现、profile、训练、评测和 exact-resume 的完整合同。后续
session 不需要恢复历史聊天来重建设计。若旧 v4 文档、旧 config、历史 ledger
中的未来式表述与本文冲突：

- 历史数值和实验事实保留；
- 当前代码、配置、训练、评测和停止条件以本文为准；
- v4 完整实现与结果见
  [`action_forecast_writer_design.md`](action_forecast_writer_design.md)；
- v4 的完整根因链见
  [`action_forecast_writer_v4_root_cause.md`](action_forecast_writer_v4_root_cause.md)。

## 0. 当前 Goal 与停止边界

当前 focused Goal 是完成并验证本 v5：

1. 原位替换 v4，不保留平行活动 Writer、runner 或 checkpoint schema；
2. 在物理 GPU 4、5、6、7 上重新 profile 最快安全配置；
3. 按约一小时一个 exact-resume segment 充分训练 AS Writer，并密集保存；
4. 在 validation 找到 observed-best，并验证：
   - same-task other teacher 的影响较小；
   - cross-suite wrong 明显更差；
   - correct 明显优于 shuffled 和 reversed；
5. absolute performance 至少达到或接近 `125/400`，目标逼近 v4 shuffled 的
   `148/400`；
6. 若任一 gate 失败，先定位最早失败层级，修改同一架构后 fresh 重试；
7. AS 全部通过后，才独立推进 cold-start RL Writer。

AS 或 RL 的“充分探索”都不能由几个略低 checkpoint 宣布完成。必须在 best
之后看到幅度明显、远超固定 400-rollout 正常波动、由多个 tasks 共同贡献，
且在独立 panel 或复测中仍成立的持续下降。否则继续相同合同的下一段训练。

完成 focused AS/RL Writer 后停止向 owner 汇报；不自动继续 final-32、
task-local RL、joint oracle 或 ViVLA。

## 1. 第一性原理目标

EMBER 的核心合同是：

```text
task language + exactly one action-hidden teacher video
  -> 对任务的高层理解
  -> complete task-specific rank-16 LoRA
  -> frozen π0.5-LIBERO source policy
```

Writer 应从一条具体示范中提取可迁移的共同逻辑：

```text
涉及什么对象和关系
任务希望达到什么状态
交互如何按先后阶段推进
这些信息怎样修正机器人闭环策略
```

它不应把一条 teacher 的绝对三维轨迹、速度、抓取角度、frame phase 或未经
校准的 future-action clock 写成部署到任意新初态的静态 controller bias。

v4 shuffled `148/400` 的启发不是“顺序不重要”。它证明：

- v4 内存在很强的、打乱帧后仍保留的任务语义；
- correct-order 路径额外写入了不适合跨初态迁移的低层
  phase/translation controller；
- 破坏这条低层路径会让已有高层语义重新主导；
- 新架构应先稳固保存高层 semantic core，再让正确的高层过程信息提供增益。

因此 v5 不是“给 v4 换一个 Temporal”，而是同时改变表示语义、顺序 owner 和
训练统计：

```text
强、顺序不变的 Semantic Core
          +
弱起步、可学习、有向的 Causal Procedure refinement
          +
同一 action 对多个同任务 teacher 的共同 functional 梯度
```

## 2. 不可改变的信息墙

Writer 每次只可读取：

- 正确 task language；
- 恰好一条 teacher video；
- 当前 sealed 数据中仅 `obs/agentview_rgb`；
- 视频自身的帧顺序、有效帧 mask 和采样 ordinal。

Writer 不得读取：

- teacher action；
- proprio/state；
- reward、terminal 或 success；
- task ID、suite ID、filename、demo label；
- source/target normalization 数值；
- policy rollout outcome；
- object pose 或其他 privileged simulator state。

AS functional policy forward 可以读取 action episode 的正常 observation、
language、state 和 action target；这些信息只属于 frozen policy loss，不得
回流为 Writer 输入。

固定数据合同：

- 24 development train tasks；
- teacher video 与 action episode 在同 task 内独立采样；
- validation/test actions 不进入 optimizer；
- frozen raw π0.5-LIBERO source base与 source-only normalization不变；
- public LoRA 挂载空间不变；
- 不使用 `pi05_libero`、MemLLM、bank、geometry、shared update subspace、
  residual escape、额外 shared adapter 或未 merge source LoRA。

训练目标只用 normal-order positive functional action loss。不得加入
shuffle/reverse classifier、contrast、margin 或其他人为制造顺序差异的 loss。

## 3. 输入视频与 teacher prompt

视频合同：

- 相机：`obs/agentview_rgb`；
- LIBERO OpenGL 图像按既有合同旋转 180°；
- render 256，模型输入 resize/pad 至 224；
- `frame_stride=5` 固定，不测试 stride 10；
- 包含 final frame；
- 视频长度 `T` 可变；
- batch 内只做 padding，所有模块都携带有效 mask；
- 不强行压成固定数量的 event tokens。

teacher 侧删除 State 整段。每帧的 PaliGemma prefix 为：

```text
image tokens
+ Task: <cleaned task language>;
  Action:
```

精确文本字符串封存为：

```json
"Task: {cleaned_task};\nAction: "
```

其中换行位于分号与 `Action:` 之间，tokenizer 使用 `add_bos=True`，尾部保留
一个空格。实现时必须用真实 sealed tokenizer 验证 prefix token layout 和最大
长度，并将该精确字符串写入 config。

teacher 侧没有 state placeholder、virtual-state token 或 imagined state。
execution policy 仍使用其原生 observation、language 和真实 state；删除 state
只发生在 Writer 的 teacher-understanding path。

## 4. 模块 A：Visual-Language Frame Encoder

### 4.1 输入

第 `t` 帧图像 `F_t` 与正确语言 `L`：

```text
F_t: uint8 [3,H,W]
L: tokenizer ids + valid mask
```

### 4.2 PaliGemma 前向

每帧正常执行：

```text
image embeddings [256,2048]
+ language/prompt embeddings [L_text,2048]
  -> frozen PaliGemma
     with trainable VL Meta-LoRA
```

prefix 内使用 PaliGemma 原生 full/bidirectional attention；不对语言或图像
位置引入 causal mask。VL Meta-LoRA：

- 18 层；
- `q_proj/k_proj/v_proj/o_proj`；
- rank 4；
- `A` 正常初始化，`B=0`，fresh 为 functional identity；
- 对完整 prefix 正常生效，不做位置 gated LoRA。

VL Meta-LoRA 的职责是适配非标准 teacher 输入，包括无 state、第三人称/
observer-view，以及未来人类视频；它不是执行时 shared adapter。

### 4.3 输出

PaliGemma language model 会返回每个 prefix position 的 final hidden。取最前面
原本由 256 个 image tokens 占据的位置：

\[
H^{img}_t
\in \mathbb{R}^{256\times2048}.
\]

这不是额外的“Gemma image output”。它是完整图文 prefix 前向后，在 image
positions 上的 final hidden，已经被 task language 条件化。

语言很重要，但初版不把 language-position hidden 直接送入下游 value：

- 语言已通过 PaliGemma attention 改写 `H^{img}_t`；
- 语言也通过 prefix KV 供 Action Expert 读取；
- 若直接把语言 positions 作为 Core value，模型可重新走
  `language -> task LoRA` 的捷径，并在每帧重复同一语言。

必须验证：

- 同视频换 task language 会改变 Core 和最终 LoRA；
- 同 language 换 wrong video 也会改变 Core 和最终 LoRA。

若实验证明 language sensitivity 不足，唯一首选补救是增加一个 pooled
language summary 作为 Core/Procedure 的 Q/K routing；它仍不得成为独立 value
或直接进入 factor heads。初版不启用该补救。

## 5. 模块 B：Permutation-Invariant Semantic Core

Semantic Core 保存 v4 shuffle 已证明有价值的高层任务内容：

- 场景和相关对象；
- language-grounded 目标对象与关系；
- 视频中出现过的交互状态；
- 初态、终态和整体操作类型；
- 不依赖具体播放顺序的任务语义。

### 5.1 空间压缩

PaliGemma 的 256 image positions 按原生 `16×16` 网格恢复。使用固定、无参数的
`2×2` average pooling：

\[
16\times16 \rightarrow 8\times8.
\]

每帧得到：

\[
\bar H^{img}_t\in\mathbb{R}^{64\times2048}.
\]

再使用共享 bias-free projection：

\[
C_t=W_C\bar H^{img}_t,\qquad
W_C\in\mathbb{R}^{256\times2048},
\]

\[
C_t\in\mathbb{R}^{64\times256}.
\]

这里不再加一层输入 RMSNorm：PaliGemma final hidden 已经过其 final norm，
compiler 自己会对 key 做 pre-norm。避免重复归一化和无依据的幅度改写。

### 5.2 跨帧集合

将所有有效帧的 `C_t` 仅按张量方式展平：

\[
C(V)\in\mathbb{R}^{(64T)\times256}.
\]

Core 路径中禁止加入：

- frame ordinal；
- temporal RoPE；
- first/last 标签；
- causal mask；
- adjacency 或 transition；
- 可随 frame order 改变的 positional value。

因此对同一组帧的任意 permutation，`C(V)` 只发生 K/V 行置换。标准
cross-attention 的加权和对该置换不变：

\[
Z_C(V)=Z_C(\operatorname{shuffle}(V))
\]

仅允许浮点 reduction order 产生数值末位差。该性质必须成为自动检查。

Core 不是把视频降格成无序任务分类器；它是稳定底座。高层顺序知识由下一条
独立 Procedure 路径提供。

## 6. 模块 C：Native Action-Expert Semantic Probe

保留 Action Expert 和 Action Meta-LoRA，但彻底删除 v4 的低层 action
forecast 语义。

### 6.1 原生固定 suffix

Action Expert 需要 suffix positions 才能从 prefix KV 读取内容。初版保留
π0.5 原生 50-token action suffix 接口：

- 一个 canonical Gaussian noise buffer
  \(\epsilon\in\mathbb{R}^{50\times32}\)；
- 由 Writer initialization seed 一次生成；
- 作为 persistent checkpoint buffer；
- 所有 frames、videos、tasks、train/eval 共用同一个 buffer；
- 固定 native flow time `t=1`；
- 使用 frozen native `action_in_proj` 和 time embedding。

它不是每样本随机 noise，也不表示要预测 50 步轨迹；它只是固定的 pretrained
Action Expert readout basis。保留 50 是为了维持原生 pretrained interface，
不是因为 v5 需要 50 个未来动作。

首版不缩到 10 或 16：

- suffix length 不改变 Action Expert 的 trainable parameter count；
- 缩短会改变 pretrained token-layout/interface；
- 只有真实 profile 证明 50-token activation 是主要吞吐瓶颈时，才单独做
  suffix-length efficiency ablation。

### 6.2 一次 Action Expert 前向

对每帧：

```text
PaliGemma prefix KV
+ fixed native 50-token noise suffix at t=1
  -> frozen Action Expert
     with trainable Action Meta-LoRA
```

Action Meta-LoRA：

- 18 层；
- `q_proj/k_proj/v_proj/o_proj`；
- rank 8；
- identity initialization；
- 只属于 teacher-understanding path。

它的语义职责是：

> 让机器人理解 observer/human teacher，并想象“假如我是视频中的 teacher，
> 当前这一帧涉及怎样的机器人交互”。

只执行一次 expert forward：

- 不做 10-step flow integration；
- 不调用 `action_out_proj`；
- 不输出 7D action；
- 不声称 suffix slot 是 future clock；
- 不构造 frame×lead action chunk。

取 Action Expert final norm 后、`action_out_proj` 前的 suffix hidden：

\[
A_t\in\mathbb{R}^{50\times1024}.
\]

### 6.3 每帧 interaction token

先对 50 个 suffix positions 做普通均值：

\[
\bar A_t=\frac{1}{50}\sum_{j=1}^{50}A_{t,j}
\in\mathbb{R}^{1024}.
\]

再使用共享 bias-free projection：

\[
s_t=W_A\bar A_t,\qquad
W_A\in\mathbb{R}^{256\times1024},
\]

\[
s_t\in\mathbb{R}^{256}.
\]

不额外做输入/输出 RMSNorm：

- expert final hidden 已经过 final norm；
- Procedure Transformer 是 pre-norm；
- 均值保留整个 frame 的 robot-semantic interaction 内容；
- 均值主动删除“第几个 action slot/lead”作为低层 future clock 的直接对应。

这一步的输出不是 state、action 或 trajectory，而是当前帧的机器人交互语义。

## 7. 明确退役的 v4 路径

v5 完整删除：

- 32-token visual-state；
- initial/anchor/local state reader；
- digit-subspace renderer；
- 10-step denoising；
- `[50,7]` future-action forecasts；
- shared robot absolute-time layout；
- Plan；
- Revision direction/strength；
- Belief；
- lead/count/strength routing；
- teacher flow-noise cursor。

这些内容只留在 v4 文档、Git 和旧 checkpoint 中，不得被兼容分支悄悄恢复。

## 8. 模块 D：Variable-Length Causal Procedure Encoder

输入：

\[
S=[s_1,\ldots,s_T]\in\mathbb{R}^{T\times256}.
\]

Procedure 表示高层“任务怎样一步步推进”，而不是 teacher 的具体连续轨迹。

### 8.1 两层全局 causal Transformer

初版使用 2 个独立参数的 pre-norm blocks：

```text
RMSNorm
-> global causal self-attention
-> residual
-> RMSNorm
-> FFN(256 -> 1024 -> 256, GELU)
-> residual
```

attention：

- width 256；
- 8 heads；
- Q/K 使用 normalized content；
- signed frame ordinal 通过 1D RoPE 进入 Q/K；
- V 和 residual 只传递真实 content；
- valid-frame padding mask 与 causal mask 同时生效；
- token `t` 只能读取 `1..t`；
- 不加入 absolute robot time；
- 不加入 learned position value；
- 不做 set pooling。

输出保留全部有效位置：

\[
E=[E_1,\ldots,E_T]\in\mathbb{R}^{T\times256}.
\]

内部长度随视频长度增长。短任务和长任务不会被强行压进同样 8 个 event tokens。
当前 LIBERO 视频直接做全局 causal attention；只有未来长视频真实出现
`O(T^2)` 成本瓶颈时，才考虑 causal window/hierarchy，不能提前改变语义。

### 8.2 初始化

Procedure blocks 使用正常非零初始化，不采用 v4 的 identity-safe zero block：

- fresh 时 normal/reverse/shuffle 已应产生不同的内部 `E`；
- public task LoRA 的 identity 由 factor heads 保证；
- Procedure 对最终 LoRA 的初始影响由下游 zero-init refiner 保证；
- 没有必要把 Procedure 本身初始化为近 identity 或零。

causal attention 只提供正确的信息方向，不自动保证高层抽象。真正避免重走
Action-Memory/v4 弯路的是三者组合：

- 单帧输入不再是 7D translation forecast；
- Semantic Core 单独保护稳定高层内容；
- 同 action 的多 demo functional 梯度压低 demo-specific路径。

## 9. 模块 E：320 个 LoRA routing identities

公开 LoRA schema固定：

```text
18 Action Expert layers × rank 16 = 288
action_in rank rows                 = 16
action_out rank rows                = 16
total routing/query slots           = 320
```

每个 slot 的静态 routing identity `R` 由：

- query identity；
- module identity；
- layer identity；
- rank identity；

组合得到，宽度 256。

强制 routing/content 分离：

- `R` 只进入 attention Q/K；
- `R` 不进入 V；
- `R` 不作为 residual content；
- factor heads 永远不能读取 `R`；
- 动态 content state 从严格的零张量开始。

因此静态 module/layer/rank 编号只能决定“读什么”，不能自己生成公共 task
LoRA。

## 10. 模块 F：Core-to-LoRA Compiler

Core compiler 先从无序 `C(V)` 生成稳定 adapter content。

初始化：

\[
Z_0=0
\in\mathbb{R}^{320\times256}.
\]

### 10.1 Core cross-attention

\[
A_C=
\operatorname{Attn}
\left(
Q=R+\operatorname{Norm}(Z_0),
K=\operatorname{Norm}(C),
V=C
\right).
\]

\[
Z_C=A_C.
\]

K 做 RMSNorm 只用于 attention addressing，V 传递 raw Core content。

### 10.2 content-only query mixing

\[
A_{self}=
\operatorname{Attn}
\left(
Q=R+\operatorname{Norm}(Z_C),
K=R+\operatorname{Norm}(Z_C),
V=Z_C
\right),
\]

\[
Z_C\leftarrow Z_C+A_{self},
\]

\[
Z_C\leftarrow
Z_C+\operatorname{FFN}(\operatorname{Norm}(Z_C)).
\]

初版只用一个 Core compiler block。它正常初始化；fresh 时可读取 Core，但最终
factor heads 仍输出 identity LoRA。

## 11. 模块 G：Procedure-to-LoRA Refiner

Procedure 不是与 Core 并列 concat 后任由 factor heads 忽略。它被定义为对
Core adapter content 的有向修正：

\[
D_0=
\operatorname{Attn}
\left(
Q=R+\operatorname{Norm}(Z_C),
K=\operatorname{Norm}(E)+\operatorname{RoPEOrdinal},
V=E
\right).
\]

随后只在 `D` 内进行 content-only query mixing：

\[
A_D=
\operatorname{Attn}
\left(
Q=R+\operatorname{Norm}(D),
K=R+\operatorname{Norm}(D),
V=D
\right),
\]

\[
D\leftarrow D+A_D+
\operatorname{FFN}(\operatorname{Norm}(D+A_D)).
\]

最终：

\[
Z=Z_C+D.
\]

约束：

- Procedure cross-attention 的 output projection zero-init；
- fresh 时严格 `D=0`，所以模型从可靠的 Core-only 起步；
- `D` 的 value/residual/FFN content 路径无 additive bias；
- Procedure 可以学习增加、删除或重定向 Core adapter content；
- factor heads只看到融合后的单一 `Z`，仍只生成一套 LoRA；
- 不存在第二套 adapter、execution branch 或人工 transition gain。

直观解释：

```text
Core:
  “目标涉及哪件物体、要形成什么关系、这是哪类操作”

Procedure:
  “这些交互状态按怎样的先后逻辑逐步推进”

Z:
  “针对上述对象，执行具有该阶段结构的闭环策略修正”
```

Procedure zero-init refinement 不表示顺序可有可无；它表示在训练开始时，不让
尚未学会的时序分量破坏已经可用的高层底座。正确 Procedure 只有在跨 demo、
跨 action episodes 持续有用时才进入 LoRA。

## 12. Compiler 深度与扩容规则

当前固定：

- 1 个 Core-to-LoRA Compiler block；
- 1 个 Procedure-to-LoRA Refiner block。

若未来参数预算增加，可以直接加层，但顺序必须是：

```text
all Core compiler blocks
-> all Procedure refiner blocks
-> factor heads
```

不交错 Core/Procedure blocks，不让低层 Procedure 在 Core 尚未稳定时反复改写
底座。每增加一个同构 block 约增加 `1,049,600` 个参数。

只有逐层诊断证明：

- `C` 和 `E` 已有正确语义；
- 但 compiler/query content 或 LoRA function 差异明显不足；

才允许加深。v4 已证明 decoder 能忠实传递差异，因此初版不靠加深解决表示
问题。

## 13. 模块 H：Factor heads 与公开 LoRA

`Z` 分为：

```text
expert:    [18,16,256]
action_in: [16,256]
action_out:[16,256]
```

使用 8 个共享 bias-free MLP heads：

```text
256 -> 420 -> target_width
GELU
final Linear zero-init
```

输出宽度：

| head | width |
|---|---:|
| `q_A` | 1024 |
| `q_B` | 2048 |
| `v_A` | 1024 |
| `v_B` | 256 |
| `action_in_A` | 32 |
| `action_in_B` | 1024 |
| `action_out_A` | 1024 |
| `action_out_B` | 32 |

tensor names、shape 和 transpose 从真实 identity template 读取，不手写推断。
公开输出保持：

- 38 targets；
- 76 A/B tensors；
- rank 16；
- alpha 16；
- dropout 0；
- `1,287,168` LoRA scalars。

identity template 的 A 是确定性非零 basis，B 为物理零。所有 factor-head final
weights 为零，因此 fresh Writer 生成的 public LoRA functionally identity。

梯度预期需要准确表述：

- 首个 functional backward 在 LoRA `B=0` 下首先打开 B-side final heads；
- 上游 Core/Procedure/Meta 梯度会在输出路径被打开后出现；
- mechanics 检查必须完成至少足以打开完整路径的连续 steps，不能错误要求
  step 0 时每个上游参数都已有非零梯度。

## 14. 参数预算

目标 comparator 是 rank-128 Source-SFT：

```text
10,297,344 trainable parameters
```

v5 机械设计预算：

| component | trainable params |
|---|---:|
| VL Meta-LoRA rank 4 | 921,600 |
| Action Meta-LoRA rank 8 | 1,253,376 |
| Core projection `2048→256` | 524,288 |
| Interaction projection `1024→256` | 262,144 |
| 2 causal Procedure blocks | 1,573,888 |
| 1 Core + 1 Procedure compiler | 2,191,104 |
| 8 factor heads，hidden 420 | 3,575,040 |
| **total** | **10,301,440** |

差异：

```text
10,301,440 - 10,297,344 = 4,096
relative difference = 0.0398%
```

公开 LoRA 仍为 `1,287,168` scalars，不计入 Writer trainable parameter
comparator。

上述数字是设计预算，不冒充真实实现证据。实现后必须从真实模型逐模块打印并
核对：

- 每个 Meta-LoRA；
- Core projection；
- interaction projection；
- 每个 Procedure block；
- routing identities；
- Core compiler；
- Procedure refiner；
- factor heads；
- total trainable；
- frozen source trainable count；
- generated public LoRA scalars。

总量应精确等于 `10,301,440`。若实现细节造成偏差，先检查重复 norm、bias 或
identity table；不得用新增科学分支凑参数。

## 15. AS 数据组织：每条 action 对 N 条视频

固定初版：

\[
N=4.
\]

每 rank 的逻辑 action batch 为 `B_a` 条 action queries。对每个 action
query `a_i`，独立从其 task 的 50 条 teacher videos 中无放回抽 4 条不同视频：

\[
\{V_{i,1},V_{i,2},V_{i,3},V_{i,4}\}.
\]

每个 `(action_i, video_{i,n})` 独立生成 one-shot program 和 LoRA：

\[
\theta_{i,n}=Writer(V_{i,n},L_i).
\]

因此逻辑上：

```text
action queries             = B_a
videos per action          = 4
logical LoRAs/pairs        = B_a × 4
functional policy losses   = B_a × 4
```

不能把整个 rank 的 action batch 共享成只有 4 个 LoRA；那是此前被 owner
否决的不同训练定义。

工程上允许按精确键：

```text
(language task, video task, demo_id, video condition/order transform)
```

对重复视频条件做 forward/cache 去重。去重只减少实际 Writer 计算，不能删除
任何逻辑 `(action,video)` pair 或改变其 loss 权重。

同一 action 的四个 pairs 必须共享：

- policy observation/state；
- action target；
- correct policy language；
- policy flow noise；
- preprocessing 和 padding。

它们之间唯一变化是 teacher video 及由它生成的 LoRA。

## 16. AS loss 与 batch/显存合同

每个 pair 的 functional behavior-cloning/flow loss 记为
\(\ell_{i,n}\)。optimizer objective：

\[
\mathcal L_{AS}
=
\frac{1}{B_aN}
\sum_{i=1}^{B_a}\sum_{n=1}^{N}\ell_{i,n}.
\]

这是有意的普通均值：

- 每条 action 等权；
- 每条 teacher video 等权；
- 不按视频长度加权；
- 不给 Core/Procedure 单独 auxiliary loss；
- 不设置对比、顺序或 policy-distance regularizer。

不使用 optimizer gradient accumulation。固定 `N=4` 后，重新 profile：

- 最大安全 per-rank `B_a`；
- 最大有效 frame microbatch；
- 是否将 `B_a×4` pairs 做单次 batched functional forward，或在同一
  optimizer step 内用现有 graph-safe pair microbatch 分块；
- 以实际 global action queries/s 和 optimizer steps/s 决定，而不是只看
  峰值显存。

若 pair 必须物理分块，所有 chunk 的 sum loss 在同一 optimizer step 内按真实
pair 数归一化后一次 `optimizer.step()`；这属于单步内存调度，不是跨 step
gradient accumulation。

world size 和 GPU 显存变化时，科学合同仍是 `每 action × N=4`。只调整
`B_a`、frame microbatch 和单步 pair chunk，不改变 N。

## 17. 推理与评测

推理保持严格 one-shot：

```text
one task language + one teacher video
-> one v5 Writer forward
-> one static task LoRA
-> all replans in that rollout reuse the same LoRA
```

训练时多 demo 共同梯度不改变推理输入数量，也不把 4 条视频拼成一个样本。

评估阶段工程并发与训练科学超参数解耦：

- Writer generation batch 单独 profile；
- rollout replicas 单独 profile；
- 相同可见视频 LoRA按精确视频键生成一次并供 aliases复用；
- 尽量保留已加载 source policy，从 Writer generation 原地切换 rollout；
- 结果同时报告 generation、rollout-only 和 end-to-end 吞吐。

## 18. 最小实现验证

只做避免无效正式实验所必需的检查：

1. exact no-state teacher prompt tokenizer layout；
2. image-position final hidden shape `[B,256,2048]`；
3. Core `16×16→8×8` pooling 和 `[64T,256]` mask；
4. same-frame-set normal/shuffle 的 Core compiler output数值相同；
5. fixed suffix buffer在train/eval/resume中逐 bit 相同；
6. Action Expert只前向一次，不调用 action-out、不产生7D forecast；
7. causal mask：位置 `t` 不读取 future；reverse/shuffle会改变 `E`；
8. Procedure refiner fresh `D=0`；
9. routing identity不能直接进入 value/residual/factor heads；
10. fresh public LoRA functionally identity；
11. 连续 steps 后 Core、Procedure、两个 Meta-LoRA 和 factor heads均有
    finite、符合预期的梯度；
12. frozen source trainable count严格为0；
13. 38 targets/76 tensors/rank16完整一致；
14. real parameter count精确符合预算；
15. checkpoint保存 Writer、fixed suffix、optimizer、scheduler、sampler、
    action/video cursors、每 rank/worker RNG，exact-resume通过；
16. information wall无 teacher action/state/reward/terminal泄漏；
17. GPU4–7真实 OOM/吞吐 profile。

不做仪式性全仓校验，不为探索性内部量新建第二套 runner。

## 19. 第一轮训练长度与 checkpoint cadence

新架构不继承 v4 的“600 steps”等价训练量。完成真实稳定 profile 后：

1. 取多步 steady-state `seconds/optimizer_step` 的中位数；
2. 估计：

\[
S_{hour}=3600/\operatorname{median}(seconds/step);
\]

3. 选择接近 `S_hour`、便于均匀保存的整数 segment size；
4. 每个约一小时 segment 均匀保存 6 个 checkpoint，即约每 10 分钟一次；
5. segment 末 checkpoint必须包含完整 exact-resume state。

scheduler horizon 在首个正式 launch 前一次封存，并保持足够长，后续
exact-resume segment 不因 stop step 改写已有 LR 轨迹。`selected_stop_step`
只是本次 invocation 的一小时边界，不是科学最大步数。

第一段完成后，先用封存的 functional validation panel安排少量候选顺序，再用
固定8-task×50 rollout panel真正选择候选。80-episode快筛已退役，不再使用。

## 20. v5 特异性 gate

对第一段 observed-best，先做低成本内部数值检查，再决定昂贵 rollout。

### 20.1 内部检查

固定多个 validation tasks、每 task 多条 videos，并严格配对 language、fixed
suffix、checkpoint 和任何 policy query。比较：

- correct normal-order；
- same-task other correct teacher；
- cross-suite wrong teacher；
- same-frame-set shuffled；
- same-frame-set reversed。

逐层记录：

```text
language-conditioned image-position hidden
Core tokens C
Core query content Z_C
Action Expert suffix hidden
per-frame interaction s_t
Causal Procedure E_t
Procedure refinement D
final query content Z
generated LoRA tensors
effective B@A
policy-function output on fixed action queries
```

必要方向：

- Core normal与same-frame-set shuffle/reverse近似相同；
- Procedure `E` 与 refinement `D` 对normal/reverse/shuffle有明显、
  跨tasks/videos稳定的差异；
- same-task other 的 Core/LoRA/function变化明显小于wrong；
- 差异必须穿过最终effective LoRA和policy function，不能只停在上游；
- v5 不应重新出现对具体前三维teacher translation的强probe相关。

内部无差异时，不浪费400 rollout；定位最早失败层并修改后fresh重试。

### 20.2 固定400 paired rollout

内部通过后，对同一 checkpoint 运行：

- correct；
- same-task other；
- cross-suite wrong；
- shuffled；
- reversed。

同task/init state/env seed/policy seed/flow noise严格配对。硬门：

```text
same-task other ≈ correct，且影响最小
correct > wrong
correct > shuffled
correct > reversed
```

“大于”必须由多个tasks共同贡献，并结合paired churn/McNemar与独立复测判断，
不能只依赖 aggregate 的1–2次差异。大部分 LIBERO manipulation tasks 都有
一定阶段顺序，因此不把 `correct ≥ shuffled/reversed` 降成无差异可接受。

若same-task other显著波动，说明仍在编码demo-specific低层路径；若wrong不差，
说明task/video语义仍由language公共路径替代；若shuffle/reverse更好，优先检查
Procedure是否重新学成低层轨迹controller，而不是添加顺序loss。

## 21. absolute performance 与继续训练

特异性通过后，再以correct-video validation观察绝对性能：

- 主要最低目标：至少达到或接近旧 Action-Forecast `125/400`；
- 目标：逼近或超过 v4 shuffled `148/400`；
- 四卡rank-128 Source-SFT `108/400`是背景参考；
- 旧八卡 `122/400`不是必须超过的独立硬门。

若第一段还在上升或没有明确峰后下降，按相同合同继续下一个约一小时 segment。
checkpoint可以密集保存，完整400评测只选有代表性的点：

- 当前新高；
- 新高附近；
- 后续明显局部低点；
- segment终点；
- 可能反弹的位置。

functional loss只能安排顺序，不能代替closed-loop选择。

若特异性通过但最优correct长期低于125：

1. 比较 Core-only、Core+Procedure 的 policy-function与逐task行为；
2. 检查多demo共同梯度是否实际成立；
3. 检查 Meta-LoRA是否再次学成低层translation/phase code；
4. 检查 compiler/factor 是否放大或压缩正确表示；
5. 只修改被证据定位的最早失败模块；
6. fresh训练并重复profile/segment/gate。

不得通过扩大总参数、增加额外adapter或对比loss追正结果。

## 22. Cold-start RL Writer

只有 AS 同时通过：

- 内部 Core/Procedure语义；
- correct/same/wrong/shuffle/reverse实际特异性；
- absolute performance；
- validation observed-best与充分峰后探索；

才启动 RL。

RL Writer：

- 使用同一最终 v5 架构与 public LoRA schema；
- 是独立路线，不从完整 AS best继续；
- 先从规定 fresh Writer 进行独立 short-AS cold start；
- 24个train tasks逐task至少获得一次official random-reset success；
- 达成24-task success coverage后永久关闭action入口；
- 随后pure official reward；
- 只使用official reward/success，不使用pose shaping或`.pruned_init`；
- 保存env/policy/worker RNG、seed schedule、interaction cursor和exact-resume；
- 同样在validation找observed-best和明显、复测稳健、多task共同贡献的峰后下降。

RL若无法启动，先判断是reward coverage、runtime mechanics还是表示问题；不得从
完整AS checkpoint接着训练来规避独立路线合同。

## 23. 代码 owner 与退役边界

唯一活动路径：

- `scripts/train_as_writer.py`：唯一AS训练入口；
- `scripts/evaluate_pi05.py`：唯一π0.5 rollout入口；
- `configs/pi05_as_writer_core_causal_v5.json`：唯一活动AS配置；
- `src/ember/writer/video_program.py`：frame prefix、两个Meta-LoRA、Core与
  Action Expert semantic probe；
- `src/ember/writer/temporal.py`：causal Procedure、Core compiler与Procedure
  refiner；
- `src/ember/writer/model.py`：完整v5组合、factor heads与public LoRA输出；
- 既有training/checkpoint/inference/evaluation owner原位适配v5 schema。

原位替换后删除：

- `src/ember/writer/visual_state.py`；
- 误导性的v4 `action_forecast.py`；
- v4 active config；
- 只验证Plan/Revision/Belief/visual-state的活动测试；
- checkpoint兼容分支；
- 独立specificity runner。

历史通过Git、v4文档、findings/progress和外部artifact保存，不在活动源码中保留
双实现。

## 24. GPU、效率、存储与交付

- 后续所有 GPU launch只使用物理4、5、6、7；0–3不进入visible set；
- launch前实时检查GPU owner、显存、温度、利用率和进程拓扑；
- 即使4–7已有他人进程，也只按owner授权共卡，不杀、不暂停、不重置；
- 一卡一DDP rank，GPU0/物理4不额外堆controller或额外模型角色；
- `/data/ymdai`总个人占用硬上限500GB；
- 启动profile/formal前测当前占用和新增checkpoint/cache峰值；
- 复用现有source base、dataset、tokenizer与evaluation infrastructure；
- frame stride 5固定；
- 训练不设总wall-clock上限；
- meaningful状态及时更新`task_plan.md`、`findings.md`、`progress.md`；
- 验证、commit并push到`origin/main`，但文档整理不阻塞已经准备好的GPU关键
  路径。

本文是后续session恢复当前设计的第一阅读入口。
