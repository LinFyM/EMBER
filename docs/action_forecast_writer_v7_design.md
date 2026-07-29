# EMBER Writer v7：Task-Aligned Semantic Trajectory + Causal Action–Effect Program

状态：2026-07-29 owner 已批准为下一条 fresh Writer 架构，尚未实现。本文先把
从第一性原理推导出的需求、已有信号、最少必要结构和可证伪判据封存下来。
实现时必须原位替换唯一 canonical Writer；v6 代码、配置和 checkpoint 只通过
Git、历史文档与实验 artifact 保留，不得形成可并行执行的 v6/v7 双路径。

v7 使用新的不兼容 schema，从 functional identity fresh 训练，不从任一 v6
Writer checkpoint resume 或 warm-start。

## 1. 设计方法

本设计严格按以下顺序推导：

```text
先回答 Writer 必须知道什么
→ 再盘点 frozen π0.5 prefix/suffix 已经提供什么
→ 已有信号能满足需求时直接使用
→ 只有存在明确缺口时才增加最少结构
→ 必要模块不因“最少”而牺牲 width、heads、depth 或计算表达能力
```

“最少结构”表示不创造没有职责的旁路、token、encoder 或 loss，不表示把必要
模块压成表达能力不足的玩具。

v7 不以模块名称宣称语义成立。Core、Procedure、effective LoRA 与 policy
action 的含义都必须由反事实和闭环行为验证。

## 2. 不可改变的任务与信息墙

核心映射保持：

```text
exact task language
+ exactly one raw action-hidden teacher video
→ one shared Writer forward
→ one complete task-specific public rank-16 LoRA
→ frozen π0.5-LIBERO source policy
```

Writer 不得读取：

- teacher action；
- proprio/state；
- reward、success、terminal；
- task/suite ID、filename、episode identity；
- hidden normalization；
- rollout outcome、object pose 或其它 simulator privileged state。

teacher action 只进入 frozen source policy 下的 AS functional loss。以下合同
不变：

- frozen π0.5-LIBERO source base 与 source-only normalization；
- fixed 24 train / 8 validation / 8 test development split；
- frame stride 5、单 agentview 与 official preprocessing；
- 320 个 public LoRA routing identities；
- 38 targets / 76 A-B tensors / rank 16 / alpha 16 / dropout 0；
- normal-order positive-only AS，不加入 contrast、order、margin 或控制臂
  supervision；
- focused GPU 工作只使用物理 GPU4–7。

## 3. Writer 真正需要的三类能力

### 3.1 Semantic Core：任务中的高层不变量

Core 必须回答：

```text
这是一个什么任务？
涉及哪些语义对象、角色、关系和约束？
任务发生在怎样的环境？
目标状态和语言中规定的子目标顺序是什么？
```

Core 表达的是高层语义身份，不是固定像素位置：

- 同一被操作对象在移动、遮挡或换视角后仍保持同一角色；
- 目标容器、目标关系和任务约束不随视频帧改变；
- 固定与移动相机都应由语义聚合而不是像素坐标硬编码来适应；
- 视觉帧集合部分对 frame permutation 不变；
- task language 中明确写出的对象、关系和子目标顺序必须保留。

Core 的有效容量不能机械固定为少数 learned slots。任务描述 token 数 `L`
自然随对象、关系和多子目标复杂度变化，因此 v7 保留 `L` 个 Core tokens。

Core 不应独自完整回答“具体怎样做”。理解任务和目标可以略微帮助 source
policy，但如果没有教学过程，不能生成一套完整而强的 task LoRA。

### 3.2 Causal Procedure：高层动作—环境效果链

Procedure 必须回答：

```text
机器人在这个阶段试图做什么？
环境中与任务相关的语义状态随后怎样改变？
这些“动作 → 效果”事件按什么因果顺序组成完整过程？
```

目标抽象层级类似：

```text
向目标对象接近
→ 建立接触并夹紧
→ 对象随机械臂移动
→ 接近目标关系
→ 释放并形成目标状态
```

而不是具体左移几厘米、某条 teacher 的速度、抓取角度、绝对 phase 或一条
50-step robot clock。

每个有意义的动作假设都应与随后出现的语义效果绑定。v7 不加入 null/no-match
token：没有额外理由要求正常 teacher 中的动作选择“无结果”。wrong、
shuffled 或 reversed 视频的失败应来自整条 action–effect program 不连贯，
而不是模型在每个局部位置任意选择一个人工 null。

### 3.3 Core 与 Procedure 的职责融合

融合顺序必须是：

```text
先借助 Core 理解“任务是什么、对象是谁、目标是什么”
→ 再以这个理解读取 Procedure 中“实际怎样完成”
→ 由 Procedure 内容生成完整 LoRA
```

结构上禁止 Core 单独产生完整有效 LoRA。Core 只负责条件化 Procedure 的
读取和 public LoRA row 的路由；真正进入 factor heads 的动态 content 必须
来自 Procedure。

期望的反事实行为：

```text
只有 Core、没有有效 Procedure
→ 更理解任务，但生成的 LoRA 接近 identity，不能获得完整教学增益

有连贯 Procedure、但任务语义不匹配
→ 无法把动作—效果链正确绑定到当前对象和目标

匹配 Core + 连贯 Procedure
→ 生成明确帮助 frozen source policy 的 LoRA

错误或不连贯教学信息
→ 倾向接近 source base，而不是强烈破坏 source policy
```

最后一条是闭环期望，不通过人为 negative loss 强制。

## 4. frozen π0.5 已经提供的信号

对每个 teacher frame，套上普通 Meta-LoRA 后，同一次 multimodal prefix
forward 已经提供：

```text
H_img[f]  ∈ R^(256×2048)  # language-conditioned image-position hidden
H_task[f] ∈ R^(L×2048)    # image-conditioned task-token hidden
prefix KV                 # 供 Action Expert suffix读取
```

PaliGemma prefix 内 image 与 text positions 双向交互。因此：

- `H_img` 已经是被任务语言查询过的视觉表征；
- `H_task` 已经包含当前 frame 对任务词语的视觉证据；
- 不需要再运行一套 text-only Gemma；
- 不需要额外视觉 encoder、3D model、optical flow、tracker 或 geometry
  module。

Action Expert suffix 已经提供 source VLA 对“当前图文条件隐含什么机器人操作”
的 pretrained interface。需要保留的不是 7D action forecast，而是 suffix
hidden 中的高层 action hypothesis。

因此 v7 只保留两个 Meta-LoRA：

```text
VL Meta-LoRA      rank4, PaliGemma 18层 q/k/v/o
Action Meta-LoRA  rank4, Action Expert 18层 q/k/v/o
```

二者均 `A` 正常初始化、`B=0`；frozen source weights 不更新。v5.1/v6 的
Text Meta-LoRA 被删除，因为现有 multimodal prefix 已满足其必要职责。

## 5. 总体拓扑

```text
task language + one action-hidden video
        │
        ├─ per-frame PaliGemma + VL Meta-LoRA
        │      ├─ task-token hidden ─┐
        │      └─ image hidden ──────┴─ task-aligned semantic trajectory G_f
        │                                      │
        │                                      ├─ frame mean
        │                                      │    → Semantic Core C
        │                                      │
        │                                      └─ forward semantic change D_f
        │
        └─ same prefix KV
               + one 8-token sparse Action Expert suffix
               + Action Meta-LoRA
               → 8 action probe tokens A_f
                       │
                       └─ bind A_f with D_f
                            → joint 8×L action–effect pooling
                            → one high-level action–effect event E_f
                            → causal Procedure P

320 public-LoRA identities
    → read Core only to form task-conditioned routing queries
    → read Procedure as the only dynamic value/content
    → one content-only slot mixer
    → eight factor heads
    → complete rank-16 public LoRA
```

## 6. Task-aligned semantic trajectory

### 6.1 Shared projection

使用一个共享、bias-free 的 `2048→256` projection：

```text
M_f = W_sem H_task[f]  ∈ R^(L×256)
X_f = W_sem H_img[f]   ∈ R^(256×256)
```

共享投影让 task-token 与 image-position hidden 进入同一语义坐标系，不为
query/evidence 创建两套任意表示。

### 6.2 frame-set-stable task query

从所有有效帧的 task-token hidden 形成：

```text
Q = Mean_f(M_f)  ∈ R^(L×256)
```

`Q`：

- 保留 task-token identity 与语言顺序；
- 对 frame permutation 不变；
- 已经利用整条视频形成稳定的任务语义；
- 不需要额外 text-only forward。

### 6.3 task-to-patch grounding

v5.1 已证明只读取 language-position hidden 会损失空间与对象证据；v5.2 已
证明 task-to-patch grounding 是有效、互补且数值稳定的。因此复用这项已有
机制，而不是创造新视觉模块：

```text
R_f = Wo Attention(
          Wq RMSNorm(Q),
          Wk RMSNorm(X_f),
          X_f)                       # no learned Wv

G_f = M_f + R_f                     # [L,256]
```

配置为 width 256、8 heads × 32。`G_f` 是唯一的逐帧 task-aligned semantic
trajectory；Core 和 Procedure 都从它派生，不再各造一套视觉表示。

## 7. Semantic Core

先做类似语义“长曝光”的 frame mean：

```text
C_0 = Mean_f(G_f)                  # [L,256]
```

这里平均的是 task-token-aligned high-level semantics，而不是原始像素。移动
物体的像素轨迹与瞬时姿态被抑制，但对象身份、角色、目标关系和跨视角反复出现
的语义证据仍可累积。

`C_0` 进入 2 个 bidirectional task-token Transformer blocks：

```text
width       256
heads       8 × 32
FFN         256 → 1024 → 256
position    task-token ordinal RoPE
depth       2
mask        valid task tokens
```

输出：

```text
C ∈ R^(L×256)
```

frame mean 对视频顺序严格不变；task-token RoPE 保留语言中“先做A、再做B”
一类目标结构。首版不再增加 v6 的 centered-frame selection residual：它不是
满足 Core 不变量需求的必要组件，也可能重新保留单帧外观和 demo-specific
瞬时状态。

## 8. 八个 sparse Action Expert probe tokens

### 8.1 精确含义

Writer teacher path 的 Action Expert suffix 从 v6 的 50 positions 改为一次
forward 中的 8 positions：

```text
native 50-position anchors:
[0, 7, 14, 21, 28, 35, 42, 49]
```

这是：

- 一次 Action Expert forward，suffix sequence length 为 8；
- 不是 8 次独立 forward；
- 不改变 execution policy 原生 50-action chunk；
- 不把 8 个位置重新编号成 `0..7`；
- 保留其在原生 50-token horizon 中的 position IDs 与固定 Gaussian rows。

同一 persistent native Gaussian buffer 仍先按 `[50,32]` 封存；Writer 只索引
上述 8 行。fixed flow time 保持 `t=1`，使用 frozen native
`action_in_proj` 与 time embedding。

### 8.2 为什么不是 1 或 50

一个 1024 维 token 虽然宽，但会完全删除 Action Expert 预训练 suffix 中的
粗粒度阶段结构，并把“完整动作摘要”压力都交给 rank-4 Meta-LoRA。

保留 50 个 token再统一均值则计算冗余，并把大量低层 action-slot clock 带回
Writer。8 个均匀锚点保留开头、中间、结尾和多因素表达能力，同时避免把
Procedure 重新定义为50步动作轨迹。

取 final norm 后、`action_out_proj` 前的 hidden，并用共享 bias-free
`1024→256` projection：

```text
A_f ∈ R^(8×256)
```

不做8-token均值；8个 action hypotheses 与task-grounded semantic change
在同一次联合注意力中直接汇成一个高层event。

## 9. Action–Effect binding

### 9.1 forward semantic change

动作应与随后出现的效果配对，因此不再使用 v6 的 backward
`G_f-G_(f-1)`。对每个有效相邻区间：

```text
D_f = RMSNorm(G_(f+1)) - RMSNorm(G_f)
      ∈ R^(L×256),  f=0..F-2
```

`D_f` 表示从当前 frame 到下一 frame 的 task-grounded semantic change。
shuffled/reversed control 必须先重排实际输入 frames，再在该顺序内重新计算
`G_f` 与 `D_f`；不得携带原视频的 transition 或 image position。

### 9.2 联合 action–effect attention pooling

8个 Action tokens 全部参与绑定，但不作为8个tokens进入Procedure。每个head
直接在全部 `8×L` 个 action–effect pairs 上归一化：

```text
A_hat[k] = RMSNorm(A_f[k])

s[h,k,l] =
    dot(Wq[h] A_hat[k], Wk[h] D_f[l]) / sqrt(32)

alpha[h,k,l] = softmax over all valid (k,l) pairs
```

value以环境语义变化为主体，Action只逐通道调制其解释：

```text
g[h,k] = 1 + tanh(split_h(Wg A_hat[k]))
V[h,k,l] = split_h(D_f[l]) ⊙ g[h,k]

e[h] = sum_(k,l) alpha[h,k,l] V[h,k,l]
E_f = Wo Concat(e[0], ..., e[7])     # [256]
```

`Wq/Wk/Wo/Wg`均为bias-free `256→256`，没有learned `Wv`；`Wg` zero-init。
这一算子具有以下结构性质：

- 所有8个suffix tokens都进入 `8×L` pair logits、softmax与梯度；
- 不提前mean，也不把8个tokens带入Procedure；
- 每个frame interval直接产生一个高层event；
- `D_f=0 → E_f=0`，Action不能单独形成Procedure或LoRA内容；
- Core完全不参与，双流直到LoRA compiler才首次相遇；
- 初始`Wg=0`，Action先通过pair selection发挥作用，不随机放大value。

相对“两阶段每个Action分别读D，再用EventRead做8→1”的方案，联合pooling少
一次人为归一化和一个learned bottleneck，也没有mean-query相消问题。其主要
可观测风险是joint softmax过早集中到少数Action probes；首轮记录每个head的
action marginal mass与pair entropy。只有真实出现跨head一致的probe collapse
且限制性能时，才考虑恢复分层聚合，而不提前保留第二个pooler。

## 10. Causal Procedure

所有区间事件：

```text
E = [E_0, ..., E_(F-2)] ∈ R^((F-1)×256)
```

进入 3 个 causal frame-interval Transformer blocks：

```text
width       256
heads       8 × 32
FFN         256 → 1024 → 256
position    当前输入序列slot对应的 sampled-frame RoPE
depth       3
mask        causal + valid intervals
```

输出：

```text
P ∈ R^((F-1)×256)
```

三层用于组织长链任务中的多级 action–effect dependency；每层仍保持规整的
256/8×32/1024容量。Procedure 不读取 absolute patch、raw Action hidden、
7D forecast、robot absolute time、normalized video progress 或 learned
position value。

## 11. Procedure-content-only LoRA compiler

公开 routing identities 保持：

```text
18 expert layers × rank16 = 288
action_in rows            = 16
action_out rows           = 16
total                     = 320
```

静态 identity `R_i` 只进入 attention Q/K，永远不进入 factor content。

### 11.1 Core只形成任务条件化query

Core与Procedure在这里第一次交互。每个 public LoRA slot先读取 Core：

```text
c_i = CoreRead(R_i, C, C)
q_i^C = RMSNorm(R_i + c_i)
```

这一结果只作为下一步 Procedure attention 的 query，不作为 LoRA slot value，
也不直接送入 factor heads。

### 11.2 Procedure提供全部动态content

```text
z_i = ProcedureRead(q_i^C, P, P)
```

两个 readers 都使用完整 8-head Q/K/V/O cross-attention，width 256。最终
dynamic slot `z_i` 的 value/content只来自 Procedure：

```text
P = 0 → z = 0
```

因此 Core-only 无法生成完整 task LoRA；Core 的作用是让每个 LoRA row知道应
从 Procedure 的哪些事件中读取什么。

### 11.3 一个 content-only slot mixer

```text
z ← z + SelfAttention(
          Q=R+RMSNorm(z),
          K=R+RMSNorm(z),
          V=z)

z ← z + FFN(RMSNorm(z))
```

配置为1 block、width256、8 heads、FFN1024。它只协调已经读取到的 Procedure
content，不能重新引入 Core residual、routing content 或公共 LoRA 旁路。

v5.1/v6 的 Core-primary AdaLN 融合被删除。它在逻辑上仍把 Core 定义为 LoRA
主体、Procedure定义为调制项，与 owner 的“Core只协助理解、Procedure承载
怎样完成”职责相反。

## 12. Factor heads 与 public LoRA

保持8个共享、bias-free factor heads：

```text
256 → 256 → target_width
GELU
final Linear zero-init
```

target widths不变：

```text
q_A/q_B/v_A/v_B                 1024/2048/1024/256
action_in_A/action_in_B           32/1024
action_out_A/action_out_B       1024/32
```

identity template 的 A 为确定性非零 basis、B 为物理零。factor finals为零，
所以 fresh Writer 生成的 public LoRA 与 source policy精确 functionally
identity；不能把 identity 误解为所有上游模块都需零初始化。

## 13. 参数预算

首版机械设计预算：

| component | trainable params |
|---|---:|
| VL Meta-LoRA rank4 | 921,600 |
| Action Meta-LoRA rank4 | 626,688 |
| shared semantic projection `2048→256` | 524,288 |
| task-to-patch grounding | 197,120 |
| 2 bidirectional Core blocks | 1,573,888 |
| Action projection `1024→256` | 262,144 |
| Action–Effect binder | 262,656 |
| 3 causal Procedure blocks | 2,360,832 |
| Procedure-content-only compiler | 1,403,904 |
| factor heads, hidden256 | 2,179,072 |
| **v7 Writer total** | **10,312,192** |

对比：

```text
rank-128 Source-SFT        10,297,344
v7 Writer                  10,312,192
difference                    +14,848
relative                    +0.144%
v6 Writer                  10,775,296
v7 relative to v6            -463,104
```

该预算不是必须机械贴齐 Source-SFT 的硬上限。它保留所有硬件友好宽度，同时
比 v6 更小。实现后必须以真实 module enumeration 为准；若不一致，先排查
bias、重复 projection/norm、routing table与共享矩阵，不通过新增模块凑数。

## 14. 明确不加入的东西

首版不加入：

- Text-only Gemma 或 Text Meta-LoRA；
- learned Core/event/null tokens；
- optical flow、3D reconstruction、geometry、tracker 或第二套视觉encoder；
- absolute patch → Procedure 旁路；
- Action-only → Procedure/LoRA residual；
- 7D action forecast、Plan/Revision/Belief 或50-step robot clock；
- Core → factor heads直接内容路径；
- Core-primary AdaLN；
- dynamic slot bank、shared update subspace、residual escape 或额外 execution
  adapter；
- contrast/order/margin/control-arm loss。

任何新增结构只能由真实内部或闭环证据指出一个现有信号无法满足的具体需求后
再提出。

## 15. 初始化和结构验证

初始化：

```text
VL/Action Meta-LoRA        A normal, B zero
feature projections        normal
attention/Transformers     normal
Action modulation Wg       zero
factor final Linear        zero
public LoRA template       functional identity
```

实现后的最短必要检查：

1. exact prompt/task-token spans与valid masks；
2. suffix长度精确为8，索引与position IDs为`[0,7,14,21,28,35,42,49]`；
3. policy execution 的50-action chunk完全不变；
4. Core对同frame set的shuffle/reverse在浮点容差内不变；
5. forward transition按各arm实际输入顺序重算；
6. binder在`D=0`时严格输出零；
7. joint action–effect pooling在`D_f=0`时严格输出零，且不读取Core；
8. Procedure满足causal prefix不变性；
9. Core-only compiler输出严格为identity delta；
10. routing identity不能进入value/content/factor heads；
11. 38 targets / 76 tensors / rank16完整匹配；
12. 连续真实steps后必要模块获得finite gradient；
13. frozen source trainable count为0；
14. checkpoint/resume覆盖fixed suffix、Writer、optimizer、scheduler、
    sampler/video cursor与四rank RNG；
15. information wall无泄漏；
16. GPU4–7最长真实视频完成吞吐/OOM profile。

## 16. 训练与快速迭代原则

v7 首个训练 recipe 不以旧 step/macro 数机械继承。实现与真实 profile后，根据
吞吐封存约一小时以内的首段；在正式小时段前允许更短的 disposable mechanism
probe，但不能用小分母 rollout 宣称性能。

实现后的真实profile已封存：B32/B24均在首个functional policy forward OOM；
B20连续3个完整macro finite，首步包含105-frame最长视频，后两步均值约
`27.48 queries/s`、`206.08 macros/hour`，峰值allocated/reserved为
`77,020,274,176/83,647,004,672 bytes`。step1→3 exact-resume保持task、
video、query、LR和cursor身份，joint binder的262,656个参数全部收到真实更新。
因此首轮正式合同为task-complete B20、fast cosine decay400、identity
fresh 0→200 macro、每25 checkpoint；除非首段出现可信absolute下降，否则
默认续到400。

首轮优先使用当前最高 single-checkpoint absolute 所支持的稳定化事实：

- task-complete能获得较高absolute，但可能弱化Procedure增益；
- old rank-rotating能放大order dependence，但absolute、wrong语义和task
  breadth较差；
- fast LR decay把v6 single-checkpoint推到143，但仍有task drift；
- Source-SFT global-8没有解决共同漂移，不能把“task数更少”当作默认答案。

因此训练修改必须围绕可观测瓶颈，而不是任意换 recipe。每个短迭代只回答一个
问题：

```text
表示是否正确形成？
→ Core/Procedure是否按职责分离？
→ Procedure content是否传到LoRA/action？
→ correct absolute是否由多个tasks共同提高？
→ 能力漂移来自梯度冲突、持续步幅还是表示/编译？
```

checkpoint筛选仍使用 paired fixed correct400；只有当前best才做正式
correct/same/wrong/shuffled/reversed full400与内部反事实。不得使用多
checkpoint融合或ensemble作为最终方法。

## 17. 成功与继续判据

focused AS最低门：

```text
single-checkpoint correct400 >= 150
single-checkpoint correct400 >= corrected Source-SFT best + 30
```

当前 corrected Source-SFT best 为109，所以两项统一为150。达到150不是自动
停止；还必须：

- same-task-other与correct同档；
- correct显著优于wrong、shuffled、reversed；
- 优势由多个tasks共同贡献；
- 独立RNG/video permutation复测成立；
- 内部Core/Procedure/effective-LoRA/action链条符合职责；
- 没有v4式控制臂捷径、language/public-LoRA旁路或validation泄漏。

若absolute仍未满足，继续训练、诊断或fresh单变量实验，直到出现明确瓶颈和
充分负证据；不能因几个略低checkpoint随意放弃。若表现仍上涨，不以机械
wall-clock停止；若出现远超400-rollout噪声、多个tasks共同贡献且独立复测的
持续post-best下降，才认为当前recipe上限已测清。

只有v7 AS通过absolute与机制门后，才推进 matched action one-shot、独立短AS
cold-start→pure-reward RL-Writer、final-32与test闭环。
