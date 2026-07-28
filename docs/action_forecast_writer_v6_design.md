# EMBER Writer v6：Task-Grounded Semantic Set + Visual-Transition Procedure

## 1. 决策与版本定位

owner 于 2026-07-28 确认采用本设计，并把原计划中的 v5.3 提升为 v6。
这是当前 fresh Writer 架构和训练 authority。2026-07-28 owner 又明确要求
v6 从 step0 直接采用 task-complete 多任务更新；因此本文早期曾记录的
“首版沿用 v5.2 one-task-per-rank update”已经失效，不再作为对照约束。

v6 不是在 v5.2 上孤立增加一个视觉 transition 的小修订，而是同时完成三项
相互配合的预算与职责重整：

1. Semantic Core 从固定 `95% mean + 5% attention` 改为
   `mean backbone + task-selected centered residual`；
2. Causal Procedure 在 Action-Expert probe 之外显式读取 task-grounded
   adjacent visual transition；
3. 不再为了机械匹配 rank-128 Source-SFT 参数量压缩 factor decoder，
   把其 hidden width 恢复为规整的256。

因此，v5.3 的 Visual-Transition Procedure 设计和原型实现只作 v6 的直接
provenance；后续实现应保持一条 canonical Writer 路径，不保留可同时执行的
v5.3/v6 双路径。

v6 的一个 macro optimizer update 固定为：

```text
4 symmetric DDP ranks
× 6 distinct tasks per rank, sequential and long-video-first
× one teacher video / one generated LoRA / B_a action queries per task
→ mean within each task
→ equal mean over all 24 tasks
→ one DDP synchronization
→ one AdamW update
```

该更新同时是新的训练架构，不再额外保留 one-task-per-rank v6 路径。

## 2. 研究目标与信息墙

v6 保持 EMBER 的核心映射不变：

```text
exact task language
+ exactly one raw action-hidden teacher video
→ understand what the task is and how it unfolds
→ compile one complete task-specific rank-16 LoRA
→ frozen π0.5-LIBERO source policy executes from new initial states
```

Writer 不得读取：

- teacher action；
- proprio/state；
- reward、success、terminal；
- task ID、filename、episode identity；
- hidden normalization；
- policy rollout outcome 或其他部署时不可得信息。

teacher action 只进入 frozen source policy 下的 AS functional loss。Writer
本身的输入始终只有任务语言和一条原始、无 action 标注的 teacher video。

以下合同全部保持不变：

- frozen π0.5-LIBERO source base 与 source-only normalization；
- development 24 train / 8 validation / 8 test split；
- frame stride=5、单 agentview 与 official preprocessing；
- 320个 public LoRA routing identities；
- 一次性生成完整 task-specific rank-16 public LoRA；
- public LoRA targets、template、rank和 functionally identity 初始化；
- 只使用物理 GPU4–7。

## 3. 第一性原理职责分离

v6 把 Writer 分成三个明确层次：

```text
Semantic Core:
  任务相关的对象、关系、位置和目标语义
  对同一 frame set 的排列严格不敏感

Causal Procedure:
  当前画面暗示的操作 + 任务相关视觉证据刚刚怎样变化
  对真实输入顺序敏感，并保持 causal

LoRA Compiler:
  把“是什么”和“怎样推进”编译到320个公开LoRA slots
  不重新读取原始图像，也不建立绕过Core/Procedure的旁路
```

Semantic Core 不承担动作时序；Procedure 不直接读取完整 absolute patches。
这样既避免静态外观/视频身份成为顺序无关捷径，也避免两个分支重复编码全部场景。

## 4. 冻结π0.5特征适配

### 4.1 Meta-LoRA

保持 v5.2 的三组 rank-4 Meta-LoRA：

| 模块 | 作用 | 参数量 |
|---|---|---:|
| Text Meta-LoRA | 18层 PaliGemma q/k/v/o 的 text-only task encoding | 921,600 |
| VL Meta-LoRA | 18层 PaliGemma q/k/v/o 的 frame+language encoding | 921,600 |
| Action Meta-LoRA | Action Expert 内的固定 probe 表征 | 626,688 |

Meta-LoRA 的 A 使用正常小随机初始化，B 为零；source policy 主干保持冻结。

### 4.2 共享投影

```text
language projection:     2048 → 256, bias-free
interaction projection:  1024 → 256, bias-free
```

对应参数分别为 `524,288` 和 `262,144`。

v6 的公共模型宽度保持256；不因为放宽总参数约束就扩到320或384。

## 5. 逐帧task-grounded证据

对 task language 的有效 token 数记为 `L`，采样视频帧数记为 `F`。

### 5.1 Text-only task queries

```text
task language
→ frozen text encoder + Text Meta-LoRA
→ shared 2048→256 projection
→ Q_text ∈ R^(L×256)
```

`Q_text` 只提供任务条件和查询身份，不携带视频内容。

### 5.2 Multimodal task-token evidence

对每帧分别执行：

```text
frame_f + exact task language
→ frozen multimodal encoder + VL Meta-LoRA
→ task-token hidden
→ shared 2048→256 projection
→ H_f ∈ R^(L×256)
```

### 5.3 Task-queried patch grounding

同一 multimodal forward 中保留 image-position hidden，并让 text-only task
queries 读取这些 patch contents：

```text
Q = Wq(RMSNorm(Q_text))            # [L,256]
K = Wk(RMSNorm(image patches_f))    # [N,256]
V = raw image patch values          # no learned Wv
G_f = Wo(Attention(Q,K,V))          # [L,256]
```

- 8 heads，每头32维；
- Q/K/O 可学习，V 无 learned projection；
- valid image positions 和 valid task tokens 严格 mask；
- 参数量 `197,120`。

逐帧完整 task-grounded evidence 为：

```text
E_f = H_f + G_f                     # [L,256]
```

同一 task-token 位置跨帧对应同一任务语义，因此 `G_f` 同时可服务于
permutation-invariant Semantic Core 和相邻 visual transition。

## 6. Semantic Core：mean backbone + centered residual

### 6.1 v5.2固定门的局限

v5.2 使用固定初始化在约0.05附近的 attention gate，把 frame mean 和
attention pooling 混合。现有内部证据中，该门在 step900–1800 仍约为
`0.0490–0.0504`，基本没有承担可学习的 task-conditioned frame selection。
继续保留这一门，会让新视觉证据主要退化为静态均值，也浪费一个本应可归因的
选择机制。

### 6.2 新的集合聚合

先建立不可丢失的静态均值骨架：

```text
M_l = Mean_f(E_f,l)                 # [256]
ΔE_f,l = E_f,l - M_l               # centered frame values
B_l = Wm(M_l)                       # mean backbone
```

再由 text-only task query 选择偏离均值的帧证据：

```text
Q_l = Wq(RMSNorm(Q_text_l))
K_f,l = Wk(RMSNorm(E_f,l))
V_f,l = ΔE_f,l                      # no learned Wv
R_l = Wo(Attention_f(Q_l,K_f,l,V_f,l))
U_l = B_l + R_l
```

性质：

- 对 frame permutation 严格不变；
- static mean 始终保留，不会因 attention 训练不稳而消失；
- 如果 attention 在帧上均匀，centered residual 精确为零；
- text query 只决定读哪一帧，不能凭空向 value 注入视频内容；
- 不需要 sigmoid gate，也不需要人为设定5%的 residual 上限。

该模块使用8头、每头32维；`Wq/Wk/Wm/Wo + two RMSNorm` 共
`262,656`参数。

### 6.3 Language-axis Semantic Core

`U ∈ R^(L×256)` 再进入2个双向 task-token Transformer blocks：

```text
width=256
heads=8
head_dim=32
FFN=256→1024→256
RoPE=task-token ordinal
depth=2
```

参数量 `1,573,888`。输出仍是 permutation-invariant Semantic Core tokens。

## 7. Causal Procedure：Action probe + visual transition

### 7.1 Action-Expert probe

每帧执行：

```text
frame_f + exact task language
+ one fixed persistent Gaussian action suffix at timestep=1
→ frozen π0.5 multimodal / Action Expert + Action Meta-LoRA
→ mean over 50 final suffix hidden tokens
→ 1024→256 interaction projection
→ A_f ∈ R^256
```

`A_f` 不是 teacher action。它只是同一固定 probe 下，π0.5 对当前画面和任务
隐含操作/交互的 latent hypothesis。

### 7.2 Task-grounded adjacent visual transition

对该 arm 的实际输入帧顺序计算：

```text
D_0 = 0
D_f = G_f - G_(f-1),  f > 0        # [L,256]
```

shuffled/reversed control 必须先变换输入帧，再在变换后的顺序内重算 `D_f`。
禁止先用正确顺序计算 transition 后再重排。

首版明确不加入：

- optical flow；
- 显式几何或 patch correspondence；
- 长程 pair matching；
- multi-scale transition；
- absolute patch Procedure 旁路；
- 第二套视觉编码器。

### 7.3 Action probe查询视觉变化

```text
Q_f = Wq(RMSNorm(A_f))             # [1,256]
K_f = Wk(RMSNorm(D_f))             # [L,256]
V_f = D_f                          # raw transition, no learned Wv
R_f = Wo(Attention(Q_f,K_f,V_f))   # [1,256]
Z_f = A_f + R_f
```

- 8 heads，每头32维；
- attention 只允许 valid task tokens；
- padding tokens 和 padding frames 严格为零；
- 参数量 `197,120`。

### 7.4 Causal Procedure encoder

`Z_1...Z_F` 进入2个 causal frame-axis Transformer blocks：

```text
width=256
heads=8
head_dim=32
FFN=256→1024→256
RoPE=sampled-frame ordinal
global causal attention
depth=2
```

参数量 `1,573,888`。它需要把：

```text
approach
→ contact
→ object moves with gripper
→ approach target relation
→ release / establish goal relation
```

一类有序变化编译为可供 LoRA compiler 使用的 causal Procedure tokens。

## 8. Slot-normalized LoRA compiler

v5.2 已证明：

- Core 对 same-frame-set order 近似不变；
- Procedure 的 order 差异能传到 effective LoRA；
- effective LoRA 差异能进一步传到 policy action。

因此 v6 不重写已通过因果传递检查的 compiler。

### 8.1 320 routing identities

```text
PaliGemma q/v targets: 18 × 16 = 288 slots
Action Expert input:               16 slots
Action Expert output:              16 slots
total:                            320 slots
```

routing identity 只进入 attention 的 Q/K，不进入 V/content。

### 8.2 Core与Procedure融合

```text
Core slot reader
→ normalized Core slot

time-centered Procedure values
+ routing identity
+ normalized Core slot
→ Procedure slot read

zero-init, bias-free AdaLN modulation from Procedure
→ modulate normalized Core
→ one post-fusion slot Transformer
```

compiler 参数量保持 `1,535,232`。AdaLN modulation projection 继续
zero-init；不加入 residual escape、shared trainable adapter 或并行 compiler。

## 9. Factor heads与public LoRA

v6 不再把与 rank-128 Source-SFT 的逐参数相等当作硬约束。八个 factor heads
统一采用硬件友好且与公共模型宽度一致的：

```text
256 → 256 → target_width
GELU
final Linear: bias-free, zero-init
```

目标宽度：

| head | target width |
|---|---:|
| `q_A` | 1024 |
| `q_B` | 2048 |
| `v_A` | 1024 |
| `v_B` | 256 |
| `action_in_A` | 32 |
| `action_in_B` | 1024 |
| `action_out_A` | 1024 |
| `action_out_B` | 32 |

八个 heads 共 `2,179,072`参数。

public LoRA 仍为 rank16 complete task-specific adapter。模板 A 非零、B 为零，
factor final projections 为零，因此 fresh step0 生成的 public LoRA 必须与
source policy 精确 functionally identity。

## 10. 精确参数预算

| 模块 | 参数量 |
|---|---:|
| Text Meta-LoRA | 921,600 |
| VL Meta-LoRA | 921,600 |
| Action Meta-LoRA | 626,688 |
| shared language projection | 524,288 |
| task-queried patch grounding | 197,120 |
| semantic set fusion | 262,656 |
| 2× bidirectional Core blocks | 1,573,888 |
| interaction projection | 262,144 |
| visual-transition fusion | 197,120 |
| 2× causal Procedure blocks | 1,573,888 |
| slot-normalized compiler | 1,535,232 |
| factor heads, hidden=256 | 2,179,072 |
| **v6 Writer total** | **10,775,296** |

rank-128 Source-SFT 的 trainable 参数量为 `10,297,344`。v6 多
`477,952`，即约 `4.64%`。这是有意的小幅偏离：

- 所有主要表示宽度统一为256；
- 8 heads × 32维对 Tensor Core 和 kernel layout 友好；
- factor decoder 不再被压到192形成非对称瓶颈；
- 没有为凑参数引入低秩投影、奇怪 hidden width 或矩阵共享。

真实实现后必须以 module enumeration 精确复核，不能只依赖本表手算。

## 11. 初始化与首版边界

初始化合同：

```text
Meta-LoRA:               A normal, B zero
feature projections:     normal initialization
attention/Transformers:  normal initialization
AdaLN modulation:        zero initialization
factor final Linear:     zero initialization
public LoRA template:    functional identity
```

首版不加入 learnable residual gate。Semantic centered residual 和 Procedure
transition residual 都应先用内部尺度与闭环结果验证；只有真实证据显示 raw
transition 幅度不稳定时，才考虑 normalized value 或小 residual gate。

不得为追求正确控制臂结果加入 contrast、order、margin 或
shuffled/reversed supervision。v6 仍只训练 normal-order positive AS。

## 12. 实现、profile与fresh训练合同

### 12.1 唯一canonical实现

- v6 使用新的、fail-closed 的 config/launch/checkpoint/evaluation schema；
- v5.2/v5.3 Writer checkpoint 不兼容，不得 resume；
- v5.3 prototype 只作实现起点，最终只保留 v6 canonical path；
- 首先完成 shape、mask、identity、freeze、gradient、真实 transition 非零；
- 完成一次 step1→3 exact-resume；
- checkpoint 只允许在完整 macro update 边界保存和恢复，不保存半个 macro。

### 12.2 Task-complete macro update

development 的 24 个 train tasks 在每个 macro update 中全量、等权覆盖一次：

```text
for micro_round in 0..5:
    每个rank接收一个不同task
    每个rank读取该task的一条teacher video
    Writer生成该task的一套rank-16 LoRA
    frozen source policy在该LoRA下处理B_a条同task action queries
    task_loss = mean(B_a functional losses)
    backward(task_loss / 6)

前5轮：完整Writer forward/backward处于DDP no_sync()
第6轮：执行本macro唯一一次DDP gradient synchronization
随后：clip_grad_norm → AdamW.step → scheduler.step
```

固定合同：

- 一个 macro 只调用一次 `zero_grad()`、一次 gradient clip、一次
  `AdamW.step()` 和一次 scheduler step；
- 每个 task 的 action loss 先独立求均值，再以 `1/24` 全局等权；长视频不能
  因运行较慢而改变统计权重；
- 每个 macro 的 task/video assignment 是 seed 与 macro cursor 的确定函数；
- 按本次实际选中 teacher video 的 stride-5 frame count做四组 cost balance，
  每组内部最长视频优先，四组随 macro cursor轮换到物理rank；
- video与action episode/chunk仍独立采样；action query保持既有
  task-balanced、episode-balanced no-replacement语义；
- exact resume只恢复完整 macro cursor、sampler/video schedule、四rank RNG、
  optimizer和scheduler。

`B_a=20`时，一个 macro 的精确计数为：

```text
24 task/video conditions
24 generated LoRAs
480 independent action queries
24 frozen-policy functional forwards
1 DDP synchronization
1 AdamW update
```

新 macro step不能与旧 optimizer step直接比较。日志和汇报必须同时记录 macro
updates、task/video conditions、action queries、functional forwards、
wall-clock、GPU-hours和显存。

### 12.3 B20/B16 profile与分段训练

只保留两个硬件友好的 profile 候选：

```text
首选 B20
→ 若真实最长105帧视频OOM，或连续完整macro出现非有限/不稳定
→ 直接回退 B16
```

不测试 B19/B18/B4 等中间档，也不通过 dummy tensor填显存。profile必须使用
GPU4–7、完整6轮 macro、真实最长视频、正常 backward/clip/step，并报告
allocated/reserved、queries/s、task-video conditions/s和macro updates/hour。

旧 v5.2 的900 updates等于3600 task/video conditions；v6中约为150 macro。
因此首个正式段以真实 profile 推导约一小时停止点，初始预计约150 macro；
warmup/checkpoint/decay按 conditions/query exposure换算，peak LR不乘6。

首段结束后做 absolute checkpoint selection、内部 Core→Procedure→effective
LoRA→policy action传递和正式五臂。owner的最终续训规则是：

```text
除非首段已经观察到明确、可信的absolute性能下降，
否则默认exact-resume第二个约一小时segment。
```

平台、小幅波动或仍上升都进入第二段；第二段之后是否再开第三段，必须依据当时
真实曲线和机制证据，不能机械外推固定总step。

### 12.4 Corrected mixed-task Source-SFT

v6确认后必须从同一 frozen source base fresh重训 capacity-matched rank-128
Source-SFT；旧 rank-pure SFT只作provenance。新SFT只有一套 shared LoRA，
因此直接使用 mixed-task physical batch：

```text
task ~ Uniform(24 tasks)
episode ~ Uniform(50 episodes within task)
chunk/frame ~ Uniform(valid chunks within episode)
→ 每个rank的physical batch混合多个tasks
→ task-balanced loss
→ one shared rank-128 LoRA update
```

不得把每rank固定为一个task，不得让长episode或chunk数更多的task获得更高
隐式权重。SFT从fresh identity LoRA训练，根据相同validation合同寻找
observed-best，并完整报告steps、action samples、optimizer updates、
wall-clock、GPU-hours和搜索上限。它在v6完成后默认执行，不再作为可选项。

## 13. 内部证据与成功判据

### 13.1 结构合同

必须直接验证：

- same frame set 的 shuffled/reversed 不改变 Semantic Core；
- visual transition 按每个 arm 的实际顺序重新计算；
- Procedure 保持 causal；
- padding task tokens/frames 不产生内容；
- public LoRA 在 step0 精确 functional identity；
- frozen source policy 除 Meta-LoRA functional path 外没有被更新。

### 13.2 Semantic Core

至少记录：

```text
RMS(mean backbone)
RMS(centered residual) / RMS(mean backbone)
frame-attention entropy
effective number of attended frames
normal / shuffled / reversed Core relative difference
Semantic fusion gradient share
```

目标不是强行制造大 residual，而是证明 task-conditioned帧选择确实工作，同时
保持 frame-set permutation invariance。

### 13.3 Procedure

至少记录：

```text
RMS(D_f)
RMS(R_f) / RMS(A_f)
transition attention entropy
visual-transition module gradient share
normal / shuffled / reversed Procedure relative difference
Procedure-only difference through effective LoRA and policy action
```

内部差异必须传递到有效 LoRA 和 policy action；只有 latent 差异不算成功。

### 13.4 闭环判定

v6 observed-best 必须用同一无放回 fixed-400 contract 与充分训练的 v5.2 比较：

```text
absolute correct performance
same-task-other robustness
correct > cross-suite wrong
correct > shuffled
correct > reversed
multi-task breadth
paired new/lost successes
Core / Procedure / effective-LoRA / policy-action transmission
```

absolute 不能只靠一个 task 抬升。目标是稳定超过 v5.2 的可信 best，并继续向
旧 v4 shuffled `148/400` 区域推进；同时不得出现 v4 式 shuffled/reversed
捷径。若视频特异性成立但 absolute 不涨，下一优先级是多task联合训练优化，
不是继续无证据地扩大 transition 或 factor decoder。

## 14. 一句话架构摘要

```text
task language + one action-hidden video
→ task-grounded per-frame semantic evidence
→ permutation-invariant “what” Core
 + causal visual-transition “how” Procedure
→ slot-normalized compiler
→ complete task-specific rank-16 LoRA
→ frozen π0.5-LIBERO executes the task
```
