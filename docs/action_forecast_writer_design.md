# Action-Forecast Writer canonical architecture design

状态：2026-07-26 owner 最新对齐。本文是当前 focused Writer 子任务的
唯一活动设计 authority。

面向外部专家的自包含问题、历史与结果入口见
[`action_forecast_writer_expert_consultation.md`](action_forecast_writer_expert_consultation.md)。

历史 Action-Memory、Action-Forecast v1/v2、28-slot Belief-v3、冻结随机
visual-state decoder、累计 transition、Plan/Revision 双 token、order contrast
和旧 `0→600` 执行口径只保留为 Git、`findings.md` 和既有结果 provenance。
它们不是活动实现，不得从旧配置、旧 checkpoint 或 Git 历史恢复。

若本文与 `task_plan.md`、`findings.md`、`progress.md` 的已完成历史条目冲突，
历史结果不改写，但当前实现和后续执行以本文及 owner 更新指令为准。

## 0. Active Goal

当前 focused Goal 是：

> 保持已经实现并通过75-step内部机制gate的Action-Forecast Writer v4不变；正式训练固定结束于step 2400，不再续训。弃用波动过大的80-episode快筛，仅在现有checkpoint中用同一固定8-task×50 validation panel充分寻找当前observed-best。随后对observed-best完成内部及完整rollout特异性检查，包括correct、same-task other correct teacher、cross-suite wrong、shuffled和reversed；重点检验同任务不同正确示范的影响是否显著小于错误任务或破坏顺序。完成证据与分析记录后停止并向owner汇报，本轮不进入cold-start RL。全程不使用对比损失，固定frame stride=5，只使用GPU 0、1、2、3且不触碰4–7。

本轮结束条件是现有checkpoint的observed-best和上述特异性证据已经完整获得、分析
并记录。它不要求证明无限训练下的全局ceiling，也不再要求通过继续训练观察峰后
下降。RL及其后续阶段均等待owner后续决定。

## 1. 科学意图与完整前向链路

Writer 不是把视频压成 task latent 后直接生成 LoRA。它必须先把 teacher 视频
翻译为机器人 future-action forecasts，再用 Plan/Revision 表示 teacher 如何
逐步完成任务，最后生成 task LoRA：

```text
correct task language + one action-hidden teacher video
  -> per-frame visual-state
  -> per-frame PI05 future-action forecasts
  -> same-absolute-time forecast alignment
  -> Plan_u + Revision_u
  -> Belief_u = concat(Plan_u, Revision_u)
  -> two-layer Temporal Transformer
  -> content-conditioned LoRA Query Decoder
  -> complete public rank-16 task LoRA
```

Action Meta-LoRA 的直观作用是让机器人根据 observer-view teacher 视频想象：
“假如我是视频中的 teacher，此刻接下来会怎么动”，并输出机器人坐标和动作空间
中的 future-action forecast。

任务、语言、物体、场景和动作方案中跨时间稳定的内容有价值，必须保留。活动
设计从未要求删除所有恒定信息。需要结构性阻止的只有：

- visual-state 的动态分支由语言或静态 query 直接生成任意 task code；
- 静态 LoRA query identity 不读取视频 memory 就直接生成 public LoRA；
- 已确认会覆盖有向 Revision 或抹掉其强度的处理。

不存在独立的 `G -> LoRA` 旁路。所有 teacher-video 信息，包括稳定和动态内容，
都必须先影响 action forecasts，再通过 Plan/Revision 和 Temporal memory 进入
LoRA decoder。

## 2. 不可改变的信息墙

- Writer 输入始终是正确 task language 加恰好一条 action-hidden teacher
  video。
- 当前 sealed teacher video 只读取 `obs/agentview_rgb`。不得静默加入 wrist
  image、第二条视频或真实 robot state。
- Writer 不得读取 teacher action、proprio/state、reward、terminal、task ID、
  filename、隐藏 normalization 或可反推出隐藏动作的元数据。
- AS action 只进入 frozen source policy 的 functional behavior loss，不进入
  Writer。
- teacher video 与 action episode 在同 task 内独立采样，不要求同 episode。
- policy 在 AS、correct/wrong/shuffled/reversed 评测中始终接收正确的
  action-query task language；变换的只有 Writer 视频。
- source base、source normalization、24 development-train tasks 和 sealed
  public LoRA schema 不变。
- 不使用 `pi05_libero`、MemLLM、bank、geometry、shared update subspace、
  residual escape、额外 shared adapter 或未 merge source LoRA。
- AS 只使用 normal positive functional action loss。不得加入 contrast、
  order classification、margin 或其他人为制造顺序差异的 loss。

AS 监督来自同任务独立 action episode，因此它不是对 teacher 隐藏 action 的
逐帧复刻。Action-Forecast 路径的作用是从 action-hidden video 提取执行方法，
再生成可在官方新初态上工作的 task policy。

## 3. 视频与 PI05 原生 prompt

- 视频按固定真实控制时间采样，`frame_stride=5` 固定，不再测试 stride 10。
- 视频长度 `T` 保持可变；batch 内只为张量化 padding，并始终携带真实原始
  frame index 和 mask。
- 每帧使用 frozen SigLIP/projector 的完整 projected image tokens
  `X_t [N_img,2048]`，不恢复旧16-token粗池化 Action-Memory。
- 同一帧的当前图像、完整正确语言和 visual-state block 进入 frozen
  PaliGemma contextual prefix。
- 使用 source-native prompt 顺序：

```text
image tokens
+ Task: <correct language>, State:
+ 32-position visual-state block
+ ;\nAction:
```

PaliGemma tokenizer 对中性状态字符串
`" 128 128 128 128 128 128 128 128"` 的实际分词已经核验为精确32 tokens：

```text
[space,1,2,8] × 8
```

因此不再使用28个 placeholder 加外部 whitespace 的旧口径。`State:` 后没有
额外未计入的空格位置；32-token block 自己以真实 whitespace token 开始。

## 4. Visual-state：初始锚点与非递归相对变化

### 4.1 语义

8个 visual-state coordinates 不声称等于真实机器人 proprio 轴。Writer 没有
state supervision，因此它们被定义为：

> source Action Expert 可以利用的、低维、连续、video-conditioned
> visual state-and-motion coordinates。

它们同时表达：

- 该视频自己的初始状态；
- 当前帧相对视频起点的状态变化；
- 当前帧相对上一帧的局部运动方向。

### 4.2 为什么不用累计积分或逐帧独立预测

纯累计：

```text
h_t = h_0 + sum(delta_h_1 ... delta_h_t)
```

会把每一步误差永久传到后续帧，视频越长漂移越大。

逐帧独立：

```text
h_t = AbsoluteReader(X_t)
```

没有漂移，但在当前 task-level AS loss 下最容易重新退化为每帧近似相同的
task latent。

活动设计使用初始锚点加每帧重新计算的相对变化，不递归依赖上一时刻的估计：

```text
h_0 = InitialStateReader(X_0)

D_anchor_t = X_t - X_0
D_local_t  = X_t - X_(t-1)

c_t = ChangeReader(X_0, X_(t-1), X_t,
                   D_anchor_t, D_local_t)

h_t = h_0 + c_t
z_t = tanh(h_t)
```

`h_t` 不使用 `h_(t-1)`，因此第 t 帧的估计误差不会传播到 t+1。

### 4.3 InitialStateReader

- 只读取第一帧 frozen image tokens `X_0`，不读取语言。
- 使用8个 coordinate routing identities。
- routing identity 只进入 attention Q/K；输出 content 只能来自 `X_0` 的
  values，query 本身不进入 residual/output。
- 当前实现使用共享bias-free `2048→128` image projection、一次8-query
  cross-attention、zero-preserving FFN和共享bias-free scalar head，得到
  `h_0 [8]`。
- `h_0` 允许包含初始机械臂、物体位置、场景状态和其他跨时间稳定信息。这是
  合法状态，不是需要删除的恒定成分。
- 8-scalar瓶颈和下述数字 embedding renderer 阻止它直接生成任意
  `32×2048` soft prompt。

### 4.4 ChangeReader

ChangeReader 同时查看相对起点和局部相邻变化。routing/key 可读取交换顺序不
改变的 pair context：

```text
(X_0 + X_t) / 2
abs(X_t - X_0)

(X_(t-1) + X_t) / 2
abs(X_t - X_(t-1))
```

value content 只能读取有符号差分：

```text
X_t - X_0
X_t - X_(t-1)
```

具体约束：

- 不读取语言；
- 静态 coordinate query 只负责 routing，不进入 value/residual/output；
- change content 路径无 additive bias；
- content FFN使用odd `tanh`而非GELU；在key固定且signed value反向时，
  reader输出也严格反向；
- 相同 pair 的 signed change 严格为零；
- 交换局部前后帧时 local value 反向；
- `t=0` 时 `c_0=0`；
- 没有 `h_(t-1) -> h_t` 递归。

这一结构保证动态分支不能凭语言、静态 query 或 bias 生成任意 task latent。
它仍可能把真实变化映射得很小，或被下游忽略；仅靠当前 AS 目标不存在能够
数学保证模型一定使用视频变化的架构。该失败必须在75-step内部诊断中被直接
发现，不能用 contrast loss 掩盖。

### 4.5 32-token native-anchor renderer

固定锚点 `E_base [32,2048]` 来自 frozen PaliGemma embedding table 对
`" 128 128 128 128 128 128 128 128"` 的真实 embeddings。

- 8个 whitespace 位置 `0,4,...,28` 始终保持对应 frozen whitespace
  embedding，不可学习。
- 每个 coordinate `z_(t,d)` 只控制其后3个数字位置。
- 数字位置的可学习方向被限制在 frozen digit token `0..9` embeddings 张成的
  子空间；不得直接生成任意2048维向量。
- 概念公式为：

```text
delta_E_(t,d,j)
  = z_(t,d) * sum_n C_(d,j,n) * DigitBasis_n

V_(t,d,j)
  = E_base_(d,j) + delta_E_(t,d,j)
```

其中 `d=0..7`、`j=0..2`、`n=0..9`。`C` 跨所有tasks/videos共享。

- fresh initialization 时 `z_t=0`，32个位置精确等于原生中性状态 prompt；
- 初始化必须同时保留到 state readers 的非零梯度，不能把 renderer 和
  state head 两侧同时置零；
- 使用连续 soft embeddings，不做 argmax、整数rounding或重新tokenize。

当前实现将reader的8维output gate置零、renderer coefficients保持非零随机值。
因此fresh前向严格为原生anchor；下游functional identity head打开后，gradient
先进入output gate，随后进入完整reader。digit basis使用冻结的10个digit
embeddings减去其均值；这不改变它们张成的子空间。

## 5. 两个 trainable Meta-LoRA

两个 Meta-LoRA 都保留为标准、全 token 位置生效、functional-identity
initialization：

- VL Meta-LoRA：PaliGemma 18层 q/k/v/o，rank 4；
- Action Meta-LoRA：Action Expert 18层 q/k/v/o，rank 8。

标准 LoRA 可以选择 layer/projection，不能天然选择只适配state token或video
token。本设计不增加position-gated LoRA或state/video mask。

VL Meta-LoRA负责理解 observer-view teacher 中的物体、场景、动作阶段，并为
未来人类视频提供视觉域适配。

Action Meta-LoRA负责将 observer/non-robot-view 表征转换成 robot-centric
future actions，即让机器人想象“假如我是视频中的teacher，我接下来会怎么动”；
同时让 Action Expert 学会解释新的 visual-state block。

两者都属于 Writer teacher-understanding path，执行生成后的public task LoRA
时不携带。

## 6. 每帧 PI05 Action Forecast

每个采样帧运行真实 frozen PI05：

```text
current image + correct language + 32 visual-state embeddings
  -> PaliGemma + VL Meta-LoRA contextual prefix/KV
  -> Action Expert + Action Meta-LoRA
  -> complete 10-step flow integration
  -> normalized action forecast [50,7]
```

固定合同：

- `num_flow_steps=10`；
- `action_horizon=50`；
- 总 action dimension 就是7，不是从更多真实action维度中“取前7维”；
- 使用 frozen source action normalization；
- 同一 video condition 内所有帧使用完全相同的可恢复 Gaussian
  `[50,32]` flow noise；
- correct/wrong/shuffled/reversed paired诊断复用相同noise；
- 每帧 PaliGemma prefix只算一次，KV在10次flow中复用；
- base weights冻结，但梯度必须穿过PaliGemma和10次flow回到
  visual-state readers及两个Meta-LoRA；
- 只保留最终 `[T,50,7]` plans，不保存10×18层hidden states。

## 7. 绝对时间对齐

采样帧的真实控制步为 `t_i`，其forecast为 `P_i[k]`，`k=0..49`。对应绝对
动作时刻：

```text
u = t_i + k
```

同一 `u` 通常被多个不同lead time的forecast覆盖。所有Plan/Revision都必须在
这个严格对齐后的集合上计算。

倒序视频不是把每帧action chunk倒序。每一帧仍预测它认为的正向未来动作；
顺序变换改变的是观察顺序、anchor/local变化、forecast集合和绝对时间对齐后的
一致性。

## 8. Plan

对每个绝对时刻 `u`，选择覆盖该时刻的最新帧预测：

```text
i*(u) = max { i | t_i <= u < t_i + 50 }
plan_raw_u = P_i*(u - t_i*)
```

它表示距离 `u` 最近、信息最充分的当前最佳动作预测。

编码保持最简单：

```text
Plan_u = bias_free_linear_7_to_128(plan_raw_u)
```

lead只允许作为Temporal Q/K routing metadata，不进入Plan value。Plan输出不做
会把所有动作幅度拉到单位尺度的最终RMSNorm。

## 9. Revision

Revision 不比较相邻forecast，也不重新编码old/new绝对action。所有更早且覆盖
同一 `u` 的forecast都相对最新Plan比较：

```text
r_(i,u) = plan_raw_u - P_i(u - t_i),  i < i*(u)
```

它表示随着新帧到来，较早预测要怎样修正到当前最佳预测，以及多次forecast是否
一致。

直接计算：

```text
signed_mean_u[d] = mean_i r_(i,u,d)
per_dim_rms_u[d] = sqrt(mean_i r_(i,u,d)^2)

z_u = bias_free_MLP_14_to_128(
        concat(signed_mean_u, per_dim_rms_u))

D_u = RMSNorm(z_u)

m_u = sqrt(mean_(i,d) r_(i,u,d)^2)

Revision_u = stopgrad(m_u) * D_u
```

约束：

- 无更早forecast或所有forecast完全一致时，Revision严格为零；
- `m_u` 是 frozen source normalization 下的无量纲原始7维action residual
  RMS；
- 不使用 `tau`、训练集分位数、人工gate上限或其他强度超参数；
- `stopgrad(m_u)` 防止AS loss通过主动放大/缩小forecasts操纵显式分歧；
- AS仍可通过方向 `D_u` 和Plan路径训练forecasts；
- count、lead、age只可作为Q/K routing metadata，不生成Revision content；
- 删除旧 old/new/delta event attention 和
  count/mean/std/max additive stability branch。

## 10. 单-token Belief

同一绝对时刻先合并：

```text
Plan_u     [128]
Revision_u [128]

Belief_u = concat(Plan_u, Revision_u) [256]
```

Temporal接收：

```text
Belief_1, Belief_2, ..., Belief_U
```

Plan/Revision不是两个交错的Temporal tokens，不使用token-type embedding，也
不做post-concat projection或分别把两个half重新归一化。

## 11. Belief与Temporal的归一化

Revision的raw magnitude不能被普通RMSNorm抹掉。Temporal attention使用：

```text
Q = Wq(Norm(Belief) + routing)
K = Wk(Norm(Belief) + routing)
V = Wv(Belief)
```

- Q/K可以读取latest lead、revision count、detached strength和absolute-time
  RoPE；
- V只读取raw Belief；
- routing、position和bias不能作为额外video content写入value residual；
- 不减去Temporal time mean；
- 不删除跨时间稳定的任务、场景、物体或动作方案内容。

## 12. Temporal Transformer

保持：

- width 256；
- 8 heads；
- 2 blocks；
- variable length和padding mask；
- 真实absolute control-time 1D RoPE；
- pre-norm attention/FFN；
- raw-content residual。

两层足够。此前主要坍缩发生在visual-state/forecast前端，不能因特异性不足盲目
加深Temporal。

attention output与FFN final projection使用identity-safe initialization，使
fresh Temporal接近传递原始Belief，不在step0随机把共同成分放大数倍。

Temporal不承担制造顺序特异性的责任；它只组织已经存在的有序Belief trajectory。
若新架构实测再次显示它显著压缩输入差异，才根据逐层证据修改。

## 13. Content-conditioned LoRA Query Decoder

public LoRA schema不变：

- 320 queries；
- 288个 `18 layers × rank16` expert queries；
- 16个action-in rank queries；
- 16个action-out rank queries；
- sealed 38 targets / 76 tensors；
- complete rank-16 public task LoRA，共1,287,168 scalars。

静态routing与视频content分开：

```text
R = query/module/layer/rank routing identities
Z_0 = 0
```

每个decoder block先读取memory：

```text
A_cross = Attention(
    Q = Norm(Z) + R,
    K = Norm(M),
    V = M)

Z = Z + A_cross
```

再让已经读到content的queries互相通信：

```text
A_self = Attention(
    Q = Norm(Z) + R,
    K = Norm(Z) + R,
    V = Z)

Z = Z + A_self
Z = Z + FFN(Norm(Z))
```

保持2个decoder blocks。

约束：

- 静态 `R` 只负责“哪个module/layer/rank应该读什么”；
- self-attention V只读Z；
- cross-attention V只读raw Temporal memory；
- factor heads只读取最终Norm(Z)，不能读取R；
- 时间稳定的视频内容可正常从M进入Z；
- 被切断的只是完全不读取视频、仅靠静态query table生成公共LoRA的旁路。

## 14. Factor heads与functional identity

8类factor heads继续按真实 `LoraTensorSpec` 生成：

```text
q_A, q_B, v_A, v_B
action_in_A, action_in_B
action_out_A, action_out_B
```

- tensor name、shape和transpose规则从真实identity template读取，不手写猜测；
- factor heads只读取query content Z；
- fresh generated delta加到真实identity template；
- public LoRA fresh状态必须functionally identity；
- initialization必须让首步仍可通过zero-B/nonzero-template-A路径获得梯度；
- 不增加独立公共LoRA、escape branch或平行schema。

## 15. 冻结与可训练对象

冻结：

- canonical PI05-LIBERO source base；
- source vision/PaliGemma/Action Expert base weights；
- source normalization；
- public LoRA mount schema和identity template；
- tokenizer embedding table本身。

可训练：

- InitialStateReader；
- ChangeReader；
- 8-coordinate digit-subspace renderer coefficients；
- VL Meta-LoRA rank4；
- Action Meta-LoRA rank8；
- Plan lift；
- Revision direction encoder；
- 2-layer Temporal；
- 2-layer LoRA Query Decoder；
- factor heads。

严禁冻结随机初始化的visual-state reader作为正式方案。两个Meta-LoRA都保持
trainable。

## 16. 参数预算

比较对象是rank-128 Source-SFT的 `10,297,344` trainable parameters。新Writer
目标仍是约10.3M，不能依靠明显更大容量解释结果。

实现后必须从真实model打印：

- Initial/Change visual-state readers与renderer；
- VL Meta-LoRA；
- Action Meta-LoRA；
- Plan/Revision；
- Temporal；
- Query Decoder；
- Factor heads；
- total trainable；
- generated public LoRA scalars。

rank4/rank8、2-layer Temporal、2-layer Query Decoder、320 queries、rank16 public
schema和上述信息流固定。若简化后的Plan/Revision使总量偏离10.297M，只允许用
不改变科学语义的hidden width（优先factor-head hidden width）做容量校准；不得
增加新分支、更多tokens、额外adapter或加深Temporal来凑参数。

参数预算在75-step launch前封存；随后已完成的正式v4轨迹至step2400始终沿用
完全相同schema。

当前真实model机械计数为：

```text
visual-state       756,992
VL Meta-LoRA       921,600
Action Meta-LoRA 1,253,376
Plan/Revision       37,376
Temporal         1,640,192
Query Decoder    2,191,104
Factor Heads     3,498,432
total           10,299,072
public LoRA      1,287,168
```

factor-head hidden width为411；总量与rank-128 Source-SFT的10,297,344只差
1,728（0.017%）。这是唯一容量校准旋钮。

## 17. 唯一代码路径与退役边界

只保留：

- `scripts/train_as_writer.py`：唯一AS入口；
- `scripts/evaluate_pi05.py`：唯一PI05 rollout入口；
- `configs/pi05_as_writer_action_forecast_v4.json`：唯一活动AS配置；
- `src/ember/writer/action_forecast.py`：PI05 forecast与Meta-LoRA owner；
- `src/ember/writer/visual_state.py`：单一职责visual-state owner；
- `src/ember/writer/temporal.py`：Plan/Revision/Temporal/Query owner；
- `src/ember/writer/model.py`：完整public LoRA schema和Writer组合。

v4原位替换并删除不兼容的Belief-v3配置与checkpoint schema；不得再新增
v5/new/experimental配置、第二runner或并行checkpoint schema。

退役：

- 28-slot content-only DETR state decoder；
- 冻结随机visual-state补丁及其测试/配置字段；
- 旧Action-Memory活动代码和只验证旧内部结构的测试；
- cumulative transition；
- per-frame independent AbsoluteReader；
- Plan/Revision interleaved tokens；
- old/new absolute-action Revision event path；
- additive stability branch；
- temporal mean removal；
- global/innovation双分支；
- order/contrast loss；
- position-gated Meta-LoRA；
- stride10。

有效工程修复可以保留：

- final frame-microbatch按最后一帧repeat-pad到固定shape后裁掉结果；
- 不改变正式authority的探索性评测provenance兼容；
- 可恢复checkpoint与真实video-condition LoRA缓存。

## 18. 最小实现验证

只做防止无效实验所必需的检查：

1. tokenizer state block精确32 positions，whitespace/digit layout正确；
2. shape和padding；
3. `c_0=0`、非递归、相同pair signed change为零、交换local pair反向；
4. visual-state、两个Meta-LoRA、Plan/Revision、Temporal、Query和factor
   heads都有预期梯度；
5. source base错误解冻为零；
6. fresh public LoRA functionally identity且首步有梯度；
7. public LoRA tensor names/shapes完整匹配；
8. frame/action batch真实OOM profile；
9. optimizer、scheduler、sampler、rank/worker RNG、flow noise、video cursor与
   checkpoint exact-resume；
10. information wall无泄漏。

不做全仓仪式性校验，不为短探索建立平行测试框架。

## 19. 75-step视频特异性闭环（已完成）

新架构从fresh identity训练到75 optimizer steps。加载成本较高，不评测更短
checkpoint。

首先做低成本内部诊断。对多个预先固定的train/validation tasks和videos构造：

- correct normal-order video；
- 同帧集合reversed；
- 同帧集合固定seed shuffled；
- same-task different-demo video；
- cross-suite wrong video。

保持正确language、policy/action query、flow noise、checkpoint和paired RNG
不变。逐层测量：

```text
frozen image tokens
-> h_0 / anchor change / local change
-> 32 visual-state embeddings
-> action forecasts
-> time-aligned residuals
-> Plan / Revision
-> Belief
-> Temporal memory
-> Query content Z
-> generated LoRA tensors
-> effective LoRA updates
```

判断要求：

- 差异不能只出现在很上游，最终effective LoRA仍须有清楚、跨多个tasks/videos
  稳定的变化；
- normal-order forecast alignment应在统计上比shuffled更一致；
- reversed/local-direction变化应沿链路保留；
- direct换视频必须改变最终LoRA；
- 不把任务/场景稳定成分本身视为失败；
- 不根据随机初始化或单个task宣称通过。

内部通过后可做paired rollout：

- rollout关注correct/wrong/reversed/shuffled之间的相对差异；
- 此阶段不要求correct arm达到最终绝对性能；
- 同task/init/policy/video seeds严格配对。

若未通过，沿最早发生坍缩或衰减的层级分析、修改同一canonical架构、fresh训练
新的75-step run并重复。不得用contrast loss挽救。若第一性原理分析后确实没有
可修方案，停止并向owner汇报，而不是堆任意模块。

本gate已经由v4通过，保留本节作为架构接受证据。此前产生的80-episode rollout
快筛由于方差和batch-composition敏感性过大，现已退役：不再运行，也不再用于
checkpoint排序或行为结论。后续行为判断只使用固定400-episode panel。

## 20. 正式AS训练与现有observed-best

v4正式run已从fresh identity训练到step 2400，且训练现已结束：

- checkpoint可每75 steps密集保存；
- frame stride固定5；
- 已采用真实吞吐和显存profile选定的batch及frame-microbatch；
- 只用GPU 0、1、2、3，即使这些卡已有进程也可按owner授权共享；
- 4–7绝不触碰。

不再续训，不使用80-episode快筛。根据完整functional轨迹、已有固定400结果和
训练时间覆盖，从现有75-step checkpoints中选择有代表性的候选，直接在完全相同
的固定8-task×50 panel上评测。候选必须覆盖已有最佳附近、后续局部functional
低点、训练后段和终点，避免只因loss或单个低点漏掉行为反弹。最终best只由paired
closed-loop success决定；functional loss只用于安排评测顺序，不能替代rollout。

本轮寻找的是截至step 2400的observed-best，不再用追加训练强求明显峰后下降，
也不据此宣称v4的无限训练ceiling。

性能参考：

- 此前Action-Forecast架构约125/400为主要参考；
- 四卡rank-128 Source-SFT observed-best是108/400；
- 旧八卡122/400不是必须超过的硬门槛，但超过更好。

无论绝对性能是否达到上述参考，最终被选best都必须重新通过内部和实际视频
特异性检查。本轮只分析未达到参考时最可能的表示与优化原因，不修改架构或重训。

### 20.1 高层任务逻辑与demo-specific低层变化

EMBER的目标不是复现teacher的逐点轨迹，而是从一条action-hidden视频提取可迁移
的任务逻辑，例如：

```text
识别目标和相关物体
-> 靠近目标
-> 抓取或操作
-> 搬运/对齐
-> 放置或完成终态
```

AS训练中的teacher video与action episode只保证来自同一任务，并非同一episode
配对。因此，同任务示范之间共享的物体角色、阶段顺序、因果关系和完成条件，才是
与监督action稳定一致的信号。某条示范独有的速度、逐点轨迹、抓取角度、细微视角
变化，以及不准确的visual-state或Action Expert forecast，则可能是与当前action
监督不匹配的干扰。

v4切断task-latent捷径并更忠实地保留visual-state、forecast和Revision差异，这是
必要的机制修复，但它可能同时放大两类信息：

- 有用的任务阶段与动作逻辑；
- 无用或不可靠的demo-specific低层变化。

因此“LoRA随视频明显变化”只证明Writer使用了视频，不证明变化对执行有益。当前
待验证假设是：v4在精确识别、接近和抓取物体的任务上，可能对后一类变化过敏，
从而使随机正确teacher造成策略摇摆。这个假设不是既定结论，本轮只通过内部量与
paired rollout量化，不直接修改架构，也不通过额外loss人为压制差异。

期望的特异性层级是：

1. `correct`与`same-task other correct teacher`可以产生差异，但内部LoRA差异
   应明显小于wrong-video和顺序破坏；两者的成功率应接近，paired episode churn
   应较小。
2. `cross-suite wrong`应改变任务语义，`shuffled/reversed`应破坏动作阶段和有向
   转移；它们应产生更大的内部变化，并在多个任务的paired rollout中稳定伤害
   表现。
3. 不要求同任务不同示范生成完全相同LoRA，因为不同合法初态、布局和策略仍可能
   提供有用调整；判断重点是变化的层级、稳定性和行为后果。

若内部差异很大但各条件rollout没有稳定层级，结论应是特异性主要属于无效变化；
若same-task other保持稳定而wrong/shuffled/reversed明显退化，才说明Writer更接近
提取了可执行的高层任务逻辑。

### 20.2 已观测结果与本轮结论

正式v4轨迹已在step2400终止。固定400-episode panel上的现有checkpoint曲线为：

| step | 675 | 825 | 900 | 1200 | 1275 | 1500 | 1875 | 2100 | 2400 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| success/400 | 100 | **109** | 82 | 96 | 94 | 92 | 90 | 90 | 89 |

因此本轮observed-best是step825，不据此声称无限训练上限。它比四卡rank-128
Source-SFT `108/400`高1，但比旧Action-Forecast observed-best `125/400`低16。
相对旧版的16次净损失全部来自两个object validation tasks：
`45→38`与`20→11`；其余六个task净变化为0。

step825的完整固定400 paired arms为：

| condition | correct | same-task other | cross-suite wrong | shuffled | reversed |
|---|---:|---:|---:|---:|---:|
| success/400 | 109 | 104 | 99 | **148** | 126 |
| 与correct的paired churn | — | 53 | 70 | 87 | 77 |
| correct-only / condition-only | — | 29 / 24 | 40 / 30 | 24 / 63 | 30 / 47 |

- same-task other只净降5且是五个条件中影响最小，说明Writer部分保留了同任务
  高层共性；但53/400条确定性翻转仍表明demo-specific变化不可忽略。
- wrong只净降10，且逐任务方向不一致，未形成稳定的语义伤害。
- shuffled净增39，exact McNemar `p=3.48e-5`；reversed净增17并沿改善方向。
  提升主要来自object tasks：shuffled使两个object task合计`49→82`，
  reversed使其`49→68`。

内部16-reference检查的effective-LoRA相对L2中位数遵循预期层级：
same-task other `0.0955` < shuffled `0.2598` < reversed `0.3255` <
wrong `0.8762`。所以v4确实读取并传播了不同视频与顺序；失败发生在“如何把这些
差异转成有益策略”，而不是又退化成公共LoRA。shuffled/reversed的effective
LoRA RMS相对correct约为`0.988/0.964`，也排除其收益只是把adapter缩回identity。

当前最符合证据的假设是：v4同时保留了高层任务阶段和不可靠的forecast/
demo-specific低层变化，其中coherent normal-order temporal信号在object精确
接近与抓取任务上被映射成了有害策略更新；打乱或倒序改变该分量后，稳定的
task/object信息反而重新占优。这一解释由任务集中性和paired结果支持，但尚未被
因果证明。并且shuffled/reversed还会改变anchor和local-transition构造，因此
不能把它们解释成纯粹的“删除时序”。

结论是：same-task鲁棒性部分成立，但实际视频特异性硬门失败，因为正确时序没有
获得行为上的优待。按owner要求，本轮不据此改架构，不再训练，也不进入RL；保留
完整证据后停下，供下一阶段及外部专家复核。

### 20.3 固定首帧shuffle反事实

owner随后只授权一个最小anchor归因实验。新增`shuffled_keep_first`条件复用
full-shuffle完全相同的确定性permutation，但把原始frame 0移回位置0，并保持
其余所有帧在full-shuffle中的相对次序；frame indices、teacher、language、
init、flow noise、policy seeds和其他合同字段均不变。

step825同一固定400 panel结果为：

```text
correct                    109/400
shuffled_keep_first        136/400
full shuffled              148/400
```

correct与keep-first的paired
both/correct-only/keep-only/both-fail为`91/18/45/246`，净增27，
exact McNemar `p=8.98e-4`。因此即使原始首帧严格保留，破坏后续顺序仍显著优于
correct；“随机阶段帧被选作anchor”不是主要解释。

full-shuffle与keep-first为`116/32/20/232`，full净高12，`p=0.126`。差异几乎
全部集中在Object-3：`37→26`，paired full-only/keep-only=`14/3`、
`p=0.0127`。两项干预可能非线性交互，不能把39严格做因果加法分解；但恢复
原始anchor只把148降到136，而固定anchor后仍相对correct净增27。因此anchor
变化可能额外帮助特定object任务，却不是异常的必要条件或主要解释。

14个full-shuffle本来就把frame 0排在首位的episodes产生了与keep-first完全相同
的LoRA，且14/14行为一致，验证该反事实实现没有引入额外随机变化。当前证据进一
步把主嫌疑收窄到正确时序及其local-transition/forecast解释，而不是首帧anchor
本身；但仍不区分“相邻transition有害”和“Temporal顺序聚合有害”。

## 21. Cold-start RL Writer（本轮暂停）

本轮完成现有AS checkpoint选择和特异性分析后立即停止，不进入RL。以下只保留
owner未来重新授权时的科学合同，不是当前执行项。

RL Writer使用同一最终架构和public LoRA schema，但不是从完整AS best继续：

1. 从规定fresh identity启动独立短AS cold start；
2. 在24个train tasks上持续official random-reset reward screen；
3. 每个task至少获得一次official success；
4. 达到24-task coverage后永久关闭action入口；
5. 转为pure official reward更新；
6. 保存env/policy/worker RNG、seed schedule、interaction cursor和exact-resume
   state；
7. 在validation上寻找observed-best及明显、复测稳健、多task共同贡献的峰后
   下降。

不得自动继续cold-start RL、final-32、test task-local RL、joint target-action
oracle或ViVLA。

## 22. 效率、存储与安全

- GPU launch前实时检查GPU0–3进程/显存/温度，4–7不查询后使用；
- 不杀、不暂停、不重置他人进程；
- `/data/ymdai` 500GB硬cap，启动新run前计算当前使用量与checkpoint峰值；
- 复用现有source base、dataset、tokenizer和video/cache；
- LoRA生成按 `(language task, video task, demo_id, condition/order transform)`
  去重，不按init state重复生成；
- LoRA生成batch与rollout并发解耦并分别profile；
- 同一评测进程尽量复用已加载policy/Writer，避免无谓卸载重载；
- 正式训练前使用`formal-training-launch`核验一次launch contract；
- meaningful里程碑更新`task_plan.md`、`findings.md`、`progress.md`，但文档不得
  阻塞GPU关键路径。
