# EMBER Writer v5.3：Task-Grounded Visual-Transition Procedure

> 2026-07-28状态：本设计及其prototype已被
> `docs/action_forecast_writer_v6_design.md`吸收并替代，只作直接provenance；
> 不再作为待独立训练的活动架构。

## 1. 决策与实验位置

v5.3 曾是 owner 于 2026-07-28 指定的下一条 fresh Writer 架构实验。
实时 v5.2 checkpoint 和评测进度仍以现场 artifact 为准；v5.2 先沿原 recipe
exact-resume 测清右端上限，v5.3 同时实现，待 GPU4–7 空闲后独立 profile 和
fresh 训练。

v5.3 不采用 task-complete optimizer update。它完整保留当前 v5.2 训练范式：

```text
4 symmetric DDP ranks
× one task / one teacher video / one generated LoRA per rank update
× B_a independent same-task action queries
→ one positive functional action loss
→ one AdamW update
```

它不从 v5.2 Writer checkpoint resume，不添加 contrast、order、margin 或
shuffled/reversed supervision。task-complete update 只保留为未来独立 recipe
对照，不混入本架构。

## 2. 不变合同

- Writer 输入仍严格为 `exact task language + exactly one raw action-hidden
  teacher video`。
- Writer 不读取 teacher action、proprio/state、reward、terminal、task ID、
  filename、hidden normalization 或 policy outcome。
- teacher action 只进入 frozen source policy 下的 AS functional loss。
- source base、normalization、24/8/8 split、frame stride=5、单 agentview、
  preprocessing、320 routing identities 和 public rank-16 complete task LoRA
  全部不变。
- Semantic Core、Causal Procedure encoder、slot-normalized compiler 和唯一
  canonical AS/evaluation runner 不分叉。
- 只使用物理 GPU4–7。

## 3. v5.2 Procedure 的剩余表示瓶颈

v5.2 的每帧 Action-Expert probe 为：

```text
frame + exact task language
+ fixed persistent Gaussian suffix at timestep=1
→ frozen π0.5 multimodal / Action-Expert forward
→ mean over 50 final suffix hidden tokens
→ A_f ∈ R^256
```

`A_f` 不是 teacher action，而是固定 probe 下的 Action-Expert latent。它让
Procedure 间接看到当前画面和任务，但 50-token mean 把对象移动、接触变化和
目标关系形成过程压缩为单向量。

与此同时，v5.2 已为每帧生成 task-token aligned patch evidence：

```text
G_f ∈ R^(L×256)
```

同一 token 位置在各帧对应同一任务语义。v5.3 把预算从已证明不是主要瓶颈的
factor decoder 移到 Procedure 上游，只增加 `G_f` 的相邻视觉变化路径。

## 4. 唯一架构改动

### 4.1 视觉 transition

对每个视频在该 arm 的实际输入顺序内计算：

```text
D_0 = 0
D_f = G_f - G_(f-1),  f > 0
```

`D_f` 保持 `[L,256]`。padding task tokens 和 padding frames 严格为零。
shuffled/reversed 必须先变换输入帧，再按变换后的相邻关系重算 `D_f`；禁止先按
正确顺序计算 transition 后再重排。

首版不加入 optical flow、显式几何、长程 pair matching、绝对 patch
Procedure 旁路或第二套视觉编码器。

### 4.2 Action-Expert probe 查询视觉变化

使用八头、每头32维的 cross-attention：

```text
Q_f = Wq(RMSNorm(A_f))             # [1,256]
K_f = Wk(RMSNorm(D_f))             # [L,256]
V_f = D_f                          # raw value，无 learned Wv
R_f = Wo(Attention(Q_f,K_f,V_f))   # [1,256]
Z_f = A_f + R_f
```

attention 只允许 valid task tokens。`Z_f` 作为原有 Causal Procedure encoder
的逐帧输入：

```text
P_1...P_F =
ExistingCausalProcedureEncoder(Z_1...Z_F, frame_positions, valid_frames)
```

职责分离为：

```text
Semantic Core:
  task-relevant objects / relations / spatial content
  invariant to permutation of the same frame set

Procedure:
  Action-Expert operation hypothesis
  + task-grounded adjacent visual change
  causally ordered over the actual input sequence
```

Procedure 不直接读取 absolute patch content，避免与 Core 重复、按静态外观识别
视频或重建 v4 shuffled 的顺序无关旁路。

## 5. 参数预算

transition fusion 新增：

```text
Wq/Wk/Wo: 3 × 256 × 256 = 196,608
two RMSNorm: 2 × 256       =     512
total                       = 197,120
```

八个 factor heads 的 hidden width 从216降至192：

```text
saved = 204,288
v5.2 Writer = 10,237,704
v5.3 Writer = 10,237,704 - 204,288 + 197,120
            = 10,230,536
```

必须由真实 module enumeration 再核验，且始终低于 rank-128 Source-SFT
`10,297,344` 参数上限。public LoRA tensor count、targets、rank、identity
template 和 compiler 均不变。

## 6. 初始化、训练与 profile

- v5.3 使用新的 config/launch/checkpoint/evaluation schema，旧 Writer
  checkpoint 必须 fail closed。
- transition Q/K/O 和 norm 使用正常 fresh 初始化；最终 factor-head projection
  继续 zero-init，因此生成 public LoRA 在 step0 仍与 source policy 完全
  functionally identity。
- 只训练 normal-order positive AS。
- 原 v5.2 的 optimizer ownership、task rotation、one-video schedule、
  action-query sampling、AdamW 与 scheduler 形式全部保留。
- 最长105帧真实视频先验证 shape、identity、gradient、freeze、B20、
  B21/上界、step1→3 exact-resume 和真实 transition 非零；正式 batch 和首段
  wall-clock 只由该 profile 决定。
- 压力 profile 暂用已封存的 `teacher_video_seed=172`，使四个 rank 的首步真实
  样本包含 stride-5 后105帧的最长 train video；profile 封存后，正式 fresh
  训练恢复 canonical `teacher_video_seed=20260722`。

## 7. 判定

v5.3 与充分训练的 v5.2 observed-best 在相同无放回 fixed-400 contract 下比较：

```text
absolute correct performance
same-task-other robustness
correct > cross-suite wrong
correct > shuffled
correct > reversed
task breadth and multi-task contribution
Core / Procedure / effective-LoRA / policy-action transmission
```

内部检查还必须证明：

- same frame-set 的 shuffled/reversed 不改变 permutation-invariant Core；
- transition 按各 arm 的实际顺序重算；
- Procedure 的 order 差异能够穿过 compiler 到 effective LoRA 和 action；
- 不能以仅有内部数值差异代替闭环正确方向；
- 不能以 v4 式 shuffled/reversed 更高的绝对分数冒充成功。

v5.3 完成后，才依据 v5.2/v5.3 的 absolute、task migration、五臂和内部证据，
决定是否尝试 task-complete recipe、cold-start RL 或下一架构。
