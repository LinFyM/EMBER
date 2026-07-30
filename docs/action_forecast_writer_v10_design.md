# EMBER Writer v10：Evidence-Preserving Dual-Stream Writer

状态（2026-07-30）：owner 已确认采用本设计。v10 已原位替换尚未提交、未正式
训练的 v9 strict-binding 草案，成为下一条唯一 canonical Writer 路径；真实
参数、B20最长视频profile和exact-resume均已通过，正式训练尚未启动。v8 及
更早版本只由 Git、checkpoint 和实验文档保存 provenance，不保留并行可执行
实现。

## 1. 决策依据

v5.2 的 single-checkpoint best 为：

```text
correct / same / wrong / shuffled / reversed
132     / 138  / 74    / 82       / 83
```

它证明稳定 task grounding、逐帧 Action-Expert latent、causal Procedure 和
强 Procedure-to-compiler readout 可以同时产生 absolute 与视频特异性。

v6 task-complete fast-decay 的 single-checkpoint best 为：

```text
correct / same / wrong / shuffled / reversed
143     / 135  / 125   / 128      / 129
```

它证明在保留 Action residual 的同时加入 visual transition 可以提高
absolute，但当前训练与融合只保留较弱的控制臂 margin。

v7/v8 改为由 8 个 Action probes 与视觉变化直接生成一个 effect-dominant
event 后，best correct 分别只有 `120/125`。内部证据为：

```text
v7 joint 8×L pair entropy / uniform entropy ≈ 99.96%
v8 EventRead entropy / uniform entropy       ≈ 99.67%
v8 vary Action, fixed Effect event L2        ≈ 8–10%
v8 fixed Action, vary Effect event L2        ≈ 147–300%
```

因此 v7/v8 的主要失效不是缺少 latent difference，而是：

1. Action 信息在 joint pooling 或 EventRead 中被平均、压缩；
2. Effect 取代了已被 v5.2/v6 证明有用的 Action path；
3. v8 的 Core gate 不能像 v5.2 compiler 一样把小而正确的 Procedure
   difference 有效编译到 LoRA。

更根本地，Action Expert probe 不是 teacher 实际执行的 action。它只是冻结
π0.5 在当前 task/frame 下的 action hypothesis；相邻视觉差分则是未知 teacher
action 造成的观察结果。信息墙内不存在逐 interval 的真实 action-effect 配对
标签，因此不得再用 strict multiplication 或 joint pair normalization 声称
二者已被识别为局部因果对。

v10 的原则为：

```text
保留已证明有效的完整证据路径
→ Action hypothesis 与 observed Effect 分流
→ 在 causal temporal model 中学习跨 interval 关系
→ Procedure 在 Core 语境下成为 LoRA 的必要内容
```

## 2. 信息墙与输出合同

Writer 输入严格为：

```text
exact task language + exactly one raw action-hidden teacher video
```

Writer 不得读取 teacher action、state/proprio、reward、terminal、task ID、
filename、episode identity、hidden normalization 或 policy outcome。teacher
action 只进入 frozen source policy 下的 AS functional loss。

输出保持一套完整 task-specific public rank-16 LoRA，覆盖 sealed 320 routing
slots、原 public targets 和 identity template。frame stride 固定为 5；只使用
物理 GPU4–7。

## 3. 稳定 task-grounded semantic trajectory

设有效 task token 数为 `L`，采样视频帧数为 `F`。

### 3.1 Text-only semantic axis

exact task tokens 单独经过 frozen Gemma + rank-4 Text Meta-LoRA 和共享
bias-free `2048→256` projection：

```text
Q_text ∈ R^(L×256)
```

`Q_text` 对所有视频帧保持不变，为跨帧 patch read 和 Semantic Set 提供稳定
task-token 坐标。

### 3.2 Multimodal evidence

每帧只执行一次带 VL/Action Meta-LoRA 的 multimodal/Action-Expert forward，
得到：

```text
H_task[f]  ∈ R^(L×2048)
H_patch[f] ∈ R^(256×2048)
A_raw[f]   ∈ R^(8×1024)
```

共享 projection 得到：

```text
M_f = Proj(H_task[f])  ∈ R^(L×256)
P_f = Proj(H_patch[f]) ∈ R^(256×256)
```

### 3.3 Task-queried patch grounding

```text
G_f = Wo Attention(
    Wq RMSNorm(Q_text),
    Wk RMSNorm(P_f),
    V=P_f,
) ∈ R^(L×256)
```

Q/K/O 为 bias-free、8 heads × 32；V 没有 learned projection。完整语义轨迹为：

```text
X_f = M_f + G_f
```

`M_f` 保存图文交互后的任务组合语义，`G_f` 保存显式 task-aligned patch
content。Core 读取 `X_f`；Effect 只对 `G_f` 做差，避免 multimodal language
context jitter 混入环境变化。

## 4. Semantic Core

先建立不可丢失的 frame mean：

```text
M_l = Mean_f(X_f,l)
```

再由稳定 `Q_text,l` 读取 frame-centered residual：

```text
R_l = Attention_f(
    Q_text,l,
    K=X_:,l,
    V=X_:,l - M_l,
)

U_l = Wm(M_l) + R_l
```

该集合聚合对 frame permutation 严格不变；若 frame attention 均匀，
centered residual 精确为零。`U` 再进入 2 个 width-256、8-head、
FFN-1024 的 bidirectional task-token blocks，RoPE 只使用 task-token
ordinal。输出：

```text
C ∈ R^(L×256)
```

保留 `L` 个 Core tokens，使容量随任务描述中的对象、关系和子目标自然变化。
语言规定的子目标顺序由 task-token ordinal 保存；Core 不读取 frame order。

## 5. Action-hypothesis stream

每帧保留 8 个 Action Expert Writer probes，覆盖 action horizon 的原生位置：

```text
[0, 7, 14, 21, 28, 35, 42, 49]
```

它们来自一次 Action-Expert forward，不经过 `action_out`，不是 teacher
actions。共享 bias-free `1024→256` projection 后：

```text
A_f,k ∈ R^256
```

建立稳定基座与位置保真的 phase residual：

```text
A_mean[f] = Mean_k(A_f,k)
δA_f,k = RMSNorm(A_f,k) - Mean_j(RMSNorm(A_f,j))

A_phase[f] = W_phase Concat_k(δA_f,k)
W_phase: 2048→256, bias-free

A_star[f] = A_mean[f] + A_phase[f]
```

`W_phase` 使用 zero-output 或极小初始化，使训练起点保留 mean fallback，同时
允许 8 个固定 horizon positions 学习不同阶段作用。8 个 probes 不进入
`8F` 时序，也不经 softmax 互相竞争。

## 6. Visual-effect stream

只在当前 arm 的实际输入顺序内计算 forward task-grounded patch change：

```text
D_f,l = G_(f+1),l - G_f,l
```

shuffled/reversed 必须先变换原始帧，再重新编码并计算 `D_f`。

Action summary 只用于选择相关变化，不与 Effect 强制相乘：

```text
V_f = Wo Attention(
    Wq RMSNorm(A_star[f]),
    Wk RMSNorm(D_f),
    V=D_f,
) ∈ R^256
```

8 heads × 32，所有 projection bias-free。性质：

```text
D_f = 0 → V_f = 0
```

但 `A_star[f]` 仍保留；Effect 不可靠时不得同时删除已证明有用的 Action path。
该 attention 表示“在当前 action hypothesis 下哪些观察变化相关”，不声称
probe 就是造成该变化的 teacher action。

## 7. Interleaved Causal Procedure

不提前执行 `A_star + V`，而是形成：

```text
A_0, V_0, A_1, V_1, ..., A_(F-2), V_(F-2), A_(F-1)
```

长度为 `2F-1`。Action 与 Effect 使用独立 input norm/projection。若 sampled
frame ordinal 为 `p_f`：

```text
pos(A_f) = 2 p_f
pos(V_f) = p_f + p_(f+1)
```

两层 width-256、8-head、FFN-1024 的 global causal pre-norm Transformer
在该序列上建模。它可以在不同时间跨度上学习：

```text
action hypothesis → observed change → next action hypothesis → next change
```

而不要求局部一一配对。Core 不参与 Procedure encoder。输出：

```text
P ∈ R^((2F-1)×256)
```

## 8. Procedure-gated Core compiler

320 routing identities 仍只进入 Q/K，不作为 value content。

每个 slot 先读取 Core：

```text
C_s = CoreRead(Q=R_s, K=C, V=C)
```

再用 Core-conditioned query 读取 Procedure：

```text
P_s = ProcedureRead(
    Q=R_s + RMSNorm(C_s),
    K=P,
    V=center_by_stream_and_time(P),
)
```

Action/Effect Procedure values 分别按时间中心化，避免固定 token-type mean
成为旁路；keys 保留完整 Procedure content 和 causal positions。

融合使用高容量、bias-free Procedure-gated Core modulation：

```text
h_s = GELU(W1 RMSNorm(P_s))       # 256→512
[γ_s, β_s] = W2(h_s)             # 512→512

Y_s = P_s + β_s
      + tanh(γ_s) ⊙ Wc RMSNorm(C_s)
```

所有线性层无 bias，因此：

```text
P_s = 0 → γ_s = β_s = Y_s = 0
```

Procedure 通过 `P_s + β_s` 直接提供 LoRA content，并通过 `γ_s` 决定哪些
Core content 可以辅助该 slot。Core 可全秩参与，但不能独自生成 public LoRA。

`Y_s` 再经过一个 content-only post-fusion slot block，随后进入原 8 个
factor heads：

```text
256 → 256 → target width
GELU
final Linear bias-free, zero-init
```

factor final zero 保证 fresh step0 public LoRA 与 identity template 逐 tensor
一致。

## 9. 参数预算

以 v6 的 `10,775,296` 为母体机械估算：

```text
8-probe phase mixer                   +524,288
compiler 256→512→(γ,β) expansion      +327,680
interleaved Procedure                  parameter-neutral
--------------------------------------------------------
v10 Writer exact enumeration        11,627,520
```

相对 corrected rank-128 Source-SFT `10,297,344` 约多 `12.9%`。上述总数已由
实际 module enumeration 核验（手算初稿少计一个 256 参数 RMSNorm）。这是软预算；
新增参数只用于 action-phase 保真和 Core/Procedure 融合，不扩大已排除为首要
瓶颈的 factor decoder。

## 10. 删除与禁止

v10 删除：

- joint `8×L` Action–Effect softmax；
- strict `Effect × Action` value；
- Procedure-only EventRead；
- v8 的简单 `Procedure × (1 + Core gate)`；
- multimodal frame-mean query 对 text-only semantic axis 的替代。

首版不加入：

- optical flow、3D reconstruction 或额外视觉 encoder；
- null token；
- Action-only public adapter branch；
- Core-only public adapter branch；
- task ID、task bank、MoE 或 shared trainable execution adapter；
- shuffled/reversed/order/contrast supervision；
- checkpoint fusion或ensemble。

## 11. 实现、训练与评测合同

v10 使用新的 fail-closed config/launch/checkpoint/eval schema，从 functional
identity fresh 训练，不加载 v8/v9 Writer 权重。

实现后先完成：

1. shape、mask、Core permutation、causal order；
2. `D=0→V=0`、`Procedure=0→LoRA delta=0`；
3. identity、freeze、完整 rank-16 target；
4. Text/VL/Action Meta-LoRA、phase mixer、Effect、Core、Procedure、
   compiler 和 factor 的真实 gradient reachability；
5. GPU4–7 最长105-frame视频 B20连续3个完整macro；
6. 完整macro边界 exact-resume。

正式训练沿当前 task-complete 方法：

```text
4 ranks
× each rank 6 distinct tasks, long-first
× one video/LoRA/task
× B20 independent action queries/task
→ mean within task
→ equal mean over all 24 tasks
→ one synchronization and one AdamW update/macro
```

从 step0 fresh 训练约两小时，初始目标为400 macro、每25 macro保存checkpoint。
训练完成后，对多个 single checkpoints 做固定、无放回、paired correct400；
只对 observed-best 做：

```text
correct / same-task-other / wrong / shuffled / reversed
```

以及内部：

```text
Core
Action stream
Effect stream
Procedure
Procedure slots
effective LoRA
policy action
```

成功目标为：

```text
single-checkpoint correct400 >= 150
correct >= corrected Source-SFT best + 30
same-task 与 correct 同档
correct 显著优于 wrong / shuffled / reversed
增益由多个tasks共同贡献
独立RNG与video permutation复测成立
```
