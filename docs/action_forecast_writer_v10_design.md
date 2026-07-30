# EMBER Writer v10：Evidence-Preserving Dual-Stream Writer

状态（2026-07-30）：owner 已确认采用本设计。v10 已原位替换 v9
strict-binding 草案，成为当前唯一 canonical Writer 路径；真实参数、
B20最长视频profile、exact-resume、identity fresh macro0→400正式训练、
12点paired correct400、observed-best五臂和内部反事实均已完成。结果为
absolute失败、视频语义/顺序行为门通过；owner要求完成v10后暂停讨论，当前
不续训、不改架构，也不启动后续候选。v8及更早版本只由Git、checkpoint和实验
文档保存 provenance，不保留并行可执行实现。

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

## 12. 正式结果与机制结论

正式run在GPU4–7按上述合同从identity fresh完成macro0→400：

```text
optimizer updates              400
one-video LoRA conditions    9,600
action queries             192,000
training wall             7,832.833 s
```

训练、gradient、24-task覆盖和long-first调度全程finite且完整；每个train task
访问400次，并按无放回cycle覆盖teacher videos。12个single checkpoints的
paired、每task 50 videos无放回correct400为：

```text
macro       25  50  75 100 150 200 225 250 300 325 350 400
correct     95 103  84  89  82  90  96  96  89  96  97  91
```

observed-best是macro50=`103/400`，低于corrected Source-SFT best=`109`，
距离absolute门`150`为47。它只显著高于macro75；其余checkpoint主要表现为
success集合交换而不是共同上升。macro50逐task为：

```text
Long-1 / Long-2 / Goal-3 / Goal-6 / Object-1 / Object-3 / Spatial-1 / Spatial-3
5      / 0      / 0      / 42     / 44       / 8        / 2         / 2
```

Goal-6与Object-1贡献`86/103`个成功；Long-2与Goal-3始终没有形成能力，
Object-3从macro25的15降到macro50的8并在macro100后归零。训练loss的25-step
均值从`1..25=.13424`持续降到`376..400=.09651`，但online validation
functional loss在macro50达到全程最低`.131935`，其后只在
`.13231–.13824`间波动。故本结果不是“右端仍上涨或只差更多updates”，而是
训练目标继续拟合、held-task闭环能力已停止泛化。

macro50正式五臂为：

```text
correct / same-task-other / wrong / shuffled / reversed
103     / 94              / 75    / 67       / 43
```

相对correct的paired switches与双侧exact McNemar为：

```text
same       26 / 17   p=.22205
wrong      52 / 24   p=.001762
shuffled   51 / 15   p=1.01e-5
reversed   68 /  8   p=5.63e-13
```

same与correct同档；wrong、shuffled、reversed三项门均通过且各有6个正向
contributing tasks。因此v10的失败不是视频完全未进入LoRA或v4式无序旁路，
而是correct adapter本身absolute与breadth不足。

8个validation tasks各取1个reference的内部中位relative-L2为：

```text
condition   Core   Procedure  ProcSlots  eff.LoRA  policy action
same       .0437     .4050      .5665     .2523       .0970
wrong      .2029     .4087     1.2923     .8832       .1674
shuffled   .0000     .0873     1.0922     .7391       .3481
reversed   .0042     .0836     1.3456     .8718       .1922
```

Core对相同frame set的shuffle严格不变、对reverse近似不变，职责边界正确。
固定correct Core只替换Procedure时几乎逐项复现全部LoRA/action差异；固定
correct Procedure只替换Core时effective-LoRA中位差异仅
`same/wrong/shuffled/reversed=.0030/.0116/.0000/.0016`。这只说明同一
task下的视频依赖由Procedure控制；不能据此推断跨task的静态Core语义无用。
将Procedure精确置零时raw/effective LoRA均严格回到identity。

Action/Effect反事实揭示了更具体的失衡。固定correct Effect只改变Action时，
shuffled/reversed effective-LoRA差异为`.6299/.8659`；固定correct Action只
改变Effect时仅`.0808/.1004`。Visual-Effect attention熵为理论均匀熵的
约`99.86%`（shuffled约`99.93%`），没有形成强task-token effect选择。v10
因而主要依赖随frame变化的frozen-policy Action hypotheses产生顺序差异，
而不是由Action和observed Effect共同形成稳定教学事件。

compiler进一步把该方差高增益化：correct的Procedure slots RMS仅`.0145`，
而Procedure生成的gated-Core RMS为`.1781`，平均比值约`14.39`；
shuffled比值约`20.53`。modulation在读取Procedure slots前执行RMSNorm，
所以“非零但很小”的Procedure证据不会自然减小adapter增益。与v5.2相比，
same-task换正确视频时Procedure/effective-LoRA/action中位差异从
`.0126/.1345/.0253`扩大到`.4050/.2523/.0970`。这给出了当前最直接的
失败解释：v10同时放大了有用顺序信号和同task示范间的非等价方差，使每次
macro抽到的新video产生更不一致的优化方向；强控制臂margin并未转化为更强
correct policy。

结论是v10在同一task-complete fast-decay合同下相对v6 best
`143→103`的40点退步主要属于架构负结果，而不是评测、调度、OOM或训练未结束。
它证伪了“只要完整保留Action并增强Procedure→Core compiler，就会同时提高
absolute与特异性”的假设。该结论不自动采用任何下一架构；按owner要求在此
暂停，等待共同讨论。

主要证据：

```text
training root:
/data/ymdai/outputs/ember/pi05_as_writer_v10_dualstream_taskcomplete_decay400_dev_r4_b20_seed7_s2400_5fd0a25_20260730

training audit:
/data/ymdai/outputs/ember/pi05_as_writer_v10_training_audit_macro400_5fd0a25_20260730.json
SHA256 6701ec353433203ef89490f0fe6b179eefddaf9e304fd60c9800e204e70ff97f

correct curve:
/data/ymdai/outputs/ember/pi05_as_writer_v10_correct400_curve_paired_5fd0a25_20260730.json
SHA256 6e9d97dcf31afdd7d867e4b3f66646db3efa68df552b625f5db2b3ba05012dfd

five-arm:
/data/ymdai/outputs/ember/pi05_as_writer_v10_single_checkpoint_macro0050_specificity400_paired_5fd0a25_20260730.json
SHA256 a2dbcacdfcfbe4ba2a3a9010c4c28664b2ff8ce4530c532560a24e680474be6b

internal:
/data/ymdai/outputs/ember/pi05_as_writer_v10_single_checkpoint_macro0050_internal_specificity_refs1_5fd0a25_20260730/summary.json
SHA256 df5b0271991b6ff95360b138dfe72dd7ab5daf34cc54383b92688acab539ec9f
```
