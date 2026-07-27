# EMBER Writer v5.2：Task-Queried Patch Grounding

状态（2026-07-27）：已在隔离 worktree 中实现并通过 CPU 合同测试，等待 v5.1
step1400 低学习率稳定段完成后，在 GPU4–7 做真实最长视频 profile。若 v5.1
稳定段仍未同时显著提高 absolute performance 和 reversed 顺序门，则 v5.2
原位取代 v5.1，成为唯一活动 Writer；不得并行保留两条执行路径。

## 1. 直接证据与修改理由

v5.1 原始训练的 observed-best step1400 在固定、无放回、逐 state/video/RNG
配对的 validation 400 五臂上得到：

| arm | success |
|---|---:|
| correct | 127/400 |
| same-task-other | 133/400 |
| cross-suite wrong | 94/400 |
| shuffled | 107/400 |
| reversed | 120/400 |

paired 结果为：

- correct 相对 wrong：`+33`，McNemar `p=1.12e-4`；
- correct 相对 shuffled：`+20`，`p=0.0225`；
- correct 相对 reversed：`+7`，`p=0.477`；
- same-task-other 相对 correct：`+6`，`p=0.504`。

因此 v5.1 已经消除了 v4 的“shuffled 明显优于 correct”漏洞，并证明视频语义
确实进入行为；但 reversed 仍与 correct 等价，而且性能几乎完全来自
`libero_goal:6` 和两个 object tasks。`libero_goal:3` 与两个 spatial tasks
合计仅 `1/150`。对 public LoRA-B 做 `1.25/1.5/1.75/2.0` 全量缩放只会把
总分从 `127` 降至 `124/119/99/82`，也没有解锁这些失败任务。内部有效 LoRA
范数在失败任务上并不小。

这组证据排除了“下游 factor decoder 输出太弱”这一主解释。当前更早的失效点
是上游语义表征：v5.1 Core 只读取 PaliGemma 最终 task-language token hidden，
完全丢弃 256 个 image-position hidden。这个语言瓶颈适合稳定任务语义，却可能
过早压掉物体身份、局部关系和空间位置等决定 LIBERO spatial/goal 成败的细节。

v5.2 因而只做一项可证伪修改：把一小部分下游 factor-head 预算搬到上游，
让 text-only task tokens 主动读取每帧的 patch content。它不增加额外 loss、
adapter、几何编码、state/action 输入或并行 Writer。

## 2. 不变的科学合同

```text
task language
+ exactly one action-hidden teacher video
→ one shared Writer
→ complete task-specific public rank-16 LoRA
→ frozen source policy
```

- Writer 仍不得读取 action、proprio/state、reward、terminal、task ID、
  filename、normalization 或 policy outcome；
- development 仍只用固定 24 train tasks 的 action functional loss；
- validation 只作 checkpoint 选择与机制评测，不产生梯度；
- frame stride 固定为 5；
- Semantic Core 仍对帧集合 permutation invariant；
- Causal Procedure、slot-normalized fusion、320 LoRA slots 和 sealed public
  LoRA targets 完全不变；
- 训练仍是 normal-order positive-only AS，不允许 contrast/order loss。

## 3. 新的 per-frame patch grounding

同一次 frozen PaliGemma multimodal forward 已产生：

```text
H_patch[f]  : [256, 2048]  final image-position hidden
H_task[f]   : [L, 2048]    final task-token hidden
Q_text      : [L, 256]     text-only task-token queries
```

v5.2 复用现有、共享的 bias-free `2048→256` language projection：

```text
P[f] = LanguageProjection(H_patch[f])       # [256,256]
M[f] = LanguageProjection(H_task[f])        # [L,256]
```

每帧执行八头 task-to-patch cross-attention：

```text
Q = Wq(RMSNorm(Q_text))
K = Wk(RMSNorm(P[f]))
V = P[f]                                    # 无 learned V projection
G[f] = Wo(softmax(Q Kᵀ / sqrt(d)) V)         # [L,256]
E[f] = M[f] + G[f]
```

其中 `Wq/Wk/Wo` 都是 bias-free `256→256`。patch 没有显式二维坐标、frame
ordinal 或 geometry token；路由只由 text-only task query 和 patch content
决定，value 只携带投影后的真实 image-position content。无效 task-token 输出
严格置零。

`E[f]` 进入原有 token-aligned frame-set attention 和 bidirectional
language-axis Core。Procedure 仍只读取原来的 per-frame Action-Expert probe；
因此本实验只检验“上游 task-grounded visual semantics 是否是缺失瓶颈”，不把
第二个同时变化混进结论。

## 4. 参数预算

新增 patch grounding：

```text
Wq + Wk + Wo        = 3 × 256 × 256 = 196,608
2 × RMSNorm         = 2 × 256       =     512
合计                                    197,120
```

八个 factor heads 的 hidden width 从 `240` 降到 `216`，减少：

```text
24 × (8 × 256 + Σ output_widths)
= 24 × (2,048 + 6,464)
= 204,288
```

Writer 总参数从 `10,244,872` 变为 `10,237,704`，低于 capacity-matched
Source-SFT 的硬上限 `10,297,344`，余量 `59,640`。因此性能变化不能归因于
扩大 Writer 总参数预算。

所有 factor head 的末层仍为严格零初始化，所以 public LoRA-B 起点保持物理
零、Writer 输出仍与 identity template 精确一致。

## 5. 训练与判定

v5.2 使用新 config/launch/checkpoint/eval schema，从 functional identity
fresh 训练；不得加载 v5.1 Writer 权重或 optimizer。

正式训练前只做：

1. config/shape/parameter-budget/identity/gradient/freeze 检查；
2. GPU4–7 上真实最长 105-frame 视频的 F32/B20 起始 profile，并向显存上限
   探索 action batch；
3. 一次最短真实 resume smoke。

首段按实测吞吐约一小时，不继承 v5.1 的 optimizer-step 坐标。首段完成后先在
fixed correct-video validation 找候选，再对 observed-best 做内部
Core/Procedure/effective-LoRA/action 检查和无放回配对 rollout。

继续第二段或第三段必须同时看到：

- correct absolute 明显超过 `127/400`，并向 `148/400` 推进；
- 增益不再只由一两个旧强任务贡献，尤其 spatial/Goal3 出现真实成功；
- same-task 稳定；
- correct 显著优于 wrong、shuffled 和 reversed；
- 多个后续 checkpoint 的下降远超固定 400 rollout 正常波动。

若 patch grounding 提升 spatial/absolute 却仍没有 reversed 方向优势，下一个
修改点应是 Procedure 的任务条件化或读出，而不是扩大 factor heads、增加 LoRA
幅度或引入 contrast/order loss。
