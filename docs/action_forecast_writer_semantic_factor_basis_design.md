# Semantic Factor-Basis Writer

状态：2026-08-02 post-seal 实现 authority。本文在 Target-Bound 首小时正式
`correct400` 已完成、内部 refs1 审计仍运行时建立；代码不得早于本文改变 canonical
Writer。历史 Target-Bound 由 Git commit `cfd26df`、frozen worktree 和正式 artifact
保存，不保留可执行的并行模型路径。

## 1. 需要解决的不是单一视频读出，而是条件能力共存

Target-Bound 已把 Semantic Core 提前绑定到 38 个真实 policy targets，并让
Action/Effect/Change 保持 private causal channels 到 rank read。它的首小时 paired
`correct400` 曲线为：

```text
macro              50    100    150    200
successes / 400     75    120     90    110
task breadth         7      7      6      7
top-2 contribution 76%   67.5%  77.8%  70.9%
```

`m50→m100` gained/lost states 为 `56/11`，`m100→m150` 为 `11/41`，
`m150→m200` 为 `31/11`；winner `m100→m200` 仍有 `22/32`。因此提前 target
绑定没有把 task 能力从轮换变成共同累积，也没有达到 v6 同期 macro200 的
`133/400`。此前 matched gradient 已显示 Target-Bound 相比 CV 把 Program 梯度份额
提高约 `2.5--4x`，但连续 task-condition 的 factor 梯度 cosine 仍约 `.151`，factor
仍占约 `90--95%` 的更新能量，task-mean/sample sketch energy 也没有改善。最早仍未
解决的接口是共享 factor generator，而不是 evidence、Core 或 private role memory
缺少另一个幅度旋钮。

winner macro100 的8-task refs1内部审计随后闭合了这个定位。remove-A、remove-D和
contextual role-memory reversal均为`8/8` tasks通过预注册门；mean effective-BA
relative L2分别为`.38865/.12374/.06850`，memory reversal的fixed-action mean为
`.02084`。Core-only/Program-only到full BA仍相差`.83840/.58622`，两路都必要。
same/wrong/shuffled/reversed相对correct的BA变化为
`.06325/.35971/.12228/.18813`，fixed-action为
`.12893/.27665/.15314/.24187`；视频与顺序信号已经真实到达函数端。A/E/D reader的
target-centered attention-energy fraction均值约`.137/.094/.129`，说明Core地址也
确实形成了target-specific activation。由此不能再把低absolute主要归因于Program
信号未传出；当前最可信的剩余结构缺口是这些已不同的条件请求仍必须穿过同一套、占
`90--95%`更新能量的factor参数。

最终需要 Writer 从 language+video 推断 task 语义，让相关 task 共享生成能力、不同
task 使用可区分的有效参数子空间，同时保持一套共享模型对 validation/test task 的
组合泛化。24 个 task-ID hard experts 不满足信息墙，也不能泛化；一个 scalar gate、
global scale 或 B-only residual 不能储存近正交的条件更新方向。

## 2. 最小完整改变：Core 选择 factor 语义基底

保留 Target-Bound 的完整前端、mean-backed Core、38-target Core read、private
A/E/D causal Program、38×16 rank reads 和 public rank-16 topology。只替换最后一个
仍承担所有 task 共存职责、却完全不读条件路由的 conventional factor MLP。

对每个真实 target 的 raw Core carrier `C_t∈R^256`，一个在八个 public factor
families、38 targets 和16 rank coordinates之间共享的 router 计算四个语义基底权重：

```text
q_t = W_q RMSNorm(C_t)
k_b = RMSNorm(K_b), b in [0,4)
p_t = softmax(q_t k_b^T / sqrt(256))
alpha_t = 4 p_t                         # mean(alpha)=1
```

`K_b` 是 learned Q/K-only basis keys。它们不进入 factor value，不与 coordinate 相加，
不能凭空产生 LoRA 内容。`alpha` 的和固定为4，使 uniform routing 严格保持每个 hidden
slice 的单位幅度；这不是失败后的全局放大，而是从 step0 就封存的幅度保持参数化，
用于避免 Target-Spectral 式增益坍缩。

每个 factor family 保持总 hidden width256 和原参数量，把单个
`Linear(1024→256)` 明确分成四个独立 value bases：

```text
h_b = GELU(W_b Z[t,r]),                 W_b: 1024 -> 64
h   = concat(alpha_t,b * h_b, b=0..3)  # 4 * 64 = 256
factor = W_out h
```

八个 factor heads 各自拥有 `W_b/W_out`，但共享同一个 router。route 对一个 target 的
16 ranks 相同；因此 task/target 语义可以选择不同 factor 子空间，而 A/B、q/v/action
仍沿同一 semantic routing 保持 coherent 写入。动态 A/E/D 仍全部存在于 `Z` 的 value
中，Core 只选择完整 factor 计算的语义基底，不能成为静态 LoRA bypass。

这是一项 factor generator 的职责重构，不是给旧输出乘一个救火 gate：没有路由时
四个独立64维 value bases仍恰好构成原256维 head；路由决定哪些参数子空间接收和表达
当前条件，而非只改变最终标量大小。

## 3. 参数、初始化和旧失败规避

每个 factor head 的 input/output参数量不变。共享 router 只新增：

```text
Core RMSNorm       256
query projection   256*256 = 65,536
four basis keys    4*256 = 1,024
basis RMSNorm      256
total              67,072
```

相对 Target-Bound `11,092,224`，预计真实 trainable enumeration 为
`11,159,296`；实现后必须由 module enumeration 证实。所有 value projections
bias-free，factor final projection 继续 exact-zero，因此 public step0 保持 template A
和 zero B 的严格 functional identity。source policy 仍为0 trainable parameters。

- 不重演 Target-Spectral：不做正交/rank/spectrum loss，不切断跨层 coherent direction，
  uniform route 不衰减 hidden amplitude。
- 不重演 v10：没有 RMS 后的大倍率 residual gate；`alpha` 有固定均值1且不直接乘
  public LoRA。
- 不重演 Recenter/Prior/Core-Program：Core mean/DC和 raw A/E/D全部保留，不做手工
  static/dynamic分解或 strict bilinear。
- 不恢复 v4 static bypass：Core只能选择 factor value bases，最终 value仍是完整
  `Z=[Core,A,E,D]` 的学习函数；必须继续用 Core-only/Program-only 与顺序反事实审计。
- 不把 post-v5 版本整体判死：本版保留 Target-Bound 已独立证明能增强 Program 梯度的
  target/private-role path，只替换被正式梯度和closed-loop共同指向的最晚共享接口。

## 4. 训练和首小时裁决

第一次正式实验保持 Target-Bound 的 task-query-keyed RAW full24、B20、fast-decay400、
一次 clip/AdamW/scheduler、single video/single LoRA、每25 macro保存。既不同时引入
Latin-Beta estimator，也不恢复 GROUP4/old six-Adam，以隔离 factor conditional
capacity 的作用。使用 fresh incompatible config、checkpoint、trainer和rank schema。

formal前只做聚焦 vertical path：shape/value provenance、route normalization、step0
identity、freeze、staged gradients、最长105-frame B20三macro与fresh/resume。首小时
fresh0→200后在同一paired panel评测50/100/150/200。

正式启动前的live vertical path已在`e87363f`完成：最长真实105-frame视频、4 ranks、
B20连续三macro耗时60.15秒，峰值reserved显存83,508,592,640 bytes；macro1按严格
identity生命周期只有factor可达，macro2起semantic frontend、Core、Program、compiler
和factor五块梯度均finite/nonzero。formal seed的fresh0→1再exact-resume1→3完成到
macro3，合同`0495a071...`保持不变，1,440 queries、72条单视频条件，validation/test
action reads均为0。因此封存B20并授权fresh0→200；这些证据只证明运行合同和主路径，
不预判closed-loop结果。

## 5. 可证伪内部预测

1. route在不同task/target间应出现非零 centered energy，不能长期完全uniform；同时不应
   全task塌到同一basis。
2. 相近语义task允许共享route；不同task的factor basis gradient应比旧整体factor
   gradient更可分且每个basis内部更稳定。
3. factor不再独占且旋转所有条件方向；checkpoint gained/lost应从轮换转为多个task共同
   累积，success-state Jaccard上升。
4. effective LoRA norm、cross-layer coherence和near-rank1高增益不能坍缩；A/E/D移除、
   role-memory reversal必须继续传到BA/action。
5. 若router近uniform或单basis collapse、task factor梯度仍近正交、四点correct400仍低且
   轮换，或absolute通过更static的Core-only路径获得，则“共享soft factor basis能解决条件
   共存”被否定。失败后重审functional estimator/closed-loop manifold，不给本模型追加
   entropy loss、load-balance loss、gate、scale或旁路。

## 6. 正式结果与裁决（2026-08-03）

clean frozen `f5ddfe3`从functional identity完成400个macro、192,000 action queries、
9,600条single-video conditions和16个every25 checkpoints；全部finite、0 clip，
validation/test action reads为0。paired correct400完整曲线为：

```text
macro 50/100/150/200/250/300/350/400
      69/ 91/118/127/117/ 81/126/120
```

single winner仍是macro200=`127`，未超过v5.2-old `132`或v6-fast `143`。第二小时没有
成熟化：200→250 gained/lost=`19/29`，250→300=`16/52`，300→350=`60/15`，
350→400=`20/26`；八点成功集合union=`193`、intersection=`39`，相对single-best的
envelope gap=`66`。因此不做昂贵行为五臂，macro200既有五条件内部分析继续作为本版
机制authority。

结构假设只得到部分支持。macro200 route的task-centered/sample energy为`.2171`，
task均值route pair relative-L2中位`.6049`，且A/E/D、causal-memory、Core-only和
Program-only反事实证明完整视频路径到达effective BA/action；这说明Writer确实能学到
task-conditioned factor routing。与此同时晚期factor占task-gradient energy约
`96.9%`，351--400的24-task mean只保留单task能量`.0420`，同task相邻CountSketch
余弦降到`-.0099`，raw mean candidate-negative tasks仍为0。Adam一阶moment的相邻
50-macro余弦约`+.011/+.024/-.001/-.033`，而二阶moment余弦约`.915--.945`。

裁决是：soft factor bases改善了早期共同增长和task条件分工，但没有稳定训练方向，也
没有提高single-checkpoint上限；不能再给router追加entropy、basis数、gate或scale来救
该checkpoint。最早剩余问题转向functional estimator的高方差，以及functional action
目标与closed-loop有效policy流形的错位。
