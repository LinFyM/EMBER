# Target-Bound Role-Preserving Program Writer

状态：2026-08-02 post-seal实验中。canonical tensor path、fresh config/checkpoint
schema、step0 identity、causal/private-role mechanics和内部重建路径已实现；移植后
48项聚焦CPU回归通过。真实longest105-frame B20四卡三macro通过，formal-seed
fresh0→1→exact-resume1→3通过；这些live evidence现已写入sealed config。首次formal
调用在任何模型加载/macro前被旧`pending_profile`状态fail-close，形成零数据失败root；
新seal从全新root fresh重启。尚无closed-loop结果，不得把profile/resume写成架构
有效性证据。本文不得反向改变已封存CV-ADR frozen run。

## 1. 目标、非目标与证据边界

目标是在一套 single-checkpoint Writer 中同时保留：

1. mean-backed Core 提供的 task/source semantic carrier；
2. teacher video 的 Action、Effect、Change 和顺序内容直到 effective BA/action；
3. 38 个真实 policy targets 先于 16 个 rank coordinates 的拓扑；
4. full-24 多任务训练下的绝对性能与 breadth，而不是只制造视频 margin；
5. 一条 video 生成一套完整 rank-16 LoRA、step0 functional identity 和现有信息墙。

本版不增加视频对比 loss、task outcome、reward、teacher action/state 输入、额外
adapter、第二套 LoRA、多视频平均、checkpoint 融合、rank/正交/谱 loss、scalar
gate、confidence、手工 prior/innovation 或固定放大系数。它也不恢复 v6 的 320
pre-topology slots，不把 functional loss 当作 closed-loop checkpoint selector。

成功不是一个漂亮内部指标。首小时必须在 paired correct400 上同时看 absolute、
breadth、checkpoint churn 和右端趋势；只有处于同期强架构同档或主路径已经被内部
证实而差距可信地属于训练成熟度，才 exact-resume 第二小时。强 single winner 后
才做五臂。

## 2. 已有量最终要求什么、当前组件为什么不足

输入已经提供 `Q_text`、multimodal task-token evidence `M_f`、task-query patch
evidence `G_f`、native 50-suffix mean Action `A_f`、frame positions 和一条完整
teacher video。问题不是缺一个未经监督的新变量，而是这些量的职责与写出顺序。

正式四格审计表明：

- v5.2 old/new 与 v6 old/new 的 normalized Procedure 顺序响应都存在；
- task-complete 后 Procedure→effective BA/action 只保留 old 约 `26--58%` /
  `20--56%`，最早共同收缩位于 Procedure 后的 slots/AdaLN/compiler；
- old recipe 每 task-cycle 六次 Adam 能恢复动态写出，但同时形成近正交参数轨迹、
  低 breadth 和能力轮换；full24 一次更新也没有解决漂移；
- v6 matched150 在新 recipe 下较 old `+16`，v5.2 为 `-81`，因此架构与训练不能
  分别判死；v6 的 Core-conditioned transition 是正证据，但其 320-slot/centered
  compiler 仍与结果混杂。

CV-ADR 又提供了更早的局部定位：

- Core-only/Program-only 距 full effective BA 为 `.6059/.8119`，两路都必要；
- Effect-only 距 full `.06744`，说明显式双读消除了 AP 的 Effect bypass；
- compiler 的 ProgramRead/CoreRead RMS 比为 `1.021`，不是单路幅度坍缩；
- 但 compiler 对 Action/Effect/Change 的 attention mass 约
  `.043/.431/.525`；remove-A 只在 `1/8` tasks 达门，remove-D 为 `5/8`；
- 真实 shuffled/reversed 会到达 BA，但反转已经 contextualized 的 Program memory
  只有 `.00607` 且 `0/8` 达门；
- same-task 50-video BA centered variance/sample energy 仅 `.10494%`；
- late factor block 占 task-gradient energy 约 `94%`，video 主效应约 `.1%`，
  query/flow 及交互约占一半局部梯度。

因此最小缺口不是再加一个强度旋钮，而是：真实 target semantic context 必须在
动态证据被池化前进入读取；Action、Effect、Change 不能继续争夺一个全局 softmax；
三个物理角色必须保留到 rank coordinate 才联合；训练估计器必须在不减少 B20
query breadth 的前提下降低可消除的 flow-time 方差。

### 2.1 post-seal复核：它是否真的提供task条件分工

24-task梯度近正交且late factor block约占94%能量，说明“一次full24平均”不是完整
根因：不同task已经提出不同请求，但共享写出路径可能无法把它们稳定保存在同一参数
点。可泛化的解决方案又不能是24个task-ID hard experts；Writer只允许从language和
video推断task语义，validation/test必须能组合复用。

本版的最小可证伪task-routing机制是：language/video形成的mean-backed Core先生成
每个真实target的`C_t`，`C_t`随后进入该target每个interval的Effect/Change地址、三路
temporal Q/K以及每个rank的最终读地址。因而task语义不只在末端与全局Program相加，
而是改变真实value被选择和传递的整条activation path；相关task可以因相近Core共享
读取，不同task可以形成不同target/role activation。video A/E/D仍是V，Core不能凭空
制造动态内容。

这不等于已经解决参数共存：evidence reader、temporal blocks、rank reader和factor
heads仍共享权重。首跑必须记录target/role activation与task-gradient block Gram，并
检验checkpoint gained/lost是否从轮换变为多task累计。若Core地址明显task-specific，
但factor-head梯度仍近正交且success继续轮换，则本版只解决了activation routing，
没有解决shared generator容量/优化；下一整体设计才有证据把Core语义提升为factor
计算的soft basis selection。首跑前不先加MoE、gate或bank，因为现有Target-Bound已是
更简单且职责完整的条件路径，尚未被真实B20/closed-loop证伪。

## 3. 选择与被拒绝的相邻方案

选择 **target-bound、role-preserving causal Program**：38 个真实 target 先读
Core；每个 target 用自己的 Core semantic read 与 interval Action 去分别读取
Effect 和 Change；A/E/D 三路独立做 causal temporal contextualization；16 个 rank
coordinate 最后分别读取三路历史，并与 Core 直接拼接进入 conventional factor
heads。

不选择单个 target-bound joint A/E/D attention。它虽然比 CV 更早引入 target，
仍允许 Action 在同一 softmax 中被 Effect/Change 淹没，无法直接回应 `.043` 的已证
失败接口。

不恢复 v6 完整路径。v6 的正证据是 Core-conditioned Action×transition read，而
不是 320 个 target 前 slots、centered-only Procedure 或 AdaLN；这些混杂组件会让
真实 38-target topology 再次过晚进入。

不采用 v7/v8 的单 event binder。v7 的 `8×L` joint attention 接近均匀，v8 又在
rank/target 之前把多个 Action anchor 过早池成 Effect 主导 event。本版保留每个
interval、每个 target、每个物理 role 的 token，直到 rank read 后才联合。

## 4. 精确数据流

### 4.1 保留的前端与 Semantic Core

前端保持 CV/v5.2 已封存合同：

```text
task language + one raw video
  -> Q_text, M_f, task-query raw-patch G_f, A_f
X_f = M_f + G_f
  -> mean-backed, task-selected centered-residual Semantic Core
```

Core 对 frame permutation 严格不变；保留 frame mean 作为 semantic basis，不删除
DC，不做手工 prior/innovation。

### 4.2 38 个 target 先读 Core

对真实 public LoRA target `t in [0,38)`：

```text
C_t = CrossAttention(Q=target_id_t, K=norm(Core), V=raw Core)
```

target identity 只进入 Q；Core raw evidence 是 V。`C_t` 是该真实 policy target 对
task/object/relation 的 semantic context，并在 16 rank coordinates 之间共享。

### 4.3 outgoing interval 与 target-bound role reads

interval `f` 的 endpoint 是 `f+1`：

```text
A_f = native Action probe at frame f
E_f = G_(f+1)
D_f = G_(f+1) - G_f
```

这个对齐表示“当前 action-like intent 后观察到的 endpoint/change”。首个 interval
只读 frames 0、1；prefix 新增未来帧不得改变已有 interval 的 raw roles。

对每个 target 和 interval，构造只用于 Q/K 的地址：

```text
H_(t,f) = norm(C_t) + norm(A_f) + target_id_t
```

然后用两个独立的调用、共享同一 evidence-reader 参数读取：

```text
E_(t,f) = RoleRead(Q=H_(t,f)+effect_role,
                   K=norm(E_f)+task_token_position,
                   V=raw E_f)
D_(t,f) = RoleRead(Q=H_(t,f)+change_role,
                   K=norm(D_f)+task_token_position,
                   V=raw D_f)
A_(t,f) = raw A_f broadcast to target t
```

target/role/token identities只进入 Q/K；没有 identity、frame ordinal 或 routing
embedding 进入 V。Action 同时作为独立 raw value 保留，并只作为 evidence 参与
Effect/Change 路由，不通过手工乘法或 scalar gate 调幅。

### 4.4 role-preserving causal temporal program

形成：

```text
P[t, f, role in {A,E,D}, 256]
```

两层共享的 temporal block 分别作用于每个 `(target, role)` 时间列。Q/K 使用
normalized content、target/role identity 与 endpoint RoPE；V 始终是当前物理
content。mask 是严格 lower-triangular（含对角）且再与 valid interval 相交。
没有 interval 内 joint role softmax，也不提前合成一个 event。跨 role 的联合计算
延迟到 factor head；因此不存在 E/D attention mass 抢走 A value channel 的路径。

这里的 causality 是**以完整、无序的 Semantic Core 为条件**的动态路径 causality，
不是在线预测式的端到端 prefix model。`C_t` 有意由整条 teacher video 的 frame set
形成；追加一帧可以经 Core 改变所有 interval 的语义地址。固定 `C_t` 后，interval
`f` 的 A/E/D 与两层 temporal memory 对任何更晚 endpoint 严格 prefix invariant。
这没有越过信息墙，因为 Writer 推理时本来就接收完整 teacher video；但它确实保留
一条合法的 order-invariant semantic carrier，因此必须用 Core-only、shuffled/reversed
和 memory-reversal 反事实检验其是否重新成为 v4 式主导旁路。若要求端到端 prefix
invariance，就必须删除全视频 Core 条件或构造 per-prefix Core；前者违背 Recenter
与 v5.2 的正证据，后者把 Core 错改为另一条时序 Procedure，当前均无证据支持。

### 4.5 rank last 与 public LoRA

对每个真实 target `t`、rank coordinate `r`，使用三个 private softmax 分别读取
完整 causal role history：

```text
R_A[t,r] = RankRead_A(Q=C_t+target_id_t+rank_id_r, K=Program_A, V=Program_A)
R_E[t,r] = RankRead_E(Q=C_t+target_id_t+rank_id_r, K=Program_E, V=Program_E)
R_D[t,r] = RankRead_D(Q=C_t+target_id_t+rank_id_r, K=Program_D, V=Program_D)
Z[t,r]   = concat(C_t, R_A[t,r], R_E[t,r], R_D[t,r])  # width 1024
```

三个 read 共享投影参数，但 role identity 进入 Q/K，softmax normalization 相互独立。
rank identity 只进入 Q。八个 hidden256、bias-free conventional factor heads联合生成
全部 A/B；最终 projection zero-init，template A/zero B 保证 step0 public functional
identity。不要对 rank coordinates 做正交、谱均匀或独立幅度约束；coherent
near-rank1 是已证的有效几何，不是缺陷。

## 5. Causality、容量与旧失败规避

- **条件时序因果**：interval `f` 的动态 value 只使用
  `A_f,G_f,G_(f+1)`；时间 block 的 token endpoint 为 `f+1`，causal mask 禁止
  读取更晚 endpoint；固定全视频 Core 的 prefix invariance 必须逐层测。Core 的
  全frame-set语义依赖单独报告，不能冒充动态路径因果性。
- **对象/关系/多子目标容量**：Effect/Change 在 target 绑定前不做 token mean；8-head
  role read 可同时选择多个 task tokens，38 targets 保留不同 semantic reads，时间轴
  保留全部 intervals。
- **软职责而非硬语义瓶颈**：Core raw carrier、A、E、D 都直接到最终 factor MLP；
  角色分开的是读取 normalizer，不是预设符号、幅度或 outcome。
- **避免 Target-Spectral**：不改变 effective rank 目标，不正交层/coordinate，不压
  coherent norm。
- **避免 v10**：没有 RMSNorm 后乘大 gate、微小 Procedure 残差放大或额外 scale。
- **避免 Recenter/Core-Program/Prior**：保留 Core mean/DC，不做 strict bilinear，
  不把动态量手工定义为 centered innovation。
- **避免 v7/v8/CV**：不做全局 Action×token/event pooling；target 在语义池化前进入，
  A/E/D 到 rank 端拥有 private channels。

预计新增 target-role activation 在最长105-frame时约
`38*3*104*256 = 3,035,136` elements/condition；bf16 raw tensor约5.8 MiB，远小于
frozen policy functional forward，仍必须用真实 B20 profile证实。factor input由
512增至1024会增加约1.05M参数；旧两层 local-role attention被删除会回收约0.39M，
真实 module enumeration 为11,092,224：semantic frontend 3,453,440、
Core 1,836,544、Program 1,641,216、compiler 409,088、factor heads
3,751,936。它比当前10.297M软上限高约795k，但保持width256/hidden256，并用这部分
容量换取三个private role histories与1024维职责完整factor input。若B20 OOM，只能
基于真实profile做职责不变的内存实现优化，不能先扫B17--B19。

## 6. 训练 operator 与无偏 flow-time 方差处理

CV-ADR同topology GROUP4裁决已经闭合：paired correct400
`82/77/73/110`低于RAW `76/111/99/117`的四点均值与single winner，task漂移仍在。
matched exact50又把A+D collective、remove-A、remove-D的预注册职责门从RAW
`8/1/5 of 8`压到`0/0/0`；Effect-only到full effective BA的mean relative L2从
`.06744`降到`.01882`，contextual-memory reversal从`.00607`降到`.00311`，same-task
BA centered variance/sample energy从`.10494%`降到`.09672%`。同时effective norm
由`64.24`升到`72.06`、stable rank仍约`1.008`，所以这是更大、更coherent却更static
的写入，不是gain/rank collapse。正式职责审计/RAW×GROUP4对比canonical SHA为
`dc01dd97...5141`/`2dc9ee29...5f4d`。

因此首小时operator固定为一次-Adam task-query RAW full24，不恢复normalized GROUP4、
未归一old six-update或CP-24。CV full400中global raw-mean candidate-negative tasks为0，
负pair不等于candidate伤害，投影没有根因依据。该选择属于耦合recipe裁决：不能从中
单独宣称grouping、Adam relinearization或phase order任一因素被孤立否定。

新的 estimator 候选只处理 Beta flow time：保持 B20 个独立同task跨episode queries，
每批对 `Beta(1.5,1)` 使用 randomized Latin strata。对每个 query，随机 permutation
使其边际仍严格是目标 Beta；批内负相关只改变 estimator variance，不改变期望
functional objective。Gaussian noise仍逐query独立；不减少query breadth，不重复
同query，不增加policy samples。实现不得改site-packages，并使用fresh incompatible
randomness/checkpoint schema。

正式采用前必须通过：边际分布/范围测试、task-query deterministic replay、CPU/CUDA
RNG restore、B20每stratum恰好一次，以及固定checkpoint matched gradient diagnostic。
只有 observed gradient variance下降且均值方向不发生超出Monte Carlo误差的系统漂移，
才把它写入首小时合同；当前尚未运行该matched diagnostic，因此canonical config继续
使用现有stateless independent Beta，Latin-stratified路径不得进入第一次正式训练。

## 7. 实现 ownership 与生命周期

保持一个 `CompleteLoRAWriter` 和现有 `train_as_writer.py`：

- `video_program.py`：复用 Meta-LoRA、patch grounding、Q/M/G/A 前端；
- `model.py`：只负责 Core→target Core→role Program→rank compiler→public A/B 编排；
- `semantic_program.py`：原位替换 CV 的 target-agnostic axial A/E/D Program，拥有
  target-bound role evidence read 与 causal role columns；
- `program_compiler.py`：原位替换 CV dual reader，拥有 target Core、private role
  rank reads 与 factor head；
- `internal_compiler.py`：只负责分析时精确重建三个private softmax及target/rank
  routing，不参与训练或推理；
- `checkpoint_schema.py`：集中拥有fresh incompatible checkpoint/trainer/rank schema
  与task-cycle family映射，训练编排不得复制schema分支；
- `functional.py`：只在上述 estimator 门通过后增加一个受config约束的 scoped
  time sampler；不复制policy forward或runner。

最终结构自审相对`51c0ba5`为active source `+1869/-992`、净增877，新增owner仅
`internal_compiler.py`与`checkpoint_schema.py`。前者从已接近上限的
`internal_path.py`抽出target/rank reader精确重建，后者从784行legacy checkpoint
owner抽出fresh schema常量；二者都有当前唯一调用者和明确职责，不是平行runner。
架构门没有hard violation；剩余review signals主要属于未继续增长的legacy大文件，
`inference.py`保持基线870行不增长，新增/增长函数均低于120行hard boundary。

CV 可执行拓扑在新实现commit中退役，只由Git、frozen config/worktree和artifact保存；
不保留runtime architecture switch。checkpoint/config/schema必须fresh incompatible。
内部分析器可在mechanics seal后按新stage names原位迁移；旧CV formal worktree继续用其
frozen analyzer，不要求新main执行历史模型。

## 8. 最短 vertical path 与证伪

正式launch前至少验证：

1. 38×16 coordinates、76 public tensors、全部真实target shape/transpose合同；
2. step0 A等于template、B严格zero、fixed policy action与identity一致；
3. source policy trainable参数为0；
4. Core frame permutation invariance；
5. role Program逐层causal prefix invariance；
6. target/rank/role/frame/token identities不进入V；
7. 单独改变A/E/D会到各自private rank read，且三路梯度finite/nonzero；
8. 主要模块在identity解除后的梯度可达；
9. checkpoint与fresh0→1→exact-resume1→3；
10. 最长105-frame真实video、4 ranks、B20连续完整task cycle无OOM/nonfinite。

最长视频profile使用专用sealed overlay
`configs/pi05_as_writer_target_bound_role_program_long105_profile_v1.json`，它只把
teacher-video schedule seed改为已验证能在首macro选择真实105-frame video的`172`；
正式训练和exact-resume仍使用task-query RAW config及正式seed`20260722`。不得把
普通formal-seed三macro中偶然出现的较短视频冒充第10项。

第10项现已由frozen`e8fb96c`完成：seed172首macro真实max105 frames，三macro各覆盖
24 tasks×B20，wall`59.07s`，峰值CUDA reserved`83,506,495,488` bytes；五个主block
在identity lifecycle后均finite/nonzero，无OOM/clip/nonfinite。随后正式seed
fresh0→1→exact-resume1→3保持同一contract/cursor/scheduler/RNG，三步loss
`.15404/.15141/.14509`连续，validation/test action reads均为0。该证据只解封formal，
不解释性能。

首小时保存cycle50/100/150/200并做同一paired correct400。内部预注册预测：

- compiler Action mass不再有定义，因为A/E/D使用private normalizers；
- remove-A、remove-D应分别在至少6/8 validation tasks产生`>=5%` BA或`>=2%`
  fixed-action变化；
- 反转已contextualized role memory应在至少6/8 tasks产生`>=2%` BA变化；
- same-task BA centered variance/sample energy应明显高于CV `.10494%`，但不能通过
  v10式不稳定放大获得；
- A/E/D→effective BA→fixed action传递应在多数tasks存在，effective norm、跨层
  coherence和near-rank1高增益不能发生Target-Spectral式坍缩；
- gain必须由多个tasks贡献，不能只靠一项Object task。

以下任一组合可证伪核心设计：四个correct400都明显低且无右端趋势；A private path
仍在多数tasks对BA/action近零；temporal memory reversal仍被压到`<2%`；Core或任一
role完全主导；held loss下降而closed-loop持续下降；task breadth收窄且checkpoint
success集合持续轮换。失败后必须定位最早接口并替换完整职责，不在本模型上加gate、
scale、旁路或额外loss。
