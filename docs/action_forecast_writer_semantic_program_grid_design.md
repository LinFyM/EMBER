# EMBER Semantic Program Grid Writer（SPG）

状态：2026-08-01 已完成独立复核、canonical实现和最长105-frame B20四卡三宏步
profile；fresh/exact-resume seal与正式训练尚未执行。当前精确参数为
`10,633,216`。profile的72个单视频条件、1,440条action queries全finite，
macro2起全部主模块梯度可达；这些仍只证明实现/容量合同，不构成性能证据。

本文取代已撤回的Coherent-Procedure/B-only residual提案。SPG不是v5.2或v6的
局部补丁，也不等待v5.2 task-complete结果才决定拓扑。它基于v1至
Target-Spectral的行为、内部传递、LoRA几何与task漂移证据，重新定义teacher
evidence、Core、Procedure、policy target、rank coordinate和多任务更新的完整
计算顺序。

## 1. 目标与信息墙

```text
exact task language
+ exactly one raw action-hidden teacher video
→ one shared Writer
→ one complete task-specific rank-16 LoRA
→ frozen source policy executes the task from unseen initial states
```

Writer不得读取teacher action、proprio/state、reward、terminal、task ID、
filename、hidden normalization或policy outcome。训练action queries只进入frozen
policy functional loss，并与teacher video同task、跨episode独立。这样不是
降低视频要求，而是迫使Writer从单条示范中抽取能跨初态迁移的方法，而非复述
同episode低层轨迹。

## 2. 设计所使用的完整证据

| 证据 | SPG决策 |
|---|---|
| v4的7D forecast、absolute-time Plan/Revision与静态旁路造成shuffled 148 | 不预测低层动作，不使用absolute robot clock，不建raw-image bypass |
| v5.2的`Q_text→patch values`、Core和causal Procedure达到132且五臂理想 | 完整保留稳定语义前端与Core/Procedure双流 |
| v6 mean-centered Core、visual transition、hidden256与task-complete达到143 | 保留其集合Core思想、视觉变化和full-width decoder容量 |
| v7 `8×L→1` joint softmax近乎完全均匀 | Action与视觉变化不做提前联合池化 |
| v8 EventRead被Effect主导 | 不把完整interval压成单一event |
| v10 Effect attention近均匀，Procedure RMSNorm把小信号放大14–20倍 | 不用Procedure gate、terminal norm或固定增益 |
| Loom teacher-policy gap/confidence缺少可靠锚点 | 不预测gap、confidence或source“是否已会” |
| Recenter删除Procedure DC后semantic-basis starvation | 保留完整Action公共程序和Core语义载体 |
| Core-Program严格乘法、Prior手工prior/innovation均失败 | 用标准attention/residual融合，不规定脆弱代数关系 |
| Target-Spectral把stable rank抬到3.32却降至34 | 不正交、不均匀谱、不强迫16个奇异方向 |
| v5.2/v6的16个固定坐标几乎全有能量且无q/v负相消 | 允许建设性coherent carrier，rank只作代数容量 |
| v5.2 same-task视频创新显著强于v6 | Procedure必须在每个真实target/rank坐标直接读取视频，而非只作统一小gate |

## 3. 共享语义前端

对有效task tokens数记为`L`，采样帧数记为`F`，公共宽度`d=256`。

```text
task language
→ frozen text Gemma + rank-4 Text Meta-LoRA
→ Q_text [L,256]

frame_f + task language
→ frozen multimodal Gemma + rank-4 VL Meta-LoRA
→ M_f [L,256]                  multimodal task-token states
→ P_f [256,256]                projected image-position states

Q_text reads raw P_f values
→ G_f [L,256]                  explicit task-grounded patch evidence

X_f = M_f + G_f
```

`M_f`保留Gemma已经完成的图文交互；`G_f`保留稳定task query从真实patch values
显式读取的证据。它们来自同一次prefix forward，不是两套视觉编码器。

同一次frame+language prefix继续送入frozen Action Expert与rank-4 Action
Meta-LoRA：

```text
native suffix hidden [50,1024]
→ mean over all 50 positions
→ bias-free 1024→256
→ A_f [256]
```

`A_f`是source policy在teacher frame上的高层action hypothesis，不是teacher
action。保留原生50-position mean，因为v5.2/v6已证明它有效；v7以后没有证据
证明8-anchor接口更好。

## 4. Semantic Core

Core保留task-token可变容量，并采用v6的mean backbone加task-selected centered
residual：

\[
\mu_l=\operatorname{Mean}_fX_{f,l}
\]

\[
R_l=\operatorname{Attention}
(Q_{\text{text},l},X_{:,l},V=X_{:,l}-\mu_l)
\]

\[
C_l^{(0)}=\mu_l+R_l
\]

随后经过两层width256、8-head、hidden1024的bidirectional task-token
Transformer，task-token ordinal只进入Q/K RoPE：

\[
C\in\mathbb{R}^{L\times256}
\]

Core对frame set permutation严格不变，表达任务、对象语义角色、目标关系、
场景不变量以及语言声明的有序子目标。它不压成固定单向量，容量随任务token和
关系复杂度自然变化。

## 5. Semantic Program Grid

SPG不再要求一个Action token立刻选择一个Effect，也不在时序推理前把两类证据
池化成单event。

对每个可观察interval `f→f+1`：

\[
D_{f,l}=G_{f+1,l}-G_{f,l}
\]

并构造：

\[
Y_f=[A_f,D_{f,1},\ldots,D_{f,L}]
\in\mathbb{R}^{(L+1)\times256}
\]

这里`A_f`与随后观察到的`D_f`只是同一teacher interval的两个软证据；模型不得
把它们当作有标签的真实action-effect pair。

interval row只有在终点frame `f+1`可见后才完整定义，因此它的temporal ordinal
使用该终点的sampled-frame位置。严格prefix合同为：前`k`个interval输出只能依赖
前`k+1`帧；扰动更晚帧不得改变这些输出。Core仍有意读取完整frame set，最终
one-video LoRA是离线教学结果，这不构成未来帧泄漏。

完整网格为：

\[
Y\in\mathbb{R}^{(F-1)\times(L+1)\times256}
\]

两个Axial Program Blocks逐层执行：

1. 每个interval内沿`Action + L个task-token changes`做self-attention；
2. 每个语义列沿interval轴做causal self-attention；
3. width256→1024→256 FFN。

全部为8 heads、pre-norm、residual。frame position、Action/visual type与
task-token ordinal只进入Q/K，不作为value。输出：

\[
P\in\mathbb{R}^{(F-1)\times(L+1)\times256}
\]

compiler读取前只flatten有效网格，不做time-centering、terminal RMSNorm、
confidence、gap或单event pooling。即使attention尚未学好，原始Action与
task-grounded changes仍由residual保留。

所有Program axial attention的value都来自未加入identity的content；ProgramRead
进一步固定为`V=P`，不先对value做RMSNorm或把position/type identity混入value。
零content即使带有非零routing identity也必须保持零value输出，防止重新制造
v4式静态旁路或v10式微小Procedure放大。

## 6. Target-first、Rank-last compiler

旧320 slots在理解真实policy target前就把rank coordinate当成语义slot。SPG先
建立38个真实public LoRA targets：

```text
18 q_proj + 18 v_proj + action_in + action_out = 38
```

target ordinal严格取自sealed `pi05_target_names()`顺序，而不是PEFT state key的
字典序；后者会把layer10排到layer2之前。A/B共享同一target/rank state，只在
最后进入各自factor head。

### 6.1 Core target states

每个target identity `r_m`先读取Core：

\[
C_m=\operatorname{CoreRead}(r_m,C,C)
\]

### 6.2 最后展开rank并直接读取Program

对每个target `m`和rank coordinate `r`：

\[
q_{m,r}=r_m+e_r+\operatorname{Norm}(C_m)
\]

\[
P_{m,r}=\operatorname{ProgramRead}
(q_{m,r},\operatorname{Norm}(P),V=P)
\]

Program value使用raw content；position/type/routing identities只影响Q/K。
标准残差融合为：

\[
H_{m,r}=C_m+P_{m,r}
\]

Core因此为同target的16个rank提供共同task-semantic carrier；Procedure允许每个
真实target/rank读取不同教学过程。模型可以继续形成有效的coherent rank1，也
可以在视频确实需要时自然产生更高rank，不接受任何rank使用率或正交约束。

### 6.3 Coordinate mixer

`H [38,16,256]`进入一个axial coordinate block：

1. 每个target内部沿16 ranks做self-attention；
2. 每个rank沿38 targets做self-attention；
3. 一个pre-norm residual FFN。

target/rank identities仍只影响Q/K，所有value来自`H`。这让q/v、不同层和
action_in/out形成闭环一致的更新，同时不强迫它们同向或正交。factor heads读取
保留残差幅度的raw block output，不加会放大小信号的terminal normalization。

## 7. Complete LoRA generation

八个factor heads保持硬件友好full width：

```text
256 → 256 → target width
GELU
bias-free

q_A / q_B
v_A / v_B
action_in_A / action_in_B
action_out_A / action_out_B
```

\[
A=A_{template}+\Delta A(H),\qquad B=\Delta B(H)
\]

全部final projections严格zero-init，故fresh step0满足`B=0`、public LoRA与source
policy functionally identity。没有QR/SVD、spectral scales、orthogonality、
rank-diversity loss、B-only residual、第二套adapter或multi-LoRA组合。

## 8. 训练架构：single-stage CP-24

SPG默认同时采用一次macro一次optimizer的Conflict-Projected Full-24更新：

```text
24 tasks / macro
每task exactly 1 video → 1 LoRA
每task B20 independent same-task cross-episode action queries
task内mean，24 tasks等权
每rank保留本地6个task gradients到macro boundary
确定性消除彼此一阶负冲突
mean projected task gradients
一次global clip + 一次AdamW + 一次scheduler step
```

实现可采用deterministic PCGrad：若`<h_i,g_j><0`，仅移除`h_i`在`g_j`负方向
上的投影；task遍历顺序由seed、macro和task确定并跨macro轮换。不存在负冲突时
CP-24严格退化为当前full24均值。

逐次投影不数学保证最终任意两个projected task direction都非负，因此正式日志
必须同时报告raw/projected pair Gram、candidate direction对各task的负内积数量和
`raw mean`/projected direction的cosine与norm；不得把“执行了PCGrad”写成“所有
冲突已消失”。实现只长期保留每rank六个local task gradients，通过跨rank通信在
macro boundary形成全24-task统计，不建立第二optimizer或跨macro gradient state。
为避免B20后再物化`24×P`全局梯度，每次只all-gather最多`1,048,576`个参数坐标
并累加full/block Gram；每个NCCL chunk在进入下一chunk前显式等待当前CUDA
stream完成。2026-08-01的共卡phase trace证明Python同步collective调用只保证
work入队：没有这个completion boundary时，快rank可连续排入全部13个chunk，
而慢rank尚未进入首个Gram exchange，形成持续NCCL starvation。显式等待后同一
最长profile连续三macro完成，因此这是一项分块通信完成性合同，不改变CP数学。
投影系数在`24×24`空间求解。为避免不同GPU在PCGrad
零点附近发生浮点分支分歧，由rank0广播最终24个系数，各rank再以本地六条梯度
形成加权方向并做一次sum all-reduce。该tiny broadcast只统一数值authority，
不改变投影定义；无冲突时权重严格为`1/24`，因此与原full24 mean是同一更新。
正式日志记录chunk all-gather和CUDA completion次数，两者在四卡CUDA路径必须
相等；CPU/Gloo路径completion计数为0。

它不是两阶段训练、gradient accumulation recipe、task-local optimizer、
loss-based task reweight、checkpoint融合、多视频或多LoRA平均。额外记录但不
进入loss的证据包括：24×24 Gradient Gram、负cosine比例、raw mean与projected
direction的cosine/norm ratio，以及Meta/Core/Program/compiler/factor分块冲突。
为让“单视频条件噪声”可被后验检验，每个macro还按全局task ID记录该次
teacher demo、采样帧数、task functional loss、raw/projected task-gradient norm、
task对candidate direction的内积，以及semantic frontend/Core/Program/compiler/
factor各自固定32维CountSketch方向。sketch使用固定坐标hash，只用于比较同task
连续单视频访问的近似梯度cosine；它不进入loss、不保留完整跨macro梯度，也不
自称精确重建高维方向。这样可把视频条件方向波动与单纯loss/norm波动区分开，
而不是把后者误称为已经测得的“噪声”。

task漂移已经跨v5.2、v6和Source-SFT出现；global8和降LR没有解决。但梯度冲突
尚未被直接证明，所以这些日志同时承担证伪职责。若`g*≈raw mean`而漂移继续，
必须降低“梯度冲突”解释，继续在AS主线检验functional surrogate与closed-loop
success错位、LoRA更新离开source-policy有效流形、单视频条件噪声和shared
hypernetwork条件路由不足，并从最早失效接口做整体职责重构；不能为保留CP而
改指标，也不能转去RL-Writer回避focused AS根因。

## 9. 预算

设计估算：

```text
shared semantic frontend + Core      ≈ 5.03M
Action projection                    ≈ 0.26M
2 × Axial Program Blocks             ≈ 2.10M
target/rank compiler                 ≈ 1.45M
8 × factor heads, hidden256          ≈ 2.18M
total                                ≈ 11.0M
```

参数上限为软参照。实现必须用真实module enumeration重新核算，保持width256、
8×32和hidden256，不为了凑Source-SFT的`10,297,344`使用奇怪维度。

2026-07-31 canonical raw-value实现的真实module enumeration为`10,633,216`：
semantic frontend `3,453,440`、mean-backed Core `1,836,544`、两层Program
`1,837,568`、target/rank compiler `1,326,592`、八个factor heads
`2,179,072`。低于早期约11.0M估算来自Program/compiler的raw-value attention
不另设value projection，而不是缩窄width、heads或factor hidden。

2026-08-01最长真实105-frame B20 profile root为
`/data/ymdai/outputs/ember/pi05_as_writer_spg_cp24_profile_b20_longseed172_sync_v2_7c1b9fc_20260801`。
三步wall为`20.5359/18.5778/18.5461s`，稳态约`25.859` queries/s、
`193.945` macros/hour；峰值allocated/reserved为
`77,203,449,344/83,529,556,160` bytes。每步24个唯一tasks、每task一条video/
一套LoRA/B20，rank内真实视频长度long-first。negative cosine pair fraction为
`.4058/.3514/.4058`，raw/projected cosine为`.8410/.9426/.9689`；macro2的
frontend/Core/Program/compiler/factor mean gradient norms分别为
`.007706/.003975/.002049/.003736/.129495`。因此CP确实看见task冲突，且SPG
主路径在identity输出层离开零点后全部可达；是否改善closed-loop漂移仍必须由
正式checkpoint曲线证伪。

### 9.1 实现owner与生命周期

- `video_program.py`只负责冻结π0.5上的`Q_text/X/G/A`证据抽取；
- `semantic_program.py`只负责mean-backed Core与因果axial Program；
- `program_compiler.py`只负责38-target-first、rank-last读取、coordinate mixer和
  八个factor heads；
- `conflict_projection.py`只负责task-gradient布局、分块Gram、确定性投影、最终
  gradient安装及四rank初始状态同步；
- `model.py`只编排上述owner并执行sealed public LoRA target/transpose合同。

旧`temporal.py`和v5.2 320-slot实现已从活动树删除，由Git历史及仍在独立frozen
worktree运行的正式v5.2实验保存provenance。旧schema/config不能被canonical
loader执行；不存在同时可选的第二Writer实现。若后续实测证明CP
`projected≈raw mean`且漂移不减，删除触发是把更新恢复为同一SPG的raw full24
mean，而不是保留两套optimizer路径。

## 10. 迭代与判定合同

SPG以及其后任何整体重构都先从identity fresh训练约一小时，再做paired
correct400并与同等一小时规模的v5.2/v6比较：

- 若明显低于有效旧架构，停止该run，不做昂贵行为五臂；做充分无rollout内部
  分析，定位最早失效层后重新从根因设计完整架构；
- 若与同期v5.2/v6同档或更好且曲线仍有价值，exact-resume第二小时；
- 只有single-checkpoint absolute达到当前强水平、接近最佳或出现明确持续提升
  价值时，才做correct/same/wrong/shuffled/reversed正式行为rollout；
- 每次都检查task breadth、能力轮换、Core/Program/target/rank/effective LoRA/
  policy action传递和LoRA几何；
- 不以`150`作为自动完成或自动停止线。即使超过150，只要内部仍有明确漏洞或
  可改进瓶颈，就继续迭代；
- 所有新设计都必须由现有输入、需求、正负证据逐层推导，禁止在失败架构上追加
  gate、scale、旁路或局部residual掩盖根因。

长期focused目标是把AS-Writer推到当前能力范围内找不到可信改进空间，而不是
机械完成某个checkpoint、某条曲线或某个阈值。
