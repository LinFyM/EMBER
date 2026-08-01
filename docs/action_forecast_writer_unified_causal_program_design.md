# Unified Causal Program Writer 设计

**状态：2026-08-01 implementation authority**

本文负责 Semantic Program Grid（SPG）一小时门失败后的下一条 canonical
AS-Writer 路径。它不是把 v7、v8、v10、Loom 或后续版本整体判死后重新命名，
而是把历史证据拆成“局部机制 × 训练合同”后，只替换已经定位到的最早失效接口。

## 1. 当前证据与结论边界

### 1.1 架构和训练方法不可分开解释

目前最关键的二乘二观察是：

| topology | old update recipe | task-complete fast-decay recipe |
|---|---:|---:|
| exact v5.2 correct400 | 132 | 120 |
| v6 correct400 | 121 | 143 |

这个表不是随机化 factorial：两条 old run 与新 run 还混杂 optimizer cadence、
exposure、scheduler，v5.2 还混杂 B21/B20。它只证明 architecture × recipe
interaction 很大，不能把“新训练一定更好”或“某个 topology 整体无效”写成结论。

v6 的干净 slow/fast task-complete scheduler 对照进一步表明：

- slow-decay2000 在 macro200 为 129，五臂 `129/131/108/111/105`；
- fast-decay400 在 macro200 为 133，macro400 达到 143；
- slow 到 macro400 只有 125。

因此 slow scheduler 可能保留更多视频差异，但没有 absolute 优势；首版新架构
不能把 slow2000 与拓扑同时锁死后再把结果归因于结构。

所有 v7、v8、v10、Loom、Recenter、Core-Program、Prior-Innovation 和
Target-Spectral 正式负结果都使用 full24/B20/one-video/fast400；没有 old-recipe
反事实。当前只允许以下强度的历史结论：

- 独立负证据较强：v7 全局 `8×L→1` 近均匀 binder；v8 过早 `8→1`
  EventRead；v10 dominant phase mixer 和 tiny Procedure 经 RMSNorm/gate 控制
  大 Core；Loom 无监督 confidence/gap；Recenter 删除 Procedure DC 并禁止 Core
  value；Core-Program strict 双必要乘法；Target-Spectral 强制正交高 rank。
- 只在当时 recipe 下失败、不能整体否定：8 个 Action anchors、局部
  Action×change 关系、causal Procedure、双流表示、Semantic Core、target-first/
  rank-last 和 target-local read。
- Prior-Innovation 的手工 prior/innovation 职责分解是中等负证据；Core prior 或
  centered innovation 作为非强制组成并未被单独否定。

### 1.2 SPG 一小时门与最早失效接口

SPG formal macro50/100/150/200 paired correct400 为：

```text
97 / 115 / 77 / 100
```

envelope union 为162，但 single-checkpoint best只有115，macro100→150 lost51、
gained13，之后又反向轮换。它不进入第二小时，也不做昂贵五臂 rollout。

SPG macro100 refs2 内部反事实的关键传递为：

| condition | Program relative L2 | Program coordinates | effective BA | fixed-query action |
|---|---:|---:|---:|---:|
| same-task-other | .967 | .355 | .066 | .022 |
| cross-suite wrong | 1.186 | .715 | .221 | .062 |
| shuffled | 1.193 | .627 | .116 | .026 |
| reversed | 1.202 | .658 | .116 | .034 |

固定 Core、只改变 Program 时，shuffled/reversed 的 effective BA 仍为
`.116/.116`，所以时序 Program 不是断路。最早失败发生在 compiler routing：

- CoreReader normalized entropy `.999992`；
- CoreReader target-centered attention energy `3.9e-5`；
- ProgramReader entropy约`.987`；
- ProgramReader target/rank-centered routing energy约`4–5e-5`；
- target/rank coordinate-centered content energy约`1e-5`；
- exact50 stable rank约`1.000001`，top singular energy超过`99.9999%`；
- B columns几乎完全相同；same-task video variance/sample energy从macro50
  `.419%`继续降到macro200 `.210%`。

当前实现把 `std=.02` 的 target/rank identity 与 RMS约1的 target Core 相加，
随后 `target_core + program_coordinates`，再跨16 ranks和38 targets做全局 mixer。
强 Program 差异因此被写成几乎所有真实policy targets共享的同一方向，并被独立
Core carrier进一步稀释。这不是 Target-Spectral 式低norm坍缩，而是路由同质化。

正式内部artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_spg_cp24_internal_specificity_macro0100_refs2_79fb7ee_20260801/
summary.json
SHA256 e3903f31afced37dc9ce4c4fe49dce813d94cf6709b5956f50483a86e28a3a93
```

### 1.3 训练端根因

SPG CP-24 能机械消除负 pair，但不是漂移解法：

- projected/raw norm约`1.25`，方向cosine约`.983`；
- raw full24 mean只保留平均单task gradient energy的`5.74%`，最后25 macros
  为`4.79%`；正交等norm基线是`1/24=4.17%`；
- 即 raw mean 的 coherent equivalent 也只有约`1.38`个task，末段约`1.15`；
- CP投影提高到`9.53%`、末段`6.99%`，仍丢失绝大多数非负但近正交的task
  innovation；
- train functional loss在24 tasks上都改善，held functional loss和closed-loop
  不跟随，说明问题不只是负梯度冲突。

因此下一版删除投影，只保留24×24 Gram和candidate-direction诊断。删除CP不是说
冲突不存在，而是当前投影主要放大同一方向，不能恢复被平均掉的正交信息。

现有B20 query schedule长期边缘近似episode内均匀，但单次task update噪声不小。
前200 macros精确重建的4,800个task-visits中：

- `6.4375%`至少漏掉一个五等分normalized-progress区间；
- 单visit五区间分布对均匀分布的TV均值`.1756`、最大`.5`；
- phase mean总体`.50015`，证明长期没有明显偏置。

artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_action_query_phase_variance_macro0200_seed7_20260801/analysis.json
SHA256 636c3072e59e7ca1df04ad438c0a837de1dcc515e6f370639519ebcbdb875c2f
```

这只证明估计器覆盖方差，不能把normalized progress解释成真实语义阶段，也不能
预先宣称分层会提高闭环成功率。

## 2. 最终需要表达什么

一条teacher video必须同时提供：

1. 当前任务里有哪些对象、关系和子目标；
2. policy在每个观察状态下隐含的操作hypothesis；
3. 随时间发生了什么task-grounded变化；
4. 哪些完整程序内容应写入38个真实policy targets和16个rank coordinates；
5. 多task训练下仍能共享的高增益coherent写入方向，以及必要的target/rank分化。

SPG已经证明未池化Program有足够强的视频/order差异；缺少的是既保留绝对语义
基底、又不允许独立静态旁路跳过Program的compiler职责。

## 3. Unified Causal Program

### 3.1 保留的前端信息墙

前端不变：

```text
task language + exactly one action-hidden teacher video
→ Q_text
→ multimodal task-token evidence M_f
→ task-query patch evidence G_f
→ X_f = M_f + G_f
→ native 50-suffix mean Action probe A_f
```

Writer仍不得读取teacher action、state、reward、terminal、task ID、filename或隐藏
normalization；frame stride仍为5。`A_f`是frozen source policy latent，不是真实
teacher action。

### 3.2 正确的interval对齐

对每个可观察interval `f→f+1`：

```text
D_f,l = G_(f+1),l - G_f,l
Y_f = [X_f,1 ... X_f,L, A_f, D_f,1 ... D_f,L]
```

因此：

- `A_f`来自interval起点frame；
- `D_f`是随后实际观察到的task-grounded变化；
- `X_f`保留该操作前的未中心化绝对语义状态；
- row只有在frame `f+1`到达后才完整可见，时间位置使用endpoint sampled-frame
  ordinal；
- shuffled/reversed必须先重排真实输入帧，再重新计算`A_f`、`X_f`和`D_f`。

这里仍不把`A_f`当成有标签的teacher action，也不监督一个显式action-effect
matcher；它们只是同一interval的软证据。相较v6的`A_f`读取incoming
`G_f-G_(f-1)`，这个定义在因果方向上无一帧错位。v6的高absolute仍支持保留
Action和transition，但其弱视频margin不能反过来证明incoming pairing更正确。

Program shape为：

```text
[F-1, 2L+1, 256]
```

它保留所有起点absolute states；最后一帧的task-grounded变化进入最后一个`D_f`。
最后一帧单独的`M_(F-1)`和`A_(F-1)`不会进入Program，因为video结束后没有可观察
outgoing effect。若内部证据证明这造成completion-state缺失，后续必须整体重审
node定义，不能临时加terminal token或静态旁路。

### 3.3 两层causal axial Program

每层依次执行：

1. 同一interval内沿`L absolute X + one Action + L changes`做全attention；
2. 每一语义列沿interval轴做causal attention；
3. width256→1024→256 FFN。

frame endpoint、value type和task-token ordinal只进入Q/K。V始终是未做terminal
RMSNorm的physical content；这样小`D_f`不会像v10一样仅因RMSNorm/gate被固定
放大14–20倍。pre-norm仍可用于Q/K和FFN稳定性。

严格prefix合同：前`k`个Program rows只依赖前`k+1`帧；改变更晚帧不得改变
这些rows。最终Writer在完整teacher video结束后读取全部Program生成LoRA，这与
causal prefix合同不冲突。

## 4. 单级target/rank raw-value reader

### 4.1 38 targets先于16 ranks

真实LoRA topology仍先枚举38个policy targets，再在每个target下展开16个rank
coordinates。target/rank identities分别为`[38,256]`和`[16,256]`，不使用
`38×16`任意独立表。

### 4.2 单级读取

对每个`(target m, rank r)`：

```text
q_mr = RMSNorm_target(t_m) + RMSNorm_rank(r_r)
z_mr = CrossAttention(
    Q = q_mr,
    K = RMSNorm(Program) + normalized type identity + two-axis RoPE,
    V = raw Program content
)
```

然后`z_mr`直接进入现有八个hidden256 coherent factor heads生成public A/B。

首版明确删除：

- permutation-invariant Core module及其独立加法value旁路；
- target Core first-hop；
- `target_core + program_coordinate`融合；
- 跨rank和跨38 targets的CoordinateMixer；
- gate、scale、confidence、null token、谱/rank loss、正交约束、B-only residual；
- 第二套LoRA或多video/LoRA平均。

target/rank/type identities只进入Q/K，不作为V或factor residual凭空制造内容。
归一化identity解决SPG里`std=.02` identity被RMS约1 content淹没的问题；它不强迫
rank正交，也不追求高stable rank。

不先加入target-context第二hop：SPG已经有Core target read→Program read两级，
却形成近均匀路由。固定target/rank query通过content-dependent K已经能产生
task/video-dependentattention；只有单级reader被内部证据明确证伪后，才重新
论证第二hop。

## 5. 已规避的旧失败与新风险

| 旧失败 | 本设计如何规避 |
|---|---|
| v4静态/absolute-time旁路 | 无独立Core或静态compiler旁路；顺序在实际Program内重算 |
| v7全局binder与Core query-only | 不先池化成一个event；最终V直接来自完整Program |
| v8 Effect主导的早期EventRead | 38×16 coordinates各自读取完整未池化Program |
| v10微小Procedure经norm/gate高增益控制Core | 动态量不通过terminal norm/gate放大大载体 |
| Loom不可识别gap/confidence | 不引入无监督中央变量 |
| Recenter semantic-basis starvation | 未中心化`X_f=M_f+G_f`作为raw Program value保留DC语义 |
| Core-Program strict bilinear | 无双必要乘法或移动basis |
| Prior手工职责 | X/A/D共同进入标准attention，不预定prior/innovation代数所有权 |
| Target-Spectral增益坍缩 | conventional coherent heads，无正交/谱目标 |
| SPG路由同质化 | normalized identities、单级read、无Core add、无global mixer |

最大新风险是“形式上统一、功能上仍静态”：`X`有L列且RMS远高于一列Action和
L列change，reader可能只读重复absolute X。这不能靠硬gate修复，必须通过type
ablation和attention mass直接证伪。另一个风险是`2L+1`列使最长视频activation
近似翻倍；参数减少不代表B20显存自动可行，必须真实profile。

## 6. 训练合同

### 6.1 raw full24 mean

每macro仍覆盖24 train tasks各一次，每task一条video、一套LoRA和B20独立
same-task action queries。task内mean、24 tasks等权，最后只有一次clip、一次
AdamW、一次scheduler update。

保留逐task gradient、24×24 Gram、block Gram、raw candidate-negative tasks、
mean/average-task energy ratio和CountSketch跨macro方向诊断，但candidate direction
严格是raw full24 mean，不再投影或放大。

### 6.2 无偏phase-stratified B20

每个task visit仍选择20个不同teacher episodes。对20条query：

1. 生成20个normalized-progress strata；
2. 用`seed/task/visit`确定性随机permutation把strata分配给20个episodes；
3. 每个stratum内部独立确定性uniform jitter；
4. `row=floor(u*episode_length)`；
5. 跨visit轮换episode与stratum映射。

因为随机stratum和bin内jitter的混合在`[0,1)`上均匀，每条episode query的边缘
仍近似原来的uniform-row estimator；变化只是在一个B20内覆盖完整进度范围，
不按loss、outcome或人工语义阶段重加权。不得固定bin中心或固定episode-bin绑定。

### 6.3 scheduler和一小时门

首版保持`warmup17 + cosine fast-decay400 + peak3e-4 + floor1e-5`，以便把主要
变化限制为结构、删除CP和低方差query estimator。fresh macro0→200，每25保存；
评测50/100/150/200 paired correct400。

只有内部主路径工作但视频innovation随LR过快消失时，才做同拓扑slow2000
counterfactual。不能先验把slow scheduler当作视频因果解法。

## 7. 实现边界与初始化

- 新文件按职责分为Program和target/rank reader；`model.py`只编排生命周期；
- 复用Meta-LoRA、patch grounding、frozen policy、functional LoRA、checkpoint、
  sampler和evaluator；
- 删除旧SPG Core/compiler的active executable path，历史由Git保存；
- fresh incompatible architecture/config/checkpoint schema；
- factor final Linear严格zero-init，step0 public LoRA逐tensor等于identity template；
- frozen source policy无trainable参数；
- public 38 targets、rank16、transpose和A/B合同不变。

参数预算仍以corrected Source-SFT约10.3M为软参照，不因机械匹配牺牲职责。
真实module enumeration为：semantic evidence frontend `3,453,440`、两层Unified
Program `1,838,592`、single-stage target/rank reader `212,224`、八个factor heads
`2,179,072`，总计`7,683,328`。容量下降来自删除已定位的Core旁路和global mixer，
不是为了机械压参数；public LoRA仍完整生成1,287,168个rank16 scalars。

canonical config为
`configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json`。live
105-frame B20 profile与formal-seed exact resume已经完成，config现已seal为
fresh macro0→200 formal authority。

现场profile使用独立detached `0d4c271` frozen worktree和只改变
`teacher_video_seed: 20260722→172`的临时overlay。真实最长task38/demo36共105个
stride-5 frames在step1实际进入rank1首个microtask；三个完整macro均严格覆盖
24 tasks、480 queries和24套单视频LoRA。step wall为
`20.394/18.494/18.504s`，峰值allocated/reserved为
`77,127,082,496/83,345,014,784` bytes，loss、梯度、LR、Writer和optimizer均
finite。zero-init使step1只有factor梯度非零；step2起semantic frontend、Program、
reader和factor四块均finite且非零，符合identity输出层先打开主路径的预期。

同一代码、canonical formal seed随后fresh运行0→1并从step1 exact-resume到3。
step1的manifest、Writer、trainer和四份rank state在resume后SHA逐项不变；三步
metric、LR、20-strata query cursor、teacher-video cursor、24-task覆盖和10组
gradient chunk gather/completion/sync连续。正式root与hash由config的
`profile_evidence`封存。因此B20被选择；只有后续真实OOM或连续非有限才允许直接
降B16。

## 8. 最短vertical path

正式运行前必须验证：

1. `Q_text/M/G/X/A/D`和Program shape；
2. `A_f`与`G_(f+1)-G_f`的interval对齐；
3. causal prefix invariance和padding；
4. frame/order/type/target/rank identities不进入V；
5. `38×16×256`coordinates及全部public A/B；
6. step0 strict functional identity；
7. source policy完全冻结；
8. 主要Program/reader/factor模块梯度finite且可达；
9. checkpoint和exact resume；
10. 最长105-frame真实video的B20 profile；只有真实OOM或连续非有限才降B16。

## 9. 内部证伪合同

除通用Core→Procedure→LoRA分析改为Program链路外，至少报告：

### 9.1 路由与类型职责

- reader entropy、top mass、target-centered和rank-centered attention energy；
- target/rank-centered coordinate content energy；
- target identity permutation与真实38-target映射；
- rank permutation是否只作gauge-compatible coordinate置换而不改变完整BA函数；
- full、X-only、dynamic-only、A-only、D-only；
- fixed X/vary A,D和fixed A,D/vary X；
- 每类attention mass、raw RMS、coordinate/BA/action贡献。

### 9.2 传递与增益

- correct/same/wrong/shuffled/reversed从Program→coordinate→effective BA→fixed-query
  action逐层relative L2/cosine及保留率；
- dynamic input按`.5/1/2`缩放的coordinate/BA/action响应，排除v10式小信号
  固定高增益；
- same-task 50-video centered variance/sample energy及task-mean energy；
- effective norm、q/v/action能量、layer CV、cross-layer cosine、B-column cosine、
  stable rank、entropy rank、top singular energy只作诊断，不设高rank目标。

若Program仍有约1量级order差异但target/rank centered routing/content保持`1e-5`
量级，单级reader被证伪。若X-only几乎复现full而A/D-only和fixed-X动态反事实接近
zero，统一tensor已退化成静态旁路。若BA/action差异充分而correct400始终不改善，
瓶颈转向functional surrogate或source-policy closed-loop manifold，不再继续改
evidence extractor。

### 9.3 训练稳定性

- raw negative pair fraction、median/min cosine；
- raw mean/average-task gradient energy；
- Meta/X/A-D/Program/reader/factor block Gram；
- 每task两个独立phase-stratified B20 estimator的gradient cosine（离线或周期性
  诊断，不把第二batch用于更新）；
- checkpoint gained/lost、breadth、success-set Jaccard、参数/Adam moment方向；
- query phase-bin覆盖和单video跨visitgradient sketch。

## 10. 决策门

macro50/100/150/200 correct400继续与v6 m200约133、v5.2新recipe m200=91、
Source-SFT约109比较。进入第二小时至少要求absolute同档、多个tasks共同右端上涨、
或内部主路径与视频/漂移证据提供可信成熟度理由。

如果不进入第二小时，不跑正式五臂；完成无rollout type/routing/geometry/action分析，
找到最早失效接口后整体重构。若进入第二小时，exact-resume到400，选择单一
checkpoint winner；只有absolute强时才做五臂。任何超过150的checkpoint仍必须
继续检查task漂移、video causality、same-task鲁棒性和closed-loop有效几何。
