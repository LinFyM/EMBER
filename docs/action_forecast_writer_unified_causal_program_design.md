# Unified Causal Program Writer 设计

**状态：2026-08-02 CPU+CUDA task/query RNG-v2已重封存；RAW fresh 0→200运行中**

本文负责 Semantic Program Grid（SPG）一小时门失败后的下一条 canonical
AS-Writer 路径。它不是把 v7、v8、v10、Loom 或后续版本整体判死后重新命名，
而是把历史证据拆成“局部机制 × 训练合同”后，只替换已经定位到的最早失效接口。

当前UCP恢复为唯一可执行路径只服务于训练受控格，不代表撤销AP的局部根因或放弃
CV-ADR。封存`b52cb54`已完成group4 B20/105-frame profile、formal-seed
fresh0→1→resume1→3→7和raw fresh0→1→resume1→3；`85a82cb`把同一运行面逐blob
恢复到canonical并退役AP/endpoint runner。随后审计证明该所谓task/query-keyed
合同只固定CUDA Gaussian noise，遗漏CPU Beta flow timestep；因此旧profile/resume
只证明shape、显存、cursor和相同rank/order可重复，不能证明跨rank/phase随机身份。
task/query RNG-v1 raw的
configured-decay400/runtime-autoscaled200消融为`81/72/107/78`；修正formal total
和stage边界后，clean `cfc2ad1` true-fast400 raw为`89/71/82/117`。scheduler让
macro200提高39但没有抬高UCP ceiling、解决breadth或消除task轮换，且参数/Adam
轨迹持续旋转，不能当作训练解。原`cfc2ad1` cycle-normalized randomized-group4
已在step307正常停止并标记invalid-contract，禁止评测/resume。canonical修复将CPU
time与CUDA noise共同按query锁定，并通过fresh v2 config/checkpoint schema阻断旧状态。
`dae13bf`的CPU全回归`241 passed`；RAW 0→1→3和GROUP4 0→1→3→7真实B20
fresh/resume均已完成。tasks12/14/34/37跨operator换rank后，functional loss和raw
task-gradient norm逐位相等，CountSketch最大绝对差仅`5.82e-11`，因此v2正式config
已重新seal。clean `55faeeb`的RAW已从fresh identity正式运行0→200；首macro合同
健康。完成后用cycle50/100/150/200同一paired correct400评测，再从新root运行
GROUP4并裁决。CV-ADR在隔离worktree等待该
operator结果，不能让训练bundle替架构背锅，也不能用架构aggregate替recipe定罪。

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

匹配每task恰好150次teacher-video visit后，交互更强而不是消失：v5.2 old
step900为`132`，task-complete macro150仅`51`，paired old-only/new-only为
`90/9`；v6 old step900为`95`，task-complete macro150为`111`，old-only/
new-only为`19/35`。两条recipe effect分别为`-81/+16`，描述性difference-in-
differences为`97`。v5.2两格仍有B21/B20导致的`75,600/72,000`query差异，且四格
optimizer update数、scheduler phase与AdamW/moment时钟没有匹配，所以这证明的是
强烈的架构×训练bundle交互，不识别某个单独recipe成分。正式审计artifact为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json
SHA256 cb54bdfa4ccbfa3e82471d05d4be3ff1e4bfb23f22eafad9bb1b41c197749369
```

旧/new内部panel还把交互定位到Procedure之后：v5.2 old shuffled/reversed的
Procedure relative L2只有`.0393/.0589`，但effective BA为`.7400/1.0346`、
action为`.0953/.1902`；task-complete winner的上游Procedure差异并未消失，
下游BA/action响应却约弱3–5倍。v6 old到task-complete也呈同方向压弱，只是v6
task-complete absolute反而提高。因此不能把新recipe概括为“没读顺序”，更准确的
待检验机制是compiler/function侧对条件innovation的利用随topology和optimizer
轨迹共同变化。

v6 的干净 slow/fast task-complete scheduler 对照进一步表明：

- slow-decay2000 在 macro200 为 129，五臂 `129/131/108/111/105`；
- fast-decay400 在 macro200 为 133，macro400 达到 143；
- slow 到 macro400 只有 125。

因此 slow scheduler 可能保留更多视频差异，但没有 absolute 优势；首版新架构
不能把 slow2000 与拓扑同时锁死后再把结果归因于结构。

严格 surrogate 审计又排除了“用 held functional loss 代替 closed-loop 来裁决
架构”的做法。v5.2-new、SPG、UCP、v6-fast/slow、v5.2-old 和 v6-old 使用的
online held panel 完全相同，均为8 tasks×8 videos×8 queries，manifest SHA256
`53cbf9e74cec9cf7a96ac435e092ff036410f08bed0b9d10d89b1b34ae8ea3a8`。在20个
正式主候选上，held loss→correct 的描述性 Pearson/Spearman 反而为
`+.346/+.484`；按架构去均值后为`+.462/+.644`，held loss→breadth 为`-.501`。
16个相邻checkpoint的一阶差分中，train25/held/norm 对下一次correct变化的
Pearson仅`+.031/+.120/-.347`；逐`architecture×task`去均值后的held→success
也只有`-.055`。这些重复checkpoint不能当独立因果样本，但方向和直接反例一致：
SPG `100→150`、UCP `100→150`都是held略改善而correct显著下降，v5.2-new
`200→400`则held恶化而correct上升。

所以 functional surrogate 只保留三项职责：检查finite、识别粗粒度训练退化、
描述teacher-state上的局部拟合；它不再承担checkpoint selector、架构裁判或task
漂移指标。正式选择必须依赖严格paired correct400、breadth、gained/lost、成功
集合Jaccard和Program/Core→effective BA→fixed action方向传递。尤其old→new使
v5.2与v6的held都改善，却让closed-loop分别`-81/+16`，进一步说明recipe改变的
是topology对条件innovation和source-policy有效流形的利用，而非统一提高同一个
可由held loss观测的目标。

训练机械审计进一步把old/new bundle量化到optimizer时钟。最干净的同topology格是
v6-old B20 slow12000与v6-full24 B20 slow2000：前150次每task teacher-video
`(demo, sampled frames)`逐项相同、query构造相同，visit150边界LR也相同。旧配方
每完整24-task exposure却做六次`mean4→clip→AdamW`，新配方只做一次
`mean24→clip→AdamW`。冻结参数、vanilla SGD的一阶近似下，旧cycle系数约为
`-6η mean24`而新cycle为`-η mean24`；实测visit150六个旧LR之和与新LR比为
`5.999553×`，前150 exposures累计比为`6.006864×`。Adam `(β1,β2)=(.9,.95)`
在旧cycle后的保留率是`(.531441,.735092)`，新cycle仍为`(.9,.95)`；旧groups
2–6还在依次更新后的参数点重新线性化，clip和weight decay也各执行六次。

这不是纯尺度解释：v6 old/new-slow在visits100→150的Writer更新方向cosine只有
`.049269`，路径norm为`9.5723/3.3109`；所有主要模块方向都接近正交，endpoint
Adam exp_avg/exp_avg_sq cosine为`.0331/.5637`，clip触发为`21/1`。因此已识别的是
“聚合+重线性化+optimizer memory/clip/WD clock”真实改变basin；尚未识别哪个子项
单独负责行为差异。任何serial收益都不能冒充“减少梯度抵消”，任何serial失败也
不能单独否定small-task updates。正式artifact为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_architecture_training_mechanics_audit_seed7_20260801/analysis.json
SHA256 c910a9335473d5a6da155db9afd4b9b1ab4ca2104f6a1da133c7e6857520e521
```

现有最小未填因果格是v5.2 full24/B20/slow2000，以及v6 old-rank-rotating/B20/
fast-exposure scheduler；只有serial closed-loop和内部结果表明这些格仍能改变决策时
才启动。`β→(.9^6,.95^6)` exposure-memory control与v5.2 old B21→B20更靠后，
不能同时混入下一版架构。

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

逐版证据边界进一步收紧为：v7只否定entropy `.99963`且Core→BA约
`.1–.2%`的global binder/Core-query-only接口，不否定8个Action anchors；v8只
否定Effect主导且过早`8→1`的single-event bottleneck，不否定Action/Effect双流；
v10五臂`103/94/75/67/43`反而证明同一full24-fast也能学到顺序特异性，强负证据
是把Procedure RMS `.0145`经RMSNorm/AdaLN放大约`14–20×`，不是interleaved
双流本身。Loom的matcher接近`1/255`随机互选、confidence/gap无条件分离，说明
这些无锚点变量不会仅靠换recipe自然可识别，但teacher-visible relation与policy
Action软比较仍未被否定。Recenter只能联合否定“删Procedure DC且禁止Core value”，
不能分别否定centering或Core carrier；Core-Program只否定strict bilinear作为唯一
factor content；Prior-Innovation没有branch反事实，且coherent norm健康、video
centered energy仅`.052–.058%`，是最符合“条件innovation被训练bundle压弱”的
历史候选之一。Target-Spectral明确否定强制正交高rank，但不否定target-first/
rank-last配conventional coherent heads。

因此serial-4之后的历史重访不是恢复整版旧代码。若serial-4先证明dynamic
innovation、breadth或漂移确实随update粒度改善，优先级为：保幅的v10式双流/
interleaved Procedure；软semantic prior+target-local innovation；不早池化的
Action anchors与局部关系；最后是target-first/rank-last加常规coherent heads。
每个候选先跑50 exposure cycles并用其既有内部失效量作先验门；Loom无锚点gap、
exact Recenter、strict Core-Program和正交Target-Spectral不因一次recipe结果自动
复活。

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

formal实现有一个必须fail-closed的边界：一小时门的`selected_stop_after_step`不能
替代scheduler构造所用的总训练尺度。LeRobot scheduler会在runtime logical total
小于`decay_steps`时自动按比例缩短warmup和decay。新task/query-keyed raw overlay曾
误写`formal.total_steps=200`，因而实际执行`warmup8 + decay200`而不是配置名暗示的
`warmup17 + decay400`；macro25/50/100/150的正式LR逐点验证了这一点。该run只能作为
scheduler消融，不能与group4做operator因果比较。

纠正后的raw必须是`total_steps=400, selected_stop_after_step=200`；group4每logical
cycle含6 updates，必须是`total_steps=2400, selected_stop_after_step=1200`。loader
必须拒绝logical total短于sealed decay或group更新数不整除cycle的formal config。
真正fast400两臂都从纠正后clean commit fresh identity启动，不得resume误缩放root。
阶段边界也属于同一合同：raw必须是`stage_stop_steps=[200,400]`，group4必须是
`[1200,2400]`；selected stop必须在列表内，最后一项必须等于formal total。第一次
纠偏launch正是在创建output root前被旧单点列表fail-closed，证明不能只在loader
检查scheduler total而忽略runtime stage parser。

误缩放raw的正式paired correct400已经完成：macro50/100/150/200为
`81/72/107/78`，single-checkpoint observed-best是macro150。其breadth为8，但逐task
只有Long `16/1`、Goal `1/28`、Object `27/32`、Spatial `1/1`，top2贡献
`60/107=56.1%`；absolute仍低于旧UCP raw117和SERIAL121。三次checkpoint转移的
gained/lost为`28/37`、`54/19`、`14/43`，成功集合Jaccard仅
`.404/.421/.529`。macro150→200在LR已接近floor时仍丢失43 states，effective BA
mean norm只从`52.94`变到`51.92`，不是单纯norm坍缩。该cell不续训、不做五臂；
它只证明过快衰减没有解决UCP漂移，不能与旧ambient-RNG fast400做单因素scheduler
归因。正式candidate analysis SHA256为`bfd580d4...0993`。

raw-full24 UCP 的四个paired correct400为`82/117/100/110`，best macro100之后
回落。四点union为`169`，比single best高`52`；50→100、100→150、150→200分别
gained/lost `64/29`、`18/35`、`39/29`。train loss继续从约`.116`降到`.101`，
held functional loss却维持`.131–.132`。因此一小时门失败且不续第二小时、不做
五臂，但存在很强的checkpoint能力轮换。

macro100 refs1真实内部纵向路径通过严格fail-close：五条件batch的canonical重算
在Program、coordinates、factor、public A/B、effective BA和fixed action上均为
bitwise相同；reader target/rank centered energy为`.240/.117`，不是SPG的
`1e-5`级同质化。same/wrong/shuffled/reversed从final Program到effective BA再到
fixed action的relative L2分别为：

```text
Program       .190 / .492 / .352 / .447
effective BA  .040 / .190 / .065 / .107
fixed action  .014 / .067 / .016 / .030
```

但当前写出主要依赖absolute `X`：固定X只替换A/D时，四个条件的effective BA
只变`.014/.021/.024/.024`，action只变`.005/.006/.007/.009`；删除全部dynamic
A/D后effective BA仅变`.049`、action`.011`。correct LoRA norm约`59.5`，q/v
跨层cosine约`.917/.923`，低于v6的高增益coherent流形。这个组合证据既排除
“UCP compiler断路”，也没有证明UCP结构已经充分；它精确支持先测full24是否把
task-specific dynamic innovation平均掉。

随后完成的exact50把refs1结论提升为每个validation task全部50条video的证据。
400 rows完整、每task reference ordinal `0..49`恰好一次、0 rollouts；pooled
same-task effective-BA centered variance/sample energy仅`.09008%`，fixed-query
action仅`.01656%`。八task BA比例范围`.0520–.1568%`，虽有约`74.8–92.9%`
变化位于task-mean正交方向，但绝对条件能量很小。same/wrong/shuffled/reversed的
Program→BA→action relative L2为
`.215/.499/.356/.440 → .043/.187/.063/.105 → .0138/.0636/.0153/.0325`；
固定X只换A/D时wrong/shuffled/reversed的BA约`.0223/.0248/.0213`、action均约
`.0055–.0058`。correct LoRA norm/stable rank/top singular energy为
`59.108/1.00319/99.714%`，q/v能量约`81.16/18.83%`。因此dynamic教学弱不是refs2
小样本偶然。artifact SHA为：

```text
analysis.json a6e40cd64dd9d0af6e648b7f746c42c3929cde6d520664cafb9c5afada5825a8
summary.json  386a04f51b56a27ae680f341885a912f7f364d7ebbf281f18876c7743836acaa
```

下一反事实冻结全部UCP topology、参数、B20、video/query/RNG和信息墙，只改变
optimizer/update granularity：令`cycle, phase = divmod(update, 6)`。每个cycle先
生成与raw-full24完全相同的cost-balanced四个rank组和rank rotation；每组六个
tasks仍按真实视频长度long-first，phase只取每rank的第phase项。因此每次全局4
tasks等权，六次更新恰好覆盖24 tasks，且每个task的`task_visit=cycle`。1,200
optimizer updates正好对应200 task cycles、4,800条单视频条件和96,000 action
queries。

LR必须按task exposure严格阶梯匹配：

```text
LR_serial(update) = LR_rawfull24(floor(update / 6))
```

实现上仍构造原`warmup17 + decay400` logical-cycle scheduler，只在每个第六次
optimizer update结束后推进一次；同一cycle的六次AdamW使用完全相同LR。禁止把
它写成连续`warmup102 + decay2400`，因为那会在一个cycle内部取六个不同LR，尤其
会改变warmup早期的实际尺度。每150 updates保存，对应cycle25；候选
300/600/900/1200对应cycle50/100/150/200。

这个反事实的不可约变量是完整更新粒度：每cycle从一次clip/AdamW/weight-decay/
moment update变成六次，后五组梯度也在已更新参数上计算。不能把结果进一步冒充
“batch size”“Adam moment”或“累计LR”某一个子因素的单独因果效应。若它仍不能
恢复absolute、dynamic写出和稳定breadth，再把责任转向UCP表达与functional
surrogate；只有证据随后单独指向LR时，才另做full24 slow2000。

还有一个必须显式报告的顺序混杂：full24里long-first只决定同一参数点上的计算
先后，而serial-4每个phase之间发生AdamW，所以长视频task更常在cycle早期、短视频
task更常在后期。若per-task改善与phase ordinal或视频长度强相关，应优先解释为
optimizer curriculum，而不是task-gradient cancellation被解决；不能为消除这个
混杂而违反已批准的long-first合同，只能用metrics和结果相关性识别它。

这个混杂不是理论上的小量：用raw-full24正式200 cycles的4,800个真实video
cost重放同一serial分组，visit-level phase ordinal与sampled-frame cost的Pearson
为`-.8331`，24个task的mean phase与mean cost为`-.8734`；phase0..5平均sampled
frames依次为`64.62/41.05/32.42/28.88/25.73/20.88`。最长task38在200 cycles中
始终位于phase0，mean sampled frames=`84.42`。因此后续per-task改善必须同时对
phase/cost做相关性审计，不能把serial结果单因归为减少同次梯度平均。

selected4 raw-mean energy ratio不能当成功指标：四个近正交等norm梯度的机械基线
就是`1/4=25%`，而full24是`1/24=4.17%`。它只验证干预确实减少了同次聚合项数。
聚合解释必须由跨cycle参数行为、A/D条件innovation、single-checkpoint breadth、
envelope gap和closed-loop共同支持。

canonical实现已完成：`cycle,phase=divmod(update,6)`、selected4 exact raw mean、
4×4只读Gram、logical-cycle scheduler、fresh serial config/checkpoint/trainer/rank
schemas和midcycle cursor均在原训练入口内实现，没有第二runner或第二Writer路径。
formal checkpoint/stage stop必须为完整cycle边界；profile/formal teacher-video seed
分别为`172/20260722`。全仓CPU回归`233 passed`，architecture guard无hard
violation。clean detached `10a71a1`上的live seal现已完成：18-update B20 profile
覆盖3个完整cycles，首update真实包含105 sampled frames，峰值allocated/reserved
为`76,971,835,904/83,647,004,672` bytes，全部finite；formal seed又完成
fresh0→1、resume1→3、resume3→7。step1/3全部checkpoint文件保持不变，前六phase
覆盖24 unique tasks，scheduler仅在phase5后推进。canonical config已seal，正式run
必须从后续新clean detached commit的fresh identity开始，不能续接smoke。

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
  该硬门检查finite与effective BA，而canonical bf16两段LoRA会因rank求和次序改变
  产生小量action execution drift，必须记录但不得误要求位级不变；
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
