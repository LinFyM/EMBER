# Amplitude-Preserving Asymmetric Dual-Read Writer 设计

**状态：2026-08-01 canonical实现、live B20/profile-resume、fresh macro0→200和
四候选评测均已完成；correct为`91/81/94/91`，一小时门失败。修复后内部反事实
直接否定当前key-only contextual Program/raw-value接口；不resume、不做五臂。**

本文负责 Unified Causal Program（UCP）raw-full24 和 exposure-matched
SERIAL-4 都未通过一小时门后的下一条 canonical AS-Writer 路径。
它不把 v7、v8、v10、Loom 或后续架构整体判死；只删除已被内部
反事实或数值传递局部证伪的接口，并把仍与 full24/B20/fast400
训练 bundle 混杂的能力保留为可复用组件或条件反事实。

## 1. 决策摘要

下一整体模型采用 **Amplitude-Preserving Asymmetric Dual Read
（AP-ADR）**：

```text
task language + one action-hidden teacher video
  -> Q_text, M_f, G_f, native mean Action A_f
  -> permutation-invariant mean-backed Semantic Core
  -> outgoing interval Program [A_f, G_(f+1), G_(f+1)-G_f]
  -> 38 target-only Core reads, broadcast across rank
  -> 38 x 16 target/rank Program reads
  -> raw [Core read ; Program read] concat
  -> eight conventional bias-free coherent factor heads
  -> one complete public rank-16 LoRA
```

两个reader使用独立softmax分母，Core不与Program在同一memory里争夺
attention mass。最后一层不对两路value做RMSNorm、AdaLN、gate、scale、
加法、乘法或global mixer；直接拼接未归一化的physical content。
这是对UCP最早失效接口的整体替换，不是在失败checkpoint上加一个
scalar补丁。

## 2. 证据边界：架构和recipe必须分层解释

### 2.1 已观测的交互

在每task 150次teacher-video exposure的近似匹配点：

| topology | old/small-task update | full24 task-complete | difference |
|---|---:|---:|---:|
| exact v5.2 | 132 | 51 | -81 |
| v6 | 95 | 111 | +16 |

这两行只证明architecture x training-bundle交互很大。v5.2还混杂B21/B20
和scheduler；两行old/new都同时改变task aggregation、每exposure的clip/AdamW/
weight-decay次数、Adam moment时钟和参数重线性化。不得将其简写为
“full24不好”、“old recipe更好”或“某架构已失效”。

UCP的同exposure反事实又表明：

```text
raw-full24 macro50/100/150/200 = 82 / 117 / 100 / 110
SERIAL-4   step300/600/900/1200 = 89 / 100 / 121 / 107
```

SERIAL-4 winner只比raw winner高4；step900->1200 gained/lost为`33/47`，
漂移和breadth漏洞仍在。selected4保留平均task-gradient energy约`27.61%`，
raw-full24末段约`4.98%`，分别接近近正交的`1/4`和`1/24`基线；但
candidate-negative tasks很少。这支持“近正交innovation在平均中被稀释”，
不证明signed conflict是唯一根因。SERIAL-4还捆绑六倍optimizer/moment/
clip时钟与由long-first引入的frame-cost/optimizer-age curriculum，因此不作为
下一架构的默认recipe。

事后按真实metrics进一步缩小bundle：到150 exposures时raw/serial LR integral为
`.037808/.226848`，严格`6x`；raw 150步clip触发0次，SERIAL全部1,200步只触发3次
且都在cycle20前。`weight_decay=1e-4`对应两轨累计纯decay收缩差仅约`1.9e-5`。
所以clip/decay仍是合同差异，却缺乏解释后期`6--10x`动态写出变化的量级；剩余
主要候选是六倍Adam moment/bias-correction与参数重线性化、以及phase-cost
curriculum。后续normalized group4必须针对这些量，而不是笼统声称“小task更新”。

但SERIAL-4 step900 exact50迫使一个重要修正：它没有解决absolute/breadth，
却确实改变了视频动态从Program到函数的传递。与raw UCP winner macro100
的exact50相比：

| internal quantity | raw macro100 | SERIAL step900 |
|---|---:|---:|
| same-task BA centered variance / sample energy | `.09008%` | `.47714%` |
| same-task fixed-action centered variance / sample energy | `.01656%` | `.63732%` |
| same Program -> coordinate -> BA -> action relative L2 | `.215/.070/.043/.014` | `.297/.205/.098/.048` |
| fixed-X shuffled BA/action relative L2 | `.0248/.0055` | `.1484/.0416` |
| fixed-X reversed BA/action relative L2 | `.0213/.0058` | `.1411/.0412` |
| correct reader X/A/D mass | `.488/.041/.471` | `.113/.021/.866` |

SERIAL的八个task BA variance全部为`.230-.798%`，不是只有一个task变化。
但fixed-action pooled `.637%`被Object-3的`5.708%`强烈主导；其他七task
的中位数只约`.019%`。这说明更细更新可以打开dynamic path，但尚不知
是可用teaching signal还是局部不稳定放大；它没有自动变成多task
closed-loop收益。

上表还混杂raw macro100的100次/task与SERIAL step900的150次/task exposure。
因此已经启动raw macro150的同exposure exact50。在该对照完成前，只能结论
“SERIAL轨迹存在更强dynamic writing”，不能将差异单因归于update grouping。
这也意味着AP-ADR后续必须同时做topology与matched-recipe反事实，不能
在任一方失败后把全部责任推给另一方。

### 2.2 后 v5 历史只能按机制判决

当前证据分三类：

1. **局部机制已否定**：内部量将最早失效局部到一个具体接口。
2. **整版recipe混杂**：低分只在full24/B20/fast400 bundle中观测，没有
   matched alternate-recipe cell。
3. **现有证据不可识别**：多个变化同时发生，无法给单个组件定罪。

局部否定边界是：

- v7：否定近均匀的global `8xL->1` binder和Core-query-only接口，不否定
  native Action evidence。
- v8：否定Effect主导的过早`8->1` EventRead，不否定Action/Effect关系。
- v10：否定将微小Procedure经RMSNorm/AdaLN约放大`14-20x`的terminal
  mixer，不否定ordered dual stream。
- Loom：否定无监督锚点的confidence/correspondence/gap作为必经中央变量，
  不否定teacher-visible relation。
- Recenter：联合否定“删Procedure DC且禁止Core value”，不分别否定
  centering或Core carrier。
- Core-Program：否定strict bilinear作为唯一factor content，不否定两种
  evidence的soft fusion。
- Prior-Innovation：中等证据否定硬编码的代数ownership，不否定语义
  prior与dynamic innovation作为非强制组成。
- Target-Spectral：否定强制正交/高rank，不否定target-first/rank-last QK
  identities和conventional coherent heads。

AP-ADR复用的是上述尚未被单独否定的能力，不是恢复任一整版旧
实现。

Git blob与formal artifact复核确认，上述八个post-v5 run的`as_step.py`、
`as_sampling.py`和launcher完全相同，且都只跑过full24/B20/fast400；没有任何
matched alternate-recipe cell。历史long-first仅发生在一次24-task聚合更新内部，
不是optimizer curriculum，不能与后来的六phase SERIAL混写。Prior-Innovation的
局部因果证据最弱，只能判完整组合失败。受控复核顺序固定为单-cycle
update-operator replay→有证据才跑cycle-normalized randomized group4→只有mean
Action被定位为容量瓶颈才移植8 anchors，不整套恢复旧实现。

## 3. 最终需要表达的计算

一条teacher video必须同时提供：

1. 任务中的对象、关系、完成状态和子目标语义；
2. frozen source policy在每个视觉状态下隐含的Action hypothesis；
3. 某个Action hypothesis之后观测到了什么task-grounded effect和change；
4. 哪些内容是38个真实policy targets共享的semantic carrier；
5. 哪些有序教学内容必须按target和rank coordinate分化；
6. 如何保留source policy已知有效的coherent高增益LoRA流形，同时让一条
   video的变化真正到达effective BA和policy action。

UCP已经具有content-selective target/rank routing，但raw-full24 exact50的same-task
effective-BA/fixed-action centered variance只有`.09008%/.01656%`；固定absolute
X只更换A/D时，BA/action只变约`2-3%/<1%`。所以缺失的不是更大
identity或更强router，而是“稳定语义载体”和“按坐标读取的有序教学内容”
不应在同一softmax、同一value向量或terminal norm中互相稀释。

## 4. 保留的信息墙和evidence frontend

输入和冻结前端不变：

```text
Q_text = text-only task-span evidence
M_f    = multimodal task-token evidence
G_f    = task-query patch-grounded evidence
X_f    = M_f + G_f
A_f    = native 50-suffix final Action hidden mean projected to width256
```

Writer仍不得读teacher action、state/proprio、reward、terminal、task ID、filename或
隐藏normalization。`A_f`是frozen policy latent，不是teacher action label。frame
stride固定5，一条video只生成一套完整rank-16 LoRA。

v7的8个sparse native Action anchors没有被单独否定，但首版不同时改变
Action capacity和compiler。v5.2的mean Action已经在old recipe下产生最强视频
特异性形态，也是SPG/UCP的共同基线。只有A-only路径、Action内部方差或
Action/Effect关系量明确将mean定位为最早瓶颈时，才做同topology的8-anchor
单因子反事实。

## 5. Mean-backed Semantic Core

对每个task-language token `l`，先在valid frames上计算：

```text
mu_l       = mean_f X_(f,l)
R_(f,l)    = X_(f,l) - mu_l
w_(f,l)    = softmax_f <Q_text_l, K(X_(f,l))>
Core_l^0   = W_mean mu_l + W_res sum_f w_(f,l) R_(f,l)
```

然后用两层task-token-axis bidirectional Transformer组合对象、关系和子目标。
frame order和frame ordinal不进入Core，所以任意frame permutation严格不变。

Core的责任是稳定语义基底，不是Action或程序顺序。raw mean避免Recenter式
semantic-basis starvation；task-selected centered residual允许它超越简单frame
average。这条路径是v6已验证的能力，不引入新gate或硬编码prior。

## 6. Outgoing Semantic Program

### 6.1 时间对齐

对每个真实可观测interval `i: f_i -> f_(i+1)`：

```text
A_i       = A_(f_i)
E_(i,l)   = G_(f_(i+1),l)
D_(i,l)   = G_(f_(i+1),l) - G_(f_i,l)
P_i       = [A_i, E_(i,1..L), D_(i,1..L)]
```

Program shape为`[F-1, 1+2L, 256]`。`A_i`是interval起点policy hypothesis；
`E_i`是紧接其后真实观测到的patch-grounded endpoint；`D_i`是对应change。
这个outgoing对齐避免v6的incoming one-frame ambiguity。

Program row `i`只在frame `f_(i+1)`到达后完整可见，其时间位置是endpoint
sampled-frame ordinal。Writer在完整teacher video后生成LoRA，因此使用该
endpoint不是未来泄漏；但Program prefix `0..k-1`必须严格只依赖前`k+1`帧。

`E_i`不是新的static bypass：它仅存在有方向的interval row中，并与`A_i/D_i`
共同经causal temporal keys读取。但它可能因raw amplitude大于`D_i`而主导
Program；所以Effect-only、D-only、A-only和attention mass是预声明的硬诊断，
不能在结果后用一个手工scale“修正”。

### 6.2 Axial contextual keys, physical raw values

两层axial Program每层依次执行：

1. 同一interval内沿`A/E/D`列全attention；
2. 每一语义列沿interval轴causal attention；
3. width256->1024->256 FFN。

frame endpoint、value type和task-token ordinal只进入Q/K。这两层产生
`ProgramKey`，用于让最终coordinate reader定位“哪个target/rank应读哪个
interval/type”。同时保留未经terminal normalization的`ProgramValue=P`。

最终reader使用：

```text
K = RMSNorm(ProgramKey) + type/order identities in QK only
V = raw ProgramValue
```

因此context决定路由，physical `A/E/D`幅度决定写入内容。小`D`不会像
v10那样因终端RMSNorm/AdaLN被固定放大，也不会像UCP那样与高能量
absolute X先融成一个value后再被reader稀释。

## 7. Asymmetric Dual Reader

### 7.1 真实LoRA拓扑

public LoRA仍先枚举38个真实policy targets：18层Q、18层V、
action-in和action-out；然后每target展开16个rank coordinates。

target identity为`[38,256]`，rank identity为`[16,256]`，不学习一个任意
`[38,16,256]`table。target/rank/type/frame identities都只影响Q/K，不进入V、
不直接加到factor content。

### 7.2 Core reader

每个真实target只读一次Core：

```text
c_m = Attention(
    Q = normalized target identity m,
    K = normalized Core semantic keys,
    V = raw Core content
)
```

`c_m`broadcast到该target的16个rank coordinates。Core的职责是提供同一真实
policy target共享的semantic carrier，不应凭rank identity制造无观测根据的
16个语义基底。这将literal symmetric 608个Core reads降为38个。

### 7.3 Program reader

每个`(target m, rank r)`独立读完整Program：

```text
p_mr = Attention(
    Q = normalized target identity m + normalized rank identity r,
    K = normalized contextual ProgramKey + type/order QK identities,
    V = raw ProgramValue
)
```

这608个reads允许不同policy layer/projection/rank coordinate选择不同教学片段，
但不通过rank loss或正交约束强迫它们分化。若source policy的有效几何需要
near-rank1 coherent writing，16个Program reads可以自然同向。

Core和Program必须使用两个独立attention normalizer。把两个memory合并成一个
softmax看似更简单，但Core token与`(F-1)(1+2L)`个Program tokens会直接
争夺mass；再修正该竞争只能重新引入没有证据的gate/bias/scale。

## 8. 无terminal amplifier的coherent factor compiler

对每个`(m,r)`：

```text
h_mr = concat(c_m, p_mr)                  # raw width512
row_mr = Linear_2(GELU(Linear_1(h_mr)))   # 512 -> 256 -> output width
```

八个bias-free heads分别生成Q-A、Q-B、V-A、V-B、action-in-A/B和
action-out-A/B。heads在所有targets/ranks上共享，保留conventional coherent
writing。每个head最后Linear严格zero-init，所以step0生成量为0；加到
template A / zero B后，public LoRA逐tensor严格是functional identity。

明确不做：

- Core/Program终端RMSNorm、AdaLN、gate、scale、sum或multiply；
- cross-target/global coordinate mixer；
- B-only residual、static bypass、confidence、gap、null token；
- rank diversity、正交、谱均匀、stable-rank loss；
- 第二套LoRA、多video平均、多LoRA平均或checkpoint融合。

没有强制Core和Program都必须非零。strict双必要已在Core-Program被证伪；
软职责应由Core-only、Program-only、fixed-Core/vary-Program和相反反事实测量，
不由代数结构硬编码。

## 9. 实现owner和生命周期

保留一条canonical Writer path：

- `video_program.py`：仅负责`Q_text/M/G/A`的frozen-policy evidence frontend；
- 新`semantic_core.py`：负责mean-backed frame-set Core；
- `semantic_program.py`：替换为`A/E/D`raw values和causal axial contextual keys；
- `program_compiler.py`：替换为asymmetric dual readers和512->256 factor input合同；
- `model.py`：只编排生命周期、public target mapping和template identity；
- 现有Meta-LoRA、functional LoRA、sampler、checkpoint、validation和evaluator全部复用。

UCP旧model/config/schema不保留可选Writer分支，历史由Git、frozen worktree和正式
artifacts保存。AP-ADR使用fresh incompatible architecture/config/checkpoint schema，
不从UCP、SPG或v5.2 Writer checkpoint warm-start。

`model.py`当前约455行，`video_program.py`约613行；新职责不得继续堆入
这两个owner。参数budget软上限仍约corrected rank-128 Source-SFT的
10.3M。纸面估算不作正式authority；实现后必须用real module enumeration
报告总数和frontend/Core/Program/readers/factor分块数。

canonical实现的真实module enumeration为：

| owner | parameters |
|---|---:|
| text/vl/action Meta-LoRA | 2,469,888 |
| language/patch/action evidence projections | 983,552 |
| mean-backed Semantic Core | 1,836,544 |
| outgoing axial Semantic Program | 1,838,592 |
| asymmetric dual readers | 409,088 |
| eight 512->256 factor heads | 2,703,360 |
| **total** | **10,241,024** |

它低于10,297,344的软参考上限56,320参数。`model.py`只增长到约483行；
新增的`semantic_core.py`独占frame-set Core责任，Program和compiler分别保留
单一owner。UCP专用config/schema/analyzer及其机制专用测试已从active tree退役，
正式历史与内部结果继续由Git和immutable artifacts保存。

## 10. 训练决策

### 10.1 首轮保持raw-full24

首个formal hour使用：

```text
raw full24 mean
x B20 phase-stratified independent same-task queries
x one video / one complete LoRA / task visit
x cost-balanced long-first
x warmup17 + cosine fast-decay400, peak 3e-4, floor 1e-5
x fresh macro0->200, checkpoint every25
```

原因不是新recipe已被证明最优，而是：

1. 与UCP raw-full24/fast只改topology，保留最干净的同recipe比较；
2. v6的最强single checkpoint `143`与该bundle相容；
3. SERIAL-4只提高UCP best 4分且未改善漂移，不足以成为默认；
4. 首轮同时改架构和recipe会再次丧失可识别性。

训练仍记录24x24 task-gradient Gram、block Gram、raw mean energy ratio、
candidate-negative task fraction、CountSketch方向、clip、Adam moment和single-video
noise。不做CP-24投影：SPG已经证明消除negative pair不会自动恢复漂移。

### 10.2 有序的matched recipe反事实

1. AP-ADR full24-fast vs UCP full24-fast：首先识别topology。
2. 若AP-ADR内部主路已工作但absolute早熟/偏弱，运行同topology
   full24-slow2000：只改scheduler。
3. 若仍有innovation dilution和漂移，才实现cycle-normalized randomized
   group4：phase与frame cost解耦，LR/6，moment/decay按exposure匹配，
   scheduler每24-task cycle只推进一次。它仍是update-granularity bundle，
   不冒充单个Adam因素。
4. 若历史识别仍会改变决策，再补v5.2 full24/B20/slow2000。

不使用held functional loss选checkpoint或覆盖closed-loop门。新的ten-step
endpoint metric在通过预声明no-gradient关联门之前不进训练；即使通过，
首先也只作held monitor，需要另行验证精确gradient方向和显存可行性。

### 10.3 Endpoint10预注册关联门

在任何endpoint数值生成前，候选和判据固定如下：

- `v5.2-new` macro150/200/350/400；
- UCP raw macro50/100/150/200；
- v6-fast macro50/100/150/200/250/300/350/400；
- v5.2-old step900；
- v6-old step900。

共18个single checkpoints；每个都必须使用同一paired correct400 state/video/RNG
authority和sealed512 validation rows。endpoint sampler固定正式rollout的无autocast
exact ten-step Euler，输入batch删除ACTION；teacher action只作为no-gradient误差
target。主指标唯一固定为
`quality = -rollout10_executed5_valid_normalized_mse`，因为正式policy每次只执行前5步。
full50/prefix10和ten-grid teacher-bridge flow MSE全部是secondary，不能替代主指标
通过门。

主指标成为可信held monitor必须同时满足：

1. 18候选task-balanced quality对correct400的Spearman `rho >= .45`；以sealed
   panel payload SHA派生固定seed、做100,000次candidate-label permutation的双侧
   `p <= .05`；
2. 对三个多checkpoint family分别去family均值后合并，Pearson和Spearman都
   `>= .30`，且三个family各自Spearman同号为正、至少两个`>= .30`；
3. 两个150-video/task等曝光recipe方向都正确：v5.2-old step900必须优于
   v5.2-new macro150，v6-fast macro150必须优于v6-old step900；
4. 按8个validation tasks分别计算跨候选Spearman，其中位数`> 0`且至少6/8
   non-negative，防止一个suite或易task单独制造aggregate相关。

若任一条失败，endpoint10只保留为负诊断，不能选checkpoint、改loss或进入训练。
即使全部通过，也先只记录AP及后续模型的held monitor；在把它用于gradient前，
还必须另做prospective checkpoint排序复验、精确gradient方向审计和B20显存验证。
不得按结果调整阈值、改主指标或用secondary metric救回失败门。

候选表在任何endpoint数值生成前做过一次文字合同修正：v5.2-new原误写为
`50/100/150/200`，但该formal实验按预先批准合同实际做paired correct400并封存的
四点始终是`150/200/350/400`。候选总数仍为18，recipe direction仍使用macro150；
该修正只让诊断绑定既有formal panel，不引入outcome-based候选选择。

## 11. 实现前vertical path

正式launch前最少必须通过：

1. `Q_text/M/G/X/A/E/D`的shape、dtype、mask和public split信息墙；
2. `A_f`与`G_(f+1)`/`G_(f+1)-G_f`的outgoing interval对齐；
3. Core frame permutation invariance；
4. Program causal prefix invariance，改未来帧不影响早期keys；
5. frame/order/type/target/rank identities不进入任一value路径；
6. Core reader严格`38x256`，Program reader严格`38x16x256`；
7. 38 targets x 16 ranks的A/B形状、transpose和全量public target完整；
8. step0 public LoRA逐tensor等于template A/zero B；
9. frozen source policy没有trainable parameters；
10. step1 factor-final与step2起frontend/Core/Program/readers/factor梯度finite且可达；
11. checkpoint manifest、sampler/RNG/data cursor和exact resume；
12. 最长105-frame真实video的B20三个完整macro profile。

B20只在真实OOM或连续非有限时直接降B16；不扫描B17-B19/B21。

## 12. 内部证伪合同

### 12.1 两路职责

对correct/same-task-other/wrong/shuffled/reversed逐task报告：

- `Q_text/M/G`、Core mean/residual/slots；
- raw `A/E/D`、两层ProgramKey和type/time attention；
- target Core reads、target/rank Program reads、raw concat和factor output；
- public A/B、effective BA和fixed-query policy action；
- 每层relative L2、cosine、RMS、上一接口到下一接口的保留率。

反事实至少包括：

- Core-only、Program-only、A-only、E-only、D-only、A+D、E+D、full；
- fixed Core/vary Program和fixed Program/vary Core；
- raw A/E/D分别按`.5/1/2`缩放，测量BA/action响应；
- Core mean删除、Core centered residual删除；
- target identity和rank identity permutation，只用effective BA/函数行为作结论；
- shuffled/reversed必须重排真实帧后完整重算，不复用correct keys。

### 12.2 有效LoRA几何

优先使用gauge-invariant量：effective BA norm/Gram/singular spectrum、fixed-query
action和完整函数行为。逐task报告：

- q/v/action energy ratio、per-layer CV、cross-layer BA cosine；
- stable rank、entropy rank、top singular energy、rank90/rank99；
- 固定坐标energy participation、component pair cosine/负pair、B-column cosine；
- same-task 50-video centered variance/sample energy和task-mean energy；
- 视频变化中scale-like和orthogonal-direction比例；
- fixed-query action RMS、relative L2和direction change。

不把高rank、低跨层cosine或特定LoRA norm本身当成成功。Target-Spectral
已经证明强迫这些数字可以同时摧毁closed-loop manifold。

### 12.3 预声明失败量

以下任一项成立都是中央责任失败：

- Core-only几乎复现full BA/action，Program成为形式分支；
- fixed-Core Program变化长期`<=3%` BA且`<=1%` action；
- same-task BA centered variance/sample energy`<=0.15%`，重演UCP；
- E-only主导，A/D在Program中无法改变写入；
- `.5/1/2`raw幅度几乎不改变输出，说明幅度又被擦除；
- 微小D经终端路径爆炸性放大，重演v10；
- 去Core mean几乎不变，但模型声称需要共同carrier；
- Program上游对order强敏感，到BA/action仍被压到近零；
- functional loss下降而correct400/breadth持续不改善。

“内部主路可信”的最低方向量是：fixed-Core Program在多个tasks上造成
`>=5%` effective-BA变化，same-task BA variance`>=0.30%`，且没有任一分支
单独复现full输出。这些只是进入续训/反事实的内部门，不替代
closed-loop表现。

## 13. 一小时门和后续决策

fresh macro0->200后严格评测50/100/150/200的paired correct400，与：

- UCP raw-full24 `82/117/100/110`；
- v6同期macro200约`133`；
- v5.2 task-complete macro200 `91`；
- corrected Source-SFT约`109`

比较。默认继续到第二小时需要best `>=133`，或best `>=125`且breadth
`>=6`、至少四个tasks共同增益、右端趋势明确并有工作中的内部主路。

若四点全部`<=120`、breadth`<=5`、主要是task轮换、或Program变化到BA/
action的传递失效，不续到400、不跑昂贵五臂；先完成exact50无rollout
内部分析并找最早失效接口。

若进入第二小时，exact-resume到macro400，评估有信息量的250/300/350/400，
选一个single-checkpoint winner。只有absolute处于强水平时才跑
correct/same/wrong/shuffled/reversed各400的正式五臂。期望方向为：

```text
same approximately correct
correct - wrong/shuffled/reversed >= 20 overall
margin from at least four tasks
```

这些不是为了制造漂亮margin的损失函数。absolute低而specificity强仍是
v10式失败；absolute高而Program不起作用仍是v4/UCP式失败。

## 14. 可证伪的后续能力，不预先恢复

只有内部量将它们定位为最早瓶颈时，才做单因子对照：

- v7的8个Action anchors：仅在mean Action capacity失效时；
- v8式局部Action x Effect relation：仅在axial keys不能形成有选择性关系时，
  且不早池化为单event；
- v10式interleaved dual stream：仅在当前列轴+时间轴分解丢失关系时，
  不恢复terminal amplifier；
- Loom中teacher-visible relation：可以作为Program key的可识别内容，但不恢复
  confidence/gap authority；
- v5.2 direct Core/Procedure concat：仅作compiler-only简化ablation，不作为
  下一整体模型，因其Procedure没有显式endpoint Effect/change。

任何后续改动都必须重新回答：需求、现有组件为什么不足、最小缺口、
历史证据、预期内部量变化和如何证伪。不允许在失败checkpoint上临时堆
gate、scale、bypass、confidence、谱约束或第二套LoRA。

## 15. 实现与live seal

canonical实现于`8306549`集成main，真实module enumeration为`10,241,024`；
source policy trainable参数0，public 76 tensors/38 targets/rank16与step0 identity
合同全部通过。受影响的Writer/model/training/evaluation focused回归为`67 passed`，
实现worktree的完整相关回归为`203 passed`，architecture gate无hard violation。

GPU4–7、NUMA node1上的longseed172真实profile完成三macro：

```text
root  /data/ymdai/outputs/ember/pi05_as_writer_ap_adr_profile_b20_longseed172_r4_8306549_20260801
steps 20.567 / 18.717 / 18.644 seconds
peak  77,227,462,656 allocated / 83,523,272,704 reserved bytes
data  72 one-video conditions / 1,440 independent queries
```

首macro真实包含task38/demo36的105 sampled frames；每步24 unique tasks且rank内
long-first。identity step只有zero-final factor路径有梯度，step2起semantic
frontend、Core、Program、compiler和factor全部finite非零，符合预声明生命周期。

formal teacher seed`20260722`另在同一frozen commit完成fresh0→1，再从step1
exact-resume到3。三步loss为`.157168/.150686/.148092`，gradient norm为
`.009444/.014368/.100302`；step1 manifest、Writer、trainer和四rank state的
size、mtime与SHA在resume后逐项不变。seal commit `7dffb6f`已push。

正式首小时从clean detached `7dffb6f` fresh identity启动macro0→200，未继承
任何profile/smoke权重：

```text
tmux   ember-ap-adr-formal-7dffb6f
frozen /data/ymdai/.codex/worktrees/EMBER-ap-adr-formal-7dffb6f-20260801
root   /data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801
log    /data/ymdai/logs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801.log
```

正式训练自然完成200 macro，run summary为96,000 queries、4,800 one-video
conditions、每task 4,000 queries/200 visits、wall `3898.217s`；200行metrics全部
finite，validation/test action读取与test video读取为0。macro50/100/150/200的
paired correct400已分别挂在GPU4/5/6/7，tmux
`ember-ap-adr-correct400-7dffb6f`；四个prepared合同均为400 states、50 videos
无放回、36 long-first shards、6 replicas/6 Writer generators。训练tmux已自然
退出，后续main改动不得影响frozen checkpoint或评测。

同曝光macro175优化动力学进一步限定Program小梯度：AP最后25步Program raw
gradient仅为UCP的`.856%`，但Adam update RMS和累计位移仍为UCP的`71.18%/85.42%`，
所以whole-block starvation不成立。最窄风险是两层temporal Q/K的二阶矩只有Adam
eps的`13.5–17.1%`、累计位移约UCP的`18–26%`；当前只能标记
`temporal Q/K→contextual key→ProgramReader K`局部routing-key starvation，必须由
trained-vs-initial key、attention和BA/action反事实结合closed-loop证伪，不能仅凭
参数梯度宣判架构。

## 16. 正式一小时裁决与最早失效接口

四个paired correct400最终为：

| macro | correct | breadth | per-task |
|---:|---:|---:|---|
| 50 | 91 | 6 | `[7,0,1,26,30,26,0,1]` |
| 100 | 81 | 6 | `[13,0,2,33,28,4,1,0]` |
| 150 | 94 | 5 | `[18,1,0,37,29,9,0,0]` |
| 200 | 91 | 7 | `[19,1,2,31,33,3,0,2]` |

相邻点gained/lost为`33/43`、`36/23`、`25/28`，没有多个task共同稳定增长；
best94也显著低于UCP raw117、SERIAL121和v6-fast同期。故不进入第二小时、不做
五臂。这个结果判的是`AP topology x raw-full24/fast`组合；以下反事实才负责把
架构责任进一步局部化。

PI05 sampler会永久把attention backend从SDPA切为eager。该副作用不进入训练，
正式evaluator也先生成全部Writer cache再rollout，但会污染交错Writer capture和
action probe的内部analyzer。`5d93af3`加入scoped backend restore后，8个validation
tasks的逐层、effective BA、fixed action重放误差全部严格为0。最终root的
analysis/summary SHA为`d42fc4eb...bc2b`/`f2c572c5...e682`。

修复后跨task均值为：

| 干预/条件 | Program block2 relL2 | Program read relL2 | BA relL2 | action relL2 |
|---|---:|---:|---:|---:|
| same-task-other | 1.1051 | .03210 | .03005 | .01668 |
| wrong | 1.3069 | .15185 | .14540 | .02926 |
| shuffled | .09066 | .02790 | .002689 | .002200 |
| reversed | .07112 | .03787 | .003903 | .002160 |
| temporal contextual keys reversed | -- | -- | .000521 | .001944 |

ProgramReader normalized entropy约`.9037`、top mass约`.0106`。Contextual Program
形成了强same/wrong变化和可测order变化，却几乎只通过高熵attention权重影响输出。
A/E/D raw value反事实进一步给出决定性定位：Effect-only距full BA仅`.00821`，
Action-only/Change-only距full为`.2761/.2832`；固定full key后数值仍为
`.2789/.2827`。Effect缩放0.5/2带来BA `.141/.289`变化，Action或Change缩放最多
`.008/.001`。所以当前中央职责错误是：causal axial stack只拥有K，raw value中的
Effect DC直接垄断写入。它不是temporal Q/K简单幅度不足，不能通过放大、gate或
B residual修补。

同时，Core-only距full BA/action`.283/.228`，Program-only距full`.961/.494`；
两路都必要。删除Core mean会改变BA`.834`，删除centered residual只变`.0128`，
mean-backed semantic carrier仍应保留。下一整体结构的最小职责变化是让causal
contextual Program本身成为target/rank reader的value content，同时保留独立Core
read、coherent factor heads和信息墙；它不同于SPG的Core加法/global mixer，也不同
于UCP取消Core carrier的单流。该结构必须另立fresh schema并重新通过一小时门，
不能在AP checkpoint上热补。
