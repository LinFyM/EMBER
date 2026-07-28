# EMBER v5.2上限测定与v6推进交接

最后更新：2026-07-28 UTC。

本文是focused AS Writer与开放式绝对性能探索的完整跨session恢复入口：
集中保存研究北极星、从最早Writer到v5.1的证据链、失败结论和当前继续训练
合同，使新session不依赖历史聊天也能接手。正在运行的v5.2精确架构以
`docs/action_forecast_writer_v5_2_design.md`为准；owner确认的下一条fresh
架构以`docs/action_forecast_writer_v6_design.md`为准。v5/v5.1只作provenance，
v5.3设计和原型是v6的直接provenance。
长期项目边界以`AGENTS.md`和`docs/execution_brief.md`为准。本文中的step和
进程状态是交接快照，接手后必须先做只读实时复核，不能据此重复启动任务。

focused AS/RL完成或本文不再承担跨session恢复作用时，更新或删除本文，不能让
过期运行状态长期成为平行authority。

## 0. 2026-07-28实时恢复摘要

- v5.1低LR phase100/300/600/900无放回correct400为
  `119/115/123/104`，均低于原best `127`；paired artifact SHA256
  `f52c9b78...543`。v5.1已停止。
- v5.2真实profile已完成：B20支持step1→3 exact-resume；B21连续3步finite，
  global84，max allocated/reserved
  `80,283,666,944/83,892,371,456` bytes；B22四rank对称OOM。
- 活动正式合同为GPU4–7、4-rank、F32/B21、frame stride5、
  `max_frames_per_encoder_call=32`。fresh step0→900已结束；correct400曲线
  `72/79/120/132`，step900为右端observed-best。
- step900 full400五臂为`132/138/74/82/83`。same鲁棒，correct显著优于
  wrong/shuffled/reversed；内部Core/Procedure/LoRA/action链也没有v4旁路。
- owner决定先沿原版v5.2训练范式从step900 exact-resume测清上限；当前tmux
  `ember-v52-resume-1800`只把1800当本次约一小时segment边界。结束后GPU4–7
  分别评测step1200/1400/1600/1800，若右端仍上涨则继续，不机械把1800当峰值。
- owner已把下一fresh架构从v5.3提升为v6。v6同时采用task-grounded adjacent
  visual transition、`mean backbone + task-selected centered residual`
  Semantic Set aggregation和hidden=256 factor heads，总参数
  `10,775,296`；设计封存在`docs/action_forecast_writer_v6_design.md`。
  当前v5.3 branch实现只作v6实现起点，尚未修改代码/config或启动v6 profile。

## 1. 当前focused objective

owner已经解除此前“特异性后停止”和“AS过门后才可探索RL”的临时边界，并创建
新的session-local Goal：在EMBER核心映射和硬约束内，自主训练、诊断、修改
架构并可做可判别的低成本RL探索，尽可能提高correct-video absolute performance。
推进期间不需要逐项审核；只要absolute尚未达到可信满意水平，或提升带有
v4-shuffled式逻辑漏洞，就不得轻易停止。

v5.1已经完整训练至step1800并完成step1400五臂、LoRA-scale扫描和额外900步
低学习率稳定阶段。原best仍为`127/400`，低LR四点
`119/115/123/104`均未提高它；reversed与correct等价的逻辑缺口也未解决。
因此v5.1已经停止。v5.2 Task-Queried Patch Grounding当前只继续测定充分训练
上限；v6 Task-Grounded Semantic Set + Visual-Transition Procedure是默认
下一fresh Writer。
新session仍须先执行第9节的只读命令核验Git、tmux、GPU4–7和artifact，但
不应重新等待owner批准。

## 2. EMBER 的研究北极星、任务和信息合同

### 2.1 核心思想

EMBER要回答的不是“视频能否改变LoRA”，而是：

> 一个已经具备基本机器人视觉、语言和闭环控制能力的frozen VLA，能否只看
> 正确任务描述和一条没有action标注的teacher视频，像人一样理解这个任务要
> 如何完成，并把这种高层理解一次性编译成一套task-specific LoRA，使机器人
> 面对不同初态亲自执行时表现更好？

核心映射：

```text
task language + exactly one action-hidden teacher video
    -> high-level task understanding / task program
    -> complete task-specific rank-16 LoRA
    -> frozen π0.5-LIBERO source policy closed-loop execution
```

视频不是执行时每一步重复输入的context；Writer在rollout前生成一次静态LoRA，
policy之后依靠当前robot observation/state闭环执行。高层理解应包括：

```text
相关对象与关系
目标终态和约束
任务按什么阶段推进
哪些闭环策略偏置能把当前状态推进到目标
```

它不应把teacher某条episode的绝对三维路径、速度、抓取角度、frame phase或
具体初态几何直接写成部署到另一初态的静态controller。

one-shot是重要合同，不因训练中需要跨示范抽象而放弃。当前假设是：同一video
生成的LoRA同时接受宽action batch监督；不同task visits轮换teacher video；
共享Writer的batch/SGD梯度会强化跨初态、跨video稳定的高层规律，使单条demo
低层细节难以持续解释监督。这个假设必须由实验验证，不能由文档直接宣布成立。

### 2.2 Benchmark与固定split

目标benchmark是四个标准LIBERO suites，共40 tasks：

| suite | train local IDs | validation local IDs | test local IDs |
|---|---|---|---|
| `libero_spatial` | 0,2,4,5,7,9 | 1,3 | 6,8 |
| `libero_object` | 2,4,5,6,8,9 | 1,3 | 0,7 |
| `libero_goal` | 0,1,2,5,8,9 | 3,6 | 4,7 |
| `libero_10` | 4,5,6,7,8,9 | 1,2 | 0,3 |

- development为24 train / 8 validation / 8 test；
- exact seal在`configs/libero_24_8_8_v1/protocol.json`；
- 不得根据outcome改变task IDs；
- development选定方法后把8 validation合入source，形成32 source / 8 test；
- 第一轮只跑一个training seed。

共同source foundation：

```text
generic lerobot/pi05_base
-> 对LIBERO-90与目标40做specification-only完整overlap audit
-> 排除19个exact semantic/composition重合task，保留71 tasks
-> 71 tasks × 每task50条success action episodes联合full-SFT
-> fresh 1,000 steps后冻结raw π0.5-LIBERO source base
```

禁止使用已经读过目标40 actions的`pi05_libero`。source-only normalization与
source base一起冻结；所有AS/RL Writer、Source-SFT和后续方法从同一base开始。
generic base的`0/400`只说明它不能直接执行LIBERO；新source base的40-task
快速screen为`46/320`且跨13 tasks/四suites，证明已有基本interface competence，
不代表EMBER成立。

### 2.3 Information wall与比较对象

Writer每次只允许读：

- 正确task language；
- 恰好一条`obs/agentview_rgb` action-hidden teacher video；
- 视频真实帧顺序、valid mask和采样ordinal。

Writer不得读teacher action、proprio/state、reward、terminal/success、
task/suite ID、filename/demo label、normalization、policy outcome、object pose
或其他simulator privileged state。AS functional policy forward可以读一条
action episode的正常observation/state/language/action target，但这些只属于
frozen policy loss，不能成为Writer输入。

主要比较：

- frozen source base；
- rank-128 Source-SFT shared LoRA：只看source actions，held不看video；
- AS-Writer one-video task LoRA；
- 独立cold-start RL-Writer one-video task LoRA；
- correct/same-task-other/cross-suite-wrong/shuffled/reversed controls。

四卡Source-SFT observed-best `108/400`与旧八卡`122/400`是背景；122不是必须
超过的硬门。旧Action-Forecast correct best约`125/400`是主要最低性能参照，
v4 shuffled `148/400`是目标区域。

## 3. 从最早版本到v5的证据链

新session必须理解每一代解决了什么，也必须理解它没有解决什么。

### 3.1 Pooled AS Writer

做法：每帧PaliGemma视觉tokens全局平均，整段视频压成4个episode tokens，
learned parameter queries与带bias factor heads直接生成LoRA。

结果：best `119/400`，source base `48/400`，但correct/wrong仅`119/115`；
换video时effective LoRA相对变化只有`7.52e-6`。

教训：functional policy本身看到正确language和当前observation，Writer可以
生成近似input-independent的公共/domain adapter而忽略视频。高absolute不等于
视频理解。

### 3.2 Conditional Spatial Writer

做法：保留`4×4`空间grid，使用condition-only attention、bias-free identity
heads并切断静态query residual；历史上还用了normal/full-language/generic
paired contrast。

结果：充分训练best correct/wrong约`99/55`。视频因果性明显增强，但absolute
从119下降。

教训：特异性与competence是两个门；contrast可以人为定义wrong应更差，却不能
证明架构自然理解视频。owner因此永久拒绝把contrast/order/margin loss作为
当前修复。

### 3.3 Action-Memory Writer

做法：让Action Expert memory tokens读取每帧图文prefix，利用source VLA的
视觉—语言—动作先验，而不是任意池化视觉后直接出LoRA。

结果：代表轨迹best约`105/400`或`91/400`；wrong/single-frame会改变LoRA，
但reverse/shuffle差异远小，视频主要被当作无序状态集合。

教训：让Action Expert“看视频”本身不等于理解过程；frame content有特异性，
时间顺序仍可能被忽略。

### 3.4 Temporal-RoPE Writer

做法：在Action-Memory上增加真实frame index的1D RoPE和4个
condition-only memory queries。

结果：step400/500=`108/98`；wrong/same/reverse/shuffle effective-LoRA相对
差约`0.2267/0.0403/0.00937/0.00699`。

教训：显式位置容量或两层Temporal不是充分条件。positive functional loss仍可
只学“这是什么任务/场景”，不学“如何一步步完成”。

### 3.5 Action-Forecast v1

做法：每帧用imagined state、两个Meta-LoRA和完整10-step flow产生`50×7`
future-action forecasts；跨frame按假想absolute robot time对齐，用Plan表示
最新最可信预测、Revision表示同一未来时刻的预测修正，再经Temporal和320
LoRA queries生成完整LoRA。

结果：correct曲线充分探索，observed-best约`125/400`；wrong `67/400`，
说明任务视频内容很重要；但shuffle/reverse约`121/124`，几乎没有顺序优势。

教训：Action-Forecast提供了很强能力和内容特异性，但“forecast是否真的有
teacher future-action语义”未经监督校准；后续路径也会丢失或误用顺序。

### 3.6 Action-Forecast v2与Belief-v3

v2切断static LoRA query到factor head的旁路，并将Revision改为
content-only directed read，避免absolute action/statistics覆盖修正方向。

Belief-v3进一步使用：

```text
Belief_u = concat(Plan_u[128], Revision_u[128])
Revision_u = stopgrad(raw residual RMS) * RMSNorm(direction)
```

不引入人工temperature/分位数强度超参数；Temporal/decoder均routing只进Q/K、
raw content进V/residual。

结果：Revision内部顺序差异已恢复，但Plan/Revision中大量有价值的跨时刻共同
成分被Temporal放大；effective LoRA的reverse/shuffle相对差又降到
`0.000297/0.000169`。反事实减去temporal mean可恢复差异，但会粗暴删除任务、
场景、物体和稳定动作语义，因此被否决。

教训：不能把“所有时间恒定信息”当作敌人。稳定语义是任务理解的重要部分；
架构应显式区分稳定Core与有向Procedure，而不是删除均值来换特异性。

### 3.7 Visual-State Action-Forecast v4

做法：32个native state tokens；初始锚点加非递归相对变化；保留两个Meta-LoRA、
7D forecasts、absolute-time Plan/Revision、Belief、两层Temporal和
content-only decoder。它解决了早期virtual state跨帧几乎不变的问题，并在
step75把reverse/shuffle差异传到effective LoRA。

充分结果：

```text
step825 fixed400
correct           109
same-task other   104
cross-suite wrong  99
shuffled          148
reversed          126
```

这不是测量波动：同一fixed panel中shuffled显著优于correct，主要收益集中在
Object-1/Object-3。固定首帧只打乱后续仍为136，首帧anchor不是主因。

全面因果诊断得到的根因链：

```text
同task但独立video/action的positive AS目标
    -> 不识别每帧应对应什么teacher future action
32-token visual-state不是必经瓶颈
    -> raw image + VL/Action Meta路径绕过它
Meta路径随训练学成demo-specific frame phase / translation code
    -> latest-is-best与Revision是真实误差修正的假设反而恶化
absolute-time Plan/Revision把未经校准的低层code解释为controller
    -> 对新初态产生错误抓取点/到达时机/运输几何
```

直接证据：

- neutral visual-state在step825只使forecast变化约`0.855%`；
- visual-state主要预测progress，robot state/action probe很弱；
- 8种random permutations产生高度共识的LoRA delta，不是幸运shuffle；
- shuffle主要去相关frame phase与前三维translation；
- 只重排translation在Object-1/3得到`79/100`，true shuffle为`82/100`，
  correct为`49/100`；
- correct常抓起或运送错误物体，shuffle改变平面到达点后让已有
  language/object semantics重新主导；
- shuffle AS loss略高且delta与AS gradient近乎正交，所以它不是更好的offline
  descent step。

最关键启示：

> shuffle没有创造更多高层语义，也不是证明顺序不重要。它结构性破坏了
> correct-order路径中有害的低层phase/translation controller，使原本已存在的
> 高层任务语义重新主导。新架构应先保护这部分稳定语义，再让正确视频中真实、
> 速度无关的高层阶段顺序提供额外增益。

### 3.8 v5早期训练合同与当前切换

Semantic Core + Causal Procedure v5已证明结构上能把normal/shuffle/reverse
差异传到Procedure、LoRA和policy function。旧`N=4 per-action`训练在step40/120
的五臂fixed400分别为`45/52/52/51/51`和`65/59/57/61/65`；模型仍在学，但每步
实际生成约24–32套LoRA、约一分钟，absolute太低且训练远未充分。

短暂的共享四视频分组把每步降到约10秒，但owner最终选择更简单、更通用的
单视频完整action-batch合同：每rank一条video只生成一套LoRA，尽可能大的action
batch共同约束它；跨step轮换video。这把训练降到常规约3–4秒/step，并保留
one-shot输入和大batch平均低层噪声的核心假设。

该未知量现已由第7–8节的正式结果回答：v5恢复到`115/400`并建立显著
wrong-video语义效应，但没有建立correct-order增益；它因此退役，不能再把
“也许继续训练会自然解决”当作未检验假设。

## 4. v5端到端架构：每个模块的输入、输出与设计初衷

完整公式与实现合同必须逐字阅读
`docs/action_forecast_writer_v5_design.md`。本节是接手时必须掌握的心智模型。

### 4.1 输入与teacher prompt

```text
一条agentview RGB video，stride=5，保留final frame，长度T可变
+ 正确task language
```

每帧teacher prefix精确为：

```text
image tokens
+ "Task: {cleaned_task};\nAction: "
```

teacher侧彻底删除state/virtual-state。原因：observer/human teacher未来没有
可用robot proprio；v4 visual-state又被证明不是必要owner。execution policy
仍正常读取自己的observation、language和真实state。

### 4.2 模块A：Visual-Language Frame Encoder

输入：第`t`帧`F_t`与language tokens。
计算：frozen PaliGemma完整图文prefix前向，加trainable identity-init
VL Meta-LoRA（18层q/k/v/o，rank4）。
输出：完整prefix final hidden中原256个image positions：
`H_img[t] ∈ R^[256,2048]`。

设计意图：

- 这些不是额外的“Gemma image output”，而是language已经通过attention改写后
  的image-position final hidden；
- VL Meta-LoRA保留是为了适配无state、observer/第三人称和未来人类视频；
- 初版不把language-position hidden作为独立value，避免
  `language -> task LoRA`绕过视频；language仍通过PaliGemma和Action Expert
  prefix KV完整生效；
- 必须同时验证换language和换video都能改变下游，不能把防捷径误写成忽视语言。

### 4.3 模块B：Permutation-Invariant Semantic Core

输入：`H_img[t]`。
计算：原生`16×16`image grid做无参数`2×2`平均池化为`8×8`，得到64 tokens/frame；
共享bias-free `2048->256`投影后，把所有有效frame-spatial tokens展平为
`C(V) ∈ R^[(64T),256]`。Core路径没有frame ordinal、temporal RoPE、first/last
标签、causal mask或任何position value。
输出：对同一帧集合的任意permutation只发生K/V行置换的Core memory。

设计意图：

- 保存v4 shuffle已经证明很强的稳定信息：对象、关系、场景、出现过的交互状态、
  初/终态和整体操作类型；
- 通过cross-attention的集合性质，在架构上保证
  `Core(normal) == Core(shuffle/reverse)`，不让错误时序分支摧毁高层底座；
- Core不是完整任务理解，顺序知识由Procedure独立拥有。

### 4.4 模块C：Native Action-Expert Semantic Probe

输入：每帧PaliGemma prefix KV，加一份全任务/全视频共享的fixed native
Gaussian suffix`[50,32]`，固定flow time`t=1`。
计算：frozen Action Expert加trainable identity-init Action Meta-LoRA
（18层q/k/v/o，rank8）只前向一次；不做10-step flow integration，不调用
`action_out_proj`，不输出7D action。取final-norm后的50个suffix hidden
`[50,1024]`，均值后经共享bias-free `1024->256`投影。
输出：每帧一个robot-semantic interaction token
`s_t ∈ R^[256]`。

设计意图：

- fixed noise suffix只是π0.5 pretrained Action Expert读取prefix所需的原生
  readout basis，不是50步forecast或随机轨迹；
- 保留50是为了不先破坏pretrained token interface，且不增加参数；只有真实
  profile证明它是瓶颈才可做独立长度ablation；
- 对50 slots求均值主动删除lead/future-clock对应，只保留“假如机器人是视频中
  teacher，这一帧涉及什么机器人交互”的整体语义；
- Action Meta-LoRA保留是为了observer/human→robot interaction semantics，
  不是让它继续输出具体teacher translation。

### 4.5 模块D：Variable-Length Causal Procedure Encoder

输入：`S=[s_1,...,s_T] ∈ R^[T,256]`与valid mask。
计算：2个独立pre-norm global causal Transformer blocks，8 heads，
FFN`256->1024->256`；signed frame ordinal只通过1D RoPE进入Q/K，
V和residual只传content；token`t`只能读取`1..t`。
输出：保留全部有效位置的`E ∈ R^[T,256]`。

设计意图：

- 表示“交互阶段如何有向推进”，不是teacher逐点路径；
- causal attention是时序最自然的结构，但历史RoPE已证明“有顺序容量”不等于
  自动使用高层顺序，因此它必须建立在非7D、非forecast的interaction语义上；
- 长度随视频增长，不把短任务和一天长任务都强塞进固定8个event tokens；
- 当前LIBERO直接全局causal；未来只有真实`O(T^2)`瓶颈时才引入causal
  window/hierarchy。

### 4.6 模块E：320个routing identities

组成：

```text
18 expert layers × rank16 = 288
action_in rank rows        = 16
action_out rank rows       = 16
total                      = 320
```

query/module/layer/rank identity组合为`R ∈ R^[320,256]`。R只进入attention
Q/K，不进入V、residual或factor heads；动态content从严格`Z_0=0`开始。

设计意图：静态identity只决定“哪个LoRA row应该读什么”，不能自己生成公共LoRA。
这是对初代公共adapter与v1 static-query residual捷径的结构性修复。

### 4.7 模块F：Core-to-LoRA Compiler

输入：Core memory`C(V)`、routing`R`与零content queries。
计算：1个content-only block：

```text
cross-attention: Q=R+Norm(Z), K=Norm(C), V=raw C
self-attention:  Q/K=R+Norm(Z), V=Z
FFN content-only
```

输出：`Z_C ∈ R^[320,256]`。

设计意图：先把稳定Core编译成每个LoRA row的内容。K归一化只负责寻址，V保留
真实content；routing永不成为value。一个block是当前容量选择，不是永远不能
加深，但只能在证据显示正确Core已存在而compiler传递不足时增加。

### 4.8 模块G：Procedure-to-LoRA Refiner

输入：`Z_C`、Procedure memory`E`和routing`R`。
计算：独立content-only cross/self/FFN生成`D`；Procedure cross-attention
output projection zero-init，所有D的value/residual/FFN内容路径无additive
bias。最终：

```text
Z = Z_C + D
```

设计意图：

- 不是Core和Procedure简单concat后让factor heads随意忽略Procedure；
- fresh时`D=0`，先从可靠Core-only起步；
- 正确顺序只有在跨action、跨video持续有用时才逐渐成为对Core adapter content
  的增加、删除或重定向；
- 不设人工transition gain、temperature或第二套adapter；
- Core表达“是什么/目标关系”，Procedure表达“怎样按阶段完成”，二者融合为
  单一任务程序。

### 4.9 模块H：Factor heads与public LoRA

输入：融合后的`Z[320,256]`。
计算：8个共享bias-free MLP heads，`256->420->target_width`，GELU，final
linear zero-init；shape/name/transpose从真实identity template读取。
输出：38 targets、76 A/B tensors、rank/alpha16、dropout0、
`1,287,168` public LoRA scalars。

设计意图：fresh Writer生成functionally identity LoRA；所有动态更新都必须来自
视频conditioned content。LoRA factor有gauge/non-identifiability，因此训练用
functional policy loss，不以raw factor MSE为主目标。

真实Writer trainable参数：

| component | params |
|---|---:|
| VL Meta-LoRA rank4 | 921,600 |
| Action Meta-LoRA rank8 | 1,253,376 |
| Core projection | 524,288 |
| interaction projection | 262,144 |
| 2 causal Procedure blocks | 1,573,888 |
| Core compiler + Procedure refiner | 2,191,104 |
| factor heads | 3,575,040 |
| **total** | **10,301,440** |

rank-128 Source-SFT为`10,297,344`，差`4,096`（0.0398%）；不是靠扩大Writer
参数获得优势。public LoRA scalars不计入Writer参数比较。

### 4.10 对齐后的不可随意改动原则

- stable task/scene/object信息有价值，不减temporal mean；
- same-task不同正确teacher应产生接近的task program和behavior，因为抽取的是
  统一高层逻辑，而不是demo低层轨迹；
- correct视频应明显优于shuffle/reverse，因为正确顺序含额外的高层完成过程；
- Core按帧集合不变，Procedure对顺序有向敏感；二者缺一不可；
- 不人工命名固定“接近/抓取/放置”token类别，以保持任务类型通用；
- 不把可变长视频压成固定8个event tokens；
- Meta-LoRA保留其无state/observer/human适配职责，但必须持续检查是否重新
  退化成translation/phase code；
- causal attention是必要的优雅顺序owner，但不是单独充分条件；
- 不用contrast/order loss、人工temperature、clip threshold或分位数尺度掩盖
  表示问题；
- 只在逐层证据定位capacity不足时增加compiler/Procedure层数，不能用堆层替代
  正确语义。

## 5. 当前唯一训练合同

owner最终拍板并已实现：

```text
每rank每optimizer step：
    1个task
    1条同task teacher video
    1套one-shot LoRA
    B_a=20条独立同task action queries全部监督这套LoRA

全局4 ranks：
    4个task
    4条video
    4套LoRA
    80条唯一action queries
```

- 每条action只计算一次；不是一条action乘4条video。
- task在4 ranks间均衡轮转；同task后续visit按确定性无放回cycle换video。
- inference仍为`task language + exactly one action-hidden video -> one LoRA`。
- frame stride固定为5。
- `max_frames_per_encoder_call=32`只是保梯度的显存安全分块，不是action
  microbatch或optimizer梯度累计。
- 不使用optimizer gradient accumulation。
- scheduler为100-step warmup、12,000-step cosine horizon；每个segment只改变
  invocation stop，不重写已有LR轨迹。

已经退役且不得恢复：

- 每action独立采`N=4` videos并生成`B_a×4` LoRA；
- 每rank共享4条videos并把action分给4套LoRA；
- B21边界搜索；
- frame stride 10；
- 全105帧一次encoder call；
- v4 visual-state / 7D forecast / Plan / Revision / Belief；
- contrast/order auxiliary loss；
- 任何新runner或平行Writer活动路径。

## 6. 已封存profile与实现证据

正式训练使用的canonical source commit：

```text
0b4e00696113cf6601d6e63b4c73734f3cea1073
Train v5 Writer with one video per task visit
```

该commit已经push到`origin/main`，正式launch前worktree clean且
`HEAD == origin/main`。后续纯文档交接commit不改变本run的训练身份。

focused tests：

```text
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_writer_data.py \
  tests/test_writer_functional.py \
  tests/test_writer_training.py \
  tests/test_writer_checkpoint.py \
  tests/test_writer_model.py

25 passed in 7.99s
```

GPU4–7最长真实视频压力测试：

- 最长样本：global task38/demo36，raw 517帧，stride5后105帧；
- `F32/B20`：最长步`6.956s`，常规步`3.109/3.527s`；
- peak allocated/reserved：
  `76,937,901,056 / 83,630,227,456 bytes`；
- `F32/B24`与`F24/B24`均在首个functional policy forward OOM；
- `F40/B20`无吞吐收益且reserved略高；
- `F105/B1`占`79,873 MiB`且90秒仍未完成；
- owner接受最长视频只保留少量显存余量，并明确停止B21。

保留profile：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v5_jointprofile_f32_b20_long105_20260727
run contract  e83dd24f5edab003280d0d8465848b7a8f45ff97606f2f66790c941c41ccb1fc
metrics       0b39de739d2eca59274ba43c8ea1679e77ead0e1228c55878b2079273799b561
ckpt manifest 993739d4a5c8323d04fc9e0eef60ac4439356a20ee1bc8061e314d2c5100cb1b
```

## 7. v5正式训练、observed-best与失败结论

v5正式root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727
contract 03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c
```

fresh step0→900和exact-resume step900→1800均正常结束；metrics连续finite，
step100至1800每100步checkpoint都包含Writer、trainer、sampler/data cursor、
optimizer/scheduler/scaler与四rank RNG。source policy trainable参数为0，
每step仍是全局80 action queries、4个one-video LoRA conditions和一次policy
forward。旧tmux `ember-v5-as-sv900`与`ember-v5-as-sv1800`均已退出，不得
resume或重复launch。

fixed400 correct-video：

```text
step       100  400  700  800  900  1000  1400  1700  1800
successes   62   64   92   76  103   115   115    71    86
```

step1000与1400并列absolute observed-best；step1400的online functional loss更低，
且是明显峰后下降前较晚一点，因此选择step1400做唯一正式特异性检查。step1700/
1800的`71/86`构成远超正常400-rollout波动的强下降，证明v5时间轴已经充分跨过
高点；不需要为寻找更好点继续同架构训练。

step1400 checkpoint：

```text
checkpoints/step_00001400
manifest d8e77a0ebbb7a7b61bbc9354fffffe02e2384d77f38bb29813146edc6ad71f57
Writer state 0febb29d821dd670c7438f3c54f37ba88a5a3bfe9d2b42eb1652c9d4d95255da
```

内部16-reference五条件root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_step1400_internal_specificity_counterfactual_val8x2_0925140_20260727
summary SHA256 962cde5cc11df528ff526602e443475d74a324c4e7d9d85e0e9cc8693a41c9e3
```

中位relative L2：

| condition | Core set | Procedure sequence | effective LoRA | policy action |
|---|---:|---:|---:|---:|
| same-task-other | 4.423% | 19.052% | 7.304% | 1.394% |
| cross-suite-wrong | 44.732% | 111.729% | 72.638% | 16.987% |
| shuffled | ~0 | 64.299% | 2.928% | 0.486% |
| reversed | 0.0157% | 72.560% | 4.773% | 0.752% |

固定Core只换Procedure时shuffle/reverse effective LoRA为`2.921%/4.767%`，
policy action为`0.415%/0.715%`，说明顺序信息确实在Procedure存在并能到达
policy，但downstream fusion把它压到小于same-demo变化。相比step900，
Procedure顺序差几乎不降，而effective LoRA与action顺序差继续缩小；最早明确
失效层就是Procedure→slot/factor编译，而不是Procedure没有学到顺序。

## 8. v5 step1400正式五臂paired rollout

所有五臂使用完全相同的400个
`(suite, task_id, init_state_id)`、language/env seed、policy-noise共同前缀、
Writer checkpoint与pairing hash。same使用`(correct demo+17)%50`；
wrong保持demo ordinal但映射到cross-suite task；shuffle/reverse使用与correct
相同video和帧集合，只改帧序。四个counterfactual各独占物理GPU4/5/6/7，
每卡6 persistent policy workers和3个Writer generators；36 shards先领取全部
long再领取普通task。四臂均400/400、worker return code全0、无OOM/traceback，
wall为`2201.9–2255.1s`，结束后GPU4–7已释放。

总体与95% Wilson区间：

| arm | success | rate | Wilson 95% |
|---|---:|---:|---:|
| correct | 115/400 | 28.75% | 24.53–33.37% |
| same-task-other | 108/400 | 27.00% | 22.88–31.55% |
| cross-suite-wrong | 74/400 | 18.50% | 15.00–22.60% |
| shuffled | 113/400 | 28.25% | 24.06–32.85% |
| reversed | 114/400 | 28.50% | 24.30–33.11% |

相对correct的exact paired结果：

| arm | both | correct-only | arm-only | both-fail | net correct | churn | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| same | 92 | 23 | 16 | 269 | +7 | 9.75% | 0.337 |
| wrong | 57 | 58 | 17 | 268 | +41 | 18.75% | 2.18e-6 |
| shuffled | 101 | 14 | 12 | 273 | +2 | 6.50% | 0.845 |
| reversed | 103 | 12 | 11 | 274 | +1 | 5.75% | 1.000 |

wrong的正向差主要由Object-1 `+16`与Object-3 `+21`贡献，correct-only flips跨
33条teacher demos；它是真实、分散但仍偏两个tasks的视频语义效应。same的
净`+7`不显著，且Object-3反向`-3`，可视为基本鲁棒但不是统计等价证明。
shuffle/reverse逐task只有`-1..+3`小幅互相抵消；19个correct-fail episode在
至少一个假顺序arm成功，而correct同时优于wrong/shuffle/reverse的episode仅3个。
因此：

```text
absolute pre-gate:   勉强达到（115/400）
same-task robustness:方向可接受，未证明等价
wrong-video semantics:通过
order specificity:  明确失败
overall AS gate:     失败
```

结果文件SHA256依次为：

```text
correct  cc0ea7390d025101b293987b0d3a46971194b78942113e40cf07a237f2f267c2
same     e5b9705e0d2f0e0e6a13fc59416c37cb9c8060d378bab538267eeb91dd2488a0
wrong    2e8b54aba1e7506ff3fa90ed02b58c35cd0ff42084eda55c7d051a719f3700c6
shuffled 514b6647bdc21187004d16581032db0f6470829b0cdc3807737724dc92880977
reversed 2f75bc7b1307fb32dbab57199b953b159c1ede4fbcb332ff0dbe7b6525f5076a
```

## 9. v5.1首段结果、特异性结论与当前停止点

v5.1仍以`docs/action_forecast_writer_v5_1_proposal.md`为唯一架构authority。
机械参数预算为`10,244,872`，比rank-128 Source-SFT少`52,472`。canonical
实现、F32/B20真实训练上限、每卡6个persistent policy workers共同生成LoRA、
全局long-first queue以及step1→2 exact-resume均已通过。

### 9.1 首段训练与observed-best

formal root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v5_1_language_axial_dev_r4_seed7_s12000_c199ad3_20260727
```

fresh step0→900正常结束：4-rank DDP、F32/B20，每step全局80 action queries、
4 one-video conditions，合计72,000 queries与3,600 video conditions；
9个checkpoint及exact-resume state完整，wall `3,622.36s`。online
validation 100..900依次为：

```text
0.1321190  0.1306069  0.1398120  0.1341999  0.1347347
0.1329643  0.1334706  0.1332438  0.1331427
```

validation已经平台化且非单调。旧有放回video采样脚本的80-rollout screen为：

```text
step:    100  200  300  400  500  700  800  900
success:  19   18   15    7   21   17   19   14
```

随后按一张物理卡负责一个checkpoint、四卡同时完成step100/500/700/900各
400条正式correct rollout，得到`82/96/98/84`。step700是observed-best，
但`98/400`明显低于约`110–120/400` absolute预门和旧Action-Forecast
`125/400`最低参照。

这批旧采样在400 rollouts中只产生259个distinct task/demo LoRAs，属于有放回
条件；它没有覆盖每条teacher video。owner最后要求做完当前特异性后停止，
因此此前讨论的无放回重测明确未执行，不得把本结果误写成无放回证据。

### 9.2 step700轻量五臂paired rollout

80条完全相同的evaluation task/state/policy RNG下，只替换teacher-video
condition，结果为：

```text
correct / same-task-other / cross-suite-wrong / shuffled / reversed
17      / 20              / 7                 / 11       / 6
```

相对correct：

| control | correct-only | control-only | exact McNemar p | task正/负/平 |
|---|---:|---:|---:|---:|
| same-task-other | 4 | 7 | 0.54883 | 2/2/4 |
| cross-suite-wrong | 12 | 2 | 0.01294 | 3/1/4 |
| shuffled | 10 | 4 | 0.17957 | 2/1/5 |
| reversed | 13 | 2 | 0.00739 | 3/0/5 |

same没有显著退化，但小样本不构成等价性证明。wrong和reversed在paired-state
层面已有信号，wrong却主要由Object-1的`+7/10`贡献；shuffled方向正确但不显著。
因此闭环视频语义与reverse顺序信号存在，仍不足以证明跨task稳定的完整order
specificity。

初次cross-suite-wrong运行遇到单worker MuJoCo/EGL 0x8cdd。resume虽补齐queue，
aggregation因两次launcher timing窗口不一致而fail-close；该root没有可信
`results.json`。正式wrong=`7/80`只来自随后使用GPU4–7、24 workers、单次调用
完成80/80 rows的fresh clean rerun。

### 9.3 内部16-reference机制检查

8 tasks × 每task 2个teacher videos的relative-L2中位数：

| condition | Semantic Core | Procedure slots | effective LoRA | policy action |
|---|---:|---:|---:|---:|
| same-task-other | 0.0403 | 0.3801 | 0.1163 | 0.0139 |
| cross-suite-wrong | 0.2041 | 1.1826 | 0.5884 | 0.1414 |
| shuffled | ~0 | 1.0516 | 0.5285 | 0.1015 |
| reversed | 0.0030 | 1.6803 | 0.7434 | 0.1668 |

wrong在全部8个task都改变Semantic Core；shuffle/reverse保持set-like Core近零，
但改变causal Procedure。固定Core、只替换Procedure时，order差异几乎完整
保留到LoRA/action；去除Procedure后shuffled严格归零，reversed action只剩
`0.0021`。因此v5.1模块分工和内部信息路径按设计成立，问题在于这些差异尚未
稳定转化为足够广泛的closed-loop成功率优势。

### 9.4 首段判定与已解除的停止边界

```text
absolute performance:    失败（observed-best 98/400）
same-task robustness:    方向可接受，未证明等价
wrong-video semantics:   paired-state成立，task breadth仍弱
order specificity:       reverse部分成立，shuffle未成立
overall v5.1 first gate:  失败
```

首段封存时没有活动训练、评测或内部probe；GPU4–7已释放。该快照原先要求停在
这里。2026-07-27稍后owner已明确解除这一临时停止边界，并建立新的session-local
Goal：在EMBER核心逻辑、信息墙、固定split/source base、one-video Writer和
GPU4–7硬约束内，自主做无放回评测、继续训练、机制诊断、架构改进，必要时可
提前探索RL-Writer；推进期间不需要逐项审核。当前执行合同与证据分叉记录在
`task_plan.md`末尾“开放式AS绝对性能探索”，接手时仍须核验实时进程、artifact
和Goal状态，不能仅凭本文重复launch。

新session第一组只读核验：

```bash
cd /data/ymdai/projects/EMBER
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
tmux list-sessions
nvidia-smi -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

GPU边界始终不变：只使用物理GPU4、5、6、7；0–3不进入visible set或查询；
不得杀、暂停、reset任何他人进程；任何新GPU launch前重新做live GPU与
`/data/ymdai` 500GB cap检查。

### 9.5 step700无放回full400与当前resume合同

新的canonical视频采样把每task demo0..49对state0..49做无放回随机双射。
step700 correct/same/wrong/shuffled/reversed=`88/97/75/65/45`。correct相对
same、wrong、shuffled、reversed的discordant pairs分别为`17/26`、
`46/33`、`46/23`、`60/17`，双侧exact p为`.2221/.1766/.00762/8.91e-7`。
因此same鲁棒、order破坏确实更差；wrong语义方向仍不显著且主要依赖Object-1，
absolute `88/400`也远未到停止标准。

逐row证据：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_1_specificity400_noreplacement_seed7_step0700_
paired_analysis_92b1e03_20260727.json
SHA256 c4a62c4c091b1262c3dbcb17382aad757b8865958212ae62b4a9e4f5986231fa
```

评测queue的global long-first不变；ordinary在不足时细分到至少两个实际worker
波次，标准四卡×6 worker现为48 long + 48 ordinary，避免成功早停后的单波尾部。
实现commit `73f171a`已push，全仓`194 passed`。

正式训练已在tmux `ember-v51-as-sv1800`沿用原root、F32/B20与4-rank DDP启动，
仅将stage stop从900推进到1800。runtime launch commit为`a92850f`；step900
resident validation逐值复现，step1000首个新增checkpoint完整落盘后仍在继续。
step900 checkpoint manifest/canonical payload/writer/trainer SHA256为：

```text
6958498bcefba9093233bde7848f3a5333686d12ce7d16d0aa9d6158e735b828
2971d3a46b7c061efcd0160725c8bb31cb4b8bcb6c5bab3c2d39d4de9cae48fe
17da429d85a5a50db49173d2e147920a38cbb343b49d512c26c2ed61978087ac
a7057a84a7ecd6e16167c27960d78bed53f9728a407f8ca1d22ed179c5b63cda
```

预计新增900 optimizer steps、72,000 action queries、3,600 one-video conditions、
9个checkpoint和约一小时wall。启动前必须确认原训练contract
`acc57fd96cace6d3a9d38a7dbfe6d8593cd29bdce1a0ff10e1f2b4239de46227`
除兼容代码commit外完全一致；该检查已通过。step1000 online validation为
`0.1373837591`，高于step900的`0.1331426708`，再次证明训练结束后不能用它
选best，必须用无放回correct400曲线。

## 10. 文档阅读、代码地图与接手验收

新session不需要历史聊天。先完整读取以下文档并核对真实Git/GPU/tmux/artifact
状态，再继续当前Goal；不得因本文中的历史命令重复启动训练或评测：

1. `README.md`
2. 本文
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_expert_consultation.md`
5. `docs/action_forecast_writer_design.md`
6. `docs/action_forecast_writer_v4_root_cause.md`
7. `docs/action_forecast_writer_v5_design.md`
8. `docs/action_forecast_writer_v5_1_proposal.md`
9. `task_plan.md`
10. `findings.md`
11. `progress.md`
12. `docs/concept.md`
13. `docs/decisions_and_open_questions.md`
14. `docs/novelty_and_landscape.md`

历史v1–v5、共享四视频profile和专家咨询只作provenance。不得从Git历史恢复旧
prompt、旧Action-Memory/Action-Forecast/v5活动架构或平行runner。

常用活动入口：

```text
scripts/train_as_writer.py
    唯一AS训练入口

scripts/evaluate_as_writer_validation_loss.py
    封存512-row functional validation panel；只安排候选顺序

scripts/evaluate_pi05.py
    唯一π0.5 rollout入口；支持correct/same/wrong/shuffled/reversed和LoRA cache

scripts/train_rl_writer.py
    唯一RL-Writer训练入口；只有AS双门通过后使用

configs/pi05_as_writer_language_axial_v5_1.json
    唯一活动v5.1合同；旧v5 config已从活动树删除

src/ember/writer/video_program.py
    task-span双路编码、三套Meta-LoRA与Action Expert probe owner

src/ember/writer/temporal.py
    frame-set attention、language Core、causal Procedure与slot fusion owner

src/ember/writer/as_step.py
    单视频/单LoRA/完整action-batch functional step owner

src/ember/writer/data.py
    task/action/video确定性schedule owner

src/ember/writer/checkpoint.py
    atomic exact-resume owner
```

接手前必须能够准确回答：

1. EMBER为什么需要共享source base，又为什么不能用`pi05_libero`？
2. one-shot Writer与执行时video-conditioned policy有什么区别？
3. 初代119/400为什么不能证明视频理解？
4. Temporal-RoPE为什么证明“位置编码存在”还不够？
5. v4 shuffle 148究竟破坏了什么，为什么不是证明顺序不重要？
6. 为什么不能直接删除temporal mean或所有稳定信息？
7. Core与Procedure分别保存什么，为什么一个按集合不变、一个必须causal？
8. fixed 50-token suffix为什么不是future-action forecast？
9. v5.1三个Meta-LoRA各做什么，又要重点防止什么退化？
10. routing identity为何只能进Q/K？
11. 为什么v5.1用中心化Procedure生成zero-init AdaLN，而不是与Core直接concat？
12. 当前为何是一video/一LoRA/大action batch，而不是N=4 Cartesian或共享四视频？
13. absolute与五臂specificity两个gate分别如何判断，顺序是什么？
14. 为什么首段900只能由v5.1实测吞吐得到而不能继承；又为什么它不意味着
    第二段固定到1800，后续段分别何时才允许开始？
15. GPU、storage、信息墙、停止边界和禁止恢复路径分别是什么？

新session应先用自己的话核对以上核心理解并报告真实代码/训练状态；owner对
v5.1推进的授权已经记录，不应重新要求批准，也不应把已经封存的设计重新当作
待owner回答的问题。

文档状态地图：

- 活动authority：`AGENTS.md`、`docs/execution_brief.md`、
  `docs/action_forecast_writer_v5_1_proposal.md`；
- 活动live ledger：本文、`task_plan.md`末尾、`findings.md`末尾、
  `progress.md`末尾；
- 历史但必须理解：`docs/action_forecast_writer_expert_consultation.md`、
  `docs/action_forecast_writer_design.md`、`docs/action_forecast_writer_v4_root_cause.md`、
  `docs/action_forecast_writer_v5_design.md`；
- 更早provenance：
  `docs/expert_plan.md`、`docs/benchmark_validity_report.md`及ledgers较早日期段落；
- 思想/claim边界：`docs/concept.md`、`docs/novelty_and_landscape.md`、
  `docs/origin_and_general_thesis.md`、`docs/prior_work_memllm_lessons.md`。

历史文档中的“当前”只指其成文日期。它们的数值证据保留，但未来式建议不能覆盖
v5.1和本文；无需从Git历史、旧prompt或历史聊天补齐任何活动合同。

## 11. 2026-07-27 21:00 UTC实时接手状态

此前第9节的首段停止状态已经过期。owner已建立开放式AS性能Goal，并明确要求：
只要absolute尚未达到可信满意水平，或提升存在v4-shuffled式逻辑漏洞，就持续
探索，不需逐项审核。

### 11.1 正式训练与完整无放回曲线

v5.1同一formal root已从step900 exact-resume至1800并正常退出。所有
checkpoints100..1800每100步完整；没有活动训练进程。无放回correct400为：

```text
step   100  500  700  900 1000 1100 1200 1300 1400 1500 1600 1700 1800
succ    83   98   88   86  114  111  114   92  127   95   92   65  126
```

step1400为observed-best。step1800与其只差1，但paired为new28/lost29；
两点不是同一批成功的稳定平台。两个spatial validation tasks全程合计最多
`2/100`，Goal-3最多`3/50`；当前absolute主要来自Object、Goal-6和Long-1，
因此仍未达到满意的多task competence。

step500与step1600曾各遇到一次EGL 0x8cdd；raw queue最终均400/400。commit
`082090f`的累计resume aggregator现已正式生成各自`results.json`：
`98/400`和`92/400`，active wall分别`2494.758s/2578.903s`，两者各2次
launcher attempts。

### 11.2 step1400内部机制判定

内部root：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_1_internal_specificity_step1400_refs2_42a9707_20260727
```

run-contract/summary/rows SHA256：

```text
39cb52068f5365de8ededfc6cf2319f97eb78c2d35b7ebaed6a052d9b3aad9d3
1749a35494d63941918f0da81e682f4b196de6f457d9e0f35394c0196f576d78
56b3314dacd0661f305dcdb6dd3bf9dddea2313479c55b06bca41212ac4d5342
```

same/wrong/shuffled/reversed的Semantic-Core中位relative L2为
`.0509/.2310/~0/.00286`，effective-LoRA为
`.0960/.6733/.5140/.6734`，policy-action为
`.0149/.1329/.1035/.1823`。固定Core、只替换Procedure时order差异完整穿过
LoRA/action；Core-only对order近似不变。结论是v5.1逻辑分工成立，没有
v4-shuffled旁路；不足来自生成修正的有用性、稳定性和task breadth。

### 11.3 当前活动进程与代码

evaluator实现commit为`082090fa5cbbbbde0796fae6642b5c8af690b4fb`；本节
ledger另由后续文档commit封存。实现commit加入：

- 每物理GPU EGL env close/create transition的用户级flock；
- 多次failed/resume launcher的累计shard与active-wall证据；
- 只缩放public LoRA B、并进入paired hash的`writer_lora_b_scale`。

完整回归`196 passed`，architecture guard无hard violation。当前tmux
`ember-v51-scale`有四个窗口：

```text
GPU5 scale125  step1400 correct400, LoRA-B ×1.25
GPU6 scale150  step1400 correct400, LoRA-B ×1.50
GPU4 scale175  step1400 correct400, LoRA-B ×1.75
GPU7 scale200  step1400 correct400, LoRA-B ×2.00
```

全部为validation、无放回、每卡6 policy workers、long-first；共同复用
identity SHA256
`40e53b52caac343275fdb63f98bb224fe48f88b32037ad50071505cfc3873043`
的原step1400 400-entry cache，不重生成LoRA。接手时先检查四个root的
`results.json`、queue和tmux，不得重复launch。

### 11.4 下一判定

1. 等四个scale full400完成，按paired、多task贡献和spatial/Goal-3 breadth
   选择；若只是把原有易task推高或损害逻辑，不算提升。
2. 用选定scale（若无scale胜出则1.0）对step1400做
   correct/same/wrong/shuffled/reversed full400，并核对paired hash与逐row RNG。
3. 当前轨迹在1400/1800 LR仍为`2.904e-4/2.842e-4`，相邻200-step更新近乎
   正交且task能力大幅迁移。若scale不能实质提高，优先从step1400开fresh
   optimizer的一小时低学习率稳定阶段，而不是机械续原轨迹。
4. 若低LR只能稳定已有易task、spatial与Goal-3仍死，按最早失效层进入单一
   canonical语义容量改进；任何架构改动前重新核对参数预算和Core/Procedure
   无旁路保证。AS门未充分通过前，RL只能作低成本可判别机制探索，不能替代AS。

### 11.5 21:40 UTC scale结论与当前实时进程

step1400的LoRA-B full400扫描已经全部完成：

```text
scale       1.00  1.25  1.50  1.75  2.00
successes    127   124   119    99    82
new/lost       — 21/24 26/34 19/47 14/59
```

四个放大scale均失败。1.25/1.50只把Long-1等局部任务推高，同时损害Object；
Spatial-1和Goal-3在所有scale仍为零。后续固定使用`1.00`，不得把scale局部
迁移当作性能改进。四个results SHA256为：

```text
b22e7854ffa7a8bff6fd9976cf8302c51685479a6497aa184715bcbd080e6c48
88d84a3d7e0af7f294b8de6b6a837330e85b8b14a7ad4738990a708cac51964b
d8e025acfc070067c3c01f1a8e2a7ef3ec3614fa37d2f117198489383bd2c378
075f9d3fb66e4339019a689972befb421e9ce7cb405f7768a85aea07cfee0f0b
```

当前活动tmux为`ember-v51-step1400-specificity`：

```text
GPU5 same_task_other
GPU6 cross_suite_wrong
GPU4 shuffled
GPU7 reversed
```

均为step1400、scale1.00、validation full400、无放回、每卡6个Writer
generators完成cache后原进程切换为6个persistent policy workers，queue继续
严格long-first。四个scale所在GPU释放后逐一自动衔接控制臂；不要重复launch。

低LR稳定合同已封存在
`configs/pi05_as_writer_language_axial_v5_1_stabilization.json`，commit
`52503e149e0512b82994904012b1eccbda7589d9`已push。它从原step1400只加载
Writer权重，fresh AdamW、peak LR `1e-4`、warmup50、phase总合同1800，
首段只到phase step900。下一动作是等四臂完整聚合并核对逐row机制，再做一次
只查询GPU4–7的live preflight并启动该一小时正式阶段。

## 12. step1400五臂封存、低LR启动与v5.2候选（2026-07-27 22:35 UTC）

四个控制臂已经全部完成，固定无放回400五臂为：

```text
correct / same-task-other / wrong / shuffled / reversed
127     / 133             / 94    / 107      / 120
```

相对correct的discordant pair与exact McNemar为：

```text
same:     correct-only 25, arm-only 31, net -6,  p=.5044
wrong:    correct-only 52, arm-only 19, net +33, p=1.1227e-4
shuffled: correct-only 45, arm-only 25, net +20, p=.02246
reversed: correct-only 39, arm-only 32, net +7,  p=.4767
```

五臂均满足8 tasks×50 states、每task demo0..49恰用一次、同env seed、
同policy seed及实际消费noise共同前缀。paired artifact为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_1_specificity400_noreplacement_seed7_step1400_
paired_analysis_082090f_20260727.json
SHA256 51c19b66e2c85501b986e57590deec7726ef19c7a71ae48279d6f551a4ec1579
```

结论不是“v5.1已通过”：same鲁棒、wrong语义性和shuffled方向成立，但
reversed与correct仍统计等价，且reversed把Long-1从15推到25、Object-3从23
推到28，只靠Goal-6从42降到23才形成aggregate净差。当前仍有顺序方向漏洞；
同时correct的Goal-3/Spatial-1/3仅`0/0/1`，absolute breadth不足。

低LR正式root已经从原step1400 Writer权重启动：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_1_stabilize1400_lr1e4_dev_r4_seed7_s1800_756bdaa_20260727
tmux: ember-v51-stabilize1400
contract canonical SHA256:
b19937ce040d98704e6ede00cacc0bcfe31db32347677c0d865999d2708dc95a
```

启动前main/origin均为`756bdaa`且tree clean，个人占用
`379,485,047,888 bytes`；只查询的GPU4–7均0MiB、无compute process。
运行使用4-rank F32/B20、fresh AdamW/scheduler/RNG、peak LR `1e-4`、
warmup50，phase总合同1800但首段只到900。step100 checkpoint已完整落盘，
训练继续；不得重复launch。

为避免等本段结束后才开始架构工作，隔离worktree
`/data/ymdai/.codex/worktrees/EMBER-v52-20260727`已经实现并push
`codex/v52-patch-grounding@4011966`。v5.2只增加text-only task query读取
每帧256个PaliGemma image-position contents的Q/K/O cross-attention，无
learned V、无geometry/新loss/新adapter；factor hidden从240降到216，
Writer总参数`10,237,704`，仍低于`10,297,344`上限。设计authority见
`docs/action_forecast_writer_v5_2_design.md`；61项focused Writer测试通过。
低LR首段结束后先并发full400评测phase100/300/600/900。若没有同时显著提高
absolute、task breadth和reversed方向门，立即用GPU4–7对v5.2做F32/B20起始
profile、向显存上限探索并fresh训练约一小时。

## 13. 低LR首段完成、四checkpoint并发评测（2026-07-27 23:30 UTC）

低LR首段已经按封存合同完整停在phase step900：

```text
completed steps:             900 / declared 1800
global action queries:       72,000
one-video conditions:        3,600
training wall:               3,616.478 s
selected_stage_stop:         true
test action/video reads:     0 / 0
```

run-contract file、run-summary、train metrics和online-validation metrics的
SHA256依次为：

```text
6e95a1d933d7ae34ab41a59787c55424e7c94e2a4155841b550062fa570d652b
238853aed306a9f5ac470966d037dab5ade79141c69fffcaca25c1dd97a0ae7b
b183b51f6441260363b45035545b2bcc60f58d2089c5a8ce4b7f3977d46e150d
e513c4f2fc7433cc377369b9d20efa0e753405f0f93570ca27d5e0e7942e6efb
```

100..900每100步的task-balanced online functional loss为
`.135287/.133725/.138699/.133650/.134800/.134450/.133997/.135060/.136773`；
原step1400为`.135241`。它没有形成可选择的下降趋势。更重要的是，连续
100-step Writer update的L2从`1.870`逐渐降至`1.176`，但从第二段起的相邻
update cosine为
`.032,-.114,-.147,-.171,-.128,-.116,-.180,-.152`。到phase900相对初始
step1400累计移动L2=`4.613`。低LR确实缩小步幅，却没有消除跨任务来回覆盖。
完整分模块证据为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_1_stabilize1400_lr1e4_dev_r4_seed7_s1800_756bdaa_20260727/
writer_drift_analysis.json
SHA256 7564fff2fa68b3d370f344d2b9e2180fe98cea640dbc9c82ed2f65af1b16ddc3
```

当前tmux `ember-v51-stabilize-correct400`已经并发启动：

```text
GPU4 phase100
GPU5 phase300
GPU6 phase600
GPU7 phase900
```

每卡独立加载一个checkpoint，使用6个Writer generators和6个persistent
rollout workers，固定无放回400且全局long-first。main/origin仍为
`756bdaa`且tree clean；自动paired分析等待四个`results.json`完整后运行。
不得根据online loss跳过或替换这四个rollout。若四点没有相对原step1400形成
显著、多task且改善breadth的absolute增益，低LR假设即关闭并立即进入v5.2
live profile。
