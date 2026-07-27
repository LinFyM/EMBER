# EMBER v5 当前活动交接

最后更新：2026-07-27 UTC。

本文是当前 focused v5 AS/RL Writer Goal 的完整跨session恢复入口：集中保存
研究北极星、从最早Writer到v5的证据链、shuffle异常带来的核心启示、v5每个
模块的设计初衷、当前训练合同、实时进程快照和下一动作，使新session不依赖
历史聊天也能接手。精确架构与科学合同仍以
`docs/action_forecast_writer_v5_design.md`为准；长期项目边界以`AGENTS.md`和
`docs/execution_brief.md`为准。本文中的step和进程状态是交接快照，接手后必须
先做只读实时复核，不能据此启动第二个训练。

focused AS/RL完成或本文不再承担跨session恢复作用时，更新或删除本文，不能让
过期运行状态长期成为平行authority。

## 1. 当前focused objective参考，不是新session启动指令

Goal是session-local状态，新session看不到也不应机械复制当前session的Goal。
新session第一阶段只负责完整理解仓库、本文与实时训练现场；此时不要创建Goal、
不要修改代码/config、不要启动训练或评测。当前session曾用的objective原文仅
作为理解工作边界的参考：

> 完成 EMBER v5 AS Writer 的高效单视频条件训练闭环：保持已对齐的 Core +
> Causal-Procedure 表示架构，重新设计为少量 task/video LoRA 各自复用多条独立
> 同任务 action supervision、取消 N=4 同一 action 跨视频 Cartesian 重复，优化
> 整帧批处理与显存/吞吐并在 GPU 4–7 上以约一小时为首轮训练尺度；从保留
> checkpoint 的完整 validation 中先寻找绝对性能，若未达到至少约
> 110–120/400 则定位并改进训练或架构后重试，若达到则完成内部数值与
> correct/same-task/wrong/shuffled/reversed rollout 特异性验证，并依据长期
> validation 峰后明显下降标准继续充分探索 AS Writer，之后再按 owner 授权
> 边界推进 cold-start RL。

正确接手顺序：

1. 完整阅读第10节规定的仓库文档；
2. 执行第8节的只读命令，核验训练是否仍运行、真实末步和Git状态；
3. 向owner用自己的话复述研究目标、历史证据链、shuffle启示、v5每个模块、
   当前训练合同、absolute/specificity双门与RL边界；
4. 等owner确认理解和允许接手后，再根据那一刻的真实训练进度与具体下一动作
   建立新的session-local Goal；不预先照抄本节objective，也不设置token budget，
   除非owner另有明确要求。

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

当前未知量正是：这一训练合同下，v5能否同时恢复absolute能力并自然建立
same-task鲁棒、wrong-video语义性与correct-order增益。不要预设答案。

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

## 7. 正在运行：step900→1800 exact resume

不要重复启动。当前tmux与resume log：

```text
ember-v5-as-sv1800
/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727_resume0900_to1800.log
```

旧tmux `ember-v5-as-sv900`已随正常stop-after-step 900退出，不能重复fresh
launch。

正式output与log：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727
/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727.log
```

source base：

```text
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000
```

首段正常完成：

```text
contract_sha256 03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c
completed steps  900
training body    3,485.1537s
global samples   72,000
video conditions 3,600
Writer params    10,301,440
source trainable 0
```

step100至900每100步的9个checkpoint全部有Writer、trainer、四rank state和
atomic manifest。step900：

```text
checkpoint       checkpoints/step_00000900
manifest SHA256  157599fc60e565570be7d711362469b7233606635437f6e219125e7e36f7b8e2
payload SHA256   c838ccba4b1125753e8f11f1ddfaf25c9d6672fe9e1601d3dbb8d7f816b375e5
per task         3,000 examples / 150 visits / 50 videos / 50 action episodes
```

resident 512-row functional loss：

```text
step100/200/300  0.136011 / 0.134911 / 0.133263
step400/500/600  0.132433 / 0.134811 / 0.135303
step700/800/900  0.132294 / 0.132574 / 0.137075
```

fixed400 correct-video代表点step100/400/700/800/900为
`62/64/92/76/103`，所以首段observed-best为step900；它仍低于absolute预门，
但step800→900 paired净增27且没有持续峰后下降。

step900内部顺序通路已增强并到达policy。owner授权的轻量8 tasks×10 states
行为诊断为：

```text
correct / same / wrong / shuffled / reversed = 21 / 25 / 14 / 23 / 23
```

它只说明wrong-video有方向性、顺序优势尚未拉开，不是full400特异性结论。

续训已从step900精确启动到step1800。launch前本文档变更已commit/push，
worktree clean，GPU4–7与存储live preflight通过。精确命令：

```bash
cd /data/ymdai/projects/EMBER
env \
  PYTHONPATH=/data/ymdai/projects/EMBER/src \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  OMP_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  scripts/train_as_writer.py \
  --config configs/pi05_as_writer_core_causal_v5.json \
  --mode formal \
  --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
  --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir /data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727 \
  --resume /data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727/checkpoints/step_00000900 \
  --stop-after-step 1800 \
  --num-workers 2 \
  --log-every 10 \
  --allow-contract-compatible-code-resume
```

该flag只允许recorded Git commit不同；run-contract其他字段必须逐项完全相同。
当前main相对训练commit只改canonical evaluator long-first调度，训练入口、
`src/ember/writer/`和v5 config无diff。

已核验的resume start event：

```text
contract_sha256 03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c
runtime commit  db2a6905cc3d7433333d4c95d08345180c9b4fc2
resume_step     900
stop_after_step 1800
tasks           24
source trainable 0
```

resume时resident模型重算step900 functional loss仍为`0.1370745508`，明确记录
optimizer updates为0、parameter gradients为false、test action reads为0。
随后step901起继续训练；最初核验到step917连续finite，常规step约`3–4s`，
每步仍为全局80 actions、4个one-video conditions和1次policy forward，四卡
物理显存约`77.9GB`、UTL接近100%。这些只作启动证据；接手时必须重读真实末步。

## 8. 新session接手后的第一组只读命令

```bash
cd /data/ymdai/projects/EMBER
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
tmux list-sessions
tail -n 80 \
  /data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727_resume0900_to1800.log
nvidia-smi \
  -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

另外检查：

- `metrics.jsonl`的optimizer step连续、loss/grad/throughput finite；
- 每100步的checkpoint只在atomic manifest完成后才算存在；
- tmux退出时先读terminal log与run summary，不能因为GPU释放就假定成功；
- 若训练仍存活，绝不重复launch；
- 若异常退出，先按checkpoint/metrics/invocation定位是否可exact-resume，不从零
  猜测重跑。

GPU边界：

- 只使用物理GPU4、5、6、7；
- 0–3绝不进入visible set；
- 不杀、不暂停、不reset任何他人进程；
- 新launch前必须重新做live GPU与`/data/ymdai` 500GB cap检查。

## 9. step900后的执行状态与后续固定动作

### 9.1 已完成：封存训练段

已验证：

1. terminal summary为正常stop-after-step 900；
2. metrics严格覆盖step1–900，无nonfinite；
3. step100/200/.../900 checkpoints的Writer、optimizer、scheduler、sampler、
   per-rank RNG和manifest完整；
4. source policy仍全冻结；
5. 24 tasks/action/video coverage符合确定性schedule；
6. 保存训练body wall、queries、task/video visits和hashes。

### 9.2 已完成首段absolute observed-best；继续第二段

- 先核验正式run中常驻模型已经写出的预封存512-row functional panel；正常
  checkpoint不得另起进程重复backfill。该panel只用于安排完整rollout顺序，
  不能用functional loss代替closed-loop选择。
- 不使用80-episode结果选择absolute observed-best。owner明确授权的step900
  轻量五臂80-panel只作续训前机制诊断，不得外推为full400硬门。
- 对step100–900分布式选择有代表性的checkpoint做固定8 tasks×50=400
  correct-video validation；逐步补齐峰值附近、segment端点和可能反弹点，直到
  当前保留checkpoint中的observed-best得到充分定位。
- 第一判断是absolute performance。若所有候选始终显著低于约
  `110–120/400`，先定位训练/架构问题，不提前花费五臂特异性rollout。
- 首段step900为`103/400`，接近但仍低于预门；内部与轻量五臂只证明机制值得
  继续。当前下一动作是同一时间轴exact-resume到step1800。

正式rollout保持官方π0.5/LIBERO合同：render256/model224、agentview与wrist
相机180°旋转、7D state/action、10 flow inference steps、执行5 actions后
replan、dummy settling10、success即停止，Spatial/Object/Goal/Long horizon
分别为`220/280/300/520`。固定400使用同一8 tasks×50 states与配对seed。

评估工程合同：

- 使用唯一`scripts/evaluate_pi05.py`的cost-balanced dynamic queue和persistent
  model/env，不恢复静态一task/一卡；
- Writer LoRA生成batch与rollout replicas是两个独立工程并发量，必须对当前v5
  checkpoint做真实显存/rollouts-per-second profile，不能机械继承v4 batch100；
- 相同可见teacher条件按
  `(language task, video task, demo_id, video condition/order transform)`去重，
  不能按init-state重复生成同一LoRA；
- cache生成完后尽量保留已经加载的source policy并原地转rollout，不无谓
  unload/reload；
- 每个worker始终先领取long shard；只有不存在未领取long shard时，空闲worker
  才领取普通task。这一规则不随一个checkpoint分配的GPU数改变；
- 每张授权卡使用相同CUDA角色数量，物理GPU4不额外堆controller/server/model；
- generation cache可以在不同rollout topology间复用，但scientific condition
  identity必须严格联锁。

### 9.3 特异性顺序

内部低成本检查条件：

```text
correct
same_task_other
cross_suite_wrong
shuffled
reversed
```

逐层检查image hidden、Core、Action Expert interaction、causal Procedure、
Procedure refinement、final query、LoRA、effective B@A和固定policy function。
内部差异通过后才跑五个fixed-400 paired rollout。

行为硬门：

```text
same-task other ≈ correct，且变化最小
correct > wrong
correct > shuffled
correct > reversed
```

方向必须由多个tasks共同贡献，并结合paired churn、McNemar和独立复测；不得以
aggregate 1–2条差异宣称通过。

### 9.4 后续训练与RL边界

- 第一段仍在上升或没有observed-best后的明显、持续、多task、独立复测成立的
  强下降时，按同一合同从best exact-resume下一个900-step segment。
- 多个checkpoint只是略低绝不能停止。
- absolute最低目标为达到或接近旧correct `125/400`；目标逼近或超过v4
  shuffled `148/400`。四卡Source-SFT `108/400`与旧八卡`122/400`只是背景，
  122不是硬门。
- 不通过时只修改证据定位到的最早失效模块；不得增加contrast/order loss来
  强行制造特异性。
- AS同时通过absolute、same-task、wrong-video和order gates后，才开始独立
  short-AS cold-start RL。
- RL先让24 train tasks逐task至少一次official random-reset success，随后永久
  关闭action入口并转pure official reward。
- focused AS/RL完成后停下向owner汇报；不自动进入final-32、task-local RL、
  joint oracle或ViVLA。

## 10. 文档阅读、代码地图与接手验收

新session不需要历史聊天。训练尚在进行时，先完整读取以下文档；在完成阅读和
下面的理解核对前，只做本文件第8节的只读监控，不改代码/config、不启动第二个
训练或提前评测：

1. `README.md`
2. 本文
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_expert_consultation.md`
5. `docs/action_forecast_writer_design.md`
6. `docs/action_forecast_writer_v4_root_cause.md`
7. `docs/action_forecast_writer_v5_design.md`
8. `task_plan.md`
9. `findings.md`
10. `progress.md`
11. `docs/concept.md`
12. `docs/decisions_and_open_questions.md`
13. `docs/novelty_and_landscape.md`

历史v1–v4、共享四视频profile和专家咨询只作provenance。不得从Git历史恢复旧
prompt、旧Action-Memory/Action-Forecast活动架构或平行runner。

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

configs/pi05_as_writer_core_causal_v5.json
    当前AS architecture/data/optimization/profile/formal合同

src/ember/writer/video_program.py
    frame encoder、Core、Action Expert probe与Procedure owner

src/ember/writer/temporal.py
    routing-only identities、Core compiler、Procedure refiner与factor heads

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
9. 两个Meta-LoRA为何保留，又要重点防止什么退化？
10. routing identity为何只能进Q/K？
11. 为什么Procedure是zero-init refinement而不是与Core直接concat？
12. 当前为何是一video/一LoRA/大action batch，而不是N=4 Cartesian或共享四视频？
13. absolute与五臂specificity两个gate分别如何判断，顺序是什么？
14. AS何时可以继续下个900-step segment，何时才允许进入cold-start RL？
15. GPU、storage、信息墙、停止边界和禁止恢复路径分别是什么？

新session应先在commentary中用自己的话简要复述以上核心理解并报告实时训练状态，
然后停在理解验收点；只有owner确认后才建立新Goal并接手后续执行。它不应把
已经封存的设计重新当作待owner回答的问题。

文档状态地图：

- 活动authority：`AGENTS.md`、`docs/execution_brief.md`、
  `docs/action_forecast_writer_v5_design.md`；
- 活动live ledger：本文、`task_plan.md`末尾、`findings.md`末尾、
  `progress.md`末尾；
- 历史但必须理解：`docs/action_forecast_writer_expert_consultation.md`、
  `docs/action_forecast_writer_design.md`、`docs/action_forecast_writer_v4_root_cause.md`；
- 更早provenance：
  `docs/expert_plan.md`、`docs/benchmark_validity_report.md`及ledgers较早日期段落；
- 思想/claim边界：`docs/concept.md`、`docs/novelty_and_landscape.md`、
  `docs/origin_and_general_thesis.md`、`docs/prior_work_memllm_lessons.md`。

历史文档中的“当前”只指其成文日期。它们的数值证据保留，但未来式建议不能覆盖
v5和本文；无需从Git历史、旧prompt或历史聊天补齐任何活动合同。
