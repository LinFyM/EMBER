# EMBER Loom 推导纪要：从原材料、需求到 Teacher–Policy Gap

状态（2026-07-30）：本文是历史推导provenance，按实际讨论顺序记录owner与
side chat如何从
信息墙内最基础的原材料出发，逐层定义 Core、Procedure 和 LoRA compiler 的
需求，经历若干中间方案与纠正，再结合 v4/v5.2/v6/v7/v8/v10 的正式证据，
收敛到：

> **EMBER Loom: Task-Grounded Teacher–Policy Gap Writer**

Loom最终历史规格见
[`action_forecast_writer_loom_design.md`](action_forecast_writer_loom_design.md)；
当前canonical设计见
[`action_forecast_writer_recenter_design.md`](action_forecast_writer_recenter_design.md)。
本文的职责不是重复一份模块手册，而是保存：

```text
问题是怎样提出的
→ 当时为什么得到某个中间结论
→ owner指出了什么遗漏或错误
→ 哪项旧实验改变了判断
→ 最终结构为何是现在的形式
```

Loom已经实现并因首段与内部负证据停止；当前canonical implementation是
Recenter。本文不再授权修改当前代码、配置、训练或评测状态。

---

## 1. 重新开始：第一性原理不是“发明更多模块”

### 1.1 讨论为何重启

此前的 Writer 迭代逐渐形成了一个难题：

- v5.2 的视频语义和顺序特异性很接近理想状态，但 absolute 不够；
- v6 的 absolute 提升到当前最强合法 single checkpoint，但 wrong、shuffled、
  reversed 也明显高于 base；
- v7/v8 为了强化 Action–Effect 关系增加了更强局部约束，absolute 反而下降；
- v10 重新保留 Action/Effect 两流后，Action hypothesis 又主导了最终 LoRA。

如果只盯着这些结果，很容易进入：

```text
看到一个症状
→ 添加一个针对症状命名的新模块
→ 用新结果再修补新模块
```

owner 指出，问题不在“第一性原理”本身，而在此前并没有严格按第一性原理的
顺序推导。

### 1.2 Owner 给出的设计方法

正确顺序应为：

```text
第一层：我们最终需要什么信息和行为？
第二层：信息墙内已经有什么原材料？
第三层：已有原材料如何最小处理后满足需求？
第四层：只有现有运算确实不能满足时，才添加必要结构。
第五层：最后才用旧实验检查每个逻辑接口是否曾被证伪。
```

这里的“最小”是拓扑职责最小，不是刻意压低表达能力。owner 后续又明确补充：

> 第一性原理不要求牺牲表达能力。必要模块一旦成立，就应给足合理宽度、head、
> FFN 和 token 容量；参数预算是软约束，不应为了少几万参数制造瓶颈。

这成为 Loom 的两条共同原则：

```text
没有独立职责的模块不创建；
有独立职责的模块不做成表达瓶颈。
```

### 1.3 旧架构在本轮推导中的位置

owner 要求最初先放下 v10 和现有实现，不让代码形状反过来决定需求。旧版本只在
候选架构推导完整后重新打开，用于回答：

1. 哪些看似合理的接口已经失败；
2. 当前最强架构有哪些优势必须继承；
3. 新架构是否只是换名字重复旧问题。

这一区分贯穿全文：

```text
第一性原理决定职责；
实验证据约束接口；
工程规格决定怎样高效落地。
```

---

## 2. 起点：每帧真正拥有的原材料

### 2.1 信息墙内的基础输入

对于 teacher video 的每一帧，Writer 真正拥有：

```text
当前帧原始图片 I_f
+ exact task description T
+ Writer自建的Action suffix probe sequence
```

它没有：

- teacher action；
- teacher proprio/state；
- reward、success、terminal；
- task ID、suite ID、filename、episode identity；
- object pose、contact label、光流真值；
- 任何 simulator privileged information。

因此后面出现的“动作”“效果”“目标进展”和“能力差距”都必须能追溯到
`I_f + T + suffix probes`。

### 2.2 Meta-LoRA 改变了问题的问法

这些输入不是经过完全原样的 Gemma 和 Action Expert。Writer 在 frozen
π0.5 主干上套有 trainable Meta-LoRA，所以输出 hidden 不能被机械解释为：

```text
image hidden = 物体
text hidden  = 任务
action hidden = 真实动作
```

owner 因此把问题改成：

> 我们借用 Gemma 和 Action Expert，究竟希望它们在训练后提供什么功能信号？

这比给 hidden 起名更严格。最终约定三类基础量：

```text
M_f [L,256]
    当前帧中沿task-token axis表达的任务语义、对象状态和进度证据

P_f [N,256]
    当前帧中沿image-patch axis保留的局部视觉、机器人、对象和关系证据

A_f [K,256]
    Action Expert面对当前teacher状态时形成的source-policy模仿hypotheses
```

其中 `K` 后来确定为 8。

### 2.3 为什么 `M_f` 也是逐帧量

讨论中 owner 特别追问：图片和 Action 显然随帧变化，但任务文本没有变化，
为什么语言轴输出也应写成 `M_f`？

关键在 Gemma 的 multimodal forward。虽然输入 task token 字符串相同，它在
每帧都与不同 image tokens 共同前向。task positions 的 hidden 已经读取了
当前帧，所以：

```text
相同task token
+ 不同frame context
→ 不同multimodal task hidden
```

因此 `M_f` 不是“第 f 帧重复一遍语言”，而是：

> task token 在第 f 帧视觉上下文中的状态化语义证据。

例如“红杯”对应的 task position 可以在不同帧表达：

- 尚未接触；
- 已被夹住；
- 正在移动；
- 接近目标；
- 已形成目标关系。

这解释了为什么 `M_f` 同时适合：

- 跨帧集合聚合成 Core；
- 相邻帧比较形成 task-semantic relation。

### 2.4 `P_f` 与 `M_f` 的区别

二者来自同一次 Gemma multimodal forward，但保留不同轴：

```text
M_f：少量task positions
     已高度语言化，容易说明“与任务有关的状态是什么”

P_f：完整image patch positions
     保留空间分布、机器人运动、对象共运动和局部关系变化
```

`M_f` 更接近高层语义，`P_f` 更适合跨帧视觉对应。两者不是两套视觉 encoder，
也不是重复制造相同信息。

### 2.5 `A_f` 的准确含义

Action Expert 读取：

```text
当前frame + task prefix KV
+ sparse suffix probes
```

输出的是 source Action Expert 对当前状态的动作空间假设。它不是 teacher
action，因为 teacher action 从未进入 Writer；它也不是一个可直接读取的
“base 成功率标量”。

最终更准确的功能解释是：

> `A_f` 表示经过 sparse-probe interface 适配后，source Action Expert 面对
> teacher 当前状态时，会怎样组织自己的模仿假设。

Meta-LoRA 是这个接口的一部分，因此其语义要靠完整训练和反事实验证，不能仅凭
预训练名称保证。

---

## 3. 先不看 `M/P/A`：Core 和 Procedure 真正需要什么

owner 要求下一步暂时抛开现有张量，避免“手里有什么就把它包装成需求”。

## 3.1 Core 的需求

Core 应回答：

```text
这是什么任务？
涉及哪些对象和语义角色？
目标关系和约束是什么？
语言要求完成几个子目标？
任务描述规定了哪些先后关系？
跨执行过程保持稳定的场景条件是什么？
```

这里的“不变”不是固定像素：

- 对象移动后仍是同一个任务角色；
- 视角变化后目标容器身份不变；
- 被遮挡不等于语义身份消失；
- 多视角视频也应能形成同一高层任务描述。

所以 Core 应是高层语义不变量，而不是背景平均图。

### 3.1.1 Core 也可以保留一种“时序”

owner 指出，一个长链任务要求 Core 说清楚：

```text
先完成子目标A，再完成子目标B
```

这不是 teacher 执行过程的 frame order，而是 task language 本身声明的目标
顺序。因此最终边界是：

```text
Core：
    保留task-token ordinal与语言声明的子目标顺序
    对video frame permutation不变

Procedure：
    保留teacher视频真实展示的执行顺序
```

二者不矛盾。

### 3.1.2 Core 容量应怎样变化

Core 不应由一个固定向量强行压缩所有任务。复杂场景未必需要更多 Core，但：

- 更多目标物体；
- 更多目标关系；
- 更长的语言子目标链；
- 更多约束

确实需要更大有效容量。

因此 Core 最终保留 `L` 个 task-axis tokens，而不是固定单 token；任务描述越
复杂，有效 task token 自然越多。

### 3.1.3 Core 的信息边界

Core 可以让 source policy 更理解任务，因此允许提供受限帮助；但它不能独自
生成完整有用 LoRA。

owner 用直觉概括：

> 只知道“要学逆旋转发球”而没有看到怎么打，可以稍微帮助理解任务，但不应像
> 看过正确教学一样获得完整技能。

结构合同因此是：

```text
Core可以解释和定位Procedure；
Core不能独立授权public LoRA。
```

## 3.2 Procedure 的需求

Procedure 应回答：

```text
teacher实际做了什么高层操作？
环境和任务对象因此发生了什么高层变化？
这些操作与变化按什么顺序推进？
source policy面对相同状态时会怎样模仿？
source相对teacher还缺哪一部分能力？
```

它不追求复原：

- 具体左移几厘米；
- 每个 controller tick；
- teacher 的精确 7D action；
- absolute robot clock。

理想粒度类似：

```text
靠近对象
→ 调整接触姿态
→ 建立抓取
→ 对象随机械臂移动
→ 靠近目标
→ 释放并形成目标关系
```

### 3.2.1 Procedure 为什么需要双流原材料

只看 Action hypothesis，不知道视频实际发生了什么；
只看视觉变化，不知道 source policy 会怎样理解或模仿。

因此 Procedure 至少需要：

```text
Teacher-visible stream：
    视频中实际展示的机器人运动、对象变化和目标进展

Policy-imitation stream：
    source Action Expert沿teacher状态序列形成的动作假设
```

最初我们把这叫“动作流 + 环境变化流”；最终更准确地叫：

```text
Teacher Procedure P_T
Policy-Imitation Procedure P_A
```

### 3.2.2 无效教学视频应怎样表现

owner 延续此前的例子：

> 想学逆旋转发球，却看到削球教学，应该是什么也没有学到，而不是把已有能力
> 弄坏。

所以：

```text
视频与任务不匹配
或变化匹配不可信
或过程不连贯
→ adapter趋近identity
→ 性能接近base
```

这后来要求我们把 teacher evidence 的内容和可信度显式分开。

## 3.3 Core 与 Procedure 怎样融合

讨论早期已经达成：

```text
先通过Core了解任务、对象、目标和约束；
再根据Procedure理解具体过程；
最后只把对任务有用的过程知识写进LoRA。
```

但这句话还不够，因为必须继续回答：

```text
Core信息是否可以直接进入LoRA？
Procedure信息由谁决定进入LoRA？
source本来已经会的部分是否还需重复写入？
```

这些问题最终把 compiler 从“Core-conditioned Procedure reader”推进为
“Teacher–Policy Gap compiler”。

---

## 4. 原材料是否够：最终答案是够，但需要职责明确的读取

把需求重新映射到基础 hidden：

| 需求 | 原材料 | 仍需的最小处理 |
|---|---|---|
| task、对象、目标、约束 | `M_f`，稳定task identity | frame-set聚合与task-token composition |
| task-grounded真实patch内容 | `P_f` + stable task identity | 一次显式patch value readout |
| 高层语义变化 | 相邻`M/G` | mean + difference relation |
| 机器人和物体运动 | 相邻`P_f` | same-grid与bidirectional correspondence |
| 环境效果和关系形成 | `M/P`跨帧变化 | task relevance与event tokenization |
| source动作假设 | `A_f` | 8-probe完整保留与temporal composition |
| teacher过程顺序 | ordered relations | causal temporal operator |
| source模仿过程顺序 | ordered `A_f` | 同一类causal temporal operator |
| 能力缺口 | `P_T`与`P_A` | task-conditioned high-level alignment与difference |
| 无效视频回到base | relation consistency/relevance | teacher confidence与final continuous scale |

所以我们并不缺“必须从信息墙外新增”的原材料。需要增加的是：

- 为现有原材料指定可验证职责；
- 保留必要的轴与顺序；
- 避免一个分支越权成为 LoRA shortcut；
- 用最少的结构把原材料变成所需表征。

---

## 5. 关于 task query 的一次重要往返

这是整次讨论中必须原样保留的修正过程。

### 5.1 最初的提议

曾提出让随帧变化的语言状态读取 image states：

\[
G_f=\operatorname{Attention}(M_f,P_f).
\]

### 5.2 Owner 的质疑

owner 指出，Gemma multimodal forward 本来就让 image/text 交互。若不说明新增
读取的独立职责，这只是“再加一层 Gemma”：

```text
已有运算已经完成图文交互
→ 不能仅因想要task grounding
→ 再复制一个同类attention
```

按照“已有东西足够就不创造新东西”的原则，纯推导阶段一度取消额外 task query，
准备直接从 `M_f/P_f` 构建 Core 和 Procedure。

### 5.3 为什么最终又恢复了 `Q_text→P_f`

完整架构推导后，我们重新打开旧实验：

- v5.1 主要读取 multimodal task positions，上游语义不足；
- v5.2 增加 video-independent task query 对真实 patch values 的显式读取后，
  absolute 与空间/对象证据一起改善；
- v6 当前最强合法 single-checkpoint 也建立在该路径上；
- v10 的失败被内部证据定位为 Action 主导和 compiler 尺度，不是 grounding
  路径失效。

这迫使我们区分：

```text
信息已经在Gemma中交互过
≠
Writer下游拥有稳定、可寻址、以task token为索引的patch value通路
```

最终恢复的模块不是另一层完整 Gemma。它只有一个必要职责：

```text
video-independent Q_text作query
+ 每帧真实P_f作value
→ 把patch evidence显式放到稳定task-token axis
```

因此：

\[
G_f=\operatorname{Attention}(Q_{\rm text},P_f,V=P_f),
\qquad
X_f=M_f+G_f.
\]

`Q_text` 只在 Q/K 中选择，不能作为视频 content 写入；value 仍全部来自当前帧。

### 5.4 这是否违反第一性原理

不违反。第一性原理不是“凡是 foundation model 可能隐含的信息都禁止再次
读取”，而是：

> 新模块必须有已有模块未履行、且实验确认重要的独立功能。

Gemma 的职责是形成 multimodal states；显式 readout 的职责是建立稳定
task-axis access bottleneck。二者不是同一个 owner。

这也是 Loom 从当前最强 v5.2/v6 学到的第一项关键优势。

---

## 6. Core 怎样从现有证据得到

### 6.1 为什么不能只取一帧

单帧会混入瞬时遮挡、机械臂位置和阶段状态，也无法稳定识别跨过程不变的对象
角色。

### 6.2 为什么 mean 是不可删除的 backbone

对于每个 task token：

\[
\mu_l=\operatorname{Mean}_f X_{f,l}.
\]

frame mean：

- 对视频帧排列严格不变；
- 保留反复出现的任务对象与环境证据；
- 不依赖 attention 是否学会选择；
- 对固定机位近似语义“长曝光”，对移动视角则在 feature space 聚合语义身份，
  而不是平均像素。

### 6.3 为什么 mean 又不够

重要对象可能只在少数帧清晰出现，简单平均会稀释它。因此加入 task-conditioned
centered residual：

\[
\Delta X_{f,l}=X_{f,l}-\mu_l,
\]

\[
R_l^C
=
\operatorname{Attention}
(Q_{{\rm text},l},X_{f,l},V=\Delta X_{f,l}).
\]

\[
U_l=W_m\mu_l+R_l^C.
\]

centered value 有一个重要安全性质：若 frame attention 均匀，
`R_l^C` 精确相消，mean backbone 仍在；attention 不能凭均匀读取制造额外静态
捷径。

### 6.4 为什么保留 `L` 个 tokens

`U[1:L]` 再经过两层 bidirectional task-token blocks，保留语言 token ordinal，
输出：

\[
C\in\mathbb{R}^{L\times256}.
\]

这样：

- 简单任务有效 token 少；
- 多对象、多关系、多子目标任务有效 token 多；
- Core 容量随任务语义复杂度自然变化；
- Core 对 frame order 不敏感，但保留 task language 的子目标顺序。

### 6.5 Core 的最终边界

Core 不读取：

- frame ordinal；
- teacher action；
- Action stream；
- Procedure temporal state。

Core 也没有直接 factor-head 路径。它只在 compiler 中帮助解释 gap，并受
Procedure-derived scale 授权。

---

## 7. Teacher Procedure 的原材料怎样得到

## 7.1 高层 task-semantic relation

仅做 `X_(f+1)-X_f` 会丢失“什么东西发生变化”。仅做相邻平均又会形成静态
旁路。因此每个 task token 使用：

```text
相邻mean：提供对象/关系上下文
相邻delta：提供实际变化并充当门
```

\[
\bar X=(X_f+X_{f+1})/2,
\qquad
\Delta X=X_{f+1}-X_f,
\]

\[
R^L
=
W_o[
W_d\Delta X
+
\tanh(W_g\Delta X)\odot W_m\bar X
].
\]

当 `ΔX=0` 时 relation 严格为零，所以绝对语义不能绕过变化直接形成 Procedure。

## 7.2 视频中本来就有动作信息

owner 特别指出：不能把“动作”只理解为 Action Expert suffix。teacher 的
机械臂明明在视频中运动，所以图像本身包含：

- 末端执行器移动；
- 夹爪开合；
- 姿态调整；
- robot/object co-motion；
- contact/release；
- 对象搬运；
- 目标关系形成。

`M_f` 的高层差分可能保留部分变化，但 task positions 太少，不能保证保留细粒度
robot/object motion。因此 `P_f` 必须承担 teacher-visible motion/effect 原材料。

## 7.3 为什么需要 same-grid 与 correspondence

只做同 patch index 差分：

\[
P_{f+1,i}-P_{f,i}
\]

能捕获固定网格变化，但对象移动后不再位于同一 patch。

只做 learned matching 又可能：

- 退化成 uniform；
- 总匹配静态背景；
- 把 shuffled 大跳变当强信号。

最终保留两类互补证据：

```text
same-grid delta
+ matched content delta
+ matched displacement
```

匹配双向共享参数，并计算 forward/backward mutual consistency 和 entropy。
坐标只进 Q/K；真正位移以 zero-at-origin Fourier representation 进入 relation
value，保证零位移不产生常数内容。

## 7.4 为什么不用 optical flow、tracker 或 3D

现有 `P_f` 已有视觉语义原材料，cross-frame correspondence 是满足移动对象关系
所需的最小新运算。没有证据表明必须引入：

- 第二视觉 encoder；
- 光流模型；
- object detector；
- tracker；
- 3D reconstruction。

这些模块会带来新的预训练假设、计算和信息边界，而当前需求尚不要求它们。

---

## 8. 为什么 relation content 与 teacher confidence 必须分开

v10 的内部结果显示：

```text
correct visual-transition RMS   ≈ .0201
shuffled visual-transition RMS  ≈ .0510
```

shuffled 跳变能量约为 correct 的 2.5 倍。因此：

```text
变化大
≠
教学价值高
```

### 8.1 Content

relation direction 使用 RMS-normalized content：

\[
\widehat R_i=R_i/(\operatorname{RMS}(R_i)+\epsilon).
\]

### 8.2 Confidence

teacher confidence 单独考虑：

```text
bounded change presence
× task relevance
× bidirectional mutual consistency
× non-uniform matching confidence
× 后续temporal coherence
```

change presence 使用饱和函数，不让 shuffled 的巨大 RMS 无限放大。
task relevance 以`LSE(task-query similarity)-log L`作为neutral baseline，再
经过zero-at-neutral的bounded nonlinearity；无匹配不能像普通sigmoid那样自动
获得`0.5` confidence。matcher confidence则使用
`1-normalized_entropy`，uniform correspondence明确为零。

三类 deterministic backbone 的 type confidence 按有效 relation 的**平均**
confidence 计算，而不是求和；否则 visual 分支仅因有 256 个 patch 就会天然比
language 分支更高。

### 8.3 为什么 confidence 绝不能读取 Action

若 teacher confidence 读取 `A_f/P_A`，trainable Action Meta-LoRA 就能同时决定：

```text
自己提供什么Action内容
+ 自己是否有权生成adapter
```

这会重建 v10 的 Action shortcut。因此：

```text
teacher confidence
= f(task, teacher relations, matching, temporal coherence)
```

严格不读取 Action stream。

---

## 9. Relation 为什么变成 8 个 Teacher Events

### 9.1 一个 Event 不够

一个 interval 可能同时包含：

- 机械臂移动；
- 夹爪状态变化；
- 对象共运动；
- 接触变化；
- 目标关系进展。

把全部 relation 压成一个 token 会形成 v8 式瓶颈。

### 9.2 全部 learned pooling 也不够

v7/v8/v10 的 attention 熵接近理论均匀值，说明 learned query 很容易退化为
平均池化。若 8 个 Event 都靠 learned attention，模型可能把它们学成八个相似
均值。

### 9.3 最终 3+5

每个 interval 输出：

```text
3 deterministic confidence-weighted backbones
    semantic
    visual-forward
    visual-backward

5 learned relation events
    在有界confidence prior下选择互补组合
```

共 8 个 Event tokens，且保留各自 scalar confidence。

这既保证三类已知证据不会因 learned selection 失败而丢失，也给模型足够容量
形成接触、搬运、释放、目标形成等复合事件。

### 9.4 为什么没有 null token

讨论中曾提出 null/zero option，让 attention 表达“没有找到匹配结果”。owner
认为一个真实动作应在视频语义变化中产生结果，不需要额外可学习 null 内容。

最终不创建 null token，而用更严格的零合同：

```text
relation全零
→ Event content全零
→ confidence全零
```

“没有可信教学证据”由 confidence 表达，不由 learned null value 表达。

---

## 10. Action probes：从 1 个到 8 个

### 10.1 为什么一个 probe 太少

最初方案只用一个 Writer action probe。owner 质疑其表达能力。问题不仅是向量
宽度，而是一个 token 会丢掉原生 50-token action horizon 的阶段结构。

### 10.2 最终 8 个 native anchors

选择：

```text
[0, 7, 14, 21, 28, 35, 42, 49]
```

保持：

- 一次 suffix length=8 的 forward；
- 原生 position IDs；
- 对应 Gaussian rows；
- 8 个输出全部进入后续计算。

这与 owner 的确认一致：

> 本质上就是 suffix 从 1 变成 8；不是 8 次 forward，也不是先跑 50 再采样。

### 10.3 为什么 Action Meta-LoRA 不能删除

后来讨论“Action 表示 source 已经会多少”时，曾出现一个过度修正：为了锚定
source 能力，建议使用不受 Meta-LoRA 干预的 frozen Action probe。

owner 立即指出：

> 不行。原始 Action Expert 适配的是 50-token suffix；没有 Meta-LoRA，
> 8-token sparse suffix 本身并不适配原接口。

因此最终决定：

```text
Action Meta-LoRA保留且可训练；
它负责让8-token interface成为有效policy-imitation probe。
```

我们不再把 `A_f` 声称为未经学习的 base competence measurement。其合理
解释是 trainable、shared、受结构约束的 policy-imitation representation。

### 10.4 为什么 8 个 probes 不再提前压成 1 个

v7/v8 已经证明：

- joint `8×L` attention 接近 uniform；
- EventRead 接近 uniform；
- 8→1 容易让 Action 或 Effect 一方被压没。

所以 8 个 Action tokens 必须完整参与局部和时序计算，不再通过 mean、
phase mixer 或 EventRead 提前汇成单 token。

---

## 11. 中间版 Loom：为什么最初曾把 Action 与 Event 混流

在纯第一性原理推导阶段，我们一度得到：

```text
8 Action tokens/frame
+ 8 Event tokens/interval
→ A0,E0,A1,E1,...交错
→ 完整causal Procedure
→ Core-conditioned compiler
```

该方案试图同时满足：

- Action 和视频效果都保留；
- 不做局部强绑定；
- 不把 8 个 tokens 压成 1 个；
- 完整时序中再理解二者关系。

当时它比 v7/v8 更合理，因为没有宣称某个 Action probe 造成某个局部视觉变化。
但它仍有两个尚未被识别的问题：

1. Action 与 Event 一旦在同一 self-attention memory 中混合，Action 可以覆盖
   teacher evidence；
2. compiler 仍在问“Procedure 应怎样生成 LoRA”，没有问“source 相对 teacher
   到底缺什么”。

这两个问题在 v10 正式内部结果出来后变得明确。

---

## 12. v10 结果如何推翻了中间版的关键先验

v10 正式 best：

```text
correct / same / wrong / shuffled / reversed
103     / 94   / 75    / 67       / 43
```

相同 task-complete 训练方法下，v6 为 `143/400`，v10 只有 `103/400`。

内部反事实：

```text
fixed Effect, vary Action LoRA    .6299 / .8659
fixed Action, vary Effect LoRA    .0808 / .1004
Effect attention entropy          ≈ 99.86% uniform
Procedure slot RMS                .0145
gated-Core RMS                    .1781
```

它说明：

```text
当前问题不是Action信息被压没；
Action hypothesis反而太容易主导最终LoRA。
```

同时 tiny Procedure 经 normalization/gating 后被放大，Core support 约为
Procedure slot RMS 的 14.39 倍，破坏了预期职责。

若中间版 Loom 原样把一个 Action token 扩成 8 个完整 Action tokens，再与
Event 混流，很可能成为“8 倍更强的 Action-v10”。

---

## 13. 主进程建议、owner 纠正与真正的概念转折

### 13.1 主进程最初的修正方向

主进程建议：

- Action 只能在 Event 可信时作为辅助；
- teacher confidence 不读 Action；
- Core 和 Action 都由 Event 授权；
- 保留连续 identity scale；
- 避免 raw energy 和 dense `16F` attention。

这些风险判断是对的，但“Action 只是帮助解释 Event”仍没有说明 Action 最独特
的价值。

### 13.2 Owner 对 Action 的重新解释

owner 提出早期 EMBER 的核心思想：

> Action 可以表示 source policy 对 teacher 的模仿程度。若 Action hypothesis
> 与视频中的 teacher procedure 越接近，说明模型本身越会这个任务；LoRA 需要
> 补充的就越少。

这使问题从：

```text
Action怎样帮助解释Event？
```

变成：

```text
teacher展示的过程
- source policy能够模仿的过程
= Writer真正需要写入LoRA的能力缺口
```

这是最终 Loom 的核心转折。

### 13.3 “不要更稳妥，要最优解”

讨论中曾以“更稳妥”为理由建议少改或保留某些路径。owner 强调：

> 目标不是选择风险最小的折中，而是从第一性原理寻找逻辑上的最优结构。

因此最终结构不是把 Action 权重调小，也不是在旧 compiler 上加一个 gate，而是
重新定义 Writer 的目标量：

\[
\Delta\theta
\sim
\text{Teacher Procedure}
-
\text{Policy-Imitation Procedure}.
\]

### 13.4 对 Action Meta 的过度修正及最终改正

随后曾错误推断：若 Action 表示 source competence，就应冻结或绕开 Action
Meta-LoRA。owner 指出 sparse suffix interface 必须适配，这个建议不可行。

最终不是靠冻结 Action 防捷径，而是靠结构分权：

```text
Action Meta可训练，负责形成有效8-probe imitation representation；
Action不能提供teacher confidence；
Action不能独立进入factor；
Action只能在与Teacher Procedure对齐后，通过gap影响LoRA。
```

这样既保留表达能力，也不给 Action 自己决定“内容 + 权限”的双重权力。

---

## 14. 最终 Procedure：两条独立 value stream，共享时序语法

## 14.1 为什么 Teacher/Policy 在 compiler 前不能混值

如果 `E/A` 在同一 self-attention memory 里交错：

- Action value 可以覆盖 Teacher value；
- 内部差异难以归因；
- v10 的 Action dominance 可能重现；
- 后续无法真正计算 teacher-policy difference。

因此最终：

```text
Teacher Events → P_T
Action probes  → P_A
```

在 compiler comparison 前不相加、不 concat 后混合 self-attention、不互读
value。

## 14.2 为什么又共享 temporal operator

两条流需要可比较的高层过程语法。如果各用完全独立大网络，它们可能学到任意
不兼容坐标系，同时参数和计算翻倍。

所以采用：

```text
stream-specific input norm/projection/type QK
+ shared axial temporal block weights
+ strictly separate value memories
```

共享的不是内容，而是“怎样把低频证据组织成高层过程”的时序算子。

## 14.3 为什么是 axial

完整保留：

```text
Teacher：[F-1,8,256]
Policy ：[F,8,256]
```

每层：

1. 每个 frame/interval 内 8-slot local self-attention；
2. 每个 slot identity 沿时间 causal attention；
3. FFN。

相对 dense `16F-8` global attention：

\[
O((16F)^2)
\rightarrow
O(F\cdot16^2+16\cdot F^2).
\]

它不需要 carrier token，不丢 8+8 证据，也避免最长视频约 1672 tokens 的
dense 全局计算。

## 14.4 输出

\[
P_T\in\mathbb{R}^{(F-1)\times8\times256},
\]

\[
P_A\in\mathbb{R}^{F\times8\times256}.
\]

Teacher scalar confidence 随 `P_T` 保持为独立通道，只能由初始 confidence
乘 bounded temporal coherence；初始为零时不能被时序网络凭空变成可信。

---

## 15. 最终 compiler：先知道任务，再比较teacher与source

## 15.1 每个 LoRA slot 先读取 Core

保留 v5.2/v6 已证明有效的 320 routing slots：

```text
18 expert layers × rank16 = 288
action_in rows           = 16
action_out rows          = 16
total                    = 320
```

每个 slot 先读 Core：

\[
C_s=\operatorname{CoreRead}(R_s,C,C).
\]

它确定当前 LoRA 功能位置面对的任务、对象、目标和约束。

## 15.2 分别读取 Teacher 与 Policy

Teacher read：

\[
T_s
=
\operatorname{TeacherRead}
(R_s+\operatorname{Norm}(C_s),P_T,P_T).
\]

同一 attention weights 读取 teacher confidence `q_s`。

Policy read：

\[
A_s
=
\operatorname{ActionRead}
(R_s+\operatorname{Norm}(C_s)+\operatorname{Norm}(T_s),P_A,P_A).
\]

这里 Teacher target 帮助选择 source hypothesis 中真正应比较的部分，但
`P_T/P_A` value 仍分别拥有。

## 15.3 learned high-level alignment

两个 bias-free full-rank projections 把读出的高层过程映到共同空间：

\[
t_s=\operatorname{Norm}(W_TT_s),
\qquad
a_s=\operatorname{Norm}(W_AA_s).
\]

比较发生在完整高层 Procedure 后，而不是局部帧、7D action 或未经校准的
absolute phase。

\[
d_s=t_s-a_s.
\]

### 为什么用 learned alignment

`P_T` 来自视频变化，`P_A` 来自 Action Expert hidden；即使共享 temporal
operator，二者的输入坐标不同。直接逐维相减没有语义保证，所以 `W_T/W_A` 是
形成共同 gap space 的必要最小映射，不是冗余 decoder。

## 15.4 imitation degree 怎样决定适配强度

\[
g_s
=
\frac{\operatorname{RMS}(d_s)}
{\operatorname{RMS}(d_s)+\tau_{\rm gap}}.
\]

\[
r_s=q_s\cdot g_s.
\]

含义：

```text
teacher可信 + source与teacher接近
→ 已经较会模仿
→ gap小，少写LoRA

teacher可信 + source与teacher差异大
→ source缺少该能力
→ gap大，重点适配

teacher不可信
→ 无论Action如何
→ q小，接近identity
```

`t/a` 在比较前归一化，所以 Action Meta 不能仅放大 hidden norm 来操纵 gap；
它若改变 gap，必须改变规范化后的 imitation content，且可由反事实观测。

## 15.5 Core 如何进入 LoRA

Core 不独立生成 adapter，而是帮助解释 gap：

\[
Z_s
=
\operatorname{Norm}(d_s)
+
\tanh(W_g\operatorname{Norm}(d_s))
\odot
\operatorname{Norm}(W_cC_s).
\]

这里：

- `d_s` 是主要内容；
- Core 指明差距对应哪个对象、目标和 LoRA 功能；
- 最终仍必须乘 `r_s`，所以 `q=0` 或 `d=0` 时 Core 不能形成更新。

这回答了此前两个问题：

```text
Procedure信息如何进入LoRA？
    由teacher-policy gap本身提供content，并由q×gap授权。

Core信息如何进入LoRA？
    只作为gap-conditioned support，不能独立进入。
```

## 15.6 为什么 scale 必须最后乘回

v10 证明 tiny Procedure 经过 RMSNorm、FFN 或 gating 后可以恢复到单位尺度。
因此不能只在 compiler 中间乘一次 gate。

最终 factor：

\[
\Delta {\rm factor}_s
=
r_s\cdot\operatorname{FactorHead}(
\operatorname{SlotCoordinate}(Z)_s
).
\]

`r_s` 在所有 normalization、FFN 和 slot coordination 之后才显式乘回。

这保证连续合同：

```text
teacher evidence → 0
或 teacher-policy gap → 0
→ public LoRA delta连续→0
```

而不只是输入精确全零时 identity。

---

## 16. 当前最强旧架构的优点是否真正被继承

owner 最后特别要求：

> 不要改回旧架构，但要确认第一性原理推导出的新架构有没有学到当前最强架构的
> 优势。

这个问题不能只回答“整体更先进”，必须逐项检查。

## 16.1 v5.2：视频特异性最理想

正式五臂：

```text
correct / same / wrong / shuffled / reversed
132     / 138  / 74    / 82       / 83
```

v5.2 已验证优势：

1. video-independent stable task query；
2. task query 对真实 patch values 的显式访问；
3. Semantic Core 对 frame permutation 不变；
4. Procedure 保留 frame order；
5. 320-slot compiler 能把上游差异传到完整 LoRA；
6. wrong/shuffled/reversed 接近 base，而不是破坏已有能力。

Loom 的继承：

| v5.2优势 | Loom对应 |
|---|---|
| stable task query | `Q_text` |
| task-grounded patch value | `G_f=Read(Q_text,P_f,V=P_f)` |
| set Core | mean backbone + centered residual |
| ordered Procedure | Teacher/Policy axial causal streams |
| strong transmission | 320 slots + full-width factor heads |
| invalid video回到base | teacher confidence × bounded gap |

所以 Loom 没有因纯粹追求简洁而删除 v5.2 最关键的正证据路径。

## 16.2 v6：当前最强合法 single checkpoint

v6 task-complete fast-decay：

```text
correct / same / wrong / shuffled / reversed
143     / 135  / 125   / 128      / 129
```

v6 的结构与工程优势：

1. stable task-grounded patch trajectory；
2. mean backbone + task-conditioned centered residual Core；
3. pretrained Action hypothesis；
4. visual transition；
5. 两层高效 causal temporal reasoning；
6. 320-slot strong compiler；
7. width256 full-width factor decoder；
8. exact functional identity；
9. hardware-friendly `256, 8×32`。

Loom 全部保留或扩展：

```text
Q_text/M/G/X                保留
Core mean+centered residual 保留
Action Meta hypothesis      扩成完整8 probes，但限制为gap输入
visual transition           扩成semantic + raw visual relation
2-layer causal temporal     改为共享axial，不增加dense瓶颈
320 slots                   保留
factor hidden 256           保留
identity                    zero-init基础上增加连续q×g
256 / 8×32                  保留
```

Loom 没有保留的，是后来证据表明有问题的职责混淆：

- Action 作为独立 LoRA content；
- raw transition energy 等同教学价值；
- compiler 不区分 source 已会和未会；
- tiny Procedure 经 normalization 被重新放大。

## 16.3 v6 的训练优势与模型优势不能混写

task-complete fast-decay 是 v6 达到 `143/400` 的重要训练条件。它不是某个模型
层，因此 Loom 不能声称“结构上继承”。

合理做法是首轮保持该 recipe：

- 给新架构已知可用的优化环境；
- 隔离模型贡献；
- 继续观测 task 漂移；
- 后续再独立修改多 task training。

## 16.4 v4 shuffled 148 不是应继承的优点

v4 shuffled `148/400` 曾经最高，但根因是：

- low-level 50×7 forecast；
- absolute phase/translation controller correction；
- raw image/Meta bypass；
- Plan/Revision 时钟路径。

它不是有效视频理解，因此 Loom 只继承：

> source Action Expert 站在 teacher states 上想象“我会怎么做”。

不继承使 shuffled 获得虚假高分的低层旁路。

## 16.5 最终审计结论

新架构不是“抛弃最强旧架构再凭直觉重建”，而是：

```text
v5.2的稳定语义访问与理想控制臂行为
+ v6的最高合法absolute骨架、容量与效率
+ teacher-visible raw motion
+ source-policy imitation comparison
+ 连续confidence/gap scale
- v4/v7/v8/v10已经暴露的shortcut与瓶颈
```

在当前证据下，没有发现哪项已被正实验支持、且与第一性原理一致的关键能力被
Loom 无理由删除。

---

## 17. v7、v8、v10 的负证据分别怎样改变 Loom

| 版本 | 关键证据 | Loom中的对应约束 |
|---|---|---|
| v7 | joint `8×L` entropy约99.96%；Core几乎不传到LoRA | 不做local Action–Effect competition；Core以gap-conditioned support进入 |
| v8 | EventRead约99.67% uniform；Effect远强于Action | 不做8→1 EventRead；保留8+8；不做strict multiplication |
| v10 | Action改变主导LoRA，Effect只占很小部分 | Teacher/Policy value分离；Action无direct factor path |
| v10 | tiny Procedure被Core gate放大约14.39× | `r=q×g`在所有norm/FFN后最终乘回 |
| v10 | shuffled transition energy大于correct | content/confidence分离；energy bounded；加入一致性和task relevance |

这些修改不是“版本越新模块越多”，而是在同一个最小职责图上补齐必要边界。

---

## 18. 最终完整职责图

```text
task T
├─ text-only Gemma + Text Meta
│    → Q_text：稳定task identity / query axis
│
teacher frames I_0...I_(F-1)
├─ multimodal Gemma + VL Meta
│    ├─ M_f：task-axis frame evidence
│    └─ P_f：patch-axis visual evidence
│
├─ Q_text reads P_f
│    → G_f
│    → X_f=M_f+G_f
│
├─ set aggregation over X
│    → Core C
│
├─ adjacent X relations
│  + bidirectional P correspondence
│    → normalized relations + action-free confidence
│    → 3 backbone + 5 learned Events
│    → shared axial temporal operator
│    → Teacher Procedure P_T
│
└─ same prefix KV + 8 native suffix probes + Action Meta
     → A_f[8]
     → same shared axial temporal operator
     → Policy-Imitation Procedure P_A

320 routing slots
├─ read C
├─ read P_T and q
├─ read matching P_A
├─ align t/a
├─ d=t-a
├─ r=q×bounded_gap(d)
├─ gap-conditioned Core support
├─ content-only slot coordination
└─ r×full-width zero-init factor heads
     → complete sealed rank-16 public LoRA
```

每种信息只有一个最终 owner：

| 信息/权力 | 唯一owner |
|---|---|
| task identity与语义约束 | Core |
| teacher展示的过程 | `P_T` |
| source的模仿假设 | `P_A` |
| teacher是否可信 | teacher relations/confidence |
| source还缺什么 | aligned gap `d` |
| 是否允许适配 | `r=q×g` |
| LoRA功能位置 | 320 routing identities |
| LoRA具体内容 | gap + gated Core support |

---

## 19. 明确排除的方案

| 排除项 | 原因 |
|---|---|
| Core-only LoRA | 任务理解不能冒充教学过程 |
| Action-only LoRA | 重建v10 shortcut |
| teacher confidence读取Action | Action同时控制内容和权限 |
| freeze/remove Action Meta | 8-token sparse suffix不适配原生50-token接口 |
| local `8×L` Action–Effect softmax | 信息墙内无局部teacher action标签，v7近uniform |
| 8→1 Action/Event read | v7/v8已显示压缩与uniform瓶颈 |
| strict Action×Effect multiplication | 错误声称局部因果，且容易把一流压没 |
| mixed Teacher/Policy value memory | Action可覆盖teacher证据，无法形成可归因gap |
| raw energy prior | shuffled大跳变被奖励 |
| learned null value | confidence已能表达无证据，不需凭空内容 |
| dense global `16F-8` attention | 最长视频计算约相对v10增64倍 |
| single carrier压缩全部局部证据 | 再造新瓶颈 |
| optical flow/tracker/3D/第二encoder | 当前原材料尚未证明不足 |
| terminal RMSNorm后直接factor | 会抹掉Procedure scale |
| weird hidden width凑参数 | 不值得牺牲kernel效率 |

---

## 20. 哪些是结构已闭合，哪些仍是科学假设

### 20.1 已闭合的结构逻辑

当前复审没有发现尚未分配 owner 的必要信息，也没有发现显式的独立
Core-only/Action-only LoRA path。

已闭合合同包括：

- Core 对 frame permutation 不变、对 task order敏感；
- shuffled/reversed arm 内重新计算 relation；
- Teacher/Policy values 在 comparison 前分离；
- Action Meta 保留但不控制 teacher confidence；
- relation content 与 confidence 分离；
- 8 Action + 8 Event 不提前压缩；
- routing/position/type 只进 Q/K，不制造 value；
- `q→0` 或 `d→0` 时 LoRA 连续趋近 identity；
- factor 容量不因第一性原理设计被无故压缩。

### 20.2 仍需实验回答的科学假设

逻辑自洽不等于性能已经被证明。以下必须由主进程实验验证：

1. `P_T/P_A` 经 shared temporal operator 和 `W_T/W_A` 后是否形成可比较的
   高层过程空间；
2. 高 imitation similarity 是否真的意味着较小必要 LoRA，而不是 representation
   collapse；
3. teacher confidence 是否对 correct 高、对 wrong/shuffled/reversed 低；
4. patch correspondence 是否捕获 robot/object co-motion，而非背景或uniform；
5. 3+5 Event tokenizer 是否形成互补事件；
6. gap compiler 是否减少 same-task demo variance 和 multi-task checkpoint drift；
7. axial Procedure 是否能在最长视频保持 B20，或需降 B16；
8. task-complete recipe 是否仍过度平均多 task gradient。

这些是不确定的研究命题，不是需要继续提前添加模块的结构缺口。

---

## 21. 最终决策链，一句话版本

```text
每帧只有图片、任务、suffix
→ Meta-LoRA后的M/P/A才是可训练原材料
→ Core需要任务不变量，Procedure需要teacher过程与source模仿过程
→ M/P足以构造teacher语义变化和视觉运动，A足以构造policy hypothesis
→ v5.2/v6证明稳定task-query patch access必须保留
→ 8个Action与8个Events都不能提前压缩
→ v7/v8否决局部强绑定，v10否决Action/Event混值和中间尺度门
→ owner把Action重新解释为source对teacher的模仿程度
→ Writer目标自然变成Teacher Procedure − Policy-Imitation Procedure
→ teacher confidence只由teacher证据决定
→ Core只解释gap，q×gap在最终factor处授权
→ 得到Loom
```

这就是最终架构为何不是 v5.2、v6、v10 的回退，也不是随意增加复杂度，而是把
各版本真正有效的能力放回一个职责单一、可以逐项证伪的数据流中。

---

## 22. 给主进程的阅读与采用边界

推荐主进程按以下顺序阅读：

1. 本文：理解推导、纠正和证据链；
2. [`action_forecast_writer_loom_design.md`](action_forecast_writer_loom_design.md)：
   获取最终张量、模块、公式、参数和验证合同；
3. v10 的正式 rollout 与内部反事实；
4. 决定是否将 Loom 原位切换为新的 canonical Writer。

若采用：

- fresh incompatible schema；
- 不从 v10 Writer checkpoint resume；
- 不维护 v10/Loom 双 canonical path；
- 首轮保持已知有效 task-complete recipe，以隔离模型架构；
- 先做最短 shape/gradient/identity/causal/resume/profile vertical path；
- 再用 absolute、五臂、内部 gap 传递和 task drift 共同判定。

在 owner 与主进程明确采用之前，两份 Loom 文档只保存设计，不改变当前实验状态。
