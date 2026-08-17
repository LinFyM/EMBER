# EMBER Architecture Reasoning After Consolidation

状态：认知重建与设计约束，不是active design，不授权实现、训练或GPU实验。

本文不从某个模块名出发，而从EMBER要传递的信息和历史最早失败接口逐步推理。任何具体架构只有在这里的分支问题
被证据裁决后，才应单独建立design authority。

## 1. 先固定问题，不先固定方法

输入：

```text
exact task language + K action-hidden correct videos
```

输出：

```text
one complete task-conditioned LoRA for one frozen source policy
```

部署：Writer运行一次，policy在未见初始化闭环执行。当前评价的是初始LoRA，不包含生成后的task-local RL。

这里没有预先规定K、memory token、rank、Action Expert调用方式或decoder。它们必须由信息流和失败证据推导。

## 2. 输入里真正存在的四类信息

### 2.1 Language query

语言确定任务对象、关系、关注点和目标。它适合提供语义query或address，不应独立成为LoRA Value。

### 2.2 Static visual context

单帧可识别对象、颜色、位置和场景，但无法单独确定“怎样从初态演化到目标态”。静态信息需要服务于语言query，
不能成为language+first-frame shortcut。

### 2.3 Directed process

视频内部的状态变化、阶段边界、接触事件和先后依赖，是correct/shuffle/reverse区别的来源。每条video必须在这一维
保序建模。

### 2.4 Cross-video invariants

多个同task videos之间共同保留的对象关系、目标和阶段结构，更可能是可迁移Program；不同的路径、速度和视角更
可能是nuisance。聚合应发生在有语义的Program证据上，而非raw frames或最终LoRA。

## 3. 输出端真正需要表达什么

LoRA不是一串任意参数。38 targets位于不同policy layers和q/v/action families，具有不同输入/输出维度和功能。
task-local experts和v6已经证明：native rank16、低stable rank、q-dominant与跨列coherent都可能policy-effective。

因此输出问题更准确地写为：

```text
task Program
    -> layer/family/target/rank-aware policy coordinates
    -> native A/B factors
    -> effective BA and action change
```

直接回归百万参数不是唯一方式；但降低rank、增加atoms或追求正交也不是自动解法。关键是Program坐标与policy功能
坐标是否对应，以及写出是否保留已有support。

## 4. 历史证据对四个接口的裁决

| 接口 | 已有正证据 | 尚未解决 |
| --- | --- | --- |
| Evidence extraction | v5/v6、LPCP能读语义、动态和顺序 | correct过程是否被理解成可迁移因果Program |
| Cross-video Program | K4、CMBG、GOMQ可形成高coherence同task表示 | 高coherence可能只是稳定错误mean；cross-task separability不稳 |
| LoRA compilation | task experts、v6/LPCP rank16、direct native heads均可material | material更新是否沿held policy-useful方向并保留support |
| Shared training | reward、endpoint和expert occupancy均可产生task-local下降 | 多task在相邻single checkpoints持续共同积累 |

这张表排除几种过度简化解释：

- 不能只说“视频没有被读取”；
- 不能只说“LoRA参数太多”；
- 不能只说“memory token失败”；
- 不能只说“训练轮数不够”；
- 不能只说“Writer容量不够”。

## 5. 为什么类SHINE路线既不能照搬，也不能简单否定

过去Dynamic-K rank8 Backbone-Memory路线在真实图文/Action-probe context中加入memory并直接生成LoRA，工程上成立，
但strict只有100左右。失败组合同时包含：弱化absolute Semantic Core、rank8 fixed-A/Direct-B tail、特定mapper和
functional credit。因此它不能否定memory correspondence一般。

成熟Hypernetwork工作真正值得继承的是：

- 输入内容由一个本来就能理解该模态的backbone处理；
- 少量memory states与目标层/参数有明确对应；
- memory通过正常内容计算获得信息，而不是无context运行；
- decoder随backbone层数/target数可扩展。

EMBER不同于纯文本Doc2LoRA：π0.5的VLM/Gemma与Action Expert原生处理图像、语言、state和action denoising。任何
memory设计都必须说明它在真实native context中读什么、写什么、是否改变原policy计算；不能为了模仿论文在缺失
图像/文字prefix时空跑Action Expert。

## 6. Memory token应被问成三个问题

不是先问“要不要memory”，而是：

1. **放置**：memory位于输入内容encoder、policy layer observer还是LoRA target grid？
2. **通信**：它只读native context，还是双向改变原policy token？如果改变，step0 identity和source行为如何保证？
3. **职责**：它承载per-video有序证据、cross-video共同Program，还是LoRA target address？一个token不应同时承担
   四个未分离职责。

历史证据偏向一个约束：memory必须处于真实图文/Action context中，且最好有layer correspondence；但它是否作为
encoder token、one-way observer或compiler query仍未裁决。

GOMQ证明learned input query有用；其151→135→131说明不能沿用当前independent rank32 Direct-B tail并期待稳定。

## 7. 多视频数据流应先分轴，再决定注意力

自然数据结构有四个不同轴：

```text
video set K
  × ordered time/stage within each video
  × policy layer/family/target
  × LoRA rank/factor coordinates
```

这些轴不意味着每一对都要做全注意力。每个交互必须有职责：

- 时间轴：提取一条video的有向过程；
- K轴：寻找same-task不变量，置换不变；
- policy轴：把Program放入冻结policy可利用的层/功能坐标；
- parameter轴：生成native LoRA factors。

横向/纵向attention只有在对应上述职责时才合理。无目的地让四个轴全互联会同时增加复杂度、数据需求和归因难度。

## 8. 当前最关键的分叉：Program问题还是tail/credit问题

现有证据存在两种仍可能同时成立的解释：

### H1：共同Program尚未形成

same-task高cosine可能来自task identity或shared nuisance；不同task表示effective rank低，导致shared map将它们压成
共同更新。PAFS address effective rank约2.16、早期rank8 task-mean高度同向支持这一风险。

### H2：Program已经足够好，但tail/credit破坏它

CMBG/GOMQ在held videos上形成很强共同坐标，GOMQ learned query还真实提高到151；随后Direct-B shared update和
task conflict使support回落。这支持“carrier/Program已有价值，compiler或credit是首因”。

不能凭偏好选择H1或H2。下一架构前应从保留artifact做stage-wise矩阵：

1. per-video ordered representation；
2. K-set Program；
3. layer/family pre-compiler state；
4. native factors/effective BA；
5. action response；
6. rollout gained/lost。

对每一stage同时量化within-task cross-video alignment、between-task separability、held/train transport和相邻update的
retention。最早从健康变坏的stage决定下一单变量。

## 9. 对未实现MCPS设想的准确定位

整理前提出过一个未实现桥接假设：保留GOMQ真实memory和K4过程，把independent rank32 Direct-B tail替换为V6的
ordered Procedure slots与native rank16 compiler。它合理地继承“memory有价值”和“V6 compiler达到143”两条证据，
也比完全类SHINE rank8重启更连续。

但它尚不能成为active design，因为：

- 如果GOMQ失败首因是reward方向而不是tail，换compiler不会解决；
- 如果V6 frozen fusion仍把Program压回AS139邻域，可能重复LPCP小改写；
- 尚未用stage-wise evidence证明GOMQ memory与V6 Procedure coordinates兼容；
- 它必须解释correct顺序为何结构必要、language-only bypass如何阻断以及multi-task support如何保留。

因此MCPS只保留为一个待比较的bridge hypothesis，不在active tree保留一份伪装成authority的设计文档。

## 10. 对训练方式的逐步约束

最终方法应能从fresh训练，而开发阶段可用强checkpoint隔离变量。训练数据不得超出固定train24以获取不公平优势；
借鉴SHINE/Doc2LoRA只借结构思想，不引入额外task数据。

一个合理训练系统需要分别解决：

1. **Representation identifiability**：same-task跨episode/video一致，但保留correct有向过程；
2. **Policy grounding**：Program对冻结policy的native action功能有作用；
3. **Shared coexistence**：不同task update在同一checkpoint共存；
4. **Closed-loop selection**：及时用strict paired400否决surrogate错位。

监督学习可以负责表示与compiler warm-start；Writer-level RL或expert successful occupancy可以负责policy-aligned credit。
这不等于预先规定三阶段pipeline，也不等于把生成后的task-local RL混进当前目标。

## 11. 下一步设计前应完成的证据问题

按顺序回答，前一个没有答案时不跳到token数/rank/LR：

1. v6/LPCP、GOMQ和Dynamic-K各自在什么stage首次丢失between-task separability？
2. GOMQ cycle2的25 gains与17 losses，在pre-compiler Program上是否可分？
3. cycle2→3的29 lost由Program漂移、compiler放大还是policy Jacobian方向改变主导？
4. V6 native rank16 compiler接收GOMQ memory-derived Program时，step0和material response能否同时成立？
5. correct/reverse/shuffle差异在Program、BA和action中哪一层首次变成“有用方向”而非任意差异？
6. train24不同task Program的effective rank和gradient conflict是否支持shared Writer容量不足，还是坐标/credit错误？

这些问题优先使用现有formal artifacts和只读分析回答。只有证据指向一个明确最早接口，才建立单变量design。

## 12. 一个未来设计必须回答的八个问题

1. 为什么能提取跨初始化成立的高层任务知识？
2. 为什么correct顺序在结构中不可删除？
3. language如何提供query但不能独立写LoRA？
4. 多video如何聚合共同语义而不平均低层信息？
5. Program如何写成material、native且support-preserving的完整LoRA？
6. 不同task/video能力如何在同一checkpoint共同积累？
7. 它继承了v6/LPCP/GOMQ哪些已证实优势，避免了哪些已测失败组合？
8. 什么机制证据会快速否决，何时必须跑strict paired400、相邻checkpoint和六臂？

在八个问题没有形成闭合答案前，memory token、rank16、V6 compiler或任何新decoder都只是候选部件，不是架构结论。
