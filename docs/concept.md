# EMBER concept

## 问题定义

人看过一段没有动作标注的教学视频，通常会先理解目标，再把视频中的条件、过程和结果迁移到自己的身体与当前场景。EMBER
研究PI0.5能否做同一件事：只看task language和`K`条action-hidden正确视频，在rollout前把观察到的知识编译成Action
Expert的一套LoRA，随后零交互完成任务。

这不是视频检索、task-ID分类、行为克隆或运行时视频条件策略。部署时没有teacher action、state、reward和第二个expert；
Writer只运行一次，输出的参数必须直接成为闭环策略的一部分。

## 为什么问题困难

原生PI0.5中，Gemma处理当前language和静态图像prefix，Action Expert把50个未来horizon位置上的noise tokens通过flow
matching推进为动作chunk。教学视频则是一串跨时间的静态帧，而且没有teacher actions。EMBER必须同时解决三个接口：

1. 从帧级PI0.5表示中提取与动作过程相关、而非只识别物体或task模板的动态证据；
2. 从可变长度、可变`K`的视频中学习保留过程信息、可支持整套策略修改的工作表示；
3. 让共同表示联合生成Action Expert完整LoRA，在共享训练后仍能迁移到新任务；native evidence用于理解与条件化，
   不再强制将输出限制为raw X/Y的signed span，也不得把held更新投影回fit-task字典。

训练task数量有限还会造成欠识别：language、video和task identity可能高度相关，模型即使完全忽略过程也能降低训练loss。因此
方法必须靠task-disjoint评测、视频controls、多个独立策略lineages和真实closed-loop结果证明因果路径。

## 方法方向与当前证据

Action Expert的原生动作时序知识应是视频过程理解的核心。Gemma逐帧提供图文语义，Action Expert提供当前视觉条件下的动作生成
响应；这两类证据都不自动构成整段视频理解。source在目标任务上可能失败，其响应不是正确动作真值。学习方向来自授权non-held
任务上的真实actions、privileged专家或训练期行为信号，不能偷渡到deployment输入。

早期v5.2/v6已证明一条端到端视频到LoRA路径能够产生真实闭环能力，后续缺少稳定积累；G1证明部分native-factor容量，G2证明
ordered response有可学习动态信息。局部接口的正证据不等于整套共享Writer通过，后期弱模型也不能抹掉早期能力。

最新owner授权复用共同过程状态P与整策略状态Q，直接联合生成38-target完整LoRA，无独立carrier。首选rank16，rank8由实际成本和行为证据决定。
active design见`docs/joint_process_policy_writer_design.md`，状态以`progress.md`为准。旧A2/P/Q实例的负证据保留，新的输出重构尚待实际检验。

## 数据流与模块职责

```text
exact language + K internally ordered action-hidden videos
  -> frozen per-frame PI0.5 image/language/action-response capture
  -> per-video learned process states P[frame, work-token, width]
       <-> whole-policy states Q[target, rank, X/Y side, width]
       repeated attention/MLP; re-read full native evidence
  -> permutation-invariant learned set read
  -> family-shared learned factor heads
  -> complete38-target A/B, one rank16 materialization
  -> frozen execution policy; no further Writer call
```

P由语言条件化图像读取形成对象/关系grounding，再读取完整原生响应并沿teacher time交换信息。Q负责协调整套参数修改，读取P，
并反馈到下一层过程读取。两者联合接受真实policy功能梯度，不使用独立冻结的固定Program tuple作为唯一中间瓶颈。
所有learned主干保持少数职责清楚、可复制扩展的attention/MLP模块；不再叠加summary、covariance solve、whitening、transport、
anchor或family gate的连续坐标链。Q跨target交流是候选机制，不能由存在attention就宣称解决多任务共存。

## 时间轴、信息墙与动态证据

teacher-video time、relative action horizon、flow time、layer depth、probe是不同轴。每个视频帧保留19个layer boundaries、
完整50 horizon及两个固定antithetic probes的原生响应，直到task-conditioned learned read才压缩；禁止horizon mean、coarse或
等价无条件平滑。s=1是噪声端点，响应不是教师未来50帧或已经去噪的正确动作；不能设`t+h`统一时钟或把网络深度当任务阶段。

语言与结构身份条件化读取，真实视频语义和有序过程共同参与完整生成。位置用于时间路由，不能由位置/帧数冒充动态证据。
完整LoRA不继承mobile-only静态零输出约束；视频动态的必要条件增量由最终冻结controls证明。G1的native局部容量正证据继续保留。
每条视频独立保序，集合阶段置换不变；不平均raw frames/features或最终LoRAs、不挑video、不拼视频时间、不重复凑K。

教学视频路径没有teacher actions、state/proprio、reward、terminal、task ID、filename或pose。执行policy按官方合同使用自身当前
观测/state。Writer只在rollout前运行一次，该调用内部允许固定只读重放同组视频/native evidence；闭环中无task-local优化或第二adapter。

## 输出与训练

唯一部署输出由Writer联合生成38-target全部A/B，首选rank16，无独立carrier或第二adapter。native responses是重要证据，
最终读出采用learned heads；非零A/零B初始化后全部可训练。解除signed/native输出限制是候选假设，不是已确认的性能根因。

最初使用正确视频生成LoRA、同task跨episode action query的真实flow loss。训练task名单、权重、normalizer、video/row occurrence与
optimizer cadence必须明确。clone/shared对照使用相同图、初始化和可训练模块；clone仅是能力诊断，不能部署或冒充共享泛化。
有共享行为信号后才按实际缺口研究成功行为保持、learner访问状态或reward；先回查SEOD/GOMQ/guard的等价尝试。

Final保留同拓扑component-init和fully-random fresh端到端候选，不强制重演G1--G3冻结课程。Action Meta首轮关闭，若证据指向
observer输入域不足再作有限matched审视；基础权重冻结，读取侧适配与执行native bank坐标必须明确，缓存不得跨learned observer更新失效。

## 裁决与未知

train24 SFT历史109/400与旧Writer143/400是实质能力参照；正式比较须对齐合同，不能只超过弱source/carrier。
最终标准仍是validation8 single-checkpoint strict paired correct严格>145/400，并由相邻稳定、低churn、breadth、四suite与
Goal/Long贡献、same-task新视频鲁棒性和最终视频因果controls共同证明。局部loss/rank/cosine改善不能替代闭环，union不能部署。
fixed validation/test不产生梯度；shuffled/reversed只在selected checkpoint冻结后检验，不用于训练、选择、内部Gate或架构修正。

仍未知的是：当前部署图的基础可学习性与共享困难分别有多大，哪种过程—策略映射能稳定积累行为，以及现有授权数据能否识别强
视频过程必要性。授权train侧的过程/组合与功能歧义审计不能由元数据数量代替；即使取得高总分，视频无必要性也不构成EMBER完成。
低于/接近baseline时应比较竞争解释，必要时多做有区分力的分析实验；不把一个可疑现象命名为根因后随手修补。
