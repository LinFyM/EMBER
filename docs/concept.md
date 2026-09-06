# EMBER concept

## 从观察到自己的策略

EMBER的出发点是：正确教学视频通常没有与接收者兼容的action labels，示范者甚至可能具有不同身体。人仍能从视频理解
“关注什么、在什么条件下做什么、过程如何推进”，再把这个知识应用到自己的动作能力上。

本项目探索一种参数化实现：以冻结的π0.5-LIBERO source policy作为具身先验，用共享Writer把exact task language与一条或多条
正确教学视频，在rollout前一次性编译为完整task-conditioned LoRA。执行时policy只根据自己的观测闭环行动；Teacher视频不再输入，
没有目标task上的试错或优化。LIBERO是当前检验平台，跨人类/跨具身泛化仍是动机，不能由该平台结果自动宣称实现。

## 三个必须接上的职责

1. **让已有动作知识帮助理解画面。** 冻结vision/Gemma产生每帧原生图文prefix，读取侧Action Expert共享Meta-LoRA适配无proprio的
   教学输入。保留各层和完整50-horizon条件响应；一次flow端点前向提供响应结构，不是已经生成完的正确动作轨迹。
2. **从连续画面形成有方向的过程证据。** 在每个计算层内，按内容、帧间隔与horizon位移建立窗口内帧对关系，两端分别读取。
   每个关系MLP同时解释对应内容和对应位置模式，再由每帧聚合自己的邻居消息并逐层更新，最后做任务条件化horizon读取。
   视频时间、action horizon与计算深度各有不同含义；对齐后的内容差为零，也可能发生了对应位置的推进。
3. **把过程编译成可闭环使用的参数。** 每条视频独立编码，集合阶段联合理解证据，用共同的整策略queries协调整套LoRA，最后
   在明确的原生目标/通道坐标上生成全部A/B。参数在rollout中固定，行为阶段由执行policy当前观测触发，不能用teacher视频时钟驱动。

这三个职责是否实际实现，必须由可复核干预和闭环证据判断。Full horizon、梯度接通或模块名本身没有完成证明。

## 当前对齐的候选数据流

```text
exact language + K条action-hidden有序videos
    → 每帧 frozen vision/Gemma native prefix
    → Action Expert + shared observer Meta-LoRA
      单个固定public Gaussian probe，flow_time=1
    → R[k,t,j,h]：18个已读图层状态 × 完整50 action horizon
    → 同层局部帧对50×50关系，两端分别softmax
    → 对应内容 + 相对位置模式 + signed gap → 关系MLP
    → 每帧最多8邻居的attention聚合，residual/FFN，同型block堆叠
    → 每(t,j)有任务条件的horizon read → E[k,t,j]
    → 置换不变的视频集合读取 + 全局整策略queries
    → 原生坐标条件MLP → 唯一38-target完整rank16 LoRA
    → 冻结source policy在自身新初始化中闭环执行
```

完整推导、shape、可执行默认、GPU梯度算法与已有代码地图见
[layered_relation_video_writer_design.md](layered_relation_video_writer_design.md)。设计已完成讨论，实际实现与科学执行授权见
[progress.md](../progress.md)。这张图尚没有新的性能证据，不继承旧Writer的分数。

- 单probe是当前最小方案；额外probe只有独立用途与实际收益时再考虑。
- Gemma语义经原生prefix进入Action Expert；不假定R或压缩E无损保留全部语义，也不无依据再叠一条R→Z读取。
- H保持relative action time，J保持计算层身份；原生action horizon注意力保持π0.5语义。视频在rollout前完整可用，
  允许双向局部读取，以signed gap和对应模式保留先后关系；不强制单位矩阵、固定平移或把斜对角形状当成动作理解证据。
- 观察侧Meta参数跨任务共享，单次编译内固定；执行侧只装生成的一套LoRA。观察侧激活不等于实际执行状态下的X/Y。
- 不额外强制读取完整raw X/Y bank，不限制最终因子处于其signed span。真实policy功能梯度提供原生参数坐标的学习信号。
- 多视频带来互补证据和削弱独立干扰的机会；相关误差、不同有效策略与学习不足会限制收益，不能保证K增大后每次性能都提升。
- 局部过程模块的双侧感受野有限；radius4堆叠4层最多前后各16个采样间隔。整段远距离证据由最终全局queries共同读取，
  不能把每个局部E说成完整任务程序；不得用全视频汇总或Q反馈暗中扩大局部上下文。

## 已有证据能支持什么

早期v5.2/v6的端到端视频到LoRA路径达到过真实闭环能力，其中v6 strict correct为143/400，但后续相邻结果下降。
Task-local rank16专家250/400说明执行LoRA存在容量；G1说明特定native-factor表示存在局部可达性；G2说明完整有序响应具有动态信息。
最近完整输出重构带来过训练侧Goal收益，同图单task学习也优于18task共享实例；这些都没有解决稳定共享迁移。

新候选吸收上述证据，同时修正“保留原生响应就等于用好动作时序先验”的推理跳跃。其主要未解问题仍是：过程表示是否足够，
共享优化是否能把它变成有用的参数，以及从多个任务学到的映射是否能迁移到未见任务。

唯一正式性能目标是validation8 strict paired correct严格 >145/400，并满足相邻与跨视频稳定性、低churn、高breadth、四suite非零、
Goal/Long贡献和冻结后视频因果controls。详细边界见 [current_owner_requirements.md](current_owner_requirements.md)，
历史脉络和证据入口见 [research_history.md](research_history.md)。
