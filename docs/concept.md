# EMBER concept

## 从教学过程到自己的策略

EMBER探索：从一条或多条action-hidden正确教学视频理解任务条件和操作变化，结合exact language，
在rollout前一次性编译成冻结π0.5 source的一套完整task-conditioned LoRA，使机器人从自己的新初始化闭环执行。
语言确定目标与关注对象，视频动态必须贡献必要信息；执行行为由机器人当前观测触发。跨具身是科学动机，LIBERO结果不自动证明跨具身泛化。

## 正确教学增量与方法证据

正确教学视频必须同时满足同任务和内部顺序正确。研究要求正常训练自然形成correct相对错任务、乱序视频的有益闭环增量，且correct高于冻结source；不能靠人为压低错误条件制造差额。

v5.2普通正样本动作监督已有正确视频与顺序依赖正证据；纯监督允许捷径不足以解释不同模型的行为。完整H、前缀依赖、attention和真实梯度只证明计算可用，不能证明操作证据已被有益消费。

Horizon与语义条件化消费C的形成过程、数学性质及未经证实的推论见[机制复核](video_mechanism_reassessment.md)。C/S只观察100/200两个节点，其后续学习潜力未被充分检验。95-task提案不构成解决视频消费问题的证据；当前授权与阶段只看[progress](../progress.md)和[task_plan](../task_plan.md)。

## 保留的完整H过去定向数据流

```text
exact language + K条独立有序视频
  → frozen vision/Gemma真实prefix：逐帧最终Z、KV与exact task-token mask
  → 单固定probe、flow_time=1、Action Expert + shared observer Meta
  → action_out_proj实际输入：最后50个post-norm hidden tokens
  → 每帧上下文task tokens经现有reader形成该帧过程条件
  → 四组：过去4帧对应 → 沿完整H联合形成query → 两端Z视觉核实
           → 按历史u从早到晚短GRU → 完整H状态
           → 临时H-read → 单向长程时间组织
           → 前三组逐H条件化回写；第四组直接送出
  → 多视频集合compiler，608个paired target/rank queries
  → native因子读出 → 唯一38-target完整rank16 A/B
  → 冻结source依据自身观测闭环执行
```

四组使用当前帧上下文task-token条件，不跨帧池化；Compiler额外静态语言query保持关闭；精确language仍通过逐帧原生task-token条件进入过程读取。语言读取共享reader参数，原生Gemma/vision仍冻结，完整时序与视觉Value通路保留。

完整数学、张量、训练与迁移合同见 [正式设计](horizon_relation_video_writer_design.md)，
原文与Owner裁决见 [讨论索引](review_materials/20260908/README.md)，实施状态见 [progress](../progress.md)。
本文记录方法合同；已实现接口、实际性能与后续执行状态只由progress及对应formal evidence确认。

## 三类有序关系与因果职责

1. **Horizon h：动作计算的相对位置。** 末层50个hidden仍不是已经采样完成的正确未来轨迹。
   软对应首先提出跨起点的候选关系；新形成的匹配内容、位移分布和不匹配模式再沿完整H联合处理，产生50个视觉query。
   H-query的作用是让一处视觉查询能参考其它位置的新对应模式；不把匹配斜带或hidden差直接解释成真实过程/速度。
2. **邻帧 u：以当前t为共同终点的历史证据。** 每条 `[u,t]` 关系先读取过去与当前的原生Z核实，再按u从早到晚进入短GRU。
   区间有重叠，递推解释证据之间的支持、冗余或修正；它不是动作积分，也不自动消除重复计数。
3. **视频 t：完整任务过程。** 每组临时H-read形成时间tokens，长程层只读当前及过去；前三组通过当前h状态条件化回写，
   让更长的过去上下文帮助下一轮局部对应和原始视觉读取。完整U始终保留到需要的最后读取，不用复制压缩token恢复H。

Owner最终选择**过去局部＋过去单向长程**；每组U_t只依赖原视频前缀。H-query允许同一帧完整H双向交互，不能混用两种mask。
专家原文曾建议双向长程，其未来证据反馈的论证不属于当前方法。计算前缀性质不能证明视频对最终行为有必要作用。

## 共同解释与完整参数生成

每条video先独立保序编码，只有集合阶段置换不变地共同读取。不混淆video内部时间和video集合次序，不平均frames、raw features或最终LoRA。
Compiler首块以task-independent target/rank身份作为残差内容，较强off候选关闭了额外language query；exact language仍通过逐帧Z/R和过程条件进入P4，第二块继续共同编译。真实P4 Value不保证视觉动态必要性。C则另用逐token视觉语义先确定过程读取条件，再融合语义与变化生成LoRA；两者的实现与证据须分开解释。
Compiler的target/rank身份决定输出位置，输入不必保留18个网络层才能生成38个目标。末层未保留的信息也不能由compiler凭空恢复。
Native D按target/rank/side独立、跨任务共享，允许更直接的因子学习通道；它仍有共享干扰和固定读出空间，不能被视为性能保证。
参数在rollout中固定，作用于随机器人观测变化的激活，因此可以形成状态条件化行为；不能按教师视频时钟播放动作。

## 学习与裁决

Writer（含内部读取模块Meta） fresh、端到端纯FM监督，source基础冻结；FM来自同task另一episode的actions。
监督阶段无RL更新、探索rollout或trust回滚，真实闭环与独立验证共同判断能力和平台。
达到有证据的平台后，才从单个保留的监督checkpoint接独立共享Writer RL，采用新optimizer/scheduler，默认不混FM。
监督充分仍弱要先定位机制缺口，不能只宣布饱和后交给RL救场。训练期共享RL不同于部署时task-local优化。

早期强Writer、task专家与G1/G2提供不同层次的正证据；后续shared/clone差距及384的失败说明共享行为仍未解决。
新图没有继承它们的分数，也没有由数学依赖证明操作理解。历史与适用边界见 [research_history](research_history.md)。
唯一正式性能线是validation8 single-checkpoint strict paired correct>145/400，并满足相邻、跨视频、breadth、四suite、Goal/Long及最终视频因果要求。


同步双视角是同一教学演示的两路RGB观察。`observer.camera_view=dual`将第三人称与腕部画面在同一时间点送入原生双相机prefix，形成共同Z/KV和一份完整horizon响应；不把相机数当作K、不平均视角LoRA。该读取能力不预先证明双视角的闭环收益。
