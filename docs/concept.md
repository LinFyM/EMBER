# EMBER concept

EMBER研究能否把exact task language与action-hidden正确教学视频，在rollout前一次编译为冻结π0.5 source的一套完整task-conditioned LoRA，
使机器人从未见初始化闭环完成任务。语言说明目标与关注对象，视频中的操作内容和顺序应提供必要条件信息。
人从他人教学迁移到自己身体的能力是科学动机；LIBERO结果本身不证明跨身体泛化。

当前授权和实际状态见[progress](../progress.md)。Owner已授权统一Writer的实现与整套比较实验，旧A/C等历史路线不自动恢复。
本轮采用[统一Writer设计](v52_evidence_based_writer_design.md)：继承v5.2的内容、条件作用、归一化和共享参数坐标，
用原生中层/末端的同构联合Z/H处理与连续参数槽，替换独立Core/P及专门融合。它尚未实现或训练，不赋予旧checkpoint分数。
下述原理中涉及Core/P的具体形式，是已实现v5.2的历史解释；最新候选的完整接口、学习和边界以上述设计为准。
Process Pullback及其它已关闭机制从[研究历史](research_history.md)追溯。

## 当前候选：统一表示与一次性参数生成

```text
exact language + K1同步双RGB，stride5和真实末帧
  → 原生前9层：真实prefix、完整H50和三组读取Meta
  → task-span/真实patch与H组成同一网格，联合block×2
  → 同时残差写回Z/H，原生后9层继续解释
  → 同构联合block×2形成唯一memory M，保留语义和全部horizon位置
  → 同构参数decoder：先从M语义位置建立内容，再以该内容读取全部M
  → 同一slot状态、归一化和八组共享完整A/B heads
  → 一套LoRA在执行时根据机器人自己的观测产生状态条件化控制
```

丰富视觉内容、跨帧证据和原生动作知识仍有明确因果作用；统一不等于删除这些作用或只保留动态差分。
首层参数读取的语义memory已与H和完整视频联合处理，不是新的静态Core。三个Meta、联合块、decoder与head共同接受跨episode真实FM。
内容保留、共享及归一化是学习偏置，不是视频必用、泛化或保持的保证；这些行为资格沿原科学目标独立验证。

## 已实现v5.2的历史处理框架

```text
exact language + 一条同步agentview／eye_in_hand RGB视频（K=1）
  → stride5完整视频及真实末帧，每时点两相机共512个真实patch tokens
  → 冻结π0.5基础 + 独立Text／VL／Action读取Meta
  → exact text queries、逐帧图文内容Z、完整50-horizon动作响应H
  → language查询真实视频patch与task-span内容，形成语义Core
  → 共享learned horizon read把H形成逐帧动作响应摘要，真实顺序Procedure理解其变化
  → 320个rank slots读取Core与居中的Procedure，后者通过AdaLN调制前者
  → 八类共享family heads输出唯一38-target、rank16完整A/B LoRA
  → 冻结source根据机器人自身当前观测闭环执行，LoRA全程固定
```

两相机是同一时点的同步观测，不是K=2，不平均视角或原始frames。Teacher action、state/proprio、reward、terminal、
task ID与文件名均不进入模型。执行policy读取机器人自身观测/state，与teacher信息墙不同。

## 历史v5.2：各模块传递什么

Text Meta只处理exact language，形成对齐的任务查询。VL Meta在真实双视角native prefix上提供任务上下文与图像位置内容；
Core的Value来自这些真实视频内容。它允许共享的对象、关系和任务语义支撑动作，不能因为后继窄出口失败而删除这条已存在能力的路径。
语言决定关注位置，却没有独立Value出口直接生成有效LoRA。

Action Meta通过相同真实prefix和一个对全部任务共用的固定50×32 Gaussian probe读取原生Action Expert。
这里的H是条件化动作计算响应，不是从视频恢复的teacher action，也不是待执行的动作标签。完整50个位置保留到实际learned read：
一个共享内容查询与50个相对位置bias对raw H加权，然后接原投影。查询与bias从零开始，初始函数等价于旧mean；
随后监督可学习各horizon内容和位置的不同作用，没有在读取前截断或平均H。

Procedure沿真实frame indices做因果注意力，表达过去条件对后续操作的限制；Core与Procedure在同一套Writer里融合。
这是一种有顺序的归纳偏置，尚不证明网络有益地理解“先A后B”。Video time、action horizon、flow time和layer depth各有独立含义。
这是历史v5.2实际计算图的因果Procedure；本轮候选已将其职责移入联合表示，不能从本段恢复旧方法选择。

共享rank slots通过Core Value产生内容，再由居中的Procedure Value调制；完整A和B均由共享family heads生成。
输出没有被限制在固定source局部PCA/span中，也不存在第二expert或并行adapter。
原生Action Expert提供理解视频的动作先验；最终参数作用仍由完整非线性policy及真实闭环判断。

## 历史v5.2：怎样共同学习

Writer与三组Meta从fresh参数／优化状态共同训练；基础LoRA为合法A模板、B=0，各head末端、Meta B和AdaLN按原图零初始化。
首步先使输出head离开零，后续更新逐步打开上游信用。实际profile须覆盖这一启动过程，不能因第一步上游为零就判定断图，
也不能仅靠随机解锁后的小图梯度宣称真实训练已连通。

唯一训练目标是同task、严格跨episode的主执行FM：每个video产生一套LoRA，监督其在同task其它episodes的真实观测上预测未来动作。
动作观测对齐遵循现行offset1合同；归一化只用冻结source统计。每更新4个不同task，各21个queries，先按task等权再统一更新。
Text/VL/Action Meta、Core、Procedure和完整A/B heads的信用均来自同一参数版本的FM。
实现可先获得完整LoRA余切，再重放Writer并用frame checkpoint限制激活内存；source冻结，适配的Z/KV/H不跨更新缓存。

跨episode监督、共享图文／动作坐标和共享heads是尝试获得可复用能力的理由，不保证迁移或保持。
继续训练可能获取新行为，也可能破坏已经成功的条件。此前同事件任务共现对照已作为历史证据保留，
不再自动继续。当前方法见上方统一候选；其新实验独立按当前active design执行。

## 怎样判断

首先看正确视频的single-checkpoint strict paired400绝对能力，长期正式资格为严格>145/400及相邻稳定、
低churn、高breadth、四suite贡献和Goal/Long能力；训练loss或较低参数漂移不替代闭环。
正式验证每task每轮全50条teacher各一次，state–video与policy RNG跨节点固定；train96是另一个明确登记的独立视频有限面板。

有能力及相邻证据后验证same-task换视频，再冻结单checkpoint做最终wrong／no-video／shuffled／reversed controls。
后者重排真实frames后完整生成，不参与训练或选点。历史A阶段另获owner授权，按预注册节点观察特异性演化并用于问题分析；
该实验现已暂停，其原合同不构成恢复授权。历史封闭诊断和Test不追溯反哺，Test默认封闭，未来使用须先登记方法冻结。
历史恢复、训练任务新获取及局部保持可以是正证据，却不等于通过最终资格。连续有信息量节点没有改善时停止该假设，
不靠无依据的种子、rank、scale或学习率小扫掩盖能力缺口。
