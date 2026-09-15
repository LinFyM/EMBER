# EMBER concept

EMBER研究能否把exact task language与action-hidden正确教学视频，在rollout前一次编译为冻结π0.5 source的一套完整task-conditioned LoRA，
使机器人从未见初始化闭环完成任务。语言说明目标与关注对象，视频中的操作内容和顺序应提供必要条件信息。
人从他人教学迁移到自己身体的能力是科学动机；LIBERO结果本身不证明跨身体泛化。

当前方法和执行边界见[v5.2恢复与保持对照](v52_return_plan.md)，状态见[progress](../progress.md)。
以下解释现行fresh基线；旧单相机／horizon mean checkpoint只作为固定历史参照，不把旧分数赋给迁移后的新模型。
Process Pullback及其它已关闭机制从[研究历史](research_history.md)追溯。

## 从视频到一次性策略参数

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

## 各模块传递什么

Text Meta只处理exact language，形成对齐的任务查询。VL Meta在真实双视角native prefix上提供任务上下文与图像位置内容；
Core的Value来自这些真实视频内容。它允许共享的对象、关系和任务语义支撑动作，不能因为后继窄出口失败而删除这条已存在能力的路径。
语言决定关注位置，却没有独立Value出口直接生成有效LoRA。

Action Meta通过相同真实prefix和一个对全部任务共用的固定50×32 Gaussian probe读取原生Action Expert。
这里的H是条件化动作计算响应，不是从视频恢复的teacher action，也不是待执行的动作标签。完整50个位置保留到实际learned read：
一个共享内容查询与50个相对位置bias对raw H加权，然后接原投影。查询与bias从零开始，初始函数等价于旧mean；
随后监督可学习各horizon内容和位置的不同作用，没有在读取前截断或平均H。

Procedure沿真实frame indices做因果注意力，表达过去条件对后续操作的限制；Core与Procedure在同一套Writer里融合。
这是一种有顺序的归纳偏置，尚不证明网络有益地理解“先A后B”。Video time、action horizon、flow time和layer depth各有独立含义。
本轮保留v5.2实际计算图的因果Procedure，不因为允许完整视频双向理解就无依据更换它。

共享rank slots通过Core Value产生内容，再由居中的Procedure Value调制；完整A和B均由共享family heads生成。
输出没有被限制在固定source局部PCA/span中，也不存在第二expert或并行adapter。
原生Action Expert提供理解视频的动作先验；最终参数作用仍由完整非线性policy及真实闭环判断。

## 怎样共同学习

Writer与三组Meta从fresh参数／优化状态共同训练；基础LoRA为合法A模板、B=0，各head末端、Meta B和AdaLN按原图零初始化。
首步先使输出head离开零，后续更新逐步打开上游信用。实际profile须覆盖这一启动过程，不能因第一步上游为零就判定断图，
也不能仅靠随机解锁后的小图梯度宣称真实训练已连通。

唯一训练目标是同task、严格跨episode的主执行FM：每个video产生一套LoRA，监督其在同task其它episodes的真实观测上预测未来动作。
动作观测对齐遵循现行offset1合同；归一化只用冻结source统计。每更新4个不同task，各21个queries，先按task等权再统一更新。
Text/VL/Action Meta、Core、Procedure和完整A/B heads的信用均来自同一参数版本的FM。
实现可先获得完整LoRA余切，再重放Writer并用frame checkpoint限制激活内存；source冻结，适配的Z/KV/H不跨更新缓存。

跨episode监督、共享图文／动作坐标和共享heads是尝试获得可复用能力的理由，不保证迁移或保持。
继续训练可能获取新行为，也可能破坏已经成功的条件。因此基线之后的有界对照只改变同一批训练事件的任务共现，
保持模型、初始化、事件、queries、更新总数和全局学习率时间轴。它是未验证的假设，不是已知修复。

## 怎样判断

首先看正确视频的single-checkpoint strict paired400绝对能力，长期正式资格为严格>145/400及相邻稳定、
低churn、高breadth、四suite贡献和Goal/Long能力；训练loss或较低参数漂移不替代闭环。
正式验证每task每轮全50条teacher各一次，state–video与policy RNG跨节点固定；train96是另一个明确登记的独立视频有限面板。

有能力及相邻证据后验证same-task换视频，再冻结单checkpoint做最终wrong／no-video／shuffled／reversed controls。
后者重排真实frames后完整生成，不参与训练、选点或架构修正。Test默认封闭，未来使用须先登记方法冻结。
历史恢复、训练任务新获取及局部保持可以是正证据，却不等于通过最终资格。连续有信息量节点没有改善时停止该假设，
不靠无依据的种子、rank、scale或学习率小扫掩盖能力缺口。
