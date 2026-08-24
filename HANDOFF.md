# EMBER temporary handoff

此文件只服务这次跨session交接。专家回复尚未到达；当前session收到回复后应更新本文件并推送。新session完整消费后删除它，
不要把它变成长期状态文档。

## 1. 当前交接点

- canonical branch：`main`；本轮起点`7ab5a04`，清理提交号见Git最新记录；
- 没有新的ECP训练或GPU job；
- 活动架构方向仍叫ECP，不叫PECS；GOMQ与ECP没有阶段依赖；
- 人工process数据路线已取消并清理；后续只使用现成授权LIBERO tasks；
- owner已向专家发送复核请求，等待回复；未经owner明确允许，不得自行联系或回复专家。

## 2. owner发给专家的要求

> 我们不制作数据集，就用现成的LIBERO task。我们没有那么多时间浪费，后续的推进直指ECP的核心。关键的几个点，第一，
> ECP的架构再次明确清楚，即使现在的设计没有问题，也再次明确一下；第二，接下来的具体的几个推进阶段规划清楚，到底用
> 什么数据，构建什么样的模型，目的是什么，要验证什么事情，怎么算通过；第三，最终的ECP应该是什么样的，如何根据前序
> 的推进阶段构建出来，用什么数据训练什么模型。我们接下来的推进，核心的目标就一个，根据之前讨论分析得出的，以现有条件
> 能分析出来的比较合理的ECP架构，经过必要的几个阶段的推进，解决之前几十版架构实现遇到的各种问题，最终实现既定输入
> 信息下性能的稳定提升。

不要再给专家重复他已经知道的完整背景，也不要给他设置“单变量下一步”“三个月计划”等限制。需要的是他对ECP核心架构、
识别问题、训练流程和可行性的直接判断。

## 3. 专家回复到达后的处理

当前session：

1. 保存回复原文或不失真的摘要；
2. 对照`docs/event_conditioned_policy_compiler_design.md`列出confirmed / changed / rejected / unresolved；
3. 把唯一最终架构具体到每层输入、中间shape、输出LoRA，以及每阶段数据、训练、冻结、实验、耗时级别、通过条件和失败分支；
4. 更新`task_plan.md`、`findings.md`、`progress.md`和本文件，提交并推送`main`；
5. 停止，不自行开始下一版本，让owner切到新session。

新session：

1. 按`AGENTS.md`完整读取mandatory docs；
2. 核对expert ruling已经进入active design和plan；
3. 删除本文件并提交；
4. 只有owner明确要求设置goal时才调用goal机制；
5. 按冻结计划自主推进，不把routine实现选择反问owner。

## 4. 不可再犯的路线混淆

- ECP核心：language+action-hidden videos -> Program posterior -> shared realizer -> one complete LoRA。
- GOMQ：独立历史baseline，rank16 136/400；不属于ECP Phase 0，也不需要重跑。
- PECS：旧privileged effect/solver诊断，已关闭；不是ECP的别名。
- v24：旧Stage 1 compiler的一版；其失败只说明那类compiler没解决映射，不定义ECP本质。
- q_pi：训练期共享Program posterior网络，不是凭空正确的teacher或手工code；必须用task-disjoint closed loop证明。
- held5：train24内部leave-task-out机制门；validation8才是正式开发选择。
- Program event slots：最大`E=8`固定，激活数量和视频段落分配动态学习。
- Action Meta-LoRA：首版可不用，但后续必须matched尝试；无负面且有净收益就启用并冻结。
- shuffled/reversed：只在最终冻结checkpoint测试时序特异性，不用于训练或选模。
- staged training不是最终割裂模型；通过后必须有冻结PI0.5 backbone的全Writer联合训练。

## 5. owner执行偏好

- 效率优先，但不能为了快丢掉专家关键意见；先思考接口，再实验。
- 不要连续盲目迭代版本，不要钻矩阵shape、低位浮点、防御性代码或冗余测试。
- 复用已有长训练资产；明确坏路线停止，不用超长续训或小超参扫挽救。
- 单job一个节点最多6张A40；EMBER总量通常不超过6，大量空闲时最多8；每次live检查两节点。
- 可安全共驻低显存/低util卡，不得kill/reset他人进程；gpu01物理0若仍prohibited则不用。
- 关键节点汇报即可，不频繁打断；遇到困难先回看专家原始意见和当前设计。
- canonical是`main`；必要时开`codex/*` worktree，验证后及时合并、推送、清理。
- 代码、文档、配置、branch、worktree和运行产物同步清理；活动树只保留一个实现面。
- owner问具体问题时先直接回答，不擅自扩大成方案或要求owner批准routine决定。

## 6. 当前仓库面

保留：source/corpus/SFT、LoRA、task experts、ECP Stage 0、video sampling、functional loss、reward occupancy、dynamic evaluator。

删除：旧Writer、functional decoder、ECP Stage 1/MDCO/PECS、fixed/two-sided realizers、projected historical adapters、人工process
代码/配置/数据，以及逐轮设计卡和重复证据JSON。精确历史通过`docs/research_history.md`和Git恢复。

专家回复前，`docs/event_conditioned_policy_compiler_design.md`中的realizer与`q_pi`训练顺序仍是开放点，不能据此launch。
