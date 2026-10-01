# EMBER task plan

## 科学目标与当前批次

从冻结π0.5 source出发，利用exact language及action-hidden教学视频生成初次即有效的一套完整task LoRA，
由严格配对闭环证明绝对能力、相邻保持和有益教学增量。稳定边界见
[Owner要求](docs/current_owner_requirements.md)、[AGENTS](AGENTS.md)与[concept](docs/concept.md)。

当前唯一active design为[条件读写§13](docs/designs/conditional_read_write_architecture.md#13-首批实施与完整学习检验2026-10-01授权)：

1. 实验执行者已完成最长视频成本检查与90完整ECP；按实际可用卡数自动分配，从完整边界承接91..450，
   保持112有效query及完整FM梯度。全部失败、占用与恢复计入40GPUh/80GiB，不fresh重做已完成更新。
2. 唯一450科学节点完成validation400、train144和固定A28，保存完整原行、分布、paired R/G/L、churn及成本。
3. 主讨论解释训练获取、held迁移、能力得失与机制预测，结合最近似完整历史决定最小后继；
   不自动900、参数小扫、controls/Test/RL，不把单点增益写成稳定性或因果确证。

工程/运行/Git由实验session闭环；主讨论负责科学判断。整批科学结果或确需裁决的实质边界按Owner要求§6回报；
其它工程阶段事实保存在已有记录。实际代码、冻结、运行与窗口状态只看[progress](progress.md)。

## 仓库维护

本轮独立整理的实际改动、保留依赖、536项CPU检查、合并后38项消费者检查、已安装skill维护与回滚位置见
[研究历史](docs/research_history.md#2026-10-02整仓整理与历史状态入口)。整理承接最新实验修复，交付身份见progress与Git。
没有改动当前实验模型、数据/标签、FM、信息墙、选择合同或预算；科学下一步仍按上面的唯一批次推进。

## 历史

多轮旧执行计划的完整快照为`git show 797ae01f:task_plan.md`；设计、实际结果、负证据和失败原因沿
[research_history](docs/research_history.md)追溯。历史未完成清单不是现行待办；不会靠整理覆盖其当时事实。
