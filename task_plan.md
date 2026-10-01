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

## 独立仓库整理的完成条件

本整理分支按Owner实际执行授权完成全仓源代码、测试、脚本、配置、入口、文档与相关临时内容的依赖和生命周期整理：

1. 删除已证实过时、重复和失活的实现/专用工具；保留当前唯一Writer及实际共享/冻结读取消费者。
2. 修正错误与失效引用，使稳定要求、当前状态、跨轮发现和历史索引各自承担唯一职责；原始科学证据不丢失。
3. 完成与改动相称的实际入口、import/config及既有CPU检查；不启动GPU实验或建设平行审计框架。
4. 将实际遇到的问题落实到已安装个人workspace-cleanup skill，精确维护并保留data1回滚及验证证据。
5. 承接最新main，与实际写入方串行集成、推送并清理task-owned临时树，向主讨论一次交付实改、证据及剩余未知。

本任务按Owner纠正使用goal；不扩展实验范围、科学预算或部署输入，不在实验的开发/冻结树中写入。

## 历史

多轮旧执行计划的完整快照为`git show 797ae01f:task_plan.md`；设计、实际结果、负证据和失败原因沿
[research_history](docs/research_history.md)追溯。历史未完成清单不是现行待办；不会靠整理覆盖其当时事实。
