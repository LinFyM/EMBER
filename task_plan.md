# EMBER task plan

## 科学目标与当前批次

从冻结π0.5 source出发，利用exact language及action-hidden教学视频生成初次即有效的一套完整task LoRA，
由严格配对闭环证明绝对能力、相邻保持和有益教学增量。稳定边界见
[Owner要求](docs/current_owner_requirements.md)、[AGENTS](AGENTS.md)与[concept](docs/concept.md)。

当前唯一active design为[条件读写§15](docs/designs/conditional_read_write_architecture.md#15-固定目标曝光的辅助任务分布短窗检验2026-10-02)：

1. 已完成§13 fresh450及§14固定图900。validation137→140/400、seen91→115/144；
   训练侧确有对象选择和多阶段获取，held主要损失仍在3/31，完整优势及相邻资格未形成。原同池续训停止。
   主讨论完整分析见机制§95/findings§270及900证据JSON；两批成本31.737894GPUh，旧原件保持。
2. 科学登记唯一D71短窗：从450恢复共同学习状态，到630新增180更新，仅将支持分布12→审计71；
   target24原事件、13,440query及2/3名义权重保持，6,720个支持query按71任务等权校正，不改模型/监督/绝对LR。
   原已存C12_630为同父同龄对照，不重复其训练，不从历史点选峰值。
3. 两个固定630各做400＋144及固定A28/被动native，共1,088环境行；新540仅供恢复。
   primary为完整held得失与保持，seen/native仅解释。没有other/wrong/时序controls/Test/RL或自动后续训练。
4. 预算12GPUh/64GiB、预计2–4小时，唯一root`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`。
   科学合同已登记，具体派发后由原实验session实施和运行；metadata测算不冒称已启动。
   整批回报后主讨论主动解释该支持分布是否带来实际迁移；只改善拟合、seen或交换能力，不自动开完整fresh或扩大数据。

工程/运行/Git由实验session闭环；主讨论负责科学判断。整批科学结果或确需裁决的实质边界按Owner要求§6回报；
其它工程阶段事实保存在已有记录。实际代码、冻结、运行与窗口状态只看[progress](progress.md)。

## 仓库维护

本轮独立整理的实际改动、保留依赖、536项CPU检查、合并后38项消费者检查、已安装skill维护与回滚位置见
[研究历史](docs/research_history.md#2026-10-02整仓整理与历史状态入口)。整理承接最新实验修复，交付身份见progress与Git。
没有改动当前实验模型、数据/标签、FM、信息墙、选择合同或预算；科学下一步仍按上面的唯一批次推进。

## 历史

多轮旧执行计划的完整快照为`git show 797ae01f:task_plan.md`；设计、实际结果、负证据和失败原因沿
[research_history](docs/research_history.md)追溯。历史未完成清单不是现行待办；不会靠整理覆盖其当时事实。
