# EMBER task plan

## 科学目标与当前批次

从冻结π0.5 source出发，利用exact language及action-hidden教学视频生成初次即有效的一套完整task LoRA，
由严格配对闭环证明绝对能力、相邻保持和有益教学增量。稳定边界见
[Owner要求](docs/current_owner_requirements.md)、[AGENTS](AGENTS.md)与[concept](docs/concept.md)。

当前唯一active design为[条件读写§14](docs/designs/conditional_read_write_architecture.md#14-固定图的第二轮教学覆盖与相邻保持2026-10-02)：

1. 已完成fresh450、50,400完整FM query、36task各50不同teacher与五个完整ECP；按实际卡数自动分配并在完整边界恢复，
   保持112有效query、完整FM及原optimizer/sampler/RNG语义，全部失败、占用和恢复已计费。
2. 唯一450节点validation137/400、完整train91/144、固定A28十二预测与八条被动记录全部验收；
   completion/readback及544原行/goal/连续trace齐备。整批19.273293GPUh、观察高水位37.058697GiB、保守峰值64GiB。
   本批新增计算已停止，工程状态和原件入口见progress。
3. 主讨论已完成原件、完整行为得失、算子及监督历史分析，见机制§94/findings§269；
   固定图从完整450继续到900，完整seen144与固定A28解释学习阶段。仅900超过MT153时补810相邻paired400，
   不倒选邻点或其它峰值；独立新增20GPUh/64GiB，先验3–5小时。
4. 原实验session按该具体合同实施；主讨论在整批回报后解释是否形成完整优势/相邻保持。
   未满足时停止该同池续训路径，无自动1350/2340、小扫、controls/Test/RL；不把方法停止等同停止研究。

工程/运行/Git由实验session闭环；主讨论负责科学判断。整批科学结果或确需裁决的实质边界按Owner要求§6回报；
其它工程阶段事实保存在已有记录。实际代码、冻结、运行与窗口状态只看[progress](progress.md)。

## 仓库维护

本轮独立整理的实际改动、保留依赖、536项CPU检查、合并后38项消费者检查、已安装skill维护与回滚位置见
[研究历史](docs/research_history.md#2026-10-02整仓整理与历史状态入口)。整理承接最新实验修复，交付身份见progress与Git。
没有改动当前实验模型、数据/标签、FM、信息墙、选择合同或预算；科学下一步仍按上面的唯一批次推进。

## 历史

多轮旧执行计划的完整快照为`git show 797ae01f:task_plan.md`；设计、实际结果、负证据和失败原因沿
[research_history](docs/research_history.md)追溯。历史未完成清单不是现行待办；不会靠整理覆盖其当时事实。
