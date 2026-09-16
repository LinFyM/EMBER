# EMBER task plan

## 当前目标与授权

Owner于2026-09-15要求设置goal自主推进，2026-09-16收敛为先建立正确对齐的source、共享SFT和v5.2 A基线。
若三者与旧版本没有实质变化，跳过新B，直接分析已有B相对A的差距；必要时才完成已登记B窗口。
完成基线比较与差距分析后停下来讨论，不自动重启历史方法、补2×2或视频controls。Goal保持active，
最新合同见[active design](docs/source_alignment_v52_plan.md)，旧goal的广泛后续范围由本次owner要求覆盖。
原C保持关闭，不恢复611后的训练；既有source、A/B/C和Process Pullback证据保留。

## 执行顺序

1. **已完成：实现与profile。** 恢复唯一source训练入口，复用现有setup／checkpoint／dataset；修正offset1，
   保持1000更新／global256／全参数训练及原optimizer。适配A40显存，不冻结参数或缩短曝光。
   A单视角／mean和B双视角／learned已整合；真实四卡global256及完整checkpoint恢复通过。
2. **已完成：source训练与官方读出。** 已据profile锁定四rank、micro8／accum8及滚动完整checkpoint；
   clean pushed detached `b8ea00e9`的fresh1000更新已完成；最终保存超时后依据完整落盘状态恢复发布元数据，保留原失败记录。
   固定raw step1000为validation50/400、train13/96，旧source47／17；两组配对区间跨零，不按结果选source或扩步。
3. **进行中：新source上的A与共享SFT。** A／必要时B共享正确数据／初始化／事件／LR及评测映射，分别fresh1200；
   A／B的新source最长视频profile已通过并封存；300／600／900／1200完整correct400＋train96，分开报告获取、保持及相对各自source的收益。
   A300已从`575c189a`完整完成：validation99/400、train36/96，四suite非零，尚无>145或相邻稳定资格；
   600完整结果为88/400、47/96；相邻验证保留57／新增31／丢失42，900已精确续训。共享SFT已完成历史rank128／global576／450更新与正确source／offset的修正；两卡真实更新和完整恢复通过，formal配置已封存。
   SFT fresh formal已启动且首步通过，固定400／425／450完整validation400；不复制旧在线validation动作监控。B尚未启动，待三个基线完成后决定。
4. **基线裁决与必要的B。** 比较对应曝光节点、任务／suite分布、覆盖与保持；三基线变化有限则跳过新B。
   变化重要则完成既定B窗口，保持预算，不追加扫参或历史架构。宽区间不当作等效证明。
5. **分析并讨论。** 用已有和必要的新证据分析B相对A的差距，区分可确认事实与无法唯一归因的候选原因；
   完成后停下来交owner讨论，不自动开展2×2、controls、Test或其它新方法。

## 资源、边界与完成判断

S+A+B原追加峰值预算112GiB，含原revision generic权重约13.5GiB；SFT单独据profile追加小型checkpoint／rows预算，
launch前重核data1独立quota、双节点GPU及实测峰值，不改变运行中的Writer冻结树。
Source有效global256及原科学规格固定，物理microbatch／累积／EMA存放可等效调整；formal jobs来自clean pushed detached树。
不改变source71审计排除、train24/validation8/test8、normalization、部署信息墙，不引入RL、第二adapter或无依据小扫。
长期方法目标仍为>145/400及相邻稳定、覆盖、同task视频鲁棒性和视频必要性；本轮完成标准是最新owner指定的基线重建、
必要的B、差距分析及讨论交付，不能以该方法目标为由无限扩展本轮。
最新命令、资源与实测结果进入[progress](progress.md)，历史证据沿research_history追溯。
