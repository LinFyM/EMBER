# EMBER task plan

## 当前目标与授权

Owner于2026-09-15要求设置goal自主推进，2026-09-16收敛为先建立正确对齐的source、共享SFT和v5.2 A基线。
若三者与旧版本没有实质变化，跳过新B，直接分析已有B相对A的差距；必要时才完成已登记B窗口。
Owner随后要求A900完整验证后立即暂停讨论；Goal已暂停。仅允许在运行的SFT继续到400步自动停止，
不启动A1200、新B、SFT后续评测或425／450续训，也不自动重启历史方法、补2×2或视频controls。
最新合同见[active design](docs/source_alignment_v52_plan.md)，旧goal的广泛后续范围由本次owner要求覆盖。
原C保持关闭，不恢复611后的训练；既有source、A/B/C和Process Pullback证据保留。

## 执行顺序

1. **已完成：实现与profile。** 恢复唯一source训练入口，复用现有setup／checkpoint／dataset；修正offset1，
   保持1000更新／global256／全参数训练及原optimizer。适配A40显存，不冻结参数或缩短曝光。
   A单视角／mean和B双视角／learned已整合；真实四卡global256及完整checkpoint恢复通过。
2. **已完成：source训练与官方读出。** 已据profile锁定四rank、micro8／accum8及滚动完整checkpoint；
   clean pushed detached `b8ea00e9`的fresh1000更新已完成；最终保存超时后依据完整落盘状态恢复发布元数据，保留原失败记录。
   固定raw step1000为validation50/400、train13/96，旧source47／17；两组配对区间跨零，不按结果选source或扩步。
3. **当前停止边界：A900读出、SFT后台至400。** A／必要时B原登记共同fresh1200预算和300／600／900／1200完整correct400＋train96；
   现依owner最新要求仅完成A900读出并讨论，A1200及新B不启动。分开报告获取、保持及相对各自source的收益。
   A300已从`575c189a`完整完成：validation99/400、train36/96，四suite非零，尚无>145或相邻稳定资格；
   600完整结果为88/400、47/96；900完整结果为140/400、54/96，验证breadth6、S/O/G/L17/60/30/33。
   600→900验证保留68／新增72／丢失20、churn92；获取回升，仍未通过>145或相邻稳定资格，当前已停在900。
   共享SFT训练入口已按历史rank128／global576／450步预算修正source／offset；两卡真实更新和完整恢复通过，formal配置已封存。
   SFT fresh formal已启动，仅继续到400并自动停下；原登记的400／425／450评测及后续训练暂不执行，不复制旧在线validation动作监控。
4. **暂停后的候选裁决，等待讨论。** 原计划比较对应曝光节点、任务／suite分布、覆盖与保持；三基线变化有限则跳过新B。
   变化重要则完成既定B窗口，保持预算，不追加扫参或历史架构。宽区间不当作等效证明。
5. **当前交付：A900结果并讨论。** 汇总已完成节点及已有B证据，区分可确认事实与无法唯一归因的候选原因；
   到此停下，不以未完成原预算为由续跑，也不自动开展2×2、controls、Test或其它新方法。

## 资源、边界与完成判断

S+A+B原追加峰值预算112GiB，含原revision generic权重约13.5GiB；SFT单独据profile追加小型checkpoint／rows预算，
launch前重核data1独立quota、双节点GPU及实测峰值，不改变运行中的Writer冻结树。
Source有效global256及原科学规格固定，物理microbatch／累积／EMA存放可等效调整；formal jobs来自clean pushed detached树。
不改变source71审计排除、train24/validation8/test8、normalization、部署信息墙，不引入RL、第二adapter或无依据小扫。
长期方法目标仍为>145/400及相邻稳定、覆盖、同task视频鲁棒性和视频必要性；当前只完成A900完整读出后暂停讨论，
并保留owner明确授权的SFT后台至400。原基线重建及必要B目标未全部完成，不以这些目标为由越过暂停边界。
最新命令、资源与实测结果进入[progress](progress.md)，历史证据沿research_history追溯。
