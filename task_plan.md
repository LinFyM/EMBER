# EMBER task plan

## 当前目标与授权

Owner于2026-09-15要求设置goal自主推进：同规格重训正确时间对齐的source，随后fresh复现v5.2 A结构并训练B，
依据source对裸policy与Writer整条链的影响决定后续。Goal已active，合同见[active design](docs/source_alignment_v52_plan.md)。
原C保持关闭，不恢复611后的训练；既有source、A/B/C和Process Pullback证据保留。

## 执行顺序

1. **已完成：实现与profile。** 恢复唯一source训练入口，复用现有setup／checkpoint／dataset；修正offset1，
   保持1000更新／global256／全参数训练及原optimizer。适配A40显存，不冻结参数或缩短曝光。
   A单视角／mean和B双视角／learned已整合；真实四卡global256及完整checkpoint恢复通过。
2. **进行中：source训练与官方读出。** 已据profile锁定四rank、micro8／accum8及滚动完整checkpoint；
   正在封存命令／quota并从clean pushed detached树启动fresh1000；
   固定raw step1000做validation400＋train96，不按结果选source或扩步。
3. **待启动：新source上的A，再B。** 共享正确数据／初始化／事件／LR及评测映射，分别fresh1200；
   300／600／900／1200完整correct400＋train96，分开报告获取、保持及相对各自source的收益。
4. **结果驱动后续。** 有重复实质改善可有限复查有局部正证据的source敏感旧架构；影响有限或A/B差异主导，
   则聚焦v5.2，必要时补2×2另两角。区间宽时保留不确定，不机械二分。
5. **冻结视频证据与交付。** 按预登记规则确认换视频鲁棒性及最终controls，Test关闭；未具资格的固定终点
   读出仅作sealed post-hoc解释，不反哺训练／选点／后继架构修正。

## 资源、边界与完成判断

S+A+B追加峰值预算112GiB，含本地缺失的原revision generic权重约13.5GiB；launch前重核data1独立quota、双节点GPU及实测峰值。
Source有效global256及原科学规格固定，物理microbatch／累积／EMA存放可等效调整；formal jobs来自clean pushed detached树。
不改变source71审计排除、train24/validation8/test8、normalization、部署信息墙，不引入RL、第二adapter或无依据小扫。
长期目标仍为>145/400及相邻稳定、覆盖、同task视频鲁棒性和视频必要性；局部涨分不能将goal标为完成。
最新命令、资源与实测结果进入[progress](progress.md)，历史证据沿research_history追溯。
