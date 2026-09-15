# EMBER task plan

## 当前目标

Owner于2026-09-15在v5.2完整历史分析后授权“计划下接下来怎么做，然后就启动吧”。
Active design为[v5.2恢复与保持对照](docs/v52_return_plan.md)。Process Pullback已完成并关闭，原证据保留。

## 执行顺序

1. **已完成：固定旧step900复核。** 原state–video映射correct125/400，原132；breadth均6，R/G/L111/14/21、churn35。
   全12 workers exit0；旧NUMA汇总判断经实际拓扑修正，原400行成功汇总。未重跑、不选点、不新增controls／Test。
2. **进行中：现行合同fresh基线。** 保留v5.2视频Core、Procedure与完整A/B能力路径，明确迁移双相机、完整50-horizon learned read、
   严格跨episode；先封存接口／采样／优化与真实profile，再按300／600／900／1200节点执行。
3. **有条件：一个任务共现对照。** 基线具备可辨认获取后，仅改变同一train24事件的四task共现，保持曝光、query数、更新和学习率一致。
   没有获取／保持共同改善则停止该假设，不作小扫。
4. 能力和相邻稳定成立后，再冻结单checkpoint、验证same-task换视频及最终视频因果controls；Test仅在登记冻结后开放。

## 预算与停止条件

A追加峰值4GiB，模型和数据不复制；单节点至多6张live合适A40，先测真实长视频，再定生成并发。
B／C每臂预设最多1,200更新、4,800视频条件、100,800 action queries；精确接口和资源预算在训练前封存。
A分叉先核对合同，不重跑挑132；B没有获取则不盲做C；C没有联合改善则结束该候选。
不改科学目标、信息墙或长期资格线，不引入RL、辅助loss、第二adapter、held梯度或未授权meta tasks。

当前实际状态与命令见[progress](progress.md)，完整边界与阶段准入见[active design](docs/v52_return_plan.md)。
