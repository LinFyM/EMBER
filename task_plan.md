# EMBER task plan

## 当前目标

Owner最新于2026-09-15要求“别继续C了，你停一下”。C已停止，不自动恢复训练或评测。
[v5.2恢复与保持对照](docs/v52_return_plan.md)的A、B已完成，C保留300／600完整结果及最后已写611更新；最新完整checkpoint为600。
该计划剩余C节点暂停，不作完整C窗口结论。Process Pullback已完成并关闭，原证据保留。

## 执行顺序

1. **已完成：固定旧step900复核。** 原state–video映射correct125/400，原132；breadth均6，R/G/L111/14/21、churn35。
   全12 workers exit0；旧NUMA汇总判断经实际拓扑修正，原400行成功汇总。未重跑、不选点、不新增controls／Test。
2. **已完成：现行合同fresh基线。** 保留v5.2视频Core、Procedure与完整A/B能力路径，明确迁移双相机、完整50-horizon learned read、
   严格跨episode；300／600／900／1200为validation97／105／77／85（均/400）、train35／49／50／56（均/96）。
   最后相邻验证保留56、新增29、丢失21；train保留38、新增18、丢失12。训练获取与breadth扩大，验证交换与迁移缺口未解决，未通过正式资格。
3. **Owner已停止：一个任务共现对照。** B300已建立可辨认train获取，仅改变同一train24事件的四task共现，保持曝光、query数、更新和学习率一致。
   C300／600为validation91／65（均/400）、train28／43（均/96）；600相对B为验证−40、训练−6，验证差额95%CI[-18.5,-2.75]pp。
   C验证相邻保留48、新增17、丢失43，未支持保持改善；其后900段已按owner指令终止，不再启动剩余900／1200训练或评测。
   B、C的GPU工作均已结束，原checkpoint、完整面板和部分训练日志保留；未完成窗口不冒充完整对照结论。
4. **未开放。** same-task-other、最终视频controls与Test均未获得本轮能力及相邻资格，不启动。

## 预算与停止条件

A追加峰值4GiB，模型和数据不复制；单节点至多6张live合适A40，先测真实长视频，再定生成并发。
B／C每臂预设最多1,200更新、4,800视频条件、100,800 action queries；精确接口和资源预算在训练前封存。
A分叉先核对合同，不重跑挑132；B没有获取则不盲做C；C没有联合改善则结束该候选。
不改科学目标、信息墙或长期资格线，不引入RL、辅助loss、第二adapter、held梯度或未授权meta tasks。

当前实际状态与命令见[progress](progress.md)，完整边界与阶段准入见[active design](docs/v52_return_plan.md)。
