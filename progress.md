# EMBER progress

## 当前状态（2026-09-14）

Goal active，按Owner五个问题讨论确定的[Process Pullback Writer](docs/process_pullback_writer_design.md)持续推进。
Owner允许多次有依据的训练、修复和迭代；是否停止坚持或回到v5.2完全由owner决定，agent不自行切换。
Owner追加的磁盘、退役源码／入口及文档正文清理已完成；最长视频真实机制基线通过，当前完成等价效率优化及新低秩出口功能核验。
计划见[task_plan](task_plan.md)，稳定目标与最新优先级见[owner requirements](docs/current_owner_requirements.md)。

本方案已完成GPU机制／成本基线，优化后profile及G P投影出口功能前提尚待结果；未启动正式新训练或评测，没有selected checkpoint或新方法闭环结果。
CPU机制检查不能代替上述结果；长期>145/400本阶段不强制，未取消能力、相邻保持、换视频及最终视频controls的要求。

## 已实现与验证

- 实现起点为`ec08b9f2`，active design登记于`c7f7e96f`；新方法代码已集成到main `f9f142e5`。
- Canonical图为同步双路K1读取、完整50-horizon、变化驱动过程Value、7维q、裸冻结source导数与PCA rank16唯一38-target A/B。
  Writer及Action Meta／VL Meta fresh共同学习，唯一loss为同task跨episode main FM；source基础冻结，无局部场辅助路径、RL或Test。
- 集成后155项针对性CPU检查通过，覆盖过程网络、固定source编译／精确q伴随、pure FM重放、采样与恢复、checkpoint身份和模拟物化。
  这些是工程合同证据，不是实际GPU吞吐、能力获取或科学资格。
- 清理实际释放212.48GiB：32根已关闭实验的可重建物化缓存178.02GiB、临时profile载荷30.29GiB、32棵clean已集成工作树4.17GiB。
  保留正式checkpoint、数据、模型、评测rows／aggregate／contract／manifest及完整来源；3棵脏树、6棵未集成树保留。
  源码清理`aeb78f94`退役10个孤立模块、4份旧配置与4个专属测试；入口为`train_writer.py`／`materialize_writer.py`。
  五份当前文档经`3abbdb7b`从1,250行减至295行；main已集成。170项相关检查、训练入口help及diff检查通过。
- `strg01`清理后data1用量846,270,296KiB，soft quota 1,073,741,824KiB，余量约217GiB。
  删除范围和重建依据见[清理记录](runs/analysis/workspace_cleanup_20260914.json)；新输出留在data1，按各阶段峰值另登记。
- 最长task38/demo36共517原帧／105个stride5帧，真实全38-target／50×7编译与两次pure-FM更新通过；
  B8基线46.61／46.86秒、峰值34.18／34.23GiB。第二次Writer／Action Meta／VL Meta及各过程模块均有finite非零梯度，
  source无可训练参数或累积梯度。部署一次编译13.68秒，无loss／optimizer；基线初始化不复用，未保存正式checkpoint。
  B16旧full-prefix FM因显存不足退出，记录保留。结果在`runs/analysis/process_pullback_writer_20260914/profile/batch8/results.json`。
- 等价优化复用裸source的CPU前缀KV、直接以固定编译器伴随重放q信用，并让main FM使用官方prefix-KV／suffix路径；
  CPU全38-target投影、q伴随及联合梯度对照通过，优化后的真实GPU成本与FM对照待测。
- 新配置为`configs/pi05_process_pullback_writer.json`；96条件低秩出口功能前提已在分数前登记，当前学习节点仍未登记。
  smoke需显式停止点，formal需完整窗口登记。
- 训练池为train24的demo16–41，共624个K1条件；采样帧数min16、median31、p90为57、max105。
  最长视频已用于上述真实profile；独立动作42–45／teacher46–49及train states32–35保持既有合同。

尚未登记的窗口、profile值和功能结果不由历史实验填补；当前无额外数据、RL或Test授权，后续顺序只在task_plan维护。

## 近期关键出处

- [Active design](docs/process_pullback_writer_design.md)：完整方法、信息墙、学习和裁决合同。
- [研究历史](docs/research_history.md)：最近关闭的固定真实纠正、语义状态路径、局部纠正场研究及其formal原件索引。
- [工作理论](docs/temporal_control_compilation_theory.md)与[findings](findings.md)：跨轮正负证据和待检验的获取／保持解释。

清理前的文档全文保存在Git `f9f142e5`；封存设计及formal原件保留原实验身份和裁决。
本文件只保存当前授权与真实快照，旧“当前／下一步／暂停”段落不再作为并列活动状态。
