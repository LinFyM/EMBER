# EMBER progress

## 当前状态（2026-09-14）

Goal active，按Owner五个问题讨论确定的[Process Pullback Writer](docs/process_pullback_writer_design.md)持续推进。
Owner允许多次有依据的训练、修复和迭代；是否停止坚持或回到v5.2完全由owner决定，agent不自行切换。
当前先完成Owner追加的占用审计、退役源码／入口与文档内容清理，随后进入真实profile和新低秩出口功能核验。
计划见[task_plan](task_plan.md)，稳定目标与最新优先级见[owner requirements](docs/current_owner_requirements.md)。

本方案尚未完成GPU profile或G P投影出口的真实功能前提，未启动正式新训练或评测，没有selected checkpoint或新方法闭环结果。
CPU机制检查不能代替上述结果；长期>145/400本阶段不强制，未取消能力、相邻保持、换视频及最终视频controls的要求。

## 已实现与验证

- 实现起点为`ec08b9f2`，active design登记于`c7f7e96f`；新方法代码已集成到main `f9f142e5`。
- Canonical图为同步双路K1读取、完整50-horizon、变化驱动过程Value、7维q、裸冻结source导数与PCA rank16唯一38-target A/B。
  Writer及Action Meta／VL Meta fresh共同学习，唯一loss为同task跨episode main FM；source基础冻结，无局部场辅助路径、RL或Test。
- 集成后155项针对性CPU检查通过，覆盖过程网络、固定source编译／精确q伴随、pure FM重放、采样与恢复、checkpoint身份和模拟物化。
  这些是工程合同证据，不是实际GPU吞吐、能力获取或科学资格。
- 清理已删除确认临时的profile大载荷及完成集成的工作树，退役旧RL／采样／panel实现、专属测试和旧Writer配置；
  入口已改名为`train_writer.py`和`materialize_writer.py`。磁盘审计与清理后的相关检查仍在进行，总释放量尚待完成后登记。
- 新配置为`configs/pi05_process_pullback_writer.json`，当前学习节点未登记，profile状态pending；smoke需显式停止点，formal需完整登记。
- 训练池为train24的demo16–41，共624个K1条件；采样帧数min16、median31、p90为57、max105。
  该统计用于选择最长视频profile，不是已经完成的GPU成本测量。独立动作42–45／teacher46–49及train states32–35保持既有合同。

当前先完成磁盘与仓库清理；新输出根依据清理后`/data1`的实时独立user quota确定，不把转移到`/data0`当作清理。
尚未登记的窗口、profile值和功能结果不由历史实验填补；当前无额外数据、RL或Test授权，后续顺序只在task_plan维护。

## 近期关键出处

- [Active design](docs/process_pullback_writer_design.md)：完整方法、信息墙、学习和裁决合同。
- [研究历史](docs/research_history.md)：最近关闭的固定真实纠正、语义状态路径、局部纠正场研究及其formal原件索引。
- [工作理论](docs/temporal_control_compilation_theory.md)与[findings](findings.md)：跨轮正负证据和待检验的获取／保持解释。

清理前的文档全文保存在Git `f9f142e5`；封存设计及formal原件保留原实验身份和裁决。
本文件只保存当前授权与真实快照，旧“当前／下一步／暂停”段落不再作为并列活动状态。
