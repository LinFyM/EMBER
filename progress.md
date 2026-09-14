# EMBER progress

## 当前状态（2026-09-14）

Goal active，按Owner五个问题讨论确定的[Process Pullback Writer](docs/process_pullback_writer_design.md)持续推进。
Owner允许多次有依据的训练、修复和迭代；是否停止坚持或回到v5.2完全由owner决定，agent不自行切换。
Owner追加的磁盘、退役源码／入口及文档正文清理已完成；真实机制、效率优化与新低秩出口功能核验通过，当前执行已登记窗口的正式fresh学习。
计划见[task_plan](task_plan.md)，稳定目标与最新优先级见[owner requirements](docs/current_owner_requirements.md)。

本方案已完成GPU机制／成本核验及G P投影出口功能前提；900更新窗口已在正式分数前登记，并已启动四卡formal首段0→300；尚无selected checkpoint或新方法闭环结果。
CPU机制检查不能代替上述结果；长期>145/400本阶段不强制，未取消能力、相邻保持、换视频及最终视频controls的要求。

## 已实现与验证

- 实现起点为`ec08b9f2`，active design登记于`c7f7e96f`；新方法代码已集成到main `f9f142e5`。
- Canonical图为同步双路K1读取、完整50-horizon、变化驱动过程Value、7维q、裸冻结source导数与PCA rank16唯一38-target A/B。
  Writer及Action Meta／VL Meta fresh共同学习，唯一loss为同task跨episode main FM；source基础冻结，无局部场辅助路径、RL或Test。
- 最新集成175项针对性CPU检查通过，覆盖过程网络、固定source编译／精确q伴随、pure FM重放、采样与恢复、checkpoint身份和模拟物化。
  这些是工程合同证据，不是实际GPU吞吐、能力获取或科学资格。
- 清理实际释放212.48GiB：32根已关闭实验的可重建物化缓存178.02GiB、临时profile载荷30.29GiB、32棵clean已集成工作树4.17GiB。
  保留正式checkpoint、数据、模型、评测rows／aggregate／contract／manifest及完整来源；3棵脏树、6棵未集成树保留。
  源码清理`aeb78f94`退役10个孤立模块、4份旧配置与4个专属测试；入口为`train_writer.py`／`materialize_writer.py`。
  五份当前文档经`3abbdb7b`从1,250行减至295行；main已集成。170项相关检查、训练入口help及diff检查通过。
- `strg01`清理后data1用量846,270,296KiB，soft quota 1,073,741,824KiB，余量约217GiB。
  删除范围和重建依据见[清理记录](runs/analysis/workspace_cleanup_20260914.json)；新输出留在data1，按各阶段峰值另登记。
- 最长task38/demo36共517原帧／105个stride5帧，真实全38-target／50×7编译与两次pure-FM更新通过。
  选用frame_chunk16／FM16，两次24.25／23.99秒、峰值37.01GiB、部署7.78秒；原基线46.86秒／13.68秒。
  Writer／Action Meta／VL Meta及各过程模块第二次均有finite非零梯度，source冻结。所有profile初始化均不复用、无正式checkpoint。
  各批量、原OOM及数值probe记录保留于`runs/analysis/process_pullback_writer_20260914/profile/`。
- 等价优化复用裸source的CPU前缀KV、直接以固定编译器伴随重放q信用，并让main FM使用官方prefix-KV／suffix路径；
  实际全38-target nonidentity FM梯度cosine .999899、norm比1.00363、loss差0.155%，保留正常BF16及高效kernel。
- 96条件功能前提通过：t1／full10改善CI下界.001426／.001918，4／3个正suite，保留原G平均改善72.56%／76.29%；
  全部finite，两个worker exit0。原件在本轮`functional/summary.json`，不计作合法Writer或闭环结果。
- `configs/pi05_process_pullback_writer.json`已登记fresh900、3,600条件／230,400queries；300／600／900做correct400及train96，
  100倍数保存。四卡训练预计4–5小时、诊断和评测另计；初段12GiB、含条件触发controls的预计峰值24GiB。
- 评测物化已支持同节点常驻GPU worker按完整条件动态分工，保持同一编译计算和单一完整manifest；
  78项bank合同与5项实际spawn并发／失败检查通过，真实四卡物化将在首个checkpoint完成后执行。
  三个节点的400＋96请求、固定state/video映射、source来源与完整配对readout已准备；候选实际配对检查待闭环完成。
- 训练池为train24的demo16–41，共624个K1条件；采样帧数min16、median31、p90为57、max105。
  最长视频已用于上述真实profile；独立动作42–45／teacher46–49及train states32–35保持既有合同。

当前formal来自clean pushed detached `d6defa69`，冻结运行树`.codex/tmp/process-pullback-runtime`，gpu01:0,1,2,3四rank。
启动记录为本轮`training_launch_contract.json`，输出`training/`，tmux `ember_process_pullback_train_20260914`。
初始24task动作诊断完成，FM .153279524；前三个实际更新19.75／22.26／19.94秒，第二次起两组Meta均有有效梯度。
NCCL同步约.016秒，峰值37.54GiB，source可训练参数0。首段结束后及时完成correct400／train96，再exact-resume同一run至600／900。
当前无额外数据、RL或Test授权，后续顺序只在task_plan维护。
Owner最新确定先保持当前四卡、完成纯FM闭环结果后再决定是否做q辅助对照；暂不开展六卡训练拆分或新增q损失。
等待期间只完成必要准备，其余等待进程结束事件，不反复读取或播报训练进度。

## 近期关键出处

- [Active design](docs/process_pullback_writer_design.md)：完整方法、信息墙、学习和裁决合同。
- [研究历史](docs/research_history.md)：最近关闭的固定真实纠正、语义状态路径、局部纠正场研究及其formal原件索引。
- [工作理论](docs/temporal_control_compilation_theory.md)与[findings](findings.md)：跨轮正负证据和待检验的获取／保持解释。

清理前的文档全文保存在Git `f9f142e5`；封存设计及formal原件保留原实验身份和裁决。
本文件只保存当前授权与真实快照，旧“当前／下一步／暂停”段落不再作为并列活动状态。
