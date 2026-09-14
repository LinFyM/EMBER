# EMBER progress

## 当前状态（2026-09-14）

本轮[Process Pullback Writer](docs/process_pullback_writer_design.md)纯FM研究已完成，完整报告与有限原因分析已落盘。
Owner允许多次有依据的训练、修复和迭代；是否停止坚持或回到v5.2完全由owner决定，agent不自行切换。
Owner追加的清理、真实机制、效率优化、低秩出口功能核验及预登记fresh学习窗口均已完成。
计划见[task_plan](task_plan.md)，稳定目标与最新优先级见[owner requirements](docs/current_owner_requirements.md)。

900更新及300／600／900的六个完整面板、1,488条闭环rows全部结束。
结论为有局部能力获取，尚未形成广泛、稳定的未见task迁移；未获前置资格，没有selected checkpoint或最终视频controls。
本轮按已放宽绝对分数后的能力／保持要求裁决；视频必要性及顺序特异性仍未知，不用CPU机制或结构性质替代。

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
  78项bank合同与5项实际spawn并发／失败检查通过；首个496条件真实四卡物化完成，576秒、约.861 LoRA/s。
  三个节点固定state/video映射；300的所有候选执行合同、normalization、逐行RNG和视频无重复检查通过。
- 训练池为train24的demo16–41，共624个K1条件；采样帧数min16、median31、p90为57、max105。
  最长视频已用于上述真实profile；独立动作42–45／teacher46–49及train states32–35保持既有合同。

本轮训练来自clean pushed detached `d6defa69`，冻结运行树`.codex/tmp/process-pullback-runtime`，gpu01:0,1,2,3四rank。
启动与分段记录为本轮`training_launch_contract.json`，输出`training/`。
三段正常退出，累计900更新、3,600条件／230,400queries，实际覆盖623/624个task/video条件；独立meta tasks仍为24。
训练墙钟合计16,433秒；独立动作诊断FM为0／300／600／900的.153279524／.149746817／.150356967／.148562361。
source可训练参数0，完整checkpoint和曝光检查通过；动作loss只是诊断，闭环证据如下。
物化／评测冻结树`.codex/tmp/process-pullback-evaluation`来自已推送`f5d9db78`，使用四卡按条件动态编译；训练树保持原样。
三次各496条件物化正常完成，合计1,663秒；六个闭环面板均12 workers exit0、合计3,928秒，完整配对检查通过。

| 节点 | Train /96 | Train breadth /24 | Validation /400 | Validation breadth /8 |
| --- | ---: | ---: | ---: | ---: |
| Source | 17 | 7 | 47 | 3 |
| 300 | 24 | 8 | 64 | 3 |
| 600 | 22 | 8 | 72 | 4 |
| 900 | 26 | 9 | 64 | 4 |

900相对source：train R/G/L14/12/3、差额task-bootstrap CI[+2.08,+17.71]pp；validation R/G/L39/25/8、CI[0,+10.75]pp。
Validation S/O/G/L从0/26/36/2→0/29/40/3→2/18/44/0，Object局部增益回落，Long归零；三个节点没有同时覆盖四suite。
Validation两次相邻R/G/L56/16/8、52/12/20，churn24／32，Jaccard .7000／.6190；
train两次为17/5/7、21/5/1，churn12／6，Jaccard .5862／.7778。不能把训练任务恢复等同于未见task保持。
固定动作诊断19/24task改善、均值约3.08%；identity首步后Writer及两组Meta的899次记录梯度均finite非零，source冻结。
这些保留局部可学性，不能唯一定位q、PCA、读取器、source或优化根因，也不能证明普遍收敛。
完整逐task、suite和配对证据见本轮[READOUT](runs/analysis/process_pullback_writer_20260914/READOUT.md)及`paired_readout.json`。
本轮已结束，无在途训练／评测，不原样追加900之后的训练或小扫。
same-task-other及wrong／no-video／shuffled／reversed未运行；没有q辅助、RL、Test、checkpoint融合或自动回退v5.2。
当前实现、完整checkpoint与正式证据保留；两棵冻结运行树供本轮复核及后续比较使用。
Owner此前决定先完成纯FM再决定q辅助；当前没有新对照的执行决定，后续范围由owner结合完整报告确定。

## 近期关键出处

- [Active design](docs/process_pullback_writer_design.md)：完整方法、信息墙、学习和裁决合同。
- [研究历史](docs/research_history.md)：最近关闭的固定真实纠正、语义状态路径、局部纠正场研究及其formal原件索引。
- [工作理论](docs/temporal_control_compilation_theory.md)与[findings](findings.md)：跨轮正负证据和待检验的获取／保持解释。

清理前的文档全文保存在Git `f9f142e5`；封存设计及formal原件保留原实验身份和裁决。
本文件只保存当前授权与真实快照，旧“当前／下一步／暂停”段落不再作为并列活动状态。
