# EMBER task plan

## 本轮完成范围

Owner授权的原因诊断、自主决定修订、实施验证、正式训练、冻结测试和对话内结果分析均已完成。
本轮[Process Pullback共享可学习出口](docs/process_pullback_writer_design.md)已冻结并关闭执行窗口；
当前没有active design或待执行实验。工作完成不代表方法获得科学资格：没有selected合格checkpoint。

## 已完成

1. 固定出口fresh900、全部节点面板、训练池视频、96条行为回放及全train24 q／完整A/B有界对照。
2. 原输出／free_q／free_AB的20／18／46支持优先修正出口参数化／优化约束；没有将PCA单独定责，也未把特权拟合作为初始化。
3. 实施共享identity乘法L/R出口，保留原过程读取、7维q、rank16、完整50-horizon与纯FM；完成相关检查、真实profile、main集成和推送。
4. Clean pushed detached 85ecfd18完成fresh900、3,600条件／230,400queries和六套节点面板：train21／20／18（/96），validation50／75／79（/400）。
5. 在新controls／Test之前登记能力non-pass并冻结方法和固定terminal900；全部视频对照79／81／53／70／78／47完成。
6. 全部视频controls结束后完成固定Test source86／correct49（/400），逐task／suite、breadth、R/G/L、churn、不确定性及配对审计齐全。
7. 新方法共3,888条正式rollouts、126个workers均正常结束；完整证据与限定结论进入项目状态、findings和research_history，结果直接在对话交付。

## 关闭结论与边界

共享出口取得局部validation增益，但训练闭环接近source，validation成功仍集中于两项奶油奶酪任务，Spatial／Long均零。
正确方向没有明确优势；Test出现局部新成功，但已有能力损失更多，86→49、R/G/L34/15/52，差额95%CI[-20.25,+.75]pp。
这些证据不支持广泛获取、迁移和保持已获解决；保留正负事实，不将有界窗口等同普遍不可学习性证明。

旧controls、所有新controls和Test均未反哺本轮架构／训练／选点；没有q辅助、RL、checkpoint融合、超参扫描或自动回退v5.2。
不从已关闭设计、日志或历史未完成项恢复运行。没有登记新的方法或训练计划。
实际状态见[progress](progress.md)，跨轮结论见[findings§102](findings.md#102-共享lr出口扩大了局部验证收益尚未建立广泛能力2026-09-15)，
原始命令、完成状态与审计以新研究final_launch_contract、paired_readout及final_paired_readout为准。
