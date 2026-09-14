# EMBER task plan

## 当前goal与边界

按Owner五个问题讨论确定的[Process Pullback Writer](docs/process_pullback_writer_design.md)推进完整研究：
以正确action-hidden视频一次编译的LoRA获得可迁移的执行能力，检验视频特异性与训练保持，并完整汇报证据和未识别范围。
Owner允许多次有依据的训练、工程修复与方法迭代；不以只训练一次为限，也不允许无信息重复。是否停止坚持或回到v5.2由owner决定。
当前授权与实际完成状态见[progress](progress.md)，稳定要求见[owner requirements](docs/current_owner_requirements.md)。

本轮固定双路K1、冻结source、完整50-horizon、7维q、固定source导数／PCA rank16唯一38-target LoRA及fresh联合纯跨episode FM。
不使用q辅助标签、额外meta tasks、RL或Test。暂不强制>145/400，不要求另训frame_set；能力、相邻保持、换视频与最终controls仍须有真实证据。

## 当前阶段与必要下一步

1. **实际清理已完成。** 已删除退役源码／配置、过时正文、可重建物化缓存、临时profile载荷及32棵完成集成的工作树。
   数据集、源模型、完整正式checkpoint和评测原件保留，实际释放212.48GiB；范围与余量见progress。
   新输出留在`/data1`，按每阶段实时独立user quota和峰值预算执行，后续及时清理完成使命的临时产物。
2. **真实机制与成本核验已完成。** 最长视频完整编译、q伴随、identity、第二次联合梯度及真实main FM对照通过；
   96条件G P功能前提通过。优化后每最长条件约24秒，frame_chunk16／FM16，峰值37.01GiB；oracle不计作合法Writer能力。
3. **学习窗口已登记。** Fresh900更新、3,600教学条件／230,400queries，100倍数保存；300／600／900做完整correct400及train96。
   预计四卡训练约4–5小时，诊断和评测另计。曝光规模和判断理由见active design§8，不能以loss代替闭环或无信息续训。
4. **执行fresh学习与配对闭环。** Formal train/eval来自clean pushed detached frozen commit；按预登记节点完成correct strict400，
   必要时用固定train96及独立动作面板区分获取与迁移，报告task／suite、breadth、R/G/L、churn、相邻重合与真实曝光。
   有能力获取和保持证据再继续；负结果先修正实际受损假设，区分工程错误、科学non-pass与证据不足。
5. **完成资格与最终视频证据。** 能力和相邻资格成立后补same-task-other，选定冻结单checkpoint，再执行完整配对wrong／no-video／shuffled／reversed。
   最终controls不反哺训练、选点或架构；未获前置资格则明确报告未执行范围，不制造selected checkpoint。

## 成功与结束判断

科学成功需要正确视频的真实闭环价值、跨task／suite获取、相邻保持、same-task换视频鲁棒性及冻结模型后的因果证据共同支持。
代码、非零梯度、loss、局部privileged正例或孤立分数峰值均不替代这些证据。

本goal的完成指授权研究工作与完整汇报完成，不要求制造正结果。合理窗口后持续缺少正向信号时，完成有限原因分析并明确报告，
不自动回退或无限追加训练。历史结果、旧计划和暂停快照由[研究历史](docs/research_history.md)、封存原件与Git保留，不在本计划续写。
