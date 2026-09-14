# EMBER task plan

## 当前goal与边界

完成原因诊断后，自主决定当前方法修订、实施、验证、正式训练和完整配对评测，再在对话中交付实际结果与原因分析。
Owner无需再次审查，持续goal仍active。当前唯一设计为[Process Pullback共享可学习出口](docs/process_pullback_writer_design.md)。
方法只改变原生LoRA出口：共享identity起步的L/R乘法变换，其余过程读取、7维q、rank16、数据和纯FM保持。
旧冻结900的视频controls不反馈新方法；不做q辅助、RL、自动回退v5.2或超参扫描。
固定Test仅在本轮方法及terminal900冻结后用于登记的source/correct读出，无梯度、不选点、不反哺设计。

## 已完成

1. 固定出口fresh900窗口、六个正式面板及完整曝光／相邻保持分析结束。
2. 旧900全部视频对照、三节点训练池视频、96条行为回放、全train24 q／完整A/B有界功能及闭环对照结束。
3. 原出口／free_q／free_AB的20／18／46支持优先修正固定出口整体约束；不单独给PCA定责，也不把特权拟合当RGB能力。
4. 相关历史与原始合同已核对；现有负例边界保留。已明确新修正的因果作用、局限、fresh曝光和冻结读出合同。

## 当前执行

1. 共享出口、精确信用、v2 metadata及冻结后Test准入已集成main并push，诊断专属路径已退役。
2. 直接autograd、bank及Test准入检查与真实最长视频两次联合更新／部署profile均通过，隔离实现工作树已清理。
3. Fresh900来自clean pushed detached `85ecfd18`、gpu01:0–3；300节点及两套闭环面板已完成，train21/96、validation50/400，已同topology exact-resume至600。
4. 在300／600／900分别完成strict paired validation400、train96及固定独立动作诊断，按原topology exact-resume。
5. 冻结预先固定的terminal900和本轮方法，完成same-task-other、wrong/no-video及最后shuffled/reversed的paired400，
   再完成固定test8 source400／correct400。终点读出不冒充性能合格选点，controls和Test均不反馈本轮设计。
6. 汇总per-task／suite、breadth、R/G/L、churn、相邻重合、曝光、成本、配对不确定性和未解决范围，直接在对话中报告。

## 完成判断

本goal要求授权工作完整结束，不要求制造正结果。科学成功仍要真实能力、跨task迁移、保持、换视频及必要视频增量共同支持。
有界特权正例、代码、非零梯度、loss或孤立分数峰值都不是成功。合理窗口后仍弱则报告具体修正未通过，不无限续训／扫参。
实际状态和资源记录见[progress](progress.md)，正式命令与来源只保存在当前研究launch contract，历史见[research_history](docs/research_history.md)。
