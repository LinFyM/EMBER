# EMBER progress

## 当前状态（2026-09-18，Owner暂停研究，整理仓库供专家讨论）

Owner最新要求是停止当前分析，整理本地仓库并推送远程，由其自行与专家讨论。
架构分析及两项只读并行核对已停止；没有active design、训练、评测或待自动执行的实验。
后续研究等待Owner新指示；旧goal、设计、配置和历史中的“下一步”不恢复执行授权。

当前接受约130–140的能力及更可信的视频特异性，允许少量正常churn。
结构必须合理，能够复制同类模块加深并自然扩展参数；设计须综合全部相关历史证据及其预算、配方和适用条件。
“保留Core/P、只改Procedure、停止整体重构”是代理过早的路线建议，现已撤回，不是Owner要求。
没有选定下一架构；原v5.2不重训的要求继续有效。稳定要求见[Owner要求](docs/current_owner_requirements.md)。

## 已完成实验与证据入口

| 实验 | 已完成事实 | 解释边界与入口 |
| --- | --- | --- |
| 统一Writer | 600/900/1200/1500 correct为90/109/67/84，各400；train为38/54/54/52，各96。1500按原合同结束 | 有界non-pass，不采纳本次改造；不证明所有统一结构不可行。findings§117、[封存设计](docs/v52_evidence_based_writer_design.md) |
| A900机制诊断 | train24固定96条件，正确58、关Procedure34、固定视频保留目标语言38/38、固定LoRA13/10、Source12、SFT47 | 旧过程路径有行为贡献，不证明顺序理解或fresh删除效果。findings§118 |
| Core/Procedure交叉 | CC58、CW34/35、WC56/56、WW38/38，各96；新增384闭环，复用288对角线 | 正确P增量能跨两个donor Core发挥；不证明Core可删、未见任务迁移或下一架构应只改P。findings§119 |

完整正负历史先读[46组证据审计及补表](docs/v52_evidence_audit_20260917.md)，再读[findings](findings.md)§117–119。
源码版本、旧专家评审、各轮逐task/suite、R/G/L/churn及formal原件由[研究历史](docs/research_history.md)索引。
统一实验及两次诊断的详细报告分别位于本地：

- `runs/analysis/unified_writer_20260917/experiment_report.md`
- `runs/analysis/v52_mechanism_audit_20260918/report.md`
- `runs/analysis/v52_core_procedure_cross_20260918/report.md`

这些`runs/`原件是ignored本地资产；远程仓库保留源码、合同、历史审计和findings结论，不包含checkpoint或数据集。
当前源码是已封存统一Writer的唯一实现；统一正式训练版本为`184947cb`，旧A冻结诊断版本为`575c189a`。
保留复现入口不表示已经选择或恢复该方法。旧A3000评测、C及其它关闭窗口均未恢复；Test保持关闭。

## 本次仓库整理

已检查源码、12个脚本入口、34个测试文件、27份配置、文档和ignored临时目录；未发现需要本轮改动的第二套Writer实现。
已合并重复的当前计划/进度叙述，补齐可扩展结构要求，纠正最近诊断的推断范围，并更新README讨论入口。
删除12个Python/pytest缓存目录的156个可重建文件，文件占用3,678,208字节；不改科研代码或配置。
保留九个含未合入commit或未提交改动的旧工作树，以及全部正式证据、唯一checkpoint、数据和模型。
未核定为可删除的历史临时记录保留；不以本次整理改变历史实验结果或扩大资产删除范围。
本轮检查范围为文档diff、受影响链接、Git状态与远程同步；不重复运行训练、GPU检查或无关测试。
