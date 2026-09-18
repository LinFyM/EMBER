# EMBER progress

## 当前状态（2026-09-18，A的learned frame-set参照goal已启动）

Owner提供专家意见，明确要求设置goal，完成专家要求的工作并把相关结果推送远程。
本范围替代此前暂停：只fresh训练与既有强A匹配的无序集合参照，不重训A、不并开新架构。
Active design为[匹配合同](docs/learned_frameset_reference_design.md)，专家原文保存在[评审材料](docs/review_materials/20260918/expert_review.md)。

已核对A原run contract、575c189a源码及四节点原始结果：correct300/600/900/1200为99/88/140/135，train36/47/54/62。
预注册新参照同1200更新、4800条件/100800queries，节点300/600/900/1200，主解释900+1200相邻证据。
已完成实现、CPU合同检查、105帧最长视频profile与三卡6更新/第4步完整恢复。
三路Meta真实信用通过，4800事件与A全字段一致；约12–14秒/更新、峰值20.65GiB。
profile权重丢弃；正式fresh已从clean pushed detached `a9d8614964abfcddef40eb82f2862623f587ffa6` 启动，当前首段0→300。
资源为gpu02三卡，固定全窗口1200；driver顺序执行四节点及各correct400/train96，尚无新闭环分数。
本地study：`runs/analysis/a_learned_frameset_20260918/`；launch、profile、严格匹配事件和逐段log均在其中。
新study计划使用data0，新增峰值预算20GiB；data0/data1独立user quota及共享容量已作规划检查，launch前刷新。
唯一干预移除视频RoPE、因果mask及读取时间寻址；为此匹配A的agentview/fixed-mean H历史口径，不改变未来新架构的长期接口原则。

完成标准是执行整个预注册诊断、保存证据与分析、推送main；不要求特定分数，不从最终controls选点。
所有稳定目标和信息墙继续有效；不恢复A3000评测、C或其它已关闭实验。详细阶段见[task_plan](task_plan.md)。

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
本轮源码正由匹配A的frame-set实现替换；已封存统一正式训练版本为`184947cb`，旧A冻结诊断版本为`575c189a`。
保留复现入口不表示已经选择或恢复该方法。旧A3000评测、C及其它关闭窗口均未恢复；Test保持关闭。

## 本次仓库整理

已检查源码、12个脚本入口、34个测试文件、27份配置、文档和ignored临时目录；未发现需要本轮改动的第二套Writer实现。
已合并重复的当前计划/进度叙述，补齐可扩展结构要求，纠正最近诊断的推断范围，并更新README讨论入口。
删除12个Python/pytest缓存目录的156个可重建文件，文件占用3,678,208字节；不改科研代码或配置。
保留九个含未合入commit或未提交改动的旧工作树，以及全部正式证据、唯一checkpoint、数据和模型。
未核定为可删除的历史临时记录保留；不以本次整理改变历史实验结果或扩大资产删除范围。
本轮检查范围为文档diff、受影响链接、Git状态与远程同步；不重复运行训练、GPU检查或无关测试。
