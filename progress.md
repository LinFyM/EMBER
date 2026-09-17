# EMBER progress

## 当前状态（2026-09-18，统一Writer整套实验结束）

本轮仓库/data1整理、统一新架构实现、六卡等效执行、正式训练、既有v5.2比较和修正可行性审视已完成。
科研窗口按预注册的1200步后停止条款在1500结束，结果为**有界scientific non-pass**；不采纳本次改造。
当前没有active design、active run或selected checkpoint，也没有待自动执行的后续节点。
已生成[完整报告](runs/analysis/unified_writer_20260917/experiment_report.md)、
[结构化裁决](runs/analysis/unified_writer_20260917/experiment_completion.json)及[逐任务表](runs/analysis/unified_writer_20260917/unified/analysis/primary_tables.md)。
原设计保留为[封存设计](docs/v52_evidence_based_writer_design.md)；历史论证及每节点事实见findings§117与research_history。

Owner本轮授权整理、实现、高效训练、比较已有v5.2，以及有具体改进依据时修正重训。
Owner随后明确要求六卡等效提速、不得重训或续训v5.2，并取消300节点评测；均按最后要求执行。
此前误启动的重复v5.2已在73步停止，无100步checkpoint或闭环结果，不纳入科学比较；处置原件为study的
`baseline/owner_scope_correction.json`。旧A3000评测、C和其它历史实验均未恢复。

## 完整行为结果与裁决

| Step | 新模型correct /400 | Train /96 | 验证/训练breadth | 既有A correct /400 | A train /96 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 600 | 90 | 38 | 4/8；17/24 | 88 | 47 |
| 900 | 109 | 54 | 6/8；18/24 | 140 | 54 |
| 1200 | 67 | 54 | 5/8；20/24 | 135 | 62 |
| 1500 | 84 | 52 | 5/8；19/24 | 112 | 56 |

1500的validation S/O/G/L为3/18/33/30，train为19/13/15/5。
1200→1500 validation R/G/L43/41/24、churn65、Jaccard .398；train41/11/13、churn24、Jaccard .631。
验证净回升17集中于Long1（13→30），其它suite净和为零；900→1500仍为109→84，train54→52。
后段未形成持续整体获取，成功保持和覆盖缺口仍在。初始上限2400不等于必须耗尽；1800/2100/2400未执行。
这是本架构/配方/有界窗口的停止判断，不证明全部统一结构不可能成功，也不把某个内部模块命名为根因。

同raw1000 source、实际teacher/state/RNG配对的A900/1200比新模型多31/68，差值95%区间不含零；
1500新−A为−28，95%区间[-22.25,+4]pp含零。A为agentview/H均值，新模型为双RGB/fullH，不能孤立归因一个模块。
原始v5.2固定900的132、同映射复核125使用不同source/video映射，只作描述性整体参照。
没有节点达到correct严格>145/400，因此没有qualification或selected checkpoint。
后续same-task-other、learned language/static和冻结后最终controls未启动；动态视频必要性与时序特异性改善未获证明。
Test未使用、held梯度为零、没有RL或用最终controls返工。

修正审视没有找到可复现工程错误或可先明确失败接口/主变量/预测的具体改动。
既有73/75-task及meta73/target18历史不支持把95-task扩展当默认修复；没有新增梯度来源或做rank/scale/seed/LR/dtype扫描。
因此没有第二轮重训。报告保留早期有效学习和Long局部反弹，不将non-pass写成完全没有获取。

## 实现、资源与正式证据

Canonical Writer为两处同构Z/H联合块、一次native双写回、连续参数读出和完整38-target A/B，13,451,008个可训练参数。
三Meta与Writer fresh共同纯FM，source trainable=0；双RGB、fullH、K1、stride5及信息墙保留。
六卡p0/1/2/3/4/6，两组三卡执行真实帧/query分片，仍是4task/global84/一次optimizer更新。
原生不等长/空分片、汇总梯度及完整恢复验证通过；最长105帧选择20帧chunk，实际峰值约34.04GiB。
相同事件8帧chunk热身六卡8.61秒对四卡12.52秒，吞吐1.45倍；不把它解释成对v5.2的整体成本优势。

正式训练commit为clean pushed detached `184947cbd9bb1f05c4a4684f390633f10374e0b2`；
1500更新累计6000条件/126000queries，程序含诊断15073.5秒（4.19h），实际allocated最高34.06GiB。
100至1500每100的15个完整checkpoint均保留，六rank状态、optimizer/scheduler/sampler/RNG连续。
四节点各400+96条件均fresh生成，1984条闭环及120个worker完成记录全部通过；评测程序合计4328.5秒。
各节点均使用五张当时合适的GPU、每卡三个persistent workers；独立评测与六卡训练拓扑分开。

Study为`/data0/user/ymdai/ember_runs/unified_writer_20260917`，仓库入口`runs/analysis/unified_writer_20260917`为symlink。
run contract、每节点命令/双节点preflight/quota、checkpoint manifest、raw rows、aggregate、completion和分析均保留。
全部本轮训练/生成/评测进程及tmux已退出；干净、已集成的临时frozen runtime已删除，代码由上述Git commit保留。

## 清理与交付

源码整理`bf8aea37`退役旧ECP捕获、SFT在线验证、重复functional wrapper及结束A配置，相关110项检查通过。
统一架构及六卡实现已集成，模型/native/训练/恢复/物化/评测检查通过；近期结果文档维护没有重跑无关测试。
删除23,224个可重建LoRA payload，准确释放96.322GiB，并移除13个干净、已集成历史工作树；
另删除8组已完成profile的107个临时/重复文件，逻辑大小1.298GiB，关键profile证据保留在study。
保留数据/source、全部formal checkpoint/raw rows、外部硬链接payload、不完整bank及9个有未合入/未提交工作树。
原件为`runs/analysis/workspace_cleanup_20260917.json`、study的`profile_cleanup_20260918.json`与`experiment_completion.json`。
收尾strg01独立quota快照：data0用121009828KiB，data1用922241376KiB，各自soft1073741824KiB；两者不混算。

详细结果、比较边界、停止/未重训理由和未执行范围以完整报告为统一交付入口；源代码继续只有一个canonical Writer实现。
