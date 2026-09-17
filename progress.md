# EMBER progress

## 当前工作（2026-09-17，统一Writer整套实验执行goal）

Owner已授权仓库/data1整理、统一新架构实现和吞吐优化、与匹配v5.2的完整训练评测、
以及结果不佳但有明确改进空间时的修正重训，全部结束后详细汇报。已设置新的执行goal；
上一轮仅分析/不实验限制已被本次授权替代。无关旧A/C等路线不恢复。

当前阶段：**源码整理完成，缓存清理与实现准备进行中**。Canonical main从clean pushed `176759a7`开始；尚无新训练或新GPU作业。
主代理负责空间、runtime、决策与集成；已完成只读源码生命周期审计。退役无生产调用的旧ECP捕获合同、SFT在线验证模块、
重复functional wrapper和已结束A配置；关键Meta梯度不变量迁入当前encoder测试。相关检查共110项通过；
其中3项旧SFT测试的“未封存profile”前提已显式化。历史正式源码/配置可按原commit恢复。
空间审计已登记76个可重建bank，保留外部硬链接后约96GiB payload正在删除；13个已集成临时工作树也在清理。
完成量以`runs/analysis/workspace_cleanup_20260917.json`及清理后quota为准；所有正式checkpoint、清单和评测原件保留。
实现将应用`code-architecture-gate`，复用现有encoder/temporal/model与runtime；状态继续归本文件/task_plan。

## Active design与执行准备

- 唯一active design：[统一Writer设计](docs/v52_evidence_based_writer_design.md)，j9/j18各两层联合Z/H块、一次双写回、
  两层连续参数decoder、三Meta共同学习及八组共享完整A/B输出。结构论证已完成，源码实现与runtime profile待完成。
- 匹配v5.2 baseline的源码可从pushed `176759a7`冻结，运行配置须匹配raw1000、双RGB/fullH、train24与优化事件；
  不把旧A的分数直接赋给新对照。不在canonical源码中保留两套生产Writer。
- 训练、显存、GPU分配、独立quota及新增资产峰值在真正launch前按实际profile封存；当前没有正式新run。
- Frozen source、action-hidden视频、跨episode纯FM、paired400、controls与Test边界仍按AGENTS/active design执行。

## 已完成设计与历史状态

第二轮设计在`176759a7`交付；完整证据审计、来源与限定见findings§115–116、
[证据审计](docs/v52_evidence_audit_20260917.md)及[研究历史](docs/research_history.md)。
设计解析13,451,008参数尚非runtime测量；静态重复不变性、地址Value边界、原生接口和信用开启已经审查，
实际获取、迁移、视频增量、保持与资源峰值均待本轮实验。

旧A3000训练已完成，2800/2900/3000完整恢复点保留；3000未评测，最新完整节点2700为108/400、train64/96。
SFT400/425/450 validation为85/89/86，其训练评测已结束。原件在`runs/analysis/source_alignment_20260915/`，
长期曲线和解释见findings§107–114与research_history。先前暂停是当时事实，不覆盖本次新goal，也不自动恢复旧待办。
此前长篇逐节点执行记录由Git `176759a7:progress.md`保留；当前状态文件只维护本轮需要的状态与入口。
