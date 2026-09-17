# EMBER progress

## 当前工作（2026-09-17，统一Writer整套实验执行goal）

Owner已授权仓库/data1整理、统一新架构实现和吞吐优化、与匹配v5.2的完整训练评测、
以及结果不佳但有明确改进空间时的修正重训，全部结束后详细汇报。已设置新的执行goal；
上一轮仅分析/不实验限制已被本次授权替代。无关旧A/C等路线不恢复。

当前阶段：**统一架构实现与实际profile**。源码整理已在clean pushed `bf8aea37`交付；尚未启动正式新训练。
已退役旧ECP捕获合同、SFT在线验证模块、重复functional wrapper及结束A配置；相关110项检查和训练入口检查通过。
已删除23,224个可重建物化payload，准确释放96.322GiB；另移除13个干净、已合入main且无运行依赖的临时工作树。
保留400个有外部硬链接的payload、一个缺少完整评测原件的bank、9个有未合入/未提交历史工作的worktree，
以及全部正式checkpoint、bank清单、raw rows、数据/source和专家材料。逐项记录：`runs/analysis/workspace_cleanup_20260917.json`。
清理后strg01 quota：data1用922,230,408KiB/soft1,073,741,824KiB，余约144.49GiB；
data0用108,483,544KiB，独立余约920.54GiB（launch前复核，非固定配额事实）。

主代理负责canonical main中的runtime/schema、训练/物化合同、评测tests、profile与集成；两个独立工作树分别实现
`video_program`原生分段与联合表示、`temporal/model`联合块及连续参数decoder。均从bf8aea37开始，写入范围不重叠。
单一active运行面整体替换，旧v5.2仅在clean detached sparse tree `.codex/tmp/unified-v52-baseline`保留本轮必要对照。
模型实现采用`code-architecture-gate`；保留现有真实FM/VJP、task等权采样、checkpoint和动态评测调度。

实现已合入main：`11e96245`负责temporal/model，`14432969`负责native encoder。实构造13,451,008参数；
encoder418行、temporal236行、model300行，旧Core/P/AdaLN已移除。集成后的native/temporal/LoRA检查35项通过；
runtime/训练/物化/评测相关196项通过。结构检查无hard；现有物化合同检查及两个模型constructor的长度/复杂度review
保留为集中接口校验，不为数字拆分其职责。正式新模型性能尚未测量。

v5.2非正式profile完成：105-frame最长train视频、21-query完整FM，8/8物理batch稳定约25.3秒/条件、峰值28.99GiB；
12/12约24.88秒但43.94GiB，frame16 OOM，故选择8/8。三卡gpu01 p0/1/4 profile完成1–3并完整恢复至6，
各step计数/optimizer/LR正常，三Meta有梯度；峰值reserved33.01GiB。profile不用于正式初始化。
脚本/原件在`.codex/tmp/unified_writer_profile.py`及`.codex/tmp/unified-writer-profile/`；其动态设备不代表未来预约。

新study输出复用`/data0/user/ymdai/ember_runs/unified_writer_20260917`，仓库入口
`runs/analysis/unified_writer_20260917`为symlink。`launch_contract.json`登记source、事件、2400预算/配对节点、
最多200GiB新增峰值估计（含主比较、必要诊断和一轮条件修正，无大资产复制）及baseline冻结bf8aea37。
Baseline为匹配双RGB/learned完整H50 fresh模型，先按旧runtime登记1200 events，再按预授权原事件prefix扩至2400；
科学曝光和12k优化时钟与新架构相同。当前准备首个600节点，正式状态以run_contract、日志及本文件后续登记为准。

## Active design与执行准备

- 唯一active design：[统一Writer设计](docs/v52_evidence_based_writer_design.md)，j9/j18各两层联合Z/H块、一次双写回、
  两层连续参数decoder、三Meta共同学习及八组共享完整A/B输出。源码已实现，真实新模型profile待完成。
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
