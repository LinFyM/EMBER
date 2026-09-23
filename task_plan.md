# EMBER task plan

## 当前：仓库收尾与第二轮checkpoint裁剪完成（2026-09-23）

Owner要求继续实质清理checkpoint，不能把每个唯一保存节点都当成必要资产。
已按研究价值裁剪45条结束路线的训练状态与中间权重，并退休Source未使用的EMA和optimizer；
额外回收360.759 GiB，登记weights-only和不可exact-resume边界，保留关键模型及当前依赖。
实验推进、新执行任务创建和后继实验派发仍停止。本次长期要求、文档、代码脚本、
可删除大资产和worktree收尾已完成；后续研究由Owner另行安排。详细快照以[progress](progress.md)为唯一入口。

## 已完成的收尾范围

- 稳定理解写入[Owner要求](docs/current_owner_requirements.md)和[concept](docs/concept.md)，区分事实、假设与授权。
- 文档按职责组织，当前计划/进度移除历史重复；独有科学证据在findings、research_history和review_materials保留。
- 已结束的专用运行面退役，保持一个canonical源码入口；按实际变更完成CPU检查。
- 大资产按生命周期核验，删除可再生缓存、无继续用途的训练状态和非关键中间权重；保留关键权重、当前依赖、数据与正式证据。
- 未完成实现封存到已推送分支，主线不合入未验证科研代码；整合清理提交后移除本任务临时worktree。
- 最终main干净且推送，进度记录空间回收、验证范围、保留边界及接手位置；直接对话汇报，不追加独立报告。

## 明确不执行

不恢复训练、GPU诊断、Validation/Test、视频controls、FT/RL，不创建Sol任务，不再给Luna派发工作。
[条件编译诊断方案](docs/designs/conditional_compilation_diagnostics_design.md)只是暂停的候选合同。
不存在active experiment design；任何旧“当前/下一步”、历史deadline或创建过的goal都不能恢复执行。
