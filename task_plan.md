# EMBER task plan

## 当前目标（2026-09-20，Owner已授权启动）

固定EMBER-1V单相机1500，完成论文实验与图文原始报告。当前active合同：[论文实验](docs/paper_experiments_design.md)。

1. **已完成：冻结资产与Test接口。** Source-71、MT-BC425、EMBER1500；development24模型Test用途已验证并推送45a39ba1。
2. **进行中：Test能力。** 三个完整400面板收齐后判断EMBER−MT-BC≥40；不足即停止下游询问Owner。
3. **条件执行：Test视频证据。** other/wrong/shuffle/reverse各400；缺失或不明确则停止询问。
4. **条件执行：单support FT。** validation/test16任务，独立演示选点集FM选点，eval50不参与选择；劣势触发三support复核。
5. **条件执行：两臂task-local RL。** 先Train接口/学习资格，按登记区段结束后读结果；不达预期暂停。
6. **条件执行：外部比较与报告。** 优先WIZARD，再ViVLA资格核查与透明复现；图、表、成本、成功失败案例及原件可追溯。

## 执行边界

不重训旧Writer、不追扫、不合并32任务；不反复轮询日志。资源按live双节点GPU与独立quota确定，无日历硬截止。
Owner阈值用于推进决策，不能以改实验直到获胜作为完成条件。常规工程问题在范围内修复，重大歧义交Owner。
维护canonical main，正式运行使用clean pushed detached worktree；运行结束清理task-owned临时树/载荷，保留证据。

## 已完成历史入口

[20260920全面清理](docs/research_history.md#workspace-cleanup-20260920)；
[上一轮总报告](docs/review_materials/20260919/final_report.md)。历史路线和清理记录不恢复执行。
