# EMBER progress

## 当前快照（2026-09-11，自主执行已恢复）

Owner最新明确授权Codex在其休息期间持续自主推进方法修正、实验验证、分析和再次修正，并要求创建goal；active goal已建立。此前咨询暂停与完整训练限制已被覆盖，后续不重复请求逐项批准。当前正在落实设计与实现，尚未启动新formal训练。

唯一active design：[普通FM下语义条件化过程消费](docs/video_consumption_writer_design.md)。首轮R/C/S比较保留普通FM、完整H、过去定向和独立D，分别检验条件分配、消费接口及主动训练无序帧集合参照；具体随机性、曝光、节点和裁决见设计。源码尚未实现部分不冒充已运行模型。

[专家第二轮原文](docs/review_materials/20260911/expert_review_round2.md)已归档。独立判断：末端改动依赖P4已有可用信息；新增语义S可能独立贡献；无序多帧仍有状态变化信息。C/S必须共享语义、融合和D，不能把不同模型分数差唯一归因于时序。目标仍未达标；Test封存。

当前执行：主agent负责架构、配置、整合与科研裁决；条件调度/梯度权重可在独立worktree完成。初轮资源预计峰值96GiB，strg01/data1最新used743343260KiB/soft1073741824KiB；大资产全部复用，GPU分配在launch前按两节点实时状态决定。代码、profile与formal身份分开。

## 方法身份与学习结果

下表为本轮新实验前的历史参照：main canonical曾启用Compiler额外语言query；126候选使用独立冻结commit45e16633的`backend_conditioning=local_h_read`关闭该route，保留local/H-read/exact language与独立D。两者不能混同。main已实现同步双视角输入，实际有分数的checkpoint全部为agentview单视角；双视角只做过输入/梯度smoke。

| 固定实验 | validation200→400 /400 | train200→400 /96 | validation自身保留/新增/丢失 |
| --- | --- | --- | --- |
| all init7 | 103→90 | 41→49 | 56/34/47 |
| Compiler-off init7 | 108→126 | 40→56 | 82/44/26 |
| all init11 | 100→108 | 41→56 | 62/46/38 |
| Compiler-off init11 | 104→119 | 44→61 | 67/52/37 |
| 同target跨rank D绑定 init7 | 115→82 | 32→60 | 59/23/56 |

Compiler删除的验证净收益经两个初始化与init7另一正确视频关联复核，但稳定保持没有修复；两个候选仍增长，不能宣布平台。D绑定只淘汰实际检验的组合。原始矩阵、逐task/suite和曝光见最新专家包；历史强模型、source47及SFT109/107的适用边界见research_history和9月7日证据包。

## 最新视频检查与机制边界

最强126候选固定macro400、train24、states32–35、teacher46/other47、单视角九臂864条，18个worker全部complete/exit0，实际输入/执行RNG配对通过。正确/另一正确/同suite错/跨suite错/乱序/倒序/首帧/中帧/末帧为**53/60/59/59/55/52/58/53/53**（各96）；对应all400为51/53/48/46/47/50/52/51/48。当前最强候选在该固定面板亦未建立稳定动态增量。输入改变具体成功集合，不能据此证明完全忽略视频或纯task记忆；见findings§49。

此前原因深化共2701条配对闭环：1728 all视频控制、640双层Q中介、256训练任务P/C/D端点、36有限更新、32 sealed BBQ端点、9橙汁回放。四次其它task真实更新使橙汁转选BBQ，零当前梯度Adam对照仍成功；P变化可独立复现目标转移。BBQ四例中只换D400使3例转向绿色瓶，只换P400为2例，只换C400为0例，另有模块交互。该现象涉及共享条件映射，不能只归Compiler，亦不证明冻结D是修复。

查询差异压缩及其向读取的传递已测到，但扩大双层差异未带来一致即时闭环收益；fresh删除收益的唯一学习中介未识别。先前真实视频/reader信用/执行诊断还降低了强梯度抵消和直接删policy-Q即可修复的解释。完整证据等级见findings§39–49；不要将几何、非零梯度或训练收益当作时序理解/迁移成立。

所有诊断无Test或held梯度。临时真实更新只有4个gradient macro/1024跨episode queries，另4次零当前梯度Adam对照；没有新增完整训练。Owner曾明确允许wrong/shuffle/reverse用于原因诊断，这些已看过结果不能冒充未触碰的最终因果资格。

## 证据与仓库整理

- 最新本地原件：`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/`。最新远程副本及源路径映射见专家包`index.json`；大checkpoint、数据和视频张量保留本地。
- canonical源码、数据/训练/eval入口经只读审计，未发现生命周期明确的新退役簇；不根据目录名或静态无引用推测删除。未采纳候选分支和用户dirty worktree保留。
- 删除137个可再生缓存/重复自检文件，共2,684,218 bytes；profile/smoke、唯一checkpoint及历史证据保留。
- README与计划/进度入口已更新；旧逐次运行状态从当前文件中收敛，历史仍由Git、research_history、findings和原件保存。
- 导出保真、119面板/13,901条统计、13组关键配对、全部新材料引用与隐私字段检查通过。没有新增训练/评测或无关测试；main交付以本文件所属提交为准。

整理前完整进度：[043b58ca的progress](https://github.com/LinFyM/EMBER/blob/043b58ca3f1f3e7ed699b876f4964be96654c7e5/progress.md)。后续待办与权限只看当前task_plan，不恢复历史命令。
