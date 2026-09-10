# EMBER task plan

## 当前目标与授权（2026-09-11）

Owner明确要求休息期间由Codex自行整理专家意见与独立判断、规划、设置goal并连续完成方法修正、实验验证、问题分析及再次修正，醒来后审查。已建立active goal。此前仅咨询、暂停科研和禁止再次完整训练的阶段限制已被本次授权覆盖；不再逐项请求批准。信息墙、固定split、source冻结、资源/Git及最终资格保持，Test封存。

目标为validation8 strict paired single-checkpoint correct>145/400，并满足相邻稳定、低churn、高breadth、四suite/Goal/Long、换正确视频和最终独立视频因果资格。不能以一次高分、机制接通或实验结束标记goal完成。

## 当前计划

唯一active design为[语义条件化过程消费](docs/video_consumption_writer_design.md)。前代[Horizon设计](docs/horizon_relation_video_writer_design.md)提供保留的完整H/过去关系细节与历史合同，不恢复其中旧待办。

1. **完成：** 归档第二轮专家原文与独立判断，落实R/C/S对照、数据/随机性/评测和资源合同。
2. **完成：** 实现一个canonical runtime下的条件分配与消费接口；共享C/S语义读取、融合和D，明确无序帧集合参照的能力与边界。
3. **完成：** CPU合同与整合、真实最长视频及恢复检查通过；formal使用clean pushed detached 7fedbe85。
4. **进行中：** R/C首轮200与完整checkpoint/独立动作验证已完成，S fresh训练及R/C 100/200物化并行；接续全部strict correct400和200训练闭环96。固定task/state/video/RNG口径，报告全部曝光和逐task/suite得失。
5. 依据真实能力和竞争解释选择继续、修正或终止候选。有实质获取则登记后续相邻节点；明确负结果不无限续训、不小扫、不同时叠加多个未经支持的改动。
6. 候选稳定后补换正确视频与公平参照，冻结选择后做独立最终因果裁决；全过程维护可审查记录、checkpoint、原始结果和Git。目标完成前持续自主推进。

## 当前已知事实与候选边界

- 既有最强Horizon off7 macro400为126/400，off11为119，尚未达标；最强训练任务九臂也没有稳定动态增量。
- v5.2普通正样本监督已有正确视频依赖；监督允许捷径不足以解释新旧差异。专家已撤回先保持再视频的顺序，见[第二轮原文](docs/review_materials/20260911/expert_review_round2.md)。
- 主假设是最后消费接口难以利用过程；竞争解释是每update教学条件分配。新增语义路径独立解释收益、P4过程不足与无序多帧仍能含过程信息，是本轮裁决必须保留的限制。
- 初轮不加辅助loss/保持约束，不换D/rank，不扩Meta/数据、不切dual。后续只有证据改变判断时才选择一个主要修正。

## 审查入口

即时进度：[progress](progress.md)；跨轮结论：[findings](findings.md)；历史：[research_history](docs/research_history.md)。专家原文、第一轮追问与最新证据保留在[9月11日材料](docs/review_materials/20260911/README.md)。旧整理及咨询工作已完成，其历史在ae606387及前序Git中保留。
