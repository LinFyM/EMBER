# EMBER task plan

## 当前goal（2026-09-18，专家建议的A learned frame-set参照）

Owner已授权执行[专家意见](docs/review_materials/20260918/expert_review.md)的唯一训练诊断并推送相关结果；此前暂停被本范围取代。
Active design：[learned frame-set匹配合同](docs/learned_frameset_reference_design.md)。

1. **进行中：登记与实现。** 核对A原件，恢复其已验证模块及初始化顺序，仅移除视频位置/因果mask/读取时间寻址。
   保留全部真实帧、语言/空间/horizon位置、三Meta、完整A/B、采样事件与优化时钟；替换已结束统一图，不并留fallback。
2. **待完成：验证与profile。** 集合不变性、真实信用、事件匹配、恢复和物化合同；真实最长视频/更新profile，选合法高效资源。
3. **待完成：正式fresh1200。** 每100保存完整checkpoint，300/600/900/1200各完成correct400及train96，原拓扑分段exact-resume。
4. **待完成：配对解释。** 复用既有A四节点，重点900+1200，报告task/suite/breadth/R/G/L/churn/相邻保持与区间，不挑峰值宣称等价。
5. **待完成：远程交付。** 推送代码、专家原文、预注册、结果汇总和精简可复核raw rows，更新状态与研究历史，确认main同步后结束goal。

不重训原A或原v5.2，不并开新架构、不用wrong/shuffle/reverse选点、不碰Test/held梯度/RL。
结果可以是正、负或不确定；目标是完成专家要求的匹配诊断，不是强行取得某种结论。
当前实际资源与进度见[progress](progress.md)，历史实验不自动恢复。
