# EMBER cross-session handoff

本文件是新session的临时入口，完整消费后删除。长期事实与要求已进入正式文档，本文件不独占研究结论、设计决定或计划。

## 接手状态

Canonical workspace：`/data1/user/ymdai/projects/EMBER`。先检查main、工作区diff及远端状态，保留后来出现的无关改动。
当前设计与授权以 [progress.md](progress.md) 为准；按 [README](README.md) 进入完整文档和代码地图。

9月7日owner已完成双向局部关系过程的数学重推，要求同步仓库并重新准备交接。当前候选已定义完整，尚未实现/验证；
新session先理解并报告计划，得到owner明确同意后才能实施/运行。旧goal、旧design、width256未做的闭环和历史待办不授权自动续跑。

设计入口为 [layered_relation_video_writer_design.md](docs/layered_relation_video_writer_design.md)。重点读懂：
独立帧对C、两端条件归一化、对应内容与rho、关系MLP、逐帧邻居聚合、同型block堆叠和有限双侧上下文。
过去单向初稿仅是Git历史；GPU布局、成本与验证条目已经随新图更新，不可照搬旧的联合历史attention。

旧专用Writer执行面已退出活动树，通用读取/LoRA/数据/VJP/评测基础保留；代码、资产与证据入口见README及设计§11。
228.50GiB可重建缓存在前次整理时退役，原checkpoint、raw rows和生成provenance保留；先从正式历史索引找原件与重建依赖。
前次139项CPU测试验证的是保留的工程基础，新图没有GPU或闭环证据。

## 可复制的启动prompt

```text
请接手 /data1/user/ymdai/projects/EMBER。第一阶段先充分理解仓库，暂不实现新架构，不启动训练、评测或恢复旧实验。

完整阅读 AGENTS.md、docs/current_owner_requirements.md、README.md、task_plan.md、progress.md、findings.md、docs/concept.md 和 docs/layered_relation_video_writer_design.md；读完 docs/research_history.md 的分层历史，再按实际问题展开原始评审、后续修正、Git快照和formal evidence。检查相关现有代码、canonical资产来源和Git状态，不重复复制资产或重扫全部历史运行。

当前候选已在9月7日完成数学重推：单probe Action Expert共享观察Meta，保留18层×50H；窗口内每个帧对一个50×50关系，两端分别归一化；关系MLP同时消费对应内容、相对位移模式和signed gap；每帧聚合最多8个邻居，同型block堆叠，再做H-read、集合Q与完整坐标A/B生成。旧past-only、跨全部历史H的联合softmax及“任何未来帧都不能影响早期E”均已退出当前合同。

先向我报告：科学目标与完整数据流、各模块职责及数学边界；历史正负证据与当前假设；可复用代码和待实现差距；具体实施、GPU算法、数据/采样、验证和闭环裁决计划。区分事实、归纳偏置、实现默认与未验证能力，然后等待我明确同意开始。

得到同意后，按记录的目标连续自主推进，实现唯一canonical路径，完成真实Meta梯度、对应/方向/rho、同步更新、有限双侧上下文、多K集合和VJP重放的验证。按最长真实K1/K4及action queries优化吞吐，尽快取得有信息量的学习与闭环证据。任务可分、确实并行且计入协调集成后能显著省时时，主动使用最少必要subagents，隔离写入，由主agent完成集成和验证；常规步骤不重复向我确认。

遇到问题先区分工程错误、科学non-pass与证据不足，结合历史定位最早失效接口，用能区分竞争解释的有限干预推进。有新证据时允许实质重构；不要靠无依据小扫、无限续训、旧fallback或层层补丁原地打转。内部loss、attention形状和参数几何不能代替闭环性能，局部失败也不能被扩大为整条科学路线失败。

最终以validation8 strict paired single-checkpoint correct严格 >145/400，以及相邻/跨视频稳定、低churn、高breadth、四suite非零、Goal/Long贡献共同裁决。遵守完整train24、经审计额外meta、固定split/信息墙、真实K覆盖、fully-random fresh候选、冻结后视频因果controls及最终32/8 fresh/test合同。shuffled/reversed不得反哺训练、选点或架构。遵守实时GPU、独立quota与main交付要求，持续更新正式账本；旧goal不构成启动授权。理解完成后删除临时HANDOFF.md，并更新README和progress中的临时入口。
```
