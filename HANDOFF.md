# EMBER cross-session handoff

本文件是新session的临时入口，完整消费后删除。长期事实与要求已进入正式文档，本文件不独占研究结论、设计决定或计划。

## 接手状态

Canonical workspace：`/data1/user/ymdai/projects/EMBER`。先检查main、工作区diff及远端状态，保留后来出现的无关改动。
当前设计与授权以 [progress.md](progress.md) 为准；按 [README](README.md) 进入完整文档和代码地图。

9月6日晚owner已接受新架构，要求先完成记录、仓库清理，再由新session接手；新session必须先理解并报告计划，
得到owner明确同意后才能实施/运行新架构。旧goal、旧design、width256的未做闭环和历史待办均不授权自动续跑。

关键交接事实：新设计已完整记录但尚未实现；旧专用Writer执行面已退出活动树，通用读取/LoRA/数据/VJP/评测基础保留；
旧原件由Git `fcdb6e43706c5fcedf10eaa5d2d459602b263016`与formal artifacts保存。228.50GiB可重建缓存已退役，
原checkpoint、raw rows和生成provenance保留，精确名单见progress引用的清理记录。先用已整理历史，按具体问题展开原件。

## 可复制的启动prompt

```text
请接手 /data1/user/ymdai/projects/EMBER。第一阶段先充分理解仓库，暂不实现新架构，不启动训练、评测或恢复旧实验。

请完整阅读 AGENTS.md、docs/current_owner_requirements.md、README.md、task_plan.md、progress.md、findings.md、docs/concept.md 和 docs/causal_layered_video_writer_design.md；读完 docs/research_history.md 的分层历史，再按当前问题查其索引中的原始评审、后续修正、Git快照和formal evidence。检查相关现有代码、canonical数据/模型来源和Git状态，不重新扫描全部历史运行，不重复复制资产。

向我说明：原始科学目标；新架构完整数据流及每个模块的因果职责；从历史到当前设计的推导、关键正负证据与边界；当前可复用代码和待实现差距；具体实施、GPU算法、数据/采样、验证和闭环裁决计划。明确区分事实、候选假设和未验证能力。然后等待我明确同意开始。

得到同意后，按已记录计划自主推进新架构及最终性能目标，不逐项重复确认。主动派发真正能并行省时的subagents，隔离写入并由主agent集成验证。先实现唯一canonical路径，核实真实Meta梯度、因果时序、完整horizon、多视频集合与重放语义，按真实最长样本优化吞吐，尽快用有信息量的学习和闭环证据判断投入。

遇到问题先区分工程错误、科学non-pass与证据不足，定位最早失效接口；结合历史用能区分竞争解释的有限干预解决根本问题。有新证据时允许实质重构，不靠无依据小扫、无限续训、恢复旧fallback或层层补丁原地打转。不要用内部loss和几何指标代替闭环性能，也不要把局部失败扩展为整条科学路线失败。

最终以validation8 strict paired single-checkpoint correct >145/400及相邻/跨视频稳定、低churn、高breadth、四suite非零和Goal/Long贡献共同裁决。遵守信息墙、固定split、真正的K覆盖、fully-random fresh候选、冻结后视频因果controls和最终32/8 fresh/test合同。shuffled/reversed不得反哺训练、选点或架构。遵守实时GPU/独立quota与main交付合同，持续更新正式账本；不要因旧goal仍active就自行启动。理解完成后删除临时HANDOFF.md，并更新README中相应临时入口。
```
