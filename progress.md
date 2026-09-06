# EMBER progress

更新时间：2026-09-07 CST，交接整理收口。

## 当前授权与方法状态

- Owner于9月6日晚完成新架构讨论，要求本session记录完整设计、全面整理仓库并准备新session接手。
  本次授权包括相关代码整理、只读核验、必要验证、Git集成与推送；不包括新架构实现或新的科学训练/评测。
- 已登记active design：[causal_layered_video_writer_design.md](docs/causal_layered_video_writer_design.md)，状态为**设计已对齐，尚未实现/验证，等待新session理解后取得owner启动指令**。
- 当前候选：frozen图文prefix、单固定probe、Action Expert共享观察Meta、18层×完整50-horizon因果局部过程、置换不变多视频compiler、
  原生坐标条件MLP生成唯一38-target完整rank16 LoRA。初版默认与待落实细节在设计§9/14。
- 未选出selected checkpoint，未达成最终科学目标。旧工具goal或旧未完成实验不覆盖当前授权，也不自动恢复width256评测或旧训练。
- 交接只读核验时两节点没有EMBER训练/评测进程。所有旧实验封存；正式推进必须依后续owner指令和当时资源状态。

## 已封存的最近科学结果

- 旧complete P/Q short4 m64 fit/held为64/62（各150）；mixed meta73四点validation screen80为15/19/19/19。
- 相同18target对照四点screen为17/17/20/16；terminal128训练侧held视频breadth为55/180、13/18tasks，对meta73的42/180、9/18。
  Object仅5/40；Goal/Long改善仍不足以证明共享与迁移问题解决。
- 相同预算task75/77 whole-Writer clones为14/20，对shared18的3/20；shared相邻四点为2/3/4/3。
  说明共享训练存在代价，但不能单凭该差距确定容量、梯度冲突或优化根因。
- 同图fully-random target18四点screen为16/16/17/19；固定训练两task4/20，未消除共同学习缺口。
- width256原run已自然结束：128updates、15,660,800参数、train.exit=0；训练732.48秒，functional Panel-B498.24秒，总计1283.05秒。
  32/64/96/128 checkpoint均保留；暂停后未启动物化或闭环评测，**没有width256闭环分数**。

完整原件、样本/预算口径与边界在 [历史§6](docs/research_history.md#recent-learning)；跨轮解释在 [findings.md](findings.md)。
上述都是旧图结果，新候选不继承其性能结论。

## 本次交接整理

- 新设计记录涵盖科学动机、完整数学与因果推导、张量shape、多视频、单probe选择、X/Y与坐标decoder、GPU staged VJP/cache与现有代码迁移。
- 重写concept、长期要求、分层history、findings、README与当前账本；原6份旧设计、11份专家原文和181节完整旧账本保留在Git
  `fcdb6e43706c5fcedf10eaa5d2d459602b263016`，历史§9可按问题定位原件。
- 退役旧P/Q、bank conditioning、Natural Program、joint primal、G1/G3与Stage 0专用训练面及专用配置/测试；
  `src/scripts/tests/configs`文件共476→123（src195→69、scripts42→8、tests42→24、configs197→22）。
- 整理通用functional重放、panel读取、task/K调度与成本分配；source/data/expert/evaluation/normalization/LoRA基础保留。
  `configs/pi05_writer_data_v1.json`集中可复用来源，历史角色不是新实验授权。新图仍无可运行训练入口。
- AGENTS新增可并行工作应主动委派、隔离写入及主agent集成责任；本次由设计记录、代码整理、存储审计三个subagents并行完成各自范围。
- 删除222组可重建派生缓存，共82,122个payload文件、245,347,917,824 allocated bytes（228.50 GiB）；
  保留所有cache/entry JSON、生成配方、run contracts、metrics、raw rows、正常化参数和唯一checkpoints。
  缺上游Writer的两个小smoke缓存、两个各约44GiB正式run root及独特轨迹证据保留。
- strg01清理后quota复查：data1 488,035,348 KiB（465.43 GiB），soft1,073,741,824 KiB、hard1,084,227,584 KiB；
  清理前727,683,088 KiB。data0 57,471,972 KiB，使用独立quota。此为当时快照，下次大增长前须现场复查。
- 精确删除范围、保留例外、重建依赖、width256完成核验与工作树清理记录位于
  `runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`。

## 验证与交付

- 在集成后的主工作区新跑 `PYTHONPATH=src .venv/bin/python -m pytest -q`：139 passed，20.75秒。
- 6个保留Python CLI的`--help`全部exit0；当前Python源码语法、22个JSON配置与两份shell脚本语法检查通过。
- 当前Markdown本地文件链接无缺失，17份退出活动树的设计/评审原件仍可由冻结Git读取；`git diff --check`通过。
- 10个已完成工作树已移除；两个写入subagent的交付范围与main集成内容一致，task分支已删除。临时启动/诊断副本与Python/pytest缓存清理完成。
- 代码整理与设计记录已合入main；本文及其余文档随交接提交推送。最终提交与remote一致性以Git实际状态为准。
- 验证覆盖当前保留的工程基础；新架构GPU forward、Meta梯度、profile和闭环尚未运行，不能据此宣称新方法有效。

## 下一步

新session先按 [README](README.md) 的阅读顺序充分理解，报告完整数据流、历史关键教训、当前代码能力与待实现差距、
执行/验证/行为裁决计划；等待owner明确同意。获准后依 [task_plan.md](task_plan.md) 连续推进，实现到真实行为证据，
遇到问题用历史与区分性诊断找根因，不以loss、局部参数几何或重复小扫代替最终性能推进。
