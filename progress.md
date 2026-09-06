# EMBER progress

更新时间：2026-09-07 CST，局部关系设计收口与再次交接。

## 当前授权与方法状态

- Owner已完成局部关系模块的数学重推对齐，并明确要求更新仓库、重新提供新session启动prompt。当前授权为文档/设计/交接同步、
  必要文档核对与Git交付，不包括新架构实现、新的科学训练/评测或旧运行恢复。
- 已登记设计：[layered_relation_video_writer_design.md](docs/layered_relation_video_writer_design.md)，状态为
  **已对齐、具备进入实施计划的完整候选定义；尚未实现/验证，未授权科研执行**。旧past-only过程定义已退出当前实施合同。
- 当前候选：冻结图文prefix、单固定probe、Action Expert共享观察Meta、18层×完整50H；窗口内每个帧对一个50×50关系，
  两端分别softmax；关系MLP消费对应内容、相对位移分布及signed gap；每帧最多8邻居聚合，同型block堆叠，H-read得到E；
  置换不变多视频compiler、坐标条件MLP生成唯一38-target完整rank16 LoRA。具体默认与待冻结配置见设计§9/14。
- 新session先充分理解仓库和历史，报告完整数据流、实施/验证与行为裁决计划；得到owner明确同意后才自主推进。
  本session只提供可复制prompt，不自动创建或启动另一个session。
- 未选出selected checkpoint，未达成最终科学目标。旧goal或未完成清单不覆盖当前授权，也不自动恢复width256评测或旧训练。
- 前次交接只读核验时两节点没有EMBER训练/评测进程。那是当时快照；本次纯文档更新未刷新GPU/进程状态，实际launch前须重新准入。

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

## 前次仓库整理

- 新设计记录涵盖科学动机、完整数学与因果推导、张量shape、多视频、单probe选择、X/Y与坐标decoder、GPU staged VJP/cache与现有代码迁移。
- 重写concept、长期要求、分层history、findings、README与当前账本；原6份旧设计、11份专家原文和181节完整旧账本保留在Git
  `fcdb6e43706c5fcedf10eaa5d2d459602b263016`，历史§9可按问题定位原件。
- 退役旧P/Q、bank conditioning、Natural Program、joint primal、G1/G3与Stage 0专用训练面及专用配置/测试；
  `src/scripts/tests/configs`文件共476→123（src195→69、scripts42→8、tests42→24、configs197→22）。
- 整理通用functional重放、panel读取、task/K调度与成本分配；source/data/expert/evaluation/normalization/LoRA基础保留。
  `configs/pi05_writer_data_v1.json`集中可复用来源，历史角色不是新实验授权。新图仍无可运行训练入口。
- 本次由设计记录、代码整理、存储审计三个subagents并行完成各自范围；通用委派规则随后按owner纠正统一维护在用户级AGENTS，删除项目内重复要求。
- 删除222组可重建派生缓存，共82,122个payload文件、245,347,917,824 allocated bytes（228.50 GiB）；
  保留所有cache/entry JSON、生成配方、run contracts、metrics、raw rows、正常化参数和唯一checkpoints。
  缺上游Writer的两个小smoke缓存、两个各约44GiB正式run root及独特轨迹证据保留。
- strg01清理后quota复查：data1 488,035,348 KiB（465.43 GiB），soft1,073,741,824 KiB、hard1,084,227,584 KiB；
  清理前727,683,088 KiB。data0 57,471,972 KiB，使用独立quota。此为当时快照，下次大增长前须现场复查。
- 精确删除范围、保留例外、重建依赖、width256完成核验与工作树清理记录位于
  `runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`。

## 前次整理的验证与交付

- 在集成后的主工作区新跑 `PYTHONPATH=src .venv/bin/python -m pytest -q`：139 passed，20.75秒。
- 6个保留Python CLI的`--help`全部exit0；当前Python源码语法、22个JSON配置与两份shell脚本语法检查通过。
- 当前Markdown本地文件链接无缺失，17份退出活动树的设计/评审原件仍可由冻结Git读取；`git diff --check`通过。
- 10个已完成工作树已移除；两个写入subagent的交付范围与main集成内容一致，task分支已删除。临时启动/诊断副本与Python/pytest缓存清理完成。
- 代码整理与设计记录已合入main；本文及其余文档随交接提交推送。最终提交与remote一致性以Git实际状态为准。
- 验证覆盖当前保留的工程基础；新架构GPU forward、Meta梯度、profile和闭环尚未运行，不能据此宣称新方法有效。

## 本次设计修订与交接

- 完整设计更名为layered_relation_video_writer_design.md，保留一个canonical方法文档；重写过程推导、shape、GPU布局/成本与验证定义。
  原past-only文本在Git 12d9689c及此前提交中保留，历史§7记录修正原因与边界。
- 同步concept、长期要求、findings、README、task_plan、分层历史和HANDOFF；内容差与对应位置变化均有明确消费者，
  允许双向教学帧读取，禁止重新使用旧的未来帧不变性测试。
- 已审阅9份文档的相关差异，核对55个本地文件引用均存在，原初稿Git引用可恢复；旧单向定义只保留在历史或明确的退役说明中。
  本次git diff --check通过；只更新文档，未重跑前次139项工程测试，未进行GPU/新模型验证。
  main交付随本次文档提交推送，Git实际状态为准。

## 下一步

新session按 [README](README.md) 和临时 [HANDOFF](HANDOFF.md) 进入，完整理解当前设计、历史关键教训、可复用代码和待实现差距，
先报告具体实施/GPU算法、数据/采样、验证与闭环裁决计划，等待owner明确启动指令。获准后依 [task_plan.md](task_plan.md) 连续推进，
遇到问题用历史与区分性诊断定位最早失效接口，不以loss、局部参数几何或重复小扫代替最终性能；消费HANDOFF后删除它并更新临时入口。
