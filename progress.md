# EMBER progress

更新时间：2026-09-07 CST，接管授权已更新，理解与实现准备进行中。

## 当前授权与方法状态

- Owner于9月7日明确授权：充分理解仓库后，连续自主开展实现、验证、训练、评测、诊断及证据支持的修正；
  进度说明不是审批节点。此前“未授权科研执行”“先报告再等待同意”等交接限制已被覆盖，跨上下文不得据此误停。
- 持续授权覆盖现有科学目标、信息墙、数据、资源与Git合同内的常规决策；不联系外部专家。
  只有改变科学目标/信息墙、未授权数据资源、不可自行裁决的重大投入分歧或越权破坏性操作才需owner决定。
- 已登记active design：[layered_relation_video_writer_design.md](docs/layered_relation_video_writer_design.md)，
  **已对齐并授权实施；新图尚未实现或获得性能证据**。唯一主线为Writer与读取侧Meta fresh端到端联合训练、fresh optimizer/scheduler，
  source基础权重冻结；不实施G1--G3冻结课程，不额外建立阶段初始化候选。LoRA合法identity初始化保持。
- 当前候选：冻结图文prefix、单固定probe、Action Expert共享观察Meta、18层×完整50H；局部帧对独立50×50关系，
  两端分别softmax；关系MLP消费内容、rho和signed gap；同步邻居聚合、4个radius4 blocks、H-read；
  置换不变多视频compiler、坐标MLP生成唯一38-target完整rank16 LoRA。首版真实K1/2/4。
- 环境长期goal已创建：实现并以学习/闭环推进稳定视频条件迁移，满足validation8 strict paired single-checkpoint >145/400、
  相邻稳定/低churn/高breadth/四suite及GoalLong/同task视频鲁棒性；选点冻结后完成因果controls，方法冻结后32/8 fresh最终训练与Test。
  实现完成、训练结束、单点高分均不代表goal完成；未自设token预算、总工期或总尝试数。
- 接手Git实测：main干净且为交接基线9ea2034037e5c70b514198a70910aac5c2fb18f5，与本地origin/main一致。
  指定当前文档已完整阅读，正在展开相关原始评审、历史修正、代码和canonical资产。HANDOFF已消费并删除，长期内容留在正式文档。
- 未选出selected checkpoint，未达成最终科学目标。不自动恢复旧width256闭环或其它旧待办。
  当前未launch新GPU工作；历史GPU/quota快照不构成实时准入，launch及大增长前重新核验。

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

完成相关历史原件、保留工程基础与canonical资产的理解，简要报告数据流、证据边界、实现缺口与近期安排后直接实施。
按 [task_plan.md](task_plan.md) 连续推进唯一新图、真实Meta/VJP/多K机制检查、最长真实视频成本测量、跨suite短学习与闭环，
再依据证据进入train24共享训练与strict400。例行检查、阶段汇报或一次实验结束不构成停止理由。
