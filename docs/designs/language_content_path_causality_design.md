# Language content path：匹配学习与能力保持干预

2026-09-26。**CPU构造已由主讨论独立复核，登记有界实现及执行。**
机器规格为`configs/language_content_path_causality_v1/experiment_spec.json`。
是否active、实际派发与实现提交只看progress。上一批flow诊断保持撤回。

## 1. 问题、理论与可否证性

依据[内容路径审计](../analyses/language_content_path_audit_20260926.md)及findings§152。
目标不是修正所有VLA的FM，而是检验EMBER视频条件生成链的具体参数化是否增加了能力保持/迁移代价。
原B/C六共同节点显示：C有时取得更多新能力，但后四节点丢失更多Source成功；630总差−19中，
新增差+5、损失差+24。训练任务侧没有全程统一劣势；不能把H_path写成“视频整体更难学习”。

原C的Core输入为A(q(L),E(L,V))，q只作为FrameRead query；图文E仍含语言。
干预仅为`A(q,E)+a*q`，a是零初始化、全部task共享的单一可学习标量，位于FrameRead后、语言blocks前。
原C是a=0的精确构造，B是a=1、FrameRead output=0、Procedure modulation=0且共享模块取B的构造。
这些只证明表达包含；有限Adam更新、迁移和视频利用都不由此得到保证。
梯度新增`dL/da=<dL/du,q>`及直接q内容项，实际有益对齐未知；a还参与原全局grad clipping，不能忽略其优化后果。

H_path预测：同预算下C+相对C0减少Source成功集合上的丢失，同时保留C0新获得的成功；
它必须在绝对闭环中体现，不能只看a、loss或表示变化。
H_information/shortcut预测：q路径至多恢复语言方案，视频的有效知识未改善；即便C+超过C0，也未必超过B，
或另一正确教学视频不兑现同样收益。只恢复B不算EMBER视频问题已解决。
若C+仅多获取新成功、原能力损失不减，只登记经验收益，不能确认H_path。
若减少损失却牺牲原C的新能力，同样不是联合修复。

旧v5.2/v6视频依赖正例说明q残差不是视频利用的必要条件；旧language query删除和shared-prior残差失败
说明增加语言/残差不保证有益。本试验不改变query路由、Procedure、heads、动作目标、数据支持或学习算子，
不把这些旧路线改名叠加。历史兼容性不是解释已成立，本批可支持、削弱或未识别该具体参数化。

## 2. 两个fresh学习臂与两个冻结参照

| 身份 | 作用 | 初始化及输入 |
| --- | --- | --- |
| C0 | 同批视频结构对照 | 原C fresh，canonical计算；a路径关闭 |
| Cplus | 唯一学习干预 | 与C0共用初始化规则，只增加零初始化a；真实语言与视频 |
| B630 | 已有语言参照，重新闭环 | 只读复用四臂B@630的既有对应bank，不重训/重物化 |
| Source | 原能力集合参照，重新闭环 | 相同冻结Source1000；不使用adapter |

B630不是与两新臂同一次物理训练；其Source/rank/数据/更新时钟匹配，但原world4与新world2不同。
Cplus−C0是主要架构因果对照；相对B630是能力基准，不假称只移除视频的单变量实验。
所有本批主配对使用新闭环行，旧结果只单列历史self差异；不拼旧Source成功集合。
不得将本诊断称为超过强MT-BC或最终方法资格。

训练与生成必须复用canonical owner，不复制Writer、policy或evaluator。候选采用明确配置开关，
默认旧路径/旧checkpoint仍能按原身份读取；不让新字段静默改变历史运行。

## 3. 数据、学习目标及固定时钟

父合同为`configs/conditional_compilation_diagnostics_v1/experiment_spec.json`及`train_C_video_fm.json`。
两新臂完全复用其Source1000、38-target rank16、agentview/stride5/末帧/full-H50、fit28及原event抽样。
fit28为2,4,5,7,12,13,17,19,22,25,28,29,32,34,35,37,42,43,51,55,56,62,64,73,95,96,97,101。
训练teacher/action仅demo0..45；teacher标签不进入Writer条件。46..49不产生梯度。
diagnostic-held8仍为0,1,14,15,20,21,36,38，官方Validation/Test不新增读取，不产生held action梯度。

每宏步4任务×(21+7)真实query，条件等权；两组均跨episode随机tau/full50真实7D FM，
保留每query 1/84和总loss尺度4/3。没有endpoint/prefix辅助、RL、任务slot改动或蒸馏。
seed7、sampler20260721、teacher20260722、teaching20260919不变；两臂同一逻辑event/query/noise流。
AdamW lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1；warmup150/decay18000/floor1e-5/
tail1350..2250 ratio.1均复用旧函数，不因缩短窗口重标定scheduler。a没有单独LR、正则或强制目标值。

每臂fresh **630宏步**，70560 query，fit每task90次条件；无公共底座、旧Writer/Adam继承。
原event owner的`data.maximum_updates=1260`表示预生成计划容量，可保留原值并严格只执行其前630步，
避免为截短窗口另写sampler；实际优化/验收/续跑上限仍为630，余下metadata不能形成启动授权。
六个完整checkpoint105/210/315/420/525/630；本阶段只评固定末点630，不做中途选峰。
保留420供有必要的后续相邻检验，但本阶段不自动评测它。Source基础权重始终冻结，全部允许Writer/Meta联合学习。
两臂均world2、同节点内DDP；允许两臂在不同GPU节点并行，每臂完整恢复锁world2。

每宏步保存原有学习证据，以及更新前/后的a、完整梯度归并后且clip前的a梯度；不能在forward读取旧的`.grad`。
在已发生的同一Writer计算中汇总有效token上的
`||a*q||`、`||A(q,E)||`和内积（FP32读出、detach），不得追加forward、改变loss或读取held标签。
这些用于区分新增路径是否实际活动，不设置“必须变大”的优化目标，不把norm当修复。

## 4. 第一级评测：固定736条，先回答是否值得扩大

不是single-checkpoint paired400方法资格；不据本screen选canonical模型。
固定节点630，无成绩驱动的checkpoint选择。完整评测矩阵如下：

| 面板 | 模型/条件 | states | 行数 |
| --- | --- | --- | --- |
| held correct | Source、B630、C0、Cplus各一格 | held8×0..9 | 320 |
| seen correct | 同四模型 | 原seen16×0..3 | 256 |
| held other | C0、Cplus各一格 | held8×0..9 | 160 |
| 合计 | 10面板 | 唯一固定模型/条件/task/state | 736 |

seen16为2,4,5,7,12,13,17,19,22,25,28,29,32,34,35,37；teacher对state0..3分别46..49。
held取原canonical seed20260911、video pool0..49、不放回排列中的对应state0..9；other offset17，
每臂内视频不重复且逐行不同于correct。此处80是明确登记的有限池screen，不冒称使用全部50视频的一轮正式400。
四模型之间同task/state、环境/策略seed7；成功即停的noise列表按共同replan前缀比较，且逐条验证完整派生规则。
其余canonical评测：render256/model224、双相机rotate180、state8/action7、10-flow、执行前5、settling10、
suite horizons220/280/300/520、成功即停，均不变。没有探索噪声或task-local更新。

复用B630原held/seen bank中精确对应条件，共144个引用；不重物化B，不混B420。
新C0/Cplus各held80＋seen64＋other80，共448条件，正常完整Writer生成，不按video挑选/平均。
other虽同task，必须分别完整forward；不复用correct LoRA冒充另一视频。
不生成deferred states10..49的bank，不做wrong/shuffle/reverse或额外动作probe。

每个面板的第一登记行是计入总额的pilot，共10行；验收工程/配对/被动采集，不按成功数决定放行。
全736行保存compact轨迹、实际动作、t0及每控制步T+1对象/fixture/region/EEF/夹爪与BDDL谓词，
复用现有passive owner及已修复arena-site身份分支。不能只让full cases有谓词。
固定full共28：held的global0/14/21/38、state0，四correct模型＋两other模型共24；
seen的global2/state0、四模型共4。full双相机，不按结果换case。

## 5. 分析与停止分支

先验收行数/身份/退出/trace，再读成功数。保存全部raw rows、success sets、per-task/suite/breadth、
Source-relative retained/gained/lost、两臂彼此R/G/L/churn/Jaccard。seen/held分别报告，不混分母。
主要比较：held正确Cplus−C0、Cplus−B630；能力保持读出为同批Source成功集合上的lost差，
以及C0在Source失败集合的新成功被Cplus保留/丢失的数量。列出Goal21与其余七任务，同时保留完整8-task总量。
另一正确视频分别对同格C0及B630；seen检验是否把任务迁移失败误称总体学不会。
对held8/seen16分别做20000次task-cluster paired bootstrap、seed20260926，所有同面板差共用重采样。
保留单训练seed、有限任务和screen尺寸限制，宽区间记为未识别，不能拿“不显著”证明等效。

- Cplus同时超过C0和B630、两正确视频方向一致、减少Source损失且保住C0的新能力：值得补正式400及相邻节点。
  仍非修复证明，下一阶段需新合同，不能自动扩量。
- 只接近/恢复B630、未见额外视频能力：可能减少结构代价，但EMBER目标未解决；不通过压低wrong补证据。
- a路径活动但未取得联合能力改善：本窗口不支持继续投入此参数化；不扫描a初值、尺度、层位、seed或延长节点。
- a路径未形成可辨识作用、或区间宽无法区分小效应：登记未识别；不把它包装成架构普遍无效。
- 训练侧改善而held不改善，或只提高新增但保持不变：按事前解释边界记录，不能事后称H_path已证实。

任何分支都先回到“能解释EMBER哪项缺口、会改变何种方法决策”再派下一项，不递归追局部指标。

## 6. 工程验证、资源和来源

CPU构造41a0f4f4：a=0复现原C、指定构造复现B的76输出maxabs均为0，非零head输出maxabs .62744；
视频扰动作用maxabs .44040，padding余项6.41e−7，a梯度与内积均 .07022614。
主讨论审阅9行源码改动与合成验证，独立复跑2项通过；这不是原生读取或性能证明。
正式GPU统一clean pushed detached提交F；训练/新bank/闭环计算同一F。
历史B bank的43d801b1来源明确只读保留，不要求重新生成以制造形式上的单提交。
仅CPU选择文件/调度/只读验收修正，若不改变输入、模型、梯度或rollout，可用新clean pushed提交完成对应阶段，
如实登记阶段来源和diff，保留有效行；不热改F、不重跑有效episode。涉及GPU科学行为的变化须主讨论重新裁决。

工程检查复用已有trainer/materializer/evaluator：两臂各最多4宏步含2→4完整恢复；Cplus最长fit视频
最多1个额外full-H宏步profile；每臂最多一个真实LoRA→canonical动作接口检查，不追加环境episode。
smoke权重与正式初始化分离。CPU测试围绕配置、活动梯度、恢复、subset物化/采集身份，
不得新建平行训练器、第二套评测器或大量只验证字段存在的测试。

预计两臂world2训练各约2.6小时、合计约10.6 GPU-hours（旧实测15.1秒/宏步，非实时保证）。
原736行相近面板的物理GPU占用约9–15秒/行，物化及初始化另计。正式启动前以真实smoke/profile核对可行性。
总预算含训练/物化/评测/加载/工程/失败 **18 GPU-hours**；不预先消费满额，不用旧计时冒充本批实测。
研究根`/data0/user/ymdai/ember_runs/language_content_path_causality_20260926`，data0新增峰值≤12GiB；
开发＋正式代码树data1≤768MiB，不复制source/data。正式最多6物理卡；建议两个world2训练并行，
评测按live余量动态队列，单节点至多6，全部launch遵守双节点实时准入与项目总上限。
每次选卡前live核对；大root创建前strg01独立quota/共享容量/实际用量及峰值。资源不足等待或按同合同串行，
不更改科学参数、不杀别人任务、不dummy占卡。

一个简洁launch合同保存实际命令/env、F/runtime、来源、调度/恢复及预算；正常长任务一次持续等待退出事件，
不例行读日志/cache。机器故障、nonfinite、数据墙/身份失配停止受影响执行，保留原件并回报。
执行结束核对2×630更新、448新bank＋144旧引用、736唯一行、28full和736 T+1，生成completion及分析索引。
由实际Sol主动Queue主讨论完成信号、原件、缺项、资源后停止；不自动追加模型/节点/矩阵。
