# EMBER progress

## 当前授权与实验

Owner已授权按[条件读写设计§13](docs/designs/conditional_read_write_architecture.md#13-首批实施与完整学习检验2026-10-01授权)
实施并自主推进首批fresh450：因果c/d、A₀感知S、最终A编译M、唯一38-target A_V/B_V及完整FM。
固定36task allowlist、source1000、normalization、信息墙、跨episode query与选择合同不变。
50,400 full query；450完整validation400、完整train144及固定A28；总上限40GPUh/80GiB。
先验整批含工程6–12小时，实际profile后的耗时与启动合同记录在运行根，不以先验充当实测。

唯一实验执行者`01a0f018-69af-7b00-b614-7e117540051b`负责代码、测试、排障、资源调度、Git及冻结运行。
主讨论`01a0ed66-cda4-7a23-90f0-e0d3a06a1d36`负责科学合同、原件解释、机制及后继/预算裁决。
既有自主授权持续生效，不恢复旧暂停段落，不等待主讨论工程复审。
沟通边界按[Owner要求§6](docs/current_owner_requirements.md#6-沟通与交接)：整批科学结果或确需裁决的实质边界只回报一次；
工程阶段、可自行修复的故障和普通调度记入已有记录。写入/Git窗口与实际冲突方直接串行协调。

## 最新已登记工程身份

源码`ad79887e`、状态`b27cf26c`已集成推送；最长视频profile的activation-checkpoint重放修复为`797ae01f`。
原冻结树`/data1/user/ymdai/projects/EMBER-conditional-read-write-formal`保持不变；
修复后版本在`/data1/user/ymdai/projects/EMBER-conditional-read-write-r2-formal`完成最长视频检查及原四卡训练。
运行根为`/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001`，恢复原件为`launch/replay_repair.json`。
旧profile首个更新失败：0完整更新、exit1、0.027361GPUh；原件保留并计入预算，不作科学负结果。
最长105帧/28query的三次fresh完整FM检查已通过，profile权重没有转入正式初值。
原四卡formal已完成90步完整ECP并安全停止，exit0、1.754886554GPUh；不是450科学节点完成。

Owner要求按实际可用卡数自动高效训练。通用target分片、完整cotangent返传、bucket SUM与来源承接已集成为
`7df6d43e`、`ddf17e9c`、`96b07612`，冻结树为
`/data1/user/ymdai/projects/EMBER-conditional-read-write-adaptive-formal`。
执行者从实际90ECP承接91..450原事件流，保留Adam/scheduler/sampler及已有rank RNG，新增rank RNG单列；
父训练与接续来源分别登记。不fresh重训、不丢更新，4×28有效query、完整FM及最终A/B公式保持不变。
实际启动、首消费者与分段成本在同一root的launch记录；物理卡数仍服从现有资源上限与整批预算。
尚未登记本架构formal450的完整科学结果；源码完成、CPU检查或profile均不表示学习资格或闭环阳性。
canonical tracked窗口已释放，冻结实验不依赖canonical后续整理；新写入仍须与实际冲突方协调。

## 独立整仓整理

整理会话`01a0f7fa-63b3-7a42-a196-4c0fd145b10c`，独占
`/data1/user/ymdai/projects/EMBER-cleanup-20261001`、`codex/ember-cleanup-20261001`。
从main `927b1498`建立，承接最新main `96b07612`的集成正在收尾；实验dev/frozen树未被本整理修改。
Owner已撤销对本独立任务的no-goal限制，本会话使用无token budget的goal持续完成整仓整理与实际skill维护；
主讨论在派发实验后无需goal的旧限定不扩大到本任务。不开新session或子代理，不做GPU实验。

已在本树退役旧Writer训练/生成与已关闭的专用诊断执行面，保留当前共享组件、sealed配置、实际bank/原行读取及科学原件；
统一README、稳定规则、当前状态和历史索引，修正Source配置导航变化导致的误拒绝。
源码提交`540fb773`；536项保留CPU测试及十个实际CLI help通过，配置、import与文档引用已检查。
已实际修改个人workspace-cleanup skill及其inventory helper，symlink/边界/CLI验证通过；回滚在
`/data1/user/ymdai/skill-maintenance/ember-cleanup-20261001/workspace-cleanup/`。
最初预估2–4小时；当前剩下最新main冲突处理、合并后消费者检查、已核实临时内容删除与串行集成推送。
整理分支承担自己的工程检查与交付，不把实验执行者的新增实现重复转给主讨论审查。

## 已完成研究与原件可用性

最近的self_read450为116/400，Context450126、T450122、强MT153；完整正负证据、task/suite与机制边界见
[研究历史](docs/research_history.md#2026-10-01共享参数两次native的fresh450完整阴性与分析后讨论)、
[机制分析](docs/analyses/feature_to_operator_mechanism_20260926.md)§87及findings§261。
原400/144 scenes、MT/T/Context/self_read原始结果和固定A28输入
`operator_chain_diagnosis_20260929/functional_credit_transport/group0`均为当前依赖，继续保留。
已完成§40及更早路线不因历史“下一步”自动恢复；科学阴性、失败原因和有效正例完整留证。

2026-09-23历史存储裁剪账目为`runs/analysis/workspace_cleanup_20260923/storage_cleanup.json`及
`checkpoint_retirement/closeout.json`；当时checkpoint累计回收413.766GiB，94个weights_only、80个metadata_only。
这些是当时的裁剪事实，不是当前存储实测。Source1000 frozen policy可读取，原训练optimizer/EMA载荷已退休。
每项资产的当前可用性由其`checkpoint_retirement.json`、`payload_retirement.json`与实际文件共同说明；
weights_only不含完整训练恢复状态，metadata_only不含权重。不得仅按原manifest宣称可重放或exact resume。
本次源代码整理不删除source/dataset、formal原始rows/metrics、关键模型、运行根或冻结树。

## 历史状态的恢复入口

旧progress与task_plan累积了多轮互相覆盖的“当前/暂停/尚未实施”。完整当时过程保存在Git：
`git show 797ae01f:progress.md`、`git show 797ae01f:task_plan.md`。
按日期/方案追溯的入口为[research_history](docs/research_history.md)，跨轮结论在[findings](findings.md)。
这些历史快照记录当时授权、成本、失败与交接，不作为今天的执行authority；不新增平行状态或in-tree archive。
