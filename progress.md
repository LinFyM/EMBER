# EMBER progress

## 当前状态：四臂完成并复核，登记冻结前段轨迹干预（2026-09-24）

Owner最新指令是由本任务接管主讨论与科学决策，以历史证据、竞争机制和可反驳干预推进；具体实验交由现有Sol执行，
每批完成主动回报，待主讨论分析后再派发下一步。这取代旧全局暂停；不恢复旧deadline或GPU特例。
Owner随后明确给予充足时间和持续优化的广泛分析/实验授权：主讨论按证据自主推进后继机制验证和改进，
不能停留在现象、随意试改法，必须逐轮保存假设、预测、原始证据与经验。授权覆盖四臂之后的研究，
当前已启动批次仍按冻结合同执行；科学信息墙、资源限制和结果可追溯要求继续适用。
Owner又明确本次授权**不向Sol转发**，让其专注具体实验；主讨论自行维护长期判断和记录，收到本批结果后再给出具体下一步。
主讨论：`01a0cd94-65da-7b22-8ca9-7ba35f454632`（接管 EMBER 科学决策与实验）。
实际执行者：`01a0cd90-ebb7-77a1-a20b-a858825d2f66`（了解 EMBER 仓库现状，Owner指定Sol）；已核对同仓库/主机，四臂已完成，新批投递状态如下。
双方保持现有模型配置；旧Luna和旧主讨论仅作历史provenance，不再作为收件人。
**当前active design为[冻结前段轨迹与视频条件因果诊断](docs/designs/frozen_prefix_causality_design.md)**，
机器规格`configs/frozen_prefix_causality_v1/experiment_spec.json`；设计提交`3258e304`已推送，已向同一Sol执行者Queue派发具体实验。
2026-09-24 00:55 UTC入队回执`01a0d0e9-1dee-79b3-abfd-2c59a20e20ab`；一次只读查询确认消息已进入新活跃轮
`01a0d0e9-1df2-70a0-94fc-0a1f7742a897`，尚未据此宣称实现完成或正式分支已启动。
精确正文、回执和查询证据在旧study的`coordination/prefix_dispatch_20260924.{txt,json}`；不重复发送或要求例行确认。
原[四臂诊断](docs/designs/conditional_compilation_diagnostics_design.md)已完成：54面板13200行、四臂全部1260更新、
selected controls及480条登记动作probe齐全；Sol主动完成回报后已停止新增实验，registration为`registered_batch_complete`。
主讨论独立原始行复核和裁决见findings§135。新批次研究根为`/data0/user/ymdai/ember_runs/frozen_prefix_causality_20260924`，
创建及资源检查由Sol执行；旧四臂runtime和产物只读保留，不改原训练、选点或结论。
Sol在独立worktree拥有代码、训练配置、测试和run产物；主讨论拥有科学解释及主线状态文档，集成前协调避免覆盖。
短期步骤、结果解释分支和派发要求见[task_plan](task_plan.md)。
接手后在既有授权内完成实现/核验/有界诊断；旧的组会deadline、无上限GPU及其它历史运行许可不恢复。
长任务正常期间等待完成或异常事件，不固定间隔读取训练进度、日志、checkpoint或共享缓存；
9月23日长期授权本身未转发Sol；现在只派发经过分析后形成的具体实验。

### 四臂结果与下一步理由

Source held58/400、seen9/64；A/B/C/D selected为420/630/420/1050，held108/124/110/123，
对应Source R/G/L为49/59/9、48/76/10、34/76/24、40/83/18。B后四节点高于A，
不支持生成链普遍不会学习；C有能力获得，但保留损失更大。C controls110/98/126、D123/110/80，
D相对wrong有优势，却尚未证明优于语言B的正确视频增量；D末段held丢60得28、seen仍改善。

实际flow probe与闭环不一致，固定画面出现错误对象和同任务反例；历史已有异质失败及空间监督的局限。
据此下一批冻结B630/C420，以两任务、50初态、两种保存动作前段、25/50步截断和四个接手条件组成1600个分支，
直接区分状态历史与同状态的策略/视频条件作用。无训练、无新增held expert actions、无官方Validation/Test或checkpoint选择。
这仍是有反驳条件的定位干预；待原件回报后由主讨论继续分析，不把分支成功率当新部署方法成绩。

### 四臂工程核验及启动历史快照（2026-09-23）

Study根：`/data0/user/ymdai/ember_runs/conditional_compilation_diagnostics_20260923`。
正式运行树：`/data1/user/ymdai/projects/EMBER-conditional-formal`，主讨论实查为clean、detached的`43d801b1`。
`launch/formal_launch_contract.json`登记精确命令、四臂共同commit、资源、预计增长和恢复合同；
`registration.json`已由Sol更新为`formal_training_CD_running`。

- Sol报告CPU针对性检查304+17+5通过。主讨论已核对源码中`SupervisedEngine.validate`归属修复、
  四臂smoke的1..4宏步记录、各自2→4恢复日志及第4步checkpoint manifest；每臂累计448条动作查询。
- `launch/profile_longest_C.json`和`profile_longest_D.json`记录最长fit视频347原始帧→71采样帧、full H50、
  21+7查询及非零活动梯度，峰值reserved约22.69GiB。`materialization_eval_interface_smoke.json`
  记录一次Writer编译、真实adapter加载及1×50×7动作输出，明确`formal_score=false`；这些只证明工程接通。
- D于10:32:12 UTC在gpu01的0/4卡启动，C于10:32:35 UTC在gpu02的0/1/2/3卡启动，合计6张物理卡。
  主讨论18:38北京时间只读核对formal run contract：同一commit、dirty为空、world size分别2/4；
  当时D已18步、C已27步。A/B尚未启动，按登记顺序随后运行；不以此瞬时步数解释科学效果。
- 已核对launch前两节点GPU记录及strg01两文件系统独立quota原件。合同预计data0新增峰值约76.5GiB，低于96GiB预算；
  每次后续launch/resume仍由执行者刷新资源状态。训练约7–8小时是基于profile的估计，评测ETA待首个完整面板实测。

本次只更新运行状态，不改变科学参数、节点或选点规则，不触碰冻结运行树；无需重复启动批准。
继续按六节点完整held400/seen64、Source相对保持/获得/丢失及冻结选点后的C/D视频controls裁决。
执行者完成本批主动回报并停止追加实验，主讨论核对原件后才派发下一项针对性干预。

### 本次接管的证据核验与判断

已按README读取当前合同、concept、findings§127–133，沿历史审计回看7–9月路线、9月11日完整专家论证、
9月19日教学提案及最终修订、近期输出空间/因果/配对报告和实际Core/Procedure/FactorHeads/采样/功能信用代码。
本轮只做历史只读复算，没有新增模型forward、梯度、rollout或Test读取。

- 原始行重算旧v5.2/v6的四个匹配访问面板，确认132/51与95/111的配方交互，仍保留Adam次数/LR/查询数混杂。
- 原始行重算C与夜间fresh各六correct节点及各自other/wrong：16个完整400面板，唯一行、50视频及worker退出通过。
  两模型selected600之间1200行逐臂核对task/state/语言、env/policy噪声共同前缀、真实teacher帧元数据、Source与normalization一致。
  correct154→142是保留94/新增48/丢失60；other160→132是88/44/72；wrong153→105是81/24/72。
  因而correct−wrong从1增至37伴随两种正确输入绝对下降；这是跨运行描述，不是隔离了单一训练机制的因果估计。
- C600局部接口五臂的160条原始行重新计数为9/16/11/11/14（每臂32），没有自由A/B显著占优的依据。
  该检查有task-local、有限预算和步幅校准限制，不能反证所有共享生成的优化问题。

小型复算证据在`coordination/scientific_recheck_20260923.json`；跨轮记录见findings§134。
当前竞争解释、可反驳预测及四臂不能回答的部分写入task_plan；未登记任何第五臂或自动后继训练。

此前Owner认为仅清理缓存不充分，明确要求进一步裁剪历史checkpoint。追加清理按关键权重/历史参照/当前依赖/恢复用途择点，
额外回收360.759 GiB；连同首轮53.007 GiB，两轮累计413.766 GiB。data1用户配额实测从846.3降至485.6/1024 GiB，
data0仍为122.1/1024 GiB。原始评测、配置、关键模型及当前诊断依赖继续保留。

两轮清理按当时暂停研究的要求执行；Luna由Owner停止，**没有创建Sol任务**。
长期理解、文档/代码/缓存与worktree整理、追加资产退休已完成并在`b711c2af`集成推送。

## 追加checkpoint裁剪

- 9月8–14日33条结束路线：回收222.737 GiB，保留70个关键权重路径；8月3–7日12条结束/作废路线：
  回收115.220 GiB，保留24个代表权重及64份完整评测原件。合计174个历史checkpoint：94个`weights_only`、80个`metadata_only`。
- Source1000预先固定使用raw policy；退休无使用用途的optimizer及未选EMA副本，回收22.802 GiB。
  raw policy的canonical inspector在删除前后返回相同结果，正式推理仍可用；Source完整训练恢复和EMA不再可用。
- 合计删除256个大载荷路径、255个独立inode：trainer约287.989 GiB、Source optimizer 14.090 GiB、
  非关键Writer权重49.968 GiB、未用EMA 8.712 GiB。保留原合同/manifest、日志/指标、原始结果和小型rank RNG记录。
- 删除trainer的历史Writer不再exact resume；其原冻结CLI会因完整文件校验而拒绝加载，今后重放需要明确支持
  weights-only的读取路径，不能伪造trainer或重写历史manifest。已导出9月模型的标量/结构元数据；8月模型所需元数据在原合同/consumed中。
  9个受影响的物化缓存退役记录已同步上游可用性。当前checkpoint retirement记录优先于旧日志的“完整保留/可直接重建”描述。
- 94个保留权重的safetensors头均可读；O1200/C600/C1200/MT-BC300的全部manifest文件和大小核对通过，
  库存无计划外消失文件。数据、teacher/native observer/stable carrier/fit19依赖、旧V5.2参照和近期完整训练轨迹未动。
  本轮没有运行模型、GPU实验或重算科研结果。data1已盘点checkpoint载荷从592.366降至231.607 GiB；
  剩余小型历史节点不一概宣称必不可少，但本轮未对缺少逐项退休依据的资产继续扩大删除。
- 精确清单及验证：`runs/analysis/workspace_cleanup_20260923/checkpoint_retirement/closeout.json`；
  每个受影响checkpoint原目录有`checkpoint_retirement.json`，历史manifest保持原内容。

## 接手先读

1. [Owner要求](docs/current_owner_requirements.md)：固定LoRA可达下界并非公共底座课程；有益视频利用、理论解释深度与协作边界。
2. [Concept](docs/concept.md)：完整pipeline，以及生成Jacobian、端点/前缀监督、任务关系覆盖三类待检验解释。
3. [Findings](findings.md)§127–133与[研究历史](docs/research_history.md)：近期原件、正负证据和不能外推的结论。
4. [接续的条件编译设计](docs/designs/conditional_compilation_diagnostics_design.md)：四臂用途、混杂与未覆盖的数据原因。

## 当前实证快照

| 方法／固定checkpoint | coverage Validation correct/other/wrong | coverage Test |
| --- | --- | --- |
| Source1000 | 51／不适用／不适用 | 75 |
| MT-BC300，rank128 | 155／不适用／不适用 | 121 |
| 原跨episode辅助Writer C600 | 154/160/153 | 未跑 |
| 夜间12任务条件fresh，selected600 | 142/132/105 | 未跑 |

所有表中分数均为每臂400条；不同task集合不能跨Val/Test逐行配对。baseline Test是Owner授权的历史暴露任务补测，
不能描述为新盲测，不反哺选点/划分/方法。Source Test更高、MT-BC更低不证明划分无效，也不支持“Test统一更难”。
夜间1200步fresh完整曲线103/117/142/141/138/107；训练侧小试收益没有转为超过155的正式泛化结果。
已完成correct/other/wrong；shuffle/reverse在完成前被停止，没有完整分数；当前EMBER Test、FT、RL均未启动。
目前没有验证出同时解决绝对性能与有益视频特异性的方案，亦没有证据证明架构理论上已到上限。

原件根：
- `/data0/user/ymdai/ember_runs/overnight_root_cause_20260922`：近期有界诊断和小试。
- `/data0/user/ymdai/ember_runs/coverage_task_mixing_20260923`：夜间fresh、完整checkpoint、配对原件与selection。
- `/data0/user/ymdai/ember_runs/coverage_baseline_test_20260923`：Source/MT-BC Test原件，交付commit`7b18030c`。
- [夜间小型结果包](docs/review_materials/20260923/overnight_results/README.md)：当时PPT使用的指标与图，不包含后续baseline Test。

## 原WIP交接与已完成集成

原分支 **`codex/conditional-compilation-diagnostics@73267f53`** 保存Luna未完成实现，仅作历史交接来源。
Sol从最新main隔离开发并逐项修复、验证后，以`43d801b1`集成推送；没有整体恢复旧文档或已退役实现。
当前设计、`experiment_spec.json`和`partition_audit.json`的科学合同保持；运行状态以本文件顶部和study登记为准。

候选fit28＝官方Train中的seen target16＋已审计aux12；diagnostic-held8是其余官方Train任务，官方24/8/8未改。
A直接共享rank16、B语言Writer、C完整视频Writer、D相同视频结构但第二监督组改tau1/前5步；均计划fresh。
这只能定位整体环节，不能唯一分离参数化/活动容量，也没有数据构成干预。旧MT-BC看过这组held8，不能充当A的未见任务结果。
完整分组、步数、节点与每个比较的边界以冻结spec和设计为准；工程验证和C/D启动不构成新的性能结论。

原WIP的`validate`曾错误缩进在`configured_endpoint`的return之后；现已恢复为`SupervisedEngine`方法，
并补齐实际forward/backward、保存恢复及物化/评测接口验证。原WIP分支不作为formal运行来源。
新主讨论依据既定四臂合同推进并独立判断结果；执行任务不能把此交接扩大为任意新实验。

## 首轮收尾记录（追加checkpoint裁剪前）

- 长期要求和concept已整合；docs按designs/analyses/review_materials分工，历史材料有明确状态。
- task_plan和progress只保留当前状态，重复历史移至既有研究历史/发现入口；原详细过程可从Git恢复。
- 已结束的stability/output-space/causal专用诊断和low-LR执行面退役；历史实现以`7b18030c`及run记录的commit恢复。
- 代码retirement净删5101行、删除15个专用文件；保留常规resume、topology migration和1500→2100 continuation。
- 108项受影响CPU检查：107项首次通过；1项引用旧文档路径的断言修正后单独通过。三个canonical CLI的`--help`均exit0。
  40份配置JSON解析通过，Markdown本地链接无失效，源码/脚本/测试无对退役诊断模块的残留引用；未运行GPU或新科研验证。
- 三个临时worktree已移除，仅剩main；已合并的cleanup-code分支删除，未合并WIP的本地/远程分支保留且远端核对为`73267f53`。
- 大资产已完成回收：52个可再生LoRA bank共9739个路径（9339个唯一inode），44.924 GiB；
  明确一次性的profile载荷0.276 GiB；worktree副本0.731 GiB、已消费handoff/下载缓存/派生预览0.246 GiB。
  追加核实574个旧物化manifest、1717个payload，回收6.794 GiB。连同生成cache和本次临时文件，总计回收53.007 GiB：
  data0为38.682 GiB、data1为14.326 GiB。Hardlink按实际inode计数；原manifest和全部重建依赖保留，载荷原目录有退役标记。
  strg01收尾配额实测data0 122.1/1024 GiB、data1 846.3/1024 GiB；这是实时用户用量，不是共享磁盘空闲量。
- 清理账目：`runs/analysis/workspace_cleanup_20260923/storage_cleanup.json`；Source/teacher/正式checkpoint、raw rows与
  不可确认可再生的authority/task-local/轨迹证据保留。data1的大头是约530 GiB formal checkpoint载荷，另有约59 GiB探索checkpoint，
  不能根据负结果或目录名当作缓存删除。277组authority/task-local/特殊转换或provenance不全的产物有逐项保留原因，
  不凭“cache”命名推断可删。旧scratch中的唯一诊断记录和可用uv安装器也保留。
- 最终main包含文档/代码清理与验证修正，按仓库交付规则推送远端；仅保留canonical worktree。
  收尾没有启动新GPU实验、创建Sol或给任何任务派发后继实验。

上述首轮收尾段是历史事实；当前接续授权、实际收件人及四臂启动状态以本文件顶部为准。
