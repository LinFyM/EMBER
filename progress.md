# EMBER progress

## Owner最新指令：三步机制研究与持续迭代（2026-09-11）

Owner明确要求建立goal：完整理解从科学精神、数学推导、专家讨论到当前架构的形成过程；随后定位为何同任务正确顺序视频没有带来必要执行收益；依据原因改架构或训练、验证，失败返回原因分析，持续到正常训练自然产生正确视频优于错任务/乱序视频的可信行为分化。不得人为压低错误视频性能制造差额，绝对能力目标仍保留。

Goal已建立。Owner随后补充correct必须至少高于冻结source；该条件已与correct高于错任务、乱序共同写入goal验收标准，不能只靠错误条件变差达成。第1步原始材料与代码审查已完成，第2步完整冻结诊断已完成，进入第3步局部表示修正与真实profile；不启动盲目覆盖实验。95-task计划不恢复，两个未提交worktree继续原样保留。后续授权含必要实验与实现，无需逐项再次批准，但每次实验须按最新task_plan登记用途、信息墙、配对与资源合同。

最新授权允许错任务/乱序开发诊断用于原因分析与架构修正，区别于最终独立冻结验证；旧禁止开发反馈的条款不覆盖本次授权。当前启动的仅为下述冻结诊断。C/S仅有100/200两个节点，其失败结论限于该观察范围；特别是C43→50不能视为已证实收敛或整个消费机制被否定。

当前唯一active design为[无变化参照局部过程更新](docs/video_change_reference_design.md)。形成链与冻结机制诊断已完成；新候选数学、128项相关CPU检查及真实四/六卡两步profile通过，进入fresh正式学习准备。原C/S仅观察100/200，未宣称收敛；95-task不恢复。

### 当前机制结果与候选实测

完整1472条新闭环及14336配对动作预测已完成，原件`runs/analysis/video_mechanism_20260911/`。C200正确/两错/乱序/静态/S/source为42/46/46/45/49/51/15，各96；固定S后的正确与静态过程同为15/32。source训练增量存在，正确动态净收益未建立。C100→200仍有获取，未宣称收敛。完整逐task/suite、配对、相邻与区间见[机制复核§10](docs/video_mechanism_reassessment.md)及findings§54。

局部无变化参照已集成main。真实六卡第二步14.21秒，比四卡25.02秒快；采用gpu02 0/1/2/3/4/6、micro8/8/8/8/6/6，原件`runs/analysis/video_change_reference_20260911/profile_summary.json`。最长93帧通过，静态中心化过程为真实视频约0.9%，上游有梯度且source无梯度。128项相关CPU合同通过；一次性profile checkpoint已删除。

正式配置`configs/pi05_video_change_reference.json`，fresh100/200后exact-resume300/400，每节点correct strict400及固定train96，200/400加开发视频对照。保留C的2K1条件/task、32queries/条件、普通FM和全部其它结构，只改局部更新参照。正式fresh100/200已从clean pushed detached64eba75b启动，gpu02六卡与NUMA/原任务权重核对通过；实际source、数据、observer、optimizer与C原合同相同，唯一model字段新增局部参照身份。命令与输入/资源合同在`runs/analysis/video_change_reference_20260911/launch_contract.json`，日志同目录`train100_200.log`。尚无新闭环结果，目标未达成。

/data1最新quota used811028252KiB、soft1073741824KiB，共享83TiB；新profile两次峰值≤12GiB，现有诊断1.4GiB、候选worktree184MiB，复用大资产。旧面板争用失败和撤回记录保留，最终完整且所有本任务GPU worker已退出。实际profile命令与设备登记在`.codex/tmp/video-change-reference-profile/`。

## 历史Owner暂停（已由上述定向授权覆盖）

Owner明确要求“停一下”，指出核心仍是正确视频的使用：off/普通FM与扩大task数量没有回答当前结构为何需要正确视频。立即暂停95-task实现、profile及训练；此前4480条闭环已完成，没有任何95-task GPU任务。子agent已停止，两个隔离worktree的未提交改动原样保留，不集成或继续。

当前回到机制讨论，没有获准继续执行的新科学设计。[95-task设计](docs/nonheld_meta_writer_design.md)为暂停的未验证提案，不能因旧登记恢复执行；既有训练/评测也不自动恢复。active goal尚未达成，用户暂停优先于此前自主推进授权。主工作区后文是暂停前快照。

## 历史快照（95-task暂停前，不构成执行授权）

Owner休息期间自主修正、验证与再次修正的授权及active goal继续。唯一active design为[非held meta-task扩展](docs/nonheld_meta_writer_design.md)；当前准备实现与验证，尚未启动新训练。目标未达到，Test封存。

本轮R/C/S及off、R两项有限续训已经全部结束。下一项主要变量为独立训练任务覆盖：保留较强off图、普通FM、agentview/K1与原64queries/条件，target24与已审计source71显式各半。新模型fresh，不继承126或R400。71项source数据已有且排除了target40重复语义，无新下载/normalization；旧meta73弱component负结果和适用边界已在新设计完整登记。

当前工程安排：数据/权重/恢复与C/S退役在两个独立worktree并行；主agent负责整合、保持off计算及初始化语义、正式配置、CPU合同和真实GPU检查。formal须来自clean pushed detached commit，200/400节点才可进入学习。还没有95-task profile、训练或成绩。

## 本轮完整结果

| 配方 | validation节点 /400 | train96 | 主要判断 |
| --- | --- | --- | --- |
| R，off图+两个K1条件 | 100/200/300/400=37/83/63/85 | 200/400=46/56 | 同target预算旧off400=126/56；无迁移收益，停止 |
| C，先语义后过程 | 100/200=43/50 | 49 | 弱且改善有限，不追加 |
| S，匹配无序帧集合 | 100/200=59/45 | 52 | 早期优势未保持，不追加 |
| 旧off7受控续训 | 400/500/600=126/73/54 | 400/600=56/64 | 训练任务提升、未见任务下降，停止 |

R400的S/O/G/L为1/45/32/7，breadth6；global1/3/11/13/23/26/31/32为0/1/39/6/0/32/2/5。R200→300保留43/新增20/丢失40、churn60/J=.41748；300→400为50/35/13、48/.51020；200→400为53/32/30、62/.46087。Long在200→300总数5→5但原成功全部更替；不能把总数稳定等同成功集合稳定。

旧off400126→R40085保留69/新增16/丢失57、churn73/J=.48592。R400 train96=56，四suite17/17/14/8、breadth20；R20046→56为36/20/10，旧off40056→R56为43/13/13。终点训练总分相同而迁移低41，更多条件没有带来最终收益。

R完整exact-resume沿原gpu02 4/6、world2、micro6/6和冻结7fedbe85；原合同、optimizer/sampler/rank RNG及真实201恢复采样通过。新增200更新/51200queries6422.31秒、平均31.75秒，累计102400queries/3200条件/383种task-video，仍只有24个独立映射。400独立动作FM=.105399（200=.110515），3072queries无梯度、289.10秒；source trainable=0。

R追加两个validation400及一个train96共896rows全部完整，canonical teacher、source/normalizer与执行RNG配对通过。300 val9worker/42shards1206.51秒；400 val15worker/60shards798.74秒、train96三worker/36shards888.14秒；400物化496条件747.14秒。所有本轮GPU工作已退出，没有下一段R或off训练。

首轮三臂共153600queries、2688闭环rows；off续训再896rows，R续训再896rows，本轮合计4480条闭环。C/S专属实现准备退役，全部历史仍可由冻结7fedbe85及run contracts审查；不将探索性off轨迹重标为formal。

## 证据入口与当前资源

- 本轮统一根：`runs/analysis/video_consumption_20260911/`；首轮`first_round_summary.json`。
- off续训：`continuation/continuation_summary.json`及step500/step600，完整126→73→54及train56→64。
- R续训：`r/continuation/continuation_summary.json`及r/step300、r/step400内逐task/suite、相邻、旧off和train96比较。
- 正式R/C/S输出：`runs/outputs/video_consumption_{r,c,s}_seed7_20260911/`；原off与独立continuation输出全部保留。
- 最近/data1 quota：used802241268KiB/soft1073741824KiB，共享83TiB；新95-task初始峰值预计40GiB，实际launch前刷新。71项source52.71GB已存在，不复制资产；最长source344帧低于已有target457帧。
- 当前没有95-task GPU任务。按live两节点、完整梯度吞吐与峰值选择最多6卡，不能恢复历史GPU快照为保留权。

## 保留的机制边界

最强off7 macro400为126/400，另初始化119；目标未达。其冻结train24九臂（correct/other/同suite错/跨suite错/乱序/倒序/首/中/末）为53/60/59/59/55/52/58/53/53，各96，没有建立稳定动态增量。既有错误/乱序结果只作历史诊断，不能冒充新最终因果资格。

此前all及最强模型的查询、中介、真实有限更新和行为诊断证明共享映射漂移及P/C/D交互存在，但没有把某个模块或几何统计识别为唯一根因。v5.2普通FM曾有视频依赖正证据；历史meta73弱图、head共享、纯语言route与多视频实验的负结果各有边界，详见findings和research_history。不因本轮负结果新增aux、RL或恢复未采纳路线。

## 交付状态

前轮实现378项相关测试覆盖通过；正式checkpoint与完整配对证据已核实。当前source仍为前轮实现，新95-task的实现/验证结果尚待产生。状态、设计登记和结果均在main推送；本轮旧详细运行记录可由251e2eb6及更早Git追溯，当前执行只看本文件与task_plan。
