# EMBER progress

## 当前快照（2026-09-11，95-task有限对照已登记）

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
