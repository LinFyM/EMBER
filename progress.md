# EMBER progress

## 当前状态（2026-09-15，Owner授权回到v5.2）

Owner在完整分析后要求“你计划下接下来怎么做，然后就启动吧”。
Active design为[v5.2恢复与保持对照](docs/v52_return_plan.md)，顺序为固定旧step900复核、现行合同fresh基线、有条件的单变量任务共现对照。
阶段A已完成，B已完成300/600/900闭环并续训最后的1200节点，C已完成300闭环并续训600；owner再次指出不能在“启动后”停止，本计划按授权连续执行。
原529da6b与兼容operator 1ce99a0c均已封存为clean pushed detached树。原checkpoint／训练合同、执行配置、RNG和400条件映射检查通过。
最长8个验证视频（63–69帧）两次真实生成通过，第二次16.458秒、0.486 LoRA/s，reserved峰值12,834,570,240bytes。
固定step900 correct400为125/400，原132；S/O/G/L15/58/41/11、breadth6，R/G/L111/14/21、churn35、Jaccard .7603。
任务配对差额95%CI[-4.25,+.75]pp；12个workers均exit0，原400行完整，评测墙钟1142.301秒。
旧汇总器将GPU3误认NUMA0导致外层exit1；实际PCI/NUMA/affinity核验后只修正此判断，原400行汇总成功，未重跑模型或闭环。
原件与修正脚本见runs/analysis/v52_return_20260915/{launch_contract.json,reference_readout.json,aggregate_reference.py}。
启动前strg01 data1约848.06GiB／soft1024GiB，本阶段追加预算4GiB，全部大资产复用。
B精确learned H-read／双视角／offset1／46+4分池／旧LR时间轴已写入active design，模型和事件采样已集成。
模型移植90c58154，确定性事件20330e64；主入口沿用train_writer/materialize_writer，旧Pullback专属路径与临时A入口已退役。
192项相关CPU检查通过，覆盖模型／Meta重放、官方FM、更新／resume、bank／pairing、workers与信息墙。
真实source最长105帧双相机视频的chunk4／8各三次完整条件更新均通过；第三次Writer及三Meta信用非零，source冻结。
选chunk8／policy microbatch8：后两次平均25.381秒／条件，reserved峰值29.004GiB，快于chunk4的27.939秒。
该profile只检验信用、吞吐与显存，不保留其初始化；不构成B的闭环能力证据。
完整metadata计划4800事件／100800queries、每task200次，同episode和held46–49训练暴露均0。
B的fresh300已从clean pushed detached `7f9f11a3`正常完成，gpu01物理0/1/2/3，四rank按实际PCI/NUMA绑定并SUM。
用时4789.652秒，1200条件／25200 queries，每task50次；第三步后298次更新的Writer与三Meta梯度均finite非零。
100/200/300完整checkpoint及0/300独立动作诊断已保存，最高reserved29.441GiB。
B300 validation97/400，source47；breadth6/8，S/O/G/L1/52/37/7，R/G/L39/58/8、churn66、Jaccard .3714。
Train35/96，source17；breadth16/24，S/O/G/L8/14/10/3，R/G/L14/21/3、churn24、Jaccard .3684。
两面板全部496行完成，21 workers exit0；task-bootstrap差额95%CI为validation[0,+30.25]pp、train[+7.2917,+31.25]pp。
Validation逐task（global1/3/11/13/23/26/31/32）为0/1/39/13/0/37/4/3；train逐task及全部配对原件见baseline/paired_readout.json。
独立动作FM .150990→.114290，24/24task下降；能力解释使用上述闭环。
B600为validation105/400、breadth6，S/O/G/L0/59/39/7；source配对R/G/L39/66/8、churn74，差额95%CI[+.25,+33]pp。
Train49/96、breadth19，S/O/G/L11/15/15/8；source配对13/36/4、churn40，差额95%CI[+18.75,+48.9583]pp。
300→600的validation R/G/L76/29/21、churn50、Jaccard .6032，差额95%CI[-.25,+5.25]pp；
train为27/22/8、churn30、Jaccard .4737，差额95%CI[+4.1667,+26.0417]pp。训练任务获取继续扩大，验证迁移增量尚不稳健。
600累计2400条件／50400 queries、每task100次，六份完整checkpoint；598次identity后更新的Writer与三Meta信用均finite非零。
本段4505.649秒，累积最高reserved35.490GiB；新节点24 workers全部exit0，496条配对审计通过，独立动作FM .106714。
600验证逐task为0/0/40/19/1/38/3/4；完整train逐task及原始配对见baseline/paired_readout.json。当前没有>145或相邻稳定资格。
B900训练段4390.034秒、exit0；累计3600条件／75600 queries，九份完整checkpoint，898次identity后更新的Writer与三Meta信用均finite非零。
同checkpoint的496条件已物化并sealed，两个面板全部完成、24 workers exit0，墙钟986.440／298.469秒，完整配对审计通过。
B900 validation77/400、breadth6，S/O/G/L2/45/17/13；source配对R/G/L20/57/27、churn84，差额95%CI[-13,+29]pp。
Train50/96、breadth19，S/O/G/L17/15/10/8；source配对13/37/4、churn41，差额95%CI[+20.8333,+46.875]pp。
600→900的validation R/G/L55/22/50、churn72、Jaccard .4331，差额95%CI[-19.5,+3]pp；
train为34/16/15、churn31、Jaccard .5231，差额95%CI[-12.5,+14.5833]pp。训练诊断总分近似保持也伴随能力交换，验证收益明显回落。
900验证逐task为0/2/39/6/0/17/9/4；完整train逐task及配对在baseline/paired_readout.json，独立动作FM .105490。
命令及PID、quota/GPU快照在`runs/analysis/v52_return_20260915/baseline/training_launch_contract.json`。
启动时data1用量890,964,648KiB，soft1,073,741,824KiB，B/C追加峰值预算32GiB；源模型及数据均复用。
全部四组validation400／train96固定请求已准备，上一节点面板完成后才继续下一段，剩余1200的学习预算不变。
C唯一六组候选已在B闭环分数前仅依据train24规范登记；实际重建验证4800事件／100800 queries与B逐项相同。
B300的train获取满足C前提。B从完整900 checkpoint精确续训1200，gpu01原四卡与world4不变，launcher PID1996342。
同一模型的C从clean pushed detached `56bf1cc0`在gpu02物理0/1两卡完成fresh300，并从完整checkpoint精确续训600，launcher PID2914268。
当前空闲资源对应两节点合计6卡上限，故C从原准备三卡改为两卡；每更新仍为4task等权SUM，C后续resume锁定两rank拓扑。
复查发现此前B300两面板并行评测合计用了7卡，漏计了已有的跨节点6卡上限；该调度偏差保留在evaluation_launch_contract中，
后续训练、物化和评测统一按B最多4卡＋C最多2卡核验，原完整配对结果不重跑。
B/C各自保持完整节点证据与原有界预算；C命令、quota/GPU和实际运行记录见cooccurrence/training_launch_contract.json。
当前主树配置为C的explicit grouping，B继续使用7f9f11a3冻结配置。
C300训练7442.261秒，1200条件／25200 queries、每task50次，峰值reserved30.650GiB；全部298次identity后梯度组有效。
C300 validation91/400、breadth5，S/O/G/L4/55/32/0；source配对R/G/L33/58/14、churn72，差额95%CI[-3.75,+28.5]pp。
Train28/96、breadth14，S/O/G/L7/7/10/4；source配对10/18/7、churn25，差额95%CI[+2.0833,+21.875]pp。
全部496行及12 workers正常完成，两个面板墙钟1713.211／530.190秒，source／normalization／完整LoRA及配对审计通过。
C300验证逐task为1/3/34/21/0/32/0/0；全部train逐task与原始配对见cooccurrence/paired_readout.json。
B300→C300的validation R/G/L69/22/28、churn50、Jaccard .5798，差额95%CI[-7,+4.75]pp；
train为19/9/16、churn25、Jaccard .4318，差额95%CI[-19.7917,+5.2083]pp。当前没有共现分组带来改善的证据，继续预注册相邻节点检验保持。
实际前300更新的1200个条件事件和全局LR时间轴逐项一致；跨臂完整证据为grouping_comparison.json，不使用checkpoint union选模型。
历史复核保留单agentview和horizon mean；新方法仍须双相机、完整50-horizon learned read和严格跨episode。
A仅使用fixed step900 correct400；B开放预注册四节点correct400及train96，未开放最终视频controls或Test。
结果在对话中交付，不新建用户报告。

Process Pullback的原有原因诊断、实现、fresh900和全部冻结读出已经完成并关闭，未通过能力资格，没有selected合格checkpoint。
其完整结果见下方已关闭记录、findings§102及research_history；旧controls／Test不反哺当前计划。

## 已关闭窗口：Process Pullback共享出口

## 本轮修改与正式训练

依据共同teacher16的train-only原输出／free_q／free_AB闭环20／18／46（均/96），
在原G P左右加入共享identity起步的L/R乘法变换，其余过程读取、7维q、完整50-horizon、数据和纯FM保持。
该诊断支持优先修正出口参数化／优化约束，不证明PCA单一根因或共享L/R必然足够；特权拟合不作初始化。

实现e1ca29b9、v2 bank合同6dffa85c、冻结Test准入6e3f9953均已集成推送。
相关直接autograd、bank／controls／Test准入检查及真实最长105帧视频两次联合更新／部署profile通过。
第二次更新24.283秒、部署7.850秒；正式新训练与全部新评测统一来自clean pushed detached 85ecfd18。
Fresh900实际3,600条件／230,400queries、623/624合法teacher条件；九份完整checkpoint保留。
三段训练墙钟15,663.382秒（4.351小时）、最大reserved43.809GiB；899次identity后更新的Writer及两Meta信用均finite非零，source冻结。
独立训练动作FM .153279524→.140811848，22/24task改善；不以这些内部数值替代实际能力。

| 节点 | Train /96 | Train breadth /24 | Validation /400 | Validation breadth /8 | Validation S/O/G/L |
| --- | ---: | ---: | ---: | ---: | --- |
| Source | 17 | 7 | 47 | 3 | 0/5/41/1 |
| 300 | 21 | 9 | 50 | 4 | 0/16/32/2 |
| 600 | 20 | 8 | 75 | 3 | 0/38/34/3 |
| 900 | 18 | 11 | 79 | 3 | 0/40/39/0 |

Validation相邻R/G/L39/36/11、63/16/12，churn47／28、Jaccard .4535／.6923；
train相邻13/7/8、12/6/8，churn15／14、Jaccard .4643／.4615。
900的77/79次成功来自两项奶油奶酪搬运，余下2次为Goal3；Spatial和Long均零。
旧固定出口900→新900为validation64→79、train26→18；两配对task区间均跨零。
验证局部收益有所保持，广泛获取、组合能力与迁移保持未建立；本阶段未额外强制>145性能线。

## 冻结终点完整读出

Correct／same-task-other／wrong／shuffled／reversed／no-video为79／81／53／70／78／47（均/400）。
Other／wrong／shuffled／reversed相对correct的task-bootstrap95%CI分别为
[-1.75,+2.75]／[-22,+2.75]／[-9.5,+2.25]／[-2.25,+1.5]pp，全部包含零。
Other R/G/L67/14/12、churn26、Jaccard .7204；正确视频的局部内容优势主要来自Object1的40对wrong11。
倒序78接近correct79，尚无有益时间方向证据。乱序／倒序均重排真实frames后完整重新编译。
No-video的8套完整零LoRA身份及400个配对初态已核验，按执行等价合同复用source47；没有新增该轮400条rollouts。

固定Test source400／correct400为86／49，breadth5／6，S/O/G/L分别19/0/45/22与5/6/27/11。
R/G/L34/15/52、churn67、Jaccard .3366；差额−9.25pp，task-bootstrap95%CI[-20.25,+.75]pp。
Object0获得6次成功、Goal7获得1次，保留这些局部正例；Spatial、Goal、Long的已有能力损失更大。
Test在方法冻结且全部视频controls完成后才准入，未改变既有non-pass裁决或产生后继设计。

六个节点面板1,488条＋四个视频臂1,600条＋Test800条，共3,888条新正式rollouts、126个workers全部正常结束。
完整checkpoint、source／normalization、400-row视频臂每task全50视频无放回、task／state／env／policy RNG、freeze及no-video身份审计通过。
各面板墙钟累计10,522.596秒，包含跨节点并行面板，不能与训练墙钟直接相加视为端到端耗时。
全部原件在runs/analysis/process_pullback_learned_outlet_20260915，final_launch_contract状态complete；
完整数值在paired_readout、paired_endpoint_revision_comparison和final_paired_readout。

## 原诊断与证据边界

旧固定出口900、2,272条新增原因诊断及96条行为回放已完成，详见findings§100–101。
训练池teacher16–19的300/600/900为18/20/22，独立teacher46–49为24/22/26，不支持只归因换视频过拟合。
回放95/96复现原成败，失败涉及对象／实例、获取、放置和组合子目标，未唯一定位模型模块。
旧900 controls为64/65/48/35/67/47，只解释旧模型，未参与本轮修订。其other外层收尾事故已按原始400 rows保留，未重跑挑结果。
本轮首次注册状态标签错误在创建runtime前退出，修正后fresh启动；失败日志独立保留，不混入正式更新。

新研究额外峰值预算32GiB；Test启动前strg01 data1用量886,996,136KiB、soft1,073,741,824KiB，
研究15,003,024,283bytes，共享data1约83TiB可用，预估峰值在独立quota内。大资产全部复用。
正式证据、唯一checkpoint和冻结运行树保留，诊断专属代码已由Git保存并退役。
限定结论见[findings§102](findings.md#102-共享lr出口扩大了局部验证收益尚未建立广泛能力2026-09-15)，历史见[research_history](docs/research_history.md)。
