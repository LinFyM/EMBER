# EMBER progress

## 当前状态（2026-09-15，Owner授权回到v5.2）

Owner在完整分析后要求“你计划下接下来怎么做，然后就启动吧”。
Active design为[v5.2恢复与保持对照](docs/v52_return_plan.md)，顺序为固定旧step900复核、现行合同fresh基线、有条件的单变量任务共现对照。
阶段A已完成，继续推进B的实现、profile与正式节点训练；owner再次指出不能在“启动后”停止，本计划按授权连续执行。
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
B的fresh300已从clean pushed detached `7f9f11a3`启动，gpu01物理0/1/2/3，四rank按实际PCI/NUMA绑定并SUM。
入口已登记step0→300及100/200/300完整保存点；先完成独立动作step0诊断，再进行更新。仍无B闭环分数。
命令及PID、quota/GPU快照在`runs/analysis/v52_return_20260915/baseline/training_launch_contract.json`。
启动时data1用量890,964,648KiB，soft1,073,741,824KiB，B/C追加峰值预算32GiB；源模型及数据均复用。
其余600/900/1200分段和全部四组validation400／train96固定请求已准备，上一节点面板完成后才继续下一段。
C唯一六组候选已在B闭环分数前仅依据train24规范登记；实际重建验证4800事件／100800 queries与B逐项相同，尚未启动C。
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
