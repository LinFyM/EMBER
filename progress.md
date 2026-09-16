# EMBER progress

## 当前状态（2026-09-16，source时间对齐goal）

Owner最新单独授权“先把新的A900测试一下视频特异性，然后告诉我结果”。本次仅固定已完成的A900，
复用correct400，补same-task-other、cross-suite-wrong、no-video、shuffled、reversed各400严格配对行。
后两臂最后评测，真实RGB重排后完整生成；no-video为零LoRA／裸source干预，不是language-only Writer。
这是未获性能资格checkpoint的封闭原因诊断，不选点、不训练、不反哺架构；A1200、新B和SFT评测／续训仍暂停。
执行与原件集中在`runs/analysis/source_alignment_20260915/A/video_specificity_step900`。
五个对照已全部完成：correct140、other136、wrong116、no-video48、shuffled128、reversed139，均为400条。
Other相对correct的R/G/L117/19/23、churn42、Jaccard .7358；任务bootstrap差值95%CI[-4,+1.5]pp。
Wrong为92/24/48、churn72，差值CI[-12,-.5]pp；相对other少20条，差值CI[-10.5,-.25]pp，支持有限内容特异性。
Shuffled／reversed相对correct为100/28/40与104/35/36，churn68／71；差值CI[-8.25,+2.25]／[-5.25,+4.5]pp。
倒序总分接近但成功集合仍有交换，不能说输出不受顺序影响；当前未建立正确时序的稳定优势或动态必要性。
No-video为完整零LoRA，48与裸source50接近；它不能替代language-only Writer基线或证明视频条件增量。
全部2000条新rollouts、48个最终workers及五个评测launcher成功；单checkpoint、source／normalization、
全50视频无放回、初态／RNG与真实RGB重排完整forward审计通过。两次无rollout的GPU准入拒绝原件保留。
完整逐task／suite、breadth、R/G/L、churn、Jaccard与区间见该目录`readout.json`／`readout.md`及findings§109。
本次诊断已结束并停止，A1200、新B、SFT评测／续训和Test仍不启动；原SFT后台至400的授权不变。
Owner再次明确不需要持续盯看：既定对照按顺序后台完成，只在阶段结束读结果，不做逐分钟状态／分数轮询或汇报。

Owner于2026-09-16最新要求A900完整验证结果出来后立即暂停讨论；Goal已由owner暂停，不启动A1200或新B。
Owner明确保留在运行的SFT后台训练，到400步自动停止；不启动其评测或425／450续训。旧基线／必要B计划保留待讨论，不自动恢复。
最新暂停边界优先于旧goal和下方执行年表。当前active design为[Source时间对齐与v5.2复核](docs/source_alignment_v52_plan.md)。
A900已在原四rank拓扑正常完成，累计3600条件／75600queries；本段3285.437秒、累计10100.668秒。
700／800／900完整恢复点检查通过，898次有效阶段更新的Writer及三Meta梯度均finite非零，source冻结。
训练动作诊断FM .103778，不作闭环替代；496条件物化及correct400／train96完整完成，12个workers与两个launcher均exit0，严格配对审计通过。
A900 validation140/400、train54/96，breadth6／19，S/O/G/L为17/60/30/33与16/18/13/7。
相对source R/G/L为36/104/14与13/41/0，churn118／41、Jaccard .2338／.2407；任务bootstrap差值95%CI[+1.5,+46.2563]／[+30.2083,+55.2083]pp。
600→900 validation保留68／新增72／丢失20，churn92、Jaccard .425，差值95%CI[0,+28.75]pp；train为39/15/8、churn23、Jaccard .6290。
Validation按global1/3/11/13/23/26/31/32为1/16/44/16/0/30/33/0；四suite均非零，但两task仍零且相邻曲线99→88→140，未通过>145或稳定资格。
两个面板墙钟1147.038／868.728秒；A训练／物化／评测进程已全部退出，当前停在完整900恢复点，没有启动1200或B。
完整逐任务与相邻原件见`A/paired_readout.json`，历史900参照及其视频配对限制见`A/node900_reference_comparisons.json`和findings§107–108；后续封闭视频诊断见§109。
SFT继续占gpu02原两卡；暂停交付时已完成59步、33984queries，loss／梯度finite，实际命令明确`--stop-after-step 400`。
共享SFT的offset1／历史rank128／global576／450步入口已整合，独立开发树已退休；fresh formal已从`16c81e29`在gpu02物理1／3启动，launcher3861019。
首步global576、loss .149509、梯度 .017673，formal合同、24task曝光与无held-action读取检查通过；原登记的400／425／450评测当前暂停，仅训练至400。
51项CPU检查通过（122.83秒），含真实三rank不等分片累积与随机checkpoint恢复；450步／259200条历史逻辑采样及900个实际LR值复核一致。
保留新的单一训练路径，移除在线held-action监控接线；旧诊断入口也拒绝此配置读取validation actions。
SFT的两卡micro64／累积5真实三步及step1→3完整恢复均exit0，梯度finite非零、source冻结；LoRA原生dtype与旧SFT一致。
后两步平均74.596秒、reserved峰值32.934GiB，450步纯训练估计9.324小时；formal配置已据此封存，正式启动不复用profile参数。
SFT使用独立冻结运行树；旧Writer冻结树保持`575c189a`，两路径的源模型引用均为新raw1000。
SFT正式root为`runs/outputs/pi05_source_sft_aligned_rank128_dev_r2_b64_seed7_s450_20260916`；命令与资源在study的`SFT/training_launch_contract.json`。
启动前data1 used961711756KiB、共享83TiB可用，SFT追加峰值3GiB与剩余Writer22GiB均在独立quota内；两节点合计6张有效GPU。
Owner询问实现是否仍有未发现错误；定向复查训练—部署索引、读取模式、LoRA注入与缩放、梯度权重和完整恢复合同，未发现新合同违例。
冻结Writer运行树的22项原生FM直接梯度、future-control、Meta重放、批量LoRA与模式／恢复相关检查新近全部通过（21.87秒）。
证据在study的`implementation_review.json`及CPU日志；此结论不证明绝对无错，SFT的真实更新与完整恢复证据见上文profile。
原C保持关闭；新source已从clean pushed detached `b8ea00e9`在gpu01物理0／1／2／3完成1000更新、256000 queries。
全部1000行计数连续、global256／micro8×accum8一致，loss／gradient／LR均finite；终点loss .0835784、gradient .165120。
训练至最后更新29290.086秒，通常约9.1 query/s；该墙钟不含最后失败的保存与元数据恢复。
最终保存遇共享存储长时间I/O等待，三个等待rank于600秒NCCL默认期限超时，原launcher exit1已保留，未误记为正常退出。
终点raw policy／EMA、805份step1000 optimizer state和四rank RNG均已完整落盘并通过格式、游标检查。
只将750恢复点的原scheduler沿已记录的250个LR转换推进至1000，补trainer／manifest／summary并原子发布；
未重新更新模型、改写任何已落盘权重／optimizer／RNG或改变采样。恢复脚本及说明在study的`recover_source_step1000.py`与`source_checkpoint_recovery.json`。
固定raw1000正式source检查通过，11个manifest文件和大小匹配，完整点31.513GiB；原生加载及三次更新验证后750已退休，250／500／750均保留retention记录。
后续source入口显式使用一小时NCCL collective timeout以容纳大型保存；11项相关CPU检查通过，原训练冻结树不改动。
strg01最新data1 used955362116KiB，soft余量112.90GiB，仅保留完整1000；Writer剩余预算充足。
官方source validation400／train96全部完成，15个workers均exit0；墙钟1017.512／925.201秒，两组严格配对检查通过。
Validation旧47→新50/400，breadth3→3，S/O/G/L为0/1/42/7；R/G/L36/14/11、churn25、Jaccard .5902，task-bootstrap差值95%CI[-2.75,+4.75]pp。
Train旧17→新13/96，breadth7→8，S/O/G/L为2/1/6/4；R/G/L8/5/9、churn14、Jaccard .3636，差值95%CI[-13.5417,+5.2083]pp。
新validation逐task（global1/3/11/13/23/26/31/32）为0/0/1/0/0/42/7/0；train逐task按固定24task顺序为1/1/0/0/0/0/0/0/0/0/0/1/0/3/0/0/3/0/2/1/0/0/1/0。
裸source尚无稳健净改善；Long局部获取与其它suite损失均保留，不能由裸policy接近判断其对Writer整条链影响小。
Train首个launcher在准备阶段因漏传显式train24诊断面板被拒，未产生rollouts；补传旧source同一注册面板后准备通过，原exit1和日志保留。
Validation Python evaluator及400行聚合成功；外层Bash因其运行时launcher文本被上述修补改变，完成后报EOF／exit2。原exit2保留，不重跑完整rollouts；后续运行期间不改共用launcher。
A／B均在正确source上通过最长105帧视频三次完整联合更新及部署，两个profile均exit0；第三步Writer和Text／VL／Action三Meta信用finite非零，source冻结。
采用共同frame chunk8／policy microbatch8；后两次A平均17.673秒、B25.414秒，reserved峰值21.148／29.004GiB，部署3.905／6.052秒。
两配置已登记实测profile；均丢弃profile参数，正式A／B继续使用相同fresh初始化流与已注册1200更新窗口。
两节点snapshot确认gpu02物理0／1现有进程仅186／148MiB且0%利用率，按共驻合同使用；本批合计6张有效GPU。
Source root为`runs/outputs/pi05_source_aligned_seed7_1k_20260915`；命令、双节点GPU／quota和日志均在`runs/analysis/source_alignment_20260915`。
两组source原件及逐task／suite配对在study的`source/{validation,train}`及对应`*_comparison.json`。
Fresh A300从clean pushed detached `575c189a`在gpu01物理0–3正常完成，训练3422.504秒、1200条件／25200queries。
100／200／300完整checkpoint与0／300独立动作诊断保存；step3起298次更新四组梯度全部finite非零，source冻结，reserved峰值21.701GiB。
共同global84、baseline分组和frame chunk8／microbatch8保持；参数、optimizer、scheduler和采样／RNG均fresh。
完整correct400＋train96为99/400与36/96；breadth5／18，S/O/G/L分别10/38/31/20与11/11/10/4。
相对新source的R/G/L为30/69/20与9/27/4，churn89／31，Jaccard .2521／.2250；差值95%CI[-3,+31.25]／[+12.5,+35.4167]pp。
Validation逐task为0/10/35/3/0/31/20/0；train按固定24task顺序为2/4/1/1/1/2/1/3/1/2/3/1/0/4/3/0/3/0/2/1/1/0/0/0。
两组496条完整、18个最终workers均exit0，全部source／checkpoint／视频／初态／RNG与信息墙检查通过；墙钟1009.883／560.788秒。
Validation首次完成准备后被实时GPU准入拒绝，未启动workers或产生rows；复查双节点后从同一队列start成功，原exit1／日志保留。
已见训练获取及四suite局部收益，尚无>145资格或相邻保持证据，不能由此提前判断source的下游效应。
本节点完整证据在`runs/analysis/source_alignment_20260915/A/paired_readout.json`及训练／评测launch contracts。
两面板完成后，A600已从300点在原四rank／GPU UUID拓扑精确续训并正常exit0；本段3392.728秒，累计2400条件／50400queries。
400／500／600完整checkpoint检查通过；step3起598次更新四组梯度全部finite非零，source冻结，reserved峰值22.854GiB。
独立train held动作诊断FM为.105642；此loss不代替闭环。累计训练6815.231秒，证据在`A/training_audit_600.json`。
600点496条件已全部物化并sealed，correct400＋train96完整完成，18个workers及两个launcher均exit0；配对与信息墙审计通过。
Validation88/400、breadth6，S/O/G/L13/35/36/4；source配对R/G/L32/56/18、churn74、Jaccard .3019，差额95%CI[-2.75,+23.75]pp。
Train47/96、breadth17，S/O/G/L11/19/13/4；source配对R/G/L12/35/1、churn36、Jaccard .25，差额95%CI[+21.875,+50]pp。
300→600 validation为R/G/L57/31/42、churn73、Jaccard .4385，差额95%CI[-13,+5.75]pp；train为29/18/7、churn25、Jaccard .5370。
验证逐task为0/13/26/9/1/35/4/0；train为3/3/1/1/3/0/4/4/1/3/4/3/2/4/3/0/4/0/2/2/0/0/0/0，顺序同上。
训练获取继续扩大，验证净下降11且Long20→4，仍有明显能力交换；没有>145或相邻保持资格，不由此改变既定窗口。
两个面板墙钟1120.709／667.014秒，完整证据已进入`A/paired_readout.json`；Writer冻结树仍为`575c189a`。
A900已从600在相同gpu01四rank／GPU UUID精确续训，launcher962964；live quota used961648756KiB，剩余Writer22GiB及SFT3GiB预算充足。
SFT两卡micro64／累积5的真实首步完成exit0：global576、loss .149509、梯度 .017673、76.263秒，reserved32.934GiB。
完整step1保存后已在相同gpu02物理1／3精确恢复至profile step3并正常完成，launcher3771217；完整恢复点、CPU与profile证据保留。
已核对旧source：71tasks、全参数SFT、1000 updates、global256、warmup333／LR5e-5、原AdamW／BF16／normalization；
后续固定raw step1000，EMA按原decay维护。本轮只修未来动作标签及必要采样支持，物理执行由profile确定。
原8×A100峰值约71GB不能直接搬到A40；已依据实际更新与恢复profile采用microbatch／累积及rank0 CPU EMA。
主agent拥有source与整合；独立codex/v52-aligned-model实现A单视角／mean和B双视角／learned，保留共有初始化流。
2026-09-16：模型与模式plumbing已整合为aa1a7b65／02160554；B恢复baseline分组，A／B共享同一实现，
config、训练、checkpoint检查和物化均识别camera／H-read。显式新source引用与eval authority已接入。
source入口恢复，接offset1、rank0 CPU EMA、逐张量AdamW、DDP gradient bucket view及deferred NCCL。
原逻辑8×32采样已与历史实现比较1000步／256000个queries，完全一致；物理打包与resume合同检查通过。
整合后的249项相关CPU检查通过（70.26秒），另保留agent的181项独立检查；尚未以这些检查代替GPU更新。
generic固定7de663972b7817d2c4cf2d84c821153dfea772e9已补齐；旧source容量smoke仅定资源，正式fresh参数不复用它。
旧source容量smoke在gpu01物理0完成：micro2三步正常、reserved33.64GiB；micro8／累积4三步正常，
后两步各32queries平均16.732秒、约1.913 query/s，reserved37.94GiB。两次均全参数可训练、rank0 CPU EMA、无checkpoint保留。
已补不整除world的query等权分配，三卡可按86／85／85维持global256；相关source／sampler／eval的37项检查通过。
source恢复合同包含GPU UUID；实际多卡、generic初始化与完整checkpoint resume均已核验。
Owner提醒gpu01有低占用卡后，复核并采用物理0–3；其中1–3随后全部空闲，不等待固定数量或排除可共驻卡。
四卡generic／global256首步完成，loss .302815、梯度finite、7.675 query/s；第二步CUDA OOM，未进入正式训练。
保留DDP bucket views原位清零后，同一四卡／micro8／accum8真实三步完成，exit0；后两步平均30.701秒、
8.339 query/s、reserved37.940GiB。完整恢复需在首个真实microbatch建立views，再按原全局权重累积。
按实测纯更新约8.53小时／1000步，另计初始化与写盘；11项source CPU检查通过。
完整checkpoint在第1步保存后，已恢复完成第2／3步并再次完整保存，两个launcher均exit0；
policy／EMA／optimizer／scheduler、四rank RNG与sampler cursor齐全，global examples为256／512／768。
完整点实测31.513GiB，恢复窗口reserved峰值39.170GiB；与先前连续窗口loss最大差2.70e-5，接受正常reduction顺序差异。
已保留小型profile／resume证据并删除31.5GiB临时权重；正式配置锁定gpu01四rank、micro8／accum8、global256，
每250步保存完整恢复点并保留最新一份，固定raw1000。记录见`runs/analysis/source_alignment_20260915/source_launch_contract.json`。
A配置`configs/pi05_writer_aligned_A.json`已单独登记；与B仅模型读取模式及相应说明不同，数据／optimizer／runtime／source相同。
失败记录保留在`.codex/tmp/source-alignment-profile/generic_full3_ddp4`，不作为训练或闭环证据。
main起点b56f0a9c；实现、配置和检查已集成推送，开发worktree已清理，source冻结运行树保留供exact-resume。
首次strg01 data1 used908275108KiB／soft1073741824KiB，项目834GiB；formal前used922315952KiB、soft余量144.41GiB，共享83TiB可用。
S+A+B追加峰值预算调整为112GiB，包含补回generic约13.5GiB；formal前再次实测quota及恢复点峰值。
旧source、A/B/C全部checkpoint与formal evidence保留；未完成的新训练／评测和后续不预写结果。

## 已关闭窗口：v5.2旧source恢复与任务共现（2026-09-15）

Owner最新明确要求“别继续C了，你停一下”。C已停止，后续训练、物化和评测不再启动。
本轮按[v5.2恢复与保持对照](docs/v52_return_plan.md)完成了阶段A、B，以及C的300／600完整闭环；C剩余计划现处于停止状态，不自动恢复。
C900段在最后已写更新611后按指令终止；最新完整checkpoint仍为600，100–600六份完整checkpoint、部分900日志及原有结果保留。
经核实的C launcher、torchrun和两个rank进程均已退出，GPU02不再有C GPU进程；exit1来自本次主动终止，不记为工程故障或完整窗口科学non-pass。
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
B1200终点为validation85/400、breadth5，S/O/G/L0/43/37/5；source配对R/G/L37/48/10、churn58，差额95%CI[-2,+28.25]pp。
Train56/96、breadth21，S/O/G/L15/20/13/8；source配对14/42/3、churn45，差额95%CI[+26.0417,+55.2083]pp。
900→1200的validation R/G/L56/29/21、churn50、Jaccard .5283，差额95%CI[-5.5,+13.25]pp；
train为38/18/12、churn30、Jaccard .5588，差额95%CI[-5.2083,+17.7083]pp。训练获取与breadth扩大，验证交换、迁移不足仍在。
1200验证逐task为0/0/41/2/1/36/5/0；全部train逐task及原始配对见baseline/paired_readout.json。
B四节点八面板共1,984条完整rows、93 workers全exit0；终点两个面板墙钟986.012／293.699秒，完整信息墙与配对审计通过。
训练累计1,200更新、4,800条件／100,800 queries、每task200条件，覆盖全部1,104个合法teacher条件各4–5次。
12份完整checkpoint、1,198次identity后四组有效梯度保留；最后训练段4531.588秒，累计18216.923秒，峰值reserved35.490GiB。
独立动作FM终点.102588、24/24task相对fresh下降；闭环最高105和终点Spatial0均不满足正式资格，不开放controls／Test。
命令及PID、quota/GPU快照在`runs/analysis/v52_return_20260915/baseline/training_launch_contract.json`。
启动时data1用量890,964,648KiB，soft1,073,741,824KiB，B/C追加峰值预算32GiB；源模型及数据均复用。
B的四组validation400／train96固定请求全部完成，各段均在上一节点面板完成后才继续；1,200更新预算未扩大。
C唯一六组候选已在B闭环分数前仅依据train24规范登记；实际重建验证4800事件／100800 queries与B逐项相同。
B300的train获取满足C前提。B所有训练、物化和闭环已结束，其释放的gpu01四卡已用于C600评测。
同一模型的C从clean pushed detached `56bf1cc0`在gpu02物理0/1完成fresh300及精确续训600；各task累计100条件，全部598次identity后四组梯度有效。
C600训练段7087.029秒、exit0，累计训练14529.289秒；其后启动的900段launcher PID3548657已按owner指令停止。
当时空闲资源对应两节点合计6卡上限，故C从原准备三卡改为两卡；每更新仍为4task等权SUM，两次resume均保持原两rank拓扑。
复查发现此前B300两面板并行评测合计用了7卡，漏计了已有的跨节点6卡上限；该调度偏差保留在evaluation_launch_contract中，
此后并行阶段按B最多4卡＋C最多2卡核验，原完整配对结果不重跑；B结束后，C600独立评测按fresh快照使用gpu01四卡，训练resume保持原world2。
B/C各自保持完整节点证据与原有界预算；C命令、quota/GPU和实际运行记录见cooccurrence/training_launch_contract.json。
当前主树配置为C的explicit grouping，B完整证据来自7f9f11a3冻结配置。
C300训练7442.261秒，1200条件／25200 queries、每task50次，峰值reserved30.650GiB；全部298次identity后梯度组有效。
C300 validation91/400、breadth5，S/O/G/L4/55/32/0；source配对R/G/L33/58/14、churn72，差额95%CI[-3.75,+28.5]pp。
Train28/96、breadth14，S/O/G/L7/7/10/4；source配对10/18/7、churn25，差额95%CI[+2.0833,+21.875]pp。
全部496行及12 workers正常完成，两个面板墙钟1713.211／530.190秒，source／normalization／完整LoRA及配对审计通过。
C300验证逐task为1/3/34/21/0/32/0/0；全部train逐task与原始配对见cooccurrence/paired_readout.json。
B300→C300的validation R/G/L69/22/28、churn50、Jaccard .5798，差额95%CI[-7,+4.75]pp；
train为19/9/16、churn25、Jaccard .4318，差额95%CI[-19.7917,+5.2083]pp。该节点没有共现分组带来改善的证据。
C600 validation65/400、breadth5，S/O/G/L1/42/21/1；source配对R/G/L25/40/22、churn62，差额95%CI[-11.75,+21.5]pp。
Train43/96、breadth19，S/O/G/L12/14/10/7；source配对13/30/4、churn34，差额95%CI[+15.625,+39.5833]pp。
300→600的validation R/G/L48/17/43、churn60、Jaccard .4444，差额95%CI[-12.75,-1.25]pp；
train为20/23/8、churn31、Jaccard .3922，差额95%CI[+4.1667,+28.125]pp。C仍有训练任务获取，但早期验证收益保持变差。
C600验证逐task为0/1/30/12/0/21/1/0；全部train逐task及配对原件在cooccurrence/paired_readout.json。
496条件sealed后，两面板使用已释放的gpu01四卡，24 workers全exit0；墙钟972.617／304.098秒，完整source／normalization／LoRA与配对审计通过。
B600→C600的validation105→65，R/G/L51/14/54、churn68、Jaccard .4286，差额95%CI[-18.5,-2.75]pp；
train49→43，32/11/17、churn28、Jaccard .5333，差额95%CI[-17.7083,+5.2083]pp。
两臂各自300相对source新增的58个验证成功，到600时B保留42、C保留28；新增train成功则分别保留15/21与12/18。
这些有限面板未支持共现分组改善获取与保持；保留其相对source的训练获取，不由此唯一归因梯度冲突。
实际前600更新的2400个条件事件／50400 queries和全局LR时间轴逐项一致；跨臂完整证据为grouping_comparison.json，不使用checkpoint union选模型。
历史复核保留单agentview和horizon mean；新方法仍须双相机、完整50-horizon learned read和严格跨episode。
A仅使用fixed step900 correct400；B已完成预注册四节点correct400及train96，C仅完成300／600两节点，未开放最终视频controls或Test。
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
