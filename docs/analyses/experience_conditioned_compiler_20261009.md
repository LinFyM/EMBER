# 经验条件完整rank128编译器：首批完整科学结果

2026-10-09。唯一预注册meta54候选完成correct strict paired400：**135/400（33.75%）**，同场景强MT为153/400、T2340为161/400。
这批没有达到Owner要求的完整固定LoRA明显超过强MT；它是有效科学负结果。全部训练、真实经历、最终权重、失败、费用和不利配对行保留。
实现完成、梯度接通或自身实践成功不构成性能资格；54更新是首个记录节点，批次交付不关闭EMBER。

## 1. 原件、身份和比较口径

唯一run root为`/data1/user/ymdai/ember_runs/experience_conditioned_compiler_20261009`（下文记root）。
原设计为[experience_conditioned_compiler_20261009](../designs/experience_conditioned_compiler_20261009.md)；原件入口：

- `training/run_contract.json`、`training/metrics.jsonl`、`training/completion.json`、完整macro155/182 checkpoint；
- `evaluation/meta27/results.json`、`evaluation/meta54/results.json`、`evaluation/formal/results.json`，各目录的合同、queue、condition原始rows/真实链/完整LoRA及worker记录；
- `registered_events.json`、`registered_panels.json`；`raw_evidence_verification.json`再次通过canonical聚合消费者核验全部原始sidecars、冻结条件、覆盖、state/video/env-policy RNG和正式scene配对；
- `scientific_analysis.json`及其可重复运行的`post_batch_scientific_analysis.py`；`train_panel_analysis.json`保留先前有限面板分析；
- 原失败`batch_execution.json`及logs、恢复`batch_attempts/formal_recovery_001/batch_execution.json`（complete、exit0）、`costs.jsonl`和`post_batch_cost_audit.json`。

共享学习来自clean pushed detached **76c136a3**，完整128warm＋54meta连续完成，唯一正式候选macro182不经小面板择优。
正式读取/评测来自 **517e0c20**：只修复validation metadata store局部SOURCE导入，并复用有效训练与面板接续正式队列。
原故障发生在teacher读取/实践之前；两个原failed job在原队列重试，最终400个独立condition各完成一次，没有重新fresh训练。
后续物理传输与adapter复用优化来自 **a692ec2b**，没有用于这400行；训练来源、正式执行来源和后续优化身份分别保留。

固定8个validation tasks，每task全部50条teacher videos各一次、50个canonical final init states各一次；K1、原state/video映射与强MT/T scene及RNG严格配对。
每condition的适应初态排除自己的final初态；共享参数冻结，只前向修订完整LoRA，final反馈不反哺。
teacher仅合法双相机视频和exact language，当前自身动作/状态/结果作为经验；无held离线标签梯度，Test封存。
源policy/native prefix冻结、physical PEFT identity、38-target rank128、alpha/r=1；MT只初始化可训练坐标并提供固定教学probe。

## 2. 完整能力、覆盖与交换

| suite（各100行） | MT | T2340 | 本批 | 对MT gain / loss |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 48 | 51 | 40 | 5 / 13 |
| Object | 46 | 50 | 43 | 10 / 13 |
| Goal | 35 | 36 | 33 | 5 / 7 |
| Long | 24 | 24 | 19 | 4 / 9 |
| 总计 | 153 | 161 | 135 | 24 / 42 |

| global task | 内容 | MT | T2340 | 本批 | 对MT gain / loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | cookie box上的黑碗放盘子 | 41 | 45 | 38 | 5 / 8 |
| 6 | cookie box旁的黑碗放盘子 | 7 | 6 | 2 | 0 / 5 |
| 11 | 奶酪放篮子 | 36 | 45 | 29 | 4 / 11 |
| 16 | 黄油放篮子 | 10 | 5 | 14 | 6 / 2 |
| 23 | 开上层抽屉并放碗 | 0 | 0 | 1 | 1 / 0 |
| 26 | 奶酪放碗 | 35 | 36 | 32 | 4 / 7 |
| 31 | 奶酪和黄油都放篮子 | 24 | 24 | 19 | 4 / 9 |
| 39 | 杯子放微波炉并关门 | 0 | 0 | 0 | 0 / 0 |

对MT保持111、增24、丢42，churn66/400（16.5%）、success-set Jaccard0.6271；对T保持105、增30、丢56，churn86/400（21.5%）、Jaccard0.5497。
六个有原能力的task中五个对MT净下降，仅task16净增4；四个suite均净下降。
breadth从MT/T的6增到7仅来自task23一行成功，不能写成已获得可靠困难任务覆盖；task39仍0/50。
task16的增益真实保留，但没有抵消原本较强task11等处的损失；不能只展示新成功或把交换说成普遍提升。

对MT差−4.5个百分点、对T差−6.5个百分点。若仅作condition独立假设下的描述性配对近似，95%区间分别为[−8.46,−0.54]、[−11.00,−2.00]个百分点；
这些是固定validation8面板的读数，不是跨task总体推断。只有一个formal checkpoint，正式相邻稳定性尚未建立。
既有MT/T无本批适应成本，绝对能力与实际额外费用可比较；本批没有等预算充分语言适应参照，不能称已完成公平效率或视频必要性比较。

## 3. 实践成功、重读与失败尾部

正式400条件实际初读400、重读714，全片等价读取**1114**（均值2.785，P50=2、P90=6，最大6）。
原生teacher编码14528帧，神经教学读取41812帧，原生编码累计735.0秒；缓存后的教学读取也计入，不能只报告编码一次。
J为事后事实：0/1/2/3/4/5次修订分别147/55/88/1/65/44个条件，没有固定J或J菜单。
实践共**252083环境steps（含settling）、936重置、0初态复用**，平均630.21steps、P50=731、P90/P95=1024。
最终评测另执行114129控制steps＋4000 settling＝118129环境steps；其结果不参与适应。

222条件在自身实践成功后停止，但其中final仅119/222成功（53.60%）；103个自身成功条件在新初态失败。
剩余178条件消耗完整1024预算，final16/178成功；它们使用182272steps（全部实践的72.31%）、618重读、796全片等价读取。
失败预算尾部不是隐藏开销，也不能因少重读或早停宣称学得更有效率。
task3/11/26各自全部50条件自身成功，final仍分别38/29/32；task6自身成功6条件的final全部失败。
task23/39共花101483实践steps、394全片读取，final只有1/100；当前经历与停止规则未兑现困难task恢复。

单condition适应wall平均61.76秒、P90=103.87、P99=108.13、最大109.45；含保存与final的condition wall平均95.33秒、P90=154.82、最大167.64。
累加工作时间分别24704.12/38131.96秒，不能当作六卡并行elapsed。正式整批parent elapsed6457.84秒（1.794小时），worker计10.6238GPUh。
J/自身成功是策略和难度共同产生的分组；重读多的条件较差不证明重读有害，也不能从自身成功与final关联推出最优停止。
formal只有end臂，不能将这些数值当作held Λend相对Λ0的修订因果效应。

## 4. 训练域与相邻读出没有显示净修订收益

固定8 train tasks×2 teacher conditions×3全新初态，16独立条件、每臂48行，不选checkpoint。
meta27：Λ0 **33/48**、Λend **30/48**，保持30、增0、丢3，breadth8→7。
meta54：Λ0/Λend/masked-null均**28/48**；end对initial和null均保持27、增1、丢1，churn2/48。
null是沿同一真实实践链与结束点屏蔽经验输入的replay；不是独立充分训练的无经历算法，也没有自己的实践/停止分布。
相邻最终meta27→54保持24、增4、丢6，churn10/48（20.83%）、Jaccard0.7059；initial相邻保持25、增3、丢8。
因此没有“先在训练域获取净修订能力、只在held迁移时丢失”的支持，且初始编译能力也出现下降。
不能从小面板相同总分推出所有经验输入无作用；一增一丢与有限条件数均保留。

两个面板真实适应分别7093/7626steps、29/31重置、全片读取32/33；meta54 masked-null缓存replay另读33次，没有重复实践。
学习实际216条meta链，固定36映射各6次；实践90307steps、206重读、378重置，初读216＋重读206＝422次全片读取。
自身实践成功172/216；两个独立SDE query共432 episodes、265成功，保留13778个真实去噪转移信用。
warm初始教学forward另512次；可微整链VJP replay另934次（warm512＋meta422），并非免费缓存读取。
不含disposable profiles的学习/面板/formal整片神经读取合计3080次；激活checkpoint内部算子重算计入GPUh，不作为新的独立观看轮数。

meta FM最终阶段减初始阶段均值为+3.95e−6；数据/时间/noise流在变，这不是固定收敛面板，也不提供控制保持保证。
β=4.84459e−6来自首个非零PG/FM共享梯度norm定标：FM1.01176、PG52210.87，初始βPG约为FM的0.25；
系数很小不能直接说RL被忽略，后续没有逐更新独立FM/PG norm序列可证明持续比例。
末点post-clip norm：decoder0.999990、reader0.002282、initializer0.003465、encoder4.916e−5、update_out0.000605、updater2.515e−5、gate3.907e−5。
这些证明实际梯度消费者存在；参数量/尺度不同，不能把norm比例直接当成因果贡献或修订无用的证明。

## 5. 具体机制、近似历史与下一判断

本批冻结prefix从合法teacher双RGB提取Φ[F,512,2048]及固定MT probe H[F,50,1024]。
真实实践保留前后RGB/proprio、实际最多5动作与结果、同次原生flow的两时点H；经验编码16slots和当前Q在空间8slot压缩前改变teacher query，
有序教学进入轴向初始化器/共享U，形成Q[38,128,256]；同一个384→512→64非线性坐标解码器直接组装38-target A/B，作用于源policy自身执行hidden。
合法跨episode FM及两个独立初态SDE结果余切，经完整A/B与全部Q阶段回传共享读取/经验/U/decoder；原始E/H停止环境梯度。
condition内没有optimizer、动作轨迹拟合或第二adapter；自身成功/预算结束后固定唯一完整LoRA退出适应。

这实际检验了“当前控制器的真实失败证据、压缩前重读与共享终局信用能教出有益参数修订”，比旧self-read（116/400）增加了真实环境后果，
比旧共享去噪回报学习（父161，末158/400）增加了condition经验、递推Q、完整可训练坐标以及FM/PG共同学习。
旧T已有正确/另一正确/public的161/150/103，不能否认已有合法视频条件作用；本批的额外机制尚未取得超过它的完整能力。
完整输出自由度、真实经历、非零功能梯度与更多修订的组合仍未给出训练域净收益，降低了“这些接口接上便会自然学成”的支持度。

一个具体竞争解释是：坐标e可直接改变共用控制起点，FM/真实信用允许通过共享解码器取得更新，却未迫使经验—教学关系承担有益修订。
identity初始化时输出不依赖Q；新输出列学起后该通路可获得梯度，实际reader/U的非零梯度排除永久断图，不能证明已学会有效使用。
另一个解释是初始共享能力漂移叠加未经学习的成功停止：实践成功只是某个适应初态的成功，不认证新初态控制。
warm/shared decoder贡献、经验吸收不足、停止分布、有限6条映射曝光与SDE/ODE信用差异尚未由本批区分；不据此指定唯一根因。

下一决策首先须解释**训练域无净修订收益、共用初始能力下降及高失败尾部**怎样在完整学习中得到改变。
本批不支持原样无限续训、固定更多重读、rank/LR/seed小扫或重跑成熟旧参照；也不证明整个经验编译族不可能。
若增加学习，应先给出预期学习速度/保持/修订效应的可失败预测；若改算子或监督，应改变经验到有益Q/LoRA更新的实际学习关系，保留强MT和完整能力比较。
主讨论消费全部证据后自主选择有依据的接续、匹配端点参照或主要干预；不再等待Owner例行许可，不把这份实验交付当研究终点。

## 6. 资源、执行效率和后台接续

实际闭合账本按物理node/GPU时间区间取并集为**20.518080GPUh**（含下述CUDA核验0.016929GPUh），最大并发6张；46行账本、23个区间全部闭合。
profile/失败/probe0.198921、连续128warm＋54meta8.168867、meta27/54面板0.642178/0.856412、原formal失败0.005023、formal恢复10.629750GPUh。
不能将重叠的rank、train内层区间、probe或worker receipt累计再次收费；正式results中的10.623823GPUh是较窄worker窗口，闭合外层账本为上述10.629750。
占卡wall并集6.326509小时；首次GPU记录至最后核验退出elapsed8.747508小时，含修复/等待；实施起至formal完整退出9.694107小时。
原Queue延迟造成失败到恢复启动1.985205小时空档，这段不是GPU计费，但必须体现在elapsed中。
首次module discovery失败及两个取消的gpu01原formal transport未生成GPU账本，不能声称它们的CUDA占用已测；保守计入全部已登记GPU transport窗口，费用上界20.590723GPUh。
费用原件为root的`post_batch_cost_audit.json`（前44行截止快照）和含最后核验的`scientific_cost_summary.json`，原failed启动和等待事实不抹掉。
现存run root allocated56.701GiB（logical56.678），strg01 data1账户used817.458GiB、soft quota2048GiB、余1230.542GiB。
过程序列存储峰没有连续测量，56.701GiB是已观测现存量；profile75.833GiB是预测，不冒称实际峰值。
实际费用和时间在80GPUh/24wall预算内；没有新增大资产副本或动历史原件，160GiB存储上界与现存量分开记录。
正式pool实际gpu01:0/6＋gpu02:0/2/4/6，共6张有用物理卡；最长/最短worker wall约6429/6311秒，队列负载尾差约118秒。
训练本批world2；代码支持1–4有用ranks，world3/4吞吐未实测，不能称2卡最优或把它固定为后继上限。
实际profile采用FM28、PG32、learned/native frame64、decoder65536、经验201；更大帧/解码块已测且未更快，记录在root实际profile。
Owner要求按有用吞吐使用可用显存与卡数；后继应按实际工作选择物理并行，并保持4condition逻辑更新/权重/流，而不是按旧world2限制资源。

a692执行优化省掉无消费者的Phi/H CPU拷贝，practice保留完整真实证据，final_many复用显存LoRA直到active batch改变。
18项针对性CPU检查通过；整批退出后在现场准入的gpu01:0上从clean frozen a692实测真实non-held query，ODE/SDE新旧动作差均0、十次flow/完整捕获消费者通过。
耗时约60.90秒、0环境steps、峰9.078GiB；legacy/practice/final每10步flow均约0.368–0.372秒，差异接近噪声，不声称整批提速比例。
该小检查是物理动作等价核验，不是训练microbatch选择或新科学评测；其成本全部记录。

Owner指出原Queue在未打开页面时卡住：原接受到实际处理6503.7秒。
canonical codex-session-messaging skill已修正为无设置覆盖地后台resume原thread，active默认Steer、idle直接turn/start，Queue只用于明确延后或恢复原条目。
隔离notLoaded且页面未打开的实际回复12.850秒；本次真实formal退出到接受0.434秒、到实际agent活动**2.715秒**，模型/推理/cwd/权限均保留。
唯一whole-batch事件等待者已经消费退出并结束；没有自Queue、阶段通知或第二持续等待者。
完整原回执、修复、7项消息测试、回滚脚本及真实本批验证均在root保留。一次整批主讨论回报以`main_delivery_receipt.json`的实际处理证据确认。

## 7. 主讨论后续消费：经验曝光与真实执行瓶颈

原`Runner.adapt`在success后先break，再进入U；因此成功经历不参与任何实际编辑。这属于原学习组织，不能冒称工程缺陷。
training metrics的216条meta链中134条J0、82条有修订；扣除其中12条masked，只有70条向U呈现未屏蔽的真实经验，覆盖24个映射。
三个18-update窗口里有修订的28/27/27条事件，配对end−initial FM均值约`1.58e-5/-8.67e-6/2.39e-5`，
相对绝对变化均值约`.00034/.00057/.00055`；同链query/noise配对，跨update不同，不能当作统一验证集上的收敛曲线。
结合train的无净收益，它降低“少量真实失败经验已经教会修订，只差held迁移”的支持。
同时，首批每映射只6条链，不能由小量未学会证明该族不可能；下一投入需要改变实际学习机会，而非只称训练不够。

独立效率原件位于`root/analysis/throughput_followup_20261009`，main读取了batch_summary、components、simulator、report和成本回执。
八个不同task/语言/实际归档LoRA/环境的同一cohort，从注册state32、seed7推进各40控制steps；B1/2/4/8再逆序各一次。
含reset/settling/scene/搬运的320控制steps平均wall为38.987/28.470/22.456/20.281秒，B8比B1快1.922倍。
每个B公平复用同组persistent环境；初始化单独计费，不能把这个B1当成原生产逐task重建环境的完整baseline。
实际Phi/H采集的20-step短段提高1.666倍；B16单replan可运行但未测持续窗口，world3/4也未测。

B1十步flow/prefix/env/reset为18.269/6.526/5.046/8.793秒，B8为2.337/4.334/5.021/8.383秒。
实际flow摊销接近8倍，整个系统只有约1.9倍：批量后prefix、环境推进和reset成为更大的份额。
这不是“显卡低UTL证明模型数学算子低效”，也不能把40-step短窗口的reset占比外推到完整1024预算。
完整非held150-decision/3次重读链的Compiler replay+VJP为1.331秒，四阶段FM VJP2.542秒；
归档gzip读取7.260秒、保存3.092秒，归档Phi重建2.485秒不是原在线训练成本。
LoRA热搬运/pack占B8短测wall不到0.2%；后续不再把复制和decoder作为主要提速论据。
6-step physics/render分解仅是未settle的冷启动小样本，原paired scene恢复被controller_goal_pos guard拒绝的失败和费用保留，不绕过guard。
新增测量4096环境steps、0optimizer、最多1卡、保守整进程0.130040GPUh、约85.7MiB，独立于首批20.518080GPUh，不改原科学结果。

## 8. 后继方法取舍与仍然可能失败的联系

[新合同](../designs/parameter_conditioned_compiler_20261009.md)选择以真实Λ为输入的经验编辑器，保留强MT为每condition初始控制，
成功/失败实际经验均在停止前参与教学重读，然后直接产生一套新的完整rank128 A/B。
原Q的意义依赖过去的共享参数；新事件记录的是实际行为参数与事实，可在χ变化后复用于共享FM，不假装旧轨迹由当前χ生成。
decoder使用共享非线性在`u=0`处的差分，使零编辑保持incoming因子且对编辑token有非零导数；
它取消“改可训练MT坐标就改变尚未实践的初始控制”的直接路径，但仍可能学到忽略输入的公共编辑，也不保证训练后能力保持。
真实FM用同task不同teacher episode的query教共享G，标签梯度经完整A/B回到读取/经验/参数编码和编辑器；新condition没有动作拟合optimizer。

这同时改变了行为起点、参数状态、成功经验消费和学习复用，不是一个可以把所有收益独归某个模块的因果消融。
decoder改变服务于学习关系；实际跨condition native batch、环境生命周期和采集/学习分离服务于效率，二者证据不能混用。
预注册144固定行为条件、180共享更新、72条件刷新及360终点，两个完整400与固定train MT/end/null，成本和提前停止见design。
若train没有净修订收益，不能先归咎held；若train有益而held无效，迁移解释上升；若完整性能提升但null同样，经历必要性尚未建立。
原生teacher特征、8-slot压缩和跨episode成功query的局限仍保留；本次没有证明这些足以表征操作知识，也不把后继成功写在实验前。
本批有界检验结束后仍须更新完整方法判断，不能围绕同一编辑decoder继续小扫，也不能仅交计划后等待Owner催促。
