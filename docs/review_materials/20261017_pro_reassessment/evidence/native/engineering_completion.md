# 固定source-H原生首写180：整批工程与科学读出

唯一有限批已结束：36套新完整38-target/rank8 LoRA、144条新query，native64/source58/reference127/Reader90089。对source R53/G11/L5、churn16、Jaccard0.768116；对Reader R60/G4/L29、churn33、Jaccard0.645161。breadth20/36，source亦20、Reader29、reference35。参考新增70只保10。

target24为19/96（source14/Reader43/reference80），support12为45/48（44/46/47）；五identity20/20均保持。Spatial6/Object0/Goal9/Long4/support45。task35从source0到3、task0从1到3等局部正例保留；Object仍0及Reader task12/14各4→0、20/22各3→0等损失同样保留。

本结果未形成有意义的广泛控制获取，并大幅落后Reader89；本pressure/原生范围组织不自动加量、改scale/QR/rank/LR/seed或另开下一窗。180不称充分训练或所有原生参数写入不可能。这里只检验固定真实source-H的首写组件，不认证完整实践修订、未见任务/strict400、视频或H因果、超过强MT。旧方法已有完整FM/反馈信用的正负证据、原强参照和交互身份均保持。

训练与真实消费者：fresh seed7共180有效更新/720事件/20160FM，每9步36任务等权，每task20不同非报告教学、560FM；每事件7非teacher episode×4分层，chunk50/offset1/源单位。完整CP90/180及rolling171含53参数张量/3144739参数、53 Adam状态、scheduler/scaler字段、完整流/Ω与4rank RNG；episode选择cursor140–147依实际跳过teacher保存，不将它误写成FM行cursor560。90无读出/选择。

完整两条固定J伴随/QR两次迭代/alpha/gauge信用经过实际native50×7 FM。720事件α中位0.617899、范围[0.000782563,1.79463]；伴随相对误差中位0.000303724、最大0.0153153，BF16/TF32正常精度，未detach Q/alpha或作低位追查。每个有效更新pressure/context/video attention梯度finite非零，但不以这些内部证据认证理解或控制。FM前/后20均值.126791/.124205，所有180点及实际流保留。

36末点实际算子改动L2中位0.0355391、范围[0.000120144,0.171549]；这不是identity输出，也不由norm推能力。source-W1090/1180/旧1180只保同视频init0的36锚（18/17/15，新16、Reader22、source13、reference32）；其余108没有伪配对。所有八臂1152逻辑case、success-set/逐task-suite/target-support/identity/RGL/churn/Jaccard及坏例见results.json与case_changes.json。

真实吞吐：4个可丢弃逻辑更新/448FM/0env，4rank support4/FM7=68.619s（含首实际非线性消费者），8/14=32.441s、16/28=19.707s；6rank16/28=19.877s，无观测wall收益，选择4rank16/28。已打包16个登记支撑和28个本condition真实query，6rank混合task pack也实际执行。临时参数/Adam不进fresh正式。正式更新中位19.700s、平均21.170s、最大82.758s，source-H冷构建/文件读取/完整保存实耗保留；正式job64.67min/4.311634977659GPUh，实际allocated峰17225MiB（16.82GiB）。

正式读取36个动态long-first condition job、8个persistent worker，4个实际就绪query/condition，不凑假episode。每次新dense编译前及最终退出都完整runner.close，36次close释放144个环境进程，全部worker/child PID跨node复核均已退出。读出6分钟量级/.667099736134GPUh；最大物理8、node最大GPU-node-B=6/GPU-node-A=4（含profile），0NCCL独立读出、未触碰foreign。各node读出worker峰PSS之和保守上界GPU-node-A=48.780GiB/GPU-node-B=51.810GiB、threads各116、最多实际episode各16；这不是同时峰值。训练/读取准入的node合计CPU及最终UID快照在admission；最终UID zombies=0，GPU无本UID任务，其他session进程不作本批残留或擅自清理。

两个pre-FM失败原件完整保留：199fc3dc自定义Function默认关闭forward AD，0.037065746784GPUh；710b5a7d实际生成FP32而非owner物理identity BF16通信，NCCL广播超时0.908953399393GPUh。分别通过同JVP入口CPU前后向、actual distributed_credit/Gloo2/4/6的76完整形状/112质量消费者窄修，再以新clean pushed detached cbd445a6完成真实GPU验证。两个均0有效更新/0FM/0env，不充值4/448或episode额外；未把科学阴性当bug。

费用：当前ROOT全部13个GPU attempt已结算（11成功/2工程失败），总6.219000091420GPUh，训练4.311634977659/读出0.667099736134/成功临时0.294246231450/失败0.946019146178。144唯一物理episode，36658control+1440dummy=38098步，unknown0/extra0/有效重跑0；原160提交/59200control/1600dummy/60800联合上限保持。0新实践/新RGB-H/held/Test/额外forward探针。

教学：本批成功journal772次完整M读取（720train+16临时+36final），两个失败另8次可证重构，共780读/25700采样帧当量；全是原始合法V，M命中不是0教学。0新M/prefix构建或重读、0producer/held M消费、旧SQLite只读。原1800视频59114采样帧与Reader900已付来源保持；固定source经历144段/40377步/52success/7808replan，选16×36=576不同decision反复消费，0新实践不等于zero-interaction。source参考/Reader/旧反馈等共享成本分别保持：[52.014113090095,52.121858595095]/13.033086503016/9.325162564582GPUh及原source费用，不混成当前ROOT从零成本或双计source-H。

存储：新增全data1；ROOT结算约2.94GiB，H工作缓存2.575GiB<4，旧共享M617130072bytes<1GiB、新M payload0。final体积加单个H原子副本/CP临时与三个小frozen源码树的保守可解释峰<3.6GiB<12；现场storage-node独立quota/个人实耗与shared余量均留admission。没有复制raw H/大模型/W/prefix/参考池，不删原formal资产。

源码结构：相对97cabedc新增1594/删除580，净+1014活跃source行、5个新模块，14文件有改动，最大320行，guard无hard、complexity review最高24。越过1000行的明确理由是固定source全算子双伴随（native）、绑定原件的真实H与有界缓存（history）、一次编译与分布式全因子FM余切（learning）、完整Adam/逻辑流（training）和原子恢复（checkpoint）均有本批真实消费者，拆散会隐藏生命周期。data承接720流，comparison/readout/run仍唯一owner；旧SourceWeightCompiler、weights.py、旧model/held菜单退役，video缩67行，只保冻结M；无平行Writer或旧loss fallback。schema/配对复杂度属于当前fail-closed authority；distributed_credit的compile/broadcast/FM/SUM/伴随是一段有finally释放的单事务，不机械分割为微模块。

实际首次本地观察02:39:51.303327Z，首次owner03:06:46.092437Z，失败恢复不重置；formal launch03:37:51.062690Z、first useful03:39:08.370669Z、training exit04:42:31.534163Z、全部owner exit04:48:36.435624Z。原预计2.5–5.5h，吞吐后余1.3–1.8h修正，实际训练/读出均在修正预期内，工程事故约20min与全部加载等待计费。家族09:05:48.038及旧事故/费用区间不重置。

Owner已因数月性能不足放弃无实践信息直接生成路线的稳定纠正，在main21a50180置顶current_owner_requirements/concept并写findings；不得再降格成实践可选或借新算子恢复该路线。后续整体重审临时稿已被main撤回，本次未集成/未升级为候选或active design，按最新消息留main归还窗口后处理，不作为执行待办。当前仅组件有限批完成，无自动下一窗。

工程验收acceptance.json、training_acceptance.json、teaching_accounting.json、resource_acceptance.json、final_architecture.json及原GPU/CPU/quota/全部job receipts保留；最终main/Git/开发树生命周期见Git_delivery.json。完成后一次可靠交main01a13dd9并归还canonical记录写窗口。

| task | suite | source | reference | Reader900 | native180 |
|---:|---|---:|---:|---:|---:|
| 0 | libero_spatial | 1 | 4 | 4 | 3 |
| 1 | libero_spatial | 0 | 4 | 1 | 0 |
| 2 | libero_spatial | 1 | 4 | 4 | 2 |
| 4 | libero_spatial | 0 | 4 | 1 | 0 |
| 5 | libero_spatial | 1 | 4 | 3 | 1 |
| 7 | libero_spatial | 0 | 4 | 2 | 0 |
| 12 | libero_object | 0 | 4 | 4 | 0 |
| 13 | libero_object | 0 | 3 | 0 | 0 |
| 14 | libero_object | 0 | 4 | 4 | 0 |
| 15 | libero_object | 0 | 4 | 0 | 0 |
| 17 | libero_object | 0 | 3 | 0 | 0 |
| 19 | libero_object | 0 | 3 | 1 | 0 |
| 20 | libero_goal | 0 | 4 | 3 | 0 |
| 21 | libero_goal | 3 | 4 | 4 | 4 |
| 22 | libero_goal | 1 | 4 | 3 | 0 |
| 25 | libero_goal | 0 | 1 | 0 | 0 |
| 28 | libero_goal | 4 | 4 | 4 | 4 |
| 29 | libero_goal | 0 | 3 | 0 | 1 |
| 32 | libero_10 | 0 | 3 | 0 | 0 |
| 34 | libero_10 | 2 | 2 | 1 | 1 |
| 35 | libero_10 | 0 | 4 | 1 | 3 |
| 36 | libero_10 | 0 | 3 | 0 | 0 |
| 37 | libero_10 | 1 | 3 | 2 | 0 |
| 38 | libero_10 | 0 | 0 | 1 | 0 |
| 42 | libero_90 | 4 | 4 | 4 | 4 |
| 43 | libero_90 | 3 | 4 | 3 | 3 |
| 51 | libero_90 | 3 | 4 | 4 | 4 |
| 55 | libero_90 | 4 | 4 | 4 | 4 |
| 56 | libero_90 | 4 | 4 | 4 | 3 |
| 62 | libero_90 | 4 | 4 | 4 | 4 |
| 64 | libero_90 | 4 | 4 | 4 | 4 |
| 73 | libero_90 | 3 | 3 | 3 | 3 |
| 95 | libero_90 | 3 | 4 | 4 | 4 |
| 96 | libero_90 | 4 | 4 | 4 | 4 |
| 97 | libero_90 | 4 | 4 | 4 | 4 |
| 101 | libero_90 | 4 | 4 | 4 | 4 |
