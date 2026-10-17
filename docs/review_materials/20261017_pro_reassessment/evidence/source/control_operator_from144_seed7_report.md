> 既有报告摘存；来源：`runs/source_policy_space_compiler_20261015/control_operator_from144_seed7/analysis/engineering_completion.md`。历史执行状态不构成当前授权。下文未入Git的原目录链接改为文字引用；可读结果投影见[证据地图](../../EVIDENCE_MAP.md)。

# §11唯一operator窗口完整交付

唯一owner315620/supervisor315187于2026-10-15T21:50:45.166926+00:00实际exit0，连续退出reader session35034/cell295已结束。派发2026-10-15T21:17:18.877Z，准备21:17:25.727Z，实际启动21:27:59.673397Z；owner wall 22.758225min，派发到退出33.438165min。比45–90min估计快，因首5min改善不足5%按合同结束学习，实施也比20–45min粗估短；没有跳过闭环。

|36×16训练域配对面板|source|reference|native144|operator1170|
|---|---:|---:|---:|---:|
|全部576|218|502|216|224|
|target24/384|48|315|45|50|
|support12/192|170|187|171|174|
|去identity31/496|138|422|136|144|

相对source R209/G15/L9，churn24；相对144 R205/G19/L11，churn30；相对reference R219/G5/L283，churn288。参考原新增288例保13（144保10），target8/270、support5/18。五identity42/55/64/96/101的80行都是source原执行复用，不能称学习。新端点至少一次成功23/36任务（source22/ref36/14422），8任务净优于source；Object仍2/96，Goal26/96，Long6/96，Spatial16/96，support174/192。task21从144的10恢复14但仍有source成功交换，task34由source4/1443到2；所有不利case及source损失集保留在`36任务表`（原运行目录：`report.md`）与`完整四方原件`（原运行目录：`results.json`）。

同原八task MT128严格配对子集仍MT67；新23/128，R20/G3/L47/churn50，0新MT执行，详见existing_MT128_pairing.json。不得拼成全36/held超MT结论。当前只在标签侧读完整参考参数，无视频/held/Test/相邻同目标稳定性资格。

## 真实学习与停止

完整144迁移到1170，新增1026更新/每task114次访问/8208逻辑参数点配对，原query cursor112/task只作来源、本段0动作/图像/noise训练query。保存220份Adam状态step1170、constant scheduler epoch1170、4rank RNG、saved mixed identity、原76S/bank、38 C、global/phase sampler、累计clock和停止状态。只将loss换为完整38层sBA/C；rank Gram FP32实际梯度，不展开矩阵、不用因子MSE、无native/BC/FM/对比辅助、无MT或参考因子旁路。两live anchors及source identity半权解析零保留，未重分配。

完整271池起点144 weighted BA=0.999999816395，首检查/唯一末点1170=0.970360511763，改善2.963931%，不足5%。累计有效训练300.887120214秒，按首完整36task周期终止；未跑15分钟或第二块，没有回选更低loss点。selected误差1.173635882→1.139940710，other .826363751→.800780313；36task均略有下降，完整task/suite/层/source基线见operator_pool_breakdown.json及诊断原rows271。

起点全池读取3.144529秒，首检查1.493748秒；末点与首检查重合，endpoint.json复用同次结果，非第三次GPU读取。源baseline约1，端点因物理point pack/FP32-TF32 reduction有约1.16e-5数值变化；C文件固定不变，不为低位一致重算。BA小幅改善没有广泛保存参考控制，不能称充分训练、唯一梯度bug或所有低维方法失败；此同目标块按授权结束，由main消费后改假说。

只保留2份完整新CP（rolling前点1134及首检查=终点1170），旧144及全部原CP/成本只读。完整恢复/计时与原子保留验收见acceptance.json，0有效行重跑。

## 吞吐、内存与生命周期

首8有用更新每配置两次，中位6rank/pack1=.555909s、4/2=.292933s、2/4=.414543s、1/8=.535644s；约14.391/27.310/19.298/14.935点/s。选4rank/pack2，在153完整周期迁移并释放其它两卡。单卡pack8已覆盖允许的全部8逻辑点但比4/2慢82.9%，有显存也不为填充扩大科学batch。此为不同任务/点和冷热状态的有用运行比较，不是严格成对profile或全局最优证明。0 OOM；物理比较峰allocated735/983/1621/2873MiB，reserved最高4744MiB。训练不加载source大policy。

6rank启动/退出合计PSS5.384/9.230GiB，4rank启动/退出3.653/6.315GiB；相应线程144/180、96/112，只有边界采样，非连续峰。两个节点容量分开核算，不把其它服务线程算作本批环境。

闭环6个persistent消费者按长任务动态就绪队列，node cap60、实际最大保留52环境槽。31真实shard中16/8/4槽分别5/18/8；按node总PSS与可用RAM而非GPU单独显存限制环境总量。中位end-to-end计划/s为7.604/6.343/4.746，native计划/s为9.101/8.438/6.196（不同task混合，非成对profile）。读出wall15.314558min。实际GPU峰allocated11362.719MiB/reserved29518MiB；single-worker PSS采样最高59.024GiB/18process/52thread，host MemAvailable最低282.012GiB，异步worker采样不伪称同刻node峰。

10份完整close回执均0残留环境PID；记录的112个owner/rank/worker/环境PID全部消失。退出现场GPU-node-A/GPU-node-B均0本任务/本用户CUDA进程，0control_space module PID；foreign进程未动。释放证据resource_exit.json及gpu_exit.json。CPU验收临时脚本曾误读native计时字段及list JSON接口，已按实际schema纠正；0GPU/环境/科学行重跑，不把这些记成模型失败。

## 交互、GPU费用与存储

新496物理提交全部完成，0环境失败/未知/有效重跑，control141326+dummy4960=146286步；独立16额外工程尝试未用。上限512starts/179840control/5120dummy/184960total保持，未刷新旧两窗额度。无新RGB/H、参考数据或视频采集。

本ROOT8个正式GPU job均exit0，费用1.785346320801GPUh；其中有用300.887秒之外的模型/因子加载、全池诊断、同步、保存、评估准备/关闭都包含在实际job成本。无未决GPU exit；没有GPUh硬cap。旧reference区间[52.01411309009465,52.12185859509464]及旧native144 2.0012681053082146独立保留；加本窗后这三个关联ROOT累计[55.800727516204,55.908473021204]GPUh。早前六loader未决上界不记0，N16和其它历史成本不并入本小计。

退出ROOT实占386764800B（后续小型验收报告另计），不是连续峰；新峰预估12GiB、reference11171725312B/旧144 662462464B只读复用。storage-node data1 latest used1748282636KiB/soft2147483648KiB，shared约80TiB。quota与shared分开核验，所有新增data1，0大资产复制/下载。

## 代码与交付

正式train/readout同一clean pushed detached528a57b19d57c6889f2cce13c582e12057fffc21，冻结树control_operator_from144_frozen_20261016。训练源旧eb78/144与其39d21读回身份保持。当前control_space为唯一active运行面，native-only训练/QueryBank/专属credit/旧四节点恢复入口已退役，旧formal在Git/frozen/原件保存；旧144只读比较通过明确phase metadata承接。

结构11个既有source文件、0新source模块、净增149行；FactorEncoder增加独立point batch、原Gram支持batch，其余owner不复制。CPU完整E/D/梯度、旧BlockLayer、流/Adam/RNG、576旧配对、终点物化和预算消费者通过；实际1026 GPU更新、两份完整CP及576新逻辑读出已验收。architecture_guard无hard；prepare/pool/周期停止/案例事务的16–20复杂度保持清楚的有限职责，不增加并行Writer。正式后续不热改冻结。

整批科学已结束，0自动第二块/视频Reader/q/RL/held/Test/参考恢复。main最终记录commit/push/clean及写窗口归还另见Git_delivery.json；完成后唯一一次向01a13dd9回报，由main自主科学消费接续。
