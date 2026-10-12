# G1296完整消费与下一项有界干预：参数块直接读取条件memory

2026-10-12 main科学决定。Owner已授权在硬边界内自主推进并要求仔细研究G架构；GPU01恢复消息已生效。
工程/Git/实际运行继续由实验session独占。本决定不恢复旧队列，不延长G1296，不把小幅收益升级为资格。
当前唯一设计增加后继节，历史§14及原失败/逾时/补测证据保持。

## 1. 完整结果更新了什么判断

我直接读了完整analysis/query48_recovery_completion_20261012的summary和2304原始逻辑行，
四臂各576，MT340/G864341/G1296=348/E455。MT→G保留310/新增38/丢失30，净+8、churn68；
旧G→新G保留305/新增43/丢失36，净+7、churn79。Spatial−7、Object+6、Goal+1、Long+5、support+3；
target train24净+5，support12净+3。任务净优/平/劣14/13/9。task13的+4、32/36的+2与5/7各−3均保留，
task36的旧高点5、本版3不隐藏。E-only134仅传22；72重合行净+4，其余504净+4。
这是真实的小幅、不均匀交换，不是明显超过强MT或相邻稳定性，更不是视频增量和完整Compiler资格。
G1296/E首5动作RMSE .088257，MT/E .088126；真实写入增加，但没有首动作更接近E的信号。

BA loss .961445、条件拟合增益 .02594551是真实进步，后者占相对MT总算子改善67.2954%。
实际BA公共能量44.2479%，不能退回“仅公共坍缩”故事；27个非零E内部仍有条件改善。
同时层等权仅改善3.855%，实际整体能量相对误差仍.990141。当前算子与控制之间还有很大缺口。
从864切BA又新增432更新，没有等时长factor延长对照，因此+7不能唯一归因loss；不为补这种归因再跑旧loss臂。

## 2. 一个真正影响结构选择的CPU结果

在既有全36 E/G864/G1296及MT初始因子上，一次4.373秒CPU SVD/低秩算术，0GPU/forward/episode。
原件operator1296_fixed_factor_geometry.py/json应归档当前ROOT/analysis/main_scientific_readout；不要重跑。
固定每层现有A，求min_B ||s BA−W_E||²；等价将W_E投影到A行空间。固定B的对偶量只作空间诊断。
全程使用同一冻结D38与等任务、等层单位，没有生成或部署oracle/hybrid LoRA。

| 保存条件 | 实际BA loss | 固定A的最优B loss | 固定B的最优A loss |
|---|---:|---:|---:|
| MT初始A/B | 1.000000 | .504372 | 1.000000 |
| G864 | .999513 | .259121 | .973152 |
| G1296 | .961445 | .326453 | .885587 |

当前A并未把B的改进空间封死。新G固定A的代数最优B范数/教师B，中位1.0609，范围.8746–1.7229，
不是必须放大数百倍才有该空间。G1296相比旧864的A空间还略退，而B方向有所改善；二者均不能称拟合充分。
Q/V固定A捕获教师能量分别.64438/.69203，接近；固定B捕获仅.01652/.17871，实际loss .99490/.92737。
因此Q尤其弱不能简单归因A没有提供可用行空间。实际B形状Q2048×8、V256×8，A/B细坐标生成难度不同；
这只是可核对的维度和拟合差异，不把高维本身或随机子空间类比当唯一根因。
固定B的最优A可有大范数，也未部署；所有最优量依赖训练教师标签，只是几何上界，不代表网络可学或闭环分数。

当前固定MT-H/共同E监督理论上可以学B；缺不同失败经历不是本次B弱拟合的原因。
H多样性仅限制未来反馈修订资格。继续增加H、先把A“完全学好”、把B手工放大或延长同一个目标，
都没有由这组证据得到充分支持。当前优先问题是合法条件怎样变成各native坐标上正确的写入方向。

## 3. 实际结构假说，以及历史反证

当前每层只由38个target token直接读memory。304个rank token和10064个64坐标块通过归一化pool/broadcast通信；
同target/rank内A/B和不同coordinate先收到共同条件消息，再靠地址与本地FFN/LN学出不同输出。
它没有硬性B零梯度或必然秩亏，但迫使细坐标相关的证据选择通过共同消息的后续非线性交互学出。
当前完整参数输出有643584有效坐标，Q的B有更多native输出坐标。全局BA标签已经提供真实写入信用，
但共享读写图仍可能使条件选择与坐标生成难以共同学到。这是可失败的学习假说，尚未唯一确诊。

我额外直接重读925e7b1:src/ember/expert_manifold/model.py全文，确认历史已有chunk×rank query直接读视频，
随后轴向通信和地址乘法绑定。不能把“参数块读视频”称作全新原理或未经检验的充分修复。
其完整历史为48→75且仍公共收缩；使用16-phase innovation/去DC因果value、旧source/rank16、raw factor+cos/logscale。
与当前完整L/V/真实H、可训练双native meta、MT中心/rank8、BA算子监督和已有455/576的共同E条件不同。
这些差别不抹去负证据，只允许检验当前明确的一处信息路径；故必须有同目标、同初始化、同学习量的匹配控制，
而不是拿新模型只胜旧失败分数就归因结构。
此前Target-Owned 76独立native heads的99→76→86→68约束单纯扩头；CRW的137/140及训练91→115约束显式A→B耦合；
动作调制122/400约束“增加动作意义就能解决”。这批不拼上独立大head、A→B新分支、native FM、动作辅助或教师重做。

## 4. 唯一架构干预：移动memory读取的位置

保留原FactorLayout、6层endpoint、256宽/64块、地址、归一化消息、rank/target自注意力、广播和共享输出头。
只将每层memory_attention由target读改为block读；复用相同维度/参数形状的attention，不增加第二并行读入路径：

    b <- b + FF(LN_b(b))
    b <- b + MHA(Q=LN_b(b), K=memory, V=memory)   # 新的唯一条件入口
    r <- r + weighted_pool(LN_b(b), target/rank)
    r <- r + rank_self_attention(LN_r(r))
    t <- t + weighted_pool(LN_r(r), target)
    t <- t + target_self_attention(LN_t(t))
    # 删除原 t <- t + MHA(LN_t(t), memory, memory)
    r <- r + LN_t(t)
    b <- b + broadcast([LN_r(r), LN_t(t)])

不同target/rank/side/block query可在softmax选择时直接读取不同证据；随后pool协调参数各部件。
本地地址和parent残差保持，末头不变。静态FactorEncoder/Actor的memory为空，语义不改变；Actor仍不训练/运行。
这没有新增输入信息、task-ID、teacher动作/权重输入、额外专家adapter或目标task轨迹拟合。
完整链仍为exact L和action-hidden有序V经真实native meta、真实MT-H经Reader形成memory；
生成A/B形成唯一sBA，在冻结MT合并基底的38目标上改变attention/flow动作计算，最终固定LoRA闭环。
梯度从同一个BA损失经每个实际A/B坐标和共享输出头到block query及memory K/V，再到Reader/native meta；
不因attention存在就宣称它选择了正确操作知识。可表示、能否学到、能否迁移和闭环收益分别裁决。

## 5. 两臂匹配的fresh有界训练

新根建议/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/coordinate_memory_read864_query16，
两臂为target_memory_control和block_memory_candidate。控制复用旧clean pushed detached代码的target入口，
候选成为唯一canonical新运行面，旧入口只留冻结control来源；不在active代码保留两套endpoint fallback。

架构计算不兼容，两臂均fresh所有ψ/双meta/Reader/static/endpoint及未训练θ、Adam/sampler/RNG，不从G1296迁权重。
相同seed7、meta seeds202610101/202610102、原初始化规则及zero endpoint head，同MT300/P0、S_G、D38、E、全部36任务、
相同逻辑task/video采样流、MT-only真实H/bank0、logical B4、LR1e-4/betas(.9,.95)/eps1e-8/wd1e-4/clip1。
既有teacher/bootstrap/assets只读引用，不重训教师、不制造新的任务内轨迹。视频读取stride5与其余输入合同不变。
两臂都从第一更新使用同一BA目标，均864更新=3456条件=每task96次；432仅完整恢复点，864唯一科学出口。
未改变科学更新量以匹配wall；物理rank/microbatch/frame-chunk可按live吞吐不同，不要求逐bit一致。
相同seed与逻辑数据流用于匹配，不能因两种图同state_dict形状就宣称checkpoint兼容。

864选择基于现有完整学习批：216出口曾主要公共拟合，864已有可判断的条件获取；本批不先做4task toy来筛全36。
这不是保证864足够拟合，也不成为后继永久步数。两臂终点无论正负都完整读出；本批不自动延长或另选中间点。
控制已就绪时可以与候选工程并行启动；不让已准备好的工作等无关阶段，不跨节点拼单个训练。

## 6. 唯一读出、成本与结果分支

每臂固定同36个teacher_demo/原MT-H/P0生成一套完整38-target LoRA，直接在原query16做576条闭环。
这批没有消费额外G-practice的Actor或G后段，不另采36个无当前消费者的G-practice；MT先实践所得H照常读取。
使用现有生成/保存/restore/query底层消费者完成；如需窄接口调整，须证明只省未消费实践，query映射和native RNG不变。
这不改变两臂端点或query因果对比，两臂同样适用；原实践与query状态仍保持分区。
预计新增1152个G query episode、0新MT/E/旧G参照episode；不是1224。MT340/E455/G1296=348及G864341均保留完整强参照。
每臂与MT、E、旧G1296以及两新臂互相给全36/576严格配对、任务/suite、RGL/churn/覆盖、E-only等四类、72/504分别报告。
现有实际首5动作和端点BA/common/conditional/A-B几何作定位，不新增native forward探针。
该面板仍train/nonheld机制读出；不能当strict held400、视频因果、或跨目标相邻稳定性。

事前分支：
- 候选相对匹配控制明显改善B/BA条件拟合且闭环收益也实质增大、保持可接受，才提高该信息路径的支持度；
  仍须相对强MT和完整分布作判断，不以仅胜新控制算成功，后继资格另登记。
- 两臂都得到相近的明显控制收益，弱化“block入口是关键”的主张，说明原BA初始化/学习组织本身也能工作；保留更有效率路径。
- 候选只改善几何而控制仍弱，说明消除这段条件写入不足以解释完整能力，不自动加动作loss或拼下一补丁。
- 两臂都仍弱，则降低并结束在当前Reader下靠全坐标共同E监督及更多入口/宽度/层数补丁取得主要进展的投入；
  消费完整结果后主讨论须重新论证学习原则与特征/功能联系，不将停止线只绑定本版本名，不等待Owner新许可。

主要终点是完整固定LoRA的实际闭环能力，不能由更小BA loss、attention差异或B范数晋级。
没有Teacher/Actor/warm/RL/held/视频controls/新probe/第三臂或>864自动授权。

## 7. 资源、wall预期与工程责任

普通双节点合计6/8、单node6、数据1独立quota/共享容量/非干扰继续有效，不设GPUh硬门槛。
两个独立训练及就绪的终点读出可并行；按真实logical B4吞吐选择每臂物理rank，不固定总并行=某臂world size。
有足够资源时优先有效4-rank或两节点各独立任务；资源不足即时按有效安排启动，不为凑卡空等。
新block attention的query数从38增到10064，代价真实增加；必须用实际最长L/V/H和同B4的可丢弃消费者测吞吐/峰值，
共享K/V、query chunk或高效SDPA只作执行安排，保持完整softmax和梯度，不截断memory或降低输入。
不得依据“注意力约265倍queries”把整个训练成本乘同系数；也不能预先承诺新路径无额外成本。
旧target控制现有chunk16/32实测可复用；新候选按实测吞吐/余量选物理安排，不铺超参扫。
main初估工程30–60min；两臂并行时训练/读出约3–5h，串行或新attention昂贵时可能更长，实际消费者profile后给准确修订。
预计新增峰值16GiB（两臂完整CP432/864、初始化、缓存及小证据），assets/P0/E/H/S_G/D不复制大资产。
出现显著超期/吞吐或资源限制须明确反馈实际原因；不因GPUh超过旧计划阻止有收益的并行，不改变科学矩阵求快。

实验session处理结构guardrail、实现/针对性实际消费者检查、branch/worktree、push/frozen来源、运行/费用/清理生命周期。
main不重复工程审查。正式来源clean pushed detached，当前记录串行合并；本CPU分析与决定一并归档。
一份整批owner记录两臂依赖/真实退出，所有已就绪任务及时运行；唯一整批完成/异常可靠交main，启动给一次真实ETA。
无心跳、阶段Queue、第二等待者。整批结果后main自主验收、理论更新及接续，不把工程ACK或Owner逐项批准作为停点。

## 8. 实验接收与持久证据

完整决定已逐节消费；两臂注册为fresh，不导入G1296参数/Adam/学生bank。
科学主讨论为01a12bb0；工程/Git/运行由实验独占，无工程ACK停点。
稳定决定原件：/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/coordinate_memory_read864_query16/analysis/coordinate_memory_read_decision.md。
固定因子分析原样归档：/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/operator_effect_learning1296_query16/analysis/main_scientific_readout/operator1296_fixed_factor_geometry.py及同名json；未重跑。
当前进入工程/正式准入，实际冻结身份、拓扑/吞吐/费用与启动时刻以coordinator和后续记录为准。

## 9. 实际消费者、物理选择与启动（2026-10-12 UTC+8）

两个正式臂已实际启动，模型结构之外的共同监督及逻辑曝光合同不变。控制训练来源c7e76b566e3d50178f9e43dbdaa977bf28fd9d5b，17:46:45启动于gpu01:0–3/world4；直接读出来源c4f53841be370a510d9f1094f60325b990ac714c，保持同一v8 target-memory图，只省掉未消费的G-practice。候选训练及直接读出来源c35c34b555ea2aa2694c1c43f23de11c6d8e6d4f，18:12:49启动于gpu01:4/5/world2，v9拒绝旧图checkpoint。三个来源均已推送、clean detached冻结，旧树未原地修改。

新接口复用真实MT bootstrap/H、教学、生成、保存与query消费者。每condition只生成一套完整端点；保存路径明确未实践的端点身份，不登记虚构G-H、Actor父节点或student-bank资格。真实task13/34原件的CPU请求检查为每task16条only-G，原场景/env/native RNG、MT/E兼容引用不变；全部36只读参照兼容。新fresh prepare实际CLI消费者核对36条原condition、MT-only/bank0、原S_G/D38/P0与864/432合同，未重算D、未导入旧共享状态。源代码有62项针对性检查通过，静态/Actor空memory行为与初始化规则保持。稳定准入原件在ROOT/analysis/admission，而非仅临时目录。

候选profile使用完整10064 blocks/643584坐标、最长实际92帧教学及104条MT-H；memory最大857 token，没有截断。chunk16两次热更新20.60172/20.60180秒，均值20.60176，allocated/reserved峰值25.20946/26.10156GiB；chunk32为21.14613/20.66695秒，均值20.90654，峰值40.30330/42.79102GiB。32没有实测吞吐增益且多占16.69GiB reserved，距45GiB卡仅约2.2GiB，故选16并停止扩大物理chunk。实际梯度finite、零头打开后双meta梯度均正、source无梯度；5更新/20条件全部丢弃，0episode。profile在18:02:30–18:04:46退出0并释放gpu01:4/5，wall135.697秒、.075387129GPUh，稳定原件ROOT/profile_candidate/attempt_001/verification.json；这只证明执行和成本，不证明拟合或控制收益。

控制首次完整36曝光的9更新平均11.65140秒，外推864训练2.796h；候选实际首次9更新/36条件平均18.64500秒、外推864为4.4748h，两个rank双meta实际梯度为正且source无梯度，完整36逻辑task与control一致。候选最长world2 profile、这一正式启动样本与旧world2平均18.47秒共同给出候选正式训练4.4–5.0h、直接查询.5–1.0h。按18:12实际启动暂估整批23:10至次日00:15结束，较原并行3–5h初估向后修订；候选两卡是当前控制4卡和普通合计6卡现场上限下的安排，不是模型或整批永久上限。控制训练退出后就绪读出立即利用释放卡；候选训练退出后读出可使用所有有效空余卡，不被world2限制。现有完整中间恢复点只有432，不为形式扩卡回退未保存更新或原地改冻结训练代码；物理迁移仍须完整checkpoint和正确消费者及真实wall收益。

实际candidate launch核双节点与strg01独立data1 quota，used1470479352/soft2147483648KiB，项目最终6卡/cap6/node6；16GiB增量与共享81TiB容量可容纳。大资产原位引用，所有新增cache/TMPDIR/output都在data1。单一owner为ROOT/coordinator/matched_coordinate_read864_01，17:46:32启动、gpu02 PID681682、tmux ember-coordinate-memory864-20261012；依赖文件采用事件等待，真实G退出才派发常驻动态query队列，整批实际退出仅一次通知实验session消费。

工程变更已集成main；task-owned codex/coordinate-memory-read864开发分支和开发树在确认等价集成、clean及profile退出后清理。正式训练/读出三个冻结树及关键恢复资产保留。当前没有CP864或576新结果，后续验收按本分析§6完整口径；不把启动、profile、测试或有限loss作为科学通过，也不自动派发任何后段。

## 10. 一次现场效率核查：六卡新共驻与ETA修订（2026-10-12 20:32 UTC+8）

Owner询问训练耗时，main提供两臂native计算约翻倍的异常线索。实验只做一次针对性双节点GPU频率/功率/温度/进程及CPU/NUMA快照，消费现有metrics；没有新模型forward、训练更新、episode或profile。稳定原件在ROOT/coordinator/matched_coordinate_read864_01/efficiency_check_20261012_2030/{gpu_preflight.json,gpu01_live.txt,gpu02_live.txt,metrics_snapshot.json,summary.json}，私有进程身份仅留run原件。

20:32:38快照：控制所有rank完整到732/864，最近72更新均21.75498秒；候选到383/864、均34.47073秒。按实际teaching frame归一，dual-meta CUDA forward由约.043升至.089秒/frame，含native重算的backward由约.128升至.265；候选endpoint forward约.25秒/rank/update，仅小部分。H decode控制1.401→1.423、候选2.827→2.846秒/rank/update，不能由H读取或更多教学帧解释整体翻倍。

现场发现另一个用户约19:42启动的作业覆盖本项目全部gpu01:0–5，每卡额外约16460–16462MiB，本项目原进程仍约27GiB；六卡util均100%。SM当前1710–1740MHz、max1740，温度61–73℃，未见当前SW/HW thermal slowdown；部分SW power cap可瞬时active，当前频率差远小于两倍。CPU近期84–86% idle、CPU/memory PSI零、可用内存约392GiB、无swap；控制四rank绑node0、候选二rank绑node1且大部分内存页本地。双节点没有空闲专用卡；gpu02少数显存有余量卡也有高负载任务，另两张低util卡不足10GiB free。**共享GPU计算争用是目前最有力解释，快照并非停掉竞争任务的因果隔离，也不是硬件故障诊断。**

没有定位到本项目可逆调度错误，故不信号任何GPU进程、不改驱动/功率/冻结代码；原owner PID681682、两个G和唯一退出接续保持。已有owner在控制真实退出后立即将释放的四卡用于动态直接query，候选最后可用全部六个项目槽位，不受训练world2限制。candidate CP432尚未到达；实际冻结v9 resume消费者明确拒绝world/rank RNG cardinality变化，没有既有正确的2→4迁移入口。同world换设备不能在当前双节点同样共享且无空闲卡的事实下证明有收益。本次不临场铺迁移工程或回退未保存更新，不中断近完成控制；432恢复only、864唯一读出及所有科学合同保持。

按这一快照最近72更新和现有共驻负载，控制剩132步约.80h，训练约21:20完成；候选剩481步约4.606h，训练约次日01:09完成。后续query仅据既有实际读出和native计算翻倍外推为1–2h，整批当前约**10月13日02:10–03:15 UTC+8**，没有新增绝对截止。原23:10–00:15预测属于启动时负载，已被此修订取代；未来外部负载或新G实际episode长度仍影响wall，不能承诺固定完成时刻。不因本异常扩大科学矩阵、设置新等待者或阶段通知；有限核查一次回main，原整批完成回报保持。

## 11. Owner11点目标与一次候选432提前诊断

Owner最终目标仍是完整Writer在固定validation/test超过强MT，训练任务MT-only初次修正只提供先行信号；11点为希望取得较好效果的交付/决策目标，未要求强杀GPU。主讨论在未读新闭环分数前明确登记候选已保存完整CP432全36/query16一次提前诊断，以尽早缩小科学未知；两864训练/完整匹配比较保持。最新完整执行范围仅见唯一design§15.1，432不与control864作结构因果比较，不构成held/反馈Writer/正式选点。总新增query期望1152→1728，0新增实践、共享更新、native probe或其它中间点。

早期root为ROOT/early_candidate432_query16，CP432/大资产原位只读引用；现有c35 frozen endpoint-query消费者合法验证G stage432，无绕校验或模型改动。CPU核原36 panel/condition及MT/E576身份通过，D与先前科学算术未重算。控制G于21:20:47实际退出0；原CPU owner未消费ready通过已有abort_dependencies接口取消，将派发权串行交给ROOT/coordinator/early_candidate432_priority_01，原在途G/实际退出收据保留。旧CPU退出按dispatch_handoff处理，只有新dispatcher派发读出与承担整批科学完成，不恢复旧队列或设置第二整批等待者。

统一owner于21:35:12启动PID1808572，四个c35/v9 resident在21:35:39–51占gpu01:0–3，原candidate训练继续4/5；slots16/chunk8/group2沿既有实测，不新profile。各job现场核普通6/cap6/node6，独立data1 used1471424172–1471424292/soft2147483648KiB/shared81TiB，预计<1GiB新增。四份Runtime记录及首8合法端点已核CP432、原MT-H、一次生成、0G-practice。稳定身份/准入/原件见新coordinator/{dispatch_revision,launch,dispatch_start,admission_cpu,startup_verified}.json；host-private ops不进Git，原冻结树未热改。

四卡优先早期432，所有就绪早期任务已派发后释放的有效slot及时服务control864；candidate864就绪后也按有效空余slot动态处理。按既有共驻成本四卡早期1–2h，当前22:35–23:35有完整早期结果的预期；尽量把早期与control读出覆盖candidate原预计至01:09的训练窗口，整批暂修订次日02:15–04:00。尚无完整早期query实测，不保证固定速度或正结果；原预测保留为历史。完整432允许一次提前科学交main，原两864整批仍交一次，分别判重，无心跳或task级消息。

后继有限CPU成本/消费者清点原件ROOT/analysis/successor_cost_inventory_20261012/inventory.md，0新forward/episode/profile。v9已经由EndpointTrainer承接BA/实际P-H的G-theta交替及固定Writer400消费者，但当前直接端点没有实践prefix，theta未训练、后继未登记，不能据接口存在称完整方法就绪。旧真实strict400六worker2.081h包含1133适应和119新final，不能用576直接query吞吐代替Compiler成本；共驻同路径粗情景约4.2h，大H/theta与新formal成本缺计时。主讨论据完整早期或整批结果自主冻结后继规模，清点不自动启动旧课程、Actor、held或Test选择。


## 12. 真实batch、局部故障恢复与阶段纠正（2026-10-12晚）

Owner明确先通过G从MT一次修订的完整闭环增益和合理保持，才登记完整Writer训练/评估。完整Writer在固定validation/test超过强MT仍是最终目标；11:00是效果和决策目标，不要求塞入后段或400，也没有新的GPU强杀截止。此次有界运行检查、两条CUDA缓存释放语句和物理组大小变化均不改变科学矩阵、模型、标签、BA目标、CP/schema或逐行policy/env RNG。

22:15核已有真实runner统计，四resident快照合计12组、384个完成G query、2209次native调用/18996行计划；平均物理batch8.599，满16调用10.73%、满16计划19.96%。resident累计7279.494秒为多worker之和，吞吐189.903 episode/GPUh；native prefix521.615秒、flow3443.251秒，合计54.47%，环境等待380.140秒/5.22%。独立环境操作时长不能加成墙钟比例。Torch allocated峰11.503GiB/reserved20.756GiB，组后设备free分别1.11/8.69/0/7.63GiB。显卡利用率和slots16不表示实际batch充分；当前主要计时是native计算，未证明改变coalesce会有收益。

early worker03在21:55:36切换task95/96环境时EGL framebuffer0x8cdd退出1；上一成功group记录设备free0，提示模型/缓存和EGL共驻余量风险，但不是唯一OOM或GPU故障证明。更明确的调度错误是旧owner的全局failures分支取消其它就绪派发，导致worker自然退出后空卡未接control864。执行者承担该错误。保留三个成功resident原544行、原失败日志/费用/保存端点；等待这些实际GPU退出收据齐全后只结束旧CPU派发owner，原候选两卡G训练及其真实receipt writer未受信号。旧回报按dispatch_handoff判重，不复活旧队列。

新唯一owner scoped_readout_recovery_02于22:28:35/PID2084323实际启动，22:28:41–53派发原两缺项及control864，22:29:12把没有领取任务而自然退出的早期worker卡立即用于control；该启动成本计账而不虚增episode。实际Runtime为项目6/cap6/node6：四卡读出、候选两卡原训练。每次launch双节点/非干扰和strg01独立data1 quota核验，used1472447812–1472448036/limit2147483648KiB，大资产原位复用。新owner限定失败root/worker，已定位设备故障避免重复同卡、失败分片至多一次工程恢复，独立就绪工作继续；没有第二等待者、额外profile或心跳。

后续worker保持slots16/chunk8/coalesce0.1，把待消费condition组2→4以减少组尾补位不足。环境重置前仅释放Torch未用缓存：v9读出eac6ebf45e02652d6e04c27ebde7d42163eb0808、v8控制62e615c9a83e23f17126a7aeff77f74c10f40982，均clean pushed detached；canonical保持单一block模型，原训练身份不变。实际CPU消费者读取已完成条件0.011秒/0新forward或episode。新真实恢复组从保存端点只补32query，332.062秒/4362步，平均batch7.755，prefix24.841秒/flow151.622秒/env wait23.299秒，allocated11.339GiB/reserved18.959GiB、组后device free5.086GiB，实际exit0。该组只有两项且任务不同，不能证明group4墙钟收益或缓存是唯一EGL根因。slots32未采用：16已出现free0和EGL失败，32的EGL/native合计峰值未测；不以模型加载前空闲显存承诺可放大，也不延迟主要读出铺profile矩阵。组4的真实收益待原整批已有物理记录消费，未证实前不称已充分优化。

完整Writer已有ready批量native、独立环境、每行LoRA/RNG和0.1s coalesce，但compile_conditions的同步teaching/Actor/G回调仍会挡ready结果消费；history observation microbatch16而响应states逐候选循环。旧适应condition_batch4同时只有至多4条真实episode，组尾等待未消除。这与当前直接query已有最多16活跃环境不同。旧strict400去重106组，编译residence合计38675.382秒、G host614.557秒/1.59%、native candidate response8658.717秒/22.39%；六worker端到端2.081h包含1133适应及119新final。不能把CFM生成说成主要耗时，或把576直接query吞吐代替完整Compiler成本。完整Writer批量结构本次未实施，先服从G阶段证据。

物理小原件：ROOT/analysis/execution_efficiency_20261012_2215/{gpu_preflight,resident_snapshot,baseline_statistics,path_timing_snapshot,first_new_group_physical}.json；handoff、实际launch/退出/source/Git生命周期见ROOT/coordinator/scoped_readout_recovery_02。恢复时早期预计22:38–23:10，实际更早22:34:58完成；整批保留02:15–04:00共驻负载情景，不将组4当已测加速，原预测保留。


## 13. 候选432提前全36/576闭环已完成：337对MT340（2026-10-12）

候选完整v9 CP432/full optimizer-RNG-sampler/world2/logicalB4在读出前显式登记为一次半程诊断，每task48次呈现。22:34:58全部36条件各原query16完成，固定教学/MT-H/P0、唯一38-target端点、初态及每行native/env RNG不变；原34项544行保留，仅task95/96读取原保存端点和H补32query，mtime及真实恢复消费者证明未重生成或实践。全部六job attempt、原失败和CPU派发交接保留，未把worker故障当模型科学失败。实际冻结CPU消费者15.575秒核配对、合法引用、schema与所有原件，0新增GPU/forward/episode；未另做算术或探针。

| 臂 | MT | G432 | 旧factor G864 | 旧BA G1296 | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 成功/576 | 340 | 337 | 341 | 348 | 455 |

MT→G432保留311/新增26/丢失29，净−3、churn55、success Jaccard .849727。旧factor G864→G432为R303/G34/L38/net−4/churn72/Jaccard .808；旧BA G1296→G432为R315/G22/L33/net−11/churn55/Jaccard .851351。它们保留强历史参照，不是同学习年龄或单变量架构比较。任务净优/平/劣13/11/12，20个task有新增、18个有丢失；至少一成功覆盖MT33/G43235/E36，覆盖增加不能抵消成功总数未增。

| suite | MT | G432 | 旧G864 | 旧G1296 | E | R/G/L | 净 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| libero_spatial | 53 | 50 | 49 | 46 | 79 | 46/4/7 | -3 |
| libero_object | 34 | 36 | 35 | 40 | 68 | 29/7/5 | +2 |
| libero_goal | 49 | 45 | 50 | 50 | 64 | 43/2/6 | -4 |
| libero_10 | 33 | 35 | 34 | 38 | 61 | 27/8/6 | +2 |
| libero_90 | 171 | 171 | 173 | 174 | 183 | 166/5/5 | +0 |

train24/384为MT169/G432166/G864168/G1296=174/E272，R145/G21/L24/net−3；support12/192为171/171/173/174/183，R166/G5/L5/net0。72条teacher选点重合为43/47/43/47/55、R42/G5/L1/net+4；504其余为297/290/298/301/400、R269/G21/L28/net−7。不能只报前72的正变化，504也仍是训练task机制读出，不称held或独立teacher泛化。

E-only134仅传17（旧G1296 22）；MT-only19保13（旧12）；both-success321保298（旧298）；both-fail102获9（旧16）。新增26分成17条E-only和9条both-fail，同时原MT成功丢29；未出现把E的大幅能力优势稳定传给G的证据。原已有首5 native action的G432/MT RMSE .004600、G432/E .088068、MT/E .088126，仅作定位，不证明视频/H因果或闭环改善原因。

main已直接读取2880条原始逻辑行，独立重汇总五臂成功集合、E-only及72/504，与上述消费一致；亦读取原始physical histogram和当前源码。随后仅消费各臂update≤432的1728条既有condition event，两臂的(update, task, demo)集合完全相同。以下为main登记的原在线pre-update BA loss，每个窗口36update/144conditions、覆盖全部36任务，MT归一均值为1；不是CP432端点重新评估，没有读取>432、额外forward或按此重新选点。

| 已有更新窗口 | target-memory control | block-memory candidate |
| --- | ---: | ---: |
| 1–36 | .9990078252 | .9989982114 |
| 73–108 | .9961377555 | .9959462064 |
| 181–216 | .9927474301 | .9921220278 |
| 289–324 | .9880925465 | .9863297621 |
| 397–432 | .9840556949 | .9806529301 |

候选末窗只比控制低.003402765，即MT归一尺度约0.34个百分点；候选相对MT基线仍仅低约1.93%。这不能称BA目标已经充分学好而只差后段选择，也不能由微小在线loss优势证明block入口已有控制收益。数值来自主讨论对已有记录的消费，实验侧未重做这项算术。

main同时核原teachers/task_0013/E.json和task_0020/E.json：两者reference=MT300、selected_update=0，weights指initial。同576中task13 MT/E7→G9（+2），task20 MT/E8→G4（−4）。这些正反例都不是恢复E新增能力，说明小幅输出扰动会混合产生局部涨跌；不能把所有G gain都称教师能力迁移，也不能由此断定纯随机或视频没有作用。真正E-only134只传17而丢原MT29，主要缺口仍是有用操作增益未稳定传到固定策略。

已有第一replan前5动作RMSE G/MT .00460045、G/E .08806758、MT/E .08812584，支持初始动作总体仍近MT、未实质更近E；不表示全轨迹行为相同，也不构成video/H因果证明。实际链为L/V/H→逐block MHA→A/B→部署BA→冻结MT基底原生动作，当前梯度监督是BA权重距离，没有动作功能或闭环梯度。细坐标直接读取放宽表示路径，但不会自动保证BA拟合或行为传递；不能将半程阴性唯一归因某模块、学习不足或后段选择。

全部任务原件，不隐去task20−4、task5−3、task0/25−2；task1+3、task13+2也保留：

| task | MT | G432 | 旧G864 | 旧G1296 | E | R/G/L | 净 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 14 | 12 | 11 | 12 | 15 | 12/0/2 | -2 |
| 1 | 4 | 7 | 3 | 6 | 15 | 4/3/0 | +3 |
| 2 | 14 | 14 | 15 | 13 | 16 | 14/0/0 | +0 |
| 4 | 8 | 7 | 9 | 8 | 7 | 7/0/1 | -1 |
| 5 | 6 | 3 | 5 | 3 | 14 | 3/0/3 | -3 |
| 7 | 7 | 7 | 6 | 4 | 12 | 6/1/1 | +0 |
| 12 | 7 | 8 | 7 | 7 | 11 | 7/1/0 | +1 |
| 13 | 7 | 9 | 9 | 11 | 7 | 6/3/1 | +2 |
| 14 | 11 | 10 | 10 | 11 | 16 | 10/0/1 | -1 |
| 15 | 3 | 3 | 4 | 3 | 9 | 2/1/1 | +0 |
| 17 | 0 | 1 | 1 | 1 | 12 | 0/1/0 | +1 |
| 19 | 6 | 5 | 4 | 7 | 13 | 4/1/2 | -1 |
| 20 | 8 | 4 | 9 | 7 | 8 | 4/0/4 | -4 |
| 21 | 14 | 14 | 14 | 14 | 16 | 14/0/0 | +0 |
| 22 | 10 | 11 | 9 | 11 | 15 | 10/1/0 | +1 |
| 25 | 2 | 0 | 1 | 1 | 6 | 0/0/2 | -2 |
| 28 | 15 | 15 | 16 | 15 | 16 | 15/0/0 | +0 |
| 29 | 0 | 1 | 1 | 2 | 3 | 0/1/0 | +1 |
| 32 | 2 | 3 | 2 | 4 | 13 | 1/2/1 | +1 |
| 34 | 11 | 10 | 10 | 12 | 10 | 9/1/2 | -1 |
| 35 | 12 | 13 | 10 | 11 | 12 | 12/1/0 | +1 |
| 36 | 1 | 1 | 5 | 3 | 12 | 0/1/1 | +0 |
| 37 | 7 | 7 | 6 | 7 | 12 | 5/2/2 | +0 |
| 38 | 0 | 1 | 1 | 1 | 2 | 0/1/0 | +1 |
| 42 | 16 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 43 | 7 | 8 | 10 | 8 | 15 | 7/1/0 | +1 |
| 51 | 14 | 15 | 14 | 15 | 16 | 14/1/0 | +1 |
| 55 | 16 | 15 | 16 | 16 | 16 | 15/0/1 | -1 |
| 56 | 13 | 14 | 13 | 15 | 13 | 13/1/0 | +1 |
| 62 | 14 | 15 | 14 | 15 | 16 | 14/1/0 | +1 |
| 64 | 16 | 15 | 16 | 16 | 16 | 15/0/1 | -1 |
| 73 | 14 | 13 | 13 | 13 | 14 | 12/1/2 | -1 |
| 95 | 13 | 13 | 13 | 13 | 13 | 13/0/0 | +0 |
| 96 | 16 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 97 | 16 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 101 | 16 | 15 | 16 | 15 | 16 | 15/0/1 | -1 |

此次早期读出新增576个完成G query/136249步，0G-practice/共享更新/参照episode。六attempt一失败，readout allocation 2.904308725GPUh包含失败.330519542，恢复32query/4362步和无领取任务的worker启动开销均计入。失败未完成环境步未知而非0，不能把136249写成全部实际交互步；CP432借用原在途864训练资产，不重复收费或把尚未退出训练费用写成已结算。首次实际query worker启动21:35:38.983至最后退出22:34:58.051，墙钟约59分19秒；失败范围/全局停派造成的空档保留，不用新成功回执改写旧失败。

半程完整读出尚未取得G从MT一次修订的净增益，没有通过当前先行阶段。它降低“将条件入口移到block即可早期显著传递E能力”的支持度，但不单独证明充分学习后的结构失败：曝光是每task48而非最终96，旧参照学习年龄与目标历史不同，未补control432。两臂864才是已登记fresh等曝光结构比较，仍由同一owner继续；不自动停止原训练、不挑最高CP、不借完整Writer/held掩盖这一阴性。完整结果优先可靠交main，后继由main科学消费裁决。

main直接消费后的阶段裁决是保持原两臂864匹配训练及完整读出，不因−3提前中断，也不因loss下降延长。若两臂864都弱，停止当前Reader+全坐标共同E监督的入口/宽深补丁投入，由main重新论证后继；不是只关闭某个版本名。当前没有后段、held或>864授权，不新增探针或CPU全量几何计算。新scoped dispatcher仍为唯一整批owner，组4收益随已登记原读出统计结算，不延迟科学交付；此记录补充随原两864整批结果带commit，不另发科研或工程回执。

稳定原件：ROOT/early_candidate432_query16/analysis/early_completion_20261012/{consume_early576.py,statistics.json,paired_raw_rows.jsonl}，raw共2880逻辑行/五臂各576，逐行保留真实源；EARLY/analysis/first_student_query16_{summary.json,rows.jsonl}为实际原生消费四臂2304逻辑行。ROOT/coordinator/scoped_readout_recovery_02/early_complete.json是实际完整事件；统计不会覆盖原失败或未来两864输出。


## 14. Owner取消固定864义务，未拟合教师的本批已停止（2026-10-12 23:00 UTC+8）

Owner最新原话：“没必要非得864步啊，难道以后每个架构都训练五六个小时？现在这结果都已经烂完了，说明教师根本就没拟合上”。main接受纠正，撤回§13末尾必须等待两864完整匹配比较的安排。半程在线BA误差约.98、完整G432337/MT340且E455已足以判定当前G尚未有效拟合教师；教师E已有真实能力，不能把G拟合失败写成教师E没有能力。原864只是当时的投入安排，不是科学义务，本批不再支付原终点成本；仍保留未完成匹配比较的限制，不为满足形式上的原方案继续低效计算。

先以能改变判断的小投入确认有效学习，再按实际缺口扩大学习及闭环读出，是后继投入原则；不机械给每个架构五六小时或靠固定步数拖延取舍，也不形成永久禁止长训练或新的固定步数菜单。G未有效拟合时不扩完整Writer。最终目标与自主研究继续，main承担完整学习原理分析；本条未授权新源代码、学生架构、训练或其它科学矩阵。

22:57:53按唯一scoped CPU owner2084323的UID/精确argv/start ticks先SIGSTOP阻止新派发/失败恢复，保留排他停止记录。23:00:09对本批7个GPU作业根PID及68个核实后代TERM；一个本批进程在短等待后KILL。23:00:19全部本批GPU身份退出，23:01:51一次双节点nvidia/proc核验本批和ymdai GPU进程均0，精确root及CPU owner也均退出。不操作其它用户/其它任务、reset或改冻结源码，不等完整group/epoch/864。旧matched回执1另外含dict重复UTC的退役CPU错误，真实G SSH退出255与原错误保留；迟到旧完成消息按handoff与Owner停止判重，不视为恢复授权。

恢复资产：控制完整v8 CP864（原432也保留）、候选完整v9 CP432/full optimizer-RNG-sampler。候选两rank metrics各641条，末update均641；源码在optimizer.step、CUDA同步后记录update+1，故最后日志证明完成641。已记录但未落盘433–641共209更新，642是否有未记录部分或已完成未知。无即时合法checkpoint保存接口，未热改或伪造CP641/648，未重复训练；原件/关键权重/H/冻结消费者保留，不将日志当可恢复状态。

| 读出范围 | 有效完成G行 | 涉及task | 16行齐备task | query组complete task | G/同身份MT |
| --- | ---: | ---: | ---: | ---: | --- |
| 候选432提前诊断 | 576 | 36 | 36 | 36 | 337/340 |
| 控制864被中止读出 | 351 | 24 | 20 | 16 | 252/253 |
| 候选864 | 0 | 0 | 0 | 0 | 无读出 |

控制351行均为runner完成后原子写入的真实G_000 query，精确原MT配对/RNG/proprio和端点CP864身份已核；16task完整query组收据覆盖256行，另外95有效完成行保留，不因组尾未到而删除或补写组complete。自然完成partial子集R241/G11/L12/net−1；长优先/组尾/停止时刻造成覆盖不完整，不能称全36/576、无偏估计或864匹配结构结果。缺225行不补，候选864无端点读出不补；早期完整576及既有强参照不重跑。具体每task覆盖和全部351原行见稳定统计/partial raw，不按成绩挑行。

全root完成新增G query927，0G-practice或后段更新/额外参照episode，完成环境步下界218834；失败/被中止episode步未知而非0。12正式attempt保留，包括早期旧失败.330519542GPUh与本次人为停止。formal allocation上界28.744634244、原discard profile.075387129，总上界28.820021373GPUh；formal下界28.733577140，总下界28.808964269。四被中止control worker原parent wait返回码无法取回，外部Owner-stop收据明确exit_code=null/complete=false，并保留精确进程/信号和退出核验；未伪造0或正常完成。四worker allocation上界1.988508808，TERM至核实退出造成上下界约.011057GPUh差。原control G实际exit0/14.268524629、candidate G实际SSH exit255/9.583292082、early读出2.904308725分别结算；引用资产与旧参照不双计。

完整停止消费CPU1.665秒/0新增GPU、模型forward或episode，没有重做main先前BA/几何算术，也没有新增探针。ROOT/analysis/owner_stop_20261012/{consume_stop.py,statistics.json,partial_control_paired_raw_rows.jsonl}保存全部账目/覆盖/CP/日志身份；ROOT/coordinator/scoped_readout_recovery_02/owner_stop_*保存精确停止/两节点核验和CPU交接。该owner/exit.json明确Owner科学停止、complete=false、无恢复授权，使原完整终点回报转换为唯一停止回报；不修改早期完整结果或旧失败。停止实际回报优先投递main，文档Git身份随后存analysis/owner_stop_20261012/git_delivery.json，不另发文档ACK或恢复自通知。

当前无执行中的active实验合同，原design§15已结束；§9–13的预测、继续864决定与资源安排均保留历史，不能恢复执行。canonical架构源码此次未改，关键恢复资产和冻结树保留；后继由main依据完整学习原理自主登记，不等待Owner再逐项批准。
