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
