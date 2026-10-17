# §21因子信用1180与完整976读出

唯一新1180：seen **306/576**，strict validation **128/400**。匹配§20为244/57；强Reader900334/117、Reader1800413/111、MT153/G165/D175168与全部历史正负例保留。当前仅改变阶段A损失；无实践、controls、Test或自动后继。

seen：{'Reader1800': 413, 'Reader900': 334, 'U36': 484, 'factor_credit1180': 306, 'reference': 502, 'source': 218, 'source_weight1180': 244}；breadth={'Reader1800': 34, 'Reader900': 32, 'U36': 36, 'factor_credit1180': 33, 'reference': 36, 'source': 22, 'source_weight1180': 27}。
相对source：R203/G103/L15；churn118；Jaccard=0.632399。
相对reference：R287/G19/L215；churn234；Jaccard=0.550864。
相对Reader900：R261/G45/L73；churn118；Jaccard=0.688654。
相对Reader1800：R266/G40/L147；churn187；Jaccard=0.587196。
相对source_weight1180：R225/G81/L19；churn100；Jaccard=0.692308。

validation：{'D144': 168, 'D72': 175, 'G165': 165, 'MT153': 153, 'Reader1800': 111, 'Reader900': 117, 'factor_credit1180': 128, 'psi180': 119, 'source_weight1180': 57}；breadth={'D144': 7, 'D72': 7, 'G165': 6, 'MT153': 6, 'Reader1800': 5, 'Reader900': 5, 'factor_credit1180': 5, 'psi180': 5, 'source_weight1180': 5}。
相对Reader900：R78/G50/L39；churn89；Jaccard=0.467066。
相对Reader1800：R54/G74/L57；churn131；Jaccard=0.291892。
相对MT153：R96/G32/L57；churn89；Jaccard=0.518919。
相对G165：R97/G31/L68；churn99；Jaccard=0.494898。
相对D72：R99/G29/L76；churn105；Jaccard=0.485294。
相对D144：R94/G34/L74；churn108；Jaccard=0.465347。
相对psi180：R82/G46/L37；churn83；Jaccard=0.496970。
相对source_weight1180：R45/G83/L12；churn95；Jaccard=0.321429。

seen参考新增：288；各臂保持：{'Reader1800': 193, 'Reader900': 116, 'U36': 252, 'factor_credit1180': 87, 'reference': 288, 'source': 0, 'source_weight1180': 35}。
target/support：{'source_seen_support12': {'Reader1800': 181, 'Reader900': 175, 'U36': 184, 'factor_credit1180': 171, 'reference': 187, 'source': 170, 'source_weight1180': 173}, 'target24': {'Reader1800': 232, 'Reader900': 159, 'U36': 300, 'factor_credit1180': 135, 'reference': 315, 'source': 48, 'source_weight1180': 71}}；identity={'Reader1800': 80, 'Reader900': 80, 'U36': 80, 'factor_credit1180': 79, 'reference': 80, 'source': 80, 'source_weight1180': 80}。
全部success-set、任务/suite/target-support、RGL/churn/Jaccard及逐case实际首plan配对均在results.json；旧MT缺proprio限制不补造，D/G含MT、ψ含实践，均非本批匹配机制消融。

| seen task | group | source | reference | U36 | Reader900 | Reader1800 | previous1180 | pair1180 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|0|target24|5|16|13|8|12|6|10|
|1|target24|0|16|16|2|12|0|5|
|2|target24|7|16|16|13|15|6|8|
|4|target24|0|14|11|8|11|0|1|
|5|target24|2|15|15|12|16|6|10|
|7|target24|0|16|15|14|15|0|1|
|12|target24|0|14|14|7|8|2|5|
|13|target24|0|13|11|1|0|0|8|
|14|target24|0|15|15|13|12|1|12|
|15|target24|0|14|14|0|12|0|5|
|17|target24|0|14|12|0|0|0|0|
|19|target24|2|14|14|8|3|3|8|
|20|target24|0|9|7|7|6|2|5|
|21|target24|14|16|16|13|15|13|10|
|22|target24|1|16|16|13|15|9|14|
|25|target24|0|10|10|0|6|0|0|
|28|target24|11|16|16|16|16|15|16|
|29|target24|0|7|7|0|5|1|1|
|32|target24|0|14|15|1|13|0|1|
|34|target24|4|7|6|3|5|2|6|
|35|target24|1|12|10|7|11|2|4|
|36|target24|0|10|11|3|12|0|0|
|37|target24|1|15|15|6|9|2|4|
|38|target24|0|6|5|4|3|1|1|
|42 identity|source_seen_support12|16|16|16|16|16|16|16|
|43|source_seen_support12|9|15|16|9|13|10|9|
|51|source_seen_support12|15|16|15|15|16|15|16|
|55 identity|source_seen_support12|16|16|16|16|16|16|16|
|56|source_seen_support12|14|16|15|15|14|15|15|
|62|source_seen_support12|16|16|16|16|14|16|16|
|64 identity|source_seen_support12|16|16|16|16|16|16|16|
|73|source_seen_support12|7|13|11|9|12|7|5|
|95|source_seen_support12|14|15|15|15|16|14|15|
|96 identity|source_seen_support12|16|16|16|16|16|16|16|
|97|source_seen_support12|15|16|16|16|16|16|16|
|101 identity|source_seen_support12|16|16|16|16|16|16|15|

| validation task | Reader900 | Reader1800 | MT | G | D72 | D144 | ψ180 | previous1180 | pair1180 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|3|21|29|41|43|45|43|24|1|32|
|6|16|35|7|8|6|7|16|4|9|
|11|42|23|36|37|37|37|44|8|38|
|16|0|0|10|21|24|20|0|0|0|
|23|0|0|0|0|1|1|0|0|0|
|26|20|16|35|38|37|33|18|42|38|
|31|18|8|24|18|25|27|17|2|11|
|39|0|0|0|0|0|0|0|0|0|

## 目标、信用、流与CPU算子读回

同4,838,160参数SourceWeightCompiler/fresh seed7/源W、A0、76S/Φ900完整M。前1000仅teacher-detached (L_A+L_B)/2；真实BA仅诊断；后180只FM+.1完整BA、同一Adam和LR，无pair辅助/重置。EA/EB只在损失，0标签入条件/gauge重排/旧U或code head/MT/H。

pair末20：L_A=0.200410，L_B=0.447443，L_pair=0.323926，真实BA=0.759277；joint末20 BA=0.685333/FM=0.106504。
完整1180有效更新、36000+720逻辑视频条件/20160跨episode FM；每task1020visit/560query、每50视频20完整无放回周期，真实flow/cursor已核。新schema拒绝旧§20恢复。1000/1090/1180和rolling1170完整保存generator/Adam/scheduler/scaler/RNG/topology/流；1000→joint同一Adam实际承接。
真实CPU dense/Gram及生成A/B梯度、teacher stopgrad/identity L_A=0/诊断不参与反传、全A/B native余切1–6rank/micro7..56、原子恢复和旧M payload身份消费者通过；GPU首pair两head非零、body0，后body和native余切非零，只是信用证据，不认证操作理解或迁移。

正式pair/joint平均0.248229/1.759487s/update，allocator峰5479.3/14935.2MiB。受限临时2pair+2joint/224query/0env、参数弃用；6rank pair pack1→6为1.791→.281s，joint实际micro7→21为2.681→1.887s。每rank已用其全部合法质量，未靠重复样本填显存；复用旧1卡pack36/micro56与6rank吞吐证据。
仅从全部576新seen保存因子，按同原D一次CPU分解：真实BA=0.653263553=公共0.034604637+任务差异0.611402039+同task视频波动0.007256877；残差-3.04e-17。逐层/全部task见operator_readout.json，0新forward/GPU/环境/held参考；不是闭环上界或因果认证。

## 成本、教学与资源退出

新976 episode/92shard/976完整38r8 LoRA；256850control+9760dummy=266610步，unknown0/有效行重跑0/工程episode extra0/16。992提交/343840control/9920dummy/353760联合上界同时满足，无旧余额转入、无旧episode或LoRA填新结果。
累计GPUh=3.579776625832；分项={'profile': 0.06309000134468079, 'readout': 2.425849497914314, 'joint_training': 0.5901970199743907, 'pair_training': 0.5006401065985362}。实际9job全部exit，失败0。首次00:32:39.024Z至科学退出0.760518h；原90–150min与修正60–100min、原家族09:05:48.038均保留，验收/Git另计。
峰6卡/node峰{'GPU-node-A': 4, 'GPU-node-B': 6}；实际16/8及末2行有用wave吞吐{'8': {'shards': 55, 'median_episodes_per_second': 0.13065353062655546, 'median_native_plans_per_second': 8.260099518347017}, '16': {'shards': 29, 'median_episodes_per_second': 0.15168454404901088, 'median_native_plans_per_second': 9.177650521292195}, '2': {'shards': 8, 'median_episodes_per_second': 0.04824509927007986, 'median_native_plans_per_second': 4.000892426523517}}。每launch双node GPU/process/真实VRAM/CPU-PSS/data1 quota；训练单node NUMA/P2P-disable/deferredNCCL、独立eval无NCCL、0重探32、不触碰foreign。
32次完整runner.close，累计316个环境PID（非同时并发），单worker子树最多18process/164threads；最低实际node MemAvailable={'GPU-node-A': 280.8521423339844, 'GPU-node-B': 366.9206085205078}。批次进程和GPU最终释放另见resource_exit.json；无用idle renderer不驻留。
旧1800原始M/59114帧＋400 held M/14528帧只读共享；旧SQLite/状态不写，W/D不复制/重算，当前M payload/partial/reserved与旧共享总额≤1GiB（charged617130072bytes）。本批0高维prefix复制/producer545读取；新M消费/命中/帧当量详teaching_consumption.json，命中不是方法零教学。976新LoRA均从本批1180实际输出。

已付source/reference/Reader及旧§20 9.817228845821GPUh/281142步等成本独立保留；参考52.0141130901–52.1218585951GPUh、Reader家族13.0330865030GPUh、source32.5445405GPUh训练下界不改。依赖gross费用不能重复累加，也不是从零总成本；旧unknown/事故/完整CP与formal原件不动。
新ROOT最终2.696GiB，16GiB峰合同、实际du及atomic余量见resource_exit.json；未连续采样磁盘，不虚称精确高水。唯一1180，没有1090/controls/Test/自动后继；无论结果如何交main科学消费，继续loss/初始化/scale/seed/曝光补丁家族停止线有效。formal frozen保留，已集成开发树清理，clean pushed main与写窗口归还见git_delivery.json。


CPU算子总误差相对旧.773562降到.653264，晚十层q误差均值1.000105→.828561；仍有较大未保存能力，不能由梯度或新loss下降认证解决。阶段A真实BA末20=.759277（旧阶段A=.705336），后段终点读回才改善，不能把新L_pair直接当旧BA或归因到某一孤立时点。

target24参考新增270保82（旧31），全seen参考新增288保87（旧35）；source218保203、失15，identity79/80。held task顺序[3,6,11,16,23,26,31,39]为[32,9,38,0,0,38,11,0]；task26较旧42降4，其余七task合90/350（旧15）。相对Reader117为R78/G50/L39、MT153为R96/G32/L57。seen Object38有正例，Spatial35/Goal46/Long16不足及task17/25/36零成功保留；单点没有相邻稳定/视频因果资格。

物理pool原拟8卡，不是锁训练world6；第7/8个worker在实际launch前空闲数10、随后8，live上限降为6而拒绝，16→8预启动回退也未绕过上限。两项均0child/0env/0GPUh/0unknown，证据unlaunched_admission.json，实际峰6。最终node UID GPU进程0、zombies0，PSS GPU-node-A≈.312/GPU-node-B≈1.696GiB；其余Codex服务线程保留，不杀foreign。旧M index最后mtime00:11:23早于新派发00:32:39；新ROOT没有source_inputs/D副本或M payload目录。

正式训练所有rank实际M读取37440次（逻辑视频36720、joint分片导致部分完整generator重复），1229564帧当量；物化976次/33459帧当量，全部只读M命中、0新prefix/Phi forward。临时profile全局88次M读取按完整有用更新推导，仅rank0的14条细项留存，不冒称完整逐rank缓存journal。旧59114/14528已付采样帧仍是方法成本，20160 FM的原生policy图像/query计算另计。

本批scope完整结束，main消费后决定§21.5停止家族与解决方向；不自动恢复loss/初始化/scale/seed/曝光补丁，也不以本轮宣布所有hypernetwork/source-r8不可能。
