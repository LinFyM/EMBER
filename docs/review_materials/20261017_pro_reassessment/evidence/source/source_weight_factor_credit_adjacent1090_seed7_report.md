# §22 已保存1090的完整相邻能力读出

唯一1090新读出：**seen290/576，validation121/400**；既存1180为306/128。1090→1180的seen为R253/G53/L37、验证为R97/G31/L24。末段有实际获得，也有大量交换；验证净增7主要来自task26的29→38，其余350例92→90，不能称广泛稳定提高或最终成功。

本批只读原§21/eadfe958的1090，0训练、Adam恢复、临时profile、额外model/action探针或环境smoke。976完整LoRA均从1090逐视频新生成，976 episode全新执行；1180和强参照只读比较。没有controls/Test/更多点/自动后继。

## 配对结果与保持

seen全部臂：{'Reader1800': 413, 'Reader900': 334, 'U36': 484, 'factor_credit1090': 290, 'factor_credit1180': 306, 'reference': 502, 'source': 218, 'source_weight1180': 244}；breadth={'Reader1800': 34, 'Reader900': 32, 'U36': 36, 'factor_credit1090': 31, 'factor_credit1180': 33, 'reference': 36, 'source': 22, 'source_weight1180': 27}。
source→1090：R206/G84/L12，churn96，Jaccard=0.682119。
reference→1090：R277/G13/L225，churn238，Jaccard=0.537864。
U36→1090：R271/G19/L213，churn232，Jaccard=0.538767。
Reader900→1090：R255/G35/L79，churn114，Jaccard=0.691057。
Reader1800→1090：R261/G29/L152，churn181，Jaccard=0.590498。
source_weight1180→1090：R226/G64/L18，churn82，Jaccard=0.733766。
1090→1180：R253/G53/L37，churn90，Jaccard=0.737609；success-set原件见results.json/adjacent_readout.json。

validation全部臂：{'D144': 168, 'D72': 175, 'G165': 165, 'MT153': 153, 'Reader1800': 111, 'Reader900': 117, 'factor_credit1090': 121, 'factor_credit1180': 128, 'psi180': 119, 'source_weight1180': 57}；breadth={'D144': 7, 'D72': 7, 'G165': 6, 'MT153': 6, 'Reader1800': 5, 'Reader900': 5, 'factor_credit1090': 5, 'factor_credit1180': 5, 'psi180': 5, 'source_weight1180': 5}。
Reader900→1090：R75/G46/L42，churn88，Jaccard=0.460123。
Reader1800→1090：R59/G62/L52，churn114，Jaccard=0.341040。
MT153→1090：R94/G27/L59，churn86，Jaccard=0.522222。
G165→1090：R97/G24/L68，churn92，Jaccard=0.513228。
D72→1090：R98/G23/L77，churn100，Jaccard=0.494949。
D144→1090：R96/G25/L72，churn97，Jaccard=0.497409。
psi180→1090：R81/G40/L38，churn78，Jaccard=0.509434。
source_weight1180→1090：R37/G84/L20，churn104，Jaccard=0.262411。
1090→1180：R97/G31/L24，churn55，Jaccard=0.638158；success-set原件见results.json/adjacent_readout.json。

target24 116→135；support12 174→171；identity80→79。参考新增288保存74→87，target270保存69→82，support18保存5→5。source原218保存206→203、损失12→15；两点都保留正例与能力损失。seen增量主要Object15→38，Spatial34→35，Goal51→46，Long16→16；不能把总体+16解释为所有任务持续增长。

验证task[3,6,11,16,23,26,31,39]：1090=[37,7,42,0,0,29,6,0]，1180=[32,9,38,0,0,38,11,0]。两点breadth均5/8，task16/23/39均0。1090对Reader117为G46/L42，对MT153为G27/L59；1180对Reader为G50/L39、对MT为G32/L57。pair相对旧57的正例保留，但弱57不是方法接受线，MT/G/D与Reader1800等强参照完整保留。

| panel / suite | source | ref | U36 | Reader900 | Reader1800 | old1180 | 1090 | 1180 | R/G/L 1090→1180 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|seen total|218|502|484|334|413|244|290|306|253/53/37|
|seen source_seen_support12|170|187|184|175|181|173|174|171|170/1/4|
|seen target24|48|315|300|159|232|71|116|135|83/52/33|
|seen libero_10|6|64|62|24|53|7|16|16|7/9/9|
|seen libero_90|170|187|184|175|181|173|174|171|170/1/4|
|seen libero_goal|26|74|72|49|63|40|51|46|43/3/8|
|seen libero_object|2|84|80|29|35|6|15|38|12/26/3|
|seen libero_spatial|14|93|86|57|81|18|34|35|21/14/13|

| validation suite | Reader900 | Reader1800 | MT | G | D72 | D144 | ψ180 | old1180 | 1090 | 1180 | R/G/L 1090→1180 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|total|117|111|153|165|175|168|119|57|121|128|97/31/24|
|libero_10|18|8|24|18|25|27|17|2|6|11|3/8/3|
|libero_goal|20|16|35|38|38|34|18|42|29|38|26/12/3|
|libero_object|42|23|46|58|61|57|44|8|42|38|36/2/6|
|libero_spatial|37|64|48|51|51|50|40|5|44|41|32/9/12|

| seen task | group | source | ref | U36 | Reader900 | Reader1800 | old1180 | 1090 | 1180 | R/G/L |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0|target24|5|16|13|8|12|6|7|10|[0, 7, 8, 13, 15]/[3, 4, 10, 11, 14]/[1, 6]|
|1|target24|0|16|16|2|12|0|1|5|[]/[3, 8, 9, 11, 12]/[10]|
|2|target24|7|16|16|13|15|6|11|8|[0, 4, 6, 9, 10, 13, 14]/[7]/[2, 8, 11, 12]|
|4|target24|0|14|11|8|11|0|3|1|[]/[4]/[0, 14, 15]|
|5|target24|2|15|15|12|16|6|10|10|[0, 5, 8, 9, 10, 11, 12, 14]/[4, 7]/[13, 15]|
|7|target24|0|16|15|14|15|0|2|1|[14]/[]/[8]|
|12|target24|0|14|14|7|8|2|1|5|[]/[3, 8, 11, 13, 14]/[1]|
|13|target24|0|13|11|1|0|0|0|8|[]/[1, 2, 3, 4, 6, 10, 11, 13]/[]|
|14|target24|0|15|15|13|12|1|8|12|[2, 6, 8, 10, 11, 13]/[0, 3, 4, 5, 12, 14]/[1, 7]|
|15|target24|0|14|14|0|12|0|0|5|[]/[1, 5, 6, 9, 11]/[]|
|17|target24|0|14|12|0|0|0|0|0|[]/[]/[]|
|19|target24|2|14|14|8|3|3|6|8|[3, 6, 7, 10, 11, 15]/[12, 13]/[]|
|20|target24|0|9|7|7|6|2|6|5|[3, 9, 10, 14]/[1]/[2, 7]|
|21|target24|14|16|16|13|15|13|12|10|[1, 2, 4, 5, 6, 7, 9, 10, 12, 14]/[]/[11, 15]|
|22|target24|1|16|16|13|15|9|13|14|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15]/[11, 13]/[10]|
|25|target24|0|10|10|0|6|0|2|0|[]/[]/[3, 11]|
|28|target24|11|16|16|16|16|15|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|29|target24|0|7|7|0|5|1|2|1|[3]/[]/[5]|
|32|target24|0|14|15|1|13|0|0|1|[]/[6]/[]|
|34|target24|4|7|6|3|5|2|5|6|[2, 6, 14]/[5, 8, 13]/[1, 3]|
|35|target24|1|12|10|7|11|2|4|4|[0, 14, 15]/[1]/[5]|
|36|target24|0|10|11|3|12|0|0|0|[]/[]/[]|
|37|target24|1|15|15|6|9|2|4|4|[2]/[5, 7, 9]/[0, 6, 14]|
|38|target24|0|6|5|4|3|1|3|1|[]/[1]/[2, 4, 8]|
|42 identity|source_seen_support12|16|16|16|16|16|16|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|43|source_seen_support12|9|15|16|9|13|10|9|9|[2, 4, 6, 7, 8, 9, 11, 12, 15]/[]/[]|
|51|source_seen_support12|15|16|15|15|16|15|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|55 identity|source_seen_support12|16|16|16|16|16|16|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|56|source_seen_support12|14|16|15|15|14|15|16|15|[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[1]|
|62|source_seen_support12|16|16|16|16|14|16|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|64 identity|source_seen_support12|16|16|16|16|16|16|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|73|source_seen_support12|7|13|11|9|12|7|7|5|[0, 3, 6, 8, 11]/[]/[12, 14]|
|95|source_seen_support12|14|15|15|15|16|14|14|15|[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[3]/[]|
|96 identity|source_seen_support12|16|16|16|16|16|16|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|97|source_seen_support12|15|16|16|16|16|16|16|16|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]/[]/[]|
|101 identity|source_seen_support12|16|16|16|16|16|16|16|15|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]/[]/[13]|

| validation task | Reader900 | Reader1800 | MT | G | D72 | D144 | ψ180 | old1180 | 1090 | 1180 | R/G/L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|3|21|29|41|43|45|43|24|1|37|32|[1, 6, 7, 9, 10, 12, 16, 17, 18, 19, 20, 21, 22, 24, 26, 29, 30, 31, 33, 34, 38, 39, 40, 42, 43, 45, 48]/[2, 14, 15, 32, 47]/[0, 3, 8, 11, 23, 25, 27, 35, 46, 49]|
|6|16|35|7|8|6|7|16|4|7|9|[6, 9, 21, 26, 41]/[1, 4, 24, 49]/[17, 36]|
|11|42|23|36|37|37|37|44|8|42|38|[0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 37, 41, 42, 43, 44, 45, 46, 47, 48]/[36, 49]/[1, 10, 18, 23, 28, 40]|
|16|0|0|10|21|24|20|0|0|0|0|[]/[]/[]|
|23|0|0|0|0|1|1|0|0|0|0|[]/[]/[]|
|26|20|16|35|38|37|33|18|42|29|38|[0, 1, 2, 6, 7, 8, 9, 12, 15, 17, 18, 19, 22, 24, 27, 30, 33, 34, 36, 38, 40, 41, 42, 46, 47, 49]/[3, 4, 14, 16, 21, 23, 25, 29, 35, 37, 44, 45]/[11, 20, 26]|
|31|18|8|24|18|25|27|17|2|6|11|[26, 33, 43]/[5, 19, 37, 38, 41, 44, 46, 48]/[13, 16, 36]|
|39|0|0|0|0|0|0|0|0|0|0|[]/[]/[]|

## 输入、消费者与工程身份

模型来源eadfe958ff17b0d1b4e1ff10265c234883b243b0，原schema ember_source_weight_factor_credit_r8_seed7_v1/stage joint/next_update1090；原manifest不改、不copy CP、不恢复Adam。readout producer d963e8c34199bd71a90b00ecf1843853c15cb086，new readout schema ember_source_weight_factor_credit_adjacent1090_readout_v1。模型4,838,160参数/完整38r8-alpha8-drop0/source1000/normalization/Φ900不变；新run身份和原训练身份显式分离。

原model.py/weights.py/video.py/cache.py无变；真实CPU消费者核1090来源、原始/held只读M、976官方映射、完整76tensor LoRA、queue adopt/unknown全额/有效拒重跑。CPU结构fixture明确来自既存1180，未冒充1090分数。当前owner/loader/readout/ledger窄改，旧training/profile/learning/checkpoint主动路径及算子几何入口退役到Git/frozen，净减615源码行/0新module/最大269行，无平行Writer或训练fallback。实际正式计算来自clean pushed detached d963e8c3。

每条件保持原seen36×16/root202610151与strict8×50/root7、video_schedule/registered scenes、state-env-policy RNG；official render256/model224、双相机rotate一次、state8/action7、native10执行5、dummy10、success即停/原horizon。最终只有source+本条件一套完整LoRA/exact L，Compiler/M/helpers退出，0交互适应/标签入条件。旧MT缺initial proprio限制、G/D含MT、ψ有实践身份均保留，历史臂不是匹配机制消融。没有新增几何/动作统计或模型forward探针。

## 费用、教学与资源退出

976有效唯一新行/976新完整LoRA/92shard；258437control+9760dummy=268197总步。unknown0/extra0/16/有效或正常失败重跑0；992提交/343840control/9920dummy/353760联合界同时满足。实际6job全exit0，GPUh=2.421583033535，全部为readout，0训练/profile。

首次现场2026-10-17T01:39:14.985237Z，owner01:44:26.915663Z→02:09:24.811810Z exit0（24.965min）；首次至科学退出30.164min，验收/Git另计。原35–65min估计与家族09:05:48.038、全部旧派发/事故/unknown/GPUh区间保持，不以resume重置。

峰6物理GPU，node峰{'GPU-node-A': 4, 'GPU-node-B': 2}；每launch两node真实GPU/process/VRAM与CPU-PSS及storage-node data1独立quota/shared均留原件，不碰foreign。dynamic long-first persistent独立evaluator无NCCL/无dummy占卡/无32重探。实际有用slots：{'8': {'shards': 53, 'median_episodes_per_second': 0.12993729770774978, 'median_native_plans_per_second': 8.13710055727463}, '16': {'shards': 30, 'median_episodes_per_second': 0.14680141408598413, 'median_native_plans_per_second': 9.164107768286033}, '4': {'shards': 1, 'median_episodes_per_second': 0.11551690478245881, 'median_native_plans_per_second': 4.491753735489169}, '2': {'shards': 8, 'median_episodes_per_second': 0.048976516688943525, 'median_native_plans_per_second': 4.298883609732832}}。16槽中位.14680episode/s、9.16411native plans/s；8槽.12994/8.13710，沿实际就绪/显存/CPU采用，末2槽为尾包，未扩科学行profile。

32次完整close、累计312个环境PID（非同时并发）全部退出；worker子树最多18进程/108线程。最低node MemAvailable={'GPU-node-A': 272.5397720336914, 'GPU-node-B': 366.0766067504883}GiB。终检owner/supervisor已退出、UID GPU进程0/两node zombie0；终检PSS GPU-node-A=0.313GiB/GPU-node-B=1.698GiB（含现有其它同UID服务，不当作本批残留）。

976次完整M消费全部只读命中，33459采样帧当量=seen18931+held14528；本批0新prefix/Reader forward/producer545/teacher离线标签。旧59114原始+14528held已付采样帧保留，缓存命中不是方法0教学；共享M charged617130072bytes≤1GiB，payload/partial/reserved统一计。新root无M/W/CP副本，旧CP/manifest/W/D/M索引mtime均早于本批首次，不写旧SQLite。详teaching_consumption.json/resource_exit.json。

ROOT终检du=2.422485GiB<新增6GiB合同；磁盘未连续采样，不虚称精确峰。旧§21 3.579776625832GPUh/266610步、§20 9.817228845821GPUh/281142步独立保留；source32.5445405训练GPUh下界、reference52.0141130901–52.1218585951GPUh区间、Reader家族13.0330865030GPUh为共享已付依赖，不能重复累加或把新root称从零总成本。

原件索引：run_contract.json/owner_exit.json/completion.json、analysis/results.json（全部原rows/配对success sets）、adjacent_readout.json、engineering_acceptance.json、resource_exit.json、teaching_consumption.json和各jobs launch/exit/fullclose/账本；完整CP仍在原§21 root，不复制。

## 交付界线

本批只回答1090→1180末段获得/保持/波动。相邻两点不足以证明持续稳定；不能按task选点、取union或按本结果自动追加1000/其它点/训练块/controls/Test。继承§21冻结Φ900+当前source-W共享行列头+36参考回归补丁家族停止线，科学后继由main消费完整原件裁决。开发树清理/最终main clean pushed与窗口归还见git_delivery.json；formal frozen d963e8c3保留。
