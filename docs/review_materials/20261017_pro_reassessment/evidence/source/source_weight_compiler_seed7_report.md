# §20完整1180与976读出交付

完整source权重条件生成器1180：seen **244/576**，strict validation **57/400**；未达到Reader900的334/117或MT153。训练域reference502、U36=484正例及旧Reader1800=413/111保留。本批无任务内实践、controls、Test或自动后继。

seen对source218：R205/G39/L13；对Reader900334：R221/G23/L113；reference新增288仅保35。target24为71/384（source48/ref315/Reader159），support12为173/192（source170/ref187/Reader175），五identity新执行80/80。breadth27/36；对照source22、reference36、Reader32。

validation对Reader117：R28/G29/L89，churn118、Jaccard.191781；对MT153：R40/G17/L113，churn130、Jaccard.235294。suite Spatial5/Object8/Goal42/Long2；Goal局部正例保留，task26独占42成功。不是普遍没有参数信用，也不能仅归因held分布：seen目标24同样尚未广泛获取。1180不称充分训练，不用代理误差认证迁移。

## 全部任务与能力得失

完整成功集合、所有臂RGL/churn/Jaccard/相邻历史重合见results.json。当前只有1180一个新点，未评1090，不能声称本候选相邻稳定。旧D/G含MT，ψ含实践；MT原缺初始proprio如实保留。

| seen task | group | source | reference | U36 | Reader900 | Reader1800 | new1180 |
|---:|---|---:|---:|---:|---:|---:|---:|
|0|target24|5|16|13|8|12|6|
|1|target24|0|16|16|2|12|0|
|2|target24|7|16|16|13|15|6|
|4|target24|0|14|11|8|11|0|
|5|target24|2|15|15|12|16|6|
|7|target24|0|16|15|14|15|0|
|12|target24|0|14|14|7|8|2|
|13|target24|0|13|11|1|0|0|
|14|target24|0|15|15|13|12|1|
|15|target24|0|14|14|0|12|0|
|17|target24|0|14|12|0|0|0|
|19|target24|2|14|14|8|3|3|
|20|target24|0|9|7|7|6|2|
|21|target24|14|16|16|13|15|13|
|22|target24|1|16|16|13|15|9|
|25|target24|0|10|10|0|6|0|
|28|target24|11|16|16|16|16|15|
|29|target24|0|7|7|0|5|1|
|32|target24|0|14|15|1|13|0|
|34|target24|4|7|6|3|5|2|
|35|target24|1|12|10|7|11|2|
|36|target24|0|10|11|3|12|0|
|37|target24|1|15|15|6|9|2|
|38|target24|0|6|5|4|3|1|
|42 identity|source_seen_support12|16|16|16|16|16|16|
|43|source_seen_support12|9|15|16|9|13|10|
|51|source_seen_support12|15|16|15|15|16|15|
|55 identity|source_seen_support12|16|16|16|16|16|16|
|56|source_seen_support12|14|16|15|15|14|15|
|62|source_seen_support12|16|16|16|16|14|16|
|64 identity|source_seen_support12|16|16|16|16|16|16|
|73|source_seen_support12|7|13|11|9|12|7|
|95|source_seen_support12|14|15|15|15|16|14|
|96 identity|source_seen_support12|16|16|16|16|16|16|
|97|source_seen_support12|15|16|16|16|16|16|
|101 identity|source_seen_support12|16|16|16|16|16|16|

| validation task | Reader900 | Reader1800 | MT | G | D72 | D144 | ψ180 | new1180 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|3|21|29|41|43|45|43|24|1|
|6|16|35|7|8|6|7|16|4|
|11|42|23|36|37|37|37|44|8|
|16|0|0|10|21|24|20|0|0|
|23|0|0|0|0|1|1|0|0|
|26|20|16|35|38|37|33|18|42|
|31|18|8|24|18|25|27|17|2|
|39|0|0|0|0|0|0|0|0|

## 学习、真实消费者与资源

4838160 fresh参数，冻结source1000/Φ900有序body；W/A0来自原source，S76只读旧合法单位，无U/code head/MT/H前向。1000BA＋180 FM/.1BA，实际36000＋720逻辑视频条件、20160跨episode queries；每task1020访问/560query。112 query分6rank，BA份额与全A/B余切均按实际query质量求和，未重复BA/除world。完整1000/1090/1180和rolling1170保留；Adam各参数step与scheduler真实承接1180，六rank CPU/CUDA RNG和全部流已核。详training_acceptance.json。

BA末20均值0.705336；joint末20 BA=0.779788/FM=0.118361。A/body在首零头后999个BA步骤及全部180 joint步骤有真梯度；仅证明信用链路，不作为控制成功。

物理比较5BA＋4joint/448额外FM，无env，临时参数弃用。1卡BA pack1/6/36为11.635/3.463/3.367s；6卡pack6为2.483s。joint1卡micro56为19.657s，6卡实际每rank至多21query为2.474s。正式均值BA2.721s/joint3.230s；峰CUDA allocator为5496/14950MiB。更大物理包已实际验证，未用重复样本填显存。

累计GPU **9.817228845821h**：M 0.806367209；profile 0.092475071；readout 3.290411293；BA_training 4.595045484；joint_training 1.032929790。19个实际job全部有exit；8个失败job（首M一个＋有效112后队列提交的7个）费用保留。父owner前三次exit1、最后538112 exit0，2026-10-17T00:12:25.660823Z。首次现场21:56:50→科学退出2.259898h；最初4–7h按实测下修，所有原时钟保留。验收/Git另计。

最终976真实新episode/92完整shard/976完整LoRA，271382control＋9760dummy＝281142；unknown0、额外episode0/16、有效行重跑0。112旧有效行在本ROOT CPU采用，未复用历史参照episode。原992/343840control/9920dummy/353760联合硬界同时满足。

实际峰7卡，node峰GPU-node-A=4/GPU-node-B=6；每launch两节点live，未触碰foreign。独立eval无NCCL，训练NUMA/P2P-disable/deferred；物理16/8/4与末2行有用波次，0重探32。76次fullclose，累计观察688个不同env PID全部退出（不是688并发）。单worker子树最多18进程/108线程；事件测得最低MemAvailable GPU-node-A 275.58/GPU-node-B 379.49GiB。最终两node UID GPU进程0、UID zombies0，PSS约GPU-node-A .314/GPU-node-B 1.699GiB，其余服务线程不乱杀。

异步worker PSS事件合计上界GPU-node-A约189.41/GPU-node-B约101.42GiB，不冒称同步精确峰。真实GPU启动快照见admission；allocator峰不包含全部EGL driver分配。最终ROOT3588620KiB≈3.422GiB，预算16GiB；全局M2200项charged617130072bytes<1GiB，无reserved/partial。原子临时已释放，磁盘未连续采样成精确高水。

## 教学读取与已付依赖

原1800 M/59114帧：首失败36ready后采用，后1764新build；失败/inflight原prefix读取帧区间[59114,59233]，费用未记零。原高维prefix只读不复制，producer545未读取。末976物化M命中576、新held构建400；held完整14528帧，raw RGB-only→纯图像prefix→M，无持久高维prefix文件/离线held标签/旧global head。缓存不能把方法教学成本说成0。

新root费用不是从零总成本。已付参考bank52.0141130901–52.1218585951GPUh/10870episode/2472414steps、Reader整家族13.0330865030GPUh/1952eval/506278steps；source原4rank训练32.5445405GPUh仅到最后更新下界，未把旧失败保存/恢复或generic pi05预训当0。旧单位/identity存档gross费用单列，不能重复加共享root。详dependency_cost_accounting.json；原数据/模型/参考/所有历史不动。

## 三次工程中断与来源

1. c7ef1372漏接已登记legacy纯图像cache，0更新/0env，0.047937221924GPUh和M帧不确定区间保留；同旧数学读取z_img/indices，stored L不读。
2. 8395a9e1完成1180后，seen环境44 union被误认为36；0env/0LoRA，按原面板投影consumer36/8，原run contract保持。
3. fb3eb80e完成112行后queue参数名不符；112行/52037步/0.455625407696GPUh全部保留；bd0e37b0实际SQLite consumer修复并CPU采用7shard，再执行864。

训练来源8395a9e1、静态输入c7ef1372、物化/评测fb3eb80e与bd0e37b0分别追溯。四formal frozen树保留；开发树集成后清理，最终clean pushed main身份见git_delivery.json。唯一owner/直接退出reader已结束，无Queue/第二waiter/自动后继。完整主讨论回执成功后归还canonical写窗口。
