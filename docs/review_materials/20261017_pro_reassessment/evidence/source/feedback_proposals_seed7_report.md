# §14第一轮完整交付

36条完整编译链、9次真实共享更新、144独立query和1008跨episode FM已结束；exit0。固定source/full38r8/basis/units，从Reader900加fresh H/Adam；无MT、held、controls、Test或自动后继窗口。

## 控制读出与适用范围

当前144为不同训练theta下的完整路径回报，不能当single-checkpoint评估。source58/reference127/generated79；source→generated R53/G26/L5、churn31、breadth20→25；参考新增70保23。五reference identity任务20/20全部source，不能称学习。
实际改变参数17task的68query：source13→generated35，R11/G24/L2。19task最终选source的76query45→44（R42/G2/L3），这些未改参数差异不归因学习；其中11task source实践4/4即止且0教学编码，另8读过教学仍选source。
唯一同视频旧900锚为init0的36行：22→19，R18/G1/L4、churn5；另外108行teacher与旧900不同，不作严格900配对。
实践source52→best90/144，R51/G39/L1；122次相邻面板R84/G67/L41。选择看practice；全部query只供共享回报，未入H/best。实践改善不等于独立初态或H/video因果已成立。

|分组|source|reference|generated|cases|
|---|---:|---:|---:|---:|
|target24|14|80|35|96|
|support12|44|47|44|48|
|libero_10|3|15|2|24|
|libero_90|44|47|44|48|
|libero_goal|8|20|14|24|
|libero_object|0|21|3|24|
|libero_spatial|3|24|16|24|

## 全部condition

|task|meta theta|candidate成功数/4（source在首）|selected|query/4|practice steps|
|---|---:|---|---|---:|---:|
|0|4|[1, 3, 2, 2, 3, 3, 3, 2, 2, 3, 1, 2]|task_0000_v0_practice_001|3|7856|
|1|2|[0, 0, 1, 0, 1, 0, 1, 0, 0]|task_0001_v0_practice_002|0|7914|
|12|7|[0, 1, 2, 0, 2, 1, 2]|task_0012_v0_practice_002|1|7533|
|13|0|[0, 0, 0, 0, 0, 0, 0]|task_0013_v0_practice_000|0|8120|
|20|4|[0, 3, 3, 3, 3, 3, 3, 3, 3, 3]|task_0020_v0_practice_001|3|7452|
|21|5|[4]|task_0021_v0_practice_000|4|835|
|32|3|[0, 0, 0]|task_0032_v0_practice_000|0|6360|
|34|5|[1, 1, 1, 0]|task_0034_v0_practice_000|0|8036|
|2|3|[0, 4]|task_0002_v0_practice_001|4|1386|
|4|8|[0, 1, 2, 1, 1, 2, 1, 2, 1]|task_0004_v0_practice_002|3|7629|
|5|4|[0, 0, 2, 1, 1, 1, 1, 2, 1, 0]|task_0005_v0_practice_002|3|8074|
|7|2|[0, 3, 2, 1, 1, 2, 0, 0, 2]|task_0007_v0_practice_001|3|7378|
|14|4|[0, 2, 1, 3, 4]|task_0014_v0_practice_004|1|5303|
|15|7|[0, 0, 0, 0, 0, 0, 0]|task_0015_v0_practice_000|0|8120|
|17|1|[0, 0, 0, 0, 0, 0, 0]|task_0017_v0_practice_000|0|8120|
|19|1|[0, 0, 2, 0, 2, 2, 1]|task_0019_v0_practice_002|1|7464|
|22|7|[1, 4]|task_0022_v0_practice_001|3|1828|
|25|5|[0, 0, 0, 0, 0, 0]|task_0025_v0_practice_000|0|7440|
|28|1|[4]|task_0028_v0_practice_000|4|490|
|29|5|[0, 0, 0, 0, 0, 0]|task_0029_v0_practice_000|0|7440|
|35|6|[0, 2, 1, 0]|task_0035_v0_practice_001|0|7423|
|36|7|[0, 1, 1, 1]|task_0036_v0_practice_001|0|7873|
|37|6|[0, 0, 0]|task_0037_v0_practice_000|2|6360|
|38|0|[0, 1, 0]|task_0038_v0_practice_001|0|6269|
|42|3|[4]|task_0042_v0_practice_000|4|602|
|43|8|[2, 3, 2, 4]|task_0043_v0_practice_003|3|4214|
|51|6|[3, 4]|task_0051_v0_practice_001|4|1239|
|55|6|[4]|task_0055_v0_practice_000|4|597|
|56|0|[4]|task_0056_v0_practice_000|3|528|
|62|2|[4]|task_0062_v0_practice_000|4|588|
|64|2|[4]|task_0064_v0_practice_000|4|535|
|73|3|[0, 3, 3, 1, 2]|task_0073_v0_practice_001|3|7082|
|95|0|[4]|task_0095_v0_practice_000|3|770|
|96|1|[4]|task_0096_v0_practice_000|4|425|
|97|8|[4]|task_0097_v0_practice_000|4|808|
|101|8|[4]|task_0101_v0_practice_000|4|713|

每个实际sample/μ/logp、decision prefix、原行为CP、完整632份原始H、各初态成功集合及所有相邻得失均保留conditions/和episodes/；精确分组见results.json与closing_readout.json。

## 梯度、流与恢复

9次均真实optimizer step且PG非零；全部122实际提案（含未选中）重放进score。beta首步固定0.00018755106293410052，之后不调。H body首步0符合零head，后8步非零（6.84e−5..4.56e−4）。FM loss约0.0885–0.1255，不能用来选点。
全路径行为均值重放最大差0.000244140625（sigma.25），logp最大差0.0023436546325683594；记录密度CPU校核差9.5367431640625e-07，保留正常BF16/物理执行差异，不称bitwise exact。
完整CP保存fresh Adam/scheduler、6rank RNG、beta/baseline、task-video-query原游标和topology；最终meta_update9，非伪936。36task visit100→101，1008合法query，全部当前teacher被排除。
81c859e6产生前3更新；无活子任务且全部GPU回执结算的meta3自然边界仅退休CPU owner，原attempt exit−15为工程迁移；7a149c3e clean pushed detached完整承接3→9。0有效重算/未知/科学字段变化，证据coordinator_handoff.json与两attempt。

## 有用工作吞吐与生命周期

真实35.6MB H档直接gzip seek读取3.188秒，一次解压0.434秒（实际loader .447秒）；改顺序解压和5/6worker pack分配，无热改。只缓存冻结raw H4GiB，dense全批16GiB上界；行为增量表示只在固定theta路径内，训练全量重算。
|meta|episode workers|env steps/s|train ranks|train秒|实际FM batch|H chunk|GPU allocated GiB|
|---|---:|---:|---:|---:|---|---:|---:|
|1|1|21.99|1|124.16|[7]|16|11.07|
|2|2|29.15|2|79.18|[14]|32|13.01|
|3|4|22.79|4|72.56|[28]|64|16.34|
|4|5|35.58|6|15.22|[14, 21]|128|14.69|
|5|5|36.50|6|19.15|[14, 21]|256|14.63|
|6|5|54.84|6|17.23|[14, 21]|512|19.43|
|7|5|42.25|6|15.14|[14, 21]|512|19.44|
|8|5|51.74|6|25.60|[14, 21]|512|19.41|
|9|5|20.53|6|17.78|[14, 21]|512|19.38|

6总卡准入时为5 episode worker+1真实提案consumer，训练改同node6rank。对1/2/4/5worker、1/2/4/6rank及H16..512都使用本批有效工作；后段5worker记录20.53..54.84step/s，训练15.14..25.60秒，无OOM。任务/H量及I/O版本不同，不声称隔离speedup或全局最优。
FM请求ceiling至112，但6rank分担固定logical112后每rank真实14/21条；不能把ceiling当实际112大batch。行为H chunk512已覆盖完整新面板；训练chunk增长到512，峰19.44GiB allocated/28.95GiB reserved。后段单更新15–26秒，评估8–14分钟主导；未为再扫物理组合中断近完成窗口。

55个GPU job全部exit0；240次完整runner.close均无残留，双节点本ROOT CUDA/owner/env均退出（resource_exit.json）。退出node用户总PSS包括app/CPU验收进程，不等于全部是本实验。全部foreign untouched。

## 成本、存储、时钟与Git

776物理episode（practice632/query144）；control199568+dummy7760=207328，正常策略失败429份practice H完整保留；0工程episode失败、unknown、有效行重跑或OOM。独立16工程额度未使用，未挪旧额度。
本ROOT完整9.325162564582GPUh，55退出回执齐全，无未决GPUh区间。旧五ROOT[69.83061998896142, 69.9383654939614]GPUh独立保留，包含原已确认区间，不强记0或重算旧费用。
632原始H压缩11.540GiB/primary18.604GiB，ratio0.6203；ROOT实占11.946GiB。最终单调原始量+16GiB临时dense+全部新code/原子余量的保守峰界30GiB<48（非连续实测峰）。storage-node独立data1 quota及共享容量见resource_exit。
接收02:25:27.046Z、准备02:25:59.716Z、首次owner03:04:28.505672Z、恢复03:53:39.218827Z、退出2026-10-16T05:08:33.734988+00:00；接收到退出2.719h，首次owner到退出2.068h，落在2–4h原预计内。旧参考2026-10-15T09:05:48.038Z不重置。
正式代码81c859e6/7a149c3e与原冻结树保留；main最终clean/pushed身份和task-owned开发树清理见Git_delivery.json。当前授权首窗结束，只向main01a13dd9一次交整批，归还记录写窗口；无自动18/36、held、controls、Test或后继实验。
