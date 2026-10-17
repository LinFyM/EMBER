> 既有报告摘存；来源：`runs/source_policy_space_compiler_20261015/incumbent_compilation_seed7/analysis/engineering_completion.md`。历史执行状态不构成当前授权。下文未入Git的原目录链接改为文字引用；可读结果投影见[证据地图](../../EVIDENCE_MAP.md)。

# §16视频初始控制保留与任务内参数改进：完整交付

实际72condition/288逻辑query、七臂1008逻辑case已完成并验收；没有新的共享训练、Reader/teacher编码、held/Test或自动后继。
video86/144低于原Reader90089；no-video59对source58。video实践83→100，改变参数的12任务query35→32；no-video实践52→63，改变参数9任务query12→13。当前完整组织没有把实践改善变成整体视频控制增益，按§16停止本固定单位局部搜索的自动加量/调参。
video相对no-video仍有净27，但主要保留了初始视频控制优势；no-video只从source起步，不是充分训练的language-only Compiler。本批不能证明H语义纠错/held能力或普遍否定交互学习。

## 完整配对结果

|面板|case/臂|source|reference|Reader900|first_best|full|video|no_video|
|---|---|---|---|---|---|---|---|---|
|全部144|144|58|127|89|73|79|86|59|
|target24|96|14|80|43|29|35|41|14|
|source-seen support12|48|44|47|46|44|44|45|45|
|五identity|20|20|20|20|20|20|20|20|
|libero_10|24|3|15|5|2|2|4|2|
|libero_90|48|44|47|46|44|44|45|45|
|libero_goal|24|8|20|14|14|14|13|7|
|libero_object|24|0|21|9|1|3|10|0|
|libero_spatial|24|3|24|15|12|16|14|5|

breadth（至少1/4成功）：{'Reader900': 29, 'first_best': 23, 'full': 25, 'no_video': 19, 'reference': 35, 'source': 20, 'video': 30}。参考新增70保留Reader29/video28/no-video3/first18/full23。全部21对RGL/churn、task成功集合/交并/Jaccard、strata和参数位移见`results.json`（原运行目录：`results.json`）。

|比较|before→after|R|G|L|churn|Jaccard|
|---|---|---|---|---|---|---|
|Reader900_to_video|89→86|84|2|5|7|0.923077|
|source_to_no_video|58→59|56|3|2|5|0.918033|
|video_to_no_video|86→59|58|1|28|29|0.666667|
|full_to_video|79→86|72|14|7|21|0.774194|
|first_best_to_video|73→86|68|18|5|23|0.747253|
|source_to_video|58→86|56|30|2|32|0.636364|
|reference_to_video|127→86|84|2|43|45|0.651163|

Object12/14均保留原Reader的4/4，当前video Object9→10的增量来自task15的一例；Object13/17仍0/4。Long5→4，task37实践0→3但query2→1；Spatial15→14、Goal14→13、support46→45。
video失去：task5/init4、task7/init0、task21/init8、task37/init4、task56/init4；新得task7/init12、task15/init12。no-video新得task2/init4、8及task51/init4；失去task34/init4、task22/init4。全部不利case见`query_changes.json`（原运行目录：`query_changes.json`），不以同分遮掩case交换。
当前video有12task/14次严格接受、120提案；24task保持初始。no-video9task/10次接受、122提案；27task保持source。video17/36四例全成功停（其中10初始即满）；no-video12/36（11为原source初始）。其余均剩余额度不足完整worst-case面板停，单condition最大8120≤8192。没有固定候选轮数。

## 执行、复用与信息墙

先完整实践确定性起点，FP32注册epsilon_j两臂配对，围绕实际incumbent+.25epsilon；严格成功数提高才移动中心，所有平局保留。候选CPU原子物化完整38/r8后实际执行；最终直接用同一已完整实践的文件，未重新生成/融合/回退。逐候选center/seed/epsilon/接受曲线与H链接保留在conditions及`candidate_trajectories.json`（原运行目录：`candidate_trajectories.json`）。
no-video只读授权source panel0/初态RNG，不含video字段；适应输入不读旧full/query/R/best/reference。query只在冻结后打开：video96条固定Reader900复用＋48真实新query；no-video108条固定canonical source复用＋36真实新query。每臂144，参数未变复用不是学习。原source144实践/40377步仍占各自8192，11个source初始4/4不重跑。
video复用36条原v0真实教学编译，共1179个有序stride5含末帧原始teacher帧；原特征/完整LoRA/metadata路径逐task保留在cost_accounting.json。当前0新Reader forward和teacher编码不等于0教学成本；原prefix/Reader共享工作费用仍属§13，无法据此分出每video实际GPUh，不虚构归零或精确摊分。
所有RGB/proprio/实际动作/mask/feedback/native0/9 hidden保留；当前m更新只消费自身参数与完整四例成功数，H没有进入语义网络。没有FM/BC/PG/动作轨迹拟合、meta9/H权重、MT、新expert/basis或其它视频control。
actual source checkpoint/basis/900/code_units全只读；依据真实原子账本核对1196个唯一case各1attempt、72条完整路径、45个最终非source完整文件及同实际实践文件身份。1112份新H和144份旧H引用齐全；两份真实成功/失败H验证了完整双相机256uint8/state8/native hidden[2,50,1024]。没有逐tensor低位/hash检查。

## 费用与优化

|臂|逻辑practice episode/step|旧practice复用|新practice episode/step|新query episode/step|全逻辑query step|提案/接受|
|---|---|---|---|---|---|---|
|video|624/161277|0/0|624/161277|48/9977|34504|120/14|
|no_video|632/184106|144/40377|488/143729|36/9467|39139|122/10|

新物理1196提交＝1112practice＋84query，339成功/857正常科学失败；312490control＋11960dummy＝324450步；**3.231518597537GPUh**，6job全exit0。工程失败0、unknown0、额外额度0/16、有效重算0；全部小于5536/658087control/55360dummy/662567联合及每condition8192。失败没有当作bug或重抽。
动态混臂pack不能精确按arm拆GPUh。专臂pack实占wall给下界，另一臂专用wall从全批扣除给上界：video[0.490161,3.155541]、no-video[0.075978,2.741357]，非可相加区间。混臂2.361820与加载/其它wall0.303560共享保留，不按step比例假称实测。
旧source实践40377步及复用204query物理成本不重计；§14原9.325162564582294GPUh/776提交/207328步、§15原0.36454247693220776GPUh及32unknown的[31211,44536]总步区间独立保留，旧source/reference/144/1170/linear64/video及更早区间也不重置或归零。详见cost_accounting和historical_five_arm_costs。
本批93个有用pack：{'4': 11, '8': 16, '12': 12, '16': 52, '24': 2}。16槽端到端中位7.623（native9.280），24槽实测两pack中位7.222（native9.626），当前按端到端选择16；24的allocated峰12586MiB/reserved32504MiB。更小8/12/4用于CPU合计准入或尾部剩余完整panel，未加profile行，也未重探32的EGL边界。不以最低显存为目标；显存空不扩大科学矩阵。
H实际20.603GiB/主要原张量33.206GiB，压缩比0.620；按最终实际practice速率外推授权最大新H约37.11GiB。归档每episode线程工作wall p50=1.289s/p95=2.319s；该并行线程工作和不是pipeline wall。当前无dense-H缓存/额外常驻actor/source复制。
ROOT实耗21.276GiB，另计有界原子临时与开发/冻结小树仍低64GiB，未声称测了OS磁盘峰。首、末storage-node data1独立quota+shared检查保留；末used1935201028KiB/soft2147483648KiB。

## 时钟、资源与代码交付

实际接收06:33:17.631Z，准备首06:33:43.357Z，owner376918于2026-10-16T06:54:55.814143+00:00启动，2026-10-16T07:29:27.207260+00:00真实exit0；执行约34.52min，派发至实际退出56.16min，CPU/Git验收另记。原估1.5–3h未超期；09:05:48.038参考首时钟及所有历史保留。
单owner/直接退出reader9269→cell748已返回，0Queue/自通知/心跳/第二waiter。六GPU（node各3）每launch live双节点/真实显存/process与quota准入，cap6/node6；pure evaluator无NCCL，source正常BF16/TF32，未杀/重置foreign。
35次完整runner.close共释放360个环境子进程；6job全实际退出、双节点本批GPU/owner/env均0、无zombie。UID PSS退出GPU-node-A约0.325GiB/GPU-node-B约1.225GiB；node分别核算，第一有用PSS快照和worker峰值/准入全保留，未把异步各worker峰之和当真实同时node峰。
正式代码56c4dc99b1be4f6d3517f49dc8f8dbd3a20d26aa，detached frozen incumbent_compilation_frozen_20261016保留。新代码已集成push；唯一control_space复用D/native/History/SQLite/atomic账本，旧FM/PG/learned-H active五模块随当前合同退役到Git/旧冻结，新增source模块0/净减782行、Architecture0hard。
task-owned两个开发worktree及分支仅在clean/已集成验证后清理，旧冻结、旧formal原件与foreign树保留。最终main clean/pushed身份及canonical记录写窗口释放见git_delivery.json；完成回报一次发主讨论01a13dd9，由其科学消费接续，本批无自动400/更多搜索或新许可停点。

验收：`acceptance.json`（原运行目录：`acceptance.json`）；费用：`cost_accounting.json`（原运行目录：`cost_accounting.json`）；吞吐/H：`useful_throughput.json`（原运行目录：`useful_throughput.json`）、`raw_history_cost.json`（原运行目录：`raw_history_cost.json`）；资源：`resource_exit.json`（原运行目录：`resource_exit.json`）；数据：`results.json`（原运行目录：`results.json`）、`report.md`（原运行目录：`report.md`）。
