# §18固定ψ180完整验证迁移：整批交付

完整400/七臂2800逻辑行验收通过。新119/400 vs Reader900117/400，R105/G14/L12、churn26、breadth5/8；vs MT153为R87/G32/L66，仍低34。实践505→647/1600，独立final仅净+2，没有清楚广泛迁移，不自动追加训练、scale、controls或Test；主讨论据原件另行科学裁决。

|task|Reader900|Reader1800|MT153|G165|D72|D144|ψ180|
|---|---|---|---|---|---|---|---|
|3|21|29|41|43|45|43|24|
|6|16|35|7|8|6|7|16|
|11|42|23|36|37|37|37|44|
|16|0|0|10|21|24|20|0|
|23|0|0|0|0|1|1|0|
|26|20|16|35|38|37|33|18|
|31|18|8|24|18|25|27|17|
|39|0|0|0|0|0|0|0|
|总计|117|111|153|165|175|168|119|

Spatial37→40、Object42→44、Goal20→18、Long18→17；task3 +3、11 +2、26 −2、31 −1，task6有交换净0，16/23/39仍0。未新增task breadth。全部21对RGL/churn/Jaccard、原始success-set及每case/per-suite见results.json与closing_readout.json。旧MT缺initial_proprio，沿registered scene path/bytes和实际noise配对；有proprio的臂仍检查。

实际改参数105条件query51→53；295保留起点条件66→66，严格指定原Reader900 row复用。初始实践零177条件8→7、非零223条件109→112；首提案最终选中56条件29→31，后续选中49条件22→22，这不是后续相对首提案的已测因果比较。1873实际提案/126严格接受，1747拒绝及21中间接受共1768未query候选不推oracle；77条件按4/4止、323按完整面板预算止，0固定轮数。

固定外部CP180训练来源0b9b8080，Reader900/U/S/saved center/code_units/source38r8均冻结，0共享更新、0held离线标签、0Test、0其它arm。每condition自己的final init排除于其practice，condition间不共享H/代码/成绩；单final冻结后只执行saved完整LoRA，query不回H/选择。仍无匹配video/H因果或相邻checkpoint稳定资格。

## 费用与具名恢复

9213物理提交：9197完整有效（9092实践+105新final）/16原reserved_unknown。确切有效2515196步（control2423226+dummy91970），旧unknown仍保留[720,8480]，真实总区间[2515916,2523676]、保守收费2523676（control2431546+dummy92130）。旧872/440542完整记录逐行SQLite比对未变；0有效行重跑。16列名unknown均恰一次恢复、8045新确切步，原unknown不清零、不称已证实失败。原64工程extra用16/余48；原全局33264/3446720不增。

|具名condition|实际物理收费|有效完整实践|原unknown|授权pending cap|实际超8192|
|---|---|---|---|---|---|
|task_0031_state_028|9550|7430|2120|9830|1358|
|task_0031_state_029|7728|5608|2120|8192|0|
|task_0031_state_030|8196|6076|2120|8351|4|
|task_0039_state_031|8480|6360|2120|8480|288|

三项实际超8192合计1650，仅已保存pending收尾；唯一有效完整实践各≤8192。四项完成后自然按原8192减全部实际/unknown收费停止，没有扩充ψ remaining、再生成候选或再提案。完整authority、原pending/receipt、原run_contract和事故快照保持；16 named original+retry的实际身份及最多两次提交已验收。

当前ROOT共39.789931998783GPUh，17实际GPU job全部有exit（16为0、1为1），0未结GPUh区间；旧5.327250757880与本恢复34.462681240903分别保留。全过程实际同时最多6GPU，单node峰GPU-node-A=3/GPU-node-B=4（不同时间，不可相加）。初次未Popen卡无GPU job，另外两个真实失败owner/所有失败job费完整保留。不同worker的residence不按环境步数虚构每condition GPUh；旧§17/Reader/basis/reference及家族区间仍独立沿原件，不重计/清零。

原首次观察2026-10-16T13:16:44Z（精确消息到达时刻未恢复）；首次正式13:34:37Z，原中断/恢复13:45:28及14:56:28均保留，最终退出2026-10-16T20:45:06.719314+00:00。派发观察至退出约7h28m，在原6–12h预期内；恢复owner实耗5.811h。家族最初2026-10-15T09:05:48.038Z不重置。原run contract不覆写，新读取26c284ee、旧读取0e775e4b/0d9d1a00和训练来源分开。

## 吞吐、教学与生命周期

原始400教学初始编译只读复用，不等于0教学：70233原帧/14528 stride5+terminal帧的既有编译成本沿旧video_initial报告保持。本批44初始4/4条件无需再构建M；其余356真实完整M构建/13204帧编码，1873次memory读取（1517 hit）/59493帧使用当量。无held高维prefix落盘、无裁帧/降分辨率。H读取1333275 events，28512档读取/26671视觉build/1841 hit；参数与信息墙不因缓存改变。

|实际useful slots|packs|端到端plans/s中位|native plans/s中位|allocated峰MiB|reserved峰MiB|worker PSS峰GiB|
|---|---|---|---|---|---|---|
|1|6|1.125|2.576|9119.7|9464.0|7.68|
|4|53|2.853|5.723|9578.9|10868.0|16.19|
|8|92|3.920|7.051|10187.3|14730.0|26.64|
|16|460|5.436|8.693|11415.6|29552.0|47.76|

复用同一Runtime/真实EpisodeRunner，live capacity16/8/4，实际就绪单final或混合pack只用1..16真实槽，不补假行。共享显存下降时先完整close后重核/缩槽；CPU按每node实际PSS/环境预测共同准入，0NCCL、0重探32。frame32及H event512/image128沿既有有用吞吐证据；表内任务/H长度未配对，不当因果batch对照。没有为填显存扩大科学行。

完整H输入均消费后再退役：9092新H累计归档167.017GiB（累计写入，不是峰值）；事前8个init0全链raw保留2.701GiB，392其余condition完成冻结/final/逐事件facts原子归档且close后退役。facts长期1.403GiB，2273套实际完整controller（400原initial只读＋1873新候选）及全部rows/actions/mask/state/feedback/order/seed/码/提案/选择/费用保留。392不再可完整H replay或适应中exact-resume，最终LoRA仍可部署。旧raw/CP/模型不动，事故证据另保。

恢复段committed raw H事件采样峰12.639GiB，未触44GiB准入边界；最终ROOT9.440GiB。H缓存清空可再生payload、控制文件0.51MiB；M约104.4MiB，分别≤16GiB/1GiB。96GiB是含atomic/cache/candidates的准入峰界，当前du/committed样本不是OS连续峰值实测。新增全部data1，storage-node独立quota/个人实耗/shared现场原件在admission及resource_exit。

678次完整close/9097环境子进程释放回执；双node本批owner/GPU/env均无残留、各UID zombie0，foreign不触碰。退出uid PSS GPU-node-A=0.383GiB、GPU-node-B=1.712GiB（包含其它自己的服务，不混两主机内存）。资源事件worker峰约47.76GiB；node最小MemAvailable分别约317/310GiB，node实际准入不以RSS重复相加。

## 原件与Git

主要入口：results.json（全部七臂原始行/完整条件/21对）、closing_readout.json（独立CPU逐case复算）、engineering_acceptance.json（400/2800/预算/档案）、cost_accounting.json、useful_throughput.json、resource_exit.json、git_delivery.json。原始输入与原pending裁决在analysis/engineering_attempt002，原16/872及两旧owner/冻结树不抹写。正式来源clean pushed detached26c284ee；开发树/本地分支已验证集成后清理，旧三冻结原件保持。

当前有限批结束，无自动后继。验收、记录/Git完成后一次回主讨论01a13dd9，归还canonical工程/科研记录/Git写窗口；由main科学消费后接续。
