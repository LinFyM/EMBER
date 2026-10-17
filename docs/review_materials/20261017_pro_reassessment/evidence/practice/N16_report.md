# N16有限恢复：整批工程验收与完整读出

验收记录UTC：2026-10-15T06:23:57.172691+00:00。本报告只消费已保存原件，0新模型/环境/GT/sealed/held/Test计算。

本批完整512逻辑行：MT67、video0 61、video1 71、no_video62（各128）。实践成功三臂均提高，但两个正确视频没有一致新初态收益；video1局部净增保留，不能由这一有限批宣布方法成功、充分收敛或否定全部自身交互学习。后续科学判断交新主讨论，不自动扩量/调参/换架构。

## 执行、恢复与退出

唯一owner practice_support16_resume_02/PID1360022，2026-10-15T05:00:51.161033Z启动，06:15:43.573128Z真实exit0，六job attempt_002均exit0。计算74.874分钟；直接等待其真实子进程完成，无第二waiter、自Queue、阶段通知或固定成绩轮询。
原首次派发2026-10-14T23:06:33.739733Z至退出为7.153小时，原2–4小时已超期事实不重置；有限恢复90–150分钟含准备/验收的估计另列，不能用74.874分钟掩盖历史停机。
冻结运行树workspace_assets/ember_worktrees/video_effect_n16_recovery_frozen_20261015，clean pushed detached78d4f73e；src与原7a146fc0相同。Phi训练8801e701/1120和原16预测不变。旧attempt/冻结树/无exit原件保留，不以主机重启单独解释更早00:25停滞。
原743个有保存row请求没有重跑；原71个未回执身份各实际恢复一次。新的3305提交全部3305返回，无新失败/未知。71仍是旧未决上界，不是证实71次完整失败。原18485未知control不由新结果回填。

## 完整原件与配对

24完整38-target/r8终点、76叶因子/Adam、流/RNG均齐；原6中断CP的真实FullFactors/Adam恢复、71所属保存draw及原预算消费者见inputs/interruption_recovery_20261015/consumer_acceptance.json。正式运行完成全部24首native完整单元、48原子更新，四格分母4、外层16；物理chunk4/8/16保持同一逻辑16均值，最多80信赖点。
56队列项全部complete；所有24适应终点冻结后再最终评测。512唯一task/arm/init行，每臂128；task/state/env seed/policy root/首native noise/初始proprio逐行与固定MT配对。init16..31实践与init0..15评测分开，最终反馈未用于更新/接受。
新实际final272行，240为预注册固定MT引用：MT128与7个未改参数condition的112行。MT来源唯一是父corrected_results.json中的官方128/67，不用旧no_video74挑高点；未改参数的7个condition不能称学习或遗忘。

## 总体、覆盖与成功集合

|臂|新初态成功/128|任务覆盖/8|更新/接受|实际改参数条件/8|实践前→后/128|
|---|---:|---:|---:|---:|---:|
|video0|61|8|16/7|6|66→77|
|video1|71|8|16/6|5|66→72|
|no_video|62|7|16/8|6|66→80|
|MT|67|8|0/0|0|66（共享实践起点）|

|candidate相对reference|保持|得到|丢失|净值|churn|Jaccard|
|---|---:|---:|---:|---:|---:|---:|
|video0_vs_MT|52|9|15|-6|24|0.6842|
|video1_vs_MT|62|9|5|+4|14|0.8158|
|no_video_vs_MT|56|6|11|-5|17|0.7671|
|video0_vs_no_video|50|11|12|-1|23|0.6849|
|video1_vs_no_video|53|18|9|+9|27|0.6625|
|video1_vs_video0|55|16|6|+10|22|0.7143|

两视频共有55个成功，video1比video0得16失6。只部署各condition唯一终点，没有相邻checkpoint的最终评测或checkpoint/video union；不能声称稳定性已充分建立。

|task|MT|video0|video1|no_video|
|---|---:|---:|---:|---:|
|0|12|11|15|12|
|1|4|3|4|3|
|12|8|8|8|7|
|13|7|7|7|7|
|20|10|10|10|9|
|21|15|15|15|15|
|32|2|2|5|0|
|34|9|5|7|9|

|suite（各32）|MT|video0|video1|no_video|
|---|---:|---:|---:|---:|
|libero_10|11|7|12|9|
|libero_goal|25|25|25|24|
|libero_object|15|15|15|14|
|libero_spatial|16|14|19|15|

video1净+4来自task0 +3、task32 +3与task34 −2，保留局部收益及负例；task1/13/20为未改MT引用。video0 task34由9降5，task0/1各−1，不能被均值或task32实践改善掩盖。no_video的6个改参数condition仍整体−5，实践66→80不能等同可迁移收益。
旧N4完整64/70/MT67/noV74及实践34/64→45/64保留；本批61/71/MT67/noV62不能解释为纯N因果差额：支撑/预算/最大更新机会改变，且旧noV74全MT的执行波动与本批预注册MT复用不同。192请求下每condition实际仅两次更新，未测充分优化。内部8训练task不等于新held任务迁移或strict paired400资格。

## 全部24个condition（每个实践分母16、评测分母16）

|task|arm|更新/接受|实践前→后|最终成功|MT未变|停止|
|---|---|---:|---:|---:|---|---|
|0|video0|2/1|12→12|11|否|budget|
|0|video1|2/1|12→12|15|否|budget|
|0|no_video|2/1|12→13|12|否|budget|
|1|video0|2/1|3→6|3|否|budget|
|1|video1|2/0|3→3|4|是|two_no_acceptable_proposals|
|1|no_video|2/1|3→4|3|否|budget|
|12|video0|2/1|6→8|8|否|budget|
|12|video1|2/1|6→7|8|否|budget|
|12|no_video|2/2|6→9|7|否|budget|
|13|video0|2/0|11→11|7|是|two_no_acceptable_proposals|
|13|video1|2/0|11→11|7|是|two_no_acceptable_proposals|
|13|no_video|2/0|11→11|7|是|two_no_acceptable_proposals|
|20|video0|2/0|11→11|10|是|two_no_acceptable_proposals|
|20|video1|2/0|11→11|10|是|two_no_acceptable_proposals|
|20|no_video|2/1|11→13|9|否|budget|
|21|video0|2/1|14→14|15|否|budget|
|21|video1|2/1|14→14|15|否|budget|
|21|no_video|2/0|14→14|15|是|two_no_acceptable_proposals|
|32|video0|2/2|3→5|2|否|budget|
|32|video1|2/2|3→4|5|否|budget|
|32|no_video|2/1|3→4|0|否|budget|
|34|video0|2/1|6→10|5|否|budget|
|34|video1|2/1|6→10|7|否|budget|
|34|no_video|2/2|6→12|9|否|budget|

总48更新/21接受；17预算停止、7连续两次无可接受提案停止。每condition逻辑请求160或176，未因物理共享/复用买更多更新。完整每更新draw/分支score/native credit/trust/candidate/接受及H路径仍在conditions/*/update_*_outcome.json与checkpoint；不另算新GT或信用探针。

## 费用、上界与资源

本ROOT全attempt物理4119提交、4048已返回；旧814/743/71与新3305/3305分列。已知656987 control+9760 dummy=666747步；旧未决control≤18485、dummy0保留，保守总685232。上限5079提交/1167285 control/50080 dummy/1217365总步均满足。原正常工程失败16未新增消费，71一次性恢复不滚动扩充。
旧依赖2184提交/451706动作步/7.005042365GPUh独立保留；加总6303物理提交、6232返回，已知1118453步、保守1136938步。不得把旧费用清零或把复用逻辑行当新物理计算。
本次恢复精确7.459967182GPUh。原无exit批GPUh区间[2.346457453,21.469853089]，由各launch到最后可证活动为下界、至03:30:32Z开机为上界；不写伪造精确退出。ROOT区间[9.806424635,28.929820271]；加旧依赖区间[16.811467000,35.934862636]。
六卡实际GPU-node-B:1/7/2/0/4/6，每launch双节点总6/node6与storage-node data1 quota现场准入。注册wave真实slots16/32均执行，原件最高reserved 35.982GiB；没继续扩大的live显存/吞吐依据在physical_slot_selection。不同wave工作长度/冷启动会影响比较，不把单次比较当普遍最优。native实际4/8/16物理chunk见update_and_resource_summary.json，0新增环境profile。
storage-node完成时du ROOT=48149954560B（44.843GiB），恢复前8.453GiB；低于本批新增80GiB预估，完成时大小不是连续实测峰值。原quota与相关目录/共享容量准入证据在inputs/interruption_recovery_20261015。

## Owner提出的主机内存/进程问题

06:14:39Z整机used412GiB、available91GiB、swap5.8GiB；可读进程的PSS抽样约306.86GiB，主要是每个数GiB的独立仿真环境，CPU私有匿名内存占大头。抽样读到91进程/2263线程（其中78环境spawn进程；非完整连续计数），环境进程各约24线程，界面线程条目会进一步放大数量。未发现第二实验owner。
源码证实：每slot一个spawn进程；每进程只缓存当前一个task的环境，切task先close，不是多task无界缓存。队列尾部只余两个final时，4个空闲worker未关闭EpisodeRunner，仍保留环境/模型直至全56项完成。线程不能全部归于已唯一定位的某个BLAS/EGL库，每进程GB也尚未分解到分配调用栈；单次快照不证明持续泄漏。
整批正常退出后owner/六模型worker/环境子进程全部释放，主机used降17GiB、available485GiB、swap125MiB；本用户CUDA0，foreign进程未触碰。CPU驻留峰值/线程总数与idle释放未被本次启动检查充分覆盖，是工程遗漏。后续若复用这套运行面，先处理空闲池释放和CPU内存并行预算；本批结果/冻结代码未热改，不自行补跑profiling或新科学计算。

## 原件与验收入口

- results.json、aggregate.json、final_freeze.json、complete.json：唯一整批读出/冻结/完成。
- coordinator/practice_support16_resume_02/exit.json与jobs/*/attempt_002/exit.json：真实退出与逐job费用。
- 本目录acceptance.json、event_and_pairing_acceptance.json：24CP/512配对/71各一次及743未重跑验收。
- 本目录conditions.json、paired_metrics.json、per_task.json、per_suite.json、update_and_resource_summary.json：全部分布、得失及更新。
- 本目录host_memory_diagnostic.json：内存排查、源码定位与释放证据边界。
- inputs/interruption_recovery_20261015/manifest.json及原contract/queue/owner快照：一次有限恢复额度、71身份和旧GPUh区间。

所有科研正负证据、旧GT/sealed时序例外保留；本轮0新GT/sealed/held/Test与Phi训练。工程交付后由[operational-id]消费并承担科学接续；执行者不自动开启任何后继。
