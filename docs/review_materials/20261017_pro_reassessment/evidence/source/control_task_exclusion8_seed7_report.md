# §19整批交付：任务排除后控制保留不足

一次28-task/208-point独立S28/U64拟合、冻结后八selected参考投影、128新query及四臂512逻辑行全部完成。U28为37/128，对source23、原U36=98、完整参考105；breadth4/8（原U36及参考8/8）。reference新增83仅保14（U36保72），source保持20/新增17/丢3；对U36保持34/新增3/丢64，churn67、Jaccard0.336634。

|task|source|reference|U36|U28|
|---|---:|---:|---:|---:|
|0|5|16|13|10|
|1|0|16|16|0|
|12|0|14|14|6|
|13|0|13|11|0|
|20|0|9|7|0|
|21|14|16|16|14|
|32|0|14|15|0|
|34|4|7|6|7|

|suite|source|reference|U36|U28|
|---|---:|---:|---:|---:|
|libero_10|4|21|21|7|
|libero_goal|14|25|23|14|
|libero_object|0|27|25|6|
|libero_spatial|5|32|29|10|

四项任务1/13/20/32均为零、不利case与成功集合完整保留。task34为7/16，超过U36的6且等于reference7；局部正例未清零。全部被测项是原train24中的八项，不是新的held split/400资格。

拟合严格排除FIRST8全部63点，仅显式28/208（203非零+5 source identity）；每task1/28、selected半权/others另半均分。76有效tensor RMS独立重算，源saved center/layout固定，未减均值，原FP64小Gram公式与rank容差沿用；没有旧U/S/code_units或八task参数/score进入fit。数值rank203、有效64、谱保留0.9931953471；fit池加权因子相对误差0.006804653。冻结后才打开八selected与旧S36/C_l诊断：被排除八点在S36下因子相对误差中位0.5901969363，完整sBA/C36相对误差中位0.7233175039。内部低残差未替代闭环。

该完整S/U构造在本排除组的直接投影显著丢控制，降低其直接跨任务覆盖的支持度；S/U共同变化，不能唯一归因U或某列。欧氏因子投影不是子空间内闭环最优，阴性不证明所有64维或所有z均无解；局部保持不证明视频能推得代码或正式held迁移。无神经训练/FM/PG/局部优化/实践适应/教学/新reference/Test，0自动后继。

原source/reference/U36均由实际128原行验证（23/105/98），旧官方init资产/字节身份、actual initial proprio、绝对env/policy noise配对；原件没有registered_scene字段，未伪造旧MT场景。新8套完整38/r8、alpha8/dropout0，只部署saved LoRA，最终运行不读basis/reference/Reader/H。首次动作统计仅消费各臂已保存的实际首5×7动作：U28-reference MSE0.025052598、U28-source0.022798005、U36-reference0.0000972224；没有补50-action响应或新forward探针。

唯一CPU fit19.173189s，CPU峰RSS3.365GiB。首次现场2026-10-16T21:05:08.224595+00:00；成功恢复owner3979821启动2026-10-16T21:26:03.688621+00:00，退出2026-10-16T21:30:50.665006+00:00，恢复286.978s，派发现场至科学退出25.71min。原35–75min为估计，完整交付另计验收/Git；家族原2026-10-15T09:05:48.038不重置。

两次工程中断均无环境行：attempt001是JSON str→Path接口，attempt002是live空闲数变化使cap8→6、未启动拒绝类型未进入withdraw。真实384旧行及captured idle10/six-worker CPU消费者通过，新clean pushed frozen接续；没有热改、重新fit、重跑有效行。首轮8输出已保存，恢复采用原输出并重复一次CPU-only post-freeze投影/BA诊断，该重复计入owner wall，0policy probes。原launch/exit/log/错误、全部admission与12个实际GPU job exit长期保留，详见engineering_attempt001/002。

实际128提交、36500 control+1280 dummy=37780步，0unknown、工程episode额度0/16，未动144/52000联合上界。总0.446570666035GPUh含失败准备加载0.098994816674及成功读出0.347575849361，12 job全有exit且均0；owner两次工程exit1单列，不能记成GPU训练失败。旧§18的16 unknown区间/39.789931998783GPUh及全部旧费用/原件独立保留，无旧额度迁入。

实时准入初始可选8，实际峰6GPU（GPU-node-A峰4/GPU-node-B峰2），单node未超6，foreign未触碰，独立eval无NCCL。七shard16槽端到端plans/s中位6.80987/native8.98177，单shard8槽5.22250/native8.18780；不同任务/长度非配对profile。最大allocated11362.72MiB/reserved27738MiB；不以最低显存为目标，每个就绪同controller shard最多16条真实行，沿既有16/8/4吞吐证据，0额外科学行/32重探/凑卡。worker+descendants最高采样PSS49.757GiB/92线程，为个体采样而非相加伪称双node峰；CPU按node分别预算。

6次完整runner.close、88子环境释放，双node本批owner/GPU/env进程0、UID zombie0。退出UID PSS约GPU-node-A 0.314/GPU-node-B 1.415GiB（包含其它UID活动及CPU验收），并非本批idle GPU进程。ROOT实耗0.683GiB<4GiB；新root/fit/launch前storage-node独立quota、relevant du、shared与CPU准入均保留，0data0新增/大资产复制/H或视频缓存。最终root大小及已知原子副本预算验收，未冒称OS连续磁盘峰测量。

完整results/engineering_acceptance/cost_accounting/useful_throughput/resource_exit和fit_manifest/evaluation_manifest/fit_completion均在本analysis/ROOT。代码来源fit537c6763、首次读取56491f82、最终读取08437d23；记录集成身份见git_delivery.json。所有task-owned开发树/本地分支已核实集成后清理，三个formal冻结保留。整批一次回main01a13dd9后归还canonical写窗口，0自Queue/心跳/第二waiter/自动后继。
