# 完整参数提案Writer工程记录（2026-10-10）

实际接收main为4e16cfea，接收记录c76615fc已push。实验session独占
`/data1/user/ymdai/ember_worktrees/video_guided_proposal_20261010`、
`codex/video-guided-proposal-writer`；主讨论只读，科学合同为active设计全文及§13。

本轮采用一个`proposal_writer`运行面，新增职责分为双meta读取、完整坐标G、三head π、
训练教师标签、状态机及condition-SUM学习。模块按实际职责组织；完整合同不能由单个旧loss或runner改名承接。
布局/通信与两个独立Reader在model；参数读取上下文在native；资产、上下文版本与执行在runtime；
教师、编译路径、学习及薄CLI分别由后续实现持有。
预计超过五个source文件/1000行，理由是实际独立的监督、生成、决策及工程恢复责任，
不是并行方案。新增前guard的base c76615fc无增长flag；完成时按实际diff复查。

环境进程/固定episode批量执行/官方10-flow为通用责任，由`writer/practice/`承接。移出的旧功能编辑、固定P出口、旧FM/PG采样、评测与CLI
在双meta/G功能消费、真实MT完整episode及v3现场重放验证后，连同仅验证旧算法的tests退役；历史通过Git和原件复核。
通用引擎每episode要求显式完整task因子及身份，不再默认MT，不包含编辑器或动作SDE。

读取meta持有在独立Module的ParameterList中，未注册进policy；policy中的38/r128物理adapter保持
合法A/nonzero、B0。读取hook上下文与执行hook互斥；whole-frame checkpoint显式传入全部meta因素并重入上下文。
读取采用实际双stream返回的post-norm contextual Z/H；raw embed_image只用作冻结自身图像特征。
所有ψ依赖特征跨更新失效；π阶段冻结全部ψ后才允许版本内CPU缓存。θ的编码每次score重算。

现场快照仅用于训练标签，包含sim、可变model scene、controller/interpolator、wrapper及RNG；
重放时核对原场景与双RGB，不以旧RGB/proprio重建或猜测。v3增加observable时钟/cache、夹爪累积动作和机器人buffer；实测同起点、同5个实际动作的RGB/proprio重放一致。

strg01现场data1 quota：使用1404602264KiB，quota2147483648KiB，limit2157969408KiB；
到quota约708GiB余量。共享data1约80TiB空余；首批新增64GiB及约0.3GiB开发checkout可容纳。
个人目录du为1438312660992bytes；共享df不替代quota。下面登记实际profile，不视为正式学习或能力结果。


实际profile位于data1的`ember_runs/video_guided_proposal_writer_20261010/pilot`。profile_01因缺LIBERO进程配置退出，0.053046GPUh；profile_02完成主要机制/吞吐，随后旧现场快照重放失败，0.139984GPUh。
profile_02最长教学84帧，frame chunk2/4/8为60.24/53.13/51.10秒、12.60/13.34/17.11GiB peak；更大物理query批量28/56为20.60/18.85秒每112queries，112在该共驻卡OOM。
两个meta均有B梯度，第二次更新出现A梯度，更新meta前后任务原生响应RMSE为0；完整16步参数积分3.48秒。这些只核必要图/上下文，不说明视频或经历已学会。
NativeMT完成一条515环境步的真实episode；旧v1/v2快照不补造标签。v3实际现场短重放通过，pre/post双RGB与8Dproprio一致、重放settling0/控制5。失败原件及费用保留。
近期共享卡性能明显慢于旧8.21–8.31秒，不直接沿用旧吞吐预测。加入冻结vision/token raw cache，Gemma及action contextualZ/H仍每次依ψ重算；下一消费者核验会选择实测microbatch/chunk和资源安排。

源码所有权：contract/panel；native双meta上下文；runtime单source与版本/冻结缓存；model完整G/独立π；teacher标签优化；path预算编译及完整score输入；learning共享更新/checkpoint；pipeline预登记阶段；batch退出事件调度/实时准入；analysis原行统计；profile一次性真实机制/吞吐。CLI仅串接。
旧experience_compiler专用源码16文件和7项旧算法tests已退役，原件/Git c76615fc保留。通用native10-flow、固定episode/实际H与现场快照统一在writer/practice包，避免继续增长writer根目录peer。

结构自审：新增surface按职责持有，不超过262行/文件；profile消费拆成读取、教师、实际head、完整原生反向的同模块函数，环境步进与退出receipt独立。guard基于c76615fc没有新增hard violation，净source减少约2000行；既有operator文件仅更新退役入口文字。固定坐标、score求和、50初态角色划分及完整10-step同调用hidden四项针对性tests通过（13.22秒）。下一profile仍待核新缓存/三head/原生反向，不宣称formal已就绪。

profile_03从clean实现commit20c43627在gpu01:1共驻运行，退出0/295.81秒/0.082168GPUh。新读取真实84帧chunk8/16为19.62/17.67秒、17.43/25.00GiB peak；source/meta执行RMSE0，两组meta第二步A/B梯度、三head实际OwnH及完整10ODE因子反向通过。112query物理28/56为10.00/10.14秒，112仍OOM；不把批量56当成有收益。
实测一份515步实际历史：无损gzip302798405bytes；不持久化冻结source的raw image embedding缓存后75818792bytes，真实RGB/hidden/动作/反馈/快照完整保留。只释放可重算缓存；共享CFM对不可变事件H做进程内复用，其下游ψ编码照常重算。
最后focused消费者将测完整10ODE功能标签的物理1/2/4批量及chunk24，并覆盖真实π的STOP到完整MT冻结出口，不执行freshG随机策略；正常G实践链留到预定G320原件。新增功能物理批量保持每item masked均值、item间均值，无监督/任务权重改变。

profile_04（7595f413）退出0/65.90秒/0.018305GPUh：chunk24为17.77秒/32.56GiB，没有比16提速；4个实际保持函数标签physical1/2/4为6.33/3.43/1.73秒，最大10.59GiB；没有逐tensor一致性追查。真实π的STOP到完整MT出口通过，明确只核这项冷启动安全分支，未执行freshG。profile_05整步退出0/38.13秒/0.010591GPUh：112query+4实际keep+prox+clip+optimizer共12.60秒/16.40GiB，任务副本L2改变0.307135，source无梯度。这些不称能力学习。
实际选择：教师physical demo28、functional4；G frame chunk16；独立执行暂8 slots。56/24未显示吞吐收益，112 OOM，故没有照显存扩大。正式读取和训练初值仍fresh，不加载这些disposable更新。
调度预算依据：gpu01:1实测约10秒/112 forward-backward、完整带keep约12.6秒；gpu02:7近期共驻约20.6秒/112。首轮教师使用已在现场可用的gpu01:1/6，gpu02:4/6/7并行采集/读出，不因高UTL排除后者。不是按world_size限制整批，预算实际约束了长梯度阶段的设备选择。启动时重新现场准入，变化自行调整物理安排。
预计完整计算暂14–18h（2张已可用的高吞吐训练卡，独立采集/读出最多再3卡，严格阶段依赖；相对原8–12h上修）。整批暂估32–40GPUh，π/G的完整历史成本仍不确定；40为硬上限，不能通过缩矩阵满足，达到具体边界给主讨论。
存储峰值64GiB计划包含cache-free真实H、12条教师端点/full optimizer、G/π完整状态、当前未消费候选、最终LoRA及指定完整U。局部数据在local128完整保存后、每RL batch在完整checkpoint后、报告在query/指定U完成后，退休已消费可重算的中间候选及H中重复参数副本，保存原始RGB/actual隐藏/动作/反馈、seed/父参数引用、raw rows和选定最终参数。该生命周期操作在相应消费者完成前不执行，必要G刷新教师/恢复资产保留。

首次formal collect四任务在恢复标签阶段失败，原件jobs/collect_*/attempt_001与initial_teachers_exit完整保留；尚未执行教师更新。event00完成，event01尚未保存真实H，故没有将丢失的H补造为可恢复原件。
v4现场快照补integration+真实微步kinematic/contact缓存，根因为mj_step后pose/cache和qpos微步时点不同；单独forward不是现场恢复。200控制步复现0.34mm差，39KB payload跨进程重放及随后5个实际动作RGB/proprio一致，profile_snapshot_04/05保留。完整MjData pickle嵌入133MB模型不采用；不放宽场景一致性guard。v1/v2/v3旧快照不补造为v4。
新采集在标签前保存实际H/代码/episode回执；四个原event00不重复。π只缓存停止梯度的原生响应预测，θ Reader/trunk/heads每次重新编码；消费者完成后候选生命周期保存最终和指定完整U、原始实践事实、全部参数改变量及固定ψ重算身份，非U旧path不能继续完整旧on-policy回放。对应source-cache live-θ梯度和候选消费/事实保留两个协议回归补入，共6tests通过。尚无G/π或教学学习能力结果。

主讨论裁决将同一首批累计硬预算40提高52GPUh，峰值64GiB与所有科学范围保持；原预算/时间/launch保留为历史。
Batch读取ROOT注册上限，显式CLI只能在该上限内；完整教师更新按所选设备实测估计，未测设备暂按22.5秒估计并计启动余量，
依据近期慢demo约20.6秒加实测功能项，不宣称固定吞吐或保证上界。
完整160 checkpoint接续只预留剩余320更新，原opt/query/RNG不重置；费用包含已发生profile/失败及其它协调器未退出的launch。
不为承接预算建立新runner/模块，batch仍是资源owner；新增检查保留于现有协议文件。跨协调器依赖只等待其退出回执文件事件，
已完成160的教师可在别的合格设备原opt续480/并行audit，原160协调器持有的设备等它全部退出后才转交，不周期读进度或重启更新。
新增协议检查核预算与外部退出事件；结构检查继续检查既有batch，不新增长期runner或parallel family。
8项协议tests通过14.41秒；随后受影响预算/事件检查通过10.55秒，最终退出处理拆分后的事件检查通过5.73秒。
最终guard相对18666净source增135行/4既有文件，没有hard violation；完成事件处理、延后设备释放仍由同一Batch资源owner承担。
当前预测45–51GPUh、恢复后主要计算16–24h；G/π完整历史实测与报告预留仍须在既定消费者/完整checkpoint边界核算。
