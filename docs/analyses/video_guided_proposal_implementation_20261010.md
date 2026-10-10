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
