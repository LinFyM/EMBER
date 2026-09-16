# EMBER task plan

## 当前目标与授权

Owner于2026-09-16明确调整goal：弄清A后续性能与视频特异性，结合原始v5.2至今近两个月实验分析原因，
据此形成改进B，完成一轮正式训练／评测并同样分析，无论解决与否直接汇报，不另写报告。
期间正常推进共享SFT。A900后暂停、SFT只到400及不运行新B的限制已解除；旧C不恢复。
本轮B是待分析后确定的改进，不自动沿用历史双视角／learned-read B；>145不构成无限扩轮理由。
权威合同见[active design](docs/source_alignment_v52_plan.md)与[current owner requirements](docs/current_owner_requirements.md)。

## 执行计划

1. **进行中：继续A并弄清变化。** 从完整A900精确续训，预注册1200／1500／1800三个节点，
   每100保存，累计7200条件／151200 queries。架构、source、数据、global84、optimizer／LR与恢复拓扑不变。
   1200沿原冻结575c189a；超过1200仅扩预算，须通过原事件前缀与完整状态连续性核验。
   各节点correct400＋train96及other／wrong／shuffled／reversed各400；时间臂最后，固定映射与RGB完整forward。
   复用已审计的source零LoRA48，不重复无视频rollouts。看逐task／suite、覆盖、相邻保持和特异性差额如何变化。
   当前300／600／900 correct为99／88／140，train36／47／54；A900视频对照为140／136／116／48／128／139。
   按owner纠正：旧强视频特异性没有在新A900复现出来。
2. **待A完整后：解释问题并形成改进。** 综合近两个月实际正负证据与代码，区分获取、保持、迁移及视频利用问题；
   列清已定位原因、竞争解释与最早失效接口，说明改进预测和相对历史失败方法的实质差别。
   可并行整理历史证据，但不在A全貌未知时预定本轮B架构。
3. **待设计：一轮正式B。** 实现、必要profile和验证后，在首个formal更新前登记B的架构、曝光、节点及停止边界。
   Fresh训练并完成完整paired评测、保持和视频特异性分析；结果正负均解释，直接向owner汇报，不另写报告。
4. **已完成：共享SFT基线。** 原两rank完成450步／259200 queries，400／425／450正式validation400为85／89／86，
   相邻R/G/L为66/23/19与64/22/25；完整审计通过，全部SFT作业已停止。结果与边界见findings§110。

## 资源、边界和交付

A继续阶段预留32GiB，SFT3GiB，B暂留40GiB、设计后重新profile核价；准备时data1独立quota余量约98.1GiB，共享83TiB。
每次launch前双节点live GPU及quota检查；A exact-resume保持gpu01原四rank，SFT保持gpu02原两rank，互不重叠。
所有formal运行来自clean pushed detached树，保留失败与原始证据，不修改冻结旧合同或静默绕过resume。
保持source71／train24／validation8／test8、normalization、信息墙；无RL、held梯度或Test。
本轮按节点视频诊断获明确授权，用于观察演化和A问题分析，不用controls选点或构造训练loss。
长任务按顺序后台完成，完整节点才读结果，异常才介入，不持续盯看或逐分钟汇报。
最终完成须包含A全貌、原因／改进论证、B完整一轮及解释、SFT基线；不以仅启动或单点涨分代替完成。
