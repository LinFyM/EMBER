# EMBER task plan

## 当前目标与授权

继续已创建的全过程goal：完整架构、正式学习、结果分析与方法迭代，直到validation8 strict paired correct>145/400、
相邻与跨视频稳定、breadth/四suite/Goal/Long和最终因果要求达标，再完成冻结方法32/8 fresh与Test。
不设token预算、总工期或总尝试次数，代码/profile/loss/单峰不是完成。

2026-09-08最新Owner安排：**当前集中K1，Writer整体先纯FM后独立共享RL**；完整架构和信息墙保持。
Meta归属Writer内部读取模块。真实采样与逻辑4task/256queries更新合同独立于GPU资源。

## 执行计划

1. **已完成基础：** 全仓理解、历史审计、完整Horizon架构及真实完整梯度；正式纯FM fresh起点与64节点correct99/other95。
   旧384停止，旧联合profile只作机制证据。当前混合K段在原128完整checkpoint安全边界结束，保留完整历史。
2. **当前落实：** sampler实际固定K1；Owner补充纠正明确fresh K1，全部Writer可训练参数与学习状态从step0重置。
   旧mixed checkpoint/result仅历史，不继承其权重、optimizer、scheduler或sampler/RNG；冻结source无需重训。
   验证全局4suite×64queries、1/4权重和SUM归约、一次clip/step/scheduler、设备无关的global cursor及曝光。
   有效同节点1–6卡只负责执行；现有完整条件分工最多4卡真正有用，禁止空rank/dummy或扩大batch。
3. **吞吐与第一K1段：** 实测完整视频/horizon的FM microbatch与帧/edge分块，解决已证实加载初始化开销；
   记录queries/s、LoRA/s、step与整段墙钟、加载/保存/评测耗时、显存峰值。有限profile后及时训练。
   用真实K1速率选择约一小时的段长，50或100倍数checkpoint，中间/末尾两个点，看到分数前登记。
4. **correct400主线：** 每段暂停训练评测两个K1 correct400，持续raw rows相邻per-task/suite/breadth/RGL/churn/J。
   train120及held FM按判断训练获取/泛化需要安排；仍低分且获取能力时不反复other。
   结合累计queries、每task条件曝光、多个有信息量节点判断平台，64/192步都不能自动代表充分监督。
5. **资格与后续：** correct接近/超目标且出现相邻稳定候选时补other；冻结单checkpoint后完成视频因果controls，
   shuffled/reversed最后且不反哺设计。K1全部通过后再登记few-shot训练/测试。
   充分纯FM后才考虑独立共享Writer RL，fresh RL optimizer/scheduler、默认无FM；充分监督仍弱先定位实质能力缺口。
6. **最终目标：** 保留单checkpoint全部资格，方法冻结后按32/8 fresh和最终Test完成全流程。
   始终一套canonical运行面，正式证据和必要checkpoint保留，按合同clean pushed detached正式launch。

## 边界与工作方式

合同内实现、正式训练、评测与证据支持的实质方法修订已授权，不重新等待批准。
GPU/存储/Git只在依赖它们时刷新；不复用旧空闲卡或quota预算。历史由Git、formal artifacts与research_history保留。
validation/test无梯度；无teacher action/state/reward/task-ID部署输入；完整视频/horizon/梯度与单完整LoRA合同保持。
当前状态看[progress](progress.md)，历史结论看[findings](findings.md)与[research_history](docs/research_history.md)。
