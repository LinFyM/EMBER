# EMBER task plan

## 当前目标与授权

**Owner最新：解除暂停，恢复全过程自主推进。** 首轮完整400 exact-resume至600，科学配方不变，500/600 canonical strict400；600不是任务终点。
并行补冻结200/400 held-video train96；仅当仍不能区分共享获取不足与视频泛化失败时，追加固定且确已曝光的熟悉视频配对诊断。
每次深入诊断预先明确竞争解释、历史已排除与未排除项、主要变量、各结果对应的下一步。小面板不等于根因，不能从历史曲线直接归因。
后续以行为证据自主继续：实质增长与breadth扩展则检验相邻稳定；连续无获取则最小有信息量干预，充分支持后可改条件组织或实质接口，保留信息墙/科学目标/纯FM。不得靠loss下降无限续训或因600单点宣布架构失败。

长期全过程目标仍为：完整架构、正式学习、结果分析与方法迭代，直到validation8 strict paired correct>145/400、
相邻与跨视频稳定、breadth/四suite/Goal/Long和最终因果要求达标，再完成冻结方法32/8 fresh与Test。
不设token预算、总工期或总尝试次数，代码/profile/loss/单峰不是完成。

2026-09-08最新Owner安排：**当前集中K1，Writer整体先纯FM后独立共享RL**；完整架构和信息墙保持。
Meta归属Writer内部读取模块。真实采样与逻辑4task/256queries更新合同独立于GPU资源。

整轮视频schedule修复已完成；fresh100/200各400条件bank实检通过，checkpoint与学习状态保留。首段两节点canonical correct400及相邻分析已完成并显示持续增长，200→400及300/400 correct400均已完整结束，400→600此前按Owner暂停且无新update；现已明确恢复原样续训与训练任务诊断。

500 strict400已完整70，600执行中；追加冻结600同口径train96，以检验400之后是否出现整体训练行为退化，避免把200→400的泛化解释直接外推。只变checkpoint、无梯度、不用于正式选点；见design§8.2.2。

## 长期路线（已恢复，以本轮证据裁决）

1. **已完成基础：** 全仓理解、历史审计、完整Horizon架构及真实完整梯度；正式纯FM fresh起点；旧混合K64的99/95为重复teacher抽样的非合规探索成绩。
   旧384停止，旧联合profile只作机制证据。当前混合K段在原128完整checkpoint安全边界结束，保留完整历史。
2. **当前落实：** sampler实际固定K1；Owner补充纠正明确fresh K1，全部Writer可训练参数与学习状态从step0重置。
   旧mixed checkpoint/result仅历史，不继承其权重、optimizer、scheduler或sampler/RNG；冻结source无需重训。
   验证全局4suite×64queries、1/4权重和SUM归约、一次clip/step/scheduler、设备无关的global cursor及曝光。
   有效同节点1–6卡只负责执行；现有完整条件分工最多4卡真正有用，禁止空rank/dummy或扩大batch。
3. **吞吐与第一K1段：** 实测完整视频/horizon的FM microbatch与帧/edge分块，解决已证实加载初始化开销；
   记录queries/s、LoRA/s、step与整段墙钟、加载/保存/评测耗时、显存峰值。有限profile后及时训练。
   用真实K1速率选择约一小时的段长，50或100倍数checkpoint，中间/末尾两个点，看到分数前登记。
4. **correct400主线：** 每段暂停训练评测两个K1 correct400，持续raw rows相邻per-task/suite/breadth/RGL/churn/J。
   允许先升后降再升的学习过程；继续观察400及之后节点，不因300单点回落停止、重启或淘汰架构。
   train96及held FM按判断训练获取/泛化需要安排；仍低分且获取能力时不反复other。
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
