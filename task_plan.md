# EMBER task plan

## 当前目标与授权

Owner继续授权完整自主科学推进。当前active design为`docs/horizon_relation_video_writer_design.md`，§8.2.5–8.2.6的首层语言内容单变量fresh对照至400已全部完成；[完整报告](docs/horizon_k1_first_query_only_20260909.md)及`runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/segment200_400/round_evidence.json`记录正式证据。下一项实验尚未登记，不能从历史脚本自动恢复。

已完成：新100/200/300/400 strict400=75/110/106/103；新200/400 train96=46/59，原同节点52/59。后两validation点较原86/87有局部收益，但BBQ持续回落、成功集合高churn、目标没有扩展。400 breadth6、Long9，两个validation任务仍0；所有训练/物化/诊断/闭环完整exit0，全部配对检查通过，GPU任务均已结束。

1. 完整交付本轮事实、适用边界与“局部收益而非整体修复”的裁决；保留四checkpoint和全部原始证据。
2. 不原样续500/600；从已经复现的训练获取与目标能力分离出发，审阅实际接口功能、竞争解释及最近等价历史。Target-Owned rank共享、旧语义路径和旧条件组织边界必须保留，不机械叠模块或重做改名方案。
3. 选择并登记一个有信息量的受控变量：明确输入/监督/共享变化，什么证据会支持或反驳，以及与已试组合的实质区别。必要时先做最小可区分诊断，再正式修改/profile/学习；现有广泛授权继续有效。
4. 根据真实行为推进，直到validation8 single-checkpoint correct>145/400、相邻稳定、breadth、四suite、Goal/Long、跨视频与最终因果全部满足，再完成方法冻结后的32/8 fresh与Test。当前结果不是完成，不预设总尝试次数。

当前canonical `first_query_only_v1`保留；没有500/600、24-task组织、额外meta、RL或新结构任务在运行。资源与即时状态以progress为准；新执行先完成相应科学/资源/clean pushed frozen合同，常规步骤无需再次询问。

## 历史：只读报告交付任务（已完成）

**2026-09-09 Owner最新：暂停所有正式实验，只深入分析已有证据并交付报告。** 不启动训练、profile、物化、闭环评测、模型forward或新干预，不继续实施24-task分叉。允许只读源码/Git/原始artifacts与CPU重算既有结果，保存分析报告和必要状态文档。

本轮交付：`docs/horizon_k1_evidence_review_20260909.md`。覆盖实际已执行架构与数据流、历史强模型及同架构配方对照、当前逐任务行为/训练轨迹/样本曝光、训练和评测工程合同、监督与泛化解释；列支持证据、反证、证据缺口及最可能原因排序。不能把训练侧改善等同于架构无问题，也不能因可测试就认定条件组织最值得改。
工作分工：main负责现行行为/数据量化与综合；三个只读审计分别核对历史对照、架构机制、工程接口。现有未提交24-task源码/config/tests保留为暂停草稿，不充当任何已执行证据，不随报告提交。

本轮报告与既有证据再分析已完成，详见上述报告和progress。后续只讨论报告；没有获准恢复实验的当前计划。

## 历史授权与已暂停的分叉计划

当前执行已转入design§8.2.3的条件组织诊断：原样500/600=70/82，训练held200/400/600=52/59/67。暂不原样续训；从完整400受控分叉，全train24每update共256queries、task目标权重1/24，保留完整学习状态与原topology，对比相同500/600节点。先完成现有采样/训练入口修改、恢复与权重验证及真实profile，再从原400正式启动。模型、数据池、source和优化超参保持，解释仍待检验；无全量熟悉视频/clone或RL自动下一步。

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

500/600 strict400已完整70/82，breadth均4，未恢复200峰值或扩展能力；暂不原样继续700/800。追加冻结600同口径train96，以检验400之后是否出现整体训练行为退化，避免把200→400的泛化解释直接外推。只变checkpoint、无梯度、不用于正式选点；见design§8.2.2。

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
