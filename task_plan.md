# EMBER task plan

## 当前目标与授权

2026-09-09 Owner要求综合语言分支、过程表示、参数生成和监督/闭环不一致，尽可能通过分析与诊断实验寻找原因并提出解决方案。**禁止正式架构或训练方式修改，禁止启动正式训练。** 允许只读/CPU分析、冻结模型forward与临时接口干预，以及事先界定的非正式探索验证；保留科学信息墙、train24梯度边界及GPU/存储合同。当前两个已启动400评测收尾继续，不恢复500/600或先前暂停的正式候选。

本轮完成标准：收齐现有400证据；用有区分力的实际实验逐步检验竞争解释；交付有原件支持的原因排序、排除边界、最早可定位接口与具体解决方案。不能承诺所有原因均可唯一识别，也不能以更多诊断数量代替信息增益。

1. 收尾上下文400 strict400/train96并完成原登记的配对比较；已有训练和held FM均完整结束。
2. 并行只读量化实际监督覆盖/尾部padding/动作事件，以及历史oracle、专家/clone、G1/G2与读出对照的可迁移边界；主agent负责当前机制实验及综合。
3. 冻结上下文200/400，在train24固定teacher46/47与32条独立动作queries上，分别隔离三处语言分支、整体语言分支、视觉读取、局部消息、长程attention和回写的功能影响。原生language、真实视频顺序和完整H保留；临时hook不改变checkpoint或canonical源码。详细注册见active design §8.2.10及诊断报告。
4. 根据上述实测选择最有信息量的下一项：分horizon/动作维度与10-step行为预测差异，或冻结接口的局部可达性探索，必要时仅在train tasks做诊断闭环。任何局部优化必须显式登记独立变量、query训练/检验划分和不可部署的oracle身份；不改变共享Writer正式学习状态，不使用validation/test梯度。
5. 汇总证据并提出解决方案；正式改图/改训练及fresh启动留在本次授权之外。最终视频controls和Test继续封存，不借诊断提前使用。

即时执行与结果见progress；当前报告为`docs/horizon_k1_causal_diagnostics_20260909.md`。先完成明确问题的诊断，再按结果决定后续，避免重复既有几何/分支分析。

## 历史：上下文条件正式保持段

Owner继续授权完整自主科学推进。active design为`docs/horizon_relation_video_writer_design.md`，当前§8.2.8逐帧上下文过程条件fresh首段全部完成：100/200 correct52/103（前轮75/110），200 train96为41（前轮46），没有整体收益或稳定资格。[完整报告](docs/horizon_k1_frame_contextual_20260909.md)和`k1_frame_contextual/first_segment_evidence.json`保留全部配对事实。

1. §8.2.9已在结果前登记：从完整200同配方exact-resume至400，保存300/400及各自correct400，400同口径train96与既有held FM。缺口由23缩至7、自身52→103，补尚未覆盖的保持区间；不将低起点增量、breadth中的单次成功或续训本身作为优势。
2. 新launch先刷新双节点GPU、独立quota及新增峰值，保持原world4/rank物理2/6/4/0和完整学习状态；复用已验证运行面与profile，不改科学配方或重复工程检查。当前同配方200→400训练已完整exit0、两个checkpoint与实际曝光核验通过；300完整79/400，400banks和冻结诊断已完成，400验证与train96并行，实际进度见progress。
3. 300/400先按自身相邻绝对能力、per-task/suite、breadth与R/G/L/churn/J判断，再对前轮106/103、train40059与原始86/87。若无实质获取/保持优势，结束该访问干预原样续训，回到有区分力的不同机制；不能靠loss或微小局部新增续500/600。
4. 保留已有历史边界：first-query-only至400仅局部收益；冻结train24功能对应持续增强，不能称普遍未学会条件编译；Target-Owned已检验rank共享，旧四任务纯FM模型也曾更强。不能用新低分抹去这些反证或机械重做改名方案。
5. 持续以证据推进至validation8 single-checkpoint correct>145、相邻稳定、breadth/四suite/Goal/Long、同task换视频与最终因果全部达标，再完成方法冻结后的32/8 fresh和Test。不预设总尝试次数，不因一次完整实验结束停止自主工作。

即时资源、命令和进度以progress为准；当前没有启动other/最终controls/meta/RL/Test，常规合同内操作无需再次询问。

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
