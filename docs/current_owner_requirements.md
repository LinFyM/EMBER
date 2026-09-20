# EMBER owner requirements

本文保存owner的稳定目标、研究原则与协作要求。Owner最新明确表达优先于本文；本文优先于[AGENTS](../AGENTS.md)中的默认合同。
当前授权和实际状态只看[progress](../progress.md)，执行计划只看[task_plan](../task_plan.md)。历史讨论与旧实验不构成重新启动授权。

## 单次覆盖重训与后继实验授权（2026-09-20，最新）

Owner明确采用专家最后的单划分＋12辅助任务方案，重新设置goal；旧冻结1500的推进暂停已被本授权取代。
只fresh训练一次EMBER、一次MT-BC；复用Source-71与normalization，保持单相机K1、架构及两项功能损失。
目标40重新按语义覆盖划分24/8/8，另12个经过核验的LIBERO-90任务由两方法共同训练，总计36个任务等权。
held所需操作对象、装置与基本操作必须有实际训练支持；不将完整held等价任务混入训练。旧划分与78/74/82结果保留。
按固定间隔完整Validation曲线决定持续下降或平台早停，各选一个验证最佳checkpoint，不设2000/600硬终点。
不做k折、额外seed、Unpaired重训或合并Validation重训；共享训练结束后使用唯一选定模型继续原FT/RL/外部比较。
新模型性能与视频特异性通过原计划要求后继续；Test相对MT-BC至少+40/400的门槛不变。不达预期即停止下游商量。
充分利用两节点资源，低负载他人GPU允许在显存余量充分且无明显干扰时共驻；原总卡数边界不变，不抢占他人进程。
推进中效率优先：在上述物理卡边界内充分使用合格设备，以实际训练更新/秒、LoRA/秒、完整评测吞吐、GPU利用率和峰值显存识别瓶颈；按测得收益优化物理batch、worker调度与实现，保持逻辑更新、配对评测及数值稳定合同。不为追求显存占满而牺牲吞吐，不原地修改正在运行的frozen代码。
只在完整训练区段和评测完成后读结果，不反复读进度，不用监控子代理；独立工程/数据审计可按仓库规则并行。
详细执行合同见progress登记的新设计；旧论文实验中的FT/RL方法、信息墙和停止条件继续适用。

## 保留的论文后继实验要求（2026-09-20）

旧协议曾冻结EMBER1500与MT-BC425，完成Test78/74/82后未达+40门槛；该阶段已封存，
其固定checkpoint和不重训限制由上方新授权替代。后继比较使用新协议各自唯一选定模型，MT-BC先合并共享LoRA。
完整Test paired400的+40门槛及冻结后视频对照要求保持；Test不用于重新挑选模型。
FT仅用一条support做梯度更新，从同task其余演示中预先划出独立选点集，按固定查询FM loss选checkpoint；
既有每task50个eval episodes仅评估、不选点。额外选点动作标签计入信息成本。默认覆盖validation/test全部16任务。
若EMBER明显落后FT，先补两条预定support分别重复（合计三条，不融合不选优）；持续劣势或结论混杂即询问Owner。
RL直接训练EMBER生成A/B，对照合并MT-BC底座上的fresh identity A/B，均38-target rank16；Writer和底座冻结，
optimizer/critic fresh。目标是全部预定交互预算点EMBER平均曲线领先；不理想时停止下一段并报告。
允许这些独立task-local实验使用目标task support动作、选点动作及局部RL reward；不得更新共享Writer/source/MT-BC，
不得回流零交互模型设计。先封存零交互结果，再开放目标适应用途。FT/RL详细预算在训练侧资格后、held实验前登记。
不按三天期限缩水实验；完整训练/评测阶段结束后统一读结果，不反复读取进度。最新要求不用监控代理，采用作业结束信号。
遇到无法定夺或不达预期结果暂停下游询问；顺利则交付可追溯的图文报告。同步清理确认无用代码、文档、分支/worktree
及大资产，保留正式证据和用途未决资产；quota不足先报告，不越限。实际状态与active合同见progress/task_plan。

## 历史开发接受标准及通用设计原则（2026-09-18）

旧划分下Owner曾接受验证能力保持在130多、140左右的候选，不要求超过145或150；这些绝对分数不移植到新划分。
重点是相对v5.2更清楚、更有益的视频特异性。允许已有能力中少量正常的获取/丢失，不能把任何churn都当失败。
历史实验仍按当时封存合同报告，不追溯改写其分数或资格裁决。

Owner要求识别能区分架构的具体机制，明确每项分析或改动针对性能、视频特异性或稳定性中的哪一项。
共同拥有语言、native响应、共享头等属性不构成成功原因。允许能快速区分解释的冻结诊断和小实验，
不把缺少因果证据仅作为停止推理的理由；是否保留或删除旧模块均须据实际作用判断。
设计必须综合全部相关历史正负证据及其预算、配方与适用条件，不能只根据最近两次诊断确定路线。
结构应职责清楚，能够复制同类模块加深，并自然扩展参数规模；这不是必须沿用旧Core/P或仅修改Procedure的要求。
是否保留、替换或删除模块均须由完整推导支持，允许实质重构；原v5.2不重训的要求继续有效。

## 经最终LoRA的教学候选授权（2026-09-19）

Owner要求设置goal完成[专家最终修订](review_materials/20260919/expert_proposal.md)、推送远程并汇报；替代上一goal完成后的无active状态。
采用一套生成LoRA的跨episode主FM与同视频短程动作辅助目标，两项共同更新整个Writer与三组Meta，source冻结。
本授权明确允许登记的同episode辅助功能误差经实际LoRA回传；主功能监督仍严格跨episode，所有部署信息墙保持。
恢复有序A骨架并加入完整H/真实相邻视觉重复读取，保留完整A/B和四任务逻辑更新；不采用已撤回的P-only教学梯度。
Owner最终明确主实验先用单相机agentview；完成后由证据决定是否值得补双相机，不自动追加。
本轮候选、原1500窗口与条件性消融见[设计](video_teaching_writer_design.md)，实际状态见progress。
Owner随后要求考虑继续训练以观察后续趋势；具体有界续训另行登记，保留完整训练状态与配对可比性，
原选定checkpoint及其controls保持原窗口的证据口径。

## 专家建议的匹配诊断授权（2026-09-18）

Owner随后提供[专家意见](review_materials/20260918/expert_review.md)，明确授权设置goal并完成其中要求的工作、推送相关结果。
本轮只fresh训练与强A匹配的learned frame-set参照，复用A既有预定节点，不重训原A，不同时开新架构路线。
仅移除video-frame位置/因果mask/读取时间寻址，保留真实帧内容、三Meta、完整A/B、采样和优化更新语义。
为解释既有A，本诊断匹配其agentview/fixed-mean H历史口径；这项限定例外不改变未来新方法的完整读取原则。
比较相邻节点、覆盖和成功集合，不能只凭总分接近宣布等价，不以wrong/shuffle/reverse选择节点或训练惩罚错误输入。
本授权在上述范围内替代暂停；执行合同与状态见[active design](learned_frameset_reference_design.md)和[progress](../progress.md)。

## 统一Writer整套实验的Owner授权（2026-09-17）

Owner明确设立新的执行goal，授权依次完成：

1. 整理整个仓库及data1空间，优化精简代码/文档，删除核实obsolete、temporary或duplicate的资产。
2. 实现已选定的统一新架构，优化真实训练和推理吞吐；Owner最新明确要求六卡等效训练提速，不改变逻辑batch和更新语义。
3. 完成新架构训练，与已经完成的v5.2训练/评测结果比较，判断能力、视频特异性与保持是否改善；不得自行重训v5.2。
4. 效果不佳但有具体改进空间时，实施有依据的修正并重新训练；充分证据显示缺乏可行改进则停止，不做无限搜索。
5. 整套实验结束后详细汇报，不能以仅实现、启动训练或单个指标结束goal。

本授权替代此前“只分析、不实现/实验”和暂停后续新方法的限制；旧A3000/C等无关实验不自动恢复。
已选定的[统一Writer设计](v52_evidence_based_writer_design.md)为本轮候选：中层/末端同构联合Z/H处理，
一次中层双写回，以内容起始的连续参数状态生成完整LoRA。实现时保留其科学合同，实际资源与节点按profile后封存。
Owner同日明确纠正：v5.2已经训练过，本轮复用既有checkpoint与正式评测原件，不能把“比较”扩成重复训练旧方法。
比较时如实列出source、输入、采样、预算和优化条件的差异，并限制因果归属；条件差异不自动授权另开fresh v5.2。

架构继承v5.2有效处理原则，允许实质重构；不能机械保留模块，也不能将普通video/native→Writer→A/B→FM框架
当作相对失败路线的充分解释。Owner希望主动检查可预见问题并给出一致的设计立场；真实反证或合同冲突可修订决定。
当前以正确闭环能力选择方法，130多的已有表现有保留价值；长期>145资格、视频必要增量和相邻保持分别裁决。
所有历史正负证据及验证强度继续有效，不将短窗口/未执行路线冒充充分失败，不由历史sealed Test反哺新设计。
新方案默认K1同步双路RGB、完整50-horizon到实际learned read、纯同task跨episode FM、三Meta fresh共同学习。
不需要teacher动作标注、专门阶段监督、RL或先验分段冻结课程；完整视频可双向理解，保留真实时间方向与物理前置关系。

旧阶段授权、A/SFT曲线与设计沿革由[research_history](research_history.md)、findings和Git保存，
不在本稳定需求文件中并列维护旧“当前”目标。具体状态、active design登记和执行计划仍归progress/task_plan。

## 1. 科学精神与目标

人可以从他人或不同身体的教学视频理解目标、条件与操作过程，再迁移到自己的身体及场景。EMBER探索将这种能力落实为
视频到策略参数的编译：从generic `lerobot/pi05_base`建立的冻结source出发，输入exact language与action-hidden正确教学视频，
一次生成完整task-conditioned LoRA，从未见初始化闭环完成任务。跨具身是科学动机；LIBERO结果本身不证明跨身体泛化。

- Writer初次生成的LoRA应立即有效；rollout期间不重复观看teacher video，不做task-local优化、环境试错或第二阶段适配。
- 语言说明目标与关注对象；视频动态必须提供相对language/static prior的必要条件增量。
- 输出是一套覆盖Action Expert全部38个目标的完整LoRA；无task-ID字典、独立carrier或第二套执行adapter。
  rank、memory tokens、FactorHeads及具体decoder均为方法选择，不是研究目标。
- teacher部署输入不得含action、state/proprio、reward、terminal、task ID、filename、pose、hidden normalization或policy outcome。
  执行policy可以读取自身当前观测和state；两者的信息边界不同。
- 一次Writer调用内部可对授权视频或native activations作固定、只读、多阶段读取与重放，包括方法明确登记的冻结source导数计算。
  共享读取侧适配属于Writer；source基础权重始终冻结，部署不存在loss或optimizer。

## 2. 架构与推导原则

- 架构与训练须由一套连贯、可检验的工作原理解释：从语言与视频理解、操作知识和能力获取，到唯一LoRA生成、未见任务迁移及训练保持。
  说明每个接口传递什么、下游怎样消费、什么监督和参数共享使其可能学成；模块名称、张量形状、attention或非零梯度不代替解释。
- 先综合完整的相关历史正负证据，再设计或修改方法。v5.2普通FM的视频依赖正例与后继局部正证据都须解释，
  不把不同模型的优点拼成一个不存在的强结果，也不由“监督允许捷径”直接推出必须增加辅助loss。
- 负结果后明确哪些预测未兑现、哪些假设应保留／降级／放弃，以及哪些投入应停止。新修正须说明相对近等价历史增加了什么，
  不同结果将怎样改变下一步；不因改名、模块重组或代码已准备就重做已失败的组合，也不以局部失败否定无关结论。
- Action Expert的原生动作知识须对视频理解有明确作用。图文语义、动作响应、过程理解和参数生成各有职责；
  观察侧与裸source执行侧的激活坐标必须区分，显式读取X/Y不等于把输出限制在其span。
- teacher-video time、relative action horizon、flow time和layer depth分别处理。stride固定5并保留真实末帧；
  完整50-horizon保留到实际learned read，不能用coarse或horizon mean冒充完整读取。
- 视频在rollout前完整可用，允许双向前后文；须理解真实时间方向、操作前置关系及语言中的“先A后B”。
  旧过去窗口或单向递推不构成永久约束；位置编码、mask和可见顺序本身不证明有益时序理解。
- 如扩展K，每条视频先独立保序编码，只在集合阶段置换不变地聚合；不平均frames、raw features或最终LoRA，不挑video。
  声称支持的cardinality必须实际训练，不重复视频凑K；one-shot/few-shot选择服从真实能力。本轮保持K=1。
- Writer内部所有应训练模块，包括Action Meta与VL Meta，都要共同获得真实信用；不能因模块命名切断梯度。
  仅冻结输入可跨参数版本缓存，已适配的Z/KV/H不能跨更新复用；具体probe、容量和投影由active design登记。
- 保持模块职责清楚。允许有证据的实质重构，不局限于小补丁；也不无依据叠加summary、gate、校准或重复读取。
  Owner只评论局部时保留其它已对齐部分，先说明完整流水线，再讨论局部选择。区分结构保证、归纳偏置和待检验假设。
- 架构须能通过复制同类模块加深、自然扩展参数规模；说明增加的容量作用于哪条实际计算路径。
  可扩展性是结构要求，不等于参数越多性能必然更好，也不要求所有职责使用同一种物理算子。

## 3. 证据与推进判断

- 当前性能接受标准服从本文开头的Owner最新取舍；新划分不套用旧130–140分数，且不要求零churn。
  仍报告single-checkpoint paired400、相邻成功保持、breadth、各suite、换视频与因果证据；历史>145资格合同保留其历史口径。
- 闭环实际能力先于loss、reconstruction、norm/rank/cosine、内部margin及surrogate。充分且可比的学习后仍弱于source／SFT参照，
  属于严重能力缺口；小幅涨分或loss下降不能将其降格成调参问题，也不能由此唯一归因某个模块。
- 能力与相邻资格成立后补same-task-other，选定并冻结单checkpoint，再做wrong／no-video／shuffled／reversed最终controls。
  controls不进入训练或checkpoint选择；常规最终controls不反哺架构；历史A问题诊断只按其当时授权解释。
- Owner单独授权的原因诊断可固定已结束窗口的末尾checkpoint，独立完成上述视频对照，无须先获方法性能资格。
  这不是合格checkpoint选择；保持冻结、不选优，不反哺下一方法训练或架构设计，诊断与原资格裁决分别解释。
- 正式评测使用single-checkpoint完整400配对行，不用80-row screen、checkpoint union或融合选模型。
  K1每task、每臂、每轮50个init对应全部50条合法teacher videos各一次，跨checkpoint和controls复用固定canonical映射。
- 主FM来自同task跨episode执行queries。固定validation/test不产生梯度；扩展non-held meta tasks须先审计held及重复specification排除，
  登记allowlist/provenance。更多同task episodes不等于更多独立meta-task映射，不制造人工process或新仿真任务绕开问题。
- 本轮Writer和Text/VL/Action三组Meta采用fresh联合监督，optimizer、scheduler、sampler和RNG均fresh；合法identity不要求每个张量随机非零。
  历史G1–G3冻结阶段不实施为当前课程；监督学习不混RL、trust回滚或部署适配。未来共享RL须另作独立阶段，不能替弱监督结果救场。
- 学习窗口由真实最长视频profile、累计条件／queries和历史曝光尺度决定，在看到正式分数前登记中间与末尾节点。
  保存点使用50或100的倍数，但不机械继承旧50/100停止点、固定一小时或任意短步数作为充分学习证明。
- 有实质获取和保持证据可继续；连续有信息量节点不改善时先综合判断，不靠无限续训或无依据的seed/LR/rank/scale/width小扫延长。
  不人为规定总轮数；Owner明确的预算与次数上限必须遵守，不能以“探索”名义绕过。
- 每轮报告per-task、per-suite、breadth、retained/gained/lost、churn、相邻success-set重合、真实曝光与墙钟。
  区分可复现工程错误、有效科学non-pass和证据不足；诊断应能区分竞争解释，不能把可疑现象或单个局部指标直接命名为根因。

## 4. 授权与自主协作

- 在owner授权、目标、信息墙与资源合同内连续完成设计、实现、分析、相关修复和有依据的重构，无需逐项询问。
  侧面问题或常规技术检查不停止已授权流程；Owner明确暂停或撤回时停止相应工作，旧授权不能覆盖新暂停。
- 长时间实验按既定顺序后台执行，结束后统一核验和报告；不持续盯看进度／分数或逐分钟汇报。
  GPU／存储只在实际launch需要时检查；发生需要处理的失败时介入，保留原失败证据。
- Active design是须完整理解和落实的起点；有充分证据需要修订时先说明理由并同步合同，不强行执行已知错误配置，
  不静默缩水或仅为实现方便偏离科学方法。是否回到v5.2由owner决定。
- 改变科学目标／信息墙、引入未授权数据或资源、重大且无法裁决的路线歧义、删除唯一或所有权不明资产时，带具体事实交owner判断；
  不创造额外审批流程，不把完整专家审查变成每轮开工前置。
- 未经owner明确授权不向外部专家或他人发送消息；需要时提供可复制prompt。只有owner明确要求时创建或设置goal。

## 5. GPU效率、工程与资产生命周期

- 从算法设计时考虑训练、物化和闭环总成本，以真实LoRA/s、samples/s、墙钟、利用率与峰值显存衡量；不以最低显存、占满显存或堆卡代替效率。
  真实长视频的明显计算失衡应先修正，吞吐阈值不能取代科学判断。
- 物理batch、chunk和设备分工不改变完整输入、任务权重、全局更新和checkpoint选择。MT-BC应能在完整更新节点按可用卡数恢复并重分片；同拓扑恢复保持原exact-resume语义，换拓扑时保留逻辑查询流、模型、optimizer和scheduler，但逐rank RNG及低位数值轨迹可以改变，须明确登记。接受正常BF16/TF32与高效kernel低位差异。
- 节点内吞吐以完整训练更新和评测完成时间衡量，避免一张慢卡或一种长任务使其他卡长期等待。训练按任务成本均衡物理分片，评测使用动态队列和persistent workers；不为消除慢任务而漏评、缩短horizon、重加权任务或降低正式行数。空出的合格GPU在下一个完整阶段边界优先复用，跨节点迁移训练须从同一条轨迹的完整checkpoint恢复。
  GPU并发上限、两节点live检查、NUMA/NCCL、quota、formal frozen commit和Git集成按[AGENTS](../AGENTS.md)执行，不在本文复制运行规则。
- 复用canonical数据、source、环境与资产。清理须覆盖仓库文件及其过时正文，不仅更换输出位置或建立in-tree archive。
  退役内容由Git、已有历史索引和formal证据保存；明确temporary／duplicate内容才删除，保留数据集、源模型、唯一checkpoint与正式原件。
- 活动树只有一个canonical实现，无平行fallback。验证按实际声明取最小有效范围；不增加防御性hash、逐tensor扫描或无意义测试。
  普通非必要整理不拖延科学节点；owner明确要求清理时完成实际清理和相关引用更新。

## 6. 沟通和交接

- 默认实用中文，先回答问题，再给关键证据和边界。Owner主要语音输入，主动修正明显同音词、术语与断句错误。
  讨论采用共同推导的方式，不把未接受建议写成owner要求，也不反复展开已对齐的adapter／信息墙边界。
- 正常训练和后台轮询不反复播报步数、句柄存活或“继续等待”；在完整结果、实质变化、需要owner处理或被询问时汇报。
  静默不表示停止工作，后台观察仍按授权继续。
- 文档单一职责：concept解释科学动机，active design规定方法，findings保存跨轮结论，research_history索引历史证据，
  task_plan保存当前计划，progress保存授权与实际状态，AGENTS保存稳定合同。旧快照不追加到当前状态文件。
- 临时handoff只在真实交接时使用，消费后删除；不能独占稳定要求或证据。跨session先读最新状态，按需追溯历史，不要求owner重讲全过程。
