# EMBER owner requirements

本文保存owner的稳定目标、研究原则与协作要求。Owner最新明确表达优先于本文；本文优先于[AGENTS](../AGENTS.md)中的默认合同。
当前授权和实际状态只看[progress](../progress.md)，执行计划只看[task_plan](../task_plan.md)。历史讨论与旧实验不构成重新启动授权。

## 当前阶段的优先级：Process Pullback Writer（Owner 2026-09-14）

五个问题讨论后确定推进[Process Pullback Writer](process_pullback_writer_design.md)。按完整方案实施、训练并作有依据的迭代，
最后汇报完整结果；“最后一轮”不限制为只训练一次。若合理学习后持续缺少正向信号，应降低对实际检验组合的支持并明确报告，
停止无信息重复；是否结束坚持或回到v5.2完全由owner决定，agent不得自行切换路线。

本阶段优先检验正确视频的实际执行价值、视频特异性与能力保持，暂不强制长期的>145/400性能线。
默认输入为一条演示的同步双路RGB（K=1）；采用纯跨episode FM，不要求教学video的动作标注或专门q监督，本轮不开展RL或Test。
允许完整视频双向理解，须保留真实顺序和操作前置关系。接受选定冻结模型上的correct相对wrong／shuffled／reversed证据，
不再强制另训独立frame_set模型；正确条件能力、相邻保持与same-task换视频鲁棒性仍须检验，不能靠错误条件退化制造收益。

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
- 一次Writer调用内部可对授权视频或native activations作固定、只读、多阶段读取与重放，包括本设计的冻结source导数计算。
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

## 3. 证据与推进判断

- 长期正式性能目标为validation8 strict single-checkpoint paired correct严格>145/400，并有相邻稳定、低churn、高breadth、
  四suite非零、Goal/Long贡献、same-task换视频及最终视频因果证据。本阶段按顶部优先级判断，不额外强制这条分数线。
- 闭环实际能力先于loss、reconstruction、norm/rank/cosine、内部margin及surrogate。充分且可比的学习后仍弱于source／SFT参照，
  属于严重能力缺口；小幅涨分或loss下降不能将其降格成调参问题，也不能由此唯一归因某个模块。
- 能力与相邻资格成立后补same-task-other，选定并冻结单checkpoint，再做wrong／no-video／shuffled／reversed最终controls。
  controls不进入训练、checkpoint选择或架构修正；旧实验按原注册标准保留，不因新要求重判。
- 正式评测使用single-checkpoint完整400配对行，不用80-row screen、checkpoint union或融合选模型。
  K1每task、每臂、每轮50个init对应全部50条合法teacher videos各一次，跨checkpoint和controls复用固定canonical映射。
- 主FM来自同task跨episode执行queries。固定validation/test不产生梯度；扩展non-held meta tasks须先审计held及重复specification排除，
  登记allowlist/provenance。更多同task episodes不等于更多独立meta-task映射，不制造人工process或新仿真任务绕开问题。
- 本轮Writer和两组Meta采用fresh联合监督，optimizer、scheduler、sampler和RNG均fresh；合法identity不要求每个张量随机非零。
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
- Active design是须完整理解和落实的起点；有充分证据需要修订时先说明理由并同步合同，不强行执行已知错误配置，
  不静默缩水或仅为实现方便偏离科学方法。是否回到v5.2由owner决定。
- 改变科学目标／信息墙、引入未授权数据或资源、重大且无法裁决的路线歧义、删除唯一或所有权不明资产时，带具体事实交owner判断；
  不创造额外审批流程，不把完整专家审查变成每轮开工前置。
- 未经owner明确授权不向外部专家或他人发送消息；需要时提供可复制prompt。只有owner明确要求时创建或设置goal。

## 5. GPU效率、工程与资产生命周期

- 从算法设计时考虑训练、物化和闭环总成本，以真实LoRA/s、samples/s、墙钟、利用率与峰值显存衡量；不以最低显存、占满显存或堆卡代替效率。
  真实长视频的明显计算失衡应先修正，吞吐阈值不能取代科学判断。
- 物理batch、chunk和设备分工不改变完整输入、任务权重、全局更新、checkpoint或exact-resume语义；接受正常BF16/TF32与高效kernel低位差异。
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
