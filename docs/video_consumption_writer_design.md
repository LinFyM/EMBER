# 普通FM下的语义条件化过程消费：自主接续设计

日期：2026-09-11。本轮及两项有限续训已完成并退为历史合同，结果见findings§51–53；当前active为[非held meta-task扩展设计](nonheld_meta_writer_design.md)。本设计在Owner最新明确授权下接续：由Codex自主整理专家与独立判断、修改方法、实验、分析并据证据再次修正，Owner醒来后审查。此前讨论暂停与禁止再次完整训练的阶段限制被本次授权覆盖；科学目标、信息墙、固定数据、资源与Git合同保持。

## 1. 目标和决策依据

目标是correct strict validation400>145，且相邻稳定、低churn、足够breadth、四suite和Goal/Long贡献、换正确视频稳健，以及冻结后独立视频因果资格；模型初次生成唯一完整38-target rank16 LoRA即执行，无部署交互。监督用普通跨episode FM，不叠加辅助loss/功能保持/RL；测试集封存。

[专家第二轮原文](review_materials/20260911/expert_review_round2.md)正面承认v5.2正常正样本动作监督已学出正确视频依赖，撤回先保持再视频的顺序。v5.2原五臂132/138/74/82/83（各400）与task-complete120/109/107/111/124共同约束解释：纯FM可行，但旧结构或分路名称不保证成功。历史不同Meta、数据池、曝光和学习率时标均非已识别原因。

当前最强Horizon为45e16633/off init7 macro400=126/400，init11=119；训练任务冻结视频面板53/60/59/59/55/52/58/53/53（各96），尚无稳定正确过程增量。已确认查询几何不是充分解释；P/C/D端点与局部更新证明功能破坏存在，不证明其是当前视频失效主因。主假设是统一P4的消费接口难以让普通FM利用过程变化；竞争解释是每更新视频条件分配不足。两者均是待检验假设。

独立判断保留三个风险：P4可能尚未学到可迁移操作信息，改末端不能凭空补齐；新增语义S可能独立解释收益；无序多帧仍包含状态变化，不能被称作单帧static或无动态模型。因此本轮只把C/S差异作为显式有序过程计算的证据，最终视频必要性另行裁决。

## 2. 架构、共享接口与变量

复用真实prefix/Z/KV、Action Meta rank4、固定probe、完整50-H、stride5、agentview及source执行图。保留现有过去关系图的定义，完整公式参考[前代设计§3–5](horizon_relation_video_writer_design.md)。当前只改编译消费及其匹配参照，不开放VL/Text Meta、不改D/rank、不切dual。

三臂均输出原76个A/B tensor，使用同一NativeFactorLoRADecoder与identity初始化。实验期一个runtime、显式有界配置三种模式；旧运行由冻结commit复现，不复制训练器或设永久fallback。Codex拥有比较分支；选择后移除活动源码中被淘汰的模式与专用测试，历史通过Git/config/结果保留。

### R：当前off与两个独立K1教学条件

过程编码与Compiler等同45e16633的local_h_read：关闭额外Compiler语言query，保留逐帧contextual language、H-read和独立D。保留原构造次序和随机初始化以便与旧off200作对照；旧off400只作能力参照。训练改变每task视频分配，见§3。

### C：先语义后过程的完整编译

```text
真实逐帧Z中的exact task tokens与image tokens
  → 逐task-token读取该帧image内容
  → 保留token轴的无时间位置集合读取及语言轴组合 → S[L,d]

原生完整R[T,50,1024] + 逐帧language/Z
  → 原四组过去关系、视觉核实、GRU、单向长程与三次回写 → P[T,d]

target/rank身份先读S → s[38,16,d]
  → 分别归一化身份与s，条件化读取P的变化Value → p[38,16,d]
  → 同槽归一化拼接+非线性融合+一个槽间组合块 → c
  → 原独立D → 唯一完整LoRA
```

S读取复用已有原生Z，不新增Gemma forward。每帧有效task tokens投影到256维，作为图文条件query读取该帧有效image tokens；保留task-token轴，不早期压为一个码。逐token沿帧集合attention无视频时钟/ordinal，随后两层语言轴组合使用task-token位置。可以用真实text embedding作为集合检索query，但残差内容来自图文证据，不把纯文本向量直接加入策略内容。

过程检索K可携带原帧索引位置，V为每视频P减其有效帧均值，先逐视频中心化再集合读取。V和输出投影无bias，不能经bias重建被去掉的共同内容；不宣称中心化剩余就是物理动态。跨视频读取使用每video−logT先验，不平均最终LoRA。全视频中心化只在最终编译发生，不回写给前面逐帧状态，保持过去前缀合同。

融合允许模型学习语义与过程的相对作用，不设强制非零动态gate，不降低静态/错误表现制造差额。显式结构职责只是归纳偏置，不能替代行为证据。

### S：匹配的主动训练无序帧集合参照

与C使用同一语义encoder、target/rank读取、中心化变化读取、融合与D。共享模块以同一固定初始化种子和稳定构造顺序初始化；不靠加载已训练checkpoint对齐。只替换过程encoder及过程K的时间位置使用：

- S每帧独立处理同样的完整R和Z，不接收P4、视频frame ordinal、长度时钟或跨帧有向边。
- 采用四组逐帧完整H self-attention、真实Z cross-attention和FFN，然后合法H-read；H位置表示动作horizon，不是视频时间。
- 最后输出每帧内容作为集合，过程检索K不加视频时间位置。保持所有真实帧，故它能比较状态并可能推断部分过程，不称为纯static。
- 所有输入帧的联合重排对S输出应保持置换不变；该合成结构测试不使用真实shuffle/reverse结果选模型。

C/S共享读取与融合，把差异集中在过去关系图相对于独立逐帧证据处理；它们的encoder函数类与计算量仍不完全相同，结果不唯一归因于顺序。S有充分逐帧处理能力，不通过删掉完整H、减少可见帧或弱decoder削弱参照。不新增第四条语义only训练线。

## 3. 数据、目标、随机性与条件分配

固定train24；teacher0–15、FM actions16–41、independent actions42–45、诊断teacher46–49，严格跨episode。无额外meta tasks、无validation/test梯度；每次生成K=1。每update仍每suite均匀抽1task，共4task/256queries；每task的64queries由两条无放回正确视频各承担32，分别生成LoRA，不做K2、不平均LoRA。

每条件FM均值乘1/8，每task合计1/4；所有条件在同一参数版本产生梯度，最后全局SUM、clip、optimizer/scheduler各一次。支持同节点1–6张真正有用的GPU，以8条件的真实视频cost分配，不把task当不可再分单元限制为4卡。

保留原独立task/video/query随机流。task序列、每task动作64样本及其完整flow noise/time先定义，再按条件切成32+32；一次更新不因物理microbatch或GPU分配改变逻辑样本。新视频采样每task一次sample2，不改变task/query流。保存条件index、query offset、原完整policy RNG seed、实际episode/frame、每task occurrences及所有sampler/RNG状态，resume锁world topology。

保留AdamW lr3e-5、betas(.9,.95)、eps1e-8、decay1e-4、clip1、warmup8后恒定。R与旧off参照不另改学习率时标；共享模块构造/初始化与C/S差异明确记录。兼容低位数值差异，不追求跨拓扑逐bit一致。

## 4. 学习和闭环节点

先CPU合同检查，后真实最长视频FM/replay profile与最短恢复smoke；均为临时证据，不初始化formal模型。formal来自clean pushed detached worktree，三臂全部fresh。以真实吞吐决定卡数与microbatch，更新不改变逻辑batch。

初轮R/C/S各100、200两个登记checkpoint，each200=51,200FM queries/1,600条件，合计153,600queries/4,800条件。该上限是首轮投入，不是整个函数类平台证明；大约一小时段的实际时长按profile记录，若明显超出先优化吞吐或分段保留合同，不能默默缩水。

每臂100/200执行canonical correct strict400（原seed20260907、per_init_ordinal、每task50视频各一次），复用固定task/state/video/env/policy RNG映射。200补train24×states32–35×held46–49固定96行；独立动作验证0/200、每task128queries、teacher46+task%4、seed20260908+task、actions42–45。只使用这些预登记正样本节点进行选择与推进，不用旧wrong/shuffle/reverse作搜索信号。

每个结果报告逐task/suite、breadth、retained/gained/lost、churn/Jaccard、unique条件覆盖及曝光/墙钟。保存完整state/optimizer/scheduler/sampler/rank RNG/source/config/command。未见task迁移只由validation结果支持，不从train96推断。

如100节点明确出现工程错误先修复；若合理训练但低分，先看200获取趋势，不因小样本强行判平台。若超过/接近目标或有广泛实质改善，追加按100间隔的相邻正式节点，结果前登记，保留一小时左右段；不机械受初轮200限制而停止有效学习。若两点明显弱且无获取改善，停止该候选，分析后才决定下一主要改动，不小扫seed/LR/rank救分。

## 5. 裁决与再次修正

- R与C相当或更强：较简单条件分配优先；不把一般能力改善自动解释为动态修复。
- C同时改善相对R的absolute、弱task和换视频，并优于充分训练S：支持该消费组合及显式过程的候选增量，进入相邻资格。
- C改善但S相当：保留语义读取进步，过程增量未分离；不能直接断言两者都不读过程。
- S更强：保留强多帧语义参照，检查当前有向关系计算的学习代价；不通过削弱S或加深时序图制造差额。
- 共同低分：先依据独立动作/训练闭环/行为阶段区分获取、控制和迁移，再在读取适配、输出消费或真实任务支持中选择一个有依据的修正。无新信息不重复同类架构。

主线不得停在surrogate或漂亮中心化统计。新增分析必须能改变下一行动，优先真实输入和行为，避免完整因果矩阵反复消耗。功能保持仅在候选有明确获取后出现可复现破坏时针对性考虑；辅助动作头、Meta扩展、decoder重构、双视角都是条件性后续，不自动同时启动。

资格阶段先依correct相邻节点冻结候选，补same-task-other strict400；最终correct/wrong/static/no-video/shuffle/reverse的用途、映射与独立性另行在结果前登记。已看过的诊断不冒充最终未触碰证据。learned language-only必须用真实文本输入；single-frame用预定首帧不附带视频时钟。当前不自动各加完整训练，正式方法需按主张补齐公平训练参照。Test保持封存到最终方法冻结。

## 6. 工程与资源

主进程拥有架构、配置和裁决；可独立委派条件调度/梯度权重实现，独立worktree、不重叠写入。现有learning_data/supervised/training/functional拥有训练，无第二个runner；semantic/readout及frame参照用至多两个凝聚模块，HorizonRelationWriter保持统一入口。材料原文保留，当前状态只写task_plan/findings/progress。

2026-09-11新工作前strg01/data1 quota used743,343,260KiB、soft1,073,741,824、hard1,084,227,584；共享84TiB。当前源树（排除runs/data/models/env/git但含旧tmp/worktrees）15GiB。新实现worktrees预计<1GiB；三臂checkpoint、临时profile、物化bank和闭环证据初轮峰值预留96GiB，低于独立quota余量。大模型/数据/env复用；每次正式launch前更新所需资源证据，GPU同时检查gpu01/gpu02，总量≤6物理卡，不占空卡、不干扰他人。

本文件是首个完整执行合同；后续实质修订记录理由和实际变化，不把专家方案当不可修改硬约束。目标是否完成由全部科学资格判断；Owner休息期间继续推进，正常轮询静默，完整结果与实质决策才进入进度。

## 7. 首轮后续：保留强候选的监督续训（2026-09-11）

R/C/S首轮全部结束：validation100→200分别37→83、43→50、59→45，train96分别46/49/52；同51,200queries旧off200为108/40。两项改动未带来未见任务净收益，C/S不追加，不从S100峰值推导第四种架构。旧off init7从200的108/40提升到400的126/56，停止来自当时阶段预算，尚未证明监督平台。当前选择检查该强候选的剩余学习，主要变量只有继续监督曝光。

预登记全局500、600两个checkpoint：各correct strict400，600补同口径train96，500/600做既有train24独立动作验证。新增200updates=51,200queries/800独立K1条件，累计600=153,600queries。原train24、每task一个K1条件/64queries、agentview、source、完整H/P4/local_h_read/D、Meta、AdamW与恒定LR均不变。不得添加保持loss、RL、换视角、seed或LR搜索。若两个节点无实质改善则不继续惯性延长；如有接近目标的广泛改善，再登记相邻节点和必要资格，当前不预许无限续训。

原checkpoint位于horizon_causal_local_h_read_seed7_20260909/macro400，runtime为clean pushed detached45e16633，mode/stage为exploratory/horizon_causal_learning_exploratory。原gpu01 1/5/6中5/6当前被他人满负载使用，不能exact-resume原拓扑。按owner允许的受控拓扑迁移，在新run root保留原三rank及其完整Writer/Meta/probe、optimizer/scheduler、sampler/400cursor和各rank Python/NumPy/Torch CPU/CUDA RNG，映射到同节点三张适用A40，micro8/8/8、NUMA与deferred NCCL保持。物理host/UUID/affinity变更明确记录；不声称逐bit等价或原拓扑exact-resume。

采用原冻结runtime的原生恢复入口，不改训练代码或建立第二套trainer。迁移仅新建有lineage的run contract、更新预登记evidence节点、复用不可变checkpoint文件并复制小型历史日志；模型/optimizer/RNG文件不重写，原run和历史标记不改。新run的native contract校验仍逐项核对config/source/model/topology，恢复仍核对stage、world、400cursor与完整学习状态。迁移报告保存父run/commit、字段变化、逻辑rank映射和下一批采样检查；失败则停止新run，原checkpoint可直接回退。新run继续标记探索性续训，不把旧探索性checkpoint转换为formal fresh或直接用于最终资格选择；如能力得到支持，正式资格须另行明确fresh复现及独立controls。

新run及材料位于runs/outputs/horizon_off7_continuation_20260911与runs/analysis/video_consumption_20260911/continuation。新增两份checkpoint约8.3GiB、validation/train96 LoRA约4.4GiB、日志及物化临时文件峰值预算16GiB；父checkpoint以同filesystem hardlink复用。launch前strg01/data1 used782,678,700KiB/soft1,073,741,824KiB，共享83TiB，预算有余量。实际卡和完整命令由launch记录固定，两节点现场复核后执行。

## 8. R的有限后续获取节点（2026-09-11）

off7追加500为73/400，低于父400的126；600训练已结束，独立动作FM400/500/600为0.104946/0.104097/0.104786，闭环与train96待完成。此前停止强旧候选确实不能证明平台，但继续曝光也未在首个追加节点保住已有能力。暂不根据这一点增加保持loss、VL Meta或另一种decoder。

R的100→200是37→83，新增60/丢失14，Goal10→37、Long0→5，仍有获取趋势；这与C43→50及S59→45不同。R虽在51,200queries时低于旧off200108，却尚未完成旧off400的102,400queries，首轮结论只否定该预算下的净迁移收益。为区分慢获取与较差的最终迁移，**只追加R的200→400段**，预登记300/400两个correct strict400、400同口径train96及冻结后的train24独立动作验证。新增51,200queries/1,600K1条件；R累计400为102,400queries/3,200条件，与旧off400动作曝光相同。C/S不追加，不恢复未采纳架构，不更改教学池或动作监督。

从R原macro200在原run root、冻结7fedbe85、原gpu02 4/6、world2、micro6/6完整exact-resume；原config/source/采样/模型/optimizer/scheduler/RNG与topology不变。仅通过原生checkpoint CLI登记本段300/400。原config自动动作诊断仍为0/200，400诊断另用同一原生validate方法只读执行，原seed20260908、每task128queries、teacher46+task%4/actions42–45，无梯度或checkpoint选择。不存在初始化迁移或新runner。若原拓扑不合适，等待此合同以外的执行选择，不擅改为另一初始化。

本段是R唯一登记的追加，不自动许可400之后继续。300/400须同时报告对200及旧off400的逐task/suite、breadth与R/G/L/churn；只有实质的广泛行为改善才讨论进一步节点，否则该条件分配路线结束，回到已有证据支持的机制判断。不能以C/S早期失败否定全部消费结构，也不把R的更多条件调用本身当成视频必要性。

材料位于runs/analysis/video_consumption_20260911/r/continuation；复用原R output root。新增两checkpoint、896个LoRA及诊断小文件峰值预留16GiB，独立quota与两节点在launch前复核。与off600评测合计始终不超过六张实际工作GPU；off600完整结果继续独立完成。
