# 对完整方法判断有用的正反证据

截至2026-10-07已完成结果。正式400均为固定Validation8；seen144与局部面板不作为选模型的替代。
R/G/L分别是相对指定参照保留、新增、丢失的成功行。不同架构和训练长度的分数不自动构成单因果比较。

## 1. 强参照与必须解释的阳性

原T训练36meta tasks。早期T/U在450为122/143，900为148/132，1080为147/128；绑定效应随学习阶段改变。
T2340为161/400，MT300为153/400；成熟后续T点147/154/160/154/159，没有持续稳定地拉开参照。
selected T correct161、same-task-other150、public103、cross-suite-wrong65。
两条合法视频相对public分别新增82/73、丢失24/26，其中共同新增64、共同丢失14；两视频间R131/G30/L19。
这包含跨合法视频重现的条件收益，也保留明显的换视频交换，不能将二者任意省略。
public-only是同一模型删去视频残差的诊断，并不是另训的强language-only baseline。

| 固定held task | 3 | 6 | 11 | 16 | 23 | 26 | 31 | 39 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT300 | 41 | 7 | 36 | 10 | 0 | 35 | 24 | 0 |
| T2340 | 45 | 6 | 45 | 5 | 0 | 36 | 24 | 0 |

优势主要集中在已相对较强任务；task23/39持续零，不能用总体微涨掩盖覆盖缺口。
原始学习曲线、每task与逐行证据直接复用[旧EVIDENCE](../20260930_t_architecture/EVIDENCE.md)、
[validation_rows.csv](../20260930_t_architecture/validation_rows.csv)、
[selected_condition_rows.csv](../20260930_t_architecture/selected_condition_rows.csv)。
旧材料也包含change_clock、public共同训练与Context，不只保留最弱参照。

## 2. 主要后续干预

| 干预及实际改变 | 已完成结果 | 支持与限制 | 可直接访问的原件/源码 |
| --- | --- | --- | --- |
| **条件读写C**：4层因果解释器运输动态，S改变A，M用最终A重编译；共同真实FM | C450/900 held137/140；seen91/115。held相邻R107/G33/L30。Context450/900为126/151，T同期122/148 | 已见映射学习明显增长，未转为held改善。Context还用了full+public损失，C为full-only，不能把C/Context差异单归S | [C450](../../analyses/conditional_read_write450_evidence_20261002.json)、[C900](../../analyses/conditional_read_write900_evidence_20261002.json)、[实际C900代码](https://github.com/LinFyM/EMBER/tree/85919994/src/ember/operator_writer)；本目录CSV补原行 |
| **support12→71**：同C450父点继续180updates，固定target13,440queries及权重；总meta tasks36→95 | C12/D71在630为154/156，R129/G27/L25；seen99/98。C完整曲线137→154→140，seen91→99→115 | 这段额外任务覆盖没有稳定收益；59个新增task只约3–4次访问。不是fresh95长期训练，也不能宣布扩任务原则失败 | [汇总](../../analyses/conditional_support_diversity_evidence_20261002.json)、[实际D71代码](https://github.com/LinFyM/EMBER/tree/796d7a9e/src/ember/operator_writer) |
| **动作校准**：先检验两帧native转移动作读出，再fresh Gamma将预测q用于C的S/M Value，真实FM+动作辅助 | P/F留task MSE .11420/.14249，原生点估计μ .13237；P相对F约99%收益来自5action区间均值。完整校准450 held122、seen77，原C450为137/91 | 局部动作读出好于静态F，不等于可迁移控制规则；P对μ区间跨0且task32不利。一次完整实例更差，未做900/Reader蒸馏 | [设计](../../designs/control_calibrated_read_write_design.md)、[实际代码](https://github.com/LinFyM/EMBER/tree/2c630fb3/src/ember/operator_writer)、本目录[功能摘录](functional_readbacks.json)及CSV |
| **实际动作/效果信用A/J**：冻结C900的β/native/解释器/A，仅学共享38组B头64；J加真实1–5步位移效果监督 | 4tasks×2videos×4inits面板：parent13/A24/J22（/32；仅16物理起点）。J相对A G2/L4。A64完整held143，parent140，MT153，T161；A64相对parentR121/G22/L19 | 局部可学性不等于shared跨任务泛化；J真实效果预测整体未优于零。不是新selected模型，不能将局部24/32拼成新架构资格 | [学习代码](https://github.com/LinFyM/EMBER/tree/b0df34ee/src/ember/operator_writer)、[A64读取代码](https://github.com/LinFyM/EMBER/tree/d56f6fa2/src/ember/operator_writer)、本目录功能摘录及CSV |
| **独立共享RL**：从T2340，fresh optimizer，72updates，1,152个train on-policy episodes；不混FM | 63/72官方ODE154/158，parent161；72相对parentR141/G17/L20。seen parent ODE/SDE109/109，72为102/109。288组中140组非零LOO | 确有实际return信用，不是全零reward。SDE自身也没进步，不能只归因ODE/SDE不匹配；只否定本预算/估计器的既得突破，非一切RL | [设计](../../designs/denoising_return_writer_learning_20261006.md)、[信用覆盖](../../analyses/denoising_credit_coverage_20261007.json)、[实际代码](https://github.com/LinFyM/EMBER/tree/1c90e7d5/src/ember/operator_writer)、本目录CSV |
| **关系G/F与输入替换P/Q**：G从native预测匿名关系再S/M，辅助监督+F蒸馏；F读取GT教学及自身关系，bare source H/v上动作残差 | G360/450 held131/126，seen G84/F66，T109。统一同链P0/Q0/P180/Q180为81/82/78/81（/144）；换GT终点+3来自保留差异 | F不是强MT/β上的oracle。不能以F66宣布正确关系无价值，也不能将GT替换看成隔离全部视觉/蒸馏/编译瓶颈；双方joint-fail新增都3 | [G完整消费](../../analyses/relation450_main_consumption_20261007.json)、[P/Q](../../analyses/relation_input_compilation_20261007.json)、[实际G/F代码](https://github.com/LinFyM/EMBER/tree/85614d9c/src/ember/relation_writer) |
| **特权state→示范动作1NN**：固定寻址几何，仅比较Value；再从T真实50步后交接相同NN | true teacher actions91、bare source动作0（/144）；校准面板原生μ6→Gamma预测22，仍低于true91。T50接力T109→91，R80/G11/L29；T失败35救11 | 11新增是真实局部控制；29丢失与原NN相同总分下的不同成功集同时存在。不能把oracle union或特权Value当可靠纠正教师，更不能部署 | [true动作](../../analyses/privileged_action_memory_control_20261007.json)、[Gamma Value](../../analyses/calibrated_action_memory_control_20261007.json)、[T50接力](../../analyses/teacher_state_handoff_20261007.json) |
| **强MT上的在线视频M/V**：同MT300继续128/14,336queries；V两层视频encoder+18层Reader直接查询自身hidden，教学β信用真实；非EMBER部署 | M93/V88（/144），R84/G4/L9；原MT93/T109。新增训练留出4tasks：M6/V3，V同task另一video5（/16） | 同时改变教师特征、地址绑定和写入位置，不能视作只解除编译瓶颈的强T上界。MT早见过4tasks，它们不是全程held。未认证强教师，停止延长/蒸馏 | [完整证据](../../analyses/native_video_control_diagnostic_20261007.json)、[实际V代码](https://github.com/LinFyM/EMBER/tree/d76e42fd/src/ember/native_video_control) |

各数字的完整配对分布优先看链接JSON及本目录outcome_summary；不能用分母相同掩盖场景、视频或参数干预不同。
C、动作校准、A/J及RL新增CSV共有3,504行，其余已在Git的逐行/成功集证据直接引用，避免多份权威副本。

## 3. 更早、最相近的机制必须继承

这些历史协议不全相同，不与当前400分数直接排名。作用是约束“某个模块一加就能解决”的推断。

| 历史机制 | 对本次判断的约束 | 入口 |
| --- | --- | --- |
| LocalField：真实X关联A、完整双向时间上下文与真实功能信用 | 不能把“层对应”“整段上下文”“可回传”当从未有过的新解法；须比较实际Value、寻址、数据和学习条件 | [历史实现](https://github.com/LinFyM/EMBER/tree/50dafb05/src/ember)、[历史索引](../../research_history.md) |
| ProcessPullback：视频PCA A、冻结G与共享L/R；旧协议有限validation 50/75/79 | 局部free q18/free AB46对原20/96并未保证共享学习；可表示、局部拟合、跨task可学必须分开 | [历史实现](https://github.com/LinFyM/EMBER/tree/85ecfd18/src/ember)、[历史索引](../../research_history.md) |
| P/I：相同新数据、不同监督匹配，124/91→108/107 | 自身访态与成功轨迹信用已有实测，不能只说“去训练失败状态”而不解释为何这次不同 | [历史实现](https://github.com/LinFyM/EMBER/tree/bc729e86/src/ember)、[findings](../../../findings.md) |
| v5.2纯FM及后来的正例复验 | FM曾产生真实条件控制，不能从最近几个阴性推导FM原则上不可能；旧split与新协议不得混排 | [FM历史回顾](../../analyses/flow_supervision_history_20260926.md)、[正例复验审计](../../analyses/frozen_positive_replication_audit.md) |
| 多种公共/条件功能分工、成功保持与跨任务adapter交换 | 不能把公共参数无条件称为“全部知识”或把任意成功adapter称为可组合操作；也不能把未做过的完整函数保持说成已被否定 | [上次公共基底评审](../20260930_t_architecture/PUBLIC_BASE_REVIEW.md)、findings§386–387 |

## 4. 当前主讨论的初步判断，允许推翻

1. **条件控制已有局部实质，跨任务可靠获取与调用尚无充分方法。** T正例与换视频敏感性并存；
   它约束了“视频完全无用”，却未区分通用操作规则、状态关联或训练任务模板调制。
2. **更强表达和更近标签未自然变成泛化。** C的seen增长、动作辅助及特权读出说明要区分学习到训练映射和建立可迁移联系；
   但当前小范围干预还不能判定数据、表示、寻址、共享梯度或闭环分布谁是主因。
3. **缺少强教师的诊断不能为编译器定罪。** F、NN和V的实际能力没有形成稳定强上界；
   它们的失败也不能替现有编译器免责，因为比较同时改变了其它环节。
4. **下一判断应是完整原理的选择。** 不再用“把某个局部读出再提高一点”自动递推到新架构；
   新投入必须说明对完整控制获取和迁移的可失败预测，解释最近似历史为何不覆盖它。

上述判断仍有不确定性，不是把专家结论预填好。最需要专家贡献的是把这些证据压缩成有预测力的机制取舍，
包括指出我们错把什么当成目标、错排了哪些假说，以及哪些现有数据已经足以停止一条路线。

## 5. 导出统计如何复核

`episode_outcomes.csv`每行包含panel、suite、task/global task、init、teacher、video ordinal、环境/策略seed root、成败及步数。
按panel求success之和可复算总分；`outcome_summary.json`登记每panel来源、角色、逐task/suite、breadth及成对R/G/L/churn。
配对键包含task/init/teacher，不将同物理起点不同teacher当独立物理场景；官方K1还核查每task50条不同teacher。
对T/MT的配对读旧CSV参照，不重复导出旧panel。seed root与case键核对不等于重新证明完整模拟器状态/噪声张量相同。

`functional_readbacks.json`是先前本地CPU分析的有限摘录，保留来源、标签/面板边界及不利分解。
它不包含全部连续轨迹或大张量；若审阅结论取决于那些尚不可远程审查的细节，应将该点标为未独立核实，
而不是用摘要或主讨论措辞代替证据。
