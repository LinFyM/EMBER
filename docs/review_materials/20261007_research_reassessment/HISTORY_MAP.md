# 从历史推理链进入本次审阅

截至2026-10-07。本页帮助专家先自行理解 EMBER 做过什么、为什么做、哪些判断应当保留或修正；
不是把主讨论的解释预设成答案。可核实的 Git 历史起点为2026-07-17，“最近两周”指2026-09-24至10-07。
更早的未归档讨论不在本页已核实范围内。

## 先校准目标和比较口径

- [最初 README](https://github.com/LinFyM/EMBER/blob/1226236beb342d56c7741164e8da1c4eb8d70725/README.md)
  将 Writer 定位为给 task-local RL 提供可用起点及适配几何，允许 language/video 任一或共同输入。
  今天的[目标](../../current_owner_requirements.md)已经聚焦 **exact language + action-hidden video → 一次生成完整 LoRA → 初次闭环即有效**。
  请审查目标演变，不把今天的全部要求倒写成7月已经冻结的合同。
- 旧 split、source、视频池和动作对齐，与9月覆盖重划后的协议不同。旧 v6 143、GOMQ 151、9月19日同视频辅助174，
  以及当前 T161/MT153，不能跨协议直接排名。每项比较回到原合同和相同面板。
- 早期 source47/48、SFT109/107、特权专家250各有自己的协议；都不能代替当前强 MT。
  旧 `>145/400` 是历史接受标准，不是本次目标。
- Test 和 shuffled/reversed 封存结果不用于本轮方法判断。历史文件混有这些内容时跳过对应节、列、行；
  原文仍保留不代表本次可以拿它们选架构。特别是9月17日审计的§6.1和历史中的 Test 条目。

## 7月至9月23日：主要路线及其理由

| 阶段 | 当时试图解决什么、为什么这样做 | 实际获得什么，哪些结论不能成立 | 远程入口 |
| --- | --- | --- | --- |
| **7月17–21日：SmolVLA到generic π0.5** | 先检验 Writer 是否可产生适配起点，再用RL改善；随后核对 foundation/source-trained base 和标准环境合同 | 早期存在 source 获取、弱迁移及 Writer-only RL退化；随后换至generic π0.5。目标和base身份同时演变，不是今天模型的一次干净消融 | [原目标](https://github.com/LinFyM/EMBER/blob/1226236beb342d56c7741164e8da1c4eb8d70725/docs/concept.md)、[早期发现](https://github.com/LinFyM/EMBER/blob/999df28c2178143a7cc339494af0b2564b07b3cf/findings.md)、[π0.5重置](https://github.com/LinFyM/EMBER/blob/6bb13dee8f971bd4ebcc4e39c9119dbf7f5b38fc/findings.md) |
| **7月下旬–8月23日：完整Writer、多视频、LPCP/GOMQ** | Core/Procedure/FactorHeads用功能监督生成完整LoRA；再用多视频、共享集合表示、原生probe和成功occupancy信用降低条件波动 | v5.2普通FM有真实能力；旧v6有143，GOMQ151→135→131。参数近似不变仍可大量交换成功行；不能把更多视频、低参数差异或一个高点当稳定控制 | [历史§2](../../research_history.md#early-writers)、[v5.2设计](https://github.com/LinFyM/EMBER/blob/529da6bbe290f7393422937aa7cc278cee732107/docs/action_forecast_writer_v5_2_design.md)、[v6设计](https://github.com/LinFyM/EMBER/blob/3a6f801d08facb3e855ab24f84e0b53cb8802e88/docs/action_forecast_writer_v6_design.md)、[9月7日原行索引](../20260907/index.json) |
| **8月19–25日：特权功能、realizer、G1/G2** | 先用task experts的成功功能检验更新空间及原生证据，再希望共享Writer获取这种功能 | 固定span确实损失行为方向，分组后有局部容量；G2有动态读出。它们主要是训练内部held5/算子面板，没有组成合法、有效的完整Writer；人工组合teacher亦非已认证oracle | [历史§3](../../research_history.md#native-capacity)、[完整冻结账本](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/research_history.md)、[当时专家意见](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/expert_review_20260824_native_factor.md) |
| **8月26日–9月2日：Program/bank/primal/routing/chart** | 假设解决当前bank的坐标、容量及寻址问题，可把局部可达功能交给自然视频共享编译器 | 多次free query、task-local或fixed-chart正控成立，接回共享自然条件仍弱；PNBTT只到E1，E2未跑。数学可解与共享获取不同，不能将未执行后段写成失败 | [9月17日完整审计§3](../../analyses/v52_evidence_audit_20260917.md)、[bank专家原文](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/expert_review_20260826_bank_conditioned_native_factor.md)、[全局重审](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/expert_review_20260902_global_route_reassessment.md) |
| **9月2–7日：full response、Axial/Unified、完整P/Q、clone** | 保留完整horizon/原生响应，减少受限输出和分段学习，检验同图共享是否能保住独立任务能力 | 完整38-target出口在四任务短窗有收益；两弱任务clone14/20，共享3或4/20，且共享并非先强后忘。容量、数据、优化仍未单独定位；width256只有训练，没有闭环成绩 | [历史§5–7](../../research_history.md)、[完整P/Q专家原文](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/expert_review_20260905_full_history_joint_process_policy_writer.md)、[既有逐行包](../20260907/README.md) |
| **9月8–11日：Horizon与条件检索** | 用完整H、局部对应、视觉核实及长程处理形成教学过程；随后区分语言内容路径与参数出口 | canonical K1曾110后回落；Compiler额外语言query删除在两初始化有收益，但续训退化、跨视频及广泛保持未解决。局部删除有效不等于唯一根因，也不是完整修复 | [9月8日原评审与修订](../20260908/README.md)、[Horizon设计](../../designs/horizon_relation_video_writer_design.md)、[9月11日原行/方法/评审](../20260911/README.md) |
| **9月12–18日：语义/动作/纠正监督、Pullback、source对齐** | 为教学特征赋予更直接的视觉、动作或原生纠正含义；用真实LoRA功能信用传到控制，并复核较强早期方法的配方 | 特权真纠正有训练侧闭环正例，合法Native Correction/Local Field未继承广泛收益；自由完整AB比受限出口更可达，但共享映射仍弱。source对齐后A曾140，后到108；标签可用、输出可达、视频可共享获取须分别判断 | [纠正审计](../../analyses/native_corrective_closed_loop_audit.md)、[9月17日审计§4–6](../../analyses/v52_evidence_audit_20260917.md)、[新增原件导出](history_records/README.md)、[9月18日专家原文](../20260918/expert_review.md) |
| **9月19–23日：教学自身信用、覆盖与完整条件比较** | 在跨episode主FM之外，将教学自身真实转移的动作信用回传到同一生成LoRA；随后核查任务支持和覆盖 | 旧协议同视频辅助900/1200/1500/1800/2100为149/174/165/160/158，对应匹配跨episode替代125/136/147/136/159。早期收益真实，晚期优势缩小；双视角分支未改善。覆盖重划后重新训练，不能直接沿用174与新MT比较 | [当时提案](../20260919/expert_proposal.md)、[学习结果](../20260919/final_report.md)、[覆盖重训合同](../../designs/coverage_retraining_design.md)、[覆盖诊断](../20260921/coverage_retraining/diagnostics/README.md)、[因果读回](../20260922/writer_causal_diagnostics/README.md) |

9月17日审计逐项记录训练曝光、冻结范围、梯度消费者和未执行部分；它的末尾建议只代表当时假说。
9月7日索引覆盖511份原件、95个rollout面板，早期并非只有文字总结；按 `exports` 找远程文件，`source` 只是原件出处。
G1/G2及部分bank分支仍主要依靠冻结账本、代码、设计、原评审与本轮小型读回，不能声称全部原始张量都可远程重算。

## 最近两周：从局部因果到同算子读写，再到当前停滞

下表给出主线，不用实验数量代表认识进展。每行原设计记录事前逻辑，历史/分析记录实际结果；
专家需要比较二者，判断后继是否真的由证据推出。名称相同的“C”“A”“B”可能属于不同阶段，不能跨行视作同一模型。

| 日期与路线 | 原机制假说与实际干预 | 已完成结果及它没有回答的问题 | 入口 |
| --- | --- | --- | --- |
| **9月24日：四臂与早期控制分解** | 区分直接参数、语言生成、视频生成及教学辅助；随后固定前段和动作通道，定位错误接近怎样影响后续 | 四臂最佳108/124/110/123，source58；Goal固定接手仅换xy27→40，换z27→19，Object无一致修复。局部早期动作有因果效应，没有推出通用训练修正 | [四臂设计](../../designs/conditional_compilation_diagnostics_design.md)、[前段](../../designs/frozen_prefix_causality_design.md)、[通道](../../designs/approach_channel_causality_design.md)、[读出](../../designs/readout_realization_causality_design.md)、[历史9月24日](../../research_history.md) |
| **9月24–25日：关系支持、原生读取和更新信用** | 固定数据规模替换两处对象—目标关系，检验组合支持；再交换读取/后段，检验单项监督、lookahead和成功方向 | Goal四格correct36/37/26/35，other40/37/29/33，完整补关系没有胜过原池；后段映射传递具体损害，但信用/lookahead未恢复完整能力。关系图的可表示性不是网络已学会组合的证据 | [关系支持](../../designs/relational_support_causality_design.md)、[读取交叉](../../designs/native_reader_transfer_causality_design.md)、[单项信用](../../designs/support_slot_credit_causality_design.md)、[lookahead](../../designs/metatask_lookahead_credit_design.md)、[成功方向](../../designs/return_credit_direction_design.md) |
| **9月26日：内容路径与完整机制回顾** | 检验首帧/原生内容和教学辅助的实际作用，避免把一般VLA性质误作EMBER根因 | C0纯跨episode FM完整held120，匹配B121；局部特征干预有收益而未形成完整方法。新的Reader只做工程接通/恢复，正式训练在启动前撤回，不能记为科研阴性 | [语言路径审计](../../analyses/language_content_path_audit_20260926.md)、[FM历史](../../analyses/flow_supervision_history_20260926.md)、[持续机制分析](../../analyses/feature_to_operator_mechanism_20260926.md)、[Reader提案](../../designs/native_conditional_reader_design.md) |
| **9月27–28日：条件速度V/L、参考对应P/I** | 将条件表示绑定到实际执行的共同状态基；另构造同教学参考的不同初始化执行数据，比较正确对应与独立对应监督 | V/L270为151/147，450退到102/112；P/I288为124/91，576为108/107。对应有短期收益，没有稳定超MT；P/I比较隔离对应方式，没有隔离新数据相对原数据的价值 | [条件速度设计](../../designs/conditional_velocity_operator_design.md)、[参考对应设计](../../designs/demonstration_transfer_learning_design.md)、[历史9月27–28日](../../research_history.md) |
| **9月28–30日：T/U与真实教学条件作用** | 用实际A同时定义教学读出和执行输入方向，检验绑定是否优于独立读取；继续到可判断相邻能力 | T/U450为122/143，900为148/132；成熟T最高161、强MT153，相邻较晚T147/154/160/154/159。selected T correct/other/public/wrong161/150/103/65，有真实条件收益和损伤，仍未形成广泛优势 | [上次整套证据](../20260930_t_architecture/README.md)、[学习曲线与原行](../20260930_t_architecture/EVIDENCE.md)、[实际机制](MECHANISM.md) |
| **9月30日–10月2日：公共项、Context、条件读写C及任务扩展** | 专家建议共同学习读取与写入；加入时序解释及条件A/S，再用最终A重编译M；检验support12→71是否帮助迁移 | C seen91→99→115，held137→154→140；Context900为151，T900为148。扩展支线630为156，对C12的154仅有限差异，新增任务曝光很短。共同学习训练映射成立，held改善没有随之兑现 | [专家原文](../20260930_t_architecture/EXPERT_RESPONSE.md)、[公共项修订](../20260930_t_architecture/PUBLIC_BASE_REVIEW.md)、[C设计](../../designs/conditional_read_write_architecture.md)、[具体结果](EVIDENCE.md) |
| **10月3–5日：动作校准、效果信用、角色/坐标与实际调用** | 用真实RGB转移动作标签约束Value，再检验动作/物体效果、角色关系是否可在自身执行中调用 | 校准450 held122低于C137；A/J局部24/22对parent13（/32），A完整held143对parent140/MT153。角色读取/分离和损失改善多次没有转为控制；部分动作误差收益主要来自夹爪。不能把内部改善自动传递到完整方法 | [校准设计](../../designs/control_calibrated_read_write_design.md)、[动作/效果](../../designs/joint_action_effect_credit_diagnostic.md)、[角色绑定](../../designs/native_role_binding_compilation_diagnostic.md)、[重编码](../../designs/native_role_centered_content_diagnostic.md)、[机制回顾§140](../../analyses/feature_to_operator_mechanism_20260926.md)、[结果与原行](EVIDENCE.md) |
| **10月5–7日：真实失败行为、RL、关系教师、直接视频控制与跨任务接手** | 回到开柜放碗/入炉等完整行为，分清读取、规划与执行；检验真实return信用、GT关系、特权动作和直接视频控制能否先形成有用教师 | 共享RL72为158对T161；关系G450为126；强MT上在线视频V88低于匹配M93（/144）；特权真动作1NN91而T50接力仍91、得11失29；最新九起点十臂全0/9。存在局部可控制性，没有得到可靠的强教师或可直接组合控制，不能唯一归罪编译器 | [本轮完整证据](EVIDENCE.md)、[RL设计](../../designs/denoising_return_writer_learning_20261006.md)、[直接控制读回](../../analyses/native_video_control_diagnostic_20261007.json)、[最新90条](../../analyses/cross_task_post_open_transfer_20261007.json) |

V/L450涉及L的24条task39记录后来发现body状态不完全匹配，相关strict比较仅376条；各自完整400总数不因此改写。
初始工程未正式训练的Reader，以及10月5日提出后撤回的前缀K适配，均不是已失败的完整模型。
专家若要讨论这些路线，应先明确真正执行了哪一段。

## 怎样核对“为什么转向”

从[研究历史](../../research_history.md)找到上述阶段，再对照原设计、[findings](../../../findings.md)及当时专家原文。
重点找出三类不同事实：事前可失败预测、实际改变的变量、结果之后主讨论如何修订假说。
主讨论可能把局部通过当成完整方案资格，也可能因短窗阴性过早撤换；两种错误都应审查，不能预先只选一种。

[持续机制分析§140](../../analyses/feature_to_operator_mechanism_20260926.md)已记录9月30日至10月5日的整体回顾及主讨论失误。
请把它当作待核实的自我解释，与实际代码和原件对照，不要仅据文档措辞认定已经理解了机制。

## 远程材料足够做什么

本路线的目标、主设计、历史专家原文、实际实现或精确Git快照、主要学习结果及关键逐行证据均有远程入口。
本轮另补40份此前只在运行目录的开发记录，复用6份既有导出，修正9月17日审计的46处入口，见
[补包目录与范围](history_records/README.md)。它服务于独立方法审阅，不宣称远程包含所有实验原件。

未包含：模型/optimizer实体、完整数据集、全部物理轨迹/RGB/activation/余切大张量、未归档的私人讨论，
以及从未运行的闭环。部分早期实验只可核对记录和代码，不能远程重新复算全部内部张量结论。
若某项核心判断确实依赖这些缺口，请说明具体材料和它会改变哪个判断；不要猜测内容，也无需因此搁置不依赖它的分析。
