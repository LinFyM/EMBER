# 实验证据地图与读取口径

本包只导出现有结果，未运行模型、仿真、训练或新的评测。所有数值仍属于原实验，导出不是新实验。请先读[当前要求](OWNER_BRIEF.md)和[推理审计](REASONING_AUDIT.md)，再按下表进入相应原件；不要把历史不同协议拼成统一排行榜。

## 1. 远程读者实际能取得什么

- [evidence_index.json](evidence_index.json)登记每个新增原件或字段投影的来源、路径、行数、分组和省略项；来源`runs/...`、`legacy_runs/...`是定位标识，不是需要远程访问的挂载路径。
- [evidence/episodes](evidence/episodes)有34组逐行结果投影、共40,552条记录，**包括跨报告复用的基线、同一结果的再次引用和参考选点池**，不能称40,552个独立episode。每组有小型`summary.json`、便于核算的`outcomes.csv`、分臂JSONL及需要时的`noise_schedules.jsonl`。优先读summary和需要的分臂文件，无须一次灌入全包。
- JSONL保留原标量、指定小数组、来源文件及原行定位；大型H、hidden、连续轨迹、模型权重不在本包。完整policy noise seed序列只对完全相同前缀作无损去重：用`noise_schedule_ref`定位序列，取前`noise_schedule_length`项。没有重新生成RNG。
- `task_id`在不少旧原件中是suite内编号。`global_task_id_resolved`只来自已存在的global字段或condition元数据，不把raw `task_id`直接当global。72条同incoming诊断缺对应global字段，保留空值、suite/task和原来源；不要把空值解释为task0。
- [evidence/learning](evidence/learning)和[legacy学习证据](evidence/legacy)保留实际共享训练指标、梯度/更新、暴露范围及相关诊断。条件字段和标量不是完整梯度张量，不能由这些文件声称已重新验证全部反传。
- [运行合同](evidence/contracts)、[代码身份](CODE_PROVENANCE.md)及[机器可读代码索引](code_provenance.json)提供输入、标签、训练范围、优化/预算、部署与实际历史实现。机器准入、主机标识和原始调度收据省略，科学成本汇总与异常范围保留。

本包之外的历史原件已经在远程Git，下面直接链接复用。大CSV或曲线若被网页预览截断，使用相同路径的raw文件或读取分臂JSONL；不要把截断显示当成原行缺失。

## 2. 数据与评测身份先行

当前coverage划分和support12名单见[coverage protocol](../../../configs/libero_24_8_8_coverage_v1/protocol.json)及其[manifest](../../../configs/libero_24_8_8_coverage_v1/manifest.json)。source过滤依据见[source manifest](../../../configs/pi05_source_corpus_v1/source_manifest.json)和[overlap audit](../../../configs/pi05_source_corpus_v1/overlap_audit.json)。更早的[原24/8/8协议](../../../configs/libero_24_8_8_v1/protocol.json)按旧实验解释，不能覆盖新协议。

train24、train24+support12=36、其它经过审计的non-held任务范围并非同一个训练分布。参考选点初态、最终报告初态、condition自己的实践初态和query初态也要分开。全包未新增Test或shuffled/reversed原行；历史封存结果不能反哺本次新设计。

## 3. 早期方法：已有原件与本次补件

| 证据及关键事实 | 原件入口 | 解释边界 |
| --- | --- | --- |
| T/U及共享A：450为122/143，900为148/132；T2340 correct161、other150、public103、wrong65，各400；MT153 | [T/U原行](../20260930_t_architecture/validation_rows.csv)、[教学controls原行](../20260930_t_architecture/selected_condition_rows.csv)、[T机制](../20261007_research_reassessment/MECHANISM.md) | rank128、无实践；A/B0/Value真实共同FM。不能把T概括成固定随机A或只训练末层，也不能从T/U局部差异识别唯一根因 |
| Native Correction真实动作oracle：90/384对source68/384，净22；合法Writer ordered50/400等，source47 | [oracle原384配对行](evidence/legacy/native_correction_oracle_rows.json)、[合法Writer配对汇总](evidence/legacy/native_correction_paired_summary.json)、[完整解释](../../analyses/native_corrective_closed_loop_audit.md) | source96条按四teacher复用；24训练task、rank16、特权动作，Object0、task35占净增11/22。不是合法教学获取或held正例 |
| LocalField：特权低秩场的函数改善；合法ordered48/47、frame-set47/50，对source47 | [特权3,072 query](evidence/legacy/local_field_privileged_function.json)、[合法共享原件](../20261007_research_reassessment/history_records/analysis/local_correction_field_writer_20260914/paired_readout.json)、[设计](../../designs/local_correction_field_design.md) | 实际层输入、完整时序与功能信用已存在；函数正例不能转写为控制获取 |
| PPW：原validation64/72/64；训练域原输出20、free-q18、free-AB46，各96；learned outlet50/75/79 | [原配对与曝光](evidence/legacy/PPW_paired_readout.json)、[特权函数前提](evidence/legacy/PPW_functional_summary.json)、[可达性诊断](evidence/legacy/PPW_reachability.json)、[free-q/free-AB函数指标](evidence/legacy/PPW_reachability_functional.json)、[learned outlet原件](../20261007_research_reassessment/history_records/analysis/process_pullback_learned_outlet_20260915/paired_readout.json) | 真实source导数和外层功能监督已经做过；79中的77来自两项奶酪搬运，旧协议不能与当前MT153直接排名 |
| 更早的其它阶段、原专家建议与主讨论修正 | [完整历史图谱](../20261007_research_reassessment/HISTORY_MAP.md)、[原件索引](../20261007_research_reassessment/EVIDENCE.md)、[10/08专家原文入口](../20261008_method_rethink/README.md)、[10/10专家原文](../20261010_writer_design_review/EXPERT_REVIEW.md) | 这些目录中“当前”“下一步”只代表当时时点；建议、未实施候选和实测结果分别引用 |

## 4. 关系、实践和真实参数修订

下面每个`episodes/<ID>/`均含原行投影、summary与来源定位。汇总只用于找原件，不代替成功集合和任务分布。

| 方法/结果 | 逐行入口 | 学习/机制证据与边界 |
| --- | --- | --- |
| G360=131/400，G450=126/400；seen G84/F66，各144 | [G360](evidence/episodes/relation_G360_correct400/summary.json)、[G450](evidence/episodes/relation_G450_correct400/summary.json)、[G seen](evidence/episodes/relation_G450_seen144/summary.json)、[F seen](evidence/episodes/relation_F450_seen144/summary.json) | [完整消费](../../analyses/relation450_main_consumption_20261007.json)；F有GT关系及真实动作FM，但从bare source输出，不是MT上的oracle |
| 匹配P/Q predicted/GT关系81/82→78/81，各144 | [完整分层与成功键](../../analyses/relation_input_compilation_20261007.json) | 共同初始失败58行各只新获3行；不是“关系标签无用”的普遍结论 |
| Experience meta27初末33→30/48；meta54 initial/end/null28/28/28；正式135/400 | [meta27](evidence/episodes/experience_conditioned_compiler_20261009_meta27/summary.json)、[meta54](evidence/episodes/experience_conditioned_compiler_20261009_meta54/summary.json)、[formal](evidence/episodes/experience_conditioned_compiler_20261009_formal/summary.json) | [完整分析](../../analyses/experience_conditioned_compiler_20261009.md)、[训练曲线](evidence/learning/experience_conditioned_compiler_20261009.jsonl)、[合同](evidence/legacy/experience_conditioned_compiler_20261009_contract.json)。128warm+54meta、216真实链/36映射各6次；null是屏蔽E的同链诊断 |
| Parameter formal180=127、360=130；两点train MT/end/null均27/25/25 | [formal180](evidence/episodes/parameter_conditioned_compiler_20261009_formal180/summary.json)、[formal360](evidence/episodes/parameter_compiler_fullreadout_20261009_formal360/summary.json)、[train180](evidence/episodes/parameter_conditioned_compiler_20261009_train180/summary.json)、[train360](evidence/episodes/parameter_conditioned_compiler_20261009_train360/summary.json) | [完整分析](../../analyses/parameter_compiler_fullreadout_20261009.md)、[学习/池](evidence/legacy/Parameter_learning_pools.json)、[曲线](evidence/learning/parameter_conditioned_compiler_20261009.jsonl)。360后补读有事后授权背景，不能当原始预注册选点；非MT incoming仅81、16映射 |
| 同incoming I/E+/E0均13/24，集合不同 | [原行](evidence/episodes/parameter_edit_credit_diagnostic_20261009_diagnostic/summary.json)、[函数与控制冲突](evidence/legacy/Parameter_same_incoming_functions.json) | [完整分析](../../analyses/parameter_edit_credit_diagnostic_20261009.md)；E+对I R11/G2/L2，对E0 R8/G5/L5，不以同分认定同一函数 |
| FR train29→25/48；正式143/400，MT153，R127/G16/L26 | [train180](evidence/episodes/functional_revision_learning_20261010_train180/summary.json)、[train360](evidence/episodes/functional_revision_learning_20261010_train360/summary.json)、[formal360](evidence/episodes/functional_revision_learning_20261010_formal360/summary.json) | [真实消费者](../../analyses/functional_revision_compiler_20261010.md)、[完整学习](../../analyses/functional_revision_learning_20261010.md)、[学习曲线](evidence/legacy/FR_learning_curve.csv)、[写入和梯度](evidence/legacy/FR_learning_summary.json)、[导数原件](evidence/legacy/FR_native_derivatives.json)、[实施更新](evidence/legacy/FR_implementation_updates.jsonl)。保留BF16差分异常和FP32 oracle；正式360为FM+Dkeep，不把实施小PG测试当正式RL |

Experience/Parameter原训练状态已退休，记录见[Experience退休](evidence/legacy/experience_conditioned_compiler_20261009_retirement.json)与[Parameter退休](evidence/legacy/parameter_conditioned_compiler_20261009_retirement.json)。仍有权重、原指标与评估；不能承诺optimizer/RNG exact-resume。

## 5. G/π、选择、功能信用及图像输入

| 结果 | 原件与报告 | 必须区分 |
| --- | --- | --- |
| G24849=425/576，MT340，R313/G112/L27；other414、wrong395；相邻G432/864/1296/8883=337/341/348/362 | [九臂原行/汇总](evidence/episodes/G24849_video/summary.json)、[科学消费](evidence/proposal/G24849_scientific_readout.json)、[原统计](evidence/proposal/G24849_original_statistics.json) | 36训练任务、固定MT/H的一次G组件；不是完整Writer或held结果。保留same-task高分，也保留wrong仍395 |
| G38349 seen427/576，MT340 | [逐task原行](evidence/episodes/G38349_seen/summary.json) | 共同E训练/参数拟合的真实正例，不强迫失败H用途 |
| 完整RL9=165/400，MT153，R129/G36/L24；旧RL9=147 | [三臂原行](evidence/episodes/full_RL9/summary.json)、[完整统计](evidence/proposal/full_RL9_statistics.json)、[训练审计](evidence/proposal/full_RL9_training_audit.json)、[代码/成本来源](evidence/proposal/full_RL9_provenance.json) | 1,465次实践、111终点仍MT；训练来源和formal恢复身份分开 |
| first-G=165/400 | [独立原行](evidence/episodes/firstG_RL9/summary.json) | 事后同核组件诊断；与完整RL9总分相同、成功集合不同，不能合并 |
| selector B18=163/400；独立价值Phi180=167/400 | [B18原行](evidence/episodes/selector_B18/summary.json)、[selector报告](evidence/proposal/selector_transfer_report.md)、[Phi180原行](evidence/episodes/value_Phi180/summary.json)、[价值报告](evidence/proposal/value_selection_report.md) | B18实践增至1,988；396个有G条件全选G。Phi对14个训练MT胜G反例仅选对2个，另6个为0，不能以小净增宣布选择机制已解决 |
| G功能匹配：F-only404/576、159/400；full-G339/150 | [两臂seen/val原行](evidence/episodes/G_function_match/summary.json)、[报告](evidence/proposal/native_function_report.md) | 同288条件/72更新，有真实功能信用；函数项不是自动有效 |
| BA-only D72=414/175；BA+native C72=393/158；D144=414/168 | [D/C原行](evidence/episodes/operator_D_C72/summary.json)、[D144原行](evidence/episodes/operator_D144/summary.json)、[匹配报告](evidence/proposal/operator_native_report.md)、[相邻报告](evidence/proposal/operator_adjacent_report.md) | 前一数字/576，后一/400。D72→144 held R149/G19/L26；175高点与相邻交换均保留 |
| 图像Gemma+H及继续学习 | [首窗报告](evidence/proposal/image_only_report.md)、[继续报告](evidence/proposal/image_history_refinement_report.md)、[对应合同](evidence/contracts) | 正确理解输入与训练标签匹配；改变读取支路不等于完整Writer取得新能力。完整逐行主要结论以原报告登记，未把该家族额外复制成新episodes集合 |
| 真实接管可行性 | [原报告](evidence/proposal/takeover_report.md)、[T选点读出](evidence/proposal/takeover_T_readout.json)、[source重放比较](evidence/proposal/takeover_replay_comparison.json) | 12槽0独有恢复、3双方成9双方败；8原失败槽均败。不能据参考初态能力假定广泛任意访态恢复 |

## 6. N4/N16：实践提升与独立query分离

| 证据 | 入口 | 范围与修正 |
| --- | --- | --- |
| 纠正N4：MT/V0/V1/noV=67/64/70/74，各128 | [四臂原行](evidence/episodes/N4_corrected/summary.json)、[工程纠正报告](evidence/practice/N4_corrected_report.md)、[完整科学读出](evidence/practice/N4_scientific_readout.md) | 内部8任务，MT起点。noV全部未改MT，74对67不可归因学习。原未纠正报告仅作事故与历史来源，不作为最终方法成绩 |
| N4信用与全失败局部正例 | [信用报告](evidence/practice/N4_credit_report.md)、[信用数据](evidence/practice/N4_credit_data.json) | task32/V1接受链实践0/4、query5/16对MT2；第四次采样信用mixed，不能说所有采样分支全失败。task1实践获益而query损失也须保留 |
| N16：MT/V0/V1/noV=67/61/71/62；实践66→77/72/80 | [四臂原行](evidence/episodes/N16/summary.json)、[报告](evidence/practice/N16_report.md)、[24条件](evidence/practice/N16_conditions.json)、[配对](evidence/practice/N16_paired_metrics.json)、[per-task](evidence/practice/N16_per_task.json)、[更新/成本](evidence/practice/N16_update_and_resource_summary.json) | 全部24条件初始已有3–14/16成功，不是全失败冷启动。task12 V0/V1实践6→8/7、query均8与MT8相同；noV实践6→9、query7 |
| 真实行为例子 | [图像](evidence/practice/unaffected_behavior_examples.png)、[对应元数据](evidence/practice/unaffected_behavior_examples.json) | 既有自身轨迹截图，无新模型或环境；手物共同运动不能单独认证完整成功。例子是解释材料，不替代完整分布 |

## 7. source/rank8完整证据链

| 阶段/结果 | 原行、报告与学习证据 | 关键口径 |
| --- | --- | --- |
| reference独立report545/648，source248 | [独立报告原行](evidence/episodes/source_reference_report/summary.json)、[全部选点池](evidence/episodes/source_reference_selection/summary.json)、[报告](evidence/source/reference_report.md) | 36任务；selection初态0–31，冻结后report32–49。9,664行选点池含多个节点，汇总成功总数不构成模型分数 |
| 36×16：source218/reference502；E/D144216、operator1170224、linear64 U36=484 | [统一主要参照](evidence/episodes/source_family_adjacent/summary.json)、[E/D144原行](evidence/episodes/source_ED144/summary.json)、[operator1170原行](evidence/episodes/source_operator1170/summary.json)、[linear报告](evidence/source/control_linear64_seed7_report.md)、[E/D曲线](evidence/learning/control_space_seed7.jsonl)、[operator曲线](evidence/learning/control_operator_from144_seed7.jsonl) | 初态0–15为已选点面板；linear成功是bank内控制可表示，不是视频或held泛化 |
| Reader900→1800：seen334→413，held117→111 | [原行](evidence/episodes/source_family_adjacent/summary.json)、[报告](evidence/source/video_initial_seed7_report.md)、[曲线](evidence/learning/video_initial_seed7.jsonl) | 训练能力上升与held交换并存，不能用seen曲线忽略验证结果 |
| 36×4同教学query：source58、reference127、Reader89、first-best73、full79、随机86、functional93、native64 | [八臂原行](evidence/episodes/native_first_write/summary.json)、[Actor报告](evidence/source/feedback_proposals_seed7_report.md)、[Actor读出报告](evidence/source/feedback_readout_seed7_report.md)、[固定Reader报告](evidence/source/incumbent_compilation_seed7_report.md) | “独立”是相对自身实践；参考127仍受到此前选点影响。full路径含不同训练θ，不冒称冻结meta9成绩 |
| 固定Reader无视频59/144 | [原行](evidence/episodes/functional_no_video/summary.json) | 与video随机86对比有范围限制，不替代全部视频/H因果控制 |
| functional ψ180 seen93/144；held119/400，Reader117 | [held与参照原行](evidence/episodes/source_family_adjacent/summary.json)、[训练报告](evidence/source/functional_feedback_seed7_report.md)、[曲线](evidence/learning/functional_feedback_seed7.jsonl)、[完整验证分层](evidence/source/functional_validation_readout.json)、[关闭读出](evidence/source/functional_validation_closing.json) | 初始实践全零177条件query8→7；改参数105条件实践156→298/query51→53；三个零task合计650候选仍无实践正例。不将实践增幅当泛化 |
| task-exclusion U28=37/128，U36=98、reference105、source23 | [四臂原行](evidence/episodes/source_task_exclusion/summary.json)、[报告](evidence/source/control_task_exclusion8_seed7_report.md) | 八项均原合法训练task的排除诊断，不等同正式held，削弱固定空间覆盖假设 |
| source-W BA1180 seen244/held57；pair1090=290/121、1180=306/128 | [统一原行](evidence/episodes/source_family_adjacent/summary.json)、[BA完整读出](evidence/source/source_weight_readout.json)、[算子读出](evidence/source/source_weight_operator_readout.json)、[因子几何](evidence/source/source_weight_factor_geometry.json)、[pair算子](evidence/source/factor_credit_operator_readout.json)、[相邻读出](evidence/source/factor_credit_adjacent_readout.json) | 真实恢复与剩余巨大参考缺口同时成立；同族末端收益集中、held三任务为零。范数/余弦不能替代闭环 |
| native180首写64/144，对source58、Reader89、reference127；Object0 | [八臂原行](evidence/episodes/native_first_write/summary.json)、[报告](evidence/native/engineering_completion.md)、[逐case变化](evidence/native/case_changes.json)、[训练消费者](evidence/native/training_acceptance.json)、[教学读取](evidence/native/teaching_accounting.json)、[曲线](evidence/learning/native180.jsonl) | source-H共144条含52次成功/40,377步，每task固定16决策支撑。真实38-target J、条件相关rank8和校准已实现；固定H首写不是完整在线适应，不是全失败冷启动 |

## 8. 仍然没有、已退役或未上传的证据

1. **未做过的科学对照不能补成已有。** Experience/Parameter/FR缺完整held视频/H因果controls，FR缺held同观测incoming/outgoing函数比较；近期source生成/native首写缺完整held视频/H因果资格。不同旧实验的局部正例不拼成一个完整合格方法。
2. **选择oracle不存在。** 未选候选通常没有独立query，不能从实践回报构造“所有候选真实query最优”的oracle；同任务共同E和参数标签存在也不证明H被用于纠错。
3. **资产生命周期有缺口。** Experience/Parameter完整训练状态已退役；functional验证392条raw H已退役，只留8条init0全链。原行成败与汇总可重算，完整物理过程不能全数重放。
4. **历史记录不齐全。** 旧MT400不含所有initial-proprio/capture字段；N16旧71未回执及费用保留区间，不能伪造精确完成记录或成本。公开投影保留缺失，不填猜测值。
5. **大资产未入Git。** source/MT/参考/Writer权重、数据集、全部视频、连续state/action、H/X/余切未上传。已有截图可说明有限例子；远程分析可以核查代码、算子、记录与分布，不能声称重新跑了完整模型或所有轨迹。
6. **封存信息有意排除。** 本包不新增Test与shuffled/reversed原行，旧Git可见性不解除不反哺设计的边界。当前没有active design，也没有任何候选因这次导出取得启动资格。

最终文件级核验范围与结果见[verification_summary.json](verification_summary.json)。它验证交接包能支持既有证据阅读和标量重算，不是新的科学有效性证明。
