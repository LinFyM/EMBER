# 固定non-held接手面板完整交接（2026-10-13）

本批完成：32个完整rollout（8次P重放+24次续跑）、12/12登记snapshot槽位、exit0。P败/E成0，P成/E败0，都成3，都败9，无缺项。原登记失败的8槽全部双方败；原登记成功的4槽为3双方成、1双方败。这是有意选择的四task/train24案例，不能估计总体恢复率或新Writer能力。

实际重放中task4/state1、task25/state33由原成功转失败；原身份、noise、角色与登记分母不变，没有补挑或补采。按实际重放成败看，失败支撑10槽中只有task4的原成功槽双方续跑成功，其余9槽双方败，仍无E独有恢复。实际重放成功支撑2槽（17/34）双方续跑均成。

| task | 所选T（主讨论事前标准query16规则） | 3槽结果 | 对当前T的范围 |
| --- | --- | --- | --- |
| 4 | P | 1双方成、2双方败 | 成功续跑由P也完成，无新增E独有恢复 |
| 17 | E | 1双方成、2双方败 | E保持这一个实际成功支撑，两个失败支撑均未恢复 |
| 25 | E | 3双方败 | 本面板没有合格成功接手支撑 |
| 34 | P | 1双方成、2双方败 | P/E都保持一个实际成功支撑，无E独有恢复 |

所选T只用于事前方法分析，不参与本次病例筛选或部署。本批不选择控制器、不生成/更新LoRA。
3个双方成功槽的首native预测5x7动作RMS差分别为task4=.341466、17=.030566、34=.013285；仅描述已执行forward产生的预测内容，不表示首chunk错误、纠错因果或新LoRA组合兼容。原P与本次P首chunk对比RMS0–.011316、最大绝对差0–.046265，初态proprio差全0、noise同源；重放成败变化保留，未要求逐位一致，也未用新forward或调参追求复刻。

| 槽位 | 原登记角色 | 实际重放成功 | pre-action控制步 | 剩余horizon | P续跑成功 | E续跑成功 | 首native动作RMS差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| task_0004_state_001_slot_0 | recorded_success | False | 105 | 115 | True | True | 0.341466 |
| task_0004_state_013_slot_0 | recorded_failure | False | 50 | 170 | False | False | 0.014935 |
| task_0004_state_013_slot_1 | recorded_failure | False | 160 | 60 | False | False | 0.012964 |
| task_0017_state_005_slot_0 | recorded_success | True | 70 | 210 | True | True | 0.030566 |
| task_0017_state_001_slot_0 | recorded_failure | False | 65 | 215 | False | False | 0.016343 |
| task_0017_state_001_slot_1 | recorded_failure | False | 205 | 75 | False | False | 0.032100 |
| task_0025_state_033_slot_0 | recorded_success | False | 145 | 155 | False | False | 0.017019 |
| task_0025_state_004_slot_0 | recorded_failure | False | 70 | 230 | False | False | 0.035640 |
| task_0025_state_004_slot_1 | recorded_failure | False | 220 | 80 | False | False | 0.006277 |
| task_0034_state_007_slot_0 | recorded_success | True | 120 | 400 | True | True | 0.013285 |
| task_0034_state_012_slot_0 | recorded_failure | False | 125 | 395 | False | False | 0.011313 |
| task_0034_state_012_slot_1 | recorded_failure | False | 385 | 135 | False | False | 0.001955 |

范围与验收：4/17/25/34全train24；固定G38349保存P与原共同E，既有source+MT300合并基础，38-target/76完整因子原身份。8个真实History已逐一读回，实际replan数/原预登记公式/index/12个v4 snapshot对应核验；24原行的实际parameter_ref、同snapshot初态、零续跑settling、完整policy noise序列、剩余官方horizon与5x7 finite预测均通过。Runtime Writer/G/theta未初始化、未更新；0新G/held/Test。诊断记录在本ROOT，原§23 H/bank只读，未喂回训练。

真实成本与时间：两GPU接口失败分别43.37秒/.012047671GPUh、53.51秒/.014864329GPUh。第二失败已有至多8次reset/settling但准确次数/步数未知，按主讨论明确amendment预扣80步，不写零交互；首字段错误发生在环境submit前。03只CPU准入因新外部占用而退出，0GPU/交互/费用。原32有效科学rollout不变，physical starts含失败≤40，科学实际环境步5951（正常dummy80含在内），含失败保守总步6031≤10640。

08:39:45Z派发，09:44:13.537Z实际有效启动，09:51:10.389Z执行exit0；准备/修复约64.5min，GPU有效执行6.95min、整批派发至执行退出71.42min，显著超过最初20–35min估计，不重置起点。全批含失败.142699175GPUh；P重放170.0秒/2306步，24续跑151.7秒/3645步。GPU01:0共驻，8环境波次，真实native最大batch8，累计计划1176，CUDA allocated峰10.005GiB/reserved13.770GiB、host RSS12.577GiB；输出约.459GiB。现场可用约22.2GiB，8波次CUDA+EGL+余量估算约19GiB，16波次估算34,200MiB不容纳；未等待卡/利用率筛选/固定24GiB筛选或增加行。全批加原§23合计5/cap6，结束后诊断GPU/环境子进程已全部退出，原owner06保持活跃。

记录局限：raw phases.cumulative_runner因可变对象引用，两阶段均保存了整批终态累计值；分阶段仅使用独立seconds/plans/environment_steps，native prefix/flow/histogram只作整批总量，不从该字段声称分阶段速度。data1上的IPC临时socket在Python退出finalizer产生NFS busy清理警告，actual32行/12snapshot/results/complete和真实exit0均齐，fresh检查无遗留诊断进程。保留日志，不为清理警告或阴性追加rollout。

科学裁决范围：这些固定失败访态没有P失败/E成功的独有目标证据，应由主讨论据此收缩恢复监督的覆盖主张，不能扩成E普遍无用、教学无用或新完整方案已失败。正常单T函数训练/更多访态的实际可用性未由此认证；两个完整控制器成功也不保证混合LoRA。结果正负均停，0自动后继。

稳定原件：panel.json、engineering_start_amendment.json、run_contract.json、replays/*/history.pt.gz与row.json、slots.json及slots/*/snapshot.pt/P.json/E.json、results.json、complete.json；acceptance.json在analysis/。失败/exit/consumed和费用在coordinator/。实际producer clean pushed detached8e718be2；临时消费者结束后从main退役，代码保留于Git/该冻结worktree和原命令。原§23独立继续B18/唯一correct400，原科学合同不变，本批没有第二等待者或阶段通知。
