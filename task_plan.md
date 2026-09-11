# EMBER task plan

## 当前目标与三步执行（Owner 2026-09-11最新授权）

Owner补充的必要验收条件：同一个候选的correct闭环性能须高于冻结source，并分别高于错任务视频和乱序视频；使用同任务、初态、env/policy RNG的配对参照，报告差额、逐task/suite和不确定性。仅错误视频下降而correct不超过source，不算成功。该条件属于当前goal的验收标准；更高的EMBER最终目标继续保留。

Owner明确要求设定goal并连续完成：先还原当前架构的科学精神、数学推导及逐轮专家讨论；再结合理论和有信息量实验定位正确视频未产生执行增量的原因；依据原因修改架构或训练、实验验证，失败返回原因分析并持续迭代。正确视频定义为同任务且内部顺序正确；错任务或乱序均属于错误视频。目标是正常训练自然产生可信的correct>wrong及correct>shuffled闭环分化，不靠人为压低错误视频表现、惩罚错误条件或削弱参照。保留绝对能力>145/400及稳定性目标，不能以接近source的微小差额宣布整个研究完成。

1. **完成形成过程审查：** 已读9月7日首轮、9月8日四轮、9月11日两轮原文及Owner裁决，核对当前实际输入、关系/视觉/GRU/回写/Compiler/D和FM重放；旧v5.2/v6推导、代码与6000条原始rows经独立只读核查。结果见[机制复核](docs/video_mechanism_reassessment.md)。未声称所有历史逐行穷举或根因已找到。
2. **完成当前冻结诊断：失效定位。** 沿视频证据→过程表示→LoRA→独立闭环建立竞争解释。先补已有C100/C200缺失的正确/错误视频机制状态，与匹配S和source比较、复用off400九臂；无新学习。结果前合同已登记于`runs/analysis/video_mechanism_20260911/registration.json`，1344主面板新rows；source复用96rows；固定历史八task另登记128条S/P4替换诊断，配对独立动作检查辅助定位。不把梯度、输入完整性、attention或一次局部拟合当成理解或根因。
3. **进行中：修正与迭代。** 1472条新闭环及配对动作已完成，C200正确42/96低于两错46、乱序45和静态49，固定S的正确/静态过程同为15/32。依据静态P4污染这一具体表示缺陷，无变化参照更新已通过128项CPU合同及真实最长视频/四六卡profile并集成main；按[active design](docs/video_change_reference_design.md)fresh100/200后exact-resume300/400，尚无正式学习结果。 每次改动说明因果依据、可辨别预测与学习观察范围；有获取趋势时补足后续节点，预算结束与候选收敛/被否定严格区分。失败回到第2步，不自动转向覆盖、种子或超参扫描。

本次最新授权允许开发阶段使用错任务和乱序视频的正常输入干预来诊断、指导修正，覆盖旧文档中禁止其反馈架构的限制。开发诊断与最终未触碰的冻结后验证分开；不新增held梯度，不用错误视频惩罚训练或按差额挑选偶然checkpoint，Test继续封存。每项新增实验先登记数据、用途、配对与判据，再执行。

95-task扩展不恢复；其两个worktree的未提交改动保留，不自动集成或清理。当前唯一active design为[无变化参照局部过程更新](docs/video_change_reference_design.md)，正式学习合同与四个节点已在结果前登记。下文仅为旧授权与覆盖实验计划，不能按其清单恢复执行。

## 历史目标与授权（暂停前）

Owner要求休息期间由Codex自主规划、修正方法、实验验证、分析和再次修正，醒来后审查；active goal继续。此前暂停与禁止再次完整训练的阶段限制已被覆盖，不重复请求逐项批准。固定split、信息墙、source冻结、资源/Git与最终资格保持；Test封存。

目标为validation8 strict paired single-checkpoint correct>145/400，并满足相邻稳定、低churn、高breadth、四suite/Goal/Long、换正确视频及最终独立视频因果资格。不能以机制接通、一次高分或实验结束标记goal完成。

## 历史覆盖实验计划（不执行）

唯一active design为[非held meta-task扩展](docs/nonheld_meta_writer_design.md)。前轮R/C/S与off、R有限续训全部完成，旧合同只作历史。

1. **完成：** 根据R400=85/400、train56/96和同target预算旧off126/56，结束条件分配路线；登记保留off图、普通FM、target24+已审计meta71各半的95-task有限对照。只改变独立任务覆盖，target恢复原一条K1/64queries。
2. **进行中：** 在隔离worktree实现审计meta loader、独立采样流、8-task逻辑batch/权重/曝光及完整恢复；并行退役C/S专属路径。主agent整合，保持off计算与初始化语义。
3. 验证固定target/held边界、71项allowlist、跨episode、target采样流不受meta消耗、8×64queries/权重1/8、完整恢复；真实混合GPU更新及最长视频，source始终无梯度。
4. 从clean pushed detached commit fresh seed7执行200/400有限节点：各correct strict400，400 train96，0/200/400独立动作验证。分别累计target51200/102400和同等meta queries，显式报告新增计算及独立task覆盖，不冒充同总预算。
5. 依per-task/suite、breadth、相邻及同target预算旧off的R/G/L/churn裁决；无广泛改善则结束该扩展，不扫比例/seed/LR或原样续训。实质改善后才登记相邻稳定与独立视频资格。目标未完成前持续推进。

## 已完成的本轮证据

- R/C/S各fresh200：validation100→200为37→83、43→50、59→45；train96=46/49/52。C/S不追加。
- off7从400受控续训至600：correct126→73→54，train56→64/96，停止未改配方；保留探索性lineage。
- R原200 exact-resume至400：correct83→63→85，train46→56/96；旧off400同102400queries为126/56。全部新增896rows及实际配对完整，停止R。
- 首轮2688rows + off续训896 + R续训896 = 4480条闭环；原始结果、checkpoint和合同保留。未新增Test、held梯度或视频因果controls。

## 审查入口

[progress](progress.md)记录实现/运行状态；[findings](findings.md)§51–53保存跨轮结果；[research_history](docs/research_history.md)保留完整历史及近等价边界。专家原文见[9月11日材料](docs/review_materials/20260911/README.md)。
