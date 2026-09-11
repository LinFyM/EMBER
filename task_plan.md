# EMBER task plan

## 最新覆盖：Owner暂停

Owner要求暂停并先回答正确视频为何被使用。95-task方案、源代码实现、profile和训练均停止；未提交改动保留，不合入main，不启动新实验。下一步只进行Owner要求的机制讨论。下文为暂停前计划，不构成恢复授权。

## 当前目标与授权（2026-09-11）

Owner要求休息期间由Codex自主规划、修正方法、实验验证、分析和再次修正，醒来后审查；active goal继续。此前暂停与禁止再次完整训练的阶段限制已被覆盖，不重复请求逐项批准。固定split、信息墙、source冻结、资源/Git与最终资格保持；Test封存。

目标为validation8 strict paired single-checkpoint correct>145/400，并满足相邻稳定、低churn、高breadth、四suite/Goal/Long、换正确视频及最终独立视频因果资格。不能以机制接通、一次高分或实验结束标记goal完成。

## 当前计划

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
